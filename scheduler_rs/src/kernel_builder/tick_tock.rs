//! TickTock Kernel Builder - Rust port of the Python tick_tock.py builder.
//!
//! This builder generates optimized instruction sequences for the tree traversal
//! and hashing kernel using the tick-tock lazy partition strategy.

use crate::types::{Op, Opcode};

use super::config::{
    fma_multiplier, KernelConfig, HASH_STAGES, SCRATCH_SIZE, STAGE5_XOR_CONST, VLEN,
};
use super::context::{ScratchAllocator, ScratchContext, ZERO_REG};
use super::emit::EmitContext;
use super::partition::{get_read_buffer, should_partition, PartitionHelper};

/// Unroll factor - process 8 groups per iteration
const UNROLL: usize = 8;

/// Depths where TICK stores happen (first half of groups)
const TICK_STORE_DEPTHS: [usize; 6] = [0, 1, 2, 4, 5, 6];

/// Depths where TOCK stores happen (second half of groups)
const TOCK_STORE_DEPTHS: [usize; 4] = [3, 7, 8, 9];

/// TickTock Kernel Builder
///
/// Generates optimized instruction sequences for tree traversal and hashing
/// using the tick-tock lazy partition strategy.
#[derive(Debug)]
pub struct TickTockKernelBuilder {
    /// Kernel configuration
    pub config: KernelConfig,
    /// Scratch memory context
    pub ctx: ScratchContext,
    /// Scratch allocator
    pub alloc: ScratchAllocator,
    /// Emission context
    pub emit: EmitContext,
    /// Whether depth 3 broadcasts have been initialized (lazy init)
    depth3_broadcasts_initialized: bool,
    /// Whether any round will use depth 3 (pre-computed)
    uses_depth3: bool,
}

impl TickTockKernelBuilder {
    /// Create a new kernel builder with the given configuration
    pub fn new(config: KernelConfig) -> Self {
        let encoding_flags = config.encoding_flags();
        // Pre-compute whether any round will use depth 3
        let tree_height = config.forest_height;
        let uses_depth3 = (0..config.rounds).any(|r| {
            let tree_depth = r % tree_height;
            tree_depth == 3
        });
        TickTockKernelBuilder {
            config,
            ctx: ScratchContext::default(),
            alloc: ScratchAllocator::default(),
            emit: EmitContext::new_with_flags(encoding_flags),
            depth3_broadcasts_initialized: false,
            uses_depth3,
        }
    }

    /// Create a builder with default configuration
    pub fn default_config() -> Self {
        Self::new(KernelConfig::default())
    }

    /// Build the kernel, returning the list of unscheduled ops
    pub fn build(&mut self) -> Vec<Op> {
        self.build_internal();
        self.emit.take_ops()
    }

    /// Build the kernel with pseudo-const support, returning both ops and pseudo-consts.
    ///
    /// Use this when enable_pseudo_consts is set to get the pseudo-const list
    /// for the scheduler's schedule_with_pseudo_consts() function.
    pub fn build_with_pseudo_consts(&mut self) -> (Vec<Op>, Vec<crate::types::PseudoConst>) {
        self.build_internal();
        self.emit.take_ops_and_pseudo_consts()
    }

    /// Internal build logic shared by both build() and build_with_pseudo_consts()
    fn build_internal(&mut self) {
        let n_groups = self.config.n_groups();
        let n_iters = self.config.n_iters();

        // Allocate scratch
        self.allocate_common(n_groups, n_iters);
        self.allocate_valu_selection();
        self.allocate_partition();  // After valu_selection so we can reuse tc_r3/v_tc_r3

        let usage = self.alloc.usage();
        if usage > SCRATCH_SIZE as i64 {
            eprintln!(
                "WARNING: Scratch usage {} exceeds budget {}!",
                usage, SCRATCH_SIZE
            );
        }

        // Emit initialization - choose method based on config
        if self.config.enable_pseudo_consts {
            self.emit_common_init_with_pseudo_consts();
        } else {
            self.emit_common_init();
        }
        self.emit_valu_selection_init();
        self.emit_partition_init();

        // Load data
        self.emit_load_data(n_groups);

        // Main loop
        let tree_height = self.config.tree_height();
        let mut tree_depth = 0;
        for rnd in 0..self.config.rounds {
            self.emit_round(rnd, tree_depth, n_groups, n_iters);

            // Update tree depth with wrap
            tree_depth += 1;
            if tree_depth >= tree_height {
                tree_depth = 0;
            }
        }

        // Convert values back from optimized form
        for g in 0..n_groups {
            self.emit.emit_valu(
                Opcode::Xor,
                self.ctx.all_val[g],
                self.ctx.all_val[g],
                self.ctx.v_stage5_xor,
            );
        }

        // Store data
        self.emit_store_data(n_groups);
    }

    // =========================================================================
    // Allocation
    // =========================================================================

    fn allocate_common(&mut self, n_groups: usize, n_iters: usize) {
        let alloc = &mut self.alloc;
        let ctx = &mut self.ctx;

        // Pointers
        ctx.forest_p = alloc.alloc("forest_p", 1);
        ctx.idx_p = alloc.alloc("idx_p", 1);
        ctx.val_p = alloc.alloc("val_p", 1);
        ctx.n_nodes_s = alloc.alloc("n_nodes", 1);

        // Temps
        for i in 0..16 {
            ctx.st[i] = alloc.alloc(&format!("st{}", i), 1);
        }

        // Constants 0-15: c[0] uses reserved ZERO_REG, allocate c[1..16]
        ctx.c[0] = ZERO_REG;
        for i in 1..16 {
            ctx.c[i] = alloc.alloc(&format!("c{}", i), 1);
        }

        // Hash constants
        for i in 0..6 {
            ctx.hc[i] = alloc.alloc(&format!("hc{}", i), 1);
        }
        for i in 0..6 {
            ctx.hs[i] = alloc.alloc(&format!("hs{}", i), 1);
        }

        // Tree cache
        for i in 0..7 {
            ctx.tc[i] = alloc.alloc(&format!("tc{}", i), 1);
        }
        // Tree cache address registers for parallel computation
        for i in 0..7 {
            ctx.tc_addr[i] = alloc.alloc(&format!("tc_addr{}", i), 1);
        }
        ctx.td_1 = alloc.alloc("td_1", 1);
        ctx.td_4 = alloc.alloc("td_4", 1);
        ctx.td_5 = alloc.alloc("td_5", 1);
        ctx.td_6 = alloc.alloc("td_6", 1);

        // Stage 5 XOR
        ctx.stage5_xor_s = alloc.alloc("stage5_xor", 1);

        // Per-group vectors
        ctx.all_idx = (0..n_groups)
            .map(|g| alloc.vec(&format!("all_idx_{}", g)))
            .collect();
        ctx.all_val = (0..n_groups)
            .map(|g| alloc.vec(&format!("all_val_{}", g)))
            .collect();
        ctx.v_nv = (0..n_iters)
            .map(|it| {
                (0..UNROLL)
                    .map(|i| alloc.vec(&format!("v_nv_{}_{}", it, i)))
                    .collect()
            })
            .collect();
        ctx.v_t = (0..n_iters)
            .map(|it| {
                (0..UNROLL)
                    .map(|i| alloc.vec(&format!("v_t_{}_{}", it, i)))
                    .collect()
            })
            .collect();

        // Vector constants
        ctx.v_0 = alloc.vec("v_0");
        ctx.v_1 = alloc.vec("v_1");
        ctx.v_2 = alloc.vec("v_2");
        ctx.v_nn = alloc.vec("v_nn");
        for i in 0..8 {
            ctx.v_const[i] = alloc.vec(&format!("v_const{}", i));
        }

        // Vector hash constants
        for i in 0..6 {
            ctx.v_hc[i] = alloc.vec(&format!("v_hc{}", i));
        }
        for i in 0..6 {
            ctx.v_hs[i] = alloc.vec(&format!("v_hs{}", i));
        }

        // Vector tree cache
        ctx.v_td_1 = alloc.vec("v_td_1");
        ctx.v_td_4 = alloc.vec("v_td_4");
        ctx.v_td_5 = alloc.vec("v_td_5");
        ctx.v_td_6 = alloc.vec("v_td_6");
        ctx.v_tc2 = alloc.vec("v_tc2");
        ctx.v_tc3 = alloc.vec("v_tc3");

        // Stage 5 XOR vector
        ctx.v_stage5_xor = alloc.vec("v_stage5_xor");

        // FMA multipliers
        for i in 0..6 {
            ctx.hash_fma_mult[i] = fma_multiplier(i);
        }
    }

    fn allocate_partition(&mut self) {
        let alloc = &mut self.alloc;
        let ctx = &mut self.ctx;

        // Memory pointers
        ctx.extra_room_p = alloc.alloc("extra_room_p", 1);

        // Partition buffer offsets (for tick-tock stores)
        ctx.partition_idx_offset = alloc.alloc("part_idx_off", 1);
        ctx.partition_val_offset = alloc.alloc("part_val_off", 1);

        // Only allocate lazy partition resources if enabled
        if self.config.enable_lazy_partition {
            // Allocate separate scratch for idx and val during partition stores
            // Each needs 8 words (VLEN) for vstore/load operations
            // NOTE: Can't reuse tc_r3[0..7] for both - they'd overwrite each other!
            ctx.part_idx_scratch = alloc.alloc("part_idx_scratch", 8);  // 8 words for idx
            ctx.tree_cache_scalars = alloc.alloc("tree_cache_scalars", 8);  // 8 words for val
            ctx.tree_cache_base_s = alloc.alloc("tree_cache_base", 1);

            // Partition counters - REUSE st[14] and st[15] to save scratch!
            ctx.part_left_count = ctx.st[14];
            ctx.part_right_count = ctx.st[15];

            // Reuse existing vectors - v_tc_r3 are only for depth 3
            // At depths 4+, they're free to reuse
            // But only if depth 3 vectors were actually allocated!
            if self.uses_depth3 {
                ctx.v_cache_base = ctx.v_tc_r3[0];
                ctx.v_tree_temp = ctx.v_tc_r3[1];
                ctx.v_part_temp[0] = ctx.v_tc_r3[2];
                ctx.v_part_temp[1] = ctx.v_tc_r3[3];
                ctx.v_part_temp[2] = ctx.v_tc_r3[4];
                ctx.v_part_temp[3] = ctx.v_tc_r3[5];
            } else {
                // Allocate fresh vectors if depth 3 wasn't allocated
                ctx.v_cache_base = alloc.vec("v_cache_base");
                ctx.v_tree_temp = alloc.vec("v_tree_temp");
                for i in 0..4 {
                    ctx.v_part_temp[i] = alloc.vec(&format!("v_part_temp_{}", i));
                }
            }
        }
    }

    fn allocate_valu_selection(&mut self) {
        let alloc = &mut self.alloc;
        let ctx = &mut self.ctx;

        // Depth 3 cache - only allocate if depth 3 will be used
        // This saves 128 words of scratch when depth 3 is never used.
        if self.uses_depth3 {
            for i in 0..8 {
                ctx.tc_r3[i] = alloc.alloc(&format!("tc_r3_{}", i), 1);
            }
            // Depth 3 address registers for parallel computation
            for i in 0..8 {
                ctx.tc_r3_addr[i] = alloc.alloc(&format!("tc_r3_addr_{}", i), 1);
            }
            for i in 0..8 {
                ctx.v_tc_r3[i] = alloc.vec(&format!("v_tc_r3_{}", i));
            }
            for i in 0..8 {
                ctx.v_const_r3[i] = alloc.vec(&format!("v_const_r3_{}", i));
            }
            // c_r3[i] will alias to c[7+i] in emit_valu_selection_init - no separate allocation needed

            // Tree-reduce deltas for depth 3 - only allocate if enabled
            // This uses 36 additional words (4 scalars + 4 vectors) but saves 1 op per group.
            // DISABLED BY DEFAULT because it exceeds the 1536 word scratch budget.
            if self.config.enable_tree_reduce_depth3 {
                ctx.td_r3_01 = alloc.alloc("td_r3_01", 1);
                ctx.td_r3_23 = alloc.alloc("td_r3_23", 1);
                ctx.td_r3_45 = alloc.alloc("td_r3_45", 1);
                ctx.td_r3_67 = alloc.alloc("td_r3_67", 1);

                ctx.v_td_r3_01 = alloc.vec("v_td_r3_01");
                ctx.v_td_r3_23 = alloc.vec("v_td_r3_23");
                ctx.v_td_r3_45 = alloc.vec("v_td_r3_45");
                ctx.v_td_r3_67 = alloc.vec("v_td_r3_67");
            }
        }

        // NOTE: Parallel selection optimization for depths 3 and 4 was explored but
        // disabled due to scratch budget constraints. The extra temp vectors for
        // parallel comparison and reduction (16+ vectors = 128+ words) push scratch
        // usage over the 1536 word limit.
        //
        // The current serialized approach (FMA accumulation) works well because:
        // 1. The scheduler effectively interleaves operations across groups
        // 2. No extra scratch required for parallel temps
        // 3. Dependency chains are hidden by group-level parallelism
    }

    // =========================================================================
    // Initialization
    // =========================================================================

    fn emit_common_init(&mut self) {
        // Register ZERO_REG as known constant 0 (no load needed - scratch is initialized to 0)
        self.emit.known_constants.insert(ZERO_REG, 0);

        // === CONSTANT DERIVATION: 3 Load ops + 12 ALU ops ===
        // Bootstrap c[1], c[2], c[4] to break dependency chain.
        // This enables earlier availability of key constants used for tree cache:
        //   - c[0..6] available by cycle 2 (was cycle 4)
        //   - All c[0..15] available by cycle 4 (was cycle 5)
        //
        // Derivation tree:
        //   Cycle 0: Load c[1]=1, c[2]=2, c[4]=4
        //   Cycle +1: c[3]=2+1, c[5]=4+1, c[6]=4+2, c[8]=4+4 (parallel)
        //   Cycle +2: c[7]=4+3, c[9..13] (parallel)
        //   Cycle +3: c[14]=7+7, c[15]=8+7 (parallel)

        // Bootstrap: Load constants 1, 2, 4 to break dependency chain
        self.emit.emit_const(self.ctx.c[1], 1);
        self.emit.emit_const(self.ctx.c[2], 2);
        self.emit.emit_const(self.ctx.c[4], 4);

        // Derive remaining constants using ALU (scheduler will parallelize)
        // Cycle +1: 3, 5, 6, 8 (all depend only on bootstrapped constants)
        self.emit
            .emit_alu(Opcode::Add, self.ctx.c[3], self.ctx.c[2], self.ctx.c[1]);
        self.emit
            .emit_alu(Opcode::Add, self.ctx.c[5], self.ctx.c[4], self.ctx.c[1]);
        self.emit
            .emit_alu(Opcode::Add, self.ctx.c[6], self.ctx.c[4], self.ctx.c[2]);
        self.emit
            .emit_alu(Opcode::Add, self.ctx.c[8], self.ctx.c[4], self.ctx.c[4]);

        // Cycle +2: 7, 9, 10, 11, 12, 13, 14, 15 (depend on cycle +1 constants)
        self.emit
            .emit_alu(Opcode::Add, self.ctx.c[7], self.ctx.c[4], self.ctx.c[3]);
        self.emit
            .emit_alu(Opcode::Add, self.ctx.c[9], self.ctx.c[8], self.ctx.c[1]);
        self.emit
            .emit_alu(Opcode::Add, self.ctx.c[10], self.ctx.c[5], self.ctx.c[5]);
        self.emit
            .emit_alu(Opcode::Add, self.ctx.c[11], self.ctx.c[8], self.ctx.c[3]);
        self.emit
            .emit_alu(Opcode::Add, self.ctx.c[12], self.ctx.c[6], self.ctx.c[6]);
        self.emit
            .emit_alu(Opcode::Add, self.ctx.c[13], self.ctx.c[8], self.ctx.c[5]);
        self.emit
            .emit_alu(Opcode::Add, self.ctx.c[14], self.ctx.c[7], self.ctx.c[7]);
        self.emit
            .emit_alu(Opcode::Add, self.ctx.c[15], self.ctx.c[8], self.ctx.c[7]);

        // Register all constants in known_constants for later derivation use
        for i in 1..16 {
            self.emit.known_constants.insert(self.ctx.c[i], i as i64);
        }

        // Load pointers
        self.emit.emit_const(self.ctx.st[0], 4);
        self.emit.emit_const(self.ctx.st[1], 5);
        self.emit.emit_load(self.ctx.forest_p, self.ctx.st[0]);
        self.emit.emit_load(self.ctx.idx_p, self.ctx.st[1]);
        self.emit.emit_const(self.ctx.st[0], 6);
        self.emit.emit_const(self.ctx.st[1], 1);
        self.emit.emit_load(self.ctx.val_p, self.ctx.st[0]);
        self.emit.emit_load(self.ctx.n_nodes_s, self.ctx.st[1]);

        // Load extra_room pointer
        self.emit.emit_const(self.ctx.st[0], 7);
        self.emit.emit_load(self.ctx.extra_room_p, self.ctx.st[0]);

        // Compute partition buffer offsets
        self.emit.emit_const(self.ctx.partition_idx_offset, 0);
        self.emit.emit_const(self.ctx.partition_val_offset, 256);

        // Load hash constants
        for i in 0..6 {
            self.emit.emit_const(self.ctx.hc[i], HASH_STAGES[i].1);
            self.emit.emit_const(self.ctx.hs[i], HASH_STAGES[i].4);
        }

        // Broadcast vector constants - use CSE to avoid duplicate broadcasts
        // v_0, v_1, v_2 are broadcast first, then v_const[0..8] will reuse them via CSE
        self.ctx.v_0 = self.emit.emit_vbroadcast(self.ctx.v_0, self.ctx.c[0]);
        self.ctx.v_1 = self.emit.emit_vbroadcast(self.ctx.v_1, self.ctx.c[1]);
        self.ctx.v_2 = self.emit.emit_vbroadcast(self.ctx.v_2, self.ctx.c[2]);
        self.ctx.v_nn = self.emit.emit_vbroadcast(self.ctx.v_nn, self.ctx.n_nodes_s);

        // v_const[0], [1], [2] will be CSE'd to v_0, v_1, v_2 (no new ops emitted)
        // v_const[3..8] will emit new broadcasts
        for i in 0..8 {
            self.ctx.v_const[i] = self.emit.emit_vbroadcast(self.ctx.v_const[i], self.ctx.c[i]);
        }

        for i in 0..6 {
            self.ctx.v_hc[i] = self.emit.emit_vbroadcast(self.ctx.v_hc[i], self.ctx.hc[i]);
            self.ctx.v_hs[i] = self.emit.emit_vbroadcast(self.ctx.v_hs[i], self.ctx.hs[i]);
        }

        // Load Stage 5 XOR constant
        self.emit.emit_const(self.ctx.stage5_xor_s, STAGE5_XOR_CONST);
        self.ctx.v_stage5_xor = self.emit.emit_vbroadcast(self.ctx.v_stage5_xor, self.ctx.stage5_xor_s);

        // Load tree cache - parallel address computation
        // First compute all addresses (can parallelize in ALU)
        for i in 0..7 {
            self.emit.emit_alu(Opcode::Add, self.ctx.tc_addr[i], self.ctx.forest_p, self.ctx.c[i]);
        }
        // Then load all values (2 loads per cycle)
        for i in 0..7 {
            self.emit.emit_load(self.ctx.tc[i], self.ctx.tc_addr[i]);
        }

        // Compute deltas
        self.emit.emit_alu(Opcode::Sub, self.ctx.td_1, self.ctx.tc[1], self.ctx.tc[2]);
        self.emit.emit_alu(Opcode::Sub, self.ctx.td_4, self.ctx.tc[4], self.ctx.tc[3]);
        self.emit.emit_alu(Opcode::Sub, self.ctx.td_5, self.ctx.tc[5], self.ctx.tc[3]);
        self.emit.emit_alu(Opcode::Sub, self.ctx.td_6, self.ctx.tc[6], self.ctx.tc[3]);

        self.ctx.v_tc2 = self.emit.emit_vbroadcast(self.ctx.v_tc2, self.ctx.tc[2]);
        self.ctx.v_td_1 = self.emit.emit_vbroadcast(self.ctx.v_td_1, self.ctx.td_1);
        self.ctx.v_tc3 = self.emit.emit_vbroadcast(self.ctx.v_tc3, self.ctx.tc[3]);
        self.ctx.v_td_4 = self.emit.emit_vbroadcast(self.ctx.v_td_4, self.ctx.td_4);
        self.ctx.v_td_5 = self.emit.emit_vbroadcast(self.ctx.v_td_5, self.ctx.td_5);
        self.ctx.v_td_6 = self.emit.emit_vbroadcast(self.ctx.v_td_6, self.ctx.td_6);

        // FMA multipliers - copy multipliers first to avoid borrow issues
        let fma_mults: Vec<(usize, i64)> = (0..6)
            .filter_map(|i| self.ctx.hash_fma_mult[i].map(|m| (i, m)))
            .collect();

        for (i, mult) in &fma_mults {
            let mult_s = self.alloc.alloc(&format!("hash_mult_{}", i), 1);
            let v_mult = self.alloc.vec(&format!("v_hash_mult_{}", i));
            self.ctx.hash_fma_mult_s.insert(*i, mult_s);
            self.emit.emit_const(mult_s, *mult);
            let actual_v_mult = self.emit.emit_vbroadcast(v_mult, mult_s);
            self.ctx.v_hash_mult.insert(*i, actual_v_mult);
        }
    }

    /// Emit common initialization using pseudo-consts for flexible constant generation.
    ///
    /// Instead of a fixed derivation chain (1 -> 2 -> 3 -> ...), this method
    /// declares the needed constants via need_const() and lets the scheduler
    /// dynamically choose between Load, Add, Mul, etc. based on resource availability.
    ///
    /// This reduces early cycle inefficiency by allowing the scheduler to:
    /// - Load critical constants directly when load slots are available
    /// - Derive constants via ALU when load slots are scarce
    /// - Schedule constant generation just-in-time when consumers need them
    ///
    /// NOTE: Temporary address values (in st[0], st[1]) still use emit_const()
    /// because they are reused for different values and don't need pseudo-const flexibility.
    fn emit_common_init_with_pseudo_consts(&mut self) {
        // Register ZERO_REG as known constant 0 (no load needed - scratch is initialized to 0)
        self.emit.known_constants.insert(ZERO_REG, 0);
        self.emit.value_to_register.insert(0, ZERO_REG);

        // === DECLARE SMALL CONSTANTS (0-15) ===
        // The scheduler will choose optimal encodings:
        // - 0: use ZERO_REG (already available)
        // - 1: must Load (no derivation possible)
        // - 2-15: Load or derive from smaller constants
        // NOTE: We must capture the return value from need_const because it may
        // return a different register if the value was already registered.
        self.ctx.c[0] = ZERO_REG;
        for i in 1..16 {
            self.ctx.c[i] = self.emit.need_const(i as i64, self.ctx.c[i]);
        }

        // === LOAD POINTERS ===
        // Use emit_const for temporary address values (st[0], st[1] are reused)
        // These are sequential operations that overwrite the same temporaries
        self.emit.emit_const(self.ctx.st[0], 4);
        self.emit.emit_const(self.ctx.st[1], 5);
        self.emit.emit_load(self.ctx.forest_p, self.ctx.st[0]);
        self.emit.emit_load(self.ctx.idx_p, self.ctx.st[1]);

        self.emit.emit_const(self.ctx.st[0], 6);
        self.emit.emit_const(self.ctx.st[1], 1);
        self.emit.emit_load(self.ctx.val_p, self.ctx.st[0]);
        self.emit.emit_load(self.ctx.n_nodes_s, self.ctx.st[1]);

        // Load extra_room pointer
        self.emit.emit_const(self.ctx.st[0], 7);
        self.emit.emit_load(self.ctx.extra_room_p, self.ctx.st[0]);

        // Compute partition buffer offsets (constant values, not addresses)
        self.emit.emit_const(self.ctx.partition_idx_offset, 0);
        self.emit.emit_const(self.ctx.partition_val_offset, 256);

        // === HASH CONSTANTS ===
        // These are large values that must be loaded directly.
        // Using need_const allows the scheduler to defer them until needed.
        // NOTE: Capture return values to handle potential value deduplication.
        for i in 0..6 {
            self.ctx.hc[i] = self.emit.need_const(HASH_STAGES[i].1, self.ctx.hc[i]);
            self.ctx.hs[i] = self.emit.need_const(HASH_STAGES[i].4, self.ctx.hs[i]);
        }

        // === BROADCASTS ===
        // Vector versions of scalar constants
        self.ctx.v_0 = self.emit.emit_vbroadcast(self.ctx.v_0, self.ctx.c[0]);
        self.ctx.v_1 = self.emit.emit_vbroadcast(self.ctx.v_1, self.ctx.c[1]);
        self.ctx.v_2 = self.emit.emit_vbroadcast(self.ctx.v_2, self.ctx.c[2]);
        self.ctx.v_nn = self.emit.emit_vbroadcast(self.ctx.v_nn, self.ctx.n_nodes_s);

        for i in 0..8 {
            self.ctx.v_const[i] = self.emit.emit_vbroadcast(self.ctx.v_const[i], self.ctx.c[i]);
        }

        for i in 0..6 {
            self.ctx.v_hc[i] = self.emit.emit_vbroadcast(self.ctx.v_hc[i], self.ctx.hc[i]);
            self.ctx.v_hs[i] = self.emit.emit_vbroadcast(self.ctx.v_hs[i], self.ctx.hs[i]);
        }

        // Stage 5 XOR constant
        // NOTE: Capture return value to handle potential value deduplication.
        self.ctx.stage5_xor_s = self.emit.need_const(STAGE5_XOR_CONST, self.ctx.stage5_xor_s);
        self.ctx.v_stage5_xor = self.emit.emit_vbroadcast(self.ctx.v_stage5_xor, self.ctx.stage5_xor_s);

        // === TREE CACHE ===
        // Load tree[0..6] - these are critical for early tree access!
        // Parallel address computation (can parallelize in ALU)
        for i in 0..7 {
            self.emit.emit_alu(Opcode::Add, self.ctx.tc_addr[i], self.ctx.forest_p, self.ctx.c[i]);
        }
        // Then load all values (2 loads per cycle)
        for i in 0..7 {
            self.emit.emit_load(self.ctx.tc[i], self.ctx.tc_addr[i]);
        }

        // Compute deltas
        self.emit.emit_alu(Opcode::Sub, self.ctx.td_1, self.ctx.tc[1], self.ctx.tc[2]);
        self.emit.emit_alu(Opcode::Sub, self.ctx.td_4, self.ctx.tc[4], self.ctx.tc[3]);
        self.emit.emit_alu(Opcode::Sub, self.ctx.td_5, self.ctx.tc[5], self.ctx.tc[3]);
        self.emit.emit_alu(Opcode::Sub, self.ctx.td_6, self.ctx.tc[6], self.ctx.tc[3]);

        self.ctx.v_tc2 = self.emit.emit_vbroadcast(self.ctx.v_tc2, self.ctx.tc[2]);
        self.ctx.v_td_1 = self.emit.emit_vbroadcast(self.ctx.v_td_1, self.ctx.td_1);
        self.ctx.v_tc3 = self.emit.emit_vbroadcast(self.ctx.v_tc3, self.ctx.tc[3]);
        self.ctx.v_td_4 = self.emit.emit_vbroadcast(self.ctx.v_td_4, self.ctx.td_4);
        self.ctx.v_td_5 = self.emit.emit_vbroadcast(self.ctx.v_td_5, self.ctx.td_5);
        self.ctx.v_td_6 = self.emit.emit_vbroadcast(self.ctx.v_td_6, self.ctx.td_6);

        // === FMA MULTIPLIERS ===
        let fma_mults: Vec<(usize, i64)> = (0..6)
            .filter_map(|i| self.ctx.hash_fma_mult[i].map(|m| (i, m)))
            .collect();

        for (i, mult) in &fma_mults {
            let mult_s = self.alloc.alloc(&format!("hash_mult_{}", i), 1);
            let v_mult = self.alloc.vec(&format!("v_hash_mult_{}", i));
            // NOTE: Capture return value to handle potential value deduplication.
            let actual_mult_s = self.emit.need_const(*mult, mult_s);
            self.ctx.hash_fma_mult_s.insert(*i, actual_mult_s);
            let actual_v_mult = self.emit.emit_vbroadcast(v_mult, actual_mult_s);
            self.ctx.v_hash_mult.insert(*i, actual_v_mult);
        }
    }

    fn emit_valu_selection_init(&mut self) {
        // Skip depth 3 initialization if it won't be used
        // Broadcasts will be done lazily in ensure_depth3_broadcasts()
        if !self.uses_depth3 {
            return;
        }

        // Reuse c[7..14] which we derived via ALU (no additional loads needed!)
        // c_r3[i] aliases to c[7+i] for broadcast sources
        for i in 0..8 {
            self.ctx.c_r3[i] = self.ctx.c[7 + i];
        }

        // Load tree[7..14] - parallel address computation
        // First compute all addresses (can parallelize in ALU)
        for i in 0..8 {
            self.emit.emit_alu(Opcode::Add, self.ctx.tc_r3_addr[i], self.ctx.forest_p, self.ctx.c_r3[i]);
        }
        // Then load all values (2 loads per cycle)
        for i in 0..8 {
            self.emit.emit_load(self.ctx.tc_r3[i], self.ctx.tc_r3_addr[i]);
        }

        // Compute tree-reduce deltas for depth 3 (only if tree-reduce enabled)
        // d01 = tree[8] - tree[7], d23 = tree[10] - tree[9], etc.
        // tc_r3[0] = tree[7], tc_r3[1] = tree[8], tc_r3[2] = tree[9], ...
        if self.config.enable_tree_reduce_depth3 {
            self.emit.emit_alu(Opcode::Sub, self.ctx.td_r3_01, self.ctx.tc_r3[1], self.ctx.tc_r3[0]);
            self.emit.emit_alu(Opcode::Sub, self.ctx.td_r3_23, self.ctx.tc_r3[3], self.ctx.tc_r3[2]);
            self.emit.emit_alu(Opcode::Sub, self.ctx.td_r3_45, self.ctx.tc_r3[5], self.ctx.tc_r3[4]);
            self.emit.emit_alu(Opcode::Sub, self.ctx.td_r3_67, self.ctx.tc_r3[7], self.ctx.tc_r3[6]);
        }

        // NOTE: Broadcasts are deferred to ensure_depth3_broadcasts() which is called
        // lazily from emit_depth3_access(). This avoids emitting broadcasts if depth 3
        // is never actually used (e.g., early termination).

        // Depth 4 cache initialization DISABLED - see allocate_valu_selection() for details.
        // The 16-way cached selection would exceed the scratch budget.
    }

    /// Ensure depth 3 broadcasts are initialized.
    ///
    /// This is called lazily from emit_depth3_access() to avoid emitting broadcasts
    /// for depth 3 values if depth 3 is never actually used.
    fn ensure_depth3_broadcasts(&mut self) {
        if self.depth3_broadcasts_initialized {
            return;
        }
        self.depth3_broadcasts_initialized = true;

        // Broadcast tree values for depth 3 (only even indices needed for tree-reduce base values)
        // v_tc_r3[0], [2], [4], [6] are the bases for FMA at level 0
        for i in 0..8 {
            self.ctx.v_tc_r3[i] =
                self.emit.emit_vbroadcast(self.ctx.v_tc_r3[i], self.ctx.tc_r3[i]);
        }

        // Broadcast constants - use CSE, some may already be broadcast (e.g., c[7] to v_const[7])
        for i in 0..8 {
            self.ctx.v_const_r3[i] =
                self.emit.emit_vbroadcast(self.ctx.v_const_r3[i], self.ctx.c_r3[i]);
        }

        // Broadcast tree-reduce deltas (only if tree-reduce enabled)
        if self.config.enable_tree_reduce_depth3 {
            self.ctx.v_td_r3_01 =
                self.emit.emit_vbroadcast(self.ctx.v_td_r3_01, self.ctx.td_r3_01);
            self.ctx.v_td_r3_23 =
                self.emit.emit_vbroadcast(self.ctx.v_td_r3_23, self.ctx.td_r3_23);
            self.ctx.v_td_r3_45 =
                self.emit.emit_vbroadcast(self.ctx.v_td_r3_45, self.ctx.td_r3_45);
            self.ctx.v_td_r3_67 =
                self.emit.emit_vbroadcast(self.ctx.v_td_r3_67, self.ctx.td_r3_67);
        }
    }

    fn emit_partition_init(&mut self) {
        // No initialization needed - we use v_const[0..7] for comparisons
        // which are already initialized in emit_common_init
    }

    // =========================================================================
    // Data Load/Store
    // =========================================================================

    fn emit_load_data(&mut self, n_groups: usize) {
        let ctx = &self.ctx;

        for g in (0..n_groups).step_by(2) {
            let offset_a = (g * VLEN) as i64;
            let offset_b = ((g + 1) * VLEN) as i64;

            self.emit.emit_const(ctx.st[0], offset_a);
            self.emit.emit_const(ctx.st[1], offset_b);

            self.emit.emit_alu(Opcode::Add, ctx.st[2], ctx.idx_p, ctx.st[0]);
            self.emit.emit_alu(Opcode::Add, ctx.st[3], ctx.idx_p, ctx.st[1]);
            self.emit.emit_alu(Opcode::Add, ctx.st[4], ctx.val_p, ctx.st[0]);
            self.emit.emit_alu(Opcode::Add, ctx.st[5], ctx.val_p, ctx.st[1]);

            self.emit.emit_vload(ctx.all_idx[g], ctx.st[2]);
            self.emit.emit_vload(ctx.all_idx[g + 1], ctx.st[3]);
            self.emit.emit_vload(ctx.all_val[g], ctx.st[4]);
            self.emit.emit_vload(ctx.all_val[g + 1], ctx.st[5]);
        }

        // Convert initial values to optimized form
        for g in 0..n_groups {
            self.emit.emit_valu(
                Opcode::Xor,
                ctx.all_val[g],
                ctx.all_val[g],
                ctx.v_stage5_xor,
            );
        }
    }

    fn emit_store_data(&mut self, n_groups: usize) {
        let ctx = &self.ctx;

        for g in (0..n_groups).step_by(2) {
            let offset_a = (g * VLEN) as i64;
            let offset_b = ((g + 1) * VLEN) as i64;

            self.emit.emit_const(ctx.st[0], offset_a);
            self.emit.emit_const(ctx.st[1], offset_b);

            self.emit.emit_alu(Opcode::Add, ctx.st[2], ctx.idx_p, ctx.st[0]);
            self.emit.emit_alu(Opcode::Add, ctx.st[3], ctx.idx_p, ctx.st[1]);
            self.emit.emit_alu(Opcode::Add, ctx.st[4], ctx.val_p, ctx.st[0]);
            self.emit.emit_alu(Opcode::Add, ctx.st[5], ctx.val_p, ctx.st[1]);

            self.emit.emit_vstore(ctx.st[2], ctx.all_idx[g]);
            self.emit.emit_vstore(ctx.st[3], ctx.all_idx[g + 1]);
            self.emit.emit_vstore(ctx.st[4], ctx.all_val[g]);
            self.emit.emit_vstore(ctx.st[5], ctx.all_val[g + 1]);
        }
    }

    // =========================================================================
    // Round Emission
    // =========================================================================

    fn emit_round(&mut self, rnd: usize, tree_depth: usize, n_groups: usize, n_iters: usize) {
        // Lazy partition for sorted vload access
        //
        // Phase 1 (current): Just emit partition stores, don't use sorted vload yet
        // This validates the partition infrastructure works before adding complexity.
        //
        // Phase 2 (future): Use sorted vload access at depths 4+ once partition stores
        // are validated to correctly reorder data.
        //
        // The sorted vload optimization is DISABLED for now - we first need to verify
        // that partition stores work correctly and measure their overhead.
        let _use_sorted_vload = false; // Disabled until partition stores are validated

        // Tree access based on depth - always use normal access patterns for now
        // Note: Depth 4 optimization disabled due to scratch budget constraints.
        // See allocate_valu_selection() for details.
        match tree_depth {
            0 => self.emit_depth0_access(n_iters),
            1 => self.emit_depth1_access(n_iters),
            2 => self.emit_depth2_access(n_iters),
            3 => self.emit_depth3_access(n_iters),
            _ => self.emit_scattered_access(n_iters),
        }

        // XOR
        self.emit_xor(n_groups);

        // Hash
        self.emit_hash(n_groups, n_iters);

        // Branch (includes partition stores)
        self.emit_branch_with_partition(n_groups, n_iters, tree_depth, rnd);
    }

    // =========================================================================
    // Tree Access by Depth
    // =========================================================================

    fn emit_depth0_access(&mut self, n_iters: usize) {
        let ctx = &self.ctx;

        // Depth 0: All at root - broadcast tree[0]
        // Use _force to bypass CSE since we need the value in each specific v_nv location
        for it in 0..n_iters {
            for i in 0..UNROLL {
                self.emit.emit_vbroadcast_force(ctx.v_nv[it][i], ctx.tc[0]);
            }
        }

        // Stage 5 opt
        for it in 0..n_iters {
            for i in 0..UNROLL {
                self.emit.emit_valu(
                    Opcode::Xor,
                    ctx.v_nv[it][i],
                    ctx.v_nv[it][i],
                    ctx.v_stage5_xor,
                );
            }
        }
    }

    fn emit_depth1_access(&mut self, n_iters: usize) {
        let ctx = &self.ctx;

        // Depth 1: Delta selection for 2 values using FMA
        // select(cond, a, b) = FMA(cond, a-b, b) = cond * (a-b) + b
        for it in 0..n_iters {
            let base_g = it * UNROLL;
            for i in 0..UNROLL {
                let t = ctx.v_t[it][i];
                self.emit.emit_valu(Opcode::Eq, t, ctx.all_idx[base_g + i], ctx.v_1);
                self.emit.emit_fma(ctx.v_nv[it][i], t, ctx.v_td_1, ctx.v_tc2);
            }
        }

        // Stage 5 opt
        for it in 0..n_iters {
            for i in 0..UNROLL {
                self.emit.emit_valu(
                    Opcode::Xor,
                    ctx.v_nv[it][i],
                    ctx.v_nv[it][i],
                    ctx.v_stage5_xor,
                );
            }
        }
    }

    fn emit_depth2_access(&mut self, n_iters: usize) {
        let ctx = &self.ctx;

        // Depth 2: Delta selection for 4 values using FMA
        // select(cond, a, b) = FMA(cond, a-b, b) = cond * (a-b) + b
        for it in 0..n_iters {
            let base_g = it * UNROLL;
            for i in 0..UNROLL {
                let t = ctx.v_t[it][i];

                // Start with tc[3], add delta if idx == 4
                self.emit.emit_valu(Opcode::Eq, t, ctx.all_idx[base_g + i], ctx.v_const[4]);
                self.emit.emit_fma(ctx.v_nv[it][i], t, ctx.v_td_4, ctx.v_tc3);

                // Add td_5 if idx == 5
                self.emit.emit_valu(Opcode::Eq, t, ctx.all_idx[base_g + i], ctx.v_const[5]);
                self.emit.emit_fma(ctx.v_nv[it][i], t, ctx.v_td_5, ctx.v_nv[it][i]);

                // Add td_6 if idx == 6
                self.emit.emit_valu(Opcode::Eq, t, ctx.all_idx[base_g + i], ctx.v_const[6]);
                self.emit.emit_fma(ctx.v_nv[it][i], t, ctx.v_td_6, ctx.v_nv[it][i]);
            }
        }

        // Stage 5 opt
        for it in 0..n_iters {
            for i in 0..UNROLL {
                self.emit.emit_valu(
                    Opcode::Xor,
                    ctx.v_nv[it][i],
                    ctx.v_nv[it][i],
                    ctx.v_stage5_xor,
                );
            }
        }
    }

    fn emit_depth3_access(&mut self, n_iters: usize) {
        // Ensure depth 3 broadcasts are initialized (lazy init)
        self.ensure_depth3_broadcasts();

        let ctx = &self.ctx;

        if self.config.enable_tree_reduce_depth3 {
            // Tree-reduce approach for 8-way selection (15 VALU ops vs 16)
            //
            // Indices 7-14 -> idx_adj = idx - 7 gives 0-7 (3 bits)
            //
            // Tree-reduce structure:
            //   Level 0: 4 FMAs for leaf pairs using bit0
            //   Level 1: 2 SUB + 2 FMA combining pairs using bit1
            //   Level 2: 1 SUB + 1 FMA for final result using bit2
            //
            // Register allocation per group:
            //   t = v_t[it][i] - general temp
            //   nv = v_nv[it][i] - result (also used for p01, then q03)
            //   s1 = v_tc_r3[1] - scratch (bit0, then p67)
            //   s2 = v_tc_r3[3] - scratch (bit1)
            //   s3 = v_tc_r3[5] - scratch (bit2)
            //   s4 = v_tc_r3[7] - scratch (p45, then q47)
            //
            // Note: We use v_tc_r3[1,3,5,7] as scratch because tree-reduce only needs
            // the EVEN indexed tree values (v_tc_r3[0,2,4,6]) as bases for level 0 FMAs.
            // The odd-indexed values are implicit in the pre-computed deltas.

            // Scratch register aliases
            let s1 = ctx.v_tc_r3[1]; // bit0, then p67
            let s2 = ctx.v_tc_r3[3]; // bit1
            let s3 = ctx.v_tc_r3[5]; // bit2
            let s4 = ctx.v_tc_r3[7]; // p45, then q47

            for it in 0..n_iters {
                let base_g = it * UNROLL;
                for i in 0..UNROLL {
                    let t = ctx.v_t[it][i];
                    let nv = ctx.v_nv[it][i];
                    let idx = ctx.all_idx[base_g + i];

                    // === Bit extraction (5 VALU ops) ===
                    // idx_adj = idx - 7 gives 0-7 for indices 7-14
                    self.emit.emit_valu(Opcode::Sub, t, idx, ctx.v_const[7]); // idx_adj
                    self.emit.emit_valu(Opcode::And, s1, t, ctx.v_1);         // bit0
                    self.emit.emit_valu(Opcode::Shr, t, t, ctx.v_1);          // idx_adj >> 1
                    self.emit.emit_valu(Opcode::And, s2, t, ctx.v_1);         // bit1
                    self.emit.emit_valu(Opcode::Shr, s3, t, ctx.v_1);         // bit2

                    // === Level 0: 4 FMAs for leaf pairs (4 VALU ops) ===
                    // p01 = bit0 * d01 + tree[7]
                    // p23 = bit0 * d23 + tree[9]
                    // p45 = bit0 * d45 + tree[11]
                    // p67 = bit0 * d67 + tree[13]
                    self.emit.emit_fma(nv, s1, ctx.v_td_r3_01, ctx.v_tc_r3[0]); // p01 in nv
                    self.emit.emit_fma(t, s1, ctx.v_td_r3_23, ctx.v_tc_r3[2]);  // p23 in t
                    self.emit.emit_fma(s4, s1, ctx.v_td_r3_45, ctx.v_tc_r3[4]); // p45 in s4
                    self.emit.emit_fma(s1, s1, ctx.v_td_r3_67, ctx.v_tc_r3[6]); // p67 in s1 (reuse)

                    // === Level 1: Combine pairs (4 VALU ops: 2 SUB + 2 FMA) ===
                    // q03 = bit1 * (p23 - p01) + p01
                    // q47 = bit1 * (p67 - p45) + p45
                    self.emit.emit_valu(Opcode::Sub, t, t, nv);     // dp03 = p23 - p01
                    self.emit.emit_fma(nv, s2, t, nv);              // q03 = bit1 * dp03 + p01
                    self.emit.emit_valu(Opcode::Sub, t, s1, s4);    // dp47 = p67 - p45
                    self.emit.emit_fma(s4, s2, t, s4);              // q47 = bit1 * dp47 + p45

                    // === Level 2: Final result (2 VALU ops: 1 SUB + 1 FMA) ===
                    // result = bit2 * (q47 - q03) + q03
                    self.emit.emit_valu(Opcode::Sub, t, s4, nv);    // d_final = q47 - q03
                    self.emit.emit_fma(nv, s3, t, nv);              // result = bit2 * d_final + q03
                }
            }
        } else {
            // Original serialized FMA approach (16 VALU ops)
            // Compare/select from preloaded tree[7..14]
            // For 8 possible index values, only ONE comparison will be true (exact match).
            // We use: first iteration with Mul to initialize, then FMA to accumulate.
            // This avoids the useless `* v_0 v_0` zero-initialization.
            for it in 0..n_iters {
                let base_g = it * UNROLL;
                for i in 0..UNROLL {
                    let t = ctx.v_t[it][i];

                    // First iteration: initialize result with t * tree_value (no zero-init needed)
                    self.emit.emit_valu(Opcode::Eq, t, ctx.all_idx[base_g + i], ctx.v_const_r3[0]);
                    self.emit.emit_valu(Opcode::Mul, ctx.v_nv[it][i], t, ctx.v_tc_r3[0]);

                    // Remaining iterations: FMA to accumulate
                    for ti in 1..8 {
                        self.emit.emit_valu(Opcode::Eq, t, ctx.all_idx[base_g + i], ctx.v_const_r3[ti]);
                        self.emit.emit_fma(ctx.v_nv[it][i], t, ctx.v_tc_r3[ti], ctx.v_nv[it][i]);
                    }
                }
            }
        }

        // Stage 5 opt
        for it in 0..n_iters {
            for i in 0..UNROLL {
                self.emit.emit_valu(
                    Opcode::Xor,
                    ctx.v_nv[it][i],
                    ctx.v_nv[it][i],
                    ctx.v_stage5_xor,
                );
            }
        }
    }

    // NOTE: emit_depth4_access was removed - the parallel 16-way cached selection
    // approach exceeded scratch budget (400+ words for cached tree values and temps).
    // Depth 4+ continues to use emit_scattered_access().

    fn emit_scattered_access(&mut self, n_iters: usize) {
        let ctx = &self.ctx;

        // Scattered loads for depths 5+
        for vi in 0..VLEN {
            for i in (0..UNROLL).step_by(2) {
                for it in 0..n_iters {
                    let base_g = it * UNROLL;
                    let st_a = ctx.st[2 * it];
                    let st_b = ctx.st[2 * it + 1];

                    self.emit.emit_alu(
                        Opcode::Add,
                        st_a,
                        ctx.forest_p,
                        ctx.all_idx[base_g + i] + vi as i64,
                    );
                    self.emit.emit_alu(
                        Opcode::Add,
                        st_b,
                        ctx.forest_p,
                        ctx.all_idx[base_g + i + 1] + vi as i64,
                    );
                    self.emit.emit_load(ctx.v_nv[it][i] + vi as i64, st_a);
                    self.emit.emit_load(ctx.v_nv[it][i + 1] + vi as i64, st_b);
                }
            }
        }

        // Stage 5 optimization: XOR gathered values with C
        for it in 0..n_iters {
            for i in 0..UNROLL {
                self.emit.emit_valu(
                    Opcode::Xor,
                    ctx.v_nv[it][i],
                    ctx.v_nv[it][i],
                    ctx.v_stage5_xor,
                );
            }
        }
    }

    // =========================================================================
    // XOR Phase
    // =========================================================================

    fn emit_xor(&mut self, n_groups: usize) {
        let ctx = &self.ctx;
        let n_iters = n_groups / UNROLL;

        for it in 0..n_iters {
            let base_g = it * UNROLL;
            for i in 0..UNROLL {
                let g = base_g + i;
                self.emit.emit_valu(Opcode::Xor, ctx.all_val[g], ctx.all_val[g], ctx.v_nv[it][i]);
            }
        }
    }

    // =========================================================================
    // Hash Phase
    // =========================================================================

    fn emit_hash(&mut self, _n_groups: usize, n_iters: usize) {
        let ctx = &self.ctx;

        for hi in 0..6 {
            let (op1, _, op2, op3, _) = HASH_STAGES[hi];

            for it in 0..n_iters {
                let base_g = it * UNROLL;
                for i in 0..UNROLL {
                    let g = base_g + i;

                    if ctx.hash_fma_mult[hi].is_some() {
                        // FMA optimization
                        let v_mult = *ctx.v_hash_mult.get(&hi).unwrap();
                        self.emit.emit_fma(ctx.all_val[g], ctx.all_val[g], v_mult, ctx.v_hc[hi]);
                    } else if hi == 5 {
                        // Stage 5 special handling
                        let t = ctx.v_t[it][i];
                        self.emit.emit_valu(Opcode::Shr, t, ctx.all_val[g], ctx.v_hs[hi]);
                        self.emit.emit_valu(Opcode::Xor, ctx.all_val[g], ctx.all_val[g], t);
                    } else {
                        // General case
                        let t = ctx.v_t[it][i];
                        let op1_code = match op1 {
                            '+' => Opcode::Add,
                            '^' => Opcode::Xor,
                            _ => Opcode::Add,
                        };
                        let op2_code = match op2 {
                            '+' => Opcode::Add,
                            '^' => Opcode::Xor,
                            _ => Opcode::Add,
                        };
                        let op3_code = match op3 {
                            '<' => Opcode::Shl,
                            '>' => Opcode::Shr,
                            _ => Opcode::Shl,
                        };

                        self.emit.emit_valu(op1_code, t, ctx.all_val[g], ctx.v_hc[hi]);
                        self.emit.emit_valu(op3_code, ctx.v_nv[it][i], ctx.all_val[g], ctx.v_hs[hi]);
                        self.emit.emit_valu(op2_code, ctx.all_val[g], t, ctx.v_nv[it][i]);
                    }
                }
            }
        }
    }

    // =========================================================================
    // Branch Phase
    // =========================================================================

    fn emit_branch_with_partition(&mut self, n_groups: usize, n_iters: usize, tree_depth: usize, rnd: usize) {
        let ctx = &self.ctx;
        let is_wrap_round = tree_depth == self.config.forest_height;

        // Emit branch computation for all groups
        for it in 0..n_iters {
            let base_g = it * UNROLL;
            for i in 0..UNROLL {
                let g = base_g + i;
                let t = ctx.v_t[it][i];

                if is_wrap_round {
                    // At leaf level: ALL items wrap to index 0
                    // Use vbroadcast_force to write 0 directly, avoiding useless `* v_0 v_0`
                    self.emit.emit_vbroadcast_force(ctx.all_idx[g], ZERO_REG);
                } else {
                    // Normal branching
                    self.emit.emit_valu(Opcode::And, t, ctx.all_val[g], ctx.v_1);
                    self.emit.emit_valu(Opcode::Sub, t, ctx.v_2, t);
                    self.emit.emit_fma(ctx.all_idx[g], ctx.all_idx[g], ctx.v_2, t);
                }
            }
        }

        // Partition stores for lazy partition optimization
        //
        // DISABLED: Partition stores add too much overhead (~13,000 ops, 4.3x slower).
        // Measurement results (2024-01):
        //   - Baseline: 1,759 cycles, 15,641 ops
        //   - With partition stores: 7,554 cycles, 28,993 ops
        //
        // The stores are NOT "free" as originally theorized - they need to be
        // properly interleaved with the other half's hash computation to overlap.
        // Current implementation emits stores sequentially, adding to critical path.
        //
        // TODO: To make this work, either:
        //   1. Actually interleave TIK stores with TOK hash (and vice versa)
        //   2. Use cheaper partition algorithm (parallel prefix-sum)
        //   3. Skip partition, use wider selection window (16-way select)
        if false && self.config.enable_lazy_partition
            && should_partition(rnd, self.config.rounds, self.config.partition_frequency)
        {
            // Emit partition stores to the write buffer (currently disabled)
            let read_buffer = get_read_buffer(rnd, self.config.partition_frequency);
            let write_buffer = read_buffer.write_target();

            let mut helper = PartitionHelper {
                ctx: &self.ctx,
                emit: &mut self.emit,
                config: &self.config,
            };
            helper.emit_partition_stores(write_buffer, n_groups);
        } else {
            // Use original tick-tock stores for load balancing
            if TICK_STORE_DEPTHS.contains(&tree_depth) {
                // Store first half: groups 0-15
                self.emit_partition_stores_batch(0, n_groups / 2);
            } else if TOCK_STORE_DEPTHS.contains(&tree_depth) {
                // Store second half: groups 16-31
                self.emit_partition_stores_batch(n_groups / 2, n_groups);
            }
        }
    }

    fn emit_partition_stores_batch(&mut self, start_g: usize, end_g: usize) {
        let ctx = &self.ctx;

        // Compute base addresses once
        let base_offset = (start_g * VLEN) as i64;
        self.emit.emit_const(ctx.st[8], base_offset);
        self.emit.emit_alu(Opcode::Add, ctx.st[9], ctx.extra_room_p, ctx.st[8]);
        self.emit.emit_alu(Opcode::Add, ctx.st[10], ctx.st[9], ctx.partition_val_offset);
        self.emit.emit_const(ctx.st[11], VLEN as i64);

        // Store all groups with incremental addressing
        for g in start_g..end_g {
            // Store indices and values
            self.emit.emit_vstore(ctx.st[9], ctx.all_idx[g]);
            self.emit.emit_vstore(ctx.st[10], ctx.all_val[g]);

            // Increment addresses for next group (except for last)
            if g < end_g - 1 {
                self.emit.emit_alu(Opcode::Add, ctx.st[9], ctx.st[9], ctx.st[11]);
                self.emit.emit_alu(Opcode::Add, ctx.st[10], ctx.st[10], ctx.st[11]);
            }
        }
    }
}
