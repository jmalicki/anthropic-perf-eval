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
}

impl TickTockKernelBuilder {
    /// Create a new kernel builder with the given configuration
    pub fn new(config: KernelConfig) -> Self {
        TickTockKernelBuilder {
            config,
            ctx: ScratchContext::default(),
            alloc: ScratchAllocator::default(),
            emit: EmitContext::new(),
        }
    }

    /// Create a builder with default configuration
    pub fn default_config() -> Self {
        Self::new(KernelConfig::default())
    }

    /// Build the kernel, returning the list of unscheduled ops
    pub fn build(&mut self) -> Vec<Op> {
        let n_groups = self.config.n_groups();
        let n_iters = self.config.n_iters();

        // Allocate scratch
        self.allocate_common(n_groups, n_iters);
        self.allocate_partition();
        self.allocate_valu_selection();

        let usage = self.alloc.usage();
        if usage > SCRATCH_SIZE as i64 {
            eprintln!(
                "WARNING: Scratch usage {} exceeds budget {}!",
                usage, SCRATCH_SIZE
            );
        }

        // Emit initialization
        self.emit_common_init();
        self.emit_valu_selection_init();

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

        // Return the ops
        self.emit.take_ops()
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

        // Tree cache for sorted vload access
        ctx.tree_cache_scalars = alloc.alloc("tree_cache_scalars", VLEN);
        ctx.tree_cache_base_s = alloc.alloc("tree_cache_base", 1);
        ctx.v_cache_base = alloc.vec("v_cache_base_tt");

        // Temp vector for tree operations
        ctx.v_tree_temp = alloc.vec("v_tree_temp");

        // Partition buffer offsets
        ctx.partition_idx_offset = alloc.alloc("part_idx_off", 1);
        ctx.partition_val_offset = alloc.alloc("part_val_off", 1);
    }

    fn allocate_valu_selection(&mut self) {
        let alloc = &mut self.alloc;
        let ctx = &mut self.ctx;

        // Depth 3 cache
        for i in 0..8 {
            ctx.tc_r3[i] = alloc.alloc(&format!("tc_r3_{}", i), 1);
        }
        for i in 0..8 {
            ctx.v_tc_r3[i] = alloc.vec(&format!("v_tc_r3_{}", i));
        }
        for i in 0..8 {
            ctx.v_const_r3[i] = alloc.vec(&format!("v_const_r3_{}", i));
        }
        for i in 0..8 {
            ctx.c_r3[i] = alloc.alloc(&format!("c_r3_{}", i), 1);
        }
    }

    // =========================================================================
    // Initialization
    // =========================================================================

    fn emit_common_init(&mut self) {
        // Register ZERO_REG as known constant 0 (no load needed - scratch is initialized to 0)
        self.emit.known_constants.insert(ZERO_REG, 0);

        // === CONSTANT DERIVATION: 1 Load op + 14 ALU ops (vs 15 Load ops before) ===
        // This frees 14 Load slots for actual data fetches!
        //
        // Derivation tree (minimum depth = 4 cycles):
        //   Cycle 0: Load 1
        //   Cycle 1: 2 = 1+1
        //   Cycle 2: 3 = 2+1, 4 = 2+2
        //   Cycle 3: 5 = 4+1, 6 = 3+3, 7 = 4+3, 8 = 4+4
        //   Cycle 4: 9=8+1, 10=5+5, 11=8+3, 12=6+6, 13=8+5, 14=7+7, 15=8+7

        // Bootstrap: Load only constant 1
        self.emit.emit_const(self.ctx.c[1], 1);

        // Derive 2-15 using ALU (scheduler will parallelize these as dependencies allow)
        // Cycle +1: 2 = 1+1
        self.emit
            .emit_alu(Opcode::Add, self.ctx.c[2], self.ctx.c[1], self.ctx.c[1]);

        // Cycle +2: 3 = 2+1, 4 = 2+2 (parallel)
        self.emit
            .emit_alu(Opcode::Add, self.ctx.c[3], self.ctx.c[2], self.ctx.c[1]);
        self.emit
            .emit_alu(Opcode::Add, self.ctx.c[4], self.ctx.c[2], self.ctx.c[2]);

        // Cycle +3: 5, 6, 7, 8 (parallel)
        self.emit
            .emit_alu(Opcode::Add, self.ctx.c[5], self.ctx.c[4], self.ctx.c[1]);
        self.emit
            .emit_alu(Opcode::Add, self.ctx.c[6], self.ctx.c[3], self.ctx.c[3]);
        self.emit
            .emit_alu(Opcode::Add, self.ctx.c[7], self.ctx.c[4], self.ctx.c[3]);
        self.emit
            .emit_alu(Opcode::Add, self.ctx.c[8], self.ctx.c[4], self.ctx.c[4]);

        // Cycle +4: 9, 10, 11, 12, 13, 14, 15 (parallel)
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

        // Load tree cache
        for i in 0..7 {
            self.emit.emit_alu(Opcode::Add, self.ctx.st[0], self.ctx.forest_p, self.ctx.c[i]);
            self.emit.emit_load(self.ctx.tc[i], self.ctx.st[0]);
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

        for (i, mult) in fma_mults {
            let mult_s = self.alloc.alloc(&format!("hash_mult_{}", i), 1);
            let v_mult = self.alloc.vec(&format!("v_hash_mult_{}", i));
            self.ctx.hash_fma_mult_s.insert(i, mult_s);
            self.emit.emit_const(mult_s, mult);
            let actual_v_mult = self.emit.emit_vbroadcast(v_mult, mult_s);
            self.ctx.v_hash_mult.insert(i, actual_v_mult);
        }
    }

    fn emit_valu_selection_init(&mut self) {
        // Reuse c[7..14] which we derived via ALU (no additional loads needed!)
        // c_r3[i] aliases to c[7+i] for broadcast sources
        for i in 0..8 {
            self.ctx.c_r3[i] = self.ctx.c[7 + i];
        }

        // Load tree[7..14]
        for i in 0..8 {
            self.emit
                .emit_alu(Opcode::Add, self.ctx.st[0], self.ctx.forest_p, self.ctx.c_r3[i]);
            self.emit.emit_load(self.ctx.tc_r3[i], self.ctx.st[0]);
        }

        // Broadcast tree values
        for i in 0..8 {
            self.ctx.v_tc_r3[i] =
                self.emit.emit_vbroadcast(self.ctx.v_tc_r3[i], self.ctx.tc_r3[i]);
        }

        // Broadcast constants - use CSE, some may already be broadcast (e.g., c[7] to v_const[7])
        for i in 0..8 {
            self.ctx.v_const_r3[i] =
                self.emit.emit_vbroadcast(self.ctx.v_const_r3[i], self.ctx.c_r3[i]);
        }
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

    fn emit_round(&mut self, _rnd: usize, tree_depth: usize, n_groups: usize, n_iters: usize) {
        // Tree access based on depth
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
        self.emit_branch(n_groups, n_iters, tree_depth);
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

        // Depth 1: Delta selection for 2 values
        for it in 0..n_iters {
            let base_g = it * UNROLL;
            for i in 0..UNROLL {
                let t = ctx.v_t[it][i];
                self.emit.emit_valu(Opcode::Eq, t, ctx.all_idx[base_g + i], ctx.v_1);
                self.emit.emit_valu(Opcode::Mul, t, ctx.v_td_1, t);
                self.emit.emit_valu(Opcode::Add, ctx.v_nv[it][i], ctx.v_tc2, t);
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

        // Depth 2: Delta selection for 4 values
        for it in 0..n_iters {
            let base_g = it * UNROLL;
            for i in 0..UNROLL {
                let t = ctx.v_t[it][i];

                // Start with tc[3]
                self.emit.emit_valu(Opcode::Eq, t, ctx.all_idx[base_g + i], ctx.v_const[4]);
                self.emit.emit_valu(Opcode::Mul, t, ctx.v_td_4, t);
                self.emit.emit_valu(Opcode::Add, ctx.v_nv[it][i], ctx.v_tc3, t);

                // Add td_5 if idx == 5
                self.emit.emit_valu(Opcode::Eq, t, ctx.all_idx[base_g + i], ctx.v_const[5]);
                self.emit.emit_valu(Opcode::Mul, t, ctx.v_td_5, t);
                self.emit.emit_valu(Opcode::Add, ctx.v_nv[it][i], ctx.v_nv[it][i], t);

                // Add td_6 if idx == 6
                self.emit.emit_valu(Opcode::Eq, t, ctx.all_idx[base_g + i], ctx.v_const[6]);
                self.emit.emit_valu(Opcode::Mul, t, ctx.v_td_6, t);
                self.emit.emit_valu(Opcode::Add, ctx.v_nv[it][i], ctx.v_nv[it][i], t);
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
        let ctx = &self.ctx;

        // Depth 3: Compare/select from preloaded tree[7..14]
        for it in 0..n_iters {
            let base_g = it * UNROLL;
            for i in 0..UNROLL {
                let t = ctx.v_t[it][i];

                // Initialize result to 0
                self.emit.emit_valu(Opcode::Mul, ctx.v_nv[it][i], ctx.v_0, ctx.v_0);

                // For each of 8 possible values
                for ti in 0..8 {
                    self.emit.emit_valu(Opcode::Eq, t, ctx.all_idx[base_g + i], ctx.v_const_r3[ti]);
                    self.emit.emit_fma(ctx.v_nv[it][i], t, ctx.v_tc_r3[ti], ctx.v_nv[it][i]);
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

    fn emit_scattered_access(&mut self, n_iters: usize) {
        let ctx = &self.ctx;

        // Scattered loads for depths 4+
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

    fn emit_branch(&mut self, n_groups: usize, n_iters: usize, tree_depth: usize) {
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
                    self.emit.emit_valu(Opcode::Mul, ctx.all_idx[g], ctx.v_0, ctx.v_0);
                } else {
                    // Normal branching
                    self.emit.emit_valu(Opcode::And, t, ctx.all_val[g], ctx.v_1);
                    self.emit.emit_valu(Opcode::Sub, t, ctx.v_2, t);
                    self.emit.emit_fma(ctx.all_idx[g], ctx.all_idx[g], ctx.v_2, t);
                }
            }
        }

        // Emit partition stores based on DEPTH
        if TICK_STORE_DEPTHS.contains(&tree_depth) {
            // Store first half: groups 0-15
            self.emit_partition_stores_batch(0, n_groups / 2);
        } else if TOCK_STORE_DEPTHS.contains(&tree_depth) {
            // Store second half: groups 16-31
            self.emit_partition_stores_batch(n_groups / 2, n_groups);
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
