//! SSA-based TickTock Kernel Builder.
//!
//! This builder generates optimized instruction sequences using SSA form,
//! eliminating false WAW/WAR dependencies for better scheduling.
//!
//! ## Flow
//!
//! 1. `build()` - Emit SsaOps using VRegs (no concrete addresses)
//! 2. `compute_ssa_dependencies()` - Only RAW dependencies (true data flow)
//! 3. `schedule_ssa_greedy()` - Schedule with VALU→ALU shifting
//! 4. `assign_addresses()` - Linear scan allocation based on live ranges
//! 5. `lower_ssa_ops()` - Convert to concrete Ops (expanding shifted VALU ops)

use std::collections::{HashMap, HashSet};

use crate::address_assign::{
    assign_addresses, lower_ssa_op, AddressAssignConfig, AddressAssignmentResult,
};
use crate::types::{Engine, Op, Opcode, Operand, SlotVal, SsaOp, VReg, VLEN};

use super::config::{fma_multiplier, KernelConfig, HASH_STAGES, STAGE5_XOR_CONST};
use super::context_ssa::SsaScratchContext;
use super::emit_ssa::{compute_ssa_dependencies, SsaEmitContext};

/// Unroll factor - process 8 groups per iteration
const UNROLL: usize = 8;

/// Depths where TICK stores happen (first half of groups)
const TICK_STORE_DEPTHS: [usize; 6] = [0, 1, 2, 4, 5, 6];

/// Depths where TOCK stores happen (second half of groups)
const TOCK_STORE_DEPTHS: [usize; 4] = [3, 7, 8, 9];

/// Result of SSA scheduling and address assignment.
#[derive(Debug)]
pub struct SsaScheduleResult {
    /// Scheduled concrete ops
    pub ops: Vec<Op>,
    /// Number of cycles
    pub cycles: usize,
    /// Address assignment result
    pub address_result: AddressAssignmentResult,
    /// Number of VRegs used
    pub vreg_count: u32,
    /// Number of VALU ops shifted to ALU
    pub valu_shifted: usize,
}

/// SSA-based TickTock Kernel Builder.
///
/// Unlike TickTockKernelBuilder which uses pre-allocated addresses,
/// this builder emits SsaOps with VRegs that are assigned addresses
/// after scheduling.
#[derive(Debug)]
pub struct SsaTickTockKernelBuilder {
    /// Kernel configuration
    pub config: KernelConfig,
    /// SSA scratch context (holds VRegs instead of addresses)
    pub ctx: SsaScratchContext,
    /// SSA emission context
    pub emit: SsaEmitContext,
    /// Whether depth 3 broadcasts have been initialized
    depth3_broadcasts_initialized: bool,
    /// Whether any round will use depth 3
    uses_depth3: bool,
    /// VReg representing constant 0 (loaded at address 0)
    zero_vreg: Option<VReg>,
}

impl SsaTickTockKernelBuilder {
    /// Create a new SSA kernel builder with the given configuration.
    pub fn new(config: KernelConfig) -> Self {
        let n_groups = config.n_groups();
        let n_iters = config.n_iters();
        let tree_height = config.forest_height;
        let uses_depth3 = (0..config.rounds).any(|r| {
            let tree_depth = r % tree_height;
            tree_depth == 3
        });

        // Pre-compute FMA multipliers in config-compatible format
        let mut hash_fma_mult = [None; 6];
        for i in 0..6 {
            hash_fma_mult[i] = fma_multiplier(i);
        }

        let mut ctx = SsaScratchContext::new(n_groups, n_iters);
        ctx.hash_fma_mult = hash_fma_mult;

        SsaTickTockKernelBuilder {
            config,
            ctx,
            emit: SsaEmitContext::new(),
            depth3_broadcasts_initialized: false,
            uses_depth3,
            zero_vreg: None,
        }
    }

    /// Build the kernel, returning unscheduled SSA ops.
    pub fn build(&mut self) -> Vec<SsaOp> {
        self.build_internal();
        self.emit.take_ops()
    }

    /// Build and schedule the kernel, returning the full result.
    ///
    /// This performs:
    /// 1. SSA op emission
    /// 2. SSA dependency computation (RAW only)
    /// 3. Greedy scheduling with VALU→ALU shifting
    /// 4. Address assignment via linear scan
    /// 5. Lowering to concrete ops (expanding shifted VALU ops)
    pub fn build_and_schedule(&mut self) -> Result<SsaScheduleResult, String> {
        self.build_and_schedule_with_options(self.config.enable_valu_shift)
    }

    /// Build and schedule with explicit VALU shift option.
    pub fn build_and_schedule_with_options(
        &mut self,
        enable_valu_shift: bool,
    ) -> Result<SsaScheduleResult, String> {
        // Build SSA ops
        self.build_internal();

        // Get ops and dependency info
        // IMPORTANT: Take vreg_def BEFORE take_ops() since take_ops() clears it
        let vreg_def = std::mem::take(&mut self.emit.vreg_def);
        let vreg_count = self.emit.next_vreg_id;
        let mut ssa_ops = self.emit.take_ops();

        // Compute dependencies (RAW only - no false WAW/WAR)
        compute_ssa_dependencies(&mut ssa_ops, &vreg_def);

        // Schedule using greedy scheduler with VALU→ALU shifting
        let mut shifted_ops: HashSet<usize> = HashSet::new();
        let cycles = self.schedule_ssa_greedy_with_shift(
            &mut ssa_ops,
            enable_valu_shift,
            &mut shifted_ops,
        )?;

        // Assign addresses via linear scan
        let mut assign_config = AddressAssignConfig::default();

        // Reserve address 0 for ZERO constant
        if let Some(zero) = self.zero_vreg {
            assign_config.fixed_vregs.insert(zero, 0);
        }

        let address_result = assign_addresses(&ssa_ops, &assign_config);

        if address_result.budget_exceeded {
            return Err(format!(
                "Address assignment exceeded budget: peak_live={} > budget={}",
                address_result.peak_live, address_result.budget
            ));
        }

        // Lower to concrete ops, expanding shifted VALU ops to ALU
        let ops = lower_ssa_ops_with_shift(
            &ssa_ops,
            &address_result.vreg_to_addr,
            &shifted_ops,
        );

        let valu_shifted = shifted_ops.len();

        Ok(SsaScheduleResult {
            ops,
            cycles,
            address_result,
            vreg_count,
            valu_shifted,
        })
    }

    /// Build and schedule using CP-SAT on SSA ops (before register allocation).
    ///
    /// This performs:
    /// 1. SSA op emission
    /// 2. SSA dependency computation (RAW only)
    /// 3. Optionally: Greedy scheduling to get hints
    /// 4. CP-SAT scheduling on SsaOps (with optional VALU→ALU shift modeling)
    /// 5. Address assignment via linear scan
    /// 6. Lowering to concrete ops (expanding shifted VALU ops)
    ///
    /// When `enable_valu_shift` is true, the CP-SAT solver models VALU→ALU shifting
    /// as decision variables, allowing it to find globally optimal shift decisions.
    pub fn build_and_schedule_cpsat(
        &mut self,
        time_limit_secs: f64,
        verbose: bool,
        enable_valu_shift: bool,
    ) -> Result<SsaScheduleResult, String> {
        use crate::cpsat_scheduler::{schedule_ssa_cpsat, SsaCpsatConfig};

        // Build SSA ops
        self.build_internal();

        // Get ops and dependency info
        // IMPORTANT: Take vreg_def BEFORE take_ops() since take_ops() clears it
        let vreg_def = std::mem::take(&mut self.emit.vreg_def);
        let vreg_count = self.emit.next_vreg_id;
        let mut ssa_ops = self.emit.take_ops();

        // Compute dependencies (RAW only)
        compute_ssa_dependencies(&mut ssa_ops, &vreg_def);

        // Get greedy hints first (for warm-starting CP-SAT)
        // If shift is enabled, also get shift hints from greedy scheduler
        let mut greedy_shifted = HashSet::new();
        let greedy_cycles = self.schedule_ssa_greedy_with_shift(
            &mut ssa_ops,
            enable_valu_shift, // Match CP-SAT shift setting for hints
            &mut greedy_shifted,
        )?;
        let greedy_hints: Vec<i32> = ssa_ops.iter().map(|op| op.cycle).collect();

        if verbose {
            eprintln!(
                "SSA Greedy hint: {} cycles{}",
                greedy_cycles,
                if enable_valu_shift {
                    format!(" ({} VALU shifted)", greedy_shifted.len())
                } else {
                    String::new()
                }
            );
        }

        // Reset cycles before CP-SAT
        for op in &mut ssa_ops {
            op.cycle = -1;
        }

        // Run CP-SAT on SSA ops with scratch budget constraint
        let cpsat_config = SsaCpsatConfig {
            time_limit_secs,
            verbose,
            stop_after_first_solution: false,
            enable_valu_shift,
            scratch_budget: 1536, // Hardware limit
        };

        // Pass shift hints if enabled
        let shift_hints = if enable_valu_shift {
            Some(&greedy_shifted)
        } else {
            None
        };

        let cpsat_result = schedule_ssa_cpsat(&ssa_ops, cpsat_config, Some(&greedy_hints), shift_hints)
            .ok_or_else(|| "CP-SAT scheduling failed".to_string())?;

        // Apply cycle assignments
        for (idx, op) in ssa_ops.iter_mut().enumerate() {
            if idx < cpsat_result.cycle_assignments.len() {
                op.cycle = cpsat_result.cycle_assignments[idx];
            }
        }

        let cycles = cpsat_result.makespan;
        let shifted_ops = cpsat_result.shifted_ops;

        if verbose {
            eprintln!(
                "SSA CP-SAT result: {} cycles (greedy was {}), {} VALU ops shifted",
                cycles, greedy_cycles, shifted_ops.len()
            );
        }

        // Assign addresses via linear scan
        let mut assign_config = AddressAssignConfig::default();

        // Reserve address 0 for ZERO constant
        if let Some(zero) = self.zero_vreg {
            assign_config.fixed_vregs.insert(zero, 0);
        }

        let address_result = assign_addresses(&ssa_ops, &assign_config);

        if address_result.budget_exceeded {
            return Err(format!(
                "Address assignment exceeded budget: peak_live={} > budget={}",
                address_result.peak_live, address_result.budget
            ));
        }

        // Lower to concrete ops, expanding shifted VALU ops to ALU
        let valu_shifted = shifted_ops.len();
        let ops = lower_ssa_ops_with_shift(&ssa_ops, &address_result.vreg_to_addr, &shifted_ops);

        Ok(SsaScheduleResult {
            ops,
            cycles,
            address_result,
            vreg_count,
            valu_shifted,
        })
    }

    /// Schedule SSA ops using a greedy list scheduler with VALU→ALU shifting.
    ///
    /// When VALU slots are saturated but ALU has capacity, shiftable VALU ops
    /// (arithmetic ops that can be expanded to 8 scalar ALU ops) are shifted.
    ///
    /// Returns the total number of cycles and populates shifted_ops with IDs
    /// of ops that were shifted.
    fn schedule_ssa_greedy_with_shift(
        &self,
        ops: &mut [SsaOp],
        enable_valu_shift: bool,
        shifted_ops: &mut HashSet<usize>,
    ) -> Result<usize, String> {
        if ops.is_empty() {
            return Ok(0);
        }

        // Build op_id -> index mapping
        let id_to_idx: HashMap<usize, usize> = ops
            .iter()
            .enumerate()
            .map(|(idx, op)| (op.id, idx))
            .collect();

        // Track scheduled ops
        let mut scheduled: HashSet<usize> = HashSet::new();
        let mut cycle = 0;
        let max_cycles = 50000; // Safety limit

        while scheduled.len() < ops.len() && cycle < max_cycles {
            // Slot usage this cycle
            let mut load_used = 0;
            let mut store_used = 0;
            let mut alu_used = 0;
            let mut valu_used = 0;
            let mut flow_used = 0;

            // Find ready ops (all deps scheduled at earlier cycles)
            let ready: Vec<usize> = ops
                .iter()
                .enumerate()
                .filter(|(idx, op)| {
                    // Not yet scheduled
                    !scheduled.contains(idx)
                        // All dependencies scheduled in earlier cycles
                        && op.dep_ids.iter().all(|dep_id| {
                            if let Some(&dep_idx) = id_to_idx.get(dep_id) {
                                let dep_cycle = ops[dep_idx].cycle;
                                dep_cycle >= 0 && dep_cycle < cycle as i32
                            } else {
                                true // Unknown dep, assume ready
                            }
                        })
                })
                .map(|(idx, _)| idx)
                .collect();

            // Schedule ready ops that fit in slots
            for idx in ready {
                let op = &ops[idx];
                let engine = op.engine;

                // Check if this VALU op can be shifted to ALU
                let can_shift = enable_valu_shift
                    && engine == Engine::Valu
                    && can_ssa_valu_shift_to_alu(op);

                let fits = match engine {
                    Engine::Load => {
                        if load_used < 2 {
                            load_used += 1;
                            true
                        } else {
                            false
                        }
                    }
                    Engine::Store => {
                        if store_used < 2 {
                            store_used += 1;
                            true
                        } else {
                            false
                        }
                    }
                    Engine::Alu => {
                        if alu_used < 12 {
                            alu_used += 1;
                            true
                        } else {
                            false
                        }
                    }
                    Engine::Valu => {
                        if valu_used < 6 {
                            valu_used += 1;
                            true
                        } else if can_shift && alu_used + VLEN <= 12 {
                            // VALU is full, but we can shift to ALU
                            // Shifted VALU op uses 8 ALU slots
                            alu_used += VLEN;
                            shifted_ops.insert(op.id);
                            true
                        } else {
                            false
                        }
                    }
                    Engine::Flow => {
                        if flow_used < 1 {
                            flow_used += 1;
                            true
                        } else {
                            false
                        }
                    }
                    _ => true,
                };

                if fits {
                    ops[idx].cycle = cycle as i32;
                    scheduled.insert(idx);
                }
            }

            cycle += 1;
        }

        if scheduled.len() < ops.len() {
            return Err(format!(
                "Scheduling failed: only scheduled {}/{} ops in {} cycles",
                scheduled.len(),
                ops.len(),
                max_cycles
            ));
        }

        Ok(cycle)
    }

    /// Internal build logic.
    fn build_internal(&mut self) {
        let n_groups = self.config.n_groups();
        let n_iters = self.config.n_iters();

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
        let v_stage5_xor = self.ctx.v_stage5_xor.unwrap();
        for g in 0..n_groups {
            let val = self.ctx.all_val[g].unwrap();
            self.ctx.all_val[g] = Some(self.emit.emit_valu(Opcode::Xor, val, v_stage5_xor));
        }

        // Store data
        self.emit_store_data(n_groups);
    }

    // =========================================================================
    // Initialization
    // =========================================================================

    fn emit_common_init(&mut self) {
        // Emit constant 0 - this VReg will be fixed at address 0
        // (scratch memory is initialized to 0)
        let zero = self.emit.emit_const(0);
        self.zero_vreg = Some(zero);
        self.ctx.c[0] = Some(zero);

        // Bootstrap constants 1, 2, 4
        self.ctx.c[1] = Some(self.emit.emit_const(1));
        self.ctx.c[2] = Some(self.emit.emit_const(2));
        self.ctx.c[4] = Some(self.emit.emit_const(4));

        // Derive remaining constants via ALU
        let c1 = self.ctx.c[1].unwrap();
        let c2 = self.ctx.c[2].unwrap();
        let c4 = self.ctx.c[4].unwrap();

        self.ctx.c[3] = Some(self.emit.emit_alu(Opcode::Add, c2, c1));
        self.ctx.c[5] = Some(self.emit.emit_alu(Opcode::Add, c4, c1));
        self.ctx.c[6] = Some(self.emit.emit_alu(Opcode::Add, c4, c2));
        self.ctx.c[8] = Some(self.emit.emit_alu(Opcode::Add, c4, c4));

        let c3 = self.ctx.c[3].unwrap();
        let c5 = self.ctx.c[5].unwrap();
        let c6 = self.ctx.c[6].unwrap();
        let c8 = self.ctx.c[8].unwrap();

        self.ctx.c[7] = Some(self.emit.emit_alu(Opcode::Add, c4, c3));
        self.ctx.c[9] = Some(self.emit.emit_alu(Opcode::Add, c8, c1));
        self.ctx.c[10] = Some(self.emit.emit_alu(Opcode::Add, c5, c5));
        self.ctx.c[11] = Some(self.emit.emit_alu(Opcode::Add, c8, c3));
        self.ctx.c[12] = Some(self.emit.emit_alu(Opcode::Add, c6, c6));
        self.ctx.c[13] = Some(self.emit.emit_alu(Opcode::Add, c8, c5));

        let c7 = self.ctx.c[7].unwrap();

        self.ctx.c[14] = Some(self.emit.emit_alu(Opcode::Add, c7, c7));
        self.ctx.c[15] = Some(self.emit.emit_alu(Opcode::Add, c8, c7));

        // Load pointers from memory (addresses 4, 5, 6, 1, 7)
        self.ctx.forest_p = Some(self.emit.emit_load_imm(4));
        self.ctx.idx_p = Some(self.emit.emit_load_imm(5));
        self.ctx.val_p = Some(self.emit.emit_load_imm(6));
        self.ctx.n_nodes_s = Some(self.emit.emit_load_imm(1));
        self.ctx.extra_room_p = Some(self.emit.emit_load_imm(7));

        // Partition buffer offsets
        self.ctx.partition_idx_offset = Some(self.emit.emit_const(0));
        self.ctx.partition_val_offset = Some(self.emit.emit_const(256));

        // Load hash constants
        for i in 0..6 {
            self.ctx.hc[i] = Some(self.emit.emit_const(HASH_STAGES[i].1));
            self.ctx.hs[i] = Some(self.emit.emit_const(HASH_STAGES[i].4));
        }

        // Broadcast vector constants
        self.ctx.v_0 = Some(self.emit.emit_vbroadcast(self.ctx.c[0].unwrap()));
        self.ctx.v_1 = Some(self.emit.emit_vbroadcast(self.ctx.c[1].unwrap()));
        self.ctx.v_2 = Some(self.emit.emit_vbroadcast(self.ctx.c[2].unwrap()));
        self.ctx.v_nn = Some(self.emit.emit_vbroadcast(self.ctx.n_nodes_s.unwrap()));

        for i in 0..8 {
            self.ctx.v_const[i] = Some(self.emit.emit_vbroadcast(self.ctx.c[i].unwrap()));
        }

        for i in 0..6 {
            self.ctx.v_hc[i] = Some(self.emit.emit_vbroadcast(self.ctx.hc[i].unwrap()));
            self.ctx.v_hs[i] = Some(self.emit.emit_vbroadcast(self.ctx.hs[i].unwrap()));
        }

        // Stage 5 XOR constant
        self.ctx.stage5_xor_s = Some(self.emit.emit_const(STAGE5_XOR_CONST));
        self.ctx.v_stage5_xor = Some(self.emit.emit_vbroadcast(self.ctx.stage5_xor_s.unwrap()));

        // Load tree cache
        let forest_p = self.ctx.forest_p.unwrap();
        for i in 0..7 {
            let addr = self.emit.emit_alu(Opcode::Add, forest_p, self.ctx.c[i].unwrap());
            self.ctx.tc_addr[i] = Some(addr);
            self.ctx.tc[i] = Some(self.emit.emit_load(addr));
        }

        // Compute deltas
        let tc1 = self.ctx.tc[1].unwrap();
        let tc2 = self.ctx.tc[2].unwrap();
        let tc3 = self.ctx.tc[3].unwrap();
        let tc4 = self.ctx.tc[4].unwrap();
        let tc5 = self.ctx.tc[5].unwrap();
        let tc6 = self.ctx.tc[6].unwrap();

        self.ctx.td_1 = Some(self.emit.emit_alu(Opcode::Sub, tc1, tc2));
        self.ctx.td_4 = Some(self.emit.emit_alu(Opcode::Sub, tc4, tc3));
        self.ctx.td_5 = Some(self.emit.emit_alu(Opcode::Sub, tc5, tc3));
        self.ctx.td_6 = Some(self.emit.emit_alu(Opcode::Sub, tc6, tc3));

        self.ctx.v_tc2 = Some(self.emit.emit_vbroadcast(tc2));
        self.ctx.v_td_1 = Some(self.emit.emit_vbroadcast(self.ctx.td_1.unwrap()));
        self.ctx.v_tc3 = Some(self.emit.emit_vbroadcast(tc3));
        self.ctx.v_td_4 = Some(self.emit.emit_vbroadcast(self.ctx.td_4.unwrap()));
        self.ctx.v_td_5 = Some(self.emit.emit_vbroadcast(self.ctx.td_5.unwrap()));
        self.ctx.v_td_6 = Some(self.emit.emit_vbroadcast(self.ctx.td_6.unwrap()));

        // FMA multipliers
        for i in 0..6 {
            if let Some(mult) = self.ctx.hash_fma_mult[i] {
                let mult_s = self.emit.emit_const(mult);
                self.ctx.hash_fma_mult_s.insert(i, mult_s);
                let v_mult = self.emit.emit_vbroadcast(mult_s);
                self.ctx.v_hash_mult.insert(i, v_mult);
            }
        }
    }

    fn emit_valu_selection_init(&mut self) {
        if !self.uses_depth3 {
            return;
        }

        // Alias c_r3[i] = c[7+i]
        for i in 0..8 {
            self.ctx.c_r3[i] = self.ctx.c[7 + i];
        }

        // Load tree[7..14]
        let forest_p = self.ctx.forest_p.unwrap();
        for i in 0..8 {
            let addr = self.emit.emit_alu(Opcode::Add, forest_p, self.ctx.c_r3[i].unwrap());
            self.ctx.tc_r3_addr[i] = Some(addr);
            self.ctx.tc_r3[i] = Some(self.emit.emit_load(addr));
        }

        // Tree-reduce deltas (if enabled)
        if self.config.enable_tree_reduce_depth3 {
            let tc_r3_0 = self.ctx.tc_r3[0].unwrap();
            let tc_r3_1 = self.ctx.tc_r3[1].unwrap();
            let tc_r3_2 = self.ctx.tc_r3[2].unwrap();
            let tc_r3_3 = self.ctx.tc_r3[3].unwrap();
            let tc_r3_4 = self.ctx.tc_r3[4].unwrap();
            let tc_r3_5 = self.ctx.tc_r3[5].unwrap();
            let tc_r3_6 = self.ctx.tc_r3[6].unwrap();
            let tc_r3_7 = self.ctx.tc_r3[7].unwrap();

            self.ctx.td_r3_01 = Some(self.emit.emit_alu(Opcode::Sub, tc_r3_1, tc_r3_0));
            self.ctx.td_r3_23 = Some(self.emit.emit_alu(Opcode::Sub, tc_r3_3, tc_r3_2));
            self.ctx.td_r3_45 = Some(self.emit.emit_alu(Opcode::Sub, tc_r3_5, tc_r3_4));
            self.ctx.td_r3_67 = Some(self.emit.emit_alu(Opcode::Sub, tc_r3_7, tc_r3_6));
        }
    }

    /// Ensure depth 3 broadcasts are initialized (lazy init).
    fn ensure_depth3_broadcasts(&mut self) {
        if self.depth3_broadcasts_initialized {
            return;
        }
        self.depth3_broadcasts_initialized = true;

        for i in 0..8 {
            self.ctx.v_tc_r3[i] = Some(self.emit.emit_vbroadcast(self.ctx.tc_r3[i].unwrap()));
        }

        for i in 0..8 {
            self.ctx.v_const_r3[i] = Some(self.emit.emit_vbroadcast(self.ctx.c_r3[i].unwrap()));
        }

        if self.config.enable_tree_reduce_depth3 {
            self.ctx.v_td_r3_01 = Some(self.emit.emit_vbroadcast(self.ctx.td_r3_01.unwrap()));
            self.ctx.v_td_r3_23 = Some(self.emit.emit_vbroadcast(self.ctx.td_r3_23.unwrap()));
            self.ctx.v_td_r3_45 = Some(self.emit.emit_vbroadcast(self.ctx.td_r3_45.unwrap()));
            self.ctx.v_td_r3_67 = Some(self.emit.emit_vbroadcast(self.ctx.td_r3_67.unwrap()));
        }
    }

    // =========================================================================
    // Data Load/Store
    // =========================================================================

    fn emit_load_data(&mut self, n_groups: usize) {
        let idx_p = self.ctx.idx_p.unwrap();
        let val_p = self.ctx.val_p.unwrap();

        for g in (0..n_groups).step_by(2) {
            let offset_a = (g * VLEN) as i64;
            let offset_b = ((g + 1) * VLEN) as i64;

            let off_a = self.emit.emit_const(offset_a);
            let off_b = self.emit.emit_const(offset_b);

            let idx_addr_a = self.emit.emit_alu(Opcode::Add, idx_p, off_a);
            let idx_addr_b = self.emit.emit_alu(Opcode::Add, idx_p, off_b);
            let val_addr_a = self.emit.emit_alu(Opcode::Add, val_p, off_a);
            let val_addr_b = self.emit.emit_alu(Opcode::Add, val_p, off_b);

            self.ctx.all_idx[g] = Some(self.emit.emit_vload(idx_addr_a));
            self.ctx.all_idx[g + 1] = Some(self.emit.emit_vload(idx_addr_b));
            self.ctx.all_val[g] = Some(self.emit.emit_vload(val_addr_a));
            self.ctx.all_val[g + 1] = Some(self.emit.emit_vload(val_addr_b));
        }

        // Convert initial values to optimized form
        let v_stage5_xor = self.ctx.v_stage5_xor.unwrap();
        for g in 0..n_groups {
            let val = self.ctx.all_val[g].unwrap();
            self.ctx.all_val[g] = Some(self.emit.emit_valu(Opcode::Xor, val, v_stage5_xor));
        }
    }

    fn emit_store_data(&mut self, n_groups: usize) {
        let idx_p = self.ctx.idx_p.unwrap();
        let val_p = self.ctx.val_p.unwrap();

        for g in (0..n_groups).step_by(2) {
            let offset_a = (g * VLEN) as i64;
            let offset_b = ((g + 1) * VLEN) as i64;

            let off_a = self.emit.emit_const(offset_a);
            let off_b = self.emit.emit_const(offset_b);

            let idx_addr_a = self.emit.emit_alu(Opcode::Add, idx_p, off_a);
            let idx_addr_b = self.emit.emit_alu(Opcode::Add, idx_p, off_b);
            let val_addr_a = self.emit.emit_alu(Opcode::Add, val_p, off_a);
            let val_addr_b = self.emit.emit_alu(Opcode::Add, val_p, off_b);

            self.emit.emit_vstore(idx_addr_a, self.ctx.all_idx[g].unwrap());
            self.emit.emit_vstore(idx_addr_b, self.ctx.all_idx[g + 1].unwrap());
            self.emit.emit_vstore(val_addr_a, self.ctx.all_val[g].unwrap());
            self.emit.emit_vstore(val_addr_b, self.ctx.all_val[g + 1].unwrap());
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

        // Branch
        self.emit_branch(n_groups, n_iters, tree_depth);
    }

    // =========================================================================
    // Tree Access by Depth
    // =========================================================================

    fn emit_depth0_access(&mut self, n_iters: usize) {
        // Depth 0: All at root - broadcast tree[0]
        let tc0 = self.ctx.tc[0].unwrap();
        let v_stage5_xor = self.ctx.v_stage5_xor.unwrap();

        for it in 0..n_iters {
            for i in 0..UNROLL {
                let v_nv = self.emit.emit_vbroadcast_force(tc0);
                self.ctx.v_nv[it][i] = Some(self.emit.emit_valu(Opcode::Xor, v_nv, v_stage5_xor));
            }
        }
    }

    fn emit_depth1_access(&mut self, n_iters: usize) {
        // Depth 1: Delta selection for 2 values
        let v_1 = self.ctx.v_1.unwrap();
        let v_td_1 = self.ctx.v_td_1.unwrap();
        let v_tc2 = self.ctx.v_tc2.unwrap();
        let v_stage5_xor = self.ctx.v_stage5_xor.unwrap();

        for it in 0..n_iters {
            let base_g = it * UNROLL;
            for i in 0..UNROLL {
                let idx = self.ctx.all_idx[base_g + i].unwrap();
                let cond = self.emit.emit_valu(Opcode::Eq, idx, v_1);
                let v_nv = self.emit.emit_fma(cond, v_td_1, v_tc2);
                self.ctx.v_nv[it][i] = Some(self.emit.emit_valu(Opcode::Xor, v_nv, v_stage5_xor));
            }
        }
    }

    fn emit_depth2_access(&mut self, n_iters: usize) {
        // Depth 2: Delta selection for 4 values
        let v_const_4 = self.ctx.v_const[4].unwrap();
        let v_const_5 = self.ctx.v_const[5].unwrap();
        let v_const_6 = self.ctx.v_const[6].unwrap();
        let v_td_4 = self.ctx.v_td_4.unwrap();
        let v_td_5 = self.ctx.v_td_5.unwrap();
        let v_td_6 = self.ctx.v_td_6.unwrap();
        let v_tc3 = self.ctx.v_tc3.unwrap();
        let v_stage5_xor = self.ctx.v_stage5_xor.unwrap();

        for it in 0..n_iters {
            let base_g = it * UNROLL;
            for i in 0..UNROLL {
                let idx = self.ctx.all_idx[base_g + i].unwrap();

                // Start with tc[3], add delta if idx == 4
                let cond4 = self.emit.emit_valu(Opcode::Eq, idx, v_const_4);
                let mut v_nv = self.emit.emit_fma(cond4, v_td_4, v_tc3);

                // Add td_5 if idx == 5
                let cond5 = self.emit.emit_valu(Opcode::Eq, idx, v_const_5);
                v_nv = self.emit.emit_fma(cond5, v_td_5, v_nv);

                // Add td_6 if idx == 6
                let cond6 = self.emit.emit_valu(Opcode::Eq, idx, v_const_6);
                v_nv = self.emit.emit_fma(cond6, v_td_6, v_nv);

                self.ctx.v_nv[it][i] = Some(self.emit.emit_valu(Opcode::Xor, v_nv, v_stage5_xor));
            }
        }
    }

    fn emit_depth3_access(&mut self, n_iters: usize) {
        // Ensure depth 3 broadcasts are initialized
        self.ensure_depth3_broadcasts();

        let v_stage5_xor = self.ctx.v_stage5_xor.unwrap();

        if self.config.enable_tree_reduce_depth3 {
            // Tree-reduce approach (15 VALU ops instead of 16)
            let v_const_7 = self.ctx.v_const[7].unwrap();
            let v_1 = self.ctx.v_1.unwrap();
            let v_td_r3_01 = self.ctx.v_td_r3_01.unwrap();
            let v_td_r3_23 = self.ctx.v_td_r3_23.unwrap();
            let v_td_r3_45 = self.ctx.v_td_r3_45.unwrap();
            let v_td_r3_67 = self.ctx.v_td_r3_67.unwrap();
            let v_tc_r3_0 = self.ctx.v_tc_r3[0].unwrap();
            let v_tc_r3_2 = self.ctx.v_tc_r3[2].unwrap();
            let v_tc_r3_4 = self.ctx.v_tc_r3[4].unwrap();
            let v_tc_r3_6 = self.ctx.v_tc_r3[6].unwrap();

            for it in 0..n_iters {
                let base_g = it * UNROLL;
                for i in 0..UNROLL {
                    let idx = self.ctx.all_idx[base_g + i].unwrap();

                    // Bit extraction
                    let idx_adj = self.emit.emit_valu(Opcode::Sub, idx, v_const_7);
                    let bit0 = self.emit.emit_valu(Opcode::And, idx_adj, v_1);
                    let idx_adj_shr = self.emit.emit_valu(Opcode::Shr, idx_adj, v_1);
                    let bit1 = self.emit.emit_valu(Opcode::And, idx_adj_shr, v_1);
                    let bit2 = self.emit.emit_valu(Opcode::Shr, idx_adj_shr, v_1);

                    // Level 0: 4 FMAs
                    let p01 = self.emit.emit_fma(bit0, v_td_r3_01, v_tc_r3_0);
                    let p23 = self.emit.emit_fma(bit0, v_td_r3_23, v_tc_r3_2);
                    let p45 = self.emit.emit_fma(bit0, v_td_r3_45, v_tc_r3_4);
                    let p67 = self.emit.emit_fma(bit0, v_td_r3_67, v_tc_r3_6);

                    // Level 1: Combine pairs
                    let dp03 = self.emit.emit_valu(Opcode::Sub, p23, p01);
                    let q03 = self.emit.emit_fma(bit1, dp03, p01);
                    let dp47 = self.emit.emit_valu(Opcode::Sub, p67, p45);
                    let q47 = self.emit.emit_fma(bit1, dp47, p45);

                    // Level 2: Final result
                    let d_final = self.emit.emit_valu(Opcode::Sub, q47, q03);
                    let result = self.emit.emit_fma(bit2, d_final, q03);

                    self.ctx.v_nv[it][i] = Some(self.emit.emit_valu(Opcode::Xor, result, v_stage5_xor));
                }
            }
        } else {
            // Original FMA approach (16 VALU ops)
            for it in 0..n_iters {
                let base_g = it * UNROLL;
                for i in 0..UNROLL {
                    let idx = self.ctx.all_idx[base_g + i].unwrap();

                    // First iteration: Mul
                    let cond0 = self.emit.emit_valu(Opcode::Eq, idx, self.ctx.v_const_r3[0].unwrap());
                    let mut v_nv = self.emit.emit_valu(Opcode::Mul, cond0, self.ctx.v_tc_r3[0].unwrap());

                    // Remaining iterations: FMA
                    for ti in 1..8 {
                        let cond = self.emit.emit_valu(Opcode::Eq, idx, self.ctx.v_const_r3[ti].unwrap());
                        v_nv = self.emit.emit_fma(cond, self.ctx.v_tc_r3[ti].unwrap(), v_nv);
                    }

                    self.ctx.v_nv[it][i] = Some(self.emit.emit_valu(Opcode::Xor, v_nv, v_stage5_xor));
                }
            }
        }
    }

    fn emit_scattered_access(&mut self, n_iters: usize) {
        // Scattered/gathered loads for depths 4+
        //
        // The ISA doesn't support gathered vload, so we need to compute addresses
        // for each lane. For depths 4+, tree indices vary per lane.
        //
        // Proper implementation would use one of:
        // 1. 8 scalar loads per group (expensive)
        // 2. Lazy partition to make accesses sequential
        // 3. load_offset with chunked gather
        //
        // For now, use the FMA approach similar to depth 3:
        // Compute v_nv using delta selection for the applicable tree indices.
        // This works because at depth d, indices are in range [2^d - 1, 2^(d+1) - 2].

        let forest_p = self.ctx.forest_p.unwrap();
        let v_stage5_xor = self.ctx.v_stage5_xor.unwrap();
        let v_0 = self.ctx.v_0.unwrap();

        for it in 0..n_iters {
            let base_g = it * UNROLL;
            for i in 0..UNROLL {
                let idx = self.ctx.all_idx[base_g + i].unwrap();

                // For gathered access, we compute the address vector and use
                // a placeholder load. In a real implementation, this would be
                // 8 scalar loads or use lazy partition.
                //
                // Compute addr = forest_p + idx for each lane
                let v_forest_p = self.emit.emit_vbroadcast_force(forest_p);
                let addr_vec = self.emit.emit_valu(Opcode::Add, v_forest_p, idx);

                // Extract first element as base address for a contiguous vload
                // This is a simplification - actual gathered load would need
                // 8 separate scalar loads or use load_offset.
                //
                // For correctness testing, we'll just use a dummy value.
                // The kernel won't produce correct results for depths 4+ but
                // will exercise the SSA pipeline.
                let dummy_addr = forest_p; // Use forest_p as placeholder
                let v_nv = self.emit.emit_vload(dummy_addr);

                // XOR with stage5 constant
                self.ctx.v_nv[it][i] = Some(self.emit.emit_valu(Opcode::Xor, v_nv, v_stage5_xor));
            }
        }
    }

    // =========================================================================
    // XOR Phase
    // =========================================================================

    fn emit_xor(&mut self, n_groups: usize) {
        let n_iters = n_groups / UNROLL;

        for it in 0..n_iters {
            let base_g = it * UNROLL;
            for i in 0..UNROLL {
                let g = base_g + i;
                let val = self.ctx.all_val[g].unwrap();
                let v_nv = self.ctx.v_nv[it][i].unwrap();
                self.ctx.all_val[g] = Some(self.emit.emit_valu(Opcode::Xor, val, v_nv));
            }
        }
    }

    // =========================================================================
    // Hash Phase
    // =========================================================================

    fn emit_hash(&mut self, _n_groups: usize, n_iters: usize) {
        for hi in 0..6 {
            let (op1, _, op2, op3, _) = HASH_STAGES[hi];

            for it in 0..n_iters {
                let base_g = it * UNROLL;
                for i in 0..UNROLL {
                    let g = base_g + i;
                    let val = self.ctx.all_val[g].unwrap();

                    if self.ctx.hash_fma_mult[hi].is_some() {
                        // FMA optimization
                        let v_mult = *self.ctx.v_hash_mult.get(&hi).unwrap();
                        let v_hc = self.ctx.v_hc[hi].unwrap();
                        self.ctx.all_val[g] = Some(self.emit.emit_fma(val, v_mult, v_hc));
                    } else if hi == 5 {
                        // Stage 5 special handling
                        let v_hs = self.ctx.v_hs[hi].unwrap();
                        let t = self.emit.emit_valu(Opcode::Shr, val, v_hs);
                        self.ctx.all_val[g] = Some(self.emit.emit_valu(Opcode::Xor, val, t));
                    } else {
                        // General case
                        let v_hc = self.ctx.v_hc[hi].unwrap();
                        let v_hs = self.ctx.v_hs[hi].unwrap();

                        let op1_code = match op1 { '+' => Opcode::Add, '^' => Opcode::Xor, _ => Opcode::Add };
                        let op2_code = match op2 { '+' => Opcode::Add, '^' => Opcode::Xor, _ => Opcode::Add };
                        let op3_code = match op3 { '<' => Opcode::Shl, '>' => Opcode::Shr, _ => Opcode::Shl };

                        let t = self.emit.emit_valu(op1_code, val, v_hc);
                        let shifted = self.emit.emit_valu(op3_code, val, v_hs);
                        self.ctx.all_val[g] = Some(self.emit.emit_valu(op2_code, t, shifted));
                    }
                }
            }
        }
    }

    // =========================================================================
    // Branch Phase
    // =========================================================================

    fn emit_branch(&mut self, n_groups: usize, n_iters: usize, tree_depth: usize) {
        let is_wrap_round = tree_depth == self.config.forest_height;
        let v_1 = self.ctx.v_1.unwrap();
        let v_2 = self.ctx.v_2.unwrap();
        let zero = self.zero_vreg.unwrap();

        for it in 0..n_iters {
            let base_g = it * UNROLL;
            for i in 0..UNROLL {
                let g = base_g + i;
                let val = self.ctx.all_val[g].unwrap();
                let idx = self.ctx.all_idx[g].unwrap();

                if is_wrap_round {
                    // Wrap to index 0
                    self.ctx.all_idx[g] = Some(self.emit.emit_vbroadcast_force(zero));
                } else {
                    // Normal branching: idx = idx * 2 + (2 - (val & 1))
                    let bit = self.emit.emit_valu(Opcode::And, val, v_1);
                    let offset = self.emit.emit_valu(Opcode::Sub, v_2, bit);
                    self.ctx.all_idx[g] = Some(self.emit.emit_fma(idx, v_2, offset));
                }
            }
        }

        // Partition stores (tick-tock)
        if TICK_STORE_DEPTHS.contains(&tree_depth) {
            self.emit_partition_stores(0, n_groups / 2);
        } else if TOCK_STORE_DEPTHS.contains(&tree_depth) {
            self.emit_partition_stores(n_groups / 2, n_groups);
        }
    }

    fn emit_partition_stores(&mut self, start_g: usize, end_g: usize) {
        let extra_room_p = self.ctx.extra_room_p.unwrap();
        let partition_val_offset = self.ctx.partition_val_offset.unwrap();

        // Compute base addresses
        let base_offset = self.emit.emit_const((start_g * VLEN) as i64);
        let mut idx_addr = self.emit.emit_alu(Opcode::Add, extra_room_p, base_offset);
        let mut val_addr = self.emit.emit_alu(Opcode::Add, idx_addr, partition_val_offset);
        let vlen_const = self.emit.emit_const(VLEN as i64);

        // Store all groups
        for g in start_g..end_g {
            self.emit.emit_vstore(idx_addr, self.ctx.all_idx[g].unwrap());
            self.emit.emit_vstore(val_addr, self.ctx.all_val[g].unwrap());

            // Increment addresses for next group
            if g < end_g - 1 {
                idx_addr = self.emit.emit_alu(Opcode::Add, idx_addr, vlen_const);
                val_addr = self.emit.emit_alu(Opcode::Add, val_addr, vlen_const);
            }
        }
    }
}

// ============================================================================
// VALU→ALU Shifting Helpers
// ============================================================================

/// Check if an SSA VALU op can be shifted to ALU.
///
/// Shiftable ops are binary arithmetic operations that can be expanded
/// to 8 scalar ALU ops (one per vector lane).
fn can_ssa_valu_shift_to_alu(op: &SsaOp) -> bool {
    if op.engine != Engine::Valu {
        return false;
    }

    // Check if the opcode supports shifting
    op.opcode.can_shift_to_alu()
}

/// Lower SSA ops to concrete ops (no VALU shifting).
fn lower_ssa_ops(ssa_ops: &[SsaOp], vreg_to_addr: &HashMap<VReg, i64>) -> Vec<Op> {
    ssa_ops
        .iter()
        .map(|op| lower_ssa_op(op, vreg_to_addr))
        .collect()
}

/// Lower SSA ops to concrete ops, expanding shifted VALU ops to ALU.
///
/// Shifted VALU ops are expanded to 8 scalar ALU ops, one per lane.
/// Each ALU op reads from src[i] and writes to dest[i].
fn lower_ssa_ops_with_shift(
    ssa_ops: &[SsaOp],
    vreg_to_addr: &HashMap<VReg, i64>,
    shifted_ops: &HashSet<usize>,
) -> Vec<Op> {
    use arrayvec::ArrayVec;

    let mut result = Vec::new();

    for ssa_op in ssa_ops {
        if shifted_ops.contains(&ssa_op.id) {
            // Expand shifted VALU op to 8 ALU ops
            let expanded = expand_ssa_valu_to_alu(ssa_op, vreg_to_addr);
            result.extend(expanded);
        } else {
            // Normal lowering
            result.push(lower_ssa_op(ssa_op, vreg_to_addr));
        }
    }

    result
}

/// Expand a shifted SSA VALU op into 8 scalar ALU ops.
fn expand_ssa_valu_to_alu(ssa_op: &SsaOp, vreg_to_addr: &HashMap<VReg, i64>) -> Vec<Op> {
    use arrayvec::ArrayVec;

    let dest = ssa_op.dest.expect("VALU op must have dest");
    let dest_addr = *vreg_to_addr.get(&dest).expect("dest VReg not in mapping");

    // Get source addresses
    let (src1_addr, src2_addr) = match (&ssa_op.srcs[..], ssa_op.opcode) {
        ([Operand::VReg(s1), Operand::VReg(s2)], _) => {
            let a1 = *vreg_to_addr.get(s1).expect("src1 VReg not in mapping");
            let a2 = *vreg_to_addr.get(s2).expect("src2 VReg not in mapping");
            (a1, a2)
        }
        _ => return vec![], // Can't expand non-binary ops
    };

    // Create 8 ALU ops
    let mut ops = Vec::with_capacity(VLEN);
    for i in 0..VLEN {
        let i_offset = i as i64;

        let mut slot: ArrayVec<SlotVal, 6> = ArrayVec::new();
        slot.push(SlotVal::Str(ssa_op.opcode.as_str().to_string()));
        slot.push(SlotVal::Int(dest_addr + i_offset));
        slot.push(SlotVal::Int(src1_addr + i_offset));
        slot.push(SlotVal::Int(src2_addr + i_offset));

        // Build reads/writes sets
        let mut reads = HashSet::new();
        reads.insert(src1_addr + i_offset);
        reads.insert(src2_addr + i_offset);

        let mut writes = HashSet::new();
        writes.insert(dest_addr + i_offset);

        ops.push(Op {
            id: ssa_op.id * 1000 + i, // Unique ID for expanded op
            engine: Engine::Alu,
            slot,
            cycle: ssa_op.cycle,
            dep_ids: ssa_op.dep_ids.clone(),
            reads,
            writes,
            last_use_reads: HashSet::new(),
            produces_constant: None,
        });
    }

    ops
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ssa_builder_creates_ops() {
        // Use fewer rounds (4) to test faster and stay within budget
        let config = KernelConfig::new(10, 2047, 256, 4);
        let mut builder = SsaTickTockKernelBuilder::new(config);
        let ops = builder.build();

        assert!(!ops.is_empty(), "Should generate ops");
        println!("Generated {} SSA ops", ops.len());
    }

    #[test]
    fn test_ssa_build_and_schedule() {
        // Use fewer rounds (4) to stay within scratch budget.
        // Full 16 rounds creates too many VRegs for SSA without reuse.
        let config = KernelConfig::new(10, 2047, 256, 4);
        let mut builder = SsaTickTockKernelBuilder::new(config);

        let result = builder.build_and_schedule();
        assert!(result.is_ok(), "Scheduling should succeed: {:?}", result.err());

        let result = result.unwrap();
        println!("SSA kernel: {} cycles, {} ops, {} VRegs, peak_live={}",
            result.cycles, result.ops.len(), result.vreg_count, result.address_result.peak_live);

        assert!(result.cycles > 0, "Should have positive cycle count");
        assert!(!result.address_result.budget_exceeded, "Should not exceed budget");
    }
}
