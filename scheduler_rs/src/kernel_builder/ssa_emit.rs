//! SSA emission methods for the TickTock kernel builder.
//!
//! This module contains all the emit_* methods that generate SSA ops
//! for different kernel phases: initialization, data load/store, tree access,
//! XOR, hash, and branch.
//!
//! NOTE: This module uses `emit_const()` instead of `emit_const_choice()` because
//! the per-section scratch budget model doesn't correctly account for ConstChoice
//! decisions yet. The CP-SAT path uses a filtering workaround to remove the AddImm
//! alternatives before scheduling. See tick_tock_ssa.rs for details.

use crate::types::{EmissionPhase, Opcode, VarType, VLEN};

use super::config::{HASH_STAGES, STAGE5_XOR_CONST};
use super::tick_tock_ssa::{SsaTickTockKernelBuilder, TICK_STORE_DEPTHS, TOCK_STORE_DEPTHS, UNROLL};

#[allow(deprecated)] // emit_const used with filtering workaround - see module docs
impl SsaTickTockKernelBuilder {
    /// Internal build logic.
    pub(super) fn build_internal(&mut self) {
        let n_groups = self.config.n_groups();
        let n_iters = self.config.n_iters();

        // Emit initialization (phase already set to Init by default)
        self.emit.set_phase(EmissionPhase::Init);
        self.emit_common_init();

        self.emit.set_phase(EmissionPhase::ValuSelectionInit);
        self.emit_valu_selection_init();

        // Load data
        self.emit.set_phase(EmissionPhase::LoadData);
        self.emit.clear_round_context();
        self.emit_load_data(n_groups);

        // Main loop
        let tree_height = self.config.tree_height();
        let mut tree_depth = 0;
        for rnd in 0..self.config.rounds {
            let is_last_round = rnd == self.config.rounds - 1;
            self.emit.set_round_context(rnd, tree_depth);
            self.emit_round(rnd, tree_depth, n_groups, n_iters, is_last_round);

            // Emit barrier between rounds for greedy scheduler
            // (barriers are elided before CP-SAT and final output)
            if !is_last_round {
                self.emit.emit_barrier();
            }

            // Update tree depth with wrap
            tree_depth += 1;
            if tree_depth >= tree_height {
                tree_depth = 0;
            }
        }

        // Convert values back from optimized form (finalization)
        self.emit.clear_round_context();
        self.emit.set_phase(EmissionPhase::StoreData);
        let v_stage5_xor = self.ctx.v_stage5_xor.unwrap();
        for g in 0..n_groups {
            self.emit.set_next_var_type(VarType::BatchData);
            let val = self.ctx.all_val[g].unwrap();
            self.ctx.all_val[g] = Some(self.emit.emit_valu(Opcode::Xor, val, v_stage5_xor));
        }

        // Store data
        self.emit_store_data(n_groups);
    }

    // =========================================================================
    // Initialization
    // =========================================================================

    pub(super) fn emit_common_init(&mut self) {
        // Emit constant 0 - this VReg will be fixed at address 0.
        // IMPORTANT: Scratch memory is ALWAYS initialized to 0 at startup, so address 0
        // already contains 0 and never needs to be loaded. We emit a zero constant op
        // Reserve address 0 for zero - scratch[0] is always 0 by hardware initialization.
        // No op is emitted - we just reserve the VReg and assign it to address 0.
        let zero = self.emit.reserve_zero_vreg();
        self.zero_vreg = Some(zero);
        self.ctx.c[0] = Some(zero);

        // Bootstrap constants 1, 2, 4 (use ConstChoice for CP-SAT optimization)
        self.emit.set_next_var_type(VarType::Constant);
        self.ctx.c[1] = Some(self.emit.emit_const_choice(1));
        self.emit.set_next_var_type(VarType::Constant);
        self.ctx.c[2] = Some(self.emit.emit_const_choice(2));
        self.emit.set_next_var_type(VarType::Constant);
        self.ctx.c[4] = Some(self.emit.emit_const_choice(4));

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
        // TODO: c[15] is unused - restructure constant generation to only emit what's needed
        // self.ctx.c[15] = Some(self.emit.emit_alu(Opcode::Add, c8, c7));

        // Load pointers from memory (addresses 4, 5, 6, 7)
        // Note: n_nodes_s not needed in SSA version (was only for v_nn which is unused)
        self.emit.set_next_var_type(VarType::Pointer);
        self.ctx.forest_p = Some(self.emit.emit_load_imm(4));
        self.emit.set_next_var_type(VarType::Pointer);
        self.ctx.idx_p = Some(self.emit.emit_load_imm(5));
        self.emit.set_next_var_type(VarType::Pointer);
        self.ctx.val_p = Some(self.emit.emit_load_imm(6));
        self.emit.set_next_var_type(VarType::Pointer);
        self.ctx.extra_room_p = Some(self.emit.emit_load_imm(7));

        // Partition buffer offsets
        // Note: partition_idx_offset not needed in SSA version (offset is 0, implicit)
        self.emit.set_next_var_type(VarType::Constant);
        self.ctx.partition_val_offset = Some(self.emit.emit_const_choice(self.config.batch_size as i64));

        // Load hash constants and create broadcasts.
        // Only emit constants/broadcasts that are actually used:
        // - FMA stages (0, 2, 4): use v_hc, but not v_hs
        // - Stage 5: uses v_hs[5], but not v_hc[5] (special XOR path)
        // - General stages (1, 3): use both v_hc and v_hs
        for i in 0..6 {
            // v_hc needed for stages 0, 1, 2, 3, 4 (not 5)
            if i != 5 {
                self.emit.set_next_var_type(VarType::HashState);
                self.ctx.hc[i] = Some(self.emit.emit_const(HASH_STAGES[i].1));
                self.emit.set_next_var_type(VarType::HashState);
                self.ctx.v_hc[i] = Some(self.emit.emit_vbroadcast(self.ctx.hc[i].unwrap()));
            }
            // v_hs needed for stages 1, 3, 5 (not 0, 2, 4 which use FMA)
            if i == 1 || i == 3 || i == 5 {
                self.emit.set_next_var_type(VarType::HashState);
                self.ctx.hs[i] = Some(self.emit.emit_const(HASH_STAGES[i].4));
                self.emit.set_next_var_type(VarType::HashState);
                self.ctx.v_hs[i] = Some(self.emit.emit_vbroadcast(self.ctx.hs[i].unwrap()));
            }
        }

        // Broadcast vector constants
        self.emit.set_next_var_type(VarType::Constant);
        self.ctx.v_1 = Some(self.emit.emit_vbroadcast(self.ctx.c[1].unwrap()));
        self.emit.set_next_var_type(VarType::Constant);
        self.ctx.v_2 = Some(self.emit.emit_vbroadcast(self.ctx.c[2].unwrap()));

        // Only broadcast v_const[4..8], as [0..3] are unused
        for i in 4..8 {
            self.emit.set_next_var_type(VarType::Constant);
            self.ctx.v_const[i] = Some(self.emit.emit_vbroadcast(self.ctx.c[i].unwrap()));
        }

        // Stage 5 XOR constant
        self.emit.set_next_var_type(VarType::HashState);
        self.ctx.stage5_xor_s = Some(self.emit.emit_const(STAGE5_XOR_CONST));
        self.emit.set_next_var_type(VarType::HashState);
        self.ctx.v_stage5_xor = Some(self.emit.emit_vbroadcast(self.ctx.stage5_xor_s.unwrap()));

        // Load tree cache
        let forest_p = self.ctx.forest_p.unwrap();
        for i in 0..7 {
            self.emit.set_next_var_type(VarType::Address);
            let addr = self.emit.emit_alu(Opcode::Add, forest_p, self.ctx.c[i].unwrap());
            self.ctx.tc_addr[i] = Some(addr);
            self.emit.set_next_var_type(VarType::TreeCache);
            self.ctx.tc[i] = Some(self.emit.emit_load(addr));
        }

        // Compute deltas
        let tc1 = self.ctx.tc[1].unwrap();
        let tc2 = self.ctx.tc[2].unwrap();
        let tc3 = self.ctx.tc[3].unwrap();
        let tc4 = self.ctx.tc[4].unwrap();
        let tc5 = self.ctx.tc[5].unwrap();
        let tc6 = self.ctx.tc[6].unwrap();

        self.emit.set_next_var_type(VarType::TreeCache);
        self.ctx.td_1 = Some(self.emit.emit_alu(Opcode::Sub, tc1, tc2));
        self.emit.set_next_var_type(VarType::TreeCache);
        self.ctx.td_4 = Some(self.emit.emit_alu(Opcode::Sub, tc4, tc3));
        self.emit.set_next_var_type(VarType::TreeCache);
        self.ctx.td_5 = Some(self.emit.emit_alu(Opcode::Sub, tc5, tc3));
        self.emit.set_next_var_type(VarType::TreeCache);
        self.ctx.td_6 = Some(self.emit.emit_alu(Opcode::Sub, tc6, tc3));

        self.emit.set_next_var_type(VarType::TreeCache);
        self.ctx.v_tc2 = Some(self.emit.emit_vbroadcast(tc2));
        self.emit.set_next_var_type(VarType::TreeCache);
        self.ctx.v_td_1 = Some(self.emit.emit_vbroadcast(self.ctx.td_1.unwrap()));
        // v_tc1 is no longer computed upfront - SelectChoice will compute it when needed
        self.emit.set_next_var_type(VarType::TreeCache);
        self.ctx.v_tc3 = Some(self.emit.emit_vbroadcast(tc3));
        self.emit.set_next_var_type(VarType::TreeCache);
        self.ctx.v_td_4 = Some(self.emit.emit_vbroadcast(self.ctx.td_4.unwrap()));
        // v_tc4 is no longer computed upfront - SelectChoice will compute it when needed
        self.emit.set_next_var_type(VarType::TreeCache);
        self.ctx.v_td_5 = Some(self.emit.emit_vbroadcast(self.ctx.td_5.unwrap()));
        // Compute v_tc5 = v_tc3 + v_td_5 for SelectChoice (true_val)
        self.emit.set_next_var_type(VarType::TreeCache);
        let v_tc5 = self.emit.emit_valu(Opcode::Add, self.ctx.v_tc3.unwrap(), self.ctx.v_td_5.unwrap());
        self.ctx.v_tc5 = Some(v_tc5);
        self.emit.set_next_var_type(VarType::TreeCache);
        self.ctx.v_td_6 = Some(self.emit.emit_vbroadcast(self.ctx.td_6.unwrap()));
        // Compute v_tc6 = v_tc3 + v_td_6 for SelectChoice (true_val)
        self.emit.set_next_var_type(VarType::TreeCache);
        let v_tc6 = self.emit.emit_valu(Opcode::Add, self.ctx.v_tc3.unwrap(), self.ctx.v_td_6.unwrap());
        self.ctx.v_tc6 = Some(v_tc6);

        // FMA multipliers
        for i in 0..6 {
            if let Some(mult) = self.ctx.hash_fma_mult[i] {
                self.emit.set_next_var_type(VarType::HashState);
                let mult_s = self.emit.emit_const(mult);
                self.ctx.hash_fma_mult_s.insert(i, mult_s);
                self.emit.set_next_var_type(VarType::HashState);
                let v_mult = self.emit.emit_vbroadcast(mult_s);
                self.ctx.v_hash_mult.insert(i, v_mult);
            }
        }
    }

    pub(super) fn emit_valu_selection_init(&mut self) {
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
            let addr = self
                .emit
                .emit_alu(Opcode::Add, forest_p, self.ctx.c_r3[i].unwrap());
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
    pub(super) fn ensure_depth3_broadcasts(&mut self) {
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

    pub(super) fn emit_load_data(&mut self, n_groups: usize) {
        let idx_p = self.ctx.idx_p.unwrap();
        let val_p = self.ctx.val_p.unwrap();

        for g in (0..n_groups).step_by(2) {
            let offset_a = (g * VLEN) as i64;
            let offset_b = ((g + 1) * VLEN) as i64;

            self.emit.set_next_var_type(VarType::Constant);
            let off_a = self.emit.emit_const(offset_a);
            self.emit.set_next_var_type(VarType::Constant);
            let off_b = self.emit.emit_const(offset_b);

            self.emit.set_next_var_type(VarType::Address);
            let idx_addr_a = self.emit.emit_alu(Opcode::Add, idx_p, off_a);
            self.emit.set_next_var_type(VarType::Address);
            let idx_addr_b = self.emit.emit_alu(Opcode::Add, idx_p, off_b);
            self.emit.set_next_var_type(VarType::Address);
            let val_addr_a = self.emit.emit_alu(Opcode::Add, val_p, off_a);
            self.emit.set_next_var_type(VarType::Address);
            let val_addr_b = self.emit.emit_alu(Opcode::Add, val_p, off_b);

            self.emit.set_next_var_type(VarType::BatchData);
            self.ctx.all_idx[g] = Some(self.emit.emit_vload(idx_addr_a));
            self.emit.set_next_var_type(VarType::BatchData);
            self.ctx.all_idx[g + 1] = Some(self.emit.emit_vload(idx_addr_b));
            self.emit.set_next_var_type(VarType::BatchData);
            self.ctx.all_val[g] = Some(self.emit.emit_vload(val_addr_a));
            self.emit.set_next_var_type(VarType::BatchData);
            self.ctx.all_val[g + 1] = Some(self.emit.emit_vload(val_addr_b));
        }

        // Convert initial values to optimized form
        let v_stage5_xor = self.ctx.v_stage5_xor.unwrap();
        for g in 0..n_groups {
            self.emit.set_next_var_type(VarType::BatchData);
            let val = self.ctx.all_val[g].unwrap();
            self.ctx.all_val[g] = Some(self.emit.emit_valu(Opcode::Xor, val, v_stage5_xor));
        }
    }

    pub(super) fn emit_store_data(&mut self, n_groups: usize) {
        let idx_p = self.ctx.idx_p.unwrap();
        let val_p = self.ctx.val_p.unwrap();

        for g in (0..n_groups).step_by(2) {
            let offset_a = (g * VLEN) as i64;
            let offset_b = ((g + 1) * VLEN) as i64;

            self.emit.set_next_var_type(VarType::Constant);
            let off_a = self.emit.emit_const(offset_a);
            self.emit.set_next_var_type(VarType::Constant);
            let off_b = self.emit.emit_const(offset_b);

            self.emit.set_next_var_type(VarType::Address);
            let idx_addr_a = self.emit.emit_alu(Opcode::Add, idx_p, off_a);
            self.emit.set_next_var_type(VarType::Address);
            let idx_addr_b = self.emit.emit_alu(Opcode::Add, idx_p, off_b);
            self.emit.set_next_var_type(VarType::Address);
            let val_addr_a = self.emit.emit_alu(Opcode::Add, val_p, off_a);
            self.emit.set_next_var_type(VarType::Address);
            let val_addr_b = self.emit.emit_alu(Opcode::Add, val_p, off_b);

            self.emit
                .emit_vstore(idx_addr_a, self.ctx.all_idx[g].unwrap());
            self.emit
                .emit_vstore(idx_addr_b, self.ctx.all_idx[g + 1].unwrap());
            self.emit
                .emit_vstore(val_addr_a, self.ctx.all_val[g].unwrap());
            self.emit
                .emit_vstore(val_addr_b, self.ctx.all_val[g + 1].unwrap());
        }
    }

    // =========================================================================
    // Round Emission
    // =========================================================================

    pub(super) fn emit_round(
        &mut self,
        _rnd: usize,
        tree_depth: usize,
        n_groups: usize,
        n_iters: usize,
        is_last_round: bool,
    ) {
        // Tree access based on depth
        self.emit.set_phase(EmissionPhase::TreeAccess);
        match tree_depth {
            0 => self.emit_depth0_access(n_iters),
            1 => self.emit_depth1_access(n_iters),
            2 => self.emit_depth2_access(n_iters),
            3 => self.emit_depth3_access(n_iters),
            _ => self.emit_scattered_access(n_iters),
        }

        // XOR
        self.emit.set_phase(EmissionPhase::Xor);
        self.emit_xor(n_groups);

        // Hash
        self.emit.set_phase(EmissionPhase::Hash);
        self.emit_hash(n_groups, n_iters);

        // Branch
        self.emit.set_phase(EmissionPhase::Branch);
        self.emit_branch(n_groups, n_iters, tree_depth, is_last_round);
    }

    // =========================================================================
    // Selection Primitives (2-way, 4-way, 8-way)
    // =========================================================================
    //
    // These build tree selection from SelectChoice pseudo-ops (vselect or FMA per op).
    // 2-way: one select. 4-way: linear chain of 3 selects or bit-tree (2 parallel + 1).
    // 8-way: bit-tree of 7 selects (4 + 2 + 1 levels).

    /// 2-way selection: cond ? true_val : false_val (one SelectChoice).
    fn select_2way(&mut self, cond: crate::types::VReg, true_val: crate::types::VReg, false_val: crate::types::VReg) -> crate::types::VReg {
        self.emit.emit_select_choice(cond, true_val, false_val)
    }

    /// 2-way selection with precomputed delta: cond ? (base + delta) : base (one SelectChoice).
    fn select_2way_with_delta(&mut self, cond: crate::types::VReg, delta: crate::types::VReg, base: crate::types::VReg) -> crate::types::VReg {
        self.emit.emit_select_choice_with_delta(cond, delta, base)
    }

    /// 4-way selection via linear chain: start with v0, then select v1 if idx==base_idx+1, v2 if base_idx+2, v3 if base_idx+3.
    fn select_4way_linear(
        &mut self,
        idx: crate::types::VReg,
        base_idx: usize,
        v0: crate::types::VReg,
        v1: crate::types::VReg,
        v2: crate::types::VReg,
        v3: crate::types::VReg,
    ) -> crate::types::VReg {
        let c1 = base_idx + 1;
        let c2 = base_idx + 2;
        let c3 = base_idx + 3;
        self.emit.set_next_var_type(VarType::Condition);
        let cond1 = self.emit.emit_valu(Opcode::Eq, idx, self.ctx.v_const[c1].unwrap());
        self.emit.set_next_var_type(VarType::Temporary);
        let mut result = self.select_2way(cond1, v1, v0);
        self.emit.set_next_var_type(VarType::Condition);
        let cond2 = self.emit.emit_valu(Opcode::Eq, idx, self.ctx.v_const[c2].unwrap());
        self.emit.set_next_var_type(VarType::Temporary);
        result = self.select_2way(cond2, v2, result);
        self.emit.set_next_var_type(VarType::Condition);
        let cond3 = self.emit.emit_valu(Opcode::Eq, idx, self.ctx.v_const[c3].unwrap());
        self.emit.set_next_var_type(VarType::Temporary);
        result = self.select_2way(cond3, v3, result);
        result
    }

    /// 4-way selection via linear chain with first pair as (base, delta) to avoid materializing base+delta.
    fn select_4way_linear_first_delta(
        &mut self,
        idx: crate::types::VReg,
        base_idx: usize,
        base: crate::types::VReg,
        delta: crate::types::VReg,
        v2: crate::types::VReg,
        v3: crate::types::VReg,
    ) -> crate::types::VReg {
        let c1 = base_idx + 1;
        let c2 = base_idx + 2;
        let c3 = base_idx + 3;
        self.emit.set_next_var_type(VarType::Condition);
        let cond1 = self.emit.emit_valu(Opcode::Eq, idx, self.ctx.v_const[c1].unwrap());
        self.emit.set_next_var_type(VarType::Temporary);
        let mut result = self.select_2way_with_delta(cond1, delta, base);
        self.emit.set_next_var_type(VarType::Condition);
        let cond2 = self.emit.emit_valu(Opcode::Eq, idx, self.ctx.v_const[c2].unwrap());
        self.emit.set_next_var_type(VarType::Temporary);
        result = self.select_2way(cond2, v2, result);
        self.emit.set_next_var_type(VarType::Condition);
        let cond3 = self.emit.emit_valu(Opcode::Eq, idx, self.ctx.v_const[c3].unwrap());
        self.emit.set_next_var_type(VarType::Temporary);
        result = self.select_2way(cond3, v3, result);
        result
    }

    /// 4-way selection via bit-tree: idx_adj = idx - base_idx (0..4), bit0 = low bit, bit1 = high bit.
    /// Level 0: two parallel 2-way (v0,v1) by bit0 and (v2,v3) by bit0. Level 1: one 2-way by bit1.
    /// Requires v_const[base_idx] for idx_adj.
    fn select_4way_bit_tree(
        &mut self,
        idx: crate::types::VReg,
        base_idx: usize,
        v0: crate::types::VReg,
        v1: crate::types::VReg,
        v2: crate::types::VReg,
        v3: crate::types::VReg,
    ) -> crate::types::VReg {
        let v_const_base = self.ctx.v_const[base_idx].unwrap();
        let v_1 = self.ctx.v_1.unwrap();
        self.emit.set_next_var_type(VarType::Temporary);
        let idx_adj = self.emit.emit_valu(Opcode::Sub, idx, v_const_base);
        let bit0 = self.emit.emit_valu(Opcode::And, idx_adj, v_1);
        let idx_adj_shr = self.emit.emit_valu(Opcode::Shr, idx_adj, v_1);
        let bit1 = self.emit.emit_valu(Opcode::And, idx_adj_shr, v_1);
        self.emit.set_next_var_type(VarType::Temporary);
        let p01 = self.select_2way(bit0, v1, v0);
        let p23 = self.select_2way(bit0, v3, v2);
        self.select_2way(bit1, p23, p01)
    }

    /// 4-way bit-tree with first pair as (base, delta) to avoid materializing v1.
    fn select_4way_bit_tree_first_delta(
        &mut self,
        idx: crate::types::VReg,
        base_idx: usize,
        base01: crate::types::VReg,
        delta1: crate::types::VReg,
        v2: crate::types::VReg,
        v3: crate::types::VReg,
    ) -> crate::types::VReg {
        let v_const_base = self.ctx.v_const[base_idx].unwrap();
        let v_1 = self.ctx.v_1.unwrap();
        self.emit.set_next_var_type(VarType::Temporary);
        let idx_adj = self.emit.emit_valu(Opcode::Sub, idx, v_const_base);
        let bit0 = self.emit.emit_valu(Opcode::And, idx_adj, v_1);
        let idx_adj_shr = self.emit.emit_valu(Opcode::Shr, idx_adj, v_1);
        let bit1 = self.emit.emit_valu(Opcode::And, idx_adj_shr, v_1);
        self.emit.set_next_var_type(VarType::Temporary);
        let p01 = self.select_2way_with_delta(bit0, delta1, base01);
        let p23 = self.select_2way(bit0, v3, v2);
        self.select_2way(bit1, p23, p01)
    }

    /// 8-way selection via bit-tree: idx_adj = idx - base_idx (0..8), bit0/bit1/bit2.
    /// Level 0: four parallel 2-way by bit0. Level 1: two 2-way by bit1. Level 2: one 2-way by bit2.
    /// Uses SelectChoice only (no direct FMA). Requires v_const[base_idx] and v_1.
    fn select_8way_bit_tree(
        &mut self,
        idx: crate::types::VReg,
        base_idx: usize,
        values: &[crate::types::VReg; 8],
    ) -> crate::types::VReg {
        let v_const_base = self.ctx.v_const[base_idx].unwrap();
        let v_1 = self.ctx.v_1.unwrap();
        self.emit.set_next_var_type(VarType::Temporary);
        let idx_adj = self.emit.emit_valu(Opcode::Sub, idx, v_const_base);
        let bit0 = self.emit.emit_valu(Opcode::And, idx_adj, v_1);
        let idx_adj_shr1 = self.emit.emit_valu(Opcode::Shr, idx_adj, v_1);
        let bit1 = self.emit.emit_valu(Opcode::And, idx_adj_shr1, v_1);
        let bit2 = self.emit.emit_valu(Opcode::Shr, idx_adj_shr1, v_1);
        self.emit.set_next_var_type(VarType::Temporary);
        let p01 = self.select_2way(bit0, values[1], values[0]);
        let p23 = self.select_2way(bit0, values[3], values[2]);
        let p45 = self.select_2way(bit0, values[5], values[4]);
        let p67 = self.select_2way(bit0, values[7], values[6]);
        let q03 = self.select_2way(bit1, p23, p01);
        let q47 = self.select_2way(bit1, p67, p45);
        self.select_2way(bit2, q47, q03)
    }

    // =========================================================================
    // Tree Access by Depth
    // =========================================================================

    pub(super) fn emit_depth0_access(&mut self, n_iters: usize) {
        // Depth 0: All at root - broadcast tree[0]
        let tc0 = self.ctx.tc[0].unwrap();
        let v_stage5_xor = self.ctx.v_stage5_xor.unwrap();

        for it in 0..n_iters {
            for i in 0..UNROLL {
                self.emit.set_iter_group(it, i);
                self.emit.set_next_var_type(VarType::Temporary);
                let v_nv = self.emit.emit_vbroadcast_force(tc0);
                self.emit.set_next_var_type(VarType::RoundState);
                self.ctx.v_nv[it][i] = Some(self.emit.emit_valu(Opcode::Xor, v_nv, v_stage5_xor));
            }
        }
        self.emit.clear_iter_group();
    }

    pub(super) fn emit_depth1_access(&mut self, n_iters: usize) {
        // Depth 1: 2-way selection (indices 1, 2) via select_2way_with_delta.
        let v_1 = self.ctx.v_1.unwrap();
        let v_td_1 = self.ctx.v_td_1.unwrap();
        let v_tc2 = self.ctx.v_tc2.unwrap();
        let v_stage5_xor = self.ctx.v_stage5_xor.unwrap();

        for it in 0..n_iters {
            let base_g = it * UNROLL;
            for i in 0..UNROLL {
                self.emit.set_iter_group(it, base_g + i);
                let idx = self.ctx.all_idx[base_g + i].unwrap();
                self.emit.set_next_var_type(VarType::Condition);
                let cond = self.emit.emit_valu(Opcode::Eq, idx, v_1);
                self.emit.set_next_var_type(VarType::Temporary);
                let v_nv = self.select_2way_with_delta(cond, v_td_1, v_tc2);
                self.emit.set_next_var_type(VarType::RoundState);
                self.ctx.v_nv[it][i] = Some(self.emit.emit_valu(Opcode::Xor, v_nv, v_stage5_xor));
            }
        }
        self.emit.clear_iter_group();
    }

    pub(super) fn emit_depth2_access(&mut self, n_iters: usize) {
        // Depth 2: 4-way selection (indices 3,4,5,6). First pair (tc3,tc4) as (base, delta); rest as full values.
        let v_tc3 = self.ctx.v_tc3.unwrap();
        let v_td_4 = self.ctx.v_td_4.unwrap();
        let v_tc5 = self.ctx.v_tc5.unwrap();
        let v_tc6 = self.ctx.v_tc6.unwrap();
        let v_stage5_xor = self.ctx.v_stage5_xor.unwrap();
        const BASE_IDX: usize = 3;

        for it in 0..n_iters {
            let base_g = it * UNROLL;
            for i in 0..UNROLL {
                self.emit.set_iter_group(it, base_g + i);
                let idx = self.ctx.all_idx[base_g + i].unwrap();
                self.emit.set_next_var_type(VarType::Temporary);
                let v_nv = self.select_4way_linear_first_delta(idx, BASE_IDX, v_tc3, v_td_4, v_tc5, v_tc6);
                self.emit.set_next_var_type(VarType::RoundState);
                self.ctx.v_nv[it][i] = Some(self.emit.emit_valu(Opcode::Xor, v_nv, v_stage5_xor));
            }
        }
        self.emit.clear_iter_group();
    }

    pub(super) fn emit_depth3_access(&mut self, n_iters: usize) {
        // Ensure depth 3 broadcasts are initialized
        self.ensure_depth3_broadcasts();

        let v_stage5_xor = self.ctx.v_stage5_xor.unwrap();

        if self.config.enable_tree_reduce_depth3 {
            // Tree-reduce: 3 levels of 2-way by bits (same shape as select_8way_bit_tree), but using direct FMA for lower op count.
            // For a SelectChoice-based 8-way use select_8way_bit_tree(idx, 7, &[v_tc_r3[0], ..., v_tc_r3[7]]).
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

                    self.ctx.v_nv[it][i] =
                        Some(self.emit.emit_valu(Opcode::Xor, result, v_stage5_xor));
                }
            }
        } else {
            // Original FMA approach (16 VALU ops)
            for it in 0..n_iters {
                let base_g = it * UNROLL;
                for i in 0..UNROLL {
                    let idx = self.ctx.all_idx[base_g + i].unwrap();

                    // First iteration: Mul
                    let cond0 = self
                        .emit
                        .emit_valu(Opcode::Eq, idx, self.ctx.v_const_r3[0].unwrap());
                    let mut v_nv = self
                        .emit
                        .emit_valu(Opcode::Mul, cond0, self.ctx.v_tc_r3[0].unwrap());

                    // Remaining iterations: FMA
                    for ti in 1..8 {
                        let cond = self
                            .emit
                            .emit_valu(Opcode::Eq, idx, self.ctx.v_const_r3[ti].unwrap());
                        v_nv = self.emit.emit_fma(cond, self.ctx.v_tc_r3[ti].unwrap(), v_nv);
                    }

                    self.ctx.v_nv[it][i] =
                        Some(self.emit.emit_valu(Opcode::Xor, v_nv, v_stage5_xor));
                }
            }
        }
    }

    pub(super) fn emit_scattered_access(&mut self, n_iters: usize) {
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

        for it in 0..n_iters {
            let base_g = it * UNROLL;
            for i in 0..UNROLL {
                let idx = self.ctx.all_idx[base_g + i].unwrap();

                // For gathered access, we need 8 separate loads (or load_offset).
                // This is a simplification that just uses a contiguous vload from
                // the forest pointer. Won't produce correct results for depths 4+
                // but exercises the SSA pipeline.
                let _ = idx; // Mark idx as used (consumed by conceptual gather)
                let v_nv = self.emit.emit_vload(forest_p);

                // XOR with stage5 constant
                self.ctx.v_nv[it][i] = Some(self.emit.emit_valu(Opcode::Xor, v_nv, v_stage5_xor));
            }
        }
    }

    // =========================================================================
    // XOR Phase
    // =========================================================================

    pub(super) fn emit_xor(&mut self, n_groups: usize) {
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

    pub(super) fn emit_hash(&mut self, _n_groups: usize, n_iters: usize) {
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

    pub(super) fn emit_branch(
        &mut self,
        n_groups: usize,
        n_iters: usize,
        tree_depth: usize,
        is_last_round: bool,
    ) {
        let is_wrap_round = tree_depth == self.config.forest_height;
        let v_1 = self.ctx.v_1.unwrap();
        let v_2 = self.ctx.v_2.unwrap();
        let zero = self.zero_vreg.unwrap();

        // Determine which groups need idx updates.
        // On the last round, ALL groups need updates because emit_store_data
        // stores all groups to the output. The partition store pattern (TICK/TOCK)
        // only affects which groups get stored to the partition buffer mid-computation,
        // but all groups must have their final values computed for the final store.
        let (idx_update_start, idx_update_end) = (0, n_groups);

        for it in 0..n_iters {
            let base_g = it * UNROLL;
            for i in 0..UNROLL {
                let g = base_g + i;

                // Skip idx update if this group won't be consumed
                if g < idx_update_start || g >= idx_update_end {
                    continue;
                }

                let val = self.ctx.all_val[g].unwrap();
                let idx = self.ctx.all_idx[g].unwrap();

                if is_wrap_round {
                    // Wrap to index 0
                    self.ctx.all_idx[g] = Some(self.emit.emit_vbroadcast_force(zero));
                } else {
                    // Normal branching: idx = idx * 2 + (2 - (val & 1))
                    let bit = self.emit.emit_valu(Opcode::And, val, v_1);
                    let offset = self.emit.emit_valu(Opcode::Sub, v_2, bit);
                    let fma_result = self.emit.emit_fma(idx, v_2, offset);
                    if std::env::var("SCHEDULER_DEBUG").is_ok() {
                        if is_last_round {
                            eprintln!("[DEBUG emit_branch] Last round g={} tree_depth={}: FMA result {:?}, old all_idx[g]={:?}",
                                g, tree_depth, fma_result, self.ctx.all_idx[g]);
                        }
                        // Also log FMAs in range 6590-6650 to find the dead stores
                        if fma_result.id >= 6590 && fma_result.id <= 6650 {
                            eprintln!("[DEBUG emit_branch] DEAD STORE CANDIDATE? g={} tree_depth={} is_last_round={}: FMA result {:?}",
                                g, tree_depth, is_last_round, fma_result);
                        }
                    }
                    self.ctx.all_idx[g] = Some(fma_result);
                }
            }
        }

        // Partition stores (tick-tock) - only if enabled
        if self.config.enable_partition_stores {
            if TICK_STORE_DEPTHS.contains(&tree_depth) {
                if std::env::var("SCHEDULER_DEBUG").is_ok() && is_last_round {
                    eprintln!("[DEBUG emit_branch] About to emit partition stores for groups 0..{}, tree_depth={}", n_groups/2, tree_depth);
                    for g in 0..n_groups/2 {
                        eprintln!("  all_idx[{}] = {:?}", g, self.ctx.all_idx[g]);
                    }
                }
                self.emit_partition_stores(0, n_groups / 2);
            } else if TOCK_STORE_DEPTHS.contains(&tree_depth) {
                self.emit_partition_stores(n_groups / 2, n_groups);
            }
        }
    }

    pub(super) fn emit_partition_stores(&mut self, start_g: usize, end_g: usize) {
        let debug = std::env::var("SCHEDULER_DEBUG").is_ok();
        let extra_room_p = self.ctx.extra_room_p.unwrap();
        let partition_val_offset = self.ctx.partition_val_offset.unwrap();

        // Compute base addresses
        let base_offset = self.emit.emit_const((start_g * VLEN) as i64);
        let mut idx_addr = self.emit.emit_alu(Opcode::Add, extra_room_p, base_offset);
        let mut val_addr = self.emit.emit_alu(Opcode::Add, idx_addr, partition_val_offset);
        let vlen_const = self.emit.emit_const(VLEN as i64);

        // Store all groups
        for g in start_g..end_g {
            let idx_vreg = self.ctx.all_idx[g].unwrap();
            let val_vreg = self.ctx.all_val[g].unwrap();
            if debug {
                eprintln!("[DEBUG emit_partition_stores] g={}: vstore(idx_addr={:?}, idx={:?}), vstore(val_addr={:?}, val={:?})",
                    g, idx_addr, idx_vreg, val_addr, val_vreg);
            }
            self.emit
                .emit_vstore(idx_addr, idx_vreg);
            self.emit
                .emit_vstore(val_addr, val_vreg);

            // Increment addresses for next group
            if g < end_g - 1 {
                idx_addr = self.emit.emit_alu(Opcode::Add, idx_addr, vlen_const);
                val_addr = self.emit.emit_alu(Opcode::Add, val_addr, vlen_const);
            }
        }
    }
}
