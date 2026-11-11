//! SSA emission methods for the TickTock kernel builder.
//!
//! This module contains all the emit_* methods that generate SSA ops
//! for different kernel phases: initialization, data load/store, tree access,
//! XOR, hash, and branch.
//!
//! The SSA path uses only emit_const_choice() for constants. The per-section scratch
//! budget model counts one VReg per ConstChoice op; scratch usage is the same whether
//! we later lower to Load or AddImm, so no extra logic is needed there.

use crate::types::{EmissionPhase, Opcode, VarType, VLEN};

use super::config::{
    level_last_index, merged_hash_stage_constants, IndexForm, HASH_STAGES, MERGED_HASH_STAGE,
    STAGE5_XOR_CONST,
};
use super::tick_tock_ssa::{
    SsaTickTockKernelBuilder, TICK_STORE_DEPTHS, TOCK_STORE_DEPTHS, UNROLL,
};

impl SsaTickTockKernelBuilder {
    /// Internal build logic.
    pub(super) fn build_internal(&mut self) {
        let n_groups = self.config.spec.n_groups();
        let n_iters = self.config.spec.n_iters();

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
        let tree_height = self.config.spec.tree_height();
        let mut tree_depth = 0;
        for rnd in 0..self.config.spec.rounds {
            let is_last_round = rnd == self.config.spec.rounds - 1;
            self.emit.set_round_context(rnd, tree_depth);
            self.emit_round(rnd, tree_depth, n_groups, n_iters, is_last_round);

            // Snapshot round-(rounds-2) hash output so StoreData xor can attach keep_live
            if self.config.spec.rounds >= 2 && rnd == self.config.spec.rounds - 2 {
                self.ctx.round_before_last_hash = Some(self.ctx.all_val.clone());
            }

            // Emit barrier between rounds as a hint: we do pre-optimization between
            // barriers (per-section), then full-kernel optimization (LTO-style).
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
            // Keep round-(rounds-2) hash output live until this xor (transitive pipeline dep)
            if let Some(ref rb) = self.ctx.round_before_last_hash {
                if let Some(v) = rb.get(g).and_then(|o| *o) {
                    self.emit.set_next_keep_live(vec![v]);
                }
            }
            let val = self.ctx.all_val[g].unwrap();
            self.ctx.all_val[g] = Some(self.emit.emit_valu(Opcode::Xor, val, v_stage5_xor));
        }

        // Capture StoreData xor results and current idx so vstores use exactly these VRegs
        // (avoids any risk of reading stale or wrong ctx slot during emit_store_data).
        let store_data_vals: Vec<crate::types::VReg> = (0..n_groups)
            .map(|g| self.ctx.all_val[g].unwrap())
            .collect();
        // `tree_depth` is now the depth the indices are at after the last round.
        let store_data_idxs: Vec<crate::types::VReg> = (0..n_groups)
            .map(|g| self.final_index(g, tree_depth))
            .collect();

        // Store data
        self.emit_store_data(n_groups, &store_data_idxs, &store_data_vals);
    }

    // =========================================================================
    // Index Form
    // =========================================================================

    /// True when the kernel tracks the mirrored index `n = L_d - idx` (`IndexForm::Mirrored`).
    pub(super) fn mirrored_index(&self) -> bool {
        self.config.spec.index_form == IndexForm::Mirrored
    }

    /// Vector of a small constant: `ConstChoice` + `vbroadcast`, both deduplicated.
    ///
    /// # Arguments
    /// * `value` - The constant to broadcast.
    ///
    /// # Returns
    /// The vector VReg holding `value` in every lane.
    pub(super) fn v_constant(&mut self, value: i64) -> crate::types::VReg {
        let scalar = if value == 0 {
            self.zero_vreg.unwrap()
        } else {
            self.emit.set_next_var_type(VarType::Constant);
            self.emit.emit_const_choice(value)
        };
        self.emit.set_next_var_type(VarType::Constant);
        self.emit.emit_vbroadcast(scalar)
    }

    /// Vector that cached-level selection compares the tracked index against to test
    /// `idx == tree_index` at `tree_depth`.
    ///
    /// # Arguments
    /// * `tree_depth` - Depth the indices are at.
    /// * `tree_index` - Reference (zero-based) tree index to test for.
    ///
    /// # Returns
    /// Vector of `tree_index` for `IndexForm::ZeroBased`, of `L_d - tree_index` for
    /// `IndexForm::Mirrored` (`idx == k` iff `n == L_d - k`, `R10-mirrored-index-uses`).
    pub(super) fn v_index_compare(
        &mut self,
        tree_depth: usize,
        tree_index: usize,
    ) -> crate::types::VReg {
        let value = if self.mirrored_index() {
            level_last_index(tree_depth) - tree_index as i64
        } else {
            tree_index as i64
        };
        self.v_constant(value)
    }

    /// Emit the position of group `g`'s indices from the left end of their level:
    /// `idx - (2^d - 1)`, in `0..2^d`.
    ///
    /// # Arguments
    /// * `g` - Group.
    /// * `v_level_first` - Vector of `2^d - 1`, the first index of the level.
    ///
    /// # Returns
    /// The vector VReg: `idx - first` for `IndexForm::ZeroBased`, `first - n` for
    /// `IndexForm::Mirrored` (equal because `L_d = 2 * first`, `R10-mirrored-index-uses`).
    pub(super) fn emit_level_position(
        &mut self,
        g: usize,
        v_level_first: crate::types::VReg,
    ) -> crate::types::VReg {
        let idx = self.ctx.all_idx[g].unwrap();
        if self.mirrored_index() {
            self.emit.emit_valu(Opcode::Sub, v_level_first, idx)
        } else {
            self.emit.emit_valu(Opcode::Sub, idx, v_level_first)
        }
    }

    /// The reference (zero-based) index vector of group `g` for the final index store.
    ///
    /// # Arguments
    /// * `g` - Group.
    /// * `tree_depth` - Depth the indices are at after the last round.
    ///
    /// # Returns
    /// `all_idx[g]` for `IndexForm::ZeroBased`. For `IndexForm::Mirrored`, `L_d - n`
    /// (one SUB per group), or a shared vector of zeros at depth 0.
    fn final_index(&mut self, g: usize, tree_depth: usize) -> crate::types::VReg {
        if !self.mirrored_index() {
            return self.ctx.all_idx[g].unwrap();
        }
        match self.ctx.all_idx[g] {
            None => self.v_constant(0),
            Some(n) => {
                let v_last = self.v_constant(level_last_index(tree_depth));
                self.emit.set_next_var_type(VarType::BatchData);
                self.emit.emit_valu(Opcode::Sub, v_last, n)
            }
        }
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
        // c[15] needed for depth 4 tree-reduce (and as base for constants 16-30)
        if self.uses_depth4 {
            self.ctx.c[15] = Some(self.emit.emit_alu(Opcode::Add, c8, c7));
        }

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
        self.ctx.partition_val_offset =
            Some(self.emit.emit_const_choice(self.config.spec.batch_size as i64));

        // Load hash constants and create broadcasts.
        // Only emit constants/broadcasts that are actually used:
        // - FMA stages (0, 2, 4): use v_hc, but not v_hs
        // - Stage 5: uses v_hs[5], but not v_hc[5] (special XOR path)
        // - General stages (1, 3): use both v_hc and v_hs
        // With merged stages 2+3 neither stage's own constant nor the stage-3 shift is used.
        let merge = self.config.spec.merge_hash_stages;
        for i in 0..6 {
            if merge && (i == MERGED_HASH_STAGE || i == MERGED_HASH_STAGE + 1) {
                continue;
            }
            // v_hc needed for stages 0, 1, 2, 3, 4 (not 5)
            if i != 5 {
                self.emit.set_next_var_type(VarType::HashState);
                self.ctx.hc[i] = Some(self.emit.emit_const_choice(HASH_STAGES[i].1));
                self.emit.set_next_var_type(VarType::HashState);
                self.ctx.v_hc[i] = Some(self.emit.emit_vbroadcast(self.ctx.hc[i].unwrap()));
            }
            // v_hs needed for stages 1, 3, 5 (not 0, 2, 4 which use FMA)
            if i == 1 || i == 3 || i == 5 {
                self.emit.set_next_var_type(VarType::HashState);
                self.ctx.hs[i] = Some(self.emit.emit_const_choice(HASH_STAGES[i].4));
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
        self.ctx.stage5_xor_s = Some(self.emit.emit_const_choice(STAGE5_XOR_CONST));
        self.emit.set_next_var_type(VarType::HashState);
        self.ctx.v_stage5_xor = Some(self.emit.emit_vbroadcast(self.ctx.stage5_xor_s.unwrap()));

        // Load tree cache
        let forest_p = self.ctx.forest_p.unwrap();
        for i in 0..7 {
            self.emit.set_next_var_type(VarType::Address);
            let addr = self
                .emit
                .emit_alu(Opcode::Add, forest_p, self.ctx.c[i].unwrap());
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
        let v_tc5 = self.emit.emit_valu(
            Opcode::Add,
            self.ctx.v_tc3.unwrap(),
            self.ctx.v_td_5.unwrap(),
        );
        self.ctx.v_tc5 = Some(v_tc5);
        self.emit.set_next_var_type(VarType::TreeCache);
        self.ctx.v_td_6 = Some(self.emit.emit_vbroadcast(self.ctx.td_6.unwrap()));
        // Compute v_tc6 = v_tc3 + v_td_6 for SelectChoice (true_val)
        self.emit.set_next_var_type(VarType::TreeCache);
        let v_tc6 = self.emit.emit_valu(
            Opcode::Add,
            self.ctx.v_tc3.unwrap(),
            self.ctx.v_td_6.unwrap(),
        );
        self.ctx.v_tc6 = Some(v_tc6);

        // FMA multipliers
        for i in 0..6 {
            if let Some(mult) = self.ctx.hash_fma_mult[i] {
                self.emit.set_next_var_type(VarType::HashState);
                let mult_s = self.emit.emit_const_choice(mult);
                self.ctx.hash_fma_mult_s.insert(i, mult_s);
                self.emit.set_next_var_type(VarType::HashState);
                let v_mult = self.emit.emit_vbroadcast(mult_s);
                self.ctx.v_hash_mult.insert(i, v_mult);
            }
        }

        // Merged stages 2+3: (a * m1 + c1) ^ (a * m2 + c2); m1 is the stage-2 multiplier above.
        if merge {
            let (_m1, c1, m2, c2) = merged_hash_stage_constants(MERGED_HASH_STAGE)
                .expect("hash stages 2 and 3 have the mergeable form");
            let mut v_merged = [None; 3];
            for (slot, value) in v_merged.iter_mut().zip([c1, m2, c2]) {
                self.emit.set_next_var_type(VarType::HashState);
                let scalar = self.emit.emit_const_choice(value);
                self.emit.set_next_var_type(VarType::HashState);
                *slot = Some(self.emit.emit_vbroadcast(scalar));
            }
            self.ctx.v_hash_merged =
                Some((v_merged[0].unwrap(), v_merged[1].unwrap(), v_merged[2].unwrap()));
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

        // Tree-reduce deltas (if enabled for depth 3)
        if self.config.use_tree_reduce_at_depth(3) {
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

        // Depth 4 initialization: constants 15-30, load tree[15..30], compute deltas
        if self.uses_depth4 {
            let c8 = self.ctx.c[8].unwrap();
            let c15 = self.ctx.c[15].unwrap();
            let c1 = self.ctx.c[1].unwrap();
            let c2 = self.ctx.c[2].unwrap();
            let c3 = self.ctx.c[3].unwrap();
            let c4 = self.ctx.c[4].unwrap();
            let c5 = self.ctx.c[5].unwrap();
            let c6 = self.ctx.c[6].unwrap();
            let c7 = self.ctx.c[7].unwrap();

            // c_r4[0] = c[15] (alias for constant 15)
            self.ctx.c_r4[0] = Some(c15);
            // Derive constants 16-30 via ALU
            let c16 = self.emit.emit_alu(Opcode::Add, c8, c8);
            self.ctx.c_r4[1] = Some(c16);
            self.ctx.c_r4[2] = Some(self.emit.emit_alu(Opcode::Add, c16, c1));
            self.ctx.c_r4[3] = Some(self.emit.emit_alu(Opcode::Add, c16, c2));
            self.ctx.c_r4[4] = Some(self.emit.emit_alu(Opcode::Add, c16, c3));
            self.ctx.c_r4[5] = Some(self.emit.emit_alu(Opcode::Add, c16, c4));
            self.ctx.c_r4[6] = Some(self.emit.emit_alu(Opcode::Add, c16, c5));
            self.ctx.c_r4[7] = Some(self.emit.emit_alu(Opcode::Add, c16, c6));
            self.ctx.c_r4[8] = Some(self.emit.emit_alu(Opcode::Add, c16, c7));
            let c24 = self.emit.emit_alu(Opcode::Add, c16, c8);
            self.ctx.c_r4[9] = Some(c24);
            self.ctx.c_r4[10] = Some(self.emit.emit_alu(Opcode::Add, c24, c1));
            self.ctx.c_r4[11] = Some(self.emit.emit_alu(Opcode::Add, c24, c2));
            self.ctx.c_r4[12] = Some(self.emit.emit_alu(Opcode::Add, c24, c3));
            self.ctx.c_r4[13] = Some(self.emit.emit_alu(Opcode::Add, c24, c4));
            self.ctx.c_r4[14] = Some(self.emit.emit_alu(Opcode::Add, c24, c5));
            self.ctx.c_r4[15] = Some(self.emit.emit_alu(Opcode::Add, c24, c6));

            // Load tree[15..30] via forest_p + c_r4[i]
            let forest_p = self.ctx.forest_p.unwrap();
            for i in 0..16 {
                self.emit.set_next_var_type(VarType::Address);
                let addr = self.emit.emit_alu(Opcode::Add, forest_p, self.ctx.c_r4[i].unwrap());
                self.ctx.tc_r4_addr[i] = Some(addr);
                self.emit.set_next_var_type(VarType::TreeCache);
                self.ctx.tc_r4[i] = Some(self.emit.emit_load(addr));
            }

            // Compute 8 pair deltas: td_r4[i] = tc_r4[2i+1] - tc_r4[2i]
            for i in 0..8 {
                self.emit.set_next_var_type(VarType::TreeCache);
                self.ctx.td_r4[i] = Some(self.emit.emit_alu(
                    Opcode::Sub,
                    self.ctx.tc_r4[2 * i + 1].unwrap(),
                    self.ctx.tc_r4[2 * i].unwrap(),
                ));
            }
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

        if self.config.use_tree_reduce_at_depth(3) {
            self.ctx.v_td_r3_01 = Some(self.emit.emit_vbroadcast(self.ctx.td_r3_01.unwrap()));
            self.ctx.v_td_r3_23 = Some(self.emit.emit_vbroadcast(self.ctx.td_r3_23.unwrap()));
            self.ctx.v_td_r3_45 = Some(self.emit.emit_vbroadcast(self.ctx.td_r3_45.unwrap()));
            self.ctx.v_td_r3_67 = Some(self.emit.emit_vbroadcast(self.ctx.td_r3_67.unwrap()));
        }
    }

    /// Ensure depth 4 broadcasts are initialized (lazy init).
    pub(super) fn ensure_depth4_broadcasts(&mut self) {
        if self.depth4_broadcasts_initialized {
            return;
        }
        self.depth4_broadcasts_initialized = true;

        // Broadcast 8 even-indexed bases: tc_r4[0], tc_r4[2], ..., tc_r4[14]
        for i in 0..8 {
            self.emit.set_next_var_type(VarType::TreeCache);
            self.ctx.v_tc_r4_even[i] =
                Some(self.emit.emit_vbroadcast(self.ctx.tc_r4[2 * i].unwrap()));
        }

        // Broadcast 8 deltas: td_r4[0..8]
        for i in 0..8 {
            self.emit.set_next_var_type(VarType::TreeCache);
            self.ctx.v_td_r4[i] =
                Some(self.emit.emit_vbroadcast(self.ctx.td_r4[i].unwrap()));
        }
    }

    // =========================================================================
    // Data Load/Store
    // =========================================================================

    pub(super) fn emit_load_data(&mut self, n_groups: usize) {
        let idx_p = self.ctx.idx_p.unwrap();
        let val_p = self.ctx.val_p.unwrap();
        // The mirrored index starts at n = 0 (`all_idx[g] = None`): the initial indices are
        // taken to be 0, as the root broadcast of round 0 already assumes, and not loaded.
        let load_idx = !self.mirrored_index();

        for g in (0..n_groups).step_by(2) {
            let offset_a = (g * VLEN) as i64;
            self.emit.set_next_var_type(VarType::Constant);
            let off_a = self.emit.emit_const_choice(offset_a);
            if load_idx {
                self.emit.set_next_var_type(VarType::Address);
                let idx_addr_a = self.emit.emit_alu(Opcode::Add, idx_p, off_a);
                self.emit.set_next_var_type(VarType::BatchData);
                self.ctx.all_idx[g] = Some(self.emit.emit_vload(idx_addr_a));
            }
            self.emit.set_next_var_type(VarType::Address);
            let val_addr_a = self.emit.emit_alu(Opcode::Add, val_p, off_a);
            self.emit.set_next_var_type(VarType::BatchData);
            self.ctx.all_val[g] = Some(self.emit.emit_vload(val_addr_a));

            if g + 1 < n_groups {
                let offset_b = ((g + 1) * VLEN) as i64;
                self.emit.set_next_var_type(VarType::Constant);
                let off_b = self.emit.emit_const_choice(offset_b);
                if load_idx {
                    self.emit.set_next_var_type(VarType::Address);
                    let idx_addr_b = self.emit.emit_alu(Opcode::Add, idx_p, off_b);
                    self.emit.set_next_var_type(VarType::BatchData);
                    self.ctx.all_idx[g + 1] = Some(self.emit.emit_vload(idx_addr_b));
                }
                self.emit.set_next_var_type(VarType::Address);
                let val_addr_b = self.emit.emit_alu(Opcode::Add, val_p, off_b);
                self.emit.set_next_var_type(VarType::BatchData);
                self.ctx.all_val[g + 1] = Some(self.emit.emit_vload(val_addr_b));
            }
        }

        // Convert initial values to optimized form
        let v_stage5_xor = self.ctx.v_stage5_xor.unwrap();
        for g in 0..n_groups {
            self.emit.set_next_var_type(VarType::BatchData);
            let val = self.ctx.all_val[g].unwrap();
            self.ctx.all_val[g] = Some(self.emit.emit_valu(Opcode::Xor, val, v_stage5_xor));
        }
    }

    pub(super) fn emit_store_data(
        &mut self,
        n_groups: usize,
        store_data_idxs: &[crate::types::VReg],
        store_data_vals: &[crate::types::VReg],
    ) {
        assert_eq!(store_data_idxs.len(), n_groups);
        assert_eq!(store_data_vals.len(), n_groups);

        let idx_p = self.ctx.idx_p.unwrap();
        let val_p = self.ctx.val_p.unwrap();

        for g in (0..n_groups).step_by(2) {
            let offset_a = (g * VLEN) as i64;
            self.emit.set_next_var_type(VarType::Constant);
            let off_a = self.emit.emit_const_choice(offset_a);
            self.emit.set_next_var_type(VarType::Address);
            let idx_addr_a = self.emit.emit_alu(Opcode::Add, idx_p, off_a);
            self.emit.set_next_var_type(VarType::Address);
            let val_addr_a = self.emit.emit_alu(Opcode::Add, val_p, off_a);
            self.emit.emit_vstore(idx_addr_a, store_data_idxs[g]);
            self.emit.emit_vstore(val_addr_a, store_data_vals[g]);

            if g + 1 < n_groups {
                let offset_b = ((g + 1) * VLEN) as i64;
                self.emit.set_next_var_type(VarType::Constant);
                let off_b = self.emit.emit_const_choice(offset_b);
                self.emit.set_next_var_type(VarType::Address);
                let idx_addr_b = self.emit.emit_alu(Opcode::Add, idx_p, off_b);
                self.emit.set_next_var_type(VarType::Address);
                let val_addr_b = self.emit.emit_alu(Opcode::Add, val_p, off_b);
                self.emit.emit_vstore(idx_addr_b, store_data_idxs[g + 1]);
                self.emit.emit_vstore(val_addr_b, store_data_vals[g + 1]);
            }
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
            0 => self.emit_depth0_access(n_iters, n_groups),
            1 => self.emit_depth1_access(n_iters, n_groups),
            2 => self.emit_depth2_access(n_iters, n_groups),
            3 if self.config.use_tree_reduce_at_depth(3) => {
                self.emit_depth3_access(n_iters, n_groups)
            }
            4 if self.config.use_tree_reduce_at_depth(4) => {
                self.emit_depth4_access(n_iters, n_groups)
            }
            _ => self.emit_scattered_access(n_iters, n_groups, tree_depth),
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
    fn select_2way(
        &mut self,
        cond: crate::types::VReg,
        true_val: crate::types::VReg,
        false_val: crate::types::VReg,
    ) -> crate::types::VReg {
        self.emit.emit_select_choice(cond, true_val, false_val)
    }

    /// 2-way selection with precomputed delta: cond ? (base + delta) : base (one SelectChoice).
    fn select_2way_with_delta(
        &mut self,
        cond: crate::types::VReg,
        delta: crate::types::VReg,
        base: crate::types::VReg,
    ) -> crate::types::VReg {
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
        let cond1 = self
            .emit
            .emit_valu(Opcode::Eq, idx, self.ctx.v_const[c1].unwrap());
        self.emit.set_next_var_type(VarType::Temporary);
        let mut result = self.select_2way(cond1, v1, v0);
        self.emit.set_next_var_type(VarType::Condition);
        let cond2 = self
            .emit
            .emit_valu(Opcode::Eq, idx, self.ctx.v_const[c2].unwrap());
        self.emit.set_next_var_type(VarType::Temporary);
        result = self.select_2way(cond2, v2, result);
        self.emit.set_next_var_type(VarType::Condition);
        let cond3 = self
            .emit
            .emit_valu(Opcode::Eq, idx, self.ctx.v_const[c3].unwrap());
        self.emit.set_next_var_type(VarType::Temporary);
        result = self.select_2way(cond3, v3, result);
        result
    }

    /// 4-way selection via linear chain with first pair as (base, delta) to avoid materializing base+delta.
    ///
    /// # Arguments
    /// * `idx` - The tracked index vector (in the kernel's index form).
    /// * `compare` - Vectors the index equals when the 2nd, 3rd and 4th value is wanted
    ///   (`v_index_compare`).
    /// * `base`, `delta` - 1st value and (2nd - 1st).
    /// * `v2`, `v3` - 3rd and 4th value.
    ///
    /// # Returns
    /// The selected vector.
    fn select_4way_linear_first_delta(
        &mut self,
        idx: crate::types::VReg,
        compare: [crate::types::VReg; 3],
        base: crate::types::VReg,
        delta: crate::types::VReg,
        v2: crate::types::VReg,
        v3: crate::types::VReg,
    ) -> crate::types::VReg {
        self.emit.set_next_var_type(VarType::Condition);
        let cond1 = self.emit.emit_valu(Opcode::Eq, idx, compare[0]);
        self.emit.set_next_var_type(VarType::Temporary);
        let mut result = self.select_2way_with_delta(cond1, delta, base);
        self.emit.set_next_var_type(VarType::Condition);
        let cond2 = self.emit.emit_valu(Opcode::Eq, idx, compare[1]);
        self.emit.set_next_var_type(VarType::Temporary);
        result = self.select_2way(cond2, v2, result);
        self.emit.set_next_var_type(VarType::Condition);
        let cond3 = self.emit.emit_valu(Opcode::Eq, idx, compare[2]);
        self.emit.set_next_var_type(VarType::Temporary);
        result = self.select_2way(cond3, v3, result);
        result
    }

    /// 4-way selection via bit-tree: idx_adj = idx - base_idx (0..4), bit0 = low bit, bit1 = high bit.
    /// Level 0: two parallel 2-way (v0,v1) by bit0 and (v2,v3) by bit0. Level 1: one 2-way by bit1.
    /// Requires v_const[base_idx] for idx_adj. Tree levels set for symmetry breaking.
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
        self.emit.set_symmetry_hint(0);
        let p01 = self.select_2way(bit0, v1, v0);
        let p23 = self.select_2way(bit0, v3, v2);
        self.emit.set_symmetry_hint(1);
        let result = self.select_2way(bit1, p23, p01);
        self.emit.clear_symmetry_hint();
        result
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
    /// Tree levels are set for symmetry breaking in CP-SAT (leaves in order, then level 1, then level 2).
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
        self.emit.set_symmetry_hint(0);
        let p01 = self.select_2way(bit0, values[1], values[0]);
        let p23 = self.select_2way(bit0, values[3], values[2]);
        let p45 = self.select_2way(bit0, values[5], values[4]);
        let p67 = self.select_2way(bit0, values[7], values[6]);
        self.emit.set_symmetry_hint(1);
        let q03 = self.select_2way(bit1, p23, p01);
        let q47 = self.select_2way(bit1, p67, p45);
        self.emit.set_symmetry_hint(2);
        let result = self.select_2way(bit2, q47, q03);
        self.emit.clear_symmetry_hint();
        result
    }

    // =========================================================================
    // Tree Access by Depth
    // =========================================================================

    pub(super) fn emit_depth0_access(&mut self, n_iters: usize, n_groups: usize) {
        // Depth 0: All at root - broadcast tree[0]
        let tc0 = self.ctx.tc[0].unwrap();
        let v_stage5_xor = self.ctx.v_stage5_xor.unwrap();

        for it in 0..n_iters {
            let base_g = it * UNROLL;
            for i in 0..UNROLL {
                if base_g + i >= n_groups {
                    continue;
                }
                self.emit.set_iter_group(it, base_g + i);
                self.emit.set_next_var_type(VarType::Temporary);
                let v_nv = self.emit.emit_vbroadcast_force(tc0);
                self.emit.set_next_var_type(VarType::RoundState);
                self.ctx.v_nv[it][i] = Some(self.emit.emit_valu(Opcode::Xor, v_nv, v_stage5_xor));
            }
        }
        self.emit.clear_iter_group();
    }

    pub(super) fn emit_depth1_access(&mut self, n_iters: usize, n_groups: usize) {
        // Depth 1: 2-way selection (indices 1, 2) via select_2way_with_delta.
        let v_is_1 = self.v_index_compare(1, 1);
        let v_td_1 = self.ctx.v_td_1.unwrap();
        let v_tc2 = self.ctx.v_tc2.unwrap();
        let v_stage5_xor = self.ctx.v_stage5_xor.unwrap();

        for it in 0..n_iters {
            let base_g = it * UNROLL;
            for i in 0..UNROLL {
                if base_g + i >= n_groups {
                    continue;
                }
                self.emit.set_iter_group(it, base_g + i);
                let idx = self.ctx.all_idx[base_g + i].unwrap();
                self.emit.set_next_var_type(VarType::Condition);
                let cond = self.emit.emit_valu(Opcode::Eq, idx, v_is_1);
                self.emit.set_next_var_type(VarType::Temporary);
                let v_nv = self.select_2way_with_delta(cond, v_td_1, v_tc2);
                self.emit.set_next_var_type(VarType::RoundState);
                self.ctx.v_nv[it][i] = Some(self.emit.emit_valu(Opcode::Xor, v_nv, v_stage5_xor));
            }
        }
        self.emit.clear_iter_group();
    }

    pub(super) fn emit_depth2_access(&mut self, n_iters: usize, n_groups: usize) {
        // Depth 2: 4-way selection (indices 3,4,5,6). First pair (tc3,tc4) as (base, delta); rest as full values.
        let v_tc3 = self.ctx.v_tc3.unwrap();
        let v_td_4 = self.ctx.v_td_4.unwrap();
        let v_tc5 = self.ctx.v_tc5.unwrap();
        let v_tc6 = self.ctx.v_tc6.unwrap();
        let v_stage5_xor = self.ctx.v_stage5_xor.unwrap();
        const BASE_IDX: usize = 3;
        let compare: [crate::types::VReg; 3] =
            std::array::from_fn(|k| self.v_index_compare(2, BASE_IDX + 1 + k));

        for it in 0..n_iters {
            let base_g = it * UNROLL;
            for i in 0..UNROLL {
                if base_g + i >= n_groups {
                    continue;
                }
                self.emit.set_iter_group(it, base_g + i);
                let idx = self.ctx.all_idx[base_g + i].unwrap();
                self.emit.set_next_var_type(VarType::Temporary);
                let v_nv =
                    self.select_4way_linear_first_delta(idx, compare, v_tc3, v_td_4, v_tc5, v_tc6);
                self.emit.set_next_var_type(VarType::RoundState);
                self.ctx.v_nv[it][i] = Some(self.emit.emit_valu(Opcode::Xor, v_nv, v_stage5_xor));
            }
        }
        self.emit.clear_iter_group();
    }

    pub(super) fn emit_depth3_access(&mut self, n_iters: usize, n_groups: usize) {
        // Ensure depth 3 broadcasts are initialized
        self.ensure_depth3_broadcasts();

        let v_stage5_xor = self.ctx.v_stage5_xor.unwrap();

        if self.config.use_tree_reduce_at_depth(3) {
            // Tree-reduce: direct FMA at level 0 (with precomputed deltas) +
            // SelectChoice at levels 1-2.
            //
            // Level 0 uses direct FMA because FMA with precomputed delta is strictly
            // better than vselect: multiply_add(bit, delta, base) = 1 VALU op, whereas
            // vselect needs add(base, delta) + vselect = 1 VALU + 1 Flow = more ops + latency.
            //
            // Levels 1-2 use SelectChoice(bit, high, low) — solver picks vselect (1 Flow)
            // or FMA (sub+fma, 2 VALU). Here vselect IS beneficial: it saves 2 VALU ops
            // at the cost of 1 Flow op, helping when VALU is saturated.
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
                    if base_g + i >= n_groups {
                        continue;
                    }
                    self.emit.set_iter_group(it, base_g + i);

                    // Bit extraction
                    self.emit.set_next_var_type(VarType::Temporary);
                    let idx_adj = self.emit_level_position(base_g + i, v_const_7);
                    let bit0 = self.emit.emit_valu(Opcode::And, idx_adj, v_1);
                    let idx_adj_shr = self.emit.emit_valu(Opcode::Shr, idx_adj, v_1);
                    let bit1 = self.emit.emit_valu(Opcode::And, idx_adj_shr, v_1);
                    let bit2 = self.emit.emit_valu(Opcode::Shr, idx_adj_shr, v_1);

                    // Level 0: 4 direct FMA with precomputed deltas (8→4)
                    // multiply_add(bit0, delta, base) = bit0 * delta + base
                    self.emit.set_symmetry_hint(0);
                    self.emit.set_next_var_type(VarType::Temporary);
                    let p01 = self.emit.emit_fma(bit0, v_td_r3_01, v_tc_r3_0);
                    self.emit.set_next_var_type(VarType::Temporary);
                    let p23 = self.emit.emit_fma(bit0, v_td_r3_23, v_tc_r3_2);
                    self.emit.set_next_var_type(VarType::Temporary);
                    let p45 = self.emit.emit_fma(bit0, v_td_r3_45, v_tc_r3_4);
                    self.emit.set_next_var_type(VarType::Temporary);
                    let p67 = self.emit.emit_fma(bit0, v_td_r3_67, v_tc_r3_6);

                    // Level 1: 2 SelectChoice (4→2)
                    // vselect saves 2 VALU ops vs FMA (sub+fma) at cost of 1 Flow op
                    self.emit.set_symmetry_hint(1);
                    self.emit.set_next_var_type(VarType::Temporary);
                    let q03 = self.select_2way(bit1, p23, p01);
                    self.emit.set_next_var_type(VarType::Temporary);
                    let q47 = self.select_2way(bit1, p67, p45);

                    // Level 2: 1 SelectChoice (2→1)
                    self.emit.set_symmetry_hint(2);
                    self.emit.set_next_var_type(VarType::Temporary);
                    let result = self.select_2way(bit2, q47, q03);
                    self.emit.clear_symmetry_hint();

                    self.ctx.v_nv[it][i] =
                        Some(self.emit.emit_valu(Opcode::Xor, result, v_stage5_xor));
                }
            }
        } else {
            // Original FMA approach (16 VALU ops). Not reachable from emit_round, which only
            // calls this function when tree-reduce is enabled at depth 3.
            assert!(
                !self.mirrored_index(),
                "depth-3 selection without tree-reduce supports only the zero-based index form"
            );
            for it in 0..n_iters {
                let base_g = it * UNROLL;
                for i in 0..UNROLL {
                    if base_g + i >= n_groups {
                        continue;
                    }
                    let idx = self.ctx.all_idx[base_g + i].unwrap();

                    // First iteration: Mul
                    let cond0 =
                        self.emit
                            .emit_valu(Opcode::Eq, idx, self.ctx.v_const_r3[0].unwrap());
                    let mut v_nv =
                        self.emit
                            .emit_valu(Opcode::Mul, cond0, self.ctx.v_tc_r3[0].unwrap());

                    // Remaining iterations: FMA
                    for ti in 1..8 {
                        let cond =
                            self.emit
                                .emit_valu(Opcode::Eq, idx, self.ctx.v_const_r3[ti].unwrap());
                        v_nv = self
                            .emit
                            .emit_fma(cond, self.ctx.v_tc_r3[ti].unwrap(), v_nv);
                    }

                    self.ctx.v_nv[it][i] =
                        Some(self.emit.emit_valu(Opcode::Xor, v_nv, v_stage5_xor));
                }
            }
        }
    }

    pub(super) fn emit_depth4_access(&mut self, n_iters: usize, n_groups: usize) {
        // Ensure depth 4 broadcasts are initialized
        self.ensure_depth4_broadcasts();

        let v_stage5_xor = self.ctx.v_stage5_xor.unwrap();

        // 4-level tree for 16-way selection (tree indices 15-30).
        // Uses precomputed broadcasts: v_tc_r4_even[0..8] (even bases) and v_td_r4[0..8] (deltas).
        //
        // Level 0: 8 direct FMA with precomputed deltas (16→8)
        //   FMA is strictly better than vselect here: multiply_add(bit, delta, base) = 1 VALU op,
        //   vs vselect needing add(base, delta) + vselect = 1 VALU + 1 Flow + extra VReg.
        // Levels 1-3: SelectChoice(bit, high, low) → solver picks vselect (1 Flow) or FMA (sub+fma, 2 VALU)
        //   Here vselect IS beneficial: saves 2 VALU ops at cost of 1 Flow op.

        // Broadcast of constant 15 for idx adjustment
        let c15 = self.ctx.c[15].unwrap();
        let v_c15 = self.emit.emit_vbroadcast(c15);
        let v_1 = self.ctx.v_1.unwrap();

        // Cache broadcast references
        let v_td_r4: [crate::types::VReg; 8] = std::array::from_fn(|i| {
            self.ctx.v_td_r4[i].unwrap()
        });
        let v_tc_r4_even: [crate::types::VReg; 8] = std::array::from_fn(|i| {
            self.ctx.v_tc_r4_even[i].unwrap()
        });

        for it in 0..n_iters {
            let base_g = it * UNROLL;
            for i in 0..UNROLL {
                if base_g + i >= n_groups {
                    continue;
                }
                self.emit.set_iter_group(it, base_g + i);

                // Bit extraction: idx_adj = idx - 15, then extract bits 0-3
                self.emit.set_next_var_type(VarType::Temporary);
                let idx_adj = self.emit_level_position(base_g + i, v_c15);
                let bit0 = self.emit.emit_valu(Opcode::And, idx_adj, v_1);
                let idx_adj_shr1 = self.emit.emit_valu(Opcode::Shr, idx_adj, v_1);
                let bit1 = self.emit.emit_valu(Opcode::And, idx_adj_shr1, v_1);
                let idx_adj_shr2 = self.emit.emit_valu(Opcode::Shr, idx_adj_shr1, v_1);
                let bit2 = self.emit.emit_valu(Opcode::And, idx_adj_shr2, v_1);
                let bit3 = self.emit.emit_valu(Opcode::Shr, idx_adj_shr2, v_1);

                // Level 0: 8 direct FMA with precomputed deltas (16→8)
                // multiply_add(bit0, delta, base) = bit0 * delta + base
                self.emit.set_symmetry_hint(0);
                self.emit.set_next_var_type(VarType::Temporary);
                let p0 = self.emit.emit_fma(bit0, v_td_r4[0], v_tc_r4_even[0]);
                self.emit.set_next_var_type(VarType::Temporary);
                let p1 = self.emit.emit_fma(bit0, v_td_r4[1], v_tc_r4_even[1]);
                self.emit.set_next_var_type(VarType::Temporary);
                let p2 = self.emit.emit_fma(bit0, v_td_r4[2], v_tc_r4_even[2]);
                self.emit.set_next_var_type(VarType::Temporary);
                let p3 = self.emit.emit_fma(bit0, v_td_r4[3], v_tc_r4_even[3]);
                self.emit.set_next_var_type(VarType::Temporary);
                let p4 = self.emit.emit_fma(bit0, v_td_r4[4], v_tc_r4_even[4]);
                self.emit.set_next_var_type(VarType::Temporary);
                let p5 = self.emit.emit_fma(bit0, v_td_r4[5], v_tc_r4_even[5]);
                self.emit.set_next_var_type(VarType::Temporary);
                let p6 = self.emit.emit_fma(bit0, v_td_r4[6], v_tc_r4_even[6]);
                self.emit.set_next_var_type(VarType::Temporary);
                let p7 = self.emit.emit_fma(bit0, v_td_r4[7], v_tc_r4_even[7]);

                // Level 1: 4 SelectChoice (8→4)
                // vselect path: vselect(bit1, p_high, p_low) (1 Flow)
                // FMA path: sub + fma (2 VALU) — sub managed by SelectChoice internally
                self.emit.set_symmetry_hint(1);
                self.emit.set_next_var_type(VarType::Temporary);
                let q0 = self.select_2way(bit1, p1, p0);
                self.emit.set_next_var_type(VarType::Temporary);
                let q1 = self.select_2way(bit1, p3, p2);
                self.emit.set_next_var_type(VarType::Temporary);
                let q2 = self.select_2way(bit1, p5, p4);
                self.emit.set_next_var_type(VarType::Temporary);
                let q3 = self.select_2way(bit1, p7, p6);

                // Level 2: 2 SelectChoice (4→2)
                self.emit.set_symmetry_hint(2);
                self.emit.set_next_var_type(VarType::Temporary);
                let r0 = self.select_2way(bit2, q1, q0);
                self.emit.set_next_var_type(VarType::Temporary);
                let r1 = self.select_2way(bit2, q3, q2);

                // Level 3: 1 SelectChoice (2→1)
                self.emit.set_symmetry_hint(3);
                self.emit.set_next_var_type(VarType::Temporary);
                let result = self.select_2way(bit3, r1, r0);
                self.emit.clear_symmetry_hint();

                self.emit.set_next_var_type(VarType::RoundState);
                self.ctx.v_nv[it][i] =
                    Some(self.emit.emit_valu(Opcode::Xor, result, v_stage5_xor));
            }
        }
        self.emit.clear_iter_group();
    }

    pub(super) fn emit_scattered_access(
        &mut self,
        n_iters: usize,
        n_groups: usize,
        tree_depth: usize,
    ) {
        // Scatter load via vector address computation + scalar decomposition.
        //
        // Each of the 8 batch items in a group has its own tree node index
        // (stored per-element in the vector VReg `all_idx[g]`). We:
        //   1. Compute all 8 addresses in one VALU add: vadd(v_idx, v_forest_p)
        //   2. Decompose the address vector into 8 scalar elements
        //   3. Load each tree node value via scalar load
        //   4. Assemble the 8 scalar results into a vector for downstream ops
        //
        // The vbroadcast of forest_p is CSE'd (broadcast_cache), so it's shared
        // across all groups and rounds. Each group costs 1 vadd (shiftable) +
        // 8 loads instead of 8 ALU adds + 8 loads.

        //
        // With the mirrored index n = L_d - idx the address is (forest_p + L_d) - n: still
        // one vector op, against a per-level constant (R10-mirrored-index-uses).

        let forest_p = self.ctx.forest_p.unwrap();
        let mirrored = self.mirrored_index();
        let v_base = if mirrored {
            self.emit.set_next_var_type(VarType::Constant);
            let last = self.emit.emit_const_choice(level_last_index(tree_depth));
            self.emit.set_next_var_type(VarType::Pointer);
            let level_end_p = self.emit.emit_alu(Opcode::Add, forest_p, last);
            self.emit.emit_vbroadcast(level_end_p)
        } else {
            self.emit.emit_vbroadcast(forest_p)
        };
        let v_stage5_xor = self.ctx.v_stage5_xor.unwrap();

        for it in 0..n_iters {
            let base_g = it * UNROLL;
            for i in 0..UNROLL {
                if base_g + i >= n_groups {
                    continue;
                }
                self.emit.set_iter_group(it, base_g + i);
                let idx = self.ctx.all_idx[base_g + i].unwrap();

                // Vector address computation: forest_p + idx[item] for all 8 items
                self.emit.set_next_var_type(VarType::Address);
                let v_addr = if mirrored {
                    self.emit.emit_valu(Opcode::Sub, v_base, idx)
                } else {
                    self.emit.emit_valu(Opcode::Add, idx, v_base)
                };

                // Decompose address vector into 8 scalar addresses
                let addr_elements = self.emit.decompose_vector(v_addr);

                // Result vector group: 8 scalar load results assembled into a vector
                self.emit.set_next_var_type(VarType::Temporary);
                let (v_nv, result_elements) = self.emit.fresh_vector_group();

                // Scalar loads from each computed address
                for elem in 0..VLEN {
                    self.emit.set_next_var_type(VarType::Temporary);
                    self.emit.emit_load_into(addr_elements[elem], result_elements[elem]);
                }

                // XOR with stage5 constant
                self.emit.set_next_var_type(VarType::RoundState);
                self.ctx.v_nv[it][i] =
                    Some(self.emit.emit_valu(Opcode::Xor, v_nv, v_stage5_xor));
            }
        }
        self.emit.clear_iter_group();
    }

    // =========================================================================
    // XOR Phase
    // =========================================================================

    pub(super) fn emit_xor(&mut self, n_groups: usize) {
        let n_iters = if n_groups == 0 {
            0
        } else {
            (n_groups + UNROLL - 1) / UNROLL
        };

        for it in 0..n_iters {
            let base_g = it * UNROLL;
            for i in 0..UNROLL {
                let g = base_g + i;
                if g >= n_groups {
                    continue;
                }
                let val = self.ctx.all_val[g].unwrap();
                let v_nv = self.ctx.v_nv[it][i].unwrap();
                self.ctx.all_val[g] = Some(self.emit.emit_valu(Opcode::Xor, val, v_nv));
            }
        }
    }

    // =========================================================================
    // Hash Phase
    // =========================================================================

    pub(super) fn emit_hash(&mut self, n_groups: usize, n_iters: usize) {
        let rnd = self.emit.current_round.unwrap_or(0);
        let enable_trace = self.config.spec.enable_trace_ids;
        let debug = std::env::var("SCHEDULER_DEBUG").is_ok();

        let merge = self.config.spec.merge_hash_stages;

        for hi in 0..6 {
            let (op1, _, op2, op3, _) = HASH_STAGES[hi];
            // Stage 3 is emitted together with stage 2 when the two are merged.
            if merge && hi == MERGED_HASH_STAGE + 1 {
                continue;
            }

            for it in 0..n_iters {
                let base_g = it * UNROLL;
                for i in 0..UNROLL {
                    let g = base_g + i;
                    if g >= n_groups {
                        continue;
                    }
                    let val = self.ctx.all_val[g].unwrap();

                    if merge && hi == MERGED_HASH_STAGE {
                        // Stages 2+3 in 3 ops: (val * m1 + c1) ^ (val * m2 + c2)
                        // (R8-hash-stage-merge in verification/rewrites.py).
                        let v_m1 = *self.ctx.v_hash_mult.get(&hi).unwrap();
                        let (v_c1, v_m2, v_c2) = self.ctx.v_hash_merged.unwrap();
                        let left = self.emit.emit_fma(val, v_m1, v_c1);
                        let right = self.emit.emit_fma(val, v_m2, v_c2);
                        self.ctx.all_val[g] = Some(self.emit.emit_valu(Opcode::Xor, left, right));
                        continue;
                    }
                    
                    // Debug: log Round N hash reading Round N-1 hash (or initial value)
                    if debug && hi == 0 && it == 0 && i == 0 {
                        eprintln!(
                            "[EMISSION DEBUG] Round {} hash phase: reading all_val[{}] = VReg {:?} (Round {} hash output)",
                            rnd, g, val, if rnd > 0 { rnd - 1 } else { 0 }
                        );
                    }

                    if self.ctx.hash_fma_mult[hi].is_some() {
                        // FMA optimization
                        if enable_trace && hi == 5 {
                            self.emit
                                .set_next_trace_id(format!("r{}_g{}_hashed_val", rnd, g));
                        }
                        let v_mult = *self.ctx.v_hash_mult.get(&hi).unwrap();
                        let v_hc = self.ctx.v_hc[hi].unwrap();
                        let new_val = self.emit.emit_fma(val, v_mult, v_hc);
                        // Debug: log Round N hash updating all_val[g]
                        if debug && hi == 5 && it == 0 && i == 0 {
                            eprintln!(
                                "[EMISSION DEBUG] Round {} hash phase (stage 5 FMA): updating all_val[{}] = VReg {:?} -> {:?}",
                                rnd, g, val, new_val
                            );
                        }
                        self.ctx.all_val[g] = Some(new_val);
                    } else if hi == 5 {
                        // Stage 5 special handling (Shr + Xor; Xor produces hashed_val)
                        let v_hs = self.ctx.v_hs[hi].unwrap();
                        let t = self.emit.emit_valu(Opcode::Shr, val, v_hs);
                        if enable_trace {
                            self.emit
                                .set_next_trace_id(format!("r{}_g{}_hashed_val", rnd, g));
                        }
                        let new_val = self.emit.emit_valu(Opcode::Xor, val, t);
                        // Debug: log Round N hash updating all_val[g]
                        if debug && it == 0 && i == 0 {
                            eprintln!(
                                "[EMISSION DEBUG] Round {} hash phase (stage 5): updating all_val[{}] = VReg {:?} -> {:?}",
                                rnd, g, val, new_val
                            );
                        }
                        self.ctx.all_val[g] = Some(new_val);
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
                        if enable_trace && hi == 5 {
                            self.emit
                                .set_next_trace_id(format!("r{}_g{}_hashed_val", rnd, g));
                        }
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
        let is_wrap_round = tree_depth == self.config.spec.forest_height;
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

                if self.mirrored_index() {
                    // Mirrored index n = L_d - idx (R9-mirrored-index-step):
                    //   wrap round: every index becomes 0, so n = 0 (None), no op;
                    //   depth 0:    n' = bit (n = 0), one op;
                    //   otherwise:  n' = 2n + bit, two ops.
                    self.ctx.all_idx[g] = if is_wrap_round {
                        None
                    } else {
                        let bit = self.emit.emit_valu(Opcode::And, val, v_1);
                        Some(match self.ctx.all_idx[g] {
                            None => bit,
                            Some(n) => self.emit.emit_fma(n, v_2, bit),
                        })
                    };
                    continue;
                }

                let idx = self.ctx.all_idx[g].unwrap();
                
                // Debug: log what VReg all_val[g] contains when branch runs
                let debug = std::env::var("SCHEDULER_DEBUG").is_ok();
                if debug {
                    let rnd = self.emit.current_round.unwrap_or(0);
                    eprintln!(
                        "[EMISSION DEBUG] Round {} branch phase: reading all_val[{}] = VReg {:?} (group {})",
                        rnd, g, val, g
                    );
                }

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
        if self.config.spec.enable_partition_stores {
            assert!(
                !self.mirrored_index(),
                "partition stores support only the zero-based index form"
            );
            if TICK_STORE_DEPTHS.contains(&tree_depth) {
                if std::env::var("SCHEDULER_DEBUG").is_ok() && is_last_round {
                    eprintln!("[DEBUG emit_branch] About to emit partition stores for groups 0..{}, tree_depth={}", n_groups/2, tree_depth);
                    for g in 0..n_groups / 2 {
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
        let base_offset = self.emit.emit_const_choice((start_g * VLEN) as i64);
        let mut idx_addr = self.emit.emit_alu(Opcode::Add, extra_room_p, base_offset);
        let mut val_addr = self
            .emit
            .emit_alu(Opcode::Add, idx_addr, partition_val_offset);
        let vlen_const = self.emit.emit_const_choice(VLEN as i64);

        // Store all groups
        for g in start_g..end_g {
            let idx_vreg = self.ctx.all_idx[g].unwrap();
            let val_vreg = self.ctx.all_val[g].unwrap();
            if debug {
                eprintln!("[DEBUG emit_partition_stores] g={}: vstore(idx_addr={:?}, idx={:?}), vstore(val_addr={:?}, val={:?})",
                    g, idx_addr, idx_vreg, val_addr, val_vreg);
            }
            self.emit.emit_vstore(idx_addr, idx_vreg);
            self.emit.emit_vstore(val_addr, val_vreg);

            // Increment addresses for next group
            if g < end_g - 1 {
                idx_addr = self.emit.emit_alu(Opcode::Add, idx_addr, vlen_const);
                val_addr = self.emit.emit_alu(Opcode::Add, val_addr, vlen_const);
            }
        }
    }
}
