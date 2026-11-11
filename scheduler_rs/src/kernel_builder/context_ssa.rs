//! SSA-based scratch memory context.
//!
//! This module contains the SsaScratchContext struct that holds VRegs
//! instead of concrete addresses. Unlike ScratchContext, VRegs are created
//! dynamically during emission and addresses are assigned after scheduling.

use std::collections::HashMap;

use crate::types::VReg;

/// SSA scratch context - holds VRegs for all kernel values.
///
/// Unlike ScratchContext which pre-allocates concrete addresses, this struct
/// holds VRegs that are created dynamically by SsaEmitContext. Addresses are
/// assigned after scheduling via linear scan allocation.
#[derive(Debug, Clone, Default)]
pub struct SsaScratchContext {
    // =========================================================================
    // Pointers (loaded from memory at init)
    // =========================================================================
    pub forest_p: Option<VReg>,  // Pointer to forest data
    pub idx_p: Option<VReg>,     // Pointer to indices
    pub val_p: Option<VReg>,     // Pointer to values
    pub n_nodes_s: Option<VReg>, // Number of nodes (scalar)

    // =========================================================================
    // Scalar Temporaries
    // =========================================================================
    pub st: [Option<VReg>; 16], // General-purpose scalar temps
    pub c: [Option<VReg>; 16],  // Constants 0-15

    // =========================================================================
    // Hash Constants
    // =========================================================================
    pub hc: [Option<VReg>; 6], // Hash constants (scalar)
    pub hs: [Option<VReg>; 6], // Hash shifts (scalar)

    // =========================================================================
    // Tree Cache (depths 0-2)
    // =========================================================================
    pub tc: [Option<VReg>; 7],      // tree[0..6] cached values
    pub tc_addr: [Option<VReg>; 7], // Address registers for parallel tree cache load
    pub td_1: Option<VReg>,         // tree[1] - tree[2] (delta for depth 1)
    pub td_4: Option<VReg>,         // tree[4] - tree[3] (delta for depth 2)
    pub td_5: Option<VReg>,         // tree[5] - tree[3]
    pub td_6: Option<VReg>,         // tree[6] - tree[3]

    // =========================================================================
    // Depth 3 Cache
    // =========================================================================
    pub tc_r3: [Option<VReg>; 8], // tree[7..14] for depth 3 VALU selection
    pub tc_r3_addr: [Option<VReg>; 8], // Address registers for parallel depth 3 cache load
    pub c_r3: [Option<VReg>; 8],  // Constants 7-14

    // =========================================================================
    // Depth 4 Cache (tree[15..30])
    // =========================================================================
    pub tc_r4: [Option<VReg>; 16],       // tree[15..30] scalar values
    pub tc_r4_addr: [Option<VReg>; 16],  // address registers (short-lived)
    pub c_r4: [Option<VReg>; 16],        // constants 15-30

    // Depth 4 Tree-Reduce Deltas (8 pairs)
    pub td_r4: [Option<VReg>; 8], // td_r4[i] = tc_r4[2i+1] - tc_r4[2i]

    // Depth 4 vector broadcasts (lazy init)
    pub v_tc_r4_even: [Option<VReg>; 8], // broadcasts of tc_r4[0,2,4,...,14] (even-indexed bases)
    pub v_td_r4: [Option<VReg>; 8],      // broadcasts of td_r4[0..8] (deltas)

    // =========================================================================
    // Stage 5 XOR Optimization
    // =========================================================================
    pub stage5_xor_s: Option<VReg>, // 0xB55A4F09 scalar

    // =========================================================================
    // Per-Group Vectors
    // These are dynamically sized based on n_groups
    // =========================================================================
    pub all_idx: Vec<Option<VReg>>,   // [n_groups] - idx vectors
    pub all_val: Vec<Option<VReg>>,   // [n_groups] - val vectors
    pub v_nv: Vec<Vec<Option<VReg>>>, // [n_iters][UNROLL] - tree value vectors
    pub v_t: Vec<Vec<Option<VReg>>>,  // [n_iters][UNROLL] - temp vectors

    // =========================================================================
    // Vector Constants
    // =========================================================================
    pub v_0: Option<VReg>,          // Vector of 0s
    pub v_1: Option<VReg>,          // Vector of 1s
    pub v_2: Option<VReg>,          // Vector of 2s
    pub v_nn: Option<VReg>,         // Vector of n_nodes
    pub v_const: [Option<VReg>; 8], // Vector constants 0-7

    // =========================================================================
    // Vector Hash Constants
    // =========================================================================
    pub v_hc: [Option<VReg>; 6], // Vector hash constants
    pub v_hs: [Option<VReg>; 6], // Vector hash shifts

    // =========================================================================
    // Vector Tree Cache
    // =========================================================================
    pub v_td_1: Option<VReg>, // Vector td_1
    pub v_td_4: Option<VReg>, // Vector td_4
    pub v_td_5: Option<VReg>, // Vector td_5
    pub v_td_6: Option<VReg>, // Vector td_6
    pub v_tc1: Option<VReg>,  // Vector tc[1] (computed as v_tc2 + v_td_1)
    pub v_tc2: Option<VReg>,  // Vector tc[2]
    pub v_tc3: Option<VReg>,  // Vector tc[3]
    pub v_tc4: Option<VReg>,  // Vector tc[4] (computed as v_tc3 + v_td_4)
    pub v_tc5: Option<VReg>,  // Vector tc[5] (computed as v_tc3 + v_td_5)
    pub v_tc6: Option<VReg>,  // Vector tc[6] (computed as v_tc3 + v_td_6)

    // =========================================================================
    // Depth 3 Vectors
    // =========================================================================
    pub v_tc_r3: [Option<VReg>; 8],    // Vector broadcasts of tc_r3
    pub v_const_r3: [Option<VReg>; 8], // Vector broadcasts of c_r3

    // =========================================================================
    // Depth 3 Tree-Reduce Deltas
    // =========================================================================
    pub td_r3_01: Option<VReg>, // tree[8] - tree[7]
    pub td_r3_23: Option<VReg>, // tree[10] - tree[9]
    pub td_r3_45: Option<VReg>, // tree[12] - tree[11]
    pub td_r3_67: Option<VReg>, // tree[14] - tree[13]

    // Vector broadcasts of tree-reduce deltas
    pub v_td_r3_01: Option<VReg>,
    pub v_td_r3_23: Option<VReg>,
    pub v_td_r3_45: Option<VReg>,
    pub v_td_r3_67: Option<VReg>,

    // Per-group scratch registers for tree-reduce
    pub v_tr_s1: Vec<Vec<Option<VReg>>>, // [n_iters][UNROLL]
    pub v_tr_s2: Vec<Vec<Option<VReg>>>, // [n_iters][UNROLL]
    pub v_tr_s3: Vec<Vec<Option<VReg>>>, // [n_iters][UNROLL]
    pub v_tr_s4: Vec<Vec<Option<VReg>>>, // [n_iters][UNROLL]

    // =========================================================================
    // Stage 5 XOR Vector
    // =========================================================================
    pub v_stage5_xor: Option<VReg>,

    // =========================================================================
    // FMA Multipliers for Hash Optimization
    // =========================================================================
    pub hash_fma_mult: [Option<i64>; 6], // Pre-computed FMA multipliers
    pub hash_fma_mult_s: HashMap<usize, VReg>, // stage -> scalar VReg
    pub v_hash_mult: HashMap<usize, VReg>, // stage -> vector VReg
    /// Merged hash stages 2+3 (`ProblemSpec::merge_hash_stages`): vectors of
    /// `(c1, m2, c2)` in `(a * m1 + c1) ^ (a * m2 + c2)`; `m1` is `v_hash_mult[2]`.
    pub v_hash_merged: Option<(VReg, VReg, VReg)>,

    // =========================================================================
    // Partition-specific (TickTock)
    // =========================================================================
    pub extra_room_p: Option<VReg>, // Pointer to extra_room memory
    pub partition_idx_offset: Option<VReg>, // Offset to idx section in partition buffer
    pub partition_val_offset: Option<VReg>, // Offset to val section in partition buffer
    pub part_left_count: Option<VReg>, // Partition counter
    pub part_right_count: Option<VReg>, // Partition counter

    /// Snapshot of all_val after round (rounds-2) hash. Used by emission to attach
    /// keep_live on the StoreData xor so the allocator keeps that hash output live
    /// until the final xor (transitive pipeline dependency).
    pub round_before_last_hash: Option<Vec<Option<VReg>>>,
}

impl SsaScratchContext {
    /// Create a new SSA scratch context with pre-sized vectors.
    pub fn new(n_groups: usize, n_iters: usize) -> Self {
        let unroll = 8;
        SsaScratchContext {
            all_idx: vec![None; n_groups],
            all_val: vec![None; n_groups],
            v_nv: (0..n_iters).map(|_| vec![None; unroll]).collect(),
            v_t: (0..n_iters).map(|_| vec![None; unroll]).collect(),
            v_tr_s1: (0..n_iters).map(|_| vec![None; unroll]).collect(),
            v_tr_s2: (0..n_iters).map(|_| vec![None; unroll]).collect(),
            v_tr_s3: (0..n_iters).map(|_| vec![None; unroll]).collect(),
            v_tr_s4: (0..n_iters).map(|_| vec![None; unroll]).collect(),
            ..Default::default()
        }
    }

    /// Get a constant VReg, panicking if not set.
    pub fn get_c(&self, i: usize) -> VReg {
        self.c[i].expect(&format!("Constant c[{}] not initialized", i))
    }

    /// Get a hash constant VReg, panicking if not set.
    pub fn get_hc(&self, i: usize) -> VReg {
        self.hc[i].expect(&format!("Hash constant hc[{}] not initialized", i))
    }

    /// Get a hash shift VReg, panicking if not set.
    pub fn get_hs(&self, i: usize) -> VReg {
        self.hs[i].expect(&format!("Hash shift hs[{}] not initialized", i))
    }

    /// Get a vector constant VReg, panicking if not set.
    pub fn get_v_const(&self, i: usize) -> VReg {
        self.v_const[i].expect(&format!("Vector constant v_const[{}] not initialized", i))
    }

    /// Get a vector hash constant VReg, panicking if not set.
    pub fn get_v_hc(&self, i: usize) -> VReg {
        self.v_hc[i].expect(&format!("Vector hash constant v_hc[{}] not initialized", i))
    }

    /// Get a vector hash shift VReg, panicking if not set.
    pub fn get_v_hs(&self, i: usize) -> VReg {
        self.v_hs[i].expect(&format!("Vector hash shift v_hs[{}] not initialized", i))
    }
}
