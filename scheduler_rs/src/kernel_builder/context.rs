//! Scratch memory allocation context.
//!
//! This module contains the ScratchContext struct that tracks all allocated
//! scratch memory locations across kernel building phases.

use std::collections::HashMap;

use super::config::VLEN;

/// Reserved address 0 always contains zero (scratch memory is initialized to 0).
/// This eliminates the need to load constant 0 - it's always available.
pub const ZERO_REG: i64 = 0;

/// Holds all allocated scratch memory locations.
///
/// This is a Rust port of the Python ScratchContext class, tracking all
/// scratch addresses needed for kernel execution.
#[derive(Debug, Clone)]
pub struct ScratchContext {
    // =========================================================================
    // Pointers (loaded from memory at init)
    // =========================================================================
    pub forest_p: i64,  // Pointer to forest data
    pub idx_p: i64,     // Pointer to indices
    pub val_p: i64,     // Pointer to values
    pub n_nodes_s: i64, // Number of nodes (scalar)

    // =========================================================================
    // Scalar Temporaries
    // =========================================================================
    pub st: [i64; 16], // General-purpose scalar temps
    pub c: [i64; 16],  // Constants 0-15

    // =========================================================================
    // Hash Constants
    // =========================================================================
    pub hc: [i64; 6], // Hash constants (scalar)
    pub hs: [i64; 6], // Hash shifts (scalar)

    // =========================================================================
    // Tree Cache (depths 0-2)
    // =========================================================================
    pub tc: [i64; 7],      // tree[0..6] cached values
    pub tc_addr: [i64; 7], // Address registers for parallel tree cache load
    pub td_1: i64,         // tree[1] - tree[2] (delta for depth 1)
    pub td_4: i64,         // tree[4] - tree[3] (delta for depth 2)
    pub td_5: i64,         // tree[5] - tree[3]
    pub td_6: i64,         // tree[6] - tree[3]

    // =========================================================================
    // Depth 3 Cache
    // =========================================================================
    pub tc_r3: [i64; 8],      // tree[7..14] for depth 3 VALU selection
    pub tc_r3_addr: [i64; 8], // Address registers for parallel depth 3 cache load
    pub c_r3: [i64; 8],       // Constants 7-14

    // =========================================================================
    // Stage 5 XOR Optimization
    // =========================================================================
    pub stage5_xor_s: i64, // 0xB55A4F09 scalar

    // =========================================================================
    // Per-Group Vectors
    // These are dynamically sized based on n_groups
    // =========================================================================
    pub all_idx: Vec<i64>,   // [n_groups] - base addresses for idx vectors
    pub all_val: Vec<i64>,   // [n_groups] - base addresses for val vectors
    pub v_nv: Vec<Vec<i64>>, // [n_iters][UNROLL] - tree value vectors
    pub v_t: Vec<Vec<i64>>,  // [n_iters][UNROLL] - temp vectors

    // =========================================================================
    // Vector Constants
    // =========================================================================
    pub v_0: i64,          // Vector of 0s
    pub v_1: i64,          // Vector of 1s
    pub v_2: i64,          // Vector of 2s
    pub v_nn: i64,         // Vector of n_nodes
    pub v_const: [i64; 8], // Vector constants 0-7

    // =========================================================================
    // Vector Hash Constants
    // =========================================================================
    pub v_hc: [i64; 6], // Vector hash constants
    pub v_hs: [i64; 6], // Vector hash shifts

    // =========================================================================
    // Vector Tree Cache
    // =========================================================================
    pub v_td_1: i64, // Vector td_1
    pub v_td_4: i64, // Vector td_4
    pub v_td_5: i64, // Vector td_5
    pub v_td_6: i64, // Vector td_6
    pub v_tc2: i64,  // Vector tc[2]
    pub v_tc3: i64,  // Vector tc[3]

    // =========================================================================
    // Depth 3 Vectors
    // =========================================================================
    pub v_tc_r3: [i64; 8],    // Vector broadcasts of tc_r3
    pub v_const_r3: [i64; 8], // Vector broadcasts of c_r3

    // =========================================================================
    // Depth 3 Tree-Reduce Deltas
    // =========================================================================
    // Scalar deltas: d01 = tree[8]-tree[7], d23 = tree[10]-tree[9], etc.
    pub td_r3_01: i64, // tree[8] - tree[7]
    pub td_r3_23: i64, // tree[10] - tree[9]
    pub td_r3_45: i64, // tree[12] - tree[11]
    pub td_r3_67: i64, // tree[14] - tree[13]

    // Vector broadcasts of tree-reduce deltas
    pub v_td_r3_01: i64,
    pub v_td_r3_23: i64,
    pub v_td_r3_45: i64,
    pub v_td_r3_67: i64,

    // Per-group scratch registers for tree-reduce (allocated like v_t)
    // These are needed to avoid inter-group dependencies when using tree-reduce
    pub v_tr_s1: Vec<Vec<i64>>, // [n_iters][UNROLL] - bit0, then p67
    pub v_tr_s2: Vec<Vec<i64>>, // [n_iters][UNROLL] - bit1
    pub v_tr_s3: Vec<Vec<i64>>, // [n_iters][UNROLL] - bit2
    pub v_tr_s4: Vec<Vec<i64>>, // [n_iters][UNROLL] - p45, then q47

    // =========================================================================
    // Stage 5 XOR Vector
    // =========================================================================
    pub v_stage5_xor: i64,

    // =========================================================================
    // FMA Multipliers for Hash Optimization
    // =========================================================================
    pub hash_fma_mult: [Option<i64>; 6], // Pre-computed FMA multipliers
    pub hash_fma_mult_s: HashMap<usize, i64>, // stage -> scalar address
    pub v_hash_mult: HashMap<usize, i64>, // stage -> vector address

    // =========================================================================
    // Partition-specific (TickTock)
    // =========================================================================
    pub extra_room_p: i64,         // Pointer to extra_room memory
    pub partition_idx_offset: i64, // Offset to idx section in partition buffer
    pub partition_val_offset: i64, // Offset to val section in partition buffer
    pub tree_cache_scalars: i64,   // Scalar storage for sorted vload cache
    pub tree_cache_base_s: i64,    // Base index of cache
    pub v_cache_base: i64,         // Vector broadcast of cache base
    pub v_tree_temp: i64,          // Temp vector for tree operations

    // =========================================================================
    // Lazy Partition Buffers (for sorted vload tree access)
    // =========================================================================
    /// Double-buffer base addresses in extra_room
    /// Buffer A: indices at offset 0, values at offset 256
    /// Buffer B: indices at offset 512, values at offset 768
    pub part_buffer_a_idx: i64, // Base address for buffer A indices
    pub part_buffer_a_val: i64, // Base address for buffer A values
    pub part_buffer_b_idx: i64, // Base address for buffer B indices
    pub part_buffer_b_val: i64, // Base address for buffer B values

    /// Partition counters (for stable partition algorithm)
    pub part_left_count: i64, // Count of items going to left partition
    pub part_right_count: i64, // Count of items going to right partition

    /// Constants for partition operations
    pub part_mask_11bit: i64, // 0x7FF for index extraction (if packed)
    pub part_batch_half: i64, // 128 (batch_size / 2)

    /// Vector temps for partition operations
    pub v_part_temp: [i64; 4], // Temp vectors for partition computation

    // =========================================================================
    // Partition Prefix Sum Support
    // =========================================================================
    /// Vector holding [0,1,2,3,4,5,6,7] for position-based comparisons
    pub v_positions: i64,
    /// Scratch space for storing branch bits during partition (8 scalars)
    pub part_branch_scratch: i64,
    /// Scratch space for storing indices during partition scatter
    pub part_idx_scratch: i64,
    /// Scratch space for storing values during partition scatter
    pub part_val_scratch: i64,
    /// Destination position vector (computed from prefix sums)
    pub v_dest_pos: i64,
    /// Additional temp vectors for prefix sum computation
    pub v_prefix_temp: [i64; 2],
    /// Vector for broadcasting individual branch bits
    pub v_branch_broadcast: i64,
    /// Left count for current group (scalar)
    pub part_left_count_temp: i64,
}

impl Default for ScratchContext {
    fn default() -> Self {
        ScratchContext {
            forest_p: 0,
            idx_p: 0,
            val_p: 0,
            n_nodes_s: 0,

            st: [0; 16],
            c: [0; 16],

            hc: [0; 6],
            hs: [0; 6],

            tc: [0; 7],
            tc_addr: [0; 7],
            td_1: 0,
            td_4: 0,
            td_5: 0,
            td_6: 0,

            tc_r3: [0; 8],
            tc_r3_addr: [0; 8],
            c_r3: [0; 8],

            stage5_xor_s: 0,

            all_idx: Vec::new(),
            all_val: Vec::new(),
            v_nv: Vec::new(),
            v_t: Vec::new(),

            v_0: 0,
            v_1: 0,
            v_2: 0,
            v_nn: 0,
            v_const: [0; 8],

            v_hc: [0; 6],
            v_hs: [0; 6],

            v_td_1: 0,
            v_td_4: 0,
            v_td_5: 0,
            v_td_6: 0,
            v_tc2: 0,
            v_tc3: 0,

            v_tc_r3: [0; 8],
            v_const_r3: [0; 8],

            td_r3_01: 0,
            td_r3_23: 0,
            td_r3_45: 0,
            td_r3_67: 0,

            v_td_r3_01: 0,
            v_td_r3_23: 0,
            v_td_r3_45: 0,
            v_td_r3_67: 0,

            v_tr_s1: Vec::new(),
            v_tr_s2: Vec::new(),
            v_tr_s3: Vec::new(),
            v_tr_s4: Vec::new(),

            v_stage5_xor: 0,

            hash_fma_mult: [None; 6],
            hash_fma_mult_s: HashMap::new(),
            v_hash_mult: HashMap::new(),

            extra_room_p: 0,
            partition_idx_offset: 0,
            partition_val_offset: 0,
            tree_cache_scalars: 0,
            tree_cache_base_s: 0,
            v_cache_base: 0,
            v_tree_temp: 0,

            part_buffer_a_idx: 0,
            part_buffer_a_val: 0,
            part_buffer_b_idx: 0,
            part_buffer_b_val: 0,
            part_left_count: 0,
            part_right_count: 0,
            part_mask_11bit: 0,
            part_batch_half: 0,
            v_part_temp: [0; 4],

            v_positions: 0,
            part_branch_scratch: 0,
            part_idx_scratch: 0,
            part_val_scratch: 0,
            v_dest_pos: 0,
            v_prefix_temp: [0; 2],
            v_branch_broadcast: 0,
            part_left_count_temp: 0,
        }
    }
}

/// Scratch allocator - tracks allocation pointer and debug info
#[derive(Debug, Clone)]
pub struct ScratchAllocator {
    /// Current allocation pointer
    pub ptr: i64,
    /// Debug map: address -> (name, length)
    pub debug_map: HashMap<i64, (String, usize)>,
}

impl Default for ScratchAllocator {
    fn default() -> Self {
        ScratchAllocator {
            // Start at 1 to reserve address 0 as the zero constant (ZERO_REG)
            ptr: 1,
            debug_map: HashMap::new(),
        }
    }
}

impl ScratchAllocator {
    /// Allocate scratch memory
    pub fn alloc(&mut self, name: &str, length: usize) -> i64 {
        let addr = self.ptr;
        self.debug_map.insert(addr, (name.to_string(), length));
        self.ptr += length as i64;
        addr
    }

    /// Allocate a vector (VLEN words)
    pub fn vec(&mut self, name: &str) -> i64 {
        self.alloc(name, VLEN)
    }

    /// Get current usage
    pub fn usage(&self) -> i64 {
        self.ptr
    }
}
