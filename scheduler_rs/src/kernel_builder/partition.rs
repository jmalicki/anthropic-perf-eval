//! Lazy partition implementation for sorted vload tree access.
//!
//! This module implements memory-based stable partition for tree access optimization.
//! At depths 4+, scattered loads (8 per group) are replaced with 1 vload + 2-way select.
//!
//! ## Key Insight: Memory-Based Scatter Works
//!
//! Although this SIMD architecture lacks cross-lane shuffle operations, we can
//! achieve stable partition through memory-based scatter:
//! 1. Store all values to scratch (1 vstore)
//! 2. For each item: compute destination via branch bit, store to that location
//! 3. Items are now physically reordered in memory
//!
//! The critical realization: **partition stores overlap with the OTHER partition's
//! hash computation** (tick-tock), so they're essentially FREE on the critical path.
//!
//! ## Configuration: k=1 (Partition Every Round)
//!
//! With `partition_frequency = 1`:
//! - Items are partitioned by branch bit every round
//! - Only 1 bit unsorted = **2-way selection** (trivial!)
//! - Position after partition tells us which 2 tree nodes the item could access
//!
//! ## Memory Layout (in extra_room)
//!
//! Double-buffered layout for reading/writing during alternating rounds:
//!
//! ```text
//! Offset    Size    Content
//! 0         256     Buffer A: indices (8 words/group x 32 groups)
//! 256       256     Buffer A: values (8 words/group x 32 groups)
//! 512       256     Buffer B: indices
//! 768       256     Buffer B: values
//! ```
//!
//! Total: 1024 words in extra_room
//!
//! ## 2-Way Selection (k=1)
//!
//! With partition_frequency=1, items are partitioned every round. After partition,
//! only the CURRENT round's branch bit is unsorted. This means:
//! - Items in left partition (bit=0) all go to tree[idx] where idx is even
//! - Items in right partition (bit=1) all go to tree[idx] where idx is odd
//!
//! For tree access: vload 8 consecutive values, then 2-way select:
//! ```text
//! tree_base = min_idx & ~1  // Round down to even
//! tree_vals = vload(forest + tree_base, 8)
//! bit = val & 1
//! result = bit ? tree_vals[idx - tree_base + 1] : tree_vals[idx - tree_base]
//! ```

use crate::types::Opcode;

use super::config::{KernelConfig, VLEN};
use super::context::ScratchContext;
use super::emit::EmitContext;

/// Buffer offsets in extra_room memory
/// TIK/TOK split layout - each partition is 128 items (16 groups)
///
/// TIK stores during TOK's hash, TOK stores during TIK's hash.
/// This natural interleaving means we only need 512 words total!
pub const TIK_IDX_OFFSET: i64 = 0;       // 128 words (8 words/group x 16 groups)
pub const TIK_VAL_OFFSET: i64 = 128;     // 128 words
pub const TOK_IDX_OFFSET: i64 = 256;     // 128 words
pub const TOK_VAL_OFFSET: i64 = 384;     // 128 words
// Total: 512 words in extra_room

/// Number of groups per partition (TIK or TOK)
pub const GROUPS_PER_PARTITION: usize = 16;

/// Items per partition
pub const ITEMS_PER_PARTITION: usize = GROUPS_PER_PARTITION * 8; // 128

/// Which half of the batch we're operating on
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartitionHalf {
    /// TIK: first 16 groups (groups 0-15, items 0-127)
    Tik,
    /// TOK: second 16 groups (groups 16-31, items 128-255)
    Tok,
}

impl PartitionHalf {
    /// Get idx offset for this partition's buffer
    pub fn idx_offset(&self) -> i64 {
        match self {
            PartitionHalf::Tik => TIK_IDX_OFFSET,
            PartitionHalf::Tok => TOK_IDX_OFFSET,
        }
    }

    /// Get val offset for this partition's buffer
    pub fn val_offset(&self) -> i64 {
        match self {
            PartitionHalf::Tik => TIK_VAL_OFFSET,
            PartitionHalf::Tok => TOK_VAL_OFFSET,
        }
    }

    /// Get the starting group index for this partition
    pub fn start_group(&self) -> usize {
        match self {
            PartitionHalf::Tik => 0,
            PartitionHalf::Tok => GROUPS_PER_PARTITION,
        }
    }

    /// Get the ending group index (exclusive) for this partition
    pub fn end_group(&self) -> usize {
        match self {
            PartitionHalf::Tik => GROUPS_PER_PARTITION,
            PartitionHalf::Tok => GROUPS_PER_PARTITION * 2,
        }
    }
}

/// Partition state tracker
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartitionState {
    /// Reading from original all_idx/all_val (before partition is ready)
    Original,
    /// Reading from partition buffer (after partition is complete)
    Partitioned,
}

/// Which partition buffer to read from
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartitionBuffer {
    /// Use original all_idx/all_val (before partition is ready)
    Original,
    /// Read from TIK partition buffer
    Tik,
    /// Read from TOK partition buffer
    Tok,
}

impl PartitionBuffer {
    /// Get idx offset for this partition's buffer
    pub fn idx_offset(&self) -> i64 {
        match self {
            PartitionBuffer::Original => 0, // Not used, reading from all_idx
            PartitionBuffer::Tik => TIK_IDX_OFFSET,
            PartitionBuffer::Tok => TOK_IDX_OFFSET,
        }
    }

    /// Get val offset for this partition's buffer
    pub fn val_offset(&self) -> i64 {
        match self {
            PartitionBuffer::Original => 256, // Not used, reading from all_val
            PartitionBuffer::Tik => TIK_VAL_OFFSET,
            PartitionBuffer::Tok => TOK_VAL_OFFSET,
        }
    }

    /// Get the write target buffer (for double-buffering)
    /// When reading from Tik, write to Tok; when reading from Tok, write to Tik
    /// Original writes to Tik initially
    pub fn write_target(&self) -> PartitionBuffer {
        match self {
            PartitionBuffer::Original => PartitionBuffer::Tik,
            PartitionBuffer::Tik => PartitionBuffer::Tok,
            PartitionBuffer::Tok => PartitionBuffer::Tik,
        }
    }
}

/// Determine which buffer to read from at this round
///
/// Before partition is ready (rounds 0-3), read from Original (all_idx/all_val).
/// After partition is ready (rounds 4+), read from the appropriate partition buffer.
/// With k=1 (partition every round), we alternate between Tik and Tok.
pub fn get_read_buffer(round: usize, _partition_freq: usize) -> PartitionBuffer {
    if round < 4 {
        // Partition not ready yet - read from original vectors
        PartitionBuffer::Original
    } else {
        // After partition is ready, use Tik as the read buffer
        // (In a more complex scheme, we'd alternate based on round parity)
        PartitionBuffer::Tik
    }
}

/// Determine if we should emit partition stores at this round
///
/// We spread partition stores across rounds 0-3 to build up the sorted buffer.
/// After round 3, partition is complete and we can use sorted vload access.
pub fn should_partition(round: usize, _total_rounds: usize, _partition_freq: usize) -> bool {
    round < 4
}

/// Determines if partition is ready for use at this round
/// Partition stores happen during rounds 0-3, ready by round 4
pub fn is_partition_ready(round: usize) -> bool {
    round >= 4
}

/// Determines if we should be storing partition data at this round
/// We spread partition stores across rounds 0-3
pub fn should_store_partition(round: usize) -> bool {
    round < 4
}

/// Get the fraction of partition stores to emit at this round
/// Returns (start_fraction, end_fraction) where each is in [0.0, 1.0]
/// This spreads 256 stores across 4 rounds (64 stores per round per partition)
pub fn get_store_range(round: usize) -> (f32, f32) {
    match round {
        0 => (0.0, 0.25),
        1 => (0.25, 0.5),
        2 => (0.5, 0.75),
        3 => (0.75, 1.0),
        _ => (1.0, 1.0), // No stores needed
    }
}

/// Partition helper functions for the kernel builder
pub struct PartitionHelper<'a> {
    pub ctx: &'a ScratchContext,
    pub emit: &'a mut EmitContext,
    pub config: &'a KernelConfig,
}

impl<'a> PartitionHelper<'a> {
    /// Emit vload-based tree access for sorted items
    ///
    /// With k=1 (partition every round), only 1 bit is unsorted, so we use 2-way selection.
    /// With k=2, 4-way selection. With k=3, 8-way (high/low split).
    ///
    /// For each group:
    /// 1. Load first index to get cache base (min index in sorted group)
    /// 2. vload 8 tree values starting at cache base
    /// 3. Compute offset = idx - cache_base
    /// 4. Select from cache based on offset
    ///
    /// After partition with k=1, indices span at most 2, so 2-way select suffices!
    pub fn emit_sorted_vload_access(
        &mut self,
        _read_buffer: PartitionBuffer,
        n_iters: usize,
    ) {
        let ctx = self.ctx;
        let k = self.config.partition_frequency;

        for it in 0..n_iters {
            let base_g = it * 8; // UNROLL = 8

            for i in 0..8 {
                let g = base_g + i;
                let t = ctx.v_t[it][i];

                // Step 1: Get minimum index of this group
                // After stable partition, indices are sorted, so first lane has min
                // Store to scratch then load first element
                self.emit.emit_vstore(ctx.part_idx_scratch, ctx.all_idx[g]);
                self.emit.emit_load(ctx.tree_cache_base_s, ctx.part_idx_scratch);

                // Step 2: vload 8 tree values starting at min_idx
                self.emit.emit_alu(Opcode::Add, ctx.st[1], ctx.forest_p, ctx.tree_cache_base_s);
                self.emit.emit_vload(ctx.tree_cache_scalars, ctx.st[1]);

                // Step 3: Broadcast cache base for offset computation
                self.emit.emit_vbroadcast_force(ctx.v_cache_base, ctx.tree_cache_base_s);

                // Step 4: Compute offset = idx - cache_base
                self.emit.emit_valu(Opcode::Sub, t, ctx.all_idx[g], ctx.v_cache_base);

                // Step 5: Selection based on partition frequency
                if k == 1 {
                    // k=1: offsets are in [0,1], 2-way selection only!
                    // This is much simpler and more efficient.
                    //
                    // result = cache[0] + (cache[1] - cache[0]) * offset
                    // Since offset is 0 or 1, this gives cache[0] or cache[1]
                    self.emit_2way_select(ctx.v_nv[it][i], t);
                } else if k == 2 {
                    // k=2: offsets are in [0,3], 4-way selection only
                    self.emit_4way_select(ctx.v_nv[it][i], t, 0);
                } else {
                    // k=3+: offsets are in [0,7], need high/low split
                    // low_offset = offset & 3
                    self.emit.emit_valu(Opcode::And, ctx.v_part_temp[0], t, ctx.v_const[3]);

                    // low_result = 4-way-select(cache[0:4], low_offset)
                    self.emit_4way_select(ctx.v_part_temp[1], ctx.v_part_temp[0], 0);

                    // high_result = 4-way-select(cache[4:8], low_offset)
                    self.emit_4way_select(ctx.v_tree_temp, ctx.v_part_temp[0], 4);

                    // high_mask = (offset >= 4) = (offset & 4) != 0
                    self.emit.emit_valu(Opcode::And, ctx.v_part_temp[0], t, ctx.v_const[4]);

                    // result = vselect(high_mask, high_result, low_result)
                    // Implement as: result = low + (high - low) * (high_mask != 0)
                    self.emit.emit_valu(Opcode::Sub, ctx.v_nv[it][i], ctx.v_tree_temp, ctx.v_part_temp[1]);

                    // Convert high_mask to 0/1: high_bit = (offset >> 2) & 1
                    self.emit.emit_valu(Opcode::Shr, ctx.v_part_temp[0], t, ctx.v_const[2]);
                    self.emit.emit_valu(Opcode::And, ctx.v_part_temp[0], ctx.v_part_temp[0], ctx.v_1);

                    // result = low + (high - low) * high_bit
                    self.emit.emit_fma(ctx.v_nv[it][i], ctx.v_nv[it][i], ctx.v_part_temp[0], ctx.v_part_temp[1]);
                }
            }
        }

        // Stage 5 optimization: XOR with constant
        for it in 0..n_iters {
            for i in 0..8 {
                self.emit.emit_valu(
                    Opcode::Xor,
                    ctx.v_nv[it][i],
                    ctx.v_nv[it][i],
                    ctx.v_stage5_xor,
                );
            }
        }
    }

    /// Emit 2-way selection: result = cache[offset] where offset in [0,1]
    /// This is MUCH simpler than 4-way and only needs 4 VALU ops!
    ///
    /// result = cache[0] + (cache[1] - cache[0]) * offset
    fn emit_2way_select(&mut self, result: i64, offset: i64) {
        let ctx = self.ctx;

        // Broadcast cache[0] and cache[1]
        self.emit.emit_vbroadcast_force(ctx.v_part_temp[0], ctx.tree_cache_scalars);     // cache[0]
        self.emit.emit_vbroadcast_force(ctx.v_part_temp[1], ctx.tree_cache_scalars + 1); // cache[1]

        // result = cache[1] - cache[0]
        self.emit.emit_valu(Opcode::Sub, result, ctx.v_part_temp[1], ctx.v_part_temp[0]);

        // result = cache[0] + (cache[1] - cache[0]) * offset
        self.emit.emit_fma(result, result, offset, ctx.v_part_temp[0]);
    }

    /// Emit 4-way selection: result = cache[offset] where offset in [0,3]
    /// cache_base_offset is the starting index in tree_cache_scalars (0 for low, 4 for high)
    fn emit_4way_select(&mut self, result: i64, offset: i64, cache_base_offset: usize) {
        let ctx = self.ctx;

        // Initialize result to 0
        self.emit.emit_valu(Opcode::Mul, result, ctx.v_0, ctx.v_0);

        // For each of 4 possible values
        for ci in 0..4 {
            // Broadcast cache[base + ci] to temp vector
            self.emit.emit_vbroadcast_force(ctx.v_tree_temp, ctx.tree_cache_scalars + (cache_base_offset + ci) as i64);
            // mask = (offset == ci)
            self.emit.emit_valu(Opcode::Eq, ctx.v_part_temp[0], offset, ctx.v_const[ci]);
            // result += mask * cache_val
            self.emit.emit_fma(result, ctx.v_part_temp[0], ctx.v_tree_temp, result);
        }
    }

    /// Emit partition stores for one half (TIK or TOK)
    ///
    /// KEY INSIGHT: Use TWO COUNTERS (left_ptr, right_ptr) for O(1) addressing!
    /// - Items with branch_bit=0 go to left_ptr++
    /// - Items with branch_bit=1 go to right_ptr++
    /// - Relative order preserved (stable partition)
    /// - Stores overlap with OTHER partition's hash (tick-tock) = FREE!
    ///
    /// The tick-tock architecture allows:
    /// - TIK stores during TOK's hash rounds
    /// - TOK stores during TIK's hash rounds
    /// - Stores use idle slots during hash computation!
    ///
    /// Parameters:
    /// - `half`: Which partition (TIK or TOK) to store
    /// - `start_group`: First group to process (relative to half)
    /// - `end_group`: Last group to process (exclusive, relative to half)
    pub fn emit_partition_stores_for_half(
        &mut self,
        half: PartitionHalf,
        start_group: usize,
        end_group: usize,
    ) {
        let ctx = self.ctx;

        // Compute write base addresses in extra_room
        let idx_offset = half.idx_offset();
        let val_offset = half.val_offset();

        // Base pointers for this partition's buffer
        // IMPORTANT: Use st[12] and st[13] for base addresses to avoid conflicts
        // with st[0]-st[6] used as temporaries in emit_stable_partition_group!
        self.emit.emit_const(ctx.st[12], idx_offset);
        self.emit.emit_const(ctx.st[13], val_offset);
        self.emit.emit_alu(Opcode::Add, ctx.st[12], ctx.extra_room_p, ctx.st[12]); // idx_base in st[12]
        self.emit.emit_alu(Opcode::Add, ctx.st[13], ctx.extra_room_p, ctx.st[13]); // val_base in st[13]

        // Initialize counters based on how many items we've already stored
        // left_ptr starts at start_group * 8 (items before start_group)
        // right_ptr starts at ITEMS_PER_PARTITION/2 + start_group * 8
        let items_before = (start_group * VLEN) as i64;
        self.emit.emit_const(ctx.part_left_count, items_before);
        self.emit.emit_const(ctx.part_right_count, (ITEMS_PER_PARTITION / 2) as i64 + items_before);

        // Get the absolute group indices for this half
        let abs_start = half.start_group() + start_group;
        let abs_end = half.start_group() + end_group;

        // Process groups with counter-based partition stores
        for g in abs_start..abs_end {
            self.emit_stable_partition_group(
                g,
                ctx.part_left_count,   // left_ptr counter (in scratch)
                ctx.part_right_count,  // right_ptr counter (in scratch)
                ctx.st[12],            // idx_base (uses st[12] to avoid conflicts)
                ctx.st[13],            // val_base (uses st[13] to avoid conflicts)
            );
        }
    }

    /// Emit partition stores spread across multiple rounds
    /// Call this with the current round to emit the appropriate fraction of stores
    pub fn emit_partition_stores_for_round(
        &mut self,
        half: PartitionHalf,
        round: usize,
    ) {
        if !should_store_partition(round) {
            return;
        }

        let (start_frac, end_frac) = get_store_range(round);
        let start_group = (start_frac * GROUPS_PER_PARTITION as f32) as usize;
        let end_group = (end_frac * GROUPS_PER_PARTITION as f32) as usize;

        self.emit_partition_stores_for_half(half, start_group, end_group);
    }

    /// Emit partition stores for all groups to the specified buffer
    ///
    /// This is the main entry point for partition stores during rounds 0-3.
    /// Stores are spread across groups to enable tick-tock interleaving.
    pub fn emit_partition_stores(
        &mut self,
        _write_buffer: PartitionBuffer,
        n_groups: usize,
    ) {
        // For simplicity, store all groups for both halves
        // This uses the global counter approach with TIK/TOK split
        let half_groups = n_groups / 2;

        // Store TIK (first half: groups 0 to n_groups/2)
        self.emit_partition_stores_for_half(PartitionHalf::Tik, 0, GROUPS_PER_PARTITION.min(half_groups));

        // Store TOK (second half: groups n_groups/2 to n_groups)
        self.emit_partition_stores_for_half(PartitionHalf::Tok, 0, GROUPS_PER_PARTITION.min(half_groups));
    }

    /// Emit stable partition for a single group using COUNTER-BASED stores
    ///
    /// KEY INSIGHT: Use two counters (left_ptr, right_ptr) instead of computing
    /// scatter addresses! Items with bit=0 go to left_ptr++, bit=1 to right_ptr++.
    ///
    /// The stores overlap with the OTHER partition's hash computation (tick-tock)
    /// so they're essentially FREE. Run 2 rounds of TOK while TIK stores!
    ///
    /// Algorithm:
    /// 1. Store idx/val to scratch for scalar access (2 vstores)
    /// 2. For each item: check bit, store both idx and val, increment counters
    ///
    /// Cost: ~40 ops per group (2 vstore + 8*(load + AND + 2 store + 2 add))
    /// These stores overlap with computation = FREE on critical path!
    fn emit_stable_partition_group(
        &mut self,
        g: usize,
        left_ptr: i64,   // Scalar: current left write position
        right_ptr: i64,  // Scalar: current right write position
        idx_base: i64,   // Base address for idx partition buffer
        val_base: i64,   // Base address for val partition buffer
    ) {
        let ctx = self.ctx;

        // Step 1: Store all_idx and all_val to scratch for scalar access
        self.emit.emit_vstore(ctx.part_idx_scratch, ctx.all_idx[g]);
        self.emit.emit_vstore(ctx.tree_cache_scalars, ctx.all_val[g]);

        // Step 2: For each lane, check bit and store BOTH idx and val
        for i in 0..8 {
            // Load val[i] and compute branch bit
            self.emit.emit_load(ctx.st[0], ctx.tree_cache_scalars + i as i64); // val[i]
            self.emit.emit_alu(Opcode::And, ctx.st[1], ctx.st[0], ctx.c[1]);   // bit = val & 1

            // Load idx[i]
            self.emit.emit_load(ctx.st[2], ctx.part_idx_scratch + i as i64);   // idx[i]

            // Compute store address:
            // addr = left_ptr + bit * (right_ptr - left_ptr)
            self.emit.emit_alu(Opcode::Sub, ctx.st[3], right_ptr, left_ptr);   // right - left
            self.emit.emit_alu(Opcode::Mul, ctx.st[3], ctx.st[1], ctx.st[3]);  // bit * diff
            self.emit.emit_alu(Opcode::Add, ctx.st[3], left_ptr, ctx.st[3]);   // + left_ptr = ptr

            // Compute final addresses
            self.emit.emit_alu(Opcode::Add, ctx.st[4], idx_base, ctx.st[3]);   // idx_addr
            self.emit.emit_alu(Opcode::Add, ctx.st[5], val_base, ctx.st[3]);   // val_addr

            // Store idx and val to partition buffer
            self.emit.emit_store(ctx.st[4], ctx.st[2]);  // store idx
            self.emit.emit_store(ctx.st[5], ctx.st[0]);  // store val

            // Increment the appropriate counter
            // left_ptr += (1 - bit), right_ptr += bit
            self.emit.emit_alu(Opcode::Sub, ctx.st[6], ctx.c[1], ctx.st[1]);   // 1 - bit
            self.emit.emit_alu(Opcode::Add, left_ptr, left_ptr, ctx.st[6]);    // left_ptr += (1-bit)
            self.emit.emit_alu(Opcode::Add, right_ptr, right_ptr, ctx.st[1]);  // right_ptr += bit
        }
    }

    /// Emit load from partition buffer back to working vectors
    pub fn emit_partition_load(
        &mut self,
        read_buffer: PartitionBuffer,
        n_groups: usize,
    ) {
        if read_buffer == PartitionBuffer::Original {
            return; // Nothing to load, already in all_idx/all_val
        }

        let ctx = self.ctx;

        let idx_offset = read_buffer.idx_offset();
        let val_offset = read_buffer.val_offset();

        self.emit.emit_const(ctx.st[8], idx_offset);
        self.emit.emit_const(ctx.st[9], val_offset);
        self.emit.emit_alu(Opcode::Add, ctx.st[10], ctx.extra_room_p, ctx.st[8]);
        self.emit.emit_alu(Opcode::Add, ctx.st[11], ctx.extra_room_p, ctx.st[9]);

        let vlen = VLEN as i64;
        self.emit.emit_const(ctx.st[12], vlen);

        for g in 0..n_groups {
            // Load indices and values
            self.emit.emit_vload(ctx.all_idx[g], ctx.st[10]);
            self.emit.emit_vload(ctx.all_val[g], ctx.st[11]);

            // Increment addresses
            if g < n_groups - 1 {
                self.emit.emit_alu(Opcode::Add, ctx.st[10], ctx.st[10], ctx.st[12]);
                self.emit.emit_alu(Opcode::Add, ctx.st[11], ctx.st[11], ctx.st[12]);
            }
        }
    }
}
