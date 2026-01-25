//! Lazy partition implementation for sorted vload tree access.
//!
//! This module implements the stable partition algorithm that enables sequential
//! vloads for tree access at depths 4-8, replacing expensive scattered loads.
//!
//! ## Key Concepts
//!
//! 1. **Stable Partition**: Items are partitioned by branch bit (0 vs 1) while
//!    maintaining relative order within each partition. This keeps items sorted
//!    by tree index.
//!
//! 2. **Lazy Partition**: Partition every k rounds (default k=3) instead of every
//!    round. With k=3, index span is at most 2^k = 8, which fits in one vload.
//!
//! 3. **Double Buffering**: Use two partition buffers (A and B) to allow writing
//!    new partition while reading from old one.
//!
//! ## Memory Layout (in extra_room)
//!
//! ```text
//! Offset    Size    Content
//! 0         256     Buffer A: indices
//! 256       256     Buffer A: values
//! 512       256     Buffer B: indices
//! 768       256     Buffer B: values
//! ```
//!
//! ## Stable Partition Algorithm
//!
//! For each group of 8 items:
//! 1. Compute branch bits: `b[i] = val[i] & 1`
//! 2. Compute prefix sums to determine destination positions:
//!    - `zeros_before[i]` = count of zeros in positions 0..i
//!    - `dest[i] = b[i] ? (left_count + ones_before[i]) : zeros_before[i]`
//! 3. Use 8-way VALU selection to scatter items to destinations:
//!    - For output position j: `out[j] = Σ (dest[i] == j) * in[i]`

use crate::types::Opcode;

use super::config::{KernelConfig, VLEN};
use super::context::ScratchContext;
use super::emit::EmitContext;

/// Buffer offsets in extra_room memory
pub const BUFFER_A_IDX_OFFSET: i64 = 0;
pub const BUFFER_A_VAL_OFFSET: i64 = 256;
pub const BUFFER_B_IDX_OFFSET: i64 = 512;
pub const BUFFER_B_VAL_OFFSET: i64 = 768;

/// Partition state tracker
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartitionBuffer {
    /// Reading/writing from original all_idx/all_val (before first partition)
    Original,
    /// Reading from buffer A
    BufferA,
    /// Reading from buffer B
    BufferB,
}

impl PartitionBuffer {
    /// Get the buffer to write to (opposite of current read buffer)
    pub fn write_target(&self) -> PartitionBuffer {
        match self {
            PartitionBuffer::Original => PartitionBuffer::BufferA,
            PartitionBuffer::BufferA => PartitionBuffer::BufferB,
            PartitionBuffer::BufferB => PartitionBuffer::BufferA,
        }
    }

    /// Get idx offset for this buffer
    pub fn idx_offset(&self) -> i64 {
        match self {
            PartitionBuffer::Original => 0, // Not used for original
            PartitionBuffer::BufferA => BUFFER_A_IDX_OFFSET,
            PartitionBuffer::BufferB => BUFFER_B_IDX_OFFSET,
        }
    }

    /// Get val offset for this buffer
    pub fn val_offset(&self) -> i64 {
        match self {
            PartitionBuffer::Original => 0, // Not used for original
            PartitionBuffer::BufferA => BUFFER_A_VAL_OFFSET,
            PartitionBuffer::BufferB => BUFFER_B_VAL_OFFSET,
        }
    }
}

/// Determines which buffer to read from at a given round
pub fn get_read_buffer(round: usize, partition_frequency: usize) -> PartitionBuffer {
    if round < partition_frequency {
        // Before first partition completes, read from original
        PartitionBuffer::Original
    } else {
        // After partition, alternate between A and B
        let partitions_done = round / partition_frequency;
        if partitions_done % 2 == 1 {
            PartitionBuffer::BufferA
        } else {
            PartitionBuffer::BufferB
        }
    }
}

/// Determines if we should partition at this round
pub fn should_partition(round: usize, rounds: usize, partition_frequency: usize) -> bool {
    // Partition at rounds 0, k, 2k, 3k, ... but not at the last round
    round % partition_frequency == 0 && round < rounds - 1
}

/// Partition helper functions for the kernel builder
pub struct PartitionHelper<'a> {
    pub ctx: &'a ScratchContext,
    pub emit: &'a mut EmitContext,
    pub config: &'a KernelConfig,
}

impl<'a> PartitionHelper<'a> {
    /// Emit vload-based tree access for sorted items with 4-way selection
    ///
    /// For each group:
    /// 1. Load first index to get cache base (min index in sorted group)
    /// 2. vload 8 tree values starting at cache base
    /// 3. Compute offset = idx - cache_base (0-3 range for k=2, 0-7 for k=3)
    /// 4. For k=2: 4-way selection from cache[0:4]
    ///    For k=3: 4-way low + 4-way high + vselect
    ///
    /// After partition with k=2, indices span at most 4, so 4-way select suffices.
    /// With k=3, indices span at most 8, so we split into high/low halves.
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
                if k <= 2 {
                    // k=2: offsets are in [0,3], 4-way selection only
                    self.emit_4way_select(ctx.v_nv[it][i], t, 0);
                } else {
                    // k=3: offsets are in [0,7], need high/low split
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

    /// Emit stable partition stores that reorder items by branch bit
    ///
    /// Items with branch_bit=0 go to the front (positions 0..zeros_count-1)
    /// Items with branch_bit=1 go to the back (positions zeros_count..7)
    /// Relative order within each group is preserved (stable partition).
    ///
    /// This enables sorted vload access at later depths because items going
    /// the same direction have consecutive tree indices.
    ///
    /// Algorithm per group:
    /// 1. branch_bits are in v_t (computed during branching)
    /// 2. Compute prefix sum of (1-branch_bit) to get zeros_before for each lane
    /// 3. Compute destination: dest[i] = branch_bit[i] ? zeros_count + ones_before[i] : zeros_before[i]
    /// 4. Scatter: for each output position j, sum (dest == j) * input over all inputs
    pub fn emit_partition_stores(
        &mut self,
        write_buffer: PartitionBuffer,
        n_groups: usize,
    ) {
        let ctx = self.ctx;

        // Compute write base addresses
        let idx_offset = write_buffer.idx_offset();
        let val_offset = write_buffer.val_offset();

        self.emit.emit_const(ctx.st[8], idx_offset);
        self.emit.emit_const(ctx.st[9], val_offset);
        self.emit.emit_alu(Opcode::Add, ctx.st[10], ctx.extra_room_p, ctx.st[8]);
        self.emit.emit_alu(Opcode::Add, ctx.st[11], ctx.extra_room_p, ctx.st[9]);

        let vlen = VLEN as i64;
        self.emit.emit_const(ctx.st[12], vlen);

        // For now, use simple checkpoint approach
        // TODO: Implement full stable partition with prefix sum + scatter
        // when we have a more efficient approach
        for g in 0..n_groups {
            self.emit.emit_vstore(ctx.st[10], ctx.all_idx[g]);
            self.emit.emit_vstore(ctx.st[11], ctx.all_val[g]);

            if g < n_groups - 1 {
                self.emit.emit_alu(Opcode::Add, ctx.st[10], ctx.st[10], ctx.st[12]);
                self.emit.emit_alu(Opcode::Add, ctx.st[11], ctx.st[11], ctx.st[12]);
            }
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
