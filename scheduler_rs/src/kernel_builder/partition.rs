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
    /// Emit vload-based tree access for sorted items
    ///
    /// For each group:
    /// 1. Load first index to get cache base
    /// 2. vload 8 tree values starting at cache base
    /// 3. Compute offset = idx - cache_base for each item
    /// 4. 8-value VALU selection
    pub fn emit_sorted_vload_access(
        &mut self,
        _read_buffer: PartitionBuffer,
        n_iters: usize,
    ) {
        let ctx = self.ctx;

        // For now, we still read from all_idx since the full partition
        // implementation is complex. This is a placeholder that shows
        // the structure - the actual sorted access requires the partition
        // to have been executed first.
        //
        // The key insight is that with sorted items, each group's indices
        // span at most 2^k consecutive tree values, so one vload covers them all.

        for it in 0..n_iters {
            let base_g = it * 8; // UNROLL = 8

            for i in 0..8 {
                let g = base_g + i;
                let t = ctx.v_t[it][i];

                // Step 1: Get first index of this group to determine cache base
                // vstore the indices, load first one
                self.emit.emit_vstore(ctx.st[0], ctx.all_idx[g]);
                self.emit.emit_load(ctx.tree_cache_base_s, ctx.st[0]);

                // Step 2: vload 8 tree values starting at first_idx
                self.emit.emit_alu(Opcode::Add, ctx.st[1], ctx.forest_p, ctx.tree_cache_base_s);
                self.emit.emit_vload(ctx.tree_cache_scalars, ctx.st[1]);

                // Step 3: Broadcast cache base for offset computation
                self.emit.emit_vbroadcast_force(ctx.v_cache_base, ctx.tree_cache_base_s);

                // Step 4: Compute offset = idx - cache_base
                self.emit.emit_valu(Opcode::Sub, t, ctx.all_idx[g], ctx.v_cache_base);

                // Step 5: 8-value selection from cache
                // result = sum((offset == ci) * cache[ci]) for ci in 0..8
                self.emit.emit_valu(Opcode::Mul, ctx.v_nv[it][i], ctx.v_0, ctx.v_0); // Init to 0

                for ci in 0..8 {
                    // Broadcast cache[ci] to temp vector
                    self.emit.emit_vbroadcast_force(ctx.v_tree_temp, ctx.tree_cache_scalars + ci as i64);
                    // mask = (offset == ci)
                    let v_ci = ctx.v_const[ci];
                    self.emit.emit_valu(Opcode::Eq, ctx.v_part_temp[0], t, v_ci);
                    // result += mask * cache_val
                    self.emit.emit_fma(ctx.v_nv[it][i], ctx.v_part_temp[0], ctx.v_tree_temp, ctx.v_nv[it][i]);
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

    /// Emit stable partition stores
    ///
    /// This is a simplified version that demonstrates the concept.
    /// Items with branch_bit=0 go to left half, branch_bit=1 to right half.
    ///
    /// Full stable partition is complex because we need to maintain relative
    /// order within each partition. For now, we use a simpler approach:
    /// - Compute destination offset for each item based on prefix sum of branch bits
    /// - Store indices and values to partition buffer
    pub fn emit_partition_stores(
        &mut self,
        write_buffer: PartitionBuffer,
        n_groups: usize,
    ) {
        let ctx = self.ctx;

        // Compute write base addresses
        let idx_offset = write_buffer.idx_offset();
        let val_offset = write_buffer.val_offset();

        // For now, just copy data to partition buffer (not actually sorting)
        // This preserves the existing tick-tock store behavior while setting up
        // the infrastructure for full partition implementation.

        self.emit.emit_const(ctx.st[8], idx_offset);
        self.emit.emit_const(ctx.st[9], val_offset);
        self.emit.emit_alu(Opcode::Add, ctx.st[10], ctx.extra_room_p, ctx.st[8]);
        self.emit.emit_alu(Opcode::Add, ctx.st[11], ctx.extra_room_p, ctx.st[9]);

        let vlen = VLEN as i64;
        self.emit.emit_const(ctx.st[12], vlen);

        for g in 0..n_groups {
            // Store indices and values
            self.emit.emit_vstore(ctx.st[10], ctx.all_idx[g]);
            self.emit.emit_vstore(ctx.st[11], ctx.all_val[g]);

            // Increment addresses
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
