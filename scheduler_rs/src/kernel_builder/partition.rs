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
    /// Emit vload-based tree access for sorted items
    ///
    /// For each group:
    /// 1. Load first index to get cache base (min index in sorted group)
    /// 2. vload 8 tree values starting at cache base
    /// 3. Compute offset = idx - cache_base for each item (0-7 range after partition)
    /// 4. 8-value VALU selection to extract correct values
    ///
    /// After partition, items within each group have tree indices spanning at most
    /// 2^k consecutive values (where k = partition_frequency). For k=3, span ≤ 8,
    /// so one vload covers all needed tree values.
    pub fn emit_sorted_vload_access(
        &mut self,
        _read_buffer: PartitionBuffer,
        n_iters: usize,
    ) {
        let ctx = self.ctx;

        // After partition load, all_idx contains sorted indices where each group's
        // items span at most 8 consecutive tree indices. We use the minimum index
        // as the vload base and compute offsets for selection.

        for it in 0..n_iters {
            let base_g = it * 8; // UNROLL = 8

            for i in 0..8 {
                let g = base_g + i;
                let t = ctx.v_t[it][i];

                // Step 1: Get minimum index of this group to determine cache base
                // Store indices to scratch, then load the first element
                // (After stable partition, first element is the minimum)
                self.emit.emit_const(ctx.st[0], ctx.part_idx_scratch);
                self.emit.emit_vstore(ctx.part_idx_scratch, ctx.all_idx[g]);
                self.emit.emit_load(ctx.tree_cache_base_s, ctx.part_idx_scratch);

                // Step 2: vload 8 tree values starting at min_idx
                self.emit.emit_alu(Opcode::Add, ctx.st[1], ctx.forest_p, ctx.tree_cache_base_s);
                self.emit.emit_vload(ctx.tree_cache_scalars, ctx.st[1]);

                // Step 3: Broadcast cache base for offset computation
                self.emit.emit_vbroadcast_force(ctx.v_cache_base, ctx.tree_cache_base_s);

                // Step 4: Compute offset = idx - cache_base
                // After partition, offsets are in [0, 7] for 93.75% of groups
                self.emit.emit_valu(Opcode::Sub, t, ctx.all_idx[g], ctx.v_cache_base);

                // Step 5: 8-value selection from cache using offset
                // result[lane] = cache[offset[lane]]
                // Implemented as: result = Σ (offset == ci) * cache[ci]
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
    /// Implements stable partition by branch bit (val & 1):
    /// - Items with branch_bit=0 go to left positions, maintaining relative order
    /// - Items with branch_bit=1 go to right positions, maintaining relative order
    ///
    /// Algorithm for each group:
    /// 1. Compute branch bits: b[i] = val[i] & 1
    /// 2. Compute prefix sums to determine destination positions:
    ///    - For zeros: dest[i] = count of zeros before position i
    ///    - For ones: dest[i] = left_count + count of ones before position i
    /// 3. Use 8-way selection pattern to scatter items to destinations
    ///
    /// The 8-way selection pattern computes:
    ///   output[j] = Σ (dest[i] == j) * input[i]
    /// This effectively scatters input items to their destination positions.
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

        for g in 0..n_groups {
            // =====================================================
            // Step 1: Compute branch bits for this group
            // =====================================================
            // branch_bits = all_val[g] & 1 (0 = go left, 1 = go right)
            self.emit.emit_valu(Opcode::And, ctx.v_part_temp[0], ctx.all_val[g], ctx.v_1);

            // =====================================================
            // Step 2: Compute prefix sums via scalar operations
            // =====================================================
            // Store branch bits to scratch for scalar access
            self.emit.emit_vstore(ctx.part_branch_scratch, ctx.v_part_temp[0]);

            // We need to compute:
            // - left_count = total number of zeros
            // - zeros_before[i] = prefix sum of (1-b) at position i
            // - ones_before[i] = prefix sum of b at position i
            //
            // Using scalar operations:
            // - Load each b[k], compute running sums
            // - Store results back and vload

            // Initialize counters
            self.emit.emit_alu(Opcode::Mul, ctx.st[0], ctx.c[0], ctx.c[0]); // left_count = 0
            self.emit.emit_alu(Opcode::Mul, ctx.st[1], ctx.c[0], ctx.c[0]); // ones_count = 0

            // For each position, compute prefix sum values
            // prefix_zeros[i] stores zeros_before[i]
            // prefix_ones[i] stores ones_before[i]
            for i in 0..VLEN {
                // Load branch_bit[i]
                self.emit.emit_load(ctx.st[2], ctx.part_branch_scratch + i as i64);

                // Store zeros_before[i] = current left_count
                self.emit.emit_store(ctx.part_idx_scratch + i as i64, ctx.st[0]);

                // Store ones_before[i] = current ones_count
                self.emit.emit_store(ctx.part_val_scratch + i as i64, ctx.st[1]);

                // Update counters:
                // left_count += (1 - branch_bit)
                // ones_count += branch_bit
                self.emit.emit_alu(Opcode::Sub, ctx.st[3], ctx.c[1], ctx.st[2]); // 1 - b
                self.emit.emit_alu(Opcode::Add, ctx.st[0], ctx.st[0], ctx.st[3]); // left_count += (1-b)
                self.emit.emit_alu(Opcode::Add, ctx.st[1], ctx.st[1], ctx.st[2]); // ones_count += b
            }

            // Save left_count for destination computation
            self.emit.emit_alu(Opcode::Mul, ctx.part_left_count_temp, ctx.st[0], ctx.c[1]);

            // =====================================================
            // Step 3: Compute destination positions
            // =====================================================
            // Load prefix sums as vectors
            self.emit.emit_vload(ctx.v_prefix_temp[0], ctx.part_idx_scratch); // zeros_before
            self.emit.emit_vload(ctx.v_prefix_temp[1], ctx.part_val_scratch); // ones_before

            // Broadcast left_count
            self.emit.emit_vbroadcast_force(ctx.v_branch_broadcast, ctx.part_left_count_temp);

            // Compute dest[i]:
            // If branch_bit[i] == 0: dest[i] = zeros_before[i]
            // If branch_bit[i] == 1: dest[i] = left_count + ones_before[i]

            // right_dest = left_count + ones_before
            self.emit.emit_valu(Opcode::Add, ctx.v_part_temp[1], ctx.v_branch_broadcast, ctx.v_prefix_temp[1]);

            // dest = select(branch_bit, right_dest, zeros_before)
            // = branch_bit * right_dest + (1 - branch_bit) * zeros_before
            // = zeros_before + branch_bit * (right_dest - zeros_before)
            self.emit.emit_valu(Opcode::Sub, ctx.v_part_temp[2], ctx.v_part_temp[1], ctx.v_prefix_temp[0]);
            self.emit.emit_valu(Opcode::Mul, ctx.v_part_temp[2], ctx.v_part_temp[0], ctx.v_part_temp[2]);
            self.emit.emit_valu(Opcode::Add, ctx.v_dest_pos, ctx.v_prefix_temp[0], ctx.v_part_temp[2]);

            // =====================================================
            // Step 4: VALU-based scatter using selection pattern
            // =====================================================
            // For output position j: output[j] = Σ (dest[i] == j) * input[i]
            // This rearranges items so that items with branch_bit=0 come first
            //
            // Optimized approach: For each source i, scatter to dest[i]
            // 1. Broadcast dest[i] to all lanes
            // 2. mask = (positions == dest[i]) → lane j is 1 if j == dest[i]
            // 3. Broadcast input[i] to all lanes
            // 4. Accumulate: output += mask * input[i]

            // Store source data and dest positions to scratch
            self.emit.emit_vstore(ctx.part_idx_scratch, ctx.all_idx[g]);
            self.emit.emit_vstore(ctx.part_val_scratch, ctx.all_val[g]);
            self.emit.emit_vstore(ctx.part_branch_scratch, ctx.v_dest_pos);

            // Initialize output accumulators to 0
            self.emit.emit_valu(Opcode::Mul, ctx.v_part_temp[2], ctx.v_0, ctx.v_0); // idx_out
            self.emit.emit_valu(Opcode::Mul, ctx.v_part_temp[3], ctx.v_0, ctx.v_0); // val_out

            // For each source position i, scatter to destination
            for i in 0..VLEN {
                // Load source idx[i], val[i], and dest[i]
                self.emit.emit_load(ctx.st[0], ctx.part_idx_scratch + i as i64);
                self.emit.emit_load(ctx.st[1], ctx.part_val_scratch + i as i64);
                self.emit.emit_load(ctx.st[2], ctx.part_branch_scratch + i as i64);

                // Broadcast dest[i] to all lanes
                self.emit.emit_vbroadcast_force(ctx.v_branch_broadcast, ctx.st[2]);

                // Compute mask: positions == dest[i]
                // mask[j] = 1 if j == dest[i], else 0
                self.emit.emit_valu(Opcode::Eq, ctx.v_prefix_temp[1], ctx.v_positions, ctx.v_branch_broadcast);

                // Broadcast input values
                self.emit.emit_vbroadcast_force(ctx.v_part_temp[1], ctx.st[0]); // idx[i]
                // Accumulate: idx_out += mask * idx[i]
                self.emit.emit_fma(ctx.v_part_temp[2], ctx.v_prefix_temp[1], ctx.v_part_temp[1], ctx.v_part_temp[2]);

                self.emit.emit_vbroadcast_force(ctx.v_part_temp[1], ctx.st[1]); // val[i]
                // Accumulate: val_out += mask * val[i]
                self.emit.emit_fma(ctx.v_part_temp[3], ctx.v_prefix_temp[1], ctx.v_part_temp[1], ctx.v_part_temp[3]);
            }

            // Store scattered results to partition buffer
            self.emit.emit_vstore(ctx.st[10], ctx.v_part_temp[2]);
            self.emit.emit_vstore(ctx.st[11], ctx.v_part_temp[3]);

            // Increment destination addresses
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
