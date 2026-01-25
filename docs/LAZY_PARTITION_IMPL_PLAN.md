# Lazy Partition Implementation Plan

## Goal

Implement lazy partitioning (every 3 rounds) to enable sequential vloads for tree access, targeting ~560 cycle savings (from ~1,760 to ~1,200 cycles, achieving 120x+ speedup).

## Executive Summary

**Current bottleneck**: Scattered loads for tree access at depths 4+ cost ~128 cycles/round (8 loads × 32 groups ÷ 2 load slots).

**Solution**: Keep items sorted by tree index via stable partition. Sorted items in a group access contiguous tree indices → one vload replaces 8 scattered loads.

**Key insight**: Partition every 3 rounds (not every round) balances partition overhead vs. vload benefits.

## Mathematical Foundation

### Index Span After k Rounds Without Partition

If items were sorted k rounds ago, their current tree indices have span:
```
new_span = ceil(2^{N-k-5}) × 2^k
```

For `new_span ≤ 8` (one vload covers group):

| k | Depths Covered | Boundary Groups |
|---|----------------|-----------------|
| 2 | 4, 5, 6, 7, 8  | ~6 (18.75%)     |
| 3 | 4, 5, 6, 7, 8  | ~14 (43.75%)    |

**Winner**: k=2 (partition every 3 rounds) covers depths 4-8 with 81% "good" groups.

### Cost Analysis

**Partition cost per event** (with index packing):
- 128 packed index stores + 256 value stores = 384 stores
- At 2 stores/cycle = 192 cycles raw
- **But**: Stores overlap with hash computation (2 rounds × 2 slots × ~100 cycles = 400 slots)
- **Net critical path**: ~40 cycles (reload + unpack)

**Savings per round** (depths 4-8):
- Scattered: 128 cycles (8 × 32 ÷ 2)
- vload + select: ~10 cycles (1 vload + 8-value VALU select per group)
- **Savings**: ~118 cycles/round

**Break-even**:
- 5 rounds benefit (depths 4-8) × 118 = 590 cycles saved
- Partition every 3 rounds = ~5 partition events × 40 = 200 cycles cost
- **Net savings: ~390 cycles per 16 rounds**

## Architecture

### Memory Layout

```
Scratch Memory (1536 words):
┌─────────────────────────────────────────────────────────┐
│ Existing allocations (indices, values, temps, etc.)     │
│ ~1200 words                                             │
├─────────────────────────────────────────────────────────┤
│ Buffer A: Sorted indices (128 packed words)             │  NEW
│ Buffer A: Sorted values (256 words)                     │  NEW
├─────────────────────────────────────────────────────────┤
│ Buffer B: Sorted indices (128 packed words)             │  NEW
│ Buffer B: Sorted values (256 words)                     │  NEW
└─────────────────────────────────────────────────────────┘
```

**Total new allocation**: 128 + 256 + 128 + 256 = 768 words

**Problem**: Current scratch usage is ~1531/1536. We need to restructure.

**Solution**: Use `extra_room` (external memory) for partition buffers since:
1. We write sequentially (good for cache)
2. Reads are via vload (8 words at once)
3. External memory has same latency as scratch for vload

### Double-Buffer Scheme

```
Round 0-2:  Read from original (unpacked), write partition to Buffer A
Round 3-5:  Read from Buffer A, write partition to Buffer B
Round 6-8:  Read from Buffer B, write partition to Buffer A
Round 9-11: Read from Buffer A, write partition to Buffer B
Round 12-14: Read from Buffer B, write partition to Buffer A
Round 15:   Read from Buffer A, final output
```

### Index Packing Format

```
32-bit word: [unused:10][idx_odd:11][idx_even:11]

Pack (2 indices → 1 word):
  packed = (idx_odd << 11) | idx_even

Unpack (1 word → 2 indices):
  idx_even = packed & 0x7FF
  idx_odd = (packed >> 11) & 0x7FF
```

## Implementation Phases

### Phase 1: Partition Infrastructure (Rust)

**File**: `scheduler_rs/src/kernel_builder/partition.rs` (new)

```rust
/// Partition buffer state
pub struct PartitionBuffers {
    /// Base addresses for double buffers (in extra_room)
    pub buffer_a_idx: i64,   // Packed indices
    pub buffer_a_val: i64,   // Values
    pub buffer_b_idx: i64,
    pub buffer_b_val: i64,

    /// Partition counters (scalar scratch)
    pub left_count: i64,     // Items with branch bit = 0
    pub right_count: i64,    // Items with branch bit = 1

    /// Constants for packing
    pub mask_11bit: i64,     // 0x7FF
    pub shift_11: i64,       // 11
}

impl PartitionBuffers {
    pub fn allocate(alloc: &mut ScratchAllocator, extra_room_base: i64) -> Self {
        // Allocate in extra_room (external memory)
        let buffer_a_idx = extra_room_base;
        let buffer_a_val = extra_room_base + 128;  // After packed indices
        let buffer_b_idx = extra_room_base + 128 + 256;
        let buffer_b_val = extra_room_base + 128 + 256 + 128;

        Self {
            buffer_a_idx,
            buffer_a_val,
            buffer_b_idx,
            buffer_b_val,
            left_count: alloc.alloc("part_left_count"),
            right_count: alloc.alloc("part_right_count"),
            mask_11bit: alloc.alloc("mask_11bit"),
            shift_11: alloc.alloc("shift_11"),
        }
    }
}
```

### Phase 2: Partition Emission

**Add to**: `scheduler_rs/src/kernel_builder/emit.rs`

```rust
impl EmitContext {
    /// Emit stable partition by branch bit
    ///
    /// Items with branch_bit=0 go to left half (maintain relative order)
    /// Items with branch_bit=1 go to right half (maintain relative order)
    pub fn emit_partition(
        &mut self,
        src_idx_base: i64,    // Source indices (packed or unpacked)
        src_val_base: i64,    // Source values
        dst_idx_base: i64,    // Destination indices (packed)
        dst_val_base: i64,    // Destination values
        branch_bits: &[i64],  // Vector registers holding branch bits per group
        is_src_packed: bool,
    ) {
        // Strategy: Two-pass stable partition
        // Pass 1: Count items going left (branch_bit = 0)
        // Pass 2: Write items to appropriate positions

        // For SIMD, we use prefix sum within each vector:
        // left_offset[i] = sum(branch_bit[0..i] == 0)
        // right_offset[i] = sum(branch_bit[0..i] == 1)

        // This requires careful ordering to maintain stability
        // ...
    }

    /// Emit sequential vload tree access for sorted items
    pub fn emit_vload_tree_access(
        &mut self,
        sorted_idx_base: i64,  // Packed sorted indices
        tree_base: i64,        // Forest pointer
        v_nv: &[i64],          // Output: tree values per group
        depth: usize,
    ) {
        // For each group:
        // 1. Unpack first index (gives cache base)
        // 2. vload tree[cache_base : cache_base+8]
        // 3. Compute offset = idx - cache_base for each item
        // 4. 8-value VALU selection

        // ...
    }
}
```

### Phase 3: Integration with Tick-Tock Builder

**Modify**: `scheduler_rs/src/kernel_builder/tick_tock.rs`

```rust
impl TickTockKernelBuilder {
    pub fn build(&mut self) -> Vec<Op> {
        // ... existing init ...

        // NEW: Allocate partition buffers
        let partition = if self.config.enable_partition {
            Some(PartitionBuffers::allocate(&mut self.alloc, extra_room_base))
        } else {
            None
        };

        // Emit partition constants
        if let Some(ref part) = partition {
            self.emit_partition_init(part);
        }

        for rnd in 0..self.config.rounds {
            let depth = rnd % (self.config.forest_height + 1);

            // Determine read source based on round
            let (read_idx, read_val, is_packed) = self.get_read_buffers(rnd, &partition);

            // PHASE 1: Tree access
            if depth <= 3 {
                // Use existing VALU selection (unchanged)
                self.emit_valu_tree_access(depth, rnd);
            } else if depth <= 8 && partition.is_some() {
                // NEW: Use vload from sorted buffer
                self.emit_vload_tree_access(read_idx, read_val, depth, rnd);
            } else {
                // Fallback: scattered loads for depth 9-10 or no partition
                self.emit_scattered_tree_access(depth, rnd);
            }

            // PHASE 2: XOR + Hash (unchanged)
            self.emit_xor_phase(rnd);
            self.emit_hash_phase(rnd);

            // PHASE 3: Branch computation (unchanged)
            self.emit_branch_phase(rnd);

            // PHASE 4: Partition stores (if this is a partition round)
            if self.should_partition(rnd) {
                let (write_idx, write_val) = self.get_write_buffers(rnd, &partition);
                self.emit_partition_stores(write_idx, write_val, rnd);
            }
        }

        // ... existing finalization ...
    }

    fn should_partition(&self, rnd: usize) -> bool {
        // Partition at rounds: 0, 3, 6, 9, 12 (every 3 rounds)
        // Don't partition at round 15 (last round)
        self.config.enable_partition && rnd % 3 == 0 && rnd < 15
    }
}
```

### Phase 4: vload Tree Access Implementation

The core algorithm for sorted tree access:

```rust
fn emit_vload_tree_access(&mut self, sorted_idx: i64, depth: usize, rnd: usize) {
    let n_groups = self.config.batch_size / VLEN;

    for g in 0..n_groups {
        // Step 1: Get first index of group (for cache base)
        // vstore group indices, load first one
        let group_idx_base = sorted_idx + (g * VLEN / 2);  // Packed: 2 indices per word

        // Unpack and get first index
        self.emit(load, vload, temp_packed, group_idx_base);
        self.emit(alu, "&", first_idx, temp_packed[0], mask_11bit);  // Extract idx_even

        // Step 2: vload 8 tree values starting at first_idx
        self.emit(alu, "+", tree_addr, forest_p, first_idx);
        self.emit(load, vload, cache, tree_addr);

        // Step 3: Broadcast cache base for offset computation
        self.emit(valu, vbroadcast, v_cache_base, first_idx);

        // Step 4: Unpack all 8 indices for this group
        // (4 packed words → 8 indices)
        for i in 0..4 {
            let packed_word = temp_packed[i];
            self.emit(alu, "&", idx[2*i], packed_word, mask_11bit);
            self.emit(alu, ">>", temp, packed_word, shift_11);
            self.emit(alu, "&", idx[2*i+1], temp, mask_11bit);
        }

        // Step 5: Broadcast indices to vector
        for i in 0..8 {
            self.emit(valu, vbroadcast, v_idx_lane[i], idx[i]);
        }
        // Interleave to form actual vector
        // (This is tricky - need to transpose 8 scalars into vector positions)

        // Step 6: Compute offset = idx - cache_base
        self.emit(valu, "-", v_offset, v_idx, v_cache_base);

        // Step 7: 8-value selection from cache
        self.emit(valu, "*", v_result, v_0, v_0);  // Init to 0
        for ci in 0..8 {
            self.emit(valu, vbroadcast, v_cache_val, cache + ci);
            self.emit(valu, "==", v_mask, v_offset, v_const[ci]);
            self.emit(valu, multiply_add, v_result, v_mask, v_cache_val, v_result);
        }

        // Result in v_result = tree value for each item in group
        self.emit(valu, "^", v_nv[g], v_result, v_stage5_xor);
    }
}
```

### Phase 5: Partition Store Implementation

Stable partition using counting sort approach:

```rust
fn emit_partition_stores(&mut self, dst_idx: i64, dst_val: i64, rnd: usize) {
    let n_groups = self.config.batch_size / VLEN;

    // Initialize counters
    self.emit(load, const, left_count, 0);
    self.emit(load, const, right_count, 128);  // Right half starts at 128

    // For each group, compute destinations and store
    for g in 0..n_groups {
        // Branch bit is in LSB of value after hash
        // branch_bit = val & 1

        // Compute prefix sum of branch bits within vector
        // This gives us the offset within left/right partition

        // For stability, process items in order:
        for lane in 0..VLEN {
            // if branch_bit[lane] == 0:
            //   store idx[lane] at left_count, increment left_count
            // else:
            //   store idx[lane] at right_count, increment right_count

            // This requires scalar processing which is slow.
            // Better: batch stores and use vstore where possible
        }
    }

    // Optimization: Group stores by destination
    // All items going left can be vstored together (if contiguous in dest)
    // This requires more complex bookkeeping but saves cycles
}
```

## Challenges and Solutions

### Challenge 1: Scalar Partition Stores

**Problem**: Stable partition requires per-item decisions, but we have limited scalar store bandwidth.

**Solution**: Batch writes using histogramming:
1. Count items going left per group (8 ALU ops)
2. Compute prefix sums across groups (serial, ~32 ALU ops)
3. Use computed offsets for batch vstore when items are contiguous

### Challenge 2: Index Unpacking Overhead

**Problem**: Unpacking 11-bit indices adds ALU operations.

**Solution**:
- Pack/unpack overlaps with other operations (ALU slots usually available)
- Only unpack when needed (once per group, not per item)
- Consider unpacking during partition write (pack during store, unpack during load)

### Challenge 3: Boundary Groups

**Problem**: ~18% of groups span partition boundaries (index span > 8).

**Solution**:
- Detect at runtime: `span = max_idx - min_idx`
- If span > 7: use 2 vloads (covers span up to 15)
- Very rare (< 1%): fallback to scattered loads

### Challenge 4: Scratch Space

**Problem**: Need ~768 words for double buffers, but scratch is nearly full.

**Solution**: Use `extra_room` (external memory pointer passed at address 7):
- Same vload latency as scratch
- Sequential access pattern is cache-friendly
- Already allocated but unused in current kernel

## Testing Strategy

### Unit Tests

1. **Partition correctness**: Verify items maintain relative order within each partition
2. **Index packing**: Verify pack/unpack round-trips correctly
3. **vload selection**: Verify 8-value selection produces correct tree values

### Integration Tests

1. **Full kernel**: Compare output against reference_kernel2()
2. **Cycle count**: Verify expected speedup (~1,200 cycles target)
3. **Edge cases**: Test with various batch sizes, tree heights

### Benchmarks

```bash
# Test partition-enabled kernel
python test_optimized.py --rust-builder --enable-partition

# Compare cycles
python -c "
import scheduler_rs

# Without partition
cycles_no_part = scheduler_rs.build_tick_tock_kernel_ops(10, 2047, 256, 16,
    enable_partition=False)

# With partition
cycles_with_part = scheduler_rs.build_tick_tock_kernel_ops(10, 2047, 256, 16,
    enable_partition=True)

print(f'Without partition: {cycles_no_part} cycles')
print(f'With partition: {cycles_with_part} cycles')
print(f'Savings: {cycles_no_part - cycles_with_part} cycles')
"
```

## Implementation Order

1. **Week 1**: Partition infrastructure
   - [ ] Add `PartitionBuffers` struct
   - [ ] Add config flag `enable_partition`
   - [ ] Allocate buffers in extra_room
   - [ ] Add pack/unpack helper ops

2. **Week 2**: vload tree access
   - [ ] Implement `emit_vload_tree_access`
   - [ ] Add 8-value VALU selection
   - [ ] Handle boundary groups (span > 8)
   - [ ] Unit tests for tree access

3. **Week 3**: Partition stores
   - [ ] Implement counting-based stable partition
   - [ ] Optimize store batching
   - [ ] Double-buffer switching logic
   - [ ] Unit tests for partition

4. **Week 4**: Integration and optimization
   - [ ] Wire into tick_tock builder
   - [ ] Full kernel testing
   - [ ] Profile and optimize hot paths
   - [ ] Final benchmarking

## Expected Results

| Metric | Current | With Partition | Improvement |
|--------|---------|----------------|-------------|
| Cycles | 1,759 | ~1,200 | -559 (32%) |
| Speedup | 84x | ~123x | +39x |
| Target gap | 272 cycles | -287 cycles | Exceeds target! |

## Risks

1. **Scratch overflow**: Mitigated by using extra_room
2. **Partition overhead > savings**: Mitigated by lazy partition (k=2)
3. **Implementation complexity**: Mitigated by phased approach
4. **Scheduler not finding parallelism**: Mitigated by interleaved emission pattern

## Open Questions

1. Should we use k=3 (every 4 rounds) for even lower partition cost?
2. Can we avoid index packing if extra_room has enough space?
3. Should boundary groups use 2 vloads or scattered loads?
4. Can partition stores fully overlap with hash computation?
