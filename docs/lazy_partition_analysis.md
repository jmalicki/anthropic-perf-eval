# Lazy Partition Analysis: When Partition-by-N-k Works

## Key Insight

If items are sorted by round (N-k)'s tree indices, they maintain **useful locality**
at round N because tree index evolution is predictable:

```
idx_N = 2^k * idx_{N-k} + (k-bit value from branch bits)
```

Items consecutive by `idx_{N-k}` will have `idx_N` differing by at most `2^k`.

## Mathematical Model

### Variables
- **N**: Current round (tree depth = N mod 11)
- **k**: Rounds since last partition (k=0 means partitioned this round)
- **Group size**: 8 items (VLEN)
- **Batch size**: 256 items

### Items Per Tree Index

At round (N-k), there are `2^{N-k}` possible tree indices:
```
items_per_index = 256 / 2^{N-k} = 2^{8-N+k}
```

### Old Indices Spanned by Group of 8

A group of 8 consecutive sorted items spans:
```
old_span = ceil(8 / items_per_index) = ceil(8 / 2^{8-N+k}) = ceil(2^{N-k-5})
```

### New Index Span After k Rounds

Each old index "spreads" to `2^k` consecutive new indices:
```
new_span = old_span * 2^k = ceil(2^{N-k-5}) * 2^k
```

### Condition for One vload to Cover Group

For a single vload (8 consecutive tree values) to cover a group:
```
new_span ≤ 8
```

## Results by Partition Frequency

### k=0: Partition Every Round
| Round N | Depth | old_span | new_span | vload works? |
|---------|-------|----------|----------|--------------|
| 4 | 4 | 1 | 1 | ✅ |
| 5 | 5 | 1 | 1 | ✅ |
| 6 | 6 | 2 | 2 | ✅ |
| 7 | 7 | 4 | 4 | ✅ |
| 8 | 8 | 8 | 8 | ✅ |
| 9 | 9 | 16 | 16 | ❌ |
| 10 | 10 | 32 | 32 | ❌ |

### k=1: Partition Every 2 Rounds
| Round N | Depth | old_span | new_span | vload works? |
|---------|-------|----------|----------|--------------|
| 4 | 4 | 1 | 2 | ✅ |
| 5 | 5 | 1 | 2 | ✅ |
| 6 | 6 | 1 | 2 | ✅ |
| 7 | 7 | 2 | 4 | ✅ |
| 8 | 8 | 4 | 8 | ✅ |
| 9 | 9 | 8 | 16 | ❌ |
| 10 | 10 | 16 | 32 | ❌ |

### k=2: Partition Every 3 Rounds
| Round N | Depth | old_span | new_span | vload works? |
|---------|-------|----------|----------|--------------|
| 4 | 4 | 1 | 4 | ✅ |
| 5 | 5 | 1 | 4 | ✅ |
| 6 | 6 | 1 | 4 | ✅ |
| 7 | 7 | 1 | 4 | ✅ |
| 8 | 8 | 2 | 8 | ✅ |
| 9 | 9 | 4 | 16 | ❌ |
| 10 | 10 | 8 | 32 | ❌ |

### k=3: Partition Every 4 Rounds
| Round N | Depth | old_span | new_span | vload works? |
|---------|-------|----------|----------|--------------|
| 4 | 4 | 1 | 8 | ✅ |
| 5 | 5 | 1 | 8 | ✅ |
| 6 | 6 | 1 | 8 | ✅ |
| 7 | 7 | 1 | 8 | ✅ |
| 8 | 8 | 1 | 8 | ✅ |
| 9 | 9 | 2 | 16 | ❌ |
| 10 | 10 | 4 | 32 | ❌ |

## Full Probability Distribution Analysis

### Setup
- 256 items partitioned into two halves by branch bit
- Each half: 128 items, sorted by old tree index
- 32 groups of 8 consecutive items

### Group Types After Stable Partition

| Group Type | Count | Description |
|------------|-------|-------------|
| Pure-0 | ~15 | All 8 items in first half (branch=0) |
| Pure-1 | ~15 | All 8 items in second half (branch=1) |
| Boundary | ~2 | Spans the half boundary |

### Span Distribution by Group Type

**Pure groups (30 of 32):**
- All items have same branch bit
- new_idx = 2 * old_idx + b (same b for all)
- Items have consecutive old indices
- Span = 2 × (items in group spanning different old indices)

**Boundary groups (2 of 32):**
- Mix of branch bits (some 0, some 1)
- new_idx alternates between even (2k) and odd (2k+1)
- Span = max(evens) - min(evens) + max(odds) - min(odds) + 1
- **Worst case: span up to 2× normal**

### Probability Tables by Depth

**Depth 4 (16 indices, 16 items/index):**
| Span | Probability | Groups |
|------|-------------|--------|
| 1 | 93.75% | 30 |
| 2 | 6.25% | 2 (boundary) |

**Depth 5 (32 indices, 8 items/index):**
| Span | Probability | Groups |
|------|-------------|--------|
| 1-2 | 93.75% | 30 |
| 2-4 | 6.25% | 2 (boundary) |

**Depth 6 (64 indices, 4 items/index):**
| Span | Probability | Groups |
|------|-------------|--------|
| 2-4 | 93.75% | 30 |
| 4-8 | 6.25% | 2 (boundary) |

**Depth 7 (128 indices, 2 items/index):**
| Span | Probability | Groups |
|------|-------------|--------|
| 4-8 | 93.75% | 30 |
| 8-12 | 6.25% | 2 (boundary) |

**Depth 8 (256 indices, 1 item/index):**
| Span | Probability | Groups |
|------|-------------|--------|
| 8 | 93.75% | 30 |
| 12-16 | 6.25% | 2 (boundary) |

### Key Insight: 93.75% of Groups Are Well-Behaved!

Even at depth 8, 30 out of 32 groups have span ≤ 8 (one vload).
Only ~2 boundary groups per round need special handling.

**Fallback strategy for boundary groups:**
- Detect at runtime (cheap: check first vs last item's partition bit)
- Use 2 vloads instead of 1
- Or fall back to scattered loads for just those 2 groups

### Multi-Round Partition (k > 1)

With partition every k rounds, there are 2^(k-1) partition boundaries:
- k=1: 1 boundary, 2 boundary groups
- k=2: 3 boundaries, 6 boundary groups
- k=3: 7 boundaries, 14 boundary groups

| k | Partition freq | Good groups | Boundary groups |
|---|----------------|-------------|-----------------|
| 1 | Every round | 30 (93.75%) | 2 (6.25%) |
| 2 | Every 2 rounds | 26 (81.25%) | 6 (18.75%) |
| 3 | Every 3 rounds | 18 (56.25%) | 14 (43.75%) |

**Trade-off:** Less frequent partition = more boundary groups, but lower partition overhead.

## Key Finding: Depths 9-10 Are Always Problematic

Regardless of partition frequency, depths 9-10 have `new_span > 8`:
- At depth 9: 512 possible indices, only 256 items → groups span ~16 indices
- At depth 10: 1024 possible indices → groups span ~32 indices

**For depths 9-10, we need either:**
1. Multiple vloads per group (2-4 vloads)
2. Scattered loads as fallback
3. Accept that these rounds are slower

## Optimal Partition Strategy

### Observation: k=2 or k=3 Works for Depths 4-8!

We can partition **every 3-4 rounds** and still get single-vload coverage for depths 4-8.

### Timing Advantage of Lazy Partition

With k=2 (partition every 3 rounds), we have **2 full rounds** to complete partition stores:

```
Round N-2: Branch completes → indices known
Round N-1: [Full round of computation]
           └─ Partition stores can happen throughout!
Round N:   Tree access uses N-2's partitioned data
           └─ Still within single-vload span!
```

**Store budget:**
- 2 rounds × ~100 cycles/round × 2 stores/cycle = 400 store slots
- Need: 512 stores (256 idx + 256 val)
- Can overlap ~80% of stores with computation!

### Index Bit Packing Optimization

Tree indices only need 11 bits (max depth 10 → max index 2047).
We can pack **2 indices per 32-bit word**:

```
Word layout: [unused:10][idx1:11][idx0:11]

# Pack
packed = (idx1 << 11) | idx0

# Extract (VALU/ALU ops)
idx0 = packed & 0x7FF        # mask 11 bits
idx1 = (packed >> 11) & 0x7FF
```

**Store count with packing:**
| Data | Items | Packing | Stores |
|------|-------|---------|--------|
| Indices | 256 | 2 per word | 128 |
| Values | 256 | 1 per word | 256 |
| **Total** | - | - | **384** |

### Partition Cost Analysis (With Packing)

| Component | Cycles | Can Overlap? |
|-----------|--------|--------------|
| 384 scalar stores | 192 | Yes (fits in 2 hash phases!) |
| 64 vloads (reload) | 32 | No (on critical path) |
| Index unpacking | ~8 | Yes (VALU during reload) |
| Serial counter updates | ~22 | Partially |
| **Net critical path** | **~40** | - |

**Two-Phase Store Schedule:**
```
Round N:   Hash phase → ~100 cycles × 2 slots = 200 stores
Round N+1: Hash phase → ~100 cycles × 2 slots = 200 stores
Total:     400 store slots available
Need:      384 stores
Result:    100% overlap possible!
```

### Break-Even Analysis

**Per-round savings from vload vs scattered:**
- Depths 4-8: 5 rounds × (128 - 8) = 600 cycles saved
- Divided by partition frequency (every 3 rounds) = 200 cycles/partition

**Per-partition cost:**
- ~80 cycles on critical path

**Net savings: ~120 cycles per partition event!**

## Depth 9-10 Strategy

For depths 9-10 where single vload doesn't work:

### Option A: Multiple vloads
- Depth 9: 2 vloads cover span of 16
- Depth 10: 4 vloads cover span of 32
- Plus 8-value selection from each vload
- Cost: ~40-80 cycles (still better than 128 for scattered)

### Option B: Preload Entire Range
At depth 9-10, the range is 512-1024 indices, BUT:
- We only need values for the 256 items we have
- With sorted items, these cluster into ~32-64 contiguous regions
- Preload with ~16-32 vloads, then select

### Option C: Hybrid Fallback
- Check span at runtime
- If span ≤ 8: fast path (vload + select)
- If span > 8: scattered loads for that group

## Recommended Implementation

### Phase 1: Lazy Partition (Every 3 Rounds)
```python
partition_rounds = [0, 3, 6, 9, 12, 15]  # Or similar

for rnd in range(16):
    if rnd in partition_rounds:
        # Partition stores can happen during previous 2 rounds
        use_partitioned_buffers()

    # Tree access
    if depth <= 8:
        vload_based_access()  # Single vload per group
    else:
        multi_vload_access()  # 2-4 vloads per group
```

### Phase 2: Double-Buffered Partition
```
Buffer A: Current sorted data (for tree access)
Buffer B: Being written with new partition

Round N:   Read from A, write partition to B
Round N+1: Read from A (still valid for 2 more rounds!)
Round N+2: Read from A (last round)
Round N+3: Swap! Read from B, write partition to A
```

## Summary

| Strategy | Partition Cost | Depths Covered | Expected Savings |
|----------|---------------|----------------|------------------|
| Every round (k=0) | 160 cycles/round | 4-8 | ~-40 cycles (loss) |
| Every 2 rounds (k=1) | 80 cycles/round | 4-8 | ~40 cycles/round |
| **Every 3 rounds (k=2)** | **53 cycles/round** | **4-8** | **~67 cycles/round** |
| Every 4 rounds (k=3) | 40 cycles/round | 4-8 | ~80 cycles/round |

**Winner: Partition every 3-4 rounds** provides best cost/benefit ratio.

With 8 rounds at depths 4+ and ~70 cycles savings per round:
**Total expected savings: ~560 cycles!**

Current: 1822 cycles → **Target: ~1260 cycles (117x speedup)**
