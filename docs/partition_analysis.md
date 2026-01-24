# Mathematical Analysis: Partitioning and Sorting Strategies for SIMD Kernel Optimization

## Problem Setup

- **Batch size**: N = 256 items
- **SIMD width**: VLEN = 8 items per vector
- **Groups**: G = N / VLEN = 32 groups
- **Tree depth**: D = 11 (forest_height + 1)
- **Rounds**: R = 16

At each round, items traverse the tree by:
1. XOR item value with tree[current_idx]
2. Hash the result
3. Branch: new_idx = 2 * current_idx + (hash & 1)

## Key Metric: Index Span per Group

For efficient tree value selection, we want to minimize the **index span** of each group:
- **span(g) = max_idx(g) - min_idx(g)**

If span ≤ k, we can use (k+1)-value selection instead of scattered loads.

---

## Strategy 1: No Partition (Current Baseline)

### Model
Items remain in original order. Tree indices within a group follow a random walk.

### Analysis
At depth d, each item's tree index is determined by d independent random bits.
For a group of 8 consecutive items with independent random indices in range [2^d - 1, 2^(d+1) - 2]:

**Expected span** = E[max - min] of 8 random samples from 2^d values

Using order statistics:
- E[max] ≈ 2^d - 1 + (2^d - 1) * (VLEN / (2^d + 1))
- E[min] ≈ 2^d - 1 + (2^d - 1) * (1 / (2^d + 1))
- E[span] ≈ (2^d - 1) * (VLEN - 1) / (2^d + 1)

| Depth | 2^d | E[span] | Strategy Needed |
|-------|-----|---------|-----------------|
| 0 | 1 | 0 | 1-value (trivial) |
| 1 | 2 | 0.58 | 2-value |
| 2 | 4 | 1.4 | 2-value |
| 3 | 8 | 2.3 | 3-value |
| 4 | 16 | 3.7 | 4-value |
| 5 | 32 | 5.4 | 6-value |
| 6 | 64 | 6.5 | 7-value |
| 7+ | 128+ | ~7 | 8-value |

**Conclusion**: Without partitioning, even at depth 4 we need 4-value selection. At depth 7+, we need 8-value selection.

### Cost per Round (depth d ≥ 4)
- Scattered loads: 256 loads @ 0.5 cycles each = 128 cycles
- 8-value selection: 32 groups × (1 vload + 16 VALU) = ~64 cycles

**Savings**: ~64 cycles/round without any sorting

---

## Strategy 2: Full Stable Partition

### Model
After each round, items are partitioned by branch bit (0 vs 1) using stable partition:
- LOW items (branch=0) maintain relative order, placed first
- HIGH items (branch=1) maintain relative order, placed second

### Analysis
After stable partition, items are sorted by tree index within each partition level.

**Partition structure after r rounds**:
- 2^r partitions, each of size N/2^r = 256/2^r
- All items in partition p have the same tree index

| Round | Partitions | Items/Partition | Groups/Partition |
|-------|------------|-----------------|------------------|
| 0 | 1 | 256 | 32 |
| 1 | 2 | 128 | 16 |
| 2 | 4 | 64 | 8 |
| 3 | 8 | 32 | 4 |
| 4 | 16 | 16 | 2 |
| 5 | 32 | 8 | 1 |
| 6 | 64 | 4 | 0.5 |
| 7 | 128 | 2 | 0.25 |
| 8+ | 256+ | ≤1 | <0.125 |

**Index span with sorted items**:

When partition_size ≥ VLEN (rounds 0-5):
- Most groups are entirely within one partition: span = 0
- Boundary groups span 2 partitions: span = 1
- Number of boundary groups = min(2^r - 1, 31)

When partition_size < VLEN (rounds 6+):
- Groups contain items from multiple partitions
- Since items are sorted, span = (items per group from different partitions) - 1
- Expected span = ceil(VLEN / partition_size) - 1

| Round | Partition Size | Boundary Groups | E[span] non-boundary | E[span] boundary |
|-------|---------------|-----------------|---------------------|------------------|
| 0-4 | ≥16 | ≤15 | 0 | 1 |
| 5 | 8 | 31 | 0 | 1 |
| 6 | 4 | 31 | 1 | 2 |
| 7 | 2 | 31 | 3 | 4 |
| 8 | 1 | 31 | 7 | 8 |

**Key insight**: With stable partition, rounds 0-6 have span ≤ 2 for 97%+ of groups!

### Partition Cost

**Per-round partition cost**:
- Read 256 values + branch bits: 32 vloads
- Write to two output buffers: 32 vstores
- Compute write indices: ~32 VALU ops

Total: ~64 cycles per round for partitioning

**Break-even analysis**:
- Partitioning cost: 64 cycles/round
- Savings from 2-value selection vs scattered: 128 - 10 = 118 cycles/round

**Net savings**: ~54 cycles/round for depths 4-6

### Total Savings Estimate

| Round | Depth | Partition Cost | Selection Savings | Net Savings |
|-------|-------|----------------|-------------------|-------------|
| 0-3 | 0-3 | 64 | 0 (already optimal) | -64 |
| 4-6 | 4-6 | 64 | 118 | +54 |
| 7-10 | 7-10 | 64 | 64 (8-value vs scattered) | 0 |
| 11-14 | 0-3 | 64 | 0 | -64 |
| 15 | 4 | 64 | 118 | +54 |

**Rounds benefiting**: 4, 5, 6, 15 (4 rounds × 54 = 216 cycles saved)
**Rounds losing**: 0-3, 7-10, 11-14 (12 rounds × 64 = 768 cycles lost)

**Conclusion**: Full stable partition is NOT worth it! The partitioning overhead exceeds savings.

---

## Strategy 3: Conditional Partition (Depth-Dependent)

### Model
Only partition when beneficial:
- Depths 0-3: No partition (VALU selection already optimal)
- Depths 4-6: Partition (enables 2-value selection)
- Depths 7+: No partition (span too high anyway)

### Analysis
This avoids partitioning overhead for rounds where it doesn't help.

**Partition only at depths 4, 5, 6, and wrapped depth 4 (round 15)**:
- 4 partition rounds × 64 cycles = 256 cycles overhead
- 4 rounds × 118 cycles savings = 472 cycles savings

**Net savings**: 472 - 256 = **216 cycles**

But there's a problem: **partition state doesn't persist!**

If we don't partition at depth 7, items become unsorted again. Then at depth 4 (round 15), we need to re-sort from scratch, which costs more than stable partition.

### Maintaining Sorted Order

**Key insight**: Once sorted, items stay sorted if we continue partitioning!

At depth 7+, partition_size < VLEN, so partitioning is "free" in terms of index span:
- Items are already small-partition organized
- Partitioning just moves them to smaller partitions
- The additional partition cost keeps them sorted for when depth wraps

**Revised strategy**:
- Rounds 0-3: No partition (span naturally small)
- Rounds 4-15: Partition (maintain sort order for depth 4-6 rounds)

Partition rounds: 4-15 (12 rounds × 64 = 768 cycles)
Benefit rounds: 4, 5, 6, 15 (4 rounds × 118 = 472 cycles)

**Net**: 768 - 472 = **276 cycles WORSE**

---

## Strategy 4: Partial Partition (Skip Expensive Groups)

### Model
Only partition groups that benefit from sorting, skip "boundary" groups.

### Analysis
At round r, there are min(2^r - 1, 31) boundary groups that span partitions.

For non-boundary groups: span = 0, so 1-value selection works.
For boundary groups: span = 1, so 2-value selection works.

**Insight**: We don't need to partition boundary groups differently - they just need 2-value selection instead of 1-value!

The key question is: **what's the cost of 2-value selection for boundary groups?**

**2-value selection cost**:
- Extract first_idx from vector: vstore + load = 2 cycles
- Load tree[first_idx] and tree[first_idx+1]: 2 loads = 1 cycle
- Broadcast both: 2 vbroadcast = 2 VALU ops
- Compute diff: 1 vsub = 1 VALU op
- vselect: 1 flow op

Total: ~5-6 cycles per boundary group

**1-value selection cost** (non-boundary):
- Same extraction + 1 load + 1 broadcast = ~4 cycles

**Difference**: ~1-2 cycles per boundary group

With up to 31 boundary groups: ~31-62 cycles overhead for 2-value selection.

**But** we save the full partitioning cost of 64 cycles/round!

---

## Strategy 5: No Partition + Adaptive Selection

### Model
Don't partition at all. Instead, for each group:
1. Extract min and max tree index
2. Compute span = max - min
3. Use (span+1)-value selection

### Analysis

**Per-group overhead**:
- Extract min/max: vstore + 2 loads = 3 cycles
- span = max - min: 1 ALU op

**Selection by span**:
| Span | Selection Type | Cost (VALU ops) |
|------|----------------|-----------------|
| 0 | 1-value broadcast | 1 |
| 1 | 2-value vselect | 5 |
| 2-3 | 4-value compare/select | 9 |
| 4-7 | 8-value compare/select | 17 |

**Expected cost at depth d** (no partition):

From Strategy 1 analysis, E[span] at each depth. Let's compute expected selection cost:

| Depth | E[span] | P(span≤1) | P(span≤3) | P(span≤7) | E[selection cost] |
|-------|---------|-----------|-----------|-----------|-------------------|
| 0-3 | <2 | >95% | ~100% | 100% | ~5 VALU |
| 4 | 3.7 | ~20% | ~70% | ~99% | ~11 VALU |
| 5 | 5.4 | ~5% | ~35% | ~95% | ~14 VALU |
| 6 | 6.5 | ~2% | ~15% | ~85% | ~16 VALU |
| 7+ | ~7 | ~1% | ~5% | ~70% | ~17 VALU |

**Cost comparison (per round, 32 groups)**:

| Depth | Scattered | Adaptive (no partition) | Savings |
|-------|-----------|------------------------|---------|
| 4 | 128 | 32 × 11/6 ≈ 59 | 69 |
| 5 | 128 | 32 × 14/6 ≈ 75 | 53 |
| 6 | 128 | 32 × 16/6 ≈ 85 | 43 |
| 7+ | 128 | 32 × 17/6 ≈ 91 | 37 |

(Note: dividing by 6 to convert VALU ops to cycles, assuming 6 VALU ops/cycle)

**Total savings over 8 "scattered load" rounds**: ~350 cycles

**This requires**:
- Min/max extraction: 32 groups × 3 cycles = 96 cycles/round
- But this can be amortized across multiple values needed per group

---

## Strategy 6: Precomputed Span Buckets

### Model
At compile time, precompute which groups fall into which span buckets based on depth.

### Analysis

**Key insight**: The distribution of spans is deterministic given:
- Depth d
- Group index g
- Partition boundaries (if any)

At depth d with N=256 items and G=32 groups:
- Number of tree indices: 2^d
- Items per index (expected): 256/2^d
- Variance in items per index: binomial variance

**Boundary probability** (group g contains items from >1 index):

P(boundary) ≈ min(1, VLEN * 2^d / N) = min(1, 8 * 2^d / 256) = min(1, 2^(d-5))

| Depth | P(boundary) | Expected boundary groups |
|-------|-------------|-------------------------|
| 0-4 | 0-0.5 | 0-16 |
| 5 | 1.0 | 31 (at partition edges) |
| 6+ | 1.0 | 32 |

For depths where items are sparse (d ≥ 5), we can precompute the exact span for each group!

**Precomputation**:
1. Simulate the random walk with seed=123
2. Record span[round][group] for all rounds and groups
3. Generate specialized code paths for each span

**Code generation**:
```python
# For each round r and group g:
if precomputed_span[r][g] == 0:
    emit_1value_selection(g)
elif precomputed_span[r][g] == 1:
    emit_2value_selection(g)
# ... etc
```

**Advantage**: No runtime span computation, optimal code per group
**Disadvantage**: Only works for fixed seed, code bloat

---

## Strategy 7: Hybrid with Sort-Once Checkpoints

### Model
Perform full sort at strategic "checkpoints" (e.g., rounds 4, 11), then maintain via cheap operations.

### Analysis

**Full sort cost**: O(N log N) comparisons
With N=256: ~2048 comparisons
Using SIMD comparators: ~256 VALU ops = ~43 cycles

**Stable partition cost**: O(N) = 256 ops = ~43 cycles

**But** sorting destroys the original item order, which may be needed for other operations!

If we can sort just the tree indices (keeping item order), we get:
- Sorted indices enable cheap selection
- Original item layout preserved for value operations

**Index-only sort**:
- Sort (tree_idx, item_position) pairs
- Apply permutation to compute tree values
- Don't actually reorder items

Cost: Sort (43 cycles) + Gather (128 cycles) = 171 cycles

**Not better than scattered loads** (128 cycles)!

---

## Conclusion: Recommended Strategy

### Best Approach: Adaptive Selection Without Partition

Based on the analysis:

1. **Don't partition** - the overhead exceeds benefits
2. **Use adaptive selection** based on runtime span
3. **Amortize span computation** by computing min/max once per group

### Implementation

```python
for each group g:
    # Extract min/max index (3 cycles amortized)
    min_idx, max_idx = extract_minmax(idx_vector[g])
    span = max_idx - min_idx

    if span == 0:
        # 1-value: just broadcast
        v_tree[g] = vbroadcast(tree[min_idx])
    elif span == 1:
        # 2-value: load pair + vselect
        low, high = tree[min_idx], tree[min_idx + 1]
        v_tree[g] = vselect(idx - min_idx, vbroadcast(high), vbroadcast(low))
    elif span <= 7:
        # N-value: vload + compare/select
        vload(cache, tree + min_idx)
        v_tree[g] = select_from_cache(idx - min_idx, cache, span + 1)
    else:
        # Fallback: scattered loads (shouldn't happen often)
        v_tree[g] = scattered_load(tree, idx)
```

### Expected Performance

| Round Type | Rounds | Current Cost | Adaptive Cost | Savings |
|------------|--------|--------------|---------------|---------|
| VALU (d≤3) | 8 | ~45 | ~45 | 0 |
| Scattered (d>3) | 8 | ~128 | ~70 | ~58 |

**Total savings**: 8 × 58 = **~464 cycles**

Current: 1884 cycles
Expected: 1884 - 464 = **~1420 cycles** (104x speedup)

This exceeds the 100x target of 1474 cycles!

---

## Appendix: Probability Calculations

### Order Statistics for Uniform Distribution

For n samples from Uniform[0, M]:
- E[max] = M * n / (n + 1)
- E[min] = M * 1 / (n + 1)
- E[span] = E[max] - E[min] = M * (n - 1) / (n + 1)

With n = VLEN = 8 and M = 2^d:
- E[span] = 2^d * 7/9 ≈ 0.78 * 2^d

### Binomial Distribution for Partition Sizes

After r rounds, items in each partition follow:
- Partition p has B(N, 2^(-r)) items on average
- E[items in partition p] = N / 2^r
- Var[items] = N * 2^(-r) * (1 - 2^(-r)) ≈ N / 2^r for large r

### Span Distribution with Sorted Items

When items are sorted by tree index within a group:
- span = (number of unique indices in group) - 1
- At depth d with partition_size = N/2^d:
  - If partition_size ≥ VLEN: span ∈ {0, 1} (boundary vs interior)
  - If partition_size < VLEN: span = ceil(VLEN / partition_size) - 1

