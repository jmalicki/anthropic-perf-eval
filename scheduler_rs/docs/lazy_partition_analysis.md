# Lazy Partition Optimization Analysis

## Overview

This document analyzes the lazy partition optimization for tree access at depths 4-8.
The goal is to replace expensive scattered loads (8 loads per group) with efficient
vloads (1 vload per group).

## Current State

- **Baseline**: 1759 cycles (83.99x speedup)
- **Target**: 1487 cycles (100x speedup)
- **Partition currently**: Disabled (adds overhead without proper stable partition)

## Key Insight: Tree Structure

Trees are binary! After stable partition by branch bit:

```
Depth 3:  Partition into LEFT (bit=0) and RIGHT (bit=1)
          LEFT group  → accesses subtree [0..7]
          RIGHT group → accesses subtree [8..15]

Depth 4:  LEFT still in [0..15], RIGHT still in [16..31]
          Indices DOUBLE but stay in same subtree

Depth 5:  [0..31] and [32..63] - still contiguous!
```

**Once in a partition, items can only go to a sub-partition.**
This means indices stay clustered within a contiguous range.

## Cost Model

### Baseline (Scattered Loads)

| Component | Calculation | Cycles |
|-----------|-------------|--------|
| Late-depth loads | 5 depths × 32 groups × 8 loads / 2 per cycle | 640 |

### With Partition (vload + selection)

| Component | Calculation | Cycles |
|-----------|-------------|--------|
| vload tree access | 5 depths × 32 groups × 1 vload / 2 per cycle | 80 |
| Partition loads | 5 partitions × 32 groups × 2 vloads / 2 per cycle | 160 |
| 4-way selection | 5 × 32 × 14 ops / 6 VALU per cycle | 373 |
| Partition stores | **FREE** (overlapped with hash via tick-tock) | 0 |

### Tick-Tock Overlap

**Critical insight**: Partition stores can overlap with hash computation!

- Hash computation per round: ~576 VALU ops = ~96 cycles
- Hash cycles between partitions: ~288 cycles
- Partition stores needed: 32 cycles
- **Stores FIT within hash! Partition stores are essentially FREE.**

### Load/VALU Overlap

During tree access:
- vload uses LOAD slot
- Selection uses VALU slots
- These run in **parallel**!

Effective tree access = max(load_cycles, select_cycles)

## Net Benefit Calculation

| Metric | Value |
|--------|-------|
| Load savings | 640 - 80 - 160 = 400 cycles |
| Selection overhead | 373 cycles (overlaps with loads) |
| Effective savings | 640 - max(80, 373) - 160 = 107 cycles |
| **Projected cycles** | 1759 - 107 = **1652** |
| **Projected speedup** | **89.4x** |

## Implementation Requirements

### 1. Stable Partition Algorithm

Current `emit_partition_stores()` just copies data. Needs to:
1. Get branch bits from v_t (computed during branching)
2. Compute prefix sum of (1-branch_bit) for zeros_before
3. Calculate destination: `dest[i] = branch_bit ? zeros_count + ones_before : zeros_before`
4. Scatter items to destinations

Cost: ~30-50 ops per group (needs implementation and measurement)

### 2. Sorted vload Access

After stable partition:
- min(indices) is at lane 0 (no reduction needed!)
- Span ≤ 2^k where k = rounds since partition
- 4-way selection for k=2, 8-way for k=3

### 3. Tick-Tock Scheduling

Partition stores must be scheduled during hash computation:
- TICK: Process groups 0-15, store to buffer A
- TOCK: Process groups 16-31, store to buffer B
- Alternate: read A/write B, then read B/write A

## Partition Frequency Trade-offs

| k | Partitions | Span | Selection | Benefit |
|---|------------|------|-----------|---------|
| 2 | 8 rounds | ≤4 | 4-way (13 ops) | Higher load overhead |
| 3 | 5 rounds | ≤8 | 4-way+vselect (31 ops) | Better balance |
| 4 | 4 rounds | ≤16 | Need 2 vloads | Worse |

**Recommended: k=3** (partition every 3 rounds)

## What Makes This Work

1. **Binary tree structure**: Partitioned items access contiguous subtrees
2. **Stable partition**: Preserves relative order, min always at lane 0
3. **Tick-tock overlap**: Partition stores are free (hidden by hash)
4. **Load/VALU parallelism**: Selection overlaps with vload

## Next Steps

1. Implement stable partition (prefix sum + scatter)
2. Measure actual overhead
3. Tune k parameter based on real performance
4. Consider conditional high/low skip for additional savings

## History

- Initial analysis: Identified 560 cycle savings potential
- Tick-tock insight: Partition stores are free (overlapped)
- Selection overhead: Nearly cancels savings without optimization
- Tree structure insight: Indices stay contiguous in subtrees
- Projected improvement: ~107 cycles (89.4x speedup)
