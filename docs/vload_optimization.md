# vload Optimization for Tree Gather

## Overview

This document analyzes the potential for using `vload` (vector load) to optimize the tree gather operations that dominate kernel performance.

## Current Performance

| Metric | Value |
|--------|-------|
| Current best | 2,390 cycles |
| Speedup | 61.8x |
| Target | 1,487 cycles (100x) |
| VALU utilization | ~87% |
| Load utilization | ~71% |

## Definitions

### Batch
The 256 input elements being processed in parallel. Each batch element has:
- An **index** into the tree (starts at 0, updates each round)
- A **value** that gets hashed each round

### Group
A set of 8 batch elements processed together as a SIMD vector. With VLEN=8 and batch_size=256:
- 32 groups total (256 / 8 = 32)
- Group g contains batch elements [g*8, g*8+1, ..., g*8+7]

### Iteration
Groups are processed in iterations of UNROLL=8 groups each:
- 4 iterations total (32 / 8 = 4)
- Iteration it processes groups [it*8, it*8+1, ..., it*8+7]

### Round
One traversal step down the tree. With 16 rounds:
- Round 0: All elements at root (index 0)
- Round 1: Elements at level 1 (indices 1-2)
- Round 2: Elements at level 2 (indices 3-6)
- Round 3+: Elements at level 3+ (indices 7+), requires gather

## Tree Memory Layout (BFS Order)

```
Index 0:      root (level 0) - 1 node
Index 1-2:    level 1 - 2 nodes
Index 3-6:    level 2 - 4 nodes
Index 7-14:   level 3 - 8 nodes    ← exactly 8, perfect for vload!
Index 15-30:  level 4 - 16 nodes
Index 31-62:  level 5 - 32 nodes
...
Level L: indices [2^L - 1, 2^(L+1) - 2], total 2^L nodes
```

## The vload Opportunity

### Key Insight: Level 3 Has Exactly 8 Nodes

At round 3, all batch elements need values from indices 7-14.

**ONE vload could load the ENTIRE level:**
```
vload(dest, forest_p + 7) → loads forest[7:15] into scratch
```

This is 1 Load op instead of 256 scalar loads!

### The Challenge: ISA Limitations

**The ISA lacks gather-from-scratch:**

After vloading `tree[7:14]` into scratch (say at address V), we need to extract `V[idx-7]` for each batch element. But:

- `load` instruction: `("load", dest, addr)` loads from `mem[scratch[addr]]`
  - Reads address FROM scratch, loads FROM memory
  - Does NOT load from scratch at a computed offset

- **There is no**: `("load_from_scratch", dest, base, offset)` that computes `scratch[base + scratch[offset]]`

### Attempted Workarounds

**Option A: Store back to memory, then gather**
```python
# 1. vload tree[7:14] into scratch at V
vload(V, forest_p + 7)

# 2. vstore V back to memory at temp location T
vstore(T_addr, V)

# 3. Per element: compute T + (idx - 7), load from memory
for element:
    addr = T + (idx - 7)   # 1 ALU
    load from mem[addr]    # 1 Load
```

Cost: 2 ALU + 1 Load per element, vs current 1 ALU + 1 Load

**Conclusion**: This approach is WORSE than direct gather!

**Option B: Broadcast + Compare chains**
```python
# After vloading tree[7:14] into scratch at V
local_idx = idx - 7
result = 0
for k in 0..7:
    mask = (local_idx == k)            # 1 VALU compare
    broadcasted = broadcast(V[k])      # 1 VALU broadcast
    result = select(mask, broadcasted, result)  # 1 flow/VALU
```

Cost: ~24 VALU ops per group (8 comparisons × 3 ops)

**Conclusion**: Also worse than direct gather (~4 cycles vs ~24 ops)

## Why Direct Gather Wins

Current gather implementation (rounds 3-15):
```python
for vi in range(VLEN):  # 8 vector elements
    # Address calculation (1 ALU per element)
    emit(alu=[("+", st_a, forest_p, all_idx[group] + vi)])
    # Memory load (1 Load per element)
    emit(load=[("load", v_nv[it][i] + vi, st_a)])
```

Per group: 8 ALU + 8 Load = 16 ops
At 2 loads/cycle and 12 ALU/cycle: ~4 cycles per group

**The ISA's scatter/gather limitation means direct scalar loads are optimal.**

## Alternative Optimizations (Implemented)

Since vload can't help with general gather, the codebase uses other strategies:

### 1. Tree Caching for Early Rounds ✅

Rounds 0-2 have predictable indices. Cache tree[0-6] and use broadcast + select.

**Implementation** (in `interleaved_unrolled_kernel.py`):
```python
if rnd == 0:
    # Round 0: just broadcast tc[0]
    emit(valu=[("vbroadcast", v_nv[it][i], tc[0])])

elif rnd == 1:
    # Round 1: v_nv = tc[2] + (idx == 1) * td_1
    emit(valu=[("==", t, all_idx[g], v_1)])
    emit(valu=[("*", t, v_td_1, t)])
    emit(valu=[("+", v_nv[it][i], v_tc2, t)])
```

### 2. VALU→ALU Shifting ✅

Move VALU ops to ALU when ALU has earlier free slots.

**Trade-off**:
- VALU: 1 slot, processes 8 elements
- ALU: 8 slots, processes 8 elements (1 each)

When VALU is saturated but ALU has capacity, shifting helps.

### 3. CP-SAT Global Scheduling ✅

Let the solver find optimal operation ordering to maximize VLIW packing.

## Future Opportunities

### Speculative Dual-Path Loading

Instead of trying to use vload for gather, load BOTH possible tree values:

```python
# Load both possible children
tree_left = load(tree[2*idx + 1])
tree_right = load(tree[2*idx + 2])

# Compute both paths
val_left = hash(val ^ tree_left)
val_right = hash(val ^ tree_right)

# Select correct result
val = select(cond, val_left, val_right)
```

**Benefits**:
- Uses both load slots productively
- Overlaps load latency with computation
- Uses "free" ALU cycles for extra address computation

## Summary

| Approach | Viable? | Notes |
|----------|---------|-------|
| vload + store-back | ❌ | More ops than direct gather |
| vload + broadcast chain | ❌ | Way more VALU ops |
| Tree caching (rounds 0-2) | ✅ | Implemented, eliminates 3 rounds of gather |
| Direct scalar gather | ✅ | Best option for rounds 3+ |
| Speculative dual-path | ⏳ | Promising, not yet implemented |

**Bottom line**: The ISA's lack of gather-from-scratch makes vload unsuitable for optimizing the tree gather. Direct scalar loads remain the best approach, optimized via CP-SAT scheduling.
