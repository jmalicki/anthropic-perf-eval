# Speculation Strategies for VLIW Tree Traversal

## Problem Structure

The kernel performs tree traversal with hash computation:
```
for each round:
    val = hash(val ^ tree[idx])
    idx = 2*idx + (1 if val%2==0 else 2)
    if idx >= n_nodes: idx = 0
```

This creates a sequential dependency chain:
- `val₁` depends on `tree[idx₀]`
- `idx₁` depends on `val₁ % 2`
- `val₂` depends on `tree[idx₁]`
- ...

## VLIW Resource Availability

| Engine | Slots/Cycle | Typical Usage | Free Slots |
|--------|-------------|---------------|------------|
| ALU | 12 | ~4 for address computation | ~8 |
| VALU | 6 | ~4 for hash (2 groups) | ~2 |
| Load | 2 | 2 for gather | 0 |
| Store | 2 | 0 during compute | 2 |
| Flow | 1 | 0-1 for select | 0-1 |

## Speculation Strategy 1: Range Preload

**Idea**: Indices are bounded! After round r, idx ∈ [2^r - 1, 2^(r+1) - 2]

| Round | Output Range | Size | Strategy |
|-------|--------------|------|----------|
| 0 | {1, 2} | 2 | Broadcast + select |
| 1 | {3..6} | 4 | 4-way select |
| 2 | {7..14} | 8 | vload + offset select |
| 3 | {15..30} | 16 | 2 vloads + select |
| 4+ | large | 2^r | Gather |

**Benefit**: Replaces 64 gather loads with 1-2 vloads for early rounds.
**Cost**: Extra select operations (~8 cycles per round)

## Speculation Strategy 2: Dual Path Execution

**Idea**: Since next_idx is EITHER 2*idx+1 OR 2*idx+2, compute both!

```
# Instead of waiting for val%2 to decide which tree node to load:
tree_left = tree[2*idx + 1]
tree_right = tree[2*idx + 2]
val_left = hash(val ^ tree_left)
val_right = hash(val ^ tree_right)
# Then select:
new_val = (orig_val%2 == 0) ? val_left : val_right
new_idx = (orig_val%2 == 0) ? 2*idx+1 : 2*idx+2
```

**Benefit**:
- Loads BOTH paths in parallel (uses both load slots)
- Hides memory latency behind 2x hash computation
- Uses idle VALU slots productively

**Cost**:
- 2x hash computation (but if VALU is underutilized, it's "free")
- 2x tree loads (but parallel is same latency as sequential)
- Extra scratch for second path

## Speculation Strategy 3: Pipeline Prefetch

**Idea**: While processing round R, prefetch tree values for round R+1.

```
# Processing round R:
val_R = hash(val_{R-1} ^ tree[idx_{R-1}])
idx_R = 2*idx_{R-1} + branch

# Simultaneously prefetch for round R+1:
prefetch tree[2*idx_{R-1} + 1]  # potential left child
prefetch tree[2*idx_{R-1} + 2]  # potential right child
```

**Benefit**: Hides memory latency completely for rounds where prefetch completes in time.
**Cost**: Requires 2 extra load slots (may conflict with current loads).

## Speculation Strategy 4: Branch Prediction

**Idea**: Use pattern history to predict branch direction.

### Static Prediction
- Always predict the more common outcome
- For random hashes, val%2 is 50/50, so either direction works
- Misprediction cost: ~1 cycle (just a select, no rollback needed)

### Profile-Guided Prediction
- Run profiling pass to collect branch statistics
- Use biased prediction if one direction dominates
- Unlikely to help for cryptographic hashes (uniform distribution)

### Pattern Prediction
- Track last N branch outcomes per element
- Predict based on pattern (e.g., alternating)
- Complex to implement, unlikely benefit for hashes

## MILP Decision Variables

The MILP solver should choose per-round:

```
strategy[r] ∈ {RANGE_PRELOAD, DUAL_PATH, PREFETCH, GATHER}
```

### Constraints
```
# Range preload only valid for small ranges
strategy[r] = RANGE_PRELOAD → 2^(r+1) <= 16

# Dual path requires extra scratch
strategy[r] = DUAL_PATH → scratch_used + 8*UNROLL*VLEN <= SCRATCH_SIZE

# Prefetch requires available load slots
strategy[r] = PREFETCH → load_slots_used < 2
```

### Objective Contribution
```
cycles[r] =
  if strategy[r] == RANGE_PRELOAD: range_preload_cycles(r)
  if strategy[r] == DUAL_PATH: dual_path_cycles(r)
  if strategy[r] == GATHER: gather_cycles(r)
```

## Expected Performance

| Strategy | Best Case | Cycles Saved |
|----------|-----------|--------------|
| Range preload (rounds 0-3) | 4 rounds × 30 cycles | ~120 |
| Dual path (rounds 4+) | 12 rounds × 16 cycles | ~192 |
| Pipeline prefetch | 12 rounds × 8 cycles | ~96 |

Combined potential: Save ~400 cycles → ~9,500 → ~5,100 cycles goal

## Implementation Priority

1. ✅ Range preload (rounds 0-3) - Already implemented
2. 🔄 Dual path execution - In progress
3. ⏳ Pipeline prefetch - Future work
4. ⏳ MILP integration - Future work
