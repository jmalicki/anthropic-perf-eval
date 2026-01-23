# Optimization Plan: Achieving 100x Speedup

## Current State
- **Best**: 12,373 cycles (12x speedup) with ultra_8x kernel
- **Target**: 1,487 cycles (100x speedup)
- **Gap**: 10,886 cycles to close

## Phase 1: Debug Smart Kernel (Tree Preload)

### Problem
The smart kernel that preloads tree[0..7] produces incorrect results.

### Root Cause Analysis
1. The vbroadcast from vector elements may be wrong
2. The conditional selection between cache and gather is buggy
3. The step loop structure may have off-by-one errors

### Fix Strategy
1. Add debug output to trace values
2. Verify tree preload is correct
3. Fix the permute logic (selecting tree[idx] from vector)
4. Test each step separately

## Phase 2: Implement Speculative Execution

### Concept
Instead of:
```
val = hash(val ^ tree[idx])
idx = 2*idx + (val%2==0 ? 1 : 2)
```

Do:
```
# Speculatively compute both branches
tree_left = tree[2*idx + 1]
tree_right = tree[2*idx + 2]
val_left = hash(val ^ tree_left)
val_right = hash(val ^ tree_right)

# Select based on original val
val = (val%2==0) ? val_left : val_right
idx = (val%2==0) ? 2*idx+1 : 2*idx+2
```

### Benefits
- Both branches loaded/computed in parallel
- No dependency on hash result for tree load
- Uses idle VALU slots productively

### Implementation
1. Create speculative kernel
2. Load both possible tree values
3. Compute hash for both
4. Select correct result

## Phase 3: MILP with Speculation

### New Variables
- `spec[i,j]` = 1 if operation i is speculatively executed for branch j
- `select[i]` = 1 if operation i is a selection (merging speculative results)

### New Constraints
- Speculative ops must have both branches scheduled
- Selection must happen after both speculative branches complete
- Resource limits include speculative operations

### Objective
Minimize cycles while allowing MILP to choose:
- When to speculate (high parallelism available)
- When to execute directly (low parallelism or high cost)

## Phase 4: Integration & Benchmarking

### Combined Optimizations
1. Tree preload for early steps (indices < 8)
2. Speculation for later steps
3. 8-way group unrolling
4. Maximum VLIW packing

### Expected Improvements
| Optimization | Expected Speedup |
|--------------|------------------|
| Tree preload (steps 0-2) | 2x over current |
| Speculation | 1.5x additional |
| Better packing | 1.3x additional |
| **Combined** | ~4x → target ~3,000 cycles |

### Stretch Goal
To reach 1,487 cycles, we may need:
- Algorithm restructuring
- Deeper speculation (3+ levels)
- Custom instruction sequences

## Execution Timeline

1. **Phase 1**: Debug smart kernel → verify tree preload works
2. **Phase 2**: Implement speculation → test correctness
3. **Phase 3**: MILP enhancement → auto-optimize speculation
4. **Phase 4**: Integrate all → benchmark against target
