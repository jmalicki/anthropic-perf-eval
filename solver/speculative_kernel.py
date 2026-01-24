"""
Speculative Dual-Path Loading - Analysis and Conclusion

This file documents the exploration of speculative dual-path loading as described
in the optimization plan. The conclusion is that speculation does NOT help for
this workload.

## Why Speculation Doesn't Work

### 1. Scratch Memory Constraints
- Current usage: 1,332 / 1,536 words
- Speculation buffers needed: 512 words (32 groups × 2 children × 8 words)
- Result: Scratch overflow (1,844 > 1,536)

### 2. Buffer Reuse Doesn't Help
Even if we reuse v_nv as v_left and v_t as v_right:
- vselect(dest, cond, a, b) needs a vector for `cond`
- Computing cond = idx & 1 requires a temp vector
- No free temp vectors available during Phase 1

### 3. vload Doesn't Help with Consecutive Children
Although children at 2*idx+1 and 2*idx+2 are consecutive in memory:
- vload(dest, addr) puts mem[addr..addr+7] into dest[0..7]
- Each batch element has DIFFERENT idx, so different addresses
- We'd need to transpose: vload gives [left_i, right_i, ...] per element
  but vselect needs [left_0..left_7] and [right_0..right_7]
- No shuffle/transpose instruction in this ISA

### 4. Load-Bound Analysis
Even if we could implement speculation:
- Current gathers: 3,328 loads (256 elements × 13 rounds)
- Speculation would ADD 3,328 loads (both children per element)
- Total: 6,656 loads / 2 per cycle = 3,328 cycles minimum
- This is WORSE than current 2,227 cycles!

The only way speculation helps is if we can overlap loads with computation,
but with 78.6% load utilization, we're already near the load bandwidth limit.

## Theoretical Limit
- Minimum gather cycles: 1,664 (3,328 gathers ÷ 2 per cycle)
- Non-gather overhead: ~565 cycles (hash, branch, bounds check)
- Total: ~2,229 cycles (matches actual!)

## Conclusion
The current 2,227 cycles (66.2× speedup) is NEAR-OPTIMAL for this algorithm
on this architecture. Further improvements would require:
1. Algorithm redesign (different traversal order)
2. Architecture changes (more load bandwidth, indexed scratch)
3. Problem relaxation (smaller batches, fewer rounds)

The baseline kernel in interleaved_unrolled_kernel.py is the production solution.
"""

# This file is kept for documentation purposes.
# The actual production kernel is in solver/interleaved_unrolled_kernel.py
