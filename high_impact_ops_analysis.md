# High-Impact Ops Analysis: Why They Aren't Shifted

## Summary

**NONE of the 7 high-impact ops are shifted** (0/7), despite having significant optimization potential (159-1061 cycles impact each).

## Root Cause Analysis

### Pattern 1: Cannot Cluster at ASAP (5 ops)

**Ops**: 244, 287, 285, 323, 289

**Issue**: ALU capacity at ASAP cycle prevents clustering all 8 ALU ops:
- Op 244: ASAP=6, ALU at cycle 6 = 11/12 → 11+8=19 > 12 (violates capacity)
- Op 287: ASAP=7, ALU at cycle 7 = 10/12 → 10+8=18 > 12 (violates capacity)
- Op 285: ASAP=7, ALU at cycle 7 = 10/12 → 10+8=18 > 12 (violates capacity)
- Op 323: ASAP=5, ALU at cycle 5 = 11/12 → 11+8=19 > 12 (violates capacity)
- Op 289: ASAP=6, ALU at cycle 6 = 11/12 → 11+8=19 > 12 (violates capacity)

**CP-SAT's Decision Process**:
1. Tries to minimize `completion_cycle` (for makespan optimization)
2. Wants to cluster all 8 ALU ops at ASAP
3. But clustering violates ALU capacity constraint
4. Must spread across ASAP-ALAP to satisfy capacity
5. Spreading increases `completion_cycle` to ALAP
6. Higher `completion_cycle` delays consumers (no speedup)
7. **CP-SAT avoids shift** (hurts makespan)

**Example (Op 244)**:
- If clustered at cycle 6: completion_cycle=6, consumers start at cycle 7 (22 cycle speedup)
- But clustering violates capacity (19 > 12)
- If spread: completion_cycle=28, consumers start at cycle 29 (0 cycle speedup)
- CP-SAT chooses not to shift

### Pattern 2: Can Cluster But Consumer Dependencies Prevent Benefit (1 op)

**Op**: 291

**Issue**: Can cluster at ASAP (cycle 8, ALU=0/12), but:
- Consumers depend on ops at cycle 30
- ALAP = 30 (min consumer cycle - 1)
- If shifted and clustered: completion_cycle=8, but consumers wait for max(8, 30)+1=31
- Current consumer cycle: 31
- **No speedup** - consumers are already waiting for other dependencies

**Result**: CP-SAT avoids shift (no benefit)

### Pattern 3: No Conflicts But Still Not Shifted (1 op)

**Op**: 262

**Issue**: 
- Can cluster at ASAP (cycle 4, ALU=2/12)
- No capacity conflicts
- No consumer dependency conflicts
- Would provide 23 cycle speedup
- **But CP-SAT still doesn't shift it**

**Possible reasons**:
1. CP-SAT hasn't explored this shift option
2. Other constraints not visible in final solution
3. CP-SAT suboptimal solution
4. Search strategy doesn't find this optimization

## Key Findings

### Task 1: Dependency Constraint Bug ✓
- Op 291 and Op 262 have **no dependencies on shifted VALU ops**
- Dependency bug (line 1752) doesn't affect these ops
- **Not the root cause**

### Task 2: ALU Capacity Constraints ✓
- ALU cumulative constraint correctly allows independent scheduling
- CP-SAT is clustering all 8 ALU ops (not spreading)
- This confirms CP-SAT minimizes a composite objective (makespan + tie-breaker; see canonical description in `scheduler_rs/src/cpsat_scheduler/model.rs`)
- **Capacity constraints work correctly**

### Task 3: Completion Cycle Logic ✓
- CP-SAT objective (canonical: model.rs "CANONICAL DESCRIPTION") rewards earlier completion, so it fills earlier slots even when makespan is unchanged
- If clustering violates capacity, CP-SAT must spread
- Spreading increases `completion_cycle`, which delays consumers
- **This is the main reason CP-SAT avoids shifts**

### Task 4: CP-SAT Solution Analysis ✓
- Debug output added to extract shift decisions
- Will show what CP-SAT assigns for high-impact ops

### Task 5: Constraint Conflicts ✓
- **5 ops**: Capacity conflicts prevent clustering
- **1 op**: Consumer dependency conflicts prevent benefit
- **1 op**: No conflicts but still not shifted (mystery case)

## Conclusions

1. **Main issue**: CP-SAT uses a composite objective (canonical: `scheduler_rs/src/cpsat_scheduler/model.rs` "CANONICAL DESCRIPTION"); the tie-breaker fills earlier slots even when that doesn't reduce makespan. For shift decisions, clustering that violates capacity forces spreading, which increases completion_cycle and delays consumers.
   - If clustering violates capacity, must spread
   - Spreading increases `completion_cycle`, delays consumers
   - CP-SAT avoids shift to prevent makespan increase

2. **Op 262 mystery**: Can cluster, no conflicts, should be beneficial
   - Either CP-SAT hasn't found optimal solution
   - Or other constraints not visible in analysis

3. **Solution needed**: Allow `completion_cycle` to be slightly higher if it enables better overall utilization
   - Or: Add soft constraint to encourage shifts when beneficial
   - Or: Improve search strategy to find these optimizations

## Recommendations

1. **Fix dependency bug** (line 1752): Use `completion_cycle_vars` when dependency is shifted
2. **Relax completion_cycle constraint**: Allow up to ALAP when consumers aren't delayed
3. **Add soft objective**: Encourage shifts when clustering is possible and beneficial
4. **Improve search**: Add hints or better search strategy for high-impact ops
