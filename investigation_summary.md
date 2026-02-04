# Investigation Summary: Why High-Impact Ops Aren't Shifted

## Key Finding

**NONE of the 7 high-impact ops are shifted** (0/7), despite having significant optimization potential.

## Root Causes Identified

### 1. Capacity Prevents Clustering (5 ops: 244, 287, 285, 323, 289)

**Issue**: ALU capacity at ASAP cycle is too high to cluster all 8 ALU ops:
- Op 244: ASAP=4 (CP-SAT), but scheduled at cycle 6 (capacity). ALU at cycle 4=2/12 (OK), but at cycle 6=11/12 (can't cluster)
- Op 287: ASAP=5, ALU at cycle 7=10/12 → can't cluster
- Op 285: ASAP=5, ALU at cycle 7=10/12 → can't cluster
- Op 323: ASAP=5, ALU at cycle 5=11/12 → can't cluster
- Op 289: ASAP=5, ALU at cycle 6=11/12 → can't cluster

**CP-SAT's Logic**:
- Wants to cluster at ASAP to minimize `completion_cycle`
- But clustering violates ALU capacity
- Must spread, which increases `completion_cycle` to ALAP
- Higher `completion_cycle` delays consumers, hurts makespan
- **Avoids shift**

### 2. Consumer Dependencies Prevent Benefit (1 op: 291)

**Op 291**: Can cluster at ASAP (cycle 5, ALU=0/12 at cycle 8), but:
- Consumers depend on ops at cycle 30
- Even if Op 291 completion_cycle=5, consumers wait for max(5, 30)+1=31
- Current consumer cycle: 31
- **No speedup** - consumers already waiting for other dependencies

### 3. Mystery Case (1 op: 262)

**Op 262**: 
- Can cluster at ASAP (cycle 4, ALU=2/12)
- No capacity conflicts
- No consumer dependency conflicts  
- Would provide 23 cycle speedup
- **But CP-SAT still doesn't shift it**

**Possible reasons**:
- CP-SAT hasn't explored this shift option
- Search strategy doesn't find this optimization
- Other constraints not visible in analysis

## Critical Insight: ASAP/ALAP vs Capacity

**CP-SAT's ASAP/ALAP calculation**:
- Based ONLY on dependencies (RAW constraints)
- Does NOT account for capacity constraints
- Op 244: ASAP=4 (dependencies allow it)
- But VALU capacity forces it to cycle 6

**When CP-SAT considers shifting**:
- Uses ASAP=4 (theoretical bound)
- Thinks: "Can cluster 8 ALU ops at cycle 4"
- ALU at cycle 4 = 2/12, so 2+8=10 <= 12 (OK)
- But CP-SAT might still avoid shift if it thinks spreading is needed

## Debug Output Findings

From CP-SAT solution extraction:
- All 7 high-impact ops: `shifted=false`
- ASAP values: 4-5 (earlier than final schedule cycles)
- ALAP values: 913 (horizon - 1, very large window)

This confirms CP-SAT is not shifting these ops.

## Recommendations

1. **Fix dependency bug** (line 1752): Use `completion_cycle_vars` when dependency is shifted
2. **Relax completion_cycle**: Allow up to ALAP when consumers aren't delayed by other dependencies
3. **Add soft constraint**: Encourage shifts when clustering is possible and beneficial
4. **Improve search**: Add hints for high-impact ops or better search strategy
5. **Partial clustering**: Allow some ALU ops at ASAP, some spread (don't require all 8 at same cycle)
