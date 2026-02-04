# Element-wise VALU Shift Dependencies Plan

## Problem Statement

Currently, when VALU operations are shifted to ALU, the system uses a `completion_cycle` variable that represents the maximum of all 8 ALU cycles. This means:

- **VALU1** → expands to 8 ALU ops: `ALU1[0..7]`
- **VALU2** depends on **VALU1** → expands to 8 ALU ops: `ALU2[0..7]`
- **Current behavior**: All 8 ALU ops of VALU2 wait for `completion_cycle(VALU1)` = `max(ALU1[0..7])`
- **Desired behavior**: `ALU2[i]` should start as soon as `ALU1[i]` completes (element-wise dependencies)

This optimization allows parts of VALU2 to start executing before all of VALU1 finishes, improving parallelism and reducing makespan.

## Current Implementation

### CP-SAT Model (scheduler_rs/src/cpsat_scheduler.rs)

1. **Shiftable VALU ops** (lines 1946-2005):
   - Creates 8 `alu_cycle_vars` per shiftable VALU op (one per ALU lane)
   - Creates 1 `completion_cycle_var` = `max(alu_cycle_vars)`

2. **Dependency constraints for ALU ops** (lines 2012-2043):
   - When VALU2 depends on VALU1 and both are shifted:
     - Each of the 8 ALU ops in VALU2 waits for `completion_cycle(VALU1)`
     - Uses conditional constraints: `alu_cycle[VALU2][i] >= completion_cycle[VALU1] + 1` when both are shifted

3. **Dependency constraints for consumers** (lines 2045-2060):
   - Non-shifted consumers of shifted VALU ops wait for `completion_cycle`

### Key Code Locations

- **CP-SAT model building**: `scheduler_rs/src/cpsat_scheduler.rs` lines 1940-2060
- **ALU cycle variables**: `alu_cycle_vars: HashMap<usize, Vec<IntVar>>` (8 vars per shifted VALU op)
- **Completion cycle**: `completion_cycle_vars: HashMap<usize, IntVar>` (max of 8 ALU cycles)
- **Dependency enforcement**: Lines 2012-2043 (ALU ops) and 2045-2060 (consumers)

## Solution Design

### Element-wise Dependency Mapping

For standard VALU operations, element `i` of VALU2 depends on element `i` of VALU1 (same lane). However, we need to handle:

1. **Standard VALU ops**: `dest[i] = f(src1[i], src2[i])`
   - Element `i` depends on element `i` of sources
   - If VALU1 writes `dest1[0..7]` and VALU2 reads `src1[0..7] = dest1[0..7]`, then `ALU2[i]` depends on `ALU1[i]`

2. **Broadcast sources**: `dest[i] = f(scalar, src2[i])`
   - Element `i` depends on the scalar (same for all lanes) and element `i` of the vector source
   - If VALU1 is a broadcast, all elements of VALU2 that read from it depend on the same scalar

3. **Mixed dependencies**: Some elements may depend on different sources

### Implementation Strategy

#### Phase 1: Track Element-wise Dependencies

1. **Extend SSA dependency tracking**:
   - Currently: `SsaOp.dep_ids` tracks VALU-level dependencies
   - Need: Track which element(s) of a VALU dependency each element depends on
   - Option A: Add `dep_elements: HashMap<usize, Vec<usize>>` to `SsaOp` (op_id → element indices)
   - Option B: Compute element-wise dependencies during CP-SAT model building from `srcs` and `dep_ids`

2. **Compute element-wise dependencies**:
   - For each VALU op `op` with dependency on `dep_op`:
     - If `op.srcs[0]` or `op.srcs[1]` reads from `dep_op.dest`:
       - Determine which elements are read (standard: element `i` reads element `i`)
       - Create mapping: `op[i]` depends on `dep_op[i]`

#### Phase 2: Modify CP-SAT Constraints

1. **Update ALU dependency constraints** (lines 2012-2043):
   - **Current**: `alu_cycle[VALU2][i] >= completion_cycle[VALU1] + 1` when both shifted
   - **New**: `alu_cycle[VALU2][i] >= alu_cycle[VALU1][j] + 1` where `j` is the element VALU2[i] depends on
   - Use conditional constraints based on shift decisions:
     - If both VALU1 and VALU2 are shifted: use element-wise dependencies
     - If VALU1 is shifted but VALU2 is not: use `completion_cycle[VALU1]`
     - If VALU1 is not shifted: use `cycle_vars[VALU1]`

2. **Handle mixed cases**:
   - If VALU2 depends on multiple shifted VALU ops, each element may have different dependencies
   - If VALU2[i] depends on VALU1[j] and VALU3[k], use: `alu_cycle[VALU2][i] >= max(alu_cycle[VALU1][j], alu_cycle[VALU3][k]) + 1`

3. **Preserve completion_cycle for non-shifted consumers**:
   - Non-shifted VALU ops that depend on shifted VALU ops still use `completion_cycle`
   - This is correct because non-shifted VALU ops execute atomically on VALU engine

#### Phase 3: Edge Cases and Validation

1. **Broadcast handling**:
   - If VALU1 is a broadcast (`vbroadcast`), all elements of VALU2 that read from it depend on the same scalar source
   - If the scalar source is also shifted, track which ALU op produces it

2. **Non-shiftable VALU ops**:
   - `multiply_add` and `vbroadcast` are not shiftable when used as dependencies
   - Handle these in the standard way (wait for completion_cycle)

3. **Validation**:
   - Ensure element-wise dependencies preserve correctness (RAW dependencies)
   - Test with chained VALU shifts: VALU1 → VALU2 → VALU3 (all shifted)
   - Verify that element-wise dependencies improve makespan

## Implementation Steps

### Step 1: Add Element-wise Dependency Tracking

**File**: `scheduler_rs/src/types.rs`

- Add helper function to compute element-wise dependencies from SSA ops
- Function signature: `compute_element_dependencies(op: &SsaOp, dep_op: &SsaOp) -> Vec<usize>`
  - Returns: For each element `i` of `op`, which element `j` of `dep_op` it depends on
  - Default: `i` depends on `i` (same lane)
  - Special cases: broadcasts, constants

### Step 2: Modify CP-SAT Model Building

**File**: `scheduler_rs/src/cpsat_scheduler.rs`

1. **Before creating completion_cycle_vars** (around line 1940):
   - Compute element-wise dependency mappings for all shiftable VALU ops
   - Store as: `element_deps: HashMap<(usize, usize), usize>` where key is `(op_idx, element)` and value is `(dep_op_idx, dep_element)`

2. **Update ALU dependency constraints** (lines 2012-2043):
   - Replace `completion_cycle[dep]` with element-wise `alu_cycle[dep][j]` when both ops are shifted
   - Use conditional constraints: `alu_cycle[op][i] >= alu_cycle[dep][j] + 1` when both shifted
   - Fall back to `completion_cycle[dep]` when dep is shifted but op is not

3. **Keep completion_cycle for non-shifted consumers** (lines 2045-2060):
   - No changes needed here (correct behavior)

### Step 3: Testing

**File**: `scheduler_rs/src/cpsat_scheduler.rs` (test module)

1. **Test chained VALU shifts**:
   - Create VALU1 → VALU2 → VALU3 chain (all shiftable)
   - Verify that ALU2[i] starts after ALU1[i] (not after completion_cycle(VALU1))
   - Verify that ALU3[i] starts after ALU2[i]

2. **Test mixed shifts**:
   - VALU1 shifted, VALU2 not shifted, VALU2 depends on VALU1
   - Verify VALU2 waits for completion_cycle(VALU1)

3. **Test broadcast dependencies**:
   - VALU1 is broadcast, VALU2 depends on VALU1, both shifted
   - Verify all ALU2[i] depend on the same scalar ALU op

### Step 4: Performance Validation

- Run on real kernels (e.g., 2-round, 4-round)
- Compare makespan before/after element-wise dependencies
- Verify correctness (same output as reference kernel)

## Expected Benefits

1. **Improved parallelism**: Parts of VALU2 can start executing before all of VALU1 finishes
2. **Reduced makespan**: Especially beneficial for long chains of VALU operations
3. **Better ALU utilization**: ALU slots can be filled earlier when dependencies allow

## Risks and Considerations

1. **Complexity**: Element-wise dependencies add complexity to the CP-SAT model
2. **Solver performance**: More constraints may slow down CP-SAT solving (need to measure)
3. **Correctness**: Must ensure element-wise dependencies preserve RAW correctness
4. **Edge cases**: Broadcasts, constants, and non-shiftable ops need special handling

## Open Questions

1. **Broadcast handling**: If VALU1 is a broadcast and VALU2 reads from it, should all ALU2[i] depend on the same scalar source, or can we optimize further?
2. **Multiple dependencies**: If VALU2[i] depends on both VALU1[j] and VALU3[k], how do we handle this efficiently?
3. **Solver hints**: Should greedy scheduler hints include element-wise cycle assignments for shifted VALU ops?

## References

- Current implementation: `scheduler_rs/src/cpsat_scheduler.rs` lines 1940-2060
- VALU expansion: `scheduler_rs/src/kernel_builder/ssa_lower.rs` `expand_ssa_valu_to_alu()`
- SSA op structure: `scheduler_rs/src/types.rs` `SsaOp` struct
