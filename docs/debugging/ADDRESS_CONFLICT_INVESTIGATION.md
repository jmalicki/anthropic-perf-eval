# Address Conflict Investigation

## Summary

**Conclusion: Address conflicts are NOT the root cause of validation failures.**

The user correctly pointed out that **same-cycle reuse is valid** according to VLIW semantics:
- Reads happen before writes within a cycle
- If VReg A ends at cycle C (last read) and VReg B starts at cycle C (written), they can share addresses
- The original expiration logic `end_cycle <= interval.start` correctly allows this

## Investigation Results

1. **Fixed conflict detection script**: Updated to exclude valid same-cycle reuse cases
2. **Result**: After excluding same-cycle reuse, **no address conflicts found**
3. **Conclusion**: The address assignment algorithm is correct

## Current State

- **Address assignment**: Correctly handles same-cycle reuse
- **Conflict detection**: Reports 0 conflicts (after excluding same-cycle reuse)
- **Validation**: Still fails, but NOT due to address conflicts

## The Real Question

If address assignment is correct, why does validation fail?

## Possible Root Causes (to investigate)

1. **Liveness interval computation bug**: Intervals might be computed incorrectly
2. **Read-after-write timing bug**: A VReg might be read after it's supposed to expire
3. **Lowering bug**: The bug might be in how SSA ops are lowered to concrete instructions
4. **Execution bug**: Values might be computed incorrectly during execution
5. **Validation bug**: The validation logic itself might have a bug

## Next Steps

1. Continue investigating why validation fails
2. Use `trace_exact_divergence.py` to find where values diverge
3. Check if the bug is in lowering, execution, or validation logic
