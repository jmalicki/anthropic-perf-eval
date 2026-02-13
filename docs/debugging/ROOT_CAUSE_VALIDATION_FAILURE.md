# Root Cause: Validation Failure in 4-Round Kernel

## Summary

The validation failure is caused by a bug in instruction bundle construction where Op 121's `vselect` instruction (scheduled at cycle 20) is incorrectly placed in cycle 21's instruction bundle instead of cycle 20's bundle. This causes both Op 121 and Op 122 to write to address 102 at cycle 21, violating VLIW semantics and producing incorrect values.

## Detailed Analysis

### The Problem

1. **Op 121** (vselect): Scheduled at cycle 20, writes to address 102
2. **Op 122** (XOR): Scheduled at cycle 21, writes to address 102
3. **Instruction bundle at cycle 21**: Contains Op 121's vselect instead of Op 122's XOR
4. **Result**: Both ops write to address 102 at cycle 21, causing incorrect values

### Evidence

From `sched_4r_8b_d3_trace.json`:
- Op 121: `cycle=20, engine=flow, opcode=vselect, writes=[102]`
- Op 122: `cycle=21, engine=valu, opcode=^, writes=[102, 103, 104, 105, 106, 107, 108, 109]`

From `kernel_to_validate.json`:
- Cycle 20 bundle: `valu: [['==', 286, 246, 21]]` (no vselect)
- Cycle 21 bundle: `flow: [['vselect', 102, 286, 254, 102]]` (Op 121's vselect, WRONG!)

Expected:
- Cycle 20 bundle: Should contain Op 121's vselect
- Cycle 21 bundle: Should contain Op 122's XOR

### Impact

1. **VReg 122 (Round 1 tree_access)** gets the wrong value because Op 122's XOR doesn't execute
2. **Round 1 XOR** computes incorrectly because it reads wrong VReg 122 value
3. **Final output** is wrong for batch indices 0, 2, 4, 5, 6, 7

### Root Cause Location

The bug is in instruction bundle construction, likely in:
- `scheduler_rs/src/python.rs::emit_scheduled_with_expanded()` (lines 237-262)
- Or in how cycles are assigned/stored in lowered ops

The `emit_scheduled_with_expanded` function groups ops by cycle:
```rust
for cycle in 0..max_cycle {
    for op in ops {
        if op.cycle == cycle as i32 {
            emit_op_to_dict(py, op, &instr)?;
        }
    }
}
```

But somehow Op 121 (cycle 20) ends up in cycle 21's bundle. This suggests:
1. Op 121's cycle field is incorrectly set to 21 after lowering
2. Or there's a type mismatch in cycle comparison
3. Or the bundle construction logic has a bug

### Next Steps

1. Check if Op 121's cycle is modified after lowering
2. Verify cycle comparison logic in `emit_scheduled_with_expanded`
3. Check if there's a cycle offset bug
4. Fix the bundle construction to ensure ops are placed in the correct cycle bundles
