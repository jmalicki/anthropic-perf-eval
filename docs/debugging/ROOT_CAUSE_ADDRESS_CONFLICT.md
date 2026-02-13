# Root Cause: Address Conflict Bug

## Summary

**Root cause identified:** The address assignment algorithm incorrectly allows two VRegs with overlapping live intervals to share the same address when one is read and the other is written at the same cycle.

## The Bug

### Critical Conflict Example

- **VReg 115** (Round 0 hash output): interval `[15, 22]` at address 94
- **VReg 123** (Round 0 XOR output): interval `[22, 23]` at address 94

Both VRegs share address 94, but:
- VReg 115 is **read** at cycle 22 (last use)
- VReg 123 is **written** at cycle 22 (definition)

### Why This Causes Wrong Values

Even though VLIW semantics guarantee reads happen before writes within a cycle, when both VRegs share the same address:

1. At cycle 22: Read VReg 115 from address 94 ✓
2. At cycle 22: Write VReg 123 to address 94 ✗ (overwrites the value)

The problem is that **both VRegs need to be live at cycle 22**, so they cannot share the same address. The expiration logic incorrectly expires VReg 115 when allocating VReg 123 because `end_cycle (22) == start (22)`, but this violates the constraint that overlapping intervals must use different addresses.

### Expiration Logic Bug

The expiration check in `assign_addresses()` (line 743-749):

```rust
let expire = if interval.start == interval.end {
    end_cycle < interval.start
} else {
    end_cycle < interval.start
        || (end_cycle == interval.start && !freed_was_point)
};
```

When allocating VReg 123 (`[22, 23]`):
- VReg 115 has `end_cycle = 22`
- `end_cycle == interval.start` (22 == 22) ✓
- `freed_was_point = false` (VReg 115 is `[15, 22]`, not a point interval) ✓
- So `expire = true` ✓

This expires VReg 115 and reuses its address, but **VReg 115 is still read at cycle 22**, so the write overwrites the value before the read completes.

## Impact

The `check_address_conflicts.py` script found **617 address conflicts** in the 4-round kernel, indicating this is a systemic bug affecting many VRegs.

## Fix Required

The expiration logic must be updated to ensure that:
- If VReg A's interval ends at cycle C (last read at C)
- And VReg B's interval starts at cycle C (written at C)
- Then VReg A should **NOT** be expired when allocating VReg B
- They must use **different addresses**

The fix should change the expiration condition to:
- Expire VReg A only if `A.end < B.start` (strictly less than, not less than or equal)

This ensures that overlapping intervals (even at a single cycle boundary) get different addresses.

## Validation

The bug manifests as:
- Wrong hash values computed in Round 1 (and subsequent rounds)
- Validation failures at batch index 1 (and other batch indices)
- Final output values don't match reference kernel

## Files Affected

- `scheduler_rs/src/address_assign.rs` - `assign_addresses()` function, expiration logic (lines 743-759)
