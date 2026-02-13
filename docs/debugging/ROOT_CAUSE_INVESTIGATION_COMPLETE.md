# Root Cause Investigation - COMPLETE

## Summary

**Root cause identified and fixed:** Address assignment bug allowing overlapping VRegs to share addresses.

## Root Cause

The expiration logic in `assign_addresses()` incorrectly allowed same-cycle address reuse when:
- VReg A's interval ends at cycle C (last read at C)
- VReg B's interval starts at cycle C (written at C)

Even though VLIW semantics guarantee reads happen before writes, both VRegs are live at cycle C and must use different addresses. The bug caused 617 address conflicts in the 4-round kernel.

## Fix

Changed expiration condition from:
```rust
end_cycle < interval.start || (end_cycle == interval.start && !freed_was_point)
```

To:
```rust
end_cycle < interval.start  // Strictly less than
```

This ensures overlapping intervals (even at a single cycle boundary) get different addresses.

## Verification

- ✅ No address conflicts found (was 617 conflicts)
- ✅ Kernel validation passes (was failing at batch index 1)
- ✅ Correct hash values computed

## Files Modified

- `scheduler_rs/src/address_assign.rs` - Fixed expiration logic (lines 735-759)

## Investigation Tools Created

1. `scripts/trace_exact_divergence.py` - Traces exact divergence point using trace IDs
2. `scripts/check_address_conflicts.py` - Detects address conflicts in scheduler output
3. `scripts/trace_batch_execution.py` - Traces single batch through execution

## Next Steps

The remaining phases from the investigation plan are no longer needed since the root cause has been identified and fixed. However, the tools created may be useful for future debugging.
