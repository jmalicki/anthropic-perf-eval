# Plan: Add Critical Missing Tests for SSA Scheduling

## Summary

Code review identified significant gaps in test coverage for SSA scheduling, particularly around budget constraints, liveness computation, and edge cases. These tests are critical because bugs in these areas can silently produce invalid kernels that exceed scratch memory limits.

## Test Categories

### Category 1: Budget Constraint Enforcement

These tests verify that schedulers actually enforce (not just track) scratch budget limits.

#### Test 1.1: CP-SAT Rejects Impossible Budget
```rust
#[test]
fn test_cpsat_rejects_impossible_budget() {
    // Create 3 vectors that MUST be live simultaneously (no valid reordering)
    // v0 = const, v1 = const, v2 = const
    // v3 = add(v0, v1)  // v0, v1, v2 all live here
    // v4 = add(v2, v3)
    //
    // Peak = 3 vectors = 24 words minimum
    // Try budget = 16 (2 vectors)
    // Expected: CP-SAT returns None or significantly increases makespan
}
```

#### Test 1.2: CP-SAT Fails (Not Warns) on Budget Exceeded
```rust
#[test]
fn test_cpsat_fails_not_warns_on_budget_exceeded() {
    // After CP-SAT scheduling + address assignment, if budget_exceeded is true,
    // the result should be an Err, not Ok with a warning
    // Currently: warns but returns success (BUG)
}
```

#### Test 1.3: Greedy Fallback Sets Budget Exceeded Flag
```rust
#[test]
fn test_greedy_fallback_sets_budget_exceeded() {
    // Create schedule that triggers fallback path (budget too tight)
    // Verify budget_exceeded flag is set to true
    // Verify the flag is propagated to final result
}
```

### Category 2: Same-Cycle Reuse Boundary Cases

These tests verify the liveness model is correctly implemented at interval boundaries.

#### Test 2.1: Intervals at Exact Boundary Share Storage
```rust
#[test]
fn test_same_cycle_reuse_exact_boundary() {
    // v0: live [0, 5] (defined cycle 0, last used cycle 5)
    // v1: live [5, 10] (defined cycle 5, last used cycle 10)
    //
    // These should share storage (same-cycle reuse)
    // Peak should be 1 VReg worth, not 2
    // Verify: address_assign gives same address to both
}
```

#### Test 2.2: Overlaps() Returns False at Boundary
```rust
#[test]
fn test_live_interval_no_overlap_at_boundary() {
    let a = LiveInterval::new(0, 5);
    let b = LiveInterval::new(5, 10);
    assert!(!a.overlaps(&b), "Intervals [0,5] and [5,10] should NOT overlap");
    assert!(!b.overlaps(&a), "Overlaps must be symmetric");
}
```

#### Test 2.3: Peak Liveness Correct at Transition
```rust
#[test]
fn test_peak_liveness_at_transition_cycle() {
    // Cycle 5: v0 dies (last read), v1 born (defined)
    // With same-cycle reuse: peak = 1
    // Without same-cycle reuse: peak = 2
    // Verify we get peak = 1
}
```

### Category 3: Address Assignment Correctness

These tests verify that address assignment produces valid, non-overlapping allocations.

#### Test 3.1: No Address Overlap for Concurrent VRegs
```rust
#[test]
fn test_address_assignment_no_overlap() {
    // Create 5+ VRegs with complex overlapping live ranges
    // After assignment, verify:
    // - For every pair of VRegs with overlapping live ranges
    // - Their address ranges [addr, addr+size) are disjoint
}
```

#### Test 3.2: Vector Addresses Are Contiguous
```rust
#[test]
fn test_vector_allocation_contiguous() {
    // Create vector VRegs
    // Verify each gets exactly 8 contiguous addresses
    // Verify no gaps or overlaps with other allocations
}
```

#### Test 3.3: Fixed VRegs Don't Collide
```rust
#[test]
fn test_fixed_vreg_no_collision() {
    // Set ZERO_REG fixed at address 0
    // Allocate other VRegs
    // Verify no other VReg uses address 0
}
```

### Category 4: Dead Store Handling

These tests verify consistent handling of VRegs that are defined but never used.

#### Test 4.1: Dead Store Excluded from Liveness
```rust
#[test]
fn test_dead_store_not_in_liveness() {
    // v0 = const(5)  // defined but never used
    // v1 = const(6)
    // v2 = add(v1, v1)
    //
    // v0 should be eliminated by DCE or not counted in liveness
    // Peak should only count v1, v2
}
```

#### Test 4.2: Dead Store with Tight Budget
```rust
#[test]
fn test_dead_store_tight_budget() {
    // If dead stores ARE counted in liveness (bug), this test fails
    // Create dead stores that would exceed budget if counted
    // Verify scheduling succeeds (dead stores not counted)
}
```

#### Test 4.3: CP-SAT and Address Assignment Agree on Dead Stores
```rust
#[test]
fn test_dead_store_consistency() {
    // CP-SAT peak and address assignment peak should match
    // for programs with dead stores
    // Currently: they may differ (BUG)
}
```

### Category 5: Budget Exceeded Flag Accuracy

These tests verify the budget_exceeded flag is set correctly at boundary values.

#### Test 5.1: Exact Budget Boundary
```rust
#[test]
fn test_budget_exactly_met() {
    // Create schedule with peak_live = 100 exactly
    // budget = 100: budget_exceeded should be false
    // budget = 99: budget_exceeded should be true
}
```

#### Test 5.2: Budget Flag Propagation
```rust
#[test]
fn test_budget_exceeded_propagates() {
    // Verify budget_exceeded from greedy scheduler
    // is combined with budget_exceeded from address assignment
    // Final result reflects either source
}
```

### Category 6: Multi-Reader Edge Cases

These tests verify correct handling of VRegs read by multiple operations.

#### Test 6.1: Multiple Readers Same Cycle
```rust
#[test]
fn test_multiple_readers_same_cycle() {
    // v0 = const(5)
    // v1 = add(v0, v0)  // reads v0 twice in same cycle
    // v2 = mul(v0, v0)  // also reads v0 twice, same cycle as v1
    //
    // last_use_cycle[v0] should be cycle of v1/v2
    // Verify liveness is correct
}
```

#### Test 6.2: Reader Order Independence
```rust
#[test]
fn test_last_use_order_independent() {
    // Two ops read same VReg at different cycles
    // Regardless of emission order, last_use should be later cycle
}
```

### Category 7: Large Scale Stress Tests

These tests verify behavior at realistic scale.

#### Test 7.1: Near Budget Limit
```rust
#[test]
fn test_schedule_near_budget_limit() {
    // Create kernel that uses ~1500 words (near 1536 limit)
    // Verify scheduling succeeds
    // Verify peak_live < 1536
}
```

#### Test 7.2: Many Small Intervals
```rust
#[test]
fn test_many_short_lived_vregs() {
    // 1000+ scalars, each live for 1-2 cycles
    // Verify address reuse works correctly
    // Verify peak is much less than total VRegs
}
```

## Implementation Priority

### Tier 1: Safety Critical (implement first)
- Test 1.2: CP-SAT fails not warns
- Test 2.1: Same-cycle reuse boundary
- Test 3.1: No address overlap
- Test 5.1: Exact budget boundary

### Tier 2: Correctness (implement second)
- Test 1.1: CP-SAT rejects impossible budget
- Test 2.2: Overlaps() boundary
- Test 4.3: Dead store consistency
- Test 6.1: Multiple readers same cycle

### Tier 3: Robustness (implement third)
- Test 1.3: Greedy fallback flag
- Test 2.3: Peak at transition
- Test 3.2: Vector contiguous
- Test 3.3: Fixed VReg collision
- Test 4.1, 4.2: Dead store handling
- Test 5.2: Flag propagation
- Test 6.2: Reader order independence
- Test 7.1, 7.2: Large scale

## Files to Modify

| File | Tests to Add |
|------|--------------|
| `scheduler_rs/src/cpsat_scheduler.rs` | 1.1, 1.2, 4.3, 6.1, 6.2 |
| `scheduler_rs/src/address_assign.rs` | 2.1, 2.3, 3.1, 3.2, 3.3, 5.1 |
| `scheduler_rs/src/types.rs` | 2.2 |
| `scheduler_rs/src/scheduler.rs` | 1.3, 5.2, 7.1, 7.2 |
| `scheduler_rs/src/kernel_builder/tick_tock_ssa.rs` | 4.1, 4.2 |

## Test Naming Convention

All new tests should follow the pattern:
```
test_<module>_<what>_<condition>
```

Examples:
- `test_cpsat_budget_rejects_impossible`
- `test_address_overlap_at_boundary_forbidden`
- `test_liveness_peak_same_cycle_reuse`

## Verification

After implementing tests:
1. All new tests pass with current code (except tests that expose known bugs)
2. **Optional/CI-only tests:** Use a **Cargo feature** (`ci-tests`) so they run only when requested. Gate with `#[cfg(feature = "ci-tests")]`; run with `cargo test --features "cpsat ci-tests"`. CI should use this for full coverage. Do not use `#[ignore]` for optional tests.
3. **Tests that expose known bugs:** Mark `#[ignore]` with a comment and issue reference; when the bug is fixed, remove `#[ignore]` and verify the test passes.
4. Run full test suite: from project root, `BUILD_FROM_PROJECT_ROOT=1 cargo test --manifest-path scheduler_rs/Cargo.toml --features cpsat`. For optional tests: `--features "cpsat ci-tests"`.
