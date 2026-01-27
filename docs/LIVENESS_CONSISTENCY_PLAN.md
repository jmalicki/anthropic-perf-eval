# Plan: Unify Liveness Constraint Semantics

## Core Principle

**All views of liveness constraints must use consistent same-cycle reuse semantics.**

The hardware model (from `problem.py`) states: "Effects like writes to memory take place after all the inputs are read." This means within a single cycle, reads happen before writes, enabling same-cycle reuse: a VReg can be freed at the end of the cycle it's last read, and that storage can be reused by a VReg defined in the same cycle.

Every module that computes or constrains liveness must implement this model identically.

## Current Inconsistencies Found

### 1. Interval Expiration Comparison (`<=` vs `<`)

**Problem:** Different modules use different comparisons for interval expiration.

| Location | Comparison | Semantics |
|----------|------------|-----------|
| `address_assign.rs:246` | `end_cycle <= interval.start` | VReg ending at cycle 5 expires when new interval starts at cycle 5 or later |
| `allocator.rs:271` | `end_cycle < interval.start` | VReg ending at cycle 5 expires when new interval starts at cycle 6 or later |

**Impact:** Intervals `[0,5]` and `[5,10]` may or may not share storage depending on which module checks.

**Fix:** Standardize on `<=` (same-cycle reuse allowed). Update `allocator.rs` to match `address_assign.rs`.

### 2. Overlaps() Implementation Conflict

**Problem:** Two different `overlaps()` implementations exist.

| Location | Logic |
|----------|-------|
| `types.rs:1244` | `self.start < other.end && other.start < self.end` |
| `allocator.rs` | Different logic (needs verification) |

**Fix:** Ensure both use the same exclusive-end semantics. Intervals `[0,5]` and `[5,10]` should NOT overlap.

### 3. Peak Liveness Calculation Offset

**Problem:** Different modules calculate peak liveness with different offsets.

| Location | Approach |
|----------|----------|
| `address_assign.rs:152-173` | No +1 offset for deallocation cycle |
| `allocator.rs` | Adds +1 to deallocation cycle |

**Fix:** Standardize the event ordering model (births before deaths at same cycle for conservative peak).

### 4. CP-SAT vs Address Assignment Dead Store Handling

**Problem:** CP-SAT excludes dead stores from scratch constraint, but address assignment includes them.

| Location | Dead Store Handling |
|----------|---------------------|
| `cpsat_scheduler.rs:1340` | Filters out VRegs with no uses |
| `address_assign.rs:102-126` | Includes all VRegs |

**Fix:** Either:
- Include dead stores in CP-SAT constraint (set `last_use_idx = def_idx` for dead stores, live for 1 cycle), OR
- Explicitly document that dead stores are excluded and ensure DCE removes them before scheduling

### 5. CP-SAT Same-Cycle Reuse Not Modeled

**Problem:** CP-SAT's cumulative constraint counts both the dying and born VRegs as live during the transition cycle, requiring budget >= 2 where greedy needs only 1.

**Fix:** This is a known design limitation. Document it prominently in CP-SAT code and consider:
- Adjusting the budget passed to CP-SAT to account for this
- Or accepting that CP-SAT produces slightly more conservative schedules

## Implementation Plan

### Phase 1: Unify Comparison Semantics

1. **Update `allocator.rs`** to use `<=` for expiration (match `address_assign.rs`)
2. **Add comments** at each location explaining the same-cycle reuse model:
   ```rust
   // Same-cycle reuse model: A VReg ending at cycle C can share storage
   // with a VReg starting at cycle C, because reads happen before writes
   // within the same cycle. Use `<=` for expiration checks.
   ```

### Phase 2: Unify Overlaps Logic

1. **Verify `allocator.rs` overlaps()** matches `types.rs`
2. **Add test** for boundary case: `[0,5].overlaps([5,10])` must be `false`

### Phase 3: Unify Peak Liveness Calculation

1. **Document the event ordering** model at both locations
2. **Add assertion** that greedy peak <= address assignment peak (sanity check)

### Phase 4: Fix Dead Store Handling

1. **Option A (Preferred):** Ensure DCE removes all dead stores before scheduling
2. **Option B:** Include dead stores in CP-SAT with 1-cycle lifetime
3. **Add test** for dead stores with tight budget

### Phase 5: Add Missing Tests

Priority tests to add:
1. `test_same_cycle_reuse_boundary` - intervals `[0,5]` and `[5,10]` share storage
2. `test_address_assignment_no_overlap` - verify all assigned addresses are disjoint
3. `test_cpsat_budget_enforcement` - verify CP-SAT fails (not warns) on impossible budget
4. `test_dead_store_liveness` - verify dead store handling is consistent

### Phase 6: Add Clarifying Comments

At each location that computes or constrains liveness, add a comment block:

```rust
// LIVENESS MODEL: Same-Cycle Reuse
// ================================
// Hardware semantics: reads before writes within a cycle.
// A VReg live during [def_cycle, last_use_cycle] can share storage
// with another VReg starting at last_use_cycle (same-cycle reuse).
//
// Comparison semantics:
// - Expiration: use `end <= start` (VReg ending at 5 is expired by cycle 5)
// - Overlap: intervals [a,b) and [b,c) do NOT overlap
// - Peak: count births before deaths at same cycle (conservative)
```

## Files to Modify

| File | Changes |
|------|---------|
| `scheduler_rs/src/allocator.rs` | Fix `<=` comparison, unify overlaps() |
| `scheduler_rs/src/address_assign.rs` | Add model documentation comments |
| `scheduler_rs/src/types.rs` | Add model documentation to LiveInterval |
| `scheduler_rs/src/cpsat_scheduler.rs` | Document same-cycle limitation, fix dead stores |
| `scheduler_rs/src/kernel_builder/tick_tock_ssa.rs` | Add model documentation |

## Verification

After fixes:
1. All existing tests pass
2. New boundary tests pass
3. `generate_ssa_kernel.py --validate` produces correct output
4. No "Budget Exceeded" warnings when using CP-SAT with default settings
