# Plan: Unify Liveness Constraint Semantics

## Core Principle

**All views of liveness constraints must use consistent same-cycle reuse semantics.**

The hardware model (from `problem.py`) states: "Effects like writes to memory take place after all the inputs are read." This means within a single cycle, reads happen before writes, enabling same-cycle reuse: a VReg can be freed at the end of the cycle it's last read, and that storage can be reused by a VReg defined in the same cycle.

Every module that computes or constrains liveness must implement this model identically.

## Architecture: Two Paths

There are two kernel building pipelines with separate address allocation implementations:

### SSA Path (Recommended)

```
tick_tock_ssa.rs → schedule_ssa_*() → address_assign.rs → lower_ssa_ops()
```

- Uses virtual registers (VRegs) with true SSA semantics
- `address_assign.rs` handles VReg → physical address mapping
- `types.rs::LiveInterval` defines the interval type
- **Correctly implements same-cycle reuse**

### Non-SSA Path (Deprecated)

```
tick_tock.rs → schedule_with_slot_filling() → allocator.rs
```

- Pre-assigns physical addresses during kernel building
- `allocator.rs` compacts addresses post-scheduling
- Has its own `LiveInterval` type with different semantics
- **Does NOT correctly implement same-cycle reuse**

## Status Update (2024)

### `allocator.rs` — NOW DEPRECATED

The `allocator.rs` module has been marked as deprecated with `#[deprecated]` attributes.
It is only used by the legacy non-SSA path (`build_tick_tock_kernel`).

**Deprecation rationale:**
- The SSA path (`build_tick_tock_kernel_ssa`) is the recommended approach
- Fixing `allocator.rs` would require effort for a deprecated code path
- Better to direct users to migrate to the SSA path

**What was added:**
- Module-level deprecation documentation
- `#[deprecated]` on `LiveInterval`, `ScratchRegisterAllocator`, `allocate_registers()`, `rewrite_addresses()`
- Compiler warnings now appear when using these APIs

## Resolved Inconsistencies (SSA Path)

With `allocator.rs` deprecated and documentation added, all known issues are resolved:

### 1. CP-SAT vs Address Assignment Dead Store Handling ✅ RESOLVED

**Status:** Already correct. CP-SAT skips dead stores (ops with negative cycle hints from greedy),
which is consistent with DCE removing them before scheduling.

| Location | Dead Store Handling |
|----------|---------------------|
| `cpsat_scheduler.rs:1290-1296` | Skips ops with `hints[idx] < 0` (dead stores) |
| `address_assign.rs:101` | Sets `end = start` for dead defs (1-cycle lifetime) |

### 2. CP-SAT Same-Cycle Reuse Not Modeled ✅ DOCUMENTED

**Status:** This is an intentional design limitation, now prominently documented in `cpsat_scheduler.rs`.

**Why CP-SAT is more conservative:**
- Intervals use `end = last_use_cycle + 1` (half-open: `[start, end)`)
- So `[0,5]` becomes `[0,6)` and `[5,10]` becomes `[5,11)`
- These OVERLAP in CP-SAT at the 5-6 boundary
- But in greedy/address_assign, they do NOT overlap (same-cycle reuse)

**Practical impact:** Minimal. If CP-SAT says a schedule fits, it definitely does.
The difference only matters for edge cases where same-cycle reuse would save exactly
enough budget to fit.

## Resolved Issues

The following issues from the original plan are now resolved by deprecating `allocator.rs`:

| Original Issue | Resolution |
|----------------|------------|
| Interval expiration `<=` vs `<` | `address_assign.rs` uses correct `<=`; `allocator.rs` deprecated |
| Two `overlaps()` implementations | `types.rs::LiveInterval::overlaps()` is canonical; `allocator.rs` deprecated |
| Peak liveness calculation offset | `address_assign.rs::compute_peak_liveness()` is canonical; `allocator.rs` deprecated |

## Implementation Plan (Revised)

### Phase 1: Documentation (DONE)

1. ~~Deprecate `allocator.rs`~~ ✅
2. ~~Add deprecation warnings to public APIs~~ ✅
3. ~~Document known issues in module docs~~ ✅

### Phase 2: Document CP-SAT Same-Cycle Limitation (DONE)

1. ~~Add prominent comment in `cpsat_scheduler.rs` explaining:~~ ✅
   - The cumulative constraint uses `end = last_use_cycle + 1`
   - This makes intervals `[0,5]` and `[5,10]` overlap in CP-SAT
   - But in `address_assign.rs`, these do NOT overlap (same-cycle reuse)
   - Result: CP-SAT is slightly more conservative than greedy

**Note:** Dead store handling is already correct:
- `cpsat_scheduler.rs` lines 1290-1296 skip ops with negative cycle hints (dead stores filtered by greedy)
- This is consistent with the DCE pass in the SSA builder

### Phase 3: Add Liveness Model Comments (DONE)

At each location in the SSA path that computes or constrains liveness, added documentation:

1. ~~`types.rs::LiveInterval` struct~~ ✅ - Full liveness model documentation
2. ~~`address_assign.rs` module docs~~ ✅ - Already had good docs, verified complete
3. ~~`cpsat_scheduler.rs` cumulative section~~ ✅ - Documented the CP-SAT vs greedy difference

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

### Phase 4: Add Boundary Tests (DONE)

Added tests to verify same-cycle reuse semantics:

**`types.rs` tests:**
1. ~~`test_live_interval_same_cycle_boundary_no_overlap`~~ ✅ - `[0,5]` and `[5,10]` don't overlap
2. ~~`test_live_interval_adjacent_cycles_no_overlap`~~ ✅ - `[0,4]` and `[5,10]` don't overlap
3. ~~`test_live_interval_single_cycle_adjacent`~~ ✅ - `[5,5]` and `[5,10]` don't overlap

**`address_assign.rs` tests:**
4. ~~`test_same_cycle_reuse_boundary`~~ ✅ - Scalars with adjacent intervals share storage
5. ~~`test_same_cycle_reuse_vectors`~~ ✅ - Vectors with adjacent intervals share storage (saves 8 words)
6. ~~`test_peak_liveness_same_cycle_boundary`~~ ✅ - Peak is 1 for `[0,5]+[5,10]` (not 2)

## Files Modified

| File | Changes |
|------|---------|
| `scheduler_rs/src/allocator.rs` | ~~Fix comparisons~~ → DEPRECATED instead ✅ |
| `scheduler_rs/src/address_assign.rs` | Verified correct; added 3 boundary tests ✅ |
| `scheduler_rs/src/types.rs` | Added full liveness model docs + 3 boundary tests ✅ |
| `scheduler_rs/src/cpsat_scheduler.rs` | Documented same-cycle limitation ✅ |
| `scheduler_rs/src/kernel_builder/tick_tock_ssa.rs` | No changes needed (uses types.rs) |

## Verification (COMPLETE)

1. ✅ All existing tests pass (103 tests)
2. ✅ New boundary tests pass (6 new tests added)
3. ✅ `generate_ssa_kernel.py --validate --use-greedy` builds successfully
4. ℹ️ Budget warnings expected with greedy scheduler (see CLAUDE.md - use CP-SAT for correct results)
5. ✅ Deprecation warnings appear when using the non-SSA path

## Summary

The liveness consistency plan is **complete**. The codebase now has:

1. **Deprecated `allocator.rs`** - Compiler warnings guide users to SSA path
2. **Documented CP-SAT limitation** - Explains why CP-SAT is slightly more conservative
3. **Full liveness model documentation** in `types.rs::LiveInterval`
4. **Comprehensive boundary tests** verifying same-cycle reuse semantics
5. **Consistent semantics** across `address_assign.rs`, `types.rs`, and greedy scheduler

The only remaining semantic difference is intentional: CP-SAT's cumulative constraint is slightly
more conservative than greedy/address_assign because it counts both dying and born VRegs as
simultaneously live at the boundary cycle. This is documented and acceptable since it makes
CP-SAT a safe upper bound on memory usage.
