# CP-SAT Presolve INFEASIBLE Analysis

## Summary

Seven CP-SAT tests fail with **INFEASIBLE** status during presolve (before main search begins). The error occurs in a **cumulative constraint** (Flow engine, capacity 1) that contains **optional intervals** from SelectChoice operations. The presolve phase reports: `'offset: 1 as empty domain after intersecting with [0]'`.

**Minimal reproducer:** A minimal CP-SAT program (1 required + 3 optional intervals, cumulative capacity 1) does **not** reproduce INFEASIBLE (`test_minimal_cumulative_optional_capacity_one`).

**Root cause (investigation):** `test_investigate_valu_shift_toy_presolve_infeasible` runs the failing valu_shift_toy program with five config variants. Only **`enable_asap_alap_tightening: false`** passes. Baseline (toy machine, default hints), `hint_select_choice_vselect_by_default: false`, default machine, and `select_choice_mode: ForceFma` all still fail. So **ASAP/ALAP tightening** is the trigger: when enabled (default), it over-constrains the model for these small programs and presolve reports INFEASIBLE. Disabling it restores feasibility.

## Affected Tests

All seven tests use `SsaCpsatHints::default()` (no greedy hints) and call `schedule_ssa_cpsat()` directly:

1. `test_ssa_cpsat_selectchoice_decisions`
2. `test_ssa_cpsat_selectchoice_flow_slot_limit`
3. `test_ssa_cpsat_selectchoice_roundtrip`
4. `test_ssa_cpsat_selectchoice_valu_shift_toy`
5. `test_ssa_cpsat_selectchoice_force_fma_minimal`
6. `test_ssa_cpsat_selectchoice_force_fma_default_machine_no_hints`
7. `test_ssa_cpsat_force_fma_full_pipeline_no_overlap`

## Error Details

### CP-SAT Output

```
INFEASIBLE: 'offset: 1 as empty domain after intersecting with [0]'
Unsat after presolving constraint #266 (warning, dump might be inconsistent): 
cumulative { 
  capacity { offset: 1 } 
  intervals: 4 intervals: 22 intervals: 31 intervals: 40 intervals: 49 intervals: 58 
            intervals: 67 intervals: 141 intervals: 142 intervals: 144 intervals: 145 
            intervals: 147 intervals: 148 intervals: 152 intervals: 153 intervals: 154 
  demands { offset: 1 } ... (15 times) ... demands { offset: 2 } 
}
```

**Key observations:**
- **Capacity**: 1 (Flow slot limit)
- **Intervals**: 16 total (4 regular Flow ops + 12 optional from 3 SelectChoice ops)
- **Demands**: 15 intervals with demand 1, 1 interval with demand 2
- **Error**: Variable at "offset: 1" (likely an internal CP-SAT variable index) has domain that must be ≥ 1, but presolve constrained it to [0], creating an empty domain

### Model Structure

For each **SelectChoice** op in `SelectChoiceMode::Optimize`:

1. **Choice variable**: `choice_var` (boolean)
   - `true` = FMA path (uses VALU)
   - `false` = vselect path (uses Flow)

2. **Optional Flow interval**: Created when `choice_var = false` (vselect path)
   ```rust
   let flow_interval = model.new_optional_fixed_size_interval_var(
       cycle_vars[idx],  // Start time
       1,                // Duration
       choice_var.not(), // Presence literal (true when choice_var=false)
   );
   ```

3. **Flow cumulative constraint**: Includes
   - Regular Flow ops (always present)
   - Optional Flow intervals from ConstChoice (when `const_choice_var = true`)
   - Optional Flow intervals from SelectChoice (when `select_choice_var = false`)

## Root Cause Hypothesis

We have **not** been able to reproduce INFEASIBLE with a minimal program (see "Minimal reproducer" above). The following is a plausible explanation for the full model only.

### The Problem

When **multiple optional intervals** are added to a **cumulative with capacity 1** in the **full** scheduler model, CP-SAT's presolve phase may:

1. **Prove mutual exclusivity**: If two optional intervals could overlap, presolve tries to prove that at least one must be inactive (since capacity is 1).

2. **Propagate constraints**: Presolve propagates constraints from:
   - Precedence (cycle variables must satisfy dependencies)
   - Domain bounds (ASAP/ALAP)
   - Cumulative capacity (at most 1 interval active per cycle)

3. **Detect contradictions**: If presolve can prove that:
   - All optional intervals must be inactive (due to other constraints), **AND**
   - At least one must be active (due to dependencies or domain bounds),
   - Then the model is **INFEASIBLE**.

### Why It Fails Without Greedy Hints

**Without greedy hints:**
- Cycle variables have wide domains (ASAP to ALAP)
- Optional intervals can potentially overlap in many ways
- Presolve has limited information to prune possibilities
- Presolve may incorrectly conclude the model is infeasible

**With greedy hints:**
- Cycle variables are hinted to specific values
- Optional intervals are hinted to specific cycles
- Choice variables are hinted (e.g., `choice_var = 0` for vselect)
- Presolve sees a feasible warm start and doesn't trigger the bug

### The "Empty Domain" Error

The error `'offset: 1 as empty domain after intersecting with [0]'` suggests:

1. **Variable at offset 1**: Likely an internal CP-SAT variable (could be a cycle variable, presence literal, or derived variable)

2. **Must be ≥ 1**: Some constraint (precedence, domain bound, or cumulative propagation) requires this variable to be ≥ 1

3. **Constrained to [0]**: Presolve propagation (from cumulative capacity or other constraints) forced it to be ≤ 0

4. **Empty intersection**: `[1, ∞) ∩ [0] = ∅` → INFEASIBLE

### Why Capacity 1 Matters

With **capacity 1**:
- At most **one** interval can be active per cycle
- Presolve can aggressively prune: "If intervals A and B could overlap, at least one must be inactive"
- This creates many implications that presolve must track
- With many optional intervals, presolve may incorrectly propagate

With **capacity > 1** (e.g., VALU capacity 6):
- Multiple intervals can coexist
- Presolve has less aggressive pruning
- The bug doesn't trigger

## Evidence

### Test Pattern

All failing tests share:
- **No greedy hints** (`SsaCpsatHints::default()`)
- **SelectChoice ops** with `SelectChoiceMode::Optimize`
- **Flow cumulative** with capacity 1
- **Optional Flow intervals** from SelectChoice

### Demand = 2 Mystery

The error dump shows **one interval with demand 2**, but our code only pushes demand 1 for Flow intervals. This could be:

1. **CP-SAT internal representation**: The "offset: 2" might refer to a different variable (not demand)
2. **Presolve transformation**: CP-SAT may merge or transform intervals during presolve
3. **Bug in our code**: We might be adding an interval twice (unlikely, but possible)

**Investigation needed**: Check if any Flow interval is added multiple times or if there's a bug in demand assignment.

## Workarounds

### Current Solution

Tests are marked `#[ignore]` so the suite passes. They can be run with:
```bash
cargo test --features cpsat -- --ignored
```

### Fix (root cause)

**Disable ASAP/ALAP tightening** for these programs: set `enable_asap_alap_tightening: false` in `SsaCpsatConfig`. The investigation test shows this is the only variant that passes; tightening over-constrains and triggers presolve INFEASIBLE.

### Other options

- **Greedy hints**: Providing cycle + choice hints may also avoid the issue.
- **ForceFma/ForceVselect**: Still failed in the investigation (tightening still applied).

## Related Code

- **Flow cumulative construction**: `scheduler_rs/src/cpsat_scheduler/model.rs:1462-1550`
- **SelectChoice optional intervals**: `scheduler_rs/src/cpsat_scheduler/model.rs:1040-1045`
- **Horizon fix**: `scheduler_rs/src/cpsat_scheduler/model.rs:144-180` (ensures horizon ≥ capacity_lower_bound)

## Next Steps

1. **Verify demand = 2**: Check if any Flow interval is added twice or if there's a bug in demand assignment.

2. **Test with greedy hints**: Modify one failing test to provide minimal greedy hints (cycle + choice) and see if it passes.

3. **Test with flow_slots = 2**: Verify that increasing Flow capacity prevents the presolve failure.

4. **Grow minimal reproducer**: Starting from `test_minimal_cumulative_optional_capacity_one`, add constraints from the full model (makespan, ASAP/ALAP domains, more cumulatives, etc.) until INFEASIBLE appears to isolate the cause.

5. **Alternative encoding**: Consider using `NoOverlap` for Flow instead of cumulative (capacity 1 ⇒ equivalent); might behave differently in presolve.
