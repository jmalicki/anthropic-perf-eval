# VALU Shift Vbroadcast Dependency Optimization Plan

**Implementation status**: Not started. (Update when in progress or done; link PRs/commits as needed.)

## Problem Statement

Currently, when a VALU operation depends on a `vbroadcast` and is shifted to ALU, the expanded ALU ops wait for the `vbroadcast` to complete. However, since `vbroadcast` simply copies a scalar value to all 8 vector lanes, the ALU ops can depend directly on the scalar source instead.

Additionally, **vbroadcast is only emitted if it has at least one non-ALU-shifted consumer**. If all consumers are shifted to ALU, vbroadcast doesn't need to be emitted at all, allowing CP-SAT to optimize it away.

**Current behavior:**

- **vbroadcast(scalar)** → writes `dest[0..7] = scalar` (all lanes same value)
- **VALU op** depends on vbroadcast → reads from `dest[0..7]`
- **VALU shifted to ALU** → expands to 8 ALU ops: `ALU[i]` reads `dest[i]`
- **Dependency**: All 8 ALU ops wait for vbroadcast to complete
- **Vbroadcast emission**: Always emitted regardless of consumer shift decisions

**Desired behavior:**

- **VALU shifted to ALU** → expands to 8 ALU ops
- **Dependency**: All 8 ALU ops depend directly on the scalar source that vbroadcast depends on
- **Vbroadcast emission**: Only if at least one consumer is NOT shifted to ALU
- **CP-SAT optimization**: Model vbroadcast as optional when all consumers are shifted
- **Benefits**: 
  - ALU ops can start as soon as the scalar is ready, without waiting for vbroadcast
  - CP-SAT can eliminate vbroadcast entirely when not needed, saving VALU slots

## Current Implementation

### CP-SAT Model (scheduler_rs/src/cpsat_scheduler.rs)

**ALU dependency constraints** (lines 2012-2043):

- When VALU op depends on another op and is shifted to ALU:
  - Each of the 8 ALU ops waits for the dependency's completion
  - If dependency is shiftable VALU: waits for `completion_cycle` or `cycle_vars` depending on shift decision
  - If dependency is non-shiftable (like vbroadcast): waits for `cycle_vars[dep_idx]`

**Vbroadcast modeling** (lines 1967-1969):

- Vbroadcast ops are modeled with optional intervals (like shiftable VALU ops)
- However, there's no logic to make vbroadcast optional based on consumer shift decisions
- Vbroadcast is always scheduled if present in the ops list

**Key observation**: The code doesn't distinguish vbroadcast dependencies from other dependencies. All dependencies are treated the same way, and vbroadcast emission is not conditional on consumer shift decisions.

### SSA Op Structure (scheduler_rs/src/types.rs)

- `SsaOp` has:
  - `srcs: Vec<Operand>` - source operands (VRegs or immediates)
  - `dep_ids: HashSet<usize>` - op IDs this op depends on (RAW dependencies)
- For vbroadcast:
  - `srcs[0]` = scalar VReg (the value to broadcast)
  - `dep_ids` contains the op that produces the scalar VReg
- For VALU op that depends on vbroadcast:
  - `dep_ids` contains the vbroadcast op ID
  - When shifted to ALU, currently waits for vbroadcast

### VALU Expansion (scheduler_rs/src/kernel_builder/ssa_lower.rs)

**`expand_ssa_valu_to_alu()`** (lines 372-430):

- Expands shifted VALU op to 8 ALU ops
- Copies `ssa_op.dep_ids` directly to each expanded ALU op
- No special handling for vbroadcast dependencies

## Solution Design

### Part 1: Dependency Rewriting Strategy

When a VALU op is shifted to ALU and depends on vbroadcast:

1. **Identify vbroadcast dependencies**:
   - For each `dep_id` in the VALU op's `dep_ids`:
     - Check if `ops[dep_idx].opcode == Opcode::Vbroadcast`
     - If yes, this is a vbroadcast dependency

2. **Find scalar source**:
   - vbroadcast has `srcs[0]` = scalar VReg
   - Find the op that produces this scalar: look up in `dep_ids` of vbroadcast
   - This is the "real" dependency we should use

3. **Rewrite dependencies**:
   - Replace vbroadcast dependency with scalar producer dependency
   - All 8 ALU ops depend on the scalar producer instead of vbroadcast

### Part 2: Conditional Vbroadcast Emission

**Key insight**: If all consumers of vbroadcast are shifted to ALU, vbroadcast doesn't need to be emitted at all.

1. **Track consumer shift decisions**:
   - For each vbroadcast op, identify all ops that depend on it (consumers)
   - Check if each consumer is shiftable and whether it will be shifted
   - Count how many consumers are shifted vs not shifted

2. **Make vbroadcast optional**:
   - If all consumers are shifted to ALU: vbroadcast is optional (can be eliminated)
   - If at least one consumer is NOT shifted: vbroadcast is required
   - Model this in CP-SAT using optional intervals with activation conditions

3. **CP-SAT constraint**:
   - Create boolean variable: `vbroadcast_needed[vbroadcast_idx]`
   - `vbroadcast_needed[vbroadcast_idx] = true` if at least one consumer is NOT shifted
   - `vbroadcast_needed[vbroadcast_idx] = false` if all consumers are shifted
   - Make vbroadcast interval optional based on `vbroadcast_needed`

### Implementation Approach

#### Option A: Rewrite in CP-SAT constraints (recommended)

**Location**: `scheduler_rs/src/cpsat_scheduler.rs` in ALU dependency constraints (lines 2012-2043) and vbroadcast modeling (around line 1967)

**Steps**:

1. **Build vbroadcast consumer analysis** (before creating intervals):
   - For each vbroadcast op, find all ops that depend on it
   - Determine which consumers are shiftable
   - Track shift decisions for shiftable consumers

2. **Make vbroadcast optional**:
   - Create `vbroadcast_needed` boolean variables
   - Set constraints: `vbroadcast_needed[vb_idx] = OR(shift[consumer] == false)` for all consumers
   - Make vbroadcast interval optional based on `vbroadcast_needed`

3. **Rewrite ALU dependencies**:
   - When adding dependency constraints for ALU ops:
     - Check if `dep_id` is a vbroadcast
     - If yes, find scalar producer and use that instead
     - Otherwise, use original dependency

**Pros**:
- No changes to dependency structure
- Only affects constraint generation
- Works seamlessly with existing code
- Allows CP-SAT to optimize away vbroadcast when not needed

**Cons**:
- Need to look up vbroadcast info during constraint building
- More complex constraint logic

## Implementation Steps

### Step 1: Build Vbroadcast Analysis

**File**: `scheduler_rs/src/cpsat_scheduler.rs`

**Location**: Before creating ALU cycle variables (around line 1940)

**Code**:

```rust
// Build mapping: vbroadcast_op_id -> scalar_producer_op_id
let mut vbroadcast_scalar_deps: HashMap<usize, usize> = HashMap::new();

// Build mapping: vbroadcast_op_id -> list of consumer op indices
let mut vbroadcast_consumers: HashMap<usize, Vec<usize>> = HashMap::new();

for (idx, op) in ops.iter().enumerate() {
    if op.opcode == Opcode::Vbroadcast {
        // Find scalar producer
        if let Some(Operand::VReg(scalar_vreg)) = op.srcs.first() {
            for (other_idx, other_op) in ops.iter().enumerate() {
                if other_op.dest == Some(*scalar_vreg) {
                    vbroadcast_scalar_deps.insert(op.id, other_op.id);
                    break;
                }
            }
        }
        
        // Find consumers (ops that depend on this vbroadcast)
        let mut consumers = Vec::new();
        for (consumer_idx, consumer_op) in ops.iter().enumerate() {
            if consumer_op.dep_ids.contains(&op.id) {
                consumers.push(consumer_idx);
            }
        }
        vbroadcast_consumers.insert(op.id, consumers);
    }
}
```

### Step 2: Make Vbroadcast Optional Based on Consumer Shifts

**File**: `scheduler_rs/src/cpsat_scheduler.rs`

**Location**: Around line 1967 (where vbroadcast intervals are created)

**Changes**:

- For each vbroadcast op:
  - Create `vbroadcast_needed` boolean variable
  - Set constraint: `vbroadcast_needed = OR(shift[consumer] == false)` for all shiftable consumers
  - Make vbroadcast interval optional based on `vbroadcast_needed`

**Code**:

```rust
// Create optional intervals for vbroadcast ops
let mut vbroadcast_needed_vars: HashMap<usize, BoolVar> = HashMap::new();

for (idx, op) in ops.iter().enumerate() {
    if op.opcode == Opcode::Vbroadcast {
        let vbroadcast_id = op.id;
        
        // Find consumers
        if let Some(consumers) = vbroadcast_consumers.get(&vbroadcast_id) {
            // Check if any consumer is shiftable
            let mut has_shiftable_consumer = false;
            let mut consumer_shift_exprs = Vec::new();
            
            for &consumer_idx in consumers {
                if shiftable_set.contains(&consumer_idx) {
                    has_shiftable_consumer = true;
                    let shift_var = shift_vars.get(&consumer_idx).unwrap();
                    // vbroadcast needed if consumer is NOT shifted
                    consumer_shift_exprs.push(shift_var.not());
                } else {
                    // Non-shiftable consumer always needs vbroadcast
                    has_shiftable_consumer = true;
                    consumer_shift_exprs.push(model.new_bool_var_with_name("always_true"));
                    model.add_eq(LinearExpr::from(consumer_shift_exprs.last().unwrap()), 1);
                }
            }
            
            if has_shiftable_consumer {
                // vbroadcast needed if at least one consumer is not shifted
                let needed = model.new_bool_var_with_name(format!("vbroadcast_needed_{}", idx));
                model.add_bool_or(consumer_shift_exprs.iter().map(|&v| v).collect(), needed);
                vbroadcast_needed_vars.insert(idx, needed);
                
                // Make vbroadcast interval optional
                let vbroadcast_interval = model.new_optional_fixed_size_interval_var(
                    cycle_vars[idx], 
                    1, 
                    needed
                );
                // Add to VALU cumulative with demand 1 when needed
            } else {
                // No shiftable consumers, vbroadcast always needed
                // Use regular interval (not optional)
            }
        }
    }
}
```

### Step 3: Modify ALU Dependency Constraints

**File**: `scheduler_rs/src/cpsat_scheduler.rs`

**Location**: Lines 2012-2043 (ALU dependency constraints)

**Changes**:

- When processing dependencies for ALU ops:
  - Check if `dep_id` is a vbroadcast
  - If yes, use the scalar producer instead
  - Otherwise, use original dependency

**Code modification**:

```rust
for &dep_id in &ops[idx].dep_ids {
    // Check if this is a vbroadcast dependency that should be rewritten
    let actual_dep_id = vbroadcast_scalar_deps.get(&dep_id).copied().unwrap_or(dep_id);
    
    if let Some(&dep_idx) = op_id_to_idx.get(&actual_dep_id) {
        // ... rest of constraint logic using dep_idx ...
        // Note: If dep_id was vbroadcast and we're using scalar producer,
        // we're depending on the scalar producer, not vbroadcast
    }
}
```

### Step 4: Handle Edge Cases

1. **Vbroadcast with immediate source**:
   - If vbroadcast source is an immediate (not a VReg), no dependency rewriting needed
   - Current behavior is correct

2. **Vbroadcast with no scalar producer**:
   - If scalar VReg is not produced by any op (e.g., input parameter), keep vbroadcast dependency
   - Fall back to original behavior

3. **Multiple vbroadcast dependencies**:
   - Each vbroadcast dependency is rewritten independently
   - ALU ops may depend on multiple scalar producers

4. **Vbroadcast with no consumers**:
   - Should not happen in valid SSA, but handle gracefully
   - Treat as always needed (or eliminate if dead code)

5. **Vbroadcast with mixed consumers**:
   - Some consumers shifted, some not shifted
   - Vbroadcast is needed (at least one non-shifted consumer)
   - ALU ops from shifted consumers depend on scalar producer
   - Non-shifted consumers depend on vbroadcast

### Step 5: Testing

See the **Tests** section below for the full test strategy.

## Expected Benefits

1. **Earlier ALU execution**: ALU ops can start as soon as scalar is ready, not waiting for vbroadcast
2. **Better parallelism**: Especially when vbroadcast and its consumer are both shifted to ALU
3. **Reduced makespan**: May improve schedule quality for kernels with many vbroadcast → VALU chains
4. **VALU slot savings**: CP-SAT can eliminate vbroadcast when all consumers are shifted, freeing VALU slots
5. **Better CP-SAT optimization**: Solver can make global decisions about vbroadcast emission based on consumer shift decisions

## Risks and Considerations

1. **Correctness**: Must ensure scalar producer is correctly identified
2. **SSA structure**: Need to verify how scalar VRegs are tracked in SSA
3. **Edge cases**: Handle immediates, input parameters, and other special cases
4. **Solver performance**: Additional constraints and variables may impact CP-SAT solving time
5. **Constraint complexity**: Conditional vbroadcast emission adds complexity to the model

---

## Risks (Deep Analysis)

### Correctness risks

| Risk | Description | Likelihood | Impact | Mitigation |
|------|-------------|------------|--------|------------|
| Wrong scalar producer | We map vbroadcast → scalar producer by matching `dest` VReg; SSA may have multiple defs or phi-like patterns | Medium | Critical (wrong schedule, wrong output) | Assert single producer per VReg in SSA; add test that compares Machine output vs reference on same input |
| Circular dependency | Rewriting vbroadcast dep to scalar could introduce or expose a cycle if scalar producer somehow depends on consumer | Low | Critical | Before rewriting, verify scalar producer is not a descendant of the consumer; add cycle check in analysis phase |
| Vbroadcast needed vs actual emission | Lowering must not emit vbroadcast when CP-SAT set `vbroadcast_needed = false`; mismatch causes missing def or wrong value | Medium | Critical | Single source of truth: pass CP-SAT result (which vbroadcasts to emit) explicitly to lowering; test that emitted kernel has no vbroadcast when all consumers shifted |
| Non-shiftable consumer missed | If we misclassify a consumer as shiftable, we may set vbroadcast_needed = false when a VALU consumer still needs the broadcast | Medium | Critical | Build consumer set from actual `dep_ids`; only treat as shiftable if op is in `shiftable_valu`; add test with multiply_add consumer (not shiftable) |
| Scalar producer not in model | Scalar may be constant (ConstChoice), input, or from another segment; if not in `op_id_to_idx` we must not rewrite | Medium | High (solver panic or wrong constraint) | Always check `op_id_to_idx.get(&actual_dep_id)`; when missing, keep vbroadcast dependency; log when we fall back |

### Model and solver risks

| Risk | Description | Likelihood | Impact | Mitigation |
|------|-------------|------------|--------|------------|
| Optional interval bugs | Vbroadcast optional interval must contribute to VALU cumulative only when `vbroadcast_needed`; wrong condition doubles or drops demand | Medium | High (infeasible or invalid schedule) | Reuse same pattern as shiftable VALU optional intervals; unit test that extracts solution and checks VALU usage per cycle |
| BoolOr / activation wrong | `vbroadcast_needed = OR(consumer not shifted)` must be correct for all consumers; missing or extra literal changes meaning | Medium | Critical | Unit test with 1, 2, 3 consumers and all combinations of shift decisions; assert vbroadcast_needed matches expected |
| More variables, slower solve | Extra booleans and optional intervals may increase search space or time per node | High | Low–Medium | Benchmark before/after on 2-round and 4-round; if >10% slowdown, consider lazy encoding or hints (e.g. prefer vbroadcast_needed = false when all shifted) |

### Integration and lifecycle risks

| Risk | Description | Likelihood | Impact | Mitigation |
|------|-------------|------------|--------|------------|
| Lowering ignores “do not emit” | Lowering/layout may still emit vbroadcast instruction when CP-SAT said not to | Medium | Critical | Explicit “vbroadcasts to emit” set from CP-SAT → lowering; integration test: run kernel, count vbroadcast in JSON/bundle, assert matches solver decision |
| Greedy vs CP-SAT divergence | Greedy may not model optional vbroadcast; hints or fallback path may emit vbroadcast always | Low | Medium | Document that optional vbroadcast is CP-SAT-only; greedy keeps current behavior; or extend greedy to skip vbroadcast when all consumers shifted |
| Regression in non-vbroadcast paths | Changes in constraint order or indexing could affect other ops | Low | High | Full regression: run existing CP-SAT and scheduler tests; run `test_optimized.py` and submission tests with and without the feature flag |

### Mitigation summary

- **Pre-merge**: All tests in the Tests section pass; use the progressive testing ladder (small rounds and batch sizes first, then build up to 2-round, 4-round, then full 16-round) so failures show up quickly on small configs.
- **Guardrails**: Feature flag or config to disable vbroadcast dependency rewriting and optional vbroadcast, so we can bisect.
- **Assertions**: In debug builds, assert no duplicate scalar producers, no cycles, and that every vbroadcast has at least one consumer (or is dead and handled).

---

## Tests

### Unit tests (Rust, `scheduler_rs`)

**Location**: `scheduler_rs/src/cpsat_scheduler.rs` (e.g. `#[cfg(test)] mod tests`) or dedicated `scheduler_rs/src/cpsat_vbroadcast_tests.rs` if preferred.

1. **Vbroadcast analysis**
   - Build small `ops` with: scalar producer (e.g. ConstChoice) → vbroadcast → one VALU add.
   - Call the same analysis used in the real model (e.g. extract to `compute_vbroadcast_scalar_deps` and `compute_vbroadcast_consumers`).
   - Assert: scalar producer id for vbroadcast is correct; single consumer, and it is the VALU add.

2. **Dependency rewriting (constraint shape)**
   - Build model with one shiftable VALU op that depends on vbroadcast; force shift (e.g. `ValuShiftMode::ForceAlu`).
   - Solve and extract cycle assignments.
   - Assert: consumer ALU op cycles are >= scalar producer cycle + 1, and not necessarily >= vbroadcast cycle (if vbroadcast is even present).

3. **Vbroadcast needed**
   - Case A: vbroadcast → single shiftable consumer. Force consumer to ALU. Assert in solution: vbroadcast_needed = false (or equivalent: vbroadcast interval not present).
   - Case B: Same, but force consumer to VALU. Assert vbroadcast_needed = true.
   - Case C: vbroadcast → two consumers, one shiftable one not (e.g. multiply_add). Assert vbroadcast_needed = true regardless of shift of the shiftable consumer.

4. **Edge cases**
   - Vbroadcast with immediate source (no scalar producer in ops): no rewrite; consumer still depends on vbroadcast.
   - Vbroadcast with no consumers (dead): no crash; treat as needed or skip.
   - Scalar producer not in `op_id_to_idx`: no rewrite; keep vbroadcast dependency.

### Integration tests (Rust)

1. **End-to-end schedule then lower**
   - Build SSA kernel that has at least one chain: scalar → vbroadcast → VALU op (shiftable).
   - Run CP-SAT with VALU shift enabled; then run existing lowering that emits instructions.
   - Assert: no panic; number of vbroadcast instructions in emitted output equals number of vbroadcasts with `vbroadcast_needed = true` in the solution.

2. **Correctness on Machine**
   - Use existing pattern (e.g. from `tick_tock_ssa` tests): build small kernel with known constants, vbroadcast, and VALU use; run on `Machine` with fixed input; assert output matches expected (or reference kernel if applicable).

### Progressive testing: small configs first, then build up

Test on minimal problem sizes first so failures are fast to reproduce and debug; only then scale to full defaults.

1. **Tier 1 – Minimal (fast, for quick iteration)**
   - **rounds=1, batch_size=8** (or 16): smallest valid kernel (batch_size multiple of VLEN 8).
   - Run CP-SAT with short time limit (e.g. 10–30 s); run full pipeline (build → schedule → lower → emit).
   - Assert: feasible, valid schedule, correctness vs reference (or known-good output) on the same config.
   - Use for: unit-level confidence that vbroadcast analysis, dependency rewriting, and optional emission don’t crash and produce a valid kernel.

2. **Tier 2 – Small (2-round, small batch)**
   - **rounds=2, batch_size=8 or 16**.
   - Run with VALU shift enabled; validate correctness and that vbroadcast count in emitted kernel matches solver’s “vbroadcast needed” decisions.
   - Use for: catching integration bugs (lowering vs CP-SAT mismatch) with short runs.

3. **Tier 3 – 2-round default batch**
   - **rounds=2, batch_size=256** (default batch).
   - Run `test_optimized.py` (or equivalent) with `--rust-builder` and VALU shift; assert correctness.
   - Optionally: compare cycles and vbroadcast stats with baseline (no optimization) to confirm no regression and possible improvement.

4. **Tier 4 – 4-round, small then default batch**
   - **rounds=4, batch_size=16** first (faster), then **rounds=4, batch_size=256**.
   - Full validation: correctness, slot counts, no invalid schedules.
   - Use for: stress-testing the model and lowering on realistic but still manageable sizes.

5. **Tier 5 – Full default (16-round, batch_size=256)**
   - Run only after Tiers 1–4 pass.
   - Full `test_optimized.py` and submission tests; record cycles and solver time for regression tracking.

**Order of execution**: Run Tiers 1 → 2 → 3 → 4 → 5 during development and in CI (Tiers 1–3 as gate; 4–5 as extended/CI). Document the exact commands and params (e.g. in this doc or in `CLAUDE.md`) so anyone can reproduce the ladder.

### Regression and performance tests (Python / harness)

1. **Correctness**
   - `uv run test_optimized.py` with `--rust-builder` and default (or explicit) VALU shift.
   - Optionally: run with and without a feature flag that enables this optimization and compare outputs (must be identical).

2. **Cycles and stability**
   - Run 2-round and 4-round with CP-SAT, e.g. short time limit; record cycles and optionally vbroadcast count.
   - After changes: re-run and assert cycles not worse (or within small tolerance); no increase in validation failures.
   - Prefer running the progressive ladder (Tiers 1–4) before full 16-round.

3. **Solver time**
   - Benchmark same config with and without optional vbroadcast + dependency rewriting; report solve time and gap if applicable.
   - Use small configs (e.g. rounds=2, batch_size=16) for quick before/after comparison; then spot-check 4-round if needed.

### Property / fuzz (optional, CI-only)

- Random SSA DAGs that include vbroadcast and shiftable VALU consumers; run CP-SAT; assert model is feasible and extracted schedule respects dependencies (scalar producer before consumer ALU ops; vbroadcast before any non-shifted consumer).

---

## Documentation

### In-code documentation

1. **`scheduler_rs/src/cpsat_scheduler.rs`**
   - At the top of the block that builds `vbroadcast_scalar_deps` and `vbroadcast_consumers`: document that vbroadcast dependencies are rewritten to scalar producer for ALU-shifted consumers, and that vbroadcast is optional when all consumers are shifted.
   - On `vbroadcast_needed` and optional interval: one-line comment that “vbroadcast is only emitted if it has at least one non-ALU-shifted consumer.”
   - For any new helper (e.g. `compute_vbroadcast_*`): doc comment with args, return value, and when we fall back to “no rewrite.”

2. **`scheduler_rs/src/kernel_builder/ssa_lower.rs`**
   - If lowering changes (e.g. skipping vbroadcast when not needed): document that the set of vbroadcasts to emit comes from CP-SAT result and must match `vbroadcast_needed` decisions.

3. **Constants and config**
   - If a feature flag or config is added (e.g. to disable this optimization), document it in `CLAUDE.md` or the same place other scheduler flags are documented.

### Plan and design docs

1. **This document (`docs/VALU_SHIFT_VBROADCAST_DEPENDENCY_OPTIMIZATION.md`)**
   - Keep it as the single design and implementation plan; update “Open Questions” and “References” as implementation progresses.
   - **Implementation status**: Maintain a short subsection at the top (e.g. “Status: not started | in progress | done”) and link to PRs or commits when applicable.

2. **Cross-references**
   - In `docs/ELEMENT_WISE_VALU_SHIFT_DEPENDENCIES.md`, add a note that vbroadcast dependency rewriting is separate and described in this doc.
   - In `docs/CPSAT_SCHEDULER_IMPLEMENTATION.md` (or equivalent), add a subsection on “Vbroadcast and VALU shift” that points to this plan and summarizes: dependency rewriting and optional vbroadcast emission.

### User- and operator-facing

- No CLI or user-facing behavior change beyond possibly different cycle counts or schedules; no new doc required unless we add an explicit flag, in which case document the flag in the same place as other scheduler options.

---

## Debugging

### Logging and verbose output

1. **Build-time analysis (Rust)**
   - When `verbose` (or a dedicated `cpsat_valu_shift_debug`) is true, log:
     - For each vbroadcast: op id, scalar producer id (or “none”), consumer op ids and whether each is shiftable.
     - When we rewrite a dependency: “rewrite dep vbroadcast_id → scalar_producer_id for consumer op_id.”
     - When we do not rewrite: “no rewrite: reason (e.g. scalar producer not in model, immediate source).”
   - Log counts: “vbroadcast_rewritten_deps = N, vbroadcast_optional_count = M.”

2. **Solution extraction**
   - In verbose mode, after solve: for each vbroadcast, print whether it was needed and its cycle (if present); for each shiftable consumer that had a vbroadcast dep, print that we used scalar producer cycle instead.

3. **Python / generator**
   - If `generate_ssa_kernel.py` or similar exposes verbose output, add a line or two when VALU shift is enabled: e.g. “Vbroadcast dependency rewriting: N deps rewritten; M vbroadcasts optional.”

### Assertions and sanity checks

1. **Debug-only assertions**
   - After building `vbroadcast_scalar_deps`: for each vbroadcast, scalar producer must be in `ops` and have `dest == vbroadcast.srcs[0]` (or equivalent).
   - After building consumer list: every vbroadcast that has consumers must have at least one op in `ops` that has this vbroadcast in `dep_ids`.
   - When applying rewrite: `actual_dep_id` must be in `op_id_to_idx` before adding constraints.

2. **Post-solve checks**
   - If vbroadcast_needed = false, no consumer that stayed on VALU may depend on this vbroadcast (all consumers must be shifted).
   - For each rewritten dep: in the solution, consumer’s ALU cycles (or completion_cycle) must be >= scalar producer cycle + 1.

### Reproducing and bisecting

1. **Determinism**
   - Run with fixed seed if CP-SAT allows; otherwise document that small differences in schedule may occur run-to-run.

2. **Bisect**
   - Provide a way to disable only vbroadcast dependency rewriting, or only optional vbroadcast (e.g. always emit vbroadcast). Use to bisect “wrong output” vs “wrong cycle count” vs “solver slow.”

3. **Minimal repro**
   - If a bug is found, add a minimal test in `scheduler_rs` that builds the smallest SSA (e.g. 3–5 ops) that exhibits the bug; run CP-SAT and assert expected constraint or solution.

### Tools and inspection

1. **Dump model or solution**
   - Option to dump (to stderr or file) a summary of vbroadcast-related decisions: for each vbroadcast, needed/not needed, cycle; for each consumer, dep used (vbroadcast vs scalar) and cycle. Helps compare good vs bad runs.

2. **Existing scheduler output JSON**
   - If `--output-json` already includes per-op cycle and engine, extend or document how to infer “vbroadcast emitted” vs “vbroadcast skipped” so that external tools or scripts can validate without reading Rust.

---

## Open Questions

1. **How to find scalar producer**: Is matching `dest` VReg sufficient, or do we need a different approach?
2. **Vbroadcast with immediate source**: Should we handle this case, or is current behavior sufficient?
3. **Interaction with element-wise dependencies**: How does this interact with the element-wise dependency optimization from the previous plan?
4. **Vbroadcast elimination**: When vbroadcast is eliminated, do we need to update any other constraints or accounting?

## References

- Current ALU dependency constraints: `scheduler_rs/src/cpsat_scheduler.rs` lines 2012-2043
- VALU expansion: `scheduler_rs/src/kernel_builder/ssa_lower.rs` `expand_ssa_valu_to_alu()`
- SSA op structure: `scheduler_rs/src/types.rs` `SsaOp` struct
- Related plan: `docs/ELEMENT_WISE_VALU_SHIFT_DEPENDENCIES.md`
