# Commit breakdown: precedence offsets and debugging

This supplements the main plan (debug slot tightening INFEASIBLE). Work is split into five commits so that (1) introducing precedence offsets on `SsaOp` is a standalone, reviewable change, (2) the buggy initial ASAP/ALAP computation is fixed using the trait, and (3) each commit has associated tests where applicable.

**Pre-requisite (no code change):** Before implementing any code, capture a concrete reproduction log (e.g. verbose run on 3r/64b config that goes INFEASIBLE with slot-only tightening) so the bug's existence is documented independently. This can be done in parallel with Commit 1.

---

## Commit 1: Precedence offset trait and impl on SsaOp

**Scope:** `scheduler_rs/src/types.rs` only. No changes to `scheduler_rs/src/cpsat_scheduler.rs`.

**Code:**

- Define trait `PrecedenceOffset` with two methods (see [ASAP_ALAP_PRECEDENCE_OFFSETS.md](ASAP_ALAP_PRECEDENCE_OFFSETS.md) for rationale):
  - `fn precedence_offset_for_asap(&self, predecessor_is_shiftable: bool) -> i32` — returns the **minimum delay** (for tightening consumer ASAP).
  - `fn precedence_offset_for_alap(&self, predecessor_is_shiftable: bool) -> i32` — returns the **maximum delay** (for tightening predecessor ALAP).
- Implement for `SsaOp`:
  - SelectChoice with `sub_temp_vreg.is_some()` → both return **2** (FMA delay: vsub at +1, FMA at +2).
  - Else if `predecessor_is_shiftable` → ASAP returns **1** (min: op might stay on VALU), ALAP returns **8** (max: shifted to 8 scalar ALU ops).
  - Else → both return **1** (normal: result visible next cycle).
- Add brief trait/impl doc comments (full doc in Commit 2). Note that SelectChoice ASAP offset 2 is conservative when mode is Optimize (true minimum is 1 for the vselect path); this is acceptable because the mode is not known at initial bounds computation time.

**Why two methods, not one?** ASAP bounds need the **minimum** delay (optimistic — assume the fastest resolution) while ALAP bounds need the **maximum** delay (pessimistic — assume the slowest resolution). Using a single conservative value (e.g. 8 for shiftable) in ASAP would artificially extend the critical path and shrink scheduling windows, potentially causing INFEASIBLE — the exact bug we are fixing. See `docs/ASAP_ALAP_PRECEDENCE_OFFSETS.md` for the full explanation.

**Tests (in `types.rs` `mod tests`):**

- **Normal op (ASAP + ALAP):** Build an `SsaOp` with e.g. `Opcode::Add`, `sub_temp_vreg: None`. Call `precedence_offset_for_asap(false)` → expect `1`. Call `precedence_offset_for_alap(false)` → expect `1`. Call both with `(true)` → expect ASAP `1`, ALAP `8`.
- **SelectChoice FMA delay:** Build `SsaOp` with `Opcode::SelectChoice` and `sub_temp_vreg: Some(VReg::...)`. Call ASAP/ALAP with `false`/`true`. Expect `2` in all cases (FMA delay dominates shiftable; and SelectChoice ops are never shiftable since `SelectChoice` is not in `can_shift_to_alu()`).
- **Shiftable, non-SelectChoice:** Build e.g. `Opcode::Add` on VALU, `sub_temp_vreg: None`. Call `precedence_offset_for_asap(true)` → expect `1`. Call `precedence_offset_for_alap(true)` → expect `8`.

Use `SsaOp::new(...)` plus direct field assignment for `opcode` and `sub_temp_vreg` so tests don't depend on emission. Run: `cargo test --manifest-path scheduler_rs/Cargo.toml precedence_offset` (or the exact test names).

**Result:** New opcodes only extend this impl; no call-site if/else anywhere.

---

## Commit 2: Documentation for precedence offsets

**Scope:** Existing doc + doc comments in `types.rs`.

**Code:**

- **docs/ASAP_ALAP_PRECEDENCE_OFFSETS.md** already exists with the correct two-method design. Align or expand it to include: pointer to CP-SAT sections 5, 5c, 5d; note on SelectChoice Optimize mode conservatism; treat the doc as the single source of truth for the trait.
- In `types.rs`: expand trait and impl doc comments to reference that doc; add one-line comments per branch in the impl (normal, SelectChoice FMA, shifted VALU).

**Tests:** No new code behavior. Optional: trivial test that a constant or path exists (e.g. that the doc file is present) if the project has that pattern; otherwise no test for this commit.

---

## Commit 3: Diagnostic (precedence violations + domain validation when verbose)

**Scope:** `scheduler_rs/src/cpsat_scheduler.rs`: after slot/scratch tightening and after `shiftable_valu` is computed, when `verbose`, check all RAW edges with the trait and log violation count and a sample. Also validate domain consistency (ASAP ≤ ALAP) for all ops.

**Code:**

- Add a function `diagnose_precedence_bounds` that takes `ops`, `asap`, `alap`, `shiftable_set`, and for each (pred, succ) edge checks **both** directions:
  - **ASAP check:** `asap[succ] >= asap[pred] + ops[pred_idx].precedence_offset_for_asap(shiftable_set.contains(&pred_idx))`
  - **ALAP check:** `alap[pred] <= alap[succ] - ops[pred_idx].precedence_offset_for_alap(shiftable_set.contains(&pred_idx))`
  - Returns `(total_violation_count, Vec<sample_violations>)` where each sample includes (pred_idx, succ_idx, expected, actual, offset, direction). Collect at most 3 samples but count all violations.
- **Domain validation (ASAP ≤ ALAP):** After tightening, check that `asap[op] <= alap[op]` for every op. If any op has `asap > alap`, its domain is empty and the CP-SAT model will be INFEASIBLE before the solver even starts. Log the count and sample of empty-domain ops. This catches the case where slot/scratch tightening narrows a window past the point of precedence consistency. **Currently there is no such validation anywhere in the code** — tightening can silently produce empty domains that are passed to the solver.
- No if/else on dependency type at the call site; reuse `precedence_offset_for_asap` / `precedence_offset_for_alap`.
- **Placement:** The diagnostic needs `shiftable_set`, which is `shiftable_valu` in the scheduler. In `cpsat_scheduler.rs`, `shiftable_valu` is computed *after* `final_asap`/`final_alap` are set (around 2085–2093). Call the check **after** the verbose VALU output block (~line 2131), before model building (~line 2133), when `verbose` is true. Pass `final_asap`/`final_alap` and `shiftable_set`.

**Tests:**

- **Unit test (preferred):** In `cpsat_scheduler.rs` tests, build a minimal `Vec<SsaOp>` with known RAW edges: set each op's `dep_ids` so the intended (pred, succ) pairs are the RAW edges (e.g. two ops with `succ.dep_ids = {pred.id}`). Inject `asap`/`alap` maps that violate one edge (e.g. pred ALAP 5, succ ASAP 6, offset 2 → need ASAP >= 7). Call the new check function and assert violation count >= 1 and that the reported sample includes that edge. For the offset-2 case, the **predecessor** must be SelectChoice with `sub_temp_vreg: Some(...)` so the trait returns 2. Also test an ALAP violation (e.g. succ ALAP 10, pred ALAP 5, max_delay 8 → need pred ALAP <= 2).
- **Domain validation test:** Build ops where tightening would produce ASAP > ALAP (e.g. inject `asap[op] = 10, alap[op] = 5`). Assert the diagnostic detects and reports the empty domain.
- **Alternative:** Integration-style test that runs `schedule_ssa_cpsat` with `verbose: true` and a small config that uses slot tightening, then parses stderr/log for the diagnostic line; heavier and more brittle.

**Result:** Violations and empty domains are visible in logs without changing solver behavior.

---

## Commit 4: Debug-fix flag and precedence-consistency pass

**Scope:** `scheduler_rs/src/cpsat_scheduler.rs`: config flag + fixpoint pass that restores ASAP/ALAP consistency using the same trait, then use result as `final_asap`/`final_alap`.

**Code:**

- Add `debug_fix_precedence_bounds: bool` to `SsaCpsatConfig` (default `false`).
- When set, after the diagnostic, run a **bidirectional** precedence-consistency fixpoint pass and set `final_asap`/`final_alap` from the corrected maps.
- No new if/else on dependency type; reuse `precedence_offset_for_asap` / `precedence_offset_for_alap`.
- **Bidirectional propagation (fixpoint):** Tightening can violate precedence in **both** directions. The pass must propagate both:
  1. **Forward (ASAP):** `ASAP[succ] = max(ASAP[succ], ASAP[pred] + min_delay)` — relaxes consumer ASAP forward when tightening pushed predecessors later without updating consumers.
  2. **Backward (ALAP):** `ALAP[pred] = min(ALAP[pred], ALAP[succ] - max_delay)` — tightens predecessor ALAP backward when tightening pushed consumers earlier without updating predecessors.
  - The backward direction is further tightening (pushing ALAP earlier), which is monotonic and safe — it can only shrink windows, not widen them.
  - Iterate until convergence (no changes). The fixpoint terminates because ASAP can only increase and ALAP can only decrease, bounded by the horizon.
- **Placement:** After the diagnostic (~line 2131), before model building (~line 2133). Update `final_asap` and `final_alap` in place.

**Tests:**

- **Integration test:** Run CP-SAT with a config that (1) enables ASAP/ALAP slot tightening and (2) sets `debug_fix_precedence_bounds: true`, on a small instance that is INFEASIBLE with slot-only and FEASIBLE with the fix. **Choose the exact config after repro** (main plan Step 1): e.g. 3r/64b or 2r/64b often shows INFEASIBLE with slot-only; 2r/32b may be FEASIBLE either way. Assert solver status is FEASIBLE (or not INFEASIBLE). If no small repro is found, the test can be skipped or relaxed (e.g. assert no crash / document as best-effort). Gate with `#[cfg(feature = "cpsat")]`; if the run is long, use `#[cfg(all(feature = "cpsat", feature = "ci-tests"))]` and document in CLAUDE.md so CI can run it when needed.

**Result:** Default behavior unchanged; with the flag, we can confirm the model becomes FEASIBLE.

---

## Commit 5: Fix initial ASAP/ALAP computation

**Scope:** `scheduler_rs/src/cpsat_scheduler.rs`: `compute_ssa_asap`, `compute_ssa_alap`, and their call sites.

**Code:**

- **Shiftable set before bounds:** In the main CP-SAT flow, compute `shiftable_valu` and `shiftable_set` **before** calling `compute_ssa_asap` / `compute_ssa_alap` (move the existing logic that builds them from ops + `enable_valu_shift` to run earlier, before the initial bounds step). This is safe because `shiftable_valu` only depends on `op.engine` and `op.opcode` — no circular dependency with ASAP/ALAP.
- **compute_ssa_asap:** Add parameter `shiftable_set: &HashSet<usize>` (op indices). Build `id_to_idx` from `ops`. When propagating to a successor, use `offset = ops[pred_idx].precedence_offset_for_asap(shiftable_set.contains(&pred_idx))` and `new_asap = my_asap + offset` instead of fixed `+ 1`.
- **compute_ssa_alap:** Add parameter `shiftable_set: &HashSet<usize>`. Build `id_to_idx` from `ops`. When propagating to a predecessor, use `offset = ops[succ_idx].precedence_offset_for_alap(...)` — wait, the trait is on the **predecessor** op. For ALAP propagation (from succ to pred): `offset = ops[pred_idx].precedence_offset_for_alap(shiftable_set.contains(&pred_idx))` and `new_alap = my_alap - offset` instead of fixed `- 1`.
- **Call sites:**
  1. **Main scheduler** (around 2034–2044): compute shiftable set first, then pass `&shiftable_set` into both functions.
  2. **`compute_ssa_asap_alap_bounds`:** Compute `shiftable_set` **inside** the function from `config.enable_valu_shift` and `ops` using the same filter as in `schedule_ssa_cpsat` (VALU ops with `opcode.can_shift_to_alu()`), then pass `&shiftable_set` into `compute_ssa_asap` and `compute_ssa_alap`. No new parameter needed; the config already has `enable_valu_shift`.
  3. **Greedy hints / warm-start consistency (`tick_tock_ssa.rs`):** The `bounds_config` at line ~543 uses `..Default::default()`, which sets `enable_valu_shift: false`. This means greedy bounds won't account for shiftable offsets even when the actual CP-SAT run has VALU shift enabled. **This matters for warm-start quality:** greedy cycles are clamped to `[ASAP, ALAP]` (line ~633–639) and passed as CP-SAT solution hints. If greedy bounds are tighter than CP-SAT bounds, the hints are still valid (they fall within the wider CP-SAT domain). But if greedy bounds use wrong offsets and produce cycles *outside* CP-SAT domains, **hints are silently rejected** and the solver starts from scratch — losing the warm-start speedup entirely.
     - **Preferred fix:** Pass `enable_valu_shift` through to `bounds_config` so greedy and CP-SAT use consistent offsets. The `enable_valu_shift` value is available in `tick_tock_ssa.rs` at the call site.
     - **Acceptable alternative:** Document that greedy bounds are an approximation. Since greedy doesn't shift ops, its cycles use offset-1 windows, which are *subsets* of the wider CP-SAT windows — so hints are always valid (just potentially suboptimal).
- **Hint rejection logging (new):** When verbose, log if any greedy hint cycle falls outside the CP-SAT `[final_asap, final_alap]` domain. Currently there is no such logging — hint rejection is silent. Add a check in the hint-application code (around where `greedy_hints` are used, before model building) that counts and reports mismatches. This is a small addition but has high diagnostic value for understanding warm-start effectiveness.

**Tests:**

- **Unit tests:** Existing or new tests that call `compute_ssa_asap` / `compute_ssa_alap` with a small DAG that includes a SelectChoice with `sub_temp_vreg.is_some()` or a shiftable VALU op; assert that the returned ASAP/ALAP reflect the correct offset:
  - FMA case: `ASAP[succ] >= ASAP[pred] + 2`, `ALAP[pred] <= ALAP[succ] - 2`.
  - Shiftable case: `ASAP[succ] >= ASAP[pred] + 1` (min delay), `ALAP[pred] <= ALAP[succ] - 8` (max delay).
  - Normal case: both use `1`.
- **Regression:** Run the same repro config (e.g. 3r/64b slot-only) without the debug-fix flag; solver should be FEASIBLE (or at least fewer violations in verbose diagnostic) because initial bounds are now correct.
- **Symmetry breaking stability:** Run a before/after comparison on a small config (e.g. 2r/32b). Check that `build_ordering_pairs_from_bounds` produces similar or fewer ordering pairs. Corrected ALAP (+8 for shiftable) changes the ALAP sort order within ordering groups — verify this doesn't regress solver time or solution quality. If symmetry breaking degrades (unlikely), consider whether ordering should prefer ASAP over ALAP.

**Result:** Initial ASAP/ALAP are valid for every resolution; slot tightening starts from correct bounds. The diagnostic (Commit 3) and debug-fix (Commit 4) remain as a safety net.

---

## Order and dependencies

```
Commit 1 (Trait + impl) ──┬──► Commit 2 (Docs) ──► Commit 5 (Initial ASAP/ALAP fix) ──► Commit 3 (Diagnostic) ──► Commit 4 (Debug-fix)
                          └──► Commit 5
                          └──► Commit 3
```

- **Commit 1** must land first (trait + impl + unit tests).
- **Commit 2** can be same PR as 1 or immediately after; no code dependency.
- **Commit 5** depends on 1 (and optionally 2); fixes the buggy initial ASAP/ALAP using the trait. Best placed before Commit 3 so the diagnostic runs on correct initial bounds.
- **Commit 3** depends on 1 (and optionally 2); adds diagnostic and unit test. Can run after 5 to confirm fewer/zero violations.
- **Commit 4** depends on 3; adds flag and pass plus integration test.

Repro on small instances (Step 1 of the main plan) can be done before or in parallel with Commit 1 (no code change, manual or scripted runs).

---

## Summary

| Commit | Content                                                            | Tests                                                                                      |
| ------ | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------ |
| 1      | `PrecedenceOffset` trait (two methods) and `SsaOp` impl in types.rs | Unit tests in types.rs: normal → (1,1), SelectChoice+sub_temp_vreg → (2,2), shiftable → (1,8) |
| 2      | Align/expand existing doc + trait/impl comments                    | Optional doc-exists check; otherwise none                                                  |
| 5      | Fix compute_ssa_asap / compute_ssa_alap with trait + shiftable_set; fix bounds_config; hint logging; add proptest | Unit tests for correct offset; regression; symmetry breaking stability; 4 property-based tests |
| 3      | Violation-check + domain validation (ASAP ≤ ALAP) + call when verbose | Unit test: precedence violations + empty domain detection                                  |
| 4      | `debug_fix_precedence_bounds` + bidirectional fixpoint pass        | Integration test: run with flag + slot tightening, assert FEASIBLE (e.g. 2r/32b or 3r/64b) |

This keeps "introducing offsets into SsaOp" as a distinct, test-covered commit; adds a required fix for the initial ASAP/ALAP bug; and avoids special if/else at call sites as in the main plan.

---

## System-wide effects and solver performance

Correcting ASAP/ALAP bounds is not an isolated change — the bounds flow through nearly every component in the scheduling pipeline. This section documents the expected effects to guide testing and performance monitoring.

### How bounds flow through the system

```
compute_ssa_asap / compute_ssa_alap (initial bounds, fixed +1 bug)
  → tighten_asap_alap_by_slots (energetic reasoning, capacity-based)
    → tighten_asap_alap_by_scratch (liveness pressure)
      → final_asap / final_alap
        ├─► cycle_vars domains [ASAP, ALAP] (CP-SAT variable creation, line 2156)
        │     ├─► interval_vars for cumulative constraints (engine slot limits)
        │     ├─► FMA cycle variable domains (derived with +1 offset, line 2648)
        │     ├─► ALU lane cycle variables for shifted ops (same domain, line 2276)
        │     └─► makespan objective (max of all cycle/completion vars)
        ├─► symmetry breaking (build_ordering_pairs_from_bounds, sorts by ALAP)
        ├─► greedy scheduler (priority queue ordering by ASAP, clamping by ALAP)
        │     └─► CP-SAT warm-start hints (greedy cycles validated against domains)
        └─► schedule_constraints JSON (Domain constraints for bulk-move simulator)
```

### Expected solver performance improvements

| Mechanism | Effect of correct bounds | Estimated impact |
|---|---|---|
| **Domain reduction** | Tighter ALAP for shiftable successors (~7 cycles per op × ~1,500 ops) | Exponential search space reduction |
| **Energetic reasoning** | Slot tightening starts from valid foundation → better capacity deductions | More effective pruning, especially for VALU (cap=6) and Flow (cap=1) |
| **Branch-and-bound** | Tighter cycle domains → tighter makespan lower bound → smaller optimality gap | Fewer branches explored; faster convergence |
| **Warm-start quality** | Greedy hints more likely to fall within CP-SAT domains → hints accepted | Solver starts closer to optimal; avoids cold-start penalty |
| **INFEASIBLE elimination** | No more empty domains from tightening → solver finds solutions | Previously wasted runs now produce results |

### Per-choice constraint interaction

The CP-SAT model uses three constraint patterns that must be consistent with initial bounds:

1. **Always-on constraints** (normal deps): `cycle[succ] >= cycle[pred] + 1`. These are consistent with +1 initial bounds.
2. **Variable-offset constraints** (SelectChoice FMA): `cycle[succ] >= cycle[pred] + 1 + choice_var` where `choice_var ∈ {0,1}`. Initial ASAP must admit the +2 case (FMA) for the domain to be wide enough.
3. **Conditional constraints** (VALU shift): `only_enforce_if(cycle[succ] >= completion_cycle + 1, shift_var)` where `completion_cycle` can be up to `pred_cycle + 7`. Initial ALAP must admit the +8 case for the consumer's domain to accommodate the shifted path.

**Key invariant:** Initial bounds must form the **convex hull** across all choice resolutions — ASAP = min delay (optimistic), ALAP = max delay (pessimistic). This ensures every choice branch has a feasible domain before the solver starts exploring.

### Components that do NOT need changes

| Component | Why it's unaffected |
|---|---|
| **ConstChoice constraints** | All encodings (Load, AddImm, ALU-derived) use offset 1; no variable delay |
| **Bulk-move simulator** | Validates against Domain constraints from JSON; wider domains = fewer false rejections (correct behavior) |
| **Python scripts** (`run_asap_alap_ablation.py`, etc.) | Pass flags to Rust; no bounds logic in Python |
| **Explicit CP-SAT precedence constraints** (sections 5, 5c, 5d) | Already correct; they use the right offsets for each choice. The fix is in the *initial bounds*, not the model constraints |

---

## Review-driven changes (from multi-agent code review)

Changes made to this plan based on a systematic review:

1. **Two-method trait (Critical fix):** Changed from a single `precedence_offset_for_consumer` method to two methods (`for_asap` / `for_alap`), aligning with the existing `docs/ASAP_ALAP_PRECEDENCE_OFFSETS.md`. The single-method design returning 8 for shiftable ASAP would have artificially extended the critical path — the exact bug this plan aims to fix.

2. **Bidirectional propagation in Commit 4 (Critical fix):** The fixpoint pass now propagates both ASAP forward (`max(ASAP[succ], ASAP[pred] + min_delay)`) and ALAP backward (`min(ALAP[pred], ALAP[succ] - max_delay)`). Slot tightening can violate precedence in both directions.

3. **Greedy hints context in Commit 5 (Important):** Flagged that `compute_ssa_asap_alap_bounds` in `tick_tock_ssa.rs` uses `..Default::default()` for bounds_config, which sets `enable_valu_shift: false`. Added explicit guidance to either pass the flag through or document the approximation.

4. **Diagnostic checks both directions (Commit 3):** Updated to check both ASAP violations (`asap[succ] >= asap[pred] + min_delay`) and ALAP violations (`alap[pred] <= alap[succ] - max_delay`).

5. **SelectChoice Optimize mode note:** Added documentation note that returning 2 for ASAP on SelectChoice is conservative when mode is Optimize (true min is 1 for vselect path), and why this is acceptable.

6. **Pre-requisite repro log:** Added guidance to capture a reproduction log before implementing, so the bug's existence is documented independently of the code changes.

### Deep review changes (second round — system-wide effects)

Additional changes from a deep review exploring CP-SAT solver internals, per-choice constraint interactions, tightening function behavior, and downstream pipeline effects:

7. **Domain validation in Commit 3 (New — High):** Added ASAP ≤ ALAP check for all ops after tightening. Currently there is **no validation** that tightening produces non-empty domains. Slot/scratch tightening can silently produce `asap > alap` (empty domain) which makes the CP-SAT model trivially INFEASIBLE. The diagnostic now catches this.

8. **Hint rejection logging in Commit 5 (New — Medium):** Added guidance to log when greedy warm-start hints fall outside CP-SAT `[ASAP, ALAP]` domains. Currently hint rejection is silent — the solver just starts from scratch without explanation. A verbose log helps diagnose warm-start failures.

9. **Warm-start consistency analysis (Expanded — Medium):** Expanded the `tick_tock_ssa.rs` bounds_config discussion with full analysis of the greedy → CP-SAT hint pipeline and why bounds consistency matters for warm-start effectiveness. Clarified that greedy-with-wrong-offsets produces hints that are *subsets* of the correct domain (still valid), not violations.

10. **Symmetry breaking stability test in Commit 5 (New — Low):** `build_ordering_pairs_from_bounds` sorts ops by ALAP within ordering groups. Corrected ALAP (+8 for shiftable) changes the sort order. Added a before/after comparison test to catch any solver regression from changed symmetry-breaking constraints.

11. **System-wide effects section (New):** Added comprehensive documentation of how bounds flow through the pipeline (cycle vars → intervals → cumulative → objective → greedy → hints → symmetry breaking → schedule constraints JSON). Includes expected solver performance improvements, per-choice constraint interaction analysis, and components that do NOT need changes (ConstChoice, bulk-move, Python scripts, explicit CP-SAT precedence constraints).

12. **ConstChoice confirmed safe (New — confirmed no-change):** Deep review confirmed all ConstChoice encodings (Load, AddImm, ALU-derived) use offset 1. No variable delay mechanism, no special handling needed in the trait.

13. **Root cause clarification:** The INFEASIBLE bug is not just wrong initial offsets — it's that slot tightening, operating without precedence knowledge, narrows windows to the point where they conflict with the CP-SAT model's actual conditional/variable-offset precedence constraints. Commit 5 (correct initial bounds) gives tightening more room; Commit 4 (bidirectional fixpoint) restores consistency after tightening.

---

## Code structure and placement guidance

Guidance on where new code should live, based on codebase conventions.

### Trait placement in types.rs

**Design decision: trait vs. plain methods.**
The codebase uses `impl SsaOp { ... }` methods exclusively — there are no existing public traits in scheduler_rs. Two options:

- **Option A (trait — chosen by plan):** Define `pub trait PrecedenceOffset` with the two methods, then `impl PrecedenceOffset for SsaOp`. This introduces a new pattern but makes the contract explicit, is extensible, and clearly separates "precedence offset" behavior from general SsaOp methods.
- **Option B (plain methods):** Add `pub fn precedence_offset_for_asap(...)` and `pub fn precedence_offset_for_alap(...)` directly to `impl SsaOp`. More consistent with existing code, but the contract is implicit.

**Recommendation:** Use Option A (trait). The trait is small, well-contained, and makes the two-method invariant (ASAP = min, ALAP = max) explicit at the type level. Place it:
- **Trait definition:** After the `Opcode` impl block (around line 400 in types.rs), or right before the `impl SsaOp` block (around line 1411). Group with a section comment.
- **Trait impl:** Right after the trait definition, or at the end of `impl SsaOp`.
- **Visibility:** `pub` — the trait is used in cpsat_scheduler.rs and potentially schedule_constraints.rs.
- **Import in cpsat_scheduler.rs:** Add `use crate::types::PrecedenceOffset;` to the existing import block (line ~164). No other new imports needed (`HashMap`, `HashSet`, `SsaOp` are already imported).

### Function placement in cpsat_scheduler.rs

cpsat_scheduler.rs is 9,641 lines, organized into sections. Place new functions near related existing code:

| New function | Place after | Rationale |
|---|---|---|
| `diagnose_precedence_bounds` (Commit 3) | After `verify_select_choice_fma_consumer_precedence` (~line 976) | Groups diagnostic/validation functions together; add section header `// Precedence Offset Diagnostics` |
| `fix_precedence_bounds_fixpoint` (Commit 4) | Before `compute_ssa_asap_alap_bounds` (~line 1889) | Groups with ASAP/ALAP tightening functions (lines 1561–1889) |

**Visibility:** Make both `fn` (private) initially. If unit tests need direct access, use `pub(crate)`. The existing pattern (`validate_ssa_slot_limits`, `verify_select_choice_fma_consumer_precedence`) uses private functions called from within `schedule_ssa_cpsat`.

**Comment style:** Use `// ============================================================================` section headers (matching existing pattern) and `///` doc comments for function signatures:
```rust
// ============================================================================
// Precedence Offset Diagnostics
// ============================================================================

/// Check precedence bound consistency after tightening.
///
/// For each RAW edge (pred → succ), verifies:
/// - ASAP[succ] >= ASAP[pred] + min_delay (forward)
/// - ALAP[pred] <= ALAP[succ] - max_delay (backward)
///
/// See docs/ASAP_ALAP_PRECEDENCE_OFFSETS.md for offset semantics.
fn diagnose_precedence_bounds(
    ops: &[SsaOp],
    asap: &HashMap<usize, i32>,
    alap: &HashMap<usize, i32>,
    shiftable_set: &HashSet<usize>,
) -> (usize, Vec<(usize, usize, i32, i32, i32, &'static str)>) {
```

### Config field placement

Add `debug_fix_precedence_bounds: bool` at the **end** of `SsaCpsatConfig` (after `enable_scratch_tightening`, ~line 1161), with default `false` in the `Default` impl. No builder pattern exists — just direct struct instantiation.

### CP-SAT model section comments to update

The solver code in `schedule_ssa_cpsat` uses numbered section comments. After implementation, update:

| Section | Current comment | Updated comment |
|---|---|---|
| **1** (~line 2034) | "Compute ASAP/ALAP bounds" | "Compute ASAP/ALAP bounds using precedence offsets (see ASAP_ALAP_PRECEDENCE_OFFSETS.md)" |
| **1b** (~line 2046) | "Tighten ASAP/ALAP" | Add note: "Tightening is capacity-only; precedence consistency restored by diagnostic (Commit 3) or fixpoint (Commit 4) if needed" |
| **2** (~line 2085) | "Identify shiftable VALU" | Reorder: compute shiftable set BEFORE section 1. Update comment to reflect new position. |
| **5** (~line 2169) | "Add precedence constraints" | Add: "NOTE: These explicit constraints are already correct per-choice. The fix is in initial bounds (Section 1), not here." |

---

## Documentation update checklist

All docs that reference ASAP/ALAP, precedence, or the constraint model, and what needs changing.

### Must update (in commit order)

| File | Commit | What to change |
|---|---|---|
| **types.rs** — trait + impl doc comments | 1 | New: trait-level doc explaining min/max delay semantics; impl doc with one-line per branch; reference to ASAP_ALAP_PRECEDENCE_OFFSETS.md |
| **docs/ASAP_ALAP_PRECEDENCE_OFFSETS.md** | 2 | Add: pointers to CP-SAT sections 5, 5c, 5d; note on SelectChoice Optimize conservatism; mark as single source of truth for trait design |
| **cpsat_scheduler.rs** — `compute_ssa_asap` doc comment | 5 | Update: "Uses RAW dependencies" → "Uses RAW dependencies with precedence offsets via `PrecedenceOffset` trait" |
| **cpsat_scheduler.rs** — `compute_ssa_alap` doc comment | 5 | Same as above |
| **cpsat_scheduler.rs** — section 1/1b/2/5 comments | 5 | See "CP-SAT model section comments to update" table above |
| **docs/CPSAT_IMPROVEMENTS.md** | 2 or 5 | Update "1. ASAP/ALAP Tightening" section: note that initial bounds must use correct offsets (not fixed +1); reference the trait |

### Should update (after all commits land)

| File | What to change |
|---|---|
| **docs/CPSAT_SCHEDULER_IMPLEMENTATION.md** | Add note in "Precedence" section: "Precedence offsets vary by dependency type; see ASAP_ALAP_PRECEDENCE_OFFSETS.md" |
| **docs/ABLATION_*.md** (4 files) | Re-run ablation studies with correct offsets; expect fewer INFEASIBLE, better propagation |
| **BUGS.md** | Add entry once bug is reproduced: "Slot tightening + incorrect ASAP/ALAP offsets → INFEASIBLE; fixed in [commit hash]" |
| **schedule_constraints.rs** — section 5 comments | Verify comments match new trait-based offset logic; add reference to ASAP_ALAP_PRECEDENCE_OFFSETS.md |

### No changes needed

| File | Why |
|---|---|
| CLAUDE.md | Problem-level docs; no ASAP/ALAP content |
| docs/ISA_REFERENCE.md | ISA definition, not scheduling |
| docs/PROBLEM_CONSTRAINTS.md | Problem parameters, not constraint model |
| docs/PER_DECILE_SLOT_UTILIZATION.md | Analysis tool docs |
| .cursor/skills/*.md | Tool usage docs, not scheduling internals |

---

## Expanded test plan

Comprehensive test plan with coverage gaps addressed, existing test impact, and property-based testing.

### Existing tests affected by signature changes

Changing `compute_ssa_asap(ops)` → `compute_ssa_asap(ops, &shiftable_set)` (and similarly for `compute_ssa_alap`) affects these call chains:

| Call site | File | Tests affected | Fix |
|---|---|---|---|
| `schedule_ssa_cpsat` lines 2035, 2044 | cpsat_scheduler.rs | **~56 CP-SAT tests** (all `#[cfg(feature = "cpsat")]`) | Update call site once; all tests pass through |
| `compute_ssa_asap_alap_bounds` line 1897 | cpsat_scheduler.rs | **~44 tick_tock_ssa tests** + **~22 ssa_greedy tests** that use bounds | Compute shiftable_set inside function; no test changes |

**Key insight:** Only **3 call sites** need updating (lines 2035, 2044, and 1897). All downstream tests (56 + 44 + 22 = ~122 tests) pass through these call sites and will automatically use the new offsets. No individual test needs modification unless it directly constructs and calls `compute_ssa_asap` or `compute_ssa_alap`.

### Test helper to add

Add a helper in types.rs tests (or cpsat_scheduler.rs tests) for building test DAGs:

```rust
fn make_test_ssa_op(
    id: usize,
    engine: Engine,
    opcode: Opcode,
    sub_temp_vreg: Option<VReg>,
    dep_ids: Vec<usize>,
) -> SsaOp {
    let mut op = SsaOp::new(id, engine, opcode, Some(VReg::scalar(id as u32)), vec![]);
    op.sub_temp_vreg = sub_temp_vreg;
    op.dep_ids = dep_ids.into_iter().collect();
    op
}
```

### Coverage gap tests to add

| Gap | Commit | Test description | Priority |
|---|---|---|---|
| **Empty shiftable_set** | 5 | Call `compute_ssa_asap(ops, &HashSet::new())` on a normal DAG. Verify all offsets are 1 (same as old behavior). | High — regression safety |
| **Mixed chain** (normal → SelectChoice → shiftable → normal) | 5 | 4-op chain with varying offsets. Verify cumulative ASAP: 0, 1, 3 (1+2), 4 (3+1). Verify cumulative ALAP uses max_delay: pred of shiftable gets ALAP 8 less than succ. | High — real-world scenario |
| **Diamond DAG** (two paths to one op) | 5 | `op0 → (op1, op2) → op3` where op1 is SelectChoice (offset 2) and op2 is normal (offset 1). Verify `ASAP[op3] = max(ASAP[op1]+2, ASAP[op2]+1)`. | Medium |
| **Fixpoint convergence** | 4 | Build bounds with cascading violations (op A → B → C, each violating precedence). Verify fixpoint terminates and all constraints satisfied. | Medium |
| **Fixpoint on consistent bounds** | 4 | Build already-consistent bounds. Verify fixpoint returns bounds unchanged in 0 iterations. | Low — fast path |
| **Single-op DAG** | 5 | One op, no deps. ASAP = 0, ALAP = horizon - 1 regardless of offset. | Low — edge case |

### Property-based testing (required — Commit 5)

Proptest is referenced in the `ci-tests` feature comment in `scheduler_rs/Cargo.toml` but is **not yet a dependency** and has no existing proptest usage in the codebase. Add it as part of Commit 5 (or a preparatory commit).

**Add dependency to `scheduler_rs/Cargo.toml`:**
```toml
[dev-dependencies]
proptest = "1.4"
```

Using `[dev-dependencies]` ensures proptest is only compiled for `cargo test`, not included in the release binary. No feature flag needed for the dependency itself — proptest tests are gated with `#[cfg(all(feature = "cpsat", feature = "ci-tests"))]` to keep `cargo test` fast by default.

**Required properties to test:**

```rust
use proptest::prelude::*;

// Strategy: generate a random DAG of 2-20 ops with random dep edges (acyclic),
// random opcodes (some SelectChoice with sub_temp_vreg), and a random shiftable_set.
// Use the existing make_test_ssa_op helper.

// Property 1: ASAP respects forward offsets
// For any valid DAG and shiftable_set:
//   ASAP[succ] >= ASAP[pred] + precedence_offset_for_asap(shiftable) for all edges
#[cfg(all(feature = "cpsat", feature = "ci-tests"))]
proptest! {
    #[test]
    fn asap_respects_precedence_offsets(dag in arb_dag(2..20)) {
        let asap = compute_ssa_asap(&dag.ops, &dag.shiftable_set);
        for (pred_idx, succ_idx) in &dag.edges {
            let offset = dag.ops[*pred_idx].precedence_offset_for_asap(
                dag.shiftable_set.contains(pred_idx)
            );
            prop_assert!(
                asap[&dag.ops[*succ_idx].id] >= asap[&dag.ops[*pred_idx].id] + offset,
                "ASAP violation: pred={} succ={} offset={}",
                pred_idx, succ_idx, offset
            );
        }
    }
}

// Property 2: ALAP respects backward offsets
// For any valid DAG and shiftable_set:
//   ALAP[pred] <= ALAP[succ] - precedence_offset_for_alap(shiftable) for all edges
#[cfg(all(feature = "cpsat", feature = "ci-tests"))]
proptest! {
    #[test]
    fn alap_respects_precedence_offsets(dag in arb_dag(2..20)) {
        let alap = compute_ssa_alap(&dag.ops, dag.horizon, &dag.shiftable_set);
        for (pred_idx, succ_idx) in &dag.edges {
            let offset = dag.ops[*pred_idx].precedence_offset_for_alap(
                dag.shiftable_set.contains(pred_idx)
            );
            prop_assert!(
                alap[&dag.ops[*pred_idx].id] <= alap[&dag.ops[*succ_idx].id] - offset,
                "ALAP violation: pred={} succ={} offset={}",
                pred_idx, succ_idx, offset
            );
        }
    }
}

// Property 3: ASAP <= ALAP for all ops (non-empty domains)
#[cfg(all(feature = "cpsat", feature = "ci-tests"))]
proptest! {
    #[test]
    fn domains_are_non_empty(dag in arb_dag(2..20)) {
        let asap = compute_ssa_asap(&dag.ops, &dag.shiftable_set);
        let alap = compute_ssa_alap(&dag.ops, dag.horizon, &dag.shiftable_set);
        for op in &dag.ops {
            let a = asap[&op.id];
            let l = alap[&op.id];
            prop_assert!(a <= l, "Empty domain: op {} ASAP={} > ALAP={}", op.id, a, l);
        }
    }
}

// Property 4: Fixpoint converges and restores consistency
#[cfg(all(feature = "cpsat", feature = "ci-tests"))]
proptest! {
    #[test]
    fn fixpoint_converges_and_is_consistent(dag in arb_dag(2..20)) {
        let asap = compute_ssa_asap(&dag.ops, &dag.shiftable_set);
        let alap = compute_ssa_alap(&dag.ops, dag.horizon, &dag.shiftable_set);
        let (fixed_asap, fixed_alap) = fix_precedence_bounds_fixpoint(
            &dag.ops, &asap, &alap, &dag.shiftable_set
        );
        // After fixpoint, no violations should remain
        let (violations, _) = diagnose_precedence_bounds(
            &dag.ops, &fixed_asap, &fixed_alap, &dag.shiftable_set
        );
        prop_assert_eq!(violations, 0, "Fixpoint left {} violations", violations);
    }
}
```

**DAG generation strategy (`arb_dag`):** Build a helper that generates:
1. Random number of ops (2..20)
2. Random topological ordering (assign ids in order; deps only point to earlier ids)
3. Random opcodes: ~10% `Opcode::SelectChoice` with `sub_temp_vreg: Some(...)`, rest `Opcode::Add`
4. Random engines: ~30% `Engine::Valu`, rest `Engine::Alu`
5. Random shiftable_set: subset of VALU ops where `opcode.can_shift_to_alu()`
6. Horizon: critical_path_length + random_slack (10..50)

Place the strategy and tests in a `mod proptest_tests` at the bottom of `cpsat_scheduler.rs` tests, gated with `#[cfg(all(feature = "cpsat", feature = "ci-tests"))]`.

### Test gating summary

| Test type | Feature gate | Run by default? |
|---|---|---|
| Trait unit tests (Commit 1) | None | Yes — trait is not CP-SAT-specific |
| Diagnostic unit tests (Commit 3) | None (function is pub(crate)) | Yes |
| Fixpoint unit tests (Commit 4) | None | Yes |
| Integration tests (Commits 4, 5) | `#[cfg(feature = "cpsat")]` | Yes with `--features cpsat` |
| Long-running regression (Commit 5) | `#[cfg(all(feature = "cpsat", feature = "ci-tests"))]` | Only in CI |
| **Property-based tests (Commit 5)** | `#[cfg(all(feature = "cpsat", feature = "ci-tests"))]` | Only in CI — requires `proptest` dev-dependency |

### Dependency changes

Add to `scheduler_rs/Cargo.toml` (Commit 5 or preparatory commit):

```toml
[dev-dependencies]
proptest = "1.4"
```

This is the first proptest usage in scheduler_rs. The `ci-tests` feature comment already references proptest but the dependency was never added. Run proptest tests with:

```bash
BUILD_FROM_PROJECT_ROOT=1 cargo test --manifest-path scheduler_rs/Cargo.toml --features "cpsat ci-tests"
```
