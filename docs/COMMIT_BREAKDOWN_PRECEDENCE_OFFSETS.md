# Commit breakdown: precedence offsets and debugging

This supplements the main plan (debug slot tightening INFEASIBLE). Work is split into five commits so that (1) introducing precedence offsets on `SsaOp` is a standalone, reviewable change, (2) the buggy initial ASAP/ALAP computation is fixed using the trait, and (3) each commit has associated tests where applicable.

---

## Commit 1: Precedence offset trait and impl on SsaOp

**Scope:** `scheduler_rs/src/types.rs` only. No changes to `scheduler_rs/src/cpsat_scheduler.rs`.

**Code:**

- Define trait `PrecedenceOffset` with `fn precedence_offset_for_consumer(&self, predecessor_is_shiftable: bool) -> i32`.
- Implement for `SsaOp`: SelectChoice with `sub_temp_vreg.is_some()` → 2; else if `predecessor_is_shiftable` → 8; else → 1.
- Add brief trait/impl doc comments (full doc in Commit 2).

**Tests (in `types.rs` `mod tests`):**

- **Normal op:** Build an `SsaOp` with e.g. `Opcode::Add`, `sub_temp_vreg: None`. Call `precedence_offset_for_consumer(false)` and `(true)`. Expect `1` and `8` (when shiftable).
- **SelectChoice FMA delay:** Build `SsaOp` with `Opcode::SelectChoice` and `sub_temp_vreg: Some(VReg::...)`. Call with `false`/`true`. Expect `2` in both cases (FMA delay dominates shiftable).
- **Shiftable, non-SelectChoice:** Build e.g. `Opcode::Add` on VALU, `sub_temp_vreg: None`. Call `precedence_offset_for_consumer(true)`. Expect `8`.

Use `SsaOp::new(...)` plus direct field assignment for `opcode` and `sub_temp_vreg` so tests don't depend on emission. Run: `cargo test --manifest-path scheduler_rs/Cargo.toml precedence_offset` (or the exact test names).

**Result:** New opcodes only extend this impl; no call-site if/else anywhere.

---

## Commit 2: Documentation for precedence offsets

**Scope:** Existing doc + doc comments in `types.rs`.

**Code:**

- **docs/ASAP_ALAP_PRECEDENCE_OFFSETS.md** already exists. Align or expand it to describe the **single-method design**: one offset per (pred, succ) — 1 (normal), 2 (SelectChoice FMA), 8 (shiftable). Document that we use the **conservative** value 8 for shiftable everywhere (valid for every resolution; ASAP on shiftable edges may be slightly looser than a min-delay approach). Ensure three dependency types, one-sentence rationale, and pointer to CP-SAT sections 5, 5c, 5d; treat the doc as the single source of truth for the trait.
- In `types.rs`: expand trait and impl doc comments to reference that doc; add one-line comments per branch in the impl (normal, SelectChoice FMA, shifted VALU).

**Tests:** No new code behavior. Optional: trivial test that a constant or path exists (e.g. that the doc file is present) if the project has that pattern; otherwise no test for this commit.

---

## Commit 3: Diagnostic (precedence violations when verbose)

**Scope:** `scheduler_rs/src/cpsat_scheduler.rs`: after slot/scratch tightening and after `shiftable_valu` is computed, when `verbose`, check all RAW edges with the trait and log violation count and a sample.

**Code:**

- Add a function that takes `ops`, `asap`, `alap`, `shiftable_set`, and for each (pred, succ) does `offset = ops[pred_idx].precedence_offset_for_consumer(shiftable_set.contains(&pred_idx))`, then checks `asap[succ] >= alap[pred] + offset`; returns or logs count and sample. No if/else on dependency type.
- **Placement:** The diagnostic needs `shiftable_set`, which is `shiftable_valu` in the scheduler. In `cpsat_scheduler.rs`, `shiftable_valu` is computed *after* `final_asap`/`final_alap` are set (around 2085–2093). Call the check **after** that block (around 2093), when `verbose` is true, passing the same `asap`/`alap` (at that point they are `final_asap`/`final_alap`) and `shiftable_valu` as `shiftable_set`. Do not call it at 2066–2082, since `shiftable_set` is not yet available there.

**Tests:**

- **Unit test (preferred):** In `cpsat_scheduler.rs` tests, build a minimal `Vec<SsaOp>` with known RAW edges: set each op's `dep_ids` so the intended (pred, succ) pairs are the RAW edges (e.g. two ops with `succ.dep_ids = {pred.id}`). Inject `asap`/`alap` maps that violate one edge (e.g. pred ALAP 5, succ ASAP 6, offset 2 → need ASAP >= 7). Call the new check function and assert violation count >= 1 and that the reported sample includes that edge. For the offset-2 case, the **predecessor** must be SelectChoice with `sub_temp_vreg: Some(...)` so the trait returns 2.
- **Alternative:** Integration-style test that runs `schedule_ssa_cpsat` with `verbose: true` and a small config that uses slot tightening, then parses stderr/log for the diagnostic line; heavier and more brittle.

**Result:** Violations are visible in logs without changing solver behavior.

---

## Commit 4: Debug-fix flag and precedence-consistency pass

**Scope:** `scheduler_rs/src/cpsat_scheduler.rs`: config flag + fixpoint pass that relaxes ASAP/ALAP using the same trait, then use result as `final_asap`/`final_alap`.

**Code:**

- Add `debug_fix_precedence_bounds: bool` to `SsaCpsatConfig` (default `false`).
- When set, after the diagnostic, run the precedence-consistency pass (same offset via trait) and set `final_asap`/`final_alap` from the relaxed maps.
- No new if/else on dependency type; reuse `precedence_offset_for_consumer`.
- **Fix direction:** Tightening only tightens (ASAP later, ALAP earlier); see main plan section "Tightening only tightens (monotonicity)". So the pass only relaxes the consumer side: `ASAP[succ] = max(ASAP[succ], ALAP[pred] + offset)`; no need to relax ALAP.

**Tests:**

- **Integration test:** Run CP-SAT with a config that (1) enables ASAP/ALAP slot tightening and (2) sets `debug_fix_precedence_bounds: true`, on a small instance that is INFEASIBLE with slot-only and FEASIBLE with the fix. **Choose the exact config after repro** (main plan Step 1): e.g. 3r/64b or 2r/64b often shows INFEASIBLE with slot-only; 2r/32b may be FEASIBLE either way. Assert solver status is FEASIBLE (or not INFEASIBLE). If no small repro is found, the test can be skipped or relaxed (e.g. assert no crash / document as best-effort). Gate with `#[cfg(feature = "cpsat")]`; if the run is long, use `#[cfg(all(feature = "cpsat", feature = "ci-tests"))]` and document in CLAUDE.md so CI can run it when needed.

**Result:** Default behavior unchanged; with the flag, we can confirm the model becomes FEASIBLE.

---

## Commit 5: Fix initial ASAP/ALAP computation

**Scope:** `scheduler_rs/src/cpsat_scheduler.rs`: `compute_ssa_asap`, `compute_ssa_alap`, and their call sites.

**Code:**

- **Shiftable set before bounds:** In the main CP-SAT flow, compute `shiftable_valu` and `shiftable_set` **before** calling `compute_ssa_asap` / `compute_ssa_alap` (move the existing logic that builds them from ops + `enable_valu_shift` to run earlier, before the initial bounds step).
- **compute_ssa_asap:** Add parameter `shiftable_set: &HashSet<usize>` (op indices). Build `id_to_idx` from `ops`. When propagating to a successor, use `offset = ops[pred_idx].precedence_offset_for_consumer(shiftable_set.contains(&pred_idx))` and `new_asap = my_asap + offset` instead of fixed `+ 1`.
- **compute_ssa_alap:** Add parameter `shiftable_set: &HashSet<usize>`. Build `id_to_idx` from `ops`. When propagating to a predecessor, use the same trait for `offset` and `new_alap = my_alap - offset` instead of fixed `- 1`.
- **Call sites:** (1) Main scheduler (around 2034–2044): compute shiftable set first, then pass `&shiftable_set` into both functions. (2) **compute_ssa_asap_alap_bounds:** Compute `shiftable_set` inside the function from `config.enable_valu_shift` and `ops` using the same filter as in `schedule_ssa_cpsat` (VALU ops with `opcode.can_shift_to_alu()`), then pass `&shiftable_set` into `compute_ssa_asap` and `compute_ssa_alap`. No new parameter needed; the config already has `enable_valu_shift`. Optional: if greedy hints should use the same precedence as CP-SAT when VALU shift is enabled, ensure the caller in `tick_tock_ssa` passes a bounds config that includes the actual `enable_valu_shift` for the run (if available at that call site).

**Tests:**

- **Unit tests:** Existing or new tests that call `compute_ssa_asap` / `compute_ssa_alap` with a small DAG that includes a SelectChoice with `sub_temp_vreg.is_some()` or a shiftable VALU op; assert that the returned ASAP/ALAP reflect the correct offset (e.g. ASAP[succ] >= ASAP[pred] + 2 for the FMA case, or + 8 when pred is shiftable). Can extend existing `compute_ssa_asap`/`compute_ssa_alap` tests in cpsat_scheduler.rs.
- **Regression:** Run the same repro config (e.g. 3r/64b slot-only) without the debug-fix flag; solver should be FEASIBLE (or at least fewer violations in verbose diagnostic) because initial bounds are now correct.

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
| 1      | `PrecedenceOffset` trait and `SsaOp` impl in types.rs               | Unit tests in types.rs: normal → 1, SelectChoice+sub_temp_vreg → 2, shiftable → 8          |
| 2      | Align/expand existing doc + trait/impl comments                    | Optional doc-exists check; otherwise none                                                  |
| 5      | Fix compute_ssa_asap / compute_ssa_alap with trait + shiftable_set | Unit tests for correct offset; regression (e.g. 3r/64b FEASIBLE without debug-fix)        |
| 3      | Violation-check function + call when verbose                      | Unit test: mock asap/alap/ops with one violation, assert count and sample                  |
| 4      | `debug_fix_precedence_bounds` + fixpoint pass                      | Integration test: run with flag + slot tightening, assert FEASIBLE (e.g. 2r/32b or 3r/64b) |

This keeps "introducing offsets into SsaOp" as a distinct, test-covered commit; adds a required fix for the initial ASAP/ALAP bug; and avoids special if/else at call sites as in the main plan.
