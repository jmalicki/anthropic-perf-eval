# Precedence Offsets and Slot-Tightening INFEASIBLE Plan

## Overview

Slot-only ASAP/ALAP tightening can make the CP-SAT model INFEASIBLE because it uses only capacity and does not preserve precedence. Bounds must respect: for each RAW edge (pred, succ), **ASAP[succ]** uses the **minimum** delay (earliest consumer can run) and **ALAP[pred]** uses the **maximum** delay (latest pred can run). Current code uses a fixed +1 everywhere and does not distinguish normal (1), SelectChoice FMA (2), or shiftable VALU (min 1, max 8). This plan adds a precedence-offset trait on the op type (two methods: for ASAP and for ALAP), debugging-only diagnostic and optional fix, and a commit breakdown with tests.

**Reference:** [docs/ASAP_ALAP_PRECEDENCE_OFFSETS.md](ASAP_ALAP_PRECEDENCE_OFFSETS.md) defines ASAP vs ALAP semantics and the min/max delay table.

---

## Root Cause

1. **Initial ASAP/ALAP** use +1 everywhere; not valid for every resolution (FMA needs 2; shiftable needs max 8 for ALAP, min 1 for ASAP).
2. **Slot tightening** uses only per-engine capacity. It does not preserve ASAP[succ] ≥ ASAP[pred] + min_delay nor ALAP[pred] ≤ ALAP[succ] − max_delay, so it can produce INFEASIBLE domains.

---

## Precedence Offset as a Trait (no if/else at call sites)

**Goal:** Centralize delay logic so diagnostic and precedence-consistency pass have no special if/else on dependency type, and adding a new op only updates the trait impl.

**Trait (types.rs):**

- `fn precedence_offset_for_asap(&self, predecessor_is_shiftable: bool) -> i32`  
  Min delay: used when tightening **ASAP[succ]** (consumer’s lower bound). Returns 1 (normal), 2 (SelectChoice FMA), 1 (shiftable).
- `fn precedence_offset_for_alap(&self, predecessor_is_shiftable: bool) -> i32`  
  Max delay: used when tightening **ALAP[pred]** (predecessor’s upper bound). Returns 1 (normal), 2 (SelectChoice FMA), 8 (shiftable).

**Impl for SsaOp:** SelectChoice with `sub_temp_vreg.is_some()` → both 2. Else if `predecessor_is_shiftable` → ASAP 1, ALAP 8. Else → both 1.

**Call sites (cpsat_scheduler.rs):** For each (pred, succ):  
`asap_offset = ops[pred].precedence_offset_for_asap(shiftable_set.contains(&pred));`  
`alap_offset = ops[pred].precedence_offset_for_alap(shiftable_set.contains(&pred));`  
Then: ASAP[succ] ≥ ASAP[pred] + asap_offset, ALAP[pred] ≤ ALAP[succ] − alap_offset. No branching on dependency type.

---

## Precedence-Consistency as Debugging Only

1. **Diagnostic (when verbose):** After slot/scratch tightening, scan RAW edges and count violations: ASAP[succ] < ASAP[pred] + asap_offset, or ALAP[pred] > ALAP[succ] − alap_offset. Log count and a sample.
2. **Optional fix (debug flag):** When `debug_fix_precedence_bounds` is true, run a fixpoint: for each edge apply the correct offsets (via trait), relax bounds until stable, then set final_asap/final_alap. Default remains unchanged.

---

## Commit Breakdown (each with tests where applicable)

### Commit 1: Precedence offset trait and impl on SsaOp

- **Scope:** [scheduler_rs/src/types.rs](scheduler_rs/src/types.rs) only.
- **Code:** Define trait with `precedence_offset_for_asap` and `precedence_offset_for_alap`; implement for `SsaOp` as above.
- **Tests (types.rs):** Unit tests: normal op → (1, 1); SelectChoice with sub_temp_vreg → (2, 2); shiftable (non-SelectChoice) → (1, 8).

### Commit 2: Documentation

- **Scope:** [docs/ASAP_ALAP_PRECEDENCE_OFFSETS.md](ASAP_ALAP_PRECEDENCE_OFFSETS.md) (already added); trait/impl doc comments in types.rs referencing it.
- **Tests:** None required for doc-only.

### Commit 3: Diagnostic (verbose)

- **Scope:** [scheduler_rs/src/cpsat_scheduler.rs](scheduler_rs/src/cpsat_scheduler.rs): after tightening, when verbose, check violations using both trait methods; log count and sample.
- **Tests:** Unit test with mock ops and asap/alap that violate one edge; assert violation count and sample (use correct asap_offset vs alap_offset per edge).

### Commit 4: Debug-fix flag and precedence-consistency pass

- **Scope:** Add `debug_fix_precedence_bounds` to config; run fixpoint using both trait methods; set final_asap/final_alap when flag is set.
- **Tests:** Integration test: run with slot tightening + flag on a config that is INFEASIBLE without fix; assert FEASIBLE (e.g. 2r/32b or 3r/64b). Gate with `cpsat` (and optionally `ci-tests` if slow).

---

## Order

1. Commit 1 (trait + impl + tests).
2. Commit 2 (docs).
3. Commit 3 (diagnostic + test).
4. Commit 4 (debug-fix + integration test).

Repro on small instances (2r/32b, 2r/64b) with slot-only can be done before or in parallel with Commit 1.
