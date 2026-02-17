# ASAP/ALAP Slot Tightening: Design and Rules

This document is the **canonical design reference** for slot-based tightening of scheduling bounds (ASAP/ALAP). It defines the guarantee we want, which rules are safe or unsafe, and how the implementation behaves.

- **Deep dive (recommended for newcomers):** [ENERGETIC_SLOT_TIGHTENING_DEEP_DIVE.md](ENERGETIC_SLOT_TIGHTENING_DEEP_DIVE.md) — explains from first principles why energetic tightening broke, the algorithm in detail, and fix options.
- **T7 counterexample investigation (2r/16b, 368 vs 36 cycles, Z3 proofs):** [SLOT_TIGHTENING_T7_COUNTEREXAMPLE.md](SLOT_TIGHTENING_T7_COUNTEREXAMPLE.md).

## Purpose of slot tightening

Before building the CP-SAT model we compute for each op a **window** [ASAP, ALAP] of cycles it may be scheduled in. **Slot tightening** narrows those windows using resource capacity: if at cycle `t` the demand from other ops (in their windows) plus this op’s slots would exceed capacity, we exclude `t` for this op. Tighter domains reduce the solver’s search space. We must not over-tighten: **no previously feasible schedule may become infeasible** (the strong guarantee).

## Strong guarantee

**Strong guarantee:** Tightening only excludes (O,t) that **no** feasible schedule uses. So every schedule that satisfies precedence and capacity is still feasible after tightening.

A weaker property (“at least one feasible schedule remains”) is insufficient: the original T7 claim in [LIVENESS_SPECIFICATION.md](LIVENESS_SPECIFICATION.md#t7) was that strong, but composition of multiple “one-step” exclusions can remove all feasible schedules (Z3 counterexample T7-two-step). We therefore design rules R that **provably** preserve the strong guarantee.

## Safe rules

### R_entailed (proved; requires a solver)

**Rule:** Only add exclusion (O,t) when **no** feasible schedule has O at t — i.e. when (precedence ∧ capacity ∧ cycle_O = t) is **UNSAT**.

- **Z3:** The strong guarantee for this rule is proved: **T7-strong-guarantee-entailed-exclusions** in `verification/theorems.py`; run `uv run python -m verification.verify -t T7-strong-guarantee-entailed-exclusions`.
- **Does not untighten:** R only removes cycles from domains.
- **Implementation:** Requires a SAT/UNSAT check per (O,t) or a batch solver query. The **Rust scheduler has no solver**, so it cannot implement R_entailed directly. Any implementation would live in a layer that calls Z3 or CP-SAT (e.g. a Python pre-pass).

### R_precedence_only (solver-free; implemented)

**Rule:** Do **not** add any slot-based exclusions. Use only precedence-derived ASAP/ALAP (and optionally greedy-based ALAP tightening, precedence fixpoint, and greedy-cap). No demand profile, no energetic loop.

- **Strong guarantee:** Trivially satisfied — we never exclude a cycle that precedence allows.
- **Does not untighten:** We only set bounds from precedence (and fixpoint).
- **Implementation:** Choose one of four modes via **`--asap-alap-tightening`**: `none`, `simple` (R_precedence_only: precedence + fixpoint only), `symmetry-aware` (default), or `energetic`. Simple = no symmetry ordering, no energetic loop. Symmetry-aware = R_symmetry_ordering. Energetic = symmetry-aware plus energetic slot loop.

**Code:** `scheduler_rs/src/cpsat_scheduler/bounds.rs` (`AsapAlapTightening` enum), `model.rs`, `kernel_build_args.py`.

### R_symmetry_ordering (solver-free; implemented; strong guarantee)

**Rule:** Use the **same ordering as symmetry breaking** (ordering groups, sort by greedy then ALAP). Within each group, for each engine, the k-th op (1-based) that definitely uses that engine cannot run before cycle `ceil(k/C) - 1`, because the model adds `cycle[i] <= cycle[j]` for consecutive pairs and at most C ops fit per cycle. So we set ASAP(op) = max(ASAP(op), ceil(k/C) - 1) for that op.

- **Strong guarantee:** Every feasible schedule that satisfies the model’s symmetry constraints already has each op at least at that cycle. So we only exclude cycles that no such schedule uses.
- **Does not untighten:** We only push ASAP forward (or leave unchanged).
- **Implementation:** Use **`--asap-alap-tightening symmetry-aware`** (default). See `tighten_asap_by_symmetry_ordering` in `bounds.rs`.

**Code:** `scheduler_rs/src/cpsat_scheduler/bounds.rs` (`tighten_asap_by_symmetry_ordering`), `model.rs`, `kernel_build_args.py`.

## Rules tried and proved unsafe (for posterity)

### R_precedence_demand (one-shot; no solver) — do not use

**Idea:** Compute the demand profile **once** from **precedence-only** ASAP/ALAP (no iteration); for each op O and each cycle t in O’s precedence window, exclude (O,t) when `demand_other(t) + slots(O) > cap`, where `demand_other(t)` = sum over other ops O′ that have t in their precedence window.

**Why it is not safe:** `demand_other(t)` counts “other ops that *could* be at t.” In any single feasible schedule, at most one op is at t. So we **overcount**: we assume all such ops are at t simultaneously, and can exclude (O,t) even when some feasible schedule has O at t.

**Counterexample:** 3 ops (O1, O2, O3), 5 cycles (0..4), capacity 1, chain precedence O2≥O1+1, O3≥O2+1. Precedence windows: O1 [0,4], O2 [1,4], O3 [2,4]. At cycle 1, O2 has 1 in its window, so `demand_other(O1, 1) = 1`. Then `1 + slots(O1) = 2 > 1`, so we would exclude (O1, 1). But the schedule (c1=1, c2=2, c3=3) is feasible and has O1 at 1. So the rule would remove a feasible schedule and **violates the strong guarantee**.

**Conclusion:** R_precedence_demand must **not** be used. Do not revisit this rule without either (a) a SAT/UNSAT check per (O,t) (R_entailed), or (b) a proved sufficient condition that makes the one-shot demand safe.

## Status of each option (proved safe / proved unsafe / in-between)

| Option or rule | Strong guarantee (all feasible schedules remain)? | Evidence |
|----------------|---------------------------------------------------|----------|
| **Slot tightening off** (`--no-enable-asap-alap-tightening`) | **Yes** (trivial) | We add no exclusions. |
| **R_precedence_only** (no slot exclusions) | **Yes** (proved) | We add no slot-based exclusions; only precedence + fixpoint + greedy-cap. Trivially preserves every feasible schedule. |
| **R_symmetry_ordering** (default when tightening on) | **Yes** (proved) | Slot exclusions from symmetry + capacity only: “k-th op in group per engine cannot run before ceil(k/C)-1.” Preserves all schedules that satisfy the model’s ordering constraints. |
| **R_entailed** (exclude only when (O,t) is UNSAT) | **Yes** (proved in Z3) | **T7-strong-guarantee-entailed-exclusions**: refuting (precedence ∧ capacity ∧ ¬exclusions) yields UNSAT → no feasible schedule removed. Not implemented in Rust (needs a solver). |
| **Energetic** (`--enable-asap-alap-tightening` + `--energetic-slot-tightening`) | **Yes** (contributing-only, Z3-proven) | Uses **contributing-only undercounting**: only excludes (O,t) when O contributes (`asap[O]==t`) and total_min_demand(t) > cap. Non-contributing case skipped (not proved). All exclusions are covered by Z3 (T7-undercount-*). |
| **R_precedence_demand** (one-shot demand from precedence windows) | **No** (proved unsafe) | Counterexample: 3-op chain, cap 1; we would exclude (O1,1) but (1,2,3) is feasible. **Do not use.** |

**Summary:**

- **Proved safe (strong guarantee):** Slot tightening off, R_precedence_only, R_symmetry_ordering, R_entailed.
- **Proved sound (contributing-only undercounting):** Energetic — all exclusions Z3-proven (T7-undercount-*); non-contributing case omitted.
- **Proved unsafe (violates strong guarantee):** R_precedence_demand (do not use).

## Implementation summary

| Option | Effect |
|--------|--------|
| Slot tightening **on** (default) | **R_symmetry_ordering** (safe slot exclusions) + R_precedence_only (precedence, fixpoint, greedy-cap). **No** energetic loop unless requested. **Proved safe (strong guarantee).** Use `--no-symmetry-aware-slot-tightening` to get R_precedence_only only; use `--no-enable-asap-alap-tightening` to disable all tightening. |
| `--no-enable-asap-alap-tightening` | No slot-based tightening; precedence + optional greedy-cap only. **Proved safe.** |
| `--enable-asap-alap-tightening` + `--energetic-slot-tightening` | Contributing-only undercounting. **Z3-proven sound** — all exclusions covered by T7-undercount-* theorems. Requires ordering (built automatically). |

## Ablation: R_precedence_only vs tightening off (2r/16b)

Two-arm ablation comparing **no tightening** vs **R_precedence_only** (tightening on, no energetic):

```bash
uv run python scripts/run_ablation.py \
  --rounds 2 --batch-size 16 --time-limit 300 \
  --arm "no-tightening:--no-enable-asap-alap-tightening" \
  --arm "precedence-only-tightening:" \
  --output ablation_2r_16b_tightening_vs_off.json
uv run python scripts/analyze_ablation.py ablation_2r_16b_tightening_vs_off.json
```

**Result (300s time limit):** Both arms **36 cycles**, OPTIMAL. Solve times ~6.3s. So R_precedence_only does not worsen makespan vs tightening off for this config; the default (tightening on without energetic) is safe and performance-neutral here.

## Ablation: Energetic vs no-energetic (3r/16b)

```bash
uv run python scripts/run_ablation.py \
  --rounds 3 --batch-size 16 --time-limit 120 \
  --arm "no-energetic:" \
  --arm "energetic:--energetic-slot-tightening" \
  --output ablation_3r_16b_energetic.json
```

**Result (120s time limit):** Both arms **55 cycles**, OPTIMAL. Energetic produces at least as good schedules as symmetry-only; no over-tightening.

## Related documents

- [ENERGETIC_SLOT_TIGHTENING_DEEP_DIVE.md](ENERGETIC_SLOT_TIGHTENING_DEEP_DIVE.md) — Deep dive: why energetic tightening broke, algorithm in detail, fix options.
- [LIVENESS_SPECIFICATION.md](LIVENESS_SPECIFICATION.md) — Axioms and theorems (T7 one-step lemma, etc.).
- [SLOT_TIGHTENING_T7_COUNTEREXAMPLE.md](SLOT_TIGHTENING_T7_COUNTEREXAMPLE.md) — Investigation: 2r/16b counterexample, Z3 proofs, ablation (energetic vs off), why 368 cycles.
- [verification/README.md](verification/README.md) — Z3 theorem list and preservation vs soundness.
