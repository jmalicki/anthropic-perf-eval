# ASAP/ALAP Slot Tightening: Design and Rules

This document is the **canonical design reference** for slot-based tightening of scheduling bounds (ASAP/ALAP). It defines the guarantee we want, which rules are safe or unsafe, and how the implementation behaves. For the specific T7 counterexample investigation (2r/16b, 368 vs 36 cycles, Z3 proofs), see [SLOT_TIGHTENING_T7_COUNTEREXAMPLE.md](SLOT_TIGHTENING_T7_COUNTEREXAMPLE.md).

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
- **Implementation:** When slot tightening is enabled (`--enable-asap-alap-tightening`), the **default** is R_precedence_only: we skip the energetic slot loop. We still run precedence propagation, greedy-cap (if hints exist), and fixpoint. To get the old energetic behavior (ASAP-only + greedy-cap, which can over-tighten), pass **`--energetic-slot-tightening`** in addition to `--enable-asap-alap-tightening`.

**Code:** `scheduler_rs/src/cpsat_scheduler/bounds.rs` (`use_energetic_slot_tightening`), `model.rs`, `kernel_build_args.py`.

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
| **R_precedence_only** (default: tightening on, no energetic) | **Yes** (proved) | We add no slot-based exclusions; only precedence + fixpoint + greedy-cap. Trivially preserves every feasible schedule. |
| **R_entailed** (exclude only when (O,t) is UNSAT) | **Yes** (proved in Z3) | **T7-strong-guarantee-entailed-exclusions**: refuting (precedence ∧ capacity ∧ ¬exclusions) yields UNSAT → no feasible schedule removed. Not implemented in Rust (needs a solver). |
| **Energetic** (`--enable-asap-alap-tightening` + `--energetic-slot-tightening`) | **No** (proved to fail) | **T7-strong-guarantee-fails-asap-only**: Z3 returns **SAT** → there exists a feasible schedule (e.g. (0,1,2)) that ASAP-only exclusions remove. So the strong guarantee is **violated**. With greedy-cap we still guarantee **at least one** schedule remains (the canonical G). So: **in-between** — not strong-guarantee safe; observed to keep G when hints are present. |
| **R_precedence_demand** (one-shot demand from precedence windows) | **No** (proved unsafe) | Counterexample: 3-op chain, cap 1; we would exclude (O1,1) but (1,2,3) is feasible. **Do not use.** |

**Summary:**

- **Proved safe (strong guarantee):** Slot tightening off, R_precedence_only, R_entailed.
- **Proved unsafe (violates strong guarantee):** R_precedence_demand (do not use), and the energetic rule (ASAP-only exclusions) — Z3 shows we remove at least one feasible schedule.
- **In-between:** Energetic + greedy-cap: we do **not** have the strong guarantee (we proved it fails), but we **do** guarantee at least one feasible schedule (G) when greedy hints are passed; no formal proof that “at least one” holds when hints are absent.

## Implementation summary

| Option | Effect |
|--------|--------|
| Slot tightening **on** (default) | **R_precedence_only**: precedence, fixpoint, greedy-cap; **no** energetic slot loop. **Proved safe (strong guarantee).** Use `--no-enable-asap-alap-tightening` to disable. |
| `--no-enable-asap-alap-tightening` | No slot-based tightening; precedence + optional greedy-cap only. **Proved safe.** |
| `--enable-asap-alap-tightening` + `--energetic-slot-tightening` | Energetic slot loop (ASAP-only + greedy-cap). **Proved to violate strong guarantee**; with greedy-cap, at least G remains. Use for experiments only. |

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

## Related documents

- [LIVENESS_SPECIFICATION.md](LIVENESS_SPECIFICATION.md) — Axioms and theorems (T7 one-step lemma, etc.).
- [SLOT_TIGHTENING_T7_COUNTEREXAMPLE.md](SLOT_TIGHTENING_T7_COUNTEREXAMPLE.md) — Investigation: 2r/16b counterexample, Z3 proofs, ablation (energetic vs off), why 368 cycles.
- [verification/README.md](verification/README.md) — Z3 theorem list and preservation vs soundness.
