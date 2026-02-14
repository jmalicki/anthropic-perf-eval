# T7 counterexample: slot tightening excludes a feasible schedule

## Claim (T7)

From [LIVENESS_SPECIFICATION.md](LIVENESS_SPECIFICATION.md#t7):  
"If we push ASAP forward or ALAP backward using slot capacity (A6) so that the demand in each op's window respects the cap, the tightened bounds **still contain at least one feasible schedule**."

## Counterexample

- **Config:** 2 rounds, batch 16, zero-gap CP-SAT (same config for both runs).
- **With slot tightening enabled:** 368-cycle schedule; cycles 37–331 are empty (295 empty cycles).
- **With slot tightening disabled** (`--no-enable-asap-alap-tightening`): **36-cycle** schedule that **passes validation**.

Therefore a feasible schedule (the 36-cycle one) exists, but the tightened domains **exclude** it when tightening is on. The current behavior contradicts T7.

See [2r_16b_zero_gap_validation_findings.md](2r_16b_zero_gap_validation_findings.md) §5a for the zero-gap + no-tightening run and the explanation (slot ASAP/ALAP tightening creating precedence violations; the fixpoint repair then pushes consumer ASAP to 349+).

## Z3 proof scope

The repo contains a Z3 verification for T7 in `verification/theorems.py` (`check_t7_slot_tightening_one_step`). It proves the **one-step** lemma: when we exclude cycle `t` for op `O` because at `t` the demand from other ops plus `slots(O)` exceeds capacity, then no feasible schedule can have `O` at `t` (the negation is UNSAT). It does **not** prove that the **full** pipeline (iterative slot tightening over all ops and windows, followed by the precedence fixpoint) preserves at least one feasible schedule.

**Two-step composition fails to verify:** A stronger claim—“after two valid one-step tightenings, a feasible schedule still exists”—is encoded in `check_t7_slot_tightening_two_step_composition` (T7-two-step). Model: 2 ops, 3 cycles, capacity 1, precedence O2 ≥ O1+1; we apply two valid one-step exclusions (O1 not at 0, O2 not at 2) and ask for a feasible (c1, c2). Z3 returns **UNSAT**: no feasible schedule exists. So the multi-step “composition preserves feasibility” theorem **fails to verify** (we have a counterexample). Run: `uv run python -m verification.verify -t T7-two-step`; the script reports `COUNTEREXAMPLE (unsat = no feasible schedule; composition theorem does not hold)`.

**Sufficient-condition exploration (Z3):** The same 2-op 3-cycle model is used to check that a **single** valid exclusion does not remove all feasible schedules: `T7-one-exclusion-asap` (only O1≠0) and `T7-one-exclusion-alap` (only O2≠2) both yield **SAT**. So one-step exclusions are safe in isolation; composition of two such exclusions (ASAP for O1, ALAP for O2) is what removes the last feasible (1,2) and yields UNSAT. A sufficient condition for composition would therefore need to restrict how multiple exclusions interact (e.g. require that after each exclusion the remaining domain still contains at least one feasible schedule, or restrict to a single "direction" per pass). Run: `uv run python -m verification.verify -t T7-one-exclusion-asap -t T7-one-exclusion-alap`.

## Narrowing the bug

**Order of operations** (in `scheduler_rs/src/cpsat_scheduler/model.rs`): (1) `tighten_asap_alap_by_slots` updates ASAP/ALAP; (2) optional scratch tightening; (3) `fix_precedence_bounds_fixpoint(ops, &final_asap, &final_alap, …)` restores precedence consistency.

- **Slot tightening** (`bounds.rs::tighten_asap_alap_by_slots`): Builds a per-engine demand profile from current op windows `[asap, alap]` (up to 3 iterations). Uses energetic reasoning: if total demand in an op’s window exceeds `cap * window_size`, it pushes that op’s ASAP forward and/or ALAP backward. So the **origin** of over-tightening can be here (e.g. profile from overlapping windows pushes one or more “tail” ops’ ASAP to 332+; or the same op gets pushed in multiple engines).

- **Precedence fixpoint** (`bounds.rs::fix_precedence_bounds_fixpoint`): After tightening, precedence can be violated (e.g. consumer’s ASAP &lt; pred’s ASAP + offset). The fixpoint only **propagates** bounds: forward it increases successor ASAP when `pred_asap + offset > succ_asap`; backward it decreases predecessor ALAP. So if slot tightening already pushed a predecessor’s ASAP to 332, the fixpoint **correctly** pushes every successor’s ASAP to at least 332+offset. That **spreads** the over-tightening to the whole tail.

**Conclusion:** The exclusion is likely **initiated** by slot tightening (wrong or overly aggressive push for some op(s)); the fixpoint then **amplifies** it by propagating those bounds along the precedence graph. A fix should ensure slot tightening never pushes an op’s ASAP/ALAP beyond what is required for feasibility (e.g. validate that the greedy schedule still lies in the tightened domains, or re-check demand using “must be at t” rather than “could be at t” where applicable). See BUGS.md entry “T7 counterexample” and the plan in `.cursor/plans/slot_tightening_and_empty_cycles_*.plan.md`.

---

## Disabling slot tightening removes the empty-cycle stretch

**Reproduction (2r/16b, same problem size):**

| Config | Scheduler JSON | Cycles | Empty stretch | Deciles 1–8 |
|--------|-----------------|--------|---------------|-------------|
| **With** slot tightening (default) | `artifacts/scheduler_2r_16b.json` | 368 | 295 empty (37–331) | All 0% usage |
| **Without** slot tightening | `scheduler_output_2r_16b_no_tightening.json` | 36 | None | Work spread across all deciles |

- **With tightening:** Per-decile slot usage shows deciles 1–8 (cycles 37–331) completely empty; only decile 0 (0–36) and decile 9 (332–367) have ops. Theoretical lower bound script: actual 368 cycles vs theoretical minimum 14 cycles (96.2% overhead).
- **Without tightening:** Same 2r/16b config with `--no-enable-asap-alap-tightening` and `--force-zero-gap --cpsat-time-limit 300`: 36-cycle schedule that **passes validation**. Per-decile usage shows work in every decile (no long empty stretch). Theoretical lower bound: actual 36 cycles vs theoretical minimum 14 (61.1% overhead).

**Conclusion:** Disabling slot tightening removes the empty-cycle phenomenon for this config. The no-tightening run is reproducible with:

```bash
uv run generate_ssa_kernel.py --rounds 2 --batch-size 16 --no-enable-asap-alap-tightening --force-zero-gap --cpsat-time-limit 300 --output-json scheduler_output_2r_16b_no_tightening.json --validate
```

Then run `uv run scripts/slot_usage_by_decile.py scheduler_output_2r_16b_no_tightening.json --markdown` and `uv run scripts/theoretical_lower_bounds.py scheduler_output_2r_16b_no_tightening.json` to confirm compact schedule and spread deciles.

---

## Ablation: with vs without tightening across configs

Ablation runs (2 arms: default vs `--no-enable-asap-alap-tightening`, CP-SAT time limit 120s) show the empty-stretch phenomenon in **all** tested configs: tightening consistently yields much higher makespans.

| Config   | With tightening (cycles) | No tightening (cycles) | Difference        |
|----------|--------------------------|-------------------------|-------------------|
| 2r/8b    | 52                       | 36                      | −16 (−30.8%)      |
| 2r/16b   | 368                      | 36                      | −332 (−90.2%)     |
| 2r/32b   | 369                      | 43                      | −326 (−88.3%)     |
| 3r/16b   | 566                      | 55                      | −511 (−90.3%)     |
| 4r/16b   | 795                      | 75                      | −720 (−90.6%)     |

So there is no small “threshold” config where tightening is safe: even 2r/8b gets 52 vs 36. Results saved as `ablation_tightening_2r_8b.json`, `ablation_tightening_2r_16b.json`, etc.

---

## Per-decile and theoretical lower bound (other configs)

For **2r/16b with tightening** (368 cycles), per-decile slot usage shows deciles 1–8 (cycles 37–331) at 0% usage; only decile 0 and decile 9 have work. The theoretical lower bound script gives actual 368 vs theoretical minimum 14 cycles (96.2% overhead). See the "Disabling slot tightening" section above.

For **other configs** (2r/8b, 2r/32b, 3r/16b, 4r/16b), the ablation table shows that "with tightening" consistently yields many more cycles than "no tightening"; the same empty-stretch pattern (work concentrated in early and late deciles, long empty middle) is expected when tightening is on. To confirm for a specific config, generate with tightening, then run:

```bash
uv run scripts/slot_usage_by_decile.py <scheduler_json> --markdown
uv run scripts/theoretical_lower_bounds.py <scheduler_json>
```

---

## What this means for bugs; strengthening conditions

### Does this show we have bugs?

**Yes**, in the sense that observed behavior violates the original T7 claim (“tightened bounds still contain at least one feasible schedule”). Concretely:

- **Spec/design bug:** The spec claimed composition preserves feasibility; we proved that is false (T7-two-step). The implementation follows the design (iterative tightening + fixpoint), so it can exclude feasible schedules.
- **Observed bug:** For 2r/16b with tightening we get a 368-cycle schedule with 295 empty cycles; without tightening the same config yields a 36-cycle **valid** schedule. So a feasible schedule exists but is excluded (or the solver ends up with a bad one because domains are tail-heavy).

So we have both a **theorem bug** (the original T7 statement was too strong) and a **behavioral bug** (bad makespans / empty stretches when tightening is on). The code is “correct” relative to the one-step lemma (each exclusion is sound) but **incorrect** relative to the intended guarantee (at least one feasible schedule remains).

### Goal: tighten beyond greedy; CP-SAT is the answer

Everything before CP-SAT (greedy, ASAP/ALAP tightening, etc.) is **cheap preprocessing** to help CP-SAT find good solutions faster. CP-SAT is the source of truth. Greedy gives a warm-start hint; slot tightening is meant to give **tighter domains than precedence alone** — ideally **tighter than where greedy put the ops** — so CP-SAT's search space is smaller. The **entire point of ASAP/ALAP tightening is to tighten beyond greedy**, not to stay at or inside greedy. We only must not over-tighten so much that we exclude every feasible schedule (the composition bug).

### Greedy-cap: adopted as the proven safe fix

We **adopt greedy-cap** as the implementation fix: when building the CP-SAT model we pass the greedy schedule as canonical and clamp slot tightening so we never exclude the canonical cycle for any op (Z3 theorem **T7-greedy-cap-preserves-G**). This guarantees at least one feasible solution (the greedy schedule G) and fixes the 368-cycle regression (no empty domains excluding G). The **ideal** — tightening that goes beyond greedy without excluding any feasible schedule — remains open; see the section below for strengths, weaknesses, and possible future work.

### What we actually need (open)

We need tightening that: (1) **goes beyond greedy** — smaller domains than precedence-only, ideally tighter than greedy's placement where slot reasoning justifies it; and (2) **does not violate feasibility** — does not exclude all feasible schedules (no composition bug). The **greedy-cap** fix ensures (2) by construction (we never exclude G). Achieving (1) while preserving (2) in a stronger way (e.g. tightening beyond greedy where provably safe) remains an open problem; see BUGS.md and the "Narrowing the bug" section above.

---

## Greedy-cap fix: strengths, weaknesses, future work

### Strengths

- **Z3-proven:** The canonical schedule G remains feasible (theorem T7-greedy-cap-preserves-G). See `verification/` and `verification/README.md`.
- **Fixes the 368-cycle regression:** CP-SAT always has at least one feasible solution (G) in the domain; we no longer exclude it via slot tightening.
- **Slot tightening still helps:** Tightening is redundant for feasibility; its role is to aid propagation. Greedy-cap keeps that benefit (we still tighten where we don't exclude G) while guaranteeing correctness.
- **Small change:** One new parameter (`greedy_cycles`) and a clamp at the point where we update ASAP/ALAP in `tighten_asap_alap_by_slots` (see `scheduler_rs/src/cpsat_scheduler/bounds.rs`).

### Weaknesses

- **Weaker propagation when capping at G:** We never exclude G(O) for any op, so for ops where domain-based tightening would have pushed past G, domains stay looser. That can mean weaker propagation and possibly longer solve times in some configs.
- **No clamp when greedy hints are absent:** When no greedy hints are available (e.g. pipeline bounds pass, or tests), we pass `None` and do not clamp; in those paths the old (domain-based) behavior remains and can still over-tighten if tightening is enabled.
- **Correctness and propagation strength depend on greedy quality:** If G is poor, "cap at G" still leaves a large search space.

### Possible future work

- **Tighter safe rule in Z3:** Prove a rule that allows more exclusions while still preserving a feasible schedule (e.g. order-based or "exclude only when no feasible schedule uses (O,t)"); then port that rule to recover propagation strength.
- **Improve greedy:** Better G implies tighter effective domains under greedy-cap and better propagation.
- **Post-tightening check:** After each engine pass, check that G is still in the domains; if not, revert the exclusions that removed it so we can apply more exclusions when they don't touch G.
- **Optional makespan cap:** Add `makespan <= greedy_makespan + slack` in the CP-SAT model as a temporary safeguard if needed.
- **Formalize batch symmetry:** Document that G is the canonical representative under batch symmetry and that preserving G preserves one schedule per equivalence class.

