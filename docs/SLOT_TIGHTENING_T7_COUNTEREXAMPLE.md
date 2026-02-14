# T7 counterexample: slot tightening excludes a feasible schedule

**Default: off.** Slot tightening is off by default so schedules are never worsened. When enabled (opt-in), the implementation uses the **ASAP-only** safe subset (Z3 T7-asap-only-composition). Use `--enable-asap-alap-tightening` to opt in.

## Claim (T7)

From [LIVENESS_SPECIFICATION.md](LIVENESS_SPECIFICATION.md#t7):  
"If we push ASAP forward or ALAP backward using slot capacity (A6) so that the demand in each op's window respects the cap, the tightened bounds **still contain at least one feasible schedule**."

**Stronger goal:** What we actually want is that **all** previously feasible schedules are still feasible—i.e. tightening only excludes (O,t) that no feasible schedule uses, so no feasible schedule is removed. The spec's "at least one" is a weaker claim; the counterexample shows we can violate even that.

**Do our Z3 proofs prove the strong guarantee?** **Yes in one case; we also try for ASAP-only and preserve the failure.** (1) **T7-strong-guarantee-entailed-exclusions:** We prove that when exclusions are **entailed by precedence**, **all** previously feasible schedules remain feasible (Z3 UNSAT). (2) **T7-strong-guarantee-fails-asap-only:** We **try** to prove the strong guarantee for ASAP-only exclusions (c1≥1, c2≥2 in the small model). The proof **fails** (Z3 SAT = there exists a feasible schedule we removed, e.g. (0,1,2)); that failure is an important result and we keep the theorem so it stays documented. (3) **One-step (T7):** For a **single** exclusion justified by demand, we only remove (O,t) that no feasible schedule uses. (4) **Composition (T7-two-step):** Two "sound" exclusions can together remove all feasible schedules. (5) **Witness checks / Greedy-cap:** At least one remains, or G remains. So we prove the strong guarantee for entailed exclusions; we attempt it for ASAP-only and preserve the fact that the proof fails (with a counterexample).

## Counterexample

- **Config:** 2 rounds, batch 16, zero-gap CP-SAT (same config for both runs).
- **With slot tightening enabled:** 368-cycle schedule; cycles 37–331 are empty (295 empty cycles).
- **With slot tightening disabled** (`--no-enable-asap-alap-tightening`): **36-cycle** schedule that **passes validation**.

Therefore a feasible schedule (the 36-cycle one) exists, but the tightened domains **exclude** it when tightening is on. The current behavior contradicts T7.

See [2r_16b_zero_gap_validation_findings.md](2r_16b_zero_gap_validation_findings.md) §5a for the zero-gap + no-tightening run and the explanation (slot ASAP/ALAP tightening creating precedence violations; the fixpoint repair then pushes consumer ASAP to 349+).

**Debugging / Z3 finding:** The SAT result from T7-strong-guarantee-fails-asap-only **proves an error** in our ASAP/ALAP tightening: it shows there exists a feasible schedule (e.g. (0,1,2)) that the ASAP-only exclusions remove, so we are over-tightening (violating the strong guarantee). Run: `uv run python -m verification.verify -t T7-strong-guarantee-fails-asap-only`. See BUGS.md entry "T7 counterexample".

## Z3 vs CP-SAT feasibility

The distinction is easy to get wrong: in **soundness** proofs we encode a violation; **Z3 SAT** = violation found (bad), **Z3 UNSAT** = theorems hold (good). In **preservation** checks we ask for a witness; **Z3 SAT** = feasible schedule exists (good), **Z3 UNSAT** = no feasible schedule (bad). Full explanation: [verification/README.md](../verification/README.md) § "Z3 vs CP-SAT: two notions of feasibility," and [CLAUDE.md](../CLAUDE.md) § "Verification (Z3)."

## Z3 proof scope

The repo contains a Z3 verification for T7 in `verification/theorems.py` (`check_t7_slot_tightening_one_step`). It proves the **one-step** lemma: when we exclude cycle `t` for op `O` because at `t` the demand from other ops plus `slots(O)` exceeds capacity, then no feasible schedule can have `O` at `t` (we assert the negation; Z3 UNSAT = no counterexample = theorem holds). It does **not** prove that the **full** pipeline (iterative slot tightening over all ops and windows, followed by the precedence fixpoint) preserves at least one feasible schedule.

**Two-step composition fails to verify:** A stronger claim—“after two valid one-step tightenings, a feasible schedule still exists”—is encoded in `check_t7_slot_tightening_two_step_composition` (T7-two-step). Model: 2 ops, 3 cycles, capacity 1, precedence O2 ≥ O1+1; we apply two valid one-step exclusions (O1 not at 0, O2 not at 2) and ask for a feasible (c1, c2). Z3 returns **UNSAT**: no feasible schedule exists. So the multi-step “composition preserves feasibility” theorem **fails to verify** (we have a counterexample). Run: `uv run python -m verification.verify -t T7-two-step`; the script reports `COUNTEREXAMPLE (unsat = no feasible schedule; composition theorem does not hold)`.

**Sufficient-condition exploration (Z3):** The same 2-op 3-cycle model is used to check that a **single** valid exclusion does not remove all feasible schedules: `T7-one-exclusion-asap` (only O1≠0) and `T7-one-exclusion-alap` (only O2≠2) both yield **SAT**. So one-step exclusions are safe in isolation; composition of two such exclusions (ASAP for O1, ALAP for O2) is what removes the last feasible (1,2) and yields UNSAT. A sufficient condition for composition would therefore need to restrict how multiple exclusions interact (e.g. require that after each exclusion the remaining domain still contains at least one feasible schedule, or restrict to a single "direction" per pass). Run: `uv run python -m verification.verify -t T7-one-exclusion-asap -t T7-one-exclusion-alap`.

## Narrowing the bug

**Order of operations** (in `scheduler_rs/src/cpsat_scheduler/model.rs`): (1) `tighten_asap_alap_by_slots` updates ASAP/ALAP; (2) optional scratch tightening; (3) `fix_precedence_bounds_fixpoint(ops, &final_asap, &final_alap, …)` restores precedence consistency.

- **Slot tightening** (`bounds.rs::tighten_asap_alap_by_slots`): Builds a per-engine demand profile from current op windows `[asap, alap]` (up to 3 iterations). Uses energetic reasoning: if total demand in an op’s window exceeds `cap * window_size`, it pushes that op’s ASAP forward (ALAP pulls are not applied; safe form is ASAP-only). So the **origin** of over-tightening could historically be here (e.g. profile from overlapping windows pushes one or more “tail” ops’ ASAP to 332+; or the same op gets pushed in multiple engines).

- **Precedence fixpoint** (`bounds.rs::fix_precedence_bounds_fixpoint`): After tightening, precedence can be violated (e.g. consumer’s ASAP &lt; pred’s ASAP + offset). The fixpoint only **propagates** bounds: forward it increases successor ASAP when `pred_asap + offset > succ_asap`; backward it decreases predecessor ALAP. So if slot tightening already pushed a predecessor’s ASAP to 332, the fixpoint **correctly** pushes every successor’s ASAP to at least 332+offset. That **spreads** the over-tightening to the whole tail.

**Conclusion:** The exclusion is likely **initiated** by slot tightening (wrong or overly aggressive push for some op(s)); the fixpoint then **amplifies** it by propagating those bounds along the precedence graph. The fix direction we focus on is **(1) safe composition**: prove a rule in Z3 that allows multiple exclusions + fixpoint while still preserving at least one feasible schedule, then implement it (see "Possible future work" below). *(2) Symmetry-based exclusion* (prove we only exclude schedules equivalent to one we keep) may be possible in future efforts. A fix should ensure slot tightening never pushes an op’s ASAP/ALAP beyond what is required for feasibility (e.g. validate that the greedy schedule still lies in the tightened domains, or re-check demand using “must be at t” rather than “could be at t” where applicable). See BUGS.md entry “T7 counterexample” and the plan in `.cursor/plans/slot_tightening_and_empty_cycles_*.plan.md`.

---

## Disabling slot tightening removes the empty-cycle stretch

**Reproduction (2r/16b, same problem size):**

| Config | Scheduler JSON | Cycles | Empty stretch | Deciles 1–8 |
|--------|-----------------|--------|---------------|-------------|
| **With** slot tightening enabled | `artifacts/scheduler_2r_16b.json` | 368 | 295 empty (37–331) | All 0% usage |
| **Without** slot tightening (default) | `scheduler_output_2r_16b_no_tightening.json` | 36 | None | Work spread across all deciles |

*(Slot tightening is now off by default; the table describes behavior when it is enabled vs disabled.)*

- **With tightening:** Per-decile slot usage shows deciles 1–8 (cycles 37–331) completely empty; only decile 0 (0–36) and decile 9 (332–367) have ops. Theoretical lower bound script: actual 368 cycles vs theoretical minimum 14 cycles (96.2% overhead).
- **Without tightening:** Same 2r/16b config with `--no-enable-asap-alap-tightening` and `--force-zero-gap --cpsat-time-limit 300`: 36-cycle schedule that **passes validation**. Per-decile usage shows work in every decile (no long empty stretch). Theoretical lower bound: actual 36 cycles vs theoretical minimum 14 (61.1% overhead).

**Conclusion:** Disabling slot tightening removes the empty-cycle phenomenon for this config. The no-tightening run is reproducible with:

```bash
uv run generate_ssa_kernel.py --rounds 2 --batch-size 16 --no-enable-asap-alap-tightening --force-zero-gap --cpsat-time-limit 300 --output-json scheduler_output_2r_16b_no_tightening.json --validate
```

Then run `uv run scripts/slot_usage_by_decile.py scheduler_output_2r_16b_no_tightening.json --markdown` and `uv run scripts/theoretical_lower_bounds.py scheduler_output_2r_16b_no_tightening.json` to confirm compact schedule and spread deciles.

---

## Ablation: 2r/16b no-tightening vs ASAP-only (Z3-informed)

Two-arm ablation to check that the **ASAP-only** safe form (Z3 T7-asap-only-composition) does not worsen makespan. **Expected:** both arms get the **same** number of cycles (if they differ, it is a bug). Ideally the ASAP-only arm converges to optimal **faster** (tighter domains aid propagation).

**Run:**

```bash
uv run python scripts/run_ablation.py \
  --rounds 2 --batch-size 16 --time-limit 300 \
  --arm "no-tightening:" \
  --arm "asap-only-tightening:--enable-asap-alap-tightening" \
  --output ablation_2r_16b_tightening.json
uv run python scripts/analyze_ablation.py ablation_2r_16b_tightening.json
```

**Result (300s time limit):**

| Arm | Cycles | Status | Solve time |
|-----|--------|--------|------------|
| no-tightening (default) | **36** | OPTIMAL | 6.99s |
| asap-only-tightening | **368** | OPTIMAL | 0.09s |

**Finding: bug.** Cycles differ (36 vs 368). The ASAP-only implementation still leads to a 368-cycle schedule; the 36-cycle schedule is either excluded by the tightened domains (plus fixpoint) or the solver returns the greedy hint without searching. So the safe-composition guarantee from the small Z3 model does not yet hold in the full pipeline for 2r/16b. Slot tightening remains **off by default**; when enabled, ASAP-only is used but may still worsen makespan for this config. See `ablation_2r_16b_tightening.json` and logs `ablation_log_no-tightening.txt`, `ablation_log_asap-only-tightening.txt`.

---

## Investigation: Z3 vs implementation — why 2r/16b still gets 368 cycles

The Z3 proofs (one-step lemma, T7-asap-only-composition) show that **certain** exclusions preserve feasibility in **small, fixed models**. The implementation can still exclude the 36-cycle schedule because several aspects of the real pipeline are **not** modeled in Z3. To make the implementation faithful to the proofs, we must either model these in Z3 and extend the proofs, or change the implementation so it stays within what we proved.

### 1. Precedence fixpoint is not in Z3

**Pipeline:** After `tighten_asap_alap_by_slots` we always run `fix_precedence_bounds_fixpoint(ops, &final_asap, &final_alap, …)` ([model.rs](scheduler_rs/src/cpsat_scheduler/model.rs) around 341). The fixpoint **propagates** bounds: for every edge (pred, succ), if `asap[succ] < asap[pred] + offset` it sets `asap[succ] = asap[pred] + offset`; similarly for ALAP backward. So one high ASAP from slot tightening is propagated to **all successors**. In 2r/16b, slot tightening pushes some ops’ ASAP to 357–367; the fixpoint then pushes every successor’s ASAP to the same range, so the whole tail ends up with ASAP in the 350s and the 36-cycle schedule is excluded.

**Z3:** T7-asap-only-composition only adds exclusions (e.g. c1≥1, c2≥2). It does **not** model “then apply precedence propagation.” So we have not proved that “ASAP-only exclusions **plus fixpoint**” preserve a feasible schedule.

**Next step:** Encode in Z3 a small model (e.g. 3–4 ops, chain precedence) where we apply ASAP-only exclusions **and** then enforce precedence. Ask: does a feasible schedule still exist? If **Z3 UNSAT** (no schedule exists), that’s a counterexample: the fixpoint is the culprit and we need either to not run it after slot tightening, or to run a restricted fixpoint.

### 2. Greedy-derived horizon and ALAP before slot tightening

**Pipeline:** When greedy hints exist, we set `horizon = greedy_max + 20` ([model.rs](scheduler_rs/src/cpsat_scheduler/model.rs) 141–155). For 2r/16b, greedy_max = 368, so horizon ≈ 388. We then compute ALAP with this horizon and apply **greedy-based ALAP tightening**: each op’s ALAP is intersected with `greedy_cycle + 15` (192–223). So the **input** to `tighten_asap_alap_by_slots` already has tail ops with ALAP in the 380s, i.e. bounds biased toward the 368-cycle greedy schedule. Slot tightening then runs with these wide windows; its demand profile sees many ops in late cycles and can push ASAP forward to fit capacity, which again favors the stretched schedule.

**Z3:** T7-asap-only-composition uses fixed domains [0,4] and fixed exclusions. It does not model “horizon and ALAP come from a prior greedy schedule,” so it does not capture this bias.

**Next step:** Either (a) when we want “safe” tightening that must not worsen makespan, avoid using greedy-based horizon/ALAP as input to slot tightening (e.g. use precedence-only bounds and a horizon not tied to greedy), or (b) model in Z3 that exclusions are applied on top of greedy-derived bounds and check whether a compact feasible schedule can still exist.

### 3. Multiple engines and iteration (follow by induction)

**Pipeline:** We tighten **per engine** (VALU, ALU, Load, Store, Flow) and merge results; the same op can be tightened by more than one engine. We run up to **3 iterations** (MAX_ITER in [bounds.rs](scheduler_rs/src/cpsat_scheduler/bounds.rs)), so a push in iteration 1 can change the profile and trigger more pushes in iteration 2.

**Induction:** The one-step lemma says each exclusion only removes a cycle (O,t) where no feasible schedule can place O at t. So after one exclusion, any feasible schedule that existed still exists in the remaining domain. By induction, any sequence of such exclusions—whether from multiple engines (each pass uses the same lemma for that engine's cap and demand) or multiple iterations (same lemma applied to updated bounds)—only removes infeasible options. So **multi-engine and multi-iteration do not break feasibility**; they follow from the one-step proof. Explicitly proving induction in Z3 is unnecessary for this.

### 4. Energetic reasoning vs discrete one-step exclusions

**Pipeline:** We use **energetic reasoning**: for an op with window [asap, alap], we compute “earliest t such that [t, alap] has demand ≤ cap × width” and set new ASAP = t. That can **jump** many cycles at once (e.g. from 26 to 357). So we are not applying “one-step” exclusions one cycle at a time; we are applying a **bulk** exclusion (all cycles before t).

**Z3:** The one-step lemma proves that excluding **one** cycle (O, t) when demand-at-t justifies it is sound. T7-asap-only-composition adds a few lower bounds (c1≥1, c2≥2). We have not proved that “bulk” ASAP pushes (new ASAP = 357) preserve feasibility when combined with the fixpoint and greedy-derived bounds.

**Next step:** Either prove in Z3 that a sequence of “bulk” ASAP pushes (each push of the form “ASAP(O) >= k”) plus fixpoint still leaves a feasible schedule under some ordering or bound, or restrict the implementation to smaller, incremental pushes that match the one-step story.

### 5. Faithfulness checklist

| Aspect | Z3 model | Implementation | Faithful? |
|--------|----------|-----------------|-----------|
| One-step exclusion soundness | T7: exclude (O,t) when demand justifies | Energetic reasoning (can push many cycles) | Same idea; implementation can push more than minimal |
| ASAP-only composition | T7-asap-only: c1≥1, c2≥2 → SAT | We only apply ASAP pushes (no ALAP pulls) | Same direction; see below |
| Precedence after tightening | Not modeled | Fixpoint propagates ASAP/ALAP | **No** — fixpoint not in Z3 |
| Initial bounds | Fixed [0,4] | Horizon from greedy; ALAP from greedy+slack | **No** — biased toward greedy |
| Multiple engines / iteration | One capacity, one round in Z3 | Per-engine, up to 3 iterations | **OK** — follows by induction from one-step |

So the implementation is **not** faithful to the Z3 models in several ways. The most plausible cause of the 2r/16b bug is the **fixpoint** (1) together with **greedy-derived bounds** (2): slot tightening pushes some ASAPs high, the fixpoint spreads them to the whole tail, and the 36-cycle schedule is excluded. Next steps are to add a Z3 model that includes precedence propagation after ASAP-only exclusions and, if needed, to change the implementation (e.g. no fixpoint after slot tightening, or precedence-only initial bounds when aiming for “safe” tightening).


### Do the other gaps follow by induction?

**No.** Only multi-engine and iteration follow by induction from the one-step lemma (each step only removes infeasible (O,t)). The others are different:

- **Fixpoint:** Precedence propagation is a **different operation**—it tightens bounds along edges (e.g. asap[succ] = max(asap[succ], asap[pred]+1)), not "exclude (O,t) because demand at t exceeds cap." So the one-step lemma does not apply. We **did** show in Z3 (T7-asap-plus-fixpoint-unsat) that aggressive exclusions + precedence can yield **no feasible schedule** (Z3 UNSAT = counterexample).
- **Greedy-derived bounds:** The lemma says each exclusion we *make* is sound; it doesn't say the *input* bounds are safe. Wide windows (from greedy horizon/ALAP) change which exclusions we compute and can lead to aggressive pushes that, after fixpoint, exclude good schedules. So this doesn't follow by induction; we could try to **model in Z3** exclusions derived from wide initial domains and check feasibility.
- **Energetic reasoning:** The implementation pushes ASAP to the earliest t such that demand in [t, alap] ≤ cap×width, which is equivalent to excluding (O,s) for all s < t. Each such (O,s) is infeasible because the window [s, alap] has total demand > cap×width (so at least one cycle in [s, alap] would exceed cap). So it's a **bulk** application of a window-based one-step. We could **prove in Z3** that this bulk condition implies each (O,s) is infeasible (window lemma), then induction would apply.

### Proposed Z3 checks

1. **T7-asap-plus-fixpoint-unsat** (implemented): Feasibility check with aggressive ASAP exclusions (c1≥3, c2≥4) + precedence. Z3 returns **UNSAT** = no feasible schedule exists in the model. So we have a **counterexample**: this combination destroys CP-SAT feasibility (the fixpoint’s effect is to propagate those bounds so the last op has no valid cycle).
2. **Energetic bulk-exclusion soundness** (optional): Prove that "total demand in [s, alap] > cap×(alap−s+1)" implies no feasible schedule places O at s (window-based one-step). Then bulk ASAP push follows by induction.
3. **Wide-domain exclusions** (optional): Model initial domains [0, H] with H large, add exclusions that would be computed by our logic from a greedy-like profile, add precedence; check whether a compact feasible schedule still exists.

### Where we stand: gap between implementation and proofs

We have proved (Z3 + induction):

- **One-step:** Each exclusion we make is sound (only removes infeasible (O,t)).
- **Multi-engine / iteration:** Follow by induction from one-step.
- **Mild ASAP-only + precedence:** T7-asap-only-composition → **Z3 SAT** (e.g. (1,2,3) is a feasible schedule) → we **preserve** CP-SAT feasibility.
- **Aggressive ASAP + precedence:** T7-asap-plus-fixpoint-unsat → **Z3 UNSAT** (no feasible schedule) → **counterexample**: this combination destroys feasibility.
- **Greedy-cap:** Capping at canonical G keeps G feasible.

The **implementation** does: greedy-derived horizon/ALAP → slot tightening (ASAP-only, per engine, up to 3 iterations) → **fixpoint**. So it can produce aggressive exclusions (wide windows → big ASAP pushes) and then the fixpoint propagates them; we have **proved** that aggressive exclusions + precedence can remove all feasible schedules.

**Gap:** We proved *when* feasibility is preserved (mild exclusions) and *when* it is lost (aggressive exclusions + fixpoint). We did **not** prove that the implementation’s choices (greedy-derived bounds + fixpoint) keep exclusions in the mild regime. In practice they do not—e.g. 2r/16b lands in the aggressive regime (368 cycles). So the gap is: **the implementation uses inputs and a fixpoint that can produce the aggressive regime we proved removes feasibility.**

To close the gap we can: **(1)** Change the implementation so it stays in the proved-safe regime (e.g. do not run fixpoint after slot tightening when tightening is on; or use precedence-only initial bounds when “safe” tightening is desired). **(2)** Add a proved sufficient condition that restricts when we run fixpoint or use greedy-derived bounds so that exclusions stay mild (e.g. only run fixpoint when exclusions are below some bound; then prove that condition holds). Until then, slot tightening remains **off by default** and greedy-cap only guarantees that the canonical (greedy) schedule remains feasible, not that a compact schedule does.

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

So we have both a **theorem bug** (the original T7 statement was too strong) and a **behavioral bug** (bad makespans / empty stretches when tightening is on). The code is “correct” relative to the one-step lemma (each exclusion is sound) but **incorrect** relative to the intended guarantee: we want **all** previously feasible schedules to remain feasible, not merely at least one.

### Goal: tighten beyond greedy; CP-SAT is the answer

Everything before CP-SAT (greedy, ASAP/ALAP tightening, etc.) is **cheap preprocessing** to help CP-SAT find good solutions faster. CP-SAT is the source of truth. Greedy gives a warm-start hint; slot tightening is meant to give **tighter domains than precedence alone** — ideally **tighter than where greedy put the ops** — so CP-SAT's search space is smaller. The **entire point of ASAP/ALAP tightening is to tighten beyond greedy**, not to stay at or inside greedy. We only must not over-tighten so much that we exclude every feasible schedule (the composition bug).

### Greedy-cap: adopted as the proven safe fix

We **adopt greedy-cap** as the implementation fix: when building the CP-SAT model we pass the greedy schedule as canonical and clamp slot tightening so we never exclude the canonical cycle for any op (Z3 theorem **T7-greedy-cap-preserves-G**). This guarantees at least one feasible solution (the greedy schedule G) and fixes the 368-cycle regression (no empty domains excluding G). The **ideal** — tightening that goes beyond greedy without excluding any feasible schedule — remains open; see the section below for strengths, weaknesses, and possible future work.

### What we actually need (open)

We need tightening that: (1) **goes beyond greedy** — smaller domains than precedence-only so CP-SAT's search is easier; and (2) **does not violate feasibility** — **all** previously feasible schedules should still be feasible (we only exclude (O,t) that no feasible schedule uses). The **greedy-cap** fix ensures we never exclude the pipeline's G, but that G can be poor (e.g. 368 cycles); good schedules (e.g. 36-cycle) can still be excluded. The open problem is to allow stronger tightening while **provably** preserving the strong property (all previously feasible schedules still feasible). We focus on **(1) safe composition** below; **(2) symmetry** is noted as a possible future direction so we don't lose sight of it.

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
- **Correctness and propagation strength depend on greedy quality:** If G is poor, "cap at G" still leaves a large search space; good compact schedules can remain excluded.

### Possible future work

**Safe composition candidate (Z3).** We added two checks in `verification/theorems.py`: `T7-asap-only-composition` (3 ops, 5 cycles, capacity 1, chain precedence; only ASAP exclusions c1≥1, c2≥2) and `T7-alap-only-composition` (same model; only ALAP exclusions c3≤3, c2≤2). Both yield **SAT**: at least one feasible schedule remains. So single-direction (ASAP-only or ALAP-only) is a candidate safe subset to implement; see `uv run python -m verification.verify -t T7-asap-only-composition -t T7-alap-only-composition` and [verification/README.md](../verification/README.md).

**Priority: (1) Safe composition (conservative focus).** Prove in Z3 a **composition rule** that allows multiple one-step exclusions (and the precedence fixpoint) while guaranteeing **all** previously feasible schedules remain feasible — **without** requiring a known schedule G in advance. For a gap analysis (why Z3 and the implementation can differ, e.g. fixpoint not in Z3, greedy-derived bounds), see the section **Investigation: Z3 vs implementation — why 2r/16b still gets 368 cycles** above. Then port that rule to the implementation. Examples: restrict to a single direction per pass (e.g. only ASAP pushes, or only ALAP pulls); or an order-based rule (e.g. process ops in a fixed order and prove the remaining domain still has a feasible schedule); or "exclude (O,t) only when no feasible schedule uses (O,t)" (expensive to check in practice unless encoded as part of the rule). This is the more conservative path: we only exclude when the combined effect is proven safe.

**Note: (2) Symmetry may be possible in future efforts.** An alternative is to prove that the exclusions we perform only remove schedules that are **symmetrically equivalent** (e.g. under batch symmetry) to some schedule we keep. Then we would not be excluding "the" compact schedule, only equivalent copies, and one representative would remain feasible. We keep this in mind so we don't lose sight of it; it may allow more aggressive tightening if the symmetry structure can be formalized.

- **Improve greedy:** Better G implies tighter effective domains under greedy-cap and better propagation.
- **Post-tightening check:** After each engine pass, check that G is still in the domains; if not, revert the exclusions that removed it so we can apply more exclusions when they don't touch G.
- **Optional makespan cap:** Add `makespan <= greedy_makespan + slack` in the CP-SAT model as a temporary safeguard if needed.
- **Formalize batch symmetry:** Document that G is the canonical representative under batch symmetry and that preserving G preserves one schedule per equivalence class; this supports both the current greedy-cap story and future (2) symmetry-based exclusion proofs.

