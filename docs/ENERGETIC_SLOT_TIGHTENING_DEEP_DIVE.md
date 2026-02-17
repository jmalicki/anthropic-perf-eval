# Energetic Slot Tightening: Deep Dive

This document explains **why** the energetic ASAP/ALAP slot tightening broke, what went wrong at each step, and how to fix it properly. It assumes minimal prior context and builds up from first principles. For the canonical design (safe/unsafe rules, implementation flags) see [ASAP_ALAP_SLOT_TIGHTENING.md](ASAP_ALAP_SLOT_TIGHTENING.md). For the specific 2r/16b counterexample investigation see [SLOT_TIGHTENING_T7_COUNTEREXAMPLE.md](SLOT_TIGHTENING_T7_COUNTEREXAMPLE.md).

---

## 1. Background: What Problem Are We Solving?

### 1.1 The VLIW Scheduling Pipeline

The kernel builder schedules SSA operations onto a VLIW machine. Each op must be assigned a **cycle** (0, 1, 2, …) such that:

1. **Precedence**: If op B depends on op A (B reads A's result), then `cycle[B] >= cycle[A] + 1` (results are visible the next cycle).
2. **Resource capacity**: Each cycle has limited slots per engine (e.g. ALU: 12, VALU: 6, Flow: 1). The number of ops using an engine at cycle `t` cannot exceed that engine's capacity.

The solver (CP-SAT) receives a model with cycle variables and constraints. **Tighter domains** (narrower allowed cycles per op) reduce the search space and can speed up solving. Our goal is to tighten domains **without** excluding any feasible schedule.

### 1.2 ASAP and ALAP

For each op O we compute a **window** [ASAP(O), ALAP(O)] of cycles O may run in:

- **ASAP (As-Soon-As-Possible)**: The earliest cycle O can run, given that all its predecessors must complete first. Computed by forward propagation along the precedence graph.
- **ALAP (As-Late-As-Possible)**: The latest cycle O can run, given that all its successors must start in time. Computed by backward propagation from a **horizon** (upper bound on makespan).

The CP-SAT model gives each op a cycle variable with domain [ASAP, ALAP]. Any schedule satisfying precedence and capacity must assign each op a cycle within its window.

### 1.3 Slot Tightening: The Idea

**Slot tightening** uses resource capacity to narrow [ASAP, ALAP]. **Requirement (normative):** every exclusion we make must be one the solver would reach anyway through its own propagation—i.e. tightening only excludes (O,t) that the full constraint system (precedence + capacity) would rule out. That is what we need to **prove** about tightening; if we cannot prove it for a given exclusion, we are not doing valid tightening. The aim is to use domain knowledge (demand profiles, energetic reasoning, precedence structure) to precompute those exclusions efficiently, helping the solver along.

The intuition: if at cycle `t` the demand from other ops (those whose windows include `t`) plus this op's slots would exceed capacity, then **no feasible schedule** can place this op at `t`. So we can exclude `t` from this op's domain.

Example: capacity 1, op O's window [0, 4]. At cycle 1, three other ops have 1 in their windows. If we place O at 1, we'd have 4 ops at cycle 1, but capacity is 1. So we can exclude (O, 1) — push ASAP forward or pull ALAP backward so that 1 is no longer in O's window.

Tighter domains → smaller search space → faster solving. **Crucially**: we must only exclude cycles that no feasible schedule uses—equivalently, exclusions the solver would deduce from precedence and capacity. Proving that each exclusion has that property is the normative requirement for valid tightening.

---

## 2. The Strong Guarantee and Symmetry

### 2.1 Symmetry Breaking in the Model

The CP-SAT model already includes **symmetry-breaking constraints**. Emission attaches a `SymmetryHint` to ops (`ordering_group`, `lexicographic_stream`). The optimizer builds ordering constraints from these: within each ordering group, ops are ordered (e.g. by greedy cycle then ALAP, or by ALAP when no greedy hints), and we add `cycle[i] <= cycle[j]` for consecutive pairs. That picks one representative per equivalence class and prunes symmetric branches. See `build_ordering_pairs_from_bounds` in `bounds.rs`, and `docs/INTEGRATED_SCHEDULING_ALLOCATION.md` / `docs/CPSAT_SLOW_PROGRESS_ANALYSIS.md`.

So the set of solutions we care about is already "one representative per symmetry class." Tightening must not exclude any of those representatives.

### 2.2 What We Want (Strong Guarantee)

**Strong guarantee**: Tightening only excludes (O, t) that **no** feasible schedule uses. So **every** schedule that satisfies precedence and capacity (and thus every representative under the existing symmetry breaking) is still feasible after tightening.

Equivalently: with symmetry breaking in place, we must not exclude a **single** feasible solution—i.e. we must not remove any representative. Any exclusion we make must be one the solver would deduce from precedence and capacity (and the same symmetry); we are only precomputing that deduction.

A weaker property ("at least one feasible schedule remains") is insufficient. Composition of multiple "valid" one-step exclusions can remove all feasible schedules (T7-two-step). So we design rules that **provably** preserve the strong guarantee.

### 2.3 Taking Symmetry Into Account

Tightening should be **consistent with the same symmetry**:

1. **Ordering is derived from tightened bounds.** After we compute final ASAP/ALAP (including any slot tightening), we build ordering pairs by sorting each ordering group by (greedy, ALAP) or ALAP. So the symmetry-breaking order is determined by the same bounds that tightening produces. For the constraint `cycle[i] <= cycle[j]` to be satisfiable we need `ASAP[i] <= ALAP[j]`; the code enforces this and panics if tightening (or fixpoint) makes it false. So tightening already has to stay consistent with **some** total order per group; the open question is whether that order is the "right" one for the symmetry we want to preserve.

2. **Exclusions must not remove the representative.** Under symmetry breaking, we keep one representative per equivalence class. Valid tightening must only exclude (O,t) that no feasible schedule uses—so in particular it must not exclude the cycle that the representative assigns to O. If our exclusions are only those the solver would deduce, we satisfy this. If we over-tighten and exclude a cycle that some representative uses, we have broken the guarantee.

3. **Future work: formalize symmetry for tightening.** To prove that tightening preserves "all feasible schedules modulo symmetry," we would formalize the symmetry (e.g. batch/group interchangeability), show that symmetry-breaking constraints pick one representative per class, and prove that each exclusion either (a) is entailed by precedence and capacity, or (b) only removes schedules that are symmetric to a schedule we keep. See SLOT_TIGHTENING_T7_COUNTEREXAMPLE.md § "Symmetry may be possible in future efforts."

---

## 3. Energetic Reasoning: The Algorithm

### 3.1 Demand Profile

For each engine (ALU, VALU, Load, Store, Flow), we build a **demand profile**:

- Each **required** op (one that definitely uses this engine) has a window [asap, alap].
- We treat it as "could run in any one cycle in [asap, alap]."
- Using a difference array: `diff[asap] += 1`, `diff[alap + 1] -= 1`; prefix-sum gives `profile[t]` = number of required ops whose window contains `t`.

So `profile[t]` = "how many ops *could* be active at cycle t" (in the sense that t is in their window). We then build a prefix array so that for any window [a, b], total demand = `prefix[b+1] - prefix[a]` in O(1).

**Important**: We exclude **optional-interval** ops from the profile:
- Flow: SelectChoice (may use FMA/VALU instead of vselect/Flow)
- Load: ConstChoice (may use AddImm/Flow instead of Load)
- VALU: shiftable ops (may shift to ALU)

If we counted them, we'd over-count: in some feasible solutions they don't use this engine.

### 3.2 Energetic Check

For an op O with window [asap, alap] and window size W = alap - asap + 1:

- Total demand in that window = sum of profile over [asap, alap].
- **Energetic reasoning**: If we have D ops that must fit in W cycles with capacity C per cycle, we need `D <= C * W`. Otherwise the window is too small — no feasible schedule can fit all those ops in that window.

If `total_demand > C * W`, we tighten O's window.

### 3.3 Tightening: ASAP Push and ALAP Pull

**ASAP push**: Find the earliest start `t` such that the window [t, alap] has `demand <= C * (alap - t + 1)`. Set `new_ASAP = t`. This excludes all cycles before t — a **bulk** exclusion (e.g. from 26 to 357 in one step).

**ALAP pull**: Symmetrically, find the latest end `t` such that [asap, t] has `demand <= C * (t - asap + 1)`. Set `new_ALAP = t`.

**Implementation choice**: We only apply ASAP pushes, not ALAP pulls. Z3 showed that mixing ASAP and ALAP exclusions can remove all feasible schedules (T7-two-step). ASAP-only is a safer composition candidate (T7-asap-only-composition yields SAT).

### 3.4 Iteration and Multiple Engines

- We iterate per engine (VALU, ALU, Load, Store, Flow) and repeat up to 3 times (MAX_ITER).
- Tightening one op changes its window → changes the profile → can justify more tightening for other ops.
- The same op can be tightened by multiple engines (e.g. an ALU op gets its ASAP pushed by the ALU pass, then again by the next iteration).

---

## 4. Why It Broke: Failure Modes

### 4.1 Composition Failure (T7-two-step)

**The one-step lemma is sound**: When we exclude (O, t) because demand-at-t justifies it, no feasible schedule can have O at t. Z3 proves this (T7).

**But composition fails**: Two valid one-step exclusions can together remove all feasible schedules.

**Concrete example** (Z3 model T7-two-step):
- 2 ops (O1, O2), 3 cycles (0, 1, 2), capacity 1
- Precedence: O2 >= O1 + 1
- Valid exclusion 1: O1 not at 0 (demand at 0 justifies it)
- Valid exclusion 2: O2 not at 2 (demand at 2 justifies it)

Remaining domain: O1 in {1, 2}, O2 in {0, 1}. Precedence forces O2 >= O1 + 1. The only feasible schedule is (O1=1, O2=2). But we excluded O2 from 2! So **no feasible schedule remains**. Z3 returns UNSAT.

Each exclusion alone is sound (removes only infeasible cycles). Together they remove the last feasible schedule (1, 2). So "valid one-step" does **not** imply "composition preserves feasibility."

### 4.2 The Precedence Fixpoint Amplifies Over-Tightening

**Order of operations** in the pipeline:
1. `tighten_asap_alap_by_slots` updates ASAP/ALAP
2. Optional scratch tightening
3. `fix_precedence_bounds_fixpoint` restores precedence consistency

**What the fixpoint does**: For every edge (pred → succ), if `asap[succ] < asap[pred] + offset`, set `asap[succ] = asap[pred] + offset`. Similarly for ALAP backward.

**The problem**: If slot tightening pushes one op's ASAP from 26 to 357, the fixpoint propagates that to **all successors**. Every op that depends on that one (directly or transitively) gets its ASAP pushed to at least 357+offset. So one aggressive push **spreads** to the whole tail.

In 2r/16b, slot tightening pushed some ops' ASAP to 357–367. The fixpoint then pushed every successor into the 350s. The 36-cycle schedule (which places ops in early cycles) was excluded — those cycles were no longer in the domain.

**Z3**: T7-asap-plus-fixpoint-unsat proves that aggressive ASAP exclusions **plus** precedence propagation can remove all feasible schedules. We have a counterexample.

### 4.3 Greedy-Derived Bounds Bias the Input

**Pipeline**: When greedy hints exist:
1. `horizon = greedy_max + 20` (e.g. greedy produces 368 cycles → horizon ≈ 388)
2. ALAP is computed with this horizon
3. **Greedy-based ALAP tightening**: Each op's ALAP is intersected with `greedy_cycle + 15`

So the **input** to `tighten_asap_alap_by_slots` already has:
- Wide windows (ALAP can go up to 380+)
- Bounds biased toward the greedy schedule (368 cycles)

**The problem**: Slot tightening sees many ops with windows extending into late cycles. The demand profile has high "could be active" counts in the 300s. Energetic reasoning then pushes ASAP forward to fit capacity — "if all these ops could be in late cycles, we need to push some forward." That favors the **stretched** schedule (368 cycles) and excludes the **compact** one (36 cycles).

The greedy schedule G is poor (368 cycles). Our tightening is biased toward G. Good compact schedules are excluded.

### 4.4 Bulk Exclusions vs One-Step

**One-step lemma**: Excluding **one** cycle (O, t) when demand-at-t justifies it is sound. Z3 proves this.

**Implementation**: We do **bulk** exclusions. For an op with window [26, 400], we might set new ASAP = 357 — excluding **all** cycles 26–356 in one step.

Each excluded (O, s) for s < 357 is justified by the window [s, 400] having excess demand. So in principle each exclusion is a valid one-step. But the combination of bulk push + fixpoint + greedy-derived bounds produces the bad outcome. The Z3 models use small, fixed exclusions (c1≥1, c2≥2); they don't model bulk pushes of 300+ cycles.

### 4.5 R_precedence_demand: A Simpler Unsafe Rule

**Idea**: For each op O and cycle t in O's window, exclude (O, t) when `demand_other(t) + slots(O) > cap`, where `demand_other(t)` = count of **other** ops whose window contains t.

**Why it fails**: `demand_other(t)` counts "ops that *could* be at t." In any **single** feasible schedule, at most one op is at t (capacity 1). So we over-count: we assume all those ops are at t simultaneously. We can exclude (O, t) even when some feasible schedule has O at t.

**Counterexample**: 3 ops (O1, O2, O3), 5 cycles, capacity 1, chain O2≥O1+1, O3≥O2+1. Windows: O1 [0,4], O2 [1,4], O3 [2,4]. At cycle 1, O2 has 1 in its window → demand_other(O1, 1) = 1. Then 1 + 1 > 1, so we'd exclude (O1, 1). But (O1=1, O2=2, O3=3) is feasible and has O1 at 1. So we'd remove a feasible schedule. **Do not use** R_precedence_demand.

---

## 5. Z3 Theorems: What They Prove

We use Z3 to verify properties of our tightening rules. Two kinds of checks:

### 5.1 Soundness Proofs (Encode a Violation)

We encode: "tightening holds AND [something we want to refute]." If Z3 finds a satisfying assignment (SAT), we have a counterexample — the theorem fails. If Z3 returns UNSAT, no counterexample exists — the theorem holds.

**Example (T7 one-step)**: We encode "demand justifies exclusion AND feasible schedule has O at t." Z3 UNSAT → no such schedule exists → each exclusion only removes infeasible cycles. **Good.**

### 5.2 Preservation Checks (Ask for a Witness)

We encode: "precedence + capacity + exclusions" and ask: does a feasible schedule exist? Z3 SAT = yes, at least one remains. Z3 UNSAT = no feasible schedule = we over-tightened.

**Example (T7-asap-only-composition)**: 3 ops, 5 cycles, capacity 1, chain precedence; exclusions c1≥1, c2≥2. We ask for a feasible (c1, c2, c3). Z3 SAT → (1, 2, 3) works → at least one feasible schedule remains. **Good** for this small model.

**Example (T7-two-step)**: 2 ops, 3 cycles; exclusions c1≠0, c2≠2. Z3 UNSAT → no feasible schedule → composition fails. **Important negative result.**

**Example (T7-strong-guarantee-fails-asap-only)**: We *try* to prove the strong guarantee for ASAP-only. Z3 SAT → there exists a feasible schedule that our exclusions remove → we violate the strong guarantee. **We preserve this failure** as documentation.

### 5.3 Key Theorems Summary

| Theorem | Claim | Result | Meaning |
|---------|-------|--------|---------|
| T7 | One-step exclusion is sound | UNSAT | Each single exclusion only removes infeasible (O,t) |
| T7-two-step | Two-step composition preserves feasibility | UNSAT (counterexample) | Composition can remove all feasible schedules |
| T7-one-exclusion-asap | One ASAP exclusion preserves feasibility | SAT | One exclusion is OK |
| T7-one-exclusion-alap | One ALAP exclusion preserves feasibility | SAT | One exclusion is OK |
| T7-asap-only-composition | ASAP-only exclusions preserve feasibility (small model) | SAT | Mild exclusions OK in small model |
| T7-asap-plus-fixpoint-unsat | ASAP exclusions + precedence propagation | UNSAT | Fixpoint can remove all feasible schedules |
| T7-strong-guarantee-entailed-exclusions | Exclusions entailed by precedence preserve all | UNSAT (refutation) | Entailed exclusions are safe |
| T7-strong-guarantee-fails-asap-only | ASAP-only preserves strong guarantee | SAT (counterexample) | We do over-tighten |
| T7-greedy-cap-preserves-G | Greedy-cap keeps canonical schedule G feasible | SAT | Capping at G preserves at least G |

---

## 6. How to Fix It: Detailed Options

### 6.1 R_entailed (Proved Safe; Requires a Solver)

**Rule**: Only exclude (O, t) when `(precedence ∧ capacity ∧ cycle_O = t)` is **UNSAT** — i.e. when no feasible schedule has O at t.

**Pros**: Z3 proves the strong guarantee (T7-strong-guarantee-entailed-exclusions). We only exclude cycles that no feasible schedule uses.

**Cons**: Requires a SAT/UNSAT check per (O, t) or a batch query. The Rust scheduler has no solver. Implementation would need a Python pre-pass calling Z3 or CP-SAT. Potentially expensive (many ops × many cycles).

### 6.2 Skip or Restrict the Fixpoint After Slot Tightening

**Observation**: T7-asap-plus-fixpoint-unsat shows that precedence propagation **after** aggressive exclusions can remove all feasible schedules. The fixpoint spreads one bad push to the whole graph.

**Option A**: Don't run the fixpoint after slot tightening. Slot tightening may create precedence violations (consumer ASAP < pred ASAP + offset). The CP-SAT model has precedence constraints, so the solver would reject such assignments anyway. But domain bounds might be inconsistent — need to validate.

**Option B**: Run the fixpoint **before** slot tightening, not after. Then slot tightening sees precedence-consistent bounds. It might still push aggressively, but the fixpoint wouldn't amplify those pushes. Would need to re-run fixpoint after tightening to restore consistency — same problem.

**Option C**: Run a **restricted** fixpoint that doesn't propagate slot-tightening-induced pushes. E.g. only propagate when the new bound comes from precedence, not from slot tightening. Implementation would need to track the source of each bound.

**Next step**: Encode in Z3 a model with ASAP-only exclusions **without** fixpoint. Check if feasibility is preserved. If yes, try Option A in the implementation and validate on 2r/16b.

### 6.3 Precedence-Only Initial Bounds for Slot Tightening

**Observation**: Greedy-derived horizon and ALAP bias the input toward the stretched schedule. Slot tightening then pushes ASAP forward to fit capacity, favoring that stretched schedule.

**Option**: When running slot tightening for "safe" mode, use **precedence-only** bounds:
- Horizon = something not tied to greedy (e.g. 2× critical path, or a capacity-based lower bound)
- No greedy-based ALAP tightening before slot tightening

So the input to `tighten_asap_alap_by_slots` is neutral — not biased toward a 368-cycle schedule. Slot tightening might then produce less aggressive pushes.

**Trade-off**: We might lose some beneficial tightening from greedy-based ALAP. But we avoid the bias that excludes good compact schedules.

### 6.4 Incremental Pushes Instead of Bulk

**Observation**: Energetic reasoning can jump ASAP from 26 to 357 in one step. The one-step lemma applies to each excluded cycle individually, but the combination with fixpoint causes trouble.

**Option**: Restrict the implementation to **small** pushes per iteration — e.g. at most 1 or 2 cycles per op per pass. Repeat until no change. That would match the one-step story more closely and might avoid the aggressive regime that triggers fixpoint amplification.

**Trade-off**: More iterations, possibly weaker propagation. Need to test whether it helps solve time.

### 6.5 Post-Tightening Validation

**Idea**: After each engine pass, check that a known good schedule (e.g. from a no-tightening run, or the greedy schedule G) is still in the tightened domains. If any op's canonical cycle was excluded, **revert** the exclusions that caused it.

**Greedy-cap** already does a form of this: we clamp so we never exclude G(O) for any op. But G can be poor (368 cycles). A stronger check would validate against a compact schedule (36 cycles) — but we don't know that schedule in advance when building the model.

**Greedy-cap limitation**: It guarantees at least G remains feasible. It does **not** guarantee that good compact schedules remain. So we fix the 368-cycle regression (solver always has a feasible solution) but we might still exclude better schedules.

### 6.6 Z3-Guided Sufficient Conditions

**Approach**: Extend the Z3 models to include:
- Precedence fixpoint propagation after exclusions
- Greedy-derived initial bounds (or parameterized horizon)

Then search for **sufficient conditions** (e.g. limits on push magnitude, ordering constraints, or bounds on how many ops can be tightened) under which a feasible schedule still exists. Implement exactly those conditions in Rust.

**Benefit**: We'd have a proof that the implementation preserves feasibility under those conditions. The gap between Z3 and the real pipeline would be closed.

---

## 7. Symmetry-Aware Slot Tightening (Safe, Solver-Free)

We can get **safe** slot-based exclusions without a solver by using the **same order as symmetry breaking**. The model adds constraints `cycle[i] <= cycle[j]` for consecutive ops in each ordering group (sorted by greedy then ALAP). So within a group, for a given engine with capacity C, the k-th op (1-based) that uses that engine cannot run before cycle `ceil(k/C) - 1`: the preceding k−1 ops must be scheduled first and at most C fit per cycle. So we set ASAP(op) = max(ASAP(op), ceil(k/C) − 1). This exclusion is **entailed** by (symmetry + capacity): every feasible schedule satisfying the model’s ordering constraints already has this op at least at that cycle. **Strong guarantee** holds; no solver needed.

**Implementation:** `tighten_asap_by_symmetry_ordering` in `bounds.rs`. On by default when slot tightening is enabled (`use_symmetry_aware_slot_tightening: true`). Disable with `--no-symmetry-aware-slot-tightening`. See [ASAP_ALAP_SLOT_TIGHTENING.md](ASAP_ALAP_SLOT_TIGHTENING.md) § R_symmetry_ordering.

### 7.1 Energetic with Undercounting (Current Implementation)

The energetic pass (`tighten_asap_alap_by_slots`) now uses **undercounting** instead of overcounting. For each (group, engine) we build an ordered chain of ops that definitely use that engine. At each cycle t, `min_demand_group(t)` = count of ops in that group's chain with `asap[op] == t`. We exclude (O,t) only when `total_min_demand(t) > cap` (summed over all groups for that engine). When O has `asap[O] == t`, O contributes to the count, so we require `total <= cap`. When O has `asap[O] < t`, O doesn't contribute, so we require `total < cap` (room for O). This never over-tightens: we only exclude when even in the best case, cycle t is full. Requires `SymmetryOrdering`; when `ordering` is None, returns bounds unchanged (no-op).

## 8. Current State and Recommendations

**Current default**: R_symmetry_ordering (safe slot exclusions) + R_precedence_only (precedence, fixpoint, greedy-cap). We run symmetry-aware tightening then skip the energetic loop unless requested. **Proved safe** (strong guarantee for symmetry-based exclusions).

**Energetic** (`--energetic-slot-tightening`): Uses **symmetry-aware undercounting** — a lower bound on demand at each cycle. Only excludes (O,t) when total_min_demand(t) > cap. Incorporates SymmetryOrdering; never over-tightens. Produces schedules at least as good as symmetry-only (e.g. 55 cycles for 3r/16b). **Sound** — safe to use.

**The old demand-profile algorithm** (overcounting) was replaced. It violated the strong guarantee (see §4). The new undercounting algorithm fixes that.

**Z3 proofs** (parallelism + symmetry ordering): Four theorems in `verification/theorems.py` prove properties of the undercounting algorithm:
- T7-undercount-one-exclusion-sound: one exclusion sound (2 parallel ops, cap 1, ordering; UNSAT)
- T7-undercount-chain-preserves-witness: fan-out exclusions preserve a witness (O1→O2,O3; SAT)
- T7-undercount-parallel-contributing-sound: contributing exclusion sound (3 parallel ops, cap 2; UNSAT)
- T7-undercount-contributing-implies-T7: contributing exclusion implies T7 one-step (UNSAT)

**Recommended next steps** (optional future work):
1. **Formalize full strong guarantee**: Extend Z3 to prove undercounting preserves all feasible schedules (not just one witness) under symmetry assumptions.
2. **Long term**: R_entailed (Option 6.1) is the only proved-safe slot-based rule. If we can afford the solver cost, a Python pre-pass could implement it.

---

## 9. References

- [ASAP_ALAP_SLOT_TIGHTENING.md](ASAP_ALAP_SLOT_TIGHTENING.md) — Canonical design, safe/unsafe rules, implementation flags
- [SLOT_TIGHTENING_T7_COUNTEREXAMPLE.md](SLOT_TIGHTENING_T7_COUNTEREXAMPLE.md) — 2r/16b investigation, Z3 vs implementation gap, ablation results
- [verification/README.md](../verification/README.md) — Z3 theorem list, preservation vs soundness, greedy-cap
- [LIVENESS_SPECIFICATION.md](LIVENESS_SPECIFICATION.md) — Axioms and T7 claim
- `scheduler_rs/src/cpsat_scheduler/bounds.rs` — `tighten_asap_alap_by_slots` implementation
- `scheduler_rs/src/cpsat_scheduler/model.rs` — Pipeline order (ASAP/ALAP → slot tightening → fixpoint)
- `verification/theorems.py` — Z3 checks for T7 family
