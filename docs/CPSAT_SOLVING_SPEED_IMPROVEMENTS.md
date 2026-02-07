# CP-SAT Solving Speed: Suggested Improvements

This document suggests concrete improvements to speed up the SSA CP-SAT scheduler, based on the current model in `scheduler_rs/src/cpsat_scheduler.rs` and the analysis in `CPSAT_SLOW_PROGRESS_ANALYSIS.md`.

---

## 1. Tighter horizon (low effort)

**Current:** `horizon = max(critical_path_len * 2, greedy_max, n)`.

**Idea:** When we have a feasible greedy hint, the optimal makespan is at most `greedy_max`. Use a horizon that is only slightly above the capacity lower bound and greedy:

```text
horizon = max(capacity_lower_bound, greedy_max) + slack
```

e.g. `slack = 50` or `min(100, capacity_lower_bound / 10)`. Tighter horizons shrink variable domains and can reduce search and propagation cost.

**Caveat:** If the solver can improve beyond the greedy solution, we must not cut off the optimum. So either:
- Set `slack` large enough that improvement is still possible, or
- Dynamically extend horizon when the solver finds a solution with makespan near the upper bound (if the API supports it).

**Where:** `schedule_ssa_cpsat` around the `horizon` computation (after `compute_capacity_lower_bound_global` is available).

---

## 2. Solver parameters (low effort)

**Already set:** `linearization_level=2`, `cp_model_presolve=true`, `cp_model_probing_level=2`, `randomize_search=true`, `num_workers=8`, `absolute_gap_limit=scale/3`.

**Worth trying:**

| Parameter | Current | Suggestion | Rationale |
|-----------|---------|------------|-----------|
| `linearization_level` | 2 | Try 1 for very large models | Level 2 adds more linearizations; for huge models presolve/solve can get heavier. A/B test: level 1 may sometimes be faster with a slightly weaker bound. |
| `relative_gap_limit` | (none) | e.g. `0.02` or `0.05` | Stop once `(best - bound) / best <= 0.02`. Gets a "good enough" schedule faster when proof of optimality is not required. |
| `max_time_in_seconds` | user | Keep | Already used; shortening for dev (e.g. 2-round) is standard. |

**Optional (experimental):** Expose `num_workers` (and maybe `linearization_level`) in `SsaCpsatConfig` so heavy runs can use more workers without code change.

---

## 3. Strengthen greedy hint (medium effort)

**Current:** We pass greedy cycle hints and choice hints (shift, ConstChoice, SelectChoice, vbroadcast elimination). CP-SAT can still reject or adjust hints if presolve tightens domains.

**Ideas:**

- **Hint only a subset of vars:** Ensure every *decision* variable (cycle, shift, const_choice, select_choice, eliminate_vb) has a hint when greedy is feasible. The code already hints many of these; audit that no decision variable is left unhinted (e.g. auxiliary vars like `chosen_cycle` are already hinted).
- **Two-phase when greedy is good:** If greedy is within scratch budget and we trust it:
  - Option A: Run with `stop_after_first_solution=true` and a short time limit to confirm feasibility, then run again with full time limit and the same (or improved) hint.
  - Option B: Fix a subset of decisions from greedy (e.g. ConstChoice and shift) and only re-solve for cycle variables; smaller search space.

**Fill-in-the-blanks:** CP-SAT can complete partial hints. If greedy sometimes leaves a few ops unscheduled (e.g. dead code), we could still pass hints for the rest and let the solver fill in the rest. Currently we skip negative cycle hints; we could still add hints for all non-negative cycles so the solver has a denser hint set.

---

## 4. Redundant linear constraints (already done; optional tightening)

**Current:** We add:
- Makespan domain `[capacity_lower_bound, horizon]`
- Redundant linear capacity constraints: Load, Store, and (when applicable) VALU, ALU, Flow, Load from choice variables (direct capacity constraints).

**Optional:** Add more redundant linear constraints that the LP can use:
- **Profile/suffix bounds as linear:** We compute profile and suffix bounds in `compute_capacity_lower_bound_global`. If we had a small set of "critical" deadlines or start times, we could add linear constraints of the form "makespan >= t + ceil(count_t / cap)" for a few t. That would require exposing those bounds from the capacity computation and adding them as linear inequalities (same idea as the existing load/store bounds).

---

## 5. Symmetry breaking (already done; extend if structure exists)

**Current:** Ordering groups and lexicographic streams from `SymmetryHint` are used to add `cycle[i] <= cycle[j]` and similar ordering constraints.

**Optional:** If there are more symmetric groups (e.g. ops that are interchangeable by engine, ASAP/ALAP, and precedence), add more lex constraints. This is only useful if the emission can tag such structure; otherwise the current symmetry breaking is already doing the job.

---

## 6. Reduce optional intervals (medium effort)

**Current:** Shift, ConstChoice, and SelectChoice are modeled with optional intervals (one or more per op). That increases the number of interval variables and the complexity of the cumulative/NoOverlap encoding.

**Ideas:**
- **Fix some choices from greedy:** For a first pass, fix ConstChoice (or shift) to the greedy choices and solve a smaller model (fewer optional intervals). Then optionally re-solve with choices free, using the first solution as a hint.
- **Merge optional intervals where equivalent:** No change to correctness; only consider if we can simplify the model (e.g. same start, same size, same literal) so that the solver sees fewer intervals. This is micro-optimization.

---

## 7. Scratch constraint (keep as is; optional tuning)

**Current:** Scratch is modeled with a cumulative over variable-size intervals (live ranges). This is expressive but heavy.

**Optional:** If we ever split into sections (see below), scratch in each section could use a smaller set of intervals (only VRegs that are live in that section), reducing constraint size per solve.

---

## 8. Sectioning / decomposition (higher effort)

**Current:** Full kernel is one big CP-SAT model (thousands of variables, 5 cumulatives, 1 NoOverlap, scratch cumulative).

**Idea:** Already used for 4-round phase: solve segment-by-segment with fixed boundaries or overlap. For the full kernel we could:
- **Time windows:** Split the horizon into segments; solve each segment (or overlapping windows) with fixed incoming/outgoing live sets; iterate to improve global makespan. Requires careful handling of cross-segment precedence and scratch.
- **Fix-and-refine:** Fix the schedule for a subset of ops (e.g. first 50% of emission order) from greedy or a previous run; re-solve only for the remaining ops with a shorter horizon. Repeat with different splits.

This is a larger design; see `CP_SAT_CUSTOM_SUBSOLVER.md` and the 4-round phase for existing patterns.

---

## 9. Greedy compaction as post-process or subsolver (planned)

**Current:** Documented in `CP_SAT_CUSTOM_SUBSOLVER.md`: greedily move ops earlier when dependencies and capacity allow.

**Use for speed:** Run CP-SAT with a **looser time limit** (or relative gap), then run greedy compaction on the best solution. You can get a better makespan without waiting for the solver to prove optimality. Optionally feed the compacted solution back as a new hint and re-run CP-SAT (hint loop).

---

## 10. When the solver reports "very slowly increasing bounds with quick_restart"

**Meaning:** The improving bound (lower bound for minimization) is coming from the `quick_restart` subsolver, and it is increasing very slowly. Often `best:inf` (no feasible solution yet).

**Implications:**
- Feasibility is hard for the model/parameters/horizon; the solver is mostly proving bounds, not finding solutions.
- **Actions:** Ensure a feasible hint is provided (greedy); tighten horizon if it's loose; try 1 worker to avoid portfolio overhead when debugging; check that the model is feasible (e.g. run with a very long time limit or with some choices fixed to greedy).

---

## Summary table

| Area | Effort | Impact | Notes |
|------|--------|--------|-------|
| Tighter horizon | Low | Medium | Reduces domains; need slack so optimum is not cut off. |
| Solver params | Low | Low–Medium | relative_gap_limit, optional linearization_level 1. |
| Stronger hints | Medium | Medium | Complete hints; optional two-phase or fix-and-refine. |
| Redundant linear | Low | Low | Optional profile/suffix linear constraints. |
| Symmetry | Low | Low | Add more if emission exposes structure. |
| Fewer optional intervals | Medium | Medium | Fix some choices from greedy; smaller model. |
| Sectioning | High | High | For very large instances; design already in use for 4-round. |
| Compaction + hint loop | Medium | Medium | Per CP_SAT_CUSTOM_SUBSOLVER.md. |

Recommendation: try **tighter horizon with slack** and **relative_gap_limit** first; then **hint audit** and **fixing a subset of choices from greedy** for a faster first solution.
