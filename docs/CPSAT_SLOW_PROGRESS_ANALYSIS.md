# CP-SAT Slow Progress Analysis

This document explains why the full-kernel CP-SAT run advances "one cycle at a time," what the solver stats mean, and how we might improve it.

## Why "One Cycle at a Time"?

**This is expected, not a bug.** The objective is **minimize makespan** (total cycles). Makespan is an integer: the number of cycles. So:

- When the solver has a feasible solution with makespan **460**, the only strictly better values are **459, 458, …**
- Each improvement is therefore **one cycle** (460 → 456 → 455 → … → 429).

So the discrete steps are inherent to the problem formulation, not a modeling error.

---

## Why Is It Slow?

From a typical long run (e.g. ~1 hour):

1. **Best solution** improves from 460 down to **429** over 27 solutions.
2. **Best bound** (LP / proof side) stays at **386**.
3. So the **gap** is **429 − 386 = 43 cycles**: the solver knows a solution of 429 but can only prove that the optimum is at least 386.

What slows things down:

### 1. Weak LP Relaxation for Cumulatives

- We use **cumulative** constraints for each engine (ALU 12, VALU 6, Load 2, Store 2, Flow 1).
- CP-SAT’s **LP relaxation of cumulative** is weak: the LP does not fully enforce “at most C ops per cycle” in a way that gives a tight lower bound.
- So `best_bound` (386) stays far below the best solution (429). Proving “no solution with makespan ≤ 428” then requires a lot of search.

We already add **extra lower bounds** in the model (capacity, profile, suffix, sparse) and use them to set the **domain** of the makespan variable. That helps initial bound and propagation, but the **dynamic LP** used during search still has a weak view of the cumulatives, so the bound improves slowly.

### 2. Progress Comes From LNS, Not From the Core MIP

From the “Solutions (27)” and “Solutions (Num, Rank)” parts of the log:

- **no_lp [hint]**: first solution is the greedy hint (460).
- **scheduling_time_window_lns**, **graph_cst_lns**, **rnd_var_lns**, **scheduling_intervals_lns**, etc.: almost all **improvements** come from **LNS** (Large Neighborhood Search), not from the main “full problem” subsolvers (default_lp, no_lp, max_lp_sym, etc.).

So:

- The **core branch-and-bound / LP** is not finding better solutions; it is slowly improving the bound.
- **Heuristics (LNS)** are doing the work of improving the incumbent.
- As the target makespan drops (e.g. 428, 427, …), proving infeasibility gets harder, so **time between improvements grows** (e.g. from a few seconds to hundreds or thousands of seconds).

### 3. Time per Improvement Grows

- Early: 460 → 456 in a few seconds.
- Later: e.g. 438 → 437 in ~25 minutes, 437 → 436 in ~2 minutes, etc.

That pattern is normal for min-makespan: each time we ask “is there a schedule with makespan ≤ M?”, the search and proof effort tends to increase as M gets closer to the true optimum.

---

## What the Printed Stats Mean

### Progress Line

```text
#9  536.59s best:449  next:[386,448]  graph_cst_lns (...)
```

- **best:449** – Best feasible makespan so far.
- **next:[386,448]** – Solver is trying to find a solution with makespan **≤ 448** (or prove none exists). **386** is the current **best lower bound** (proof that optimum ≥ 386).
- **graph_cst_lns** – This improving solution was found by the “graph constraint LNS” heuristic.

### Presolved Model

- **#Variables: 8,795** – After presolve.
- **#kCumulative: 5** – Five cumulative constraints (one per engine).
- **#kInterval: 8,858** – Interval variables (one per op; optional intervals for shift/ConstChoice/Flow).
- **#kNoOverlap: 1 (#intervals: 132, #optional: 132)** – One NoOverlap for Flow (capacity 1); 132 optional intervals (e.g. add_imm vs load, or select vs FMA).

### Symmetry

- **276 orbits on 552 variables** – Presolve found symmetry; 552 variables lie in 276 symmetric orbits.
- **Orbitope 4×2** – A small orbitope structure was detected. CP-SAT will add symmetry-breaking constraints.

### Task Timing / LNS Stats

- **scheduling_time_window_lns**, **scheduling_intervals_lns**, **graph_cst_lns**, etc. – Time and number of calls for each LNS/heuristic. Most **improving** solutions come from these.
- **default_lp**, **no_lp**, **max_lp_sym** – Full problem subsolvers; they spend time improving the bound and searching, but in this run they rarely report new solutions.

### Lp Debug

- **Bad / BadScaling** – Indicates numerical issues in the LP (e.g. fixed, max_lp_sym with large “Bad” counts). Can sometimes be alleviated by scaling or different LP options; worth monitoring but not necessarily the main cause of slowness.

---

## Possible Improvements

### 1. Strengthen the LP (Redundant Linear Constraints) — **Implemented**

The cumulative constraint’s LP is weak. We set **makespan ≥ computed_lower_bound** via the variable domain and **also add explicit linear constraints** that the LP can use directly:

- For each engine: **makespan ≥ ceil(num_ops_engine / capacity)**.

These redundant constraints are added in both `schedule_cpsat` (non-SSA) and `schedule_ssa_cpsat` (SSA) in `scheduler_rs/src/cpsat_scheduler.rs`. Same information as the capacity bound used for the makespan domain, but as linear inequalities they can be used more effectively in the LP relaxation and cut generation. See also **Lower bound strengthening** in `docs/CPSAT_SCHEDULER_IMPLEMENTATION.md`.

### 2. Symmetry Breaking

We have 276 orbits. If we can identify **equivalent ops** (same engine, same ASAP/ALAP, no precedence between them), we can add **lexicographic ordering** constraints (e.g. “op i runs before or at same time as op j” for equivalent pairs). That can reduce the search space without changing the optimum.

### 3. Solver Parameters

- **num_workers** – Already 8 for SSA; keep as is for parallelism.
- **LNS / heuristics** – The solver is already relying on LNS. We could try:
  - Slightly longer **LNS time limits** or more **LNS subsolvers** (if the API allows).
  - **Feasibility pump** – Already in the mix; no change needed unless we see a different bottleneck.
- **linearization_level** – Already 2. Going higher may tighten the LP a bit but increase model size and presolve time; worth a try on a few runs.
- **relative_gap_limit** – Already 0.02. We could stop earlier (e.g. 0.05) if we only need a “good” solution and not a proof.

### 4. Sectioning / Decomposition

For the **full kernel**, we currently build one big CP-SAT model. The docs describe **per-section** CP-SAT (e.g. 4-round phase) where we solve segment-by-segment. For the **single full-kernel** run that you observed:

- We are not sectioning; it’s one large model (8,795 variables, 8,726 cumulative intervals).
- If we could **fix** a subset of ops (e.g. from a greedy or previous run) and only re-optimize a **window** of cycles or a **subset of ops**, we could reduce model size and possibly get faster improvements. That would require a different orchestration (e.g. fix-and-refine loops) and care not to cut off the global optimum.

### 5. Accept “Good Enough” Sooner

If the goal is a **good schedule** rather than **proven optimal**:

- Use **relative_gap_limit** (e.g. 2–5%) and **max_time_in_seconds** so the run stops once we have a solution within that gap.
- The current run already improves from 460 to 429; stopping at 2% gap (or at a time limit) would still yield a strong schedule without the long tail of 1-cycle improvements.

---

## Summary

| Observation | Explanation |
|------------|-------------|
| Improves by 1 cycle at a time | Normal: objective is integer makespan. |
| Slow progress | Weak LP for cumulatives → bound (386) far below best solution (429). |
| Who finds solutions? | Mostly LNS (scheduling_*, graph_*, rnd_*); core MIP improves the bound. |
| Time between improvements grows | Proving “no solution with makespan ≤ M” gets harder as M decreases. |

So nothing is “wrong” with the formulation in the sense of incorrectness; the slowness comes from the **difficulty of the problem** (many ops, many cumulatives, integer makespan) and the **weak LP for cumulatives**. We have implemented **redundant linear capacity constraints** to strengthen the LP. **Symmetry breaking (implemented):** We use `SymmetryHint` on `SsaOp`: emission attaches `ordering_group` and `lexicographic_stream` keys; the optimizer adds ordering constraints (cycle[i] <= cycle[j] within groups, first_op ordering across stream items). This prunes redundant search when CP-SAT reports many orbits (e.g. 276). See `scheduler_rs/src/types.rs` (`SymmetryHint`) and `docs/INTEGRATED_SCHEDULING_ALLOCATION.md` §5.5.

Remaining levers: **stopping earlier** (gap or time) when a good-enough solution is enough.
