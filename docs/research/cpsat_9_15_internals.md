# How CP-SAT 9.15 handles our scheduling model, and what would make it faster

Date: 2026-11-24. Reading only; no solver, build or test was run.

## How to read this file

Every claim carries one of these tags:

- **[S]** verified by reading the OR-Tools source at tag `v9.15` (tarball from
  `codeload.github.com/google/or-tools/tar.gz/refs/tags/v9.15`, directory `ortools/sat`). The file and
  function are named. Line numbers are from that tag.
- **[S10]** verified in the source or release notes of `v10.0-rc.1` (published 2026-11-23).
- **[R]** verified in this repository (`scheduler_rs/src/cpsat_scheduler/`, `docs/`).
- **[I]** my inference from [S] facts. Not measured. Each one has a test in section 9.
- **[M]** recollection of the literature or of maintainer statements, not re-verified in this session.
  Treat as a lead, not as evidence. See "Gaps" at the end.

Source paths below are relative to `ortools/sat/` unless they start with another directory.

## 0. Short answers

1. **Bound.** The LP relaxation of each engine cumulative is one aggregate energy row,
   `sum(presence) <= capacity * (makespan - min_start)` [S]. That row is the capacity floor. Nothing
   else in the default configuration can beat it: time-tabling only reasons about fixed unit tasks, and
   the stronger propagators are off or size-capped [S]. The bound sitting on the floor is the designed
   behaviour, not a symptom.
2. **Full size never improves.** The scheduling LNS neighbourhoods never fix a start time. They fix
   presence literals and add ordering precedences, so each fragment is still the whole 109K-variable
   model, copied and presolved per call, with a default budget of 0.1 deterministic time [S]. That this
   budget is too small at full size is [I], but it fits the observation, and the three parameters that
   control it have never been measured here [R].
3. **Interleaved search spends about 40% of the budget on full-model searches** regardless of
   `num_workers`, and gives no violation local search at 8 workers [S].
4. **Linearization level 2 at the top level leaks** into `fixed`, `quick_restart` and the "stalling"
   half of the LNS calls [S]. The scratch cumulative gets an energy-cut generator that the source itself
   describes as cubic, with no size cut-off [S].
5. **There is no unit-duration special case** for capacity 2 or more [S].
6. **OR-Tools 10.0-rc.1 exists** (2026-11-23). It still drops the bound workers under
   `interleave_search`, fixes a non-determinism bug in interleaved LNS, and adds LNS presolve limits and
   a presolve rule that splits a cumulative using precedences [S10].

---

## 1. Cumulative over thousands of unit intervals

### 1.1 Presolve (`cp_model_presolve.cc`, `CpModelPresolver::PresolveCumulative`, line 6964 onward)

- Merges identical intervals that appear twice in the same cumulative by summing demands (line 6991) [S].
  It does not merge different intervals that share a start variable.
- Removes absent intervals, zero demands, zero sizes (line 7063) [S].
- Splits the constraint into time-disjoint components (line 7182) [S]. Our op windows overlap in a
  chain across the whole horizon, so this does nothing for us [I].
- Removes intervals that cannot meet a possible overload ("Presolving techniques and linear relaxations
  for cumulative scheduling", Heinz; line 7253) [S].
- **Capacity 1 only:** if every demand exceeds half the capacity, the constraint becomes `no_overlap`,
  or `all_different` on the starts when every size is 1 and no interval is optional (line 7430) [S].
  Our Flow engine has optional intervals, so it becomes `no_overlap` at best.
- **No rule exists for unit size and unit demand with capacity 2 or more.** The full list of
  `cumulative:` rule names in the file has none [S]. Capacity 12, 6 and 2 stay generic cumulatives.
- `RunPropagatorsForConstraint` (line 8019) loads each cumulative alone with every extra propagator on
  (overload checker, timetable edge finding up to 1,000 intervals, hard precedences) and keeps the
  level-zero bound changes [S]. This is root-only.
- The cumulative is never rewritten to a time-indexed form. `CumulativeTimeDecomposition` in
  `cumulative.cc` builds exactly that encoding (one Boolean per task and time point, one pseudo-Boolean
  row per time point), but nothing outside tests calls it in 9.15 [S]. `encode_cumulative_as_reservoir`
  (`cp_model_expand.cc:2844`) is documented as test-only [S].

### 1.2 Propagators at load time (`cumulative.cc`, `Cumulative()`)

In order, as the function builds them [S]:

| Propagator | Default | Condition and cost |
|---|---|---|
| Per-task `demand <= capacity` linear rows | always | one row per task |
| `Disjunctive` on tasks with `2 * demand_min > capacity` | on (`use_disjunctive_constraint_in_cumulative`) | only fires at capacity 1 for us; if every task qualifies, no cumulative propagator is created at all |
| `CumulativeIsAfterSubsetConstraint` | off (`use_hard_precedences_in_cumulative`) | the precedence list is filled only if `exploit_all_precedences` is also true; otherwise the loop body never runs (confirmed; the repo already sets both [R]) |
| `TimeTablingPerTask` | **always** | the only default propagator for capacity 2 or more |
| Overload checker | off (`use_overload_checker_in_cumulative`) | energy over task intervals; raises the capacity lower bound or fails |
| DFF overload checker | off (`use_conservative_scale_overload_checker`) | skipped unless some demand exceeds 1, so never for our engine cumulatives |
| `TimeTableEdgeFinding` | off (`use_timetable_edge_finding_in_cumulative`) | **also requires `NumTasks() <= max_num_intervals_for_timetable_edge_finding`, default 100.** O(n^2) per call (`timetable_edgefinding.h`) |

Consequences:

- `use_timetable_edge_finding_in_cumulative=true` alone is a silent no-op on any cumulative with more
  than 100 intervals [S]. The repo has the override (`timetable_edge_finding_max_intervals`) [R].
- There is no parameter named `use_energetic_reasoning_in_cumulative`. The `use_energetic_reasoning...`
  parameters are all `..._in_no_overlap_2d` (`sat_parameters.proto` lines 1062 to 1071) [S]. The repo's
  "energetic reasoning" switch is its own redundant rows, not a CP-SAT propagator [R].
- `use_dynamic_precedence_in_cumulative` (default off, marked experimental) changes branching, not
  propagation: it adds `CumulativePrecedenceSearchHeuristic` to the scheduling heuristic
  (`cp_model_search.cc:365`) [S].

### 1.3 What time-tabling can and cannot do with unit tasks

`TimeTablingPerTask::BuildProfile` (`timetable.cc:411`) adds a task to the profile only when it is
present and `StartMax < EndMin` [S]. For a size-1 task that means the start is fixed. So for the engine
cumulatives:

- Time-tabling prunes cycle `t` from other tasks only after `capacity` tasks are already fixed at `t` [S].
- It has no pigeonhole reasoning. Twenty-five ALU ops whose windows are all the two cycles `t` and
  `t + 1` (24 slots) are infeasible, but time-tabling sees nothing until search has fixed 12 of them
  in one cycle. The same holds for any Hall-set violation over a longer window [I from S].
- The header calls it "a strongly quadratic version of Time Tabling" (`timetable.h:112`) [S]. Each call
  rebuilds the profile and sweeps all tasks twice (forward and reversed).

The scratch cumulative has variable-length intervals, so its tasks do get compulsory parts as soon as
`start_max < end_min`, and time-tabling is meaningful there [S].

### 1.4 The linear relaxation of a cumulative (`linear_relaxation.cc`)

**Level 1 and above: one static row per cumulative** (`AddCumulativeRelaxation`, line 815; called from
`TryToLinearizeConstraint` line 1517 with no level check) [S]:

```
sum over optional tasks(energy_min * presence) + sum over present tasks(energy)
    <= capacity_max * (span_end - span_start)
```

- `span_start` is the constant minimum start. `span_end` is the makespan expression if one is detected,
  else the constant maximum end (line 1002) [S].
- The makespan is detected in two ways: a "makespan interval" (always present, demand equal to the
  capacity, end fixed at the horizon; `DetectMakespan`, line 648), or a variable that the precedence
  closure shows to be after every end (`DetectMakespanFromPrecedences`, line 688, which requires every
  end to be `1 * var + const`) [S]. Whether our `max_cycle` is detected for each cumulative is not
  measured [R: listed under "Not measured" in `CPSAT_CONFIG_ABLATION.md`].
- If no task is optional and nothing is variable, the row is skipped entirely (line 878) [S].
- **For unit sizes and unit demands this row is `count of present ops <= capacity * (makespan - 0)`,
  the capacity floor.** With VALU shift it is the pair of rows whose joint LP optimum the repo already
  computes as "vector-op equivalents / 7.5" [I from S, consistent with R].

**Level 2 only: cut generators** (`AddCumulativeCutGenerator`, line 1721). All are marked
`only_run_at_level_zero` (`scheduling_cuts.cc`) [S]:

| Generator | Created when | What it adds | Use to us |
|---|---|---|---|
| Time-table cuts (`CreateCumulativeTimeTableCutGenerator`, `scheduling_cuts.cc:747`) | always | at each time point, mandatory demands (tasks with `start_max < end_min`) sum to at most the capacity | none for unit tasks at the root: nothing has a mandatory part unless fixed |
| Completion-time cuts (`CreateCumulativeCompletionTimeCutGenerator`, line 1904) | always | exhaustive on subsets of about 6 tasks (`kCtExhaustiveTargetSize = 6`) plus Queyranne-style energy cuts; predecessor detection is skipped above 100 events (line 1146) | weak at our scale |
| Pairwise precedence cuts (`CreateCumulativePrecedenceCutGenerator`, line 1002) | always | Balas disjunctive cuts on pairs that cannot overlap | none at capacity 2 or more with unit demands |
| Energy cuts (`CreateCumulativeEnergyCutGenerator`, line 606) | **only if some size or demand is not fixed, or the capacity is not fixed** (`linear_relaxation.cc:1749`) | energy over every time window | not created for the engine cumulatives (fixed size and demand; optional presence does not count). **Created for the scratch cumulative.** |

Cost of the energy cut generator [S]: `GenerateCumulativeEnergeticCuts` (line 518) loops over every
pair of time points and, inside, over every event. The source comment says "The complexity of this loop
is n^3 ... Still, this is not visible in the profile as we only [run] this method at the root node".
The only guard is a wall-time-limit check once per outer iteration when there are more than 50 events.
There is no size cut-off.

[I] At full size the scratch cumulative has on the order of 10^4 events and up to about 1,400 distinct
time points per bound type, so one round is of the order of 10^10 inner steps, per worker that runs
level 2. This is my best explanation for "max_lp workers spend minutes building their LP" and for the
50 to 59 deterministic-time first tasks recorded in `CPSAT_CONFIG_ABLATION.md` section 9. Not measured.

### 1.5 What level 2 adds beyond cuts

`LinearizeEnforcedConstraints` (line 1262) returns true only above level 1 [S]. So:

- Level 1: an enforced linear constraint (`OnlyEnforceIf`) is **not in the LP at all**.
- Level 2: it is added as a big-M style relaxation using the activity bounds
  (`AppendLinearConstraintRelaxation`, line 1318) [S].

Our model has many enforced precedences (shift lanes, select paths, CSE subs) [R], so level 2 adds a
large number of weak rows.

---

## 2. Presolve, loader and the named parameters

- **Precedences.** Two-variable linear rows are recorded as precedences when the linear constraint is
  loaded (`cp_model_loader.cc`, `LoadLinearConstraint`, line 1275) and feed `PrecedenceRelations` and
  the transitive closure, which is capped by `transitive_precedences_work_limit` (default 10^6;
  `precedences.cc:555`) [S]. With about 10^4 ops the full closure does not fit in 10^6 work units [I].
- **`exploit_all_precedences`** has exactly one use in the solver: `cumulative.cc:226`, to fill the list
  for `use_hard_precedences_in_cumulative`. It does nothing on its own [S].
- **`use_precedences_in_disjunctive_constraint`** (default true) and
  **`use_strong_propagation_in_disjunctive`** (default false; creates one literal per pair of tasks when
  the disjunctive has at most `max_size_to_create_precedence_literals_in_disjunctive` = 60 intervals)
  are read only in `disjunctive.cc` [S]. They affect our Flow engine only (capacity 1), and the strong
  mode not at all above 60 intervals.
- **`detect_table_with_cost`** is about table constraints from weighted CSPs (proto comment, line 524) [S].
  Irrelevant.
- **`symmetry_level`** (default 2). Level 1: detect in presolve and fix Booleans. Level 2: "some form of
  dynamic symmetry breaking during search". Level 3: also for very large models. Level 4: break as much
  as possible in presolve (proto line 1609) [S]. The symmetry graph does encode intervals, no_overlap
  and cumulative (`cp_model_symmetries.cc` lines 440, 481, 518) [S]. But the dynamic breaking is
  Boolean-only and is loaded only when `linearization_level == 0` and the worker is not core
  (`cp_model_solver_helpers.cc:1333`) [S]. With LP on, symmetry is used for LP folding in `max_lp_sym`
  only. So among our workers only `no_lp` and `quick_restart_no_lp` do dynamic symmetry breaking.
- **`AddMaxEquality` on a makespan.** Presolve has "lin_max: rewrite with precedences" for a target used
  only in the objective (`cp_model_presolve.cc:1305`) [S]. The repo already writes
  `max_cycle >= issue_i` rows directly [R], so there is nothing to gain there.
- **Hint.** A complete feasible hint is pushed as solution "complete_hint" before any worker starts
  (`cp_model_solver.cc:991`) [S], so LNS can start without a full-model worker. Each full worker also
  replays the hint for `hint_conflict_limit` (10) conflicts in its first chunk (`QuickSolveWithHint`,
  `cp_model_solver_helpers.cc:1872`); the core worker skips the hint [S].

---

## 3. Time-indexed Booleans against intervals plus cumulative

### 3.1 What CP-SAT does with "at most k of n Booleans" [S]

`SatSolver::AddLinearConstraintInternal` (`sat_solver.cc`, about line 325):

- `k = n - 1`: stored as a clause.
- `k = 1`: stored as an at-most-one in the binary implication graph (strong: clique reasoning, used by
  probing and by cut lifting).
- otherwise: handed to `pb_constraints_` (`pb_constraint.cc`), a counter/slack pseudo-Boolean
  propagator with pseudo-Boolean conflict explanation.

There is no totalizer, sorting-network or other cardinality encoding. In the LP the row goes in as is
at level 1 or above (`AppendLinearConstraintRelaxation`, unenforced branch) [S]. Non-Boolean linear rows
longer than `linear_split_size` (100) are split into sqrt(n) partial sums for propagation; pure
pseudo-Boolean rows are not (`cp_model_loader.cc:1358`) [S].

### 3.2 What that means for us

- **Propagation.** A per-cycle row `sum_i [issue_i == t] <= cap` propagates when `cap` literals are
  true, that is when `cap` ops are fixed at `t`. That is the same strength as time-tabling on unit
  tasks (section 1.3) [I]. A time-indexed model does not propagate more; it propagates the same thing
  with cheaper code and with explanations in terms of equality literals.
- **LP.** This is where the difference is. The interval model's LP has one aggregate row per engine.
  A time-indexed LP has one row per cycle and engine, and any sum of consecutive cycle rows is the
  energy inequality for that window. So the time-indexed LP contains every window energy bound, and with
  disaggregated precedences (`sum_{s<=t} x[j][s] <= sum_{s<=t-1} x[i][s]`) it also couples them with
  heads and tails [I; the general result that time-indexed LPs are the strongest known for
  machine scheduling is M: Sousa and Wolsey 1992, Queyranne and Schulz 1994].
- **Size.** With domains `[ASAP, greedy + 15]` the number of Booleans is `sum of window widths`. For a
  25-cycle window model or a 1,000 to 2,000 op instance with a tight cap this is 10^4 to 10^5, which
  CP-SAT's LP handles. At full size without windows it is not viable [I].
- CP-SAT already creates `[x >= v]` literals lazily and can fully encode a variable; a redundant
  per-cycle row can be added on top of the existing integer starts without removing the intervals
  (hybrid). This keeps LNS's scheduling neighbourhoods, which need interval constraints
  (`cp_model_solver.cc:1947`) [S].

### 3.3 What maintainers say

I could not re-verify quotations in this session (see "Gaps"). My recollection [M]: the maintainers'
standing advice is intervals plus `AddCumulative`/`AddNoOverlap` for general scheduling because the
time-indexed form scales with the horizon, with the caveat that for unit durations and short horizons a
Boolean model can win. The source-level facts above are consistent with that: CP-SAT has no
unit-duration specialisation and never builds the time-indexed LP itself.

---

## 4. Lower-bound machinery

### 4.1 What each worker does [S]

All defined in `GetNamedParameters` (`cp_model_search.cc:505`).

- **`objective_shaving`** (`shaving_solver.cc`, `ObjectiveShavingSolver`). Copies the model, replaces the
  objective by the constraint `objective in [lb, target]`, clears the objective, presolves the copy from
  scratch and solves it as a feasibility problem. When `ub - lb <= shaving_search_threshold` (64) the
  target is `lb` itself, otherwise `lb + LogUniform(0, (ub - lb) / 2)`. INFEASIBLE raises the shared
  lower bound to `target + 1`; a solution is posted as a new incumbent (lines 73 to 240). It runs with
  probing level 0, symmetry level 0 and the extra scheduling propagators. **This is a destructive lower
  bound loop.** The comment says "Not having the objective open up more presolve reduction."
- **`AddExtraSchedulingPropagators`** (line 179) is what `use_dual_scheduling_heuristics` (default true)
  switches on for the bound workers: `exploit_all_precedences`, `use_hard_precedences_in_cumulative`,
  `use_overload_checker_in_cumulative`, `use_strong_propagation_in_disjunctive`,
  `use_timetable_edge_finding_in_cumulative`, `use_conservative_scale_overload_checker`. It does **not**
  raise `max_num_intervals_for_timetable_edge_finding`, so even the bound workers run no edge finding
  on a cumulative with more than 100 intervals.
- **`lb_tree_search`** (`lb_tree_search.h:47`): "a classic MIP tree search by having an exhaustive list
  of open nodes ... It is all about improving the lower bound." Level 2 LP, lazy constraints off,
  100,000 root LP iterations, does not import objective bounds.
- **`objective_lb_search`** (`integer_search.cc:437`, `ShaveObjectiveLb`): at decision level 0 the first
  decision is `objective <= lb + LogUniform(0, (ub - lb) / 2)`; the normal search then tries to refute
  it, and a refutation raises the bound through conflict learning.
- **`probing`** (`ContinuousProber`, `integer_search.h:329`): loops over variables and probes and shaves
  their bounds, with a small shaving budget (0.001 dtime) that it increases.
- **`variables_shaving`** (`VariablesShavingSolver`): minimises or maximises single variables on a model
  copy to shrink domains. Level is `num_workers / 20` unless `variables_shaving_level` is set
  (`cp_model_solver.cc:1858`), so it is off below 20 workers.
- **`core`**: `optimize_with_core`, level 0. **Dropped when the objective has one term**
  (`cp_model_search.cc:958`, "Disable core search if there is only 1 term in the objective") [S]. So with
  `--cpsat-objective makespan` there is no core worker in any mode. With the default objective it runs,
  but the source says of the core path "This doesn't work with splitting in chunk for now"
  (`cp_model_solver_helpers.cc:1838`), and interleaved search runs every full worker in chunks of 1
  deterministic time (`cp_model_solver.cc:1177`). What the core worker achieves per chunk under
  interleave is therefore doubtful [I].

### 4.2 Why interleave drops them (confirmed) [S]

`GetFullWorkerParameters` (`cp_model_search.cc`, lines 942 to 975):

```
// TODO(user): Enable probing_search in deterministic mode. Currently it timeouts on small
// problems as the deterministic time limit never hits the sharding limit.
if (params.use_probing_search() && params.interleave_search()) continue;

// TODO(user): Enable shaving search in interleave mode.
// Currently it do not respect ^C, and has no per chunk time limit.
if ((params.use_objective_shaving_search()) && params.interleave_search()) continue;

// Disable subsolvers that do not implement the deterministic mode.
// TODO(user): Enable lb_tree_search in deterministic mode.
if (params.interleave_search() &&
    (params.optimize_with_lb_tree_search() || params.use_objective_lb_search())) continue;
```

The same three filters are still present in `v10.0-rc.1` (`cp_model_search.cc` lines 1030 to 1057) [S10].

### 4.3 What a user can do instead, reproducibly

- **Destructive lower bound by hand.** Build the model with no objective and the makespan domain
  `[lb, H]`, and solve for feasibility, `H = lb, lb + 1, ...`. This is what `objective_shaving` does
  internally (4.1). Run it with `num_workers = 1` (one named strategy, deterministic) or with
  `interleave_search`, and with the propagators of `AddExtraSchedulingPropagators` plus a raised
  `max_num_intervals_for_timetable_edge_finding`, which no built-in worker gives. Without an objective
  the LNS workers are not created (`cp_model_solver.cc:1898`) and the whole budget goes to the proof.
- **Single worker, chosen subsolver.** `num_workers = 1` with `subsolvers: ["objective_lb_search"]` or
  the equivalent parameters directly (`use_objective_lb_search = true`). With one worker the solver
  runs sequentially (`SequentialLoop`) and is deterministic for a fixed seed [S for the loop; I for
  end-to-end determinism].
- **Assumptions.** `cp_model.assumptions` are passed to the hint solve and the main solve
  (`ResetAndSolveIntegerProblem(mapping.Literals(model_proto.assumptions()), model)`). A model with
  assumptions gets no shared-tree workers (`cp_model_solver.cc:1814`) [S]. A literal `makespan <= H`
  as an assumption would give an UNSAT core over assumptions, but with a single assumption that is no
  more than the fixed-horizon solve.
- **`stop_after_first_solution`** exists and is what the fixed-horizon feasibility solve wants on the
  SAT side; the repo already forces one worker in that mode [R].

### 4.4 Why none of this is likely to prove much at full size [I]

Every bound worker ends in the same propagators and the same LP. Sections 1.3 and 1.4 show that for
unit tasks those give the capacity floor and little else. A fixed-horizon solve at `H = floor` has to
refute by search what no propagator sees. The realistic uses are (a) windows, where proofs already
succeed in seconds [R], and (b) the mid-size instances with edge finding or a time-indexed LP switched
on, to learn whether the floor is in fact the optimum.

---

## 5. LNS

### 5.1 How the budget is divided under `interleave_search` [S]

- `GetFullWorkerParameters` (`cp_model_search.cc:993`): "In interleaved mode, we run all of them" (returns before the
  `num_full_subsolvers` truncation). With the default list and the filters of 4.2 the full workers are
  `default_lp, fixed, core (multi-term objective only), no_lp, max_lp (or max_lp_sym plus max_lp),
  quick_restart, reduced_costs, quick_restart_no_lp, pseudo_costs`: 8 to 10. This matches the "ten
  full-problem subsolvers" in `CPSAT_CONFIG_ABLATION.md` [R].
- Incomplete subsolvers at 8 workers: `feasibility_pump`, `rins/rens`, `rnd_var_lns`, `rnd_cst_lns`,
  `graph_var_lns`, `graph_arc_lns`, `graph_cst_lns`, `graph_dec_lns`, `scheduling_intervals_lns`,
  `scheduling_time_window_lns`, `scheduling_resource_windows_lns` (if there are more than two distinct
  interval sets), `scheduling_precedences_lns` (`cp_model_solver.cc` lines 1849 to 2020).
  `lb_relax_lns` needs 16 workers (`lb_relax_num_workers_threshold`).
- **Violation local search: zero.** `num_violation_ls` defaults to
  `(num_workers - number of full workers + 2) / 3` (line 2071). With 8 workers and 8 to 10 full workers
  that is 0. Under interleave the full-worker count does not shrink with `num_workers`, so the "one LS
  per three threads" rule gives nothing unless `num_violation_ls` is set explicitly.
- **Scheduling.** `DeterministicLoop` (`subsolver.cc:131`) builds batches of `3 * num_workers` tasks
  (`cp_model_solver.cc:827`), picking each task by `NextSubsolverToSchedule`, which takes the subsolver
  with the lowest `GetSelectionScore` = `num_scheduled_tasks * max(0.1, deterministic_time /
  num_scheduled_tasks)` (`subsolver.h:142`). That is approximately "least deterministic time used so
  far". Every subsolver, full or LNS, therefore converges to an equal share of deterministic time.
- [I] With about 9 full workers and about 12 incomplete ones, roughly 40% of the budget is full-model
  search, roughly 20% is the four scheduling neighbourhoods, and the rest is generic neighbourhoods,
  RINS and the feasibility pump. At full size the full-model share has never produced an improvement [R].

### 5.2 What each neighbourhood relaxes [S] (`cp_model_lns.cc`)

Generic (all build on `FixGivenVariables`, line 1070, which fixes every non-relaxed variable to its
value in the base solution, except that a single objective variable is never fixed):

- `rnd_var_lns`: a uniform random `difficulty` fraction of active variables (line 1371).
- `rnd_cst_lns`: variables of randomly chosen constraints until the target count (line 1382).
- `graph_var_lns`: breadth-first over the variable-constraint graph from a random "improvable objective
  variable" (line 1428). `graph_arc_lns`, `graph_cst_lns`, `graph_dec_lns` are variants.
- `rins/rens`: fixes variables on which the LP solution and the incumbent agree.

Scheduling (all build on `GenerateSchedulingNeighborhoodFromRelaxedIntervals`, line 2167, or
`...FromIntervalPrecedences`, line 2105):

- They **start from the full model** (`helper.FullNeighborhood()`), fix the **presence literal** of
  every non-relaxed optional interval that is present in the base solution, and add **precedences
  `end(a) <= start(b)`** between non-relaxed intervals. **No start variable is fixed.**
- The precedences come from `GetSchedulingPrecedences`: for each cumulative the capacity is split
  recursively in two and tasks are dealt to the halves until every pair in a part is disjoint, then
  consecutive tasks in a part are chained (`ProcessDemandListFromCumulativeConstraint`, line 713; the
  comments cite the ICAPS 2005 partial-order-schedule paper). For the scratch cumulative this freezes
  the relative order of value lifetimes that share a "lane".
- `scheduling_intervals_lns`: relaxes a random `difficulty` fraction of active intervals (line 2249).
- `scheduling_time_window_lns`: sorts intervals by start, picks a random position, and relaxes the
  next `floor(difficulty * n)` intervals in order of end time, a contiguous window in time
  (`PartitionIndicesAroundRandomTimeWindow`, line 589). One window per call ("TODO: Consider relaxing
  more than one time window").
- `scheduling_resource_windows_lns`: an independent random window per distinct interval set (line 2316).
- `scheduling_precedences_lns`: keeps a random `1 - difficulty` fraction of the chain precedences and
  relaxes everything else (line 2260).
- **`push_all_tasks_toward_start`** is read in exactly two places, both here (lines 2301 and 2331): the
  intervals before the window are left out of the precedence chains and **all their variables are fixed
  to the base solution**. So with it on, the time-window neighbourhood becomes "prefix frozen, window
  free, suffix free to slide but order-preserving". Without it, the prefix is order-preserving too.
  It has no other effect anywhere in 9.15.

### 5.3 What follows for a makespan objective

- [I from S] A **generic** neighbourhood pins every non-relaxed op at its cycle. `makespan` is the
  maximum over all ops, so it can only drop if the neighbourhood happens to relax every op in the last
  cycle and everything that holds them there. `graph_var_lns` starts from the objective variable, whose
  constraints are the 12K `max_cycle >= issue_i` rows [R], so its "local" neighbourhood is a random
  sample of all ops. With the makespan-only objective the six generic neighbourhoods can almost never
  improve the incumbent. With the tie-breaker objective they register improvements (earlier
  completions) that do not shorten the schedule. Either way their budget share is close to wasted for
  the makespan.
- [I from S] A **scheduling** neighbourhood can shorten the makespan, because the suffix can slide.
  But its fragment is the whole model: every start variable stays free, so presolve cannot shrink it.
  At full size each call copies 585K constraints, adds chain precedences for six cumulatives,
  presolves and loads (`LnsSolver::GenerateTask`, `cp_model_solver.cc:1339` to 1560:
  `ImportAndSimplifyConstraints`, `PresolveCpModel`, `LoadCpModel`, `QuickSolveWithHint`,
  `SolveLoadedCpModel`). The window-probe result that a pinned 25-cycle window solves to optimality in
  seconds [R] is the same search on a model that presolve can actually reduce.
- [S] LNS fragments run with `use_sat_inprocessing = false`, probing level 0, symmetry level 0
  ("lns" base, `cp_model_search.cc:770`), and the objective is restricted to `<=` the base value, not
  `<` (`RestrictObjectiveUsingHint`).

### 5.4 Difficulty and time limit [S]

- Each generator has its own `difficulty` (initial `lns_initial_difficulty`, default 0.5) and its own
  `deterministic_limit` (initial `lns_initial_deterministic_limit`, default 0.1) (`cp_model_lns.h:394`).
- After each batch, `NeighborhoodGenerator::Synchronize` (line 1252) counts fragments that ended
  OPTIMAL or INFEASIBLE as "fully solved" and calls `difficulty_.Update(num_not_fully_solved,
  num_fully_solved)`. `AdaptiveParameterValue` (`ortools/util/adaptative_parameter_value.h`) multiplies
  or divides by `1 + 1 / sqrt(num_changes + 1)`, so steps shrink as changes accumulate. It converges to
  the difficulty at which half the fragments are fully solved within the limit.
- The limit only grows by 2% each time 50 more consecutive calls fail to improve, capped at 60.
  From 0.1 it takes about 35 such bumps (1,750 non-improving calls of that generator) to reach 0.2.
- After 10 consecutive non-improving calls, every other call uses `lns_stalling` parameters (portfolio
  search, random variable pool 5) and half of all calls drop the hint (`cp_model_solver.cc` lines 1390 and 1472).
- [I] At full size, 0.1 deterministic time is spent before a 109K-variable fragment finishes loading
  and replaying its hint. Every call is "not fully solved", so difficulty falls step by step, but for
  the scheduling neighbourhoods a lower difficulty does not make the fragment smaller (5.3). They can
  stay unproductive for the whole run.

### 5.5 What a user can bias [S]

| Parameter | Effect in 9.15 |
|---|---|
| `lns_initial_difficulty`, `lns_initial_deterministic_limit` | initial values of 5.4; the only LNS tuning knobs |
| `use_lns_only` | name filter `fj*`, `fs*`, `*lns` (`SubsolverNameFilter`, `cp_model_search.cc:1141`): drops every full worker, `rins/rens` and `feasibility_pump` |
| `subsolvers`, `extra_subsolvers`, `ignore_subsolvers`, `filter_subsolvers` | fnmatch patterns on names; LNS generators can be dropped individually by name |
| `subsolver_params` with `name: "lns"` | merged into the LNS base parameters (`cp_model_search.cc:791`); this is how to give LNS fragments their own settings |
| `num_violation_ls` | number of violation-LS workers; must be set explicitly under interleave (5.1) |
| `push_all_tasks_toward_start` | freezes the prefix in the two window neighbourhoods (5.2) |
| `solution_pool_size` (3), `alternative_pool_size` (1) | which base solution a fragment starts from |
| `fix_variables_to_their_hinted_value` | fixes every hinted variable before presolve (`cp_model_solver.cc:2632`). With a partial hint this is a user-side LNS primitive: hint only what should stay fixed |
| `repair_hint` | replaces the hint replay by `MinimizeL1DistanceWithHint`; irrelevant with a feasible complete hint |
| `hint_conflict_limit` (10) | conflicts allowed in the hint replay |
| `diversify_lns_params` | **dead: defined in the proto, referenced by no source file** |
| search strategy | used by `fixed` and by hint completion; LNS fragments use automatic search (`lns_base`) or portfolio (`lns_stalling`) |

### 5.6 What maintainers recommend when only LNS improves

Not re-verified in this session (see "Gaps"). My recollection [M] is that the standing advice for large
scheduling models is: more workers (more LNS), and beyond that a user-side decomposition or LNS loop
(re-solve with most of the schedule fixed and a hint), because the built-in neighbourhoods are generic.
The source facts in 5.3 give the mechanism for why a user-side window loop beats the built-in ones on
this model: real fixing lets presolve reduce the fragment.

---

## 6. After 9.15

`gh api repos/google/or-tools/releases` on 2026-11-24 lists **`v10.0-rc.1`, published 2026-11-23**, as
the only release after `v9.15` (2026-01-12). There is no 9.16. CP-SAT lines of its release notes that
bear on this model [S10, release notes]:

- "Added dedicated scheduling local search subsolver (`ls_scheduling`) implementing Nowicki and
  Smutnicki's tabu search for job-shop/machine scheduling." Job-shop neighbourhoods need disjunctive
  machines; relevance to a capacity-12 cumulative is doubtful [I].
- "Added `ContinuousProber` and new shaving/probing/branching subsolvers (`shaving*`, `probing_no_lp`,
  `alt_fixed*`, ... `ROUND_ROBIN_SHAVING_SEARCH`, `ALT_FIXED_SEARCH`)."
- "new 2D packing/scheduling presolve rules (... **splitting `cumulative` with precedences**)."
- "Reworked `ModelCopy` / `DenseModelCopy` for LNS and Local Search with **adaptive LNS presolve
  deterministic limits**, rate-limited Violation LS rebuilds, and **optional skipping of redundant
  scheduling constraints in LNS** and Feasibility Jump." New parameters
  `lns_initial_presolve_deterministic_limit` (0.5), `lns_max_presolve_deterministic_limit` (10),
  `lns_ignore_redundant_constraints` (true) [S10, `sat_parameters.proto`]. This is direct evidence that
  the per-fragment copy and presolve cost of 5.3 was a known problem.
- "Enabled `use_new_integer_conflict_resolution` by default."
- "Improved LP relaxations and MIP cuts ... upfront GCD simplification / density capping /
  sparsification in `LinearConstraintManager`."
- New parameters `try_hint_as_assumptions` (default true: "Try first to use the hint as assumption ...
  way faster if the hint is completable"), `objective_shaving_search_threshold` (1000),
  `shaving_search_use_random_search`, `cp_model_direct_solve` [S10].
- Bug fixes: "**non-determinism in interleaved LNS (#5423)**", "shaving solver deadlock (#5532, #5541)",
  "silent loss of enforcement literals in presolve (#5294)", "scheduling/2D explanation bugs".
  The first means 9.15's interleaved mode has a known reproducibility defect in LNS. The repo's three
  identical reruns [R] did not hit it, but the guarantee is weaker than the docs assume.
- **Unchanged:** the interleave filters of 4.2 [S10]. Upgrading does not give reproducible bound
  workers.
- Breaking changes for a Rust binding: C++20 required; monolithic `libortools` split into component
  libraries; `boolean_problem.*` moved out of `ortools/sat`; DRAT parameters removed.

---

## 7. Enforcement literals, shared presence literals, auxiliaries

- **Enforced linear constraints** are absent from the LP below level 2 and are big-M rows at level 2
  (1.5) [S]. In propagation they are ordinary half-reified constraints. Every one is copied into every
  LNS fragment.
- **Optional intervals** in a cumulative: counted in the profile only when present (1.3); in the LP
  energy row as `energy_min * presence` (1.4); **they do not trigger the energy-cut generator** [S].
- **Eight lane intervals sharing one presence literal.** Nothing in `cumulative.cc`,
  `scheduling_helpers` or the relaxation groups intervals by presence literal [S, by absence]. Each
  lane is an independent optional task. The LP row gets `8 * shift` because the eight terms add up,
  which is correct and as tight as that row can be. Presolve's "merged demands of identical interval"
  applies only to the same interval listed twice, so eight lanes with different start variables stay
  eight tasks even when they are hinted to the same cycle. The scheduling LNS requires exactly one
  enforcement literal per interval (`DCHECK_EQ(interval_ct.enforcement_literal().size(), 1)`,
  `cp_model_lns.cc:2132`), which our lanes satisfy.
  [I] The cost of lanes is mostly volume: 8 start variables, 8 intervals and the enforced precedence
  rows per shiftable op, in every worker and every LNS fragment.
- **Variable-size intervals** (scratch lifetimes) are what trigger the cubic energy-cut generator at
  level 2 (1.4) [S].
- **`AddMaxEquality`.** A `lin_max` gets a relaxation at level 1 and cut generators at level 2, the
  generic one only below 100 expressions (`linear_relaxation.cc:1475`) [S]. `max_lin_max_size_for_expansion`
  defaults to 0 (no Boolean expansion). The repo's makespan uses plain `>=` rows [R].
- **The makespan hub.** `max_cycle >= x` for every issue, completion and multiply-add variable [R] is a
  12K-row star. It costs little in propagation, but it makes the objective variable adjacent to
  everything, which defeats the locality of the graph neighbourhoods (5.3) [I]. Only ops with no
  successor need a row; the others are implied by precedences.

---

## 8. Configuration facts that follow from the above [S unless marked]

- `fixed`, `quick_restart`, `reduced_costs`, `pseudo_costs` and `lns_stalling` do not set their own
  linearization level in every case: `fixed`, `quick_restart` and `lns_stalling` **inherit the top-level
  value** (`GetNamedParameters`). The repo's top-level default is 2 [R: `config.rs:411`]. So with the
  repo default, two full workers and, after ten stalls, half of all LNS fragments build a level-2
  relaxation. `lns_base` forces 0.
- `reduced_costs` and `pseudo_costs` force level 2 themselves.
- Section 9 of `CPSAT_CONFIG_ABLATION.md` found level 1 against 2 "changes almost nothing" at 4x64 and
  6x64 with a budget of 20 [R]. That budget is about one batch, so `lns_stalling` (which needs ten
  non-improving calls per generator) had hardly started. The measurement does not cover the leak.

---

## 9. Ranked changes (at most 8), each with mechanism and cheapest test

Ranking is by expected effect on the two observed failures (full size never improves; mid-size bound
stuck) per unit of work. Items 1, 2 and 4 are parameter-only.

**1. Give the budget to LNS and make the fragments solvable.**
Set `use_lns_only = true` (or `ignore_subsolvers` for the full workers, `rnd_*` and `graph_*`), keep
the four `scheduling_*` neighbourhoods, set `push_all_tasks_toward_start = true`,
`lns_initial_deterministic_limit` 2 to 10 and `lns_initial_difficulty` 0.02 to 0.05.
Mechanism: 5.1 (equal deterministic-time share per subsolver, about 40% to full workers), 5.3 (generic
neighbourhoods cannot shorten a makespan), 5.4 (0.1 per whole-model fragment). The frozen prefix is
the only built-in way to let presolve shrink a fragment.
Test: 16 rounds, wavefront warm start, the existing budget of 60. Read the "LNS stats" table of the
log (calls, improving calls, fully-solved proportion `p=`, difficulty `d=`, limit `t=`): today I expect
`p` near 0 for the scheduling generators. Success is any improving call. No code change beyond passing
three parameters through `CpsatTuning`.

**2. Top-level `linearization_level = 1` and drop the level-2 workers at 16 rounds.**
`ignore_subsolvers: [max_lp, max_lp_sym, reduced_costs, pseudo_costs]`.
Mechanism: 1.4 (cubic energy-cut generator on the scratch cumulative, no cut-off), 1.5 (every enforced
row becomes a big-M row), 8 (the leak into `fixed`, `quick_restart`, `lns_stalling`).
Test: one 16-round run with `log_search_progress`; compare the "Task timing" table (first-task
deterministic time of each LP worker) and the cut table (count of `CumulativeEnergy*` cuts) with and
without. If the 50 to 59 first-task figures disappear, the mechanism is confirmed.

**3. Make the user-side rolling window the primary optimiser at 16 rounds.**
Mechanism: 5.3. Pinning (not ordering) the outside lets presolve reduce the fragment to the window;
the probe already shows proofs in seconds and 1,399 to 1,393 over the first 370 cycles [R].
`fix_variables_to_their_hinted_value` with a partial hint is the CP-SAT-native way to pin without
rebuilding domains. v10 moves the built-in LNS the same way (presolve limits, skipping redundant
scheduling constraints).
Test: already half done; finish the rolling pass over all 1,399 cycles with W = 25 and W = 50 and
report cycles saved per deterministic time, against item 1 at the same budget.

**4. Reproducible bound proving as a separate fixed-horizon solve.**
No objective, makespan domain `[floor, H]`, `num_workers = 1`, `stop_after_first_solution`,
`use_overload_checker_in_cumulative`, `use_timetable_edge_finding_in_cumulative` with
`max_num_intervals_for_timetable_edge_finding` above the largest cumulative, hard precedences with
`exploit_all_precedences`, linearization 0 or 1.
Mechanism: 4.1 (this is `objective_shaving`, which interleave drops; no objective frees presolve and
removes the LNS workers), 1.2 (edge finding is otherwise capped at 100 intervals even in the bound
workers).
Test: 4 rounds / batch 64 (incumbent 119 to 139, bound 95), `H = 95, 96, 97`, budget 20 each. Outcome
INFEASIBLE moves the bound for the first time; outcome FEASIBLE at 96 would be an 18% better schedule.
Either is informative; UNKNOWN on all three says search cannot refute what propagation misses (4.4)
and the next step is item 5.

**5. Add per-cycle cardinality rows on equality literals (hybrid time-indexed), for windows and
mid-size bound proofs only.**
`sum_i [issue_i == t] <= cap_e` per engine and cycle, lanes and optional encodings included through
their literals; optionally disaggregated precedences.
Mechanism: 3.2. The LP then contains every window energy bound instead of one aggregate row; Load,
Store and Flow rows (capacity 2, 2, 1) become near-clique constraints that probing and cut lifting use
(3.1). Propagation strength is unchanged; the gain is LP bound and explanations.
Test: on the window model with W = 25 (which solves to optimality today), add the rows and compare
deterministic time to OPTIMAL over the same 107 windows. Then the item 4 instance at linearization 1:
does the root LP bound exceed 95? A model-building change of one function; no change to lowering.

**6. Shrink the makespan hub to sink ops.**
Emit `max_cycle >= x` only for ops (and completion or multiply-add variables) with no successor.
Mechanism: 7. Removes about 12K rows from every fragment and stops `graph_*` neighbourhoods and
`DetectMakespanFromPrecedences` from walking a star. Sound because every other op precedes a sink with
latency at least 0 (must be checked for latency-0 edges and dead stores).
Test: count rows before and after at 16 rounds; Z3 or the existing `makespan_rows` proof harness for
soundness; rerun item 1 and compare fragment presolve time in the log.

**7. Set `num_violation_ls` explicitly (2 to 4) under interleave.**
Mechanism: 5.1, the default formula yields 0 at 8 workers. Violation LS evaluates cumulatives as "area
above capacity" (`constraint_violation.cc:1769`) and treats objective improvement as a hard constraint;
it needs no model copy per move, so it scales to the full model where LNS does not.
Test: one 16-round and one 4x64 run; the log's LS table shows moves and improving solutions.

**8. Evaluate `v10.0-rc.1` for LNS only.**
Mechanism: 6 (adaptive LNS presolve limits, skipping redundant scheduling constraints in LNS, cumulative
splitting with precedences, interleaved-LNS determinism fix). It does not help the bound side under
interleave.
Test: only after items 1 to 3, and only if the Rust binding can follow the library split; export one
16-round model with `ExportToFile` and solve it with the v10 `sat_runner` at the same parameters.

---

## 10. Maintainer statements and issue tracker (added after the sub-agent's report arrived)

Tag **[W]**: read by the web sub-agent via `gh api` or page fetch, with its quotations; I did not
re-read the pages myself. Google Groups pages were truncated on fetch. This section supersedes every
"not re-verified" remark in sections 3.3 and 5.6.

### 10.1 Things that change the recommendations

- **9.15 has open correctness bugs in presolve** [W]. Wrong OPTIMAL: issues #5293 and #5582
  (hint-dependent); both reporters say `presolve_inclusion_work_limit: 0` avoids it. False INFEASIBLE:
  #5564, same `ProcessAtMostOneAndLinear` family, "Should be fixed on `main`" (vitor1001). **Item 4
  relies on INFEASIBLE answers and item 5 adds at-most-k rows, so both must be run with
  `presolve_inclusion_work_limit: 0` on 9.15, and any INFEASIBLE should be cross-checked** (for
  example with presolve off on a window-sized model).
- **Crash with a complete feasible hint in 9.15**: #5593, avoided by `symmetry_level=0` per the
  reporter; "fixed on main" (lperron) [W].
- **Violation LS is not deterministic under interleave**: #5601, lperron: "ls is non deterministic."
  ... "This will be for rc2" [W]. Item 7 therefore costs reproducibility on 9.15 and on rc.1.
- **10.0-rc.1 LNS regression**: #5571 reports "every LNS fragment is discarded on large models"
  (1 of 256 LNS tasks completed against 267 on 9.15); closed 2026-11-09, before the rc.1 tag; whether
  the fix is in rc.1 is unverified [W]. `ls_scheduling` "is never enabled" with optional intervals
  (#5636, opened 2026-11-24, no reply) [W]. Item 8 drops further: check #5571 on our model first.
- **Capacity-1 unit intervals**: user ozan-cb (compiler scheduling, #5059) reports that the
  `no_overlap` to `all_diff` conversion made the model "stuck for hours"; rc.1 adds a toggle through
  `infer_all_diffs` (#5337) [W]. Our Flow engine has optional intervals, so the conversion should not
  fire (section 1.1), but it is worth one look at the presolve rule counts in a log.
- **Perron's own parameter review** (discussion #5259) [W]: "interleave_search:true <- very costly,
  try to remove", "randomize_search:true <- remove", "linearization_level:1 <- remove" (that is, leave
  the default). And (or-tools-discuss, 2023): interleaved search "is usually way slower than the non
  deterministic version". On getting determinism without the slowdown (#4783): "This is an open
  problem." The repo sets `randomize_search=true` unmeasured [R]; it should go into the next ablation.

### 10.2 Support for the ranked items

- Item 3 (user-side windows): lperron, discussion #2940: "after the first solution has been found, you
  can select random (or not) subsets of the intervals, and run some (external) LNS on each subset."
  Issue #1314: "the best option is to decompose the problem". Slides (CMU EWO, UPenn 2025): "Better than
  commercial solvers on small to medium scheduling problems, misses heuristics for large instances."
  Primer LNS chapter: CP-SAT "has to be relatively problem-agnostic... You on the other hand know a lot
  about your problem" [W].
- Item 1 (LNS share): lperron on worker tiers (or-tools-discuss, 2022): "17 workers and more: only add
  LNS workers. Useful if finding very good solutions is really hard/slow", "we tune for 16 workers".
  Discussion #2614: side constraints "make the LNS much harder". Primer: `use_lns_only` "needs to be
  combined with a time limit" [W].
- Item 2 (LP): primer, "`max_lp` is probably a waste of resources if you know that your model has a
  terrible linear relaxation." lperron's suggested set for another compiler-scheduling model (#4166):
  `num_workers:16,cp_model_probing_level:0,linearization_level:0` [W].
- Item 4 (bounds): user ozan-cb in #4946: "`objective_lb_search` is highly effective at proving lower
  bounds for my makespan minimization problems." The request to enable the bound workers under
  interleave (#4946) has no maintainer reply. lperron: "Single thread search should be deterministic."
  No maintainer advice was found on fixed-horizon feasibility solves or on the cumulative propagator
  flags; one user experiment on 2D packing found edge finding plus overload checker slower than
  defaults (discussion #3177) [W].
- Interleave mechanics confirmed by fdid-git: batches of independent chunks, synchronise, next batch;
  "For a fixed batch size, the result should be deterministic, independently of the number of
  workers." A reporter (#5199) confirms the 1.0 dtime chunk [W]. Note a discrepancy: lperron says
  feasibility_pump and rins/rens are excluded under interleave (discussion #4541); I did not see such
  a filter in `SolveCpModelParallel` at v9.15, so either it is elsewhere or it changed. Unresolved.
- Model hygiene: lperron (#4166), "use the new_fixed_size_interval or
  new_optional_fixed_size_interval ... This will speed up everything"; primer, optional intervals are
  "the most expensive" kind. The repo already uses fixed-size intervals for engine ops [R].

### 10.3 Question 3, time-indexed against intervals

No maintainer statement and no published benchmark compares the two for unit-duration cumulative
scheduling [W]. Nearest statements, lperron: "Do not add interval variables unless you really need
them ... In timetabling, most of the times, the time slots are fixed. Just use Boolean variables"
(2019); but also "it would be better not to use Boolean vars if the crux of the problem is a temporal
one" and "use one cumulative constraint with a capacity of 2" rather than n no_overlaps (discussion
#4348). And a cumulative inferred inside the solver has "not linear relaxation ... added", unlike an
explicit `AddCumulative` (discussion #3107). So item 5 stays an experiment with no external prior.
Nothing was found from Perron on how `sum(Booleans) <= k` is handled; section 3.1 is from source.

### 10.4 Release notes before 9.15 that explain current behaviour [W]

- 9.15: "Detect, store and use precedences in presolve, propagation, and cuts"; "Improved scheduling
  cuts". 9.13: same precedence work; "Speedup feasibility jump and violation ls". 9.12: "Improve
  Cumulative propagation and cuts"; "Feasible and complete hints should now be preserved after
  presolve"; "Speed up LNS". 9.7: `objective_shaving_search` introduced; recommended violation-LS
  counts "(8, 1), (16, 2), (24, 3), (32, 4)" for (workers, `num_violation_ls`), which is what the
  default formula would give in non-interleaved mode and does not give under interleave (5.1).

## Gaps

- **Maintainer statements** are now in section 10, as reported by the sub-agent; I did not re-read
  those pages. Not reachable: the Perron-Didier-Gay "Scheduling with CP-SAT" slides and the CP 2023
  CP-SAT-LP paper. Nothing found on `repair_hint`, `hint_conflict_limit`,
  `push_all_tasks_toward_start`, `detect_table_with_cost`, or scheduling-specific symmetry.
- **Not read:** `pb_constraint.cc` internals, `cumulative_energy.cc` (overload checker complexity),
  `feasibility_jump.cc`, `rins.cc`, `lb_tree_search.cc` beyond its header.
- **Deterministic-time accounting of LNS copy and presolve** (whether it is charged against the 0.1
  limit) was not traced; item 1's test reads it off the log instead.
- Every [I] is a prediction. Items 1, 2, 4 and 7 can each be settled by one run.
