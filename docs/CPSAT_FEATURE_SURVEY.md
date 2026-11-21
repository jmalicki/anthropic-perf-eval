# CP-SAT feature survey (November 2026)

At 16 rounds CP-SAT does not fail to improve the warm start because it searches badly. It
hardly searches at all. In the 16-round log examined here, seven of the ten full-problem
workers each spent their whole share of a 60-unit deterministic budget in one task: loading
the model and replaying the hint that presolve had already accepted. That task took 3
minutes to 1.5 hours of wall time per worker. In the meantime the ten LNS neighbourhoods ran
once or twice each (section 2). So the most promising items are about getting search to
happen at all:

1. **Stop the LP-heavy full-problem workers from gating the interleaved batches.**
   `ignore_subsolvers`, which `--cpsat-ignore-subsolvers` already passes through, or
   `use_lns_only`. With fewer workers in a batch, more LNS calls fit in the same budget.
   This costs nothing to try on OR-Tools 9.15.
2. **Write our own windowed LNS around CP-SAT**, using the per-section repair machinery we
   already have: fix every op outside a window of cycles and solve the window. The built-in
   scheduling neighbourhoods fix the presence literals of the intervals they do not relax,
   but they keep those intervals' cycle variables and only add ordering constraints between
   them. Each fragment is therefore still a copy of the whole 115,000-variable model, and each
   call costs minutes of wall time at 16 rounds (section 3.2).
   Perron's own slides list "missing heuristics for large instances" as CP-SAT's weak spot in
   scheduling.
3. **Tune the LNS sub-solves through `subsolver_params`** (for example `lns_initial_difficulty`
   or the fragment presolve of `lns_base`) and **switch on violation local search with
   `num_violation_ls`**.
   Our configuration gets zero violation-LS workers today, because 10 full-problem subsolvers
   leave none of the 8 threads free. Both parameters are in 9.15 and in our crate fork, but
   each needs a `CpsatTuning` field. Item 1 done with the `--cpsat-ignore-subsolvers` list
   already brings violation local search back without one; a short run confirmed it
   (section 5). Done with `use_lns_only` it would not, because that filter drops `ls` and
   `ls_lin`.
4. **Upgrading past 9.15 is not available yet.** 9.15 (January 2026) is the newest release
   and the newest Homebrew bottle. The unreleased `main` branch has exactly the fixes for our
   symptoms: the hint is tried as assumptions instead of being replayed
   (`try_hint_as_assumptions`), LNS gets an adaptive presolve budget, and redundant
   scheduling constraints are dropped from LNS fragments. These are worth tracking, and a
   `--HEAD` build is possible, but they are not a parameter change.
5. **Leave the modelling primitives alone.** A reservoir would not model scratch better than
   the cumulative does (9.15 expands it into pairwise Booleans by default). The scheduling
   propagators were already measured. The modelling change that does matter is the one the
   modelling review already ranks first: fewer, block-shaped intervals for shifted ops (the
   RCPSP multi-mode pattern of one interval with a variable demand). It also shrinks every
   LNS fragment.

Evidence labels follow
[CPSAT_VARIABLE_MODELLING_REVIEW.md](CPSAT_VARIABLE_MODELLING_REVIEW.md):

- **[V]** verified: read in the OR-Tools v9.15 or `main` sources, the installed 9.15 header,
  the `cp_sat` crate fork's proto, a solver log, or a run made for this survey.
- **[I]** inferred from verified facts.
- **[S]** speculative.

Already measured, and not repeated here, in [CPSAT_CONFIG_ABLATION.md](CPSAT_CONFIG_ABLATION.md):
worker count, interleaved and non-interleaved search, the bound workers, the makespan-only
objective, the greedy ALAP cap, the horizon guards, the makespan interval,
`push_all_tasks_toward_start`, the extra cumulative propagators (`overload_checker`,
`timetable_edge_finding`, and `hard_precedences` together with `exploit_all_precedences`),
and removing the redundant rows.

## 1. Candidates

"In 9.15" means the parameter exists in the 9.15 `SatParameters` (installed header
`sat_parameters.pb.h`) **[V]**. "Fork" means our `cp_sat` crate fork's `sat_parameters.proto`
has the field. That proto stops at tag 324 ("NEXT TAG: 325"), so the 9.15 fields with tags 325
to 355 cannot be set from Rust today **[V]**. "Switch" means we would need a new
`CpsatTuning` field. "Flag" means an existing command-line option can already express it.
"Since" names a release only where the release notes say so; otherwise it says "before 9.15",
and the proto tag number gives the rough age (tags are assigned in order).

| # | Feature (parameter / API) | Since | In 9.15? | Why it might help here | Cost | Test (ablation row) and expected signal |
|---|---|---|---|---|---|---|
| 1 | `ignore_subsolvers` / `subsolvers`: drop the LP-heavy full-problem workers (`default_lp`, `fixed`, `max_lp`, `max_lp_sym`, `pseudo_costs`, `quick_restart`, `reduced_costs`) | before 9.15 | yes, fork, **flag** | Each of these spent 50 to 59 deterministic time in its first task at 16 rounds and gated every batch; the LNS ran once or twice each (section 2) | none | `lns-heavy: --cpsat-objective makespan --cpsat-ignore-subsolvers default_lp,fixed,max_lp,max_lp_sym,pseudo_costs,quick_restart,reduced_costs`. Signal: many more LNS calls per budget in the log's "LNS stats" table, and cycles below the warm start at 16 rounds |
| 2 | `use_lns_only` (keeps only `fj*`, `fs*` and `*lns`) | before 9.15 | yes, fork, switch | Pure version of #1: no full-problem worker at all. Note that its filter also drops the violation-LS workers `ls` and `ls_lin` (they do not match `*lns`) | small switch | `use_lns_only` itself needs a switch. The row that works today is not the same portfolio: `lns-only: --cpsat-ignore-subsolvers core,default_lp,fixed,max_lp,max_lp_sym,no_lp,pseudo_costs,quick_restart,quick_restart_no_lp,reduced_costs` drops the full-problem workers but keeps `ls`/`ls_lin` (section 5). Signal as #1. The bound stays at its root value |
| 3 | External windowed LNS (the CP-SAT primer's pattern; `fix_variables_to_their_hinted_value` for a quick prototype) | n/a | n/a | Fragments with fixed variables presolve down to the window; the built-in scheduling LNS keep all 115k variables (section 3.2) | medium: a Rust loop over windows, reusing the per-section solve | New mode, not an ablation row: 16 rounds, windows of 64 to 128 cycles sliding over the wavefront warm start. Signal: cycles below the run's own warm start (the "SSA Greedy hint" line of its log; 1,398 for the default wavefront start, 1,314 with `--greedy-split-shift`) within an hour |
| 4 | `subsolver_params` with name `lns_base` / `lns_stalling` (e.g. `linearization_level`, `cp_model_presolve`) | before 9.15 | yes, fork (tag 210), switch | `lns_base` (linearization 0) runs every call until a neighbourhood has had 10 consecutive non-improving calls; only then does `lns_stalling` (our global `linearization_level: 2`) take every other call. The 16-round calls all ran with `h=base`, so the linearization level is not what made them slow. What can be tuned is the fragment presolve (`lns_base`: `cp_model_presolve`, `max_presolve_iterations`) and, once stalling, `lns_stalling`'s linearization | small switch (repeated message) | `lns-nopresolve`: `lns_base` with `cp_model_presolve: false`. Signal: lower wall time per LNS call, more calls. An `lns_stalling` override matters only in runs long enough to stall |
| 5 | `num_violation_ls` (violation-based local search on the feasibility jump code) | 9.7 | yes, fork (244), switch | Default is `(num_workers - full_subsolvers + 2) / 3` = `(8 - 10 + 2) / 3` = 0 for us. The local search supports `cumulative` and `no_overlap` | small switch; the `lns-only` ignore list of #2 turns it on with no switch (section 5), `use_lns_only` does not | `vls-2: num_violation_ls: 2` with and without #1. Signal: solutions credited to `ls`/`ls_lin` in the "Solutions" table |
| 6 | Scheduling LNS neighbourhoods `scheduling_time_window_lns`, `scheduling_resource_windows_lns`, `scheduling_intervals_lns`, `scheduling_precedences_lns` | before 9.15 (`packing_precedences_lns` added in 9.8) | yes, already on | Already in our portfolio (log). At 16 rounds only the two window ones improved anything, and only the tie-breaker | nothing to enable; `ignore_subsolvers` can keep only the window ones | `windows-only: --cpsat-objective makespan --cpsat-ignore-subsolvers 'default_lp,fixed,max_lp,max_lp_sym,pseudo_costs,quick_restart,reduced_costs,graph_*,rnd_*,scheduling_intervals_lns,scheduling_precedences_lns'` (#1's list plus the other neighbourhoods, in one value). Signal: calls per window neighbourhood up |
| 7 | `lns_initial_difficulty`, `lns_initial_deterministic_limit` (0.5, 0.1) | before 9.15 | yes, fork (307, 308), switch | The first LNS calls at 16 rounds ran at d=0.5 with t=0.10. Half the intervals relaxed with 0.1 deterministic time is a lot for a 115k-variable fragment | small | `lns-small: lns_initial_difficulty: 0.05`. Signal: more improving calls early |
| 8 | Hint replay per worker; `repair_hint`, `hint_conflict_limit` | before 9.15 | yes, fork | Every full-problem worker replays the complete, already-accepted hint with `HINT_SEARCH` in its first task (section 3.1). `hint_conflict_limit` limits conflicts, not decisions, so it cannot shorten a conflict-free replay | none to try; no 9.15 parameter skips the replay | Covered by #1/#2: those workers go away. `repair_hint` changes nothing for a feasible hint **[I]** |
| 9 | `try_hint_as_assumptions` (default true) | `main` only (tag 363) | **no** | Replaces the replay of #8 with assumptions: "This should be way faster if the hint is completable" | needs an OR-Tools build from `main` and a fork proto update | After an upgrade: the same 16-round run; signal: first full-problem chunk in seconds, not 50+ deterministic time |
| 10 | LNS presolve budget `lns_initial_presolve_deterministic_limit`, `lns_max_presolve_deterministic_limit`; `lns_ignore_redundant_constraints`; scheduling model detection (`scheduling_model.h`) | `main` only (tags 378, 379, 716) | **no** | Separates the LNS fragment's presolve budget from its search budget and grows it adaptively. It also drops detected redundant cumulatives from LNS fragments, which addresses large-fragment LNS directly | upgrade | As #9 |
| 11 | `fix_variables_to_their_hinted_value` | before 9.15 | yes, fork (192), switch | Turns the hinted variables into fixed ones at presolve. With a hint restricted to the ops outside a window, this is a one-call prototype of #3 | small | Prototype of #3 on 4 and 16 rounds |
| 12 | `solution_pool_size` (3); `alternative_pool_size` (1; tag 325); `solution_pool_diversity_limit` (tag 329) | before 9.15 | yes; 325+ **not in fork** | Lets LNS start from older or different solutions. Useless while the LNS barely runs | small; the fork proto must be synced for 325+ | Only after #1 helps |
| 13 | `objective_lb_search`, `objective_shaving`, `lb_tree_search`, `variables_shaving_level` | before 9.15 | yes; the first three are skipped under `interleave_search` | The bound sits at the capacity floor. The first three only run non-interleaved (already measured: bound unchanged). `variables_shaving_level` defaults to `num_workers / 20` = 0 and does run interleaved | small switch | `vshave-1: variables_shaving_level: 1`. Expected: nothing for the objective bound **[I]**; a better bound needs model rows (modelling review section 9) |
| 14 | `use_dynamic_precedence_in_cumulative` / `_in_disjunctive` | before 9.15 | yes, fork (268, 263), switch | Branch on "A before B" between intervals of a resource. Marked experimental. With 82,000 intervals, likely to cost more than it saves **[S]** | small switch | `dynprec`: at 4x64 only. Expected: no gain |
| 15 | `use_conservative_scale_overload_checker` | before 9.15 | yes, fork (286), switch | Helps cumulatives with non-unit demands. Our engine demands are 1, the shifted lanes are 1, and scratch is 1 or 8 against 1,536. The proto says it "does not help much to find a solution" | small | Not worth a row |
| 16 | `use_area_energetic_reasoning_in_no_overlap_2d`, `use_timetabling_in_no_overlap_2d`, packing LNS | before 9.15 | yes | Only for `no_overlap_2d`, which we do not use | n/a | none |
| 17 | `max_presolve_iterations` (3), `cp_model_probing_level`, `presolve_probing_deterministic_time_limit` | before 9.15 | yes, fork | 16-round presolve takes about 38 s, of which two probing passes take 31 s. That is small against an hour-long solve | small | `probe-1: cp_model_probing_level: 1`. Signal: presolve wall time; cycles unchanged |
| 18 | `transitive_precedences_work_limit` (1e6), `share_linear2_bounds` | 9.15 or earlier (tags 327, 326) | yes, **not in fork** | Root transitive closure of precedences. Our 26,800 RAW edges are mostly unconditional chains. `share_linear2_bounds` is documented as non-deterministic | fork proto sync | Low priority |
| 19 | `AddReservoirConstraint` for scratch | before 9.15 | yes | With variable-size liveness intervals, a cumulative is the same constraint. 9.15 expands reservoirs into pairwise precedence Booleans by default (`expand_reservoir_constraints`), which is quadratic in about 12,000 values | rewrite of section 8b | Not recommended |
| 20 | One interval per op with a mode-dependent demand (the `rcpsp_sat.py` multi-mode pattern), instead of 1 + 8 optional intervals | n/a | yes | 82,641 intervals, 65,593 of them optional. One interval with demand `1 - s` on VALU and `8 s` (or `4 s` per half block) on ALU removes most of them. CP-SAT then derives the energy from the mode literals itself (Perron, "Dealing with Energy") | large; it is the modelling review's proposal 2/7 at K = 1 or 2 | The modelling review's toy and 4-round tests |
| 21 | `NewOptionalIntervalVar` + `AddExactlyOne` for alternatives | before 9.15 | yes | We already use optional intervals with presence literals. For two modes, exactly-one is the literal and its negation. Presolve already finds the six real exactly-ones | none | none |

Not found anywhere in the 9.8 to 9.15 release notes or the `main` proto: a scheduling-specific
presolve that would compact our model, or any per-neighbourhood parameters for the
scheduling LNS **[V]**. The 9.12 to 9.15 notes mention "scheduling cuts", "detect, store and
use precedences" and `no_overlap_2d` work, none of it with a switch.

## 2. What the 16-round solver log shows

Source: a 16-round CP-SAT run of 2026-11-21 (`--cpsat-deterministic-time 60`, the wavefront
warm start with `--greedy-split-shift`, hint 1,314 cycles, 5,410 s wall). It was made on
another branch for the hint work, and the log was read for this survey **[V log]**:

- The hint was complete and feasible and was accepted before search (`#1 37.84s complete_hint`).
- Portfolio: 10 full-problem subsolvers `[core, default_lp, fixed, max_lp, max_lp_sym, no_lp,
  pseudo_costs, quick_restart, quick_restart_no_lp, reduced_costs]` and 10 LNS
  `[graph_arc_lns, graph_cst_lns, graph_dec_lns, graph_var_lns, rnd_cst_lns, rnd_var_lns,
  scheduling_intervals_lns, scheduling_precedences_lns, scheduling_resource_windows_lns,
  scheduling_time_window_lns]`. There were no violation-LS workers.
- Task timing: every full-problem subsolver ran **exactly one task**. Seven of them spent
  50.7 to 58.6 deterministic time in it, with wall times from 3.4 min (`max_lp_sym`) to
  1.49 h (`reduced_costs`). `core` spent 0.15 and `no_lp` / `quick_restart_no_lp` 1.5, each
  in 0.5 to 12.5 min of wall time.
- Search statistics: 0 conflicts in seven of the ten, and 13 in each of the other three. The number of
  branches is about the number of Booleans (for example 15,230 `BacktrackToRoot` for each).
  This is the hint replay, not search.
- LNS: 14 calls in total, 1 or 2 per neighbourhood, each with a 0.1 deterministic-time limit
  and 35 s to 12 min of wall time. Two calls improved (`scheduling_time_window_lns`,
  `scheduling_resource_windows_lns`), both only on the tie-breaker. The makespan stayed at
  1,314.
- LP size: `max_lp_sym` had 359,712 rows; the others had 24,000 to 39,000 rows over 107,000 to
  125,000 columns.
- The bound moved from the capacity floor 1,271 to 1,271 plus the tie-breaker. The
  `reduced_costs` worker supplied it.

The 4-round / batch 256 run of the same series (budget 20, hint 423 cycles, 578 s wall) shows
the same shape **[V log]**. Every full-problem subsolver ran one task (`fixed`, `max_lp`,
`max_lp_sym`, `pseudo_costs` and `quick_restart` took 14.6 to 19.9 deterministic time and 7.7 to
8.6 min each). The LNS made 14 calls. The "LNS stats" table credits 9 of them with an improvement, but only
two new solutions were reported (`rnd_var_lns` at 42 s and `graph_cst_lns` at 72 s), and both
improved only the tie-breaker. The makespan stayed at 423.

**Reading [I]:** with interleaved search a batch waits for all of its tasks. The first task
of a full-problem worker is not split into 1-deterministic-time chunks (section 3.1), so the
whole budget went into one batch dominated by seven LP workers replaying the hint. Whatever
the LNS can do at this size, they barely got to run. That would explain why no
configuration in the ablation study improved the warm start at 16 rounds while CP-SAT
improves it by 0 to 10% at 3 to 6 rounds. It is a hypothesis for items 1 and 2 to test,
not a measured cause.

## 3. Details

### 3.1 The first chunk of a full-problem worker replays the hint [V source]

`FullProblemSolver::GenerateTask` (`cp_model_solver.cc`, v9.15, lines 1102 to 1190): on the
first chunk it calls `LoadCpModel` and then `QuickSolveWithHint` (or
`MinimizeL1DistanceWithHint` under `repair_hint`). Only after that first chunk does it set
`max_deterministic_time(1)` per chunk under `interleave_search`. `QuickSolveWithHint`
(`cp_model_solver_helpers.cc`, 1872 onwards) sets `max_number_of_conflicts =
hint_conflict_limit` and `search_branching = HINT_SEARCH` and solves. A complete, feasible
hint produces no conflicts, so the conflict limit never fires and the worker walks the
whole hint with its full propagation, LP included. This happens even though presolve
already injected the same solution. On `main` the function first tries the hint as
assumptions (`try_hint_as_assumptions`, default true; `cp_model_solver_helpers.cc` around
line 1825).

### 3.2 How the built-in scheduling LNS build their fragments [V source]

From `cp_model_lns.h` v9.15 (lines 673 to 726):

- `scheduling_intervals_lns`: "select a random set of interval of the problem according to the
  difficulty. Then, for each scheduling constraints, it adds strict relation order between the
  non-relaxed intervals."
- `scheduling_precedences_lns`: keeps a random set of the precedences of the current
  solution and adds them to the model.
- `scheduling_time_window_lns`: relaxes the intervals of one random time window.
- `scheduling_resource_windows_lns`: one window per resource.

The helper "Returns a full neighborhood enriched with the set or precedences passed to the
generate method". `GenerateSchedulingNeighborhoodFromIntervalPrecedences` and
`...FromRelaxedIntervals` (`cp_model_lns.cc` 2105 to 2240) fix the presence literal of every
interval that is not relaxed, then add ordering constraints between those intervals; their
start variables stay free. Only with `push_all_tasks_toward_start` do the two window
generators also fix the variables of the intervals before the window (lines 2301 and 2331).
For us that means every shift and select mode outside the window is fixed, but the cycle
variables of all 12,000 ops remain in the fragment **[I]**. So `--cpsat-push-tasks-toward-start`
deserves one 16-round row on top of item 1: there it would freeze everything before the window.
The ablation study measured it only up to 6 rounds, where it changed nothing. In 9.15
the LNS fragment is copied from the full model and presolved inside the same 0.1
deterministic-time limit as its search (`cp_model_solver.cc` 1401 to 1470) **[V]**. At
115,000 variables and 390,000 constraints that explains minutes of wall time per call
**[I]**. Perron's scheduling slides (undated, "CP-SAT for scheduling") say "Scheduling LNS are
incomplete (only one uses the current solution)" and list scheduling as "missing heuristics
for large instances" **[V]**.

LNS parameter sets (`cp_model_search.cc` 770 to 810): `lns` turns off SAT inprocessing,
probing and symmetry; `lns_base` sets `linearization_level: 0`; `lns_stalling` keeps the base
linearization level (2 for us) and uses `PORTFOLIO_SEARCH`. A `subsolver_params` entry with
the same name is merged into these **[V]**.

### 3.3 Why there is no violation local search [V source]

`cp_model_solver.cc` 2067 to 2092: without an explicit `num_violation_ls`, the count is
`(num_workers - full_worker_subsolvers.size() + 2) / 3`, and with 8 workers and 10
full-problem subsolvers that is 0. First-solution workers (`fj`, `fs_random*`) are added
only into the threads left over, and are stopped at the first solution anyway, which the hint
provides. `constraint_violation.cc` v9.15 has violation evaluators for `kNoOverlap` and
`kCumulative` **[V]**. Whether they work well with 65,000 optional intervals is open **[S]**.

### 3.4 Hints [V source]

- `fix_variables_to_their_hinted_value` (`cp_model_solver.cc` 2632): fixes every hinted
  variable before presolve. The CP-SAT primer recommends it for checking a hint. Combined
  with a partial hint it is a cheap way to freeze everything outside a window.
- `debug_crash_if_presolve_breaks_hint` is already wired up (`model.rs`, environment
  variable). The 9.12 notes say "Feasible and complete hints should now be preserved after
  presolve", and our 16-round log confirms it ("complete and is feasible").
- `repair_hint` switches the first-chunk hint phase to `MinimizeL1DistanceWithHint`, which is
  meant for infeasible hints. Ours is feasible, so it only adds work **[I]**.

### 3.5 Upgrading

- **Releases [V]:** `gh api repos/google/or-tools/releases` lists v9.15 (2026-01-12) as the
  newest; there is no 9.16 or 10.x tag. `brew info or-tools` says "stable 9.15 (bottled),
  HEAD". The installed library is 9.15 (`9.15_16`) and the solver logs say "v9.15.9999".
- **`main` since 9.15 [V]:** 24 new `SatParameters` fields (tags up to 379). For us the
  relevant ones are `try_hint_as_assumptions`, `lns_initial_presolve_deterministic_limit`,
  `lns_max_presolve_deterministic_limit`, `lns_ignore_redundant_constraints`,
  `feasibility_jump_ignore_redundant_constraints`, `objective_shaving_search_threshold`,
  `shaving_search_use_random_search` and `shaving_search_restart_limit`. It also adds a new
  `scheduling_model.h` that detects resources, task recipes (alternative modes), barrier
  (makespan) intervals and redundant cumulatives. Commit subjects since the release include
  "[CP-SAT] simplify lns model when possible", "LNS and LS model reductions when possible" and
  "[CP-SAT] new subsolver: shaving".
- **Cost [I]:** `brew install --HEAD or-tools` (a source build), or a CMake build into a
  prefix; then point `ORTOOLS_PREFIX` at it. The fork's `sat_parameters.proto` must be
  replaced with the new one and the bindings regenerated. The fork already lags 9.15 itself
  (tags 325 to 355 are missing), so a proto sync is due in any case. Determinism must be
  re-checked (`tests/test_determinism.py`). An unreleased solver is fine for experiments,
  not for recorded results.

### 3.6 Modelling primitives

- **Reservoir [V proto, I conclusion]:** `expand_reservoir_constraints` (default true) "expand
  the reservoir constraints by creating booleans for all possible precedences between event".
  At about 12,000 live values that is out of reach. Without expansion the reservoir propagator
  enforces the same thing as our cumulative over `[def, last read)` intervals, without the
  cumulative's energy relaxation. Not recommended.
- **One interval with a mode-dependent demand [V example, I benefit]:** `rcpsp_sat.py` (v9.15,
  lines 183 to 235) gives each multi-mode task one interval whose duration and demand are
  variables tied to the mode literals. It feeds the energy `sum(literal * fixed_energy)` to the
  cumulative. Perron's slide "Dealing with Energy" says this structure "is discovered by
  inspecting the model" and is "very effective on the PSPLIB". For shifted ops this is the
  block form (K = 1 or 2) of the modelling review (sections 4.4 and 10, proposals 2 and 7).
  The review also records the cost: K = 1 raises the capacity floor from N/7.5 to N/7.
- **Makespan in the LP:** the `rcpsp_sat.py` makespan interval was measured (no gain).

## 4. Literature on CP for instruction scheduling

- Lozano, Carlsson, Hjort Blindell and Schulte, "Combinatorial Register Allocation and
  Instruction Scheduling", TOPLAS 2019 ([arXiv:1804.02452](https://arxiv.org/abs/1804.02452)):
  the Unison CP model, integrated with LLVM. It uses Gecode/Chuffed, not CP-SAT. It handles
  basic blocks of hundreds of instructions; its decomposition into a global problem and local
  per-block problems is the analogue of solving our kernel by windows.
- Gang Chen, "Optimal and Heuristic Min-Reg Scheduling Algorithms for GPU Programs", 2023
  ([arXiv:2303.06855](https://arxiv.org/abs/2303.06855)): CP-SAT with a time-indexed Boolean
  model (an N x N matrix per DAG). It solved DAGs of 48 to 511 nodes to optimality within two
  hours. A time-indexed model does not scale to our 12,000 ops (the modelling review, section 3).
- Roorda, "Optimal Software Pipelining using an SMT-Solver", 2026
  ([arXiv:2601.21842](https://arxiv.org/abs/2601.21842)): SMT, not CP-SAT. It is already cited
  in the pickup assessment for lazy register constraints, which is the "lazy scratch" idea of
  the modelling review, section 6.3.
- Heinz, Zvára, Knobloch, Hanzálek and Vilím, "Accelerating Constraint Programming Solver with
  Parallel External Heuristics", CP 2026
  ([LIPIcs.CP.2026.28](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.CP.2026.28)):
  OptalCP with external heuristics exchanging solutions during the run. They report clear
  gains on flow-shop scheduling. This supports the "our heuristic plus CP-SAT on fragments"
  direction of item 3.
- Perron, Didier and Gay, "The CP-SAT-LP Solver", CP 2023 invited talk
  ([LIPIcs.CP.2023.3](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.CP.2023.3)):
  the architecture (SAT plus simplex plus a worker portfolio). It describes no
  scheduling-specific parameters.

No CP-SAT paper on VLIW instruction scheduling or modulo scheduling at our size was found.

## 5. Commands for the top items

Table rows 1 and 2 (row 2 as the `lns-only` ignore list) need no code. These are the commands,
not results. They should be run at 16 rounds, where the problem is, with the ablation script
([use-cpsat-ablation](../.cursor/skills/use-cpsat-ablation/SKILL.md)):

```bash
WF="--greedy-barriers none --greedy-priority wavefront --cpsat-objective makespan"
LP_HEAVY="default_lp,fixed,max_lp,max_lp_sym,pseudo_costs,quick_restart,reduced_costs"
ALL_FULL="core,default_lp,fixed,max_lp,max_lp_sym,no_lp,pseudo_costs,quick_restart,quick_restart_no_lp,reduced_costs"
# The 16-round run in section 2 took 5,410 s of wall time; the script's default backstop is
# 3,600 s, so raise it.
uv run python -m scripts.run_cpsat_ablation --instances 16x256@60 --time-limit 21600 \
    --out-dir build/cpsat_survey \
    --name lns --config "baseline:$WF" \
    --config "lns-heavy:$WF --cpsat-ignore-subsolvers $LP_HEAVY" \
    --config "lns-only:$WF --cpsat-ignore-subsolvers $ALL_FULL" \
    --config "lns-heavy+push:$WF --cpsat-ignore-subsolvers $LP_HEAVY --cpsat-push-tasks-toward-start"
```

First check the results table: a row marked "(wall-clock limit hit: invalid)" stopped on the
backstop, not the budget, and does not count. What to read in each log: the "Task timing" rows of the LNS subsolvers (how many calls), the
"LNS stats" Improv/Calls column, and the "Solutions" table. A configuration that gives the
LNS 10 times the calls without improving the makespan would show that the built-in
neighbourhoods do not help at this size, and point to item 3.

A smaller version is the same command with `--instances 4x64,6x64` (budget 20; a few
minutes of wall time per cell in the ablation study). It is a cheap first look, because the
4-round / batch 256 log in section 2 shows the same pattern at budget 20. The 16-round run is
the one that tests the hypothesis.

**Short check: the configuration works on 9.15 [V run].** A short run at 2 rounds / batch 16
confirmed that the flag is accepted and that the solver runs without any full-problem
worker:

```bash
uv run generate_ssa_kernel.py --rounds 2 --batch-size 16 --cpsat-deterministic-time 2 \
    --cpsat-objective makespan --validate \
    --cpsat-ignore-subsolvers core,default_lp,fixed,max_lp,max_lp_sym,no_lp,pseudo_costs,quick_restart,quick_restart_no_lp,reduced_costs
```

- The log lists "10 ignored subsolvers", "8 first solution subsolvers" and "13 interleaved
  subsolvers: [... ls(2), ls_lin, ... scheduling_time_window_lns]". With no full-problem
  worker, the default violation-LS count becomes `(8 - 0 + 2) / 3 = 3`: two `ls` workers and
  one `ls_lin` (section 3.3).
- The run went from the hint at 43 cycles to 36 at 0.69 s. The improving solutions came
  from `ls_restart`, `scheduling_precedences_lns` (twice) and `ls_lin_restart_decay` (twice).
  36 is the critical path, so the status is OPTIMAL. The kernel was PROVED by the dataflow
  check and PASSED validation.
- This only shows that the configuration runs, and that the local search and LNS find
  solutions on their own. It is not a comparison: the default portfolio also reaches 36 on
  this instance (0.62 s in the modelling review, section 7.2).

## 6. Sources

Primary sources read for this survey:

- OR-Tools release notes, v9.6 to v9.15: <https://github.com/google/or-tools/releases>,
  <https://developers.google.com/optimization/support/release_notes>.
- OR-Tools v9.15 sources: `ortools/sat/sat_parameters.proto`, `cp_model_solver.cc`,
  `cp_model_solver_helpers.cc`, `cp_model_search.cc`, `cp_model_lns.h`, `cp_model_lns.cc`,
  `constraint_violation.cc`, `examples/python/rcpsp_sat.py`
  (<https://github.com/google/or-tools/tree/v9.15/ortools/sat>).
- OR-Tools `main` as of 2026-11-20: `sat_parameters.proto` (NEXT TAG 380),
  `cp_model_solver.cc`, `cp_model_solver_helpers.cc`, `cp_model_lns.cc`, `scheduling_model.h`,
  and the commit log of `ortools/sat` since 2026-01-12.
- Installed 9.15: `/opt/homebrew/Cellar/or-tools/9.15_16/include/ortools/sat/sat_parameters.pb.h`
  (the field-number constants).
- The `cp_sat` crate fork's `src/sat_parameters.proto` at `165da6f` (NEXT TAG 325).
- L. Perron, "CP-SAT for scheduling" (slides, Scheduling Seminar):
  <https://schedulingseminar.com/presentations/SchedulingSeminar_LaurentPerron.pdf>.
- D. Krupke et al., The CP-SAT Primer: parameters
  (<https://d-krupke.github.io/cpsat-primer/parameters.html>) and LNS
  (<https://d-krupke.github.io/cpsat-primer/lns.html>).
- The papers in section 4.

Read but not used beyond what is cited: or-tools-discuss threads on hints and workers.
