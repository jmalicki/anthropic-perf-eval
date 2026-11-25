# Energetic reasoning for unit tasks: what it can and cannot do for this scheduler

This document records what the November 2026 investigation of "CP-SAT is not improving" found
about energetic (capacity-based) reasoning: what the literature and the OR-Tools source say,
what was wrong with the rule we had, the rule that replaced it, and what was measured.

- The rule in the code: `scheduler_rs/src/cpsat_scheduler/resource_bounds.rs`
  (`--asap-alap-tightening energetic`, the default).
- Its soundness checks: `verification/resource_bounds.py` (`./scripts/test.sh proofs`).
- The research notes this summarises, with per-claim evidence tags and every source:
  [research/](research/README.md).

Evidence labels used below: **measured** (a run in this repository), **source** (read in the
OR-Tools 9.15 source), **literature** (read in a paper; the research notes say how much of each
was read), **argument** (our own reasoning, checked by Z3 where stated).

## 1. Summary

1. For tasks of duration 1 and demand 1, every energy-based cumulative propagator (overload
   check, edge finding, timetable edge finding, energetic reasoning and their faster variants)
   reduces to one counting test over time windows. With wide windows that test is the capacity
   floor. A stronger cumulative propagator cannot raise our bound. (literature, argument)
2. CP-SAT 9.15's bound sits on the capacity floor by construction: the linear relaxation of a
   cumulative is one aggregate row, and time-tabling counts only tasks whose cycle is fixed.
   (source)
3. Capacity combined with precedence does say more: everything an op depends on has to run
   before it. That is the Langevin and Cerny recursion, the static form of the predecessor
   constraints of Malik, McInnes and van Beek. It is what `energetic` now computes. (literature)
4. The rule `energetic` used before was not that. It counted ops with the same earliest cycle
   and pushed "the first of them", which excludes schedules that run those first. It cut off
   better schedules at 6 rounds / batch 64. (measured, Z3)
5. The correct rule is sound for every schedule of the model, but in this model it has to be
   weaker than the textbook statement in two places, because a shifted op runs lane by lane.
   (argument, Z3 finds a violating schedule for each stronger variant) This is a statement
   about the resource-aware pass. `energetic` also runs the symmetry-aware pass, which is
   valid only together with the declared ordering rows and is skipped when they are not in
   the model (section 5).
6. Measured effect of the correct rule on CP-SAT in 120 s: none that exceeds run-to-run noise.
   The gain is that the default no longer excludes good schedules. What did move the result by
   20 or more cycles is the warm start (section 6).

## 2. Why stronger cumulative propagation does not help

A cumulative over tasks of duration 1 and demand 1 with capacity `C` says "at most `C` tasks per
cycle". That is the upper-bound half of a global cardinality constraint. When every task's
domain is an interval of cycles, as ours are (`[ASAP, ALAP]`), it is satisfiable exactly when,
for every window `[a, b)`, the tasks whose whole domain lies inside the window number at most
`C * (b - a)` (Quimper et al., CP 2003; a Hall-interval condition). For domains with holes the
condition is necessary but not sufficient: three tasks with domain `{0, 2}` and capacity 1 pass
it for every window and cannot be scheduled.

Substituting duration 1 into the energy a task must spend inside a window gives "1 if its domain
lies inside the window, else 0". So for our engine cumulatives:

| Rule | What it is for unit tasks |
|---|---|
| Time-tabling | Counts only tasks with a fixed cycle; weaker than the window count |
| Overload check | Exactly the window-count failure test |
| Edge finding, extended and timetable edge finding | Exactly window-count pruning |
| Energetic reasoning, horizontally elastic rules | The same; newer algorithms are faster, not stronger |

Our op domains are hundreds of cycles wide, so the only windows that bind are the whole horizon
and its two ends. The window count over the whole horizon is the capacity floor.

What CP-SAT 9.15 does with such a cumulative (source; details in
[research/cpsat_9_15_internals.md](research/cpsat_9_15_internals.md)):

- **Linear relaxation:** one row per cumulative, `sum(presence) <= capacity * (makespan - min
  start)`. For unit tasks that row is the capacity floor.
- **Time-tabling:** a task enters the profile only when `start_max < end_min`, which for a unit
  task means its cycle is fixed. There is no pigeonhole reasoning over a window.
- **Timetable edge finding:** created only for cumulatives of at most 100 intervals
  (`max_num_intervals_for_timetable_edge_finding`).
- **No unit-duration special case** for capacity 2 or more, and no cardinality constraint.

This is why the bound printed for 4 rounds / batch 64 is 95 and stays there, and why the
propagator switches in [CPSAT_CONFIG_ABLATION.md](CPSAT_CONFIG_ABLATION.md) changed nothing.

## 3. Where capacity does say more: precedence plus capacity

The window count ignores precedence. Combining the two gives rules that are stronger than both:

- **Predecessor and successor constraints** (Malik, McInnes, van Beek 2008): if more
  immediate predecessors of one type than fit in a cycle all start at `r` or later, the op starts
  at least `ceil(count / capacity)` cycles after `r`. They describe it as an adaptation of edge
  finding, and report it (with their distance constraints and dominance rules) as critical for
  solving blocks of up to 2,600 instructions.
- **Langevin and Cerny 1996:** apply a capacity lower bound to the subgraph of all predecessors
  of each node, in topological order, to get an "ASAP under constraint". Reported to improve the
  Rim and Jain bound in more than half of their experiments.
- **All-different with precedences** (Bessiere et al. 2011): the capacity-1 case, with a
  polynomial bounds-consistency algorithm that is strictly stronger than propagating the two
  constraints separately.

The statement all three share, for an op `o` with ancestor set `A`:

```text
start(o) >= r + ceil( W / cap )     W = slots of one engine needed by { a in A : start(a) >= r }
```

for every threshold `r` and every engine, and the mirror image over descendants. It is an
*implied* constraint: it follows from a relaxation (one engine at a time, cheapest encoding of
every op, minimum latency of every edge), so adding the scratch limit or encoding choices cannot
make it false. Dominance rules ("move this op earlier and nothing gets worse") do not have that
property and have to be re-proved for every such addition; Shobaki et al. 2022 measured that
only 50.2 % of the edges that are safe for makespan are also safe for register pressure.

The Rim and Jain bound itself (release times, deadlines, one type at a time) is, for unit tasks,
the window count again: it equals the static row of `--cpsat-makespan-rows static`. Implementing
it separately would add nothing (argument; proof obligation 9.1 of the literature note).

## 4. What was wrong with the old rule

`tighten_asap_alap_by_slots` (removed) worked per ordering group and engine:

1. count the ops of the group whose earliest cycle is `t`;
2. if the count exceeds the capacity, raise the earliest cycle of the first `count - capacity`
   ops in group order by one;
3. repeat up to three times, then propagate the raised bounds to all successors.

The count proves that *some* ops cannot run at `t`. It does not say *which*. The ops it picked
are the first in the group's order, which are the ones with the smallest latest cycle: the most
urgent ones.

- **At 6 rounds / batch 64 (measured):** the 32 level-0 `multiply_add` ops of round 3 form one
  group (the group key has no batch group in it) with earliest cycle 55 against 6 VALU slots.
  Batch group 0 was pushed from 55 to 58, and its dependency cone with it. A validated 160-cycle
  schedule had 24 ops outside the resulting domains, while the greedy schedule the model was
  hinted with (188 cycles) was inside, so nothing ever looked infeasible.
- **Minimal counterexample (Z3, `RB-old-counting-rule`):** one VALU slot; `R`; two
  multiply_adds `A`, `B` that read `R` and share an ordering group; `C` reads `A`. The best
  schedule is 3 cycles (`R`, `A`, then `B` and `C`). The rule raises `A` to cycle 2, and the best
  schedule left is 4 cycles. The ordering row `A <= B` is satisfied by the excluded schedule.
- **Why the old proofs did not catch it:** the `T7-undercount-*` theorems model one exclusion
  on two or three parallel ops without a reader. The failure needs a reader of the pushed op.

Full analysis: [research/energetic_tightening_root_cause.md](research/energetic_tightening_root_cause.md).

## 5. The rule now, and where this model forces it to be weaker

`resource_bounds.rs` computes, in one topological pass each way:

- `issue_lb(o)`: the ancestor rule of section 3, over all ancestors and six engines (VALU, ALU,
  Load, Store, Flow, and VALU+ALU jointly with a vector op as 8 units against `8 * 6 + 12`);
- `finish_tail(o)`: the mirror image, so that the op's execution end (the completion of a
  shiftable op, the issue cycle of any other, as `op_times.rs` defines it) is at most
  `horizon - 1 - finish_tail(o)`. For a select on the multiply-add path that is the issue
  cycle, not the multiply_add at issue + 1. Only issue-cycle domains are tightened.

Work is counted with the cheapest encoding (`makespan_rows::min_work`): selects, constants and
vbroadcasts count nothing because their engine, or whether they run at all, is the solver's
choice.

Two things in the model (`op_times.rs`) make the textbook rule false as stated. Both come from
VALU shift: a shifted op is eight scalar ops, and lane `i` of a reader waits only for lane `i` of
a shifted producer.

| Textbook assumption | What the model allows | What the rule does |
|---|---|---|
| An ancestor's slots are at or after its own start bound | A shifted ancestor's lanes can run before its issue cycle (the issue variable is only bounded by the completion of the producers; the lanes follow lane by lane) | Counts a shiftable ancestor from a separate *lane* bound, which follows the lane rows and is never above its issue bound |
| All descendants run after the op completes | Lanes of a shifted reader can run before a shiftable producer completes | In the tail of a shiftable op (or a vbroadcast, which may be eliminated) only descendants that cannot be shifted are counted |

Z3 finds a schedule of the model that violates each of the two stronger variants
(`RB-issue-bound-for-shifted-ancestor`, `RB-full-tail-for-shiftable`), and none that violates the
rule in the code (`RB1` to `RB3`, 120 random graphs with every kind of op plus the two graphs
that break the variants). The Python port the proof uses is pinned to the Rust code by shared
test instances (`RB0`).

**What `energetic` does besides this rule.** It also runs the symmetry-aware pass
(`tighten_asap_by_symmetry_ordering`): the k-th op of a declared ordering group on an engine
with `cap` slots starts at `ceil(k / cap) - 1` or later. That bound holds only together with
the declared ordering rows (`sym_ord`), which are in the model only with
`--cpsat-symmetry-rows declared` (the default) and are themselves not justified by a proven
symmetry (BUGS.md). With `none` or `checked` the rows are absent, and `schedule_ssa_cpsat` now
skips the pass; before, `none` still applied the bounds and excluded schedules of the model
(regression test
`test_tightening_ignores_the_declared_order_when_its_rows_are_not_in_the_model`). So "excludes
no schedule" is a property of the resource-aware pass, and of `energetic` as a whole only
relative to the model with its declared rows.

A first prototype that treated a shifted op as one block reported 11.5 % of start-domain values
removed at 6 rounds / batch 64. Part of that came from the two unsound assumptions above; the
figure should not be quoted.

## 6. Measurements

Machine: 8 workers, 120 s per solve, one solve at a time, `--greedy-barriers none`, makespan
objective. Wall-clock limited, so differences of a few cycles are within run-to-run variation.

**Domains (measured).**

| Instance | Old rule | New rule |
|---|---|---|
| 4 rounds / batch 64 | 179 earliest cycles raised (unsound) | none raised, none lowered |
| 6 rounds / batch 64 | 529 raised, of which 24 exclude a 160-cycle schedule | 352 raised (1,512 cycles in total, at most 6 per op), none lowered |

- **Heads** bind only where a group is locally wider than a capacity. Here that is the gather:
  8 loads per group against 2 load slots. 4 rounds has no gather round, so nothing changes.
- **Tails** are large: 1,052 of 1,284 ops at 6 rounds have a resource tail above their
  precedence tail, by up to 46 cycles (82 ops and up to 17 cycles at 4 rounds). They come from
  early ops whose descendants span every group. But the model's horizon is the greedy makespan
  + 20, far above any good schedule, so as static upper bounds they remove nothing.

**Tails as rows relative to the makespan (measured, on exported models, not in the code).**
One row per op, `makespan >= completion + 1 + finish_tail`:

| Instance | Without rows | With rows |
|---|---|---|
| 6 rounds / batch 64, four seeds | 183, 183, 185, 186 cycles; bound 153 | 185, 185, 188, 188 cycles; bound 153 |
| 6 rounds / batch 64, window-LNS calls per run | 132 to 158 | 110 to 127 |
| 4 rounds / batch 64, one seed | 115 cycles, bound 95 | 115 cycles, bound 95 |

No gain, and probably a small loss, so the rows were not added to the model. They exclude no
schedule, so the difference is search speed: every improvement in these runs comes from an LNS
call, each call copies and presolves the whole model, and the rows (735 constraints after
presolve, all on the makespan variable) left about 20 % fewer calls in the same time. The bound
was 153 either way. Four seeds per arm is thin; the runs without rows alone span 183 to 186.

**What does move the result (measured earlier in the same investigation).**

| Start for CP-SAT | 4 rounds / batch 64 | 6 rounds / batch 64 |
|---|---|---|
| Greedy schedule (hint) | 119 | 188 |
| CP-SAT 120 s from the greedy | 113 to 116 | 183 |
| Greedy + forward/backward justification, no solver | 108 | 160 |
| Justification iterated with random priorities, no solver | 104 | 159 |
| CP-SAT 120 s from the justified schedule | 107 | 160 |
| Solver bound | 95 | 153 |

Forward/backward justification (list-schedule the reversed graph latest-first, mirror it, list-
schedule forward again) takes milliseconds and gains more than CP-SAT does in two minutes. These
figures come from an experiment outside the generator; the justification pass is a separate
change and is not part of this one.

## 7. Why per-op counting cannot see our bottleneck

The survey predicted the small effect of section 6 before it was measured: a group has about 1.4
ops per level of its critical path against 20 or more slots per cycle, and an op's ancestors lie
in its own group (plus shared constants). The capacity bottleneck exists only in the *sum over
the 32 groups*, and no single op has ancestors in two groups. The ancestor rule sees fan-in
inside a group (the gather) and fan-out from shared ops (the tails), and nothing else.

Two more facts from the investigation point the same way:

- The dependency graph is almost asymmetric: colour refinement gives 879 classes among the 883
  ops of 4 rounds / batch 64, so the groups are not structurally interchangeable and symmetry
  breaking has little to work with.
- Freezing the encoding choices (shift, select, constant) lifts the solver's bound at 4 rounds /
  batch 64 from 95 to 104 or 105: part of the gap is the relaxation over encodings, which the
  cheapest-encoding count cannot avoid.

## 8. What the literature suggests next

Ranked in the research notes; the items that concern bounds and propagation:

1. **A time-indexed relaxation over op classes.** One integer per (op position within a group,
   cycle) counting how many groups have completed it. It is 32 times smaller than a per-op
   time-indexed model and has no group symmetry. It is the one candidate for a bound above the
   floor. Not found in the literature for this problem class; exactness is our own argument and
   fails as stated when an encoding changes an op's duration, so it needs a Z3 treatment first.
   A first attempt in this investigation was invalid (it confused op ids with positions) and
   its result was withdrawn. The near-asymmetry above means classes have to be built from the
   interior of the groups, not from the emitted group tags.
2. **Distance constraints for regions** (Malik et al.): `start(j) >= start(i) + d` for a
   gather (address, 8 loads, reader) or a flow-op select tree. Relative, so they hold wherever
   the region is scheduled, unlike a static head. The ancestor rule already yields the same
   number for the absolute bound; the relative row is not in the model.
3. **Per-cycle cardinality rows on equality literals** for windows and mid-size proofs. One
   experiment helped (a 124-cycle target at 4 rounds / batch 64 was satisfied in 27 s against
   unknown in 120 s) and one hurt; inconclusive.
4. **Not worth building for the bound:** column generation, Lagrangian decomposition, logic-
   based Benders, decision diagrams. A 2025 study finds the Dantzig-Wolfe master no stronger than
   the disaggregated time-indexed relaxation for a single mode and 0.42 % stronger for several
   modes, at 26 to 77 times the cost.

## 9. Sources

Read in full or in part during the investigation (the research notes give the reading status of
each and many more):

- Malik, McInnes, van Beek, "Optimal basic block instruction scheduling for multiple-issue
  processors using constraint programming", IJAIT 17, 2008.
  <https://cs.uwaterloo.ca/~vanbeek/Publications/ijait07.pdf>
- Quimper, van Beek, López-Ortiz, Golynski, Sadjad, "An efficient bounds consistency algorithm
  for the global cardinality constraint", CP 2003.
  <https://cs.uwaterloo.ca/~vanbeek/Publications/cp03.pdf>
- Bessiere, Narodytska, Quimper, Walsh, "The AllDifferent constraint with precedences",
  CPAIOR 2011. <https://ar5iv.arxiv.org/html/1103.3745>
- Levey, Rothvoss, "A (1+epsilon)-approximation for makespan scheduling with precedence
  constraints using LP hierarchies", STOC 2016. <https://ar5iv.arxiv.org/html/1509.07808>
- Ouellet, Quimper, "A O(n log^2 n) checker and O(n^2 log n) filtering algorithm for the
  energetic reasoning", CPAIOR 2018. <https://cquimper.github.io/publications/energetic.pdf>
- Castañeda Lozano, Carlsson, Blindell, Schulte, "Combinatorial register allocation and
  instruction scheduling", TOPLAS 41(3), 2019. <https://arxiv.org/pdf/1804.02452v5>
- Shobaki, Bassett, Heffernan, Kerbow, "Graph transformations for register-pressure-aware
  instruction scheduling", CC 2022. <https://par.nsf.gov/servlets/purl/10379551>
- Kolter, Grunow, Kolisch, "On branch-and-price for project scheduling", arXiv 2501.04563.
  <https://arxiv.org/pdf/2501.04563>
- OR-Tools 9.15 source, `ortools/sat/` (`cumulative.cc`, `timetable.cc`,
  `linear_relaxation.cc`, `scheduling_cuts.cc`, `cp_model_lns.cc`).

Abstract only (the algorithms as described here are our reconstruction and are marked so in the
notes): Rim and Jain, IEEE TCAD 13(4), 1994; Langevin and Cerny, ACM TODAES 1(4), 1996.
