# Optimal instruction scheduling: what the literature says makes exact solving fast

24 November 2026. Reading and reasoning only; nothing was solved, built or run.

Markers used throughout:

- **[R]** read in full text (URL given).
- **[A]** abstract only.
- **[2]** read through a secondary source that restates it (named).
- **[M]** from background knowledge, not checked against the source in this session. Treat as a lead, not a fact.
- **[D]** my own derivation or judgement. Every [D] claim that matters is a proof obligation in section 9.

"Our extras" means: typed capacities (12/6/2/2/1), alternative implementations that change an
op's class and count, latency 0 and 2 edges, and the 1,536-word scratch limit.

One rule covers most of the "does it survive" questions, so it is stated once:

> **Implied constraints survive every extra; dominance constraints do not.** An implied
> constraint is derived from a relaxation (fewer constraints, cheapest mode of every op,
> minimum latency of every edge). Adding the scratch limit or mode choices only removes
> schedules, so the constraint stays valid. A dominance constraint is justified by an
> exchange argument ("move this op earlier / swap these two and nothing gets worse"). Each
> extra adds something that can get worse in the exchange, so each dominance rule needs its
> exchange argument redone with memory and modes included. Shobaki et al. 2022 measured this
> directly: of 627,715 edges that satisfy the makespan-only dominance conditions, 315,221
> (50.2 %) also satisfy the register-pressure conditions.

## 0. What could and could not be read

| Source | Status | URL |
|---|---|---|
| Malik, McInnes, van Beek, IJAIT 17:37-54, 2008 (journal version of ICTAI 2006) | [R] | https://cs.uwaterloo.ca/~vanbeek/Publications/ijait07.pdf |
| Malik, PhD thesis, Waterloo 2008 (ch. 3 and 5 read, rest skimmed) | [R] | https://cs.uwaterloo.ca/~vanbeek/Publications/malik.pdf |
| van Beek and Wilken, CP 2001 | [R] | https://cs.uwaterloo.ca/~vanbeek/Publications/cp01.pdf |
| Malik, Chase, Russell, van Beek, CP 2008 (superblocks; skimmed) | [R] | https://cs.uwaterloo.ca/~vanbeek/Publications/cp08.pdf |
| Castañeda Lozano, Carlsson, Blindell, Schulte, TOPLAS 41(3), 2019 (sections 10, 11.3 read) | [R] | https://arxiv.org/pdf/1804.02452v5 |
| Castañeda Lozano and Schulte, survey, 2014/2019 (section 4 read) | [R] | https://arxiv.org/pdf/1409.7628 |
| Shobaki, Bassett, Heffernan, Kerbow, CC 2022 | [R] | https://par.nsf.gov/servlets/purl/10379551 |
| Le, Roussel, Lecoutre, Chan, CP 2024 (section 4 read) | [R] | https://drops.dagstuhl.de/storage/00lipics/lipics-vol307-cp2024/LIPIcs.CP.2024.18/LIPIcs.CP.2024.18.pdf |
| Zheng, Qu, Cai, Mao, arXiv 2608.19711, 2026 (abstract, intro, theorem statements) | [R] partial | https://arxiv.org/pdf/2608.19711 |
| Wilken, Liu, Heffernan, PLDI 2000 | [A] + [2] survey | https://doi.org/10.1145/349299.349318 (paywalled) |
| Heffernan and Wilken, J. Scheduling 8:427-451, 2005 | [2] Malik (theorem restated), survey, Shobaki 2022 | https://doi.org/10.1007/s10951-005-2862-8 (paywalled) |
| Rim and Jain, IEEE TCAD 13(4), 1994 | [A] | https://doi.org/10.1109/43.275355 |
| Langevin and Cerny, ACM TODAES 1(4), 1996 | [A] | https://doi.org/10.1145/238997.239002 |
| Touati, register saturation papers (IJPP 2005, CC 2001, "On the optimality of RS") | [A] via the HAL API | https://hal.science/hal-00130633v1 , https://inria.hal.science/hal-00646752v1 , https://inria.hal.science/inria-00637277v1 |
| Govindarajan et al., minimum register instruction sequencing, 2003 | [2] Malik thesis ch. 5 | |
| Edwards, Baatar, Smith-Miles, Ernst, JORS 72(8), 2021 | title and abstract snippet only | https://ideas.repec.org/a/taf/tjorxx/v72y2021i8p1822-1843.html |
| Chaudhuri, Walker, Mitchell 1994; Timmer and Jess 1995; Narasimhan and Ramanujam 2000; Fernandez and Bussell 1973; Sharma and Jain; Hu; Coffman-Graham; Bertsimas and Sethuraman 2002; Kovacs and Vancza 2006; Möhring et al. 2003 | **not read** | |

Publisher sites, HAL documents and the aggregator copies refused automated download. Four
fetch sub-agents sent after the paywalled papers returned nothing usable. Everything under
"not read" appears below only as [M] and only where it changes a recommendation.

**Correction to the brief.** Malik, McInnes and van Beek do not use the Rim and Jain or
Langevin and Cerny bounds. Their lower bound is whatever constraint propagation yields on
their own model (section 1.5). Those two bounds are used in the Wilken-group enumeration
schedulers: Heffernan and Wilken compute a lower bound per block and Shobaki and Wilken
(2004) use "the cost lower bound obtained from a relaxation of the problem [Langevin and
Cerny; Rim and Jain]" as one of four pruning tests ([2] survey section 4, thesis section 3.1).

## 1. Malik, McInnes, van Beek: the CP model that solved 2,600-instruction blocks [R]

### 1.1 Model

- One integer variable per instruction: its issue cycle, domain `[lower(i), upper(i)]`
  inside `{1..m}`. Bounds representation only.
- **Latency**: `j >= i + l(i,j)` per edge. van Beek and Wilken add an edge only if it is not
  dominated by a path: drop `(i,j)` if some `k` has `l(i,j) <= l(i,k) + cp(k,j)`.
- **Resources**: one global cardinality constraint (gcc) per functional-unit type `t` over
  all variables of that type: every cycle value is taken at most `k_t` times. Bounds
  consistency with the Quimper et al. algorithm, `O(n)` after sorting. They state that for
  this problem bounds consistency "is dramatically better than other, more expensive, forms
  of consistency". A second gcc over all variables models an issue width smaller than the
  number of units (thesis 3.3.1). Multi-cycle occupancy is modelled by auxiliary "pipeline
  variables" `p_{i,j}` with bounds `[a_i + j, b_i + j]` added to the same gcc.
- Typed units are therefore handled by one gcc per type and by taking the maximum over
  types in every bound below. There is no joint reasoning across types.

### 1.2 Distance constraints (implied)

A pair `(i, j)` is a **region** if there is more than one path from `i` to `j` and no other
node lies on every path. For each region add `j >= i + d(i,j)` when `d(i,j) > cp(i,j)`.

- Regions of at most 25 nodes (50 in the 2001 single-issue paper): solve the region exactly
  in isolation with the same solver, inside-out so inner distances are available to outer
  regions.
- Larger regions: with `op(i,j,t)` the type-`t` nodes on some path from `i` to `j`
  (including `i` or `j` when they are of type `t`),

  ```
  d(i,j) = max over t of  r1 + r2 + r3 - 1
  r1 = min{ cp(i,k) : k in op(i,j,t) }
  r2 = ceil( |op(i,j,t)| / k_t )
  r3 = min{ cp(k,j) : k in op(i,j,t) }
  ```

  improved by trying to drop one to three nodes from `op(i,j,t)` when that raises `r1 + r3`
  by more than it lowers `r2`.

**Sound when**: always (relaxation: one resource type at a time, all other constraints
dropped). **Our extras**: survives all of them if `|op|` counts each op at its cheapest
usage of the type (0 if some mode avoids the type) and `cp` uses the minimum latency over
modes. Scratch: survives (implied). **Effect**: no isolated number. Stated as "key factors"
and, with safe pruning and dominance, "critical": without them "many individual basic blocks
could not be solved within the amount of time that we can now solve the entire ensemble".

**What it gives on our DAG [D].** A gather is a region: address vector, 8 scalar loads, the
consumer. With load capacity 2, `d = 1 + ceil(8/2) + 1 - 1 = 5` against a critical path of
2. A level-k select tree done as flow ops is another (flow capacity 1). Every gathered round
therefore lengthens the true per-group critical path by 3 cycles that the static heads and
tails in `makespan_rows.rs` do not see. Whether the existing slot tightening of ASAP/ALAP
already finds this should be checked before implementing; the static heads used by the
makespan rows explicitly do not ("these are not the model's ASAP/ALAP").

### 1.3 Predecessor and successor constraints (implied, propagated during search)

For node `i`, type `t`, and every subset `P` of the type-`t` immediate predecessors with
`|P| > k_t`:

```
lower(i) >= min{ lower(k) : k in P } + ceil(|P| / k_t) - 1 + min{ l(k,i) : k in P }
upper(i) <= max{ upper(k) : k in P } - ceil(|P| / k_t) + 1 - min{ l(i,k) : k in P }   (successors)
```

Propagated in `O(|pred|^2)` by sorting predecessors by lower bound. They call it an
adaptation of edge finding. **Sound**: always. **Extras**: as 1.2. **In CP-SAT**: this is
the "variable after a subset of tasks of a cumulative" rule. OR-Tools 9.15 has a parameter
`use_hard_precedences_in_cumulative` (present in the installed `sat_parameters.pb.h`;
described upstream as experimental [M]) that is meant to create exactly this propagator.
The `ortools-internals` teammate should confirm what it does before it is ablated.

### 1.4 Safe pruning (dominance)

Theorem 3.1. After propagating latency and resource constraints, if for a type `t` and an
interval `[a,b]`, with `I` the type-`t` variables whose domain meets `[a,b]`:

1. every `i` in `I` has `lower(i) = a`;
2. for every `i` in `I` and every predecessor `k`: `upper(k) + l(k,i) <= lower(i)`;
3. `|I| <= (b - a + 1) * k_t`;

then set `upper(i) = min(upper(i), b)` for all `i` in `I`. Symmetric corollary for lower
bounds. Proof: any op of `I` placed after `b` can be moved into `[a,b]`.

**Sound when**: no constraint penalises moving an op earlier. **Our extras**: typed
capacities fine (the rule is per type). Modes: an op that may change class is not reliably
in `I`; apply only to ops whose class is fixed. **Scratch: unsafe as stated.** Moving an op
earlier makes its result live longer; it frees its operands earlier only if it was their
last reader. It is safe for an op whose move cannot raise the live count at any cycle, for
example a single-result op that is the only remaining reader of an operand at least as wide
as its result [D]. It is also a propagation rule on current domains, so in CP-SAT it could
only run as a presolve on the initial domains. Not recommended.

### 1.5 Dominance constraints (Heffernan and Wilken's superior subgraphs)

Theorem 3.2 (restated from Heffernan and Wilken). Let `A`, `B` be isomorphic induced
subgraphs (same types, edges, latencies), `a_i <-> b_i`. If

1. `a_i` is neither a predecessor nor a successor of `b_i`;
2. for every predecessor `k` of `a_i` outside `A`: `l(k,a_i) <= cp(k,b_i)`;
3. for every successor `k` of `b_i` outside `B`: `l(b_i,k) <= cp(a_i,k)`;
4. for every edge `(b_i, a_j)`: `l(b_i,a_j) <= cp(a_i,b_j)`;

then adding `a_i <= b_i` for all `i` is safe. Applied iteratively (added edges change `cp`
and enable more). They search for `A`, `B` among the connected components left when a
region's source and sink are removed, with a sort-based isomorphism test, and report pairs
"sometimes with several hundred nodes each". The special case of many same-type nodes with
identical neighbours is plain symmetry breaking (order them).

**Sound when**: makespan only, renewable typed capacities. **Our extras**: section 6.

### 1.6 Search

- Lower bound `m` on the length from propagation of the model itself (critical path, node
  count over capacity, distance constraints). Upper bound from critical-path list scheduling.
- **Iterate over the makespan from below**: fix `m`, set upper bounds from `m`, run a
  complete backtracking search for a feasible schedule; if none, `m += 1`. Stop at the first
  feasible `m` or when `m` reaches the list-schedule length. Each step is a satisfaction
  problem with the tightest possible domains.
- Phase 1 (5 s): variable order smallest domain first, ties by degree; values ascending.
- Phase 2: **singleton bounds consistency** during search (tentatively fix a variable to its
  lower or upper bound, propagate, remove the bound on failure) and branch on the variable
  whose probe caused the most domain reductions.
- Phase 3 (blocks of at most 300 instructions, after a further 15 s): singleton consistency
  to depth two.
- Ideas tried and dropped in 2001 because the stronger model made them unnecessary: cycle
  cutset decomposition (cutsets of 2 to 20 nodes split the larger blocks roughly evenly),
  singleton consistency before search, symmetry-breaking constraints.

### 1.7 Results, and what regime they are in

- SPEC 2000, 168,999 blocks before register allocation, 183,112 after; largest 2,600 (2,750
  in the thesis tables). Four idealised machines: 1, 2, 4, 6 issue; the 6-issue one has 2
  integer, 1 floating point, 2 load/store, 1 branch unit. Latencies 1 to 38.
- Unsolved within 10 minutes: 0 to 22 blocks per configuration. Solved within 1 s: 99.77 to
  99.79 %; within 10 min: 99.993 to 99.999 % (thesis table 3.8).
- Against Heffernan and Wilken on the same benchmarks and a 100 s limit: 4 failures against
  more than 200, "a 50-fold improvement".
- CP 2001 single-issue: 517 non-trivial blocks solved in 4.5 s total, 22 times faster than
  the PLDI 2000 integer program. A model with only latency and all-different "does not scale
  beyond 50 instructions".
- **No per-technique ablation is published.** The only statements are that the minimal model
  stops at about 40 instructions and that distance, safe pruning and dominance constraints
  were "critical".
- **Regime.** Critical-path list scheduling was already optimal on 97.7 to 99.6 % of blocks.
  Where the optimum was better, the mean gain was 4.4 to 6.6 %, and for the largest blocks
  (251 to 2,600) 0.2 to 3.4 %. So on almost every block the propagation lower bound equals
  the optimum, and the work is a short feasibility search at `m = LB` or `LB + 1`.

**[D] Consequence for us.** These solvers succeed because the bound is tight and the
schedule is latency-bound and narrow. Our instances are capacity-bound and wide (32
independent chains), with a 20 to 30 % gap. Nothing in this literature demonstrates closing
a gap of that size at any scale; the techniques are bound-tighteners and search-space
reducers for instances that are already nearly closed. That makes the ports in section 8
worth doing (they shrink the gap from both sides) but not a route to proving optimality at
12K ops.

## 2. Wilken, Liu, Heffernan (PLDI 2000) and Heffernan and Wilken (2005)

### 2.1 Time-indexed integer program [A] + [2] survey

Binary `x_{i,k}` = instruction `i` issues in cycle `k`; each instruction exactly once; one
instruction per cycle; dependency `(i,j)` with latency `l` in aggregated form
`sum_k k*x_{i,k} + l <= sum_k k*x_{j,k}` (one row per edge). Before solving, propagation
removes variables outside `[earliest, latest]`. During solving, violated dependency
constraints in the linear relaxation are repaired with "adaptively added constraints
(cuts)". The exact cut formulas were not available to me.

### 2.2 DAG transformations [2] survey

- **Partitioning**: split the DAG at nodes every path goes through and solve the parts
  independently.
- **Redundant dependency elimination**: remove an edge implied by a path of at least the
  same latency.
- **Region linearisation**: for a region, determine an order in which its instructions can
  appear in some optimal schedule and add it as edges. This is a single-issue result; the
  exact condition was not available to me.
- **Superior nodes** (2005; extends Chou and Chung's dominance): two independent
  instructions with equal resource usage can be ordered by an artificial latency-0 edge when
  the first is at least as constrained by its predecessors and at most as constraining for
  its successors. Shobaki et al. 2022 give the simple sufficient form [R]:
  `Pred(x) subset of Pred(y)` and `Succ(y) subset of Succ(x)` (all predecessors and
  successors, not only immediate), same type, then add `x -> y`.
- **Superior subgraphs** (2005): Theorem 3.2 above.

**Results**: PLDI 2000: every block of the SPEC95 floating-point benchmarks scheduled
optimally, up to 1,000 instructions, compile time +14 % [A]. 2005: 1-, 2- and 4-issue,
latency up to 4, SPEC 2000 FP and MediaBench, blocks up to about 1,200 instructions; "up to
13.2 % of the non-trivial basic blocks were improved after graph transformations" ([2]
Malik thesis 3.1); more than 200 blocks unsolved at 100 s on the 2-issue model ([2] Malik).

**Our extras**: partitioning does not apply (no articulation structure between 32 parallel
groups) [D]. Redundant-edge elimination is always sound if the replacing path's minimum
latency over modes is at least the edge's maximum latency over modes [D]. Superior nodes:
section 6.3. Region linearisation: single issue only.

### 2.3 The stronger time-indexed precedence rows [M]

Chaudhuri, Walker and Mitchell were not readable. What I recall of the result, to be
verified before use: with `x_{i,t}` binary, the disaggregated precedence rows

```
sum_{s <= t} x_{j,s}  <=  sum_{s <= t - l(i,j)} x_{i,s}        for every edge (i,j) and cycle t
```

together with the assignment rows describe an integral polytope (the constraint matrix is
an interval/network matrix), so all fractionality comes from the resource rows; the
aggregated row of 2.1 is much weaker. CP-SAT builds this encoding lazily for integer
variables (order literals `[start_i <= t]`), so the practical question is only whether its
linear relaxation contains these rows. That is a question for `ortools-internals`, not a
literature finding.

## 3. Unison (Castañeda Lozano et al., TOPLAS 2019) [R]

- **Model**: operations with optional activation and alternative instructions (`ins(o)`),
  issue cycles, temporaries with live ranges. Resources: one `cumulative` per processor
  resource and block over `<issue(o), dur(ins(o), s), con(ins(o), s)>` for active
  operations. Alternative implementations are therefore tasks whose duration and
  consumption are indexed by the chosen instruction, which is our mode structure.
- **Register pressure** is not a count limit but a packing: each temporary is a rectangle
  (live range by register width) placed in a register array without overlap. An implied
  **cumulative over live ranges**, each consuming `width(t)` units of the register
  resource, is added on top.
- **Decomposition**: a master problem assigns registers to the operands that cross block
  boundaries; each block is then an independent subproblem; iterate with a no-good on the
  master solution and an objective cut. Not applicable to one straight-line block [D].
- **Implied constraints** relevant to us: the live-range cumulative; **distance
  constraints from region analysis** (the van Beek and Wilken / Malik ones, reused
  directly); a **cost lower bound per subproblem** from an optimally solved relaxation.
- **Symmetry**: interchangeable registers and interchangeable copies, broken with value
  precedence. Not our symmetry.
- **Dominance**: variables of inactive operations and dead temporaries "are assigned to
  arbitrary values as they do not affect the constraints in which they occur".
- **Probing** before search on the instruction choices and on objective values.
- **Portfolio**: Gecode decomposition solver with an internal portfolio of search strategies
  (schedule as early as possible, or as close as possible to users; assign instructions per
  operation, or all instruction choices first), racing Chuffed (lazy clause generation).
  49.7 % of solutions come from the decomposition solver, the rest from Chuffed.
- **Effect of all improvements together**: functions solved optimally 44.7 % to 61.7 %;
  largest function solved 647 to 946 instructions; 2.5 times more medium-sized functions
  solved; time-outs on small functions 2.1 times rarer. 15 minute limit. No finer ablation.

**Port [D]**: the inactive-variable dominance. Every timing variable of an unselected
alternative (lane cycles of an op that is not shifted, the multiply-add cycle of a select
done as a flow op, the `completion` variable the ablation doc reports as drifting) should
be a function of the selected alternative's variables. Sound with every extra provided the
variable occurs only in constraints enforced by its literal, which is the proof obligation.

## 4. Lower bounds

### 4.1 Rim and Jain 1994 [A]; algorithm [M]

Abstract: a lower bound on completion time for resource-constrained scheduling of a data
flow graph, handling chaining, multi-cycle and pipelined operations, "very fast" and "very
tight". Algorithm as I recall it, unverified: relax precedence to release times (ASAP) and
deadlines (ALAP for a target length `T`), then, per resource type, schedule the unit jobs
greedily (earliest deadline first among released jobs, `k_t` per cycle). The bound is
`T + max tardiness`, maximised over types.

**[D] This is the bound `makespan_rows.rs` already has.** For unit jobs with integer
release times and deadlines on `k` parallel units, a feasible schedule exists iff every
interval `[t1, t2]` contains at most `k * (t2 - t1 + 1)` jobs whose whole window lies inside
it (the job/slot graph is convex, so Hall's condition on intervals suffices). Maximum
tardiness is therefore `max over (t, q) of t + q + ceil(W(head >= t, tail >= q) / cap) - T`,
which is the static constant row. So implementing Rim and Jain buys nothing new. Proof
obligation 9.1 checks this equivalence.

### 4.2 Langevin and Cerny 1996 [A]; algorithm [M]

Abstract: compute for every node an "ASAP under constraint" value from the values of its
predecessors, by applying the lower-bound estimator to the subgraph of each node's
predecessors; with Rim and Jain's estimator as the inner bound it "improves the complete
lower bound in more than 50 % of the experiments reported in Rim and Jain", for "about
twice" the CPU time; "can be extended to exploit other lower-bound techniques, for example
... the number of busses or registers".

Implementable statement [D], consistent with the abstract and with 1.2 and 1.3. In
topological order, with `anc(i)` all ancestors of `i`:

```
head'(i) = max( max over edges (k,i) of head'(k) + lat_min(k,i),
                max over types e, thresholds t of
                    t + ceil( N_e(i,t) / cap_e ) - 1 + lat_min_e(i) )
N_e(i,t)     = cheapest-mode type-e slots needed by { k in anc(i) : head'(k) >= t }
lat_min_e(i) = min latency from a type-e ancestor to i (1 for us, 0 only through
               eliminable broadcasts)
```

and symmetrically `tail'`. Then feed `head'`, `tail'` to the existing `(t, q)` rows and to
the model's ASAP/ALAP. This subsumes Malik's region estimate (take the region's nodes as a
subset of the ancestors) and the static part of the predecessor constraints.

**Sound**: always (each term relaxes to one type and to release times). **Extras**: all
survive with cheapest-mode counts and minimum latencies. **Complexity**: `O(n * |anc|)`
naive; for us ancestors are confined to one group plus shared constants, so it is small.

**What it does and does not give us [D].** It lengthens per-group critical paths wherever a
group is locally wider than a capacity: gathers (+3 per gathered round at load capacity 2),
flow-op select trees, 8-lane shifted ops only if ALU capacity were below 8. It cannot see
contention between groups, because no node has ancestors in two groups. Hu's theorem [M]
says the tail-window bound is exact for unit jobs on identical machines when the precedence
graph is an in-forest; so on the single-type, forest-like part of our problem the bound we
have is already the optimum, and any real slack in the bound comes from (a) typed
interplay, (b) fan-in inside a group (what 4.2 addresses), (c) modes, (d) scratch.

### 4.3 Other bounds from the brief

- **Fernandez and Bussell 1973** [M]: the interval-load bound with critical-path heads and
  tails. Same family as the `(t, q)` rows. Not read.
- **Hu 1961, Coffman and Graham 1972** [M]: level scheduling is optimal for in-forests and
  out-forests on `m` identical units; Coffman-Graham is optimal for `m = 2` and within
  `2 - 2/m` otherwise. Neither extends to typed units; I recall Jaffe's 1980 bound for typed
  task systems (list scheduling within about `k + 1` of optimal for `k` types) but did not
  check it. The practical content: with one type and forest precedence the capacity bound
  is tight, so do not expect a cleverer single-type bound.
- **Narasimhan and Ramanujam 2000** [A, search snippet]: claim bounds provably tighter than
  the above and, for the time-constrained problem, "the tightest bounds obtainable when
  precedence constraints are ignored" via a linear program. Not read.
- **Timmer and Jess 1995, Sharma and Jain, Chaudhuri et al.**: not read. Section 2.3 is all
  I can offer on the last.
- **Time-indexed LP / Lagrangian bound** [M]: Möhring, Schulz, Stork, Uetz (2003) solve the
  Lagrangian relaxation of the time-indexed RCPSP as a minimum cut and report LP-strength
  bounds and good list schedules on instances with thousands of jobs. For 32 identical
  groups the LP optimum can spread every group fractionally, so I expect it to land on the
  capacity floor plus ramp-up and ramp-down, like the rows we have [D]. Low priority.

## 5. Many identical sub-DAGs sharing a resource

### 5.1 Safe ordering constraints

**Le, Roussel, Lecoutre, Chan (CP 2024)** [R], building on Edwards et al. 2021 (abstract
only): high-multiplicity RCPSP, classes of identical projects, cumulative resources,
precedence inside each project. Result (Lemmas 4 to 13): for identical projects `p < q` and
**every** atomic activity `a`,

```
start(a_p) <= start(a_q)
```

can be imposed on all activities at once without losing an optimal makespan. The exchange
is per activity: swap the two copies' instances of `a` whenever they are out of order
(a "(p,q)-permutation"); precedence holds by a four-case argument, resource use at every
time is unchanged. A precondition is **start-end consistency** (Definition 3):
`start(a_p) < start(a_q) iff end(a_p) < end(a_q)`; they need a separate lemma because their
durations vary. Edwards et al. report that the symmetry breaking let all solving methods
find and prove more optima (abstract).

This is the same statement as Heffernan and Wilken's Theorem 3.2 with `A`, `B` two whole
groups: conditions 1 and 4 hold because groups are unconnected, 2 and 3 hold iff every
node outside the groups (shared constants, cached tree nodes, the final stores) is
connected identically to both.

**Note for the repo.** `CPSAT_VARIABLE_MODELLING_REVIEW.md` 8.1 states that the safe form is
one ordered op per pair of groups. The literature result is stronger: every op, in every
round, provided the conditions below hold. Two rounds of the same group are *not* a valid
`A`, `B` (condition 2 fails: round `r` feeds round `r+1`), and neither are single rounds of
two groups taken separately (the previous round of `A` is an external predecessor of `A`
that is not a predecessor of `B`).

**Conditions for our model [D]** (proof obligation 9.2):

1. *Typed capacities*: fine. The per-op swap exchanges two ops of the same class.
2. *Latency 0 and 2*: fine if the latency of each edge is the same in both groups.
3. *Scratch*: fine, and the argument is short. Sort every op's cycles across the groups.
   For a value defined by `a` and last read by the latest of its readers `b, c, ...`:
   the number of copies defined by cycle `t` is unchanged; the number of copies dead by
   `t` becomes `min(#b <= t, #c <= t, ...)`, which is at least the number of copies that
   had all their readers done by `t` before. Live words can only go down, at every cycle.
   A value read by all groups dies at the maximum over all readers, which a permutation
   does not change.
4. *Alternative implementations*: **this is where it breaks.** The sort must move an op
   together with its mode, and two instances in different modes are not ordered by one
   number (a shifted op is eight lane cycles, a select has a result cycle that depends on
   its mode). Sorting by start cycle then violates start-end consistency. Safe variants:
   (a) order by a mode-independent key only ops that have a single mode in both groups;
   but the exchange argument needs every op on a path between two sorted ops to be sorted
   consistently too, so (a) is sound only if the multi-mode ops are sorted by a key that
   implies both their "operands needed by" and "result ready at" orders, or
   (b) fall back to a pure symmetry break on whole groups: one lexicographic key per pair
   of adjacent groups (for example the cycle of one designated single-mode op, with later
   positions only as tie-breakers). (b) is sound for any true automorphism of the model and
   needs no exchange argument, but prunes far less.
5. *Identical emission*: the groups must be isomorphic including their emission plan; under
   a per-group `--emission-plan` only groups with the same plan may be chained.
6. *Other symmetry breaking*: lane ordering (5.2) and group ordering come from different
   exchange arguments. They commute here (one permutes lanes inside an op, the other
   permutes op instances across groups) but that is an obligation, not a given.

### 5.2 Interchangeable nodes inside a group (superior nodes) [D on top of [2]]

The 8 scalar loads of a gather, and the 8 lanes of a shifted op, have identical
predecessor and successor sets, so by the superior-node rule they can be ordered:
`t_0 <= t_1 <= ... <= t_7`. Scratch: each lane is one word with the same producer and
consumer structure, so a swap exchanges two identical lifetimes; safe **if** the model
frees address and result words per lane. If a vector is freed only as a block of 8, the
swap is trivially neutral. This removes `8!` per gather and per shifted op.

### 5.3 Periodic schedules and a matching bound

- **Fluid attainment** (Zheng et al. 2026 [R] statements only; Bertsimas and Sethuraman
  2002 [2]): for job shops with many copies of a few job types, the fluid (pure workload)
  lower bound is attained exactly by a repeating schedule after scaling (Theorem 1), and
  for growing instances of fixed composition the optimum stays within an additive constant
  of the fluid bound (Theorem 2). Bertsimas and Sethuraman's constant is
  `(I + 2) * Pmax * Jmax` (`I` job types, `Jmax` operations per job). Their template method
  ("search job-type templates with a model whose size does not grow with the number of
  copies, then replay") reached a 1.54 % mean gap to the fluid bound on 676 instances up to
  1e5 operations, against 5.32 % for a job-indexed adaptive LNS and 6.18 % for a genetic
  algorithm.
- **Caveats for us [D]**: these are job-shop results (one unit-capacity machine per
  operation, no memory limit, no fan-in). Our multiplicity is 32 with very long jobs, so
  the additive constant is useless as a number. What transfers is the shape of the
  argument: when the per-round recurrence (a group's round `r+1` after its round `r`) needs
  far fewer cycles than one period, the steady state is resource-bound and a periodic
  schedule at the resource rate exists.
- **Modulo scheduling** [M]: the minimum period is `max(ResMII, RecMII)`. For one round of
  32 groups, `ResMII` is that round's capacity floor (about 80 cycles at 16 rounds) and
  `RecMII` is the per-group chain latency of one round (about 17 cycles; 20 with the gather
  distance of 1.2). With no binding recurrence and unit-time ops on flat typed units, a
  modulo reservation table with `cap * II >= work` can always be filled. So a periodic
  schedule at the capacity floor exists for a stream of identical rounds *if the scratch
  limit is ignored*. Exact modulo schedulers with register minimisation exist as time-indexed
  integer programs (Govindarajan, Altman, Gao 1994; Eichenberger and Davidson 1997) [2/M].
- **What this says about the gap [D, speculation until checked]**: the capacity floor is
  probably close to the true optimum of the steady state when scratch is not binding, so
  most of the 16-round gap is likely upper-bound slack plus ramp-up, ramp-down, the
  non-identical rounds (cached levels against gathered levels) and scratch. The repo's
  window probe (windows improvable by a cycle each, almost only with free encodings) is
  consistent with that. The check is cheap in model size: one period of one round type as
  a cyclic CP-SAT model (capacity on `t mod P`, precedence with wrap-around, peak live
  words of the periodic schedule) has about 1/16 of the ops; decreasing `P` until
  infeasible gives the steady-state optimum for that round type *with* scratch, and the
  difference between that `P` and the round's capacity floor is a measured answer to "is
  memory binding in steady state".

## 6. Register pressure and makespan bounds

**No source I could reach gives a makespan lower bound under a register limit.** What
exists:

- **Register saturation** (Touati) [A]: `RS(G)` = the maximum register need over all valid
  schedules, independent of functional units. If `RS <= R` the register constraint is
  redundant and can be dropped. Otherwise serial arcs are added until `RS <= R`, trying not
  to lengthen the critical path. Computing `RS` is NP-complete; reducing it is NP-hard; an
  exact integer program with `O(n^2)` variables and `O(m + n^2)` constraints exists;
  heuristics are reported as nearly optimal. It is an upper bound on *need*, not a lower
  bound on *time*.
- **Lineages** (Govindarajan et al., via Malik thesis ch. 5) [2]: chains of
  def-to-last-use instructions that can share a register; fusing lineages by adding
  latency-0 edges; registers needed is at most the colouring number of the lineage
  interference graph. Malik's exact method enumerates lineage formations and schedules each
  transformed DAG with the chapter-3 solver; his register lower bound is just the largest
  number of immediate predecessors of a node. With the no-spill register constraint the
  method solved 97.496 % of SPEC 2000 blocks and scaled only to blocks "as large as 50
  instructions" within 10 minutes, against 2,600 without it [R thesis abstract and 1.5].
  That is the clearest published measure of what a register limit does to exact
  scheduling. A heuristic and a slow exact method; no bound.
- **RP-superiority** (Shobaki et al. 2022) [R]: adding `x -> y` preserves optimal peak
  pressure and sum of live-interval lengths if (legality) `Pred(x) subset of Pred(y)` and
  `Succ(y) subset of Succ(x)`, `x`, `y` same type, and (pressure) for every register type
  an upper bound on live ranges lengthened by the swap is at most a lower bound on live
  ranges shortened. Lengthened: everything `x` defines, and everything `y` reads unless `y`
  is provably not the last reader. Shortened: everything `y` defines, and everything `x`
  reads for which `x` is provably the last reader (all other readers are predecessors of
  `x`). The proof of their Theorem 1 is omitted in the paper. To keep both makespan and
  pressure optimal, both this and the Heffernan-Wilken conditions must hold. Effect: 45 %
  fewer time-outs and solved instances 84 % faster in their branch and bound; regions up to
  5,197 instructions (most far smaller).

**Two things that follow [D]:**

1. *A redundancy test for the scratch cumulative.* The set of values live at one instant is
   an antichain of the order "`u` before `v` iff every reader of `u` is an ancestor of, or
   is, the producer of `v`". So the maximum-weight antichain of that order (a min-flow
   computation) is an upper bound on live words in any schedule. Restricted to the ops that
   can overlap a given cycle window according to ASAP/ALAP, it tells where the scratch
   constraint cannot bind, and the cumulative can omit those ops or cycles. It gives no
   makespan bound, but it shrinks the most expensive constraint. Obligation 9.5.
2. *A space-time area bound.* Scratch is a renewable resource of capacity 1,536, and a value
   of `w` words occupies it for at least `life_min` cycles (the resource-aware distance from
   its producer to its last reader, 4.2). Hence
   `makespan >= ceil( sum_v w_v * life_min(v) / (1536 - words live throughout) )`.
   On the 16-round instance my rough count is far below 1,271, so it does not bind by
   itself. It is the only memory-aware makespan bound available and costs one pass.
   Langevin and Cerny's abstract anticipates this ("other constraints such as the number of
   ... registers").

   > **Editor's note (2026-11-24, from review of the pull request that added this note).**
   > The formula as written is not a valid bound when the sum runs over all values: it
   > subtracts the words live throughout from the capacity but keeps their area in the
   > numerator. Counterexample: 768 words live for all 10 cycles of a schedule and another
   > 768 words live for 9 cycles fit in 1,536 words, yet the formula gives
   > `ceil((768 * 10 + 768 * 9) / (1536 - 768)) = 19 > 10`. Leaving the values that are live
   > throughout out of the sum gives `ceil(768 * 9 / 768) = 9` on this example. Neither form
   > has been implemented or proved; the derivation above is kept as it was written.

## 7. Search and solver configuration lessons

- **Destructive lower bound** (Malik, van Beek, Shobaki): prove infeasibility of `m`, then
  `m + 1`. Works when `LB` is within a cycle or two of the optimum. With our gap it would
  stall on the first value. The usable half is the other direction: solve the decision
  problem `makespan <= UB - k` with ALAP recomputed from that target, so presolve sees the
  tight domains, which the objective bound alone does not give it [D].
- **Singleton consistency on bounds** was Malik's fallback for every hard block. CP-SAT
  equivalents present in OR-Tools 9.15's parameters: `use_objective_shaving_search`,
  `variables_shaving_level`, `shaving_search_deterministic_time`,
  `shaving_search_threshold`. Whether they run in the configured portfolio is for
  `ortools-internals`.
- **Bounds consistency beat stronger filtering** for the cardinality constraint. The
  ablation doc's finding that extra cumulative propagators did not pay agrees.
- **Portfolios of value orders** (Unison): as early as possible against as close as
  possible to the users. The second is the memory-friendly order.

## 8. Ranked shortlist

| # | Technique | Kind | Survives extras | Reported effect |
|---|---|---|---|---|
| 1 | Resource-aware heads and tails (Langevin-Cerny recursion; Malik distance constraints) fed to the existing makespan rows, ASAP/ALAP, and as explicit `j >= i + d` rows for gather and select-tree regions | implied | yes, with cheapest-mode counts and minimum latencies | improves Rim-Jain in more than 50 % of cases at 2x CPU; "critical" in Malik |
| 2 | Order every single-mode op across identical whole groups (`start(a_g) <= start(a_g+1)`), not one op per pair | dominance | capacities yes; scratch yes by the sorting argument; modes only under 5.1 item 4 | Edwards et al.: more optima found and proved; Malik: "greatly improved" |
| 3 | Order interchangeable lanes (8 loads of a gather, lanes of a shifted op) | dominance | yes if lanes are freed per word or per block uniformly | part of Heffernan-Wilken's gains |
| 4 | Periodic template for the steady state (cyclic model of one round type, replay, CP-SAT windows at transitions); its minimum period is a per-round optimum with scratch | restriction (upper bound) plus a per-round bound | by construction | template replay 1.54 % from the fluid bound against 5.32 % for LNS |
| 5 | Pin every variable of an unselected alternative to a function of the selected one | dominance on don't-cares | yes | part of Unison's 44.7 % to 61.7 % |
| 6 | Scratch redundancy test by maximum-weight antichain; space-time area row | implied | yes | Touati: many DAGs need no register constraint |
| 7 | Decision-version solves at `UB - k` with recomputed ALAP; shaving subsolvers; `use_hard_precedences_in_cumulative` | search | yes | Malik: singleton consistency closed the hard blocks |
| 8 | Gate every future dominance edge on Shobaki's pressure condition as well as the makespan condition | discipline | this is the scratch-safety test itself | only 50.2 % of makespan-safe edges were pressure-safe |

Not on the list: safe pruning (unsafe with scratch, and dynamic), region linearisation
(single issue), DAG partitioning (no articulation structure), Rim-Jain itself (already
implemented, 4.1), a time-indexed LP bound (expected to equal the floor we have).

## 9. Proof obligations

Each to be checked with Z3 on small instances before the technique is trusted.

1. **Rim-Jain equals the static row.** For random small typed instances with heads and
   tails: `max over (t,q) of t + q + ceil(W/cap)` equals the optimum of the relaxation
   "unit jobs, release `head`, deadline `T - tail`, capacity `cap`, minimise `T`".
2. **Resource-aware heads.** For every op `i`: no schedule of the full model (all modes,
   all capacities, scratch dropped) has `start(i) < head'(i)`; same for `tail'`. For every
   emitted distance row `(i, j, d)`: no schedule has `start(j) - start(i) < d`. Include an
   instance where a mode removes an op from the counted type, and one with a latency-0
   edge out of an eliminable broadcast.
3. **Group ordering.** On 2 or 3 groups by 2 rounds with scratch and all modes enabled:
   (a) the optimum with the ordering rows equals the optimum without;
   (b) the exchange lemma itself: for any feasible schedule, the per-op sorted schedule is
   feasible with the same makespan and no cycle with more live words. Run (b) separately
   for single-mode ops only, and with multi-mode ops, expecting a counterexample for the
   latter when the sort key is the start cycle. Run (a) again with lane ordering on, to
   check the two orderings together.
4. **Lane ordering.** For one gather and one shifted op inside a 2-group instance with
   scratch: optimum unchanged, and the swap of two lanes never raises live words under the
   model's actual freeing rule (per word or per block).
5. **Antichain bound.** For small instances: maximum live words over all schedules (Z3
   maximisation) is at most the maximum-weight antichain of the order in section 6; and the
   windowed version: dropping the ops it declares non-binding from the scratch cumulative
   does not change the optimum.
6. **Area row.** `1536 * makespan >= sum of w * life_min` in every schedule, with
   `life_min` from obligation 2 and same-cycle reuse counted as the model counts it.
7. **Pinned variables.** For each pinned variable: every constraint in which it occurs is
   enforced by the literal of its alternative (a syntactic check on the exported model),
   and the optimum is unchanged on 2 rounds.
8. **Periodic template.** Not an optimality claim. The replayed schedule must pass the
   existing validators (kernel dataflow proof, capacity, scratch). The per-round period
   bound is valid only for a schedule constrained to be periodic; it must not be reported
   as a bound on the whole problem.
