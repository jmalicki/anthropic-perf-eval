# Recent OR / CP work for our scheduling problem, and what CP-SAT does not do (November 2026)

Reading and reasoning only; no solver, build or test was run.

**Problem class.** Multi-mode RCPSP with unit durations, five typed renewable resources (12 / 6 / 2 / 2 / 1), one storage-like resource (1,536 live words), latency-1 precedences, makespan objective, and 32 structurally identical groups per round coupled only through capacities.

**How this was produced.** Four sub-agents each read the web for one block of topics and wrote a part file with per-paper detail; I read all four, checked two source-code claims against the OR-Tools 9.15 tree, added the cross-cutting analysis, and ranked. The part files hold the full per-paper notes:

- `docs/research/recent_or_parts/part1_search_and_roadmap.md` (topics 1 and 7)
- `docs/research/recent_or_parts/part2_propagation_and_bounds.md` (topics 2 and 3)
- `docs/research/recent_or_parts/part3_structure_and_paradigms.md` (topics 4 and 5)
- `docs/research/recent_or_parts/part4_hls_and_compilers.md` (topic 6)

**Evidence labels.** Every citation says how much was read: *full text*, *partial*, *abstract only*, *snippet* (search excerpt), *source* (code or docs). "[reasoning]" marks our own argument, not a literature result. "[memory]" marks background knowledge with no page opened. All reading was done by the sub-agents unless marked "checked by me".

**Verification status.**
- Checked by me in `ortools/sat/cumulative.cc` and `sat_parameters.proto` at v9.15: the two gating facts in section 2.3.
- OR-Tools v10.0-rc.1 exists (its source tree is in the session scratchpad). The year of its 23 November tag date was not shown on the page; its release-note wording was read through a summariser.
- Three items rest on one sub-agent read each, and my re-check returned no report: the arXiv paper 2608.19711 (section 4.2), the 975-cycle public repository (section 6.3), and the release-note wording above. Treat their numbers as single-source.

---

## 0. What the survey says, in five points

1. **No stronger cumulative propagator will move our bound.** For unit-duration, unit-demand tasks every energy-based rule (overload check, edge finding, timetable edge finding, energetic reasoning, horizontally elastic variants) reduces to the Hall-interval count "tasks whose window lies inside [a,b) <= C(b-a)". With windows about 1,000 cycles wide, the only binding intervals are the whole horizon and its two ends. That is the capacity floor (section 2).
2. **No other solver's search is built for this regime either.** Failure-directed search and lazy-clause-generation RCPSP results come from scarce-resource, near-disjunctive benchmarks. A CP 2025 paper reports that instances with low resource constrainedness "almost never benefit", and Perron's 2025 slides list the unit-task capacity "pigeon hole" as an open propagation problem (section 1).
3. **The bound that can beat the floor is a time-indexed relaxation, and the 32 identical groups make it small.** A "counting" formulation with one integer in 0..32 per (op class, cycle) is 32 times smaller than the per-op model, has no group symmetry, has the same LP bound as the disaggregated time-indexed LP, and by our argument is exact for fixed-latency precedence, capacity and memory (section 3.2). We found no paper that applies this to RCPSP.
4. **Column generation, Lagrangian decomposition and logic-based Benders will not beat that LP.** A 2025 study shows the Dantzig-Wolfe master equals the disaggregated LP for single-mode and is 0.42 % stronger on average for multi-mode, at 26 to 77 times the cost (section 3.3).
5. **For better schedules at 12K ops, the evidence points away from whole-model exact search.** Perron: CP-SAT is "missing heuristics for large instances". One public kernel of this challenge is reported at 975 cycles on a 967 floor with a priority-searched list scheduler plus backward/forward justification and no solver. Our gap to our own floor is 43 cycles or more (sections 5.4 and 6.3).

---

## 1. Search in other CP solvers

### 1.1 Failure-directed search (CP Optimizer, OptalCP)

- **What.** A complete restarting tree search that assumes no better solution exists. Choices are binary splits of start-time domains; each branch carries a rating that falls when the branch fails quickly, and the lowest-rated choice is taken first. It has no conflict analysis; it relies on strong propagation (timetable, edge finding, not-first/not-last).
- **CP-SAT 9.15?** No. No FDS subsolver in `sat_parameters.pb.h`. The nearest are `objective_shaving`, `lb_tree_search`, `objective_lb_search` and `core`, which we measured (bound unchanged).
- **Reported effect.** Heinz, Vilím, Hanzálek 2025: lower bounds improved on 226 of 393 open PSPLIB RCPSP instances and 78 of 84 open job-shop instances at 900 s; 7 instances closed. These are destructive bounds on instances of at most 120 activities with durations 1 to 10.
- **Fit [reasoning].** Poor. With 12 + 6 slots per cycle, splitting one op's start time causes no failure until a cycle is completely full, so ratings carry no signal. Refuting "floor + k" is a counting argument over thousands of interchangeable unit tasks.
- **Benefit.** (i) none, it is a prover; (ii) none expected.
- **Sources.** Vilím, Laborie, Shaw, "Failure-Directed Search for Constraint-Based Scheduling", CPAIOR 2015, https://dblp.uni-trier.de/rec/conf/cpaior/VilimLS15.html (abstract only). Heinz, Vilím, Hanzálek, "Reinforcement Learning for Search Tree Size Minimization in Constraint Programming: New Results on Scheduling Benchmarks", Computers & Industrial Engineering 209, 2025, https://arxiv.org/html/2508.20056 (partial, first 100K characters).

### 1.2 CP Optimizer: self-adapting LNS over partial-order schedules, temporal linear relaxation

- **What.** The default search is LNS plus FDS. The LNS keeps the unrelaxed tasks as a partial-order schedule (POS): precedence arcs between consecutive users of each resource unit instead of fixed start times, so the frozen part can slide. A learning layer picks neighbourhoods by reward [memory for the last two sentences; Godard, Laborie, Nuijten 2005 abstract confirms the POS step]. Temporal linear relaxation is an LP over start/end times that guides search for earliness/tardiness and allocation costs.
- **CP-SAT 9.15?** Partly. `scheduling_precedences_lns` keeps a random set of the current solution's precedences, which is POS-like, but each fragment is still the whole model (docs/CPSAT_FEATURE_SURVEY.md section 3.2). No adaptive per-neighbourhood learning beyond difficulty adjustment.
- **Fit.** POS relaxation generalises our "everything after the window moves as one block". Temporal linear relaxation has nothing to guide with a pure makespan objective.
- **Reported effect against CP-SAT.** Lan and Berkhout 2025 (OR-Tools 9.11 against CP Optimizer 22.1.1, 900 s): RCPSP deviation from best-known 0.93 % against 0.46 %, and optimality gap 3.46 % against 4.26 %; multi-mode RCPSP 0.18 % against 0.27 %, and gap 0.94 % against 6.65 %; multi-project RCPSP gap 14.91 % against 67.36 %. So CP Optimizer finds slightly better schedules on large single-mode instances, and CP-SAT has the better bounds and the better multi-mode results.
- **Benefit.** (i) moderate, as a neighbourhood design for our external LNS (section 5.4); (ii) none.
- **Sources.** Laborie, Rogerie, Shaw, Vilím, "IBM ILOG CP Optimizer for scheduling", Constraints 23, 2018, https://dblp.uni-trier.de/rec/journals/constraints/LaborieRSV18.html (abstract only). Laborie, Rogerie, "Temporal linear relaxation in IBM ILOG CP Optimizer", J. Scheduling 19, 2016 (snippet). Lan, Berkhout, "PyJobShop: Solving scheduling problems with constraint programming in Python", arXiv 2502.13483, https://arxiv.org/html/2502.13483v1 (full text). Laborie and Godard 2007: not found.

### 1.3 OptalCP

- **What.** Vilím's commercial CP scheduling solver: LNS plus FDS workers, optional intervals with `alternative`, cumulative through `pulse`, reservoir through `stepAtStart`.
- **What it has that CP-SAT does not (vendor docs).** `usePrecedenceEnergy` ("precedence energy algorithm for intervals with multiple predecessors or successors on the same resource", off by default); `cumulPropagationLevel` up to 4 with "max-energy reasoning"; `FDSDual` (FDS on objective bounds); a `Large` preset that runs every worker as LNS; warm start from an external solution.
- **Licence and API.** Proprietary. A free "Preview" returns objective values but not solutions; a free academic licence is requested by email. Python, TypeScript and C# APIs.
- **Reported effect.** Vendor benchmark against CP Optimizer only: median 5.4x speed-up on 136 job-shop instances both solve. **No RCPSP comparison against CP-SAT was found.** One non-peer-reviewed 2026 job-shop preprint (snippet only) ranks OptalCP most robust.
- **Fit.** Our model maps directly (unit intervals, alternatives for modes, pulse for memory). Whether pulse height may depend on mode was not confirmed.
- **Benefit.** (i) plausible at 12K ops; (ii) a few cycles at most from precedence energy [reasoning: per-group ancestor counts are far below capacity, section 2.2].
- **Route.** Other solver; export one instance through the Python API.
- **Sources.** https://optalcp.com/ , https://optalcp.com/docs/benchmarks , https://optalcp.com/docs/api , https://github.com/ScheduleOpt/optalcp-benchmarks (vendor docs).

### 1.4 Chuffed and the lazy-clause-generation RCPSP line

- **What gave the proofs.** Timetable cumulative with explanations; redundant pairwise disjunctions for tasks with `r_i + r_j > capacity`; objective written as `S_i + tail_i <= makespan`; VSIDS with restarts after a 500-node "hot start"; destructive lower bounding. Timetable edge finding with explanations improved bounds on 264 of 432 open PSPLIB instances in 10 minutes.
- **CP-SAT 9.15?** Yes, all of it or stronger; Perron's slides say CP-SAT "is inspired by Chuffed".
- **Fit.** The pairwise-disjunction step needs `r_i + r_j > capacity`; with unit demands only the single flow slot qualifies. Two cheap carry-overs: the `S_i + tail_i <= makespan` rows, and destructive bounding on windows (satisfaction at makespan = optimum - 1).
- **Benefit.** (i) none; (ii) none at whole-program scale.
- **Sources.** Schutt, Feydy, Stuckey, Wallace, "Solving RCPSP/max by lazy clause generation", J. Scheduling 16, 2013, https://arxiv.org/pdf/1009.0347 (partial, sections 2.3 to 5.3). Schutt, Feydy, Stuckey, "Explaining Time-Table-Edge-Finding Propagation for the Cumulative Resource Constraint", CPAIOR 2013, https://ar5iv.labs.arxiv.org/html/1208.3015 (full text via summary). "Explaining the cumulative propagator", Constraints 16, 2011 (abstract only).

### 1.5 Pumpkin and Huub

- Pumpkin's cumulative is timetable only (source docs); its distinguishing feature is proof logging. Huub: timetable cumulative, fuller disjunctive (snippet).
- Sidorov, Marijnissen, Demirović, "Unite and Lead: Finding Disjunctive Cliques for Scheduling Problems", CP 2025, https://drops.dagstuhl.de/storage/00lipics/lipics-vol340-cp2025/html/LIPIcs.CP.2025.35/LIPIcs.CP.2025.35.html (full text via summary): against OR-Tools 9.11, 4 new upper and 4 new lower bounds; gains only "when resource contention is high".
- Sidorov, "On inferring cumulative constraints", arXiv 2602.15635, https://arxiv.org/abs/2602.15635 (abstract only): derives new cumulatives as linear combinations of existing ones; 25 new RCPSP lower bounds.
- **Transferable idea [reasoning].** A mode-free combined resource: per cycle `alu + 8*valu <= 60`, in which a shiftable vector op has demand 8 in either mode. Its whole-horizon energy is our joint floor (vector-equivalents / 7.5), but as a cumulative it lets interval reasoning ignore mode literals. Check first whether the energetic rows of the modelling review already contain it.

---

## 2. Propagation and bounds for unit-duration, high-capacity cumulatives

### 2.1 Everything energy-based collapses to Hall intervals

- **Fact.** A cumulative with durations 1, demands 1 and capacity C is the upper-bound half of a global cardinality constraint. Quimper et al. (Lemma 1): it is satisfiable iff for every value set S, the number of variables with domain inside S is at most the capacity of S. Bounds consistency is polynomial and complete for one resource.
- **Consequence [reasoning].** Substituting p = 1 into the minimum-intersection-energy formula gives "h if the window lies inside the interval, else 0". So:

| Rule | For unit, demand-1 tasks |
|---|---|
| Timetabling | Only fixed tasks count; strictly weaker than Hall |
| Overload check | Exactly the Hall failure test |
| Edge finding, extended edge finding, timetable edge finding | Exactly Hall pruning |
| Energetic reasoning (Baptiste, Tesch, Ouellet-Quimper, Carlier et al.) | Same; the newer algorithms are faster, not stronger |
| Horizontally elastic rules (Gingras-Quimper, Kameugne et al.) | Same |

- **CP-SAT 9.15.** Timetabling always. Overload checker optional (it raises the capacity minimum, not task bounds). Timetable edge finding optional and **created only when the cumulative has at most `max_num_intervals_for_timetable_edge_finding` tasks, default 100** (checked by me, `cumulative.cc` at v9.15). No plain edge finding, not-first/not-last or energetic reasoning for 1-D cumulative. `AllDifferentBoundsPropagator` does Hall intervals for `all_different` only; there is no unit-duration special case and no GCC.
- **Exception.** A demand-8 task on the 12-wide ALU: elastic energy lets its 8 units spread, while at most one such task fits per cycle. A dual-feasible function sees this; CP-SAT has `use_conservative_scale_overload_checker` for it. It matters only if a shifted op is one demand-8 interval.
- **Benefit of implementing any of these.** (i) none; (ii) none. This closes the "stronger propagator" direction for the op-class resources.
- **Sources.** Quimper, van Beek, López-Ortiz, Golynski, Sadjad, "An Efficient Bounds Consistency Algorithm for the Global Cardinality Constraint", CP 2003, https://cs.uwaterloo.ca/~vanbeek/Publications/cp03.pdf (partial, pp. 1-6). Ouellet, Quimper, "A O(n log^2 n) Checker and O(n^2 log n) Filtering Algorithm for the Energetic Reasoning", CPAIOR 2018, https://cquimper.github.io/publications/energetic.pdf (partial, pp. 1-4). Gingras, Quimper, IJCAI 2016, https://ijcai.org/Abstract/16/440 (abstract). Kameugne et al., CP 2023, https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.CP.2023.20 (abstract only). Tesch CP 2018; Carlier et al. EJOR 2020; Gay, Hartert, Schaus CP 2015 (snippets only). Vilím CP 2009 and Ouellet-Quimper CP 2013: not fetched.

### 2.2 Precedence combined with capacity

- **What.** Bessiere, Narodytska, Quimper, Walsh: all-different with precedences; bounds consistency is polynomial, and strictly stronger than the decomposition (two predecessors in {1,2,3} force the successor past 2). The capacity-C form is "if op j has k ancestors of type r, all with head >= a, then start_j >= a + ceil(k / C_r)".
- **Why it can beat the LP.** The integrality-gap instance of the time-indexed LP (Levey and Rothvoss: layers of m+1 unit jobs on m machines, gap 2 - 2/(m+1)) is solved exactly by this rule.
- **CP-SAT 9.15?** Yes, one-directional: `CumulativeIsAfterSubsetConstraint` under `use_hard_precedences_in_cumulative`, **which creates constraints only when `exploit_all_precedences` is also set** (checked by me, `cumulative.cc` lines 194 to 252). The ablation used both together. OptalCP has `usePrecedenceEnergy`. No "GCC with precedences" global constraint was found in the literature.
- **Fit [reasoning].** Little. A group has about 375 ops over a critical path of 264, about 1.4 ops per level against 20+ slots per cycle. An op's ancestors lie in its own group, so ancestor counts are far below capacity. Our bottleneck exists only in the sum over 32 groups, which per-op ancestor counting cannot see.
- **Benefit.** (i) none; (ii) a few cycles at most.
- **Sources.** Bessiere et al., "The AllDifferent Constraint with Precedences", CPAIOR 2011, https://ar5iv.arxiv.org/html/1103.3745 (full text). Levey, Rothvoss, STOC 2016, https://ar5iv.arxiv.org/html/1509.07808 (full text). Bofill, Coll, Suy, Villaret "energy precedences" (Coll's thesis, https://core.ac.uk/download/87633485.pdf): table of contents only, definition not read.

### 2.3 What CP-SAT's LP sees

- Static row per cumulative: sum of energies <= capacity x (makespan - min start), with presence literals. That is the capacity floor minimised over modes, in LP form, and it is the bound we observe.
- Cut generators at `linearization_level > 1`: timetable, completion-time, precedence; energy cuts only when some size or demand is variable.
- Perron's slides: multi-mode energy cuts use the mode Booleans only when the recipe structure "is discovered by inspecting the model". Whether our 1 + 8 optional-interval encoding of VALU shift is detected is a question for the ortools-internals agent.

---

## 3. Mathematical-programming bounds

### 3.1 Time-indexed formulations

- **What.** Pulse or step variables per (op, cycle). Aggregated precedence (one row per edge) is weak; disaggregated precedence (one row per edge and cycle, Christofides et al. 1987) is stronger. With disaggregated rows the precedence-only polytope is integral for a single mode, and solvable as a minimum cut (Möhring et al.).
- **Numbers (Kolter, Grunow, Kolisch 2025, PSPLIB j30).** Aggregated LP on average 7.50 % weaker than disaggregated; disaggregated LP plus solver cuts on average 6.81 % stronger than the column-generation master.
- **CP-SAT 9.15?** No time-indexed LP; its LP is over start variables.
- **Fit.** Per-op size at 16 rounds: 12K ops x about 1,050 window cycles = 12.6M variables. That is the reason it was dismissed. Section 3.2 removes the factor 32.
- **Does it capture what the floor misses [reasoning]?** Yes for typed pipeline fill at start-up and drain at the tail (a class cannot complete faster than its predecessors while capacity is charged), fractionally for modes, and weakly for memory. It does not capture per-cycle integrality.
- **Sources.** Kolter, Grunow, Kolisch, "On Branch-and-Price for Project Scheduling", arXiv 2501.04563, https://arxiv.org/pdf/2501.04563 (full text, pp. 1-24). Artigues, "On the strength of time-indexed formulations for the resource-constrained project scheduling problem", OR Letters 45, 2017, https://hal-univ-tlse3.archives-ouvertes.fr/hal-01461447 (snippet only; content as reported by Kolter et al.).

### 3.2 The counting (high-multiplicity) time-indexed model [reasoning; not found in the literature]

Let a *class* be one op position within a group, and y[c,t] in {0..32} the number of groups that have completed class c by cycle t.

- Precedence c' -> c, latency 1: `y[c,t] <= y[c',t-1]`.
- Capacity of type r: `sum over classes of type r of (y[c,t] - y[c,t-1]) <= C_r`.
- Memory: `L[v,t] >= y[def(v),t] - y[c,t]` for each consumer c of value v; `sum_v w_v L[v,t] <= 1536`.
- Shared op feeding every group: `y[c,t] <= 32 * y[shared,t-1]`.
- Modes: split y by mode; capacity rows take the mode's demand.

Claims:
1. **Valid relaxation**: sum the per-group step constraints.
2. **Same LP bound as the per-op disaggregated LP**: average any LP solution over group permutations.
3. **Exact as an integer program when latencies do not depend on mode**: let group g complete class c at `min{t : y[c,t] >= g}`. Precedence holds, capacity is unchanged, and because the assignment is nested the number of live copies of v is exactly `y[def] - min_c y[c]`.
4. **Not exact as stated when a mode changes an op's duration** (the 2-cycle select, lanes spread over cycles). Counter-example sketch: copy A starts at 1 and is available at 4, copy B starts at 2 and is available at 3. The second-available copy has start rank 1, and the count rows only guarantee one predecessor by cycle 1. It stays a valid bound. This needs a Z3 or checker treatment before the integer model is trusted for such classes.

Size: 2 rounds / batch 256 is about 30 to 45 classes x 200 cycles, under 10K integers. 16 rounds is about 375 classes x 1,050 cycles, about 0.4M variables and 1M rows: a feasible LP, a heavy MILP.

- **Benefit.** (ii) The one candidate for a bound above the floor, and the instrument that splits the gap into ramp, mode integrality and memory. (i) On mid-size instances the integer model may simply solve, since the 32! symmetry is gone. If emission plans differ between groups, use one count per group type.
- **Route.** Model reformulation, outside CP-SAT first (HiGHS LP), then MILP or a CP-SAT model over y.
- **Corollary for the existing model.** Claim 3 gives a dominance far stronger than one lex row: for every fixed-duration class c, `start[c,1] <= start[c,2] <= ... <= start[c,32]`. Prove it for the mode cases before posting (section 4.4).

### 3.3 Lagrangian relaxation, column generation, destructive bounds

- **Möhring, Schulz, Stork, Uetz 2003.** Dualise the resource rows of the time-indexed IP; the rest is a minimum cut. Same value as the disaggregated LP for a single mode. Here it would decompose into 32 identical group-sized min-cuts, memory can be priced as start-time-dependent costs on a "kill" event [reasoning], and the prices give a list-scheduling priority. https://dspace.mit.edu/handle/1721.1/693 (abstract only).
- **Dantzig-Wolfe over group patterns.** Kolter et al.: master LP equals the disaggregated LP for single mode; 0.42 % stronger on average for multi-mode (MMLIB50) at 26 times the cost; 77 times slower for single mode. Do not build it for the bound.
- **Brucker and Knust 2000; Baptiste and Demassey 2004.** Destructive bounds from column generation over feasible subsets (antichains) of the preemptive relaxation. With unit durations this is a per-cycle configuration LP; it sees per-cycle packing integrality (8 + 4 on a 12-wide ALU) and is blind to ordering. Not read in full (secondary source and snippet).
- **Event-based MILP (Koné et al. 2011, snippet).** With unit durations events are cycles; nothing is saved.
- **CP-SAT 9.15?** None of these.

### 3.4 Unit-time theory

- Time-indexed LP has integrality gap at least 2 - 2/(m+1) for P|prec,p=1|Cmax (Levey and Rothvoss), from layers slightly wider than capacity. LP hierarchies close it in theory only (Levey-Rothvoss; Garg, arXiv 1708.04369, abstract only).
- Polynomial cases (in/out-trees, interval orders, two machines) and typed task systems (Jaffe 1980): [memory]. Typed task systems are reported NP-complete even for unit-length fork graphs (snippet).
- Our regime is the opposite of the gap instance: narrow groups, load from 32 parallel copies. A fluid relaxation is nearly realisable when 10 % of every group can be replaced by 3 whole groups [reasoning].

---

## 4. Decomposition and structure exploitation

### 4.1 High-multiplicity scheduling, n-fold IP

- Hochbaum and Shamir 1991; Clifford and Posner 2001; Brauner et al. 2005; Knop and Koutecký 2018 (all abstract only). None handles precedence inside a job type, and n-fold algorithms are parameterised by brick size, which here is a 375-op schedule. Useful as a framing only; section 3.2 is the usable form.

### 4.2 Fluid bounds, periodic and modulo scheduling

- **Zheng, Qu, Cai, Mao, "High-Multiplicity Flexible Job Shops: From Exact Recurrent Fluid Attainment to Structure-Guided Finite-Horizon Scheduling", arXiv 2608.19711, 2026** (skimmed by a sub-agent; single-source, see verification note). Claims: the optimal makespan stays within an additive constant of the fluid bound as multiplicity grows; an algorithm (TCTR) that solves a compact job-type template with CP-SAT and replays it; mean gap to the fluid bound 1.54 % on 676 instances against 5.32 % for a job-indexed adaptive LNS. Disjunctive machines, chain jobs, no memory resource.
- **Cyclic identical-job shops** (Roundy 1992; Lee and Posner 1997; Hanen and Munier 1995; all abstract only) optimise cycle time for an infinite stream, not finite makespan.
- **Modulo scheduling ILPs.** Oppermann et al., TRETS 2019, https://www.esa.informatik.tu-darmstadt.de/assets/publications/materials/2019/2019_TRETS_JO.pdf (full text): 188 loops, median 54 ops, median 4 resource-limited ops; pairwise overlap binaries that would be quadratic for us. The Eichenberger-Davidson counting form is the right shape.
- **Which period [reasoning].** One group every p cycles gives 31p + 264 with p >= 39.7, at least 1,495: worse than the incumbent. The round is the natural period: about 80 cycles of capacity against 16.5 cycles of critical path per round, about 750 ops, the size our windows already solve.
- **CP-SAT 9.15?** No periodic machinery; model reformulation plus outer loop.
- **Benefit.** (i) a structurally different incumbent at 12K ops; (ii) none directly, but "is the floor-rate period feasible for a steady-state round?" tells where the gap lives.

### 4.3 Logic-based Benders

- Hooker's 2019 survey, https://ar5iv.labs.arxiv.org/html/1910.11944 (full text): works when the subproblem decouples into facilities; analytical cuts are "weak for cumulative scheduling"; the master needs a subproblem relaxation, which for cumulative is the energy inequality.
- **Fit.** Master = modes, subproblem = one coupled schedule. No decoupling, the master relaxation is the capacity floor we already have, and a makespan nogood excludes one mode vector among thousands of equivalents. Not recommended.

### 4.4 Symmetry and dominance

- **CP-SAT 9.15 (headers read).** Automatic detection (`symmetry_level`), Boolean orbit and orbitope fixing in presolve, and `SymmetryPropagator` (Devriendt et al., ICTAI 2012 symmetry propagation, on Boolean literals). `use_symmetry_in_lp` folds orbits in the LP (default off). Not present: SBDS-1UIP (Chu, Stuckey, de la Banda, Mears, IJCAI 2011, abstract only), symmetric explanation learning (Devriendt et al., SAT 2017, abstract only), lightweight dynamic symmetry breaking (Mears et al. 2014, abstract only).
- **Fit.** Group-swap generators have support of about 375 start variables plus mode literals; the header itself recommends small supports. Symmetry handling shrinks trees; it does not lift a relaxation bound at the floor.
- **Stronger than lex [reasoning].** Per-class nested ordering from section 3.2, and a round-boundary suffix exchange (if group A finishes round r before B but starts round r+1 after it, swap their suffixes). Both are conditional dominances and must be proved first.
- **Warning.** Chu and Stuckey, CP 2012, https://people.eng.unimelb.edu.au/pstuckey/papers/dominance.pdf (full text): on PSPLIB J60 in Chuffed, learning alone 4.07 s, learning plus static left-shift dominance 32.84 s. Do not post left-shift rows.
- **State dominance.** Cohen et al., arXiv 2611.14426 (abstract only): graph-based A* with duplicate-state detection proves 480 of 480 RCPSP instances against 445 for best-first branch-and-bound with the same bound. CP-SAT has no state memoisation. For us a state is a multiset of per-group progress, which removes group symmetry exactly; the counting model is the same idea in MILP form.

---

## 5. Newer exact paradigms and learned search

### 5.1 Domain-independent dynamic programming; assembly line balancing

- Kuroiwa and Beck, arXiv 2401.13883, https://arxiv.org/pdf/2401.13883v4 (full text of experiments): on SALBP-1 (which with unit times is P|prec,p=1|Cmax), complete anytime beam search solves 1,802 of 2,100 against 1,584 for CP Optimizer and 1,431 for Gurobi. CP-SAT is not compared.
- Branch-bound-and-remember (Sewell and Jacobson 2012, abstract only) and bin packing with precedences (Dell'Amico et al. 2012; Pereira 2016; Kramer et al. 2017; abstracts only). No version with several typed capacities, modes or storage was found.
- **Fit.** The only cheap dual bound is the capacity floor, so proofs do not terminate. The beam search with duplicate detection over multiset states is a possible second incumbent generator.
- **Benefit.** (i) moderate; (ii) none. **Route.** didp-rs, separate from CP-SAT.

### 5.2 Decision diagrams

- Peel-and-bound (Rudich et al., CP 2022, abstract only), ddo. Published successes are sequencing problems. For 20+ ops per cycle, merging states leaves the capacity floor. Not applicable.

### 5.3 Core-guided search; SAT and SMT encodings

- Gange, Berg, Demirović, Stuckey, CPAIOR 2020, https://cs.helsinki.fi/u/jezberg/papers/CPAIOR2020_Gange_Berg_Demirovic_Stuckey.pdf (full text). CP-SAT has it (`core`, `optimize_with_core`). With a single-variable makespan objective it reduces to bottom-up search on makespan. No benefit.
- Bofill, Coll, Suy, Villaret, IJCAI 2017, https://ijcai.org/proceedings/2017/78 (full text): ops on one precedence path are mutually exclusive, which shrinks pseudo-Boolean encodings (281K to 45K Booleans on MMLIB50). With unit durations each (cycle, class) row is a plain cardinality row over time-indexed Booleans; usable in windows, not for 12K x 1,300.

### 5.4 LNS design; learned and adaptive LNS

- **Learned LNS for MIP** (Song et al. NeurIPS 2020; Sonnerat et al. 2021; Wu et al. NeurIPS 2021; Huang et al. ICML 2023; abstracts only) train on a distribution of instances. We have one family with strong structure; not recommended.
- **Adaptive selection.** Hendel, Math. Prog. Computation 2022 (abstract only): bandit over eight LNS heuristics in SCIP. Cheap to copy.
- **Designs the scheduling literature found effective.** Partial-order-schedule relaxation (Godard, Laborie, Nuijten, ICAPS 2005, abstract only); iterative flattening (Cesta, Oddi, Smith, AAAI 2000, abstract only); forward-backward improvement or double justification (Valls, Ballestín, Quintanilla, EJOR 2005, abstract only: added to 22 algorithms on j120, it "pays" generally at no extra time).
- **CP-SAT 9.15?** Time-window, resource-window, random-interval and random-precedence neighbourhoods on whole-model fragments. No mode-centred neighbourhood, no justification pass, no bandit over user neighbourhoods.
- **Fit [reasoning].** Window gains come from mode re-decisions, so define neighbourhoods on modes: (a) windows with the frozen part as a partial order, (b) free k of 32 groups over the whole horizon, (c) free all modes with every start confined to +/- 2 cycles, (d) double justification between rounds, (e) bandit selection by improvement per deterministic second.
- **Benefit.** (i) the highest of anything in sections 4 and 5; (ii) none. **Route.** Our external LNS loop.

---

## 6. Compiler and hardware scheduling since 2015

### 6.1 SDC + SAT (exact HLS scheduling)

- Dai, Liu, Zhang, FPGA 2018, https://www.csl.cornell.edu/~zhiruz/pdfs/sds-fpga2018.pdf (full text); Dai, Zhang, DAC 2019, https://www.csl.cornell.edu/~zhiruz/pdfs/sdsp-dac2019.pdf (full text). SAT chooses resource orderings; an incremental difference-constraint graph returns negative cycles as learned clauses; binary search on latency. Over 100x faster than ILP on 44 to 857 ops with 1 to 6 units per resource; typically under 10 % of ops are in contention.
- **CP-SAT 9.15?** The difference-logic half, yes (`PrecedencesPropagator` with explained cycles, header read). The resource half is a binding-plus-pairwise-ordering encoding, weaker than a cumulative at capacity 12. Not subsumed: lazy introduction of resource constraints only where an unconstrained schedule violates them.
- **Fit.** Our contended fraction is near 100 %. The transferable part is a lazy memory constraint: solve without the sixth cumulative, add it only where violated. (i) unknown, possibly large if that cumulative is what makes the model unsearchable; (ii) none.

### 6.2 Graph reduction, modulo scheduling, XLS, memory-aware ordering

- Oppermann et al., FPL 2018 (full text): collapse non-resource ops into longest-path edges; 4.37x on a time-indexed ILP. For us every op is resource-limited. Small benefit.
- Sittel, Fiege, Kumm, Zipf, "Isomorphic Subgraph-based Problem Reduction for Resource Minimal Modulo Scheduling", ReConFig 2019, DOI 10.1109/ReConFig48160.2019.8994768 (**abstract only**; mechanism not read): about 5x average, up to 102x. Objective is resource minimisation at fixed period, not makespan. The paper closest to our structure; worth reading with IEEE access.
- Google XLS (https://google.github.io/xls/scheduling/, full page) and ISDC (arXiv 2401.12343, full text): SDC without resource constraints. Not applicable.
- Memory-aware ordering: Liberis and Lane 2019; Ahn et al., MLSys 2020 (both full text): DP over frontiers, up to about 100 nodes, sequential order only. Steiner et al., OLLA / MODeL (ICML 2023, partial): time-indexed ILP with created/preserved Booleans per (tensor, step), restricted to ASAP-ALAP spans; up to 2,116 operators in seconds on chain-like graphs. Transferable: per-(value, cycle) live Booleans with one linear memory row per cycle, as an alternative to variable-length intervals inside windows.
- Castañeda Lozano and Schulte, ACM CSUR 2019, https://arxiv.org/html/1409.7628v3 (partial): optimal local scheduling to 2,600 instructions for issue widths 1 to 6, where precedence dominates. Not our regime.
- COGENT is a tensor-contraction code generator: not relevant. "Rijkeboer": not found. CGRA mappers (SAT-MapIt, DATE 2023, full text): tiny kernels, placement dominated.

### 6.3 Public write-ups of this challenge (searched by its public name only)

No write-up using CP-SAT, ILP, MiniZinc, annealing or a genetic algorithm over schedules was found.

| Source | Cycles | Scheduling method |
|---|---|---|
| github.com/Trav55555/original_performance_takehome (README and docs; single-source) | 975, on a floor of 967 for its ops | List scheduling backward then forward, three priority orders, about 2,000 emission candidates scored; Z3 used once for a 527-variable repair; a whole-graph exact query hit a memory cap after 16 h 52 min |
| austinwallace.ca/kernel (full page) | 1,137 | Priority list scheduler plus beam search (width 2, 25-cycle horizon); about 43 cycles above its floor |
| sambhavg.github.io/blog/performance (full page) | 1,218 | Greedy ready-list scheduling |
| github.com/0xquinto/vliw-kernel-optimization (README) | 1,285 | Multi-trial scheduler; reports 11 cycles of scheduler noise |
| paradigm.xyz/puzzles/anthropic-challenge (full page) | leaderboard 861 to 865 | Not described |
| news.ycombinator.com/item?id=46700594 (thread) | 1,023 to 1,137 | Manual or agent-driven |

**Reading [reasoning].** A searched list scheduler with justification is reported within 0.8 % of its floor on this DAG family, and exact search was useful only as a repair of about 500 variables. The best kernels are designed so that all engine floors nearly coincide; the lever is the emission and mode mix.

### 6.4 Metaheuristic warm starts

- Wang, Gong, DeRenzi, Kastner, IEEE TCAD 2007, https://cseweb.ucsd.edu/~kastner/papers/tcad07-aco_operation_scheduling.pdf (partial): learn a priority list, evaluate with a list scheduler; up to 22 % better than plain list scheduling on graphs of at most 354 nodes. The mechanism transfers; the percentages do not.

---

## 7. What the CP-SAT maintainers say, and what is coming

- **Perron, "OR-Tools and CP-SAT", guest lecture slides, UPenn CIS 1921, spring 2025**, https://cis.upenn.edu/~cis1921/files25s/guestlecture.pdf (full, 56 slides):
  - "Better than commercial solvers on small to medium scheduling problems, missing heuristics for large instances."
  - What's next: "Improved scheduling (branching heuristics, LNS, propagation, modeling)", "Reduce encoding size".
  - "Stronger Scheduling Propagation ... Pigeon hole problem", with mitigation ideas (gcd, subset-sum, super-additive functions) posed as open.
  - Energy cuts for recipes: "This structure is discovered by inspecting the model."
- **Perron, Didier, Gay, "The CP-SAT-LP Solver", CP 2023**, https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.CP.2023.3 (full; a 2-page abstract with no algorithms). No paper on CP-SAT scheduling cuts was found.
- **Release notes** (https://github.com/google/or-tools/releases, through a summariser). 9.12: improved cumulative propagation and cuts, hints preserved through presolve. 9.13: precedences detected and used in presolve, propagation and cuts. **v10.0-rc.1** (pre-release, 23 November): presolve splits cumulatives that have precedences; new `ls_scheduling` subsolver (present in `cp_model_solver.cc` of the rc tree, checked by me); new shaving and probing subsolvers; shared-tree rework; LNS model-copy rework with deterministic presolve limits and skipping of redundant scheduling constraints; a determinism fix for interleaved LNS. No FDS, no POS-style LNS, no unit-duration specialisation, no symmetry items.
- **GitHub issues and discussions** on FDS, unit-duration cumulative or large-instance LNS: nothing relevant found.
- **Independent comparisons.** PyJobShop (section 1.2) is the only RCPSP comparison read in full. Naderi, Ruiz, Roshanaei (INFORMS JoC 2023, abstract only) compares CP Optimizer with MILP solvers, not with CP-SAT. Da Col and Teppan (Operations Research Perspectives 2022, abstract only): job shop up to one million operations; tables not accessible. MiniZinc Challenge 2025: CP-SAT gold in all categories; no RCPSP problem that year.

---

## 8. Not recommended, with the reason

| Idea | Reason |
|---|---|
| Stronger cumulative propagators (edge finding, energetic reasoning, horizontally elastic) | Collapse to Hall intervals for unit tasks (2.1) |
| Failure-directed search, or porting to Chuffed / Pumpkin / Huub | Built for scarce-resource regimes; no signal with 18 slots per cycle (1.1, 1.4, 1.5) |
| Dantzig-Wolfe / branch-and-price | Same bound as the compact LP, 26 to 77 times slower (3.3) |
| Logic-based Benders on modes | No decoupling; master relaxation is the floor (4.3) |
| Decision diagrams, core-guided search | Bottom out in the capacity floor (5.2, 5.3) |
| Learned LNS policies | One instance family; hand-built neighbourhoods plus a bandit suffice (5.4) |
| SDC + SAT, dependence-graph preprocessing | Subsumed by CP-SAT, or aimed at under 10 % contended ops (6.1, 6.2) |
| Static left-shift dominance rows | Measured to slow a learning solver 8x on RCPSP (4.4) |
| Group-periodic (one group per period) schedules | At least 1,495 cycles by arithmetic (4.2) |

---

## 9. Ranked shortlist

Ranked by expected value per unit of effort. "Queue" marks an experiment that is a heavy job for this machine.

1. **Counting (high-multiplicity) time-indexed model (3.2).** The only bound candidate above the floor; removes group symmetry; may solve mid-size instances outright. *Experiment:* build the aggregated LP for 2 rounds / batch 256 (under 10K variables) in HiGHS, then with memory rows, then as a MILP. Compare with the floor and the best schedule. Prove claim 3 for mode-dependent durations with Z3 on a 2-group toy first.
2. **Backward/forward justification and priority search in the greedy scheduler (5.4, 6.3).** Reported at 0.8 % above the floor where we are at 3.4 % or more. *Experiment:* double-justify an existing 16-round schedule JSON and count cycles saved; run the lift-and-reschedule tool on the 975 kernel to compare our scheduler with its author's on identical ops.
3. **Mode-centred and partial-order LNS neighbourhoods with bandit selection (1.2, 5.4).** Extends the one mechanism that already improves incumbents. *Experiment:* at one mid-size instance and equal deterministic time, compare the 25-cycle window with "free 2 of 32 groups" and "all modes free, starts +/- 2".
4. **OR-Tools v10.0-rc.1 unchanged (7).** LNS rework, cumulative splitting and new subsolvers target our symptoms at no modelling cost. *Experiment:* rerun two ablation rows after a proto sync of the `cp_sat` fork. Queue.
5. **Per-class nested ordering in place of whole-group lex (3.2, 4.4).** *Experiment:* prove with Z3, then measure window-proof times on a 2-round instance with the existing ablation harness.
6. **Round-periodic template probe (4.2).** *Experiment:* three unrolled copies of one steady-state round with equal offsets; test feasibility at the floor-rate period and at +1, +2, with the folded memory limit.
7. **Lazy memory constraint, and time-indexed liveness rows in windows (6.1, 6.2).** *Experiment:* solve 2 to 4 rounds without the sixth cumulative and measure peak live words; add rows only for offending ranges.
8. **OptalCP trial (1.3).** *Experiment:* export one 1,500-op instance; the free preview returns objective values and the propagation-only lower bound with `usePrecedenceEnergy`.
9. **Check that CP-SAT's recipe-energy detection fires on our mode encoding; add the mode-free `alu + 8*valu <= 60` resource (1.5, 2.3).** *Experiment:* one 25-cycle window, three encodings, compare root bound and time to optimal.
10. **Lagrangian or LP-dual prices as a greedy priority and window selector (3.3).** *Experiment:* take the duals of the LP from item 1; order ops by priced start time; feed to the greedy.
