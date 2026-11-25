# Part 2 - Propagation for unit-duration cumulatives, and mathematical-programming lower bounds

Scope: Topic A (what cumulative propagators can do when every task has duration 1) and Topic B (LP / Lagrangian / column-generation bounds that could beat the capacity floor). Reading and reasoning only; nothing was run.

Evidence labels: **[full text]**, **[abstract only]**, **[snippet]** (search-result snippet only), **[source code]**, **[from memory, unverified]**. Paragraphs marked **Reasoning** are my own derivations, not claims from a paper.

---

## Summary of conclusions

1. For unit-duration, demand-1 tasks, a cumulative of capacity C is exactly the upper-bound half of a global cardinality constraint (GCC). Its satisfiability is characterised completely by the Hall condition. Overload checking, edge finding, timetable edge finding, energetic reasoning and the horizontally elastic variants all collapse to that Hall-interval test (or its pruning form). None can prune beyond it on a single resource. Timetabling is strictly weaker.
2. With our wide windows (slack about 1,000 cycles), the only Hall intervals that bind are whole-horizon ones, which is the capacity floor. This explains why CP-SAT's bound sits at the floor plus 1 or 2 whichever energy flags are set.
3. What can add something is reasoning that combines precedence with capacity. CP-SAT 9.15 has one such propagator (`CumulativeIsAfterSubsetConstraint`). Because one group is far narrower than the capacity, per-group ancestor counting is dominated by the critical path, so I expect little from it here.
4. Because the 32 groups are identical and coupled only through capacities and memory, a "counting" time-indexed formulation (one integer in 0..32 per op class per cycle) is, by my argument below, an exact reformulation for precedence, capacity and memory. It is 32 times smaller, has no group symmetry, and its LP relaxation equals the disaggregated (DDT) LP bound. This is my top recommendation.
5. Published evidence says column generation / Dantzig-Wolfe is not worth it for this shape of problem; the compact disaggregated LP plus solver cuts is as strong and far faster.

---

## Topic A - Propagation for unit tasks / high capacity

### A1. Unit-duration cumulative as a GCC

**What it is.** Quimper, van Beek, Lopez-Ortiz, Golynski, Sadjad, "An Efficient Bounds Consistency Algorithm for the Global Cardinality Constraint", CP 2003, https://cs.uwaterloo.ca/~vanbeek/Publications/cp03.pdf **[full text, pp. 1-6]**. A GCC splits into a lower-bound constraint and an upper-bound constraint (ubc: value v used at most u_v times). They generalise Hall intervals: C(S) is the number of variables whose domain is contained in S and the capacity of S is the sum of u_v over S. Their Lemma 1: "A ubc is satisfiable if and only if for any set S, C(S) <= capacity(S)" (proved by duplicating each value u_v times and reducing to all-different). Bounds consistency runs in O(t + n), t being the time to sort the bounds.

**Reasoning - mapping.** A cumulative with all durations 1, all demands 1 and capacity C is a ubc with u_t = C for every cycle t. With interval domains only interval sets matter, so the constraint is feasible iff for every [a,b): #(tasks with window inside [a,b)) <= C(b-a). Bounds consistency is therefore polynomial and complete for a single resource: no other single-resource rule can tighten bounds further.

**Multi-resource.** With modes fixed, each op uses one type, so the five typed cumulatives are five independent GCCs. With modes free, an op's domain is a union of intervals on different resources and the shifted mode consumes 8 units, so it is a weighted b-matching rather than a GCC; its LP relaxation is the "capacity floor minimised over modes" we already have.

### A2. Which propagators collapse for unit tasks (Reasoning)

Ouellet and Quimper, "A O(n log^2 n) Checker and O(n^2 log n) Filtering Algorithm for the Energetic Reasoning", CPAIOR 2018, https://cquimper.github.io/publications/energetic.pdf **[full text, pp. 1-4]** state that energetic reasoning "dominates the overload check, the time-tabling, the edge-finder, and the time-tabling-edge-finder" and is incomparable with not-first/not-last. They define the minimum intersection energy LSRS(i,l,u) = h_i * max(min(ect_i - l, u - lst_i, p_i, u - l), 0).

Substituting p_i = 1 and integer bounds: LSRS(i,l,u) = h_i if est_i >= l and lct_i <= u, else 0. A unit task has no partial intersection with an interval. Hence:

| Rule | What it becomes for unit, demand-1 tasks |
|---|---|
| Timetabling (mandatory parts) | A unit task has a mandatory part only when fixed. Reduces to "cycle t has C fixed tasks, so remove t from other tasks' bounds". Strictly weaker than Hall. |
| Overload check (fully elastic) | Exactly the Hall-interval failure test. |
| Edge finding / extended edge finding | Exactly Hall-interval pruning. If [a,b) holds C(b-a) tasks, the update est_i >= est + ceil(rest/c_i) evaluates to est_i >= b. |
| Timetable edge finding | Same as edge finding plus fixed tasks; fixed unit tasks are already counted by Hall. |
| Energetic reasoning (all variants: Baptiste et al., Derrien-Petit, Tesch, Ouellet-Quimper, Carlier et al.) | Check = Hall test; filtering = Hall pruning. The faster algorithms change cost, not strength. |
| Horizontally elastic overload check / edge finder | Differs from fully elastic only through the per-instant cap h_i, which is irrelevant when p_i = 1 and h_i = 1. Same as Hall. |
| Not-first / not-last | Incomparable in general, but GCC bounds consistency is already complete here. |

So for our five op-class resources every energy-style rule is GCC bounds consistency. With ASAP/ALAP windows about 1,000 cycles wide, the binding Hall intervals are [0,T) and intervals near the two ends (heads and tails), i.e. the Fernandez-Bussell-style bound. I did not obtain the Fernandez-Bussell formula from a page I read (bibliographic record and abstract only: https://bibsonomy.org/bibtex/1055349dd163559280862e4a6f40be35f **[abstract only]**); my statement that it is the interval-load bound over earliest/latest times is **[from memory, unverified]**.

**Exception: demand-8 unit tasks.** If a shifted vector op is modelled as one demand-8 task on the 12-wide scalar ALU, elastic energy lets the 8 units spread over cycles, while in reality at most one such task fits per cycle (8 + 4). A dual-feasible function (f(8) = 1, f(d <= 4) = 0, capacity 1) captures this. If instead the 8 lanes are 8 independent demand-1 ops, there is nothing to gain. I do not know which our model uses.

### A3. Papers on stronger cumulative filtering (regime check)

- Gingras and Quimper, "Generalizing the Edge-Finder Rule for the Cumulative Constraint", IJCAI 2016, https://ijcai.org/Abstract/16/440 **[abstract page]**. Horizontally elastic relaxation: O(n^2) overload check, O(kn^2) edge-finder. Reported: on BL (highly cumulative) runtime improved on 85%, 82%, 63% of instances for three search heuristics; on PSPLIB 75%. Adds nothing for unit tasks (A2).
- Kameugne, Fetgo Betmbe, Noulamo, Tayou Djamegni, "Horizontally Elastic Edge Finder Rule for Cumulative Constraint Based on Slack and Density", CP 2023, https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.CP.2023.20 **[abstract only]**. O(n^2); subsumes basic edge finding, incomparable with Gingras-Quimper and TTEF; no numbers in the abstract. The CPAIOR 2018 horizontally elastic not-first/not-last paper: **[snippet]** only.
- Tesch, "Improving Energetic Propagations for Cumulative Scheduling", CP 2018, ZIB report 18-29, https://opus4.kobv.de/opus4-zib/frontdoor/index/index/docId/6933 **[snippet of abstract]**: an energetic edge-finding rule generalising both energetic reasoning and edge finding, O(n^2 log n), relaxed O(n^2) version. Tesch CP 2016: **[snippet]** (title/URN only).
- Carlier, Pinson, Sahli, Jouglet, EJOR 286 (2020) 468-476, https://ideas.repec.org/a/eee/ejores/v286y2020i2p468-476.html **[snippet]**: energetic adjustments in O(n^2); a 2021 follow-up gives an O(n alpha(n) log n) checker (https://hal.archives-ouvertes.fr/hal-03364774 **[snippet]**).
- Gay, Hartert, Schaus, "Simple and Scalable Time-Table Filtering for the Cumulative Constraint", CP 2015, https://rd.springer.com/chapter/10.1007/978-3-319-23219-5_11 **[snippet]**; the author PDF returned 404. Results not verified.
- Vilim CP 2009 (edge finding O(kn log n)) and Ouellet-Quimper CP 2013 (timetable extended edge finding): **not fetched**. Vilim CPAIOR 2011 (timetable edge finding) is confirmed only as the reference named in CP-SAT's `timetable_edgefinding.h` **[source code]**.

All of these target long durations with partial overlaps and compulsory parts. For unit tasks they are at best faster ways to compute the Hall test. Expected benefit for us: none for bounds, none for incumbents.

### A4. Precedence-aware versions - the part that can add something

**AllDiffPrec.** Bessiere, Narodytska, Quimper, Walsh, "The AllDifferent Constraint with Precedences", CPAIOR 2011, https://ar5iv.arxiv.org/html/1103.3745 **[full text via HTML]**. All-different plus strict precedences; motivated by single-machine unit-time instruction scheduling. Domain consistency is NP-hard; bounds consistency is polynomial (O(n^2)). Lemma 1 shows the decomposition is weaker: X1, X2 in {1,2,3} both before X3 in {2,3,4}; the decomposition is bounds consistent but X3 = 2 is impossible because two distinct predecessors must both come earlier. No experiments. A GCC extension is listed as future work; I found no later "GCC with precedences" or "cumulative with precedences" global constraint (**not found**).

**Reasoning - capacity-C generalisation.** If op j has k (transitive) predecessors of type r, all with est >= a, then start_j >= a + ceil(k / C_r). A tail-aware form: for each k, start_j >= min est + ceil(#{ancestors of type r with path-to-j >= k} / C_r) + k (a Hu-style bound applied to an ancestor cone; the Hu formula itself is **[from memory, unverified]**). The other agent covers the Rim-Jain / Langevin-Cerny line.

**Why it can beat Hall and the time-indexed LP.** Levey and Rothvoss (full citation in B4) give the integrality-gap instance for the time-indexed LP: k layers of m+1 unit jobs, each layer preceding the next. Integral schedules need 2 cycles per layer; the LP needs k(m+1)/m; gap at least 2 - 2/(m+1). Hall with heads and tails also gives about k(m+1)/m. The ancestor-count rule gives the exact answer: each job has m+1 predecessors, so its head moves by ceil((m+1)/m) = 2 per layer. Precedence-energy captures "layer slightly wider than capacity" effects that the LP misses.

**Why I expect little on our DAG.** A group contributes about 375 ops over a critical path of 264, roughly 1.4 ops per level against 23 slots per cycle. Ancestors of an op are (shared constants aside) inside its own group, so ceil(#ancestors of type r / C_r) is far below the path length. The bottleneck arises only from summing over 32 groups, which per-op ancestor counting cannot see. Running it to fixpoint and then applying the Hall test (a destructive bound) is cheap and worth one confirming measurement, but I predict a gain of a few cycles at most.

**Bofill, Coll, Suy, Villaret.** CP 2017, "An Efficient SMT Approach to Solve MRCPSP/max Instances with Tight Constraints on Resources" and Coll's thesis (section "Extended Precedence Set: Energy Precedences", https://core.ac.uk/download/87633485.pdf): **[snippet]**, table of contents and a one-line description (minimum time lags updated using renewable-resource demands, "related to energy-based reasoning"). I could not read the definition, so I cannot say whether it differs from the ancestor-count rule.

### A5. What CP-SAT 9.15 has

Sources: installed headers and `ortools/sat/cumulative.cc`, `cumulative_energy.cc`, `linear_relaxation.cc`, `sat_parameters.proto` at tag v9.15 on raw.githubusercontent.com **[source code]**. The .cc files were read through a fetch tool that excerpts; items marked "verify" should be re-checked against the raw file.

| Technique | CP-SAT 9.15 | Evidence |
|---|---|---|
| Timetabling | Yes, always: `TimeTablingPerTask` | `timetable.h`; unconditional in `cumulative.cc` |
| Overload check (= Hall failure test for unit tasks) | Yes, off by default: `AddCumulativeOverloadChecker` / `CumulativeEnergyConstraint` (theta-lambda tree), `use_overload_checker_in_cumulative`. The code comment says it "increases the minimum of the capacity variable"; it does not push task bounds. | `cumulative_energy.h`, `cumulative.cc` |
| DFF / conservative-scale overload | Yes: `AddCumulativeOverloadCheckerDff` / `CumulativeDualFeasibleEnergyConstraint`, `use_conservative_scale_overload_checker`; created only if some demand_min > 1 | `cumulative.cc` |
| Timetable edge finding | Yes: `TimeTableEdgeFinding` (cites Vilim CPAIOR 2011, O(n^2)), `use_timetable_edge_finding_in_cumulative`. **Created only when `helper->NumTasks() <= max_num_intervals_for_timetable_edge_finding`, default 100.** | `timetable_edgefinding.h`, `cumulative.cc` |
| Plain edge finding, not-first/not-last, energetic reasoning, horizontally elastic rules for 1-D cumulative | No class in the cumulative headers. Energetic reasoning parameters exist only for `no_overlap_2d`. | header listing, `sat_parameters.proto` |
| Hall-interval pruning for unit cumulative with C > 1 | No. `AllDifferentBoundsPropagator` ("detect the so called Hall intervals", O(n log n)) serves `all_different` only; `cumulative.cc` has no unit-duration special case; the constraint proto has no GCC. | `all_different.h`, `cumulative.cc`, `cp_model.pb.h` |
| Precedence + energy | Yes: `CumulativeIsAfterSubsetConstraint`, under `use_hard_precedences_in_cumulative`. It sweeps est values and pushes `var >= time + CeilRatio(energy_after_time, capa_max) + min_offset`, i.e. the ancestor-count rule of A4 for one cumulative, forward direction only, using the minimum offset rather than per-task tails. In the excerpt I read, `full_precedences` is filled only inside `if (parameters.exploit_all_precedences())`, whereas the proto comment says "by default we just use direct precedences" - verify whether `use_hard_precedences_in_cumulative` alone creates anything. Header comment: "I am not sure this is the best way, but it does at least push the level zero bound on the large cumulative instances." | `cumulative_energy.h/.cc`, `cumulative.cc`, `sat_parameters.proto` |
| LP side | Static row "sum of energies <= capacity * (makespan - min start)" always (`AddCumulativeRelaxation`). At `linearization_level > 1`: `CreateCumulativeTimeTableCutGenerator`, `CreateCumulativeCompletionTimeCutGenerator`, `CreateCumulativePrecedenceCutGenerator`; `CreateCumulativeEnergyCutGenerator` only if some size or demand is variable or capacity is not fixed. | `linear_relaxation.cc`, `scheduling_cuts.h` |

Consequences for us:
- The static energy row with presence literals is the capacity floor minimised over modes, in LP form. That is the bound we observe.
- The timetable-edge-finding flag is silently inert on any cumulative with more than 100 intervals unless `max_num_intervals_for_timetable_edge_finding` is raised. If the earlier trial did not raise it, the trial tested nothing on our op-class cumulatives. By A2 I still expect no bound gain.
- Hall pruning could be emulated by `all_different` on y_i = C*s_i + k_i (k_i in [0,C)), which by the proof of Quimper et al.'s Lemma 1 equals GCC-ubc bounds consistency on s_i. It adds slot symmetry and, by the wide-window argument, I do not expect it to help.

---

## Topic B - Mathematical-programming lower bounds

### B1. Time-indexed formulations and their strength

Primary source read: Kolter, Grunow, Kolisch, "On Branch-and-Price for Project Scheduling", arXiv:2501.04563 (2025), https://arxiv.org/pdf/2501.04563 **[full text, pp. 1-24]**. Artigues, "On the strength of time-indexed formulations for the resource-constrained project scheduling problem", Operations Research Letters 45(2) 154-159, 2017 was **not readable** (HAL returned access-denied; https://hal-univ-tlse3.archives-ouvertes.fr/hal-01461447 **[snippet of abstract]**); what follows about it is as reported by Kolter et al.

- PDT (pulse, aggregated precedence: sum_t (t + d) x_it <= sum_t t x_jt) versus PDDT (disaggregated: sum_{s >= t-d+1} x_is + sum_{s <= t} x_js <= 1 for each edge and t; attributed to Christofides et al. 1987). "The PDT requires fewer constraints but results in weaker linear relaxations (Artigues, 2017)."
- Their Proposition 2: for a single mode, the precedence-only problem with disaggregated constraints "possesses the integrality property" (they cite Mohring et al. 2001 for proofs) and is solvable as a min-cut (Mohring et al. 2003). By their Theorem 2 (Geoffrion 1974), Lagrangian relaxation or Dantzig-Wolfe over such a subproblem gives exactly the PDDT LP bound.
- Their Proposition 3: with two or more modes the precedence-only subproblem does **not** have the integrality property, so decomposition can in principle beat the PDDT LP.
- Numbers. 480 PSPLIB j30 instances, Gurobi 11: column-generation master (LMP) equals PDDT-LP; PDT-LP is on average 7.50% weaker; PDDT-LP with solver cuts is on average 6.81% stronger than LMP; column generation is on average 77 times slower than PDDT-LP. 200 MMLIB50 multi-mode instances: LMP is on average 0.42% stronger than PDDT-LP and on average 26 times slower; PDDT-LP with cuts is not significantly different from LMP.
- Their conclusion: for pricing problems without resource constraints, Dantzig-Wolfe "does not strengthen lower bounds beyond the compact disaggregated PDDT formulation"; multi-mode gains are "of a rather theoretical nature".

CP-SAT 9.15: has no time-indexed LP; its LP uses start variables with the cuts in A5. Its precedence cuts are closer to the aggregated form.

### B2. Lagrangian relaxation by minimum cut

Mohring, Schulz, Stork, Uetz, "Solving Project Scheduling Problems by Minimum Cut Computations", Management Science 49(3) 330-350, 2003; MIT working paper record https://dspace.mit.edu/handle/1721.1/693 **[abstract only]**. Lagrangian relaxation of the resource constraints of a time-indexed IP; the relaxed problem is a minimum cut; the Lagrangian solution drives relaxation-based list scheduling. The abstract claims "strong lower bounds"; I read no numbers. By B1 the bound equals the PDDT LP bound for a single mode.

**Reasoning - fit to our problem.** Three properties make it attractive here:
1. Dualising the five capacities and the memory limit leaves 32 **identical, independent** subproblems. One min-cut on about 375 classes x 1,050 window cycles (about 400K nodes) per subgradient step, multiplied by 32.
2. Memory fits. A value's live interval is [def, last use). Add a dummy "kill" event after all consumers; with nonnegative memory prices mu_t the cost is M(kill) - M(def), M(t) = w * sum_{s<t} mu_s. That is a pair of start-time-dependent costs on precedence-linked events, exactly the structure the min-cut handles.
3. It yields prices per (resource, cycle) and per (memory, cycle). List scheduling by Lagrangian start times is a different priority rule from our greedy and could be used for LNS neighbourhood selection.

It cannot beat the PDDT LP for fixed modes. With modes the subproblem is not integral, so an exactly solved subproblem (small: one group) could give more; the published evidence (0.42% on MMLIB50) says not to expect much.

### B3. Destructive bounds and column generation over antichains

- Brucker and Knust, "A linear programming and constraint propagation-based lower bound for the RCPSP", EJOR 127(2) 355-362, 2000: **not read**; described in Kolter et al. **[full text of the secondary source]**: column generation on the antichain (feasible-subset) LP of the preemptive problem for a trial makespan T, "If this relaxation turns out to be infeasible, the approach proves that T is a valid lower bound", after constraint propagation has excluded antichains.
- Baptiste and Demassey, "Tight LP bounds for resource constrained project scheduling", OR Spectrum 26(2) 251-262, 2004, https://www.doi.org/10.1007/S00291-003-0155-1 **[snippet]**: destructive bounds from constraint propagation plus LP; a listing reports 16 PSPLIB j60 instances closed. Not verified in full text.

**Reasoning.** The antichain LP is preemptive. With unit durations preemption changes nothing, so it becomes a per-cycle "configuration" LP: choose how many cycles use each pattern of simultaneously running ops. Aggregated over classes this is a small cutting-stock-like LP. Its distinctive strength is integrality within a cycle (for example "at most one demand-8 shifted op per cycle", see A2), which a time-indexed LP sees only fractionally. It is blind to ordering apart from windows.

### B4. Unit-time theory: where the LP is tight, and LP hierarchies

- Levey and Rothvoss, "A (1+epsilon)-approximation for makespan scheduling with precedence constraints using LP hierarchies", STOC 2016, https://ar5iv.arxiv.org/html/1509.07808 **[full text via HTML]**. The time-indexed LP "has a constant integrality gap" (at least 2 - 2/(m+1), instance in A4). (log n)^{O((m^2/eps^2) log log n)} Sherali-Adams rounds reduce the gap to 1 + eps for fixed m. Cited there: Graham list scheduling 2 - 1/m; Lam-Sethi (Coffman-Graham) 2 - 2/m, optimal for m = 2; Lenstra-Rinnooy Kan 4/3 hardness; Svensson (2 - eps) hardness under a Unique-Games variant when m is part of the input.
- Garg, "Quasi-PTAS for Scheduling with Precedences using LP Hierarchies", arXiv:1708.04369, https://arxiv.org/abs/1708.04369 **[abstract only]**: (log n)^{O(1)} rounds. Li 2021: **not searched / not found**.
- Jansen, "Precedence Constraint Scheduling on a Fixed Number of Machines", arXiv:2611.09392 (Nov 2026), https://arxiv.org/abs/2611.09392 **[abstract only]**: claims an exact polynomial algorithm for Pm|prec,p_j=1|Cmax for every fixed m >= 3 with exponent 6m^2 + 13m + 1 (degree 94 for m = 3). A very recent preprint; I cannot assess it and it has no practical use at our sizes.
- Busing, Draeger, Mathwieser, arXiv:2511.12395, https://arxiv.org/html/2511.12395v1 **[partial full text]**: cites Coffman-Graham for two machines and Bodlaender-Fellows for W[2]-hardness in width plus machines.
- Polynomial special cases - in/out-trees (Hu 1961), interval orders (Papadimitriou-Yannakakis 1979), two machines (Fujii et al.; Coffman-Graham) - and typed task systems (Jaffe, SIAM J. Comput. 9(3) 1980; record only at https://dspace.mit.edu/handle/1721.1/148939): **[from memory, unverified]** apart from what is cited above. A search snippet (Verriet 1998) says typed task systems are NP-complete even for unit-length fork graphs; I did not read the report.

**Relevance.** These results are about small m with capacity saturated by wide layers. Our regime is the opposite: each group is narrow, and the load comes from 32 parallel copies. The hierarchy results are not implementable. The useful takeaway is the gap instance: LP weakness comes from layers slightly wider than capacity, which is not our structure.

### B5. Event-based formulations

Kone, Artigues, Lopez, Mongeau, "Event-based MILP models for resource-constrained project scheduling problems", Computers & Operations Research 38(1) 3-13, 2011 **[snippet]** (HAL copies returned access-denied). Two event-indexed models (start/end events and on/off events) with variables indexed by events instead of time; a later correction note concerns the start/end model. The claim that event-based models suit long horizons but have weak LP relaxations is **[from memory, unverified]**. For us, with unit durations and 20+ ops per cycle, events are cycles, so nothing is saved. Not recommended.

### B6. Analytical answer for our instance (Reasoning)

Setting: 32 identical groups of about 375 ops, critical path 264, floor 1,271, best about 1,314.

**Is anything strictly stronger than "work / capacity with heads and tails"?**

| Bound | Captures (1) modes x integrality | (2) start/tail ramps | (3) memory | Verdict |
|---|---|---|---|---|
| Hall / heads-and-tails per type, modes fixed | No | Only as counts of ops with early heads | No | Baseline |
| Mode-aware Hall LP: mode fractions per class, one Hall row per type per interval [a,b) | Fractionally: one mode split must satisfy all interval rows for all types at once | Yes, jointly across types | No | Strictly stronger than minimising each interval separately; small LP |
| Per-cycle configuration LP / DFF | Yes (per-cycle packing integrality such as 8 + 4 on a 12-wide ALU) | No | No | Stronger only if the shifted mode is a demand-8 task |
| Ancestor-count precedence energy to fixpoint, then Hall (destructive) | No | In principle; here dominated by the critical path | No | Cheap; expect a few cycles at most |
| PDDT LP with mode variables | Fractionally | Yes: cumulative completion of a class at t is bounded by its predecessors at t-1 while capacity is charged, so cross-type pipeline fill is modelled | Weakly, via rows L_{v,t} >= F_def(t) - F_c(t) per consumer and sum w_v L_{v,t} <= 1,536 | Strongest polynomial candidate; dominates the Hall LPs |
| Lagrangian / min-cut | Same as PDDT LP for fixed modes | Same | Yes, priced | Same value; cheaper per iteration; gives prices |
| Counting MILP (below), integer | Yes | Yes | Yes | Exact |

**Will the PDDT LP capture the ramp?** It captures the fractional version. Advancing "10% of every group" in the LP corresponds to really advancing about 3 groups while the others wait, so with 32 copies the fluid relaxation is close to realisable for precedence and capacity. I therefore expect the LP bound without memory rows to land near the heads-and-tails floor. The remaining gap would then come from (a) per-cycle integrality of mode patterns and (b) memory limiting how many groups can be in flight. That is a hypothesis; the LP is the instrument that splits the 43-cycle gap into these parts.

**Size.** Disaggregated: 12K ops x about 1,050 window cycles (T about 1,314 minus critical path 264) is about 12.6M step variables and over 25M precedence rows: too big, consistent with the earlier dismissal. Aggregated by class: 375 classes x 1,050 is about 0.4M variables, about 1M precedence rows, 5 x 1,400 capacity rows, plus memory rows. That is within reach of an LP solver (interior point or first-order), though it is a heavy job that must be queued.

**Aggregation over the 32 groups - the counting formulation.** Let y_{c,t} in {0..32} be the number of copies of class c completed by cycle t.
- Precedence c' -> c with latency 1: y_{c,t} <= y_{c',t-1} (sum of the per-group step constraints, hence valid).
- Capacity: sum over classes of type r of (y_{c,t} - y_{c,t-1}) <= C_r.
- Memory: L_{v,t} >= y_{def,t} - y_{c,t} for each consumer c; sum_v w_v L_{v,t} <= 1,536.
- A shared op (multiplicity 1) feeding all copies: y_{c,t} <= 32 * y_{shared,t-1}.

Two claims:
1. *Same LP bound as the disaggregated PDDT LP.* Averaging any LP solution over the group permutations gives a symmetric solution of equal value, and symmetric solutions correspond one-to-one with aggregated ones.
2. *Exact as an integer program.* Given integer y, let group g complete class c at tau_c(g) = min{t : y_{c,t} >= g} (group 1 always first). Then y_{c,t} <= y_{c',t-1} gives tau_c(g) >= tau_{c'}(g) + 1, capacity is unchanged, and because the assignment is nested, the number of groups holding value v live at t equals y_{def,t} - min_c y_{c,t}. So every aggregated integer solution is a real schedule with the same makespan and memory profile. Mode-dependent latencies need per-mode counts; I have verified validity as a relaxation but not exactness in that case.

This replaces 32! group symmetry by integer counts (far stronger than lex constraints), shrinks the model 32-fold, and gives a compact LP bound. Conditions: groups must be exactly isomorphic and interact only through capacities, memory and shared ops. It is a standard high-multiplicity idea; I did not find a paper applying it to RCPSP (**not found**, limited search).

---

## Ranked ideas with cheap experiments

1. **Counting (high-multiplicity) time-indexed model over the 32 identical groups.** Valid bound by summation; exact for fixed-latency precedence, capacity and memory by the nested argument. *Experiment:* on a 2-round instance (about 1,000 ops, about 30 classes per group, horizon about 200) build the aggregated LP and compare with the capacity floor and the best known schedule; then add memory rows to see how much of the gap is memory. Then solve it as a MILP (or a CP-SAT model with y in 0..32) and compare with the 20-30% gap of the per-op model. Must be queued as a heavy job at 16 rounds.
2. **Mode-aware Hall LP plus per-cycle pattern integrality.** A tiny LP: mode counts per class, one row per type per interval over distinct heads/tails, plus "shifted ops <= cycles" style pattern rows where the shifted mode is a demand-8 task. *Experiment:* a script that reads class counts, heads and tails from an existing schedule JSON and prints the bound next to 1,271; runs in seconds. If it stays at 1,271, capacity-plus-window reasoning is exhausted and the gap is ordering or memory.
3. **Lagrangian prices by per-group min-cut, memory priced too.** Same value as the PDDT LP for fixed modes, but only one group-sized min-cut per iteration, and it produces (resource, cycle) prices and alpha-point orderings. *Experiment:* take the duals of the aggregated LP from idea 1 as prices without implementing min-cut, order ops by priced start time, and feed that as a greedy priority or to choose LNS windows.
4. **Destructive bound: ancestor-count precedence energy (with tails) to fixpoint, then Hall per type.** *Experiment:* offline script on the op DAG; report how far any head moves beyond the longest-path value. Predicted near zero; if confirmed it also settles that `use_hard_precedences_in_cumulative` / `exploit_all_precedences` cannot help.
5. **Two CP-SAT configuration checks from the source.** (a) Timetable edge finding is skipped above 100 intervals unless `max_num_intervals_for_timetable_edge_finding` is raised. (b) In the excerpt read, `CumulativeIsAfterSubsetConstraint` is created only when `exploit_all_precedences` is also true. *Experiment:* read the raw `cumulative.cc` at v9.15 to confirm (b), and check which values the earlier ablations used. Expected effect on bounds is none (A2), but it corrects the record of what was actually tested.

Not recommended: stronger energetic or edge-finding propagators (collapse to Hall for unit tasks), event-based MILP (events are cycles), column generation / branch-and-price (Kolter et al.: same bound as PDDT LP for single mode, 0.42% average gain for multi-mode on MMLIB50, 26 to 77 times slower on average), LP hierarchies (theoretical).
