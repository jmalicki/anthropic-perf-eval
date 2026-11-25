# Part 1 - Search in other CP solvers, and the CP-SAT scheduling roadmap

Evidence labels: **full text** (I read the paper or slides), **abstract only** (abstract or search snippet), **source/docs** (code or API docs), **[from memory, unverified]** (not read in this session). Assessments marked "My assessment" are my reasoning, not a source's claim.

## Headline findings

1. **No solver's search is designed for our bound regime.** Failure-directed search (FDS) and lazy clause generation (LCG) with VSIDS produced their makespan proofs on benchmarks where resources are scarce and tasks are near-disjunctive. Two sources I read say that high capacity is the bad case: the Pumpkin CP 2025 paper reports that instances with resource constrainedness under 15% "almost never benefit", and Perron's own 2025 slides list a unit-like "pigeon hole" cumulative as an open propagation problem. Do not expect FDS or another LCG solver to lift our bound above the capacity floor.
2. **For finding better schedules at 12K ops, CP Optimizer and OptalCP are the credible alternatives.** Perron's 2025 slides state that CP-SAT is "missing heuristics for large instances". OptalCP has a free academic licence, a Python API, and a "Large" preset that runs all workers as LNS.
3. **OR-Tools v10.0-rc.1 was tagged on 23 Nov** (year not shown on the page; presumably 2026). It contains several scheduling changes that are testable without code changes, including splitting cumulative constraints that have precedences.
4. **CP-SAT's multi-mode energy cuts depend on a model shape it detects by inspection.** That makes "does our VALU-shift encoding get detected?" the cheapest high-value check from this survey.

---

## TOPIC A - Search in other CP solvers

### A1. Failure-directed search (CP Optimizer, OptalCP)

**Sources**
- Vilím, Laborie, Shaw, "Failure-Directed Search for Constraint-Based Scheduling", CPAIOR 2015, LNCS 9075, pp. 437-453. **Abstract only**, via https://dblp.uni-trier.de/rec/conf/cpaior/VilimLS15.html and search snippets; the Springer page redirected to a login and the author PDF URL returned 404. The snippet says FDS and LNS+FDS were tested on job-shop, job-shop with operators, flexible job-shop, RCPSP, RCPSP/max, multi-mode RCPSP and multi-mode RCPSP/max, and that it "often improves best-known lower and upper bounds and closes many open instances". Exact counts: not found.
- Heinz, Vilím, Hanzálek, "Reinforcement Learning for Search Tree Size Minimization in Constraint Programming: New Results on Scheduling Benchmarks", Computers & Industrial Engineering 209, 2025. **Full text, first 100K of 179K characters**, https://arxiv.org/html/2508.20056. This is the best available description of FDS as implemented in OptalCP.

**What it is.** FDS is a complete restarting tree search that assumes no better solution exists and tries to prove it quickly. A choice is a binary split of a variable's domain, for example "start <= 5 or start > 5". Each branch has a rating updated as `rating := (1-alpha)*rating + alpha*localRating`, where `localRating` is 0 if the branch failed immediately and `1 + 0.5^n` otherwise (n = number of domains reduced by propagation). The lowest-rated choice is branched on first. Restarts are geometric, ratings persist, and each restart adds nogoods that block the explored branch path. The paper calls FDS "Plan B" beside LNS: LNS supplies the upper bound, FDS proves bounds. It relies on timetable filtering, edge finding, detectable precedences and not-first/not-last. There is no conflict analysis or explanation.

**Does CP-SAT 9.15 have it?** No. I found no FDS subsolver in the installed `sat_parameters.pb.h`, and a search of OR-Tools discussions for "failure directed search" returned nothing. The closest CP-SAT analogues, all present in the 9.15 header, are `use_objective_shaving_search`, `use_probing_search`, `variables_shaving_level`, `optimize_with_lb_tree_search` and `optimize_with_core`. These attack the bound with learned clauses and LP.

**Reported effects** (Heinz et al., 900 s per instance)
- Enhanced FDS vs original FDS in OptalCP: 1.7x faster on JSSP, 2.1x on RCPSP.
- Enhanced FDS vs FDS in CP Optimizer 22.1: 3.5x faster on JSSP, 2.1x on RCPSP.
- Lower bounds improved on 78 of 84 open JSSP instances and 226 of 393 open RCPSP instances (PSPLIB J60/J90/J120); 7 instances closed (4 JSSP, 3 RCPSP).
- Each bound was obtained destructively: one unit above the highest makespan proven infeasible in 900 s. The authors "make no claims about the overall efficiency" against bounds obtained under other time limits.

**Caveat: FDS and our capacity bound.** My assessment, not a claim from the papers:
- FDS learns which choices fail quickly. On job-shop or scarce-resource RCPSP, a start-time split forces orderings and edge finding fails within a few levels.
- In our problem a cycle holds 12 scalar plus 6 vector ops. Splitting one op's start time changes nothing until a cycle is completely full, so nearly every branch gets the same no-failure rating and the heuristic has no signal.
- Refuting makespan = floor + k is a counting (pigeonhole) argument over thousands of interchangeable unit tasks. A tree of binary start-time splits without a global counting bound needs exponentially many nodes.
- PSPLIB J120 has 120 activities, 4 resources and durations 1 to 10. We have 1,000 to 12,000 unit tasks. Nothing in the FDS literature I could read covers that regime.

Expected benefit: (i) better schedules: none directly, FDS is a prover. (ii) Bounds: I expect none beyond what root propagation gives. It would only help if a tight memory limit or latency chains made the problem locally disjunctive, which the 264-cycle critical path against a 1,271 floor suggests is not the case.

### A2. CP Optimizer: self-adapting LNS, POS, temporal linear relaxation

**Sources**
- Laborie, Rogerie, Shaw, Vilím, "IBM ILOG CP Optimizer for scheduling", Constraints 23(2):210-250, 2018. **Abstract only**, https://dblp.uni-trier.de/rec/journals/constraints/LaborieRSV18.html plus snippet. The abstract claims only that automatic search "provides good performance out of the box". Full text was behind a login.
- Laborie, Rogerie, "Temporal linear relaxation in IBM ILOG CP Optimizer", J. Scheduling 19:391-400, 2016. **Abstract only** (snippet). TLR is a linear relaxation over start/end times and most resource-assignment decisions, used to guide search for problems with temporal and allocation costs, evaluated on 15 benchmarks with irregular temporal costs or presence costs.
- Laborie & Godard 2007 (self-adapting LNS): not found online in this session.

**What it does that CP-SAT does not** [from memory, unverified, except where a source is named]
- The default search is LNS plus FDS (confirmed by the FDS snippet above).
- The LNS relaxes a fragment while keeping the remaining tasks as a partial-order schedule (POS): precedence arcs between consecutive users of each resource unit, not fixed start times. The frozen part can therefore slide in time.
- A learning layer picks neighbourhoods and completion strategies by reward.
- CP-SAT's scheduling LNS neighbourhoods are documented by the other agent. The relevant difference is that CP Optimizer's LNS is the primary engine and is tuned for scale.

**Fit to our problem**
- TLR targets earliness/tardiness and allocation costs. With a pure makespan objective and no costs it has little to guide. Low relevance.
- POS-style relaxation is relevant in principle. With capacity 12 and unit tasks, a per-unit POS adds one chain per resource unit, so the frozen part behaves like a set of chains that can compress. That is the "everything after the window moves as one block" behaviour we already built by hand, generalised.
- Mode re-decision is where our gains come from. CP Optimizer models this with `alternative` over optional intervals and its LNS re-decides alternatives inside the fragment [from memory, unverified].

**Effect sizes with CP-SAT as comparator**
- Lan and Berkhout, "PyJobShop: Solving scheduling problems with constraint programming in Python", arXiv:2502.13483, 2025. **Full text**, https://arxiv.org/html/2502.13483v1. OR-Tools 9.11 vs CP Optimizer 22.1.1, 900 s:

| Problem | Instances | Relative deviation from best-known, % (OR-Tools / CPO) | Optimality gap, % (OR-Tools / CPO) |
|---|---|---|---|
| RCPSP | 2,520 | 0.93 / 0.46 | 3.46 / 4.26 |
| MMRCPSP | 1,080 | 0.18 / 0.27 | 0.94 / 6.65 |
| RCMPSP | 2,254 | -0.52 / -0.85 | 14.91 / 67.36 |

- Reading: CP Optimizer finds slightly better schedules on RCPSP and multi-project RCPSP; CP-SAT has much better bounds everywhere and is better on multi-mode. The authors write that "CP Optimizer generally handles large-scale instances more efficiently". On non-permutation flow shop (up to 48,000 tasks) the deviation was 8.88% for CPO vs 13.53% for OR-Tools.
- Implication for us: switching to CP Optimizer could plausibly help the 12K-op primal side and would hurt the bound side. It is commercial; IBM's academic terms were not checked.

### A3. OptalCP

**Sources**: https://optalcp.com/ (product page), https://optalcp.com/docs/benchmarks, https://optalcp.com/docs/api and the `Parameters` reference, https://github.com/ScheduleOpt/optalcp-benchmarks (README). All **docs/vendor pages**.

- **Licence.** Proprietary. "Preview (free)" gives the full modelling API and objective values, but solution values need a licence. "Academic (free)" gives full solutions under a cite-and-use licence, requested by email. No published prices. The benchmark repository is MIT.
- **API.** Python, TypeScript/JavaScript, C#. A DOcplex migration layer exists (`optalcp-cpo`).
- **Modelling fit.**
  - Optional interval variables, `alternative`, `span`, `noOverlap`, precedence constraints.
  - Cumulative via `pulse` with `<=` on sums of pulses. Unit tasks are ordinary length-1 intervals.
  - A reservoir-style resource via `stepAtStart` steps, with its own `reservoirPropagationLevel` (1 to 2). The API index I read lists `pulse` and `stepAtStart` only.
  - Our memory resource can be a pulse over a variable-length "live" interval (demand 1 or 8), or +d at definition and -d at last use. The first matches our current sixth cumulative. Whether pulse height may depend on mode was not confirmed.
- **Search.** `searchType` is one of `LNS`, `FDS`, `FDSDual`, `SetTimes`, `FDSLB`, set per worker.
  - `FDSDual` is "Failure-Directed Search working on objective bounds", with `fdsDualStrategy` Minimum/Random/Split.
  - `simpleLBWorker` computes a propagation-only lower bound before search.
  - `cumulPropagationLevel` 1 to 4 adds edge finding, disjunctive reasoning and "max-energy reasoning".
  - `usePrecedenceEnergy` enables "the precedence energy algorithm for intervals with multiple predecessors or successors on the same resource". That is the precedence-plus-energy combination our problem needs, and it is off by default.
  - `preset=Large` (automatic above 100,000 variables) uses all-LNS workers with weaker propagation.
  - `lnsUseWarmStartOnly` and external-solution support mean our greedy schedule can seed it.
- **Benchmarks vs CP Optimizer (vendor).** 4 threads, CPO at defaults, OptalCP split between LNS and FDS. On 242 job-shop instances, median speedup 5.4x on the 136 both prove optimal. Across 4,010 instances in five families both reach the same objective on 3,195; where they differ OptalCP is better in 62 to 95% of cases depending on family. RCPSP, MMRCPSP (PSPLIB, MMLIB50, MMLIB100) and RCPSP-CPR pages exist, but I could not open the per-family numbers (404).
- **Benchmarks vs CP-SAT.** The vendor pages do not include CP-SAT. One independent source: "A Comprehensive Benchmark of Constraint Programming Solvers for the Makespan-Minimisation Job Shop Scheduling Problem", Preprints.org 202605.1319, May 2026, not peer reviewed. **Search snippet only** (page returned 403); authors not seen. It says OptalCP "is the most robust engine overall, certifying optimality on 57.5% of the instances with the smallest average optimality gap (3.55%)", while Hexaly dominates industrial-scale instances. Job-shop only, so weak evidence for us.
- **No RCPSP comparison of OptalCP against CP-SAT was found.**

Expected benefit: (i) schedules: plausible, as it is the one alternative with a free licence, a large-instance LNS preset and native alternatives. (ii) Bounds: `usePrecedenceEnergy` plus max-energy reasoning might move the root bound by a few cycles. FDSDual is unlikely to add more, for the reason in A1.

### A4. Chuffed and the LCG RCPSP line

**Sources**
- Schutt, Feydy, Stuckey, Wallace, "Solving RCPSP/max by lazy clause generation", J. Scheduling 16(3):273-289, 2013. **Full text of sections 2.3 to 5.3** from the preprint, https://arxiv.org/pdf/1009.0347.
- Schutt, Feydy, Stuckey, "Explaining Time-Table-Edge-Finding Propagation for the Cumulative Resource Constraint", CPAIOR 2013. **Full text via summary**, https://ar5iv.labs.arxiv.org/html/1208.3015.
- Schutt, Feydy, Stuckey, Wallace, "Explaining the cumulative propagator", Constraints 16(3):250-282, 2011. **Abstract only** (catalogue snippet): closes "a substantial number of open problems from the PSPLIB". Count not found.
- Yang, Schutt, Stuckey, "Time Table Edge Finding with Energy Variables" (CPAIOR 2019 per the file name; venue not verified). **Pages 1-2 read**, https://people.eng.unimelb.edu.au/pstuckey/ttefen/ttefen.pdf.

**What gave the proofs**
- Timetable cumulative with explanations, plus redundant reified disjunctions for every pair with `r_i + r_j > capacity` (section 3.2).
- Start-time domains tightened by Bellman-Ford and the objective written as `S_i + tail_i <= objective` (section 3.1), which the authors say "gave major improvements".
- Search: VSIDS over bound literals with geometric restarts (base 250, factor 2.0), after a "hot start" of 500 nodes of smallest-start-time branching to seed activities. The combination is "HotRestart".
- Phase 2 re-posts the model with horizon = the upper bound found, because a tight horizon shrinks the Boolean encoding.
- The 2013 TTEF paper computes bounds destructively: prove infeasibility at makespan M, increment, repeat. That used HotStart without restarts.

**Effects**
- RCPSP/max: 2,278 of 2,340 instances solved with HotRestart vs 2,237 for start-time branching (Table 1); 573 open instances closed (abstract).
- TTEF explanations, PSPLIB, 10 minutes: bounds improved on 264 of 432 open instances (61.1%) against 151 with the consistency check only; 6 instances closed.

**Does CP-SAT have it?** Yes, and more. Perron's slides say CP-SAT "is inspired by Chuffed". The 9.15 header has `use_timetable_edge_finding_in_cumulative`, `use_disjunctive_constraint_in_cumulative`, `use_hard_precedences_in_cumulative`, `use_dynamic_precedence_in_cumulative` and `use_overload_checker_in_cumulative`.

**Fit to our problem.** Poor for bounds.
- The pairwise-disjunction trick needs `r_i + r_j > capacity`. With unit demands and capacity 12 or 6 no pair qualifies; only the single flow slot is disjunctive.
- The Yang et al. introduction says timetable propagation "is usually superior for highly disjunctive problems" and edge finding "more appropriate for highly cumulative problems". We are at the far cumulative end, where explanations are large and reuse poorly.
- The energy-variable idea is relevant to modes: with rotation or modes, the minimum duration times minimum demand underestimates energy, so the propagator carries an explicit energy variable. CP-SAT does the equivalent in its cuts (see B1).
- Two cheap carry-overs remain: the `S_i + tail_i <= makespan` objective form, and destructive bounding (makespan fixed, satisfaction only) on windows rather than the whole program.

### A5. Pumpkin (TU Delft) and Huub (Monash)

**Sources**
- Sidorov, Marijnissen, Demirović, "Unite and Lead: Finding Disjunctive Cliques for Scheduling Problems", CP 2025, LIPIcs 340, article 35. **Full text via summary**, https://drops.dagstuhl.de/storage/00lipics/lipics-vol340-cp2025/html/LIPIcs.CP.2025.35/LIPIcs.CP.2025.35.html.
- Sidorov, "On inferring cumulative constraints", arXiv:2602.15635, Feb 2026 (submitted to CP 2026). **Abstract only**, https://arxiv.org/abs/2602.15635.
- Pumpkin source docs, https://docs.rs/crate/pumpkin-propagators/0.5.0/source/src/propagators/cumulative/mod.rs. **Source/docs**: the cumulative module "contains solely time-tabling propagators".
- Dekker, Ignatiev, Stuckey, Zhong, "Towards Modern and Modular SAT for LCG", CP 2025, LIPIcs 340, article 42. **Snippet only**. Huub docs (snippet): cumulative uses a time-table profile; disjunctive has overload checking, edge finding, not-first/not-last and detectable precedences.
- MiniZinc Challenge 2025 results, https://www.minizinc.org/challenge/2025/results.

**Findings**
- Pumpkin's cumulative is timetable only: weaker than CP-SAT's. Its strength is proof logging (DRCP certificates).
- "Unite and Lead" adds Booleans for "tasks i and j are disjoint", finds cliques, and runs disjunctive overload checking on them. Compared with OR-Tools 9.11: 4 new upper bounds, 4 new lower bounds, one instance closed (Pack-d #3). The authors report the gain only "when resource contention is high"; resource constrainedness under 15% "almost never" benefits and 40% or more benefits most. Per the summary, the paper also proves an exponential lower bound for a pigeonhole-like "3n" problem stated for unit durations, the regime we are in.
- "On inferring cumulative constraints" treats cumulatives as linear inequalities over occupancy vectors and derives new ones from covers with lifting: 25 new lower bounds on RCPSP and RCPSP/max, 8 directly from the inferred constraints without search. Solver not named in the abstract.
- MiniZinc Challenge 2025: OR-Tools CP-SAT took gold in fixed, free and parallel. Pumpkin tied for bronze in fixed. Huub entered late and unranked. None of the 20 problems is labelled RCPSP; `ihtc-2024-*` and `carpet-cutting` use `cumulative`. Per-problem scores were not retrievable. Perron's slides say CP-SAT "won all gold medals in the last 6 years".

**Fit to our problem.** Neither solver is worth porting to. One idea transfers: linear combinations of our typed capacities give mode-independent redundant cumulatives. My own derivation in that spirit: per cycle `alu <= 12` and `valu <= 6` imply `alu + 8*valu <= 60`, in which a shiftable vector op has demand 8 in either mode. Energy reasoning on that resource does not depend on mode literals. It may be equivalent to the energetic rows already tried; check before spending time.

---

## TOPIC B - CP-SAT maintainers' statements, roadmap, benchmarks

### B1. What the maintainers say

**Source**: Laurent Perron, "OR-Tools and CP-SAT", guest lecture slides, UPenn CIS 1921, spring 2025. **Full text, 56 slides read**, https://cis.upenn.edu/~cis1921/files25s/guestlecture.pdf. The slides describe themselves as a follow-up to the CPAIOR 2020 master class. Also: Perron, Didier, Gay, "The CP-SAT-LP Solver (Invited Talk)", CP 2023, LIPIcs 280, 3:1-3:2. **Full text, but it is a 2-page abstract with no algorithms**, https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.CP.2023.3. No full paper on CP-SAT scheduling cuts was found.

Direct quotes from the slides:
- "Where we are": "Scheduling: competitive or better than state of the art on academic benchmarks. Better than commercial solvers on small to medium scheduling problems, **missing heuristics for large instances**."
- "What's next?": "Improved scheduling (branching heuristics, LNS, propagation, modeling)", "Reduce encoding size", "Non violation based Local Search?".
- "Dealing with Energy": "In the PSPLIB, the duration and the demand are fixed per recipe. Thus the energy of a demand can be linearized as sum(recipe_i_bool_var * fixed_energy_i). **This structure is discovered by inspecting the model.** This is very effective on the PSPLIB as the energetic cuts take into account the Boolean variables of the selected recipes."
- "Stronger Scheduling Propagation": "2k+1 tasks of duration p, cumulative of horizon (k+1/2)*p, minimize capacity used. **Pigeon hole problem.** Mitigation ideas: 1. divide by gcd 2. subset-sum 3. use MIR like super-additive techniques applied to linear relaxation, cuts or propagation?" Posed as an open question.
- "Completion Time Cuts" (Smith rule): "Can be adapted to cumulative by dividing pi by the capacity max".
- "More Scheduling Cuts": energetic cuts lifted with intervals that must intersect the window.
- Flexible job-shop: with `sum(presence_literal * interval_size) <= span` in the LP, "8 threads, 15 min, from 205 to 246 proven out of 313".
- Conclusion: CP-SAT "could incorporate different relaxation/specialized (expensive) propagation algorithms, provided they can be explained, or provided they can be useful at the root node."

**Reading for our problem**
- The maintainers confirm both of our symptoms: large-instance primal heuristics are a known gap, and capacity pigeonhole bounds are a known open propagation problem.
- The "Dealing with Energy" slide is actionable. Energy cuts see mode Booleans only if the model has the recipe shape CP-SAT detects. Our VALU shift is a recipe with energies (1 VALU, 0 ALU) or (0 VALU, 8 ALU). If the 8 scalar ops are 8 separate optional unit intervals, detection may fail; one interval on the ALU cumulative with demand `8 * shift_literal` is closer to the PSPLIB shape. Which form is detected is a question for the CP-SAT-internals agent.
- The completion-time cut "divided by capacity" is weak at capacity 12, consistent with the bound sitting at the floor.

### B2. OR-Tools release notes

**Source**: https://github.com/google/or-tools/releases and the v10.0-rc.1 tag page. **Release notes, read through a summariser**; names below are as it reported them.

- **9.12** (Feb 2025): "Improve Cumulative propagation and cuts"; hints now preserved through presolve; "Speed up LNS".
- **9.13** (Jun 2025): "Detect, store and use precedences, with their bounds ... in presolve, propagation, and cuts"; "Improvement to scheduling cuts".
- **9.15** (12 Jan 2026): "Improved scheduling cuts and memory management"; "Improve shared tree workers and clause sharing".
- **v10.0-rc.1** (23 Nov, pre-release):
  - Presolve "splits `cumulative` constraints that have precedences".
  - New `ls_scheduling` subsolver, described as tabu search for job-shop and machine scheduling. Probably disjunctive only.
  - New shaving, probing and branching subsolvers (`shaving*`, `probing_no_lp`, `alt_fixed*`, `ROUND_ROBIN_SHAVING_SEARCH`).
  - Shared-tree rework with per-node lower-bound sharing.
  - LNS model-copy rework, deterministic presolve limits for adaptive LNS, and the ability to skip redundant scheduling constraints in LNS and Feasibility Jump.
  - A fix for non-determinism in interleaved LNS.
  - No symmetry items.

No FDS, no POS-style LNS, and no unit-duration specialisation appear in any release note read.

### B3. GitHub issues and discussions

Not found. Two searches for discussions on FDS, unit-duration cumulative, or large-instance scheduling LNS returned no relevant threads. I am not reporting anything from memory here.

### B4. Independent comparisons

- **PyJobShop** (A2 table) is the only independent RCPSP/MMRCPSP comparison I read in full.
- **Naderi, Ruiz, Roshanaei**, INFORMS J. Computing 35(4):817-843, 2023. **Abstract only** (snippet). It compares CP Optimizer 20.1 with CPLEX 20.1 and Gurobi 9.1.2 on 12 shop problems and 6,623 experiments; CP Optimizer wins except on pure assignment. CP-SAT is not in the abstract, so it says nothing about CP-SAT vs CPO.
- **Da Col and Teppan**, Operations Research Perspectives 9, 2022, 100249. **Abstract only**, https://ideas.repec.org/a/eee/oprepe/v9y2022ics2214716022000215.html. OR-Tools vs CP Optimizer on job-shop up to one million operations, single and quad core. Result tables not accessible (403). Their ICLP 2019 paper (arXiv:1909.08247): abstract only, no numbers.
- **Hexaly vendor page**, https://www.hexaly.com/benchmarks/hexaly-vs-cp-optimizer-vs-or-tools-on-the-job-shop-scheduling-problem-jssp. On the ten 1,000 x 1,000 instances it states "OR-Tools, Gurobi, and Cplex fail to deliver feasible solutions within 6 hours", attributing the baseline runs to Da Col and Teppan. Vendor claim, different hardware.

---

## Ranked ideas for our problem

**1. Check that CP-SAT's recipe-energy detection fires on our mode encoding; add the mode-free combined resource.**
- Why: maintainers say energetic cuts use mode Booleans only when the structure is "discovered by inspecting the model", and that this is "very effective" on multi-mode PSPLIB. Our gains come from mode re-decisions.
- Cheap experiment: on one 25-cycle window, compare (a) the current encoding, (b) one ALU-side interval per shiftable op with demand `8 * shift_literal`, (c) adding the redundant cumulative `alu + 8*valu <= 60`. Compare root bound and time to optimal. Seconds each.

**2. Try OR-Tools v10.0-rc.1 unchanged.**
- Why: cumulative splitting with precedences, shaving subsolvers, per-node bound sharing and LNS rework all target our model class, at no modelling cost.
- Cheap experiment: rerun the existing CP-SAT ablation on two mid-size instances with the same deterministic-time budget. Note that the Rust `cp_sat` fork needs a rebuild.

**3. Benchmark OptalCP on one mid-size and one 12K instance.**
- Why: free academic licence, Python API, alternatives, cumulative, reservoir, warm start, all-LNS large preset, and a precedence-energy propagator. Perron's slide concedes CP-SAT lacks large-instance heuristics.
- Cheap experiment: export one 1,500-op instance to the Python API. The free Preview edition returns objective values, enough to see whether LNS beats our incumbent and whether `simpleLBWorker` with `usePrecedenceEnergy=1` and `cumulPropagationLevel=4` beats floor + 2, before requesting a licence.

**4. POS-style neighbourhoods in our external LNS.**
- Why: CP Optimizer's primal strength at scale is attributed to LNS over partial-order schedules [from memory, unverified]. Our window LNS fixes absolute times outside the window.
- Cheap experiment: one neighbourhood that frees all mode literals in a round, keeps only relative order (precedence arcs between frozen ops in successive cycles), and leaves times free. Compare cycles gained per second with the 25-cycle window.

**5. Stop spending on search-based bound proofs; use destructive bounds only on windows.**
- Why: FDS, LCG and disjunctive-clique reasoning all report gains under high contention. The CP 2025 paper reports little benefit below 15% resource constrainedness, and Perron lists the capacity pigeonhole as open.
- Cheap experiment: on windows that solve in seconds, pose "makespan <= optimum - 1" as pure satisfaction and record conflicts to refute. If that grows quickly with window size, global proof effort belongs with relaxation-based bounds, not search.
