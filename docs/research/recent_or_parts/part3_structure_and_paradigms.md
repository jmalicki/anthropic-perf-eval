# Part 3 - Structure exploitation (identical groups) and newer exact paradigms

Scope: Topic A (decomposition and structure when the instance is 32 identical sub-DAGs coupled by capacities) and Topic B (DIDP, decision diagrams, core-guided search, SAT encodings, LNS design). Web reading and reading of installed headers only; nothing was run.

Evidence tags used below: **full text** (I read the paper text), **abstract only** (abstract page or search-result snippet), **source code** (OR-Tools header or source), **[from memory, unverified]** (background knowledge, no page opened). Paragraphs headed "Own reasoning" are my arithmetic or argument, not a literature result, and need a checker before anyone relies on them.

## Summary of findings

- **Dantzig-Wolfe over group patterns will not lift the bound.** A 2025 analysis shows that with precedence-only pricing problems the master LP equals the disaggregated time-indexed LP (single mode) or beats it by 0.42 % on average (multi-mode), at 26 to 77 times the cost.
- **Fluid / high-multiplicity theory says the capacity floor is asymptotically the right bound.** For identical-job streams the gap between optimum and fluid bound is an additive constant, so the remaining gap is start-up and tail, not steady state. The same paper gives a template-and-replay heuristic that maps well onto our problem.
- **A group-staggered periodic pipeline is arithmetically worse than the incumbent** at 16 rounds (own reasoning from the brief's numbers). The useful periodic unit is the round, or a small "group-round" template.
- **CP-SAT 9.15 already has dynamic symmetry propagation and orbitope presolve**; SBDS-1UIP, symmetric explanation learning and LDSB are not present. The round-boundary structure admits a stronger, conditional exchange dominance than whole-group lex (own reasoning).
- **Static dominance breaking can hurt a clause-learning solver on RCPSP**: Chu and Stuckey measured 4.07 s with learning alone against 32.84 s with learning plus dominance constraints.
- **Topic B is mostly negative for bounds.** DIDP, decision diagrams and core-guided search all bottom out in the same capacity-floor dual bound. DIDP's beam search and the classical scheduling-LNS neighbourhood designs are the usable parts.

---

## Topic A - Many identical sub-DAGs coupled only by shared capacities

### A1. High-multiplicity scheduling and n-fold IP

**What it is.** High-multiplicity (HM) scheduling encodes the input as job types with counts and asks for algorithms polynomial in the number of types. Hochbaum and Shamir give strongly polynomial algorithms for unit-time jobs on a single machine under several objectives. Clifford and Posner treat parallel machines (identical, proportional, unrelated) for total completion time and makespan. Brauner, Crama, Grigoriev and van de Klundert give a complexity framework and explicitly restrict it to non-preemptive one-machine problems. N-fold IP (Knop and Koutecký) gives fixed-parameter algorithms for R||Cmax and R||sum w_j C_j, with later work extending to the HM encoding via a configuration LP.

**In CP-SAT 9.15?** No. Nothing in the installed headers or `sat_parameters.proto` refers to multiplicity or n-fold structure.

**Fit to our problem.** Poor as an algorithm, useful as a framing. None of the HM results I found handles precedence constraints inside a job type; they cover independent jobs. Our model does have n-fold shape (32 identical bricks, a few global capacity rows per cycle), but the brick is a time-indexed schedule of about 375 ops over about 1,300 cycles, and n-fold algorithms are FPT in brick dimension and coefficient size, so the theory gives no usable procedure.

**Benefit.** (i) Schedules: none directly. (ii) Bounds: none. The transferable idea is the configuration LP (columns are per-cycle or per-group configurations), covered in A3.

Sources:
- Hochbaum, Shamir, "Strongly polynomial algorithms for the high multiplicity scheduling problem", Operations Research 39(4), 1991. https://ideas.repec.org/a/inm/oropre/v39y1991i4p648-653.html (abstract only)
- Clifford, Posner, "Parallel machine scheduling with high multiplicity", Mathematical Programming 89, 2001. https://link.springer.com/article/10.1007/PL00011403 (abstract only, partial snippet)
- Brauner, Crama, Grigoriev, van de Klundert, "A framework for the complexity of high-multiplicity scheduling problems", J. Combinatorial Optimization 9(3), 2005. https://link.springer.com/article/10.1007/s10878-005-1414-7 (abstract only)
- Knop, Koutecký, "Scheduling meets n-fold integer programming", J. Scheduling 21(5), 2018. https://arxiv.org/pdf/1603.02611 (abstract only, search snippet)
- Knop, Koutecký, Levin, Mnich, Onn, "High-multiplicity N-fold IP via configuration LP", Mathematical Programming 2023 (seen only in a search snippet; abstract only).

### A2. Fluid relaxations, periodic schedules, modulo scheduling

**Fluid bounds for identical-job streams.** Bertsimas and Gamarnik solve the fluid relaxation of the job shop in closed form, obtaining a lower bound C_max, and construct a schedule of makespan at most C_max + O(sqrt(C_max)); they report schedules within 1 % of optimal on moderate instances (abstract only: https://www.mit.edu/~dbertsim/papers/Stochastic%20Networks/Asymptotically%20optimal%20algorithms%20for%20job%20shop%20scheduling%20and%20packet%20routing.pdf, J. Algorithms 33(2), 1999, via search snippet).

The most relevant recent paper is Zheng, Qu, Cai, Mao, "High-Multiplicity Flexible Job Shops: From Exact Recurrent Fluid Attainment to Structure-Guided Finite-Horizon Scheduling", arXiv 2608.19711, August 2026 (full text skimmed: abstract, theorem statements, algorithm section; https://arxiv.org/abs/2608.19711):
- Theorem 1 ("exact recurrent fluid attainment"): after a finite integer scaling, the optimal fluid allocation is realised exactly by a repeating discrete schedule. Machine intervals are packed first; job precedence is restored afterwards by relabelling which job copy owns which slot, without moving any slot.
- Theorem 2: under fixed-composition growth the optimal makespan stays within a constant additive gap of the fluid bound, so the relative gap vanishes.
- Algorithm TCTR (type-based cyclic template replay): search a compact job-type template with a model whose size does not grow with the number of copies (they solve it with CP-SAT, 2-second limit, one worker, feasible incumbent suffices), then replay it on the full instance with earliest-gap insertion.
- Reported: on 676 multiplicity-expanded public flexible-job-shop instances, feasible in all cases, mean gap to the fluid bound 1.54 %, against 5.32 % for a job-indexed adaptive LNS and 6.18 % for a hybrid genetic algorithm.

Caveats for us: their machines are disjunctive, jobs are chains, and there is no memory resource. The slot-relabelling trick is nevertheless the same move as the exchange argument in A5.

**Cyclic job shop with identical jobs.** Roundy (Math. of OR 17(4), 1992) studies structural properties of cyclic schedules when all jobs are identical and proposes a search algorithm (abstract only: https://pubsonline.informs.org/doi/fpi/10.1287/moor.17.4.842). Lee and Posner (Operations Research 45(1), 1997) show a class of schedules that minimises cycle time and repeats an identical timing pattern every minimal part set (abstract only: https://ideas.repec.org/a/inm/oropre/v45y1997i1p72-91.html). Hanen and Munier (Discrete Applied Mathematics, 1995) are reported to show that periodic schedules are dominant for the cyclic problem on parallel processors (abstract only, bibliographic database snippet). Kimbrel and Sviridenko (OR Letters 36, 2008) are reported by a secondary source to show the HM cyclic job shop is APX-hard (search snippet only). Hall, Lee, Posner "The complexity of cyclic shop scheduling problems" and McCormick and Rao: **not found** beyond bibliography entries.

All of these optimise cycle time (throughput) for an infinite stream. None addresses finite-horizon makespan with start-up and tail, which is our objective; Zheng et al. is the only paper found that does.

**Modulo scheduling ILPs.**
- Eichenberger and Davidson (PLDI 1997): time-indexed, VLIW-centric formulation that counts ops per modulo slot. Original abstract **not found**; characterised here from the Oppermann papers.
- Oppermann, Reuter-Oppermann, Sommer, Koch, Sinnen, "Exact and Practical Modulo Scheduling for High-Level Synthesis", ACM TRETS 12(2), 2019 (full text: https://www.esa.informatik.tu-darmstadt.de/assets/publications/materials/2019/2019_TRETS_JO.pdf). Moovac puts the initiation interval into the ILP and models resources with pairwise overlap binaries between resource-limited ops instead of counting per congruence class. Instances: 188 loops; operations min 19, median 54, mean 125, max 2,654; resource-limited ops median 4, max 221. Solved with Gurobi 8.0. Fastest configuration: under 16 minutes total, proven optimal II on 179 loops.
- Ayala, Benabid, Artigues, Hanen, "The resource-constrained modulo scheduling problem: an experimental study", Computational Optimization and Applications 54(3), 2013 (abstract only: https://ideas.repec.org/a/spr/coopap/v54y2013i3p645-673.html): decomposed software pipelining to get a retiming, then a reduced ILP; gap below 1 % on most instances. A companion (HAL hal-00538821, search snippet only) states that the two practitioner ILPs have equivalent LP relaxations and that Dantzig-Wolfe variants are stronger for non-binary resource requirements.

Fit: Moovac's pairwise overlap variables are designed for a handful of limited ops and would be quadratic for hundreds of ops per class at capacity 6 or 12. The Eichenberger-Davidson counting form is the right shape for us. Neither handles modes or a memory resource.

**In CP-SAT 9.15?** No periodic or modulo machinery. It is a model reformulation plus an outer loop.

**Own reasoning: which periodic unit is viable at 16 rounds.** Using the brief's numbers: capacity floor 1,271 over 32 groups is about 39.7 cycles of machine time per group; one group's critical path is 264; best known is 1,314.
- A pipeline that starts one identically-shaped group every p cycles has makespan 31p + L with p >= 39.7 and L >= 264, so at least 1,495. That is worse than the incumbent. The start-up and tail triangles waste more than steady state saves, because only about 6.6 groups would be in flight out of 32.
- Batches of k groups: makespan is roughly (32 - k) x 39.7 + L_k, so beating 1,314 needs L_k <= 43 + 39.7k together with L_k >= 264, hence k >= 6, and each batch must have an almost rectangular resource footprint.
- The round is the natural period: about 79 cycles of capacity per round against 16.5 cycles of critical path per round. That is modulo scheduling with II near 80 and a loop-carried dependence per group. Rounds are not identical (tree depth changes the access encoding), so it is only approximately periodic.
- The fluid results then suggest the 43-cycle gap at 16 rounds (3.4 %) is start-up and tail plus the memory limit. The 20 to 30 % gap on 2-round instances is the same boundary effect at a small horizon.

**Benefit.** (i) Schedules: moderate to high at 12K ops, where nothing currently improves the greedy start; a template gives a structurally different incumbent. (ii) Bounds: none directly, but it reframes the question as "how large must start-up and tail be", which a small LP can answer (experiment 2).

### A3. Column generation / Dantzig-Wolfe over group patterns

**What it is.** Column = one group's complete schedule (with modes) at an absolute time; master = one convexity row for the aggregated 32 groups plus capacity rows per cycle and class plus memory rows; pricing = single-group scheduling with a dual price per cycle. The master LP value equals the Lagrangian dual from relaxing the capacity rows.

**What the literature says about the bound.** Kolter, Grunow, Kolisch, "On Branch-and-Price for Project Scheduling", arXiv 2501.04563, January 2025 (full text of the relevant sections: https://arxiv.org/abs/2501.04563) answers this question directly:
- Proposition 2: with one mode, the precedence-only pricing problem in the disaggregated-precedence time-indexed form has the integrality property, so the master LP equals the compact LP (z_LMP = z_PDDT-LP). Their experiment confirms identical bounds, with column generation on average 77 times slower. Adding cutting planes to the compact LP gives bounds on average 6.81 % stronger than the master LP.
- Multi-mode, no resources in pricing: the integrality property fails in theory, but the master LP is only 0.42 % stronger on average and on average 26 times slower. The aggregated compact LP with cuts was more than 171 times faster.
- Resource-constrained pricing (possible only with dedicated resources per block) is not integral and does give substantially stronger bounds, but pricing is NP-hard; the cited branch-and-price has difficulty with 50 activities.
- The master is highly degenerate, which slows convergence. They note that identical pricing problems can be aggregated to break symmetry, which is our case.
- Their summary table lists Möhring et al.-style and Drexl-Kimms-style decompositions as z_LMP = z_PDDT-LP.

Möhring, Schulz, Stork, Uetz, "Solving Project Scheduling Problems by Minimum Cut Computations", Management Science 49(3), 2003 (abstract only: https://ideas.repec.org/a/inm/ormnsc/v49y2003i3p330-350.html): Lagrangian relaxation of the time-indexed IP, solved as a minimum cut, with relaxation-based list scheduling. Its integrality-property status is taken from Kolter et al., not from the paper itself.

**Fit to our problem.** Our groups share all resources, so pricing is a multi-mode, precedence-only problem. Expect the group-pattern master to beat the disaggregated time-indexed LP by well under 1 %. With unit durations and capacities 6 and 12 that LP is itself close to max(capacity floor, critical path). The memory row is a further coupling row in the master and does not change the pricing structure. A per-cycle configuration LP (column = multiset of op classes in one cycle) also collapses to the capacity floor when every demand is 1 or 8.

**In CP-SAT 9.15?** No column generation.

**Benefit.** (ii) Bounds: essentially none; do not build this for the bound. (i) Schedules: the restricted master, used as a heuristic over a small pool of templates, is the same idea as TCTR and is worth trying (experiment 1).

**Route.** Outer loop with an LP solver if anyone insists on the bound. The Möhring min-cut Lagrangian is a far cheaper way to reach the same number.

### A4. Logic-based Benders decomposition (LBBD)

**What it is.** A master fixes some variables (assignments, modes); a subproblem, usually CP, schedules under them; the subproblem's inference dual yields a cut. Hooker's planning-and-scheduling version uses a MILP master assigning jobs to facilities and one cumulative CP subproblem per facility.

Source: Hooker, "Logic-Based Benders Decomposition for Large-Scale Optimization", 2019 survey (full text: https://ar5iv.labs.arxiv.org/html/1910.11944). Also Hooker, "Planning and Scheduling by Logic-Based Benders Decomposition", Operations Research 55(3), 2007 (abstract only: https://ideas.repec.org/a/inm/oropre/v55y2007i3p588-602.html). Points from the survey:
- Makespan cut: M_i >= M_i* (1 - sum over a reduced job set of (1 - x_ij)), strengthened by removing jobs and re-solving; plus "analytical" cuts.
- The survey states that analytical cuts are weak for cumulative scheduling and more effective for disjunctive scheduling.
- Success "often depends on" a relaxation of the subproblem in the master; for cumulative this is an energy inequality of the form M >= t + (1/C) sum p c x.
- The benefit comes from the subproblem decoupling into independent facilities. With no natural decoupling, the horizon is cut into segments and the master assigns jobs to segments.
- Reported effect: orders of magnitude over MILP and CP on assignment-plus-scheduling; one problem where plain CP beat LBBD.
- Branch-and-check and its extended form are described as close to conflict-driven clause learning.

LBBD for multi-mode RCPSP with a mode-assignment master in 2022 to 2024: **not found** as a verified paper. Search snippets mention a generalised Benders method for MRCPSP (Ramos et al.) and an older Maniezzo-Mingozzi heuristic with a mode-assignment master; neither was opened.

**In CP-SAT 9.15?** Not as an outer decomposition. Clause learning with lazy explanations plays the branch-and-check role internally [from memory, unverified].

**Fit to our problem.** Master = mode choices (VALU shift, const, select); subproblem = one single-mode schedule. Problems:
- The subproblem does not decouple, because all 32 groups share everything.
- The natural master relaxation is the capacity floor over modes, which is already the bound CP-SAT sits on.
- A makespan nogood only excludes one mode vector out of thousands of near-equivalent ones.
- Hooker's own remark about weak cumulative cuts applies fully.

The only variant with a chance is the segmented-horizon form: master allocates per-segment counts of shifted ops, subproblem solves each segment as a window. That is the existing windowed LNS plus a coordinating master; it gives no bound, since nothing proves a segment allocation is necessary.

**Benefit.** (i) Low to moderate, as a mode-first heuristic. (ii) None.

### A5. Symmetry beyond static lex constraints

**What CP-SAT 9.15 does.**
- `symmetry_level` (default 2): level 1 detects symmetries in presolve and fixes some Booleans; level 2 adds dynamic symmetry breaking during search; level 3 extends detection to very large models; level 4 tries to break as much as possible in presolve. Source: `sat_parameters.proto` at tag v9.15, comments read through a fetch-tool summary (source code, summarised).
- Other parameters: `use_symmetry_in_lp` (default false; folds each orbit into one LP variable), `keep_symmetry_in_presolve` (default false, experimental), `symmetry_detection_deterministic_time_limit` (default 1.0). All four names confirmed in the installed `sat_parameters.pb.h` (source code).
- Dynamic part: `symmetry.h` (installed header, read directly) says `SymmetryPropagator` implements "more or less" Devriendt, Bogaerts, De Cat, Denecker, Mears, "Symmetry propagation: Improved Dynamic Symmetry Breaking in SAT" (ICTAI 2012). When a literal is propagated and its image under a generator is not, the image is propagated using the permuted reason. The header calls the method heuristic and not exhaustive, and it works on Boolean literals and generators only.
- Orbitopes: `symmetry_util.h` (installed, read directly) has `BasicOrbitopeExtraction`, citing Kaibel and Pfetsch's packing and partitioning orbitopes. It requires generators made only of transpositions listed in canonical order, and detects at most one orbitope. The Schreier-Sims stabiliser-chain inequalities (Salvagnin) are an unimplemented stub (`TransformToGeneratorOfStabilizer` is empty).
- `cp_model_symmetries.cc` at v9.15 (source code, read through a fetch-tool summary, so treat details as second-hand): interval, no_overlap and cumulative constraints are encoded in the symmetry graph; an unsupported constraint type aborts detection for the whole model; presolve fixes Booleans from orbits or the orbitope; column-ordering inequalities on an orbitope are added only at level 4, behind a comment that these transformations seem to hurt more than they help.

So CP-SAT has automatic detection, Boolean orbit/orbitope fixing, and symmetry propagation. It does not have SBDS, SBDD, LDSB, symmetric explanation learning, or SBDS-1UIP.

**Literature.**
- Chu, Stuckey, Garcia de la Banda, Mears, "Symmetries and Lazy Clause Generation", IJCAI 2011 (abstract only: https://ijcai.org/proceedings/2011/94), and "Symmetries, almost symmetries, and lazy clause generation", Constraints 19(4), 2014 (abstract only: https://link.springer.com/article/10.1007/s10601-014-9163-9). SBDS-1UIP posts symmetric images of 1UIP nogoods; because the nogoods are more precise it exploits symmetries no earlier method could, and extends to almost-symmetries.
- Devriendt, Bogaerts, Bruynooghe, "Symmetric explanation learning: effective dynamic symmetry handling for SAT", SAT 2017 (abstract only: https://lirias.kuleuven.be/retrieve/440155 via search snippet). Learns symmetric images of explanation clauses only when unit or conflicting; described as the first symmetric-learning scheme competitive with static breaking.
- Mears, Garcia de la Banda, Demoen, Wallace, "Lightweight dynamic symmetry breaking", Constraints 19, 2014 (abstract only: https://link.springer.com/doi/10.1007/s10601-013-9154-2). Dynamic, so it does not interfere with the search heuristic; restricted to compactly representable symmetry classes.
- Pfetsch, Rehn, "A computational comparison of symmetry handling methods for mixed integer programs", Mathematical Programming Computation (abstract only: https://optimization-online.org/2015/11/5209/). On MIPLIB, isomorphism pruning, orbital fixing or symmetry-breaking inequalities give about 15 % speed-up and more instances solved.
- Margot, and Ostrowski et al. "Orbital branching": not opened; nothing claimed.

**Fit to our problem.** The automatic machinery sees S_32 as permutations of whole blocks of about 375 start variables plus mode literals. The generators that swap two groups are involutions with very large support; `symmetry.h` itself recommends supports as small as possible, so symmetry propagation will be expensive and weak here. If the user's lex constraints are in the model, the symmetry is no longer present in the proto, so the two mechanisms cannot be combined.

**Own reasoning: a stronger, conditional exchange dominance at round boundaries.** The brief notes that precedence chains tie a group's identity across rounds. That tie can be loosened:
- If groups were plain chains, a suffix swap shows there is always an optimal schedule with start(g, k) <= start(g+1, k) for every position k at once: if A's op k is before B's but A's op k+1 is after B's, exchange the two suffixes. That is 375 ordering constraints per adjacent pair, not one lex row.
- For DAG groups the swap works at any cut where it keeps every cross-cut edge satisfied. The round boundary is a narrow cut (the carried index and value vectors). If A finishes round r no later than B, but A starts round r+1 later than B, hand B's round-(r+1)-onward time slots and modes to A and vice versa.
- The live intervals of the boundary values are re-paired, but the multiset of live cycles is unchanged, so memory usage per cycle is unchanged.
- Consequence: the pairing of round-r prefixes with round-(r+1) suffixes can be taken first-in-first-out, so the usable symmetry is close to a separate S_32 per round. This is the slot-relabelling move of Zheng et al.'s Theorem 1.
- Caveat: with two carried values defined at different cycles, a swap is valid only when both cross edges stay satisfied, so this is a conditional dominance in Chu and Stuckey's sense. It must be proved (Z3 or a checker) before it is posted as a constraint.

**Benefit.** (i) Schedules: little. (ii) Proofs: symmetry handling shrinks the tree but does not raise a relaxation bound that sits at the capacity floor, so do not expect it to close the gap. Its value is in window proofs and in making group-level neighbourhoods and templates legitimate.

**Route.** Parameters (`symmetry_level` 3 or 4, `use_symmetry_in_lp`) for the automatic path with the manual lex rows removed; model rows for the exchange dominance; SBDS-1UIP or SEL would need an OR-Tools fork.

### A6. Dominance breaking

- Chu, Stuckey, "A Generic Method for Identifying and Exploiting Dominance Relations", CP 2012 (full text: https://people.eng.unimelb.edu.au/pstuckey/papers/dominance.pdf); journal version "Dominance breaking constraints", Constraints 20, 2015 (abstract only: https://link.springer.com/doi/10.1007/s10601-014-9173-7). Choose a mapping sigma on assignments (for makespan problems the paper suggests shifting items earlier), derive the condition under which sigma preserves feasibility and does not worsen the objective, and post its negation. Symmetry and conditional symmetry breaking are special cases.
- Their RCPSP result is a warning for a clause-learning solver. On non-trivial J60 instances with the left-shift rule ("each task starts at 0 or when another ends"), geometric means in Chuffed were: 358.95 s with no dominance and no learning, 279.74 s with dominance only, 4.07 s with learning only, and 32.84 s with both. The authors say the dominances learning derives dynamically are stronger than the static ones, and the static constraints interfere.
- Chu, Stuckey, "Dominance driven search", CP 2013: only the DOI appeared in a result list; not opened.
- Lee, Zhong, "Automatic Dominance Breaking for a Class of Constraint Optimization Problems", IJCAI 2020 (abstract only: https://www.ijcai.org/proceedings/2020/166). Generates dominance-breaking nogoods of length 2 to 4 by solving an auxiliary CSP; needs separable or sub/supermodular objectives; reports up to three orders of magnitude over manual dominance breaking on knapsack variants, concert-hall scheduling and max-cut. A makespan objective over start variables is outside that class. "Exploiting functional constraints in automatic dominance breaking" (AIJ 2023): **not found** under that title; a 2023 journal preprint by the same authors appeared in results but was not opened.
- Mears, Garcia de la Banda, "Towards Automatic Dominance Breaking for Constraint Optimization Problems", IJCAI 2015 (seen only as a search hit: https://ijcai.org/proceedings/2015/57).
- Demeulemeester and Herroelen: the 1992 Management Science branch-and-bound combines bounds with dominance pruning (abstract only: https://ideas.repec.org/a/inm/ormnsc/v38y1992i12p1803-1818.html). The cutset dominance rule is attributed to their 1997 paper by a secondary source, which also notes it stores all cutsets and is memory-hungry. Exact rule statements **not found**.
- Cohen, Kress, Nehrke, Pesch, Shtern, "Two Communities, One Problem: Branch-and-Bound vs. A* for Resource-Constrained Project Scheduling", arXiv 2611.14426, November 2026 (abstract only: https://arxiv.org/abs/2611.14426). With the same lower bound and best-first selection, graph-based A* with duplicate-state detection and extra pruning rules proves optimality on all 480 instances of their benchmark; best-first branch-and-bound proves 445. The conclusion is that detecting equivalent states reached by different decision orders is a major factor. This is cutset dominance in modern form.
- Hu's level rule and Coffman-Graham for unit-time jobs on identical machines: [from memory, unverified]. They are optimal only for special cases (forests; two machines) and do not carry over to five typed resources with modes.

**In CP-SAT 9.15?** No dominance-breaking facility found in the headers. Left-shift dominance is implicit through the objective and `push_all_tasks_toward_start` (already tried per the brief).

**Fit.** For unit-duration tasks the useful dominances are: left-shift (probably counter-productive to post, per Chu and Stuckey's RCPSP row); the suffix-exchange in A5; and state dominance (same set of completed ops and same live set means same subproblem), which is what DIDP and A* exploit and CP-SAT cannot.

**Benefit.** (i) None. (ii) Small, and possibly negative for static left-shift rows.

---

## Topic B - Newer exact paradigms and learned search

### B1. Domain-independent dynamic programming, SALBP-1, bin packing with precedences

**What it is.** DIDP models a problem as a state-transition system in DyPDL (state variables, transitions with preconditions and costs, dominance between states, dual-bound expressions) and solves it by heuristic state-space search; the flagship solver is complete anytime beam search (CABS).

Source: Kuroiwa, Beck, "Domain-Independent Dynamic Programming", arXiv 2401.13883v4 (full text, experiments section: https://arxiv.org/pdf/2401.13883v4). Setup: Gurobi 11.0.2 and IBM CP Optimizer 22.1.0, single thread, 8 GB, 30 minutes. CP-SAT is not in the comparison.
- SALBP-1 (2,100 instances): MIP 1,431 solved, CP 1,584, CABS 1,802 with zero memory-outs.
- Bin packing (1,615 instances): MIP 1,160, CP 1,234, CABS 1,167; CP has the highest coverage.
- CABS has the highest coverage among the DIDP solvers in every class.

Earlier ICAPS-era version, Kuroiwa and Beck, arXiv 2211.14409 (full text: https://ar5iv.labs.arxiv.org/html/2211.14409): the SALBP-1 model has state (set of unscheduled tasks U, remaining station time r), transitions "assign task" and "open station" (allowed only when nothing fits), dominance on r, and the dual bound max of three bin-packing bounds from Scholl and Klein. With that solver (CAASDy): MIP 1,360, CP 1,585, DP 1,652; on the "very large" set MIP 0, CP 155, DP 204.

Parallel version: Kuroiwa, Beck, "Parallel Beam Search Algorithms for Domain-Independent Dynamic Programming", AAAI 2024 (abstract only: https://ojs.aaai.org/index.php/AAAI/article/view/30062): outperforms multi-thread MIP and CP solvers on four of six problem classes.

**SALBP exact literature.**
- Scholl, Klein, "SALOME: A Bidirectional Branch-and-Bound Procedure for Assembly Line Balancing", INFORMS J. Computing 9(4), 1997 (abstract only: https://pubsonline.informs.org/doi/fpi/10.1287/ijoc.9.4.319): local-lower-bound branching, bidirectional branching, bounding and dominance rules.
- Sewell, Jacobson, "A Branch, Bound, and Remember Algorithm for the Simple Assembly Line Balancing Problem", INFORMS J. Computing 2012 (abstract only: https://pubsonline.informs.org/doi/10.1287/ijoc.1110.0462): branch-and-bound with memory to eliminate redundant subproblems; verifies optimality on the combined Hoffmann, Talbot and Scholl benchmarks in under half a second per problem on average; the previous best algorithm solved 257 of 269.
- Morrison, Sewell, Jacobson, EJOR 236(2), 2014 (abstract only: https://ideas.repec.org/a/eee/ejores/v236y2014i2p403-409.html): cyclic best-first search plus a memory-saving backtracking rule; exact solutions for 86 % of the unsolved problems in the 2013 dataset.

**Typed and lagged versions.**
- Dell'Amico, Díaz Díaz, Iori, "The Bin Packing Problem with Precedence Constraints", Operations Research 60(6), 2012 (abstract only: https://ideas.repec.org/a/inm/oropre/v60y2012i6p1491-1504.html): first exact treatment; reduction criteria, a large set of lower bounds, variable neighbourhood search, branch-and-bound.
- Pereira, "Procedures for the bin packing problem with precedence constraints", EJOR 250(3), 2016 (abstract only: https://ideas.repec.org/a/eee/ejores/v250y2016i3p794-806.html): DP-based heuristic and modified enumeration with new lower bounds and dominance rules.
- Kramer, Dell'Amico, Iori, IJPR 55(21), 2017 (abstract only: https://ideas.repec.org/a/taf/tprsxx/v55y2017i21p6288-6304.html): bin packing with generalised precedences carrying heterogeneous non-negative integer values, which also models SALBP-1; iterated local search with adapted lower bounds.

With unit weights and lags in {0, 1, 2}, Kramer et al.'s problem is our problem restricted to one op class and one mode. I found no version with several typed capacities per bin, alternative modes, or a storage resource.

**In CP-SAT 9.15?** No. didp-rs is a separate Rust library.

**Fit to our problem.** A DyPDL model is easy to write: state = set of scheduled ops plus residual capacities of the open cycle; transition = schedule a ready op in some mode, or close the cycle. Three obstacles:
- The only cheap dual bound is ceil(remaining work / capacity) per class, the capacity floor again, so almost every state ties on f and A*-style proof never terminates.
- The state space is the set of down-closed subsets of a 12K-op DAG with antichains hundreds wide. The solved SALBP-1 "very large" instances have 1,000 tasks with much narrower structure [from memory, unverified, for the 1,000 figure].
- Memory-limit feasibility depends on the live set; it is a function of the state, so it can be modelled, but it removes the simple resource-dominance that makes SALBP states prune.

One real attraction: a DP state can be canonicalised as a multiset of per-group progress states, which removes the 32-group symmetry exactly, with no lex constraints.

**Benefit.** (i) Moderate: CABS is a principled beam search with duplicate detection and would be an independent source of incumbents beside the greedy scheduler. (ii) None beyond the capacity floor.

### B2. Decision-diagram bounds and branch-and-bound

**What it is.** Relaxed decision diagrams of bounded width merge DP states to give dual bounds; restricted diagrams give primal solutions; branch-and-bound proceeds on exact cut sets. Peel-and-bound reuses a peeled sub-diagram as the starting diagram for later iterations.

Sources: Rudich, Cappart, Rousseau, "Peel-and-Bound", CP 2022 (abstract only: https://arxiv.org/pdf/2205.05216), tighter bounds than branch-and-bound with the same propagators on the sequence ordering problem; their JAIR 2023 extension is reported to outperform ddo on TSPTW (search snippet). Kuroiwa and Beck (above, full text) compare CABS with ddo on TSPTW-M and talent scheduling: ddo ran out of 8 GB on every unsolved TSPTW-M instance, and CABS has higher coverage on both. A search for decision diagrams applied to cumulative scheduling or RCPSP in 2022 to 2024 returned only single-machine and multi-machine sequencing work (van den Bogaerdt and de Weerdt 2018; Hooker's Lagrangian bounds on relaxed diagrams for job sequencing) and Cire and van Hoeve's disjunctive MDD propagation. Bergman, Cire, van Hoeve, Hooker (INFORMS JoC 2016) and Cire and van Hoeve (OR 2013): not opened.

**In CP-SAT 9.15?** No.

**Assessment.** Not applicable in this regime. The published successes are sequencing problems where a layer is "the next job" and merged states still carry useful bound information. For a cumulative problem with 20 or more ops per cycle, either layers are ops (12K layers, and merging down-sets destroys all precedence information, leaving the capacity floor) or layers are cycles (the domain of a layer is every feasible subset of ready ops). No published evidence was found that relaxed diagrams give useful makespan bounds for high-capacity cumulative scheduling.

### B3. Core-guided optimisation; SAT and SMT encodings

**Core-guided search.** Gange, Berg, Demirović, Stuckey, "Core-Guided and Core-Boosted Search for CP", CPAIOR 2020 (full text: https://cs.helsinki.fi/u/jezberg/papers/CPAIOR2020_Gange_Berg_Demirovic_Stuckey.pdf). Extends MaxSAT core-guided search to integer variables in the lazy-clause-generation solver geas. Benchmarks are all MiniZinc Challenge 2015 to 2018 optimisation models with a linear objective (48 models, 249 instances, 600 s). Findings: branch-and-bound is slightly better than basic core-guided search; variable-based reformulation makes core-guided better than branch-and-bound; core-boosted search (10 % of the time core-guided, then branch-and-bound) is best overall; the two are "fairly orthogonal"; pure core-guided is much weaker as an anytime method.

**In CP-SAT 9.15?** Yes. `optimize_with_core` (default false; core-based approach "as in MaxSAT" to raise the lower bound), `optimize_with_max_hs`, `core_minimization_level` (default 2), `cover_optimization` (default true, "core exhaustion"), `max_sat_assumption_order`, plus `use_objective_lb_search` and `use_objective_shaving_search`. Source: `sat_parameters.proto` at v9.15, comments read through a fetch-tool summary (source code, summarised). The brief lists subsolver lists as already tried.

**Fit.** Core-guided search pays off when the objective is a sum of many terms, where cores identify small subsets that cannot all be at their lower bound. Our objective is one makespan variable, so core-guided search reduces to bottom-up search on that variable, which needs a proof that makespan = floor + k is infeasible. That is the proof CP-SAT cannot produce. No benefit expected for (i) or (ii).

**SAT / SMT encodings of RCPSP.**
- Horbach, "A Boolean satisfiability approach to the resource-constrained project scheduling problem", Annals of OR 181, 2010 (abstract only: https://ideas.repec.org/a/spr/annopr/v181y2010i1p89-10710.1007-s10479-010-0693-2.html): RCPSP as SAT with a domain-adapted solver; closed several open PSPLIB instances.
- Bofill, Coll, Suy, Villaret, "Compact MDDs for Pseudo-Boolean Constraints with At-Most-One Relations in Resource-Constrained Scheduling Problems", IJCAI 2017 (full text: https://ijcai.org/proceedings/2017/78). Activities on a common path of the precedence closure cannot overlap, so a minimum path cover gives at-most-one groups inside each per-time resource constraint, which shrinks its decision-diagram encoding. Reported: Boolean variables 281k to 45k on MRCPSP MMLIB50 and 3,450k to 360k on RCPSP/t j120; MMLIB50 mean time 108.2 s against 143.1 s for the plain encoding and 162.4 s for a lazy-clause-generation solver; timeouts 88 against 105 and 115.
- Bofill, Coll, Suy, Villaret, "SMT encodings for Resource-Constrained Project Scheduling Problems", Computers & Industrial Engineering 149, 2020 (abstract only: https://dblp.org/rec/journals/candie/BofillCSV20.html): replacing linear-arithmetic resource constraints by SAT encodings is described as crucial for efficiency.

**Fit.** With unit durations, each (cycle, class) row is a pure cardinality constraint (at most 6, 12, 2, 2 or 1) over time-indexed Booleans; with VALU shift it is a pseudo-Boolean row with coefficients 1 and 8. In CP-SAT that is a linear constraint over Booleans, which is an alternative to `AddCumulative` available by reformulation. It gives the LP relaxation and clause learning direct access to "op i at cycle t" literals. Costs and limits:
- The Boolean count is ops x window width, so it is plausible for the 1,000 to 2,000-op instances and for windows, not for 12K ops over a 1,300-cycle horizon.
- The Bofill at-most-one trick transfers (ops on one precedence chain never share a cycle) but matters less when coefficients are already unit.
- These papers improve solve time on instances with 30 to 120 activities and capacities well below ours. None claims better bounds in a high-parallelism regime.

**Benefit.** (i) Unknown, perhaps small on mid-size instances. (ii) None expected.

### B4. Learned and adaptive LNS; neighbourhood design for scheduling

**Learned LNS for MIP (abstract only for all four).**
- Song, Lanka, Yue, Dilkina, NeurIPS 2020 (https://papers.nips.cc/paper/2020/hash/e769e03a9d329b2e864b4bf4ff54ff39-Abstract.html): decomposition-based LNS with a neighbourhood selector learned by imitation or reinforcement learning; reported to beat Gurobi in wall-clock time.
- Sonnerat et al., arXiv 2107.10201 (https://arxiv.org/abs/2107.10201): learned initial solution plus imitation-learned destroy policy; 2x to 37.8x better average primal gap than the best baseline on three datasets at large running times.
- Wu, Song, Cao, Zhang, NeurIPS 2021 (https://neurips.cc/virtual/2021/poster/28200): reinforcement-learned destroy policy factorised per variable; better solutions than SCIP in less time on four problem classes.
- Huang, Ferber, Tian, Dilkina, Steiner, ICML 2023 (https://proceedings.mlr.press/v202/huang23g.html): contrastive learning from a slow expert heuristic.

All train on a distribution of instances of one family. We have one family with a handful of sizes and a strong structural prior, so a learned policy has little to add over a hand-built neighbourhood set with adaptive selection. A search for RL-chosen neighbourhoods on top of CP-SAT for RCPSP or job shop found nothing verified.

**Adaptive selection (abstract only).**
- Hendel, "Adaptive large neighborhood search for mixed integer programming", Mathematical Programming Computation 14, 2022 (https://link.springer.com/article/10.1007/s12532-021-00209-7): eight LNS heuristics under a multi-armed-bandit selector with an LNS-specific reward, in SCIP.
- Laborie, Godard, "Self-adapting large neighborhood search: application to single-mode scheduling problems", MISTA 2007: original **not found**; known only through citations in Thomas and Schaus, CPAIOR 2018, which revisits its weight-update rule.

**Neighbourhood designs the scheduling-LNS literature found effective.**
- Partial-order-schedule relaxation. Godard, Laborie, Nuijten, "Randomized Large Neighborhood Search for Cumulative Scheduling", ICAPS 2005 (abstract only: https://aaai.org/?p=79652). The key step is deriving a partial-order schedule from a fixed-start-time schedule, so that unrelaxed tasks keep their relative order but can still slide. Best known results at the time on cumulative job shop benchmarks.
- Iterative flattening. Cesta, Oddi, Smith, AAAI 2000 (abstract only: https://www.ri.cmu.edu/publications/iterative-flattening-a-scalable-method-for-solving-multi-capacity-scheduling-problems): alternately remove some resource-ordering decisions and re-flatten; 100 to 900 activities, within 10 % of computed upper bounds. Later variants by Oddi, Cesta, Policella, Smith (J. Intelligent Manufacturing 2010) compare relaxation strategies.
- Forward-backward improvement (double justification). Valls, Ballestín, Quintanilla, "Justification and RCPSP: A technique that pays", EJOR 165(2), 2005 (abstract only: https://ideas.repec.org/a/eee/ejores/v165y2005i2p375-386.html): right-shift everything, then left-shift everything. Added to 22 algorithms on the j120 set; fifteen of the resulting algorithms outperform seven of the best heuristics that do not use it, generally at no extra time.
- Mode-reassignment neighbourhoods for multi-mode RCPSP: no source opened; [from memory, unverified] the usual design frees modes of a subset together with their start times.
- What CP-SAT's built-in scheduling LNS contains is the other agent's topic.

**Fit to our problem.** The brief's observation that nearly all window gains come from re-deciding modes says the neighbourhood should be defined on the mode dimension:
1. Keep time windows, and relax the frozen part to a partial order (Godard et al.) rather than fixed start times, so a one-cycle gain inside a window propagates without the "everything after moves as one block" restriction.
2. Group-subset neighbourhood: free k of the 32 groups over the whole horizon and fix the others' capacity usage. This is orthogonal to time windows. Because the groups are identical, a pattern that improves one group can be copied to the others, which is the template idea of A2.
3. Mode-budget neighbourhood: fix every op's cycle to within plus or minus 1 or 2 and free all modes globally. This is a large but shallow model aimed at the dimension where gains occur.
4. Double justification of the incumbent between LNS rounds. It costs two greedy passes. The memory limit makes the right-shift pass non-trivial, so it must run through the existing budget-aware scheduler.
5. Choose among these with a bandit reward of improvement per deterministic second (Hendel; Laborie and Godard); no learning stack is needed.

**Benefit.** (i) High relative to everything else in this part: LNS is the only thing that currently improves incumbents, and none of items 1 to 3 is what a generic solver neighbourhood does. (ii) None.

**Route.** External LNS loop through the solution-hint API, as the current windowed LNS already does.

---

## Ranked ideas and cheap experiments

1. **Group-round templates with replay (TCTR-style restricted master).** Treat each (group, round) unit of about 23 ops (my estimate: 12K ops / 512 units) as a macro-task with a few alternative internal shapes and mode choices. Decision variables are 512 offsets plus template choices, with capacity rows per cycle, first-in-first-out linking between rounds, and the memory row. *Experiment:* take the best 2-round and 4-round schedules already on disk, cut them into group-round units, and count distinct relative shapes (resource profile and modes after translating to the unit's first cycle). If a handful of shapes covers most units, build the small offset model for 4 rounds and compare its optimum with the existing incumbent. Pure analysis plus one small solve.

2. **Start-up and tail bound from a time-bucketed fluid LP.** Variables: fraction of each op class (grouped by ASAP/ALAP level and mode) executed in each cycle or bucket, with capacity rows and aggregated precedence on levels. This measures how much of the 43-cycle gap at 16 rounds, and of the 20 to 30 % gap at 2 rounds, is unavoidable ramp-up and ramp-down. *Experiment:* one LP per instance, a few thousand rows, in HiGHS. If it returns the capacity floor plus only 2 or 3, the remaining gap is in the memory limit and integrality, and no decomposition bound will find it either.

3. **Mode-centred and group-centred LNS neighbourhoods with bandit selection** (items 1 to 5 in B4). *Experiment:* on one mid-size instance with a fixed deterministic-time budget, compare the current window neighbourhood against (a) free 2 of 32 groups over the whole horizon and (b) free all modes with starts confined to plus or minus 2 cycles; record improvement per deterministic second for each.

4. **Round-boundary suffix-exchange dominance.** *Experiment, before any solver change:* write a checker that, on existing schedules, finds pairs of groups whose round-r completion order and round-(r+1) start order cross, tests whether the swap is feasible (cross edges and memory), and counts how often it is not. Then state the condition for Z3. Only if the dominance is proved, replace whole-group lex with per-round ordering on a 2-round instance and measure window-proof times with the existing ablation harness.

5. **CABS beam search over multiset states (didp-rs) as a second incumbent generator.** *Experiment:* model one class-restricted, single-mode 2-round instance in DyPDL with state "multiset of per-group progress" and the capacity-floor dual bound; compare the first few CABS incumbents with the greedy scheduler. Stop if it does not match greedy within minutes.

Not recommended: Dantzig-Wolfe for the bound (A3), classical LBBD (A4), decision diagrams (B2), core-guided search (B3), learned LNS policies (B4), posting static left-shift dominance rows (A6).
