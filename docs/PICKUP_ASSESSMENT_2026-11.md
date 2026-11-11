# Pickup assessment (November 2026)

State of the project after a nine-month gap, measured on branch `pickup/assessment`
(from `origin/design/visitor-trait-pattern`, 7d3272b). Every number below was produced in
this session unless marked otherwise. Commands are in the last section.

## 1. Where things stand

| Item | Result |
|------|--------|
| 16-round kernel, greedy, before this session | Does not build (aborts past 5 rounds) |
| 16-round kernel, greedy, after the liveness fix | **1994 cycles**, values and indices correct on 3 seeds |
| Same kernel on the earliest committed simulator (`dceff11:problem.py`) | 1994 cycles, correct |
| 16-round kernel, whole-program CP-SAT, 600 s | 1994 (returns the greedy hint unchanged), bound 1,271 |
| Rust tests (`--features cpsat`) | 334 passed, 11 ignored |
| Best documented public results | 1,023 to 1,160 (hand-tuned list schedulers) |
| Community leaderboard (no write-ups) | 863 |

"TBD" in CLAUDE.md for 16 rounds was literal: the pipeline had never produced a full kernel.
The 4-round figures (435 to 485) are not comparable to the 1,487 target, and the "304.6x
speedup" in CLAUDE.md divides a 16-round baseline by a 4-round result.

### 1.1 The bug that blocked 16 rounds

`ssa_greedy/mod.rs` frees a value when its last reader is scheduled. Gather rounds use
vector groups: an address vector decomposed into 8 lanes, and 8 loaded lanes assembled into
a vector. Readers name only one side of each group, so the address vector (8 words) and the
loaded lanes (8 words) were never released. That is 512 phantom words per gather round:

| Rounds | Gather rounds | Before | After |
|--------|---------------|--------|-------|
| 5 | 1 | 572 | 572 |
| 6 | 2 | 870 (throttled) | 716 |
| 8 | 4 | abort | 1004 |
| 16 | 8 | abort | 2007 via CLI, 1994 via the test harness |

The fix adds a lane/alias release map (`liveness_release_vregs`). The regression test
`test_vector_group_vregs_released_after_last_use` fails without it (peak 4808) and passes
with it. The CLI and harness differ by 13 cycles; the CLI prints `n_nodes=1023` for
`forest_height=10` where the harness passes 2047. Not yet investigated.

### 1.2 Whole-program CP-SAT on 16 rounds

`uv run generate_ssa_kernel.py --rounds 16 --cpsat-time-limit 600 --validate -q`:

- Model: 111,276 variables, 80,337 intervals (65,593 optional), 6 cumulatives, about
  470,000 linear constraints before presolve.
- The first solution arrived at 193 s and was the greedy hint (1994). No improvement in the
  remaining time.
- Final status FEASIBLE, 1994 cycles, bound 1,271. The bound is the capacity floor of
  section 1.3, supplied by the model's own redundant capacity constraint; the search did
  not move it.
- The kernel validates.

So at full size the whole-program model currently adds nothing over the greedy schedule in
ten minutes. A longer run was not tried.

### 1.3 Where the 1994 cycles go

Rounds execute strictly one after another (each round starts one cycle after the previous
one ends).

| Rounds | Cycles each | Vector ALU | Load | Scalar ALU |
|--------|-------------|-----------|------|-----------|
| Gather (4 to 10, 15) | 140 to 144 | about 61% | 85 to 100% | 0 to 3% |
| Select (0 to 3, 11 to 14) | 80 to 137 | 81 to 96% | 0% | 43 to 65% |

A gather round cannot beat 128 cycles (256 loads at 2 per cycle), and while it runs the
vector ALU is one-third idle. Select rounds leave the load slots idle. The groups are
independent chains, so nothing but the round-by-round structure prevents some groups from
gathering while others hash.

Op counts for the 16-round kernel as emitted today:

| Engine | Ops | Floor (ops / capacity) |
|--------|-----|------------------------|
| Vector ALU | 8,783 (plus 723 shifted to scalar) | 1,464 |
| Load | 2,136 | 1,068 |
| Flow | 488 | 488 |
| Scalar ALU | 5,946 (162 native) | 496 |

With vector ops free to move to the scalar ALU (8 scalar ops each), the joint capacity
floor is about **1,270 cycles**: 9,506 vector-op equivalents over 7.5 per cycle. So of the
gap from 1,994 to the 1,487 target:

- about 720 cycles are schedule structure (1,994 against a 1,270 floor), and
- the floor itself is above the public results because the kernel has too many ops. The
  1,137-cycle public write-up reports about 6,563 vector ALU ops after offloading; an 863
  result implies at most about 6,470 vector-op equivalents and at most 1,726 loads, fewer
  than the 2,048 gather loads emitted here (inference from slot arithmetic, not from a
  write-up).

## 2. Design assessment

### What is sound

- SSA with explicit alternative encodings (ConstChoice, SelectChoice, vector-to-scalar
  shift) is the right representation. Leupers and Marwedel (1997), Unison (2019) and
  Eriksson and Kessler (2012) all model alternatives inside the scheduler this way.
- The liveness cumulative over def-to-last-use intervals matches Unison's implied register
  cumulative and Moccasin's retention intervals, the two formulations that scale best in
  the literature.
- Greedy first, solver second, with the greedy schedule as a hint.
- Using Z3 for the tightening theorems caught a real unsoundness (T7).

### What works against the problem

1. **The monolith is far past the published frontier.** The largest instances solved
   optimally in the literature are about 2,600 instructions for scheduling alone (Malik,
   McInnes, van Beek) and 946 for integrated allocation and scheduling (Unison). The
   16-round kernel has 17,417 ops, and each shiftable vector op adds eight scalar
   intervals. Proofs of optimality from this model should not be expected.
2. **The solver is configured so that it cannot report a gap.** `model.rs` sets
   `interleave_search = true` and `num_workers = 8` on a 12-core machine. Per the 9.15
   source (read by the review agent, parameters confirmed here), interleaved mode drops
   the probing, shaving and lower-bound workers, and 8 workers leave no slot for them
   either way. The extra cumulative propagators are only enabled inside those workers.
3. **Liveness is implemented four times** (greedy counter, `round_budget.rs`, the CP-SAT
   scratch model, `address_assign.rs`). Most open entries in BUGS.md, and the leak fixed
   here, are disagreements between those copies.
4. **Round barriers shape the schedule.** The greedy emits and schedules by round, so the
   hint CP-SAT starts from has no cross-round overlap, and the model must discover a
   completely different schedule shape from a serialized warm start.
5. **Op count was never the target.** Most effort went into the scheduler; the largest
   remaining lever is in emission.

## 3. Recommended direction

The thesis that survives contact with the literature is not "one solver call schedules the
program". It is **solvers at the levels where they are exact, with bounds connecting the
levels**.

1. **Encoding selection as a small exact problem.** Choose per-level tree access strategy
   (cached and selected, or gathered), select encoding, constant offload and shift counts
   to minimise the capacity floor `max(vector, scalar, load, flow)`. This is an integer
   program with tens of variables. Eriksson and Kessler balance resources in their
   objective the same way. Its optimum is a lower bound for that emission and tells you
   whether a schedule is worth solving for.
2. **A schedule shape that can reach the floor.** Remove the round barriers and stagger
   groups so gather and hash phases overlap (the public 1,137 solution runs about 20 groups
   one round apart). Because the groups are identical chains, the steady state is a
   periodic pattern, which is modulo scheduling. A template of one group-round is tens of
   ops and can be solved exactly (Eichenberger and Davidson 1997; Oppermann et al. 2019;
   Dai and Zhang 2019).
3. **CP-SAT as neighbourhood search, not monolith.** Relax a time window, keep the rest of
   the incumbent as precedences rather than fixed cycles (Godard, Laborie, Nuijten 2005;
   this is what CP-SAT's own `scheduling_time_window_lns` does), and choose windows at the
   ramp-up, the tail and liveness peaks.
4. **Report a gap on every run.** Capacity floor with modes, per-engine head/tail bounds
   (Rim and Jain 1994; Langevin and Cerny 1996), then destructive probes: ask for
   makespan at most M-1 with no objective and take UNSAT as the certificate (Klein and
   Scholl 1999; the same scheme Malik et al. use).

### Solver configuration experiments (cheap, measurable)

| Change | Expected mechanism | Measure |
|--------|-------------------|---------|
| `interleave_search = false`, workers = cores, add `objective_shaving_*`, `objective_lb_search_*` | Restores bound-proving workers and their propagators | `best_bound` at fixed wall time |
| Verify the hint is complete and feasible (`fix_variables_to_their_hinted_value` dry run) | An incomplete hint gets about 10 conflicts of guided search, then is dropped | "complete_hint" in the log; time to first solution |
| Makespan interval in each cumulative (as in OR-Tools `rcpsp_sat.py`) | Lets the LP relaxation and energy cuts see the makespan | Root bound |
| Group ordering as progressive constraints over cycles and modes (Kovács and Váncza 2006) | Removes group-permutation symmetry | Conflicts to target |
| Double justification on the greedy result (Valls et al. 2005) | Better incumbent, tighter ALAP | Greedy cycles |

None of these has published evidence on a model like this one; the review found only
source comments and Google's own worker configurations. Treat them as experiments.

## 4. Proof obligations

Ordered by value. The first two would have prevented most of BUGS.md.

1. **Dataflow replay checker (decision procedure, no solver).** Execute the emitted bundles
   with each scratch word tagged by the VReg that wrote it; at every read, require the tag
   to equal the SSA operand. Linear time, input-independent, and it names the first bad
   (cycle, address, op). It proves that scheduling, lowering and address assignment
   preserve the SSA dataflow for all inputs, where random-seed validation only samples.
   It subsumes about a dozen of the trace scripts.
2. **Rewrite soundness in Z3 bit-vectors.** Each emission identity as a 32-bit theorem:
   the three multiply-add hash fusions, the deferred stage-5 XOR, arithmetic select
   `(t - f) * c + f`, and each new identity before it is used (1-based index so the update
   is `2j + bit`, pointer-form indices, constant offload). These are small quantifier-free
   queries.
3. **SSA against the reference, per round.** With 1 and 2, an inductive invariant relating
   the kernel's (value, index) representation to the reference state after each round
   gives end-to-end correctness without trusting an edited `reference_kernel2`.
4. **One liveness definition, checked.** Until the four copies are unified, assert after
   every greedy run that its peak equals the interval-based peak. The leak fixed here is
   exactly a counterexample to that equality, and a bounded Z3 model over small op
   patterns (vector groups, shifted ops, FMA selects) would have found it.
   *Done for the greedy counter (spec T16): checked after every run, with a property test
   and a bounded Z3 statement. The first runs found three more undercounts (up to 256
   words) in real kernels; see BUGS.md.*
5. **Per-instance certificates for tightening.** T7 showed that composing individually
   valid exclusions is unsound. Emit, for every excluded (op, cycle), the overloaded
   interval that justifies it, and check those independently. Checking instances is easier
   than proving composition in general.
6. **Lower-bound certificates.** The capacity floor is an LP; its dual solution is a
   checkable certificate. State bounds as conditional on an emission. A public claim of a
   Lean-proved 1,081 bound was later withdrawn by its author for proving something other
   than intended, and is contradicted by the leaderboard.
7. **Periodic schedule validity.** If a modulo template is used, conflict-freedom for all
   groups reduces to a finite check over one period.

## 5. Tooling findings

| Finding | Effect | Suggested fix |
|---------|--------|---------------|
| `cargo test` as documented fails: PyO3 0.27 rejects system Python 3.15 | Rust tests look broken | Set `PYO3_PYTHON=.venv/bin/python` in the build script and CLAUDE.md |
| No CI | Regressions surface months later | One workflow: build, Rust tests, 2-round and 16-round greedy with validation |
| `-q` printed 253,250 lines on a failing 16-round run | The real error is buried | Dump the liveness breakdown once, at the first overflow |
| Abort message "peak_live=1268 > budget=1536" | Misleading; the real figure was the greedy's 4552 | Report which accounting tripped |
| `frozen_problem.py` re-exports the edited `problem.py`; header 7 to 8, reference kernel rewritten | The validation oracle is not the upstream one | Vendor the upstream `tests/` unchanged and validate against it |
| `SSAKernelBuilder` (the submission path) hardcodes CP-SAT 600 s, shift off | `tests/submission_tests.py` does not exercise the best configuration | Load a saved kernel, or share the CLI defaults |
| About 55 single-purpose scripts, 69 docs, several stale (OPTIMIZATION_PLAN.md describes a 2,390-cycle Python pipeline) | Hard to know what is current | Replace with the replay checker and one `explain` report: floors, per-round spans, utilisation over time, gap |
| `model.rs` builds and solves in one 3,400-line function | Hard to test pieces or build windowed models | Split model construction from solve; needed for LNS anyway |
| 32 compiler warnings, mostly dead code | Noise | Delete dead paths |
| No per-run record of floor against achieved cycles | "Is this good?" has no answer | Print the capacity floor and the gap on every run |

## 6. Literature notes

From three recall passes (Opus and Sonnet) checked against what was measured here. Depth
labels are the reviewers': FT = full text read, ABS = abstract only.

**Scale of exact methods**
- Malik, McInnes, van Beek, ICTAI 2006 / IJAIT 2008 (FT): CP, up to 2,600 instructions,
  solved bottom-up from a lower bound. Their minimal model "does not scale beyond 40
  instructions"; region distance constraints and dominance between isomorphic subgraphs
  made the difference.
- Castañeda Lozano, Carlsson, Hjort Blindell, Schulte, TOPLAS 2019 (FT): Unison, up to 946
  instructions; decomposition on boundary state, implied register cumulative, probing,
  solver portfolio.
- Eriksson and Kessler, TECS 2012 (FT): integrated ILP, 142 nodes; integrated models often
  return nothing where phase-separated ones return a solution.
- Shobaki, Kerbow, Mekhanoshin, CGO 2020 (FT): two-pass branch and bound, register
  pressure first, then length.
- Chen, arXiv 2303.06855 (FT) and Bartan et al., ICML 2023 (FT): the only CP-SAT
  scheduling-under-memory papers found, up to 511 and 1,000 nodes.
- Roorda, arXiv 2601.21842 (FT): SMT software pipelining for a VLIW; register-pressure
  constraints added lazily, unsat cores used for diagnosis.

**Bounds and search**
- Klein and Scholl, EJOR 1999 (ABS): destructive lower bounds.
- Rim and Jain, TCAD 1994; Langevin and Cerny, TODAES 1996 (ABS): fast bounds for unit ops
  on typed units, which is this setting.
- Godard, Laborie, Nuijten, ICAPS 2005 (ABS): LNS that freezes order, not time.
- Valls, Ballestín, Quintanilla, EJOR 2005 (ABS): double justification.
- Kovács and Váncza, CPAIOR 2006 (FT): dominance for identical subprojects.
- Touati, CC 2001 (ABS): register saturation; add arcs until the register constraint is
  redundant.

**Not found by any pass:** CP-SAT applied to VLIW or SIMD instruction scheduling; exact
methods on straight-line code of 5,000 ops or more; equality saturation with schedule
length in the extraction cost; any solver-based entry for this challenge. The public
entries all use list schedulers, one with a small beam search.

**Public technique catalogue for this kernel** (from write-ups, not verified in code):
wavefront staggering of groups; arithmetic select on the first level to spare the flow
slot; stage-1 and stage-3 constants moved to the scalar ALU; pointer-form indices so
gathers need no address add; skipping the index store (worth about 15 cycles in one
write-up; the upstream test checks values only); loading distinct nodes once for levels
where there are fewer nodes than lanes.

**Version cautions** (from issue trackers, not reproduced): OR-Tools 9.15 has a reported
presolve overflow that produced a wrong OPTIMAL (issue 5293), and 10.0 RC1 a reported LNS
regression (issue 5571).

## 7. Suggested order of work

1. Replay checker, upstream tests vendored, CI. Everything after this is faster.
2. Capacity floor and gap printed on every run.
3. Remove round barriers in the greedy and stagger groups. Target: within 10% of the
   1,270 floor without the solver.
4. Emission reductions, each with a Z3 identity, driven by the encoding-selection model.
5. Solver configuration experiments on 2 and 4 rounds, where bounds are provable.
6. Windowed CP-SAT over the 16-round schedule; modulo template for the steady state.

## 8. Reproduction

```bash
./scripts/build_scheduler_rs.sh --release
uv run generate_ssa_kernel.py --rounds 16 --use-greedy --validate -q --output-json scheduler_output_g16.json
PYO3_PYTHON=$PWD/.venv/bin/python BUILD_FROM_PROJECT_ROOT=1 ORTOOLS_PREFIX=$(brew --prefix or-tools) \
  cargo test --manifest-path scheduler_rs/Cargo.toml --features cpsat
```
