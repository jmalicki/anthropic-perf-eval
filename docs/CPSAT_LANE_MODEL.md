# Lane model of shiftable ops (`--cpsat-lane-model`)

A shiftable vector op `o` has, in the CP-SAT model (`model.rs` section 6), a cycle variable
`c_o`, a shift literal `shift_o`, eight lane variables `alu_c{o}_{i}` (`a_o,i`: the cycle
of each scalar op when `o` is shifted, each the start of an optional size-1 interval in the
scalar cumulative) and `completion_o >= c_o, a_o,i`. Review
[CPSAT_VARIABLE_MODELLING_REVIEW.md](CPSAT_VARIABLE_MODELLING_REVIEW.md) 4.2 (items 1, 2, 3, 5)
and 4.4 (a), (b) proposed two restrictions; both are now switches, default off:

| `--cpsat-lane-model` | Adds | Removes |
|---|---|---|
| `free` (default) | nothing: the exported model is byte-identical to `main` | |
| `sorted` | `lane_order_{o}_{i}`: `a_o,i <= a_o,i+1` (7 rows per shiftable op) | the 8! lane permutations of each op |
| `pinned` | `lane_pin_{o}_{i}`: `!shift_o => a_o,i = c_o` (8 enforced rows per op) | the free lanes of an unshifted op; each lane precedence pair `shift_d => a_o,i >= a_d,i + 1`, `!shift_d => a_o,i >= c_d + 1` becomes one unconditional row `shift_alu_prec_{o}_{i}_dep_{d}` |
| `sorted-pinned` | both | both |

## 1. Why neither loses a schedule

Both are dominance restrictions: every feasible schedule of the free model maps to a
feasible schedule of the restricted model with the same cycle variables, shift literals,
completions, liveness starts and, per op, the same multiset of lane times. So the makespan,
the tie-breaker objective and the per-cycle scalar-slot usage are unchanged, and the
restricted model keeps an optimal schedule of the free one.

The mapping is: sort the lanes of every shiftable op (sorted); set the lanes of every
unshifted op to its cycle (pinned); both, in that order (sorted-pinned).

**Every row of today's model that mentions a lane** has one of these shapes (from
`model.rs`; confirmed on exported models, below):

| Rows | Shape | Under a common lane permutation | Under per-op sorting | Under pinning (op unshifted) |
|---|---|---|---|---|
| `shift_alu_prec_*_on` | `a_o,i >= a_d,i + 1`, enforced by `shift_d`; **same lane index** (`element_dep_map` is the identity) | kept | kept: if `x_i >= y_i + 1` for all i then `sort(x)_k >= sort(y)_k + 1` | `c_o >= completion_d + 1 >= a_d,i + 1` (`consumer_after_completion`, `completion_ge_alu`) |
| `shift_alu_prec_*_off` | `a_o,i >= c_d + 1`, enforced by `!shift_d` | same bound on every lane | kept | `c_o >= completion_d + 1 >= c_d + 1` |
| `shift_alu_prec_*_valu` | `a_o,i >= c_p + 1`, non-shiftable producer `p` (includes the 8 scalar loads of a gather: each bounds every lane) | same bound on every lane | kept | twin row `prec_raw` on `c_o` |
| `select_prec_*_{fma,opt}` | `a_o,i >= c_s + 1 (+ extra)` | same bound on every lane | kept | twin row on `c_o` |
| `vb_elim_*_prec_{elim,keep}` | `a_o,i >= c_q + 1` / `>= c_vb + 1`, enforced by `eliminate_vb` / its negation | same bound on every lane | kept | `eliminate_vb => shift_o`; keep: twin precedence on `c_o` |
| `completion_ge_alu_*` | `completion_o >= a_o,i` | max | kept | `completion_ge_start` |
| `scratch_shifted_start_ge_alu_*` | `start_o <= a_o,i` | min | kept | `!shift_o => start_o = c_o` |
| lane intervals in `cumul_alu` | size 1, presence `shift_o`, demand 1 | same multiset per cycle | kept (a permutation) | absent when unshifted |
| interval binding | `a - a = 0` (the binding's start + size = end row) | trivial | trivial | trivial |

Nothing else mentions a lane: no objective term, no decision-strategy entry, no `lin_max`,
no `no_overlap`. Readers of a shifted op's result (vector or a decomposed lane) wait for
`completion_o`, which is permutation-invariant. The lane domains are `[asap, alap]`; the
cycle variable can only take values in that range in a feasible schedule (ASAP is sound for
precedence, and `c_o <= completion_o <= alap`), so pinning does not cut the cycle's domain
either.

**Where per-op sorting would be invalid.** If a row tied lane i of a reader to a producer
that differs per lane (for example lane-wise precedence from the eight scalar loads of a
gather into the lanes of a shifted reader, or a decomposed vector whose lane-i reader waited
for lane i of a shifted producer), sorting one op without the other would break it
(`LANE-sort-lane-specific-producers` is the Z3 counterexample). No such row exists today.
If one is added, sort only ops whose lanes have no lane-specific producer, or sort
consistently along each chain; `tests/test_cpsat_lane_model.py` fails until then.

### 1.1 Mechanical checks

- **Z3** (`verification/lane_model.py`, run by `./scripts/test.sh proofs`; see
  [verification/README.md](../verification/README.md)). Each check encodes a free schedule
  that satisfies every row and a mapped schedule that breaks one; UNSAT = no schedule lost.
  Unbounded integers, so each holds for all values. Sorting is a fixed compare-exchange
  network (Batcher, 19 comparators) applied to every op at once: one step keeps every row of
  a DAG with all shapes above, for each comparator, in the free and the pinned model
  (`LANE-sort-dag`, `LANE-sorted-pinned-dag`), and the network sorts every input
  (`LANE-network-sorts`); by induction a sorted schedule exists. `LANE-pin-dag` proves the
  pin mapping, `LANE-pin-unconditional-precedence` that the unconditional row equals the
  enforced pair once lanes are pinned.
- **Structure of the real model** (`tests/test_cpsat_lane_model.py`). On exported models
  (3 and 5 rounds, the latter with a gather round; forced multiply-add selects; eliminable
  broadcasts), every row that mentions a lane is classified: a uniform row must exist for all
  eight lanes of the op, a lane-wise row must use the same index on both sides and exist for
  all eight, lane intervals share one presence literal and demand 1. Any other row fails
  the test. The premises of the pinning proof (a twin row on the cycle variable, or
  `consumer_after_completion` for lane-wise rows; the lane domain containing the cycle's)
  are checked on the free model.
- **Optimum unchanged on a toy** (`tuning_lane_model_tests` in `cpsat_scheduler/tests.rs`):
  every lane model proves the same optimal makespan under both objectives.
- **Hint** (`tests/test_cpsat_hint_complete.py`): under every lane model the greedy hint is
  complete and feasible (each hinted variable fixed to its hint, CP-SAT finds the fixed
  model feasible). The greedy hints all eight lanes of an op at the op's cycle, which is
  sorted and pinned. A future per-lane hint (the split-shift hint) must hint lanes in
  non-decreasing order and an unshifted op's lanes at its cycle.

## 2. Measurements

Measured with `scripts/run_cpsat_ablation.py` (skill
[use-cpsat-ablation](../.cursor/skills/use-cpsat-ablation/SKILL.md)) at a budget of **20
deterministic time** on the instances of [CPSAT_CONFIG_ABLATION.md](CPSAT_CONFIG_ABLATION.md),
each lane model with the default objective and with `--cpsat-objective makespan`. Code: this
branch on `main` at `ca5087a` (the rebase onto `50db776` brought only address-allocator and
architecture changes, which do not touch the CP-SAT model). Every kernel below was **PROVED**
by the dataflow replay and **PASSED** numeric validation. Wall times were taken on a machine
shared with other jobs (load 130 to 300 on 12 cores) and are indicative only; deterministic
times are not affected by load. "Bound" is the solver's bound for the emitted ops inside the
model's domains, not a bound on the problem.

### 2.1 Summary

- **No lane model changes the makespan consistently.** With the default objective all four
  give the same cycles on 2r/b256, 3r/b256, 4r/b256 and 4r/b64 (one run 140 against 139);
  on 6r/b64 pinned gave 228 and sorted-pinned 233 against 232. With the makespan-only
  objective the spread is 126 to 134 at 4r/b64 and 222 to 227 at 6r/b64. These are single
  trajectories (one seed each); differences of a few cycles that do not repeat across
  instances are search noise, as in [CPSAT_CONFIG_ABLATION.md](CPSAT_CONFIG_ABLATION.md).
- **The bound never moved**: the same `lb_cycles` in every configuration of an instance.
- **Sorting removes the lane symmetry CP-SAT detects.** With free lanes presolve reports
  orbits of size 8 (batch 256: e.g. 1,955 orbits on 15,634 variables at 4r/b256, orbitope
  6 x 8) or 16 (batch 64); with sorted lanes 1 orbit of 2 variables remains at batch 256 and
  orbits of size 2 (orbitope 14 x 2, a symmetry between pairs of groups, not lanes) at batch
  64. When no symmetry is left, interleaved search drops the `max_lp_sym` worker.
- **Sorting cuts solver work at the same budget by 22 to 45 % on batch 256** (solver
  deterministic time summed over workers: 61 -> 37 at 2r, 65 -> 47 at 3r, 101 -> 61 at
  4r/b256 with the default objective; 37 -> 29, 47 -> 26, 62 -> 42 with makespan-only), with
  the same cycles and a first solution as early (within load noise). At batch 64 it changes nothing
  (20 to 25 in every configuration: the solve ends on the per-worker budget there).
- **Pinning changes little.** It adds 8 enforced rows per shiftable op (about +5 % rows
  after presolve) and makes lane precedences unconditional, but presolve keeps the same
  number of variables; solver work and first-solution time are within noise of free.
- **First solution**: always the complete hint (`complete_hint`), within seconds in every
  configuration; the hint stays complete and feasible under every lane model.
- **OPTIMAL** was reached only on the small instance (2 rounds / batch 16, budget 5): 36
  cycles in every lane model, solver work 10.7 to 11.0 (default objective) and 0.86 to 1.59
  (makespan-only; pinned 0.86, the others 1.56 to 1.59).
- **16 rounds** (shaped greedy start, budget 60): see 2.4.

**Recommendation**: keep `free` as the default for now. `sorted` is safe (proved), removes
the lane symmetry and the work spent on it, and never cost a cycle here except within noise;
it is the candidate to make default together with the makespan-only objective once a study
with several seeds confirms the work saving turns into cycles at longer budgets.

### 2.2 Cycles by configuration and instance (budget 20)

| Configuration | 2r / b256 (budget 20) | 3r / b256 (budget 20) | 4r / b64 (budget 20) | 6r / b64 (budget 20) | 4r / b256 (budget 20) |
|---|---|---|---|---|---|
| *greedy hint* | 194 | 327 | 143 | 240 | 458 |
| baseline | 193 | 325 | 139 | 232 | 455 |
| lane-sorted | 193 | 325 | 139 | 232 | 455 |
| lane-pinned | 193 | 325 | 140 | 228 | 455 |
| lane-sorted-pinned | 193 | 325 | 139 | 233 | 455 |
| makespan | 192 | 325 | 130 | 225 | 455 |
| makespan-sorted | 193 | 325 | 126 | 227 | 455 |
| makespan-pinned | 192 | 325 | 130 | 222 | 455 |
| makespan-sorted-pinned | 192 | 325 | 134 | 227 | 455 |
| *capacity floor* | 163 | 249..250 | 94..96 | 130..132 | 357..358 |

### 2.3 Detail per cell

"Vars" is the model handed to CP-SAT; "Presolved vars/cons" the model after presolve
(`#Model` line). "Symmetry" is the first `[Symmetry]` line of presolve. "First solution" and
"Final incumbent" are solver wall seconds (load-dependent); "Solver dtime" is the solver's
deterministic time summed over workers. Peak is the peak scratch liveness of the kernel.

| Instance | Config | Cycles | Status | Bound | Vars | Presolved vars/cons | Symmetry (presolve) | First solution s | Final incumbent s | Solver dtime | Solver wall s | max_lp_sym | Valid | Peak |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2r_b256 | baseline | 193 | FEASIBLE | 164 | 14315 | 14273/49951 | 897 orbits / 7170 vars (sizes 8,8,8,8,8,8...), orbitope 6 x 8 | 9.54 (complete_hint) | 9.54 | 61.3746 | 548.608 | True | PROVED, PASSED | 1129 |
| 2r_b256 | lane-pinned | 193 | FEASIBLE | 164 | 14315 | 14273/52255 | 897 orbits / 7170 vars (sizes 8,8,8,8,8,8...), orbitope 6 x 8 | 9.53 (complete_hint) | 9.53 | 62.5127 | 458.246 | True | PROVED, PASSED | 1114 |
| 2r_b256 | lane-sorted-pinned | 193 | FEASIBLE | 164 | 14315 | 14272/58525 | 1 orbits / 2 vars (sizes 2...) | 9.57 (complete_hint) | 9.57 | 38.6986 | 397.217 | False | PROVED, PASSED | 1114 |
| 2r_b256 | lane-sorted | 193 | FEASIBLE | 164 | 14315 | 14272/56221 | 1 orbits / 2 vars (sizes 2...) | 9.95 (complete_hint) | 9.95 | 36.5152 | 492.635 | False | PROVED, PASSED | 1123 |
| 2r_b256 | makespan-pinned | 192 | FEASIBLE | 164 | 13387 | 13377/50463 | 897 orbits / 7170 vars (sizes 8,8,8,8,8,8...), orbitope 6 x 8 | 6.78 (complete_hint) | 72.05 | 42.5102 | 286.767 | True | PROVED, PASSED | 1113 |
| 2r_b256 | makespan-sorted-pinned | 192 | FEASIBLE | 164 | 13387 | 13376/56733 | 1 orbits / 2 vars (sizes 2...) | 11.82 (complete_hint) | 97.73 | 28.2952 | 241.439 | False | PROVED, PASSED | 1113 |
| 2r_b256 | makespan-sorted | 193 | FEASIBLE | 164 | 13387 | 13376/54429 | 1 orbits / 2 vars (sizes 2...) | 6.71 (complete_hint) | 6.71 | 28.534 | 231.958 | False | PROVED, PASSED | 1113 |
| 2r_b256 | makespan | 192 | FEASIBLE | 164 | 13387 | 13377/48159 | 897 orbits / 7170 vars (sizes 8,8,8,8,8,8...), orbitope 6 x 8 | 5.77 (complete_hint) | 47.48 | 36.5611 | 289.789 | True | PROVED, PASSED | 1113 |
| 3r_b256 | baseline | 325 | FEASIBLE | 249 | 22940 | 22674/77159 | 1411 orbits / 11282 vars (sizes 8,8,8,8,8,8...), orbitope 6 x 8 | 9.47 (complete_hint) | 9.47 | 65.0769 | 1562.74 | True | PROVED, PASSED | 1305 |
| 3r_b256 | lane-pinned | 325 | FEASIBLE | 249 | 22940 | 22674/81271 | 1411 orbits / 11282 vars (sizes 8,8,8,8,8,8...), orbitope 6 x 8 | 17.37 (complete_hint) | 17.37 | 61.7955 | 1877.78 | True | PROVED, PASSED | 1274 |
| 3r_b256 | lane-sorted-pinned | 325 | FEASIBLE | 249 | 22940 | 22673/91139 | 1 orbits / 2 vars (sizes 2...) | 24.15 (complete_hint) | 24.15 | 38.0608 | 1531.0 | False | PROVED, PASSED | 1273 |
| 3r_b256 | lane-sorted | 325 | FEASIBLE | 249 | 22940 | 22673/87027 | 1 orbits / 2 vars (sizes 2...) | 10.57 (complete_hint) | 10.57 | 47.2731 | 1307.65 | False | PROVED, PASSED | 1273 |
| 3r_b256 | makespan-pinned | 325 | FEASIBLE | 249 | 21402 | 21200/78323 | 1411 orbits / 11282 vars (sizes 8,8,8,8,8,8...), orbitope 6 x 8 | 25.78 (complete_hint) | 25.78 | 39.3507 | 1184.53 | True | PROVED, PASSED | 1271 |
| 3r_b256 | makespan-sorted-pinned | 325 | FEASIBLE | 249 | 21402 | 21199/88191 | 1 orbits / 2 vars (sizes 2...) | 18.25 (complete_hint) | 18.25 | 27.8759 | 978.063 | False | PROVED, PASSED | 1271 |
| 3r_b256 | makespan-sorted | 325 | FEASIBLE | 249 | 21402 | 21199/84079 | 1 orbits / 2 vars (sizes 2...) | 22.96 (complete_hint) | 22.96 | 26.0023 | 930.808 | False | PROVED, PASSED | 1271 |
| 3r_b256 | makespan | 325 | FEASIBLE | 249 | 21402 | 21200/74211 | 1411 orbits / 11282 vars (sizes 8,8,8,8,8,8...), orbitope 6 x 8 | 27.19 (complete_hint) | 27.19 | 47.2411 | 1152.45 | True | PROVED, PASSED | 1271 |
| 4r_b256 | baseline | 455 | FEASIBLE | 357 | 32630 | 32046/107620 | 1955 orbits / 15634 vars (sizes 8,8,8,8,8,8...), orbitope 6 x 8 | 41.17 (complete_hint) | 41.17 | 101.155 | 3181.1 | True | PROVED, PASSED | 1303 |
| 4r_b256 | lane-pinned | 455 | FEASIBLE | 357 | 32630 | 32046/112756 | 1955 orbits / 15634 vars (sizes 8,8,8,8,8,8...), orbitope 6 x 8 | 40.05 (complete_hint) | 40.05 | 107.153 | 3399.22 | True | PROVED, PASSED | 1297 |
| 4r_b256 | lane-sorted-pinned | 455 | FEASIBLE | 357 | 32630 | 32045/126432 | 1 orbits / 2 vars (sizes 2...) | 42.50 (complete_hint) | 42.5 | 64.1791 | 2946.42 | False | PROVED, PASSED | 1280 |
| 4r_b256 | lane-sorted | 455 | FEASIBLE | 357 | 32630 | 32045/121296 | 1 orbits / 2 vars (sizes 2...) | 41.41 (complete_hint) | 41.41 | 60.8511 | 2221.21 | False | PROVED, PASSED | 1324 |
| 4r_b256 | makespan-pinned | 455 | FEASIBLE | 357 | 30452 | 29932/108528 | 1955 orbits / 15634 vars (sizes 8,8,8,8,8,8...), orbitope 6 x 8 | 36.97 (complete_hint) | 36.97 | 67.53 | 1605.15 | True | PROVED, PASSED | 1266 |
| 4r_b256 | makespan-sorted-pinned | 455 | FEASIBLE | 357 | 30452 | 29931/122204 | 1 orbits / 2 vars (sizes 2...) | 35.80 (complete_hint) | 35.8 | 45.7138 | 1161.29 | False | PROVED, PASSED | 1266 |
| 4r_b256 | makespan-sorted | 455 | FEASIBLE | 357 | 30452 | 29931/117068 | 1 orbits / 2 vars (sizes 2...) | 46.87 (complete_hint) | 46.87 | 41.9582 | 1325.77 | False | PROVED, PASSED | 1266 |
| 4r_b256 | makespan | 455 | FEASIBLE | 357 | 30452 | 29932/103392 | 1955 orbits / 15634 vars (sizes 8,8,8,8,8,8...), orbitope 6 x 8 | 102.51 (complete_hint) | 102.51 | 61.6416 | 2131.62 | True | PROVED, PASSED | 1266 |
| 4r_b64 | baseline | 139 | FEASIBLE | 95 | 8390 | 8234/27586 | 546 orbits / 4054 vars (sizes 16,16,16,16...), orbitope 14 x 2 | 5.11 (complete_hint) | 222.25 | 20.3109 | 332.016 | True | PROVED, PASSED | 570 |
| 4r_b64 | lane-pinned | 140 | FEASIBLE | 95 | 8390 | 8234/28882 | 546 orbits / 4054 vars (sizes 16,16,16,16...), orbitope 14 x 2 | 3.78 (complete_hint) | 3.78 | 21.7564 | 392.722 | True | PROVED, PASSED | 569 |
| 4r_b64 | lane-sorted-pinned | 139 | FEASIBLE | 95 | 8390 | 8234/32312 | 155 orbits / 310 vars (sizes 2,2,2,2,2,2...), orbitope 14 x 2 | 4.92 (complete_hint) | 214.31 | 21.6292 | 404.353 | True | PROVED, PASSED | 586 |
| 4r_b64 | lane-sorted | 139 | FEASIBLE | 95 | 8390 | 8234/31016 | 155 orbits / 310 vars (sizes 2,2,2,2,2,2...), orbitope 14 x 2 | 6.63 (complete_hint) | 196.4 | 20.2214 | 296.684 | True | PROVED, PASSED | 563 |
| 4r_b64 | makespan-pinned | 130 | FEASIBLE | 95 | 7844 | 7704/27822 | 535 orbits / 4032 vars (sizes 16,16,16,16...), orbitope 13 x 2 | 6.16 (complete_hint) | 563.97 | 23.2293 | 740.823 | True | PROVED, PASSED | 553 |
| 4r_b64 | makespan-sorted-pinned | 134 | FEASIBLE | 95 | 7844 | 7704/31252 | 144 orbits / 288 vars (sizes 2,2,2,2,2,2...), orbitope 13 x 2 | 7.71 (complete_hint) | 510.54 | 22.0827 | 724.313 | True | PROVED, PASSED | 545 |
| 4r_b64 | makespan-sorted | 126 | FEASIBLE | 95 | 7844 | 7704/29956 | 144 orbits / 288 vars (sizes 2,2,2,2,2,2...), orbitope 13 x 2 | 5.19 (complete_hint) | 432.58 | 23.3687 | 519.466 | True | PROVED, PASSED | 545 |
| 4r_b64 | makespan | 130 | FEASIBLE | 95 | 7844 | 7704/26526 | 535 orbits / 4032 vars (sizes 16,16,16,16...), orbitope 13 x 2 | 7.13 (complete_hint) | 317.41 | 21.0631 | 353.215 | True | PROVED, PASSED | 570 |
| 6r_b64 | baseline | 232 | FEASIBLE | 155 | 11785 | 11629/39955 | 942 orbits / 6118 vars (sizes 16,16,16,16...), orbitope 14 x 2 | 21.70 (complete_hint) | 628.71 | 21.9386 | 807.476 | True | PROVED, PASSED | 565 |
| 6r_b64 | lane-pinned | 228 | FEASIBLE | 155 | 11785 | 11629/41763 | 942 orbits / 6118 vars (sizes 16,16,16,16...), orbitope 14 x 2 | 12.57 (complete_hint) | 552.35 | 23.4763 | 684.816 | True | PROVED, PASSED | 557 |
| 6r_b64 | lane-sorted-pinned | 233 | FEASIBLE | 155 | 11785 | 11629/46649 | 451 orbits / 902 vars (sizes 2,2,2,2,2,2...), orbitope 14 x 2 | 19.23 (complete_hint) | 506.38 | 25.2392 | 677.938 | True | PROVED, PASSED | 556 |
| 6r_b64 | lane-sorted | 232 | FEASIBLE | 155 | 11785 | 11629/44841 | 451 orbits / 902 vars (sizes 2,2,2,2,2,2...), orbitope 14 x 2 | 20.74 (complete_hint) | 548.51 | 24.5776 | 737.312 | True | PROVED, PASSED | 597 |
| 6r_b64 | makespan-pinned | 222 | FEASIBLE | 155 | 11031 | 10891/40287 | 919 orbits / 6072 vars (sizes 16,16,16,16...), orbitope 13 x 2 | 22.46 (complete_hint) | 454.84 | 20.0614 | 539.559 | True | PROVED, PASSED | 567 |
| 6r_b64 | makespan-sorted-pinned | 227 | FEASIBLE | 155 | 11031 | 10891/45173 | 428 orbits / 856 vars (sizes 2,2,2,2,2,2...), orbitope 13 x 2 | 26.24 (complete_hint) | 394.67 | 22.3542 | 484.229 | True | PROVED, PASSED | 544 |
| 6r_b64 | makespan-sorted | 227 | FEASIBLE | 155 | 11031 | 10891/43365 | 428 orbits / 856 vars (sizes 2,2,2,2,2,2...), orbitope 13 x 2 | 13.73 (complete_hint) | 389.62 | 21.3823 | 506.983 | True | PROVED, PASSED | 544 |
| 6r_b64 | makespan | 225 | FEASIBLE | 155 | 11031 | 10891/38479 | 919 orbits / 6072 vars (sizes 16,16,16,16...), orbitope 13 x 2 | 12.96 (complete_hint) | 480.56 | 21.3976 | 638.996 | True | PROVED, PASSED | 544 |

### 2.4 16 rounds, shaped greedy start

PENDING16

### 2.5 Validation runs

Dataflow replay PROVED and `--validate` PASSED for every lane model on 2 rounds / batch 16,
3 rounds / batch 64 and 4 rounds / batch 64 (budget 5), on the 40 cells above, and on the
16-round shaped-greedy builds of 2.4.

| Instance (budget 5) | free | sorted | pinned | sorted-pinned |
|---|---|---|---|---|
| 2r / b16 | 36, OPTIMAL | 36, OPTIMAL | 36, OPTIMAL | 36, OPTIMAL |
| 2r / b16, makespan-only | 36, OPTIMAL | 36, OPTIMAL | 36, OPTIMAL | 36, OPTIMAL |
| 3r / b64 | 97 | 97 | 96 | 96 |
| 4r / b64 | 140 | 140 | 140 | 140 |

## 3. Reproduction

```bash
./scripts/build_scheduler_rs.sh --release
uv run python -m scripts.run_cpsat_ablation --out-dir build/lane_ablation --jobs 4 \
    --instances 2x256,3x256,4x64,6x64,4x256 --budget 20 \
    --config "baseline:" --config "lane-sorted:--cpsat-lane-model sorted" \
    --config "lane-pinned:--cpsat-lane-model pinned" --config "lane-sorted-pinned:--cpsat-lane-model sorted-pinned" \
    --config "makespan:--cpsat-objective makespan" \
    --config "makespan-sorted:--cpsat-objective makespan --cpsat-lane-model sorted" \
    --config "makespan-pinned:--cpsat-objective makespan --cpsat-lane-model pinned" \
    --config "makespan-sorted-pinned:--cpsat-objective makespan --cpsat-lane-model sorted-pinned"
uv run python -m verification.verify -q -t LANE-sort-dag          # or ./scripts/test.sh proofs
uv run pytest tests/test_cpsat_lane_model.py tests/test_cpsat_hint_complete.py -k lane
```
