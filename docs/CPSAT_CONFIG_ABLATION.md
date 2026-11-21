# CP-SAT configuration ablation (November 2026)

Each CP-SAT configuration or modelling change proposed in
[CPSAT_VARIABLE_MODELLING_REVIEW.md](CPSAT_VARIABLE_MODELLING_REVIEW.md) (sections 3, 7, 8, 10)
and [PICKUP_ASSESSMENT_2026-11.md](PICKUP_ASSESSMENT_2026-11.md) (section 2) that did not need
a re-encoding is now a switch, off by default, and was measured alone against the default
configuration. Defaults were not changed.

## Summary

- **One switch helps: `--cpsat-objective makespan`** (drop the sum-of-cycles tie-breaker). At
  a budget of 20 deterministic time it returned 130 instead of 139 cycles at 4 rounds /
  batch 64, 230 instead of 235 at 6 rounds / batch 64, 192 instead of 193 at 2 rounds /
  batch 256, and the baseline's 325 and 455 at 3 and 4 rounds / batch 256; it was never worse
  than the default. Small instances reach OPTIMAL with 2.5 to 7 times less solver work at the same
  makespan. It is safe (section 4), but it leaves the `completion` variable of some shifted
  ops above their last lane, which makes address assignment hold their operands longer than
  needed.
- **Nothing else moved the makespan consistently** at that budget: worker count, greedy ALAP
  slack (5, 50, none), the horizon guards, the makespan interval,
  `push_all_tasks_toward_start`, the extra cumulative propagators, dropping the redundant
  rows. Single runs differ from the baseline by 0 to 2 cycles in either direction; the
  propagators gave 5 cycles on one instance at 1.5 to 5 times the wall time.
- **Non-interleaved search is the one lead worth following up, but it is not a like-for-like
  comparison.** At 4 rounds / batch 64 with the makespan-only objective it reached 120 / 121 /
  121 cycles over three runs against 130 interleaved, but the same per-worker budget buys it
  about six times the total solver work, and it is not reproducible (section 3.1).
- **The warm start matters far more than the solver configuration.** The greedy schedule
  without round barriers is better than anything CP-SAT reached from the round-barrier hint
  (119 against 127 to 139 cycles at 4 rounds / batch 64; 188 against 230 to 235 at 6 rounds /
  batch 64), and on that warm start CP-SAT added 0 to 5 cycles. At 16 rounds the comparison
  is deferred (section 2.3).
- **The bound never moved** beyond the capacity floor plus one or two cycles (or the critical
  path) in any configuration. OR-Tools 9.15 has no reproducible multi-worker configuration
  with bound workers (section 5).
- **The default objective excludes feasible schedules that makespan-only allows** (section
  4.3): an auxiliary variable of the tie-breaker forbids a select scheduled at its latest
  cycle from taking the multiply-add path. This is a model defect, not search behaviour, and
  it is a second, independent difference between the two objectives.
- **Recommended default** (section 6): `--cpsat-objective makespan`. Leave everything else as
  it is.

## 1. How it was measured

- Script: `uv run python -m scripts.run_cpsat_ablation` (skill
  [use-cpsat-ablation](../.cursor/skills/use-cpsat-ablation/SKILL.md)). It runs every
  configuration on every instance with the same `--cpsat-deterministic-time` budget and a
  generous wall-clock backstop, with `--validate` on.
- **Budget: 20 deterministic time** for every instance, the same for baseline and variant.
  The budget applies per worker; the solver reports the sum over workers, which is the
  "Solver dtime" column. No run ended on the wall-clock backstop.
- **Every kernel in the tables was PROVED** by the symbolic dataflow check
  (`verification/kernel_dataflow.py`) **and PASSED** numeric validation against
  `reference_kernel2`.
- Instances (forest height 10): 2 rounds / batch 256, 3 rounds / batch 256, 4 rounds /
  batch 64, 4 rounds / batch 256, 6 rounds / batch 64, and 16 rounds / batch 256 on the
  wavefront warm start, deferred (section 2.3). At 4 rounds / batch 256 only the baseline,
  the makespan-only objective and the ALAP cap off with the tight horizon were run.
- **Code measured:** the switches as merged in PRs #23 to #28 and the code of #39 (the
  lower-bound conversion gives the same values for the integral bounds CP-SAT reports), on
  `main` at `ab9f998`: before #38 (an address-assignment fix that cannot change cycle
  counts), before #40 (section 4.3; measured separately there) and before #41 (the CSE hint
  fix; see section 5).
- **Reproducibility.** With interleaved search (the default) the same command gives the same
  kernel. This was confirmed three times in this study: the 4 rounds / batch 64 cells were
  run on 2026-11-10, 2026-11-20 and 2026-11-21, under machine loads between 4 and 190 and
  across two rebases, and returned the same cycles and the same solver deterministic time to
  four decimals each time (baseline 139 cycles, 21.0486; makespan-only 130, 21.7252; 16
  workers 140, 28.4894). Configurations with `--no-cpsat-interleave` and more than one worker
  are not reproducible and were run three times; the tables give min / median / max.
- Wall times were taken on a machine shared with other jobs and are indicative only.
  Deterministic times are not affected by load.
- "Bound" is the solver's lower bound for the ops this build emitted, **inside the model's
  domains** (greedy ALAP cap, horizon). It is not a bound on the problem. "Floor" is the
  capacity floor of `scripts.schedule_report` for the emitted ops and chosen encodings, which
  is why it differs by a cycle or two between rows of one instance.
- "Final incumbent at": CP-SAT prints wall time, not deterministic time, for each solution.
  The deterministic-time figure is an estimate that assumes deterministic time advanced
  evenly over the solve.
- Each reproducible configuration is one trajectory (one seed). Differences of a few cycles
  between configurations are noise unless they repeat across instances.

In interleaved mode a budget of 20 is small: at 2 rounds / batch 256 it is one batch, in which
each of the 10 full-problem subsolvers runs once and each LNS neighbourhood once or twice.
The study answers "what does each switch do at a budget of a few minutes", not "what does it
do in an hour".

## 2. Results

### 2.1 Baseline and each switch alone

#### Cycles by configuration and instance

| Configuration | 2r / b256 (budget 20) | 3r / b256 (budget 20) | 4r / b64 (budget 20) | 4r / b256 (budget 20) | 6r / b64 (budget 20) |
|---|---|---|---|---|---|
| *greedy hint* | 194 | 327 | 143 | 458 | 240 |
| baseline | 193 | 325 | 139 | 455 | 235 |
| workers-16 | 193 | 325 | 140 | - | 235 |
| makespan-only | 192 | 325 | 130 | 455 | 230 |
| alap-slack-5 | 193 | 325 | 139 | - | 235 |
| alap-slack-50 | 193 | 325 | 139 | - | 235 |
| alap-slack-none | 193 | 325 | 139 | - | 235 |
| tight-horizon | 193 | 325 | 139 | - | 235 |
| alap-slack-none-tight-horizon | 193 | 325 | 137 | 455 | 235 |
| makespan-interval | 193 | 325 | 140 | - | 235 |
| push-toward-start | 193 | 325 | 139 | - | 235 |
| propagators-all | - | - | 139 | - | 230 |
| no-redundant-rows | 193 | 325 | 139 | - | 235 |
| *capacity floor* | 163 | 249 | 93..95 | 357 | 130..132 |

Non-reproducible configurations show min / median / max over their runs. "optimal" is optimal for the emitted ops inside the model's domains, not for the problem.

#### 2 rounds, batch 256 (budget 20 deterministic time; greedy hint 194 cycles)

| Configuration | Cycles | vs baseline | Status | Bound (lb_cycles) | Floor | Final incumbent at: dtime (est) | wall s | Solver dtime | Solver wall s | Peak live | Valid |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 193 | 0 | FEASIBLE | 164 | 163 | 1.2 | 13.9 | 61.4 | 733.8 | 1129 | PROVED, PASSED |
| workers-16 | 193 | 0 | FEASIBLE | 164 | 163 | 1.8 | 13.2 | 95.1 | 703.0 | 1129 | PROVED, PASSED |
| makespan-only | 192 | -1 | FEASIBLE | 164 | 163 | 9.2 | 113.9 | 36.6 | 453.0 | 1113 | PROVED, PASSED |
| alap-slack-5 | 193 | 0 | FEASIBLE | 163 | 163 | 1.5 | 18.2 | 52.5 | 639.4 | 1129 | PROVED, PASSED |
| alap-slack-50 | 193 | 0 | FEASIBLE | 164 | 163 | 0.5 | 6.2 | 52.9 | 599.9 | 1138 | PROVED, PASSED |
| alap-slack-none | 193 | 0 | FEASIBLE | 164 | 163 | 0.6 | 8.7 | 45.1 | 658.1 | 1121 | PROVED, PASSED |
| tight-horizon | 193 | 0 | FEASIBLE | 164 | 163 | 0.7 | 6.0 | 60.4 | 499.3 | 1129 | PROVED, PASSED |
| alap-slack-none-tight-horizon | 193 | 0 | FEASIBLE | 164 | 163 | 0.7 | 8.9 | 52.2 | 644.1 | 1121 | PROVED, PASSED |
| makespan-interval | 193 | 0 | FEASIBLE | 163 | 163 | 0.6 | 4.8 | 58.1 | 496.7 | 1122 | PROVED, PASSED |
| push-toward-start | 193 | 0 | FEASIBLE | 164 | 163 | 0.8 | 6.2 | 61.4 | 502.9 | 1129 | PROVED, PASSED |
| no-redundant-rows | 193 | 0 | FEASIBLE | 164 | 163 | 0.8 | 6.5 | 61.4 | 521.4 | 1129 | PROVED, PASSED |

#### 3 rounds, batch 256 (budget 20 deterministic time; greedy hint 327 cycles)

| Configuration | Cycles | vs baseline | Status | Bound (lb_cycles) | Floor | Final incumbent at: dtime (est) | wall s | Solver dtime | Solver wall s | Peak live | Valid |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 325 | 0 | FEASIBLE | 249 | 249 | 3.3 | 42.7 | 63.9 | 818.9 | 1271 | PROVED, PASSED |
| workers-16 | 325 | 0 | FEASIBLE | 249 | 249 | 1.7 | 15.1 | 79.2 | 708.7 | 1271 | PROVED, PASSED |
| makespan-only | 325 | 0 | FEASIBLE | 249 | 249 | 4.4 | 44.2 | 64.3 | 642.1 | 1271 | PROVED, PASSED |
| alap-slack-5 | 325 | 0 | FEASIBLE | 249 | 249 | 3.0 | 32.2 | 55.2 | 584.6 | 1271 | PROVED, PASSED |
| alap-slack-50 | 325 | 0 | FEASIBLE | 249 | 249 | 2.5 | 27.1 | 75.2 | 826.9 | 1271 | PROVED, PASSED |
| alap-slack-none | 325 | 0 | FEASIBLE | 249 | 249 | 1.0 | 18.4 | 60.4 | 1089.5 | 1271 | PROVED, PASSED |
| tight-horizon | 325 | 0 | FEASIBLE | 249 | 249 | 2.1 | 19.3 | 78.1 | 727.0 | 1271 | PROVED, PASSED |
| alap-slack-none-tight-horizon | 325 | 0 | FEASIBLE | 249 | 249 | 2.8 | 24.6 | 77.3 | 673.0 | 1271 | PROVED, PASSED |
| makespan-interval | 325 | 0 | FEASIBLE | 250 | 249 | 2.7 | 28.3 | 58.9 | 609.0 | 1271 | PROVED, PASSED |
| push-toward-start | 325 | 0 | FEASIBLE | 249 | 249 | 3.4 | 27.6 | 63.9 | 523.4 | 1271 | PROVED, PASSED |
| no-redundant-rows | 325 | 0 | FEASIBLE | 249 | 249 | 4.1 | 31.2 | 63.9 | 488.7 | 1271 | PROVED, PASSED |

#### 4 rounds, batch 64 (budget 20 deterministic time; greedy hint 143 cycles)

| Configuration | Cycles | vs baseline | Status | Bound (lb_cycles) | Floor | Final incumbent at: dtime (est) | wall s | Solver dtime | Solver wall s | Peak live | Valid |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 139 | 0 | FEASIBLE | 95 | 95 | 16.6 | 110.9 | 21.0 | 140.4 | 569 | PROVED, PASSED |
| workers-16 | 140 | +1 | FEASIBLE | 95 | 93 | 0.4 | 1.8 | 28.5 | 126.6 | 577 | PROVED, PASSED |
| makespan-only | 130 | -9 | FEASIBLE | 95 | 94 | 19.2 | 125.2 | 21.7 | 142.0 | 530 | PROVED, PASSED |
| alap-slack-5 | 139 | 0 | FEASIBLE | 93 | 95 | 18.4 | 133.6 | 21.0 | 152.1 | 569 | PROVED, PASSED |
| alap-slack-50 | 139 | 0 | FEASIBLE | 95 | 95 | 19.6 | 145.5 | 23.7 | 176.0 | 578 | PROVED, PASSED |
| alap-slack-none | 139 | 0 | FEASIBLE | 95 | 95 | 19.1 | 132.9 | 21.8 | 151.0 | 569 | PROVED, PASSED |
| tight-horizon | 139 | 0 | FEASIBLE | 95 | 95 | 14.8 | 104.7 | 21.2 | 149.9 | 569 | PROVED, PASSED |
| alap-slack-none-tight-horizon | 137 | -2 | FEASIBLE | 95 | 94 | 20.1 | 147.2 | 23.2 | 170.1 | 561 | PROVED, PASSED |
| makespan-interval | 140 | +1 | FEASIBLE | 95 | 95 | 0.4 | 2.7 | 21.7 | 128.6 | 577 | PROVED, PASSED |
| push-toward-start | 139 | 0 | FEASIBLE | 95 | 95 | 16.3 | 105.8 | 21.0 | 136.1 | 569 | PROVED, PASSED |
| propagators-all | 139 | 0 | FEASIBLE | 95 | 94 | 18.0 | 245.5 | 21.1 | 287.0 | 540 | PROVED, PASSED |
| no-redundant-rows | 139 | 0 | FEASIBLE | 95 | 95 | 15.8 | 102.8 | 21.0 | 136.5 | 569 | PROVED, PASSED |

#### 4 rounds, batch 256 (budget 20 deterministic time; greedy hint 458 cycles)

| Configuration | Cycles | vs baseline | Status | Bound (lb_cycles) | Floor | Final incumbent at: dtime (est) | wall s | Solver dtime | Solver wall s | Peak live | Valid |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 455 | 0 | FEASIBLE | 357 | 357 | 5.7 | 42.9 | 100.5 | 750.8 | 1266 | PROVED, PASSED |
| makespan-only | 455 | 0 | FEASIBLE | 357 | 357 | 5.9 | 41.7 | 103.2 | 733.1 | 1266 | PROVED, PASSED |
| alap-slack-none-tight-horizon | 455 | 0 | FEASIBLE | 358 | 357 | 7.6 | 47.2 | 114.3 | 713.4 | 1266 | PROVED, PASSED |

#### 6 rounds, batch 64 (budget 20 deterministic time; greedy hint 240 cycles)

| Configuration | Cycles | vs baseline | Status | Bound (lb_cycles) | Floor | Final incumbent at: dtime (est) | wall s | Solver dtime | Solver wall s | Peak live | Valid |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 235 | 0 | FEASIBLE | 155 | 131 | 0.7 | 7.5 | 22.0 | 233.6 | 565 | PROVED, PASSED |
| workers-16 | 235 | 0 | FEASIBLE | 155 | 130 | 0.3 | 3.9 | 26.0 | 392.7 | 539 | PROVED, PASSED |
| makespan-only | 230 | -5 | FEASIBLE | 155 | 130 | 20.4 | 318.7 | 20.8 | 324.2 | 544 | PROVED, PASSED |
| alap-slack-5 | 235 | 0 | FEASIBLE | 155 | 131 | 0.4 | 8.7 | 21.2 | 435.5 | 557 | PROVED, PASSED |
| alap-slack-50 | 235 | 0 | FEASIBLE | 155 | 131 | 1.4 | 24.2 | 23.0 | 412.7 | 557 | PROVED, PASSED |
| alap-slack-none | 235 | 0 | FEASIBLE | 155 | 131 | 1.0 | 20.0 | 20.4 | 398.3 | 557 | PROVED, PASSED |
| tight-horizon | 235 | 0 | FEASIBLE | 155 | 131 | 0.7 | 11.0 | 23.3 | 374.0 | 565 | PROVED, PASSED |
| alap-slack-none-tight-horizon | 235 | 0 | FEASIBLE | 155 | 130 | 0.9 | 18.9 | 21.1 | 444.7 | 539 | PROVED, PASSED |
| makespan-interval | 235 | 0 | FEASIBLE | 140 | 131 | 0.9 | 16.9 | 22.0 | 418.2 | 557 | PROVED, PASSED |
| push-toward-start | 235 | 0 | FEASIBLE | 155 | 131 | 1.0 | 22.4 | 21.8 | 482.0 | 565 | PROVED, PASSED |
| propagators-all | 230 | -5 | FEASIBLE | 156 | 132 | 18.8 | 304.1 | 22.3 | 359.8 | 561 | PROVED, PASSED |
| no-redundant-rows | 235 | 0 | FEASIBLE | 155 | 131 | 0.9 | 20.7 | 22.0 | 485.7 | 565 | PROVED, PASSED |

Configurations:

- `baseline`: `(defaults)`
- `workers-16`: `--cpsat-workers 16`
- `makespan-only`: `--cpsat-objective makespan`
- `alap-slack-5`: `--cpsat-greedy-alap-slack 5`
- `alap-slack-50`: `--cpsat-greedy-alap-slack 50`
- `alap-slack-none`: `--cpsat-greedy-alap-slack none`
- `tight-horizon`: `--no-cpsat-horizon-num-ops --no-cpsat-horizon-max-latency`
- `alap-slack-none-tight-horizon`: `--cpsat-greedy-alap-slack none --no-cpsat-horizon-num-ops --no-cpsat-horizon-max-latency`
- `makespan-interval`: `--cpsat-makespan-interval`
- `push-toward-start`: `--cpsat-push-tasks-toward-start`
- `propagators-all`: `--cpsat-cumulative-propagators all --cpsat-ttef-max-intervals 1000000`
- `no-redundant-rows`: `--disable-cpsat-constraints conditional_makespan,flow_cumulative`

Columns: *Cycles* is the schedule length of the validated kernel. *Bound* is the solver's lower bound for the ops this run emitted, inside the model's domains; it is not a bound on the problem. *Floor* is the capacity floor of scripts.schedule_report for the emitted ops. *Final incumbent at* is when the final makespan was first reached: wall seconds from the solver log, and a deterministic-time estimate (the log has no per-solution deterministic time; the estimate assumes it advanced evenly). *Solver dtime* is the total the solver reports, summed over workers, so it exceeds the per-worker budget.

### 2.2 Combinations with the makespan-only objective, and search modes

#### Cycles by configuration and instance

| Configuration | 2r / b256 (budget 20) | 4r / b64 (budget 20) | 6r / b64 (budget 20) |
|---|---|---|---|
| *greedy hint* | 194 | 143 | 240 |
| baseline | 193 | 139 | 235 |
| makespan-only | 192 | 130 | 230 |
| makespan+tight-horizon | 192 | 130 | 230 |
| makespan+alap-5 | 192 | 127 | 230 |
| makespan+alap-50 | 192 | 131 | 235 |
| makespan+alap-none+tight-horizon | 193 | 133 | 235 |
| makespan+workers-16 | 192 | 137 | 235 |
| makespan+push | 192 | 130 | 230 |
| makespan+makespan-interval | 193 | 129 | 234 |
| w1+makespan | - | 140 | - |
| no-interleave (not reproducible) | - | 129 | - |
| no-interleave+makespan (not reproducible) | - | 120 / 121 / 121 | - |
| no-interleave-w12-lb+makespan (not reproducible) | - | 121 / 121 / 122 | - |
| *capacity floor* | 163 | 93..96 | 130..131 |

Non-reproducible configurations show min / median / max over their runs. "optimal" is optimal for the emitted ops inside the model's domains, not for the problem.

#### 2 rounds, batch 256 (budget 20 deterministic time; greedy hint 194 cycles)

| Configuration | Cycles | vs baseline | Status | Bound (lb_cycles) | Floor | Final incumbent at: dtime (est) | wall s | Solver dtime | Solver wall s | Peak live | Valid |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 193 | 0 | FEASIBLE | 164 | 163 | 1.2 | 13.9 | 61.4 | 733.8 | 1129 | PROVED, PASSED |
| makespan-only | 192 | -1 | FEASIBLE | 164 | 163 | 9.2 | 113.9 | 36.6 | 453.0 | 1113 | PROVED, PASSED |
| makespan+tight-horizon | 192 | -1 | FEASIBLE | 164 | 163 | 8.0 | 35.3 | 36.6 | 161.7 | 1113 | PROVED, PASSED |
| makespan+alap-5 | 192 | -1 | FEASIBLE | 163 | 163 | 10.7 | 37.5 | 30.7 | 107.8 | 1113 | PROVED, PASSED |
| makespan+alap-50 | 192 | -1 | FEASIBLE | 164 | 163 | 8.5 | 32.0 | 32.8 | 124.0 | 1113 | PROVED, PASSED |
| makespan+alap-none+tight-horizon | 193 | 0 | FEASIBLE | 164 | 163 | 1.4 | 5.5 | 36.9 | 144.7 | 1113 | PROVED, PASSED |
| makespan+workers-16 | 192 | -1 | FEASIBLE | 164 | 163 | 25.5 | 32.8 | 69.6 | 89.6 | 1113 | PROVED, PASSED |
| makespan+push | 192 | -1 | FEASIBLE | 164 | 163 | 8.8 | 16.6 | 36.6 | 69.0 | 1113 | PROVED, PASSED |
| makespan+makespan-interval | 193 | 0 | FEASIBLE | 163 | 163 | 0.7 | 1.1 | 36.5 | 58.4 | 1113 | PROVED, PASSED |

#### 4 rounds, batch 64 (budget 20 deterministic time; greedy hint 143 cycles)

| Configuration | Cycles | vs baseline | Status | Bound (lb_cycles) | Floor | Final incumbent at: dtime (est) | wall s | Solver dtime | Solver wall s | Peak live | Valid |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 139 | 0 | FEASIBLE | 95 | 95 | 16.6 | 110.9 | 21.0 | 140.4 | 569 | PROVED, PASSED |
| makespan-only | 130 | -9 | FEASIBLE | 95 | 94 | 19.2 | 125.2 | 21.7 | 142.0 | 530 | PROVED, PASSED |
| makespan+tight-horizon | 130 | -9 | FEASIBLE | 95 | 94 | 20.9 | 87.5 | 22.6 | 94.6 | 530 | PROVED, PASSED |
| makespan+alap-5 | 127 | -12 | FEASIBLE | 94 | 95 | 21.3 | 86.9 | 22.8 | 93.0 | 546 | PROVED, PASSED |
| makespan+alap-50 | 131 | -8 | FEASIBLE | 95 | 96 | 18.5 | 100.6 | 21.9 | 119.4 | 561 | PROVED, PASSED |
| makespan+alap-none+tight-horizon | 133 | -6 | FEASIBLE | 95 | 93 | 18.0 | 78.2 | 23.4 | 101.9 | 561 | PROVED, PASSED |
| makespan+workers-16 | 137 | -2 | FEASIBLE | 95 | 93 | 25.5 | 54.1 | 36.5 | 77.5 | 569 | PROVED, PASSED |
| makespan+push | 130 | -9 | FEASIBLE | 95 | 94 | 19.1 | 87.0 | 21.7 | 99.0 | 530 | PROVED, PASSED |
| makespan+makespan-interval | 129 | -10 | FEASIBLE | 95 | 94 | 19.0 | 75.4 | 21.5 | 85.1 | 569 | PROVED, PASSED |
| w1+makespan | 140 | +1 | FEASIBLE | 95 | 93 | 1.9 | 9.1 | 20.0 | 94.8 | 561 | PROVED, PASSED |
| no-interleave (not reproducible) | 129 (all 3 runs) | -10 | FEASIBLE | 95 | 93..95 | 32.0..43.8 | 21.6..30.9 | 137.3..138.3 | 82.1..98.1 | 522..601 | PROVED, PASSED |
| no-interleave+makespan (not reproducible) | 120 / 121 / 121 (min / median / max of 3 runs) | -19..-18 | FEASIBLE | 95 | 94..96 | 28.9..96.1 | 48.4..141.1 | 133.5..133.8 | 196.5..232.3 | 531..570 | PROVED, PASSED |
| no-interleave-w12-lb+makespan (not reproducible) | 121 / 121 / 122 (min / median / max of 3 runs) | -18..-17 | FEASIBLE | 95 | 95..96 | 38.9..138.2 | 53.1..138.1 | 156.2..158.5 | 158.3..213.0 | 554..576 | PROVED, PASSED |

#### 6 rounds, batch 64 (budget 20 deterministic time; greedy hint 240 cycles)

| Configuration | Cycles | vs baseline | Status | Bound (lb_cycles) | Floor | Final incumbent at: dtime (est) | wall s | Solver dtime | Solver wall s | Peak live | Valid |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 235 | 0 | FEASIBLE | 155 | 131 | 0.7 | 7.5 | 22.0 | 233.6 | 565 | PROVED, PASSED |
| makespan-only | 230 | -5 | FEASIBLE | 155 | 130 | 20.4 | 318.7 | 20.8 | 324.2 | 544 | PROVED, PASSED |
| makespan+tight-horizon | 230 | -5 | FEASIBLE | 155 | 130 | 18.5 | 96.8 | 22.0 | 115.4 | 544 | PROVED, PASSED |
| makespan+alap-5 | 230 | -5 | FEASIBLE | 155 | 130 | 21.1 | 92.9 | 21.3 | 93.9 | 544 | PROVED, PASSED |
| makespan+alap-50 | 235 | 0 | FEASIBLE | 155 | 130 | 0.9 | 4.3 | 20.6 | 99.4 | 539 | PROVED, PASSED |
| makespan+alap-none+tight-horizon | 235 | 0 | FEASIBLE | 155 | 130 | 0.9 | 4.2 | 24.2 | 116.5 | 539 | PROVED, PASSED |
| makespan+workers-16 | 235 | 0 | FEASIBLE | 155 | 130 | 0.6 | 2.8 | 21.9 | 101.7 | 539 | PROVED, PASSED |
| makespan+push | 230 | -5 | FEASIBLE | 155 | 130 | 20.8 | 119.6 | 20.8 | 119.9 | 544 | PROVED, PASSED |
| makespan+makespan-interval | 234 | -1 | FEASIBLE | 141 | 130 | 19.4 | 86.6 | 23.0 | 102.6 | 539 | PROVED, PASSED |

Configurations:

- `baseline`: `(defaults)`
- `makespan-only`: `--cpsat-objective makespan`
- `makespan+tight-horizon`: `--cpsat-objective makespan --no-cpsat-horizon-num-ops --no-cpsat-horizon-max-latency`
- `makespan+alap-5`: `--cpsat-objective makespan --cpsat-greedy-alap-slack 5`
- `makespan+alap-50`: `--cpsat-objective makespan --cpsat-greedy-alap-slack 50`
- `makespan+alap-none+tight-horizon`: `--cpsat-objective makespan --cpsat-greedy-alap-slack none --no-cpsat-horizon-num-ops --no-cpsat-horizon-max-latency`
- `makespan+workers-16`: `--cpsat-objective makespan --cpsat-workers 16`
- `makespan+push`: `--cpsat-objective makespan --cpsat-push-tasks-toward-start`
- `makespan+makespan-interval`: `--cpsat-objective makespan --cpsat-makespan-interval`
- `w1+makespan`: `--no-cpsat-interleave --cpsat-workers 1 --cpsat-objective makespan`
- `no-interleave`: `--no-cpsat-interleave`
- `no-interleave+makespan`: `--no-cpsat-interleave --cpsat-objective makespan`
- `no-interleave-w12-lb+makespan`: `--no-cpsat-interleave --cpsat-workers 12 --cpsat-extra-subsolvers objective_lb_search,objective_shaving --cpsat-objective makespan`

### 2.3 CP-SAT on better warm starts

`nb-*` rows use `--greedy-barriers none`. The greedy schedules alone
(`--use-greedy`, validated) are 119 cycles at 4 rounds / batch 64 and 188 at 6 rounds /
batch 64 without barriers (143 and 240 with them), and 1,398 at 16 rounds with the wavefront
priority (2,007 by default). `--greedy-barriers none` alone does not build at 16 rounds
(peak liveness 2,113 words against a budget of 1,536 in address assignment).

**16 rounds is deferred.** Three configurations were run at 16 rounds on the wavefront warm
start with a budget of 60 (default objective; makespan-only; makespan-only with the ALAP cap
off and the tight horizon). All three returned 1,399 cycles, the repaired warm start and one
cycle more than the greedy schedule, with bound 1,271 (the capacity floor), after 2.2 to 2.5
hours of wall time each on a loaded machine; the first solution arrived only after about 27
minutes. The solver reported the hint as incomplete (114,988 of 115,628 variables hinted),
which is why it had to repair it rather than accept it. A fix that hints the missing CSE
shared-subtract scratch variables is in progress in another PR; the 16-round rows are
deferred until it is on `main`, because measuring the solver switches against a warm start
the solver cannot use says nothing about the switches.

#### Cycles by configuration and instance

| Configuration | 4r / b64 (budget 20) | 6r / b64 (budget 20) |
|---|---|---|
| *greedy hint* | 119 | 188 |
| nb-baseline | 119 | 188 |
| nb-makespan-only | 114 | 188 |
| nb-alap-slack-none-tight-horizon | 119 | 188 |
| nb-makespan+alap-5 | 116 | 188 |
| nb-makespan+alap-50 | 116 | 188 |
| nb-makespan+alap-none+tight-horizon | 115 | 188 |
| *capacity floor* | 94..95 | 130..131 |

Non-reproducible configurations show min / median / max over their runs. "optimal" is optimal for the emitted ops inside the model's domains, not for the problem.

#### 4 rounds, batch 64 (budget 20 deterministic time; greedy hint 119 cycles)

| Configuration | Cycles | vs baseline | Status | Bound (lb_cycles) | Floor | Final incumbent at: dtime (est) | wall s | Solver dtime | Solver wall s | Peak live | Valid |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nb-baseline | 119 | 0 | FEASIBLE | 93 | 95 | 0.9 | 4.0 | 21.8 | 102.5 | 577 | PROVED, PASSED |
| nb-makespan-only | 114 | -5 | FEASIBLE | 94 | 95 | 22.3 | 108.2 | 23.0 | 111.9 | 569 | PROVED, PASSED |
| nb-alap-slack-none-tight-horizon | 119 | 0 | FEASIBLE | 95 | 95 | 1.0 | 4.7 | 22.1 | 101.7 | 585 | PROVED, PASSED |
| nb-makespan+alap-5 | 116 | -3 | FEASIBLE | 94 | 94 | 12.4 | 62.0 | 22.0 | 109.5 | 569 | PROVED, PASSED |
| nb-makespan+alap-50 | 116 | -3 | FEASIBLE | 94 | 94 | 13.4 | 72.2 | 20.5 | 110.7 | 569 | PROVED, PASSED |
| nb-makespan+alap-none+tight-horizon | 115 | -4 | FEASIBLE | 95 | 94 | 20.4 | 90.0 | 22.9 | 101.5 | 569 | PROVED, PASSED |

#### 6 rounds, batch 64 (budget 20 deterministic time; greedy hint 188 cycles)

| Configuration | Cycles | vs baseline | Status | Bound (lb_cycles) | Floor | Final incumbent at: dtime (est) | wall s | Solver dtime | Solver wall s | Peak live | Valid |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nb-baseline | 188 | 0 | FEASIBLE | 155 | 131 | 1.3 | 6.0 | 20.9 | 97.3 | 585 | PROVED, PASSED |
| nb-makespan-only | 188 | 0 | FEASIBLE | 155 | 130 | 1.6 | 5.8 | 21.8 | 80.5 | 577 | PROVED, PASSED |
| nb-alap-slack-none-tight-horizon | 188 | 0 | FEASIBLE | 155 | 130 | 1.1 | 3.6 | 34.1 | 108.3 | 577 | PROVED, PASSED |
| nb-makespan+alap-5 | 188 | 0 | FEASIBLE | 155 | 130 | 1.3 | 5.1 | 20.2 | 82.3 | 577 | PROVED, PASSED |
| nb-makespan+alap-50 | 188 | 0 | FEASIBLE | 155 | 130 | 1.3 | 6.1 | 20.6 | 93.6 | 577 | PROVED, PASSED |
| nb-makespan+alap-none+tight-horizon | 188 | 0 | FEASIBLE | 155 | 130 | 2.3 | 5.3 | 47.2 | 110.1 | 577 | PROVED, PASSED |

Configurations:

- `nb-baseline`: `--greedy-barriers none`
- `nb-makespan-only`: `--greedy-barriers none --cpsat-objective makespan`
- `nb-alap-slack-none-tight-horizon`: `--greedy-barriers none --cpsat-greedy-alap-slack none --no-cpsat-horizon-num-ops --no-cpsat-horizon-max-latency`
- `nb-makespan+alap-5`: `--greedy-barriers none --cpsat-objective makespan --cpsat-greedy-alap-slack 5`
- `nb-makespan+alap-50`: `--greedy-barriers none --cpsat-objective makespan --cpsat-greedy-alap-slack 50`
- `nb-makespan+alap-none+tight-horizon`: `--greedy-barriers none --cpsat-objective makespan --cpsat-greedy-alap-slack none --no-cpsat-horizon-num-ops --no-cpsat-horizon-max-latency`

## 3. Option by option

### 3.1 Search mode and workers (`--cpsat-workers`, `--cpsat-interleave`, subsolver lists)

- **Worker count under interleaved search** (`--cpsat-workers 16`): 0 or +1 cycle. The
  subsolver set does not depend on the worker count in this mode; only the batch size does.
  One worker (`--no-cpsat-interleave --cpsat-workers 1`, reproducible) is worse: with the
  makespan-only objective 140 at 4 rounds / batch 64, where 8 interleaved workers reach 130.
- **Non-interleaved search** (`--no-cpsat-interleave`, 8 workers), 4 rounds / batch 64:
  129 / 129 / 129 cycles (3 runs), solver deterministic time 137 to 138, wall 82 to 98 s
  with the default objective; 120 / 121 / 121 cycles (3 runs), solver deterministic time 133 to
  134, wall 197 to 232 s with the makespan-only objective, against 139 and 130 interleaved. The
  comparison is not like for like: the budget is per worker, and non-interleaved workers all
  run for the whole budget at once, so the solver reported about 138 deterministic time in
  total against 21 interleaved (the wall times above were taken under different machine
  loads and do not compare). At 2 rounds / batch 256 (one
  run each, first session, result files lost): 193 with 8 workers and 193 with 12, the same as
  interleaved.
- **Bound workers** (`objective_lb_search`, `objective_shaving`, named in
  `--cpsat-extra-subsolvers` with 12 non-interleaved workers and the makespan-only objective):
  121 / 121 / 122 cycles (3 runs), solver deterministic time 156 to 158, wall 158 to 213 s;
  bound 95 / 95 / 95. The log lists the full-problem subsolvers that ran: [default_lp, fixed,
  max_lp_sym, no_lp, objective_lb_search, objective_shaving, quick_restart, reduced_costs].
- **A reproducible configuration with bound workers does not exist in 9.15** (section 5).
- On a small instance (2 rounds / batch 16) the non-interleaved portfolio proves optimality
  in 0.18 to 0.20 s of wall time against 1.8 s interleaved (first session, two runs).

Reading: without interleaving, all workers run for the whole per-worker budget at once, so a
budget of 20 is about 134 to 158 units of total solver work against about 21 interleaved. The
better makespans (120 to 122 against 130 with the makespan-only objective) are at least partly
that extra work. Adding `objective_lb_search` and `objective_shaving` (they do run in this mode,
as the subsolver list shows) did not move the bound from 95.

Verdict: no change of default; reproducible runs are worth more for development. For a
production run, `--no-cpsat-interleave` with the makespan-only objective deserves a comparison
at equal total work or equal wall time (an interleaved run with about six times the budget);
that was not done here.

### 3.2 Makespan-only objective (`--cpsat-objective makespan`)

The objective is `scale * makespan + sum(effective completion cycle of every op)`. With
`--cpsat-objective makespan` the tie-breaker, its auxiliary variables and their enforced
equalities are not created (4 rounds / batch 64: 8,390 to 7,844 variables, 38,738 to 37,661
constraints, 884 objective terms to 1).

- Cycles: sections 2.1 to 2.3. Better or equal on every pair measured, on both warm starts.
- The runs were still improving when the budget ran out: at 4 rounds / batch 64 the incumbent
  makespans were 140, 139, 138, 137, 135, 131, 130, the last one 21 s before the end of a
  135 s solve; the default objective produced 140 and 139.
- Time to OPTIMAL without a budget, same makespan, all validated (solver deterministic time,
  tie-breaker against makespan-only): 2 rounds / batch 16: 10.87 and 1.59; 3 rounds /
  batch 16: 11.19 and 1.61; 2 rounds / batch 32: 11.29 and 4.50.
- Peak liveness was lower than the default's in every pair measured (2r/b256 1113 against
  1129, 4r/b64 530 against 569, 6r/b64 544 against 565), although the schedule is not
  left-packed.
- Combined with each other switch (section 2.2) nothing beats makespan-only alone
  consistently.
- Safety and the two model differences: section 4.

Why the search does better without the tie-breaker was not investigated here; a separate
study is looking at that (incumbent traces, LNS, a two-phase solve). Section 4.3 is input for
it: part of the difference between the two objectives is not search behaviour at all.

Verdict: recommended as the default, after the follow-up of section 4.2.

### 3.3 Greedy ALAP cap and horizon (`--cpsat-greedy-alap-slack`, `--no-cpsat-horizon-num-ops`, `--no-cpsat-horizon-max-latency`)

Every op's latest cycle is capped at its greedy cycle + 15, and the horizon is
`max(greedy makespan + 20, max-latency critical path, number of ops)`. The op count sets the
horizon on every benchmark instance (2r/b256: 1,543 against a tight horizon of 213; 4r/b64:
883 against 160; 6r/b64: 1,284 against 255).

- Slack 5, 50 and none, and the tight horizon (both guards off): the baseline's cycles on
  every instance measured.
- Slack none with the tight horizon: 137 instead of 139 at 4 rounds / batch 64, no change
  elsewhere.
- Slack 5 lowers the reported bound by 1 or 2 on two instances (163 against 164, 93 against
  95).
- With the makespan-only objective (section 2.2): no direction.
- On the better warm start (section 2.3) the cap does not matter either.
- Nothing broke without the horizon guards: all tests pass and every run produced a validated
  kernel. The tight horizon relies on the greedy (plus per-section repair) schedule being
  feasible.
- Slack 0 exposed the defect of section 4.3; that is a property of the default objective,
  not of the cap.

Verdict: the cap is not what holds the solver back at these budgets, on either warm start.
It still makes "OPTIMAL" and the bound relative to a neighbourhood of the greedy schedule, so
keep the switch and re-measure with longer budgets. No change of default.

### 3.4 Makespan interval and `push_all_tasks_toward_start`

- `--cpsat-makespan-interval`: cycles 0 or +1 against the baseline, and a **lower** reported
  bound: 140 instead of 155 at 6 rounds / batch 64 and 163 instead of 164 at 2 rounds /
  batch 256. With the makespan-only objective: 129 / 234 / 193 against 130 / 230 / 192
  (4r/b64, 6r/b64, 2r/b256). Whether CP-SAT's makespan detection (`DetectMakespan`) fires
  with or without the interval is logged only at `VLOG(2)` and was not observed.
- `--cpsat-push-tasks-toward-start`: no change anywhere. With the makespan-only objective it
  is the same search as makespan-only alone (same cycles; solver deterministic time 21.7251
  against 21.7252 and 20.8022 against 20.8024).

Verdict: leave both off.

### 3.5 Extra cumulative propagators (`--cpsat-cumulative-propagators`)

All of them (`overload_checker`, `timetable_edge_finding` with the interval limit raised from
100 to 1,000,000, `hard_precedences`, which sets both `use_hard_precedences_in_cumulative`
and `exploit_all_precedences`) on every worker:

| Instance | Cycles: default | Cycles: all propagators | Wall time ratio |
|---|---|---|---|
| 2 rounds / batch 256 (first session) | 193 | 193 | 1.4 |
| 4 rounds / batch 64 | 139 | 139 | 2.0 (this session); see note |
| 6 rounds / batch 64 | 235 | 230 (bound 156 against 155) | 1.5 (this session); see note |

Deterministic time does not charge for the propagators in proportion to their wall time, so a
fixed deterministic budget hides their cost. The wall-time ratios are against the baseline
cell of the same instance run at a different time and machine load, so they are indicative
only; in an earlier session, with both runs under the same load, the ratios were 4.2 and 5.2.
The one gain (5 cycles at 6 rounds / batch 64)
is what the makespan-only objective gets there without the extra wall time. The propagators
were not measured one at a time, nor on a dedicated subsolver. `hard_precedences` without
`exploit_all_precedences` would have been a silent no-op in OR-Tools 9.15 (`cumulative.cc`
fills the precedence list only under the second parameter); the switch therefore sets both,
and no measurement here used one without the other.

Verdict: leave off.

### 3.6 Redundant rows (`--disable-cpsat-constraints conditional_makespan,flow_cumulative`)

The per-op `makespan >= cycle + 1` rows are implied by `makespan = max_cycle + 1` and the
unconditional `max_cycle >= ...` rows; the capacity-1 Flow cumulative repeats
`no_overlap_flow`. Leaving both out removes 1,414 of 38,738 constraints at 4 rounds /
batch 64 and changes nothing: the run is the same search as the default one (same cycles,
same bound, same solver deterministic time to four decimals: 61.3746 at 2r/b256, 21.0486 at
4r/b64, 21.9528 at 6r/b64). CP-SAT's presolve already removes them.

Verdict: no effect; no change of default. Whether presolve time at 16 rounds (about 20,000
such rows) is affected was not measured.

## 4. The makespan-only objective: safety, drift, and a defect in the default objective

### 4.1 Can an auxiliary variable drift unsafely without the tie-breaker?

Several auxiliary variables are bounded by inequalities only. With the tie-breaker gone,
nothing pulls them. Can one drift so that address assignment rejects the schedule
(`BudgetExceeded`) or a post-solve check fires?

**No, by construction, and no failure was observed in 43 validated
makespan-only builds.** Every such variable can only drift in the direction in which the
model reserves *more* than the schedule needs, and where a drifted value is read back out of
the solution, address assignment reads the same value the model accounted for.

| Variable | Constraint | Free direction | Read from the solution? | Effect of drift |
|---|---|---|---|---|
| `completion_c{i}` (shiftable op) | `>= c_i`, `>= every lane` | upward, up to `min(reader cycle) - 1` | Yes: it becomes the op's recorded cycle when shifted (`cycle_assignments`) | Readers are constrained against the same variable (`reader >= completion + 1`), so precedence holds. The op's operands are modelled live until `completion + 1`; address assignment times those reads at the recorded cycle, i.e. the same value (`address_assign.rs`, `use_cycles`: "each src VReg read at op.cycle"). Model and allocator agree; the operands are held longer than the lanes need. |
| `shifted_dest_start_{i}` | `<= every lane`; `= c_i` when not shifted | downward | No (address assignment uses `min(lanes)`) | The model reserves the destination earlier than its first write: over-reservation only. |
| `cse_sub_end` | `>= consumer cycle + 2` for active consumers | upward | No | The shared subtract temporary is modelled live longer: over-reservation only. |
| `live_end_{v}` (several readers) | `= max` over readers (`add_max_eq`) | none | No | Exact. |
| `select_fma_end_{i}` | two enforced equalities | none | No | Exact. |
| `live_size_{v}` | `start + size = end` | none | No | Determined by its endpoints. |
| `max_cycle` | `>=` every cycle, completion and FMA cycle; `makespan = max_cycle + 1` | none at the optimum: the objective is `makespan` | Yes (`makespan`) | Cannot exceed what the objective allows. |
| `shift_chosen_cycle_*`, `select_chosen_cycle_*` | tie-breaker only | | | Not created in this mode. |

So a makespan-only solution that satisfies the scratch cumulative is within budget for
address assignment to exactly the extent that a default solution is: the objective was never
part of that argument. The comment in the scratch model that "the objective will naturally
push `start_var` to the minimum" describes tightness, not soundness.

The checks after the solve ran on every build in the tables: per-cycle slot limits, RAW
precedence including the multiply-add latency, CSE subtract precedence, the scratch budget
recomputed from the solution with every reader counted (`verify_scratch_budget_all_readers`),
then address assignment itself, the dataflow proof and numeric validation.

Not shown: that the scratch model and address assignment agree in every other respect. That
is the pre-existing obligation of the scratch model and does not depend on the objective.

### 4.2 The drift is real and costs scratch

Shifted ops whose recorded cycle (the `completion` variable) is later than their last lane:

| Instance | Objective | Shifted ops | Recorded later than the last lane | Largest gap (cycles) | Peak live words | Cycles |
|---|---|---|---|---|---|---|
| 2 rounds / batch 256 | makespan+tiebreak | 180 | 0 | 0 | 1129 | 193 |
| | makespan | 168 | 10 | 14 | 1113 | 192 |
| 4 rounds / batch 64 | makespan+tiebreak | 71 | 0 | 0 | 569 | 139 |
| | makespan | 71 | 13 | 58 | 530 | 130 |
| 6 rounds / batch 64 | makespan+tiebreak | 61 | 0 | 0 | 565 | 235 |
| | makespan | 63 | 4 | 18 | 544 | 230 |

With the tie-breaker, `completion` of a shifted op is in the objective and sits on the last
lane. Without it, some shifted ops report a cycle after they actually finished: `completion`
is above the last lane. Consequences: (1) address assignment times those ops' reads at the
recorded cycle, so their operands stay allocated until then; the model accounts for exactly
the same value, so the budget holds; (2) `scheduler_output.json` and the tools that read it
show a cycle for the op at which nothing of it executes. Neither is a correctness problem,
and peak liveness was still lower with makespan-only in all three pairs.

The default objective is not immune when the solve stops with the tie-breaker unconverged:
the `--cpsat-makespan-interval` run at 2 rounds / batch 256 (default objective) has 2 of 176
shifted ops recorded up to 88 cycles after their last lane.

Follow-up, not done here: set the recorded cycle of a shifted op to its last lane after the
solve (always valid: it only shortens lifetimes, and readers already come later), or state
`completion = max(lanes)` in the model. It should precede making makespan-only the default at
sizes where scratch is tight. It was left out of these PRs because it changes what the
default configuration emits whenever a default solve stops early.

### 4.3 The default objective forbids the multiply-add path for a select at its latest cycle

Found by a zero-slack test of the greedy ALAP cap. `select_chosen_cycle_{i}`, an auxiliary of
the tie-breaker, is declared with the domain `[asap_i, alap_i]`, but on the multiply-add
(FMA) path it is constrained to equal the op's cycle + 1. A select scheduled at its ALAP
therefore cannot take the FMA path under the default objective: the tie-breaker's own
bookkeeping variable removes feasible schedules. The makespan-only objective does not create
the variable and has no such restriction.

Evidence: a proven-optimal schedule of a small program (4 selects, 7 shifted ops, 2 selects
on the FMA path), given back as the warm start with `--cpsat-greedy-alap-slack 0`, is
INFEASIBLE at presolve under the default objective, with every tightening mode, and solves
to the same makespan under `--cpsat-objective makespan`. With slack 15 or no cap both
objectives solve. With the default slack of 15 the edge (a select exactly 15 cycles after its
greedy cycle, or at the horizon) is rarely reached, which is why it went unnoticed.

Meaning for the comparison of the two objectives: part of the difference is not search
behaviour. The default model is slightly smaller than the makespan-only model in the
schedules it admits. On the benchmark runs of this study the effect is nil: the fix (domain
to `alap + 1`, PR #40) leaves the default objective's cycles, bound and select encodings
unchanged on all five instances at a budget of 20 (the solver's deterministic time moves in
the third or fourth digit on four of them, so the model did change). It can bind only when a
select sits at its ALAP, i.e. 15 cycles after its warm-start cycle by default.

The sibling variables were checked for the same pattern (a value that is op cycle + k on some
path needs a domain reaching `alap + k`): `fma_cycle` has `[asap + 1, alap + 1]`;
`completion` and the lane cycles equal a cycle of the op itself (k = 0) and have
`[asap, alap]`; `shift_chosen_cycle` equals `completion` or the op cycle (k = 0); the
ConstChoice tie-breaker term is the op's cycle variable itself; `select_fma_end`, `live_end`
and `cse_sub_end` have `[0, horizon + 2]`; `cse_sub_cycle` reaches `horizon - 1`;
`live_size` and `cse_sub_size` reach `horizon + 1` and `horizon + 2`. Only
`select_chosen_cycle` was short by one.

## 5. What OR-Tools 9.15 does with the search parameters

Read from `ortools/sat/cp_model_search.cc`, `parameters_validation.cc` and `cumulative.cc` at
tag `v9.15`, and confirmed in solver logs ("N full problem subsolvers: [...]").

- With `interleave_search`, the subsolvers `probing*`, `objective_shaving*`,
  `lb_tree_search*` and `objective_lb_search*` are skipped for any model with an objective,
  including when they are named in `subsolvers` or `extra_subsolvers`. Run with
  `--cpsat-extra-subsolvers objective_lb_search,objective_shaving,lb_tree_search,probing`
  the log lists the same ten subsolvers as without. **There is no reproducible multi-worker
  configuration with bound workers in 9.15.**
- With `interleave_search`, every remaining subsolver runs whatever `num_workers` is; the
  worker count only sets the batch size (24 tasks at 8 workers, 48 at 16).
- Without it, the list is cut to a length derived from `num_workers` (6 full-problem
  subsolvers at 8 workers, 8 at 12). The default order puts the bound workers after the first
  nine, so they do not run at 8 or 12 workers unless named in `--cpsat-extra-subsolvers`,
  which go to the front of the list.
- With a one-term objective (makespan-only) the `core` subsolver is dropped (9 full-problem
  subsolvers instead of 10).
- An unknown name in `subsolvers` / `extra_subsolvers` aborts the process inside CP-SAT, and
  `num_workers` above 10,000 is rejected as MODEL_INVALID; the binding checks both first.
- `use_hard_precedences_in_cumulative` adds constraints only together with
  `exploit_all_precedences`.
- All parameter names used by the switches exist in the `SatParameters` proto of the `cp_sat`
  fork (`interleave_search`, `num_workers`, `subsolvers`, `extra_subsolvers`,
  `ignore_subsolvers`, `push_all_tasks_toward_start`, `use_overload_checker_in_cumulative`,
  `use_timetable_edge_finding_in_cumulative`,
  `max_num_intervals_for_timetable_edge_finding`, `use_hard_precedences_in_cumulative`,
  `exploit_all_precedences`).
- **The hint is incomplete from 3 rounds up.** At 16 rounds the solver reports "The solution
  hint is incomplete: 114988 out of 115628 non fixed variables hinted" and the first solution
  arrives from a `[hint]`-following worker only after the first batch, about 25 minutes into
  the solve on the loaded machine, at 1,399 cycles: one more than the greedy schedule it was
  given. At 2 rounds / batch 256 the hint is complete and feasible and is accepted before
  search starts; at 4 and 6 rounds / batch 64 it is not. According to the greedy work running
  in parallel with this study, the unhinted variables are the CSE shared-subtract scratch
  variables (`cse_sub_end_N`, `cse_sub_size_N`), and with them hinted the 3-round log reads
  "complete and is feasible"; that fix is a separate PR and is **not** in the build measured
  here. All rows from 3 rounds up, and the 16-round rows in particular, were therefore
  measured with an incomplete hint and should be rerun once it lands.

## 6. Recommendation

1. **Make `--cpsat-objective makespan` the default**, after pinning the recorded cycle of a
   shifted op to its last lane (section 4.2). It was better or equal on every instance and
   warm start measured and it is the only switch for which that is true. Until the pin is in
   it is still safe to use; it only wastes scratch.
2. **Fix `select_chosen_cycle`** (section 4.3) regardless: the default model should not
   exclude feasible schedules.
3. **Keep interleaved search and 8 workers** as the default: reproducible results are worth
   more for development. Compare non-interleaved search at equal total work before using it
   for production runs (section 3.1).
4. **Leave the greedy ALAP cap and the horizon guards as they are**, and re-measure them with
   longer budgets. They cost nothing and help nothing at 20, on either warm start.
5. **Do not enable** the makespan interval, `push_all_tasks_toward_start`, the cumulative
   propagators, or the removal of the redundant rows.
6. **Spend effort on the warm start and on the hint, not on solver parameters.** The greedy
   shape options moved the schedule by tens of percent; no solver switch moved it by more
   than a few cycles. Once the hint is complete, rerun the 16-round rows (the comparison
   that matters for the project).

## 7. Not measured

- Budgets other than 20. Several runs were still improving at the end.
- 16 rounds (deferred until the hint is complete; section 2.3).
- Wall-clock-equal comparisons. Interleaved and non-interleaved runs with the same
  deterministic budget do very different amounts of work.
- Non-interleaved search on anything but 4 rounds / batch 64 (three runs) and 2 rounds /
  batch 256 (one run each, first session; result files lost).
- 4 rounds / batch 256 for the switches other than the makespan-only objective and the ALAP
  cap with the tight horizon.
- Each cumulative propagator on its own; propagators on a dedicated subsolver
  (`subsolver_params`); propagators beyond 2r/b256, 4r/b64 and 6r/b64.
- Whether CP-SAT detects the makespan of each cumulative, with or without the makespan
  interval; the root LP bound as such (only the bound at the end of the budget is recorded).
- `--cpsat-subsolvers` (replacing the list) and `--cpsat-ignore-subsolvers`: exercised by
  tests only.
- Why the makespan-only objective searches better; a two-phase solve (makespan, then
  tie-breaker). The effect of the section 4.3 fix on the makespan-only comparison (the
  measured build does not contain it).
- Presolve time and model size effects of the horizon guards and the redundant rows at 16
  rounds.
- The 16-round warm start with all greedy shape options (`--greedy-split-shift` and
  `--pressure-inflation-threshold` were not on `main`).
- The remaining items of section 10 of the modelling review: lane re-encoding (proposal 2,
  excluded on purpose), hint completeness, `select_fma_extra` / `fma_cycle` / constant
  variables, prefix and suffix capacity rows.
- Variance over seeds.

## 8. Reproduction

```bash
./scripts/build_scheduler_rs.sh --release

# Baseline and every switch alone (reproducible configurations may run in parallel)
uv run python -m scripts.run_cpsat_ablation --out-dir build/cpsat_ablation --jobs 3

# Combinations with the makespan-only objective
MK="--cpsat-objective makespan"
TIGHT="--no-cpsat-horizon-num-ops --no-cpsat-horizon-max-latency"
uv run python -m scripts.run_cpsat_ablation --instances 4x64,6x64,2x256 --out-dir build/cpsat_ablation \
    --name combos --config "baseline:" --config "makespan-only:$MK" \
    --config "makespan+tight-horizon:$MK $TIGHT" \
    --config "makespan+alap-5:$MK --cpsat-greedy-alap-slack 5" \
    --config "makespan+alap-50:$MK --cpsat-greedy-alap-slack 50" \
    --config "makespan+alap-none+tight-horizon:$MK --cpsat-greedy-alap-slack none $TIGHT" \
    --config "makespan+workers-16:$MK --cpsat-workers 16" \
    --config "makespan+push:$MK --cpsat-push-tasks-toward-start" \
    --config "makespan+makespan-interval:$MK --cpsat-makespan-interval"

# Better warm start (greedy without round barriers)
NB="--greedy-barriers none"
uv run python -m scripts.run_cpsat_ablation --instances 4x64,6x64 --out-dir build/cpsat_ablation \
    --name nobarriers --config "nb-baseline:$NB" --config "nb-makespan-only:$NB $MK"

# Default model unchanged: export and diff
CPSAT_EXPORT_MODEL=a.pb uv run generate_ssa_kernel.py --rounds 2 --batch-size 16 \
    --cpsat-deterministic-time 0.5 --no-validate -q
uv run python -m scripts.diff_cpsat_models a.pb b.pb
```

`build/` is git-ignored; keep run outputs there rather than in `/tmp`.

OR-Tools sources read: tag `v9.15`, `ortools/sat/cp_model_search.cc`,
`ortools/sat/parameters_validation.cc`, `ortools/sat/cumulative.cc`.
