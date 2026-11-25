# Greedy schedule shape: ablation

November 2026. Every number here was produced in this session with the commands in
section 10, greedy scheduler unless a row says CP-SAT, `forest_height=10`, `batch_size=256`.
Every build was run with `--validate`; a row with numbers was PROVED (symbolic dataflow
replay) and PASSED (numeric validation), and a row that says "fails" did not produce a
kernel. Tables come from `uv run python -m scripts.schedule_report --table`.

## 1. Result

| 16 rounds, greedy | Schedule cycles | Simulator cycles | Gap to the 1,271 floor | Scratch peak |
|---|---|---|---|---|
| Default | 2,007 | 1,994 | 736 (57.9%) | 1,268 |
| Wavefront + split shift + inflation threshold 0.95 | **1,313** | **1,315** | 42 (3.3%) | 1,244 |
| Same with stagger 10 and 20 groups in flight | 1,308 | 1,310 | 37 (2.9%) | 1,364 |

```bash
uv run generate_ssa_kernel.py --rounds 16 --use-greedy --validate \
  --greedy-barriers none --greedy-priority wavefront --greedy-split-shift \
  --pressure-inflation-threshold 0.95
```

Rails for that command: PROVED, PASSED, greedy counter agrees with interval liveness, slot
limits hold, address assignment peak 1,244 of 1,536, two runs give the same kernel, and
`./scripts/check_upstream.sh --strict` with the same options passes upstream's CorrectnessTests and
every speed threshold at 1,315 cycles.

**This is a greedy-only result.** By default split shift is switched off whenever CP-SAT
follows the greedy, because CP-SAT's hint then carries one cycle per op, and CP-SAT starts
from the wavefront schedule without split shifts (1,398 cycles at 16 rounds, not 1,313).
`--cpsat-split-shift-hint` hands CP-SAT the split-shift schedule lane by lane; section 6
has the measurements.

Defaults are unchanged. The four switches are independent options; sections 4 and 5 give
each alone and combined.

The floor (1,271) is for the ops this emission contains. It moves when emission changes;
the shape options do not depend on the op count.

## 2. What serialises rounds in the default greedy

Measured by switching each suspect off separately.

1. **Barrier dependencies make it strict.** Every op after a barrier depends on it, and it
   depends on every op of the round before, so consecutive rounds overlap by exactly 0
   cycles. A gather round then takes about 143 cycles because 256 loads need 128 cycles of
   the two load slots, while the vector ALU is at about 61%; a select round leaves the load
   slots idle.
2. **Ready-queue order keeps rounds abreast without barriers.** The queue is ordered by the
   cycle an op became ready, then emission order, and emission is round-major (all 32
   groups of a round, then the next round). An op that has waited goes before one that
   just became ready, so no group gets ahead. With `--greedy-barriers none` alone all 32
   groups still reach each gather round together: 8 rounds improve from 1,004 to 951
   cycles, the gather rounds are still paced by the load slots (100% busy, vector ALU about
   71%), and at 16 rounds the temporaries of 32 groups in flight exceed the scratch budget
   (peak 2,113) and the build fails.
3. **The pressure policy does not serialise rounds**, but its throttle decides what happens
   when scratch is tight. ASAP inflation defers any producer with slack by up to 10 cycles
   once live words pass 80% of the budget. With barriers every op's slack is bounded by its
   round; without them almost every op has slack, so the throttle defers almost everything.
   This is the cliff in section 5.

So the barrier is the hard cause, the queue order is why removing the barrier is not
enough, and the throttle is why naive mixing of rounds gets slower instead of faster.

## 3. The options

| Option | Default | What it does |
|---|---|---|
| `--greedy-barriers {round,none}` | `round` | `none` drops the barrier pseudo-ops before dependencies are computed. |
| `--greedy-priority {fifo,wavefront}` | `fifo` | `wavefront` tries ready ops in the order of `depth + stagger * group rank`. |
| `--greedy-stagger-cycles N` | 12 | Wavefront: delay between consecutive groups. |
| `--greedy-groups-in-flight N` | 16 | Wavefront: groups allowed to run at once (0 = no limit). |
| `--greedy-split-shift` | off | A vector op run as 8 scalar ops may straddle two cycles. Greedy-only. |
| `--greedy-justify N` | 0 (off) | Forward/backward justification of the finished greedy schedule, at most N iterations (section 11). |
| `--pressure-inflation-threshold F` | policy's 0.80 | Fraction of the budget above which producers with slack are deferred. |
| `--pressure-soft-threshold F`, `--pressure-critical-threshold F` | 0.85, 0.95 | The policy's other two thresholds. |
| `--greedy-budget-headroom W` | 0 | Scratch words the greedy keeps free. |

**Wavefront priority.** The 32 groups are independent chains that share only constants and
cached tree vectors. `depth` is an op's longest dependency path from the start; `group rank`
is the position of its chain in emission order. Trying ready ops in key order makes group g
run as if it started `stagger * g` cycles after group 0, so at any time some groups are
gathering (using the load slots) while others hash or select (using the ALUs). The scheduler
stays work-conserving: a lower-priority op still runs when a slot is free. Chains are found
from the dependency graph (an op that reaches more final stores than the median op does is
shared; chains are the connected components of the rest), because the emission's group tag
is only set during tree access. The window on groups in flight bounds scratch: a group that has not
started holds nothing and a finished group has stored its results. "Priority that fills idle
engines" and "explicit group staggering" turned out to be this one key; a separate rule
preferring load-feeding ops was not needed, because with staggered groups the load slots
always have a group to serve (81% busy overall, 100% for long stretches).

**Split shift.** The greedy shifts a vector op to the scalar ALU only when the six vector
slots are full, and a whole shifted op needs 8 of the 12 scalar slots in one cycle, so the
greedy tops out at 7 vector-op equivalents per cycle while the floor assumes 7.5. With split
shifts the leftover scalar slots take the first lanes of one more op and the next cycle takes
the rest (8 + 4, then 4 + 8). The op's cycle is its last lane's; the per-lane cycles go to
`shifted_alu_cycles`, which lowering, address assignment and the dataflow proof already
handle for CP-SAT's shifted ops. It is switched off when CP-SAT follows, because CP-SAT's
hint has one cycle per op.

In the default schedule the scalar ALU is idle in gather rounds for a different reason: the
load slots pace the round, the vector slots are never full, and nothing asks to shift. The
pressure policy's `allow_shift` only suppresses shifts above 85% of the budget and is not
the cause.

**Inflation threshold.** PR #18 made the greedy's live-word counter exact. The 0.80
threshold was tuned against a counter that undercounted, so it now fires earlier than
intended. See section 5.

## 4. Ablation tables

Columns: schedule length, simulator cycles (what `problem.Machine` and the upstream tests
count), capacity floor and gap from `schedule_report`, scratch peak from address
assignment, utilisation of each engine over the whole schedule, and total round overlap
(cycles each round shares with the previous one, summed; 0 = strictly one after another).
"Wavefront" rows use `--greedy-barriers none`; wavefront with barriers left on is in
section 8.

### 2 rounds

| Run | Rounds | Cycles | Simulator | Floor | Gap | Peak words | valu % | alu % | load % | flow % | Round overlap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| default | 2 | 193 | 194 | 163 | 30 (18.4%) | 1113 | 89.5 | 62.0 | 20.7 | 35.2 | 0 |
| barriers none | 2 | 191 | 193 | 163 | 28 (17.2%) | 1089 | 90.1 | 64.0 | 20.9 | 35.6 | 83 |
| wavefront (barriers none) | 2 | 197 | 199 | 163 | 34 (20.9%) | 556 | 90.4 | 49.9 | 21.3 | 32.5 | 173 |
| split shift | 2 | 200 | 201 | 163 | 37 (22.7%) | 1098 | 81.2 | 80.2 | 21.0 | 32.0 | 0 |
| inflation 0.95 | 2 | 193 | 194 | 163 | 30 (18.4%) | 1113 | 89.5 | 62.0 | 20.7 | 35.2 | 0 |
| wavefront + split shift | 2 | 189 | 191 | 163 | 26 (16.0%) | 580 | 92.1 | 60.5 | 22.2 | 33.9 | 165 |
| wavefront + split shift + inflation 0.95 | 2 | 189 | 191 | 163 | 26 (16.0%) | 580 | 92.1 | 60.5 | 22.2 | 33.9 | 165 |
| same, stagger 10, 20 in flight | 2 | 194 | 196 | 163 | 31 (19.0%) | 684 | 90.5 | 55.5 | 22.2 | 32.0 | 170 |

### 3 rounds

| Run | Rounds | Cycles | Simulator | Floor | Gap | Peak words | valu % | alu % | load % | flow % | Round overlap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| default | 3 | 326 | 326 | 249 | 77 (30.9%) | 1271 | 82.0 | 53.7 | 12.6 | 50.9 | 0 |
| barriers none | 3 | 319 | 321 | 249 | 70 (28.1%) | 1265 | 83.3 | 56.6 | 12.9 | 52.0 | 87 |
| wavefront (barriers none) | 3 | 300 | 302 | 249 | 51 (20.5%) | 740 | 91.0 | 50.6 | 14.0 | 54.7 | 530 |
| split shift | 3 | 324 | 324 | 249 | 75 (30.1%) | 1274 | 77.5 | 74.0 | 13.6 | 49.4 | 0 |
| inflation 0.95 | 3 | 289 | 289 | 249 | 40 (16.1%) | 1489 | 92.6 | 60.1 | 14.2 | 57.4 | 0 |
| wavefront + split shift | 3 | 287 | 289 | 249 | 38 (15.3%) | 764 | 93.0 | 61.5 | 15.7 | 55.1 | 504 |
| wavefront + split shift + inflation 0.95 | 3 | 287 | 289 | 249 | 38 (15.3%) | 764 | 93.0 | 61.5 | 15.7 | 55.1 | 504 |
| same, stagger 10, 20 in flight | 3 | 290 | 292 | 249 | 41 (16.5%) | 876 | 92.5 | 59.0 | 15.7 | 54.1 | 511 |

### 4 rounds

| Run | Rounds | Cycles | Simulator | Floor | Gap | Peak words | valu % | alu % | load % | flow % | Round overlap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| default | 4 | 456 | 455 | 357 | 99 (27.7%) | 1266 | 84.3 | 54.3 | 9.6 | 57.9 | 0 |
| barriers none | 4 | 454 | 456 | 357 | 97 (27.2%) | 1265 | 84.4 | 55.7 | 9.7 | 58.1 | 375 |
| wavefront (barriers none) | 4 | 425 | 427 | 357 | 68 (19.0%) | 812 | 93.0 | 47.9 | 11.5 | 59.8 | 1128 |
| split shift | 4 | 455 | 454 | 357 | 98 (27.5%) | 1274 | 78.9 | 76.8 | 10.5 | 56.3 | 0 |
| inflation 0.95 | 4 | 437 | 436 | 357 | 80 (22.4%) | 1482 | 88.8 | 53.1 | 10.1 | 60.4 | 0 |
| wavefront + split shift | 4 | 423 | 425 | 357 | 66 (18.5%) | 828 | 92.2 | 53.0 | 12.1 | 59.1 | 1110 |
| wavefront + split shift + inflation 0.95 | 4 | 423 | 425 | 357 | 66 (18.5%) | 828 | 92.2 | 53.0 | 12.1 | 59.1 | 1110 |
| same, stagger 10, 20 in flight | 4 | 409 | 411 | 357 | 52 (14.6%) | 956 | 92.6 | 66.1 | 11.2 | 63.6 | 1067 |

### 8 rounds

| Run | Rounds | Cycles | Simulator | Floor | Gap | Peak words | valu % | alu % | load % | flow % | Round overlap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| default | 8 | 1004 | 999 | 648 | 356 (54.9%) | 1267 | 74.2 | 25.6 | 55.4 | 26.3 | 0 |
| barriers none | 8 | 951 | 953 | 648 | 303 (46.8%) | 1265 | 78.4 | 26.8 | 58.5 | 27.8 | 1171 |
| wavefront (barriers none) | 8 | 746 | 748 | 648 | 98 (15.1%) | 780 | 94.0 | 57.9 | 74.7 | 35.1 | 4264 |
| split shift | 8 | 988 | 983 | 648 | 340 (52.5%) | 1275 | 72.7 | 36.6 | 56.7 | 25.9 | 0 |
| inflation 0.95 | 8 | 987 | 982 | 648 | 339 (52.3%) | 1483 | 75.7 | 24.9 | 56.3 | 26.7 | 0 |
| wavefront + split shift | 8 | 709 | 711 | 648 | 61 (9.4%) | 828 | 93.6 | 82.2 | 78.6 | 37.0 | 4003 |
| wavefront + split shift + inflation 0.95 | 8 | 709 | 711 | 648 | 61 (9.4%) | 828 | 93.6 | 82.2 | 78.6 | 37.0 | 4003 |
| same, stagger 10, 20 in flight | 8 | 706 | 708 | 648 | 58 (9.0%) | 900 | 93.6 | 83.9 | 79.8 | 35.3 | 3933 |

### 16 rounds

| Run | Rounds | Cycles | Simulator | Floor | Gap | Peak words | valu % | alu % | load % | flow % | Round overlap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| default | 16 | 2007 | 1994 | 1271 | 736 (57.9%) | 1268 | 72.9 | 24.7 | 53.2 | 24.3 | 0 |
| barriers none | 16 | fails: address assignment peak 2113 > 1536 | | | | | | | | | |
| wavefront (barriers none) | 16 | 1398 | 1400 | 1271 | 127 (10.0%) | 1156 | 97.7 | 63.5 | 76.5 | 34.7 | 17024 |
| split shift | 16 | 1970 | 1957 | 1271 | 699 (55.0%) | 1276 | 71.6 | 35.8 | 54.4 | 24.4 | 0 |
| inflation 0.95 | 16 | 1965 | 1952 | 1271 | 694 (54.6%) | 1490 | 74.7 | 24.5 | 54.4 | 24.8 | 0 |
| wavefront + split shift | 16 | 1314 | 1316 | 1271 | 43 (3.4%) | 1260 | 97.4 | 93.6 | 81.5 | 36.8 | 15929 |
| wavefront + split shift + inflation 0.95 | 16 | 1313 | 1315 | 1271 | 42 (3.3%) | 1244 | 97.5 | 93.7 | 81.5 | 36.8 | 15921 |
| same, stagger 10, 20 in flight | 16 | 1308 | 1310 | 1271 | 37 (2.9%) | 1364 | 97.8 | 94.3 | 82.3 | 36.1 | 15235 |

Reading the tables:

- The simulator counts only bundles that contain an op. With barriers each barrier leaves
  an empty bundle, so the default's simulator count is below its schedule length; without
  barriers it is the schedule length plus the two pause bundles. Compare simulator cycles.
- The wavefront is the large effect wherever there are gather rounds (8 and 16 rounds).
  At 2 to 4 rounds there are none; the gain there comes from running the select rounds of
  some groups beside the hash of others, and it is smaller. At 2 rounds the wavefront alone
  is 5 cycles worse than the default.
- Split shift alone is worth 1 to 2% (and is 7 cycles worse at 2 rounds): the default
  schedule is not short of scalar slots. On top of the wavefront it is worth 6% at 16
  rounds, because there both ALUs are the bottleneck.
- The inflation threshold alone is the cheapest change: 3 rounds 326 to 289, 4 rounds 456
  to 437, 16 rounds 2,007 to 1,965. With the default wavefront window it changes almost
  nothing, because 16 groups in flight stay under 80% of the budget most of the time; its
  value there is robustness (section 5).
- At 16 rounds with the full combination both ALUs are at 100% from cycle 110 to cycle
  1,150. What is left over the floor is the first 110 cycles (no group has reached a gather
  round, so the load slots are half idle; that costs nothing since the ALUs are full) and
  the last 80 cycles, where the final groups run at critical-path speed with slots free.

## 5. Parameter sweeps (16 rounds)

All sweeps in this section were re-run on the current scheduler (after the #29 review
fixes and the #32 review fixes), one `generate_ssa_kernel.py --use-greedy --validate` run per
cell; every cell with numbers is PROVED and PASSED.

Wavefront stagger and window, without split shift, default pressure policy. Schedule
cycles, scratch peak in parentheses:

| Stagger (rows), groups in flight (columns) | 12 | 14 | 16 | 18 | 20 | 24 | no limit |
|---|---|---|---|---|---|---|---|
| 8 | 1538 (1012) | 1487 (1112) | 1441 (1228) | 2687 (1334) | 2777 (1398) | 2844 (1413) | 2834 (1378) |
| 10 | 1483 (988) | 1434 (1088) | 1395 (1196) | 1563 (1288) | 2796 (1388) | 2826 (1396) | 2840 (1354) |
| 12 | 1460 (980) | 1409 (1088) | 1398 (1156) | 1398 (1232) | 1398 (1268) | 2737 (1388) | 2808 (1360) |
| 14 | 1438 (980) | 1405 (1056) | 1404 (1108) | 1404 (1152) | 1404 (1220) | 1411 (1268) | 2813 (1394) |

The cliff on the right is the throttle of section 2: once about 1,280 words are live
(80% of the budget plus what is in flight) almost every op has enough slack to be
deferred, and the schedule roughly doubles. A smaller stagger reaches that peak with fewer
groups in flight. With split shift the peak is higher and the cliff is closer; with
`--pressure-inflation-threshold 0.95` it is gone:

| Wavefront + split shift | built-in 0.80 | inflation 0.95 |
|---|---|---|
| stagger 12, 14 in flight | 1347 (1128) | 1347 (1128) |
| stagger 12, 16 in flight | 1314 (1260) | 1313 (1244) |
| stagger 12, 18 in flight | 2885 (1352) | 1313 (1262) |
| stagger 12, 20 in flight | 2800 (1444) | 1313 (1276) |
| stagger 12, 24 in flight | 2793 (1448) | 1313 (1330) |
| stagger 12, no limit | 2816 (1434) | 1313 (1336) |
| stagger 10, 18 in flight | 1453 (1298) | 1309 (1296) |
| stagger 10, 20 in flight | 2749 (1398) | 1308 (1364) |
| stagger 11 / 13 / 14, 16 in flight | 1314 / 1317 / 1323 | 1314 / 1317 / 1323 |

The first version of this table (before the #29 review fixes) had 1,549 and 1,770 at 18
and 20 in flight with the built-in threshold; on the current scheduler those cells are on
the far side of the cliff (2,885 and 2,800). The conclusion is the same and stronger: with
the built-in threshold the window must stay at 16 or below; with 0.95 it does not matter.

Inflation threshold on the default shape (barriers on, first-come-first-served queue).
Schedule cycles, scratch peak in parentheses (re-run: identical to the first version):

| Threshold | 3 rounds | 4 rounds | 16 rounds |
|---|---|---|---|
| 0.80 (built-in) | 326 (1271) | 456 (1266) | 2007 (1268) |
| 0.85 | 317 (1345) | 446 (1338) | 1988 (1340) |
| 0.90 | 304 (1417) | 442 (1418) | 1973 (1420) |
| 0.93 | 295 (1465) | 439 (1466) | 1967 (1466) |
| 0.95 | 289 (1489) | 437 (1482) | 1965 (1490) |
| 0.97 | 289 (1513) | 434 (1506) | 1962 (1514) |
| 0.98 | fails (1537) | 434 (1521) | fails (1540) |
| 0.99, 1.0 (off) | fails | fails | fails |

- The peak follows the threshold (0.95 of 1,535 is 1,458; the peak is 30 to 40 words above
  the threshold at every setting). 0.95 leaves 46 words; 0.97 leaves 22 and is one step
  from failing. 0.95 is the value to use.
- With inflation off the greedy runs into the hard budget, and either force-schedules past
  it or address assignment needs 1 to 4 words more than the counter's cycle-end value. The
  pressure policies without inflation (`greedy`, `throttled`, `gated`) fail the same way at
  3, 4 and 16 rounds on the default shape. Inflation is the mechanism that keeps the default
  greedy inside the budget; it is only set too early.
- `--pressure-soft-threshold 0.95 --pressure-critical-threshold 1.0` changed no schedule
  (same cycles as the built-in values at 3, 4 and 16 rounds).
- `--greedy-budget-headroom` 8, 16 or 32 with inflation off still fails: headroom moves
  the wall, it does not stop the greedy from deadlocking at it. With inflation 0.95,
  headroom 16 gives 293 / 437 / 1,965: no gain over 0.95 alone.

Against the numbers quoted for PR #18 (greedy 446 to 456 at 4 rounds): inflation 0.85
gives 446 again and 0.95 gives 437, both inside the budget with the exact counter.

## 6. CP-SAT started from these greedy schedules

The greedy schedule is CP-SAT's warm start and bounds its search (every op's latest cycle
is capped at its greedy cycle + 15). Without `--cpsat-split-shift-hint`,
`--greedy-split-shift` is ignored in this pipeline, so the start is the wavefront schedule
without split shifts. All runs `--cpsat-deterministic-time 20` with a generous wall-clock
backstop (none stopped on it), PROVED and PASSED. The table below was measured before #41,
when the hint handed to CP-SAT from 3 rounds up was incomplete (the shared CSE
subtractions' scratch intervals were not hinted); the split-shift hint measurements further
down are on main after #41, with a complete hint.

| Rounds | Greedy start | Greedy cycles | CP-SAT cycles (simulator) |
|---|---|---|---|
| 3 | default | 326 | 325 (327) |
| 3 | inflation 0.95 (default shape) | 289 | **288 (290)** |
| 3 | wavefront + inflation 0.95 | 300 | 297 (299) |
| 4 | default | 456 | 455 (457) |
| 4 | wavefront + inflation 0.95 | 425 | **425 (427)** |

- In this budget the solver takes 0 to 3 cycles off whatever it is given. It polishes the
  shape of its start; it does not find a different shape. The quality of the greedy start
  is the quality of the result.
- At 3 rounds the best start is the default shape with the retuned threshold (288), 9
  cycles better than the wavefront start (297). There are no gather rounds at 3 rounds, so
  staggering has little to offer and the wavefront without split shifts runs at 7
  vector-op equivalents per cycle at best.
- At 4 rounds the wavefront start is 30 cycles (6.6%) better than the default start. The
  4-round run from "inflation 0.95, default shape" (greedy 437) was not made.
- An earlier attempt at the 3-round "inflation 0.95" row failed without an error message.
  It does not reproduce: the rerun above is the same command. The failed attempt coincided
  with a rebuild of the extension in the same checkout.
- Not measured: 16 rounds (the whole-program model did not move in 600 s before this
  work), longer solves, and CP-SAT from a split-shift start (below).

### Handing CP-SAT a split-shift schedule

`--cpsat-split-shift-hint` (with `--greedy-split-shift`) does this. The model already had
what a split shift needs: one cycle variable per lane of every shiftable op
(`alu_cycle_vars`), a completion variable that is their maximum, a scratch interval for the
destination that starts at the earliest lane, and lane-by-lane precedence. The change is on
the hint side only:

1. `SsaCpsatHints::shifted_lane_cycle_hints` maps an op id to its lane cycles and is filled
   from the greedy's `shifted_alu_cycles`.
2. Each lane variable is hinted at its own cycle (before, all eight lanes of a shifted op
   were hinted at the op's cycle).
3. The hint verifier ("Merged hint violates engine capacity") counts scalar slots lane by
   lane. Counted whole, a split-shift schedule has 16 scalar slots in a cycle and is
   rejected.
4. The hinted scratch interval of a split op's destination starts at its first lane.
5. `GreedySchedulePass` keeps split shifts for the CP-SAT pipeline when the flag is given.

Measured on main after #41 and #43 (complete greedy hints), shaped start
(`--greedy-barriers none --greedy-priority wavefront --pressure-inflation-threshold 0.95
--greedy-split-shift`), with and without `--cpsat-split-shift-hint`. Without the flag the
greedy that feeds CP-SAT drops its split shifts, so the start is the wavefront schedule
without them. Every run: the solver log says "The solution hint is complete and is
feasible", the first solution is the hint (`complete_hint`), PROVED and PASSED.

| Rounds | Budget (dtime) | Start without the hint | CP-SAT without (simulator) | Start with the hint | CP-SAT with (simulator) |
|---|---|---|---|---|---|
| 3 | 20 | 297 | 297 (299) | 289 | **289 (291)** |
| 4 | 20 | 425 | 425 (427) | 423 | **423 (425)** |
| 16 | 60 | 1399 | 1399 (1401) | 1314 | **1314 (1316)** |

- In every case CP-SAT returns its start unchanged in this budget, so what the hint buys is
  the start: 8, 2 and 85 cycles at 3, 4 and 16 rounds.
- The 16-round kernel with the hint (1,316 simulator cycles, scratch peak 1,182) passes
  `./scripts/check_upstream.sh --strict --kernel ...`: correctness and every speed threshold.
  It is one cycle longer than the greedy-only kernel (1,313 / 1,315): with CP-SAT following,
  the greedy runs with full bounds (ASAP deferral and cycle clamping), which costs a cycle.
- The hint is also complete and feasible under `--cpsat-objective makespan`,
  `--cpsat-makespan-interval` and `--select-choice fma` (3 rounds, batch 64).
- Wall time at 16 rounds, on a machine whose load average was 100 to 200: 1.5 hours with the
  hint (the solve ended on the deterministic budget), 6 hours without. The run without the
  hint ended on the 21,600 s wall-clock backstop, so its search depended on load; it found
  nothing better than its start after 1,012 s, and its start does not depend on load.
  The solver reported about 400 deterministic time units for both runs against a budget of
  60, so the "stopped on the wall-clock limit" warning, which compares those two numbers,
  did not fire (see BUGS.md).

## 7. With the emission options

The shape options do not depend on which ops are emitted, but chain detection does depend
on the dependency graph, so it was checked against the emission options (PRs #21, #31 and
#33 to #35, not on `main` when this was written; measured on a local merge of that stack
with this one, not on a pushed branch). 16 rounds, greedy; "default" is the default greedy
shape, "shaped" is the command of section 1. Schedule cycles (simulator cycles), all PROVED
and PASSED unless it says fails. The wavefront found 32 chains in every row.

| Emission options | default | shaped | Peak (shaped) |
|---|---|---|---|
| none | 2007 (1994) | 1313 (1315) | 1244 |
| `--merge-hash-stages` | 1973 (1960) | 1251 (1253) | 1236 |
| merge + `--index-form mirrored` | 1931 (1918) | 1197 (1199) | 1118 |
| merge + `--prune-dead-index` | 1959 (1946) | 1242 (1244) | 1061 |
| merge + `--prune-dead-index --values-only` (changes the output) | 1957 (1944) | 1229 (1231) | 1068 |
| merge + `--cached-select bits` | 2005 (1992) | 1183 (1185) | 803 |
| merge + `--cached-select bits-flow` | 2365 (2352) | 1172 (1174) | 851 |
| merge + `--cached-select mobius` | 1987 (1974) | 1213 (1215) | 819 |
| merge + `--cached-select bits --select-bit-source path` | fails (peak 1579) | 1168 (1170) | 859 |
| merge + `--tree-reduce-max-depth 4 --depth4-select-rounds after-wrap` | 2124 (2111) | 1308 (1310) | 1253 |
| merge + `--index-form mirrored --cached-select bits` | 1860 (1847) | 1153 (1155) | 771 |

- Emission savings that barely show in the default shape (the round-by-round schedule is
  paced by loads in gather rounds whatever the ALU work) show in full once the shape is
  fixed: the best row is 1,155 simulator cycles.
- This check found a bug in the first version of chain detection. With
  `--index-form mirrored` or `--prune-dead-index` all groups share one initial index
  vector, which reaches exactly half of the final stores (every index store, no value
  store). The first rule ("shared = reaches more than half of the stores") called it
  private, which joined all 32 groups into one chain; the build then failed the scratch
  budget or was hundreds of cycles longer. The rule is now "reaches more stores than the
  median op", with a regression test.
- Scratch peaks with the cached-select options are far below the budget (771 to 859), so
  more groups in flight should fit there; not swept.

## 8. Tried and not kept, and not done

- **Wavefront with barriers left on**: 207 / 356 / 529 / 1,078 / 2,150 cycles at 2 / 3 / 4 /
  8 / 16 rounds, longer than the default at every size. Through the barrier chain every op
  reaches every final store, so no groups are found and the key is plain dependency depth
  inside each round.
- **Last groups abreast** (the final 4 to 8 groups share one rank so they finish together):
  5 to 7 cycles at 16 rounds (1,314 to 1,307), 12 or more groups abreast is worse (1,348).
  Not kept. The tail is the remaining 3%.
- **A separate load-feeding priority**: not needed (section 3).
- **Forward-backward justification after the greedy**: not done. It would have to move
  split-shift lanes and keep the live-word counter exact while shifting ops later; the
  tail it could recover is at most the 37 to 43 cycles left.
- **Priority-rule sampling** (several staggers and windows, keep the best): not built. The
  sweeps in section 5 show what it would find: about 5 cycles at 16 rounds, 14 at 4 rounds.
- **The 2-lane minimum for starting a split** was not tuned.

## 9. Recommendation

1. For a 16-round greedy kernel use the command in section 1. It is a clear win with all
   rails green: 1,994 to 1,315 simulator cycles, below the 1,487 target and past every
   upstream speed threshold, from the greedy alone (about a second to build).
2. `--pressure-inflation-threshold 0.95` is safe to make the default on its own: it is
   better at every size measured, on the default shape and with the wavefront, and it keeps
   46 words of budget. This PR stack does not change it.
3. The wavefront should stay an option for now. It is the right start where there are
   gather rounds (5 rounds and up; at 4 rounds it still wins, 30 cycles in section 6,
   from running select rounds of some groups beside the hash of others), but at 3 rounds
   CP-SAT does better from the default
   shape with the retuned threshold (section 6), and CP-SAT cannot yet take the split-shift
   schedule.
4. The per-round barrier and the per-section repair it exists for are not needed by the
   wavefront greedy: it has never exceeded the budget at any size measured here.

## 10. Reproduction

```bash
# One row (any rounds, any options); the JSON feeds the table
uv run generate_ssa_kernel.py --rounds 16 --use-greedy -q --validate \
  --greedy-barriers none --greedy-priority wavefront --greedy-split-shift \
  --pressure-inflation-threshold 0.95 --output-json best16.json
uv run generate_ssa_kernel.py --rounds 16 --use-greedy -q --validate --output-json base16.json

# Table rows and a before/after diff
uv run python -m scripts.schedule_report --table default=base16.json best=best16.json
uv run python -m scripts.schedule_report base16.json --compare best16.json

# Upstream tests on the recommended kernel
./scripts/check_upstream.sh --strict --use-greedy --greedy-barriers none --greedy-priority wavefront \
  --greedy-split-shift --pressure-inflation-threshold 0.95

# The single-option rows
uv run generate_ssa_kernel.py --rounds 16 --use-greedy -q --validate --greedy-barriers none
uv run generate_ssa_kernel.py --rounds 16 --use-greedy -q --validate --greedy-barriers none --greedy-priority wavefront
uv run generate_ssa_kernel.py --rounds 16 --use-greedy -q --validate --greedy-split-shift
uv run generate_ssa_kernel.py --rounds 16 --use-greedy -q --validate --pressure-inflation-threshold 0.95

# CP-SAT from a greedy start (section 6)
uv run generate_ssa_kernel.py --rounds 3 --cpsat-deterministic-time 20 --validate -q \
  --greedy-barriers none --greedy-priority wavefront --pressure-inflation-threshold 0.95
```

## 11. Forward/backward justification (`--greedy-justify N`)

A pass after the greedy (`GreedyJustify`, `scheduler_rs/src/kernel_builder/ssa_greedy/justify.rs`)
that shortens the finished schedule with two list-scheduling passes per iteration:

1. **Backward**: schedule the reversed dependency graph, trying the ops that complete latest
   in the current schedule first, and mirror the times. Every op is now as late as possible.
2. **Forward**: schedule the original graph in the order of that schedule's start times.

Before the first iteration one forward pass runs in the order of the greedy's own schedule
(the same order without the pressure heuristics). The shorter schedule of each iteration is
where the next one starts, even when it is no shorter than the last (lengths are not
monotone); the shortest schedule seen that passes every check is the result. The search
ends after N iterations, when a schedule reproduces itself, or after 4 iterations in a row
in which no shorter schedule was accepted (`DEFAULT_MAX_STALLED_ITERATIONS`; without it an
orbit that never returns to exactly the same schedule runs all N iterations). Only cycles
and the shift decision (vector slot, 8 scalar lanes, or lanes split over two consecutive
cycles) change. The greedy's const and select encodings are kept.

The list scheduler is not trusted. Every candidate shorter than the best so far is written
in the pipeline's schedule representation (`op.cycle`, shifted set, `shifted_alu_cycles`)
and checked there:

- `verify_schedule`: slot limits in every cycle, every dependency with its precedence
  offset (two cycles after a select on the multiply-add path), every memory edge with its
  own latency, shape of the shift records. The greedy's own schedule must pass the same
  check first, otherwise the pass does nothing.
- Scratch: peak live words by `interval_liveness_curve` (the accounting the greedy's
  counter is checked against) within the greedy's budget, the pool condition under
  `--scratch-model pools`, and a trial run of address assignment within the scratch size.

- Domains, only when CP-SAT follows: every op inside its `[ASAP, ALAP]` domain, since the
  schedule becomes the solver's hint.

A candidate that fails is dropped. If none passes, the greedy's schedule is kept unchanged.
The summary line (`Greedy justify: ...`) reports cycles before and after, iterations,
shifted and split ops, peak live words, candidates checked and how many were rejected for
scratch, by verification (which would be a bug in the list scheduler) and for ASAP/ALAP
domains (which is not: the list scheduler does not know the domains), each with its own
counter. `SCHEDULER_DEBUG=1` prints the length of every pass.

### Measured

Shaped greedy (`--greedy-barriers none --greedy-priority wavefront --greedy-split-shift
--pressure-inflation-threshold 0.95`), `forest_height=10`, with and without
`--greedy-justify 10`. Every row was built with `--validate`: PROVED and PASSED.

| Rounds x batch | Capacity floor | Shaped greedy (simulator) | + justify 10 (simulator) | Gap to floor | Scratch peak | Scratch rejections |
|---|---|---|---|---|---|---|
| 4 x 64 | 93 | 135 (137) | **105** (107) | 12 | 544 -> 457 | 0 of 3 |
| 6 x 64 | 130 | 180 (182) | **158** (160) | 28 | 552 -> 542 | 0 of 1 |
| 4 x 256 | 357 | 423 (425) | **364** (366) | 7 | 827 -> 1,137 | 0 of 6 |
| 6 x 256 | 503 | 556 (558) | **511** (513) | 8 | 827 -> 1,078 | 0 of 4 |
| 16 x 256 | 1,271 | 1,313 (1,315) | **1,279** (1,281) | 8 | 1,243 -> 1,321 | 0 of 1 |
| 16 x 256, stagger 10, 20 groups in flight | 1,271 | 1,308 (1,310) | **1,278** (1,280) | 7 | 1,363 -> 1,394 | 0 of 2 |

The 16 x 256 kernel (1,281 simulator cycles) passes the strict upstream check (commands
below), and two builds give the same kernel file. The pass adds about 0.1 s to a 16-round
build. In every row the shortest schedule is found before the stall limit ends the search
(16 x 256: in the first iteration; the search stops after iteration 5).

### With CP-SAT following

The pass also runs when CP-SAT follows the greedy; the justified schedule is then the hint
(and, through the greedy ALAP cap and the horizon, the neighbourhood) the solver starts
from. Measured with a deterministic-time budget, `--validate` (PROVED and PASSED), no
per-section repair needed, and CP-SAT reporting the hint complete and feasible in every
build:

| Build | Greedy hint | + justify 10 hint | CP-SAT result | CP-SAT result with justify 10 |
|---|---|---|---|---|
| 2 x 16, `--greedy-barriers none --greedy-priority wavefront`, `--cpsat-deterministic-time 5` | 42 | 37 | 36 (optimal) | 36 (optimal) |
| 4 x 64, shaped greedy with `--cpsat-split-shift-hint`, `--cpsat-deterministic-time 10` | 135 | 105 | 135 | 105 |

`tests/test_scheduler_rs_interface.py::test_greedy_justify_with_cpsat` builds the first row
both ways and checks the replay proof, the reference output, and that the result is no
longer than the hint or than the build without justification. Nothing larger has been run
with CP-SAT.

### Limits

- **It is not pressure-aware.** The passes ignore scratch and a candidate is only accepted
  or rejected afterwards. From the wavefront schedule this is enough (table above). From a
  schedule that runs all 32 groups abreast every candidate is rejected and nothing changes:
  16 rounds with the default shape, 9 of 9 candidates rejected before the stall limit ends
  the search (the last at a peak of 1,603 words against 1,535), result 2,007 cycles as
  without the pass; 4 rounds with `--greedy-barriers none` alone, 9 of 9 rejected (peak
  1,649). With `--greedy-budget-headroom 250` on the shaped 16-round build (budget 1,285)
  8 of 8 candidates are rejected and the result stays at 1,313 cycles; before the stall
  limit that build ran all 10 iterations, rejected 18 of 19 candidates and accepted one of
  1,280 cycles, so under scratch pressure the limit can end the search before a late
  candidate that would fit. Making the list scheduler respect the budget is the follow-up.
- **Memory edges with latency 0** (a store after a load of the same word) are scheduled with
  latency 1. The verifier checks the real latency, so this can only cost cycles.
- **CP-SAT pipeline: measured at two small sizes only** (table above). An accepted schedule
  must keep every op inside its ASAP/ALAP domain. Justification can raise peak scratch
  (4 x 256: 827 to 1,137 words), so at larger sizes a justified hint could send a section
  to the per-section repair where the greedy's would not; that has not been observed or
  measured.
- Shift decisions change only when the greedy owned them (no forced shift encoding, no hint
  pass); ALU-derived constants and eliminated vbroadcasts make the pass skip.

```bash
# 16 x 256: 1,279 cycles (1,281 on the simulator), PROVED and PASSED
uv run generate_ssa_kernel.py --rounds 16 --use-greedy --validate \
  --greedy-barriers none --greedy-priority wavefront --greedy-split-shift \
  --pressure-inflation-threshold 0.95 --greedy-justify 10 --save-kernel kernel_justify.json
# The same kernel on the unmodified upstream simulator and tests
./scripts/check_upstream.sh --strict --kernel kernel_justify.json
# CP-SAT following the justified greedy (first row of the CP-SAT table)
uv run generate_ssa_kernel.py --rounds 2 --batch-size 16 --validate \
  --greedy-barriers none --greedy-priority wavefront --greedy-justify 10 \
  --cpsat-deterministic-time 5 --cpsat-time-limit 600
```
