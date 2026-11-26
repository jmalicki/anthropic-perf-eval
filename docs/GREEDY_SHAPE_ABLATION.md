# Greedy schedule shape: ablation

November 2026. Every number here was produced in this session with the commands in
section 10, greedy scheduler unless a row says CP-SAT, `forest_height=10`, `batch_size=256`.
Every build was run with `--validate`; a row with numbers was PROVED (symbolic dataflow
replay) and PASSED (numeric validation), and a row that says "fails" did not produce a
kernel. Tables come from `uv run python -m scripts.schedule_report --table`.

## 0. Latest: what limits useful shifting (section 12)

Every vector op the greedy does not shift takes one vector slot, so `V - s = 6T - I` exactly
(V vector ops, s shifted, T cycles, I idle vector slots). The identity ties the shift count to
the cycle count only for a fixed number of idle slots: at the same I, a schedule that is longer
for any reason shifts fewer ops, but an idle cycle adds 1 to T and 6 to I and leaves s
unchanged. The greedy's vector ALU is busy in every steady-state cycle, so here I is small and
nearly constant. Forcing more shifts (`pace`, `reserve`, PR #73) made these schedules slower,
which rules out too few shifts as the cause for those policies only; a shift decision can still
delay a schedule under policies not tested. For `flow90.json` (1,039): 32 of its 42 cycles above
the floor are needed by any schedule of that emission (static head/tail bound 1,029, a
bound (row arithmetic argued by hand, checked by Z3 on random small DAGs)); 10 are the gap above the static bound (an upper bound on scheduling loss: the bound is a lower bound, not the optimum). In shifts: the floor's split shifts 1,465 ops and the greedy
1,307, 158 fewer. A schedule at 1,029 cycles **with the same 115 idle vector slots** could shift
at most 1,367; one with more idle slots could shift more, so "98 of the 158 are out of reach" is
conditional on I, not a bound. Each number below is labelled **removal** (a build with the
cause removed or changed, A/B) or **accounting** (a decomposition of one schedule, which says
where cycles went, not what removing a cause would buy):

- **The one flow slot.** The greedy runs every SelectChoice as a flow `vselect` and never takes
  the `multiply_add` form. With path bits and `bits-flow` in round 4 (probe A, 1,101), any
  schedule that runs every select on flow needs at least 1,050 cycles (bound on the run's
  encodings). New flag `--greedy-select-overflow valu` (removal): 1,054 as first measured, but
  that build relied on a live-word counter that undercounted multiply-add selects; with the
  counter fixed it exceeds the scratch budget (peak 1,761), so the 47 cycles are not available
  within the budget with this flag.
- **Round-1 `bits-flow` with index bits.** Established by removal: the A/B pair that differs
  only in round 1 (`bits` 1,039, `bits-flow` 1,053) costs 14 cycles, and relaxing scratch
  leaves 1,053 (removal: not scratch). The mechanism is a **hypothesis**: the accounting shows
  a longer flow queue in mid-schedule and late groups entering the load stream late, but both
  interventions aimed at that queue made the plan slower (`--greedy-flow-priority
  feeds-load-first` 1,053 -> 1,074; `--greedy-select-overflow valu` 1,053 -> 1,067, the latter
  with the undercounting counter).

Neither new flag is a default win. A calibrated heuristic of the greedy's cycles
(`es.GREEDY_MODEL`, reproducible with `uv run python -m scripts.greedy_model_fit --check`)
ranks index-bit plans **in-sample** (Spearman +0.95 over 28 plans; +0.96 with the constants
refitted leave-one-out, but the form was chosen on the same plans; +0.71 on the 13 within 30
cycles of the best, a subset selected on the outcome). The only out-of-sample test is negative:
the four plans stage 1 chose with it ran 20 to 42 cycles slower than predicted (1,051 and 1,058
predicted 1,030; path 1,053 and 1,054 predicted 1,011 and 1,034). It found no plan better than
1,039 / 1,033.

## 0.1 Previous: the load-bound plans (section 11)

On `flow90.json`, 16 rounds, the shaped greedy goes from 1,104 / 1,106 to **1,039 / 1,041**
cycles. Two changes interact to get there:
- Retuning the existing window and stagger, with no load-priority change, gives 28 cycles:
  `--greedy-groups-in-flight 32` alone reaches 1,088, and adding `--greedy-stagger-cycles 9`
  reaches 1,076.
- `--greedy-load-priority final-last` (new) adds 37 cycles on top of that shape (1,076 to
  1,039). It adds 9 at stagger 12 (1,088 to 1,079), and alone with the default window of 16
  it is 83 cycles worse (1,187).

Stagger 9 was picked by a sweep on this one emission. The same flags take
`path_flow90.json` from 1,097 / 1,099 to **1,033 / 1,035**, and the four other PR #69
plans from 1,104 to 1,039. All of these are PROVED and PASSED; the two best kernels also pass
upstream `--strict`.

1,039 is 10 cycles above this emission's static head/tail bound of 1,029 (load row).
Shifting toward the floor's VALU/ALU split (`--greedy-shift-policy pace`) reaches that split
and is slower: a negative result.

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
| `--greedy-justify N` | 0 (off) | Forward/backward justification of the finished greedy schedule, at most N iterations (section 13). |
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

## 11. The load-bound plans: why the shaped greedy stopped at 1,104

**Conclusion.** On the structure-model plans (`docs/emission_plans/flow90.json`, and the
other PR #69 plans that all landed at 1,104), two changes take the shaped greedy from 1,104
to 1,039. They interact, so their effects do not add separately. Both are flag-gated and off
by default.

| `flow90.json`, shaped greedy plus | stagger 12 (default) | stagger 9 |
|---|---|---|
| 16 groups in flight (default) | 1,104 | 1,125 |
| 16 in flight, `final-last` | 1,187 | |
| 32 in flight | 1,088 | 1,076 |
| 32 in flight, `final-last` | 1,079 | **1,039** |

- **Window and stagger retuning, no load-priority change: 28 cycles** (1,104 to 1,076).
  - The default window of 16 groups was tuned on main's default emission, whose scratch peak
    is higher; these plans peak near 1,050 words.
  - Stagger 9 was chosen by a sweep on this one emission (section 11.2).
- **`--greedy-load-priority final-last`: 9 to 37 more, depending on the shape.**
  - What it does: loads that no later load depends on (round 15's gathers) are offered the
    load slots after every other ready load.
  - It adds 9 cycles at stagger 12 and 37 at stagger 9 (both with 32 in flight).
  - It needs the open window. With 16 in flight, a group whose round-15 gathers are deferred
    keeps its place in the window, and the run is 83 cycles slower (1,187).

| Flags on top of the shaped greedy, 16 rounds, batch 256 | Schedule / simulator cycles |
|---|---|
| `flow90.json` (today's best, 1,104 / 1,106) | **1,039 / 1,041** |
| `path_flow90.json` + `--select-bit-source path` (1,097 / 1,099) | **1,033 / 1,035** |
| the four other PR #69 plans at 1,104 | 1,039 / 1,041 each |

The same flags were used unchanged on the other plans; stagger and window were not re-swept
there.

The 1,039 kernel is:
- 10 cycles above this emission's static head/tail bound of 1,029 (load row, t = 51, q = 14;
  the same with any const/select encoding);
- 25 cycles above its capacity floor (1,014, load).

Every kernel quoted here is PROVED and PASSED. The 1,041 and 1,035 kernels also pass
`./scripts/check_upstream.sh --strict`. Two builds of the 1,039 kernel are byte-identical.

**The leading hypothesis is a negative result.** The hypothesis was: give the greedy a target
split and shift toward it all along.
- `--greedy-shift-policy pace` reaches the floor's split exactly: 1,465 ops shifted, VALU
  and ALU both at 89.2%. It is 10 cycles *slower* (1,114).
- `reserve` shifts when the free vector slots are wanted by multiply_adds later in the
  cycle. It gains 3 cycles alone and loses 10 on top of the new shape.

PR #76 explains why: the shift count follows the schedule's length rather than limiting it
(see 11.1).

### 11.1 What the measurements show

Measured on the 1,104-cycle schedule with `scripts.schedule_report` (static head/tail bound
included) and `scripts/greedy_gap_diagnosis.py` (new).

**The diagnosis script's waiting counts are upper bounds.** They use operand readiness only:
SSA `dep_ids` with unit latency. They ignore the in-flight window, bounds and inflation
deferral, the scratch budget, memory edges and the extra cycle of a multiply-add select. They
locate idle capacity and queues; they do not prove that another order would have filled a
slot.

1. **Shift split.**
   - The floor-optimal resolution (`choice_floor`) runs 1,465 shiftable vector ops as 8 scalar
     ops; the greedy ran 1,039.
   - With the greedy's split the vector ALU needs 6,387 / 6 = 1,065 cycles.
   - PR #76 shows the identity behind this. Every unshifted vector op takes one vector slot, so
     V - s = 6T - I exactly (V vector ops, s shifted, T cycles, I idle vector slots), and the
     vector ALU is busy in every steady-state cycle. A longer schedule therefore shifts fewer
     ops, whatever made it longer. The missing shifts are a consequence of the length, not
     shown to be its cause.
2. **Declined shifts.** The pressure policy never suppressed a shift: the peak, 1,049 words,
   stays below the adaptive policy's 85% soft threshold. The greedy shifts only an op that
   finds the six vector slots full, and tries ready ops in wavefront order.
   - In cycles 0 to 300, on average 17 to 21 operand-ready `multiply_add`s wait, against
     under 0.5 shiftable ops (upper bounds), while the scalar ALU has 344 to 767 idle slots
     per 100 cycles.
   - That pattern is what `reserve` targets. Removing it gains 3 cycles alone (1,101) and costs
     10 on top of the new shape (1,049).
3. **Engines over time.**
   - In the 1,104 schedule the load slots are fully used from cycle 100 to 700 and lose only
     8 slots between 700 and 1,000. The vector ALU is also 100% busy there.
   - Of the 146 load slots idle after cycle 51, 116 are in the last 104 cycles. Any
     schedule's tail looks like that, so the count alone does not identify a cause.
   - In none of the 146 was a data load operand-ready. The figure of 22 "with a ready load
     waiting" in the first version of this section counted constants (ConstChoice ops, which
     the greedy may lower to flow `add_imm`); it is withdrawn.
   - The idle count barely moves under any shift policy (146 / 134 / 149 for `when-full` /
     `reserve` / `pace`), and cycles do not improve.
   - The static head/tail bound of 1,029 comes from the load row.
4. **What `final-last` changes, as measured.**
   - In the 1,039 schedule the last data load issues at cycle 1,024 (it was at 1,089), and
     only 35 load slots are idle after cycle 51 (it was 146).
   - Under the wavefront key, each group's round-15 gathers run as soon as they are ready, in
     key order. Per group, the last round-10 gather and the first round-15 gather are about
     56 cycles apart, since rounds 11 to 14 have no loads.
   - That `final-last` helps is consistent with this ordering costing load slots late in the
     schedule, but this was not isolated beyond the ablation in 11.2.
5. **The plateau.**
   - The PR #69 plans differ only in rounds 0 to 4 and 11 to 14, and all landed at 1,104
     before and at 1,039 after.
   - After the change, the vector ALU and the loads are both 100% busy from about cycle 55 to
     990. The vector ALU's work is 6,119 / 6 = 1,020 cycles; with 7 cycles of start-up idle
     and 12 of tail idle that sums to 1,039.
   - Which engine sets the 1,039 plateau is **open**.

### 11.2 Ablation (flow90.json)

All rows: `--emission-plan docs/emission_plans/flow90.json --merge-hash-stages --index-form
mirrored --greedy-barriers none --greedy-priority wavefront --greedy-split-shift
--pressure-inflation-threshold 0.95`, 16 rounds, plus the flags in the first column. Floor
is `schedule_report`'s (the greedy's own const/select resolution; the choice floor is 997).
Every row PROVED and PASSED.

| Run | Rounds | Cycles | Simulator | Floor | Gap | Peak words | valu % | alu % | load % | flow % | Round overlap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| shaped (baseline) | 16 | 1104 | 1106 | 1014 | 90 (8.9%) | 1049 | 96.4 | 64.3 | 91.8 | 80.2 | 13635 |
| shift reserve | 16 | 1101 | 1103 | 1014 | 87 (8.6%) | 1023 | 94.8 | 72.1 | 92.1 | 80.4 | 13552 |
| shift pace (auto 1465) | 16 | 1114 | 1116 | 1014 | 100 (9.9%) | 1065 | 89.2 | 89.2 | 91.0 | 79.4 | 13777 |
| shift reserve-pace | 16 | 1115 | 1117 | 1014 | 101 (10.0%) | 1081 | 89.1 | 89.1 | 90.9 | 79.5 | 13787 |
| loads final-last | 16 | 1187 | 1189 | 1012 | 175 (17.3%) | 1139 | 94.7 | 39.7 | 85.3 | 74.9 | 14917 |
| loads tail, stagger 9, 32 in flight | 16 | 1204 | 1206 | 1007 | 197 (19.6%) | 1441 | 90.5 | 50.6 | 83.6 | 74.7 | 8470 |
| 32 in flight | 16 | 1088 | 1090 | 1013 | 75 (7.4%) | 1121 | 96.4 | 71.0 | 93.1 | 81.6 | 13370 |
| stagger 9, 32 in flight | 16 | 1076 | 1078 | 1013 | 63 (6.2%) | 1207 | 96.7 | 74.8 | 94.1 | 82.4 | 12584 |
| final-last, 32 in flight | 16 | 1079 | 1081 | 1013 | 66 (6.5%) | 1343 | 96.5 | 74.3 | 93.8 | 82.3 | 13255 |
| **final-last, stagger 9, 32 in flight** | 16 | **1039** | **1041** | 1014 | 25 (2.5%) | 1431 | 98.2 | 85.5 | 97.5 | 85.3 | 12017 |
| + shift reserve | 16 | 1049 | 1051 | 1014 | 35 (3.5%) | 1447 | 96.8 | 86.5 | 96.6 | 84.5 | 12147 |
| + shift pace (auto) | 16 | 1078 | 1080 | 1013 | 65 (6.4%) | 1481 | 92.2 | 92.2 | 93.9 | 82.4 | 12285 |
| + shift pace 1300 | 16 | 1044 | 1046 | 1013 | 31 (3.1%) | 1423 | 97.7 | 85.2 | 97.0 | 85.0 | 11915 |

Stagger with `final-last` and 32 in flight (schedule cycles, peak): 7: 1,051 (1,465); 8:
1,048 (1,481); 9: 1,039 (1,431); 10: 1,058 (1,353); 11: 1,067 (1,361); 9 with 28 in flight:
1,110 (1,315). Without `final-last` (stagger, in flight): 6/24 1,085, 6/32 1,077, 9/20 1,088,
9/24 1,084, 9/32 1,076, 12/20 1,093, 12/24 1,091, 12/32 1,088.

Other plans, same flags:

| Run | Rounds | Cycles | Simulator | Floor | Gap | Peak words | valu % | alu % | load % | flow % | Round overlap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| path bits, shaped (baseline) | 16 | 1097 | 1099 | 994 | 103 (10.4%) | 1050 | 95.6 | 63.7 | 90.6 | 78.8 | 13389 |
| path bits, stagger 9, 32 in flight | 16 | 1066 | 1068 | 995 | 71 (7.1%) | 1399 | 96.4 | 73.8 | 93.3 | 81.0 | 12445 |
| **path bits, final-last, stagger 9, 32 in flight** | 16 | **1033** | **1035** | 995 | 38 (3.8%) | 1465 | 98.1 | 81.3 | 96.3 | 83.5 | 11984 |
| caps-underfilled2, shaped | 16 | 1104 | 1106 | 1010 | 94 (9.3%) | 1043 | 96.4 | 64.6 | 91.5 | 81.5 | 13636 |
| caps-underfilled2, final-last s9 g32 | 16 | 1039 | 1041 | 1010 | 29 (2.9%) | 1447 | 98.2 | 85.7 | 97.2 | 86.7 | 12074 |
| caps-underfilled3, shaped | 16 | 1104 | 1106 | 1014 | 90 (8.9%) | 1049 | 96.4 | 64.3 | 91.8 | 80.2 | 13635 |
| caps-underfilled3, final-last s9 g32 | 16 | 1039 | 1041 | 1014 | 25 (2.5%) | 1431 | 98.2 | 85.5 | 97.5 | 85.3 | 12017 |
| nogood-capped0, shaped | 16 | 1104 | 1106 | 1010 | 94 (9.3%) | 1043 | 96.4 | 64.6 | 91.5 | 81.5 | 13636 |
| nogood-capped0, final-last s9 g32 | 16 | 1039 | 1041 | 1010 | 29 (2.9%) | 1447 | 98.2 | 85.7 | 97.2 | 86.7 | 12074 |
| nogood-capped5, shaped | 16 | 1104 | 1106 | 1010 | 94 (9.3%) | 1051 | 96.4 | 64.6 | 91.5 | 81.5 | 13636 |
| nogood-capped5, final-last s9 g32 | 16 | 1039 | 1041 | 1010 | 29 (2.9%) | 1447 | 98.2 | 85.7 | 97.2 | 86.7 | 12074 |

"path bits" is `docs/emission_plans/path_flow90.json` with `--select-bit-source path`; its
static head/tail bound is 1,009, so 1,033 is 24 above it. The PR #69 plans were written from
`loop_report.json` on that branch (`plan` strings) into plan files with the same context as
`flow90.json`.

On main's default emission (no plan; ALU-bound, 1,313 with the shaped greedy) `final-last`
gives 1,309 to 1,312 with 20 or 24 in flight and fails the scratch budget with 32 in flight
(peak 1,656) or stagger 9 and 24 (1,632): it is for load-bound emissions.

### 11.3 The new options

| Option | Default | What it does |
|---|---|---|
| `--greedy-load-priority {key,tail,final-last}` | `key` | Wavefront only. Which ready loads are offered the two load slots first; other engines keep the wavefront order. `tail`: longest dependency path to the end first. `final-last`: loads no later load depends on go after every other ready load. |
| `--greedy-shift-policy {when-full,reserve,pace,reserve-pace}` | `when-full` | When a shiftable vector op runs as 8 scalar ops. `reserve`: also when the free vector slots are wanted by non-shiftable vector ops later in the cycle's order. `pace`: also whenever the run is behind `shifted / vector ops placed = target / vector ops`. |
| `--greedy-shift-target N` | balanced split | Target for `pace`: default `(2V - A) / 10` of the counts the greedy sees (1,465 here, as `choice_floor`). |

Shift policies:
- Both proactive shift policies respect `allow_shift`. They use a split shift when a whole
  one does not fit, and fall back to the vector ALU when no scalar slot is free.
- They apply only when the greedy decides shifts itself, not with fixed shift encodings.
- Pace accounting (documented at its definition in `ssa_greedy/mod.rs`):
  - "vector ops" are ops whose own engine is `Valu`, the same set in the total and in the
    placed count;
  - placements are counted on every path: normal, split start and force-schedule;
  - the pace is a floor on shifting, not a cap: shifts made because the vector slots are
    full continue after the target is reached.
- `GreedyDiagnostics::proactive_shifts` counts shifts made while a vector slot was free.

`final-last`, exactly:
- Once per cycle, before the first placement pass, the `Load`-engine ops are re-sorted among
  the positions they hold in the wavefront order.
- Constants (`ConstChoice`) are sorted with them. A constant no load depends on counts as
  final, even if it is then lowered to a flow `add_imm`.
- Only the first pass is ordered. A feeder load deferred for the scratch budget is retried
  after the final loads of that pass, so a final load can take a slot ahead of it
  (`test_load_priority_final_last_budget_deferral`).
- The 1,039 numbers were measured with exactly this behaviour; changing it needs
  re-measurement.
- `--greedy-load-priority` without `--greedy-priority wavefront` is an error (CLI and binding).

Tests in `ssa_greedy/tests.rs`:
- slot limits on the normal path (`test_shift_policies_keep_slot_limits`);
- slot limits on runs that both force-schedule and make proactive shifts
  (`test_shift_policies_keep_slot_limits_when_force_scheduling`, `test_proactive_shift_counter`);
- `test_shift_reserve_*`, `test_shift_pace_*`;
- `test_load_priority_final_last` and `..._budget_deferral`.

### 11.4 Ruled out, and not verified

- **Ruled out:** shifting toward the floor's VALU/ALU split as a remedy (pace reaches the
  split and is slower; idle load slots do not change; PR #76: the shift count follows the
  schedule's length); pressure-policy shift
  suppression (never active below 85% of the budget); `tail` load priority (sorting loads by
  their tail keeps every group's gather of one round abreast and loses the stagger:
  1,204); `final-last` with the 16-group window (1,187).
- **Not verified:** that the load ordering in the tail is *the* cause of the 1,104 plateau
  (the ablation shows `final-last` helps only together with the open window, and the
  idle-slot counts cannot isolate a cause); how many of the last 10 cycles to the 1,029
  bound a scheduler can recover; which of the loads and the vector ALU sets the 1,039
  plateau (both are saturated from about cycle 55 to 990; open); the start-up (16 load slots idle before cycle 51 and 7 after,
  while 13 loads of dependency depth below 51 run after cycle 51) was not addressed; no
  CP-SAT run from the new start.

### 11.5 Reproduction

```bash
uv run generate_ssa_kernel.py --rounds 16 --use-greedy -q --validate \
  --emission-plan docs/emission_plans/flow90.json --merge-hash-stages --index-form mirrored \
  --greedy-barriers none --greedy-priority wavefront --greedy-split-shift \
  --pressure-inflation-threshold 0.95 \
  --greedy-load-priority final-last --greedy-stagger-cycles 9 --greedy-groups-in-flight 32 \
  --output-json best.json --save-kernel best_kernel.json
uv run python -m scripts.greedy_gap_diagnosis best.json --from 51 --slice 100
./scripts/check_upstream.sh --strict --kernel best_kernel.json
```

## 12. What limits useful shifting

**Conclusion.** The number of vector ops the greedy shifts to the scalar ALU does not limit the
schedule. It follows from the schedule's length. Every vector op left unshifted takes one
vector slot, so for vector ops V, shifted s, cycles T and idle vector slots I:

```
V - s = 6 T - I        (exact; checked on every schedule below)
```

The vector ALU is busy in every steady-state cycle (I is 115 to 250 slots, all in the start-up
and the tail). So a schedule that is longer for any reason shifts fewer ops. PR #74 found
cycles = VALU ops left / 6 + 19 to 32 (Spearman +0.83 to +0.99). That is this identity, not a
cause. PR #73 also found that forcing more shifts (`pace`, `reserve`) makes schedules slower.
The causes are elsewhere. "Needed by any schedule" is the static head/tail bound, a bound (row arithmetic argued by hand, checked by Z3 on random small DAGs):
SB1 in verification/static_bounds_proof.py (150 random DAGs of 3 to 7 ops at VLEN 2) and
makespan_rows.py, not a proof for all graphs. In the last column each
number is labelled: **removal** = a build with the cause removed or changed (an A/B), the only
kind that says what removing it buys; **accounting** = a decomposition of one schedule.

| Plan (cycles) | Cycles above the floor | Needed by any schedule | The rest: what is known |
|---|---|---|---|
| `flow90.json`, index (1,039) | 42 (floor 997) | 32 (static head/tail bound 1,029, load row) | 10 cycles above the static bound (cycles minus bound: an upper bound on scheduling loss, the optimum is not known); cause not identified |
| `r1 bits-flow` variant of a 1,039 plan, index (1,053) | 60 (floor 993) | 32 (bound 1,025) | Removal: `bits` instead of `bits-flow` in round 1 gives 1,039 (14 cycles); scratch-relaxed leaves 1,053 (not scratch). Hypothesis, from accounting: the longer flow queue in mid-schedule delays late groups into the load stream. Both interventions aimed at it are slower (feeds-load-first 1,074, overflow 1,067), so it is not established |
| `path_flow90.json`, path (1,033) | 53 (floor 980) | 29 (bound 1,009) | 24 cycles above the static bound (an upper bound on scheduling loss); removal: scratch-relaxed 1,034 (not scratch). #82 accounts for them (start-up load idle slots) |
| probe A: `path_flow90.json` with `bits-flow` in round 4, path (1,101) | 124 (floor 977) | 32 (load bound 1,009); 73 for any schedule that runs every select on flow (flow row 1,050) | Removal: `--greedy-select-overflow valu` gave 1,054 with the undercounting live-word counter; with the counter fixed it exceeds the budget (peak 1,761). Removal: scratch-relaxed 1,093 |
| `flow90.json`, path (1,105) | 108 (floor 997) | 32 (bound 1,029) | Removal (analysis only): scratch-relaxed gives 1,046, so the 1,536-word budget, or how the greedy meets it, costs up to 59 |

In shift counts, for `flow90.json`: the floor's split is 1,465 and the greedy's is 1,307,
158 "missing" shifts (`scripts.shift_limit_diagnosis` prints a different count, 6 x 42 - I =
137: the shifts a schedule at the floor with **no** idle vector slot would take beyond this one's.
Closing the 42 cycles while keeping this schedule's 115 idle slots would need 6 x 42 = 252). The
identity gives s = V - 6T + I, so the shifts a schedule can take depend on its idle vector
slots I as well as on T. A schedule at the static bound of 1,029 cycles **with the same 115
idle slots** could shift at most 1,367, 98 short of the floor's split; a schedule at 1,029 with
more idle slots could shift more. So the 98 is conditional on I and is not a bound. What is
unconditional is the cycle statement: 32 of the 42 cycles are needed by any schedule. "Needed by any schedule" is
`scripts.static_bounds` (minimum work over every encoding; a randomized Z3 check against a
reference schedule model, `verification/static_bounds_proof.py`, not a proof for all graphs). The probe-A flow row uses the run's encodings, so it
holds for every schedule that runs every SelectChoice as a vselect, as the default greedy
does.

All builds: 16 rounds, batch 256, `--merge-hash-stages --index-form mirrored`, the shaped
greedy flags plus the load-tail flags (`loop run --load-tail`). They are greedy development
measurements. Every kernel quoted is PROVED (dataflow replay) and PASSED (`--validate`).
"Scratch-relaxed" runs (`--scratch-budget-override 4096`) are analysis only, not kernels. Of
the 50-build budget, 31 sixteen-round builds were used, one at a time.

### 12.1 Candidates, ruled in or out

| Candidate | Test | Result |
|---|---|---|
| Op kinds that cannot shift (multiply_add, vbroadcast; selects) | per-round counts from the schedule (accounting) | `bits` adds one fixed multiply_add per group and select; `bits-flow` moves it to a flow `vselect`. Never-shiftable work does not set the count: shiftable ops ready and waiting while the scalar ALU has room ("declined") is 0 to 16 shift equivalents per schedule (0 on `flow90.json` and the 1,039 plan, 16 on the 1,053 plan). An earlier "18 to 41" counted split-shifted producers ready one cycle early (readiness used the first lane; review of #76) |
| Shifted op = 8 scalar slots, lane rules, split shift | idle scalar slots by class (accounting) | Most idle scalar slots are in cycles where `multiply_add`s wait and shiftable ops hold vector slots (priority inversion: 149 shift equivalents on `flow90.json`, 133 before the readiness fix). Removing the inversion (`reserve`, PR #73; removal) is slower: the loads are full in those cycles |
| Timing mismatch (H2) | round bands, below | Idle scalar slots sit in the round-15 tail and the gather band. Ready shiftable work sits in the select bands. They rarely coincide |
| Scratch pressure policy: soft (85%) and critical (95%) thresholds | `--pressure-soft-threshold 0.999 --pressure-critical-threshold 0.999` (removal), inflation threshold left at 0.95 | **These two thresholds: no gain** (probe A 1,101 -> 1,106; `r1 bits-flow` plan 1,053 -> 1,055; path `flow90.json` 1,105 -> 1,118). This did **not** test the policy's ASAP inflation (threshold 0.95, the one that matters), so the pressure policy as a whole is not ruled out. PR #82 (`docs/SCRATCH_CEILING_DIAGNOSIS.md`) found that inflation costs 49 to 78 cycles on the plans at the scratch ceiling, because it defers operand-ready gather loads; it is needed today because the `words` budget check force-schedules budget-deferred ops instead of deferring them (`--greedy-budget-force stall`) |
| The 1,536-word budget itself | `--scratch-budget-override 4096` (analysis) | **Ruled in for path `flow90.json`** (1,105 -> 1,046). Not for probe A (1,093), `path_flow90.json` (1,034), or the index plans (1,039 and 1,053 unchanged) |
| Flow saturation, all selects on flow (H1) | select resolution per round; `--greedy-select-overflow valu` | **A cause for flow-heavy path plans by bound** (with every select on flow, probe A needs 1,050 cycles); **the flag's measured gain did not survive the counter fix (below)**. The greedy put 1,004 of 1,004 selects on flow. Flow was busy every cycle from 0 to 900, with 90 to 140 flow ops waiting. Overflow puts 348 on the vector ALU: 1,101 -> 1,054, and the shift count rises from 809 to 1,369. **But it is not a default win:** `path_flow90.json` 1,033 -> 1,059, `flow90.json` 1,039 -> 1,066, `r1 bits-flow` plan 1,053 -> 1,067, path floor-optimal plan over the scratch budget. **Counter fix (review of #76):** these overflow numbers were measured with a live-word counter that did not hold a multiply_add select's delta and operands to cycle + 1 as address assignment does; its check was blind to it too. With the base binary the counter read 1,408 to 1,464 words where address assignment needed 1,481 to 1,577. The path floor-optimal plan (`floor_optimal.json`, path bits) failed address assignment at 1,577 words because of that undercount. Rebuilt with the fixed counter (same command, section 12.5, plus `--select-bit-source path --greedy-select-overflow valu`): `path_flow90.json` 1,059 -> 1,061; probe A 1,054 -> over the budget (the greedy force-schedules, peak 1,761); path floor-optimal still over the budget (peak 1,771, now seen by the greedy). In overflow mode the fixed counter over-counts while scheduling (it holds the delta of every select that may still take the multiply-add form until it is resolved), so "over budget, peak 1,761" means this greedy did not fit it, not that overflow cannot fit. Default flags are byte-identical (no overflow, no change). The index-plan overflow numbers were not rebuilt |
| Flow order: early-round selects first | `--greedy-flow-priority feeds-load-first` | **Negative:** `r1 bits-flow` plan 1,053 -> 1,074; 1,039 plan -> 1,052 |
| `--select-choice fma` as a resolution hook | probe A | Ignored by the greedy-only pipeline (same 1,101 kernel). There was no per-op hook before `--greedy-select-overflow` |

**H1 (is `bits` vs `bits-flow` only a select-resolution choice?): partly, and not within the
budget.** On probe A the resolution was worth 47 of the 68 cycles (1,101 -> 1,054 against
1,033 for `bits`), but that build relied on the undercounting live-word counter; with the
counter fixed it exceeds the scratch budget (peak 1,761). The limit rows below were measured
with the old counter too and were not rebuilt. No resolution rule tried recovers the rest or
wins elsewhere. A count limit set to the floor's
resolution (81 selects) gives 1,083, and 160 gives 1,072. On the gm-path plan below, `bits`
and `bits-flow` in round 4 run 1,054 and 1,053. So with this greedy the choice still changes
cycles through timing, not only through the op set. It should stay a stage-1 choice until a
scheduler resolves selects well; then it can become a scheduler variable. Select resolution
by round in the winning and losing builds (default greedy):

| Build | Round 1 | 2 | 3 | 4 | 12 | 13 | 14 | Mean flow wait (rounds 1-4 / 12-14) |
|---|---|---|---|---|---|---|---|---|
| `path_flow90.json` (1,033) | flow 32 | flow 96 | flow 224 | flow 140 (bits x20 use multiply_adds) | flow 32 | flow 96 | flow 224 | 7 to 24 / 1 to 5 |
| probe A (1,101) | flow 32 | flow 96 | flow 224 | flow 300 | flow 32 | flow 96 | flow 224 | 97 to 147 / 48 to 66 |
| probe A + overflow (1,054, undercounting counter; over budget with the fix) | 30 + fma 2 | 81 + 15 | 137 + 87 | 186 + 114 | 26 + 6 | 64 + 32 | 132 + 92 | 0.5 to 6 / 0.6 to 2 |
| index 1,039 plan | bits | flow 96 | flow 224 | flow 240 | bits | flow 96 | flow 224 | 10 to 14 / 15 to 17 |
| same, `bits-flow` in round 1 (1,053) | flow 32 | flow 96 | flow 224 | flow 240 | bits | flow 96 | flow 224 | 9 to 29 / 20 to 24 |

**H2 (by phase).** Bands are the round of the vector work in each cycle. The table gives
shiftable ops, shifts taken, op-cycles that shiftable ops spent ready and waiting, and idle
scalar slots:

| Band | 1,039 plan (index, `nogood-capped0`): shiftable / shifted / ready-wait / scalar idle | 1,053 plan (`nogood-capped4`, `r1 bits-flow`): same |
|---|---|---|
| setup | 110 / 34 / 44 / 106 | 111 / 37 / 93 / 92 |
| rounds 0-4 | 1,440 / 593 / 611 / 446 | 1,440 / 505 / 310 / 756 |
| rounds 5-10 | 1,888 / 318 / 590 / 488 | 1,888 / 287 / 290 / 1,064 |
| rounds 11-14 | 1,088 / 352 / 683 / 152 | 1,088 / 366 / 462 / 216 |
| round 15 | 320 / 12 / 41 / 596 | 320 / 15 / 24 / 620 |

Regenerated after the readiness fix (both plans rebuilt, same cycles; ready-wait was 721 / 629
/ 732 and 412 / 323 / 520 before). This is accounting. In the 1,053 plan, idle scalar slots
grow in the gather band (488 -> 1,064). Ready shiftable work there shrinks (590 -> 290),
consistent with the load stream starving, so nothing is ready to shift.
Per group, the last group ends its round-4 work 13 cycles later (821 -> 834). Its round-10
gathers end 12 cycles later. Load slots idle 26 times in cycles 700 to 800, and the schedule
ends 14 cycles later.

### 12.2 The new options (off by default; default kernels byte-identical)

| Option | Default | What it does |
|---|---|---|
| `--greedy-select-overflow {never,valu}` | `never` | `valu`: a SelectChoice with a precomputed delta (one `multiply_add`) runs on the vector ALU when the flow slot is taken in that cycle and a vector slot is free. The greedy reports these selects as multiply-add selects for lowering and address assignment. It refuses to run if the caller does not collect them |
| `--greedy-select-overflow-limit N` | none | At most N selects take the multiply_add form |
| `--greedy-flow-priority {key,feeds-load-first}` | `key` | Wavefront only: ready flow ops that a non-final load depends on go first |

Tests (`ssa_greedy/tests.rs`): slot limits (vector 6, scalar 12, flow 1, with multiply-add
selects counted as vector ops) under every overflow setting, with and without split shift,
under both flow priorities, on the normal path and on the force-schedule path. Also covered:
a caller that does not collect the multiply-add selects is refused, and `feeds_nonfinal_load`
marks the right ops. `flow90.json` built with the new code and default flags is
byte-identical to the build before the change.

### 12.3 Stage 1: a greedy pipeline term (heuristic)

No valid bound ranks the plans. The bounds checked never exceed the cycles over the 44
recorded builds:

- the floor with selects pinned to flow;
- the flow row with selects on flow plus its 27-cycle tail;
- the static bound.

Near the optimum they rank the plans no better than the floor (index Spearman -0.54 to
+0.50). So the stage-1 term is a **calibrated heuristic** of the load-tail greedy, not a
bound. In `es.GREEDY_MODEL` / `solve_plan(greedy_model={})` / `solve --greedy-model`
(data: `docs/emission_plans/greedy_model_data.json`; fit, leave-one-out, rankings and
out-of-sample errors: `uv run python -m scripts.greedy_model_fit [--check]`, enforced by
`tests/test_greedy_model_fit.py`):

```
predicted = max(floor + 32,  (loads + consts) / 2 + 19 + 0.75 * max(0, flow ops + selects - 900))
```

The story behind the form (a hypothesis, not measured): the load engine paces the kernel; the
selects of all groups' first and last rounds share the flow slot, and above about 900 of them
(about 0.9 per load cycle) every extra select costs about 0.75 cycles. The constants are a grid
fit (minimum total absolute error) on the 24 index-bit loop builds under 1,200 cycles of
`loop_report_load_tail.json` (their rows and counts are in the data file). Leave-one-out
(constants refitted without each plan): MAE 6.2 cycles, maximum 18. It covers the constants,
not the form, which was chosen after looking at these plans; so every number in the table is
in-sample, and the "near" column is a subset selected on the outcome. `a_floor` is barely identified: the grid minimum (total absolute
error 126) is tied between 32 and 36, and the floor branch is active on 1 of the 24 training
plans, so 32 is the first of a tie, not a measured offset.

| Surrogate (Spearman) | Index, all (28) | Index, loop (22) | Index, near (13) | Path, all (16) |
|---|---|---|---|---|
| floor | +0.24 | -0.09 | -0.54 | -0.47 |
| capped floor `flow=0.92` | +0.79 | +0.71 | +0.19 | -0.12 |
| greedy model (in-sample) | +0.95 | +0.94 | +0.71 | +0.05 |
| greedy model (constants leave-one-out, form in-sample; 24 plans) | +0.96 | | +0.71 | |

- **(a) Ranking.** Index bits: the plans are ranked (+0.71 near the best, where the floor gave
  -0.54). Path bits: they are not (+0.05). 12 of the 16 path builds run near the scratch
  ceiling (1,489 to 1,505 words), and the model does not include scratch.
- **(b) The path round-4 mistake.** The model ranks probe A (predicted 1,101, actual 1,101)
  behind `path_flow90.json` (predicted 1,023, actual 1,033). Stage 1 with the term picks
  `r0 bits; r1 bits-flow; r2 bits; r3 bits-flow; r4 gather x9 + bits-flow x23; r11 bits; r12
  bits x21 + bits-flow x11; r13 bits; r14 bits-flow`: **1,053 / 1,055**. The best path plan
  any earlier stage-1 mode found ran 1,099. The same plan with `bits` in round 4 runs
  1,054 / 1,056, so its round-4 `bits-flow` is not a mistake there.
- **(c) A new best: no.** Index bits: the model's plan runs 1,051 / 1,053 (predicted 1,030).
  Fixing rounds 1 and 12 to `bits` gives 1,058 / 1,060 (predicted 1,030). Path bits: 1,053
  against 1,033 (predicted 1,011), and with `bits` in round 4 1,054 (predicted 1,034). These
  four builds are the only out-of-sample evidence, and it is negative: the model is optimistic
  by 20 to 42 cycles on every plan it chose (they are in the data file, use
  `out_of_sample`). Minimising a fitted model selects its errors. No plan is checked in.

### 12.4 Not verified

- Where the 1,039 schedules' 10 cycles above the static bound go (an upper bound on what scheduling could recover).
- A select-resolution rule that wins on both the index and the path plans. Overflow keyed
  to a target count or to the scalar ALU's slack was not tried beyond the limits above.
- A scratch term for path-bit plans. The 1,505-word ceiling is explained in
  docs/SCRATCH_CEILING_DIAGNOSIS.md: the pressure policy, needed because the `words` model
  forces budget-deferred ops in the same cycle (`--greedy-budget-force stall` removes the
  same-cycle force; not safe on every plan: 3 of 28 path-bit plans exceed the budget, see
  docs/STALL_DEADLOCK.md on #88).
- The model in other contexts (rounds, batch, values-only).
- CP-SAT on any of these emissions (no whole-program CP-SAT, by the machine rules).

### 12.5 Reproduction

```bash
# One plan with the load-tail greedy, its schedule JSON kept
uv run generate_ssa_kernel.py --rounds 16 --use-greedy -q --validate \
  --emission-plan docs/emission_plans/flow90.json --merge-hash-stages --index-form mirrored \
  --greedy-barriers none --greedy-priority wavefront --greedy-split-shift \
  --pressure-inflation-threshold 0.95 \
  --greedy-load-priority final-last --greedy-stagger-cycles 9 --greedy-groups-in-flight 32 \
  --output-json s.json --save-kernel k.json
uv run python -m scripts.shift_limit_diagnosis s.json --floor 997
# The flags of section 12.2 on top: --greedy-select-overflow valu [--greedy-select-overflow-limit N],
# --greedy-flow-priority feeds-load-first. Scratch-relaxed analysis: --scratch-budget-override 4096.
# Stage 1 with the greedy model
uv run python -m scripts.emission_structure solve --greedy-model [--select-bit-source path] --out plan.json
# (the loop builds plans with the load-tail greedy and stores builds and flags in the state file)
uv run python -m scripts.emission_structure loop run --load-tail --state loop_state.json \
  --modes nogood --budget 4 --max-builds 8
uv run python -m scripts.emission_structure loop report --state loop_state.json --near 30   # greedy_model row
```

## 13. Forward/backward justification (`--greedy-justify N`)

A pass after the greedy (`GreedyJustify`, `scheduler_rs/src/kernel_builder/ssa_greedy/justify.rs`)
that shortens the finished schedule with two list-scheduling passes per iteration:

1. **Backward**: schedule the reversed dependency graph, trying the ops that complete latest
   in the current schedule first, and mirror the times. Every op is now as late as possible.
2. **Forward**: schedule the original graph in the order of that schedule's start times.

Before the first iteration one forward pass runs in the order of the greedy's own schedule
(the same order without the pressure heuristics). The shorter schedule of each iteration is
where the next one starts, even when it is no shorter than the last (lengths are not
monotone); the shortest schedule seen that passes every check is the result. The search
ends after N iterations, when a schedule reproduces itself, or after 10 iterations in a row
in which no shorter schedule was accepted (`DEFAULT_MAX_STALLED_ITERATIONS`). The last only
bounds a large N, where an orbit that never returns to exactly the same schedule would
otherwise run all N iterations; up to N = 10 it changes nothing. Only cycles
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
build. At 16 x 256 the shortest schedule is found in the first iteration.

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
  16 rounds with the default shape, 21 of 21 candidates rejected (peak 1,611 words against
  1,535), result 2,007 cycles as without the pass; 4 rounds with `--greedy-barriers none`
  alone, 21 of 21 rejected (peak 1,649). With `--greedy-budget-headroom 250` on the shaped
  16-round build (budget 1,285) 18 of 19 candidates are rejected and the one accepted has
  1,280 cycles. Making the list scheduler respect the budget is the follow-up.
- **Memory edges with latency 0** (a store after a load of the same word) are scheduled with
  latency 1. The verifier checks the real latency, so this can only cost cycles.
- **CP-SAT pipeline: measured at two small sizes only** (table above). An accepted schedule
  must keep every op inside its ASAP/ALAP domain. Justification can raise peak scratch
  (4 x 256: 827 to 1,137 words), so at larger sizes a justified hint could send a section
  to the per-section repair where the greedy's would not; that has not been observed or
  measured.
- Shift decisions change only when the greedy owned them (no forced shift encoding, no hint
  pass); ALU-derived constants, eliminated vbroadcasts and individually forced shifts
  (what-if runs, whose forced set the greedy does not see) make the pass skip.

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
