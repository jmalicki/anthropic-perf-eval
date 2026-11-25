# Scratch ceiling diagnosis (path-bit plans)

Questions: where do the 24 cycles between the path-bit plan `path_flow90.json` (1,033) and its
static bound (1,009) go; what is the "1,505-word ceiling" that 7 of 16 path-bit plans of the
stage-1 loop sat at (docs/EMISSION_STRUCTURE_MODEL.md section 10.0.4); how much is scratch
worth; and is there a small, sound scheduler fix.

All numbers are greedy development measurements at 16 rounds, batch 256, with the shaped
greedy flags of the best kernels (below). Every kernel quoted is PROVED (dataflow replay)
and PASSED (`--validate`); the six kernels in section 5 also pass `./scripts/check_upstream.sh
--strict`. Relaxed-scratch numbers (`--scratch-budget-override`, `--no-scratch-constraint`)
are analysis only, never kernels.

## Answers

1. **The 24 cycles of `path_flow90.json` are not scratch.** The bound is the load row
   (t 51 + q 14 + ceil(1,888 / 2)). The tail is exact (last load at T - q - 1). The loss is
   19 cycles of idle load slots in the start-up ramp (cycles 51 to 91) and 5 cycles of setup
   loads (head below t) that issued inside the load window (an exact split, checked). The
   idle slots are **attributed** (accounting, not counterfactual) upstream: the one flow slot
   (round 1 to 3 `bits-flow` selects) 9.5 cycles, the scalar ALU (setup) 4, the load engine
   (setup loads) 2.5, plus 3 cycles where a ready load waited. Attribution is not what
   removal buys: removing the flow starvation (`--greedy-flow-priority chain-first`,
   docs/STARTUP_AND_STALL_LOOP.md) bought 2 measured cycles, not 9.5. Inflation
   (231 deferrals), shift suppression and the budget play no part: relaxed scratch gives
   1,034 at every budget from 1,600 to unlimited, and so does turning the pressure policy off.
2. **There is no 1,505-word limit in the code.** The greedy's hard check
   (`live words after the op <= budget`, budget 1,535 = 1,536 less the zero word) never
   rejects an op on these runs (0 deferrals by the checked-in counter, printed as "Greedy
   budget check: N deferrals, M ops forced past the budget"); the counter agrees with interval liveness (0 mismatches) and
   address assignment has 0 fragmentation. What keeps the default runs below the budget is
   the pressure policy (ASAP inflation and shift suppression, both from 0.95 x 1,536 = 1,459
   words), and it has to: in the `words` scratch model, an op the hard check defers is
   **forced in the same cycle** whenever it has a slot (`schedule_ssa_greedy_core`, the
   emergency force-schedule block), so a run that reaches the budget overshoots it and
   address assignment rejects the kernel. That is why inflation thresholds of 0.99 and 1.0
   fail (peak 1,537 > 1,536) instead of running at the budget. The value 1,505 itself is the
   natural peak of those plans (1,537 with no scratch limit for `nogood-capped0`) less the
   few vectors the inflation feedback removes; live words move in 8-word steps (s0 plus whole
   vectors: 1,489, 1,497, 1,505, 1,513), so several plans land on the same step. Holding them
   there costs 49 to 78 cycles, because inflation defers operand-ready gather loads: on
   `nogood-capped0`, 152 load slots (76 cycles) idle while a ready load waited.
   Ruled out as the cause: groups-in-flight admission (32 = all groups), `pools` vs `words`
   (identical, 1,099 / 1,505), `--greedy-budget-headroom` (0), address fragmentation (0),
   counter vs allocator (agree).
3. **Scratch is worth nothing to the two best plans and up to 50 cycles to the ceiling
   plans.** `path_flow90.json` 1,033 -> 1,034 and `flow90.json` (index bits) 1,039 -> 1,039
   at any budget up to unlimited. A denser start (stagger 6) needs 1,700 words and still
   gives 1,036. The ceiling plan `nogood-capped0`: 1,099 at 1,536, 1,053 at 1,600, 1,050 from
   1,700 to unlimited (peak 1,537). Its 49 cycles were the policy, not the budget: section 5
   gets 1,049 within 1,536 words.
4. **Fix (flag, default unchanged): `--greedy-budget-force stall`.** An op the budget check
   defers waits while other ops are placed in the cycle and is forced only in a cycle where
   nothing was placed (a real deadlock), as the `pools` model already does. With it the hard
   check holds the budget by itself: the ceiling plan with the pressure policy off runs at
   peak 1,535 and 1,049 cycles. The best kernels do not change (1,033 / 1,034, 1,039): their
   gap is not scratch. Every ceiling plan improves by 8 to 78 cycles, none below 1,033.

## 1. The 24 cycles of `path_flow90.json`

`scripts/scratch_ceiling_diagnosis.py gap` takes t, q, W and the window loads from the static
bound's own load row and splits `T - bound` exactly into idle load slots in [t, T - q) and
extra loads in that window (loads W does not count, here all setup loads with head below t):
`2 * gap = idle + extra - (2 * ceil(W/2) - W)`, checked, with every window load verified to lie
in the window. It then **attributes** each idle slot by a blame chain: the loads issuing at the
next load cycle that could have run there share the slot equally; from each, back through its
latest producer (ties: smallest id) to the first op that issued later than its operands
allowed, charged to that op's engine with "no room" if its engine had no room in every cycle
it waited (a shiftable vector op: vector ALU full and fewer than 8 scalar slots free). This is
accounting, not a counterfactual: on `path_flow90.json` the 9.5 attributed flow cycles,
removed by chain-first, bought 2 measured cycles.

| Build (T / bound) | idle load slots in window | extra loads | flow no room | ALU no room | load no room | vector no room | ready load waited |
|---|---|---|---|---|---|---|---|
| `path_flow90.json` path (1,033 / 1,009) | 38 = 19.0 | 10 = 5.0 | 9.5 | 4.0 | 2.5 | 0 | 3.0 |
| `flow90.json` index (1,039 / 1,029) | 7 = 3.5 | 13 = 6.5 | 3.0 | 0 | 0.5 | 0 | 0 |
| `nogood-capped0` path, default (1,099 / 1,009) | 167 = 83.5 | 13 = 6.5 | 2.0 | 5.0 | 0.5 | 0 | **76.0** |
| `nogood-capped0` path, `stall` + 0.98 (1,049 / 1,009) | 67 = 33.5 | 13 = 6.5 | 8.0 | 5.0 | 0.5 | 19.0 | 1.0 |

Attributed cause columns are cycles; every "ready load waited" here is a dependent load (no
dependency-free constant load). The last load issues at T - q - 1 in all four (no tail
beyond q; such cycles would appear as idle slots). For `path_flow90.json` 35 of the 38 idle slots are in cycles 51 to
91. Every group's first gather load is dependency-ready at cycle 51 (round 4) or 62 (round
5); group 0 issues it at 59, group 1 at 70, group 2 at 81, 11 cycles apart: the start-up
fills the load pipe one group at a time. Changing the order does not help:

| `path_flow90.json` variant | cycles | peak |
|---|---|---|
| baseline (stagger 9) | 1,033 | 1,465 |
| `--greedy-flow-priority feeds-load-first` | 1,042 (flow starvation unchanged at 19 slots; 12 cycles of VALU starvation added) | 1,463 |
| stagger 12 | 1,065 | 1,281 |
| stagger 6 | fails the budget (peak 1,713) | - |
| stagger 6, relaxed 1,700 words or more (analysis) | 1,036 | 1,579 |

So the start-up is a trade: a denser start needs more scratch and gives nothing back.
What would close it is less work on the one flow slot before the first gathers (rounds 1 to
3 are `bits-flow`: 352 flow selects) or a start-up that runs the first groups' early rounds
before the setup; both are stage-1 or emission questions, not scratch.

## 2. The 1,505-word "ceiling"

`nogood-capped0` (`docs/emission_plans/path_nogood_capped0.json`, the stage-1 loop plan of that
name). Deferrals and forced ops are the greedy's own counters (`GreedyDiagnostics::
budget_check_deferrals`, `forced_past_budget`), printed in the build log as "Greedy budget
check: ..."; every count below was re-measured with them.

| Inflation threshold | Budget-check deferrals | Forced past budget | Peak (words) | Cycles | Result |
|---|---|---|---|---|---|
| 0.90 | 0 | 0 | 1,441 | 1,176 | PASSED |
| 0.95 (the shaped default) | 0 | 0 | 1,505 | 1,099 | PASSED |
| 0.96 | 0 | 0 | 1,505 | 1,060 | PASSED |
| 0.97 | 0 | 0 | 1,505 | 1,053 | PASSED |
| 0.98 | 0 | 0 | 1,513 | 1,049 | PASSED |
| 0.99 | 6 | 1 | 1,537 | - | fails address assignment (1,537 > 1,536) |
| 1.0 (inflation off) | 8 | 1 | 1,537 | - | fails |
| 0.97, soft and critical 0.999 | 18 | 2 | 1,537 | - | fails |
| 0.95, `--scratch-model pools` | 0 | 0 | 1,505 | 1,099 | PASSED |
| 0.99, `--scratch-model pools` | 6 | 1 | 1,537 | - | fails (the pools retry starts from the words run) |
| 0.99, `--greedy-budget-force stall` | 6 | 0 | 1,529 | 1,049 | PASSED |
| 1.0, `--greedy-budget-force stall` | 7 | 0 | 1,529 | 1,051 | PASSED |

Where the threshold is 0.98 or below the hard check never fires; above, it fires a handful
of times and one rejected op is forced in the same cycle (1,536 counter words > 1,535, plus
the zero word: 1,537). With 0.95 the interval liveness curve of the scheduler JSON tops out
at 1,481 (14 cycles), 1,489 (9), 1,497 (4), 1,505 (3).
The same 1,505 at 0.95, 0.96 and 0.97 with very different amounts of deferral (10,343,
1,673 and 185) shows it is not a threshold: it is these plans' natural peak clipped by a
few vectors, on the 8-word grid.

The cost of clipping at 0.95 is the inflation itself: from 1,459 words up it defers every
producer with slack, gather loads included, so ready loads wait (76 cycles on
`nogood-capped0`). At 0.97 the clip is the same and costs 4 cycles over the unconstrained run.

## 3. Cycles as a function of budget (relaxed scratch, analysis only)

Thresholds are fractions of the budget, so they scale with it.

| Plan, stagger | 1,536 | 1,600 | 1,700 | 2,000 | 4,096 | unlimited |
|---|---|---|---|---|---|---|
| `path_flow90.json` path, 9 | **1,033** (1,465) | 1,034 | 1,034 | 1,034 | 1,034 | 1,034 (1,465) |
| `flow90.json` index, 9 | **1,039** (1,431) | 1,039 | 1,039 | 1,039 | 1,039 | 1,039 (1,431) |
| `nogood-capped0` path, 9 | 1,099 (1,505) | 1,053 | 1,050 | 1,050 | 1,050 | 1,050 (1,537) |
| `path_flow90.json` path, 6 | fails (1,713) | 1,093 | 1,036 | 1,036 | 1,036 | 1,036 (1,579) |
| `flow90.json` index, 6 | 1,070 (1,481) | 1,048 | 1,048 | 1,048 | 1,048 | 1,048 (1,513) |
| `nogood-capped0` path, 6 | 1,179 (1,507) | 1,151 | 1,103 | 1,046 | 1,046 | 1,046 (1,721) |

Peaks in parentheses. Neither best plan gets closer to its bound with more scratch.

## 4. The flag

`--greedy-budget-force {same-cycle,stall}` (default `same-cycle`, the historical behaviour;
`GreedyShape::budget_force`, `BudgetForce` in `scheduler_rs/src/kernel_builder/ssa_greedy/
mod.rs`). `stall` changes one condition of the force-schedule block: in the `words` model the
deferred op is forced only when `scheduled_this_cycle == 0`, otherwise it goes back on the
ready heap for the next cycle (the existing carry-over). The `pools` model already did this.
Soundness is unchanged: the force path, slot limits and the counter are the same code;
`stall` only forces less often, and a schedule over budget still fails address assignment.
A cycle whose only placement is the first part of a split shift also counts as progress
(defensive: a split starts only with 1 to 7 scalar slots free, which takes a counted
placement on the scalar ALU in the same cycle, so it does not occur; the test checks the
counter `split_only_cycles` stays 0). Tests:
`test_budget_force_stall_keeps_words_budget_where_same_cycle_does_not` (three broadcasts ready
together, budget for two: `same-cycle` goes over, `stall` does not),
`test_budget_force_stall_still_breaks_deadlock`, and
`test_budget_force_stall_holds_the_budget_in_the_shaped_pipeline` (adaptive policy with ASAP
inflation over policy bounds, wavefront, VALU shift with and without split shifts, four
binding budgets: `stall` never forces and its peak stays within the budget; control:
`same-cycle` overshoots).

## 5. Results with the flag

| Plan (static bound) | default | `stall`, 0.95 | `stall`, 0.98 | `stall`, 0.99 | `stall`, 1.0 |
|---|---|---|---|---|---|
| `path_flow90.json` path (1,009) | **1,033** / 1,035 | 1,033 | 1,034 | 1,034 | 1,034 / 1,036 (U) |
| `flow90.json` index (1,029) | **1,039** / 1,041 | 1,039 | 1,039 | 1,039 | 1,039 / 1,041 (U) |
| `nogood-capped0` path (1,009) | 1,099 / 1,101 | 1,099 | **1,049** / 1,051 (U) | 1,049 | 1,051 / 1,053 (U) |
| `nogood-capped2` path (1,009) | 1,118 | - | 1,053 | - | 1,051 (U) |
| `nogood-capped3` path (1,009) | 1,118 | - | 1,053 | - | 1,051 |
| floor-optimal `caps-underfilled0` path (1,005) | 1,142 | - | 1,064 | - | 1,064 (U) |
| `caps-underfilled5` path | 1,130 | - | 1,098 | - | 1,098 |
| probe A (`path_flow90.json`, round 4 `bits-flow`) | 1,101 | - | 1,093 | - | 1,093 |

Scheduled / simulated where both are given. (U): the saved kernel passes
`./scripts/check_upstream.sh --strict`. `nogood-capped0` with `stall`, inflation 1.0 and
soft/critical 0.999 (pressure policy off): 1,049, peak 1,535. `path_flow90.json` the same way:
1,034.

The flag removes the ceiling and is worth 8 to 78 cycles on plans that hit it, but the best
kernel stays `path_flow90.json` at 1,033. Default stays `same-cycle`: with the shaped
default threshold (0.95) the two rules give identical kernels, and the gain needs a higher
threshold too, which costs 1 cycle on `path_flow90.json`.

## What next

- For stage 1: rerun the path-bit loop with `--greedy-budget-force stall
  --pressure-inflation-threshold 0.98`. The plans the loop scored at 1,099 to 1,142 now score
  1,049 to 1,064, which changes which path plans look promising, though none here beats
  1,033.
- For `path_flow90.json` the remaining 24 cycles are the start-up: 9.5 on the flow slot
  (round 1 to 3 selects before the first gathers) and 5 of setup loads inside the window.
  Candidates: `multiply_add` selects (`bits`) in round 1 to 3 for the first few groups only,
  or ordering the setup loads before t. Neither is a scratch question.

## Reproduce

```bash
# Shaped greedy flags of the best kernels:
F="--rounds 16 --use-greedy --validate --greedy-barriers none --greedy-priority wavefront \
  --greedy-split-shift --pressure-inflation-threshold 0.95 --greedy-load-priority final-last \
  --greedy-groups-in-flight 32 --greedy-stagger-cycles 9 --merge-hash-stages --index-form mirrored"
uv run generate_ssa_kernel.py $F --emission-plan docs/emission_plans/path_flow90.json \
  --select-bit-source path --output-json path.json --save-kernel path_kernel.json
uv run python -m scripts.scratch_ceiling_diagnosis gap path.json
uv run python -m scripts.scratch_ceiling_diagnosis peak path.json
# The ceiling plan, default and with the flag:
uv run generate_ssa_kernel.py $F --emission-plan docs/emission_plans/path_nogood_capped0.json \
  --select-bit-source path --output-json ceil.json
uv run generate_ssa_kernel.py $F --emission-plan docs/emission_plans/path_nogood_capped0.json \
  --select-bit-source path --greedy-budget-force stall --pressure-inflation-threshold 0.98 \
  --output-json ceil_stall.json --save-kernel ceil_stall_kernel.json
./scripts/check_upstream.sh --kernel ceil_stall_kernel.json --strict
# Relaxed scratch (analysis only): add --scratch-budget-override 1600 (or --no-scratch-constraint).
```

`gap` reads t, q and W from the static bound itself (the generator prints the same row as
"Static head/tail bound"). The other loop
plans of section 5 are the `plan` strings of `docs/emission_plans/loop_report_load_tail.json`
(`path_bits`), written as plan JSON the same way as `path_nogood_capped0.json`. About 30 s per
build including proof and validation; about 100 builds were used, one at a time.
