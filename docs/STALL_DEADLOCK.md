# Why `--greedy-budget-force stall` fails on some plans

Under the stall judge (shaped load-tail greedy, `--greedy-budget-force stall
--pressure-inflation-threshold 0.98`), 3 of the 28 path-bit plans of the top-k rerun
(docs/STARTUP_AND_STALL_LOOP.md section A) fail the scratch budget: the greedy forces an op at
1,528 live words and then stays over budget for 265 to 874 cycles (peaks 1,761 to 1,817). The
plans are checked in as `docs/emission_plans/stall_deadlock_{floor10,gm6,gm14}.json`.

All numbers are greedy development measurements (16 rounds, batch 256, `--merge-hash-stages
--index-form mirrored`, path bits). "Measured" = built; the state at the failure is read from
the greedy itself (`SCHEDULER_DEADLOCK_DUMP`), not reconstructed.

## Answers

1. **The first forced cycle is empty because ASAP inflation deferred the waiting work, not
   because nothing can run.** On all three plans, at the first forced cycle (491, 353, 504)
   the engines are idle and 1,528 words are live (7 below the 1,535 assignable words, less than
   one vector). The hard budget check refused 3 ready ops, each writing a vector (8 words) and,
   by the script's approximate count, freeing none. Every other op in the ready heap (201, 266,
   182) was deferred by ASAP inflation in that same cycle: above 0.98 x 1,535 = 1,504 words it
   defers every op with slack and lets 3 per cycle through (read from the dump:
   `inflation_deferred` equals the heap on all three). The heap holds no op for any other
   reason. `stall` forces in any cycle where nothing was placed, so it forces here, before
   inflation's own breaker (3 empty cycles) acts. Whether the inflation-deferred ops would
   have freed enough words is not shown.
2. **Waiting did not save `floor10` in one experiment.** A rule that forces only when nothing
   waits in the heap (implemented and measured on all three plans, then removed) saved none of
   them. Its state was read for `floor10` only: cycle 509, 1,528 live words, nothing waiting,
   213 ready ops (selects and hash ops of the late groups 26 to 31 and shared round-0/1 ops),
   every one adding 8 words and, by the approximate count of
   `scripts/stall_deadlock_diagnosis.py`, freeing none. With 7 words free **no single ready
   op fits, so the greedy, which places ops one at a time, cannot proceed**. This does not
   show that no in-budget schedule exists from that state: a set of ops issued in one cycle
   that together free as many words as they write is not excluded. The state came from a
   force rule that was removed again, so nothing in the tree reproduces it. The live words are
   the late groups' path bits and tree-access values of rounds 2 to 4 plus the early groups'
   values of rounds 9 to 15: all 32 groups are in flight.
3. **Admitting fewer groups avoids it.** With 24 or 20 groups in flight all three plans build
   within the budget (measured, PROVED, PASSED; 20 in flight: 1,394 / 1,159 / 1,281 cycles;
   24: 1,444 / 1,176 / 1,279). With 32 they fail under both judges tried, `stall` + 0.98 and
   the default (`same-cycle`, inflation 0.95), so the failure is not specific to `stall`. With
   28 in flight `floor10` fails differently: the counter stays within the budget (0 forced) but
   address assignment finds no layout (the `words` model's known limit; `--scratch-model
   pools`). Whether 32 groups make the failure unavoidable is not shown.
4. **Decision: leave `stall` as is and document it.** None of the three rules tried saves
   these plans:
   - forcing only when no op waits in the heap: identical on every passing plan (1,030,
     1,039, the ceiling plans 1,049 / 1,053 / 1,064 / 1,098) and saves none of the three, so
     it was not kept;
   - a reserve (`--greedy-budget-headroom` 16 / 48 / 96): all three still fail (their states
     were not dumped);
   - admission (fewer groups in flight) avoids it, at 159 to 414 cycles above the best kernels.

   A different choice of which op to force does not help once the force is past the budget,
   since that kernel is invalid; a rule that placed several ops in one cycle to free words
   first was not tried. `stall` is therefore not a safe judge for plan search: a plan that
   fails under it is a failed build at 32 groups in flight, which a loop should record as a
   failure (as it does), not as evidence about the plan's floor.

## Tools

- `SCHEDULER_DEADLOCK_DUMP=state.json`: the greedy writes its state at the first cycle of a run
  that reaches the force path without `same-cycle` (with `stall`, or with scratch model
  `pools`); the dump is written before the slot check, so that cycle may force nothing. Fields:
  cycle, live words, live values (counted group keys, not lanes) with remaining readers, the
  ready ops deferred this cycle with their reason (`budget`: the hard check; `pressure`: the
  policy's soft deferral), the ops ASAP inflation deferred this cycle, the ready heap
  (`priority` is the heap's order key, not a wait: every entry whose operands are visible and
  that is admitted is offered again each cycle) with admission, and slots used. The counter is
  per process and per path and counts dumps, not greedy runs: the first dump goes to the given
  path, the k-th to `<path>.<k-1>`; a long-lived process (a plan-search loop) keeps counting
  across builds.
- `uv run python -m scripts.stall_deadlock_diagnosis state.json scheduler_output_error.json
  [--live]`: joins it with the failed build's exported schedule; prints the deferred ops with
  their reason and words added and freed, and the live words by the state of each value's
  first remaining reader (found through source operands only: values read through a lane
  alias, `keep_live` or a delta are reported as "readers not resolved", 80 to 176 words on
  these plans); with `--live` every live value and its first readers.
- `tests/test_stall_deadlock.py`: the three plans under the stall judge fail with
  `ScratchBudgetExceeded`, with ops deferred by inflation and 3 vector writes refused by the
  hard check in the first forced cycle (pinned limitation), and build within the budget with
  20 groups in flight at the cycle counts above.

## Reproduce

```bash
J="--rounds 16 --use-greedy --validate --greedy-barriers none --greedy-priority wavefront \
  --greedy-split-shift --greedy-load-priority final-last --greedy-groups-in-flight 32 \
  --greedy-stagger-cycles 9 --merge-hash-stages --index-form mirrored --select-bit-source path \
  --greedy-budget-force stall --pressure-inflation-threshold 0.98"
SCHEDULER_DEADLOCK_DUMP=state.json uv run generate_ssa_kernel.py $J \
  --emission-plan docs/emission_plans/stall_deadlock_floor10.json      # fails: ScratchBudgetExceeded
uv run python -m scripts.stall_deadlock_diagnosis state.json scheduler_output_error.json
uv run generate_ssa_kernel.py $J --emission-plan docs/emission_plans/stall_deadlock_floor10.json \
  --greedy-groups-in-flight 20                                         # 1,394 cycles, PASSED
```
