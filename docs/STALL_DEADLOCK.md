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

1. **The first forced cycle is not a deadlock.** On all three plans, at the first forced
   cycle (491, 353, 504) the engines are idle, 1,528 words are live (7 below the 1,535
   assignable words, less than one vector), the budget check has deferred 3 ready ops, each of
   which adds 8 words and frees none, and **182 to 266 other ops wait in the ready heap until 8
   cycles later**: ASAP inflation (above 0.98 x 1,535 = 1,504 words it defers every op with
   slack by up to 10 cycles, letting 3 ops per cycle through) parked them, including the
   consumers whose reads would free words. `stall` forces in any cycle where nothing was
   placed, so it forces here, before inflation's own breaker (3 empty cycles) acts.
2. **Waiting does not save these plans either: the greedy has painted itself in.** A rule that
   forces only when nothing waits in the heap (implemented and measured, then removed) gets
   one step further on each plan and reaches a real deadlock: e.g. `floor10` at cycle 509,
   1,528 live words, nothing waiting, 213 ready ops (selects and hash ops of the late groups
   26 to 31 and shared round-0/1 ops), **every one adding 8 words and freeing none** (their
   inputs all have other readers left). With 7 words free no op fits, and no op frees words
   until one runs: no in-budget schedule exists from that state. The live words are the late
   groups' path bits and tree-access values of rounds 2 to 4 plus the early groups' values of
   rounds 9 to 15: all 32 groups are in flight.
3. **Which choice made it unavoidable:** admitting all 32 groups. With 24 or 20 groups in
   flight all three plans build within the budget (measured, PROVED, PASSED; 20 in flight:
   1,394 / 1,159 / 1,281 cycles; 24: 1,444 / 1,176 / 1,279). The plans also fail under the
   default judge (`same-cycle`, inflation 0.95): they are plans the shaped greedy cannot run
   with 32 groups in flight, not a defect of `stall` alone. With 28 in flight `floor10` fails
   differently: the counter stays within the budget (0 forced) but address assignment finds no
   layout (the `words` model's known limit; `--scratch-model pools`).
4. **Decision: leave `stall` as is and document it.** No small rule fixes these plans:
   - a smarter force rule cannot: once any op is forced over the budget the kernel is invalid;
   - forcing only in a real deadlock (no op waiting) is sound, changes nothing on any passing
     plan (1,030, 1,039, the ceiling plans 1,049 / 1,053 / 1,064 / 1,098: identical), and saves
     none of the three, so it was not kept;
   - a reserve (`--greedy-budget-headroom` 16 / 48 / 96) only moves the same deadlock lower:
     all three still fail;
   - admission (fewer groups in flight) avoids it at 159 to 414 cycles above the best kernels.

   `stall` is therefore not a safe judge for plan search: a plan that fails under it is a plan
   the shaped greedy cannot run at 32 groups in flight, which a loop should record as a
   failure (as it does), not as evidence about the plan's floor.

## Tools

- `SCHEDULER_DEADLOCK_DUMP=state.json`: the greedy writes its state at the first cycle the
  force path runs without `same-cycle` (cycle, live words, live values with remaining readers,
  budget-deferred ops, the ready heap with each op's ready cycle and admission, slots used).
- `uv run python -m scripts.stall_deadlock_diagnosis state.json scheduler_output_error.json
  [--live]`: joins it with the failed build's exported schedule; prints the deferred ops with
  words added and freed, the live words by the state of each value's first remaining reader,
  and with `--live` every live value and why its readers cannot run.
- `tests/test_stall_deadlock.py`: the three plans under the stall judge fail with
  `ScratchBudgetExceeded` after an inflation-emptied cycle (pinned limitation), and build
  within the budget with 20 groups in flight at the cycle counts above.

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
