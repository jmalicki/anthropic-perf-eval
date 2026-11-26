---
name: use-whatif-shift
description: Use the what-if shift script to force a specific VALU op to shift to ALU, re-optimize with CP-SAT, and compare. Use when debugging why the solver did or didn't shift specific ops, or testing the impact of forcing shift (cycles, ALU/VALU balance).
---

# What-if shift: debug VALU shift decisions

The **what-if shift** script forces a specific VALU op (by ID) to shift to ALU, re-runs CP-SAT to re-optimize the rest of the schedule, and optionally compares the result to a baseline. Use it to debug shift decisions and to test whether forcing shift improves or worsens cycles.

## When to use

- **Debug why the solver didn't shift** — After per-decile analysis, if "VALU other" is high and ALU has headroom in a decile, pick a candidate op and force it to shift; see if the resulting schedule is feasible and whether cycles improve or worsen.
- **Test impact of forcing shift** — Compare baseline cycles (and ALU/VALU usage) to the what-if result; see how many ops change cycle and whether the forced op is actually ALU in the output.
- **Follow up from schedule-debug** — Use [use-schedule-debug](../../use-schedule-debug/SKILL.md) (per-decile) to find candidate "VALU other" ops; then run what-if on one or more op IDs.

## Prerequisite

Scheduler output JSON from a run you care about (e.g. `uv run generate_ssa_kernel.py --output-json scheduler_output.json`). Use it to get op IDs and, optionally, as the baseline for `--compare-with`.

## How to run

```bash
# Force op to shift and re-optimize (default: 600s CP-SAT, rounds=1; use --rounds 2 for 2-round kernel)
uv run scripts/what_if_shift.py --op-id <id> [--cpsat-time-limit 600] [--rounds 2]

# Compare what-if result to a baseline JSON
uv run scripts/what_if_shift.py --op-id <id> --compare-with scheduler_output.json

# Custom output path (default: scheduler_output_whatif.json)
uv run scripts/what_if_shift.py --op-id <id> --output-json my_whatif.json
```

Output is written to `scheduler_output_whatif.json` by default. The script prints total cycles and, when a shift was forced, whether the op appears as ALU in the result. With `--compare-with`, it prints cycle difference, ops that changed cycle, and ALU/VALU / shifted-op counts.

## Getting op IDs

- **From scheduler output JSON:** Open the JSON (e.g. `scheduler_output.json`) and use the `ops` array; each op has an `id` field. Filter by `engine: "valu"` and optionally by opcode (non-vbroadcast, non-multiply_add are "VALU other" / shiftable).
- **From per-decile workflow:** Run [slot_usage_by_decile.py](../../../scripts/slot_usage_by_decile.py) to see which deciles have high "VALU other"; then inspect the JSON for ops in that decile (cycle range) with engine VALU and shiftable opcode.

## How the forced shift is applied

`forced_shifts=[op ids]` is a hard constraint of the CP-SAT model (the op's shift literal is fixed to 1) and the greedy scheduler honours it too:

- **Greedy:** a forced op is placed as 8 scalar ALU ops in one cycle on every path (normal, force-schedule, and when the pressure policy blocks shifting); it waits for a cycle with 8 free scalar slots and never takes a vector slot. The greedy log line reports it (`N VALU shifted`).
- **CP-SAT:** the greedy schedule is the warm start and bounds the model (ALAP cap = greedy cycle + slack, horizon). Because the greedy placed the forced ops shifted, the solver log should read `The solution hint is complete and is feasible`. If it reads `complete, but it is infeasible`, see the open BUGS.md entry on per-section repair.
- **Greedy alone** (`use_cpsat=False` on the binding): the emitted kernel runs the forced ops as scalar ops as well.
- **Where it applies:** only to shiftable vector ops (not `vbroadcast`, `multiply_add`, or non-vector ops), only with VALU shift enabled, and only under `--valu-shift optimize`; under `alu` / `valu` the set is ignored, by the model and by the greedy, with or without CP-SAT (`PipelineConfig::forced_shift_ids` is the one place that decides). Ids that do not qualify are ignored silently.

## Limitation

Only **force shift** (op must shift to ALU) is supported. **Force not shift** (`--force-shift false`) is not implemented; the script reports this and runs normal optimization.

## References

- Script: [scripts/what_if_shift.py](../../../scripts/what_if_shift.py)
- Schedule debug (per-decile, find candidates): [.cursor/skills/use-schedule-debug/SKILL.md](../../use-schedule-debug/SKILL.md)
- Bulk-move (compaction): [.cursor/skills/use-bulk-move/SKILL.md](../../use-bulk-move/SKILL.md)
- API: `scheduler_rs.build_tick_tock_kernel_ssa(..., forced_shifts=[op_id])` — see [scheduler_rs/src/python.rs](../../../scheduler_rs/src/python.rs).
