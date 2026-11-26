---
name: use-window-probe
description: Measure whether a schedule is locally near-optimal by re-optimising windows of W cycles with the full CP-SAT model while everything outside the window stays fixed (ops after it move as one block). Use when deciding on windowed / rolling-horizon CP-SAT, when whole-program CP-SAT fails to improve a warm start, or to find where in a schedule (start-up, steady state, tail) cycles can still be saved.
---

# Window optimality probe

**Script:** `scripts/window_optimality_probe.py`. **Rust:** `scheduler_rs/src/cpsat_scheduler/window.rs` (`run_window_probe`, `apply_window_constraints`). **Results and reading:** [docs/WINDOW_OPTIMALITY_PROBE.md](../../../docs/WINDOW_OPTIMALITY_PROBE.md).

It is an analysis tool, off by default. The generator only runs it when `CPSAT_WINDOW_PROBE` names a spec file; the script writes that file for you.

## How to run

Rebuild `scheduler_rs` first if Rust changed (`./scripts/build_scheduler_rs.sh --release`). Run one or two probes at a time on a shared machine; every window solve is on a deterministic-time budget, so results do not depend on load (wall times do).

```bash
# 16 rounds, shaped greedy warm start, W=50: sweep (stride W/2, free and fixed encodings),
# rolling pass, then lower and validate the best rolling schedule.
uv run python -m scripts.window_optimality_probe --rounds 16 --window-sizes 50 \
    --deterministic-time 30 --report window_probe_w50.json

# Several sizes, sweep only, encodings free, back-to-back windows:
uv run python -m scripts.window_optimality_probe --window-sizes 100 200 --stride-fraction 1 \
    --encodings free --no-rolling --report window_probe_big.json

# Only part of the schedule:
uv run python -m scripts.window_optimality_probe --window-sizes 25 --cycle-range 600 800 ...

# Summarise reports (Markdown tables; reports of one warm start can be merged):
uv run python -m scripts.window_optimality_probe --summarize window_probe_w50.json window_probe_big.json
```

Generator options go after `--` (for example `-- --tree-reduce-max-depth 2`). `--no-shaped-greedy` uses the default round-barrier greedy as the warm start.

## What the model is

The full-kernel model (`schedule_ssa_cpsat`) with every constraint it normally has: slot capacities, precedence offsets (SelectChoice FMA +2, shifted lanes element-wise), VALU shift, ConstChoice and SelectChoice encodings, CSE shared SUBs, the scratch cumulative over the whole schedule. On top of it, per op of the reference schedule:

| Region | Rule |
|---|---|
| Before (a slot before the window) | Pinned at its cycles, encodings fixed |
| Inside (all slots in `[a, a+W)`) | Free within the window; encodings free (`free`) or fixed (`fixed`) |
| After (all slots at or after `a`, some at or after `a+W`) | Pinned relative to each other, encodings fixed, shifted earlier by one shared `delta` in `[0, W]` |

The objective is the makespan alone, so the solver maximises `delta`: the saving is the number of cycles the rest of the schedule can start earlier, i.e. a real makespan gain. A window with length 0 pins everything; the probe solves it first ("all-fixed check") to prove the warm start is a solution of the model.

## Keeping it fast

- **Reduced model** (default; `--no-prune` solves the full model instead, for checks): ops far from the window are left out of the cumulatives and, apart from the def and latest reader of each value that reaches the window, out of the model (`restrict_to_window`). It is meant to be exact (same presolved model and result as the unreduced one on a 16-round window), but on a path-bit kernel the two disagreed (BUGS.md, unresolved): re-check any reduced-model gain with `--no-prune`.
- **Max-LP workers left out** (default `--ignore-subsolvers max_lp max_lp_sym reduced_costs pseudo_costs`; pass `--ignore-subsolvers` with no names to keep them): on big window models they spend many minutes building LP relaxations at almost no deterministic time, and interleaved search waits for them.
- `--workers 2` and a deterministic budget of 10 to 20 per window are enough: most W=25 windows close in seconds.
- The all-fixed check runs on the full model with one worker; the rolling pass's final schedule is re-checked the same way (`rolling_checks` in the report).

## Reading the report

- `saved` is proven optimal for that window when `status` is `optimal`; otherwise `saved_bound` (reference makespan minus CP-SAT's bound) is the most it could save.
- `fixed` vs `free` separates pure reordering from gains that need different encodings; `shift_changes`, `select_changes`, `const_changes` count changed ops inside each window.
- The rolling rows are "what rolling-horizon CP-SAT would get": non-overlapping windows from cycle 0, each solved against the schedule the previous ones improved.
- The generator continues with the best rolling schedule (or the warm start), so `--validate` checks the improved kernel end to end.

## Gotchas

- Not pinned: the cycle variable of an op that runs shifted (its lanes and completion are), and CSE shared SUBs (no op of their own). Both still satisfy every full-model constraint.
- `greedy_alap_slack` (default 15) still caps every op at reference cycle + 15, as in the full solve.
- The report is rewritten after every window, so `--summarize` works on a run that is still going.
