# Window optimality probe

**Question.** At 16 rounds the whole-program CP-SAT solve never improves on its greedy warm start. Is the warm start locally near-optimal, or is the model too big for the solver to find improvements that exist? The answer feeds the decision on windowed or rolling-horizon CP-SAT.

**Short answer (partial data, see [Status](#status)).** Every gain below was found on the reduced model and is **not confirmed by the full model** (the two models later disagreed on a path-bit kernel: BUGS.md, docs/STARTUP_WINDOW_PROBE.md); read them as reduced-model results.
- On the reduced model, the warm start is not locally optimal. With free encodings, about half of all 25-cycle windows and most 50-cycle windows can each be shortened by one cycle, proven optimal in most cases.
- Applied in sequence, those reduced-model gains add up. A rolling pass (incomplete, not re-checked on the full model) of 25-cycle windows took 1,399 to 1,393 cycles over the first 370 cycles of the schedule.
- Almost all of the gain needs different encodings, mainly VALU shift. With encodings fixed, 1 of 107 windows improves.
- If the full model confirms them, whole-program CP-SAT is failing on scale: the improvements exist and its search does not reach them. Not confirmed.

Tooling: `scripts/window_optimality_probe.py`, `scheduler_rs/src/cpsat_scheduler/window.rs`, skill `.cursor/skills/use-window-probe/SKILL.md`.

## Method

### Warm start

"Current best" below is as of this measurement. The best recorded schedules since are 1,012
and 1,033 cycles (greedy only, [CRITICAL_PATH.md](CRITICAL_PATH.md));
[STARTUP_WINDOW_PROBE.md](STARTUP_WINDOW_PROBE.md) part B probes those.

The current best 16-round schedule, the shaped greedy (main's default emission):

```bash
uv run python -m scripts.window_optimality_probe --rounds 16 --window-sizes 25 \
    --deterministic-time 20 --workers 2 --report window_probe_w25.json
# runs: CPSAT_WINDOW_PROBE=<spec> generate_ssa_kernel.py --rounds 16 --greedy-barriers none \
#       --greedy-priority wavefront --cpsat-deterministic-time 20 --cpsat-time-limit 900 --validate ...
```

The reference is 12,356 SSA ops with a makespan of 1,399 cycles. The generator builds its warm start as usual. Instead of the whole-program solve, it hands the warm start to `run_window_probe` (Rust). The probe first solves the model with every op pinned, the "all-fixed check". The check proves that the warm start is a solution of the CP-SAT model (status optimal, makespan 1,399, 11 to 16 s), and its solution becomes the reference. It gives a cycle to every op, including dead stores, and to every CSE shared SUB.

### The window model

Window `[a, a+W)`. The model is the full-kernel model from `schedule_ssa_cpsat`, with every constraint it normally has:
- slot capacities;
- precedence offsets: SelectChoice FMA +2, and element-wise shifted lanes;
- VALU shift;
- ConstChoice and SelectChoice encodings, including ALU-derived constants;
- CSE shared SUBs;
- the scratch cumulative.

`apply_window_constraints` adds the following, per op of the reference, by the cycles its slots occupy:

| Region | Constraint |
|---|---|
| Before (a slot before `a`) | every cycle variable, lane and encoding pinned to the reference |
| Inside (all slots in `[a, a+W)`) | kept in the window; encodings free (`free`) or pinned (`fixed`) |
| After (all slots at or after `a`, some at or after `a+W`) | pinned relative to each other, encodings pinned, moved by one shared `delta` in `[0, W]`: `cycle = reference - delta` |

**Objective: makespan alone.** When the last op is after the window, minimising the makespan maximises `delta`, the number of cycles by which the rest of the schedule can start earlier. That is the real makespan gain from compacting the window. The other option is to minimise the window's own span with the rest pinned. A shorter span need not move anything after the window, so it overstates the gain. Moving the After block keeps every constraint among After ops intact. The model still checks everything that crosses a boundary: precedence from Before and Inside ops; slot capacity where Inside and After ops now share cycles; and the scratch budget over the whole schedule, so a value defined before the window and read after it is counted at every cycle it is live.

**Reduced model (meant to be exact; see below).** Every op outside the window is pinned, so an op far from it only matters through constraints shared with nearby ops. `CpsatWindow::pruned_ops` marks the ops whose slots end before `a - W - 1`, or start at or after `max(a + 2W + 2, last Before slot + W + 3)`. Nothing that can move ever reaches those cycles:
- Inside ops stay in `[a, a+W)`;
- the After block moves at most `W` earlier;
- the SUBs are pinned or kept in the window;
- every dependency edge between a marked op and an unmarked one is at least two cycles long, the largest latency.

`restrict_to_window` keeps the unmarked ops, plus, for each value that touches them, its def and its latest marked reader (so its scratch interval is exact), plus the op with the last slot. Marked ops that remain are left out of every cumulative. A 16-round W=50 window drops from 12,356 to 1,585 ops. It produces the same presolved model (4,082 variables, 13,716 constraints) and the same result (saved 1, optimal at 8.5 deterministic time) as the unreduced model on that window. On a path-bit kernel the reduced and full models later disagreed (unresolved: BUGS.md, docs/STARTUP_WINDOW_PROBE.md), so treat reduced-model gains as unconfirmed until re-checked with `--no-prune`. `pruning_far_ops_keeps_the_window_optimum` and `reduced_model_with_cse_group_matches_full_model` check the same on small programs, including re-solving each expanded schedule on the full model with every op pinned. The rolling pass re-checks its final schedule the same way.

**Budgets.**
- 20 units of deterministic time per window, with a 900 s wall-clock backstop and 2 workers.
- The max-LP subsolvers (`max_lp`, `max_lp_sym`, `reduced_costs`, `pseudo_costs`) are left out. On the unreduced 16-round window model they spent 10 to 13 minutes of wall time building their LP relaxations at about 0.2 deterministic time. Interleaved search waits for every subsolver in a batch, so neither the budget nor LNS ever got going.
- The sweep strides by W/2. Every window is solved against the unchanged reference.
- The rolling pass places non-overlapping windows from cycle 0. Each next window starts where the previous one now ends. Each window is solved against the schedule the earlier windows improved, with encodings free, and each improvement is adopted.

## Results

Reference: 1,399 cycles. "Saved" is per window, against the reference. "Bound" is the most the window could save: the reference makespan minus CP-SAT's proven bound.

### Independent sweep

Reduced model (`restrict_to_window`); the gains are unconfirmed by the full model (`--no-prune`), which was not run on these windows.

| W | encodings | windows | proven optimal | improved | saved per improved window | max bound (open windows) | improved by phase | median / max s |
|---|---|---|---|---|---|---|---|---|
| 25 | free | 107 (all) | 87 | 49 | 1 (one window 2) | 8 | start-up 2, steady 40, tail 7 | 12 / 638 |
| 25 | fixed | 107 (all) | 102 | 1 | 1 | 0 | tail 1 | 0.4 / 4.7 |
| 50 | free | 37 of 56 (cycles 0 to 900) | 21 | 29 | 1 | 8 | start-up 3, steady 26 | 205 / 901 |
| 50 | fixed | not run yet | | | | | | |
| 100, 200 | | not run yet | | | | | | |

Phases: start-up is a window starting in the first 10 % of the schedule, tail is one ending in the last 10 %, steady state is everything else.

Notes:
- **Where.** W=25 windows improve at about the same rate in every decile of the schedule (improved/total per decile: 2/11, 6/11, 7/11, 1/11, 6/10, 6/11, 7/11, 5/11, 4/10, 5/10). Gains are not concentrated in the start-up or the tail. The fourth decile, around cycles 420 to 560, is the exception, with 1 window out of 11.
- **Encodings.** Every improving window with free encodings changed the shift of some of its ops: a median of 10 ops at W=25 and 14 at W=50. Across all improving windows, SelectChoice changed on 10 ops at W=25 and 10 at W=50, and ConstChoice on 4 at W=25. With encodings fixed, reordering alone found one cycle in one window (W=25 at cycle 1,313). The greedy's order is close to optimal for its own encodings; the slack is in its VALU-shift decisions.
- **Proofs.** At W=25, 87 of 107 free windows close to optimality, median 4 s. The open ones each found 1 cycle, with a bound of 2 to 8, and cluster around cycles 150 to 380 and 1,200 to 1,290. At W=50, 21 of 37 close, median 78 s for closed windows and 400 s for open ones.
- **Window boundary follow-up.** The original W=25 sweep reported four infeasible free windows and five infeasible fixed windows among starts 26, 481, 624, 793 and 962. In the saved fixed model at 481, the reference used a one-cycle vselect at 505. An unconditional constraint also required its unused FMA cycle (506) to fit before the window ended. The window now applies that constraint only when the FMA executes. With this change and the preceding reference-domain fix, all five starts solved to optimality at 1,399 cycles in both modes (one worker, deterministic budget 5). All five generator runs passed symbolic replay and numeric validation of the reference kernel. The table retains the original sweep counts and timings.

### Rolling pass (what rolling-horizon CP-SAT would get)

Reduced model, unconfirmed: the pass stopped at cycle 370, so its schedule was never re-checked on the full model or lowered.

| W | windows | improved | final makespan | saved | complete |
|---|---|---|---|---|---|
| 25 | 15 | 6 | 1,393 | 6 | no: stopped at cycle 370 of 1,399 |

The gain compounds: 6 cycles from the first 26 % of the schedule, from 6 of 15 windows. Every gain came from shift changes (6 to 38 ops per window). At the W=25 sweep's rate of 49 improving windows out of 107, a full pass would plausibly land near 1,375 to 1,380. That figure is an extrapolation, not a result. The final schedule of a complete pass is re-checked on the full model and lowered and validated by the generator (`--validate`). On the 2-round kernel this produced a validated 196-cycle kernel from the 197-cycle warm start.

### Solve time per window size

On a machine at load 130 to 180, about a quarter of a core per solve:

| W | ops in the reduced model | median wall s (free) | median wall s (fixed) |
|---|---|---|---|
| 25 | about 900 | 12 (closed windows: 4) | 0.4 |
| 50 | about 1,600 | 205 (closed: 78, open: 400) | not run |

A sweep needs one model build per window. Before PR #60 that build alone took 456 s at 16 rounds, almost all of it in `compute_capacity_lower_bound_global`; it now takes 5 s.

## What this means for windowed CP-SAT

1. **Build it.** The whole-program solve's failure is one of scale. Improvements exist in roughly every other 25-cycle window and in most 50-cycle windows, and CP-SAT proves them in seconds to minutes once the model is the window's. The whole-program model has 109,240 variables and 584,831 constraints, and its LP workers spend their budget loading. It does not find them.
2. **Keep encodings free inside the window.** That is where the gain is: fixed encodings find 1 cycle in 107 windows, free ones 49. A windowed pass that only reorders is not worth building.
3. **Use the reduced model and leave out the max-LP workers.** The reduction is meant to be exact (it was on the window checked here, but disagreed with the full model on a path-bit kernel: BUGS.md, docs/STARTUP_WINDOW_PROBE.md; re-check improving windows with `--no-prune`) and cuts the model about 8 times at W=50. Without it, and with the max-LP workers in, a single window ran out of wall time at 2.7 deterministic time.
4. **Window size.** W=25 closes most windows in seconds. W=50 finds an improvement in more windows (29 of 37) but takes about 15 times longer per window. A rolling pass at W=25 to 50 with stride W, roughly 30 to 60 solves, is the practical shape. The expected gain from the data so far is on the order of 20 cycles (1.5 %). That is useful but modest; the capacity floor for this emission is 1,271.
5. **Also look at the greedy's shift policy.** Nearly every gain comes from re-deciding VALU shifts locally. A better shift rule in the greedy could capture part of this without any solver.

## Status

Partial, because the shared machine's CPU is reserved for higher-priority work:
- complete: the W=25 sweep, both modes;
- partial: W=50 with free encodings, 37 of 56 windows; the W=25 rolling pass, stopped at cycle 370;
- not run: W=50 fixed, W=100, W=200, the W=50 rolling pass, and the full-model re-check of a complete rolling pass.

The reports are JSON files rewritten after every window: `uv run python -m scripts.window_optimality_probe --summarize REPORT...`.
