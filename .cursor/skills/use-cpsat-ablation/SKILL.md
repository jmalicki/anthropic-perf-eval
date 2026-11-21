---
name: use-cpsat-ablation
description: Ablate CP-SAT configuration and modelling switches (workers, interleaved search, subsolvers, objective, greedy ALAP cap, horizon, makespan interval, cumulative propagators, redundant rows) over a set of instances with a deterministic-time budget, and get a results table (JSON + Markdown). Use when asked whether a solver option helps, when adding a new CP-SAT switch, or when re-measuring after a model change.
---

# CP-SAT configuration ablation

**Script:** `scripts/run_cpsat_ablation.py` (run as a module from the project root; build
`scheduler_rs` first with `./scripts/build_scheduler_rs.sh --release`).

It runs every configuration on every instance with the same `--cpsat-deterministic-time`
budget, validates every kernel (symbolic dataflow proof and numeric validation), and writes
`<out-dir>/<name>.json` and `<out-dir>/<name>.md`. Recorded results and the recommendation
from the last study are in [docs/CPSAT_CONFIG_ABLATION.md](../../../docs/CPSAT_CONFIG_ABLATION.md).

For comparing arbitrary `generate_ssa_kernel.py` options on one instance with a wall-clock
limit, use `scripts/run_ablation.py` (skill `use-ablation-analysis`) instead.

## Usage

```bash
# The standard study: baseline, then each switch alone, on 2x256, 3x256, 4x64, 4x256, 6x64
uv run python -m scripts.run_cpsat_ablation --out-dir build/cpsat_ablation --jobs 2

# See the standard configurations
uv run python -m scripts.run_cpsat_ablation --list-configs

# A subset of the standard list (the baseline is always kept), one instance, own budget
uv run python -m scripts.run_cpsat_ablation --instances 4x64@10 --only makespan-only,tight-horizon

# Add a combination to the standard list
uv run python -m scripts.run_cpsat_ablation \
    --add-config "combo:--cpsat-objective makespan --no-cpsat-horizon-num-ops --no-cpsat-horizon-max-latency"

# Own list ("name:generate_ssa_kernel.py options"; the first one is the baseline)
uv run python -m scripts.run_cpsat_ablation --instances 2x256,16x256@60 \
    --config "baseline:" --config "w16:--cpsat-workers 16"

# Rebuild the tables from finished runs (no solving)
uv run python -m scripts.run_cpsat_ablation --out-dir build/cpsat_ablation --report-only
```

Options: `--instances ROUNDSxBATCH[@BUDGET],...`, `--budget` (default 20), `--config`,
`--add-config`, `--only`, `--shared-args`, `--time-limit` (wall-clock backstop, default
3600 s), `--nondeterministic-repeats` (default 3), `--jobs`, `--out-dir`, `--name`,
`--force`, `--report-only`, `--list-configs`.

Finished cells are saved as `<instance>__<config>.result.json` next to their log, schedule
JSON and kernel, and reused when the same instance, options and budget are requested again
with the same `scheduler_rs` build (the reuse key includes a hash of the installed extension,
so rebuilding after a model change solves every cell again). So a study can be extended (more
configurations, more instances) or resumed after an interruption. `--force` reruns anyway;
use it after changing Python code that affects the build (e.g. `kernel_build_args.py`), which
the key does not cover. `--report-only` tabulates whatever is saved, whatever build made it.

## Reading the table

| Column | Meaning |
|---|---|
| Cycles | Schedule length of the validated kernel. For a non-reproducible configuration: min / median / max over its runs. |
| vs baseline | Difference to the first configuration's cycles. |
| Status | CP-SAT status. `OPTIMAL` is optimal **for the emitted ops inside the model's domains** (greedy ALAP cap, horizon), not for the problem. |
| Bound (lb_cycles) | The solver's lower bound in cycles, same scope as above. |
| Floor | Capacity floor for the emitted ops (`scripts.schedule_report`). |
| Final incumbent at | When the final makespan was first reached: wall seconds (from the solver log) and a deterministic-time estimate. CP-SAT does not print deterministic time per solution; the estimate assumes it advanced evenly over the solve. |
| Solver dtime / wall s | Totals reported by CP-SAT. The deterministic time is summed over workers, so it exceeds the budget, which applies per worker. |
| Valid | Must read `PROVED, PASSED`. Anything else means the run is not a result. |

## Rules

- **Budget in deterministic time, never wall clock.** A run flagged `wall-clock limit hit`
  is invalid: raise `--time-limit` and run the study again. Such cells (and cells that did not
  validate) are never reused; they are solved again on the next invocation.
- **The runner owns** `--cpsat-deterministic-time`, `--cpsat-time-limit`, `--rounds`,
  `--batch-size`, `--output-json`, `--save-kernel` and `--validate`; a configuration that sets
  one is rejected. Give budgets per instance (`4x64@10`). Each `ROUNDSxBATCH` may appear once
  per invocation; use another `--out-dir` to compare budgets.
- `--report-only` refuses a saved cell whose options or budget differ from what is requested.
- **Same budget for baseline and variant.** Budgets are per instance (`@BUDGET`).
- **Reproducible configurations may run in parallel** (`--jobs`): load does not change what
  they return, only how long they take.
- **`--no-cpsat-interleave` with more than one worker is not reproducible.** The script runs
  such configurations `--nondeterministic-repeats` times, one at a time after the others,
  and reports the spread. Never quote a single run. Their result also depends on machine
  load, and the same budget buys them more work (the budget is per worker and all workers
  run concurrently rather than in turn), so compare wall time as well.
- **OR-Tools 9.15 has no reproducible configuration with bound workers**: with
  `interleave_search` the `probing`, `objective_shaving*`, `lb_tree_search*` and
  `objective_lb_search*` subsolvers are dropped whatever `--cpsat-subsolvers` /
  `--cpsat-extra-subsolvers` say (`GetFullWorkerParameters` in `cp_model_search.cc`). Check
  the "N full problem subsolvers: [...]" line of the log for what actually ran.
- If two runs of a reproducible configuration differ, that is a bug in the variant: export
  the models (`CPSAT_EXPORT_MODEL=a.pb`) and diff them with `scripts.diff_cpsat_models`.

## Adding a switch

1. Add the field to `CpsatTuning` (`scheduler_rs/src/cpsat_scheduler/config.rs`) with a
   default that leaves the model unchanged; use it in `model.rs`.
2. Add the CLI flag (default `None`) and an entry in `CPSAT_TUNING_OPTIONS`
   (`kernel_build_args.py`). Nothing else needs plumbing: the options travel as one JSON
   object and `run_options.cpsat_tuning` records the effective values.
3. Add tests in `tests/test_cpsat_tuning.py` (flag reaches the model or the solver
   parameters; explicit default equals the default model) and document the flag in CLAUDE.md.
4. Add a configuration to `STANDARD_CONFIGS` in the script and run the study.
