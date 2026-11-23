---
name: use-emission-ablation
description: Measure what an emission option (--merge-hash-stages, --index-form, --cached-select, --values-only, ...) does to the ops the kernel emits - vector-op equivalents, loads, flow ops, capacity floor, cycles - with every kernel proved against its SSA program and validated. Use when adding or changing an emission option, when asked which options to turn on, or to refresh docs/EMISSION_ABLATION.md.
---

# Emission ablation

**Module:** `scripts/emission_ablation.py`. Results and recommendation: [docs/EMISSION_ABLATION.md](../../../docs/EMISSION_ABLATION.md).

## How to run

```bash
./scripts/build_scheduler_rs.sh --release      # the script uses the installed extension

uv run python -m scripts.emission_ablation                                  # 16 rounds, greedy, all configurations
uv run python -m scripts.emission_ablation --rounds 4 --only baseline,merge-hash
uv run python -m scripts.emission_ablation --upstream --markdown            # plus the upstream simulator; Markdown table
uv run python -m scripts.emission_ablation --rounds 3 --batch-size 64 --cpsat-deterministic-time 10 \
  --only baseline,all-safe-bits --select-choice fma                         # through CP-SAT, forced encodings
uv run python -m scripts.emission_ablation --keep out/ --json rows.json     # keep scheduler JSONs and kernels
```

Run it as a module from the project root. Exit status is 1 if any configuration fails to build, is not PROVED, does not PASS, or fails upstream.

## Columns

| Column | Meaning |
|---|---|
| Vector-op equiv. | Vector ops + scalar ops / 8, as `scripts.schedule_report` counts them |
| Loads, Flow | Lowered ops on those engines (constants and selects as the scheduler encoded them) |
| Floor | Capacity floor for the ops this run emitted (`scripts.schedule_report`). Not a bound on the problem |
| Binds | The engine whose term is the floor (`alu+valu`, `load`, `flow`, `store`). Only work removed from that engine lowers the floor |
| Floor, best split | The same with the selects divided between the flow engine and the vector ALU in the best way (`best_select_split_floor`). Differs from Floor when the scheduler's division is far from the best, e.g. greedy puts every SelectChoice on flow |
| Cycles (sched / sim) | Schedule length and what `problem.Machine` counts |
| PROVED | Dataflow replay: the kernel equals its SSA program for all inputs |
| PASSED | Numeric validation against `reference_kernel2`: values and indices, or "(values only)" for configurations with `values_only` |
| Upstream | `./scripts/check_upstream.sh --kernel` on the saved kernel (with `--upstream`; needs 16 rounds, batch 256, height 10) |

## Adding an option

1. Prove the identity in `verification/rewrites.py` (register it in `REWRITE_CHECKS`, add a `*-needs-*` check for each precondition) before emitting anything that relies on it.
2. Add the option to `ProblemSpec`, to the emission block of the binding (keyword-only; its own line in the `#[pyo3(signature)]` and its own block in the parameter list, ahead of the scheduler options), and to `EMISSION_DEFAULTS` and `add_emission_args` in `kernel_build_args.py`. That dict feeds `run_options`, the binding kwargs and the run header, so nothing else needs the name.
3. Check the default emission did not move: build 16 rounds with no option on `main` and on the branch and compare the `ssa_ops` and `ops` of the two scheduler JSONs. Reusing a constant through a different helper, or reordering two ops, renumbers everything and shifts the greedy schedule.
4. Add the configuration to `CONFIGS` here, a test in `tests/test_emission_options.py` (PROVED, PASSED, fewer ops), and a row in `docs/EMISSION_ABLATION.md`.

## Reading the result

- The floor falls by 1 cycle per 7.5 vector-op equivalents only while the ALUs bind. With `--merge-hash-stages --index-form mirrored --cached-select bits` loads bind (2,114 loads, floor 1,057) and further vector-op savings do not show.
- Greedy cycles may rise while the floor falls (`--cached-select bits` alone: floor -86, greedy +47). Compare floors across emissions and cycles across schedulers.
- A configuration that fails to build with `scratch budget exceeded` is reported as a row, not an exception (`--select-bit-source path` at 16 rounds).
