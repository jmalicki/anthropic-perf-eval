# CLAUDE.md

[![CI](https://github.com/jmalicki/anthropic-perf-eval/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/jmalicki/anthropic-perf-eval/actions/workflows/ci.yml)

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **VLIW SIMD Kernel Optimization Challenge**. The goal is to minimize execution cycles for a tree traversal + hashing kernel on a custom VLIW architecture.

**Performance Status:**
- Baseline: 147,734 cycles
- 16 rounds (the benchmark size): no recorded CP-SAT result yet. Greedy reference, not a recorded result (see "Scheduler Selection Guide"): 2,007 cycles (73.6x speedup; `--use-greedy` via the CLI, validated). Whole-program CP-SAT did not improve on the greedy schedule in 600 s. With the non-default shape options (`--greedy-barriers none --greedy-priority wavefront --greedy-split-shift --pressure-inflation-threshold 0.95`) the greedy gives 1,313 cycles (1,315 on the simulator), validated and passing `check_upstream.sh`; see [docs/GREEDY_SHAPE_ABLATION.md](docs/GREEDY_SHAPE_ABLATION.md).
- 4 rounds (development size, not comparable with the 16-round baseline): 485 cycles (CP-SAT 1800s, no VALU shift)
- Capacity floor for the ops we emit today (16 rounds): 1,271 cycles. This is a floor for this emission (9,526 vector-op equivalents), not a bound on the problem: public kernels emit fewer ops and reach 863. See [docs/BOUND_GAP_ANALYSIS.md](docs/BOUND_GAP_ANALYSIS.md).
- Target: 1,487 cycles (100x speedup)

## Default Problem Size

**Current development: use 2 rounds** for kernel generation, CP-SAT debugging, bulk-move, and iteration (e.g. `uv run generate_ssa_kernel.py --rounds 2`). The script default is still 16 for benchmarking.

Benchmark/default parameters:
- `forest_height=10` - Tree depth (levels 0..10, so 2^11 - 1 = 2047 nodes)
- `batch_size=256` - Number of parallel traversals
- `rounds=16` - Hash rounds per traversal (use `--rounds 2` for dev)

**Batch size constraint:** `batch_size` must be a positive multiple of VLEN (8), because the kernel processes vectors of 8 elements. Invalid values (e.g. 12, 20) produce wrong output or are rejected. Use e.g. 8, 16, 24, 32.

**Terminal / scripting:** Do not use `tail` (or similar) to truncate optimizer or build output. We should see the whole output—run commands without piping to `tail` so the full output is displayed. You may read only the last part of a file or buffer when needed, but present the full output (or the full relevant portion) to the user.

## Project skills

When working in this repo, read and follow these project-specific skills when they apply:

- **Inspect a schedule (what is at address A, which SSA op owns a slot)** — Read [.cursor/skills/use-inspect-schedule/SKILL.md](.cursor/skills/use-inspect-schedule/SKILL.md). `scripts/inspect_schedule.py` answers, lane-aware, from a scheduler JSON: `addr A [--cycle C]` (values covering A, who holds it, last write), `op ID` (operands, lowered ops incl. shifted lanes and select subtract/multiply_add, consumers), `slot CYCLE DEST` (SSA op behind a kernel slot), `value NAME`. Run as `uv run python -m scripts.inspect_schedule sched.json ...`.
- **Kernel dataflow replay (first step for any wrong kernel)** — Read [.cursor/skills/use-kernel-dataflow/SKILL.md](.cursor/skills/use-kernel-dataflow/SKILL.md). `verification/kernel_dataflow.py` executes the emitted kernel symbolically and proves it equals its SSA program for all inputs, or names the first cycle, slot and operand where it does not. Runs automatically under `--validate`; standalone: `uv run python -m verification.kernel_dataflow --kernel kern.json --scheduler sched.json [--explain-cycle N]`.
- **SSA equivalence (transformed emission == canonical emission)** — Read [.cursor/skills/use-ssa-equivalence/SKILL.md](.cursor/skills/use-ssa-equivalence/SKILL.md). `verification/ssa_equivalence.py` proves two SSA programs (scheduler JSONs) write the same final memory for every input satisfying the preconditions, or gives the first differing word with a concrete input. Use after adding or changing an emission rewrite or plan: `uv run python -m verification.ssa_equivalence default.json transformed.json [--values-only]`.
- **Reproducibility (same command, same kernel)** — Builds are deterministic: two runs with the same options hand CP-SAT byte-identical models and, with a `--cpsat-deterministic-time` budget that the solve ends on (not the wall-clock backstop), emit byte-identical kernels (`tests/test_determinism.py`). If two runs disagree, export each model with `CPSAT_EXPORT_MODEL=path.pb` and run `uv run python -m scripts.diff_cpsat_models a.pb b.pb`; it reports the first constraint, variable or hint that differs and whether it is only a reordering. `CPSAT_EXPORT_MODEL_DIR=dir` keeps every model of a run (per-section solves included). A change that claims not to alter the model (a refactor of `cpsat_scheduler/`) is checked with `scripts/cpsat_model_identity.py record` on the base and on the change, then `compare`: every model and kernel of a matrix of configurations must be identical. Times of an op (issue, visible, last read, finish) are defined once in `cpsat_scheduler/op_times.rs` (`OpTimes` for the model, `ScheduledTimes` for validators); constraint generators must not use the raw time variables. Memory edges (`SsaOp::mem_deps`) are sorted per op and built over a `BTreeMap`; keep it that way. New code that creates solver variables/constraints or allocates addresses must not iterate a `HashMap`/`HashSet` directly: use `sorted_by_key` (in `cpsat_scheduler/model.rs`), a `BTreeMap`/`BTreeSet`, or sort first.
- **Debugging makespan / empty cycles** — Read [.cursor/skills/use-bulk-move/SKILL.md](.cursor/skills/use-bulk-move/SKILL.md). Use the bulk-move simulator to see if empty cycles can be compacted (ReachedEnd) or which constraint blocks (Blocked). Requires CP-SAT run with `--output-json`; run `uv run scripts/simulate_bulk_move.py` on that JSON.
- **Schedule debug / per-decile slot utilization** — Read [.cursor/skills/use-schedule-debug/SKILL.md](.cursor/skills/use-schedule-debug/SKILL.md). Use per-decile slot utilization to find optimization opportunities (slot balance, encoding improvements). Run `uv run scripts/slot_usage_by_decile.py` on scheduler output JSON; see [docs/PER_DECILE_SLOT_UTILIZATION.md](docs/PER_DECILE_SLOT_UTILIZATION.md).
- **What-if shift (debug VALU shift decisions)** — Read [.cursor/skills/use-whatif-shift/SKILL.md](.cursor/skills/use-whatif-shift/SKILL.md). Force a specific VALU op to shift to ALU, re-optimize with CP-SAT, and optionally compare to baseline. Run `uv run scripts/what_if_shift.py --op-id <id> [--compare-with scheduler_output.json]`. The forced set (`forced_shifts=[op ids]` on the binding) is honoured by the greedy as well as by CP-SAT: a forced op takes 8 scalar slots on every greedy path, so the warm start, the ALAP cap and the horizon come from a schedule of the model being solved, and `use_cpsat=False` emits the forced shifts too. It applies only where the CP-SAT model applies it: shiftable vector ops, shifting enabled, `--valu-shift optimize`.
- **Bug tracking** — Read [.cursor/skills/use-bugs-md/SKILL.md](.cursor/skills/use-bugs-md/SKILL.md). Use BUGS.md at the repo root: read before fixing, update when fixing or reporting, keep entries short.
- **Bisecting validation failures / minimal repros** — Read [.cursor/skills/use-bisection-tools/SKILL.md](.cursor/skills/use-bisection-tools/SKILL.md). Use `scripts/bisect_cpsat_validation.py` to find which choice options (ConstChoice, SelectChoice, valu shift) correlate with failure, and `scripts/bisect_minimal_fma.py` for smallest rounds/batch/height that reproduces.
- **CP-SAT debug (INFEASIBLE, active set, budget exceeded)** — Read [.cursor/skills/use-cpsat-debug/SKILL.md](.cursor/skills/use-cpsat-debug/SKILL.md). Use `CPSAT_EXPORT_INFEASIBLE=1` to export model on INFEASIBLE; `scripts/inspect_cpsat_model.py` to inspect it; `scripts/inspect_binding_constraints.py` to see binding constraints from saved JSON. **Budget exceeded errors automatically export schedule JSON** to `scheduler_output_error.json` (override with `SCHEDULER_EXPORT_ON_ERROR_PATH`).
- **Validation / execution trace (where did the kernel go wrong?)** — Read [.cursor/skills/use-validation-trace/SKILL.md](.cursor/skills/use-validation-trace/SKILL.md). Use `scripts/debug_validation_trace.py` to see which batch indices and groups mismatch vs reference; use `scripts/trace_group20.py` to get the exact cycle and instruction that wrote a wrong value and trace producers. Use when a kernel runs but fails validation.
- **Compare kernel scratch (first divergence between two kernels)** — Read [.cursor/skills/use-compare-kernel-scratch/SKILL.md](.cursor/skills/use-compare-kernel-scratch/SKILL.md). Use `scripts/compare_kernel_scratch_cycle.py` to find the first cycle and scratch address where two kernels (e.g. passing vs failing) differ on the same input. Use when debugging validation failure between two kernels (e.g. zero-gap vs default).
- **Find wrong scratch use (root-cause wrong read address)** — Read [.cursor/skills/use-find-wrong-scratch-use/SKILL.md](.cursor/skills/use-find-wrong-scratch-use/SKILL.md). Use `scripts/find_wrong_scratch_use.py` on scheduler output JSON to find the first lowered op that reads from a scratch address not matching vreg_to_addr[SSA source]. Use when root-causing "consumer reads from wrong slot" (e.g. after compare_kernel_scratch_cycle gives divergent addresses).
- **Profile kernel generation** — Read [.cursor/skills/use-profile-kernel-generation/SKILL.md](.cursor/skills/use-profile-kernel-generation/SKILL.md). Use `scripts/profile_kernel_generation.sh` to run the generator with profile output paths (scheduler_output_profile.json, kernel_profile.json) or under `USE_PERF=1` for CPU profiling. Use when profiling generate_ssa_kernel.py without overwriting normal outputs.
- **Trace IDs (semantic debugging with trace_input/trace_output)** — Read [.cursor/skills/use-trace-ids/SKILL.md](.cursor/skills/use-trace-ids/SKILL.md). Use `--enable-trace-ids` to generate kernels with semantic trace points; run with `enable_debug=True` to capture `trace_io_log` and compare trace outputs to reference to pinpoint where values diverge. Use when you need to see exact values at semantic boundaries (e.g., hash outputs per round) or verify intermediate computations.
- **Hypothesis quantification (schedule optimization ideas)** — Read [.cursor/skills/use-schedule-hypotheses/SKILL.md](.cursor/skills/use-schedule-hypotheses/SKILL.md). Use `scripts/quantify_schedule_hypotheses.py` on scheduler output JSON to estimate impact of vbroadcast elimination, element-wise chaining, and ConstChoice; use A/B runs and bulk-move/what-if for VALU shift and precedence.
- **Ablation analysis (ASAP/ALAP tightening)** — Read [.cursor/skills/use-ablation-analysis/SKILL.md](.cursor/skills/use-ablation-analysis/SKILL.md). Run 4-arm ablation study (baseline, slot-aware, scratch-aware, both) with `scripts/run_asap_alap_ablation.py`; analyze results with `scripts/analyze_ablation_results.py`.
- **CP-SAT configuration ablation** — Read [.cursor/skills/use-cpsat-ablation/SKILL.md](.cursor/skills/use-cpsat-ablation/SKILL.md). Run `uv run python -m scripts.run_cpsat_ablation` to measure CP-SAT switches (workers, interleaved search, subsolvers, objective, greedy ALAP cap, horizon, makespan interval, cumulative propagators, redundant rows) over a set of instances with a deterministic-time budget; it validates every kernel and writes a JSON and Markdown results table. Results and the recommended configuration: [docs/CPSAT_CONFIG_ABLATION.md](docs/CPSAT_CONFIG_ABLATION.md).
- **Schedule report (how good is this schedule?)** — Read [.cursor/skills/use-schedule-report/SKILL.md](.cursor/skills/use-schedule-report/SKILL.md). `uv run python -m scripts.schedule_report sched.json` prints per-engine floors, the capacity floor for the ops this run emitted, the gap to it, per-round cycle spans and utilisation over time; `--compare` diffs two runs. The generator prints the floor and gap after every build. The floor (and `lb_cycles`) is for this emission only, never a bound on the problem.
- **Emission ablation (does an emission option remove ops?)** — Read [.cursor/skills/use-emission-ablation/SKILL.md](.cursor/skills/use-emission-ablation/SKILL.md). `uv run python -m scripts.emission_ablation [--rounds N] [--only a,b] [--upstream] [--markdown]` builds each emission configuration, requires PROVED and PASSED, and tabulates vector-op equivalents, loads, flow, the capacity floor (as encoded, and with the best division of selects between flow and the vector ALU) and cycles. Results and recommendation: [docs/EMISSION_ABLATION.md](docs/EMISSION_ABLATION.md).
- **Emission structure model (which encoding mix is best?)** — Read [.cursor/skills/use-emission-structure/SKILL.md](.cursor/skills/use-emission-structure/SKILL.md). `uv run python -m scripts.emission_structure solve --out plan.json` chooses the tree-access encoding (gather, bits, bits-flow, mobius) of every group in every round to minimise the capacity floor, over op counts calibrated exactly from the emitter (`calibrate`, `check`); `measure --plan plan.json --shaped --upstream` emits it (`--emission-plan`), proves, validates and schedules it. Design and results: [docs/EMISSION_STRUCTURE_MODEL.md](docs/EMISSION_STRUCTURE_MODEL.md).
- **Public kernel lift and reschedule (emission vs scheduling)** — Read [.cursor/skills/use-public-kernel-reschedule/SKILL.md](.cursor/skills/use-public-kernel-reschedule/SKILL.md). `scripts/capture_public_kernel.py` captures a third-party kernel builder's instructions in a sandbox; `scripts/kernel_lift.py` lifts any straight-line kernel into a dependency graph (exact scratch RAW/WAR/WAW per word, conservative Z3-derived memory edges from `verification/memory_alias.py` under Assumption A) and reports ops, floors, critical path and fill; `scripts/reschedule_kernel.py` reschedules the same ops with greedy or CP-SAT and validates on the simulator. Results in [docs/PUBLIC_KERNEL_RESCHEDULE.md](docs/PUBLIC_KERNEL_RESCHEDULE.md).
- **Window optimality probe (is the schedule locally optimal? windowed / rolling-horizon CP-SAT)** — Read [.cursor/skills/use-window-probe/SKILL.md](.cursor/skills/use-window-probe/SKILL.md). `scripts/window_optimality_probe.py` re-optimises windows of W cycles with the full CP-SAT model, everything outside pinned (ops after the window move as one block), and reports cycles saved per window, proven or bounded, with free vs fixed encodings, plus a rolling pass; results in [docs/WINDOW_OPTIMALITY_PROBE.md](docs/WINDOW_OPTIMALITY_PROBE.md). Off by default (`CPSAT_WINDOW_PROBE=spec.json`).
- **Round progress tracking** — Read [.cursor/skills/use-round-progress/SKILL.md](.cursor/skills/use-round-progress/SKILL.md). Run kernel generation with validation and update [docs/round-progress.md](docs/round-progress.md) to track SHA vs cycles over time. Use when benchmarking different round counts, batch sizes, or configurations.

## Build & Development Commands

### Python Setup
```bash
# Install dependencies (uses uv package manager)
uv pip install -e .

# Install with dev dependencies (wandb)
uv pip install -e ".[dev]"
```

### Rust Scheduler (scheduler_rs/)
**Always build from the project root** so the extension installs into the project venv (used by `uv run`). Building from inside `scheduler_rs/` installs into `scheduler_rs/.venv` and will fail with an explicit error.

**Why uv doesn't manage scheduler_rs:** uv and maturin integrate (PEP 517): uv uses maturin to build when installing a maturin package. Here we use a separate script because (1) scheduler_rs is not a dependency of the root project—the root `pyproject.toml` does not list it—so `uv sync` / `uv run` never install or rebuild it; (2) scheduler_rs's `build.rs` requires `BUILD_FROM_PROJECT_ROOT=1` and running from the project root (so the wheel installs into the project venv). When uv builds a path dependency it typically runs the build from the package directory, which would hit that check. So scheduler_rs is installed manually via the script below (or `./scripts/run_generate_ssa_kernel.sh` to auto-rebuild when Rust changes, then run the generator).
```bash
# From project root (required)
BUILD_FROM_PROJECT_ROOT=1 uv run maturin develop --manifest-path scheduler_rs/Cargo.toml

# Or use the helper script (from project root)
./scripts/build_scheduler_rs.sh

# Release build
BUILD_FROM_PROJECT_ROOT=1 uv run maturin build --release --manifest-path scheduler_rs/Cargo.toml
```

### Running Tests

**Testing strategy:** Heavy testing of **scheduler_rs** (scheduling, budget, CP-SAT, greedy, correctness) lives in **Rust** (`cargo test --manifest-path scheduler_rs/Cargo.toml --features cpsat`). Python tests for scheduler_rs **only exercise the PyO3 interface and its options** (e.g. `build_tick_tock_kernel_ssa`, `enable_valu_shift`, `use_cpsat`): they build via the binding → run on `Machine` → compare to `reference_kernel2`. They do **not** test underlying scheduler_rs functionality (that is for Rust tests). `tests/submission_tests.py` exercises the end-to-end pipeline.

**One entry point:** `./scripts/test.sh [rust|rust-shallow|python|python-shallow|python-deep|proofs|kernel|upstream|ci-only|all]` sets the environment each suite needs (OR-Tools prefix, `PYO3_PYTHON` pointing at the project venv, `BUILD_FROM_PROJECT_ROOT`). Arguments after the suite name are passed through.

**Machine load: test slots, and when to run suites locally.** Several agents in different worktrees share this machine; when they all run suites at once, timing-sensitive tests (pytest-timeout, subprocess timeouts, Z3 timeouts) fail for load alone. So:
- Every `./scripts/test.sh` suite holds one of a few machine-wide test slots while it runs (`scripts/test_slot.py`; an `flock` on files in `~/.cache/anthropic-perf-eval/test-slots`, released automatically when the run ends or is killed). `PERF_EVAL_TEST_SLOTS` sets how many (default 2; `0` disables); CI takes no slot. The wrapper `exec`s the suite with the lock, so the slot is held exactly while the suite (and anything it leaves running) lives. A run that has to wait prints who holds the slots.
- Run suites locally when you change code or are exploring. After a rebase or merge that only resolves conflicts (your commits' changes are the same, checked with `git range-diff`), do not rerun the full suites locally: push, let CI run them, and quote the range-diff.
- A local failure that is a timeout under load (`uptime` load average far above the core count) is not a test failure; rerun it later or rely on CI, and say so.

```bash
./scripts/test.sh                 # everything
./scripts/test.sh rust            # cargo test --features "cpsat deep-tests" (all Rust tests)
./scripts/test.sh rust-shallow    # without deep-tests: the slow CP-SAT tests are #[ignore]d (PR CI)
./scripts/test.sh rust -- test_vector_group      # one Rust test
./scripts/test.sh python          # pytest: every tests/test_*.py (discovery) + solver/test_slot_utilization.py
./scripts/test.sh python tests/test_kernel_dataflow.py -k lane
./scripts/test.sh python-shallow  # without @pytest.mark.deep tests (PR CI)
./scripts/test.sh python-deep     # only @pytest.mark.deep tests (deep CI)
./scripts/test.sh proofs          # Z3: liveness theorems + rewrite identities
./scripts/test.sh kernel          # greedy 2r and 16r builds with dataflow replay + validation
./scripts/test.sh upstream        # 16-round greedy kernel on the unmodified upstream simulator and tests
./scripts/test.sh ci-only         # pytest -m ci_only: 16-round pin builds (CI step; not part of 'all')

# Rust: include optional/CI-only tests (slow proptest, minimal failing config, etc.)
./scripts/test.sh rust --features "cpsat ci-tests"

# Python: submission tests (end-to-end pipeline)
uv run tests/submission_tests.py

# Upstream check: 16-round kernel on the unmodified upstream simulator and tests (vendored in upstream/)
./scripts/check_upstream.sh                 # greedy smoke check; pass generate_ssa_kernel.py options for CP-SAT
./scripts/check_upstream.sh --kernel kernel_to_validate.json   # test a saved kernel
```

`uv run pytest` works directly: pytest, pytest-timeout and z3-solver are in the default `dev` dependency group. Running `cargo test` by hand needs `PYO3_PYTHON=$PWD/.venv/bin/python` if the system Python is newer than PyO3 supports ("the configured Python version is newer than PyO3's maximum supported version").

**Test tiers (shallow vs deep):** a test that runs a CP-SAT solve is *deep*. In Python mark it `@pytest.mark.deep` (or mark only its CP-SAT parameter: `pytest.param("cpsat", marks=pytest.mark.deep)`, also on fixture `params`). In Rust, if it takes more than a few seconds in CI, add `#[cfg_attr(not(feature = "deep-tests"), ignore = "...")]` under `#[test]`. Z3 proofs and greedy builds stay shallow: they take milliseconds to seconds. Two cheap 2-round CP-SAT solves stay shallow on purpose, so every PR exercises the Python → binding → CP-SAT path: `tests/test_kernel_dataflow.py::test_clean_cpsat_kernel_is_proved` (in-process build, kernel proved against its SSA program) and `tests/test_cpsat_hint_complete.py::test_greedy_hint_is_complete[rounds 2]` (CLI build, the greedy hint is complete and CP-SAT accepts it). Keep at least these; `rust-shallow` also keeps every Rust CP-SAT test except the six behind `deep-tests`. The plain suites (`rust`, `python`) run both tiers, so local runs cover everything; PR CI runs the `-shallow` suites and the deep tier runs in `ci-deep.yml`. CI prints `--durations=30`: a slow test in the shallow tier's list is a candidate for `deep`.

**Optional tests:** Prefer a **Cargo feature** (`ci-tests`) for optional/CI-only Rust tests so they run when requested (e.g. `cargo test --features "cpsat ci-tests"`). Use `#[ignore]` only for temporarily disabled tests that expose known bugs (with a comment and issue reference). CI should run with `--features "cpsat ci-tests"` for full coverage. For Python, mark the test `@pytest.mark.ci_only`: `pyproject.toml` deselects it in every pytest run (`addopts = "-m 'not ci_only'"`), and CI runs them in their own step, `./scripts/test.sh ci-only` (`pytest -m ci_only`; a later `-m` replaces the default).

### CI

GitHub Actions ([.github/workflows/ci.yml](.github/workflows/ci.yml)) runs on every pull request and on pushes to `main`, on Linux (`ubuntu-24.04`, x86_64; Linux runners queue far less than macOS ones). OR-Tools comes from Google's prebuilt C++ archive (it bundles protobuf and abseil), installed and cached by [.github/actions/setup-ortools](.github/actions/setup-ortools/action.yml), which pins the version and checksum. Two jobs:

- **Rust tests (scheduler_rs)** — `./scripts/test.sh rust-shallow` (debug profile, `--features cpsat`).
- **Python tests, proofs, kernel builds** — `./scripts/build_scheduler_rs.sh --release`, then `./scripts/test.sh python-shallow`, `./scripts/test.sh proofs`, `./scripts/test.sh kernel`, `./scripts/test.sh upstream`, `./scripts/test.sh ci-only`, one step each. The python suite collects `tests/` by discovery, so every `tests/test_*.py` runs in CI without being listed; `tests/test_suite_coverage.py` fails on a test file outside `tests/` that the suite does not run.

**Deep tier** ([.github/workflows/ci-deep.yml](.github/workflows/ci-deep.yml)): `./scripts/test.sh rust` (with `deep-tests`) and `./scripts/test.sh python-deep`. Runner minutes are rationed, so it runs daily on `main` (06:00 UTC). If `main` has not moved since the last finished deep run, it repeats that run's result without running the deep jobs: green with the jobs skipped (the run summary links the run that tested the commit) if it passed, red if it failed, by hand (Actions → CI (deep) → Run workflow), and on PRs labelled `deep-ci`. A PR without the label starts no deep job. Add the label to a PR that changes the CP-SAT model, its tuning, hints, or anything else the deep tests guard; otherwise a deep failure shows up only in the next daily run.

To reproduce a failure locally, run the suite named in the failing step: `./scripts/test.sh <suite>` (rebuild first with `./scripts/build_scheduler_rs.sh --release` for the Python-side suites). CI differs from local development in platform (Linux x86_64, not macOS arm64) and in the OR-Tools build (the pinned archive in setup-ortools, not Homebrew's); if a failure does not reproduce locally, suspect those first, especially for tests that pin CP-SAT results.

Not run in CI: the `ci-tests` Cargo feature, `tests/submission_tests.py`, and any full CP-SAT kernel build (the kernel suite is greedy-only).

### Generating Kernels

**Important: Always use `uv run` to execute Python scripts** to ensure the correct Python environment and dependencies.

**Rebuilding scheduler_rs:** `uv run generate_ssa_kernel.py` does **not** rebuild the Rust extension. After changing Rust code, run `./scripts/build_scheduler_rs.sh` first, or use `./scripts/run_generate_ssa_kernel.sh` which rebuilds scheduler_rs when Rust sources are newer than the installed extension, then runs the generator.

**Recommended: SSA builder with CP-SAT** (best quality):
```bash
# Default: 1 hour CP-SAT, VALU shift enabled
# Shows full optimization progress (solver stats, cycle improvements)
uv run generate_ssa_kernel.py

# With validation against reference kernel
uv run generate_ssa_kernel.py --validate

# Quick dev iteration (2 rounds, greedy, ~2 seconds, may exceed budget)
uv run generate_ssa_kernel.py --rounds 2 --use-greedy

# 4-round with CP-SAT and validation (30 min, good for development)
uv run generate_ssa_kernel.py --rounds 4 --cpsat-time-limit 1800 --validate

# Auto-rebuild scheduler_rs if Rust changed, then run
./scripts/run_generate_ssa_kernel.sh --cpsat-time-limit 7200 --validate --rounds 4

# Output schedule JSON for analysis (default: scheduler_output.json)
uv run generate_ssa_kernel.py --output-json scheduler_output.json
```

**Scheduler output JSON (`--output-json`, e.g. `scheduler_output.json`):** When summarizing or analyzing a run, **always prefer using the scheduler output JSON** when available. It contains summarizable information (per-cycle slot usage, liveness curve, VALU shift debug, usage accounting, etc.) that supports slot usage over time, decile analysis, and other reports without re-deriving from the kernel alone. Generate it with `--output-json scheduler_output.json` (or another path); then use that file for any summarization or analysis tasks.

**CP-SAT solver output:** The script shows full CP-SAT progress by default. **Always report** (and ensure the generator prints):
- **CP-SAT choices:** ConstChoice (AddImm vs Load counts), SelectChoice (FMA vs vselect counts), and how many VALU ops were shifted. This explains what the solver chose for optional encodings.
- **Slot usage:** For each engine (ALU, VALU, Load, Store, Flow), report used slots, total available slots, and utilization percentage (e.g. `ALU=1774/3696 (48.0%)`). This shows how well the schedule uses each slot type.
- **VALU breakdown:** Always summarize VALU slot usage by source: FMA (SelectChoice FMA path; not shiftable), vbroadcast (not shiftable), and other (shiftable VALU ops). This clarifies how much of VALU is fixed vs optimizable via shifting.
- Kernel building phase (SSA ops, dead code elimination), solver progress (objective bounds, conflicts, branches), and final statistics (cycles, speedup, scratch usage).

**Options for `generate_ssa_kernel.py`:**

Scheduler options:
- `--use-greedy` - Use greedy scheduler instead of CP-SAT (fast, ~2s, may exceed budget)
- `--cpsat-time-limit SECONDS` - CP-SAT wall-clock time limit (default: 3600 = 1 hour). How far the solver gets in a wall-clock limit depends on machine load, so two runs can return different schedules.
- `--cpsat-deterministic-time DTIME` - CP-SAT deterministic-time budget per solve (the solver's own work counter). **Use this for any comparison between runs or configurations:** the same command then produces the same kernel regardless of load, provided the solve ends on this budget (or on optimality). The wall-clock limit stays as a backstop; if it fires first the result depends on load again and a `WARNING: CP-SAT stopped on the wall-clock limit` line is printed, so keep `--cpsat-time-limit` generous. A few units is seconds of solving on small kernels; solver logs print `deterministic_time` for calibration.
- `--enable-valu-shift` / `--no-valu-shift` - VALU→ALU shifting (enabled by default)
- `--tree-reduce-max-depth N` - Max tree depth for tree-reduce selection, 2 or 3 (default: 3)
- `--const-choice MODE` - ConstChoice handling: `optimize` (default), `add_imm`, or `load`
- `--select-choice MODE` - SelectChoice handling: `optimize` (default), `fma`, or `vselect`
- `--valu-shift MODE` - VALU shift handling: `optimize` (default), `alu`, or `valu`. Under `--use-greedy`, `alu` puts every shiftable vector op on the scalar ALU, about one per cycle: 7,148 cycles at 16 rounds against 2,007 (a warning is printed); `valu` gives 2,104.
- `--per-section-optimize-secs SECONDS` - Per-section CP-SAT optimization time (0 or omit = feasibility only)
- `--force-zero-gap` - Force 0 integral gap: prove the schedule optimal for the emitted ops and encoding choices, not for the problem (auto uses 2h time limit)
- `--asap-alap-tightening {none,simple,symmetry-aware,energetic}` - ASAP/ALAP tightening mode (default: energetic). `energetic` is symmetry-aware plus resource-aware earliest and latest issue cycles: everything an op depends on, and everything that depends on it, needs slots (`cpsat_scheduler/resource_bounds.rs`). The resource-aware pass excludes no schedule of the model (Z3 checks `RB*` in `verification/resource_bounds.py`); the counting rule it replaced did. The symmetry-aware part of `energetic` is valid only together with the declared ordering rows (themselves not justified by a proven symmetry, see BUGS.md) and is skipped when those rows are not in the model (`--cpsat-symmetry-rows none` or `checked`). Design, literature and measurements: [docs/ENERGETIC_REASONING_FOR_UNIT_TASKS.md](docs/ENERGETIC_REASONING_FOR_UNIT_TASKS.md); older rules: [docs/ASAP_ALAP_SLOT_TIGHTENING.md](docs/ASAP_ALAP_SLOT_TIGHTENING.md).
- `--enable-scratch-tightening` - Non-default debugging: scratch-aware domain tightening (default: disabled)
- `--bulk-move` - Enable bulk-move compaction after CP-SAT to eliminate empty cycles. **Off by default:** CP-SAT should produce schedules without empty cycles; bulk-move merely masks solver/model bugs rather than fixing them, and is buggy (can cause same-cycle scratch write conflicts). Use only for experimenting; do not rely on it for production.
- `--hint-passes` - Enable all three greedy hint passes (ConstChoice, SelectChoice, VALU shift). Off by default. See [docs/GREEDY_HINT_PASSES_ABLATION.md](docs/GREEDY_HINT_PASSES_ABLATION.md).
- `--const-choice-hint-pass`, `--select-choice-hint-pass`, `--valu-shift-hint-pass` - Enable hint passes individually.
- `--schedule-with-fixed-choices` - After hint passes, re-run greedy with encodings fixed to assign cycles.
- `--scratch-model {words,pools}` - What the schedulers' scratch budget bounds. `words` (default): live words at every point, which does not guarantee that addresses fit (for 1- and 8-word values a per-point fit does not imply a layout exists; docs/ALLOCATION_CONJECTURES.md). `pools`: `8 * (most vectors live at once) + (most scalars live at once) <= budget`, which the two-pass address allocator always fits. Greedy: keeps the words schedule when it already meets the condition, otherwise reschedules with fixed vector/scalar capacities from the first run. CP-SAT: a vector cumulative and a scalar cumulative with capacities `N_v`, `N_s`, `8*N_v + N_s <= budget`. Off by default.

Greedy schedule shape options (ablation switches, defaults unchanged; measurements, sweeps and the recommended combination in [docs/GREEDY_SHAPE_ABLATION.md](docs/GREEDY_SHAPE_ABLATION.md)):
- `--greedy-barriers {round,none}` - Round barriers in the greedy scheduler. `round` (default): every op of round r+1 waits for all of round r, so rounds run strictly one after another. `none`: the barrier pseudo-ops are dropped before dependencies are computed, the ready queue may mix rounds, and the kernel is a single section for the per-section budget check. On its own this does not help at 16 rounds (the ready queue stays round-major and the greedy exceeds the scratch budget); it is the prerequisite for the other shape options.
  - The option also applies when CP-SAT follows the greedy (no `--use-greedy`). The CP-SAT model itself never contains barriers (they are always removed before it is built); what changes is the greedy schedule it gets as its warm start, which also caps every op's latest cycle (greedy cycle + 15) and sets the horizon, and the per-section repair, which then sees one section instead of one per round.
- `--greedy-priority {fifo,wavefront}` - Order in which the greedy tries ready ops. `fifo` (default): by ready cycle, then emission order, which keeps all 32 groups in the same round. `wavefront`: key = dependency depth + stagger × group rank, so group g runs as if it started `stagger × g` cycles after group 0 and some groups gather while others hash or select (see `scheduler_rs/src/kernel_builder/ssa_greedy/wavefront.rs`; groups are found structurally: an op that reaches more final stores than the median op does is shared, and groups are the connected components of the rest). Use with `--greedy-barriers none`: with barriers every op reaches every store through the barrier chain, no groups are found, the key is plain dependency depth, and the schedule is longer than the default at every size (16 rounds: 2,150 cycles).
  - `--greedy-stagger-cycles N` - Stagger between consecutive groups in cycles (default 12, about three quarters of one round's critical path; 0 = all groups abreast).
  - `--greedy-groups-in-flight N` - Groups allowed to run at once (default 16; 0 = no limit). Bounds scratch: a group that has not started holds nothing, a finished group has stored its results.
  - 16 rounds, greedy: `uv run generate_ssa_kernel.py --rounds 16 --use-greedy --validate --greedy-barriers none --greedy-priority wavefront` gives 1,398 cycles (1,400 on the simulator) against 2,007 (1,994) by default.
- `--greedy-split-shift` - Greedy-only. A vector op run as 8 scalar ops may straddle two consecutive cycles (first lanes in the leftover scalar slots of one cycle, the rest in the next), so all 12 scalar slots carry shifted work: 3 shifted ops per 2 cycles instead of 2, i.e. 7.5 vector-op equivalents per cycle instead of 7. The op's cycle is its last lane's; per-lane cycles go to `shifted_alu_cycles` as for CP-SAT. Ignored when CP-SAT follows unless `--cpsat-split-shift-hint` is given: that keeps the greedy's split shifts and hints CP-SAT each lane at its own cycle (`SsaCpsatHints::shifted_lane_cycle_hints`), so the solver starts from the split-shift schedule. With the wavefront priority at 16 rounds: 1,314 cycles (1,316 on the simulator), 43 above the 1,271 floor.
- `--greedy-justify N` - Forward/backward justification of the finished greedy schedule, at most N iterations (default 0 = off): a backward list-scheduling pass on the reversed graph, then a forward pass in the order of its start times; only cycles and shift decisions change. Every candidate is checked (slot limits, precedences, memory edges, scratch) and dropped if it fails, so the result is the greedy's schedule or a shorter valid one. The search also stops after 10 iterations in a row without a shorter accepted schedule (a bound for large N only). With the shape options above at 16 rounds: 1,313 -> 1,279 cycles (1,281 on the simulator), validated and passing `check_upstream.sh --strict`. Not pressure-aware: from the default greedy shape every candidate is rejected for scratch. Also applies when CP-SAT follows the greedy: the justified schedule is the hint (it must then keep every op in its ASAP/ALAP domain); measured only at 2 rounds / batch 16 and 4 rounds / batch 64. See [docs/GREEDY_SHAPE_ABLATION.md](docs/GREEDY_SHAPE_ABLATION.md) section 13.
- `--greedy-load-priority {key,tail,final-last}` - Wavefront only (an error otherwise). Which ready loads get the two load slots first; other engines keep the wavefront order. `key` (default): the wavefront key. `tail`: longest dependency path to the end of the program first. `final-last`: loads that no later load depends on (the last gather round) only after every other ready load; only the first placement pass of a cycle is ordered. On `docs/emission_plans/flow90.json` with the shape options, `--greedy-groups-in-flight 32 --greedy-stagger-cycles 9` and `final-last` take 16 rounds from 1,104 to 1,039 cycles (an interaction: see docs/GREEDY_SHAPE_ABLATION.md section 11).
- `--greedy-flow-priority {key,feeds-load-first,chain-first}` - Wavefront only (an error otherwise). Which ready flow op gets the one flow slot: `key` (default) the wavefront key; `feeds-load-first` flow ops that a non-final load depends on (negative on the measured plans, docs/GREEDY_SHAPE_ABLATION.md section 12); `chain-first` flow ops by group (chain) rank, then key, so the first groups' early-round selects are not overtaken (measured with `--greedy-alu-priority native-first`, below).
- `--greedy-select-overflow {never,valu}` - `valu`: a SelectChoice with a precomputed delta (one `multiply_add`) runs on the vector ALU when the flow slot is taken and a vector slot is free; the greedy's live-word counter holds its delta and operands to cycle + 1 as address assignment does. An error together with `--select-choice fma|vselect`, a select-choice hint pass or `--schedule-with-fixed-choices` (they own the select encoding). Not a default win (section 12.1).
- `--greedy-select-overflow-limit N` - With `--greedy-select-overflow valu` only (an error otherwise): at most N selects take the multiply-add form.
- `--greedy-shift-policy {when-full,reserve,pace,reserve-pace}` - When the greedy runs a shiftable vector op as 8 scalar ops (only when it decides shifts itself). `when-full` (default): only when the six vector slots are full. `reserve`: also when the free vector slots are wanted by non-shiftable vector ops later in the cycle's ready order. `pace`: also whenever the run is behind the pace toward `--greedy-shift-target`. `reserve-pace`: either. Measured on flow90 (section 11): none beats `when-full` with the best shape (`reserve` 1,049, `pace` 1,078 against 1,039).
- `--greedy-shift-target N` - Pace target for `pace` / `reserve-pace`: vector ops to run as 8 scalar ops over the schedule (default: the balanced split `(2V - A) / 10` of the emitted counts, which equalises the vector and scalar ALU floors).
- `--pressure-inflation-threshold F`, `--pressure-soft-threshold F`, `--pressure-critical-threshold F` - Override the pressure policy's thresholds (fractions of the scratch budget; built-in 0.80 / 0.85 / 0.95, tuned before the greedy's live-word counter was exact). The inflation threshold is the one that matters: above it, producers with slack are deferred by up to 10 cycles. `--pressure-inflation-threshold 0.95` shortens the default greedy schedule (3 rounds 326 to 289, 4 rounds 456 to 437, 16 rounds 2,007 to 1,965) and makes the wavefront priority insensitive to `--greedy-groups-in-flight`; 0.98 and above fail the scratch budget at 3 and 16 rounds. The soft and critical thresholds changed nothing measurable.
- `--greedy-budget-force {same-cycle,stall}` - Scratch model `words`: when an op the greedy's budget check deferred is forced past the budget. `same-cycle` (default) forces one budget-deferred op per cycle whenever it has a slot. Reaching the budget is not itself a failure (later placements can release enough for their results), but a forced op takes the schedule past the budget, and then address assignment can fail; only the pressure policy keeps default runs from forcing. `stall` forces only in a cycle where nothing else was placed (as `pools` does), but it forces in any empty cycle, including one that ASAP inflation emptied by deferring the waiting ops, and can then stay over budget: 3 of 28 path-bit loop plans under `stall` + 0.98 did (docs/STARTUP_AND_STALL_LOOP.md), all 32 groups in flight; with 20 or 24 in flight they build; docs/STALL_DEADLOCK.md, `SCHEDULER_DEADLOCK_DUMP` + `scripts/stall_deadlock_diagnosis.py`). With `--pressure-inflation-threshold 0.98` it takes path-bit plans that sat at a 1,505-word peak from 1,099-1,142 to 1,049-1,064 cycles; the best plans (1,033, 1,039) are unchanged. See [docs/SCRATCH_CEILING_DIAGNOSIS.md](docs/SCRATCH_CEILING_DIAGNOSIS.md).
- `--greedy-alu-priority {key,native-first}` - Wavefront only. `native-first`: native scalar-ALU ops (address adds) are tried before the cycle's vector ops (lanes of a split shift carried from the previous cycle are still placed first and can take scalar slots). `--greedy-flow-priority chain-first`: flow ops by group (chain) rank, then key, so the first groups' early-round selects are not overtaken. Together on `path_flow90.json` (path bits): 1,033 -> **1,030** (1,032 simulated; PROVED, PASSED, upstream `--strict`); `flow90.json` (index bits): chain-first is worse (1,065), native-first alone changes nothing. See [docs/STARTUP_AND_STALL_LOOP.md](docs/STARTUP_AND_STALL_LOOP.md).
- `--greedy-const-policy {default,alu-first,alu-when-load-full}` - Greedy-only: where a ConstChoice is materialised. `alu-first` derives it on the scalar ALU (emitter alternatives, else `a + b` of two live constants) from producers scheduled earlier and still live. Moves 23 of 35 load-engine constants on `path_flow90.json`; without justification it changes no cycle count (1,030 / 1,039). Justification models derived constants since #93 (it skipped them before): `alu-first` + `--greedy-justify 10` gives 1,016, the same as without `alu-first`, and #91 records 1,026 for `alu-first` + `--reassociate-gather-xor` as a fragile interaction. `--greedy-early-stagger K:S` - Wavefront: the first K groups S cycles apart. No setting beats the default. `--const-choice`, `--select-choice`, `--valu-shift` now also take effect under `--use-greedy` (they were ignored). See [docs/GREEDY_CONST_CHOICE.md](docs/GREEDY_CONST_CHOICE.md).
- `--greedy-budget-headroom WORDS` - Scratch words the greedy keeps free below the budget (default 0). It scales the thresholds down with the budget; it does not replace the inflation threshold (with inflation off the greedy deadlocks at the budget and force-schedules past it whatever the headroom).

Relaxed-scratch reference (analysis only, off by default; see [docs/RELAXED_SCRATCH_REFERENCE.md](docs/RELAXED_SCRATCH_REFERENCE.md)):
- `--scratch-budget-override WORDS` - Scratch words the greedy, the per-section repair, CP-SAT and address assignment may use instead of 1,536.
- `--no-scratch-constraint` - Drop the CP-SAT scratch cumulative; the greedy and the allocator get 1,048,576 words unless `--scratch-budget-override` is given.
- Above 1,536 words (or with no constraint) the kernel does not run on the machine: the generator prints a banner, writes `kernel_relaxed_scratch.json` / `scheduler_output_relaxed_scratch.json` (marked `relaxed_scratch` / `scratch_relaxed`), refuses the normal kernel paths, and validates on a simulator with that much scratch. `check_upstream.sh` refuses such kernels. Never record these as results.

Output options:
- `--output-json PATH` - Write scheduler output JSON (default: `scheduler_output.json`). Use this file for summarization and slot/liveness analysis; see "Scheduler output JSON" above.
- `--validate` - Validate against reference kernel (final values and final indices; values only with `--values-only`). **The kernel is always saved before validation** (to `kernel_to_validate.json` unless `--save-kernel PATH` is set) so failed runs can be analyzed or re-validated without rebuilding.
- `--no-dataflow-check` - With `--validate`, skip the symbolic kernel-vs-SSA replay that runs before numeric validation.
- `--validate-only PATH` - Load kernel from JSON and run validation only (no build).
- `--save-kernel PATH` - Save kernel to this path (default when `--validate`: `kernel_to_validate.json`).
- `--liveness-detail LEVEL` - Liveness detail in JSON: `none`, `summary`, or `full` (default)
- `--comment TEXT` - Comment stored in schedule JSON
- `--seed N` - Random seed for validation (default: 123)
- `-q, --quiet` - Suppress verbose Rust scheduler output

CP-SAT scratch accounting (full-kernel and per-section solves, recorded under `run_options.cpsat_scratch_time`):
- `--cpsat-scratch-time {phase,cycle}` - `phase` (default) lets address assignment reuse a scratch word in the cycle of its last read. A value written and read within the section occupies `[def, last_read)` on the write-slot timeline (proof A8 in verification/allocation_conjectures.py). Live-ins also occupy the initial read phase, and live-through values retain their words through the section's final write. `cycle` counts through the last read cycle, adding 46–80 words on measured schedules. Compare the timelines with `uv run python -m scripts.scratch_time_overcount sched.json` (requires full liveness detail, the default).

CP-SAT tuning switches (full-kernel solve only; per-section feasibility solves are unaffected). Each is off / unchanged by default, independent of the others, and recorded with its effective value under `run_options.cpsat_tuning` in the scheduler JSON. They travel to Rust as one JSON object (`cpsat_tuning_json` → `CpsatTuning` in `scheduler_rs/src/cpsat_scheduler/config.rs`, which owns the defaults):
- `--cpsat-workers N` - `num_workers` (default: 8).
- `--cpsat-interleave` / `--no-cpsat-interleave` - `interleave_search` (default: on). On = deterministic round-robin of subsolvers: the same command gives the same kernel provided the solve ends on the `--cpsat-deterministic-time` budget or on optimality (if the wall-clock limit fires first, the result depends on machine load, as above); OR-Tools 9.15 then starts no probing, shaving, `lb_tree_search` or `objective_lb_search` worker. **`--no-cpsat-interleave` makes the result non-reproducible** (ordinary parallel portfolio): report the spread over several runs, never a single run.
- `--cpsat-subsolvers NAME,...` / `--cpsat-extra-subsolvers NAME,...` / `--cpsat-ignore-subsolvers NAME,...` - replace, extend or prune CP-SAT's subsolver list (e.g. `--cpsat-extra-subsolvers objective_lb_search,objective_shaving`). The solver log lists the subsolvers that actually run ("N full problem subsolvers: [...]").
- `--cpsat-objective {makespan+tiebreak,makespan}` - Objective (default: `makespan+tiebreak` = `scale * makespan + sum of op completion cycles`). `makespan` drops the tie-breaker and its auxiliary "chosen cycle" variables: the solve stops once the makespan is proved, and the returned schedule is any schedule of that makespan rather than the left-packed one (peak liveness can differ). No constraint depends on the tie-breaker: the scratch model's slack variables (`shifted_dest_start`, `completion`) can only make a value's modelled lifetime longer than its real one, so a makespan-only solution is never rejected by address assignment for that reason.
- `--cpsat-greedy-alap-slack K|none` - Every op's latest cycle is capped at its greedy cycle + K (default: 15). This confines the model to a neighbourhood of the greedy schedule: "OPTIMAL" and the bound are relative to it. `none` removes the cap (ALAP then comes from the horizon).
- `--cpsat-horizon-max-latency` / `--no-cpsat-horizon-max-latency` - The horizon (`greedy makespan + 20`) is raised to the max-latency critical path (every shiftable op at its worst-case latency) by default; that path is several times the makespan (183 against 43 at 2 rounds / batch 16). `--no-cpsat-horizon-max-latency` drops the guard. Use it together with `--no-cpsat-horizon-num-ops` for a horizon of exactly `greedy makespan + 20`.
- `--cpsat-horizon-num-ops` / `--no-cpsat-horizon-num-ops` - The horizon is `greedy makespan + 20`, raised to the number of ops by default. The op count (1,543 against a makespan of about 194 at 2 rounds) only widens the domains of auxiliary variables, and of ALAP when the cap is off. `--no-cpsat-horizon-num-ops` keeps the tight horizon; it relies on the greedy (plus per-section repair) schedule being feasible.
- `--cpsat-makespan-interval` - Add one interval `[makespan, horizon)` with demand = capacity to each engine cumulative (ALU, VALU, Load, Store, Flow), the pattern of OR-Tools' `rcpsp_sat.py`, so CP-SAT's LP relaxation and energy cuts see the makespan. Excludes no schedule. Off by default.
- `--cpsat-push-tasks-toward-start` - CP-SAT `push_all_tasks_toward_start` (experimental solver parameter, also from `rcpsp_sat.py`). Off by default.
- `--cpsat-cumulative-propagators NAME,...` - Extra cumulative propagators for every worker: `all`, or any of `overload_checker` (`use_overload_checker_in_cumulative`), `timetable_edge_finding` (`use_timetable_edge_finding_in_cumulative`), `hard_precedences` (`use_hard_precedences_in_cumulative` together with `exploit_all_precedences`; OR-Tools 9.15 builds the precedence list only when both are set). Default: none.
- `--cpsat-ttef-max-intervals N` - `max_num_intervals_for_timetable_edge_finding` (CP-SAT default 100). Timetable edge finding is skipped on cumulatives with more intervals, which is every engine cumulative here, so raise it together with `timetable_edge_finding`.
- `--cpsat-lane-model {free,sorted,pinned,sorted-pinned}` - How the 8 lane variables of a shiftable op's scalar alternative are modelled (default: `free`, today's model). `sorted` adds `lane_0 <= ... <= lane_7` per op (removes the 8! lane permutations); `pinned` sets an unshifted op's lanes to its cycle and states each lane precedence as one unconditional row instead of two rows enforced by the producer's shift literal; `sorted-pinned` does both. Neither loses a schedule: Z3 proofs per row shape in `verification/lane_model.py` (`./scripts/test.sh proofs`), and `tests/test_cpsat_lane_model.py` checks that every row of a real model that mentions a lane has a proved shape. A future row that ties lane i of a reader to a lane-specific producer (e.g. the loads of a gather) would make per-op sorting invalid; that test then fails. Validity argument and measurements: [docs/CPSAT_LANE_MODEL.md](docs/CPSAT_LANE_MODEL.md).
- `--cpsat-linearization-level {0,1,2}` / `--cpsat-probing-level {0..3}` - CP-SAT `linearization_level` and `cp_model_probing_level` (default: 2 and 2). Level 1 against 2 changed nothing measurable at 4x64 and 6x64 (most LP subsolvers set their own level); see [docs/CPSAT_CONFIG_ABLATION.md](docs/CPSAT_CONFIG_ABLATION.md) section 9.
- `--cpsat-makespan-rows {none,static,bucketed}` - Extra makespan lower-bound rows from static longest-path heads and tails (`cpsat_scheduler/makespan_rows.rs`, soundness proof `verification/makespan_rows.py`, default: none). `static`: one constant row (the head/tail bound, printed as "Static head/tail bound") plus per-engine head/tail window rows whose work depends on the shift, select and const literals. `bucketed`: also prefix-end and suffix-start variables per 32 cycles of static head, with energy rows. Raised the presolve bound from 130 to 152 at 6x64; section 9 of the ablation doc.
- `--cpsat-symmetry-rows {declared,none,checked}` - Symmetry-breaking ordering rows (default: `declared`, the `sym_ord`/`sym_lex` rows emission declares, which are not justified by a proven symmetry; see BUGS.md). `none`: no ordering rows. `checked`: vector groups that a mechanical isomorphism check proves identical (`cpsat_scheduler/group_symmetry.rs`) get one anchor op each, ordered in a chain per orbit; corresponding ops get the same domain, and the declared ordering is not used for tightening either. A pinned window probe can distinguish otherwise interchangeable groups, so it skips the checked anchor rows and domain unions. Per-section solves retain their existing defaults. Proof: `verification/group_symmetry.py`.
- `--disable-cpsat-constraints GROUP,...` - Leave out redundant constraint groups (recorded under `run_options.disabled_cpsat_constraints`). Besides the capacity rows (`redundant_linear_bounds`, `capacity_valu`, `capacity_alu`, `capacity_flow`, `capacity_load`) there are two groups that only remove duplicates: `conditional_makespan` (the per-op `makespan >= cycle + 1` rows, implied by `makespan = max_cycle + 1` and `max_cycle >= ...`; about 1.5 rows per op, most of them enforced) and `flow_cumulative` (the capacity-1 Flow cumulative, which repeats `no_overlap_flow`). All groups are on by default.

Emission options (change which ops the kernel contains; all off by default, each backed by a Z3 proof in `verification/rewrites.py`; measurements in [docs/EMISSION_ABLATION.md](docs/EMISSION_ABLATION.md)):
- `--merge-hash-stages` - Hash stages 2 and 3 as two multiply_adds and an XOR: 10 vector ops per hash instead of 11 (proof `R8-hash-stage-merge`).
- `--index-form {zero-based,mirrored}` - Tree index representation between rounds (default: `zero-based`). `mirrored` tracks `n = L_d - idx` (position in the level from the right, `L_d = 2^(d+1) - 2`): index update in 2 vector ops instead of 3: 1 at depth 0 (the AND; `n` is the branch bit itself, so no multiply_add) and none in the wrap round. The initial indices are assumed 0 (as the root broadcast in round 0 already assumes) and not loaded; the final indices are still stored and validated (proofs `R9` to `R11`).
- `--prune-dead-index` - Zero-based index form only: do not emit index work whose result is known. At depth 0 every index is 0, so the update needs no multiply_add and the initial indices are not loaded; the wrap round shares one zero vector. Output unchanged (proof `R12-index-update-from-root`). No effect with `--index-form mirrored`, which has none of this work.
- `--values-only` - **Changes the output:** the final indices are not stored, only the values (all the upstream tests compare). The last round's index update and everything else that only fed the index store is then removed by DCE. `--validate` compares values only under this flag and says so; the saved kernel JSON records `values_only` so `--validate-only` does the same.
- `--cached-select {standard,bits,bits-flow,mobius}` - Node selection on the cached tree levels (depth 0 up to `--tree-reduce-max-depth`; default `standard`). The other modes select with the branch bits over a node table that already carries the deferred hash constant, so there is no per-group compare, root broadcast or XOR. `bits`: binary select tree, leaf selects as multiply_adds with precomputed deltas, selects above them as SelectChoice (0, 1, 3, 7 per group at depths 1 to 4 against `standard`'s 1, 3, 3, 7, so fewer flow choices at depths 1 and 2 and the same at 3 and 4). `bits-flow`: leaf selects are SelectChoice too, so all `2^d - 1` selects can go to the flow engine (fewest vector ops; greedy puts every select on flow and gets slower). `mobius`: `2^d - 1` multiply_adds over Mobius coefficients, no flow ops (proofs `R13`, `R14`).
- `--select-bit-source {index,path}` - With `--cached-select bits|bits-flow|mobius`: `index` (default) extracts the older bits used at depths 3 and 4 from the index (2 vector ops per group at depth 3, 4 at depth 4; one more for a zero-based index) instead of keeping them live; the newest bit, and at depth 2 the depth-0 bit, are still read directly (with a zero-based index the depth-0 bit stays live one extra round); `path` keeps every branch bit live until the select (no ops, 8 scratch words per group and earlier round; greedy exceeds the scratch budget at 16 rounds) (proof `R15-bits-from-index`).
- `--depth4-select-rounds {all,after-wrap}` - With `--tree-reduce-max-depth 4`: which visits of tree level 4 select among the 16 cached nodes instead of gathering (default `all`: rounds 4 and 15 of 16). `after-wrap` gathers on the first pass (round 4) and selects only after the wrap (round 15): 15 selects per group instead of 8 loads per group, where loads are what binds the floor.
- `--reassociate-gather-xor` - Gather rounds: XOR the deferred hash constant into the value instead of the loaded node, `val ^ (node ^ K) = (val ^ K) ^ node` (R16): one XOR after the load, q -1, same op count. `--index-early-terms {none,final,first-gather}` (mirrored index only) - write consumers of `n' = 2n + bit` as `madd(n, -2, V) - bit` (R17): `final` (index store) q -1, +1 op; `first-gather` also t -1, +1 op per first gather. `--select-xor-pushdown {none,first-pass,all}` - `val ^ sel(c, a, b) = sel(c, val ^ a, val ^ b)` (R18): t -1 per select round, +1 op per group and round. Best uses: path `--reassociate-gather-xor --greedy-const-policy alu-first` 1,026 (fragile: an interaction, worse at stagger 8); index `--index-early-terms final` 1,038. See [docs/CRITICAL_PATH.md](docs/CRITICAL_PATH.md).
- `--emission-plan PATH` - JSON plan (format: `solver/emission_plan.py`; written by `scripts/emission_structure.py solve`) of the tree-access encoding of every group in every round: `gather` (depth 1 to 10), `bits`, `bits-flow`, `mobius` (depth 0 to 4), mixed freely within a round. Replaces `--cached-select`, `--tree-reduce-max-depth` and `--depth4-select-rounds` for the tree access (combining it with the first or last is an error). A plan that spells out what the flags choose emits the same SSA program. See [docs/EMISSION_STRUCTURE_MODEL.md](docs/EMISSION_STRUCTURE_MODEL.md).

**Why SSA builder?**
- Eliminates false dependencies between operations
- Enables better scheduling with CP-SAT
- VALU shift enabled by default for better slot utilization

**Important limitations:**
- The **greedy scheduler** (`--use-greedy`) tries to respect the scratch budget via deferred scheduling, but may exceed it as a last resort (reported via `budget_exceeded` flag). When sections exceed budget, per-section CP-SAT runs automatically fix them.
- **CP-SAT** (default) is required for guaranteed correct results with the SSA builder.

**Scheduler Selection Guide:**
- **SSA + Greedy** (`--use-greedy`): Quick iteration only (~2s), may exceed budget. The greedy scheduler always runs as Phase 1 even with CP-SAT; `--use-greedy` skips Phases 2-3. **Never use `--use-greedy` for recorded results or benchmarking.**
- **SSA + CP-SAT 1 hour** (default): Production runs, benchmarking (recommended, correct output)
- **SSA + CP-SAT 30 min** (`--cpsat-time-limit 1800`): Good for 4-round development iteration

**4-round and full-kernel validation:**
- All full 4-round tests (and 16-round when applicable) **must pass with CP-SAT** when recording results or benchmarking. Do not use `--use-greedy` for recorded results.
- CP-SAT is required to find a **valid feasible schedule** that respects engine slot limits (ALU 12, VALU 6, Load 2, Store 2, **Flow 1**). The model enforces Flow capacity 1 via cumulative + NoOverlap; after solve we validate per-cycle slot counts. **If slot limits are ever violated in a CP-SAT solution, that indicates a serious bug in the constraints** and must be fixed.
- **Per-section feasibility**: When greedy exceeds scratch budget for a section, a per-section CP-SAT run automatically finds a feasible schedule within budget. This adds a few cycles but ensures correctness.
- **Recorded progress** (options run, cycles, peak liveness, before/after treeselect): see **[docs/4-round-phase-results.md](docs/4-round-phase-results.md)**.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Entry Points                                 │
│  perf_takehome.py   - Test harness & performance measurement   │
│  problem.py         - VLIW machine simulator & ISA definition  │
│  generate_ssa_kernel.py - CLI for kernel generation             │
└─────────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┴────────────────────┐
         │                                         │
┌────────▼─────────────────────┐   ┌───────────────▼───────────────┐
│   Kernel Builders             │   │   Rust Module (scheduler_rs/) │
│                               │   │                               │
│   scheduler_rs/ (SSA only)    │   │   kernel_builder/             │
│   └── build_tick_tock_kernel_ssa │   │   └── tick_tock_ssa/ (main)  │
│                               │   │   └── emit_ssa.rs (SSA emit) │
│   solver/kernel_builder/      │   │   └── ssa_greedy/ (greedy)   │
│   └── SSAKernelBuilder (wrapper)│   │   cpsat_scheduler/ (CP-SAT)  │
│                               │   │   types.rs (SsaOp, traits)    │
└───────────────────────────────┘   └───────────────────────────────┘
```

### Three-Phase Hint-Based Scheduling

The scheduling pipeline has three phases:

1. **Greedy scheduler** (~100ms): Produces per-op cycle hints, VALU shift decisions, and SelectChoice/ConstChoice. May exceed scratch budget in some sections — this is expected, not a bug.
2. **Per-section feasibility CP-SAT** (seconds–minutes): Re-schedules only sections that exceed scratch budget. Merges results back, potentially changing SelectChoice decisions.
3. **Full-kernel CP-SAT** (minutes–hours): Uses greedy+repair hints as warm start. Optimizes across all sections simultaneously for minimum cycles.

Better greedy hints → faster CP-SAT convergence. Hint quality is measured by cycles, peak liveness, and budget headroom. See [docs/GREEDY_SCHEDULER_DESIGN.md](docs/GREEDY_SCHEDULER_DESIGN.md) for detailed analysis of pressure policies and improvement roadmap.

### Rust Kernel Builder (SSA only)

The only Rust kernel builder is the SSA builder via `scheduler_rs`:

**SSA Builder** - Eliminates false dependencies for better scheduling:
```python
import scheduler_rs

# SSA builder with CP-SAT (best quality)
instrs, stats = scheduler_rs.build_tick_tock_kernel_ssa(
    forest_height=10,
    n_nodes=2047,
    batch_size=256,
    rounds=16,
    enable_valu_shift=True,   # VALU→ALU shifting (default)
    verbose=True,
    use_cpsat=True,
    cpsat_time_limit=3600.0,  # 1 hour (default)
    output_json=None
)
print(f"Cycles: {stats['cycles']}, Budget exceeded: {stats['budget_exceeded']}")
```

Benefits of Rust builders:
- **40x faster** than Python builder (parallel multipass scheduling)
- No Python→Rust parsing overhead (ops created as native Rust structs)
- SSA builder enables optimal CP-SAT scheduling

## ISA Quick Reference

See **[docs/ISA_REFERENCE.md](docs/ISA_REFERENCE.md)** for complete instruction set documentation.

### Execution Model
- **VLIW**: Multiple operations execute in parallel each cycle
- **SIMD**: Vector operations process VLEN=8 elements simultaneously
- **Timing**: All operations complete in 1 cycle; results visible next cycle
- **Memory**: 32-bit words; 1536-word scratch space (registers/cache)
- **Special Address s0**: Address 0 is always initialized to 0 at startup. `add_imm` reads from s0 to materialize constants without using Load slots. No precedence constraints needed for s0 usage.

### Slot Limits (per cycle)
| Engine | Slots | Operations |
|--------|-------|------------|
| ALU    | 12    | `+`, `-`, `*`, `//`, `^`, `&`, `\|`, `<<`, `>>`, `%`, `<`, `==`, `cdiv` |
| VALU   | 6     | Same as ALU on 8 elements + `vbroadcast`, `multiply_add` |
| Load   | 2     | `load`, `vload`, `const`, `load_offset` |
| Store  | 2     | `store`, `vstore` |
| Flow   | 1     | `select`, `vselect`, `add_imm`, `jump`, `cond_jump`, `halt` |

### Instruction Bundle Format
Each cycle executes one bundle (dict of engine → list of operations):
```python
{
    "alu": [("+", dest, a, b), ("-", dest2, c, d)],
    "valu": [("^", vdest, va, vb)],  # operates on [vdest:vdest+8]
    "load": [("vload", dest, addr)],
}
```

### Key Optimization Techniques
1. **Full unrolling** - All 16 rounds × 32 groups explicit (no loops)
2. **Interleaved emission** - Mix operations from different iterations
3. **Tree value caching** - Predictable early rounds use broadcasts
4. **Fused multiply_add** - `idx = idx * 2 + offset` in one op
5. **VALU→ALU shifting** - Move ops to underutilized ALU slots
6. **Constant derivation** - Generate constants via ALU instead of loads

## Verification (Z3)

Z3 is used to prove that tightening (e.g. slot-aware ASAP/ALAP) **preserves CP-SAT feasibility**. The property we want is **strong**: **all** previously feasible schedules should still be feasible (we only exclude (O,t) that no feasible schedule uses). That is stronger than "at least one feasible schedule remains." **Do not confuse the two notions of feasibility:**

- **Soundness proofs (most checks):** We encode a **violation** of the theorem (counterexample). **Z3 SAT** = it found a violation = **bad**. **Z3 UNSAT** = no assignment violates our theorems = theorems hold = **good**.
- **Preservation checks** (e.g. T7-asap-only-composition): We ask for a **witness** (a feasible schedule). **Z3 SAT** = witness found = we preserved feasibility = **good**. **Z3 UNSAT** = no witness = no feasible schedule = **bad**.

Full explanation: [verification/README.md](verification/README.md) § "Z3 vs CP-SAT: two notions of feasibility."

## Key Files

| File | Purpose |
|------|---------|
| `problem.py` | Machine simulator, ISA definition, reference kernel |
| `perf_takehome.py` | Main test harness for performance measurement |
| `test_optimized.py` | Test kernel correctness and performance |
| `generate_ssa_kernel.py` | **Generate kernels with SSA builder + CP-SAT (recommended)** |
| `scheduler_rs/src/kernel_builder/tick_tock_ssa/` | Main SSA kernel builder (orchestration, per-section feasibility) |
| `scheduler_rs/src/kernel_builder/ssa_greedy/` | SSA greedy scheduler (budget-aware, precedence offsets) |
| `scheduler_rs/src/kernel_builder/emit_ssa.rs` | SSA op emission (ConstChoice, SelectChoice, tree-reduce) |
| `scheduler_rs/src/kernel_builder/memory_deps.rs` | Main-memory locations, exact memory edges (RAW/WAR/WAW) and the filter that drops edges data flow implies |
| `scheduler_rs/src/cpsat_scheduler/` | CP-SAT model (config, bounds, validation, model, legacy) |
| `scheduler_rs/src/types.rs` | SsaOp struct, PrecedenceOffset trait, Engine/Opcode enums |
| `solver/kernel_builder/ssa_wrapper.py` | Python wrapper (SSAKernelBuilder) calling build_tick_tock_kernel_ssa |
| `docs/ISA_REFERENCE.md` | Complete instruction set documentation |
| `docs/4-round-phase-results.md` | Recorded 4-round optimization progress |
| `docs/EMISSION_ABLATION.md` | Emission options (fewer ops per kernel): what each removes, its proof, the 16-round ablation table, recommendation |
| `scripts/measure_baseline.py` | Measures the original scalar baseline (`perf_takehome.py` at `dceff1151fb8…`, pinned by full sha and sha256) on the vendored upstream simulator: 147,734 cycles at 16 rounds / batch 256; `--save` writes the kernel for `scripts/kernel_lift.py` (docs/BASELINE_TO_BEST_GAP.md) |
| `scripts/emission_ablation.py` | Rebuilds that table: per configuration builds, proves, validates, runs upstream, reports ops, loads, flow, floor, cycles |
| `docs/ENERGETIC_REASONING_FOR_UNIT_TASKS.md` | Why stronger cumulative propagation cannot raise the bound for unit tasks, what precedence plus capacity adds, the unsound rule that was removed, the rule in `resource_bounds.rs` and where shifted lanes weaken it, measurements; research notes with sources in `docs/research/` |
| `docs/ASAP_ALAP_SLOT_TIGHTENING.md` | **Canonical design:** slot tightening rules (R_entailed, R_precedence_only), strong guarantee, tried-and-unsafe (R_precedence_demand), implementation flags |
| `docs/SLOT_TIGHTENING_T7_COUNTEREXAMPLE.md` | T7 investigation: 2r/16b counterexample, Z3 proofs, ablation; design lives in ASAP_ALAP_SLOT_TIGHTENING.md |
| `docs/GREEDY_HINT_PASSES_ABLATION.md` | Hint-pass lowering (greedy-only uses greedy_decisions), ablation config, Phase 1/2 commands |
| `docs/GREEDY_SHAPE_ABLATION.md` | Greedy schedule shape options (barriers, wavefront priority, split shift, pressure thresholds): what serialises rounds, ablation tables for 2 to 16 rounds, sweeps, CP-SAT from each greedy start, recommendation |
| `verification/` | Z3 verification of liveness/slot-tightening theorems; see [verification/README.md](verification/README.md) (Z3 vs CP-SAT feasibility) |

## Important Constraints

- **Vector length (VLEN) = 8** - All vector operations process 8 elements
- **Batch size must be a multiple of 8** - `batch_size` must equal VLEN × n_groups (e.g. 8, 16, 32)
- **Scratch space = 1536 words** - Limited register/cache space
- **Write-after-read timing** - Results visible next cycle
- **Dependencies** - `dep_ids` are VReg RAW edges only (latency >= 1 everywhere they are read). Main-memory ordering is separate: every load/store carries a `MemLocation` set by the emitter (`set_next_mem_loc`; a memory op without one is an error), and `compute_memory_dependencies` keeps in `SsaOp::mem_deps` the RAW (latency 1), WAR (latency 0: same cycle allowed, the simulator reads before it writes) and WAW (1) edges that no `dep_ids` path implies. Empty for every kernel emitted today. Code that orders ops (greedy readiness, ASAP/ALAP, CP-SAT precedence, validation, section split, DCE) must read both; memory edges carry no scratch value and never affect liveness.
- **Primary bottleneck: VALU saturation** - 98%+ VALU utilization in optimized schedules; gather operations (8 non-contiguous loads per group) are secondary

## Validation

All kernels are validated against `reference_kernel2()` in `problem.py`. The kernel must produce identical outputs to be considered correct.

`problem.py` here differs from upstream (8-word header instead of 7, rewritten `reference_kernel2`, extra debug ops), and the root `frozen_problem.py` re-exports it, so `tests/submission_tests.py` is **not** the upstream oracle. The upstream oracle is `./scripts/check_upstream.sh`: it runs the unmodified upstream `tests/submission_tests.py` and `frozen_problem.py` (vendored in `upstream/`, pinned by `upstream/UPSTREAM_COMMIT` and `upstream/SHA256SUMS`) on a saved kernel. Never edit the vendored files.

```bash
# Validate (default uses KernelBuilder = SSA; --rust-builder calls build_tick_tock_kernel_ssa directly)
uv run test_optimized.py
uv run test_optimized.py --rust-builder
```
