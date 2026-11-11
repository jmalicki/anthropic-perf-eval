# CLAUDE.md

[![CI](https://github.com/jmalicki/anthropic-perf-eval/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/jmalicki/anthropic-perf-eval/actions/workflows/ci.yml)

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **VLIW SIMD Kernel Optimization Challenge**. The goal is to minimize execution cycles for a tree traversal + hashing kernel on a custom VLIW architecture.

**Performance Status:**
- Baseline: 147,734 cycles
- 16 rounds (the benchmark size): no recorded CP-SAT result yet. Greedy reference, not a recorded result (see "Scheduler Selection Guide"): 2,007 cycles (73.6x speedup; `--use-greedy` via the CLI, validated). Whole-program CP-SAT did not improve on the greedy schedule in 600 s.
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
- **Reproducibility (same command, same kernel)** — Builds are deterministic: two runs with the same options hand CP-SAT byte-identical models and, with a `--cpsat-deterministic-time` budget that the solve ends on (not the wall-clock backstop), emit byte-identical kernels (`tests/test_determinism.py`). If two runs disagree, export each model with `CPSAT_EXPORT_MODEL=path.pb` and run `uv run python -m scripts.diff_cpsat_models a.pb b.pb`; it reports the first constraint, variable or hint that differs and whether it is only a reordering. New code that creates solver variables/constraints or allocates addresses must not iterate a `HashMap`/`HashSet` directly: use `sorted_by_key` (in `cpsat_scheduler/model.rs`), a `BTreeMap`/`BTreeSet`, or sort first.
- **Debugging makespan / empty cycles** — Read [.cursor/skills/use-bulk-move/SKILL.md](.cursor/skills/use-bulk-move/SKILL.md). Use the bulk-move simulator to see if empty cycles can be compacted (ReachedEnd) or which constraint blocks (Blocked). Requires CP-SAT run with `--output-json`; run `uv run scripts/simulate_bulk_move.py` on that JSON.
- **Schedule debug / per-decile slot utilization** — Read [.cursor/skills/use-schedule-debug/SKILL.md](.cursor/skills/use-schedule-debug/SKILL.md). Use per-decile slot utilization to find optimization opportunities (slot balance, encoding improvements). Run `uv run scripts/slot_usage_by_decile.py` on scheduler output JSON; see [docs/PER_DECILE_SLOT_UTILIZATION.md](docs/PER_DECILE_SLOT_UTILIZATION.md).
- **What-if shift (debug VALU shift decisions)** — Read [.cursor/skills/use-whatif-shift/SKILL.md](.cursor/skills/use-whatif-shift/SKILL.md). Force a specific VALU op to shift to ALU, re-optimize with CP-SAT, and optionally compare to baseline. Run `uv run scripts/what_if_shift.py --op-id <id> [--compare-with scheduler_output.json]`.
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
- **Schedule report (how good is this schedule?)** — Read [.cursor/skills/use-schedule-report/SKILL.md](.cursor/skills/use-schedule-report/SKILL.md). `uv run python -m scripts.schedule_report sched.json` prints per-engine floors, the capacity floor for the ops this run emitted, the gap to it, per-round cycle spans and utilisation over time; `--compare` diffs two runs. The generator prints the floor and gap after every build. The floor (and `lb_cycles`) is for this emission only, never a bound on the problem.
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

**One entry point:** `./scripts/test.sh [rust|python|proofs|kernel|upstream|all]` sets the environment each suite needs (OR-Tools prefix, `PYO3_PYTHON` pointing at the project venv, `BUILD_FROM_PROJECT_ROOT`). Arguments after the suite name are passed through.

```bash
./scripts/test.sh                 # everything
./scripts/test.sh rust            # cargo test --features cpsat
./scripts/test.sh rust -- test_vector_group      # one Rust test
./scripts/test.sh python          # pytest: interface, dataflow replay, forced modes, rewrite proofs
./scripts/test.sh python tests/test_kernel_dataflow.py -k lane
./scripts/test.sh proofs          # Z3: liveness theorems + rewrite identities
./scripts/test.sh kernel          # greedy 2r and 16r builds with dataflow replay + validation
./scripts/test.sh upstream        # 16-round greedy kernel on the unmodified upstream simulator and tests

# Rust: include optional/CI-only tests (slow proptest, minimal failing config, etc.)
./scripts/test.sh rust --features "cpsat ci-tests"

# Python: submission tests (end-to-end pipeline)
uv run tests/submission_tests.py

# Upstream check: 16-round kernel on the unmodified upstream simulator and tests (vendored in upstream/)
./scripts/check_upstream.sh                 # greedy smoke check; pass generate_ssa_kernel.py options for CP-SAT
./scripts/check_upstream.sh --kernel kernel_to_validate.json   # test a saved kernel
```

`uv run pytest` works directly: pytest, pytest-timeout and z3-solver are in the default `dev` dependency group. Running `cargo test` by hand needs `PYO3_PYTHON=$PWD/.venv/bin/python` if the system Python is newer than PyO3 supports ("the configured Python version is newer than PyO3's maximum supported version").

**Optional tests:** Prefer a **Cargo feature** (`ci-tests`) for optional/CI-only Rust tests so they run when requested (e.g. `cargo test --features "cpsat ci-tests"`). Use `#[ignore]` only for temporarily disabled tests that expose known bugs (with a comment and issue reference). CI should run with `--features "cpsat ci-tests"` for full coverage. For Python, use a pytest marker (e.g. `@pytest.mark.ci_only`) and run default with `-m "not ci_only"`; document in `pyproject.toml`.

### CI

GitHub Actions ([.github/workflows/ci.yml](.github/workflows/ci.yml)) runs on every pull request and on pushes to `main`, on a macOS arm64 runner (`macos-latest`) with OR-Tools from Homebrew, the same setup as local development. Two jobs:

- **Rust tests (scheduler_rs)** — `./scripts/test.sh rust` (debug profile, `--features cpsat`).
- **Python tests, proofs, kernel builds** — `./scripts/build_scheduler_rs.sh --release`, then `./scripts/test.sh python`, `./scripts/test.sh proofs`, `./scripts/test.sh kernel`, `./scripts/test.sh upstream`, one step each.

To reproduce a failure locally, run the suite named in the failing step: `./scripts/test.sh <suite>` (rebuild first with `./scripts/build_scheduler_rs.sh --release` for the Python-side suites). CI installs whatever OR-Tools version Homebrew currently ships (printed by the "Install OR-Tools" step); if a failure does not reproduce, compare that against `brew list --versions or-tools`.

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
- `--valu-shift MODE` - VALU shift handling: `optimize` (default), `alu`, or `valu`
- `--per-section-optimize-secs SECONDS` - Per-section CP-SAT optimization time (0 or omit = feasibility only)
- `--force-zero-gap` - Force 0 integral gap: prove the schedule optimal for the emitted ops and encoding choices, not for the problem (auto uses 2h time limit)
- `--asap-alap-tightening {none,simple,symmetry-aware,energetic}` - ASAP/ALAP tightening mode (default: energetic). See [docs/ASAP_ALAP_SLOT_TIGHTENING.md](docs/ASAP_ALAP_SLOT_TIGHTENING.md).
- `--enable-scratch-tightening` - Non-default debugging: scratch-aware domain tightening (default: disabled)
- `--bulk-move` - Enable bulk-move compaction after CP-SAT to eliminate empty cycles. **Off by default:** CP-SAT should produce schedules without empty cycles; bulk-move merely masks solver/model bugs rather than fixing them, and is buggy (can cause same-cycle scratch write conflicts). Use only for experimenting; do not rely on it for production.
- `--hint-passes` - Enable all three greedy hint passes (ConstChoice, SelectChoice, VALU shift). Off by default. See [docs/GREEDY_HINT_PASSES_ABLATION.md](docs/GREEDY_HINT_PASSES_ABLATION.md).
- `--const-choice-hint-pass`, `--select-choice-hint-pass`, `--valu-shift-hint-pass` - Enable hint passes individually.
- `--schedule-with-fixed-choices` - After hint passes, re-run greedy with encodings fixed to assign cycles.

Greedy schedule shape options (ablation switches, defaults unchanged):
- `--greedy-barriers {round,none}` - Round barriers in the greedy scheduler. `round` (default): every op of round r+1 waits for all of round r, so rounds run strictly one after another. `none`: the barrier pseudo-ops are dropped before dependencies are computed, the ready queue may mix rounds, and the kernel is a single section for the per-section budget check. On its own this does not help at 16 rounds (the ready queue stays round-major and the greedy exceeds the scratch budget); it is the prerequisite for the other shape options.
  - The option also applies when CP-SAT follows the greedy (no `--use-greedy`). The CP-SAT model itself never contains barriers (they are always removed before it is built); what changes is the greedy schedule it gets as its warm start, which also caps every op's latest cycle (greedy cycle + 15) and sets the horizon, and the per-section repair, which then sees one section instead of one per round.
- `--greedy-priority {fifo,wavefront}` - Order in which the greedy tries ready ops. `fifo` (default): by ready cycle, then emission order, which keeps all 32 groups in the same round. `wavefront`: key = dependency depth + stagger × group rank, so group g runs as if it started `stagger × g` cycles after group 0 and some groups gather while others hash or select (see `scheduler_rs/src/kernel_builder/ssa_greedy/wavefront.rs`; groups are found structurally as the connected components of ops that reach few final stores). Use with `--greedy-barriers none`; with barriers it has nothing to reorder across rounds and fails the scratch budget at 8 and 16 rounds.
  - `--greedy-stagger-cycles N` - Stagger between consecutive groups in cycles (default 12, about three quarters of one round's critical path; 0 = all groups abreast).
  - `--greedy-groups-in-flight N` - Groups allowed to run at once (default 16; 0 = no limit). Bounds scratch: a group that has not started holds nothing, a finished group has stored its results.
  - 16 rounds, greedy: `uv run generate_ssa_kernel.py --rounds 16 --use-greedy --validate --greedy-barriers none --greedy-priority wavefront` gives 1,398 cycles (1,400 on the simulator) against 2,007 (1,994) by default.
- `--greedy-split-shift` - Greedy-only. A vector op run as 8 scalar ops may straddle two consecutive cycles (first lanes in the leftover scalar slots of one cycle, the rest in the next), so all 12 scalar slots carry shifted work: 3 shifted ops per 2 cycles instead of 2, i.e. 7.5 vector-op equivalents per cycle instead of 7. The op's cycle is its last lane's; per-lane cycles go to `shifted_alu_cycles` as for CP-SAT. Ignored when CP-SAT follows (its hints carry one cycle per op). With the wavefront priority at 16 rounds: 1,314 cycles (1,316 on the simulator), 43 above the 1,271 floor.
- `--pressure-inflation-threshold F`, `--pressure-soft-threshold F`, `--pressure-critical-threshold F` - Override the pressure policy's thresholds (fractions of the scratch budget; built-in 0.80 / 0.85 / 0.95, tuned before the greedy's live-word counter was exact). The inflation threshold is the one that matters: above it, producers with slack are deferred by up to 10 cycles. `--pressure-inflation-threshold 0.95` shortens the default greedy schedule (3 rounds 326 to 289, 4 rounds 456 to 437, 16 rounds 2,007 to 1,965) and makes the wavefront priority insensitive to `--greedy-groups-in-flight`; 0.98 and above fail the scratch budget at 3 and 16 rounds. The soft and critical thresholds changed nothing measurable.
- `--greedy-budget-headroom WORDS` - Scratch words the greedy keeps free below the budget (default 0). It scales the thresholds down with the budget; it does not replace the inflation threshold (with inflation off the greedy deadlocks at the budget and force-schedules past it whatever the headroom).

Output options:
- `--output-json PATH` - Write scheduler output JSON (default: `scheduler_output.json`). Use this file for summarization and slot/liveness analysis; see "Scheduler output JSON" above.
- `--validate` - Validate against reference kernel. **The kernel is always saved before validation** (to `kernel_to_validate.json` unless `--save-kernel PATH` is set) so failed runs can be analyzed or re-validated without rebuilding.
- `--no-dataflow-check` - With `--validate`, skip the symbolic kernel-vs-SSA replay that runs before numeric validation.
- `--validate-only PATH` - Load kernel from JSON and run validation only (no build).
- `--save-kernel PATH` - Save kernel to this path (default when `--validate`: `kernel_to_validate.json`).
- `--liveness-detail LEVEL` - Liveness detail in JSON: `none`, `summary`, or `full` (default)
- `--comment TEXT` - Comment stored in schedule JSON
- `--seed N` - Random seed for validation (default: 123)
- `-q, --quiet` - Suppress verbose Rust scheduler output

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
| `scheduler_rs/src/cpsat_scheduler/` | CP-SAT model (config, bounds, validation, model, legacy) |
| `scheduler_rs/src/types.rs` | SsaOp struct, PrecedenceOffset trait, Engine/Opcode enums |
| `solver/kernel_builder/ssa_wrapper.py` | Python wrapper (SSAKernelBuilder) calling build_tick_tock_kernel_ssa |
| `docs/ISA_REFERENCE.md` | Complete instruction set documentation |
| `docs/4-round-phase-results.md` | Recorded 4-round optimization progress |
| `docs/ASAP_ALAP_SLOT_TIGHTENING.md` | **Canonical design:** slot tightening rules (R_entailed, R_precedence_only), strong guarantee, tried-and-unsafe (R_precedence_demand), implementation flags |
| `docs/SLOT_TIGHTENING_T7_COUNTEREXAMPLE.md` | T7 investigation: 2r/16b counterexample, Z3 proofs, ablation; design lives in ASAP_ALAP_SLOT_TIGHTENING.md |
| `docs/GREEDY_HINT_PASSES_ABLATION.md` | Hint-pass lowering (greedy-only uses greedy_decisions), ablation config, Phase 1/2 commands |
| `verification/` | Z3 verification of liveness/slot-tightening theorems; see [verification/README.md](verification/README.md) (Z3 vs CP-SAT feasibility) |

## Important Constraints

- **Vector length (VLEN) = 8** - All vector operations process 8 elements
- **Batch size must be a multiple of 8** - `batch_size` must equal VLEN × n_groups (e.g. 8, 16, 32)
- **Scratch space = 1536 words** - Limited register/cache space
- **Write-after-read timing** - Results visible next cycle
- **Primary bottleneck: VALU saturation** - 98%+ VALU utilization in optimized schedules; gather operations (8 non-contiguous loads per group) are secondary

## Validation

All kernels are validated against `reference_kernel2()` in `problem.py`. The kernel must produce identical outputs to be considered correct.

`problem.py` here differs from upstream (8-word header instead of 7, rewritten `reference_kernel2`, extra debug ops), and the root `frozen_problem.py` re-exports it, so `tests/submission_tests.py` is **not** the upstream oracle. The upstream oracle is `./scripts/check_upstream.sh`: it runs the unmodified upstream `tests/submission_tests.py` and `frozen_problem.py` (vendored in `upstream/`, pinned by `upstream/UPSTREAM_COMMIT` and `upstream/SHA256SUMS`) on a saved kernel. Never edit the vendored files.

```bash
# Validate (default uses KernelBuilder = SSA; --rust-builder calls build_tick_tock_kernel_ssa directly)
uv run test_optimized.py
uv run test_optimized.py --rust-builder
```
