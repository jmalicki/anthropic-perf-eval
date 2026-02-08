# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **VLIW SIMD Kernel Optimization Challenge**. The goal is to minimize execution cycles for a tree traversal + hashing kernel on a custom VLIW architecture.

**Performance Status:**
- Baseline: 147,734 cycles
- Current best (4 rounds): 485 cycles (304.6x speedup, CP-SAT 1800s, no VALU shift)
- Current best (16 rounds): TBD
- Target: 1,487 cycles (100x speedup)

## Default Problem Size

**Current development: use 2 rounds** for kernel generation, CP-SAT debugging, bulk-move, and iteration (e.g. `uv run generate_ssa_kernel.py --rounds 2`). The script default is still 16 for benchmarking.

Benchmark/default parameters:
- `forest_height=10` - Tree depth (2^10 - 1 = 2047 nodes)
- `batch_size=256` - Number of parallel traversals
- `rounds=16` - Hash rounds per traversal (use `--rounds 2` for dev)

**Batch size constraint:** `batch_size` must be a positive multiple of VLEN (8), because the kernel processes vectors of 8 elements. Invalid values (e.g. 12, 20) produce wrong output or are rejected. Use e.g. 8, 16, 24, 32.

**Terminal / scripting:** Do not use `tail` (or similar) to truncate optimizer or build output. We should see the whole output—run commands without piping to `tail` so the full output is displayed. You may read only the last part of a file or buffer when needed, but present the full output (or the full relevant portion) to the user.

## Project skills

When working in this repo, read and follow these project-specific skills when they apply:

- **Debugging makespan / empty cycles** — Read [.cursor/skills/use-bulk-move/SKILL.md](.cursor/skills/use-bulk-move/SKILL.md). Use the bulk-move simulator to see if empty cycles can be compacted (ReachedEnd) or which constraint blocks (Blocked). Requires CP-SAT run with `--output-json`; run `uv run scripts/simulate_bulk_move.py` on that JSON.
- **Schedule debug / per-decile slot utilization** — Read [.cursor/skills/use-schedule-debug/SKILL.md](.cursor/skills/use-schedule-debug/SKILL.md). Use per-decile slot utilization to find optimization opportunities (slot balance, encoding improvements). Run `uv run scripts/slot_usage_by_decile.py` on scheduler output JSON; see [docs/PER_DECILE_SLOT_UTILIZATION.md](docs/PER_DECILE_SLOT_UTILIZATION.md).
- **What-if shift (debug VALU shift decisions)** — Read [.cursor/skills/use-whatif-shift/SKILL.md](.cursor/skills/use-whatif-shift/SKILL.md). Force a specific VALU op to shift to ALU, re-optimize with CP-SAT, and optionally compare to baseline. Run `uv run scripts/what_if_shift.py --op-id <id> [--compare-with scheduler_output.json]`.
- **Bug tracking** — Read [.cursor/skills/use-bugs-md/SKILL.md](.cursor/skills/use-bugs-md/SKILL.md). Use BUGS.md at the repo root: read before fixing, update when fixing or reporting, keep entries short.
- **Bisecting validation failures / minimal repros** — Read [.cursor/skills/use-bisection-tools/SKILL.md](.cursor/skills/use-bisection-tools/SKILL.md). Use `scripts/bisect_cpsat_validation.py` to find which choice options (ConstChoice, SelectChoice, valu shift) correlate with failure, and `scripts/bisect_minimal_fma.py` for smallest rounds/batch/height that reproduces.
- **Hypothesis quantification (schedule optimization ideas)** — Read [.cursor/skills/use-schedule-hypotheses/SKILL.md](.cursor/skills/use-schedule-hypotheses/SKILL.md). Use `scripts/quantify_schedule_hypotheses.py` on scheduler output JSON to estimate impact of vbroadcast elimination, element-wise chaining, and ConstChoice; use A/B runs and bulk-move/what-if for VALU shift and precedence.
- **Ablation analysis (ASAP/ALAP tightening)** — Read [.cursor/skills/use-ablation-analysis/SKILL.md](.cursor/skills/use-ablation-analysis/SKILL.md). Run 4-arm ablation study (baseline, slot-aware, scratch-aware, both) with `scripts/run_asap_alap_ablation.py`; analyze results with `scripts/analyze_ablation_results.py`.

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

```bash
# Rust: full scheduler_rs test suite (from project root)
BUILD_FROM_PROJECT_ROOT=1 cargo test --manifest-path scheduler_rs/Cargo.toml --features cpsat

# Rust: include optional/CI-only tests (slow proptest, minimal failing config, etc.)
BUILD_FROM_PROJECT_ROOT=1 cargo test --manifest-path scheduler_rs/Cargo.toml --features "cpsat ci-tests"

# Python: interface test only (scheduler_rs build → Machine → reference)
uv run pytest tests/test_scheduler_rs_interface.py -v

# Python: submission tests (end-to-end pipeline)
uv run tests/submission_tests.py

# Python: remaining solver utility tests
uv run pytest solver/test_slot_utilization.py -v

# Manual / CI: optimized kernel with optional --rust-builder
uv run test_optimized.py
```

**Optional tests:** Prefer a **Cargo feature** (`ci-tests`) for optional/CI-only Rust tests so they run when requested (e.g. `cargo test --features "cpsat ci-tests"`). Use `#[ignore]` only for temporarily disabled tests that expose known bugs (with a comment and issue reference). CI should run with `--features "cpsat ci-tests"` for full coverage. For Python, use a pytest marker (e.g. `@pytest.mark.ci_only`) and run default with `-m "not ci_only"`; document in `pyproject.toml`.

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
- `--cpsat-time-limit SECONDS` - CP-SAT time limit (default: 3600 = 1 hour)
- `--enable-valu-shift` / `--no-valu-shift` - VALU→ALU shifting (enabled by default)
- `--tree-reduce-max-depth N` - Max tree depth for tree-reduce selection, 2 or 3 (default: 3)
- `--const-choice MODE` - ConstChoice handling: `optimize` (default), `add_imm`, or `load`
- `--select-choice MODE` - SelectChoice handling: `optimize` (default), `fma`, or `vselect`
- `--valu-shift MODE` - VALU shift handling: `optimize` (default), `alu`, or `valu`
- `--per-section-optimize-secs SECONDS` - Per-section CP-SAT optimization time (0 or omit = feasibility only)
- `--force-zero-gap` - Force 0 integral gap for proven optimality (auto uses 2h time limit)
- `--enable-asap-alap-tightening` - Experimental: tighten ASAP/ALAP by slot capacity
- `--enable-scratch-tightening` - Experimental: scratch-aware domain tightening

Output options:
- `--output-json PATH` - Write scheduler output JSON (default: `scheduler_output.json`). Use this file for summarization and slot/liveness analysis; see "Scheduler output JSON" above.
- `--validate` - Validate against reference kernel. **The kernel is always saved before validation** (to `kernel_to_validate.json` unless `--save-kernel PATH` is set) so failed runs can be analyzed or re-validated without rebuilding.
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

## Important Constraints

- **Vector length (VLEN) = 8** - All vector operations process 8 elements
- **Batch size must be a multiple of 8** - `batch_size` must equal VLEN × n_groups (e.g. 8, 16, 32)
- **Scratch space = 1536 words** - Limited register/cache space
- **Write-after-read timing** - Results visible next cycle
- **Primary bottleneck: VALU saturation** - 98%+ VALU utilization in optimized schedules; gather operations (8 non-contiguous loads per group) are secondary

## Validation

All kernels are validated against `reference_kernel2()` in `problem.py`. The kernel must produce identical outputs to be considered correct.

```bash
# Validate (default uses KernelBuilder = SSA; --rust-builder calls build_tick_tock_kernel_ssa directly)
uv run test_optimized.py
uv run test_optimized.py --rust-builder
```
