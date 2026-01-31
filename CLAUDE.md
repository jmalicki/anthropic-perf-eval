# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **VLIW SIMD Kernel Optimization Challenge**. The goal is to minimize execution cycles for a tree traversal + hashing kernel on a custom VLIW architecture.

**Performance Status:**
- Baseline: 147,734 cycles
- Current best: 1,764 cycles (83.7x speedup)
- Target: 1,487 cycles (100x speedup)

## Default Problem Size

The benchmark uses these default parameters:
- `forest_height=10` - Tree depth (2^10 - 1 = 2047 nodes)
- `batch_size=256` - Number of parallel traversals
- `rounds=16` - Hash rounds per traversal

These values are consistent across all test scripts and should be used when benchmarking.

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

**Testing strategy:** Heavy testing of **scheduler_rs** (scheduling, budget, CP-SAT, greedy, correctness) lives in **Rust** (`cargo test --manifest-path scheduler_rs/Cargo.toml --features cpsat`). Python tests for scheduler_rs **only exercise the PyO3 interface and its options** (e.g. `build_tick_tock_kernel`, `build_tick_tock_kernel_ssa`, `enable_valu_shift`, `use_cpsat`): they build via the binding → run on `Machine` → compare to `reference_kernel2`. They do **not** test underlying scheduler_rs functionality (that is for Rust tests). Solver tests in `solver/` and `tests/submission_tests.py` exercise the **Python** kernel builders and pipeline.

```bash
# Rust: full scheduler_rs test suite (from project root)
BUILD_FROM_PROJECT_ROOT=1 cargo test --manifest-path scheduler_rs/Cargo.toml --features cpsat

# Python: interface test only (scheduler_rs build → Machine → reference)
uv run pytest tests/test_scheduler_rs_interface.py -v

# Python: solver / submission tests (Python pipeline, not Rust)
uv run pytest solver/
uv run tests/submission_tests.py

# Manual / CI: optimized kernel with optional --rust-builder
uv run test_optimized.py
```

### Generating Kernels

**Important: Always use `uv run` to execute Python scripts** to ensure the correct Python environment and dependencies.

**Rebuilding scheduler_rs:** `uv run generate_ssa_kernel.py` does **not** rebuild the Rust extension. After changing Rust code, run `./scripts/build_scheduler_rs.sh` first, or use `./scripts/run_generate_ssa_kernel.sh` which rebuilds scheduler_rs when Rust sources are newer than the installed extension, then runs the generator.

**Recommended: SSA builder with CP-SAT** (best quality):
```bash
# Default: 10 min CP-SAT, VALU shift disabled
# Shows full optimization progress (solver stats, cycle improvements)
uv run generate_ssa_kernel.py

# With validation against reference kernel
uv run generate_ssa_kernel.py --validate

# Auto-rebuild scheduler_rs if Rust changed, then run (e.g. long CP-SAT + validate)
./scripts/run_generate_ssa_kernel.sh --enable-valu-shift --cpsat-time-limit 7200 --validate --rounds 5

# Quick iteration with greedy scheduler (~2 seconds, may exceed budget)
uv run generate_ssa_kernel.py --use-greedy

# Longer optimization (diminishing returns after 10 min)
uv run generate_ssa_kernel.py --cpsat-time-limit 3600

# Output schedule JSON for analysis
uv run generate_ssa_kernel.py --output-json schedule.json
```

**CP-SAT solver output:** The script shows full CP-SAT progress by default, including:
- Kernel building phase (SSA ops, dead code elimination)
- Solver progress (objective bounds, conflicts, branches)
- Final statistics (cycles, speedup, scratch usage)

**Options for `generate_ssa_kernel.py`:**
- `--use-greedy` - Use greedy scheduler instead of CP-SAT (fast, ~2s, may exceed budget)
- `--cpsat-time-limit SECONDS` - CP-SAT time limit (default: 600 = 10 min)
- `--enable-valu-shift` - Enable VALU→ALU shifting (disabled by default)
- `--output-json PATH` - Write schedule JSON for analysis
- `--validate` - Validate against reference kernel
- `-q, --quiet` - Suppress verbose Rust scheduler output

**Why SSA builder?**
- Eliminates false dependencies between operations
- Enables better scheduling with CP-SAT
- VALU shift disabled by default to avoid scratch budget issues

**Important limitations:**
- The **greedy scheduler** (`--use-greedy`) does not respect the scratch budget (1536 words) and will produce incorrect results. Use it only for quick iteration, not for correct output.
- **CP-SAT** (default) is required for correct results with the SSA builder.
- VALU shift is disabled by default because it increases scratch pressure.

**Legacy: Python builder** (deprecated):
```bash
uv run generate_precomputed_kernel.py --use-greedy
uv run generate_precomputed_kernel.py --full-solve-time 600
```

**Scheduler Selection Guide:**
- **SSA + Greedy** (`--use-greedy`): Quick iteration only (~2s), results may exceed budget
- **SSA + CP-SAT 10 min** (default): Production runs, benchmarking (recommended, correct output)
- **SSA + CP-SAT 1 hour** (`--cpsat-time-limit 3600`): Final optimization (diminishing returns)

### Running Experiments
```bash
uv run run_experiment.py           # With W&B logging
uv run run_experiment.py --offline # Without W&B
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Entry Points                                 │
│  perf_takehome.py   - Test harness & performance measurement   │
│  problem.py         - VLIW machine simulator & ISA definition  │
└─────────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┴────────────────────┐
         │                                         │
┌────────▼─────────────────────┐   ┌───────────────▼───────────────┐
│   Kernel Builders             │   │   Rust Module (scheduler_rs/) │
│                               │   │                               │
│   scheduler_rs/ (RECOMMENDED) │   │   kernel_builder/             │
│   └── build_tick_tock_kernel  │   │   └── tick_tock.rs (builder)  │
│                               │   │   └── emit.rs (Op emission)   │
│   solver/kernel_builder/      │   │                               │
│   (Python, DEPRECATED)        │   │   scheduler.rs (parallel)     │
│   └── tick_tock.py            │   │   constants.rs (derivation)   │
└───────────────────────────────┘   └───────────────────────────────┘
```

### Rust Kernel Builders

Two Rust kernel builders are available via `scheduler_rs`:

**SSA Builder (Recommended)** - Eliminates false dependencies for better scheduling:
```python
import scheduler_rs

# SSA builder with CP-SAT (best quality)
instrs, stats = scheduler_rs.build_tick_tock_kernel_ssa(
    forest_height=10,
    n_nodes=2047,
    batch_size=256,
    rounds=16,
    enable_valu_shift=False,  # Disabled to avoid budget issues
    verbose=True,
    use_cpsat=True,
    cpsat_time_limit=600.0,   # 10 minutes
    output_json=None
)
print(f"Cycles: {stats['cycles']}, Budget exceeded: {stats['budget_exceeded']}")
```

**Non-SSA Builder** - Fast greedy scheduling:
```python
# Non-SSA with greedy scheduler (~2 seconds)
instrs = scheduler_rs.build_tick_tock_kernel(
    forest_height=10,
    n_nodes=2047,
    batch_size=256,
    rounds=16,
    enable_valu_shift=True,
    verbose=True
)
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
| `scheduler_rs/src/kernel_builder/` | Rust kernel builders (SSA and non-SSA) |
| `scheduler_rs/src/scheduler.rs` | Rust greedy list scheduler with parallel multipass |
| `solver/kernel_builder/tick_tock.py` | Python kernel builder (deprecated) |
| `docs/ISA_REFERENCE.md` | Complete instruction set documentation |

## Important Constraints

- **Vector length (VLEN) = 8** - All vector operations process 8 elements
- **Scratch space = 1536 words** - Limited register/cache space
- **Write-after-read timing** - Results visible next cycle
- **Primary bottleneck: Gather operations** - 8 non-contiguous loads per group

## Validation

All kernels are validated against `reference_kernel2()` in `problem.py`. The kernel must produce identical outputs to be considered correct.

```bash
# Validate with Rust builder (fastest, ~2 seconds)
uv run test_optimized.py --rust-builder

# Validate with Python builder (slower, ~80 seconds)
uv run test_optimized.py --tick-tock
```
