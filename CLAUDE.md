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

### Rust Scheduler (in scheduler_rs/)
```bash
cd scheduler_rs
maturin develop          # Build and install for development
maturin build --release  # Build optimized release
```

### Running Tests
```bash
# Run all tests
pytest

# Run solver tests
pytest solver/

# Run submission tests
python tests/submission_tests.py

# Test the optimized kernel
python test_optimized.py
```

### Generating Kernels
```bash
# Generate with default settings (greedy scheduler - fast)
python generate_precomputed_kernel.py --use-greedy

# Generate with CP-SAT optimization (10 min recommended - worth the wait)
python generate_precomputed_kernel.py --full-solve-time 600

# For best results (1 hour, diminishing returns after 10 min)
python generate_precomputed_kernel.py --full-solve-time 3600

# Common options:
#   --use-greedy                Use fast greedy scheduler
#   --enable-valu-shift         Enable VALU→ALU shifting (default)
#   --enable-alu-hash-parallel  Parallel hash using ALU and VALU
#   --enable-chunked-gather     Use chunked vload for tree access
#   --output PATH               Output file (default: solver/precomputed_kernel.py)
```

**Scheduler Selection Guide:**
- **Greedy** (`--use-greedy`): Development iteration, testing (2-3 seconds)
- **CP-SAT 10 min** (`--full-solve-time 600`): Production runs, benchmarking (recommended)
- **CP-SAT 1 hour** (`--full-solve-time 3600`): Final optimization attempts (diminishing returns)

### Running Experiments
```bash
python run_experiment.py           # With W&B logging
python run_experiment.py --offline # Without W&B
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

### Rust Kernel Builder (Recommended)

The fastest way to generate and schedule kernels:

```python
import scheduler_rs

# Build and schedule in ~2 seconds (vs 80+ seconds with Python)
instrs = scheduler_rs.build_tick_tock_kernel(
    forest_height=10,
    n_nodes=2047,
    batch_size=256,
    rounds=16,
    enable_valu_shift=True,
    verbose=True
)
```

Benefits:
- **40x faster** than Python builder (parallel multipass scheduling)
- No Python→Rust parsing overhead (ops created as native Rust structs)
- Identical output quality (same cycle count as Python builder)

## ISA Quick Reference

See **[docs/ISA_REFERENCE.md](docs/ISA_REFERENCE.md)** for complete instruction set documentation.

### Execution Model
- **VLIW**: Multiple operations execute in parallel each cycle
- **SIMD**: Vector operations process VLEN=8 elements simultaneously
- **Timing**: All operations complete in 1 cycle; results visible next cycle
- **Memory**: 32-bit words; 1536-word scratch space (registers/cache)

### Slot Limits (per cycle)
| Engine | Slots | Operations |
|--------|-------|------------|
| ALU    | 12    | `+`, `-`, `*`, `//`, `^`, `&`, `\|`, `<<`, `>>`, `%`, `<`, `==`, `cdiv` |
| VALU   | 6     | Same as ALU on 8 elements + `vbroadcast`, `multiply_add` |
| Load   | 2     | `load`, `vload`, `const`, `load_offset` |
| Store  | 2     | `store`, `vstore` |
| Flow   | 1     | `select`, `vselect`, `jump`, `cond_jump`, `halt` |

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
| `scheduler_rs/src/kernel_builder/` | **Rust kernel builder (recommended)** |
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
python test_optimized.py --rust-builder

# Validate with Python builder (slower, ~80 seconds)
python test_optimized.py --tick-tock
```
