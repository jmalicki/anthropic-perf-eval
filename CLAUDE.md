# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **VLIW SIMD Kernel Optimization Challenge**. The goal is to minimize execution cycles for a tree traversal + hashing kernel on a custom VLIW architecture.

**Performance Status:**
- Baseline: 147,734 cycles
- Current best: ~2,390 cycles (61.8x speedup)
- Target: 1,487 cycles (100x speedup)

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

# Generate with CP-SAT optimization (slow but better)
python generate_precomputed_kernel.py --full-solve-time 3600

# Common options:
#   --use-greedy                Use fast greedy scheduler
#   --enable-valu-shift         Enable VALU→ALU shifting (default)
#   --enable-alu-hash-parallel  Parallel hash using ALU and VALU
#   --enable-chunked-gather     Use chunked vload for tree access
#   --output PATH               Output file (default: solver/precomputed_kernel.py)
```

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
│   Kernel Builders             │   │   Scheduling Engines          │
│   (solver/kernel_builder/)    │   │                               │
│                               │   │   scheduler_rs/ (Rust)        │
│   • TickTockKernelBuilder     │   │   └── scheduler.rs (greedy)   │
│   • ModularKernelBuilder      │   │   └── constants.rs (derive)   │
│   • Strategies:               │   │                               │
│     - tree_access.py          │   │   solver/ (Python, deprecated)│
│     - hash.py                 │   │   └── cp_scheduler.py         │
│     - chunked.py              │   │   └── flexible_scheduler.py   │
└───────────────────────────────┘   └───────────────────────────────┘
```

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
| `generate_precomputed_kernel.py` | Offline kernel generation with all optimizations |
| `solver/precomputed_kernel.py` | The generated optimized kernel (compressed) |
| `solver/kernel_builder/tick_tock.py` | Current best kernel builder |
| `scheduler_rs/src/scheduler.rs` | Rust greedy list scheduler |
| `docs/ISA_REFERENCE.md` | Complete instruction set documentation |
| `docs/OPTIMIZATION_PLAN.md` | Current optimization state and roadmap |

## Important Constraints

- **Vector length (VLEN) = 8** - All vector operations process 8 elements
- **Scratch space = 1536 words** - Limited register/cache space
- **Write-after-read timing** - Results visible next cycle
- **Primary bottleneck: Gather operations** - 8 non-contiguous loads per group

## Validation

All kernels are validated against `reference_kernel2()` in `problem.py`. The kernel must produce identical outputs to be considered correct.

```bash
# Validate current kernel
python test_optimized.py
```
