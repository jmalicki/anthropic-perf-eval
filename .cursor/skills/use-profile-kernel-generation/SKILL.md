---
name: use-profile-kernel-generation
description: Run kernel generation with profile-specific output paths (scheduler_output_profile.json, kernel_profile.json) so profiling runs do not overwrite normal runs; optionally run under perf for CPU profiling. Use when profiling generate_ssa_kernel.py or comparing optimized vs profile runs.
---

# Profile Kernel Generation

**Script:** `scripts/profile_kernel_generation.sh`

Wrapper around `uv run generate_ssa_kernel.py` that uses **profile-specific** output files by default, so you can profile without overwriting `scheduler_output.json` or `kernel_to_validate.json`. Optional **perf** recording for CPU flamegraphs.

## When to Use

- Profiling kernel generation (Python + Rust) without clobbering your usual scheduler/kernel outputs.
- Recording a perf profile for `generate_ssa_kernel.py` (e.g. `perf report -g` or flamegraph).

## How to Run

```bash
# Profile run with default profile paths (scheduler_output_profile.json, kernel_profile.json)
./scripts/profile_kernel_generation.sh -- --rounds 5 --batch-size 256 --tree-reduce-max-depth 3 --cpsat-time-limit 300 --validate

# With perf (record only; analyze with perf report -g or perf script | flamegraph)
USE_PERF=1 ./scripts/profile_kernel_generation.sh -- --rounds 5 --batch-size 256 --cpsat-time-limit 300
```

Pass any `generate_ssa_kernel.py` arguments after `--`. If you pass your own `--output-json` or `--save-kernel`, those paths are used instead of the profile defaults.

## Optional: Rust symbols in perf

For useful Rust symbols in `perf report`, build scheduler_rs with debuginfo first:

```bash
RUSTFLAGS="-C debuginfo=2" BUILD_FROM_PROJECT_ROOT=1 uv run maturin develop --manifest-path scheduler_rs/Cargo.toml
```

Then run with `USE_PERF=1` as above.
