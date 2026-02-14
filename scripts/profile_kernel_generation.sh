#!/usr/bin/env bash
# Run kernel generation with profile-specific output files (and optionally under perf).
#
# Output files default to profile-specific names so profiling runs don't overwrite
# normal runs (scheduler_output.json / kernel_to_validate.json):
#   --output-json scheduler_output_profile.json
#   --save-kernel kernel_profile.json
#
# Pass any generate_ssa_kernel.py args after -- (or set EXTRA_ARGS). If you pass
# your own --output-json or --save-kernel, those are used instead.
#
# Usage:
#   ./scripts/profile_kernel_generation.sh -- --rounds 5 --batch-size 256 --tree-reduce-max-depth 3 --cpsat-time-limit 300 --validate
#
# With perf (record only; report with: perf report -g or perf script | flamegraph):
#   USE_PERF=1 ./scripts/profile_kernel_generation.sh -- --rounds 5 --batch-size 256 --tree-reduce-max-depth 3 --cpsat-time-limit 300
#
# For useful Rust symbols in perf, build scheduler_rs with debuginfo first:
#   RUSTFLAGS="-C debuginfo=2" BUILD_FROM_PROJECT_ROOT=1 uv run maturin develop --manifest-path scheduler_rs/Cargo.toml
set -e
cd "$(dirname "$0")/.."

PROFILE_JSON="${PROFILE_JSON:-scheduler_output_profile.json}"
PROFILE_KERNEL="${PROFILE_KERNEL:-kernel_profile.json}"
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "--" ]]; then
    shift
    EXTRA_ARGS+=("$@")
    break
  fi
  shift
done

# Default profile output paths (overridden if user passes --output-json / --save-kernel in EXTRA_ARGS)
args=(
  --output-json "$PROFILE_JSON"
  --save-kernel "$PROFILE_KERNEL"
  "${EXTRA_ARGS[@]}"
)

if [[ "${USE_PERF}" == "1" ]]; then
  echo "Profiling with perf; output: ${PROFILE_JSON} / ${PROFILE_KERNEL}"
  exec perf record -g -F 99 --call-graph dwarf -- uv run generate_ssa_kernel.py "${args[@]}"
else
  echo "Profile output files: ${PROFILE_JSON} / ${PROFILE_KERNEL}"
  exec uv run generate_ssa_kernel.py "${args[@]}"
fi
