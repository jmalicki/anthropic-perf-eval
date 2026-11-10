#!/usr/bin/env sh
# Run the project's test suites with the environment they need.
#
#   ./scripts/test.sh            # everything below
#   ./scripts/test.sh rust       # scheduler_rs unit tests (cargo, --features cpsat)
#   ./scripts/test.sh python     # pytest: interface, dataflow replay, forced modes, proofs
#   ./scripts/test.sh proofs     # Z3: liveness theorems and rewrite identities
#   ./scripts/test.sh kernel     # greedy 2-round and 16-round builds with replay + validation
#
# Extra arguments after the suite name are passed through, e.g.
#   ./scripts/test.sh rust -- test_vector_group
#   ./scripts/test.sh python tests/test_kernel_dataflow.py -k lane
#
# Python suites use the installed scheduler_rs extension; run ./scripts/build_scheduler_rs.sh
# after changing Rust code.
set -e
cd "$(dirname "$0")/.."

# Same environment as build_scheduler_rs.sh: OR-Tools for cp_sat, venv python for PyO3.
if [ -z "${ORTOOLS_PREFIX}" ] && command -v brew >/dev/null 2>&1; then
  ORTOOLS_PREFIX=$(brew --prefix or-tools 2>/dev/null || true)
fi
export ORTOOLS_PREFIX="${ORTOOLS_PREFIX:-/opt/ortools}"
if [ -z "${PYO3_PYTHON}" ] && [ -x ".venv/bin/python" ]; then
  export PYO3_PYTHON="$(pwd)/.venv/bin/python"
fi
export BUILD_FROM_PROJECT_ROOT=1

suite="${1:-all}"
[ $# -gt 0 ] && shift

run_rust() {
  cargo test --manifest-path scheduler_rs/Cargo.toml --features cpsat "$@"
}
run_python() {
  if [ $# -eq 0 ]; then
    set -- tests/test_kernel_dataflow.py tests/test_rewrite_proofs.py tests/test_forced_modes.py \
      tests/test_scheduler_rs_interface.py tests/test_kernel_bundle_consistency.py \
      tests/test_trace_io.py tests/test_trace_io_integration.py solver/test_slot_utilization.py
  fi
  uv run pytest -q "$@"
}
run_proofs() {
  uv run python -m verification.verify -q "$@"
}
run_kernel() {
  for rounds in 2 16; do
    uv run generate_ssa_kernel.py --rounds "$rounds" --use-greedy --validate -q \
      --output-json "scheduler_output_test_${rounds}r.json" \
      --save-kernel "kernel_test_${rounds}r.json" "$@"
  done
}

case "$suite" in
  rust) run_rust "$@" ;;
  python) run_python "$@" ;;
  proofs) run_proofs "$@" ;;
  kernel) run_kernel "$@" ;;
  all) run_rust; run_python; run_proofs; run_kernel ;;
  *) echo "unknown suite '$suite' (rust | python | proofs | kernel | all)" >&2; exit 2 ;;
esac
