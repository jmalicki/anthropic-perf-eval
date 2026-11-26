#!/usr/bin/env sh
# Run the project's test suites with the environment they need.
#
#   ./scripts/test.sh            # everything below
#   ./scripts/test.sh rust       # scheduler_rs unit tests (cargo, --features "cpsat deep-tests")
#   ./scripts/test.sh python     # pytest: every tests/test_*.py, plus solver/test_slot_utilization.py
#   ./scripts/test.sh proofs     # Z3: liveness theorems, rewrite identities, allocation conjectures, static makespan bounds, memory-alias rules
#   ./scripts/test.sh kernel     # greedy 2-round and 16-round builds with replay + validation
#   ./scripts/test.sh upstream   # 16-round greedy kernel on the unmodified upstream simulator and tests
#   ./scripts/test.sh ci-only    # pytest -m ci_only: 16-round pin builds (CI; not part of 'all')
#
# Test tiers. The plain suites above run everything (local default). CI splits them:
#   ./scripts/test.sh rust-shallow    # PR CI: rust without the deep-tests feature (slow solves #[ignore]d)
#   ./scripts/test.sh python-shallow  # PR CI: python without tests marked @pytest.mark.deep
#   ./scripts/test.sh python-deep     # deep CI: only tests marked @pytest.mark.deep
# A test is deep when it runs a CP-SAT solve, except two 2-round CP-SAT smoke tests kept shallow
# (CLAUDE.md, test tiers). Z3 proofs are milliseconds and stay shallow.
#
# Tests marked @pytest.mark.ci_only are deselected by every other pytest run (pyproject addopts).
#
# Extra arguments after the suite name are passed through, e.g.
#   ./scripts/test.sh rust -- test_vector_group
#   ./scripts/test.sh python tests/test_kernel_dataflow.py -k lane
#
# Python suites use the installed scheduler_rs extension; run ./scripts/build_scheduler_rs.sh
# after changing Rust code.
#
# Each suite runs while holding one of a few machine-wide test slots (scripts/test_slot.py),
# so agents in different worktrees do not all run suites at once and time each other out.
# PERF_EVAL_TEST_SLOTS sets the number of slots (default 2; 0 = no slot); CI takes none.
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
  cargo test --manifest-path scheduler_rs/Cargo.toml --features "cpsat deep-tests" "$@"
}
run_rust_shallow() {
  cargo test --manifest-path scheduler_rs/Cargo.toml --features cpsat "$@"
}
# run_python [-m MARKEXPR] [pytest args]: a leading -m replaces the default 'not ci_only'.
run_python() {
  marks="not ci_only"
  if [ "$1" = "-m" ]; then
    marks="$2"
    shift 2
  fi
  if [ $# -eq 0 ]; then
    # Discovery, not a list: pytest collects every tests/test_*.py, so a new test file runs in
    # CI without editing this script. tests/submission_tests.py does not match the pattern (it
    # is not run in CI, see CLAUDE.md). The one test file outside tests/ is named here;
    # tests/test_suite_coverage.py fails if another appears outside tests/.
    set -- tests solver/test_slot_utilization.py
  fi
  rc=0
  uv run pytest -q -m "$marks" "$@" || rc=$?
  if [ "$rc" -eq 5 ]; then
    echo "test.sh python: no test selected ($marks). Tests marked ci_only are deselected by this suite;" \
      "if that is what you named, run: ./scripts/test.sh ci-only $*" >&2
  fi
  return "$rc"
}
run_ci_only() {
  if [ $# -eq 0 ]; then
    set -- tests
  fi
  rc=0
  uv run pytest -q -m ci_only "$@" || rc=$?
  # Exit 5 = nothing collected: no ci_only test in the selection, which is not a failure.
  if [ "$rc" -eq 5 ]; then
    echo "test.sh ci-only: no ci_only test in $*" >&2
    rc=0
  fi
  return "$rc"
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

run_upstream() {
  ./scripts/check_upstream.sh "$@"
}

# Re-enter this script under a test slot: "./scripts/test.sh SUITE ..." runs
# "test_slot.py SUITE -- ./scripts/test.sh --in-slot SUITE ...".
if [ "$suite" = "--in-slot" ]; then
  suite="$1"
  shift
elif [ "$suite" != "all" ]; then
  exec python3 scripts/test_slot.py "$suite" -- "$(pwd)/scripts/test.sh" --in-slot "$suite" "$@"
fi

case "$suite" in
  rust) run_rust "$@" ;;
  rust-shallow) run_rust_shallow "$@" ;;
  python) run_python "$@" ;;
  python-shallow) run_python -m "not ci_only and not deep" "$@" ;;
  python-deep) run_python -m "deep and not ci_only" "$@" ;;
  proofs) run_proofs "$@" ;;
  kernel) run_kernel "$@" ;;
  upstream) run_upstream "$@" ;;
  ci-only) run_ci_only "$@" ;;
  all) for s in rust python proofs kernel upstream; do "$(pwd)/scripts/test.sh" "$s"; done ;;
  *) echo "unknown suite '$suite' (rust | rust-shallow | python | python-shallow | python-deep | proofs | kernel | upstream | ci-only | all)" >&2; exit 2 ;;
esac
