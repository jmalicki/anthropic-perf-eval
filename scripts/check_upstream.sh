#!/usr/bin/env sh
# Run a kernel from this repository against the unmodified upstream take-home tests
# (upstream/tests/, vendored from the commit in upstream/UPSTREAM_COMMIT).
#
#   ./scripts/check_upstream.sh                       # build 16-round greedy kernel, then test
#   ./scripts/check_upstream.sh --cpsat-time-limit 3600   # any generate_ssa_kernel.py build options
#   ./scripts/check_upstream.sh --kernel kernel.json  # test an already saved kernel, no build
#   ./scripts/check_upstream.sh --strict ...          # also fail if any upstream speed threshold fails
#
# Steps: (1) verify the vendored files against upstream/SHA256SUMS, (2) build and save the
# kernel unless --kernel is given, (3) run upstream's CorrectnessTests (gating), (4) run
# upstream's SpeedTests (reported; gating only with --strict).
#
# The kernel is built for forest_height=10, batch_size=256, rounds=16, the only
# configuration upstream tests. With no build options the greedy scheduler is used, which
# is a smoke check only; pass CP-SAT options for a recorded result.
set -e
cd "$(dirname "$0")/.."

kernel=""
strict=0
while [ $# -gt 0 ]; do
  case "$1" in
    --kernel) kernel="$2"; shift 2 ;;
    --strict) strict=1; shift ;;
    *) break ;;
  esac
done

echo "== Vendored upstream files =="
(cd upstream && shasum -a 256 -c SHA256SUMS)

if [ -z "$kernel" ]; then
  kernel="kernel_upstream_check.json"
  [ $# -eq 0 ] && set -- --use-greedy
  echo "== Building kernel: generate_ssa_kernel.py --rounds 16 $* =="
  uv run generate_ssa_kernel.py --forest-height 10 --batch-size 256 --rounds 16 -q \
    --output-json scheduler_output_upstream_check.json --save-kernel "$kernel" "$@"
fi
case "$kernel" in
  /*) ;;
  *) kernel="$(pwd)/$kernel" ;;
esac
export UPSTREAM_KERNEL_JSON="$kernel"

# -E -S: no site-packages and no PYTHON* environment, so only the standard library and
# upstream/ are importable; this repository's problem.py cannot be picked up.
py=python3
[ -x .venv/bin/python ] && py=.venv/bin/python

echo "== Upstream CorrectnessTests (kernel: $kernel) =="
"$py" -E -S upstream/tests/submission_tests.py CorrectnessTests

echo "== Upstream SpeedTests =="
if "$py" -E -S upstream/tests/submission_tests.py -v SpeedTests; then
  echo "All upstream speed thresholds pass."
elif [ "$strict" -eq 1 ]; then
  echo "Some upstream speed thresholds fail (--strict)." >&2
  exit 1
else
  echo "Correct on the upstream simulator; some speed thresholds fail (see above)."
fi
