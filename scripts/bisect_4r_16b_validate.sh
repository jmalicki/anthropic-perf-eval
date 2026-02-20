#!/usr/bin/env bash
# Run 4-round 16-batch kernel build + validate for git bisect.
# Exit 0 = pass (good), non-zero = fail (bad).
# Use from project root:
#   git bisect start
#   git bisect bad HEAD
#   git bisect good <known-good-rev>
#   git bisect run ./scripts/bisect_4r_16b_validate.sh

set -e
cd "$(git rev-parse --show-toplevel)"

echo "=== Building scheduler_rs ==="
BUILD_FROM_PROJECT_ROOT=1 uv run maturin develop --manifest-path scheduler_rs/Cargo.toml -q 2>/dev/null || true

echo "=== Running 4r/16b with --validate ==="
uv run generate_ssa_kernel.py --rounds 4 --batch-size 16 --validate -q --cpsat-time-limit 300
exit 0
