#!/usr/bin/env sh
# Build and install scheduler_rs into the project venv (used by uv run).
# Run from project root: ./scripts/build_scheduler_rs.sh
# Do not run maturin from inside scheduler_rs/ - this script sets BUILD_FROM_PROJECT_ROOT
# so the build succeeds and installs to the correct venv.
set -e
cd "$(dirname "$0")/.."
BUILD_FROM_PROJECT_ROOT=1 uv run maturin develop --manifest-path scheduler_rs/Cargo.toml "$@"
