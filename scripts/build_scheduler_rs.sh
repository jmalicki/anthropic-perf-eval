#!/usr/bin/env sh
# Build and install scheduler_rs into the project venv (used by uv run).
# Run from project root: ./scripts/build_scheduler_rs.sh
# Do not run maturin from inside scheduler_rs/ - this script sets BUILD_FROM_PROJECT_ROOT
# so the build succeeds and installs to the correct venv.
# Locates OR-Tools (for cp_sat dependency) and exports ORTOOLS_PREFIX so the cp_sat build finds headers.
# Use this script from any worktree or when invoking maturin directly; otherwise cp_sat build may
# default to /opt/ortools and fail with "ortools/sat/cp_model.h not found".
set -e
cd "$(dirname "$0")/.."

# Pass OR-Tools prefix to cp_sat build (cp_sat uses ORTOOLS_PREFIX, default /opt/ortools)
if [ -z "${ORTOOLS_PREFIX}" ]; then
  if command -v brew >/dev/null 2>&1; then
    ORTOOLS_PREFIX=$(brew --prefix or-tools 2>/dev/null || true)
  fi
  if [ -z "${ORTOOLS_PREFIX}" ] || [ ! -d "${ORTOOLS_PREFIX}/include" ]; then
    ORTOOLS_PREFIX=/opt/ortools
  fi
  export ORTOOLS_PREFIX
fi

# PyO3 picks the first python on PATH, which may be newer than it supports (e.g. a system
# 3.15 with PyO3 0.27). Build against the project venv's interpreter unless told otherwise.
if [ -z "${PYO3_PYTHON}" ] && [ -x ".venv/bin/python" ]; then
  export PYO3_PYTHON="$(pwd)/.venv/bin/python"
fi

BUILD_FROM_PROJECT_ROOT=1 uv run maturin develop --manifest-path scheduler_rs/Cargo.toml "$@"
