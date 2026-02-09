#!/usr/bin/env python3
"""Bisect CP-SAT validation failures by running all 8 combinations of choice options.

Runs generate_ssa_kernel.py with --validate -q for:
- One greedy baseline (expected to pass)
- Eight CP-SAT runs: each combination of --const-choice, --select-choice, --valu-shift
  (const_choice in {optimize, load}, select_choice in {optimize, vselect}, valu_shift in {alu, valu})

Uses forced options (alu, valu) for valu_shift rather than optimize, to isolate which choice correlates with failure.

For full axis options (choice + constraints + size), see scripts/bisect.py.

Usage (from project root):
    uv run python scripts/bisect_cpsat_validation.py
    uv run python scripts/bisect_cpsat_validation.py --rounds 2 --batch-size 256 --cpsat-time-limit 90
"""

import os
import subprocess
import sys


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bisect_py = os.path.join(project_root, "scripts", "bisect.py")
    if not os.path.isfile(bisect_py):
        print(f"Error: bisect.py not found at {bisect_py}", file=sys.stderr)
        return 1

    result = subprocess.run(
        [sys.executable, bisect_py, "--sweep-choice"] + sys.argv[1:],
        cwd=project_root,
    )
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
