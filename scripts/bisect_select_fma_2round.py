#!/usr/bin/env python3
"""Minimal repro for 2-round SelectChoice FMA validation failure.

Bisect results:
- rounds=1, select=fma: PASS (no SelectChoice ops in 1-round kernel)
- rounds=2, select=vselect: PASS
- rounds=2, select=fma: FAIL (value mismatch at batch index 1)
- rounds=2, select=optimize + shift=valu: FAIL when solver picks FMA for any SelectChoice

So the bug is in the SelectChoice FMA path when rounds >= 2. This script runs
the minimal failing config for debugging.

Usage (from project root):
  uv run python scripts/bisect_select_fma_2round.py
"""
import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Minimal SelectChoice FMA repro")
    parser.add_argument("--rounds", type=int, default=2, help="Number of rounds (default 2)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default 8)")
    parser.add_argument("--cpsat-time-limit", type=float, default=60.0)
    parser.add_argument("--no-validate", action="store_true", help="Skip validation (build only)")
    args = parser.parse_args()

    if args.batch_size % 8 != 0:
        print("batch_size must be a multiple of 8", file=sys.stderr)
        return 1

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = os.path.join(root, "generate_ssa_kernel.py")
    cmd = [
        sys.executable,
        script,
        "--rounds", str(args.rounds),
        "--batch-size", str(args.batch_size),
        "--const-choice", "optimize",
        "--select-choice", "fma",
        "--valu-shift", "valu",
        "--cpsat-time-limit", str(args.cpsat_time_limit),
        "-q",
    ]
    if not args.no_validate:
        cmd.append("--validate")

    print("Running: rounds={} batch_size={} select_choice=fma valu_shift=valu".format(args.rounds, args.batch_size))
    result = subprocess.run(cmd, cwd=root, text=True)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
