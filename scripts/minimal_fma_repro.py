#!/usr/bin/env python3
"""Minimal reproducer for CP-SAT + SelectChoice FMA validation failure.

Smallest config that:
- Uses the full SSA kernel (tree + hash, so has SelectChoice ops)
- PASSES with greedy + fma
- FAILS with CP-SAT + fma (when validated against reference)

Usage (from project root):
    uv run python scripts/minimal_fma_repro.py              # CP-SAT + fma, validate (expect FAIL)
    uv run python scripts/minimal_fma_repro.py --greedy     # Greedy + fma (expect PASS)
    uv run python scripts/minimal_fma_repro.py --vselect    # CP-SAT + vselect (expect PASS when solver finds valid schedule)
"""

import argparse
import subprocess
import sys
import os


def main():
    ap = argparse.ArgumentParser(description="Minimal FMA validation reproducer")
    ap.add_argument("--greedy", action="store_true", help="Use greedy scheduler (then fma passes)")
    ap.add_argument("--vselect", action="store_true", help="Force vselect instead of fma")
    ap.add_argument("--seed", type=int, default=123, help="Random seed (default 123)")
    ap.add_argument("--rounds", type=int, default=2, help="Hash rounds (default 2)")
    ap.add_argument("--batch-size", type=int, default=16, help="Batch size (default 16)")
    ap.add_argument("--cpsat-time-limit", type=float, default=60.0, help="CP-SAT time limit when not --greedy")
    ap.add_argument("--save", type=str, default="kernel_minimal_fma.json", help="Save kernel to this path")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = os.path.join(root, "generate_ssa_kernel.py")
    if not os.path.isfile(script):
        print(f"Error: {script} not found", file=sys.stderr)
        return 1

    # Minimal params: rounds=2, batch_size=16 (default forest_height=10, n_nodes=2047)
    cmd = [
        sys.executable,
        script,
        "--rounds", str(args.rounds),
        "--batch-size", str(args.batch_size),
        "--seed", str(args.seed),
        "--valu-shift", "valu",  # avoid shift variability
        "--validate",
        "-q",
        "--save-kernel", args.save,
    ]
    if args.greedy:
        cmd.append("--use-greedy")
    else:
        cmd.extend(["--cpsat-time-limit", str(args.cpsat_time_limit)])
    if args.vselect:
        cmd.extend(["--select-choice", "vselect"])
    else:
        cmd.extend(["--select-choice", "fma"])

    print("Minimal FMA reproducer:")
    print(f"  rounds={args.rounds}, batch_size={args.batch_size}, seed={args.seed}")
    print(f"  scheduler={'greedy' if args.greedy else 'CP-SAT'}, select_choice={'vselect' if args.vselect else 'fma'}")
    print(f"  kernel save path: {args.save}")
    print()
    result = subprocess.run(cmd, cwd=root)
    if result.returncode != 0:
        print()
        print("Re-run with: uv run python scripts/minimal_fma_repro.py [--greedy|--vselect]")
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
