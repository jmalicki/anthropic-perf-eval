#!/usr/bin/env python3
"""Bisect CP-SAT validation failures by running all 8 combinations of choice options.

Runs generate_ssa_kernel.py with --validate -q for:
- One greedy baseline (expected to pass)
- Eight CP-SAT runs: each combination of --const-choice, --select-choice, --valu-shift
  (const_choice in {optimize, load}, select_choice in {optimize, vselect}, valu_shift in {optimize, valu})

Reports which combinations pass or fail so we can see which choice (if any) correlates with the failure.

Usage (from project root):
    uv run python scripts/bisect_cpsat_validation.py
    uv run python scripts/bisect_cpsat_validation.py --rounds 2 --batch-size 256 --cpsat-time-limit 90
"""

import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Bisect CP-SAT validation: run greedy + 8 CP-SAT flag combinations and report pass/fail"
    )
    parser.add_argument("--rounds", type=int, default=2, help="Hash rounds (default: 2)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument(
        "--cpsat-time-limit",
        type=float,
        default=60.0,
        help="CP-SAT time limit in seconds (default: 60)",
    )
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = os.path.join(project_root, "generate_ssa_kernel.py")
    if not os.path.isfile(script):
        print(f"Error: generate_ssa_kernel.py not found at {script}", file=sys.stderr)
        return 1

    base_cmd = [
        sys.executable,
        script,
        "--rounds",
        str(args.rounds),
        "--batch-size",
        str(args.batch_size),
        "--validate",
        "-q",
    ]

    # Greedy baseline
    print("Greedy (baseline)...", flush=True)
    cmd = base_cmd + ["--use-greedy"]
    result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)
    greedy_ok = result.returncode == 0
    print(f"  Greedy (baseline): {'PASS' if greedy_ok else 'FAIL'}")
    if not greedy_ok and result.stderr:
        for line in result.stderr.strip().splitlines()[-5:]:
            print(f"    {line}")
    print()

    # CP-SAT: 8 combinations (const_choice, select_choice, valu_shift)
    print("CP-SAT (8 combinations of choice options):")
    rows = []
    for const_choice in ("optimize", "load"):
        for select_choice in ("optimize", "vselect"):
            for valu_shift in ("optimize", "valu"):
                cmd = base_cmd + [
                    "--cpsat-time-limit",
                    str(args.cpsat_time_limit),
                    "--const-choice",
                    const_choice,
                    "--select-choice",
                    select_choice,
                    "--valu-shift",
                    valu_shift,
                ]

                result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)
                ok = result.returncode == 0
                label = f"const={const_choice} select={select_choice} shift={valu_shift}"
                status = "PASS" if ok else "FAIL"
                rows.append((label, status))
                print(f"  {label}  {status}")

    print()
    print("Interpretation:")
    if not greedy_ok:
        print("  Greedy failed; it is expected to pass for this config. Check environment.")
    passed = sum(1 for _, s in rows if s == "PASS")
    if passed == 0:
        print("  All 8 CP-SAT combinations failed → bug likely in core CP-SAT schedule (cycle assignment/precedence).")
    elif passed == 8:
        print("  All 8 CP-SAT combinations passed.")
    else:
        print(f"  {passed}/8 CP-SAT passed. Compare passing vs failing rows to see which choice(s) correlate with failure.")

    return 0 if greedy_ok else 1


if __name__ == "__main__":
    sys.exit(main())
