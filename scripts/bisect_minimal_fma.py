#!/usr/bin/env python3
"""Bisect full-kernel config (batch_size, rounds, forest_height) to find a minimal FMA failure case.

Runs generate_ssa_kernel.py with --validate -q and --select-choice fma:
- Greedy + fma: expected to pass (baseline).
- CP-SAT + fma: we want to find the smallest config where this still FAILS (minimal repro).

Machine config (VALU/ALU slot counts) is NOT exposed via the CLI; the scheduler uses
MachineConfig::default() (6 VALU, 12 ALU, etc.). So we only sweep problem size here.

Usage (from project root):
    # Default: sweep rounds in [1,2], batch in [8,16,32], forest_height in [8,9,10]
    uv run python scripts/bisect_minimal_fma.py

    # Single combo (quick check)
    uv run python scripts/bisect_minimal_fma.py --rounds 2 --batch-size 16 --forest-height 10

    # Narrow sweep to find boundary
    uv run python scripts/bisect_minimal_fma.py --rounds 1 2 --batch-size 8 16 --forest-height 9 10

    # Shrink to find minimal failing (try halving until CP-SAT passes, then report)
    uv run python scripts/bisect_minimal_fma.py --shrink
"""

import argparse
import os
import subprocess
import sys


def n_nodes(forest_height: int) -> int:
    return (1 << forest_height) - 1


def run_one(
    project_root: str,
    script: str,
    rounds: int,
    batch_size: int,
    forest_height: int,
    use_greedy: bool,
    cpsat_time_limit: float,
    seed: int = 123,
) -> tuple[bool, str]:
    """Run generate_ssa_kernel.py with given params. Returns (success, stderr_snippet)."""
    cmd = [
        sys.executable,
        script,
        "--rounds",
        str(rounds),
        "--batch-size",
        str(batch_size),
        "--forest-height",
        str(forest_height),
        "--n-nodes",
        str(n_nodes(forest_height)),
        "--seed",
        str(seed),
        "--validate",
        "-q",
        "--select-choice",
        "fma",
        "--valu-shift",
        "valu",
    ]
    if use_greedy:
        cmd.append("--use-greedy")
    else:
        cmd.extend(["--cpsat-time-limit", str(cpsat_time_limit)])
    result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True, timeout=cpsat_time_limit + 30)
    snippet = ""
    if result.stderr:
        lines = result.stderr.strip().splitlines()
        snippet = "\n".join(lines[-8:]) if len(lines) > 8 else result.stderr.strip()
    return result.returncode == 0, snippet


def main():
    parser = argparse.ArgumentParser(
        description="Bisect (batch_size, rounds, forest_height) to find minimal CP-SAT + FMA validation failure"
    )
    parser.add_argument("--rounds", type=int, nargs="+", default=[1, 2], help="Rounds to try (default: 1 2)")
    parser.add_argument(
        "--batch-size",
        type=int,
        nargs="+",
        default=[8, 16, 32],
        help="Batch sizes to try (default: 8 16 32)",
    )
    parser.add_argument(
        "--forest-height",
        type=int,
        nargs="+",
        default=[8, 9, 10],
        help="Forest heights to try (default: 8 9 10)",
    )
    parser.add_argument("--cpsat-time-limit", type=float, default=90.0, help="CP-SAT time limit per run (default: 90)")
    parser.add_argument("--seed", type=int, default=123, help="Random seed (default: 123)")
    parser.add_argument(
        "--shrink",
        action="store_true",
        help="Shrink from (rounds=2, batch=16, height=10) until CP-SAT passes; report minimal failing",
    )
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = os.path.join(project_root, "generate_ssa_kernel.py")
    if not os.path.isfile(script):
        print(f"Error: generate_ssa_kernel.py not found at {script}", file=sys.stderr)
        return 1

    if args.shrink:
        # Start from known failing config (2, 16, 10); try smaller configs and report pass/fail
        r0, b0, h0 = 2, 16, 10
        print("Shrink mode: try smaller configs to find minimal failing (rounds, batch_size, forest_height)")
        print(f"  Reference: rounds={r0}, batch_size={b0}, forest_height={h0}")
        print()

        # Greedy baseline at reference
        ok, _ = run_one(project_root, script, r0, b0, h0, True, args.cpsat_time_limit, args.seed)
        if not ok:
            print("  Greedy + fma failed at reference; cannot use as baseline.", file=sys.stderr)
            return 1

        # Try each dimension smaller; record pass/fail
        print("  Varying batch_size (rounds=2, forest_height=10):")
        for batch in [16, 8, 4, 2]:
            ok, _ = run_one(project_root, script, r0, batch, h0, False, args.cpsat_time_limit, args.seed)
            print(f"    batch_size={batch}: CP-SAT+fma {'PASS' if ok else 'FAIL'}")

        print("  Varying rounds (batch_size=16, forest_height=10):")
        for rounds in [2, 1]:
            ok, _ = run_one(project_root, script, rounds, b0, h0, False, args.cpsat_time_limit, args.seed)
            print(f"    rounds={rounds}: CP-SAT+fma {'PASS' if ok else 'FAIL'}")

        print("  Varying forest_height (rounds=2, batch_size=16):")
        for height in [10, 9, 8, 7]:
            ok, _ = run_one(project_root, script, r0, b0, height, False, args.cpsat_time_limit, args.seed)
            print(f"    forest_height={height}: CP-SAT+fma {'PASS' if ok else 'FAIL'}")

        print()
        print("To reproduce minimal failure, use the smallest failing combo, e.g.:")
        print("  uv run python scripts/minimal_fma_repro.py --rounds 2 --batch-size 16")
        return 0

    # Grid sweep
    rounds_list = sorted(set(args.rounds))
    batch_list = sorted(set(args.batch_size))
    height_list = sorted(set(args.forest_height))

    print("Bisect full kernel (CP-SAT + FMA validation failure)")
    print(f"  Rounds: {rounds_list}, Batch sizes: {batch_list}, Forest heights: {height_list}")
    print(f"  Seed: {args.seed}, CP-SAT time limit: {args.cpsat_time_limit}s")
    print("  Machine config: not exposed (using default 6 VALU, 12 ALU, etc.)")
    print()

    # Greedy baseline at one config
    r0, b0, h0 = rounds_list[-1], batch_list[-1], height_list[-1]
    print(f"Greedy baseline (rounds={r0}, batch={b0}, forest_height={h0})...", flush=True)
    ok, _ = run_one(project_root, script, r0, b0, h0, True, args.cpsat_time_limit, args.seed)
    print(f"  {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("  Greedy failed; fix baseline before bisecting.", file=sys.stderr)
        return 1
    print()

    results = []
    for rounds in rounds_list:
        for batch in batch_list:
            for height in height_list:
                key = (rounds, batch, height)
                ok, snippet = run_one(
                    project_root, script, rounds, batch, height, False, args.cpsat_time_limit, args.seed
                )
                results.append((*key, "PASS" if ok else "FAIL", snippet))
                status = "PASS" if ok else "FAIL"
                print(f"  rounds={rounds} batch={batch} forest_height={height}  CP-SAT+fma: {status}")

    print()
    print("Summary (CP-SAT + fma):")
    failed = [(r, b, h) for r, b, h, s, _ in results if s == "FAIL"]
    passed = [(r, b, h) for r, b, h, s, _ in results if s == "PASS"]
    print(f"  Passed: {len(passed)}, Failed: {len(failed)}")
    if failed:
        print("  Failing configs (rounds, batch_size, forest_height):")
        for (r, b, h) in sorted(failed):
            print(f"    ({r}, {b}, {h})")
        # Smallest failing by (rounds, batch, height) lex order
        r, b, h = min(failed, key=lambda t: (t[0], t[1], t[2]))
        print(f"  Smallest failing (lex): rounds={r}, batch_size={b}, forest_height={h}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
