#!/usr/bin/env python3
"""
CLI tool for kernel validation.

Usage:
    python solver/validate_kernel.py --rounds 4 --batch-size 8
    python solver/validate_kernel.py --verbose
    python solver/validate_kernel.py --final-only
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Validate kernel against reference implementation"
    )
    parser.add_argument(
        "--rounds", type=int, default=16,
        help="Number of traversal rounds (default: 16)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="Batch size (default: 256)"
    )
    parser.add_argument(
        "--forest-height", type=int, default=6,
        help="Tree height (default: 6)"
    )
    parser.add_argument(
        "--seed", type=int, default=123,
        help="Random seed (default: 123)"
    )
    parser.add_argument(
        "--max-failures", type=int, default=10,
        help="Maximum failures to collect (default: 10)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--final-only", action="store_true",
        help="Only validate final output (no debug instructions)"
    )
    parser.add_argument(
        "--all-failures", action="store_true",
        help="Show all collected failures (not just first)"
    )

    args = parser.parse_args()

    # Import here to avoid issues with PYTHONPATH
    from solver.debug_validator import validate_kernel, validate_final_output

    print(f"Kernel Validation")
    print(f"  Rounds: {args.rounds}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Forest height: {args.forest_height}")
    print(f"  Seed: {args.seed}")
    print()

    if args.final_only:
        print("Running final output validation...")
        passed, mismatches = validate_final_output(
            rounds=args.rounds,
            batch_size=args.batch_size,
            forest_height=args.forest_height,
            seed=args.seed,
            verbose=args.verbose,
        )

        if passed:
            print("PASSED: All final outputs match reference")
            return 0
        else:
            print(f"FAILED: {len(mismatches)} mismatches found")
            for name, expected, got in mismatches[:10]:
                print(f"  {name}: expected=0x{expected:08X}, got=0x{got:08X}")
            if len(mismatches) > 10:
                print(f"  ... and {len(mismatches) - 10} more")
            return 1

    print("Running debug validation...")
    result = validate_kernel(
        rounds=args.rounds,
        batch_size=args.batch_size,
        forest_height=args.forest_height,
        seed=args.seed,
        max_failures=args.max_failures,
        verbose=args.verbose,
    )

    print()
    print(result.summary())

    if not result.passed and args.all_failures:
        print()
        print("All failures:")
        for i, failure in enumerate(result.failures):
            print(f"\n[{i+1}] {failure}")

    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
