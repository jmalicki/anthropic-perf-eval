#!/usr/bin/env python3
"""
Run the Rust bulk-move simulator on an SSA schedule JSON.

Uses the same constraint logic as the CP-SAT model (precedence, ordering, domain).
The JSON must be from: uv run generate_ssa_kernel.py --output-json <path> (with CP-SAT).

Usage:
  uv run scripts/simulate_bulk_move.py [path]
  uv run scripts/simulate_bulk_move.py --verbose path.json

Default path: scheduler_output.json (must be SSA format with schedule_constraints).
"""

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Rust bulk-move simulator on SSA schedule JSON (same constraints as CP-SAT)."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="scheduler_output.json",
        help="Path to SSA schedule JSON (default: scheduler_output.json)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each bulk-move step (C and ops moved)",
    )
    args = parser.parse_args()

    try:
        import scheduler_rs
    except ImportError:
        print("scheduler_rs not found. Build with: BUILD_FROM_PROJECT_ROOT=1 uv run maturin develop --manifest-path scheduler_rs/Cargo.toml", file=sys.stderr)
        return 1

    try:
        result = scheduler_rs.run_bulk_move_simulator_from_json(args.path, verbose=args.verbose)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except OSError as e:
        print(f"IO error: {e}", file=sys.stderr)
        return 1

    kind = result.get("result", "")
    if kind == "NoEmptyCycles":
        print("No empty cycles.")
        return 0
    if kind == "ReachedEnd":
        print("Reached end: all empty cycles pushed to the end.")
        print(
            f"  Original makespan = {result.get('original_makespan')}, "
            f"final = {result.get('final_makespan')}, steps = {result.get('steps')}."
        )
        print("  Makespan could have been reduced; schedule had unnecessary empty cycles.")
        return 1
    if kind == "Blocked":
        print(
            f"Blocked at empty C={result.get('empty_c')}: {result.get('violation_message')}"
        )
        print(f"  Steps before block: {result.get('steps')}")
        return 1
    print(f"Unknown result: {result}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
