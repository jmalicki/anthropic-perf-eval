#!/usr/bin/env python3
"""
Inspect active set (binding constraints) from a saved schedule JSON.

Reimplements binding_schedule_constraints from scheduler_rs: precedence, ordering,
and domain constraints that hold with equality at the CP-SAT solution.

Usage (from project root):
  uv run python -m scripts.inspect_binding_constraints scheduler_output.json

Run as module (-m) to avoid shadowing stdlib 'bisect' by scripts/bisect.py.

Input: Scheduler output JSON with schedule_constraints, ssa_cycle_assignments
(CP-SAT run with --output-json). shifted_alu_cycles is optional but needed for
correct dep_effective_cycle when dep_is_shifted.
"""

import argparse
import json
import sys


def dep_effective_cycle(
    dep_idx: int,
    dep_is_shifted: bool,
    cycle_assignments: list[int],
    shifted_alu_cycles: dict[int, list[int]],
    ops: list[dict],
) -> int:
    """Compute effective completion cycle for dep (handles shifted VALU)."""
    if dep_idx >= len(cycle_assignments):
        return -1
    if not dep_is_shifted:
        return cycle_assignments[dep_idx]
    # dep is shifted: use max of ALU cycles, else cycle_assignments
    op_id = ops[dep_idx]["id"] if dep_idx < len(ops) else None
    if op_id is None:
        return cycle_assignments[dep_idx]
    alu_cycles = shifted_alu_cycles.get(op_id)
    if alu_cycles:
        return max(alu_cycles)
    return cycle_assignments[dep_idx]


def binding_schedule_constraints(
    ops: list[dict],
    cycle_assignments: list[int],
    shifted_alu_cycles: dict[int, list[int]],
    constraints: list[dict],
) -> list[tuple[dict, str]]:
    """Return (constraint, description) for constraints that are binding."""
    out = []
    for c in constraints:
        if "Precedence" in c:
            p = c["Precedence"]
            consumer_idx = p["consumer_idx"]
            dep_idx = p["dep_idx"]
            result_offset = p["result_offset"]
            dep_is_shifted = p.get("dep_is_shifted", False)
            consumer_cycle = (
                cycle_assignments[consumer_idx]
                if consumer_idx < len(cycle_assignments)
                else -1
            )
            dep_cycle = dep_effective_cycle(
                dep_idx, dep_is_shifted, cycle_assignments, shifted_alu_cycles, ops
            )
            if consumer_cycle == dep_cycle + result_offset:
                out.append(
                    (
                        c,
                        f"consumer {consumer_idx} cycle {consumer_cycle} = dep {dep_idx} effective {dep_cycle} + {result_offset}",
                    )
                )
        elif "Ordering" in c:
            o = c["Ordering"]
            before_idx = o["before_idx"]
            after_idx = o["after_idx"]
            before_c = (
                cycle_assignments[before_idx]
                if before_idx < len(cycle_assignments)
                else -1
            )
            after_c = (
                cycle_assignments[after_idx]
                if after_idx < len(cycle_assignments)
                else -1
            )
            if before_c == after_c:
                out.append(
                    (
                        c,
                        f"before {before_idx} cycle {before_c} = after {after_idx} cycle {after_c}",
                    )
                )
        elif "Domain" in c:
            d = c["Domain"]
            idx = d["idx"]
            lb = d["lb"]
            ub = d["ub"]
            cy = cycle_assignments[idx] if idx < len(cycle_assignments) else -1
            if cy == lb or cy == ub:
                out.append(
                    (
                        c,
                        f"op {idx} cycle {cy} at bound (lb={lb}, ub={ub})",
                    )
                )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect binding constraints from SSA schedule JSON"
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="scheduler_output.json",
        help="Path to scheduler output JSON (default: scheduler_output.json)",
    )
    args = parser.parse_args()

    try:
        with open(args.path) as f:
            data = json.load(f)
    except OSError as e:
        print(f"Error: cannot read {args.path}: {e}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON in {args.path}: {e}", file=sys.stderr)
        return 1

    constraints = data.get("schedule_constraints")
    cycle_assignments = data.get("ssa_cycle_assignments")
    if not constraints:
        print("Error: no schedule_constraints in JSON", file=sys.stderr)
        print("Re-run with CP-SAT and --output-json", file=sys.stderr)
        return 1
    if not cycle_assignments:
        print("Error: no ssa_cycle_assignments in JSON", file=sys.stderr)
        print("Re-run with CP-SAT and --output-json", file=sys.stderr)
        return 1

    # shifted_alu_cycles: JSON keys are strings (op id)
    raw_shifted = data.get("shifted_alu_cycles") or {}
    shifted_alu_cycles = {int(k): v for k, v in raw_shifted.items()}

    # ops: use ssa_ops_bulk_move if present (id, dep_ids), else ops (id, ...)
    ops = data.get("ssa_ops_bulk_move") or data.get("ops")
    if not ops:
        print("Warning: no ssa_ops_bulk_move or ops; dep_effective_cycle for shifted deps may be wrong", file=sys.stderr)
        ops = [{"id": i} for i in range(len(cycle_assignments))]

    binding = binding_schedule_constraints(
        ops, cycle_assignments, shifted_alu_cycles, constraints
    )

    max_cycle = max(cycle_assignments) if cycle_assignments else -1
    n_prec = sum(1 for c, _ in binding if "Precedence" in c)
    n_ord = sum(1 for c, _ in binding if "Ordering" in c)
    n_dom = sum(1 for c, _ in binding if "Domain" in c)

    print(f"Max cycle (makespan-1): {max_cycle}")
    print(f"Binding constraints: {len(binding)} total")
    print(f"  Precedence: {n_prec}")
    print(f"  Ordering: {n_ord}")
    print(f"  Domain: {n_dom}")

    # Ops at last cycle
    at_last = [i for i, cy in enumerate(cycle_assignments) if cy == max_cycle]
    print(f"\nOps at last cycle ({max_cycle}): {len(at_last)}")
    if at_last and len(at_last) <= 30:
        print(f"  indices: {at_last}")
    elif at_last:
        print(f"  indices (first 30): {at_last[:30]} ...")

    # Sample binding constraints
    print("\nSample binding constraints (first 20):")
    for c, desc in binding[:20]:
        print(f"  {desc}")

    if len(binding) > 20:
        print(f"  ... and {len(binding) - 20} more")

    return 0


if __name__ == "__main__":
    sys.exit(main())
