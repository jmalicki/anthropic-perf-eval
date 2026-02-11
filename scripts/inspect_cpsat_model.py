#!/usr/bin/env python3
"""
Inspect a CP-SAT model exported on INFEASIBLE (binary .pb format).

Use when CP-SAT reports INFEASIBLE and you've run with CPSAT_EXPORT_INFEASIBLE=1
to export the model. This script loads the model and prints a summary: constraint
count/type, cumulative capacity/demands, variable stats.

Usage (from project root):
  uv run python -m scripts.inspect_cpsat_model cpsat_infeasible_model.pb
  uv run python -m scripts.inspect_cpsat_model cpsat_infeasible_model.pb --validate

Run as module (-m) to avoid shadowing stdlib 'bisect' by scripts/bisect.py.
Requires ortools (installed with the project).
"""

import argparse
import sys

# Ensure project root is on path
import os
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect a CP-SAT model exported on INFEASIBLE (.pb binary)"
    )
    parser.add_argument(
        "model_pb",
        help="Path to exported model file (e.g. cpsat_infeasible_model.pb)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run ortools validation and print any errors",
    )
    args = parser.parse_args()

    try:
        from ortools.sat.python import cp_model
    except ImportError as e:
        print(f"Error: ortools not installed: {e}", file=sys.stderr)
        print("Install with: uv pip install ortools", file=sys.stderr)
        return 1

    if not os.path.isfile(args.model_pb):
        print(f"Error: file not found: {args.model_pb}", file=sys.stderr)
        print(
            "Export a model with: CPSAT_EXPORT_INFEASIBLE=1 uv run generate_ssa_kernel.py ...",
            file=sys.stderr,
        )
        return 1

    with open(args.model_pb, "rb") as f:
        data = f.read()

    from ortools.sat import cp_model_pb2

    proto = cp_model_pb2.CpModelProto()
    try:
        proto.ParseFromString(data)
    except Exception as e:
        print(f"Error: failed to parse model: {e}", file=sys.stderr)
        return 1

    # Variables
    n_vars = len(proto.variables)
    print(f"Variables: {n_vars}")

    # Constraints by type
    constraints = proto.constraints
    n_constraints = len(constraints)
    print(f"Constraints: {n_constraints}")

    # Cumulative constraints (embedded in constraints oneof)
    cumul_list = []
    for i, c in enumerate(constraints):
        if c.HasField("cumulative"):
            cu = c.cumulative
            cap = "?"
            if cu.HasField("capacity"):
                cap_expr = cu.capacity
                if not cap_expr.vars and hasattr(cap_expr, "offset"):
                    cap = str(cap_expr.offset)
                else:
                    cap = "(expr)"
            intervals = list(cu.intervals) if cu.intervals else []
            demands = list(cu.demands) if cu.demands else []
            n_int = len(intervals)
            n_dem = len(demands)
            demand_vals = []
            for d in demands:
                if not d.vars and hasattr(d, "offset"):
                    demand_vals.append(d.offset)
                else:
                    demand_vals.append(None)
            valid_d = [x for x in demand_vals if x is not None]
            max_d = max(valid_d) if valid_d else "?"
            min_d = min(valid_d) if valid_d else "?"
            name = c.name if c.name else f"cumul_{i}"
            cumul_list.append((name, n_int, n_dem, cap, min_d, max_d))
    print(f"\nCumulative constraints: {len(cumul_list)}")
    for name, n_int, n_dem, cap, min_d, max_d in cumul_list:
        print(f"  {name}: intervals={n_int} demands={n_dem} capacity={cap} demand_range=[{min_d},{max_d}]")

    # Constraint names
    named = sum(1 for c in constraints if c.name)
    if named > 0:
        print(f"\nNamed constraints: {named}/{n_constraints}")

    # Validate if requested
    if args.validate:
        try:
            model = cp_model.CpModel()
            model.Proto().CopyFrom(proto)
            if hasattr(model, "Validate"):
                validation_result = model.Validate()
                if validation_result:
                    print(f"\nValidation error: {validation_result}")
                    return 1
                else:
                    print("\nValidation: OK")
            else:
                print("\nValidation: (Validate not available in this ortools version)")
        except Exception as e:
            print(f"\nValidation error: {e}")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
