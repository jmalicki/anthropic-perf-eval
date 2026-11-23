#!/usr/bin/env python3
"""Check that the solution hint of an exported CP-SAT model is a solution.

Every hinted variable is fixed to its hint and the fixed model is solved; FEASIBLE or
OPTIMAL means the hint satisfies every constraint. Unhinted variables stay free (check
completeness separately, e.g. tests/test_cpsat_hint_complete.py).

    CPSAT_EXPORT_MODEL=m.pb uv run generate_ssa_kernel.py --rounds 3 --batch-size 32 \\
        --cpsat-deterministic-time 0.2 --no-validate -q
    uv run python -m scripts.check_cpsat_hint m.pb

Run it in its own process: importing OR-Tools' Python solver next to the scheduler_rs
extension (which links its own OR-Tools) crashes the interpreter.

Exit status 0 if the hint is feasible, 1 otherwise.
"""

from __future__ import annotations

import argparse
import sys

from google.protobuf import text_format
from ortools.sat import cp_model_pb2
from ortools.sat.python import cp_model


def hint_status(model: cp_model_pb2.CpModelProto, time_limit: float = 60.0) -> str:
    """CP-SAT status of the model with every hinted variable fixed to its hint."""
    fixed = cp_model_pb2.CpModelProto()
    fixed.CopyFrom(model)
    for var, value in zip(fixed.solution_hint.vars, fixed.solution_hint.values):
        del fixed.variables[var].domain[:]
        fixed.variables[var].domain.extend([value, value])
    fixed.ClearField("solution_hint")
    fixed.ClearField("search_strategy")
    solver_model = cp_model.CpModel()
    solver_model.proto.parse_text_format(text_format.MessageToString(fixed))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = 1
    return solver.StatusName(solver.Solve(solver_model))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", nargs="+", help="exported model(s) (CPSAT_EXPORT_MODEL)")
    parser.add_argument("--time-limit", type=float, default=60.0)
    args = parser.parse_args()
    ok = True
    for path in args.model:
        model = cp_model_pb2.CpModelProto()
        with open(path, "rb") as f:
            model.ParseFromString(f.read())
        status = hint_status(model, args.time_limit)
        print(f"{path}: {status} ({len(model.solution_hint.vars)} of {len(model.variables)} variables hinted)")
        ok &= status in ("FEASIBLE", "OPTIMAL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
