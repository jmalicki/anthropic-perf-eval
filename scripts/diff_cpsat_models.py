#!/usr/bin/env python3
"""Compare two exported CP-SAT models and report where they first differ.

Two builds of the same configuration must hand CP-SAT the same model: same variables,
constraints, hints and search strategy, in the same order. If they do not, the solver
explores differently and returns different schedules, and no two runs are comparable.

Export a model with CPSAT_EXPORT_MODEL:

    CPSAT_EXPORT_MODEL=a.pb uv run generate_ssa_kernel.py --rounds 2 --batch-size 16 \\
        --cpsat-deterministic-time 0.5 --no-validate -q
    CPSAT_EXPORT_MODEL=b.pb uv run generate_ssa_kernel.py ...same options...
    uv run python -m scripts.diff_cpsat_models a.pb b.pb

Exit status 0 if the models are identical, 1 otherwise.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

from ortools.sat import cp_model_pb2


def load(path: str) -> cp_model_pb2.CpModelProto:
    model = cp_model_pb2.CpModelProto()
    with open(path, "rb") as f:
        model.ParseFromString(f.read())
    return model


def _kind(constraint) -> str:
    return constraint.WhichOneof("constraint") or "empty"


def sections(model: cp_model_pb2.CpModelProto) -> dict[str, list[tuple[str, bytes]]]:
    """Ordered (label, serialised bytes) entries for each part of the model."""
    hint = model.solution_hint
    return {
        "variables": [(v.name or f"#{i}", v.SerializeToString()) for i, v in enumerate(model.variables)],
        "constraints": [
            (f"{c.name or '(unnamed)'} [{_kind(c)}]", c.SerializeToString()) for c in model.constraints
        ],
        "hints": [(f"var {var}", str(val).encode()) for var, val in zip(hint.vars, hint.values)],
        "search_strategy": [(f"strategy {i}", s.SerializeToString()) for i, s in enumerate(model.search_strategy)],
        "objective": [("objective", model.objective.SerializeToString())],
    }


def compare(a: cp_model_pb2.CpModelProto, b: cp_model_pb2.CpModelProto) -> list[str]:
    """Human-readable differences, one block per differing section (empty = identical)."""
    out: list[str] = []
    sa, sb = sections(a), sections(b)
    for name in sa:
        ea, eb = sa[name], sb[name]
        if ea == eb:
            continue
        first = next((i for i, (x, y) in enumerate(zip(ea, eb)) if x != y), min(len(ea), len(eb)))
        same_multiset = Counter(ea) == Counter(eb)
        verdict = (
            "same entries in a different order (nondeterministic construction order)"
            if same_multiset
            else "different entries"
        )
        out.append(f"{name}: {len(ea)} vs {len(eb)} entries; {verdict}; first difference at index {first}")
        for label, entries in (("a", ea), ("b", eb)):
            window = entries[first:first + 3]
            out.append(f"    {label}: " + ", ".join(lbl for lbl, _ in window) if window else f"    {label}: (end)")
        if not same_multiset:
            only_a = list((Counter(ea) - Counter(eb)).elements())[:3]
            only_b = list((Counter(eb) - Counter(ea)).elements())[:3]
            if only_a:
                out.append("    only in a: " + ", ".join(lbl for lbl, _ in only_a))
            if only_b:
                out.append("    only in b: " + ", ".join(lbl for lbl, _ in only_b))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("a", help="first exported model (.pb)")
    ap.add_argument("b", help="second exported model (.pb)")
    args = ap.parse_args()
    diffs = compare(load(args.a), load(args.b))
    if not diffs:
        print("models are identical")
        return 0
    print("models differ:")
    for line in diffs:
        print("  " + line)
    return 1


if __name__ == "__main__":
    sys.exit(main())
