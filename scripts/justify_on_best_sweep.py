#!/usr/bin/env python3
"""Justification on the best greedy kernels: the sweep behind docs/CRITICAL_PATH.md
("Justification on the best kernels"), as data.

    uv run python -m scripts.justify_on_best_sweep                    # rebuild, print, compare with the JSON
    uv run python -m scripts.justify_on_best_sweep --write            # rebuild and rewrite the JSON
    uv run python -m scripts.justify_on_best_sweep --only rewrites    # one table

Every row is one 16-round greedy build (deterministic), proved by dataflow replay and
validated against the reference; "fail" = the build exceeded the scratch budget. The JSON
(docs/emission_plans/justify_on_best_sweep.json) records each row's command line, so any row
can be rerun alone. tests/test_justify_on_best_sweep.py (ci_only) rebuilds the sweep and
compares. Greedy development measurements, not recorded benchmark results.
"""

from __future__ import annotations

import os
import sys

# Before any other import: run as a file, scripts/ would shadow stdlib modules (scripts/bisect.py).
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
sys.path.insert(0, ROOT)

import argparse  # noqa: E402
import json  # noqa: E402
import tempfile  # noqa: E402

DATA = os.path.join(ROOT, "docs", "emission_plans", "justify_on_best_sweep.json")

SHAPED = dict(rounds=16, use_greedy=True, greedy_barriers="none", greedy_priority="wavefront", greedy_split_shift=True,
              pressure_inflation_threshold=0.95, greedy_load_priority="final-last", greedy_groups_in_flight=32,
              greedy_stagger_cycles=9, merge_hash_stages=True, index_form="mirrored", greedy_justify=10)
PATH = dict(emission_plan="docs/emission_plans/path_flow90.json", select_bit_source="path")
INDEX = dict(emission_plan="docs/emission_plans/flow90.json")
PRIO = dict(greedy_flow_priority="chain-first", greedy_alu_priority="native-first")
R16 = dict(reassociate_gather_xor=True)
R17 = dict(index_early_terms="final")
R18 = dict(select_xor_pushdown="first-pass")
ALL3 = dict(R16, **R17, **R18)


def configurations() -> dict[str, list[tuple[str, dict]]]:
    """Table name -> [(row label, options on top of SHAPED)]."""
    sweep = []
    for stagger in (8, 9, 10):
        for in_flight in (24, 32):
            shape = dict(greedy_stagger_cycles=stagger, greedy_groups_in_flight=in_flight)
            sweep += [
                (f"stagger {stagger} / {in_flight}, path, prio", dict(PATH, **PRIO, **R16, greedy_const_policy="alu-first", **shape)),
                (f"stagger {stagger} / {in_flight}, path, no prio", dict(PATH, **R16, greedy_const_policy="alu-first", **shape)),
                (f"stagger {stagger} / {in_flight}, index, prio", dict(INDEX, **PRIO, **R17, **shape)),
                (f"stagger {stagger} / {in_flight}, index, no prio", dict(INDEX, **R17, **shape)),
            ]
    rewrites = []
    for label, rw in (("none", {}), ("R16", R16), ("R17 final", R17), ("R16 + R17 final", dict(R16, **R17)),
                      ("R18 first-pass", R18), ("R18 first-pass + R16 + R17 final", ALL3)):
        for policy in ("default", "alu-first"):
            rewrites += [(f"{label}, path, {policy}", dict(PATH, **PRIO, **rw, greedy_const_policy=policy)),
                         (f"{label}, index, {policy}", dict(INDEX, **rw, greedy_const_policy=policy))]
    neighbours = []
    for label, shape in (("best", {}), ("stagger 8", dict(greedy_stagger_cycles=8)),
                         ("stagger 10", dict(greedy_stagger_cycles=10)), ("28 in flight", dict(greedy_groups_in_flight=28)),
                         ("24 in flight", dict(greedy_groups_in_flight=24)), ("justify 0", dict(greedy_justify=0))):
        neighbours += [(f"{label}, path", dict(PATH, **PRIO, **ALL3, **shape)),
                       (f"{label}, index", dict(INDEX, **ALL3, **shape))]
    neighbours += [("path without the priorities", dict(PATH, **ALL3)),
                   ("index with the priorities", dict(INDEX, **PRIO, **ALL3))]
    return {"sweep": sweep, "rewrites": rewrites, "neighbours": neighbours}


def command(options: dict) -> str:
    """The generate_ssa_kernel.py command line of a row."""
    parts = ["uv run generate_ssa_kernel.py --validate"]
    for key, value in {**SHAPED, **options}.items():
        flag = "--" + key.replace("_", "-")
        if value is True:
            parts.append(flag)
        elif value is not False:
            parts.append(f"{flag} {value}")
    return " ".join(parts)


def build(options: dict) -> dict:
    """Build, prove and validate one row; {"cycles", "simulator_cycles", "peak_live",
    "justify"} or {"fail": message}."""
    import scheduler_rs
    from generate_ssa_kernel import simulator_cycles, validate_kernel
    from kernel_build_args import build_kernel_with_args
    from solver.emission_plan import expand_plan, load_plan
    from verification.kernel_dataflow import check

    opts = {**SHAPED, **options}
    opts["emission_plan"] = expand_plan(load_plan(os.path.join(ROOT, opts["emission_plan"])), 16, 32, 10)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "sched.json")
        try:
            instrs, stats = build_kernel_with_args(verbose=False, output_json=path, **opts)
        except scheduler_rs.ScratchBudgetExceeded as e:
            return {"fail": f"scratch budget exceeded (peak_live {e.args[1]} > {e.args[2]})"}
        with open(path) as f:
            sched = json.load(f)
    instrs = json.loads(json.dumps(list(instrs)))
    if stats["budget_exceeded"]:
        return {"fail": f"scratch budget exceeded (peak_live {stats.get('peak_live')})"}
    if not check({"instrs": instrs}, sched).ok:
        raise AssertionError(f"dataflow replay failed: {command(options)}")
    if not validate_kernel(instrs, 10, 256, 16, 123):
        raise AssertionError(f"validation failed: {command(options)}")
    return {"cycles": stats["cycles"], "simulator_cycles": simulator_cycles(instrs)[0],
            "peak_live": stats.get("peak_live"), "justify": stats.get("greedy_justify_summary")}


def run(tables=None, log=print) -> dict:
    out = {}
    for table, rows in configurations().items():
        if tables and table not in tables:
            continue
        out[table] = []
        for label, options in rows:
            res = build(options)
            out[table].append({"row": label, "command": command(options), **res})
            log(f"  {table}: {label}: {res.get('cycles', res.get('fail'))}")
    return out


def outcome(row: dict):
    return row.get("cycles", "fail" if "fail" in row else None)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", action="append", choices=list(configurations()), help="rebuild these tables only")
    ap.add_argument("--write", action="store_true", help="rewrite the JSON with the rebuilt rows")
    ap.add_argument("--data", default=DATA)
    args = ap.parse_args(argv)
    res = run(args.only)
    if args.write:
        recorded = {}
        if os.path.exists(args.data):
            with open(args.data) as f:
                recorded = json.load(f).get("tables", {})
        recorded.update(res)
        with open(args.data, "w") as f:
            json.dump({"note": __doc__.splitlines()[0], "tables": recorded}, f, indent=1)
        print(f"wrote {args.data}")
        return 0
    with open(args.data) as f:
        recorded = json.load(f)["tables"]
    diffs = [(t, a["row"], outcome(b), outcome(a)) for t, rows in res.items()
             for a, b in zip(rows, recorded[t]) if outcome(a) != outcome(b)]
    for t, row, was, now in diffs:
        print(f"CHANGED {t}: {row}: recorded {was}, now {now}")
    return 1 if diffs else 0


if __name__ == "__main__":
    sys.exit(main())
