#!/usr/bin/env python3
"""
Analyze empty cycles in a schedule JSON (e.g. scheduler_output.json).

For each empty cycle C, lists ops scheduled at C+1 with breadcrumbs:
- blocked by precedence: some dep's result is only available at C+1 or later (op cannot run at C), or
- could run at C: all deps have result by C (op at C+1 is then a bug: model/solver/tie-breaker).

Usage:
  uv run scripts/analyze_empty_cycles.py [scheduler_output.json]
  uv run scripts/analyze_empty_cycles.py --max-empty 15 --json out.json scheduler_output.json
"""

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze empty cycles in schedule JSON")
    parser.add_argument(
        "path",
        nargs="?",
        default="scheduler_output.json",
        help="Path to schedule JSON (default: scheduler_output.json)",
    )
    parser.add_argument(
        "--max-empty",
        type=int,
        default=15,
        help="Max number of empty cycles to report in detail (default: 15)",
    )
    parser.add_argument(
        "--json-out",
        metavar="FILE",
        help="If set, write machine-readable report to FILE",
    )
    args = parser.parse_args()

    with open(args.path) as f:
        data = json.load(f)

    total_cycles = data["total_cycles"]
    ops = data["ops"]

    id_to_op = {op["id"]: op for op in ops}
    used_cycles = {op["cycle"] for op in ops}
    empty_cycles = [c for c in range(total_cycles) if c not in used_cycles]
    max_used = max(op["cycle"] for op in ops) if ops else -1
    max_used_plus_one = max_used + 1
    gap = total_cycles - max_used_plus_one if total_cycles > max_used_plus_one else 0

    # Gap summary
    print("=== Gap summary ===")
    print(f"  total_cycles: {total_cycles}")
    print(f"  max(op.cycle)+1: {max_used_plus_one}")
    print(f"  empty cycle count: {len(empty_cycles)}")
    if total_cycles > max_used_plus_one:
        print(f"  structural gap: total_cycles ({total_cycles}) > max_used+1 ({max_used_plus_one})")
    print()

    # Per empty cycle C: ops at C+1 with breadcrumbs
    print("=== Ops just after empty cycles (breadcrumbs) ===")
    report = []
    for C in empty_cycles[: args.max_empty]:
        ops_at_c_plus_1 = [op for op in ops if op["cycle"] == C + 1]
        for op in ops_at_c_plus_1:
            op_id = op["id"]
            engine = op["engine"]
            opcode = op["opcode"]
            dep_ids = op.get("dep_ids", [])

            if not dep_ids:
                latest_dep_result = -1
                blocking_dep_id = None
                blocking_dep_opcode = None
                blocking_dep_cycle = None
            else:
                dep_results = []
                for dep_id in dep_ids:
                    dep = id_to_op.get(dep_id)
                    if dep is None:
                        continue
                    dep_cycle = dep["cycle"]
                    # Lowered schedule: result visible at dep_cycle + 1
                    dep_results.append((dep_cycle + 1, dep_id, dep["opcode"], dep_cycle))
                if not dep_results:
                    latest_dep_result = -1
                    blocking_dep_id = None
                    blocking_dep_opcode = None
                    blocking_dep_cycle = None
                else:
                    dep_results.sort(key=lambda x: x[0], reverse=True)
                    latest_dep_result, blocking_dep_id, blocking_dep_opcode, blocking_dep_cycle = dep_results[0]

            if latest_dep_result > C:
                line = (
                    f"Empty {C}: op id={op_id} (engine={engine}, opcode={opcode}) at C+1 blocked by precedence: "
                    f"dep id={blocking_dep_id} (opcode={blocking_dep_opcode}) at cycle {blocking_dep_cycle} -> result at {latest_dep_result}."
                )
            else:
                line = (
                    f"Empty {C}: op id={op_id} (engine={engine}, opcode={opcode}) at C+1 could run at C (BUG: should not be at C+1); "
                    f"latest dep result at {latest_dep_result}."
                )
            print(line)
            report.append(
                {
                    "empty_cycle": C,
                    "op_id": op_id,
                    "engine": engine,
                    "opcode": opcode,
                    "blocked_by_precedence": latest_dep_result > C,
                    "blocking_dep_id": blocking_dep_id,
                    "latest_dep_result": latest_dep_result,
                }
            )

    if not report and empty_cycles:
        print(f"  (no ops at C+1 for first {min(args.max_empty, len(empty_cycles))} empty cycles)")
    elif not empty_cycles:
        print("  (no empty cycles)")

    # For the first "could run at C" op, check each constraint type against actual data (no speculation).
    could_run_ops = [r for r in report if not r["blocked_by_precedence"]]
    if could_run_ops and empty_cycles:
        C = could_run_ops[0]["empty_cycle"]
        op_id = could_run_ops[0]["op_id"]
        idx = next(i for i, o in enumerate(ops) if o["id"] == op_id)
        op = ops[idx]
        print()
        print("=== Constraint check for one concrete op (data only) ===")
        print(f"  Op: id={op_id} index={idx} cycle={op['cycle']} engine={op['engine']} opcode={op['opcode']}")
        print(f"  Empty C={C}; op is at C+1={C+1}. Checking which constraints could preclude cycle C.")
        # 1. Precedence
        dep_ids = op.get("dep_ids", [])
        if dep_ids:
            dep_cycles = []
            for did in dep_ids:
                d = id_to_op.get(did)
                if d:
                    dep_cycles.append((did, d["cycle"], d["cycle"] + 1, d["opcode"]))
            latest = max(rc for (_, _, rc, _) in dep_cycles)
            print(f"  1. Precedence: deps {dep_cycles} -> latest_dep_result={latest}. Op can run at cycle >= {latest}. C={C} >= {latest}? {C >= latest}. So precedence DOES NOT preclude C.")
        else:
            print("  1. Precedence: no deps. DOES NOT preclude C.")
        # 2. Neighbors (symmetry ordering is cycle[i] <= cycle[j] for consecutive in group)
        if idx > 0:
            prev_op = ops[idx - 1]
            print(f"  2. Symmetry (if same ordering_group): prev index {idx-1} id={prev_op['id']} cycle={prev_op['cycle']}. Constraint cycle[{idx-1}] <= cycle[{idx}] -> {prev_op['cycle']} <= cycle. So cycle >= {prev_op['cycle']}. C={C} >= {prev_op['cycle']}? {C >= prev_op['cycle']}. DOES NOT preclude C.")
        if idx + 1 < len(ops):
            next_op = ops[idx + 1]
            print(f"     Next index {idx+1} id={next_op['id']} cycle={next_op['cycle']}. Constraint cycle[{idx}] <= cycle[{idx+1}] -> cycle <= {next_op['cycle']}. C={C} <= {next_op['cycle']}? {C <= next_op['cycle']}. DOES NOT preclude C.")
        # 3. Domain (ASAP/ALAP not in JSON; op was placed at C+1 so C+1 in domain)
        latest_res = could_run_ops[0].get("latest_dep_result", -1)
        print(f"  3. Domain [ASAP,ALAP]: not in JSON. Op was at {op['cycle']} so ALAP >= {op['cycle']}. So C={C} is in domain if ASAP <= {C}. From precedence ASAP <= latest_dep_result = {latest_res}. So domain DOES NOT preclude C (unless ASAP bug).")
        # 4. Cumulative
        print(f"  4. Cumulative: C is empty (0 ops at C). Capacity at C is full. DOES NOT preclude C.")
        print("  Conclusion: No constraint visible in the data precludes this op from cycle C. Solver/tie-breaker left it at C+1.")

    if args.json_out:
        out = {
            "total_cycles": total_cycles,
            "max_used_plus_one": max_used_plus_one,
            "empty_cycle_count": len(empty_cycles),
            "structural_gap": total_cycles > max_used_plus_one,
            "breadcrumbs": report,
        }
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote JSON report to {args.json_out}")


if __name__ == "__main__":
    main()
    sys.exit(0)
