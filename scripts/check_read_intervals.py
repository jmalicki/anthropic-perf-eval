#!/usr/bin/env python3
"""Check that every op that reads scratch has a live VReg at each read address at op.cycle.

For each op with reads: at op.cycle, for each read address A, at least one VReg
assigned to A must have interval containing op.cycle. Reports violations.

  uv run python scripts/check_read_intervals.py --scheduler sched.json

Output directory (for plan): artifacts/root_cause/1C_read_intervals/
"""

import argparse
import json
import os
import sys

# Project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_scheduler(path):
    with open(path) as f:
        return json.load(f)


def vreg_covers_cycle(vreg_info, cycle):
    """True if VReg's interval contains cycle (inclusive)."""
    s = vreg_info.get("interval_start", -1)
    e = vreg_info.get("interval_end", -1)
    return s <= cycle <= e


def run(scheduler_path, output_dir=None):
    sched = load_scheduler(scheduler_path)
    ops = sched.get("ops", [])
    vregs = sched.get("vregs") or []
    # Build: address -> list of VReg infos at that address
    addr_to_vregs = {}
    for v in vregs:
        addr = v.get("address", -1)
        if addr < 0:
            continue
        addr_to_vregs.setdefault(addr, []).append(v)

    violations = []
    for op in ops:
        op_id = op.get("id", "?")
        cycle = op.get("cycle", -1)
        reads = op.get("reads", [])
        for addr in reads:
            vregs_at_addr = addr_to_vregs.get(addr, [])
            any_live = any(vreg_covers_cycle(v, cycle) for v in vregs_at_addr)
            if not any_live:
                violations.append({
                    "op_id": op_id,
                    "cycle": cycle,
                    "address": addr,
                    "vregs_at_addr": [
                        (v.get("id"), v.get("interval_start"), v.get("interval_end"))
                        for v in vregs_at_addr
                    ],
                })
    report_lines = [
        f"Scheduler: {scheduler_path}",
        f"Ops: {len(ops)}, VRegs in JSON: {len(vregs)}",
        f"Read-interval violations: {len(violations)}",
        "",
    ]
    for v in violations[:50]:
        report_lines.append(
            f"  op id={v['op_id']} cycle={v['cycle']} reads addr={v['address']} "
            f"(VRegs at addr: {v['vregs_at_addr']})"
        )
    if len(violations) > 50:
        report_lines.append(f"  ... and {len(violations) - 50} more")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "report.txt"), "w") as f:
            f.write("\n".join(report_lines) + "\n")

    for line in report_lines:
        print(line)
    return violations


def main():
    parser = argparse.ArgumentParser(
        description="Check that every read has a live VReg at that address at op.cycle."
    )
    parser.add_argument("--scheduler", "-s", required=True, help="Path to scheduler JSON")
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help="Write report to this dir (e.g. artifacts/root_cause/1C_read_intervals/)",
    )
    args = parser.parse_args()
    violations = run(args.scheduler, output_dir=args.output_dir)
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
