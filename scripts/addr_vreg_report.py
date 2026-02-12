#!/usr/bin/env python3
"""Report all VRegs at a given scratch address and ops that read/write it.

Use to rule in/out address conflicts (e.g. 4r/256 hash-as-offset: value and
offset at same address with overlapping need).

  uv run python scripts/addr_vreg_report.py --scheduler sched.json --address 10

Output directory (for plan): artifacts/root_cause/1E_addr_vreg_report/
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


def run(scheduler_path, address, output_dir=None):
    sched = load_scheduler(scheduler_path)
    vregs = sched.get("vregs") or []
    ops = sched.get("ops", [])

    vregs_at_addr = [v for v in vregs if v.get("address") == address]
    reads = [(op.get("id"), op.get("cycle"), "read", op.get("reads", [])) for op in ops if address in op.get("reads", [])]
    writes = [(op.get("id"), op.get("cycle"), "write", op.get("writes", [])) for op in ops if address in op.get("writes", [])]

    report = [
        f"Scheduler: {scheduler_path}",
        f"Address: {address}",
        "",
        f"VRegs at address {address}: {len(vregs_at_addr)}",
    ]
    for v in vregs_at_addr:
        report.append(
            f"  id={v.get('id')} [{v.get('interval_start')},{v.get('interval_end')}] "
            f"round={v.get('round')} phase={v.get('phase')}"
        )
    report.append("")
    report.append(f"Ops reading {address}: {len(reads)}")
    for op_id, cycle, _, _ in reads[:30]:
        report.append(f"  op id={op_id} cycle={cycle}")
    if len(reads) > 30:
        report.append(f"  ... and {len(reads) - 30} more")
    report.append("")
    report.append(f"Ops writing {address}: {len(writes)}")
    for op_id, cycle, _, _ in writes[:30]:
        report.append(f"  op id={op_id} cycle={cycle}")
    if len(writes) > 30:
        report.append(f"  ... and {len(writes) - 30} more")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "report.txt"), "w") as f:
            f.write("\n".join(report) + "\n")

    return {"vregs": vregs_at_addr, "reads": reads, "writes": writes, "report": report}


def main():
    parser = argparse.ArgumentParser(
        description="List VRegs and ops at a scratch address from scheduler JSON."
    )
    parser.add_argument("--scheduler", "-s", required=True, help="Path to scheduler JSON")
    parser.add_argument("--address", "-a", type=int, required=True, help="Scratch address")
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help="Write report to this dir (e.g. artifacts/root_cause/1E_addr_vreg_report/)",
    )
    args = parser.parse_args()
    result = run(args.scheduler, args.address, output_dir=args.output_dir)
    for line in result["report"]:
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
