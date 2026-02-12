#!/usr/bin/env python3
"""Report which VRegs are live at a given cycle from scheduler JSON.

Use to rule in/out liveness as cause of validation failures (e.g. round-1 hash
VRegs not live at cycle 50).

  uv run python scripts/check_liveness_at_cycle.py --scheduler sched.json --cycle 50
  uv run python scripts/check_liveness_at_cycle.py --scheduler sched.json --cycle 50 --round 1

Output directory (for plan): artifacts/root_cause/1B_liveness_at_cycle/
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


def vreg_live_at_cycle(vreg_info, cycle):
    """True if VReg's interval contains cycle (inclusive).
    Scheduler JSON: interval_start = def cycle, interval_end = last use cycle (or def+1 for dead).
    """
    s = vreg_info.get("interval_start", -1)
    e = vreg_info.get("interval_end", -1)
    return s <= cycle <= e


def run(scheduler_path, cycle, round_filter=None, phase_filter=None, output_dir=None):
    sched = load_scheduler(scheduler_path)
    vregs = sched.get("vregs")
    if not vregs:
        msg = "Scheduler JSON has no 'vregs' (generate with liveness_detail=full)."
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "report.txt"), "w") as f:
                f.write(msg + "\n")
        return {"error": msg, "live_at_cycle": [], "live_count": 0}

    live = []
    for v in vregs:
        if not vreg_live_at_cycle(v, cycle):
            continue
        if round_filter is not None:
            r = v.get("round")
            if r is None or r != round_filter:
                continue
        if phase_filter is not None:
            ph = v.get("phase") or ""
            if phase_filter not in ph:
                continue
        live.append(v)

    report_lines = [
        f"Scheduler: {scheduler_path}",
        f"Cycle: {cycle}",
        f"Round filter: {round_filter!r}  Phase filter: {phase_filter!r}",
        f"Total VRegs in JSON: {len(vregs)}",
        f"Live at cycle {cycle}: {len(live)}",
        "",
    ]
    if live:
        report_lines.append("Live VRegs (id, interval_start, interval_end, address, round, phase):")
        for v in live[:100]:  # cap for readability
            report_lines.append(
                f"  id={v.get('id')} [{v.get('interval_start')},{v.get('interval_end')}] "
                f"addr={v.get('address')} round={v.get('round')} phase={v.get('phase')}"
            )
        if len(live) > 100:
            report_lines.append(f"  ... and {len(live) - 100} more")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "report.txt"), "w") as f:
            f.write("\n".join(report_lines) + "\n")

    return {"live_at_cycle": live, "live_count": len(live), "report_lines": report_lines}


def main():
    parser = argparse.ArgumentParser(
        description="Report VRegs live at a given cycle from scheduler JSON."
    )
    parser.add_argument("--scheduler", "-s", required=True, help="Path to scheduler JSON (--output-json)")
    parser.add_argument("--cycle", "-c", type=int, required=True, help="Cycle number")
    parser.add_argument("--round", "-r", type=int, default=None, help="Filter VRegs by round (e.g. 1 for round-1 hash)")
    parser.add_argument("--phase", "-p", type=str, default=None, help="Filter VRegs by phase substring")
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help="Write report to this dir (e.g. artifacts/root_cause/1B_liveness_at_cycle/)",
    )
    args = parser.parse_args()

    result = run(
        args.scheduler,
        args.cycle,
        round_filter=args.round,
        phase_filter=args.phase,
        output_dir=args.output_dir,
    )
    if result.get("error"):
        print(result["error"])
        return 1
    for line in result.get("report_lines", []):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
