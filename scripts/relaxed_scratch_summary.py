#!/usr/bin/env python3
"""One row per scheduler JSON for the relaxed-scratch reference (docs/RELAXED_SCRATCH_REFERENCE.md).

    uv run python -m scripts.relaxed_scratch_summary real.json relaxed.json [...]

Columns: schedule cycles, peak live scratch words, highest scratch address, the scratch size the
run was allowed and whether CP-SAT modelled it, CP-SAT's proven bound (`lb_cycles`, CP-SAT
runs only), the capacity floor for the ops the run emitted (scripts/schedule_report.py), and
the batch groups in flight at the peak-liveness cycle and at most.

A group is in flight at cycle c when it has an op at or before c and an op at or after c (the
window `--greedy-groups-in-flight` bounds). With round barriers every group is in flight for
the whole kernel. The cycle count of a relaxed run is a schedule, not a floor; `lb_cycles` is
the floor (for that emission and the model's domains).
"""

import argparse
import json
import sys


def groups_in_flight(d):
    """(groups in flight at the peak-liveness cycle, most groups in flight at any cycle), or
    (None, None) when the JSON has no SSA ops with group info."""
    spans = {}
    for op in d.get("ssa_ops") or []:
        group = (op.get("source_info") or {}).get("group")
        cycle = op.get("cycle", -1)
        if group is None or cycle is None or cycle < 0:
            continue
        lo, hi = spans.get(group, (cycle, cycle))
        spans[group] = (min(lo, cycle), max(hi, cycle))
    if not spans:
        return None, None
    delta = [0] * (d["total_cycles"] + 2)
    for lo, hi in spans.values():
        delta[lo] += 1
        delta[hi + 1] -= 1
    running, per_cycle = 0, []
    for step in delta[:-1]:
        running += step
        per_cycle.append(running)
    peak_cycle = (d.get("liveness_summary") or {}).get("peak_cycle")
    at_peak = per_cycle[peak_cycle] if peak_cycle is not None and 0 <= peak_cycle < len(per_cycle) else None
    return at_peak, max(per_cycle)


def summarize(d):
    """The row for one scheduler JSON as a dict."""
    from scripts import schedule_report

    report = schedule_report.analyze(d)
    at_peak, most = groups_in_flight(d)
    opts = d.get("run_options") or {}
    return {
        "cycles": d["total_cycles"],
        "simulator_cycles": report["run"]["simulator_cycles"],
        "peak_live_words": d.get("peak_live_words"),
        "max_addr": d.get("max_addr"),
        "scratch_size": d.get("scratch_size", 1536),
        "cpsat_scratch_budget": d.get("cpsat_scratch_budget"),
        "relaxed": bool(d.get("scratch_relaxed")),
        "lb_cycles": d.get("lb_cycles"),
        "capacity_floor": report["floor"]["cycles"],
        "groups_in_flight_at_peak": at_peak,
        "groups_in_flight_max": most,
        "scheduler": report["run"]["scheduler"],
        "rounds": opts.get("rounds"),
        "comment": d.get("comment"),
    }


COLUMNS = [
    ("cycles", "cycles"), ("simulator_cycles", "sim"), ("peak_live_words", "peak"), ("max_addr", "max_addr"),
    ("scratch_size", "scratch"), ("cpsat_scratch_budget", "cpsat_budget"), ("lb_cycles", "lb_cycles"),
    ("capacity_floor", "floor"), ("groups_in_flight_at_peak", "groups@peak"),
    ("groups_in_flight_max", "groups_max"), ("scheduler", "sched"),
]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("json", nargs="+", help="scheduler output JSONs")
    parser.add_argument("--json-out", action="store_true", help="print the rows as JSON")
    args = parser.parse_args(argv)
    rows = []
    for path in args.json:
        with open(path) as f:
            rows.append((path, summarize(json.load(f))))
    if args.json_out:
        json.dump({path: row for path, row in rows}, sys.stdout, indent=2)
        print()
        return 0
    header = ["file"] + [title for _, title in COLUMNS]
    table = [header] + [[path] + ["-" if row[key] is None else str(row[key]) for key, _ in COLUMNS]
                        for path, row in rows]
    widths = [max(len(r[i]) for r in table) for i in range(len(header))]
    for r in table:
        print("  ".join(cell.ljust(w) for cell, w in zip(r, widths)))
    for path, row in rows:
        if row["relaxed"] and row["lb_cycles"] is not None:
            print(f"{path}: relaxed run; lb_cycles {row['lb_cycles']} is CP-SAT's floor, "
                  f"cycles {row['cycles']} is a schedule")
    return 0


if __name__ == "__main__":
    sys.exit(main())
