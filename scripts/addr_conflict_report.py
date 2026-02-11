#!/usr/bin/env python3
"""Report overlapping VReg intervals at the same scratch address.

Loads scheduler output JSON and finds addresses where two or more VRegs have
overlapping live intervals—indicating an allocation bug (one value overwrites
another before it is read).

Usage:
  uv run python -m scripts.addr_conflict_report scheduler_output.json
  uv run python -m scripts.addr_conflict_report sched.json --addr 13
"""

import sys
import os

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
if os.path.dirname(os.path.abspath(__file__)) in sys.path:
    sys.path.remove(os.path.dirname(os.path.abspath(__file__)))

import json
import argparse


def intervals_conflict(a_start, a_end, b_start, b_end):
    """True if B writes before A's last read (problematic reuse).
    Same-cycle reuse (b_start == a_end) is OK: read then write within cycle."""
    return a_start <= b_end and b_start < a_end


def main():
    ap = argparse.ArgumentParser(description="Report overlapping VReg intervals at same address")
    ap.add_argument("json_path", help="Scheduler output JSON")
    ap.add_argument("--addr", type=int, default=None, help="Only check this address")
    args = ap.parse_args()

    with open(args.json_path) as f:
        data = json.load(f)

    vregs = data.get("vregs")
    if not vregs:
        print("No 'vregs' in JSON; generate with --liveness-detail full")
        return 1

    # Group by address
    by_addr = {}
    for v in vregs:
        addr = v.get("address")
        if addr is None:
            continue
        start = v.get("interval_start")
        end = v.get("interval_end")
        if start is None or end is None:
            continue
        by_addr.setdefault(addr, []).append({
            "id": v.get("id"),
            "width": v.get("width", "?"),
            "phase": v.get("phase", ""),
            "start": start,
            "end": end,
        })

    addrs = [args.addr] if args.addr is not None else sorted(by_addr.keys())
    any_conflict = False

    for addr in addrs:
        if addr not in by_addr:
            continue
        items = by_addr[addr]
        if len(items) < 2:
            continue
        items = sorted(items, key=lambda x: (x["start"], x["end"]))

        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i], items[j]
                if intervals_conflict(a["start"], a["end"], b["start"], b["end"]):
                    any_conflict = True
                    print(f"CONFLICT addr {addr}: id={a['id']} [{a['start']},{a['end']}] "
                          f"overlaps id={b['id']} [{b['start']},{b['end']}]")

    if not any_conflict:
        print("No overlapping intervals found.")
    return 0 if not any_conflict else 1


if __name__ == "__main__":
    sys.exit(main())
