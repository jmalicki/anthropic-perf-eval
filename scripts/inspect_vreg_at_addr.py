#!/usr/bin/env python3
"""Inspect which VRegs were assigned a given scratch address in scheduler output JSON.

Use when debugging validation failures due to scratch reuse (e.g. hash value and
offset sharing the same address). Run from project root:

  uv run python scripts/inspect_vreg_at_addr.py scheduler_output.json
  uv run python scripts/inspect_vreg_at_addr.py scheduler_output.json --addr 10
  uv run python scripts/inspect_vreg_at_addr.py scheduler_output.json --addr 10 --vregs

With --vregs, also shows interval [start, end] for each VReg if the JSON has
liveness detail (vregs array with interval_start/interval_end).
"""

import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser(description="List VRegs at a given scratch address")
    ap.add_argument("json_path", help="Path to scheduler output JSON")
    ap.add_argument("--addr", type=int, default=10, help="Scratch address (default 10)")
    ap.add_argument("--vregs", action="store_true", help="Show intervals from vregs if present")
    args = ap.parse_args()

    with open(args.json_path) as f:
        data = json.load(f)

    vreg_to_addr = data.get("vreg_to_addr")
    if not vreg_to_addr:
        print("No vreg_to_addr in JSON", file=sys.stderr)
        return 1

    at_addr = [vreg for vreg, addr in vreg_to_addr.items() if addr == args.addr]
    if not at_addr:
        print(f"No VReg at address {args.addr}")
        return 0

    print(f"VRegs at scratch address {args.addr} ({len(at_addr)}):")
    for vreg in sorted(at_addr):
        print(f"  {vreg}")

    if args.vregs:
        vregs_list = data.get("vregs")
        if not vregs_list:
            print("\n(No 'vregs' in JSON; run with --liveness-detail full to get intervals)")
        else:
            # vregs: list of { id, width, address, interval_start, interval_end, ... }
            at_addr_vregs = [v for v in vregs_list if v.get("address") == args.addr]
            print("\nIntervals for VRegs at this address (from vregs):")
            for v in at_addr_vregs:
                i = v.get("interval_start"), v.get("interval_end")
                print(f"  id={v.get('id')} width={v.get('width')} interval=[{i[0]}, {i[1]}]")
            if not at_addr_vregs and at_addr:
                print("  (vregs has no entry for this address; addresses may differ from vreg_to_addr)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
