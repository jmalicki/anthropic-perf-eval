#!/usr/bin/env python3
"""List ops at a given cycle that read a given scratch address, and VRegs at that address with intervals.

Use when debugging validation failures: find which op at cycle C reads address A,
and which VRegs were assigned A and whether any ended before the read.

  uv run python scripts/inspect_cycle_addr.py sched_4r8b_d3.json --cycle 785 --addr 19
  uv run python scripts/inspect_cycle_addr.py scheduler_output.json --cycle 786 --addr 19 --vregs
"""

import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser(
        description="Ops at cycle C that read addr A, and VRegs at A with intervals"
    )
    ap.add_argument("json_path", help="Path to scheduler output JSON")
    ap.add_argument("--cycle", type=int, required=True, help="Cycle to inspect")
    ap.add_argument("--addr", type=int, required=True, help="Scratch address (or vector base)")
    ap.add_argument(
        "--vregs",
        action="store_true",
        help="Show intervals and live-at-cycle from vregs array",
    )
    args = ap.parse_args()

    with open(args.json_path) as f:
        data = json.load(f)

    ops = data.get("ops", [])
    cycle_ops = [o for o in ops if o.get("cycle") == args.cycle]
    # Vector: address A covers A..A+7
    addr_end = args.addr + 8
    reads_addr = []
    for o in cycle_ops:
        r = o.get("reads") or []
        if any(args.addr <= x < addr_end for x in r):
            reads_addr.append(o)
    if not reads_addr:
        print(f"No ops at cycle {args.cycle} with reads containing address {args.addr} (or {args.addr}..{args.addr+7})")
    else:
        print(f"Ops at cycle {args.cycle} that read address {args.addr} (or vector {args.addr}..{args.addr+7}):")
        for o in reads_addr:
            print(f"  id={o.get('id')} opcode={o.get('opcode')} reads={o.get('reads')} writes={o.get('writes')}")

    vreg_to_addr = data.get("vreg_to_addr")
    if not vreg_to_addr:
        print("\nNo vreg_to_addr in JSON", file=sys.stderr)
        return 0
    at_addr = [v for v, a in vreg_to_addr.items() if a == args.addr]
    print(f"\nVRegs at scratch address {args.addr}: {len(at_addr)}")
    for v in sorted(at_addr):
        print(f"  {v}")

    if args.vregs:
        vregs_list = data.get("vregs")
        if not vregs_list:
            print("\n(No 'vregs' in JSON; run with --liveness-detail full)")
        else:
            at_addr_vregs = [v for v in vregs_list if v.get("address") == args.addr]
            print("\nIntervals and live-at-cycle:")
            for v in at_addr_vregs:
                start = v.get("interval_start")
                end = v.get("interval_end")
                # Half-open [start, end): live at cycle C iff start <= C < end
                live = start is not None and end is not None and start <= args.cycle < end
                ended_before = end is not None and end <= args.cycle
                flags = []
                if live:
                    flags.append("LIVE at cycle")
                if ended_before:
                    flags.append("ended before read")
                print(f"  id={v.get('id')} width={v.get('width')} interval=[{start}, {end}) {' '.join(flags)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
