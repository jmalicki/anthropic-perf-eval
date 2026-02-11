#!/usr/bin/env python3
"""Find the last instruction that wrote to a scratch address before a given cycle.

Walks kernel instructions backward from (cycle-1) to 0 and reports the first
write to the address. Use when debugging: "what value was at addr X when we read it at cycle C?"

Usage:
  uv run python -m scripts.who_wrote_addr kernel.json --addr 13 --before-cycle 72
"""

import sys
import os

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
if os.path.dirname(os.path.abspath(__file__)) in sys.path:
    sys.path.remove(os.path.dirname(os.path.abspath(__file__)))

import json
import argparse


def get_writes(instr):
    """Extract (addr, size) written by each slot. size=8 for vector, 1 for scalar."""
    writes = []
    for engine, slots in instr.items():
        if engine in ("debug", "pause"):
            continue
        for slot in slots:
            if not isinstance(slot, (list, tuple)) or len(slot) < 2:
                continue
            op = slot[0] if isinstance(slot[0], str) else str(slot[0])
            if op == "vstore":
                continue  # writes to mem
            dest = slot[1]
            if not isinstance(dest, (int, float)):
                continue
            dest = int(dest)
            size = 8 if (engine == "valu" or op in ("vload", "vbroadcast", "vselect", "multiply_add")) else 1
            writes.append((dest, size))
    return writes


def main():
    ap = argparse.ArgumentParser(description="Last write to addr before cycle")
    ap.add_argument("kernel", help="Kernel JSON")
    ap.add_argument("--addr", type=int, required=True)
    ap.add_argument("--before-cycle", type=int, required=True, help="Cycle at which we read (search backwards)")
    args = ap.parse_args()

    with open(args.kernel) as f:
        data = json.load(f)
    instrs = data.get("instrs", data) if isinstance(data, dict) else data

    addr = args.addr
    before = args.before_cycle
    addr_end = addr + 8  # vector span

    for c in range(before - 1, -1, -1):
        instr = instrs[c]
        for (dest, size) in get_writes(instr):
            dest_end = dest + size
            if dest < addr_end and dest_end > addr:
                print(f"Cycle {c}: {instr}")
                return 0

    print(f"No write to addr {addr} (or {addr}..{addr+7}) before cycle {before}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
