#!/usr/bin/env python3
"""Dump instructions at specific cycles from a kernel JSON."""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", default="kernel_treeselect_4round.json")
    parser.add_argument("--cycles", type=int, nargs="+", default=[369, 370, 371, 372, 373, 374, 375])
    args = parser.parse_args()

    with open(args.kernel) as f:
        data = json.load(f)
    if isinstance(data, dict) and "instrs" in data:
        instrs = data["instrs"]
    else:
        instrs = data

    for c in args.cycles:
        if c >= len(instrs):
            print(f"\nCycle {c}: OUT OF RANGE (only {len(instrs)} instructions)")
            continue
        instr = instrs[c]
        print(f"\n=== Cycle {c} ===")
        for engine, slots in instr.items():
            if engine in ("debug", "pause"):
                continue
            for slot in slots:
                # Format: engine: (op, dest, src1, src2, ...)
                print(f"  {engine}: {slot}")


if __name__ == "__main__":
    main()
