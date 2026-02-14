#!/usr/bin/env python3
"""Compare scratch state cycle-by-cycle between two kernels on same input.

Finds the first cycle at which scratch differs. Use to debug validation
failure (e.g. zero-gap with shift vs without shift).

Usage:
  uv run python scripts/compare_kernel_scratch_cycle.py kernel_a.json kernel_b.json \\
    --rounds 2 --batch-size 16
"""

import argparse
import json
import sys
from copy import copy

from problem import (
    Machine,
    Tree,
    Input,
    build_mem_image,
    N_CORES,
    DebugInfo,
)


def load_instrs(path):
    with open(path) as f:
        data = json.load(f)
    return data.get("instrs", data) if isinstance(data, dict) else data


def run_and_capture_scratch(instrs, mem, max_cycles, scratch_len=256):
    """Run kernel; return list of scratch snapshots at END of each cycle."""
    machine = Machine(
        copy(mem),
        list(instrs),
        DebugInfo(scratch_map={}),
        n_cores=N_CORES,
        trace=False,
    )
    machine.enable_pause = False
    machine.enable_debug = False
    machine.prints = False
    core = machine.cores[0]

    snapshots = []
    for cycle_idx in range(min(max_cycles, len(instrs))):
        instr = instrs[cycle_idx]
        machine.step(instr, core)
        snapshots.append(core.scratch[:scratch_len].copy())
    return snapshots


def main():
    ap = argparse.ArgumentParser(description="First cycle where scratch differs between two kernels")
    ap.add_argument("kernel_a", help="First kernel JSON (e.g. passing)")
    ap.add_argument("kernel_b", help="Second kernel JSON (e.g. failing)")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--scratch-len", type=int, default=256)
    ap.add_argument("--max-cycles", type=int, default=None, help="Max cycles to compare (default: min of two lengths)")
    args = ap.parse_args()

    import random
    random.seed(args.seed)
    forest = Tree.generate(10)
    inp = Input.generate(forest, args.batch_size, args.rounds)
    mem = build_mem_image(forest, inp)

    instrs_a = load_instrs(args.kernel_a)
    instrs_b = load_instrs(args.kernel_b)
    max_c = args.max_cycles
    if max_c is None:
        max_c = min(len(instrs_a), len(instrs_b))
    print(f"Comparing scratch for cycles 0..{max_c-1} (kernel_a has {len(instrs_a)} cycles, kernel_b has {len(instrs_b)})")

    snap_a = run_and_capture_scratch(instrs_a, mem, max_c, args.scratch_len)
    snap_b = run_and_capture_scratch(instrs_b, mem, max_c, args.scratch_len)

    for c in range(max_c):
        for addr in range(args.scratch_len):
            if snap_a[c][addr] != snap_b[c][addr]:
                print(f"First scratch difference at cycle {c}, address {addr}: a={snap_a[c][addr]} (0x{snap_a[c][addr]&0xffffffff:x}) b={snap_b[c][addr]} (0x{snap_b[c][addr]&0xffffffff:x})")
                # Show a few surrounding addresses
                lo = max(0, addr - 2)
                hi = min(args.scratch_len, addr + 3)
                print(f"  s[{lo}..{hi}] kernel_a: {[snap_a[c][i] for i in range(lo, hi)]}")
                print(f"  s[{lo}..{hi}] kernel_b: {[snap_b[c][i] for i in range(lo, hi)]}")
                return 0
    print("No scratch difference in compared cycles")
    return 0


if __name__ == "__main__":
    sys.exit(main())
