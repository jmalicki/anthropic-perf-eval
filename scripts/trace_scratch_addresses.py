#!/usr/bin/env python3
"""Trace specific scratch addresses cycle-by-cycle during kernel execution.

Runs the kernel step-by-step and logs every write to monitored addresses,
plus their values at specific "snapshot" cycles. Also logs every READ from
monitored addresses by scanning instructions.
"""

import json
import random
import sys
import os
from copy import copy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from problem import (
    Machine,
    Tree,
    Input,
    build_mem_image,
    N_CORES,
    DebugInfo,
    VLEN,
)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", default="kernel_treeselect_4round.json")
    parser.add_argument("--forest-height", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--watch-addrs",
        type=int,
        nargs="+",
        default=None,
        help="Scratch addresses to monitor (default: 174-181,214-221,190-197)",
    )
    parser.add_argument(
        "--snapshot-cycles",
        type=int,
        nargs="+",
        default=None,
        help="Cycles at which to dump watched addresses",
    )
    args = parser.parse_args()

    if args.watch_addrs is None:
        args.watch_addrs = list(range(174, 182)) + list(range(214, 222)) + list(range(190, 198))
    if args.snapshot_cycles is None:
        args.snapshot_cycles = [6, 13, 14, 100, 200, 300, 369, 370, 371, 372, 373, 374, 375]

    # Load kernel
    print(f"Loading kernel from {args.kernel}...")
    with open(args.kernel) as f:
        data = json.load(f)
    if isinstance(data, dict) and "instrs" in data:
        instrs = data["instrs"]
    else:
        instrs = data
    print(f"  {len(instrs)} instructions")

    # Generate same input as validation
    random.seed(args.seed)
    forest = Tree.generate(args.forest_height)
    inp = Input.generate(forest, args.batch_size, args.rounds)
    mem = build_mem_image(forest, inp)

    # Create machine
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

    watch_set = set(args.watch_addrs)
    snapshot_set = set(args.snapshot_cycles)
    core = machine.cores[0]

    watch_bases = sorted(set(a - (a % 8) for a in watch_set))

    print(f"\nWatching addresses: {sorted(watch_set)}")
    print(f"Snapshot cycles: {sorted(snapshot_set)}")
    print()

    # Monkey-patch step to intercept scratch_write
    original_step = Machine.step.__get__(machine)

    def traced_step(instr, core_arg, cycle_num):
        """Execute step and intercept writes to watched addresses."""
        # Execute all engine functions (builds scratch_write and mem_write)
        ENGINE_FNS = {
            "alu": machine.alu,
            "valu": machine.valu,
            "load": machine.load,
            "store": machine.store,
            "flow": machine.flow,
        }
        machine.scratch_write = {}
        machine.mem_write = {}
        for name, slots in instr.items():
            if name == "debug":
                continue
            if name == "pause":
                continue
            for slot in slots:
                ENGINE_FNS[name](core_arg, *slot)

        # Check for writes to watched addresses BEFORE applying them
        watched_writes = {a: v for a, v in machine.scratch_write.items() if a in watch_set}
        if watched_writes:
            # Find which instruction caused the write
            for a in sorted(watched_writes):
                old = core_arg.scratch[a]
                new = watched_writes[a]
                if old != new:
                    print(f"  WRITE cycle={cycle_num}: s[{a}] = {new} (was {old})")

        # Apply writes
        for addr, val in machine.scratch_write.items():
            core_arg.scratch[addr] = val
        for addr, val in machine.mem_write.items():
            machine.mem[addr] = val

        del machine.scratch_write
        del machine.mem_write

    # Step through cycle by cycle
    total_cycles = len(instrs)
    for cycle_idx in range(total_cycles):
        instr = instrs[cycle_idx]

        # Take snapshot BEFORE executing this cycle
        if cycle_idx in snapshot_set:
            print(f"\n=== Snapshot BEFORE cycle {cycle_idx} ===")
            for base in watch_bases:
                vals = [core.scratch[base + j] for j in range(8)]
                # Format as compact list
                print(f"  s[{base:3d}..{base+7}]: {vals}")

        # Execute this cycle with tracing
        traced_step(instr, core, cycle_idx)

        # Take snapshot AFTER executing this cycle
        if cycle_idx in snapshot_set:
            print(f"=== Snapshot AFTER cycle {cycle_idx} ===")
            for base in watch_bases:
                vals = [core.scratch[base + j] for j in range(8)]
                print(f"  s[{base:3d}..{base+7}]: {vals}")

        core.pc += 1

    # Final state
    print(f"\n=== Final state (after all {total_cycles} cycles) ===")
    for base in watch_bases:
        vals = [core.scratch[base + j] for j in range(8)]
        print(f"  s[{base:3d}..{base+7}]: {vals}")


if __name__ == "__main__":
    main()
