#!/usr/bin/env python3
"""Compare scratch values at specific cycles to reference trace.

Runs the kernel and at specified cycles dumps scratch[base..base+7] (vector).
For a given group, fetches expected values from the reference and prints
got vs expected side-by-side. Pinpoints exactly where values diverge.

Usage:
  uv run python -m scripts.scratch_vs_ref kernel.json --rounds 4 --batch-size 8 \\
    --addr 21 --cycles 71 72 --group 0
"""

import sys
import os

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
if os.path.dirname(os.path.abspath(__file__)) in sys.path:
    sys.path.remove(os.path.dirname(os.path.abspath(__file__)))

import json
import random
import argparse

from problem import (
    Machine,
    Tree,
    Input,
    build_mem_image,
    reference_kernel2,
    N_CORES,
    DebugInfo,
)
from copy import copy


def main():
    ap = argparse.ArgumentParser(description="Compare scratch at cycles to reference")
    ap.add_argument("kernel", help="Kernel JSON")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--addr", type=int, nargs="+", default=[21], help="Vector base address(s) to dump")
    ap.add_argument("--cycles", type=int, nargs="+", default=[70, 71, 72, 73, 74])
    ap.add_argument("--group", type=int, default=0, help="Group for ref trace (batch group*8..group*8+7)")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    with open(args.kernel) as f:
        data = json.load(f)
    instrs = data.get("instrs", data) if isinstance(data, dict) else data

    random.seed(args.seed)
    forest = Tree.generate(10)
    inp = Input.generate(forest, args.batch_size, args.rounds)
    mem = build_mem_image(forest, inp)

    # Run reference for trace
    trace_ref = {}
    ref_results = list(reference_kernel2(copy(mem), trace_ref))

    addrs = args.addr if isinstance(args.addr, list) else [args.addr]
    group = args.group
    batch_start = group * 8
    target_cycles = set(args.cycles)

    # Run kernel, capture scratch at START of each target cycle (before writes)
    mem_opt = copy(mem)
    machine2 = Machine(
        mem_opt,
        list(instrs),
        DebugInfo(scratch_map={}),
        n_cores=N_CORES,
        trace=False,
    )
    machine2.enable_pause = False
    machine2.enable_debug = False
    machine2.prints = False
    core = machine2.cores[0]

    captures = {}  # cycle -> {addr: [vals]}
    for cycle_idx in range(len(instrs)):
        if cycle_idx in target_cycles:
            captures[cycle_idx] = {
                base: [core.scratch[base + j] for j in range(8)]
                for base in addrs
            }
        instr = instrs[cycle_idx]
        core.pc += 1
        machine2.step(instr, core)

    # trace_ref keys: (round, batch_i, "hashed_val") or (round, batch_i, "val"), etc.
    n_rounds = args.rounds
    round_idx = n_rounds - 1

    print(f"Scratch at cycles {args.cycles} (group {group}, batch {batch_start}..{batch_start+7})")
    print()

    for c in sorted(captures.keys()):
        for base in addrs:
            got = captures[c][base]
            print(f"Cycle {c} s[{base}..{base+7}] GOT: {[hex(x) for x in got]}")
        ref_vals = []
        for lane in range(8):
            bi = batch_start + lane
            key = (round_idx, bi, "hashed_val")
            ref_vals.append(trace_ref.get(key))
        if any(x is not None for x in ref_vals) and addrs:
            base = addrs[0]
            got = captures[c][base]
            print(f"        REF (round {round_idx} hashed_val): {[hex(x) if x is not None else '?' for x in ref_vals]}")
            for lane in range(8):
                g, r = got[lane], ref_vals[lane]
                match = "OK" if r is not None and g == r else "MISMATCH"
                print(f"        lane {lane}: got={hex(g)} ref={hex(r) if r else '?'} [{match}]")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
