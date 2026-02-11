#!/usr/bin/env python3
"""Trace single batch index through entire kernel execution.

Records all scratch reads/writes for that batch and compares to reference.

Usage:
    uv run python scripts/trace_batch_execution.py --kernel kernel.json --rounds 4 --batch-size 8 --batch-idx 1
"""

import sys
import os

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir in sys.path:
    sys.path.remove(_script_dir)

import json
import random
from copy import copy

from problem import (
    Machine,
    Tree,
    Input,
    build_mem_image,
    reference_kernel2,
    N_CORES,
    DebugInfo,
    CoreState,
    VLEN,
)


def trace_kernel_step_by_step(instrs, mem, batch_idx):
    """Run kernel step-by-step, recording scratch state each cycle."""
    machine = Machine(
        copy(mem),
        list(instrs),
        DebugInfo(scratch_map={}),
        n_cores=N_CORES,
        trace=False,
    )
    machine.enable_pause = False
    machine.enable_debug = True
    machine.prints = False
    
    core = machine.cores[0]
    scratch_history = []
    
    for cycle_idx in range(len(instrs)):
        if core.state != CoreState.RUNNING:
            break
        
        if core.pc >= len(instrs):
            break
        
        old_scratch = list(core.scratch)
        instr = instrs[core.pc]
        
        core.pc += 1
        machine.step(instr, core)
        if any(name != "debug" for name in instr.keys()):
            machine.cycle += 1
        
        # Record scratch changes
        scratch_changes = {}
        for addr in range(len(core.scratch)):
            if core.scratch[addr] != old_scratch[addr]:
                scratch_changes[addr] = core.scratch[addr]
        
        if scratch_changes:
            scratch_history.append((cycle_idx, dict(scratch_changes)))
    
    return scratch_history, core.scratch


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", default="kernel_to_validate.json")
    parser.add_argument("--forest-height", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch-idx", type=int, default=1)
    parser.add_argument("--scheduler-json", help="Scheduler JSON with address mappings")
    args = parser.parse_args()

    # Load kernel
    print(f"Loading kernel from {args.kernel}...")
    with open(args.kernel) as f:
        data = json.load(f)
    if isinstance(data, dict) and "instrs" in data:
        instrs = data["instrs"]
    else:
        instrs = data
    print(f"  {len(instrs)} instructions")

    # Generate input
    random.seed(args.seed)
    forest = Tree.generate(args.forest_height)
    inp = Input.generate(forest, args.batch_size, args.rounds)
    mem = build_mem_image(forest, inp)

    # Run reference
    mem_ref = copy(mem)
    trace_ref = {}
    list(reference_kernel2(mem_ref, trace_ref))

    # Trace kernel execution
    scratch_history, final_scratch = trace_kernel_step_by_step(instrs, mem, args.batch_idx)
    
    batch_idx = args.batch_idx
    print(f"\nTracing batch index {batch_idx}:")
    print("=" * 80)
    
    # Show reference trace
    print("\nReference kernel trace:")
    for h in range(args.rounds):
        idx = trace_ref.get((h, batch_idx, "idx"), "?")
        val = trace_ref.get((h, batch_idx, "val"), "?")
        hashed = trace_ref.get((h, batch_idx, "hashed_val"), "?")
        next_idx = trace_ref.get((h, batch_idx, "next_idx"), "?")
        print(f"  Round {h}: idx={idx}, val={val}, hashed={hashed}, next_idx={next_idx}")
    
    # Load scheduler JSON to map addresses to VRegs
    vreg_to_addr = {}
    if args.scheduler_json:
        with open(args.scheduler_json) as f:
            sched_data = json.load(f)
        vreg_to_addr = sched_data.get('vreg_to_addr', {})
        print(f"\nLoaded {len(vreg_to_addr)} VReg address mappings")
    
    # Show scratch changes at key cycles
    print(f"\nScratch changes (showing cycles with vector writes):")
    for cycle, changes in scratch_history[:50]:  # First 50 cycles
        # Filter to addresses that might be relevant (vector addresses)
        vector_addrs = [a for a in changes.keys() if a < 100]  # Rough filter
        if vector_addrs:
            print(f"  Cycle {cycle}: {len(changes)} addresses changed")
            # Show first few
            for addr in sorted(changes.keys())[:5]:
                print(f"    addr {addr}: {changes[addr]}")


if __name__ == "__main__":
    main()
