#!/usr/bin/env python3
"""Trace kernel execution step-by-step to find where values diverge.

Compares intermediate scratch values to reference kernel to find the first divergence.
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
from copy import copy, deepcopy

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


def trace_kernel_step_by_step(instrs, mem):
    """Run kernel step-by-step, recording scratch state each cycle."""
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
        
        # Record scratch state
        scratch_history.append((cycle_idx, list(core.scratch)))
    
    return scratch_history, core.scratch, machine.mem


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", default="kernel_to_validate.json")
    parser.add_argument("--forest-height", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch-idx", type=int, default=0)
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

    # Run reference kernel
    mem_ref = deepcopy(mem)
    trace_ref = {}
    list(reference_kernel2(mem_ref, trace_ref))

    # Trace optimized kernel
    scratch_history, final_scratch, final_mem = trace_kernel_step_by_step(instrs, mem)
    
    batch_idx = args.batch_idx
    group = batch_idx // 8
    lane = batch_idx % 8
    
    print(f"\nTracing batch index {batch_idx} (group {group}, lane {lane}):")
    print("=" * 80)
    
    # Load scheduler JSON to map VRegs to addresses
    vreg_to_addr = {}
    if args.scheduler_json:
        with open(args.scheduler_json) as f:
            sched_data = json.load(f)
        vreg_to_addr = sched_data.get('vreg_to_addr', {})
        print(f"Loaded {len(vreg_to_addr)} VReg address mappings")
    
    # Check reference values for each round
    print("\nReference kernel values:")
    for h in range(args.rounds):
        hashed = trace_ref.get((h, batch_idx, "hashed_val"))
        if hashed is not None:
            print(f"  Round {h} hash: {hashed}")
    
    # Check final values
    out_values_p = mem_ref[6]
    got_final = final_mem[out_values_p + batch_idx]
    ref_final = mem_ref[out_values_p + batch_idx]
    print(f"\nFinal values:")
    print(f"  Got:  {got_final}")
    print(f"  Ref:  {ref_final}")
    print(f"  Match: {'YES' if got_final == ref_final else 'NO'}")
    
    # Try to find where the hash values are stored in scratch
    # For Round 0 hash, check what address it should be at
    # We need to find which VReg stores Round 0 hash and what address it's at
    
    if args.scheduler_json:
        print("\nChecking Round 0 hash storage:")
        # Find Round 0 hash VReg from SSA ops
        ssa_ops = sched_data.get('ssa_ops', [])
        round0_hash_ops = [o for o in ssa_ops if o.get('source_info', {}).get('round') == 0 and o.get('source_info', {}).get('phase') == 'hash']
        if round0_hash_ops:
            # Find the last hash op (produces final hash value)
            round0_hash_ops.sort(key=lambda x: x.get('cycle', -1))
            last_hash = round0_hash_ops[-1]
            hash_vreg = last_hash.get('dest')
            print(f"  Round 0 hash VReg: {hash_vreg}")
            
            # Find address via exact key match. vreg_to_addr keys are Rust Debug format
            # ("VReg { id: N, width: Scalar|Vector }"); ssa_ops dest is "vN_s" or "vN_v".
            def vreg_dest_to_map_key(dest_str):
                if not dest_str or not isinstance(dest_str, str):
                    return None
                parts = dest_str.split("_", 1)
                if len(parts) != 2 or not parts[0].startswith("v"):
                    return None
                try:
                    vreg_id = int(parts[0][1:], 10)
                except ValueError:
                    return None
                width = "Vector" if parts[1] == "v" else "Scalar" if parts[1] == "s" else None
                if width is None:
                    return None
                return f"VReg {{ id: {vreg_id}, width: {width} }}"
            map_key = vreg_dest_to_map_key(hash_vreg)
            hash_addr = vreg_to_addr.get(map_key) if map_key else None
            
            if hash_addr is not None:
                print(f"  Round 0 hash address: {hash_addr} (vector, spans {hash_addr} to {hash_addr+7})")
                print(f"  Lane {lane} value at address {hash_addr + lane}:")
                
                # Check scratch at key cycles
                for cycle, scratch in scratch_history:
                    if cycle >= 15 and cycle <= 20:  # Around Round 0 hash completion
                        idx = hash_addr + lane
                        if 0 <= idx < len(scratch):
                            val = scratch[idx]
                            print(f"    Cycle {cycle}: {val}")
                        else:
                            print(f"    Cycle {cycle}: (index {idx} out of range [0, {len(scratch)}), skipped)")


if __name__ == "__main__":
    main()
