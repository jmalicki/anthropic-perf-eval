#!/usr/bin/env python3
"""Trace exact divergence point using trace IDs.

Finds the first round/phase/operation where values diverge from reference.

Usage:
    uv run python scripts/trace_exact_divergence.py --kernel kernel.json --rounds 4 --batch-size 8 --batch-idx 1
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
)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", default="kernel_to_validate.json")
    parser.add_argument("--forest-height", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch-idx", type=int, default=1, help="Batch index to trace")
    parser.add_argument("--scheduler-json", help="Scheduler JSON with SSA ops (optional)")
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

    # Run optimized kernel with trace IDs
    mem_opt = copy(mem)
    machine = Machine(
        mem_opt,
        list(instrs),
        DebugInfo(scratch_map={}),
        n_cores=N_CORES,
        trace=False,
    )
    machine.enable_pause = False
    machine.enable_debug = True  # Required for trace_io_log
    machine.prints = False
    machine.run()

    # Run reference kernel
    mem_ref = copy(mem)
    trace_ref = {}
    list(reference_kernel2(mem_ref, trace_ref))

    # Get trace IO log
    if not hasattr(machine, 'trace_io_log') or not machine.trace_io_log:
        print("ERROR: Kernel was not generated with --enable-trace-ids")
        print("Re-generate kernel with: --enable-trace-ids")
        sys.exit(1)

    trace_io_log = machine.trace_io_log
    batch_idx = args.batch_idx
    group = batch_idx // 8
    lane = batch_idx % 8

    print(f"\nTracing batch index {batch_idx} (group {group}, lane {lane}):")
    print("=" * 80)

    # Compare trace outputs round-by-round
    divergence_found = False
    for entry in trace_io_log:
        cycle, io_type, trace_id, values = entry
        
        if io_type != 'output':
            continue
        
        # Parse trace_id: r0_g0_hashed_val -> round 0, group 0
        parts = trace_id.split('_')
        if len(parts) < 3:
            continue
        
        try:
            round_num = int(parts[0][1:])  # r0 -> 0
            group_num = int(parts[1][1:])  # g0 -> 0
        except (ValueError, IndexError):
            continue
        
        if group_num != group:
            continue
        
        # Get value for this lane
        if lane >= len(values):
            continue
        
        got_val = values[lane]
        ref_key = (round_num, batch_idx, 'hashed_val')
        ref_val = trace_ref.get(ref_key)
        
        if ref_val is None:
            continue
        
        match = "✓" if got_val == ref_val else "✗"
        status = "MATCH" if got_val == ref_val else "DIVERGENCE"
        
        print(f"Cycle {cycle:3d}: Round {round_num}, Group {group}, Lane {lane}")
        print(f"  Trace ID: {trace_id}")
        print(f"  Got:  {got_val}")
        print(f"  Ref:  {ref_val}")
        print(f"  {match} {status}")
        
        if got_val != ref_val and not divergence_found:
            divergence_found = True
            print(f"\n>>> FIRST DIVERGENCE FOUND <<<")
            print(f"  Round: {round_num}")
            print(f"  Cycle: {cycle}")
            print(f"  Trace ID: {trace_id}")
            print(f"  Got value: {got_val}")
            print(f"  Ref value: {ref_val}")
            print(f"  Difference: {got_val - ref_val}")
            
            # Try to find the SSA op that produced this value
            if args.scheduler_json:
                print(f"\n  Looking up SSA op for trace_id={trace_id}...")
                with open(args.scheduler_json) as f:
                    sched_data = json.load(f)
                ssa_ops = sched_data.get('ssa_ops', [])
                
                # Find ops with matching trace_id or round/phase
                matching_ops = []
                for op in ssa_ops:
                    source_info = op.get('source_info', {})
                    op_round = source_info.get('round')
                    op_phase = source_info.get('phase', '')
                    op_cycle = op.get('cycle', -1)
                    
                    if op_round == round_num and op_phase == 'hash' and op_cycle <= cycle:
                        matching_ops.append(op)
                
                if matching_ops:
                    # Find the hash output op (last hash op in this round)
                    hash_ops = [o for o in matching_ops if o.get('opcode') in ['^', 'multiply_add']]
                    if hash_ops:
                        hash_ops.sort(key=lambda x: x.get('cycle', -1))
                        last_hash = hash_ops[-1]
                        print(f"    Found hash op: Op {last_hash['id']} at cycle {last_hash.get('cycle')}")
                        print(f"      Dest: {last_hash.get('dest')}")
                        print(f"      Sources: {[s.get('VReg', {}).get('id') if s.get('VReg') else s.get('Imm') for s in last_hash.get('srcs', [])]}")
        
        print()

    if not divergence_found:
        print("No divergence found in trace outputs - check final memory values")
    
    # Check final values
    out_values_p = mem_ref[6]
    got_final = machine.mem[out_values_p + batch_idx]
    ref_final = mem_ref[out_values_p + batch_idx]
    print(f"\nFinal values:")
    print(f"  Got:  {got_final}")
    print(f"  Ref:  {ref_final}")
    print(f"  Match: {'YES' if got_final == ref_final else 'NO'}")


if __name__ == "__main__":
    main()
