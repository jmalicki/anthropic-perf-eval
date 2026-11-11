#!/usr/bin/env python3
"""Debug validation failure by tracing reference kernel vs optimized kernel.

Loads a saved kernel and compares per-round per-batch intermediate values
against the reference kernel to find the exact point of divergence.

Usage when you have a kernel that fails validation:
  uv run python scripts/debug_validation_trace.py --kernel kernel_to_validate.json --rounds 5 --batch-size 16
Use the same --rounds and --batch-size as the run that produced the kernel.
The script prints reference trace for --batch-idx, final got vs ref, and all
mismatching batch indices with affected groups.

Note: For 5-round builds, failing combos (e.g. --valu-shift alu, --const-choice load)
often make CP-SAT infeasible, so no kernel is saved and there is nothing to trace.
Use a config that produces a kernel but fails validation (e.g. wrong rounds match).
"""

import sys
import os

# Ensure project root is on path and script dir is not, so stdlib "bisect" is not shadowed by scripts/bisect.py
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir in sys.path:
    sys.path.remove(_script_dir)

import json
import random

from problem import (
    Machine,
    Tree,
    Input,
    build_mem_image,
    reference_kernel2,
    N_CORES,
    DebugInfo,
    SCRATCH_SIZE,
)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", default="kernel_to_validate.json")
    parser.add_argument("--forest-height", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch-idx", type=int, default=160, help="Batch index to trace")
    args = parser.parse_args()

    # Load the saved kernel
    print(f"Loading kernel from {args.kernel}...")
    with open(args.kernel) as f:
        data = json.load(f)
    # Handle both formats: raw instruction list or dict with 'instrs' key
    if isinstance(data, dict) and "instrs" in data:
        instrs = data["instrs"]
    else:
        instrs = data
    print(f"  {len(instrs)} instructions")

    # Generate the same tree/input as validation
    random.seed(args.seed)
    forest = Tree.generate(args.forest_height)
    inp = Input.generate(forest, args.batch_size, args.rounds)
    mem = build_mem_image(forest, inp)

    # Make a copy of mem BEFORE running anything on it
    from copy import copy
    mem_for_machine = copy(mem)
    mem_for_ref = copy(mem)

    # Run optimized kernel FIRST (on original mem copy)
    print(f"\nRunning optimized kernel...")
    machine = Machine(
        mem_for_machine,
        list(instrs),
        DebugInfo(scratch_map={}),
        n_cores=N_CORES,
        trace=False,
    )
    machine.enable_pause = False
    machine.enable_debug = False
    machine.prints = False
    machine.run()

    # Run reference kernel on separate copy
    print(f"\nRunning reference kernel (rounds={args.rounds}, batch={args.batch_size})...")
    trace_ref = {}
    ref_results = list(reference_kernel2(mem_for_ref, trace_ref))
    ref_mem = ref_results[-1]

    BATCH_IDX = args.batch_idx
    n_nodes = len(forest.values)  # 2**(height+1) - 1, as problem.Tree.generate

    print(f"\nReference kernel trace for batch index {BATCH_IDX}:")
    for h in range(args.rounds):
        idx = trace_ref.get((h, BATCH_IDX, "idx"), "?")
        val = trace_ref.get((h, BATCH_IDX, "val"), "?")
        node_val = trace_ref.get((h, BATCH_IDX, "node_val"), "?")
        val_after_xor = trace_ref.get((h, BATCH_IDX, "val_after_xor"), "?")
        hashed = trace_ref.get((h, BATCH_IDX, "hashed_val"), "?")
        next_idx = trace_ref.get((h, BATCH_IDX, "next_idx"), "?")
        wrapped = trace_ref.get((h, BATCH_IDX, "wrapped_idx"), "?")
        print(
            f"  Round {h}: idx={idx} -> node_val={node_val} -> "
            f"val_after_xor={val_after_xor} -> hashed={hashed} -> "
            f"next_idx={next_idx} -> wrapped={wrapped}"
        )

    # Extract results
    # Memory layout: header[0:8], forest_values at mem[4], inp_indices at mem[5], inp_values at mem[6]
    out_indices_p = ref_mem[5]  # inp_indices_p
    out_values_p = ref_mem[6]   # inp_values_p

    print(f"\nOutput memory locations: values at {out_values_p}, indices at {out_indices_p}")

    # Compare final values
    got_val = machine.mem[out_values_p + BATCH_IDX]
    ref_val = ref_mem[out_values_p + BATCH_IDX]
    got_idx = machine.mem[out_indices_p + BATCH_IDX]
    ref_idx = ref_mem[out_indices_p + BATCH_IDX]

    print(f"\nBatch index {BATCH_IDX}:")
    print(f"  Got  value={got_val}, idx={got_idx}")
    print(f"  Ref  value={ref_val}, idx={ref_idx}")
    print(f"  Match: value={'YES' if got_val == ref_val else 'NO'}, idx={'YES' if got_idx == ref_idx else 'NO'}")

    # Now let's check the scratch memory to find intermediate values
    # After the kernel runs, the scratch memory contains the final state
    # We need to look at the scratch addresses used for the BATCH_IDX element
    scratch = machine.cores[0].scratch

    print(f"\nScratch memory dump (first 64 words):")
    for i in range(0, min(64, SCRATCH_SIZE), 8):
        vals = [f"{scratch[i+j]:10d}" for j in range(8)]
        print(f"  [{i:4d}]: {' '.join(vals)}")

    # Check all batch elements for mismatches
    print(f"\nAll mismatching batch indices:")
    mismatches = []
    for i in range(args.batch_size):
        got_v = machine.mem[out_values_p + i]
        ref_v = ref_mem[out_values_p + i]
        got_i = machine.mem[out_indices_p + i]
        ref_i = ref_mem[out_indices_p + i]
        if got_v != ref_v or got_i != ref_i:
            mismatches.append(i)
            if len(mismatches) <= 20:
                print(f"  batch[{i}]: got_val={got_v} ref_val={ref_v} got_idx={got_i} ref_idx={ref_i}")

    if len(mismatches) > 20:
        print(f"  ... and {len(mismatches) - 20} more")
    print(f"  Total mismatches: {len(mismatches)} of {args.batch_size}")

    # Show which groups are affected
    groups = sorted(set(i // 8 for i in mismatches))
    print(f"  Affected groups: {groups}")

    # For each affected group, show all 8 elements
    for g in groups[:5]:
        print(f"\n  Group {g} (batch [{g*8}:{g*8+8}]):")
        for lane in range(8):
            bi = g * 8 + lane
            got_v = machine.mem[out_values_p + bi]
            ref_v = ref_mem[out_values_p + bi]
            got_i = machine.mem[out_indices_p + bi]
            ref_i = ref_mem[out_indices_p + bi]
            match = "OK" if (got_v == ref_v and got_i == ref_i) else "MISMATCH"
            print(f"    lane {lane} (batch {bi}): got_idx={got_i:4d} ref_idx={ref_i:4d} got_val={got_v:10d} ref_val={ref_v:10d} {match}")


if __name__ == "__main__":
    main()
