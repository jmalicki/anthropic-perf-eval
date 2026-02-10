#!/usr/bin/env python3
"""Compare depth-3 round: kernel output vs reference node_val / val_after_xor / hashed_val.

Use when debugging depth-3 (4r/8b) wrong value: determine whether the wrong value
comes from (A) wrong tree selection (value matches ref node_val for another lane),
(B) wrong XOR, or (C) wrong hash/branch (value matches ref hashed_val or something else).

Usage:
  uv run python scripts/compare_depth3_tree_value.py --kernel kernel_to_validate.json --rounds 4 --batch-size 8
  uv run python scripts/compare_depth3_tree_value.py --rounds 4 --batch-size 8 --group 0

Output:
- Reference trace for the depth-3 round (round 3 for 4 rounds): node_val, val_after_xor, hashed_val per batch index.
- Kernel run: final got vs ref, list of mismatching batch indices.
- For each wrong value store: comparison of got value to ref node_val, val_after_xor, hashed_val
  (same lane and other lanes in the group) to suggest tree selection vs hash vs other.
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
    VLEN,
    CoreState,
)


def run_kernel_tracing_mem(instrs, mem):
    """Run kernel; return final memory and list of (cycle, addr, value) for each mem write."""
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
    mem_writes = []
    for cycle_idx in range(len(instrs)):
        if core.state != CoreState.RUNNING:
            break
        instr = instrs[cycle_idx]
        old_mem = list(machine.mem)
        core.pc += 1
        machine.step(instr, core)
        machine.cycle += 1
        for addr in range(len(machine.mem)):
            if machine.mem[addr] != old_mem[addr]:
                mem_writes.append((cycle_idx, addr, machine.mem[addr]))
    return machine.mem, mem_writes


def depth3_round_index(rounds: int, forest_height: int) -> int:
    """Round index where tree_depth == 3 (mirrors reference: depth increments after each round)."""
    tree_height = forest_height + 1
    # At start of round 0 depth=0, round 1 depth=1, round 2 depth=2, round 3 depth=3
    if rounds <= 3:
        return -1
    return 3


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Compare depth-3 round: reference node_val/val_after_xor/hashed vs kernel wrong values.",
    )
    parser.add_argument("--kernel", default="kernel_to_validate.json")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--forest-height", type=int, default=5)
    parser.add_argument("--group", type=int, default=None, help="If set, only print this group's lanes (batch group*8..group*8+7)")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    if args.batch_size % VLEN != 0:
        print(f"Error: batch_size must be a multiple of VLEN ({VLEN})")
        return 1

    random.seed(args.seed)
    forest = Tree.generate(args.forest_height)
    inp = Input.generate(forest, args.batch_size, args.rounds)
    mem = build_mem_image(forest, inp)

    # Reference trace
    trace_ref = {}
    ref_mem = None
    for m in reference_kernel2(deepcopy(mem), trace_ref):
        ref_mem = m

    inp_values_p = ref_mem[6]
    inp_indices_p = ref_mem[5]
    n_nodes = ref_mem[1]
    batch_size = ref_mem[2]

    depth3_r = depth3_round_index(args.rounds, args.forest_height)
    if depth3_r < 0:
        print("No depth-3 round for this rounds/height.")
        return 0

    # Reference table for depth-3 round
    print("=" * 80)
    print(f"REFERENCE: depth-3 round (round {depth3_r}) — node_val, val_after_xor, hashed_val")
    print("=" * 80)

    if args.group is not None:
        start = args.group * 8
        end = min(start + 8, batch_size)
        indices = range(start, end)
    else:
        indices = range(batch_size)

    def fmt(v):
        return f"0x{v:08x}" if isinstance(v, int) else (str(v) if v is not None else "?")

    for i in indices:
        node_val = trace_ref.get((depth3_r, i, "node_val"))
        val_after_xor = trace_ref.get((depth3_r, i, "val_after_xor"))
        hashed_val = trace_ref.get((depth3_r, i, "hashed_val"))
        idx = trace_ref.get((depth3_r, i, "idx"), "?")
        print(
            f"  batch {i:>3d}: idx={idx}  node_val={fmt(node_val)}  "
            f"val_after_xor={fmt(val_after_xor)}  hashed_val={fmt(hashed_val)}"
        )

    # Load and run kernel
    if not os.path.isfile(args.kernel):
        print(f"Kernel not found: {args.kernel}")
        return 1
    with open(args.kernel) as f:
        data = json.load(f)
    instrs = data.get("instrs", data) if isinstance(data, dict) else data

    print(f"\nRunning kernel ({len(instrs)} cycles)...")
    final_mem, mem_writes = run_kernel_tracing_mem(instrs, deepcopy(mem))

    # Final comparison
    print("\n" + "=" * 80)
    print("FINAL: got vs ref (value and index)")
    print("=" * 80)
    mismatches = []
    for i in indices:
        ref_val = ref_mem[inp_values_p + i]
        got_val = final_mem[inp_values_p + i]
        ref_idx = ref_mem[inp_indices_p + i]
        got_idx = final_mem[inp_indices_p + i]
        val_ok = ref_val == got_val
        idx_ok = ref_idx == got_idx
        if not val_ok or not idx_ok:
            mismatches.append(i)
        print(
            f"  batch {i:>3d}: val got=0x{got_val:08x} ref=0x{ref_val:08x} {'OK' if val_ok else 'WRONG'}  "
            f"idx got={got_idx} ref={ref_idx} {'OK' if idx_ok else 'WRONG'}"
        )

    if not mismatches:
        print("\nNo mismatches; kernel matches reference for this config.")
        return 0

    # For each wrong value, compare to ref node_val / val_after_xor / hashed_val
    print("\n" + "=" * 80)
    print("WRONG VALUE ANALYSIS: compare got value to reference quantities")
    print("=" * 80)

    group_mem_addrs = set()
    for i in range(batch_size):
        group_mem_addrs.add(inp_values_p + i)
        group_mem_addrs.add(inp_indices_p + i)

    wrong_value_stores = []
    for cycle, addr, val in mem_writes:
        if addr not in group_mem_addrs or addr < inp_values_p or addr >= inp_values_p + batch_size:
            continue
        batch_i = addr - inp_values_p
        ref_val = ref_mem[addr]
        if val != ref_val:
            wrong_value_stores.append((cycle, batch_i, val, ref_val))

    for cycle, batch_i, got_val, ref_val in wrong_value_stores:
        print(f"\n  WRONG value: cycle {cycle} batch {batch_i}  got=0x{got_val:08x}  ref=0x{ref_val:08x}")

        # Same lane: ref quantities for this batch at depth-3 round and last round
        node_val_bi = trace_ref.get((depth3_r, batch_i, "node_val"))
        xor_bi = trace_ref.get((depth3_r, batch_i, "val_after_xor"))
        hash_bi = trace_ref.get((depth3_r, batch_i, "hashed_val"))

        if node_val_bi is not None and got_val == node_val_bi:
            print(f"    -> got matches REF node_val for same batch {batch_i} (depth-3 round) [tree selection correct for lane, wrong elsewhere?]")
        elif xor_bi is not None and got_val == xor_bi:
            print(f"    -> got matches REF val_after_xor for same batch {batch_i} [bug likely in hash or after]")
        elif hash_bi is not None and got_val == hash_bi:
            print(f"    -> got matches REF hashed_val for same batch {batch_i} [suggests wrong round or index]")
        else:
            # Other lanes: does got match another lane's node_val? (tree selection mix-up)
            found = False
            for other in range(batch_size):
                if other == batch_i:
                    continue
                no = trace_ref.get((depth3_r, other, "node_val"))
                if no is not None and got_val == no:
                    print(f"    -> got matches REF node_val for *other* batch {other} [TREE SELECTION MIX-UP: wrong lane selected]")
                    found = True
                    break
            if not found:
                for other in range(batch_size):
                    ho = trace_ref.get((depth3_r, other, "hashed_val"))
                    if ho is not None and got_val == ho:
                        print(f"    -> got matches REF hashed_val for other batch {other}")
                        found = True
                        break
            if not found:
                print(f"    -> got does not match any ref node_val/val_after_xor/hashed_val for depth-3 round (may be from another round or corrupted)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
