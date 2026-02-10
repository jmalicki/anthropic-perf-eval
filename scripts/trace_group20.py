#!/usr/bin/env python3
"""
Trace group 20 (batch indices 160-167) through the kernel to find where
the tree traversal diverges from the expected path.

Both greedy and CP-SAT kernels produce identical (wrong) results for group 20,
so the bug is in the kernel builder (SSA emission), not in the scheduler.

Approach:
1. Run the kernel step by step, recording scratch writes per cycle
2. After all cycles, compare final memory to reference
3. For each store to memory, trace which scratch address held the stored value
4. For the wrong values, trace backwards through scratch to find the first
   wrong intermediate value
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
    SCRATCH_SIZE,
    CoreState,
)


def run_kernel_tracing_scratch(instrs, mem):
    """
    Run kernel step by step. After each cycle, record ALL scratch writes
    and ALL memory writes.

    Returns:
      scratch_writes: list of (cycle, dict of addr->value) for each cycle
      mem_writes: list of (cycle, dict of addr->value) for each cycle
      final_scratch: the final scratch array
      final_mem: the final memory
    """
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

    scratch_writes = []
    mem_writes = []

    for cycle_idx in range(len(instrs)):
        if core.state != CoreState.RUNNING:
            break
        instr = instrs[cycle_idx]
        # step() sets scratch_write and mem_write dicts, applies them, then deletes
        # We need to intercept them. Easiest: call step and check before/after.
        # Actually, let's just patch.

        old_scratch = list(core.scratch)
        old_mem = list(machine.mem)

        core.pc += 1
        machine.step(instr, core)
        machine.cycle += 1

        # Find scratch changes
        sw = {}
        for addr in range(len(core.scratch)):
            if core.scratch[addr] != old_scratch[addr]:
                sw[addr] = core.scratch[addr]

        # Find mem changes
        mw = {}
        for addr in range(len(machine.mem)):
            if machine.mem[addr] != old_mem[addr]:
                mw[addr] = machine.mem[addr]

        if sw:
            scratch_writes.append((cycle_idx, sw))
        if mw:
            mem_writes.append((cycle_idx, mw))

    return scratch_writes, mem_writes, core.scratch, machine.mem


def find_scratch_addr_for_value(scratch_writes, target_value, before_cycle=None):
    """Find which scratch address(es) held a given value, searching backwards."""
    results = []
    for cycle, writes in reversed(scratch_writes):
        if before_cycle is not None and cycle >= before_cycle:
            continue
        for addr, val in writes.items():
            if val == target_value:
                results.append((cycle, addr))
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Trace a batch group through the kernel; find where results diverge from reference."
    )
    parser.add_argument("--kernel", default=None, help="Kernel JSON (default: kernel_treeselect_4round.json or kernel_to_validate.json)")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--group", type=int, default=20, help="Group index (batch indices group*8 .. group*8+7)")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--forest-height", type=int, default=10)
    args = parser.parse_args()

    if args.group * 8 + 8 > args.batch_size:
        print(f"Error: group {args.group} implies batch indices {args.group*8}-{args.group*8+7}, but batch_size={args.batch_size}")
        return 1

    seed = args.seed
    random.seed(seed)
    forest = Tree.generate(args.forest_height)
    inp = Input.generate(forest, args.batch_size, args.rounds)
    mem = build_mem_image(forest, inp)

    # Get reference trace
    trace = {}
    ref_mem = None
    for m in reference_kernel2(deepcopy(mem), trace):
        ref_mem = m

    inp_values_p = ref_mem[6]
    inp_indices_p = ref_mem[5]
    forest_values_p = ref_mem[4]
    n_nodes = ref_mem[1]
    batch_size = ref_mem[2]

    group_start = args.group * 8
    group_end = group_start + 8

    print("=" * 80)
    print(f"REFERENCE TRACE FOR GROUP {args.group} (batch {group_start}-{group_end-1})")
    print("=" * 80)
    for h in range(args.rounds):
        print(f"\n--- Round {h} ---")
        for i in range(group_start, group_end):
            idx = trace.get((h, i, "idx"), "?")
            val = trace.get((h, i, "val"), "?")
            node_val = trace.get((h, i, "node_val"), "?")
            val_after_xor = trace.get((h, i, "val_after_xor"), "?")
            hashed_val = trace.get((h, i, "hashed_val"), "?")
            next_idx = trace.get((h, i, "next_idx"), "?")
            wrapped_idx = trace.get((h, i, "wrapped_idx"), "?")
            print(
                f"  batch {i}: idx={idx:>4d}  val=0x{val:08x}  node_val=0x{node_val:08x}"
                f"  xor=0x{val_after_xor:08x}  hash=0x{hashed_val:08x}"
                f"  next={next_idx:>4d}  wrapped={wrapped_idx:>4d}"
            )

    # Load kernel
    if args.kernel and os.path.isfile(args.kernel):
        kernel_path = args.kernel
    else:
        kernel_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "kernel_treeselect_4round.json",
        )
        if not os.path.exists(kernel_path):
            kernel_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "kernel_to_validate.json",
            )
    print(f"\nUsing kernel: {kernel_path}")

    with open(kernel_path) as f:
        kernel_data = json.load(f)
    instrs = kernel_data["instrs"]

    print(f"\nRunning kernel ({len(instrs)} cycles)...")
    scratch_writes, mem_writes, final_scratch, final_mem = run_kernel_tracing_scratch(
        instrs, deepcopy(mem)
    )
    print(f"Total cycles with scratch writes: {len(scratch_writes)}")
    print(f"Total cycles with mem writes: {len(mem_writes)}")

    # Find memory addresses for group 20 outputs
    # inp_indices are at mem[inp_indices_p + i] for i in 160..167
    # inp_values are at mem[inp_values_p + i] for i in 160..167
    print("\n" + "=" * 80)
    print(f"FINAL MEMORY COMPARISON FOR GROUP {args.group}")
    print("=" * 80)
    for i in range(group_start, group_end):
        ref_val = ref_mem[inp_values_p + i]
        got_val = final_mem[inp_values_p + i]
        ref_idx = ref_mem[inp_indices_p + i]
        got_idx = final_mem[inp_indices_p + i]
        val_ok = "OK" if ref_val == got_val else "WRONG"
        idx_ok = "OK" if ref_idx == got_idx else "WRONG"
        print(
            f"  batch {i}: val got=0x{got_val:08x} exp=0x{ref_val:08x} [{val_ok}]"
            f"  idx got={got_idx} exp={ref_idx} [{idx_ok}]"
        )

    # Find which cycles wrote to group 20's memory addresses
    print("\n" + "=" * 80)
    print(f"MEMORY WRITES TO GROUP {args.group} ADDRESSES")
    print("=" * 80)
    group_mem_addrs = set()
    for i in range(group_start, group_end):
        group_mem_addrs.add(inp_values_p + i)
        group_mem_addrs.add(inp_indices_p + i)

    for cycle, writes in mem_writes:
        for addr, val in sorted(writes.items()):
            if addr in group_mem_addrs:
                if addr >= inp_values_p and addr < inp_values_p + batch_size:
                    batch_i = addr - inp_values_p
                    kind = "value"
                else:
                    batch_i = addr - inp_indices_p
                    kind = "index"
                ref_val = ref_mem[addr]
                ok = "OK" if val == ref_val else "WRONG"
                print(
                    f"  cycle {cycle:>3d}: batch {batch_i} {kind:>5s} = 0x{val:08x}"
                    f" (ref=0x{ref_val:08x}) [{ok}]"
                )

    # Now trace backwards: for each WRONG memory store, find the scratch address
    # that was the source
    print("\n" + "=" * 80)
    print("TRACING WRONG VALUES BACKWARDS")
    print("=" * 80)

    # To trace the source of a store, we need to look at the store instruction
    # The store instruction format is: ("store", addr_scratch, src_scratch) or
    # ("vstore", addr_scratch, src_scratch)
    # So we need the instruction at the cycle that wrote the wrong value

    for cycle, writes in mem_writes:
        for addr, val in sorted(writes.items()):
            if addr not in group_mem_addrs:
                continue
            ref_val = ref_mem[addr]
            if val == ref_val:
                continue

            if addr >= inp_values_p and addr < inp_values_p + batch_size:
                batch_i = addr - inp_values_p
                kind = "value"
            else:
                batch_i = addr - inp_indices_p
                kind = "index"

            print(f"\n  WRONG: cycle {cycle} batch {batch_i} {kind}")
            print(f"    got=0x{val:08x} expected=0x{ref_val:08x}")

            # Look at the instruction at that cycle
            instr = instrs[cycle]
            print(f"    instruction: {instr}")

            # Find the store that wrote to this mem address
            # Store ops: ("store", addr_reg, src_reg) or ("vstore", addr_reg, src_reg)
            if "store" in instr:
                for slot in instr["store"]:
                    print(f"    store slot: {slot}")

    # Now let's do something more targeted: find ALL scratch addresses that
    # hold values matching group 20's intermediate values, round by round

    print("\n" + "=" * 80)
    print("SEARCHING FOR INTERMEDIATE VALUES IN SCRATCH")
    print("=" * 80)

    # Build a map: cycle -> full scratch state
    # We need the full scratch at each cycle. Let's rebuild it.
    scratch_at_cycle = {}
    scratch_state = [0] * SCRATCH_SIZE
    for cycle, writes in scratch_writes:
        for addr, val in writes.items():
            scratch_state[addr] = val
        scratch_at_cycle[cycle] = dict(enumerate(scratch_state))

    # Get a list of all cycles that had writes
    write_cycles = sorted(scratch_at_cycle.keys())

    # For each reference intermediate value, find when/where it appears in scratch
    STAGE5_XOR_CONST = 0xB55A4F09

    # We care about: initial idx (all 0), initial val, node_val, val_after_xor, hashed_val, next_idx
    # The kernel uses optimized form (hashed_val has stage5 xor), and tree indices

    # Let's check the initial memory to see what values group 20 starts with
    print("\nInitial memory for group 20:")
    for i in range(group_start, group_end):
        init_val = mem[inp_values_p + i]
        init_idx = mem[inp_indices_p + i]
        print(f"  batch {i}: val=0x{init_val:08x} idx={init_idx}")

    # The tree values at the relevant indices
    print("\nTree values at traversal indices:")
    for h in range(4):
        for i in range(group_start, group_end):
            idx = trace.get((h, i, "idx"), 0)
            tree_val = mem[forest_values_p + idx] if idx < n_nodes else "OOB"
            print(f"  round {h} batch {i}: tree[{idx}] = 0x{tree_val:08x}" if isinstance(tree_val, int) else f"  round {h} batch {i}: tree[{idx}] = {tree_val}")

    # Now let's look at VSTORE operations that write to inp_indices and inp_values
    # memory ranges. The kernel uses vstores to write back results.
    print("\n" + "=" * 80)
    print("ALL VSTORE/STORE INSTRUCTIONS WRITING TO GROUP 20 MEM RANGE")
    print("=" * 80)

    for cycle_idx, instr in enumerate(instrs):
        if "store" not in instr:
            continue
        for slot in instr["store"]:
            if slot[0] == "vstore":
                # ("vstore", addr_scratch, src_scratch)
                # addr_scratch holds the base memory address
                # Need to know what scratch[addr_scratch] is at this cycle
                pass
            elif slot[0] == "store":
                pass

    # Better approach: look at the actual mem_writes and correlate with instructions
    # We already have mem_writes. Let's also track which scratch addresses are
    # read by each store instruction.

    print("\n" + "=" * 80)
    print("DETAILED STORE ANALYSIS FOR GROUP 20")
    print("=" * 80)

    # Rebuild scratch state incrementally for accurate reading
    scratch_state2 = [0] * SCRATCH_SIZE
    write_idx = 0

    for cycle_idx in range(len(instrs)):
        # Apply writes up to this cycle
        while write_idx < len(scratch_writes) and scratch_writes[write_idx][0] < cycle_idx:
            for addr, val in scratch_writes[write_idx][1].items():
                scratch_state2[addr] = val
            write_idx += 1

        # The scratch state BEFORE this cycle's execution is scratch_state2
        # (writes from this cycle haven't been applied yet)
        # Actually, step() reads from current scratch, then applies writes at end
        # So the reads happen from scratch state AFTER previous cycle's writes

        # Apply this cycle's writes
        if write_idx < len(scratch_writes) and scratch_writes[write_idx][0] == cycle_idx:
            this_cycle_writes = scratch_writes[write_idx][1]
        else:
            this_cycle_writes = {}

        instr = instrs[cycle_idx]
        if "store" not in instr:
            # Apply writes and continue
            for addr, val in this_cycle_writes.items():
                scratch_state2[addr] = val
            if write_idx < len(scratch_writes) and scratch_writes[write_idx][0] == cycle_idx:
                write_idx += 1
            continue

        for slot in instr["store"]:
            if slot[0] == "vstore":
                addr_reg = slot[1]
                src_reg = slot[2]
                mem_base = scratch_state2[addr_reg]
                # This writes mem[mem_base..mem_base+8] from scratch[src_reg..src_reg+8]
                for vi in range(VLEN):
                    target_mem = mem_base + vi
                    if target_mem in group_mem_addrs:
                        src_val = scratch_state2[src_reg + vi]
                        if target_mem >= inp_values_p and target_mem < inp_values_p + batch_size:
                            batch_i = target_mem - inp_values_p
                            kind = "value"
                        else:
                            batch_i = target_mem - inp_indices_p
                            kind = "index"
                        ref_val = ref_mem[target_mem]
                        ok = "OK" if src_val == ref_val else "WRONG"
                        print(
                            f"  cycle {cycle_idx:>3d}: vstore batch {batch_i} {kind}"
                            f" scratch[{src_reg + vi}]=0x{src_val:08x}"
                            f" (ref=0x{ref_val:08x}) [{ok}]"
                            f"  addr_reg=s{addr_reg}(=0x{mem_base:x})"
                        )

            elif slot[0] == "store":
                addr_reg = slot[1]
                src_reg = slot[2]
                target_mem = scratch_state2[addr_reg]
                if target_mem in group_mem_addrs:
                    src_val = scratch_state2[src_reg]
                    if target_mem >= inp_values_p and target_mem < inp_values_p + batch_size:
                        batch_i = target_mem - inp_values_p
                        kind = "value"
                    else:
                        batch_i = target_mem - inp_indices_p
                        kind = "index"
                    ref_val = ref_mem[target_mem]
                    ok = "OK" if src_val == ref_val else "WRONG"
                    print(
                        f"  cycle {cycle_idx:>3d}: store batch {batch_i} {kind}"
                        f" scratch[{src_reg}]=0x{src_val:08x}"
                        f" (ref=0x{ref_val:08x}) [{ok}]"
                        f"  addr_reg=s{addr_reg}(=0x{target_mem:x})"
                    )

        # Apply writes
        for addr, val in this_cycle_writes.items():
            scratch_state2[addr] = val
        if write_idx < len(scratch_writes) and scratch_writes[write_idx][0] == cycle_idx:
            write_idx += 1

    # Now trace backwards from the wrong stores to find the producer chain
    print("\n" + "=" * 80)
    print("BACKWARDS TRACE FROM WRONG STORES")
    print("=" * 80)

    # For each wrong store, find when the scratch address (src_reg) was last written
    # Then find what operation wrote it, and trace its inputs

    # Build a reverse map: for each scratch address, list of (cycle, value) when it was written
    addr_write_history = {}  # addr -> [(cycle, value), ...]
    for cycle, writes in scratch_writes:
        for addr, val in writes.items():
            if addr not in addr_write_history:
                addr_write_history[addr] = []
            addr_write_history[addr].append((cycle, val))

    def trace_scratch_addr(addr, before_cycle):
        """Find the most recent write to a scratch address before a given cycle."""
        if addr not in addr_write_history:
            return None, 0  # Never written = initial value 0
        history = addr_write_history[addr]
        for cycle, val in reversed(history):
            if cycle < before_cycle:
                return cycle, val
        return None, 0

    # For the detailed trace, focus on first elements of the group
    print(f"\nDetailed backwards trace for batch {group_start} and {group_start+2}:")

    # Redo the store scan but now trace backwards
    scratch_state3 = [0] * SCRATCH_SIZE
    write_idx3 = 0

    stores_for_group = []  # (cycle, batch_i, kind, src_addr, value, ref_value)

    for cycle_idx in range(len(instrs)):
        while write_idx3 < len(scratch_writes) and scratch_writes[write_idx3][0] < cycle_idx:
            for addr, val in scratch_writes[write_idx3][1].items():
                scratch_state3[addr] = val
            write_idx3 += 1

        if write_idx3 < len(scratch_writes) and scratch_writes[write_idx3][0] == cycle_idx:
            this_writes = scratch_writes[write_idx3][1]
        else:
            this_writes = {}

        instr = instrs[cycle_idx]
        if "store" in instr:
            for slot in instr["store"]:
                if slot[0] == "vstore":
                    addr_reg = slot[1]
                    src_reg = slot[2]
                    mem_base = scratch_state3[addr_reg]
                    for vi in range(VLEN):
                        target_mem = mem_base + vi
                        if target_mem in group_mem_addrs:
                            src_val = scratch_state3[src_reg + vi]
                            if target_mem >= inp_values_p and target_mem < inp_values_p + batch_size:
                                batch_i = target_mem - inp_values_p
                                kind = "value"
                            else:
                                batch_i = target_mem - inp_indices_p
                                kind = "index"
                            stores_for_group.append(
                                (cycle_idx, batch_i, kind, src_reg + vi, src_val, ref_mem[target_mem])
                            )
                elif slot[0] == "store":
                    addr_reg = slot[1]
                    src_reg = slot[2]
                    target_mem = scratch_state3[addr_reg]
                    if target_mem in group_mem_addrs:
                        src_val = scratch_state3[src_reg]
                        if target_mem >= inp_values_p and target_mem < inp_values_p + batch_size:
                            batch_i = target_mem - inp_values_p
                            kind = "value"
                        else:
                            batch_i = target_mem - inp_indices_p
                            kind = "index"
                        stores_for_group.append(
                            (cycle_idx, batch_i, kind, src_reg, src_val, ref_mem[target_mem])
                        )

        for addr, val in this_writes.items():
            scratch_state3[addr] = val
        if write_idx3 < len(scratch_writes) and scratch_writes[write_idx3][0] == cycle_idx:
            write_idx3 += 1

    for cycle, batch_i, kind, src_addr, val, ref_val in stores_for_group:
        ok = "OK" if val == ref_val else "WRONG"
        if val != ref_val:
            print(
                f"\n  cycle {cycle}: batch {batch_i} {kind} s{src_addr}=0x{val:08x}"
                f" ref=0x{ref_val:08x} [{ok}]"
            )

            # Trace backwards: when was s{src_addr} written?
            prod_cycle, prod_val = trace_scratch_addr(src_addr, cycle)
            print(f"    <- s{src_addr} last written at cycle {prod_cycle} with value 0x{prod_val:08x}")

            if prod_cycle is not None:
                prod_instr = instrs[prod_cycle]
                print(f"    <- instruction at cycle {prod_cycle}: {prod_instr}")

                # Find the specific operation that wrote to s{src_addr}
                for engine, slots in prod_instr.items():
                    if engine == "debug":
                        continue
                    for slot in slots:
                        if engine == "valu":
                            # dest is slot[1], writes to slot[1]..slot[1]+7
                            if len(slot) >= 2 and isinstance(slot[1], int):
                                dest = slot[1]
                                if dest <= src_addr < dest + VLEN:
                                    print(f"    <- matched {engine} slot: {slot}")
                                    # Trace the inputs
                                    if slot[0] == "vbroadcast":
                                        src = slot[2]
                                        sc, sv = trace_scratch_addr(src, prod_cycle)
                                        print(f"       input s{src} = 0x{sv:08x} (written at cycle {sc})")
                                    elif slot[0] == "multiply_add":
                                        offset = src_addr - dest
                                        for inp_idx, inp_name in [(2, "a"), (3, "b"), (4, "c")]:
                                            inp_addr = slot[inp_idx] + offset
                                            sc, sv = trace_scratch_addr(inp_addr, prod_cycle)
                                            print(f"       input {inp_name}: s{inp_addr} = 0x{sv:08x} (written at cycle {sc})")
                                    elif len(slot) == 4:
                                        offset = src_addr - dest
                                        for inp_idx, inp_name in [(2, "a"), (3, "b")]:
                                            inp_addr = slot[inp_idx] + offset
                                            sc, sv = trace_scratch_addr(inp_addr, prod_cycle)
                                            print(f"       input {inp_name}: s{inp_addr} = 0x{sv:08x} (written at cycle {sc})")
                        elif engine in ("alu", "flow"):
                            if len(slot) >= 2 and isinstance(slot[1], int):
                                dest = slot[1]
                                if dest == src_addr:
                                    print(f"    <- matched {engine} slot: {slot}")
                                    if slot[0] == "select":
                                        cond_addr = slot[2]
                                        a_addr = slot[3]
                                        b_addr = slot[4]
                                        for name, addr in [("cond", cond_addr), ("a", a_addr), ("b", b_addr)]:
                                            sc, sv = trace_scratch_addr(addr, prod_cycle)
                                            print(f"       input {name}: s{addr} = 0x{sv:08x} (written at cycle {sc})")
                                    elif slot[0] == "vselect":
                                        offset = src_addr - dest
                                        for inp_idx, inp_name in [(2, "cond"), (3, "a"), (4, "b")]:
                                            inp_addr = slot[inp_idx] + offset
                                            sc, sv = trace_scratch_addr(inp_addr, prod_cycle)
                                            print(f"       input {inp_name}: s{inp_addr} = 0x{sv:08x} (written at cycle {sc})")
                                    elif slot[0] == "add_imm":
                                        inp_addr = slot[2]
                                        imm = slot[3]
                                        sc, sv = trace_scratch_addr(inp_addr, prod_cycle)
                                        print(f"       input a: s{inp_addr} = 0x{sv:08x} (written at cycle {sc})")
                                        print(f"       imm: {imm}")
                                    elif len(slot) == 4:
                                        for inp_idx, inp_name in [(2, "a"), (3, "b")]:
                                            inp_addr = slot[inp_idx]
                                            sc, sv = trace_scratch_addr(inp_addr, prod_cycle)
                                            print(f"       input {inp_name}: s{inp_addr} = 0x{sv:08x} (written at cycle {sc})")

    # Now let's find the FIRST wrong scratch write by comparing against reference values
    print("\n" + "=" * 80)
    print("FINDING SCRATCH ADDRESSES USED BY GROUP 20")
    print("=" * 80)

    # Strategy: find all scratch addresses that participate in the dataflow
    # leading to the wrong stores.
    # Start from the wrong store src_addrs, then trace backwards recursively.

    # Collect all wrong store source addresses
    wrong_store_addrs = set()
    for cycle, batch_i, kind, src_addr, val, ref_val in stores_for_group:
        if val != ref_val:
            wrong_store_addrs.add((src_addr, cycle))

    # For each wrong address, trace the full producer tree (up to depth 10)
    traced_addrs = set()  # (addr, cycle) pairs we've traced

    def trace_producer_tree(addr, before_cycle, depth=0, prefix=""):
        """Recursively trace the producer of a scratch address."""
        key = (addr, before_cycle)
        if key in traced_addrs or depth > 15:
            return
        traced_addrs.add(key)

        prod_cycle, prod_val = trace_scratch_addr(addr, before_cycle)
        if prod_cycle is None:
            print(f"{prefix}s{addr} = 0x{prod_val:08x} (initial/never written)")
            return

        prod_instr = instrs[prod_cycle]
        # Find which op wrote to this addr
        found = False
        for engine, slots in prod_instr.items():
            if engine == "debug":
                continue
            for slot in slots:
                if engine == "valu":
                    if len(slot) >= 2 and isinstance(slot[1], int):
                        dest = slot[1]
                        if dest <= addr < dest + VLEN:
                            offset = addr - dest
                            print(f"{prefix}s{addr} = 0x{prod_val:08x} <- cycle {prod_cycle} {engine} {slot[0]} (offset={offset})")
                            found = True
                            if slot[0] == "vbroadcast":
                                trace_producer_tree(slot[2], prod_cycle, depth + 1, prefix + "  ")
                            elif slot[0] == "multiply_add":
                                for inp_idx, inp_name in [(2, "a"), (3, "b"), (4, "c")]:
                                    inp_addr = slot[inp_idx] + offset
                                    trace_producer_tree(inp_addr, prod_cycle, depth + 1, prefix + "  ")
                            elif len(slot) == 4:
                                for inp_idx in [2, 3]:
                                    inp_addr = slot[inp_idx] + offset
                                    trace_producer_tree(inp_addr, prod_cycle, depth + 1, prefix + "  ")
                elif engine == "load":
                    if len(slot) >= 2 and isinstance(slot[1], int):
                        if slot[0] == "vload":
                            dest = slot[1]
                            if dest <= addr < dest + VLEN:
                                offset = addr - dest
                                # vload loads from mem[scratch[addr_reg] + offset]
                                addr_reg = slot[2]
                                ac, av = trace_scratch_addr(addr_reg, prod_cycle)
                                mem_addr = av + offset
                                print(f"{prefix}s{addr} = 0x{prod_val:08x} <- cycle {prod_cycle} vload mem[{mem_addr}] (s{addr_reg}={av})")
                                found = True
                        elif slot[0] == "load":
                            dest = slot[1]
                            if dest == addr:
                                addr_reg = slot[2]
                                ac, av = trace_scratch_addr(addr_reg, prod_cycle)
                                print(f"{prefix}s{addr} = 0x{prod_val:08x} <- cycle {prod_cycle} load mem[{av}] (s{addr_reg})")
                                found = True
                        elif slot[0] == "load_offset":
                            dest = slot[1]
                            offset = slot[3]
                            if dest + offset == addr:
                                addr_reg = slot[2]
                                ac, av = trace_scratch_addr(addr_reg + offset, prod_cycle)
                                print(f"{prefix}s{addr} = 0x{prod_val:08x} <- cycle {prod_cycle} load_offset mem[{av}] (s{addr_reg + offset})")
                                found = True
                        elif slot[0] == "const":
                            dest = slot[1]
                            if dest == addr:
                                print(f"{prefix}s{addr} = 0x{prod_val:08x} <- cycle {prod_cycle} const {slot[2]}")
                                found = True
                elif engine in ("alu",):
                    if len(slot) >= 2 and isinstance(slot[1], int):
                        dest = slot[1]
                        if dest == addr:
                            print(f"{prefix}s{addr} = 0x{prod_val:08x} <- cycle {prod_cycle} {engine} {slot[0]}")
                            found = True
                            if len(slot) == 4:
                                for inp_idx in [2, 3]:
                                    trace_producer_tree(slot[inp_idx], prod_cycle, depth + 1, prefix + "  ")
                elif engine == "flow":
                    if len(slot) >= 2 and isinstance(slot[1], int):
                        dest = slot[1]
                        if dest == addr:
                            found = True
                            if slot[0] == "vselect":
                                print(f"{prefix}s{addr} = 0x{prod_val:08x} <- cycle {prod_cycle} vselect")
                                offset = addr - dest
                                for inp_idx, inp_name in [(2, "cond"), (3, "a"), (4, "b")]:
                                    inp_addr = slot[inp_idx] + offset
                                    trace_producer_tree(inp_addr, prod_cycle, depth + 1, prefix + "  ")
                            elif slot[0] == "select":
                                print(f"{prefix}s{addr} = 0x{prod_val:08x} <- cycle {prod_cycle} select")
                                for inp_idx in [2, 3, 4]:
                                    trace_producer_tree(slot[inp_idx], prod_cycle, depth + 1, prefix + "  ")
                            elif slot[0] == "add_imm":
                                print(f"{prefix}s{addr} = 0x{prod_val:08x} <- cycle {prod_cycle} add_imm +{slot[3]}")
                                trace_producer_tree(slot[2], prod_cycle, depth + 1, prefix + "  ")
                            else:
                                print(f"{prefix}s{addr} = 0x{prod_val:08x} <- cycle {prod_cycle} {engine} {slot}")

                if found:
                    break
            if found:
                break

        if not found:
            print(f"{prefix}s{addr} = 0x{prod_val:08x} <- cycle {prod_cycle} (op not found in instruction)")

    # Trace the producer tree for each wrong store
    print("\nProducer trees for wrong index stores:")
    for cycle, batch_i, kind, src_addr, val, ref_val in stores_for_group:
        if val == ref_val:
            continue
        if kind != "index":
            continue
        # Only trace first couple to keep output manageable
        if batch_i > group_start + 2:
            continue
        print(f"\n  batch {batch_i} {kind}: got=0x{val:08x} expected=0x{ref_val:08x}")
        trace_producer_tree(src_addr, cycle + 1, 0, "    ")

    # Most importantly: compare scratch states between the two kernels at each cycle
    # Since both have 445 cycles and produce the same (wrong) results, let's
    # compare with a REFERENCE execution instead.
    #
    # Strategy: find all scratch addresses where the kernel writes a value that
    # matches a known WRONG intermediate value (one that differs from reference)

    # First, let's check if the tree node value loaded for group 20 in each round is correct
    print("\n" + "=" * 80)
    print("CHECKING NODE VALUE LOADS FOR GROUP 20")
    print("=" * 80)

    # The kernel loads tree[idx] for each element. For group 20, the idx values
    # determine which tree node is loaded. If idx is wrong, node_val will be wrong
    # and everything downstream will be wrong.
    #
    # Reference round 0: all idx=0, so tree[0] is loaded for all
    # Reference round 1: idx depends on round 0 result
    #
    # The KEY question is: does the kernel use the RIGHT idx for each round?
    # If not, WHEN does it first use a wrong idx?

    # Let's search for the tree values that should be loaded
    print("\nExpected tree values per round:")
    for h in range(4):
        print(f"  Round {h}:")
        for i in range(group_start, group_end):
            idx = trace[(h, i, "idx")]
            tree_val = mem[forest_values_p + idx]
            print(f"    batch {i}: tree[{idx}] = 0x{tree_val:08x}")

    # Search for these tree values in scratch at appropriate times
    # The tree values are loaded via vload or load_offset instructions
    # Let's find when tree values for group 20 appear in scratch

    # For round 0, all elements use idx=0, so tree[0] should be loaded
    tree_val_0 = mem[forest_values_p + 0]  # 0x06b3d9b3
    print(f"\nTree[0] = 0x{tree_val_0:08x}")

    # Find all scratch writes of this value
    print("Scratch writes of tree[0]:")
    count = 0
    for cycle, writes in scratch_writes:
        for addr, val in writes.items():
            if val == tree_val_0:
                count += 1
                if count <= 20:
                    print(f"  cycle {cycle}: s{addr} = 0x{val:08x}")
    if count > 20:
        print(f"  ... and {count - 20} more")

    # For the scatter load approach, tree values are loaded via scalar loads
    # after decomposing the address vector. Let's find the loads for group 20
    # specifically by looking at load_offset and load instructions near
    # addresses that could hold group 20 data

    # Actually, let's take a completely different, more efficient approach:
    # Find the FIRST cycle where group 20's data diverges from expected

    print("\n" + "=" * 80)
    print("CYCLE-BY-CYCLE DIVERGENCE ANALYSIS")
    print("=" * 80)

    # Run reference kernel to get ALL intermediate values per batch element per round
    # Then search scratch for these values to see when they appear/disappear

    # For each round, the reference has:
    # - idx (tree index, used to load node_val)
    # - val (input value)
    # - node_val = tree[idx]
    # - val_after_xor = val ^ node_val (with stage5 adjustment for rounds > 0)
    # - hashed_val = myhash_opt(val_after_xor)
    # - next_idx = 2*idx + branch_offset
    # - wrapped_idx = 0 if next_idx >= n_nodes else next_idx

    # The kernel processes all elements in parallel (vector ops), so for group 20
    # (elements 160-167), these values should appear as a vector in 8 consecutive
    # scratch addresses at some point.

    # Let's search for the expected hashed values for each round
    for h in range(4):
        print(f"\nRound {h}: expected hashed values for group 20:")
        expected_hash = []
        for i in range(group_start, group_end):
            hv = trace[(h, i, "hashed_val")]
            expected_hash.append(hv)
            print(f"  batch {i}: 0x{hv:08x}")

        # Search for the first element of this hash vector in scratch
        target = expected_hash[0]
        print(f"  Searching for first element 0x{target:08x}...")
        found = False
        for cycle, writes in scratch_writes:
            for addr, val in writes.items():
                if val == target:
                    # Check if the next 7 addresses match too
                    # Need full scratch state at this cycle
                    # Build it incrementally... too slow for full scan
                    # Just report the finding
                    print(f"    Found at cycle {cycle}, s{addr}")
                    found = True
        if not found:
            print(f"    NOT FOUND anywhere in scratch!")

    # Now let's look at the expected idx values after each round
    for h in range(4):
        print(f"\nRound {h}: expected wrapped_idx for group 20:")
        expected_idx = []
        for i in range(group_start, group_end):
            wi = trace[(h, i, "wrapped_idx")]
            expected_idx.append(wi)
        print(f"  {expected_idx}")

        # Search for these as a vector pattern
        # The idx values should appear as 8 consecutive scratch words
        target = expected_idx[0]
        print(f"  Searching for first element idx={target}...")
        found_cycles = []
        for cycle, writes in scratch_writes:
            for addr, val in writes.items():
                if val == target:
                    found_cycles.append((cycle, addr))
        if found_cycles:
            print(f"    Found at: {found_cycles[:10]}{'...' if len(found_cycles) > 10 else ''}")
        else:
            print(f"    NOT FOUND")

    # Final analysis: check if the tree values loaded for group 20 are correct
    # by looking at what memory addresses the loads use
    print("\n" + "=" * 80)
    print("LOAD INSTRUCTION ANALYSIS FOR SCATTER LOADS")
    print("=" * 80)

    # In scatter load, the kernel decomposes the address vector into 8 scalars,
    # then does 8 scalar loads. Let's find load_offset instructions.

    load_offset_count = 0
    scalar_load_count = 0
    for cycle_idx, instr in enumerate(instrs):
        if "load" not in instr:
            continue
        for slot in instr["load"]:
            if slot[0] == "load_offset":
                load_offset_count += 1
            elif slot[0] == "load":
                scalar_load_count += 1

    print(f"Total load_offset instructions: {load_offset_count}")
    print(f"Total scalar load instructions: {scalar_load_count}")

    # Find all scalar loads and check if any load from wrong addresses
    # for group 20's tree node values

    # Expected memory addresses for tree node loads:
    print("\nExpected tree load addresses for group 20:")
    for h in range(4):
        print(f"  Round {h}:")
        for i in range(group_start, group_end):
            idx = trace[(h, i, "idx")]
            mem_addr = forest_values_p + idx
            print(f"    batch {i}: mem[{mem_addr}] = tree[{idx}]")


if __name__ == "__main__":
    main()
