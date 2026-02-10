#!/usr/bin/env python3
"""
Check whether any instruction between vbroadcast cycles and FMA completion
overwrites the vector addresses used by Level 0 tree-select broadcasts.

Analyzes kernel_to_validate.json to find:
1. vbroadcast instructions at cycles 291-293 and confirm addresses
2. Any writes to those 8-word vector ranges between broadcast and last relevant FMA
3. FMA (multiply_add) instructions that reference those addresses as SOURCE operands
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

KERNEL_PATH = Path(__file__).parent.parent / "kernel_to_validate.json"

# All vector addresses we're tracking (8-word ranges)
TRACKED_ADDRS = {590, 102, 94, 672, 655, 663, 760, 768}
TRACKED_LABELS = {
    590: "tc_r3[0]",
    102: "tc_r3[2]",
    94:  "tc_r3[4]",
    672: "tc_r3[6]",
    655: "td_r3_01 delta",
    663: "td_r3_23 delta",
    760: "td_r3_45 delta",
    768: "td_r3_67 delta",
}


def is_vector_op(engine, op_name):
    """Determine if an operation writes a vector (8 words) or scalar (1 word)."""
    if engine == "valu":
        return True
    if engine == "load":
        return op_name == "vload"
    if engine == "flow":
        return op_name == "vselect"
    if engine == "alu":
        return False
    return False


def get_write_range(engine, op):
    """Get (start, end_exclusive) of addresses written, or None."""
    op_name = op[0]
    if engine == "store":
        return None
    if len(op) < 2:
        return None
    if op_name in ("jump", "cond_jump", "halt", "pause", "noop"):
        return None
    dest = op[1]
    if is_vector_op(engine, op_name):
        return (dest, dest + 8)
    else:
        return (dest, dest + 1)


def ranges_overlap(w_start, w_end, t_start, t_end):
    return w_start < t_end and t_start < w_end


def main():
    with open(KERNEL_PATH) as f:
        kernel = json.load(f)

    instrs = kernel["instrs"]
    print(f"Kernel: {len(instrs)} cycles, rounds={kernel.get('rounds')}, "
          f"batch_size={kernel.get('batch_size')}")
    print()

    # =========================================================================
    # 1. Find ALL vbroadcast instructions that write to tracked addresses
    # =========================================================================
    print("=" * 80)
    print("STEP 1: Find vbroadcast instructions writing to tracked addresses")
    print("=" * 80)

    # Find ALL broadcasts to tracked addrs (anywhere in kernel)
    found_broadcasts = {}  # addr -> list of (cycle,)
    for cycle_idx in range(len(instrs)):
        bundle = instrs[cycle_idx]
        for engine, ops in bundle.items():
            for op in ops:
                if op[0] == "vbroadcast" and op[1] in TRACKED_ADDRS:
                    addr = op[1]
                    if addr not in found_broadcasts:
                        found_broadcasts[addr] = []
                    found_broadcasts[addr].append(cycle_idx)

    # Show specifically around cycles 290-295
    print("\n  Broadcasts to tracked addresses near cycles 290-295:")
    for addr in sorted(TRACKED_ADDRS):
        label = TRACKED_LABELS[addr]
        cycles = found_broadcasts.get(addr, [])
        relevant = [c for c in cycles if 280 <= c <= 300]
        all_str = ", ".join(str(c) for c in cycles)
        rel_str = ", ".join(str(c) for c in relevant)
        print(f"    addr {addr:4d} ({label:20s}): near 290-295 = [{rel_str}], all = [{all_str}]")

    # Use the LAST broadcast near 290-295 as the "Level 0 broadcast"
    level0_broadcast_cycle = {}
    for addr in TRACKED_ADDRS:
        cycles = found_broadcasts.get(addr, [])
        relevant = [c for c in cycles if 280 <= c <= 300]
        if relevant:
            level0_broadcast_cycle[addr] = max(relevant)

    print(f"\n  Level 0 broadcast cycles identified:")
    for addr in sorted(level0_broadcast_cycle.keys()):
        print(f"    addr {addr}: cycle {level0_broadcast_cycle[addr]}")

    # =========================================================================
    # 2. Find multiply_add (FMA) instructions that READ tracked addresses
    #    as SOURCE operands (positions 3 or 4: multiply_add dest a b c)
    #    Only consider FMAs AFTER the broadcast cycle
    # =========================================================================
    print()
    print("=" * 80)
    print("STEP 2: Find multiply_add (FMA) using tracked addrs as SOURCE operands")
    print("=" * 80)

    # multiply_add format: ["multiply_add", dest, a, b, c]
    # FMA = a * b + c. Source operands are at indices 2, 3, 4.
    # We want FMAs where index 3 or 4 (or 2) matches a tracked addr,
    # and the FMA is AFTER the broadcast cycle for that addr.

    fma_reads_by_addr = defaultdict(list)  # addr -> [(cycle, op, which_src)]

    for cycle_idx in range(len(instrs)):
        bundle = instrs[cycle_idx]
        if "valu" not in bundle:
            continue
        for op in bundle["valu"]:
            if op[0] != "multiply_add":
                continue
            # Check source operands (indices 2, 3, 4)
            for src_idx in range(2, min(len(op), 5)):
                src_val = op[src_idx]
                if src_val in TRACKED_ADDRS:
                    bc = level0_broadcast_cycle.get(src_val)
                    if bc is not None and cycle_idx > bc:
                        src_label = ["", "dest", "a", "b", "c"][src_idx]
                        fma_reads_by_addr[src_val].append((cycle_idx, op, src_label))

    for addr in sorted(TRACKED_ADDRS):
        label = TRACKED_LABELS[addr]
        bc = level0_broadcast_cycle.get(addr)
        refs = fma_reads_by_addr.get(addr, [])
        if refs:
            cycles_list = [c for c, _, _ in refs]
            print(f"\n  addr {addr} ({label}), broadcast@{bc}: "
                  f"{len(refs)} FMA reads, cycles {min(cycles_list)}-{max(cycles_list)}")
            for cycle_idx, op, src_label in refs:
                print(f"    Cycle {cycle_idx}: {op}  (addr {addr} is src '{src_label}')")
        else:
            print(f"\n  addr {addr} ({label}), broadcast@{bc}: NO FMA reads after broadcast")

    # =========================================================================
    # 3. Scan for ANY write to tracked ranges between broadcast and last FMA read
    # =========================================================================
    print()
    print("=" * 80)
    print("STEP 3: Check for overwrites between broadcast and last FMA read")
    print("=" * 80)

    problems_found = 0

    for addr in sorted(TRACKED_ADDRS):
        label = TRACKED_LABELS[addr]
        bc = level0_broadcast_cycle.get(addr)
        if bc is None:
            print(f"\n  addr {addr} ({label}): NO broadcast found, skipping")
            continue

        refs = fma_reads_by_addr.get(addr, [])
        if not refs:
            print(f"\n  addr {addr} ({label}): NO FMA reads after broadcast@{bc}, skipping")
            continue

        last_fma_cycle = max(c for c, _, _ in refs)
        first_fma_cycle = min(c for c, _, _ in refs)

        # Danger window: (bc, last_fma_cycle] -- anything that writes to [addr, addr+8)
        # EXCLUDING the broadcast itself
        danger_writes = []
        for cycle_idx in range(bc + 1, last_fma_cycle + 1):
            if cycle_idx >= len(instrs):
                break
            bundle = instrs[cycle_idx]
            for engine, ops in bundle.items():
                for op in ops:
                    wr = get_write_range(engine, op)
                    if wr is None:
                        continue
                    w_start, w_end = wr
                    if ranges_overlap(w_start, w_end, addr, addr + 8):
                        # Is this one of the FMA reads themselves writing the dest?
                        # If the FMA reads addr as source AND writes to a different addr, it's fine.
                        # But if something writes TO [addr, addr+8) BEFORE the FMA reads it, that's bad.
                        # Check: is this write BEFORE any FMA that reads addr at this or later cycle?
                        fma_at_or_after = [c for c, _, _ in refs if c >= cycle_idx]
                        if fma_at_or_after:
                            # There's a FMA reading addr at or after this write cycle.
                            # But the write happens in the SAME cycle or BEFORE.
                            # Same-cycle: the write result isn't visible until next cycle,
                            # but the FMA read also happens this cycle. So same-cycle is OK
                            # (read-before-write in VLIW: both see the OLD value).
                            # But a write at cycle C means the value is corrupted from cycle C+1.
                            # So any FMA at cycle > C is reading corrupted data.
                            fma_after = [c for c, _, _ in refs if c > cycle_idx]
                            if fma_after:
                                danger_writes.append((cycle_idx, engine, op, w_start, w_end))
                            # If the only remaining FMAs are at exactly this cycle, it's OK
                            # (VLIW same-cycle reads see old values)

        if danger_writes:
            problems_found += 1
            print(f"\n  *** PROBLEM: addr {addr} ({label}) ***")
            print(f"      Broadcast at cycle {bc}")
            print(f"      FMA reads at cycles: {', '.join(str(c) for c,_,_ in refs)}")
            print(f"      {len(danger_writes)} writes in danger window "
                  f"(cycles {bc+1} to {last_fma_cycle}):")
            for c, e, o, ws, we in danger_writes:
                fma_after = sorted(set(fc for fc, _, _ in refs if fc > c))
                print(f"        Cycle {c} [{e}]: {o[0]} dest={o[1]} "
                      f"(words {ws}-{we-1}) -- FMAs reading AFTER: {fma_after}")
        else:
            print(f"\n  addr {addr} ({label}): SAFE")
            print(f"      Broadcast at cycle {bc}")
            print(f"      FMA reads at cycles: {', '.join(str(c) for c,_,_ in refs)}")
            print(f"      No intervening writes before last FMA read")

    # =========================================================================
    # 4. Also show what's at the cycles with the broadcasts for context
    # =========================================================================
    print()
    print("=" * 80)
    print("STEP 4: Full bundle contents at broadcast cycles (context)")
    print("=" * 80)

    bc_cycles = sorted(set(level0_broadcast_cycle.values()))
    for bc in bc_cycles:
        bundle = instrs[bc]
        print(f"\n  Cycle {bc}:")
        for engine in ["alu", "valu", "load", "store", "flow"]:
            if engine in bundle:
                print(f"    {engine}: ({len(bundle[engine])} ops)")
                for op in bundle[engine]:
                    tracked_marker = ""
                    if len(op) >= 2 and op[1] in TRACKED_ADDRS:
                        tracked_marker = f" <-- TRACKED ({TRACKED_LABELS[op[1]]})"
                    print(f"      {op}{tracked_marker}")

    print()
    if problems_found:
        print(f"RESULT: {problems_found} addresses have writes between broadcast and FMA read!")
        return 1
    else:
        print("RESULT: All broadcast addresses are safe -- no overwrites before FMA reads.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
