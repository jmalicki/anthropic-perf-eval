#!/usr/bin/env python3
"""Find first instruction that reads from a scratch address not matching its SSA source VReg.

Used for root-cause analysis of 2r/16b zero-gap: some use reads from an address
that holds a different SSA value (e.g. consumer of constant 5 reads from slot that got 4).

Usage:
  uv run python -m scripts.find_wrong_scratch_use artifacts/scheduler_2r_16b_zero_gap.json [--addrs 1 2]
"""

import json
import argparse
import re


def vreg_key_from_operand(src):
    """Build vreg_to_addr key from SSA operand (dict with VReg id/width)."""
    if isinstance(src, dict) and "VReg" in src:
        v = src["VReg"]
        wid = v.get("width", "scalar")
        w = "Scalar" if wid == "scalar" else "Vector"
        return f"VReg {{ id: {v['id']}, width: {w} }}"
    return None


def get_expected_addrs_for_op(ssa_op, v2a, ssa_ops_by_id):
    """Return list of (slot_index, expected_addr) for each source that could be addr 1 or 2."""
    opcode = ssa_op.get("opcode")
    srcs = ssa_op.get("srcs", [])
    # Slot layout by opcode: [+, dest, src1, src2], [load, dest, addr], [vbroadcast, dest, src], etc.
    if opcode in ("+", "-", "*", "//", "^", "&", "|", "<<", ">>", "%", "<", "=="):
        # Binary ALU: slot = [opcode, dest, src1, src2]; read slots 2, 3
        if len(srcs) >= 2:
            k0 = vreg_key_from_operand(srcs[0])
            k1 = vreg_key_from_operand(srcs[1])
            a0 = v2a.get(k0) if k0 else None
            a1 = v2a.get(k1) if k1 else None
            return [(2, a0), (3, a1)]
    if opcode == "load":
        # load dest, addr -> slot [load, dest, addr]; read slot 2
        if len(srcs) >= 1:
            k = vreg_key_from_operand(srcs[0])
            return [(2, v2a.get(k) if k else None)]
    if opcode == "vbroadcast":
        # vbroadcast dest, src -> slot [vbroadcast, dest, src]; read slot 2
        if len(srcs) >= 1:
            k = vreg_key_from_operand(srcs[0])
            return [(2, v2a.get(k) if k else None)]
    if opcode == "const_choice":
        # ConstChoice ALU derive: slot [opcode, dest, left_addr, right_addr]. Use dep_ids as producers.
        dep_ids = ssa_op.get("dep_ids", [])
        if len(dep_ids) >= 2:
            left_dest = None
            right_dest = None
            for o in ssa_ops_by_id:
                if o.get("id") == dep_ids[0]:
                    left_dest = o.get("dest")
                    break
            for o in ssa_ops_by_id:
                if o.get("id") == dep_ids[1]:
                    right_dest = o.get("dest")
                    break
            if left_dest and right_dest:
                def dest_to_key(dest):
                    s = str(dest)
                    if s.startswith("v") and "_" in s:
                        id_part, w = s[1:].split("_")
                        vid = int(id_part)
                        width = "Scalar" if w == "s" else "Vector"
                        return f"VReg {{ id: {vid}, width: {width} }}"
                    return None
                k1 = dest_to_key(left_dest)
                k2 = dest_to_key(right_dest)
                return [(2, v2a.get(k1) if k1 else None), (3, v2a.get(k2) if k2 else None)]
    return []


def slot_int(slot, i):
    """Get slot[i] as int if possible."""
    if i >= len(slot):
        return None
    v = slot[i]
    if isinstance(v, int):
        return v
    if isinstance(v, str) and v.isdigit():
        return int(v)
    return None


def main():
    ap = argparse.ArgumentParser(description="Find first wrong scratch use (read addr != vreg_to_addr[SSA src])")
    ap.add_argument("json_path", help="Scheduler output JSON (failing run)")
    ap.add_argument("--addrs", type=int, nargs="+", default=[1, 2], help="Addresses to check (default: 1 2)")
    ap.add_argument("-v", "--verbose", action="store_true", help="Print every op that reads target addrs and its expected/actual")
    args = ap.parse_args()

    with open(args.json_path, encoding="utf-8") as f:
        data = json.load(f)

    v2a = data.get("vreg_to_addr", {})
    ssa_ops = data.get("ssa_ops", [])
    ssa_ops_by_id = ssa_ops  # iterate to find by id
    ops = data.get("ops", [])
    addrs = set(args.addrs)

    for op in ops:
        reads = op.get("reads", [])
        slot = op.get("slot", [])
        if not addrs.intersection(set(reads)):
            # Also check slot in case reads is incomplete
            for i in range(1, len(slot)):
                v = slot_int(slot, i)
                if v is not None and v in addrs:
                    break
            else:
                continue

        op_id = op.get("id")
        # Decode shifted ALU op id: id = ssa_id * 1000 + lane
        ssa_op_id = (op_id // 1000) if op_id >= 1000 else op_id
        cycle = op.get("cycle")
        engine = op.get("engine")
        opcode = op.get("opcode")

        # Find SSA op with this id (use decoded id for shifted ALU ops)
        ssa_op = None
        for o in ssa_ops:
            if o.get("id") == ssa_op_id:
                ssa_op = o
                break
        if not ssa_op:
            continue

        expected = get_expected_addrs_for_op(ssa_op, v2a, ssa_ops_by_id)
        if not expected:
            continue

        for slot_idx, expected_addr in expected:
            if expected_addr is None:
                continue
            actual = slot_int(slot, slot_idx)
            if actual is not None and actual in addrs:
                if args.verbose:
                    match = "OK" if actual == expected_addr else "MISMATCH"
                    print(f"  cycle={cycle} op_id={op_id} {opcode} slot[{slot_idx}] actual={actual} expected={expected_addr} {match}")
            # Only report when expected is one of the target addrs (e.g. 1 or 2) and actual is wrong
            if expected_addr not in addrs:
                continue
            if actual is not None and actual != expected_addr:
                print(
                    f"INCORRECT USE: cycle={cycle} engine={engine} opcode={opcode} op_id={op_id} ssa_op_id={ssa_op_id}\n"
                    f"  slot={slot}\n"
                    f"  slot[{slot_idx}] (read addr) = {actual}  expected (vreg_to_addr[SSA src]) = {expected_addr}\n"
                    f"  SSA op id={op_id} opcode={ssa_op.get('opcode')} srcs={ssa_op.get('srcs', [])[:4]}"
                )
                return 0

    print("No incorrect use found among ops reading the given addrs.")
    return 0


if __name__ == "__main__":
    exit(main())
