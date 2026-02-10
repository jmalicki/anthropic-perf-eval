#!/usr/bin/env python3
"""Print the SSA op that defines a given VReg (op with dest == VReg) from scheduler output JSON.

Use when debugging validation: confirm which op produces the value at a scratch address
(e.g. depth-3 tree-select result for value VReg 213).

  uv run python scripts/inspect_vreg_definer.py scheduler_output.json --vreg 213
  uv run python scripts/inspect_vreg_definer.py sched_4r8b.json --vreg 213 --width vector
"""

import argparse
import json
import sys


def find_vreg_key(vreg_to_addr, vreg_id: int, width: str) -> str | None:
    """Return the vreg_to_addr key for the given id and width, or None."""
    width_sub = "Vector" if width == "vector" else "Scalar"
    for k in vreg_to_addr:
        if f"id: {vreg_id}" in k and width_sub in k:
            return k
    return None


def main():
    ap = argparse.ArgumentParser(
        description="Find the op that defines a VReg (writes to its scratch address)"
    )
    ap.add_argument("json_path", help="Path to scheduler output JSON")
    ap.add_argument("--vreg", type=int, default=213, help="VReg id (default 213)")
    ap.add_argument(
        "--width",
        choices=("scalar", "vector"),
        default="vector",
        help="VReg width (default vector)",
    )
    args = ap.parse_args()

    with open(args.json_path) as f:
        data = json.load(f)

    vreg_to_addr = data.get("vreg_to_addr")
    if not vreg_to_addr:
        print("No vreg_to_addr in JSON", file=sys.stderr)
        return 1

    key = find_vreg_key(vreg_to_addr, args.vreg, args.width)
    if not key:
        print(f"No VReg with id={args.vreg} width={args.width} in vreg_to_addr", file=sys.stderr)
        return 1

    addr = vreg_to_addr[key]
    print(f"VReg id={args.vreg} width={args.width} -> scratch address {addr}")

    ops = data.get("ops", [])
    ops_writing_addr = [o for o in ops if addr in (o.get("writes") or [])]
    if not ops_writing_addr:
        print(f"No op with writes containing address {addr}")
        return 0

    # Prefer the op whose cycle matches this VReg's interval_start (from vregs array)
    def_cycle = None
    vregs_list = data.get("vregs")
    if vregs_list:
        entry = next(
            (v for v in vregs_list if v.get("id") == args.vreg and v.get("width", "").lower() == args.width),
            None,
        )
        if entry is not None:
            def_cycle = entry.get("interval_start")

    definers = [o for o in ops_writing_addr if o.get("cycle") == def_cycle] if def_cycle is not None else ops_writing_addr
    if not definers:
        definers = ops_writing_addr
    # If multiple ops at same cycle write to this address, that's a conflict
    if definers:
        c = definers[0].get("cycle")
        same_cycle = [o for o in ops_writing_addr if o.get("cycle") == c]
        if len(same_cycle) > 1:
            print(f"Warning: {len(same_cycle)} ops at cycle {c} write to address {addr} (same-cycle conflict)")
    for o in definers:
        print(f"\nDefining op: id={o.get('id')}  cycle={o.get('cycle')}  engine={o.get('engine')}  opcode={o.get('opcode')}")
        print(f"  slot={o.get('slot')}")
        print(f"  reads={o.get('reads')}")
        print(f"  writes={o.get('writes')}")

    if def_cycle is not None and len(ops_writing_addr) > 1:
        other = [o.get("id") for o in ops_writing_addr if o.get("cycle") != def_cycle]
        if other:
            print(f"\n(Other ops that reuse address {addr} at other cycles: {other})")

    # If vregs array present (liveness-detail full), show interval and metadata for this VReg
    if vregs_list:
        entry = next(
            (v for v in vregs_list if v.get("id") == args.vreg and v.get("width", "").lower() == args.width),
            None,
        )
        if entry:
            print(f"\nVReg from 'vregs' array: interval=[{entry.get('interval_start')}, {entry.get('interval_end')}]  var_type={entry.get('var_type')}  phase={entry.get('phase')}  round={entry.get('round')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
