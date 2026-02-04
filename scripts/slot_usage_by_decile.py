#!/usr/bin/env python3
"""
Report slot usage and scratch by decile of the program (first 10% of cycles, next 10%, etc.)
from a scheduler output JSON (e.g. from generate_ssa_kernel.py --output-json scheduler_output.json).

Usage (from project root):
  uv run python scripts/slot_usage_by_decile.py [scheduler_output.json]
  uv run python scripts/slot_usage_by_decile.py   # uses scheduler_output.json

Requires scheduler output with `ops` (and optionally `liveness_curve` for scratch).
Generate with: uv run generate_ssa_kernel.py --output-json scheduler_output.json
"""

import argparse
import json
import sys

# Slot limits per cycle (match problem.py / ISA)
SLOT_LIMITS = {
    "alu": 12,
    "valu": 6,
    "load": 2,
    "store": 2,
    "flow": 1,
}


def main():
    parser = argparse.ArgumentParser(
        description="Report slot usage and scratch by decile from scheduler output JSON"
    )
    parser.add_argument(
        "input_json",
        nargs="?",
        default="scheduler_output.json",
        help="Scheduler output JSON path (default: scheduler_output.json)",
    )
    args = parser.parse_args()

    try:
        with open(args.input_json) as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: file not found: {args.input_json}", file=sys.stderr)
        print(
            "Generate it with: uv run generate_ssa_kernel.py --output-json scheduler_output.json",
            file=sys.stderr,
        )
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON in {args.input_json}: {e}", file=sys.stderr)
        return 1

    total_cycles = data.get("total_cycles")
    ops = data.get("ops", [])
    liveness_curve = data.get("liveness_curve")

    if not total_cycles and ops:
        total_cycles = max(op["cycle"] for op in ops) + 1
    if not total_cycles:
        print("Error: no total_cycles and no ops in JSON", file=sys.stderr)
        return 1

    # Per-cycle slot counts (indexed by cycle)
    n = total_cycles
    alu = [0] * n
    valu_vbroadcast = [0] * n
    valu_fma = [0] * n
    valu_other = [0] * n
    load = [0] * n
    store = [0] * n
    flow = [0] * n
    scratch = [0] * n  # filled from liveness_curve if present

    for op in ops:
        c = op["cycle"]
        if c < 0 or c >= n:
            continue
        eng = op.get("engine", "").lower()
        opcode = (op.get("opcode") or "").strip()

        if eng == "alu":
            alu[c] += 1
        elif eng == "valu":
            if opcode == "vbroadcast":
                valu_vbroadcast[c] += 1
            elif opcode in ("multiply_add", "-"):
                valu_fma[c] += 1
            else:
                valu_other[c] += 1
        elif eng == "load":
            load[c] += 1
        elif eng == "store":
            store[c] += 1
        elif eng == "flow":
            flow[c] += 1

    if liveness_curve:
        for entry in liveness_curve:
            c = entry.get("cycle", 0)
            if 0 <= c < n:
                scratch[c] = entry.get("total_live_words", 0)

    # Deciles: 0 = first 10% of cycles, 1 = next 10%, ... 9 = last 10%
    num_deciles = 10
    decile_cycles = [[] for _ in range(num_deciles)]
    for cycle in range(n):
        d = (cycle * num_deciles) // n
        if d >= num_deciles:
            d = num_deciles - 1
        decile_cycles[d].append(cycle)

    # Table header
    print(f"Slot usage and scratch by decile (total cycles = {n})")
    print(f"Input: {args.input_json}")
    print()

    # Column headers: Decile | Cycle range | ALU | VALU (vbroadcast | FMA | other) | Flow | Load | Store | Scratch
    header = (
        "Decile | Cycle range      | ALU (used/cap %)   | VALU total (vbroadcast | FMA | other) | Flow   | Load  | Store | Scratch (words)"
    )
    print(header)
    print("-" * len(header))

    for d in range(num_deciles):
        cycles = decile_cycles[d]
        if not cycles:
            continue
        c_lo, c_hi = min(cycles), max(cycles)
        cycle_range = f"{c_lo}-{c_hi}"
        n_cy = len(cycles)

        # Sum slot usage in this decile
        alu_used = sum(alu[c] for c in cycles)
        vb_used = sum(valu_vbroadcast[c] for c in cycles)
        fma_used = sum(valu_fma[c] for c in cycles)
        other_used = sum(valu_other[c] for c in cycles)
        valu_used = vb_used + fma_used + other_used
        load_used = sum(load[c] for c in cycles)
        store_used = sum(store[c] for c in cycles)
        flow_used = sum(flow[c] for c in cycles)

        alu_cap = SLOT_LIMITS["alu"] * n_cy
        valu_cap = SLOT_LIMITS["valu"] * n_cy
        load_cap = SLOT_LIMITS["load"] * n_cy
        store_cap = SLOT_LIMITS["store"] * n_cy
        flow_cap = SLOT_LIMITS["flow"] * n_cy

        alu_pct = (100.0 * alu_used / alu_cap) if alu_cap else 0
        valu_pct = (100.0 * valu_used / valu_cap) if valu_cap else 0
        load_pct = (100.0 * load_used / load_cap) if load_cap else 0
        store_pct = (100.0 * store_used / store_cap) if store_cap else 0
        flow_pct = (100.0 * flow_used / flow_cap) if flow_cap else 0

        if liveness_curve:
            scratch_avg = sum(scratch[c] for c in cycles) / n_cy
            scratch_max = max(scratch[c] for c in cycles)
            scratch_str = f"avg {scratch_avg:.0f} max {scratch_max}"
        else:
            scratch_str = "N/A (no liveness_curve)"

        alu_str = f"{alu_used}/{alu_cap} ({alu_pct:.1f}%)"
        valu_break = f"{valu_used} ({vb_used} | {fma_used} | {other_used})"
        flow_str = f"{flow_used}/{flow_cap} ({flow_pct:.0f}%)"
        load_str = f"{load_used}/{load_cap} ({load_pct:.0f}%)"
        store_str = f"{store_used}/{store_cap} ({store_pct:.0f}%)"

        row = (
            f"  {d}     | {cycle_range:15} | {alu_str:18} | {valu_break:28} | {flow_str:6} | {load_str:5} | {store_str:5} | {scratch_str}"
        )
        print(row)

    print()
    print("VALU breakdown: vbroadcast and FMA (multiply_add / sub) are not shiftable; other is shiftable.")
    if not liveness_curve:
        print("Scratch: run with liveness_detail=full (default when --output-json is set) to include per-cycle scratch.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
