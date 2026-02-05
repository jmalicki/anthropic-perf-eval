#!/usr/bin/env python3
"""
Explore whether high utilization (e.g. 95%) is remotely feasible for a schedule.

Reads scheduler output JSON and computes:
- Total ops per engine (ALU, VALU, Load, Store, Flow)
- Capacity-bound minimum cycles: if an engine were 100% utilized, how many cycles
  would it need? (e.g. total_alu / 12 for ALU)
- Current utilization (over all cycles and excluding cycle 0)
- "At 95% utilization" lower bound: min cycles needed if that engine ran at 95%
  (total_ops / (0.95 * slots_per_cycle)). The max over engines is a theoretical
  lower bound on makespan at 95% util (ignoring precedence and that all engines
  must align).

Usage:
  uv run scripts/utilization_feasibility.py [scheduler_output.json]
"""

import argparse
import json
import math
import sys

SLOT_LIMITS = {"alu": 12, "valu": 6, "load": 2, "store": 2, "flow": 1}


def main():
    parser = argparse.ArgumentParser(description="Utilization feasibility from schedule JSON")
    parser.add_argument("input_json", nargs="?", default="scheduler_output.json")
    args = parser.parse_args()
    try:
        with open(args.input_json) as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: not found {args.input_json}", file=sys.stderr)
        return 1
    total_cycles = data.get("total_cycles") or (max(op["cycle"] for op in data.get("ops", [])) + 1)
    ops = data.get("ops", [])
    if not ops:
        print("Error: no ops in JSON", file=sys.stderr)
        return 1

    totals = {k: 0 for k in SLOT_LIMITS}
    cycle0 = {k: 0 for k in SLOT_LIMITS}
    for op in ops:
        c = op.get("cycle", 0)
        eng = op.get("engine", "").lower()
        if eng not in SLOT_LIMITS:
            continue
        totals[eng] += 1
        if c == 0:
            cycle0[eng] += 1

    n = total_cycles
    slots = SLOT_LIMITS
    print(f"Input: {args.input_json}")
    print(f"Total cycles: {n}")
    print()
    print("Total ops per engine:")
    for eng, cap in slots.items():
        print(f"  {eng}: {totals[eng]} (capacity {cap}/cycle)")
    print()
    print("Cycle 0 usage (often light due to deps/constants):")
    for eng in slots:
        print(f"  {eng}: {cycle0[eng]}/{slots[eng]}")
    print()

    # Capacity-bound minimum cycles at 100% util
    min_cycles_100 = {}
    for eng, cap in slots.items():
        if totals[eng] == 0:
            min_cycles_100[eng] = 0
        else:
            min_cycles_100[eng] = (totals[eng] + cap - 1) // cap
    print("Capacity-bound minimum cycles (if 100% utilized):")
    for eng in slots:
        print(f"  {eng}: {min_cycles_100[eng]}")
    bottleneck_100 = max(min_cycles_100.values()) if any(min_cycles_100.values()) else 0
    print(f"  -> Bottleneck (max): {bottleneck_100} cycles")
    print()

    # Current utilization
    print("Current utilization (all cycles):")
    for eng, cap in slots.items():
        used = totals[eng]
        avail = cap * n
        pct = (100.0 * used / avail) if avail else 0
        print(f"  {eng}: {used}/{avail} = {pct:.1f}%")
    print("Current utilization (excluding cycle 0):")
    n_rest = max(0, n - 1)
    for eng, cap in slots.items():
        used_rest = totals[eng] - cycle0[eng]
        avail_rest = cap * n_rest
        pct = (100.0 * used_rest / avail_rest) if avail_rest else 0
        print(f"  {eng}: {used_rest}/{avail_rest} = {pct:.1f}%")
    print()

    # At 95% utilization, minimum cycles per engine
    target_util = 0.95
    min_cycles_95 = {}
    for eng, cap in slots.items():
        if totals[eng] == 0:
            min_cycles_95[eng] = 0
        else:
            # total = 0.95 * cap * C  =>  C = total / (0.95 * cap)
            min_cycles_95[eng] = math.ceil(totals[eng] / (target_util * cap))
    print(f"Theoretical minimum makespan at {target_util*100:.0f}% utilization (capacity-bound only):")
    for eng in slots:
        print(f"  {eng}: {min_cycles_95[eng]} cycles")
    bottleneck_95 = max(min_cycles_95.values()) if any(min_cycles_95.values()) else 0
    print(f"  -> Bottleneck: {bottleneck_95} cycles")
    print()
    if bottleneck_95 <= n:
        print(f"  So {target_util*100:.0f}% utilization is capacity-wise feasible: need at least {bottleneck_95} cycles, we have {n}.")
        print("  (Precedence and other constraints will raise the actual minimum; this is an upper bound on feasibility.)")
    else:
        print(f"  So {target_util*100:.0f}% utilization is not achievable: would need at least {bottleneck_95} cycles, we have {n}.")
    print()
    # What utilization is "max" we could have at current makespan?
    print("If we kept makespan at current level, max achievable utilization (capacity-bound):")
    for eng, cap in slots.items():
        avail = cap * n
        if avail:
            pct = 100.0 * totals[eng] / avail
            print(f"  {eng}: {pct:.1f}% (we have {totals[eng]} ops, capacity {avail})")
    print()
    # Excluding cycle 0: is 95% on ALU/VALU feasible?
    n_no0 = max(0, n - 1)
    if n_no0:
        print("Scenario: first cycle unused for ALU/VALU (as is), 95% util on remaining cycles:")
        for eng in ("alu", "valu"):
            cap = slots[eng]
            total_eng = totals[eng] - cycle0[eng]
            need_cycles = math.ceil(total_eng / (target_util * cap)) if total_eng else 0
            ok = need_cycles <= n_no0
            print(f"  {eng}: need {need_cycles} packable cycles at 95%, have {n_no0} -> {'feasible' if ok else 'not feasible'} (capacity-bound)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
