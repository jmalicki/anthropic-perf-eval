#!/usr/bin/env python3
"""
Report hypothesis quantification metrics from a scheduler output JSON.

Reads one JSON (from generate_ssa_kernel.py --output-json) and reports:
1. Vbroadcast eliminability: vbroadcasts whose consumers are all shifted to ALU
   (could be replaced by scalar loads/add_imm; count and VALU slot saving).
2. Element-wise slack: for shifted ops with dependents, wasted slack (max−min ALU
   cycles) and a theoretical cycle-saving upper bound if dependents were scheduled
   per-lane.
3. Decile 0 ConstChoice: add_imm vs Load const counts in the first decile.

Theoretical cycle saving formula: for each consumer of a shifted op, we use
  completion_cycle = max(dep_alu_cycles). Current model schedules consumer at
  earliest completion_cycle+1. If we could schedule per-lane, consumer could start
  at min(dep_alu_cycles)+1. We sum over consumers: max(0, consumer_cycle - (min(dep_alu_cycles)+1)).
  Consumer cycle is from ops (or min(consumer's alu_cycles) if consumer is also shifted).

Usage (from project root):
  uv run scripts/quantify_schedule_hypotheses.py [scheduler_output.json]
  uv run scripts/quantify_schedule_hypotheses.py --markdown scheduler_output.json

Requires scheduler output with `ops`. For vbroadcast and slack, `shifted_alu_cycles`
must be present (run with --enable-valu-shift and output-json).
"""

import argparse
import json
import sys

VLEN = 8


def _shifted_map(data):
    """Return shifted_alu_cycles as dict[int, list[int]]. JSON keys may be strings."""
    raw = data.get("shifted_alu_cycles") or {}
    return {int(k): v for k, v in raw.items() if isinstance(v, list) and len(v) >= VLEN}


def _consumer_map(ops):
    """Map producer op id -> list of consumer op dicts (ops that have producer in dep_ids)."""
    consumers = {}
    for op in ops:
        op_id = op.get("id")
        if op_id is None:
            continue
        for dep in op.get("dep_ids") or []:
            consumers.setdefault(dep, []).append(op)
    return consumers


def vbroadcast_eliminability(data, shifted, consumers_by_id):
    """Return (eliminable_list, total_slots_saved_by_cycle)."""
    ops = data.get("ops", [])
    op_by_id = {op["id"]: op for op in ops if "id" in op}
    eliminable = []
    slots_by_cycle = {}

    for op in ops:
        if op.get("engine", "").lower() != "valu":
            continue
        if (op.get("opcode") or "").strip() != "vbroadcast":
            continue
        op_id = op["id"]
        consumers = consumers_by_id.get(op_id, [])
        if not consumers:
            eliminable.append((op_id, op.get("cycle", 0)))
            c = op.get("cycle", 0)
            slots_by_cycle[c] = slots_by_cycle.get(c, 0) + 1
            continue
        if not shifted:
            continue
        if all(c["id"] in shifted for c in consumers):
            eliminable.append((op_id, op.get("cycle", 0)))
            c = op.get("cycle", 0)
            slots_by_cycle[c] = slots_by_cycle.get(c, 0) + 1

    return eliminable, slots_by_cycle


def element_wise_slack(data, shifted):
    """Return (slack_list per (producer_id, consumer_id), theoretical_cycle_saving)."""
    ops = data.get("ops", [])
    op_by_id = {op["id"]: op for op in ops if "id" in op}
    consumers_by_id = _consumer_map(ops)
    slack_pairs = []
    theoretical_saving = 0

    for op_id, alu_cycles in shifted.items():
        if len(alu_cycles) < VLEN:
            continue
        completion = max(alu_cycles)
        min_cycle = min(alu_cycles)
        wasted_slack = completion - min_cycle
        consumers = consumers_by_id.get(op_id, [])
        for c in consumers:
            c_id = c["id"]
            slack_pairs.append((op_id, c_id, wasted_slack))
            # Consumer's scheduled cycle: use op cycle, or min(consumer's alu_cycles) if shifted
            c_cycle = c.get("cycle")
            if c_id in shifted and len(shifted[c_id]) >= VLEN:
                c_cycle = min(shifted[c_id])
            if c_cycle is None:
                continue
            # Earliest start if per-lane: min_cycle + 1
            earliest_per_lane = min_cycle + 1
            if c_cycle > earliest_per_lane:
                theoretical_saving += c_cycle - earliest_per_lane

    return slack_pairs, theoretical_saving


def early_slot_slack(data, total_cycles, num_deciles=2):
    """Return list of (decile, cycle_lo, cycle_hi, alu_slack, valu_slack, cycles_with_alu_slack)."""
    ops = data.get("ops", [])
    n = total_cycles
    alu_cap, valu_cap = 12, 6
    alu = [0] * n
    valu = [0] * n
    for op in ops:
        c = op.get("cycle", 0)
        if 0 <= c < n:
            e = (op.get("engine") or "").lower()
            if e == "alu":
                alu[c] += 1
            elif e == "valu":
                valu[c] += 1
    result = []
    for d in range(min(num_deciles, 10)):
        c_lo = (n * d) // 10
        c_hi = (n * (d + 1)) // 10
        if c_lo >= c_hi:
            continue
        alu_slack = sum(alu_cap - alu[c] for c in range(c_lo, c_hi) if alu[c] < alu_cap)
        valu_slack = sum(valu_cap - valu[c] for c in range(c_lo, c_hi) if valu[c] < valu_cap)
        cycles_with_alu = [c for c in range(c_lo, c_hi) if alu[c] < alu_cap]
        result.append((d, c_lo, c_hi - 1, alu_slack, valu_slack, cycles_with_alu))
    return result


def decile0_constchoice(data):
    """Return (add_imm_count, load_const_count) for first decile."""
    total_cycles = data.get("total_cycles")
    ops = data.get("ops", [])
    if not total_cycles and ops:
        total_cycles = max(op.get("cycle", 0) for op in ops) + 1
    if not total_cycles:
        return 0, 0
    decile0_end = (total_cycles * 10) // 100
    if decile0_end <= 0:
        decile0_end = 1
    add_imm = 0
    load_const = 0
    for op in ops:
        c = op.get("cycle", 0)
        if c < 0 or c >= decile0_end:
            continue
        eng = (op.get("engine") or "").lower()
        opcode = (op.get("opcode") or "").strip()
        if eng == "flow" and opcode == "add_imm":
            add_imm += 1
        elif eng == "load" and opcode == "const":
            load_const += 1
    return add_imm, load_const


def main():
    parser = argparse.ArgumentParser(
        description="Report hypothesis quantification metrics from scheduler output JSON"
    )
    parser.add_argument(
        "input_json",
        nargs="?",
        default="scheduler_output.json",
        help="Scheduler output JSON path (default: scheduler_output.json)",
    )
    parser.add_argument(
        "--markdown",
        "-m",
        action="store_true",
        help="Output as markdown (renders in docs/notebooks)",
    )
    args = parser.parse_args()

    try:
        with open(args.input_json) as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: file not found: {args.input_json}", file=sys.stderr)
        print(
            "Generate with: uv run generate_ssa_kernel.py --output-json scheduler_output.json",
            file=sys.stderr,
        )
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON in {args.input_json}: {e}", file=sys.stderr)
        return 1

    ops = data.get("ops", [])
    if not ops:
        print("Error: no ops in JSON", file=sys.stderr)
        return 1

    total_cycles = data.get("total_cycles") or (max(op.get("cycle", 0) for op in ops) + 1)
    shifted = _shifted_map(data)
    consumers_by_id = _consumer_map(ops)

    md = args.markdown

    # --- Vbroadcast eliminability ---
    if md:
        print("## Vbroadcast eliminability\n")
    else:
        print("=== Vbroadcast eliminability ===\n")
    if not shifted:
        print("N/A (run with --enable-valu-shift and output-json to get shifted_alu_cycles).\n")
    else:
        eliminable, slots_by_cycle = vbroadcast_eliminability(
            data, shifted, consumers_by_id
        )
        print(f"Eliminable vbroadcasts (all consumers shifted): {len(eliminable)}")
        if eliminable:
            total_slots = sum(slots_by_cycle.values())
            print(f"VALU slots that could be saved (if removed): {total_slots} total")
            if len(eliminable) <= 20:
                if md:
                    print("\n| op id | cycle |")
                    print("|-------|-------|")
                    for op_id, cy in sorted(eliminable, key=lambda x: (x[1], x[0])):
                        print(f"| {op_id} | {cy} |")
                else:
                    print("(id, cycle):", sorted(eliminable, key=lambda x: (x[1], x[0])))
            else:
                if md:
                    print("\n| op id | cycle |")
                    print("|-------|-------|")
                    for op_id, cy in sorted(eliminable, key=lambda x: (x[1], x[0]))[:20]:
                        print(f"| {op_id} | {cy} |")
                    print(f"... and {len(eliminable) - 20} more")
                else:
                    print(
                        "First 20 (id, cycle):",
                        sorted(eliminable, key=lambda x: (x[1], x[0]))[:20],
                    )
        print()

    # --- Element-wise slack ---
    if md:
        print("## Element-wise slack (shifted ops)\n")
    else:
        print("=== Element-wise slack (shifted ops) ===\n")
    if not shifted:
        print("N/A (run with --enable-valu-shift and output-json).\n")
    else:
        slack_pairs, theoretical_saving = element_wise_slack(data, shifted)
        shifted_with_consumers = len({p for p, _c, _s in slack_pairs})
        print(f"Shifted ops that have at least one consumer: {shifted_with_consumers}")
        print(f"Producer–consumer pairs (shifted producer): {len(slack_pairs)}")
        if slack_pairs:
            slacks = [s for _p, _c, s in slack_pairs]
            print(
                f"Wasted slack (max−min ALU cycles) — min: {min(slacks)}, max: {max(slacks)}, mean: {sum(slacks)/len(slacks):.2f}"
            )
            print(
                f"Theoretical cycle saving (upper bound if dependents scheduled per-lane): {theoretical_saving}"
            )
        print()

    # --- Early-slot slack (fillable capacity) ---
    if md:
        print("## Early-slot slack (fillable capacity)\n")
    else:
        print("=== Early-slot slack (fillable capacity) ===\n")
    for d, c_lo, c_hi, alu_slack, valu_slack, cycles_with_alu in early_slot_slack(
        data, total_cycles, num_deciles=2
    ):
        n_cy = c_hi - c_lo + 1
        print(
            f"Decile {d} (cycles {c_lo}–{c_hi}): ALU slack {alu_slack} slot-cycles, VALU slack {valu_slack} slot-cycles"
        )
        if cycles_with_alu and len(cycles_with_alu) <= 25:
            print(f"  Cycles with ALU slack: {cycles_with_alu}")
        elif cycles_with_alu:
            print(f"  Cycles with ALU slack: {cycles_with_alu[:15]} ... and {len(cycles_with_alu) - 15} more")
    print(
        "  (Filling earlier slots would need precedence-feasible ops, e.g. more VALU→ALU shift or moving ALU ops whose deps are already in place.)"
    )
    print()

    # --- Decile 0 ConstChoice ---
    if md:
        print("## Decile 0 ConstChoice\n")
    else:
        print("=== Decile 0 ConstChoice ===\n")
    add_imm, load_const = decile0_constchoice(data)
    decile0_end = (total_cycles * 10) // 100 if total_cycles else 0
    print(
        f"Decile 0 (cycles 0–{decile0_end - 1 if decile0_end else 0}): add_imm {add_imm}, Load const {load_const}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
