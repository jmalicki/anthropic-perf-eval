#!/usr/bin/env python3
"""
Analyze scheduler output: could ALU-derived constants (Add/Sub/Mul/Shift on two
existing consts) let dependent ops move to earlier cycles?

Part 1: Identify opportunities (ConstChoice ops that could use ALU-derived).
Part 2: Simulate moving ops into freed slots and estimate cycle savings.

Correctness: Uses lowered ops for slot accounting. Phase 1 moves ConstChoice
only when the actual engine (Load or Flow) has capacity to remove and ALU has
capacity at the new cycle. Phase 2 only moves consumers that map 1:1 to a
single lowered op (excludes expanded vector ops). Precedence: earliest =
max(dep_cycles) + 1 (write-after-read). Never decrements usage below zero.

Usage:
  uv run scripts/analyze_alu_derive_opportunity.py [scheduler_output.json]
  uv run scripts/analyze_alu_derive_opportunity.py --simulate scheduler_output.json  # default
"""

import argparse
import json
import sys

SLOT_LIMITS = {"alu": 12, "valu": 6, "load": 2, "store": 2, "flow": 1}


def _alu_derive_encodings(value: int, available: set[int], max_count: int = 5) -> list[tuple[str, int, int]]:
    """Return [(encoding_type, dep_a, dep_b), ...] for ALU-derived encodings."""
    result = []
    if max_count <= 0:
        return result

    if value > 0:
        max_operand = min(value, 512)
        for a in range(max_operand + 1):
            b = value - a
            if b >= a >= 0 and b <= 512 and a in available and b in available:
                result.append(("add", a, b))
                if len(result) >= max_count:
                    return result

    if value > 0:
        for shift in range(1, 10):
            divisor = 1 << shift
            if value % divisor == 0:
                base = value // divisor
                if 0 < base <= 256 and base in available and shift in available:
                    result.append(("shift", base, shift))
                    if len(result) >= max_count:
                        return result

    if value > 0:
        for a in range(2, 17):
            if value % a == 0:
                b = value // a
                if b >= a and b <= 256 and a in available and b in available:
                    result.append(("mul", a, b))
                    if len(result) >= max_count:
                        return result

    if value < 0 and 0 in available and -value in available:
        result.append(("sub", 0, -value))
    return result


def _engine_for_op(op: dict) -> str:
    """Return engine string for slot accounting."""
    eng = (op.get("engine") or "").lower()
    if eng in ("alu", "valu", "load", "store", "flow"):
        return eng
    return "alu"  # fallback


def _build_slot_usage_from_ops(ops: list[dict], n_cycles: int) -> dict[str, list[int]]:
    """Build per-cycle slot usage from lowered ops (ground truth)."""
    usage = {e: [0] * n_cycles for e in SLOT_LIMITS}
    for op in ops:
        c = op.get("cycle", 0)
        if c < 0 or c >= n_cycles:
            continue
        eng = _engine_for_op(op)
        usage[eng][c] += 1
    return usage


def _slot_free(usage: dict[str, list[int]], engine: str, cycle: int) -> bool:
    limit = SLOT_LIMITS[engine]
    if cycle < 0 or cycle >= len(usage[engine]):
        return False
    return usage[engine][cycle] < limit


def _simulate(
    ssa_ops: list[dict],
    ops: list[dict],
    opportunities: list[dict],
    value_to_earliest: dict[int, tuple[int, int]],
    consumers: dict[int, list[dict]],
    original_makespan: int,
) -> tuple[int, int, list[tuple[str, int, int, int]]]:
    """
    Simulate moving ConstChoice ops to ALU-derived. Only makes correct moves:
    - Uses ops (lowered) for slot accounting (ground truth)
    - Phase 1: Match ConstChoice by id; decrement actual engine (Load or Flow)
    - Phase 2: Only move consumers that map 1:1 to a single lowered op
    """
    op_cycles = {o["id"]: o.get("cycle", 0) for o in ssa_ops}
    n_cycles = original_makespan + 50
    moves_log: list[tuple[str, int, int, int]] = []

    # Build initial slot usage from lowered ops (ground truth)
    usage = _build_slot_usage_from_ops(ops, n_cycles)

    # Map ssa_op_id -> lowered op. ConstChoice keeps ssa_op.id in lowered op.
    # Prefer from_const_choice if present, else match by id (ConstChoice lowers to 1 op with same id)
    op_id_to_lowered_map: dict[int, dict] = {}
    for op in ops:
        oid = op.get("id")
        if oid is not None:
            op_id_to_lowered_map[oid] = op

    moved_const: set[int] = set()
    const_choice_ids = {o["op_id"] for o in opportunities}

    # Phase 1: Move ConstChoice ops to ALU-derived at earliest cycle
    sorted_opps = sorted(opportunities, key=lambda o: o["earliest_alu_cycle"])
    for o in sorted_opps:
        op_id = o["op_id"]
        old_cycle = op_cycles[op_id]
        new_cycle = o["earliest_alu_cycle"]
        if new_cycle >= old_cycle:
            continue
        if not o.get("has_slot", True):
            continue
        if op_id in moved_const:
            continue
        lowered = op_id_to_lowered_map.get(op_id)
        if not lowered or op_id not in const_choice_ids:
            continue
        old_eng = _engine_for_op(lowered)
        if not _slot_free(usage, "alu", new_cycle):
            continue
        if usage[old_eng][old_cycle] <= 0:
            continue
        usage[old_eng][old_cycle] -= 1
        usage["alu"][new_cycle] += 1
        op_cycles[op_id] = new_cycle
        moved_const.add(op_id)
        moves_log.append(("const_choice", op_id, old_cycle, new_cycle))

    # Phase 2: Cascade consumer moves - only when 1:1 with lowered op (excludes expanded vector ops)
    op_id_to_lowered: dict[int, list[dict]] = {}
    for op in ops:
        oid = op.get("id")
        if oid is not None:
            op_id_to_lowered.setdefault(oid, []).append(op)

    changed = True
    while changed:
        changed = False
        for op in ssa_ops:
            op_id = op["id"]
            deps = op.get("dep_ids") or []
            if not deps:
                continue
            earliest = max(op_cycles.get(d, 0) for d in deps) + 1
            current = op_cycles[op_id]
            if earliest >= current:
                continue
            lowered_list = op_id_to_lowered.get(op_id, [])
            if len(lowered_list) != 1:
                continue
            lowered = lowered_list[0]
            eng = _engine_for_op(lowered)
            if not _slot_free(usage, eng, earliest):
                continue
            if usage[eng][current] <= 0:
                continue
            usage[eng][current] -= 1
            usage[eng][earliest] += 1
            op_cycles[op_id] = earliest
            moves_log.append(("consumer", op_id, current, earliest))
            changed = True

    new_makespan = max(op_cycles.values()) if op_cycles else 0
    cycles_saved = original_makespan - new_makespan
    return new_makespan, cycles_saved, moves_log


def main():
    parser = argparse.ArgumentParser(description="Analyze ALU-derived constant opportunities")
    parser.add_argument("input_json", nargs="?", default="scheduler_output.json")
    parser.add_argument(
        "--no-simulate",
        action="store_true",
        help="Skip simulation (only report opportunities)",
    )
    args = parser.parse_args()

    with open(args.input_json) as f:
        data = json.load(f)

    ssa_ops = data.get("ssa_ops", [])
    ops = data.get("ops", [])
    if not ssa_ops or not ops:
        print("Error: need both ssa_ops and ops", file=sys.stderr)
        return 1

    total_cycles = data.get("total_cycles") or (max(o["cycle"] for o in ops) + 1)

    # Build value -> earliest (op_id, cycle) for const_choice ops
    value_to_earliest: dict[int, tuple[int, int]] = {}
    const_choice_ops: list[dict] = []

    for op in ssa_ops:
        if (op.get("opcode") or "").strip() != "const_choice":
            continue
        srcs = op.get("srcs") or []
        value = None
        for s in srcs:
            if "Imm" in s:
                value = s["Imm"]
                break
        if value is None:
            continue
        const_choice_ops.append(op)
        cycle = op.get("cycle", 0)
        op_id = op.get("id")
        if value not in value_to_earliest or cycle < value_to_earliest[value][1]:
            value_to_earliest[value] = (op_id, cycle)

    consumers: dict[int, list[dict]] = {}
    for op in ssa_ops:
        for dep in op.get("dep_ids") or []:
            consumers.setdefault(dep, []).append(op)

    alu_per_cycle = [0] * total_cycles
    for op in ops:
        c = op.get("cycle", 0)
        if 0 <= c < total_cycles and (op.get("engine") or "").lower() == "alu":
            alu_per_cycle[c] += 1

    ssa_cycle = {o["id"]: o.get("cycle", 0) for o in ssa_ops}

    opportunities = []
    total_const_choice = 0
    has_alu_option = 0
    could_move_earlier = 0
    consumer_could_move = 0

    for op in const_choice_ops:
        op_id = op["id"]
        cycle = op.get("cycle", 0)
        srcs = op.get("srcs") or []
        value = None
        for s in srcs:
            if "Imm" in s:
                value = s["Imm"]
                break
        if value is None:
            continue

        total_const_choice += 1
        available_before = set()
        for v, (_, prod_cycle) in value_to_earliest.items():
            if prod_cycle < cycle:
                available_before.add(v)
        available_before.add(0)

        alu_encodings = _alu_derive_encodings(value, available_before, max_count=3)
        if not alu_encodings:
            continue
        has_alu_option += 1

        best_earliest = cycle
        best_enc = None
        for enc_type, dep_a, dep_b in alu_encodings:
            cycle_a = value_to_earliest[dep_a][1] if dep_a in value_to_earliest else (-1 if dep_a == 0 else cycle)
            cycle_b = value_to_earliest[dep_b][1] if dep_b in value_to_earliest else (-1 if dep_b == 0 else cycle)
            if dep_a == 0:
                cycle_a = -1
            if dep_b == 0:
                cycle_b = -1
            earliest = max(cycle_a, cycle_b, 0) + 1
            if earliest < best_earliest:
                best_earliest = earliest
                best_enc = (enc_type, dep_a, dep_b)

        if best_enc is None or best_earliest >= cycle:
            continue

        could_move_earlier += 1
        enc_type, dep_a, dep_b = best_enc
        free_alu = SLOT_LIMITS["alu"] - alu_per_cycle[best_earliest] if best_earliest < total_cycles else 0
        has_slot = free_alu >= 1

        consumer_moves = []
        for consumer in consumers.get(op_id, []):
            c_id = consumer["id"]
            c_cycle = consumer.get("cycle", 0)
            deps = consumer.get("dep_ids") or []
            other_deps = [d for d in deps if d != op_id]
            other_max = max(ssa_cycle.get(d, 0) for d in other_deps) if other_deps else -1
            new_consumer_earliest = max(best_earliest, other_max) + 1
            if new_consumer_earliest < c_cycle:
                free_at_new = SLOT_LIMITS["alu"] - alu_per_cycle[new_consumer_earliest] if new_consumer_earliest < total_cycles else 0
                eng = (consumer.get("engine") or "").lower()
                slot_lim = SLOT_LIMITS.get(eng, SLOT_LIMITS["alu"])
                free_at_new = slot_lim - (
                    alu_per_cycle[new_consumer_earliest] if eng == "alu" else 0
                )  # simplified
                consumer_moves.append((c_id, c_cycle, new_consumer_earliest, free_at_new >= 1, eng))
                if free_at_new >= 1:
                    consumer_could_move += 1

        opportunities.append({
            "op_id": op_id,
            "value": value,
            "current_cycle": cycle,
            "earliest_alu_cycle": best_earliest,
            "encoding": f"{enc_type}({dep_a}, {dep_b})",
            "free_alu_at_earliest": free_alu,
            "has_slot": has_slot,
            "consumer_moves": consumer_moves,
        })

    print("=== ALU-derived ConstChoice opportunity analysis ===\n")
    print(f"Total ConstChoice ops: {total_const_choice}")
    print(f"With ALU-derived option (deps available earlier): {has_alu_option}")
    print(f"Could move to earlier cycle: {could_move_earlier}")
    print(f"Consumers that could move earlier (with free slot): {consumer_could_move}\n")

    if opportunities:
        print("Opportunities (ConstChoice ops that could use ALU-derived and move earlier):\n")
        for o in opportunities[:20]:
            print(f"  Op {o['op_id']} value={o['value']}: cycle {o['current_cycle']} -> {o['earliest_alu_cycle']} "
                  f"via {o['encoding']} | ALU free at earliest: {o['free_alu_at_earliest']}/12 "
                  f"({'yes' if o['has_slot'] else 'no slot'})")
            for cm in o["consumer_moves"][:3]:
                print(f"    -> consumer op {cm[0]} could move {cm[1]} -> {cm[2]} "
                      f"({'slot ok' if cm[3] else 'no slot'}) engine={cm[4]}")
        if len(opportunities) > 20:
            print(f"  ... and {len(opportunities) - 20} more")
    else:
        print("No opportunities found.")

    if not args.no_simulate and opportunities:
        print("\n=== Simulation: moving deps into freed slots ===\n")
        new_makespan, cycles_saved, moves_log = _simulate(
            ssa_ops, ops, opportunities, value_to_earliest, consumers, total_cycles
        )
        const_moves = [m for m in moves_log if m[0] == "const_choice"]
        consumer_moves = [m for m in moves_log if m[0] == "consumer"]
        print(f"Original makespan: {total_cycles}")
        print(f"Simulated makespan: {new_makespan}")
        print(f"Cycles saved (estimated): {cycles_saved}")
        print("(Estimate uses dependency chains and per-engine slot limits; critical path may be VALU/Store/Flow.)")
        print(f"ConstChoice ops moved: {len(const_moves)}")
        print(f"Consumer ops moved (cascaded): {len(consumer_moves)}")
        if moves_log:
            print("\nFirst 20 moves:")
            for typ, oid, old_c, new_c in moves_log[:20]:
                print(f"  {typ} op {oid}: cycle {old_c} -> {new_c}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
