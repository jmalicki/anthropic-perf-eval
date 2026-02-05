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


def vbroadcast_elimination_asap_impact(data, shifted, consumers_by_id):
    """
    Find chains vbroadcast → valu1 (shifted only) → valu2. If we eliminate the vbroadcast
    (feed valu1 via 8 scalar constants at cycle 0), valu1 can complete at cycle 1,
    so valu2's ASAP drops from 1+completion(valu1) to 2.
    Returns (list of (vb_id, vb_cycle, valu1_id, completion_valu1, valu2_id, asap_current, asap_if_elim, reduction), set of vbroadcast ids that are profitably eliminable).
    """
    ops = data.get("ops", [])
    op_by_id = {op["id"]: op for op in ops if "id" in op}
    EARLIEST_SCALAR = 0
    VALU1_COMPLETION_IF_ELIM = EARLIEST_SCALAR + 1
    VALU2_ASAP_IF_ELIM = VALU1_COMPLETION_IF_ELIM + 1

    chains = []
    vb_profitable = set()

    for op in ops:
        if op.get("engine", "").lower() != "valu" or (op.get("opcode") or "").strip() != "vbroadcast":
            continue
        vb_id = op["id"]
        vb_cycle = op.get("cycle", 0)
        valu1_list = consumers_by_id.get(vb_id, [])
        for valu1 in valu1_list:
            v1_id = valu1["id"]
            if v1_id not in shifted or len(shifted[v1_id]) < VLEN:
                continue
            completion_v1 = max(shifted[v1_id])
            valu2_list = consumers_by_id.get(v1_id, [])
            for valu2 in valu2_list:
                v2_id = valu2["id"]
                asap_current = 1 + completion_v1
                asap_if_elim = VALU2_ASAP_IF_ELIM
                reduction = asap_current - asap_if_elim
                if reduction <= 0:
                    continue
                chains.append((vb_id, vb_cycle, v1_id, completion_v1, v2_id, asap_current, asap_if_elim, reduction))
                vb_profitable.add(vb_id)

    return chains, vb_profitable


def vbroadcast_shift_elimination_chains(data, shifted, consumers_by_id):
    """
    Find all chains vbroadcast → valu1 → valu2 (valu1 any VALU op, not necessarily shifted).
    Combined transform: shift valu1 to ALU + eliminate vbroadcast (feed valu1 via 8 scalars at cycle 0)
    → valu1 completes at 1, so valu2 ASAP = 2.
    Returns (chains with (vb_id, vb_cycle, valu1_id, valu1_cycle, completion_v1, valu1_already_shifted, valu2_id, asap_current, asap_if_combined, reduction), set of vb ids with positive reduction).
    """
    ops = data.get("ops", [])
    op_by_id = {op["id"]: op for op in ops if "id" in op}
    EARLIEST_SCALAR = 0
    VALU1_COMPLETION_IF_ELIM = EARLIEST_SCALAR + 1
    VALU2_ASAP_IF_ELIM = VALU1_COMPLETION_IF_ELIM + 1

    chains = []
    vb_profitable = set()

    for op in ops:
        if op.get("engine", "").lower() != "valu" or (op.get("opcode") or "").strip() != "vbroadcast":
            continue
        vb_id = op["id"]
        vb_cycle = op.get("cycle", 0)
        valu1_list = consumers_by_id.get(vb_id, [])
        for valu1 in valu1_list:
            v1_id = valu1["id"]
            # valu1 must be VALU (consumer of vbroadcast is typically VALU)
            if (valu1.get("engine") or "").lower() != "valu":
                continue
            if v1_id in shifted and len(shifted[v1_id]) >= VLEN:
                completion_v1 = max(shifted[v1_id])
                v1_already_shifted = True
            else:
                completion_v1 = valu1.get("cycle", 0)
                v1_already_shifted = False
            v1_cycle = valu1.get("cycle", 0)
            valu2_list = consumers_by_id.get(v1_id, [])
            for valu2 in valu2_list:
                v2_id = valu2["id"]
                asap_current = 1 + completion_v1
                asap_if_elim = VALU2_ASAP_IF_ELIM
                reduction = asap_current - asap_if_elim
                if reduction <= 0:
                    continue
                chains.append((
                    vb_id, vb_cycle, v1_id, v1_cycle, completion_v1, v1_already_shifted,
                    v2_id, asap_current, asap_if_elim, reduction,
                ))
                vb_profitable.add(vb_id)

    return chains, vb_profitable


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


def asap_bounds(data, shifted):
    """
    Compute precedence-based ASAP for each op.
    - ASAP (current): 1 + max over deps of completion_cycle. Shifted dep completion = max(alu_cycles).
    - ASAP_per_lane: 1 + max over deps of (min(alu_cycles)+0 for shifted, else dep_cycle); i.e. consumer
      could start at 1 + max(min(alu_cycles) for shifted deps, dep_cycle for others). So completion
      for per-lane = min(alu_cycles) for shifted (earliest lane completes), else dep_cycle.
    Returns list of (op, asap, asap_per_lane, limiting_dep_id, completion_max, completion_min).
    """
    ops = data.get("ops", [])
    op_by_id = {op["id"]: op for op in ops if "id" in op}
    results = []
    for op in ops:
        op_id = op["id"]
        dep_ids = op.get("dep_ids") or []
        if not dep_ids:
            asap = 0
            asap_per_lane = 0
            limiting = None
            comp_max_lim = comp_min_lim = None
        else:
            completions_max = []
            completions_min = []
            for d in dep_ids:
                if d in shifted and len(shifted[d]) >= VLEN:
                    completions_max.append(max(shifted[d]))
                    completions_min.append(min(shifted[d]))
                elif d in op_by_id:
                    c = op_by_id[d].get("cycle", 0)
                    completions_max.append(c)
                    completions_min.append(c)
                else:
                    completions_max.append(-1)
                    completions_min.append(-1)
            comp_max = max(completions_max) if completions_max else -1
            comp_min = max(completions_min) if completions_min else -1
            asap = 1 + comp_max
            asap_per_lane = 1 + comp_min
            # Which dep limited (the one with max completion for current ASAP)
            limiting = dep_ids[completions_max.index(comp_max)] if dep_ids else None
            comp_max_lim = comp_min_lim = None
            if limiting and limiting in shifted and len(shifted.get(limiting, [])) >= VLEN:
                comp_max_lim = max(shifted[limiting])
                comp_min_lim = min(shifted[limiting])
            elif limiting and limiting in op_by_id:
                comp_max_lim = op_by_id[limiting].get("cycle")
                comp_min_lim = comp_max_lim
        results.append((op, asap, asap_per_lane, limiting, comp_max_lim, comp_min_lim))
    return results


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

    # --- Vbroadcast elimination + VALU shift (combined) ---
    if md:
        print("## Vbroadcast elimination + VALU shift (combined)\n")
    else:
        print("=== Vbroadcast elimination + VALU shift (combined) ===\n")
    vb_count = len([o for o in data.get("ops", []) if (o.get("engine") or "").lower() == "valu" and (o.get("opcode") or "").strip() == "vbroadcast"])
    combined_chains, vb_profitable_combined = vbroadcast_shift_elimination_chains(data, shifted if shifted else {}, consumers_by_id)
    print(
        "Chains vb → valu1 → valu2: if we **combine** (1) shift valu1 to ALU and (2) eliminate vbroadcast (feed valu1 via 8 scalars at cycle 0), then valu1 completes at 1 and valu2's ASAP drops to 2."
    )
    print(f"Vbroadcasts that can be profitably eliminated with this combined transform: {len(vb_profitable_combined)} (of {vb_count} total)")
    print(f"Chains with positive ASAP reduction: {len(combined_chains)}")
    if combined_chains:
        already_shifted = sum(1 for c in combined_chains if c[5])
        print(f"Of those chains: valu1 already shifted in {already_shifted}, would need to shift in {len(combined_chains) - already_shifted}")
        chains = combined_chains
        chains.sort(key=lambda c: (-c[9], c[1]))
        vb_id, vb_cycle, v1_id, v1_cycle, completion_v1, v1_already_shifted, v2_id, asap_cur, asap_elim, reduction = chains[0]
        op_by_id = {op["id"]: op for op in data.get("ops", []) if "id" in op}
        v1 = op_by_id.get(v1_id, {})
        v2 = op_by_id.get(v2_id, {})
        print("\n**Example chain:**")
        print(f"  vbroadcast id={vb_id} at cycle {vb_cycle}")
        print(f"  → valu1 id={v1_id} ({v1.get('opcode', '')}), cycle={v1_cycle}, completion={completion_v1}, already_shifted={v1_already_shifted}")
        print(f"  → valu2 id={v2_id} ({v2.get('opcode', '')}), scheduled cycle {v2.get('cycle', '?')}")
        print(f"  Current: valu2 ASAP = 1 + completion(valu1) = {asap_cur}. Combine: shift valu1 to ALU (if not already) + eliminate vb (feed valu1 via 8 scalars at 0) → valu1 completes at 1, valu2 ASAP = 2.")
        print(f"  **ASAP reduction for valu2: {asap_cur} → 2 (reduction = {reduction} cycles).**")
        n_show = min(12, len(chains))
        print(f"\nChains with largest ASAP reduction (up to {n_show}):")
        if md:
            print("\n| vb id | vb cycle | valu1 id | valu1 shifted? | completion(valu1) | valu2 id | ASAP current | ASAP if combined | reduction |")
            print("|-------|----------|----------|-----------------|-------------------|----------|--------------|------------------|-----------|")
        for c in chains[:n_show]:
            vb_id, vb_cycle, v1_id, v1_cycle, completion_v1, v1_already_shifted, v2_id, asap_cur, asap_elim, reduction = c
            ys = "yes" if v1_already_shifted else "no"
            if md:
                print(f"| {vb_id} | {vb_cycle} | {v1_id} | {ys} | {completion_v1} | {v2_id} | {asap_cur} | {asap_elim} | {reduction} |")
            else:
                print(f"  vb {vb_id} (cycle {vb_cycle}) → valu1 {v1_id} (shifted={v1_already_shifted}, completion {completion_v1}) → valu2 {v2_id}: ASAP {asap_cur} → {asap_elim}, reduction {reduction}")
        if len(chains) > n_show:
            print(f"  ... and {len(chains) - n_show} more.")
    else:
        print(
            "No chains vb → valu1 → valu2 with positive ASAP reduction (every such valu2 already has ASAP ≤ 2, or there are no downstream consumers)."
        )
        print(
            "Combine: shift valu1 to ALU and eliminate vbroadcast (feed valu1 via 8 scalars at cycle 0) → valu1 completes at 1, valu2 ASAP = 2."
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

    # --- ASAP bounds that could be reduced (per-lane completion) ---
    if md:
        print("## ASAP bounds that could be reduced (per-lane completion)\n")
    else:
        print("=== ASAP bounds that could be reduced (per-lane completion) ===\n")
    if not shifted:
        print("N/A (run with --enable-valu-shift and output-json for shifted_alu_cycles).\n")
    else:
        asap_list = asap_bounds(data, shifted)
        reducible = [(op, asap, asap_per_lane, lim, c_max, c_min) for op, asap, asap_per_lane, lim, c_max, c_min in asap_list if asap_per_lane < asap and (op.get("dep_ids"))]
        print(f"Ops whose ASAP could be reduced (per-lane completion of shifted dep): {len(reducible)}")
        if reducible:
            # Sort by reduction amount (largest first)
            reducible.sort(key=lambda r: (r[1] - r[2], -r[0].get("cycle", 0)))
            n_show = min(15, len(reducible))
            print(f"Showing {n_show} examples (largest ASAP reduction first):\n")
            if md:
                print("| op id | engine | opcode | current cycle | ASAP (current) | ASAP (per-lane) | limiting dep | dep completion max→min |")
                print("|-------|--------|--------|---------------|----------------|-----------------|--------------|------------------------|")
            for op, asap, asap_per_lane, lim, c_max, c_min in reducible[:n_show]:
                oid = op.get("id")
                eng = (op.get("engine") or "").lower()
                opcode = (op.get("opcode") or "").strip()[:12]
                cur = op.get("cycle", 0)
                lim_str = str(lim) if lim is not None else ""
                comp_str = f"{c_max}→{c_min}" if c_max is not None and c_min is not None and c_max != c_min else str(c_max) if c_max is not None else ""
                if md:
                    print(f"| {oid} | {eng} | {opcode} | {cur} | {asap} | {asap_per_lane} | {lim_str} | {comp_str} |")
                else:
                    print(f"  op {oid} ({eng} {opcode}): cycle={cur} ASAP={asap} → {asap_per_lane} (limiting dep {lim_str}, completion {comp_str})")
            if len(reducible) > n_show:
                print(f"\n... and {len(reducible) - n_show} more.")
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
