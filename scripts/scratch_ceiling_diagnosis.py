"""Where a load-bound greedy schedule loses cycles against the static head/tail bound.

The static bound of a load-bound schedule (`scripts.static_bounds`, the row `schedule_report`
prints) is `T >= t + q + ceil(W / 2)`: the W loads with head >= t and tail >= q must all issue
in the window [t, T - q). The script takes t, q, W and the window loads from that same row
(`static_bounds.ops_from_schedule_json(sched, "run")`), checks that every window load issued in
the window, and splits the gap exactly:

    2 * (T - bound) = idle + extra - (2 * ceil(W / 2) - W)

- **idle**: idle load slots in [t, T - q) (a last load before T - q - 1 shows up here: the
  slots after it are idle; there is no separate tail term);
- **extra**: loads issued in the window that W does not count (head < t: setup loads issued
  late; tail < q), which took slots W needed.

Given the counts the identity is arithmetic; what is checked (and fails loudly) is that every
window load issued inside [t, T - q) and that W equals the window loads issued, i.e. that the
row and the schedule agree.

Each idle slot is then **attributed** (accounting, not a counterfactual; removing a cause can
buy fewer cycles than it is charged, e.g. the 9.5 flow cycles on `path_flow90.json` bought 2):

- "ordering, dependent load ready" / "ordering, dependency-free load ready": a load whose
  operands were ready (producer cycle + 1) issued later. Dependency-free loads (constants) are
  kept apart: they are ready from cycle 0 by construction.
- otherwise "starved": the loads issued at the next cycle with a load whose dependency head
  (static head, no resources) is <= the idle cycle share the slot equally (fractional, so the
  result does not depend on which of them is listed first). For each, walk back through its
  latest-issued producer (ties: smallest op id) to the first op that issued later than its
  operands allowed, and charge that op's engine: "full" if in every cycle it waited its
  engine had no room for it, "not full" otherwise (priority, stagger, inflation, scratch). A
  shiftable vector op has no room when the vector ALU is full and fewer than 8 scalar slots
  are free (a split shift could start in fewer: an approximation); a non-shiftable vector op
  when the vector ALU is full. The walk stops at the nearest delayed op; an earlier delay on
  the chain is not charged.

Usage (from the project root):
    uv run python -m scripts.scratch_ceiling_diagnosis gap sched.json
    uv run python -m scripts.scratch_ceiling_diagnosis peak sched.json

docs/SCRATCH_CEILING_DIAGNOSIS.md and docs/STARTUP_AND_STALL_LOOP.md use both.
"""
import argparse
import collections
import json
import os
import sys
from fractions import Fraction

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
sys.path.insert(0, os.path.dirname(_HERE))

from scripts import static_bounds  # noqa: E402
from scripts.schedule_report import lowered_with_ssa  # noqa: E402

CAP = {"load": 2, "valu": 6, "alu": 12, "flow": 1, "store": 2}
VLEN = 8


def load_row(sched: dict) -> dict:
    """The static bound's load row for this schedule: t, q, W, the window ops and op heads.

    Raises ValueError when the strongest row is not the load row (the split assumes it)."""
    bops = static_bounds.ops_from_schedule_json(sched, "run")
    rows = static_bounds.makespan_rows(bops)
    if rows["engine"] != "load":
        raise ValueError(f"the static bound's row is {rows['engine']}, not load: this split is for load-bound schedules")
    t, q, w = rows["t"], rows["q"], rows["work"]
    h, tl = static_bounds.heads(bops), static_bounds.tails(bops)
    window_ids = {o.id for o in bops if o.work.get("load", 0) and h[o.id] >= t and tl[o.id] >= q}
    return {"t": t, "q": q, "w": w, "bound": rows["bound"], "window_ids": window_ids, "heads": h, "tails": tl}


def _uses(sched: dict):
    """Per-engine slot use per cycle, the lowered load ops with their SSA op, shifted SSA ids."""
    total = sched["total_cycles"]
    use = {e: [0] * (total + 1) for e in CAP}
    loads = []
    shifted = set()
    for low, s, lane in lowered_with_ssa(sched):
        if low["engine"] in use:
            use[low["engine"]][low["cycle"]] += 1
        if lane and s is not None:
            shifted.add(s["id"])
        if low["engine"] == "load":
            loads.append((low["cycle"], s["id"] if s is not None else None))
    return use, loads, shifted


def _operand_ready(op: dict, ops: dict) -> int:
    return max((ops[d]["cycle"] + 1 for d in op["dep_ids"] if d in ops), default=0)


def _no_room(op: dict, cycle: int, use: dict) -> bool:
    """Whether `op`'s engine had no room for it in `cycle` (see the module docstring)."""
    e = op["engine"]
    if e == "valu":
        if use["valu"][cycle] < CAP["valu"]:
            return False
        shiftable = op["opcode"] in static_bounds.SHIFTABLE_OPCODES
        return not shiftable or use["alu"][cycle] > CAP["alu"] - VLEN
    if e in CAP:
        return use[e][cycle] >= CAP[e]
    return False


def blame(load_id: int, ops: dict, use: dict) -> tuple:
    """(engine, no_room, op id) of the first delayed op on the load's latest-producer chain, or
    ("none", False, None) when the chain is tight."""
    op = ops[load_id]
    while True:
        ready = _operand_ready(op, ops)
        if op["cycle"] > ready:
            full = all(_no_room(op, c, use) for c in range(ready, op["cycle"]))
            return op["engine"], full, op["id"]
        preds = [ops[d] for d in op["dep_ids"] if d in ops]
        if not preds:
            return "none", False, None
        op = max(preds, key=lambda p: (p["cycle"], -p["id"]))


def gap_report(sched: dict, row: dict | None = None) -> dict:
    """The exact split of T - bound (see the module docstring). `row` defaults to `load_row`."""
    row = row or load_row(sched)
    t, q, w = row["t"], row["q"], row["w"]
    total = sched["total_cycles"]
    ops = {o["id"]: o for o in sched["ssa_ops"] if o["cycle"] >= 0}
    use, loads, _ = _uses(sched)
    end = total - q
    misplaced = [i for c, i in loads if i in row["window_ids"] and not t <= c < end]
    if misplaced:
        raise AssertionError(f"window loads outside [t, T - q) = [{t}, {end}): {misplaced[:5]}")
    in_window = [(c, i) for c, i in loads if t <= c < end]
    if sum(1 for _, i in in_window if i in row["window_ids"]) != w:
        raise AssertionError(f"W = {w} but {sum(1 for _, i in in_window if i in row['window_ids'])} window loads issued")
    extra = [(c, i) for c, i in in_window if i not in row["window_ids"]]
    extra_head = sum(1 for _, i in extra if i is not None and row["heads"].get(i, 0) < t)
    idle = {c: CAP["load"] - use["load"][c] for c in range(t, end) if use["load"][c] < CAP["load"]}
    n_idle = sum(idle.values())
    bound = t + q + -(-w // CAP["load"])
    gap = total - bound
    if 2 * gap != n_idle + len(extra) - (2 * -(-w // CAP["load"]) - w):
        raise AssertionError(f"identity fails: 2*{gap} != {n_idle} + {len(extra)} - {2 * -(-w // 2) - w}")

    # Attribution of the idle slots.
    load_ops = sorted({i for _, i in loads if i is not None and i in ops}, key=lambda i: (ops[i]["cycle"], i))
    classes = collections.defaultdict(Fraction)
    for c, n in sorted(idle.items()):
        later = [i for i in load_ops if ops[i]["cycle"] > c]
        ready = [i for i in later if _operand_ready(ops[i], ops) <= c]
        if any(ops[i]["dep_ids"] for i in ready):
            classes["ordering, dependent load ready"] += n
            continue
        if ready:
            classes["ordering, dependency-free load ready"] += n
            continue
        possible = [i for i in later if row["heads"].get(i, 0) <= c]
        if not possible:
            classes["no load's dependency head allows it"] += n
            continue
        first = min(ops[i]["cycle"] for i in possible)
        group = [i for i in possible if ops[i]["cycle"] == first]
        for i in group:
            engine, full, _ = blame(i, ops, use)
            label = "starved: tight chain" if engine == "none" else \
                f"starved: delayed {engine} op, {'no room' if full else 'room'}"
            classes[label] += Fraction(n, len(group))
    return {
        "cycles": total, "bound": bound, "row_bound": row["bound"], "gap": gap, "t": t, "q": q, "w": w,
        "idle_slots": n_idle, "idle_cycles": sorted(idle),
        "extra_loads": len(extra), "extra_head_below_t": extra_head,
        "last_load": max(c for c, _ in loads),
        "classes": {k: float(v) for k, v in classes.items()},
    }


def peak_report(sched: dict) -> dict:
    curve = [c["total_live_words"] for c in sched["liveness_curve"]]
    peak = max(curve)
    pc = curve.index(peak)
    top = collections.Counter(x for x in curve if x >= peak - 32)
    live = [v for v in sched["vregs"] if v["interval_start"] <= pc < v["interval_end"]]
    kinds = collections.Counter()
    for v in live:
        kinds[v["var_type"] or v["phase"] or "unknown"] += v["size_words"]
    return {"peak": peak, "peak_cycle": pc, "top_values": sorted(top.items()),
            "live_by_kind": dict(kinds.most_common())}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gap", help="split T - static bound into idle load slots and extra loads; attribute idle slots")
    g.add_argument("schedule")
    p = sub.add_parser("peak", help="liveness peak and what is live there")
    p.add_argument("schedule")
    args = parser.parse_args(argv)
    with open(args.schedule) as f:
        sched = json.load(f)
    if args.cmd == "gap":
        r = gap_report(sched)
        print(f"{r['cycles']} cycles, static bound {r['bound']} (load row t={r['t']}, q={r['q']}, W={r['w']}): "
              f"gap {r['gap']}  [2*gap = idle + extra - rounding: checked]")
        print(f"  idle load slots in [t, T-q): {r['idle_slots']} = {r['idle_slots'] / 2:.1f} cycles "
              f"(last load at {r['last_load']}; idle at cycles {r['idle_cycles']})")
        print(f"  extra loads in the window (not in W): {r['extra_loads']} = {r['extra_loads'] / 2:.1f} cycles "
              f"({r['extra_head_below_t']} with head < t)")
        print("  idle slots by attributed cause (accounting, not counterfactual):")
        for k, n in sorted(r["classes"].items(), key=lambda x: -x[1]):
            print(f"    {n:6.1f} slots ({n / 2:5.2f} cycles)  {k}")
    else:
        r = peak_report(sched)
        print(f"peak {r['peak']} words at cycle {r['peak_cycle']}; curve values within 32 of the peak: "
              f"{r['top_values']}")
        for k, n in r["live_by_kind"].items():
            print(f"  {k:>24}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
