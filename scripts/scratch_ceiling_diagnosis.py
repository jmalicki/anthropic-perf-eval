"""Where a load-bound greedy schedule loses cycles against the static head/tail bound.

The static bound of a load-bound schedule (`scripts.schedule_report`, row load) is
`T >= t + q + ceil(W / 2)`: W loads cannot issue before cycle t and must leave q cycles
after them. So every cycle of `T - bound` is one of:

- **tail**: the last load issues before `T - q - 1` (cycles after it not explained by q);
- **idle load slots in [t, T - q)**: two idle slots are one cycle;
- **early loads in the window**: loads whose head is below t (W does not count them) that
  issued inside the window and took slots W needed. Two are one cycle.

Each idle load slot is then classified by a blame chain. Take the load that issued next and
was dependency-ready (operand-ready by longest path, no resources) at or before the idle
cycle; walk back through its latest-issued producer until reaching an op that issued later
than its own operands allowed (`delayed op`). The idle slot is charged to that op's engine,
split into "engine full" (every slot of that engine was used in every cycle the op waited:
throughput) and "engine not full" (the op waited although its engine had room: priority,
wavefront stagger, inflation or scratch policy). If a ready load was waiting at the idle
cycle the slot is "ordering" (the load engine picked nothing although a load was ready).

"Ready" uses operand availability (producer cycle + 1), like `scripts.greedy_gap_diagnosis`.

Usage (from the project root):
    uv run python -m scripts.scratch_ceiling_diagnosis gap sched.json --t 51 --q 14 --w 1888
    uv run python -m scripts.scratch_ceiling_diagnosis peak sched.json

`peak` prints the liveness peak, the cycles at the top of the curve and what is live at the
peak by value kind. docs/SCRATCH_CEILING_DIAGNOSIS.md uses both.
"""
import argparse
import collections
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
sys.path.insert(0, os.path.dirname(_HERE))

from scripts.schedule_report import lowered_with_ssa  # noqa: E402

CAP = {"load": 2, "valu": 6, "alu": 12, "flow": 1, "store": 2}


def _engine_use(sched: dict) -> dict:
    total = sched["total_cycles"]
    use = {e: [0] * (total + 1) for e in CAP}
    for op in sched["ops"]:
        if op["engine"] in use:
            use[op["engine"]][op["cycle"]] += 1
    return use


def _ssa(sched: dict) -> dict:
    return {o["id"]: o for o in sched["ssa_ops"] if o["cycle"] >= 0}


def _operand_ready(op: dict, ops: dict) -> int:
    return max((ops[d]["cycle"] + 1 for d in op["dep_ids"] if d in ops), default=0)


def _dep_asap(ops: dict) -> dict:
    """Longest-path earliest cycle of every op, unit latency (ids are topological)."""
    asap = {}
    for i in sorted(ops):
        asap[i] = max((asap[d] + 1 for d in ops[i]["dep_ids"] if d in asap), default=0)
    return asap


def blame(load: dict, ops: dict, use: dict, shifted: set) -> tuple:
    """First op on `load`'s latest-producer chain that issued after its operands were ready.

    Returns (engine, engine_full, op) or ("none", False, None) if the chain is tight.
    """
    op = load
    while True:
        ready = _operand_ready(op, ops)
        if op["cycle"] > ready:
            engine = "alu" if op["id"] in shifted else op["engine"]
            full = all(use[engine][c] >= CAP[engine] for c in range(ready, op["cycle"]))
            if op["engine"] == "valu" and op["id"] not in shifted:
                # A vector op could have run on the vector ALU or shifted: full only if both were.
                full = all(use["valu"][c] >= CAP["valu"] for c in range(ready, op["cycle"]))
            return engine, full, op
        preds = [ops[d] for d in op["dep_ids"] if d in ops]
        if not preds:
            return "none", False, None
        op = max(preds, key=lambda p: p["cycle"])


def gap_report(sched: dict, t: int, q: int, w: int) -> dict:
    total = sched["total_cycles"]
    ops = _ssa(sched)
    use = _engine_use(sched)
    asap = _dep_asap(ops)
    shifted = {s["id"] for _, s, lane in lowered_with_ssa(sched) if lane}
    loads = sorted((o for o in ops.values() if o["engine"] == "load"), key=lambda o: o["cycle"])
    last = max(o["cycle"] for o in loads)
    window = range(t, total - q)
    idle = {c: CAP["load"] - use["load"][c] for c in window if use["load"][c] < CAP["load"]}
    in_window = [o for o in loads if t <= o["cycle"] < total - q]
    early = sum(1 for o in in_window if asap[o["id"]] < t)
    classes = collections.Counter()
    examples = {}
    for c, n in sorted(idle.items()):
        waiting = [o for o in loads if o["cycle"] > c and _operand_ready(o, ops) <= c]
        if waiting:
            classes["ordering (a ready load waited)"] += n
            continue
        nxt = [o for o in loads if o["cycle"] > c and asap[o["id"]] <= c]
        if not nxt:
            classes["no load could be ready (dependency heads)"] += n
            continue
        engine, full, culprit = blame(nxt[0], ops, use, shifted)
        key = f"starved: delayed {engine} op, engine {'full' if full else 'not full'}"
        classes[key] += n
        examples.setdefault(key, (c, culprit["id"] if culprit else None,
                                  culprit["opcode"] if culprit else None,
                                  (culprit.get("source_info") or {}).get("group") if culprit else None))
    bound = t + q + -(-w // CAP["load"])
    return {
        "cycles": total, "bound": bound, "gap": total - bound,
        "tail_cycles": (total - q - 1) - last,
        "idle_slots": sum(idle.values()), "idle_cycles": sorted(idle),
        "early_loads_in_window": early,
        "loads_in_window": len(in_window), "w": w,
        "classes": dict(classes), "examples": examples,
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
    g = sub.add_parser("gap", help="split T - static bound into tail, idle load slots, early loads")
    g.add_argument("schedule")
    g.add_argument("--t", type=int, required=True, help="head t of the load row (schedule_report)")
    g.add_argument("--q", type=int, required=True, help="tail q of the load row")
    g.add_argument("--w", type=int, required=True, help="W of the load row")
    p = sub.add_parser("peak", help="liveness peak and what is live there")
    p.add_argument("schedule")
    args = parser.parse_args(argv)
    with open(args.schedule) as f:
        sched = json.load(f)
    if args.cmd == "gap":
        r = gap_report(sched, args.t, args.q, args.w)
        print(f"{r['cycles']} cycles, static bound {r['bound']}: gap {r['gap']}")
        print(f"  tail beyond q: {r['tail_cycles']} cycles")
        print(f"  idle load slots in [t, T-q): {r['idle_slots']} = {r['idle_slots'] / 2:.1f} cycles "
              f"(at cycles {r['idle_cycles']})")
        print(f"  loads with head < t issued in the window: {r['early_loads_in_window']} = "
              f"{r['early_loads_in_window'] / 2:.1f} cycles "
              f"({r['loads_in_window']} loads in the window, W = {r['w']})")
        print("  idle slots by cause:")
        for k, n in sorted(r["classes"].items(), key=lambda x: -x[1]):
            print(f"    {n:4d} slots ({n / 2:5.1f} cycles)  {k}  e.g. {r['examples'].get(k)}")
    else:
        r = peak_report(sched)
        print(f"peak {r['peak']} words at cycle {r['peak_cycle']}; curve values within 32 of the peak: "
              f"{r['top_values']}")
        for k, n in r["live_by_kind"].items():
            print(f"  {k:>24}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
