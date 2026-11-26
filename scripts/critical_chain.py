"""The dependency chains behind the static bound's head t and tail q, op by op.

The load-bound static bound is t + q + ceil(W / 2) (`scripts.static_bounds`, the row
`schedule_report` prints): t is the longest latency-weighted path from the start to the
first window gather load, q the longest path from the last one to the end. Every cycle
removed from either chain lowers the bound. For a scheduler output JSON this prints:

- the chain to a window load with head t, from its first op (latency of every edge, the
  op's head, its region: setup, or round / phase / group);
- the chain from a window load with tail q to the end;
- per region of the t-chain, how many cycles it contributes.

Usage (from the project root):
    uv run python -m scripts.critical_chain sched.json [--json out.json]
"""
import argparse
import collections
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
sys.path.insert(0, os.path.dirname(_HERE))

from scripts import static_bounds  # noqa: E402


def region(op: dict) -> str:
    si = op.get("source_info") or {}
    if si.get("round") is None:
        return f"setup/{si.get('phase')}"
    return f"r{si['round']}/{si.get('phase')}"


def chains(sched: dict) -> dict:
    bops = static_bounds.ops_from_schedule_json(sched, "run")
    rows = static_bounds.makespan_rows(bops)
    t, q = rows["t"], rows["q"]
    by_id = {o.id: o for o in bops}
    h, tl = static_bounds.heads(bops), static_bounds.tails(bops)
    succ = collections.defaultdict(list)
    for o in bops:
        for p, lat in o.deps:
            if p in by_id:
                succ[p].append((o.id, lat))
    ssa = {o["id"]: o for o in sched["ssa_ops"]}
    window = [o.id for o in bops if o.work.get("load", 0) and h[o.id] >= t and tl[o.id] >= q]
    first = min((i for i in window if h[i] == t), key=lambda i: i)
    last = min((i for i in window if tl[i] == q), key=lambda i: i)

    def row(i, lat):
        o = ssa.get(i, {})
        si = o.get("source_info") or {}
        return {"id": i, "opcode": o.get("opcode"), "engine": o.get("engine"), "latency_in": lat,
                "head": h[i], "tail": tl[i], "region": region(o) if o else "?", "group": si.get("group")}

    # Backward from `first`: the predecessor that realises the head.
    back = []
    i, lat_in = first, None
    while True:
        preds = [(p, lat) for p, lat in by_id[i].deps if p in by_id and h[p] + lat == h[i]]
        back.append((i, None))
        if not preds:
            break
        p, lat = min(preds)
        back[-1] = (i, lat)
        i = p
    back.reverse()
    head_chain = [row(i, lat) for i, lat in back]
    # Forward from `last`: the successor that realises the tail.
    fwd = []
    i = last
    while True:
        nxt = [(c, lat) for c, lat in succ[i] if tl[c] + lat == tl[i]]
        fwd.append(row(i, None))
        if not nxt:
            break
        c, lat = min(nxt)
        fwd[-1]["latency_out"] = lat
        i = c
    contrib = collections.Counter()
    for r in head_chain:
        contrib[r["region"].split("/")[0]] += r["latency_in"] or 0
    return {"t": t, "q": q, "w": rows["work"], "bound": rows["bound"], "engine": rows["engine"],
            "head_chain": head_chain, "tail_chain": fwd, "head_by_round": dict(contrib)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("schedule")
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    with open(args.schedule) as f:
        r = chains(json.load(f))
    print(f"static bound {r['bound']} ({r['engine']} row): t {r['t']}, q {r['q']}, W {r['w']}")
    print(f"head chain ({len(r['head_chain'])} ops; latency-weighted length {r['t']}):")
    for x in r["head_chain"]:
        print(f"  head {x['head']:4d}  +{x['latency_in'] if x['latency_in'] is not None else '-'}  "
              f"{x['opcode']:16s} {x['engine']:5s} {x['region']:28s} g{x['group']}")
    print("head cycles by region: " + ", ".join(f"{k} {v}" for k, v in r["head_by_round"].items()))
    print(f"tail chain ({len(r['tail_chain'])} ops; length {r['q']}):")
    for x in r["tail_chain"]:
        print(f"  tail {x['tail']:4d}  {x['opcode']:16s} {x['engine']:5s} {x['region']:28s} g{x['group']} "
              f"-> +{x.get('latency_out', '-')}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(r, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
