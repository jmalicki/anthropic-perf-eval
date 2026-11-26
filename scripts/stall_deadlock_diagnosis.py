"""Why a `--greedy-budget-force stall` run got stuck at the scratch budget.

Inputs: the state the greedy writes at the first cycle `stall` forces an op
(`SCHEDULER_DEADLOCK_DUMP=state.json`) and the failed build's exported schedule
(`scheduler_output_error.json`: every op's eventual cycle, operands and destination).

Prints, for that cycle (the first stall force; not necessarily a deadlock):
- the ready ops deferred this cycle, each with its reason ("budget": the hard budget check;
  "pressure": the pressure policy's soft deferral) and the words it would add and free.
  "Frees" is approximate: it counts a source operand whose only remaining reader is this op,
  by the operand's own id; it does not resolve vector-group lane aliases and ignores
  `keep_live`, a select's delta and a value listed twice as a source. Treat "adds more than
  it frees" as an indication, not a proof that no ready op fits;
- live words grouped by the state of each value's first remaining reader (by op id):
  "deferred" (ready, with its reason), "inflation" (deferred by ASAP inflation this cycle),
  "in heap" (in the ready heap, operands not yet visible), "not admitted" (outside the
  groups-in-flight window), or "blocked on op X" (an unscheduled operand producer; its own
  state is given). Readers are found through source operands only: a value whose remaining
  readers read it through a lane alias, `keep_live` or a select's delta has the dump's
  `remaining_users` above 0 but no reader here, and is reported as "readers not resolved".
  With `--live`, every live value and its first readers.

Usage (from the project root):
    SCHEDULER_DEADLOCK_DUMP=state.json uv run generate_ssa_kernel.py ... --greedy-budget-force stall
    uv run python -m scripts.stall_deadlock_diagnosis state.json scheduler_output_error.json
"""
import argparse
import collections
import json


def vreg_id(x):
    """VReg id from a dest string like 'v124_s' / 'v7_v' or an operand dict."""
    if isinstance(x, dict):
        x = x.get("VReg", x)
        return x.get("id") if isinstance(x, dict) else None
    if isinstance(x, str) and x.startswith("v"):
        return int(x[1:].split("_")[0])
    return None


def analyse(state: dict, sched: dict) -> dict:
    cyc = state["cycle"]
    ops = {o["id"]: o for o in sched["ssa_ops"]}
    producer = {}
    readers = collections.defaultdict(list)
    for o in ops.values():
        d = vreg_id(o.get("dest"))
        if d is not None:
            producer[d] = o["id"]
        for src in o.get("srcs", []):
            v = vreg_id(src)
            if v is not None:
                readers[v].append(o["id"])
    # Older dumps list ids only (no reason).
    deferred = {(d["id"] if isinstance(d, dict) else d): (d.get("reason", "unknown") if isinstance(d, dict) else "unknown")
                for d in state["budget_deferred"]}
    inflated = set(state.get("inflation_deferred", []))
    heap = {h["id"]: h for h in state["heap"]}
    live = {v["vreg"]: v for v in state["live"]}
    done = lambda i: 0 <= ops[i]["cycle"] < cyc  # noqa: E731  (scheduled before the dumped cycle)

    def status(i, depth=0):
        if i in deferred:
            return f"deferred (ready, {deferred[i]})"
        if i in inflated:
            return "inflation (deferred this cycle)"
        if i in heap:
            if not heap[i]["in_window"]:
                return "not admitted (groups in flight)"
            # The heap key is an order, not a wait: the op is offered again every cycle.
            return "in heap (operands not yet visible)"
        pending = [d for d in ops[i]["dep_ids"] if d in ops and not done(d)]
        if not pending:
            return "ready but not tried"
        first = min(pending)
        inner = status(first, depth + 1) if depth < 6 else "..."
        return f"blocked on op {first} ({ops[first]['opcode']}, {inner})"

    def words_added(i):
        d = vreg_id(ops[i].get("dest"))
        return 0 if d is None else (8 if str(ops[i].get("dest", "")).endswith("_v") else 1)

    def words_freed(i):
        freed = 0
        for src in ops[i].get("srcs", []):
            v = vreg_id(src)
            if v in live and live[v]["remaining_users"] == 1:
                freed += live[v]["words"]
        return freed

    deferred_rows = [{"id": i, "opcode": ops[i]["opcode"], "engine": ops[i]["engine"],
                      "group": (ops[i].get("source_info") or {}).get("group"),
                      "round": (ops[i].get("source_info") or {}).get("round"),
                      "reason": deferred[i], "adds": words_added(i), "frees": words_freed(i)}
                     for i in sorted(deferred)]
    live_rows = []
    for v, info in live.items():
        rem = [r for r in readers.get(v, []) if not done(r)]
        p = producer.get(v)
        si = (ops[p].get("source_info") or {}) if p in ops else {}
        live_rows.append({"vreg": v, "words": info["words"], "producer": p,
                          "group": si.get("group"), "round": si.get("round"), "phase": si.get("phase"),
                          "readers": [(r, status(r)) for r in sorted(rem)]})
    kinds = collections.Counter()
    for row in live_rows:
        if not row["readers"]:
            if live[row["vreg"]]["remaining_users"] > 0:
                kinds["readers not resolved (lane alias, keep_live or delta)"] += row["words"]
            else:
                kinds["no reader left (released next cycle)"] += row["words"]
            continue
        st = row["readers"][0][1]
        key = ("deferred" if st.startswith("deferred") else "inflation" if st.startswith("inflation")
               else "in heap" if st.startswith("in heap")
               else "not admitted" if st.startswith("not admitted")
               # "inflation (deferred this cycle)" contains "deferred": test inflation first.
               else "blocked behind an inflation-deferred op" if "inflation" in st
               else "blocked behind a deferred op" if "deferred" in st
               else "blocked behind an op in the heap" if "in heap" in st
               else "blocked (other)")
        kinds[key] += row["words"]
    return {"cycle": cyc, "live_words": state["live_words"], "budget": state["budget"], "slots": state["slots"],
            "inflation_deferred": len(inflated), "deferred": deferred_rows, "live": live_rows, "words_by_first_reader_state": dict(kinds)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("state")
    ap.add_argument("schedule")
    ap.add_argument("--live", action="store_true", help="also list every live value")
    args = ap.parse_args(argv)
    with open(args.state) as f:
        state = json.load(f)
    with open(args.schedule) as f:
        sched = json.load(f)
    r = analyse(state, sched)
    print(f"first stall force at cycle {r['cycle']}: {r['live_words']} live words, budget {r['budget']}, "
          f"slots used {r['slots']}; {r['inflation_deferred']} ops deferred by ASAP inflation this cycle")
    print(f"deferred ready ops ({len(r['deferred'])}):")
    for d in r["deferred"]:
        print(f"  op {d['id']} {d['opcode']} ({d['engine']}, group {d['group']}, round {d['round']}, {d['reason']}): "
              f"adds {d['adds']}, frees {d['frees']}")
    print("live words by the state of each value's first remaining reader:")
    for k, w in sorted(r["words_by_first_reader_state"].items(), key=lambda x: -x[1]):
        print(f"  {w:5d}  {k}")
    if args.live:
        for row in sorted(r["live"], key=lambda x: (str(x["group"]), str(x["round"]))):
            print(f"  v{row['vreg']} {row['words']}w g{row['group']} r{row['round']} {row['phase']}: "
                  + "; ".join(f"op {i}: {st}" for i, st in row["readers"][:2]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
