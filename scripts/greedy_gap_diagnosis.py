"""Where a schedule loses cycles: idle slots per engine over time, and whether work was waiting.

For each slice of the schedule this prints, from a scheduler output JSON:

- idle slots of the load, vector ALU, scalar ALU and flow engines;
- for the load engine, how many idle slots had a load that was ready (every operand
  available) and still unissued, and how many had none: a load slot idle with nothing ready
  is starved by the work upstream of the loads, one idle with a load ready is the
  scheduler's ordering;
- the average number of ready, unissued vector ops that can run as 8 scalar ops
  ("shiftable") and that cannot (`multiply_add`, `vbroadcast`): many non-shiftable ops
  waiting while the scalar ALU idles means shiftable ops took vector slots the
  non-shiftable ones needed.

"Ready" uses operand availability only (producer cycle + 1). It ignores the wavefront's
in-flight window, so ops of chains not yet admitted count as ready.

Usage (from the project root):
    uv run python -m scripts.greedy_gap_diagnosis sched.json [--slice 50] [--from 51]

docs/GREEDY_SHAPE_ABLATION.md section 11 uses it.
"""
import argparse
import collections
import json

NON_SHIFTABLE = {"multiply_add", "vbroadcast"}


def engine_use(sched: dict) -> dict:
    """Lowered slots used per engine per cycle."""
    total = sched["total_cycles"]
    use = collections.defaultdict(lambda: [0] * total)
    for op in sched["ops"]:
        use[op["engine"]][op["cycle"]] += 1
    return use


def waiting_counts(sched: dict, select) -> list:
    """Per cycle, SSA ops chosen by `select` that were ready and not yet issued."""
    total = sched["total_cycles"]
    by_id = {op["id"]: op for op in sched["ssa_ops"]}
    waiting = [0] * total
    for op in sched["ssa_ops"]:
        if op["cycle"] < 0 or not select(op):
            continue
        ready = max((by_id[d]["cycle"] + 1 for d in op["dep_ids"]
                     if d in by_id and by_id[d]["cycle"] >= 0), default=0)
        for t in range(ready, op["cycle"]):
            waiting[t] += 1
    return waiting


def report(sched: dict, slice_len: int, start: int) -> str:
    total = sched["total_cycles"]
    use = engine_use(sched)
    loads_waiting = waiting_counts(sched, lambda op: op["engine"] == "load")
    shift_waiting = waiting_counts(
        sched, lambda op: op["engine"] == "valu" and op["opcode"] not in NON_SHIFTABLE)
    fixed_waiting = waiting_counts(
        sched, lambda op: op["engine"] == "valu" and op["opcode"] in NON_SHIFTABLE)
    cap = {"load": 2, "valu": 6, "alu": 12, "flow": 1}

    lines = []
    idle_ready = idle_starved = 0
    for t in range(start, total):
        idle = cap["load"] - use["load"][t]
        idle_ready += min(idle, loads_waiting[t])
        idle_starved += max(0, idle - loads_waiting[t])
    load_cycles = [t for t in range(total) if use["load"][t]]
    lines.append(f"{total} cycles; last load at cycle {load_cycles[-1] if load_cycles else None}")
    lines.append(f"load slots idle from cycle {start}: {idle_ready} with a ready load waiting, "
                 f"{idle_starved} with none ready")
    lines.append("")
    lines.append(f"{'from':>5} {'load idle':>9} {'(ready)':>7} {'valu idle':>9} {'alu idle':>8} "
                 f"{'flow idle':>9} {'shiftable waiting':>17} {'non-shiftable waiting':>21}")
    for b in range(0, total, slice_len):
        rng = range(b, min(total, b + slice_len))
        n = len(rng)
        idle = {e: sum(cap[e] - use[e][t] for t in rng) for e in cap}
        ready = sum(min(cap["load"] - use["load"][t], loads_waiting[t]) for t in rng)
        lines.append(f"{b:>5} {idle['load']:>9} {ready:>7} {idle['valu']:>9} {idle['alu']:>8} "
                     f"{idle['flow']:>9} {sum(shift_waiting[t] for t in rng) / n:>17.1f} "
                     f"{sum(fixed_waiting[t] for t in rng) / n:>21.1f}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("schedule", help="scheduler output JSON (--output-json)")
    parser.add_argument("--slice", type=int, default=50, help="cycles per row (default 50)")
    parser.add_argument("--from", dest="start", type=int, default=0,
                        help="count idle load slots from this cycle (e.g. the static bound's head t)")
    args = parser.parse_args(argv)
    with open(args.schedule) as f:
        print(report(json.load(f), args.slice, args.start))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
