"""Where a schedule leaves slots idle, per engine over time, and how much work was waiting.

For each slice of the schedule this prints, from a scheduler output JSON:

- idle slots of the load, vector ALU, scalar ALU and flow engines (all lowered ops count,
  constants included);
- for the load engine, how many idle slots had a data load (not a constant) whose operands
  were available and which had not issued yet ("operand-ready"), and how many had none;
- the average number of operand-ready, unissued vector ops that can run as 8 scalar ops
  ("shiftable") and that cannot (`multiply_add`, `vbroadcast`).

**What "operand-ready" means, and what it does not.** An op is operand-ready at cycle t when
every producer in its SSA `dep_ids` issued at t - 1 or earlier (unit latency). That ignores
what else can hold an op back in the greedy: the wavefront's in-flight window, ASAP/bounds
and inflation deferral, the scratch budget, memory-ordering edges (not in the JSON), the
extra cycle of a select on the multiply-add path, and the completion cycle of a split shift
when it is not the producer's recorded cycle. Every one of these makes an op ready *later*
than counted here, so the operand-ready counts are upper bounds on the work the scheduler
could really have placed, and the "none operand-ready" count of idle load slots is a lower
bound on load slots the scheduler could not have filled. The counts locate idle capacity and
work queues; they do not show that a different order would have filled a slot.

Constants (`const_choice`, which the greedy lowers to a `const` load or an `add_imm` on the
flow engine) are excluded from the load queue and reported separately.

Usage (from the project root):
    uv run python -m scripts.greedy_gap_diagnosis sched.json [--slice 50] [--from 51]

docs/GREEDY_SHAPE_ABLATION.md section 11 uses it.
"""
import argparse
import collections
import json

NON_SHIFTABLE = {"multiply_add", "vbroadcast"}
CONSTANT_OPCODES = {"const_choice", "const"}
CAPACITY = {"load": 2, "valu": 6, "alu": 12, "flow": 1}


def engine_use(sched: dict) -> dict:
    """Lowered slots used per engine per cycle."""
    total = sched["total_cycles"]
    use = collections.defaultdict(lambda: [0] * total)
    for op in sched["ops"]:
        use[op["engine"]][op["cycle"]] += 1
    return use


def is_data_load(op: dict) -> bool:
    """A load that is not a constant (constants may be lowered to flow add_imm)."""
    return op["engine"] == "load" and op["opcode"] not in CONSTANT_OPCODES


def operand_ready_waiting(sched: dict, select) -> list:
    """Per cycle, SSA ops chosen by `select` that were operand-ready and not yet issued.

    Operand-ready = every producer in `dep_ids` issued at least one cycle earlier. An upper
    bound on what was really placeable (see the module docstring).
    """
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


def summary(sched: dict, start: int) -> dict:
    """Idle load slots from `start`, split by whether a data load was operand-ready."""
    total = sched["total_cycles"]
    use = engine_use(sched)
    waiting = operand_ready_waiting(sched, is_data_load)
    with_ready = without_ready = 0
    for t in range(start, total):
        idle = CAPACITY["load"] - use["load"][t]
        with_ready += min(idle, waiting[t])
        without_ready += max(0, idle - waiting[t])
    data_cycles = [op["cycle"] for op in sched["ssa_ops"] if is_data_load(op) and op["cycle"] >= 0]
    constants = collections.Counter(op["opcode"] for op in sched["ops"]
                                    if op["opcode"] in ("const", "add_imm"))
    return {"cycles": total, "last_data_load": max(data_cycles, default=None),
            "idle_with_operand_ready_load": with_ready,
            "idle_without_operand_ready_load": without_ready,
            "constants_as_load": constants.get("const", 0),
            "constants_as_add_imm": constants.get("add_imm", 0)}


def report(sched: dict, slice_len: int, start: int) -> str:
    total = sched["total_cycles"]
    use = engine_use(sched)
    shift_waiting = operand_ready_waiting(
        sched, lambda op: op["engine"] == "valu" and op["opcode"] not in NON_SHIFTABLE)
    fixed_waiting = operand_ready_waiting(
        sched, lambda op: op["engine"] == "valu" and op["opcode"] in NON_SHIFTABLE)
    s = summary(sched, start)

    lines = [
        f"{s['cycles']} cycles; last data load at cycle {s['last_data_load']}; constants: "
        f"{s['constants_as_load']} as const loads, {s['constants_as_add_imm']} as flow add_imm",
        f"load slots idle from cycle {start}: {s['idle_without_operand_ready_load']} with no data "
        f"load operand-ready (a lower bound on slots no order could fill), "
        f"{s['idle_with_operand_ready_load']} with one operand-ready (an upper bound; the load may "
        f"still have been held by the window, deferral, the budget or a memory edge)",
        "",
        "Waiting columns: average operand-ready, unissued vector ops (upper bounds).",
        f"{'from':>5} {'load idle':>9} {'valu idle':>9} {'alu idle':>8} {'flow idle':>9} "
        f"{'shiftable waiting':>17} {'non-shiftable waiting':>21}",
    ]
    for b in range(0, total, slice_len):
        rng = range(b, min(total, b + slice_len))
        n = len(rng)
        idle = {e: sum(CAPACITY[e] - use[e][t] for t in rng) for e in CAPACITY}
        lines.append(f"{b:>5} {idle['load']:>9} {idle['valu']:>9} {idle['alu']:>8} "
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
