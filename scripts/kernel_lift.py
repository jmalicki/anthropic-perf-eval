#!/usr/bin/env python3
"""Lift a straight-line kernel (bundles) into a dependency graph over its ops, and report on it.

    uv run python -m scripts.kernel_lift kernel.json [more.json ...] [--table] [--json out.json]

Works on any kernel JSON with bundles under ``instrs`` (``--save-kernel`` output, or a public
kernel captured by ``scripts/capture_public_kernel.py``). Every slot becomes one op.

Edges, each with a minimum distance in cycles (``cycle[dst] >= cycle[src] + latency``):

* Scratch, per word (vector ops are 8 words, so lanes are exact): read-after-write from the
  last writer (latency 1: results are visible the next cycle); write-after-read from every
  read since that write (latency 0: a slot reads before the cycle's writes commit, so the
  overwrite may share the reader's cycle); write-after-write from the last writer
  (latency 1: two writes of one word in one cycle are a conflict). A word never written
  reads the machine's initial 0, and its first writer is ordered after those reads.
* Main memory: exact, from ``verification.memory_alias`` - an edge only between ops whose
  addresses can be equal for some valid input (Z3), same latencies as for scratch.

The report gives per-engine op counts, capacity floors, the critical path of the graph,
the cycles the kernel takes on the upstream simulator, the gap to the floor, and how full
the cycles are over time (start-up and tail).
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys
from dataclasses import dataclass

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.schedule_report import emission_floor  # noqa: E402
from verification.memory_alias import KOp, MemLayout, SymbolicKernel, concrete_check, load_bundles  # noqa: E402

ENGINES = ("alu", "valu", "load", "store", "flow")
CAP = {"alu": 12, "valu": 6, "load": 2, "store": 2, "flow": 1}
VLEN = 8
NON_SHIFTABLE = ("multiply_add", "vbroadcast")


@dataclass
class LiftedKernel:
    ops: list[KOp]
    edges: dict[tuple[int, int], int]  # (src, dst) -> latency
    mem_edges: int
    sym: SymbolicKernel
    raw: set[tuple[int, int]] = None  # the edges that carry a value (read-after-write)

    @property
    def n(self) -> int:
        return len(self.ops)

    def preds(self) -> list[list[tuple[int, int]]]:
        out: list[list[tuple[int, int]]] = [[] for _ in self.ops]
        for (s, d), lat in sorted(self.edges.items()):
            out[d].append((s, lat))
        return out

    def succs(self) -> list[list[tuple[int, int]]]:
        out: list[list[tuple[int, int]]] = [[] for _ in self.ops]
        for (s, d), lat in sorted(self.edges.items()):
            out[s].append((d, lat))
        return out


def scratch_edges(ops: list[KOp]) -> tuple[dict[tuple[int, int], int], set[tuple[int, int]]]:
    """Exact RAW/WAR/WAW edges on scratch words; ops in one cycle read before any writes.
    Returns (edges with latency, the subset that is read-after-write)."""
    edges: dict[tuple[int, int], int] = {}
    raw: set[tuple[int, int]] = set()
    last_writer: dict[int, int] = {}
    readers: dict[int, list[int]] = collections.defaultdict(list)

    def add(s: int, d: int, lat: int) -> None:
        if s != d and edges.get((s, d), -1) < lat:
            edges[(s, d)] = lat

    by_cycle: dict[int, list[KOp]] = collections.defaultdict(list)
    for op in ops:
        by_cycle[op.cycle].append(op)
    for cycle in sorted(by_cycle):
        group = by_cycle[cycle]
        for op in group:
            for a in op.reads:
                if a in last_writer:
                    add(last_writer[a], op.index, 1)
                    raw.add((last_writer[a], op.index))
                readers[a].append(op.index)
        for op in group:
            for a in op.writes:
                for r in readers.pop(a, []):
                    add(r, op.index, 0)
                if a in last_writer:
                    add(last_writer[a], op.index, 1)
                last_writer[a] = op.index
    return edges, raw


def lift(bundles: list[dict], layout: MemLayout | None = None) -> LiftedKernel:
    sym = SymbolicKernel(bundles, layout).run()
    edges, raw = scratch_edges(sym.ops)
    for (s, d), lat in sym.mem_edges.items():
        if edges.get((s, d), -1) < lat:
            edges[(s, d)] = lat
        if sym.ops[s].mem == "store" and sym.ops[d].mem == "load":
            raw.add((s, d))
    return LiftedKernel(sym.ops, edges, len(sym.mem_edges), sym, raw)


def save_graph(lk: LiftedKernel, path: str, name: str = "") -> None:
    """Write the lifted graph (ops with their original cycle, and edges) as JSON, so a
    rescheduler can read it without repeating the alias analysis."""
    data = {
        "name": name,
        "ops": [{"cycle": o.cycle, "engine": o.engine, "slot": o.slot} for o in lk.ops],
        "edges": [[s, d, lat] for (s, d), lat in sorted(lk.edges.items())],
        "memory_edges": lk.mem_edges,
        "raw": sorted(lk.raw),
    }
    with open(path, "w") as f:
        json.dump(data, f)


@dataclass
class Graph:
    """A lifted graph read back from `save_graph` (no symbolic state)."""

    ops: list[KOp]
    edges: dict[tuple[int, int], int]
    mem_edges: int
    raw: set[tuple[int, int]]

    n = LiftedKernel.n
    preds = LiftedKernel.preds
    succs = LiftedKernel.succs


def load_graph(path: str) -> Graph:
    with open(path) as f:
        data = json.load(f)
    ops = [KOp(i, o["cycle"], o["engine"], o["slot"]) for i, o in enumerate(data["ops"])]
    return Graph(ops, {(s, d): lat for s, d, lat in data["edges"]}, data["memory_edges"],
                 {tuple(e) for e in data.get("raw", [])})


# =============================================================================
# Graph measures
# =============================================================================


@dataclass
class Condensed:
    """The graph with each co-issue group (ops joined by a cycle of latency-0 edges, which
    must share a cycle) collapsed to one node. Kept fixed addresses can force this: op A
    reads w and writes v while op B reads v and writes w, so each must not be later than
    the other."""

    group: list[int]  # op -> group id
    members: list[list[int]]  # group id -> ops
    preds: list[list[tuple[int, int]]]  # group -> (pred group, latency)
    succs: list[list[tuple[int, int]]]
    order: list[int]  # groups in topological order


def condense(lk) -> Condensed:
    n = lk.n
    zero_succ: list[list[int]] = [[] for _ in range(n)]
    for (s, d), lat in lk.edges.items():
        if lat == 0:
            zero_succ[s].append(d)
    # Tarjan's strongly connected components on latency-0 edges, iteratively.
    index = [-1] * n
    low = [0] * n
    on_stack = [False] * n
    stack: list[int] = []
    group = [-1] * n
    members: list[list[int]] = []
    counter = 0
    for root in range(n):
        if index[root] != -1:
            continue
        work = [(root, 0)]
        while work:
            u, k = work.pop()
            if k == 0:
                index[u] = low[u] = counter
                counter += 1
                stack.append(u)
                on_stack[u] = True
            if k < len(zero_succ[u]):
                work.append((u, k + 1))
                v = zero_succ[u][k]
                if index[v] == -1:
                    work.append((v, 0))
                elif on_stack[v]:
                    low[u] = min(low[u], index[v])
                continue
            if low[u] == index[u]:
                comp = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    group[w] = len(members)
                    comp.append(w)
                    if w == u:
                        break
                members.append(sorted(comp))
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[u])
    m = len(members)
    gp: dict[tuple[int, int], int] = {}
    for (s, d), lat in lk.edges.items():
        gs, gd = group[s], group[d]
        if gs == gd:
            if lat > 0:
                raise ValueError(f"ops {s} and {d} must share a cycle but need {lat} apart: infeasible graph")
            continue
        if gp.get((gs, gd), -1) < lat:
            gp[(gs, gd)] = lat
    preds: list[list[tuple[int, int]]] = [[] for _ in range(m)]
    succs: list[list[tuple[int, int]]] = [[] for _ in range(m)]
    for (gs, gd), lat in sorted(gp.items()):
        preds[gd].append((gs, lat))
        succs[gs].append((gd, lat))
    indeg = [len(p) for p in preds]
    order = [g for g in range(m) if indeg[g] == 0]
    head = 0
    while head < len(order):
        u = order[head]
        head += 1
        for v, _ in succs[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                order.append(v)
    if len(order) != m:
        raise ValueError("dependency cycle with a positive latency: infeasible graph")
    return Condensed(group, members, preds, succs, order)


def topo_order(lk) -> list[int]:
    """Ops in a topological order of the condensed graph (co-issue groups contiguous)."""
    c = condense(lk)
    return [op for g in c.order for op in c.members[g]]


def asap(lk, c: Condensed | None = None) -> list[int]:
    c = c or condense(lk)
    t = [0] * len(c.members)
    for g in c.order:
        t[g] = max((t[p] + lat for p, lat in c.preds[g]), default=0)
    return [t[c.group[i]] for i in range(lk.n)]


def heights(lk, c: Condensed | None = None) -> list[int]:
    """Cycles from an op's issue to the end of the program along the longest path (>= 1)."""
    c = c or condense(lk)
    h = [1] * len(c.members)
    for g in reversed(c.order):
        h[g] = max((h[v] + lat for v, lat in c.succs[g]), default=1)
    return [h[c.group[i]] for i in range(lk.n)]


def dataflow_critical_path(lk) -> int:
    """Longest chain of read-after-write edges, in cycles: the critical path of the values
    alone, before the write-after-read/write-after-write edges that reusing scratch
    addresses adds. A different address assignment could approach it; the full graph's
    critical path holds for these addresses."""
    pred: dict[int, list[int]] = collections.defaultdict(list)
    for s, d in lk.raw:
        pred[d].append(s)
    depth = [1] * lk.n
    for i in sorted(range(lk.n), key=lambda i: lk.ops[i].cycle):
        depth[i] = max((depth[p] + 1 for p in pred[i]), default=1)
    return max(depth, default=0)


def check_schedule(lk: LiftedKernel, cycles: list[int]) -> list[str]:
    """Violated edges and slot overflows of a schedule (empty list = valid)."""
    bad = [f"edge {s}->{d} needs +{lat}, got {cycles[d] - cycles[s]}"
           for (s, d), lat in lk.edges.items() if cycles[d] < cycles[s] + lat]
    use = collections.Counter((cycles[o.index], o.engine) for o in lk.ops)
    bad += [f"cycle {c}: {n} {e} slots > {CAP[e]}" for (c, e), n in use.items() if n > CAP[e]]
    return bad


# =============================================================================
# Report
# =============================================================================


def simulator_cycles(bundles: list[dict]) -> int:
    """Cycles as the upstream Machine counts them: bundles with a non-debug engine."""
    return sum(1 for b in bundles if any(e != "debug" and s for e, s in b.items()))


def fill_profile(lk: LiftedKernel, cycles: list[int], buckets: int = 10) -> dict:
    """Per-cycle fill; start-up/tail = leading/trailing cycles whose vector-op-equivalent
    throughput (VALU + ALU/8) is below 90% of the joint peak of 7.5."""
    T = max(cycles) + 1 if cycles else 0
    per = [collections.Counter() for _ in range(T)]
    for o in lk.ops:
        per[cycles[o.index]][o.engine] += 1
    veq = [p["valu"] + p["alu"] / VLEN for p in per]
    full = [v >= 0.9 * (CAP["valu"] + CAP["alu"] / VLEN) for v in veq]
    lead = next((i for i, f in enumerate(full) if f), T)
    tail = next((i for i, f in enumerate(reversed(full)) if f), T)
    deciles = []
    for b in range(buckets):
        lo, hi = b * T // buckets, (b + 1) * T // buckets
        n = max(1, hi - lo)
        deciles.append({e: round(100 * sum(per[c][e] for c in range(lo, hi)) / (CAP[e] * n), 1) for e in ENGINES}
                       | {"veq_pct": round(100 * sum(veq[lo:hi]) / (7.5 * n), 1)})
    return {
        "cycles": T,
        "empty_cycles": sum(1 for p in per if not p),
        "underfilled_cycles": sum(1 for f in full if not f),
        "startup_cycles": lead,
        "tail_cycles": tail,
        "deciles": deciles,
    }


def report(lk: LiftedKernel, bundles: list[dict], name: str = "") -> dict:
    counts = collections.Counter(o.engine for o in lk.ops)
    opcodes = collections.Counter(f"{o.engine}.{o.slot[0]}" for o in lk.ops)
    vector = counts["valu"]
    non_shiftable = sum(1 for o in lk.ops if o.engine == "valu" and o.slot[0] in NON_SHIFTABLE)
    fixed = emission_floor(vector, counts["alu"], counts["load"], counts["flow"], counts["store"], max_shift=0)
    joint = emission_floor(vector, counts["alu"], counts["load"], counts["flow"], counts["store"],
                           max_shift=vector - non_shiftable)
    c = condense(lk)
    cp = max(heights(lk, c), default=0)
    cycles = [o.cycle for o in lk.ops]
    sim = simulator_cycles(bundles)
    s = lk.sym.stats
    return {
        "name": name,
        "ops": dict(sorted(counts.items())),
        "total_ops": lk.n,
        "opcodes": dict(sorted(opcodes.items())),
        "vector_op_equivalents": vector + counts["alu"] / VLEN,
        "engine_floor": {e: math.ceil(counts[e] / CAP[e]) for e in ENGINES},
        "floor_fixed_encoding": fixed["cycles"],
        "floor_fixed_binding": fixed["binding"],
        "floor_with_valu_shift": joint["cycles"],
        "critical_path": cp,
        "dataflow_critical_path": dataflow_critical_path(lk),
        "edges": len(lk.edges),
        "memory_edges": lk.mem_edges,
        "memory_alias": {
            "mem_ops": s.mem_ops, "op_pairs_with_store": s.op_pairs_total, "op_edges": s.op_edges,
            "z3_disjoint_word_pairs": s.disjoint_z3, "z3_may_alias_word_pairs": s.may_alias_z3,
            "must_alias_word_pairs": s.must_alias, "seconds": round(s.seconds, 1),
        },
        "simulator_cycles": sim,
        "gap_to_fixed_floor": sim - fixed["cycles"],
        "fill": fill_profile(lk, cycles),
        "co_issue_groups": sum(1 for m in c.members if len(m) > 1),
        "original_schedule_violations": len(check_schedule(lk, cycles)),
    }


def format_report(r: dict) -> str:
    f = r["fill"]
    m = r["memory_alias"]
    lines = [
        f"== {r['name']}: {r['simulator_cycles']} cycles on the simulator, {r['total_ops']} ops",
        "  ops:    " + "  ".join(f"{e}={r['ops'].get(e, 0)}" for e in ENGINES)
        + f"  (vector-op equivalents {r['vector_op_equivalents']:.0f})",
        "  floors: " + "  ".join(f"{e}={v}" for e, v in r["engine_floor"].items())
        + f"  -> fixed encoding {r['floor_fixed_encoding']} ({'+'.join(r['floor_fixed_binding'])}),"
        f" with VALU->ALU shift {r['floor_with_valu_shift']}",
        f"  critical path {r['critical_path']} cycles with these addresses, {r['dataflow_critical_path']} "
        f"through values only; gap to fixed floor {r['gap_to_fixed_floor']}"
        f" ({100 * r['gap_to_fixed_floor'] / max(1, r['floor_fixed_encoding']):.1f}%)",
        f"  graph: {r['edges']} edges ({r['memory_edges']} through memory: {m['mem_ops']} memory ops, "
        f"{m['op_pairs_with_store']} ordered pairs with a store, Z3 word pairs: {m['z3_disjoint_word_pairs']} disjoint, "
        f"{m['z3_may_alias_word_pairs']} may alias, {m['must_alias_word_pairs']} must alias)",
        f"  fill: {f['empty_cycles']} empty, {f['underfilled_cycles']} below 90% of VALU+ALU peak; "
        f"start-up {f['startup_cycles']}, tail {f['tail_cycles']} cycles",
        "  vector-op-equivalent % by decile: " + " ".join(f"{d['veq_pct']:.0f}" for d in f["deciles"]),
        "  load % by decile:                 " + " ".join(f"{d['load']:.0f}" for d in f["deciles"]),
    ]
    if "concrete_check" in r:
        c = r["concrete_check"]
        lines.append(f"  concrete check ({c['runs']} random inputs): {c['observed_alias_pairs']} aliasing pairs seen, "
                     f"{c['observed_without_edge']} without an edge, {c['edges_never_observed']} memory edges never seen, "
                     f"{c['addresses_outside_value_set']} addresses outside their value set, outputs ok={c['outputs_ok']}")
    if r["original_schedule_violations"]:
        lines.append(f"  WARNING: the kernel's own schedule violates {r['original_schedule_violations']} lifted constraints")
    return "\n".join(lines)


def table_row(r: dict) -> str:
    o = r["ops"]
    return (f"| {r['name']} | {o.get('valu', 0)} | {o.get('alu', 0)} | {o.get('load', 0)} | {o.get('store', 0)} "
            f"| {o.get('flow', 0)} | {r['vector_op_equivalents']:.0f} | {r['floor_fixed_encoding']} "
            f"| {r['floor_with_valu_shift']} | {r['critical_path']} | {r['simulator_cycles']} "
            f"| {r['gap_to_fixed_floor']} | {r['fill']['startup_cycles']} / {r['fill']['tail_cycles']} |")


TABLE_HEAD = ("| kernel | VALU | ALU | load | store | flow | vec-op eq. | floor (fixed) | floor (shift) "
              "| critical path | cycles | gap | start-up / tail |\n"
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("kernels", nargs="+", help="kernel JSON files; NAME=PATH sets the row name")
    ap.add_argument("--table", action="store_true", help="print one Markdown row per kernel")
    ap.add_argument("--json", help="write all reports to this JSON file")
    ap.add_argument("--concrete-runs", type=int, default=0,
                    help="also run each kernel on N random inputs and check every observed memory "
                         "alias has an edge (verification.memory_alias.concrete_check)")
    ap.add_argument("--save-graph-dir", help="write NAME.graph.json for each kernel here "
                                             "(input to scripts/reschedule_kernel.py)")
    args = ap.parse_args()
    reports = []
    for spec in args.kernels:
        name, _, path = spec.rpartition("=")
        name = name or os.path.basename(path)
        bundles = load_bundles(path)
        lk = lift(bundles)
        if args.save_graph_dir:
            save_graph(lk, os.path.join(args.save_graph_dir, f"{name}.graph.json"), name)
        r = report(lk, bundles, name)
        if args.concrete_runs:
            c = concrete_check(lk.sym, args.concrete_runs)
            r["concrete_check"] = {"runs": c.runs, "observed_alias_pairs": c.observed_pairs,
                                   "observed_without_edge": len(c.missing_edges),
                                   "edges_never_observed": c.unseen_edges,
                                   "addresses_outside_value_set": len(c.outside_value_set),
                                   "outputs_ok": c.outputs_ok}
        reports.append(r)
        if not args.table:
            print(format_report(r), flush=True)
        if args.json:  # rewritten after every kernel, so a long run keeps what it finished
            with open(args.json, "w") as f:
                json.dump(reports, f, indent=1)
    if args.table:
        print(TABLE_HEAD)
        for r in reports:
            print(table_row(r))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(reports, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
