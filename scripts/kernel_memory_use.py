#!/usr/bin/env python3
"""How a kernel uses main memory mid-kernel, and how much scratch it needs.

    uv run python -I scripts/kernel_memory_use.py NAME=kernel.json [...] [--json out.json]

Main memory is the only place this ISA addresses by a register value, so a kernel can
scatter by index (store to an address computed from data) and then read contiguously
(``vload``), or memoise values keyed by index. This tool classifies every store by
pattern, using the exact store-to-load data flow from ``verification.memory_alias``:

* address: constant, or data-dependent (the number of words it can reach and their region);
* value: what the stored word is a function of - constants, tree values (copied or
  selected), working values derived from the input values (hash state), or index/pointer
  values (finitely many possibilities);
* reader: which later loads read the stored words (``vload`` / ``load`` / ``load_offset``),
  how many distinct stores feed one load, and whether it reads the words back at all.

It also reports scratch use: distinct words touched, and the peak number of live words
(a word is live from the cycle after it is written until its last reader), for the
kernel's own schedule.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verification.memory_alias import SymbolicKernel, load_bundles  # noqa: E402

VLEN = 8


# =============================================================================
# Scratch liveness
# =============================================================================


def scratch_values(ops) -> list[tuple[int | None, int, list[int]]]:
    """(writer op or None for the initial zero, word, reader ops) for every scratch value."""
    current: dict[int, int] = {}  # word -> index into out
    out: list[tuple[int | None, int, list[int]]] = []
    by_cycle: dict[int, list] = collections.defaultdict(list)
    for op in ops:
        by_cycle[op.cycle].append(op)
    for cycle in sorted(by_cycle):
        group = by_cycle[cycle]
        for op in group:
            for a in op.reads:
                if a not in current:
                    current[a] = len(out)
                    out.append((None, a, []))
                out[current[a]][2].append(op.index)
        for op in group:
            for a in op.writes:
                current[a] = len(out)
                out.append((op.index, a, []))
    return out


def peak_live_words(values, start: list[int]) -> tuple[int, int]:
    """(peak, cycle of the peak) of words holding a value that is still to be read, under
    the schedule `start`. A word is occupied from the cycle after its write until its last
    read (a new write may land in the last reader's cycle). Initial zeros count from 0."""
    if not start:
        return 0, 0
    T = max(start) + 2
    diff = [0] * (T + 1)
    for writer, _, readers in values:
        if not readers:
            continue
        lo = 0 if writer is None else start[writer] + 1
        hi = max(start[r] for r in readers)
        if hi >= lo:
            diff[lo] += 1
            diff[hi + 1] -= 1
    best, at, cur = 0, 0, 0
    for t in range(T):
        cur += diff[t]
        if cur > best:
            best, at = cur, t
    return best, at


# =============================================================================
# Memory patterns
# =============================================================================


def _leaf_kinds(sym: SymbolicKernel):
    """Memoised: term -> frozenset of leaf kinds {const, tree, input, gathered, header, other}."""
    terms, layout = sym.terms, sym.layout
    memo: dict[int, frozenset] = {}

    def kinds(t: int) -> frozenset:
        stack = [t]
        while stack:
            cur = stack[-1]
            if cur in memo:
                stack.pop()
                continue
            term = terms.defs[cur]
            if term[0] == "const":
                memo[cur] = frozenset(["const"])
                stack.pop()
                continue
            if term[0] == "mem":
                a = terms.const_value(term[1])
                if a is None:
                    memo[cur] = frozenset(["gathered"])
                else:
                    region = layout.region(a)
                    memo[cur] = frozenset([{"tree": "tree", "values": "input"}.get(region, region)])
                stack.pop()
                continue
            kids = [k for k in term[1:] if k not in memo]
            if kids:
                stack.extend(kids)
                continue
            stack.pop()
            memo[cur] = frozenset().union(*(memo[k] for k in term[1:]))
        return memo[t]

    return kinds


def _value_class(sym, kinds, tids: list[int]) -> str:
    leaves = frozenset().union(*(kinds(t) for t in tids))
    heads = {sym.terms.defs[t][0] for t in tids}
    if leaves <= {"const"}:
        return "constant"
    finite = [sym.vsets.get(t) for t in tids]
    if all(f is not None for f in finite):
        n = len(frozenset().union(*finite))
        return f"index/pointer ({n} possible values)"
    if "input" in leaves:
        return "working value (derived from input values)"
    if heads & {"sel", "ite_eq"}:
        return "selected tree values"
    return "function of tree values (no input values)"


def _addr_class(sym, base: int, width: int) -> str:
    if sym.terms.const_value(base) is not None:
        a = sym.terms.const_value(base)
        return f"constant ({sym.layout.region(a)})"
    vs = sym.vsets.get(base)
    if vs is None:
        return "data-dependent (unbounded)"
    regions = sorted({sym.layout.region(v + k) for v in vs for k in range(width)})
    return f"data-dependent ({len(vs)} bases, {'+'.join(regions)})"


def _shape(sym, tid: int, depth: int = 3) -> str:
    return re.sub(r"0x[0-9a-f]+|\b\d+\b", "N", sym.terms.show(tid, depth))


def memory_use(sym: SymbolicKernel, examples: int = 1) -> dict:
    kinds = _leaf_kinds(sym)
    ops = sym.ops
    store_words: dict[int, list] = collections.defaultdict(list)  # store op -> events
    for e, ev in enumerate(sym.events):
        if ev.kind == "store":
            store_words[ev.op].append(e)
    readers: dict[int, set[int]] = collections.defaultdict(set)  # store op -> load ops
    feeders: dict[int, set[int]] = collections.defaultdict(set)  # load op -> store ops
    lanes_fed: dict[int, set[int]] = collections.defaultdict(set)
    for s_op, _, l_op, lane, _ in sym.flows:
        readers[s_op].add(l_op)
        feeders[l_op].add(s_op)
        lanes_fed[l_op].add(lane)

    groups: dict[tuple, dict] = {}
    for s_op, evs in sorted(store_words.items()):
        op = ops[s_op]
        values = [sym.events[e].value for e in evs]
        rd = sorted(readers.get(s_op, ()))
        reader_kinds = tuple(sorted({ops[r].slot[0] for r in rd})) or ("never read back",)
        key = (op.slot[0], _addr_class(sym, op.mem_addr, op.mem_width), _value_class(sym, kinds, values),
               reader_kinds)
        g = groups.setdefault(key, {"stores": 0, "store_cycles": [], "loads": set(), "examples": []})
        g["stores"] += 1
        g["store_cycles"].append(op.cycle)
        g["loads"].update(rd)
        if len(g["examples"]) < examples:
            ex = {"cycle": op.cycle, "slot": op.slot, "address": sym.terms.show(op.mem_addr, 6),
                  "address_shape": _shape(sym, op.mem_addr),
                  "value": sym.terms.show(values[0], 4)}
            if rd:
                r = ops[rd[0]]
                ex["reader"] = {"cycle": r.cycle, "slot": r.slot, "address": sym.terms.show(r.mem_addr, 6),
                                "stores_feeding_it": len(feeders[r.index]),
                                "lanes_from_stores": len(lanes_fed[r.index])}
            g["examples"].append(ex)
    out = []
    for (opcode, addr, value, rkinds), g in sorted(groups.items(), key=lambda kv: -kv[1]["stores"]):
        loads = sorted(g["loads"])
        multi = sum(1 for l in loads if len(feeders[l]) > 1)
        out.append({
            "store": opcode, "address": addr, "value": value, "readers": list(rkinds),
            "stores": g["stores"], "store_cycles": [min(g["store_cycles"]), max(g["store_cycles"])],
            "loads": len(loads), "load_cycles": [ops[loads[0]].cycle, ops[loads[-1]].cycle] if loads else None,
            "loads_fed_by_several_stores": multi,
            "examples": g["examples"],
        })
    return {"patterns": out}


def scratch_report(sym: SymbolicKernel) -> dict:
    vals = scratch_values(sym.ops)
    touched = {a for o in sym.ops for a in o.reads + o.writes}
    peak, at = peak_live_words(vals, [o.cycle for o in sym.ops])
    # Words held in memory between a store and its last reader (the memory round trips).
    last_read: dict[int, int] = {}
    for s_op, e, l_op, _, _ in sym.flows:
        last_read[e] = max(last_read.get(e, -1), sym.ops[l_op].cycle)
    diff = collections.Counter()
    for e, hi in last_read.items():
        lo = sym.ops[sym.events[e].op].cycle + 1
        if hi >= lo:
            diff[lo] += 1
            diff[hi + 1] -= 1
    mem_peak, cur = 0, 0
    for t in range(len(sym.bundles) + 2):
        cur += diff[t]
        mem_peak = max(mem_peak, cur)
    return {"words_touched": len(touched), "highest_word": max(touched) if touched else None,
            "peak_live_words": peak, "peak_cycle": at, "peak_words_parked_in_memory": mem_peak}


def format_memory_use(name: str, mu: dict, sr: dict) -> str:
    lines = [f"== {name}: scratch words touched {sr['words_touched']} (highest {sr['highest_word']}), "
             f"peak live {sr['peak_live_words']} at cycle {sr['peak_cycle']}; "
             f"peak words parked in memory {sr['peak_words_parked_in_memory']}"]
    for p in mu["patterns"]:
        lines.append(f"  {p['stores']:5d} x {p['store']:6s} addr {p['address']}; value {p['value']}; "
                     f"read by {'/'.join(p['readers'])}: {p['loads']} loads "
                     f"({p['loads_fed_by_several_stores']} fed by several stores); "
                     f"store cycles {p['store_cycles'][0]}..{p['store_cycles'][1]}"
                     + (f", load cycles {p['load_cycles'][0]}..{p['load_cycles'][1]}" if p["load_cycles"] else ""))
        for ex in p["examples"]:
            lines.append(f"        e.g. cycle {ex['cycle']} {ex['slot']}: mem[{ex['address']}] = {ex['value']}")
            if "reader" in ex:
                r = ex["reader"]
                lines.append(f"             read at cycle {r['cycle']} by {r['slot']} from {r['address']} "
                             f"({r['lanes_from_stores']} lanes from {r['stores_feeding_it']} stores)")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("kernels", nargs="+", help="NAME=PATH kernel JSON files")
    ap.add_argument("--json", help="write the reports here")
    args = ap.parse_args()
    out = {}
    for spec in args.kernels:
        name, _, path = spec.rpartition("=")
        name = name or os.path.basename(path)
        sym = SymbolicKernel(load_bundles(path)).run()
        out[name] = {"memory_use": memory_use(sym), "scratch": scratch_report(sym)}
        print(format_memory_use(name, out[name]["memory_use"], out[name]["scratch"]), flush=True)
        if args.json:
            with open(args.json, "w") as f:
                json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
