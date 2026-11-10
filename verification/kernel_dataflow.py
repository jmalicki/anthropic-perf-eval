#!/usr/bin/env python3
"""Dataflow replay: prove an emitted kernel computes exactly what its SSA program computes.

The kernel is executed symbolically. Every scratch word holds a hash-consed term instead of
a number, e.g. ``^(mem[+(p, idx)], c)``. The same terms are built from the SSA ops (no
cycles, no addresses, no liveness). The kernel is correct with respect to the SSA program,
for every input, iff both store the same terms to the same memory addresses.

This is independent of the scheduler's own metadata: it uses only the SSA ops (opcode,
srcs, dest), the vector-group lane map, and the emitted instruction bundles. It therefore
catches wrong cycles, wrong bundles, premature address reuse, wrong operand addresses and
same-cycle write conflicts, and names the first place the kernel computes a value the SSA
program never does.

What it does not prove: that the SSA program equals the reference kernel (run --validate
for that), and that select conditions lowered to multiply_add are 0/1 (those are checked
structurally and listed if not recognisably boolean).

Usage:
    uv run generate_ssa_kernel.py --rounds 2 --use-greedy --output-json sched.json --save-kernel kern.json
    uv run python -m verification.kernel_dataflow --kernel kern.json --scheduler sched.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field

VLEN = 8
MASK = 2**32
COMMUTATIVE = {"+", "*", "^", "&", "|", "=="}


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


_FOLD = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "//": lambda a, b: a // b,
    "cdiv": _cdiv,
    "^": lambda a, b: a ^ b,
    "&": lambda a, b: a & b,
    "|": lambda a, b: a | b,
    "<<": lambda a, b: a << b,
    ">>": lambda a, b: a >> b,
    "%": lambda a, b: a % b,
    "<": lambda a, b: int(a < b),
    "==": lambda a, b: int(a == b),
}


class Terms:
    """Hash-consed term store shared by the SSA side and the kernel side.

    Both sides must build terms through the same constructors so that every normalisation
    (constant folding, commutative ordering, select lowering) applies identically.
    """

    def __init__(self) -> None:
        self._ids: dict[tuple, int] = {}
        self.defs: list[tuple] = []
        # Conditions of selects that were recovered from c*(t-f)+f and are not provably 0/1.
        self.unproven_bool_conds: set[int] = set()

    def _intern(self, term: tuple) -> int:
        tid = self._ids.get(term)
        if tid is None:
            tid = len(self.defs)
            self._ids[term] = tid
            self.defs.append(term)
        return tid

    def const(self, value: int) -> int:
        return self._intern(("const", value % MASK))

    def const_value(self, tid: int) -> int | None:
        term = self.defs[tid]
        return term[1] if term[0] == "const" else None

    def binop(self, op: str, a: int, b: int) -> int:
        if op not in _FOLD:
            raise ValueError(f"unknown binary op {op!r}")
        ca, cb = self.const_value(a), self.const_value(b)
        if ca is not None and cb is not None and not (op in ("//", "cdiv", "%") and cb == 0):
            return self.const(_FOLD[op](ca, cb))
        # (x - y) + y = x and (x + y) - y = x, modulo 2^32. Emission precomputes select
        # deltas as t - f and rebuilds t as f + delta; both spellings must be one term.
        if op == "+":
            for diff, other in ((a, b), (b, a)):
                dterm = self.defs[diff]
                if dterm[0] == "-" and dterm[2] == other:
                    return dterm[1]
        if op == "-":
            aterm = self.defs[a]
            if aterm[0] == "+" and b in aterm[1:]:
                return aterm[2] if aterm[1] == b else aterm[1]
        if op in COMMUTATIVE and b < a:
            a, b = b, a
        return self._intern((op, a, b))

    def is_bool(self, tid: int) -> bool:
        term = self.defs[tid]
        if term[0] == "const":
            return term[1] in (0, 1)
        if term[0] in ("==", "<"):
            return True
        if term[0] == "&":
            return self.const_value(term[1]) == 1 or self.const_value(term[2]) == 1
        if term[0] == "sel":
            return self.is_bool(term[2]) and self.is_bool(term[3])
        return False

    def select(self, cond: int, if_true: int, if_false: int) -> int:
        return self._intern(("sel", cond, if_true, if_false))

    def fma(self, a: int, b: int, c: int) -> int:
        """a * b + c, with the SelectChoice FMA lowering recognised.

        ``cond * (t - f) + f`` is how a select is lowered onto the vector ALU. It equals
        ``select(cond, t, f)`` when cond is 0 or 1, so it is normalised to the select term.
        """
        ca, cb, cc = self.const_value(a), self.const_value(b), self.const_value(c)
        if ca is not None and cb is not None and cc is not None:
            return self.const(ca * cb + cc)
        for cond, delta in ((a, b), (b, a)):
            dterm = self.defs[delta]
            if dterm[0] == "-" and dterm[2] == c:
                if not self.is_bool(cond):
                    self.unproven_bool_conds.add(cond)
                return self.select(cond, dterm[1], c)
        if b < a:
            a, b = b, a
        return self._intern(("fma", a, b, c))

    def mem(self, addr: int) -> int:
        return self._intern(("mem", addr))

    def show(self, tid: int, depth: int = 3) -> str:
        term = self.defs[tid]
        if term[0] == "const":
            return f"{term[1]:#x}" if term[1] > 9 else str(term[1])
        if depth == 0:
            return "…"
        return f"{term[0]}(" + ", ".join(self.show(t, depth - 1) for t in term[1:]) + ")"


# =============================================================================
# SSA side
# =============================================================================


@dataclass
class SsaValues:
    """Terms the SSA program computes, and the memory writes it performs."""

    names: dict[int, str] = field(default_factory=dict)  # term id -> "v123[lane] (where)"
    stores: dict[int, tuple[int, str]] = field(default_factory=dict)  # addr term -> (value term, where)
    undefined_vregs: list[int] = field(default_factory=list)


def _where(op: dict) -> str:
    info = op.get("source_info") or {}
    parts = [f"op {op['id']} {op['opcode']}"]
    for key in ("round", "phase", "group"):
        if info.get(key) is not None:
            parts.append(f"{key}={info[key]}")
    return " ".join(parts)


def ssa_values(sched: dict, terms: Terms) -> SsaValues:
    """Build the term of every SSA value, lane by lane."""
    ops = sched["ssa_ops"]
    out = SsaValues()
    def_op: dict[int, dict] = {}
    for op in ops:
        if op.get("dest"):
            vid = int(op["dest"][1:].split("_")[0])
            # ConstChoice may list an AddImm alternative with the same dest; first wins.
            def_op.setdefault(vid, op)

    if "vector_group_lanes" not in sched or "ssa_ops" not in sched:
        raise ValueError("scheduler JSON lacks ssa_ops/vector_group_lanes; regenerate it with the "
                         "current scheduler_rs (--output-json)")
    groups = {int(k): v for k, v in sched["vector_group_lanes"].items()}
    lane_of: dict[int, tuple[int, int]] = {}
    for group, lanes in groups.items():
        if group in def_op:  # read group: lanes are slices of a defined vector
            for k, lane in enumerate(lanes):
                lane_of[lane] = (group, k)

    cache: dict[int, list[int]] = {}

    def operand(src) -> list[int]:
        if "Imm" in src:
            return [terms.const(src["Imm"])]
        return value(src["VReg"]["id"])

    def lanes_of(src, n: int) -> list[int]:
        tids = operand(src)
        return tids if len(tids) == n else tids * n

    def value(vid: int) -> list[int]:
        if vid in cache:
            return cache[vid]
        if vid in lane_of:
            group, k = lane_of[vid]
            result = [value(group)[k]]
        elif vid in def_op:
            result = compute(def_op[vid])
            where = _where(def_op[vid])
            for k, tid in enumerate(result):
                out.names.setdefault(tid, f"v{vid}[{k}] ({where})" if len(result) > 1 else f"v{vid} ({where})")
        elif vid in groups:  # write group: assembled from scalar lanes
            result = [value(lane)[0] for lane in groups[vid]]
        else:  # reserved zero VReg: scratch[0], never written
            out.undefined_vregs.append(vid)
            result = [terms.const(0)]
        cache[vid] = result
        return result

    def compute(op: dict) -> list[int]:
        opcode, srcs = op["opcode"], op["srcs"]
        width = VLEN if op["dest"].endswith("_v") else 1
        if opcode in ("const_choice", "const"):
            return [terms.const(srcs[0]["Imm"])]
        if opcode == "vbroadcast":
            return operand(srcs[0]) * VLEN
        if opcode == "load":
            return [terms.mem(operand(srcs[0])[0])]
        if opcode == "vload":
            base = operand(srcs[0])[0]
            return [terms.mem(terms.binop("+", base, terms.const(k))) for k in range(VLEN)]
        if opcode == "add_imm":
            return [terms.binop("+", operand(srcs[0])[0], operand(srcs[1])[0])]
        if opcode == "multiply_add":
            a, b, c = (lanes_of(s, width) for s in srcs)
            return [terms.fma(a[k], b[k], c[k]) for k in range(width)]
        if opcode in ("select_choice", "vselect", "select"):
            c, t, f = (lanes_of(s, width) for s in srcs)
            return [terms.select(c[k], t[k], f[k]) for k in range(width)]
        if opcode in _FOLD:
            a, b = (lanes_of(s, width) for s in srcs)
            return [terms.binop(opcode, a[k], b[k]) for k in range(width)]
        raise ValueError(f"SSA opcode {opcode!r} not modelled (op {op['id']})")

    sys.setrecursionlimit(max(sys.getrecursionlimit(), 100_000))
    for op in ops:
        if op["cycle"] is not None and op["cycle"] < 0:
            continue  # dead op, never emitted
        if op["opcode"] == "vstore":
            base = operand(op["srcs"][0])[0]
            vals = operand(op["srcs"][1])
            for k in range(VLEN):
                out.stores[terms.binop("+", base, terms.const(k))] = (vals[k], _where(op))
        elif op["opcode"] == "store":
            out.stores[operand(op["srcs"][0])[0]] = (operand(op["srcs"][1])[0], _where(op))
        elif op.get("dest"):
            value(int(op["dest"][1:].split("_")[0]))
    # Only the reserved zero VReg may be read without a definition.
    if len(set(out.undefined_vregs)) > 1:
        raise ValueError(f"SSA program reads {len(set(out.undefined_vregs))} undefined VRegs "
                         f"(e.g. {sorted(set(out.undefined_vregs))[:5]}); expected only the zero VReg")
    return out


# =============================================================================
# Kernel side
# =============================================================================


@dataclass
class Finding:
    kind: str
    cycle: int
    message: str


@dataclass
class Report:
    findings: list[Finding]
    cycles: int
    kernel_ops: int
    ssa_values: int
    unproven_bool_conds: list[str]

    @property
    def ok(self) -> bool:
        return not self.findings


def _execute(terms: Terms, read, engine: str, slot: list) -> tuple[list[int], list[int], list[list[int]]] | None:
    """Symbolically execute one slot against the pre-cycle state.

    Returns (written addresses, result terms, operand addresses per result), or None for
    slots with no scratch effect (stores are handled by the caller; pause/halt do nothing).
    Entry k of the third list holds, in operand order, the addresses read to produce
    result k, so a diagnostic can name the exact lane that went wrong.
    """
    name = slot[0]
    lanes = range(VLEN)
    if engine == "alu":
        _, d, a, b = slot
        return [d], [terms.binop(name, read(a), read(b))], [[a, b]]
    if engine == "valu":
        d = slot[1]
        dests = [d + k for k in lanes]
        if name == "vbroadcast":
            return dests, [read(slot[2])] * VLEN, [[slot[2]]] * VLEN
        if name == "multiply_add":
            _, _, a, b, c = slot
            return (dests, [terms.fma(read(a + k), read(b + k), read(c + k)) for k in lanes],
                    [[a + k, b + k, c + k] for k in lanes])
        _, _, a, b = slot
        return dests, [terms.binop(name, read(a + k), read(b + k)) for k in lanes], [[a + k, b + k] for k in lanes]
    if engine == "load":
        d = slot[1]
        if name == "const":
            return [d], [terms.const(slot[2])], [[]]
        if name == "load":
            return [d], [terms.mem(read(slot[2]))], [[slot[2]]]
        if name == "load_offset":
            return [d + slot[3]], [terms.mem(read(slot[2] + slot[3]))], [[slot[2] + slot[3]]]
        if name == "vload":
            base = read(slot[2])
            return ([d + k for k in lanes],
                    [terms.mem(terms.binop("+", base, terms.const(k))) for k in lanes], [[slot[2]]] * VLEN)
    if engine == "flow":
        if name in ("pause", "halt"):
            return None
        d = slot[1]
        if name == "add_imm":
            return [d], [terms.binop("+", read(slot[2]), terms.const(slot[3]))], [[slot[2]]]
        if name == "select":
            _, _, c, a, b = slot
            return [d], [terms.select(read(c), read(a), read(b))], [[c, a, b]]
        if name == "vselect":
            _, _, c, a, b = slot
            return ([d + k for k in lanes],
                    [terms.select(read(c + k), read(a + k), read(b + k)) for k in lanes],
                    [[c + k, a + k, b + k] for k in lanes])
    raise ValueError(f"{engine} slot {slot} not modelled (the kernel must be straight-line)")


class _Replay:
    """Kernel state during symbolic execution: a term and a last writer per scratch word."""

    def __init__(self, terms: Terms, ssa: SsaValues) -> None:
        self.terms = terms
        self.ssa = ssa
        self.zero = terms.const(0)
        self.scratch: dict[int, int] = {}  # absent = never written = 0
        self.writer: dict[int, tuple[int, str, list]] = {}

    def read(self, addr: int) -> int:
        return self.scratch.get(addr, self.zero)

    def name(self, tid: int) -> str:
        if tid in self.ssa.names:
            return self.ssa.names[tid]
        kind = "constant" if self.terms.const_value(tid) is not None else "non-SSA value"
        return f"{kind} {self.terms.show(tid)}"

    def describe(self, addr: int) -> str:
        what = self.name(self.read(addr))
        if addr in self.writer:
            cycle, engine, slot = self.writer[addr]
            return f"s[{addr}] = {what}  [written at cycle {cycle} by {engine} {slot}]"
        return f"s[{addr}] = {what}  [never written]"

    def commit(self, cycle: int, writes: list[tuple[int, int, str, list]]) -> list[Finding]:
        """Apply end-of-cycle writes; two different slots writing one address conflict."""
        findings = []
        seen: dict[int, tuple[str, list]] = {}
        for addr, tid, engine, slot in writes:
            if addr in seen and seen[addr][1] is not slot:
                findings.append(Finding(
                    "write-conflict", cycle,
                    f"cycle {cycle}: s[{addr}] is written twice in one cycle, by {seen[addr][0]} "
                    f"{seen[addr][1]} and {engine} {slot}"))
            seen[addr] = (engine, slot)
            self.scratch[addr] = tid
            self.writer[addr] = (cycle, engine, slot)
        return findings


def _near_miss(terms: Terms, ssa: SsaValues, tid: int) -> str:
    """If an SSA value differs from `tid` in exactly one operand, say which operand."""
    term = terms.defs[tid]
    for cand, name in ssa.names.items():
        other = terms.defs[cand]
        if other[0] != term[0] or len(other) != len(term):
            continue
        diff = [i for i in range(1, len(term)) if other[i] != term[i]]
        if len(diff) == 1:
            want = ssa.names.get(other[diff[0]]) or terms.show(other[diff[0]])
            return f" Closest SSA value is {name}: it differs only in operand {diff[0]}, which should be {want}."
    return ""


def check(kernel: dict | list, sched: dict) -> Report:
    """Replay the kernel symbolically and compare with the SSA program.

    # Arguments
    * `kernel` - Kernel JSON from --save-kernel (or a bare list of bundles)
    * `sched` - Scheduler JSON from --output-json (needs `ssa_ops`, `vector_group_lanes`)

    # Returns
    A Report whose findings are ordered by cycle; the first is the earliest point at which
    the kernel departs from the SSA program.
    """
    instrs = kernel["instrs"] if isinstance(kernel, dict) else kernel
    terms = Terms()
    ssa = ssa_values(sched, terms)
    state = _Replay(terms, ssa)

    # Non-SSA terms computed by the kernel and not yet shown to be lowering temporaries
    # (a temporary is consumed by an op that produces an SSA value).
    foreign: dict[int, tuple[int, str, list, str, list[str]]] = {}
    kernel_stores: dict[int, tuple[int, int, list]] = {}
    findings: list[Finding] = []
    n_ops = 0

    for cycle, bundle in enumerate(instrs):
        writes: list[tuple[int, int, str, list]] = []
        for engine, slots in bundle.items():
            if engine == "debug":
                continue
            for slot in slots:
                n_ops += 1
                if engine == "store":
                    _, addr, src = slot
                    base = state.read(addr)
                    for k in range(VLEN if slot[0] == "vstore" else 1):
                        target = terms.binop("+", base, terms.const(k)) if slot[0] == "vstore" else base
                        kernel_stores[target] = (state.read(src + k), cycle, slot)
                    continue
                effect = _execute(terms, state.read, engine, slot)
                if effect is None:
                    continue
                dests, results, operands = effect
                for lane, (dest, tid) in enumerate(zip(dests, results)):
                    writes.append((dest, tid, engine, slot))
                    if tid in ssa.names or terms.const_value(tid) is not None:
                        for addr in operands[lane]:
                            foreign.pop(state.read(addr), None)
                    elif tid not in foreign:
                        where = f" lane {lane}" if len(dests) > 1 else ""
                        foreign[tid] = (cycle, engine, slot, where, [state.describe(a) for a in operands[lane]])
        findings += state.commit(cycle, writes)

    for tid, (cycle, engine, slot, where, held) in foreign.items():
        findings.append(Finding(
            "foreign-value", cycle,
            f"cycle {cycle}: {engine} {slot}{where} computes a value no SSA op computes: {terms.show(tid)}."
            + _near_miss(terms, ssa, tid)
            + "".join(f"\n      operand {i + 1}: {h}" for i, h in enumerate(held))))

    for target, (want, where) in ssa.stores.items():
        got = kernel_stores.get(target)
        if got is None:
            findings.append(Finding("missing-store", len(instrs),
                                    f"SSA {where} stores to mem[{terms.show(target)}]; the kernel never does"))
        elif got[0] != want:
            findings.append(Finding(
                "wrong-store", got[1],
                f"cycle {got[1]}: {got[2]} stores {state.name(got[0])} to mem[{terms.show(target)}]; "
                f"SSA {where} stores {state.name(want)}"))
    for target, (_, cycle, slot) in kernel_stores.items():
        if target not in ssa.stores:
            findings.append(Finding("extra-store", cycle,
                                    f"cycle {cycle}: {slot} stores to mem[{terms.show(target)}]; no SSA op does"))

    findings.sort(key=lambda f: f.cycle)
    return Report(
        findings=findings,
        cycles=len(instrs),
        kernel_ops=n_ops,
        ssa_values=len(ssa.names),
        unproven_bool_conds=sorted(terms.show(t) for t in terms.unproven_bool_conds),
    )


def explain_cycle(kernel: dict | list, sched: dict, cycle: int) -> list[str]:
    """Describe what each operand of each slot in `cycle` held when the cycle began."""
    instrs = kernel["instrs"] if isinstance(kernel, dict) else kernel
    terms = Terms()
    state = _Replay(terms, ssa_values(sched, terms))
    for c, bundle in enumerate(instrs[:cycle]):
        writes = []
        for engine, slots in bundle.items():
            if engine in ("debug", "store"):
                continue
            for slot in slots:
                effect = _execute(terms, state.read, engine, slot)
                if effect is not None:
                    writes += [(dest, tid, engine, slot) for dest, tid in zip(effect[0], effect[1])]
        state.commit(c, writes)
    lines = []
    for engine, slots in instrs[cycle].items():
        if engine == "debug":
            continue
        for slot in slots:
            lines.append(f"{engine} {slot}")
            if engine == "store":
                width = VLEN if slot[0] == "vstore" else 1
                per_lane = [[slot[1], slot[2] + k] for k in range(width)]
            else:
                effect = _execute(terms, state.read, engine, slot)
                per_lane = effect[2] if effect is not None else []
                if effect is not None:
                    # Name the SSA value this slot produces, so a slot can be traced to its op.
                    made = [state.name(tid) for tid in effect[1]]
                    lines.append(f"    -> {made[0]}" + (f" … {made[-1]}" if len(made) > 1 and made[-1] != made[0] else ""))
            shown: set[int] = set()
            for lane, addrs in enumerate(per_lane):
                for addr in addrs:
                    if addr not in shown:
                        shown.add(addr)
                        lines.append(f"    {state.describe(addr)}")
    return lines


def format_report(report: Report, limit: int = 5) -> str:
    head = (f"dataflow replay: {report.kernel_ops} kernel ops over {report.cycles} cycles "
            f"against {report.ssa_values} SSA values")
    if report.ok:
        lines = [head, "  PROVED - every store matches the SSA program for all inputs"]
    else:
        lines = [head, f"  FAILED - {len(report.findings)} finding(s); earliest first:"]
        lines += [f"  [{f.kind}] {f.message}" for f in report.findings[:limit]]
        if len(report.findings) > limit:
            lines.append(f"  … {len(report.findings) - limit} more (later findings are usually consequences)")
    if report.unproven_bool_conds:
        lines.append(f"  note: {len(report.unproven_bool_conds)} select condition(s) lowered to multiply_add "
                     f"are not structurally 0/1, e.g. {report.unproven_bool_conds[0]}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--kernel", required=True, help="kernel JSON from --save-kernel")
    ap.add_argument("--scheduler", required=True, help="scheduler JSON from --output-json")
    ap.add_argument("--explain-cycle", type=int, help="show what each operand address held before this cycle")
    ap.add_argument("--limit", type=int, default=5, help="max findings to print")
    args = ap.parse_args()
    with open(args.kernel) as f:
        kernel = json.load(f)
    with open(args.scheduler) as f:
        sched = json.load(f)
    if args.explain_cycle is not None:
        print("\n".join(explain_cycle(kernel, sched, args.explain_cycle)))
        return 0
    report = check(kernel, sched)
    print(format_report(report, args.limit))
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
