#!/usr/bin/env python3
"""Lane-aware inspection of a scheduler output JSON: addresses, SSA ops and kernel slots.

One tool for the questions that come up when a kernel reads the wrong value:

  addr A [--cycle C]   Every value whose address range covers A (a vector at 27..34 covers
                       30), with its live interval, defining SSA op and source info. With
                       --cycle: which value holds A at C, the last write to A before C, and
                       the lowered ops that read or write A in C.
  op ID                An SSA op: operands (address, interval, definer), its lowered ops with
                       cycles (the 8 scalar lanes of a shifted op; the subtract and
                       multiply_add of a select lowered to the vector ALU), and its consumers.
  slot CYCLE DEST      Which SSA op the kernel slot writing DEST in CYCLE belongs to.
  value NAME           A value (v123_v, v45_s or a bare id): address, interval, definer, readers.

Run from the project root (scripts/bisect.py shadows the stdlib `bisect` if a script is run
from inside scripts/):

  uv run generate_ssa_kernel.py --rounds 3 --batch-size 16 --use-greedy -q \
      --output-json /tmp/s.json --save-kernel /tmp/k.json
  uv run python -m scripts.inspect_schedule /tmp/s.json addr 30 --cycle 21
  uv run python -m scripts.inspect_schedule /tmp/s.json op 123
  uv run python -m scripts.inspect_schedule /tmp/s.json --kernel /tmp/k.json slot 28 117

Cycles are schedule cycles (the `cycle` fields of the JSON). The kernel has one leading
`pause` bundle, so kernel bundle index = schedule cycle + 1; pass --bundle to give CYCLE as a
bundle index (what `verification.kernel_dataflow` reports).

Intervals are [def, last_use]: the value is written at the end of cycle `def` and last read
in cycle `last_use`, so it occupies its address for reads in cycles def+1 .. last_use.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field

VLEN = 8
# Lowered-op id conventions (scheduler_rs/src/kernel_builder/ssa_lower.rs, types.rs SsaId).
SHIFT_LANE_FACTOR = 1000  # shifted VALU op: ssa_id * 1000 + lane
SELECT_FMA_OFFSET = 1_000_000  # multiply_add of a select lowered to the vector ALU
CSE_SHARED_SUB_PREFIX = 0xC5E0_0000  # shared subtract of a CSE group of selects
KERNEL_LEADING_BUNDLES = 1  # the kernel starts with one pause bundle

_VREG_KEY = re.compile(r"id:\s*(\d+),\s*width:\s*(\w+)")
_VALUE_NAME = re.compile(r"^v?(\d+)(?:_([vs]))?$")


def value_name(vid: int, width: str) -> str:
    return f"v{vid}_{'v' if width.lower() == 'vector' else 's'}"


@dataclass
class Value:
    """One SSA value (VReg) with its scratch placement."""

    name: str
    vid: int
    width: str  # "scalar" | "vector"
    size: int
    addr: int | None
    start: int | None  # cycle of the defining op
    end: int | None  # cycle of the last reader
    definer: int | None = None  # SSA op id
    lane_of: tuple[str, int] | None = None  # (group value name, lane) for a vector-group lane
    lanes: list[str] = field(default_factory=list)  # lane value names for a vector group
    note: str = ""

    def covers(self, addr: int) -> bool:
        return self.addr is not None and self.addr <= addr < self.addr + self.size

    def words(self) -> str:
        if self.addr is None:
            return "no address"
        if self.size == 1:
            return f"address {self.addr}"
        return f"addresses {self.addr}..{self.addr + self.size - 1}"

    def interval(self) -> str:
        if self.start is None or self.end is None:
            return "interval unknown"
        return f"interval [{self.start}, {self.end}]"

    def holds_at(self, cycle: int) -> bool:
        """True if a read in `cycle` sees this value (written before, not yet dead)."""
        return self.start is not None and self.end is not None and self.start < cycle <= self.end


def slot_ints(slot: list) -> list:
    """Scheduler JSON slots are lists of strings; kernel slots are ints. Normalise to ints."""
    out = [slot[0]]
    for x in slot[1:]:
        try:
            out.append(int(x))
        except (TypeError, ValueError):
            out.append(x)
    return out


def slot_effects(engine: str, slot: list) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """(writes, reads) of a slot as (address, words) ranges, lane-exact.

    Mirrors verification.kernel_dataflow._execute. Stores write memory, not scratch: their
    pointer and source operands are reads.
    """
    s = slot_ints(slot)
    name = s[0]
    if engine == "alu":
        return [(s[1], 1)], [(s[2], 1), (s[3], 1)]
    if engine == "valu":
        if name == "vbroadcast":
            return [(s[1], VLEN)], [(s[2], 1)]
        return [(s[1], VLEN)], [(a, VLEN) for a in s[2:]]
    if engine == "load":
        if name == "const":
            return [(s[1], 1)], []
        if name == "load":
            return [(s[1], 1)], [(s[2], 1)]
        if name == "load_offset":
            return [(s[1] + s[3], 1)], [(s[2] + s[3], 1)]
        if name == "vload":
            return [(s[1], VLEN)], [(s[2], 1)]
    if engine == "flow":
        if name == "add_imm":
            return [(s[1], 1)], [(s[2], 1)]
        if name == "select":
            return [(s[1], 1)], [(a, 1) for a in s[2:5]]
        if name == "vselect":
            return [(s[1], VLEN)], [(a, VLEN) for a in s[2:5]]
        return [], []
    if engine == "store":
        return [], [(s[1], 1), (s[2], VLEN if name == "vstore" else 1)]
    return [], []


def _covers(ranges: list[tuple[int, int]], addr: int) -> bool:
    return any(a <= addr < a + n for a, n in ranges)


class Schedule:
    """Indexes a scheduler output JSON for the queries below."""

    def __init__(self, sched: dict, kernel: dict | list | None = None) -> None:
        self.sched = sched
        self.ssa: dict[int, dict] = {op["id"]: op for op in sched.get("ssa_ops") or []}
        self.lowered: list[dict] = sorted(sched.get("ops") or [], key=lambda o: (o["cycle"], o["id"]))
        self.instrs = None
        if kernel is not None:
            self.instrs = kernel["instrs"] if isinstance(kernel, dict) else kernel
        self.values: dict[str, Value] = {}
        self._build_values()
        self.readers: dict[str, list[int]] = {}
        for op in self.ssa.values():
            for name in self.operand_names(op):
                self.readers.setdefault(name, []).append(op["id"])
        self.lowered_of: dict[int, list[tuple[dict, str]]] = {}
        self.owner: dict[int, tuple[int | None, str]] = {}  # index in self.lowered -> (ssa id, role)
        for i, low in enumerate(self.lowered):
            ssa_id, role = self._owner_of(low)
            self.owner[i] = (ssa_id, role)
            if ssa_id is not None:
                self.lowered_of.setdefault(ssa_id, []).append((low, role))

    # ---- values -----------------------------------------------------------------------

    def _build_values(self) -> None:
        vregs = self.sched.get("vregs")
        if vregs:
            for v in vregs:
                name = value_name(v["id"], v["width"])
                self.values[name] = Value(name, v["id"], v["width"].lower(), v.get("size_words") or
                                          (VLEN if v["width"].lower() == "vector" else 1),
                                          v.get("address"), v.get("interval_start"), v.get("interval_end"))
        else:  # --liveness-detail none/summary: addresses only
            for key, addr in (self.sched.get("vreg_to_addr") or {}).items():
                m = _VREG_KEY.search(key)
                if not m:
                    continue
                vid, width = int(m.group(1)), m.group(2).lower()
                name = value_name(vid, width)
                self.values[name] = Value(name, vid, width, VLEN if width == "vector" else 1, addr, None, None)

        for op in self.ssa.values():
            dest = op.get("dest")
            if not dest:
                continue
            if dest not in self.values:
                m = _VALUE_NAME.match(dest)
                width = "vector" if dest.endswith("_v") else "scalar"
                self.values[dest] = Value(dest, int(m.group(1)) if m else -1, width,
                                          VLEN if width == "vector" else 1, None, None, None,
                                          note="not in the JSON's vregs (eliminated or never assigned)")
            self.values[dest].definer = op["id"]

        # Vector groups: 8 scalar lanes alias the 8 words of a vector. A decomposed vector is
        # defined by an op and its lanes are read; an assembled vector has no defining op and
        # each lane is written by its own scalar op. Lanes have no entry of their own in
        # `vregs`: lane k lives at the group's address + k.
        for gid, lane_ids in (self.sched.get("vector_group_lanes") or {}).items():
            gname = value_name(int(gid), "vector")
            group = self.values.get(gname)
            if group is None:
                group = self.values[gname] = Value(gname, int(gid), "vector", VLEN, None, None, None,
                                                   note="vector group without an entry in vregs")
            assembled = group.definer is None
            group.note = ("vector group assembled from 8 scalar lanes" if assembled
                          else "vector group decomposed into 8 scalar lanes")
            for k, lid in enumerate(lane_ids):
                lname = value_name(lid, "scalar")
                lane = self.values.get(lname)
                if lane is None:
                    lane = self.values[lname] = Value(lname, lid, "scalar", 1, None, None, None)
                lane.lane_of = (gname, k)
                lane.note = ""  # a lane is placed through its group, not through vregs
                if lane.addr is None and group.addr is not None:
                    lane.addr = group.addr + k
                if lane.end is None:
                    lane.end = group.end
                if lane.start is None:
                    own = self.ssa.get(lane.definer) if lane.definer is not None else None
                    lane.start = own["cycle"] if own is not None else group.start
                if lane.definer is None and group.definer is not None:
                    lane.definer = group.definer
                group.lanes.append(lname)

    def find_value(self, text: str) -> Value | None:
        m = _VALUE_NAME.match(text.strip())
        if not m:
            return None
        vid, suffix = int(m.group(1)), m.group(2)
        for name in ([f"v{vid}_{suffix}"] if suffix else [f"v{vid}_v", f"v{vid}_s"]):
            if name in self.values:
                return self.values[name]
        return None

    @staticmethod
    def operand_names(op: dict) -> list[str]:
        names = []
        for src in op.get("srcs") or []:
            if isinstance(src, dict) and "VReg" in src:
                names.append(value_name(src["VReg"]["id"], src["VReg"]["width"]))
        return names

    def values_at(self, addr: int) -> list[Value]:
        found = [v for v in self.values.values() if v.covers(addr)]
        return sorted(found, key=lambda v: (v.start if v.start is not None else -1, v.name))

    # ---- lowered ops ------------------------------------------------------------------

    def _dest_addr(self, ssa_id: int) -> int | None:
        dest = self.ssa[ssa_id].get("dest")
        value = self.values.get(dest) if dest else None
        return value.addr if value else None

    def _owner_of(self, low: dict) -> tuple[int | None, str]:
        """SSA op a lowered op implements, and its role in that op's lowering."""
        lid, engine, opcode = low["id"], low["engine"], low["opcode"]
        parent = low.get("from_select_choice")
        if parent is not None:
            if opcode in ("vselect", "select"):
                return parent, "select"
            if opcode == "multiply_add":
                return parent, "select as multiply_add: cond * (true - false) + false"
            return parent, "select subtract: true - false"
        if low.get("from_const_choice") is not None:
            return low["from_const_choice"], f"constant as {opcode}"
        if lid >= CSE_SHARED_SUB_PREFIX:
            return None, f"shared select subtract of CSE group {lid - CSE_SHARED_SUB_PREFIX}"
        if engine == "debug":
            return None, "debug trace"

        writes, _ = slot_effects(engine, low["slot"])
        wrote = writes[0][0] if writes else None
        direct = lid if lid in self.ssa else None
        lane_parent = None
        if lid >= SHIFT_LANE_FACTOR and engine == "alu" and lid % SHIFT_LANE_FACTOR < VLEN:
            cand = lid // SHIFT_LANE_FACTOR
            if cand in self.ssa and self.ssa[cand]["engine"] == "valu" and self.ssa[cand]["opcode"] == opcode:
                lane_parent = cand
        lane = lid % SHIFT_LANE_FACTOR
        if direct is not None and lane_parent is not None:
            # Ambiguous id (op 5000 or lane 0 of shifted op 5): the written address decides.
            base = self._dest_addr(lane_parent)
            if base is not None and wrote == base + lane and self._dest_addr(direct) != wrote:
                direct = None
            else:
                lane_parent = None
        if lane_parent is not None:
            return lane_parent, f"lane {lane} of {VLEN} (vector op shifted to the scalar ALU)"
        if direct is not None:
            return direct, ""
        return None, "no SSA op found for this lowered op"

    def lowered_touching(self, addr: int, cycle: int) -> list[tuple[int, dict, str]]:
        out = []
        for i, low in enumerate(self.lowered):
            if low["cycle"] != cycle:
                continue
            writes, reads = slot_effects(low["engine"], low["slot"])
            kinds = [k for k, r in (("writes", writes), ("reads", reads)) if _covers(r, addr)]
            if kinds:
                out.append((i, low, " and ".join(kinds)))
        return out

    def last_write_before(self, addr: int, cycle: int) -> tuple[int, dict] | None:
        best = None
        for i, low in enumerate(self.lowered):
            if low["cycle"] >= cycle:
                break
            if _covers(slot_effects(low["engine"], low["slot"])[0], addr):
                best = (i, low)
        return best

    def shared_subtract_of(self, low: dict) -> tuple[int, dict] | None:
        """The shared (CSE group) subtract feeding a select's multiply_add, if it has one."""
        if low["opcode"] != "multiply_add" or low.get("from_select_choice") is None:
            return None
        delta = slot_ints(low["slot"])[3]
        last = self.last_write_before(delta, low["cycle"])
        if last is not None and last[1]["id"] >= CSE_SHARED_SUB_PREFIX:
            return last
        return None

    def shared_subtract_users(self, index: int) -> list[int]:
        """SSA select ops whose multiply_add reads the shared subtract at self.lowered[index]."""
        users = []
        for low in self.lowered:
            feeder = self.shared_subtract_of(low)
            if feeder is not None and feeder[0] == index:
                users.append(low["from_select_choice"])
        return sorted(set(users))

    def kernel_slots(self, cycle: int) -> list[tuple[str, list]]:
        """Slots of the kernel bundle for a schedule cycle (empty without --kernel)."""
        if self.instrs is None:
            return []
        index = cycle + KERNEL_LEADING_BUNDLES
        if not 0 <= index < len(self.instrs):
            return []
        return [(engine, slot) for engine, slots in self.instrs[index].items()
                if engine != "debug" for slot in slots]

    # ---- formatting -------------------------------------------------------------------

    def where(self, op: dict) -> str:
        info = op.get("source_info") or {}
        parts = [f"{key}={info[key]}" for key in ("round", "phase", "group", "tree_depth", "iter")
                 if info.get(key) is not None]
        return ", ".join(parts) if parts else "no source info"

    def op_brief(self, ssa_id: int | None) -> str:
        if ssa_id is None or ssa_id not in self.ssa:
            return "no SSA op"
        op = self.ssa[ssa_id]
        return f"SSA op {op['id']} {op['opcode']} at cycle {op['cycle']} ({self.where(op)})"

    def defined_by(self, v: Value) -> str:
        if v.definer is None and v.lanes:
            return "defined lane by lane (each lane is written by its own scalar op)"
        if v.definer is None:
            return "no defining SSA op (program input, e.g. the zero word)"
        return f"defined by {self.op_brief(v.definer)}"

    def value_line(self, v: Value, addr: int | None = None) -> str:
        parts = [v.name, v.words()]
        if addr is not None and v.size > 1 and v.addr is not None:
            parts.append(f"lane {addr - v.addr} is address {addr}")
        parts.append(v.interval())
        if v.lane_of:
            parts.append(f"lane {v.lane_of[1]} of vector group {v.lane_of[0]}")
        if v.note:
            parts.append(v.note)
        return "  ".join(parts)

    def lowered_line(self, i: int, low: dict) -> str:
        ssa_id, role = self.owner[i]
        owner = f"SSA op {ssa_id}" if ssa_id is not None else "no SSA op"
        text = (f"cycle {low['cycle']} (bundle {low['cycle'] + KERNEL_LEADING_BUNDLES})  "
                f"{low['engine']} {slot_ints(low['slot'])}  <- {owner}")
        return text + (f"; {role}" if role else "")


# ---- subcommands ----------------------------------------------------------------------


def cmd_addr(s: Schedule, addr: int, cycle: int | None) -> list[str]:
    values = s.values_at(addr)
    if not values:
        return [f"No value's address range covers address {addr}."]
    lines = [f"Values whose address range covers address {addr} ({len(values)}):"]
    holder = None
    for v in values:
        mark = ""
        if cycle is not None:
            if v.holds_at(cycle):
                mark = f"   <== holds address {addr} for reads in cycle {cycle}"
                holder = v
            elif v.start == cycle:
                mark = f"   <== written in cycle {cycle} (readable from cycle {cycle + 1})"
        lines.append(f"  {s.value_line(v, addr)}{mark}")
        lines.append(f"      {s.defined_by(v)}")
    if cycle is None:
        return lines
    lines.append("")
    if holder is None:
        lines.append(f"No value is live at address {addr} for reads in cycle {cycle} "
                     f"(a value is readable in cycles def+1 .. last_use).")
    last = s.last_write_before(addr, cycle)
    if last is None:
        lines.append(f"Last write to address {addr} before cycle {cycle}: none")
    else:
        lines.append(f"Last write to address {addr} before cycle {cycle}:")
        lines.append(f"  {s.lowered_line(*last)}")
    touching = s.lowered_touching(addr, cycle)
    lines.append(f"Lowered ops in cycle {cycle} that touch address {addr}: {len(touching)}")
    for i, low, kind in touching:
        lines.append(f"  {kind}: {s.lowered_line(i, low)}")
    return lines


def _value_detail(s: Schedule, v: Value, indent: str) -> list[str]:
    lines = [f"{indent}{s.value_line(v)}", f"{indent}    {s.defined_by(v)}"]
    if v.lanes:
        for lname in v.lanes:
            lane = s.values[lname]
            if lane.definer is not None and lane.definer != v.definer:
                lines.append(f"{indent}    lane {lane.lane_of[1]} {lname} ({lane.words()}): "
                             f"{s.op_brief(lane.definer)}")
    return lines


def cmd_op(s: Schedule, ssa_id: int) -> list[str]:
    op = s.ssa.get(ssa_id)
    if op is None:
        return [f"No SSA op with id {ssa_id} (the JSON has {len(s.ssa)} ssa_ops)."]
    lines = [f"SSA op {ssa_id}: {op['opcode']} on {op['engine']}, cycle {op['cycle']}  ({s.where(op)})"]
    dest = op.get("dest")
    if dest:
        lines.append("Result:")
        lines += _value_detail(s, s.values[dest], "  ")
    lines.append("Operands:")
    definers = set()
    for k, src in enumerate(op.get("srcs") or []):
        if isinstance(src, dict) and "VReg" in src:
            v = s.values.get(value_name(src["VReg"]["id"], src["VReg"]["width"]))
            if v is None:
                lines.append(f"  [{k}] {src} (unknown value)")
                continue
            definers.add(v.definer)
            for j, text in enumerate(_value_detail(s, v, "      ")):
                lines.append(f"  [{k}] {text.strip()}" if j == 0 else text)
        else:
            lines.append(f"  [{k}] immediate {src.get('Imm') if isinstance(src, dict) else src}")
    extra = [d for d in op.get("dep_ids") or [] if d not in definers]
    if extra:
        lines.append("Other dependencies (ordering only, or through a vector group): "
                     + ", ".join(s.op_brief(d) for d in sorted(extra)))

    lowered = list(s.lowered_of.get(ssa_id, []))
    for low, _ in list(lowered):
        feeder = s.shared_subtract_of(low)
        if feeder is not None:
            group = feeder[1]["id"] - CSE_SHARED_SUB_PREFIX
            others = [u for u in s.shared_subtract_users(feeder[0]) if u != ssa_id]
            shared = f", shared with select op(s) {', '.join(map(str, others))}" if others else ""
            lowered.append((feeder[1], f"select subtract: true - false (CSE group {group}{shared})"))
    lowered.sort(key=lambda pair: (pair[0]["cycle"], pair[0]["id"]))
    has_subtract = any(role.startswith("select subtract") for _, role in lowered)
    lines.append(f"Lowered ops ({len(lowered)}):")
    for low, role in lowered:
        text = (f"  cycle {low['cycle']} (bundle {low['cycle'] + KERNEL_LEADING_BUNDLES})  "
                f"{low['engine']} {slot_ints(low['slot'])}")
        lines.append(text + (f"  [{role}]" if role else ""))
        if role.startswith("select as multiply_add") and not has_subtract:
            # No subtract was emitted: (true - false) already exists as a value.
            delta = slot_ints(low["slot"])[3]
            held = [v for v in s.values_at(delta) if v.holds_at(low["cycle"]) and v.addr == delta]
            what = f"{held[0].name}, {s.defined_by(held[0])}" if held else f"address {delta}"
            lines.append(f"      no subtract: (true - false) is precomputed in {what}")
    if not lowered:
        lines.append("  none (dead, eliminated, or lowered under a shared op)")

    consumers: dict[int, list[str]] = {}
    if dest:
        names = [dest] + s.values[dest].lanes
        if s.values[dest].lane_of:
            names.append(s.values[dest].lane_of[0])
        for name in names:
            for reader in s.readers.get(name, []):
                consumers.setdefault(reader, []).append(name)
    lines.append(f"Consumers ({len(consumers)}):")
    for reader in sorted(consumers, key=lambda r: (s.ssa[r]["cycle"], r)):
        via = ", ".join(sorted(set(consumers[reader])))
        lines.append(f"  {s.op_brief(reader)} reads {via}")
    return lines


def cmd_slot(s: Schedule, cycle: int, dest: int) -> list[str]:
    hits = []
    for i, low in enumerate(s.lowered):
        if low["cycle"] != cycle:
            continue
        writes, _ = slot_effects(low["engine"], low["slot"])
        if _covers(writes, dest) or (low["engine"] == "store" and slot_ints(low["slot"])[1] == dest):
            hits.append((i, low))
    header = f"schedule cycle {cycle} (kernel bundle {cycle + KERNEL_LEADING_BUNDLES})"
    if not hits:
        return [f"No slot in {header} writes address {dest} "
                f"(for a store, DEST is its pointer operand)."]
    lines = []
    for i, low in hits:
        ssa_id, role = s.owner[i]
        lines.append(f"Slot in {header}: {low['engine']} {slot_ints(low['slot'])}")
        if low["engine"] == "store":
            lines.append("  (a store: DEST matched its pointer operand; it writes memory, not scratch)")
        if ssa_id is None:
            lines.append(f"  belongs to: {role}")
            for user in s.shared_subtract_users(i):
                lines.append(f"    feeds the multiply_add of {s.op_brief(user)}")
        else:
            lines.append(f"  belongs to: {s.op_brief(ssa_id)}" + (f"; {role}" if role else ""))
            result = s.ssa[ssa_id].get("dest")
            if result:
                lines.append(f"  result: {s.value_line(s.values[result], dest)}")
        if s.instrs is not None:
            mine = (low["engine"], slot_ints(low["slot"]))
            found = any(engine == mine[0] and slot_ints(list(slot)) == mine[1]
                        for engine, slot in s.kernel_slots(cycle))
            lines.append("  kernel: bundle contains this slot" if found else
                         "  kernel: WARNING: this slot is not in the kernel bundle "
                         "(kernel and scheduler JSON are from different builds?)")
    return lines


def cmd_value(s: Schedule, text: str) -> list[str]:
    v = s.find_value(text)
    if v is None:
        return [f"No value named {text} (use v123_v, v45_s or a bare id)."]
    lines = _value_detail(s, v, "")
    names = [v.name] + v.lanes + ([v.lane_of[0]] if v.lane_of else [])
    readers = sorted({r for name in names for r in s.readers.get(name, [])},
                     key=lambda r: (s.ssa[r]["cycle"], r))
    lines.append(f"Readers ({len(readers)}):")
    lines += [f"  {s.op_brief(r)}" for r in readers]
    return lines


def load(path: str):
    with open(path) as f:
        return json.load(f)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scheduler_json", help="Scheduler output JSON (generate_ssa_kernel.py --output-json)")
    ap.add_argument("--kernel", help="Kernel JSON from the same build (--save-kernel); used to cross-check slots")
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("addr", help="values whose address range covers an address")
    p.add_argument("addr", type=int)
    p.add_argument("--cycle", type=int, help="also show what holds, last wrote and touches the address at this cycle")
    p.add_argument("--bundle", action="store_true", help="--cycle is a kernel bundle index (schedule cycle + 1)")
    p = sub.add_parser("op", help="an SSA op: operands, lowered ops, consumers")
    p.add_argument("id", type=int)
    p = sub.add_parser("slot", help="the SSA op behind the slot writing DEST in CYCLE")
    p.add_argument("cycle", type=int)
    p.add_argument("dest", type=int)
    p.add_argument("--bundle", action="store_true", help="CYCLE is a kernel bundle index (schedule cycle + 1)")
    p = sub.add_parser("value", help="a value: address, interval, definer, readers")
    p.add_argument("name")
    args = ap.parse_args(argv)

    sched = load(args.scheduler_json)
    if not sched.get("ssa_ops"):
        print("Scheduler JSON has no ssa_ops; regenerate it with the current scheduler_rs.", file=sys.stderr)
        return 1
    s = Schedule(sched, load(args.kernel) if args.kernel else None)
    cycle = getattr(args, "cycle", None)
    if cycle is not None and getattr(args, "bundle", False):
        cycle -= KERNEL_LEADING_BUNDLES
    if args.command == "addr":
        lines = cmd_addr(s, args.addr, cycle)
    elif args.command == "op":
        lines = cmd_op(s, args.id)
    elif args.command == "slot":
        lines = cmd_slot(s, cycle, args.dest)
    else:
        lines = cmd_value(s, args.name)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
