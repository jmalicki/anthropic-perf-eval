#!/usr/bin/env python3
"""Static head-body-tail lower bounds on the makespan of one emission.

The capacity floor in ``scripts/schedule_report.py`` divides all the work by the slot
capacity, as if every engine could be busy from cycle 0 to the last cycle. A real schedule
cannot: the ops at the top of the dependency graph cannot start late, the ops far from the
end cannot finish late, and nothing at depth t can run before cycle t. The rows below
charge that ramp-up and tail. They are the classical head-body-tail (release date / tail)
bounds of machine scheduling, generalised to parallel slots.

Definitions (all edges of the SSA graph have latency 1 unless noted; see "Latencies")
------------------------------------------------------------------------------------
head(op)  longest dependency path from any source to op, in cycles. Every slot of op lies
          at a cycle >= head(op).
tail(op)  longest dependency path from op to any sink, in cycles: the number of cycles that
          must exist strictly AFTER every slot of op. A sink has tail 0. (Counting op's own
          cycle instead gives tail + 1; the row below then needs ``q - 1``. Writing it as
          ``q + ceil(W/cap)`` with the own-cycle tail is unsound by one cycle; the
          counterexample is kept in verification/static_bound_counterexamples.json.)

Rows. For a row r with capacity cap_r (slots per cycle) and op work w_r(op):
    makespan >= t + q + ceil(W_r(t, q) / cap_r),
    W_r(t, q) = sum of w_r(op) over the ops with head(op) >= t and tail(op) >= q,
because those ops' slots all lie in cycles [t, makespan - 1 - q], a window of
makespan - t - q cycles holding at most cap_r slots each. t = q = 0 is the whole-schedule
capacity floor; q = 0 is the head row; t = 0 is the tail row. The bound is the maximum over
every row and every (t, q) with W > 0; the grid is small (critical path squared), so all
pairs are evaluated.

Rows and work (minimum over the op's encodings, so the bound holds for any choice):
    valu       6   ops that must use a VALU slot: non-shiftable vector ops (multiply_add,
                   vbroadcast that cannot be eliminated, cdiv). Shiftable 0, select 0.
    alu        12  native scalar ops.
    load       2   loads. ConstChoice 0 (it may be add_imm or ALU-derived).
    store      2   stores.
    flow       1   flow ops. SelectChoice 0 (the FMA path uses no flow), ConstChoice 0.
    alu+valu   60  8 * VALU slots + ALU slots (= vector-op equivalents at 7.5 per cycle):
                   every vector op 8 whether it runs on VALU or as 8 scalar lanes,
                   native scalar 1, select 0, const 0.
The valu, alu and alu+valu rows at t = q = 0 together give exactly the integer joint floor
of ``schedule_report.joint_alu_floor``: max(ceil(N/6), ceil(S/12), ceil((8V+S)/60)).

Latencies. The CP-SAT model's precedence offsets (model.rs sections 5 to 5d) are 1 for
every edge, 2 after a SelectChoice that takes the FMA path, and for a vector op run as 8
scalar lanes, lane i of a shifted consumer waits for lane i of its producer and everyone
else waits for the last lane. The minimum over the choices is 1 everywhere, which is what
head and tail use. One exception: the edge out of an eliminable vbroadcast (a broadcast of
a scalar whose consumers are all shiftable, when the run enables vbroadcast elimination)
has latency 0 here. The model today still makes
the consumers wait for the broadcast's cycle variable, but a consumer that reads the scalar
directly only needs the scalar's producer, so latency 0 keeps the bound valid either way.

These are bounds on scheduling THIS emission (these ops, these dependencies), never bounds
on the problem: an emission with fewer ops or shorter chains has lower ones.

Proof: verification/static_bounds_proof.py checks with Z3, on random small graphs against
a reference schedule model that is more permissive than CP-SAT's (lanes at any cycles,
selects on flow or VALU, optional broadcasts), that no schedule is shorter than the bound.

The core (``BoundOp``, ``heads``, ``tails``, ``makespan_rows``) is engine-agnostic and has
no I/O, so the CP-SAT model can mirror it: ``ops_from_schedule_json`` is the only part that
knows the scheduler JSON.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field

import numpy as np

VLEN = 8
CAP = {"valu": 6, "alu": 12, "load": 2, "store": 2, "flow": 1}
JOINT = "alu+valu"
ROWS = ("valu", "alu", "load", "store", "flow", JOINT)
SCOPE_NOTE = "These are bounds on scheduling the emitted ops, not bounds on the problem."

# Vector opcodes the scheduler may run as VLEN scalar lanes (Opcode::can_shift_to_alu in
# scheduler_rs/src/types.rs, by their JSON names).
SHIFTABLE_OPCODES = frozenset(
    ("+", "-", "*", "/", "%", "&", "|", "^", "<<", ">>", "==", "!=", "<", "<=", ">", ">=")
)
# Engines that hold no slot in the CP-SAT model.
NO_SLOT_ENGINES = frozenset(("debug", "eliminated_broadcast", "shifted_valu"))


def row_caps(cap=None, vlen=VLEN):
    """Slots per cycle for each row; the joint row counts vlen per VALU slot plus ALU slots."""
    cap = dict(CAP if cap is None else cap)
    return {**{e: cap[e] for e in ("valu", "alu", "load", "store", "flow")}, JOINT: vlen * cap["valu"] + cap["alu"]}


@dataclass(frozen=True)
class BoundOp:
    """One op for the bound: its predecessors and the minimum work it puts on each row.

    ``deps`` holds (predecessor id, latency) pairs. ``work`` maps a row name to the fewest
    slots of that row the op uses under any of its encodings (absent = 0).
    """

    id: int
    deps: tuple[tuple[int, int], ...] = ()
    work: dict = field(default_factory=dict)


def _topo(ops):
    """Ops in dependency order. Dependencies on ids outside ``ops`` are ignored (the model
    ignores them too). Raises on a cycle."""
    by_id = {o.id: o for o in ops}
    indeg = {o.id: 0 for o in ops}
    succ = collections.defaultdict(list)
    for o in ops:
        for p, lat in o.deps:
            if p in by_id:
                indeg[o.id] += 1
                succ[p].append((o.id, lat))
    ready = collections.deque(sorted(i for i, d in indeg.items() if d == 0))
    order = []
    while ready:
        i = ready.popleft()
        order.append(i)
        for c, _ in succ[i]:
            indeg[c] -= 1
            if indeg[c] == 0:
                ready.append(c)
    if len(order) != len(ops):
        raise ValueError("dependency graph has a cycle")
    return order, by_id, succ


def heads(ops):
    """head(op): longest path (sum of latencies) from a source to op."""
    order, by_id, _ = _topo(ops)
    h = {}
    for i in order:
        h[i] = max((h[p] + lat for p, lat in by_id[i].deps if p in by_id), default=0)
    return h


def tails(ops):
    """tail(op): longest path from op to a sink, i.e. cycles needed strictly after op."""
    order, _, succ = _topo(ops)
    q = {}
    for i in reversed(order):
        q[i] = max((q[c] + lat for c, lat in succ[i]), default=0)
    return q


def _best_cell(grid, cap):
    """Strongest t + q + ceil(W/cap) over the cells of a suffix-summed work grid with W > 0."""
    t_idx = np.arange(grid.shape[0])[:, None]
    q_idx = np.arange(grid.shape[1])[None, :]
    value = np.where(grid > 0, t_idx + q_idx + -(-grid // cap), 0)
    flat = int(np.argmax(value))
    t, q = divmod(flat, grid.shape[1])
    return int(value[t, q]), t, q


def makespan_rows(ops, caps=None):
    """Evaluate every row. Returns a dict:

    bound          the strongest row (fewest cycles any schedule of these ops can take)
    engine, t, q   where it comes from (engine is the row name; "alu+valu" is the joint row);
                   work = W of that row in that window
    rows           per row: best value overall, best with q = 0 (head row), best with t = 0
                   (tail row), and the whole-schedule floor (t = q = 0)
    critical_path  longest path + 1 over ops with work (a one-op window)
    """
    caps = row_caps() if caps is None else caps
    if not ops:
        return {"bound": 0, "engine": None, "t": 0, "q": 0, "work": 0, "rows": {}, "critical_path": 0}
    h, tl = heads(ops), tails(ops)
    n_t, n_q = max(h.values()) + 1, max(tl.values()) + 1
    out_rows = {}
    best = (0, None, 0, 0, 0)
    for r, cap in caps.items():
        grid = np.zeros((n_t + 1, n_q + 1), dtype=np.int64)
        for o in ops:
            w = o.work.get(r, 0)
            if w:
                grid[h[o.id], tl[o.id]] += w
        # Suffix sums: grid[t, q] = work of ops with head >= t and tail >= q.
        grid = grid[::-1, ::-1].cumsum(0).cumsum(1)[::-1, ::-1]
        value, t, q = _best_cell(grid, cap)
        head_only, th, _ = _best_cell(grid[:, :1], cap)
        tail_only, _, qt = _best_cell(grid[:1, :], cap)
        total = int(grid[0, 0])
        out_rows[r] = {
            "cap": cap,
            "work": total,
            "floor": -(-total // cap),
            "head_row": {"bound": head_only, "t": th},
            "tail_row": {"bound": tail_only, "q": qt},
            "best": {"bound": value, "t": t, "q": q, "work": int(grid[t, q])},
        }
        if value > best[0]:
            best = (value, r, t, q, int(grid[t, q]))
    crit = max((h[o.id] + tl[o.id] + 1 for o in ops if any(o.work.values())), default=0)
    value, r, t, q, w = best
    return {"bound": value, "engine": r, "t": t, "q": q, "work": w, "rows": out_rows, "critical_path": crit}


# ----------------------------------------------------------------------------
# Scheduler output JSON -> BoundOps
# ----------------------------------------------------------------------------
def _eliminable_vbroadcasts(ssa, consumers, shiftable):
    """Vbroadcasts the model may drop (model.rs 5d): one dependency, on a non-VALU scalar
    producer, and every consumer shiftable."""
    out = set()
    for o in ssa.values():
        if o["engine"] != "valu" or o["opcode"] != "vbroadcast":
            continue
        deps = [d for d in o["dep_ids"] if d in ssa]
        if len(o["dep_ids"]) != 1 or not deps or ssa[deps[0]]["engine"] == "valu":
            continue
        cons = consumers.get(o["id"], [])
        if cons and all(c in shiftable for c in cons):
            out.add(o["id"])
    return out


def _min_work(o, shiftable, eliminable, vlen):
    """Fewest slots per row the SSA op uses under any encoding."""
    e, opc = o["engine"], o["opcode"]
    if e in NO_SLOT_ENGINES or e == "barrier" or opc in ("select_choice", "const_choice"):
        return {}
    if e == "valu":
        if o["id"] in eliminable:
            return {}
        if o["id"] in shiftable:
            return {JOINT: vlen}
        return {"valu": 1, JOINT: vlen}
    if e == "alu":
        return {"alu": 1, JOINT: 1}
    return {e: 1}


def _run_work(d, ssa, shiftable, vlen):
    """Slots per row each SSA op used in THIS run's encodings (lowered ops attributed to the
    SSA op they came from), with the VALU->ALU shift still relaxed. Summed over all ops this
    reproduces the counts behind schedule_report's emission floor."""
    work = collections.defaultdict(collections.Counter)
    for lo in d["ops"]:
        sid = lo.get("from_select_choice", lo.get("from_const_choice", lo["id"]))
        eng = lo["engine"]
        # A vector op run as scalar lanes appears as `alu` ops with id = ssa_id * 1000 + lane
        # (the same rule as schedule_report.lowered_with_ssa).
        if eng == "alu" and (sid not in ssa or ssa[sid]["engine"] == "valu"):
            parent = ssa.get(lo["id"] // 1000)
            if parent is not None and parent["engine"] == "valu":
                work[parent["id"]][JOINT] += 1  # one lane; vlen lanes = one vector op
                continue
        if sid not in ssa:
            sid = None
        if eng == "valu":
            if sid not in shiftable:
                work[sid]["valu"] += 1
            work[sid][JOINT] += vlen
        elif eng == "alu":
            work[sid]["alu"] += 1
            work[sid][JOINT] += 1
        elif eng in CAP:
            work[sid][eng] += 1
    return work


def ops_from_schedule_json(d, encodings="any", vlen=VLEN):
    """BoundOps for the SSA ops of a scheduler output JSON.

    encodings="any": minimum work over every encoding (the bound holds for any choice; what
    a CP-SAT row may use). encodings="run": the work of the encodings this run chose, shift
    relaxed (comparable with schedule_report's emission floor). Barrier pseudo-ops are
    dropped with their edges: the CP-SAT model never contains them.
    """
    if encodings not in ("any", "run"):
        raise ValueError(f"encodings must be 'any' or 'run', not {encodings!r}")
    ssa = {o["id"]: o for o in d.get("ssa_ops") or [] if o["engine"] != "barrier"}
    if not ssa:
        raise ValueError("scheduler JSON has no ssa_ops (needed for dependencies)")
    opts = d.get("run_options") or {}
    shift_on = opts.get("enable_valu_shift", True) and opts.get("valu_shift", "optimize") != "valu"
    shiftable = {i for i, o in ssa.items() if shift_on and o["engine"] == "valu" and o["opcode"] in SHIFTABLE_OPCODES}
    consumers = collections.defaultdict(list)
    for o in ssa.values():
        for p in o["dep_ids"]:
            consumers[p].append(o["id"])
    # Absent options (a JSON without run_options, the proof's instances) allow elimination,
    # which only weakens the bound.
    elim_on = shift_on and opts.get("enable_vbroadcast_elimination", True)
    eliminable = _eliminable_vbroadcasts(ssa, consumers, shiftable) if elim_on else set()
    run = _run_work(d, ssa, shiftable, vlen) if encodings == "run" else None
    out = []
    for i, o in ssa.items():
        deps = tuple((p, 0 if p in eliminable else 1) for p in o["dep_ids"] if p in ssa)
        work = dict(run.get(i, {})) if run is not None else _min_work(o, shiftable, eliminable, vlen)
        out.append(BoundOp(i, deps, work))
    if run is not None and run.get(None):
        # Lowered ops not attributed to any SSA op: no dependencies known, so only the
        # whole-schedule (t = q = 0) cell sees them. A pseudo-op with head = tail = 0.
        out.append(BoundOp(-1, (), dict(run[None])))
    return out


def static_makespan_bound(d, encodings="any"):
    """The rows for a scheduler output JSON (see ``makespan_rows``)."""
    return makespan_rows(ops_from_schedule_json(d, encodings))
