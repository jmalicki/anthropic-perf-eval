"""Soundness of resource-aware issue-cycle bounds (`--asap-alap-tightening energetic`,
scheduler_rs/src/cpsat_scheduler/resource_bounds.rs).

The bounds narrow the domain of every op's issue cycle. They are sound if no schedule of the
CP-SAT model (with every cycle below the horizon) has an op outside them. Each check builds
small dependency graphs with every kind of op the rule treats differently, encodes the
schedules of the model in Z3 with the rows `op_times.rs` and `model.rs` state, computes the
bounds with a port of the Rust code, and asks Z3 for a schedule that violates one.

* UNSAT on every graph (expected for RB1 to RB3): no schedule violates a bound. Three
  bounds are checked per op: the issue cycle is at least `issue_lb`; every lane of a shifted
  op is at least `lane_lb`; the op's execution end (`end` below: the completion of a
  shiftable op, the issue cycle of any other) is at most `horizon - 1 - finish_tail`. For a
  select on the multiply-add path `end` is the issue cycle; its multiply_add at issue + 1 is
  not bounded and not checked here. Only issue-cycle domains are tightened in the model.
* SAT (expected) for the variants the rule deliberately avoids, so the checks can tell a
  wrong bound from a right one:
  - RB-full-tail-for-shiftable: counting shiftable descendants in the tail of a shiftable op.
    A shifted reader's lanes wait lane by lane, so they can run before the op completes.
  - RB-issue-bound-for-shifted-ancestor: counting a shiftable ancestor from its issue bound
    instead of its lane bound. Its lanes can run before its issue bound.
  - RB-old-counting-rule: the rule this one replaces (count the ops of an ordering group
    with the same earliest cycle; when they exceed the capacity, raise the earliest cycle of
    the first ones). On four ops it excludes a schedule that is feasible, satisfies the
    ordering row and is one cycle shorter than anything it leaves.

Model semantics encoded (the rows the argument uses; where the real model has more rows the
encoding leaves them out, which only admits more schedules):
* every op has an issue cycle c; a shiftable op also has a shift literal, one cycle per lane
  and a completion cycle with completion >= c and completion >= every lane;
* the op executes until `end`: its completion if shiftable, else c;
* reader.c >= producer.end + 1 for every dependency, whatever is shifted (`prec_raw_*`,
  `consumer_after_completion_*`), + 1 more after a select on the multiply-add path;
* each lane of a shifted reader >= lane i of a shifted producer + 1, or >= producer.c + 1
  when the producer is not shifted or cannot be (`shift_alu_prec_*`);
* an eliminable vbroadcast is eliminated exactly when all its readers are shifted; then it
  uses no slot and their lanes wait for the scalar producer instead;
* slots: a vector op not shifted, one VALU slot at c; shifted, one ALU slot per lane; a
  select, one Flow slot at c or VALU slots at c and c + 1; a ConstChoice, a Load or a Flow
  slot at c; everything else one slot of its engine at c;
* every cycle is in [0, horizon).

Run standalone:

    uv run python -m verification.resource_bounds
"""

from __future__ import annotations

import random
import time

from z3 import And, Bool, If, Implies, Int, Not, Or, Solver, Sum, sat, unsat

LANES = 2  # vector length in the proofs (the rule uses VLEN; any value works)
CAPS = {"valu": 1, "alu": 2, "load": 1, "store": 1, "flow": 1}
KINDS = ["valu", "valu_shift", "alu", "load", "store", "flow", "select", "const"]
ENGINES = ["valu", "alu", "load", "store", "flow", "joint"]


# ----------------------------------------------------------------------------
# Port of resource_bounds.rs
# ----------------------------------------------------------------------------


def _cap(caps, lanes, engine):
    return lanes * caps["valu"] + caps["alu"] if engine == "joint" else caps[engine]


def min_work(kind, engine, lanes, vb_optional):
    """Slots under the cheapest encoding (makespan_rows.rs min_work)."""
    if kind in ("select", "const") or (kind == "vb" and vb_optional):
        return 0
    if kind in ("valu", "valu_shift", "vb"):
        if engine == "joint":
            return lanes
        return int(engine == "valu" and kind != "valu_shift")
    if kind == "alu":
        return int(engine in ("alu", "joint"))
    return int(engine == kind)


def _window_bound(members, key, work, caps, lanes):
    """max over r and engines of r + ceil(W(key >= r) / cap); None without work."""
    best = None
    for e in ENGINES:
        cap = _cap(caps, lanes, e)
        if cap <= 0:
            continue
        for r in sorted({key[m] for m in members}):
            w = sum(work[m][e] for m in members if key[m] >= r)
            if w > 0:
                bound = r + (w + cap - 1) // cap
                best = bound if best is None else max(best, bound)
    return best


def resource_bounds(ops, caps=CAPS, lanes=LANES, lane_key="lane", soft_tail="unshiftable"):
    """(issue_lb, lane_lb, finish_tail) as resource_bounds.rs computes them.

    `ops` is a list of (kind, deps) in dependency order. The two keyword arguments select
    the unsound variants: `lane_key="issue"` counts a shiftable ancestor from its issue
    bound; `soft_tail="all"` counts shiftable descendants in the tail of a shiftable op.
    """
    n = len(ops)
    kinds = [k for k, _ in ops]
    preds = [list(d) for _, d in ops]
    succs = [[] for _ in range(n)]
    for i, deps in enumerate(preds):
        for d in deps:
            succs[d].append(i)
    shiftable = [k == "valu_shift" for k in kinds]
    vb_optional = any(shiftable)
    optional_vb = [k == "vb" and vb_optional for k in kinds]
    work = [{e: min_work(kinds[i], e, lanes, vb_optional) for e in ENGINES} for i in range(n)]

    ancestors = [set() for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            ancestors[i] |= ancestors[p] | {p}
    issue_lb, lane_lb, lane_ready = [0] * n, [0] * n, [0] * n
    for i in range(n):
        lb = max([issue_lb[p] + 1 for p in preds[i]], default=0)
        bound = _window_bound(ancestors[i], lane_lb, work, caps, lanes)
        if bound is not None:
            lb = max(lb, bound)
        issue_lb[i] = lb
        from_producers = max([lane_ready[p] for p in preds[i]], default=0)
        lane_lb[i] = min(lb, from_producers) if shiftable[i] and lane_key == "lane" else lb
        lane_ready[i] = from_producers if optional_vb[i] else lane_lb[i] + 1

    descendants = [set() for _ in range(n)]
    for i in reversed(range(n)):
        for s in succs[i]:
            descendants[i] |= descendants[s] | {s}
    finish_tail = [0] * n
    for i in reversed(range(n)):
        tail = max([finish_tail[s] + 1 for s in succs[i]], default=0)
        only_unshiftable = (shiftable[i] or optional_vb[i]) and soft_tail == "unshiftable"
        members = {d for d in descendants[i] if not (only_unshiftable and shiftable[d])}
        bound = _window_bound(members, finish_tail, work, caps, lanes)
        if bound is not None:
            tail = max(tail, bound)
        finish_tail[i] = tail
    return issue_lb, lane_lb, finish_tail


# The same instances, with the same expected numbers, as the unit tests of resource_bounds.rs
# (VLEN 8 there and here), so that the port and the Rust code are pinned to each other.
def _port_cases():
    real = {"valu": 6, "alu": 12, "load": 2, "store": 2, "flow": 1}
    fan_in = [["load", []] for _ in range(5)] + [["alu", [0, 1, 2, 3, 4]]]
    yield "head-fan-in", resource_bounds(fan_in, real, 8), ([0, 0, 0, 0, 0, 3], None, [1, 1, 1, 1, 1, 0])
    fan_out = [["load", []]] + [["alu", [0]] for _ in range(7)]
    got = resource_bounds(fan_out, dict(real, alu=2), 8)
    yield "tail-fan-out", (got[0][1:], None, got[2][:1]), ([1] * 7, None, [4])
    one_vector = dict(real, valu=1, alu=0)
    for name, producer, reader, want in [
        ("tail-hard-producer", "valu", "valu_shift", 4),
        ("tail-shiftable-producer", "valu_shift", "valu_shift", 1),
        ("tail-unshiftable-readers", "valu_shift", "valu", 4),
    ]:
        g = [[producer, []]] + [[reader, [0]] for _ in range(4)]
        yield name, (None, None, resource_bounds(g, one_vector, 8)[2][:1]), (None, None, [want])
    lanes = [["load", []] for _ in range(5)] + [["valu_shift", [0, 1, 2, 3, 4]], ["valu", [5]]]
    got = resource_bounds(lanes, real, 8)
    yield "lane-bound", (got[0][5:], got[1][5:], None), ([3, 4], [1, 4], None)


def check_port_matches_rust_tests(ctx=None):
    """RB0: the port gives the numbers the Rust unit tests assert."""
    start = time.time()
    wrong = [(name, got, want) for name, got, want in _port_cases()
             if any(w is not None and g != w for g, w in zip(got, want))]
    return "RB0-port-matches-rust-tests", "sat" if wrong else "unsat", time.time() - start, {"mismatches": wrong}


# ----------------------------------------------------------------------------
# Schedules of the model
# ----------------------------------------------------------------------------


def _graph(rng: random.Random, n: int, with_vb: bool):
    """Random DAG as a list of [kind, deps], optionally with an eliminable vbroadcast."""
    ops = []
    for i in range(n):
        kind = rng.choice(KINDS)
        deps = sorted({rng.randrange(i) for _ in range(rng.randint(0, min(2, i)))}) if i else []
        ops.append([kind, deps])
    if with_vb and n >= 3:
        scalar = len(ops)
        ops.append(["alu", [rng.randrange(scalar)] if rng.random() < 0.5 else []])
        vb = len(ops)
        ops.append(["vb", [scalar]])
        for _ in range(rng.randint(1, 2)):
            ops.append(["valu_shift", [vb] + ([rng.randrange(scalar)] if rng.random() < 0.5 else [])])
        if rng.random() < 0.5:
            ops.append([rng.choice(["valu", "valu_shift"]), [len(ops) - 1]])
    return ops


def _encode(ops, horizon, caps=CAPS):
    """Z3 schedules of the model. Returns (solver, terms)."""
    n = len(ops)
    s = Solver()
    c = [Int(f"c{i}") for i in range(n)]
    shift = [Bool(f"shift{i}") for i in range(n)]
    lane = [[Int(f"l{i}_{j}") for j in range(LANES)] for i in range(n)]
    comp = [Int(f"comp{i}") for i in range(n)]
    fma = [Bool(f"fma{i}") for i in range(n)]
    load_path = [Bool(f"ld{i}") for i in range(n)]
    elim = [Bool(f"elim{i}") for i in range(n)]
    readers = [[j for j in range(n) if i in ops[j][1]] for i in range(n)]
    shiftable = [kind == "valu_shift" for kind, _ in ops]

    def end(p):
        return comp[p] if shiftable[p] else c[p]

    for i, (kind, deps) in enumerate(ops):
        s.add(c[i] >= 0, c[i] < horizon)
        if shiftable[i]:
            s.add(comp[i] >= c[i], comp[i] < horizon)
            for j in range(LANES):
                s.add(lane[i][j] >= 0, lane[i][j] < horizon, comp[i] >= lane[i][j])
        else:
            s.add(Not(shift[i]))
        if kind != "select":
            s.add(Not(fma[i]))
        if kind != "const":
            s.add(Not(load_path[i]))
        if kind == "vb" and readers[i] and all(shiftable[r] for r in readers[i]):
            s.add(elim[i] == And([shift[r] for r in readers[i]]))
        else:
            s.add(Not(elim[i]))
        for p in deps:
            # The reader's issue cycle waits for the whole result, shifted or not.
            s.add(c[i] >= end(p) + 1 + (If(fma[p], 1, 0) if ops[p][0] == "select" else 0))
            if not shiftable[i]:
                continue
            for j in range(LANES):
                if shiftable[p]:
                    ready = If(shift[p], lane[p][j] + 1, c[p] + 1)
                elif ops[p][0] == "vb":
                    scalar = ops[p][1][0]
                    ready = If(elim[p], end(scalar) + 1, c[p] + 1)
                else:
                    ready = c[p] + 1
                s.add(Implies(shift[i], lane[i][j] >= ready))
    occupy = {e: [[] for _ in range(horizon + 1)] for e in caps}
    for i, (kind, _) in enumerate(ops):
        for t in range(horizon + 1):
            def at(x):
                return If(x == t, 1, 0)
            if kind == "valu":
                occupy["valu"][t].append(at(c[i]))
            elif kind == "valu_shift":
                occupy["valu"][t].append(If(Not(shift[i]), at(c[i]), 0))
                for j in range(LANES):
                    occupy["alu"][t].append(If(shift[i], at(lane[i][j]), 0))
            elif kind == "vb":
                occupy["valu"][t].append(If(Not(elim[i]), at(c[i]), 0))
            elif kind == "select":
                occupy["flow"][t].append(If(Not(fma[i]), at(c[i]), 0))
                occupy["valu"][t].append(If(fma[i], at(c[i]) + at(c[i] + 1), 0))
            elif kind == "const":
                occupy["load"][t].append(If(load_path[i], at(c[i]), 0))
                occupy["flow"][t].append(If(Not(load_path[i]), at(c[i]), 0))
            else:
                occupy[kind][t].append(at(c[i]))
    for e in caps:
        for t in range(horizon + 1):
            if occupy[e][t]:
                s.add(Sum(occupy[e][t]) <= caps[e])
    return s, dict(c=c, shift=shift, lane=lane, end=end, shiftable=shiftable)


def _bound_violations(ops, horizon, terms, caps=CAPS, **variant):
    """Some op is outside its bounds."""
    issue_lb, lane_lb, finish_tail = resource_bounds(ops, caps, LANES, **variant)
    c, shift, lane, end = terms["c"], terms["shift"], terms["lane"], terms["end"]
    viol = []
    for i in range(len(ops)):
        viol.append(c[i] < issue_lb[i])
        viol.append(end(i) > horizon - 1 - finish_tail[i])
        if terms["shiftable"][i]:
            viol += [And(shift[i], lane[i][j] < lane_lb[i]) for j in range(LANES)]
    return Or(viol)


def _check(name, graphs, caps=CAPS, **variant):
    start = time.time()
    found = None
    for label, ops in graphs:
        horizon = 2 * len(ops) + 3
        s, terms = _encode(ops, horizon, caps)
        s.add(_bound_violations(ops, horizon, terms, caps, **variant))
        outcome = s.check()
        if outcome == sat:
            found = (label, ops)
            break
        if outcome != unsat:
            # A timeout proves nothing: report it, and the runner fails the check.
            return name, "unknown", time.time() - start, {"graphs": len(graphs), "unresolved": label,
                                                           "reason": s.reason_unknown()}
    return name, "sat" if found else "unsat", time.time() - start, {"graphs": len(graphs), "counterexample": found}


def _random_graphs(seeds, with_vb):
    out = []
    for seed in seeds:
        rng = random.Random(seed)
        out.append((f"seed {seed}", _graph(rng, rng.randint(3, 6), with_vb)))
    return out


# One scalar root, six shiftable ops that read it, one shiftable op that reads all six, and a
# reader of that. Lane 0 of the six can be done two cycles before all of them are complete.
_SHIFT_FAN = [["alu", []]] + [["valu_shift", [0]] for _ in range(6)] + [["valu_shift", [1, 2, 3, 4, 5, 6]], ["valu", [7]]]
# A shiftable op with five shiftable readers: their lanes can run while its completion waits.
_SHIFT_READERS = [["valu_shift", []]] + [["valu_shift", [0]] for _ in range(5)]
_CRAFTED = [("shift fan", _SHIFT_FAN), ("shift readers", _SHIFT_READERS)]


def check_bounds_sound(ctx=None):
    """RB1: no schedule has an op outside its bounds."""
    return _check("RB1-bounds-sound", _random_graphs(range(70), with_vb=False))


def check_bounds_sound_with_vbroadcast(ctx=None):
    """RB2: as RB1 with an eliminable vbroadcast."""
    return _check("RB2-bounds-sound-with-vbroadcast", _random_graphs(range(100, 150), with_vb=True))


def check_bounds_sound_on_crafted_graphs(ctx=None):
    """RB3: as RB1 on the graphs that break the two variants below."""
    return _check("RB3-bounds-sound-crafted", _CRAFTED)


def check_full_tail_for_shiftable_unsound(ctx=None):
    """RB-full-tail-for-shiftable: shiftable descendants in a shiftable op's tail (SAT)."""
    return _check("RB-full-tail-for-shiftable", _CRAFTED, soft_tail="all")


def check_issue_bound_for_shifted_ancestor_unsound(ctx=None):
    """RB-issue-bound-for-shifted-ancestor: shiftable ancestors counted from their issue bound (SAT)."""
    return _check("RB-issue-bound-for-shifted-ancestor", _CRAFTED, lane_key="issue")


def _old_counting_rule(asap, alap, chain, cap, max_iter=3):
    """The ASAP pass of the removed rule for one engine and one ordering group: while more
    ops of the group have earliest cycle t than the engine has slots, the first of them (in
    group order) move to t + 1, or to the first cycle that is not over capacity."""
    new = dict(asap)
    for _ in range(max_iter):
        changed = False
        count = {}
        for o in chain:
            count[new[o]] = count.get(new[o], 0) + 1
        for o in chain:
            a = new[o]
            if count[a] <= cap or a >= alap[o]:
                continue
            first_ok = next((t for t in range(a, alap[o] + 1) if count.get(t, 0) <= cap), a + 1)
            moved = min(a + 1, first_ok)
            count[a] -= 1
            count[moved] = count.get(moved, 0) + 1
            new[o] = moved
            changed = True
        if not changed:
            break
    return new


def check_old_counting_rule_unsound(ctx=None):
    """RB-old-counting-rule: the removed rule excludes a feasible schedule that satisfies the
    ordering row and is shorter than every schedule it leaves (SAT)."""
    start = time.time()
    # R (scalar) feeds two multiply_adds A and B of one ordering group; C reads A. One VALU slot.
    ops = [["alu", []], ["valu", [0]], ["valu", [0]], ["alu", [1]]]
    r, a, b, reader = range(4)
    horizon = 5
    asap = {r: 0, a: 1, b: 1, reader: 2}
    alap = {r: 2, a: 3, b: 4, reader: 4}
    pushed = _old_counting_rule(asap, alap, chain=[a, b], cap=CAPS["valu"])
    pushed[reader] = max(pushed[reader], pushed[a] + 1)

    unresolved = []

    def best(lower_bounds):
        """Shortest makespan with A before or with B (the ordering row) and c >= lower bounds."""
        for makespan in range(1, horizon + 1):
            s, terms = _encode(ops, makespan)
            c = terms["c"]
            s.add(c[a] <= c[b], *[c[i] >= lower_bounds[i] for i in range(len(ops))])
            outcome = s.check()
            if outcome == sat:
                return makespan
            if outcome != unsat:
                # A timeout proves nothing about this makespan: the check is then unresolved.
                unresolved.append((makespan, s.reason_unknown()))
                return None
        return None

    free, restricted = best(asap), best(pushed)
    if unresolved:
        return "RB-old-counting-rule", "unknown", time.time() - start, {"unresolved": unresolved}
    # Every makespan below `free` (and below `restricted`, or up to the horizon) is proved
    # infeasible, so `free` is a witnessed optimum and the comparison is exact.
    result = "sat" if free is not None and (restricted is None or restricted > free) else "unsat"
    return "RB-old-counting-rule", result, time.time() - start, {
        "counterexample": {"asap": asap, "old rule": pushed, "best makespan": free, "with the old rule": restricted}}


RESOURCE_BOUND_CHECKS = [
    ("RB0-port-matches-rust-tests", "The port of resource_bounds.rs gives the numbers its unit tests assert", check_port_matches_rust_tests),
    ("RB1-bounds-sound", "Resource-aware issue, lane and completion bounds hold in every schedule", check_bounds_sound),
    ("RB2-bounds-sound-with-vbroadcast", "The bounds hold with an eliminable vbroadcast", check_bounds_sound_with_vbroadcast),
    ("RB3-bounds-sound-crafted", "The bounds hold on the graphs that break the unsound variants", check_bounds_sound_on_crafted_graphs),
    ("RB-full-tail-for-shiftable", "Shiftable descendants in a shiftable op's tail give an unsound bound (SAT)", check_full_tail_for_shiftable_unsound),
    ("RB-issue-bound-for-shifted-ancestor", "Counting a shiftable ancestor from its issue bound is unsound (SAT)", check_issue_bound_for_shifted_ancestor_unsound),
    ("RB-old-counting-rule", "The removed same-earliest-cycle counting rule excludes the optimum (SAT)", check_old_counting_rule_unsound),
]

EXPECT_SAT = {"RB-full-tail-for-shiftable", "RB-issue-bound-for-shifted-ancestor", "RB-old-counting-rule"}


if __name__ == "__main__":
    import sys

    failed = 0
    for name, desc, fn in RESOURCE_BOUND_CHECKS:
        _, result, secs, stats = fn()
        want = "sat" if name in EXPECT_SAT else "unsat"
        failed += result != want
        detail = stats.get("counterexample") or stats.get("mismatches") or ""
        print(f"{'PASS' if result == want else 'FAIL'} [{name}] {desc}: {result} ({secs:.1f}s) {detail}")
    sys.exit(1 if failed else 0)
