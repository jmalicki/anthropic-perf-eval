"""Soundness of the makespan rows (`--cpsat-makespan-rows`, scheduler_rs/src/cpsat_scheduler/makespan_rows.rs).

The rows are lower bounds on the makespan from static heads and tails. They are sound if no
schedule of the CP-SAT model violates them. Each check below builds random small dependency
graphs with every kind of op the rows treat differently, encodes the schedules of the model
in Z3 (issue cycles, shifted lanes, the multiply-add path of a select, ConstChoice on Load or
Flow, an eliminable vbroadcast, slot capacities, precedence, the makespan), computes the rows
exactly as the Rust code does, and asks Z3 for a schedule that violates one of them.

* UNSAT on every graph (expected for MR1 to MR3): no schedule violates any row.
* MR1-old-tail-definition (expected SAT): the first draft of the definition counted the op's
  own cycle in its tail (makespan >= t + q + W/cap with q one larger). Z3 finds a schedule
  that violates it, so the checks can tell a wrong row from a right one.

Model semantics encoded (see op_times.rs):
* a VALU op not shifted uses one VALU slot at its issue cycle c; shifted, its lanes use one
  ALU slot each at their own cycles; its result is ready at c + 1, lane i at lane_i + 1;
* a select on the vselect path uses one Flow slot at c; on the multiply-add path two VALU
  slots (c and c + 1) and its result is ready at c + 2; every select has a multiply_add
  cycle variable c + 1 that the makespan covers on both paths (max_cycle >= fma_cycle);
* a ConstChoice uses one Load or one Flow slot at c;
* an eliminable vbroadcast is eliminated exactly when all its readers are shifted; then it
  uses no slot and their lanes read the scalar producer (ready at its c + 1);
* readers that are not shifted wait for the producer's result, shifted lanes wait lane by
  lane for a shifted producer and for the whole result otherwise;
* makespan = 1 + the largest cycle any op occupies (lanes and multiply_add cycles included).
"""

from __future__ import annotations

import random
import time

from z3 import And, Bool, If, Implies, Int, Not, Or, Solver, Sum, sat

LANES = 2  # vector length in the proofs (the rows use VLEN; any value works)
CAPS = {"valu": 1, "alu": 2, "load": 1, "store": 1, "flow": 1}
KINDS = ["valu", "valu_shift", "alu", "load", "store", "flow", "select", "const"]


def _graph(rng: random.Random, n: int, with_vb: bool):
    """Random DAG: list of (kind, deps). Optionally an eliminable vbroadcast."""
    ops = []
    for i in range(n):
        kind = rng.choice(KINDS)
        deps = sorted({rng.randrange(i) for _ in range(rng.randint(0, min(2, i)))}) if i else []
        ops.append([kind, deps])
    if with_vb and n >= 3:
        # scalar producer -> vbroadcast -> shiftable readers
        scalar = len(ops)
        ops.append(["alu", []])
        vb = len(ops)
        ops.append(["vb", [scalar]])
        for _ in range(rng.randint(1, 2)):
            ops.append(["valu_shift", [vb] + ([rng.randrange(scalar)] if rng.random() < 0.5 else [])])
    return ops


def static_paths(ops, tail_counts_own_cycle=False, vb_latency=0):
    """Heads and tails as makespan_rows.rs computes them (vbroadcast out-edges latency 0)."""
    n = len(ops)
    succs = [[] for _ in range(n)]
    for i, (_, deps) in enumerate(ops):
        for d in deps:
            succs[d].append(i)
    lat = [vb_latency if ops[i][0] == "vb" else 1 for i in range(n)]
    head = [0] * n
    for i in range(n):
        for d in ops[i][1]:
            head[i] = max(head[i], head[d] + lat[d])
    tail = [0] * n
    for i in reversed(range(n)):
        for s in succs[i]:
            tail[i] = max(tail[i], tail[s] + lat[i])
    if tail_counts_own_cycle:
        tail = [t + 1 for t in tail]
    return head, tail


def min_work(kind, engine):
    """Cheapest-encoding slots (makespan_rows.rs min_work)."""
    if kind in ("select", "const", "vb"):
        return 0
    table = {
        "valu": {"valu": 1, "joint": LANES},
        "valu_shift": {"joint": LANES},
        "alu": {"alu": 1, "joint": 1},
        "load": {"load": 1},
        "store": {"store": 1},
        "flow": {"flow": 1},
    }
    return table[kind].get(engine, 0)


def _cap(engine):
    return LANES * CAPS["valu"] + CAPS["alu"] if engine == "joint" else CAPS[engine]


def _encode(ops, horizon):
    """Z3 schedule of the model. Returns (solver, makespan, per-op dict of terms)."""
    n = len(ops)
    s = Solver()
    c = [Int(f"c{i}") for i in range(n)]
    shift = [Bool(f"shift{i}") for i in range(n)]
    lane = [[Int(f"l{i}_{j}") for j in range(LANES)] for i in range(n)]
    fma = [Bool(f"fma{i}") for i in range(n)]
    load_path = [Bool(f"ld{i}") for i in range(n)]
    elim = [Bool(f"elim{i}") for i in range(n)]
    readers = [[j for j in range(n) if i in ops[j][1]] for i in range(n)]
    for i, (kind, _) in enumerate(ops):
        s.add(c[i] >= 0, c[i] < horizon)
        for j in range(LANES):
            s.add(lane[i][j] >= 0, lane[i][j] < horizon)
        if kind != "valu_shift":
            s.add(Not(shift[i]))
        if kind != "select":
            s.add(Not(fma[i]))
        if kind != "const":
            s.add(Not(load_path[i]))
        if kind == "vb":
            s.add(elim[i] == And([shift[r] for r in readers[i]]) if readers[i] else Not(elim[i]))
        else:
            s.add(Not(elim[i]))
        if kind == "valu_shift":
            # Not shifted: lanes follow the issue cycle (the model leaves them free; this is
            # one of its solutions). Shifted: the issue cycle is the first lane.
            for j in range(LANES):
                s.add(Implies(Not(shift[i]), lane[i][j] == c[i]))
            s.add(Implies(shift[i], c[i] == lane[i][0]))
    # Result ready times.
    def ready(p):
        kind = ops[p][0]
        if kind == "select":
            return c[p] + If(fma[p], 2, 1)
        if kind == "valu_shift":
            return If(shift[p], lane_max(p) + 1, c[p] + 1)
        return c[p] + 1

    def lane_max(p):
        m = lane[p][0]
        for j in range(1, LANES):
            m = If(lane[p][j] > m, lane[p][j], m)
        return m

    for i, (kind, deps) in enumerate(ops):
        for p in deps:
            if ops[p][0] == "vb":
                scalar = ops[p][1][0]
                # Eliminated: lanes read the scalar; kept: they read the vbroadcast.
                for j in range(LANES):
                    s.add(Implies(And(shift[i], elim[p]), lane[i][j] >= c[scalar] + 1))
                    s.add(Implies(And(shift[i], Not(elim[p])), lane[i][j] >= c[p] + 1))
                s.add(Implies(Not(shift[i]), c[i] >= c[p] + 1))
                continue
            if kind == "valu_shift":
                for j in range(LANES):
                    lane_ready = If(shift[p], lane[p][j] + 1, ready(p)) if ops[p][0] == "valu_shift" else ready(p)
                    s.add(Implies(shift[i], lane[i][j] >= lane_ready))
                s.add(Implies(Not(shift[i]), c[i] >= ready(p)))
            else:
                s.add(c[i] >= ready(p))
    # Occupancy per engine per cycle.
    occupy = {e: [[] for _ in range(horizon + 1)] for e in CAPS}
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
    for e in CAPS:
        for t in range(horizon + 1):
            if occupy[e][t]:
                s.add(Sum(occupy[e][t]) <= CAPS[e])
    # Makespan: every occupied cycle and every multiply_add cycle variable is below it.
    M = Int("M")
    for i, (kind, _) in enumerate(ops):
        s.add(M >= c[i] + 1)
        if kind == "valu_shift":
            for j in range(LANES):
                s.add(M >= lane[i][j] + 1)
        if kind == "select":
            s.add(M >= c[i] + 2)
    terms = dict(c=c, shift=shift, lane=lane, fma=fma, load_path=load_path, elim=elim, lane_max=lane_max)
    return s, M, terms


def _actual_work(ops, i, engine, terms):
    """Slots op i uses on `engine` in the schedule, as rows.rs op_work counts them."""
    kind = ops[i][0]
    shift, fma, load_path = terms["shift"], terms["fma"], terms["load_path"]
    if kind == "select":
        return If(Not(fma[i]), 1, 0) if engine == "flow" else 0
    if kind == "const":
        if engine == "load":
            return If(load_path[i], 1, 0)
        if engine == "flow":
            return If(Not(load_path[i]), 1, 0)
        return 0
    if kind == "valu_shift":
        if engine == "valu":
            return If(Not(shift[i]), 1, 0)
        if engine == "alu":
            return If(shift[i], LANES, 0)
        if engine == "joint":
            return LANES
        return 0
    return min_work(kind, engine)


def _row_violations(ops, M, terms, tail_counts_own_cycle=False, bucket=2, vb_latency=0):
    """Disjunction of every row's violation."""
    n = len(ops)
    head, tail = static_paths(ops, tail_counts_own_cycle, vb_latency)
    viol = []
    engines = ["valu", "alu", "load", "store", "flow", "joint"]
    # Constant rows over (t, q).
    for e in engines:
        for t in range(max(head) + 1):
            for q in range(max(tail) + 1):
                w = sum(min_work(ops[i][0], e) for i in range(n) if head[i] >= t and tail[i] >= q)
                if w > 0:
                    viol.append(M < t + q + (w + _cap(e) - 1) // _cap(e))
    if tail_counts_own_cycle:
        return Or(viol) if viol else False
    # Literal rows over head windows and tail windows (all thresholds).
    for e in ["valu", "alu", "load", "flow"]:
        for side, values in (("head", head), ("tail", tail)):
            for t in range(1, max(values) + 1):
                work = [_actual_work(ops, i, e, terms) for i in range(n) if values[i] >= t]
                if work:
                    viol.append(_cap(e) * (M - t) < Sum([w if not isinstance(w, int) else w for w in work] + [0]))
    # Bucketed rows: prefix_end = max finish term, suffix_start = min start.
    c, shift, lane, fma = terms["c"], terms["shift"], terms["lane"], terms["fma"]
    lane_max = terms["lane_max"]
    buckets = max(head) // bucket + 1
    if buckets >= 2:
        def finish_terms(i):
            kind = ops[i][0]
            out = [(c[i], 0)]
            if kind == "valu_shift":
                out.append((If(shift[i], lane_max(i), c[i]), 0))  # completion
            if kind == "select":
                out.append((c[i] + 1, 1))  # multiply_add cycle
            return out

        def start(i):
            if ops[i][0] == "valu_shift":
                m = lane[i][0]
                for j in range(1, LANES):
                    m = If(lane[i][j] < m, lane[i][j], m)
                return If(shift[i], m, c[i])
            return c[i]

        def zmax(xs):
            m = xs[0]
            for x in xs[1:]:
                m = If(x > m, x, m)
            return m

        def zmin(xs):
            m = xs[0]
            for x in xs[1:]:
                m = If(x < m, x, m)
            return m

        for k in range(buckets - 1):
            prefix = [i for i in range(n) if head[i] // bucket <= k]
            if not prefix:
                continue
            fts = [ft for i in prefix for ft in finish_terms(i)]
            E = zmax([x for x, _ in fts])
            mt = min(max(tail[i] - early, 0) for i in prefix for _, early in finish_terms(i))
            viol.append(M < E + 1 + mt)
            for e in engines:
                w = Sum([_actual_work(ops, i, e, terms) for i in prefix] + [0])
                viol.append(_cap(e) * (E + 1) < w)
        for k in range(1, buckets):
            suffix = [i for i in range(n) if head[i] // bucket >= k]
            if not suffix:
                continue
            S = zmin([start(i) for i in suffix])
            viol.append(S < k * bucket)
            for e in engines:
                w = Sum([_actual_work(ops, i, e, terms) for i in suffix] + [0])
                viol.append(_cap(e) * (M - S) < w)
    return Or(viol) if viol else False


def _check(name, seeds, with_vb, tail_counts_own_cycle=False, bucket=2, vb_latency=0):
    start = time.time()
    found = None
    for seed in seeds:
        rng = random.Random(seed)
        ops = _graph(rng, rng.randint(3, 5), with_vb)
        horizon = 2 * len(ops) + 3
        s, M, terms = _encode(ops, horizon)
        s.add(_row_violations(ops, M, terms, tail_counts_own_cycle, bucket, vb_latency))
        if s.check() == sat:
            found = (seed, ops)
            break
    result = "sat" if found else "unsat"
    return name, result, time.time() - start, {"graphs": len(seeds), "counterexample": found}


def check_rows_sound(ctx=None):
    """MR1: no schedule violates a constant, literal or bucketed row (bucket of 2 cycles)."""
    return _check("MR1-rows-sound", range(60), with_vb=False)


def check_rows_sound_bucket_1(ctx=None):
    """MR2: as MR1 with buckets of one cycle."""
    return _check("MR2-rows-sound-bucket-1", range(60, 100), with_vb=False, bucket=1)


def check_rows_sound_with_vbroadcast(ctx=None):
    """MR3: as MR1 with an eliminable vbroadcast (latency-0 out-edges in the heads)."""
    return _check("MR3-rows-sound-with-vbroadcast", range(100, 140), with_vb=True)


def check_old_tail_definition_unsound(ctx=None):
    """MR1-old-tail-definition: counting the op's own cycle in its tail is unsound (SAT)."""
    return _check("MR1-old-tail-definition", range(60), with_vb=False, tail_counts_own_cycle=True)


def check_vbroadcast_latency_one_unsound(ctx=None):
    """MR3-vbroadcast-latency-1: heads through an eliminable vbroadcast at latency 1 are too
    large when it is eliminated (its shifted readers wait only for the scalar) (SAT)."""
    return _check("MR3-vbroadcast-latency-1", range(100, 200), with_vb=True, vb_latency=1)


MAKESPAN_ROW_CHECKS = [
    ("MR1-rows-sound", "Makespan rows (constant, literal, bucketed) hold in every schedule", check_rows_sound),
    ("MR2-rows-sound-bucket-1", "Makespan rows hold with one-cycle buckets", check_rows_sound_bucket_1),
    ("MR3-rows-sound-with-vbroadcast", "Makespan rows hold with an eliminable vbroadcast", check_rows_sound_with_vbroadcast),
    ("MR1-old-tail-definition", "A tail that counts the op's own cycle gives an unsound row (SAT)", check_old_tail_definition_unsound),
    ("MR3-vbroadcast-latency-1", "Latency 1 out of an eliminable vbroadcast gives an unsound row (SAT)", check_vbroadcast_latency_one_unsound),
]

EXPECT_SAT = {"MR1-old-tail-definition", "MR3-vbroadcast-latency-1"}


if __name__ == "__main__":
    import sys

    failed = 0
    for name, desc, fn in MAKESPAN_ROW_CHECKS:
        _, result, secs, stats = fn()
        want = "sat" if name in EXPECT_SAT else "unsat"
        failed += result != want
        print(f"{'PASS' if result == want else 'FAIL'} [{name}] {desc}: {result} ({secs:.1f}s) {stats.get('counterexample') or ''}")
    sys.exit(1 if failed else 0)
