"""Soundness of the static head-body-tail makespan bounds (scripts/static_bounds.py), by Z3.

Claim: for any emission, no schedule of its ops is shorter than ``static_makespan_bound``.
The bound is computed from the scheduler-JSON form of the ops (``ops_from_schedule_json``),
so the check covers the JSON adapter (work and latency rules) as well as the row arithmetic.

Reference model
---------------
Each random instance is a small DAG of SSA ops in scheduler-JSON form. A schedule assigns
cycles to *slots*; the makespan is one more than the last slot's cycle. The model is at
least as permissive as the CP-SAT model (scheduler_rs/src/cpsat_scheduler/model.rs), so a
bound no schedule here beats is also never beaten by a CP-SAT schedule:

    plain op (alu, load, store, flow, non-shiftable valu)  one slot on its engine at x.
    shiftable valu op   either one VALU slot at x, or VLEN ALU lanes at independent cycles
                        (the model's split shift). A shifted consumer's lane k waits for
                        lane k of a shifted producer; everything else waits for the last
                        lane.
    select_choice       vselect: one flow slot at x, consumers wait x + 1; or FMA: one VALU
                        slot at x + 1 (the subtract may be shared, so it is not charged),
                        consumers wait x + 2.
    delta select        a select_choice with a precomputed delta (``"delta": true`` on the
                        instance op; ssa_lower.rs gives it no separate FMA cycle): vselect as
                        above, or its one multiply_add on a VALU slot at x itself, consumers
                        wait x + 1 either way. Opt-in kind (``random_instance(kinds=...)``),
                        used by verification/window_rows_proof.py.
    const_choice        one slot at x on flow (add_imm), load or alu.
    vbroadcast          one VALU slot at x. If eliminable (one dependency on a non-VALU
                        producer, every consumer shiftable) it may be dropped: then it holds
                        no slot and its consumers wait only for its producer (x stays as a
                        virtual position, x >= producer + 1, consumers >= x).
    Precedence          a consumer's slots wait for the producer's result as above
                        (results visible next cycle).
    Capacity            per engine and cycle, slots <= capacity.

Machines are scaled down (VLEN 2, small capacities) so rows bind on tiny graphs and Z3
stays fast; the bound takes the machine as a parameter, so this exercises the same code.

Checks (result convention of verification/verify.py)
----------------------------------------------------
SB1-static-bound-sound     random instances: "a schedule with makespan < bound exists"
                           must be UNSAT on every one. Result "unsat" = sound on all.
SB2-own-cycle-tail-unsound the tail row written with the own-cycle tail and no -1 (as first
                           specified) on the stored counterexample: a schedule shorter
                           than that "bound" exists. Result "sat" = counterexample confirmed.
SB3-bound-attained         non-vacuity: on the stored tight instances the bound is attained
                           (a schedule of exactly that length exists) and beats the
                           whole-schedule capacity floor, and one cycle fewer is UNSAT.
                           Result "sat".
SB4-eliminated-vbroadcast-latency  the bound with latency 1 out of an eliminable vbroadcast
                           on the stored counterexample: a schedule that drops the broadcast
                           and reads the scalar directly beats it. Result "sat". Today's
                           CP-SAT model keeps the broadcast's precedence even when it drops
                           the broadcast, so latency 1 would hold for it, but not for a model
                           without that restriction; static_bounds uses latency 0, valid for
                           both.

Mutation check (not a stored check, run while writing this): adding one cycle to every row
that uses t > 0 or q > 0 is caught on 311 of 400 random instances, so SB1 has teeth.

Counterexamples and tight instances are data in static_bound_counterexamples.json.

Run:
    uv run python -m verification.static_bounds_proof            # fast set (proofs suite)
    uv run python -m verification.static_bounds_proof --slow     # 2,000 instances
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
import time

from z3 import And, Bool, If, Implies, Int, Not, Solver, Sum, sat, unsat

from scripts import static_bounds as sb

DATA_PATH = pathlib.Path(__file__).with_name("static_bound_counterexamples.json")

# Op kinds of a random instance, as (engine, opcode) in scheduler-JSON form.
KINDS = {
    "alu": ("alu", "+"),
    "load": ("load", "load"),
    "store": ("store", "store"),
    "flow": ("flow", "add_imm"),
    "valu": ("valu", "multiply_add"),
    "shift": ("valu", "^"),
    "select": ("flow", "select_choice"),
    "const": ("load", "const_choice"),
    "vbroadcast": ("valu", "vbroadcast"),
}


# Kinds a random instance may use only when asked (random_instance(kinds=...)), so the
# instances of SB1 stay what they were.
EXTRA_KINDS = {"delta_select": ("flow", "select_choice")}


def _op_kind(o):
    if o.get("delta"):
        return "delta_select"
    return next(k for k, v in KINDS.items() if v == (o["engine"], o["opcode"]))


def instance_json(kinds, deps):
    """A scheduler JSON holding only what the bound reads: ssa_ops with engine, opcode, deps
    (and "delta" on a delta select, which the bound does not read)."""
    all_kinds = {**KINDS, **EXTRA_KINDS}
    return {
        "ssa_ops": [
            {"id": i, "engine": all_kinds[k][0], "opcode": all_kinds[k][1], "dep_ids": sorted(deps[i]),
             **({"delta": True} if k == "delta_select" else {})}
            for i, k in enumerate(kinds)
        ],
        "ops": [],
        "run_options": {},
    }


def bound_for(inst, cap, vlen):
    ops = sb.ops_from_schedule_json(inst, "any", vlen=vlen)
    return sb.makespan_rows(ops, sb.row_caps(cap, vlen))


def schedule_exists(inst, cap, vlen, makespan):
    """Z3: does a reference-model schedule with all slots in [0, makespan) exist?"""
    ops = inst["ssa_ops"]
    by_id = {o["id"]: o for o in ops}
    kind = {o["id"]: _op_kind(o) for o in ops}
    consumers = {o["id"]: [] for o in ops}
    for o in ops:
        for p in o["dep_ids"]:
            consumers[p].append(o["id"])
    shiftable = {i for i in kind if kind[i] == "shift"}
    eliminable = sb._eliminable_vbroadcasts(by_id, consumers, shiftable)

    s = Solver()
    x = {i: Int(f"x{i}") for i in kind}
    shifted = {i: Bool(f"shift{i}") for i in shiftable}
    lanes = {i: [Int(f"l{i}_{k}") for k in range(vlen)] for i in shiftable}
    fma = {i: Bool(f"fma{i}") for i in kind if kind[i] == "select"}
    delta_fma = {i: Bool(f"dfma{i}") for i in kind if kind[i] == "delta_select"}
    cchoice = {i: Int(f"cc{i}") for i in kind if kind[i] == "const"}
    elim = {i: Bool(f"elim{i}") for i in eliminable}
    slots = []  # (engine, cycle expr, present)
    for i, k in kind.items():
        s.add(x[i] >= 0)
        if k == "shift":
            slots.append(("valu", x[i], Not(shifted[i])))
            for lane in lanes[i]:
                s.add(lane >= 0)
                slots.append(("alu", lane, shifted[i]))
        elif k == "select":
            slots.append(("flow", x[i], Not(fma[i])))
            slots.append(("valu", x[i] + 1, fma[i]))
        elif k == "delta_select":
            slots.append(("flow", x[i], Not(delta_fma[i])))
            slots.append(("valu", x[i], delta_fma[i]))
        elif k == "const":
            s.add(cchoice[i] >= 0, cchoice[i] <= 2)
            for c, eng in enumerate(("flow", "load", "alu")):
                slots.append((eng, x[i], cchoice[i] == c))
        elif i in eliminable:
            slots.append(("valu", x[i], Not(elim[i])))
        else:
            slots.append((by_id[i]["engine"], x[i], True))

    def ready(p, lane=None):
        """Earliest cycle a consumer position may take after producer p (lane: the
        consumer is shifted and this is its lane `lane`)."""
        if p in shiftable:
            last = lanes[p][0]
            for ln in lanes[p][1:]:
                last = If(ln > last, ln, last)
            done = lanes[p][lane] + 1 if lane is not None else last + 1
            return If(shifted[p], done, x[p] + 1)
        if p in fma:
            return x[p] + 1 + If(fma[p], 1, 0)
        if p in elim:
            return If(elim[p], x[p], x[p] + 1)
        return x[p] + 1

    for i in kind:
        for p in by_id[i]["dep_ids"]:
            if i in shiftable:
                s.add(Implies(Not(shifted[i]), x[i] >= ready(p)))
                for k_, lane in enumerate(lanes[i]):
                    s.add(Implies(shifted[i], lane >= ready(p, k_)))
            else:
                s.add(x[i] >= ready(p))
    for _, cyc, present in slots:
        s.add(Implies(present, cyc < makespan))
    for eng in sb.CAP:
        mine = [(cyc, present) for e, cyc, present in slots if e == eng]
        if not mine:
            continue
        for t in range(makespan):
            s.add(Sum([If(And(present, cyc == t), 1, 0) for cyc, present in mine]) <= cap[eng])
    return s.check()


def optimum(inst, cap, vlen, start):
    """Fewest cycles of any reference schedule, searching upward from `start`."""
    m = max(start, 1)
    while schedule_exists(inst, cap, vlen, m) != sat:
        m += 1
    return m


def random_instance(rng, kinds=None):
    """A random small DAG and machine. `kinds`: the op kinds to draw from (with repeats as
    weights); default every kind of KINDS."""
    pool = list(KINDS) if kinds is None else list(kinds)
    n = rng.randint(3, 7)
    kinds = [rng.choice(pool) for _ in range(n)]
    deps = []
    for i in range(n):
        d = {j for j in range(i) if rng.random() < 0.35}
        if kinds[i] == "vbroadcast" and d and rng.random() < 0.7:
            d = {max(d)}  # one dependency: a candidate for elimination
        deps.append(d)
    cap = {"valu": rng.choice((1, 2)), "alu": rng.choice((1, 2, 3)), "load": 1, "store": 1, "flow": 1}
    return instance_json(kinds, deps), cap


def check_random(count, seed=0):
    """SB1 body. Returns (violations, stats)."""
    rng = random.Random(seed)
    violations, tight, beats_floor, pair_binds = [], 0, 0, 0
    for _ in range(count):
        inst, cap = random_instance(rng)
        vlen = 2
        rows = bound_for(inst, cap, vlen)
        b = rows["bound"]
        if b <= 0:
            continue
        if schedule_exists(inst, cap, vlen, b - 1) != unsat:
            violations.append({"instance": inst, "cap": cap, "vlen": vlen, "bound": b})
            continue
        opt = optimum(inst, cap, vlen, b)
        tight += opt == b
        floor = max(r["floor"] for r in rows["rows"].values())
        beats_floor += b > floor
        pair_binds += rows["t"] > 0 and rows["q"] > 0 and b > max(
            max(r["head_row"]["bound"], r["tail_row"]["bound"]) for r in rows["rows"].values())
    stats = {"instances": count, "violations": len(violations), "bound_equals_optimum": tight,
             "bound_above_capacity_floor": beats_floor, "only_a_head_tail_pair_attains_bound": pair_binds}
    return violations, stats


def _result(name, ok_result, ok, t0, stats):
    return name, ok_result if ok else ("sat" if ok_result == "unsat" else "unsat"), time.perf_counter() - t0, stats


def check_sb1_sound(count=150, seed=0):
    t0 = time.perf_counter()
    violations, stats = check_random(count, seed)
    if violations:
        stats["first_violation"] = json.dumps(violations[0])
    return _result("SB1-static-bound-sound", "unsat", not violations, t0, stats)


def load_data():
    with open(DATA_PATH) as f:
        return json.load(f)


def own_cycle_tail_row(inst, cap, vlen):
    """The tail row as first written: makespan >= q' + ceil(W(tail' >= q') / cap), with tail'
    counting the op's own cycle (tail' = tail + 1). Unsound; kept to show why."""
    ops = sb.ops_from_schedule_json(inst, "any", vlen=vlen)
    tl = sb.tails(ops)
    caps = sb.row_caps(cap, vlen)
    best = 0
    for r, c in caps.items():
        for q in range(1, max(tl.values()) + 2):
            w = sum(o.work.get(r, 0) for o in ops if tl[o.id] + 1 >= q)
            if w:
                best = max(best, q + -(-w // c))
    return best


def check_sb2_own_cycle_tail_unsound():
    t0 = time.perf_counter()
    stats, ok = {}, True
    for case in load_data()["own_cycle_tail_counterexamples"]:
        wrong = own_cycle_tail_row(case["instance"], case["cap"], case["vlen"])
        res = schedule_exists(case["instance"], case["cap"], case["vlen"], wrong - 1)
        stats[case["name"]] = f"wrong row {wrong}, schedule of {wrong - 1} cycles: {res}"
        ok &= wrong == case["wrong_bound"] and res == sat
    return _result("SB2-own-cycle-tail-unsound", "sat", ok, t0, stats)


def check_sb3_attained():
    t0 = time.perf_counter()
    stats, ok = {}, True
    for case in load_data()["tight_instances"]:
        rows = bound_for(case["instance"], case["cap"], case["vlen"])
        floor = max(r["floor"] for r in rows["rows"].values())
        res = schedule_exists(case["instance"], case["cap"], case["vlen"], rows["bound"])
        stats[case["name"]] = f"bound {rows['bound']} ({rows['engine']}, t={rows['t']}, q={rows['q']}), floor {floor}: {res}"
        ok &= rows["bound"] == case["bound"] and rows["bound"] > floor and res == sat
        ok &= schedule_exists(case["instance"], case["cap"], case["vlen"], rows["bound"] - 1) == unsat
    return _result("SB3-bound-attained", "sat", ok, t0, stats)


def bound_with_unit_latencies(inst, cap, vlen):
    """The bound with latency 1 on every edge, including the edge out of an eliminable
    vbroadcast (unsound when the eliminated broadcast's consumers read the scalar directly)."""
    ops = [sb.BoundOp(o.id, tuple((p, 1) for p, _ in o.deps), o.work)
           for o in sb.ops_from_schedule_json(inst, "any", vlen=vlen)]
    return sb.makespan_rows(ops, sb.row_caps(cap, vlen))["bound"]


def check_sb4_vbroadcast_latency():
    t0 = time.perf_counter()
    stats, ok = {}, True
    for case in load_data()["vbroadcast_latency_counterexamples"]:
        wrong = bound_with_unit_latencies(case["instance"], case["cap"], case["vlen"])
        right = bound_for(case["instance"], case["cap"], case["vlen"])["bound"]
        res = schedule_exists(case["instance"], case["cap"], case["vlen"], wrong - 1)
        stats[case["name"]] = f"latency-1 bound {wrong}, latency-0 bound {right}, schedule of {wrong - 1} cycles: {res}"
        ok &= wrong == case["wrong_bound"] and right < wrong and res == sat
    return _result("SB4-eliminated-vbroadcast-latency", "sat", ok, t0, stats)


STATIC_BOUND_CHECKS = [
    ("SB1-static-bound-sound", "Static head/tail makespan rows: no schedule beats them (150 random DAGs)",
     check_sb1_sound),
    ("SB2-own-cycle-tail-unsound", "Tail row with own-cycle tail and no -1 is unsound (SAT = counterexample)",
     check_sb2_own_cycle_tail_unsound),
    ("SB3-bound-attained", "Stored tight instances: bound attained and above the capacity floor (SAT)",
     check_sb3_attained),
    ("SB4-eliminated-vbroadcast-latency",
     "Latency 1 out of an eliminable vbroadcast is unsound (SAT = counterexample)",
     check_sb4_vbroadcast_latency),
]
EXPECT_SAT = ("SB2-own-cycle-tail-unsound", "SB3-bound-attained", "SB4-eliminated-vbroadcast-latency")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--slow", action="store_true", help="2,000 random instances instead of 150")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    failed = 0
    for name, desc, fn in STATIC_BOUND_CHECKS:
        if name == "SB1-static-bound-sound":
            _, res, el, stats = check_sb1_sound(2000 if args.slow else 150, args.seed)
        else:
            _, res, el, stats = fn()
        want = "sat" if name in EXPECT_SAT else "unsat"
        print(f"[{name}] {desc}: {'PASS' if res == want else 'FAIL'} ({res}) in {el:.1f}s")
        for k, v in stats.items():
            print(f"    {k}: {v}")
        failed += res != want
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
