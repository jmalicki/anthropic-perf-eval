"""Soundness of the window rows of the structure loop (scripts/emission_loop.py), by Z3.

Claim: for any set of ops, no schedule is shorter than

    max over (t, q) of  t + q + window_floor(ops with head >= t and tail >= q)

where head and tail are those of scripts/static_bounds.py (minimum latency over every
encoding) and ``window_floor`` (scripts/emission_structure.py) is the fewest cycles that hold
the window's ops with every choice resolved jointly: selects on flow or the VALU, shiftable
vector ops on the VALU or as VLEN scalar lanes; ConstChoices and eliminable vbroadcasts are
not charged. It is the per-engine head-body-tail rows of static_bounds.py with the engines
coupled through the choices, so it can be stronger than each of them.

What is proved and what is checked:

1. Containment (argued by hand, not by Z3): in any schedule, every slot of an op with
   head >= t and tail >= q lies in cycles [t, M - 1 - q]. By induction along the edges: a
   slot is never before its op's earliest position, every edge out of a slot has latency
   >= 1 to every slot of the consumer it constrains (the eliminable vbroadcast, which holds
   no slot, has latency 0 in the heads and tails), and that holds for every encoding: a
   vselect (flow at x, consumers x + 1), a two-op FMA select (VALU at x + 1, consumers
   x + 2), a one-op delta select (VALU at x, consumers x + 1; ssa_lower.rs gives it no
   separate FMA cycle), and a shifted op's lanes (a shifted consumer's lane k after lane k
   of a shifted producer, anything else after its last lane).
2. Coupling (WR3, proved by Z3 for all counts and spans): ops whose slots fit in S cycles
   can be counted per engine with some split of selects (flow / VALU) and of shiftable
   vector ops (VALU / VLEN ALU lanes) within S * capacity; whenever such a split exists,
   ``_window_fits(counts, S)`` holds, and it is monotone in S. So window_floor <= M - t - q.
   Proved over symbolic non-negative counts and span for the real machine (VLEN 8) and the
   scaled machines of the random instances; the Z3 transcription of ``_window_fits`` is
   checked against the Python function on random points.
3. Composition (WR1, a randomized check, not a proof): on random 3-7 op DAGs at VLEN 2,
   including delta selects, "a reference schedule with makespan < bound exists" is UNSAT.
   The reference model is static_bounds_proof.schedule_exists, which is at least as
   permissive as the CP-SAT model for every op kind it has (with delta selects added for
   this check). WR1c draws instances where the window row is above every per-engine row
   (the coupling binds), so the part 2 adds is exercised, not just the per-engine rows.

Checks (result convention of verification/verify.py):
WR1-window-rows-sound      random instances (150 in the suite): must be UNSAT on every one.
WR1c-window-rows-coupled   random instances on which the coupling binds (40 in the suite):
                           UNSAT on every one.
WR2-window-rows-have-teeth non-vacuity: adding 1 to every window with t > 0 or q > 0 is
                           violated on some instance, and on some instance the window row
                           is above every per-engine row. Result "sat".
WR3-window-coupling-lemma  part 2 for all counts and spans: the negation is UNSAT.

What this does NOT prove: that the per-class work, heads and tails the loop learns from
built plans are lower bounds for plans not built. That part is observed (every build is
checked against the table it was solved with), so a window row fed by the table is not a
proved bound for an unbuilt plan.

Run:
    uv run --extra verify python -m verification.window_rows_proof [--count 2000 --coupled 400]
"""

from __future__ import annotations

import argparse
import collections
import random
import sys
import time

from z3 import And, If, Implies, Int, Not, Solver, sat, unsat

from scripts import emission_structure as es
from scripts import static_bounds as sb
from scripts.emission_loop import op_window_category
from verification import static_bounds_proof as sbp

VLEN = 2
# Op kinds of the random instances: every kind of the reference model plus the delta select.
WINDOW_KINDS = tuple(sbp.KINDS) + tuple(sbp.EXTRA_KINDS)
# Kinds weighted towards what couples engines (selects, shiftable vector ops, flow, ALU).
COUPLING_KINDS = ("select", "delta_select", "shift", "shift", "flow", "alu", "valu", "load")


def window_rows_bound(inst, cap, vlen=VLEN, bump=0):
    """The window rows for a scheduler-JSON instance; `bump` is added to every window with
    t > 0 or q > 0 (a deliberately wrong row, for WR2)."""
    ops = sb.ops_from_schedule_json(inst, "any", vlen=vlen)
    heads, tails = sb.heads(ops), sb.tails(ops)
    ssa = {o["id"]: o for o in inst["ssa_ops"]}
    consumers = collections.defaultdict(list)
    for o in ssa.values():
        for p in o["dep_ids"]:
            consumers[p].append(o["id"])
    shiftable = {i for i, o in ssa.items() if o["engine"] == "valu" and o["opcode"] in sb.SHIFTABLE_OPCODES}
    eliminable = sb._eliminable_vbroadcasts(ssa, consumers, shiftable)
    cats = {i: op_window_category(o, shiftable, eliminable) for i, o in ssa.items()}
    best = 0
    for t in sorted(set(heads.values())):
        for q in sorted(set(tails.values())):
            c = collections.Counter(cats[i] for i in ssa if cats[i] and heads[i] >= t and tails[i] >= q)
            if not c:
                continue
            value = t + q + es.window_floor(c, cap, vlen) + (bump if (t or q) else 0)
            best = max(best, value)
    return best


def check(count=150, seed=0):
    rng = random.Random(seed)
    violations, above_rows, teeth, deltas = [], 0, 0, 0
    for _ in range(count):
        inst, cap = sbp.random_instance(rng, WINDOW_KINDS)
        deltas += any(o.get("delta") for o in inst["ssa_ops"])
        b = window_rows_bound(inst, cap)
        if b <= 0:
            continue
        if sbp.schedule_exists(inst, cap, VLEN, b - 1) != unsat:
            violations.append({"instance": inst, "cap": cap, "bound": b})
            continue
        rows = sbp.bound_for(inst, cap, VLEN)["bound"]
        above_rows += b > rows
        bumped = window_rows_bound(inst, cap, bump=1)
        if bumped > b and sbp.schedule_exists(inst, cap, VLEN, bumped - 1) == sat:
            teeth += 1
    return violations, {"instances": count, "with_delta_select": deltas, "violations": len(violations),
                        "window_row_above_per_engine_rows": above_rows, "bumped_row_violated": teeth}


def check_coupled(count=40, seed=0, max_draws=200_000):
    """WR1 on `count` instances where the window row is above every per-engine row (drawn by
    rejection from COUPLING_KINDS)."""
    rng = random.Random(seed + 10_007)
    violations, found, draws, deltas = [], 0, 0, 0
    while found < count and draws < max_draws:
        draws += 1
        inst, cap = sbp.random_instance(rng, COUPLING_KINDS)
        b = window_rows_bound(inst, cap)
        if b <= 0 or b <= sbp.bound_for(inst, cap, VLEN)["bound"]:
            continue
        found += 1
        deltas += any(o.get("delta") for o in inst["ssa_ops"])
        if sbp.schedule_exists(inst, cap, VLEN, b - 1) != unsat:
            violations.append({"instance": inst, "cap": cap, "bound": b})
    return violations, {"coupled_instances": found, "draws": draws, "with_delta_select": deltas,
                        "violations": len(violations)}


# ----------------------------------------------------------------------------
# WR3: the coupling inequality for all counts and spans
# ----------------------------------------------------------------------------
LEMMA_MACHINES = [dict(es.CAP, vlen=es.VLEN)] + [
    {"valu": v, "alu": a, "load": 1, "store": 1, "flow": 1, "vlen": VLEN} for v in (1, 2) for a in (1, 2, 3)]


def _z3_window_fits(c, span, cap, vlen):
    """Transcription of emission_structure._window_fits over Z3 integers."""
    def zmin(a, b):
        return If(a <= b, a, b)

    on_flow = zmin(c["select"], cap["flow"] * span - c["flow"])
    vector = c["vec_fixed"] + c["vec_shiftable"] + c["select"] - on_flow
    alu_room = cap["alu"] * span - c["alu"]
    over = If(vector - cap["valu"] * span > 0, vector - cap["valu"] * span, 0)
    return And(span >= 0, c["load"] <= cap["load"] * span, c["store"] <= cap["store"] * span,
               c["flow"] <= cap["flow"] * span, alu_room >= 0,
               over <= zmin(c["vec_shiftable"], alu_room / vlen))


def lemma_holds(machine):
    """(aggregate split exists at S => fits(S)) and (fits(S) => fits(S + 1)), for all counts."""
    vlen = machine["vlen"]
    cap = {k: v for k, v in machine.items() if k != "vlen"}
    c = {k: Int(k) for k in es.WINDOW_CATEGORIES}
    span, sf, sh = Int("span"), Int("select_on_flow"), Int("shifted")
    nonneg = And(*[v >= 0 for v in c.values()], span >= 0)
    split = And(0 <= sf, sf <= c["select"], 0 <= sh, sh <= c["vec_shiftable"],
                c["load"] <= cap["load"] * span, c["store"] <= cap["store"] * span,
                c["flow"] + sf <= cap["flow"] * span,
                c["vec_fixed"] + c["vec_shiftable"] + c["select"] - sf - sh <= cap["valu"] * span,
                c["alu"] + vlen * sh <= cap["alu"] * span)
    fits = _z3_window_fits(c, span, cap, vlen)
    fits_next = _z3_window_fits(c, span + 1, cap, vlen)
    s = Solver()
    s.add(nonneg, Not(And(Implies(split, fits), Implies(fits, fits_next))))
    return s.check()


def transcription_matches(machine, n=2000, seed=0):
    """The Z3 transcription agrees with the Python _window_fits on random points."""
    rng = random.Random(seed)
    vlen = machine["vlen"]
    cap = {k: v for k, v in machine.items() if k != "vlen"}
    for _ in range(n):
        c = {k: rng.randint(0, 40) for k in es.WINDOW_CATEGORIES}
        span = rng.randint(0, 12)
        s = Solver()
        s.add(_z3_window_fits(c, span, cap, vlen))
        if (s.check() == sat) != es._window_fits(c, span, cap, vlen):
            return False
    return True


def check_wr1c_coupled(count=40, seed=0):
    t0 = time.perf_counter()
    violations, stats = check_coupled(count, seed)
    if violations:
        stats["first_violation"] = str(violations[0])
    ok = not violations and stats["coupled_instances"] == count
    return "WR1c-window-rows-coupled", "unsat" if ok else "sat", time.perf_counter() - t0, stats


def check_wr3_lemma():
    t0 = time.perf_counter()
    stats, ok = {}, True
    for m in LEMMA_MACHINES:
        res = lemma_holds(m)
        tr = transcription_matches(m, n=300)
        stats[str(m)] = f"lemma negation {res}, transcription matches: {tr}"
        ok &= res == unsat and tr
    return "WR3-window-coupling-lemma", "unsat" if ok else "sat", time.perf_counter() - t0, stats


def check_wr1_sound(count=150, seed=0):
    t0 = time.perf_counter()
    violations, stats = check(count, seed)
    if violations:
        stats["first_violation"] = str(violations[0])
    return "WR1-window-rows-sound", "unsat" if not violations else "sat", time.perf_counter() - t0, stats


def check_wr2_teeth(count=150, seed=0):
    t0 = time.perf_counter()
    _, stats = check(count, seed)
    ok = stats["bumped_row_violated"] > 0 and stats["window_row_above_per_engine_rows"] > 0
    return "WR2-window-rows-have-teeth", "sat" if ok else "unsat", time.perf_counter() - t0, stats


WINDOW_ROW_CHECKS = [
    ("WR1-window-rows-sound",
     "Window rows of the structure loop: no schedule beats them (randomized: 150 random DAGs, delta selects included)",
     check_wr1_sound),
    ("WR1c-window-rows-coupled",
     "Window rows where the engine coupling binds: no schedule beats them (randomized: 40 such DAGs)",
     check_wr1c_coupled),
    ("WR3-window-coupling-lemma",
     "Engine coupling of the window floor: any per-engine split within S cycles passes _window_fits (all counts)",
     check_wr3_lemma),
    ("WR2-window-rows-have-teeth", "Window rows bumped by 1 are violated; some beat every per-engine row (SAT)",
     check_wr2_teeth),
]
EXPECT_SAT = ("WR2-window-rows-have-teeth",)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, default=150)
    ap.add_argument("--coupled", type=int, default=40, help="instances on which the coupling binds (WR1c)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
    violations, stats = check(args.count, args.seed)
    wr1 = "unsat" if not violations else "sat"
    wr2 = "sat" if stats["bumped_row_violated"] and stats["window_row_above_per_engine_rows"] else "unsat"
    print(f"WR1-window-rows-sound: {wr1} (expected unsat) {stats} [{time.perf_counter() - t0:.1f}s]")
    print(f"WR2-window-rows-have-teeth: {wr2} (expected sat)")
    if violations:
        print("first violation:", violations[0])
    _, wr1c, dt, cstats = check_wr1c_coupled(args.coupled, args.seed)
    print(f"WR1c-window-rows-coupled: {wr1c} (expected unsat) {cstats} [{dt:.1f}s]")
    _, wr3, dt, lstats = check_wr3_lemma()
    print(f"WR3-window-coupling-lemma: {wr3} (expected unsat) [{dt:.1f}s]")
    for k, v in lstats.items():
        print(f"  {k}: {v}")
    return 0 if wr1 == "unsat" and wr2 == "sat" and wr1c == "unsat" and wr3 == "unsat" else 1


if __name__ == "__main__":
    sys.exit(main())
