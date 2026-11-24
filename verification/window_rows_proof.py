"""Soundness of the window rows of the structure loop (scripts/emission_loop.py), by Z3.

Claim (WR1): for any set of ops, no schedule is shorter than

    max over (t, q) of  t + q + window_floor(ops with head >= t and tail >= q)

where head and tail are those of scripts/static_bounds.py (minimum latency over every
encoding) and ``window_floor`` (scripts/emission_structure.py) is the fewest cycles that hold
the window's ops with every choice resolved jointly: selects on flow or the VALU, shiftable
vector ops on the VALU or as VLEN scalar lanes; ConstChoices and eliminable vbroadcasts are
not charged. It is the per-engine head-body-tail rows of static_bounds.py (proved in
verification/static_bounds_proof.py) with the engines coupled through the choices, so it can
be stronger than each of them.

The reference schedule model is static_bounds_proof.schedule_exists (more permissive than
the CP-SAT model; see that module), on the same random small instances.

Checks (result convention of verification/verify.py):
WR1-window-rows-sound      "a schedule with makespan < bound exists" must be UNSAT on every
                           instance. Result "unsat" = sound on all.
WR2-window-rows-have-teeth non-vacuity: adding 1 to every window with t > 0 or q > 0 is
                           violated on some instance (a schedule exists below it), and on
                           some instance the window row is above every per-engine row.
                           Result "sat".

What this does NOT prove: that the per-class work, heads and tails the loop learns from
built plans are lower bounds for plans not built. That part is observed (every build is
checked against the table it was solved with).

Run:
    uv run --extra verify python -m verification.window_rows_proof
"""

from __future__ import annotations

import argparse
import collections
import random
import sys
import time

from z3 import sat, unsat

from scripts import emission_structure as es
from scripts import static_bounds as sb
from scripts.emission_loop import op_window_category
from verification import static_bounds_proof as sbp

VLEN = 2


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
    violations, above_rows, teeth = [], 0, 0
    for _ in range(count):
        inst, cap = sbp.random_instance(rng)
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
    return violations, {"instances": count, "violations": len(violations),
                        "window_row_above_per_engine_rows": above_rows, "bumped_row_violated": teeth}


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
    ("WR1-window-rows-sound", "Window rows of the structure loop: no schedule beats them (150 random DAGs)",
     check_wr1_sound),
    ("WR2-window-rows-have-teeth", "Window rows bumped by 1 are violated; some beat every per-engine row (SAT)",
     check_wr2_teeth),
]
EXPECT_SAT = ("WR2-window-rows-have-teeth",)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
    violations, stats = check(args.count, args.seed)
    wr1 = "unsat" if not violations else "sat"
    wr2 = "sat" if stats["bumped_row_violated"] and stats["window_row_above_per_engine_rows"] else "unsat"
    dt = time.perf_counter() - t0
    print(f"WR1-window-rows-sound: {wr1} (expected unsat) {stats} [{dt:.1f}s]")
    print(f"WR2-window-rows-have-teeth: {wr2} (expected sat)")
    if violations:
        print("first violation:", violations[0])
    return 0 if wr1 == "unsat" and wr2 == "sat" else 1


if __name__ == "__main__":
    sys.exit(main())
