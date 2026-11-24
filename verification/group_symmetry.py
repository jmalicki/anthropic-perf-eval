"""Group symmetry breaking (`--cpsat-symmetry-rows checked`, cpsat_scheduler/group_symmetry.rs).

The model contains several identical vector groups: copies of one dependency graph that
share some ops (constants, cached tree values) and compete for the same slots. Swapping two
identical groups maps schedules to schedules of the same makespan. The checked rows put one
anchor op per group in a chain, `anchor(G_1) <= anchor(G_2) <= ...`. That excludes no
makespan: in any schedule, sort the groups by their anchor's cycle and give the k-th group
the k-th group's cycles.

Each check builds random small instances: m identical groups of a random template graph
(ops on VALU, ALU, Load), shared producers feeding every group, and an optional shared
reader of corresponding ops of every group. Slot capacities are small, so the groups
contend. Z3 computes the shortest makespan without rows, then asks for a schedule of that
makespan that also satisfies the rows.

* GS1-anchor-chain (expected UNSAT, i.e. never worse): the checked rows never raise the
  optimum.
* GS2-two-anchors (expected SAT): two anchors per group, ordered independently (what one
  emission-declared stream per round amounts to when the greedy orders the groups
  differently in two rounds), can raise the optimum.
* GS3-within-group (expected SAT): an order between two different ops of one group (what the
  declared `sym_ord` chains over a whole level of a select tree include) can raise the
  optimum.
"""

from __future__ import annotations

import random
import time

from z3 import If, Int, Optimize, Solver, Sum, unknown, unsat

CAPS = {"valu": 1, "alu": 1, "load": 1}


def _instance(rng: random.Random):
    """(template ops [(engine, deps)], shared ops [engine], m groups, shared reader?)."""
    n = rng.randint(2, 4)
    template = []
    for i in range(n):
        deps = sorted({rng.randrange(i) for _ in range(rng.randint(0, min(2, i)))}) if i else []
        template.append((rng.choice(list(CAPS)), deps))
    n_shared = rng.randint(0, 2)
    shared = [rng.choice(list(CAPS)) for _ in range(n_shared)]
    # each template op may read shared op s (same for every group)
    shared_reads = {i: sorted({rng.randrange(n_shared)}) if n_shared and rng.random() < 0.5 else [] for i in range(n)}
    reader = rng.random() < 0.4
    return template, shared, shared_reads, rng.randint(2, 3), reader


def _encode(inst, horizon):
    template, shared, shared_reads, m, reader = inst
    s = []
    c = {}
    engine = {}
    for k, e in enumerate(shared):
        c[("s", k)] = Int(f"s{k}")
        engine[("s", k)] = e
    for g in range(m):
        for i, (e, _) in enumerate(template):
            c[(g, i)] = Int(f"g{g}_{i}")
            engine[(g, i)] = e
    rpos = len(template) - 1
    if reader:
        c["r"] = Int("r")
        engine["r"] = "alu"
    for v in c.values():
        s += [v >= 0, v < horizon]
    for g in range(m):
        for i, (_, deps) in enumerate(template):
            for d in deps:
                s.append(c[(g, i)] >= c[(g, d)] + 1)
            for k in shared_reads[i]:
                s.append(c[(g, i)] >= c[("s", k)] + 1)
    if reader:
        for g in range(m):
            s.append(c["r"] >= c[(g, rpos)] + 1)
    for e, cap in CAPS.items():
        for t in range(horizon):
            terms = [If(v == t, 1, 0) for key, v in c.items() if engine[key] == e]
            if terms:
                s.append(Sum(terms) <= cap)
    M = Int("M")
    s += [M >= v + 1 for v in c.values()]
    return s, M, c


def _optimum(constraints, M):
    o = Optimize()
    o.add(constraints)
    o.minimize(M)
    outcome = o.check()
    if outcome == unknown:
        return None, o.reason_unknown()
    if outcome == unsat:
        raise RuntimeError("generated group-symmetry instance has no baseline schedule")
    return o.model()[M].as_long(), None


def _check(name, seeds, rows_of):
    """Find a counterexample (SAT), rule them out on the checked seeds (UNSAT), or report UNKNOWN."""
    start = time.time()
    for seed in seeds:
        rng = random.Random(seed)
        inst = _instance(rng)
        horizon = sum(1 for _ in inst[0]) * inst[3] + len(inst[1]) + 2
        cons, M, c = _encode(inst, horizon)
        best, reason = _optimum(cons, M)
        if reason is not None:
            return name, "unknown", time.time() - start, {
                "unresolved_seed": seed, "stage": "baseline optimum", "reason": reason,
            }
        rows = rows_of(inst, c, rng)
        if rows is None:
            continue
        s = Solver()
        s.add(cons + rows + [M <= best])
        outcome = s.check()
        if outcome == unsat:
            return name, "sat", time.time() - start, {"counterexample_seed": seed, "instance": inst, "optimum": best}
        if outcome == unknown:
            return name, "unknown", time.time() - start, {
                "unresolved_seed": seed, "stage": "schedule with ordering rows", "reason": s.reason_unknown(),
            }
    return name, "unsat", time.time() - start, {"instances": len(seeds)}


def _anchor_chain(inst, c, rng):
    template, _, _, m, _ = inst
    k = rng.randrange(len(template))
    return [c[(g, k)] <= c[(g + 1, k)] for g in range(m - 1)]


def _two_anchors(inst, c, rng):
    template, _, _, m, _ = inst
    if len(template) < 2:
        return None
    k1, k2 = rng.sample(range(len(template)), 2)
    # first anchor ascending, second descending: two streams sorted by different greedy orders
    return [c[(g, k1)] <= c[(g + 1, k1)] for g in range(m - 1)] + [c[(g + 1, k2)] <= c[(g, k2)] for g in range(m - 1)]


def _within_group(inst, c, rng):
    template, _, _, m, _ = inst
    if len(template) < 2:
        return None
    p, q = rng.sample(range(len(template)), 2)
    return [c[(0, p)] <= c[(0, q)]]


def check_anchor_chain(ctx=None):
    """GS1: one anchor per group in a chain never raises the optimum."""
    return _check("GS1-anchor-chain", range(80), _anchor_chain)


def check_two_anchors_can_cut(ctx=None):
    """GS2: two independently ordered anchors per group can raise the optimum (SAT)."""
    return _check("GS2-two-anchors", range(300), _two_anchors)


def check_within_group_order_can_cut(ctx=None):
    """GS3: an order between two ops of the same group can raise the optimum (SAT)."""
    return _check("GS3-within-group", range(300), _within_group)


GROUP_SYMMETRY_CHECKS = [
    ("GS1-anchor-chain", "One anchor per identical group in a chain keeps the optimum", check_anchor_chain),
    ("GS2-two-anchors", "Two independently ordered anchors per group can raise the optimum (SAT)", check_two_anchors_can_cut),
    ("GS3-within-group", "An order between two ops of one group can raise the optimum (SAT)", check_within_group_order_can_cut),
]

EXPECT_SAT = {"GS2-two-anchors", "GS3-within-group"}


if __name__ == "__main__":
    import sys

    failed = 0
    for name, desc, fn in GROUP_SYMMETRY_CHECKS:
        _, result, secs, stats = fn()
        want = "sat" if name in EXPECT_SAT else "unsat"
        failed += result != want
        print(f"{'PASS' if result == want else 'FAIL'} [{name}] {desc}: {result} ({secs:.1f}s) {stats}")
    sys.exit(1 if failed else 0)
