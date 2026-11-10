#!/usr/bin/env python3
"""Toy experiments comparing CP-SAT encodings for docs/CPSAT_VARIABLE_MODELLING_REVIEW.md.

Standalone: needs only `ortools` (9.15 used). Does not import the project.

    uv run python -P scripts/cpsat_encoding_experiments.py            # all experiments
    uv run python -P scripts/cpsat_encoding_experiments.py --only A   # one experiment
    uv run python -P scripts/cpsat_encoding_experiments.py --time-limit 30

(`-P` is required: scripts/bisect.py would otherwise shadow the standard-library `bisect`.)

The instance is a miniature of the kernel: G independent groups, each a load followed by
S stages; a stage is three vector ops `a = f(x); b = g(x); x' = a ^ b`. Every vector op can
run as one vector-ALU slot (capacity 6) or as eight scalar-ALU slots (capacity 12). Loads
have capacity 2. Every value is 8 words; `x` has two readers, so "which reader is last" is
not known in advance.

Encodings of the vector/scalar alternative:

* ``current``  - replica of model.rs: main cycle var, shift Boolean, 8 lane vars, completion
                 var, 1 + 8 optional intervals, precedence stated per mode with enforcement
                 literals, inactive lanes left free.
* ``unified K`` - K lane-block vars (K in 1, 2, 8) with demand 8/K each; inactive blocks are
                 pinned to block 0; precedence is stated once, block-wise, with no
                 enforcement literal.
* ``sorted K``  - ``unified K`` plus t_0 <= t_1 <= ... <= t_{K-1} per op (a dominance: sorting
                 the blocks of every op preserves block-wise precedence and capacity use), so
                 "first write" is t_0 and "last read/write" is t_{K-1} with no min/max variable.

Liveness encodings:

* ``single``   - replica of model.rs: end = main cycle var of ONE designated reader + 1.
* ``all``      - end >= (every block of every reader) + 1.
* ``allmax``   - end == max(...) via AddMaxEquality.
* ``exact``    - (sorted only) start = t_0 of the producer; end = t_last(reader) + 1 directly
                 when there is one reader, else AddMaxEquality over the readers' last blocks.

Each run reports the solver's result and, independently, the true peak liveness recomputed
from the solution (first write to last read), so an under-approximating model is visible.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field

from ortools.sat.python import cp_model

VLEN = 8
VALU_CAP, ALU_CAP, LOAD_CAP = 6, 12, 2


@dataclass
class Op:
    idx: int
    kind: str  # "load" | "vec"
    deps: list[int] = field(default_factory=list)


def make_instance(groups: int, stages: int) -> list[Op]:
    ops: list[Op] = []
    for _ in range(groups):
        x = len(ops)
        ops.append(Op(x, "load"))
        for _ in range(stages):
            a = len(ops)
            ops.append(Op(a, "vec", [x]))
            b = len(ops)
            ops.append(Op(b, "vec", [x]))
            x = len(ops)
            ops.append(Op(x, "vec", [a, b]))
    return ops


@dataclass
class Built:
    model: cp_model.CpModel
    makespan: cp_model.IntVar
    blocks: dict[int, list[cp_model.IntVar]]  # op -> execution-time vars (lanes / blocks)
    shift: dict[int, cp_model.IntVar]
    main: dict[int, cp_model.IntVar]
    scale: int
    aux: list[cp_model.IntVar]  # auxiliary vars (for the hint experiment)


def build(
    ops: list[Op],
    enc: str = "current",
    k: int = 8,
    tiebreak: bool = False,
    liveness: str | None = None,
    budget: int = 0,
    makespan_interval: bool = False,
    redundant_capacity: bool = True,
    sorted_blocks: bool = False,
) -> Built:
    m = cp_model.CpModel()
    n = len(ops)
    readers: dict[int, list[int]] = {o.idx: [] for o in ops}
    for o in ops:
        for d in o.deps:
            readers[d].append(o.idx)
    depth = [0] * n
    for o in ops:
        depth[o.idx] = 1 + max((depth[d] for d in o.deps), default=-1)
    n_vec = sum(o.kind == "vec" for o in ops)
    horizon = max(depth) + 1 + math.ceil(n_vec / VALU_CAP) + 4

    main: dict[int, cp_model.IntVar] = {}
    shift: dict[int, cp_model.IntVar] = {}
    blocks: dict[int, list[cp_model.IntVar]] = {}
    completion: dict[int, cp_model.IntVar] = {}
    aux: list[cp_model.IntVar] = []
    valu_iv, valu_dem, alu_iv, alu_dem, load_iv = [], [], [], [], []

    for o in ops:
        c = m.new_int_var(depth[o.idx], horizon - 1, f"c{o.idx}")
        main[o.idx] = c
        if o.kind == "load":
            blocks[o.idx] = [c]
            load_iv.append(m.new_fixed_size_interval_var(c, 1, f"ld{o.idx}"))
            continue
        s = m.new_bool_var(f"shift{o.idx}")
        shift[o.idx] = s
        valu_iv.append(m.new_optional_fixed_size_interval_var(c, 1, s.Not(), f"v{o.idx}"))
        valu_dem.append(1)
        if enc == "current":
            lanes = [m.new_int_var(depth[o.idx], horizon - 1, f"a{o.idx}_{i}") for i in range(VLEN)]
            comp = m.new_int_var(depth[o.idx], horizon - 1, f"comp{o.idx}")
            aux += lanes + [comp]
            m.add(comp >= c)
            for a in lanes:
                m.add(comp >= a)
                alu_iv.append(m.new_optional_fixed_size_interval_var(a, 1, s, ""))
                alu_dem.append(1)
            blocks[o.idx] = lanes
            completion[o.idx] = comp
        else:
            # Block 0 is the main var. Extra blocks exist only for k > 1 and are pinned to
            # block 0 when the op is not shifted (inactive alternative is fixed).
            bl = [c] + [m.new_int_var(depth[o.idx], horizon - 1, f"t{o.idx}_{j}") for j in range(1, k)]
            aux += bl[1:]
            if sorted_blocks:
                for t_prev, t_next in zip(bl, bl[1:]):
                    m.add(t_prev <= t_next)
                if k > 1:
                    m.add(bl[-1] == c).only_enforce_if(s.Not())
            else:
                for t in bl[1:]:
                    m.add(t == c).only_enforce_if(s.Not())
            for t in bl:
                alu_iv.append(m.new_optional_fixed_size_interval_var(t, 1, s, ""))
                alu_dem.append(VLEN // k)
            blocks[o.idx] = bl

    # ---- precedence -------------------------------------------------------------------
    for o in ops:
        for d in o.deps:
            if enc == "current":
                if d in shift:
                    m.add(main[o.idx] >= completion[d] + 1)  # 5c
                    for i, a in enumerate(blocks[o.idx]):  # 5b
                        m.add(a >= blocks[d][i] + 1).only_enforce_if(shift[d])
                        m.add(a >= main[d] + 1).only_enforce_if(shift[d].Not())
                else:
                    m.add(main[o.idx] >= main[d] + 1)  # 5
                    for a in blocks[o.idx]:
                        m.add(a >= main[d] + 1)
            else:
                # One statement for every mode combination: block j of the reader after
                # block j of the producer (a load has a single block).
                bo, bd = blocks[o.idx], blocks[d]
                for j, t in enumerate(bo):
                    m.add(t >= bd[j if len(bd) > 1 else 0] + 1)

    # ---- makespan ---------------------------------------------------------------------
    cap_lb = math.ceil(n_vec / (VALU_CAP + ALU_CAP / VLEN))
    makespan = m.new_int_var(1, horizon, "makespan")
    for o in ops:
        m.add(makespan >= main[o.idx] + 1)
        if enc == "current":
            if o.idx in completion:
                m.add(makespan >= completion[o.idx] + 1)
        else:
            for t in blocks[o.idx][1:]:
                m.add(makespan >= t + 1)
    if redundant_capacity:
        s_sum = sum(shift.values())
        m.add(VALU_CAP * makespan >= n_vec - s_sum)
        m.add(ALU_CAP * makespan >= VLEN * s_sum)
        m.add(makespan >= cap_lb)

    if makespan_interval:
        # OR-Tools rcpsp_sat.py pattern: an always-present interval [makespan, horizon+1)
        # with demand == capacity in every engine cumulative.
        size = m.new_int_var(1, horizon + 1, "mk_size")
        aux.append(size)
        mk = m.new_interval_var(makespan, size, horizon + 1, "mk")
        valu_iv.append(mk), valu_dem.append(VALU_CAP)
        alu_iv.append(mk), alu_dem.append(ALU_CAP)
        m.add_cumulative(load_iv + [mk], [1] * len(load_iv) + [LOAD_CAP], LOAD_CAP)
    else:
        m.add_cumulative(load_iv, [1] * len(load_iv), LOAD_CAP)
    m.add_cumulative(valu_iv, valu_dem, VALU_CAP)
    m.add_cumulative(alu_iv, alu_dem, ALU_CAP)

    # ---- liveness ---------------------------------------------------------------------
    if liveness:
        ivs = []
        for o in ops:
            rs = readers[o.idx]
            if not rs:
                continue
            if liveness == "exact":
                assert sorted_blocks
                st = blocks[o.idx][0]
                lasts = [blocks[r][-1] for r in rs]
                if len(lasts) == 1:
                    end = lasts[0] + 1
                else:
                    end = m.new_int_var(1, horizon + 1, f"end{o.idx}")
                    aux.append(end)
                    m.add_max_equality(end, [t + 1 for t in lasts])
                size = m.new_int_var(1, horizon + 1, f"sz{o.idx}")
                aux.append(size)
                ivs.append(m.new_interval_var(st, size, end, f"live{o.idx}"))
                continue
            # start
            if o.idx in shift and len(blocks[o.idx]) > 1:
                st = m.new_int_var(0, horizon, f"st{o.idx}")
                aux.append(st)
                for t in blocks[o.idx]:
                    m.add(st <= t)
                if enc == "current":
                    m.add(st == main[o.idx]).only_enforce_if(shift[o.idx].Not())
            else:
                st = main[o.idx]
            # end
            if liveness == "single":
                end = main[max(rs)] + 1  # designated reader, its MAIN var (as model.rs)
            else:
                reads = [t for r in rs for t in blocks[r]]
                if enc == "current":
                    reads += [main[r] for r in rs]
                e = m.new_int_var(1, horizon + 1, f"end{o.idx}")
                aux.append(e)
                if liveness == "allmax":
                    m.add_max_equality(e, [t + 1 for t in reads])
                else:
                    for t in reads:
                        m.add(e >= t + 1)
                end = e
            size = m.new_int_var(1, horizon + 1, f"sz{o.idx}")
            aux.append(size)
            ivs.append(m.new_interval_var(st, size, end, f"live{o.idx}"))
        m.add_cumulative(ivs, [VLEN] * len(ivs), budget)

    # ---- objective --------------------------------------------------------------------
    scale = 1
    if tiebreak:
        terms = []
        for o in ops:
            if o.idx in shift and len(blocks[o.idx]) > 1:
                if enc == "current":
                    ch = m.new_int_var(0, horizon - 1, f"chosen{o.idx}")
                    aux.append(ch)
                    m.add(ch == completion[o.idx]).only_enforce_if(shift[o.idx])
                    m.add(ch == main[o.idx]).only_enforce_if(shift[o.idx].Not())
                else:
                    ch = m.new_int_var(0, horizon - 1, f"fin{o.idx}")
                    aux.append(ch)
                    for t in blocks[o.idx]:
                        m.add(ch >= t)
                terms.append(ch)
            else:
                terms.append(main[o.idx])
        while scale <= n * horizon:
            scale *= 10
        m.minimize(scale * makespan + sum(terms))
    else:
        m.minimize(makespan)
    return Built(m, makespan, blocks, shift, main, scale, aux)


class Trace(cp_model.CpSolverSolutionCallback):
    def __init__(self, mk):
        super().__init__()
        self.mk, self.t0, self.best, self.t_best = mk, time.time(), None, None

    def on_solution_callback(self):
        v = self.value(self.mk)
        if self.best is None or v < self.best:
            self.best, self.t_best = v, time.time() - self.t0


def true_peak(ops, b: Built, solver) -> int:
    """Peak live words recomputed from the solution: first write .. last read, inclusive."""
    ev = []
    readers = {o.idx: [] for o in ops}
    for o in ops:
        for d in o.deps:
            readers[d].append(o.idx)

    def times(i):
        if i in b.shift and solver.value(b.shift[i]):
            return [solver.value(t) for t in b.blocks[i]]
        return [solver.value(b.main[i])]

    for o in ops:
        if not readers[o.idx]:
            continue
        start = min(times(o.idx))
        last = max(max(times(r)) for r in readers[o.idx])
        ev += [(start, VLEN), (last + 1, -VLEN)]
    cur = peak = 0
    for _, d in sorted(ev):
        cur += d
        peak = max(peak, cur)
    return peak


def solve(ops, b: Built, time_limit: float, project_params: bool = True, log=None):
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = time_limit
    s.parameters.num_workers = 8
    s.parameters.random_seed = 1
    if project_params:  # the settings model.rs uses today
        s.parameters.interleave_search = True
        s.parameters.linearization_level = 2
        s.parameters.cp_model_probing_level = 2
        s.parameters.randomize_search = True
    if b.scale > 1:
        s.parameters.absolute_gap_limit = b.scale // 3
    if log is not None:
        s.parameters.log_search_progress = True
        s.parameters.log_to_stdout = False
        s.log_callback = log.append
    cb = Trace(b.makespan)
    t0 = time.time()
    st = s.solve(b.model, cb)
    wall = time.time() - t0
    res = {
        "status": s.status_name(st),
        "makespan": cb.best,
        "bound": math.ceil(s.best_objective_bound / b.scale - 1e-9),
        "t_best": cb.t_best,
        "wall": wall,
        "vars": len(b.model.proto.variables),
        "cons": len(b.model.proto.constraints),
        "conflicts": s.num_conflicts,
        "peak": true_peak(ops, b, s) if cb.best is not None else None,
        "solver": s,
    }
    return res


def row(label, r, extra=""):
    tb = f"{r['t_best']:.1f}" if r["t_best"] is not None else "-"
    print(
        f"| {label} | {r['vars']} | {r['cons']} | {r['status']} | {r['makespan']} | {r['bound']} "
        f"| {tb} | {r['wall']:.1f} | {r['peak']} {extra}|",
        flush=True,
    )


HEADER = (
    "| variant | vars | constraints | status | makespan | bound | s to best makespan | wall s | true peak words |\n"
    "|---|---|---|---|---|---|---|---|---|"
)


def exp_a(tl):
    print("\n## A. Encoding of the vector/scalar alternative (no liveness, makespan objective)\n")
    for g, s in [(4, 6), (8, 8), (16, 6), (32, 6)]:
        ops = make_instance(g, s)
        n_vec = sum(o.kind == "vec" for o in ops)
        print(f"\nInstance G={g} S={s}: {len(ops)} ops, critical path {2 * s + 1}, "
              f"capacity floor free lanes {math.ceil(n_vec / 7.5)}, co-scheduled {math.ceil(n_vec / 7)}\n")
        print(HEADER)
        row("current (8 free lanes, per-mode precedence)", solve(ops, build(ops, "current"), tl))
        for k in (8, 2, 1):
            row(f"unified, {k} block(s) of demand {VLEN // k}", solve(ops, build(ops, "unified", k), tl))


def exp_b(tl):
    print("\n## B. Tie-breaker in the objective (sum of completion cycles) on/off\n")
    ops = make_instance(8, 8)
    print(HEADER)
    for enc, k in (("current", 8), ("unified", 2)):
        for tb in (False, True):
            row(f"{enc} k={k} tiebreak={tb}", solve(ops, build(ops, enc, k, tiebreak=tb), tl))


def exp_c(tl):
    print("\n## C. Liveness encodings under a binding scratch budget\n")
    g, s = 8, 8
    ops = make_instance(g, s)
    for budget in (8 * 3 * g, 8 * 2 * g + 16):
        print(f"\nG={g} S={s}, budget {budget} words (a model is unsound if true peak > budget)\n")
        print(HEADER)
        row("current + single designated reader", solve(ops, build(ops, "current", liveness="single", budget=budget), tl))
        row("current + all readers (>=)", solve(ops, build(ops, "current", liveness="all", budget=budget), tl))
        row("unified k=2 + all readers (>=)", solve(ops, build(ops, "unified", 2, liveness="all", budget=budget), tl))
        row("unified k=2 + all readers (max eq)", solve(ops, build(ops, "unified", 2, liveness="allmax", budget=budget), tl))
        row("unified k=1 + all readers (>=)", solve(ops, build(ops, "unified", 1, liveness="all", budget=budget), tl))


def exp_d(tl):
    print("\n## D. Makespan interval in each cumulative vs redundant linear capacity rows\n")
    print("Bound-only test: no hint; `bound` is the best lower bound on the makespan at the time limit.\n")
    ops = make_instance(16, 6)
    print(HEADER)
    for red in (True, False):
        for mk in (False, True):
            b = build(ops, "unified", 2, makespan_interval=mk, redundant_capacity=red)
            row(f"redundant_capacity={red} makespan_interval={mk}", solve(ops, b, tl))
    print("\nSame, default CP-SAT parameters (no interleave; bound workers allowed):\n")
    print(HEADER)
    for red in (True, False):
        for mk in (False, True):
            b = build(ops, "unified", 2, makespan_interval=mk, redundant_capacity=red)
            row(f"redundant_capacity={red} makespan_interval={mk}", solve(ops, b, tl, project_params=False))


def exp_f(tl):
    print("\n## F. Sorted blocks (dominance) with exact liveness endpoints\n")
    for g, s in [(8, 8), (16, 6)]:
        ops = make_instance(g, s)
        print(f"\nG={g} S={s}, no liveness\n")
        print(HEADER)
        for k in (8, 2):
            row(f"sorted, {k} blocks", solve(ops, build(ops, "unified", k, sorted_blocks=True), tl))
    g, s = 8, 8
    ops = make_instance(g, s)
    for budget in (8 * 3 * g, 8 * 2 * g + 16):
        print(f"\nG={g} S={s}, budget {budget} words\n")
        print(HEADER)
        for k in (8, 2):
            b = build(ops, "unified", k, liveness="exact", budget=budget, sorted_blocks=True)
            row(f"sorted, {k} blocks + exact liveness", solve(ops, b, tl))
        b = build(ops, "unified", 1, liveness="exact", budget=budget, sorted_blocks=True)
        row("1 block + exact liveness", solve(ops, b, tl))


def exp_e(tl):
    print("\n## E. Hint completeness (current encoding)\n")
    ops = make_instance(16, 6)
    b0 = build(ops, "current", tiebreak=True)
    r0 = solve(ops, b0, tl)
    s0 = r0["solver"]
    vals = {v.index: s0.value(v) for v in
            [b0.makespan, *b0.main.values(), *b0.shift.values(), *b0.aux]}
    aux_idx = {v.index for v in b0.aux}
    print("| hint | solver message | first solution line |\n|---|---|---|")
    for label, drop in (("complete", set()), ("auxiliary vars (lanes, completion, chosen) unhinted", aux_idx)):
        b = build(ops, "current", tiebreak=True)
        for v in [b.makespan, *b.main.values(), *b.shift.values(), *b.aux]:
            if v.index not in drop:
                b.model.add_hint(v, vals[v.index])
        log: list[str] = []
        solve(ops, b, min(tl, 5.0), log=log)
        msg = next((l.strip() for l in log if "solution hint" in l), "(none)")
        first = next((l.strip() for l in log if l.startswith("#1 ")), "(no solution)")
        print(f"| {label} | {msg} | `{first}` |", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-limit", type=float, default=20.0)
    ap.add_argument("--only", default="ABCDEF")
    a = ap.parse_args()
    import ortools

    print(f"ortools {ortools.__version__}, time limit {a.time_limit}s per run, 8 workers")
    for key, fn in (("A", exp_a), ("B", exp_b), ("C", exp_c), ("D", exp_d), ("E", exp_e), ("F", exp_f)):
        if key in a.only:
            fn(a.time_limit)


if __name__ == "__main__":
    main()
