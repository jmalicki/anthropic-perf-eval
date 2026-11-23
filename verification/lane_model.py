"""Z3 proofs that the lane models of the CP-SAT scheduler lose no schedule.

A shiftable vector op ``o`` has a cycle variable ``c_o`` (its time as one vector op), a
shift literal ``shift_o``, eight lane variables ``a_o,0 .. a_o,7`` (its times as eight
scalar ops when shifted) and ``completion_o`` (at least every lane and ``c_o``). See
``scheduler_rs/src/cpsat_scheduler/model.rs`` section 6 and ``CpsatLaneModel`` in
``config.rs``. Two restrictions of the default ("free") lane model are proved here:

* **sorted**: ``a_o,0 <= a_o,1 <= ... <= a_o,7`` for every shiftable op.
* **pinned**: ``!shift_o => a_o,i = c_o`` for every lane, and the pair of lane
  precedences ``shift_d => a_o,i >= a_d,i + 1``, ``!shift_d => a_o,i >= c_d + 1`` is
  replaced by the single unconditional ``a_o,i >= a_d,i + 1``.

Each check maps a feasible schedule of the free model to a schedule of the restricted model
(sort the lanes of every shiftable op; set the lanes of every unshifted op to its cycle;
both) and encodes a **violation**: the free schedule satisfies every constraint and the
mapped one breaks at least one. **UNSAT = no schedule is lost** (good). The mapping leaves
every cycle variable, completion, liveness start, shift literal and the multiset of
lane times of every op unchanged, so the makespan, the tie-breaker objective and the
per-cycle scalar-slot usage are unchanged too: the restricted model keeps an optimal
schedule of the free one (a dominance argument, not just "some schedule survives").

The constraint shapes are those of ``model.rs`` (constraint names in comments). All
variables are unbounded integers, so a check is a proof for every value, not for a small
domain. Sorting is done by the fixed compare-exchange network ``NETWORK`` applied to every
op at once; ``LANE-sort-dag`` shows that one such step keeps every row of a DAG carrying all
lane shapes, for each step of the network, and ``LANE-network-sorts`` that the steps end
sorted, so by induction a sorted schedule exists. The composition over a whole model is
per row: each row of the model has one of the shapes in the DAG, which
``tests/test_cpsat_lane_model.py`` checks on exported models of real kernels.

Shapes:

* lane-wise precedence with the **same lane index** on both sides
  (``shift_alu_prec_*_on``; with pinning the unconditional ``shift_alu_prec_*``);
* a bound that is **the same for every lane** of one op (``shift_alu_prec_*_off``,
  ``shift_alu_prec_*_valu``, ``select_prec_*_fma``/``_opt``, ``vb_elim_*_prec_*``,
  ``scratch_shifted_start_ge_alu_*``, ``completion_ge_alu_*``);
* the scalar cumulative, where every lane interval has demand 1 and the op's shift literal
  as presence.

``LANE-sort-lane-specific-producers`` is the shape that is *not* safe (expected SAT): a
reader whose lane i waits for its own scalar producer p_i, as the loads of a gather would
if their lanes were tied to the lanes of a shifted reader. The model has no such
constraint today (the gather's loads bound every lane of a reader equally, and the
readers of a decomposed vector wait for ``completion``); the structural test fails if one
appears.

Run:
    uv run python -m verification.verify -t LANE-sort-dag
    uv run python -m verification.lane_model
"""

from __future__ import annotations

from z3 import And, Bool, If, Implies, Int, Not, Solver, Sum

K = 8  # lanes per vector op (VLEN)
ALU_CAP = 12


def _stats(solver) -> dict:
    try:
        s = solver.statistics()
        return {str(k): s.get_key_value(k) for k in s.keys()}
    except Exception:
        return {}


def _lanes(name: str) -> list:
    return [Int(f"{name}_{i}") for i in range(K)]


# Batcher's odd-even merge sort for 8 inputs: 19 compare-exchange pairs (i, j), i < j.
NETWORK = [
    (0, 1), (2, 3), (4, 5), (6, 7),
    (0, 2), (1, 3), (4, 6), (5, 7),
    (1, 2), (5, 6),
    (0, 4), (1, 5), (2, 6), (3, 7),
    (2, 4), (3, 5),
    (1, 2), (3, 4), (5, 6),
]


def _min(a, b):
    return If(a <= b, a, b)


def _max(a, b):
    return If(a <= b, b, a)


def network_sort(xs: list) -> list:
    """``xs`` sorted ascending by the compare-exchange network ``NETWORK`` (Z3 terms)."""
    out = list(xs)
    for i, j in NETWORK:
        out[i], out[j] = _min(out[i], out[j]), _max(out[i], out[j])
    return out


def network_sorts_every_input() -> bool:
    """0-1 principle: a comparator network sorts every input iff it sorts every 0-1 input."""
    for bits in range(1 << K):
        out = [(bits >> i) & 1 for i in range(K)]
        for i, j in NETWORK:
            out[i], out[j] = min(out[i], out[j]), max(out[i], out[j])
        if any(out[i] > out[i + 1] for i in range(K - 1)):
            return False
    return True


def _sorted_copy(solver: Solver, name: str, xs: list) -> list:
    """``xs`` sorted ascending, as terms of the network (``solver`` and ``name`` unused)."""
    del solver, name
    return network_sort(xs)


def _count_at(lanes: list, t) -> object:
    return Sum([If(a == t, 1, 0) for a in lanes])


def _result(name: str, solver: Solver) -> tuple[str, str, float, dict]:
    return (name, str(solver.check()), 0.0, _stats(solver))


# ----------------------------------------------------------------------------
# Single shapes
# ----------------------------------------------------------------------------


def check_sort_lanewise_precedence(ctx=None):
    """Sorting both ops keeps ``a_o,i >= a_d,i + delta`` for every lane (any delta).

    Violation: the unsorted lanes satisfy it lane by lane and the sorted ones do not.
    """
    solver = Solver(ctx=ctx)
    x, y, delta = _lanes("o"), _lanes("d"), Int("delta")
    solver.add([x[i] >= y[i] + delta for i in range(K)])
    sx, sy = _sorted_copy(solver, "o", x), _sorted_copy(solver, "d", y)
    solver.add(Not(And([sx[i] >= sy[i] + delta for i in range(K)])))
    return _result("LANE-sort-lanewise-precedence", solver)


def check_sort_uniform_bounds(ctx=None):
    """Sorting keeps a bound that is the same for every lane, and the min and max.

    ``lo <= a_i <= hi`` (domain, ``a_i >= c_dep + 1``, ``start <= a_i``), ``completion >= a_i``.
    """
    solver = Solver(ctx=ctx)
    x, lo, hi, completion = _lanes("o"), Int("lo"), Int("hi"), Int("completion")
    solver.add([And(lo <= x[i], x[i] <= hi, completion >= x[i]) for i in range(K)])
    sx = _sorted_copy(solver, "o", x)
    solver.add(Not(And([And(lo <= sx[i], sx[i] <= hi, completion >= sx[i]) for i in range(K)])))
    return _result("LANE-sort-uniform-bounds", solver)


def check_sort_cycle_counts(ctx=None):
    """Each compare-exchange of the network keeps the number of lanes in every cycle, so
    sorting keeps the scalar-slot usage (composition over the network's steps)."""
    x, t = _lanes("o"), Int("t")
    for i, j in NETWORK:
        solver = Solver(ctx=ctx)
        solver.add(_count_at(x, t) != _count_at(_compare_exchange(x, i, j), t))
        result = str(solver.check())
        if result != "unsat":
            return ("LANE-sort-cycle-counts", f"{result} at ({i}, {j})", 0.0, {})
    return ("LANE-sort-cycle-counts", "unsat", 0.0, {})


def check_sort_lane_specific_producers(ctx=None):
    """Counterexample (expected SAT): lane i of a reader waits for its own producer p_i.

    Sorting the reader's lanes alone breaks ``a_i >= p_i + 1``. This shape must not
    appear in the model while lanes are sorted per op.
    """
    solver = Solver(ctx=ctx)
    x, p = _lanes("o"), _lanes("p")
    solver.add([x[i] >= p[i] + 1 for i in range(K)])
    sx = _sorted_copy(solver, "o", x)
    solver.add(Not(And([sx[i] >= p[i] + 1 for i in range(K)])))
    return _result("LANE-sort-lane-specific-producers", solver)


def check_pin_unconditional_precedence(ctx=None):
    """With pinned lanes, the unconditional lane precedence equals the enforced pair.

    Premises (the model's other rows): ``c_o >= completion_d + 1`` (consumer_after_completion),
    ``completion_d >= c_d`` and ``completion_d >= a_d,i``, and pinning of both ops.
    Violation: the pair and the single row disagree in either direction.
    """
    solver = Solver(ctx=ctx)
    a_o, a_d = _lanes("o"), _lanes("d")
    c_o, c_d, comp_d = Int("c_o"), Int("c_d"), Int("completion_d")
    shift_o, shift_d = Bool("shift_o"), Bool("shift_d")
    solver.add(c_o >= comp_d + 1, comp_d >= c_d)
    for i in range(K):
        solver.add(comp_d >= a_d[i])
        solver.add(Implies(Not(shift_o), a_o[i] == c_o), Implies(Not(shift_d), a_d[i] == c_d))
    pair = And([And(Implies(shift_d, a_o[i] >= a_d[i] + 1), Implies(Not(shift_d), a_o[i] >= c_d + 1))
                for i in range(K)])
    single = And([a_o[i] >= a_d[i] + 1 for i in range(K)])
    solver.add(pair != single)
    return _result("LANE-pin-unconditional-precedence", solver)


# ----------------------------------------------------------------------------
# A small DAG with every lane shape of the model at once
# ----------------------------------------------------------------------------


def _dag_rows(ctx=None):
    """Variables of a small DAG and a function giving every lane row of the model on it.

    Ops: ``p`` (non-shiftable producer), ``q`` -> ``vb`` (an eliminable broadcast),
    ``sel`` (a select whose multiply-add path adds one cycle, ``x``), shiftable ``d``,
    shiftable ``o`` reading ``p``, ``d``, ``sel`` and ``vb``, shiftable ``e`` reading ``o``.
    ``n_t`` native scalar ops share the scalar engine with the lanes at an arbitrary cycle
    ``t``. Every shift literal is free.
    """
    names = ["d", "o", "e"]
    c = {n: Int(f"c_{n}") for n in names + ["p", "q", "vb", "sel"]}
    shift = {n: Bool(f"shift_{n}") for n in names}
    lanes = {n: _lanes(f"a_{n}") for n in names}
    comp = {n: Int(f"completion_{n}") for n in names}
    start = {n: Int(f"start_{n}") for n in names}
    asap = {n: Int(f"asap_{n}") for n in names}
    alap = {n: Int(f"alap_{n}") for n in names}
    x = Int("fma_extra")  # select_fma_extra: 1 when the select takes the multiply-add path
    elim = Bool("eliminate_vb")
    t, n_t = Int("t"), Int("n_t")
    deps_shiftable = {"o": ["d"], "e": ["o"], "d": []}  # lane-wise precedences

    def rows(a, pinned: bool, sorted_: bool):
        """Every row of the model that mentions lanes, plus the cycle rows they rely on."""
        out = [x >= 0, x <= 1, Implies(elim, shift["o"])]  # vb_elim_*_shift_ge_c*
        for n in names:
            out += [comp[n] >= c[n], asap[n] <= comp[n], comp[n] <= alap[n]]  # completion_ge_start
            out.append(Implies(Not(shift[n]), start[n] == c[n]))  # scratch_shifted_start_eq_cycle
            # A sound ASAP: the cycle variable of a feasible schedule is never below it (T3/T5).
            out.append(c[n] >= asap[n])
            for i in range(K):
                out += [asap[n] <= a[n][i], a[n][i] <= alap[n]]  # lane domain
                out.append(comp[n] >= a[n][i])  # completion_ge_alu
                out.append(start[n] <= a[n][i])  # scratch_shifted_start_ge_alu
                if pinned:
                    out.append(Implies(Not(shift[n]), a[n][i] == c[n]))  # lane_pin
                if sorted_ and i + 1 < K:
                    out.append(a[n][i] <= a[n][i + 1])  # lane_order
            for dep in deps_shiftable[n]:
                out.append(c[n] >= comp[dep] + 1)  # consumer_after_completion
                for i in range(K):
                    if pinned:
                        out.append(a[n][i] >= a[dep][i] + 1)  # shift_alu_prec (unconditional)
                    else:
                        out.append(Implies(shift[dep], a[n][i] >= a[dep][i] + 1))  # _on
                        out.append(Implies(Not(shift[dep]), a[n][i] >= c[dep] + 1))  # _off
        # o's other producers: p (plain), sel (select, +x), vb (eliminable broadcast of q).
        out += [c["o"] >= c["p"] + 1, c["o"] >= c["sel"] + 1 + x, c["o"] >= c["vb"] + 1,
                c["vb"] >= c["q"] + 1]
        for i in range(K):
            out.append(a["o"][i] >= c["p"] + 1)  # shift_alu_prec_*_valu
            out.append(a["o"][i] >= c["sel"] + 1 + x)  # select_prec_*_opt
            out.append(Implies(elim, a["o"][i] >= c["q"] + 1))  # vb_elim_*_prec_elim
            out.append(Implies(Not(elim), a["o"][i] >= c["vb"] + 1))  # vb_elim_*_prec_keep
        # Scalar engine at cycle t: native ops plus the present lane intervals (demand 1).
        used = n_t + Sum([If(shift[n], _count_at(a[n], t), 0) for n in names])
        out += [n_t >= 0, used <= ALU_CAP]
        return And(out)

    return names, lanes, shift, c, rows


def _compare_exchange(lanes: list, i: int, j: int) -> list:
    out = list(lanes)
    out[i], out[j] = _min(lanes[i], lanes[j]), _max(lanes[i], lanes[j])
    return out


def _dag_pin(ctx=None):
    """Free schedule feasible; lanes of unshifted ops set to their cycle; violation of the pinned model."""
    solver = Solver(ctx=ctx)
    names, lanes, shift, c, rows = _dag_rows(ctx)
    solver.add(rows(lanes, pinned=False, sorted_=False))
    pinned = {n: [If(shift[n], lanes[n][i], c[n]) for i in range(K)] for n in names}
    solver.add(Not(rows(pinned, pinned=True, sorted_=False)))
    return solver


def _dag_compare_exchange_steps(pinned: bool, ctx=None) -> str:
    """One compare-exchange of lanes (i, j), applied to every shiftable op at once, keeps
    every row; checked for every pair of ``NETWORK``. Applying the whole network this way
    sorts every op (``LANE-network-sorts``), so by induction a sorted schedule exists.

    Returns "unsat" if every step is UNSAT (no step loses a schedule), else the first other
    result with the pair.
    """
    for i, j in NETWORK:
        solver = Solver(ctx=ctx)
        names, lanes, _, _, rows = _dag_rows(ctx)
        solver.add(rows(lanes, pinned=pinned, sorted_=False))
        stepped = {n: _compare_exchange(lanes[n], i, j) for n in names}
        solver.add(Not(rows(stepped, pinned=pinned, sorted_=False)))
        result = str(solver.check())
        if result != "unsat":
            return f"{result} at ({i}, {j})"
    return "unsat"


def check_network_sorts(ctx=None):
    """The network sorts every integer input (violation: some output pair out of order)."""
    solver = Solver(ctx=ctx)
    out = network_sort(_lanes("x"))
    solver.add(Not(And([out[i] <= out[i + 1] for i in range(K - 1)])))
    assert network_sorts_every_input(), "0-1 principle check failed"
    return _result("LANE-network-sorts", solver)


def check_dag_sort(ctx=None):
    return ("LANE-sort-dag", _dag_compare_exchange_steps(pinned=False, ctx=ctx), 0.0, {})


def check_dag_pin(ctx=None):
    return _result("LANE-pin-dag", _dag_pin(ctx))


def check_dag_pin_sort(ctx=None):
    return ("LANE-sorted-pinned-dag", _dag_compare_exchange_steps(pinned=True, ctx=ctx), 0.0, {})


LANE_CHECKS = [
    ("LANE-sort-lanewise-precedence", "Sorted lanes keep same-index lane precedences", check_sort_lanewise_precedence),
    ("LANE-sort-uniform-bounds", "Sorted lanes keep per-op uniform bounds, min and max", check_sort_uniform_bounds),
    ("LANE-sort-cycle-counts", "Sorted lanes keep the lanes per cycle (scalar capacity)", check_sort_cycle_counts),
    (
        "LANE-sort-lane-specific-producers",
        "Per-lane producers break per-op sorting (SAT = counterexample; shape must stay absent)",
        check_sort_lane_specific_producers,
    ),
    (
        "LANE-pin-unconditional-precedence",
        "Pinned lanes: unconditional lane precedence equals the enforced on/off pair",
        check_pin_unconditional_precedence,
    ),
    ("LANE-network-sorts", "The compare-exchange network sorts every input", check_network_sorts),
    (
        "LANE-sort-dag",
        "Free model, all lane shapes: each compare-exchange step on every op keeps feasibility",
        check_dag_sort,
    ),
    ("LANE-pin-dag", "Free schedule -> lanes of unshifted ops at their cycle stays feasible", check_dag_pin),
    (
        "LANE-sorted-pinned-dag",
        "Pinned model, all lane shapes: each compare-exchange step keeps feasibility",
        check_dag_pin_sort,
    ),
]

EXPECT_SAT = {"LANE-sort-lane-specific-producers"}


def main() -> int:
    failures = 0
    for name, desc, check in LANE_CHECKS:
        _, result, _, _ = check()
        want = "sat" if name in EXPECT_SAT else "unsat"
        ok = result == want
        failures += not ok
        print(f"{'PASS' if ok else 'FAIL'} {name}: {result} (want {want}) - {desc}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
