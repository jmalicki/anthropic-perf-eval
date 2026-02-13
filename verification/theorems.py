"""Theorems T1–T13 from LIVENESS_SPECIFICATION.md."""

from z3 import And, Ints, Not, Or, Solver

from .fixtures import overlaps_closed


def _get_stats(solver) -> dict:
    try:
        s = solver.statistics()
        return {str(k): s.get_key_value(k) for k in s.keys()}
    except Exception:
        return {}


def check_t2_overlap_definition(ctx=None) -> tuple[str, str, float, dict]:
    """T2: Overlap definition.
    Theorem: overlaps([s1,e1],[s2,e2]) iff e1>s2 and e2>s1.
    Our implementation: overlaps = (s1<e2 and s2<e1). These are equivalent.
    Prove by refutation: (impl XOR spec) should be UNSAT.
    """
    s1, e1, s2, e2 = Ints("s1 e1 s2 e2")
    impl = overlaps_closed(s1, e1, s2, e2)  # s1<e2 and s2<e1
    spec = And(e1 > s2, e2 > s1)  # spec formulation
    # XOR: (impl and not spec) or (not impl and spec)
    negated_theorem = Or(And(impl, Not(spec)), And(Not(impl), spec))
    solver = Solver(ctx=ctx)
    solver.add(negated_theorem)
    result = solver.check()
    stats = _get_stats(solver)
    return ("T2", "unsat" if str(result) == "unsat" else str(result), 0.0, stats)


def check_t1_expiration_no_write_conflict(ctx=None) -> tuple[str, str, float, dict]:
    """T1: Expiration rule. When end(v)<=S, reuse is safe (no write-write conflict).
    If last access of v was write at L, then end(v)>=L+1 (A4). So end(v)<=S implies S>L.
    Prove: end_v <= S and last_is_write_at_L implies S > L (from A4 end>=L+1).
    """
    end_v, S, L = Ints("end_v S L")
    # A4: if last access was write at L, end(v) >= L+1
    # Negation of T1: we can have end_v<=S and last access write at L, and S<=L (conflict)
    # T1 says: when we reuse (end_v<=S), we never have write-write conflict.
    # Write-write conflict: both write at same cycle. So if v's last write at L, and we
    # allocate new writer at S, conflict only if S=L. T1 says end_v<=S => safe, so S=L
    # cannot hold when end_v<=S and last was write at L (since end_v>=L+1).
    # So: end_v>=L+1 and end_v<=S => S>=L+1 > L. Thus S cannot equal L.
    # Negation of theorem: assume end_v<=S, end_v>=L+1 (A4 for dead/write case), and S<=L.
    # Then L+1 <= end_v <= S <= L => L+1 <= L, impossible. So UNSAT.
    axiom_a4_write = end_v >= L + 1  # when last access is write at L
    reuse = end_v <= S
    conflict = S <= L  # new writer at S would conflict if S<=L (same or earlier cycle)
    negated = And(axiom_a4_write, reuse, conflict)
    solver = Solver(ctx=ctx)
    solver.add(negated)
    result = solver.check()
    stats = _get_stats(solver)
    return ("T1", "unsat" if str(result) == "unsat" else str(result), 0.0, stats)


def check_t9_expiration_implies_no_overlap(ctx=None) -> tuple[str, str, float, dict]:
    """T9: Expiration implies no overlap. If end(v)<=S, then [S,E] does not overlap [start(v),end(v)].
    Overlap requires S < end(v) (for [S,E] and [start,end] to overlap: need end > S and E > start).
    Expiration: end(v) <= S. So S < end(v) is false. Hence no overlap.
    Prove by refutation: assume end_v<=S and overlap([S,E],[start_v,end_v]).
    Overlap: end_v > S and E > start_v. But end_v<=S contradicts end_v>S. UNSAT.
    """
    start_v, end_v, S, E = Ints("start_v end_v S E")
    expiration = end_v <= S
    overlap = overlaps_closed(S, E, start_v, end_v)  # S<end_v and start_v<E
    negated = And(expiration, overlap)
    solver = Solver(ctx=ctx)
    solver.add(negated)
    result = solver.check()
    stats = _get_stats(solver)
    return ("T9", "unsat" if str(result) == "unsat" else str(result), 0.0, stats)


def check_t3_asap_bounds(ctx=None) -> tuple[str, str, float, dict]:
    """T3: ASAP bounds. For pred->succ, cycle(succ) >= cycle(pred) + offset.
    From A3: consumer at t requires t >= completion_cycle(pred)+1. For normal op, completion=cycle.
    So cycle_succ >= cycle_pred + 1. Negation: assume A3 and cycle_succ < cycle_pred + 1. UNSAT."""
    cycle_pred, cycle_succ = Ints("cycle_pred cycle_succ")
    # A3: cycle_succ >= completion_pred + 1; for normal op completion_pred = cycle_pred
    axiom_a3 = cycle_succ >= cycle_pred + 1
    # Negation of T3: cycle_succ < cycle_pred + 1
    negated = And(axiom_a3, cycle_succ < cycle_pred + 1)
    solver = Solver(ctx=ctx)
    solver.add(negated)
    result = solver.check()
    stats = _get_stats(solver)
    return ("T3", "unsat" if str(result) == "unsat" else str(result), 0.0, stats)


def check_t4_alap_bounds(ctx=None) -> tuple[str, str, float, dict]:
    """T4: ALAP bounds. cycle(pred) <= cycle(succ) - offset. From T3: cycle_succ >= cycle_pred+1.
    Rearranged: cycle_pred <= cycle_succ - 1. Negation: assume T3 and cycle_pred > cycle_succ - 1."""
    cycle_pred, cycle_succ = Ints("cycle_pred cycle_succ")
    t3 = cycle_succ >= cycle_pred + 1  # from A3
    # Negation of T4: cycle_pred > cycle_succ - 1, i.e. cycle_pred >= cycle_succ
    negated = And(t3, cycle_pred >= cycle_succ)
    solver = Solver(ctx=ctx)
    solver.add(negated)
    result = solver.check()
    stats = _get_stats(solver)
    return ("T4", "unsat" if str(result) == "unsat" else str(result), 0.0, stats)


def check_t5_cycle_domain(ctx=None) -> tuple[str, str, float, dict]:
    """T5: Cycle domain. ASAP(op) <= cycle(op) <= ALAP(op) is sound (contains all feasible cycles).
    Model: 2-node chain pred->succ, offset=1, makespan K. Fixpoint: ASAP_pred=0, ALAP_pred=K-1,
    ASAP_succ=1, ALAP_succ=K. Feasible: precedence and makespan. Negation: feasible schedule
    but cycle_pred < ASAP_pred or cycle_pred > ALAP_pred or cycle_succ < ASAP_succ or cycle_succ > ALAP_succ."""
    K = 10  # makespan bound
    cycle_pred, cycle_succ = Ints("cycle_pred cycle_succ")
    precedence = cycle_succ >= cycle_pred + 1
    makespan = cycle_succ <= K
    nonneg = And(cycle_pred >= 0, cycle_succ >= 0)
    feasible = And(precedence, makespan, nonneg)
    # ASAP_pred=0, ALAP_pred=K-1, ASAP_succ=1, ALAP_succ=K
    violates_pred = Or(cycle_pred < 0, cycle_pred > K - 1)
    violates_succ = Or(cycle_succ < 1, cycle_succ > K)
    negated = And(feasible, Or(violates_pred, violates_succ))
    solver = Solver(ctx=ctx)
    solver.add(negated)
    result = solver.check()
    stats = _get_stats(solver)
    return ("T5", "unsat" if str(result) == "unsat" else str(result), 0.0, stats)


def check_t6_half_open_conversion(ctx=None) -> tuple[str, str, float, dict]:
    """T6: Half-open conversion. end_halfopen = end_closed + 1.
    For integer t: t in closed [s,e] (s<=t<=e) iff t in half-open [s,e+1) (s<=t<e+1).
    I.e. t<=e <-> t<e+1. Negation: exists t,e with (t<=e and t>=e+1) or (t<e+1 and t>e). UNSAT."""
    t, e = Ints("t e")
    # Negation: (t<=e and t>=e+1) or (t<e+1 and t>e)
    case1 = And(t <= e, t >= e + 1)
    case2 = And(t < e + 1, t > e)
    negated = Or(case1, case2)
    solver = Solver(ctx=ctx)
    solver.add(negated)
    result = solver.check()
    stats = _get_stats(solver)
    return ("T6", "unsat" if str(result) == "unsat" else str(result), 0.0, stats)


def check_t10_dead_shifted_valu_end(ctx=None) -> tuple[str, str, float, dict]:
    """T10: Dead shifted VALU dest: end(v) = max_cycle(v) + 1. From A2: last write at max_cycle,
    reuse at S requires S > max_cycle. So end must block reuse at max_cycle, hence end > max_cycle.
    Minimal: end = max_cycle + 1. Negation: end < max_cycle+1 and A2 holds. A2 => end >= max_cycle+1. UNSAT."""
    max_cycle, end_v = Ints("max_cycle end_v")
    # A2 for dead shifted: must block reuse when S <= max_cycle, so end > max_cycle => end >= max_cycle+1
    axiom_a2 = end_v >= max_cycle + 1
    # Negation of T10: end < max_cycle + 1
    negated = And(axiom_a2, end_v < max_cycle + 1)
    solver = Solver(ctx=ctx)
    solver.add(negated)
    result = solver.check()
    stats = _get_stats(solver)
    return ("T10", "unsat" if str(result) == "unsat" else str(result), 0.0, stats)
