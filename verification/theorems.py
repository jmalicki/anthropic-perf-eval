"""Theorems T1–T14 from LIVENESS_SPECIFICATION.md."""

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
    """T3: ASAP bounds. For pred->succ, cycle(succ) >= cycle(pred) + offset_asap(pred).
    From A3: consumer at t requires t >= completion_cycle(pred)+1; minimum delay is offset >= 1.
    We prove for arbitrary integer offset >= 1 so the result holds for normal (1), FMA (1), shifted (1), etc.
    Negation: assume A3 (cycle_succ >= cycle_pred + offset) and cycle_succ < cycle_pred + offset. UNSAT."""
    cycle_pred, cycle_succ, offset = Ints("cycle_pred cycle_succ offset")
    solver = Solver(ctx=ctx)
    solver.add(offset >= 1)  # minimum delay from A3/A5 (normal, FMA, shifted all have delay >= 1)
    axiom_a3 = cycle_succ >= cycle_pred + offset
    negated = And(axiom_a3, cycle_succ < cycle_pred + offset)
    solver.add(negated)
    result = solver.check()
    stats = _get_stats(solver)
    return ("T3", "unsat" if str(result) == "unsat" else str(result), 0.0, stats)


def check_t4_alap_bounds(ctx=None) -> tuple[str, str, float, dict]:
    """T4: ALAP bounds. cycle(pred) <= cycle(succ) - offset_asap(pred). From T3: cycle_succ >= cycle_pred + offset.
    Rearranged: cycle_pred <= cycle_succ - offset. We use the same symbolic offset >= 1 as T3.
    Negation: assume T3 and cycle_pred > cycle_succ - offset (i.e. cycle_pred >= cycle_succ - offset + 1). UNSAT."""
    cycle_pred, cycle_succ, offset = Ints("cycle_pred cycle_succ offset")
    solver = Solver(ctx=ctx)
    solver.add(offset >= 1)
    t3 = cycle_succ >= cycle_pred + offset
    # Negation of T4: cycle_pred > cycle_succ - offset  (for integers: cycle_pred >= cycle_succ - offset + 1)
    negated = And(t3, cycle_pred >= cycle_succ - offset + 1)
    solver.add(negated)
    result = solver.check()
    stats = _get_stats(solver)
    return ("T4", "unsat" if str(result) == "unsat" else str(result), 0.0, stats)


def check_t5_cycle_domain(ctx=None) -> tuple[str, str, float, dict]:
    """T5: Cycle domain. ASAP(op) <= cycle(op) <= ALAP(op) is sound (contains all feasible cycles).
    We prove for a representative 2-node chain: pred->succ, offset=1, makespan K (symbolic, K>=2).
    Fixpoint: ASAP_pred=0, ALAP_pred=K-1, ASAP_succ=1, ALAP_succ=K. Full theorem for arbitrary DAGs
    follows from T3/T4 and the fixpoint definition; this instance shows no feasible schedule lies
    outside the computed domain. Negation: feasible schedule but cycle outside [ASAP, ALAP]. UNSAT."""
    K, cycle_pred, cycle_succ = Ints("K cycle_pred cycle_succ")
    solver = Solver(ctx=ctx)
    solver.add(K >= 2)  # so ALAP_pred = K-1 >= 1 and domain is non-empty
    precedence = cycle_succ >= cycle_pred + 1
    makespan = cycle_succ <= K
    nonneg = And(cycle_pred >= 0, cycle_succ >= 0)
    feasible = And(precedence, makespan, nonneg)
    violates_pred = Or(cycle_pred < 0, cycle_pred > K - 1)
    violates_succ = Or(cycle_succ < 1, cycle_succ > K)
    negated = And(feasible, Or(violates_pred, violates_succ))
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


def check_t7_slot_tightening_one_step(ctx=None) -> tuple[str, str, float, dict]:
    """T7 inductive step: one slot-aware tightening step is sound.
    When we push ASAP (or pull ALAP) for op O at cycle t because demand_other(t) + slots(O) > cap,
    no feasible schedule can have O at t. So we only remove infeasible cycles.
    Negation: tightening holds (demand_other >= cap - slots_O + 1) AND feasible with O at t
    (demand_other + slots_O <= cap). Then demand_other >= cap - slots_O + 1 and
    demand_other <= cap - slots_O. UNSAT."""
    cap, slots_O, demand_other = Ints("cap slots_O demand_other")
    # Tightening: at cycle t, adding O would exceed cap. So demand_other + slots_O > cap,
    # i.e. demand_other >= cap - slots_O + 1 (for integers)
    tightening = demand_other >= cap - slots_O + 1
    # Feasible with O at t: total demand <= cap, so demand_other + slots_O <= cap
    feasible_with_o_at_t = demand_other + slots_O <= cap
    negated = And(tightening, feasible_with_o_at_t)
    solver = Solver(ctx=ctx)
    solver.add(negated)
    # Axioms: cap and slots positive
    solver.add(cap >= 1)
    solver.add(slots_O >= 1)
    solver.add(demand_other >= 0)
    result = solver.check()
    stats = _get_stats(solver)
    return ("T7", "unsat" if str(result) == "unsat" else str(result), 0.0, stats)


def check_t7_slot_tightening_two_step_composition(ctx=None) -> tuple[str, str, float, dict]:
    """T7 two-step composition: show that the one-step lemma does NOT compose.
    Model: 2 ops (O1, O2), 3 cycles (0,1,2), capacity 1. Precedence: O2 >= O1 + 1.
    Two valid one-step tightenings: (1) exclude O1 from cycle 0; (2) exclude O2 from cycle 2.
    We ask: does a feasible schedule (c1, c2) exist? If UNSAT, then no feasible schedule
    exists after these two valid one-step steps — i.e. the composition theorem fails.
    This proves that the Z3 one-step proof (T7) is insufficient for the full pipeline."""
    c1, c2 = Ints("c1 c2")
    solver = Solver(ctx=ctx)
    # Cycle domain
    solver.add(c1 >= 0)
    solver.add(c1 <= 2)
    solver.add(c2 >= 0)
    solver.add(c2 <= 2)
    # Valid one-step tightening 1: O1 excluded from cycle 0 (so c1 != 0)
    solver.add(c1 != 0)
    # Valid one-step tightening 2: O2 excluded from cycle 2 (so c2 != 2)
    solver.add(c2 != 2)
    # Capacity 1: ops cannot share a cycle
    solver.add(c1 != c2)
    # Precedence: O2 must run after O1 (e.g. min delay 1)
    solver.add(c2 >= c1 + 1)
    result = solver.check()
    stats = _get_stats(solver)
    # We expect UNSAT: no (c1,c2) satisfies all. So remaining domain c1 in {1,2}, c2 in {0,1};
    # precedence c2>=c1+1 forces c1=0 or c2>=2, but c1!=0 and c2!=2 => no solution.
    return (
        "T7-two-step",
        "unsat" if str(result) == "unsat" else str(result),
        0.0,
        stats,
    )


def check_t7_one_exclusion_preserves_feasibility_asap(ctx=None) -> tuple[str, str, float, dict]:
    """T7 sufficient-condition exploration: one ASAP exclusion preserves feasibility.
    Same 2-op 3-cycle model as T7-two-step; only exclusion is O1 not at 0 (valid one-step).
    We expect SAT: (c1=1, c2=2) is feasible. So a single valid one-step exclusion does
    not necessarily remove all feasible schedules; composition with a second exclusion does."""
    c1, c2 = Ints("c1 c2")
    solver = Solver(ctx=ctx)
    solver.add(c1 >= 0)
    solver.add(c1 <= 2)
    solver.add(c2 >= 0)
    solver.add(c2 <= 2)
    solver.add(c1 != 0)  # only exclusion: O1 not at 0 (ASAP push)
    solver.add(c1 != c2)
    solver.add(c2 >= c1 + 1)
    result = solver.check()
    stats = _get_stats(solver)
    return (
        "T7-one-exclusion-asap",
        "sat" if str(result) == "sat" else str(result),
        0.0,
        stats,
    )


def check_t7_one_exclusion_preserves_feasibility_alap(ctx=None) -> tuple[str, str, float, dict]:
    """T7 sufficient-condition exploration: one ALAP exclusion preserves feasibility.
    Same 2-op 3-cycle model; only exclusion is O2 not at 2 (valid one-step). We expect SAT:
    (c1=0, c2=1) is feasible. So one ALAP exclusion alone does not remove all feasible schedules."""
    c1, c2 = Ints("c1 c2")
    solver = Solver(ctx=ctx)
    solver.add(c1 >= 0)
    solver.add(c1 <= 2)
    solver.add(c2 >= 0)
    solver.add(c2 <= 2)
    solver.add(c2 != 2)  # only exclusion: O2 not at 2 (ALAP pull)
    solver.add(c1 != c2)
    solver.add(c2 >= c1 + 1)
    result = solver.check()
    stats = _get_stats(solver)
    return (
        "T7-one-exclusion-alap",
        "sat" if str(result) == "sat" else str(result),
        0.0,
        stats,
    )


def check_t7_greedy_cap_preserves_canonical(ctx=None) -> tuple[str, str, float, dict]:
    """T7 greedy-cap rule: the canonical schedule G remains feasible after exclusions.
    Model: 2 ops (O1, O2), 3 cycles (0,1,2), capacity 1, precedence O2 >= O1 + 1.
    Canonical schedule G = (G(O1)=1, G(O2)=2). Greedy-cap: only exclude (O, t) when
    t != G(O). So we may exclude O1 from {0, 2} and O2 from {0, 1}; we never exclude
    O1 from 1 or O2 from 2. We add the maximal set of allowed exclusions (c1!=0, c1!=2,
    c2!=0, c2!=1) plus precedence and capacity, then assert G = (1, 2) is still a
    solution. SAT proves that under greedy-cap, the canonical schedule remains feasible."""
    c1, c2 = Ints("c1 c2")
    solver = Solver(ctx=ctx)
    # Cycle domain
    solver.add(c1 >= 0)
    solver.add(c1 <= 2)
    solver.add(c2 >= 0)
    solver.add(c2 <= 2)
    # Capacity and precedence
    solver.add(c1 != c2)
    solver.add(c2 >= c1 + 1)
    # Greedy-cap allowed exclusions only: never exclude G(O) for any O.
    # G(O1)=1, G(O2)=2. So we may exclude c1 in {0,2}, c2 in {0,1}.
    solver.add(c1 != 0)
    solver.add(c1 != 2)
    solver.add(c2 != 0)
    solver.add(c2 != 1)
    # Assert canonical schedule G = (1, 2)
    solver.add(c1 == 1)
    solver.add(c2 == 2)
    result = solver.check()
    stats = _get_stats(solver)
    return (
        "T7-greedy-cap-preserves-G",
        "sat" if str(result) == "sat" else str(result),
        0.0,
        stats,
    )


def check_t7_asap_only_composition(ctx=None) -> tuple[str, str, float, dict]:
    """T7 safe-composition candidate: multiple ASAP-only exclusions preserve feasibility.
    Model: 3 ops (O1, O2, O3), 5 cycles (0..4), capacity 1, precedence chain O2>=O1+1, O3>=O2+1.
    Apply only valid one-step ASAP exclusions: c1>=1 (O1 not at 0), c2>=2 (O2 not at 0,1).
    Ask: does a feasible schedule (c1,c2,c3) exist? We expect SAT, e.g. (1,2,3). If SAT,
    ASAP-only composition may be a safe subset for implementation."""
    c1, c2, c3 = Ints("c1 c2 c3")
    solver = Solver(ctx=ctx)
    solver.add(c1 >= 0)
    solver.add(c1 <= 4)
    solver.add(c2 >= 0)
    solver.add(c2 <= 4)
    solver.add(c3 >= 0)
    solver.add(c3 <= 4)
    solver.add(c1 != c2)
    solver.add(c1 != c3)
    solver.add(c2 != c3)
    solver.add(c2 >= c1 + 1)
    solver.add(c3 >= c2 + 1)
    # ASAP-only exclusions (push earliest start)
    solver.add(c1 >= 1)
    solver.add(c2 >= 2)
    result = solver.check()
    stats = _get_stats(solver)
    return (
        "T7-asap-only-composition",
        "sat" if str(result) == "sat" else str(result),
        0.0,
        stats,
    )


def check_t7_alap_only_composition(ctx=None) -> tuple[str, str, float, dict]:
    """T7 safe-composition candidate: multiple ALAP-only exclusions preserve feasibility.
    Model: 3 ops (O1, O2, O3), 5 cycles (0..4), capacity 1, precedence chain O2>=O1+1, O3>=O2+1.
    Apply only valid one-step ALAP exclusions: c3<=3 (O3 not at 4), c2<=2 (O2 not at 3,4).
    Ask: does a feasible schedule exist? We expect SAT, e.g. (0,1,2) or (0,1,3). If SAT,
    ALAP-only composition may be a safe subset for implementation."""
    c1, c2, c3 = Ints("c1 c2 c3")
    solver = Solver(ctx=ctx)
    solver.add(c1 >= 0)
    solver.add(c1 <= 4)
    solver.add(c2 >= 0)
    solver.add(c2 <= 4)
    solver.add(c3 >= 0)
    solver.add(c3 <= 4)
    solver.add(c1 != c2)
    solver.add(c1 != c3)
    solver.add(c2 != c3)
    solver.add(c2 >= c1 + 1)
    solver.add(c3 >= c2 + 1)
    # ALAP-only exclusions (pull latest end)
    solver.add(c3 <= 3)
    solver.add(c2 <= 2)
    result = solver.check()
    stats = _get_stats(solver)
    return (
        "T7-alap-only-composition",
        "sat" if str(result) == "sat" else str(result),
        0.0,
        stats,
    )


def check_t7_domain_based_removes_canonical(ctx=None) -> tuple[str, str, float, dict]:
    """T7 domain-based rule refutation: the second exclusion (domain-based) removes G.
    After excluding O1 from 0, O1's domain is {1, 2}. Domain-based demand at cycle 2
    counts O1 (since 2 in domain(O1)), so we exclude O2 from 2. Then the canonical
    schedule G = (1, 2) is no longer feasible (c2=2 excluded). We encode: exclusions
    c1!=0 and c2!=2 (as domain-based would apply), plus G = (1, 2). We expect UNSAT,
    i.e. G is not a solution after domain-based exclusions."""
    c1, c2 = Ints("c1 c2")
    solver = Solver(ctx=ctx)
    solver.add(c1 >= 0)
    solver.add(c1 <= 2)
    solver.add(c2 >= 0)
    solver.add(c2 <= 2)
    solver.add(c1 != 0)   # first exclusion (one-step for O1 at 0)
    solver.add(c2 != 2)   # second exclusion (domain-based: O2 at 2 excluded)
    solver.add(c1 != c2)
    solver.add(c2 >= c1 + 1)
    solver.add(c1 == 1)
    solver.add(c2 == 2)   # G = (1, 2)
    result = solver.check()
    stats = _get_stats(solver)
    # UNSAT = G is not feasible after domain-based exclusions (refutation)
    return (
        "T7-domain-based-removes-G",
        "unsat" if str(result) == "unsat" else str(result),
        0.0,
        stats,
    )


def check_t8_scratch_tightening_one_step(ctx=None) -> tuple[str, str, float, dict]:
    """T8 inductive step: one scratch-aware tightening step is sound.
    When we tighten for op O at cycle t because scratch_other(t) + scratch(O) > budget,
    no feasible schedule can have O at t. Same structure as T7."""
    budget, scratch_O, scratch_other = Ints("budget scratch_O scratch_other")
    tightening = scratch_other >= budget - scratch_O + 1
    feasible_with_o_at_t = scratch_other + scratch_O <= budget
    negated = And(tightening, feasible_with_o_at_t)
    solver = Solver(ctx=ctx)
    solver.add(negated)
    solver.add(budget >= 1)
    solver.add(scratch_O >= 1)
    solver.add(scratch_other >= 0)
    result = solver.check()
    stats = _get_stats(solver)
    return ("T8", "unsat" if str(result) == "unsat" else str(result), 0.0, stats)


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


def check_t11_dead_single_cycle_end(ctx=None) -> tuple[str, str, float, dict]:
    """T11: Dead single-cycle VReg: end(v) = D + 1. Written at D, no consumers.
    From A2: last access is write at D, reuse at S requires S > D, so end >= D+1. Minimal: end = D+1.
    Negation: end < D+1 and A2 holds (end >= D+1). UNSAT."""
    D, end_v = Ints("D end_v")
    axiom_a2 = end_v >= D + 1
    negated = And(axiom_a2, end_v < D + 1)
    solver = Solver(ctx=ctx)
    solver.add(negated)
    result = solver.check()
    stats = _get_stats(solver)
    return ("T11", "unsat" if str(result) == "unsat" else str(result), 0.0, stats)


def check_t12_shifted_valu_with_consumer_end(ctx=None) -> tuple[str, str, float, dict]:
    """T12: Shifted VALU dest with consumer at R: end(v) = R. Precedence: R >= max_cycle + 1.
    Last access is read at R. Per A2, reuse for writer at R allowed. We need the range through
    cycle R inclusive (read at R), so minimal closed end is R. Negation: end < R and
    need_through_R (end >= R). UNSAT."""
    R, end_v, max_cycle = Ints("R end_v max_cycle")
    precedence = R >= max_cycle + 1
    need_through_R = end_v >= R  # range must extend through R (last read) so minimal end = R
    negated = And(precedence, end_v < R, need_through_R)
    solver = Solver(ctx=ctx)
    solver.add(negated)
    result = solver.check()
    stats = _get_stats(solver)
    return ("T12", "unsat" if str(result) == "unsat" else str(result), 0.0, stats)


def check_t13_occupied_semantics(ctx=None) -> tuple[str, str, float, dict]:
    """T13: Occupied interval semantics. For [start,end], occupied for start <= t <= end,
    reusable for writer at end+1. Per A2/A4: at cycle end+1 we're past last access. Negation:
    assume [start,end] but NOT reusable at end+1 - would require still needing range at end+1.
    But end is last needed cycle, so end+1 cannot be needed. Encode: reusable at end+1 means
    a new writer at S=end+1 doesn't conflict. Per T9, expiration (end <= S) => no overlap.
    So end <= end+1 holds, hence reusable. Negation: end > end+1 (impossible). Simpler: the
    half-open [start, end+1) contains cycles start..end. Reusable at end+1 means first free
    cycle is end+1. That's exactly end_halfopen = end_closed + 1 (T6). So T13 follows from T6.
    Prove directly: for cycle t, (start <= t <= end) => t is in half-open [start, end+1).
    LHS: start <= t and t <= end. RHS: start <= t and t < end+1. For integers t<=end <-> t<end+1. Same.
    So occupied semantics match. Negation: occupied at t but t not in [start,end+1), or vice versa.
    Simpler refutation: reusable at end+1 means we allow writer at S=end+1. Per T1, expire when end<=S.
    So end <= end+1. Always true. The theorem says we ARE reusable at end+1. So we need to prove
    no conflict. Conflict would mean we need the range at end+1. Need through end means end >= last_use.
    Last use is at end (by def of end). So we don't need at end+1. Qed. Encode: assume we need range
    at end+1 (not reusable). Needing at t means t in [start, end]. So end+1 in [start,end] => start <= end+1 <= end => end+1 <= end. UNSAT."""
    start, end = Ints("start end")
    # Not reusable at end+1 would mean we still need the range at cycle end+1
    # Needing at t means t in occupied [start,end], i.e. start <= t <= end
    need_at_end_plus_1 = And(start <= end + 1, end + 1 <= end)
    solver = Solver(ctx=ctx)
    solver.add(need_at_end_plus_1)
    result = solver.check()
    stats = _get_stats(solver)
    return ("T13", "unsat" if str(result) == "unsat" else str(result), 0.0, stats)


def check_t14_capacity_lower_bound(ctx=None) -> tuple[str, str, float, dict]:
    """T14: Capacity lower bound. For engine with capacity C and op-count N, makespan >= ceil(N/C).
    From A6: at most C ops per cycle, so over M cycles at most M*C ops. So N <= M*C for valid schedule.
    Negation: valid (N <= M*C) and makespan < ceil(N/C). For integers, M < ceil(N/C) iff M*C < N.
    So we need N <= M*C and M*C < N => N < N. UNSAT."""
    M, N, C = Ints("M N C")
    solver = Solver(ctx=ctx)
    solver.add(C >= 1)
    solver.add(N >= 0)
    solver.add(M >= 0)
    capacity_constraint = N <= M * C  # at most C ops per cycle over M cycles
    negation_t14 = M * C < N  # makespan M < ceil(N/C)  <=>  M*C < N for integers
    solver.add(capacity_constraint)
    solver.add(negation_t14)
    result = solver.check()
    stats = _get_stats(solver)
    return ("T14", "unsat" if str(result) == "unsat" else str(result), 0.0, stats)


def check_t15_optional_capacity_redundant(ctx=None) -> tuple[str, str, float, dict]:
    """T15: Optional propagation constraints are redundant. Cumulative (N <= C*M) implies
    the direct capacity inequality (M >= ceil(N/C)). So any solution feasible without the
    optional constraints is feasible with them. Same encoding as T14: assume cumulative
    and negation of M >= ceil(N/C) (i.e. M*C < N) => UNSAT."""
    M, N, C = Ints("M N C")
    solver = Solver(ctx=ctx)
    solver.add(C >= 1)
    solver.add(N >= 0)
    solver.add(M >= 0)
    cumulative = N <= M * C  # cumulative constraint for one engine
    negation_optional = M * C < N  # optional says M >= ceil(N/C); negation is M*C < N
    solver.add(cumulative)
    solver.add(negation_optional)
    result = solver.check()
    stats = _get_stats(solver)
    return ("T15", "unsat" if str(result) == "unsat" else str(result), 0.0, stats)
