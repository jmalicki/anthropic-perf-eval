"""Conjectures about scratch address allocation for fixed live intervals, checked with Z3.

After scheduling, every value has a fixed live interval and a size. Address assignment
(``scheduler_rs/src/address_assign.rs``) gives each value one contiguous block of words for
its whole interval; two values live at the same time must not share a word. The schedulers
only guarantee that the live words at every cycle fit the budget. Is that enough? The problem
is Dynamic Storage Allocation (DSA, Garey and Johnson problem SR2). Statements, proofs,
bounds and literature: docs/ALLOCATION_CONJECTURES.md.

Model
-----
An *instance* is a list of :class:`Value` (closed interval ``[start, end]`` of abstract time
steps, size in words) and a budget ``B`` (words ``0 .. B-1``). Two values *overlap* when their
intervals share a step. A *layout* gives each value an address ``a`` with
``0 <= a <= B - size`` (and ``a % align == offset`` for values larger than one word, under an
:class:`Alignment`) such that overlapping values get disjoint blocks. The *load* at a step is
the total size of the values live then; the *per-cycle check* is load <= B at every step.

The real liveness semantics (closed ``[def, last_read]``, a value read at C may share with one
written at C, except that a point interval ``[C, C]`` shares with nothing live at C) is this
model on a doubled time line, checks A3-*. Every result below therefore holds for the real
allocator too.

Conjectures and checks
----------------------
C1 (sufficiency): sizes {1, k}; the per-cycle check implies that a layout exists.
    FALSE for every k >= 2: unaligned for every B >= 2k, k-aligned for every multiple of k
    from 2k, and under the real semantics (:func:`pinning_counterexample`). It also fails for
    two sizes that do not divide each other ({2, 3}). In the bounds searched it holds when
    B < 2k (at most one vector live at a time).
      A1-c1-refuted-*   witness passes the per-cycle check and Z3 proves no layout: UNSAT.
      A2-c1-tight-*     the same witness has a layout with one more word: SAT.
      A6-c1-bounded-*   exhaustive over every instance with at most T steps, any number of
                        values: "unsat" = C1 holds in the bound, "sat" = counterexample
                        (minimised and re-proved by Z3).
      A6-c1-cegis-*     the same question by CEGIS (Z3 synthesis, layout check): cross-check.
C2 (algorithm): some polynomial allocator always succeeds under the per-cycle check.
    FALSE, since C1 is false; "find a layout whenever one exists" is NP-hard (SR2). Each
    candidate greedy has a stored instance that has a layout but on which it fails.
      A7-greedy-witnesses  each stored instance passes the per-cycle check, has a layout
                           (SAT) and its greedy fails.
C3 (two-pass bound): vectors in start order into the lowest free k-aligned slot, then scalars
    in start order into any word free over their interval, succeed whenever
    k * max_v + max_s <= B (max_v, max_s: most vectors / scalars live at one step).
    TRUE; the proof in the doc rests on A4.
      A4-start-order-lemma  a value placed earlier in start order overlaps the new one iff it
                            is live at the new one's start: UNSAT.
      A5-c3-exhaustive-*    every instance in the bound satisfying the hypothesis: the two-pass
                            allocator succeeds ("unsat" = no failure found).

Result convention (as in verification/verify.py): soundness checks encode a violation, so
UNSAT means the statement holds; the names in ``EXPECT_SAT`` ask for a witness and want SAT.

Counterexamples are data in ``allocation_counterexamples.json`` (rows
``[start, end, size, count]``), shared with the allocator's unit tests.

Run:
    uv run python -m verification.allocation_conjectures             # fast checks (proofs suite)
    uv run python -m verification.allocation_conjectures --slow      # + bounded checks, minutes
    uv run python -m verification.allocation_conjectures --search-greedy  # regenerate C2 witnesses
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import random
import sys
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass

from z3 import And, If, Int, Not, Or, Solver, Sum, sat, unsat

COUNTEREXAMPLES_PATH = pathlib.Path(__file__).with_name("allocation_counterexamples.json")

CheckResult = tuple[str, str, float, dict]

# =============================================================================
# Instances
# =============================================================================


@dataclass(frozen=True, order=True)
class Value:
    """A value live over the closed interval [start, end] (abstract steps), `size` words wide."""

    start: int
    end: int
    size: int

    def overlaps(self, other: Value) -> bool:
        """True when the two intervals share a step."""
        return self.start <= other.end and other.start <= self.end


@dataclass(frozen=True)
class Alignment:
    """Address rule for values wider than one word: ``addr % align == offset``.

    ``Alignment()`` is what our allocator needs: address_assign.rs places vectors at any
    address and the ISA only asks vload/vstore blocks to be contiguous
    (docs/ISA_REFERENCE.md). ``Alignment(8)`` models 8-aligned slots. ``offset`` only sets
    the slot phase: ``Alignment(8, 1)`` puts slots at 1, 9, 17, ... but scalars may still use
    word 0. A reserved address 0 is modelled by a budget one word smaller (every address
    shifted down by one), not by the offset.
    """

    align: int = 1
    offset: int = 0

    def allows(self, size: int, addr: int) -> bool:
        """Whether a value of `size` words may start at `addr`."""
        return size == 1 or addr % self.align == self.offset % self.align

    def starts(self, size: int, budget: int) -> range:
        """Allowed start addresses for a value of `size` words, increasing."""
        if size == 1 or self.align == 1:
            return range(0, budget - size + 1)
        return range(self.offset % self.align, budget - size + 1, self.align)


UNALIGNED = Alignment()


def per_cycle_load(values: Sequence[Value]) -> list[int]:
    """Total size of the values live at each step 0 .. max end."""
    load = [0] * (max((v.end for v in values), default=-1) + 1)
    for v in values:
        for t in range(v.start, v.end + 1):
            load[t] += v.size
    return load


def fits_per_cycle(values: Sequence[Value], budget: int) -> bool:
    """The schedulers' guarantee: load <= budget at every step."""
    return all(words <= budget for words in per_cycle_load(values))


def max_live(values: Sequence[Value], size: int) -> int:
    """Most values of `size` live at one step."""
    return max(per_cycle_load([Value(v.start, v.end, 1) for v in values if v.size == size]), default=0)


def is_valid_layout(values: Sequence[Value], budget: int, addrs: Sequence[int],
                    alignment: Alignment = UNALIGNED) -> bool:
    """In range, aligned, and overlapping values get disjoint blocks."""
    for v, a in zip(values, addrs, strict=True):
        if not (0 <= a <= budget - v.size and alignment.allows(v.size, a)):
            return False
    return not any(v.overlaps(w) and a < b + w.size and b < a + v.size
                   for (v, a), (w, b) in itertools.combinations(zip(values, addrs), 2))


def instance_from_rows(rows: Iterable[Sequence[int]]) -> list[Value]:
    """Decode ``[[start, end, size, count], ...]`` (the JSON format)."""
    return [Value(s, e, size) for s, e, size, count in rows for _ in range(count)]


def instance_to_rows(values: Iterable[Value]) -> list[list[int]]:
    """Encode values as sorted ``[start, end, size, count]`` rows."""
    counts: dict[Value, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return [[v.start, v.end, v.size, n] for v, n in sorted(counts.items())]


def load_counterexamples(path: pathlib.Path = COUNTEREXAMPLES_PATH) -> dict:
    """The stored counterexamples (see the file's "format" entry)."""
    return json.loads(path.read_text())


def alignment_of(entry: dict) -> Alignment:
    """Alignment of a stored entry (default unaligned)."""
    return Alignment(entry.get("align", 1), entry.get("offset", 0))


# =============================================================================
# Layout existence
# =============================================================================


def layout_solver(values: Sequence[Value], budget: int, alignment: Alignment = UNALIGNED) -> tuple[Solver, list]:
    """Z3 encoding of "a layout exists": one integer address per value.

    Identical values are ordered by address (any layout can be permuted into that order),
    which keeps the UNSAT proofs small.
    """
    solver = Solver()
    addrs = [Int(f"a{i}") for i in range(len(values))]
    for i, (v, a) in enumerate(zip(values, addrs)):
        solver.add(a >= 0, a <= budget - v.size)
        if v.size > 1 and alignment.align > 1:
            solver.add(a == alignment.align * Int(f"q{i}") + alignment.offset % alignment.align)
    for (i, v), (j, w) in itertools.combinations(enumerate(values), 2):
        if v == w:
            solver.add(addrs[i] + v.size <= addrs[j])
        elif v.overlaps(w):
            solver.add(Or(addrs[i] + v.size <= addrs[j], addrs[j] + w.size <= addrs[i]))
    return solver, addrs


def find_layout(values: Sequence[Value], budget: int, alignment: Alignment = UNALIGNED) -> list[int] | None:
    """A layout found by Z3, or None when Z3 proves there is none."""
    solver, addrs = layout_solver(values, budget, alignment)
    result = solver.check()
    if result == unsat:
        return None
    if result != sat:
        raise RuntimeError(f"layout check returned {result}")
    layout = [solver.model().eval(a).as_long() for a in addrs]
    assert is_valid_layout(values, budget, layout, alignment)
    return layout


def start_order(values: Sequence[Value]) -> list[int]:
    """Indices by start; ties: wider first, then end, then index (deterministic)."""
    return sorted(range(len(values)), key=lambda i: (values[i].start, -values[i].size, values[i].end, i))


def search_layout(values: Sequence[Value], budget: int, alignment: Alignment = UNALIGNED) -> list[int] | None:
    """Exact backtracking layout search, much faster than Z3 on small instances.

    Places values in start order and tries every allowed address, so None means no layout.
    By A4 only earlier values live at the new value's start can block it. Identical values
    (adjacent in start order) take increasing addresses. The bounded searches use this and
    re-prove every counterexample with :func:`find_layout`.
    """
    order = start_order(values)
    vals = [values[i] for i in order]
    masks = [0] * len(vals)
    addrs = [0] * len(vals)

    def place(i: int) -> bool:
        if i == len(vals):
            return True
        v = vals[i]
        blocked = 0
        for j in range(i):
            if vals[j].end >= v.start:
                blocked |= masks[j]
        lowest = addrs[i - 1] + 1 if i and vals[i - 1] == v else 0
        for a in alignment.starts(v.size, budget):
            mask = ((1 << v.size) - 1) << a
            if a >= lowest and not blocked & mask:
                masks[i], addrs[i] = mask, a
                if place(i + 1):
                    return True
        return False

    if not place(0):
        return None
    layout = [0] * len(values)
    for pos, i in enumerate(order):
        layout[i] = addrs[pos]
    assert is_valid_layout(values, budget, layout, alignment)
    return layout


# =============================================================================
# C1 counterexamples: the pinning construction
# =============================================================================


def pinning_counterexample(big: int, budget: int, alignment: Alignment = UNALIGNED, small: int = 1) -> list[Value]:
    """An instance that passes the per-cycle check and has no layout. Needs big >= 2 * small.

    Let q = big // small and p the first step with small values. Small values: q-1 live over
    [p, p+1], one "pin" over [p, p+2], one over [p+1, p+2]; at p+1 these are q+1 values,
    more than `big` words. At p and p+2 one big value is live, and it sits at 0 or at big
    (the two halves of 2*big words), so the small values live then are in the other half.
    The pin is live at both, so it is the same half both times, and at p+1 all q+1 small
    values need that one half: impossible. Unaligned, a second big value at p-1 and at p+3
    fills all 2*big words, which forces the halves; big-aligned, the slots are the halves.

    For budget > 2*big, fillers live over the whole horizon take the extra words: scalars
    unaligned (removing the filler words from any layout, order-preserving, keeps every
    other block contiguous), whole big values aligned (budget must then be a multiple of big).
    """
    if big < 2 * small or budget < 2 * big:
        raise ValueError("needs big >= 2 * small and budget >= 2 * big")
    aligned = alignment.align > 1
    if aligned and (small != 1 or alignment.align != big or budget % big):
        raise ValueError("aligned witness needs small == 1, align == big and budget a multiple of big")
    q = big // small
    p = 0 if aligned else 1
    values = [Value(p, p + 1, small)] * (q - 1) + [Value(p, p + 2, small), Value(p + 1, p + 2, small)]
    if aligned:
        values += [Value(p, p, big), Value(p + 2, p + 2, big)]
    else:
        values += [Value(0, 0, big), Value(0, 1, big), Value(3, 4, big), Value(4, 4, big)]
    horizon = max(v.end for v in values)
    if aligned:
        values += [Value(0, horizon, big)] * ((budget - 2 * big) // big)
    else:
        values += [Value(0, horizon, 1)] * (budget - 2 * big)
    return sorted(values)


# =============================================================================
# Real liveness semantics as intervals on a doubled time line
# =============================================================================


def real_conflict(s1, e1, s2, e2):
    """Whether two real intervals [s1, e1] and [s2, e2] may not share a word (Z3 term).

    This is the pairwise safety rule address_assign.rs's expiry test applies: the
    earlier-starting value A frees its words for B when A.end < B.start, or A.end == B.start
    and neither is a point interval (reads happen before writes within a cycle). It is not
    a model of the scan itself, which can be more conservative: it stops expiring at the
    first heap entry it may not free, so a point interval at the head keeps later-ending
    entries' words busy. Layouts here are judged by the pairwise rule; a "no layout" result
    under it is all the more true for the scan. Accepts ints or Z3 integer terms.
    """

    def frees(sa, ea, sb, eb):
        return Or(ea < sb, And(ea == sb, sa < ea, sb < eb))

    return Not(If(s1 <= s2, frees(s1, e1, s2, e2), frees(s2, e2, s1, e1)))


def real_phase_interval(start, end):
    """Closed doubled-time interval of a real interval; phase 2C reads, 2C+1 writes.

    A non-point [s, e] holds its words from its write to its last read: [2s+1, 2e]. A point
    [C, C] holds them for the whole cycle: [2C, 2C+1]. Returns ints for int arguments and
    Z3 terms otherwise.
    """
    if isinstance(start, int) and isinstance(end, int):
        return (2 * start, 2 * start + 1) if start == end else (2 * start + 1, 2 * end)
    point = start == end
    return If(point, 2 * start, 2 * start + 1), If(point, 2 * start + 1, 2 * end)


def real_as_abstract(values: Sequence[Value]) -> list[Value]:
    """Real intervals as abstract values on the doubled time line (A3-real-conflict-is-overlap),
    so the layout search applies the real same-cycle-reuse rule."""
    return [Value(*real_phase_interval(v.start, v.end), v.size) for v in values]


def embed_in_real(values: Sequence[Value]) -> list[Value]:
    """Real intervals [2s, 2e+1] for abstract values (A3-real-embedding)."""
    return [Value(2 * v.start, 2 * v.end + 1, v.size) for v in values]


def check_real_conflict_is_overlap() -> CheckResult:
    """A3-real-conflict-is-overlap: real conflict iff the doubled intervals overlap. UNSAT = holds."""
    s1, e1, s2, e2 = Int("s1"), Int("e1"), Int("s2"), Int("e2")
    lo1, hi1 = real_phase_interval(s1, e1)
    lo2, hi2 = real_phase_interval(s2, e2)
    solver = Solver()
    solver.add(s1 <= e1, s2 <= e2, real_conflict(s1, e1, s2, e2) != And(lo1 <= hi2, lo2 <= hi1))
    return _solve("A3-real-conflict-is-overlap", solver)


def check_real_load_bound() -> CheckResult:
    """A3-real-load-bound: a value live at phase p is counted at cycle floor(p/2) by a closed
    per-cycle count, so the schedulers' check implies the doubled-time per-cycle check.

    The CP-SAT scratch cumulative uses [def, last_read + 1) half-open, i.e. closed
    [def, last_read] (cpsat_scheduler/model.rs). UNSAT = holds.
    """
    s, e, p = Int("s"), Int("e"), Int("p")
    lo, hi = real_phase_interval(s, e)
    cycle = p / 2  # Z3 integer division rounds down for p >= 0
    solver = Solver()
    solver.add(s >= 0, s <= e, lo <= p, p <= hi, Not(And(s <= cycle, cycle <= e)))
    return _solve("A3-real-load-bound", solver)


def check_real_embedding() -> CheckResult:
    """A3-real-embedding: abstract [s, e] -> real [2s, 2e+1] keeps conflicts and loads.

    Real conflict iff abstract overlap, and the value is in the closed count at real cycle C
    iff it is live at abstract step floor(C/2); so every abstract witness is a real one with
    cycles doubled. UNSAT = holds.
    """
    s1, e1, s2, e2, c = Int("s1"), Int("e1"), Int("s2"), Int("e2"), Int("c")
    conflict = real_conflict(2 * s1, 2 * e1 + 1, 2 * s2, 2 * e2 + 1)
    live_real = And(2 * s1 <= c, c <= 2 * e1 + 1)
    live_abstract = And(s1 <= c / 2, c / 2 <= e1)
    solver = Solver()
    solver.add(s1 >= 0, s2 >= 0, c >= 0, s1 <= e1, s2 <= e2,
               Or(conflict != And(s1 <= e2, s2 <= e1), live_real != live_abstract))
    return _solve("A3-real-embedding", solver)


# =============================================================================
# Candidate allocators (C2, C3)
# =============================================================================

Allocator = Callable[[Sequence[Value], int, Alignment], "list[int] | None"]
Chooser = Callable[[Value, list[int], list[tuple[Value, int]]], int]


def _place_in_order(values: Sequence[Value], budget: int, alignment: Alignment, order: Iterable[int],
                    choose: Chooser) -> list[int] | None:
    """Place values in `order`, never moving one; `choose` picks among the addresses free over
    the value's interval given the placed (value, address) pairs. None when one does not fit."""
    placed: list[tuple[Value, int]] = []
    addrs = [-1] * len(values)
    for i in order:
        v = values[i]
        free = [a for a in alignment.starts(v.size, budget)
                if not any(v.overlaps(w) and a < b + w.size and b < a + v.size for w, b in placed)]
        if not free:
            return None
        addrs[i] = choose(v, free, placed)
        placed.append((v, addrs[i]))
    return addrs


def _lowest(v: Value, free: list[int], placed: list[tuple[Value, int]]) -> int:
    return free[0]


def start_order_lowest(values: Sequence[Value], budget: int, alignment: Alignment = UNALIGNED) -> list[int] | None:
    """Start order, lowest free address (first fit; with Alignment(8), lowest free slot)."""
    return _place_in_order(values, budget, alignment, start_order(values), _lowest)


def start_order_most_recently_freed(values: Sequence[Value], budget: int,
                                    alignment: Alignment = UNALIGNED) -> list[int] | None:
    """Start order; the free address whose block was vacated last (largest end among placed
    values touching it, -1 if none), ties to the lowest address."""

    def choose(v: Value, free: list[int], placed: list[tuple[Value, int]]) -> int:
        def vacated(a: int) -> int:
            return max((w.end for w, b in placed if a < b + w.size and b < a + v.size), default=-1)

        return max(free, key=lambda a: (vacated(a), -a))

    return _place_in_order(values, budget, alignment, start_order(values), choose)


def longest_lifetime_first(values: Sequence[Value], budget: int, alignment: Alignment = UNALIGNED) -> list[int] | None:
    """Vectors by decreasing lifetime at the lowest free address, then scalars by decreasing
    lifetime at the highest free word (ties: start, index)."""

    def by_lifetime(wide: bool) -> list[int]:
        idx = [i for i, v in enumerate(values) if (v.size > 1) == wide]
        return sorted(idx, key=lambda i: (values[i].start - values[i].end, values[i].start, i))

    return _place_in_order(values, budget, alignment, by_lifetime(True) + by_lifetime(False),
                           lambda v, free, placed: free[0] if v.size > 1 else free[-1])


def vectors_then_scalars(values: Sequence[Value], budget: int, alignment: Alignment = UNALIGNED) -> list[int] | None:
    """C3's allocator: vectors in start order into the lowest free k-aligned slot (k = widest
    size; `alignment` only sets the slot offset), then scalars in start order into the lowest
    word free over their interval."""
    k = max((v.size for v in values), default=1)
    slots = Alignment(k, alignment.offset)
    order = start_order(values)
    vectors = [i for i in order if values[i].size > 1]
    scalars = [i for i in order if values[i].size == 1]
    return _place_in_order(values, budget, slots, vectors + scalars, _lowest)


GREEDY_ALLOCATORS: dict[str, Allocator] = {
    "start_order_lowest": start_order_lowest,
    "start_order_most_recently_freed": start_order_most_recently_freed,
    "longest_lifetime_first": longest_lifetime_first,
    "vectors_then_scalars": vectors_then_scalars,
}

# (allocator, alignment) pairs with a stored C2 witness; vectors_then_scalars always uses slots.
GREEDY_CANDIDATES: list[tuple[str, Alignment]] = [
    *[(name, align) for name in ("start_order_lowest", "start_order_most_recently_freed", "longest_lifetime_first")
      for align in (UNALIGNED, Alignment(8))],
    ("vectors_then_scalars", Alignment(8)),
]


def search_greedy_witness(name: str, alignment: Alignment, k: int, budgets: Iterable[int], steps: int,
                          max_values: int, trials_per_budget: int, seed: int) -> tuple[int, list[Value]] | None:
    """Seeded random search for an instance that passes the per-cycle check and has a layout
    (same alignment) on which greedy `name` fails. Budgets are tried in increasing order; the
    result is 1-minimal (dropping any value makes the greedy succeed or the layout vanish)."""
    greedy = GREEDY_ALLOCATORS[name]

    def is_witness(values: list[Value], budget: int) -> bool:
        return (fits_per_cycle(values, budget) and greedy(values, budget, alignment) is None
                and search_layout(values, budget, alignment) is not None)

    rng = random.Random(seed)
    types = value_types((1, k), steps)
    for budget in budgets:
        for _ in range(trials_per_budget):
            values = sorted(rng.choice(types) for _ in range(rng.randint(2, max_values)))
            if is_witness(values, budget):
                return budget, _minimise(values, lambda vs: is_witness(vs, budget))
    return None


def greedy_search(min_budget: int) -> list[dict]:
    """The seeded search behind the "greedy_failures" entries (k = 8, 8 steps)."""
    entries = []
    for name, alignment in GREEDY_CANDIDATES:
        found = search_greedy_witness(name, alignment, k=8, budgets=range(min_budget, 33), steps=8,
                                      max_values=16, trials_per_budget=20_000, seed=1)
        if found is not None:
            budget, values = found
            entries.append({"allocator": name, "align": alignment.align, "budget": budget,
                            "values": instance_to_rows(values)})
    return entries


# =============================================================================
# Bounded C1: every instance with at most T steps
# =============================================================================


def value_types(sizes: Sequence[int], steps: int) -> list[Value]:
    """Every (interval, size) within `steps` steps, sorted."""
    return sorted(Value(s, e, size) for size in sizes for s in range(steps) for e in range(s, steps))


def maximal_instances(sizes: Sequence[int], budget: int, steps: int) -> Iterator[list[Value]]:
    """Every instance within `steps` steps with load <= budget to which no value can be added.

    Adding values never creates a layout, so a bound contains a counterexample iff it
    contains a maximal one.
    """
    types = value_types(sizes, steps)
    load = [0] * steps
    chosen: list[Value] = []

    def room(v: Value) -> int:
        return min(budget - load[t] for t in range(v.start, v.end + 1)) // v.size

    def rec(i: int) -> Iterator[list[Value]]:
        if i == len(types):
            if all(room(v) == 0 for v in types):
                yield list(chosen)
            return
        v = types[i]
        for n in range(room(v), -1, -1):
            for t in range(v.start, v.end + 1):
                load[t] += n * v.size
            chosen.extend([v] * n)
            yield from rec(i + 1)
            del chosen[len(chosen) - n:]
            for t in range(v.start, v.end + 1):
                load[t] -= n * v.size

    yield from rec(0)


def exhaustive_c1(sizes: Sequence[int], budget: int, steps: int, alignment: Alignment = UNALIGNED,
                  slack: int = 0) -> tuple[str, list[Value] | None, int]:
    """Decide C1 in the bound by checking every maximal instance.

    With `slack` > 0 layouts get budget + slack words (load still <= budget): "sat" then
    means some instance needs more than `slack` extra words. Returns ("unsat", None,
    instances checked) or ("sat", minimised counterexample, instances checked).
    """
    checked = 0
    for values in maximal_instances(sizes, budget, steps):
        checked += 1
        if search_layout(values, budget + slack, alignment) is None:
            return "sat", _confirmed_counterexample(values, budget + slack, alignment), checked
    return "unsat", None, checked


def saturate(values: Sequence[Value], addrs: Sequence[int], budget: int, alignment: Alignment,
             types: Sequence[Value]) -> list[Value]:
    """Add values of `types` to a layout without moving any, until none fits anywhere.

    The result still has a layout, so every sub-multiset of it has one (a CEGIS lemma).
    """
    placed = list(zip(values, addrs))
    grew = True
    while grew:
        grew = False
        for t in types:
            a = next((a for a in alignment.starts(t.size, budget)
                      if not any(t.overlaps(w) and a < b + w.size and b < a + t.size for w, b in placed)), None)
            if a is not None:
                placed.append((t, a))
                grew = True
    return [v for v, _ in placed]


def cegis_c1(sizes: Sequence[int], budget: int, steps: int, alignment: Alignment = UNALIGNED,
             max_iterations: int = 100_000) -> tuple[str, list[Value] | None, int]:
    """Decide C1 in the bound by counterexample-guided synthesis.

    Synthesis (Z3): a count per value type, load <= budget at every step, and for each
    learned layout, more of some type than it holds. Check: does the proposed instance have
    a layout? If not, it is a counterexample; if so, saturate the layout and learn it and
    its time reversal. Same result shape as :func:`exhaustive_c1`; the third field is the
    number of iterations.
    """
    types = value_types(sizes, steps)
    counts = [Int(f"n{i}") for i in range(len(types))]
    synth = Solver()
    synth.add(*[n >= 0 for n in counts], Sum(counts) >= 1)
    for t in range(steps):
        synth.add(Sum([n * v.size for n, v in zip(counts, types) if v.start <= t <= v.end]) <= budget)
    for iteration in range(1, max_iterations + 1):
        result = synth.check()
        if result == unsat:
            return "unsat", None, iteration
        if result != sat:
            return f"unknown ({synth.reason_unknown()})", None, iteration
        model = synth.model()
        values = [v for n, v in zip(counts, types) for _ in range(model.eval(n).as_long())]
        layout = search_layout(values, budget, alignment)
        if layout is None:
            return "sat", _confirmed_counterexample(values, budget, alignment), iteration
        held = saturate(values, layout, budget, alignment, types)
        for lemma in (held, [Value(steps - 1 - v.end, steps - 1 - v.start, v.size) for v in held]):
            synth.add(Or([n > lemma.count(v) for n, v in zip(counts, types)]))
    raise RuntimeError(f"CEGIS did not converge in {max_iterations} iterations")


def _minimise(values: list[Value], keep: Callable[[list[Value]], bool]) -> list[Value]:
    """Drop single values (sorted order) while `keep` still holds, rescanning from the start
    after every drop, until no single drop keeps it: the result is 1-minimal even when `keep`
    is not monotone (greedy failure is not)."""
    values = sorted(values)
    dropped = True
    while dropped:
        dropped = False
        for i in range(len(values)):
            trial = values[:i] + values[i + 1:]
            if trial and keep(trial):
                values, dropped = trial, True
                break
    return values


def _confirmed_counterexample(values: list[Value], budget: int, alignment: Alignment) -> list[Value]:
    """Minimise a counterexample found by search_layout and re-prove it with Z3."""
    values = _minimise(values, lambda vs: search_layout(vs, budget, alignment) is None)
    if find_layout(values, budget, alignment) is not None:
        raise AssertionError("search_layout missed a layout that Z3 found")
    return values


# =============================================================================
# C3: start-order lemma and exhaustive bounded check
# =============================================================================


def check_start_order_lemma() -> CheckResult:
    """A4-start-order-lemma: if s_j <= s_x, [s_j, e_j] meets [s_x, e_x] iff s_j <= s_x <= e_j.

    In start order every value placed earlier starts no later, so those that block the new
    value are exactly those live at its start: at most (count live then) - 1. UNSAT = holds.
    """
    sj, ej, sx, ex = Int("sj"), Int("ej"), Int("sx"), Int("ex")
    solver = Solver()
    solver.add(sj <= ej, sx <= ex, sj <= sx, And(sj <= ex, sx <= ej) != And(sj <= sx, sx <= ej))
    return _solve("A4-start-order-lemma", solver)


def c3_failures(k: int, budget: int, steps: int, max_values: int) -> Iterator[list[Value]]:
    """Instances (sizes {1, k}, <= `steps` steps, <= `max_values` values) satisfying
    k * max_v + max_s <= budget on which vectors_then_scalars fails. C3 says: none."""
    types = value_types((1, k), steps)
    for n in range(1, max_values + 1):
        for combo in itertools.combinations_with_replacement(types, n):
            values = list(combo)
            if k * max_live(values, k) + max_live(values, 1) <= budget and vectors_then_scalars(values, budget) is None:
                yield values


# =============================================================================
# Checks
# =============================================================================


def _solve(name: str, solver: Solver) -> CheckResult:
    t0 = time.perf_counter()
    result = str(solver.check())
    return (name, result, time.perf_counter() - t0, {})


def _tag(big: int, budget: int, alignment: Alignment, small: int = 1) -> str:
    sizes = f"k{big}" if small == 1 else f"s{small}k{big}"
    return f"{sizes}-B{budget}-{'aligned' if alignment.align > 1 else 'unaligned'}"


def check_c1_refuted(big: int, budget: int, alignment: Alignment = UNALIGNED, *, small: int = 1,
                     real: bool = False) -> Callable[[], CheckResult]:
    """A1: the pinning witness passes the per-cycle check and Z3 proves it has no layout (UNSAT).

    With `real`, the witness is embedded as real intervals: the per-cycle check is the closed
    count on those, and layouts are judged by the real same-cycle-reuse rule.
    """
    name = f"A1-c1-refuted-{_tag(big, budget, alignment, small)}{'-real' if real else ''}"
    values = pinning_counterexample(big, budget, alignment, small)
    if real:
        values = embed_in_real(values)
    return lambda: _check_no_layout(name, values, budget, alignment, real)


def _check_no_layout(name: str, values: list[Value], budget: int, alignment: Alignment, real: bool) -> CheckResult:
    """UNSAT when `values` passes the (closed) per-cycle check and Z3 proves no layout exists;
    `real` values are laid out under the real rule (on the doubled time line)."""
    if not fits_per_cycle(values, budget):
        return (name, "load-exceeds-budget", 0.0, {})
    laid_out = real_as_abstract(values) if real else values
    return _solve(name, layout_solver(laid_out, budget, alignment)[0])


def check_c1_tight(big: int, alignment: Alignment = UNALIGNED) -> Callable[[], CheckResult]:
    """A2: the 2*big-word witness has a layout with one more word (SAT)."""
    values = pinning_counterexample(big, 2 * big, alignment)
    name = f"A2-c1-tight-{_tag(big, 2 * big + 1, alignment)}"
    return lambda: _solve(name, layout_solver(values, 2 * big + 1, alignment)[0])


def check_stored_c1_counterexamples() -> CheckResult:
    """A1-stored: each stored C1 counterexample passes the per-cycle check and has no layout
    (entries with "semantics": "real" under the real same-cycle-reuse rule)."""
    t0 = time.perf_counter()
    for entry in load_counterexamples()["c1_counterexamples"]:
        real = entry.get("semantics", "abstract") == "real"
        _, result, _, _ = _check_no_layout(entry["name"], instance_from_rows(entry["values"]), entry["budget"],
                                           alignment_of(entry), real)
        if result != "unsat":
            return ("A1-stored", f"{entry['name']}:{result}", time.perf_counter() - t0, {})
    return ("A1-stored", "unsat", time.perf_counter() - t0, {})


def check_greedy_witnesses() -> CheckResult:
    """A7: each stored greedy witness passes the per-cycle check, its greedy fails, and Z3
    finds a layout (SAT). Any other result names the bad entry."""
    t0 = time.perf_counter()
    for entry in load_counterexamples()["greedy_failures"]:
        values, budget, alignment = instance_from_rows(entry["values"]), entry["budget"], alignment_of(entry)
        tag = f"{entry['allocator']}-B{budget}-align{alignment.align}"
        if not fits_per_cycle(values, budget):
            return ("A7", f"{tag}:load-exceeds-budget", time.perf_counter() - t0, {})
        if GREEDY_ALLOCATORS[entry["allocator"]](values, budget, alignment) is not None:
            return ("A7", f"{tag}:greedy-succeeds", time.perf_counter() - t0, {})
        if find_layout(values, budget, alignment) is None:
            return ("A7", f"{tag}:no-layout", time.perf_counter() - t0, {})
    return ("A7", "sat", time.perf_counter() - t0, {})


def check_c1_bounded(sizes: Sequence[int], budget: int, steps: int, alignment: Alignment = UNALIGNED, *,
                     slack: int = 0, method: str = "exhaustive") -> Callable[[], CheckResult]:
    """A6: decide C1 over every instance in the bound ("unsat" = holds, "sat" = counterexample)."""
    search = {"exhaustive": exhaustive_c1, "cegis": cegis_c1}[method]

    def run() -> CheckResult:
        t0 = time.perf_counter()
        kwargs = {"slack": slack} if slack else {}
        result, witness, work = search(sizes, budget, steps, alignment, **kwargs)
        stats: dict = {"instances" if method == "exhaustive" else "iterations": work}
        if witness is not None:
            stats["counterexample"] = instance_to_rows(witness)
        return ("A6", result, time.perf_counter() - t0, stats)

    return run


def check_c3_exhaustive(k: int, budget: int, steps: int, max_values: int) -> Callable[[], CheckResult]:
    """A5: the two-pass allocator succeeds on every instance in the bound that satisfies C3's
    hypothesis ("unsat" = no failure)."""

    def run() -> CheckResult:
        t0 = time.perf_counter()
        failure = next(c3_failures(k, budget, steps, max_values), None)
        stats = {} if failure is None else {"counterexample": instance_to_rows(failure)}
        return ("A5", "unsat" if failure is None else "sat", time.perf_counter() - t0, stats)

    return run


def _bounded(name_prefix: str, sizes: tuple[int, ...], budget: int, steps: int,
             alignment: Alignment = UNALIGNED, *, slack: int = 0, method: str = "exhaustive") -> tuple:
    """A (name, description, check) row for an A6 bounded check; the name encodes the bound."""
    size_tag = "-".join(map(str, sizes))
    align_tag = "aligned" if alignment.align > 1 else "unaligned"
    slack_tag = f"-slack{slack}" if slack else ""
    name = f"{name_prefix}-sizes{size_tag}-B{budget}-T{steps}-{align_tag}{slack_tag}"
    desc = (f"every instance with sizes {set(sizes)}, load <= {budget}, <= {steps} steps, {align_tag}: "
            f"layout in {budget + slack} words?")
    return (name, desc, check_c1_bounded(sizes, budget, steps, alignment, slack=slack, method=method))


ALIGNED8 = Alignment(8)

# Fast checks: verification/verify.py and ./scripts/test.sh proofs run these (seconds in total).
ALLOCATION_CHECKS = [
    ("A1-c1-refuted-k8-B16-unaligned", "C1 false: load <= 16 every cycle, no layout (vectors anywhere)",
     check_c1_refuted(8, 16)),
    ("A1-c1-refuted-k8-B16-aligned", "C1 false with 8-aligned vectors",
     check_c1_refuted(8, 16, ALIGNED8)),
    ("A1-c1-refuted-k8-B20-unaligned", "C1 false for B > 2k (filler scalars)",
     check_c1_refuted(8, 20)),
    ("A1-c1-refuted-k8-B24-aligned", "C1 false for B > 2k, aligned (filler vector)",
     check_c1_refuted(8, 24, ALIGNED8)),
    ("A1-c1-refuted-k8-B16-unaligned-real", "C1 false under the real same-cycle-reuse semantics",
     check_c1_refuted(8, 16, real=True)),
    *[(f"A1-c1-refuted-k{k}-B{2 * k}-unaligned", f"C1 false for sizes {{1, {k}}}", check_c1_refuted(k, 2 * k))
      for k in (2, 3, 4)],
    ("A1-c1-refuted-s2k5-B10-unaligned", "C1 false for sizes {2, 5} (sizes not dividing each other)",
     check_c1_refuted(5, 10, small=2)),
    ("A1-stored", "Stored C1 counterexamples have no layout (allocation_counterexamples.json)",
     check_stored_c1_counterexamples),
    ("A2-c1-tight-k8-B17-unaligned", "The unaligned witness fits in 2k+1 words", check_c1_tight(8)),
    ("A2-c1-tight-k8-B17-aligned", "The aligned witness fits in 2k+1 words", check_c1_tight(8, ALIGNED8)),
    ("A3-real-conflict-is-overlap", "Real expiry rule = overlap on the doubled time line",
     check_real_conflict_is_overlap),
    ("A3-real-load-bound", "Closed per-cycle count bounds the doubled-time load", check_real_load_bound),
    ("A3-real-embedding", "[s,e] -> [2s,2e+1] embeds abstract instances in the real semantics",
     check_real_embedding),
    ("A4-start-order-lemma", "Start order: an earlier value blocks the new one iff live at its start",
     check_start_order_lemma),
    ("A5-c3-exhaustive-k2-B6-T3-n5", "C3: two-pass succeeds, k=2, B=6, <= 3 steps, <= 5 values",
     check_c3_exhaustive(2, 6, 3, 5)),
    _bounded("A6-c1-bounded", (1, 2), 4, 4),
    _bounded("A6-c1-bounded", (1, 2), 4, 2, Alignment(2)),
    _bounded("A6-c1-bounded", (1, 2), 4, 3, Alignment(2)),
    _bounded("A6-c1-cegis", (1, 2), 4, 2, Alignment(2), method="cegis"),
    _bounded("A6-c1-cegis", (1, 2), 4, 3, Alignment(2), method="cegis"),
    ("A7-greedy-witnesses", "Stored greedy witnesses: a layout exists and the greedy fails",
     check_greedy_witnesses),
]

# Slow checks (--slow): minutes in total; each bound is recorded in docs/ALLOCATION_CONJECTURES.md.
SLOW_ALLOCATION_CHECKS = [
    _bounded("A6-c1-bounded", (1, 2), 4, 5),
    _bounded("A6-c1-cegis", (1, 2), 4, 4, method="cegis"),
    _bounded("A6-c1-bounded", (1, 2), 3, 6),
    _bounded("A6-c1-bounded", (1, 3), 4, 5),
    _bounded("A6-c1-bounded", (1, 3), 5, 5),
    _bounded("A6-c1-bounded", (1, 3), 6, 4),
    _bounded("A6-c1-bounded", (1, 4), 8, 4),
    _bounded("A6-c1-bounded", (1, 4), 8, 3, Alignment(4)),
    _bounded("A6-c1-bounded", (2, 3), 6, 6),
    _bounded("A6-c1-bounded", (2, 3), 7, 5),
    _bounded("A6-c1-bounded", (1, 2), 4, 5, slack=1),
    _bounded("A6-c1-bounded", (1, 3), 6, 5, slack=1),
    ("A5-c3-exhaustive-k3-B8-T3-n6", "C3: two-pass succeeds, k=3, B=8, <= 3 steps, <= 6 values",
     check_c3_exhaustive(3, 8, 3, 6)),
]

EXPECT_SAT = {
    "A2-c1-tight-k8-B17-unaligned",
    "A2-c1-tight-k8-B17-aligned",
    "A6-c1-bounded-sizes1-2-B4-T3-aligned",
    "A6-c1-cegis-sizes1-2-B4-T3-aligned",
    "A6-c1-bounded-sizes1-2-B4-T5-unaligned",
    "A6-c1-bounded-sizes1-4-B8-T3-aligned",
    "A6-c1-bounded-sizes2-3-B7-T5-unaligned",
    "A7-greedy-witnesses",
}


def run_checks(checks: Sequence[tuple], *, verbose: bool = True) -> int:
    """Run (name, description, check) rows and print PASS/FAIL. Returns the number of failures."""
    failed = 0
    for name, desc, check in checks:
        _, result, elapsed, stats = check()
        ok = result == ("sat" if name in EXPECT_SAT else "unsat")
        failed += not ok
        extra = f" {stats}" if stats and verbose else ""
        print(f"{'PASS' if ok else 'FAIL'} [{name}] {desc}: {result} ({elapsed:.2f}s){extra}", flush=True)
    return failed


def main() -> int:
    parser = argparse.ArgumentParser(description="Scratch allocation conjectures (docs/ALLOCATION_CONJECTURES.md)")
    parser.add_argument("--slow", action="store_true", help="Also run the slow bounded checks (minutes)")
    parser.add_argument("--search-greedy", action="store_true",
                        help="Re-run the seeded C2 witness search and print the entries")
    parser.add_argument("-q", "--quiet", action="store_true", help="Do not print witnesses and counts")
    args = parser.parse_args()
    if args.search_greedy:
        for min_budget in (8, 16):
            for entry in greedy_search(min_budget):
                print(json.dumps(entry))
        return 0
    checks = ALLOCATION_CHECKS + (SLOW_ALLOCATION_CHECKS if args.slow else [])
    return 1 if run_checks(checks, verbose=not args.quiet) else 0


if __name__ == "__main__":
    sys.exit(main())
