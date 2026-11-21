"""Scratch allocation conjectures (verification/allocation_conjectures.py, docs/ALLOCATION_CONJECTURES.md).

Runs the fast checks and guards the pieces they rely on: the backtracking layout search
agrees with Z3, the doubled-time model matches address_assign.rs's expiry rule, the greedy
allocators return valid layouts, and C3's guarantee holds on random instances.
"""

import itertools
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("z3")

import z3  # noqa: E402

from verification import allocation_conjectures as ac  # noqa: E402


@pytest.mark.parametrize("name,desc,check", ac.ALLOCATION_CHECKS, ids=[c[0] for c in ac.ALLOCATION_CHECKS])
def test_allocation_check(name, desc, check):
    _, result, _, _ = check()
    assert result == ("sat" if name in ac.EXPECT_SAT else "unsat"), desc


def _random_instance(rng: random.Random, k: int, steps: int, max_values: int) -> list[ac.Value]:
    values = []
    for _ in range(rng.randint(1, max_values)):
        s = rng.randrange(steps)
        values.append(ac.Value(s, rng.randrange(s, steps), rng.choice((1, 1, k))))
    return values


@pytest.mark.parametrize("alignment", [ac.UNALIGNED, ac.Alignment(8)], ids=["unaligned", "aligned"])
def test_search_layout_agrees_with_z3(alignment):
    """The exhaustive searches trust search_layout's None; it must match Z3 exactly."""
    rng = random.Random(7)
    checked = 0
    while checked < 150:
        budget = rng.choice((16, 17, 18, 24))
        values = _random_instance(rng, 8, 6, 12)
        if not ac.fits_per_cycle(values, budget):
            continue
        checked += 1
        assert (ac.search_layout(values, budget, alignment) is None) == (ac.find_layout(values, budget, alignment) is None), values


def _rust_shares(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """address_assign.rs: may the later-starting interval reuse the earlier one's words?"""
    first, second = sorted((a, b))
    if second[0] == second[1]:  # allocating a point interval: only values ended before it
        return first[1] < second[0]
    return first[1] < second[0] or (first[1] == second[0] and first[0] != first[1])


def test_real_conflict_matches_address_assign_rule():
    intervals = [(s, e) for s in range(5) for e in range(s, 5)]
    for a, b in itertools.product(intervals, repeat=2):
        conflict = z3.is_true(z3.simplify(ac.real_conflict(a[0], a[1], b[0], b[1])))
        assert conflict == (not _rust_shares(a, b)), (a, b)


@pytest.mark.parametrize("k", range(2, 9))
def test_pinning_counterexample_family(k):
    for alignment, budget in ((ac.UNALIGNED, 2 * k), (ac.UNALIGNED, 2 * k + 3), (ac.Alignment(k), 3 * k)):
        values = ac.pinning_counterexample(k, budget, alignment)
        assert ac.fits_per_cycle(values, budget)
        assert ac.search_layout(values, budget, alignment) is None


@pytest.mark.parametrize("name", sorted(ac.GREEDY_ALLOCATORS))
def test_greedy_layouts_are_valid(name):
    rng = random.Random(11)
    allocate = ac.GREEDY_ALLOCATORS[name]
    for _ in range(300):
        values, budget = _random_instance(rng, 8, 8, 14), rng.choice((16, 24, 32))
        addrs = allocate(values, budget, ac.UNALIGNED)
        if addrs is not None:
            assert ac.is_valid_layout(values, budget, addrs)


def test_c3_two_pass_bound_on_random_instances():
    """C3: k * max_v + max_s <= B implies the two-pass allocator succeeds (k = 8)."""
    rng = random.Random(3)
    for _ in range(500):
        values = _random_instance(rng, 8, 10, 20)
        budget = 8 * ac.max_live(values, 8) + ac.max_live(values, 1)
        addrs = ac.vectors_then_scalars(values, budget)
        assert addrs is not None and ac.is_valid_layout(values, budget, addrs, ac.Alignment(8)), values


def test_real_entries_need_the_real_rule():
    """The abstract witness read as real intervals has a layout, so A1-stored must apply the
    real rule to "semantics": "real" entries rather than the abstract overlap."""
    values = ac.pinning_counterexample(8, 16)
    assert ac.search_layout(ac.real_as_abstract(values), 16) is not None
    assert ac.search_layout(ac.real_as_abstract(ac.embed_in_real(values)), 16) is None


def test_greedy_witnesses_reproduce_from_seeded_search():
    """--search-greedy regenerates every stored greedy witness (deterministic search)."""
    found = [{k: e[k] for k in ("allocator", "align", "budget", "values")}
             for b in (8, 16) for e in ac.greedy_search(b)]
    for entry in ac.load_counterexamples()["greedy_failures"]:
        stored = {"allocator": entry["allocator"], "align": entry.get("align", 1),
                  "budget": entry["budget"], "values": entry["values"]}
        assert stored in found, entry["allocator"]


def test_stored_counterexamples_are_documented():
    data = ac.load_counterexamples()
    for entry in data["c1_counterexamples"] + data["greedy_failures"]:
        assert entry["demonstrates"] and entry["budget"] > 0 and entry["values"]
    assert {e["allocator"] for e in data["greedy_failures"]} == set(ac.GREEDY_ALLOCATORS)
