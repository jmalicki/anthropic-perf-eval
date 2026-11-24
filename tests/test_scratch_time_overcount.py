"""Tests for scripts.scratch_time_overcount (CP-SAT cycle count vs exact phase count)."""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import scratch_time_overcount as st  # noqa: E402


def _sched(rows):
    return {"total_cycles": 10, "vregs": [{"interval_start": s, "interval_end": e, "size_words": w}
                                          for s, e, w in rows]}


def test_chain_handoff_is_counted_once_by_phase_and_twice_by_cycle():
    # v0 [0,1] is read at 1 while v1 [1,2] is written at 1: one 8-word block suffices.
    rep = st.report(_sched([(0, 1, 8), (1, 2, 8), (2, 3, 8)]))
    assert rep["peak_phase_count"] == 8
    assert rep["peak_cycle_count"] == 16
    assert rep["max_overcount"] == 8


def test_point_interval_holds_its_cycle():
    # A point value [3,3] and one written at 3 overlap in both counts.
    rep = st.report(_sched([(3, 3, 8), (3, 5, 8)]))
    assert rep["peak_phase_count"] == 16 and rep["peak_cycle_count"] == 16


def test_phase_never_exceeds_cycle():
    rng = random.Random(7)
    for _ in range(200):
        rows = []
        for _ in range(rng.randint(1, 8)):
            s = rng.randint(0, 6)
            rows.append((s, s + rng.randint(0, 4), rng.choice((1, 8))))
        for c, p in st.per_cycle_counts(st.value_intervals(_sched(rows))):
            assert p <= c
