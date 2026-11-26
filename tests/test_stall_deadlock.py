"""The three path-bit plans on which `--greedy-budget-force stall` + inflation 0.98 fails the
scratch budget (docs/STALL_DEADLOCK.md). 16-round greedy builds take about a second each.

- Under stall + 0.98 each build fails with ScratchBudgetExceeded: a known limitation, pinned
  so a change to the greedy that fixes (or moves) it is noticed.
- The first forced cycle (SCHEDULER_DEADLOCK_DUMP) is not a deadlock: ASAP inflation has
  every op that would free words waiting for a later cycle, and the ops it let through all add
  words and free none.
- With 20 groups in flight each plan builds within the budget (proved and validated).
"""
import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scheduler_rs  # noqa: E402
from generate_ssa_kernel import validate_kernel  # noqa: E402
from kernel_build_args import build_kernel_with_args  # noqa: E402
from scripts import stall_deadlock_diagnosis as sd  # noqa: E402
from solver.emission_plan import expand_plan, load_plan  # noqa: E402
from verification.kernel_dataflow import check  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JUDGE = dict(use_greedy=True, greedy_barriers="none", greedy_priority="wavefront", greedy_split_shift=True,
             greedy_load_priority="final-last", greedy_groups_in_flight=32, greedy_stagger_cycles=9,
             merge_hash_stages=True, index_form="mirrored", select_bit_source="path",
             greedy_budget_force="stall", pressure_inflation_threshold=0.98)
PLANS = {"floor10": 1394, "gm6": 1159, "gm14": 1281}  # cycles with 20 groups in flight


def rows(name):
    return expand_plan(load_plan(os.path.join(ROOT, f"docs/emission_plans/stall_deadlock_{name}.json")), 16, 32, 10)


@pytest.mark.parametrize("name", sorted(PLANS))
def test_stall_fails_after_an_inflation_emptied_cycle(name, monkeypatch, tmp_path):
    state_path = tmp_path / "state.json"
    monkeypatch.setenv("SCHEDULER_DEADLOCK_DUMP", str(state_path))
    sched_path = tmp_path / "sched.json"
    monkeypatch.setenv("SCHEDULER_EXPORT_ON_ERROR_PATH", str(sched_path))
    with pytest.raises(scheduler_rs.ScratchBudgetExceeded):
        build_kernel_with_args(emission_plan=rows(name), verbose=False, **JUDGE)
    state = json.loads(state_path.read_text())
    assert state["live_words"] > state["budget"] - 8  # less than one vector free
    assert any(h["ready_at"] > state["cycle"] for h in state["heap"])  # work waiting: not a deadlock
    r = sd.analyse(state, json.loads(sched_path.read_text()))
    # Every deferred op writes a vector (its 8 words are what the budget check refused). The
    # script's "frees" count is approximate (no lane aliases, keep_live or deltas): not asserted.
    assert r["deferred"] and all(d["adds"] == 8 for d in r["deferred"])


@pytest.mark.parametrize("name,cycles", sorted(PLANS.items()))
def test_fewer_groups_in_flight_builds_within_budget(name, cycles):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "sched.json")
        instrs, stats = build_kernel_with_args(emission_plan=rows(name), verbose=False, output_json=path,
                                               **dict(JUDGE, greedy_groups_in_flight=20))
        with open(path) as f:
            sched = json.load(f)
    instrs = json.loads(json.dumps(list(instrs)))
    assert not stats["budget_exceeded"] and stats["cycles"] == cycles
    assert check({"instrs": instrs}, sched).ok
    assert validate_kernel(instrs, 10, 256, 16, 123)
