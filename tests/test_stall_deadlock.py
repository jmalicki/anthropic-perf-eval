"""The three path-bit plans on which `--greedy-budget-force stall` + inflation 0.98 fails the
scratch budget (docs/STALL_DEADLOCK.md). 16-round greedy builds take about a second each.

- Under stall + 0.98 each build fails with ScratchBudgetExceeded: a known limitation, pinned
  so a change to the greedy that fixes (or moves) it is noticed.
- At the first forced cycle (SCHEDULER_DEADLOCK_DUMP) ASAP inflation deferred ops in that
  same cycle, and the ready ops the hard budget check refused each write a vector. (Whether the
  inflation-deferred ops would have freed words is not asserted: the script's frees count is
  approximate.)
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
    # Inflation deferred ops in the forced cycle itself: the cycle was not empty for lack of work.
    assert state["inflation_deferred"]
    assert state["budget_deferred"] and all(d["reason"] == "budget" for d in state["budget_deferred"])
    r = sd.analyse(state, json.loads(sched_path.read_text()))
    # Every deferred op writes a vector (its 8 words are what the budget check refused). The
    # script's "frees" count is approximate (no lane aliases, keep_live or deltas): not asserted.
    assert r["deferred"] and all(d["adds"] == 8 for d in r["deferred"])


@pytest.mark.parametrize("name,cycles", sorted(PLANS.items()))
def test_fewer_groups_in_flight_builds_within_budget(name, cycles, monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        # A regression to a failed build must not write scheduler_output_error.json to the cwd.
        monkeypatch.setenv("SCHEDULER_EXPORT_ON_ERROR_PATH", os.path.join(tmp, "error.json"))
        path = os.path.join(tmp, "sched.json")
        instrs, stats = build_kernel_with_args(emission_plan=rows(name), verbose=False, output_json=path,
                                               **dict(JUDGE, greedy_groups_in_flight=20))
        with open(path) as f:
            sched = json.load(f)
    instrs = json.loads(json.dumps(list(instrs)))
    assert not stats["budget_exceeded"] and stats["cycles"] == cycles
    assert check({"instrs": instrs}, sched).ok
    assert validate_kernel(instrs, 10, 256, 16, 123)


def test_live_group_read_through_a_lane_is_not_reported_released():
    """Review of #88: a live vector group whose remaining reader reads one of its lanes has no
    direct reader under the group's id. With `remaining_users` > 0 in the dump the script must
    not claim it is released next cycle."""
    state = {"cycle": 5, "live_words": 8, "budget": 8, "slots": {}, "budget_deferred": [], "heap": [],
             "live": [{"vreg": 10, "words": 8, "remaining_users": 1},
                      {"vreg": 20, "words": 1, "remaining_users": 0}]}
    sched = {"ssa_ops": [
        {"id": 1, "opcode": "load", "engine": "load", "cycle": 2, "dep_ids": [], "dest": "v10_v", "srcs": []},
        {"id": 2, "opcode": "+", "engine": "alu", "cycle": 9, "dep_ids": [1], "dest": "v30_s",
         "srcs": [{"VReg": {"id": 11}}]},  # reads lane 11 of group 10
        {"id": 3, "opcode": "load", "engine": "load", "cycle": 1, "dep_ids": [], "dest": "v20_s", "srcs": []}]}
    kinds = sd.analyse(state, sched)["words_by_first_reader_state"]
    assert kinds.get("no reader left (released next cycle)") == 1  # vreg 20: no users left
    assert kinds.get("readers not resolved (lane alias, keep_live or delta)") == 8
