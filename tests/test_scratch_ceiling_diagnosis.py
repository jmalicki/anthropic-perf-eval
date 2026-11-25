"""scripts/scratch_ceiling_diagnosis.py on a hand-made schedule."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.scratch_ceiling_diagnosis import gap_report, peak_report  # noqa: E402


def _schedule():
    # Cycle 0: a const load. Two selects are ready at cycle 1; the one flow slot runs one at
    # cycle 1 and the other at cycle 2. Each feeds a load, so the load fed by the late select
    # runs at cycle 3.
    ssa_ops = [
        {"id": 0, "engine": "load", "opcode": "const", "dep_ids": [], "cycle": 0},
        {"id": 1, "engine": "flow", "opcode": "select_choice", "dep_ids": [0], "cycle": 2},
        {"id": 2, "engine": "flow", "opcode": "select_choice", "dep_ids": [0], "cycle": 1},
        {"id": 3, "engine": "load", "opcode": "load", "dep_ids": [1], "cycle": 3},
        {"id": 4, "engine": "load", "opcode": "load", "dep_ids": [2], "cycle": 2},
    ]
    ops = [{"id": op["id"], "engine": op["engine"], "cycle": op["cycle"]} for op in ssa_ops]
    return {"total_cycles": 5, "ops": ops, "ssa_ops": ssa_ops}


def test_gap_splits_idle_load_slots_by_cause():
    r = gap_report(_schedule(), t=1, q=1, w=2)
    assert (r["bound"], r["gap"], r["tail_cycles"]) == (3, 2, 0)
    # Window [1, 4): two idle slots at cycle 1, one each at cycles 2 and 3.
    assert r["idle_slots"] == 4
    assert r["early_loads_in_window"] == 0
    assert r["classes"] == {
        # Cycle 1 and 3: no load's dependency head allows it there.
        "no load could be ready (dependency heads)": 3,
        # Cycle 2: load 3 could run (head 2), but its select waited for the full flow slot.
        "starved: delayed flow op, engine full": 1,
    }


def test_peak_reports_cycle_and_kinds():
    sched = {
        "liveness_curve": [{"total_live_words": w} for w in (1, 9, 17, 9)],
        "vregs": [
            {"interval_start": 0, "interval_end": 4, "size_words": 1, "var_type": None, "phase": None},
            {"interval_start": 1, "interval_end": 4, "size_words": 8, "var_type": "HashState", "phase": "hash"},
            {"interval_start": 2, "interval_end": 3, "size_words": 8, "var_type": None, "phase": "hash"},
        ],
    }
    r = peak_report(sched)
    assert (r["peak"], r["peak_cycle"]) == (17, 2)
    assert r["live_by_kind"] == {"HashState": 8, "hash": 8, "unknown": 1}
