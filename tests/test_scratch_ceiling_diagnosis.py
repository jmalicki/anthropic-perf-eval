"""scripts/scratch_ceiling_diagnosis.py on hand-made schedules."""
import os
import sys

import pytest

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


# The load row: t = 1, q = 0, W = the two loads with head >= 1.
ROW = {"t": 1, "q": 0, "w": 2, "bound": 2, "window_ids": {3, 4}, "heads": {0: 0, 1: 1, 2: 1, 3: 2, 4: 2}}


def test_gap_identity_and_attribution():
    r = gap_report(_schedule(), dict(ROW))
    # Window [1, 5): idle 2 at cycle 1, 1 at 2, 1 at 3, 2 at 4. gap = 5 - 2 = 3 = 6 / 2.
    assert (r["bound"], r["gap"], r["idle_slots"], r["extra_loads"]) == (2, 3, 6, 0)
    assert r["classes"] == {
        # Cycles 1, 3 and 4: no load's dependency head allows one there.
        "no load's dependency head allows it": 5.0,
        # Cycle 2: load 3 could run (head 2); its select waited for the full flow slot.
        "starved: delayed flow op, no room": 1.0,
    }


def test_row_and_schedule_must_agree():
    """Given the counts the identity is arithmetic; what can disagree is the row: a window
    load outside [t, T - q), or W differing from the window loads issued. Both fail loudly."""
    with pytest.raises(AssertionError, match="outside"):
        gap_report(_schedule(), dict(ROW, window_ids={0, 3, 4}))  # load 0 issued at cycle 0 < t
    with pytest.raises(AssertionError, match="W = 3"):
        gap_report(_schedule(), dict(ROW, w=3))
    # W = 1 with one window load is consistent: load 4 is then an extra load.
    r = gap_report(_schedule(), dict(ROW, w=1, window_ids={3}))
    assert (r["extra_loads"], r["gap"]) == (1, 3)


def test_attribution_splits_a_slot_between_loads_of_the_same_cycle():
    """Two loads issue at the next load cycle; the idle slot is split between their chains,
    whatever order they are listed in."""
    ssa_ops = [
        {"id": 0, "engine": "flow", "opcode": "select_choice", "dep_ids": [], "cycle": 1},
        {"id": 1, "engine": "flow", "opcode": "select_choice", "dep_ids": [], "cycle": 0},
        {"id": 2, "engine": "alu", "opcode": "+", "dep_ids": [], "cycle": 1},
        {"id": 3, "engine": "load", "opcode": "load", "dep_ids": [0], "cycle": 2},
        {"id": 4, "engine": "load", "opcode": "load", "dep_ids": [2], "cycle": 2},
    ]
    row = {"t": 1, "q": 0, "w": 2, "bound": 2, "window_ids": {3, 4}, "heads": {0: 0, 1: 0, 2: 0, 3: 1, 4: 1}}
    for order in (ssa_ops, list(reversed(ssa_ops))):
        sched = {"total_cycles": 3, "ops": [{"id": o["id"], "engine": o["engine"], "cycle": o["cycle"]} for o in order],
                 "ssa_ops": order}
        r = gap_report(sched, dict(row))
        # Cycle 1 has 2 idle load slots: half to the select (flow full at cycle 0), half to
        # the add (scalar ALU had room at cycle 0).
        assert r["classes"] == {"starved: delayed flow op, no room": 1.0, "starved: delayed alu op, room": 1.0}


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


def test_blame_attributes_a_vector_select_to_the_vector_alu():
    """Review of #83: a SelectChoice lowered to a multiply_add keeps SSA engine flow but runs on
    the vector ALU. With the vector ALU full at cycle 0 and flow empty, the select issued at 1
    feeding a load is a delayed vector op with no room, not a flow op with room."""
    from scripts.scratch_ceiling_diagnosis import _uses, blame, executed_ops

    def op(i, engine, opcode, cycle, deps=()):
        return {"id": i, "engine": engine, "opcode": opcode, "cycle": cycle, "dep_ids": list(deps)}

    ssa = [op(i, "valu", "multiply_add", 0) for i in range(6)]
    ssa += [op(6, "flow", "select_choice", 1), op(7, "load", "load", 2, deps=[6])]
    lowered = [dict(o) for o in ssa[:6]]
    lowered += [{"id": 6, "engine": "valu", "opcode": "multiply_add", "cycle": 1, "from_select_choice": 6},
                {"id": 7, "engine": "load", "opcode": "load", "cycle": 2}]
    sched = {"total_cycles": 3, "ssa_ops": ssa, "ops": lowered}
    use, _, _ = _uses(sched)
    assert blame(7, executed_ops(sched), use) == ("valu", True, 6)
    # A select lowered to a vselect stays a flow op.
    lowered[6] = {"id": 6, "engine": "flow", "opcode": "vselect", "cycle": 1, "from_select_choice": 6}
    use, _, _ = _uses(sched)
    assert blame(7, executed_ops(sched), use) == ("flow", False, 6)
