"""scripts/greedy_gap_diagnosis.py on a hand-made schedule."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.greedy_gap_diagnosis import engine_use, report, waiting_counts  # noqa: E402


def _schedule():
    # Cycle 0: a const load; cycle 1: one gather (the other load slot idle while load 3 is
    # ready); cycle 2: load 3 and a multiply_add that was ready since cycle 1.
    ssa_ops = [
        {"id": 0, "engine": "load", "opcode": "const", "dep_ids": [], "cycle": 0},
        {"id": 1, "engine": "load", "opcode": "load", "dep_ids": [0], "cycle": 1},
        {"id": 2, "engine": "valu", "opcode": "multiply_add", "dep_ids": [0], "cycle": 2},
        {"id": 3, "engine": "load", "opcode": "load", "dep_ids": [0], "cycle": 2},
        {"id": 4, "engine": "valu", "opcode": "^", "dep_ids": [0], "cycle": 1},
    ]
    ops = [{"engine": op["engine"], "cycle": op["cycle"]} for op in ssa_ops]
    return {"total_cycles": 3, "ops": ops, "ssa_ops": ssa_ops}


def test_engine_use_and_waiting():
    sched = _schedule()
    use = engine_use(sched)
    assert use["load"] == [1, 1, 1]
    assert use["valu"] == [0, 1, 1]
    assert waiting_counts(sched, lambda op: op["engine"] == "load") == [0, 1, 0]
    assert waiting_counts(sched, lambda op: op["opcode"] == "multiply_add") == [0, 1, 0]


def test_report_splits_idle_load_slots():
    text = report(_schedule(), slice_len=3, start=1)
    # From cycle 1: one idle load slot with load 3 waiting (cycle 1), one with none (cycle 2).
    assert "1 with a ready load waiting, 1 with none ready" in text
    assert "last load at cycle 2" in text
