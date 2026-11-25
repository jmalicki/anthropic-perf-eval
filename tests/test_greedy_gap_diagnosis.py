"""scripts/greedy_gap_diagnosis.py on hand-made schedules."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.greedy_gap_diagnosis import (  # noqa: E402
    engine_use, is_data_load, operand_ready_waiting, report, summary)


def _ssa(op_id, engine, opcode, deps, cycle):
    return {"id": op_id, "engine": engine, "opcode": opcode, "dep_ids": deps, "cycle": cycle}


def _schedule():
    # Cycle 0: a constant on the load engine; cycle 1: one gather (the other load slot idle
    # while load 3 is operand-ready); cycle 2: load 3 and a multiply_add that was
    # operand-ready since cycle 1.
    ssa_ops = [
        _ssa(0, "load", "const_choice", [], 0),
        _ssa(1, "load", "load", [0], 1),
        _ssa(2, "valu", "multiply_add", [0], 2),
        _ssa(3, "load", "load", [0], 2),
        _ssa(4, "valu", "^", [0], 1),
    ]
    ops = [{"engine": op["engine"], "cycle": op["cycle"],
            "opcode": "const" if op["opcode"] == "const_choice" else op["opcode"]} for op in ssa_ops]
    return {"total_cycles": 3, "ops": ops, "ssa_ops": ssa_ops}


def test_engine_use_and_waiting():
    sched = _schedule()
    use = engine_use(sched)
    assert use["load"] == [1, 1, 1]
    assert use["valu"] == [0, 1, 1]
    assert operand_ready_waiting(sched, is_data_load) == [0, 1, 0]
    assert operand_ready_waiting(sched, lambda op: op["opcode"] == "multiply_add") == [0, 1, 0]


def test_report_splits_idle_load_slots():
    text = report(_schedule(), slice_len=3, start=1)
    # From cycle 1: one idle load slot with load 3 operand-ready (cycle 1), one with none (2).
    assert "1 with no data load operand-ready (a lower bound" in text
    assert "1 with one operand-ready (an upper bound" in text
    assert "last data load at cycle 2" in text
    assert "1 as const loads, 0 as flow add_imm" in text


def test_constants_are_not_load_queue_entries():
    """A constant lowered to add_imm (flow) is not waiting for a load slot."""
    ssa_ops = [
        _ssa(0, "load", "load", [], 0),
        _ssa(1, "valu", "^", [0], 1),
        _ssa(2, "load", "const_choice", [], 3),  # issued late, on flow
    ]
    ops = [{"engine": "load", "cycle": 0, "opcode": "load"},
           {"engine": "valu", "cycle": 1, "opcode": "^"},
           {"engine": "flow", "cycle": 3, "opcode": "add_imm"}]
    sched = {"total_cycles": 4, "ops": ops, "ssa_ops": ssa_ops}
    assert operand_ready_waiting(sched, is_data_load) == [0, 0, 0, 0]
    s = summary(sched, start=0)
    # Load slots idle in cycles 1-3 (6) plus one in cycle 0: none has a data load waiting.
    assert s["idle_without_operand_ready_load"] == 7
    assert s["idle_with_operand_ready_load"] == 0
    assert (s["constants_as_load"], s["constants_as_add_imm"]) == (0, 1)


def test_operand_ready_is_an_upper_bound():
    """Only SSA dep_ids with unit latency are used: a load held back by anything else (here,
    nothing in the JSON explains why load 2 waited until cycle 4) still counts as
    operand-ready from cycle 1, so 'with one operand-ready' over-counts and 'with no data
    load operand-ready' under-counts."""
    ssa_ops = [
        _ssa(0, "alu", "+", [], 0),
        _ssa(1, "load", "load", [0], 4),
        _ssa(2, "load", "load", [], 4),
    ]
    ops = [{"engine": "alu", "cycle": 0, "opcode": "+"},
           {"engine": "load", "cycle": 4, "opcode": "load"},
           {"engine": "load", "cycle": 4, "opcode": "load"}]
    sched = {"total_cycles": 5, "ops": ops, "ssa_ops": ssa_ops}
    assert operand_ready_waiting(sched, is_data_load) == [1, 2, 2, 2, 0]
    s = summary(sched, start=0)
    # Cycles 0-3: 8 idle slots, all counted as "operand-ready" (7 + 1 capped by 2 waiting).
    assert s["idle_with_operand_ready_load"] == 7
    assert s["idle_without_operand_ready_load"] == 1
