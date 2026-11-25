"""scripts/shift_limit_diagnosis.py on a hand-made schedule."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.shift_limit_diagnosis import analyze, band_of, main, report  # noqa: E402


def _si(rnd):
    return {"round": rnd, "phase": None, "group": 0, "tree_depth": None, "iter": None}


def _schedule():
    """Three cycles.

    cycle 0: six vector ops on the vector ALU (five shiftable XORs of round 0 and one
             multiply_add); a seventh XOR (id 6) is ready and shifted to 8 scalar lanes.
    cycle 1: six XORs of round 6 fill the vector ALU while a multiply_add (id 13, ready at 1)
             waits for cycle 2: priority inversion, 12 scalar slots idle.
    cycle 2: the multiply_add and a select of round 13 on flow; the vector ALU has 5 idle slots.
    """
    ssa, ops = [], []

    def add(i, engine, opcode, cycle, rnd, deps=(), lanes=False, select=False):
        ssa.append({"id": i, "engine": engine, "opcode": opcode, "dep_ids": list(deps), "cycle": cycle,
                    "source_info": _si(rnd), "srcs": [], "dest": None, "var_type": "x"})
        if lanes:
            for lane in range(8):
                ops.append({"id": i * 1000 + lane, "engine": "alu", "opcode": opcode, "cycle": cycle})
        elif select:
            ops.append({"id": i, "engine": "flow", "opcode": "vselect", "cycle": cycle, "from_select_choice": i})
        else:
            ops.append({"id": i, "engine": engine, "opcode": opcode, "cycle": cycle})

    for i in range(5):
        add(i, "valu", "^", 0, 0)
    add(5, "valu", "multiply_add", 0, 0)
    add(6, "valu", "^", 0, 0, lanes=True)
    for i in range(7, 13):
        add(i, "valu", "^", 1, 6, deps=[0])
    add(13, "valu", "multiply_add", 2, 6, deps=[5])
    add(14, "flow", "select_choice", 2, 13, deps=[5], select=True)
    ssa[-1]["select_fma_vector_ops"] = 1
    return {"total_cycles": 3, "ops": ops, "ssa_ops": ssa, "peak_live_words": 10}


def test_band_of():
    assert band_of(None) == "setup"
    assert band_of(4) == "r0-4"
    assert band_of(5) == "r5-10"
    assert band_of(14) == "r11-14"
    assert band_of(15) == "r15"


def test_analyze_accounting_and_classes():
    a = analyze(_schedule())
    assert (a["cycles"], a["vector_ops"], a["shifted"]) == (3, 14, 1)
    # 13 vector slots used = 6 * 3 - 5 idle.
    assert a["valu_idle"] == 5 and a["identity_holds"]
    assert a["alu_idle"]["inversion (fixed waiting)"] == 12  # cycle 1
    assert a["alu_idle"]["full, nothing waiting"] == 4  # cycle 0: 8 lanes used
    assert a["alu_idle"]["valu not full"] == 12  # cycle 2
    assert a["bands"]["r0-4"]["shiftable"] == 6 and a["bands"]["r0-4"]["shifted"] == 1
    assert a["bands"]["r5-10"]["alu_idle"] == 24  # cycles 1 and 2 hold round-6 vector work
    assert a["selects"] == {13: {"flow": 1}}
    assert a["flow_wait"] == {13: 1.0}


def test_report_and_cli(tmp_path, capsys):
    text = report("s", analyze(_schedule()), floor=2, bound=None)
    # 1 cycle above the floor: 6 * 1 - 5 idle = 1 missing shift.
    assert "1 cycles above it = 1 shifts 'missing'" in text
    assert "identity V - s = 6T - I: holds" in text
    import json
    path = tmp_path / "s.json"
    path.write_text(json.dumps(_schedule()))
    assert main([str(path), "--floor", "2", "--no-static-bound"]) == 0
    assert "inversion (fixed waiting)" in capsys.readouterr().out


def test_ready_waits_for_the_last_lane_of_a_split_shift():
    """A split-shifted producer (lanes in cycles 0 and 1) is ready for its consumer at cycle 2,
    not 1: the consumer issued at cycle 2 did not wait (review of #76)."""
    ssa = [{"id": 0, "engine": "valu", "opcode": "^", "dep_ids": [], "cycle": 1, "source_info": _si(0),
            "srcs": [], "dest": None, "var_type": "x"},
           {"id": 1, "engine": "valu", "opcode": "^", "dep_ids": [0], "cycle": 2, "source_info": _si(0),
            "srcs": [], "dest": None, "var_type": "x"}]
    ops = [{"id": lane, "engine": "alu", "opcode": "^", "cycle": 0 if lane < 4 else 1} for lane in range(8)]
    ops.append({"id": 1, "engine": "valu", "opcode": "^", "cycle": 2})
    d = {"total_cycles": 3, "ops": ops, "ssa_ops": ssa, "peak_live_words": 0,
         "shifted_alu_cycles": {"0": [0, 0, 0, 0, 1, 1, 1, 1]}}
    a = analyze(d)
    assert a["shifted"] == 1
    assert a["bands"]["r0-4"]["ready_wait"] == 0


def test_static_bound_import_error_is_reported(monkeypatch, capsys):
    """Without scripts.static_bounds (e.g. numpy missing) the bound is skipped with a warning,
    and the report says so instead of silently dropping the forced/loss split."""
    from scripts import shift_limit_diagnosis as sld

    monkeypatch.setitem(sys.modules, "scripts.static_bounds", None)
    assert sld.static_bound(_schedule()) is None
    assert "static head/tail bound not computed" in capsys.readouterr().err
    assert "static head/tail bound not computed" in report("s", analyze(_schedule()), floor=2, bound=None)
