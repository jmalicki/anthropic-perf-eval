"""Forced encoding modes must build a kernel that equals its SSA program.

These are the options the bisection tools rely on (--valu-shift, --const-choice,
--select-choice). Forcing a mode used to make CP-SAT report INFEASIBLE, because domains
were bounded by a greedy schedule that used different encodings (BUGS.md).
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scheduler_rs  # noqa: E402
from verification.kernel_dataflow import check  # noqa: E402

FORCED = [
    {"valu_shift": "alu"},
    {"valu_shift": "valu"},
    {"const_choice": "add_imm"},
    {"const_choice": "load"},
    {"select_choice": "fma"},
    {"select_choice": "vselect"},
]


@pytest.mark.parametrize("mode", FORCED, ids=lambda m: "-".join(f"{k}={v}" for k, v in m.items()))
def test_forced_mode_builds_and_matches_ssa(mode, tmp_path):
    sched_path = tmp_path / "sched.json"
    instrs, stats = scheduler_rs.build_tick_tock_kernel_ssa(
        forest_height=10, n_nodes=2047, batch_size=16, rounds=3, enable_valu_shift=True,
        verbose=False, use_cpsat=True, cpsat_time_limit=30.0, output_json=str(sched_path), **mode,
    )
    assert not stats["budget_exceeded"]
    with open(sched_path) as f:
        sched = json.load(f)
    report = check({"instrs": json.loads(json.dumps(list(instrs)))}, sched)
    assert report.ok, report.findings[:2]

    lowered = sched["ops"]
    if mode.get("valu_shift") == "valu":
        assert sched["valu_shifted"] == 0
    if mode.get("valu_shift") == "alu":
        assert sched["valu_shifted"] > 0
    if mode.get("const_choice") == "load":
        assert not any(op["opcode"] == "add_imm" for op in lowered)
    if mode.get("select_choice") == "vselect":
        assert not any(op.get("from_select_choice") is not None and op["engine"] == "valu" for op in lowered)
    if mode.get("select_choice") == "fma":
        assert not any(op["opcode"] == "vselect" for op in lowered)
