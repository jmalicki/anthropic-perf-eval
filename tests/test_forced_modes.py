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


@pytest.mark.deep
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


@pytest.mark.deep
def test_forced_fma_hint_is_feasible_at_3_rounds_batch_64(tmp_path):
    """--select-choice fma at 3 rounds / batch 64 used to hand CP-SAT an infeasible hint.

    The greedy scheduled readers of a multiply-add select one cycle after it (the vselect
    latency) and counted one vector slot for it; the hint gave the select's chosen-cycle and
    the CSE shared subtract the vselect timing too. CP-SAT reported "The solution hint is
    complete, but it is infeasible" and found no solution within the budget (BUGS.md).
    """
    import subprocess

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ)
    env.pop("CPSAT_MAX_DETERMINISTIC_TIME", None)
    result = subprocess.run(
        [sys.executable, os.path.join(root, "generate_ssa_kernel.py"), "--rounds", "3", "--batch-size", "64",
         "--select-choice", "fma", "--cpsat-deterministic-time", "2", "--cpsat-time-limit", "300",
         "--no-validate", "--output-json", str(tmp_path / "sched.json"),
         "--save-kernel", str(tmp_path / "kernel.json")],
        cwd=root, env=env, capture_output=True, text=True, timeout=600,
    )
    log = result.stdout + result.stderr
    assert result.returncode == 0, log[-3000:]
    hint_lines = [line for line in log.splitlines() if line.startswith("The solution hint is")]
    assert hint_lines and "complete and is feasible" in hint_lines[-1], hint_lines
