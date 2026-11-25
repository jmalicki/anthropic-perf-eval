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


# ---------------------------------------------------------------------------
# Individually forced shifts (forced_shifts=[op ids], scripts/what_if_shift.py)
# ---------------------------------------------------------------------------

SMALL = dict(forest_height=10, n_nodes=2047, batch_size=16, rounds=2, enable_valu_shift=True)

_FORCED_SHIFT_BUILD = """
import json, sys
import scheduler_rs
kwargs = json.loads(sys.argv[1])
instrs, stats = scheduler_rs.build_tick_tock_kernel_ssa(**kwargs)
with open(sys.argv[2], "w") as f:
    json.dump({"instrs": list(instrs), "budget_exceeded": stats["budget_exceeded"]}, f)
"""


def _two_vector_ops_the_greedy_runs_together(tmp_path):
    """Ids of two shiftable vector ops that the unforced greedy leaves on vector slots in one
    cycle. Shifted, they need 16 scalar slots, so they cannot share a cycle."""
    path = tmp_path / "free.json"
    scheduler_rs.build_tick_tock_kernel_ssa(**SMALL, verbose=False, use_cpsat=False, output_json=str(path))
    with open(path) as f:
        sched = json.load(f)
    assert sched["valu_shifted"] == 0, "the unforced greedy is expected to shift nothing at this size"
    by_cycle = {}
    for op in sched["ssa_ops"]:
        if op["engine"] == "valu" and op["opcode"] not in ("vbroadcast", "multiply_add"):
            by_cycle.setdefault(op["cycle"], []).append(op["id"])
    pair = next(ids[:2] for _, ids in sorted(by_cycle.items()) if len(ids) >= 2)
    return pair


def _lowered_lanes(sched, op_id):
    """Lowered scalar ops of a shifted SSA op (ids op_id * 1000 + lane)."""
    return [op for op in sched["ops"] if op["id"] // 1000 == op_id and op["engine"] == "alu"]


def _forced_shift_build(tmp_path, forced, **kwargs):
    """Build in a subprocess (the solver log goes to the process's stdout/stderr)."""
    import subprocess

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sched_path, kernel_path = tmp_path / "sched.json", tmp_path / "kernel.json"
    args = dict(SMALL, forced_shifts=forced, output_json=str(sched_path), **kwargs)
    env = dict(os.environ)
    env.pop("CPSAT_MAX_DETERMINISTIC_TIME", None)
    result = subprocess.run(
        [sys.executable, "-c", _FORCED_SHIFT_BUILD, json.dumps(args), str(kernel_path)],
        cwd=root, env=env, capture_output=True, text=True, timeout=600,
    )
    log = result.stdout + result.stderr
    assert result.returncode == 0, log[-3000:]
    with open(sched_path) as f:
        sched = json.load(f)
    with open(kernel_path) as f:
        kernel = json.load(f)
    assert not kernel["budget_exceeded"]
    report = check({"instrs": kernel["instrs"]}, sched)
    assert report.ok, report.findings[:2]
    return sched, log


def test_forced_shifts_reach_cpsat_as_a_feasible_hint(tmp_path):
    """The greedy must place individually forced shifts as 8 scalar ops.

    The CP-SAT model fixes their shift literals to 1 and hints their lanes at the greedy's
    cycle. The greedy used to ignore the forced set, so two forced ops it had put on vector
    slots of one cycle were hinted as 16 scalar ops in that cycle: CP-SAT reported "The
    solution hint is complete, but it is infeasible", and domains and horizon were bounded by
    a schedule that is not a schedule of the model (BUGS.md).
    """
    forced = _two_vector_ops_the_greedy_runs_together(tmp_path)
    sched, log = _forced_shift_build(
        tmp_path, forced, verbose=True, use_cpsat=True, cpsat_time_limit=300.0, cpsat_deterministic_time=2.0,
    )
    hint_lines = [line for line in log.splitlines() if line.startswith("The solution hint is")]
    assert hint_lines and all("complete and is feasible" in line for line in hint_lines), hint_lines
    assert "SSA Greedy hint:" in log and "(2 VALU shifted)" in log, [
        line for line in log.splitlines() if "Greedy hint" in line
    ]
    for op_id in forced:
        assert str(op_id) in sched["shifted_alu_cycles"], (op_id, sched["shifted_alu_cycles"])
        assert len(_lowered_lanes(sched, op_id)) == 8


def test_forced_shifts_are_honoured_by_the_greedy_alone(tmp_path):
    """use_cpsat=False: the emitted kernel runs the forced ops as scalar ops too."""
    forced = _two_vector_ops_the_greedy_runs_together(tmp_path)
    sched, _ = _forced_shift_build(tmp_path, forced, verbose=False, use_cpsat=False)
    assert sched["valu_shifted"] == 2
    lanes = [_lowered_lanes(sched, op_id) for op_id in forced]
    assert [len(ops) for ops in lanes] == [8, 8]
    # Each forced op is placed whole (all 8 lanes in one cycle), and one whole shift fits
    # per cycle, so the two ops no longer share a cycle.
    cycles = [{op["cycle"] for op in ops} for ops in lanes]
    assert [len(c) for c in cycles] == [1, 1], cycles
    assert cycles[0].isdisjoint(cycles[1])


@pytest.mark.parametrize("valu_shift", ["valu", "alu"])
def test_forced_shifts_are_ignored_outside_optimize_by_the_greedy_alone(tmp_path, valu_shift):
    """use_cpsat=False with --valu-shift valu|alu: the forced set is not applied.

    The CP-SAT model reads the forced set only under `optimize`, and the greedy must agree
    in both pipelines. (Whether the greedy-only build applies `alu` itself is a separate
    open entry in BUGS.md: today it does not, so nothing is shifted in either mode.)
    """
    forced = _two_vector_ops_the_greedy_runs_together(tmp_path)
    sched, _ = _forced_shift_build(tmp_path, forced, verbose=False, use_cpsat=False, valu_shift=valu_shift)
    assert [len(_lowered_lanes(sched, op_id)) for op_id in forced] == [0, 0]
    assert sched["valu_shifted"] == 0
