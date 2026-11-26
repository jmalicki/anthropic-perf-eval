"""Greedy-only builds: forced encodings take effect, and the ConstChoice policies emit kernels
that are proved against their SSA program and validated.

Before this, `--const-choice`, `--select-choice` and `--valu-shift` were read only by the
CP-SAT pipeline, so under `--use-greedy` a forced choice produced the same kernel as no choice.
"""
import collections
import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate_ssa_kernel import validate_kernel  # noqa: E402
from kernel_build_args import build_kernel_with_args  # noqa: E402
from verification.kernel_dataflow import check  # noqa: E402

ROUNDS, BATCH, HEIGHT = 2, 16, 10


def _build(**options):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "sched.json")
        instrs, stats = build_kernel_with_args(rounds=ROUNDS, batch_size=BATCH, use_greedy=True, verbose=False,
                                               output_json=path, **options)
        with open(path) as f:
            sched = json.load(f)
    instrs = json.loads(json.dumps(list(instrs)))
    assert check({"instrs": instrs}, sched).ok, "dataflow replay: kernel differs from its SSA program"
    assert validate_kernel(instrs, HEIGHT, BATCH, ROUNDS, 123)
    encodings = collections.Counter(op["engine"] for op in sched["ops"] if op.get("from_const_choice") is not None)
    n_const = sum(1 for op in sched["ssa_ops"] if op["opcode"] == "const_choice")
    sched["_stats"] = dict(stats)
    return encodings, n_const, sched


@pytest.mark.parametrize("mode,engine", [("add_imm", "flow"), ("load", "load")])
def test_forced_const_choice_takes_effect_under_greedy(mode, engine):
    encodings, n_const, _ = _build(const_choice=mode)
    assert n_const > 0
    assert encodings == {engine: n_const}, encodings


def test_forced_select_choice_takes_effect_under_greedy():
    _, _, default = _build(select_choice="optimize")
    _, _, forced = _build(select_choice="fma")
    flow = lambda s: sum(1 for op in s["ops"] if op["engine"] == "flow" and op["slot"][0] == "vselect")  # noqa: E731
    assert flow(default) > 0, "control: the default greedy runs selects as vselect"
    assert flow(forced) == 0


@pytest.mark.parametrize("policy", ["alu-first", "alu-when-load-full"])
def test_const_policy_kernels_are_proved_and_valid(policy):
    """The derived constants read their operands at the derive's cycle: the dataflow replay
    proves the kernel equals the SSA program, and the simulator validates it (in _build)."""
    encodings, n_const, _ = _build(greedy_const_policy=policy)
    assert sum(encodings.values()) == n_const


def test_justify_with_alu_derived_constants():
    """`--greedy-justify` models ALU-derived constants (their reads of producer values): with
    `--greedy-const-policy alu-first` it runs, keeps every derivation, and the kernel is still
    proved against its SSA program and validated (in _build)."""
    shape = dict(greedy_barriers="none", greedy_priority="wavefront", greedy_split_shift=True,
                 pressure_inflation_threshold=0.95, greedy_const_policy="alu-first")
    _, _, plain = _build(**shape)
    _, _, justified = _build(greedy_justify=10, **shape)
    derived = lambda s: sorted(op["from_const_choice"] for op in s["ops"]  # noqa: E731
                               if op.get("from_const_choice") is not None and op["engine"] == "alu")
    assert derived(plain), "control: alu-first derives a constant on the scalar ALU"
    assert derived(justified) == derived(plain)
    # The pass ran on this input (before, it skipped any schedule with a derived constant).
    summary = justified["_stats"]["greedy_justify_summary"]
    assert plain["_stats"]["greedy_justify_summary"] in (None, "off")
    assert summary and summary != "off" and not summary.startswith("skipped"), summary
    assert "fell back" not in summary and "failed verification" not in summary, summary
    assert justified["_stats"]["cycles"] < plain["_stats"]["cycles"], (summary, plain["_stats"]["cycles"])
    cycles = lambda s: [op["cycle"] for op in s["ssa_ops"]]  # noqa: E731
    assert cycles(justified) != cycles(plain)


@pytest.mark.parametrize("policy", ["alu-first", "alu-when-load-full"])
@pytest.mark.parametrize("owner", ["enable_schedule_with_fixed_choices", "enable_select_choice_hint_pass",
                                   "enable_valu_shift_hint_pass"])
def test_const_policy_with_a_fixed_choices_rerun(policy, owner):
    """Review of #86: a hint pass or --schedule-with-fixed-choices reruns the greedy with the
    first run's encodings fixed. A derived constant must then be scheduled as an ALU op after
    its producers (or recorded as the load the rerun made it): before, the rerun placed it as a
    load while lowering still emitted the ALU derivation, at a cycle nothing ordered after its
    producers and outside the slot count (13 scalar ops in a cycle at 4 rounds, batch 32)."""
    rounds, batch = 4, 32
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "sched.json")
        instrs, stats = build_kernel_with_args(rounds=rounds, batch_size=batch, use_greedy=True, verbose=False,
                                               output_json=path, greedy_const_policy=policy, **{owner: True})
        with open(path) as f:
            sched = json.load(f)
    instrs = json.loads(json.dumps(list(instrs)))
    assert any(op.get("from_const_choice") is not None and op["engine"] == "alu" for op in sched["ops"]), \
        "control: derivations are kept (honoured by the rerun), not all turned into loads"
    limits = {"alu": 12, "valu": 6, "load": 2, "store": 2, "flow": 1}
    for c, bundle in enumerate(instrs):
        for engine, slots in bundle.items():
            assert len(slots) <= limits.get(engine, 99), f"cycle {c}: {len(slots)} {engine} slots"
    assert check({"instrs": instrs}, sched).ok, "dataflow replay: kernel differs from its SSA program"
    assert validate_kernel(instrs, HEIGHT, batch, rounds, 123)
