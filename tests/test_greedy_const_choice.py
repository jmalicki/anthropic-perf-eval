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


def _build(rounds=ROUNDS, batch=BATCH, **options):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "sched.json")
        instrs, stats = build_kernel_with_args(rounds=rounds, batch_size=batch, use_greedy=True, verbose=False,
                                               output_json=path, **options)
        with open(path) as f:
            sched = json.load(f)
    instrs = json.loads(json.dumps(list(instrs)))
    assert check({"instrs": instrs}, sched).ok, "dataflow replay: kernel differs from its SSA program"
    assert validate_kernel(instrs, HEIGHT, batch, rounds, 123)
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
    proves the kernel equals the SSA program, and the simulator validates it (in _build).
    4 rounds, batch 32: both policies derive constants there (at 2 rounds, batch 16 the
    default shape derives none, so the test proved nothing about derivations)."""
    encodings, n_const, _ = _build(rounds=4, batch=32, greedy_const_policy=policy)
    assert sum(encodings.values()) == n_const
    assert encodings["alu"] > 0, f"control: {policy} derived nothing ({dict(encodings)})"


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
    # Control: without the policy, justification changes this schedule too.
    default = dict(shape, greedy_const_policy="default")
    assert cycles(_build(greedy_justify=10, **default)[2]) != cycles(_build(**default)[2])


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


def test_forced_valu_shift_takes_effect_under_greedy():
    """--valu-shift alu shifts every shiftable vector op, --valu-shift valu none (review of #86).
    4 rounds, batch 32: the smallest size here where the default greedy shifts some (8)."""
    def shifted(mode):
        with tempfile.TemporaryDirectory() as tmp:
            instrs, stats = build_kernel_with_args(rounds=4, batch_size=32, use_greedy=True, verbose=False,
                                                   valu_shift=mode, output_json=os.path.join(tmp, "s.json"))
        instrs = json.loads(json.dumps(list(instrs)))
        assert validate_kernel(instrs, HEIGHT, 32, 4, 123)
        return stats["valu_shifted"], instrs
    default, _ = shifted("optimize")
    alu, alu_instrs = shifted("alu")
    valu, _ = shifted("valu")
    assert 0 < default < alu, "control: the default shifts some, alu more"
    assert valu == 0
    shiftable = {"+", "-", "*", "^", "&", "|", "<<", ">>", "%", "<", "==", "//", "cdiv"}
    on_valu = [s for b in alu_instrs for s in b.get("valu", []) if s[0] in shiftable]
    assert not on_valu, f"--valu-shift alu left {len(on_valu)} shiftable ops on the vector ALU"


def test_forced_vselect_takes_effect_under_greedy():
    _, _, forced = _build(select_choice="vselect")
    n_select = sum(1 for op in forced["ssa_ops"] if op["opcode"] == "select_choice")
    vselects = sum(1 for op in forced["ops"] if op["engine"] == "flow" and op["slot"][0] == "vselect")
    assert n_select > 0 and vselects == n_select


@pytest.mark.parametrize("options, match", [
    (dict(const_choice="load", enable_const_choice_hint_pass=True), "hint pass"),
    (dict(const_choice="add_imm", hint_passes=True), "hint pass"),
    (dict(select_choice="vselect", enable_select_choice_hint_pass=True), "hint pass"),
    (dict(valu_shift="alu", enable_valu_shift_hint_pass=True), "hint pass"),
    (dict(greedy_early_stagger=(8, 3)), "wavefront"),
    (dict(greedy_const_policy="alu-first", const_choice="load"), "owns the ConstChoice"),
    (dict(greedy_const_policy="alu-first", enable_const_choice_hint_pass=True), "owns the ConstChoice"),
])
def test_overridden_or_ignored_options_are_errors(options, match):
    """Review of #86: options another option would silently override or ignore."""
    with pytest.raises(ValueError, match=match):
        _build(**options)


def test_const_policy_under_cpsat_is_an_error():
    with pytest.raises(ValueError, match="greedy-only|owns the ConstChoice"):
        build_kernel_with_args(rounds=ROUNDS, batch_size=BATCH, use_greedy=False, verbose=False,
                               greedy_const_policy="alu-first")


def test_binding_rejects_overridden_options():
    """Direct binding calls get the same check (Rust config::check_encoding_option_combinations)."""
    import scheduler_rs

    base = dict(forest_height=2, n_nodes=7, batch_size=8, rounds=1, verbose=False, use_cpsat=False)
    with pytest.raises(ValueError, match="const_choice"):
        scheduler_rs.build_tick_tock_kernel_ssa(**base, const_choice="load", enable_const_choice_hint_pass=True)
    with pytest.raises(ValueError, match="greedy_early_stagger"):
        scheduler_rs.build_tick_tock_kernel_ssa(**base, greedy_early_stagger=(8, 3))
    with pytest.raises(ValueError, match="greedy_const_policy"):
        scheduler_rs.build_tick_tock_kernel_ssa(**dict(base, use_cpsat=True), greedy_const_policy="alu-first")


def test_valu_shift_alu_warning_only_when_shifting_is_on(capsys):
    """The slow-mode warning for --valu-shift alu under the greedy is not printed with
    --no-valu-shift (enable_valu_shift=False), where the mode shifts nothing."""
    from kernel_build_args import args_to_build_kwargs

    args_to_build_kwargs(None, use_greedy=True, valu_shift="alu", use_git_sha=False)
    assert "--valu-shift alu" in capsys.readouterr().err
    args_to_build_kwargs(None, use_greedy=True, valu_shift="alu", enable_valu_shift=False, use_git_sha=False)
    assert "--valu-shift alu" not in capsys.readouterr().err
