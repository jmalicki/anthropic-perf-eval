"""Tests for the CLI's derived build arguments and cycle reporting."""

import json
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scheduler_rs  # noqa: E402
from generate_ssa_kernel import simulator_cycles  # noqa: E402
from kernel_build_args import args_to_build_kwargs, n_nodes_for_height  # noqa: E402
from problem import DebugInfo, Input, Machine, N_CORES, Tree, build_mem_image  # noqa: E402


@pytest.mark.parametrize("height", [1, 3, 10])
def test_n_nodes_matches_the_tree_the_harness_generates(height):
    """The CLI used to derive 2**height - 1 (1023 for height 10); the tree has 2**(height+1) - 1."""
    assert n_nodes_for_height(height) == len(Tree.generate(height).values)


def test_cli_kwargs_default_n_nodes_is_2047():
    kwargs = args_to_build_kwargs(None, use_greedy=True, use_git_sha=False)
    assert kwargs["forest_height"] == 10
    assert kwargs["n_nodes"] == 2047
    assert json.loads(kwargs["run_options_json"])["n_nodes"] == 2047
    assert args_to_build_kwargs(None, n_nodes=31, use_git_sha=False)["n_nodes"] == 31


def test_greedy_barriers_flag_reaches_kwargs_and_run_options():
    """--greedy-barriers defaults to 'round' and is passed to the binding and recorded."""
    import argparse

    from kernel_build_args import add_kernel_build_args

    default = args_to_build_kwargs(None, use_greedy=True, use_git_sha=False)
    assert default["greedy_barriers"] == "round"
    assert json.loads(default["run_options_json"])["greedy_barriers"] == "round"

    parser = argparse.ArgumentParser()
    add_kernel_build_args(parser)
    args = parser.parse_args(["--use-greedy", "--greedy-barriers", "none"])
    kwargs = args_to_build_kwargs(args, use_git_sha=False)
    assert kwargs["greedy_barriers"] == "none"
    assert json.loads(kwargs["run_options_json"])["greedy_barriers"] == "none"
    with pytest.raises(SystemExit):
        parser.parse_args(["--greedy-barriers", "sometimes"])


def test_greedy_priority_flags_reach_kwargs_and_run_options():
    """--greedy-priority and its two parameters default to fifo/12/16 and are passed on and recorded."""
    import argparse

    from kernel_build_args import add_kernel_build_args

    default = args_to_build_kwargs(None, use_greedy=True, use_git_sha=False)
    recorded = json.loads(default["run_options_json"])
    for where in (default, recorded):
        assert where["greedy_priority"] == "fifo"
        assert where["greedy_stagger_cycles"] == 12
        assert where["greedy_groups_in_flight"] == 16

    parser = argparse.ArgumentParser()
    add_kernel_build_args(parser)
    args = parser.parse_args(["--use-greedy", "--greedy-barriers", "none", "--greedy-priority", "wavefront",
                              "--greedy-stagger-cycles", "9", "--greedy-groups-in-flight", "0"])
    kwargs = args_to_build_kwargs(args, use_git_sha=False)
    recorded = json.loads(kwargs["run_options_json"])
    for where in (kwargs, recorded):
        assert where["greedy_priority"] == "wavefront"
        assert where["greedy_stagger_cycles"] == 9
        assert where["greedy_groups_in_flight"] == 0
    for bad in (["--greedy-priority", "lifo"], ["--greedy-stagger-cycles", "-1"],
                ["--greedy-groups-in-flight", "many"]):
        with pytest.raises(SystemExit):
            parser.parse_args(bad)


def test_greedy_justify_flag_reaches_kwargs_and_run_options():
    """--greedy-justify defaults to 0 (off) and is passed to the binding and recorded."""
    import argparse

    from kernel_build_args import add_kernel_build_args

    default = args_to_build_kwargs(None, use_greedy=True, use_git_sha=False)
    assert default["greedy_justify"] == 0
    assert json.loads(default["run_options_json"])["greedy_justify"] == 0

    parser = argparse.ArgumentParser()
    add_kernel_build_args(parser)
    kwargs = args_to_build_kwargs(parser.parse_args(["--use-greedy", "--greedy-justify", "10"]), use_git_sha=False)
    assert kwargs["greedy_justify"] == 10
    assert json.loads(kwargs["run_options_json"])["greedy_justify"] == 10
    with pytest.raises(SystemExit):
        parser.parse_args(["--greedy-justify", "-1"])


def test_greedy_split_shift_flag_reaches_kwargs_and_run_options():
    """--greedy-split-shift defaults to off and is passed to the binding and recorded."""
    import argparse

    from kernel_build_args import add_kernel_build_args

    default = args_to_build_kwargs(None, use_greedy=True, use_git_sha=False)
    assert default["greedy_split_shift"] is False
    assert json.loads(default["run_options_json"])["greedy_split_shift"] is False

    parser = argparse.ArgumentParser()
    add_kernel_build_args(parser)
    kwargs = args_to_build_kwargs(parser.parse_args(["--use-greedy", "--greedy-split-shift"]), use_git_sha=False)
    assert kwargs["greedy_split_shift"] is True
    assert json.loads(kwargs["run_options_json"])["greedy_split_shift"] is True

    # The CP-SAT hint for split shifts is a separate switch, off by default.
    assert default["cpsat_split_shift_hint"] is False
    assert kwargs["cpsat_split_shift_hint"] is False
    hinted = args_to_build_kwargs(
        parser.parse_args(["--greedy-split-shift", "--cpsat-split-shift-hint"]), use_git_sha=False)
    assert hinted["cpsat_split_shift_hint"] is True
    assert json.loads(hinted["run_options_json"])["cpsat_split_shift_hint"] is True


def test_cpsat_scratch_time_reaches_kwargs_and_run_options():
    """--cpsat-scratch-time defaults to phase, is passed on and recorded, and rejects other values."""
    import argparse

    from kernel_build_args import add_kernel_build_args

    default = args_to_build_kwargs(None, use_greedy=False, use_git_sha=False)
    assert default["cpsat_scratch_time"] == "phase"
    assert json.loads(default["run_options_json"])["cpsat_scratch_time"] == "phase"
    parser = argparse.ArgumentParser()
    add_kernel_build_args(parser)
    kwargs = args_to_build_kwargs(parser.parse_args(["--cpsat-scratch-time", "cycle"]), use_git_sha=False)
    assert kwargs["cpsat_scratch_time"] == "cycle"
    assert json.loads(kwargs["run_options_json"])["cpsat_scratch_time"] == "cycle"
    with pytest.raises(SystemExit):
        parser.parse_args(["--cpsat-scratch-time", "half"])


def test_pressure_threshold_flags_reach_kwargs_and_run_options():
    """The threshold overrides default to None (policy built-ins), headroom to 0; all are passed on and recorded."""
    import argparse

    from kernel_build_args import add_kernel_build_args

    names = ("pressure_soft_threshold", "pressure_critical_threshold", "pressure_inflation_threshold")
    default = args_to_build_kwargs(None, use_greedy=True, use_git_sha=False)
    recorded = json.loads(default["run_options_json"])
    for where in (default, recorded):
        assert all(where[name] is None for name in names)
        assert where["greedy_budget_headroom"] == 0

    parser = argparse.ArgumentParser()
    add_kernel_build_args(parser)
    args = parser.parse_args(["--use-greedy", "--pressure-soft-threshold", "0.9", "--pressure-critical-threshold", "1",
                              "--pressure-inflation-threshold", "0.95", "--greedy-budget-headroom", "16"])
    kwargs = args_to_build_kwargs(args, use_git_sha=False)
    recorded = json.loads(kwargs["run_options_json"])
    for where in (kwargs, recorded):
        assert [where[name] for name in names] == [0.9, 1.0, 0.95]
        assert where["greedy_budget_headroom"] == 16
    for bad in (["--pressure-inflation-threshold", "0"], ["--pressure-soft-threshold", "nan"],
                ["--greedy-budget-headroom", "-1"]):
        with pytest.raises(SystemExit):
            parser.parse_args(bad)


def test_scratch_model_reaches_kwargs_and_run_options():
    """--scratch-model defaults to words, is passed on and recorded, and rejects other values."""
    import argparse

    from kernel_build_args import add_kernel_build_args

    default = args_to_build_kwargs(None, use_greedy=True, use_git_sha=False)
    assert default["scratch_model"] == "words"
    assert json.loads(default["run_options_json"])["scratch_model"] == "words"

    parser = argparse.ArgumentParser()
    add_kernel_build_args(parser)
    kwargs = args_to_build_kwargs(parser.parse_args(["--use-greedy", "--scratch-model", "pools"]), use_git_sha=False)
    assert kwargs["scratch_model"] == "pools"
    assert json.loads(kwargs["run_options_json"])["scratch_model"] == "pools"
    with pytest.raises(SystemExit):
        parser.parse_args(["--scratch-model", "bytes"])


def _build(n_nodes, rounds=2, batch_size=16):
    instrs, stats = scheduler_rs.build_tick_tock_kernel_ssa(
        forest_height=10, n_nodes=n_nodes, batch_size=batch_size, rounds=rounds, verbose=False, use_cpsat=False)
    return json.loads(json.dumps(list(instrs))), stats


def test_n_nodes_does_not_change_the_schedule():
    """n_nodes is stored in the spec but emission keys off forest_height only."""
    a, stats_a = _build(2047)
    b, stats_b = _build(1023)
    assert stats_a["cycles"] == stats_b["cycles"]
    assert len(a) == len(b)
    assert [sorted((e, len(s)) for e, s in bundle.items()) for bundle in a] == \
           [sorted((e, len(s)) for e, s in bundle.items()) for bundle in b]


def test_simulator_cycles_matches_machine():
    """CLI 'Cycles' is the schedule length; the simulator counts bundles that contain an op."""
    instrs, stats = _build(2047, rounds=2, batch_size=256)
    counted, empty = simulator_cycles(instrs)
    random.seed(123)
    forest = Tree.generate(10)
    mem = build_mem_image(forest, Input.generate(forest, 256, 2))
    machine = Machine(mem, instrs, DebugInfo(scratch_map={}), n_cores=N_CORES)
    machine.enable_pause = False
    machine.enable_debug = False
    machine.run()
    assert machine.cycle == counted
    assert len(instrs) == stats["cycles"] + 2, "schedule cycles plus the two pause bundles"
    assert counted == stats["cycles"] + 2 - empty
    assert empty == sum(1 for bundle in instrs if not bundle)


def test_relaxed_scratch_flags_reach_kwargs_and_run_options():
    """--scratch-budget-override and --no-scratch-constraint reach the binding and are recorded;
    by default the machine's 1536 words with the constraint."""
    import argparse

    from kernel_build_args import (
        DEFAULT_SCHEDULER_JSON,
        RELAXED_SCHEDULER_JSON,
        UNBOUNDED_SCRATCH_WORDS,
        add_kernel_build_args,
    )

    parser = argparse.ArgumentParser()
    add_kernel_build_args(parser)
    cases = [
        ([], 1536, True, False),
        (["--scratch-budget-override", "3072"], 3072, True, True),
        (["--no-scratch-constraint"], UNBOUNDED_SCRATCH_WORDS, False, True),
        (["--no-scratch-constraint", "--scratch-budget-override", "4096"], 4096, False, True),
        (["--scratch-budget-override", "1024"], 1024, True, False),
    ]
    for argv, size, constraint, relaxed in cases:
        kwargs = args_to_build_kwargs(parser.parse_args(["--use-greedy", *argv]), use_git_sha=False)
        recorded = json.loads(kwargs["run_options_json"])
        assert (kwargs["scratch_size"], kwargs["scratch_constraint"]) == (size, constraint), argv
        assert (recorded["scratch_size"], recorded["scratch_constraint"], recorded["relaxed_scratch"]) == \
            (size, constraint, relaxed), argv
        assert kwargs["output_json"] == (RELAXED_SCHEDULER_JSON if relaxed else DEFAULT_SCHEDULER_JSON), argv
    with pytest.raises(ValueError, match="a normal kernel/result path"):
        args_to_build_kwargs(None, scratch_constraint=False, output_json=DEFAULT_SCHEDULER_JSON)
    with pytest.raises(SystemExit):
        parser.parse_args(["--scratch-budget-override", "0"])


def test_relaxed_scratch_build_is_marked_and_kept_off_normal_paths(tmp_path, monkeypatch, capsys):
    """A relaxed-scratch build refuses normal kernel paths, prints the warning, saves a kernel
    marked as not valid on the machine, and validates it on a simulator with more scratch."""
    import generate_ssa_kernel

    monkeypatch.chdir(tmp_path)
    base = ["generate_ssa_kernel.py", "--use-greedy", "-q", "--forest-height", "5", "--batch-size", "64",
            "--rounds", "2", "--scratch-budget-override", "3072"]
    monkeypatch.setattr(sys, "argv", base + ["--save-kernel", "kernel_to_validate.json"])
    assert generate_ssa_kernel.main() == 2
    assert not (tmp_path / "kernel_to_validate.json").exists()

    monkeypatch.setattr(sys, "argv", base)
    assert generate_ssa_kernel.main() == 0
    out = capsys.readouterr().out
    assert "RELAXED SCRATCH" in out and "NOT valid on the real machine" in out
    saved = json.loads((tmp_path / "kernel_relaxed_scratch.json").read_text())
    assert saved["relaxed_scratch"]["valid_on_machine"] is False
    assert saved["relaxed_scratch"]["scratch_size"] >= 1536
    sched = json.loads((tmp_path / "scheduler_output_relaxed_scratch.json").read_text())
    assert sched["scratch_size"] == 3072 and sched["scratch_relaxed"] is True
    assert sched["run_options"]["relaxed_scratch"] is True

    monkeypatch.setattr(sys, "argv", ["generate_ssa_kernel.py", "--validate-only", "kernel_relaxed_scratch.json"])
    assert generate_ssa_kernel.main() == 0
    assert "RELAXED-SCRATCH kernel" in capsys.readouterr().out
