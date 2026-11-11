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
