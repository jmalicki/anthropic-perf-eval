"""
PyO3 interface tests for scheduler_rs only.

These tests exercise the Python bindings and their options (build_tick_tock_kernel_ssa,
enable_valu_shift, use_cpsat, etc.). They do NOT test underlying scheduler_rs
functionality—that belongs in Rust tests. Each test: build via the binding → run
on Machine → assert output matches reference_kernel2. Small configs only so runs stay fast.
"""

import pytest

from problem import (
    Machine,
    Tree,
    Input,
    build_mem_image,
    reference_kernel2,
    N_CORES,
    DebugInfo,
)

# Small config for all interface tests (fast, same across tests)
FOREST_HEIGHT = 5
N_NODES = 63
BATCH_SIZE = 64
ROUNDS = 4


def _run_kernel_and_assert_matches_reference(instrs, mem, batch_size):
    """Run kernel on Machine and assert output matches reference_kernel2."""
    machine = Machine(mem, list(instrs), DebugInfo(scratch_map={}), n_cores=N_CORES)
    machine.enable_pause = False
    machine.enable_debug = False
    machine.run()
    ref_mem = list(reference_kernel2(mem))[-1]
    inp_values_p = ref_mem[6]
    inp_indices_p = ref_mem[5]
    assert machine.mem[inp_values_p : inp_values_p + batch_size] == ref_mem[
        inp_values_p : inp_values_p + batch_size
    ], "Output values must match reference"
    assert machine.mem[inp_indices_p : inp_indices_p + batch_size] == ref_mem[
        inp_indices_p : inp_indices_p + batch_size
    ], "Output indices must match reference"


def test_build_tick_tock_kernel_ssa_greedy():
    """PyO3: build_tick_tock_kernel_ssa with use_cpsat=False returns (instrs, stats); instrs correct."""
    import scheduler_rs

    forest = Tree.generate(FOREST_HEIGHT)
    inp = Input.generate(forest, BATCH_SIZE, ROUNDS)
    mem = build_mem_image(forest, inp)
    instrs, stats = scheduler_rs.build_tick_tock_kernel_ssa(
        FOREST_HEIGHT, N_NODES, BATCH_SIZE, ROUNDS,
        enable_valu_shift=False,
        verbose=False,
        use_cpsat=False,
    )
    _run_kernel_and_assert_matches_reference(instrs, mem, BATCH_SIZE)


# Production-sized 4-round config: same as submission (forest_height=10, batch=256) but rounds=4.
# Used for 4-round optimization effort submission-style correctness.
FOUR_ROUND_FOREST_HEIGHT = 10
FOUR_ROUND_N_NODES = 2047
FOUR_ROUND_BATCH_SIZE = 256
FOUR_ROUNDS = 4


def test_4_round_submission_correctness():
    """4-round kernel (Rust-generated, production-sized) must match reference_kernel2.

    Submission-style correctness: forest_height=10, n_nodes=2047, batch_size=256, rounds=4.
    Build via scheduler_rs, run on Machine, assert output == reference_kernel2.
    Uses default Adaptive pressure policy (graduated sort, no cascading deferral).
    """
    import scheduler_rs

    forest = Tree.generate(FOUR_ROUND_FOREST_HEIGHT)
    inp = Input.generate(forest, FOUR_ROUND_BATCH_SIZE, FOUR_ROUNDS)
    mem = build_mem_image(forest, inp)
    instrs, stats = scheduler_rs.build_tick_tock_kernel_ssa(
        FOUR_ROUND_FOREST_HEIGHT,
        FOUR_ROUND_N_NODES,
        FOUR_ROUND_BATCH_SIZE,
        FOUR_ROUNDS,
        enable_valu_shift=False,
        verbose=False,
        use_cpsat=False,
    )
    _run_kernel_and_assert_matches_reference(instrs, mem, FOUR_ROUND_BATCH_SIZE)


# Edge cases: small batches and few rounds (n_groups=1, partial iters)
def test_ssa_1_round_batch_8_correctness():
    """1 round, batch 8 (n_groups=1): kernel must build and match reference.

    Exercises n_iters ceiling (1 iter for 1 group), single-group load/store,
    and g < n_groups guards in xor/hash/depth phases.
    """
    import scheduler_rs

    forest_height = 10
    n_nodes = 2047
    batch_size = 8
    rounds = 1
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)
    instrs, stats = scheduler_rs.build_tick_tock_kernel_ssa(
        forest_height,
        n_nodes,
        batch_size,
        rounds,
        enable_valu_shift=False,
        verbose=False,
        use_cpsat=False,
    )
    assert not stats.get("budget_exceeded", False), "1r8b should stay within budget"
    _run_kernel_and_assert_matches_reference(instrs, mem, batch_size)


def test_ssa_2_rounds_batch_16_correctness():
    """2 rounds, batch 16 (n_groups=2): kernel must build and match reference.

    Minimal small config; exercises ceiling n_iters and partial-iter guards.
    """
    import scheduler_rs

    forest_height = 10
    n_nodes = 2047
    batch_size = 16
    rounds = 2
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)
    instrs, stats = scheduler_rs.build_tick_tock_kernel_ssa(
        forest_height,
        n_nodes,
        batch_size,
        rounds,
        enable_valu_shift=False,
        verbose=False,
        use_cpsat=False,
    )
    assert not stats.get("budget_exceeded", False), "2r16b should stay within budget"
    _run_kernel_and_assert_matches_reference(instrs, mem, batch_size)


def test_scheduling_pressure_policy_throttled():
    """PyO3: scheduling_pressure_policy='throttled' kwarg is accepted and produces correct output."""
    import scheduler_rs

    forest = Tree.generate(FOREST_HEIGHT)
    inp = Input.generate(forest, BATCH_SIZE, ROUNDS)
    mem = build_mem_image(forest, inp)
    instrs, stats = scheduler_rs.build_tick_tock_kernel_ssa(
        FOREST_HEIGHT, N_NODES, BATCH_SIZE, ROUNDS,
        enable_valu_shift=False,
        verbose=False,
        use_cpsat=False,
        scheduling_pressure_policy="throttled",
    )
    _run_kernel_and_assert_matches_reference(instrs, mem, BATCH_SIZE)


def test_scheduling_pressure_policy_gated():
    """PyO3: scheduling_pressure_policy='gated' kwarg is accepted and produces correct output."""
    import scheduler_rs

    forest = Tree.generate(FOREST_HEIGHT)
    inp = Input.generate(forest, BATCH_SIZE, ROUNDS)
    mem = build_mem_image(forest, inp)
    instrs, stats = scheduler_rs.build_tick_tock_kernel_ssa(
        FOREST_HEIGHT, N_NODES, BATCH_SIZE, ROUNDS,
        enable_valu_shift=False,
        verbose=False,
        use_cpsat=False,
        scheduling_pressure_policy="gated",
    )
    _run_kernel_and_assert_matches_reference(instrs, mem, BATCH_SIZE)


def test_scheduling_pressure_policy_greedy():
    """PyO3: scheduling_pressure_policy='greedy' kwarg is accepted and produces correct output."""
    import scheduler_rs

    forest = Tree.generate(FOREST_HEIGHT)
    inp = Input.generate(forest, BATCH_SIZE, ROUNDS)
    mem = build_mem_image(forest, inp)
    instrs, stats = scheduler_rs.build_tick_tock_kernel_ssa(
        FOREST_HEIGHT, N_NODES, BATCH_SIZE, ROUNDS,
        enable_valu_shift=False,
        verbose=False,
        use_cpsat=False,
        scheduling_pressure_policy="greedy",
    )
    _run_kernel_and_assert_matches_reference(instrs, mem, BATCH_SIZE)


def test_depth4_validates():
    """Depth 4 tree-reduce: kernel with tree_reduce_max_depth=4 must match reference.

    Uses forest_height=5 (tree_height=6), rounds=5, batch_size=8 so round 4 hits depth 4.
    Greedy scheduler (fast) — correctness is the goal, not optimal cycles.
    """
    import scheduler_rs

    forest_height = 5
    n_nodes = 31
    batch_size = 8
    rounds = 5
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)
    instrs, stats = scheduler_rs.build_tick_tock_kernel_ssa(
        forest_height,
        n_nodes,
        batch_size,
        rounds,
        enable_valu_shift=False,
        verbose=False,
        use_cpsat=False,
        tree_reduce_max_depth=4,
    )
    assert not stats.get("budget_exceeded", False), "depth4 small config should stay within budget"
    _run_kernel_and_assert_matches_reference(instrs, mem, batch_size)
