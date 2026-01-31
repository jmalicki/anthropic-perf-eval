"""
PyO3 interface tests for scheduler_rs only.

These tests exercise the Python bindings and their options (build_tick_tock_kernel,
build_tick_tock_kernel_ssa, enable_valu_shift, use_cpsat, etc.). They do NOT test
underlying scheduler_rs functionality—that belongs in Rust tests. Each test: build
via the binding → run on Machine → assert output matches reference_kernel2.
Small configs only so runs stay fast.
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


def test_build_tick_tock_kernel_greedy_no_valu_shift():
    """PyO3: build_tick_tock_kernel with enable_valu_shift=False produces correct kernel."""
    import scheduler_rs

    forest = Tree.generate(FOREST_HEIGHT)
    inp = Input.generate(forest, BATCH_SIZE, ROUNDS)
    mem = build_mem_image(forest, inp)
    instrs = scheduler_rs.build_tick_tock_kernel(
        FOREST_HEIGHT, N_NODES, BATCH_SIZE, ROUNDS,
        enable_valu_shift=False,
        verbose=False,
    )
    _run_kernel_and_assert_matches_reference(instrs, mem, BATCH_SIZE)


def test_build_tick_tock_kernel_greedy_with_valu_shift():
    """PyO3: build_tick_tock_kernel with enable_valu_shift=True produces correct kernel."""
    import scheduler_rs

    forest = Tree.generate(FOREST_HEIGHT)
    inp = Input.generate(forest, BATCH_SIZE, ROUNDS)
    mem = build_mem_image(forest, inp)
    instrs = scheduler_rs.build_tick_tock_kernel(
        FOREST_HEIGHT, N_NODES, BATCH_SIZE, ROUNDS,
        enable_valu_shift=True,
        verbose=False,
    )
    _run_kernel_and_assert_matches_reference(instrs, mem, BATCH_SIZE)


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
