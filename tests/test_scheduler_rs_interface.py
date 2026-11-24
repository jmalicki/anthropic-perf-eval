"""
PyO3 interface tests for scheduler_rs only.

These tests exercise the Python bindings and their options (build_tick_tock_kernel_ssa,
enable_valu_shift, use_cpsat, etc.). They do NOT test underlying scheduler_rs
functionality—that belongs in Rust tests. Each test: build via the binding → run
on Machine → assert output matches reference_kernel2. Small configs only so runs stay fast.
"""

import json

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


def _run_kernel_and_assert_matches_reference(instrs, mem, batch_size, scratch_size=None):
    """Run kernel on Machine and assert output matches reference_kernel2.

    scratch_size: scratch words of the simulated machine (default: the real machine's); only
    relaxed-scratch kernels pass more.
    """
    extra = {} if scratch_size is None else {"scratch_size": scratch_size}
    machine = Machine(mem, list(instrs), DebugInfo(scratch_map={}), n_cores=N_CORES, **extra)
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


def test_greedy_barriers_none():
    """PyO3: greedy_barriers='none' is accepted, produces correct output, and bad values are rejected."""
    import scheduler_rs

    forest = Tree.generate(FOREST_HEIGHT)
    inp = Input.generate(forest, BATCH_SIZE, ROUNDS)
    mem = build_mem_image(forest, inp)
    instrs, stats = scheduler_rs.build_tick_tock_kernel_ssa(
        FOREST_HEIGHT, N_NODES, BATCH_SIZE, ROUNDS,
        verbose=False,
        use_cpsat=False,
        greedy_barriers="none",
    )
    assert not stats.get("budget_exceeded", False)
    _run_kernel_and_assert_matches_reference(instrs, mem, BATCH_SIZE)
    with pytest.raises(ValueError, match="greedy_barriers"):
        scheduler_rs.build_tick_tock_kernel_ssa(
            FOREST_HEIGHT, N_NODES, BATCH_SIZE, ROUNDS,
            verbose=False,
            use_cpsat=False,
            greedy_barriers="sometimes",
        )


def test_greedy_priority_wavefront():
    """PyO3: greedy_priority='wavefront' with its parameters is accepted and produces correct output."""
    import scheduler_rs

    for extra in ({}, {"greedy_stagger_cycles": 0, "greedy_groups_in_flight": 0},
                  {"greedy_stagger_cycles": 30, "greedy_groups_in_flight": 2}):
        # A fresh image per build: the reference kernel updates the image it is given.
        forest = Tree.generate(FOREST_HEIGHT)
        inp = Input.generate(forest, BATCH_SIZE, ROUNDS)
        mem = build_mem_image(forest, inp)
        instrs, stats = scheduler_rs.build_tick_tock_kernel_ssa(
            FOREST_HEIGHT, N_NODES, BATCH_SIZE, ROUNDS,
            verbose=False,
            use_cpsat=False,
            greedy_barriers="none",
            greedy_priority="wavefront",
            **extra,
        )
        assert not stats.get("budget_exceeded", False), extra
        _run_kernel_and_assert_matches_reference(instrs, mem, BATCH_SIZE)
    with pytest.raises(ValueError, match="greedy_priority"):
        scheduler_rs.build_tick_tock_kernel_ssa(
            FOREST_HEIGHT, N_NODES, BATCH_SIZE, ROUNDS,
            verbose=False,
            use_cpsat=False,
            greedy_priority="lifo",
        )


def _alu_slots_per_cycle(instrs):
    return [len(bundle.get("alu", [])) for bundle in instrs]


def test_greedy_split_shift():
    """PyO3: greedy_split_shift=True is accepted, fills scalar slots past 8, and the kernel is correct."""
    import scheduler_rs

    builds = {}
    for split in (False, True):
        forest = Tree.generate(FOREST_HEIGHT)
        inp = Input.generate(forest, BATCH_SIZE, ROUNDS)
        mem = build_mem_image(forest, inp)
        instrs, stats = scheduler_rs.build_tick_tock_kernel_ssa(
            FOREST_HEIGHT, N_NODES, BATCH_SIZE, ROUNDS,
            verbose=False,
            use_cpsat=False,
            greedy_barriers="none",
            greedy_priority="wavefront",
            greedy_split_shift=split,
        )
        assert not stats.get("budget_exceeded", False)
        _run_kernel_and_assert_matches_reference(instrs, mem, BATCH_SIZE)
        builds[split] = (list(instrs), stats)
    assert max(_alu_slots_per_cycle(builds[True][0])) <= 12
    assert sum(_alu_slots_per_cycle(builds[True][0])) > sum(_alu_slots_per_cycle(builds[False][0])), \
        "split shifts move more vector work onto the scalar slots"
    assert builds[True][1]["cycles"] <= builds[False][1]["cycles"]


def test_cpsat_split_shift_hint():
    """PyO3: with cpsat_split_shift_hint the split-shift greedy schedule is accepted by CP-SAT and the kernel is correct."""
    import scheduler_rs

    forest = Tree.generate(FOREST_HEIGHT)
    inp = Input.generate(forest, BATCH_SIZE, ROUNDS)
    mem = build_mem_image(forest, inp)
    instrs, stats = scheduler_rs.build_tick_tock_kernel_ssa(
        FOREST_HEIGHT, N_NODES, BATCH_SIZE, ROUNDS,
        verbose=False,
        use_cpsat=True,
        cpsat_time_limit=120.0,
        cpsat_deterministic_time=1.0,
        greedy_barriers="none",
        greedy_priority="wavefront",
        greedy_split_shift=True,
        cpsat_split_shift_hint=True,
    )
    assert not stats.get("budget_exceeded", False)
    assert max(_alu_slots_per_cycle(list(instrs))) <= 12
    _run_kernel_and_assert_matches_reference(instrs, mem, BATCH_SIZE)


def test_pressure_thresholds_and_headroom():
    """PyO3: threshold overrides and budget headroom are accepted, give a correct kernel, and bad values raise."""
    import scheduler_rs

    forest = Tree.generate(FOREST_HEIGHT)
    inp = Input.generate(forest, BATCH_SIZE, ROUNDS)
    mem = build_mem_image(forest, inp)
    instrs, stats = scheduler_rs.build_tick_tock_kernel_ssa(
        FOREST_HEIGHT, N_NODES, BATCH_SIZE, ROUNDS,
        verbose=False,
        use_cpsat=False,
        pressure_soft_threshold=0.9,
        pressure_critical_threshold=1.0,
        pressure_inflation_threshold=0.95,
        greedy_budget_headroom=16,
    )
    assert not stats.get("budget_exceeded", False)
    _run_kernel_and_assert_matches_reference(instrs, mem, BATCH_SIZE)
    for bad in ({"pressure_inflation_threshold": 0.0}, {"pressure_soft_threshold": float("nan")},
                {"pressure_critical_threshold": -1.0}):
        with pytest.raises(ValueError, match="must be a positive finite number"):
            scheduler_rs.build_tick_tock_kernel_ssa(
                FOREST_HEIGHT, N_NODES, BATCH_SIZE, ROUNDS,
                verbose=False,
                use_cpsat=False,
                **bad,
            )


def test_depth3_validates():
    """Depth 3 tree-reduce: kernel with tree_reduce_max_depth=3 must match reference.

    Uses forest_height=5 (tree_height=6), rounds=4, batch_size=8 so round 3 hits depth 3
    (8-way selection). Greedy scheduler (fast) — correctness is the goal.
    """
    import scheduler_rs

    forest_height = 5
    n_nodes = 31
    batch_size = 8
    rounds = 4
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
        tree_reduce_max_depth=3,
    )
    assert not stats.get("budget_exceeded", False), "depth3 small config should stay within budget"
    _run_kernel_and_assert_matches_reference(instrs, mem, batch_size)


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


def test_scratch_size_and_scratch_constraint(tmp_path):
    """PyO3: scratch_size / scratch_constraint reach the allocator, are reported in stats and
    give correct kernels (relaxed ones on a simulator with that much scratch)."""
    import scheduler_rs

    def build(**extra):
        forest = Tree.generate(FOREST_HEIGHT)
        inp = Input.generate(forest, 16, 2)
        mem = build_mem_image(forest, inp)
        instrs, stats = scheduler_rs.build_tick_tock_kernel_ssa(
            FOREST_HEIGHT, N_NODES, 16, 2, verbose=False, **extra)
        assert not stats["budget_exceeded"], extra
        assert stats["max_addr"] < stats["scratch_size"], extra
        _run_kernel_and_assert_matches_reference(
            instrs, mem, 16, scratch_size=max(1536, stats["max_addr"] + 1))
        return stats

    default = build(use_cpsat=False)
    assert (default["scratch_size"], default["scratch_constraint"], default["relaxed_scratch"]) == (1536, True, False)

    smaller = build(use_cpsat=False, scratch_size=1024)
    assert (smaller["scratch_size"], smaller["relaxed_scratch"]) == (1024, False)

    relaxed_path = tmp_path / "relaxed.json"
    relaxed_greedy = build(use_cpsat=False, scratch_size=1024, scratch_constraint=False,
                           output_json=str(relaxed_path))
    assert relaxed_greedy["relaxed_scratch"] is True
    assert json.loads(relaxed_path.read_text())["scratch_relaxed"] is True

    larger = build(use_cpsat=False, scratch_size=3072)
    assert (larger["scratch_size"], larger["relaxed_scratch"]) == (3072, True)

    # CP-SAT side (the Rust test covers the constrained budgets): no scratch section at all.
    unbounded = build(use_cpsat=True, cpsat_time_limit=120.0, cpsat_deterministic_time=0.2,
                      scratch_constraint=False)
    assert unbounded["scratch_size"] == 1 << 20
    assert unbounded["relaxed_scratch"]
    assert unbounded["cpsat_scratch_budget"] == 0, "no scratch section in the CP-SAT model"

    with pytest.raises(ValueError, match="scratch_size"):
        scheduler_rs.build_tick_tock_kernel_ssa(
            FOREST_HEIGHT, N_NODES, BATCH_SIZE, 2, verbose=False, use_cpsat=False, scratch_size=4)


def test_direct_binding_protects_normal_outputs_from_relaxed_scratch(tmp_path):
    import scheduler_rs
    from kernel_build_args import NORMAL_KERNEL_PATHS

    for name in sorted(NORMAL_KERNEL_PATHS):
        normal_path = tmp_path / name
        normal_path.write_text("existing artifact")
        with pytest.raises(ValueError, match="separate from normal kernel artifacts"):
            scheduler_rs.build_tick_tock_kernel_ssa(
                FOREST_HEIGHT, N_NODES, 16, 2, verbose=False,
                output_json=str(normal_path), scratch_size=3072)
        assert normal_path.read_text() == "existing artifact"

    normal_path = tmp_path / "scheduler_output.json"
    scheduler_rs.build_tick_tock_kernel_ssa(
        FOREST_HEIGHT, N_NODES, 16, 2, verbose=False, output_json=str(normal_path))
    assert json.loads(normal_path.read_text())["scratch_relaxed"] is False
