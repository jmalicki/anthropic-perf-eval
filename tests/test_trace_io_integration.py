"""Integration test for trace_input/trace_output with SSA kernel builder."""

import pytest
from problem import Machine, Tree, Input, build_mem_image, DebugInfo, N_CORES
from generate_ssa_kernel import load_kernel_from_json
from kernel_build_args import build_kernel_with_args
import random


def test_trace_io_integration_enabled():
    """Build 2-round kernel with --enable-trace-ids, run with enable_debug, assert trace_io_log non-empty."""
    instrs, stats = build_kernel_with_args(
        rounds=2,
        batch_size=8,
        enable_trace_ids=True,
        use_greedy=True,
        verbose=False,
    )
    assert instrs is not None
    assert len(instrs) > 0

    random.seed(123)
    forest = Tree.generate(10)
    inp = Input.generate(forest, 8, 2)
    mem = build_mem_image(forest, inp)

    machine = Machine(mem, list(instrs), DebugInfo(scratch_map={}), n_cores=N_CORES)
    machine.enable_debug = True
    machine.enable_pause = False
    machine.run()

    assert len(machine.trace_io_log) > 0, "trace_io_log should have entries when enable_trace_ids=True"
    hashed_entries = [e for e in machine.trace_io_log if "hashed_val" in str(e[2])]
    assert len(hashed_entries) >= 2, "Should have at least 2 hashed_val trace entries (r0 and r1)"


def test_trace_io_integration_disabled():
    """When enable_trace_ids=False, kernel has no trace ops; trace_io_log stays empty."""
    instrs, stats = build_kernel_with_args(
        rounds=2,
        batch_size=8,
        enable_trace_ids=False,
        use_greedy=True,
        verbose=False,
    )

    random.seed(123)
    forest = Tree.generate(10)
    inp = Input.generate(forest, 8, 2)
    mem = build_mem_image(forest, inp)

    machine = Machine(mem, list(instrs), DebugInfo(scratch_map={}), n_cores=N_CORES)
    machine.enable_debug = True
    machine.enable_pause = False
    machine.run()

    assert len(machine.trace_io_log) == 0, "trace_io_log should be empty when enable_trace_ids=False"
