"""Test that verify_kernel_bundle_consistency passes when kernel and scheduler JSON are from the same run."""

import os
import tempfile

import pytest

from generate_ssa_kernel import save_kernel_to_json
from kernel_build_args import build_kernel_with_args
from scripts.verify_kernel_bundle_consistency import run_consistency_check


def test_bundle_consistency_2r8_same_run():
    """Build 2r/8 kernel with output_json and save_kernel from same run; verify consistency passes."""
    with tempfile.TemporaryDirectory() as tmp:
        sched_path = os.path.join(tmp, "sched.json")
        kernel_path = os.path.join(tmp, "kernel.json")
        forest_height = 10
        batch_size = 8
        rounds = 2
        seed = 123
        instrs, stats = build_kernel_with_args(
            rounds=rounds,
            batch_size=batch_size,
            use_greedy=True,
            verbose=False,
            output_json=sched_path,
        )
        assert instrs is not None and len(instrs) > 0
        save_kernel_to_json(
            instrs, kernel_path, forest_height, batch_size, rounds, seed
        )
        passed, mismatches = run_consistency_check(kernel_path, sched_path)
        assert passed, f"Bundle consistency failed: {mismatches}"
