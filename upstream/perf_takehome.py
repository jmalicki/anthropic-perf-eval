"""
Shim, NOT an upstream file: lets the unmodified upstream tests/submission_tests.py
run a kernel produced by this repository.

The upstream test does `from perf_takehome import KernelBuilder` from the parent of
its tests/ directory, then calls KernelBuilder().build_kernel(forest_height, n_nodes,
batch_size, rounds) and reads .instrs and .debug_info(). Everything else it uses
(Machine, memory image, reference kernel) comes from the vendored upstream
tests/frozen_problem.py, so the simulator and the oracle are upstream's.

This shim only loads a kernel saved by `generate_ssa_kernel.py --save-kernel`; it
does not import scheduler_rs or this repository's edited problem.py. Point it at the
kernel with UPSTREAM_KERNEL_JSON (scripts/check_upstream.sh does this).
"""

import json
import os

from problem import DebugInfo  # the vendored upstream problem.py next to this file

KERNEL_JSON_ENV = "UPSTREAM_KERNEL_JSON"


class KernelBuilder:
    def __init__(self):
        self.instrs = []

    def build_kernel(self, forest_height: int, n_nodes: int, batch_size: int, rounds: int):
        path = os.environ.get(KERNEL_JSON_ENV)
        if not path:
            raise RuntimeError(
                f"{KERNEL_JSON_ENV} is not set; run scripts/check_upstream.sh, or point it "
                "at a kernel saved by generate_ssa_kernel.py --save-kernel"
            )
        with open(path) as f:
            data = json.load(f)
        requested = {"forest_height": forest_height, "batch_size": batch_size, "rounds": rounds}
        saved = {k: data.get(k) for k in requested}
        if saved != requested:
            raise RuntimeError(f"{path} was built for {saved}, but the test asked for {requested}")
        self.instrs = [
            {engine: [tuple(op) for op in ops] for engine, ops in bundle.items()}
            for bundle in data["instrs"]
        ]

    def debug_info(self):
        return DebugInfo(scratch_map={})
