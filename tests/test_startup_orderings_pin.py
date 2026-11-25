"""Pin the start-up orderings' headline (docs/STARTUP_AND_STALL_LOOP.md): on
`path_flow90.json` with path bits and the shaped load-tail greedy, `--greedy-flow-priority
chain-first --greedy-alu-priority native-first` gives 1,030 cycles against 1,033 without them,
and each alone 1,031 / 1,033 (the measured-by-removal table). Greedy builds are deterministic,
so a change in these numbers means the scheduler changed. 16-round builds: CI only.
"""
import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate_ssa_kernel import validate_kernel  # noqa: E402
from kernel_build_args import build_kernel_with_args  # noqa: E402
from solver.emission_plan import expand_plan, load_plan  # noqa: E402
from verification.kernel_dataflow import check  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHAPED = dict(use_greedy=True, greedy_barriers="none", greedy_priority="wavefront", greedy_split_shift=True,
              pressure_inflation_threshold=0.95, greedy_load_priority="final-last", greedy_groups_in_flight=32,
              greedy_stagger_cycles=9, merge_hash_stages=True, index_form="mirrored", select_bit_source="path")


@pytest.mark.ci_only
@pytest.mark.parametrize("flow,alu,cycles", [
    ("key", "key", 1033),
    ("chain-first", "key", 1031),
    ("key", "native-first", 1033),
    ("chain-first", "native-first", 1030),
])
def test_startup_orderings_cycles(flow, alu, cycles):
    rows = expand_plan(load_plan(os.path.join(ROOT, "docs/emission_plans/path_flow90.json")), 16, 32, 10)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "sched.json")
        instrs, stats = build_kernel_with_args(emission_plan=rows, greedy_flow_priority=flow, greedy_alu_priority=alu,
                                               verbose=False, output_json=path, **SHAPED)
        with open(path) as f:
            sched = json.load(f)
    instrs = json.loads(json.dumps(list(instrs)))
    assert stats["cycles"] == cycles
    assert not stats["budget_exceeded"]
    assert check({"instrs": instrs}, sched).ok
    assert validate_kernel(instrs, 10, 256, 16, 123)
