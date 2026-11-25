"""Pin forward/backward justification on the best greedy kernels (docs/CRITICAL_PATH.md,
"Justification on the best kernels"). Measured by removal: `--greedy-justify` 0 against 10,
every other flag fixed.

- `path_flow90.json` with path bits, the shaped load-tail greedy, chain-first + native-first
  and R16 (`--reassociate-gather-xor`): 1,031 -> 1,016.
- The same with R16 + R17 final + R18 first-pass: 1,044 -> 1,012.
- `flow90.json` (index) with R16 + R17 final + R18 first-pass: 1,047 -> 1,033.

Greedy builds are deterministic, so a change in these numbers means the scheduler changed.
16-round builds: CI only.
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
              greedy_stagger_cycles=9, merge_hash_stages=True, index_form="mirrored")
PATH = dict(plan="path_flow90.json", select_bit_source="path", greedy_flow_priority="chain-first",
            greedy_alu_priority="native-first", reassociate_gather_xor=True)
REWRITES = dict(reassociate_gather_xor=True, index_early_terms="final", select_xor_pushdown="first-pass")
PATH_ALL = dict(PATH, **REWRITES)
INDEX_ALL = dict(plan="flow90.json", **REWRITES)


@pytest.mark.ci_only
@pytest.mark.parametrize("name,options,justify,cycles", [
    ("path R16", PATH, 0, 1031),
    ("path R16", PATH, 10, 1016),
    ("path R16+R17+R18", PATH_ALL, 0, 1044),
    ("path R16+R17+R18", PATH_ALL, 10, 1012),
    ("index R16+R17+R18", INDEX_ALL, 0, 1047),
    ("index R16+R17+R18", INDEX_ALL, 10, 1033),
])
def test_justify_on_best_cycles(name, options, justify, cycles):
    options = dict(options)
    rows = expand_plan(load_plan(os.path.join(ROOT, "docs/emission_plans", options.pop("plan"))), 16, 32, 10)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "sched.json")
        instrs, stats = build_kernel_with_args(emission_plan=rows, greedy_justify=justify, verbose=False,
                                               output_json=path, **SHAPED, **options)
        with open(path) as f:
            sched = json.load(f)
    instrs = json.loads(json.dumps(list(instrs)))
    assert stats["cycles"] == cycles, name
    assert not stats["budget_exceeded"]
    assert check({"instrs": instrs}, sched).ok
    assert validate_kernel(instrs, 10, 256, 16, 123)
