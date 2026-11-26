"""Region tags (`SsaOp::region`) change nothing about the emitted program.

The SSA program (without the `region` field) and the greedy kernel of a small matrix of builds
are compared with fingerprints recorded on main before tags existed (commit 1fcf628). A change
that alters the emission on purpose must update the fingerprints and say why.
"""

import hashlib
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernel_build_args import build_kernel_with_args  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHAPED = dict(greedy_barriers="none", greedy_priority="wavefront", greedy_split_shift=True,
              pressure_inflation_threshold=0.95)
MIRRORED = dict(merge_hash_stages=True, index_form="mirrored")

# name -> (build options, sha256 prefix of the ssa_ops without `region`, of the kernel instrs)
MATRIX = {
    "r2_default": (dict(rounds=2), "86bda95c612ef005", "90d4c2ba7ef09f6c"),
    "r4_default": (dict(rounds=4), "6264f623350489fa", "c26f32421fe591a0"),
    "r16_plan_flow90": (dict(rounds=16, **MIRRORED, **SHAPED,
                             emission_plan=os.path.join(ROOT, "docs", "emission_plans", "flow90.json")),
                        "989a6d6c6e353f1f", "85f4dfd740813798"),
    "r16_zero_mobius": (dict(rounds=16, cached_select="mobius", prune_dead_index=True, **SHAPED),
                        "af87789a07be12ef", "cf858893dc0356a1"),
}


def build(tmp_path, name, options):
    path = tmp_path / f"sched_{name}.json"
    instrs, _ = build_kernel_with_args(None, verbose=False, output_json=str(path), use_git_sha=False,
                                       use_greedy=True, **options)
    with open(path) as f:
        sched = json.load(f)
    return json.loads(json.dumps(list(instrs))), sched


def sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


@pytest.mark.parametrize("name", sorted(MATRIX))
def test_program_and_kernel_match_main_without_region(tmp_path, name):
    options, ssa_sha, kernel_sha = MATRIX[name]
    if "emission_plan" in options:
        from solver.emission_plan import load_plan
        options = dict(options, emission_plan=load_plan(options["emission_plan"]))
    instrs, sched = build(tmp_path, name, options)
    ops = [{k: v for k, v in op.items() if k != "region"} for op in sched["ssa_ops"]]
    assert sha(ops) == ssa_sha
    assert hashlib.sha256(json.dumps(instrs).encode()).hexdigest()[:16] == kernel_sha


def test_region_field_in_the_scheduler_json(tmp_path):
    _, sched = build(tmp_path, "r2", dict(rounds=2))
    tagged = [op for op in sched["ssa_ops"] if "region" in op]
    assert tagged and len(tagged) < len(sched["ssa_ops"])  # init and stores are untagged
    kinds = {op["region"]["kind"] for op in tagged}
    assert {"node_xor", "hash", "index_step"} <= kinds
    assert any(k.startswith("tree_access:") for k in kinds)
    for op in tagged:
        region = op["region"]
        assert set(region) <= {"id", "kind", "round", "group", "depth"}
        assert isinstance(region["id"], int)
        if region["kind"] != "cached_level_setup":
            assert region["round"] == op["source_info"]["round"]
