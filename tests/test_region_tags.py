"""Region tags (`SsaOp::region`) change nothing about the emitted program.

The SSA program of a small matrix of builds, without the `region` field and without anything
the scheduler decides (`cycle`), is compared with fingerprints of the emitted program, recorded
on main just before tags existed (e6f2579) and unchanged with them (61d7798). They
pin emission only: a scheduler change does not touch them, a change that alters the emission
on purpose must update them and say why. The kernel itself is not fingerprinted (it depends
on the schedule and the allocation; the scheduler reads no region tag).
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

# Fields of an SSA op left out of the fingerprint: the tag itself, and what the scheduler decides.
NOT_EMISSION = {"region", "cycle"}

# name -> (build options, sha256 prefix of the ssa_ops without NOT_EMISSION)
MATRIX = {
    "r2_default": (dict(rounds=2), "85822d1d80955e7e"),
    "r4_default": (dict(rounds=4), "988343cd32747388"),
    "r16_plan_flow90": (dict(rounds=16, **MIRRORED, **SHAPED,
                             emission_plan=os.path.join(ROOT, "docs", "emission_plans", "flow90.json")),
                        "3e1b14d2726bcc96"),
    "r16_zero_mobius": (dict(rounds=16, cached_select="mobius", prune_dead_index=True, **SHAPED),
                        "235da91a3694ed01"),
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
def test_program_matches_without_region(tmp_path, name):
    options, ssa_sha = MATRIX[name]
    if "emission_plan" in options:
        from solver.emission_plan import load_plan
        options = dict(options, emission_plan=load_plan(options["emission_plan"]))
    _, sched = build(tmp_path, name, options)
    ops = [{k: v for k, v in op.items() if k not in NOT_EMISSION} for op in sched["ssa_ops"]]
    assert sha(ops) == ssa_sha


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
