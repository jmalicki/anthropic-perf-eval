"""Emission plans (solver/emission_plan.py, docs/EMISSION_STRUCTURE_MODEL.md): the tree-access
encoding chosen per round and per group.

- A plan that names, for every group, what the flags would choose emits exactly the SSA
  program of those flags (same ops, operands and order).
- Plans that mix encodings within a round build kernels that are PROVED equal to their SSA
  program and PASS numeric validation, at every cached depth and with all three bit-driven
  encodings in one round.
- Invalid plans are rejected before anything is built.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scheduler_rs  # noqa: E402
from generate_ssa_kernel import validate_kernel  # noqa: E402
from kernel_build_args import args_to_build_kwargs  # noqa: E402
from solver.emission_plan import compact_round, describe_plan, expand_plan, plan_document  # noqa: E402
from verification.kernel_dataflow import check  # noqa: E402

SEED = 123
SAFE = dict(merge_hash_stages=True, index_form="mirrored")


def build(tmp_path, name, rounds, batch_size, prove=True, **options):
    """Greedy build through kernel_build_args (as the CLI does). Returns (instrs, sched)."""
    path = tmp_path / f"sched_{name}.json"
    kwargs = args_to_build_kwargs(None, output_json=str(path), use_git_sha=False, use_greedy=True,
                                  rounds=rounds, batch_size=batch_size, **options)
    kwargs["verbose"] = False
    instrs, stats = scheduler_rs.build_tick_tock_kernel_ssa(**kwargs)
    assert not stats["budget_exceeded"]
    instrs = json.loads(json.dumps(list(instrs)))
    with open(path) as f:
        sched = json.load(f)
    if prove:
        report = check({"instrs": instrs}, sched)
        assert report.ok, report.findings[:2]  # PROVED
        assert validate_kernel(instrs, 10, batch_size, rounds, SEED, values_only=bool(options.get("values_only")))
    return instrs, sched


def program(sched):
    """The SSA program without scheduling results."""
    return [{k: v for k, v in op.items() if k != "cycle"} for op in sched["ssa_ops"]]


def uniform_plan(rounds, rule):
    return [rule(r, r % 11) for r in range(rounds)]


@pytest.mark.parametrize("flags,rule", [
    (dict(cached_select="bits"), lambda r, d: "bits" if d <= 3 else "gather"),
    (dict(cached_select="bits-flow"), lambda r, d: "bits-flow" if d <= 3 else "gather"),
    (dict(cached_select="mobius", tree_reduce_max_depth=4, depth4_select_rounds="after-wrap"),
     lambda r, d: "mobius" if d <= 3 or (d == 4 and r >= 11) else "gather"),
])
def test_plan_reproduces_flag_emission(tmp_path, flags, rule):
    """16 rounds, 32 groups: same SSA program as the flags the plan spells out."""
    _, by_flags = build(tmp_path, "flags", 16, 256, prove=False, **SAFE, **flags)
    plan_flags = {k: v for k, v in flags.items() if k == "tree_reduce_max_depth"}
    _, by_plan = build(tmp_path, "plan", 16, 256, prove=False, **SAFE, **plan_flags,
                       emission_plan={"access": uniform_plan(16, rule)})
    assert program(by_plan) == program(by_flags)


def test_mixed_plan_is_proved_and_passes(tmp_path):
    """Every cached depth mixes encodings; one round has all four; levels 1 to 4 also gather."""
    groups = 4
    access = [
        "bits",                                                   # 0
        {"gather": 1, "bits": 1, "bits-flow": 1, "mobius": 1},    # 1
        ["mobius", "bits-flow", "gather", "bits"],                # 2
        {"bits-flow": 2, "gather": 2},                            # 3
        {"gather": 3, "bits": 1},                                 # 4: level 4 cached in one group
        "gather", "gather", "gather", "gather", "gather", "gather",  # 5 to 10
        "mobius",                                                 # 11
        {"bits": 2, "gather": 2},                                 # 12
        "bits-flow",                                              # 13
        {"mobius": 1, "bits": 3},                                 # 14
        {"bits-flow": 1, "mobius": 1, "gather": 2},               # 15
    ]
    _, sched = build(tmp_path, "mixed", 16, groups * 8, **SAFE, emission_plan={"access": access})
    by_round = {}
    for op in sched["ssa_ops"]:
        info = op["source_info"]
        if info["phase"] == "tree_access" and info["group"] is not None:
            by_round.setdefault(info["round"], set()).add(info["group"])
    # Every bit-driven group above depth 0 and every gathering group emits its own access ops.
    assert by_round[1] == {0, 1, 2, 3}
    assert by_round[4] == {0, 1, 2, 3}


def test_mixed_plan_zero_based_index(tmp_path):
    access = ["bits", {"gather": 1, "mobius": 1}, "bits-flow", {"bits": 1, "gather": 1},
              {"gather": 1, "bits-flow": 1}] + ["gather"] * 6 + ["bits", "gather", "mobius", "bits", "gather"]
    build(tmp_path, "zero", 16, 16, merge_hash_stages=True, emission_plan={"access": access})


def test_plan_at_another_forest_height(tmp_path):
    """Height 3, 4 rounds: round 3 visits depth 3 (rounds wrap at forest_height + 1). The level-3
    nodes must be loaded for a plan that selects there (they were not when the depth schedule
    used forest_height as its modulus, and the build panicked)."""
    access = ["bits", {"gather": 1, "bits-flow": 1}, "mobius", {"bits": 1, "gather": 1}]
    path = tmp_path / "sched_h3.json"
    kwargs = args_to_build_kwargs(None, output_json=str(path), use_git_sha=False, use_greedy=True, rounds=4,
                                  batch_size=16, forest_height=3, **SAFE, emission_plan={"access": access})
    kwargs["verbose"] = False
    instrs, stats = scheduler_rs.build_tick_tock_kernel_ssa(**kwargs)
    assert not stats["budget_exceeded"]
    instrs = json.loads(json.dumps(list(instrs)))
    with open(path) as f:
        sched = json.load(f)
    assert check({"instrs": instrs}, sched).ok
    assert validate_kernel(instrs, 3, 16, 4, SEED)


def test_huge_group_count_is_rejected_without_expanding():
    with pytest.raises(ValueError, match="sum to 1000000000"):
        expand_plan({"access": [{"bits": 10 ** 9}]}, 1, 2, 10)
    with pytest.raises(ValueError, match="non-negative integer"):
        expand_plan({"access": [{"bits": True, "gather": 1}]}, 1, 2, 10)


@pytest.mark.parametrize("access,message", [
    (["gather"] + ["bits"] * 15, "depth 0"),
    (["bits"] * 16, "depth 5"),
    (["bits"] * 15, "15 rounds"),
    (["bits", ["bits"]] + ["gather"] * 14, "1 entries"),
    (["bits", {"bits": 1}] + ["gather"] * 14, "sum to 1"),
    (["bits", "fetch"] + ["gather"] * 14, "unknown encoding"),
])
def test_invalid_plans_are_rejected(access, message):
    with pytest.raises(ValueError, match=message):
        expand_plan({"access": access}, 16, 2, 10)


def test_binding_rejects_invalid_plan_and_flag_conflicts():
    base = dict(forest_height=10, n_nodes=2047, batch_size=16, rounds=3, verbose=False)
    with pytest.raises(ValueError, match="depth 0"):
        scheduler_rs.build_tick_tock_kernel_ssa(**base, emission_plan=[["gather", "gather"]] * 3)
    with pytest.raises(ValueError, match="do not combine"):
        scheduler_rs.build_tick_tock_kernel_ssa(**base, cached_select="bits", emission_plan=[["bits", "bits"]] * 3)
    with pytest.raises(ValueError, match="do not combine"):
        args_to_build_kwargs(None, use_git_sha=False, rounds=3, batch_size=16, cached_select="bits",
                             emission_plan={"access": ["bits"] * 3})


def test_plan_file_round_trip(tmp_path):
    rows = expand_plan({"access": ["bits", {"gather": 1, "bits-flow": 1}, ["mobius", "bits"]]}, 3, 2, 10)
    assert rows == [["bits", "bits"], ["gather", "bits-flow"], ["mobius", "bits"]]
    assert compact_round(["a", "b", "a"]) == ["a", "b", "a"]
    doc = plan_document(rows, 10, comment="x")
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(doc))
    kwargs = args_to_build_kwargs(None, use_git_sha=False, rounds=3, batch_size=16, emission_plan=str(path))
    assert kwargs["emission_plan"] == rows
    assert json.loads(kwargs["run_options_json"])["emission_plan"] == rows
    assert describe_plan(rows) == "r0 bits; r1 gather x1 + bits-flow x1; r2 mobius x1 + bits x1"
