"""Emission options (docs/EMISSION_ABLATION.md): each must build a kernel that is PROVED equal
to its SSA program, PASSES numeric validation against the reference, and emits fewer ops than
the default emission.

Builds use the greedy scheduler at 3 rounds (root, depth-1 select, depth-2 select) or 6 rounds
(adds the depth-3 select and two gather rounds), 2 groups.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scheduler_rs  # noqa: E402
from generate_ssa_kernel import validate_kernel  # noqa: E402
from scripts import schedule_report as sr  # noqa: E402
from verification.kernel_dataflow import check  # noqa: E402

BATCH = 16
SEED = 123


def build(tmp_path, rounds, **options):
    """Build with the greedy scheduler; require PROVED and PASSED.

    Returns the schedule report (scripts.schedule_report.analyze) and the scheduler JSON.
    """
    path = tmp_path / ("sched_" + "_".join(f"{k}-{v}" for k, v in sorted(options.items())) + f"_{rounds}.json")
    instrs, stats = scheduler_rs.build_tick_tock_kernel_ssa(
        forest_height=10, n_nodes=2047, batch_size=BATCH, rounds=rounds, enable_valu_shift=True,
        verbose=False, use_cpsat=False, output_json=str(path), **options,
    )
    assert not stats["budget_exceeded"]
    instrs = json.loads(json.dumps(list(instrs)))
    with open(path) as f:
        sched = json.load(f)
    report = check({"instrs": instrs}, sched)
    assert report.ok, report.findings[:2]  # PROVED: kernel equals the SSA program
    assert validate_kernel(instrs, 10, BATCH, rounds, SEED)  # PASSED: values and indices match the reference
    return sr.analyze(sched), sched


def vector_op_equivalents(rep):
    return rep["emission"]["vector_op_equivalents"]


def phase_ops(sched, phase, opcode=None):
    """SSA ops emitted in one phase (hash, branch, tree_access, xor), optionally one opcode."""
    return [o for o in sched["ssa_ops"]
            if (o.get("source_info") or {}).get("phase") == phase and (opcode is None or o["opcode"] == opcode)]


# ---------------------------------------------------------------------------
# --merge-hash-stages
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rounds", [2, 3])
def test_merge_hash_stages_saves_one_op_per_hash(tmp_path, rounds):
    base, base_sched = build(tmp_path, rounds)
    merged, merged_sched = build(tmp_path, rounds, merge_hash_stages=True)
    groups = BATCH // 8
    assert len(phase_ops(base_sched, "hash")) == 11 * groups * rounds
    assert len(phase_ops(merged_sched, "hash")) == 10 * groups * rounds
    assert vector_op_equivalents(merged) < vector_op_equivalents(base)
    assert merged["floor"]["cycles"] <= base["floor"]["cycles"]
    # Two multiply_adds and an XOR replace multiply_add, add, shift, XOR: no shift by 9 is left.
    assert len(phase_ops(merged_sched, "hash", "multiply_add")) == 4 * groups * rounds
    assert len(phase_ops(base_sched, "hash", "multiply_add")) == 3 * groups * rounds
