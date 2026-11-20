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
    values_only = bool(options.get("values_only"))
    # PASSED: values and indices match the reference (values only for --values-only kernels).
    assert validate_kernel(instrs, 10, BATCH, rounds, SEED, values_only=values_only)
    if values_only:
        # The flag changes the output: the indices in memory are not the reference's.
        assert not validate_kernel(instrs, 10, BATCH, rounds, SEED)
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


# ---------------------------------------------------------------------------
# --index-form mirrored
# ---------------------------------------------------------------------------
def test_mirrored_index_updates_in_two_ops(tmp_path):
    """6 rounds visit depths 0 to 5: root, three cached levels, two gathers."""
    rounds, groups = 6, BATCH // 8
    base, base_sched = build(tmp_path, rounds)
    mirrored, mirrored_sched = build(tmp_path, rounds, index_form="mirrored")
    # Zero-based: AND, SUB, multiply_add every round. Mirrored: AND every round, plus one
    # multiply_add except at depth 0, where n is the bit itself.
    assert len(phase_ops(base_sched, "branch")) == 3 * groups * rounds
    assert len(phase_ops(mirrored_sched, "branch")) == groups * (rounds + rounds - 1)
    assert not phase_ops(mirrored_sched, "branch", "-")
    # The initial indices are not loaded (taken to be 0); the final ones are still stored.
    assert len(phase_ops(base_sched, "load_data", "vload")) == 2 * groups
    assert len(phase_ops(mirrored_sched, "load_data", "vload")) == groups
    assert len(phase_ops(mirrored_sched, "store_data", "vstore")) == 2 * groups
    assert vector_op_equivalents(mirrored) < vector_op_equivalents(base)


@pytest.mark.parametrize("rounds", [1, 3, 11, 12, 13])
def test_mirrored_index_stores_reference_indices(tmp_path, rounds):
    """Final depth 1 and 3; the wrap round last (11), and one and two rounds after it."""
    build(tmp_path, rounds, index_form="mirrored")  # PROVED, and PASSED compares the stored indices


def test_mirrored_index_has_no_wrap_broadcast(tmp_path):
    """After the bottom level every index is 0: zero-based broadcasts 0 per group, mirrored emits nothing."""
    rounds, groups = 11, BATCH // 8
    _, base_sched = build(tmp_path, rounds)
    _, mirrored_sched = build(tmp_path, rounds, index_form="mirrored")
    wrap_round = [o for o in phase_ops(base_sched, "branch") if o["source_info"]["round"] == 10]
    assert [o["opcode"] for o in wrap_round] == ["vbroadcast"] * groups
    assert not [o for o in phase_ops(mirrored_sched, "branch") if o["source_info"]["round"] == 10]


def test_index_form_rejects_unknown_value(tmp_path):
    with pytest.raises(ValueError, match="index_form"):
        scheduler_rs.build_tick_tock_kernel_ssa(
            forest_height=10, n_nodes=2047, batch_size=BATCH, rounds=2, verbose=False, index_form="one-based")


# ---------------------------------------------------------------------------
# --prune-dead-index
# ---------------------------------------------------------------------------
def test_prune_dead_index_drops_known_index_work(tmp_path):
    """12 rounds: depth 0 twice (rounds 0 and 11) and the wrap round (10)."""
    rounds, groups = 12, BATCH // 8
    base, base_sched = build(tmp_path, rounds)
    pruned, pruned_sched = build(tmp_path, rounds, prune_dead_index=True)

    def branch(sched, rnd):
        return sorted(o["opcode"] for o in phase_ops(sched, "branch") if o["source_info"]["round"] == rnd)

    for rnd in (0, 11):  # depth 0: no multiply_add, the offset is the index
        assert branch(base_sched, rnd) == sorted(["&", "-", "multiply_add"] * groups)
        assert branch(pruned_sched, rnd) == sorted(["&", "-"] * groups)
    assert branch(base_sched, 10) == ["vbroadcast"] * groups
    # The zeros of the wrap round are not read (round 11 is at depth 0), so nothing is emitted.
    assert branch(pruned_sched, 10) == []
    assert branch(pruned_sched, 5) == branch(base_sched, 5)
    assert len(phase_ops(base_sched, "load_data", "vload")) == 2 * groups
    assert len(phase_ops(pruned_sched, "load_data", "vload")) == groups
    assert vector_op_equivalents(pruned) < vector_op_equivalents(base)


def test_prune_dead_index_shares_one_zero_vector_when_the_wrap_round_is_last(tmp_path):
    rounds, groups = 11, BATCH // 8
    _, base_sched = build(tmp_path, rounds)
    _, pruned_sched = build(tmp_path, rounds, prune_dead_index=True)
    wrap = lambda sched: [o["opcode"] for o in phase_ops(sched, "branch") if o["source_info"]["round"] == 10]  # noqa: E731
    assert wrap(base_sched) == ["vbroadcast"] * groups
    assert wrap(pruned_sched) == ["vbroadcast"]


def test_prune_dead_index_is_a_no_op_for_the_mirrored_form(tmp_path):
    _, mirrored = build(tmp_path, 3, index_form="mirrored")
    _, both = build(tmp_path, 3, index_form="mirrored", prune_dead_index=True)
    assert [(o["opcode"], o["engine"]) for o in mirrored["ssa_ops"]] == \
           [(o["opcode"], o["engine"]) for o in both["ssa_ops"]]


# ---------------------------------------------------------------------------
# --values-only
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("index_form", ["zero-based", "mirrored"])
def test_values_only_stores_no_indices_and_drops_what_fed_them(tmp_path, index_form):
    """3 rounds (depths 0, 1, 2): the selects of rounds 1 and 2 read the index, round 2's update is dead."""
    rounds, groups = 3, BATCH // 8
    full, full_sched = build(tmp_path, rounds, index_form=index_form)
    values, values_sched = build(tmp_path, rounds, index_form=index_form, values_only=True)
    assert len(phase_ops(full_sched, "store_data", "vstore")) == 2 * groups
    assert len(phase_ops(values_sched, "store_data", "vstore")) == groups
    last = [o for o in phase_ops(values_sched, "branch") if o["source_info"]["round"] == rounds - 1]
    assert not last, "the last round's index update only fed the index store"
    assert [o for o in phase_ops(values_sched, "branch") if o["source_info"]["round"] == rounds - 2]
    assert vector_op_equivalents(values) < vector_op_equivalents(full)
    assert values["engines"]["store"]["ops"] == groups


def test_validate_only_remembers_values_only(tmp_path):
    """A kernel saved by a --values-only build is re-validated on values only."""
    from generate_ssa_kernel import load_kernel_from_json, save_kernel_to_json

    path = tmp_path / "kernel.json"
    save_kernel_to_json([{"alu": [("+", 1, 2, 3)]}], str(path), 10, BATCH, 2, SEED, values_only=True)
    _, params = load_kernel_from_json(str(path))
    assert params["values_only"] is True
    save_kernel_to_json([{"alu": [("+", 1, 2, 3)]}], str(path), 10, BATCH, 2, SEED)
    assert load_kernel_from_json(str(path))[1]["values_only"] is False
