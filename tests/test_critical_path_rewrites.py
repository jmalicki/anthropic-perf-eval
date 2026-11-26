"""Critical-path rewrites (docs/CRITICAL_PATH.md): each is proved against its SSA program and
validated, changes the op count by what it claims, and shortens the dependency chain it
targets in the static bound's heads and tails. 6 rounds, batch 16, mirrored index, path bits,
bits-flow on the cached levels: rounds 1 to 3 select, rounds 4 and 5 gather.
"""
import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate_ssa_kernel import validate_kernel  # noqa: E402
from kernel_build_args import build_kernel_with_args  # noqa: E402
from scripts import static_bounds as sb  # noqa: E402
from verification.kernel_dataflow import check  # noqa: E402

ROUNDS, BATCH, HEIGHT = 6, 16, 10
BASE = dict(rounds=ROUNDS, batch_size=BATCH, use_greedy=True, merge_hash_stages=True, index_form="mirrored",
            cached_select="bits-flow", select_bit_source="path")


def build(**options):
    opts = dict(BASE, **options)
    height, batch, rounds = opts.get("forest_height", HEIGHT), opts["batch_size"], opts["rounds"]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "sched.json")
        instrs, stats = build_kernel_with_args(verbose=False, output_json=path, **opts)
        with open(path) as f:
            sched = json.load(f)
    instrs = json.loads(json.dumps(list(instrs)))
    assert check({"instrs": instrs}, sched).ok, f"dataflow replay failed for {options}"
    for seed in (123, 7):
        assert validate_kernel(instrs, height, batch, rounds, seed), f"validation failed for {options} (seed {seed})"
    bops = sb.ops_from_schedule_json(sched, "run")
    h, tl = sb.heads(bops), sb.tails(bops)
    ssa = {o["id"]: o for o in sched["ssa_ops"]}
    gathers = [o.id for o in bops if ssa.get(o.id, {}).get("opcode") == "load"
               and (ssa[o.id].get("source_info") or {}).get("group") is not None]
    vector_ops = sum(1 for o in sched["ssa_ops"] if o["engine"] == "valu")
    madds = {}
    for o in sched["ssa_ops"]:
        if o["engine"] == "valu" and o["opcode"] == "multiply_add":
            r = (o.get("source_info") or {}).get("round")
            madds[r] = madds.get(r, 0) + 1
    return {"t": min(h[i] for i in gathers), "q": min(tl[i] for i in gathers), "vector_ops": vector_ops,
            "madds_by_round": madds}


@pytest.fixture(scope="module")
def base():
    return build()


def test_gather_xor_reassociation(base):
    r = build(reassociate_gather_xor=True)
    assert r["vector_ops"] == base["vector_ops"]
    assert r["q"] == base["q"] - 1 and r["t"] == base["t"]


def test_index_early_terms_final(base):
    r = build(index_early_terms="final")
    assert r["vector_ops"] == base["vector_ops"] + 1  # the broadcast of -2
    assert r["q"] == base["q"] - 1


def test_index_early_terms_first_gather(base):
    r = build(index_early_terms="first-gather")
    assert r["t"] == base["t"] - 1
    assert r["vector_ops"] > base["vector_ops"] + 1  # one more multiply_add per first gather


def test_select_xor_pushdown(base):
    r = build(select_xor_pushdown="first-pass")
    assert r["t"] == base["t"] - 3  # rounds 1, 2 and 3 select
    groups = BATCH // 8
    assert r["vector_ops"] == base["vector_ops"] + 3 * groups  # one more XOR per group and round


def test_index_early_terms_needs_mirrored():
    with pytest.raises(ValueError, match="mirrored"):
        build_kernel_with_args(verbose=False, **dict(BASE, index_form="zero-based", index_early_terms="final"))


# ----------------------------------------------------------------------------
# After the wrap. Height 3 (depths 0, 1, 2, 3, 0, ...; the depth-3 round is the wrap round),
# depths 0 to 2 cached (tree_reduce_max_depth 2), depth 3 gathered, batch 16. With 9 rounds
# the last round is at depth 0; with 8 it is the wrap round. Every build is proved by
# dataflow replay and validated against the reference (two seeds), so a wrong rewrite after
# the wrap fails here. The op-count checks show each rewrite really fires after the wrap.
# ----------------------------------------------------------------------------
WRAP = dict(forest_height=3, tree_reduce_max_depth=2, rounds=9, batch_size=16)
GROUPS = 16 // 8


def vector_ops(**options):
    return build(**dict(WRAP, **options))["vector_ops"]


@pytest.mark.parametrize("variant", [
    {},
    {"cached_select": "bits"},
    {"select_bit_source": "index"},
    {"merge_hash_stages": False},
])
def test_select_xor_pushdown_after_the_wrap(variant):
    none = vector_ops(**variant)
    first = vector_ops(select_xor_pushdown="first-pass", **variant)
    every = vector_ops(select_xor_pushdown="all", **variant)
    # Select rounds: 1, 2 before the wrap and 5, 6 after it with bits-flow (depth 1 too); bits
    # pushes only from depth 2 (rounds 2 and 6). One more XOR per group and pushed round.
    pushed_before, pushed_after = (1, 1) if variant.get("cached_select") == "bits" else (2, 2)
    assert first == none + pushed_before * GROUPS
    assert every == none + (pushed_before + pushed_after) * GROUPS


@pytest.mark.parametrize("rounds", [8, 9])
def test_index_early_terms_after_the_wrap(rounds):
    none = build(**dict(WRAP, rounds=rounds))
    first = build(**dict(WRAP, rounds=rounds, index_early_terms="first-gather"))
    # 'final' with the last round at depth 0 (9 rounds: n' = bit) or at the wrap round (8:
    # index 0) has no early form to use: rejected as a no-op (review of #91).
    with pytest.raises(ValueError, match="applies to no op"):
        build(**dict(WRAP, rounds=rounds, index_early_terms="final"))
    # 'first-gather': the gathers of round 3 and, after the wrap, round 7 (depth 3, the wrap
    # round) take madd(n, -2, F + L) - bit: one more multiply_add per group in those rounds.
    # n' is then read by nothing (the wrap round resets the index), so the update's
    # multiply_add in the round before (2 and 6) is dead: net only the broadcast of -2.
    for gather_round in (3, 7):
        assert first["madds_by_round"][gather_round] == none["madds_by_round"][gather_round] + GROUPS
        assert first["madds_by_round"][gather_round - 1] == none["madds_by_round"][gather_round - 1] - GROUPS
    assert first["vector_ops"] == none["vector_ops"] + 1


def test_index_early_terms_final_after_the_wrap_is_used():
    """7 rounds: the last round (6) is at depth 2, after the wrap, so the final store reads
    n' = madd(n, 2, bit) and 'final' rewrites it as madd(n, -2, L) - bit (n' then dead)."""
    none = build(**dict(WRAP, rounds=7))
    final = build(**dict(WRAP, rounds=7, index_early_terms="final"))
    # The round-6 update's multiply_add per group moves to the output store (no round), and
    # the broadcast of -2 is added; the tail of the gathers is one cycle shorter.
    assert final["madds_by_round"][6] == none["madds_by_round"][6] - GROUPS
    assert final["madds_by_round"].get(None, 0) == none["madds_by_round"].get(None, 0) + GROUPS
    assert final["vector_ops"] == none["vector_ops"] + 1
    assert final["q"] == none["q"] - 1


@pytest.mark.parametrize("rounds", [8, 9])
@pytest.mark.parametrize("merge", [True, False])
def test_all_three_after_the_wrap(rounds, merge):
    build(**dict(WRAP, rounds=rounds, merge_hash_stages=merge, reassociate_gather_xor=True,
                 index_early_terms="first-gather", select_xor_pushdown="all"))
    build(**dict(WRAP, rounds=7, merge_hash_stages=merge, reassociate_gather_xor=True,
                 index_early_terms="final", select_xor_pushdown="all", select_bit_source="index"))


@pytest.mark.parametrize("options, name", [
    ({"select_xor_pushdown": "all", "cached_select": "standard"}, "select_xor_pushdown"),
    ({"select_xor_pushdown": "all", "cached_select": "mobius"}, "select_xor_pushdown"),
    ({"reassociate_gather_xor": True, "tree_reduce_max_depth": 3}, "reassociate_gather_xor"),
])
def test_options_that_would_be_no_ops_are_rejected(options, name):
    """Rejected by ProblemSpec::check_critical_path_options, so a Rust caller gets the same
    error (Rust test test_check_critical_path_options)."""
    with pytest.raises(ValueError, match=name):
        build_kernel_with_args(verbose=False, **{**BASE, **WRAP, **options})


def test_pushdown_with_bits_only_at_depth_1_is_rejected():
    """bits at depth 1 has no SelectChoice at the top of its tree (a plan can ask for it; the
    flags clamp tree_reduce_max_depth to 2 or more)."""
    depth_enc = {0: "bits", 1: "bits", 2: "gather", 3: "gather"}
    plan = [[depth_enc[r % 4]] * 2 for r in range(9)]
    opts = {k: v for k, v in {**BASE, **WRAP}.items() if k not in ("cached_select", "tree_reduce_max_depth")}
    build_kernel_with_args(verbose=False, **opts, emission_plan=plan)  # the plan itself is fine
    with pytest.raises(ValueError, match="select_xor_pushdown"):
        build_kernel_with_args(verbose=False, **opts, emission_plan=plan, select_xor_pushdown="all")


def test_index_early_terms_final_with_values_only_is_rejected():
    """--values-only drops the final index store, so 'final' has nothing to rewrite."""
    with pytest.raises(ValueError, match="applies to no op"):
        build_kernel_with_args(verbose=False, **{**BASE, **WRAP, "rounds": 7, "values_only": True,
                                                 "index_early_terms": "final"})
