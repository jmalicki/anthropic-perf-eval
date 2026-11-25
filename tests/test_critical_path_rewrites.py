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
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "sched.json")
        instrs, stats = build_kernel_with_args(verbose=False, output_json=path, **dict(BASE, **options))
        with open(path) as f:
            sched = json.load(f)
    instrs = json.loads(json.dumps(list(instrs)))
    assert check({"instrs": instrs}, sched).ok, f"dataflow replay failed for {options}"
    assert validate_kernel(instrs, HEIGHT, BATCH, ROUNDS, 123), f"validation failed for {options}"
    bops = sb.ops_from_schedule_json(sched, "run")
    h, tl = sb.heads(bops), sb.tails(bops)
    ssa = {o["id"]: o for o in sched["ssa_ops"]}
    gathers = [o.id for o in bops if ssa.get(o.id, {}).get("opcode") == "load"
               and (ssa[o.id].get("source_info") or {}).get("group") is not None]
    vector_ops = sum(1 for o in sched["ssa_ops"] if o["engine"] == "valu")
    return {"t": min(h[i] for i in gathers), "q": min(tl[i] for i in gathers), "vector_ops": vector_ops}


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
