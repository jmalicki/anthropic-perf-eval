"""Tests for verification.ssa_equivalence (SSA program vs SSA program, for all valid inputs).

Positive: the default emission is proved equal to every single emission option and to
combinations. Soundness: one op of a program is mutated (wrong constant, swapped select arms,
off-by-one load address) consistently in its SSA program and its kernel (kernel_dataflow proves
the two still agree); the checker must reject the mutant with a counterexample input, and the
two kernels must really write different words on that input on problem.Machine.
"""

import copy
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("z3")

import scheduler_rs  # noqa: E402
from kernel_build_args import EMISSION_DEFAULTS  # noqa: E402
from problem import N_CORES, DebugInfo, Machine  # noqa: E402
from verification import ssa_equivalence as se  # noqa: E402
from verification.kernel_dataflow import check as dataflow_check  # noqa: E402

ROUNDS, BATCH, HEIGHT = 16, 8, 10
SHAPE = (ROUNDS, BATCH, HEIGHT)
HASH_STAGE2_CONST = 0x165667B1


def _build(tmp_path_factory, name, rounds=ROUNDS, batch=BATCH, **emission):
    path = tmp_path_factory.mktemp(name) / "sched.json"
    kwargs = dict(forest_height=HEIGHT, n_nodes=2 ** (HEIGHT + 1) - 1, batch_size=batch, rounds=rounds,
                  enable_valu_shift=True, verbose=False, use_cpsat=False, output_json=str(path),
                  **EMISSION_DEFAULTS)
    kwargs.update(emission)
    instrs, _ = scheduler_rs.build_tick_tock_kernel_ssa(**kwargs)
    with open(path) as f:
        sched = json.load(f)
    return {"instrs": json.loads(json.dumps(list(instrs)))}, sched


@pytest.fixture(scope="module")
def default(tmp_path_factory):
    return _build(tmp_path_factory, "default")


@pytest.fixture(scope="module")
def bits(tmp_path_factory):
    return _build(tmp_path_factory, "bits", cached_select="bits")


VARIANTS = {
    "merge-hash": {"merge_hash_stages": True},
    "mirrored": {"index_form": "mirrored"},
    "prune-dead-index": {"prune_dead_index": True},
    "bits": {"cached_select": "bits"},
    "bits-flow": {"cached_select": "bits-flow"},
    "mobius": {"cached_select": "mobius"},
    "bits-path": {"cached_select": "bits", "select_bit_source": "path"},
    "values-only": {"values_only": True},
    "merge+mirrored+bits": {"merge_hash_stages": True, "index_form": "mirrored", "cached_select": "bits"},
    "merge+mirrored+bits+depth4": {"merge_hash_stages": True, "index_form": "mirrored", "cached_select": "bits",
                                   "tree_reduce_max_depth": 4},
}


@pytest.mark.parametrize("name", list(VARIANTS))
def test_default_equals_each_emission_option(tmp_path_factory, default, name):
    options = VARIANTS[name]
    _, sched = _build(tmp_path_factory, name, **options)
    result = se.check_equivalence(default[1], sched, values_only=bool(options.get("values_only")), shape=SHAPE)
    assert result.verdict == "proved", se.format_result(result)
    assert result.proved_words == result.compared == (BATCH if options.get("values_only") else 2 * BATCH)


def test_two_round_combination_is_proved(tmp_path_factory):
    _, a = _build(tmp_path_factory, "r2a", rounds=2, batch=32)
    _, b = _build(tmp_path_factory, "r2b", rounds=2, batch=32, merge_hash_stages=True, index_form="mirrored",
                  cached_select="bits")
    result = se.check_equivalence(a, b, shape=(2, 32, HEIGHT))
    assert result.verdict == "proved", se.format_result(result)


# =============================================================================
# Mutations: the same change in the SSA program and in the kernel
# =============================================================================


def _run_machine(instrs, image):
    machine = Machine(list(image), list(instrs), DebugInfo(scratch_map={}), n_cores=N_CORES, trace=False)
    machine.enable_pause = False
    machine.enable_debug = False
    machine.prints = False
    machine.run()
    return machine.mem


def _mutate(built, pick, change):
    """Apply `change(ssa_op, kernel_slot)` to the first SSA op `pick` accepts (and its one
    lowered slot), and return the mutant (kernel, sched)."""
    kernel, sched = copy.deepcopy(built[0]), copy.deepcopy(built[1])
    lowered = {}
    for op in sched["ops"]:
        lowered.setdefault(op["id"], []).append(op)
    for op in sched["ssa_ops"]:
        lows = lowered.get(op["id"], [])
        if len(lows) == 1 and pick(op, lows[0]):
            low = lows[0]
            want = [str(v) for v in low["slot"]]
            # The kernel starts with a pause bundle, so schedule cycle c is bundle c + 1.
            bundle = kernel["instrs"][low["cycle"] + 1].get(low["engine"], [])
            slot = next(s for s in bundle if [str(v) for v in s] == want)
            change(op, slot)
            assert dataflow_check(kernel, sched).ok, "mutation is not the same in the SSA program and the kernel"
            return kernel, sched
    raise AssertionError("no op to mutate")


def _assert_rejected_on_machine(base, mutant, values_only=False):
    result = se.check_equivalence(base[1], mutant[1], values_only=values_only, shape=SHAPE)
    assert result.verdict == "counterexample", se.format_result(result)
    d = result.difference
    got_a = _run_machine(base[0]["instrs"], d.image)
    got_b = _run_machine(mutant[0]["instrs"], d.image)
    assert got_a[d.address] == d.value_a
    assert got_b[d.address] == d.value_b
    assert d.value_a != d.value_b
    return result


def _vreg_addr(sched, vid):
    for key, addr in sched["vreg_to_addr"].items():
        if key.startswith(f"VReg {{ id: {vid},"):
            return addr
    raise KeyError(vid)


def test_wrong_hash_constant_is_rejected(default):
    def change(op, slot):
        op["srcs"][0]["Imm"] += 1
        slot[-1] = int(slot[-1]) + 1

    mutant = _mutate(default, lambda op, low: op["opcode"] == "const_choice"
                     and op["srcs"][0]["Imm"] == HASH_STAGE2_CONST, change)
    _assert_rejected_on_machine(default, mutant)


@pytest.mark.parametrize("which", ["default", "bits"])
def test_swapped_select_arms_are_rejected(default, bits, which):
    built = {"default": default, "bits": bits}[which]

    def change(op, slot):
        op["srcs"][1], op["srcs"][2] = op["srcs"][2], op["srcs"][1]
        slot[3], slot[4] = slot[4], slot[3]

    mutant = _mutate(built, lambda op, low: op["opcode"] == "select_choice" and low["slot"][0] == "vselect", change)
    _assert_rejected_on_machine(built, mutant)
    # A rewritten program against the mutant of another emission is rejected too.
    if which == "bits":
        _assert_rejected_on_machine(default, mutant)


def test_off_by_one_load_address_is_rejected(default):
    sched = default[1]
    one = next(op for op in sched["ssa_ops"] if op["opcode"] == "const_choice" and op["srcs"][0]["Imm"] == 1)
    one_vid = int(one["dest"][1:].split("_")[0])

    def change(op, slot):
        op["srcs"][1] = {"VReg": {"id": one_vid, "width": "scalar"}}
        slot[3] = _vreg_addr(sched, one_vid)

    # The address of the root node, forest_values_p + 0 (the zero VReg), becomes + 1.
    mutant = _mutate(default, lambda op, low: op["opcode"] == "+" and low["engine"] == "alu"
                     and op["srcs"][1].get("VReg", {}).get("id") == 0, change)
    _assert_rejected_on_machine(default, mutant)


def test_values_only_build_differs_in_indices_unless_declared(tmp_path_factory, default):
    vo = _build(tmp_path_factory, "vo", values_only=True)
    result = _assert_rejected_on_machine(default, vo)
    assert result.difference.region == "indices[0]"
    assert se.check_equivalence(default[1], vo[1], values_only=True, shape=SHAPE).verdict == "proved"


def test_z3_unknown_is_never_a_pass(monkeypatch, default, bits):
    monkeypatch.setattr(se.Sweeper, "_prove", lambda self, x, y, deep: "unknown")
    result = se.check_equivalence(default[1], bits[1], shape=SHAPE)
    assert result.verdict == "unproved"


# =============================================================================
# Normaliser and memory model
# =============================================================================


def _terms():
    layout = se.Layout.from_params(2, 8, 2)
    return se.NormTerms(layout, layout.sample_images(8, 0)), layout


def test_normal_forms_merge_rewritten_hash_stages():
    """R8 by normalisation alone: stage 3 of stage 2 equals the merged form."""
    from verification.rewrites import merged_stage_constants

    T, layout = _terms()
    a = T.mem(T.const(layout.values_p))
    x = T.fma(a, T.const(33), T.const(0x165667B1))
    stage3 = T.binop("^", T.binop("+", x, T.const(0xD3A2646C)), T.binop("<<", x, T.const(9)))
    m1, c1, m2, c2 = merged_stage_constants(2)
    merged = T.binop("^", T.fma(a, T.const(m1), T.const(c1)), T.fma(a, T.const(m2), T.const(c2)))
    assert stage3 == merged


def test_fma_with_non_boolean_factor_is_not_a_select():
    T, layout = _terms()
    a, b = T.mem(T.const(layout.values_p)), T.mem(T.const(layout.values_p + 1))
    assert T.defs[T.fma(a, b, T.const(0))][0] == "*"
    bit = T.binop("&", a, T.const(1))
    assert T.defs[T.fma(bit, b, a)][0] == "sel"


def test_precondition_folding_matches_layout():
    T, layout = _terms()
    assert T.const_value(T.mem(T.const(4))) == layout.forest_p
    assert T.const_value(T.mem(T.const(layout.indices_p))) == 0
    assert T.const_value(T.mem(T.const(layout.values_p))) is None


def _op(i, opcode, dest, srcs):
    return {"id": i, "opcode": opcode, "dest": dest, "cycle": 0, "source_info": {},
            "srcs": [{"Imm": s} if isinstance(s, int) else {"VReg": {"id": s[0], "width": "scalar"}} for s in srcs]}


def _program(store_value_from_load: bool):
    """Store to values[0]; the program also loads values[0]. The load precedes the store only
    if the stored value depends on it."""
    layout = se.Layout.from_params(2, 8, 2)
    ops = [_op(0, "const_choice", "v1_s", [layout.values_p]),
           _op(1, "load", "v2_s", [(1,)]),
           _op(2, "const_choice", "v3_s", [7]),
           _op(3, "+", "v4_s", [(2,), (3,)]),
           _op(4, "store", None, [(1,), (4,) if store_value_from_load else (3,)])]
    return {"ssa_ops": ops, "vector_group_lanes": {}}


def test_memory_order_is_checked():
    ok = se.check_equivalence(_program(True), _program(True), shape=(2, 8, 2))
    assert ok.verdict == "proved", se.format_result(ok)
    bad = se.check_equivalence(_program(False), _program(False), shape=(2, 8, 2))
    assert bad.verdict == "unproved"
    assert any("not ordered before the store" in u for u in bad.unproved)
