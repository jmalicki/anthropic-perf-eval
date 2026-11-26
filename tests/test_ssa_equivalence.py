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


# =============================================================================
# Small synthetic programs: A computes values[0] + 7 and stores it to values[0]
# =============================================================================

TINY = (2, 8, 2)
TINY_LAYOUT = se.Layout.from_params(*TINY)


class _Prog:
    """Scalar SSA program builder: x = values[0]; ...; store(result)."""

    def __init__(self):
        self.ops = []
        self.vp = self.const(TINY_LAYOUT.values_p)
        self.x = self.op("load", self.vp)

    def _add(self, opcode, srcs, dest=True):
        vid = 100 + len(self.ops)
        self.ops.append({"id": len(self.ops), "opcode": opcode, "dest": f"v{vid}_s" if dest else None,
                         "cycle": 0, "source_info": {}, "srcs": srcs})
        return vid

    def const(self, c):
        return self._add("const_choice", [{"Imm": c}])

    def op(self, opcode, *srcs):
        return self._add(opcode, [{"VReg": {"id": s, "width": "scalar"}} for s in srcs])

    def plus7(self):
        return self.op("+", self.x, self.const(7))

    def store(self, value, addr=None):
        self._add("store", [{"VReg": {"id": self.vp if addr is None else addr, "width": "scalar"}},
                            {"VReg": {"id": value, "width": "scalar"}}], dest=False)
        return {"ssa_ops": self.ops, "vector_group_lanes": {}}


def _a():
    p = _Prog()
    return p.store(p.plus7())


def _samples_agree(a, b):
    """The final memories agree on every sample input the checker uses, so a verdict on these
    programs comes from the prover, not from sampling."""
    for image in TINY_LAYOUT.sample_images(32, 0):
        image = [int(v) for v in image]
        if se.final_memory(a, image) != se.final_memory(b, image):
            return False
    return True


def _check(b):
    return se.check_equivalence(_a(), b, shape=TINY)


def test_z3_rejects_select_on_a_rare_value():
    p = _Prog()
    r = p.plus7()
    b = p.store(p.op("select", p.op("==", p.x, p.const(0x12345)), p.op("+", p.x, p.const(8)), r))
    assert _samples_agree(_a(), b)
    result = _check(b)
    assert result.verdict == "unproved" and result.stats.sat > 0


def test_z3_rejects_multiply_add_of_a_rare_comparison():
    p = _Prog()
    b = p.store(p.op("multiply_add", p.op("==", p.x, p.const(0x777)), p.const(1), p.plus7()))
    assert _samples_agree(_a(), b)
    assert _check(b).verdict == "unproved"


def test_z3_rejects_xor_of_a_rare_masked_comparison():
    p = _Prog()
    masked = p.op("&", p.x, p.const(0xFFFFF))
    b = p.store(p.op("^", p.plus7(), p.op("==", masked, p.const(0x1234))))
    assert _samples_agree(_a(), b)
    assert _check(b).verdict == "unproved"


def test_difference_only_outside_the_precondition_is_proved():
    """x == 2^30 is never true for inputs below 2^30: proved by the interval fact on the load."""
    p = _Prog()
    b = p.store(p.op("+", p.plus7(), p.op("==", p.x, p.const(1 << 30))))
    assert _samples_agree(_a(), b)
    result = _check(b)
    assert result.verdict == "proved", se.format_result(result)
    assert result.stats.queries > 0


def test_case_split_proves_a_select_identity():
    """sel(x & 1, x + 8, x + 7) == x + 7 + (x & 1): needs the split on the 0/1 condition."""
    p = _Prog()
    bit = p.op("&", p.x, p.const(1))
    b = p.store(p.op("select", bit, p.op("+", p.x, p.const(8)), p.plus7()))
    q = _Prog()
    bit_q = q.op("&", q.x, q.const(1))
    a = q.store(q.op("+", q.plus7(), bit_q))
    result = se.check_equivalence(a, b, shape=TINY)
    assert result.verdict == "proved", se.format_result(result)
    assert result.stats.split_cases > 0


def test_memo_does_not_share_answers_across_ranges():
    """(c & 1) == c holds for c in [0, 1] but not for c in [0, 3]: same structure, different
    cut-point ranges, so the two queries must not share a memo entry."""
    T = se.NormTerms(TINY_LAYOUT, TINY_LAYOUT.sample_images(4, 0))
    x = T.mem(T.const(TINY_LAYOUT.values_p))
    sweeper = se.Sweeper(T, 5000, 100)
    answers = []
    for mask in (1, 3):
        c = T.binop("&", x, T.const(mask))
        sweeper.mask[c] = se.BOTH  # shared by both programs: a cut point
        answers.append(sweeper._prove(T.binop("&", c, T.const(1)), c, False))
    assert answers == ["unsat", "sat"]
    assert len(sweeper.memo) == 2


def test_dangling_vreg_is_an_error():
    p = _Prog()
    b = p.store(p.op("+", p.plus7(), 999))
    with pytest.raises(se.EquivalenceError, match="undefined VReg"):
        _check(b)


def test_value_dead_division_that_may_trap_is_not_proved():
    p = _Prog()
    divisor = p.op("-", p.x, p.const(5))
    b = p.store(p.op("multiply_add", p.op("//", p.x, divisor), p.const(0), p.plus7()))
    result = _check(b)
    assert result.verdict == "unproved"
    assert any("may be 0" in u for u in result.unproved)


def test_value_dead_load_outside_memory_is_not_proved():
    p = _Prog()
    wild = p.op("load", p.op("+", p.vp, p.x))
    b = p.store(p.op("multiply_add", wild, p.const(0), p.plus7()))
    result = _check(b)
    assert result.verdict == "unproved"
    assert any("outside memory" in u for u in result.unproved)


def test_word_stored_twice_is_an_error():
    p = _Prog()
    r = p.plus7()
    p.store(r)
    b = p.store(r)
    with pytest.raises(se.EquivalenceError, match="more than once"):
        _check(b)


def test_non_constant_store_address_is_an_error():
    p = _Prog()
    b = p.store(p.plus7(), addr=p.op("+", p.vp, p.op("&", p.x, p.const(1))))
    with pytest.raises(se.EquivalenceError, match="non-constant address"):
        _check(b)


def test_shape_must_match_recorded_run_options():
    a = dict(_a(), run_options={"rounds": 2, "batch_size": 8, "forest_height": 2})
    with pytest.raises(se.EquivalenceError, match="disagree"):
        se.check_equivalence(a, _a(), shape=(2, 16, 2))
    b = dict(_a(), run_options={"rounds": 2, "batch_size": 16, "forest_height": 2})
    with pytest.raises(se.EquivalenceError, match="disagree"):
        se.check_equivalence(a, b)
    assert se.check_equivalence(a, _a()).verdict == "proved"


# =============================================================================
# Malformed programs are input errors or not proved, never PROVED (pre-merge review)
# =============================================================================


def _redefine(prog, vid, opcode, srcs, cycle=0):
    """Append an op that defines v`vid` again."""
    prog.ops.append({"id": len(prog.ops), "opcode": opcode, "dest": f"v{vid}_s", "cycle": cycle,
                     "source_info": {}, "srcs": [{"VReg": {"id": s, "width": "scalar"}} for s in srcs]})


def test_two_live_definitions_of_one_vreg_are_an_error():
    # B: r = x + 7, then a second live r = x + 8; the store reads r.
    p = _Prog()
    r = p.plus7()
    _redefine(p, r, "+", [p.x, p.const(8)])
    b = p.store(r)
    with pytest.raises(se.EquivalenceError, match="not single assignment"):
        _check(b)


def test_a_dead_definition_is_not_part_of_the_program():
    # B: a dead (cycle -1) r = x + 7, then the live r = x + 8: B stores x + 8, not A's x + 7.
    p = _Prog()
    c7, c8 = p.const(7), p.const(8)
    r = p.op("+", p.x, c7)
    p.ops[-1]["cycle"] = -1
    _redefine(p, r, "+", [p.x, c8])
    b = p.store(r)
    assert _check(b).verdict == "counterexample"


def test_a_read_of_a_dead_only_definition_is_an_error():
    # B's live op reads a constant whose only definition is dead.
    p = _Prog()
    c7 = p.const(7)
    p.ops[-1]["cycle"] = -1
    b = p.store(p.op("+", p.x, c7))
    with pytest.raises(se.EquivalenceError, match="undefined VReg"):
        _check(b)


def test_constant_alternatives_with_one_immediate_are_allowed():
    p = _Prog()
    c7 = p.const(7)
    p.ops.append({"id": len(p.ops), "opcode": "const", "dest": f"v{c7}_s", "cycle": 0, "source_info": {},
                  "srcs": [{"Imm": 7}]})
    b = p.store(p.op("+", p.x, c7))
    assert _check(b).verdict == "proved"


def _with_store_outside_memory(p):
    r = p.plus7()
    p.store(r)
    return p.store(r, addr=p.const(TINY_LAYOUT.mem_size + 5))


def test_a_store_outside_memory_is_a_trap():
    b = _with_store_outside_memory(_Prog())
    for result in (se.check_equivalence(_a(), b, values_only=True, shape=TINY),  # B only, values only
                   se.check_equivalence(_a(), b, shape=TINY),                    # B only, all words
                   se.check_equivalence(b, b, shape=TINY)):                      # both
        assert result.verdict == "unproved", se.format_result(result)
        assert any("outside memory" in u for u in result.unproved)


@pytest.mark.parametrize("opcode", ["halt", "jump", "cond_jump", "pause"])
def test_dest_less_control_flow_is_an_error(opcode):
    p = _Prog()
    r = p.plus7()
    p._add(opcode, [], dest=False)
    b = p.store(r)
    with pytest.raises(se.EquivalenceError, match="no destination and is not a store"):
        _check(b)


def test_nothing_to_compare_is_an_error():
    empty = {"ssa_ops": [], "vector_group_lanes": {}}
    with pytest.raises(se.EquivalenceError, match="no word would be compared"):
        se.check_equivalence(empty, empty, shape=TINY)


# =============================================================================
# main(): exit codes 0 proved, 1 counterexample, 2 not proved, 3 input error / checker error
# =============================================================================


def _main(tmp_path, a, b, *extra):
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(json.dumps(a))
    pb.write_text(json.dumps(b))
    return se.main([str(pa), str(pb), "--shape", *map(str, TINY), *extra])


def test_main_exit_codes(tmp_path):
    assert _main(tmp_path, _a(), _a()) == 0
    p = _Prog()
    assert _main(tmp_path, _a(), p.store(p.op("+", p.x, p.const(8)))) == 1
    p = _Prog()
    rare = p.store(p.op("select", p.op("==", p.x, p.const(0x12345)), p.op("+", p.x, p.const(8)), p.plus7()))
    assert _main(tmp_path, _a(), rare) == 2
    p = _Prog()
    r = p.plus7()
    _redefine(p, r, "+", [p.x, p.const(8)])
    assert _main(tmp_path, _a(), p.store(r)) == 3                      # EquivalenceError
    p = _Prog()
    assert _main(tmp_path, _a(), p.store(p.op("frob", p.x))) == 3      # any other exception
    empty = {"ssa_ops": [], "vector_group_lanes": {}}
    assert _main(tmp_path, empty, empty) == 3


# =============================================================================
# Inequivalent pairs for the remaining opcodes
# =============================================================================


@pytest.mark.parametrize("opcode,k_a,k_b", [("<", 1 << 29, 1 << 28), (">>", 3, 4), ("<<", 3, 4),
                                            ("//", 3, 5), ("%", 3, 5), ("cdiv", 3, 5)])
def test_inequivalent_pairs_for_each_opcode(opcode, k_a, k_b):
    def prog(k):
        p = _Prog()
        return p.store(p.op(opcode, p.x, p.const(k)))
    assert se.check_equivalence(prog(k_a), prog(k_a), shape=TINY).verdict == "proved"
    result = se.check_equivalence(prog(k_a), prog(k_b), shape=TINY)
    assert result.verdict == "counterexample", se.format_result(result)


# =============================================================================
# Real-emission mutants: lanes of vector groups, multiply_add / Mobius coefficients
# =============================================================================


def _groups(sched, read):
    """Vector groups of `sched`: read groups (the key has a definer) or write groups."""
    defined = {op["dest"].split("_")[0][1:] for op in sched["ssa_ops"] if op.get("dest")}
    return sorted((k for k in sched["vector_group_lanes"] if (k in defined) == read), key=int)


@pytest.mark.parametrize("read", [True, False], ids=["read-group (gather address lanes)", "write-group (gathered lanes)"])
def test_swapped_lanes_of_a_vector_group_are_rejected(default, read):
    mutant = copy.deepcopy(default[1])
    key = _groups(mutant, read)[0]
    lanes = mutant["vector_group_lanes"][key]
    lanes[0], lanes[1] = lanes[1], lanes[0]
    result = se.check_equivalence(default[1], mutant, shape=SHAPE)
    assert result.verdict == "counterexample", se.format_result(result)


def _swap_constant_operand(sched, pick):
    """Replace operand 1 (a broadcast constant) of the first multiply_add `pick` accepts by
    another broadcast constant vector of the program."""
    mutant = copy.deepcopy(sched)
    ops = mutant["ssa_ops"]
    consts = {op["dest"]: op for op in ops if op["opcode"] == "vbroadcast"}
    vecs = [op["dest"] for op in ops if op["opcode"] == "vbroadcast"]
    for op in ops:
        if op["opcode"] == "multiply_add" and pick(op):
            cur = op["srcs"][1]["VReg"]["id"]
            other = next(d for d in vecs if int(d[1:].split("_")[0]) != cur)
            op["srcs"][1] = {"VReg": {"id": int(other[1:].split("_")[0]), "width": "vector"}}
            assert other in consts
            return mutant
    raise AssertionError("no multiply_add to mutate")


def test_wrong_multiply_add_coefficient_is_rejected(default):
    hash_fma = lambda op: (op.get("source_info") or {}).get("phase") == "hash"  # noqa: E731
    result = se.check_equivalence(default[1], _swap_constant_operand(default[1], hash_fma), shape=SHAPE)
    assert result.verdict == "counterexample", se.format_result(result)


def test_wrong_mobius_coefficient_is_rejected(tmp_path_factory):
    _, mobius = _build(tmp_path_factory, "mobius-mutant", cached_select="mobius")
    tree_fma = lambda op: (op.get("source_info") or {}).get("phase") == "tree_access"  # noqa: E731
    result = se.check_equivalence(mobius, _swap_constant_operand(mobius, tree_fma), shape=SHAPE)
    assert result.verdict == "counterexample", se.format_result(result)
