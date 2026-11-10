"""Tests for verification.kernel_dataflow (symbolic kernel-vs-SSA replay).

Clean kernels must be proved; each kind of corruption that has caused a real validation
failure (see BUGS.md) must be caught at, or at the first use after, the corrupted cycle.
"""

import copy
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scheduler_rs  # noqa: E402
from verification.kernel_dataflow import Terms, check, explain_cycle  # noqa: E402


def _build(tmp_path_factory, name, **kwargs):
    path = tmp_path_factory.mktemp(name) / "sched.json"
    args = dict(forest_height=10, n_nodes=2047, batch_size=16, rounds=5, enable_valu_shift=True,
                verbose=False, use_cpsat=False, output_json=str(path))
    args.update(kwargs)
    instrs, _ = scheduler_rs.build_tick_tock_kernel_ssa(**args)
    with open(path) as f:
        sched = json.load(f)
    return {"instrs": json.loads(json.dumps(list(instrs)))}, sched


@pytest.fixture(scope="module")
def greedy(tmp_path_factory):
    """5 rounds: covers cached levels, tree-reduce selects and one gather round."""
    return _build(tmp_path_factory, "greedy")


def _first_slot(instrs, engine, start, pred=lambda slot: True):
    for cycle in range(start, len(instrs)):
        for i, slot in enumerate(instrs[cycle].get(engine, [])):
            if pred(slot):
                return cycle, i
    raise AssertionError(f"no {engine} slot from cycle {start}")


def _mutated(kernel, mutate):
    mutant = copy.deepcopy(kernel)
    cycle = mutate(mutant["instrs"])
    return mutant, cycle


def test_clean_greedy_kernel_is_proved(greedy):
    kernel, sched = greedy
    report = check(kernel, sched)
    assert report.ok, report.findings[:3]
    assert report.ssa_values > 1000


def test_clean_cpsat_kernel_is_proved(tmp_path_factory):
    """CP-SAT exercises the alternative encodings (shifted ops, FMA selects, add_imm)."""
    kernel, sched = _build(tmp_path_factory, "cpsat", rounds=2, use_cpsat=True, cpsat_time_limit=20.0)
    report = check(kernel, sched)
    assert report.ok, report.findings[:3]


def test_op_in_next_cycles_bundle_is_caught(greedy):
    """BUGS.md: an op scheduled at cycle N but emitted in bundle N+1."""
    kernel, sched = greedy

    def mutate(instrs):
        cycle, i = _first_slot(instrs, "valu", len(instrs) // 2)
        instrs[cycle + 1].setdefault("valu", []).append(instrs[cycle]["valu"].pop(i))
        return cycle

    mutant, cycle = _mutated(kernel, mutate)
    report = check(mutant, sched)
    assert not report.ok
    assert cycle <= report.findings[0].cycle <= cycle + 2


def test_wrong_operand_address_is_caught_at_that_cycle(greedy):
    kernel, sched = greedy

    def mutate(instrs):
        cycle, i = _first_slot(instrs, "valu", len(instrs) // 2, lambda s: s[0] == "^")
        instrs[cycle]["valu"][i][3] += 8
        return cycle

    mutant, cycle = _mutated(kernel, mutate)
    report = check(mutant, sched)
    assert not report.ok
    first = report.findings[0]
    assert first.cycle == cycle and first.kind == "foreign-value"
    assert "operand 2" in first.message


def test_premature_overwrite_is_caught(greedy):
    """BUGS.md: a live value's address is reused before its last read."""
    kernel, sched = greedy

    def mutate(instrs):
        cycle, i = _first_slot(instrs, "valu", len(instrs) // 2, lambda s: s[0] == "^")
        instrs[cycle]["valu"][i][1] += 8  # write lands on a neighbouring vector
        return cycle

    mutant, cycle = _mutated(kernel, mutate)
    report = check(mutant, sched)
    assert not report.ok
    assert report.findings[0].cycle >= cycle


def test_dropped_op_is_caught(greedy):
    kernel, sched = greedy

    def mutate(instrs):
        cycle, i = _first_slot(instrs, "valu", len(instrs) // 3)
        instrs[cycle]["valu"].pop(i)
        return cycle

    mutant, cycle = _mutated(kernel, mutate)
    assert not check(mutant, sched).ok


def test_wrong_gather_lane_is_caught(greedy):
    kernel, sched = greedy

    def mutate(instrs):
        cycle, i = _first_slot(instrs, "load", len(instrs) // 2, lambda s: s[0] == "load")
        instrs[cycle]["load"][i][2] += 1
        return cycle

    mutant, _ = _mutated(kernel, mutate)
    assert not check(mutant, sched).ok


def test_wrong_constant_is_caught(greedy):
    kernel, sched = greedy

    def mutate(instrs):
        cycle, i = _first_slot(instrs, "load", 0, lambda s: s[0] == "const")
        instrs[cycle]["load"][i][2] += 1
        return cycle

    mutant, _ = _mutated(kernel, mutate)
    assert not check(mutant, sched).ok


def test_same_cycle_write_conflict_is_reported(greedy):
    kernel, sched = greedy

    def mutate(instrs):
        cycle, _ = _first_slot(instrs, "valu", len(instrs) // 2)
        a, b = instrs[cycle]["valu"][0], instrs[cycle]["valu"][-1]
        if a is b:
            instrs[cycle]["valu"].append(list(a))
        else:
            b[1] = a[1]
        return cycle

    mutant, cycle = _mutated(kernel, mutate)
    report = check(mutant, sched)
    assert any(f.kind == "write-conflict" and f.cycle == cycle for f in report.findings)


def test_explain_cycle_names_ssa_values(greedy):
    kernel, sched = greedy
    lines = explain_cycle(kernel, sched, len(kernel["instrs"]) // 2)
    assert any("written at cycle" in line for line in lines)
    assert any("(op " in line for line in lines), "operands should be named by SSA op"


def test_fma_select_lowering_normalises_to_select():
    terms = Terms()
    c = terms.binop("==", terms.mem(terms.const(1)), terms.const(0))
    t, f = terms.mem(terms.const(2)), terms.mem(terms.const(3))
    assert terms.fma(c, terms.binop("-", t, f), f) == terms.select(c, t, f)
    assert terms.fma(terms.binop("-", t, f), c, f) == terms.select(c, t, f)
    assert not terms.unproven_bool_conds
    # A non-boolean multiplicand is still normalised, but recorded for the report.
    terms.fma(t, terms.binop("-", t, f), f)
    assert terms.unproven_bool_conds


def test_constants_fold_with_machine_semantics():
    terms = Terms()
    assert terms.binop("-", terms.const(1), terms.const(2)) == terms.const(2**32 - 1)
    assert terms.binop("+", terms.const(5), terms.const(0)) == terms.const(5)
    assert terms.fma(terms.const(3), terms.const(4), terms.const(5)) == terms.const(17)


def test_delta_spelling_of_select_true_value_is_one_term():
    """Emission builds t as f + (t - f); the kernel's FMA select recovers t directly."""
    terms = Terms()
    t, f = terms.mem(terms.const(2)), terms.mem(terms.const(3))
    delta = terms.binop("-", t, f)
    assert terms.binop("+", f, delta) == t
    assert terms.binop("+", delta, f) == t
    assert terms.binop("-", terms.binop("+", t, f), f) == t


def test_single_lane_corruption_names_that_lane(greedy):
    """A scalar write landing inside a live vector corrupts one lane; the report must name it."""
    kernel, sched = greedy
    mutant = copy.deepcopy(kernel)
    instrs = mutant["instrs"]
    cycle, i = _first_slot(instrs, "valu", len(instrs) // 2, lambda s: s[0] == "^")
    src = instrs[cycle]["valu"][i][2]
    # One cycle earlier, clobber lane 5 of the first operand with a constant.
    instrs[cycle - 1].setdefault("load", []).append(["const", src + 5, 12345])
    report = check(mutant, sched)
    first = next(f for f in report.findings if f.kind == "foreign-value")
    assert first.cycle == cycle
    assert "lane 5" in first.message
    assert f"s[{src + 5}]" in first.message and "0x3039" in first.message
