"""Z3 proofs of the algebraic rewrites the kernel emission and the dataflow checker rely on.

Every check encodes a *violation* of the identity over 32-bit bit-vectors (the machine
reduces every result modulo 2**32), so UNSAT means the identity holds for all inputs.
Checks named ``*-needs-*`` are the opposite: they drop a precondition and expect SAT, to
record that the precondition is necessary.

Constants come from ``problem.HASH_STAGES``, so the proofs track the real kernel.

Run:
    uv run --extra verify python -m verification.verify -t R1-hash-fma-fusion
    uv run --extra verify python -m verification.rewrites
"""

from __future__ import annotations

import os
import sys

from z3 import And, BitVec, BitVecVal, If, Int, LShR, Not, Or, Solver, URem

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from problem import HASH_STAGES  # noqa: E402

WIDTH = 32
# The constant XORed in by the last hash stage; emission defers it into the next round.
STAGE5_XOR_CONST = HASH_STAGES[5][1]


def _bv(name: str):
    return BitVec(name, WIDTH)


def _const(value: int):
    return BitVecVal(value, WIDTH)


def _stats(solver) -> dict:
    try:
        s = solver.statistics()
        return {str(k): s.get_key_value(k) for k in s.keys()}
    except Exception:
        return {}


def _refute(name: str, *violations, ctx=None) -> tuple[str, str, float, dict]:
    """Check that none of `violations` is satisfiable. Returns the first non-UNSAT result."""
    stats: dict = {}
    for violation in violations:
        solver = Solver(ctx=ctx)
        solver.add(violation)
        result = str(solver.check())
        stats = _stats(solver)
        if result != "unsat":
            return (name, result, 0.0, stats)
    return (name, "unsat", 0.0, stats)


def _reference_stage(a, stage: int):
    """One stage of problem.myhash on a bit-vector."""
    op1, val1, op2, op3, val3 = HASH_STAGES[stage]
    left = a + _const(val1) if op1 == "+" else a ^ _const(val1)
    right = a << val3 if op3 == "<<" else LShR(a, val3)
    return left + right if op2 == "+" else left ^ right


# =============================================================================
# Hash
# =============================================================================


def fma_multiplier(stage: int) -> int | None:
    """Multiplier m such that stage(a) = a * m + c, or None if the stage does not fuse."""
    op1, _, op2, op3, shift = HASH_STAGES[stage]
    if (op1, op2, op3) == ("+", "+", "<<"):
        return 1 + (1 << shift)
    return None


def check_hash_fma_fusion(ctx=None) -> tuple[str, str, float, dict]:
    """R1: stages of the form (a + c) + (a << k) equal multiply_add(a, 1 + 2^k, c).

    Emission uses one multiply_add for stages 0, 2 and 4 instead of three vector ops.
    """
    a = _bv("a")
    violations = []
    for stage in range(len(HASH_STAGES)):
        mult = fma_multiplier(stage)
        if mult is not None:
            fused = a * _const(mult) + _const(HASH_STAGES[stage][1])
            violations.append(fused != _reference_stage(a, stage))
    assert len(violations) == 3, "expected stages 0, 2, 4 to fuse"
    return _refute("R1-hash-fma-fusion", *violations, ctx=ctx)


def merged_stage_constants(first: int) -> tuple[int, int, int, int] | None:
    """Constants (m1, c1, m2, c2) with stage[first+1](stage[first](a)) = (a*m1 + c1) ^ (a*m2 + c2).

    Applies when stage `first` fuses to a*m + c (see fma_multiplier) and the next stage has
    the form (x + d) ^ (x << k): then m1 = m, c1 = c + d, m2 = m << k, c2 = c << k, all
    modulo 2^32. Returns None for any other pair of stages. Emission uses the pair (2, 3)
    under --merge-hash-stages (config.rs merged_hash_stage_constants).
    """
    if first + 1 >= len(HASH_STAGES):
        return None
    mult = fma_multiplier(first)
    op1, d, op2, op3, k = HASH_STAGES[first + 1]
    if mult is None or (op1, op2, op3) != ("+", "^", "<<"):
        return None
    c = HASH_STAGES[first][1]
    mask = (1 << WIDTH) - 1
    return (mult, (c + d) & mask, (mult << k) & mask, (c << k) & mask)


def check_hash_stage_merge(ctx=None) -> tuple[str, str, float, dict]:
    """R8: stages 2 and 3 together equal (a*33 + C2+C3) ^ (a*16896 + (C2 << 9)).

    Three ops (two multiply_adds and an XOR) instead of four (multiply_add, add, shift,
    XOR). Stage 3 is (x + C3) ^ (x << 9) with x = 33a + C2; the left shift distributes over
    the multiply_add modulo 2^32, so x << 9 = a*(33 << 9) + (C2 << 9).
    """
    a = _bv("a")
    consts = merged_stage_constants(2)
    assert consts is not None, "expected stages 2 and 3 to merge"
    m1, c1, m2, c2 = consts
    merged = (a * _const(m1) + _const(c1)) ^ (a * _const(m2) + _const(c2))
    return _refute("R8-hash-stage-merge", merged != _reference_stage(_reference_stage(a, 2), 3), ctx=ctx)


def check_hash_stage_merge_needs_left_shift(ctx=None) -> tuple[str, str, float, dict]:
    """R8-needs-left-shift: the same merge is wrong when the second stage shifts right (SAT).

    A right shift does not distribute over a*m + c modulo 2^32, which is why stages 0+1 and
    4+5 (both followed by a right-shift stage) are not merged.
    """
    a = _bv("a")
    m, c = _const(fma_multiplier(0)), _const(HASH_STAGES[0][1])
    k = HASH_STAGES[1][4]
    merged = (a * m + c + _const(HASH_STAGES[1][1])) ^ (a * LShR(m, k) + LShR(c, k))
    reference = (a * m + c + _const(HASH_STAGES[1][1])) ^ LShR(a * m + c, k)
    solver = Solver(ctx=ctx)
    solver.add(merged != reference)
    return ("R8-hash-stage-merge-needs-left-shift", str(solver.check()), 0.0, _stats(solver))


def check_stage5_xor_deferral(ctx=None) -> tuple[str, str, float, dict]:
    """R2: the stage-5 constant can be left out and folded into the next round's node XOR.

    The kernel keeps values in "optimised form" v' = v ^ C. Stage 5 becomes a ^ (a >> 16)
    (two ops, not three), tree values are pre-XORed with C, and C is applied once at the
    end. Three obligations: the stage itself, the next round's XOR, and the entry/exit
    conversions.
    """
    a, node, v = _bv("a"), _bv("node"), _bv("v")
    c = _const(STAGE5_XOR_CONST)
    kernel_stage5 = a ^ LShR(a, HASH_STAGES[5][4])
    return _refute(
        "R2-stage5-xor-deferral",
        # Optimised-form output differs from the reference output by exactly C.
        kernel_stage5 ^ c != _reference_stage(a, 5),
        # Next round: v' ^ (node ^ C) is the true v ^ node.
        (v ^ c) ^ (node ^ c) != v ^ node,
        # Exit: one final XOR with C restores the true value.
        (v ^ c) ^ c != v,
        ctx=ctx,
    )


def check_branch_offset_on_optimised_value(ctx=None) -> tuple[str, str, float, dict]:
    """R3: idx*2 + (2 - (v' & 1)) is the reference update when v' = v ^ C and C is odd.

    The reference takes child 1 when the hashed value is even and child 2 when odd. The
    kernel tests the optimised-form value, whose parity is flipped because C is odd, and
    compensates by subtracting the bit from 2.
    """
    v, idx = _bv("v"), _bv("idx")
    v_opt = v ^ _const(STAGE5_XOR_CONST)
    kernel = idx * _const(2) + (_const(2) - (v_opt & _const(1)))
    reference = _const(2) * idx + If(URem(v, _const(2)) == _const(0), _const(1), _const(2))
    return _refute("R3-branch-offset", kernel != reference, ctx=ctx)


def check_branch_offset_needs_odd_constant(ctx=None) -> tuple[str, str, float, dict]:
    """R3-needs-odd: with an even deferred constant the same code picks the wrong child (SAT)."""
    v, idx, c = _bv("v"), _bv("idx"), _bv("c")
    kernel = idx * _const(2) + (_const(2) - ((v ^ c) & _const(1)))
    reference = _const(2) * idx + If(URem(v, _const(2)) == _const(0), _const(1), _const(2))
    solver = Solver(ctx=ctx)
    solver.add(c & _const(1) == _const(0), kernel != reference)
    return ("R3-branch-offset-needs-odd-constant", str(solver.check()), 0.0, _stats(solver))


# =============================================================================
# Select
# =============================================================================


def check_fma_select(ctx=None) -> tuple[str, str, float, dict]:
    """R4: cond * (t - f) + f equals select(cond, t, f) when cond is 0 or 1.

    This is the SelectChoice FMA lowering, and the normalisation the dataflow checker
    applies (verification/kernel_dataflow.py, Terms.fma).
    """
    cond, t, f = _bv("cond"), _bv("t"), _bv("f")
    fma = cond * (t - f) + f
    select = If(cond != _const(0), t, f)
    return _refute("R4-fma-select", And(Or(cond == _const(0), cond == _const(1)), fma != select), ctx=ctx)


def check_fma_select_needs_boolean(ctx=None) -> tuple[str, str, float, dict]:
    """R4-needs-boolean: for a condition outside {0, 1} the lowering is wrong (SAT)."""
    cond, t, f = _bv("cond"), _bv("t"), _bv("f")
    solver = Solver(ctx=ctx)
    solver.add(cond * (t - f) + f != If(cond != _const(0), t, f))
    return ("R4-fma-select-needs-boolean", str(solver.check()), 0.0, _stats(solver))


def check_delta_roundtrip(ctx=None) -> tuple[str, str, float, dict]:
    """R5: (x - y) + y = x and (x + y) - y = x modulo 2^32.

    Emission precomputes select deltas as t - f and rebuilds t as f + delta; the dataflow
    checker treats both spellings as one term (Terms.binop).
    """
    x, y = _bv("x"), _bv("y")
    return _refute("R5-delta-roundtrip", (x - y) + y != x, (x + y) - y != x, ctx=ctx)


def check_comparison_is_boolean(ctx=None) -> tuple[str, str, float, dict]:
    """R6: the forms the checker accepts as 0/1 conditions really are 0 or 1.

    Mirrors Terms.is_bool: a comparison result, a value ANDed with 1, and a select between
    two 0/1 values.
    """
    x, y, c = _bv("x"), _bv("y"), _bv("c")
    eq = If(x == y, _const(1), _const(0))
    bit = x & _const(1)
    sel = If(c != _const(0), eq, bit)

    def not_bool(e):
        return And(e != _const(0), e != _const(1))

    return _refute("R6-boolean-forms", not_bool(eq), not_bool(bit), not_bool(sel), ctx=ctx)


# =============================================================================
# Index range (bounds-check elimination and the wrap round)
# =============================================================================


def check_index_range_by_depth(ctx=None) -> tuple[str, str, float, dict]:
    """R7: at tree depth d the index lies in [2^d - 1, 2^(d+1) - 2], so it never needs a
    bounds check before the last level and always wraps after it.

    Induction over depth with p = 2^d as an unbounded integer: if idx is in [p - 1, 2p - 2]
    then both children 2*idx + 1 and 2*idx + 2 are in [2p - 1, 4p - 2], which is the range
    for depth d + 1. A complete tree of height h has 2^(h+1) - 1 nodes, so the range for
    depth h + 1 starts exactly at n_nodes: every index wraps to 0 there and none earlier.
    """
    p, idx, child, n_nodes = Int("p"), Int("idx"), Int("child"), Int("n_nodes")
    in_range = And(p >= 1, idx >= p - 1, idx <= 2 * p - 2)
    is_child = Or(child == 2 * idx + 1, child == 2 * idx + 2)
    return _refute(
        "R7-index-range-by-depth",
        # Base: depth 0 (p = 1) is exactly index 0.
        And(p == 1, idx >= p - 1, idx <= 2 * p - 2, idx != 0),
        # Step: children land in the next depth's range.
        And(in_range, is_child, Not(And(child >= 2 * p - 1, child <= 4 * p - 2))),
        # Below the last level (2p - 1 <= n_nodes means depth d exists): no index is out of bounds.
        And(in_range, 2 * p - 1 <= n_nodes, idx >= n_nodes),
        # Leaving the last level (n_nodes = 2p - 1): every child is out of bounds, so wraps.
        And(in_range, n_nodes == 2 * p - 1, is_child, child < n_nodes),
        ctx=ctx,
    )


# =============================================================================
# Mirrored index (--index-form mirrored)
# =============================================================================
#
# The kernel may track n = L_d - idx instead of idx, where L_d = 2^(d+1) - 2 is the last
# index of tree level d: n is the node's position in its level counted from the right.


def level_last_index(depth: int) -> int:
    """L_d = 2^(d+1) - 2, the largest index at tree depth d (config.rs level_last_index)."""
    return (1 << (depth + 1)) - 2


def _reference_next_index(idx, v):
    """The reference update before the wrap: child 1 when the hashed value is even, else 2."""
    return _const(2) * idx + If(URem(v, _const(2)) == _const(0), _const(1), _const(2))


def check_mirrored_index_step(ctx=None) -> tuple[str, str, float, dict]:
    """R9: with n = L - idx, the reference update becomes n' = 2n + (v' & 1), L' = 2L + 2.

    v' = v ^ C is the optimised-form value and C is odd, so v' & 1 is 1 exactly when the
    reference takes child 1. Mirroring the index absorbs that inversion: two ops (AND,
    multiply_add) instead of three (AND, SUB, multiply_add; R3). Holds for every L, so in
    particular for L_d = 2^(d+1) - 2, whose successor 2 L_d + 2 is L_(d+1). At depth 0
    (n = 0) the update is the bit itself, so that round needs no multiply_add.
    """
    v, idx, level = _bv("v"), _bv("idx"), _bv("L")
    bit = (v ^ _const(STAGE5_XOR_CONST)) & _const(1)
    n = level - idx
    return _refute(
        "R9-mirrored-index-step",
        (_const(2) * level + _const(2)) - _reference_next_index(idx, v) != _const(2) * n + bit,
        # Depth 0: idx = 0 and L = 0, so n = 0 and the next n is the bit.
        (_const(2) - _reference_next_index(_const(0), v)) != bit,
        ctx=ctx,
    )


def check_mirrored_index_step_needs_odd_constant(ctx=None) -> tuple[str, str, float, dict]:
    """R9-needs-odd: with an even deferred constant the mirrored update is wrong (SAT)."""
    v, idx, level, c = _bv("v"), _bv("idx"), _bv("L"), _bv("c")
    bit = (v ^ c) & _const(1)
    solver = Solver(ctx=ctx)
    solver.add(
        c & _const(1) == _const(0),
        (_const(2) * level + _const(2)) - _reference_next_index(idx, v) != _const(2) * (level - idx) + bit,
    )
    return ("R9-mirrored-index-step-needs-odd-constant", str(solver.check()), 0.0, _stats(solver))


def check_mirrored_index_uses(ctx=None) -> tuple[str, str, float, dict]:
    """R10: every use of idx can be written with n = L - idx at the same op count.

    Gather address: forest_p + idx = (forest_p + L) - n, one SUB against a per-level constant.
    Cached-level compare: idx == k iff n == L - k. Position from the left end of the level:
    idx - c = c - n when L = 2c (c = 2^d - 1, the first index of the level). Final index
    store: idx = L - n.
    """
    idx, level, p, k, c = _bv("idx"), _bv("L"), _bv("p"), _bv("k"), _bv("c")
    n = level - idx
    return _refute(
        "R10-mirrored-index-uses",
        (p + level) - n != p + idx,
        (idx == k) != (n == level - k),
        And(level == _const(2) * c, idx - c != c - n),
        level - n != idx,
        ctx=ctx,
    )


def check_mirrored_index_range(ctx=None) -> tuple[str, str, float, dict]:
    """R11: at depth d, n lies in [0, 2^d - 1] and L_d - n is the index range of R7.

    Induction with p = 2^d as an unbounded integer and L = 2p - 2. So n is 0 at depth 0
    (and again after the wrap), 0 or 1 at depth 1, and the level constants never overflow.
    """
    p, n, bit, level = Int("p"), Int("n"), Int("bit"), Int("L")
    in_range = And(p >= 1, n >= 0, n <= p - 1, level == 2 * p - 2)
    return _refute(
        "R11-mirrored-index-range",
        # Base: depth 0 (p = 1) is n = 0, idx = L - n = 0.
        And(p == 1, n >= 0, n <= p - 1, level == 2 * p - 2, Or(n != 0, level - n != 0)),
        # Step: 2n + bit stays in the next level's range.
        And(in_range, Or(bit == 0, bit == 1), Not(And(2 * n + bit >= 0, 2 * n + bit <= 2 * p - 1))),
        # The level constant follows L' = 2L + 2.
        And(in_range, 2 * level + 2 != 2 * (2 * p) - 2),
        # idx = L - n covers exactly [p - 1, 2p - 2], the range R7 proves for depth d.
        And(in_range, Not(And(level - n >= p - 1, level - n <= 2 * p - 2))),
        ctx=ctx,
    )


REWRITE_CHECKS = [
    ("R1-hash-fma-fusion", "Hash stages 0/2/4 equal one multiply_add", check_hash_fma_fusion),
    ("R2-stage5-xor-deferral", "Stage-5 constant deferred into the next round's node XOR", check_stage5_xor_deferral),
    ("R3-branch-offset", "Branch on the optimised-form value picks the reference child", check_branch_offset_on_optimised_value),
    ("R3-branch-offset-needs-odd-constant", "Branch trick fails for an even deferred constant (SAT)", check_branch_offset_needs_odd_constant),
    ("R4-fma-select", "cond*(t-f)+f equals select for 0/1 cond", check_fma_select),
    ("R4-fma-select-needs-boolean", "FMA select is wrong for non-0/1 cond (SAT)", check_fma_select_needs_boolean),
    ("R5-delta-roundtrip", "(x-y)+y = x and (x+y)-y = x", check_delta_roundtrip),
    ("R6-boolean-forms", "Forms the dataflow checker accepts as 0/1 are 0/1", check_comparison_is_boolean),
    ("R7-index-range-by-depth", "Index range per depth: no bounds check before the last level, always wrap after", check_index_range_by_depth),
    ("R8-hash-stage-merge", "Hash stages 2+3 equal two multiply_adds and an XOR", check_hash_stage_merge),
    ("R8-hash-stage-merge-needs-left-shift", "Stage merge fails across a right shift (SAT)", check_hash_stage_merge_needs_left_shift),
    ("R9-mirrored-index-step", "Mirrored index n = L - idx updates as 2n + (v' & 1)", check_mirrored_index_step),
    ("R9-mirrored-index-step-needs-odd-constant", "Mirrored update fails for an even deferred constant (SAT)", check_mirrored_index_step_needs_odd_constant),
    ("R10-mirrored-index-uses", "Gather address, compares and the final store written with the mirrored index", check_mirrored_index_uses),
    ("R11-mirrored-index-range", "Mirrored index range per depth and the level constants", check_mirrored_index_range),
]

# Checks that are expected to find a counterexample.
EXPECT_SAT = {
    "R3-branch-offset-needs-odd-constant",
    "R4-fma-select-needs-boolean",
    "R8-hash-stage-merge-needs-left-shift",
    "R9-mirrored-index-step-needs-odd-constant",
}


def main() -> int:
    failed = 0
    for name, desc, fn in REWRITE_CHECKS:
        _, result, _, _ = fn()
        want = "sat" if name in EXPECT_SAT else "unsat"
        ok = result == want
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'} [{name}] {desc}: {result}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
