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

    Proved operand by operand: asked as one query over the whole expression, Z3 needs 25 to
    90 s on a loaded machine (more than `verification.verify`'s 60 s timeout), while the four
    obligations below take milliseconds. Together they give the identity by substitution:

        stage3(stage2(a)) = (x + C3) ^ (x << 9)                      [stage 3, x = stage2(a)]
                          = ((a*m + C2) + C3) ^ ((a*m + C2) << 9)    [stage 2 is a*m + C2, R1]
                          = (a*m1 + c1) ^ (a*m2 + c2)                [left and right operands]
    """
    a, x = _bv("a"), _bv("x")
    consts = merged_stage_constants(2)
    assert consts is not None, "expected stages 2 and 3 to merge"
    m1, c1, m2, c2 = consts
    mult, c = fma_multiplier(2), HASH_STAGES[2][1]
    op1, d, op2, op3, k = HASH_STAGES[3]
    assert (op1, op2, op3) == ("+", "^", "<<"), "stage 3 must be (x + d) ^ (x << k)"
    stage2 = a * _const(mult) + _const(c)
    # Same theorem as the single query stage3(stage2(a)) == merged, split for speed. Sound:
    # the four obligations below together imply it by substitution (the chain in the
    # docstring), so proving them proves the original equality; nothing is weakened.
    return _refute(
        "R8-hash-stage-merge",
        # Stage 3 has the form (x + d) ^ (x << k) for every x.
        _reference_stage(x, 3) != (x + _const(d)) ^ (x << k),
        # Stage 2 is the multiply_add a*m + c (as in R1).
        _reference_stage(a, 2) != stage2,
        # The two operands of the XOR.
        stage2 + _const(d) != a * _const(m1) + _const(c1),
        (stage2 << k) != a * _const(m2) + _const(c2),
        ctx=ctx,
    )


def check_hash_stage_merge_general(ctx=None) -> tuple[str, str, float, dict]:
    """R8-general: for all a, m, c, d and every shift 0 <= k < 32,
    x = a*m + c gives (x + d) ^ (x << k) = (a*m + (c+d)) ^ (a*(m << k) + (c << k)) mod 2^32.

    The identity the `hash-merge` rewrite rule (scheduler_rs transform/hash_merge.rs) applies
    to any site it matches, whatever its constants; R8-hash-stage-merge is its instance for
    hash stages 2 and 3. One obligation per k (the rule rejects k outside 0..32).
    """
    a, m, c, d = _bv("a"), _bv("m"), _bv("c"), _bv("d")
    x = a * m + c
    violations = []
    for k in range(WIDTH):
        kk = _const(k)
        violations.append(((x + d) ^ (x << kk)) != ((a * m + (c + d)) ^ (a * (m << kk) + (c << kk))))
    return _refute("R8-hash-stage-merge-general", *violations, ctx=ctx)


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


# =============================================================================
# Index work with a known result (--prune-dead-index)
# =============================================================================


def check_index_update_from_root(ctx=None) -> tuple[str, str, float, dict]:
    """R12: from idx = 0 the zero-based update idx*2 + (2 - (v' & 1)) is 2 - (v' & 1).

    At depth 0 every index is 0 (R7: before the first round by assumption, and again after
    the wrap), so the multiply_add of the update has a known first operand and the offset
    is the new index. The result is the reference child (R3 with idx = 0).
    """
    v = _bv("v")
    offset = _const(2) - ((v ^ _const(STAGE5_XOR_CONST)) & _const(1))
    return _refute(
        "R12-index-update-from-root",
        _const(0) * _const(2) + offset != offset,
        offset != _reference_next_index(_const(0), v),
        ctx=ctx,
    )


def check_cse_identical_broadcast(ctx=None) -> tuple[str, str, float, dict]:
    """CSE-identical-broadcast: two vbroadcasts of one scalar hold the same 8 lanes.

    The justification of the prune-index rule's `shared-constant` sites (rule 5): a later
    `vbroadcast(s)` is replaced by an earlier one of the same scalar `s`. Plain common
    subexpression elimination of a pure op; R12 says nothing about it. Lane-wise model: lane i
    of vbroadcast(s) is s.
    """
    s = _bv("s")
    first = [s for _ in range(8)]
    second = [s for _ in range(8)]
    return _refute("CSE-identical-broadcast", Or(*[a != b for a, b in zip(first, second, strict=True)]), ctx=ctx)


def check_fma_zero_factor(ctx=None) -> tuple[str, str, float, dict]:
    """R12-fma-zero-factor: multiply_add(0, m, c) = multiply_add(m, 0, c) = c for all m, c.

    The identity the `prune-index` rewrite rule (scheduler_rs transform/prune_index.rs)
    applies to any multiply_add with a factor it shows to be 0; R12-index-update-from-root is
    its instance on the zero-based index update.
    """
    m, c = _bv("m"), _bv("c")
    return _refute(
        "R12-fma-zero-factor",
        _const(0) * m + c != c,
        m * _const(0) + c != c,
        ctx=ctx,
    )


def check_index_update_from_root_needs_zero(ctx=None) -> tuple[str, str, float, dict]:
    """R12-needs-zero: for a nonzero index the offset alone is not the new index (SAT)."""
    v, idx = _bv("v"), _bv("idx")
    offset = _const(2) - ((v ^ _const(STAGE5_XOR_CONST)) & _const(1))
    solver = Solver(ctx=ctx)
    solver.add(idx != _const(0), idx * _const(2) + offset != offset)
    return ("R12-index-update-from-root-needs-zero", str(solver.check()), 0.0, _stats(solver))


# =============================================================================
# Cached-level selection from the branch bits (--cached-select bits|bits-flow|mobius)
# =============================================================================
#
# Mirrors ssa_emit.rs emit_cached_select_init / emit_cached_level_select / emit_select_bits.

MAX_CACHED_DEPTH = 4  # config.rs MAX_TREE_REDUCE_DEPTH


def _bit_vars(depth: int):
    """Branch bits b_0 .. b_(d-1), oldest first, and the constraint that each is 0 or 1."""
    bits = [_bv(f"b{i}") for i in range(depth)]
    return bits, And(*[Or(b == _const(0), b == _const(1)) for b in bits])


def _position(bits):
    """n after the rounds that produced `bits`: n' = 2n + bit from n = 0 (R9)."""
    n = _const(0)
    for bit in bits:
        n = _const(2) * n + bit
    return n


def _reference_node(nodes, depth: int, n):
    """tree[idx] ^ C for idx = L_d - n, as a chain of compares on the reference index."""
    idx = _const(level_last_index(depth)) - n
    first = (1 << depth) - 1
    out = nodes[first]
    for k in range(first + 1, 2 * first + 1):
        out = If(idx == _const(k), nodes[k], out)
    return out ^ _const(STAGE5_XOR_CONST)


def select_table(nodes, depth: int):
    """U[m] = tree[L_d - reverse_d(m)] ^ C: the node table indexed by the bits, oldest lowest."""
    table = []
    for m in range(1 << depth):
        n = 0
        for i in range(depth):
            n = (n << 1) | ((m >> i) & 1)
        table.append(nodes[level_last_index(depth) - n] ^ _const(STAGE5_XOR_CONST))
    return table


def select_tree(table, bits):
    """The select tree: bits[0] selects within leaf pairs, bits[-1] at the root."""
    cur = list(table)
    for bit in bits:
        cur = [If(bit != _const(0), cur[2 * k + 1], cur[2 * k]) for k in range(len(cur) // 2)]
    return cur[0]


def mobius_coefficients(table):
    """a[m] = sum over subsets s of m of (-1)^(|m|-|s|) U[s] (in-place subset transform)."""
    coeff = list(table)
    depth = len(coeff).bit_length() - 1
    for bit in range(depth):
        for m in range(len(coeff)):
            if m & (1 << bit):
                coeff[m] = coeff[m] - coeff[m ^ (1 << bit)]
    return coeff


def mobius_eval(coeff, bits):
    """2^d - 1 multiply_adds: fold the coefficients pairwise, one bit per level."""
    cur = list(coeff)
    for bit in bits:
        cur = [bit * cur[2 * k + 1] + cur[2 * k] for k in range(len(cur) // 2)]
    return cur[0]


def _nodes(depth: int):
    first = (1 << depth) - 1
    return {k: _bv(f"tree{k}") for k in range(first, 2 * first + 1)}


def check_select_by_path_bits(ctx=None) -> tuple[str, str, float, dict]:
    """R13: the select tree over the pre-XORed node table, driven by the branch bits oldest
    first, yields tree[idx] ^ C for the reference index idx = L_d - n (depths 1 to 4).

    n is built from the bits by the mirrored update (R9), so this ties the selection to the
    reference index in either index form. Depth 0 is the root for every lane (R7). Each
    two-way select is a SelectChoice (vselect, or multiply_add by R4).
    """
    violations = []
    for depth in range(1, MAX_CACHED_DEPTH + 1):
        bits, boolean = _bit_vars(depth)
        nodes = _nodes(depth)
        selected = select_tree(select_table(nodes, depth), bits)
        violations.append(And(boolean, selected != _reference_node(nodes, depth, _position(bits))))
    return _refute("R13-select-by-path-bits", *violations, ctx=ctx)


def check_select_by_path_bits_needs_bit_order(ctx=None) -> tuple[str, str, float, dict]:
    """R13-needs-bit-order: feeding the bits newest first into the same table is wrong (SAT)."""
    depth = 2
    bits, boolean = _bit_vars(depth)
    nodes = _nodes(depth)
    selected = select_tree(select_table(nodes, depth), list(reversed(bits)))
    solver = Solver(ctx=ctx)
    solver.add(boolean, selected != _reference_node(nodes, depth, _position(bits)))
    return ("R13-select-by-path-bits-needs-bit-order", str(solver.check()), 0.0, _stats(solver))


def check_mobius_select(ctx=None) -> tuple[str, str, float, dict]:
    """R14: 2^d - 1 multiply_adds over the Mobius coefficients equal the table entry the
    bits address, for every 0/1 assignment of the bits (depths 1 to 4).

    The bits are enumerated (that they are 0 or 1 is the precondition), which leaves an
    identity that is linear in the table entries; together with R13 the result is
    tree[idx] ^ C.
    """
    violations = []
    for depth in range(1, MAX_CACHED_DEPTH + 1):
        table = [_bv(f"u{m}") for m in range(1 << depth)]
        coeff = mobius_coefficients(table)
        for m in range(1 << depth):
            bits = [_const((m >> i) & 1) for i in range(depth)]
            violations.append(mobius_eval(coeff, bits) != table[m])
            violations.append(mobius_eval(coeff, bits) != select_tree(table, bits))
    return _refute("R14-mobius-select", *violations, ctx=ctx)


def check_mobius_select_needs_boolean(ctx=None) -> tuple[str, str, float, dict]:
    """R14-needs-boolean: for a bit outside {0, 1} the multiply_add form is not the select (SAT)."""
    bit = _bv("b0")
    table = [_bv("u0"), _bv("u1")]
    solver = Solver(ctx=ctx)
    solver.add(mobius_eval(mobius_coefficients(table), [bit]) != select_tree(table, [bit]))
    return ("R14-mobius-select-needs-boolean", str(solver.check()), 0.0, _stats(solver))


def extract_bits(n, count: int):
    """The `count` older bits from n = b_0 .. b_(count-1) in binary, as emit_select_bits does:
    AND 1 and shift right for all but the top bit, which is what remains."""
    bits = [None] * count
    cur = n
    for k in range(count - 1, 0, -1):
        bits[k] = cur & _const(1)
        cur = LShR(cur, 1)
    bits[0] = cur
    return bits


def check_bits_from_index(ctx=None) -> tuple[str, str, float, dict]:
    """R15: the older branch bits can be read back from the index (1 to 3 bits).

    With n = b_0 .. b_(k-1) in binary (the mirrored index after k rounds, R9), AND/shift
    extraction returns exactly those bits; the top bit needs no AND because n < 2^k (R11).
    A zero-based index is mirrored first: n = L - idx (R10).
    """
    violations = []
    for count in range(1, MAX_CACHED_DEPTH):
        bits, boolean = _bit_vars(count)
        n = _position(bits)
        extracted = extract_bits(n, count)
        violations.append(And(boolean, Or(*[e != b for e, b in zip(extracted, bits, strict=True)])))
        idx = _const(level_last_index(count)) - n
        from_zero_based = extract_bits(_const(level_last_index(count)) - idx, count)
        violations.append(And(boolean, Or(*[e != b for e, b in zip(from_zero_based, bits, strict=True)])))
    return _refute("R15-bits-from-index", *violations, ctx=ctx)


def check_bits_from_index_needs_range(ctx=None) -> tuple[str, str, float, dict]:
    """R15-needs-range: for n >= 2^k the unmasked top "bit" is not 0 or 1 (SAT)."""
    n = _bv("n")
    top = extract_bits(n, 2)[0]
    solver = Solver(ctx=ctx)
    solver.add(And(top != _const(0), top != _const(1)))
    return ("R15-bits-from-index-needs-range", str(solver.check()), 0.0, _stats(solver))


def check_gather_xor_reassociation(ctx=None) -> tuple[str, str, float, dict]:
    """R16: val ^ (node ^ K) = (val ^ K) ^ node for every 32-bit val, node and the deferred
    stage-5 constant K: the constant can go into the value (ready before the gather returns)
    instead of the loaded node, so one XOR follows the load. Same op count."""
    val, node = _bv("val"), _bv("node")
    k = _const(STAGE5_XOR_CONST)
    return _refute("R16-gather-xor-reassociation", val ^ (node ^ k) != (val ^ k) ^ node, ctx=ctx)


def check_index_early_terms(ctx=None) -> tuple[str, str, float, dict]:
    """R17: a consumer V - n' of the mirrored update n' = 2n + bit equals
    multiply_add(n, -2, V) - bit (mod 2^32), whose multiply_add needs only n: the final index
    store (V = L) and a gather address (V = forest_p + L) keep one op after the newest bit."""
    v, n, bit = _bv("V"), _bv("n"), _bv("bit")
    minus2 = _const(0xFFFFFFFE)
    return _refute("R17-index-early-terms", v - (_const(2) * n + bit) != (n * minus2 + v) - bit, ctx=ctx)


def check_index_early_terms_needs_wrap(ctx=None) -> tuple[str, str, float, dict]:
    """R17-needs-wrap: with -2 replaced by 2 the early form is wrong (SAT): the constant is
    the two's complement of 2, not 2."""
    v, n, bit = _bv("V"), _bv("n"), _bv("bit")
    solver = Solver(ctx=ctx)
    solver.add(v - (_const(2) * n + bit) != (n * _const(2) + v) - bit)
    return ("R17-index-early-terms-needs-wrap", str(solver.check()), 0.0, _stats(solver))


def check_select_xor_pushdown(ctx=None) -> tuple[str, str, float, dict]:
    """R18: val ^ select(c, hi, lo) = select(c, val ^ hi, val ^ lo) for every condition, so
    the round XOR can run on both select operands in parallel with the branch bit."""
    val, hi, lo, c = _bv("val"), _bv("hi"), _bv("lo"), _bv("c")
    cond = c != _const(0)
    return _refute("R18-select-xor-pushdown", val ^ If(cond, hi, lo) != If(cond, val ^ hi, val ^ lo), ctx=ctx)


# =============================================================================
# Normal forms used by the SSA equivalence checker (verification/ssa_equivalence.py)
# =============================================================================
#
# The checker builds terms in normal form: sums of constant multiples (linear forms), XOR
# sets, equality as a difference compared with zero, and selects. Each rule below is one of
# the rewrites its constructors apply; constructors cite these ids.


def check_linear_normal_form(ctx=None) -> tuple[str, str, float, dict]:
    """R21: +, -, multiplication by a constant and shifts left by a constant are linear forms.

    Z/2^32 is a commutative ring, so any expression built from +, - and multiplication by
    constants equals c + sum(k_i * x_i) with one coefficient per distinct atom x_i. A left
    shift by a constant k is multiplication by 2^k (by 0 for k >= 32, as the machine reduces
    modulo 2^32), and multiply_add(a, b, c) with a constant factor is a linear form too.
    """
    x, y, z, k, c1, c2 = (_bv(n) for n in ("x", "y", "z", "k", "c1", "c2"))
    from z3 import UGE, ULT

    return _refute(
        "R21-linear-normal-form",
        And(ULT(k, _const(WIDTH)), (x << k) != x * (_const(1) << k)),
        And(UGE(k, _const(WIDTH)), (x << k) != _const(0)),
        (x + y) + z != x + (y + z),
        x + y != y + x,
        c1 * x + c2 * x != (c1 + c2) * x,
        c1 * (x + y) != c1 * x + c1 * y,
        c1 * (c2 * x) != (c1 * c2) * x,
        x - y != x + _const(2**WIDTH - 1) * y,
        _const(1) * x != x,
        _const(0) * x != _const(0),
        ctx=ctx,
    )


def check_xor_normal_form(ctx=None) -> tuple[str, str, float, dict]:
    """R22: XOR is associative and commutative, x ^ x = 0 and x ^ 0 = x, so any XOR
    expression equals c ^ (XOR of a set of atoms)."""
    x, y, z = _bv("x"), _bv("y"), _bv("z")
    return _refute(
        "R22-xor-normal-form",
        (x ^ y) ^ z != x ^ (y ^ z),
        x ^ y != y ^ x,
        x ^ x != _const(0),
        x ^ _const(0) != x,
        ctx=ctx,
    )


def check_equality_as_difference(ctx=None) -> tuple[str, str, float, dict]:
    """R23: (a == b) iff (a - b == 0) iff (b - a == 0) modulo 2^32, so an equality test is
    normalised to one linear form compared with zero, with a canonical sign."""
    a, b = _bv("a"), _bv("b")
    return _refute(
        "R23-equality-as-difference",
        (a == b) != (a - b == _const(0)),
        (a - b == _const(0)) != (b - a == _const(0)),
        ctx=ctx,
    )


def check_select_forms(ctx=None) -> tuple[str, str, float, dict]:
    """R24: the select forms the normaliser produces.

    select(c, t, t) = t for every c; for cond in {0, 1}, multiply_add(cond, d, f) =
    select(cond, f + d, f) (R4 with t = f + d, R5) and cond * d = select(cond, d, 0).
    """
    c, t, f, d = _bv("c"), _bv("t"), _bv("f"), _bv("d")
    boolean = Or(c == _const(0), c == _const(1))

    def sel(cond, if_true, if_false):
        return If(cond != _const(0), if_true, if_false)

    return _refute(
        "R24-select-forms",
        sel(c, t, t) != t,
        And(boolean, c * d + f != sel(c, f + d, f)),
        And(boolean, c * d != sel(c, d, _const(0))),
        ctx=ctx,
    )


def check_interval_transfer(ctx=None) -> tuple[str, str, float, dict]:
    """R25: the interval rules of the normaliser are sound (unsigned 32-bit).

    Used to recognise 0/1 conditions (R4 and R24 need them), to bound load addresses, and
    as facts on cut points. For an operand x in [lo, hi]:
      * x & y <= min(x, y);  x | y >= max(x, y);
      * x, y < 2^m implies x | y < 2^m and x ^ y < 2^m;
      * x >> k lies in [lo >> k, hi >> k];
      * a comparison is 0 or 1; select(c, t, f) is t or f;
      * a linear form: the 32-bit result equals the integer value of c + sum(k_i x_i)
        modulo 2^32 (addition and multiplication commute with reduction modulo 2^32), so if
        that integer lies in [0, 2^32) the result is that integer.
    Division and remainder get no interval rule (the checker treats them as unbounded):
    with a symbolic divisor Z3 does not decide their monotonicity within a minute.
    """
    from z3 import Extract, If as ZIf, ULE, ULT, ZeroExt

    x, y, lo, hi, k, m = (_bv(n) for n in ("x", "y", "lo", "hi", "k", "m"))
    in_range = And(ULE(lo, x), ULE(x, hi))
    p = _const(1) << m
    xw, yw, kw = ZeroExt(WIDTH, x), ZeroExt(WIDTH, y), ZeroExt(WIDTH, k)
    return _refute(
        "R25-interval-transfer",
        Not(And(ULE(x & y, x), ULE(x & y, y))),
        Not(And(ULE(x, x | y), ULE(y, x | y))),
        And(ULT(m, _const(WIDTH)), ULT(x, p), ULT(y, p), Not(And(ULT(x | y, p), ULT(x ^ y, p)))),
        And(in_range, Not(And(ULE(LShR(lo, k), LShR(x, k)), ULE(LShR(x, k), LShR(hi, k))))),
        Not(ULE(ZIf(ULT(x, y), _const(1), _const(0)), _const(1))),
        Not(ULE(ZIf(x == y, _const(1), _const(0)), _const(1))),
        And(in_range, ULE(lo, y), ULE(y, hi), Not(And(ULE(lo, ZIf(k != _const(0), x, y)),
                                                      ULE(ZIf(k != _const(0), x, y), hi)))),
        # Reduction modulo 2^32 commutes with + and *: the low 32 bits of the exact (64-bit,
        # no overflow) sum and product are the 32-bit sum and product.
        Extract(WIDTH - 1, 0, xw + yw) != x + y,
        Extract(WIDTH - 1, 0, kw * xw) != k * x,
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
    ("R8-hash-stage-merge-general", "Stage merge holds for any constants m, c, d and shift 0 <= k < 32", check_hash_stage_merge_general),
    ("R8-hash-stage-merge-needs-left-shift", "Stage merge fails across a right shift (SAT)", check_hash_stage_merge_needs_left_shift),
    ("R9-mirrored-index-step", "Mirrored index n = L - idx updates as 2n + (v' & 1)", check_mirrored_index_step),
    ("R9-mirrored-index-step-needs-odd-constant", "Mirrored update fails for an even deferred constant (SAT)", check_mirrored_index_step_needs_odd_constant),
    ("R10-mirrored-index-uses", "Gather address, compares and the final store written with the mirrored index", check_mirrored_index_uses),
    ("R11-mirrored-index-range", "Mirrored index range per depth and the level constants", check_mirrored_index_range),
    ("R12-index-update-from-root", "From idx = 0 the index update is its offset", check_index_update_from_root),
    ("R12-index-update-from-root-needs-zero", "The offset alone is wrong for a nonzero index (SAT)", check_index_update_from_root_needs_zero),
    ("CSE-identical-broadcast", "Two vbroadcasts of one scalar are equal lane for lane", check_cse_identical_broadcast),
    ("R12-fma-zero-factor", "multiply_add with a zero factor is its addend", check_fma_zero_factor),
    ("R13-select-by-path-bits", "Select tree driven by the branch bits yields the reference node", check_select_by_path_bits),
    ("R13-select-by-path-bits-needs-bit-order", "Select tree with the bits in the wrong order is wrong (SAT)", check_select_by_path_bits_needs_bit_order),
    ("R14-mobius-select", "2^d - 1 multiply_adds over Mobius coefficients equal the select tree", check_mobius_select),
    ("R14-mobius-select-needs-boolean", "Mobius form is wrong for a non-0/1 bit (SAT)", check_mobius_select_needs_boolean),
    ("R15-bits-from-index", "Older branch bits read back from the index", check_bits_from_index),
    ("R15-bits-from-index-needs-range", "Unmasked top bit is not 0/1 outside the index range (SAT)", check_bits_from_index_needs_range),
    ("R16-gather-xor-reassociation", "Deferred constant XORed into the value instead of the gathered node", check_gather_xor_reassociation),
    ("R17-index-early-terms", "V - (2n + bit) = madd(n, -2, V) - bit: one op after the newest bit", check_index_early_terms),
    ("R17-index-early-terms-needs-wrap", "With +2 instead of -2 the early form is wrong (SAT)", check_index_early_terms_needs_wrap),
    ("R18-select-xor-pushdown", "Round XOR pushed below the top select", check_select_xor_pushdown),
    ("R21-linear-normal-form", "+, -, constant multiples and constant left shifts form linear forms", check_linear_normal_form),
    ("R22-xor-normal-form", "XOR expressions are a constant XOR a set of atoms", check_xor_normal_form),
    ("R23-equality-as-difference", "a == b is a - b == 0 (either sign)", check_equality_as_difference),
    ("R24-select-forms", "select(c,t,t) = t; boolean multiply_add and product are selects", check_select_forms),
    ("R25-interval-transfer", "Interval rules used for 0/1 conditions, load addresses and cut facts", check_interval_transfer),
]

# Checks that are expected to find a counterexample.
EXPECT_SAT = {
    "R3-branch-offset-needs-odd-constant",
    "R4-fma-select-needs-boolean",
    "R8-hash-stage-merge-needs-left-shift",
    "R9-mirrored-index-step-needs-odd-constant",
    "R12-index-update-from-root-needs-zero",
    "R13-select-by-path-bits-needs-bit-order",
    "R14-mobius-select-needs-boolean",
    "R15-bits-from-index-needs-range",
    "R17-index-early-terms-needs-wrap",
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
