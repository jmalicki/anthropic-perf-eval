#!/usr/bin/env python3
"""Exact memory dependencies for a straight-line kernel: which loads and stores can alias.

A kernel that uses main memory as temporary storage (public kernels store selected tree
values and pointers to memory mid-kernel and load them back) can only be rescheduled if
we know which memory ops may touch the same word. This module decides that per pair, and
adds an ordering edge only when the two addresses CAN be equal for some valid input.

How
---
1. The kernel is executed symbolically (`verification.kernel_dataflow` terms), with an
   exact memory model: a load returns the value of the latest store that must alias it,
   wrapped in ``ite(addr == store_addr, store_value, ...)`` for every later store that
   may alias it, and the initial memory below that.
2. Initial memory follows the upstream generator (``build_mem_image`` in
   ``upstream/tests/frozen_problem.py``) - the PRECONDITIONS:
     * header words 0..6 are ``rounds, n_nodes, batch_size, forest_height,
       forest_values_p, inp_indices_p, inp_values_p``, all fixed by the parameters;
     * ``forest_values_p = 7``; word 7 holds the root's value (the generator's write of
       ``extra_room`` to word 7 is overwritten by the tree values);
     * tree values and input values are arbitrary integers in [0, 2**30);
     * every input index is 0 (``Input.generate``);
     * memory has ``inp_values_p + batch_size`` words.
   Constant-address loads of the header and index regions therefore fold to constants.
3. Every memory address becomes a term over those inputs. Each candidate pair is decided:
     * value sets: an over-approximation of the values a term can take (constants,
       ``x & c``, ``x % c``, ``x >> c``, ``select`` and arithmetic on small sets). Disjoint
       sets prove the pair disjoint;
     * otherwise the Z3 query over 32-bit bit-vectors: SAT means the addresses can be equal
       (edge), UNSAT means they never are (no edge). ``unknown`` raises: it is a bug to fix,
       never a reason to add an edge. Before calling the solver, the query's formula is
       evaluated on 1,024 fixed assignments of its free variables (numpy, the same
       operations as the Z3 expression); an assignment that satisfies it is a model, so the
       answer is SAT without the solver. Only queries with no sampled model reach Z3, and
       those decide UNSAT (or a SAT the samples missed).
4. The abstraction behind (3), ASSUMPTION A: a subterm whose value set is unbounded
   (a hash value, a tree value) is replaced by a fresh 32-bit variable, shared by every
   occurrence of that subterm. This over-approximates the behaviours, so UNSAT (disjoint)
   is a proof for the real kernel. A SAT witness is a choice of those fresh values, in
   practice the branch bits ``h & 1`` of the hashes; treating different hashes' bits as
   independent is the assumption, and the concrete cross-check (`concrete_check`) runs
   the kernel on random inputs and reports how many SAT pairs were actually seen to alias.

Latencies: store -> load 1 (a load reads memory as it was before the cycle); load ->
store 0; store -> store 0 (stores of one cycle commit in slot order, so the later store may
share the cycle if it stays later in the bundle).

Dependencies on one word, in program order (cycle, then slot order): a load depends on
every earlier store that may alias it, back to the first one that must alias it (older
stores are ordered before that one by store-store edges); a store depends on every earlier
load and store that may alias it, back to the first store that must alias it.

Usage:
    uv run python -m verification.memory_alias kernel.json [--concrete-runs 8]
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import z3

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verification.kernel_dataflow import VLEN, Terms, _execute  # noqa: E402

MASK = 2**32
VALUE_BOUND = 2**30  # tree and input values are drawn from [0, 2**30)
SET_CAP = 4096  # largest value set kept explicitly
MOD_CAP = 64  # x % c with x unbounded is bounded by range(c) for constant c <= MOD_CAP (MA2)
PRODUCT_CAP = 1 << 16  # largest cartesian product evaluated for a binary op
Z3_TIMEOUT_MS = 20_000
N_SAMPLES = 1024  # assignments of the fresh variables tried before asking Z3 (witness search)
RANGE_OPS = {"&", "%", ">>", "<", "==", "//", "cdiv"}


# =============================================================================
# Preconditions
# =============================================================================


@dataclass(frozen=True)
class MemLayout:
    """Initial memory as the upstream generator builds it (see module docstring)."""

    forest_height: int = 10
    batch_size: int = 256
    rounds: int = 16

    @classmethod
    def from_metadata(cls, data: dict, *, require_complete: bool = False) -> MemLayout:
        """Read upstream layout fields, defaulting absent or null kernel metadata."""
        if not isinstance(data, dict):
            raise ValueError("layout metadata must be a JSON object")
        fields = {}
        for name, minimum in (("forest_height", 0), ("batch_size", 1), ("rounds", 0)):
            value = data.get(name)
            if value is None and not require_complete:
                continue
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
            fields[name] = value
        return cls(**fields)

    @property
    def n_nodes(self) -> int:
        return 2 ** (self.forest_height + 1) - 1

    @property
    def forest_p(self) -> int:
        return 7

    @property
    def idx_p(self) -> int:
        return self.forest_p + self.n_nodes

    @property
    def val_p(self) -> int:
        return self.idx_p + self.batch_size

    @property
    def size(self) -> int:
        return self.val_p + self.batch_size

    def header(self) -> list[int]:
        return [self.rounds, self.n_nodes, self.batch_size, self.forest_height,
                self.forest_p, self.idx_p, self.val_p]

    def region(self, addr: int) -> str:
        if addr < self.forest_p:
            return "header"
        if addr < self.idx_p:
            return "tree"
        if addr < self.val_p:
            return "indices"
        if addr < self.size:
            return "values"
        return "outside"


# =============================================================================
# Terms with a sound select lowering and memory ites
# =============================================================================


class AliasTerms(Terms):
    """`Terms` whose multiply_add only becomes a select when the condition is provably 0/1."""

    def is_bool(self, tid: int) -> bool:
        term = self.defs[tid]
        if term[0] == "%" and self.const_value(term[2]) == 2:
            return True
        return super().is_bool(tid)

    def fma(self, a: int, b: int, c: int) -> int:
        ca, cb, cc = self.const_value(a), self.const_value(b), self.const_value(c)
        if ca is not None and cb is not None and cc is not None:
            return self.const(ca * cb + cc)
        for cond, delta in ((a, b), (b, a)):
            dterm = self.defs[delta]
            if dterm[0] == "-" and dterm[2] == c and self.is_bool(cond):
                return self.select(cond, dterm[1], c)
        if b < a:
            a, b = b, a
        return self._intern(("fma", a, b, c))

    def select(self, cond: int, if_true: int, if_false: int) -> int:
        cv = self.const_value(cond)
        if cv is not None:
            return if_true if cv != 0 else if_false
        if if_true == if_false:
            return if_true
        return super().select(cond, if_true, if_false)

    def ite_eq(self, a: int, b: int, then: int, other: int) -> int:
        """``then if a == b else other`` (a load that may read a store's value)."""
        if then == other:
            return then
        return self._intern(("ite_eq", a, b, then, other))


# =============================================================================
# Value sets (sound over-approximation, used as a fast disjointness filter)
# =============================================================================


def _submasks(c: int):
    sub = c
    while True:
        yield sub
        if sub == 0:
            return
        sub = (sub - 1) & c


_SET_FOLD = {
    "+": lambda a, b: (a + b) % MASK,
    "-": lambda a, b: (a - b) % MASK,
    "*": lambda a, b: (a * b) % MASK,
    "^": lambda a, b: a ^ b,
    "&": lambda a, b: a & b,
    "|": lambda a, b: a | b,
    "<<": lambda a, b: (a << b) % MASK,
    ">>": lambda a, b: a >> b,
    "%": lambda a, b: a % b if b else None,
    "//": lambda a, b: a // b if b else None,
    "cdiv": lambda a, b: (a + b - 1) // b if b else None,
    "<": lambda a, b: int(a < b),
    "==": lambda a, b: int(a == b),
}


class ValueSets:
    """`get(tid)` is a frozenset that contains every value the term can take, or None
    (unbounded). Leaves that are inputs (tree/input values, gathered words) are None."""

    def __init__(self, terms: AliasTerms) -> None:
        self.terms = terms
        self.cache: dict[int, frozenset | None] = {}

    def get(self, tid: int) -> frozenset | None:
        if tid in self.cache:
            return self.cache[tid]
        # Iterative post-order: terms can be thousands deep (hash chains).
        stack = [tid]
        while stack:
            t = stack[-1]
            if t in self.cache:
                stack.pop()
                continue
            kids = [k for k in self._children(t) if k not in self.cache]
            if kids:
                stack.extend(kids)
                continue
            stack.pop()
            self.cache[t] = self._compute(t)
        return self.cache[tid]

    def _children(self, t: int) -> list[int]:
        term = self.terms.defs[t]
        if term[0] in ("const", "mem"):
            return []
        return list(term[1:])

    def _compute(self, t: int) -> frozenset | None:
        terms = self.terms
        term = terms.defs[t]
        op = term[0]
        if op == "const":
            return frozenset([term[1]])
        if op == "mem":
            return None
        if op == "sel":
            c, a, b = (self.cache[k] for k in term[1:])
            if c is not None and 0 not in c:
                return a
            if c is not None and c == {0}:
                return b
            return None if a is None or b is None else _cap(a | b)
        if op == "ite_eq":
            v, r = self.cache[term[3]], self.cache[term[4]]
            return None if v is None or r is None else _cap(v | r)
        if op == "fma":
            a, b, c = (self.cache[k] for k in term[1:])
            if a is None or b is None or c is None or len(a) * len(b) * len(c) > PRODUCT_CAP:
                return None
            return _cap(frozenset((x * y + z) % MASK for x in a for y in b for z in c))
        a, b = self.cache[term[1]], self.cache[term[2]]
        cb = terms.const_value(term[2])
        ca = terms.const_value(term[1])
        if a is not None and b is not None and len(a) * len(b) <= PRODUCT_CAP:
            fold = _SET_FOLD[op]
            return _cap(frozenset(v for x in a for y in b if (v := fold(x, y)) is not None))
        # One side unbounded: only range-reducing ops with a constant still bound the result.
        if op == "&":
            c = cb if cb is not None else ca
            if c is not None and bin(c).count("1") <= 12:
                return frozenset(_submasks(c))
        if op == "%" and cb is not None and 0 < cb <= MOD_CAP:
            return frozenset(range(cb))
        if op == ">>" and cb is not None and cb >= 20:
            return frozenset(range(1 << max(0, 32 - cb))) if cb < 32 else frozenset([0])
        if op in ("<", "=="):
            return frozenset([0, 1])
        return None


def _cap(s: frozenset) -> frozenset | None:
    return s if len(s) <= SET_CAP else None


# =============================================================================
# Z3 translation (Assumption A: unbounded subterms become shared fresh variables)
# =============================================================================


class Z3Encoder:
    """Translate terms to 32-bit bit-vectors; see Assumption A in the module docstring."""

    def __init__(self, terms: AliasTerms, vsets: ValueSets, layout: MemLayout) -> None:
        self.terms = terms
        self.vsets = vsets
        self.layout = layout
        self.ctx = z3.Context()
        self.cache: dict[int, z3.BitVecRef] = {}
        self.fresh: dict[int, z3.BitVecRef] = {}
        self.side: dict[int, z3.BoolRef] = {}  # fresh var id -> its range constraint
        self.deps: dict[int, frozenset[int]] = {}  # term -> fresh vars it uses
        self._inc: tuple[int, z3.Solver] | None = None  # (anchor term, incremental solver)
        # Witness search: every fresh variable gets one fixed vector of N_SAMPLES values (in
        # its range), and every encoded term the vector of its values, computed with the same
        # operations as its Z3 expression. An index where the formula holds is a model of the
        # Z3 query, so a SAT answer found this way is the answer Z3 would give.
        self.form: dict[int, tuple] = {}  # term -> ("fresh",) | ("const", v) | (op, kids...)
        self.samples: dict[int, np.ndarray] = {}
        self.sampled_sat = 0  # queries answered by a sampled witness
        self.z3_queries = 0

    def _val(self, v: int):
        return z3.BitVecVal(v % MASK, 32, self.ctx)

    def _fresh(self, t: int):
        if t not in self.fresh:
            var = z3.BitVec(f"x{t}", 32, self.ctx)
            self.fresh[t] = var
            term = self.terms.defs[t]
            if term[0] == "mem":
                addr = self.terms.const_value(term[1])
                if addr is not None and self.layout.region(addr) in ("tree", "values"):
                    self.side[t] = z3.ULT(var, self._val(VALUE_BOUND))
        return self.fresh[t]

    def _is_cut(self, t: int, top: bool) -> bool:
        term = self.terms.defs[t]
        if term[0] == "const":
            return False
        if term[0] == "mem":
            return True
        return not top and self.vsets.get(t) is None

    def encode(self, t: int):
        """Expression for term t, and the fresh variables it depends on."""
        stack = [(t, True)]
        while stack:
            cur, top = stack[-1]
            if cur in self.cache:
                stack.pop()
                continue
            if self._is_cut(cur, top):
                stack.pop()
                self.cache[cur] = self._fresh(cur)
                self.deps[cur] = frozenset([cur])
                self.form[cur] = ("fresh",)
                continue
            term = self.terms.defs[cur]
            if term[0] == "const":
                stack.pop()
                self.cache[cur] = self._val(term[1])
                self.deps[cur] = frozenset()
                self.form[cur] = ("const", term[1])
                continue
            kids = [k for k in term[1:] if k not in self.cache]
            if kids:
                stack.extend((k, False) for k in kids)
                continue
            stack.pop()
            self.cache[cur] = self._build(term)
            self.deps[cur] = frozenset().union(*(self.deps[k] for k in term[1:]))
            self.form[cur] = term
        return self.cache[t]

    def sample(self, t: int) -> np.ndarray:
        """Values of encoded term t under the fixed sample assignments (uint64, < 2**32)."""
        self.encode(t)
        stack = [t]
        while stack:
            cur = stack[-1]
            if cur in self.samples:
                stack.pop()
                continue
            form = self.form[cur]
            if form[0] == "fresh":
                rng = np.random.default_rng(1_000_003 + cur)
                hi = VALUE_BOUND if cur in self.side else MASK
                self.samples[cur] = rng.integers(0, hi, N_SAMPLES, dtype=np.uint64)
                stack.pop()
                continue
            if form[0] == "const":
                self.samples[cur] = np.full(N_SAMPLES, form[1] % MASK, dtype=np.uint64)
                stack.pop()
                continue
            kids = [k for k in form[1:] if k not in self.samples]
            if kids:
                stack.extend(kids)
                continue
            stack.pop()
            self.samples[cur] = _np_op(form[0], [self.samples[k] for k in form[1:]])
        return self.samples[t]

    def _build(self, term: tuple):
        op = term[0]
        args = [self.cache[k] for k in term[1:]]
        one, zero = self._val(1), self._val(0)
        if op == "sel":
            return z3.If(args[0] != zero, args[1], args[2])
        if op == "ite_eq":
            return z3.If(args[0] == args[1], args[2], args[3])
        if op == "fma":
            return args[0] * args[1] + args[2]
        a, b = args
        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op == "*":
            return a * b
        if op == "^":
            return a ^ b
        if op == "&":
            return a & b
        if op == "|":
            return a | b
        if op == "<<":
            return a << b
        if op == ">>":
            return z3.LShR(a, b)
        if op == "%":
            return z3.URem(a, b)
        if op == "//":
            return z3.UDiv(a, b)
        if op == "cdiv":
            wide = z3.ZeroExt(1, a) + z3.ZeroExt(1, b) - z3.BitVecVal(1, 33, self.ctx)
            return z3.Extract(31, 0, z3.UDiv(wide, z3.ZeroExt(1, b)))
        if op == "<":
            return z3.If(z3.ULT(a, b), one, zero)
        if op == "==":
            return z3.If(a == b, one, zero)
        raise ValueError(f"no Z3 encoding for {op!r}")

    def _solve(self, formula, *tids: int) -> str:
        self.z3_queries += 1
        return self._solve_z3(formula, *tids)

    def _solve_z3(self, formula, *tids: int) -> str:
        """Check `formula` with the range constraints of the fresh variables of `tids`.
        Queries come in runs with the same first term (one load or store address against
        each candidate), so one incremental solver per first term is kept and each query
        is pushed and popped on it."""
        anchor = tids[0]
        if self._inc is None or self._inc[0] != anchor:
            solver = z3.Solver(ctx=self.ctx)
            solver.set("timeout", Z3_TIMEOUT_MS)
            for v in self.deps[anchor]:
                if v in self.side:
                    solver.add(self.side[v])
            self._inc = (anchor, solver)
        solver = self._inc[1]
        solver.push()
        solver.add(formula)
        for t in tids[1:]:
            for v in self.deps[t]:
                if v in self.side:
                    solver.add(self.side[v])
        try:
            result = str(solver.check())
        finally:
            solver.pop()
        if result == "unknown":
            raise RuntimeError(f"Z3 returned unknown ({solver.reason_unknown()}) for "
                               f"{self.terms.show(tids[0], 4)} vs {self.terms.show(tids[-1], 4)}")
        return result

    def ranges_overlap(self, a: int, wa: int, b: int, wb: int) -> bool:
        """Can [a, a+wa) and [b, b+wb) share a word? (mod 2**32; proof in check_overlap_rule)"""
        ea, eb = self.encode(a), self.encode(b)
        diff = (self.sample(a) - self.sample(b) + np.uint64(wa - 1)) & np.uint64(MASK - 1)
        if bool(np.any(diff < np.uint64(wa + wb - 1))):
            self.sampled_sat += 1
            return True
        return self._solve(overlap_rule(ea, wa, eb, wb, self.ctx), a, b) == "sat"

    def always_equal(self, a: int, b: int) -> bool:
        ea, eb = self.encode(a), self.encode(b)
        if bool(np.any(self.sample(a) != self.sample(b))):
            self.sampled_sat += 1  # a witness of a != b
            return False
        return self._solve(ea != eb, a, b) == "unsat"


_M = np.uint64(MASK - 1)


def _np_op(op: str, args: list[np.ndarray]) -> np.ndarray:
    """The Z3 encoding of `op` (Z3Encoder._build), on uint64 vectors of 32-bit values."""
    if op == "sel":
        return np.where(args[0] != 0, args[1], args[2])
    if op == "ite_eq":
        return np.where(args[0] == args[1], args[2], args[3])
    if op == "fma":
        return (args[0] * args[1] + args[2]) & _M
    a, b = args
    if op == "+":
        return (a + b) & _M
    if op == "-":
        return (a - b) & _M
    if op == "*":
        return (a * b) & _M  # exact mod 2**32: 2**32 divides 2**64
    if op == "^":
        return a ^ b
    if op == "&":
        return a & b
    if op == "|":
        return a | b
    if op == "<<":
        return np.where(b >= 32, np.uint64(0), (a << np.minimum(b, 31)) & _M)
    if op == ">>":
        return np.where(b >= 32, np.uint64(0), a >> np.minimum(b, 31))
    nz = b != 0
    safe = np.where(nz, b, np.uint64(1))
    if op == "%":
        return np.where(nz, a % safe, a)  # Z3: urem(x, 0) = x
    if op == "//":
        return np.where(nz, a // safe, _M)  # Z3: udiv(x, 0) = all ones
    if op == "cdiv":
        return np.where(nz, ((a + safe - np.uint64(1)) // safe) & _M, _M)
    if op == "<":
        return (a < b).astype(np.uint64)
    if op == "==":
        return (a == b).astype(np.uint64)
    raise ValueError(f"no sample evaluation for {op!r}")


def overlap_rule(a, wa: int, b, wb: int, ctx=None):
    """[a, a+wa) and [b, b+wb) share a word, modulo 2**32: a - b + (wa - 1) <u wa + wb - 1.
    (a + k == b + j  <=>  a - b == j - k in [-(wa - 1), wb - 1]; proved in MA1.)"""
    return z3.ULT(a - b + z3.BitVecVal(wa - 1, 32, ctx), z3.BitVecVal(wa + wb - 1, 32, ctx))


# =============================================================================
# Symbolic execution with exact memory
# =============================================================================


@dataclass
class KOp:
    """One slot of the kernel, in program order."""

    index: int
    cycle: int
    engine: str
    slot: list
    reads: list[int] = field(default_factory=list)  # scratch words read
    writes: list[int] = field(default_factory=list)  # scratch words written
    mem: str | None = None  # "load" / "store"
    mem_addr: int | None = None  # term of the first word
    mem_width: int = 0


@dataclass
class AliasStats:
    mem_ops: int = 0
    candidate_word_pairs: int = 0  # word pairs not separated by the value-set index
    disjoint_value_sets: int = 0  # op pairs (store involved) separated by value sets alone
    disjoint_z3: int = 0  # word pairs Z3 proved disjoint
    may_alias_z3: int = 0  # word pairs Z3 found can be equal
    must_alias: int = 0  # word pairs that are always equal
    op_pairs_total: int = 0  # ordered op pairs with at least one store
    op_edges: int = 0  # op pairs with a memory edge
    z3_queries: int = 0  # alias questions Z3 had to answer (no sampled witness)
    sampled_witnesses: int = 0  # alias questions answered by a sampled model
    seconds: float = 0.0


@dataclass
class _MemEvent:
    op: int
    kind: str  # "load" / "store"
    addr: int  # word address term
    value: int | None  # stored value term (stores)
    vset: frozenset | None


class SymbolicKernel:
    """Run a kernel symbolically; collect scratch read/write sets and exact memory edges."""

    def __init__(self, bundles: list[dict], layout: MemLayout | None = None) -> None:
        self.bundles = bundles
        self.layout = layout or MemLayout()
        self.terms = AliasTerms()
        self.vsets = ValueSets(self.terms)
        self.z3 = Z3Encoder(self.terms, self.vsets, self.layout)
        self.ops: list[KOp] = []
        self.mem_edges: dict[tuple[int, int], int] = {}  # (src, dst) -> latency
        self.stats = AliasStats()
        self._events: list[_MemEvent] = []
        self._by_value: dict[int, list[int]] = {}  # concrete word -> event indices
        self._unbounded: list[int] = []  # events whose address set is unknown
        # Store-to-load data flow per word: (store op, store event, load op, lane, relation).
        self.flows: list[tuple[int, int, int, int, str]] = []
        self._z3_cache: dict[tuple[int, int], str] = {}

    # -- initial memory --------------------------------------------------------
    def _initial(self, addr: int) -> int:
        a = self.terms.const_value(addr)
        if a is not None:
            region = self.layout.region(a)
            if region == "header":
                return self.terms.const(self.layout.header()[a])
            if region == "indices":
                return self.terms.const(0)
        return self.terms.mem(addr)

    # -- alias decisions ---------------------------------------------------------
    def _relation(self, a: int, b: int) -> str:
        """'must', 'may' or 'never' for two word address terms."""
        if a == b:
            return "must"
        key = (a, b) if a < b else (b, a)
        if key in self._z3_cache:
            return self._z3_cache[key]
        va, vb = self.vsets.get(a), self.vsets.get(b)
        if va is not None and vb is not None and not (va & vb):
            rel = "never"
        elif not self.z3.ranges_overlap(a, 1, b, 1):
            rel = "never"
            self.stats.disjoint_z3 += 1
        elif self.z3.always_equal(a, b):
            rel = "must"
            self.stats.must_alias += 1
        else:
            rel = "may"
            self.stats.may_alias_z3 += 1
        self._z3_cache[key] = rel
        return rel

    def _candidates(self, vset: frozenset | None) -> list[int]:
        """Earlier events whose address may equal one of `vset`, newest first."""
        if vset is None:
            found = range(len(self._events))
        else:
            hit = set(self._unbounded)
            for v in vset:
                hit.update(self._by_value.get(v, ()))
            found = hit
        return sorted(found, reverse=True)

    def _add_edge(self, src: int, dst: int, latency: int) -> None:
        if src == dst:
            return
        old = self.mem_edges.get((src, dst))
        if old is None or latency > old:
            self.mem_edges[(src, dst)] = latency

    @property
    def events(self) -> list[_MemEvent]:
        """Every memory word access in program order (loads of a cycle before its stores)."""
        return self._events

    def _load_word(self, op: int, addr: int, lane: int = 0) -> int:
        """Value of one loaded word; records store->load edges."""
        vset = self.vsets.get(addr)
        hits = []
        for e in self._candidates(vset):
            ev = self._events[e]
            if ev.kind != "store":
                continue
            self.stats.candidate_word_pairs += 1
            rel = self._relation(addr, ev.addr)
            if rel == "never":
                continue
            self._add_edge(ev.op, op, 1)
            self.flows.append((ev.op, e, op, lane, rel))
            hits.append((rel, ev))
            if rel == "must":
                break
        value = self._initial(addr) if not hits or hits[-1][0] != "must" else hits[-1][1].value
        for rel, ev in reversed(hits[:-1] if hits and hits[-1][0] == "must" else hits):
            value = self.terms.ite_eq(addr, ev.addr, ev.value, value)
        return value

    def _store_word(self, op: int, addr: int) -> None:
        """Records load->store and store->store edges for one stored word."""
        for e in self._candidates(self.vsets.get(addr)):
            ev = self._events[e]
            self.stats.candidate_word_pairs += 1
            rel = self._relation(addr, ev.addr)
            if rel == "never":
                continue
            # load -> store and store -> store: latency 0. Memory writes commit at the end of
            # the cycle in slot order (the simulator's mem_write dict), so two stores to one word
            # may share a cycle as long as the later one stays later in the bundle; emitters
            # keep program order within a cycle.
            self._add_edge(ev.op, op, 0)
            if rel == "must" and ev.kind == "store":
                break

    def _record(self, op: int, kind: str, addr: int, value: int | None) -> None:
        vset = self.vsets.get(addr)
        idx = len(self._events)
        self._events.append(_MemEvent(op, kind, addr, value, vset))
        if vset is None:
            self._unbounded.append(idx)
        else:
            for v in vset:
                self._by_value.setdefault(v, []).append(idx)

    # -- main loop ----------------------------------------------------------------
    def run(self) -> "SymbolicKernel":
        t0 = time.perf_counter()
        terms = self.terms
        scratch: dict[int, int] = {}
        zero = terms.const(0)

        def read(a: int) -> int:
            return scratch.get(a, zero)

        for cycle, bundle in enumerate(self.bundles):
            writes: list[tuple[int, int]] = []
            pending_events: list[tuple[int, str, int, int | None]] = []
            for engine, slots in bundle.items():
                if engine == "debug":
                    continue
                for slot in slots:
                    op = KOp(len(self.ops), cycle, engine, list(slot))
                    self.ops.append(op)
                    name = slot[0]
                    if engine == "store":
                        width = VLEN if name == "vstore" else 1
                        base = read(slot[1])
                        op.reads = [slot[1]] + [slot[2] + k for k in range(width)]
                        op.mem, op.mem_addr, op.mem_width = "store", base, width
                        for k in range(width):
                            w = terms.binop("+", base, terms.const(k))
                            pending_events.append((op.index, "store", w, read(slot[2] + k)))
                        continue
                    if engine == "load" and name in ("load", "vload", "load_offset"):
                        if name == "load_offset":
                            src, dests, width = slot[2] + slot[3], [slot[1] + slot[3]], 1
                        else:
                            src, width = slot[2], (VLEN if name == "vload" else 1)
                            dests = [slot[1] + k for k in range(width)]
                        base = read(src)
                        op.reads, op.writes = [src], dests
                        op.mem, op.mem_addr, op.mem_width = "load", base, width
                        for k, d in enumerate(dests):
                            w = terms.binop("+", base, terms.const(k))
                            writes.append((d, self._load_word(op.index, w, k)))
                            pending_events.append((op.index, "load", w, None))
                        continue
                    effect = _execute(terms, read, engine, slot)
                    if effect is None:
                        continue
                    dests, results, operands = effect
                    op.writes = list(dests)
                    op.reads = sorted({a for lane in operands for a in lane})
                    writes.extend(zip(dests, results))
            # Loads in this cycle read memory before this cycle's stores commit, so they
            # are recorded first: a same-cycle store then gets a load->store edge.
            for op_index, kind, w, value in pending_events:
                if kind == "load":
                    self._record(op_index, kind, w, value)
            for op_index, kind, w, value in pending_events:
                if kind == "store":
                    self._store_word(op_index, w)
                    self._record(op_index, kind, w, value)
            for d, t in writes:
                scratch[d] = t
        self._finish_stats()
        self.stats.seconds = time.perf_counter() - t0
        return self

    def _finish_stats(self) -> None:
        mem = [o for o in self.ops if o.mem]
        n_st = sum(o.mem == "store" for o in mem)
        n_ld = len(mem) - n_st
        self.stats.mem_ops = len(mem)
        # Ordered pairs (earlier, later) where at least one is a store.
        self.stats.op_pairs_total = n_st * (n_st - 1) // 2 + n_st * n_ld
        self.stats.op_edges = len(self.mem_edges)
        self.stats.z3_queries = self.z3.z3_queries
        self.stats.sampled_witnesses = self.z3.sampled_sat


# =============================================================================
# Concrete cross-check
# =============================================================================


def _frozen_problem():
    import importlib.util

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "upstream", "tests", "frozen_problem.py")
    spec = importlib.util.spec_from_file_location("frozen_problem_upstream", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def trace_memory(bundles: list[dict], seed: int, layout: MemLayout | None = None):
    """Run the kernel on the upstream simulator; return (words touched per op, machine, ref)."""
    fp = _frozen_problem()
    layout = layout or MemLayout()
    random.seed(seed)
    forest = fp.Tree.generate(layout.forest_height)
    inp = fp.Input.generate(forest, layout.batch_size, layout.rounds)
    mem = fp.build_mem_image(forest, inp)
    program = [{e: [tuple(s) for s in slots] for e, slots in b.items()} for b in bundles]
    machine = fp.Machine(mem, program, fp.DebugInfo(scratch_map={}), n_cores=1)
    machine.enable_pause = False
    machine.enable_debug = False
    core = machine.cores[0]
    touched: list[tuple[str, list[int]]] = []
    for instr in program:
        for engine, slots in instr.items():
            if engine == "debug":
                continue
            for slot in slots:
                name = slot[0]
                if engine == "store":
                    width = VLEN if name == "vstore" else 1
                    base = core.scratch[slot[1]]
                    touched.append(("store", [base + k for k in range(width)]))
                elif engine == "load" and name in ("load", "vload", "load_offset"):
                    if name == "load_offset":
                        words = [core.scratch[slot[2] + slot[3]]]
                    else:
                        width = VLEN if name == "vload" else 1
                        words = [core.scratch[slot[2]] + k for k in range(width)]
                    touched.append(("load", words))
                else:
                    touched.append(("", []))
        machine.step(instr, core)
        machine.cycle += 1
    for ref_mem in fp.reference_kernel2(mem):
        pass
    return touched, machine, ref_mem


@dataclass
class ConcreteReport:
    runs: int
    observed_pairs: int  # concretely conflicting (earlier, later) op pairs
    missing_edges: list[tuple[int, int]]
    unseen_edges: int  # memory edges never seen to alias in any run
    outside_value_set: list[tuple[int, int]]  # (op, word) not in the op's value set
    outputs_ok: bool
    ordered_transitively: int = 0  # observed pairs ordered through a must-alias store


def concrete_check(sym: SymbolicKernel, runs: int = 8, seed0: int = 1) -> ConcreteReport:
    """Run the kernel on random inputs with address tracing.

    * Every pair of memory ops that touched a common word (at least one a store) in some run
      must be ordered: by a direct edge, or through stores that must alias both (where the
      scan stops). An unordered pair would mean the preconditions are wrong.
    * Every concrete address must lie in its value set (soundness of the value-set filter).
    * Memory edges whose two ops never shared a word in any run are counted: aliasing the
      analysis proved possible but random inputs did not produce (Assumption A's slack, or
      rare inputs).
    """
    observed: set[tuple[int, int]] = set()  # aliased and ordered by a direct edge
    transitive: set[tuple[int, int]] = set()  # aliased and ordered through must-alias stores
    missing_set: set[tuple[int, int]] = set()  # aliased and not ordered: a bug
    outside: list[tuple[int, int]] = []
    ok = True
    layout = sym.layout
    order = sorted(range(len(sym.ops)), key=lambda i: (sym.ops[i].cycle, sym.ops[i].mem != "load", i))
    for r in range(runs):
        touched, machine, ref = trace_memory(sym.bundles, seed0 + r, layout)
        ok &= machine.mem[layout.val_p:layout.size] == ref[layout.val_p:layout.size]
        # Per word, the ops that touched it, in program order (loads of a cycle first).
        by_word: dict[int, list[int]] = {}
        for i in order:
            kind, words = touched[i]
            if not kind:
                continue
            vs = sym.vsets.get(sym.ops[i].mem_addr)
            if vs is not None and (words[0] % MASK) not in vs:
                outside.append((i, words[0]))
            for w in set(words):
                by_word.setdefault(w, []).append(i)
        for ops in by_word.values():
            # Ordering may be direct or through a store that must alias both (the scan stops
            # there): such a path runs through ops that touched this same word, so the
            # closure is taken over this word's ops only. reach[k] = bitmask of later ops.
            n = len(ops)
            is_store = [sym.ops[i].mem == "store" for i in ops]
            reach = [0] * n
            for k in range(n - 1, -1, -1):
                acc = 0
                for m in range(k + 1, n):
                    if (ops[k], ops[m]) in sym.mem_edges:
                        acc |= (1 << m) | reach[m]
                reach[k] = acc
            for k in range(n):
                for m in range(k + 1, n):
                    if not (is_store[k] or is_store[m]):
                        continue
                    pair = (ops[k], ops[m])
                    if pair in observed or pair in transitive or pair in missing_set:
                        continue
                    if pair in sym.mem_edges:
                        observed.add(pair)
                    elif reach[k] >> m & 1:
                        transitive.add(pair)
                    else:
                        missing_set.add(pair)
    missing = sorted(missing_set)
    unseen = sum(1 for p in sym.mem_edges if p not in observed)
    return ConcreteReport(runs, len(observed) + len(transitive) + len(missing), missing, unseen, outside, ok,
                          len(transitive))


# =============================================================================
# Z3 lemmas behind the rules above (run by ./scripts/test.sh proofs)
# =============================================================================


def _refute(name: str, violation) -> tuple[str, str, float, dict]:
    solver = z3.Solver()
    solver.add(violation)
    return (name, str(solver.check()), 0.0, {})


def check_overlap_rule():
    """MA1: overlap_rule is exactly "some word of one range equals some word of the other",
    for every width pair the ISA has (scalar 1, vector 8), with wrap-around."""
    a, b = z3.BitVecs("a b", 32)
    for wa in (1, 8):
        for wb in (1, 8):
            exists = z3.Or([a + k == b + j for k in range(wa) for j in range(wb)])
            name, result, *_ = _refute("MA1", exists != overlap_rule(a, wa, b, wb))
            if result != "unsat":
                return (name, f"{result} (wa={wa}, wb={wb})", 0.0, {})
    return ("MA1", "unsat", 0.0, {})


def check_value_set_rules():
    """MA2: the value-set rules for one unbounded operand x and a constant c: x & c is a
    submask of c; x % c < c; x >> c < 2**(32 - c); x % 2 == x & 1 (one branch bit).
    One query per rule; % for each constant 1..MOD_CAP the rule accepts (a symbolic divisor
    is too slow to bit-blast), >> for a symbolic shift in [20, 32)."""
    x, c, sh = z3.BitVecs("x c s", 32)
    violations = [
        (x & c) & ~c != 0,
        *[z3.Not(z3.ULT(z3.URem(x, k), k)) for k in range(1, MOD_CAP + 1)],
        z3.And(z3.UGE(sh, 20), z3.ULT(sh, 32),
               z3.Not(z3.ULT(z3.LShR(x, sh), z3.BitVecVal(1, 32) << (32 - sh)))),
        z3.URem(x, 2) != x & 1,
    ]
    for i, v in enumerate(violations):
        name, result, *_ = _refute("MA2", v)
        if result != "unsat":
            return (name, f"{result} (rule {i})", 0.0, {})
    return ("MA2", "unsat", 0.0, {})


def check_fresh_variable_abstraction():
    """MA3: replacing a subterm by a fresh variable only adds behaviours. If the abstract
    pair can never be equal, neither can the concrete one: for f(x) = x & 1 (a hash's
    branch bit), equality of g(f(x)) and h(f(x)) implies satisfiability with f(x) := v."""
    x, v = z3.BitVecs("x v", 32)
    base = z3.BitVecVal(2310, 32)
    concrete = base + (x & 1) == base + 1 - (x & 1)
    abstract = base + (v & 1) == base + 1 - (v & 1)
    # concrete SAT would give a witness for abstract (v := x); so abstract UNSAT must imply
    # concrete UNSAT. Check the instance: both are UNSAT.
    s = z3.Solver()
    s.add(z3.Or(concrete, abstract))
    return ("MA3", str(s.check()), 0.0, {})


MEMORY_ALIAS_CHECKS = [
    ("MA1-overlap-rule", "Word-range overlap test equals per-word equality (mod 2^32)", check_overlap_rule),
    ("MA2-value-set-rules", "Value-set rules for &, %, >> with a constant; x%2 == x&1", check_value_set_rules),
    ("MA3-fresh-variable", "Branch-bit abstraction instance: 2310+b == 2311-b has no solution",
     check_fresh_variable_abstraction),
]


# =============================================================================
# CLI
# =============================================================================


def load_bundles(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data["instrs"] if isinstance(data, dict) else data


def load_kernel(path: str) -> tuple[list[dict], MemLayout]:
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data["instrs"], MemLayout.from_metadata(data)
    return data, MemLayout()


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("kernel", help="kernel JSON (bundles under 'instrs')")
    ap.add_argument("--concrete-runs", type=int, default=8)
    args = ap.parse_args()
    try:
        bundles, layout = load_kernel(args.kernel)
    except ValueError as e:
        ap.error(str(e))
    sym = SymbolicKernel(bundles, layout).run()
    s = sym.stats
    print(f"{s.mem_ops} memory ops; {s.op_pairs_total} ordered op pairs with a store")
    print(f"  memory edges: {s.op_edges}; all other pairs proved disjoint or ordered through a must-alias store")
    print(f"  word pairs: {s.candidate_word_pairs} candidates after value sets; Z3: {s.disjoint_z3} disjoint, "
          f"{s.may_alias_z3} may alias, {s.must_alias} must alias  ({s.seconds:.1f}s)")
    if args.concrete_runs:
        rep = concrete_check(sym, args.concrete_runs)
        print(f"  concrete ({rep.runs} runs): {rep.observed_pairs} aliasing pairs observed, "
              f"{len(rep.missing_edges)} without an edge, {rep.unseen_edges} edges never observed, "
              f"{len(rep.outside_value_set)} addresses outside their value set, outputs ok={rep.outputs_ok}")
        if rep.missing_edges or rep.outside_value_set:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
