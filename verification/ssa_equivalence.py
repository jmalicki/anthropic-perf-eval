#!/usr/bin/env python3
"""SSA equivalence: prove that two SSA programs write the same final memory for every valid input.

Given two scheduler output JSONs (``--output-json``; the ``ssa_ops`` and ``vector_group_lanes``
that ``verification.kernel_dataflow`` also consumes), decide whether program B writes the same
final memory as program A for every input that satisfies the problem's preconditions, or name
the first output word where they differ with a concrete input that shows it.

Together with ``kernel_dataflow`` (kernel = its SSA program) this gives a chain
kernel_A = SSA_A = SSA_B = kernel_B: a rewritten emission (``--merge-hash-stages``,
``--index-form mirrored``, ``--cached-select ...``, ``--prune-dead-index``, ``--values-only``,
``--emission-plan``) computes what the canonical emission computes.

What is proved
--------------
PROVED means: for every initial memory image that satisfies the PRECONDITIONS, the final value
of every compared word is the same in A and B. Compared words are the union of the words either
program stores (a word a program does not store keeps its initial value), or with
``--values-only`` only the ``batch_size`` output values (the indices are not compared).

PRECONDITIONS (``problem.build_mem_image``; the layout is taken from it, not hard-coded):
  * header words 0 .. forest_values_p - 1 hold the parameters (rounds, n_nodes, batch_size,
    height, the region pointers), all fixed by the problem shape;
  * every input index is 0 (``Input.generate``);
  * tree values and input values are arbitrary integers in [0, 2^30); every other word
    (extra room) is 0.

SEMANTICS of an SSA program (the same model ``kernel_dataflow`` uses): ops are pure functions
of their operands with machine semantics modulo 2^32; every load reads the INITIAL memory;
stores write the final memory. This equals the machine's behaviour when every load of a word
that the program also stores happens before that store. That is checked, not assumed: a
constant-address load of a stored word must be in the dependency cone of that store (so every
schedule orders it first), and a load with a non-constant address must have an address
interval (R20) inside memory and disjoint from the stored words. A program that fails the
check is reported UNPROVED.

How
---
1. Normalised symbolic execution. Both programs are executed into one hash-consed term store
   whose constructors apply proved identities, so equal computations become the same term:
   constant folding (machine semantics), linear forms over Z/2^32 for +, -, constant
   multiples, constant left shifts and multiply_add with a constant factor (R16), XOR sets
   (R17), equality as a difference compared with zero (R18), a multiply_add or product whose
   factor is 0/1 as a select (R4, R5, R19; 0/1-ness from intervals, R20), select(c, t, t) = t
   (R19), and the preconditions (header loads and index loads become constants). The merged
   hash stages (R8), the mirrored index (R9-R11), the pruned index work (R12) and the
   stage-5 constant deferral (R2) all normalise to identical terms this way.
2. Sweeping with cut points (as in combinational equivalence checking). Every term is
   evaluated on a set of random valid inputs; terms of A and B with the same values on all
   samples are candidates. In topological order each candidate pair is proved equal by Z3
   over 32-bit bit-vectors, with every subterm shared by both programs replaced by a fresh
   variable constrained to its interval (an over-approximation, so UNSAT is a proof). Proven
   pairs are merged (union-find) and later terms are rebuilt over the representatives, so
   cut points move up round by round and every query stays small (the cached-level select
   trees of R13-R15 are the typical residue). Queries equal up to renaming of their cut
   variables are answered once.
3. Every output word whose A and B terms end in one class is proved. Otherwise the samples
   give a concrete input on which the programs differ, confirmed by a plain concrete
   interpreter of the SSA ops (no normalisation); the first such word (lowest address) is
   reported. If no sample differs and some word is not merged, the result is UNPROVED (never
   a pass). Z3 ``unknown`` and timeouts count as not proved.

Usage:
    uv run python -m verification.ssa_equivalence a.json b.json [--values-only]
        [--samples 32] [--seed 0] [--timeout-ms 20000] [--counterexample-out cex.json]

Exit status: 0 proved, 1 counterexample, 2 not proved (or an input error).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import z3

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verification.kernel_dataflow import _FOLD, VLEN, Terms, ssa_values  # noqa: E402

MOD = 1 << 32
MASK = MOD - 1
VALUE_BOUND = 1 << 30  # input values and tree values are drawn from [0, 2^30)
TOP = (0, MASK)
SIGNED_HALF = 1 << 31
PROGRAM_A, PROGRAM_B, BOTH = 1, 2, 3
_COMMUTATIVE_ATOMS = {"&", "|", "*"}


class EquivalenceError(ValueError):
    """The inputs cannot be compared (different problem shapes, unsupported ops, ...)."""


# =============================================================================
# Memory layout and the preconditions
# =============================================================================


@dataclass(frozen=True)
class Layout:
    """Initial memory layout of one problem shape, taken from ``problem.build_mem_image``."""

    rounds: int
    batch_size: int
    forest_height: int
    n_nodes: int
    image: tuple[int, ...]  # the image with all inputs (tree values, input values) zero
    forest_p: int
    indices_p: int
    values_p: int

    @classmethod
    def from_params(cls, rounds: int, batch_size: int, forest_height: int) -> "Layout":
        from problem import Input, Tree, build_mem_image

        n_nodes = (1 << (forest_height + 1)) - 1
        mem = build_mem_image(Tree(forest_height, [0] * n_nodes),
                              Input([0] * batch_size, [0] * batch_size, rounds))
        return cls(rounds, batch_size, forest_height, n_nodes, tuple(mem), mem[4], mem[5], mem[6])

    @property
    def mem_size(self) -> int:
        return len(self.image)

    def is_input(self, addr: int) -> bool:
        """True for the words the preconditions leave free (tree values and input values)."""
        return (self.forest_p <= addr < self.forest_p + self.n_nodes
                or self.values_p <= addr < self.values_p + self.batch_size)

    def fixed_word(self, addr: int) -> int | None:
        """The initial value of `addr` if the preconditions fix it, else None."""
        if 0 <= addr < self.mem_size and not self.is_input(addr):
            return self.image[addr]
        return None

    def region(self, addr: int) -> str:
        for name, base, size in (("values", self.values_p, self.batch_size),
                                 ("indices", self.indices_p, self.batch_size),
                                 ("tree", self.forest_p, self.n_nodes)):
            if base <= addr < base + size:
                return f"{name}[{addr - base}]"
        return "header" if addr < self.forest_p else "extra"

    def value_words(self) -> list[int]:
        return list(range(self.values_p, self.values_p + self.batch_size))

    def sample_images(self, n: int, seed: int) -> np.ndarray:
        """`n` valid initial images: all inputs 0, all inputs 2^30 - 1, then random ones."""
        rng = np.random.default_rng(seed)
        base = np.array(self.image, dtype=np.uint64)
        images = np.tile(base, (n, 1))
        regions = [slice(self.forest_p, self.forest_p + self.n_nodes),
                   slice(self.values_p, self.values_p + self.batch_size)]
        for s in range(n):
            for region in regions:
                size = region.stop - region.start
                if s == 0:
                    images[s, region] = 0
                elif s == 1:
                    images[s, region] = VALUE_BOUND - 1
                else:
                    images[s, region] = rng.integers(0, VALUE_BOUND, size=size, dtype=np.uint64)
        return images


def layout_of(sched: dict, shape: tuple[int, int, int] | None = None) -> Layout:
    """Layout for `sched`: from `shape` (rounds, batch_size, forest_height) if given, else from
    the run_options the generator writes."""
    if shape is not None:
        return Layout.from_params(*shape)
    opts = sched.get("run_options") or {}
    try:
        return Layout.from_params(int(opts["rounds"]), int(opts["batch_size"]), int(opts["forest_height"]))
    except KeyError as missing:
        raise EquivalenceError(f"scheduler JSON lacks run_options[{missing}]; regenerate it with the "
                               "current generator (--output-json) or give the shape") from None


# =============================================================================
# Normalising term store
# =============================================================================


def _children(term: tuple) -> tuple[int, ...]:
    kind = term[0]
    if kind == "const":
        return ()
    if kind == "lin":
        return tuple(a for a, _ in term[2])
    if kind == "xor":
        return term[2]
    return term[1:]


def _raw(op: str, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Machine semantics of a binary ALU op, lane-wise on uint64 arrays of 32-bit values."""
    if op == "+":
        return (a + b) & MASK
    if op == "-":
        return (a - b) & MASK
    if op == "*":
        return (a * b) & MASK
    if op == "^":
        return a ^ b
    if op == "&":
        return a & b
    if op == "|":
        return a | b
    if op in ("<<", ">>"):
        k = np.minimum(b, 63)
        shifted = ((a << k) & MASK) if op == "<<" else (a >> k)
        return np.where(b >= 32, np.uint64(0), shifted)
    if op == "<":
        return (a < b).astype(np.uint64)
    if op == "==":
        return (a == b).astype(np.uint64)
    safe = np.where(b == 0, np.uint64(1), b)
    if op == "//":
        return a // safe
    if op == "%":
        return a % safe
    if op == "cdiv":
        return ((a + safe - 1) // safe) & MASK
    raise EquivalenceError(f"unknown binary op {op!r}")


class NormTerms(Terms):
    """Hash-consed terms in normal form, each with its values on the sample inputs and an
    unsigned interval.

    Term shapes: ("const", v); ("lin", c, ((atom, k), ...)) = c + sum k*atom (R16);
    ("xor", c, (atoms...)) (R17); ("eqz", d) = (d == 0) (R18); ("sel", c, t, f);
    ("mem", addr) for a load of an input word; (op, a, b) for the remaining ALU ops.
    Every constructor checks its result against the machine semantics of the op on the
    samples, so a wrong normalisation fails loudly instead of proving something false.
    """

    def __init__(self, layout: Layout, images: np.ndarray) -> None:
        super().__init__()
        self.layout = layout
        self.images = images
        self.rows = np.arange(images.shape[0])
        self.vals: list[np.ndarray] = []
        self.iv: list[tuple[int, int]] = []
        self.out_of_range_loads = 0
        self.zero_divisions = 0

    # ---- interning, values, intervals ------------------------------------------------

    def _intern(self, term: tuple) -> int:
        tid = self._ids.get(term)
        if tid is None:
            tid = len(self.defs)
            self._ids[term] = tid
            self.defs.append(term)
            self.vals.append(self._eval(term))
            self.iv.append(self._interval(term))
        return tid

    def _eval(self, term: tuple) -> np.ndarray:
        kind, vals = term[0], self.vals
        n = len(self.rows)
        if kind == "const":
            return np.full(n, term[1], dtype=np.uint64)
        if kind == "lin":
            acc = np.full(n, term[1], dtype=np.uint64)
            for atom, k in term[2]:
                acc = acc + vals[atom] * np.uint64(k)  # wraps modulo 2^64, a multiple of 2^32
            return acc & MASK
        if kind == "xor":
            acc = np.full(n, term[1], dtype=np.uint64)
            for atom in term[2]:
                acc = acc ^ vals[atom]
            return acc
        if kind == "eqz":
            return (vals[term[1]] == 0).astype(np.uint64)
        if kind == "sel":
            return np.where(vals[term[1]] != 0, vals[term[2]], vals[term[3]])
        if kind == "mem":
            addr = vals[term[1]]
            bad = addr >= self.layout.mem_size
            if bad.any():
                self.out_of_range_loads += 1
            return np.where(bad, np.uint64(0), self.images[self.rows, np.where(bad, 0, addr).astype(np.int64)])
        a, b = vals[term[1]], vals[term[2]]
        if kind in ("//", "%", "cdiv") and (b == 0).any():
            self.zero_divisions += 1
        return _raw(kind, a, b)

    def _lin_interval(self, c: int, coeffs) -> tuple[int, int]:
        lo = hi = 0
        for atom, k in coeffs:
            s = k - MOD if k >= SIGNED_HALF else k
            alo, ahi = self.iv[atom]
            lo += s * alo if s > 0 else s * ahi
            hi += s * ahi if s > 0 else s * alo
        for rep in (c, c - MOD):  # either representative of c (R20: reduction commutes)
            if 0 <= lo + rep and hi + rep <= MASK:
                return (lo + rep, hi + rep)
        return TOP

    def _interval(self, term: tuple) -> tuple[int, int]:
        """Sound unsigned interval of a term (rules proved in R20)."""
        kind, iv = term[0], self.iv
        if kind == "const":
            return (term[1], term[1])
        if kind == "lin":
            return self._lin_interval(term[1], term[2])
        if kind == "xor":
            bits = max([term[1].bit_length()] + [iv[a][1].bit_length() for a in term[2]])
            return (0, (1 << bits) - 1)
        if kind in ("eqz", "<"):
            return (0, 1)
        if kind == "sel":
            clo, chi = iv[term[1]]
            if clo > 0:
                return iv[term[2]]
            if chi == 0:
                return iv[term[3]]
            return (min(iv[term[2]][0], iv[term[3]][0]), max(iv[term[2]][1], iv[term[3]][1]))
        if kind == "mem":
            # Every initial word is below 2^30: inputs by precondition, the rest by the layout.
            return (0, VALUE_BOUND - 1) if iv[term[1]][1] < self.layout.mem_size else TOP
        a, b = iv[term[1]], iv[term[2]]
        if kind == "&":
            return (0, min(a[1], b[1]))
        if kind == "|":
            return (max(a[0], b[0]), (1 << max(a[1].bit_length(), b[1].bit_length())) - 1)
        if kind == ">>" and b[0] == b[1]:
            k = b[0]
            return (0, 0) if k >= 32 else (a[0] >> k, a[1] >> k)
        if kind == ">>":
            return (0, a[1])
        return TOP

    def is_bool(self, tid: int) -> bool:
        return self.iv[tid][1] <= 1

    def _checked(self, tid: int, raw: np.ndarray, what: str) -> int:
        if not np.array_equal(self.vals[tid], raw):
            raise AssertionError(f"normaliser bug: {what} -> {self.show(tid)} disagrees with machine semantics")
        return tid

    # ---- linear forms (R16) -------------------------------------------------------------

    def _lin_parts(self, t: int) -> tuple[int, dict[int, int]]:
        term = self.defs[t]
        if term[0] == "const":
            return term[1], {}
        if term[0] == "lin":
            return term[1], dict(term[2])
        return 0, {t: 1}

    def linear(self, c: int, parts: list[tuple[int, int]]) -> int:
        """c + sum k*t over terms t (each flattened into its own linear form)."""
        coeffs: dict[int, int] = {}
        for t, k in parts:
            tc, tco = self._lin_parts(t)
            c += k * tc
            for atom, ak in tco.items():
                coeffs[atom] = (coeffs.get(atom, 0) + k * ak) % MOD
        c %= MOD
        items = tuple(sorted((a, k) for a, k in coeffs.items() if k))
        if not items:
            return self.const(c)
        if c == 0 and len(items) == 1 and items[0][1] == 1:
            return items[0][0]
        return self._intern(("lin", c, items))

    def add(self, a: int, b: int) -> int:
        return self.linear(0, [(a, 1), (b, 1)])

    def scale(self, a: int, k: int) -> int:
        return self.linear(0, [(a, k % MOD)])

    # ---- XOR sets (R17) -----------------------------------------------------------------

    def _xor_parts(self, t: int) -> tuple[int, set[int]]:
        term = self.defs[t]
        if term[0] == "const":
            return term[1], set()
        if term[0] == "xor":
            return term[1], set(term[2])
        return 0, {t}

    def xor(self, items: list[int]) -> int:
        c, atoms = 0, set()
        for t in items:
            tc, ta = self._xor_parts(t)
            c ^= tc
            atoms ^= ta
        if not atoms:
            return self.const(c)
        if c == 0 and len(atoms) == 1:
            return next(iter(atoms))
        return self._intern(("xor", c, tuple(sorted(atoms))))

    # ---- equality (R18), selects (R19) ---------------------------------------------------

    def eqz(self, d: int) -> int:
        cd = self.const_value(d)
        if cd is not None:
            return self.const(int(cd == 0))
        _, coeffs = self._lin_parts(d)
        lead = coeffs[min(coeffs)]
        if lead > SIGNED_HALF or (lead == SIGNED_HALF and self.scale(d, MASK) < d):
            d = self.scale(d, MASK)
        return self._intern(("eqz", d))

    def sel(self, c: int, t: int, f: int) -> int:
        cc = self.const_value(c)
        if cc is not None:
            return t if cc else f
        lo, hi = self.iv[c]
        if lo > 0:
            return t
        if hi == 0:
            return f
        if t == f:
            return t
        return self._intern(("sel", c, t, f))

    # ---- the Terms interface used by kernel_dataflow.ssa_values -------------------------

    def binop(self, op: str, a: int, b: int) -> int:
        if op not in _FOLD:
            raise EquivalenceError(f"unknown binary op {op!r}")
        return self._checked(self._binop(op, a, b), _raw(op, self.vals[a], self.vals[b]), op)

    def _binop(self, op: str, a: int, b: int) -> int:
        ca, cb = self.const_value(a), self.const_value(b)
        if ca is not None and cb is not None and not (op in ("//", "cdiv", "%") and cb == 0):
            return self.const(_FOLD[op](ca, cb))
        if op == "+":
            return self.add(a, b)
        if op == "-":
            return self.linear(0, [(a, 1), (b, MASK)])
        if op == "*":
            if ca is not None or cb is not None:
                return self.scale(b, ca) if ca is not None else self.scale(a, cb)
            if self.is_bool(a):
                return self.sel(a, b, self.const(0))
            if self.is_bool(b):
                return self.sel(b, a, self.const(0))
        if op == "<<" and cb is not None:
            return self.scale(a, 0 if cb >= 32 else 1 << cb)
        if op == "^":
            return self.xor([a, b])
        if op == "==":
            return self.eqz(self.linear(0, [(a, 1), (b, MASK)]))
        if op in _COMMUTATIVE_ATOMS and b < a:
            a, b = b, a
        return self._intern((op, a, b))

    def select(self, cond: int, if_true: int, if_false: int) -> int:
        v = self.vals
        return self._checked(self.sel(cond, if_true, if_false),
                             np.where(v[cond] != 0, v[if_true], v[if_false]), "select")

    def fma(self, a: int, b: int, c: int) -> int:
        v = self.vals
        raw = ((v[a] * v[b]) & MASK) + v[c] & MASK
        return self._checked(self._fma(a, b, c), raw, "multiply_add")

    def _fma(self, a: int, b: int, c: int) -> int:
        ca, cb = self.const_value(a), self.const_value(b)
        if ca is not None or cb is not None:
            return self.linear(0, [(b, ca), (c, 1)] if ca is not None else [(a, cb), (c, 1)])
        if self.is_bool(a):
            return self.sel(a, self.add(c, b), c)
        if self.is_bool(b):
            return self.sel(b, self.add(c, a), c)
        lo, hi = (a, b) if a < b else (b, a)
        return self.add(self._intern(("*", lo, hi)), c)

    def mem(self, addr: int) -> int:
        ca = self.const_value(addr)
        if ca is not None and self.layout.fixed_word(ca) is not None:
            tid = self.const(self.layout.fixed_word(ca))
        else:
            tid = self._intern(("mem", addr))
        addrs = self.vals[addr]
        bad = addrs >= self.layout.mem_size
        raw = np.where(bad, np.uint64(0), self.images[self.rows, np.where(bad, 0, addrs).astype(np.int64)])
        return self._checked(tid, raw, "load")

    def show(self, tid: int, depth: int = 3) -> str:
        term = self.defs[tid]
        if term[0] == "const":
            return f"{term[1]:#x}" if term[1] > 9 else str(term[1])
        if depth == 0:
            return "…"
        if term[0] == "lin":
            parts = [f"{k if k < SIGNED_HALF else k - MOD}*{self.show(a, depth - 1)}" for a, k in term[2]]
            return "(" + " + ".join(parts + ([str(term[1])] if term[1] else [])) + ")"
        if term[0] == "xor":
            return "(" + " ^ ".join([self.show(a, depth - 1) for a in term[2]] + ([f"{term[1]:#x}"] if term[1] else [])) + ")"
        return f"{term[0]}(" + ", ".join(self.show(t, depth - 1) for t in term[1:]) + ")"


# =============================================================================
# Programs
# =============================================================================


def _store_words(sched: dict) -> int:
    n = 0
    for op in sched["ssa_ops"]:
        if op["cycle"] is not None and op["cycle"] < 0:
            continue
        n += VLEN if op["opcode"] == "vstore" else op["opcode"] == "store"
    return n


def build_program(sched: dict, terms: Terms, name: str) -> tuple[dict[int, tuple[int, str]], int]:
    """Run ``kernel_dataflow.ssa_values`` with `terms`; return {address: (value, where)}.

    Store addresses must be constants under the preconditions and every word may be stored
    at most once (an SSA program has no order between two stores of one word).
    """
    values = ssa_values(sched, terms)
    stores: dict[int, tuple[int, str]] = {}
    for addr_tid, (val, where) in values.stores.items():
        addr = terms.const_value(addr_tid) if isinstance(terms, NormTerms) else addr_tid[0]
        if addr is None:
            raise EquivalenceError(f"{name}: {where} stores to a non-constant address {terms.show(addr_tid)}")
        stores[addr] = (val, where)
    if len(stores) != _store_words(sched):
        raise EquivalenceError(f"{name}: some word is stored more than once ({_store_words(sched)} store "
                               f"words, {len(stores)} distinct addresses); the SSA program does not order them")
    return stores, len(sched["ssa_ops"])


class ConcreteTerms:
    """Plain concrete interpreter for ``ssa_values``: a value is (int, loads it depends on).

    No normalisation: it confirms counterexamples independently of NormTerms, and its
    dependency sets check the memory-order condition (a load of a stored word must precede
    the store). Only loads of words in `watch` are tracked.
    """

    def __init__(self, image, watch: set[int] | frozenset = frozenset()) -> None:
        self.image = image
        self.watch = watch
        self.loads: list[tuple[int, int]] = []  # (load id, address) of watched loads

    def const(self, value: int):
        return (value % MOD, frozenset())

    def const_value(self, t):
        return t[0]

    def show(self, t) -> str:
        return str(t[0])

    def binop(self, op, a, b):
        if op in ("//", "%", "cdiv") and b[0] == 0:
            raise EquivalenceError(f"division by zero ({op}) on the counterexample input")
        return (_FOLD[op](a[0], b[0]) % MOD, a[1] | b[1])

    def select(self, c, t, f):
        return (t[0] if c[0] else f[0], c[1] | t[1] | f[1])

    def fma(self, a, b, c):
        return ((a[0] * b[0] + c[0]) % MOD, a[1] | b[1] | c[1])

    def mem(self, addr):
        if not 0 <= addr[0] < len(self.image):
            raise EquivalenceError(f"load from address {addr[0]} outside memory on the counterexample input")
        deps = addr[1]
        if addr[0] in self.watch:
            self.loads.append((len(self.loads), addr[0]))
            deps = deps | {len(self.loads) - 1}
        return (int(self.image[addr[0]]), deps)


def final_memory(sched: dict, image) -> dict[int, int]:
    """Concrete final values of the words `sched` stores, on initial memory `image`."""
    stores, _ = build_program(sched, ConcreteTerms(image), "program")
    return {addr: val[0] for addr, (val, _) in stores.items()}


def memory_order_problems(sched: dict, terms: NormTerms, stores: dict[int, tuple[int, str]],
                          reach: set[int], layout: Layout, name: str) -> list[str]:
    """Loads that might observe a store of the same program (the model reads initial memory)."""
    stored = set(stores)
    problems = []
    for t in sorted(reach):
        term = terms.defs[t]
        if term[0] != "mem" or terms.const_value(term[1]) is not None:
            continue
        lo, hi = terms.iv[term[1]]
        if hi >= layout.mem_size:
            problems.append(f"{name}: load {terms.show(t)} may read outside memory (address in [{lo}, {hi}])")
        elif any(lo <= s <= hi for s in stored):
            problems.append(f"{name}: load {terms.show(t)} (address in [{lo}, {hi}]) may read a word the program stores")
    # Constant-address loads (including those folded by the preconditions): exact, via the
    # dependency sets of the concrete interpreter.
    interp = ConcreteTerms(layout.image, frozenset(stored))
    concrete, _ = build_program(sched, interp, name)
    deps_of = {addr: val[1] for addr, (val, _) in concrete.items()}
    for load_id, addr in interp.loads:
        if addr in deps_of and load_id not in deps_of[addr]:
            problems.append(f"{name}: a load of mem[{addr}] ({layout.region(addr)}) is not ordered before "
                            f"the store to that word ({stores[addr][1]})")
    return problems


# =============================================================================
# Sweeping: merge proven-equal terms bottom-up
# =============================================================================


@dataclass
class SweepStats:
    terms: int = 0
    swept: int = 0
    candidates: int = 0
    merged: int = 0
    queries: int = 0
    cache_hits: int = 0
    sat: int = 0
    unknown: int = 0
    too_large: int = 0
    z3_seconds: float = 0.0
    max_query_seconds: float = 0.0
    max_query_nodes: int = 0


class Sweeper:
    """Union-find over terms of both programs; merges a pair only after Z3 proves it.

    Which pairs are tried (the alignment of cut points) is separate from how a pair is
    proved: pairs come from equal values on the sample inputs (`sweep`), which needs no
    emitter metadata, so programs from different builders (a lifted kernel, the upstream
    baseline) can be compared; `prove` accepts any pair and is sound whatever pairs it gets.
    """

    def __init__(self, terms: NormTerms, timeout_ms: int, node_budget: int) -> None:
        self.terms = terms
        self.timeout_ms = timeout_ms
        self.node_budget = node_budget
        self.parent: dict[int, int] = {}
        self.mask: dict[int, int] = {}
        self.by_sig: dict[bytes, list[int]] = {}
        self.memo: dict[tuple, str] = {}
        self.max_split_bits = 8
        self.stats = SweepStats()

    def find(self, t: int) -> int:
        root = t
        while root in self.parent:
            root = self.parent[root]
        while t in self.parent and self.parent[t] != root:
            self.parent[t], t = root, self.parent[t]
        return root

    def rebuild(self, t: int) -> int:
        """`t` with every child replaced by its representative, re-normalised."""
        T, term = self.terms, self.terms.defs[t]
        kind = term[0]
        kids = _children(term)
        if all(self.find(k) == k for k in kids):
            return t
        f = self.find
        if kind == "lin":
            return T.linear(term[1], [(f(a), k) for a, k in term[2]])
        if kind == "xor":
            return T.xor([T.const(term[1])] + [f(a) for a in term[2]])
        if kind == "eqz":
            return T.eqz(f(term[1]))
        if kind == "sel":
            return T.sel(f(term[1]), f(term[2]), f(term[3]))
        if kind == "mem":
            return T.mem(f(term[1]))
        if kind == "*":
            return T._fma(f(term[1]), f(term[2]), T.const(0))
        return T._binop(kind, f(term[1]), f(term[2]))

    def sweep(self, order: list[int], origin: dict[int, int]) -> None:
        T = self.terms
        for t in order:
            self.stats.swept += 1
            c = self.find(self.rebuild(t))
            if not np.array_equal(T.vals[c], T.vals[t]):
                raise AssertionError(f"checker bug: rebuilt term {T.show(c)} differs from {T.show(t)} on the samples")
            if c != t and t not in self.parent:
                self.parent[t] = c
            mask = self.mask.get(c, 0) | origin[t]
            self.mask[c] = mask
            cands = self.by_sig.setdefault(T.vals[c].tobytes(), [])
            reps = [self.find(u) for u in cands]
            if c in reps:
                continue
            if mask != BOTH:
                for u in reps:
                    if self.mask.get(u, 0) & ~mask == 0:
                        continue  # u is from the same program; nothing to align
                    self.stats.candidates += 1
                    if self.prove(c, u) == "unsat":
                        self.parent[c] = u
                        self.mask[u] = self.mask.get(u, 0) | mask
                        self.stats.merged += 1
                        c = u
                        break
            if self.find(c) == c and c not in reps and len(cands) < 8:
                cands.append(c)

    # ---- Z3 ------------------------------------------------------------------------------

    def _serialise(self, roots: tuple[int, int], deep: bool = False):
        """The query as a node list (cut points as interval-constrained variables).

        Returns (nodes, root indices, node intervals) or None if it exceeds the node budget.
        Node indices are assigned in DFS order, so queries equal up to renaming of cut points
        serialise alike (intervals are a function of the structure and the cut intervals).
        """
        T = self.terms
        index: dict[int, int] = {}
        nodes: list[tuple] = []
        ivs: list[tuple[int, int]] = []
        root_set = set(roots)

        def is_cut(n: int) -> bool:
            if n in root_set or self.mask.get(n, 0) != BOTH or T.defs[n][0] == "const":
                return False
            # Deep queries look through shared terms of bounded range (index arithmetic,
            # branch bits) and cut only at unbounded ones (hash values, loaded words).
            return not deep or T.iv[n] == TOP or T.defs[n][0] == "mem"

        out = []
        for root in roots:
            stack = [(root, False)]
            while stack:
                n, expanded = stack.pop()
                if n in index:
                    continue
                term = T.defs[n]
                if is_cut(n):
                    index[n] = len(nodes)
                    nodes.append(("var", *T.iv[n]))
                    ivs.append(T.iv[n])
                    continue
                kids = [self.find(k) for k in _children(term)]
                if not expanded:
                    stack.append((n, True))
                    stack.extend((k, False) for k in reversed(kids) if k not in index)
                    continue
                if len(nodes) >= self.node_budget:
                    return None
                ki = tuple(index[k] for k in kids)
                kind = term[0]
                if kind == "const":
                    nodes.append(("const", term[1]))
                elif kind == "lin":
                    nodes.append(("lin", term[1], tuple(zip(ki, (k for _, k in term[2])))))
                elif kind == "xor":
                    nodes.append(("xor", term[1], ki))
                elif kind == "mem":
                    nodes.append(("mem", *ki, *T.iv[n]))
                else:
                    nodes.append((kind, *ki))
                ivs.append(T.iv[n])
                index[n] = len(nodes) - 1
            out.append(index[root])
        return nodes, out, ivs

    def _z3(self, nodes: list[tuple], fixed: dict[int, int] | None = None) -> tuple[list, list]:
        """Z3 expressions of `nodes`; nodes in `fixed` are replaced by those constants."""
        bv = lambda v: z3.BitVecVal(v, 32)  # noqa: E731
        mem = z3.Function("mem", z3.BitVecSort(32), z3.BitVecSort(32))
        ex: list = []
        facts: list = []
        one, zero = bv(1), bv(0)
        fixed = fixed or {}
        for i, node in enumerate(nodes):
            kind = node[0]
            if i in fixed:
                e = bv(fixed[i])
            elif kind == "var":
                e = z3.BitVec(f"x{i}", 32)
                if (node[1], node[2]) != TOP:
                    facts += [z3.ULE(bv(node[1]), e), z3.ULE(e, bv(node[2]))]
            elif kind == "const":
                e = bv(node[1])
            elif kind == "lin":
                e = bv(node[1])
                for j, k in node[2]:
                    e = e + bv(k) * ex[j]
            elif kind == "xor":
                e = bv(node[1])
                for j in node[2]:
                    e = e ^ ex[j]
            elif kind == "eqz":
                e = z3.If(ex[node[1]] == zero, one, zero)
            elif kind == "sel":
                e = z3.If(ex[node[1]] != zero, ex[node[2]], ex[node[3]])
            elif kind == "mem":
                e = mem(ex[node[1]])
                if (node[2], node[3]) != TOP:
                    facts += [z3.ULE(e, bv(node[3]))]
            else:
                a, b = ex[node[1]], ex[node[2]]
                e = {"&": lambda: a & b, "|": lambda: a | b, "*": lambda: a * b, "<<": lambda: a << b,
                     ">>": lambda: z3.LShR(a, b), "//": lambda: z3.UDiv(a, b), "%": lambda: z3.URem(a, b),
                     "<": lambda: z3.If(z3.ULT(a, b), one, zero),
                     "cdiv": lambda: z3.Extract(31, 0, z3.UDiv(z3.ZeroExt(1, a) + z3.ZeroExt(1, b) - 1,
                                                                z3.ZeroExt(1, b)))}[kind]()
            ex.append(e)
        return ex, facts

    def prove(self, x: int, y: int) -> str:
        """'unsat' if x == y for all values of the cut points (a proof), else 'sat',
        'unknown' or 'too-large'.

        Cut strategies are tried in order: first every term shared by both programs is a
        cut point; if that is not enough (SAT), shared terms of bounded range are expanded
        (deep). Either UNSAT is a proof: any choice of cut points over-approximates.
        """
        result = "sat"
        for deep in (False, True):
            result = self._prove(x, y, deep)
            if result == "unsat":
                break
        return result

    def _prove(self, x: int, y: int, deep: bool) -> str:
        query = self._serialise((x, y), deep)
        if query is None:
            self.stats.too_large += 1
            return "too-large"
        nodes, (rx, ry), ivs = query
        key = (tuple(nodes), rx, ry)
        if key in self.memo:
            self.stats.cache_hits += 1
            return self.memo[key]
        self.stats.queries += 1
        self.stats.max_query_nodes = max(self.stats.max_query_nodes, len(nodes))
        # Case split on the select conditions known to be 0 or 1 (R20): every case is a
        # select-free query, and the cases are exhaustive, so all UNSAT is a proof. Z3 is
        # slow on sums of selects (the Mobius form) but instant once the conditions are fixed.
        conds = sorted({n[1] for n in nodes if n[0] == "sel" and ivs[n[1]][1] <= 1})
        if len(conds) > self.max_split_bits:
            conds = []
        t0 = time.perf_counter()
        result = "unsat"
        for case in range(1 << len(conds)):
            fixed = {c: (case >> i) & 1 for i, c in enumerate(conds)}
            ex, facts = self._z3(nodes, fixed)
            solver = z3.Solver()
            solver.set("timeout", self.timeout_ms)
            solver.add(*facts)
            # The fixed nodes must take their case values (a cut condition is otherwise free).
            whole, _ = self._z3(nodes) if fixed else (ex, None)
            solver.add(*[whole[c] == v for c, v in fixed.items()])
            solver.add(ex[rx] != ex[ry])
            result = str(solver.check())
            if result != "unsat":
                break
        dt = time.perf_counter() - t0
        self.stats.z3_seconds += dt
        self.stats.max_query_seconds = max(self.stats.max_query_seconds, dt)
        if result == "sat":
            self.stats.sat += 1
        elif result != "unsat":
            self.stats.unknown += 1
        self.memo[key] = result
        return result


# =============================================================================
# Top level
# =============================================================================


@dataclass
class Difference:
    address: int
    region: str
    value_a: int
    value_b: int
    image: list[int]
    sample: int


@dataclass
class Result:
    verdict: str  # "proved" | "counterexample" | "unproved"
    layout: Layout
    values_only: bool
    compared: int
    proved_words: int
    difference: Difference | None = None
    unproved: list[str] = field(default_factory=list)
    stats: SweepStats = field(default_factory=SweepStats)
    seconds: float = 0.0
    ops: tuple[int, int] = (0, 0)

    @property
    def ok(self) -> bool:
        return self.verdict == "proved"


def _reach(terms: NormTerms, roots) -> set[int]:
    seen: set[int] = set()
    stack = list(roots)
    while stack:
        t = stack.pop()
        if t in seen:
            continue
        seen.add(t)
        stack.extend(_children(terms.defs[t]))
    return seen


def check_equivalence(sched_a: dict, sched_b: dict, values_only: bool = False, samples: int = 32,
                      seed: int = 0, timeout_ms: int = 20000, node_budget: int = 4000,
                      shape: tuple[int, int, int] | None = None) -> Result:
    """Decide whether SSA program B writes the same final memory as SSA program A.

    # Arguments
    * `sched_a`, `sched_b` - scheduler JSONs (``--output-json``) of one problem shape
    * `values_only` - compare only the output values, not the indices
    * `samples` - random valid inputs used to pair terms and to find counterexamples
    * `seed` - seed of those inputs
    * `timeout_ms` - Z3 timeout per query (a timeout is "not proved", never a pass)
    * `node_budget` - largest query (term nodes) handed to Z3
    * `shape` - (rounds, batch_size, forest_height) for JSONs without run_options (both
      programs must be for this shape)

    # Returns
    A Result whose verdict is "proved", "counterexample" (with the first differing word and
    an input image that shows it, confirmed by a concrete interpreter) or "unproved".
    """
    t0 = time.perf_counter()
    layout = layout_of(sched_a, shape)
    if shape is None and layout != layout_of(sched_b):
        raise EquivalenceError("the two programs are for different problem shapes (rounds, batch_size, forest_height)")
    images = layout.sample_images(max(samples, 2), seed)
    terms = NormTerms(layout, images)
    stores_a, n_a = build_program(sched_a, terms, "A")
    stores_b, n_b = build_program(sched_b, terms, "B")

    words = layout.value_words() if values_only else sorted(set(stores_a) | set(stores_b))

    def final(stores, addr):
        return stores[addr][0] if addr in stores else terms.mem(terms.const(addr))

    pairs = [(addr, final(stores_a, addr), final(stores_b, addr)) for addr in words]
    reach_a = _reach(terms, [a for _, a, _ in pairs] + [v for v, _ in stores_a.values()])
    reach_b = _reach(terms, [b for _, _, b in pairs] + [v for v, _ in stores_b.values()])
    origin = {t: (PROGRAM_A if t in reach_a else 0) | (PROGRAM_B if t in reach_b else 0)
              for t in reach_a | reach_b}

    result = Result("unproved", layout, values_only, len(words), 0, ops=(n_a, n_b))
    problems = (memory_order_problems(sched_a, terms, stores_a, reach_a, layout, "A")
                + memory_order_problems(sched_b, terms, stores_b, reach_b, layout, "B"))

    sweeper = Sweeper(terms, timeout_ms, node_budget)
    sweeper.sweep(sorted(origin), origin)
    result.stats = sweeper.stats
    result.stats.terms = len(terms.defs)

    unproved = []
    for addr, a, b in pairs:
        ra, rb = sweeper.find(a), sweeper.find(b)
        if ra != rb:
            # Last attempt with the final representatives, unless a sample already separates
            # them (then the word is a counterexample and Z3 would only time out).
            same = np.array_equal(terms.vals[a], terms.vals[b])
            if same and sweeper.prove(ra, rb) == "unsat":
                sweeper.parent[rb] = ra
                continue
            unproved.append((addr, a, b))
    result.proved_words = len(words) - len(unproved)

    for addr, a, b in unproved:
        diff = np.nonzero(terms.vals[a] != terms.vals[b])[0]
        if len(diff):
            s = int(diff[0])
            image = [int(v) for v in images[s]]
            got_a, got_b = final_memory(sched_a, image), final_memory(sched_b, image)
            value_a, value_b = got_a.get(addr, image[addr]), got_b.get(addr, image[addr])
            if value_a == value_b:
                raise AssertionError(f"checker bug: sample {s} separates mem[{addr}] symbolically but not concretely")
            result.verdict = "counterexample"
            result.difference = Difference(addr, layout.region(addr), value_a, value_b, image, s)
            break
    else:
        result.unproved = [f"mem[{addr}] ({layout.region(addr)}): A {terms.show(a)}, B {terms.show(b)}"
                           for addr, a, b in unproved]
        result.unproved += problems
        if sweeper.stats.unknown:
            result.unproved.append(f"{sweeper.stats.unknown} Z3 quer(y/ies) returned unknown (timeout {timeout_ms} ms)")
        result.verdict = "proved" if not unproved and not problems else "unproved"
    result.seconds = time.perf_counter() - t0
    return result


def format_result(result: Result, name_a: str = "A", name_b: str = "B", limit: int = 5) -> str:
    L, s = result.layout, result.stats
    what = "values only (indices not compared)" if result.values_only else "every stored word"
    lines = [f"ssa equivalence: {name_a} ({result.ops[0]} ops) vs {name_b} ({result.ops[1]} ops); "
             f"{L.rounds} rounds, batch {L.batch_size}, height {L.forest_height}",
             f"  compared {result.compared} words: {what}"]
    if result.verdict == "proved":
        lines.append("  PROVED - B writes the same final memory as A for every input satisfying the preconditions")
    elif result.verdict == "counterexample":
        d = result.difference
        lines.append(f"  COUNTEREXAMPLE - first differing word mem[{d.address}] = {d.region}: "
                     f"A writes {d.value_a:#x}, B writes {d.value_b:#x}")
        lines.append(f"    on sample input {d.sample} (write it with --counterexample-out)")
    else:
        lines.append(f"  NOT PROVED - {result.proved_words}/{result.compared} words proved; no sample input separates the rest")
        lines += [f"    {u}" for u in result.unproved[:limit]]
        if len(result.unproved) > limit:
            lines.append(f"    … {len(result.unproved) - limit} more")
    lines.append(f"  {s.terms} terms, {s.swept} swept, {s.merged} merged by Z3 ({s.queries} queries, "
                 f"{s.cache_hits} answered from cache, {s.sat} sat, {s.unknown} unknown, {s.too_large} too large; "
                 f"{s.z3_seconds:.1f} s in Z3, slowest {s.max_query_seconds:.2f} s, largest {s.max_query_nodes} nodes); "
                 f"{result.seconds:.1f} s")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("a", help="scheduler JSON of the reference program (e.g. the default emission)")
    ap.add_argument("b", help="scheduler JSON of the transformed program")
    ap.add_argument("--values-only", action="store_true",
                    help="compare only the output values (for --values-only builds, which do not store indices)")
    ap.add_argument("--samples", type=int, default=32, help="random valid inputs (default 32)")
    ap.add_argument("--seed", type=int, default=0, help="seed of the sample inputs")
    ap.add_argument("--timeout-ms", type=int, default=20000, help="Z3 timeout per query")
    ap.add_argument("--node-budget", type=int, default=4000, help="largest Z3 query, in term nodes")
    ap.add_argument("--shape", nargs=3, type=int, metavar=("ROUNDS", "BATCH", "HEIGHT"),
                    help="problem shape, for JSONs written without run_options (by the binding directly)")
    ap.add_argument("--counterexample-out", metavar="PATH",
                    help="write the counterexample (initial memory image, word, both values) as JSON")
    args = ap.parse_args(argv)
    with open(args.a) as f:
        sched_a = json.load(f)
    with open(args.b) as f:
        sched_b = json.load(f)
    try:
        result = check_equivalence(sched_a, sched_b, args.values_only, args.samples, args.seed,
                                   args.timeout_ms, args.node_budget,
                                   tuple(args.shape) if args.shape else None)
    except EquivalenceError as e:
        print(f"ssa equivalence: error: {e}", file=sys.stderr)
        return 2
    print(format_result(result, args.a, args.b))
    if result.difference is not None and args.counterexample_out:
        d = result.difference
        with open(args.counterexample_out, "w") as f:
            json.dump({"address": d.address, "region": d.region, "value_a": d.value_a,
                       "value_b": d.value_b, "mem": d.image}, f)
        print(f"  counterexample written to {args.counterexample_out}")
    return {"proved": 0, "counterexample": 1}.get(result.verdict, 2)


if __name__ == "__main__":
    sys.exit(main())
