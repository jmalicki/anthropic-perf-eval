"""Shared symbolic variables and helpers for Z3 liveness verification."""

from z3 import And, Int, Ints

# Timeout in milliseconds (1 minute)
DEFAULT_TIMEOUT_MS = 60_000


def overlaps_closed(s1: "z3.ArithRef", e1: "z3.ArithRef", s2: "z3.ArithRef", e2: "z3.ArithRef"):
    """Overlap predicate for closed intervals [s1,e1] and [s2,e2].
    Matches scheduler_rs LiveInterval::overlaps: e1 > s2 and e2 > s1.
    Adjacent intervals [0,5] and [5,10] do NOT overlap."""
    return And(s1 < e2, s2 < e1)


def fresh_interval_vars(prefix: str = "i"):
    """Create symbolic vars for closed interval [start, end]."""
    return Ints(f"{prefix}_start {prefix}_end")


def fresh_cycle_var(name: str = "t"):
    """Create a symbolic cycle (non-negative int)."""
    return Int(name)
