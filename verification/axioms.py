"""Axioms A1–A7 from LIVENESS_SPECIFICATION.md."""

# A1 (VLIW order): Read-before-write within cycle. Encoded implicitly via
# half-open intervals and reuse rules; no explicit formula needed.
#
# A2 (Reuse rule): Encoded per-theorem when needed.
# A3 (Consumer readiness): Precedence; encoded per-theorem.
# A4 (End by VReg kind): Schema; encoded per-theorem.
# A5 (Completion cycle): Encoded per-theorem.
# A6 (Slot capacity): For T7; deferred.
# A7 (Scratch budget): Overlapping intervals => disjoint ranges; encoded in T2/T9.


def axioms_for_t2():
    """No extra axioms needed for T2 (overlap definition is self-contained)."""
    return []


def axioms_for_t1():
    """Axioms needed for T1 (expiration rule): A2, A4."""
    return []  # Will be inlined in theorem encoding
