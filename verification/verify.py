"""CLI to run Z3 verification of liveness theorems."""

import argparse
import sys
import time

import z3

from .fixtures import DEFAULT_TIMEOUT_MS
from .theorems import (
    check_t1_expiration_no_write_conflict,
    check_t2_overlap_definition,
    check_t3_asap_bounds,
    check_t4_alap_bounds,
    check_t6_half_open_conversion,
    check_t9_expiration_implies_no_overlap,
    check_t10_dead_shifted_valu_end,
)

THEOREM_CHECKS = [
    ("T2", "Overlap definition", check_t2_overlap_definition),
    ("T1", "Expiration rule (no write-write conflict)", check_t1_expiration_no_write_conflict),
    ("T3", "ASAP bounds (precedence)", check_t3_asap_bounds),
    ("T4", "ALAP bounds (precedence)", check_t4_alap_bounds),
    ("T6", "Half-open conversion", check_t6_half_open_conversion),
    ("T9", "Expiration implies no overlap", check_t9_expiration_implies_no_overlap),
    ("T10", "Dead shifted VALU dest end", check_t10_dead_shifted_valu_end),
]


def run_verification(
    *,
    theorems: list[str] | None = None,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
    verbose: bool = True,
    seed: int = 42,
) -> int:
    """Run Z3 verification. Returns 0 if all pass, 1 otherwise."""
    z3.set_param("timeout", timeout_ms)
    try:
        z3.set_param("sat.random_seed", seed)
    except (z3.Z3Exception, TypeError):
        pass  # seed optional
    if verbose:
        z3.set_param("verbose", 1)
        z3.set_param("stats", True)
    print(f"Z3 verification (timeout={timeout_ms}ms, seed={seed})")
    print("-" * 60)
    failures = []
    for name, desc, check_fn in THEOREM_CHECKS:
        if theorems and name not in theorems:
            continue
        print(f"\n[{name}] {desc}...", end=" ", flush=True)
        t0 = time.perf_counter()
        try:
            result_name, result_str, _, stats = check_fn()
            elapsed = time.perf_counter() - t0
        except Exception as e:
            print(f"ERROR: {e}")
            failures.append((name, str(e)))
            continue
        passed = result_str == "unsat"
        status = "PASS" if passed else "FAIL"
        print(f"{status} ({result_str}) in {elapsed:.3f}s")
        if verbose and stats:
            for k, v in sorted(stats.items())[:8]:
                print(f"    {k}: {v}")
        if not passed:
            failures.append((name, result_str))
    print("\n" + "-" * 60)
    if failures:
        print(f"FAILED: {len(failures)} theorem(s)")
        for name, msg in failures:
            print(f"  - {name}: {msg}")
        return 1
    print("All theorems verified (UNSAT = proven).")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Z3 verification of liveness spec theorems")
    parser.add_argument(
        "-t",
        "--theorem",
        action="append",
        help="Run only these theorems (e.g. -t T1 -t T2)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_MS,
        help=f"Timeout in ms (default: {DEFAULT_TIMEOUT_MS})",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Less output (no Z3 stats)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for Z3")
    args = parser.parse_args()
    return run_verification(
        theorems=args.theorem,
        timeout_ms=args.timeout,
        verbose=not args.quiet,
        seed=args.seed,
    )


if __name__ == "__main__":
    sys.exit(main())
