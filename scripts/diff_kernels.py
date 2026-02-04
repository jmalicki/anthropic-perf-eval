#!/usr/bin/env python3
"""Compare two kernel JSON files (e.g. passing vs failing) to aid debugging.

Usage (from project root):
  uv run python scripts/diff_kernels.py kernel_vselect.json kernel_fma.json
  uv run python scripts/diff_kernels.py kernel_to_validate.json other.json --brief
"""

import argparse
import json
import sys


def load_instrs(path):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "instrs" in data:
        return data["instrs"], data.get("rounds"), data.get("batch_size")
    return data, None, None


def bundle_repr(bundle):
    """Normalize a bundle to a comparable string (sorted engines, then ops)."""
    parts = []
    for engine in sorted(bundle.keys()):
        ops = bundle[engine]
        for op in ops:
            op_str = " ".join(str(x) for x in op)
            parts.append(f"  {engine}: {op_str}")
    return "\n".join(parts) if parts else "  (empty)"


def main():
    parser = argparse.ArgumentParser(
        description="Diff two kernel JSON files (e.g. passing vs failing config)."
    )
    parser.add_argument("a", help="First kernel JSON path (e.g. passing)")
    parser.add_argument("b", help="Second kernel JSON path (e.g. failing)")
    parser.add_argument(
        "--brief",
        action="store_true",
        help="Only print first difference and summary",
    )
    args = parser.parse_args()

    instrs_a, rounds_a, batch_a = load_instrs(args.a)
    instrs_b, rounds_b, batch_b = load_instrs(args.b)

    n_a, n_b = len(instrs_a), len(instrs_b)
    print(f"  {args.a}: {n_a} cycles (rounds={rounds_a}, batch_size={batch_a})")
    print(f"  {args.b}: {n_b} cycles (rounds={rounds_b}, batch_size={batch_b})")
    if n_a != n_b:
        print(f"  Length mismatch: {n_a} vs {n_b}")

    first_diff = None
    diff_count = 0
    for i in range(min(n_a, n_b)):
        ba, bb = instrs_a[i], instrs_b[i]
        ra, rb = bundle_repr(ba), bundle_repr(bb)
        if ra != rb:
            diff_count += 1
            if first_diff is None:
                first_diff = i
            if not args.brief:
                print(f"\n--- Cycle {i} ---")
                print(f"  {args.a}:")
                print(ra)
                print(f"  {args.b}:")
                print(rb)
    if first_diff is not None:
        print(f"\nFirst difference at cycle {first_diff}; total differing cycles: {diff_count}")
        if args.brief and first_diff is not None:
            print(f"\nCycle {first_diff} detail:")
            print(f"  {args.a}:")
            print(bundle_repr(instrs_a[first_diff]))
            print(f"  {args.b}:")
            print(bundle_repr(instrs_b[first_diff]))
    else:
        if n_a != n_b:
            print("\nBundles match up to min length; trailing cycles differ.")
        else:
            print("\nKernels are identical (same cycle count and bundle content).")
    return 0 if diff_count == 0 and n_a == n_b else 1


if __name__ == "__main__":
    sys.exit(main())
