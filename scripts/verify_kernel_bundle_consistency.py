#!/usr/bin/env python3
"""Verify that every op in the scheduler JSON appears in the emitted kernel at exactly op.cycle.

Use after generating a kernel with both --output-json and --save-kernel from the same run:

  uv run generate_ssa_kernel.py ... --output-json sched.json --save-kernel kern.json
  uv run python scripts/verify_kernel_bundle_consistency.py --kernel kern.json --scheduler sched.json

Exits 0 if every op is in the correct bundle; else prints mismatches and exits 1.
Output directory (for plan): artifacts/root_cause/1A_bundle_consistency/
"""

import argparse
import json
import os
import sys

# Project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_kernel_bundles(path):
    """Load kernel JSON; return (list of bundles for cycle 0,1,..., total_cycles), total_cycles.

    Kernel format: first and last elements are pause wrappers; middle elements are
    cycle 0, cycle 1, ... So bundle at cycle c is at index 1+c.
    """
    with open(path) as f:
        data = json.load(f)
    raw = data.get("instrs", data)
    if not raw:
        return [], 0
    # Strip leading and trailing pause bundles
    start = 0
    end = len(raw)
    if len(raw) >= 1 and _is_pause_bundle(raw[0]):
        start = 1
    if len(raw) >= 2 and _is_pause_bundle(raw[-1]):
        end = len(raw) - 1
    bundles = [raw[i] for i in range(start, end)]
    return bundles, len(bundles)


def _is_pause_bundle(bundle):
    """True if bundle is only flow with ['pause']."""
    if not isinstance(bundle, dict):
        return False
    flow = bundle.get("flow")
    if not flow or len(flow) != 1:
        return False
    slot = flow[0]
    return isinstance(slot, (list, tuple)) and len(slot) >= 1 and slot[0] == "pause"


def slot_matches(sched_slot, kernel_slot):
    """Compare scheduler slot (list of str from JSON) with kernel slot (list of str/int)."""
    if len(sched_slot) != len(kernel_slot):
        return False
    for a, b in zip(sched_slot, kernel_slot):
        # Scheduler JSON serializes slot as Vec<String> (all elements string).
        # Kernel may have ints (e.g. scratch addresses). Normalize to string.
        if str(a) != str(b):
            return False
    return True


def _assign_ops_to_bundles_one_to_one(ops, bundles):
    """One-to-one assignment: each op is assigned to at most one kernel slot at (op.cycle, op.engine).
    Returns a dict op_id -> actual_cycle (or None if not assigned, or 'duplicate' if no free slot).
    """
    from collections import defaultdict
    # Group ops by (cycle, engine)
    by_ce = defaultdict(list)
    for op in ops:
        c, e = op.get("cycle", -1), op.get("engine", "")
        if c < 0 or c >= len(bundles):
            continue
        by_ce[(c, e)].append(op)

    result = {}  # op id -> actual_cycle or None
    for (c, e), group in by_ce.items():
        bundle = bundles[c] if c < len(bundles) else {}
        slots = list(bundle.get(e, []))  # list of kernel slots at this (cycle, engine)
        used = [False] * len(slots)
        for op in group:
            op_id = op.get("id", "?")
            slot = op.get("slot", [])
            if not slot:
                result[op_id] = None
                continue
            found = False
            for i, kslot in enumerate(slots):
                if used[i]:
                    continue
                klist = list(kslot) if isinstance(kslot, (list, tuple)) else [kslot]
                if slot_matches(slot, klist):
                    used[i] = True
                    result[op_id] = c
                    found = True
                    break
            if not found:
                result[op_id] = None  # no matching slot or all taken
    return result


def run_consistency_check(kernel_path, scheduler_path, output_dir=None):
    """Run the consistency check. Returns (passed: bool, list of mismatch messages)."""
    bundles, total_cycles = load_kernel_bundles(kernel_path)
    with open(scheduler_path) as f:
        sched = json.load(f)
    ops = sched.get("ops", [])
    # Kernel cycle count vs JSON: max(op.cycle)+1 should equal number of bundles (1I)
    max_cycle = max((op.get("cycle", -1) for op in ops), default=-1)
    expected_bundles = max_cycle + 1 if max_cycle >= 0 else 0
    if ops and total_cycles != expected_bundles:
        mismatches = [
            f"Kernel bundle count {total_cycles} != max(op.cycle)+1 = {expected_bundles}"
        ]
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "report.txt"), "w") as f:
                f.write("\n".join(mismatches) + "\n")
        return False, mismatches
    if not ops:
        return True, []  # vacuously consistent

    assignment = _assign_ops_to_bundles_one_to_one(ops, bundles)
    mismatches = []
    for op in ops:
        op_id = op.get("id", "?")
        expected_cycle = op.get("cycle", -1)
        actual = assignment.get(op_id)
        if actual is None:
            mismatches.append(f"op id={op_id} expected_cycle={expected_cycle}: NOT FOUND in kernel")
            continue
        if actual != expected_cycle:
            mismatches.append(
                f"op id={op_id} expected_cycle={expected_cycle} actual_cycle={actual}"
            )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, "report.txt")
        with open(report_path, "w") as f:
            f.write(f"Kernel: {kernel_path}\nScheduler: {scheduler_path}\n")
            f.write(f"Total cycles (bundles): {total_cycles}\n")
            f.write(f"Ops in scheduler: {len(ops)}\n")
            if mismatches:
                f.write(f"Mismatches: {len(mismatches)}\n")
                for m in mismatches:
                    f.write(m + "\n")
            else:
                f.write("PASS: all ops in correct bundle.\n")

    return len(mismatches) == 0, mismatches


def main():
    parser = argparse.ArgumentParser(
        description="Verify kernel bundle consistency with scheduler JSON (same run)."
    )
    parser.add_argument("--kernel", "-k", required=True, help="Path to kernel JSON (--save-kernel)")
    parser.add_argument("--scheduler", "-s", required=True, help="Path to scheduler JSON (--output-json)")
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help="Write report to this dir (e.g. artifacts/root_cause/1A_bundle_consistency/)",
    )
    args = parser.parse_args()

    passed, mismatches = run_consistency_check(
        args.kernel, args.scheduler, output_dir=args.output_dir
    )
    if mismatches:
        for m in mismatches:
            print(m)
        print(f"\nTotal mismatches: {len(mismatches)}")
        return 1
    print("PASS: every op appears in the bundle for exactly op.cycle")
    return 0


if __name__ == "__main__":
    sys.exit(main())
