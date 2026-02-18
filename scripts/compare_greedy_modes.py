#!/usr/bin/env python3
"""
Compare PolicyOnly vs Full greedy schedules and report which CP-SAT constraints
the PolicyOnly (shorter, validating) schedule violates.

Slot limits are checked on *lowered* ops (what the Machine runs), not SSA ops—
ConstChoice/SelectChoice affect engine assignment after lowering.

Usage:
  uv run generate_ssa_kernel.py --rounds 3 --batch-size 64 --use-greedy --output-json policy_only.json
  uv run generate_ssa_kernel.py --rounds 3 --batch-size 64 --use-greedy --greedy-full-bounds --output-json full_greedy.json
  uv run generate_ssa_kernel.py --rounds 3 --batch-size 64 --cpsat-time-limit 30 --output-json cpsat_ref.json -q
  uv run python -m scripts.compare_greedy_modes policy_only.json full_greedy.json cpsat_ref.json

Or with defaults: policy_only.json, full_greedy.json, cpsat_ref.json
"""

import argparse
import json
import sys
from typing import Any


# Slot limits (from Engine::slot_limit)
SLOT_LIMITS = {"load": 2, "store": 2, "alu": 12, "valu": 6, "flow": 1}


def engine_from_str(s: str) -> str:
    return s.lower() if s else ""


def check_slot_limits_lowered(lowered_ops: list[dict]) -> list[str]:
    """Check per-cycle slot limits using lowered ops (what the Machine actually runs)."""
    violations = []
    if not lowered_ops:
        return violations
    max_cycle = max(op.get("cycle", -1) for op in lowered_ops)
    if max_cycle < 0:
        return violations
    n = max_cycle + 1
    alu_per = [0] * n
    valu_per = [0] * n
    load_per = [0] * n
    store_per = [0] * n
    flow_per = [0] * n

    for op in lowered_ops:
        cy = op.get("cycle", -1)
        if cy < 0 or cy >= n:
            continue
        engine = engine_from_str(op.get("engine", ""))
        if engine == "alu":
            alu_per[cy] += 1
        elif engine == "valu":
            valu_per[cy] += 1
        elif engine == "load":
            load_per[cy] += 1
        elif engine == "store":
            store_per[cy] += 1
        elif engine == "flow":
            flow_per[cy] += 1

    for cycle in range(n):
        if alu_per[cycle] > SLOT_LIMITS["alu"]:
            violations.append(
                f"ALU slot limit at cycle {cycle}: {alu_per[cycle]} > {SLOT_LIMITS['alu']}"
            )
        if valu_per[cycle] > SLOT_LIMITS["valu"]:
            violations.append(
                f"VALU slot limit at cycle {cycle}: {valu_per[cycle]} > {SLOT_LIMITS['valu']}"
            )
        if load_per[cycle] > SLOT_LIMITS["load"]:
            violations.append(
                f"Load slot limit at cycle {cycle}: {load_per[cycle]} > {SLOT_LIMITS['load']}"
            )
        if store_per[cycle] > SLOT_LIMITS["store"]:
            violations.append(
                f"Store slot limit at cycle {cycle}: {store_per[cycle]} > {SLOT_LIMITS['store']}"
            )
        if flow_per[cycle] > SLOT_LIMITS["flow"]:
            violations.append(
                f"Flow slot limit at cycle {cycle}: {flow_per[cycle]} > {SLOT_LIMITS['flow']}"
            )
    return violations


def dep_effective_cycle(
    dep_idx: int,
    dep_is_shifted: bool,
    cycle_assignments: list[int],
    shifted_alu_cycles: dict[int, list[int]],
    ops: list[dict],
) -> int:
    if dep_idx >= len(cycle_assignments):
        return -1
    if not dep_is_shifted:
        return cycle_assignments[dep_idx]
    op_id = ops[dep_idx]["id"] if dep_idx < len(ops) else None
    if op_id is None:
        return cycle_assignments[dep_idx]
    alu_cycles = shifted_alu_cycles.get(op_id)
    if alu_cycles:
        return max(alu_cycles)
    return cycle_assignments[dep_idx]


def check_constraints(
    ops: list[dict],
    cycle_assignments: list[int],
    shifted_alu_cycles: dict[int, list[int]],
    constraints: list[dict],
) -> list[tuple[str, str]]:
    """Check schedule against constraints. Returns [(constraint_type, message), ...]."""
    violations = []
    for c in constraints:
        if "Precedence" in c:
            p = c["Precedence"]
            consumer_idx = p["consumer_idx"]
            dep_idx = p["dep_idx"]
            result_offset = p["result_offset"]
            dep_is_shifted = p.get("dep_is_shifted", False)
            consumer_cycle = (
                cycle_assignments[consumer_idx]
                if consumer_idx < len(cycle_assignments)
                else -1
            )
            dep_cycle = dep_effective_cycle(
                dep_idx, dep_is_shifted, cycle_assignments, shifted_alu_cycles, ops
            )
            if consumer_cycle < dep_cycle + result_offset:
                violations.append(
                    (
                        "Precedence",
                        f"op {consumer_idx} cycle {consumer_cycle} < dep {dep_idx} result_at {dep_cycle + result_offset}",
                    )
                )
        elif "Ordering" in c:
            o = c["Ordering"]
            before_idx = o["before_idx"]
            after_idx = o["after_idx"]
            before_c = (
                cycle_assignments[before_idx]
                if before_idx < len(cycle_assignments)
                else -1
            )
            after_c = (
                cycle_assignments[after_idx]
                if after_idx < len(cycle_assignments)
                else -1
            )
            if before_c > after_c:
                violations.append(
                    (
                        "Ordering",
                        f"op {before_idx} cycle {before_c} > op {after_idx} cycle {after_c}",
                    )
                )
        elif "Domain" in c:
            d = c["Domain"]
            idx = d["idx"]
            lb = d["lb"]
            ub = d["ub"]
            cy = cycle_assignments[idx] if idx < len(cycle_assignments) else -1
            if cy < lb or cy > ub:
                violations.append(
                    (
                        "Domain",
                        f"op {idx} cycle {cy} not in [{lb}, {ub}]",
                    )
                )
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare PolicyOnly vs Full greedy and report constraint violations"
    )
    parser.add_argument(
        "policy_only",
        nargs="?",
        default="policy_only.json",
        help="PolicyOnly greedy output JSON",
    )
    parser.add_argument(
        "full_greedy",
        nargs="?",
        default="full_greedy.json",
        help="Full greedy output JSON",
    )
    parser.add_argument(
        "cpsat_ref",
        nargs="?",
        default="cpsat_ref.json",
        help="CP-SAT output JSON (has schedule_constraints)",
    )
    args = parser.parse_args()

    def load(p: str) -> dict:
        try:
            with open(p) as f:
                return json.load(f)
        except OSError as e:
            print(f"Error: cannot read {p}: {e}", file=sys.stderr)
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: invalid JSON in {p}: {e}", file=sys.stderr)
            sys.exit(1)

    policy = load(args.policy_only)
    full = load(args.full_greedy)
    cpsat = load(args.cpsat_ref)

    constraints = cpsat.get("schedule_constraints")
    if not constraints:
        print("Error: cpsat_ref has no schedule_constraints. Run with CP-SAT.", file=sys.stderr)
        return 1

    ssa_ops = policy.get("ssa_ops") or cpsat.get("ssa_ops")
    if not ssa_ops:
        print("Error: no ssa_ops in policy or cpsat_ref", file=sys.stderr)
        return 1

    # Cycle assignments from PolicyOnly
    policy_cycles = [op.get("cycle", -1) for op in ssa_ops]
    full_cycles = [op.get("cycle", -1) for op in (full.get("ssa_ops") or ssa_ops)]

    print("=== Schedule comparison ===")
    p_cycles = max(policy_cycles) + 1 if policy_cycles else 0
    f_cycles = max(full_cycles) + 1 if full_cycles else 0
    print(f"PolicyOnly: {p_cycles} cycles")
    print(f"Full greedy: {f_cycles} cycles")
    n = min(len(policy_cycles), len(full_cycles))
    if n > 0:
        diff = sum(1 for i in range(n) if policy_cycles[i] != full_cycles[i])
        print(f"Ops with different cycle (first {n}): {diff} / {n}")
    print()

    # Slot limits: use lowered ops (what the Machine runs). SSA ops would overcount
    # Load (ConstChoice→add_imm counted as load) and mis-count Flow/VALU.
    slot_violations = check_slot_limits_lowered(policy.get("ops", []))
    print("=== Slot limit violations (PolicyOnly) ===")
    if slot_violations:
        for v in slot_violations[:20]:
            print(f"  {v}")
        if len(slot_violations) > 20:
            print(f"  ... and {len(slot_violations) - 20} more")
    else:
        print("  None")
    print()

    # Domain, Ordering, Precedence. Policy has no shifted_alu_cycles; use empty so
    # dep_effective = policy_cycles[dep_idx] (one cycle per op in greedy output).
    constraint_violations = check_constraints(
        ssa_ops, policy_cycles, {}, constraints
    )
    print("=== Constraint violations (PolicyOnly vs CP-SAT model) ===")
    by_type: dict[str, list[str]] = {}
    for ctype, msg in constraint_violations:
        by_type.setdefault(ctype, []).append(msg)
    for ctype in ["Domain", "Ordering", "Precedence"]:
        msgs = by_type.get(ctype, [])
        print(f"  {ctype}: {len(msgs)} violations")
        for m in msgs[:5]:
            print(f"    {m}")
        if len(msgs) > 5:
            print(f"    ... and {len(msgs) - 5} more")
    if not constraint_violations:
        print("  None")
    print()

    total = len(slot_violations) + len(constraint_violations)
    if total > 0:
        print("Summary: PolicyOnly violates slot limits (lowered ops) or model constraints.")
    else:
        print("Summary: PolicyOnly schedule satisfies slot limits and checked constraints.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
