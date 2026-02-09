#!/usr/bin/env python3
"""Unified bisection across choice, constraint, and size axes.

Bisect CP-SAT validation failures by sweeping any combination of:
- Choice: const_choice, select_choice, valu_shift (instruction encoding options)
- Constraints: which optional CP-SAT constraint groups to disable
- Size: rounds, batch_size, forest_height (problem size)

Usage (from project root):
  # Which choice options cause failure? (default 8 combinations, backward compat with bisect_cpsat_validation)
  uv run python scripts/bisect.py --sweep-choice

  # Sweep choice with explicit values (e.g. include fma)
  uv run python scripts/bisect.py --sweep-choice --const-choice optimize load --select-choice optimize fma vselect --valu-shift alu valu

  # Which constraint group correlates with failure? (none + each single group disabled)
  uv run python scripts/bisect.py --sweep-constraints single

  # Sweep size (rounds × batch × forest_height) with fixed select=fma, valu=valu
  uv run python scripts/bisect.py --sweep-size --rounds 1 2 --batch-size 8 16 32 --forest-height 8 9 10 --select-choice fma --valu-shift valu

  # Shrink to find minimal failing config
  uv run python scripts/bisect.py --shrink --select-choice fma --valu-shift valu

  # Fixed config only (single run)
  uv run python scripts/bisect.py --rounds 2 --batch-size 16
"""

import argparse
import itertools
import os
import sys

# Allow importing bisect_common when run as uv run python scripts/bisect.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bisect_common import run_one


def _constraint_group_names():
    try:
        import scheduler_rs
        return list(scheduler_rs.optional_constraint_group_names())
    except Exception:
        return [
            "redundant_linear_bounds",
            "capacity_valu",
            "capacity_alu",
            "capacity_flow",
            "capacity_load",
        ]


def _choice_axis_default():
    """Default grid for --sweep-choice (backward compat: 8 combos like bisect_cpsat_validation)."""
    return (
        ["optimize", "load"],   # const_choice
        ["optimize", "vselect"],  # select_choice
        ["alu", "valu"],         # valu_shift
    )


def _constraint_axis_single():
    """(label, disable_constraints) for sweep-constraints single: none + each group disabled."""
    names = _constraint_group_names()
    yield ("none", None)
    for g in names:
        yield (f"disable_{g}", [g])


def _constraint_axis_all():
    """Single run with all constraint groups disabled."""
    return [("disable_all", _constraint_group_names())]


def main():
    parser = argparse.ArgumentParser(
        description="Bisect validation failures across choice, constraint, and/or size axes"
    )
    # Base config (single value when not sweeping size)
    parser.add_argument("--rounds", type=int, nargs="+", default=[2], help="Rounds (default: 2)")
    parser.add_argument("--batch-size", type=int, nargs="+", default=[16], help="Batch size, multiple of 8 (default: 16)")
    parser.add_argument("--forest-height", type=int, nargs="+", default=[10], help="Forest height (default: 10)")
    parser.add_argument("--cpsat-time-limit", type=float, default=60.0, help="CP-SAT time limit per run (default: 60)")
    parser.add_argument("--seed", type=int, default=123, help="Random seed (default: 123)")
    parser.add_argument("--use-greedy", action="store_true", help="Run greedy only (one run, no CP-SAT)")

    # Sweep axes
    parser.add_argument("--sweep-choice", action="store_true", help="Sweep const/select/valu choice options")
    parser.add_argument(
        "--const-choice",
        type=str,
        nargs="*",
        default=None,
        metavar="C",
        help="Const choice values (with --sweep-choice). Default: optimize load",
    )
    parser.add_argument(
        "--select-choice",
        type=str,
        nargs="*",
        default=None,
        metavar="S",
        help="Select choice values (with --sweep-choice) or single fixed value. Default sweep: optimize vselect",
    )
    parser.add_argument(
        "--valu-shift",
        type=str,
        nargs="*",
        default=None,
        metavar="V",
        help="VALU shift values (with --sweep-choice) or single fixed value. Default sweep: alu valu",
    )
    parser.add_argument(
        "--sweep-constraints",
        nargs="*",
        default=None,
        metavar="MODE|SET",
        help="Sweep constraint groups: none (1 run), single (none + each group disabled), all (all disabled), or list of disable sets e.g. capacity_valu capacity_valu,capacity_alu",
    )
    parser.add_argument(
        "--disable-cpsat-constraints",
        type=str,
        default=None,
        metavar="G1,G2",
        help="Fixed disable set when not sweeping constraints (e.g. capacity_valu,capacity_alu)",
    )
    parser.add_argument("--sweep-size", action="store_true", help="Sweep rounds × batch_size × forest_height grid")
    parser.add_argument(
        "--shrink",
        action="store_true",
        help="Shrink from reference (2,16,10): vary one dimension at a time, report pass/fail",
    )

    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = os.path.join(project_root, "generate_ssa_kernel.py")
    if not os.path.isfile(script):
        print(f"Error: generate_ssa_kernel.py not found at {script}", file=sys.stderr)
        return 1

    # Validate batch sizes (must be multiple of 8)
    for b in args.batch_size:
        if b % 8 != 0:
            print(f"Error: batch_size must be a multiple of 8 (VLEN). Got {b}.", file=sys.stderr)
            return 1

    # Resolve single-value vs sweep for choice
    if args.sweep_choice:
        const_list = args.const_choice if args.const_choice is not None else _choice_axis_default()[0]
        select_list = args.select_choice if args.select_choice is not None else _choice_axis_default()[1]
        valu_list = args.valu_shift if args.valu_shift is not None else _choice_axis_default()[2]
        choice_tuples = list(itertools.product(const_list, select_list, valu_list))
        choice_axis = [
            (f"const={c} select={s} shift={v}", {"const_choice": c, "select_choice": s, "valu_shift": v})
            for c, s, v in choice_tuples
        ]
    else:
        const = (args.const_choice or ["optimize"])[0]
        select = (args.select_choice or ["optimize"])[0]
        valu = (args.valu_shift or ["optimize"])[0]
        choice_axis = [("", {"const_choice": const, "select_choice": select, "valu_shift": valu})]

    # Constraint axis
    if args.sweep_constraints is not None:
        if len(args.sweep_constraints) == 0:
            # --sweep-constraints with no args: treat as "single"
            constraint_axis = list(_constraint_axis_single())
        elif len(args.sweep_constraints) == 1:
            mode = args.sweep_constraints[0].lower()
            if mode == "none":
                constraint_axis = [("none", None)]
            elif mode == "single":
                constraint_axis = list(_constraint_axis_single())
            elif mode == "all":
                constraint_axis = _constraint_axis_all()
            else:
                # Single disable set
                constraint_axis = [(mode, [s.strip() for s in mode.split(",") if s.strip()])]
        else:
            # Explicit list of disable sets
            constraint_axis = []
            for i, s in enumerate(args.sweep_constraints):
                groups = [x.strip() for x in s.split(",") if x.strip()]
                label = s if len(s) < 40 else f"set{i}"
                constraint_axis.append((label, groups if groups else None))
            # If any entry is "none", use None for disable_constraints
            constraint_axis = [(lbl, None if grps is not None and len(grps) == 0 else grps) for lbl, grps in constraint_axis]
            # Fix: "none" string could have been passed as one set
            for i, (lbl, grps) in enumerate(constraint_axis):
                if grps is not None and len(grps) == 1 and grps[0].lower() == "none":
                    constraint_axis[i] = (lbl, None)
    else:
        fixed_disable = None
        if args.disable_cpsat_constraints:
            fixed_disable = [s.strip() for s in args.disable_cpsat_constraints.split(",") if s.strip()]
        constraint_axis = [("", fixed_disable)]

    # Size axis
    if args.shrink:
        # Handled in separate code path below
        size_axis = None
    elif args.sweep_size:
        rounds_list = sorted(set(args.rounds))
        batch_list = sorted(set(args.batch_size))
        height_list = sorted(set(args.forest_height))
        size_axis = [
            (f"r={r} b={b} h={h}", {"rounds": r, "batch_size": b, "forest_height": h})
            for r in rounds_list
            for b in batch_list
            for h in height_list
        ]
    else:
        r, b, h = args.rounds[0], args.batch_size[0], args.forest_height[0]
        size_axis = [("", {"rounds": r, "batch_size": b, "forest_height": h})]

    # ---- Shrink mode ----
    if args.shrink:
        r0, b0, h0 = 2, 16, 10
        const = (args.const_choice or ["optimize"])[0]
        select = (args.select_choice or ["fma"])[0]
        valu = (args.valu_shift or ["valu"])[0]
        fixed_disable = None
        if args.disable_cpsat_constraints:
            fixed_disable = [s.strip() for s in args.disable_cpsat_constraints.split(",") if s.strip()]

        print("Shrink mode: try smaller configs to find minimal failing (rounds, batch_size, forest_height)")
        print(f"  Reference: rounds={r0}, batch_size={b0}, forest_height={h0}")
        print(f"  Choice: const={const} select={select} shift={valu}")
        print()

        base_kw = {
            "rounds": r0,
            "batch_size": b0,
            "forest_height": h0,
            "use_greedy": False,
            "cpsat_time_limit": args.cpsat_time_limit,
            "seed": args.seed,
            "const_choice": const,
            "select_choice": select,
            "valu_shift": valu,
            "disable_constraints": fixed_disable,
            "validate": True,
            "quiet": True,
        }

        ok, _ = run_one(project_root, script, use_greedy=True, **{**base_kw, "use_greedy": True})
        if not ok:
            print("  Greedy failed at reference; cannot use as baseline.", file=sys.stderr)
            return 1

        print("  Varying batch_size (rounds=2, forest_height=10):")
        for batch in [16, 8, 4, 2]:
            if batch % 8 != 0:
                continue
            ok, _ = run_one(project_root, script, **{**base_kw, "batch_size": batch})
            print(f"    batch_size={batch}: CP-SAT {'PASS' if ok else 'FAIL'}")

        print("  Varying rounds (batch_size=16, forest_height=10):")
        for rounds in [2, 1]:
            ok, _ = run_one(project_root, script, **{**base_kw, "rounds": rounds})
            print(f"    rounds={rounds}: CP-SAT {'PASS' if ok else 'FAIL'}")

        print("  Varying forest_height (rounds=2, batch_size=16):")
        for height in [10, 9, 8, 7]:
            ok, _ = run_one(project_root, script, **{**base_kw, "forest_height": height})
            print(f"    forest_height={height}: CP-SAT {'PASS' if ok else 'FAIL'}")

        print()
        print("To reproduce minimal failure, use the smallest failing combo, e.g.:")
        print("  uv run python scripts/bisect.py --rounds 2 --batch-size 16 --select-choice fma --valu-shift valu")
        return 0

    # ---- Product of axes ----
    if args.use_greedy:
        # Single run
        (clbl, ckw) = choice_axis[0]
        (conlbl, condisable) = constraint_axis[0]
        (slbl, skw) = size_axis[0]
        label = " ".join(filter(None, [clbl, conlbl, slbl])).strip() or "greedy"
        kwargs = {**ckw, **skw, "disable_constraints": condisable, "use_greedy": True, "cpsat_time_limit": args.cpsat_time_limit, "seed": args.seed, "validate": True, "quiet": True}
        print(f"Greedy (baseline): {label}...", flush=True)
        ok, _ = run_one(project_root, script, **kwargs)
        print(f"  {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1

    # Optional greedy baseline when sweeping choice (like bisect_cpsat_validation)
    if args.sweep_choice and not args.sweep_size and len(constraint_axis) == 1 and constraint_axis[0][1] is None:
        (_, skw) = size_axis[0]
        print("Greedy (baseline)...", flush=True)
        ok, _ = run_one(
            project_root,
            script,
            **skw,
            const_choice="optimize",
            select_choice="optimize",
            valu_shift="optimize",
            disable_constraints=None,
            use_greedy=True,
            cpsat_time_limit=args.cpsat_time_limit,
            seed=args.seed,
            validate=True,
            quiet=True,
        )
        print(f"  Greedy (baseline): {'PASS' if ok else 'FAIL'}")
        if not ok:
            print("  Greedy failed; fix baseline before bisecting.", file=sys.stderr)
            return 1
        print()

    results = []
    for (clbl, ckw) in choice_axis:
        for (conlbl, condisable) in constraint_axis:
            for (slbl, skw) in size_axis:
                label_parts = [p for p in (clbl, conlbl, slbl) if p]
                label = " | ".join(label_parts) if label_parts else "run"
                kwargs = {
                    **ckw,
                    **skw,
                    "disable_constraints": condisable,
                    "use_greedy": False,
                    "cpsat_time_limit": args.cpsat_time_limit,
                    "seed": args.seed,
                    "validate": True,
                    "quiet": True,
                }
                print(f"  {label}...", end=" ", flush=True)
                ok, _ = run_one(project_root, script, **kwargs)
                status = "PASS" if ok else "FAIL"
                results.append((label, status))
                print(status)

    # Summary
    print()
    passed = sum(1 for _, s in results if s == "PASS")
    failed_list = [lbl for lbl, s in results if s == "FAIL"]
    print(f"Summary: {passed}/{len(results)} passed.")
    if failed_list:
        print("Failing configs:")
        for lbl in failed_list:
            print(f"  {lbl}")
    return 0 if not failed_list else 1


if __name__ == "__main__":
    sys.exit(main())
