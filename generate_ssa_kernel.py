#!/usr/bin/env python3
"""Generate optimized kernel using SSA builder with CP-SAT.

This is the recommended approach for generating high-quality kernels.
The SSA builder eliminates false dependencies, and CP-SAT finds optimal schedules.

Default settings:
- enable_valu_shift=True (VALU→ALU shifting for better slot use)
- use_cpsat=True (optimal scheduling)
- cpsat_time_limit=600 (10 minutes - good default)
- validate=True (validate against reference; use --no-validate to skip)

Example usage:
    # Default (CP-SAT with VALU shift, validation on)
    python generate_ssa_kernel.py

    # Quick iteration with greedy scheduler (no validation)
    python generate_ssa_kernel.py --use-greedy --no-validate

    # Longer optimization
    python generate_ssa_kernel.py --cpsat-time-limit 3600

    # Kernel written to kernel_to_validate.json when validating; re-validate without rebuilding:
    python generate_ssa_kernel.py --validate-only kernel_to_validate.json
"""

import argparse
import json
import os
import random
import sys
import time

from kernel_build_args import (
    DEFAULT_SCHEDULER_JSON,
    add_kernel_build_args,
    args_to_build_kwargs,
    build_kernel_with_args,
    n_nodes_for_height,
)

BASELINE = 147734


def main():
    parser = argparse.ArgumentParser(
        description="Generate optimized kernel using SSA builder with CP-SAT"
    )
    add_kernel_build_args(parser)

    # Script-specific options (not passed to Rust)
    parser.add_argument("--validate", action="store_true", default=True,
                        help="Validate against reference kernel (default: True). Kernel is always saved before validation for debugging.")
    parser.add_argument("--no-validate", action="store_false", dest="validate",
                        help="Skip validation (faster for iteration).")
    parser.add_argument("--no-dataflow-check", action="store_false", dest="dataflow_check", default=True,
                        help="With --validate, skip the symbolic kernel-vs-SSA replay (verification/kernel_dataflow.py)")
    parser.add_argument("--validate-only", type=str, metavar="PATH", default=None,
                        help="Load kernel from JSON and run validation only (no build). Use same forest_height, batch_size, rounds, seed.")
    parser.add_argument("--save-kernel", type=str, metavar="PATH", default=None,
                        help="Save instruction list to JSON (default when --validate: kernel_to_validate.json). Rerun with --validate-only PATH.")
    parser.add_argument("--seed", type=int, default=123,
                        help="Random seed for validation (default: 123)")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress verbose output from Rust scheduler")

    args = parser.parse_args()

    # Validate-only: load kernel and run validation, then exit
    if args.validate_only:
        return run_validate_only(
            args.validate_only,
            args.forest_height,
            args.batch_size,
            args.rounds,
            args.seed,
        )

    use_cpsat = not args.use_greedy
    verbose = not args.quiet  # Show Rust scheduler progress by default

    # Derive n_nodes if not set
    n_nodes_val = getattr(args, "n_nodes", None)
    if n_nodes_val is None:
        n_nodes_val = n_nodes_for_height(args.forest_height)
    tree_reduce_max_depth = max(2, min(4, args.tree_reduce_max_depth))
    if tree_reduce_max_depth != args.tree_reduce_max_depth and verbose:
        print(f"  tree_reduce_max_depth clamped to {tree_reduce_max_depth} (allowed 2..4)")
    per_section_optimize = (
        None if args.per_section_optimize_secs is None or args.per_section_optimize_secs <= 0
        else args.per_section_optimize_secs
    )
    cpsat_time_limit = args.cpsat_time_limit
    # force_zero_gap defaults to 2h time limit, but allow explicit override
    if args.force_zero_gap and cpsat_time_limit == 3600.0:  # Only auto-set if using default
        cpsat_time_limit = 7200
        if verbose:
            print(f"  (force_zero_gap: using cpsat_time_limit={cpsat_time_limit:.0f}s to allow proving optimality)")
    elif args.force_zero_gap and verbose:
        print(f"  (force_zero_gap: using explicit cpsat_time_limit={cpsat_time_limit:.0f}s)")

    scheduler_name = "greedy" if args.use_greedy else f"CP-SAT ({cpsat_time_limit:.0f}s)"
    print(f"Building SSA kernel with {scheduler_name}...")
    print(f"  forest_height={args.forest_height}, n_nodes={n_nodes_val}")
    print(f"  batch_size={args.batch_size}, rounds={args.rounds}")
    print(f"  enable_valu_shift={args.enable_valu_shift}")
    print(f"  tree_reduce_max_depth={tree_reduce_max_depth}")
    if per_section_optimize is not None:
        print(f"  per_section_optimize_secs={per_section_optimize}")
    if args.const_choice != "optimize" or args.select_choice != "optimize" or args.valu_shift != "optimize":
        print(f"  const_choice={args.const_choice}, select_choice={args.select_choice}, valu_shift={args.valu_shift}")
    build_kwargs = args_to_build_kwargs(args)
    if build_kwargs.get("optional_constraint_groups_to_disable"):
        names = [g.name for g in build_kwargs["optional_constraint_groups_to_disable"]]
        print(f"  disable_optional_constraint_groups={names}")
    if args.force_zero_gap:
        print(f"  force_zero_gap=True")
    print(f"  asap_alap_tightening={getattr(args, 'asap_alap_tightening', 'symmetry-aware')}")
    if args.enable_scratch_tightening:
        print(f"  enable_scratch_tightening=True")
    if args.scheduling_pressure_policy != "adaptive":
        print(f"  scheduling_pressure_policy={args.scheduling_pressure_policy}")
    print()  # Blank line before Rust output
    sys.stdout.flush()  # Ensure header appears before Rust output

    output_json_path = args.output_json if args.output_json is not None else DEFAULT_SCHEDULER_JSON
    start_time = time.time()
    instrs, stats = build_kernel_with_args(
        args,
        verbose=verbose,
        output_json=output_json_path,
    )
    build_time = time.time() - start_time

    cycles = stats["cycles"]
    speedup = BASELINE / cycles

    print(f"\nResults:")
    print(f"  Build time: {build_time:.1f}s")
    print(f"  Cycles: {cycles}")
    print(f"  Speedup: {speedup:.1f}x over baseline ({BASELINE})")
    print(f"  Instructions: {len(instrs)}")
    sim_cycles, empty_bundles = simulator_cycles(instrs)
    print(f"  Simulator cycles: {sim_cycles} (what problem.Machine and the test harness count: "
          f"{len(instrs)} bundles including the 2 pause bundles, minus {empty_bundles} with no ops)")
    print(f"  VRegs: {stats.get('vreg_count', 'N/A')}")
    print(f"  Peak live: {stats.get('peak_live', 'N/A')} words")
    print(f"  Budget exceeded: {stats.get('budget_exceeded', False)}")

    budget_exceeded = stats.get("budget_exceeded", False)
    if budget_exceeded:
        print("\n  WARNING: Scratch budget exceeded! Results will be incorrect.")
        print("  The greedy scheduler does not currently respect budget constraints.")
        print("  Use CP-SAT scheduler (default) for correct results.")

    print(f"\n  Schedule JSON written to: {output_json_path}")
    print_schedule_report(output_json_path, full=verbose)

    # Always save kernel before validating so failures can be analyzed without re-running.
    kernel_path = args.save_kernel
    if args.validate and kernel_path is None:
        kernel_path = "kernel_to_validate.json"
    if kernel_path:
        save_kernel_to_json(
            instrs,
            kernel_path,
            args.forest_height,
            args.batch_size,
            args.rounds,
            args.seed,
        )
        print(f"\n  Kernel saved to {kernel_path} (rerun with --validate-only {kernel_path})")

    if args.validate:
        if budget_exceeded:
            print("\nSkipping validation - budget exceeded (results known to be incorrect)")
            return 1
        if args.dataflow_check:
            # Symbolic replay first: it holds for all inputs and, on failure, names the
            # first cycle where the kernel departs from the SSA program.
            print("\nChecking kernel dataflow against the SSA program...")
            if not check_kernel_dataflow(instrs, output_json_path):
                return 1
        print("\nValidating against reference kernel...")
        correct = validate_kernel(
            instrs,
            args.forest_height,
            args.batch_size,
            args.rounds,
            args.seed,
        )
        if correct:
            print(f"  PASSED - kernel produces correct output")
        else:
            print(f"  FAILED - kernel produces incorrect output")
            return 1

    return 0


def print_schedule_report(scheduler_json_path, full):
    """Print the capacity floor for the ops this run emitted, what binds it and the gap.

    Three lines with -q, the whole report (scripts/schedule_report.py) otherwise. The floor
    is conditional on the emission; it is not a bound on the problem."""
    try:
        from scripts import schedule_report

        with open(scheduler_json_path) as f:
            report = schedule_report.analyze(json.load(f))
        if full:
            lines = schedule_report.format_report(report).splitlines()
        else:
            lines = schedule_report.summary_lines(report)
    except Exception as e:  # advisory output: must never abort the build or skip the kernel save
        print(f"\n  (schedule report skipped: {type(e).__name__}: {e})")
        return
    if full:
        print(f"\nSchedule report (uv run python -m scripts.schedule_report {scheduler_json_path}):")
    else:
        print()
    for line in lines:
        print(f"  {line}")


def check_kernel_dataflow(instrs, scheduler_json_path):
    """Prove the emitted kernel computes what its SSA program computes (all inputs).

    Returns False only when the replay finds a divergence; a scheduler JSON without SSA
    ops (e.g. written by an older build) skips the check with a message.
    """
    from verification.kernel_dataflow import check, format_report

    try:
        with open(scheduler_json_path) as f:
            sched = json.load(f)
        # Round-trip through JSON so slots are plain lists, as in a saved kernel.
        report = check(json.loads(json.dumps(list(instrs))), sched)
    except (OSError, ValueError) as e:
        print(f"  SKIPPED - {e}")
        return True
    print("  " + format_report(report).replace("\n", "\n  "))
    return report.ok


def load_kernel_from_json(path):
    """Load instruction list from JSON. If file has 'instrs' key, use saved params; else treat as raw list."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "instrs" in data:
        raw = data["instrs"]
        params = {
            "forest_height": data.get("forest_height"),
            "batch_size": data.get("batch_size"),
            "rounds": data.get("rounds"),
            "seed": data.get("seed"),
        }
    else:
        raw = data
        params = None
    instrs = []
    for bundle in raw:
        restored = {}
        for engine, ops in bundle.items():
            restored[engine] = [tuple(op) for op in ops]
        instrs.append(restored)
    return instrs, params


def simulator_cycles(instrs):
    """Cycles problem.Machine counts for a kernel, and the number of bundles it does not count.

    The scheduler's `cycles` is the schedule length. The simulator executes every bundle but
    only counts those with at least one non-debug slot, and the kernel is wrapped in two
    `pause` bundles (which count). So simulator cycles = schedule cycles + 2 - empty bundles;
    e.g. the 16-round greedy kernel is 2007 schedule cycles and 1994 simulator cycles.

    Returns (simulator_cycles, empty_bundles).
    """
    counted = sum(1 for bundle in instrs if any(engine != "debug" for engine in bundle))
    return counted, len(instrs) - counted


def save_kernel_to_json(instrs, path, forest_height, batch_size, rounds, seed):
    """Save instruction list and problem params to JSON so --validate-only uses same params."""
    serializable = []
    for bundle in instrs:
        sb = {}
        for k, v in bundle.items():
            sb[k] = [list(op) for op in v]
        serializable.append(sb)
    payload = {
        "forest_height": forest_height,
        "batch_size": batch_size,
        "rounds": rounds,
        "seed": seed,
        "instrs": serializable,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def run_validate_only(kernel_path, forest_height, batch_size, rounds, seed):
    """Load kernel from JSON and run validation only (no build). Uses params from file if saved."""
    print(f"Loading kernel from {kernel_path}...")
    instrs, saved_params = load_kernel_from_json(kernel_path)
    if saved_params and all(saved_params.get(k) is not None for k in ("forest_height", "batch_size", "rounds", "seed")):
        forest_height = saved_params["forest_height"]
        batch_size = saved_params["batch_size"]
        rounds = saved_params["rounds"]
        seed = saved_params["seed"]
        print(f"  Loaded {len(instrs)} instructions; params from file: forest_height={forest_height}, batch_size={batch_size}, rounds={rounds}, seed={seed}")
    else:
        print(f"  Loaded {len(instrs)} instructions; params from CLI: forest_height={forest_height}, batch_size={batch_size}, rounds={rounds}, seed={seed}")
    print("\nValidating against reference kernel...")
    correct = validate_kernel(instrs, forest_height, batch_size, rounds, seed)
    if correct:
        print("  PASSED - kernel produces correct output")
        return 0
    print("  FAILED - kernel produces incorrect output")
    return 1


def validate_kernel(instrs, forest_height, batch_size, rounds, seed):
    """Validate kernel against reference implementation."""
    from problem import (
        Machine, Tree, Input, build_mem_image, reference_kernel2, N_CORES, DebugInfo
    )

    random.seed(seed)

    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    machine = Machine(
        mem,
        list(instrs),
        DebugInfo(scratch_map={}),
        n_cores=N_CORES,
        trace=False,
    )
    machine.enable_pause = False
    machine.enable_debug = False
    machine.prints = False  # Enable to debug

    try:
        machine.run()
    except Exception as e:
        # Print last few cycles for debug
        print(f"\nCrash at cycle {machine.cycle}, core 0 pc={machine.cores[0].pc}")
        print(f"Last instruction: {instrs[machine.cores[0].pc - 1] if machine.cores[0].pc > 0 else 'N/A'}")
        raise
    except IndexError as e:
        print(f"\nIndexError details: {e}")
        print(f"  mem_write entries: {len(machine.mem_write) if hasattr(machine, 'mem_write') else 'N/A'}")
        print(f"  mem length: {len(machine.mem)}")
        if hasattr(machine, 'mem_write'):
            for addr, val in machine.mem_write.items():
                if addr >= len(machine.mem):
                    print(f"  OUT OF RANGE: addr={addr}, val={val}")
                else:
                    print(f"  addr={addr}, val={val}")
        raise

    # Verify against reference
    ref_mem = list(reference_kernel2(mem))[-1]

    inp_values_p = ref_mem[6]
    inp_indices_p = ref_mem[5]

    got_values = machine.mem[inp_values_p : inp_values_p + batch_size]
    ref_values = ref_mem[inp_values_p : inp_values_p + batch_size]
    got_indices = machine.mem[inp_indices_p : inp_indices_p + batch_size]
    ref_indices = ref_mem[inp_indices_p : inp_indices_p + batch_size]

    values_correct = got_values == ref_values
    indices_correct = got_indices == ref_indices

    if not values_correct:
        first_bad = next(i for i in range(batch_size) if got_values[i] != ref_values[i])
        print(f"\n  Value mismatch: first difference at batch index {first_bad} (of {batch_size})")
        print(f"    Got:      {got_values[first_bad]!r}")
        print(f"    Expected: {ref_values[first_bad]!r}")
        # Show a window around the first mismatch
        lo = max(0, first_bad - 2)
        hi = min(batch_size, first_bad + 6)
        print(f"    Window [{lo}:{hi}]: Got {got_values[lo:hi]}")
        print(f"                 Expected {ref_values[lo:hi]}")

    if not indices_correct:
        first_bad = next(i for i in range(batch_size) if got_indices[i] != ref_indices[i])
        print(f"\n  Index mismatch: first difference at batch index {first_bad} (of {batch_size})")
        print(f"    Got:      {got_indices[first_bad]!r}")
        print(f"    Expected: {ref_indices[first_bad]!r}")
        lo = max(0, first_bad - 2)
        hi = min(batch_size, first_bad + 6)
        print(f"    Window [{lo}:{hi}]: Got {got_indices[lo:hi]}")
        print(f"                 Expected {ref_indices[lo:hi]}")

    return values_correct and indices_correct


if __name__ == "__main__":
    exit(main())
