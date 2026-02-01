#!/usr/bin/env python3
"""Generate optimized kernel using SSA builder with CP-SAT.

This is the recommended approach for generating high-quality kernels.
The SSA builder eliminates false dependencies, and CP-SAT finds optimal schedules.

Default settings:
- enable_valu_shift=False (avoids scratch budget issues)
- use_cpsat=True (optimal scheduling)
- cpsat_time_limit=600 (10 minutes - good default)

Example usage:
    # Default (10 min CP-SAT, no VALU shift)
    python generate_ssa_kernel.py

    # Quick iteration with greedy scheduler
    python generate_ssa_kernel.py --use-greedy

    # Longer optimization
    python generate_ssa_kernel.py --cpsat-time-limit 3600

    # With validation
    python generate_ssa_kernel.py --validate
"""

import argparse
import json
import random
import sys
import time

import scheduler_rs

BASELINE = 147734


def main():
    parser = argparse.ArgumentParser(
        description="Generate optimized kernel using SSA builder with CP-SAT"
    )

    # Problem parameters
    parser.add_argument("--forest-height", type=int, default=10,
                        help="Tree depth (default: 10, gives 2047 nodes)")
    parser.add_argument("--n-nodes", type=int, default=2047,
                        help="Number of tree nodes (default: 2047 = 2^10 - 1)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Number of parallel traversals (default: 256)")
    parser.add_argument("--rounds", type=int, default=16,
                        help="Hash rounds per traversal (default: 16)")

    # Scheduler options
    parser.add_argument("--cpsat-time-limit", type=float, default=3600.0,
                        help="CP-SAT time limit in seconds (default: 3600 = 1 hour)")
    parser.add_argument("--per-section-optimize-secs", type=float, default=None, metavar="SECS",
                        help="If set, run each per-section CP-SAT for SECS seconds to optimize cycles (no VALU shift in per-section). 0 or omit = feasibility only.")
    parser.add_argument("--use-greedy", action="store_true",
                        help="Use greedy scheduler instead of CP-SAT (fast, ~2s)")
    parser.add_argument("--enable-valu-shift", action="store_true",
                        help="Enable VALU->ALU shifting (disabled by default to avoid budget issues)")
    parser.add_argument("--enable-tree-reduce-depth3", action="store_true",
                        help="Enable tree-reduce for depth-3 (8-way) selection (uses more scratch)")

    # Output options
    parser.add_argument("--output-json", type=str, default=None,
                        help="Output schedule JSON for analysis")
    parser.add_argument("--liveness-detail", type=str, default=None,
                        choices=["none", "summary", "full"],
                        help="Liveness detail level in JSON: none, summary, or full (default: full)")
    parser.add_argument("--validate", action="store_true",
                        help="Validate against reference kernel")
    parser.add_argument("--validate-only", type=str, metavar="PATH", default=None,
                        help="Load kernel from JSON and run validation only (no build). Use same forest_height, batch_size, rounds, seed.")
    parser.add_argument("--save-kernel", type=str, metavar="PATH", default=None,
                        help="After building, save instruction list to JSON so you can rerun validation with --validate-only PATH")
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
    per_section_optimize = (
        None if args.per_section_optimize_secs is None or args.per_section_optimize_secs <= 0
        else args.per_section_optimize_secs
    )

    scheduler_name = "greedy" if args.use_greedy else f"CP-SAT ({args.cpsat_time_limit:.0f}s)"
    print(f"Building SSA kernel with {scheduler_name}...")
    print(f"  forest_height={args.forest_height}, n_nodes={args.n_nodes}")
    print(f"  batch_size={args.batch_size}, rounds={args.rounds}")
    print(f"  enable_valu_shift={args.enable_valu_shift}")
    print(f"  enable_tree_reduce_depth3={args.enable_tree_reduce_depth3}")
    if per_section_optimize is not None:
        print(f"  per_section_optimize_secs={per_section_optimize}")
    print()  # Blank line before Rust output
    sys.stdout.flush()  # Ensure header appears before Rust output

    start_time = time.time()
    instrs, stats = scheduler_rs.build_tick_tock_kernel_ssa(
        forest_height=args.forest_height,
        n_nodes=args.n_nodes,
        batch_size=args.batch_size,
        rounds=args.rounds,
        enable_valu_shift=args.enable_valu_shift,
        verbose=verbose,
        use_cpsat=use_cpsat,
        cpsat_time_limit=args.cpsat_time_limit,
        per_section_optimize_secs=per_section_optimize,
        output_json=args.output_json,
        enable_tree_reduce_depth3=args.enable_tree_reduce_depth3,
        # Note: liveness_detail parameter not available in Python bindings yet
        # It's only used when output_json is provided anyway
    )
    build_time = time.time() - start_time

    cycles = stats["cycles"]
    speedup = BASELINE / cycles

    print(f"\nResults:")
    print(f"  Build time: {build_time:.1f}s")
    print(f"  Cycles: {cycles}")
    print(f"  Speedup: {speedup:.1f}x over baseline ({BASELINE})")
    print(f"  Instructions: {len(instrs)}")
    print(f"  VRegs: {stats.get('vreg_count', 'N/A')}")
    print(f"  Peak live: {stats.get('peak_live', 'N/A')} words")
    print(f"  Budget exceeded: {stats.get('budget_exceeded', False)}")

    budget_exceeded = stats.get("budget_exceeded", False)
    if budget_exceeded:
        print("\n  WARNING: Scratch budget exceeded! Results will be incorrect.")
        print("  The greedy scheduler does not currently respect budget constraints.")
        print("  Use CP-SAT scheduler (default) for correct results.")

    if args.output_json:
        print(f"\n  Schedule JSON written to: {args.output_json}")

    if args.save_kernel:
        save_kernel_to_json(
            instrs,
            args.save_kernel,
            args.forest_height,
            args.batch_size,
            args.rounds,
            args.seed,
        )
        print(f"\n  Kernel saved to {args.save_kernel} (rerun with --validate-only {args.save_kernel})")

    if args.validate:
        if budget_exceeded:
            print("\nSkipping validation - budget exceeded (results known to be incorrect)")
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
