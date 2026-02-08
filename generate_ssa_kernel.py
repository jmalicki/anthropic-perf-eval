#!/usr/bin/env python3
"""Generate optimized kernel using SSA builder with CP-SAT.

This is the recommended approach for generating high-quality kernels.
The SSA builder eliminates false dependencies, and CP-SAT finds optimal schedules.

Default settings:
- enable_valu_shift=True (VALU→ALU shifting for better slot use)
- use_cpsat=True (optimal scheduling)
- cpsat_time_limit=600 (10 minutes - good default)

Example usage:
    # Default (CP-SAT with VALU shift)
    python generate_ssa_kernel.py

    # Quick iteration with greedy scheduler
    python generate_ssa_kernel.py --use-greedy

    # Longer optimization
    python generate_ssa_kernel.py --cpsat-time-limit 3600

    # With validation (kernel is always saved before validation for debugging)
    python generate_ssa_kernel.py --validate
    # Kernel written to kernel_to_validate.json; re-validate without rebuilding:
    python generate_ssa_kernel.py --validate-only kernel_to_validate.json
"""

import argparse
import json
import random
import subprocess
import sys
import time

import scheduler_rs

DEFAULT_SCHEDULER_JSON = "scheduler_output.json"

BASELINE = 147734


def get_git_sha():
    """Return git HEAD SHA (short) or None if not a repo or git unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if out.returncode == 0 and out.stdout:
            return out.stdout.strip()[:12]  # short SHA
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


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
                        help="Number of parallel traversals; must be a multiple of 8 (VLEN) (default: 256)")
    parser.add_argument("--rounds", type=int, default=16,
                        help="Hash rounds per traversal (default: 16)")

    # Scheduler options
    parser.add_argument("--cpsat-time-limit", type=float, default=3600.0,
                        help="CP-SAT time limit in seconds (default: 3600 = 1 hour)")
    parser.add_argument("--per-section-optimize-secs", type=float, default=None, metavar="SECS",
                        help="If set, run each per-section CP-SAT for SECS seconds to optimize cycles (no VALU shift in per-section). 0 or omit = feasibility only.")
    parser.add_argument("--use-greedy", action="store_true",
                        help="Use greedy scheduler instead of CP-SAT (fast, ~2s)")
    parser.add_argument("--enable-valu-shift", action="store_true", default=True,
                        help="Enable VALU->ALU shifting (default: True)")
    parser.add_argument("--no-valu-shift", action="store_false", dest="enable_valu_shift",
                        help="Disable VALU->ALU shifting (e.g. if hitting scratch budget)")
    parser.add_argument("--tree-reduce-max-depth", type=int, default=3, metavar="N",
                        help="Max tree depth for tree-reduce selection (2..4). Default 3; use 4 for 16-way at depth 4 (more VALU ops).")
    parser.add_argument("--const-choice", choices=["optimize", "add_imm", "load"], default="optimize",
                        help="CP-SAT ConstChoice: optimize, force add_imm, or force load (default: optimize)")
    parser.add_argument("--select-choice", choices=["optimize", "fma", "vselect"], default="optimize",
                        help="CP-SAT SelectChoice: optimize, force fma, or force vselect (default: optimize)")
    parser.add_argument("--valu-shift", choices=["optimize", "alu", "valu"], default="optimize",
                        help="CP-SAT VALU shift: optimize, force alu, or force valu (default: optimize)")
    parser.add_argument("--force-zero-gap", action="store_true",
                        help="Force 0 integral gap for proven optimality (debugging only, slower). Uses longer time limit (2h) so CP-SAT can prove optimality.")
    parser.add_argument("--enable-asap-alap-tightening", action="store_true",
                        help="Enable ASAP/ALAP tightening by considering slot capacity (experimental, for ablation studies)")
    parser.add_argument("--enable-scratch-tightening", action="store_true",
                        help="Enable scratch-aware tightening. Considers scratch constraints when tightening bounds (experimental, for ablation studies). Can be used independently of --enable-asap-alap-tightening.")
    parser.add_argument("--scheduling-pressure-policy", choices=["greedy", "throttled", "gated", "adaptive"], default="adaptive",
                        help="Scheduling pressure policy: adaptive (default, transitions automatically), greedy (no pressure awareness), throttled (soft re-sort), gated (hard gate)")

    # Output options
    parser.add_argument("--output-json", type=str, default=None,
                        help=f"Output schedule JSON path (default: {DEFAULT_SCHEDULER_JSON})")
    parser.add_argument("--comment", type=str, default=None,
                        help="Comment stored in the schedule JSON to describe the run")
    parser.add_argument("--liveness-detail", type=str, default=None,
                        choices=["none", "summary", "full"],
                        help="Liveness detail level in JSON: none, summary, or full (default: full)")
    parser.add_argument("--validate", action="store_true",
                        help="Validate against reference kernel (kernel is always saved before validation for debugging)")
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
    # Clamp to allowed range (min 2, max 4)
    tree_reduce_max_depth = max(2, min(4, args.tree_reduce_max_depth))
    if tree_reduce_max_depth != args.tree_reduce_max_depth and verbose:
        print(f"  tree_reduce_max_depth clamped to {tree_reduce_max_depth} (allowed 2..3)")
    per_section_optimize = (
        None if args.per_section_optimize_secs is None or args.per_section_optimize_secs <= 0
        else args.per_section_optimize_secs
    )

    # With --force-zero-gap, CP-SAT must prove optimality; use longer time limit if needed
    cpsat_time_limit = args.cpsat_time_limit
    if args.force_zero_gap and cpsat_time_limit < 7200:
        cpsat_time_limit = 7200  # 2 hours so solver has time to prove optimality
        if verbose:
            print(f"  (force_zero_gap: using cpsat_time_limit={cpsat_time_limit:.0f}s to allow proving optimality)")

    scheduler_name = "greedy" if args.use_greedy else f"CP-SAT ({cpsat_time_limit:.0f}s)"
    print(f"Building SSA kernel with {scheduler_name}...")
    print(f"  forest_height={args.forest_height}, n_nodes={args.n_nodes}")
    print(f"  batch_size={args.batch_size}, rounds={args.rounds}")
    print(f"  enable_valu_shift={args.enable_valu_shift}")
    print(f"  tree_reduce_max_depth={tree_reduce_max_depth}")
    if per_section_optimize is not None:
        print(f"  per_section_optimize_secs={per_section_optimize}")
    if args.const_choice != "optimize" or args.select_choice != "optimize" or args.valu_shift != "optimize":
        print(f"  const_choice={args.const_choice}, select_choice={args.select_choice}, valu_shift={args.valu_shift}")
    if args.force_zero_gap:
        print(f"  force_zero_gap=True")
    if args.enable_asap_alap_tightening:
        print(f"  enable_asap_alap_tightening=True")
    if args.enable_scratch_tightening:
        print(f"  enable_scratch_tightening=True")
    if args.scheduling_pressure_policy != "adaptive":
        print(f"  scheduling_pressure_policy={args.scheduling_pressure_policy}")
    print()  # Blank line before Rust output
    sys.stdout.flush()  # Ensure header appears before Rust output

    # Always write schedule JSON (default path or user-specified); include run metadata
    output_json_path = args.output_json if args.output_json is not None else DEFAULT_SCHEDULER_JSON
    run_options = {
        "forest_height": args.forest_height,
        "n_nodes": args.n_nodes,
        "batch_size": args.batch_size,
        "rounds": args.rounds,
        "use_cpsat": use_cpsat,
        "cpsat_time_limit": cpsat_time_limit,
        "per_section_optimize_secs": per_section_optimize,
        "enable_valu_shift": args.enable_valu_shift,
        "tree_reduce_max_depth": tree_reduce_max_depth,
        "const_choice": args.const_choice,
        "select_choice": args.select_choice,
        "valu_shift": args.valu_shift,
        "force_zero_gap": args.force_zero_gap and use_cpsat,
        "enable_asap_alap_tightening": args.enable_asap_alap_tightening and use_cpsat,
        "enable_scratch_tightening": args.enable_scratch_tightening and use_cpsat,
        "scheduling_pressure_policy": args.scheduling_pressure_policy,
    }
    git_sha = get_git_sha()

    start_time = time.time()
    instrs, stats = scheduler_rs.build_tick_tock_kernel_ssa(
        forest_height=args.forest_height,
        n_nodes=args.n_nodes,
        batch_size=args.batch_size,
        rounds=args.rounds,
        enable_valu_shift=args.enable_valu_shift,
        verbose=verbose,
        use_cpsat=use_cpsat,
        cpsat_time_limit=cpsat_time_limit,
        per_section_optimize_secs=per_section_optimize,
        output_json=output_json_path,
        tree_reduce_max_depth=tree_reduce_max_depth,
        const_choice=args.const_choice,
        select_choice=args.select_choice,
        valu_shift=args.valu_shift,
        force_zero_gap=args.force_zero_gap if use_cpsat else None,
        enable_asap_alap_tightening=args.enable_asap_alap_tightening if use_cpsat else None,
        enable_scratch_tightening=args.enable_scratch_tightening if use_cpsat else None,
        run_options_json=json.dumps(run_options),
        git_sha=git_sha,
        comment=args.comment,
        scheduling_pressure_policy=args.scheduling_pressure_policy,
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

    print(f"\n  Schedule JSON written to: {output_json_path}")

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
