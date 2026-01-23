#!/usr/bin/env python3
"""
Run optimization experiments with wandb tracking.

Usage:
    python run_experiment.py              # Run with wandb (requires login)
    python run_experiment.py --offline    # Run in offline mode
    wandb login                           # To log in first
"""

import argparse
import random
import time
import os

# Set wandb to offline mode if needed before importing
parser = argparse.ArgumentParser()
parser.add_argument("--offline", action="store_true", help="Run wandb in offline mode")
args, _ = parser.parse_known_args()

if args.offline:
    os.environ["WANDB_MODE"] = "offline"

import wandb
from problem import Machine, Tree, Input, build_mem_image, reference_kernel2, N_CORES

BASELINE = 147734


def run_kernel_test(
    kernel_builder,
    kernel_name: str,
    forest_height: int = 10,
    rounds: int = 16,
    batch_size: int = 256,
    seed: int = 123,
    log_to_wandb: bool = True,
):
    """Run a kernel test and log results to wandb."""
    random.seed(seed)

    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    # Build kernel
    start_build = time.time()
    kernel_builder.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    build_time = time.time() - start_build

    n_instructions = len(kernel_builder.instrs)

    # Run machine
    machine = Machine(mem, kernel_builder.instrs, kernel_builder.debug_info(), n_cores=N_CORES)
    machine.enable_pause = False
    machine.enable_debug = False

    start_run = time.time()
    machine.run()
    run_time = time.time() - start_run

    # Verify correctness
    ref_mem = list(reference_kernel2(mem))[-1]
    inp_values_p = ref_mem[6]
    inp_indices_p = ref_mem[5]

    values_correct = (
        machine.mem[inp_values_p : inp_values_p + batch_size]
        == ref_mem[inp_values_p : inp_values_p + batch_size]
    )
    indices_correct = (
        machine.mem[inp_indices_p : inp_indices_p + batch_size]
        == ref_mem[inp_indices_p : inp_indices_p + batch_size]
    )

    is_correct = values_correct and indices_correct
    cycles = machine.cycle
    speedup = BASELINE / cycles

    results = {
        "kernel_name": kernel_name,
        "cycles": cycles,
        "speedup": speedup,
        "baseline": BASELINE,
        "n_instructions": n_instructions,
        "build_time_s": build_time,
        "run_time_s": run_time,
        "correct": is_correct,
        "values_correct": values_correct,
        "indices_correct": indices_correct,
        "forest_height": forest_height,
        "rounds": rounds,
        "batch_size": batch_size,
    }

    if log_to_wandb:
        wandb.log(results)

    return results


def main():
    # Initialize wandb
    wandb.init(
        project="vliw-kernel-optimization",
        config={
            "forest_height": 10,
            "rounds": 16,
            "batch_size": 256,
            "baseline_cycles": BASELINE,
            "target_cycles": 1487,  # Claude Opus 4.5 best
        }
    )

    print("=" * 60)
    print("VLIW Kernel Optimization Experiment")
    print("=" * 60)
    print(f"Baseline: {BASELINE} cycles")
    print(f"Target: 1487 cycles (Opus 4.5 best)")
    print()

    # Test 1: Baseline kernel
    print("Testing baseline kernel...")
    from perf_takehome import KernelBuilder
    baseline_kb = KernelBuilder()
    baseline_results = run_kernel_test(baseline_kb, "baseline")
    print(f"  Cycles: {baseline_results['cycles']}")
    print(f"  Correct: {baseline_results['correct']}")
    print()

    # Test 2: Optimized kernel (unrolled vectorized)
    print("Testing optimized kernel (unrolled)...")
    from solver.kernel_gen import OptimizedKernelBuilder
    opt_kb = OptimizedKernelBuilder()
    opt_results = run_kernel_test(opt_kb, "optimized_unrolled")
    print(f"  Cycles: {opt_results['cycles']}")
    print(f"  Speedup: {opt_results['speedup']:.2f}x")
    print(f"  Correct: {opt_results['correct']}")
    print()

    # Test 3: Highly optimized kernel (looped)
    print("Testing highly optimized kernel (looped)...")
    from solver.optimized_kernel import HighlyOptimizedKernelBuilder
    highly_opt_kb = HighlyOptimizedKernelBuilder()
    highly_opt_results = run_kernel_test(highly_opt_kb, "optimized_looped")
    print(f"  Cycles: {highly_opt_results['cycles']}")
    print(f"  Speedup: {highly_opt_results['speedup']:.2f}x")
    print(f"  Correct: {highly_opt_results['correct']}")
    print()

    # Test 4: Aggressive kernel (max VLIW packing)
    print("Testing aggressive kernel (max VLIW)...")
    from solver.aggressive_kernel import AggressiveKernelBuilder
    aggressive_kb = AggressiveKernelBuilder()
    aggressive_results = run_kernel_test(aggressive_kb, "aggressive_vliw")
    print(f"  Cycles: {aggressive_results['cycles']}")
    print(f"  Speedup: {aggressive_results['speedup']:.2f}x")
    print(f"  Correct: {aggressive_results['correct']}")
    print()

    # Test 5: Pipelined kernel (software pipelining)
    print("Testing pipelined kernel...")
    from solver.pipelined_kernel import PipelinedKernelBuilder
    pipelined_kb = PipelinedKernelBuilder()
    pipelined_results = run_kernel_test(pipelined_kb, "pipelined")
    print(f"  Cycles: {pipelined_results['cycles']}")
    print(f"  Speedup: {pipelined_results['speedup']:.2f}x")
    print(f"  Correct: {pipelined_results['correct']}")
    print()

    # Test 6: Deeply unrolled kernel (4 groups at once)
    print("Testing unrolled kernel (4 groups)...")
    try:
        from solver.unrolled_kernel import UnrolledKernelBuilder
        unrolled_kb = UnrolledKernelBuilder()
        unrolled_results = run_kernel_test(unrolled_kb, "unrolled_4x")
        print(f"  Cycles: {unrolled_results['cycles']}")
        print(f"  Speedup: {unrolled_results['speedup']:.2f}x")
        print(f"  Correct: {unrolled_results['correct']}")
    except Exception as e:
        print(f"  Error: {e}")
        unrolled_results = {"kernel_name": "unrolled_4x", "cycles": 999999, "speedup": 0.1, "correct": False}
    print()

    # Test 7: Ultra kernel (8 groups at once)
    print("Testing ultra kernel (8 groups)...")
    try:
        from solver.ultra_kernel import UltraKernelBuilder
        ultra_kb = UltraKernelBuilder()
        ultra_results = run_kernel_test(ultra_kb, "ultra_8x")
        print(f"  Cycles: {ultra_results['cycles']}")
        print(f"  Speedup: {ultra_results['speedup']:.2f}x")
        print(f"  Correct: {ultra_results['correct']}")
    except Exception as e:
        print(f"  Error: {e}")
        ultra_results = {"kernel_name": "ultra_8x", "cycles": 999999, "speedup": 0.1, "correct": False}
    print()

    # Test 8: Ultimate kernel (blocked + parallel + tree preload)
    print("Testing ultimate kernel (blocked + parallel + tree preload)...")
    try:
        from solver.ultimate_kernel import UltimateKernelBuilder
        ultimate_kb = UltimateKernelBuilder()
        ultimate_results = run_kernel_test(ultimate_kb, "ultimate")
        print(f"  Cycles: {ultimate_results['cycles']}")
        print(f"  Speedup: {ultimate_results['speedup']:.2f}x")
        print(f"  Correct: {ultimate_results['correct']}")
    except Exception as e:
        print(f"  Error: {e}")
        ultimate_results = {"kernel_name": "ultimate", "cycles": 999999, "speedup": 0.1, "correct": False}
    print()

    # Test 9: Dual-path kernel
    print("Testing dual-path kernel...")
    try:
        from solver.dual_path_kernel import DualPathKernelBuilder
        dual_kb = DualPathKernelBuilder()
        dual_results = run_kernel_test(dual_kb, "dual_path")
        print(f"  Cycles: {dual_results['cycles']}")
        print(f"  Speedup: {dual_results['speedup']:.2f}x")
        print(f"  Correct: {dual_results['correct']}")
    except Exception as e:
        print(f"  Error: {e}")
        dual_results = {"kernel_name": "dual_path", "cycles": 999999, "speedup": 0.1, "correct": False}
    print()

    all_results = [baseline_results, opt_results, highly_opt_results, aggressive_results, pipelined_results, unrolled_results, ultra_results, ultimate_results, dual_results]

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Kernel':<25} {'Cycles':>10} {'Speedup':>10} {'Correct':>10}")
    print("-" * 60)
    for r in all_results:
        print(f"{r['kernel_name']:<25} {r['cycles']:>10} {r['speedup']:>10.2f}x {str(r['correct']):>10}")

    best_speedup = max(r['speedup'] for r in all_results)
    best_cycles = min(r['cycles'] for r in all_results if r['correct'])

    print("-" * 60)
    print(f"Best: {best_cycles} cycles ({best_speedup:.1f}x speedup)")
    print(f"Target: 1487 cycles (100x speedup)")
    print(f"Gap: {best_cycles - 1487} cycles")

    # Log summary table
    wandb.log({
        "summary/baseline_cycles": baseline_results['cycles'],
        "summary/optimized_unrolled_cycles": opt_results['cycles'],
        "summary/optimized_looped_cycles": highly_opt_results['cycles'],
        "summary/aggressive_cycles": aggressive_results['cycles'],
        "summary/pipelined_cycles": pipelined_results['cycles'],
        "summary/unrolled_4x_cycles": unrolled_results['cycles'],
        "summary/ultra_8x_cycles": ultra_results['cycles'],
        "summary/best_cycles": best_cycles,
        "summary/best_speedup": best_speedup,
        "summary/gap_to_target": best_cycles - 1487,
    })

    wandb.finish()


if __name__ == "__main__":
    main()
