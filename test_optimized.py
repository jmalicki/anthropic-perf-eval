#!/usr/bin/env python3
"""
Test the optimized kernel against baseline.
"""

import argparse
import random
import time
from problem import (
    Machine, Tree, Input, build_mem_image, reference_kernel2, N_CORES, DebugInfo
)

BASELINE = 147734


class RustKernelBuilderWrapper:
    """Wrapper to make Rust kernel builder compatible with test harness."""

    def __init__(self, instrs, scratch_debug=None):
        self.instrs = instrs
        self._scratch_debug = scratch_debug or {}

    def debug_info(self):
        return DebugInfo(scratch_map=self._scratch_debug)


def test_optimized_kernel(
    forest_height: int = 10,
    rounds: int = 16,
    batch_size: int = 256,
    seed: int = 123,
    trace: bool = False,
    use_precomputed: bool = False,
    use_tick_tock: bool = True,
    use_rust_builder: bool = False,
):
    print(f"Testing optimized kernel: {forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)

    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    # Build optimized kernel
    build_start = time.time()

    if use_rust_builder:
        # Use pure Rust kernel builder (fastest)
        import scheduler_rs
        print("Using Rust kernel builder with parallel multipass scheduling...")
        instrs = scheduler_rs.build_tick_tock_kernel(
            forest.height, len(forest.values), len(inp.indices), rounds,
            enable_valu_shift=True, verbose=True
        )
        kb = RustKernelBuilderWrapper(list(instrs))
    elif use_precomputed:
        # Use precomputed (pre-scheduled) kernel
        from solver.kernel_gen import OptimizedKernelBuilder
        from solver.precomputed_kernel import build_kernel as precomputed_build_kernel
        kb = OptimizedKernelBuilder()
        precomputed_build_kernel(kb, forest.height, len(forest.values), len(inp.indices), rounds)
    elif use_tick_tock:
        # Use tick-tock kernel builder with Rust scheduler (default)
        from solver.kernel_builder.tick_tock import TickTockKernelBuilder
        kb = TickTockKernelBuilder()
        kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    else:
        # Generate with basic builder (unscheduled)
        from solver.kernel_gen import OptimizedKernelBuilder
        kb = OptimizedKernelBuilder()
        kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)

    build_time = time.time() - build_start
    print(f"Build time: {build_time:.2f}s")

    print(f"Generated {len(kb.instrs)} instructions")

    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        trace=trace,
    )
    machine.enable_pause = False
    machine.enable_debug = False

    # Run
    machine.run()

    # Verify against reference
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

    print(f"CYCLES: {machine.cycle}")
    print(f"Speedup over baseline: {BASELINE / machine.cycle:.2f}x")
    print(f"Values correct: {values_correct}")
    print(f"Indices correct: {indices_correct}")

    if not values_correct:
        print("\nValue mismatch (first 8):")
        print(f"  Got:      {machine.mem[inp_values_p : inp_values_p + 8]}")
        print(f"  Expected: {ref_mem[inp_values_p : inp_values_p + 8]}")

    if not indices_correct:
        print("\nIndex mismatch (first 8):")
        print(f"  Got:      {machine.mem[inp_indices_p : inp_indices_p + 8]}")
        print(f"  Expected: {ref_mem[inp_indices_p : inp_indices_p + 8]}")

    return machine.cycle, values_correct and indices_correct


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test optimized kernel")

    # Kernel builder selection
    parser.add_argument("--rust-builder", action="store_true",
                       help="Use pure Rust kernel builder (fastest, parallel scheduling)")
    parser.add_argument("--tick-tock", action="store_true", default=True,
                       help="Use tick-tock kernel builder with Rust scheduler (default)")
    parser.add_argument("--no-tick-tock", dest="tick_tock", action="store_false",
                       help="Disable tick-tock, use basic builder")
    parser.add_argument("--precomputed", action="store_true",
                       help="Use precomputed (pre-scheduled) kernel")

    # Problem parameters
    parser.add_argument("--forest-height", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--trace", action="store_true", help="Enable execution trace")

    args = parser.parse_args()

    cycles, correct = test_optimized_kernel(
        forest_height=args.forest_height,
        rounds=args.rounds,
        batch_size=args.batch_size,
        seed=args.seed,
        trace=args.trace,
        use_precomputed=args.precomputed,
        use_tick_tock=args.tick_tock and not args.precomputed and not args.rust_builder,
        use_rust_builder=args.rust_builder,
    )
    if correct:
        print(f"\n✓ Test passed with {cycles} cycles ({BASELINE/cycles:.1f}x speedup)")
    else:
        print(f"\n✗ Test failed - results incorrect")
