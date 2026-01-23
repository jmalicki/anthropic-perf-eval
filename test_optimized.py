#!/usr/bin/env python3
"""
Test the optimized kernel against baseline.
"""

import random
from problem import (
    Machine, Tree, Input, build_mem_image, reference_kernel2, N_CORES
)
from solver.kernel_gen import OptimizedKernelBuilder

BASELINE = 147734


def test_optimized_kernel(
    forest_height: int = 10,
    rounds: int = 16,
    batch_size: int = 256,
    seed: int = 123,
    trace: bool = False,
):
    print(f"Testing optimized kernel: {forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)

    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    # Build optimized kernel
    kb = OptimizedKernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)

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
    cycles, correct = test_optimized_kernel()
    if correct:
        print(f"\n✓ Test passed with {cycles} cycles ({BASELINE/cycles:.1f}x speedup)")
    else:
        print(f"\n✗ Test failed - results incorrect")
