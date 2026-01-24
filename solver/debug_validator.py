"""
Debug validation module for kernel verification.

Runs the kernel with debug instructions enabled and compares intermediate
values against reference implementation to identify exact points of divergence.
"""

from dataclasses import dataclass
from typing import Any, Optional
import random

from problem import (
    Machine, Tree, Input, DebugInfo, build_mem_image, reference_kernel2,
    VLEN, SCRATCH_SIZE
)


@dataclass
class ValidationFailure:
    """Details about a validation failure."""
    round: int
    group: int
    element: int
    phase: str
    expected: int
    got: int
    trace_key: tuple

    def __str__(self):
        return (
            f"FAILED at: round={self.round}, group={self.group}, "
            f"element={self.element}, phase={self.phase}\n"
            f"  Expected: 0x{self.expected:08X}\n"
            f"  Got:      0x{self.got:08X}\n"
            f"  Key: {self.trace_key}"
        )


@dataclass
class ValidationResult:
    """Result of kernel validation."""
    passed: bool
    first_failure: Optional[ValidationFailure]
    total_comparisons: int
    failed_comparisons: int
    failures: list  # List of ValidationFailure

    def summary(self) -> str:
        if self.passed:
            return f"PASSED: {self.total_comparisons} comparisons verified"
        return (
            f"FAILED: {self.failed_comparisons}/{self.total_comparisons} comparisons failed\n"
            f"First failure:\n{self.first_failure}"
        )


def generate_reference_trace(
    forest_height: int,
    batch_size: int,
    rounds: int,
    seed: int = 123,
) -> tuple[list[int], dict[Any, int]]:
    """
    Generate reference trace by running reference_kernel2.

    Returns:
        (mem, trace): Final memory and trace dictionary
    """
    random.seed(seed)
    tree = Tree.generate(forest_height)
    inp = Input.generate(tree, batch_size, rounds)
    mem = build_mem_image(tree, inp)

    trace = {}
    gen = reference_kernel2(mem, trace)
    next(gen)  # Initial yield (before rounds)
    try:
        next(gen)  # Final yield (after rounds) - trace should be populated now
    except StopIteration:
        pass  # Generator finished

    return mem, trace


def validate_kernel(
    rounds: int = 16,
    batch_size: int = 256,
    forest_height: int = 6,
    seed: int = 123,
    max_failures: int = 10,
    verbose: bool = False,
) -> ValidationResult:
    """
    Run kernel with debug validation and return first point of divergence.

    Args:
        rounds: Number of traversal rounds
        batch_size: Batch size (must be multiple of VLEN)
        forest_height: Height of the tree
        seed: Random seed for reproducibility
        max_failures: Maximum number of failures to collect
        verbose: Print progress information

    Returns:
        ValidationResult with details of any failures
    """
    from solver.interleaved_unrolled_kernel import InterleavedUnrolledKernelBuilder

    if verbose:
        print(f"Generating reference trace (rounds={rounds}, batch={batch_size})...")

    # Generate reference trace
    random.seed(seed)
    tree = Tree.generate(forest_height)
    inp = Input.generate(tree, batch_size, rounds)
    mem = build_mem_image(tree, inp)

    trace = {}
    gen = reference_kernel2(mem, trace)
    next(gen)  # Initial yield (before rounds)
    next(gen)  # Final yield (after rounds) - trace is now populated

    if verbose:
        print(f"Reference trace has {len(trace)} entries")

    # Reset and rebuild memory for kernel execution
    random.seed(seed)
    tree = Tree.generate(forest_height)
    inp = Input.generate(tree, batch_size, rounds)
    mem = build_mem_image(tree, inp)

    n_nodes = 2 ** (forest_height + 1) - 1

    if verbose:
        print("Building kernel with debug enabled...")

    # Build kernel with debug enabled
    builder = InterleavedUnrolledKernelBuilder(enable_debug=True)
    builder.build_kernel(
        forest_height=forest_height,
        n_nodes=n_nodes,
        batch_size=batch_size,
        rounds=rounds,
        use_greedy=True,  # Fast scheduling for validation
        require_optimal=False,
    )

    if verbose:
        print(f"Kernel has {len(builder.instrs)} instructions")
        debug_count = sum(1 for instr in builder.instrs if 'debug' in instr)
        print(f"Debug instructions: {debug_count}")

    # Run machine with value trace
    machine = Machine(
        mem_dump=mem,
        program=builder.instrs,
        debug_info=builder.debug_info(),
        value_trace=trace,
        trace=False,  # Disable file tracing
    )
    machine.enable_debug = False  # Disable machine's debug assertions - we check manually
    machine.enable_pause = False  # Disable pause instructions for validation

    failures = []
    total_comparisons = 0

    # Custom step-through to catch assertion errors
    try:
        for core in machine.cores:
            if core.state.value == 2:  # PAUSED
                core.state = machine.cores[0].state.__class__(1)  # RUNNING

        while any(c.state.value == 1 for c in machine.cores):  # RUNNING
            for core in machine.cores:
                if core.state.value != 1:
                    continue
                if core.pc >= len(machine.program):
                    core.state = core.state.__class__(3)  # STOPPED
                    continue

                instr = machine.program[core.pc]
                core.pc += 1

                # Execute the instruction first (skip debug engine)
                machine.step(instr, core)

                # Now check debug instructions after values are in scratch
                if 'debug' in instr:
                    for slot in instr['debug']:
                        if slot[0] == 'vcompare':
                            loc, keys = slot[1], slot[2]
                            for vi, key in enumerate(keys):
                                total_comparisons += 1
                                if key not in trace:
                                    if verbose:
                                        print(f"Warning: trace key not found: {key}")
                                    continue
                                expected = trace[key]
                                got = core.scratch[loc + vi]
                                if got != expected:
                                    # Parse key to extract round, element, phase
                                    rnd = key[0]
                                    batch_i = key[1]
                                    phase = key[2] if len(key) > 2 else "unknown"
                                    if len(key) > 3:
                                        phase = f"{phase}_{key[3]}"
                                    group = batch_i // VLEN
                                    element = batch_i % VLEN

                                    failure = ValidationFailure(
                                        round=rnd,
                                        group=group,
                                        element=element,
                                        phase=phase,
                                        expected=expected,
                                        got=got,
                                        trace_key=key,
                                    )
                                    failures.append(failure)
                                    if len(failures) >= max_failures:
                                        return ValidationResult(
                                            passed=False,
                                            first_failure=failures[0],
                                            total_comparisons=total_comparisons,
                                            failed_comparisons=len(failures),
                                            failures=failures,
                                        )
                        elif slot[0] == 'compare':
                            loc, key = slot[1], slot[2]
                            total_comparisons += 1
                            if key not in trace:
                                if verbose:
                                    print(f"Warning: trace key not found: {key}")
                                continue
                            expected = trace[key]
                            got = core.scratch[loc]
                            if got != expected:
                                rnd = key[0]
                                batch_i = key[1]
                                phase = key[2] if len(key) > 2 else "unknown"
                                group = batch_i // VLEN
                                element = batch_i % VLEN

                                failure = ValidationFailure(
                                    round=rnd,
                                    group=group,
                                    element=element,
                                    phase=phase,
                                    expected=expected,
                                    got=got,
                                    trace_key=key,
                                )
                                failures.append(failure)
                                if len(failures) >= max_failures:
                                    return ValidationResult(
                                        passed=False,
                                        first_failure=failures[0],
                                        total_comparisons=total_comparisons,
                                        failed_comparisons=len(failures),
                                        failures=failures,
                                    )

            machine.cycle += 1

    except Exception as e:
        if verbose:
            print(f"Execution error: {e}")
        raise

    if failures:
        return ValidationResult(
            passed=False,
            first_failure=failures[0],
            total_comparisons=total_comparisons,
            failed_comparisons=len(failures),
            failures=failures,
        )

    return ValidationResult(
        passed=True,
        first_failure=None,
        total_comparisons=total_comparisons,
        failed_comparisons=0,
        failures=[],
    )


def validate_final_output(
    rounds: int = 16,
    batch_size: int = 256,
    forest_height: int = 6,
    seed: int = 123,
    verbose: bool = False,
) -> tuple[bool, list[tuple[int, int, int]]]:
    """
    Validate final kernel output against reference (without debug instructions).

    Returns:
        (passed, mismatches): Tuple of (bool, list of (index, expected, got))
    """
    from solver.interleaved_unrolled_kernel import InterleavedUnrolledKernelBuilder

    # Generate reference
    random.seed(seed)
    tree = Tree.generate(forest_height)
    inp = Input.generate(tree, batch_size, rounds)
    ref_mem = build_mem_image(tree, inp)

    gen = reference_kernel2(ref_mem, {})
    next(gen)
    next(gen)

    # Run kernel
    random.seed(seed)
    tree = Tree.generate(forest_height)
    inp = Input.generate(tree, batch_size, rounds)
    test_mem = build_mem_image(tree, inp)

    n_nodes = 2 ** (forest_height + 1) - 1

    builder = InterleavedUnrolledKernelBuilder(enable_debug=False)
    builder.build_kernel(
        forest_height=forest_height,
        n_nodes=n_nodes,
        batch_size=batch_size,
        rounds=rounds,
        use_greedy=True,
        require_optimal=False,
    )

    machine = Machine(
        mem_dump=test_mem,
        program=builder.instrs,
        debug_info=builder.debug_info(),
        trace=False,
    )
    machine.enable_pause = False  # Run through pause instructions
    machine.run()

    # Compare outputs
    inp_indices_p = test_mem[5]
    inp_values_p = test_mem[6]

    mismatches = []
    for i in range(batch_size):
        idx_expected = ref_mem[inp_indices_p + i]
        idx_got = machine.mem[inp_indices_p + i]
        if idx_expected != idx_got:
            mismatches.append((f"idx[{i}]", idx_expected, idx_got))

        val_expected = ref_mem[inp_values_p + i]
        val_got = machine.mem[inp_values_p + i]
        if val_expected != val_got:
            mismatches.append((f"val[{i}]", val_expected, val_got))

    return len(mismatches) == 0, mismatches
