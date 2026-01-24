"""
Test helper utilities for kernel validation.

Provides utilities for building minimal test kernels, counting operations,
and verifying scratch memory bounds.
"""

from typing import Optional
from problem import VLEN, SCRATCH_SIZE, SLOT_LIMITS


def build_mini_kernel(
    rounds: int = 1,
    batch_size: int = 64,  # Minimum 64 (UNROLL * VLEN) for kernel's iteration structure
    forest_height: int = 6,
    use_greedy: bool = True,
    enable_debug: bool = False,
):
    """
    Build a minimal kernel for fast testing.

    Uses greedy scheduler by default for speed. Returns the builder
    and generated instructions.

    Args:
        rounds: Number of traversal rounds
        batch_size: Batch size (must be multiple of VLEN)
        forest_height: Height of the tree
        use_greedy: Use fast greedy scheduler (default True)
        enable_debug: Enable debug instruction emission

    Returns:
        builder: The InterleavedUnrolledKernelBuilder instance
    """
    from solver.interleaved_unrolled_kernel import InterleavedUnrolledKernelBuilder

    n_nodes = 2 ** (forest_height + 1) - 1
    builder = InterleavedUnrolledKernelBuilder(enable_debug=enable_debug)
    builder.build_kernel(
        forest_height=forest_height,
        n_nodes=n_nodes,
        batch_size=batch_size,
        rounds=rounds,
        use_greedy=use_greedy,
        require_optimal=False,
    )
    return builder


def count_ops_by_type(instrs: list, engine: str, op_type: Optional[str] = None) -> int:
    """
    Count operations of a specific type in instruction list.

    Args:
        instrs: List of instruction dictionaries
        engine: Engine name ('alu', 'valu', 'load', 'store', 'flow', 'debug')
        op_type: Optional operation type to filter (e.g., '+', 'vbroadcast', 'multiply_add')

    Returns:
        Count of matching operations
    """
    count = 0
    for instr in instrs:
        if engine in instr:
            for slot in instr[engine]:
                if op_type is None:
                    count += 1
                elif slot[0] == op_type:
                    count += 1
    return count


def verify_scratch_bounds(builder) -> tuple[bool, list[str]]:
    """
    Check that all allocated scratch addresses are within SCRATCH_SIZE.

    Args:
        builder: InterleavedUnrolledKernelBuilder instance

    Returns:
        (passed, errors): Tuple of (bool, list of error messages)
    """
    errors = []

    if builder.scratch_ptr > SCRATCH_SIZE:
        errors.append(
            f"Scratch pointer {builder.scratch_ptr} exceeds SCRATCH_SIZE {SCRATCH_SIZE}"
        )

    for addr, (name, length) in builder.scratch_debug.items():
        end_addr = addr + length
        if end_addr > SCRATCH_SIZE:
            errors.append(
                f"Allocation '{name}' at {addr} with length {length} "
                f"extends to {end_addr}, exceeding SCRATCH_SIZE {SCRATCH_SIZE}"
            )

    return len(errors) == 0, errors


def extract_phase_ops(instrs: list, phase_name: str) -> list:
    """
    Extract instructions that contain debug markers for a specific phase.

    Args:
        instrs: List of instruction dictionaries
        phase_name: Phase name to look for in debug markers

    Returns:
        List of matching instructions
    """
    matching = []
    for instr in instrs:
        if 'debug' in instr:
            for slot in instr['debug']:
                if slot[0] == 'vcompare' and len(slot) >= 3:
                    keys = slot[2]
                    if any(phase_name in str(key) for key in keys):
                        matching.append(instr)
                        break
    return matching


def get_scratch_allocation_info(builder) -> dict:
    """
    Get information about scratch memory allocations.

    Args:
        builder: InterleavedUnrolledKernelBuilder instance

    Returns:
        Dict with allocation statistics
    """
    scalar_count = 0
    vector_count = 0
    scalar_words = 0
    vector_words = 0

    for addr, (name, length) in builder.scratch_debug.items():
        if length == VLEN:
            vector_count += 1
            vector_words += length
        else:
            scalar_count += 1
            scalar_words += length

    return {
        'total_words': builder.scratch_ptr,
        'scalar_allocations': scalar_count,
        'vector_allocations': vector_count,
        'scalar_words': scalar_words,
        'vector_words': vector_words,
        'utilization': builder.scratch_ptr / SCRATCH_SIZE,
    }


def validate_slot_limits(instrs: list) -> tuple[bool, list[str]]:
    """
    Validate that no instruction exceeds slot limits for any engine.

    Args:
        instrs: List of instruction dictionaries

    Returns:
        (passed, errors): Tuple of (bool, list of error messages)
    """
    errors = []

    for i, instr in enumerate(instrs):
        for engine, slots in instr.items():
            if engine in SLOT_LIMITS:
                limit = SLOT_LIMITS[engine]
                if len(slots) > limit:
                    errors.append(
                        f"Instruction {i}: {engine} has {len(slots)} slots, "
                        f"exceeds limit {limit}"
                    )

    return len(errors) == 0, errors
