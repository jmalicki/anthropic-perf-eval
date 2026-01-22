"""
Chunked scheduler - solves smaller subproblems optimally with CP-SAT.

Strategy:
1. Divide operations into chunks based on dependencies
2. Solve each chunk optimally
3. Concatenate results
"""

from solver.reschedule import Op, extract_ops, compute_dependencies, schedule_greedy, emit_scheduled
from solver.cp_scheduler import schedule_cp, emit_scheduled_cp


def schedule_chunked(ops: list[Op], op_by_id: dict[int, Op],
                     chunk_size: int = 1000, time_limit: float = 30.0) -> tuple[int, list[dict]]:
    """Schedule operations in chunks."""

    if len(ops) <= chunk_size:
        # Small enough for direct CP-SAT
        result = schedule_cp(ops, op_by_id, time_limit=time_limit)
        if result.success:
            return result.cycles, emit_scheduled_cp(ops, result.cycles)
        else:
            # Fall back to greedy
            num_cycles = schedule_greedy(ops, op_by_id)
            return num_cycles, emit_scheduled(ops, num_cycles)

    # Find natural chunk boundaries based on dependency structure
    # Strategy: chunk ops by their earliest possible start time

    # First pass: compute earliest start time for each op
    earliest = {}
    def compute_earliest(op_id):
        if op_id in earliest:
            return earliest[op_id]
        op = op_by_id[op_id]
        if not op.dep_ids:
            earliest[op_id] = 0
        else:
            earliest[op_id] = 1 + max(compute_earliest(d) for d in op.dep_ids)
        return earliest[op_id]

    for op in ops:
        compute_earliest(op.id)

    # Group ops by their earliest start time range
    max_earliest = max(earliest.values())
    chunk_ranges = (max_earliest + 1) // chunk_size + 1

    # Actually, let's just chunk by original order with overlap handling
    # Sort ops topologically
    sorted_ops = sorted(ops, key=lambda op: (earliest[op.id], op.id))

    all_instrs = []
    offset = 0

    i = 0
    while i < len(sorted_ops):
        # Take next chunk
        chunk_end = min(i + chunk_size, len(sorted_ops))
        chunk_ops = sorted_ops[i:chunk_end]

        # Extend chunk to include all ops that depend on ops in chunk
        # (to avoid splitting dependency chains)
        chunk_ids = {op.id for op in chunk_ops}

        # Find all ops that are depended upon by chunk ops
        # These must be scheduled before the chunk
        prereqs = set()
        for op in chunk_ops:
            prereqs.update(op.dep_ids)
        prereqs -= chunk_ids

        # Adjust cycles for ops in this chunk based on offset
        for op in chunk_ops:
            # Update dep_ids to point to prereqs with known cycles
            pass

        # Create modified op list for this chunk
        # Prereqs already have assigned cycles from previous chunks
        chunk_op_by_id = {op.id: op for op in chunk_ops}

        # Adjust dependencies: prereqs have cycles, use them as constraints
        for op in chunk_ops:
            # dep_ids pointing to prereqs are already scheduled
            pass

        # Schedule this chunk
        result = schedule_cp(chunk_ops, chunk_op_by_id, time_limit=time_limit)

        if result.success:
            # Adjust cycles by offset
            for op in chunk_ops:
                op.cycle += offset

            chunk_max = max(op.cycle for op in chunk_ops) + 1
            offset = chunk_max

            print(f"  Chunk {i//chunk_size}: {len(chunk_ops)} ops -> {result.cycles} cycles")
        else:
            # Fall back to greedy for this chunk
            num_cycles = schedule_greedy(chunk_ops, chunk_op_by_id)
            for op in chunk_ops:
                op.cycle += offset
            offset += num_cycles
            print(f"  Chunk {i//chunk_size}: {len(chunk_ops)} ops -> {num_cycles} cycles (greedy fallback)")

        i = chunk_end

    # Emit all instructions
    max_cycle = max(op.cycle for op in ops) + 1
    return max_cycle, emit_scheduled(ops, max_cycle)


def schedule_by_round(instrs: list[dict], n_groups: int, rounds: int) -> list[dict]:
    """Schedule by processing each round separately, then stitching."""
    # This would require understanding the structure of the kernel
    # For now, use chunked approach
    pass
