"""
Round-chunked scheduler - solves each round with CP-SAT, then stitches.

Key insight: Operations within a round can be optimized independently,
with only the boundary constraints mattering for stitching.
"""

from solver.reschedule import Op, schedule_greedy, emit_scheduled
from solver.cp_scheduler import schedule_cp, emit_scheduled_cp


def schedule_round_chunked(ops: list[Op], op_by_id: dict[int, Op],
                           rounds: int, n_iters: int, unroll: int,
                           time_limit: float = 30.0) -> tuple[int, list[dict]]:
    """Schedule operations in round-sized chunks."""

    if len(ops) <= 3000:
        # Small enough for direct CP-SAT
        print(f"  Direct CP-SAT on {len(ops)} ops...")
        result = schedule_cp(ops, op_by_id, time_limit=time_limit)
        if result.success:
            return result.cycles, emit_scheduled_cp(ops, result.cycles)
        else:
            print(f"  CP-SAT failed ({result.status}), falling back to greedy")
            num_cycles = schedule_greedy(ops, op_by_id)
            return num_cycles, emit_scheduled(ops, num_cycles)

    # Compute earliest start time for each op (for topological ordering)
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

    # Sort ops by earliest start time
    sorted_ops = sorted(ops, key=lambda op: (earliest[op.id], op.id))

    # Estimate ops per round to create chunks
    ops_per_round = len(ops) // rounds if rounds > 0 else len(ops)
    chunk_size = max(1500, ops_per_round)  # At least 1500 ops per chunk for CP-SAT efficiency

    offset = 0
    i = 0
    chunk_num = 0

    while i < len(sorted_ops):
        # Take next chunk
        chunk_end = min(i + chunk_size, len(sorted_ops))
        chunk_ops = sorted_ops[i:chunk_end]

        # Build dependency map for this chunk
        chunk_ids = {op.id for op in chunk_ops}
        chunk_op_by_id = {op.id: op for op in chunk_ops}

        # Find prerequisites (ops in previous chunks that this chunk depends on)
        prereqs = set()
        for op in chunk_ops:
            for dep_id in op.dep_ids:
                if dep_id not in chunk_ids:
                    prereqs.add(dep_id)

        # For this chunk, compute minimum start time based on prereqs
        # Each op's local start must be >= (prereq.cycle + 1 - offset) for all prereqs
        min_starts = {}
        for op in chunk_ops:
            local_min = 0
            for dep_id in op.dep_ids:
                if dep_id in prereqs:
                    prereq_op = op_by_id[dep_id]
                    local_min = max(local_min, prereq_op.cycle + 1 - offset)
            min_starts[op.id] = local_min

        min_start = min(min_starts.values()) if min_starts else 0

        # Adjust op dependencies to remove prereqs (they're already scheduled)
        for op in chunk_ops:
            op.dep_ids = [d for d in op.dep_ids if d in chunk_ids]

        # Schedule this chunk
        print(f"  Chunk {chunk_num}: {len(chunk_ops)} ops, min_start={min_start}...")

        # Try CP-SAT first
        try:
            result = schedule_cp(chunk_ops, chunk_op_by_id, time_limit=time_limit//(rounds//2 + 1))
            if result.success:
                # Adjust cycles by offset and per-op min_start
                for op in chunk_ops:
                    local_cycle = result.schedule[op.id]
                    op_min = min_starts.get(op.id, 0)
                    op.cycle = offset + max(local_cycle, op_min)

                chunk_max = max(op.cycle for op in chunk_ops) + 1
                cycles_used = chunk_max - offset

                print(f"    -> {cycles_used} cycles (CP-SAT {result.status})")
                offset = chunk_max
            else:
                raise ValueError("CP-SAT failed")
        except Exception as e:
            # Fall back to greedy
            # Reset cycles for greedy
            for op in chunk_ops:
                op.cycle = None
            num_cycles = schedule_greedy(chunk_ops, chunk_op_by_id)

            # Apply per-op offsets
            for op in chunk_ops:
                op_min = min_starts.get(op.id, 0)
                op.cycle = offset + max(op.cycle, op_min)

            chunk_max = max(op.cycle for op in chunk_ops) + 1
            cycles_used = chunk_max - offset
            offset = chunk_max
            print(f"    -> {cycles_used} cycles (greedy fallback)")

        i = chunk_end
        chunk_num += 1

    # Emit all instructions
    max_cycle = max(op.cycle for op in ops) + 1
    return max_cycle, emit_scheduled(ops, max_cycle)
