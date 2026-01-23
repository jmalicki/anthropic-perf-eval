"""
Round-chunked scheduler - solves each round with CP-SAT, then stitches.

Key insight: Operations within a round can be optimized independently,
with only the boundary constraints mattering for stitching.

Two-phase approach:
1. Chunked solve for initial solution
2. Full solve using chunked as warm start (optional, for better results)
"""

from solver.reschedule import Op, emit_scheduled
from solver.cp_scheduler import schedule_cp, emit_scheduled_cp


def schedule_round_chunked(ops: list[Op], op_by_id: dict[int, Op],
                           rounds: int, n_iters: int, unroll: int,
                           time_limit: float = 30.0,
                           full_solve_time: float = 0.0) -> tuple[int, list[dict]]:
    """Schedule operations in round-sized chunks."""

    if len(ops) <= 3000:
        # Small enough for direct CP-SAT
        print(f"  Direct CP-SAT on {len(ops)} ops...")
        result = schedule_cp(ops, op_by_id, time_limit=time_limit, log_file='cpsat_direct.log')
        if not result.success:
            raise RuntimeError(f"CP-SAT failed: {result.status}")
        return result.cycles, emit_scheduled_cp(ops, result.cycles)

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

        # Schedule with CP-SAT
        result = schedule_cp(chunk_ops, chunk_op_by_id, time_limit=time_limit//(rounds//2 + 1),
                            log_file='cpsat_chunks.log')
        if not result.success:
            raise RuntimeError(f"CP-SAT failed for chunk {chunk_num}: {result.status}")

        # Adjust cycles by offset and per-op min_start
        for op in chunk_ops:
            local_cycle = result.schedule[op.id]
            op_min = min_starts.get(op.id, 0)
            op.cycle = offset + max(local_cycle, op_min)

        chunk_max = max(op.cycle for op in chunk_ops) + 1
        cycles_used = chunk_max - offset

        print(f"    -> {cycles_used} cycles (CP-SAT {result.status})")
        offset = chunk_max

        i = chunk_end
        chunk_num += 1

    # Phase 1 complete - we have chunked solution
    chunked_cycles = max(op.cycle for op in ops) + 1
    print(f"Chunked solution: {chunked_cycles} cycles")

    # Phase 2 (optional): Use chunked as hint for full solve
    # KEY: We must restore ORIGINAL dependencies and let the solver
    # freely explore - the chunked hint is just a starting point,
    # NOT a set of boundary constraints.
    if full_solve_time > 0:
        print(f"Phase 2: Full solve with chunked hint ({full_solve_time}s)...")
        chunked_hint = {op.id: op.cycle for op in ops}

        # CRITICAL: Restore original dependencies for full solve
        # The chunked solve modified dep_ids to filter external deps,
        # which artificially constrained the search space.
        # We need fresh, complete dependencies.
        from solver.reschedule import compute_dependencies as recompute_deps

        # Reset all dependencies and recompute from scratch
        for op in ops:
            op.dep_ids = set()
            op.cycle = -1  # Clear cycle assignments

        # Rebuild op_by_id with fresh dependency computation
        fresh_op_by_id = recompute_deps(ops)

        # Now run full solve with complete dependencies
        # The hint helps start from a good solution but doesn't lock boundaries
        result = schedule_cp(ops, fresh_op_by_id, time_limit=full_solve_time, greedy_hint=chunked_hint,
                            log_file='cpsat_fullsolve.log')
        if result.success and result.cycles < chunked_cycles:
            print(f"Full solve improved: {result.cycles} cycles ({result.status})")
            return result.cycles, emit_scheduled(ops, result.cycles)
        else:
            status_msg = f"{result.cycles} cycles" if result.success else "failed"
            print(f"Full solve didn't improve (got {status_msg})")
            # Restore chunked solution
            for op in ops:
                op.cycle = chunked_hint[op.id]

    # Emit all instructions
    max_cycle = max(op.cycle for op in ops) + 1
    return max_cycle, emit_scheduled(ops, max_cycle)


def compute_dependencies_inplace(ops: list[Op]):
    """Recompute dependencies for ops (modifies in place)."""
    last_writer = {}
    readers_since_write = {}

    for op in ops:
        op.dep_ids = set()

        # RAW: depend on writers of values we read
        for addr in op.reads:
            if addr in last_writer:
                op.dep_ids.add(last_writer[addr].id)
            if addr not in readers_since_write:
                readers_since_write[addr] = []
            readers_since_write[addr].append(op)

        # WAR + WAW: depend for values we write
        for addr in op.writes:
            if addr in last_writer:
                op.dep_ids.add(last_writer[addr].id)
            if addr in readers_since_write:
                for reader in readers_since_write[addr]:
                    if reader.id != op.id:
                        op.dep_ids.add(reader.id)
                readers_since_write[addr] = []
            last_writer[addr] = op
