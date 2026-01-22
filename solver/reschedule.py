"""
Re-scheduler for VLIW kernels.

Takes a list of instructions and re-schedules them for better packing
while respecting data dependencies.
"""

from dataclasses import dataclass, field
from typing import Optional
from solver.ir import IRProgram, OpType, Value, Operation, OP_TO_ENGINE

# Vector length for SIMD operations
VLEN = 8


@dataclass
class Op:
    """Single operation extracted from instruction."""
    id: int
    engine: str
    slot: tuple
    cycle: int = -1
    dep_ids: set[int] = field(default_factory=set)  # IDs of ops we depend on

    # Track which values this op reads/writes
    reads: set[int] = field(default_factory=set)
    writes: set[int] = field(default_factory=set)

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        return isinstance(other, Op) and self.id == other.id


def extract_ops(instrs: list[dict]) -> list[Op]:
    """Extract all operations from instructions with their dependencies."""
    ops = []
    op_id = 0

    for cycle, instr in enumerate(instrs):
        for engine, slots in instr.items():
            for slot in slots:
                op = Op(id=op_id, engine=engine, slot=slot, cycle=cycle)
                op_id += 1

                # Parse reads/writes based on operation type
                # Vector operations touch VLEN consecutive addresses
                if engine == 'load':
                    op_type = slot[0]
                    if op_type == 'const':
                        op.writes.add(slot[1])
                    elif op_type == 'load':
                        op.writes.add(slot[1])
                        op.reads.add(slot[2])
                    elif op_type == 'vload':
                        # Vector load writes VLEN consecutive addresses
                        for i in range(VLEN):
                            op.writes.add(slot[1] + i)
                        op.reads.add(slot[2])
                elif engine == 'store':
                    op_type = slot[0]
                    if op_type == 'store':
                        op.reads.add(slot[1])  # address
                        op.reads.add(slot[2])  # data
                    elif op_type == 'vstore':
                        op.reads.add(slot[1])  # address
                        # Vector store reads VLEN consecutive addresses
                        for i in range(VLEN):
                            op.reads.add(slot[2] + i)
                elif engine == 'alu':
                    op.writes.add(slot[1])
                    op.reads.add(slot[2])
                    op.reads.add(slot[3])
                elif engine == 'valu':
                    op_type = slot[0]
                    if op_type == 'vbroadcast':
                        # Broadcasts scalar to all VLEN elements
                        for i in range(VLEN):
                            op.writes.add(slot[1] + i)
                        op.reads.add(slot[2])
                    elif op_type in ('vmultiply_add', 'multiply_add'):
                        # dest = src1 * src2 + src3 (all vectors)
                        for i in range(VLEN):
                            op.writes.add(slot[1] + i)
                            op.reads.add(slot[2] + i)
                            op.reads.add(slot[3] + i)
                            op.reads.add(slot[4] + i)
                    else:
                        # Binary vector ops: dest = src1 op src2
                        for i in range(VLEN):
                            op.writes.add(slot[1] + i)
                            op.reads.add(slot[2] + i)
                            op.reads.add(slot[3] + i)
                elif engine == 'flow':
                    op_type = slot[0]
                    if op_type == 'vselect':
                        # Vector select: all VLEN elements
                        for i in range(VLEN):
                            op.writes.add(slot[1] + i)
                            op.reads.add(slot[2] + i)
                            op.reads.add(slot[3] + i)
                            op.reads.add(slot[4] + i)
                    elif op_type == 'cond_jump':
                        op.reads.add(slot[1])
                    # pause has no reads/writes

                ops.append(op)

    return ops


def compute_dependencies(ops: list[Op]) -> dict[int, Op]:
    """Compute data dependencies between operations.

    Tracks three types of dependencies:
    - RAW (Read-After-Write): read depends on prior write
    - WAR (Write-After-Read): write must wait for prior reads to complete
    - WAW (Write-After-Write): write must wait for prior write
    """
    # Map: address -> last op that wrote to it
    last_writer: dict[int, Op] = {}
    # Map: address -> list of ops that read it since last write
    readers_since_write: dict[int, list[Op]] = {}
    # Map: op id -> Op
    op_by_id: dict[int, Op] = {op.id: op for op in ops}

    for op in ops:
        # RAW: Add dependencies on writers of values we read
        for addr in op.reads:
            if addr in last_writer:
                op.dep_ids.add(last_writer[addr].id)
            # Track this read for WAR dependencies
            if addr not in readers_since_write:
                readers_since_write[addr] = []
            readers_since_write[addr].append(op)

        # WAR + WAW: Add dependencies for values we write
        for addr in op.writes:
            # WAW: depend on previous write
            if addr in last_writer:
                op.dep_ids.add(last_writer[addr].id)
            # WAR: depend on all readers since last write (except self!)
            if addr in readers_since_write:
                for reader in readers_since_write[addr]:
                    if reader.id != op.id:  # Don't create self-dependency
                        op.dep_ids.add(reader.id)
                readers_since_write[addr] = []  # Clear readers

            # Update writer
            last_writer[addr] = op

    return op_by_id


def schedule_greedy(ops: list[Op], op_by_id: dict[int, Op]) -> int:
    """Schedule operations using greedy list scheduling with priority."""
    SLOT_LIMITS = {
        'load': 2,
        'store': 2,
        'alu': 12,
        'valu': 6,
        'flow': 1,
    }

    # Compute critical path length for each op (longest path to any sink)
    # This helps prioritize operations on the critical path
    critical_path = {}

    # Build successor lists and compute out-degree
    successors = {op.id: [] for op in ops}
    out_degree = {op.id: 0 for op in ops}
    for op in ops:
        for dep_id in op.dep_ids:
            successors[dep_id].append(op.id)
            out_degree[dep_id] += 1

    # Compute critical path iteratively in reverse topological order
    # Start from sinks (ops with no successors)
    work_queue = [op.id for op in ops if not successors[op.id]]
    for op_id in work_queue:
        critical_path[op_id] = 1

    # Process in reverse order
    remaining_out = dict(out_degree)
    while work_queue:
        op_id = work_queue.pop(0)
        op = op_by_id[op_id]
        my_cp = critical_path[op_id]
        for dep_id in op.dep_ids:
            # Update predecessor's critical path
            if dep_id not in critical_path:
                critical_path[dep_id] = 0
            critical_path[dep_id] = max(critical_path[dep_id], my_cp + 1)
            remaining_out[dep_id] -= 1
            if remaining_out[dep_id] == 0:
                work_queue.append(dep_id)

    # Topological sort based on dependencies
    in_degree = {op.id: len(op.dep_ids) for op in ops}
    ready = [op for op in ops if in_degree[op.id] == 0]

    # Track slot usage per cycle
    slot_usage: dict[int, dict[str, int]] = {}

    def can_schedule(op: Op, cycle: int) -> bool:
        if cycle not in slot_usage:
            return True
        usage = slot_usage[cycle]
        limit = SLOT_LIMITS.get(op.engine, 1)
        return usage.get(op.engine, 0) < limit

    def earliest_cycle(op: Op) -> int:
        if not op.dep_ids:
            return 0
        return max(op_by_id[dep_id].cycle + 1 for dep_id in op.dep_ids)

    def priority(op: Op) -> tuple:
        # Higher priority = scheduled first
        # 1. Critical path (higher = more important)
        # 2. Original order (lower = earlier) - preserves locality
        # 3. VALU ops (to fill the 6 slots)
        return (-critical_path.get(op.id, 0), op.id, op.engine != 'valu')

    scheduled = []
    while ready:
        # Sort ready queue by priority
        ready.sort(key=priority)
        op = ready.pop(0)
        cycle = earliest_cycle(op)
        while not can_schedule(op, cycle):
            cycle += 1

        op.cycle = cycle
        scheduled.append(op)

        if cycle not in slot_usage:
            slot_usage[cycle] = {}
        slot_usage[cycle][op.engine] = slot_usage[cycle].get(op.engine, 0) + 1

        # Update ready list
        for other in ops:
            if op.id in other.dep_ids:
                in_degree[other.id] -= 1
                if in_degree[other.id] == 0:
                    ready.append(other)

    return max(op.cycle for op in ops) + 1 if ops else 0


def emit_scheduled(ops: list[Op], max_cycle: int) -> list[dict]:
    """Emit instructions from scheduled operations."""
    instrs = []

    for cycle in range(max_cycle):
        instr = {}
        for op in ops:
            if op.cycle == cycle:
                if op.engine not in instr:
                    instr[op.engine] = []
                instr[op.engine].append(op.slot)
        if instr:
            instrs.append(instr)

    return instrs


def reschedule(instrs: list[dict]) -> list[dict]:
    """Re-schedule instructions for better packing."""
    # Extract operations
    ops = extract_ops(instrs)

    # Compute dependencies
    op_by_id = compute_dependencies(ops)

    # Schedule
    max_cycle = schedule_greedy(ops, op_by_id)

    # Emit
    return emit_scheduled(ops, max_cycle)


def reschedule_section(instrs: list[dict], start: int, end: int) -> list[dict]:
    """Re-schedule a section of instructions."""
    section = instrs[start:end]
    rescheduled = reschedule(section)
    return instrs[:start] + rescheduled + instrs[end:]
