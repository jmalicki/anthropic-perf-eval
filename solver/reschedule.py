"""
Re-scheduler for VLIW kernels.

Takes a list of instructions and re-schedules them for better packing
while respecting data dependencies.

Provides two scheduling approaches:
1. schedule_greedy() - Fast O(V+E) list scheduling with critical path priority
2. CP-SAT (external) - Optimal but slow ILP-based scheduling

For most cases, the greedy scheduler achieves near-optimal results in seconds,
while CP-SAT can take hours for large kernels.
"""

from dataclasses import dataclass, field
from typing import Optional

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
                elif engine == 'debug':
                    # Debug ops read the values they're comparing
                    op_type = slot[0]
                    if op_type == 'vcompare':
                        # vcompare reads VLEN consecutive addresses from loc
                        loc = slot[1]
                        for i in range(VLEN):
                            op.reads.add(loc + i)
                    elif op_type == 'compare':
                        # compare reads a single address
                        op.reads.add(slot[1])

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


# VALU ops that can be converted to scalar ALU ops (binary operations)
VALU_TO_ALU_OPS = {'+', '-', '*', '/', '%', '&', '|', '^', '<<', '>>', '==', '!=', '<', '<=', '>', '>='}


def can_convert_valu_to_alu(op: Op) -> bool:
    """Check if a VALU op can be converted to 8 scalar ALU ops."""
    if op.engine != 'valu':
        return False
    op_type = op.slot[0]
    # Binary vector ops can be converted
    return op_type in VALU_TO_ALU_OPS


def expand_valu_to_alu(op: Op, next_op_id: int) -> list[Op]:
    """Expand a VALU op into 8 scalar ALU ops.

    Returns a list of 8 new Op objects that perform the same computation
    element-wise on the ALU.
    """
    op_type, dest, src1, src2 = op.slot
    new_ops = []

    for i in range(VLEN):
        new_op = Op(
            id=next_op_id + i,
            engine='alu',
            slot=(op_type, dest + i, src1 + i, src2 + i),
            dep_ids=set(op.dep_ids),  # Copy dependencies
            reads={src1 + i, src2 + i},
            writes={dest + i},
        )
        new_ops.append(new_op)

    return new_ops


def schedule_greedy(
    ops: list[Op],
    op_by_id: dict[int, Op],
    enable_valu_shift: bool = False,
    priority_mode: str = "critical_path",
) -> tuple[int, list[Op]]:
    """Schedule operations using greedy list scheduling with configurable priority.

    This is an O(V+E) algorithm that achieves near-optimal results for structured
    problems with high parallelism (like VLIW kernels with independent batch groups).

    Algorithm:
    1. Compute critical path length for each op (longest path to any sink)
    2. Initialize ready queue with ops that have no dependencies
    3. Greedily schedule highest-priority ops that fit in available slots
    4. Update ready queue as dependencies are satisfied

    Args:
        ops: List of operations to schedule
        op_by_id: Dictionary mapping op IDs to Op objects
        enable_valu_shift: If True, convert VALU ops to ALU ops when VALU is
                          saturated but ALU has free slots
        priority_mode: Priority strategy for ready queue ordering:
            - "critical_path": Prioritize ops on critical path (default)
            - "critical_path_valu": Same but prioritize VALU ops
            - "engine_balance": Balance engine utilization
            - "original_order": Schedule in original emission order

    Returns:
        Tuple of (num_cycles, scheduled_ops) where scheduled_ops includes any
        expanded ALU ops from VALU shifting.
    """
    SLOT_LIMITS = {
        'load': 2,
        'store': 2,
        'alu': 12,
        'valu': 6,
        'flow': 1,
        'debug': 64,  # High limit since debug ops don't affect performance
    }

    # Track expanded ops (VALU ops that were converted to ALU)
    # Maps original op ID -> list of expanded ALU op IDs
    expanded_ops = {}
    next_op_id = max(op.id for op in ops) + 1 if ops else 0

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

    def get_slot_usage(cycle: int, engine: str) -> int:
        if cycle not in slot_usage:
            return 0
        return slot_usage[cycle].get(engine, 0)

    def can_schedule(op: Op, cycle: int) -> bool:
        usage = get_slot_usage(cycle, op.engine)
        limit = SLOT_LIMITS.get(op.engine, 1)
        return usage < limit

    def get_free_alu_slots(start_cycle: int, num_cycles: int = 2) -> int:
        """Count free ALU slots across multiple cycles."""
        total = 0
        for c in range(start_cycle, start_cycle + num_cycles):
            total += SLOT_LIMITS['alu'] - get_slot_usage(c, 'alu')
        return total

    def earliest_cycle(op: Op) -> int:
        if not op.dep_ids:
            return 0
        return max(op_by_id[dep_id].cycle + 1 for dep_id in op.dep_ids)

    def priority(op: Op) -> tuple:
        # Higher priority = scheduled first (lower tuple = higher priority)
        cp = critical_path.get(op.id, 0)

        if priority_mode == "critical_path":
            # 1. Critical path (higher = more important)
            # 2. Original order (lower = earlier) - preserves locality
            # 3. VALU ops (to fill the 6 slots)
            return (-cp, op.id, op.engine != 'valu')
        elif priority_mode == "critical_path_valu":
            # Prioritize VALU first, then critical path
            return (op.engine != 'valu', -cp, op.id)
        elif priority_mode == "engine_balance":
            # Try to balance engine utilization by deprioritizing saturated engines
            # This can help find better schedules when one engine is overloaded
            engine_load = {
                'valu': 0.6,   # VALU has fewer slots (6), deprioritize slightly
                'alu': 1.0,    # ALU has many slots (12)
                'load': 0.8,   # LOAD has few slots (2)
                'store': 0.8,  # STORE has few slots (2)
                'flow': 0.5,   # FLOW is usually sparse
            }
            load_factor = engine_load.get(op.engine, 1.0)
            return (-cp * load_factor, op.id)
        elif priority_mode == "original_order":
            # Simple FIFO order with engine preference
            return (op.id, op.engine != 'valu')
        else:
            # Default to critical path
            return (-cp, op.id, op.engine != 'valu')

    def schedule_op(op: Op, cycle: int):
        """Mark an op as scheduled at the given cycle."""
        op.cycle = cycle
        if cycle not in slot_usage:
            slot_usage[cycle] = {}
        slot_usage[cycle][op.engine] = slot_usage[cycle].get(op.engine, 0) + 1

    scheduled = []
    valu_shifted_count = 0

    while ready:
        # Sort ready queue by priority
        ready.sort(key=priority)
        op = ready.pop(0)
        cycle = earliest_cycle(op)

        # Check if we can schedule as-is
        while not can_schedule(op, cycle):
            # VALU→ALU shifting: if VALU is full but ALU has capacity
            if (enable_valu_shift and op.engine == 'valu' and
                can_convert_valu_to_alu(op) and
                get_free_alu_slots(cycle) >= VLEN):

                # Expand VALU op to 8 ALU ops
                alu_ops = expand_valu_to_alu(op, next_op_id)
                next_op_id += VLEN

                # Schedule the 8 ALU ops across available cycles
                for alu_op in alu_ops:
                    alu_cycle = cycle
                    while not can_schedule(alu_op, alu_cycle):
                        alu_cycle += 1
                    schedule_op(alu_op, alu_cycle)
                    op_by_id[alu_op.id] = alu_op
                    scheduled.append(alu_op)
                    # Update critical path for new ops
                    critical_path[alu_op.id] = critical_path.get(op.id, 1)

                # Mark original VALU op as "done" by setting cycle to max of ALU ops
                op.cycle = max(a.cycle for a in alu_ops)
                expanded_ops[op.id] = [a.id for a in alu_ops]
                valu_shifted_count += 1
                break
            else:
                cycle += 1
        else:
            # Normal scheduling (didn't expand)
            schedule_op(op, cycle)
            scheduled.append(op)

        # Update ready list
        for other in ops:
            if op.id in other.dep_ids:
                in_degree[other.id] -= 1
                if in_degree[other.id] == 0:
                    ready.append(other)

    if valu_shifted_count > 0:
        print(f"  VALU→ALU shifted: {valu_shifted_count} ops")

    num_cycles = max(op.cycle for op in scheduled) + 1 if scheduled else 0
    return num_cycles, scheduled


def schedule_greedy_multipass(
    ops: list[Op],
    op_by_id: dict[int, Op],
    enable_valu_shift: bool = False,
) -> tuple[int, list[Op]]:
    """Run multiple greedy passes with different priority orderings and return the best.

    This can find better schedules by exploring different orderings of the ready queue.
    Each pass uses a different priority_mode to potentially find different local optima.

    Args:
        ops: List of operations to schedule
        op_by_id: Dictionary mapping op IDs to Op objects
        enable_valu_shift: If True, convert VALU ops to ALU ops when VALU is saturated

    Returns:
        Tuple of (num_cycles, scheduled_ops) from the best pass
    """
    priority_modes = [
        "critical_path",
        "critical_path_valu",
        "engine_balance",
        "original_order",
    ]

    best_cycles = float('inf')
    best_scheduled = None
    best_mode = None

    for mode in priority_modes:
        # Deep copy ops to avoid state pollution between passes
        ops_copy = [Op(
            id=op.id,
            engine=op.engine,
            slot=op.slot,
            cycle=-1,
            dep_ids=set(op.dep_ids),
            reads=set(op.reads),
            writes=set(op.writes),
        ) for op in ops]
        op_by_id_copy = {op.id: op for op in ops_copy}

        try:
            num_cycles, scheduled = schedule_greedy(
                ops_copy, op_by_id_copy, enable_valu_shift, priority_mode=mode
            )
            if num_cycles < best_cycles:
                best_cycles = num_cycles
                best_scheduled = scheduled
                best_mode = mode
        except Exception as e:
            print(f"  Warning: {mode} pass failed: {e}")
            continue

    if best_scheduled is None:
        raise RuntimeError("All greedy passes failed")

    print(f"  Best scheduling mode: {best_mode} ({best_cycles} cycles)")
    return best_cycles, best_scheduled


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


def reschedule(instrs: list[dict], enable_valu_shift: bool = False) -> list[dict]:
    """Re-schedule instructions for better packing."""
    # Extract operations
    ops = extract_ops(instrs)

    # Compute dependencies
    op_by_id = compute_dependencies(ops)

    # Schedule
    max_cycle, scheduled_ops = schedule_greedy(ops, op_by_id, enable_valu_shift)

    # Emit
    return emit_scheduled(scheduled_ops, max_cycle)


def reschedule_section(instrs: list[dict], start: int, end: int) -> list[dict]:
    """Re-schedule a section of instructions."""
    section = instrs[start:end]
    rescheduled = reschedule(section)
    return instrs[:start] + rescheduled + instrs[end:]
