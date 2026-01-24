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

from solver.shift_analysis import (
    analyze_shifts, can_eliminate_broadcast, ShiftAnalysis,
    compute_lookahead_eliminations
)

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
# NOTE: 'multiply_add' is intentionally EXCLUDED because:
# - VALU multiply_add is 1 fused op
# - ALU would need 2 separate ops (multiply then add)
# - Shifting FMA to ALU INCREASES total work, which is counterproductive
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


def expand_valu_to_alu_smart(
    op: Op,
    next_op_id: int,
    broadcast_sources: dict[int, int],
    broadcast_op_ids: dict[int, int],
    debug: bool = False
) -> tuple[list[Op], int]:
    """Expand a VALU op into 8 scalar ALU ops with broadcast forwarding.

    When a source operand comes from a vbroadcast AND the broadcast is the
    defining operation for that source (checked via dependencies), use the
    original scalar source directly instead of reading from the broadcast
    destination. This eliminates redundant intermediate copies.

    IMPORTANT: We only substitute if the current op DIRECTLY depends on the
    broadcast operation. This ensures the broadcast is the last write to
    those addresses before this op reads them. Addresses like v_nv that are
    broadcast in some rounds but overwritten in others won't be incorrectly
    substituted.

    Args:
        op: The VALU operation to expand
        next_op_id: The next available operation ID
        broadcast_sources: Map from broadcast dest base addr to scalar source addr
        broadcast_op_ids: Map from broadcast dest base addr to broadcast op ID
        debug: If True, print debug information about substitutions

    Returns:
        Tuple of (list of 8 ALU ops, count of broadcast forwards applied)
    """
    op_type, dest, src1, src2 = op.slot
    new_ops = []

    # Check if src1 or src2 came from a broadcast AND we depend on that broadcast
    # Only substitute if the broadcast is the actual defining operation for this read
    src1_is_broadcast = (src1 in broadcast_sources and
                         broadcast_op_ids.get(src1) in op.dep_ids)
    src2_is_broadcast = (src2 in broadcast_sources and
                         broadcast_op_ids.get(src2) in op.dep_ids)

    # Count how many forwards we're applying
    forwards = (1 if src1_is_broadcast else 0) + (1 if src2_is_broadcast else 0)

    if debug and (src1_is_broadcast or src2_is_broadcast):
        print(f"  [DEBUG] Broadcast forwarding for op {op.id}: {op.slot}")
        if src1_is_broadcast:
            print(f"    src1: {src1} → {broadcast_sources[src1]} (dep on broadcast {broadcast_op_ids[src1]})")
        if src2_is_broadcast:
            print(f"    src2: {src2} → {broadcast_sources[src2]} (dep on broadcast {broadcast_op_ids[src2]})")

    for i in range(VLEN):
        # If source came from broadcast AND we depend on it, use scalar directly
        if src1_is_broadcast:
            actual_src1 = broadcast_sources[src1]
        else:
            actual_src1 = src1 + i

        if src2_is_broadcast:
            actual_src2 = broadcast_sources[src2]
        else:
            actual_src2 = src2 + i

        new_op = Op(
            id=next_op_id + i,
            engine='alu',
            slot=(op_type, dest + i, actual_src1, actual_src2),
            dep_ids=set(op.dep_ids),  # Copy dependencies
            reads={actual_src1, actual_src2},
            writes={dest + i},
        )
        new_ops.append(new_op)

    return new_ops, forwards


def expand_vbroadcast_to_alu(op: Op, next_op_id: int) -> list[Op]:
    """Expand a vbroadcast op into 8 scalar ALU copy ops.

    Uses AND with the same operand twice to effectively copy:
    dest[i] = scalar & scalar = scalar

    Returns a list of 8 new Op objects.
    """
    _, dest, scalar_src = op.slot
    new_ops = []

    for i in range(VLEN):
        new_op = Op(
            id=next_op_id + i,
            engine='alu',
            slot=('&', dest + i, scalar_src, scalar_src),
            dep_ids=set(op.dep_ids),  # Copy dependencies
            reads={scalar_src},
            writes={dest + i},
        )
        new_ops.append(new_op)

    return new_ops


def should_emit_broadcast(
    dest: int,
    op: Op,
    scheduled_ops: list[Op],
    ready_ops: list[Op],
    remaining_ops: list[Op],
    shifted_op_ids: set[int]
) -> bool:
    """Check if a vbroadcast needs to be emitted.

    A broadcast can be skipped if ALL consumers of its destination addresses
    will be shifted to ALU (and thus can read the scalar directly).

    For simplicity, we always emit broadcasts for now since tracking all
    consumers requires more analysis. The broadcast forwarding optimization
    still helps by allowing shifted ops to read scalars directly.
    """
    # For now, always emit broadcasts - the forwarding optimization still helps
    # by allowing shifted binary ops to read the scalar source directly instead
    # of the broadcast destination.
    #
    # TODO: Full dead broadcast elimination requires tracking all consumers
    # and checking if they will all be shifted. This is complex because:
    # 1. Some consumers might not be shiftable (vselect, multiply_add)
    # 2. Consumer shifting depends on VALU slot availability at schedule time
    # 3. Dependencies need updating if broadcast is skipped
    return True


def schedule_greedy(
    ops: list[Op],
    op_by_id: dict[int, Op],
    enable_valu_shift: bool = False,
    priority_mode: str = "critical_path",
    enable_broadcast_lookahead: bool = False,
) -> tuple[int, list[Op]]:
    """Schedule operations using greedy list scheduling with configurable priority.

    This is an O(V+E) algorithm that achieves near-optimal results for structured
    problems with high parallelism (like VLIW kernels with independent batch groups).

    Algorithm:
    1. Compute critical path length for each op (longest path to any sink)
    2. Initialize ready queue with ops that have no dependencies
    3. Greedily schedule highest-priority ops that fit in available slots
    4. Update ready queue as dependencies are satisfied

    Smart Broadcast Forwarding:
    When VALU ops are shifted to ALU, we track vbroadcast sources. If a shifted
    VALU op reads from a broadcast destination, we substitute the original scalar
    source directly instead of reading from the broadcast destination. This
    eliminates 8 redundant ALU copies per shifted op that reads from a broadcast.

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
        enable_broadcast_lookahead: If True, use lookahead analysis to
            proactively eliminate broadcasts whose consumers will shift.
            Experimental - may increase cycle count in some cases.

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

    # Smart Broadcast Forwarding: Track vbroadcast sources
    # Maps broadcast dest base address -> original scalar source address
    # When a shifted VALU op reads from a broadcast dest, we substitute the scalar
    broadcast_sources: dict[int, int] = {}
    # Also track broadcast op IDs to verify dependencies
    # Maps broadcast dest base address -> op ID of the broadcast
    broadcast_op_ids: dict[int, int] = {}
    broadcast_ops_shifted = 0  # Count of vbroadcast ops that were shifted

    # NEW: Run static shift analysis for vbroadcast elimination
    shift_analysis: ShiftAnalysis = analyze_shifts(ops, op_by_id)
    eliminated_broadcasts: set[int] = set()  # Track which broadcasts were eliminated
    shifted_op_ids: set[int] = set()  # Track which ops have been shifted to ALU
    must_shift: set[int] = set()  # Ops that MUST be shifted (broadcast eliminated)

    # NEW: Lookahead analysis - predict which broadcasts can be eliminated proactively
    # based on future VALU pressure (optional, can cause regressions)
    proactive_eliminations: set[int] = set()
    if enable_valu_shift and shift_analysis.eliminable_broadcasts:
        if enable_broadcast_lookahead:
            proactive_eliminations = compute_lookahead_eliminations(ops, op_by_id, shift_analysis)

            # Pre-mark consumers of proactively eliminated broadcasts for shifting
            for bc_id in proactive_eliminations:
                consumers = shift_analysis.broadcast_consumers.get(bc_id, set())
                must_shift.update(consumers)

        print(f"  Shift analysis: {len(shift_analysis.shiftable_ops)} shiftable, "
              f"{len(shift_analysis.broadcast_ops)} broadcasts, "
              f"{len(shift_analysis.eliminable_broadcasts)} eliminable" +
              (f", {len(proactive_eliminations)} proactive" if enable_broadcast_lookahead else ""))

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

    # Pre-scan for vbroadcast ops to build broadcast_sources mapping
    # This allows smart expansion to substitute scalar sources during shifting
    # We also track op IDs so we can verify dependency relationships
    for op in ops:
        if op.engine == 'valu' and len(op.slot) >= 3 and op.slot[0] == 'vbroadcast':
            _, dest, scalar_src = op.slot
            broadcast_sources[dest] = scalar_src
            broadcast_op_ids[dest] = op.id

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
    broadcast_forward_count = 0  # Track how many broadcasts were forwarded
    eliminated_broadcast_count = 0  # Track eliminated broadcasts

    while ready:
        # Sort ready queue by priority
        ready.sort(key=priority)
        op = ready.pop(0)
        cycle = earliest_cycle(op)

        # NEW: Check if this is an eliminable vbroadcast
        # A vbroadcast can be eliminated if:
        # 1. It's in proactive_eliminations (lookahead predicted consumers will shift), OR
        # 2. VALU is saturated (broadcast would be shifted to ALU anyway)
        #    AND all consumers are shiftable
        # This avoids emitting 8 ALU copy ops for the broadcast itself
        is_proactive = op.id in proactive_eliminations
        is_reactive = (not can_schedule(op, cycle) and  # VALU is saturated
                       get_free_alu_slots(cycle) >= VLEN and  # ALU has capacity
                       can_eliminate_broadcast(op.id, shift_analysis, shifted_op_ids))

        if (enable_valu_shift and op.engine == 'valu' and
            len(op.slot) >= 3 and op.slot[0] == 'vbroadcast' and
            (is_proactive or is_reactive)):

            # ELIMINATE: Don't emit anything, just mark as done
            _, dest, scalar = op.slot
            broadcast_sources[dest] = scalar
            broadcast_op_ids[dest] = op.id
            eliminated_broadcasts.add(op.id)
            eliminated_broadcast_count += 1

            # Mark consumers as MUST SHIFT - they need to read scalar directly
            consumers = shift_analysis.broadcast_consumers.get(op.id, set())
            must_shift.update(consumers)

            # Mark as "done" at earliest possible cycle (no actual work)
            op.cycle = cycle

            # Update ready list for dependent ops
            for other in ops:
                if op.id in other.dep_ids:
                    in_degree[other.id] -= 1
                    if in_degree[other.id] == 0:
                        ready.append(other)

            continue  # Skip normal scheduling

        # NEW: Force-shift ops that must be shifted (their broadcast was eliminated)
        if (op.id in must_shift and op.engine == 'valu' and
            can_convert_valu_to_alu(op)):

            # Must shift to ALU - broadcast was eliminated, consumer needs scalar
            alu_ops, forwards = expand_valu_to_alu_smart(
                op, next_op_id, broadcast_sources, broadcast_op_ids, debug=False
            )
            broadcast_forward_count += forwards
            next_op_id += VLEN

            # Schedule the 8 ALU ops across available cycles
            for alu_op in alu_ops:
                alu_cycle = cycle
                while not can_schedule(alu_op, alu_cycle):
                    alu_cycle += 1
                schedule_op(alu_op, alu_cycle)
                op_by_id[alu_op.id] = alu_op
                scheduled.append(alu_op)
                critical_path[alu_op.id] = critical_path.get(op.id, 1)

            op.cycle = max(a.cycle for a in alu_ops)
            expanded_ops[op.id] = [a.id for a in alu_ops]
            valu_shifted_count += 1
            shifted_op_ids.add(op.id)
            must_shift.discard(op.id)  # Done shifting this op

            # Update ready list for dependent ops
            for other in ops:
                if op.id in other.dep_ids:
                    in_degree[other.id] -= 1
                    if in_degree[other.id] == 0:
                        ready.append(other)

            continue  # Skip normal scheduling

        # Check if we can schedule as-is
        while not can_schedule(op, cycle):
            # VALU→ALU shifting: if VALU is full but ALU has capacity
            if (enable_valu_shift and op.engine == 'valu' and
                can_convert_valu_to_alu(op) and
                get_free_alu_slots(cycle) >= VLEN):

                # Smart Broadcast Forwarding: Expand VALU op to 8 ALU ops,
                # substituting scalar sources for any operands from vbroadcast
                alu_ops, forwards = expand_valu_to_alu_smart(
                    op, next_op_id, broadcast_sources, broadcast_op_ids, debug=False
                )
                broadcast_forward_count += forwards
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
                shifted_op_ids.add(op.id)  # Track for broadcast elimination
                break
            else:
                cycle += 1
        else:
            # Normal scheduling (didn't shift/expand)
            schedule_op(op, cycle)
            scheduled.append(op)

        # Update ready list for ops that depend on current op
        for other in ops:
            if op.id in other.dep_ids:
                in_degree[other.id] -= 1
                if in_degree[other.id] == 0:
                    ready.append(other)

    if valu_shifted_count > 0 or eliminated_broadcast_count > 0:
        print(f"  VALU→ALU shifted: {valu_shifted_count} ops")
        if eliminated_broadcast_count > 0:
            ops_saved = eliminated_broadcast_count * VLEN
            print(f"  Broadcasts eliminated: {eliminated_broadcast_count} (saves {ops_saved} ALU ops)")
        if broadcast_sources:
            print(f"  Broadcast sources tracked: {len(broadcast_sources)} vbroadcasts")
        if broadcast_forward_count > 0:
            print(f"  Broadcast forwards applied: {broadcast_forward_count} (saves {broadcast_forward_count * 7} redundant reads)")

    num_cycles = max(op.cycle for op in scheduled) + 1 if scheduled else 0
    return num_cycles, scheduled


def schedule_greedy_multipass(
    ops: list[Op],
    op_by_id: dict[int, Op],
    enable_valu_shift: bool = False,
    enable_broadcast_lookahead: bool = False,
) -> tuple[int, list[Op]]:
    """Run multiple greedy passes with different priority orderings and return the best.

    This can find better schedules by exploring different orderings of the ready queue.
    Each pass uses a different priority_mode to potentially find different local optima.

    Args:
        ops: List of operations to schedule
        op_by_id: Dictionary mapping op IDs to Op objects
        enable_valu_shift: If True, convert VALU ops to ALU ops when VALU is saturated
        enable_broadcast_lookahead: If True, use lookahead to proactively eliminate broadcasts

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
                ops_copy, op_by_id_copy, enable_valu_shift, priority_mode=mode,
                enable_broadcast_lookahead=enable_broadcast_lookahead
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
