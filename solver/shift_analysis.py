"""Static analysis for VALU→ALU shifting decisions.

This module provides pure functions for analyzing which VALU operations can be
shifted to ALU and which vbroadcast operations can be eliminated entirely.

The key insight is that vbroadcast operations exist ONLY to replicate a scalar
value across VLEN vector lanes. If ALL consumers of that broadcast are shifted
to ALU (where they read scalars directly via broadcast forwarding), the
broadcast itself becomes dead code and can be eliminated.

Flow:
1. analyze_shifts() - Static pre-pass before scheduling
2. can_eliminate_broadcast() - Dynamic check during scheduling
"""

from dataclasses import dataclass, field
from typing import Set, Dict, List, Optional

# Vector length for SIMD operations
VLEN = 8

# VALU ops that can be converted to scalar ALU ops (binary operations)
# NOTE: 'multiply_add' and 'vmultiply_add' are EXCLUDED because:
# - VALU multiply_add is 1 fused op
# - ALU would need 2 separate ops (multiply then add)
# - Shifting FMA to ALU INCREASES total work
VALU_TO_ALU_OPS = {'+', '-', '*', '/', '%', '&', '|', '^', '<<', '>>',
                   '==', '!=', '<', '<=', '>', '>='}


@dataclass
class ShiftAnalysis:
    """Results of static shift analysis.

    Attributes:
        shiftable_ops: Op IDs that CAN be converted from VALU to ALU
        broadcast_ops: Maps broadcast op_id → scalar source address
        broadcast_consumers: Maps broadcast op_id → set of consumer op IDs
        eliminable_broadcasts: Broadcasts where ALL consumers are shiftable
        broadcast_dest_to_id: Maps broadcast dest base addr → broadcast op_id
    """
    shiftable_ops: Set[int] = field(default_factory=set)
    broadcast_ops: Dict[int, int] = field(default_factory=dict)
    broadcast_consumers: Dict[int, Set[int]] = field(default_factory=dict)
    eliminable_broadcasts: Set[int] = field(default_factory=set)
    broadcast_dest_to_id: Dict[int, int] = field(default_factory=dict)


def is_shiftable_op(op) -> bool:
    """Check if a VALU op can be converted to 8 scalar ALU ops.

    Only binary vector operations with operators in VALU_TO_ALU_OPS can be shifted.
    vbroadcast itself is NOT shiftable (but can be eliminated).
    """
    if op.engine != 'valu':
        return False
    if not op.slot or len(op.slot) < 1:
        return False
    return op.slot[0] in VALU_TO_ALU_OPS


def is_vbroadcast(op) -> bool:
    """Check if op is a vbroadcast operation."""
    return (op.engine == 'valu' and
            op.slot and
            len(op.slot) >= 3 and
            op.slot[0] == 'vbroadcast')


def get_broadcast_dest_range(op) -> Set[int]:
    """Get the destination addresses written by a vbroadcast.

    vbroadcast writes VLEN consecutive addresses starting at slot[1].
    """
    if not is_vbroadcast(op):
        return set()
    _, dest, _ = op.slot
    return {dest + i for i in range(VLEN)}


def get_broadcast_scalar_source(op) -> Optional[int]:
    """Get the scalar source address of a vbroadcast."""
    if not is_vbroadcast(op):
        return None
    return op.slot[2]


def analyze_shifts(ops: list, op_by_id: dict) -> ShiftAnalysis:
    """Analyze which VALU ops can be shifted and which broadcasts eliminated.

    This is a STATIC analysis run BEFORE scheduling. It determines:
    1. Which ops are shiftable (binary VALU ops in VALU_TO_ALU_OPS)
    2. Which ops are vbroadcasts and their scalar sources
    3. Which ops consume each vbroadcast (via address-based reads AND dep_ids)
    4. Which vbroadcasts have ALL shiftable consumers (eliminable candidates)

    Args:
        ops: List of Op objects to analyze
        op_by_id: Dictionary mapping op ID to Op object

    Returns:
        ShiftAnalysis with all computed information
    """
    analysis = ShiftAnalysis()

    # Pass 1: Identify shiftable ops and broadcasts
    for op in ops:
        if is_shiftable_op(op):
            analysis.shiftable_ops.add(op.id)
        if is_vbroadcast(op):
            _, dest, scalar = op.slot
            analysis.broadcast_ops[op.id] = scalar
            analysis.broadcast_consumers[op.id] = set()
            analysis.broadcast_dest_to_id[dest] = op.id

    # Pass 2: Find consumers of each broadcast
    # A consumer must:
    # 1. Read from addresses written by the broadcast (via op.reads)
    # 2. Have a dependency on the broadcast (via op.dep_ids) - ensures it's the
    #    defining write, not a stale address that was overwritten
    for op in ops:
        for bc_id in analysis.broadcast_ops:
            # Check if this op depends on the broadcast
            if bc_id not in op.dep_ids:
                continue
            # Check if this op reads any broadcast destination addresses
            bc_op = op_by_id[bc_id]
            bc_dest_range = get_broadcast_dest_range(bc_op)
            if op.reads & bc_dest_range:  # Intersection
                analysis.broadcast_consumers[bc_id].add(op.id)

    # Pass 3: Determine eliminable broadcasts
    # A broadcast is eliminable if:
    # 1. It has at least one consumer (no consumers = already dead code)
    # 2. ALL consumers are shiftable VALU ops
    for bc_id, consumers in analysis.broadcast_consumers.items():
        if not consumers:
            # No consumers means the broadcast result is never used
            # This shouldn't happen in well-formed code, but handle it
            continue

        all_consumers_shiftable = all(
            c in analysis.shiftable_ops for c in consumers
        )

        if all_consumers_shiftable:
            analysis.eliminable_broadcasts.add(bc_id)

    return analysis


def can_eliminate_broadcast(
    bc_id: int,
    analysis: ShiftAnalysis,
    shifted_ops: Set[int],
    pending_shift_ops: Optional[Set[int]] = None
) -> bool:
    """Check if a broadcast can be eliminated given current shifting state.

    A broadcast can be eliminated if ALL its consumers:
    - Have already been shifted to ALU, OR
    - Will definitely be shifted (are in pending_shift_ops or shiftable_ops)

    This is called during scheduling when making the actual decision.

    Args:
        bc_id: The op ID of the broadcast to check
        analysis: The ShiftAnalysis from static analysis
        shifted_ops: Set of op IDs that have already been shifted to ALU
        pending_shift_ops: Optional set of ops that will definitely be shifted

    Returns:
        True if the broadcast can be safely eliminated
    """
    # Must be a known eliminable candidate from static analysis
    if bc_id not in analysis.eliminable_broadcasts:
        return False

    consumers = analysis.broadcast_consumers.get(bc_id, set())
    if not consumers:
        return False  # No consumers - don't eliminate (should be handled elsewhere)

    # All consumers must be shifted or will be shifted
    pending = pending_shift_ops or set()
    for consumer_id in consumers:
        is_shifted = consumer_id in shifted_ops
        will_shift = consumer_id in analysis.shiftable_ops or consumer_id in pending
        if not (is_shifted or will_shift):
            return False

    return True


def predict_shifted_ops(
    ops: list,
    op_by_id: dict,
    analysis: ShiftAnalysis,
    valu_slots_per_cycle: int = 6,
    alu_slots_per_cycle: int = 12
) -> Set[int]:
    """Predict which VALU ops will be shifted to ALU due to VALU saturation.

    This is a lookahead analysis that estimates future VALU pressure and
    predicts which ops will be forced to shift. Used to proactively
    eliminate broadcasts whose consumers will all be shifted anyway.

    The prediction is conservative: it identifies ops that will DEFINITELY
    shift due to VALU capacity constraints.

    Args:
        ops: List of all ops
        op_by_id: Op ID to Op mapping
        analysis: ShiftAnalysis with shiftable ops identified
        valu_slots_per_cycle: VALU capacity (default 6)
        alu_slots_per_cycle: ALU capacity (default 12)

    Returns:
        Set of op IDs predicted to be shifted to ALU
    """
    # Count total VALU ops (excluding broadcasts which might be eliminated)
    valu_ops = [op for op in ops if op.engine == 'valu']
    shiftable_valu = [op for op in valu_ops if op.id in analysis.shiftable_ops]
    non_shiftable_valu = [op for op in valu_ops if op.id not in analysis.shiftable_ops
                         and op.slot[0] != 'vbroadcast']
    broadcast_ops = [op for op in valu_ops if op.slot[0] == 'vbroadcast']

    # Estimate minimum cycles needed based on critical path / dependencies
    # For now, use a simple estimate: total ops / slots per cycle
    # This is conservative (actual cycles may be higher due to dependencies)
    min_cycles_for_non_shiftable = (len(non_shiftable_valu) + valu_slots_per_cycle - 1) // valu_slots_per_cycle

    # Available VALU slots for shiftable ops after non-shiftable are scheduled
    # This is a rough estimate - actual availability depends on scheduling order
    total_valu_capacity = min_cycles_for_non_shiftable * valu_slots_per_cycle
    valu_slots_for_shiftable = max(0, total_valu_capacity - len(non_shiftable_valu))

    # If we have more shiftable ops than available slots, some will shift
    excess_shiftable = len(shiftable_valu) - valu_slots_for_shiftable

    if excess_shiftable <= 0:
        return set()  # All shiftable ops can fit in VALU

    # We need to predict WHICH specific ops will shift.
    # The scheduler shifts ops when VALU is full at their earliest start time.
    # Without running the full schedule, we can't know exactly which ops shift.
    #
    # Conservative approach: only mark ops as "will shift" if we're CERTAIN.
    # This means we need a high confidence that the op will hit VALU saturation.
    #
    # Heuristic: ops with lower critical path priority are more likely to shift
    # because they're scheduled later when VALU is more likely to be full.
    #
    # For now, we use a simple approach: if excess is > 50% of shiftable ops,
    # we predict the bottom 50% by original emission order will shift.
    # This is still conservative - we may miss some elimination opportunities.

    shift_ratio = excess_shiftable / len(shiftable_valu)

    if shift_ratio < 0.3:
        # Less than 30% will shift - too uncertain to predict which ones
        return set()

    # Sort shiftable ops by original ID (emission order)
    # Ops emitted later are more likely to be scheduled later and hit saturation
    sorted_shiftable = sorted(shiftable_valu, key=lambda op: op.id)

    # Predict that the last N ops will shift (where N = excess)
    # Add some margin for uncertainty - only predict if we're very confident
    confident_excess = int(excess_shiftable * 0.8)  # 80% confidence margin

    if confident_excess <= 0:
        return set()

    # The ops most likely to shift are those emitted last
    predicted_shifts = {op.id for op in sorted_shiftable[-confident_excess:]}

    return predicted_shifts


def compute_lookahead_eliminations(
    ops: list,
    op_by_id: dict,
    analysis: ShiftAnalysis
) -> Set[int]:
    """Compute which broadcasts can be eliminated with lookahead.

    Uses predict_shifted_ops to identify broadcasts whose consumers
    will all be shifted due to future VALU pressure.

    Args:
        ops: List of all ops
        op_by_id: Op ID to Op mapping
        analysis: ShiftAnalysis with eliminable broadcasts identified

    Returns:
        Set of broadcast op IDs that can be eliminated proactively
    """
    predicted_shifts = predict_shifted_ops(ops, op_by_id, analysis)

    if not predicted_shifts:
        return set()

    proactive_eliminations = set()

    for bc_id in analysis.eliminable_broadcasts:
        consumers = analysis.broadcast_consumers.get(bc_id, set())
        if not consumers:
            continue

        # Check if ALL consumers are predicted to shift
        all_consumers_will_shift = all(c in predicted_shifts for c in consumers)

        if all_consumers_will_shift:
            proactive_eliminations.add(bc_id)

    return proactive_eliminations


def get_broadcast_source_for_dest(dest: int, analysis: ShiftAnalysis, op_by_id: dict) -> Optional[int]:
    """Get the scalar source for a broadcast destination address.

    Given a destination address that might be part of a broadcast output,
    returns the scalar source address if this is from an eliminable broadcast.

    Args:
        dest: The destination base address (not offset)
        analysis: The ShiftAnalysis from static analysis
        op_by_id: Dictionary mapping op ID to Op object

    Returns:
        The scalar source address if eliminable, None otherwise
    """
    bc_id = analysis.broadcast_dest_to_id.get(dest)
    if bc_id is None:
        return None
    return analysis.broadcast_ops.get(bc_id)
