"""
Iterative rescheduling with VALU→ALU optimization.

Key insight: Move VALU ops to ALU when ALU has EARLIER free slots, not just for balance.
This directly reduces makespan by allowing earlier execution.

Algorithm:
1. Start with initial greedy schedule
2. For each VALU op (in schedule order):
   a. Find earliest cycle where it could run on ALU (8 slots available, deps satisfied)
   b. If earlier than current VALU schedule, move to ALU
   c. Cascade: use CP-SAT to reschedule only the LATER ops (at or after the old cycle)
3. Repeat until no improvement
"""

from solver.reschedule import Op, VLEN, schedule_greedy, emit_scheduled
from solver.cp_scheduler import schedule_cp
from solver.flexible_scheduler import (
    SHIFTABLE_OPS, SPECIAL_SHIFTABLE_OPS,
    get_alu_slot_demand, expand_to_alu
)
from dataclasses import dataclass


@dataclass
class IterativeResult:
    success: bool
    cycles: int
    schedule: dict[int, int]
    valu_shifted: int
    iterations: int


def iterative_reschedule(ops: list[Op], op_by_id: dict[int, Op],
                         max_iterations: int = 100,
                         cpsat_time_per_iter: float = 60.0) -> IterativeResult:
    """Iteratively move VALU ops to ALU when it enables earlier execution.

    After moving ops, uses CP-SAT to reschedule only the affected (later) ops,
    keeping earlier ops fixed. This gives CP-SAT quality without full-problem cost.
    """

    SLOT_LIMITS = {
        'load': 2,
        'store': 2,
        'alu': 12,
        'valu': 6,
        'flow': 1,
    }

    if not ops:
        return IterativeResult(True, 0, {}, 0, 0)

    # Identify shiftable VALU ops
    all_shiftable = SHIFTABLE_OPS | SPECIAL_SHIFTABLE_OPS
    shiftable_ids = {
        op.id for op in ops
        if op.engine == 'valu' and op.slot and op.slot[0] in all_shiftable
    }

    # Initial greedy schedule
    for op in ops:
        op.cycle = -1
    schedule_greedy(ops, op_by_id)

    initial_cycles = max(op.cycle for op in ops) + 1
    print(f"  Initial greedy: {initial_cycles} cycles")

    # Track slot usage per cycle
    def compute_usage():
        usage = {}
        for op in ops:
            c = op.cycle
            if c not in usage:
                usage[c] = {e: 0 for e in SLOT_LIMITS}
            if isinstance(op.slot, list):
                usage[c][op.engine] = usage[c].get(op.engine, 0) + len(op.slot)
            else:
                usage[c][op.engine] = usage[c].get(op.engine, 0) + 1
        return usage

    def earliest_dep_cycle(op):
        """Earliest cycle this op can start based on dependencies."""
        if not op.dep_ids:
            return 0
        return max(op_by_id[d].cycle + 1 for d in op.dep_ids if d in op_by_id)

    valu_shifted = 0
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        ops_moved_this_iter = 0
        min_old_cycle = float('inf')  # Track earliest cycle we moved FROM
        usage = compute_usage()

        # Sort by cycle (earliest first) so we process in dependency order
        sorted_ops = sorted(ops, key=lambda o: (o.cycle, o.id))

        for op in sorted_ops:
            if op.id not in shiftable_ids or op.engine != 'valu':
                continue

            current_cycle = op.cycle
            min_cycle = earliest_dep_cycle(op)

            if min_cycle >= current_cycle:
                continue  # Can't move earlier

            alu_demand = get_alu_slot_demand(op.slot[0])

            # Find earliest cycle with ALU capacity
            best_cycle = None
            for c in range(min_cycle, current_cycle):
                alu_used = usage.get(c, {}).get('alu', 0)
                if alu_used + alu_demand <= SLOT_LIMITS['alu']:
                    best_cycle = c
                    break

            if best_cycle is not None:
                # Move to ALU
                old_slot = op.slot
                old_cycle = op.cycle

                op.engine = 'alu'
                op.cycle = best_cycle
                op.slot = expand_to_alu(old_slot)

                # Track the earliest cycle we moved FROM
                min_old_cycle = min(min_old_cycle, old_cycle)

                # Update usage: remove from old position, add to new
                if old_cycle in usage:
                    usage[old_cycle]['valu'] = max(0, usage[old_cycle].get('valu', 0) - 1)
                if best_cycle not in usage:
                    usage[best_cycle] = {e: 0 for e in SLOT_LIMITS}
                usage[best_cycle]['alu'] = usage[best_cycle].get('alu', 0) + len(op.slot)

                valu_shifted += 1
                ops_moved_this_iter += 1
                shiftable_ids.discard(op.id)

        if ops_moved_this_iter == 0:
            break

        # After moving ops, reschedule only ops at or after min_old_cycle using CP-SAT
        # Earlier ops are kept fixed - they're already well-scheduled
        later_ops = [op for op in ops if op.cycle >= min_old_cycle]
        earlier_ops = [op for op in ops if op.cycle < min_old_cycle]

        if later_ops:
            print(f"  Iter {iteration}: moved {ops_moved_this_iter} ops, "
                  f"rescheduling {len(later_ops)} later ops with CP-SAT...")

            # For CP-SAT, we need to handle dependencies on earlier (fixed) ops
            # Create modified dep_ids that only reference ops within later_ops
            # But track the minimum start time based on earlier op dependencies
            later_op_ids = {op.id for op in later_ops}
            earlier_op_ids = {op.id for op in ops if op.cycle < min_old_cycle}

            # Compute min start for each later op based on fixed earlier ops
            min_start_from_earlier = {}
            for op in later_ops:
                min_start = 0
                for dep_id in op.dep_ids:
                    if dep_id in earlier_op_ids:
                        # This dependency is on a fixed earlier op
                        earlier_op = op_by_id[dep_id]
                        min_start = max(min_start, earlier_op.cycle + 1)
                min_start_from_earlier[op.id] = min_start

            # Create copies of ops with filtered dep_ids for CP-SAT
            # (only deps within later_ops subset)
            cp_ops = []
            for op in later_ops:
                cp_op = Op(
                    id=op.id,
                    engine=op.engine,
                    slot=op.slot,
                    dep_ids={d for d in op.dep_ids if d in later_op_ids},
                    reads=op.reads.copy() if op.reads else set(),
                    writes=op.writes.copy() if op.writes else set(),
                )
                cp_ops.append(cp_op)

            cp_op_by_id = {op.id: op for op in cp_ops}

            # Create hint from current schedule (relative to min_old_cycle)
            later_hint = {op.id: max(0, op.cycle - min_old_cycle) for op in later_ops}

            # Compute ALU demands for ops with multi-slot usage (shifted VALU ops)
            alu_demands = {}
            for op in cp_ops:
                if op.engine == 'alu' and isinstance(op.slot, list):
                    # Shifted VALU op - demand is the number of ALU slots (8)
                    alu_demands[op.id] = len(op.slot)

            # Run CP-SAT with adequate time, logging to file
            result = schedule_cp(cp_ops, cp_op_by_id,
                                time_limit=cpsat_time_per_iter,
                                greedy_hint=later_hint,
                                alu_demands=alu_demands,
                                log_file='cpsat_reschedule.log')

            if result.success:
                # Apply schedule back to original ops
                # Need to offset by min_old_cycle AND respect min_start_from_earlier
                for op in later_ops:
                    scheduled_cycle = result.schedule[op.id]
                    min_from_earlier = min_start_from_earlier[op.id]
                    # Offset and ensure it respects earlier deps
                    op.cycle = max(scheduled_cycle + min_old_cycle, min_from_earlier)

                new_cycles = max(op.cycle for op in ops) + 1
                print(f"    CP-SAT: {new_cycles} cycles (status={result.status})")
            else:
                print(f"    CP-SAT failed: {result.status} - keeping existing schedule")
                new_cycles = max(op.cycle for op in ops) + 1
        else:
            new_cycles = max(op.cycle for op in ops) + 1

    final_cycles = max(op.cycle for op in ops) + 1

    # Validate final schedule doesn't exceed slot limits
    final_usage = compute_usage()
    violations = []
    for cycle, cycle_usage in final_usage.items():
        for engine, used in cycle_usage.items():
            limit = SLOT_LIMITS.get(engine, 1)
            if used > limit:
                violations.append((cycle, engine, used, limit))

    if violations:
        print(f"  WARNING: {len(violations)} slot limit violations detected!")
        for cycle, engine, used, limit in violations[:5]:
            print(f"    Cycle {cycle}: {engine} uses {used}/{limit} slots")

    print(f"  After {iteration} iterations: {final_cycles} cycles, shifted {valu_shifted} ops")

    schedule = {op.id: op.cycle for op in ops}
    return IterativeResult(
        success=True,
        cycles=final_cycles,
        schedule=schedule,
        valu_shifted=valu_shifted,
        iterations=iteration
    )


def emit_iterative_scheduled(ops: list[Op], max_cycle: int) -> list[dict]:
    """Emit instructions from iteratively scheduled operations."""
    instrs = []

    for cycle in range(max_cycle):
        instr = {}
        for op in ops:
            if op.cycle == cycle:
                if op.engine not in instr:
                    instr[op.engine] = []
                # Handle expanded ALU ops (list of slots)
                if isinstance(op.slot, list):
                    instr[op.engine].extend(op.slot)
                else:
                    instr[op.engine].append(op.slot)
        if instr:
            instrs.append(instr)

    return instrs
