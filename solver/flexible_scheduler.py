"""
Flexible CP-SAT scheduler that allows VALU↔ALU substitution.

Key insight: Some VALU operations can alternatively be done on ALU:
- VALU: 1 slot, processes 8 elements (SIMD)
- ALU: 8 slots, processes 8 elements (scalar expanded)

With 6 VALU + 12 ALU slots per cycle:
- Pure VALU: 6 ops/cycle = 48 elements/cycle
- If VALU saturated, shift to ALU: 1.5 extra ops possible

This scheduler adds boolean choice variables for VALU ops.
"""

from ortools.sat.python import cp_model
from dataclasses import dataclass
from solver.reschedule import Op, VLEN


@dataclass
class FlexScheduleResult:
    success: bool
    cycles: int
    schedule: dict[int, int]  # op_id -> cycle
    engine_choice: dict[int, str]  # op_id -> chosen engine
    status: str
    valu_count: int
    alu_count: int


# VALU ops that can be done on ALU (basic arithmetic)
SHIFTABLE_OPS = {'+', '-', '*', '//', '^', '&', '|', '<', '<=', '==', '!=', '>=', '>', '<<', '>>'}

# Special VALU ops with non-standard expansion patterns
# vbroadcast: scalar src → 8 copies to vector dest (uses 8 ALU slots)
# Note: multiply_add is NOT shiftable because it needs 16 ALU slots (8 muls + 8 adds)
#       which exceeds the 12 ALU slots/cycle limit
SPECIAL_SHIFTABLE_OPS = {'vbroadcast'}


def get_alu_slot_demand(op_type: str) -> int:
    """Get the number of ALU slots required when shifting a VALU op to ALU."""
    if op_type in SHIFTABLE_OPS:
        return 8  # Standard vector op expands to 8 scalar ops
    elif op_type == 'vbroadcast':
        return 8  # 8 copies from same scalar source
    elif op_type == 'multiply_add':
        return 16  # 8 multiplies + 8 adds (uses internal temps)
    return 0  # Not shiftable


def expand_to_alu(slot: tuple) -> list[tuple]:
    """Expand a VALU slot to equivalent ALU ops.

    Returns a list of ALU slot tuples that together implement the VALU operation.
    """
    op_type = slot[0]

    if op_type in SHIFTABLE_OPS:
        # Standard binary vector op: (op, dest, src1, src2)
        _, dest, src1, src2 = slot
        return [(op_type, dest + i, src1 + i, src2 + i) for i in range(VLEN)]

    elif op_type == 'vbroadcast':
        # vbroadcast(dest, src): copy scalar src to all 8 elements of vector dest
        # Use bitwise AND with itself as a copy: dest[i] = src & src
        _, dest, src = slot
        return [('&', dest + i, src, src) for i in range(VLEN)]

    elif op_type == 'multiply_add':
        # multiply_add(dest, a, b, c): dest = a * b + c for all 8 elements
        # Expand to 8 muls + 8 adds, using dest as temp storage for intermediate mul results
        # Step 1: dest[i] = a[i] * b[i]  (8 multiplies)
        # Step 2: dest[i] = dest[i] + c[i]  (8 adds)
        # Note: This produces all ops in one list - the scheduler will place them
        _, dest, a, b, c = slot
        muls = [('*', dest + i, a + i, b + i) for i in range(VLEN)]
        adds = [('+', dest + i, dest + i, c + i) for i in range(VLEN)]
        return muls + adds

    return []  # Not shiftable


def schedule_flexible(ops: list[Op], op_by_id: dict[int, Op], time_limit: float = 60.0,
                      greedy_hint: dict[int, int] = None) -> FlexScheduleResult:
    """Schedule with VALU↔ALU flexibility."""

    SLOT_LIMITS = {
        'load': 2,
        'store': 2,
        'alu': 12,
        'valu': 6,
        'flow': 1,
    }

    if not ops:
        return FlexScheduleResult(True, 0, {}, {}, "empty", 0, 0)

    model = cp_model.CpModel()

    # Identify ops that can be shifted from VALU to ALU
    # Include both standard shiftable ops and special ones (vbroadcast, multiply_add)
    all_shiftable = SHIFTABLE_OPS | SPECIAL_SHIFTABLE_OPS
    shiftable_ops = []
    for op in ops:
        if op.engine == 'valu' and op.slot and op.slot[0] in all_shiftable:
            shiftable_ops.append(op)

    print(f"  Shiftable VALU ops: {len(shiftable_ops)} / {sum(1 for o in ops if o.engine == 'valu')}")

    # Compute horizon
    if greedy_hint:
        horizon = max(greedy_hint.values()) + 1
    else:
        from solver.reschedule import schedule_greedy as quick_greedy
        local_ids = {op.id for op in ops}
        temp_ops = [Op(id=op.id, engine=op.engine, slot=op.slot,
                       dep_ids={d for d in op.dep_ids if d in local_ids},
                       reads=op.reads.copy(), writes=op.writes.copy()) for op in ops]
        temp_by_id = {op.id: op for op in temp_ops}
        horizon = quick_greedy(temp_ops, temp_by_id)
        greedy_hint = {op.id: op.cycle for op in temp_ops}

    # Variables
    starts = {}
    ends = {}
    intervals = {}
    use_valu = {}  # Boolean: true = use VALU, false = use ALU (only for shiftable)

    shiftable_ids = {op.id for op in shiftable_ops}

    for op in ops:
        start = model.NewIntVar(0, horizon - 1, f'start_{op.id}')
        end = model.NewIntVar(1, horizon, f'end_{op.id}')
        interval = model.NewIntervalVar(start, 1, end, f'interval_{op.id}')
        starts[op.id] = start
        ends[op.id] = end
        intervals[op.id] = interval

        if op.id in shiftable_ids:
            use_valu[op.id] = model.NewBoolVar(f'use_valu_{op.id}')

    # Dependency constraints
    for op in ops:
        for dep_id in op.dep_ids:
            if dep_id in ends:
                model.Add(starts[op.id] >= ends[dep_id])

    # Resource constraints with flexibility
    # For VALU: demand = 1 if use_valu else 0
    # For ALU: demand = 8 if not use_valu (for shiftable), else original demand

    # Group ops by engine (original assignment)
    valu_ops = [op for op in ops if op.engine == 'valu']
    alu_ops = [op for op in ops if op.engine == 'alu']
    load_ops = [op for op in ops if op.engine == 'load']
    store_ops = [op for op in ops if op.engine == 'store']
    flow_ops = [op for op in ops if op.engine == 'flow']

    # VALU cumulative: shiftable ops contribute 1 if use_valu, 0 otherwise
    # Non-shiftable VALU ops always contribute 1
    valu_intervals = []
    valu_demands = []
    for op in valu_ops:
        valu_intervals.append(intervals[op.id])
        if op.id in shiftable_ids:
            # Demand is 1 when use_valu[op.id] is true, 0 otherwise
            demand = model.NewIntVar(0, 1, f'valu_demand_{op.id}')
            model.Add(demand == 1).OnlyEnforceIf(use_valu[op.id])
            model.Add(demand == 0).OnlyEnforceIf(use_valu[op.id].Not())
            valu_demands.append(demand)
        else:
            # Non-shiftable (like vbroadcast, multiply_add): always 1
            valu_demands.append(1)

    if valu_intervals:
        model.AddCumulative(valu_intervals, valu_demands, SLOT_LIMITS['valu'])

    # ALU cumulative: original ALU ops + shiftable VALU ops (8 slots each when shifted)
    alu_intervals = [intervals[op.id] for op in alu_ops]
    alu_demands = [1] * len(alu_ops)

    for op in shiftable_ops:
        alu_intervals.append(intervals[op.id])
        # Get ALU slot demand for this op type (8 for standard, 8 for vbroadcast, 16 for multiply_add)
        alu_slots = get_alu_slot_demand(op.slot[0])
        # Demand is alu_slots when NOT using VALU (shifted to ALU), 0 otherwise
        demand = model.NewIntVar(0, alu_slots, f'alu_demand_{op.id}')
        model.Add(demand == 0).OnlyEnforceIf(use_valu[op.id])
        model.Add(demand == alu_slots).OnlyEnforceIf(use_valu[op.id].Not())
        alu_demands.append(demand)

    if alu_intervals:
        model.AddCumulative(alu_intervals, alu_demands, SLOT_LIMITS['alu'])

    # Other engines: simple cumulative
    if load_ops:
        model.AddCumulative([intervals[op.id] for op in load_ops],
                           [1] * len(load_ops), SLOT_LIMITS['load'])
    if store_ops:
        model.AddCumulative([intervals[op.id] for op in store_ops],
                           [1] * len(store_ops), SLOT_LIMITS['store'])
    if flow_ops:
        model.AddCumulative([intervals[op.id] for op in flow_ops],
                           [1] * len(flow_ops), SLOT_LIMITS['flow'])

    # Objective: minimize makespan
    makespan = model.NewIntVar(0, horizon, 'makespan')
    model.AddMaxEquality(makespan, [ends[op.id] for op in ops])
    model.Minimize(makespan)

    # Hints
    if greedy_hint:
        for op in ops:
            if op.id in greedy_hint:
                model.AddHint(starts[op.id], greedy_hint[op.id])
        # Default to using VALU (original behavior)
        for op_id in use_valu:
            model.AddHint(use_valu[op_id], 1)

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    import os
    solver.parameters.num_search_workers = os.cpu_count() or 8
    solver.parameters.interleave_search = True
    solver.parameters.log_search_progress = False  # Quiet for now
    solver.parameters.linearization_level = 2
    solver.parameters.randomize_search = True
    solver.parameters.cp_model_presolve = True
    solver.parameters.cp_model_probing_level = 2

    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        schedule = {}
        engine_choice = {}
        valu_count = 0
        alu_count = 0

        for op in ops:
            cycle = solver.Value(starts[op.id])
            schedule[op.id] = cycle
            op.cycle = cycle

            if op.id in use_valu:
                if solver.Value(use_valu[op.id]):
                    engine_choice[op.id] = 'valu'
                    valu_count += 1
                else:
                    engine_choice[op.id] = 'alu'
                    alu_count += 1
                    # Update op's engine for emission
                    op.engine = 'alu'
                    # Expand to ALU ops using the appropriate expansion pattern
                    op.slot = expand_to_alu(op.slot)
            else:
                engine_choice[op.id] = op.engine
                if op.engine == 'valu':
                    valu_count += 1

        result_cycles = solver.Value(makespan)
        bound = solver.BestObjectiveBound()
        gap = (result_cycles - bound) / result_cycles * 100 if result_cycles > 0 else 0

        print(f"    Flexible: {result_cycles} cycles, bound={bound:.0f}, gap={gap:.1f}%")
        print(f"    VALU ops: {valu_count}, shifted to ALU: {alu_count}")

        return FlexScheduleResult(
            success=True,
            cycles=result_cycles,
            schedule=schedule,
            engine_choice=engine_choice,
            status="optimal" if status == cp_model.OPTIMAL else "feasible",
            valu_count=valu_count,
            alu_count=alu_count
        )
    else:
        return FlexScheduleResult(
            success=False,
            cycles=-1,
            schedule={},
            engine_choice={},
            status=solver.StatusName(status),
            valu_count=0,
            alu_count=0
        )


def emit_flexible_scheduled(ops: list[Op], max_cycle: int) -> list[dict]:
    """Emit instructions from flexibly scheduled operations."""
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
