"""
CP-SAT based scheduler using Google OR-Tools.

Much faster than ILP for scheduling problems because:
1. Uses interval variables (not binary time-indexed)
2. Cumulative constraints for resource limits
3. Sophisticated constraint propagation
"""

from ortools.sat.python import cp_model
from dataclasses import dataclass
from solver.reschedule import Op, VLEN


@dataclass
class CPScheduleResult:
    success: bool
    cycles: int
    schedule: dict[int, int]  # op_id -> cycle
    status: str


def schedule_cp(ops: list[Op], op_by_id: dict[int, Op], time_limit: float = 60.0,
                greedy_hint: dict[int, int] = None,
                alu_demands: dict[int, int] = None,
                log_file: str = None) -> CPScheduleResult:
    """Schedule operations using CP-SAT solver.

    Args:
        ops: List of operations to schedule
        op_by_id: Dictionary mapping op IDs to ops
        time_limit: Max solve time in seconds
        greedy_hint: Initial schedule hint {op_id: cycle}
        alu_demands: Optional dict {op_id: demand} for ops with non-unit ALU demand
                    (e.g., shifted VALU ops use 8 ALU slots)
        log_file: Optional file path to redirect CP-SAT verbose logs
    """

    SLOT_LIMITS = {
        'load': 2,
        'store': 2,
        'alu': 12,
        'valu': 6,
        'flow': 1,
    }

    if not ops:
        return CPScheduleResult(True, 0, {}, "empty")

    model = cp_model.CpModel()

    # Compute tight horizon using greedy as upper bound
    if greedy_hint:
        horizon = max(greedy_hint.values()) + 1
    else:
        # Quick greedy pass to get horizon
        # Filter out external dependencies for this local solve
        from solver.reschedule import schedule_greedy as quick_greedy
        local_ids = {op.id for op in ops}
        temp_ops = [Op(id=op.id, engine=op.engine, slot=op.slot,
                       dep_ids={d for d in op.dep_ids if d in local_ids},  # Filter external deps
                       reads=op.reads.copy(), writes=op.writes.copy()) for op in ops]
        temp_by_id = {op.id: op for op in temp_ops}
        horizon = quick_greedy(temp_ops, temp_by_id)
        greedy_hint = {op.id: op.cycle for op in temp_ops}

    # Create interval variables for each operation
    # Each op takes 1 cycle, so start = end - 1
    starts = {}
    ends = {}
    intervals = {}

    for op in ops:
        start = model.NewIntVar(0, horizon - 1, f'start_{op.id}')
        end = model.NewIntVar(1, horizon, f'end_{op.id}')
        interval = model.NewIntervalVar(start, 1, end, f'interval_{op.id}')
        starts[op.id] = start
        ends[op.id] = end
        intervals[op.id] = interval

    # Dependency constraints: if op depends on dep, then start[op] >= end[dep]
    # Only add constraints for deps that are in our op set
    for op in ops:
        for dep_id in op.dep_ids:
            if dep_id in ends:  # Only if dep is in this problem
                model.Add(starts[op.id] >= ends[dep_id])

    # Resource constraints using cumulative
    # Group ops by engine
    ops_by_engine = {}
    for op in ops:
        if op.engine not in ops_by_engine:
            ops_by_engine[op.engine] = []
        ops_by_engine[op.engine].append(op)

    for engine, engine_ops in ops_by_engine.items():
        limit = SLOT_LIMITS.get(engine, 1)
        # Compute demands - use alu_demands if provided, otherwise 1 per op
        if engine == 'alu' and alu_demands:
            demands = [alu_demands.get(op.id, 1) for op in engine_ops]
        else:
            demands = [1] * len(engine_ops)
        engine_intervals = [intervals[op.id] for op in engine_ops]
        model.AddCumulative(engine_intervals, demands, limit)

    # Objective: minimize makespan
    makespan = model.NewIntVar(0, horizon, 'makespan')
    model.AddMaxEquality(makespan, [ends[op.id] for op in ops])
    model.Minimize(makespan)

    # Add hints from greedy solution to help solver
    if greedy_hint:
        for op in ops:
            if op.id in greedy_hint:
                model.AddHint(starts[op.id], greedy_hint[op.id])

    # Solve with optimized parameters
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    # Use all cores with interleaved search strategies
    import os
    num_cores = os.cpu_count() or 8
    solver.parameters.num_search_workers = num_cores
    solver.parameters.interleave_search = True  # Run different strategies in parallel
    solver.parameters.log_search_progress = True  # Enable progress logging
    solver.parameters.linearization_level = 2  # Stronger LP relaxations
    solver.parameters.randomize_search = True  # Help escape local optima
    solver.parameters.cp_model_presolve = True  # Enable presolve
    solver.parameters.cp_model_probing_level = 2  # More aggressive probing

    # Solve, optionally writing verbose output to a log file instead of stdout
    if log_file:
        # Disable stdout logging, enable logging to response
        solver.parameters.log_to_stdout = False
        solver.parameters.log_to_response = True
        status = solver.Solve(model)
        # Write the log to file
        with open(log_file, 'a') as f:
            f.write(solver.ResponseStats() + '\n')
    else:
        status = solver.Solve(model)

    # Log solution quality
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(f"    CP-SAT: {solver.Value(makespan)} cycles, "
              f"bound={solver.BestObjectiveBound():.0f}, "
              f"gap={(solver.Value(makespan)-solver.BestObjectiveBound())/solver.Value(makespan)*100:.1f}%")

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        schedule = {}
        for op in ops:
            cycle = solver.Value(starts[op.id])
            schedule[op.id] = cycle
            op.cycle = cycle

        return CPScheduleResult(
            success=True,
            cycles=solver.Value(makespan),
            schedule=schedule,
            status="optimal" if status == cp_model.OPTIMAL else "feasible"
        )
    else:
        return CPScheduleResult(
            success=False,
            cycles=-1,
            schedule={},
            status=solver.StatusName(status)
        )


def emit_scheduled_cp(ops: list[Op], max_cycle: int) -> list[dict]:
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
