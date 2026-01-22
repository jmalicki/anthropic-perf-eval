"""
Multi-Path IR - Operations can have multiple implementation choices.

Each logical operation can be implemented in different ways (e.g., VALU vs ALU),
and the scheduler chooses which path to use. Constraints ensure exactly one
implementation is selected per operation.

Example: A scalar multiply could be:
  - ALU: uses ALU slot, 1 cycle
  - VALU broadcast + multiply: uses VALU slot, 2 cycles but frees ALU

The scheduler picks based on resource availability and dependencies.
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class Engine(Enum):
    ALU = "alu"
    VALU = "valu"
    LOAD = "load"
    STORE = "store"
    FLOW = "flow"


@dataclass
class Implementation:
    """One way to implement a logical operation."""
    engine: Engine
    slots: list[tuple]  # The actual instruction slots
    latency: int = 1    # Cycles until result is ready
    throughput: int = 1  # Cycles this impl occupies the engine


@dataclass
class LogicalOp:
    """A logical operation with one or more implementation choices."""
    id: int
    name: str
    implementations: list[Implementation]
    dep_ids: list[int] = field(default_factory=list)  # Dependencies on other LogicalOps

    # After scheduling:
    chosen_impl: Optional[int] = None  # Index into implementations
    cycle: Optional[int] = None


@dataclass
class MultiPathIR:
    """IR that supports multiple implementation paths."""
    ops: list[LogicalOp] = field(default_factory=list)
    op_counter: int = 0

    def add_op(self, name: str, implementations: list[Implementation],
               deps: list[int] = None) -> int:
        """Add a logical operation with multiple implementation choices."""
        op_id = self.op_counter
        self.op_counter += 1
        self.ops.append(LogicalOp(
            id=op_id,
            name=name,
            implementations=implementations,
            dep_ids=deps or []
        ))
        return op_id

    # Convenience methods for common operations

    def scalar_add(self, dest: int, src1: int, src2: int, deps: list[int] = None) -> int:
        """Scalar add - can use ALU."""
        return self.add_op(
            f"add_s_{dest}",
            [Implementation(Engine.ALU, [("+", dest, src1, src2)])],
            deps
        )

    def scalar_mul(self, dest: int, src1: int, src2: int, deps: list[int] = None) -> int:
        """Scalar multiply - can use ALU."""
        return self.add_op(
            f"mul_s_{dest}",
            [Implementation(Engine.ALU, [("*", dest, src1, src2)])],
            deps
        )

    def vector_add(self, dest: int, src1: int, src2: int, deps: list[int] = None) -> int:
        """Vector add - VALU only."""
        return self.add_op(
            f"vadd_{dest}",
            [Implementation(Engine.VALU, [("+", dest, src1, src2)])],
            deps
        )

    def vector_mul(self, dest: int, src1: int, src2: int, deps: list[int] = None) -> int:
        """Vector multiply - VALU only."""
        return self.add_op(
            f"vmul_{dest}",
            [Implementation(Engine.VALU, [("*", dest, src1, src2)])],
            deps
        )

    def vector_xor(self, dest: int, src1: int, src2: int, deps: list[int] = None) -> int:
        """Vector XOR - VALU only."""
        return self.add_op(
            f"vxor_{dest}",
            [Implementation(Engine.VALU, [("^", dest, src1, src2)])],
            deps
        )

    def broadcast(self, dest: int, src: int, deps: list[int] = None) -> int:
        """Broadcast scalar to vector - VALU only."""
        return self.add_op(
            f"vbcast_{dest}",
            [Implementation(Engine.VALU, [("vbroadcast", dest, src)])],
            deps
        )

    def load_const(self, dest: int, value: int, deps: list[int] = None) -> int:
        """Load constant - LOAD only."""
        return self.add_op(
            f"const_{dest}",
            [Implementation(Engine.LOAD, [("const", dest, value)])],
            deps
        )

    def load_mem(self, dest: int, addr: int, deps: list[int] = None) -> int:
        """Load from memory - LOAD only."""
        return self.add_op(
            f"load_{dest}",
            [Implementation(Engine.LOAD, [("load", dest, addr)])],
            deps
        )

    def vload(self, dest: int, addr: int, deps: list[int] = None) -> int:
        """Vector load - LOAD only."""
        return self.add_op(
            f"vload_{dest}",
            [Implementation(Engine.LOAD, [("vload", dest, addr)])],
            deps
        )

    def store_mem(self, addr: int, src: int, deps: list[int] = None) -> int:
        """Store to memory - STORE only."""
        return self.add_op(
            f"store_{addr}",
            [Implementation(Engine.STORE, [("store", addr, src)])],
            deps
        )

    def vstore(self, addr: int, src: int, deps: list[int] = None) -> int:
        """Vector store - STORE only."""
        return self.add_op(
            f"vstore_{addr}",
            [Implementation(Engine.STORE, [("vstore", addr, src)])],
            deps
        )

    # Multi-path operations (the interesting ones!)

    def flexible_add(self, dest: int, src1: int, src2: int,
                     is_vector: bool = False, deps: list[int] = None) -> int:
        """Add that can use either ALU or VALU (if scalar broadcast is acceptable)."""
        impls = [Implementation(Engine.ALU, [("+", dest, src1, src2)])]
        if is_vector:
            impls = [Implementation(Engine.VALU, [("+", dest, src1, src2)])]
        return self.add_op(f"flex_add_{dest}", impls, deps)

    def index_calc_flexible(self, dest: int, base: int, idx_addr: int,
                           deps: list[int] = None) -> int:
        """
        Index calculation: dest = base + mem[idx_addr]

        Can be done as:
        1. ALU path: add dest, base, idx_addr (if idx is already in scratch)
        2. Could potentially fuse with subsequent load
        """
        return self.add_op(
            f"idx_calc_{dest}",
            [Implementation(Engine.ALU, [("+", dest, base, idx_addr)])],
            deps
        )

    def compare_flexible(self, dest: int, src1: int, src2: int, op: str = "==",
                        is_vector: bool = True, deps: list[int] = None) -> int:
        """
        Compare operation - can use VALU or ALU depending on context.

        For vectors, must use VALU.
        For scalars, could use ALU (if available) or VALU.
        """
        impls = []
        if is_vector:
            impls.append(Implementation(Engine.VALU, [(op, dest, src1, src2)]))
        else:
            # Scalar: can use either ALU or VALU
            impls.append(Implementation(Engine.ALU, [(op, dest, src1, src2)]))
            impls.append(Implementation(Engine.VALU, [(op, dest, src1, src2)]))
        return self.add_op(f"cmp_{dest}", impls, deps)

    def vector_op_flexible(self, dest: int, src1: int, src2: int, op: str,
                          deps: list[int] = None) -> int:
        """
        Vector operation that can use VALU or ALU.

        Key insight: ALU has 12 slots vs VALU's 6!
        For element-wise vector ops where we process each element separately,
        we can choose to use ALU for some elements and VALU for others.

        This is especially powerful with full unrolling - different trees
        can use different execution units to maximize parallelism.
        """
        # Both ALU and VALU can do basic arithmetic
        return self.add_op(
            f"vop_{op}_{dest}",
            [
                Implementation(Engine.VALU, [(op, dest, src1, src2)]),
                Implementation(Engine.ALU, [(op, dest, src1, src2)]),
            ],
            deps
        )

    def arithmetic_flexible(self, dest: int, src1: int, src2: int, op: str,
                           prefer_valu: bool = True, deps: list[int] = None) -> int:
        """
        Arithmetic op that scheduler can assign to ALU or VALU.

        With 6 VALU + 12 ALU slots, we can do 18 ops per cycle if we mix!
        The scheduler will balance based on what other ops need each engine.
        """
        if prefer_valu:
            impls = [
                Implementation(Engine.VALU, [(op, dest, src1, src2)]),
                Implementation(Engine.ALU, [(op, dest, src1, src2)]),
            ]
        else:
            impls = [
                Implementation(Engine.ALU, [(op, dest, src1, src2)]),
                Implementation(Engine.VALU, [(op, dest, src1, src2)]),
            ]
        return self.add_op(f"arith_{op}_{dest}", impls, deps)

    def multiply_or_shift(self, dest: int, src: int, by_two: int,
                         deps: list[int] = None) -> int:
        """
        Multiply by 2 - can be done as multiply or left shift.

        Both use same engine but demonstrates the concept.
        In a real scenario, shift might be faster or use different resources.
        """
        return self.add_op(
            f"mul2_{dest}",
            [
                Implementation(Engine.VALU, [("*", dest, src, by_two)]),
                Implementation(Engine.VALU, [("<<", dest, src, by_two)]),  # Shift by 1
            ],
            deps
        )

    def branch_bit_calc(self, bit_dest: int, idx_dest: int,
                       val: int, idx: int, v_1: int, v_2: int, v_nn: int,
                       deps: list[int] = None) -> list[int]:
        """
        Branch computation with flexible implementation.

        Standard path (6 VALU ops):
          idx = idx * 2
          bit = val & 1
          bit = bit + 1
          idx = idx + bit
          in_bounds = idx < n_nodes
          idx = idx * in_bounds

        Alternative: Could use multiply_add fusion for some steps.
        """
        op_ids = []

        # idx = idx * 2 (or shift)
        op1 = self.add_op(
            f"idx_mul2_{idx_dest}",
            [
                Implementation(Engine.VALU, [("*", idx_dest, idx, v_2)]),
                Implementation(Engine.VALU, [("<<", idx_dest, idx, v_1)]),
            ],
            deps
        )
        op_ids.append(op1)

        # bit = val & 1
        op2 = self.add_op(
            f"bit_and_{bit_dest}",
            [Implementation(Engine.VALU, [("&", bit_dest, val, v_1)])],
            deps  # Can run in parallel with op1
        )
        op_ids.append(op2)

        # bit = bit + 1
        op3 = self.add_op(
            f"bit_inc_{bit_dest}",
            [Implementation(Engine.VALU, [("+", bit_dest, bit_dest, v_1)])],
            [op2]
        )
        op_ids.append(op3)

        # idx = idx + bit
        op4 = self.add_op(
            f"idx_add_bit_{idx_dest}",
            [Implementation(Engine.VALU, [("+", idx_dest, idx_dest, bit_dest)])],
            [op1, op3]
        )
        op_ids.append(op4)

        # in_bounds = idx < n_nodes (reuse bit_dest)
        op5 = self.add_op(
            f"in_bounds_{bit_dest}",
            [Implementation(Engine.VALU, [("<", bit_dest, idx_dest, v_nn)])],
            [op4]
        )
        op_ids.append(op5)

        # idx = idx * in_bounds
        op6 = self.add_op(
            f"idx_clamp_{idx_dest}",
            [Implementation(Engine.VALU, [("*", idx_dest, idx_dest, bit_dest)])],
            [op5]
        )
        op_ids.append(op6)

        return op_ids

    def hash_stage_flexible(self, dest: int, val: int, hc: int, hs: int, tmp: int,
                           op1: str, op2: str, op3: str,
                           deps: list[int] = None) -> int:
        """
        Hash stage with multiple implementation options:

        Standard (2 VALU ops):
          t = val op1 hc
          v = val op3 hs
          dest = t op2 v

        Could potentially be done differently based on available resources.
        """
        # For now, single implementation
        # TODO: Add alternative implementations
        return self.add_op(
            f"hash_{dest}",
            [
                Implementation(
                    Engine.VALU,
                    [
                        (op1, tmp, val, hc),
                        (op3, dest, val, hs),  # Use dest as second temp
                    ],
                    latency=1
                ),
            ],
            deps
        )


def schedule_multi_path(ir: MultiPathIR, time_limit: float = 60.0) -> tuple[int, list[dict]]:
    """
    Schedule a multi-path IR using CP-SAT.

    For each logical op:
    - Create a boolean variable for each implementation choice
    - Exactly one implementation must be chosen
    - Create interval variables for each implementation
    - Only the chosen implementation's interval is active
    - Dependencies are between logical ops (any impl satisfies dep)
    """
    from ortools.sat.python import cp_model

    SLOT_LIMITS = {
        Engine.ALU: 12,
        Engine.VALU: 6,
        Engine.LOAD: 2,
        Engine.STORE: 2,
        Engine.FLOW: 1,
    }

    model = cp_model.CpModel()
    ops = ir.ops
    n_ops = len(ops)

    # Estimate horizon
    horizon = n_ops * 2

    # Variables for each op
    op_vars = {}  # op_id -> {impl_idx -> (chosen_bool, start, end, interval)}
    op_completion = {}  # op_id -> completion_time (IntVar)

    for op in ops:
        op_vars[op.id] = {}

        # Completion time for this logical op
        completion = model.NewIntVar(0, horizon, f"completion_{op.id}")
        op_completion[op.id] = completion

        impl_chosen = []
        impl_completions = []

        for impl_idx, impl in enumerate(op.implementations):
            # Boolean: is this implementation chosen?
            chosen = model.NewBoolVar(f"chosen_{op.id}_{impl_idx}")
            impl_chosen.append(chosen)

            # Start time for this implementation
            start = model.NewIntVar(0, horizon - 1, f"start_{op.id}_{impl_idx}")
            end = model.NewIntVar(1, horizon, f"end_{op.id}_{impl_idx}")

            # Duration based on throughput
            model.Add(end == start + impl.throughput)

            # Completion time for this impl
            impl_completion = model.NewIntVar(0, horizon, f"impl_comp_{op.id}_{impl_idx}")
            model.Add(impl_completion == start + impl.latency)
            impl_completions.append((chosen, impl_completion))

            # Optional interval (only active if chosen)
            interval = model.NewOptionalIntervalVar(
                start, impl.throughput, end, chosen,
                f"interval_{op.id}_{impl_idx}"
            )

            op_vars[op.id][impl_idx] = {
                'chosen': chosen,
                'start': start,
                'end': end,
                'interval': interval,
                'impl': impl,
                'completion': impl_completion,
            }

        # Exactly one implementation must be chosen
        model.AddExactlyOne(impl_chosen)

        # Link completion time to chosen implementation's completion
        for chosen, impl_comp in impl_completions:
            model.Add(completion == impl_comp).OnlyEnforceIf(chosen)

    # Dependencies: op must start after all deps complete
    for op in ops:
        for dep_id in op.dep_ids:
            dep_completion = op_completion[dep_id]
            for impl_idx, impl_data in op_vars[op.id].items():
                # If this impl is chosen, it must start after dep completes
                model.Add(impl_data['start'] >= dep_completion).OnlyEnforceIf(impl_data['chosen'])

    # Resource constraints: cumulative per engine
    for engine in Engine:
        engine_intervals = []
        engine_demands = []

        for op in ops:
            for impl_idx, impl_data in op_vars[op.id].items():
                if impl_data['impl'].engine == engine:
                    engine_intervals.append(impl_data['interval'])
                    engine_demands.append(1)

        if engine_intervals:
            model.AddCumulative(engine_intervals, engine_demands, SLOT_LIMITS[engine])

    # Objective: minimize makespan
    makespan = model.NewIntVar(0, horizon, "makespan")
    for op in ops:
        model.Add(makespan >= op_completion[op.id])

    model.Minimize(makespan)

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # Extract solution
        for op in ops:
            for impl_idx, impl_data in op_vars[op.id].items():
                if solver.Value(impl_data['chosen']):
                    op.chosen_impl = impl_idx
                    op.cycle = solver.Value(impl_data['start'])
                    break

        # Emit instructions
        max_cycle = solver.Value(makespan)
        instructions = emit_multi_path(ops, max_cycle)

        return max_cycle, instructions
    else:
        raise ValueError(f"CP-SAT failed with status {status}")


def emit_multi_path(ops: list[LogicalOp], max_cycle: int) -> list[dict]:
    """Emit instructions from scheduled multi-path ops."""
    # Group ops by cycle
    by_cycle = {}
    for op in ops:
        if op.cycle not in by_cycle:
            by_cycle[op.cycle] = []
        by_cycle[op.cycle].append(op)

    instructions = []
    for cycle in range(max_cycle):
        if cycle not in by_cycle:
            continue

        instr = {}
        for op in by_cycle[cycle]:
            impl = op.implementations[op.chosen_impl]
            engine_name = impl.engine.value

            if engine_name not in instr:
                instr[engine_name] = []
            instr[engine_name].extend(impl.slots)

        if instr:
            instructions.append(instr)

    return instructions
