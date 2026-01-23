"""
Intermediate Representation for the optimization solver.

This module defines the IR that represents operations before they are
scheduled and packed into VLIW instructions.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Any


class OpType(Enum):
    """Types of operations in the IR"""
    # ALU operations
    ALU_ADD = auto()
    ALU_SUB = auto()
    ALU_MUL = auto()
    ALU_DIV = auto()
    ALU_XOR = auto()
    ALU_AND = auto()
    ALU_OR = auto()
    ALU_SHL = auto()
    ALU_SHR = auto()
    ALU_MOD = auto()
    ALU_LT = auto()
    ALU_EQ = auto()

    # Vector ALU operations
    VALU_ADD = auto()
    VALU_SUB = auto()
    VALU_MUL = auto()
    VALU_XOR = auto()
    VALU_AND = auto()
    VALU_OR = auto()
    VALU_SHL = auto()
    VALU_SHR = auto()
    VALU_MOD = auto()
    VALU_LT = auto()
    VALU_EQ = auto()
    VALU_BROADCAST = auto()
    VALU_MULTIPLY_ADD = auto()

    # Load operations
    LOAD = auto()
    VLOAD = auto()
    CONST = auto()

    # Store operations
    STORE = auto()
    VSTORE = auto()

    # Flow operations
    SELECT = auto()
    VSELECT = auto()
    JUMP = auto()
    COND_JUMP = auto()
    ADD_IMM = auto()


# Map from OpType to engine name
OP_TO_ENGINE = {
    OpType.ALU_ADD: "alu", OpType.ALU_SUB: "alu", OpType.ALU_MUL: "alu",
    OpType.ALU_DIV: "alu", OpType.ALU_XOR: "alu", OpType.ALU_AND: "alu",
    OpType.ALU_OR: "alu", OpType.ALU_SHL: "alu", OpType.ALU_SHR: "alu",
    OpType.ALU_MOD: "alu", OpType.ALU_LT: "alu", OpType.ALU_EQ: "alu",

    OpType.VALU_ADD: "valu", OpType.VALU_SUB: "valu", OpType.VALU_MUL: "valu",
    OpType.VALU_XOR: "valu", OpType.VALU_AND: "valu", OpType.VALU_OR: "valu",
    OpType.VALU_SHL: "valu", OpType.VALU_SHR: "valu", OpType.VALU_MOD: "valu",
    OpType.VALU_LT: "valu", OpType.VALU_EQ: "valu", OpType.VALU_BROADCAST: "valu",
    OpType.VALU_MULTIPLY_ADD: "valu",

    OpType.LOAD: "load", OpType.VLOAD: "load", OpType.CONST: "load",
    OpType.STORE: "store", OpType.VSTORE: "store",

    OpType.SELECT: "flow", OpType.VSELECT: "flow", OpType.JUMP: "flow",
    OpType.COND_JUMP: "flow", OpType.ADD_IMM: "flow",
}

# Map from string op to OpType
STR_TO_ALU_OP = {
    "+": OpType.ALU_ADD, "-": OpType.ALU_SUB, "*": OpType.ALU_MUL,
    "//": OpType.ALU_DIV, "^": OpType.ALU_XOR, "&": OpType.ALU_AND,
    "|": OpType.ALU_OR, "<<": OpType.ALU_SHL, ">>": OpType.ALU_SHR,
    "%": OpType.ALU_MOD, "<": OpType.ALU_LT, "==": OpType.ALU_EQ,
}

STR_TO_VALU_OP = {
    "+": OpType.VALU_ADD, "-": OpType.VALU_SUB, "*": OpType.VALU_MUL,
    "^": OpType.VALU_XOR, "&": OpType.VALU_AND, "|": OpType.VALU_OR,
    "<<": OpType.VALU_SHL, ">>": OpType.VALU_SHR, "%": OpType.VALU_MOD,
    "<": OpType.VALU_LT, "==": OpType.VALU_EQ,
}

ALU_OP_TO_STR = {v: k for k, v in STR_TO_ALU_OP.items()}
VALU_OP_TO_STR = {v: k for k, v in STR_TO_VALU_OP.items()}


@dataclass
class Value:
    """Represents a value in the IR (like SSA)"""
    id: int
    name: str = ""
    is_vector: bool = False

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, Value) and self.id == other.id


@dataclass
class Operation:
    """Represents a single operation in the IR"""
    id: int
    op_type: OpType
    dest: Optional[Value] = None
    operands: list[Value] = field(default_factory=list)
    immediate: Optional[int] = None  # For const, jump targets, etc.

    # Dependencies (set during analysis)
    data_deps: set['Operation'] = field(default_factory=set)

    # Scheduling info (set by scheduler)
    scheduled_cycle: int = -1

    def engine(self) -> str:
        return OP_TO_ENGINE[self.op_type]

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, Operation) and self.id == other.id


@dataclass
class IRProgram:
    """A program in IR form"""
    operations: list[Operation] = field(default_factory=list)
    values: dict[int, Value] = field(default_factory=dict)

    _next_op_id: int = 0
    _next_value_id: int = 0

    def new_value(self, name: str = "", is_vector: bool = False) -> Value:
        v = Value(self._next_value_id, name, is_vector)
        self.values[v.id] = v
        self._next_value_id += 1
        return v

    def add_op(self, op_type: OpType, dest: Optional[Value] = None,
               operands: list[Value] = None, immediate: int = None) -> Operation:
        op = Operation(
            id=self._next_op_id,
            op_type=op_type,
            dest=dest,
            operands=operands or [],
            immediate=immediate
        )
        self.operations.append(op)
        self._next_op_id += 1
        return op

    def compute_dependencies(self):
        """Compute data dependencies between operations"""
        # Map from value to the operation that produces it
        producers: dict[int, Operation] = {}

        for op in self.operations:
            # Add dependencies on operand producers
            for operand in op.operands:
                if operand.id in producers:
                    op.data_deps.add(producers[operand.id])

            # Register this op as producer of its dest
            if op.dest is not None:
                producers[op.dest.id] = op

    def topological_sort(self) -> list[Operation]:
        """Return operations in topological order"""
        self.compute_dependencies()

        in_degree = {op: len(op.data_deps) for op in self.operations}
        ready = [op for op in self.operations if in_degree[op] == 0]
        result = []

        while ready:
            op = ready.pop(0)
            result.append(op)

            for other in self.operations:
                if op in other.data_deps:
                    in_degree[other] -= 1
                    if in_degree[other] == 0:
                        ready.append(other)

        return result
