"""
Base kernel builder infrastructure.

This module provides the BaseKernelBuilder class that handles scratch allocation,
instruction emission, debug tracking, and control flow with labels and jumps.
"""

from typing import Dict, List, Tuple

from problem import DebugInfo, VLEN

from .config import KernelConfig


class BaseKernelBuilder:
    """Base class providing common kernel building infrastructure.

    Handles scratch allocation, instruction emission, debug tracking,
    and control flow with labels and jumps.
    """

    def __init__(self, config: KernelConfig):
        self.config = config
        self.instrs: List[Dict] = []
        self.scratch: Dict[str, int] = {}
        self.scratch_debug: Dict[int, Tuple[str, int]] = {}
        self.scratch_ptr = 0
        self.debug_ops: List[Tuple] = []
        # Control flow infrastructure
        self.labels: Dict[str, int] = {}  # label_name -> instruction index
        self.fixups: List[Tuple[int, str]] = []  # (instr_idx, label_name) for forward refs

    def debug_info(self) -> DebugInfo:
        return DebugInfo(scratch_map=self.scratch_debug)

    def alloc(self, name: str = None, length: int = 1) -> int:
        """Allocate scratch memory."""
        addr = self.scratch_ptr
        if name:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        return addr

    def vec(self, name: str) -> int:
        """Allocate a vector (VLEN words) in scratch."""
        return self.alloc(name, VLEN)

    def emit(self, **engines):
        """Emit an instruction with operations for one or more engines."""
        instr = {k: v for k, v in engines.items() if v}
        if instr:
            if self.debug_ops:
                instr["debug"] = list(self.debug_ops)
                self.debug_ops = []
            self.instrs.append(instr)

    def emit_debug(self):
        """Emit a debug-only instruction."""
        if self.debug_ops:
            self.instrs.append({"debug": list(self.debug_ops)})
            self.debug_ops = []

    def emit_compare(self, loc: int, key: tuple):
        """Emit scalar comparison with reference value."""
        if self.config.enable_debug:
            self.debug_ops.append(("compare", loc, key))

    def emit_vcompare(self, loc: int, keys: list):
        """Emit vector comparison with reference values."""
        if self.config.enable_debug:
            self.debug_ops.append(("vcompare", loc, keys))

    # =========================================================================
    # Control Flow Infrastructure (labels and jumps)
    # =========================================================================

    def label(self, name: str) -> None:
        """Define a label at the current instruction position.

        Labels are targets for jump instructions. The label points to the
        NEXT instruction that will be emitted.
        """
        self.labels[name] = len(self.instrs)

    def emit_jump(self, label: str) -> None:
        """Emit an unconditional jump to a label.

        If the label is not yet defined (forward reference), the address
        will be fixed up later by resolve_fixups().
        """
        if label in self.labels:
            addr = self.labels[label]
        else:
            addr = 0  # Placeholder, will be fixed up later
            self.fixups.append((len(self.instrs), label))
        self.emit(flow=[("jump", addr)])

    def emit_cond_jump(self, cond: int, label: str) -> None:
        """Emit a conditional jump to a label.

        Jump is taken if scratch[cond] != 0.
        Forward references are resolved by resolve_fixups().
        """
        if label in self.labels:
            addr = self.labels[label]
        else:
            addr = 0  # Placeholder, will be fixed up later
            self.fixups.append((len(self.instrs), label))
        self.emit(flow=[("cond_jump", cond, addr)])

    def resolve_fixups(self) -> None:
        """Resolve forward references to labels.

        Called after all instructions are emitted to patch jump addresses.
        """
        for instr_idx, label_name in self.fixups:
            if label_name not in self.labels:
                raise ValueError(f"Undefined label: {label_name}")
            addr = self.labels[label_name]
            instr = self.instrs[instr_idx]
            if "flow" in instr:
                new_slots = []
                for slot in instr["flow"]:
                    if slot[0] == "jump":
                        new_slots.append(("jump", addr))
                    elif slot[0] == "cond_jump":
                        new_slots.append(("cond_jump", slot[1], addr))
                    else:
                        new_slots.append(slot)
                instr["flow"] = new_slots
        self.fixups.clear()
