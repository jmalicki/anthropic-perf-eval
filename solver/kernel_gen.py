"""
Optimized Kernel Generator.

This module generates optimized VLIW instructions using the optimization
results from the scheduler, vectorizer, and register allocator.
"""

from dataclasses import dataclass, field
from typing import Optional, Any
from problem import (
    DebugInfo, SLOT_LIMITS, VLEN, SCRATCH_SIZE, HASH_STAGES
)


@dataclass
class OptimizedKernelBuilder:
    """
    Builds an optimized kernel using vectorization and ILP.

    Key optimizations:
    1. Vectorize batch processing (process 8 elements per vector op)
    2. Pack multiple operations per cycle (exploit all slots)
    3. Software pipelining (overlap iterations)
    4. Minimize memory traffic (reuse scratch values)
    """

    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def alloc_scratch(self, name: str = None, length: int = 1) -> int:
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def alloc_vector(self, name: str) -> int:
        """Allocate a vector register (8 words)"""
        return self.alloc_scratch(name, VLEN)

    def emit(self, instr: dict):
        """Emit a single instruction"""
        self.instrs.append(instr)

    def emit_packed(self, *slots):
        """
        Emit a VLIW instruction with multiple slots packed together.
        Each slot is (engine, op_tuple).
        """
        instr = {}
        for engine, slot in slots:
            if engine not in instr:
                instr[engine] = []
            instr[engine].append(slot)
        self.instrs.append(instr)

    def const(self, val: int, name: str = None) -> int:
        """Get or create a constant in scratch"""
        if val not in self.const_map:
            addr = self.alloc_scratch(name or f"const_{val}")
            self.emit({"load": [("const", addr, val)]})
            self.const_map[val] = addr
        return self.const_map[val]

    def vconst(self, val: int, name: str = None) -> int:
        """Get or create a vector constant (broadcast)"""
        key = ("vec", val)
        if key not in self.const_map:
            scalar = self.const(val)
            vec_addr = self.alloc_vector(name or f"vconst_{val}")
            self.emit({"valu": [("vbroadcast", vec_addr, scalar)]})
            self.const_map[key] = vec_addr
        return self.const_map[key]

    def build_kernel(
        self,
        forest_height: int,
        n_nodes: int,
        batch_size: int,
        rounds: int
    ):
        """
        Build the optimized kernel.

        Strategy:
        1. Process batch in groups of 8 (VLEN)
        2. Vectorize the hash computation
        3. Use gather for tree node lookups (can't vectorize random access)
        4. Pack operations into VLIW bundles
        """
        n_groups = batch_size // VLEN

        # Allocate pointers and constants
        forest_values_p = self.alloc_scratch("forest_values_p")
        inp_indices_p = self.alloc_scratch("inp_indices_p")
        inp_values_p = self.alloc_scratch("inp_values_p")
        n_nodes_s = self.alloc_scratch("n_nodes")

        # Load memory layout from header
        tmp = self.alloc_scratch("tmp")
        for i, (name, addr) in enumerate([
            ("forest_values_p", forest_values_p),
            ("inp_indices_p", inp_indices_p),
            ("inp_values_p", inp_values_p),
        ]):
            self.emit_packed(
                ("load", ("const", tmp, 4 + i)),
                ("load", ("load", addr, tmp)) if i > 0 else ("load", ("const", tmp, 4 + i)),
            )
            if i == 0:
                self.emit({"load": [("load", addr, tmp)]})
            else:
                pass  # Already emitted

        # Actually load all pointers properly
        self.instrs = []  # Reset and do it right
        tmp = self.alloc_scratch("tmp")

        # Load all pointers in parallel where possible
        self.emit({"load": [("const", tmp, 4)]})
        self.emit({"load": [("load", forest_values_p, tmp)]})
        self.emit({"load": [("const", tmp, 5)]})
        self.emit({"load": [("load", inp_indices_p, tmp)]})
        self.emit({"load": [("const", tmp, 6)]})
        self.emit({"load": [("load", inp_values_p, tmp)]})
        self.emit({"load": [("const", tmp, 1)]})  # n_nodes at offset 1
        self.emit({"load": [("load", n_nodes_s, tmp)]})

        # Constants
        c0 = self.const(0, "c0")
        c1 = self.const(1, "c1")
        c2 = self.const(2, "c2")

        # Hash constants
        hash_consts = []
        for i, (_, val, _, _, _) in enumerate(HASH_STAGES):
            hash_consts.append(self.const(val, f"hash_{i}"))

        # Shift amounts
        shift_amts = []
        for i, (_, _, _, _, amt) in enumerate(HASH_STAGES):
            shift_amts.append(self.const(amt, f"shift_{i}"))

        # Vector constants
        vc0 = self.vconst(0, "vc0")
        vc1 = self.vconst(1, "vc1")
        vc2 = self.vconst(2, "vc2")

        # Broadcast hash constants to vectors
        vhash_consts = []
        for i, (_, val, _, _, _) in enumerate(HASH_STAGES):
            vhash_consts.append(self.vconst(val, f"vhash_{i}"))

        vshift_amts = []
        for i, (_, _, _, _, amt) in enumerate(HASH_STAGES):
            vshift_amts.append(self.vconst(amt, f"vshift_{i}"))

        # Vector registers for processing
        v_indices = self.alloc_vector("v_indices")
        v_values = self.alloc_vector("v_values")
        v_node_vals = self.alloc_vector("v_node_vals")
        v_tmp1 = self.alloc_vector("v_tmp1")
        v_tmp2 = self.alloc_vector("v_tmp2")
        v_next_idx = self.alloc_vector("v_next_idx")

        # Scalar temps for address computation
        s_addr = self.alloc_scratch("s_addr")
        s_base = self.alloc_scratch("s_base")
        s_tmp = self.alloc_scratch("s_tmp")

        self.emit({"flow": [("pause",)]})

        # Main loop: rounds x groups
        for round_idx in range(rounds):
            for group_idx in range(n_groups):
                group_offset = group_idx * VLEN

                # 1. Load indices vector: vload(v_indices, inp_indices_p + group_offset)
                self.emit_packed(
                    ("alu", ("+", s_addr, inp_indices_p, self.const(group_offset))),
                )
                self.emit({"load": [("vload", v_indices, s_addr)]})

                # 2. Load values vector
                self.emit_packed(
                    ("alu", ("+", s_addr, inp_values_p, self.const(group_offset))),
                )
                self.emit({"load": [("vload", v_values, s_addr)]})

                # 3. Gather tree node values (scalar loop - can't vectorize random access)
                for vi in range(VLEN):
                    # addr = forest_values_p + indices[vi]
                    self.emit_packed(
                        ("alu", ("+", s_addr, forest_values_p, v_indices + vi)),
                    )
                    self.emit({"load": [("load", v_node_vals + vi, s_addr)]})

                # 4. XOR values with node values
                self.emit({"valu": [("^", v_values, v_values, v_node_vals)]})

                # 5. Hash computation (6 stages)
                for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    # tmp1 = op1(val, const)
                    # tmp2 = op3(val, shift)
                    # val = op2(tmp1, tmp2)
                    self.emit({"valu": [
                        (op1, v_tmp1, v_values, vhash_consts[hi]),
                        (op3, v_tmp2, v_values, vshift_amts[hi]),
                    ]})
                    self.emit({"valu": [(op2, v_values, v_tmp1, v_tmp2)]})

                # 6. Compute next index: 2*idx + (1 if val%2==0 else 2)
                # tmp1 = val % 2
                self.emit({"valu": [("%", v_tmp1, v_values, vc2)]})
                # tmp1 = (tmp1 == 0)
                self.emit({"valu": [("==", v_tmp1, v_tmp1, vc0)]})
                # tmp2 = select(tmp1, 1, 2)
                self.emit({"flow": [("vselect", v_tmp2, v_tmp1, vc1, vc2)]})
                # next_idx = 2 * idx + tmp2
                self.emit({"valu": [("*", v_next_idx, v_indices, vc2)]})
                self.emit({"valu": [("+", v_next_idx, v_next_idx, v_tmp2)]})

                # 7. Bounds check: idx = (idx < n_nodes) ? idx : 0
                # Broadcast n_nodes to vector
                if group_idx == 0 and round_idx == 0:
                    vn_nodes = self.alloc_vector("vn_nodes")
                    self.emit({"valu": [("vbroadcast", vn_nodes, n_nodes_s)]})
                else:
                    vn_nodes = self.scratch["vn_nodes"]

                self.emit({"valu": [("<", v_tmp1, v_next_idx, vn_nodes)]})
                self.emit({"flow": [("vselect", v_indices, v_tmp1, v_next_idx, vc0)]})

                # 8. Store results
                self.emit_packed(
                    ("alu", ("+", s_addr, inp_indices_p, self.const(group_offset))),
                )
                self.emit({"store": [("vstore", s_addr, v_indices)]})

                self.emit_packed(
                    ("alu", ("+", s_addr, inp_values_p, self.const(group_offset))),
                )
                self.emit({"store": [("vstore", s_addr, v_values)]})

        self.emit({"flow": [("pause",)]})


class VLIWPacker:
    """
    Packs operations into VLIW bundles respecting slot limits.
    """

    def __init__(self):
        self.slot_limits = SLOT_LIMITS.copy()

    def pack(self, operations: list[tuple[str, tuple]]) -> list[dict]:
        """
        Pack a list of (engine, slot) pairs into VLIW instructions.
        Uses a greedy algorithm to maximize packing.
        """
        result = []
        remaining = list(operations)

        while remaining:
            bundle = {}
            used = {engine: 0 for engine in self.slot_limits}
            packed = []

            for i, (engine, slot) in enumerate(remaining):
                if engine not in self.slot_limits:
                    continue
                if used[engine] < self.slot_limits[engine]:
                    if engine not in bundle:
                        bundle[engine] = []
                    bundle[engine].append(slot)
                    used[engine] += 1
                    packed.append(i)

            # Remove packed operations (in reverse to preserve indices)
            for i in reversed(packed):
                remaining.pop(i)

            if bundle:
                result.append(bundle)
            elif remaining:
                # Can't pack more - emit one at a time
                engine, slot = remaining.pop(0)
                result.append({engine: [slot]})

        return result
