"""
Multi-Path Kernel - Uses instruction selection + scheduling via CP-SAT.

Key insight: For some operations, we can choose between:
1. VALU: 6 slots, processes 8 elements per op (SIMD)
2. ALU: 12 slots, processes 1 element per op (scalar)

When VALU is saturated but ALU has capacity, it may be faster to
"unpack" some vector ops into scalar ALU ops.

Example: 32 groups × XOR operation
- VALU path: 32 ops / 6 slots = 6 cycles
- Mixed: 6 VALU (48 elements) + 26×8=208 ALU ops... wait, that's worse

Actually, the opportunity is different:
- For operations where we ALREADY process elements separately (like gather),
  we can better pack with the scheduler choosing optimal slot assignment.
"""

from dataclasses import dataclass, field
from solver.multi_path_ir import MultiPathIR, Implementation, Engine, schedule_multi_path
from problem import VLEN, HASH_STAGES


@dataclass
class MultiPathKernelBuilder:
    """Build kernel using multi-path IR for instruction selection."""

    ir: MultiPathIR = field(default_factory=MultiPathIR)
    scratch_ptr: int = 0
    scratch_debug: dict = field(default_factory=dict)

    def alloc(self, name: str = None, length: int = 1) -> int:
        addr = self.scratch_ptr
        if name:
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        return addr

    def vec(self, name: str) -> int:
        return self.alloc(name, VLEN)

    def build_kernel(self, forest_height: int, n_nodes: int,
                     batch_size: int, rounds: int):
        """Build kernel with multi-path instruction selection."""

        n_groups = batch_size // VLEN  # 32
        UNROLL = 8
        n_iters = n_groups // UNROLL  # 4

        ir = self.ir

        # Allocate scratch
        forest_p = self.alloc("forest_p")
        idx_p = self.alloc("idx_p")
        val_p = self.alloc("val_p")
        n_nodes_s = self.alloc("n_nodes")
        st = [self.alloc(f"st{i}") for i in range(16)]
        c = [self.alloc(f"c{i}") for i in range(16)]
        hc = [self.alloc(f"hc{i}") for i in range(6)]
        hs = [self.alloc(f"hs{i}") for i in range(6)]
        tc = [self.alloc(f"tc{i}") for i in range(7)]

        all_idx = [self.vec(f"all_idx_{g}") for g in range(n_groups)]
        all_val = [self.vec(f"all_val_{g}") for g in range(n_groups)]
        v_nv = [[self.vec(f"v_nv_{it}_{i}") for i in range(UNROLL)] for it in range(n_iters)]
        v_t = [[self.vec(f"v_t_{it}_{i}") for i in range(UNROLL)] for it in range(n_iters)]

        v_0 = self.vec("v_0")
        v_1 = self.vec("v_1")
        v_2 = self.vec("v_2")
        v_nn = self.vec("v_nn")
        v_hc = [self.vec(f"v_hc{i}") for i in range(6)]
        v_hs = [self.vec(f"v_hs{i}") for i in range(6)]
        v_tc = [self.vec(f"v_tc{i}") for i in range(3)]

        print(f"Scratch usage: {self.scratch_ptr}")

        # === INIT PHASE ===
        # Load constants (LOAD engine only) - no dependencies
        const_ops = []
        for i in range(8):
            const_ops.append(ir.load_const(c[i], i))

        # Load pointers (after some constants loaded)
        load_forest = ir.load_mem(forest_p, c[4], deps=const_ops[:5])
        load_idx = ir.load_mem(idx_p, c[5], deps=const_ops[:6])
        load_val = ir.load_mem(val_p, c[6], deps=const_ops[:7])
        load_nn = ir.load_mem(n_nodes_s, c[1], deps=const_ops[:2])

        # Load hash constants
        hc_ops = []
        for i in range(6):
            hc_ops.append(ir.load_const(hc[i], HASH_STAGES[i][1]))
            hc_ops.append(ir.load_const(hs[i], HASH_STAGES[i][4]))

        # Broadcasts (VALU only - true SIMD benefit)
        bcast_deps = const_ops + [load_nn] + hc_ops
        bcast_0 = ir.broadcast(v_0, c[0], deps=bcast_deps)
        bcast_1 = ir.broadcast(v_1, c[1], deps=bcast_deps)
        bcast_2 = ir.broadcast(v_2, c[2], deps=bcast_deps)
        bcast_nn = ir.broadcast(v_nn, n_nodes_s, deps=[load_nn])

        bcast_hc = [ir.broadcast(v_hc[i], hc[i], deps=[hc_ops[i*2]]) for i in range(6)]
        bcast_hs = [ir.broadcast(v_hs[i], hs[i], deps=[hc_ops[i*2+1]]) for i in range(6)]

        # Load tree cache (simplified - just load tc[0])
        tc_load = ir.load_mem(tc[0], forest_p, deps=[load_forest])
        tc_bcast = [ir.broadcast(v_tc[0], tc[0], deps=[tc_load])]

        init_complete = [bcast_0, bcast_1, bcast_2, bcast_nn] + bcast_hc + bcast_hs + tc_bcast + [load_idx, load_val]

        # === LOAD DATA PHASE ===
        load_complete = []

        for g in range(n_groups):
            # Address calculations (ALU - scalar)
            const_g = ir.load_const(st[g % 8], g * VLEN, deps=init_complete)
            addr_idx = ir.scalar_add(st[8 + g % 4], idx_p, st[g % 8], deps=[const_g])
            addr_val = ir.scalar_add(st[12 + g % 4], val_p, st[g % 8], deps=[const_g])

            # Vector loads (LOAD engine)
            vload_idx = ir.vload(all_idx[g], st[8 + g % 4], deps=[addr_idx])
            vload_val = ir.vload(all_val[g], st[12 + g % 4], deps=[addr_val])
            load_complete.extend([vload_idx, vload_val])

        # === ROUNDS PHASE (simplified for testing) ===
        prev_round_ops = load_complete

        for rnd in range(min(rounds, 2)):  # Just 2 rounds for testing
            # Track ops for this round
            bcast_ops = {}  # (it, i) -> op_id
            xor_ops = {}
            hash_ops = {}
            branch_ops = {}

            # Broadcast phase
            for it in range(n_iters):
                for i in range(UNROLL):
                    op = ir.broadcast(v_nv[it][i], tc[0], deps=prev_round_ops)
                    bcast_ops[(it, i)] = op

            # XOR phase
            for it in range(n_iters):
                base_g = it * UNROLL
                for i in range(UNROLL):
                    op = ir.vector_xor(all_val[base_g+i], all_val[base_g+i],
                                      v_nv[it][i], deps=[bcast_ops[(it, i)]])
                    xor_ops[(it, i)] = op

            # Hash phase (1 stage)
            for it in range(n_iters):
                base_g = it * UNROLL
                for i in range(UNROLL):
                    t = v_t[it][i]
                    val_addr = all_val[base_g+i]

                    h1 = ir.vector_mul(t, val_addr, v_hc[0], deps=[xor_ops[(it, i)]])
                    h2 = ir.vector_xor(val_addr, val_addr, t, deps=[h1])
                    hash_ops[(it, i)] = h2

            # Branch phase - FLEXIBLE OPS!
            for it in range(n_iters):
                base_g = it * UNROLL
                for i in range(UNROLL):
                    idx_addr = all_idx[base_g+i]
                    val_addr = all_val[base_g+i]
                    t = v_t[it][i]

                    # These can use VALU OR ALU - scheduler picks!
                    b1 = ir.arithmetic_flexible(idx_addr, idx_addr, v_2, '*', deps=[hash_ops[(it, i)]])
                    b2 = ir.arithmetic_flexible(t, val_addr, v_1, '&', deps=[hash_ops[(it, i)]])
                    b3 = ir.arithmetic_flexible(t, t, v_1, '+', deps=[b2])
                    b4 = ir.arithmetic_flexible(idx_addr, idx_addr, t, '+', deps=[b1, b3])
                    branch_ops[(it, i)] = b4

            prev_round_ops = list(branch_ops.values())

        # === STORE PHASE ===
        store_deps = prev_round_ops
        for g in range(n_groups):
            const_g = ir.load_const(st[g % 8], g * VLEN, deps=store_deps)
            addr_idx = ir.scalar_add(st[8 + g % 4], idx_p, st[g % 8], deps=[const_g])
            addr_val = ir.scalar_add(st[12 + g % 4], val_p, st[g % 8], deps=[const_g])
            ir.vstore(st[8 + g % 4], all_idx[g], deps=[addr_idx])
            ir.vstore(st[12 + g % 4], all_val[g], deps=[addr_val])

        print(f"Created {len(ir.ops)} operations")
        return ir


def test_multi_path_kernel():
    """Test the multi-path kernel."""
    builder = MultiPathKernelBuilder()
    ir = builder.build_kernel(
        forest_height=8,
        n_nodes=255,
        batch_size=256,
        rounds=3  # Small for testing
    )

    print(f"\nScheduling {len(ir.ops)} ops with instruction selection...")
    cycles, instrs = schedule_multi_path(ir, time_limit=120.0)

    print(f"\nResult: {cycles} cycles, {len(instrs)} instructions")

    # Count engine usage
    valu_total = 0
    alu_total = 0
    load_total = 0
    store_total = 0

    for op in ir.ops:
        if op.chosen_impl is not None:
            impl = op.implementations[op.chosen_impl]
            if impl.engine == Engine.VALU:
                valu_total += 1
            elif impl.engine == Engine.ALU:
                alu_total += 1
            elif impl.engine == Engine.LOAD:
                load_total += 1
            elif impl.engine == Engine.STORE:
                store_total += 1

    print(f"\nEngine selection:")
    print(f"  VALU: {valu_total} ops")
    print(f"  ALU: {alu_total} ops")
    print(f"  LOAD: {load_total} ops")
    print(f"  STORE: {store_total} ops")

    # Show a few cycles
    print(f"\nFirst 10 cycles:")
    for i, instr in enumerate(instrs[:10]):
        parts = []
        for eng, ops in instr.items():
            parts.append(f"{eng}={len(ops)}")
        print(f"  Cycle {i}: {', '.join(parts)}")

    return cycles, instrs


if __name__ == "__main__":
    from solver.multi_path_ir import Engine
    test_multi_path_kernel()
