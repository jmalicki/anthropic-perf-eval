"""
Multi-Path Interleaved Kernel - Uses multi-path IR for VALU/ALU selection.

Key insight: Some vector operations can be done on either:
- VALU: 1 slot, processes 8 elements
- ALU: 8 slots, processes 8 elements (less efficient but uses idle ALU capacity)

With 6 VALU + 12 ALU slots, when VALU is saturated, we can shift work to ALU.
The scheduler picks the best assignment based on resource availability.
"""

from problem import DebugInfo, VLEN, SCRATCH_SIZE, HASH_STAGES
from solver.multi_path_ir import MultiPathIR, Implementation, Engine, schedule_multi_path


class MultiPathInterleavedKernelBuilder:
    """Build kernel using multi-path IR for instruction selection."""

    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.ir = MultiPathIR()

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def alloc(self, name: str = None, length: int = 1) -> int:
        addr = self.scratch_ptr
        if name:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        return addr

    def vec(self, name: str) -> int:
        return self.alloc(name, VLEN)

    def build_kernel(
        self,
        forest_height: int,
        n_nodes: int,
        batch_size: int,
        rounds: int
    ):
        """Build kernel with VALU/ALU selection via multi-path IR."""
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

        print(f"Scratch usage: {self.scratch_ptr} / {SCRATCH_SIZE}")

        # === BUILD MULTI-PATH IR ===

        # INIT: Load constants (LOAD only)
        const_ops = []
        for i in range(8):
            const_ops.append(ir.load_const(c[i], i))

        # Load pointers
        ptr_ops = [
            ir.load_mem(forest_p, c[4], deps=const_ops[:5]),
            ir.load_mem(idx_p, c[5], deps=const_ops[:6]),
            ir.load_mem(val_p, c[6], deps=const_ops[:7]),
            ir.load_mem(n_nodes_s, c[1], deps=const_ops[:2]),
        ]

        # Hash constants
        hc_ops = []
        for i in range(6):
            hc_ops.append(ir.load_const(hc[i], HASH_STAGES[i][1]))
            hc_ops.append(ir.load_const(hs[i], HASH_STAGES[i][4]))

        # Broadcasts (VALU only - true SIMD benefit)
        bcast_deps = const_ops + ptr_ops + hc_ops
        bcast_ops = [
            ir.broadcast(v_0, c[0], deps=bcast_deps),
            ir.broadcast(v_1, c[1], deps=bcast_deps),
            ir.broadcast(v_2, c[2], deps=bcast_deps),
            ir.broadcast(v_nn, n_nodes_s, deps=[ptr_ops[3]]),
        ]
        bcast_ops.extend([ir.broadcast(v_hc[i], hc[i], deps=[hc_ops[i*2]]) for i in range(6)])
        bcast_ops.extend([ir.broadcast(v_hs[i], hs[i], deps=[hc_ops[i*2+1]]) for i in range(6)])

        # Tree cache load
        tc_load = ir.load_mem(tc[0], forest_p, deps=[ptr_ops[0]])
        tc_bcast = ir.broadcast(v_nv[0][0], tc[0], deps=[tc_load])

        init_complete = bcast_ops + [tc_bcast]

        # LOAD DATA (LOAD engine)
        load_ops = {}
        for g in range(n_groups):
            const_g = ir.load_const(st[g % 8], g * VLEN, deps=init_complete)
            addr_idx = ir.scalar_add(st[8 + g % 4], idx_p, st[g % 8], deps=[const_g])
            addr_val = ir.scalar_add(st[12 + g % 4], val_p, st[g % 8], deps=[const_g])
            load_ops[g] = (
                ir.vload(all_idx[g], st[8 + g % 4], deps=[addr_idx]),
                ir.vload(all_val[g], st[12 + g % 4], deps=[addr_val])
            )

        load_complete = [op for g in load_ops for op in load_ops[g]]

        # ROUNDS - simplified to just a few rounds for testing
        prev_round = load_complete

        for rnd in range(min(rounds, 4)):  # Limit for testing
            round_ops = {}

            for it in range(n_iters):
                base_g = it * UNROLL

                for i in range(UNROLL):
                    g = base_g + i
                    t = v_t[it][i]
                    nv = v_nv[it][i]

                    # Broadcast tree value (VALU)
                    nv_op = ir.broadcast(nv, tc[0], deps=prev_round)

                    # XOR (VALU only for efficiency)
                    xor_op = ir.vector_xor(all_val[g], all_val[g], nv, deps=[nv_op])

                    # Hash - 2 stages simplified (can expand)
                    hash_dep = xor_op
                    for hi in range(2):  # Just 2 hash stages for testing
                        op1, _, op2, op3, _ = HASH_STAGES[hi]
                        # Stage op1 and op3 - VALU or ALU
                        h1 = ir.arithmetic_flexible(t, all_val[g], v_hc[hi], op1, deps=[hash_dep])
                        h2 = ir.arithmetic_flexible(nv, all_val[g], v_hs[hi], op3, deps=[hash_dep])
                        # Stage op2 - VALU or ALU
                        h3 = ir.arithmetic_flexible(all_val[g], t, nv, op2, deps=[h1, h2])
                        hash_dep = h3

                    # Branch - all FLEXIBLE (scheduler picks VALU vs ALU)
                    b1 = ir.arithmetic_flexible(all_idx[g], all_idx[g], v_2, '*', deps=[hash_dep])
                    b2 = ir.arithmetic_flexible(t, all_val[g], v_1, '&', deps=[hash_dep])
                    b3 = ir.arithmetic_flexible(t, t, v_1, '+', deps=[b2])
                    b4 = ir.arithmetic_flexible(all_idx[g], all_idx[g], t, '+', deps=[b1, b3])
                    b5 = ir.arithmetic_flexible(t, all_idx[g], v_nn, '<', deps=[b4])
                    b6 = ir.arithmetic_flexible(all_idx[g], all_idx[g], t, '*', deps=[b5])

                    round_ops[(it, i)] = b6

            prev_round = list(round_ops.values())

        # STORE
        for g in range(n_groups):
            const_g = ir.load_const(st[g % 8], g * VLEN, deps=prev_round)
            addr_idx = ir.scalar_add(st[8 + g % 4], idx_p, st[g % 8], deps=[const_g])
            addr_val = ir.scalar_add(st[12 + g % 4], val_p, st[g % 8], deps=[const_g])
            ir.vstore(st[8 + g % 4], all_idx[g], deps=[addr_idx])
            ir.vstore(st[12 + g % 4], all_val[g], deps=[addr_val])

        print(f"Created {len(ir.ops)} operations")

        # Schedule with instruction selection
        print(f"Scheduling with CP-SAT (instruction selection)...")
        cycles, instrs = schedule_multi_path(ir, time_limit=120.0)

        self.instrs = [{"flow": [("pause",)]}] + instrs + [{"flow": [("pause",)]}]
        print(f"Result: {cycles} cycles, {len(self.instrs)} instructions")

        return cycles


def test_multipath_kernel():
    """Test the multi-path kernel."""
    import random
    from problem import Machine, Tree, Input, build_mem_image, reference_kernel2, N_CORES

    FOREST_HEIGHT = 10
    ROUNDS = 4  # Limited for testing
    BATCH_SIZE = 256
    BASELINE = 147734

    random.seed(123)
    tree = Tree.generate(FOREST_HEIGHT)
    n_nodes = len(tree.values)
    inp = Input.generate(tree, BATCH_SIZE, ROUNDS)
    mem = build_mem_image(tree, inp)

    print(f"Testing multi-path kernel")
    print(f"  forest_height={FOREST_HEIGHT}, rounds={ROUNDS}, batch_size={BATCH_SIZE}")
    print()

    builder = MultiPathInterleavedKernelBuilder()
    cycles = builder.build_kernel(FOREST_HEIGHT, n_nodes, BATCH_SIZE, ROUNDS)

    # Run and verify
    machine = Machine(mem, builder.instrs, builder.debug_info(), n_cores=N_CORES)
    machine.enable_pause = False
    machine.enable_debug = False
    machine.run()

    actual_cycles = machine.cycle
    print(f"\nActual execution: {actual_cycles} cycles")

    # Verify correctness
    ref_mem = list(reference_kernel2(mem))[-1]
    inp_values_p = ref_mem[6]
    inp_indices_p = ref_mem[5]

    values_correct = machine.mem[inp_values_p:inp_values_p+BATCH_SIZE] == ref_mem[inp_values_p:inp_values_p+BATCH_SIZE]
    indices_correct = machine.mem[inp_indices_p:inp_indices_p+BATCH_SIZE] == ref_mem[inp_indices_p:inp_indices_p+BATCH_SIZE]

    print(f"Correct: values={values_correct}, indices={indices_correct}")

    # Engine usage
    valu_count = 0
    alu_count = 0
    for op in builder.ir.ops:
        if op.chosen_impl is not None:
            impl = op.implementations[op.chosen_impl]
            if impl.engine == Engine.VALU:
                valu_count += 1
            elif impl.engine == Engine.ALU:
                alu_count += 1

    print(f"\nEngine selection:")
    print(f"  VALU: {valu_count} ops")
    print(f"  ALU: {alu_count} ops")

    return actual_cycles


if __name__ == "__main__":
    test_multipath_kernel()
