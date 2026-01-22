"""
Interleaved Unrolled Kernel - Emits operations in interleaved order to enable pipelining.

Key insight: Emit operations from different iterations interleaved so the scheduler
can overlap independent operations. Uses per-iteration temps to avoid conflicts.
"""

from problem import DebugInfo, SLOT_LIMITS, VLEN, SCRATCH_SIZE, HASH_STAGES
from solver.reschedule import extract_ops, compute_dependencies, schedule_greedy, emit_scheduled


class InterleavedUnrolledKernelBuilder:
    """Build a fully unrolled kernel with interleaved emission for better pipelining."""

    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0

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

    def emit(self, **engines):
        instr = {k: v for k, v in engines.items() if v}
        if instr:
            self.instrs.append(instr)

    def build_kernel(
        self,
        forest_height: int,
        n_nodes: int,
        batch_size: int,
        rounds: int
    ):
        """Build fully unrolled kernel with interleaved emission."""
        n_groups = batch_size // VLEN
        UNROLL = 8
        n_iters = n_groups // UNROLL

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
        td_1 = self.alloc("td_1")
        td_4 = self.alloc("td_4")
        td_5 = self.alloc("td_5")
        td_6 = self.alloc("td_6")
        all_idx = [self.vec(f"all_idx_{g}") for g in range(n_groups)]
        all_val = [self.vec(f"all_val_{g}") for g in range(n_groups)]
        v_nv = [[self.vec(f"v_nv_{it}_{i}") for i in range(UNROLL)] for it in range(n_iters)]

        # Per-group temps to enable parallelism within iterations
        # Use 1 temp per group, reuse across phases (hash and branch don't overlap)
        # Total: 4 iters × 8 groups = 32 temp vectors = 256 words
        v_t = [[self.vec(f"v_t_{it}_{i}") for i in range(UNROLL)] for it in range(n_iters)]
        # v_t[it][i] is the temp for group i in iteration it

        v_0 = self.vec("v_0")
        v_1 = self.vec("v_1")
        v_2 = self.vec("v_2")
        v_nn = self.vec("v_nn")
        v_const = [self.vec(f"v_const{i}") for i in range(8)]
        v_td_1 = self.vec("v_td_1")
        v_td_4 = self.vec("v_td_4")
        v_td_5 = self.vec("v_td_5")
        v_td_6 = self.vec("v_td_6")
        v_tc2 = self.vec("v_tc2")
        v_tc3 = self.vec("v_tc3")
        v_hc = [self.vec(f"v_hc{i}") for i in range(6)]
        v_hs = [self.vec(f"v_hs{i}") for i in range(6)]

        print(f"Scratch usage: {self.scratch_ptr} / {SCRATCH_SIZE}")

        # INIT
        for i in range(16):
            self.emit(load=[("const", c[i], i)])
        self.emit(load=[("const", st[0], 4), ("const", st[1], 5)])
        self.emit(load=[("load", forest_p, st[0]), ("load", idx_p, st[1])])
        self.emit(load=[("const", st[0], 6), ("const", st[1], 1)])
        self.emit(load=[("load", val_p, st[0]), ("load", n_nodes_s, st[1])])
        for i in range(6):
            self.emit(load=[("const", hc[i], HASH_STAGES[i][1]), ("const", hs[i], HASH_STAGES[i][4])])
        self.emit(valu=[("vbroadcast", v_0, c[0]), ("vbroadcast", v_1, c[1]), ("vbroadcast", v_2, c[2]), ("vbroadcast", v_nn, n_nodes_s)])
        for i in range(0, 8, 2):
            self.emit(valu=[("vbroadcast", v_const[i], c[i]), ("vbroadcast", v_const[i+1], c[i+1])])
        for i in range(0, 6, 2):
            self.emit(valu=[("vbroadcast", v_hc[i], hc[i]), ("vbroadcast", v_hc[i+1], hc[i+1]), ("vbroadcast", v_hs[i], hs[i]), ("vbroadcast", v_hs[i+1], hs[i+1])])
        for i in range(7):
            self.emit(alu=[("+", st[0], forest_p, c[i])])
            self.emit(load=[("load", tc[i], st[0])])
        self.emit(alu=[("-", td_1, tc[1], tc[2]), ("-", td_4, tc[4], tc[3]), ("-", td_5, tc[5], tc[3]), ("-", td_6, tc[6], tc[3])])
        self.emit(valu=[("vbroadcast", v_tc2, tc[2]), ("vbroadcast", v_td_1, td_1), ("vbroadcast", v_tc3, tc[3]), ("vbroadcast", v_td_4, td_4), ("vbroadcast", v_td_5, td_5), ("vbroadcast", v_td_6, td_6)])
        init_end = len(self.instrs)

        # LOAD DATA - interleaved across groups
        for g in range(0, n_groups, 2):
            self.emit(load=[("const", st[0], g * VLEN), ("const", st[1], (g+1) * VLEN)])
            self.emit(alu=[("+", st[2], idx_p, st[0]), ("+", st[3], idx_p, st[1]), ("+", st[4], val_p, st[0]), ("+", st[5], val_p, st[1])])
            self.emit(load=[("vload", all_idx[g], st[2]), ("vload", all_idx[g+1], st[3])])
            self.emit(load=[("vload", all_val[g], st[4]), ("vload", all_val[g+1], st[5])])
        load_end = len(self.instrs)

        # FULLY UNROLLED ROUNDS - INTERLEAVED EMISSION
        for rnd in range(rounds):
            # PHASE 1: Compute or Gather node values into v_nv
            if rnd == 0:
                # Round 0: just broadcast tc[0]
                for it in range(n_iters):
                    for i in range(UNROLL):
                        self.emit(valu=[("vbroadcast", v_nv[it][i], tc[0])])

            elif rnd == 1:
                # Round 1: v_nv = tc[2] + (idx == 1) * td_1
                # Use per-group temps: v_t[it][i] for group i
                for it in range(n_iters):
                    base_g = it * UNROLL
                    for i in range(UNROLL):
                        t = v_t[it][i]  # Per-group temp
                        self.emit(valu=[("==", t, all_idx[base_g+i], v_1)])
                        self.emit(valu=[("*", t, v_td_1, t)])
                        self.emit(valu=[("+", v_nv[it][i], v_tc2, t)])

            elif rnd == 2:
                # Round 2: v_nv = tc[3] + select based on idx == 4, 5, 6
                # Use per-group temp and v_nv as additional temps
                # t0 = v_t[it][i], t1 = v_nv[it][(i+1)%UNROLL], t2 = v_nv[it][(i+2)%UNROLL]
                # But this creates conflicts... instead, do sequential accumulation
                for it in range(n_iters):
                    base_g = it * UNROLL
                    for i in range(UNROLL):
                        t = v_t[it][i]
                        # Sequential: compare, multiply, accumulate step by step
                        self.emit(valu=[("==", t, all_idx[base_g+i], v_const[4])])
                        self.emit(valu=[("*", t, v_td_4, t)])
                        self.emit(valu=[("+", v_nv[it][i], v_tc3, t)])
                        # Now add contribution from idx==5
                        self.emit(valu=[("==", t, all_idx[base_g+i], v_const[5])])
                        self.emit(valu=[("*", t, v_td_5, t)])
                        self.emit(valu=[("+", v_nv[it][i], v_nv[it][i], t)])
                        # Now add contribution from idx==6
                        self.emit(valu=[("==", t, all_idx[base_g+i], v_const[6])])
                        self.emit(valu=[("*", t, v_td_6, t)])
                        self.emit(valu=[("+", v_nv[it][i], v_nv[it][i], t)])

            else:
                # Gather from forest - use per-iteration st slots for interleaving
                for vi in range(VLEN):
                    for i in range(0, UNROLL, 2):
                        for it in range(n_iters):
                            base_g = it * UNROLL
                            st_a, st_b = st[2*it], st[2*it+1]
                            self.emit(alu=[
                                ("+", st_a, forest_p, all_idx[base_g+i] + vi),
                                ("+", st_b, forest_p, all_idx[base_g+i+1] + vi)
                            ])
                            self.emit(load=[
                                ("load", v_nv[it][i] + vi, st_a),
                                ("load", v_nv[it][i+1] + vi, st_b)
                            ])

            # PHASE 2: XOR all values with v_nv (ALL rounds)
            for it in range(n_iters):
                base_g = it * UNROLL
                for i in range(UNROLL):
                    self.emit(valu=[("^", all_val[base_g+i], all_val[base_g+i], v_nv[it][i])])

            # PHASE 3: Hash - 6 stages (ALL rounds)
            # Use per-group temps: v_t[it][i] for hash temp
            for hi in range(6):
                op1, _, op2, op3, _ = HASH_STAGES[hi]
                for it in range(n_iters):
                    base_g = it * UNROLL
                    for i in range(UNROLL):
                        t = v_t[it][i]  # Per-group temp for hash
                        self.emit(valu=[(op1, t, all_val[base_g+i], v_hc[hi]),
                                        (op3, v_nv[it][i], all_val[base_g+i], v_hs[hi])])
                        self.emit(valu=[(op2, all_val[base_g+i], t, v_nv[it][i])])

            # PHASE 4: Branch computation (ALL rounds)
            # Use per-group temp v_t[it][i] for bit and reuse for in_bounds
            # (sequential ops so no conflict)
            for it in range(n_iters):
                base_g = it * UNROLL
                for i in range(UNROLL):
                    t = v_t[it][i]  # Per-group temp
                    self.emit(valu=[("*", all_idx[base_g+i], all_idx[base_g+i], v_2)])
                    self.emit(valu=[("&", t, all_val[base_g+i], v_1)])
                    self.emit(valu=[("+", t, t, v_1)])
                    self.emit(valu=[("+", all_idx[base_g+i], all_idx[base_g+i], t)])
                    # Reuse t for in_bounds (previous value no longer needed)
                    self.emit(valu=[("<", t, all_idx[base_g+i], v_nn)])
                    self.emit(valu=[("*", all_idx[base_g+i], all_idx[base_g+i], t)])

        rounds_end = len(self.instrs)

        # STORE DATA
        for g in range(0, n_groups, 2):
            self.emit(load=[("const", st[0], g * VLEN), ("const", st[1], (g+1) * VLEN)])
            self.emit(alu=[("+", st[2], idx_p, st[0]), ("+", st[3], idx_p, st[1]), ("+", st[4], val_p, st[0]), ("+", st[5], val_p, st[1])])
            self.emit(store=[("vstore", st[2], all_idx[g]), ("vstore", st[3], all_idx[g+1])])
            self.emit(store=[("vstore", st[4], all_val[g]), ("vstore", st[5], all_val[g+1])])
        store_end = len(self.instrs)

        print(f"Before scheduling: {store_end} instructions")

        # RESCHEDULE - try chunked CP-SAT, fall back to greedy
        ops = extract_ops(self.instrs)
        op_by_id = compute_dependencies(ops)
        print(f"Scheduling {len(ops)} ops...")

        SCHEDULER = "chunked"  # "cp", "greedy", or "chunked"

        if SCHEDULER == "chunked" and len(ops) > 2000:
            try:
                from solver.round_chunked_scheduler import schedule_round_chunked
                num_cycles, scheduled = schedule_round_chunked(
                    ops, op_by_id, rounds, n_iters, UNROLL, time_limit=30.0
                )
                self.instrs = [{"flow": [("pause",)]}] + scheduled + [{"flow": [("pause",)]}]
                print(f"Round-chunked scheduling: {len(self.instrs)} instructions ({num_cycles} cycles)")
            except ImportError:
                print("Round-chunked scheduler not available, falling back to greedy")
                SCHEDULER = "greedy"

        if SCHEDULER == "greedy" or (SCHEDULER == "chunked" and len(ops) <= 2000):
            num_cycles = schedule_greedy(ops, op_by_id)
            scheduled = emit_scheduled(ops, num_cycles)
            self.instrs = [{"flow": [("pause",)]}] + scheduled + [{"flow": [("pause",)]}]
            print(f"Greedy scheduling: {len(self.instrs)} instructions ({num_cycles} cycles)")
