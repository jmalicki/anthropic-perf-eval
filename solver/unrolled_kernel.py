"""
Fully Unrolled Kernel - No loops, let the scheduler optimize everything.

Key insight: All sizes known at kernel creation time = complete unrolling.
The scheduler can then find globally optimal packing across ALL operations.
"""

from problem import DebugInfo, SLOT_LIMITS, VLEN, SCRATCH_SIZE, HASH_STAGES
from solver.reschedule import extract_ops, compute_dependencies, schedule_greedy, emit_scheduled


class UnrolledKernelBuilder:
    """Build a fully unrolled kernel with scheduler optimization."""

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
        """Build fully unrolled kernel."""
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
        # Temps per iteration to avoid WAW serialization
        # Each iteration needs ~4 temps for hash/branch
        v_t = [[self.vec(f"v_t_{it}_{i}") for i in range(4)] for it in range(n_iters)]
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

        # LOAD DATA
        for g in range(0, n_groups, 2):
            self.emit(load=[("const", st[0], g * VLEN), ("const", st[1], (g+1) * VLEN)])
            self.emit(alu=[("+", st[2], idx_p, st[0]), ("+", st[3], idx_p, st[1]), ("+", st[4], val_p, st[0]), ("+", st[5], val_p, st[1])])
            self.emit(load=[("vload", all_idx[g], st[2]), ("vload", all_idx[g+1], st[3])])
            self.emit(load=[("vload", all_val[g], st[4]), ("vload", all_val[g+1], st[5])])
        load_end = len(self.instrs)

        # FULLY UNROLLED ROUNDS - use per-iteration temps to enable parallelism
        for rnd in range(rounds):
            for it in range(n_iters):
                base_g = it * UNROLL
                v = v_nv[it]
                t = v_t[it]  # Per-iteration temps!
                if rnd == 0:
                    # Round 0: just broadcast tc[0] to all v_nv
                    for i in range(UNROLL):
                        self.emit(valu=[("vbroadcast", v[i], tc[0])])
                elif rnd == 1:
                    # Round 1: v_nv = tc[2] + (idx == 1) * td_1
                    for i in range(UNROLL):
                        self.emit(valu=[("==", t[0], all_idx[base_g+i], v_1)])
                        self.emit(valu=[("*", t[0], v_td_1, t[0])])
                        self.emit(valu=[("+", v[i], v_tc2, t[0])])
                elif rnd == 2:
                    # Round 2: more complex - select based on idx == 4, 5, 6
                    for i in range(UNROLL):
                        self.emit(valu=[("==", t[0], all_idx[base_g+i], v_const[4]),
                                        ("==", t[1], all_idx[base_g+i], v_const[5]),
                                        ("==", t[2], all_idx[base_g+i], v_const[6])])
                        self.emit(valu=[("*", t[0], v_td_4, t[0]),
                                        ("*", t[1], v_td_5, t[1]),
                                        ("*", t[2], v_td_6, t[2])])
                        self.emit(valu=[("+", t[0], t[0], t[1])])
                        self.emit(valu=[("+", t[0], t[0], t[2])])
                        self.emit(valu=[("+", v[i], v_tc3, t[0])])
                else:
                    # Gather: Load each element separately
                    for vi in range(VLEN):
                        for i in range(0, UNROLL, 2):
                            self.emit(alu=[
                                ("+", st[0], forest_p, all_idx[base_g+i] + vi),
                                ("+", st[1], forest_p, all_idx[base_g+i+1] + vi)
                            ])
                            self.emit(load=[
                                ("load", v[i] + vi, st[0]),
                                ("load", v[i+1] + vi, st[1])
                            ])
                # XOR all values with v_nv
                for i in range(UNROLL):
                    self.emit(valu=[("^", all_val[base_g+i], all_val[base_g+i], v[i])])

                # Hash: 6 stages - use per-iteration temps
                for hi in range(6):
                    op1, _, op2, op3, _ = HASH_STAGES[hi]
                    for i in range(UNROLL):
                        self.emit(valu=[(op1, t[i%2], all_val[base_g+i], v_hc[hi]),
                                        (op3, v[i], all_val[base_g+i], v_hs[hi])])
                        self.emit(valu=[(op2, all_val[base_g+i], t[i%2], v[i])])

                # Branch computation - use per-iteration temps
                for i in range(UNROLL):
                    self.emit(valu=[("*", all_idx[base_g+i], all_idx[base_g+i], v_2)])
                    self.emit(valu=[("&", t[0], all_val[base_g+i], v_1)])
                    self.emit(valu=[("+", t[0], t[0], v_1)])
                    self.emit(valu=[("+", all_idx[base_g+i], all_idx[base_g+i], t[0])])
                    self.emit(valu=[("<", t[1], all_idx[base_g+i], v_nn)])
                    self.emit(valu=[("*", all_idx[base_g+i], all_idx[base_g+i], t[1])])
        rounds_end = len(self.instrs)

        # STORE DATA
        for g in range(0, n_groups, 2):
            self.emit(load=[("const", st[0], g * VLEN), ("const", st[1], (g+1) * VLEN)])
            self.emit(alu=[("+", st[2], idx_p, st[0]), ("+", st[3], idx_p, st[1]), ("+", st[4], val_p, st[0]), ("+", st[5], val_p, st[1])])
            self.emit(store=[("vstore", st[2], all_idx[g]), ("vstore", st[3], all_idx[g+1])])
            self.emit(store=[("vstore", st[4], all_val[g]), ("vstore", st[5], all_val[g+1])])
        store_end = len(self.instrs)

        print(f"Before scheduling: {store_end} instructions")

        # RESCHEDULE - try parallel CP-SAT, then single CP-SAT, then greedy
        SCHEDULER = "parallel"  # "parallel", "cp", "greedy", or "none"

        if SCHEDULER == "parallel":
            try:
                from solver.parallel_scheduler import schedule_parallel
                ops = extract_ops(self.instrs)
                op_by_id = compute_dependencies(ops)
                print(f"Running parallel scheduler on {len(ops)} ops...")
                num_cycles, scheduled = schedule_parallel(ops, op_by_id, chunk_size=2000, time_limit=60.0)
                self.instrs = [{"flow": [("pause",)]}] + scheduled + [{"flow": [("pause",)]}]
                print(f"Parallel scheduling: {len(self.instrs)} instructions ({num_cycles} cycles)")
            except Exception as e:
                print(f"Parallel scheduler failed ({e}), falling back to greedy")
                SCHEDULER = "greedy"

        if SCHEDULER == "cp":
            try:
                from solver.cp_scheduler import schedule_cp, emit_scheduled_cp
                ops = extract_ops(self.instrs)
                op_by_id = compute_dependencies(ops)
                print(f"Running CP-SAT scheduler on {len(ops)} ops...")
                result = schedule_cp(ops, op_by_id, time_limit=120.0)
                if result.success:
                    scheduled = emit_scheduled_cp(ops, result.cycles)
                    self.instrs = [{"flow": [("pause",)]}] + scheduled + [{"flow": [("pause",)]}]
                    print(f"CP-SAT scheduling: {len(self.instrs)} instructions ({result.cycles} cycles, status={result.status})")
                else:
                    print(f"CP-SAT failed ({result.status}), falling back to greedy")
                    SCHEDULER = "greedy"
            except ImportError:
                print("CP-SAT not available, falling back to greedy")
                SCHEDULER = "greedy"

        if SCHEDULER == "greedy":
            ops = extract_ops(self.instrs)
            op_by_id = compute_dependencies(ops)
            num_cycles = schedule_greedy(ops, op_by_id)
            scheduled = emit_scheduled(ops, num_cycles)
            self.instrs = [{"flow": [("pause",)]}] + scheduled + [{"flow": [("pause",)]}]
            print(f"Greedy scheduling: {len(self.instrs)} instructions ({num_cycles} cycles)")

        if SCHEDULER == "none":
            self.instrs = [{"flow": [("pause",)]}] + self.instrs + [{"flow": [("pause",)]}]
            print(f"Scheduling disabled: {len(self.instrs)} instructions")
