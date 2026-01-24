"""
Interleaved Unrolled Kernel - Emits operations in interleaved order to enable pipelining.

Key insight: Emit operations from different iterations interleaved so the scheduler
can overlap independent operations. Uses per-iteration temps to avoid conflicts.
"""

import math

from problem import DebugInfo, SLOT_LIMITS, VLEN, SCRATCH_SIZE, HASH_STAGES
from solver.reschedule import extract_ops, compute_dependencies, schedule_greedy, schedule_greedy_multipass, emit_scheduled


class InterleavedUnrolledKernelBuilder:
    """Build a fully unrolled kernel with interleaved emission for better pipelining."""

    def __init__(self, enable_debug: bool = False):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.enable_debug = enable_debug
        self.debug_ops = []  # Pending debug operations to attach to next emit()

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
            # Attach any pending debug ops to this instruction
            if self.debug_ops:
                instr["debug"] = list(self.debug_ops)
                self.debug_ops = []
            self.instrs.append(instr)

    def emit_compare(self, loc: int, key: tuple):
        """Emit scalar comparison with reference value.

        Args:
            loc: Scratch address of value to compare
            key: Trace key tuple (round, batch_i, phase_name, ...)
        """
        if self.enable_debug:
            self.debug_ops.append(("compare", loc, key))

    def emit_vcompare(self, loc: int, keys: list):
        """Emit vector comparison with reference values.

        Args:
            loc: Base scratch address of vector to compare
            keys: List of trace key tuples, one per VLEN element
        """
        if self.enable_debug:
            self.debug_ops.append(("vcompare", loc, keys))

    def emit_debug(self):
        """Emit a debug-only instruction immediately.

        This ensures debug comparisons are processed at the correct point
        in the dependency graph (right after the operation they trace),
        rather than being attached to the next emit() which may contain
        operations that write to the same locations.

        Debug-only instructions don't count toward cycle count.
        """
        if self.debug_ops:
            self.instrs.append({"debug": list(self.debug_ops)})
            self.debug_ops = []

    def build_kernel(
        self,
        forest_height: int,
        n_nodes: int,
        batch_size: int,
        rounds: int,
        full_solve_time: float = 3600.0,
        require_optimal: bool = True,
        enable_valu_shift: bool = True,
        valu_shift_time: float = None,
        use_greedy: bool = False,
        enable_round3_valu_select: bool = True,
        extend_valu_selection: int = 0,
    ):
        """Build fully unrolled kernel with interleaved emission.

        This kernel uses tree_depth-based dispatch: depths 0-3 use VALU/delta methods
        (reusing preloaded tree values), depths 4+ use scattered loads. The tree_depth
        wraps at tree_height, so post-wrap rounds automatically reuse the same efficient
        code paths. This eliminates ~178 cycles (8% improvement) by avoiding redundant
        scattered loads after all elements wrap to the tree root.

        Args:
            full_solve_time: Time limit for CP-SAT full solve in seconds (default: 3600 = 1 hour)
            require_optimal: Require OPTIMAL status from CP-SAT (default: True)
            enable_valu_shift: Enable VALU→ALU shifting optimization (default: True)
            valu_shift_time: Time per VALU→ALU shift iteration (default: same as full_solve_time)
            use_greedy: Use fast greedy scheduler instead of CP-SAT (default: False)
            enable_round3_valu_select: Use VALU selection for depth 3 tree access instead of
                scattered loads. Preloads tree[7..14] and uses compare+multiply_add to select.
                Applies to both initial traversal AND post-wrap rounds. (default: True)
            extend_valu_selection: EXPERIMENTAL - Extend VALU selection pattern to depths 4-5.
                Testing shows this is COUNTERPRODUCTIVE (increases cycles) because VALU ops
                become more expensive than scattered loads for larger index ranges.
                NOT RECOMMENDED. (default: 0 = disabled)
        """
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
        # Round 3 optimization: preload tree[7..14] (8 values) for VALU selection
        tc_r3 = [self.alloc(f"tc_r3_{i}") for i in range(8)] if enable_round3_valu_select else None
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
        # Round 3 optimization: vectors for tree[7..14] and comparison constants
        v_tc_r3 = [self.vec(f"v_tc_r3_{i}") for i in range(8)] if enable_round3_valu_select else None
        v_const_r3 = [self.vec(f"v_const_r3_{i}") for i in range(8)] if enable_round3_valu_select else None

        # Extended VALU selection: preload tree values for rounds 4-5
        # Round 4: indices 15-30 (16 values), Round 5: indices 31-62 (32 values)
        v_tc_r4 = None
        v_const_r4 = None
        v_tc_r5 = None
        v_const_r5 = None
        if extend_valu_selection >= 4:
            # Round 4: 16 tree values (indices 15-30)
            v_tc_r4 = [self.vec(f"v_tc_r4_{i}") for i in range(16)]
            v_const_r4 = [self.vec(f"v_const_r4_{i}") for i in range(16)]
        if extend_valu_selection >= 5:
            # Round 5: 32 tree values (indices 31-62)
            v_tc_r5 = [self.vec(f"v_tc_r5_{i}") for i in range(32)]
            v_const_r5 = [self.vec(f"v_const_r5_{i}") for i in range(32)]

        # FMA optimization: (a + const) + (a << shift) = a * (1 + 2^shift) + const
        # Applicable to stages where op1='+', op2='+', op3='<<'
        hash_fma_mult = [None] * 6  # multiplier or None if not FMA-eligible
        v_hash_mult = {}  # stage_idx -> vector address
        for i, (op1, const, op2, op3, shift) in enumerate(HASH_STAGES):
            if op1 == '+' and op2 == '+' and op3 == '<<':
                hash_fma_mult[i] = 1 + (1 << shift)

        # Stage 5 XOR optimization:
        # Stage 5: (a ^ C) ^ (a >> 16) — we simplify to a ^ (a >> 16)
        # Key: val_orig = val_opt ^ C is maintained across rounds
        # 1. Pre-XOR tc[1-6] with C (not tc[0])
        # 2. Pre-XOR gathered nodes with C (rounds 3+)
        # 3. Simplify stage 5 to just a ^ (a >> 16)
        # 4. Flip branch direction (C's LSB = 1)
        # 5. When wrapping (idx → 0), XOR val with C
        # 6. At the end, XOR stored val with C
        STAGE5_XOR_CONST = 0xB55A4F09
        stage5_xor_s = self.alloc("stage5_xor")
        v_stage5_xor = self.vec("v_stage5_xor")

        if self.scratch_ptr > SCRATCH_SIZE:
            print(f"WARNING: Scratch usage {self.scratch_ptr} exceeds budget {SCRATCH_SIZE}!")
        else:
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

        # Initialize FMA multiplier vectors for eligible hash stages
        for i in range(6):
            if hash_fma_mult[i] is not None:
                mult_s = self.alloc(f"hash_mult_{i}")
                v_mult = self.vec(f"v_hash_mult_{i}")
                v_hash_mult[i] = v_mult
                self.emit(load=[("const", mult_s, hash_fma_mult[i])])
                self.emit(valu=[("vbroadcast", v_mult, mult_s)])

        # Load Stage 5 XOR constant
        self.emit(load=[("const", stage5_xor_s, STAGE5_XOR_CONST)])
        self.emit(valu=[("vbroadcast", v_stage5_xor, stage5_xor_s)])

        for i in range(7):
            self.emit(alu=[("+", st[0], forest_p, c[i])])
            self.emit(load=[("load", tc[i], st[0])])

        # Round 3 optimization: load tree[7..14] and broadcast to vectors
        # These 8 values will be used for VALU selection instead of scattered loads
        c_r3 = None
        if enable_round3_valu_select:
            c_r3 = [self.alloc(f"c_r3_{i}") for i in range(8)]  # Constants 7-14
            for i in range(8):
                self.emit(load=[("const", c_r3[i], 7 + i)])  # Load constant 7, 8, ..., 14
            for i in range(8):
                self.emit(alu=[("+", st[0], forest_p, c_r3[i])])
                self.emit(load=[("load", tc_r3[i], st[0])])
            # Broadcast tree values and comparison constants to vectors
            for i in range(0, 8, 2):
                self.emit(valu=[("vbroadcast", v_tc_r3[i], tc_r3[i]),
                               ("vbroadcast", v_tc_r3[i+1], tc_r3[i+1])])
            for i in range(0, 8, 2):
                self.emit(valu=[("vbroadcast", v_const_r3[i], c_r3[i]),
                               ("vbroadcast", v_const_r3[i+1], c_r3[i+1])])

        # Extended VALU selection: load tree values for rounds 4-5
        # Round 4: tree[15..30] (16 values), Round 5: tree[31..62] (32 values)
        tc_r4 = None
        c_r4 = None
        tc_r5 = None
        c_r5 = None
        if extend_valu_selection >= 4:
            tc_r4 = [self.alloc(f"tc_r4_{i}") for i in range(16)]
            c_r4 = [self.alloc(f"c_r4_{i}") for i in range(16)]
            for i in range(16):
                self.emit(load=[("const", c_r4[i], 15 + i)])  # Constants 15-30
            for i in range(16):
                self.emit(alu=[("+", st[0], forest_p, c_r4[i])])
                self.emit(load=[("load", tc_r4[i], st[0])])
            # Broadcast tree values and comparison constants to vectors
            for i in range(0, 16, 2):
                self.emit(valu=[("vbroadcast", v_tc_r4[i], tc_r4[i]),
                               ("vbroadcast", v_tc_r4[i+1], tc_r4[i+1])])
            for i in range(0, 16, 2):
                self.emit(valu=[("vbroadcast", v_const_r4[i], c_r4[i]),
                               ("vbroadcast", v_const_r4[i+1], c_r4[i+1])])

        if extend_valu_selection >= 5:
            tc_r5 = [self.alloc(f"tc_r5_{i}") for i in range(32)]
            c_r5 = [self.alloc(f"c_r5_{i}") for i in range(32)]
            for i in range(32):
                self.emit(load=[("const", c_r5[i], 31 + i)])  # Constants 31-62
            for i in range(32):
                self.emit(alu=[("+", st[0], forest_p, c_r5[i])])
                self.emit(load=[("load", tc_r5[i], st[0])])
            # Broadcast tree values and comparison constants to vectors
            for i in range(0, 32, 2):
                self.emit(valu=[("vbroadcast", v_tc_r5[i], tc_r5[i]),
                               ("vbroadcast", v_tc_r5[i+1], tc_r5[i+1])])
            for i in range(0, 32, 2):
                self.emit(valu=[("vbroadcast", v_const_r5[i], c_r5[i]),
                               ("vbroadcast", v_const_r5[i+1], c_r5[i+1])])

        # Note: We do NOT pre-XOR tc[1-6] here because the delta computation
        # (td_1 = tc[1] - tc[2]) doesn't preserve XOR. Instead, we XOR v_nv
        # after computing it for depths 1-2.

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

        # Tree traversal depth tracking
        # tree_height = forest_height + 1 (rounds needed to traverse full tree)
        # After tree_height rounds, all elements wrap to idx=0 and restart from root
        # This is because at the maximum tree depth, all child indices exceed n_nodes
        tree_height = forest_height + 1

        # FULLY UNROLLED ROUNDS - INTERLEAVED EMISSION
        # Use tree_depth to dispatch based on position in tree, not absolute round
        tree_depth = 0
        for rnd in range(rounds):
            # PHASE 1: Compute or Gather node values into v_nv
            # Dispatch based on tree_depth (0 = root, 1 = depth 1, etc.)
            # This handles both initial traversal AND post-wrap traversal

            if tree_depth == 0:
                # Depth 0: broadcast tc[0] (root node)
                for it in range(n_iters):
                    for i in range(UNROLL):
                        self.emit(valu=[("vbroadcast", v_nv[it][i], tc[0])])
                # Stage 5 opt: XOR v_nv with C (same as other depths)
                for it in range(n_iters):
                    for i in range(UNROLL):
                        self.emit(valu=[("^", v_nv[it][i], v_nv[it][i], v_stage5_xor)])
                # DEBUG: emit node_val comparison for depth 0
                for it in range(n_iters):
                    base_g = it * UNROLL
                    for i in range(UNROLL):
                        g = base_g + i
                        keys = [(rnd, g * VLEN + vi, "node_val_xored") for vi in range(VLEN)]
                        self.emit_vcompare(v_nv[it][i], keys)
                self.emit_debug()

            elif tree_depth == 1:
                # Depth 1: v_nv = tc[2] + (idx == 1) * td_1
                # idx can be 1 or 2, use delta computation
                for it in range(n_iters):
                    base_g = it * UNROLL
                    for i in range(UNROLL):
                        t = v_t[it][i]
                        self.emit(valu=[("==", t, all_idx[base_g+i], v_1)])
                        self.emit(valu=[("*", t, v_td_1, t)])
                        self.emit(valu=[("+", v_nv[it][i], v_tc2, t)])
                # Stage 5 opt: XOR v_nv with C
                for it in range(n_iters):
                    for i in range(UNROLL):
                        self.emit(valu=[("^", v_nv[it][i], v_nv[it][i], v_stage5_xor)])
                # DEBUG: emit node_val comparison for depth 1
                for it in range(n_iters):
                    base_g = it * UNROLL
                    for i in range(UNROLL):
                        g = base_g + i
                        keys = [(rnd, g * VLEN + vi, "node_val_xored") for vi in range(VLEN)]
                        self.emit_vcompare(v_nv[it][i], keys)
                self.emit_debug()

            elif tree_depth == 2:
                # Depth 2: v_nv = tc[3] + delta selection for idx == 3, 4, 5, 6
                for it in range(n_iters):
                    base_g = it * UNROLL
                    for i in range(UNROLL):
                        t = v_t[it][i]
                        self.emit(valu=[("==", t, all_idx[base_g+i], v_const[4])])
                        self.emit(valu=[("*", t, v_td_4, t)])
                        self.emit(valu=[("+", v_nv[it][i], v_tc3, t)])
                        self.emit(valu=[("==", t, all_idx[base_g+i], v_const[5])])
                        self.emit(valu=[("*", t, v_td_5, t)])
                        self.emit(valu=[("+", v_nv[it][i], v_nv[it][i], t)])
                        self.emit(valu=[("==", t, all_idx[base_g+i], v_const[6])])
                        self.emit(valu=[("*", t, v_td_6, t)])
                        self.emit(valu=[("+", v_nv[it][i], v_nv[it][i], t)])
                # Stage 5 opt: XOR v_nv with C
                for it in range(n_iters):
                    for i in range(UNROLL):
                        self.emit(valu=[("^", v_nv[it][i], v_nv[it][i], v_stage5_xor)])

            elif tree_depth == 3 and enable_round3_valu_select:
                # Depth 3: VALU selection from preloaded tree[7..14]
                for it in range(n_iters):
                    base_g = it * UNROLL
                    for i in range(UNROLL):
                        t = v_t[it][i]
                        self.emit(valu=[("*", v_nv[it][i], v_0, v_0)])
                        for ti in range(8):
                            self.emit(valu=[("==", t, all_idx[base_g+i], v_const_r3[ti])])
                            self.emit(valu=[("multiply_add", v_nv[it][i], t, v_tc_r3[ti], v_nv[it][i])])
                # Stage 5 opt: XOR v_nv with C
                for it in range(n_iters):
                    for i in range(UNROLL):
                        self.emit(valu=[("^", v_nv[it][i], v_nv[it][i], v_stage5_xor)])

            elif tree_depth == 4 and extend_valu_selection >= 4:
                # Depth 4: VALU selection from preloaded tree[15..30]
                for it in range(n_iters):
                    base_g = it * UNROLL
                    for i in range(UNROLL):
                        t = v_t[it][i]
                        self.emit(valu=[("*", v_nv[it][i], v_0, v_0)])
                        for ti in range(16):
                            self.emit(valu=[("==", t, all_idx[base_g+i], v_const_r4[ti])])
                            self.emit(valu=[("multiply_add", v_nv[it][i], t, v_tc_r4[ti], v_nv[it][i])])
                # Stage 5 opt: XOR v_nv with C
                for it in range(n_iters):
                    for i in range(UNROLL):
                        self.emit(valu=[("^", v_nv[it][i], v_nv[it][i], v_stage5_xor)])

            elif tree_depth == 5 and extend_valu_selection >= 5:
                # Depth 5: VALU selection from preloaded tree[31..62]
                for it in range(n_iters):
                    base_g = it * UNROLL
                    for i in range(UNROLL):
                        t = v_t[it][i]
                        self.emit(valu=[("*", v_nv[it][i], v_0, v_0)])
                        for ti in range(32):
                            self.emit(valu=[("==", t, all_idx[base_g+i], v_const_r5[ti])])
                            self.emit(valu=[("multiply_add", v_nv[it][i], t, v_tc_r5[ti], v_nv[it][i])])
                # Stage 5 opt: XOR v_nv with C
                for it in range(n_iters):
                    for i in range(UNROLL):
                        self.emit(valu=[("^", v_nv[it][i], v_nv[it][i], v_stage5_xor)])

            else:
                # Gather from forest - scattered loads for tree_depth >= 4 (or >= 3 if round3 disabled)
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
                # Stage 5 optimization: XOR gathered values with C
                for it in range(n_iters):
                    base_g = it * UNROLL
                    for i in range(UNROLL):
                        self.emit(valu=[("^", v_nv[it][i], v_nv[it][i], v_stage5_xor)])

            # Increment tree_depth, wrapping at tree_height (post-wrap restarts from root)
            tree_depth += 1
            if tree_depth >= tree_height:
                tree_depth = 0

            # PHASE 2: XOR all values with v_nv (ALL rounds)
            for it in range(n_iters):
                base_g = it * UNROLL
                for i in range(UNROLL):
                    self.emit(valu=[("^", all_val[base_g+i], all_val[base_g+i], v_nv[it][i])])
            # DEBUG: emit val_after_xor comparison
            for g in range(n_groups):
                keys = [(rnd, g * VLEN + vi, "val_after_xor") for vi in range(VLEN)]
                self.emit_vcompare(all_val[g], keys)
            self.emit_debug()

            # PHASE 3: Hash - 6 stages (ALL rounds)
            # Use per-group temps: v_t[it][i] for hash temp
            # FMA optimization: (a + const) + (a << shift) = a * (1 + 2^shift) + const
            # Stage 5 optimization: simplified to a ^ (a >> 16)
            for hi in range(6):
                op1, _, op2, op3, _ = HASH_STAGES[hi]
                for it in range(n_iters):
                    base_g = it * UNROLL
                    for i in range(UNROLL):
                        if hash_fma_mult[hi] is not None:
                            # FMA: val = val * mult + const (single multiply_add)
                            self.emit(valu=[("multiply_add", all_val[base_g+i],
                                            all_val[base_g+i], v_hash_mult[hi], v_hc[hi])])
                        elif hi == 5:
                            # Stage 5 optimized: just a ^ (a >> 16)
                            t = v_t[it][i]
                            self.emit(valu=[(">>", t, all_val[base_g+i], v_hs[hi])])
                            self.emit(valu=[("^", all_val[base_g+i], all_val[base_g+i], t)])
                        else:
                            # Original: 2 ops per stage
                            t = v_t[it][i]  # Per-group temp for hash
                            self.emit(valu=[(op1, t, all_val[base_g+i], v_hc[hi]),
                                            (op3, v_nv[it][i], all_val[base_g+i], v_hs[hi])])
                            self.emit(valu=[(op2, all_val[base_g+i], t, v_nv[it][i])])
                # DEBUG: emit hash_stage comparison after each stage
                for g in range(n_groups):
                    keys = [(rnd, g * VLEN + vi, "hash_stage", hi) for vi in range(VLEN)]
                    self.emit_vcompare(all_val[g], keys)
                self.emit_debug()

            # PHASE 4: Branch computation (ALL rounds)
            # Use per-group temp v_t[it][i] for bit and reuse for in_bounds
            # (sequential ops so no conflict)
            # Uses multiply_add fusion: idx = idx * 2 + t
            #
            # Stage 5 optimization: val_opt = val_orig ^ C, so LSB is flipped.
            # Original: branch = (val_orig & 1) + 1
            # Optimized: branch = ((val_opt & 1) ^ 1) + 1 = 2 - (val_opt & 1)
            #
            # Optimization: Skip bounds check for early rounds where all indices
            # are guaranteed in-bounds. At round r, max index is 2^(r+1) - 1.
            # Bounds check needed when 2^(r+1) - 1 >= n_nodes.
            max_idx_this_round = (1 << (rnd + 2)) - 1  # After branch: max is 2^(rnd+2) - 1
            needs_bounds_check = max_idx_this_round >= n_nodes

            for it in range(n_iters):
                base_g = it * UNROLL
                for i in range(UNROLL):
                    t = v_t[it][i]  # Per-group temp
                    # Stage 5 opt: t = 2 - (val & 1) = flipped branch
                    self.emit(valu=[("&", t, all_val[base_g+i], v_1)])
                    self.emit(valu=[("-", t, v_2, t)])
                    # idx = idx * 2 + t (fused multiply_add saves 1 VALU op)
                    self.emit(valu=[("multiply_add", all_idx[base_g+i], all_idx[base_g+i], v_2, t)])
                    # Only do bounds check when needed (compile-time decision)
                    if needs_bounds_check:
                        # Reuse t for in_bounds (previous value no longer needed)
                        self.emit(valu=[("<", t, all_idx[base_g+i], v_nn)])
                        # idx = idx * in_bounds (zeroes if out of bounds)
                        self.emit(valu=[("*", all_idx[base_g+i], all_idx[base_g+i], t)])
            # DEBUG: emit wrapped_idx comparison after branch/bounds
            for g in range(n_groups):
                keys = [(rnd, g * VLEN + vi, "wrapped_idx") for vi in range(VLEN)]
                self.emit_vcompare(all_idx[g], keys)
            self.emit_debug()

        rounds_end = len(self.instrs)

        # Stage 5 optimization: XOR all values with C before storing
        # This converts val_opt back to val_orig
        for g in range(n_groups):
            self.emit(valu=[("^", all_val[g], all_val[g], v_stage5_xor)])

        # STORE DATA
        for g in range(0, n_groups, 2):
            self.emit(load=[("const", st[0], g * VLEN), ("const", st[1], (g+1) * VLEN)])
            self.emit(alu=[("+", st[2], idx_p, st[0]), ("+", st[3], idx_p, st[1]), ("+", st[4], val_p, st[0]), ("+", st[5], val_p, st[1])])
            self.emit(store=[("vstore", st[2], all_idx[g]), ("vstore", st[3], all_idx[g+1])])
            self.emit(store=[("vstore", st[4], all_val[g]), ("vstore", st[5], all_val[g+1])])
        store_end = len(self.instrs)

        print(f"Before scheduling: {store_end} instructions")

        # RESCHEDULE - either with fast greedy or optimal CP-SAT
        ops = extract_ops(self.instrs)
        op_by_id = compute_dependencies(ops)
        print(f"Scheduling {len(ops)} ops...")

        if use_greedy:
            # Fast greedy scheduler - runs in seconds
            import time
            start = time.time()
            num_cycles, scheduled_ops = schedule_greedy(ops, op_by_id, enable_valu_shift=enable_valu_shift)
            scheduled = emit_scheduled(scheduled_ops, num_cycles)
            elapsed = time.time() - start
            print(f"Greedy scheduling completed in {elapsed:.2f}s")
        else:
            # CP-SAT optimal scheduler - may take hours for large kernels
            from solver.round_chunked_scheduler import schedule_round_chunked
            # Multi-phase: chunked → full solve → VALU shifting
            num_cycles, scheduled = schedule_round_chunked(
                ops, op_by_id, rounds, n_iters, UNROLL,
                time_limit=60.0, full_solve_time=full_solve_time,
                require_optimal=require_optimal, enable_valu_shift=enable_valu_shift,
                valu_shift_time=valu_shift_time
            )

        self.instrs = [{"flow": [("pause",)]}] + scheduled + [{"flow": [("pause",)]}]
        print(f"Final scheduling: {len(self.instrs)} instructions ({num_cycles} cycles)")
