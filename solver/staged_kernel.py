"""
Staged Tree Loading Kernel - EXPERIMENTAL (DOES NOT IMPROVE PERFORMANCE)

*** NOTE: This approach was found to be SLOWER than the original kernel! ***
*** Original: ~2,227 cycles | Staged: ~3,990 cycles                      ***

The original plan assumed staging could overlap with VALU computation, but there's
a fundamental data dependency: staging for round N+1 requires the NEW INDICES from
round N's branch. So staging MUST wait until after the branch completes - there's
no overlap with the current round's VALU ops.

Why staging doesn't help:
1. We still need 256 scattered loads (to get tree values for staging)
2. We ADD 256 stores (to put values into staging buffer)
3. The stores can't overlap with VALU due to the branch dependency
4. Net result: more work, not less

The original kernel already achieves good overlap:
- Different batch groups interleave naturally
- Group 0's hash can overlap with Group 1's gather
- The scheduler handles this efficiently

Kept for educational purposes - shows why this optimization approach fails.
"""

from problem import DebugInfo, SLOT_LIMITS, VLEN, SCRATCH_SIZE, HASH_STAGES
from solver.reschedule import extract_ops, compute_dependencies, schedule_greedy, emit_scheduled


class StagedKernelBuilder:
    """Build a staged tree loading kernel with pipelined staging."""

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
        rounds: int,
        full_solve_time: float = 3600.0,
        require_optimal: bool = True,
        enable_valu_shift: bool = True,
        valu_shift_time: float = None,
        use_greedy: bool = True
    ):
        """Build staged kernel with pipelined tree loading.

        The key optimization is using double-buffered staging areas in memory:
        - During round N, we vload tree values from staged buffer (16 cycles)
        - While computing hash/branch, we prepare staged buffer for round N+1
        - This overlaps the expensive scattered loads with VALU computation
        """
        n_groups = batch_size // VLEN  # 32 groups of 8 elements
        UNROLL = 8
        n_iters = n_groups // UNROLL  # 4 iterations

        # ========== SCRATCH ALLOCATION ==========
        forest_p = self.alloc("forest_p")
        idx_p = self.alloc("idx_p")
        val_p = self.alloc("val_p")
        n_nodes_s = self.alloc("n_nodes")
        extra_room_p_scratch = self.alloc("extra_room_p")

        # Staging buffer base addresses (read from memory at runtime)
        staged_A_base = self.alloc("staged_A_base")
        staged_B_base = self.alloc("staged_B_base")
        const_256 = self.alloc("const_256")  # Constant for staged_B offset

        st = [self.alloc(f"st{i}") for i in range(24)]  # Extra scratch for staging

        # Per-iteration scratch for staging operations (to avoid conflicts after reordering)
        # For each group pair (0,1), (2,3), ..., (30,31) we need separate scratch
        # st_addr[g//2] = scratch for address computation for groups g and g+1
        # st_val[g//2] = scratch for loaded values for groups g and g+1
        st_addr_a = [self.alloc(f"st_addr_a_{i}") for i in range(n_groups // 2)]
        st_addr_b = [self.alloc(f"st_addr_b_{i}") for i in range(n_groups // 2)]
        st_val_a = [self.alloc(f"st_val_a_{i}") for i in range(n_groups // 2)]
        st_val_b = [self.alloc(f"st_val_b_{i}") for i in range(n_groups // 2)]
        st_dest_a = [self.alloc(f"st_dest_a_{i}") for i in range(n_groups // 2)]
        st_dest_b = [self.alloc(f"st_dest_b_{i}") for i in range(n_groups // 2)]
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

        # Per-group temps
        v_t = [[self.vec(f"v_t_{it}_{i}") for i in range(UNROLL)] for it in range(n_iters)]

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

        # FMA optimization multipliers
        hash_fma_mult = [None] * 6
        v_hash_mult = {}
        for i, (op1, const, op2, op3, shift) in enumerate(HASH_STAGES):
            if op1 == '+' and op2 == '+' and op3 == '<<':
                hash_fma_mult[i] = 1 + (1 << shift)

        # Stage 5 XOR optimization constant
        STAGE5_XOR_CONST = 0xB55A4F09
        stage5_xor_s = self.alloc("stage5_xor")
        v_stage5_xor = self.vec("v_stage5_xor")

        print(f"Scratch usage: {self.scratch_ptr} / {SCRATCH_SIZE}")

        # ========== INITIALIZATION ==========
        for i in range(16):
            self.emit(load=[("const", c[i], i)])

        # Load pointers from memory header
        self.emit(load=[("const", st[0], 4), ("const", st[1], 5)])
        self.emit(load=[("load", forest_p, st[0]), ("load", idx_p, st[1])])
        self.emit(load=[("const", st[0], 6), ("const", st[1], 1)])
        self.emit(load=[("load", val_p, st[0]), ("load", n_nodes_s, st[1])])

        # Load extra_room pointer and set up staging buffer addresses
        # extra_room_p is at memory[7], staged_A at extra_room_p+0, staged_B at extra_room_p+256
        self.emit(load=[("const", st[0], 7), ("const", const_256, 256)])
        self.emit(load=[("load", extra_room_p_scratch, st[0])])
        self.emit(alu=[("+", staged_A_base, extra_room_p_scratch, c[0])])  # staged_A = extra_room + 0
        self.emit(alu=[("+", staged_B_base, extra_room_p_scratch, const_256)])  # staged_B = extra_room + 256

        # Load hash constants
        for i in range(6):
            self.emit(load=[("const", hc[i], HASH_STAGES[i][1]), ("const", hs[i], HASH_STAGES[i][4])])

        # Broadcast vector constants
        self.emit(valu=[("vbroadcast", v_0, c[0]), ("vbroadcast", v_1, c[1]),
                        ("vbroadcast", v_2, c[2]), ("vbroadcast", v_nn, n_nodes_s)])
        for i in range(0, 8, 2):
            self.emit(valu=[("vbroadcast", v_const[i], c[i]), ("vbroadcast", v_const[i+1], c[i+1])])
        for i in range(0, 6, 2):
            self.emit(valu=[("vbroadcast", v_hc[i], hc[i]), ("vbroadcast", v_hc[i+1], hc[i+1]),
                            ("vbroadcast", v_hs[i], hs[i]), ("vbroadcast", v_hs[i+1], hs[i+1])])

        # Initialize FMA multipliers
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

        # Load tree constants for early rounds
        for i in range(7):
            self.emit(alu=[("+", st[0], forest_p, c[i])])
            self.emit(load=[("load", tc[i], st[0])])

        self.emit(alu=[("-", td_1, tc[1], tc[2]), ("-", td_4, tc[4], tc[3]),
                       ("-", td_5, tc[5], tc[3]), ("-", td_6, tc[6], tc[3])])
        self.emit(valu=[("vbroadcast", v_tc2, tc[2]), ("vbroadcast", v_td_1, td_1),
                        ("vbroadcast", v_tc3, tc[3]), ("vbroadcast", v_td_4, td_4),
                        ("vbroadcast", v_td_5, td_5), ("vbroadcast", v_td_6, td_6)])

        # ========== LOAD INPUT DATA ==========
        for g in range(0, n_groups, 2):
            self.emit(load=[("const", st[0], g * VLEN), ("const", st[1], (g+1) * VLEN)])
            self.emit(alu=[("+", st[2], idx_p, st[0]), ("+", st[3], idx_p, st[1]),
                          ("+", st[4], val_p, st[0]), ("+", st[5], val_p, st[1])])
            self.emit(load=[("vload", all_idx[g], st[2]), ("vload", all_idx[g+1], st[3])])
            self.emit(load=[("vload", all_val[g], st[4]), ("vload", all_val[g+1], st[5])])

        # ========== FULLY UNROLLED ROUNDS ==========
        for rnd in range(rounds):
            # Determine which staging buffer is "current" vs "next" for this round
            # Rounds 0-2: no staging needed (tree values computed/precomputed)
            # Round 3+: alternate between buffers A and B
            # Round 3: reads from A (bootstrap filled it), writes to B
            # Round 4: reads from B, writes to A
            # etc.
            if rnd >= 3:
                if rnd % 2 == 1:
                    current_staged_base = staged_A_base
                    next_staged_base = staged_B_base
                else:
                    current_staged_base = staged_B_base
                    next_staged_base = staged_A_base

            # ---------- PHASE 1: Get node values ----------
            if rnd == 0:
                # Round 0: just broadcast tc[0]
                for it in range(n_iters):
                    for i in range(UNROLL):
                        self.emit(valu=[("vbroadcast", v_nv[it][i], tc[0])])

            elif rnd == 1:
                # Round 1: v_nv = tc[2] + (idx == 1) * td_1
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

            elif rnd == 2:
                # Round 2: v_nv = tc[3] + select based on idx == 4, 5, 6
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

                # ---------- BOOTSTRAP: Stage tree values for round 3 ----------
                # After round 2 branch, indices are in range [7,14] (level 3 nodes)
                # We need to stage these values into staged_A for round 3 to vload
                # NOTE: This staging happens at end of round 2, BEFORE we call the
                # normal round 2 hash/branch (so indices are still from round 2 start)
                # Actually, we need to do this AFTER round 2 branch (see below)

            else:
                # Rounds 3-15: vload from staged buffer (filled by previous round or bootstrap)
                # Use per-group-pair scratch slots to avoid conflicts after scheduling
                for g in range(0, n_groups, 2):
                    pair_idx = g // 2
                    offset_a = g * VLEN
                    offset_b = (g + 1) * VLEN
                    # Use st_addr for this operation (won't conflict with staging which uses different slots)
                    self.emit(load=[("const", st_addr_a[pair_idx], offset_a),
                                    ("const", st_addr_b[pair_idx], offset_b)])
                    self.emit(alu=[("+", st_val_a[pair_idx], current_staged_base, st_addr_a[pair_idx]),
                                  ("+", st_val_b[pair_idx], current_staged_base, st_addr_b[pair_idx])])
                    it = g // UNROLL
                    i = g % UNROLL
                    # Proper indexing for v_nv based on group number
                    nv_a = v_nv[it][i]
                    nv_b = v_nv[it][i+1] if i+1 < UNROLL else v_nv[it+1][0]
                    self.emit(load=[("vload", nv_a, st_val_a[pair_idx]),
                                    ("vload", nv_b, st_val_b[pair_idx])])
                # Stage 5 opt: XOR v_nv with C
                for it in range(n_iters):
                    for i in range(UNROLL):
                        self.emit(valu=[("^", v_nv[it][i], v_nv[it][i], v_stage5_xor)])

            # ---------- PHASE 2: XOR values with node values ----------
            for it in range(n_iters):
                base_g = it * UNROLL
                for i in range(UNROLL):
                    self.emit(valu=[("^", all_val[base_g+i], all_val[base_g+i], v_nv[it][i])])

            # ---------- PHASE 3: Hash (6 stages) ----------
            for hi in range(6):
                op1, _, op2, op3, _ = HASH_STAGES[hi]
                for it in range(n_iters):
                    base_g = it * UNROLL
                    for i in range(UNROLL):
                        if hash_fma_mult[hi] is not None:
                            self.emit(valu=[("multiply_add", all_val[base_g+i],
                                            all_val[base_g+i], v_hash_mult[hi], v_hc[hi])])
                        elif hi == 5:
                            t = v_t[it][i]
                            self.emit(valu=[(">>", t, all_val[base_g+i], v_hs[hi])])
                            self.emit(valu=[("^", all_val[base_g+i], all_val[base_g+i], t)])
                        else:
                            t = v_t[it][i]
                            self.emit(valu=[(op1, t, all_val[base_g+i], v_hc[hi]),
                                            (op3, v_nv[it][i], all_val[base_g+i], v_hs[hi])])
                            self.emit(valu=[(op2, all_val[base_g+i], t, v_nv[it][i])])

            # ---------- PHASE 4: Branch computation ----------
            max_idx_this_round = (1 << (rnd + 2)) - 1
            needs_bounds_check = max_idx_this_round >= n_nodes

            for it in range(n_iters):
                base_g = it * UNROLL
                for i in range(UNROLL):
                    t = v_t[it][i]
                    self.emit(valu=[("&", t, all_val[base_g+i], v_1)])
                    self.emit(valu=[("-", t, v_2, t)])
                    self.emit(valu=[("multiply_add", all_idx[base_g+i], all_idx[base_g+i], v_2, t)])
                    if needs_bounds_check:
                        self.emit(valu=[("<", t, all_idx[base_g+i], v_nn)])
                        self.emit(valu=[("*", all_idx[base_g+i], all_idx[base_g+i], t)])

            # ---------- PHASE 5: Stage tree values for NEXT round ----------
            # After branch, we have new indices. Stage tree values for next round.
            # This uses scattered loads + stores, but they overlap with the next
            # round's VALU operations due to the scheduler.
            #
            # For round 2: bootstrap staged_A for round 3
            # For rounds 3-14: stage to next buffer for rounds 4-15
            # Round 15 (last): no staging needed

            if rnd == 2:
                # Bootstrap: after round 2 branch, stage tree values into staged_A
                # Indices are now in range [7,14] for the next round (round 3)
                # Use per-group-pair scratch to avoid conflicts after scheduling
                for vi in range(VLEN):
                    for g in range(0, n_groups, 2):
                        pair_idx = g // 2
                        # Compute tree addresses (use per-pair scratch)
                        self.emit(alu=[
                            ("+", st_addr_a[pair_idx], forest_p, all_idx[g] + vi),
                            ("+", st_addr_b[pair_idx], forest_p, all_idx[g+1] + vi)
                        ])
                        # Scattered loads from tree
                        self.emit(load=[
                            ("load", st_val_a[pair_idx], st_addr_a[pair_idx]),
                            ("load", st_val_b[pair_idx], st_addr_b[pair_idx])
                        ])
                        # Compute destination addresses in staged buffer
                        dest_offset_a = g * VLEN + vi
                        dest_offset_b = (g+1) * VLEN + vi
                        self.emit(load=[
                            ("const", st_dest_a[pair_idx], dest_offset_a),
                            ("const", st_dest_b[pair_idx], dest_offset_b)
                        ])
                        self.emit(alu=[
                            ("+", st_addr_a[pair_idx], staged_A_base, st_dest_a[pair_idx]),
                            ("+", st_addr_b[pair_idx], staged_A_base, st_dest_b[pair_idx])
                        ])
                        # Store to staging buffer
                        self.emit(store=[
                            ("store", st_addr_a[pair_idx], st_val_a[pair_idx]),
                            ("store", st_addr_b[pair_idx], st_val_b[pair_idx])
                        ])

            elif 3 <= rnd < rounds - 1:
                # Rounds 3-14: stage tree values for next round
                # Use per-group-pair scratch to avoid conflicts after scheduling
                for vi in range(VLEN):
                    for g in range(0, n_groups, 2):
                        pair_idx = g // 2
                        # Compute tree addresses from NEW indices
                        self.emit(alu=[
                            ("+", st_addr_a[pair_idx], forest_p, all_idx[g] + vi),
                            ("+", st_addr_b[pair_idx], forest_p, all_idx[g+1] + vi)
                        ])
                        # Scattered loads from tree
                        self.emit(load=[
                            ("load", st_val_a[pair_idx], st_addr_a[pair_idx]),
                            ("load", st_val_b[pair_idx], st_addr_b[pair_idx])
                        ])
                        # Compute destination addresses in NEXT staged buffer
                        dest_offset_a = g * VLEN + vi
                        dest_offset_b = (g+1) * VLEN + vi
                        self.emit(load=[
                            ("const", st_dest_a[pair_idx], dest_offset_a),
                            ("const", st_dest_b[pair_idx], dest_offset_b)
                        ])
                        self.emit(alu=[
                            ("+", st_addr_a[pair_idx], next_staged_base, st_dest_a[pair_idx]),
                            ("+", st_addr_b[pair_idx], next_staged_base, st_dest_b[pair_idx])
                        ])
                        # Store to staging buffer
                        self.emit(store=[
                            ("store", st_addr_a[pair_idx], st_val_a[pair_idx]),
                            ("store", st_addr_b[pair_idx], st_val_b[pair_idx])
                        ])

        # ========== FINAL XOR AND STORE ==========
        # Stage 5 optimization: XOR all values with C before storing
        for g in range(n_groups):
            self.emit(valu=[("^", all_val[g], all_val[g], v_stage5_xor)])

        # Store results
        for g in range(0, n_groups, 2):
            self.emit(load=[("const", st[0], g * VLEN), ("const", st[1], (g+1) * VLEN)])
            self.emit(alu=[("+", st[2], idx_p, st[0]), ("+", st[3], idx_p, st[1]),
                          ("+", st[4], val_p, st[0]), ("+", st[5], val_p, st[1])])
            self.emit(store=[("vstore", st[2], all_idx[g]), ("vstore", st[3], all_idx[g+1])])
            self.emit(store=[("vstore", st[4], all_val[g]), ("vstore", st[5], all_val[g+1])])

        print(f"Before scheduling: {len(self.instrs)} instructions")

        # ========== RESCHEDULE ==========
        ops = extract_ops(self.instrs)
        op_by_id = compute_dependencies(ops)
        print(f"Scheduling {len(ops)} ops...")

        if use_greedy:
            import time
            start = time.time()
            num_cycles, scheduled_ops = schedule_greedy(ops, op_by_id, enable_valu_shift=enable_valu_shift)
            scheduled = emit_scheduled(scheduled_ops, num_cycles)
            elapsed = time.time() - start
            print(f"Greedy scheduling completed in {elapsed:.2f}s")
        else:
            from solver.round_chunked_scheduler import schedule_round_chunked
            num_cycles, scheduled = schedule_round_chunked(
                ops, op_by_id, rounds, n_iters, UNROLL,
                time_limit=60.0, full_solve_time=full_solve_time,
                require_optimal=require_optimal, enable_valu_shift=enable_valu_shift,
                valu_shift_time=valu_shift_time
            )

        self.instrs = [{"flow": [("pause",)]}] + scheduled + [{"flow": [("pause",)]}]
        print(f"Final scheduling: {len(self.instrs)} instructions ({num_cycles} cycles)")
