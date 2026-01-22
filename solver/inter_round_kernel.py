"""
Inter-Round Pipelined Kernel - Overlap rounds for maximum engine utilization.

Key insight: While processing (VALU) the last iter of round R,
we can gather (Load+ALU) the first iter of round R+1.

This eliminates VALU-only bundles from process_only and Load+ALU-only
bundles from gather_only by overlapping them!

Structure:
- Warm-up: gather round 3 iter 0
- Loop body:
  - gather_and_process(iter 1) for round R
  - gather_and_process(iter 2) for round R
  - gather_and_process(iter 3) for round R
  - gather(iter 0 for R+1) + process(iter 3 for R)  <- KEY PACKING!
  - R++
- Cool-down: process round 15 iter 3
"""

from problem import DebugInfo, SLOT_LIMITS, VLEN, SCRATCH_SIZE, HASH_STAGES


class InterRoundKernelBuilder:
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

    def slot_utilization(self, verbose: bool = True, actual_cycles: int = None) -> dict:
        """
        Compute slot utilization statistics for the emitted instructions.

        Args:
            verbose: Print summary table
            actual_cycles: If provided, also compute dynamic utilization based on
                          actual executed cycles (for kernels with loops)

        Returns a dict with:
        - used: {engine: total slots used in static instructions}
        - available: {engine: total slots available (cycles * limit)}
        - unused: {engine: available - used}
        - utilization: {engine: used / available as percentage}
        - per_cycle: list of per-cycle usage dicts
        """
        n_instructions = len(self.instrs)

        # Count used slots per engine (excluding debug which isn't a real constraint)
        engines = ['valu', 'alu', 'load', 'store', 'flow']
        used = {engine: 0 for engine in engines}
        per_cycle = []

        for instr in self.instrs:
            cycle_usage = {}
            for engine in engines:
                count = len(instr.get(engine, []))
                used[engine] += count
                cycle_usage[engine] = count
            per_cycle.append(cycle_usage)

        # Compute available and unused based on static instruction count
        available = {engine: n_instructions * SLOT_LIMITS[engine] for engine in engines}
        unused = {engine: available[engine] - used[engine] for engine in engines}
        utilization = {
            engine: (used[engine] / available[engine] * 100) if available[engine] > 0 else 0
            for engine in engines
        }

        if verbose:
            print(f"\n=== Slot Utilization ({n_instructions} instructions) ===")
            print(f"{'Engine':<8} {'Limit':<6} {'Used':<8} {'Available':<10} {'Unused':<8} {'Util%':<8}")
            print("-" * 60)
            for engine in engines:
                print(f"{engine:<8} {SLOT_LIMITS[engine]:<6} {used[engine]:<8} {available[engine]:<10} {unused[engine]:<8} {utilization[engine]:<8.1f}")

            total_used = sum(used.values())
            total_available = sum(available.values())
            total_unused = sum(unused.values())
            total_util = (total_used / total_available * 100) if total_available > 0 else 0
            print("-" * 60)
            print(f"{'TOTAL':<8} {'':<6} {total_used:<8} {total_available:<10} {total_unused:<8} {total_util:<8.1f}")

            # If actual_cycles provided, show dynamic utilization
            if actual_cycles is not None:
                print(f"\n=== Dynamic Utilization ({actual_cycles} executed cycles) ===")
                dyn_available = {engine: actual_cycles * SLOT_LIMITS[engine] for engine in engines}
                # Estimate dynamic used based on ratio of actual to static cycles
                # (This is approximate since loops execute instructions multiple times)
                ratio = actual_cycles / n_instructions if n_instructions > 0 else 1
                dyn_used = {engine: int(used[engine] * ratio) for engine in engines}
                dyn_unused = {engine: dyn_available[engine] - dyn_used[engine] for engine in engines}
                dyn_util = {
                    engine: (dyn_used[engine] / dyn_available[engine] * 100) if dyn_available[engine] > 0 else 0
                    for engine in engines
                }
                print(f"{'Engine':<8} {'Limit':<6} {'~Used':<8} {'Available':<10} {'~Unused':<8} {'~Util%':<8}")
                print("-" * 60)
                for engine in engines:
                    print(f"{engine:<8} {SLOT_LIMITS[engine]:<6} {dyn_used[engine]:<8} {dyn_available[engine]:<10} {dyn_unused[engine]:<8} {dyn_util[engine]:<8.1f}")

        return {
            'instructions': n_instructions,
            'used': used,
            'available': available,
            'unused': unused,
            'utilization': utilization,
            'per_cycle': per_cycle,
        }

    def build_kernel(
        self,
        forest_height: int,
        n_nodes: int,
        batch_size: int,
        rounds: int
    ):
        n_groups = batch_size // VLEN  # 32
        UNROLL = 8
        n_iters = n_groups // UNROLL  # 4

        # SCRATCH
        forest_p = self.alloc("forest_p")
        idx_p = self.alloc("idx_p")
        val_p = self.alloc("val_p")
        n_nodes_s = self.alloc("n_nodes")
        round_i = self.alloc("round_i")

        st = [self.alloc(f"st{i}") for i in range(16)]
        c = [self.alloc(f"c{i}") for i in range(16)]
        c_vlen = self.alloc("c_vlen")
        c_rounds = self.alloc("c_rounds")
        c_rounds_minus_1 = self.alloc("c_rounds_m1")

        hc = [self.alloc(f"hc{i}") for i in range(6)]
        hs = [self.alloc(f"hs{i}") for i in range(6)]

        tc = [self.alloc(f"tc{i}") for i in range(7)]
        td_1 = self.alloc("td_1")
        td_4 = self.alloc("td_4")
        td_5 = self.alloc("td_5")
        td_6 = self.alloc("td_6")

        all_idx = [self.vec(f"all_idx_{g}") for g in range(n_groups)]
        all_val = [self.vec(f"all_val_{g}") for g in range(n_groups)]

        # v_nv for each iteration (double buffer for overlap)
        v_nv = [[self.vec(f"v_nv_{it}_{i}") for i in range(UNROLL)] for it in range(n_iters)]
        # Separate gather buffers for initial gather during cached rounds
        v_nv_gather = [[self.vec(f"v_nv_g_{it}_{i}") for i in range(UNROLL)] for it in range(n_iters)]

        v_t1 = self.vec("v_t1")
        v_t2 = self.vec("v_t2")
        v_t3 = self.vec("v_t3")

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

        # Pre-broadcast hash constants
        v_hc = [self.vec(f"v_hc{i}") for i in range(6)]
        v_hs = [self.vec(f"v_hs{i}") for i in range(6)]

        # INIT
        for i in range(0, 16, 2):
            self.emit(load=[("const", c[i], i), ("const", c[i+1], i+1)])

        self.emit(load=[("const", c_vlen, VLEN), ("const", c_rounds, rounds)])
        self.emit(load=[("const", c_rounds_minus_1, rounds - 1)])
        self.emit(load=[("const", st[0], 4), ("const", st[1], 5)])
        self.emit(load=[("load", forest_p, st[0]), ("load", idx_p, st[1])])
        self.emit(load=[("const", st[0], 6), ("const", st[1], 1)])
        self.emit(load=[("load", val_p, st[0]), ("load", n_nodes_s, st[1])])

        for i in range(0, 6, 2):
            self.emit(load=[("const", hc[i], HASH_STAGES[i][1]), ("const", hc[i+1], HASH_STAGES[i+1][1])])
        for i in range(0, 6, 2):
            self.emit(load=[("const", hs[i], HASH_STAGES[i][4]), ("const", hs[i+1], HASH_STAGES[i+1][4])])

        self.emit(valu=[
            ("vbroadcast", v_0, c[0]),
            ("vbroadcast", v_1, c[1]),
            ("vbroadcast", v_2, c[2]),
            ("vbroadcast", v_nn, n_nodes_s),
            ("vbroadcast", v_const[0], c[0]),
            ("vbroadcast", v_const[1], c[1]),
        ])
        self.emit(valu=[
            ("vbroadcast", v_const[2], c[2]),
            ("vbroadcast", v_const[3], c[3]),
            ("vbroadcast", v_const[4], c[4]),
            ("vbroadcast", v_const[5], c[5]),
            ("vbroadcast", v_const[6], c[6]),
            ("vbroadcast", v_const[7], c[7]),
        ])

        # Pre-broadcast hash constants
        self.emit(valu=[("vbroadcast", v_hc[i], hc[i]) for i in range(6)])
        self.emit(valu=[("vbroadcast", v_hs[i], hs[i]) for i in range(6)])

        # Load tree cache
        for i in range(0, 6, 2):
            self.emit(alu=[("+", st[0], forest_p, c[i]), ("+", st[1], forest_p, c[i+1])])
            self.emit(load=[("load", tc[i], st[0]), ("load", tc[i+1], st[1])])
        self.emit(alu=[("+", st[0], forest_p, c[6])])
        self.emit(load=[("load", tc[6], st[0])])

        # Compute diffs
        self.emit(alu=[("-", td_1, tc[1], tc[2])])
        self.emit(alu=[("-", td_4, tc[4], tc[3]), ("-", td_5, tc[5], tc[3]), ("-", td_6, tc[6], tc[3])])

        self.emit(valu=[
            ("vbroadcast", v_tc2, tc[2]),
            ("vbroadcast", v_td_1, td_1),
            ("vbroadcast", v_tc3, tc[3]),
            ("vbroadcast", v_td_4, td_4),
            ("vbroadcast", v_td_5, td_5),
            ("vbroadcast", v_td_6, td_6),
        ])

        self.emit(flow=[("pause",)])

        # LOAD ALL DATA
        for g in range(0, n_groups, 2):
            self.emit(load=[("const", st[0], g * VLEN), ("const", st[1], (g+1) * VLEN)])
            self.emit(alu=[
                ("+", st[2], idx_p, st[0]),
                ("+", st[3], idx_p, st[1]),
                ("+", st[4], val_p, st[0]),
                ("+", st[5], val_p, st[1]),
            ])
            self.emit(load=[("vload", all_idx[g], st[2]), ("vload", all_idx[g+1], st[3])])
            self.emit(load=[("vload", all_val[g], st[4]), ("vload", all_val[g+1], st[5])])

        # CACHED ROUNDS 0-2
        for rnd in range(min(3, rounds)):
            for it in range(n_iters):
                base_g = it * UNROLL
                self._emit_cached_round(rnd, base_g, UNROLL, all_idx, all_val, tc,
                                        v_nv[it], v_t1, v_t2, v_t3, v_hc, v_hs,
                                        v_0, v_1, v_2, v_nn, v_const, v_td_1, v_tc2,
                                        v_tc3, v_td_4, v_td_5, v_td_6)

        # GATHER ROUNDS 3+ with inter-round pipelining
        if rounds > 3:
            # Warm-up: gather round 3 iter 0
            self._emit_gather_only(0 * UNROLL, UNROLL, all_idx, forest_p, v_nv[0], st)

            self.emit(load=[("const", round_i, 3)])
            round_start = len(self.instrs)

            # gather iter 1 + process iter 0
            self._emit_gather_and_process(
                1 * UNROLL, 0 * UNROLL, UNROLL, all_idx, all_val, forest_p,
                v_nv[1], v_nv[0], v_t1, v_t2, v_hc, v_hs, v_0, v_1, v_2, v_nn, st
            )

            # gather iter 2 + process iter 1
            self._emit_gather_and_process(
                2 * UNROLL, 1 * UNROLL, UNROLL, all_idx, all_val, forest_p,
                v_nv[2], v_nv[1], v_t1, v_t2, v_hc, v_hs, v_0, v_1, v_2, v_nn, st
            )

            # gather iter 3 + process iter 2
            self._emit_gather_and_process(
                3 * UNROLL, 2 * UNROLL, UNROLL, all_idx, all_val, forest_p,
                v_nv[3], v_nv[2], v_t1, v_t2, v_hc, v_hs, v_0, v_1, v_2, v_nn, st
            )

            # KEY: gather iter 0 (next round) + process iter 3 (current round)
            self._emit_gather_and_process(
                0 * UNROLL, 3 * UNROLL, UNROLL, all_idx, all_val, forest_p,
                v_nv[0], v_nv[3], v_t1, v_t2, v_hc, v_hs, v_0, v_1, v_2, v_nn, st
            )

            self.emit(alu=[("+", round_i, round_i, c[1])])
            self.emit(alu=[("<", st[0], round_i, c_rounds_minus_1)])
            self.emit(flow=[("cond_jump", st[0], round_start)])

            # Cool-down: process round 15 iterations 1-3 (iter 0 already processed in loop!)
            # Actually no - we need to process all iters for round 15

            # After last loop iteration:
            # - We've gathered iter 0 for round 16 (but round 16 doesn't exist)
            # - We've processed iter 3 for round 14
            # Wait, that's wrong. Let me re-think...

            # Loop runs for rounds 3 to 14 (round_i < rounds-1 = 15)
            # After loop, round_i = 15
            # For round 15:
            # - iter 0 was gathered in the last "gather iter 0 next round" but not processed
            # - iters 1,2,3 need gather and process

            # Need to complete round 15:
            # - process iter 0 (already gathered)
            # - gather + process iter 1
            # - gather + process iter 2
            # - gather + process iter 3
            # - process iter 3

            # Actually the loop structure is confusing. Let me trace through:
            # round_i = 3:
            #   gather_and_process(iter 1, 0) for round 3
            #   gather_and_process(iter 2, 1) for round 3
            #   gather_and_process(iter 3, 2) for round 3
            #   gather_and_process(iter 0 round 4, iter 3 round 3)
            # round_i = 4:
            #   ... processes round 4, gathers iter 0 for round 5
            # ...
            # round_i = 14:
            #   ... processes round 14, gathers iter 0 for round 15
            # round_i = 15, exit loop
            #
            # After loop: iter 0 for round 15 is gathered, need to process it
            # Then continue with iters 1,2,3 for round 15

            # Final round (15) processing:
            # gather iter 1 + process iter 0
            self._emit_gather_and_process(
                1 * UNROLL, 0 * UNROLL, UNROLL, all_idx, all_val, forest_p,
                v_nv[1], v_nv[0], v_t1, v_t2, v_hc, v_hs, v_0, v_1, v_2, v_nn, st
            )

            # gather iter 2 + process iter 1
            self._emit_gather_and_process(
                2 * UNROLL, 1 * UNROLL, UNROLL, all_idx, all_val, forest_p,
                v_nv[2], v_nv[1], v_t1, v_t2, v_hc, v_hs, v_0, v_1, v_2, v_nn, st
            )

            # gather iter 3 + process iter 2
            self._emit_gather_and_process(
                3 * UNROLL, 2 * UNROLL, UNROLL, all_idx, all_val, forest_p,
                v_nv[3], v_nv[2], v_t1, v_t2, v_hc, v_hs, v_0, v_1, v_2, v_nn, st
            )

            # process iter 3 (no more gather)
            self._emit_process_only(3 * UNROLL, UNROLL, all_idx, all_val,
                                    v_nv[3], v_t1, v_t2, v_hc, v_hs, v_0, v_1, v_2, v_nn)

        # STORE
        for g in range(0, n_groups, 2):
            self.emit(load=[("const", st[0], g * VLEN), ("const", st[1], (g+1) * VLEN)])
            self.emit(alu=[
                ("+", st[2], idx_p, st[0]),
                ("+", st[3], idx_p, st[1]),
                ("+", st[4], val_p, st[0]),
                ("+", st[5], val_p, st[1]),
            ])
            self.emit(store=[("vstore", st[2], all_idx[g]), ("vstore", st[3], all_idx[g+1])])
            self.emit(store=[("vstore", st[4], all_val[g]), ("vstore", st[5], all_val[g+1])])

        self.emit(flow=[("pause",)])

    def _emit_cached_round_staged(self, rnd, n_iters, UNROLL, all_idx, all_val, tc,
                                   v_nv_all, v_t1, v_t2, v_t3, v_hc, v_hs,
                                   v_0, v_1, v_2, v_nn, v_const, v_td_1, v_tc2,
                                   v_tc3, v_td_4, v_td_5, v_td_6):
        """Process cached round with stage-first ordering for better VALU packing.

        Key insight: iterations within a round are independent, so we can
        process each STAGE for all iterations before moving to the next stage.
        This allows merging 2-op bundles from group 6-7 across iterations.
        """

        def emit_ops_packed(ops):
            """Emit ops in 6-op bundles."""
            for i in range(0, len(ops), 6):
                self.emit(valu=ops[i:i+6])

        # Stage 1: Set v_nv values for all iterations
        nv_ops = []
        for it in range(n_iters):
            base_g = it * UNROLL
            v_nv = v_nv_all[it]
            if rnd == 0:
                nv_ops.extend([("vbroadcast", v_nv[lg], tc[0]) for lg in range(UNROLL)])
            elif rnd == 1:
                temps = [v_t1, v_t2, v_t3, v_nv[4], v_nv[5], v_nv[6]]
                nv_ops.extend([("==", temps[i], all_idx[base_g+i], v_1) for i in range(6)])
        emit_ops_packed(nv_ops)

        # Round 1 has additional steps that depend on the == result
        if rnd == 1:
            mul_ops = []
            add_ops = []
            for it in range(n_iters):
                base_g = it * UNROLL
                v_nv = v_nv_all[it]
                temps = [v_t1, v_t2, v_t3, v_nv[4], v_nv[5], v_nv[6]]
                mul_ops.extend([("*", temps[i], v_td_1, temps[i]) for i in range(6)])
            emit_ops_packed(mul_ops)

            for it in range(n_iters):
                base_g = it * UNROLL
                v_nv = v_nv_all[it]
                temps = [v_t1, v_t2, v_t3, v_nv[4], v_nv[5], v_nv[6]]
                add_ops.extend([("+", v_nv[i], v_tc2, temps[i]) for i in range(6)])
            emit_ops_packed(add_ops)

            # Groups 6-7 for all iterations
            eq67_ops = []
            mul67_ops = []
            add67_ops = []
            for it in range(n_iters):
                base_g = it * UNROLL
                v_nv = v_nv_all[it]
                eq67_ops.extend([("==", v_t1, all_idx[base_g+6], v_1), ("==", v_t2, all_idx[base_g+7], v_1)])
            emit_ops_packed(eq67_ops)

            for it in range(n_iters):
                v_nv = v_nv_all[it]
                mul67_ops.extend([("*", v_t1, v_td_1, v_t1), ("*", v_t2, v_td_1, v_t2)])
            emit_ops_packed(mul67_ops)

            for it in range(n_iters):
                v_nv = v_nv_all[it]
                add67_ops.extend([("+", v_nv[6], v_tc2, v_t1), ("+", v_nv[7], v_tc2, v_t2)])
            emit_ops_packed(add67_ops)

        elif rnd == 2:
            # Round 2 has a more complex dependency chain
            for local_g in range(UNROLL):
                eq_ops = []
                mul_ops = []
                add1_ops = []
                add2_ops = []
                final_ops = []
                for it in range(n_iters):
                    base_g = it * UNROLL
                    g = base_g + local_g
                    v_nv = v_nv_all[it]
                    eq_ops.extend([("==", v_t1, all_idx[g], v_const[4]), ("==", v_t2, all_idx[g], v_const[5]), ("==", v_t3, all_idx[g], v_const[6])])
                emit_ops_packed(eq_ops)

                for it in range(n_iters):
                    mul_ops.extend([("*", v_t1, v_td_4, v_t1), ("*", v_t2, v_td_5, v_t2), ("*", v_t3, v_td_6, v_t3)])
                emit_ops_packed(mul_ops)

                for it in range(n_iters):
                    add1_ops.append(("+", v_t1, v_t1, v_t2))
                emit_ops_packed(add1_ops)

                for it in range(n_iters):
                    add2_ops.append(("+", v_t1, v_t1, v_t3))
                emit_ops_packed(add2_ops)

                for it in range(n_iters):
                    v_nv = v_nv_all[it]
                    final_ops.append(("+", v_nv[local_g], v_tc3, v_t1))
                emit_ops_packed(final_ops)

        # Stage 2: XOR all iterations
        xor_ops = []
        for it in range(n_iters):
            base_g = it * UNROLL
            v_nv = v_nv_all[it]
            xor_ops.extend([("^", all_val[base_g+lg], all_val[base_g+lg], v_nv[lg]) for lg in range(UNROLL)])
        emit_ops_packed(xor_ops)

        # Stage 3: Hash - process each sub-stage for all iterations
        for hi, (op1, _, op2, op3, _) in enumerate(HASH_STAGES):
            # Step A: stepA for groups 0-5 (6 ops per iter, 4 iters = 24 ops = 4 bundles)
            stepA_05 = []
            for it in range(n_iters):
                base_g = it * UNROLL
                v_nv = v_nv_all[it]
                g, g2 = base_g, base_g + 2
                stepA_05.extend([
                    (op1, v_t1, all_val[g], v_hc[hi]),
                    (op3, v_t2, all_val[g], v_hs[hi]),
                    (op1, v_nv[0], all_val[g+1], v_hc[hi]),
                    (op3, v_nv[1], all_val[g+1], v_hs[hi]),
                    (op1, v_nv[2], all_val[g2], v_hc[hi]),
                    (op3, v_nv[3], all_val[g2], v_hs[hi]),
                ])
            emit_ops_packed(stepA_05)

            # Step B for groups 0-1 + step A for groups 3-4
            stepB_01_A_34 = []
            for it in range(n_iters):
                base_g = it * UNROLL
                v_nv = v_nv_all[it]
                g, g2 = base_g, base_g + 2
                stepB_01_A_34.extend([
                    (op2, all_val[g], v_t1, v_t2),
                    (op2, all_val[g+1], v_nv[0], v_nv[1]),
                    (op1, v_t1, all_val[g2+1], v_hc[hi]),
                    (op3, v_t2, all_val[g2+1], v_hs[hi]),
                    (op1, v_nv[4], all_val[base_g+4], v_hc[hi]),
                    (op3, v_nv[5], all_val[base_g+4], v_hs[hi]),
                ])
            emit_ops_packed(stepB_01_A_34)

            # Step B for groups 2-3 + step A for groups 5-6
            stepB_23_A_56 = []
            for it in range(n_iters):
                base_g = it * UNROLL
                v_nv = v_nv_all[it]
                g2 = base_g + 2
                stepB_23_A_56.extend([
                    (op2, all_val[g2], v_nv[2], v_nv[3]),
                    (op2, all_val[g2+1], v_t1, v_t2),
                    (op1, v_t1, all_val[base_g+5], v_hc[hi]),
                    (op3, v_t2, all_val[base_g+5], v_hs[hi]),
                    (op1, v_nv[6], all_val[base_g+6], v_hc[hi]),
                    (op3, v_nv[7], all_val[base_g+6], v_hs[hi]),
                ])
            emit_ops_packed(stepB_23_A_56)

            # Step B for groups 4-5 + step A for group 7
            stepB_45_A_7 = []
            for it in range(n_iters):
                base_g = it * UNROLL
                v_nv = v_nv_all[it]
                stepB_45_A_7.extend([
                    (op2, all_val[base_g+4], v_nv[4], v_nv[5]),
                    (op2, all_val[base_g+5], v_t1, v_t2),
                    (op1, v_t1, all_val[base_g+7], v_hc[hi]),
                    (op3, v_t2, all_val[base_g+7], v_hs[hi]),
                ])
            emit_ops_packed(stepB_45_A_7)

            # Step B for groups 6-7
            stepB_67 = []
            for it in range(n_iters):
                base_g = it * UNROLL
                v_nv = v_nv_all[it]
                stepB_67.extend([
                    (op2, all_val[base_g+6], v_nv[6], v_nv[7]),
                    (op2, all_val[base_g+7], v_t1, v_t2),
                ])
            emit_ops_packed(stepB_67)

        # Stage 4: Branch - process each step for all iterations
        # Step 1: Compute branch offset first (val & 1) + 1
        branch_and = []
        for it in range(n_iters):
            base_g = it * UNROLL
            v_nv = v_nv_all[it]
            g0, g1, g2 = base_g, base_g + 2, base_g + 4
            branch_and.extend([
                ("&", v_t1, all_val[g0], v_1), ("&", v_t2, all_val[g0+1], v_1),
                ("&", v_nv[0], all_val[g1], v_1), ("&", v_nv[1], all_val[g1+1], v_1),
                ("&", v_nv[2], all_val[g2], v_1), ("&", v_nv[3], all_val[g2+1], v_1),
            ])
        emit_ops_packed(branch_and)

        branch_add1 = []
        for it in range(n_iters):
            v_nv = v_nv_all[it]
            branch_add1.extend([
                ("+", v_t1, v_t1, v_1), ("+", v_t2, v_t2, v_1),
                ("+", v_nv[0], v_nv[0], v_1), ("+", v_nv[1], v_nv[1], v_1),
                ("+", v_nv[2], v_nv[2], v_1), ("+", v_nv[3], v_nv[3], v_1),
            ])
        emit_ops_packed(branch_add1)

        # Step 2: Fused multiply-add: idx = idx * 2 + branch (saves one op!)
        branch_fma = []
        for it in range(n_iters):
            base_g = it * UNROLL
            v_nv = v_nv_all[it]
            g0, g1, g2 = base_g, base_g + 2, base_g + 4
            branch_fma.extend([
                ("multiply_add", all_idx[g0], all_idx[g0], v_2, v_t1), ("multiply_add", all_idx[g0+1], all_idx[g0+1], v_2, v_t2),
                ("multiply_add", all_idx[g1], all_idx[g1], v_2, v_nv[0]), ("multiply_add", all_idx[g1+1], all_idx[g1+1], v_2, v_nv[1]),
                ("multiply_add", all_idx[g2], all_idx[g2], v_2, v_nv[2]), ("multiply_add", all_idx[g2+1], all_idx[g2+1], v_2, v_nv[3]),
            ])
        emit_ops_packed(branch_fma)

        branch_lt = []
        for it in range(n_iters):
            base_g = it * UNROLL
            v_nv = v_nv_all[it]
            g0, g1, g2 = base_g, base_g + 2, base_g + 4
            branch_lt.extend([
                ("<", v_t1, all_idx[g0], v_nn), ("<", v_t2, all_idx[g0+1], v_nn),
                ("<", v_nv[0], all_idx[g1], v_nn), ("<", v_nv[1], all_idx[g1+1], v_nn),
                ("<", v_nv[2], all_idx[g2], v_nn), ("<", v_nv[3], all_idx[g2+1], v_nn),
            ])
        emit_ops_packed(branch_lt)

        branch_final = []
        for it in range(n_iters):
            base_g = it * UNROLL
            v_nv = v_nv_all[it]
            g0, g1, g2 = base_g, base_g + 2, base_g + 4
            branch_final.extend([
                ("*", all_idx[g0], all_idx[g0], v_t1), ("*", all_idx[g0+1], all_idx[g0+1], v_t2),
                ("*", all_idx[g1], all_idx[g1], v_nv[0]), ("*", all_idx[g1+1], all_idx[g1+1], v_nv[1]),
                ("*", all_idx[g2], all_idx[g2], v_nv[2]), ("*", all_idx[g2+1], all_idx[g2+1], v_nv[3]),
            ])
        emit_ops_packed(branch_final)

        # Groups 6-7 branch (the "last pair") - use multiply_add fusion
        branch67_and = []
        for it in range(n_iters):
            base_g = it * UNROLL
            g3 = base_g + 6
            branch67_and.extend([
                ("&", v_t1, all_val[g3], v_1), ("&", v_t2, all_val[g3+1], v_1),
            ])
        emit_ops_packed(branch67_and)

        branch67_add1 = []
        for it in range(n_iters):
            branch67_add1.extend([("+", v_t1, v_t1, v_1), ("+", v_t2, v_t2, v_1)])
        emit_ops_packed(branch67_add1)

        branch67_fma = []
        for it in range(n_iters):
            base_g = it * UNROLL
            g3 = base_g + 6
            branch67_fma.extend([("multiply_add", all_idx[g3], all_idx[g3], v_2, v_t1), ("multiply_add", all_idx[g3+1], all_idx[g3+1], v_2, v_t2)])
        emit_ops_packed(branch67_fma)

        branch67_lt = []
        for it in range(n_iters):
            base_g = it * UNROLL
            g3 = base_g + 6
            branch67_lt.extend([("<", v_t1, all_idx[g3], v_nn), ("<", v_t2, all_idx[g3+1], v_nn)])
        emit_ops_packed(branch67_lt)

        branch67_final = []
        for it in range(n_iters):
            base_g = it * UNROLL
            g3 = base_g + 6
            branch67_final.extend([("*", all_idx[g3], all_idx[g3], v_t1), ("*", all_idx[g3+1], all_idx[g3+1], v_t2)])
        emit_ops_packed(branch67_final)

    def _build_cached_round_valu(self, rnd, base_g, UNROLL, all_idx, all_val, tc,
                                  v_nv, v_t1, v_t2, v_t3, v_hc, v_hs,
                                  v_0, v_1, v_2, v_nn, v_const, v_td_1, v_tc2,
                                  v_tc3, v_td_4, v_td_5, v_td_6):
        """Build VALU ops for cached round (returns list of op bundles)."""
        valu_ops = []

        if rnd == 0:
            valu_ops.append([("vbroadcast", v_nv[lg], tc[0]) for lg in range(6)])
            valu_ops.append([("vbroadcast", v_nv[lg], tc[0]) for lg in range(6, UNROLL)])
        elif rnd == 1:
            # PACKED: groups 0-5 use v_t1,v_t2,v_t3,v_nv[4],v_nv[5],v_nv[6] as temps
            temps = [v_t1, v_t2, v_t3, v_nv[4], v_nv[5], v_nv[6]]
            valu_ops.append([("==", temps[i], all_idx[base_g+i], v_1) for i in range(6)])
            valu_ops.append([("*", temps[i], v_td_1, temps[i]) for i in range(6)])
            valu_ops.append([("+", v_nv[i], v_tc2, temps[i]) for i in range(6)])
            # Groups 6-7
            valu_ops.append([("==", v_t1, all_idx[base_g+6], v_1), ("==", v_t2, all_idx[base_g+7], v_1)])
            valu_ops.append([("*", v_t1, v_td_1, v_t1), ("*", v_t2, v_td_1, v_t2)])
            valu_ops.append([("+", v_nv[6], v_tc2, v_t1), ("+", v_nv[7], v_tc2, v_t2)])
        elif rnd == 2:
            for local_g in range(UNROLL):
                g = base_g + local_g
                valu_ops.append([("==", v_t1, all_idx[g], v_const[4]), ("==", v_t2, all_idx[g], v_const[5]), ("==", v_t3, all_idx[g], v_const[6])])
                valu_ops.append([("*", v_t1, v_td_4, v_t1), ("*", v_t2, v_td_5, v_t2), ("*", v_t3, v_td_6, v_t3)])
                valu_ops.append([("+", v_t1, v_t1, v_t2)])
                valu_ops.append([("+", v_t1, v_t1, v_t3)])
                valu_ops.append([("+", v_nv[local_g], v_tc3, v_t1)])

        # XOR
        valu_ops.append([("^", all_val[base_g+lg], all_val[base_g+lg], v_nv[lg]) for lg in range(6)])
        valu_ops.append([("^", all_val[base_g+lg], all_val[base_g+lg], v_nv[lg]) for lg in range(6, UNROLL)])

        # Hash
        for hi, (op1, _, op2, op3, _) in enumerate(HASH_STAGES):
            g = base_g
            g2 = base_g + 2

            valu_ops.append([
                (op1, v_t1, all_val[g], v_hc[hi]),
                (op3, v_t2, all_val[g], v_hs[hi]),
                (op1, v_nv[0], all_val[g+1], v_hc[hi]),
                (op3, v_nv[1], all_val[g+1], v_hs[hi]),
                (op1, v_nv[2], all_val[g2], v_hc[hi]),
                (op3, v_nv[3], all_val[g2], v_hs[hi]),
            ])

            valu_ops.append([
                (op2, all_val[g], v_t1, v_t2),
                (op2, all_val[g+1], v_nv[0], v_nv[1]),
                (op1, v_t1, all_val[g2+1], v_hc[hi]),
                (op3, v_t2, all_val[g2+1], v_hs[hi]),
                (op1, v_nv[4], all_val[base_g+4], v_hc[hi]),
                (op3, v_nv[5], all_val[base_g+4], v_hs[hi]),
            ])

            valu_ops.append([
                (op2, all_val[g2], v_nv[2], v_nv[3]),
                (op2, all_val[g2+1], v_t1, v_t2),
                (op1, v_t1, all_val[base_g+5], v_hc[hi]),
                (op3, v_t2, all_val[base_g+5], v_hs[hi]),
                (op1, v_nv[6], all_val[base_g+6], v_hc[hi]),
                (op3, v_nv[7], all_val[base_g+6], v_hs[hi]),
            ])

            valu_ops.append([
                (op2, all_val[base_g+4], v_nv[4], v_nv[5]),
                (op2, all_val[base_g+5], v_t1, v_t2),
                (op1, v_t1, all_val[base_g+7], v_hc[hi]),
                (op3, v_t2, all_val[base_g+7], v_hs[hi]),
            ])

            valu_ops.append([
                (op2, all_val[base_g+6], v_nv[6], v_nv[7]),
                (op2, all_val[base_g+7], v_t1, v_t2),
            ])

        # Branch - WIDE (using multiply_add fusion)
        g0, g1, g2, g3 = base_g, base_g + 2, base_g + 4, base_g + 6

        # Compute branch offset first
        valu_ops.append([
            ("&", v_t1, all_val[g0], v_1), ("&", v_t2, all_val[g0+1], v_1),
            ("&", v_nv[0], all_val[g1], v_1), ("&", v_nv[1], all_val[g1+1], v_1),
            ("&", v_nv[2], all_val[g2], v_1), ("&", v_nv[3], all_val[g2+1], v_1),
        ])
        valu_ops.append([
            ("+", v_t1, v_t1, v_1), ("+", v_t2, v_t2, v_1),
            ("+", v_nv[0], v_nv[0], v_1), ("+", v_nv[1], v_nv[1], v_1),
            ("+", v_nv[2], v_nv[2], v_1), ("+", v_nv[3], v_nv[3], v_1),
        ])
        # Fused multiply-add: idx = idx * 2 + branch
        valu_ops.append([
            ("multiply_add", all_idx[g0], all_idx[g0], v_2, v_t1), ("multiply_add", all_idx[g0+1], all_idx[g0+1], v_2, v_t2),
            ("multiply_add", all_idx[g1], all_idx[g1], v_2, v_nv[0]), ("multiply_add", all_idx[g1+1], all_idx[g1+1], v_2, v_nv[1]),
            ("multiply_add", all_idx[g2], all_idx[g2], v_2, v_nv[2]), ("multiply_add", all_idx[g2+1], all_idx[g2+1], v_2, v_nv[3]),
        ])
        valu_ops.append([
            ("<", v_t1, all_idx[g0], v_nn), ("<", v_t2, all_idx[g0+1], v_nn),
            ("<", v_nv[0], all_idx[g1], v_nn), ("<", v_nv[1], all_idx[g1+1], v_nn),
            ("<", v_nv[2], all_idx[g2], v_nn), ("<", v_nv[3], all_idx[g2+1], v_nn),
        ])
        valu_ops.append([
            ("*", all_idx[g0], all_idx[g0], v_t1), ("*", all_idx[g0+1], all_idx[g0+1], v_t2),
            ("*", all_idx[g1], all_idx[g1], v_nv[0]), ("*", all_idx[g1+1], all_idx[g1+1], v_nv[1]),
            ("*", all_idx[g2], all_idx[g2], v_nv[2]), ("*", all_idx[g2+1], all_idx[g2+1], v_nv[3]),
        ])

        # Pair 3 (using multiply_add fusion)
        valu_ops.append([("&", v_t1, all_val[g3], v_1), ("&", v_t2, all_val[g3+1], v_1)])
        valu_ops.append([("+", v_t1, v_t1, v_1), ("+", v_t2, v_t2, v_1)])
        valu_ops.append([("multiply_add", all_idx[g3], all_idx[g3], v_2, v_t1), ("multiply_add", all_idx[g3+1], all_idx[g3+1], v_2, v_t2)])
        valu_ops.append([("<", v_t1, all_idx[g3], v_nn), ("<", v_t2, all_idx[g3+1], v_nn)])
        valu_ops.append([("*", all_idx[g3], all_idx[g3], v_t1), ("*", all_idx[g3+1], all_idx[g3+1], v_t2)])

        return valu_ops

    def _emit_cached_round(self, rnd, base_g, UNROLL, all_idx, all_val, tc,
                           v_nv, v_t1, v_t2, v_t3, v_hc, v_hs,
                           v_0, v_1, v_2, v_nn, v_const, v_td_1, v_tc2,
                           v_tc3, v_td_4, v_td_5, v_td_6):
        """Cached round with maximum VALU packing."""
        if rnd == 0:
            self.emit(valu=[("vbroadcast", v_nv[lg], tc[0]) for lg in range(6)])
            self.emit(valu=[("vbroadcast", v_nv[lg], tc[0]) for lg in range(6, UNROLL)])
        elif rnd == 1:
            # PACKED: groups 0-5 use v_t1,v_t2,v_t3,v_nv[4],v_nv[5],v_nv[6] as temps
            temps = [v_t1, v_t2, v_t3, v_nv[4], v_nv[5], v_nv[6]]
            self.emit(valu=[("==", temps[i], all_idx[base_g+i], v_1) for i in range(6)])
            self.emit(valu=[("*", temps[i], v_td_1, temps[i]) for i in range(6)])
            self.emit(valu=[("+", v_nv[i], v_tc2, temps[i]) for i in range(6)])
            # Groups 6-7 use v_t1, v_t2
            self.emit(valu=[("==", v_t1, all_idx[base_g+6], v_1), ("==", v_t2, all_idx[base_g+7], v_1)])
            self.emit(valu=[("*", v_t1, v_td_1, v_t1), ("*", v_t2, v_td_1, v_t2)])
            self.emit(valu=[("+", v_nv[6], v_tc2, v_t1), ("+", v_nv[7], v_tc2, v_t2)])
        elif rnd == 2:
            for local_g in range(UNROLL):
                g = base_g + local_g
                self.emit(valu=[("==", v_t1, all_idx[g], v_const[4]), ("==", v_t2, all_idx[g], v_const[5]), ("==", v_t3, all_idx[g], v_const[6])])
                self.emit(valu=[("*", v_t1, v_td_4, v_t1), ("*", v_t2, v_td_5, v_t2), ("*", v_t3, v_td_6, v_t3)])
                self.emit(valu=[("+", v_t1, v_t1, v_t2)])
                self.emit(valu=[("+", v_t1, v_t1, v_t3)])
                self.emit(valu=[("+", v_nv[local_g], v_tc3, v_t1)])

        # XOR
        self.emit(valu=[("^", all_val[base_g+lg], all_val[base_g+lg], v_nv[lg]) for lg in range(6)])
        self.emit(valu=[("^", all_val[base_g+lg], all_val[base_g+lg], v_nv[lg]) for lg in range(6, UNROLL)])

        # Hash
        for hi, (op1, _, op2, op3, _) in enumerate(HASH_STAGES):
            g = base_g
            g2 = base_g + 2

            self.emit(valu=[
                (op1, v_t1, all_val[g], v_hc[hi]),
                (op3, v_t2, all_val[g], v_hs[hi]),
                (op1, v_nv[0], all_val[g+1], v_hc[hi]),
                (op3, v_nv[1], all_val[g+1], v_hs[hi]),
                (op1, v_nv[2], all_val[g2], v_hc[hi]),
                (op3, v_nv[3], all_val[g2], v_hs[hi]),
            ])

            self.emit(valu=[
                (op2, all_val[g], v_t1, v_t2),
                (op2, all_val[g+1], v_nv[0], v_nv[1]),
                (op1, v_t1, all_val[g2+1], v_hc[hi]),
                (op3, v_t2, all_val[g2+1], v_hs[hi]),
                (op1, v_nv[4], all_val[base_g+4], v_hc[hi]),
                (op3, v_nv[5], all_val[base_g+4], v_hs[hi]),
            ])

            self.emit(valu=[
                (op2, all_val[g2], v_nv[2], v_nv[3]),
                (op2, all_val[g2+1], v_t1, v_t2),
                (op1, v_t1, all_val[base_g+5], v_hc[hi]),
                (op3, v_t2, all_val[base_g+5], v_hs[hi]),
                (op1, v_nv[6], all_val[base_g+6], v_hc[hi]),
                (op3, v_nv[7], all_val[base_g+6], v_hs[hi]),
            ])

            self.emit(valu=[
                (op2, all_val[base_g+4], v_nv[4], v_nv[5]),
                (op2, all_val[base_g+5], v_t1, v_t2),
                (op1, v_t1, all_val[base_g+7], v_hc[hi]),
                (op3, v_t2, all_val[base_g+7], v_hs[hi]),
            ])

            self.emit(valu=[
                (op2, all_val[base_g+6], v_nv[6], v_nv[7]),
                (op2, all_val[base_g+7], v_t1, v_t2),
            ])

        # Branch (using multiply_add fusion)
        # WIDE branch: 3 pairs at once
        g0, g1, g2, g3 = base_g, base_g + 2, base_g + 4, base_g + 6

        # Compute branch offset first
        self.emit(valu=[
            ("&", v_t1, all_val[g0], v_1), ("&", v_t2, all_val[g0+1], v_1),
            ("&", v_nv[0], all_val[g1], v_1), ("&", v_nv[1], all_val[g1+1], v_1),
            ("&", v_nv[2], all_val[g2], v_1), ("&", v_nv[3], all_val[g2+1], v_1),
        ])
        self.emit(valu=[
            ("+", v_t1, v_t1, v_1), ("+", v_t2, v_t2, v_1),
            ("+", v_nv[0], v_nv[0], v_1), ("+", v_nv[1], v_nv[1], v_1),
            ("+", v_nv[2], v_nv[2], v_1), ("+", v_nv[3], v_nv[3], v_1),
        ])
        # Fused multiply-add: idx = idx * 2 + branch
        self.emit(valu=[
            ("multiply_add", all_idx[g0], all_idx[g0], v_2, v_t1), ("multiply_add", all_idx[g0+1], all_idx[g0+1], v_2, v_t2),
            ("multiply_add", all_idx[g1], all_idx[g1], v_2, v_nv[0]), ("multiply_add", all_idx[g1+1], all_idx[g1+1], v_2, v_nv[1]),
            ("multiply_add", all_idx[g2], all_idx[g2], v_2, v_nv[2]), ("multiply_add", all_idx[g2+1], all_idx[g2+1], v_2, v_nv[3]),
        ])
        self.emit(valu=[
            ("<", v_t1, all_idx[g0], v_nn), ("<", v_t2, all_idx[g0+1], v_nn),
            ("<", v_nv[0], all_idx[g1], v_nn), ("<", v_nv[1], all_idx[g1+1], v_nn),
            ("<", v_nv[2], all_idx[g2], v_nn), ("<", v_nv[3], all_idx[g2+1], v_nn),
        ])
        self.emit(valu=[
            ("*", all_idx[g0], all_idx[g0], v_t1), ("*", all_idx[g0+1], all_idx[g0+1], v_t2),
            ("*", all_idx[g1], all_idx[g1], v_nv[0]), ("*", all_idx[g1+1], all_idx[g1+1], v_nv[1]),
            ("*", all_idx[g2], all_idx[g2], v_nv[2]), ("*", all_idx[g2+1], all_idx[g2+1], v_nv[3]),
        ])

        # Pair 3 (last pair, using multiply_add fusion)
        self.emit(valu=[("&", v_t1, all_val[g3], v_1), ("&", v_t2, all_val[g3+1], v_1)])
        self.emit(valu=[("+", v_t1, v_t1, v_1), ("+", v_t2, v_t2, v_1)])
        self.emit(valu=[("multiply_add", all_idx[g3], all_idx[g3], v_2, v_t1), ("multiply_add", all_idx[g3+1], all_idx[g3+1], v_2, v_t2)])
        self.emit(valu=[("<", v_t1, all_idx[g3], v_nn), ("<", v_t2, all_idx[g3+1], v_nn)])
        self.emit(valu=[("*", all_idx[g3], all_idx[g3], v_t1), ("*", all_idx[g3+1], all_idx[g3+1], v_t2)])

    def _emit_cached_with_gather(self, rnd, base_g, UNROLL, all_idx, all_val, tc,
                                   v_nv_proc, v_nv_gather, v_t1, v_t2, v_t3, v_hc, v_hs,
                                   v_0, v_1, v_2, v_nn, v_const, v_td_1, v_tc2,
                                   v_tc3, v_td_4, v_td_5, v_td_6, forest_p, st):
        """Cached round with gather interleaved - maximum utilization."""
        # Build VALU ops for this cached round
        valu_ops = []

        if rnd == 0:
            valu_ops.append([("vbroadcast", v_nv_proc[lg], tc[0]) for lg in range(6)])
            valu_ops.append([("vbroadcast", v_nv_proc[lg], tc[0]) for lg in range(6, UNROLL)])
        elif rnd == 1:
            for local_g in range(0, UNROLL, 2):
                g = base_g + local_g
                valu_ops.append([("==", v_t1, all_idx[g], v_1), ("==", v_t2, all_idx[g+1], v_1)])
                valu_ops.append([("*", v_t1, v_td_1, v_t1), ("*", v_t2, v_td_1, v_t2)])
                valu_ops.append([("+", v_nv_proc[local_g], v_tc2, v_t1), ("+", v_nv_proc[local_g+1], v_tc2, v_t2)])
        elif rnd == 2:
            for local_g in range(UNROLL):
                g = base_g + local_g
                valu_ops.append([("==", v_t1, all_idx[g], v_const[4]), ("==", v_t2, all_idx[g], v_const[5]), ("==", v_t3, all_idx[g], v_const[6])])
                valu_ops.append([("*", v_t1, v_td_4, v_t1), ("*", v_t2, v_td_5, v_t2), ("*", v_t3, v_td_6, v_t3)])
                valu_ops.append([("+", v_t1, v_t1, v_t2)])
                valu_ops.append([("+", v_t1, v_t1, v_t3)])
                valu_ops.append([("+", v_nv_proc[local_g], v_tc3, v_t1)])

        # XOR
        valu_ops.append([("^", all_val[base_g+lg], all_val[base_g+lg], v_nv_proc[lg]) for lg in range(6)])
        valu_ops.append([("^", all_val[base_g+lg], all_val[base_g+lg], v_nv_proc[lg]) for lg in range(6, UNROLL)])

        # Hash
        for hi, (op1, _, op2, op3, _) in enumerate(HASH_STAGES):
            g = base_g
            g2 = base_g + 2

            valu_ops.append([
                (op1, v_t1, all_val[g], v_hc[hi]),
                (op3, v_t2, all_val[g], v_hs[hi]),
                (op1, v_nv_proc[0], all_val[g+1], v_hc[hi]),
                (op3, v_nv_proc[1], all_val[g+1], v_hs[hi]),
                (op1, v_nv_proc[2], all_val[g2], v_hc[hi]),
                (op3, v_nv_proc[3], all_val[g2], v_hs[hi]),
            ])

            valu_ops.append([
                (op2, all_val[g], v_t1, v_t2),
                (op2, all_val[g+1], v_nv_proc[0], v_nv_proc[1]),
                (op1, v_t1, all_val[g2+1], v_hc[hi]),
                (op3, v_t2, all_val[g2+1], v_hs[hi]),
                (op1, v_nv_proc[4], all_val[base_g+4], v_hc[hi]),
                (op3, v_nv_proc[5], all_val[base_g+4], v_hs[hi]),
            ])

            valu_ops.append([
                (op2, all_val[g2], v_nv_proc[2], v_nv_proc[3]),
                (op2, all_val[g2+1], v_t1, v_t2),
                (op1, v_t1, all_val[base_g+5], v_hc[hi]),
                (op3, v_t2, all_val[base_g+5], v_hs[hi]),
                (op1, v_nv_proc[6], all_val[base_g+6], v_hc[hi]),
                (op3, v_nv_proc[7], all_val[base_g+6], v_hs[hi]),
            ])

            valu_ops.append([
                (op2, all_val[base_g+4], v_nv_proc[4], v_nv_proc[5]),
                (op2, all_val[base_g+5], v_t1, v_t2),
                (op1, v_t1, all_val[base_g+7], v_hc[hi]),
                (op3, v_t2, all_val[base_g+7], v_hs[hi]),
            ])

            valu_ops.append([
                (op2, all_val[base_g+6], v_nv_proc[6], v_nv_proc[7]),
                (op2, all_val[base_g+7], v_t1, v_t2),
            ])

        # Branch - WIDE (using multiply_add fusion)
        g0, g1, g2, g3 = base_g, base_g + 2, base_g + 4, base_g + 6

        # Compute branch offset first
        valu_ops.append([
            ("&", v_t1, all_val[g0], v_1), ("&", v_t2, all_val[g0+1], v_1),
            ("&", v_nv_proc[0], all_val[g1], v_1), ("&", v_nv_proc[1], all_val[g1+1], v_1),
            ("&", v_nv_proc[2], all_val[g2], v_1), ("&", v_nv_proc[3], all_val[g2+1], v_1),
        ])
        valu_ops.append([
            ("+", v_t1, v_t1, v_1), ("+", v_t2, v_t2, v_1),
            ("+", v_nv_proc[0], v_nv_proc[0], v_1), ("+", v_nv_proc[1], v_nv_proc[1], v_1),
            ("+", v_nv_proc[2], v_nv_proc[2], v_1), ("+", v_nv_proc[3], v_nv_proc[3], v_1),
        ])
        # Fused multiply-add: idx = idx * 2 + branch
        valu_ops.append([
            ("multiply_add", all_idx[g0], all_idx[g0], v_2, v_t1), ("multiply_add", all_idx[g0+1], all_idx[g0+1], v_2, v_t2),
            ("multiply_add", all_idx[g1], all_idx[g1], v_2, v_nv_proc[0]), ("multiply_add", all_idx[g1+1], all_idx[g1+1], v_2, v_nv_proc[1]),
            ("multiply_add", all_idx[g2], all_idx[g2], v_2, v_nv_proc[2]), ("multiply_add", all_idx[g2+1], all_idx[g2+1], v_2, v_nv_proc[3]),
        ])
        valu_ops.append([
            ("<", v_t1, all_idx[g0], v_nn), ("<", v_t2, all_idx[g0+1], v_nn),
            ("<", v_nv_proc[0], all_idx[g1], v_nn), ("<", v_nv_proc[1], all_idx[g1+1], v_nn),
            ("<", v_nv_proc[2], all_idx[g2], v_nn), ("<", v_nv_proc[3], all_idx[g2+1], v_nn),
        ])
        valu_ops.append([
            ("*", all_idx[g0], all_idx[g0], v_t1), ("*", all_idx[g0+1], all_idx[g0+1], v_t2),
            ("*", all_idx[g1], all_idx[g1], v_nv_proc[0]), ("*", all_idx[g1+1], all_idx[g1+1], v_nv_proc[1]),
            ("*", all_idx[g2], all_idx[g2], v_nv_proc[2]), ("*", all_idx[g2+1], all_idx[g2+1], v_nv_proc[3]),
        ])

        # Pair 3 (using multiply_add fusion)
        valu_ops.append([("&", v_t1, all_val[g3], v_1), ("&", v_t2, all_val[g3+1], v_1)])
        valu_ops.append([("+", v_t1, v_t1, v_1), ("+", v_t2, v_t2, v_1)])
        valu_ops.append([("multiply_add", all_idx[g3], all_idx[g3], v_2, v_t1), ("multiply_add", all_idx[g3+1], all_idx[g3+1], v_2, v_t2)])
        valu_ops.append([("<", v_t1, all_idx[g3], v_nn), ("<", v_t2, all_idx[g3+1], v_nn)])
        valu_ops.append([("*", all_idx[g3], all_idx[g3], v_t1), ("*", all_idx[g3+1], all_idx[g3+1], v_t2)])

        # Build gather ops
        gather_alu = []
        gather_load = []

        for vi in range(VLEN):
            for local_g in range(0, UNROLL, 2):
                gather_alu.append([
                    ("+", st[0], forest_p, all_idx[base_g + local_g] + vi),
                    ("+", st[1], forest_p, all_idx[base_g + local_g + 1] + vi)
                ])
                gather_load.append((local_g, vi))

        # Interleave: ALU + VALU, then Load + ALU + VALU
        n_gather = len(gather_alu)
        v_idx = 0

        # First bundle: ALU + VALU
        bundle = {'alu': gather_alu[0]}
        if v_idx < len(valu_ops):
            bundle['valu'] = valu_ops[v_idx]
            v_idx += 1
        self.emit(**bundle)

        # Remaining bundles: Load + ALU + VALU
        for i in range(1, n_gather):
            local_g, vi = gather_load[i-1]
            bundle = {
                'load': [("load", v_nv_gather[local_g] + vi, st[0]), ("load", v_nv_gather[local_g + 1] + vi, st[1])],
                'alu': gather_alu[i]
            }
            if v_idx < len(valu_ops):
                bundle['valu'] = valu_ops[v_idx]
                v_idx += 1
            self.emit(**bundle)

        # Final load
        local_g, vi = gather_load[n_gather-1]
        bundle = {'load': [("load", v_nv_gather[local_g] + vi, st[0]), ("load", v_nv_gather[local_g + 1] + vi, st[1])]}
        if v_idx < len(valu_ops):
            bundle['valu'] = valu_ops[v_idx]
            v_idx += 1
        self.emit(**bundle)

        # Any remaining VALU ops
        while v_idx < len(valu_ops):
            self.emit(valu=valu_ops[v_idx])
            v_idx += 1

    def _emit_wide_gather_and_process(self, curr_bases, prev_base, UNROLL, all_idx, all_val, forest_p,
                                        v_nv_curr_list, v_nv_prev, v_t1, v_t2, v_hc, v_hs, v_0, v_1, v_2, v_nn, st):
        """Gather 2 iters + Process 1 iter - maximizes Load slot utilization."""
        # Build gather ops for 2 iterations
        gather_alu = []
        gather_load = []

        for curr_idx, (curr_base, v_nv_curr) in enumerate(zip(curr_bases, v_nv_curr_list)):
            for vi in range(VLEN):
                for local_g in range(0, UNROLL, 2):
                    gather_alu.append([
                        ("+", st[0], forest_p, all_idx[curr_base + local_g] + vi),
                        ("+", st[1], forest_p, all_idx[curr_base + local_g + 1] + vi)
                    ])
                    gather_load.append((curr_idx, local_g, vi))

        # VALU ops for processing (same as before)
        valu_ops = []

        # XOR
        valu_ops.append([("^", all_val[prev_base+lg], all_val[prev_base+lg], v_nv_prev[lg]) for lg in range(6)])
        valu_ops.append([("^", all_val[prev_base+lg], all_val[prev_base+lg], v_nv_prev[lg]) for lg in range(6, UNROLL)])

        # Hash (max packed)
        for hi, (op1, _, op2, op3, _) in enumerate(HASH_STAGES):
            g = prev_base
            g2 = prev_base + 2

            valu_ops.append([
                (op1, v_t1, all_val[g], v_hc[hi]),
                (op3, v_t2, all_val[g], v_hs[hi]),
                (op1, v_nv_prev[0], all_val[g+1], v_hc[hi]),
                (op3, v_nv_prev[1], all_val[g+1], v_hs[hi]),
                (op1, v_nv_prev[2], all_val[g2], v_hc[hi]),
                (op3, v_nv_prev[3], all_val[g2], v_hs[hi]),
            ])

            valu_ops.append([
                (op2, all_val[g], v_t1, v_t2),
                (op2, all_val[g+1], v_nv_prev[0], v_nv_prev[1]),
                (op1, v_t1, all_val[g2+1], v_hc[hi]),
                (op3, v_t2, all_val[g2+1], v_hs[hi]),
                (op1, v_nv_prev[4], all_val[prev_base+4], v_hc[hi]),
                (op3, v_nv_prev[5], all_val[prev_base+4], v_hs[hi]),
            ])

            valu_ops.append([
                (op2, all_val[g2], v_nv_prev[2], v_nv_prev[3]),
                (op2, all_val[g2+1], v_t1, v_t2),
                (op1, v_t1, all_val[prev_base+5], v_hc[hi]),
                (op3, v_t2, all_val[prev_base+5], v_hs[hi]),
                (op1, v_nv_prev[6], all_val[prev_base+6], v_hc[hi]),
                (op3, v_nv_prev[7], all_val[prev_base+6], v_hs[hi]),
            ])

            valu_ops.append([
                (op2, all_val[prev_base+4], v_nv_prev[4], v_nv_prev[5]),
                (op2, all_val[prev_base+5], v_t1, v_t2),
                (op1, v_t1, all_val[prev_base+7], v_hc[hi]),
                (op3, v_t2, all_val[prev_base+7], v_hs[hi]),
            ])

            valu_ops.append([
                (op2, all_val[prev_base+6], v_nv_prev[6], v_nv_prev[7]),
                (op2, all_val[prev_base+7], v_t1, v_t2),
            ])

        # Branch - WIDE (using multiply_add fusion)
        g0, g1, g2 = prev_base, prev_base + 2, prev_base + 4

        # Compute branch offset first
        valu_ops.append([
            ("&", v_t1, all_val[g0], v_1), ("&", v_t2, all_val[g0+1], v_1),
            ("&", v_nv_prev[0], all_val[g1], v_1), ("&", v_nv_prev[1], all_val[g1+1], v_1),
            ("&", v_nv_prev[2], all_val[g2], v_1), ("&", v_nv_prev[3], all_val[g2+1], v_1),
        ])
        valu_ops.append([
            ("+", v_t1, v_t1, v_1), ("+", v_t2, v_t2, v_1),
            ("+", v_nv_prev[0], v_nv_prev[0], v_1), ("+", v_nv_prev[1], v_nv_prev[1], v_1),
            ("+", v_nv_prev[2], v_nv_prev[2], v_1), ("+", v_nv_prev[3], v_nv_prev[3], v_1),
        ])
        # Fused multiply-add: idx = idx * 2 + branch
        valu_ops.append([
            ("multiply_add", all_idx[g0], all_idx[g0], v_2, v_t1), ("multiply_add", all_idx[g0+1], all_idx[g0+1], v_2, v_t2),
            ("multiply_add", all_idx[g1], all_idx[g1], v_2, v_nv_prev[0]), ("multiply_add", all_idx[g1+1], all_idx[g1+1], v_2, v_nv_prev[1]),
            ("multiply_add", all_idx[g2], all_idx[g2], v_2, v_nv_prev[2]), ("multiply_add", all_idx[g2+1], all_idx[g2+1], v_2, v_nv_prev[3]),
        ])
        valu_ops.append([
            ("<", v_t1, all_idx[g0], v_nn), ("<", v_t2, all_idx[g0+1], v_nn),
            ("<", v_nv_prev[0], all_idx[g1], v_nn), ("<", v_nv_prev[1], all_idx[g1+1], v_nn),
            ("<", v_nv_prev[2], all_idx[g2], v_nn), ("<", v_nv_prev[3], all_idx[g2+1], v_nn),
        ])
        valu_ops.append([
            ("*", all_idx[g0], all_idx[g0], v_t1), ("*", all_idx[g0+1], all_idx[g0+1], v_t2),
            ("*", all_idx[g1], all_idx[g1], v_nv_prev[0]), ("*", all_idx[g1+1], all_idx[g1+1], v_nv_prev[1]),
            ("*", all_idx[g2], all_idx[g2], v_nv_prev[2]), ("*", all_idx[g2+1], all_idx[g2+1], v_nv_prev[3]),
        ])

        # Pair 3 (using multiply_add fusion)
        g3 = prev_base + 6
        valu_ops.append([("&", v_t1, all_val[g3], v_1), ("&", v_t2, all_val[g3+1], v_1)])
        valu_ops.append([("+", v_t1, v_t1, v_1), ("+", v_t2, v_t2, v_1)])
        valu_ops.append([("multiply_add", all_idx[g3], all_idx[g3], v_2, v_t1), ("multiply_add", all_idx[g3+1], all_idx[g3+1], v_2, v_t2)])
        valu_ops.append([("<", v_t1, all_idx[g3], v_nn), ("<", v_t2, all_idx[g3+1], v_nn)])
        valu_ops.append([("*", all_idx[g3], all_idx[g3], v_t1), ("*", all_idx[g3+1], all_idx[g3+1], v_t2)])

        # Interleave: Load + ALU + VALU (now with 2x gather ops)
        n_gather = len(gather_alu)
        v_idx = 0

        bundle = {'alu': gather_alu[0]}
        if v_idx < len(valu_ops):
            bundle['valu'] = valu_ops[v_idx]
            v_idx += 1
        self.emit(**bundle)

        for i in range(1, n_gather):
            curr_idx, local_g, vi = gather_load[i-1]
            v_nv_curr = v_nv_curr_list[curr_idx]
            bundle = {
                'load': [("load", v_nv_curr[local_g] + vi, st[0]), ("load", v_nv_curr[local_g + 1] + vi, st[1])],
                'alu': gather_alu[i]
            }
            if v_idx < len(valu_ops):
                bundle['valu'] = valu_ops[v_idx]
                v_idx += 1
            self.emit(**bundle)

        curr_idx, local_g, vi = gather_load[n_gather-1]
        v_nv_curr = v_nv_curr_list[curr_idx]
        bundle = {'load': [("load", v_nv_curr[local_g] + vi, st[0]), ("load", v_nv_curr[local_g + 1] + vi, st[1])]}
        if v_idx < len(valu_ops):
            bundle['valu'] = valu_ops[v_idx]
            v_idx += 1
        self.emit(**bundle)

        while v_idx < len(valu_ops):
            self.emit(valu=valu_ops[v_idx])
            v_idx += 1

    def _emit_gather_only(self, base_g, UNROLL, all_idx, forest_p, v_nv, st):
        """Gather with Load+ALU packing."""
        gather_alu = []
        gather_load = []

        for vi in range(VLEN):
            for local_g in range(0, UNROLL, 2):
                gather_alu.append([
                    ("+", st[0], forest_p, all_idx[base_g + local_g] + vi),
                    ("+", st[1], forest_p, all_idx[base_g + local_g + 1] + vi)
                ])
                gather_load.append((local_g, vi))

        n = len(gather_alu)
        self.emit(alu=gather_alu[0])
        for i in range(1, n):
            local_g, vi = gather_load[i-1]
            self.emit(
                load=[("load", v_nv[local_g] + vi, st[0]), ("load", v_nv[local_g + 1] + vi, st[1])],
                alu=gather_alu[i]
            )
        local_g, vi = gather_load[n-1]
        self.emit(load=[("load", v_nv[local_g] + vi, st[0]), ("load", v_nv[local_g + 1] + vi, st[1])])

    def _emit_gather_and_process(self, curr_base, prev_base, UNROLL, all_idx, all_val, forest_p,
                                  v_nv_curr, v_nv_prev, v_t1, v_t2, v_hc, v_hs, v_0, v_1, v_2, v_nn, st):
        """Gather curr + Process prev with Load+ALU+VALU packing."""
        # Gather ops
        gather_alu = []
        gather_load = []

        for vi in range(VLEN):
            for local_g in range(0, UNROLL, 2):
                gather_alu.append([
                    ("+", st[0], forest_p, all_idx[curr_base + local_g] + vi),
                    ("+", st[1], forest_p, all_idx[curr_base + local_g + 1] + vi)
                ])
                gather_load.append((local_g, vi))

        # VALU ops
        valu_ops = []

        # XOR
        valu_ops.append([("^", all_val[prev_base+lg], all_val[prev_base+lg], v_nv_prev[lg]) for lg in range(6)])
        valu_ops.append([("^", all_val[prev_base+lg], all_val[prev_base+lg], v_nv_prev[lg]) for lg in range(6, UNROLL)])

        # Hash (max packed)
        for hi, (op1, _, op2, op3, _) in enumerate(HASH_STAGES):
            g = prev_base
            g2 = prev_base + 2

            valu_ops.append([
                (op1, v_t1, all_val[g], v_hc[hi]),
                (op3, v_t2, all_val[g], v_hs[hi]),
                (op1, v_nv_prev[0], all_val[g+1], v_hc[hi]),
                (op3, v_nv_prev[1], all_val[g+1], v_hs[hi]),
                (op1, v_nv_prev[2], all_val[g2], v_hc[hi]),
                (op3, v_nv_prev[3], all_val[g2], v_hs[hi]),
            ])

            valu_ops.append([
                (op2, all_val[g], v_t1, v_t2),
                (op2, all_val[g+1], v_nv_prev[0], v_nv_prev[1]),
                (op1, v_t1, all_val[g2+1], v_hc[hi]),
                (op3, v_t2, all_val[g2+1], v_hs[hi]),
                (op1, v_nv_prev[4], all_val[prev_base+4], v_hc[hi]),
                (op3, v_nv_prev[5], all_val[prev_base+4], v_hs[hi]),
            ])

            valu_ops.append([
                (op2, all_val[g2], v_nv_prev[2], v_nv_prev[3]),
                (op2, all_val[g2+1], v_t1, v_t2),
                (op1, v_t1, all_val[prev_base+5], v_hc[hi]),
                (op3, v_t2, all_val[prev_base+5], v_hs[hi]),
                (op1, v_nv_prev[6], all_val[prev_base+6], v_hc[hi]),
                (op3, v_nv_prev[7], all_val[prev_base+6], v_hs[hi]),
            ])

            valu_ops.append([
                (op2, all_val[prev_base+4], v_nv_prev[4], v_nv_prev[5]),
                (op2, all_val[prev_base+5], v_t1, v_t2),
                (op1, v_t1, all_val[prev_base+7], v_hc[hi]),
                (op3, v_t2, all_val[prev_base+7], v_hs[hi]),
            ])

            valu_ops.append([
                (op2, all_val[prev_base+6], v_nv_prev[6], v_nv_prev[7]),
                (op2, all_val[prev_base+7], v_t1, v_t2),
            ])

        # Branch - WIDE: 3 pairs at once (using multiply_add fusion)
        # Pairs 0,1,2 use v_t1/v_t2 for pair 0, v_nv_prev[0]/[1] for pair 1, [2]/[3] for pair 2
        g0, g1, g2 = prev_base, prev_base + 2, prev_base + 4

        # Compute branch offset first
        valu_ops.append([
            ("&", v_t1, all_val[g0], v_1), ("&", v_t2, all_val[g0+1], v_1),
            ("&", v_nv_prev[0], all_val[g1], v_1), ("&", v_nv_prev[1], all_val[g1+1], v_1),
            ("&", v_nv_prev[2], all_val[g2], v_1), ("&", v_nv_prev[3], all_val[g2+1], v_1),
        ])
        valu_ops.append([
            ("+", v_t1, v_t1, v_1), ("+", v_t2, v_t2, v_1),
            ("+", v_nv_prev[0], v_nv_prev[0], v_1), ("+", v_nv_prev[1], v_nv_prev[1], v_1),
            ("+", v_nv_prev[2], v_nv_prev[2], v_1), ("+", v_nv_prev[3], v_nv_prev[3], v_1),
        ])
        # Fused multiply-add: idx = idx * 2 + branch
        valu_ops.append([
            ("multiply_add", all_idx[g0], all_idx[g0], v_2, v_t1), ("multiply_add", all_idx[g0+1], all_idx[g0+1], v_2, v_t2),
            ("multiply_add", all_idx[g1], all_idx[g1], v_2, v_nv_prev[0]), ("multiply_add", all_idx[g1+1], all_idx[g1+1], v_2, v_nv_prev[1]),
            ("multiply_add", all_idx[g2], all_idx[g2], v_2, v_nv_prev[2]), ("multiply_add", all_idx[g2+1], all_idx[g2+1], v_2, v_nv_prev[3]),
        ])
        valu_ops.append([
            ("<", v_t1, all_idx[g0], v_nn), ("<", v_t2, all_idx[g0+1], v_nn),
            ("<", v_nv_prev[0], all_idx[g1], v_nn), ("<", v_nv_prev[1], all_idx[g1+1], v_nn),
            ("<", v_nv_prev[2], all_idx[g2], v_nn), ("<", v_nv_prev[3], all_idx[g2+1], v_nn),
        ])
        valu_ops.append([
            ("*", all_idx[g0], all_idx[g0], v_t1), ("*", all_idx[g0+1], all_idx[g0+1], v_t2),
            ("*", all_idx[g1], all_idx[g1], v_nv_prev[0]), ("*", all_idx[g1+1], all_idx[g1+1], v_nv_prev[1]),
            ("*", all_idx[g2], all_idx[g2], v_nv_prev[2]), ("*", all_idx[g2+1], all_idx[g2+1], v_nv_prev[3]),
        ])

        # Pair 3 (groups 6-7, using multiply_add fusion)
        g3 = prev_base + 6
        valu_ops.append([("&", v_t1, all_val[g3], v_1), ("&", v_t2, all_val[g3+1], v_1)])
        valu_ops.append([("+", v_t1, v_t1, v_1), ("+", v_t2, v_t2, v_1)])
        valu_ops.append([("multiply_add", all_idx[g3], all_idx[g3], v_2, v_t1), ("multiply_add", all_idx[g3+1], all_idx[g3+1], v_2, v_t2)])
        valu_ops.append([("<", v_t1, all_idx[g3], v_nn), ("<", v_t2, all_idx[g3+1], v_nn)])
        valu_ops.append([("*", all_idx[g3], all_idx[g3], v_t1), ("*", all_idx[g3+1], all_idx[g3+1], v_t2)])

        # Interleave: Load + ALU + VALU
        n_gather = len(gather_alu)
        v_idx = 0

        bundle = {'alu': gather_alu[0]}
        if v_idx < len(valu_ops):
            bundle['valu'] = valu_ops[v_idx]
            v_idx += 1
        self.emit(**bundle)

        for i in range(1, n_gather):
            local_g, vi = gather_load[i-1]
            bundle = {
                'load': [("load", v_nv_curr[local_g] + vi, st[0]), ("load", v_nv_curr[local_g + 1] + vi, st[1])],
                'alu': gather_alu[i]
            }
            if v_idx < len(valu_ops):
                bundle['valu'] = valu_ops[v_idx]
                v_idx += 1
            self.emit(**bundle)

        local_g, vi = gather_load[n_gather-1]
        bundle = {'load': [("load", v_nv_curr[local_g] + vi, st[0]), ("load", v_nv_curr[local_g + 1] + vi, st[1])]}
        if v_idx < len(valu_ops):
            bundle['valu'] = valu_ops[v_idx]
            v_idx += 1
        self.emit(**bundle)

        while v_idx < len(valu_ops):
            self.emit(valu=valu_ops[v_idx])
            v_idx += 1

    def _emit_process_only(self, base_g, UNROLL, all_idx, all_val, v_nv, v_t1, v_t2, v_hc, v_hs, v_0, v_1, v_2, v_nn):
        """Process-only phase with WIDE packing."""
        self.emit(valu=[("^", all_val[base_g+lg], all_val[base_g+lg], v_nv[lg]) for lg in range(6)])
        self.emit(valu=[("^", all_val[base_g+lg], all_val[base_g+lg], v_nv[lg]) for lg in range(6, UNROLL)])

        for hi, (op1, _, op2, op3, _) in enumerate(HASH_STAGES):
            g = base_g
            g2 = base_g + 2

            self.emit(valu=[
                (op1, v_t1, all_val[g], v_hc[hi]),
                (op3, v_t2, all_val[g], v_hs[hi]),
                (op1, v_nv[0], all_val[g+1], v_hc[hi]),
                (op3, v_nv[1], all_val[g+1], v_hs[hi]),
                (op1, v_nv[2], all_val[g2], v_hc[hi]),
                (op3, v_nv[3], all_val[g2], v_hs[hi]),
            ])

            self.emit(valu=[
                (op2, all_val[g], v_t1, v_t2),
                (op2, all_val[g+1], v_nv[0], v_nv[1]),
                (op1, v_t1, all_val[g2+1], v_hc[hi]),
                (op3, v_t2, all_val[g2+1], v_hs[hi]),
                (op1, v_nv[4], all_val[base_g+4], v_hc[hi]),
                (op3, v_nv[5], all_val[base_g+4], v_hs[hi]),
            ])

            self.emit(valu=[
                (op2, all_val[g2], v_nv[2], v_nv[3]),
                (op2, all_val[g2+1], v_t1, v_t2),
                (op1, v_t1, all_val[base_g+5], v_hc[hi]),
                (op3, v_t2, all_val[base_g+5], v_hs[hi]),
                (op1, v_nv[6], all_val[base_g+6], v_hc[hi]),
                (op3, v_nv[7], all_val[base_g+6], v_hs[hi]),
            ])

            self.emit(valu=[
                (op2, all_val[base_g+4], v_nv[4], v_nv[5]),
                (op2, all_val[base_g+5], v_t1, v_t2),
                (op1, v_t1, all_val[base_g+7], v_hc[hi]),
                (op3, v_t2, all_val[base_g+7], v_hs[hi]),
            ])

            self.emit(valu=[
                (op2, all_val[base_g+6], v_nv[6], v_nv[7]),
                (op2, all_val[base_g+7], v_t1, v_t2),
            ])

        # WIDE branch (using multiply_add fusion): process 3 pairs at once using v_nv as extra temps
        # Pairs 0,1,2 (groups 0-5): use v_t1/v_t2 for pair 0, v_nv[0]/v_nv[1] for pair 1, v_nv[2]/v_nv[3] for pair 2
        g0, g1, g2 = base_g, base_g + 2, base_g + 4

        # Compute branch offset first
        self.emit(valu=[
            ("&", v_t1, all_val[g0], v_1), ("&", v_t2, all_val[g0+1], v_1),
            ("&", v_nv[0], all_val[g1], v_1), ("&", v_nv[1], all_val[g1+1], v_1),
            ("&", v_nv[2], all_val[g2], v_1), ("&", v_nv[3], all_val[g2+1], v_1),
        ])
        self.emit(valu=[
            ("+", v_t1, v_t1, v_1), ("+", v_t2, v_t2, v_1),
            ("+", v_nv[0], v_nv[0], v_1), ("+", v_nv[1], v_nv[1], v_1),
            ("+", v_nv[2], v_nv[2], v_1), ("+", v_nv[3], v_nv[3], v_1),
        ])
        # Fused multiply-add: idx = idx * 2 + branch
        self.emit(valu=[
            ("multiply_add", all_idx[g0], all_idx[g0], v_2, v_t1), ("multiply_add", all_idx[g0+1], all_idx[g0+1], v_2, v_t2),
            ("multiply_add", all_idx[g1], all_idx[g1], v_2, v_nv[0]), ("multiply_add", all_idx[g1+1], all_idx[g1+1], v_2, v_nv[1]),
            ("multiply_add", all_idx[g2], all_idx[g2], v_2, v_nv[2]), ("multiply_add", all_idx[g2+1], all_idx[g2+1], v_2, v_nv[3]),
        ])
        self.emit(valu=[
            ("<", v_t1, all_idx[g0], v_nn), ("<", v_t2, all_idx[g0+1], v_nn),
            ("<", v_nv[0], all_idx[g1], v_nn), ("<", v_nv[1], all_idx[g1+1], v_nn),
            ("<", v_nv[2], all_idx[g2], v_nn), ("<", v_nv[3], all_idx[g2+1], v_nn),
        ])
        self.emit(valu=[
            ("*", all_idx[g0], all_idx[g0], v_t1), ("*", all_idx[g0+1], all_idx[g0+1], v_t2),
            ("*", all_idx[g1], all_idx[g1], v_nv[0]), ("*", all_idx[g1+1], all_idx[g1+1], v_nv[1]),
            ("*", all_idx[g2], all_idx[g2], v_nv[2]), ("*", all_idx[g2+1], all_idx[g2+1], v_nv[3]),
        ])

        # Pair 3 (groups 6-7, using multiply_add fusion)
        g3 = base_g + 6
        self.emit(valu=[("&", v_t1, all_val[g3], v_1), ("&", v_t2, all_val[g3+1], v_1)])
        self.emit(valu=[("+", v_t1, v_t1, v_1), ("+", v_t2, v_t2, v_1)])
        self.emit(valu=[("multiply_add", all_idx[g3], all_idx[g3], v_2, v_t1), ("multiply_add", all_idx[g3+1], all_idx[g3+1], v_2, v_t2)])
        self.emit(valu=[("<", v_t1, all_idx[g3], v_nn), ("<", v_t2, all_idx[g3+1], v_nn)])
        self.emit(valu=[("*", all_idx[g3], all_idx[g3], v_t1), ("*", all_idx[g3+1], all_idx[g3+1], v_t2)])


def interleaved_kernel(forest_height: int, n_nodes: int, batch_size: int, rounds: int):
    builder = InterRoundKernelBuilder()
    builder.build_kernel(forest_height, n_nodes, batch_size, rounds)
    return builder.instrs, builder.debug_info()
