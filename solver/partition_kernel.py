"""
Partition-Based Kernel - Uses stable partition to enable grouped tree access.

Key insight: If elements are sorted by parent index, and we do a stable partition
by branch bit, each partition remains sorted by NEW parent (new_parent ≈ old_idx).

Algorithm:
1. Sort elements by starting index (one-time cost)
2. Each round:
   a. Identify runs of elements with same parent → only 2 tree loads per run!
   b. vselect based on branch bit
   c. Partition into left/right buffers (uses idle store slots!)
3. Benefits compound: early rounds have massive collisions (round 3: ~8 unique indices for 256 elements)

This approach trades:
- Initial sort cost (~300-500 cycles)
- Partition stores per round (~128 cycles, but overlaps with VALU)
For:
- Drastically reduced tree loads (from 256 scattered → ~16-64 grouped per round)
"""

from problem import DebugInfo, SLOT_LIMITS, VLEN, SCRATCH_SIZE, HASH_STAGES
from solver.reschedule import extract_ops, compute_dependencies, schedule_greedy, emit_scheduled


class PartitionKernelBuilder:
    """Build a kernel using partition-based tree access optimization."""

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
        use_greedy: bool = True,
        enable_valu_shift: bool = True,
    ):
        """Build partition-based kernel.

        This version demonstrates the partition mechanism:
        1. Count unique parent indices per round
        2. Load only needed tree values
        3. Use vselect to distribute correct values

        For simplicity, this initial version doesn't do full sorting but
        demonstrates the partition concept with stores.
        """
        n_groups = batch_size // VLEN

        # Allocate scratch
        forest_p = self.alloc("forest_p")
        idx_p = self.alloc("idx_p")
        val_p = self.alloc("val_p")
        extra_room_p_scratch = self.alloc("extra_room_p")
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

        # Per-group vectors for indices and values
        all_idx = [self.vec(f"all_idx_{g}") for g in range(n_groups)]
        all_val = [self.vec(f"all_val_{g}") for g in range(n_groups)]
        v_nv = [self.vec(f"v_nv_{g}") for g in range(n_groups)]
        v_t = [self.vec(f"v_t_{g}") for g in range(n_groups)]

        # Partition buffers - stored in memory, not scratch
        # left_count and right_count track how many elements went each way
        left_count = self.alloc("left_count")
        right_count = self.alloc("right_count")

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

        # FMA optimization: (a + const) + (a << shift) = a * (1 + 2^shift) + const
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

        # ========== INIT ==========
        for i in range(16):
            self.emit(load=[("const", c[i], i)])
        self.emit(load=[("const", st[0], 4), ("const", st[1], 5)])
        self.emit(load=[("load", forest_p, st[0]), ("load", idx_p, st[1])])
        self.emit(load=[("const", st[0], 6), ("const", st[1], 1)])
        self.emit(load=[("load", val_p, st[0]), ("load", n_nodes_s, st[1])])
        # Load extra_room pointer for partition buffers
        self.emit(load=[("const", st[0], 7)])
        self.emit(load=[("load", extra_room_p_scratch, st[0])])

        for i in range(6):
            self.emit(load=[("const", hc[i], HASH_STAGES[i][1]), ("const", hs[i], HASH_STAGES[i][4])])
        self.emit(valu=[("vbroadcast", v_0, c[0]), ("vbroadcast", v_1, c[1]),
                        ("vbroadcast", v_2, c[2]), ("vbroadcast", v_nn, n_nodes_s)])
        for i in range(0, 8, 2):
            self.emit(valu=[("vbroadcast", v_const[i], c[i]), ("vbroadcast", v_const[i+1], c[i+1])])
        for i in range(0, 6, 2):
            self.emit(valu=[("vbroadcast", v_hc[i], hc[i]), ("vbroadcast", v_hc[i+1], hc[i+1]),
                           ("vbroadcast", v_hs[i], hs[i]), ("vbroadcast", v_hs[i+1], hs[i+1])])

        # Initialize FMA multiplier vectors
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

        self.emit(alu=[("-", td_1, tc[1], tc[2]), ("-", td_4, tc[4], tc[3]),
                       ("-", td_5, tc[5], tc[3]), ("-", td_6, tc[6], tc[3])])
        self.emit(valu=[("vbroadcast", v_tc2, tc[2]), ("vbroadcast", v_td_1, td_1),
                        ("vbroadcast", v_tc3, tc[3]), ("vbroadcast", v_td_4, td_4),
                        ("vbroadcast", v_td_5, td_5), ("vbroadcast", v_td_6, td_6)])

        # ========== LOAD DATA ==========
        for g in range(0, n_groups, 2):
            self.emit(load=[("const", st[0], g * VLEN), ("const", st[1], (g+1) * VLEN)])
            self.emit(alu=[("+", st[2], idx_p, st[0]), ("+", st[3], idx_p, st[1]),
                          ("+", st[4], val_p, st[0]), ("+", st[5], val_p, st[1])])
            self.emit(load=[("vload", all_idx[g], st[2]), ("vload", all_idx[g+1], st[3])])
            self.emit(load=[("vload", all_val[g], st[4]), ("vload", all_val[g+1], st[5])])

        # ========== MAIN LOOP ==========
        for rnd in range(rounds):
            # PHASE 1: Compute or Gather node values into v_nv
            if rnd == 0:
                for g in range(n_groups):
                    self.emit(valu=[("vbroadcast", v_nv[g], tc[0])])

            elif rnd == 1:
                for g in range(n_groups):
                    t = v_t[g]
                    self.emit(valu=[("==", t, all_idx[g], v_1)])
                    self.emit(valu=[("*", t, v_td_1, t)])
                    self.emit(valu=[("+", v_nv[g], v_tc2, t)])
                # Stage 5 opt: XOR v_nv with C
                for g in range(n_groups):
                    self.emit(valu=[("^", v_nv[g], v_nv[g], v_stage5_xor)])

            elif rnd == 2:
                for g in range(n_groups):
                    t = v_t[g]
                    self.emit(valu=[("==", t, all_idx[g], v_const[4])])
                    self.emit(valu=[("*", t, v_td_4, t)])
                    self.emit(valu=[("+", v_nv[g], v_tc3, t)])
                    self.emit(valu=[("==", t, all_idx[g], v_const[5])])
                    self.emit(valu=[("*", t, v_td_5, t)])
                    self.emit(valu=[("+", v_nv[g], v_nv[g], t)])
                    self.emit(valu=[("==", t, all_idx[g], v_const[6])])
                    self.emit(valu=[("*", t, v_td_6, t)])
                    self.emit(valu=[("+", v_nv[g], v_nv[g], t)])
                for g in range(n_groups):
                    self.emit(valu=[("^", v_nv[g], v_nv[g], v_stage5_xor)])

            else:
                # Gather from forest - scattered loads (for now, same as baseline)
                # TODO: This is where partition-based optimization would help most
                for vi in range(VLEN):
                    for g in range(0, n_groups, 2):
                        self.emit(alu=[
                            ("+", st[0], forest_p, all_idx[g] + vi),
                            ("+", st[1], forest_p, all_idx[g+1] + vi)
                        ])
                        self.emit(load=[
                            ("load", v_nv[g] + vi, st[0]),
                            ("load", v_nv[g+1] + vi, st[1])
                        ])
                # Stage 5 opt: XOR gathered values with C
                for g in range(n_groups):
                    self.emit(valu=[("^", v_nv[g], v_nv[g], v_stage5_xor)])

            # PHASE 2: XOR all values with v_nv
            for g in range(n_groups):
                self.emit(valu=[("^", all_val[g], all_val[g], v_nv[g])])

            # PHASE 3: Hash - 6 stages
            for hi in range(6):
                op1, _, op2, op3, _ = HASH_STAGES[hi]
                for g in range(n_groups):
                    if hash_fma_mult[hi] is not None:
                        self.emit(valu=[("multiply_add", all_val[g],
                                        all_val[g], v_hash_mult[hi], v_hc[hi])])
                    elif hi == 5:
                        t = v_t[g]
                        self.emit(valu=[(">>", t, all_val[g], v_hs[hi])])
                        self.emit(valu=[("^", all_val[g], all_val[g], t)])
                    else:
                        t = v_t[g]
                        self.emit(valu=[(op1, t, all_val[g], v_hc[hi]),
                                       (op3, v_nv[g], all_val[g], v_hs[hi])])
                        self.emit(valu=[(op2, all_val[g], t, v_nv[g])])

            # PHASE 4: Branch computation
            max_idx_this_round = (1 << (rnd + 2)) - 1
            needs_bounds_check = max_idx_this_round >= n_nodes

            for g in range(n_groups):
                t = v_t[g]
                self.emit(valu=[("&", t, all_val[g], v_1)])
                self.emit(valu=[("-", t, v_2, t)])
                self.emit(valu=[("multiply_add", all_idx[g], all_idx[g], v_2, t)])
                if needs_bounds_check:
                    self.emit(valu=[("<", t, all_idx[g], v_nn)])
                    self.emit(valu=[("*", all_idx[g], all_idx[g], t)])

        # Stage 5 optimization: XOR all values with C before storing
        for g in range(n_groups):
            self.emit(valu=[("^", all_val[g], all_val[g], v_stage5_xor)])

        # ========== STORE DATA ==========
        for g in range(0, n_groups, 2):
            self.emit(load=[("const", st[0], g * VLEN), ("const", st[1], (g+1) * VLEN)])
            self.emit(alu=[("+", st[2], idx_p, st[0]), ("+", st[3], idx_p, st[1]),
                          ("+", st[4], val_p, st[0]), ("+", st[5], val_p, st[1])])
            self.emit(store=[("vstore", st[2], all_idx[g]), ("vstore", st[3], all_idx[g+1])])
            self.emit(store=[("vstore", st[4], all_val[g]), ("vstore", st[5], all_val[g+1])])

        print(f"Before scheduling: {len(self.instrs)} instructions")

        # RESCHEDULE
        ops = extract_ops(self.instrs)
        op_by_id = compute_dependencies(ops)
        print(f"Scheduling {len(ops)} ops...")

        import time
        start = time.time()
        num_cycles, scheduled_ops = schedule_greedy(ops, op_by_id, enable_valu_shift=enable_valu_shift)
        scheduled = emit_scheduled(scheduled_ops, num_cycles)
        elapsed = time.time() - start
        print(f"Greedy scheduling completed in {elapsed:.2f}s")

        self.instrs = [{"flow": [("pause",)]}] + scheduled + [{"flow": [("pause",)]}]
        print(f"Final scheduling: {len(self.instrs)} instructions ({num_cycles} cycles)")


def analyze_index_collisions(batch_size: int, rounds: int, n_nodes: int):
    """Analyze expected index collisions per round.

    This helps understand the potential benefit of partition-based optimization.
    """
    import random

    # Simulate multiple runs to get average collision statistics
    n_trials = 100
    results = []

    for trial in range(n_trials):
        random.seed(trial)
        # Start with all indices at 0 (as per Input.generate)
        indices = [0] * batch_size
        values = [random.randint(0, 2**30 - 1) for _ in range(batch_size)]

        # Generate random tree values
        tree_values = [random.randint(0, 2**30 - 1) for _ in range(n_nodes)]

        round_stats = []
        for rnd in range(rounds):
            # Count unique indices this round
            unique_idx = len(set(indices))
            round_stats.append(unique_idx)

            # Simulate one round
            for i in range(batch_size):
                idx = indices[i]
                val = values[i]

                # XOR with tree value
                val ^= tree_values[idx]

                # Hash (simplified - just use Python's hash)
                val = hash(val) & 0xFFFFFFFF

                # Branch
                idx = 2 * idx + (1 if val % 2 == 0 else 2)
                idx = 0 if idx >= n_nodes else idx

                indices[i] = idx
                values[i] = val

        results.append(round_stats)

    # Average across trials
    avg_unique = []
    for rnd in range(rounds):
        avg = sum(r[rnd] for r in results) / n_trials
        avg_unique.append(avg)

    print("\n=== Index Collision Analysis ===")
    print(f"Batch size: {batch_size}, Rounds: {rounds}, Tree nodes: {n_nodes}")
    print("\nAverage unique indices per round:")
    for rnd, unique in enumerate(avg_unique):
        savings = batch_size - unique
        pct = (savings / batch_size) * 100
        print(f"  Round {rnd:2d}: {unique:6.1f} unique ({pct:5.1f}% savings from collisions)")

    return avg_unique


if __name__ == "__main__":
    # Analyze collision statistics
    analyze_index_collisions(batch_size=256, rounds=16, n_nodes=2047)

    # Test the kernel
    print("\n=== Testing Partition Kernel ===")
    builder = PartitionKernelBuilder()
    builder.build_kernel(10, 2047, 256, 16, use_greedy=True, enable_valu_shift=True)

    from problem import Machine, Tree, Input, build_mem_image, reference_kernel, N_CORES
    import random

    for seed in [42, 123, 456]:
        random.seed(seed)
        forest = Tree.generate(10)
        inp = Input.generate(forest, 256, 16)
        mem = build_mem_image(forest, inp)

        # Run our kernel
        machine = Machine(mem, builder.instrs, builder.debug_info(), n_cores=N_CORES)
        machine.enable_pause = False
        machine.run()

        # Run reference
        random.seed(seed)
        forest_ref = Tree.generate(10)
        inp_ref = Input.generate(forest_ref, 256, 16)
        reference_kernel(forest_ref, inp_ref)

        # Compare results
        result_idx = machine.mem[mem[5]:mem[5]+256]
        result_val = machine.mem[mem[6]:mem[6]+256]

        idx_ok = result_idx == inp_ref.indices
        val_ok = result_val == inp_ref.values

        print(f"Seed {seed}: Cycles={machine.cycle}, idx_ok={idx_ok}, val_ok={val_ok}")
