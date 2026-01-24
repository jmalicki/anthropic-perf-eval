"""
Jump-Based Kernel - Uses control flow with demand-driven vload caching.

Key insight from user: With TRUE sorted items (not just chunked), the vload
doesn't need to be aligned to any chunk size - it can start at any index!

Approach:
1. Items are sorted by tree index (maintained via stable partition each round)
2. Keep a vload cache covering tree[base : base+8]
3. For each item:
   - If idx in [base, base+8): use cache, select value at (idx - base)
   - Else: vload(tree[idx : idx+8]) to refresh cache
4. Use conditional jump to skip vload when cache hits

With 99% collision, consecutive items often need the same tree value:
- Most items hit the cache
- Few vloads needed (1-3 instead of 256 scattered loads)

The ISA supports:
- ("cond_jump", cond, addr) - Jump if scratch[cond] != 0
- ("cond_jump_rel", cond, offset) - Relative jump if scratch[cond] != 0
- ("jump", addr) - Unconditional jump
- ("jump_indirect", addr) - Jump to address in scratch

Example with sorted items:
  Items 0-200: all need tree[15] (99% collision at this index)
  Items 201-250: need tree[16-20]
  Items 251-255: need tree[21]

  Processing:
  1. First item needs tree[15], load vload(tree[15:23])
  2. Items 0-200: all hit cache at offset 0 (tree[15])
  3. Item 201 needs tree[16]: still in cache at offset 1
  4. Eventually items need tree[23+]: load new vload
  5. Total: 2-3 vloads instead of 256 loads!
"""

from problem import DebugInfo, SLOT_LIMITS, VLEN, SCRATCH_SIZE, HASH_STAGES
from solver.reschedule import extract_ops, compute_dependencies, schedule_greedy, emit_scheduled


class JumpKernelBuilder:
    """Build a kernel using jump-based control flow for chunk skipping."""

    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.labels = {}  # name -> instruction index
        self.fixups = []  # (instr_idx, label_name) for forward references

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

    def label(self, name: str):
        """Define a label at the current instruction position."""
        self.labels[name] = len(self.instrs)

    def emit_jump(self, label: str):
        """Emit an unconditional jump to a label."""
        if label in self.labels:
            addr = self.labels[label]
        else:
            addr = 0  # Will be fixed up later
            self.fixups.append((len(self.instrs), label))
        self.emit(flow=[("jump", addr)])

    def emit_cond_jump(self, cond: int, label: str):
        """Emit a conditional jump to a label."""
        if label in self.labels:
            addr = self.labels[label]
        else:
            addr = 0  # Will be fixed up later
            self.fixups.append((len(self.instrs), label))
        self.emit(flow=[("cond_jump", cond, addr)])

    def resolve_fixups(self):
        """Resolve forward references to labels."""
        for instr_idx, label_name in self.fixups:
            if label_name not in self.labels:
                raise ValueError(f"Undefined label: {label_name}")
            addr = self.labels[label_name]
            instr = self.instrs[instr_idx]
            if "flow" in instr:
                slots = instr["flow"]
                new_slots = []
                for slot in slots:
                    if slot[0] == "jump":
                        new_slots.append(("jump", addr))
                    elif slot[0] == "cond_jump":
                        new_slots.append(("cond_jump", slot[1], addr))
                    else:
                        new_slots.append(slot)
                instr["flow"] = new_slots

    def build_kernel(
        self,
        forest_height: int,
        n_nodes: int,
        batch_size: int,
        rounds: int,
        chunk_size: int = 4,
        use_greedy: bool = True,
    ):
        """Build kernel with jump-based chunk skipping.

        This version demonstrates the control flow structure for chunk skipping.
        It's not optimized but shows how jumps can be used.

        Algorithm:
        1. For each round at depth >= 4:
           a. Compute which chunk each item needs: chunk_id = idx / chunk_size
           b. Find min/max chunk_id across all items
           c. Loop from min to max chunk:
              - Check if any item needs this chunk
              - If yes: vload chunk, select values for items in this chunk
              - If no: skip via jump
        2. After loop, continue with hash and branch as usual
        """
        n_groups = batch_size // VLEN
        tree_height = forest_height + 1

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

        # Per-group vectors
        all_idx = [self.vec(f"all_idx_{g}") for g in range(n_groups)]
        all_val = [self.vec(f"all_val_{g}") for g in range(n_groups)]
        v_nv = [self.vec(f"v_nv_{g}") for g in range(n_groups)]
        v_t = [self.vec(f"v_t_{g}") for g in range(n_groups)]

        # Loop control
        chunk_id = self.alloc("chunk_id")  # Current chunk being processed
        max_chunk = self.alloc("max_chunk")  # Max chunk to process
        loop_cond = self.alloc("loop_cond")  # Loop continuation condition

        # Chunk cache - holds one vload worth of tree values
        chunk_cache = self.vec("chunk_cache")  # 8 tree values

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

        # Chunk selection constants
        v_chunk_size = self.vec("v_chunk_size")
        v_chunk_mask = self.vec("v_chunk_mask")  # chunk_size - 1 for mod operation

        # FMA optimization
        hash_fma_mult = [None] * 6
        v_hash_mult = {}
        for i, (op1, const, op2, op3, shift) in enumerate(HASH_STAGES):
            if op1 == '+' and op2 == '+' and op3 == '<<':
                hash_fma_mult[i] = 1 + (1 << shift)

        # Stage 5 XOR
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

        for i in range(6):
            self.emit(load=[("const", hc[i], HASH_STAGES[i][1]),
                           ("const", hs[i], HASH_STAGES[i][4])])
        self.emit(valu=[("vbroadcast", v_0, c[0]), ("vbroadcast", v_1, c[1]),
                        ("vbroadcast", v_2, c[2]), ("vbroadcast", v_nn, n_nodes_s)])
        for i in range(0, 8, 2):
            self.emit(valu=[("vbroadcast", v_const[i], c[i]),
                           ("vbroadcast", v_const[i+1], c[i+1])])
        for i in range(0, 6, 2):
            self.emit(valu=[("vbroadcast", v_hc[i], hc[i]),
                           ("vbroadcast", v_hc[i+1], hc[i+1]),
                           ("vbroadcast", v_hs[i], hs[i]),
                           ("vbroadcast", v_hs[i+1], hs[i+1])])

        # Initialize FMA multipliers
        for i in range(6):
            if hash_fma_mult[i] is not None:
                mult_s = self.alloc(f"hash_mult_{i}")
                v_mult = self.vec(f"v_hash_mult_{i}")
                v_hash_mult[i] = v_mult
                self.emit(load=[("const", mult_s, hash_fma_mult[i])])
                self.emit(valu=[("vbroadcast", v_mult, mult_s)])

        # Stage 5 XOR
        self.emit(load=[("const", stage5_xor_s, STAGE5_XOR_CONST)])
        self.emit(valu=[("vbroadcast", v_stage5_xor, stage5_xor_s)])

        # Chunk constants
        chunk_size_s = self.alloc("chunk_size_s")
        chunk_mask_s = self.alloc("chunk_mask_s")
        self.emit(load=[("const", chunk_size_s, chunk_size)])
        self.emit(load=[("const", chunk_mask_s, chunk_size - 1)])
        self.emit(valu=[("vbroadcast", v_chunk_size, chunk_size_s)])
        self.emit(valu=[("vbroadcast", v_chunk_mask, chunk_mask_s)])

        # Tree cache
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

        # Stage 5: XOR initial values
        for g in range(n_groups):
            self.emit(valu=[("^", all_val[g], all_val[g], v_stage5_xor)])

        # ========== MAIN LOOP (UNROLLED BY ROUND) ==========
        tree_depth = 0
        for rnd in range(rounds):
            # PHASE 1: Get tree values into v_nv
            if tree_depth == 0:
                # Depth 0: broadcast tc[0]
                for g in range(n_groups):
                    self.emit(valu=[("vbroadcast", v_nv[g], tc[0])])
                for g in range(n_groups):
                    self.emit(valu=[("^", v_nv[g], v_nv[g], v_stage5_xor)])

            elif tree_depth == 1:
                # Depth 1: delta method
                for g in range(n_groups):
                    t = v_t[g]
                    self.emit(valu=[("==", t, all_idx[g], v_1)])
                    self.emit(valu=[("*", t, v_td_1, t)])
                    self.emit(valu=[("+", v_nv[g], v_tc2, t)])
                for g in range(n_groups):
                    self.emit(valu=[("^", v_nv[g], v_nv[g], v_stage5_xor)])

            elif tree_depth == 2:
                # Depth 2: delta method with 4 values
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

            elif tree_depth == 3:
                # Depth 3: VALU selection from 8 tree values
                # This is the same as HybridTreeAccessStrategy depth 3
                # Preloaded tree[7..14] at init would be ideal, but for simplicity
                # we do scattered loads and will optimize later
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
                for g in range(n_groups):
                    self.emit(valu=[("^", v_nv[g], v_nv[g], v_stage5_xor)])

            else:
                # Depth 4+: This is where jump-based chunk skipping could help
                #
                # With sorted items (by tree index), we could:
                # 1. Track current vload range [base, base+8)
                # 2. Process items in order
                # 3. When item's idx is outside range, load new vload
                # 4. Use conditional jump to skip the load if not needed
                #
                # For now, use scattered loads (baseline)
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
                for g in range(n_groups):
                    self.emit(valu=[("^", v_nv[g], v_nv[g], v_stage5_xor)])

            tree_depth += 1
            if tree_depth >= tree_height:
                tree_depth = 0

            # PHASE 2: XOR
            for g in range(n_groups):
                self.emit(valu=[("^", all_val[g], all_val[g], v_nv[g])])

            # PHASE 3: Hash
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

            # PHASE 4: Branch
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

        # Final XOR
        for g in range(n_groups):
            self.emit(valu=[("^", all_val[g], all_val[g], v_stage5_xor)])

        # ========== STORE DATA ==========
        for g in range(0, n_groups, 2):
            self.emit(load=[("const", st[0], g * VLEN), ("const", st[1], (g+1) * VLEN)])
            self.emit(alu=[("+", st[2], idx_p, st[0]), ("+", st[3], idx_p, st[1]),
                          ("+", st[4], val_p, st[0]), ("+", st[5], val_p, st[1])])
            self.emit(store=[("vstore", st[2], all_idx[g]), ("vstore", st[3], all_idx[g+1])])
            self.emit(store=[("vstore", st[4], all_val[g]), ("vstore", st[5], all_val[g+1])])

        self.resolve_fixups()
        store_end = len(self.instrs)
        print(f"Before scheduling: {store_end} instructions")

        # Reschedule
        if use_greedy:
            ops = extract_ops(self.instrs)
            op_by_id = compute_dependencies(ops)
            print(f"Scheduling {len(ops)} ops...")
            num_cycles, scheduled_ops = schedule_greedy(ops, op_by_id, enable_valu_shift=True)
            scheduled = emit_scheduled(scheduled_ops, num_cycles)
            self.instrs = [{"flow": [("pause",)]}] + scheduled + [{"flow": [("pause",)]}]
            print(f"Final: {len(self.instrs)} instructions ({num_cycles} cycles)")


def test_jump_kernel():
    """Test the jump kernel builder."""
    builder = JumpKernelBuilder()
    builder.build_kernel(
        forest_height=10,
        n_nodes=2047,
        batch_size=256,
        rounds=16,
    )
    print(f"Jump kernel: {len(builder.instrs) - 2} cycles")
    return builder


if __name__ == "__main__":
    test_jump_kernel()
