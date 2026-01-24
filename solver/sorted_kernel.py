"""
Sorted Kernel - Maintains sorted order via stable partition for efficient tree access.

Key insight: With items sorted by tree index, we can use demand-driven vload caching:
1. Process items in order (sorted by tree index)
2. Keep a vload cache covering tree[base : base+8]
3. Most consecutive items hit the cache (high collision = nearby indices)
4. Only reload when cache miss

Stable partition maintains sort order:
- Initial: items sorted by starting index
- Each round: partition by branch bit (even/odd)
- Stable partition preserves relative order within each partition
- Result: items remain sorted by NEW index (new_idx = 2*old_idx + branch)

With 99% collision, most items access same tree value:
- Cache hit rate is very high
- Few vloads needed per round (1-5 instead of 256 scattered loads)

Implementation approach:
1. Scalar loop over items (not vectorized) with cache tracking
2. Process items one-by-one, checking cache on each
3. Use conditional jump to skip vload when cache hits
4. Store results back in sorted order

This trades vectorization for cache efficiency. With high collision, the
reduced loads can offset the scalar loop overhead.
"""

from problem import DebugInfo, SLOT_LIMITS, VLEN, SCRATCH_SIZE, HASH_STAGES


class SortedKernelBuilder:
    """Build a kernel using sorted items with demand-driven vload caching."""

    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.labels = {}
        self.fixups = []

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
        """Define a label at current position."""
        self.labels[name] = len(self.instrs)

    def emit_jump(self, label: str):
        """Emit unconditional jump."""
        addr = self.labels.get(label, 0)
        if label not in self.labels:
            self.fixups.append((len(self.instrs), label))
        self.emit(flow=[("jump", addr)])

    def emit_cond_jump(self, cond: int, label: str):
        """Emit conditional jump."""
        addr = self.labels.get(label, 0)
        if label not in self.labels:
            self.fixups.append((len(self.instrs), label))
        self.emit(flow=[("cond_jump", cond, addr)])

    def resolve_fixups(self):
        """Resolve forward references."""
        for instr_idx, label_name in self.fixups:
            if label_name not in self.labels:
                raise ValueError(f"Undefined label: {label_name}")
            addr = self.labels[label_name]
            instr = self.instrs[instr_idx]
            if "flow" in instr:
                new_slots = []
                for slot in instr["flow"]:
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
    ):
        """Build kernel with scalar loop and demand-driven cache.

        NOTE: This is a conceptual implementation showing the structure.
        The scalar loop approach is likely slower than vectorized scattered
        loads due to loop overhead and lack of ILP. The goal is to demonstrate
        the algorithm, not achieve best performance.

        For actual speedup, we'd need:
        1. Hardware support for gather operations, OR
        2. Pre-sorted input with run-length encoding, OR
        3. Hybrid approach using vectorized code for common case
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

        # Loop variables for scalar processing
        loop_idx = self.alloc("loop_idx")  # Current item index
        loop_end = self.alloc("loop_end")  # batch_size
        loop_cond = self.alloc("loop_cond")  # Loop condition

        # Cache variables
        cache_base = self.alloc("cache_base")  # Start of current cache range
        cache_end = self.alloc("cache_end")  # End of cache range (base + 8)
        cache_valid = self.alloc("cache_valid")  # 1 if cache is valid
        cache = self.vec("cache")  # 8 cached tree values

        # Current item data
        cur_idx = self.alloc("cur_idx")  # Current tree index
        cur_val = self.alloc("cur_val")  # Current value
        cur_nv = self.alloc("cur_nv")  # Current node value (from tree)
        cur_t = self.alloc("cur_t")  # Temp for hash

        # Per-group vectors (for vectorized depths 0-3)
        all_idx = [self.vec(f"all_idx_{g}") for g in range(n_groups)]
        all_val = [self.vec(f"all_val_{g}") for g in range(n_groups)]
        v_nv = [self.vec(f"v_nv_{g}") for g in range(n_groups)]
        v_t = [self.vec(f"v_t_{g}") for g in range(n_groups)]

        v_0 = self.vec("v_0")
        v_1 = self.vec("v_1")
        v_2 = self.vec("v_2")
        v_nn = self.vec("v_nn")
        v_const = [self.vec(f"v_const{i}") for i in range(8)]

        # Constants for loop
        const_8 = self.alloc("const_8")

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

        # Loop constants
        self.emit(load=[("const", loop_end, batch_size)])
        self.emit(load=[("const", const_8, 8)])

        # Tree cache
        for i in range(7):
            self.emit(alu=[("+", st[0], forest_p, c[i])])
            self.emit(load=[("load", tc[i], st[0])])

        # ========== LOAD DATA (vectorized) ==========
        for g in range(0, n_groups, 2):
            self.emit(load=[("const", st[0], g * VLEN), ("const", st[1], (g+1) * VLEN)])
            self.emit(alu=[("+", st[2], idx_p, st[0]), ("+", st[3], idx_p, st[1]),
                          ("+", st[4], val_p, st[0]), ("+", st[5], val_p, st[1])])
            self.emit(load=[("vload", all_idx[g], st[2]), ("vload", all_idx[g+1], st[3])])
            self.emit(load=[("vload", all_val[g], st[4]), ("vload", all_val[g+1], st[5])])

        # ========== MAIN LOOP ==========
        # For demonstration, we'll use vectorized code for depths 0-3
        # and show the scalar loop structure for depths 4+

        tree_depth = 0
        for rnd in range(rounds):
            if tree_depth <= 3:
                # Use vectorized code for depths 0-3 (same as hybrid)
                self._emit_vectorized_round(
                    tree_depth, rnd, n_groups, forest_p, n_nodes,
                    all_idx, all_val, v_nv, v_t, tc, v_0, v_1, v_2, v_nn, v_const,
                    hc, hs, st
                )
            else:
                # Demonstrate scalar loop structure for depths 4+
                # NOTE: This is slower than vectorized scattered loads!
                # It's here to show the algorithm, not for performance.
                self._emit_scalar_round_with_cache(
                    tree_depth, rnd, batch_size, forest_p, n_nodes,
                    idx_p, val_p, loop_idx, loop_end, loop_cond,
                    cache_base, cache_end, cache_valid, cache, const_8,
                    cur_idx, cur_val, cur_nv, cur_t, hc, hs, st, c
                )

            tree_depth += 1
            if tree_depth >= tree_height:
                tree_depth = 0

        # ========== STORE DATA (vectorized) ==========
        for g in range(0, n_groups, 2):
            self.emit(load=[("const", st[0], g * VLEN), ("const", st[1], (g+1) * VLEN)])
            self.emit(alu=[("+", st[2], idx_p, st[0]), ("+", st[3], idx_p, st[1]),
                          ("+", st[4], val_p, st[0]), ("+", st[5], val_p, st[1])])
            self.emit(store=[("vstore", st[2], all_idx[g]), ("vstore", st[3], all_idx[g+1])])
            self.emit(store=[("vstore", st[4], all_val[g]), ("vstore", st[5], all_val[g+1])])

        self.resolve_fixups()
        self.emit(flow=[("pause",)])

        print(f"Total: {len(self.instrs)} instructions")

    def _emit_vectorized_round(
        self, tree_depth, rnd, n_groups, forest_p, n_nodes,
        all_idx, all_val, v_nv, v_t, tc, v_0, v_1, v_2, v_nn, v_const,
        hc, hs, st
    ):
        """Emit vectorized code for depths 0-3."""
        # Similar to hybrid strategy - simplified here
        if tree_depth == 0:
            for g in range(n_groups):
                self.emit(valu=[("vbroadcast", v_nv[g], tc[0])])
        elif tree_depth == 1:
            # Simplified - actual implementation uses delta method
            for g in range(n_groups):
                self.emit(valu=[("vbroadcast", v_nv[g], tc[1])])
        elif tree_depth == 2:
            for g in range(n_groups):
                self.emit(valu=[("vbroadcast", v_nv[g], tc[3])])
        elif tree_depth == 3:
            for g in range(n_groups):
                self.emit(valu=[("vbroadcast", v_nv[g], tc[6])])

        # XOR
        for g in range(n_groups):
            self.emit(valu=[("^", all_val[g], all_val[g], v_nv[g])])

        # Hash (simplified - just XOR with constant)
        for g in range(n_groups):
            self.emit(valu=[("^", all_val[g], all_val[g], v_const[0])])

        # Branch (simplified)
        max_idx_this_round = (1 << (rnd + 2)) - 1
        needs_bounds_check = max_idx_this_round >= n_nodes

        for g in range(n_groups):
            t = v_t[g]
            self.emit(valu=[("&", t, all_val[g], v_1)])
            self.emit(valu=[("+", t, t, v_1)])
            self.emit(valu=[("*", all_idx[g], all_idx[g], v_2)])
            self.emit(valu=[("+", all_idx[g], all_idx[g], t)])
            if needs_bounds_check:
                self.emit(valu=[("<", t, all_idx[g], v_nn)])
                self.emit(valu=[("*", all_idx[g], all_idx[g], t)])

    def _emit_scalar_round_with_cache(
        self, tree_depth, rnd, batch_size, forest_p, n_nodes,
        idx_p, val_p, loop_idx, loop_end, loop_cond,
        cache_base, cache_end, cache_valid, cache, const_8,
        cur_idx, cur_val, cur_nv, cur_t, hc, hs, st, c
    ):
        """Emit scalar loop with demand-driven cache for depths 4+.

        This demonstrates the algorithm structure:
        1. loop_idx = 0
        2. cache_valid = 0 (no cache yet)
        3. LOOP:
           a. Load cur_idx, cur_val from memory
           b. Check if cur_idx in [cache_base, cache_end)
           c. If not: vload new cache starting at cur_idx
           d. Select cur_nv from cache based on (cur_idx - cache_base)
           e. Hash and branch
           f. Store cur_idx, cur_val
           g. loop_idx++
           h. If loop_idx < batch_size: goto LOOP
        """
        # Initialize loop
        self.emit(load=[("const", loop_idx, 0)])
        self.emit(load=[("const", cache_valid, 0)])
        self.emit(load=[("const", cache_base, 0)])

        self.label(f"loop_start_d{tree_depth}_r{rnd}")

        # Load current item's idx and val
        self.emit(alu=[("+", st[0], idx_p, loop_idx)])
        self.emit(load=[("load", cur_idx, st[0])])
        self.emit(alu=[("+", st[0], val_p, loop_idx)])
        self.emit(load=[("load", cur_val, st[0])])

        # Check cache validity and range
        # If cache_valid == 0 OR cur_idx < cache_base OR cur_idx >= cache_end: reload
        # Simplified: always reload for demonstration
        self.emit(alu=[("+", st[0], forest_p, cur_idx)])
        self.emit(load=[("vload", cache, st[0])])  # Load 8 values
        self.emit(alu=[("+", cache_end, cur_idx, const_8)])
        self.emit(load=[("const", cache_valid, 1)])

        # Select from cache (simplified - just use first value)
        # Real implementation would compute offset = cur_idx - cache_base
        self.emit(alu=[("+", cur_nv, cache, c[0])])  # cur_nv = cache[0]

        # XOR
        self.emit(alu=[("^", cur_val, cur_val, cur_nv)])

        # Hash (simplified)
        for hi in range(6):
            self.emit(alu=[("^", cur_val, cur_val, hc[hi])])

        # Branch
        self.emit(alu=[("&", cur_t, cur_val, c[1])])
        self.emit(alu=[("+", cur_t, cur_t, c[1])])
        self.emit(alu=[("*", cur_idx, cur_idx, c[2])])
        self.emit(alu=[("+", cur_idx, cur_idx, cur_t)])

        # Bounds check
        max_idx_this_round = (1 << (rnd + 2)) - 1
        if max_idx_this_round >= n_nodes:
            self.emit(alu=[("<", cur_t, cur_idx, c[1])])  # Simplified
            self.emit(alu=[("*", cur_idx, cur_idx, cur_t)])

        # Store results
        self.emit(alu=[("+", st[0], idx_p, loop_idx)])
        self.emit(store=[("store", st[0], cur_idx)])
        self.emit(alu=[("+", st[0], val_p, loop_idx)])
        self.emit(store=[("store", st[0], cur_val)])

        # Increment and loop
        self.emit(alu=[("+", loop_idx, loop_idx, c[1])])
        self.emit(alu=[("<", loop_cond, loop_idx, loop_end)])
        self.emit_cond_jump(loop_cond, f"loop_start_d{tree_depth}_r{rnd}")


def test_sorted_kernel():
    """Test the sorted kernel builder."""
    builder = SortedKernelBuilder()
    builder.build_kernel(
        forest_height=10,
        n_nodes=2047,
        batch_size=256,
        rounds=16,
    )
    print(f"Sorted kernel: {len(builder.instrs)} instructions")
    return builder


if __name__ == "__main__":
    test_sorted_kernel()
