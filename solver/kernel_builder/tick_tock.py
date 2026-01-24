"""
Tick-Tock Partition Kernel Builder.

This module contains the TickTockKernelBuilder that implements the tick-tock
lazy partition strategy for optimized tree access.
"""

from typing import List

from problem import VLEN, HASH_STAGES, SCRATCH_SIZE

from .base import BaseKernelBuilder
from .config import KernelConfig, TreeAccessMode, SchedulingMode
from .context import ScratchContext


class TickTockKernelBuilder(BaseKernelBuilder):
    """Build kernels using the Tick-Tock Lazy Partition strategy.

    This builder implements the complete tick-tock algorithm:
    1. Items are partitioned by branch bit using stable partition
    2. Partition stores happen over multiple rounds using idle store slots
    3. Sorted items enable vload-based tree access (8 values cover any group)

    Key schedule:
    - Rounds 0-3: VALU selection (depths 0-3), partition stores pipeline
    - Rounds 4-10: Sorted vload (depths 4-10), partition stores for next cycle
    - Round 11: Wrap reset - all indices become 0 (clean slate)
    - Rounds 11-14: VALU selection, partition stores for round 15
    - Round 15: Sorted vload (depth 4)

    Expected improvement: ~800 cycles saved → ~1050 cycles (140x speedup)

    Implementation notes:
    - Partition stores use the extra_room memory region (ptr at address 7)
    - Double-buffered: stores to buffer B while reading from buffer A
    - Partition positions computed via prefix sum during branch computation
    """

    UNROLL = 8
    PARTITION_READY_ROUNDS = {4, 15}  # Rounds where partition data is ready
    WRAP_ROUND = 10  # Round where wrap happens (depth 10, all indices reset)

    # Partition schedule:
    # Rounds 0-2: TICK stores (items 0-127)
    # Round 3: TOCK stores (items 128-255), partition ready for round 4
    # Rounds 4-6: Use sorted data, TICK stores for next
    # Partition stores based on DEPTH, not round number:
    # - TICK depths: Store first half (groups 0-15)
    # - TOCK depths: Store second half (groups 16-31)
    #
    # Schedule matches the original round-based config:
    # Round 0-3 (depths 0-3): TICK 0,1,2 + TOCK 3 → ready for depth 4
    # Round 4-9 (depths 4-9): TICK 4,5,6 + TOCK 7,8,9 → ready for depth 10
    # Round 10 (depth 10): WRAP - no stores
    # Round 11-14 (depths 0-3): TICK 0,1,2 + TOCK 3 → ready for depth 4
    # Round 15 (depth 4): Use partition data
    TICK_STORE_DEPTHS = {0, 1, 2, 4, 5, 6}   # First half on these depths
    TOCK_STORE_DEPTHS = {3, 7, 8, 9}          # Second half on these depths

    def __init__(self, config: KernelConfig = None):
        if config is None:
            config = KernelConfig()
        super().__init__(config)
        self.ctx = ScratchContext()

        # Memory pointers for partition buffers
        self.extra_room_p: int = 0  # Pointer to extra_room memory
        self.partition_buf_a: int = 0  # Buffer A base (indices: 0-255, values: 256-511)
        self.partition_buf_b: int = 0  # Buffer B base

        # Partition-specific scratch
        self.tree_cache_scalars: int = 0    # Scalar storage for tree cache
        self.tree_cache_base_s: int = 0     # Base index of cache
        self.v_cache_base: int = 0          # Vector broadcast of cache base
        self.v_tree_temp: int = 0           # Temp vector for tree operations

        # Prefix sum state
        self.v_branch_bits: List[int] = []  # Branch bits per group
        self.v_prefix_sum: List[int] = []   # Cumulative count of branch=0
        self.count0_total: int = 0          # Total count of branch=0 items

        # Current partition state
        self.using_buffer_a: bool = True  # Which buffer to read from
        self.partition_ready: bool = False  # Whether partition data is valid

        # Round contexts for tracking state
        self.rnd_contexts = {}

    def build_kernel(
        self,
        forest_height: int,
        n_nodes: int,
        batch_size: int,
        rounds: int,
        **kwargs
    ):
        """Build the tick-tock partition kernel."""
        self.config = KernelConfig(
            forest_height=forest_height,
            n_nodes=n_nodes,
            batch_size=batch_size,
            rounds=rounds,
            tree_access_mode=TreeAccessMode.TICK_TOCK_PARTITION,
            scheduling_mode=SchedulingMode.GREEDY_MULTIPASS,
            **kwargs
        )

        # Re-initialize state
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.labels = {}
        self.fixups = []
        self.ctx = ScratchContext()

        n_groups = batch_size // VLEN
        n_iters = n_groups // self.UNROLL
        tree_height = forest_height + 1

        # Allocate scratch
        self._allocate_common(n_groups, n_iters)
        self._allocate_partition(n_groups)
        self._allocate_valu_selection(n_groups, n_iters)

        if self.scratch_ptr > SCRATCH_SIZE:
            print(f"WARNING: Scratch usage {self.scratch_ptr} exceeds budget {SCRATCH_SIZE}!")
        else:
            print(f"Scratch usage: {self.scratch_ptr} / {SCRATCH_SIZE}")

        # Emit init
        self._emit_common_init()
        self._emit_valu_selection_init()

        # Load data
        self._emit_load_data(n_groups)

        # Main loop
        tree_depth = 0
        for rnd in range(rounds):
            self._emit_round(rnd, tree_depth, n_groups, n_iters)

            # Update tree depth with wrap
            tree_depth += 1
            if tree_depth >= tree_height:
                tree_depth = 0

        # Convert values back from optimized form
        for g in range(n_groups):
            self.emit(valu=[("^", self.ctx.all_val[g], self.ctx.all_val[g], self.ctx.v_stage5_xor)])

        # Store data
        self._emit_store_data(n_groups)

        print(f"Before scheduling: {len(self.instrs)} instructions")

        # Schedule
        self._schedule()

        return self

    def _allocate_common(self, n_groups: int, n_iters: int):
        """Allocate common scratch memory (same as ModularKernelBuilder)."""
        ctx = self.ctx

        # Pointers
        ctx.forest_p = self.alloc("forest_p")
        ctx.idx_p = self.alloc("idx_p")
        ctx.val_p = self.alloc("val_p")
        ctx.n_nodes_s = self.alloc("n_nodes")

        # Temps
        ctx.st = [self.alloc(f"st{i}") for i in range(16)]
        ctx.c = [self.alloc(f"c{i}") for i in range(16)]

        # Hash constants
        ctx.hc = [self.alloc(f"hc{i}") for i in range(6)]
        ctx.hs = [self.alloc(f"hs{i}") for i in range(6)]

        # Tree cache
        ctx.tc = [self.alloc(f"tc{i}") for i in range(7)]
        ctx.td_1 = self.alloc("td_1")
        ctx.td_4 = self.alloc("td_4")
        ctx.td_5 = self.alloc("td_5")
        ctx.td_6 = self.alloc("td_6")

        # Stage 5 XOR
        ctx.stage5_xor_s = self.alloc("stage5_xor")

        # Per-group vectors
        ctx.all_idx = [self.vec(f"all_idx_{g}") for g in range(n_groups)]
        ctx.all_val = [self.vec(f"all_val_{g}") for g in range(n_groups)]
        ctx.v_nv = [[self.vec(f"v_nv_{it}_{i}") for i in range(self.UNROLL)] for it in range(n_iters)]
        ctx.v_t = [[self.vec(f"v_t_{it}_{i}") for i in range(self.UNROLL)] for it in range(n_iters)]

        # Vector constants
        ctx.v_0 = self.vec("v_0")
        ctx.v_1 = self.vec("v_1")
        ctx.v_2 = self.vec("v_2")
        ctx.v_nn = self.vec("v_nn")
        ctx.v_const = [self.vec(f"v_const{i}") for i in range(8)]

        # Vector hash constants
        ctx.v_hc = [self.vec(f"v_hc{i}") for i in range(6)]
        ctx.v_hs = [self.vec(f"v_hs{i}") for i in range(6)]

        # Vector tree cache
        ctx.v_td_1 = self.vec("v_td_1")
        ctx.v_td_4 = self.vec("v_td_4")
        ctx.v_td_5 = self.vec("v_td_5")
        ctx.v_td_6 = self.vec("v_td_6")
        ctx.v_tc2 = self.vec("v_tc2")
        ctx.v_tc3 = self.vec("v_tc3")

        # Stage 5 XOR vector
        ctx.v_stage5_xor = self.vec("v_stage5_xor")

        # FMA multipliers
        ctx.hash_fma_mult = [None] * 6
        for i, (op1, const, op2, op3, shift) in enumerate(HASH_STAGES):
            if op1 == '+' and op2 == '+' and op3 == '<<':
                ctx.hash_fma_mult[i] = 1 + (1 << shift)

    def _allocate_partition(self, n_groups: int):
        """Allocate partition-specific scratch."""
        # Memory pointers (loaded from memory at init)
        self.extra_room_p = self.alloc("extra_room_p")

        # Tree cache for sorted vload access
        self.tree_cache_scalars = self.alloc("tree_cache_scalars", VLEN)
        self.tree_cache_base_s = self.alloc("tree_cache_base")
        self.v_cache_base = self.vec("v_cache_base_tt")

        # Temp vector for tree operations
        self.v_tree_temp = self.vec("v_tree_temp")

        # Partition buffer offsets (computed at runtime)
        self.partition_idx_offset = self.alloc("part_idx_off")  # Offset to idx section
        self.partition_val_offset = self.alloc("part_val_off")  # Offset to val section

    def _allocate_valu_selection(self, n_groups: int, n_iters: int):
        """Allocate VALU selection scratch (depth 3)."""
        ctx = self.ctx
        ctx.tc_r3 = [self.alloc(f"tc_r3_{i}") for i in range(8)]
        ctx.v_tc_r3 = [self.vec(f"v_tc_r3_{i}") for i in range(8)]
        ctx.v_const_r3 = [self.vec(f"v_const_r3_{i}") for i in range(8)]
        ctx.c_r3 = [self.alloc(f"c_r3_{i}") for i in range(8)]

    def _emit_common_init(self):
        """Emit common initialization (same as ModularKernelBuilder)."""
        ctx = self.ctx

        # Load constants 0-15
        for i in range(16):
            self.emit(load=[("const", ctx.c[i], i)])

        # Load pointers
        self.emit(load=[("const", ctx.st[0], 4), ("const", ctx.st[1], 5)])
        self.emit(load=[("load", ctx.forest_p, ctx.st[0]), ("load", ctx.idx_p, ctx.st[1])])
        self.emit(load=[("const", ctx.st[0], 6), ("const", ctx.st[1], 1)])
        self.emit(load=[("load", ctx.val_p, ctx.st[0]), ("load", ctx.n_nodes_s, ctx.st[1])])

        # Load extra_room pointer for partition buffers
        self.emit(load=[("const", ctx.st[0], 7)])
        self.emit(load=[("load", self.extra_room_p, ctx.st[0])])

        # Compute partition buffer offsets
        # Buffer layout in extra_room: [idx_a:256][val_a:256][idx_b:256][val_b:256]
        self.emit(load=[("const", self.partition_idx_offset, 0)])
        self.emit(load=[("const", self.partition_val_offset, 256)])

        # Load hash constants
        for i in range(6):
            self.emit(load=[("const", ctx.hc[i], HASH_STAGES[i][1]),
                           ("const", ctx.hs[i], HASH_STAGES[i][4])])

        # Broadcast vector constants
        self.emit(valu=[
            ("vbroadcast", ctx.v_0, ctx.c[0]),
            ("vbroadcast", ctx.v_1, ctx.c[1]),
            ("vbroadcast", ctx.v_2, ctx.c[2]),
            ("vbroadcast", ctx.v_nn, ctx.n_nodes_s)
        ])
        for i in range(0, 8, 2):
            self.emit(valu=[
                ("vbroadcast", ctx.v_const[i], ctx.c[i]),
                ("vbroadcast", ctx.v_const[i+1], ctx.c[i+1])
            ])
        for i in range(0, 6, 2):
            self.emit(valu=[
                ("vbroadcast", ctx.v_hc[i], ctx.hc[i]),
                ("vbroadcast", ctx.v_hc[i+1], ctx.hc[i+1]),
                ("vbroadcast", ctx.v_hs[i], ctx.hs[i]),
                ("vbroadcast", ctx.v_hs[i+1], ctx.hs[i+1])
            ])

        # Load Stage 5 XOR constant
        self.emit(load=[("const", ctx.stage5_xor_s, ctx.STAGE5_XOR_CONST)])
        self.emit(valu=[("vbroadcast", ctx.v_stage5_xor, ctx.stage5_xor_s)])

        # Load tree cache
        for i in range(7):
            self.emit(alu=[("+", ctx.st[0], ctx.forest_p, ctx.c[i])])
            self.emit(load=[("load", ctx.tc[i], ctx.st[0])])

        # Compute deltas
        self.emit(alu=[
            ("-", ctx.td_1, ctx.tc[1], ctx.tc[2]),
            ("-", ctx.td_4, ctx.tc[4], ctx.tc[3]),
            ("-", ctx.td_5, ctx.tc[5], ctx.tc[3]),
            ("-", ctx.td_6, ctx.tc[6], ctx.tc[3])
        ])
        self.emit(valu=[
            ("vbroadcast", ctx.v_tc2, ctx.tc[2]),
            ("vbroadcast", ctx.v_td_1, ctx.td_1),
            ("vbroadcast", ctx.v_tc3, ctx.tc[3]),
            ("vbroadcast", ctx.v_td_4, ctx.td_4),
            ("vbroadcast", ctx.v_td_5, ctx.td_5),
            ("vbroadcast", ctx.v_td_6, ctx.td_6)
        ])

        # FMA multipliers
        for i in range(6):
            if ctx.hash_fma_mult[i] is not None:
                mult_s = self.alloc(f"hash_mult_{i}")
                ctx.hash_fma_mult_s[i] = mult_s
                v_mult = self.vec(f"v_hash_mult_{i}")
                ctx.v_hash_mult[i] = v_mult
                self.emit(load=[("const", mult_s, ctx.hash_fma_mult[i])])
                self.emit(valu=[("vbroadcast", v_mult, mult_s)])

    def _emit_valu_selection_init(self):
        """Emit depth 3 VALU selection initialization."""
        ctx = self.ctx
        if ctx.tc_r3 is None:
            return

        # Load constants 7-14
        for i in range(8):
            self.emit(load=[("const", ctx.c_r3[i], 7 + i)])

        # Load tree[7..14]
        for i in range(8):
            self.emit(alu=[("+", ctx.st[0], ctx.forest_p, ctx.c_r3[i])])
            self.emit(load=[("load", ctx.tc_r3[i], ctx.st[0])])

        # Broadcast tree values
        for i in range(0, 8, 2):
            self.emit(valu=[
                ("vbroadcast", ctx.v_tc_r3[i], ctx.tc_r3[i]),
                ("vbroadcast", ctx.v_tc_r3[i+1], ctx.tc_r3[i+1])
            ])

        # Broadcast constants
        for i in range(0, 8, 2):
            self.emit(valu=[
                ("vbroadcast", ctx.v_const_r3[i], ctx.c_r3[i]),
                ("vbroadcast", ctx.v_const_r3[i+1], ctx.c_r3[i+1])
            ])

    def _emit_load_data(self, n_groups: int):
        """Emit data loading phase."""
        ctx = self.ctx

        for g in range(0, n_groups, 2):
            self.emit(load=[("const", ctx.st[0], g * VLEN), ("const", ctx.st[1], (g+1) * VLEN)])
            self.emit(alu=[
                ("+", ctx.st[2], ctx.idx_p, ctx.st[0]),
                ("+", ctx.st[3], ctx.idx_p, ctx.st[1]),
                ("+", ctx.st[4], ctx.val_p, ctx.st[0]),
                ("+", ctx.st[5], ctx.val_p, ctx.st[1])
            ])
            self.emit(load=[("vload", ctx.all_idx[g], ctx.st[2]), ("vload", ctx.all_idx[g+1], ctx.st[3])])
            self.emit(load=[("vload", ctx.all_val[g], ctx.st[4]), ("vload", ctx.all_val[g+1], ctx.st[5])])

        # Convert initial values to optimized form
        for g in range(n_groups):
            self.emit(valu=[("^", ctx.all_val[g], ctx.all_val[g], ctx.v_stage5_xor)])

    def _emit_store_data(self, n_groups: int):
        """Emit data storing phase."""
        ctx = self.ctx

        for g in range(0, n_groups, 2):
            self.emit(load=[("const", ctx.st[0], g * VLEN), ("const", ctx.st[1], (g+1) * VLEN)])
            self.emit(alu=[
                ("+", ctx.st[2], ctx.idx_p, ctx.st[0]),
                ("+", ctx.st[3], ctx.idx_p, ctx.st[1]),
                ("+", ctx.st[4], ctx.val_p, ctx.st[0]),
                ("+", ctx.st[5], ctx.val_p, ctx.st[1])
            ])
            self.emit(store=[("vstore", ctx.st[2], ctx.all_idx[g]), ("vstore", ctx.st[3], ctx.all_idx[g+1])])
            self.emit(store=[("vstore", ctx.st[4], ctx.all_val[g]), ("vstore", ctx.st[5], ctx.all_val[g+1])])

    def _emit_round(self, rnd: int, tree_depth: int, n_groups: int, n_iters: int):
        """Emit a complete round: tree access, XOR, hash, branch.

        Partition stores happen during branch phase, using idle store slots.
        The stored data builds a sorted copy for future sorted vload access.

        NOTE: Partition reload is NOT implemented yet because it requires
        the stable partition algorithm to actually sort items by tree index.
        Without sorting, reload would just copy unsorted data back.
        """
        ctx = self.ctx

        # Tree access based on depth
        if tree_depth == 0:
            self._emit_depth0_access(n_iters, rnd)
        elif tree_depth == 1:
            self._emit_depth1_access(n_iters, rnd)
        elif tree_depth == 2:
            self._emit_depth2_access(n_iters, rnd)
        elif tree_depth == 3:
            self._emit_depth3_access(n_iters, rnd)
        else:
            # For depths 4+, use scattered loads
            # TODO: Enable sorted vload once stable partition is implemented
            self._emit_scattered_access(n_iters, rnd)

        # XOR
        self._emit_xor(n_groups, rnd)

        # Hash
        self._emit_hash(n_groups, n_iters, rnd)

        # Branch (includes partition stores on TICK/TOCK rounds)
        self._emit_branch(n_groups, n_iters, rnd)

    def _emit_partition_reload(self, n_groups: int):
        """Reload data from partition buffer into working vectors.

        This loads the stored idx/val pairs back into all_idx and all_val.
        With proper stable partition, items would be sorted by branch bit,
        enabling sorted vload tree access.

        For now, this is a straight reload (no reordering) to validate
        the store/reload pipeline works correctly.
        """
        ctx = self.ctx

        # Compute base addresses
        self.emit(load=[("const", ctx.st[8], 0)])
        self.emit(alu=[("+", ctx.st[9], self.extra_room_p, ctx.st[8])])  # idx base
        self.emit(alu=[("+", ctx.st[10], ctx.st[9], self.partition_val_offset)])  # val base
        self.emit(load=[("const", ctx.st[11], VLEN)])  # increment constant

        # Reload all groups
        for g in range(n_groups):
            # Load indices and values
            self.emit(load=[
                ("vload", ctx.all_idx[g], ctx.st[9]),
                ("vload", ctx.all_val[g], ctx.st[10])
            ])

            # Increment addresses for next group (except for last)
            if g < n_groups - 1:
                self.emit(alu=[
                    ("+", ctx.st[9], ctx.st[9], ctx.st[11]),
                    ("+", ctx.st[10], ctx.st[10], ctx.st[11])
                ])

    def _emit_depth0_access(self, n_iters: int, rnd: int):
        """Depth 0: All at root - broadcast tree[0]."""
        ctx = self.ctx
        for it in range(n_iters):
            for i in range(self.UNROLL):
                self.emit(valu=[("vbroadcast", ctx.v_nv[it][i], ctx.tc[0])])

        # Stage 5 opt
        for it in range(n_iters):
            for i in range(self.UNROLL):
                self.emit(valu=[("^", ctx.v_nv[it][i], ctx.v_nv[it][i], ctx.v_stage5_xor)])

    def _emit_depth1_access(self, n_iters: int, rnd: int):
        """Depth 1: Delta selection for 2 values."""
        ctx = self.ctx
        for it in range(n_iters):
            base_g = it * self.UNROLL
            for i in range(self.UNROLL):
                t = ctx.v_t[it][i]
                self.emit(valu=[("==", t, ctx.all_idx[base_g+i], ctx.v_1)])
                self.emit(valu=[("*", t, ctx.v_td_1, t)])
                self.emit(valu=[("+", ctx.v_nv[it][i], ctx.v_tc2, t)])

        # Stage 5 opt
        for it in range(n_iters):
            for i in range(self.UNROLL):
                self.emit(valu=[("^", ctx.v_nv[it][i], ctx.v_nv[it][i], ctx.v_stage5_xor)])

    def _emit_depth2_access(self, n_iters: int, rnd: int):
        """Depth 2: Delta selection for 4 values."""
        ctx = self.ctx
        for it in range(n_iters):
            base_g = it * self.UNROLL
            for i in range(self.UNROLL):
                t = ctx.v_t[it][i]
                # Start with tc[3]
                self.emit(valu=[("==", t, ctx.all_idx[base_g+i], ctx.v_const[4])])
                self.emit(valu=[("*", t, ctx.v_td_4, t)])
                self.emit(valu=[("+", ctx.v_nv[it][i], ctx.v_tc3, t)])
                # Add td_5 if idx == 5
                self.emit(valu=[("==", t, ctx.all_idx[base_g+i], ctx.v_const[5])])
                self.emit(valu=[("*", t, ctx.v_td_5, t)])
                self.emit(valu=[("+", ctx.v_nv[it][i], ctx.v_nv[it][i], t)])
                # Add td_6 if idx == 6
                self.emit(valu=[("==", t, ctx.all_idx[base_g+i], ctx.v_const[6])])
                self.emit(valu=[("*", t, ctx.v_td_6, t)])
                self.emit(valu=[("+", ctx.v_nv[it][i], ctx.v_nv[it][i], t)])

        # Stage 5 opt
        for it in range(n_iters):
            for i in range(self.UNROLL):
                self.emit(valu=[("^", ctx.v_nv[it][i], ctx.v_nv[it][i], ctx.v_stage5_xor)])

    def _emit_depth3_access(self, n_iters: int, rnd: int):
        """Depth 3: Compare/select from preloaded tree[7..14]."""
        ctx = self.ctx
        for it in range(n_iters):
            base_g = it * self.UNROLL
            for i in range(self.UNROLL):
                t = ctx.v_t[it][i]
                # Initialize result to 0
                self.emit(valu=[("*", ctx.v_nv[it][i], ctx.v_0, ctx.v_0)])
                # For each of 8 possible values
                for ti in range(8):
                    self.emit(valu=[("==", t, ctx.all_idx[base_g+i], ctx.v_const_r3[ti])])
                    self.emit(valu=[("multiply_add", ctx.v_nv[it][i], t, ctx.v_tc_r3[ti], ctx.v_nv[it][i])])

        # Stage 5 opt
        for it in range(n_iters):
            for i in range(self.UNROLL):
                self.emit(valu=[("^", ctx.v_nv[it][i], ctx.v_nv[it][i], ctx.v_stage5_xor)])

    def _emit_scattered_access(self, n_iters: int, rnd: int):
        """Scattered loads for depths 4+."""
        ctx = self.ctx
        n_groups = n_iters * self.UNROLL

        # Gather from forest - scattered loads
        for vi in range(VLEN):
            for i in range(0, self.UNROLL, 2):
                for it in range(n_iters):
                    base_g = it * self.UNROLL
                    st_a, st_b = ctx.st[2*it], ctx.st[2*it+1]
                    self.emit(alu=[
                        ("+", st_a, ctx.forest_p, ctx.all_idx[base_g+i] + vi),
                        ("+", st_b, ctx.forest_p, ctx.all_idx[base_g+i+1] + vi)
                    ])
                    self.emit(load=[
                        ("load", ctx.v_nv[it][i] + vi, st_a),
                        ("load", ctx.v_nv[it][i+1] + vi, st_b)
                    ])

        # Stage 5 optimization: XOR gathered values with C
        for it in range(n_iters):
            for i in range(self.UNROLL):
                self.emit(valu=[("^", ctx.v_nv[it][i], ctx.v_nv[it][i], ctx.v_stage5_xor)])

    def _emit_sorted_vload_access(self, n_iters: int, tree_depth: int, rnd: int):
        """Optimized sorted vload-based tree access.

        With items sorted by tree index, all groups span a small index range.
        We load the tree cache ONCE per round (not per group), then all groups
        select from this shared cache.

        Cost analysis:
        - 1 vstore + 1 load (extract min index) = ~2 cycles
        - 1 vload (tree cache) = 1 cycle
        - 1 broadcast (cache base) = 1 cycle
        - Per group: 1 subtract + 8-value selection = ~17 VALU ops
        - Total: ~4 cycles setup + ~90 cycles VALU = ~94 cycles

        Compare to scattered loads: 128 cycles
        Savings: ~34 cycles per round

        NOTE: This assumes items are sorted. If not sorted, some items may
        have offset > 7 and get incorrect values. Full partition mechanism
        ensures sorted property.
        """
        ctx = self.ctx

        # Step 1: Extract minimum index from first group
        # With sorted items, first element of first group is the minimum
        self.emit(store=[("vstore", ctx.idx_p, ctx.all_idx[0])])
        self.emit(load=[("load", self.tree_cache_base_s, ctx.idx_p)])

        # Step 2: vload 8 tree values starting at cache_base
        self.emit(alu=[("+", ctx.st[0], ctx.forest_p, self.tree_cache_base_s)])
        self.emit(load=[("vload", self.tree_cache_scalars, ctx.st[0])])

        # Step 3: Broadcast cache_base to vector for offset computation
        self.emit(valu=[("vbroadcast", self.v_cache_base, self.tree_cache_base_s)])

        # Step 4: Broadcast all 8 cache values to vectors
        # This is done once, shared across all groups
        v_tc = []
        for ci in range(8):
            v_tc.append(ctx.v_t[ci // self.UNROLL][ci % self.UNROLL])
            self.emit(valu=[("vbroadcast", v_tc[ci], self.tree_cache_scalars + ci)])

        # Step 5: For each group, compute offset and select
        for it in range(n_iters):
            base_g = it * self.UNROLL
            for i in range(self.UNROLL):
                g = base_g + i
                # Use a temp from a different iteration to avoid conflicts
                t_offset = ctx.v_nv[it][i]  # Reuse v_nv as temp

                # Compute offset = idx - cache_base
                self.emit(valu=[("-", t_offset, ctx.all_idx[g], self.v_cache_base)])

                # 8-value selection: result = sum((offset == i) * tree_cache[i])
                t_result = self.v_tree_temp
                self.emit(valu=[("*", t_result, ctx.v_0, ctx.v_0)])  # init to 0

                for ci in range(8):
                    # mask = (offset == ci)
                    t_mask = ctx.v_t[(it + 2) % n_iters][i]  # Use different temp
                    self.emit(valu=[("==", t_mask, t_offset, ctx.v_const[ci])])
                    # result += mask * tree_cache[ci]
                    self.emit(valu=[("multiply_add", t_result, t_mask, v_tc[ci], t_result)])

                # Copy result to v_nv and apply Stage 5 XOR
                self.emit(valu=[("^", ctx.v_nv[it][i], t_result, ctx.v_stage5_xor)])

    def _emit_xor(self, n_groups: int, rnd: int):
        """Emit XOR of all_val with v_nv."""
        ctx = self.ctx
        n_iters = n_groups // self.UNROLL

        for it in range(n_iters):
            base_g = it * self.UNROLL
            for i in range(self.UNROLL):
                g = base_g + i
                self.emit(valu=[("^", ctx.all_val[g], ctx.all_val[g], ctx.v_nv[it][i])])

    def _emit_hash(self, n_groups: int, n_iters: int, rnd: int):
        """Emit 6-stage hash."""
        ctx = self.ctx

        for hi in range(6):
            op1, _, op2, op3, _ = HASH_STAGES[hi]

            for it in range(n_iters):
                base_g = it * self.UNROLL
                for i in range(self.UNROLL):
                    g = base_g + i

                    if ctx.hash_fma_mult[hi] is not None:
                        self.emit(valu=[("multiply_add", ctx.all_val[g],
                                        ctx.all_val[g], ctx.v_hash_mult[hi], ctx.v_hc[hi])])
                    elif hi == 5:
                        t = ctx.v_t[it][i]
                        self.emit(valu=[(">>", t, ctx.all_val[g], ctx.v_hs[hi])])
                        self.emit(valu=[("^", ctx.all_val[g], ctx.all_val[g], t)])
                    else:
                        t = ctx.v_t[it][i]
                        self.emit(valu=[(op1, t, ctx.all_val[g], ctx.v_hc[hi]),
                                       (op3, ctx.v_nv[it][i], ctx.all_val[g], ctx.v_hs[hi])])
                        self.emit(valu=[(op2, ctx.all_val[g], t, ctx.v_nv[it][i])])

    def _emit_branch(self, n_groups: int, n_iters: int, rnd: int):
        """Emit branch computation with static wrap handling and partition stores.

        The branch computation updates indices:
            new_idx = 2 * old_idx + (2 - (val & 1))

        Partition stores happen AFTER branch computation using idle store slots,
        based on tree DEPTH (not round) for proper wrap handling:
        - Depths 0-2: TICK stores (first half)
        - Depth 3: TOCK stores (second half)
        """
        ctx = self.ctx

        tree_height = self.config.forest_height + 1
        tree_depth = rnd % tree_height
        is_wrap_round = (tree_depth == self.config.forest_height)

        # Emit branch computation for all groups
        for it in range(n_iters):
            base_g = it * self.UNROLL
            for i in range(self.UNROLL):
                g = base_g + i
                t = ctx.v_t[it][i]

                if is_wrap_round:
                    # At leaf level: ALL items wrap to index 0
                    self.emit(valu=[("*", ctx.all_idx[g], ctx.v_0, ctx.v_0)])
                else:
                    # Normal branching
                    self.emit(valu=[("&", t, ctx.all_val[g], ctx.v_1)])
                    self.emit(valu=[("-", t, ctx.v_2, t)])
                    self.emit(valu=[("multiply_add", ctx.all_idx[g], ctx.all_idx[g], ctx.v_2, t)])

        # Emit partition stores based on DEPTH (handles wrap correctly)
        # The scheduler will interleave these with hash computation of next round
        if tree_depth in self.TICK_STORE_DEPTHS:
            # Store first half: groups 0-15
            self._emit_partition_stores_batch(0, n_groups // 2, rnd)
        elif tree_depth in self.TOCK_STORE_DEPTHS:
            # Store second half: groups 16-31
            self._emit_partition_stores_batch(n_groups // 2, n_groups, rnd)

    def _emit_partition_stores_batch(self, start_g: int, end_g: int, rnd: int):
        """Emit partition stores for a range of groups with optimized addressing.

        Uses incremental address computation to minimize overhead:
        - Compute base address once
        - Increment by VLEN for each group

        This minimizes the load/ALU overhead that would otherwise eat into
        the "free" store slots.
        """
        ctx = self.ctx
        n_groups_to_store = end_g - start_g

        # Compute base addresses once
        base_offset = start_g * VLEN
        self.emit(load=[("const", ctx.st[8], base_offset)])
        self.emit(alu=[("+", ctx.st[9], self.extra_room_p, ctx.st[8])])  # idx base
        self.emit(alu=[("+", ctx.st[10], ctx.st[9], self.partition_val_offset)])  # val base
        self.emit(load=[("const", ctx.st[11], VLEN)])  # increment constant

        # Store all groups with incremental addressing
        for i, g in enumerate(range(start_g, end_g)):
            # Store indices and values
            self.emit(store=[
                ("vstore", ctx.st[9], ctx.all_idx[g]),
                ("vstore", ctx.st[10], ctx.all_val[g])
            ])

            # Increment addresses for next group (except for last)
            if g < end_g - 1:
                self.emit(alu=[
                    ("+", ctx.st[9], ctx.st[9], ctx.st[11]),
                    ("+", ctx.st[10], ctx.st[10], ctx.st[11])
                ])

    def _schedule(self):
        """Apply scheduling."""
        from solver.reschedule import (
            extract_ops, compute_dependencies,
            schedule_greedy_multipass, emit_scheduled
        )

        ops = extract_ops(self.instrs)
        op_by_id = compute_dependencies(ops)
        print(f"Scheduling {len(ops)} ops...")

        import time
        start = time.time()
        num_cycles, scheduled_ops = schedule_greedy_multipass(
            ops, op_by_id,
            enable_valu_shift=self.config.enable_valu_shift,
            enable_broadcast_lookahead=self.config.enable_broadcast_lookahead
        )
        scheduled = emit_scheduled(scheduled_ops, num_cycles)
        print(f"Multipass greedy scheduling completed in {time.time() - start:.2f}s")

        self.instrs = [{"flow": [("pause",)]}] + scheduled + [{"flow": [("pause",)]}]
        print(f"Final scheduling: {len(self.instrs)} instructions ({num_cycles} cycles)")
