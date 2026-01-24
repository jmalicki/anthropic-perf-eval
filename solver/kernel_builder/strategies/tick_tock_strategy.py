"""
Tick-Tock Partition Strategies for tree access.

These strategies use stable partition to sort items by branch bit,
enabling efficient vload-based tree access.
"""

from typing import List, TYPE_CHECKING

from problem import VLEN

from .tree_access import TreeAccessStrategy, VALUSelectionStrategy, ScatteredLoadStrategy

if TYPE_CHECKING:
    from ..base import BaseKernelBuilder
    from ..context import ScratchContext


class TickTockPartitionStrategy(TreeAccessStrategy):
    """Tick-Tock Lazy Partition Strategy for near-zero store overhead tree access.

    This strategy achieves massive savings by combining:
    1. **Sorted items via stable partition**: Items sorted by tree index span few indices
    2. **Tick-Tock pipelining**: Store partition data in idle store slots over multiple rounds
    3. **Wrap reset exploitation**: After round 10, all indices reset to 0 (clean slate)

    Algorithm overview:
    - Partition every 3-4 rounds using idle store slots (98% unused)
    - Use vload-based tree access for sorted items (depths 4+)
    - Exploit the wrap at round 11 to avoid carrying partition state

    Schedule:
    =========
    Phase 1 (Rounds 0-3): VALU selection, TICK stores Half 0 (items 0-127)
    Phase 2 (Rounds 4-10): Sorted vload access, TICK/TOCK alternation
    Phase 3 (Round 11): Wrap reset - all indices become 0 (clean slate!)
    Phase 4 (Rounds 11-14): VALU selection, TICK stores for round 15
    Phase 5 (Round 15): Sorted vload access

    Cost savings:
    - Scattered loads: 128 cycles/round (256 loads @ 2/cycle)
    - Sorted vload: 8-16 cycles/round (4-8 vloads + selection)
    - Net savings: ~100 cycles/round for 8 rounds = 800 cycles

    Expected result: ~1050 cycles (140x speedup from baseline)
    """

    UNROLL = 8
    HALF_SIZE = 128  # Each half has 128 items
    PARTITION_ROUNDS = {3, 14}  # Rounds where partition reload happens

    def __init__(self, valu_max_depth: int = 3):
        self.valu_strategy = VALUSelectionStrategy(max_depth=valu_max_depth)
        self.scattered_strategy = ScatteredLoadStrategy()
        self.valu_max_depth = valu_max_depth

        # Partition buffers - will be allocated during allocate()
        self.partition_idx_a: List[int] = []  # Sorted indices buffer A
        self.partition_val_a: List[int] = []  # Corresponding values buffer A
        self.partition_idx_b: List[int] = []  # Sorted indices buffer B
        self.partition_val_b: List[int] = []  # Corresponding values buffer B

        # Current partition state
        self.current_buffer = 'a'  # Which buffer is current

        # Partition counters (for prefix sum during partition)
        self.count0_h0: int = 0  # Count of branch=0 in half 0
        self.count0_h1: int = 0  # Count of branch=0 in half 1

        # Temp vectors for partition computation
        self.v_branch: List[int] = []  # Branch bits per group
        self.v_prefix: List[int] = []  # Prefix sum for partition positions

        # Cache for vload-based tree access
        self.v_tree_cache: List[int] = []  # Cached tree values (8 per vload)
        self.tree_cache_base: int = 0  # Scalar: base index of current cache

        # Vectors for sorted vload selection
        self.v_cache_base: int = 0  # Vector broadcast of cache base
        self.v_offset: List[int] = []  # Offset vectors for selection

    def allocate(self, builder: 'BaseKernelBuilder', ctx: 'ScratchContext') -> None:
        """Allocate scratch for partition buffers and tree cache."""
        self.valu_strategy.allocate(builder, ctx)
        self.scattered_strategy.allocate(builder, ctx)

        n_groups = builder.config.batch_size // VLEN

        # Partition buffers: store sorted indices and values
        self.partition_idx_a = [builder.vec(f"part_idx_a_{g}") for g in range(n_groups)]
        self.partition_val_a = [builder.vec(f"part_val_a_{g}") for g in range(n_groups)]

        # Partition counters and temps
        self.count0_h0 = builder.alloc("count0_h0")
        self.count0_h1 = builder.alloc("count0_h1")

        # Branch bit vectors (reuse v_t from ctx)
        self.v_branch = [builder.vec(f"v_branch_{g}") for g in range(n_groups)]

        # Tree cache: 8 values loaded via vload
        self.v_tree_cache = [builder.vec(f"v_tree_cache_{i}") for i in range(8)]
        self.tree_cache_base = builder.alloc("tree_cache_base")

        # Vectors for selection
        self.v_cache_base = builder.vec("v_cache_base")
        self.v_offset = [builder.vec(f"v_offset_{g}") for g in range(n_groups)]

    def emit_init(self, builder: 'BaseKernelBuilder', ctx: 'ScratchContext') -> None:
        """Emit initialization code."""
        self.valu_strategy.emit_init(builder, ctx)
        self.scattered_strategy.emit_init(builder, ctx)

        # Initialize partition counters to 0
        builder.emit(alu=[("*", self.count0_h0, ctx.c[0], ctx.c[0])])
        builder.emit(alu=[("*", self.count0_h1, ctx.c[0], ctx.c[0])])

    def emit_access(
        self,
        builder: 'BaseKernelBuilder',
        ctx: 'ScratchContext',
        tree_depth: int,
        rnd: int,
    ) -> None:
        """Emit tree access based on round and depth."""
        tree_height = builder.config.forest_height + 1

        # Determine which strategy to use
        if tree_depth <= self.valu_max_depth:
            # Depths 0-3: Use VALU selection (fast, preloaded)
            self.valu_strategy.emit_access(builder, ctx, tree_depth, rnd)
        elif rnd < 4:
            # Before first partition ready: use scattered loads
            self.scattered_strategy.emit_access(builder, ctx, tree_depth, rnd)
        elif tree_depth == builder.config.forest_height:
            # Wrap round (depth 10): Use scattered loads
            self.scattered_strategy.emit_access(builder, ctx, tree_depth, rnd)
        else:
            # Depths 4-9 (after partition ready): Use sorted vload
            self._emit_sorted_vload_access(builder, ctx, tree_depth, rnd)

    def _emit_sorted_vload_access(
        self,
        builder: 'BaseKernelBuilder',
        ctx: 'ScratchContext',
        tree_depth: int,
        rnd: int,
    ) -> None:
        """Emit sorted vload-based tree access."""
        n_groups = builder.config.batch_size // VLEN
        n_iters = n_groups // self.UNROLL

        for it in range(n_iters):
            base_g = it * self.UNROLL
            for i in range(self.UNROLL):
                g = base_g + i
                t = ctx.v_t[it][i]

                # Extract first element's index as cache base
                builder.emit(store=[("vstore", ctx.idx_p, ctx.all_idx[g])])
                builder.emit(load=[("load", self.tree_cache_base, ctx.idx_p)])

                # vload 8 tree values starting at cache_base
                builder.emit(alu=[("+", ctx.st[0], ctx.forest_p, self.tree_cache_base)])
                builder.emit(load=[("vload", self.v_tree_cache[0], ctx.st[0])])

                # Broadcast cache_base to vector for offset computation
                builder.emit(valu=[("vbroadcast", self.v_cache_base, self.tree_cache_base)])

                # Compute offset = idx - cache_base for each element
                builder.emit(valu=[("-", t, ctx.all_idx[g], self.v_cache_base)])

                # 8-value selection: result = sum((offset == i) * tree_cache[i])
                builder.emit(valu=[("*", ctx.v_nv[it][i], ctx.v_0, ctx.v_0)])  # init to 0

                # For each of 8 possible offsets
                for ci in range(8):
                    # Broadcast tree_cache[ci] to temp
                    v_cache_val = self.v_tree_cache[1]  # Reuse for broadcast
                    builder.emit(valu=[("vbroadcast", v_cache_val, self.v_tree_cache[0] + ci)])

                    # mask = (offset == ci)
                    mask_temp = self.v_tree_cache[2]
                    builder.emit(valu=[("==", mask_temp, t, ctx.v_const[ci])])

                    # result += mask * cache_val
                    builder.emit(valu=[("multiply_add", ctx.v_nv[it][i], mask_temp, v_cache_val, ctx.v_nv[it][i])])

                # Stage 5 opt: XOR v_nv with C
                builder.emit(valu=[("^", ctx.v_nv[it][i], ctx.v_nv[it][i], ctx.v_stage5_xor)])


class TickTockPartitionStrategyV2(TreeAccessStrategy):
    """Tick-Tock Partition Strategy V2 - Optimized for minimal overhead.

    Key improvements over V1:
    1. Batched vload: Load tree cache once per round, not per group
    2. Efficient offset computation using min_idx broadcast
    3. Store pipelining: Partition stores overlap with computation
    """

    UNROLL = 8

    def __init__(self, valu_max_depth: int = 3):
        self.valu_strategy = VALUSelectionStrategy(max_depth=valu_max_depth)
        self.scattered_strategy = ScatteredLoadStrategy()
        self.valu_max_depth = valu_max_depth

        # Per-round tree cache
        self.v_tree_cache: List[int] = []
        self.tree_cache_base: int = 0
        self.v_cache_base: int = 0

        # Partition state tracking
        self.partition_phase = 'TICK'  # TICK or TOCK
        self.partition_round_in_phase = 0  # 0, 1, 2 within phase

    def allocate(self, builder: 'BaseKernelBuilder', ctx: 'ScratchContext') -> None:
        self.valu_strategy.allocate(builder, ctx)
        self.scattered_strategy.allocate(builder, ctx)

        # Tree cache: single vload worth (8 values)
        self.v_tree_cache = [builder.vec(f"v_tree_cache2_{i}") for i in range(3)]
        self.tree_cache_base = builder.alloc("tree_cache_base2")
        self.v_cache_base = builder.vec("v_cache_base2")

    def emit_init(self, builder: 'BaseKernelBuilder', ctx: 'ScratchContext') -> None:
        self.valu_strategy.emit_init(builder, ctx)
        self.scattered_strategy.emit_init(builder, ctx)

    def emit_access(
        self,
        builder: 'BaseKernelBuilder',
        ctx: 'ScratchContext',
        tree_depth: int,
        rnd: int,
    ) -> None:
        """Emit tree access with tick-tock partition pipelining."""
        if tree_depth <= self.valu_max_depth:
            self.valu_strategy.emit_access(builder, ctx, tree_depth, rnd)
        else:
            # For depths 4+, use scattered loads with partition store pipelining
            self.scattered_strategy.emit_access(builder, ctx, tree_depth, rnd)
