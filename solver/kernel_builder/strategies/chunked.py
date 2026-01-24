"""
Chunked and demand-driven tree access strategies.

This module contains experimental strategies that use vload-based chunking:
- ChunkedVloadStrategy: Preload tree chunks at init, VALU select at runtime
- ChunkedDemandDrivenStrategy: Demand-driven vload with cache
- AdaptiveSelectionStrategy: Currently equivalent to HYBRID
"""

from typing import Dict, List, TYPE_CHECKING

from problem import VLEN

from .tree_access import TreeAccessStrategy, VALUSelectionStrategy, ScatteredLoadStrategy

if TYPE_CHECKING:
    from ..base import BaseKernelBuilder
    from ..context import ScratchContext


class ChunkedVloadStrategy(TreeAccessStrategy):
    """Use vload to preload tree chunks at init, then VALU select at runtime.

    EXPERIMENTAL: This strategy is currently SLOWER than HybridTreeAccessStrategy.
    It serves as a foundation for future jump-based optimization.

    Key insight: vload loads 8 contiguous values in 1 cycle (vs 4 cycles for
    8 scalar loads). By using vload at init time, we amortize the load cost
    across all rounds.

    With chunk_size=4 and vload=8, one vload covers two chunks:
    - Mode 1: Item in current chunk (select from vload[0:3])
    - Mode 2: Item in next chunk of same vload (select from vload[4:7])
    - Mode 3: Item beyond vload range (need new vload)

    The vload doesn't need 8-byte alignment - we can load tree[idx:idx+8]
    for any idx.

    Cost analysis for depth 4 (16 tree values):
    - Init: 2 vloads + 16 broadcasts = ~10 cycles (amortized over rounds)
    - Runtime: 16 comparisons × 2 ops × 32 groups = 1024 ops = ~170 cycles

    This is WORSE than scattered loads (128 cycles) for depths 4+ because
    the VALU selection cost scales with number of tree values.
    """

    def __init__(
        self,
        valu_max_depth: int = 3,
        chunked_min_depth: int = 4,
        chunked_max_depth: int = 10,
        chunk_size: int = 4,
    ):
        self.valu_strategy = VALUSelectionStrategy(max_depth=valu_max_depth)
        self.scattered_strategy = ScatteredLoadStrategy()
        self.valu_max_depth = valu_max_depth
        self.chunked_min_depth = chunked_min_depth
        self.chunked_max_depth = chunked_max_depth
        self.chunk_size = chunk_size
        self.UNROLL = 8

        # Will be populated during allocate()
        self.chunked_tc: Dict[int, List[int]] = {}  # depth -> scalar tree values
        self.chunked_v_tc: Dict[int, List[int]] = {}  # depth -> vector tree values
        self.chunked_c: Dict[int, List[int]] = {}  # depth -> scalar comparison constants
        self.chunked_v_c: Dict[int, List[int]] = {}  # depth -> vector comparison constants

    def allocate(self, builder: 'BaseKernelBuilder', ctx: 'ScratchContext') -> None:
        """Allocate scratch for VALU selection + chunked gather."""
        self.valu_strategy.allocate(builder, ctx)
        self.scattered_strategy.allocate(builder, ctx)

        # Allocate for each depth in chunked range
        tree_height = builder.config.forest_height + 1
        n_nodes = builder.config.n_nodes

        for depth in range(self.chunked_min_depth, min(self.chunked_max_depth + 1, tree_height)):
            # Calculate tree index range for this depth
            start_idx = (1 << depth) - 1  # 2^depth - 1
            end_idx = (1 << (depth + 1)) - 2  # 2^(depth+1) - 2
            n_vals = end_idx - start_idx + 1  # 2^depth values

            # Only process if within n_nodes
            if start_idx >= n_nodes:
                continue

            # Clamp to n_nodes
            actual_n_vals = min(n_vals, n_nodes - start_idx)

            # Allocate scratch
            self.chunked_tc[depth] = [builder.alloc(f"ctc_d{depth}_{i}") for i in range(actual_n_vals)]
            self.chunked_v_tc[depth] = [builder.vec(f"cv_tc_d{depth}_{i}") for i in range(actual_n_vals)]
            self.chunked_c[depth] = [builder.alloc(f"cc_d{depth}_{i}") for i in range(actual_n_vals)]
            self.chunked_v_c[depth] = [builder.vec(f"cv_c_d{depth}_{i}") for i in range(actual_n_vals)]

    def emit_init(self, builder: 'BaseKernelBuilder', ctx: 'ScratchContext') -> None:
        """Emit init: vload tree chunks and broadcast to vectors."""
        self.valu_strategy.emit_init(builder, ctx)
        self.scattered_strategy.emit_init(builder, ctx)

        # Initialize each depth's tree values using vload
        for depth in sorted(self.chunked_tc.keys()):
            start_idx = (1 << depth) - 1  # First tree index at this depth
            n_vals = len(self.chunked_tc[depth])
            tc_list = self.chunked_tc[depth]
            v_tc_list = self.chunked_v_tc[depth]
            c_list = self.chunked_c[depth]
            v_c_list = self.chunked_v_c[depth]

            # Load comparison constants (tree indices)
            for i in range(n_vals):
                builder.emit(load=[("const", c_list[i], start_idx + i)])

            # Use vload to load tree values in chunks of 8 (VLEN)
            n_chunks = (n_vals + VLEN - 1) // VLEN
            for chunk_i in range(n_chunks):
                chunk_start = chunk_i * VLEN
                chunk_idx = start_idx + chunk_start
                # Compute address: forest_p + chunk_idx
                builder.emit(load=[("const", ctx.st[0], chunk_idx)])
                builder.emit(alu=[("+", ctx.st[1], ctx.forest_p, ctx.st[0])])
                # vload into contiguous scratch starting at tc_list[chunk_start]
                builder.emit(load=[("vload", tc_list[chunk_start], ctx.st[1])])

            # Broadcast tree values to vectors
            for i in range(0, n_vals, 2):
                if i + 1 < n_vals:
                    builder.emit(valu=[
                        ("vbroadcast", v_tc_list[i], tc_list[i]),
                        ("vbroadcast", v_tc_list[i+1], tc_list[i+1])
                    ])
                else:
                    builder.emit(valu=[("vbroadcast", v_tc_list[i], tc_list[i])])

            # Broadcast comparison constants to vectors
            for i in range(0, n_vals, 2):
                if i + 1 < n_vals:
                    builder.emit(valu=[
                        ("vbroadcast", v_c_list[i], c_list[i]),
                        ("vbroadcast", v_c_list[i+1], c_list[i+1])
                    ])
                else:
                    builder.emit(valu=[("vbroadcast", v_c_list[i], c_list[i])])

    def emit_access(
        self,
        builder: 'BaseKernelBuilder',
        ctx: 'ScratchContext',
        tree_depth: int,
        rnd: int,
    ) -> None:
        """Emit tree access: VALU selection for shallow, chunked for mid, scattered for deep."""
        if tree_depth <= self.valu_max_depth:
            # Use existing VALU selection for depths 0-3
            self.valu_strategy.emit_access(builder, ctx, tree_depth, rnd)
        elif tree_depth in self.chunked_tc:
            # Use chunked gather with VALU selection
            self._emit_chunked_access(builder, ctx, tree_depth, rnd)
        else:
            # Fall back to scattered loads for very deep or unconfigured depths
            self.scattered_strategy.emit_access(builder, ctx, tree_depth, rnd)

    def _emit_chunked_access(
        self,
        builder: 'BaseKernelBuilder',
        ctx: 'ScratchContext',
        tree_depth: int,
        rnd: int,
    ) -> None:
        """Emit VALU selection from preloaded chunked tree values."""
        n_groups = builder.config.batch_size // VLEN
        n_iters = n_groups // self.UNROLL

        v_tc_list = self.chunked_v_tc[tree_depth]
        v_c_list = self.chunked_v_c[tree_depth]
        n_vals = len(v_tc_list)

        for it in range(n_iters):
            base_g = it * self.UNROLL
            for i in range(self.UNROLL):
                t = ctx.v_t[it][i]
                # Initialize result to 0
                builder.emit(valu=[("*", ctx.v_nv[it][i], ctx.v_0, ctx.v_0)])
                # For each tree value, check if idx matches and accumulate
                for ti in range(n_vals):
                    builder.emit(valu=[("==", t, ctx.all_idx[base_g+i], v_c_list[ti])])
                    builder.emit(valu=[("multiply_add", ctx.v_nv[it][i], t, v_tc_list[ti], ctx.v_nv[it][i])])

        # Stage 5 opt: XOR v_nv with C
        for it in range(n_iters):
            for i in range(self.UNROLL):
                builder.emit(valu=[("^", ctx.v_nv[it][i], ctx.v_nv[it][i], ctx.v_stage5_xor)])


class ChunkedDemandDrivenStrategy(TreeAccessStrategy):
    """Demand-driven vload with 4-value selection.

    This strategy exploits the key insight that with sorted items (maintained
    via stable partition each round), most items need tree values from a small
    range. We use demand-driven vload to cache 8 tree values, then use
    speculative 4-value selection from both chunks (0-3 and 4-7).

    Algorithm:
    1. For depths 0-3: Use existing VALU selection (already optimal)
    2. For depths 4+:
       a. Check if first item's index is in cache range [cache_base, cache_base+8)
       b. If not: vload new cache, update cache_base, broadcast to v_cache
       c. For each group:
          - Compute offset = (idx - cache_base) & 3  (position within chunk)
          - Compute chunk = ((idx - cache_base) >> 2) & 1  (which chunk: 0 or 1)
          - Select from chunk 0: 4-value selection using offset
          - Select from chunk 1: 4-value selection using offset
          - Blend with vselect based on chunk bit

    Cost analysis:
    - Cache check: 1 vstore + 1 load + few ALU ops = ~2 cycles (once per round)
    - 4-value selection per chunk: 8 VALU ops (== + multiply_add × 4)
    - Both chunks + vselect: 8 + 8 + 1 = 17 ops per group
    - With 32 groups: 544 ops = ~91 VALU cycles

    With sorted items and high collision, cache hits are frequent:
    - Most groups: All items in same chunk → can skip the other chunk (future opt)
    - Rare: Items span chunks → compute both and blend
    - Very rare: Cache miss → reload (1-3 times per round)

    Compared to scattered loads (128 cycles + 21 ALU = 149 cycles), this
    approach can be faster when combined with mode optimization.
    """

    VLOAD_SIZE = 8  # Fixed at VLEN
    CHUNK_SIZE = 4  # Two chunks per vload

    def __init__(self, valu_max_depth: int = 3):
        self.valu_strategy = VALUSelectionStrategy(max_depth=valu_max_depth)
        self.scattered_strategy = ScatteredLoadStrategy()
        self.valu_max_depth = valu_max_depth
        self.UNROLL = 8

        # Cache state - allocated during allocate()
        self.cache_base_s: int = 0  # Scalar: current cache base index
        self.cache_s: int = 0       # Scalar array: cached tree values [0:8]
        self.v_cache: List[int] = []  # Vector broadcasts of cache values
        self.v_cache_base: int = 0  # Vector broadcast of cache_base
        self.v_3: int = 0           # Vector constant 3 (for & 3)
        self.v_4: int = 0           # Vector constant 4 (for chunk selection)
        self.temp_mem: int = 0      # Memory location for vstore/load

    def allocate(self, builder: 'BaseKernelBuilder', ctx: 'ScratchContext') -> None:
        """Allocate scratch for VALU selection + scattered loads."""
        self.valu_strategy.allocate(builder, ctx)
        self.scattered_strategy.allocate(builder, ctx)

        # Reserve cache vectors for future sorted-items optimization
        self.v_cache = [builder.vec(f"v_dd_cache_{i}") for i in range(3)]

    def emit_init(self, builder: 'BaseKernelBuilder', ctx: 'ScratchContext') -> None:
        """Emit initialization code."""
        self.valu_strategy.emit_init(builder, ctx)
        self.scattered_strategy.emit_init(builder, ctx)

    def emit_access(
        self,
        builder: 'BaseKernelBuilder',
        ctx: 'ScratchContext',
        tree_depth: int,
        rnd: int,
    ) -> None:
        """Emit tree access code."""
        if tree_depth <= self.valu_max_depth:
            # Depths 0-3: Use existing VALU selection (already optimal)
            self.valu_strategy.emit_access(builder, ctx, tree_depth, rnd)
        else:
            # Depths 4+: Use 8-value vload cache + selection
            self._emit_cached_8value_access(builder, ctx, tree_depth, rnd)

    def _emit_cached_8value_access(
        self,
        builder: 'BaseKernelBuilder',
        ctx: 'ScratchContext',
        tree_depth: int,
        rnd: int,
    ) -> None:
        """Emit 8-value tree access using vload cache."""
        n_groups = builder.config.batch_size // VLEN
        n_iters = n_groups // self.UNROLL

        # Step 1: Extract first item's index as cache base
        first_idx_s = ctx.st[10]
        cache_base_s = ctx.st[11]
        builder.emit(store=[("vstore", ctx.idx_p, ctx.all_idx[0])])
        builder.emit(load=[("load", first_idx_s, ctx.idx_p)])

        builder.emit(alu=[("=", cache_base_s, first_idx_s, ctx.c[0])])  # cache_base = first_idx

        # Step 2: vload 8 tree values starting at cache_base
        builder.emit(alu=[("+", ctx.st[0], ctx.forest_p, cache_base_s)])
        cache_scalars = self.v_cache[0]  # 8 consecutive scalar locations
        builder.emit(load=[("vload", cache_scalars, ctx.st[0])])

        # Broadcast cache_base to vector for offset computation
        v_cache_base = self.v_cache[2]
        builder.emit(valu=[("vbroadcast", v_cache_base, cache_base_s)])

        # Step 4: For each group, compute offset and do 8-value selection
        for it in range(n_iters):
            base_g = it * self.UNROLL
            for i in range(self.UNROLL):
                g = base_g + i
                t = ctx.v_t[it][i]

                # Compute offset = idx - cache_base (should be 0-7 for cache hits)
                builder.emit(valu=[("-", t, ctx.all_idx[g], v_cache_base)])

                # 8-value selection: result = sum((offset == i) * cache[i]) for i in 0..7
                builder.emit(valu=[("*", ctx.v_nv[it][i], ctx.v_0, ctx.v_0)])  # init to 0
                for ci in range(8):
                    # Broadcast cache[ci] to temp vector
                    v_cache_val = self.v_cache[0]  # Reuse for broadcast
                    builder.emit(valu=[("vbroadcast", v_cache_val, cache_scalars + ci)])
                    # mask = (offset == ci)
                    mask_temp = self.v_cache[1]
                    builder.emit(valu=[("==", mask_temp, t, ctx.v_const[ci])])
                    # result += mask * cache_val
                    builder.emit(valu=[("multiply_add", ctx.v_nv[it][i], mask_temp, v_cache_val, ctx.v_nv[it][i])])

                # Stage 5 opt: XOR v_nv with C
                builder.emit(valu=[("^", ctx.v_nv[it][i], ctx.v_nv[it][i], ctx.v_stage5_xor)])


class AdaptiveSelectionStrategy(TreeAccessStrategy):
    """Adaptive selection strategy - currently equivalent to HYBRID.

    After extensive experimentation, we found that:
    1. VALU N-value selection scales poorly (2N ops per group)
    2. Per-group branching is blocked by flow engine (1 slot/cycle)
    3. Extending VALU to depth 4 (16 values) costs 171 cycles vs 128 for scattered

    The HYBRID approach (VALU for depths 0-3, scattered for 4+) remains optimal.

    Future optimizations would require:
    - Sorted items (via stable partition) to enable broadcast path
    - ISA changes (faster scatter/gather operations)
    """

    def __init__(self, valu_max_depth: int = 3):
        """Initialize adaptive selection strategy (uses HYBRID internally)."""
        from .tree_access import HybridTreeAccessStrategy
        self.hybrid = HybridTreeAccessStrategy(valu_max_depth=valu_max_depth)
        self.valu_max_depth = valu_max_depth

    def allocate(self, builder: 'BaseKernelBuilder', ctx: 'ScratchContext') -> None:
        self.hybrid.allocate(builder, ctx)

    def emit_init(self, builder: 'BaseKernelBuilder', ctx: 'ScratchContext') -> None:
        self.hybrid.emit_init(builder, ctx)

    def emit_access(
        self,
        builder: 'BaseKernelBuilder',
        ctx: 'ScratchContext',
        tree_depth: int,
        rnd: int,
    ) -> None:
        self.hybrid.emit_access(builder, ctx, tree_depth, rnd)
