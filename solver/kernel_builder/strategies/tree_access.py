"""
Tree access strategies for kernel building.

This module contains the base TreeAccessStrategy and simple implementations:
- ScatteredLoadStrategy: Baseline approach using individual scalar loads
- VALUSelectionStrategy: Uses VALU compare+select for shallow depths
- HybridTreeAccessStrategy: Combines VALU selection with scattered loads
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from problem import VLEN

if TYPE_CHECKING:
    from ..base import BaseKernelBuilder
    from ..context import ScratchContext


class TreeAccessStrategy(ABC):
    """Abstract base class for tree access strategies.

    Tree access is the most expensive part of the kernel - 256 batch elements
    need to read from potentially different tree locations each round.

    Strategies trade off between:
    - Scattered loads: Simple but 128 cycles/round (256 loads @ 2/cycle)
    - VALU selection: Preload + compare/select, faster for small index ranges
    - Chunked gather: vload + selection, exploits collision patterns
    """

    @abstractmethod
    def allocate(self, builder: 'BaseKernelBuilder', ctx: 'ScratchContext') -> None:
        """Allocate strategy-specific scratch memory."""
        pass

    @abstractmethod
    def emit_init(self, builder: 'BaseKernelBuilder', ctx: 'ScratchContext') -> None:
        """Emit initialization code (load constants, preload tree values)."""
        pass

    @abstractmethod
    def emit_access(
        self,
        builder: 'BaseKernelBuilder',
        ctx: 'ScratchContext',
        tree_depth: int,
        rnd: int,
    ) -> None:
        """Emit tree access code for all groups at given depth."""
        pass


class ScatteredLoadStrategy(TreeAccessStrategy):
    """Load tree values using individual scalar loads.

    This is the baseline approach: for each of 256 batch elements, compute
    the tree address and issue a load instruction.

    Cost: 256 loads at 2 loads/cycle = 128 cycles per round
    Plus 256 ALU ops for address calculation = 21 cycles (12 ALU/cycle)
    Total: ~150 cycles per round for tree access

    Simple but expensive. Used as fallback for deep tree levels where
    VALU selection would be even more expensive.
    """

    def __init__(self):
        self.UNROLL = 8

    def allocate(self, builder: 'BaseKernelBuilder', ctx: 'ScratchContext') -> None:
        # No additional allocation needed - uses shared temps
        pass

    def emit_init(self, builder: 'BaseKernelBuilder', ctx: 'ScratchContext') -> None:
        # No initialization needed for scattered loads
        pass

    def emit_access(
        self,
        builder: 'BaseKernelBuilder',
        ctx: 'ScratchContext',
        tree_depth: int,
        rnd: int,
    ) -> None:
        """Emit scattered load access for all groups."""
        n_groups = builder.config.batch_size // VLEN
        n_iters = n_groups // self.UNROLL

        # Gather from forest - scattered loads
        for vi in range(VLEN):
            for i in range(0, self.UNROLL, 2):
                for it in range(n_iters):
                    base_g = it * self.UNROLL
                    st_a, st_b = ctx.st[2*it], ctx.st[2*it+1]
                    builder.emit(alu=[
                        ("+", st_a, ctx.forest_p, ctx.all_idx[base_g+i] + vi),
                        ("+", st_b, ctx.forest_p, ctx.all_idx[base_g+i+1] + vi)
                    ])
                    builder.emit(load=[
                        ("load", ctx.v_nv[it][i] + vi, st_a),
                        ("load", ctx.v_nv[it][i+1] + vi, st_b)
                    ])

        # Stage 5 optimization: XOR gathered values with C
        for it in range(n_iters):
            for i in range(self.UNROLL):
                builder.emit(valu=[("^", ctx.v_nv[it][i], ctx.v_nv[it][i], ctx.v_stage5_xor)])


class VALUSelectionStrategy(TreeAccessStrategy):
    """Use VALU compare+select for tree access at shallow depths.

    For depths 0-3, preload all possible tree values and use compare/multiply_add
    to select the correct value. Much faster than scattered loads when the number
    of possible values is small.

    Cost analysis per group:
    - Depth 0 (1 value): 1 broadcast = 1 VALU op
    - Depth 1 (2 values): delta method = 3 VALU ops
    - Depth 2 (4 values): delta method = 7 VALU ops
    - Depth 3 (8 values): compare/select = 16 VALU ops (2 per value)

    At 6 VALU slots/cycle with 32 groups:
    - Depth 0: 32 ops / 6 = 6 cycles
    - Depth 1: 96 ops / 6 = 16 cycles
    - Depth 2: 224 ops / 6 = 38 cycles
    - Depth 3: 512 ops / 6 = 86 cycles

    Compare to scattered loads: 150 cycles
    Depth 3 VALU selection (86 cycles) beats scattered loads (150 cycles)!
    """

    def __init__(self, max_depth: int = 3):
        self.max_depth = max_depth
        self.UNROLL = 8

    def allocate(self, builder: 'BaseKernelBuilder', ctx: 'ScratchContext') -> None:
        """Allocate depth 3 cache if using VALU selection for depth 3."""
        if self.max_depth >= 3:
            ctx.tc_r3 = [builder.alloc(f"tc_r3_{i}") for i in range(8)]
            ctx.v_tc_r3 = [builder.vec(f"v_tc_r3_{i}") for i in range(8)]
            ctx.v_const_r3 = [builder.vec(f"v_const_r3_{i}") for i in range(8)]
            ctx.c_r3 = [builder.alloc(f"c_r3_{i}") for i in range(8)]

    def emit_init(self, builder: 'BaseKernelBuilder', ctx: 'ScratchContext') -> None:
        """Emit depth 3 initialization: load tree[7..14] and broadcast."""
        if self.max_depth >= 3 and ctx.tc_r3 is not None:
            # Load constants 7-14
            for i in range(8):
                builder.emit(load=[("const", ctx.c_r3[i], 7 + i)])

            # Load tree[7..14]
            for i in range(8):
                builder.emit(alu=[("+", ctx.st[0], ctx.forest_p, ctx.c_r3[i])])
                builder.emit(load=[("load", ctx.tc_r3[i], ctx.st[0])])

            # Broadcast tree values to vectors
            for i in range(0, 8, 2):
                builder.emit(valu=[
                    ("vbroadcast", ctx.v_tc_r3[i], ctx.tc_r3[i]),
                    ("vbroadcast", ctx.v_tc_r3[i+1], ctx.tc_r3[i+1])
                ])

            # Broadcast comparison constants to vectors
            for i in range(0, 8, 2):
                builder.emit(valu=[
                    ("vbroadcast", ctx.v_const_r3[i], ctx.c_r3[i]),
                    ("vbroadcast", ctx.v_const_r3[i+1], ctx.c_r3[i+1])
                ])

    def emit_access(
        self,
        builder: 'BaseKernelBuilder',
        ctx: 'ScratchContext',
        tree_depth: int,
        rnd: int,
    ) -> None:
        """Emit VALU selection access for depths 0-3."""
        n_groups = builder.config.batch_size // VLEN
        n_iters = n_groups // self.UNROLL

        if tree_depth == 0:
            self._emit_depth0(builder, ctx, n_iters, rnd)
        elif tree_depth == 1:
            self._emit_depth1(builder, ctx, n_iters, rnd)
        elif tree_depth == 2:
            self._emit_depth2(builder, ctx, n_iters, rnd)
        elif tree_depth == 3:
            self._emit_depth3(builder, ctx, n_iters, rnd)
        else:
            raise ValueError(f"VALU selection only supports depths 0-3, got {tree_depth}")

    def _emit_depth0(self, builder, ctx, n_iters, rnd):
        """Depth 0: All at root - just broadcast tree[0]."""
        for it in range(n_iters):
            for i in range(self.UNROLL):
                builder.emit(valu=[("vbroadcast", ctx.v_nv[it][i], ctx.tc[0])])

        # Stage 5 opt: XOR v_nv with C
        for it in range(n_iters):
            for i in range(self.UNROLL):
                builder.emit(valu=[("^", ctx.v_nv[it][i], ctx.v_nv[it][i], ctx.v_stage5_xor)])

        # Debug
        for it in range(n_iters):
            base_g = it * self.UNROLL
            for i in range(self.UNROLL):
                g = base_g + i
                keys = [(rnd, g * VLEN + vi, "node_val_xored") for vi in range(VLEN)]
                builder.emit_vcompare(ctx.v_nv[it][i], keys)
        builder.emit_debug()

    def _emit_depth1(self, builder, ctx, n_iters, rnd):
        """Depth 1: v_nv = tc[2] + (idx == 1) * (tc[1] - tc[2])."""
        for it in range(n_iters):
            base_g = it * self.UNROLL
            for i in range(self.UNROLL):
                t = ctx.v_t[it][i]
                builder.emit(valu=[("==", t, ctx.all_idx[base_g+i], ctx.v_1)])
                builder.emit(valu=[("*", t, ctx.v_td_1, t)])
                builder.emit(valu=[("+", ctx.v_nv[it][i], ctx.v_tc2, t)])

        # Stage 5 opt: XOR v_nv with C
        for it in range(n_iters):
            for i in range(self.UNROLL):
                builder.emit(valu=[("^", ctx.v_nv[it][i], ctx.v_nv[it][i], ctx.v_stage5_xor)])

        # Debug
        for it in range(n_iters):
            base_g = it * self.UNROLL
            for i in range(self.UNROLL):
                g = base_g + i
                keys = [(rnd, g * VLEN + vi, "node_val_xored") for vi in range(VLEN)]
                builder.emit_vcompare(ctx.v_nv[it][i], keys)
        builder.emit_debug()

    def _emit_depth2(self, builder, ctx, n_iters, rnd):
        """Depth 2: delta selection for idx == 3, 4, 5, 6."""
        for it in range(n_iters):
            base_g = it * self.UNROLL
            for i in range(self.UNROLL):
                t = ctx.v_t[it][i]
                # Start with tc[3]
                builder.emit(valu=[("==", t, ctx.all_idx[base_g+i], ctx.v_const[4])])
                builder.emit(valu=[("*", t, ctx.v_td_4, t)])
                builder.emit(valu=[("+", ctx.v_nv[it][i], ctx.v_tc3, t)])
                # Add td_5 if idx == 5
                builder.emit(valu=[("==", t, ctx.all_idx[base_g+i], ctx.v_const[5])])
                builder.emit(valu=[("*", t, ctx.v_td_5, t)])
                builder.emit(valu=[("+", ctx.v_nv[it][i], ctx.v_nv[it][i], t)])
                # Add td_6 if idx == 6
                builder.emit(valu=[("==", t, ctx.all_idx[base_g+i], ctx.v_const[6])])
                builder.emit(valu=[("*", t, ctx.v_td_6, t)])
                builder.emit(valu=[("+", ctx.v_nv[it][i], ctx.v_nv[it][i], t)])

        # Stage 5 opt: XOR v_nv with C
        for it in range(n_iters):
            for i in range(self.UNROLL):
                builder.emit(valu=[("^", ctx.v_nv[it][i], ctx.v_nv[it][i], ctx.v_stage5_xor)])

    def _emit_depth3(self, builder, ctx, n_iters, rnd):
        """Depth 3: compare/select from preloaded tree[7..14]."""
        if ctx.v_tc_r3 is None:
            raise ValueError("Depth 3 VALU selection requires tc_r3 allocation")

        for it in range(n_iters):
            base_g = it * self.UNROLL
            for i in range(self.UNROLL):
                t = ctx.v_t[it][i]
                # Initialize result to 0
                builder.emit(valu=[("*", ctx.v_nv[it][i], ctx.v_0, ctx.v_0)])
                # For each of 8 possible values
                for ti in range(8):
                    builder.emit(valu=[("==", t, ctx.all_idx[base_g+i], ctx.v_const_r3[ti])])
                    builder.emit(valu=[("multiply_add", ctx.v_nv[it][i], t, ctx.v_tc_r3[ti], ctx.v_nv[it][i])])

        # Stage 5 opt: XOR v_nv with C
        for it in range(n_iters):
            for i in range(self.UNROLL):
                builder.emit(valu=[("^", ctx.v_nv[it][i], ctx.v_nv[it][i], ctx.v_stage5_xor)])


class HybridTreeAccessStrategy(TreeAccessStrategy):
    """Combine VALU selection (shallow) with scattered loads (deep).

    Uses VALU selection for depths 0-3 where it's faster, falls back to
    scattered loads for depths 4+ where VALU selection becomes too expensive.

    This is the current best strategy, achieving 1880 cycles with:
    - Depths 0-3: VALU selection (preloaded at init)
    - Depths 4+: Scattered loads

    The tree_depth wraps at tree_height (forest_height + 1 = 11), so
    post-wrap rounds automatically reuse the same efficient VALU selection
    code paths for depths 0-3.
    """

    def __init__(self, valu_max_depth: int = 3):
        self.valu_strategy = VALUSelectionStrategy(max_depth=valu_max_depth)
        self.scattered_strategy = ScatteredLoadStrategy()
        self.valu_max_depth = valu_max_depth
        self.UNROLL = 8

    def allocate(self, builder: 'BaseKernelBuilder', ctx: 'ScratchContext') -> None:
        self.valu_strategy.allocate(builder, ctx)
        self.scattered_strategy.allocate(builder, ctx)

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
        if tree_depth <= self.valu_max_depth:
            self.valu_strategy.emit_access(builder, ctx, tree_depth, rnd)
        else:
            self.scattered_strategy.emit_access(builder, ctx, tree_depth, rnd)
