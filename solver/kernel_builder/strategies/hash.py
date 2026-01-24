"""
Hash computation strategies for kernel building.

This module contains hash strategies that compute the 6-stage hash function.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from problem import VLEN, HASH_STAGES

if TYPE_CHECKING:
    from ..base import BaseKernelBuilder
    from ..context import ScratchContext


class HashStrategy(ABC):
    """Abstract base class for hash computation strategies.

    The hash function has 6 stages, each with operations:
        stage(a) = (a op1 C) op2 (a op3 S)

    Where op1, op2, op3 are operations (+, ^, <<, >>), C is a constant, S is a shift.

    FMA optimization: For stages where op1='+', op2='+', op3='<<':
        (a + C) + (a << S) = a * (1 + 2^S) + C
    This reduces 3 ops to 1 multiply_add instruction.

    Stage 5 optimization: Skip XOR with constant C, compensate by:
    1. Pre-XOR tree values with C
    2. Use flipped branch direction (2 - bit instead of 1 + bit)
    3. XOR stored values with C at the end
    """

    @abstractmethod
    def allocate(self, builder: 'BaseKernelBuilder', ctx: 'ScratchContext') -> None:
        pass

    @abstractmethod
    def emit_init(self, builder: 'BaseKernelBuilder', ctx: 'ScratchContext') -> None:
        pass

    @abstractmethod
    def emit_xor(
        self,
        builder: 'BaseKernelBuilder',
        ctx: 'ScratchContext',
        rnd: int,
    ) -> None:
        """Emit XOR of all_val with v_nv (tree values)."""
        pass

    @abstractmethod
    def emit_hash(
        self,
        builder: 'BaseKernelBuilder',
        ctx: 'ScratchContext',
        rnd: int,
    ) -> None:
        """Emit 6-stage hash computation."""
        pass

    @abstractmethod
    def emit_branch(
        self,
        builder: 'BaseKernelBuilder',
        ctx: 'ScratchContext',
        rnd: int,
    ) -> None:
        """Emit branch computation: idx = idx * 2 + (1 or 2 depending on val LSB)."""
        pass


class VALUHashStrategy(HashStrategy):
    """Compute hash using only VALU operations (baseline).

    This is the standard approach using vector ALU for all hash computation.
    Uses FMA optimization for eligible stages and Stage 5 XOR optimization.
    """

    def __init__(self):
        self.UNROLL = 8

    def allocate(self, builder: 'BaseKernelBuilder', ctx: 'ScratchContext') -> None:
        # Calculate FMA eligibility for each stage
        ctx.hash_fma_mult = [None] * 6
        for i, (op1, const, op2, op3, shift) in enumerate(HASH_STAGES):
            if op1 == '+' and op2 == '+' and op3 == '<<':
                ctx.hash_fma_mult[i] = 1 + (1 << shift)

    def emit_init(self, builder: 'BaseKernelBuilder', ctx: 'ScratchContext') -> None:
        """Emit FMA multiplier initialization."""
        for i in range(6):
            if ctx.hash_fma_mult[i] is not None:
                mult_s = builder.alloc(f"hash_mult_{i}")
                ctx.hash_fma_mult_s[i] = mult_s
                v_mult = builder.vec(f"v_hash_mult_{i}")
                ctx.v_hash_mult[i] = v_mult
                builder.emit(load=[("const", mult_s, ctx.hash_fma_mult[i])])
                builder.emit(valu=[("vbroadcast", v_mult, mult_s)])

    def emit_xor(
        self,
        builder: 'BaseKernelBuilder',
        ctx: 'ScratchContext',
        rnd: int,
    ) -> None:
        """Emit XOR of all_val with v_nv."""
        n_groups = builder.config.batch_size // VLEN
        n_iters = n_groups // self.UNROLL

        for it in range(n_iters):
            base_g = it * self.UNROLL
            for i in range(self.UNROLL):
                g = base_g + i
                builder.emit(valu=[("^", ctx.all_val[g], ctx.all_val[g], ctx.v_nv[it][i])])

        # Debug
        for g in range(n_groups):
            keys = [(rnd, g * VLEN + vi, "val_after_xor") for vi in range(VLEN)]
            builder.emit_vcompare(ctx.all_val[g], keys)
        builder.emit_debug()

    def emit_hash(
        self,
        builder: 'BaseKernelBuilder',
        ctx: 'ScratchContext',
        rnd: int,
    ) -> None:
        """Emit 6-stage hash with FMA and Stage 5 optimizations."""
        n_groups = builder.config.batch_size // VLEN
        n_iters = n_groups // self.UNROLL

        for hi in range(6):
            op1, _, op2, op3, _ = HASH_STAGES[hi]

            for it in range(n_iters):
                base_g = it * self.UNROLL
                for i in range(self.UNROLL):
                    g = base_g + i

                    if ctx.hash_fma_mult[hi] is not None:
                        # FMA: val = val * mult + const
                        builder.emit(valu=[("multiply_add", ctx.all_val[g],
                                          ctx.all_val[g], ctx.v_hash_mult[hi], ctx.v_hc[hi])])
                    elif hi == 5:
                        # Stage 5 optimized: a ^ (a >> 16)
                        t = ctx.v_t[it][i]
                        builder.emit(valu=[(">>", t, ctx.all_val[g], ctx.v_hs[hi])])
                        builder.emit(valu=[("^", ctx.all_val[g], ctx.all_val[g], t)])
                    else:
                        # Standard: (a op1 C) op2 (a op3 S)
                        t = ctx.v_t[it][i]
                        builder.emit(valu=[(op1, t, ctx.all_val[g], ctx.v_hc[hi]),
                                          (op3, ctx.v_nv[it][i], ctx.all_val[g], ctx.v_hs[hi])])
                        builder.emit(valu=[(op2, ctx.all_val[g], t, ctx.v_nv[it][i])])

            # Debug
            for g in range(n_groups):
                keys = [(rnd, g * VLEN + vi, "hash_stage", hi) for vi in range(VLEN)]
                builder.emit_vcompare(ctx.all_val[g], keys)
            builder.emit_debug()

    def emit_branch(
        self,
        builder: 'BaseKernelBuilder',
        ctx: 'ScratchContext',
        rnd: int,
    ) -> None:
        """Emit branch computation with static wrap handling.

        Key insight: We know at compile time exactly when wrapping occurs.
        - At depth == forest_height (the leaf level), ALL items wrap to 0
        - At all other depths, indices stay valid (no bounds check needed)

        This eliminates runtime bounds checking entirely.
        """
        n_groups = builder.config.batch_size // VLEN
        n_iters = n_groups // self.UNROLL

        tree_height = builder.config.forest_height + 1
        tree_depth = rnd % tree_height
        is_wrap_round = (tree_depth == builder.config.forest_height)

        for it in range(n_iters):
            base_g = it * self.UNROLL
            for i in range(self.UNROLL):
                g = base_g + i
                t = ctx.v_t[it][i]

                if is_wrap_round:
                    # At leaf level: ALL items wrap to index 0
                    builder.emit(valu=[("*", ctx.all_idx[g], ctx.v_0, ctx.v_0)])
                else:
                    # Normal branching - no bounds check needed
                    builder.emit(valu=[("&", t, ctx.all_val[g], ctx.v_1)])
                    builder.emit(valu=[("-", t, ctx.v_2, t)])
                    builder.emit(valu=[("multiply_add", ctx.all_idx[g], ctx.all_idx[g], ctx.v_2, t)])

        # Debug
        for g in range(n_groups):
            keys = [(rnd, g * VLEN + vi, "wrapped_idx") for vi in range(VLEN)]
            builder.emit_vcompare(ctx.all_idx[g], keys)
        builder.emit_debug()
