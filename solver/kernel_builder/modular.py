"""
Modular Kernel Builder.

This module provides the main entry point for building kernels with
composable strategies.
"""

from problem import VLEN, HASH_STAGES, SCRATCH_SIZE

from .base import BaseKernelBuilder
from .config import KernelConfig, TreeAccessMode, SchedulingMode
from .context import ScratchContext
from .strategies import (
    TreeAccessStrategy,
    ScatteredLoadStrategy,
    VALUSelectionStrategy,
    HybridTreeAccessStrategy,
    ChunkedVloadStrategy,
    ChunkedDemandDrivenStrategy,
    AdaptiveSelectionStrategy,
    TickTockPartitionStrategy,
    HashStrategy,
    VALUHashStrategy,
)


def create_tree_strategy(config: KernelConfig) -> TreeAccessStrategy:
    """Create tree access strategy based on config."""
    mode = config.tree_access_mode
    if mode == TreeAccessMode.SCATTERED_LOADS:
        return ScatteredLoadStrategy()
    elif mode == TreeAccessMode.VALU_SELECTION:
        return VALUSelectionStrategy(max_depth=config.valu_selection_max_depth)
    elif mode == TreeAccessMode.HYBRID:
        return HybridTreeAccessStrategy(valu_max_depth=config.valu_selection_max_depth)
    elif mode == TreeAccessMode.CHUNKED_GATHER:
        return ChunkedVloadStrategy(
            valu_max_depth=config.valu_selection_max_depth,
            chunked_min_depth=config.chunked_gather_min_depth,
            chunked_max_depth=config.chunked_gather_max_depth,
            chunk_size=config.chunk_size,
        )
    elif mode == TreeAccessMode.DEMAND_DRIVEN:
        return ChunkedDemandDrivenStrategy(
            valu_max_depth=config.valu_selection_max_depth,
        )
    elif mode == TreeAccessMode.ADAPTIVE_SELECTION:
        return AdaptiveSelectionStrategy(
            valu_max_depth=config.valu_selection_max_depth,
        )
    elif mode == TreeAccessMode.TICK_TOCK_PARTITION:
        return TickTockPartitionStrategy(
            valu_max_depth=config.valu_selection_max_depth,
        )
    else:
        raise ValueError(f"Unknown tree access mode: {mode}")


def create_hash_strategy(config: KernelConfig) -> HashStrategy:
    """Create hash strategy based on config."""
    # For now, only VALU is implemented
    return VALUHashStrategy()


class ModularKernelBuilder(BaseKernelBuilder):
    """Build kernels using composable strategies.

    This is the main entry point for building kernels with the modular architecture.

    Example usage (new style):
        config = KernelConfig(
            tree_access_mode=TreeAccessMode.HYBRID,
            hash_mode=HashMode.VALU_ONLY,
            scheduling_mode=SchedulingMode.GREEDY_MULTIPASS,
        )
        builder = ModularKernelBuilder(config)
        builder.build()

    Example usage (legacy style, compatible with old KernelBuilder):
        builder = ModularKernelBuilder()
        builder.build_kernel(forest_height=10, n_nodes=2047, batch_size=256, rounds=16)

        # Access results
        instrs = builder.instrs
        cycles = len(instrs) - 2  # Subtract pause instructions
    """

    UNROLL = 8

    def __init__(self, config: KernelConfig = None):
        if config is None:
            config = KernelConfig()  # Use defaults, will be overwritten by build_kernel()
        super().__init__(config)
        self.tree_strategy = None  # Initialized lazily in build()
        self.hash_strategy = None
        self.ctx = ScratchContext()

    def build_kernel(
        self,
        forest_height: int,
        n_nodes: int,
        batch_size: int,
        rounds: int,
        **kwargs
    ):
        """Legacy interface for compatibility with old KernelBuilder.

        Creates a config and calls build().
        """
        self.config = KernelConfig(
            forest_height=forest_height,
            n_nodes=n_nodes,
            batch_size=batch_size,
            rounds=rounds,
            tree_access_mode=TreeAccessMode.HYBRID,
            scheduling_mode=SchedulingMode.GREEDY_MULTIPASS,
            **kwargs
        )
        # Re-initialize base class state
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.labels = {}
        self.fixups = []
        self.ctx = ScratchContext()

        return self.build()

    def build(self):
        """Build the complete kernel."""
        # Initialize strategies if not already done (e.g., when using build_kernel())
        if self.tree_strategy is None:
            self.tree_strategy = create_tree_strategy(self.config)
        if self.hash_strategy is None:
            self.hash_strategy = create_hash_strategy(self.config)

        n_groups = self.config.batch_size // VLEN
        n_iters = n_groups // self.UNROLL
        tree_height = self.config.forest_height + 1

        # Phase 1: Allocate scratch
        self._allocate_common(n_groups, n_iters)
        self.tree_strategy.allocate(self, self.ctx)
        self.hash_strategy.allocate(self, self.ctx)

        if self.scratch_ptr > SCRATCH_SIZE:
            print(f"WARNING: Scratch usage {self.scratch_ptr} exceeds budget {SCRATCH_SIZE}!")
        else:
            print(f"Scratch usage: {self.scratch_ptr} / {SCRATCH_SIZE}")

        # Phase 2: Init
        self._emit_common_init()
        self.tree_strategy.emit_init(self, self.ctx)
        self.hash_strategy.emit_init(self, self.ctx)

        # Phase 3: Load data
        self._emit_load_data(n_groups)

        # Phase 4: Rounds
        tree_depth = 0
        for rnd in range(self.config.rounds):
            # Tree access
            self.tree_strategy.emit_access(self, self.ctx, tree_depth, rnd)

            # XOR, hash, branch
            self.hash_strategy.emit_xor(self, self.ctx, rnd)
            self.hash_strategy.emit_hash(self, self.ctx, rnd)
            self.hash_strategy.emit_branch(self, self.ctx, rnd)

            # Update tree depth
            tree_depth += 1
            if tree_depth >= tree_height:
                tree_depth = 0

        # Phase 5: Convert values back from optimized form
        for g in range(n_groups):
            self.emit(valu=[("^", self.ctx.all_val[g], self.ctx.all_val[g], self.ctx.v_stage5_xor)])

        # Phase 6: Store data
        self._emit_store_data(n_groups)

        store_end = len(self.instrs)
        print(f"Before scheduling: {store_end} instructions")

        # Phase 7: Schedule
        self._schedule()

        return self

    def _allocate_common(self, n_groups: int, n_iters: int):
        """Allocate common scratch memory."""
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

    def _emit_common_init(self):
        """Emit common initialization code."""
        ctx = self.ctx

        # Load constants 0-15
        for i in range(16):
            self.emit(load=[("const", ctx.c[i], i)])

        # Load pointers
        self.emit(load=[("const", ctx.st[0], 4), ("const", ctx.st[1], 5)])
        self.emit(load=[("load", ctx.forest_p, ctx.st[0]), ("load", ctx.idx_p, ctx.st[1])])
        self.emit(load=[("const", ctx.st[0], 6), ("const", ctx.st[1], 1)])
        self.emit(load=[("load", ctx.val_p, ctx.st[0]), ("load", ctx.n_nodes_s, ctx.st[1])])

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

    def _schedule(self):
        """Apply scheduling based on config using Rust scheduler."""
        import scheduler_rs
        import time

        print(f"Scheduling with Rust scheduler...")

        mode = self.config.scheduling_mode
        if mode == SchedulingMode.GREEDY_SINGLE:
            start = time.time()
            scheduled = scheduler_rs.schedule(
                self.instrs,
                enable_valu_shift=self.config.enable_valu_shift,
            )
            num_cycles = len(scheduled)
            print(f"Greedy scheduling completed in {time.time() - start:.2f}s")

        elif mode == SchedulingMode.GREEDY_MULTIPASS:
            start = time.time()
            scheduled = scheduler_rs.schedule_multipass(
                self.instrs,
                enable_valu_shift=self.config.enable_valu_shift,
            )
            num_cycles = len(scheduled)
            print(f"Multipass greedy scheduling completed in {time.time() - start:.2f}s")

        else:  # CPSAT_OPTIMAL - uses constraint solver, keep Python implementation
            from solver.round_chunked_scheduler import schedule_round_chunked
            from solver.reschedule import extract_ops, compute_dependencies
            ops = extract_ops(self.instrs)
            op_by_id = compute_dependencies(ops)
            n_groups = self.config.batch_size // VLEN
            n_iters = n_groups // self.UNROLL
            num_cycles, scheduled = schedule_round_chunked(
                ops, op_by_id, self.config.rounds, n_iters, self.UNROLL,
                time_limit=60.0,
                full_solve_time=self.config.solve_time,
                require_optimal=self.config.require_optimal,
                enable_valu_shift=self.config.enable_valu_shift,
                valu_shift_time=self.config.valu_shift_time
            )

        self.instrs = [{"flow": [("pause",)]}] + list(scheduled) + [{"flow": [("pause",)]}]
        print(f"Final scheduling: {len(self.instrs)} instructions ({num_cycles} cycles)")
