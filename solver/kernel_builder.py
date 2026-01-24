"""
Modular Kernel Builder Framework

This module provides a composable architecture for building kernels with different
optimization strategies. Instead of monolithic kernel implementations, we separate:

1. **KernelConfig** - All configuration options in one place
2. **BaseKernelBuilder** - Common infrastructure (alloc, emit, scheduling)
3. **Strategy classes** - Composable pieces for different optimizations:
   - TreeAccessStrategy: How to load tree values (scattered, VALU selection, chunked)
   - HashStrategy: How to compute hashes (VALU, ALU-parallel)

The key insight is that the kernel has well-defined phases:
1. INIT: Load constants, preload tree values, setup hash constants
2. LOAD DATA: vload batch idx/val vectors
3. ROUNDS: For each round: tree_access -> XOR -> hash -> branch
4. STORE DATA: vstore results

Each phase can use different strategies while sharing common infrastructure.

Usage:
    config = KernelConfig(
        tree_access_mode=TreeAccessMode.VALU_SELECTION,
        hash_mode=HashMode.VALU_ONLY,
        scheduling_mode=SchedulingMode.GREEDY_MULTIPASS,
    )
    builder = ModularKernelBuilder(config)
    builder.build()

    # Access results
    instrs = builder.instrs
    cycles = len(instrs) - 2  # Subtract pause instructions
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Optional, Tuple, Any, Callable

from problem import DebugInfo, SLOT_LIMITS, VLEN, SCRATCH_SIZE, HASH_STAGES


# =============================================================================
# Configuration Enums
# =============================================================================

class TreeAccessMode(Enum):
    """Strategy for loading tree values during traversal."""
    SCATTERED_LOADS = auto()    # 256 scalar loads per round (baseline)
    VALU_SELECTION = auto()     # Preload + compare/select (depths 0-3)
    CHUNKED_GATHER = auto()     # vload chunks + selection (experimental)
    HYBRID = auto()             # VALU for shallow, scattered for deep


class HashMode(Enum):
    """Strategy for computing hash function."""
    VALU_ONLY = auto()          # All hash ops in VALU (baseline)
    ALU_PARALLEL = auto()       # Split work between ALU and VALU


class SchedulingMode(Enum):
    """Strategy for scheduling operations."""
    GREEDY_SINGLE = auto()      # Single-pass greedy (fast)
    GREEDY_MULTIPASS = auto()   # Multi-pass greedy with different strategies
    CPSAT_OPTIMAL = auto()      # CP-SAT solver (slow but optimal)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class KernelConfig:
    """Configuration for kernel generation.

    All kernel parameters in one place for easy experimentation.
    """
    # Problem parameters
    forest_height: int = 10
    n_nodes: int = 2047
    batch_size: int = 256
    rounds: int = 16

    # Tree access strategy
    tree_access_mode: TreeAccessMode = TreeAccessMode.VALU_SELECTION
    valu_selection_max_depth: int = 3  # Use VALU selection up to this depth

    # Chunked gather (only used with CHUNKED_GATHER or HYBRID modes)
    chunked_gather_min_depth: int = 4
    chunked_gather_max_depth: int = 10
    chunk_size: int = 8

    # Hash computation strategy
    hash_mode: HashMode = HashMode.VALU_ONLY
    alu_hash_fraction: float = 0.5  # Fraction of groups using ALU hash

    # Scheduling strategy
    scheduling_mode: SchedulingMode = SchedulingMode.GREEDY_MULTIPASS
    solve_time: float = 3600.0  # Time limit for CP-SAT
    require_optimal: bool = True

    # VALU→ALU shifting
    enable_valu_shift: bool = True
    valu_shift_time: Optional[float] = None
    enable_broadcast_lookahead: bool = False

    # Debug options
    enable_debug: bool = False


# =============================================================================
# Scratch Allocation Context
# =============================================================================

@dataclass
class ScratchContext:
    """Holds all allocated scratch memory locations.

    This is passed between phases and strategies to share allocations.
    Using a dedicated class makes it easy to track and debug scratch usage.
    """
    # Pointers
    forest_p: int = 0
    idx_p: int = 0
    val_p: int = 0
    n_nodes_s: int = 0

    # Temporaries
    st: List[int] = field(default_factory=list)  # Scalar temps
    c: List[int] = field(default_factory=list)   # Constants 0-15

    # Hash constants
    hc: List[int] = field(default_factory=list)  # Hash constants
    hs: List[int] = field(default_factory=list)  # Hash shifts

    # Tree cache (depths 0-2)
    tc: List[int] = field(default_factory=list)  # tree[0..6]
    td_1: int = 0  # tree[1] - tree[2]
    td_4: int = 0  # tree[4] - tree[3]
    td_5: int = 0  # tree[5] - tree[3]
    td_6: int = 0  # tree[6] - tree[3]

    # Depth 3 cache (optional)
    tc_r3: Optional[List[int]] = None  # tree[7..14]
    c_r3: Optional[List[int]] = None   # Constants 7-14

    # Stage 5 XOR optimization
    stage5_xor_s: int = 0
    STAGE5_XOR_CONST: int = 0xB55A4F09

    # Per-group vectors
    all_idx: List[int] = field(default_factory=list)
    all_val: List[int] = field(default_factory=list)
    v_nv: List[List[int]] = field(default_factory=list)  # [iter][unroll]
    v_t: List[List[int]] = field(default_factory=list)   # [iter][unroll] temps

    # Vector constants
    v_0: int = 0
    v_1: int = 0
    v_2: int = 0
    v_nn: int = 0
    v_const: List[int] = field(default_factory=list)  # [0..7]

    # Vector hash constants
    v_hc: List[int] = field(default_factory=list)
    v_hs: List[int] = field(default_factory=list)

    # Vector tree cache
    v_td_1: int = 0
    v_td_4: int = 0
    v_td_5: int = 0
    v_td_6: int = 0
    v_tc2: int = 0
    v_tc3: int = 0

    # Depth 3 vectors (optional)
    v_tc_r3: Optional[List[int]] = None
    v_const_r3: Optional[List[int]] = None

    # Stage 5 XOR vector
    v_stage5_xor: int = 0

    # FMA multipliers for hash optimization
    hash_fma_mult: List[Optional[int]] = field(default_factory=list)
    hash_fma_mult_s: Dict[int, int] = field(default_factory=dict)
    v_hash_mult: Dict[int, int] = field(default_factory=dict)

    # ALU hash parallel temps
    alu_temps: Optional[List[int]] = None
    alu_hash_start_iter: int = 0


# =============================================================================
# Base Kernel Builder
# =============================================================================

class BaseKernelBuilder:
    """Base class providing common kernel building infrastructure.

    Handles scratch allocation, instruction emission, and debug tracking.
    """

    def __init__(self, config: KernelConfig):
        self.config = config
        self.instrs: List[Dict] = []
        self.scratch: Dict[str, int] = {}
        self.scratch_debug: Dict[int, Tuple[str, int]] = {}
        self.scratch_ptr = 0
        self.debug_ops: List[Tuple] = []

    def debug_info(self) -> DebugInfo:
        return DebugInfo(scratch_map=self.scratch_debug)

    def alloc(self, name: str = None, length: int = 1) -> int:
        """Allocate scratch memory."""
        addr = self.scratch_ptr
        if name:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        return addr

    def vec(self, name: str) -> int:
        """Allocate a vector (VLEN words) in scratch."""
        return self.alloc(name, VLEN)

    def emit(self, **engines):
        """Emit an instruction with operations for one or more engines."""
        instr = {k: v for k, v in engines.items() if v}
        if instr:
            if self.debug_ops:
                instr["debug"] = list(self.debug_ops)
                self.debug_ops = []
            self.instrs.append(instr)

    def emit_debug(self):
        """Emit a debug-only instruction."""
        if self.debug_ops:
            self.instrs.append({"debug": list(self.debug_ops)})
            self.debug_ops = []

    def emit_compare(self, loc: int, key: tuple):
        """Emit scalar comparison with reference value."""
        if self.config.enable_debug:
            self.debug_ops.append(("compare", loc, key))

    def emit_vcompare(self, loc: int, keys: list):
        """Emit vector comparison with reference values."""
        if self.config.enable_debug:
            self.debug_ops.append(("vcompare", loc, keys))


# =============================================================================
# Tree Access Strategies
# =============================================================================

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
    def allocate(self, builder: BaseKernelBuilder, ctx: ScratchContext) -> None:
        """Allocate strategy-specific scratch memory."""
        pass

    @abstractmethod
    def emit_init(self, builder: BaseKernelBuilder, ctx: ScratchContext) -> None:
        """Emit initialization code (load constants, preload tree values)."""
        pass

    @abstractmethod
    def emit_access(
        self,
        builder: BaseKernelBuilder,
        ctx: ScratchContext,
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

    def allocate(self, builder: BaseKernelBuilder, ctx: ScratchContext) -> None:
        # No additional allocation needed - uses shared temps
        pass

    def emit_init(self, builder: BaseKernelBuilder, ctx: ScratchContext) -> None:
        # No initialization needed for scattered loads
        pass

    def emit_access(
        self,
        builder: BaseKernelBuilder,
        ctx: ScratchContext,
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

    def allocate(self, builder: BaseKernelBuilder, ctx: ScratchContext) -> None:
        """Allocate depth 3 cache if using VALU selection for depth 3."""
        if self.max_depth >= 3:
            ctx.tc_r3 = [builder.alloc(f"tc_r3_{i}") for i in range(8)]
            ctx.v_tc_r3 = [builder.vec(f"v_tc_r3_{i}") for i in range(8)]
            ctx.v_const_r3 = [builder.vec(f"v_const_r3_{i}") for i in range(8)]
            ctx.c_r3 = [builder.alloc(f"c_r3_{i}") for i in range(8)]

    def emit_init(self, builder: BaseKernelBuilder, ctx: ScratchContext) -> None:
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
        builder: BaseKernelBuilder,
        ctx: ScratchContext,
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

    def allocate(self, builder: BaseKernelBuilder, ctx: ScratchContext) -> None:
        self.valu_strategy.allocate(builder, ctx)
        self.scattered_strategy.allocate(builder, ctx)

    def emit_init(self, builder: BaseKernelBuilder, ctx: ScratchContext) -> None:
        self.valu_strategy.emit_init(builder, ctx)
        self.scattered_strategy.emit_init(builder, ctx)

    def emit_access(
        self,
        builder: BaseKernelBuilder,
        ctx: ScratchContext,
        tree_depth: int,
        rnd: int,
    ) -> None:
        if tree_depth <= self.valu_max_depth:
            self.valu_strategy.emit_access(builder, ctx, tree_depth, rnd)
        else:
            self.scattered_strategy.emit_access(builder, ctx, tree_depth, rnd)


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

    FUTURE OPTIMIZATION: Jump-based chunk skipping
    =============================================
    With sorted items and jump-based control flow, we could exploit the high
    collision rate (99%+ in some rounds) to skip chunks that have no items:

    1. Sort batch items by tree index within each round
    2. Process items in chunk order, checking only the first/last item per chunk
    3. Use jump instruction to skip empty chunks entirely
    4. Three processing modes:
       - Same chunk: continue processing
       - Next chunk in same vload: switch selection range, no new load
       - New vload needed: emit vload, reset selection range

    With 99% collision (1-2 unique indices), typically only 1-2 chunks are
    non-empty, reducing effective work from 170 cycles to ~10 cycles.

    This optimization requires:
    - Sorted items infrastructure (see partition_kernel.py)
    - Jump-based control flow (ISA supports jumps)
    - Per-chunk item counts or boundary detection

    For now, this strategy demonstrates the structure but doesn't implement
    jump-based skipping. Use HybridTreeAccessStrategy for best performance.
    """

    def __init__(
        self,
        valu_max_depth: int = 3,
        chunked_min_depth: int = 4,
        chunked_max_depth: int = 10,
        chunk_size: int = 4,
    ):
        """Initialize chunked vload strategy.

        Args:
            valu_max_depth: Use VALU selection up to this depth (default: 3)
            chunked_min_depth: Start chunked vload at this depth (default: 4)
            chunked_max_depth: End chunked vload at this depth (default: 10)
            chunk_size: Chunk size for chunked gather (default: 4)
        """
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

    def allocate(self, builder: BaseKernelBuilder, ctx: ScratchContext) -> None:
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

    def emit_init(self, builder: BaseKernelBuilder, ctx: ScratchContext) -> None:
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
            # This is faster than individual scalar loads: 1 vload vs 4 scalar loads
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
        builder: BaseKernelBuilder,
        ctx: ScratchContext,
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
        builder: BaseKernelBuilder,
        ctx: ScratchContext,
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


# =============================================================================
# Hash Computation Strategies
# =============================================================================

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
    def allocate(self, builder: BaseKernelBuilder, ctx: ScratchContext) -> None:
        pass

    @abstractmethod
    def emit_init(self, builder: BaseKernelBuilder, ctx: ScratchContext) -> None:
        pass

    @abstractmethod
    def emit_xor(
        self,
        builder: BaseKernelBuilder,
        ctx: ScratchContext,
        rnd: int,
    ) -> None:
        """Emit XOR of all_val with v_nv (tree values)."""
        pass

    @abstractmethod
    def emit_hash(
        self,
        builder: BaseKernelBuilder,
        ctx: ScratchContext,
        rnd: int,
    ) -> None:
        """Emit 6-stage hash computation."""
        pass

    @abstractmethod
    def emit_branch(
        self,
        builder: BaseKernelBuilder,
        ctx: ScratchContext,
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

    def allocate(self, builder: BaseKernelBuilder, ctx: ScratchContext) -> None:
        # Calculate FMA eligibility for each stage
        ctx.hash_fma_mult = [None] * 6
        for i, (op1, const, op2, op3, shift) in enumerate(HASH_STAGES):
            if op1 == '+' and op2 == '+' and op3 == '<<':
                ctx.hash_fma_mult[i] = 1 + (1 << shift)

    def emit_init(self, builder: BaseKernelBuilder, ctx: ScratchContext) -> None:
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
        builder: BaseKernelBuilder,
        ctx: ScratchContext,
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
        builder: BaseKernelBuilder,
        ctx: ScratchContext,
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
        builder: BaseKernelBuilder,
        ctx: ScratchContext,
        rnd: int,
    ) -> None:
        """Emit branch computation with bounds checking."""
        n_groups = builder.config.batch_size // VLEN
        n_iters = n_groups // self.UNROLL

        # Check if bounds check needed
        max_idx_this_round = (1 << (rnd + 2)) - 1
        needs_bounds_check = max_idx_this_round >= builder.config.n_nodes

        for it in range(n_iters):
            base_g = it * self.UNROLL
            for i in range(self.UNROLL):
                g = base_g + i
                t = ctx.v_t[it][i]

                # Stage 5 opt: t = 2 - (val & 1) = flipped branch
                builder.emit(valu=[("&", t, ctx.all_val[g], ctx.v_1)])
                builder.emit(valu=[("-", t, ctx.v_2, t)])
                # idx = idx * 2 + t (fused multiply_add)
                builder.emit(valu=[("multiply_add", ctx.all_idx[g], ctx.all_idx[g], ctx.v_2, t)])

                if needs_bounds_check:
                    builder.emit(valu=[("<", t, ctx.all_idx[g], ctx.v_nn)])
                    builder.emit(valu=[("*", ctx.all_idx[g], ctx.all_idx[g], t)])

        # Debug
        for g in range(n_groups):
            keys = [(rnd, g * VLEN + vi, "wrapped_idx") for vi in range(VLEN)]
            builder.emit_vcompare(ctx.all_idx[g], keys)
        builder.emit_debug()


# =============================================================================
# Factory Functions
# =============================================================================

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
    else:
        raise ValueError(f"Unknown tree access mode: {mode}")


def create_hash_strategy(config: KernelConfig) -> HashStrategy:
    """Create hash strategy based on config."""
    # For now, only VALU is implemented
    return VALUHashStrategy()


# =============================================================================
# Modular Kernel Builder
# =============================================================================

class ModularKernelBuilder(BaseKernelBuilder):
    """Build kernels using composable strategies.

    This is the main entry point for building kernels with the modular architecture.

    Example usage:
        config = KernelConfig(
            tree_access_mode=TreeAccessMode.HYBRID,
            hash_mode=HashMode.VALU_ONLY,
            scheduling_mode=SchedulingMode.GREEDY_MULTIPASS,
        )
        builder = ModularKernelBuilder(config)
        builder.build()

        # Access results
        instrs = builder.instrs
        cycles = len(instrs) - 2  # Subtract pause instructions
    """

    UNROLL = 8

    def __init__(self, config: KernelConfig):
        super().__init__(config)
        self.tree_strategy = create_tree_strategy(config)
        self.hash_strategy = create_hash_strategy(config)
        self.ctx = ScratchContext()

    def build(self):
        """Build the complete kernel."""
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
        """Apply scheduling based on config."""
        from solver.reschedule import (
            extract_ops, compute_dependencies,
            schedule_greedy, schedule_greedy_multipass, emit_scheduled
        )

        ops = extract_ops(self.instrs)
        op_by_id = compute_dependencies(ops)
        print(f"Scheduling {len(ops)} ops...")

        mode = self.config.scheduling_mode
        if mode == SchedulingMode.GREEDY_SINGLE:
            import time
            start = time.time()
            num_cycles, scheduled_ops = schedule_greedy(
                ops, op_by_id,
                enable_valu_shift=self.config.enable_valu_shift,
                enable_broadcast_lookahead=self.config.enable_broadcast_lookahead
            )
            scheduled = emit_scheduled(scheduled_ops, num_cycles)
            print(f"Greedy scheduling completed in {time.time() - start:.2f}s")

        elif mode == SchedulingMode.GREEDY_MULTIPASS:
            import time
            start = time.time()
            num_cycles, scheduled_ops = schedule_greedy_multipass(
                ops, op_by_id,
                enable_valu_shift=self.config.enable_valu_shift,
                enable_broadcast_lookahead=self.config.enable_broadcast_lookahead
            )
            scheduled = emit_scheduled(scheduled_ops, num_cycles)
            print(f"Multipass greedy scheduling completed in {time.time() - start:.2f}s")

        else:  # CPSAT_OPTIMAL
            from solver.round_chunked_scheduler import schedule_round_chunked
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

        self.instrs = [{"flow": [("pause",)]}] + scheduled + [{"flow": [("pause",)]}]
        print(f"Final scheduling: {len(self.instrs)} instructions ({num_cycles} cycles)")
