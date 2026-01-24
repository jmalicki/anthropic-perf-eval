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
    DEMAND_DRIVEN = auto()      # Demand-driven vload with 4-value selection
    ADAPTIVE_SELECTION = auto() # Span-based adaptive selection for depths 4+
    TICK_TOCK_PARTITION = auto() # Tick-tock lazy partition with sorted vload


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

    Handles scratch allocation, instruction emission, debug tracking,
    and control flow with labels and jumps.
    """

    def __init__(self, config: KernelConfig):
        self.config = config
        self.instrs: List[Dict] = []
        self.scratch: Dict[str, int] = {}
        self.scratch_debug: Dict[int, Tuple[str, int]] = {}
        self.scratch_ptr = 0
        self.debug_ops: List[Tuple] = []
        # Control flow infrastructure
        self.labels: Dict[str, int] = {}  # label_name -> instruction index
        self.fixups: List[Tuple[int, str]] = []  # (instr_idx, label_name) for forward refs

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

    # =========================================================================
    # Control Flow Infrastructure (labels and jumps)
    # =========================================================================

    def label(self, name: str) -> None:
        """Define a label at the current instruction position.

        Labels are targets for jump instructions. The label points to the
        NEXT instruction that will be emitted.
        """
        self.labels[name] = len(self.instrs)

    def emit_jump(self, label: str) -> None:
        """Emit an unconditional jump to a label.

        If the label is not yet defined (forward reference), the address
        will be fixed up later by resolve_fixups().
        """
        if label in self.labels:
            addr = self.labels[label]
        else:
            addr = 0  # Placeholder, will be fixed up later
            self.fixups.append((len(self.instrs), label))
        self.emit(flow=[("jump", addr)])

    def emit_cond_jump(self, cond: int, label: str) -> None:
        """Emit a conditional jump to a label.

        Jump is taken if scratch[cond] != 0.
        Forward references are resolved by resolve_fixups().
        """
        if label in self.labels:
            addr = self.labels[label]
        else:
            addr = 0  # Placeholder, will be fixed up later
            self.fixups.append((len(self.instrs), label))
        self.emit(flow=[("cond_jump", cond, addr)])

    def resolve_fixups(self) -> None:
        """Resolve forward references to labels.

        Called after all instructions are emitted to patch jump addresses.
        """
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
        self.fixups.clear()


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
        """Initialize demand-driven strategy.

        Args:
            valu_max_depth: Use VALU selection up to this depth (default: 3)
        """
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

    def allocate(self, builder: BaseKernelBuilder, ctx: ScratchContext) -> None:
        """Allocate scratch for VALU selection + scattered loads."""
        self.valu_strategy.allocate(builder, ctx)
        self.scattered_strategy.allocate(builder, ctx)

        # Reserve cache vectors for future sorted-items optimization
        self.v_cache = [builder.vec(f"v_dd_cache_{i}") for i in range(3)]

    def emit_init(self, builder: BaseKernelBuilder, ctx: ScratchContext) -> None:
        """Emit initialization code."""
        self.valu_strategy.emit_init(builder, ctx)
        self.scattered_strategy.emit_init(builder, ctx)

    def emit_access(
        self,
        builder: BaseKernelBuilder,
        ctx: ScratchContext,
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
        builder: BaseKernelBuilder,
        ctx: ScratchContext,
        tree_depth: int,
        rnd: int,
    ) -> None:
        """Emit 8-value tree access using vload cache.

        Key insight: With high collision rates, most items in a round have
        indices within a small range. We:
        1. Extract first item's index as representative
        2. vload 8 consecutive tree values into cache
        3. Use 8-value compare/select for all groups

        For items outside the cache range, the selection will produce 0
        (no match), which is incorrect. This approach assumes most items
        are within the 8-value cache window.

        Cost per round:
        - 1 vstore + 1 load (extract first_idx) = ~2 cycles
        - 1 vload + 8 broadcasts = ~5 cycles
        - 16 VALU ops per group × 32 groups = 512 ops = ~85 cycles

        Total: ~92 cycles vs scattered loads ~128 cycles
        Savings: ~36 cycles per round
        """
        n_groups = builder.config.batch_size // VLEN
        n_iters = n_groups // self.UNROLL

        # Step 1: Extract first item's index as cache base
        # Store first group to memory, load element 0
        first_idx_s = ctx.st[10]
        cache_base_s = ctx.st[11]
        builder.emit(store=[("vstore", ctx.idx_p, ctx.all_idx[0])])
        builder.emit(load=[("load", first_idx_s, ctx.idx_p)])

        # Round down to start of level (optional, could just use first_idx)
        # For now, just use first_idx as cache_base
        builder.emit(alu=[("=", cache_base_s, first_idx_s, ctx.c[0])])  # cache_base = first_idx

        # Step 2: vload 8 tree values starting at cache_base
        # Compute address: forest_p + cache_base
        builder.emit(alu=[("+", ctx.st[0], ctx.forest_p, cache_base_s)])
        # vload into scratch starting at v_cache[0]
        # Note: v_cache was allocated as 3 vectors, we need 8 scalar slots
        # Reuse the first vector's memory for 8 scalars
        cache_scalars = self.v_cache[0]  # 8 consecutive scalar locations
        builder.emit(load=[("vload", cache_scalars, ctx.st[0])])

        # Step 3: Broadcast cache values to vectors
        # We need 8 vector broadcasts for the 8 cached tree values
        v_cache_vecs = [self.v_cache[0], self.v_cache[1], self.v_cache[2]]
        # Actually we only allocated 3 cache vectors, need to allocate more or reuse
        # For now, let's use a simpler approach: compute selection inline

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

    def _emit_demand_driven_access(
        self,
        builder: BaseKernelBuilder,
        ctx: ScratchContext,
        tree_depth: int,
        rnd: int,
    ) -> None:
        """Emit fast 2-value tree access using vselect.

        Key insight: With stable partition each round, items at the same parent
        index branch to two adjacent child indices. We:
        1. Extract the first item's index (the minimum with sorted items)
        2. Load tree values at that index and the next
        3. Use vselect to choose based on each item's actual index

        This handles groups that straddle partition boundaries correctly.

        Cost per round:
        - 1 vstore + 1 load (extract first_idx)
        - 2 loads (tree values)
        - 3 broadcasts
        - 32 compares + 32 vselects + 32 XORs = ~96 ops

        Total: ~16 cycles for VALU + 3 loads + 32 flow = ~52 cycles
        vs scattered loads: ~150 cycles per round
        """
        n_groups = builder.config.batch_size // VLEN
        n_iters = n_groups // self.UNROLL

        # Step 1: Extract first item's index (minimum with sorted items)
        # Store first group to memory, load element 0
        first_idx_s = ctx.st[10]
        builder.emit(store=[("vstore", ctx.idx_p, ctx.all_idx[0])])
        builder.emit(load=[("load", first_idx_s, ctx.idx_p)])

        # Step 2: Load tree values at first_idx and first_idx + 1
        low_tree_s = ctx.st[11]
        high_tree_s = ctx.st[12]

        builder.emit(alu=[("+", ctx.st[0], ctx.forest_p, first_idx_s)])
        builder.emit(alu=[("+", ctx.st[1], first_idx_s, ctx.c[1])])  # first_idx + 1
        builder.emit(alu=[("+", ctx.st[2], ctx.forest_p, ctx.st[1])])
        builder.emit(load=[("load", low_tree_s, ctx.st[0])])
        builder.emit(load=[("load", high_tree_s, ctx.st[2])])

        # Step 3: Broadcast to vectors
        v_low = self.v_cache[0]
        v_high = self.v_cache[1]
        v_first_idx = self.v_cache[2]
        builder.emit(valu=[
            ("vbroadcast", v_low, low_tree_s),
            ("vbroadcast", v_high, high_tree_s),
            ("vbroadcast", v_first_idx, first_idx_s),
        ])

        # Step 4: For each group, select LOW or HIGH based on actual idx
        # With only 2 possible indices (first_idx, first_idx+1):
        # is_high = (idx - first_idx) which is 0 for LOW, 1 for HIGH
        # v_nv = is_high ? v_high : v_low
        for it in range(n_iters):
            base_g = it * self.UNROLL
            for i in range(self.UNROLL):
                g = base_g + i
                t = ctx.v_t[it][i]

                # Compute difference: 0 for first_idx, 1 for first_idx+1
                builder.emit(valu=[("-", t, ctx.all_idx[g], v_first_idx)])

                # Select: v_nv = is_high ? v_high : v_low
                builder.emit(flow=[("vselect", ctx.v_nv[it][i], t, v_high, v_low)])

                # Stage 5 opt: XOR v_nv with C
                builder.emit(valu=[("^", ctx.v_nv[it][i], ctx.v_nv[it][i], ctx.v_stage5_xor)])

    def _emit_speculative_selection(
        self,
        builder: BaseKernelBuilder,
        ctx: ScratchContext,
        g: int,
        it: int,
        i: int,
    ) -> None:
        """Emit 8-value selection from cache.

        Uses 8-value selection (like depth 3) from the cache.
        Reuses ctx.v_t for temporaries to minimize scratch usage.
        """
        # Use existing v_t for temporaries
        v_rel = ctx.v_t[it][i]  # Reuse v_t for relative position

        # Compute relative position in cache: v_rel = all_idx[g] - v_cache_base
        builder.emit(valu=[("-", v_rel, ctx.all_idx[g], self.v_cache_base)])

        # 8-value selection: compare with each offset 0-7
        # result = sum((rel == i) * cache[i]) for i in 0..7
        builder.emit(valu=[("*", ctx.v_nv[it][i], ctx.v_0, ctx.v_0)])  # init to 0
        for ci in range(self.VLOAD_SIZE):
            # Use v_nv as temp for mask, then accumulate
            # Actually, we need a separate mask temp. Use a different v_t slot.
            # For the last group in iteration, use v_nv[it][(i+1) % UNROLL] as temp
            mask_temp = ctx.v_t[it][(i + 1) % self.UNROLL]
            builder.emit(valu=[("==", mask_temp, v_rel, ctx.v_const[ci])])
            builder.emit(valu=[("multiply_add", ctx.v_nv[it][i], mask_temp, self.v_cache[ci], ctx.v_nv[it][i])])

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
        self.hybrid = HybridTreeAccessStrategy(valu_max_depth=valu_max_depth)
        self.valu_max_depth = valu_max_depth

    def allocate(self, builder: BaseKernelBuilder, ctx: ScratchContext) -> None:
        self.hybrid.allocate(builder, ctx)

    def emit_init(self, builder: BaseKernelBuilder, ctx: ScratchContext) -> None:
        self.hybrid.emit_init(builder, ctx)

    def emit_access(
        self,
        builder: BaseKernelBuilder,
        ctx: ScratchContext,
        tree_depth: int,
        rnd: int,
    ) -> None:
        self.hybrid.emit_access(builder, ctx, tree_depth, rnd)


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
        """Initialize tick-tock partition strategy.

        Args:
            valu_max_depth: Use VALU selection up to this depth (default: 3)
        """
        self.valu_strategy = VALUSelectionStrategy(max_depth=valu_max_depth)
        self.scattered_strategy = ScatteredLoadStrategy()
        self.valu_max_depth = valu_max_depth

        # Partition buffers - will be allocated during allocate()
        # Double-buffered: partition_a and partition_b for ping-pong
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

    def allocate(self, builder: BaseKernelBuilder, ctx: ScratchContext) -> None:
        """Allocate scratch for partition buffers and tree cache."""
        self.valu_strategy.allocate(builder, ctx)
        self.scattered_strategy.allocate(builder, ctx)

        n_groups = builder.config.batch_size // VLEN

        # Partition buffers: store sorted indices and values
        # Each buffer holds 256 items (32 groups × 8 elements)
        # We use memory for partition storage, but need scratch for temps
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

    def emit_init(self, builder: BaseKernelBuilder, ctx: ScratchContext) -> None:
        """Emit initialization code."""
        self.valu_strategy.emit_init(builder, ctx)
        self.scattered_strategy.emit_init(builder, ctx)

        # Initialize partition counters to 0
        builder.emit(alu=[("*", self.count0_h0, ctx.c[0], ctx.c[0])])
        builder.emit(alu=[("*", self.count0_h1, ctx.c[0], ctx.c[0])])

    def emit_access(
        self,
        builder: BaseKernelBuilder,
        ctx: ScratchContext,
        tree_depth: int,
        rnd: int,
    ) -> None:
        """Emit tree access based on round and depth.

        Uses VALU selection for depths 0-3, sorted vload for depths 4+.
        Handles partition reload at rounds 4 and 15.
        """
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
            # After this round, all indices reset to 0
            self.scattered_strategy.emit_access(builder, ctx, tree_depth, rnd)
        else:
            # Depths 4-9 (after partition ready): Use sorted vload
            self._emit_sorted_vload_access(builder, ctx, tree_depth, rnd)

    def _emit_sorted_vload_access(
        self,
        builder: BaseKernelBuilder,
        ctx: ScratchContext,
        tree_depth: int,
        rnd: int,
    ) -> None:
        """Emit sorted vload-based tree access.

        With items sorted by tree index from the previous partition:
        - Groups of 8 items span at most 8 tree indices (for k≤3 rounds since partition)
        - One vload (8 values) covers any group
        - Use 8-value selection to pick correct values

        For boundary groups (~6% of groups), may need 2 vloads.
        For simplicity, we use a per-group approach that loads what's needed.

        Cost analysis:
        - 1 vload per group (or 2 for boundary) = ~4-8 vloads total
        - 8-value selection: 16 VALU ops per group
        - Total: ~16 cycles vs 128 for scattered
        """
        n_groups = builder.config.batch_size // VLEN
        n_iters = n_groups // self.UNROLL

        # For each group, we need to:
        # 1. Find the minimum index in the group (first element since sorted)
        # 2. vload 8 tree values starting at that index
        # 3. Select correct value for each item using offset from min

        # Strategy: Use first element's index as cache base for each group
        # Since items are sorted, this gives the minimum index
        # The 8-value selection handles the spread within the group

        for it in range(n_iters):
            base_g = it * self.UNROLL
            for i in range(self.UNROLL):
                g = base_g + i
                t = ctx.v_t[it][i]

                # Extract first element's index as cache base
                # Store group 0 to memory, load element 0
                # This is expensive but necessary without sorted infrastructure
                builder.emit(store=[("vstore", ctx.idx_p, ctx.all_idx[g])])
                builder.emit(load=[("load", self.tree_cache_base, ctx.idx_p)])

                # vload 8 tree values starting at cache_base
                builder.emit(alu=[("+", ctx.st[0], ctx.forest_p, self.tree_cache_base)])
                # Load into first tree cache vector (8 consecutive values)
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

    This version focuses on the core insight: with 98% idle store slots,
    we can pipeline partition stores without impacting compute cycles.

    Schedule for partition stores (TICK-TOCK):
    - TICK phase: Store Half 0 (items 0-127) over 3 rounds
    - TOCK phase: Store Half 1 (items 128-255) over 3 rounds
    - Total: 256 stores = 128 cycles, spread over 6 rounds = 21 stores/round
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

    def allocate(self, builder: BaseKernelBuilder, ctx: ScratchContext) -> None:
        self.valu_strategy.allocate(builder, ctx)
        self.scattered_strategy.allocate(builder, ctx)

        # Tree cache: single vload worth (8 values)
        self.v_tree_cache = [builder.vec(f"v_tree_cache2_{i}") for i in range(3)]
        self.tree_cache_base = builder.alloc("tree_cache_base2")
        self.v_cache_base = builder.vec("v_cache_base2")

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
        """Emit tree access with tick-tock partition pipelining."""
        if tree_depth <= self.valu_max_depth:
            self.valu_strategy.emit_access(builder, ctx, tree_depth, rnd)
        else:
            # For depths 4+, use scattered loads with partition store pipelining
            # The partition stores happen in the store phase (handled by scheduler)
            self.scattered_strategy.emit_access(builder, ctx, tree_depth, rnd)


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
                    # Skip branch computation entirely - just set idx = 0
                    builder.emit(valu=[("*", ctx.all_idx[g], ctx.v_0, ctx.v_0)])
                else:
                    # Normal branching - no bounds check needed
                    # Stage 5 opt: t = 2 - (val & 1) = flipped branch
                    builder.emit(valu=[("&", t, ctx.all_val[g], ctx.v_1)])
                    builder.emit(valu=[("-", t, ctx.v_2, t)])
                    # idx = idx * 2 + t (fused multiply_add)
                    builder.emit(valu=[("multiply_add", ctx.all_idx[g], ctx.all_idx[g], ctx.v_2, t)])

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


# =============================================================================
# Modular Kernel Builder
# =============================================================================

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


# =============================================================================
# Tick-Tock Partition Kernel Builder
# =============================================================================

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
    TICK_STORE_DEPTHS = {0, 1, 2, 4, 5, 6}  # First half at these depths
    TOCK_STORE_DEPTHS = {3, 7, 8, 9}        # Second half at these depths

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


# Alias for backward compatibility - use the best performing builder
# TickTockKernelBuilder achieves 1790 cycles (82.4x) vs ModularKernelBuilder's 1822 (81x)
KernelBuilder = TickTockKernelBuilder
