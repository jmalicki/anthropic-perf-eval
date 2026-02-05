"""
Modular Kernel Builder Framework

This package provides a composable architecture for building kernels with different
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

Default production builder: KernelBuilder uses the Rust SSA path (build_tick_tock_kernel_ssa).

Usage:
    from solver.kernel_builder import (
        KernelConfig, TreeAccessMode, SchedulingMode,
        ModularKernelBuilder, KernelBuilder,
    )

    # Default: SSA Rust builder (recommended)
    builder = KernelBuilder()
    builder.build_kernel(forest_height=10, n_nodes=2047, batch_size=256, rounds=16)

    # Modular approach
    config = KernelConfig(
        tree_access_mode=TreeAccessMode.HYBRID,
        scheduling_mode=SchedulingMode.GREEDY_MULTIPASS,
    )
    builder = ModularKernelBuilder(config)
    builder.build()

    # Access results
    instrs = builder.instrs
    cycles = len(instrs) - 2  # Subtract pause instructions
"""

from .config import (
    TreeAccessMode,
    HashMode,
    SchedulingMode,
    KernelConfig,
)
from .context import ScratchContext
from .base import BaseKernelBuilder
from .modular import (
    ModularKernelBuilder,
    create_tree_strategy,
    create_hash_strategy,
)
from .ssa_wrapper import SSAKernelBuilder
from .strategies import (
    TreeAccessStrategy,
    ScatteredLoadStrategy,
    VALUSelectionStrategy,
    HybridTreeAccessStrategy,
    ChunkedVloadStrategy,
    ChunkedDemandDrivenStrategy,
    AdaptiveSelectionStrategy,
    TickTockPartitionStrategy,
    TickTockPartitionStrategyV2,
    HashStrategy,
    VALUHashStrategy,
)

# Default production builder: SSA Rust path (CP-SAT scheduling)
KernelBuilder = SSAKernelBuilder

__all__ = [
    # Configuration
    'TreeAccessMode',
    'HashMode',
    'SchedulingMode',
    'KernelConfig',
    # Context
    'ScratchContext',
    # Builders
    'BaseKernelBuilder',
    'ModularKernelBuilder',
    'SSAKernelBuilder',
    'KernelBuilder',  # Alias for SSAKernelBuilder
    # Factory functions
    'create_tree_strategy',
    'create_hash_strategy',
    # Tree access strategies
    'TreeAccessStrategy',
    'ScatteredLoadStrategy',
    'VALUSelectionStrategy',
    'HybridTreeAccessStrategy',
    'ChunkedVloadStrategy',
    'ChunkedDemandDrivenStrategy',
    'AdaptiveSelectionStrategy',
    'TickTockPartitionStrategy',
    'TickTockPartitionStrategyV2',
    # Hash strategies
    'HashStrategy',
    'VALUHashStrategy',
]
