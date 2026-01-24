"""
Configuration for kernel generation.

This module contains all configuration enums and the KernelConfig dataclass
that control how kernels are built.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


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

    # VALU->ALU shifting
    enable_valu_shift: bool = True
    valu_shift_time: Optional[float] = None
    enable_broadcast_lookahead: bool = False

    # Debug options
    enable_debug: bool = False
