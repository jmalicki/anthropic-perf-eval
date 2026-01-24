"""
Scratch memory allocation context.

This module contains the ScratchContext dataclass that tracks all allocated
scratch memory locations across kernel building phases.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


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
