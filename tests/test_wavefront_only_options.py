"""Wavefront-only greedy orderings are an error without the wavefront priority (they used to be
ignored silently under the FIFO priority)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernel_build_args import build_kernel_with_args  # noqa: E402


@pytest.mark.parametrize("option", [{"greedy_alu_priority": "native-first"},
                                    {"greedy_flow_priority": "chain-first"},
                                    {"greedy_load_priority": "final-last"}])
def test_wavefront_only_orderings_need_the_wavefront(option):
    with pytest.raises(ValueError, match="orders the wavefront priority only"):
        build_kernel_with_args(rounds=2, batch_size=8, use_greedy=True, verbose=False, **option)
    # With the wavefront priority the same option builds.
    instrs, stats = build_kernel_with_args(rounds=2, batch_size=8, use_greedy=True, verbose=False,
                                           greedy_barriers="none", greedy_priority="wavefront", **option)
    assert stats["cycles"] > 0
