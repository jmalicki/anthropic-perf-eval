"""
Kernel Builder - SSA Rust path wrapper.

Default production builder: KernelBuilder uses the Rust SSA path
(scheduler_rs.build_tick_tock_kernel_ssa).

Usage:
    from solver.kernel_builder import KernelBuilder

    builder = KernelBuilder()
    builder.build_kernel(forest_height=10, n_nodes=2047, batch_size=256, rounds=16)
    instrs = builder.instrs
"""

from .ssa_wrapper import SSAKernelBuilder

KernelBuilder = SSAKernelBuilder

__all__ = [
    'SSAKernelBuilder',
    'KernelBuilder',
]
