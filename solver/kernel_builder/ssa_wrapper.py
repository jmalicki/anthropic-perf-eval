"""
SSA-based kernel builder wrapper.

Calls scheduler_rs.build_tick_tock_kernel_ssa and exposes the same interface
as the legacy KernelBuilder (build_kernel, .instrs, .debug_info()) for use
by perf_takehome, submission_tests, and run_experiment.
"""

from problem import DebugInfo


class SSAKernelBuilder:
    """Kernel builder that uses the Rust SSA path (build_tick_tock_kernel_ssa)."""

    def __init__(self):
        self.instrs: list = []
        self._stats: dict | None = None

    def build_kernel(
        self,
        forest_height: int,
        n_nodes: int,
        batch_size: int,
        rounds: int,
    ) -> None:
        import scheduler_rs

        instrs, stats = scheduler_rs.build_tick_tock_kernel_ssa(
            forest_height=forest_height,
            n_nodes=n_nodes,
            batch_size=batch_size,
            rounds=rounds,
            enable_valu_shift=False,
            verbose=False,
            use_cpsat=True,
            cpsat_time_limit=600.0,
        )
        self.instrs = list(instrs)
        self._stats = stats

    def debug_info(self) -> DebugInfo:
        return DebugInfo(scratch_map={})
