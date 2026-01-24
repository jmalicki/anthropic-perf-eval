"""
Kernel Analysis Tool - Analyzes operation counts, slot utilization, and theoretical limits.

This tool helps understand where cycles are spent and what the theoretical minimum
cycle count could be for various kernel configurations.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from problem import SLOT_LIMITS, VLEN


@dataclass
class KernelAnalysis:
    """Analysis results for a kernel configuration."""
    config_name: str
    total_cycles: int

    # Op counts by engine (before VALU shift)
    ops_by_engine: dict[str, int]

    # Op counts after VALU shift (if enabled)
    ops_after_shift: Optional[dict[str, int]]
    valu_shifted_count: int

    # Theoretical minimums (ops / slots per cycle)
    theoretical_min_cycles: dict[str, int]

    # Utilization percentages
    utilization: dict[str, float]

    # Scheduling overhead
    scheduling_overhead_cycles: int
    scheduling_overhead_pct: float

    def print_report(self):
        """Print a formatted analysis report."""
        print(f"\n{'='*60}")
        print(f"Configuration: {self.config_name}")
        print(f"{'='*60}")

        print(f"\nTotal Cycles: {self.total_cycles}")

        print(f"\n--- Operation Counts (Before VALU Shift) ---")
        for engine, count in sorted(self.ops_by_engine.items()):
            slots = SLOT_LIMITS.get(engine, 1)
            min_cycles = (count + slots - 1) // slots
            print(f"  {engine:6s}: {count:6d} ops → min {min_cycles:4d} cycles ({slots} slots/cycle)")

        if self.ops_after_shift:
            print(f"\n--- Operation Counts (After VALU Shift) ---")
            print(f"  VALU ops shifted to ALU: {self.valu_shifted_count}")
            for engine, count in sorted(self.ops_after_shift.items()):
                slots = SLOT_LIMITS.get(engine, 1)
                min_cycles = (count + slots - 1) // slots
                print(f"  {engine:6s}: {count:6d} ops → min {min_cycles:4d} cycles ({slots} slots/cycle)")

        print(f"\n--- Theoretical Minimums ---")
        bottleneck_engine = max(self.theoretical_min_cycles.items(), key=lambda x: x[1])
        for engine, min_cycles in sorted(self.theoretical_min_cycles.items(), key=lambda x: -x[1]):
            marker = " ← BOTTLENECK" if engine == bottleneck_engine[0] else ""
            print(f"  {engine:6s}: {min_cycles:4d} cycles{marker}")

        print(f"\n--- Slot Utilization ---")
        for engine, util in sorted(self.utilization.items(), key=lambda x: -x[1]):
            print(f"  {engine:6s}: {util:5.1f}%")

        print(f"\n--- Scheduling Overhead ---")
        print(f"  Theoretical min: {bottleneck_engine[1]} cycles")
        print(f"  Actual:          {self.total_cycles} cycles")
        print(f"  Overhead:        {self.scheduling_overhead_cycles} cycles ({self.scheduling_overhead_pct:.1f}%)")
        print()


def count_ops_by_engine(instrs: list[dict]) -> dict[str, int]:
    """Count operations by engine type from instruction list."""
    counts = defaultdict(int)
    for instr in instrs:
        for engine, slots in instr.items():
            if engine != 'debug':
                counts[engine] += len(slots)
    return dict(counts)


def count_ops_from_ops_list(ops: list) -> dict[str, int]:
    """Count operations by engine type from Op list."""
    counts = defaultdict(int)
    for op in ops:
        counts[op.engine] += 1
    return dict(counts)


def compute_theoretical_minimums(ops_by_engine: dict[str, int]) -> dict[str, int]:
    """Compute theoretical minimum cycles for each engine."""
    minimums = {}
    for engine, count in ops_by_engine.items():
        slots = SLOT_LIMITS.get(engine, 1)
        minimums[engine] = (count + slots - 1) // slots
    return minimums


def compute_utilization(ops_by_engine: dict[str, int], total_cycles: int) -> dict[str, float]:
    """Compute slot utilization percentage for each engine."""
    utilization = {}
    for engine, count in ops_by_engine.items():
        slots = SLOT_LIMITS.get(engine, 1)
        max_possible_ops = total_cycles * slots
        utilization[engine] = (count / max_possible_ops * 100) if max_possible_ops > 0 else 0.0
    return utilization


def analyze_kernel_build(
    config_name: str,
    forest_height: int = 10,
    n_nodes: int = 2047,
    batch_size: int = 256,
    rounds: int = 16,
    use_greedy: bool = True,
    enable_valu_shift: bool = True,
    **kwargs
) -> KernelAnalysis:
    """Build a kernel and analyze its characteristics.

    Args:
        config_name: Human-readable name for this configuration
        forest_height: Height of the decision tree
        n_nodes: Number of nodes in the tree
        batch_size: Batch size for parallel processing
        rounds: Number of tree traversal rounds
        use_greedy: Use greedy scheduler (True) or CP-SAT (False)
        enable_valu_shift: Enable VALU→ALU shifting optimization
        **kwargs: Additional arguments passed to build_kernel

    Returns:
        KernelAnalysis with detailed metrics
    """
    from solver.interleaved_unrolled_kernel import InterleavedUnrolledKernelBuilder
    from solver.reschedule import extract_ops, compute_dependencies

    # Build the kernel
    builder = InterleavedUnrolledKernelBuilder()
    builder.build_kernel(
        forest_height=forest_height,
        n_nodes=n_nodes,
        batch_size=batch_size,
        rounds=rounds,
        use_greedy=use_greedy,
        enable_valu_shift=enable_valu_shift,
        **kwargs
    )

    # Total cycles (excluding pause instructions)
    total_cycles = len(builder.instrs) - 2

    # Count ops from final scheduled instructions
    ops_by_engine = count_ops_by_engine(builder.instrs)

    # For ops before shift, we need to rebuild without VALU shift to count original ops
    # This is a bit wasteful but gives accurate counts
    builder_no_shift = InterleavedUnrolledKernelBuilder()
    # Don't print during analysis
    import sys
    from io import StringIO
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        builder_no_shift.build_kernel(
            forest_height=forest_height,
            n_nodes=n_nodes,
            batch_size=batch_size,
            rounds=rounds,
            use_greedy=use_greedy,
            enable_valu_shift=False,
            **kwargs
        )
    finally:
        sys.stdout = old_stdout

    ops_before_shift = count_ops_by_engine(builder_no_shift.instrs)

    # Compute VALU shift count
    valu_shifted = 0
    if enable_valu_shift:
        valu_before = ops_before_shift.get('valu', 0)
        valu_after = ops_by_engine.get('valu', 0)
        valu_shifted = valu_before - valu_after

    # Use the actual ops from the final kernel for analysis
    ops_after_shift = ops_by_engine if enable_valu_shift else None

    # Compute theoretical minimums based on final ops
    theoretical_mins = compute_theoretical_minimums(ops_by_engine)

    # Compute utilization
    utilization = compute_utilization(ops_by_engine, total_cycles)

    # Scheduling overhead
    bottleneck_min = max(theoretical_mins.values()) if theoretical_mins else 0
    overhead_cycles = total_cycles - bottleneck_min
    overhead_pct = (overhead_cycles / total_cycles * 100) if total_cycles > 0 else 0

    return KernelAnalysis(
        config_name=config_name,
        total_cycles=total_cycles,
        ops_by_engine=ops_before_shift,
        ops_after_shift=ops_after_shift,
        valu_shifted_count=valu_shifted,
        theoretical_min_cycles=theoretical_mins,
        utilization=utilization,
        scheduling_overhead_cycles=overhead_cycles,
        scheduling_overhead_pct=overhead_pct,
    )


def analyze_slot_utilization_per_cycle(instrs: list[dict]) -> dict[str, list[int]]:
    """Compute slot utilization for each cycle.

    Returns a dict mapping engine -> list of slot counts per cycle.
    """
    # Skip pause instructions
    core_instrs = [i for i in instrs if 'flow' not in i or i.get('flow') != [('pause',)]]

    utilization = defaultdict(list)
    for instr in core_instrs:
        for engine in SLOT_LIMITS:
            if engine == 'debug':
                continue
            count = len(instr.get(engine, []))
            utilization[engine].append(count)

    return dict(utilization)


def print_utilization_histogram(instrs: list[dict]):
    """Print a histogram of slot utilization."""
    util = analyze_slot_utilization_per_cycle(instrs)

    print("\n--- Slot Utilization Histogram ---")
    for engine in ['valu', 'alu', 'load', 'store']:
        if engine not in util:
            continue
        counts = util[engine]
        max_slots = SLOT_LIMITS[engine]

        # Bucket by utilization level
        buckets = defaultdict(int)
        for c in counts:
            buckets[c] += 1

        print(f"\n{engine.upper()} (max {max_slots} slots):")
        for slots in range(max_slots + 1):
            count = buckets[slots]
            pct = count / len(counts) * 100
            bar = '█' * int(pct / 2)
            print(f"  {slots:2d} slots: {count:4d} cycles ({pct:5.1f}%) {bar}")


def compare_configurations():
    """Compare different kernel configurations."""
    configs = [
        ("Baseline (no VALU shift)", {"enable_valu_shift": False}),
        ("With VALU→ALU shift", {"enable_valu_shift": True}),
        ("No round3 VALU select", {"enable_valu_shift": True, "enable_round3_valu_select": False}),
    ]

    results = []
    for name, kwargs in configs:
        print(f"\nAnalyzing: {name}...")
        analysis = analyze_kernel_build(name, **kwargs)
        results.append(analysis)

    # Print comparison table
    print("\n" + "=" * 80)
    print("CONFIGURATION COMPARISON")
    print("=" * 80)

    print(f"\n{'Configuration':<35} {'Cycles':>8} {'VALU Ops':>10} {'ALU Ops':>10} {'Overhead':>10}")
    print("-" * 80)
    for r in results:
        valu_ops = r.ops_after_shift.get('valu', r.ops_by_engine.get('valu', 0)) if r.ops_after_shift else r.ops_by_engine.get('valu', 0)
        alu_ops = r.ops_after_shift.get('alu', r.ops_by_engine.get('alu', 0)) if r.ops_after_shift else r.ops_by_engine.get('alu', 0)
        print(f"{r.config_name:<35} {r.total_cycles:>8} {valu_ops:>10} {alu_ops:>10} {r.scheduling_overhead_pct:>9.1f}%")

    # Print detailed reports
    for r in results:
        r.print_report()


def analyze_bounds_check_savings():
    """Analyze potential savings from tree_depth-based bounds check optimization."""
    # Current: bounds check based on absolute round number
    # Proposed: bounds check based on tree_depth (which wraps)

    forest_height = 10
    n_nodes = 2047
    rounds = 16
    tree_height = forest_height + 1  # 11

    print("\n" + "=" * 60)
    print("BOUNDS CHECK OPTIMIZATION ANALYSIS")
    print("=" * 60)

    print(f"\nParameters: forest_height={forest_height}, n_nodes={n_nodes}, rounds={rounds}")
    print(f"Tree height (wraps at): {tree_height}")

    print(f"\n{'Round':<6} {'tree_depth':<12} {'max_idx (rnd)':<15} {'max_idx (depth)':<16} {'Current Check':<15} {'Optimized Check':<15}")
    print("-" * 85)

    tree_depth = 0
    current_checks = 0
    optimized_checks = 0

    for rnd in range(rounds):
        # Current formula (based on absolute round)
        max_idx_rnd = (1 << (rnd + 2)) - 1
        needs_check_current = max_idx_rnd >= n_nodes

        # Optimized formula (based on tree_depth)
        max_idx_depth = (1 << (tree_depth + 2)) - 1
        needs_check_optimized = max_idx_depth >= n_nodes

        if needs_check_current:
            current_checks += 1
        if needs_check_optimized:
            optimized_checks += 1

        print(f"{rnd:<6} {tree_depth:<12} {max_idx_rnd:<15} {max_idx_depth:<16} {'YES' if needs_check_current else 'no':<15} {'YES' if needs_check_optimized else 'no':<15}")

        # Update tree_depth with wrapping
        tree_depth += 1
        if tree_depth >= tree_height:
            tree_depth = 0

    print(f"\nCurrent bounds checks: {current_checks} rounds")
    print(f"Optimized bounds checks: {optimized_checks} rounds")
    print(f"Savings: {current_checks - optimized_checks} rounds")

    # Estimate VALU op savings
    n_groups = 256 // VLEN  # 32 groups
    ops_per_check = 2  # compare + multiply
    saved_ops = (current_checks - optimized_checks) * n_groups * ops_per_check
    print(f"Estimated VALU ops saved: {saved_ops}")
    print(f"Estimated cycle savings: ~{saved_ops // SLOT_LIMITS['valu']} cycles (if VALU-bound)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze kernel performance")
    parser.add_argument("--compare", action="store_true", help="Compare configurations")
    parser.add_argument("--bounds", action="store_true", help="Analyze bounds check optimization")
    parser.add_argument("--histogram", action="store_true", help="Show slot utilization histogram")
    parser.add_argument("--all", action="store_true", help="Run all analyses")

    args = parser.parse_args()

    if args.all or args.bounds:
        analyze_bounds_check_savings()

    if args.all or args.compare:
        compare_configurations()

    if args.histogram:
        from solver.interleaved_unrolled_kernel import InterleavedUnrolledKernelBuilder
        print("\nBuilding kernel for histogram analysis...")
        builder = InterleavedUnrolledKernelBuilder()
        builder.build_kernel(10, 2047, 256, 16, use_greedy=True, enable_valu_shift=True)
        print_utilization_histogram(builder.instrs)

    if not any([args.compare, args.bounds, args.histogram, args.all]):
        # Default: run single analysis
        print("Running default analysis...")
        analysis = analyze_kernel_build("Default (greedy + VALU shift)")
        analysis.print_report()
