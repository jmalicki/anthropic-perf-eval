"""
Test the modular kernel builder against the original implementation.
"""

import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver.kernel_builder import (
    KernelConfig,
    ModularKernelBuilder,
    TreeAccessMode,
    HashMode,
    SchedulingMode,
)


class TestModularKernelBuilder(unittest.TestCase):
    """Test cases for the modular kernel builder."""

    def test_basic_build(self):
        """Test that the builder produces a valid kernel."""
        config = KernelConfig(
            forest_height=10,
            n_nodes=2047,
            batch_size=256,
            rounds=16,
            tree_access_mode=TreeAccessMode.HYBRID,
            scheduling_mode=SchedulingMode.GREEDY_MULTIPASS,
        )
        builder = ModularKernelBuilder(config)
        builder.build()

        # Check that instructions were generated
        self.assertGreater(len(builder.instrs), 0)
        print(f"Generated {len(builder.instrs)} instructions")

        # Check that we have pause instructions at start and end
        self.assertIn("flow", builder.instrs[0])
        self.assertIn("flow", builder.instrs[-1])

        # Check cycle count (should be around 1880 for optimal)
        cycles = len(builder.instrs) - 2
        print(f"Cycles: {cycles}")
        self.assertLess(cycles, 3000, "Too many cycles - something is wrong")

    def test_scattered_loads_only(self):
        """Test with scattered loads only (no VALU selection)."""
        config = KernelConfig(
            tree_access_mode=TreeAccessMode.SCATTERED_LOADS,
            scheduling_mode=SchedulingMode.GREEDY_SINGLE,
        )
        builder = ModularKernelBuilder(config)
        builder.build()

        cycles = len(builder.instrs) - 2
        print(f"Scattered loads only: {cycles} cycles")
        # Should be slower than hybrid
        self.assertGreater(cycles, 1800)

    def test_valu_selection_only(self):
        """Test with VALU selection for depths 0-3."""
        config = KernelConfig(
            tree_access_mode=TreeAccessMode.VALU_SELECTION,
            valu_selection_max_depth=3,
            scheduling_mode=SchedulingMode.GREEDY_SINGLE,
        )
        builder = ModularKernelBuilder(config)
        # This should fail because VALU selection only handles depths 0-3
        # but rounds 4+ need depths 4+
        with self.assertRaises(ValueError):
            builder.build()

    def test_hybrid_mode(self):
        """Test hybrid mode (VALU selection + scattered loads)."""
        config = KernelConfig(
            tree_access_mode=TreeAccessMode.HYBRID,
            valu_selection_max_depth=3,
            scheduling_mode=SchedulingMode.GREEDY_SINGLE,
        )
        builder = ModularKernelBuilder(config)
        builder.build()

        cycles = len(builder.instrs) - 2
        print(f"Hybrid mode: {cycles} cycles")

    def test_scratch_usage(self):
        """Test that scratch usage is within budget."""
        config = KernelConfig(
            tree_access_mode=TreeAccessMode.HYBRID,
        )
        builder = ModularKernelBuilder(config)
        builder.build()

        from problem import SCRATCH_SIZE
        self.assertLessEqual(builder.scratch_ptr, SCRATCH_SIZE,
                            f"Scratch usage {builder.scratch_ptr} exceeds budget {SCRATCH_SIZE}")


def run_comparison():
    """Compare modular builder output with original implementation."""
    print("=" * 60)
    print("Comparing Modular Builder with Original Implementation")
    print("=" * 60)

    # Build with modular builder
    config = KernelConfig(
        forest_height=10,
        n_nodes=2047,
        batch_size=256,
        rounds=16,
        tree_access_mode=TreeAccessMode.HYBRID,
        scheduling_mode=SchedulingMode.GREEDY_MULTIPASS,
        enable_valu_shift=True,
    )
    print("\nBuilding with ModularKernelBuilder...")
    modular_builder = ModularKernelBuilder(config)
    modular_builder.build()
    modular_cycles = len(modular_builder.instrs) - 2
    print(f"Modular cycles: {modular_cycles}")

    # Build with original implementation
    print("\nBuilding with InterleavedUnrolledKernelBuilder...")
    from solver.interleaved_unrolled_kernel import InterleavedUnrolledKernelBuilder
    original_builder = InterleavedUnrolledKernelBuilder()
    original_builder.build_kernel(
        forest_height=10,
        n_nodes=2047,
        batch_size=256,
        rounds=16,
        use_greedy=True,
        use_multipass_scheduler=True,
        enable_valu_shift=True,
    )
    original_cycles = len(original_builder.instrs) - 2
    print(f"Original cycles: {original_cycles}")

    print("\n" + "=" * 60)
    print(f"Comparison:")
    print(f"  Modular:  {modular_cycles} cycles")
    print(f"  Original: {original_cycles} cycles")
    print(f"  Diff:     {modular_cycles - original_cycles} cycles ({(modular_cycles - original_cycles) / original_cycles * 100:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    # Run comparison first
    run_comparison()

    # Then run unit tests
    print("\n\nRunning unit tests...")
    unittest.main(verbosity=2, exit=False)
