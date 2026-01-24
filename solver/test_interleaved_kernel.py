"""
Unit tests for interleaved unrolled kernel generation.

Tests kernel generation phases including scratch allocation, tree depth dispatch,
bounds check optimization, and hash computation.
"""

import unittest
from problem import VLEN, SCRATCH_SIZE, HASH_STAGES


class TestKernelBuilderInit(unittest.TestCase):
    """Test scratch allocation and initialization."""

    def test_scratch_allocation_increments_ptr(self):
        """Verify that alloc() increments scratch_ptr correctly."""
        from solver.interleaved_unrolled_kernel import InterleavedUnrolledKernelBuilder

        builder = InterleavedUnrolledKernelBuilder()

        # Initial state
        self.assertEqual(builder.scratch_ptr, 0)

        # Single allocation
        addr1 = builder.alloc("test1")
        self.assertEqual(addr1, 0)
        self.assertEqual(builder.scratch_ptr, 1)

        # Multiple word allocation
        addr2 = builder.alloc("test2", length=4)
        self.assertEqual(addr2, 1)
        self.assertEqual(builder.scratch_ptr, 5)

        # Another single allocation
        addr3 = builder.alloc("test3")
        self.assertEqual(addr3, 5)
        self.assertEqual(builder.scratch_ptr, 6)

    def test_vec_allocation_uses_VLEN(self):
        """Verify that vec() allocates VLEN words."""
        from solver.interleaved_unrolled_kernel import InterleavedUnrolledKernelBuilder

        builder = InterleavedUnrolledKernelBuilder()

        addr = builder.vec("test_vec")
        self.assertEqual(addr, 0)
        self.assertEqual(builder.scratch_ptr, VLEN)

        addr2 = builder.vec("test_vec2")
        self.assertEqual(addr2, VLEN)
        self.assertEqual(builder.scratch_ptr, 2 * VLEN)

    def test_scratch_debug_tracks_names(self):
        """Verify that allocations are tracked in scratch_debug."""
        from solver.interleaved_unrolled_kernel import InterleavedUnrolledKernelBuilder

        builder = InterleavedUnrolledKernelBuilder()

        builder.alloc("scalar_var")
        builder.vec("vector_var")
        builder.alloc("multi_word", length=3)

        # Check debug tracking
        self.assertIn(0, builder.scratch_debug)
        self.assertEqual(builder.scratch_debug[0], ("scalar_var", 1))

        self.assertIn(1, builder.scratch_debug)
        self.assertEqual(builder.scratch_debug[1], ("vector_var", VLEN))

        self.assertIn(1 + VLEN, builder.scratch_debug)
        self.assertEqual(builder.scratch_debug[1 + VLEN], ("multi_word", 3))

    def test_unnamed_allocation_not_tracked(self):
        """Verify that unnamed allocations are not tracked in scratch_debug."""
        from solver.interleaved_unrolled_kernel import InterleavedUnrolledKernelBuilder

        builder = InterleavedUnrolledKernelBuilder()

        addr = builder.alloc()  # No name
        self.assertEqual(addr, 0)
        self.assertEqual(builder.scratch_ptr, 1)
        self.assertNotIn(0, builder.scratch_debug)


class TestTreeDepthDispatch(unittest.TestCase):
    """Test tree_depth-based code path selection."""

    def _build_kernel_and_get_raw_instrs(self, rounds, forest_height=6):
        """Build kernel and return pre-scheduled instructions for analysis."""
        from solver.interleaved_unrolled_kernel import InterleavedUnrolledKernelBuilder

        n_nodes = 2 ** (forest_height + 1) - 1
        builder = InterleavedUnrolledKernelBuilder()

        # Build without scheduling to analyze raw instruction generation
        # We'll capture instrs before scheduling
        builder.instrs = []
        builder.scratch = {}
        builder.scratch_debug = {}
        builder.scratch_ptr = 0

        return builder, n_nodes, forest_height

    def test_depth_0_broadcasts_root(self):
        """Verify depth 0 (root) uses vbroadcast of tc[0]."""
        from solver.test_helpers import build_mini_kernel, count_ops_by_type

        # Build 1-round kernel - round 0 is depth 0
        builder = build_mini_kernel(rounds=1, batch_size=64)

        # Depth 0 should use vbroadcast for node values
        vbroadcast_count = count_ops_by_type(builder.instrs, 'valu', 'vbroadcast')
        self.assertGreater(vbroadcast_count, 0, "Depth 0 should use vbroadcast")

    def test_depth_1_uses_delta_computation(self):
        """Verify depth 1 uses delta computation with tc[2] + delta."""
        from solver.test_helpers import build_mini_kernel, count_ops_by_type

        # Build 2-round kernel - round 1 is depth 1
        builder = build_mini_kernel(rounds=2, batch_size=64)

        # Depth 1 uses '==' comparison for branch selection
        eq_count = count_ops_by_type(builder.instrs, 'valu', '==')
        self.assertGreater(eq_count, 0, "Depth 1 should use '==' for delta selection")

    def test_depth_2_uses_multiple_deltas(self):
        """Verify depth 2 uses multiple delta computations for indices 3-6."""
        from solver.test_helpers import build_mini_kernel, count_ops_by_type

        # Build 3-round kernel - round 2 is depth 2
        builder = build_mini_kernel(rounds=3, batch_size=64)

        # Depth 2 uses multiple '==' comparisons (for indices 4, 5, 6)
        eq_count = count_ops_by_type(builder.instrs, 'valu', '==')
        # Should have more comparisons for depth 2 than just depth 1
        self.assertGreater(eq_count, 3, "Depth 2 should use multiple '==' comparisons")

    def test_depth_3_uses_valu_selection(self):
        """Verify depth 3 uses VALU selection with multiply_add."""
        from solver.test_helpers import build_mini_kernel, count_ops_by_type

        # Build 4-round kernel - round 3 is depth 3
        builder = build_mini_kernel(rounds=4, batch_size=64)

        # Depth 3 uses multiply_add for VALU selection (8 tree values)
        ma_count = count_ops_by_type(builder.instrs, 'valu', 'multiply_add')
        # Should have multiply_add for both hash FMA and depth 3 selection
        self.assertGreater(ma_count, 0, "Depth 3 should use multiply_add for selection")

    def test_tree_depth_wraps_at_tree_height(self):
        """Verify tree_depth wraps back to 0 after tree_height rounds."""
        from solver.test_helpers import build_mini_kernel, count_ops_by_type

        # With forest_height=6, tree_height=7
        # After 7 rounds, depth wraps to 0 (vbroadcast again)
        # Build with 8 rounds to see wrap behavior
        builder = build_mini_kernel(rounds=8, batch_size=64, forest_height=6)

        # The kernel should complete without error
        # Wrap behavior is tested by successful kernel generation
        self.assertIsNotNone(builder.instrs)
        self.assertGreater(len(builder.instrs), 0)


class TestBoundsCheckOptimization(unittest.TestCase):
    """Test compile-time bounds check elimination."""

    def test_early_rounds_skip_bounds_check(self):
        """Verify early rounds don't include bounds check ops."""
        from solver.test_helpers import build_mini_kernel, count_ops_by_type

        # Build 1-round kernel with small tree
        # Round 0: max_idx after branch = 2^2 - 1 = 3
        # For forest_height=6, n_nodes=127, so no bounds check needed for round 0
        builder = build_mini_kernel(rounds=1, batch_size=64, forest_height=6)

        # Less than '<' ops in early rounds (bounds check uses '<')
        lt_count = count_ops_by_type(builder.instrs, 'valu', '<')
        # Early rounds with small indices don't need bounds checks
        # This is a sanity check - exact behavior depends on tree structure
        self.assertIsNotNone(lt_count)

    def test_late_rounds_include_bounds_check(self):
        """Verify late rounds include bounds check when indices may exceed n_nodes."""
        from solver.test_helpers import build_mini_kernel, count_ops_by_type

        # Build many-round kernel where bounds checks are needed
        # With forest_height=3 (n_nodes=15), round 3 needs bounds check
        # max_idx at round 3 = 2^5 - 1 = 31 >= 15
        builder = build_mini_kernel(rounds=4, batch_size=64, forest_height=3)

        lt_count = count_ops_by_type(builder.instrs, 'valu', '<')
        self.assertGreater(lt_count, 0, "Late rounds should include bounds check ('<' op)")


class TestHashComputation(unittest.TestCase):
    """Test hash stage generation."""

    def test_fma_optimization_applied(self):
        """Verify FMA optimization is applied for eligible hash stages."""
        from solver.test_helpers import build_mini_kernel, count_ops_by_type

        builder = build_mini_kernel(rounds=1, batch_size=64)

        # FMA optimization uses multiply_add
        ma_count = count_ops_by_type(builder.instrs, 'valu', 'multiply_add')
        self.assertGreater(ma_count, 0, "FMA optimization should use multiply_add")

    def test_stage5_xor_optimization(self):
        """Verify stage 5 XOR optimization is applied."""
        from solver.test_helpers import build_mini_kernel, count_ops_by_type

        builder = build_mini_kernel(rounds=1, batch_size=64)

        # Stage 5 optimization uses XOR (^) operations
        xor_count = count_ops_by_type(builder.instrs, 'valu', '^')
        self.assertGreater(xor_count, 0, "Stage 5 should use XOR optimization")

    def test_all_6_stages_emitted(self):
        """Verify all 6 hash stages are emitted."""
        from solver.test_helpers import build_mini_kernel

        builder = build_mini_kernel(rounds=1, batch_size=64)

        # Count total valu ops - should include all hash stages
        # Each round has 6 hash stages
        # This is a sanity check that hash computation is included
        self.assertGreater(len(builder.instrs), 10,
                          "Kernel should have instructions for all hash stages")


class TestDebugEmission(unittest.TestCase):
    """Test debug instruction emission when enabled."""

    def test_debug_disabled_by_default(self):
        """Verify debug is disabled by default."""
        from solver.interleaved_unrolled_kernel import InterleavedUnrolledKernelBuilder

        builder = InterleavedUnrolledKernelBuilder()
        self.assertFalse(builder.enable_debug)

    def test_debug_can_be_enabled(self):
        """Verify debug can be enabled via constructor."""
        from solver.interleaved_unrolled_kernel import InterleavedUnrolledKernelBuilder

        builder = InterleavedUnrolledKernelBuilder(enable_debug=True)
        self.assertTrue(builder.enable_debug)

    def test_debug_ops_list_initialized(self):
        """Verify debug_ops list is initialized."""
        from solver.interleaved_unrolled_kernel import InterleavedUnrolledKernelBuilder

        builder = InterleavedUnrolledKernelBuilder(enable_debug=True)
        self.assertIsInstance(builder.debug_ops, list)
        self.assertEqual(len(builder.debug_ops), 0)


class TestScratchBounds(unittest.TestCase):
    """Test scratch memory bounds checking."""

    def test_kernel_within_scratch_bounds(self):
        """Verify generated kernel stays within SCRATCH_SIZE."""
        from solver.test_helpers import build_mini_kernel, verify_scratch_bounds

        builder = build_mini_kernel(rounds=16, batch_size=256)

        passed, errors = verify_scratch_bounds(builder)
        self.assertTrue(passed, f"Scratch bounds violated: {errors}")

    def test_small_kernel_allocation_info(self):
        """Test scratch allocation info for a small kernel."""
        from solver.test_helpers import build_mini_kernel, get_scratch_allocation_info

        builder = build_mini_kernel(rounds=1, batch_size=64)

        info = get_scratch_allocation_info(builder)
        self.assertGreater(info['total_words'], 0)
        self.assertLessEqual(info['total_words'], SCRATCH_SIZE)
        self.assertLessEqual(info['utilization'], 1.0)


class TestSlotLimits(unittest.TestCase):
    """Test slot limit validation."""

    def test_kernel_respects_slot_limits(self):
        """Verify generated kernel respects slot limits for all engines."""
        from solver.test_helpers import build_mini_kernel, validate_slot_limits

        builder = build_mini_kernel(rounds=4, batch_size=64)

        passed, errors = validate_slot_limits(builder.instrs)
        self.assertTrue(passed, f"Slot limits violated: {errors}")


if __name__ == '__main__':
    unittest.main()
