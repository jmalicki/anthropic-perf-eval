"""Unit tests for shift analysis module.

Tests the static analysis functions that determine which VALU operations
can be shifted to ALU and which vbroadcast operations can be eliminated.
"""

import unittest
import sys
import os

# Add the parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver.shift_analysis import (
    ShiftAnalysis, is_shiftable_op, is_vbroadcast,
    get_broadcast_dest_range, get_broadcast_scalar_source,
    analyze_shifts, can_eliminate_broadcast,
    VALU_TO_ALU_OPS, VLEN
)
from solver.reschedule import Op


class TestIsShiftableOp(unittest.TestCase):
    """Tests for is_shiftable_op function."""

    def test_binary_valu_addition_is_shiftable(self):
        """VALU + op should be shiftable."""
        op = Op(id=1, engine='valu', slot=('+', 10, 20, 30))
        self.assertTrue(is_shiftable_op(op))

    def test_binary_valu_subtraction_is_shiftable(self):
        """VALU - op should be shiftable."""
        op = Op(id=1, engine='valu', slot=('-', 10, 20, 30))
        self.assertTrue(is_shiftable_op(op))

    def test_binary_valu_multiply_is_shiftable(self):
        """VALU * op should be shiftable."""
        op = Op(id=1, engine='valu', slot=('*', 10, 20, 30))
        self.assertTrue(is_shiftable_op(op))

    def test_binary_valu_bitwise_and_is_shiftable(self):
        """VALU & op should be shiftable."""
        op = Op(id=1, engine='valu', slot=('&', 10, 20, 30))
        self.assertTrue(is_shiftable_op(op))

    def test_all_binary_ops_are_shiftable(self):
        """All ops in VALU_TO_ALU_OPS should be shiftable."""
        for op_type in VALU_TO_ALU_OPS:
            op = Op(id=1, engine='valu', slot=(op_type, 10, 20, 30))
            self.assertTrue(is_shiftable_op(op), f"{op_type} should be shiftable")

    def test_vbroadcast_not_shiftable(self):
        """vbroadcast is NOT shiftable (but can be eliminated)."""
        op = Op(id=1, engine='valu', slot=('vbroadcast', 10, 5))
        self.assertFalse(is_shiftable_op(op))

    def test_multiply_add_not_shiftable(self):
        """multiply_add is NOT shiftable (would expand to 2 ops)."""
        op = Op(id=1, engine='valu', slot=('multiply_add', 10, 20, 30, 40))
        self.assertFalse(is_shiftable_op(op))

    def test_vmultiply_add_not_shiftable(self):
        """vmultiply_add is NOT shiftable."""
        op = Op(id=1, engine='valu', slot=('vmultiply_add', 10, 20, 30, 40))
        self.assertFalse(is_shiftable_op(op))

    def test_non_valu_ops_not_shiftable(self):
        """ALU, flow, load, store ops are not shiftable."""
        test_cases = [
            Op(id=1, engine='alu', slot=('+', 10, 20, 30)),
            Op(id=2, engine='flow', slot=('vselect', 10, 20, 30, 40)),
            Op(id=3, engine='load', slot=('const', 10, 100)),
            Op(id=4, engine='store', slot=('store', 10, 20)),
        ]
        for op in test_cases:
            self.assertFalse(is_shiftable_op(op), f"{op.engine} should not be shiftable")


class TestIsVbroadcast(unittest.TestCase):
    """Tests for is_vbroadcast function."""

    def test_vbroadcast_detected(self):
        """vbroadcast should be detected."""
        op = Op(id=1, engine='valu', slot=('vbroadcast', 10, 5))
        self.assertTrue(is_vbroadcast(op))

    def test_binary_valu_not_vbroadcast(self):
        """Binary VALU ops are not vbroadcast."""
        op = Op(id=1, engine='valu', slot=('+', 10, 20, 30))
        self.assertFalse(is_vbroadcast(op))

    def test_non_valu_not_vbroadcast(self):
        """Non-VALU ops are not vbroadcast."""
        op = Op(id=1, engine='alu', slot=('vbroadcast', 10, 5))
        self.assertFalse(is_vbroadcast(op))


class TestBroadcastDestRange(unittest.TestCase):
    """Tests for get_broadcast_dest_range function."""

    def test_returns_vlen_addresses(self):
        """Should return VLEN consecutive addresses."""
        op = Op(id=1, engine='valu', slot=('vbroadcast', 100, 5))
        result = get_broadcast_dest_range(op)
        expected = {100 + i for i in range(VLEN)}
        self.assertEqual(result, expected)

    def test_non_broadcast_returns_empty(self):
        """Non-broadcast ops should return empty set."""
        op = Op(id=1, engine='valu', slot=('+', 10, 20, 30))
        self.assertEqual(get_broadcast_dest_range(op), set())


class TestBroadcastScalarSource(unittest.TestCase):
    """Tests for get_broadcast_scalar_source function."""

    def test_returns_scalar_source(self):
        """Should return the scalar source address."""
        op = Op(id=1, engine='valu', slot=('vbroadcast', 100, 5))
        self.assertEqual(get_broadcast_scalar_source(op), 5)

    def test_non_broadcast_returns_none(self):
        """Non-broadcast ops should return None."""
        op = Op(id=1, engine='valu', slot=('+', 10, 20, 30))
        self.assertIsNone(get_broadcast_scalar_source(op))


class TestAnalyzeShifts(unittest.TestCase):
    """Tests for analyze_shifts function."""

    def test_identifies_shiftable_ops(self):
        """Should identify all shiftable ops."""
        ops = [
            Op(id=0, engine='valu', slot=('+', 10, 20, 30),
               writes={10+i for i in range(8)}, reads={20+i for i in range(8)} | {30+i for i in range(8)}),
            Op(id=1, engine='valu', slot=('-', 40, 50, 60),
               writes={40+i for i in range(8)}, reads={50+i for i in range(8)} | {60+i for i in range(8)}),
            Op(id=2, engine='valu', slot=('multiply_add', 70, 80, 90, 100),
               writes={70+i for i in range(8)}, reads={80+i for i in range(8)} | {90+i for i in range(8)} | {100+i for i in range(8)}),
        ]
        op_by_id = {op.id: op for op in ops}

        analysis = analyze_shifts(ops, op_by_id)

        self.assertIn(0, analysis.shiftable_ops)
        self.assertIn(1, analysis.shiftable_ops)
        self.assertNotIn(2, analysis.shiftable_ops)  # multiply_add not shiftable

    def test_identifies_broadcasts(self):
        """Should identify all vbroadcast ops."""
        ops = [
            Op(id=0, engine='valu', slot=('vbroadcast', 10, 5),
               writes={10+i for i in range(8)}, reads={5}),
            Op(id=1, engine='valu', slot=('vbroadcast', 20, 6),
               writes={20+i for i in range(8)}, reads={6}),
        ]
        op_by_id = {op.id: op for op in ops}

        analysis = analyze_shifts(ops, op_by_id)

        self.assertEqual(analysis.broadcast_ops, {0: 5, 1: 6})

    def test_broadcast_with_all_shiftable_consumers_is_eliminable(self):
        """Broadcast is eliminable when ALL consumers are shiftable."""
        ops = [
            Op(id=0, engine='valu', slot=('vbroadcast', 10, 5),
               writes={10+i for i in range(8)}, reads={5}),
            Op(id=1, engine='valu', slot=('+', 20, 10, 30),
               writes={20+i for i in range(8)},
               reads={10+i for i in range(8)} | {30+i for i in range(8)},
               dep_ids={0}),
        ]
        op_by_id = {op.id: op for op in ops}

        analysis = analyze_shifts(ops, op_by_id)

        self.assertIn(0, analysis.broadcast_ops)
        self.assertIn(1, analysis.shiftable_ops)
        self.assertEqual(analysis.broadcast_consumers[0], {1})
        self.assertIn(0, analysis.eliminable_broadcasts)

    def test_broadcast_with_non_shiftable_consumer_not_eliminable(self):
        """Broadcast NOT eliminable if ANY consumer can't be shifted."""
        ops = [
            Op(id=0, engine='valu', slot=('vbroadcast', 10, 5),
               writes={10+i for i in range(8)}, reads={5}),
            Op(id=1, engine='flow', slot=('vselect', 20, 10, 30, 40),
               writes={20+i for i in range(8)},
               reads={10+i for i in range(8)} | {30+i for i in range(8)} | {40+i for i in range(8)},
               dep_ids={0}),
        ]
        op_by_id = {op.id: op for op in ops}

        analysis = analyze_shifts(ops, op_by_id)

        self.assertIn(0, analysis.broadcast_ops)
        self.assertNotIn(1, analysis.shiftable_ops)  # vselect not shiftable
        self.assertNotIn(0, analysis.eliminable_broadcasts)

    def test_broadcast_with_mixed_consumers(self):
        """Broadcast NOT eliminable if even ONE consumer can't shift."""
        ops = [
            Op(id=0, engine='valu', slot=('vbroadcast', 10, 5),
               writes={10+i for i in range(8)}, reads={5}),
            Op(id=1, engine='valu', slot=('+', 20, 10, 30),  # Can shift
               writes={20+i for i in range(8)},
               reads={10+i for i in range(8)} | {30+i for i in range(8)},
               dep_ids={0}),
            Op(id=2, engine='valu', slot=('multiply_add', 40, 10, 50, 60),  # Can't shift
               writes={40+i for i in range(8)},
               reads={10+i for i in range(8)} | {50+i for i in range(8)} | {60+i for i in range(8)},
               dep_ids={0}),
        ]
        op_by_id = {op.id: op for op in ops}

        analysis = analyze_shifts(ops, op_by_id)

        self.assertIn(1, analysis.shiftable_ops)
        self.assertNotIn(2, analysis.shiftable_ops)  # multiply_add not shiftable
        self.assertEqual(analysis.broadcast_consumers[0], {1, 2})
        self.assertNotIn(0, analysis.eliminable_broadcasts)

    def test_broadcast_with_multiple_shiftable_consumers(self):
        """Broadcast eliminable when ALL multiple consumers are shiftable."""
        ops = [
            Op(id=0, engine='valu', slot=('vbroadcast', 10, 5),
               writes={10+i for i in range(8)}, reads={5}),
            Op(id=1, engine='valu', slot=('+', 20, 10, 30),
               writes={20+i for i in range(8)},
               reads={10+i for i in range(8)} | {30+i for i in range(8)},
               dep_ids={0}),
            Op(id=2, engine='valu', slot=('*', 40, 10, 50),
               writes={40+i for i in range(8)},
               reads={10+i for i in range(8)} | {50+i for i in range(8)},
               dep_ids={0}),
            Op(id=3, engine='valu', slot=('-', 60, 10, 70),
               writes={60+i for i in range(8)},
               reads={10+i for i in range(8)} | {70+i for i in range(8)},
               dep_ids={0}),
        ]
        op_by_id = {op.id: op for op in ops}

        analysis = analyze_shifts(ops, op_by_id)

        self.assertEqual(analysis.broadcast_consumers[0], {1, 2, 3})
        self.assertIn(0, analysis.eliminable_broadcasts)

    def test_consumer_without_dependency_not_counted(self):
        """Consumer must have dep_ids on broadcast to be counted."""
        ops = [
            Op(id=0, engine='valu', slot=('vbroadcast', 10, 5),
               writes={10+i for i in range(8)}, reads={5}),
            # This op reads address 10 but doesn't depend on op 0
            # (e.g., address 10 was overwritten by another op first)
            Op(id=1, engine='valu', slot=('+', 20, 10, 30),
               writes={20+i for i in range(8)},
               reads={10+i for i in range(8)} | {30+i for i in range(8)},
               dep_ids=set()),  # No dependency on broadcast!
        ]
        op_by_id = {op.id: op for op in ops}

        analysis = analyze_shifts(ops, op_by_id)

        # op 1 is NOT a consumer of broadcast 0 because it doesn't depend on it
        self.assertEqual(analysis.broadcast_consumers[0], set())


class TestCanEliminateBroadcast(unittest.TestCase):
    """Tests for can_eliminate_broadcast function."""

    def setUp(self):
        """Set up common test fixtures."""
        self.ops = [
            Op(id=0, engine='valu', slot=('vbroadcast', 10, 5),
               writes={10+i for i in range(8)}, reads={5}),
            Op(id=1, engine='valu', slot=('+', 20, 10, 30),
               writes={20+i for i in range(8)},
               reads={10+i for i in range(8)} | {30+i for i in range(8)},
               dep_ids={0}),
        ]
        self.op_by_id = {op.id: op for op in self.ops}
        self.analysis = analyze_shifts(self.ops, self.op_by_id)

    def test_eliminable_with_no_shifts_yet(self):
        """Eliminable broadcast can be eliminated before any shifts."""
        # Consumer is shiftable, so elimination is safe
        self.assertTrue(can_eliminate_broadcast(0, self.analysis, set()))

    def test_eliminable_with_consumer_already_shifted(self):
        """Eliminable broadcast can be eliminated after consumer shifted."""
        self.assertTrue(can_eliminate_broadcast(0, self.analysis, {1}))

    def test_non_eliminable_broadcast_cannot_be_eliminated(self):
        """Non-eliminable broadcast cannot be eliminated."""
        ops = [
            Op(id=0, engine='valu', slot=('vbroadcast', 10, 5),
               writes={10+i for i in range(8)}, reads={5}),
            Op(id=1, engine='flow', slot=('vselect', 20, 10, 30, 40),
               writes={20+i for i in range(8)},
               reads={10+i for i in range(8)} | {30+i for i in range(8)} | {40+i for i in range(8)},
               dep_ids={0}),
        ]
        op_by_id = {op.id: op for op in ops}
        analysis = analyze_shifts(ops, op_by_id)

        self.assertFalse(can_eliminate_broadcast(0, analysis, set()))

    def test_unknown_broadcast_cannot_be_eliminated(self):
        """Unknown broadcast ID returns False."""
        self.assertFalse(can_eliminate_broadcast(999, self.analysis, set()))


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and boundary conditions."""

    def test_empty_ops_list(self):
        """Should handle empty ops list."""
        analysis = analyze_shifts([], {})
        self.assertEqual(analysis.shiftable_ops, set())
        self.assertEqual(analysis.broadcast_ops, {})
        self.assertEqual(analysis.eliminable_broadcasts, set())

    def test_no_broadcasts(self):
        """Should handle case with no broadcasts."""
        ops = [
            Op(id=0, engine='valu', slot=('+', 10, 20, 30),
               writes={10+i for i in range(8)}, reads={20+i for i in range(8)} | {30+i for i in range(8)}),
        ]
        op_by_id = {op.id: op for op in ops}

        analysis = analyze_shifts(ops, op_by_id)

        self.assertEqual(analysis.broadcast_ops, {})
        self.assertEqual(analysis.eliminable_broadcasts, set())

    def test_broadcast_with_no_consumers(self):
        """Broadcast with no consumers is not marked eliminable."""
        ops = [
            Op(id=0, engine='valu', slot=('vbroadcast', 10, 5),
               writes={10+i for i in range(8)}, reads={5}),
            # No consumers of broadcast destination 10-17
        ]
        op_by_id = {op.id: op for op in ops}

        analysis = analyze_shifts(ops, op_by_id)

        self.assertIn(0, analysis.broadcast_ops)
        self.assertEqual(analysis.broadcast_consumers[0], set())
        self.assertNotIn(0, analysis.eliminable_broadcasts)


if __name__ == '__main__':
    unittest.main()
