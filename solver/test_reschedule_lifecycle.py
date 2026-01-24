"""Unit tests for memory lifecycle tracking in reschedule module."""

import unittest

from solver.reschedule import (
    Op,
    compute_dependencies,
    compute_first_writers,
    compute_last_readers,
    compute_last_users,
    compute_memory_accesses,
    emit_scheduled_with_debug,
    extract_ops,
)


class TestComputeFirstWriters(unittest.TestCase):
    """Tests for compute_first_writers function."""

    def test_single_op_single_write(self):
        """Single op writing to one address."""
        op = Op(id=0, engine="load", slot=("const", 10, 42))
        op.writes = {10}
        ops = [op]
        result = compute_first_writers(ops)
        self.assertEqual(result, {10: 0})

    def test_multiple_ops_same_address(self):
        """Multiple ops writing to same address - first one wins."""
        op1 = Op(id=0, engine="alu", slot=("op",))
        op1.writes = {10}
        op2 = Op(id=1, engine="alu", slot=("op",))
        op2.writes = {10}
        ops = [op1, op2]
        result = compute_first_writers(ops)
        self.assertEqual(result, {10: 0})

    def test_multiple_ops_different_addresses(self):
        """Multiple ops writing to different addresses."""
        op1 = Op(id=0, engine="alu", slot=("op",))
        op1.writes = {10}
        op2 = Op(id=1, engine="alu", slot=("op",))
        op2.writes = {20}
        ops = [op1, op2]
        result = compute_first_writers(ops)
        self.assertEqual(result, {10: 0, 20: 1})

    def test_vector_write(self):
        """Op writing to multiple addresses (vector)."""
        op = Op(id=0, engine="valu", slot=("vbroadcast", 10, 0))
        op.writes = {10, 11, 12, 13, 14, 15, 16, 17}  # VLEN=8
        ops = [op]
        result = compute_first_writers(ops)
        for addr in range(10, 18):
            self.assertEqual(result[addr], 0)


class TestComputeLastUsers(unittest.TestCase):
    """Tests for compute_last_users function."""

    def test_single_read(self):
        """Single op reading one address."""
        op = Op(id=0, engine="alu", slot=("op",))
        op.reads = {10}
        ops = [op]
        result = compute_last_users(ops)
        self.assertEqual(result, {10: 0})

    def test_multiple_reads_same_address(self):
        """Multiple ops reading same address - last one wins."""
        op1 = Op(id=0, engine="alu", slot=("op",))
        op1.reads = {10}
        op2 = Op(id=1, engine="alu", slot=("op",))
        op2.reads = {10}
        ops = [op1, op2]
        result = compute_last_users(ops)
        self.assertEqual(result, {10: 1})

    def test_read_and_write_same_address(self):
        """Read and write to same address - later one wins."""
        op1 = Op(id=0, engine="load", slot=("const", 10, 42))
        op1.writes = {10}
        op2 = Op(id=1, engine="alu", slot=("op",))
        op2.reads = {10}
        ops = [op1, op2]
        result = compute_last_users(ops)
        self.assertEqual(result, {10: 1})


class TestComputeMemoryAccesses(unittest.TestCase):
    """Tests for compute_memory_accesses function."""

    def test_load_operation(self):
        """Load op should be tracked."""
        op = Op(id=0, engine="load", slot=("load", 10, 20))
        op.reads = {20}
        op.writes = {10}
        ops = [op]
        loads, stores = compute_memory_accesses(ops)
        self.assertEqual(loads, {0: [20]})
        self.assertEqual(stores, {})

    def test_vload_operation(self):
        """Vector load should be tracked."""
        op = Op(id=0, engine="load", slot=("vload", 10, 20))
        op.reads = {20}
        op.writes = set(range(10, 18))
        ops = [op]
        loads, stores = compute_memory_accesses(ops)
        self.assertEqual(loads, {0: [20]})

    def test_store_operation(self):
        """Store op should be tracked."""
        op = Op(id=0, engine="store", slot=("store", 10, 20))
        op.reads = {10, 20}
        ops = [op]
        loads, stores = compute_memory_accesses(ops)
        self.assertEqual(loads, {})
        self.assertEqual(stores, {0: [10]})

    def test_vstore_operation(self):
        """Vector store should be tracked."""
        op = Op(id=0, engine="store", slot=("vstore", 10, 20))
        op.reads = {10} | set(range(20, 28))
        ops = [op]
        loads, stores = compute_memory_accesses(ops)
        self.assertEqual(stores, {0: [10]})

    def test_const_not_tracked(self):
        """Const operations should not be tracked as memory access."""
        op = Op(id=0, engine="load", slot=("const", 10, 42))
        op.writes = {10}
        ops = [op]
        loads, stores = compute_memory_accesses(ops)
        self.assertEqual(loads, {})
        self.assertEqual(stores, {})


class TestEmitScheduledWithDebug(unittest.TestCase):
    """Tests for emit_scheduled_with_debug function."""

    def test_simple_allocation(self):
        """Should emit scratch_alloc for first write."""
        op = Op(id=0, engine="load", slot=("const", 10, 42), cycle=0)
        op.writes = {10}
        ops = [op]
        op_by_id = {0: op}

        instrs = emit_scheduled_with_debug(ops, 1, op_by_id)

        self.assertEqual(len(instrs), 1)
        self.assertIn("debug", instrs[0])
        debug_slots = instrs[0]["debug"]
        alloc_events = [s for s in debug_slots if s[0] == "scratch_alloc"]
        self.assertEqual(len(alloc_events), 1)
        self.assertEqual(alloc_events[0], ("scratch_alloc", 10, 1))

    def test_simple_retirement(self):
        """Should emit scratch_retire for last read."""
        op1 = Op(id=0, engine="load", slot=("const", 10, 42), cycle=0)
        op1.writes = {10}
        op2 = Op(id=1, engine="alu", slot=("+", 20, 10, 10), cycle=1)
        op2.reads = {10}
        op2.writes = {20}
        ops = [op1, op2]
        op_by_id = {0: op1, 1: op2}

        instrs = emit_scheduled_with_debug(ops, 2, op_by_id)

        # Cycle 0: alloc for addr 10
        debug_0 = instrs[0].get("debug", [])
        allocs_0 = [s for s in debug_0 if s[0] == "scratch_alloc"]
        self.assertTrue(any(s[1] == 10 for s in allocs_0))

        # Cycle 1: alloc for addr 20, retire for addr 10
        debug_1 = instrs[1].get("debug", [])
        retires_1 = [s for s in debug_1 if s[0] == "scratch_retire"]
        self.assertTrue(any(s[1] == 10 for s in retires_1))

    def test_memory_load_event(self):
        """Should emit mem_load for load operations."""
        op = Op(id=0, engine="load", slot=("load", 10, 20), cycle=0)
        op.reads = {20}
        op.writes = {10}
        ops = [op]
        op_by_id = {0: op}

        instrs = emit_scheduled_with_debug(ops, 1, op_by_id)

        debug_slots = instrs[0].get("debug", [])
        load_events = [s for s in debug_slots if s[0] == "mem_load"]
        self.assertEqual(len(load_events), 1)
        self.assertEqual(load_events[0], ("mem_load", 20))

    def test_memory_store_event(self):
        """Should emit mem_store for store operations."""
        op = Op(id=0, engine="store", slot=("store", 10, 20), cycle=0)
        op.reads = {10, 20}
        ops = [op]
        op_by_id = {0: op}

        instrs = emit_scheduled_with_debug(ops, 1, op_by_id)

        debug_slots = instrs[0].get("debug", [])
        store_events = [s for s in debug_slots if s[0] == "mem_store"]
        self.assertEqual(len(store_events), 1)
        self.assertEqual(store_events[0], ("mem_store", 10))

    def test_with_scratch_debug_grouping(self):
        """Should group addresses by named regions when scratch_debug provided."""
        # Simulate a vector allocation (8 addresses)
        op = Op(id=0, engine="load", slot=("vload", 10, 100), cycle=0)
        op.reads = {100}
        op.writes = set(range(10, 18))
        ops = [op]
        op_by_id = {0: op}

        # scratch_debug maps base addr -> (name, length)
        scratch_debug = {10: ("vec_a", 8)}

        instrs = emit_scheduled_with_debug(ops, 1, op_by_id, scratch_debug)

        debug_slots = instrs[0].get("debug", [])
        alloc_events = [s for s in debug_slots if s[0] == "scratch_alloc"]

        # Should have one grouped allocation for the whole region
        self.assertEqual(len(alloc_events), 1)
        self.assertEqual(alloc_events[0], ("scratch_alloc", 10, 8))


if __name__ == "__main__":
    unittest.main()
