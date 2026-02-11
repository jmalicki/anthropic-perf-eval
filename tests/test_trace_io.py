"""Tests for trace_input and trace_output debug ops in problem.py Machine."""

import pytest
from problem import Machine, DebugInfo


def test_trace_io_log_initialized_empty():
    """trace_io_log is initialized empty and append-only."""
    machine = Machine(
        mem_dump=[],
        program=[],
        debug_info=DebugInfo(scratch_map={}),
    )
    assert machine.trace_io_log == []
    assert isinstance(machine.trace_io_log, list)


def test_trace_input_output_scalar():
    """trace_input and trace_output record scalar (n_elems=1) values."""
    # Cycle 0: load 5 and 7 into scratch 1 and 2
    # Cycle 1: alu 3 = 1 + 2, with trace_input (addrs 1,2) and trace_output (addr 3)
    # Cycle 2: halt
    program = [
        {"load": [("const", 1, 5), ("const", 2, 7)]},
        {
            "alu": [("+", 3, 1, 2)],
            "debug": [
                ("trace_input", "r0_g0_test", 1, 1, 2),
                ("trace_output", "r0_g0_test", 1, 3),
            ],
        },
        {"flow": [("halt",)]},
    ]
    machine = Machine(
        mem_dump=[],
        program=program,
        debug_info=DebugInfo(scratch_map={}),
    )
    machine.enable_debug = True
    machine.run()

    assert len(machine.trace_io_log) == 2
    # trace_input at cycle 1: inputs from addrs 1,2 -> (5, 7)
    input_entry = next(e for e in machine.trace_io_log if e[1] == "input")
    assert input_entry[0] == 1
    assert input_entry[2] == "r0_g0_test"
    assert input_entry[3] == (5, 7)

    # trace_output at cycle 1: output from addr 3 -> (12,)
    output_entry = next(e for e in machine.trace_io_log if e[1] == "output")
    assert output_entry[0] == 1
    assert output_entry[2] == "r0_g0_test"
    assert output_entry[3] == (12,)


def test_trace_input_output_vector():
    """trace_input and trace_output record vector (n_elems=8) values."""
    # Load 8 values into addrs 10-17 (4 cycles, 2 load slots each)
    # Load 8 values into addrs 20-27
    # valu add 30 = 10 + 20, then trace
    program = [
        {"load": [("const", 10, 1), ("const", 11, 2)]},
        {"load": [("const", 12, 3), ("const", 13, 4)]},
        {"load": [("const", 14, 5), ("const", 15, 6)]},
        {"load": [("const", 16, 7), ("const", 17, 8)]},
        {"load": [("const", 20, 10), ("const", 21, 20)]},
        {"load": [("const", 22, 30), ("const", 23, 40)]},
        {"load": [("const", 24, 50), ("const", 25, 60)]},
        {"load": [("const", 26, 70), ("const", 27, 80)]},
        {
            "valu": [("+", 30, 10, 20)],
            "debug": [
                ("trace_input", "r0_g0_vec", 8, 10, 20),
                ("trace_output", "r0_g0_vec", 8, 30),
            ],
        },
        {"flow": [("halt",)]},
    ]
    machine = Machine(
        mem_dump=[],
        program=program,
        debug_info=DebugInfo(scratch_map={}),
    )
    machine.enable_debug = True
    machine.run()

    input_entry = next(e for e in machine.trace_io_log if e[1] == "input")
    assert input_entry[3] == (1, 2, 3, 4, 5, 6, 7, 8, 10, 20, 30, 40, 50, 60, 70, 80)

    output_entry = next(e for e in machine.trace_io_log if e[1] == "output")
    assert output_entry[3] == (11, 22, 33, 44, 55, 66, 77, 88)


def test_enable_debug_false_skips_trace_ops():
    """When enable_debug=False, trace ops are skipped and log stays empty."""
    program = [
        {"load": [("const", 1, 5), ("const", 2, 7)]},
        {
            "alu": [("+", 3, 1, 2)],
            "debug": [
                ("trace_input", "r0_g0_test", 1, 1, 2),
                ("trace_output", "r0_g0_test", 1, 3),
            ],
        },
        {"flow": [("halt",)]},
    ]
    machine = Machine(
        mem_dump=[],
        program=program,
        debug_info=DebugInfo(scratch_map={}),
    )
    machine.enable_debug = False
    machine.run()

    assert machine.trace_io_log == []


def test_trace_id_tuple():
    """trace_id can be a tuple for cross-run correlation."""
    program = [
        {"load": [("const", 1, 42)]},
        {
            "flow": [("add_imm", 2, 0, 0)],  # no-op to advance cycle
            "debug": [
                ("trace_output", (1, 0, "hashed_val"), 1, 1),
            ],
        },
        {"flow": [("halt",)]},
    ]
    machine = Machine(
        mem_dump=[],
        program=program,
        debug_info=DebugInfo(scratch_map={}),
    )
    machine.enable_debug = True
    machine.run()

    assert len(machine.trace_io_log) == 1
    assert machine.trace_io_log[0][2] == (1, 0, "hashed_val")
    assert machine.trace_io_log[0][3] == (42,)
