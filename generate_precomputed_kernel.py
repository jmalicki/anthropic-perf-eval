#!/usr/bin/env python3
"""
Generate a precomputed kernel file with optimized schedule.

This script:
1. Builds the kernel using InterleavedUnrolledKernelBuilder
2. Runs CP-SAT optimization to find the best schedule
3. Outputs solver/precomputed_kernel.py with the precomputed instructions

Usage:
    python generate_precomputed_kernel.py [--full-solve-time SECONDS]
"""

import argparse
import sys
from datetime import datetime


def generate_precomputed_kernel(
    forest_height: int = 10,
    n_nodes: int = 2047,
    batch_size: int = 256,
    rounds: int = 16,
    full_solve_time: float = 3600.0,
    require_optimal: bool = True,
    enable_valu_shift: bool = True,
    valu_shift_time: float = None,
    use_greedy: bool = False,
    enable_alu_hash_parallel: bool = False,
    alu_hash_fraction: float = 0.5,
    use_multipass_scheduler: bool = False,
    enable_broadcast_lookahead: bool = False,
    enable_chunked_gather: bool = False,
    chunked_gather_min_depth: int = 4,
    chunked_gather_max_depth: int = 10,
    chunk_size: int = 8,
    output_file: str = "solver/precomputed_kernel.py"
):
    """Generate the precomputed kernel file.

    Args:
        use_greedy: Use fast greedy scheduler instead of CP-SAT.
                    Greedy runs in seconds but may produce ~5-10% more cycles.
    """

    print(f"=" * 60)
    print(f"Generating precomputed kernel")
    print(f"=" * 60)
    print(f"Parameters:")
    print(f"  forest_height: {forest_height}")
    print(f"  n_nodes: {n_nodes}")
    print(f"  batch_size: {batch_size}")
    print(f"  rounds: {rounds}")
    if use_greedy:
        print(f"  scheduler: GREEDY (fast, near-optimal)")
    else:
        print(f"  scheduler: CP-SAT (slow, optimal)")
        print(f"  full_solve_time: {full_solve_time}s")
        print(f"  require_optimal: {require_optimal}")
        print(f"  enable_valu_shift: {enable_valu_shift}")
        print(f"  valu_shift_time: {valu_shift_time if valu_shift_time else 'same as full_solve_time'}s")
    print(f"  enable_alu_hash_parallel: {enable_alu_hash_parallel}")
    if enable_alu_hash_parallel:
        print(f"  alu_hash_fraction: {alu_hash_fraction}")
    print(f"  enable_chunked_gather: {enable_chunked_gather}")
    if enable_chunked_gather:
        print(f"  chunked_gather_min_depth: {chunked_gather_min_depth}")
        print(f"  chunked_gather_max_depth: {chunked_gather_max_depth}")
        print(f"  chunk_size: {chunk_size}")
    print()

    # Import here to avoid issues if dependencies aren't installed
    from solver.interleaved_unrolled_kernel import InterleavedUnrolledKernelBuilder

    # Build the kernel
    print("Building kernel...")
    builder = InterleavedUnrolledKernelBuilder()
    builder.build_kernel(forest_height, n_nodes, batch_size, rounds,
                        full_solve_time=full_solve_time,
                        require_optimal=require_optimal,
                        enable_valu_shift=enable_valu_shift,
                        valu_shift_time=valu_shift_time,
                        use_greedy=use_greedy,
                        enable_alu_hash_parallel=enable_alu_hash_parallel,
                        alu_hash_fraction=alu_hash_fraction,
                        use_multipass_scheduler=use_multipass_scheduler,
                        enable_broadcast_lookahead=enable_broadcast_lookahead,
                        enable_chunked_gather=enable_chunked_gather,
                        chunked_gather_min_depth=chunked_gather_min_depth,
                        chunked_gather_max_depth=chunked_gather_max_depth,
                        chunk_size=chunk_size)

    # Get the results
    instrs = builder.instrs
    scratch_ptr = builder.scratch_ptr
    scratch_debug = builder.scratch_debug

    # Calculate cycles (subtract 2 for pause instructions)
    num_cycles = len(instrs) - 2 if len(instrs) > 2 else len(instrs)

    print()
    print(f"Results:")
    print(f"  Instructions: {len(instrs)}")
    print(f"  Cycles: {num_cycles}")
    print(f"  Scratch usage: {scratch_ptr}")
    print(f"  Speedup: {147734 / num_cycles:.1f}x")
    print()

    # Generate the output file
    print(f"Writing {output_file}...")

    # Compress instructions using zlib + base64 for compact storage
    import json
    import zlib
    import base64

    # Convert instructions to JSON-serializable format
    def convert_for_json(obj):
        if isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_for_json(x) for x in obj]
        else:
            return obj

    instrs_json = json.dumps(convert_for_json(instrs), separators=(',', ':'))
    compressed = zlib.compress(instrs_json.encode('utf-8'), level=9)
    encoded = base64.b64encode(compressed).decode('ascii')

    # Split into 100-char lines for readability
    encoded_lines = [encoded[i:i+100] for i in range(0, len(encoded), 100)]

    print(f"  Uncompressed: {len(instrs_json)} bytes")
    print(f"  Compressed: {len(compressed)} bytes ({100*len(compressed)//len(instrs_json)}%)")

    with open(output_file, 'w') as f:
        f.write(f'''"""
Precomputed optimized kernel.

Generated by generate_precomputed_kernel.py
Parameters: forest_height={forest_height}, n_nodes={n_nodes}, batch_size={batch_size}, rounds={rounds}
Results: cycles={num_cycles}, speedup={147734 / num_cycles:.1f}x
"""

from problem import DebugInfo


def build_kernel(self, forest_height, n_nodes, batch_size, rounds):
    """Build the kernel using precomputed instructions."""
    import json
    import zlib
    import base64

    # Assert parameters match
    assert forest_height == {forest_height}, f"Expected forest_height={forest_height}, got {{forest_height}}"
    assert n_nodes == {n_nodes}, f"Expected n_nodes={n_nodes}, got {{n_nodes}}"
    assert batch_size == {batch_size}, f"Expected batch_size={batch_size}, got {{batch_size}}"
    assert rounds == {rounds}, f"Expected rounds={rounds}, got {{rounds}}"

    # Precomputed scratch mapping
    self.scratch_ptr = {scratch_ptr}
    self.scratch_debug = {{
''')
        # Write scratch_debug
        for addr, (name, length) in sorted(scratch_debug.items()):
            f.write(f"        {addr}: ({repr(name)}, {length}),\n")
        f.write(f'''    }}

    # Compressed instructions ({len(instrs)} total, {num_cycles} cycles)
    _COMPRESSED = (
''')
        for line in encoded_lines:
            f.write(f'        "{line}"\n')
        f.write('''    )

    # Decompress and parse instructions
    compressed = base64.b64decode(_COMPRESSED)
    json_data = zlib.decompress(compressed).decode('utf-8')
    raw = json.loads(json_data)

    # Convert lists back to tuples for slots
    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            if obj and isinstance(obj[0], str) and obj[0] in (
                'const', 'load', 'vload', 'vstore', 'pause',
                '+', '-', '*', '/', '%', '&', '|', '^', '<<', '>>',
                '==', '!=', '<', '<=', '>', '>=',
                'vbroadcast', 'vselect', 'multiply_add'
            ):
                return tuple(convert(x) for x in obj)
            return [convert(x) for x in obj]
        else:
            return obj

    self.instrs = convert(raw)
    return self


def debug_info(self):
    return DebugInfo(scratch_map=self.scratch_debug)
''')

    print(f"Done! Generated {output_file}")
    print()
    print(f"To use the precomputed kernel:")
    print(f"  from solver.precomputed_kernel import build_kernel")


def main():
    parser = argparse.ArgumentParser(description="Generate precomputed kernel")
    parser.add_argument("--forest-height", type=int, default=10)
    parser.add_argument("--n-nodes", type=int, default=2047)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--rounds", type=int, default=16)
    parser.add_argument("--full-solve-time", type=float, default=3600.0,
                        help="Time limit for CP-SAT full solve in seconds (default: 3600)")
    parser.add_argument("--require-optimal", action="store_true", default=True,
                        help="Require OPTIMAL status from CP-SAT (default: True)")
    parser.add_argument("--no-require-optimal", action="store_false", dest="require_optimal",
                        help="Allow FEASIBLE status from CP-SAT")
    parser.add_argument("--enable-valu-shift", action="store_true", default=True,
                        help="Enable VALU→ALU shifting optimization (default: True)")
    parser.add_argument("--no-valu-shift", action="store_false", dest="enable_valu_shift",
                        help="Disable VALU→ALU shifting optimization")
    parser.add_argument("--valu-shift-time", type=float, default=None,
                        help="Time per VALU→ALU shift iteration (default: same as full-solve-time)")
    parser.add_argument("--use-greedy", action="store_true",
                        help="Use fast greedy scheduler instead of CP-SAT (seconds vs hours)")
    parser.add_argument("--enable-alu-hash-parallel", action="store_true",
                        help="Enable parallel hash using both ALU and VALU (fills idle ALU slots)")
    parser.add_argument("--alu-hash-fraction", type=float, default=0.5,
                        help="Fraction of groups to process with ALU (default: 0.5)")
    parser.add_argument("--use-multipass-scheduler", action="store_true",
                        help="Try multiple scheduling strategies and pick the best")
    parser.add_argument("--enable-broadcast-lookahead", action="store_true",
                        help="EXPERIMENTAL: Proactively eliminate broadcasts based on predicted VALU pressure")
    parser.add_argument("--enable-chunked-gather", action="store_true",
                        help="Use chunked vload instead of scattered loads for tree access")
    parser.add_argument("--chunked-gather-min-depth", type=int, default=4,
                        help="Minimum tree depth to use chunked gather (default: 4, since 0-3 use VALU)")
    parser.add_argument("--chunked-gather-max-depth", type=int, default=10,
                        help="Maximum tree depth to use chunked gather (default: 10)")
    parser.add_argument("--chunk-size", type=int, default=8, choices=[8, 16, 32],
                        help="Chunk size for chunked gather (default: 8)")
    parser.add_argument("--output", type=str, default="solver/precomputed_kernel.py",
                        help="Output file path")

    args = parser.parse_args()

    generate_precomputed_kernel(
        forest_height=args.forest_height,
        n_nodes=args.n_nodes,
        batch_size=args.batch_size,
        rounds=args.rounds,
        full_solve_time=args.full_solve_time,
        require_optimal=args.require_optimal,
        enable_valu_shift=args.enable_valu_shift,
        valu_shift_time=args.valu_shift_time,
        use_greedy=args.use_greedy,
        enable_alu_hash_parallel=args.enable_alu_hash_parallel,
        alu_hash_fraction=args.alu_hash_fraction,
        use_multipass_scheduler=args.use_multipass_scheduler,
        enable_broadcast_lookahead=args.enable_broadcast_lookahead,
        enable_chunked_gather=args.enable_chunked_gather,
        chunked_gather_min_depth=args.chunked_gather_min_depth,
        chunked_gather_max_depth=args.chunked_gather_max_depth,
        chunk_size=args.chunk_size,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
