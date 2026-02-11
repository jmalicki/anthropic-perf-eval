#!/usr/bin/env python3
"""Trace VReg producers to understand dependency chains.

Usage:
    uv run scripts/trace_vreg_producers.py <scheduler_output.json> <vreg_id>
"""

import json
import sys
from typing import Dict, List, Optional


def find_vreg_producer(ssa_ops: List[Dict], vreg_id: int, width: str = "vector") -> Optional[Dict]:
    """Find the op that produces a given VReg (by ID and width)."""
    # VReg dest format: "v{id}_{width}" where width is "v" for vector, "s" for scalar
    expected_dest = f"v{vreg_id}_{'v' if width == 'vector' else 's'}"
    for op in ssa_ops:
        if op.get("dest") == expected_dest:
            return op
    return None


def trace_vreg_chain(ssa_ops: List[Dict], vreg_id: int, width: str = "vector", depth: int = 0, max_depth: int = 10) -> List[Dict]:
    """Trace back the chain of VReg producers."""
    if depth > max_depth:
        return []
    
    producer = find_vreg_producer(ssa_ops, vreg_id, width)
    if not producer:
        return []
    
    chain = [producer]
    
    # Trace sources
    srcs = producer.get("srcs", [])
    for src in srcs:
        if src.get("VReg"):
            src_vreg_info = src["VReg"]
            src_vreg_id = src_vreg_info.get("id")
            src_width = src_vreg_info.get("width", "scalar")
            
            sub_chain = trace_vreg_chain(ssa_ops, src_vreg_id, src_width, depth + 1, max_depth)
            chain.extend(sub_chain)
    
    return chain


def main():
    if len(sys.argv) < 3:
        print("Usage: uv run scripts/trace_vreg_producers.py <scheduler_output.json> <vreg_id> [width] [max_depth]")
        print("Example: uv run scripts/trace_vreg_producers.py sched_2r_8b_d3_inspect.json 103 vector")
        sys.exit(1)
    
    json_path = sys.argv[1]
    vreg_id = int(sys.argv[2])
    width = sys.argv[3] if len(sys.argv) > 3 else "vector"
    max_depth = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    
    with open(json_path, "r") as f:
        data = json.load(f)
    
    ssa_ops = data.get("ssa_ops")
    if not ssa_ops:
        print(f"ERROR: {json_path} does not contain 'ssa_ops' field.")
        sys.exit(1)
    
    print(f"Tracing VReg {vreg_id} ({width}):\n")
    
    chain = trace_vreg_chain(ssa_ops, vreg_id, width, max_depth=max_depth)
    
    if not chain:
        print(f"No producer found for {vreg_id}")
        sys.exit(1)
    
    print("Producer chain (most recent first):")
    seen = set()
    for op in chain:
        op_id = op["id"]
        if op_id in seen:
            continue
        seen.add(op_id)
        
        source_info = op.get("source_info", {})
        round_num = source_info.get("round")
        phase = source_info.get("phase", "unknown")
        
        print(f"  Op {op_id}: {op.get('opcode', 'unknown')} -> {op.get('dest', 'none')}")
        print(f"    Round: {round_num}, Phase: {phase}, Cycle: {op.get('cycle', -1)}")
        print()


if __name__ == "__main__":
    main()
