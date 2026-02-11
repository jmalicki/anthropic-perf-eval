#!/usr/bin/env python3
"""Inspect SSA dependencies to check if Round 2 hash ops depend on Round 1 hash VReg.

This script answers Phase 1 of the root cause investigation:
- Do Round 2 hash ops have Round 1 hash VReg as a source operand?

Usage:
    uv run scripts/inspect_ssa_dependencies.py <scheduler_output.json>

The JSON file must be generated with --output-json and include ssa_ops.
"""

import json
import sys
from typing import Dict, List, Optional, Set


def find_round_hash_ops(ssa_ops: List[Dict], round_num: int) -> List[Dict]:
    """Find all hash-phase ops for a given round."""
    hash_ops = []
    for op in ssa_ops:
        source_info = op.get("source_info")
        if not source_info:
            continue
        if source_info.get("round") != round_num:
            continue
        phase = source_info.get("phase", "")
        if phase == "hash":
            hash_ops.append(op)
    return hash_ops


def find_round_final_hash_vreg(ssa_ops: List[Dict], round_num: int) -> Optional[str]:
    """Find the final hash VReg produced in a round (last hash op's dest)."""
    hash_ops = find_round_hash_ops(ssa_ops, round_num)
    if not hash_ops:
        return None
    
    # Sort by op id to get last hash op
    hash_ops.sort(key=lambda op: op["id"])
    last_hash_op = hash_ops[-1]
    
    return last_hash_op.get("dest")


def get_vreg_from_operand(operand: Dict) -> Optional[str]:
    """Extract VReg string from an operand.
    
    Returns format matching SsaOpDebugOut dest names:
    - "v{id}_v" for vector width
    - "v{id}_s" for scalar width
    """
    if operand.get("VReg"):
        vreg_info = operand["VReg"]
        width = vreg_info.get("width", "scalar")
        vreg_id = vreg_info.get("id")
        if vreg_id is not None:
            suffix = "v" if width == "vector" else "s"
            return f"v{vreg_id}_{suffix}"
    return None


def check_round2_depends_on_round1(ssa_ops: List[Dict]) -> Dict:
    """Check if Round 2 hash ops depend on Round 1 hash VReg."""
    # Find Round 1 final hash VReg
    round1_final_vreg = find_round_final_hash_vreg(ssa_ops, 0)  # Round 0 = Round 1
    if not round1_final_vreg:
        return {
            "round1_final_vreg": None,
            "round2_hash_ops": [],
            "depends": False,
            "reason": "No Round 1 hash ops found"
        }
    
    # Find Round 2 hash ops
    round2_hash_ops = find_round_hash_ops(ssa_ops, 1)  # Round 1 = Round 2
    if not round2_hash_ops:
        return {
            "round1_final_vreg": round1_final_vreg,
            "round2_hash_ops": [],
            "depends": False,
            "reason": "No Round 2 hash ops found"
        }
    
    # Check if any Round 2 hash op has Round 1 final hash VReg as source
    depends = False
    dependent_ops = []
    
    for op in round2_hash_ops:
        srcs = op.get("srcs", [])
        for src in srcs:
            src_vreg = get_vreg_from_operand(src)
            if src_vreg == round1_final_vreg:
                depends = True
                dependent_ops.append({
                    "op_id": op["id"],
                    "opcode": op.get("opcode", "unknown"),
                    "cycle": op.get("cycle", -1),
                    "src_vreg": src_vreg
                })
                break
    
    # Analyze what VRegs Round 2 hash ops actually read
    round2_hash_srcs = []
    for op in round2_hash_ops:
        srcs = op.get("srcs", [])
        op_srcs = []
        for src in srcs:
            src_vreg = get_vreg_from_operand(src)
            if src_vreg:
                op_srcs.append(src_vreg)
        round2_hash_srcs.append({
            "op_id": op["id"],
            "opcode": op.get("opcode", "unknown"),
            "src_vregs": op_srcs
        })
    
    return {
        "round1_final_vreg": round1_final_vreg,
        "round2_hash_ops_count": len(round2_hash_ops),
        "round2_hash_op_ids": [op["id"] for op in round2_hash_ops],
        "depends": depends,
        "dependent_ops": dependent_ops,
        "round2_hash_srcs": round2_hash_srcs,
        "reason": "Round 2 hash ops DO depend on Round 1 hash VReg" if depends else "Round 2 hash ops DO NOT depend on Round 1 hash VReg"
    }


def find_round_branch_ops(ssa_ops: List[Dict], round_num: int) -> List[Dict]:
    """Find all branch-phase ops for a given round."""
    branch_ops = []
    for op in ssa_ops:
        source_info = op.get("source_info")
        if not source_info:
            continue
        if source_info.get("round") != round_num:
            continue
        phase = source_info.get("phase", "")
        if phase == "branch":
            branch_ops.append(op)
    return branch_ops


def check_round2_branch_reads(ssa_ops: List[Dict], round1_final_hash_vreg: Optional[str]) -> Dict:
    """Check what VRegs Round 2 branch ops read."""
    round2_branch_ops = find_round_branch_ops(ssa_ops, 1)  # Round 1 = Round 2
    if not round2_branch_ops:
        return {
            "round2_branch_ops_count": 0,
            "round2_branch_op_ids": [],
            "reads_round1_hash": False,
            "branch_srcs": []
        }
    
    reads_round1_hash = False
    branch_srcs = []
    
    for op in round2_branch_ops:
        srcs = op.get("srcs", [])
        op_srcs = []
        for src in srcs:
            src_vreg = get_vreg_from_operand(src)
            if src_vreg:
                op_srcs.append(src_vreg)
                if src_vreg == round1_final_hash_vreg:
                    reads_round1_hash = True
        branch_srcs.append({
            "op_id": op["id"],
            "opcode": op.get("opcode", "unknown"),
            "cycle": op.get("cycle", -1),
            "src_vregs": op_srcs
        })
    
    return {
        "round2_branch_ops_count": len(round2_branch_ops),
        "round2_branch_op_ids": [op["id"] for op in round2_branch_ops],
        "reads_round1_hash": reads_round1_hash,
        "branch_srcs": branch_srcs
    }


def main():
    if len(sys.argv) != 2:
        print("Usage: uv run scripts/inspect_ssa_dependencies.py <scheduler_output.json>")
        sys.exit(1)
    
    json_path = sys.argv[1]
    
    with open(json_path, "r") as f:
        data = json.load(f)
    
    ssa_ops = data.get("ssa_ops")
    if not ssa_ops:
        print(f"ERROR: {json_path} does not contain 'ssa_ops' field.")
        print("Generate the kernel with --output-json to include SSA ops.")
        sys.exit(1)
    
    print(f"Loaded {len(ssa_ops)} SSA ops from {json_path}\n")
    
    result = check_round2_depends_on_round1(ssa_ops)
    
    print("=" * 70)
    print("SSA Dependency Analysis: Round 2 Hash → Round 1 Hash")
    print("=" * 70)
    print()
    print(f"Round 1 final hash VReg: {result['round1_final_vreg']}")
    print(f"Round 2 hash ops count: {result['round2_hash_ops_count']}")
    if result['round2_hash_op_ids']:
        print(f"Round 2 hash op IDs: {result['round2_hash_op_ids'][:10]}{'...' if len(result['round2_hash_op_ids']) > 10 else ''}")
    print()
    print(f"RESULT: {result['reason']}")
    print()
    
    if result['depends']:
        print("Dependent ops:")
        for dep_op in result['dependent_ops']:
            print(f"  Op {dep_op['op_id']} ({dep_op['opcode']}) at cycle {dep_op['cycle']} reads {dep_op['src_vreg']}")
    else:
        print("No Round 2 hash ops read Round 1 final hash VReg.")
        print()
        print("Round 2 hash ops read these VRegs:")
        for op_info in result['round2_hash_srcs']:
            print(f"  Op {op_info['op_id']} ({op_info['opcode']}): {', '.join(op_info['src_vregs']) if op_info['src_vregs'] else 'no VReg sources'}")
        print()
        print("This suggests:")
        print("  - Round 2 hash ops may read a different VReg (intermediate value?)")
        print("  - Or the dependency chain is broken in emission")
        print("  - Or Round 2 hash ops read Round 1 hash via a different path")
    
    print()
    print("=" * 70)
    print()
    
    # Check Round 2 branch ops
    branch_result = check_round2_branch_reads(ssa_ops, result['round1_final_vreg'])
    print("Round 2 Branch Analysis:")
    print(f"  Round 2 branch ops count: {branch_result['round2_branch_ops_count']}")
    if branch_result['round2_branch_op_ids']:
        print(f"  Round 2 branch op IDs: {branch_result['round2_branch_op_ids'][:10]}{'...' if len(branch_result['round2_branch_op_ids']) > 10 else ''}")
    print(f"  Reads Round 1 hash directly: {branch_result['reads_round1_hash']}")
    if branch_result['branch_srcs']:
        print("  Round 2 branch ops read these VRegs:")
        for op_info in branch_result['branch_srcs'][:5]:  # Show first 5
            print(f"    Op {op_info['op_id']} ({op_info['opcode']}) at cycle {op_info['cycle']}: {', '.join(op_info['src_vregs']) if op_info['src_vregs'] else 'no VReg sources'}")
    
    print()
    print("=" * 70)
    
    # Return exit code: 0 if dependency exists (expected), 1 if not (unexpected)
    sys.exit(0 if result['depends'] else 1)


if __name__ == "__main__":
    main()
