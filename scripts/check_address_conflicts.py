#!/usr/bin/env python3
"""Check for address conflicts in scheduler output.

Verifies that no two VRegs with overlapping live intervals share addresses.

Usage:
    uv run python scripts/check_address_conflicts.py --scheduler-json scheduler_output.json
"""

import sys
import os

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir in sys.path:
    sys.path.remove(_script_dir)

import json
from collections import defaultdict


def check_address_conflicts(sched_data):
    """Check for address conflicts between VRegs.
    
    Excludes valid same-cycle reuse cases where:
    - VReg A ends at cycle C (last read at C)
    - VReg B starts at cycle C (written at C)
    This is valid because VLIW semantics: reads happen before writes.
    """
    vregs = sched_data.get('vregs', [])
    conflicts = []
    
    # Group VRegs by address
    addr_to_vregs = defaultdict(list)
    for vreg in vregs:
        addr = vreg.get('address')
        if addr is not None:
            width = vreg.get('width', 'scalar')
            if width == 'vector':
                # Vector spans 8 addresses
                for i in range(8):
                    addr_to_vregs[addr + i].append((vreg, i))
            else:
                addr_to_vregs[addr].append((vreg, 0))
    
    # Check each address for conflicts
    for addr, vreg_list in addr_to_vregs.items():
        if len(vreg_list) <= 1:
            continue
        
        # Check if any two VRegs have overlapping intervals
        for i, (vreg1, offset1) in enumerate(vreg_list):
            for j, (vreg2, offset2) in enumerate(vreg_list[i+1:], i+1):
                start1 = vreg1.get('interval_start')
                end1 = vreg1.get('interval_end')
                start2 = vreg2.get('interval_start')
                end2 = vreg2.get('interval_end')
                
                if start1 is None or end1 is None or start2 is None or end2 is None:
                    continue
                
                # Check if intervals overlap (not just adjacent)
                # Adjacent intervals [D1, C] and [C, E2] are valid same-cycle reuse
                if end1 < start2 or end2 < start1:
                    # No overlap - valid
                    continue
                
                # Check for valid same-cycle reuse:
                # Case 1: VReg1 ends at C (read), VReg2 starts at C (write)
                # Case 2: VReg2 ends at C (read), VReg1 starts at C (write)
                is_same_cycle_reuse = (
                    (end1 == start2 and end1 >= start1) or  # VReg1 read at C, VReg2 written at C
                    (end2 == start1 and end2 >= start2)      # VReg2 read at C, VReg1 written at C
                )
                
                if is_same_cycle_reuse:
                    # Valid same-cycle reuse - skip
                    continue
                
                # Real conflict: intervals overlap beyond boundary
                overlap_start = max(start1, start2)
                overlap_end = min(end1, end2)
                
                conflicts.append({
                    'address': addr,
                    'vreg1': {
                        'id': vreg1.get('id'),
                        'width': vreg1.get('width'),
                        'interval': [start1, end1],
                        'offset': offset1,
                    },
                    'vreg2': {
                        'id': vreg2.get('id'),
                        'width': vreg2.get('width'),
                        'interval': [start2, end2],
                        'offset': offset2,
                    },
                    'overlap': [overlap_start, overlap_end],
                })
    
    return conflicts


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler-json", required=True, help="Scheduler output JSON")
    args = parser.parse_args()

    print(f"Loading scheduler JSON from {args.scheduler_json}...")
    with open(args.scheduler_json) as f:
        sched_data = json.load(f)
    
    print(f"Checking for address conflicts...")
    conflicts = check_address_conflicts(sched_data)
    
    if not conflicts:
        print("✓ No address conflicts found!")
        return
    
    print(f"\n⚠️  Found {len(conflicts)} address conflict(s):\n")
    
    for i, conflict in enumerate(conflicts, 1):
        print(f"Conflict {i}:")
        print(f"  Address: {conflict['address']}")
        print(f"  VReg {conflict['vreg1']['id']} ({conflict['vreg1']['width']}):")
        print(f"    Interval: [{conflict['vreg1']['interval'][0]}, {conflict['vreg1']['interval'][1]}]")
        print(f"    Offset: {conflict['vreg1']['offset']}")
        print(f"  VReg {conflict['vreg2']['id']} ({conflict['vreg2']['width']}):")
        print(f"    Interval: [{conflict['vreg2']['interval'][0]}, {conflict['vreg2']['interval'][1]}]")
        print(f"    Offset: {conflict['vreg2']['offset']}")
        print(f"  Overlap: [{conflict['overlap'][0]}, {conflict['overlap'][1]}]")
        print()
    
    # Load SSA ops to get more context
    ssa_ops = sched_data.get('ssa_ops', [])
    if ssa_ops:
        print("\nDetails about conflicting VRegs:")
        for conflict in conflicts[:5]:  # First 5
            for vreg_id in [conflict['vreg1']['id'], conflict['vreg2']['id']]:
                producer = [o for o in ssa_ops if o.get('dest') == f"v{vreg_id}_{conflict['vreg1']['width'][0] if vreg_id == conflict['vreg1']['id'] else conflict['vreg2']['width'][0]}"]
                if producer:
                    p = producer[0]
                    print(f"\n  VReg {vreg_id}:")
                    print(f"    Produced by Op {p['id']} ({p.get('opcode')}) at cycle {p.get('cycle')}")
                    print(f"    Round: {p.get('source_info', {}).get('round')}, Phase: {p.get('source_info', {}).get('phase')}")


if __name__ == "__main__":
    main()
