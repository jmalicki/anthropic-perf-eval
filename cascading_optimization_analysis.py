#!/usr/bin/env python3
"""
Cascading Optimization Analysis: Analyze effects of manually shifting VALU ops.

This script analyzes what happens when VALU ops are shifted to ALU:
- Which VALU slots are freed
- Which ops can move earlier to use freed slots
- How consumers can start earlier
- Overall makespan reduction potential
"""

import json
import sys
from collections import defaultdict
from typing import List, Dict, Tuple, Set


def load_schedule(json_path: str) -> Dict:
    """Load schedule from JSON file."""
    with open(json_path) as f:
        return json.load(f)


def find_op(ops: List[Dict], op_id: int) -> Dict:
    """Find an op by ID."""
    return next((op for op in ops if op.get('id') == op_id), None)


def compute_asap(op: Dict, ops: List[Dict]) -> int:
    """Compute ASAP (earliest possible cycle) for an op."""
    deps = op.get('dep_ids', [])
    if not deps:
        return 0
    
    dep_cycles = []
    for dep_id in deps:
        dep_op = find_op(ops, dep_id)
        if dep_op:
            dep_cycles.append(dep_op.get('cycle', -1))
    
    return max(dep_cycles) + 1 if dep_cycles else 0


def compute_alap(op: Dict, ops: List[Dict]) -> int:
    """Compute ALAP (latest possible cycle) for an op."""
    op_id = op.get('id')
    consumers = [c for c in ops if op_id in c.get('dep_ids', [])]
    if not consumers:
        return 999  # No consumers, can be scheduled late
    
    consumer_cycles = [c.get('cycle', 999) for c in consumers]
    return min(consumer_cycles) - 1


def analyze_shift_impact(op_id: int, ops: List[Dict]) -> Dict:
    """Analyze the impact of shifting a specific VALU op."""
    op = find_op(ops, op_id)
    if not op:
        return None
    
    cycle = op.get('cycle')
    asap = compute_asap(op, ops)
    alap = compute_alap(op, ops)
    
    # Find consumers
    consumers = [c for c in ops if op_id in c.get('dep_ids', [])]
    consumer_cycles = [c.get('cycle', 999) for c in consumers]
    min_consumer_cycle = min(consumer_cycles) if consumer_cycles else 999
    
    # Consumer speedup if shifted
    consumer_speedup = min_consumer_cycle - (asap + 1) if consumer_cycles else 0
    total_consumer_speedup = consumer_speedup * len(consumers)
    
    # Find ops that could use the freed VALU slot
    candidates = []
    for other_op in ops:
        if (other_op.get('engine') == 'valu' 
            and other_op.get('cycle', 999) > cycle
            and other_op.get('opcode') not in ['vbroadcast', 'multiply_add']):
            other_asap = compute_asap(other_op, ops)
            if other_asap <= cycle:
                speedup = other_op.get('cycle') - cycle
                candidates.append((other_op.get('id'), other_op.get('cycle'), speedup))
    
    slot_reuse = len(candidates)
    op_speedup = sum(speedup for _, _, speedup in candidates)
    
    return {
        'op_id': op_id,
        'cycle': cycle,
        'asap': asap,
        'alap': alap,
        'num_consumers': len(consumers),
        'consumer_speedup': consumer_speedup,
        'total_consumer_speedup': total_consumer_speedup,
        'slot_reuse': slot_reuse,
        'op_speedup': op_speedup,
        'total_impact': total_consumer_speedup + op_speedup
    }


def analyze_cascading_effects(json_path: str):
    """Perform full cascading optimization analysis."""
    data = load_schedule(json_path)
    ops = data.get('ops', [])
    total_cycles = data.get('total_cycles', 0)
    
    print("=" * 80)
    print("Cascading Optimization Analysis")
    print("=" * 80)
    print(f"\nCurrent makespan: {total_cycles} cycles\n")
    
    # Find all early shiftable VALU ops
    early_shiftable = []
    for op in ops:
        if (op.get('engine') == 'valu' 
            and op.get('opcode') not in ['vbroadcast', 'multiply_add']
            and op.get('cycle', 999) <= 10):
            impact = analyze_shift_impact(op.get('id'), ops)
            if impact:
                early_shiftable.append(impact)
    
    # Sort by total impact
    early_shiftable.sort(key=lambda x: x['total_impact'], reverse=True)
    
    print("Shiftable VALU ops (cycles 0-10) ranked by impact:")
    print("\n" + "Rank".ljust(6) + "Op ID".ljust(8) + "Cycle".ljust(8) + 
          "Consumers".ljust(12) + "Consumer Speedup".ljust(18) + 
          "Slot Reuse".ljust(12) + "Op Speedup".ljust(12) + "Total Impact")
    print("-" * 100)
    
    for i, impact in enumerate(early_shiftable[:15], 1):
        print(f"{i:<6}{impact['op_id']:<8}{impact['cycle']:<8}"
              f"{impact['num_consumers']:<12}{impact['consumer_speedup']:<18}"
              f"{impact['slot_reuse']:<12}{impact['op_speedup']:<12}{impact['total_impact']}")
    
    # Analyze specific scenarios
    print("\n" + "=" * 80)
    print("Scenario Analysis")
    print("=" * 80)
    
    # Scenario 1: Shift Op 244
    impact_244 = next((x for x in early_shiftable if x['op_id'] == 244), None)
    if impact_244:
        print("\nScenario 1: Shift Op 244")
        print(f"  - Frees VALU slot at cycle {impact_244['cycle']}")
        print(f"  - {impact_244['slot_reuse']} VALU ops can move to cycle {impact_244['cycle']}")
        print(f"  - {impact_244['num_consumers']} consumers move {impact_244['consumer_speedup']} cycles earlier")
        print(f"  - Total impact: {impact_244['total_impact']} cycles")
    
    # Scenario 2: Shift Op 244 and 287
    impact_287 = next((x for x in early_shiftable if x['op_id'] == 287), None)
    if impact_244 and impact_287:
        print("\nScenario 2: Shift Op 244 AND Op 287")
        print(f"  - Frees VALU slots at cycles {impact_244['cycle']} and {impact_287['cycle']}")
        print(f"  - {impact_244['slot_reuse'] + impact_287['slot_reuse']} VALU ops can move earlier")
        print(f"  - {impact_244['num_consumers']} consumers move {impact_287['consumer_speedup']} cycles earlier")
        print(f"  - Total impact: {impact_244['total_impact'] + impact_287['total_impact']} cycles")
    
    # Scenario 3: Shift top 5 ops
    top_5 = early_shiftable[:5]
    total_freed_slots = len(top_5)
    total_op_speedup = sum(x['op_speedup'] for x in top_5)
    total_consumer_speedup = sum(x['total_consumer_speedup'] for x in top_5)
    
    print("\nScenario 3: Shift top 5 ops by impact")
    print(f"  - Ops: {[x['op_id'] for x in top_5]}")
    print(f"  - Freed VALU slots: {total_freed_slots}")
    print(f"  - Total op speedup: {total_op_speedup} cycles")
    print(f"  - Total consumer speedup: {total_consumer_speedup} cycles")
    print(f"  - Total impact: {sum(x['total_impact'] for x in top_5)} cycles")
    
    # Optimal sequence
    print("\n" + "=" * 80)
    print("Optimal Shift Sequence")
    print("=" * 80)
    print("\nRecommended sequence (maximizing cascading effects):")
    
    sequence = []
    freed_slots_by_cycle = defaultdict(int)
    
    for impact in sorted(early_shiftable, key=lambda x: (x['cycle'], -x['total_impact'])):
        cycle = impact['cycle']
        
        # Check if slot is already freed
        if freed_slots_by_cycle[cycle] > 0:
            freed_slots_by_cycle[cycle] -= 1
        else:
            freed_slots_by_cycle[cycle] += 1
        
        sequence.append(impact['op_id'])
        
        if len(sequence) <= 10:
            print(f"\n  Step {len(sequence)}: Shift Op {impact['op_id']} (cycle {cycle})")
            print(f"    - Frees VALU slot at cycle {cycle}")
            print(f"    - {impact['slot_reuse']} ops can use this slot")
            print(f"    - Consumer speedup: {impact['total_consumer_speedup']} cycles")
            print(f"    - Total impact: {impact['total_impact']} cycles")
    
    print(f"\n\nFull sequence ({len(sequence)} ops): {sequence}")
    print(f"Estimated total impact: {sum(x['total_impact'] for x in early_shiftable)} cycles")
    print(f"Estimated makespan reduction: ~{max(x['consumer_speedup'] for x in early_shiftable[:5])} cycles")


if __name__ == '__main__':
    json_path = sys.argv[1] if len(sys.argv) > 1 else 'scheduler_output_fixed.json'
    analyze_cascading_effects(json_path)
