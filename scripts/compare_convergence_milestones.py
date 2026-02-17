#!/usr/bin/env python3
"""Compare convergence milestones between two CP-SAT runs.

Extracts objective improvements from solver logs and compares timing.
"""

import argparse
import re
import json
from pathlib import Path
from typing import List, Tuple, Optional


def parse_objective_improvements(log_file: Path) -> List[Tuple[float, int]]:
    """Parse CP-SAT log to extract (time, objective_value) pairs for solution improvements.
    
    Returns list sorted by time.
    """
    improvements = []
    
    with open(log_file, 'r') as f:
        for line in f:
            # Match solution improvement lines: #N TIME best:OBJ
            match = re.search(r'#(\d+)\s+([\d.]+)s\s+best:(\d+)', line)
            if match:
                time_secs = float(match.group(2))
                objective = int(match.group(3))
                improvements.append((time_secs, objective))
    
    return improvements


def get_final_cycles(json_file: Optional[Path]) -> Optional[int]:
    """Extract final cycles from scheduler JSON."""
    if not json_file or not json_file.exists():
        return None
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data.get('stats', {}).get('cycles')
    except Exception:
        return None


def compute_milestones(improvements: List[Tuple[float, int]], final_cycles: Optional[int]) -> dict:
    """Compute convergence milestones from objective improvements.
    
    Since objective values are encoded, we report:
    - Time to first feasible solution
    - Time to each objective improvement (relative improvement %)
    - Time to optimal (last improvement)
    """
    if not improvements:
        return {}
    
    first_time, first_obj = improvements[0]
    last_time, last_obj = improvements[-1]
    obj_range = first_obj - last_obj  # Should be positive (decreasing)
    
    result = {
        'time_to_first_feasible': first_time,
        'time_to_optimal': last_time,
        'num_improvements': len(improvements),
        'first_objective': first_obj,
        'last_objective': last_obj,
        'objective_range': obj_range,
    }
    
    # Add times for each improvement milestone
    improvement_times = []
    for i, (time_secs, obj) in enumerate(improvements):
        if obj_range > 0:
            improvement_pct = ((first_obj - obj) / obj_range) * 100
        else:
            improvement_pct = 100.0 if i == len(improvements) - 1 else 0.0
        
        improvement_times.append({
            'solution_num': i + 1,
            'time_secs': time_secs,
            'objective': obj,
            'improvement_pct': improvement_pct,
        })
    
    result['improvements'] = improvement_times
    
    return result


def format_comparison(arm1_name: str, arm1_milestones: dict, 
                     arm2_name: str, arm2_milestones: dict):
    """Format a comparison table of convergence milestones."""
    print(f"\n{'='*80}")
    print("CONVERGENCE MILESTONES COMPARISON")
    print(f"{'='*80}\n")
    
    # Summary
    print("Summary:")
    print(f"  {arm1_name}:")
    print(f"    First feasible: {arm1_milestones.get('time_to_first_feasible', 0):.2f}s")
    print(f"    Optimal: {arm1_milestones.get('time_to_optimal', 0):.2f}s")
    print(f"    Improvements: {arm1_milestones.get('num_improvements', 0)}")
    
    print(f"  {arm2_name}:")
    print(f"    First feasible: {arm2_milestones.get('time_to_first_feasible', 0):.2f}s")
    print(f"    Optimal: {arm2_milestones.get('time_to_optimal', 0):.2f}s")
    print(f"    Improvements: {arm2_milestones.get('num_improvements', 0)}")
    
    # Speedup
    if arm1_milestones.get('time_to_optimal') and arm2_milestones.get('time_to_optimal'):
        speedup = arm1_milestones['time_to_optimal'] / arm2_milestones['time_to_optimal']
        print(f"\n  Speedup ({arm2_name} vs {arm1_name}): {speedup:.2f}x")
    
    # Detailed improvement timeline
    print(f"\n{'='*80}")
    print("OBJECTIVE IMPROVEMENT TIMELINE")
    print(f"{'='*80}\n")
    
    print(f"{'Solution':<10} {'Time (s)':<12} {'Objective':<15} {'Improvement %':<15}")
    print(f"{'':10} {'':12} {'':15} {'':15}")
    print(f"{'-'*52}")
    
    # Merge and compare improvements
    arm1_improvements = arm1_milestones.get('improvements', [])
    arm2_improvements = arm2_milestones.get('improvements', [])
    
    # Show first few improvements from each
    max_show = max(len(arm1_improvements), len(arm2_improvements), 10)
    
    for i in range(min(max_show, len(arm1_improvements))):
        imp = arm1_improvements[i]
        print(f"{arm1_name:>10} {imp['time_secs']:>11.2f} {imp['objective']:>14} {imp['improvement_pct']:>14.1f}%")
    
    if len(arm2_improvements) > 0:
        print()
        for i in range(min(max_show, len(arm2_improvements))):
            imp = arm2_improvements[i]
            print(f"{arm2_name:>10} {imp['time_secs']:>11.2f} {imp['objective']:>14} {imp['improvement_pct']:>14.1f}%")
    
    # Time to reach specific improvement percentages
    print(f"\n{'='*80}")
    print("TIME TO REACH IMPROVEMENT MILESTONES")
    print(f"{'='*80}\n")
    
    for pct in [25, 50, 75, 90, 95, 99, 100]:
        arm1_time = None
        arm2_time = None
        
        for imp in arm1_improvements:
            if imp['improvement_pct'] >= pct:
                arm1_time = imp['time_secs']
                break
        
        for imp in arm2_improvements:
            if imp['improvement_pct'] >= pct:
                arm2_time = imp['time_secs']
                break
        
        if arm1_time or arm2_time:
            print(f"{pct:>3}% improvement:")
            if arm1_time:
                print(f"  {arm1_name}: {arm1_time:.2f}s")
            if arm2_time:
                print(f"  {arm2_name}: {arm2_time:.2f}s")
            if arm1_time and arm2_time:
                speedup = arm1_time / arm2_time
                print(f"  Speedup: {speedup:.2f}x")


def main():
    parser = argparse.ArgumentParser(
        description="Compare convergence milestones between two CP-SAT runs"
    )
    parser.add_argument("log1", type=Path, help="First solver log file")
    parser.add_argument("log2", type=Path, help="Second solver log file")
    parser.add_argument("--name1", default="arm1", help="Name for first arm")
    parser.add_argument("--name2", default="arm2", help="Name for second arm")
    parser.add_argument("--json1", type=Path, help="First scheduler JSON (for final cycles)")
    parser.add_argument("--json2", type=Path, help="Second scheduler JSON (for final cycles)")
    
    args = parser.parse_args()
    
    # Parse improvements
    imp1 = parse_objective_improvements(args.log1)
    imp2 = parse_objective_improvements(args.log2)
    
    # Get final cycles
    final_cycles1 = get_final_cycles(args.json1)
    final_cycles2 = get_final_cycles(args.json2)
    
    # Compute milestones
    milestones1 = compute_milestones(imp1, final_cycles1)
    milestones2 = compute_milestones(imp2, final_cycles2)
    
    # Format comparison
    format_comparison(args.name1, milestones1, args.name2, milestones2)
    
    return 0


if __name__ == "__main__":
    exit(main())
