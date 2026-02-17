#!/usr/bin/env python3
"""Extract convergence milestones from CP-SAT solver logs.

Parses solver logs to find when each objective improvement occurred,
mapping to cycle milestones.
"""

import argparse
import re
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional


def parse_cpsat_log(log_file: Path) -> List[Tuple[float, int]]:
    """Parse CP-SAT log to extract (time, objective_value) pairs.
    
    CP-SAT logs have lines like:
    #1       0.41s best:74010802 next:[54007247,74010801] quick_restart_no_lp [hint]
    #2       2.38s best:74010757 next:[56007352,74010756] graph_arc_lns
    
    Returns list of (time_seconds, objective_value) tuples in chronological order.
    """
    milestones = []
    
    with open(log_file, 'r') as f:
        for line in f:
            # Match solution improvement lines: #N TIME best:OBJ next:[...]
            match = re.search(r'#(\d+)\s+([\d.]+)s\s+best:(\d+)', line)
            if match:
                solution_num = int(match.group(1))
                time_secs = float(match.group(2))
                objective = int(match.group(3))
                milestones.append((time_secs, objective))
            
            # Also match bound improvements: #Bound TIME best:OBJ next:[...]
            # These show when the lower bound improves, which is also useful
            bound_match = re.search(r'#Bound\s+([\d.]+)s\s+best:(\d+)', line)
            if bound_match:
                time_secs = float(bound_match.group(1))
                objective = int(bound_match.group(2))
                # Only add if it's better than the last milestone
                if not milestones or objective < milestones[-1][1]:
                    milestones.append((time_secs, objective))
    
    return milestones


def find_cycle_milestones(log_file: Path, final_cycles: int) -> Dict[str, float]:
    """Find time to reach cycle milestones.
    
    Since CP-SAT objective values are encoded, we track objective improvements
    and assume they map linearly to cycle improvements. We'll report:
    - Time to first feasible solution
    - Time to reach milestones relative to final cycles (e.g., 90%, 95%, 99%, 100%)
    """
    milestones = parse_cpsat_log(log_file)
    
    if not milestones:
        return {}
    
    # Get objective range
    first_obj = milestones[0][1]
    last_obj = milestones[-1][1]
    obj_range = first_obj - last_obj  # Should be positive (decreasing objective)
    
    if obj_range == 0:
        # No improvement, just return first feasible time
        return {
            'time_to_first_feasible': milestones[0][0],
            'time_to_optimal': milestones[-1][0] if len(milestones) > 1 else milestones[0][0],
        }
    
    # Map objective improvements to cycle improvements
    # Assume linear mapping: cycles = final_cycles + (obj - last_obj) / scale
    # We don't know the exact scale, but we can estimate milestones based on
    # when we're X% of the way from first to last objective
    
    result = {
        'time_to_first_feasible': milestones[0][0],
        'time_to_optimal': milestones[-1][0],
    }
    
    # Find times to reach percentage milestones of improvement
    for pct in [90, 95, 99, 100]:
        target_obj = last_obj + (obj_range * (100 - pct) / 100)
        
        # Find first milestone where objective <= target_obj
        for time_secs, obj in milestones:
            if obj <= target_obj:
                result[f'time_to_{pct}pct'] = time_secs
                break
    
    return result


def extract_from_scheduler_json(json_file: Path) -> Optional[Dict]:
    """Extract cycle information from scheduler output JSON if available."""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        stats = data.get('stats', {})
        cycles = stats.get('cycles')
        solve_time = stats.get('solve_time_secs')
        
        if cycles is not None:
            return {
                'final_cycles': cycles,
                'solve_time_secs': solve_time,
            }
    except Exception as e:
        print(f"Warning: Could not parse JSON {json_file}: {e}")
    
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Extract convergence milestones from CP-SAT logs"
    )
    parser.add_argument("log_file", type=Path, help="CP-SAT solver log file")
    parser.add_argument("--json", type=Path, help="Scheduler output JSON (for final cycles)")
    parser.add_argument("--final-cycles", type=int, help="Final cycle count (if not in JSON)")
    
    args = parser.parse_args()
    
    # Get final cycles
    final_cycles = args.final_cycles
    if args.json:
        json_data = extract_from_scheduler_json(args.json)
        if json_data and json_data.get('final_cycles'):
            final_cycles = json_data['final_cycles']
    
    if final_cycles is None:
        print("Error: Need --final-cycles or --json with cycles")
        return 1
    
    # Extract milestones
    milestones = parse_cpsat_log(args.log_file)
    cycle_milestones = find_cycle_milestones(args.log_file, final_cycles)
    
    print(f"Final cycles: {final_cycles}")
    print(f"\nObjective improvements: {len(milestones)}")
    if milestones:
        print(f"  First feasible: {milestones[0][0]:.2f}s (obj={milestones[0][1]})")
        print(f"  Last improvement: {milestones[-1][0]:.2f}s (obj={milestones[-1][1]})")
    
    print(f"\nCycle milestones:")
    for key, value in sorted(cycle_milestones.items()):
        print(f"  {key}: {value:.2f}s")
    
    return 0


if __name__ == "__main__":
    exit(main())
