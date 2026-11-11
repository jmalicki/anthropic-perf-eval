#!/usr/bin/env python3
"""Analyze ablation study results from ablation_results.json.

Extracts and summarizes:
- Makespan comparison
- Solve time comparison
- Solver statistics (if available)
- Slot usage patterns
- VALU shift statistics
- Liveness patterns
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

def load_results(json_path: str) -> Dict[str, Any]:
    """Load ablation results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)

def extract_solver_stats(json_data: Optional[Dict]) -> Dict[str, Any]:
    """Extract solver statistics from json_data."""
    if not json_data:
        return {}
    
    stats = {}
    
    # Basic metrics
    stats['total_cycles'] = json_data.get('total_cycles')
    stats['total_ops'] = json_data.get('total_ops')
    stats['vreg_count'] = json_data.get('vreg_count')
    stats['peak_live_words'] = json_data.get('peak_live_words')
    stats['max_addr'] = json_data.get('max_addr')
    stats['budget_exceeded'] = json_data.get('budget_exceeded', False)
    stats['valu_shifted'] = json_data.get('valu_shifted', 0)
    
    # Solver stats
    if 'solver_stats' in json_data:
        solver_stats = json_data['solver_stats']
        stats['solver_time'] = solver_stats.get('solve_time_seconds')
        stats['num_branches'] = solver_stats.get('num_branches')
        stats['num_conflicts'] = solver_stats.get('num_conflicts')
        stats['num_restarts'] = solver_stats.get('num_restarts')
        stats['objective_value'] = solver_stats.get('objective_value')
        stats['best_bound'] = solver_stats.get('best_bound')
        stats['time_to_feasibility'] = solver_stats.get('time_to_feasibility_seconds')
    
    # Slot usage
    if 'slot_usage' in json_data:
        slot_usage = json_data['slot_usage']
        stats['slot_usage'] = {}
        for engine in ['alu', 'valu', 'load', 'store', 'flow']:
            if engine in slot_usage:
                stats['slot_usage'][engine] = {
                    'used': slot_usage[engine].get('used', 0),
                    'total': slot_usage[engine].get('total', 0),
                    'utilization': slot_usage[engine].get('utilization_percent', 0),
                }
    
    # Usage accounting
    if 'usage_accounting' in json_data:
        usage = json_data['usage_accounting']
        stats['usage_accounting'] = {
            'total_vregs': usage.get('total_vregs'),
            'total_words': usage.get('total_words'),
            'peak_live_words': usage.get('peak_live_words'),
        }
    
    return stats

def compare_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare results between configurations."""
    comparison = {
        'num_configs': len(results),
        'configs': [],
        'makespan_comparison': {},
        'time_comparison': {},
        'solver_stats_comparison': {},
    }
    
    # Extract baseline (tightening disabled)
    baseline = None
    tightening = None
    
    for result in results:
        config_info = {
            'name': result['config_name'],
            'enable_tightening': result['enable_tightening'],
            'makespan': result['makespan'],
            'elapsed_time': result['elapsed_time'],
            'solver_stats': extract_solver_stats(result.get('json_data')),
        }
        comparison['configs'].append(config_info)
        
        if not result['enable_tightening']:
            baseline = config_info
        else:
            tightening = config_info
    
    # Compare makespan
    if baseline and tightening:
        comparison['makespan_comparison'] = {
            'baseline': baseline['makespan'],
            'with_tightening': tightening['makespan'],
            'difference': tightening['makespan'] - baseline['makespan'],
            'percent_change': ((tightening['makespan'] - baseline['makespan']) / baseline['makespan'] * 100) if baseline['makespan'] else 0,
        }
    
    # Compare solve time
    if baseline and tightening:
        comparison['time_comparison'] = {
            'baseline': baseline['elapsed_time'],
            'with_tightening': tightening['elapsed_time'],
            'difference': tightening['elapsed_time'] - baseline['elapsed_time'],
            'percent_change': ((tightening['elapsed_time'] - baseline['elapsed_time']) / baseline['elapsed_time'] * 100) if baseline['elapsed_time'] else 0,
        }
        
        # Compare solver stats if available
        baseline_stats = baseline['solver_stats']
        tightening_stats = tightening['solver_stats']
        
        if baseline_stats.get('solver_time') and tightening_stats.get('solver_time'):
            comparison['solver_stats_comparison'] = {
                'solver_time': {
                    'baseline': baseline_stats['solver_time'],
                    'with_tightening': tightening_stats['solver_time'],
                    'difference': tightening_stats['solver_time'] - baseline_stats['solver_time'],
                    'percent_change': ((tightening_stats['solver_time'] - baseline_stats['solver_time']) / baseline_stats['solver_time'] * 100) if baseline_stats['solver_time'] else 0,
                },
                'num_branches': {
                    'baseline': baseline_stats.get('num_branches'),
                    'with_tightening': tightening_stats.get('num_branches'),
                },
                'num_conflicts': {
                    'baseline': baseline_stats.get('num_conflicts'),
                    'with_tightening': tightening_stats.get('num_conflicts'),
                },
            }
    
    return comparison

def analyze_liveness_curve(liveness_curve: List[Any]) -> Dict[str, Any]:
    """Analyze liveness curve patterns."""
    if not liveness_curve:
        return {}
    
    # Handle both list of ints and list of dicts
    if isinstance(liveness_curve[0], dict):
        live_words = [entry.get('total_live_words', 0) for entry in liveness_curve]
    else:
        live_words = liveness_curve
    
    if not live_words:
        return {}
    
    peak_value = max(live_words)
    peak_idx = live_words.index(peak_value)
    
    return {
        'peak': peak_value,
        'average': sum(live_words) / len(live_words),
        'min': min(live_words),
        'cycles': len(live_words),
        'peak_cycle': peak_idx,
    }

def analyze_valu_shift(valu_shift_debug: Optional[Dict]) -> Dict[str, Any]:
    """Analyze VALU shift patterns."""
    if not valu_shift_debug:
        return {}
    
    analysis = {}
    
    # Count shifts by type
    if 'shifted_ops' in valu_shift_debug:
        shifted = valu_shift_debug['shifted_ops']
        analysis['total_shifted'] = len(shifted)
        
        # Group by operation type
        by_type = {}
        for op in shifted:
            op_type = op.get('op_type', 'unknown')
            by_type[op_type] = by_type.get(op_type, 0) + 1
        analysis['by_type'] = by_type
    
    return analysis

def print_summary(data: Dict[str, Any]):
    """Print a formatted summary of the ablation study."""
    config = data['config']
    results = data['results']
    
    print("=" * 80)
    print("ABLATION STUDY ANALYSIS")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Rounds: {config['rounds']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Time limit: {config['time_limit']}s")
    print(f"\nNumber of results: {len(results)}")
    
    # Print each configuration
    print("\n" + "-" * 80)
    print("CONFIGURATION RESULTS")
    print("-" * 80)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['config_name']}")
        print(f"   Tightening enabled: {result['enable_tightening']}")
        print(f"   Makespan: {result['makespan']} cycles")
        print(f"   Elapsed time: {result['elapsed_time']:.2f}s")
        
        if result.get('json_data'):
            json_data = result['json_data']
            print(f"   Total ops: {json_data.get('total_ops', 'N/A')}")
            print(f"   Peak liveness: {json_data.get('peak_live_words', 'N/A')} words")
            print(f"   VALU shifted: {json_data.get('valu_shifted', 0)} ops")
            print(f"   Budget exceeded: {json_data.get('budget_exceeded', False)}")
            print(f"   Lower bound cycles (this emission): {json_data.get('lb_cycles', 'N/A')}")
            
            # Liveness analysis
            if 'liveness_curve' in json_data:
                liveness_analysis = analyze_liveness_curve(json_data['liveness_curve'])
                if liveness_analysis:
                    print(f"\n   Liveness Analysis:")
                    print(f"     Peak: {liveness_analysis['peak']} words (cycle {liveness_analysis['peak_cycle']})")
                    print(f"     Average: {liveness_analysis['average']:.1f} words")
                    print(f"     Min: {liveness_analysis['min']} words")
            
            # Usage accounting breakdown
            if 'usage_accounting' in json_data:
                usage = json_data['usage_accounting']
                print(f"\n   Usage Accounting:")
                print(f"     Total vregs: {usage.get('total_vregs', 'N/A')}")
                print(f"     Total words: {usage.get('total_words', 'N/A')}")
                
                if 'by_round' in usage:
                    print(f"     By round:")
                    for round_data in usage['by_round']:
                        rnd = round_data.get('round', '?')
                        stats = round_data.get('stats', {})
                        print(f"       Round {rnd}: {stats.get('vreg_count', 'N/A')} vregs, "
                              f"{stats.get('total_words', 'N/A')} words "
                              f"({stats.get('scalar_count', 0)} scalar, {stats.get('vector_count', 0)} vector)")
                
                if 'by_var_type' in usage:
                    print(f"     By variable type:")
                    for var_data in usage['by_var_type']:
                        var_type = var_data.get('var_type', '?')
                        stats = var_data.get('stats', {})
                        print(f"       {var_type}: {stats.get('vreg_count', 'N/A')} vregs, "
                              f"{stats.get('total_words', 'N/A')} words")
            
            # VALU shift analysis
            if 'valu_shift_debug' in json_data:
                shift_analysis = analyze_valu_shift(json_data['valu_shift_debug'])
                if shift_analysis:
                    print(f"\n   VALU Shift Analysis:")
                    print(f"     Total shifted: {shift_analysis.get('total_shifted', 0)}")
                    if 'by_type' in shift_analysis:
                        print(f"     By operation type:")
                        for op_type, count in shift_analysis['by_type'].items():
                            print(f"       {op_type}: {count}")
            
            # Solver stats
            if 'solver_stats' in json_data:
                solver_stats = json_data['solver_stats']
                print(f"\n   Solver Statistics:")
                print(f"     Solver time: {solver_stats.get('solve_time_seconds', 'N/A'):.2f}s")
                print(f"     Branches: {solver_stats.get('num_branches', 'N/A')}")
                print(f"     Conflicts: {solver_stats.get('num_conflicts', 'N/A')}")
                print(f"     Restarts: {solver_stats.get('num_restarts', 'N/A')}")
                if solver_stats.get('time_to_feasibility_seconds'):
                    print(f"     Time to feasibility: {solver_stats['time_to_feasibility_seconds']:.2f}s")
            
            # Slot usage
            if 'slot_usage' in json_data:
                print(f"\n   Slot Usage:")
                for engine, usage in json_data['slot_usage'].items():
                    used = usage.get('used', 0)
                    total = usage.get('total', 0)
                    util = usage.get('utilization_percent', 0)
                    print(f"     {engine.upper()}: {used}/{total} ({util:.1f}%)")
    
    # Compare configurations
    comparison = compare_results(results)
    
    if len(comparison['configs']) >= 2:
        print("\n" + "-" * 80)
        print("COMPARISON")
        print("-" * 80)
        
        if comparison['makespan_comparison']:
            makespan = comparison['makespan_comparison']
            print(f"\nMakespan:")
            print(f"  Baseline: {makespan['baseline']} cycles")
            print(f"  With tightening: {makespan['with_tightening']} cycles")
            diff = makespan['difference']
            pct = makespan['percent_change']
            if diff < 0:
                print(f"  ✓ Improvement: {abs(diff)} cycles ({abs(pct):.1f}% better)")
            elif diff > 0:
                print(f"  ✗ Regression: +{diff} cycles ({pct:.1f}% worse)")
            else:
                print(f"  → No change")
        
        if comparison['time_comparison']:
            time_comp = comparison['time_comparison']
            print(f"\nSolve Time:")
            print(f"  Baseline: {time_comp['baseline']:.2f}s")
            print(f"  With tightening: {time_comp['with_tightening']:.2f}s")
            diff = time_comp['difference']
            pct = time_comp['percent_change']
            if diff < 0:
                print(f"  ✓ Improvement: {abs(diff):.2f}s ({abs(pct):.1f}% faster)")
            elif diff > 0:
                print(f"  ✗ Regression: +{diff:.2f}s ({pct:.1f}% slower)")
            else:
                print(f"  → No change")
        
        if comparison['solver_stats_comparison']:
            solver_comp = comparison['solver_stats_comparison']
            print(f"\nSolver Statistics:")
            
            if 'solver_time' in solver_comp:
                st = solver_comp['solver_time']
                print(f"  Solver time:")
                print(f"    Baseline: {st['baseline']:.2f}s")
                print(f"    With tightening: {st['with_tightening']:.2f}s")
                diff = st['difference']
                pct = st['percent_change']
                if diff < 0:
                    print(f"    ✓ Improvement: {abs(diff):.2f}s ({abs(pct):.1f}% faster)")
                elif diff > 0:
                    print(f"    ✗ Regression: +{diff:.2f}s ({pct:.1f}% slower)")
            
            if 'num_branches' in solver_comp:
                nb = solver_comp['num_branches']
                print(f"  Branches:")
                print(f"    Baseline: {nb['baseline']}")
                print(f"    With tightening: {nb['with_tightening']}")
            
            if 'num_conflicts' in solver_comp:
                nc = solver_comp['num_conflicts']
                print(f"  Conflicts:")
                print(f"    Baseline: {nc['baseline']}")
                print(f"    With tightening: {nc['with_tightening']}")
    
    print("\n" + "=" * 80)

def main():
    if len(sys.argv) < 2:
        json_path = "ablation_results.json"
    else:
        json_path = sys.argv[1]
    
    if not Path(json_path).exists():
        print(f"Error: File not found: {json_path}")
        return 1
    
    try:
        data = load_results(json_path)
        print_summary(data)
        return 0
    except Exception as e:
        print(f"Error analyzing results: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
