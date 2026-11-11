#!/usr/bin/env python3
"""General-purpose ablation analyzer.

Reads the JSON output from run_ablation.py and produces a detailed comparison
of all arms. The first arm is treated as the baseline.

Usage:
    uv run python scripts/analyze_ablation.py
    uv run python scripts/analyze_ablation.py ablation_general_results.json
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_results(json_path: str) -> Dict[str, Any]:
    """Load ablation results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def analyze_liveness_curve(liveness_curve: List[Any]) -> Dict[str, Any]:
    """Analyze liveness curve patterns."""
    if not liveness_curve:
        return {}

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
    if 'shifted_ops' in valu_shift_debug:
        shifted = valu_shift_debug['shifted_ops']
        analysis['total_shifted'] = len(shifted)
        by_type = {}
        for op in shifted:
            op_type = op.get('op_type', 'unknown')
            by_type[op_type] = by_type.get(op_type, 0) + 1
        analysis['by_type'] = by_type

    return analysis


def extract_solver_stats(json_data: Optional[Dict]) -> Dict[str, Any]:
    """Extract solver statistics from scheduler output JSON."""
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
    stats['lb_cycles'] = json_data.get('lb_cycles')

    # Solver stats
    if 'solver_stats' in json_data:
        ss = json_data['solver_stats']
        stats['solver_time'] = ss.get('solve_time_seconds')
        stats['num_branches'] = ss.get('num_branches')
        stats['num_conflicts'] = ss.get('num_conflicts')
        stats['num_restarts'] = ss.get('num_restarts')
        stats['objective_value'] = ss.get('objective_value')
        stats['best_bound'] = ss.get('best_bound')
        stats['time_to_feasibility'] = ss.get('time_to_feasibility_seconds')

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

    return stats


def print_arm_detail(idx: int, result: Dict[str, Any], is_baseline: bool):
    """Print detailed metrics for a single arm."""
    label = "BASELINE" if is_baseline else f"arm {idx}"
    name = result['arm_name']
    status = result.get('solver_status', '?')

    print(f"\n{idx}. [{label}] {name}")
    if result.get('extra_args'):
        print(f"   Options: {result['extra_args']}")

    # Failure / validation
    if result.get('failure_reason'):
        print(f"   Failure reason: {result['failure_reason']}")
    if result.get('validation_passed') is False:
        print(f"   Validation: FAILED")
    elif result.get('validation_passed') is True:
        print(f"   Validation: PASSED")
    if result.get('return_code', 0) != 0:
        print(f"   Return code: {result['return_code']}")

    # Basic metrics
    makespan = result.get('makespan')
    if makespan is not None:
        print(f"   Makespan: {makespan} cycles")
    elif status == 'INFEASIBLE':
        print(f"   Makespan: -- (INFEASIBLE: model proven to have no solution)")
    elif status == 'UNKNOWN':
        print(f"   Makespan: -- (UNKNOWN: no solution within time limit)")
    else:
        print(f"   Makespan: -- (solver status: {status})")

    if result.get('elapsed_time'):
        print(f"   Elapsed time: {result['elapsed_time']:.1f}s")
    if result.get('solve_time_secs'):
        print(f"   Solve time: {result['solve_time_secs']:.2f}s")
    if result.get('solver_status'):
        print(f"   Solver status: {result['solver_status']}")
    if result.get('optimality_gap') is not None:
        print(f"   Optimality gap: {result['optimality_gap']:.4f}")
    if result.get('solver_iterations'):
        print(f"   Branches: {result['solver_iterations']}")
    if result.get('solution_count'):
        print(f"   Solutions found: {result['solution_count']}")
    if result.get('time_to_feasibility'):
        rounds = result.get('rounds', 1)
        print(f"   Time to feasibility: {result['time_to_feasibility']:.2f}s "
              f"({result['time_to_feasibility']/rounds:.2f}s/round)")
    if result.get('time_to_optimum'):
        rounds = result.get('rounds', 1)
        print(f"   Time to optimum: {result['time_to_optimum']:.2f}s "
              f"({result['time_to_optimum']/rounds:.2f}s/round)")

    # JSON-derived stats
    json_data = result.get('json_data')
    if not json_data:
        return

    print(f"   Total ops: {json_data.get('total_ops', 'N/A')}")
    print(f"   Peak liveness: {json_data.get('peak_live_words', 'N/A')} words")
    print(f"   VALU shifted: {json_data.get('valu_shifted', 0)} ops")
    print(f"   Budget exceeded: {json_data.get('budget_exceeded', False)}")
    if json_data.get('lb_cycles') is not None:
        print(f"   Lower bound cycles (this emission): {json_data['lb_cycles']}")

    # Liveness analysis
    if 'liveness_curve' in json_data:
        la = analyze_liveness_curve(json_data['liveness_curve'])
        if la:
            print(f"\n   Liveness Analysis:")
            print(f"     Peak: {la['peak']} words (cycle {la['peak_cycle']})")
            print(f"     Average: {la['average']:.1f} words")
            print(f"     Min: {la['min']} words")

    # Usage accounting
    if 'usage_accounting' in json_data:
        usage = json_data['usage_accounting']
        print(f"\n   Usage Accounting:")
        print(f"     Total vregs: {usage.get('total_vregs', 'N/A')}")
        print(f"     Total words: {usage.get('total_words', 'N/A')}")

        if 'by_round' in usage:
            print(f"     By round:")
            for rd in usage['by_round']:
                rnd = rd.get('round', '?')
                s = rd.get('stats', {})
                print(f"       Round {rnd}: {s.get('vreg_count', 'N/A')} vregs, "
                      f"{s.get('total_words', 'N/A')} words "
                      f"({s.get('scalar_count', 0)} scalar, {s.get('vector_count', 0)} vector)")

        if 'by_var_type' in usage:
            print(f"     By variable type:")
            for vd in usage['by_var_type']:
                vt = vd.get('var_type', '?')
                s = vd.get('stats', {})
                print(f"       {vt}: {s.get('vreg_count', 'N/A')} vregs, "
                      f"{s.get('total_words', 'N/A')} words")

    # VALU shift analysis
    if 'valu_shift_debug' in json_data:
        sa = analyze_valu_shift(json_data['valu_shift_debug'])
        if sa:
            print(f"\n   VALU Shift Analysis:")
            print(f"     Total shifted: {sa.get('total_shifted', 0)}")
            if 'by_type' in sa:
                print(f"     By operation type:")
                for op_type, count in sa['by_type'].items():
                    print(f"       {op_type}: {count}")

    # Solver stats from JSON
    if 'solver_stats' in json_data:
        ss = json_data['solver_stats']
        print(f"\n   Solver Statistics (from JSON):")
        if ss.get('solve_time_seconds') is not None:
            print(f"     Solver time: {ss['solve_time_seconds']:.2f}s")
        if ss.get('num_branches') is not None:
            print(f"     Branches: {ss['num_branches']}")
        if ss.get('num_conflicts') is not None:
            print(f"     Conflicts: {ss['num_conflicts']}")
        if ss.get('num_restarts') is not None:
            print(f"     Restarts: {ss['num_restarts']}")
        if ss.get('time_to_feasibility_seconds') is not None:
            print(f"     Time to feasibility: {ss['time_to_feasibility_seconds']:.2f}s")

    # Slot usage
    if 'slot_usage' in json_data:
        print(f"\n   Slot Usage:")
        for engine, usage in json_data['slot_usage'].items():
            used = usage.get('used', 0)
            total = usage.get('total', 0)
            util = usage.get('utilization_percent', 0)
            print(f"     {engine.upper()}: {used}/{total} ({util:.1f}%)")


def print_comparison(results: List[Dict[str, Any]]):
    """Print pairwise comparison of each arm against the baseline (first arm)."""
    if len(results) < 2:
        return

    baseline = results[0]
    b_makespan = baseline.get('makespan')
    b_elapsed = baseline.get('elapsed_time')
    b_json = baseline.get('json_data') or {}
    b_solver = b_json.get('solver_stats') or {}

    print(f"\n{'='*80}")
    print("COMPARISON (vs baseline)")
    print(f"{'='*80}")

    for r in results[1:]:
        name = r['arm_name']
        r_makespan = r.get('makespan')
        r_elapsed = r.get('elapsed_time')
        r_json = r.get('json_data') or {}
        r_solver = r_json.get('solver_stats') or {}

        print(f"\n  {baseline['arm_name']}  vs  {name}")
        print(f"  {'-'*60}")

        # Makespan
        if b_makespan is not None and r_makespan is not None:
            diff = r_makespan - b_makespan
            pct = (diff / b_makespan) * 100 if b_makespan else 0
            better = diff < 0
            indicator = "better" if better else ("worse" if diff > 0 else "same")
            print(f"  Makespan: {b_makespan} -> {r_makespan} ({diff:+d} cycles, {pct:+.1f}%, {indicator})")
        elif b_makespan is not None:
            print(f"  Makespan: {b_makespan} -> -- ({r.get('solver_status', '?')})")
        elif r_makespan is not None:
            print(f"  Makespan: -- -> {r_makespan}")

        # Elapsed time
        if b_elapsed and r_elapsed:
            tdiff = r_elapsed - b_elapsed
            tpct = (tdiff / b_elapsed) * 100 if b_elapsed else 0
            print(f"  Elapsed: {b_elapsed:.1f}s -> {r_elapsed:.1f}s ({tdiff:+.1f}s, {tpct:+.1f}%)")

        # Solve time
        b_solve = baseline.get('solve_time_secs')
        r_solve = r.get('solve_time_secs')
        if b_solve and r_solve:
            sdiff = r_solve - b_solve
            spct = (sdiff / b_solve) * 100 if b_solve else 0
            print(f"  Solve time: {b_solve:.2f}s -> {r_solve:.2f}s ({sdiff:+.2f}s, {spct:+.1f}%)")

        # Peak liveness
        b_peak = b_json.get('peak_live_words')
        r_peak = r_json.get('peak_live_words')
        if b_peak is not None and r_peak is not None:
            pdiff = r_peak - b_peak
            print(f"  Peak liveness: {b_peak} -> {r_peak} ({pdiff:+d} words)")

        # VALU shifted
        b_shifted = b_json.get('valu_shifted', 0)
        r_shifted = r_json.get('valu_shifted', 0)
        if b_shifted or r_shifted:
            print(f"  VALU shifted: {b_shifted} -> {r_shifted} ({r_shifted - b_shifted:+d})")

        # Solver stats comparison
        b_branches = b_solver.get('num_branches')
        r_branches = r_solver.get('num_branches')
        if b_branches is not None and r_branches is not None:
            print(f"  Branches: {b_branches} -> {r_branches}")

        b_conflicts = b_solver.get('num_conflicts')
        r_conflicts = r_solver.get('num_conflicts')
        if b_conflicts is not None and r_conflicts is not None:
            print(f"  Conflicts: {b_conflicts} -> {r_conflicts}")

        # Slot usage comparison
        b_slots = b_json.get('slot_usage', {})
        r_slots = r_json.get('slot_usage', {})
        if b_slots and r_slots:
            engines = sorted(set(b_slots) | set(r_slots))
            slot_diffs = []
            for engine in engines:
                b_util = b_slots.get(engine, {}).get('utilization_percent', 0)
                r_util = r_slots.get(engine, {}).get('utilization_percent', 0)
                diff_util = r_util - b_util
                if abs(diff_util) > 0.1:
                    slot_diffs.append(f"{engine.upper()}: {b_util:.1f}% -> {r_util:.1f}% ({diff_util:+.1f}pp)")
            if slot_diffs:
                print(f"  Slot utilization changes:")
                for sd in slot_diffs:
                    print(f"    {sd}")


def print_summary_table(results: List[Dict[str, Any]]):
    """Print a compact summary table for quick comparison."""
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")

    # Header
    name_width = max(len(r['arm_name']) for r in results)
    name_width = max(name_width, 4)  # minimum "Name"
    note_width = 18  # failure_reason or validation
    print(f"  {'Name':<{name_width}}  {'Cycles':>7}  {'Status':>8}  {'Solve(s)':>8}  "
          f"{'Peak Live':>9}  {'VALU Shift':>10}  {'vs Base':>8}  {'Note':<{note_width}}")
    print(f"  {'-'*name_width}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*10}  {'-'*8}  {'-'*note_width}")

    baseline_makespan = results[0].get('makespan') if results else None

    for i, r in enumerate(results):
        name = r['arm_name']
        makespan = r.get('makespan')
        status = r.get('solver_status', '?')
        solve = r.get('solve_time_secs')
        json_data = r.get('json_data') or {}
        peak = json_data.get('peak_live_words', '')
        shifted = json_data.get('valu_shifted', '')

        ms_str = str(makespan) if makespan is not None else '--'
        solve_str = f"{solve:.1f}" if solve is not None else '--'
        peak_str = str(peak) if peak != '' else '--'
        shift_str = str(shifted) if shifted != '' else '--'

        if i == 0 or baseline_makespan is None or makespan is None:
            vs_str = '--'
        else:
            diff = makespan - baseline_makespan
            vs_str = f"{diff:+d}"

        # Note: failure reason or validation status
        note = r.get('failure_reason') or ''
        if not note and r.get('validation_passed') is False:
            note = 'validation_failed'
        if not note and r.get('validation_passed') is True:
            note = 'OK'
        note_str = (note or '')[:note_width]

        print(f"  {name:<{name_width}}  {ms_str:>7}  {status:>8}  {solve_str:>8}  "
              f"{peak_str:>9}  {shift_str:>10}  {vs_str:>8}  {note_str:<{note_width}}")


def main():
    if len(sys.argv) < 2:
        json_path = "ablation_general_results.json"
    else:
        json_path = sys.argv[1]

    if not Path(json_path).exists():
        print(f"Error: File not found: {json_path}")
        print(f"Run 'uv run python scripts/run_ablation.py ...' first to generate results.")
        return 1

    try:
        data = load_results(json_path)
    except Exception as e:
        print(f"Error loading results: {e}")
        return 1

    config = data.get('config', {})
    results = data.get('results', [])

    if not results:
        print("No results found in the ablation file.")
        return 1

    # Header
    print("=" * 80)
    print("GENERAL ABLATION STUDY ANALYSIS")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Rounds: {config.get('rounds', '?')}")
    print(f"  Batch size: {config.get('batch_size', '?')}")
    print(f"  Time limit: {config.get('time_limit', '?')}s")
    if config.get('shared_args'):
        print(f"  Shared args: {config['shared_args']}")
    print(f"  Arms: {len(results)}")

    # Summary table first
    print_summary_table(results)

    # Per-arm details
    print(f"\n{'-'*80}")
    print("PER-ARM DETAILS")
    print(f"{'-'*80}")

    for i, result in enumerate(results):
        print_arm_detail(i + 1, result, is_baseline=(i == 0))

    # Pairwise comparison
    print_comparison(results)

    print(f"\n{'='*80}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
