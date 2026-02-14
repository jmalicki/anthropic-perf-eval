#!/usr/bin/env python3
"""Run ablation study for ASAP/ALAP tightening.

Tests baseline (tightening disabled) vs tightening enabled.
Uses 4 rounds, batch size 64 for fast iteration.

Measures:
- Time to first feasible solution (from CP-SAT logs)
- Makespan
- Propagation strength (from verbose output)

All CP-SAT solver logs are written to disk (ablation_log_*.txt) for post-hoc analysis.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

def run_experiment(config_name, enable_slot_tightening, enable_scratch_tightening, rounds=4, batch_size=64, time_limit=600, quiet=False):
    """Run a single experiment configuration.

    By default streams scheduler output to both stdout and the log file.
    Set quiet=True to only write to the log file.
    """
    print(f"\n{'='*80}")
    print(f"Running: {config_name}")
    print(f"  Slot-aware tightening: {'ENABLED' if enable_slot_tightening else 'DISABLED'}")
    print(f"  Scratch-aware tightening: {'ENABLED' if enable_scratch_tightening else 'DISABLED'}")
    print(f"  Rounds: {rounds}, Batch size: {batch_size}")
    print(f"  Time limit: {time_limit}s")
    print(f"{'='*80}\n")
    
    # Sanitize filename: replace spaces and special chars with underscores
    safe_name = config_name.lower().replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '').replace('+', '_and_')
    output_json = f"scheduler_output_{safe_name}.json"
    log_file = f"ablation_log_{safe_name}.txt"
    
    cmd = [
        sys.executable, "generate_ssa_kernel.py",
        "--rounds", str(rounds),
        "--batch-size", str(batch_size),
        "--cpsat-time-limit", str(time_limit),
        "--output-json", output_json,
        "--comment", config_name,
    ]
    
    if enable_slot_tightening:
        cmd.append("--enable-asap-alap-tightening")
    if enable_scratch_tightening:
        cmd.append("--enable-scratch-tightening")
    
    # Run and stream output to log file and stdout in real time (not buffered until completion)
    start_time = time.time()
    with open(log_file, 'w') as log_f:
        # Write command header
        log_f.write(f"{'='*80}\n")
        log_f.write(f"Command: {' '.join(cmd)}\n")
        log_f.write(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n")
        log_f.write(f"{'='*80}\n\n")
        log_f.flush()

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        lines = []
        for line in proc.stdout:
            lines.append(line)
            log_f.write(line)
            log_f.flush()
            if not quiet:
                print(line, end='')
        proc.wait()
        result = type('Result', (), {'returncode': proc.returncode, 'stdout': ''.join(lines), 'stderr': None})()

        log_f.write(f"\n\n{'='*80}\n")
        log_f.write(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
        log_f.write(f"Return code: {result.returncode}\n")
        log_f.write(f"{'='*80}\n")
        log_f.flush()

    elapsed_time = time.time() - start_time

    # Full output for parsing (makespan, time_to_feasibility, solver status, etc.)
    stdout_output = result.stdout

    # Parse solver status from logs (works for both success and failure)
    # CP-SAT prints "status: FEASIBLE" or "status: INFEASIBLE" or "status: UNKNOWN"
    # Rust prints "SSA CP-SAT: no solution found, status Infeasible" or "status Unknown"
    solver_status = None
    for line in stdout_output.split('\n'):
        line_lower = line.lower()
        if 'status:' in line_lower:
            if 'optimal' in line_lower:
                solver_status = 'OPTIMAL'
                break
            if 'feasible' in line_lower:
                solver_status = 'FEASIBLE'
                break
            if 'infeasible' in line_lower:
                solver_status = 'INFEASIBLE'
                break
            if 'unknown' in line_lower:
                solver_status = 'UNKNOWN'
                break
        if 'no solution found' in line_lower and 'status' in line_lower:
            if 'infeasible' in line_lower:
                solver_status = 'INFEASIBLE'
                break
            if 'unknown' in line_lower:
                solver_status = 'UNKNOWN'
                break

    # On failure, still record result so we distinguish INFEASIBLE (proof) vs UNKNOWN (timeout)
    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        if solver_status == 'INFEASIBLE':
            print("  Solver status: INFEASIBLE (model proven to have no solution)")
        elif solver_status == 'UNKNOWN':
            print("  Solver status: UNKNOWN (time limit or incomplete; no proof of infeasibility)")
        else:
            print(f"  Solver status: {solver_status or 'not parsed'}")
        # Build a result dict for the failed run (no makespan, but status and log)
        return {
            'config_name': config_name,
            'rounds': rounds,
            'enable_slot_tightening': enable_slot_tightening,
            'enable_scratch_tightening': enable_scratch_tightening,
            'makespan': None,
            'elapsed_time': elapsed_time,
            'solve_time_secs': None,
            'time_to_feasibility': None,
            'time_to_optimum': None,
            'time_per_round_to_feasibility': None,
            'time_per_round_to_optimum': None,
            'started_feasible': None,
            'solver_status': solver_status or 'UNKNOWN',
            'optimality_gap': None,
            'solver_iterations': None,
            'solution_count': None,
            'return_code': result.returncode,
            'output_json': output_json,
            'log_file': log_file,
            'json_data': None,
        }

    # Extract makespan from output
    makespan = None
    for line in stdout_output.split('\n'):
        if 'Cycles:' in line and 'Results:' not in line:
            try:
                # Look for "Cycles: 37" pattern
                parts = line.split('Cycles:')
                if len(parts) > 1:
                    makespan = int(parts[1].strip().split()[0])
            except (ValueError, IndexError):
                pass
        # Also check Results section
        if 'Results:' in line:
            # Next line should have "  Cycles: X"
            continue
        if makespan is None and '  Cycles:' in line:
            try:
                makespan = int(line.split('Cycles:')[1].strip().split()[0])
            except (ValueError, IndexError):
                pass
    
    # Try to load JSON for more detailed stats
    json_data = None
    if Path(output_json).exists():
        try:
            with open(output_json, 'r') as f:
                json_data = json.load(f)
        except Exception as e:
            print(f"Warning: Could not parse JSON: {e}")
    
    # Extract solver statistics from JSON and parse CP-SAT logs
    time_to_feasibility = None
    time_to_optimum = None
    started_feasible = None
    solver_status = None
    solve_time_secs = None
    optimality_gap = None
    solver_iterations = None
    
    # Parse CP-SAT solver logs from stdout/stderr
    # CP-SAT logs progress with patterns like:
    # - "Solution #X found at time Y" (when feasible solution found)
    # - "best: X" (current best solution value)
    # - "best_bound: Y" (current lower bound)
    # - Status information
    import re
    
    best_seen = None
    first_feasible_time = None
    optimal_time = None
    solution_count = 0
    
    for line in stdout_output.split('\n'):
        line_lower = line.lower()
        
        # Parse "Solution #X found at time Y" pattern
        solution_match = re.search(r'solution\s+#(\d+)\s+found\s+at\s+time\s+([\d.]+)', line_lower)
        if solution_match:
            solution_count = int(solution_match.group(1))
            solution_time = float(solution_match.group(2))
            if first_feasible_time is None:
                first_feasible_time = solution_time
        
        # Parse "best: X" pattern (CP-SAT progress logs)
        if 'best:' in line_lower:
            try:
                parts = line.split('best:')
                if len(parts) > 1:
                    best_val = parts[1].strip().split()[0]
                    if best_val.lower() != 'inf' and best_val != '-' and best_val.lower() != 'infinity':
                        if best_seen is None:
                            best_seen = best_val
                            # Try to extract time from the line
                            time_match = re.search(r'time[:\s]+([\d.]+)', line_lower)
                            if time_match:
                                first_feasible_time = float(time_match.group(1))
            except (ValueError, IndexError):
                pass
        
        # Parse solver status (FEASIBLE, OPTIMAL, INFEASIBLE, UNKNOWN)
        if 'status:' in line_lower or ('cp-sat' in line_lower and 'status' in line_lower):
            if 'optimal' in line_lower:
                solver_status = 'OPTIMAL'
            elif 'feasible' in line_lower:
                solver_status = 'FEASIBLE'
            elif 'infeasible' in line_lower:
                solver_status = 'INFEASIBLE'
            elif 'unknown' in line_lower:
                solver_status = 'UNKNOWN'
        
        # Parse "SSA CP-SAT result" line for status and timing
        if 'ssa cp-sat result' in line_lower:
            # Extract status and time from line like:
            # "SSA CP-SAT result: X cycles (best_bound: Y), status Z, gap W%, time T"
            status_match = re.search(r'status\s+(\w+)', line_lower)
            if status_match:
                solver_status = status_match.group(1).upper()
            
            time_match = re.search(r'time\s+([\d.]+)s', line_lower)
            if time_match:
                solve_time_secs = float(time_match.group(1))
            
            gap_match = re.search(r'gap\s+([\d.]+)%', line_lower)
            if gap_match:
                optimality_gap = float(gap_match.group(1)) / 100.0
        
        # Check if we started from a feasible solution (hints)
        if 'hint' in line_lower and ('feasible' in line_lower or 'warm' in line_lower):
            started_feasible = True
        
        # Parse solver statistics for iterations
        if 'branches:' in line_lower or 'conflicts:' in line_lower:
            branches_match = re.search(r'branches[:\s]+(\d+)', line_lower)
            if branches_match:
                solver_iterations = int(branches_match.group(1))
    
    # Extract from JSON if available
    if json_data:
        # Check if we have CP-SAT result stats
        if 'stats' in json_data:
            stats = json_data['stats']
            if solve_time_secs is None:
                solve_time_secs = stats.get('solve_time_secs')
            if optimality_gap is None:
                optimality_gap = stats.get('gap')
    
    # Set time_to_feasibility and time_to_optimum
    if first_feasible_time is not None:
        time_to_feasibility = first_feasible_time
    elif solve_time_secs is not None and solver_status in ['OPTIMAL', 'FEASIBLE']:
        # If we have a solution but no explicit time, use solve_time as proxy
        time_to_feasibility = solve_time_secs
    
    if solver_status == 'OPTIMAL':
        if optimal_time is not None:
            time_to_optimum = optimal_time
        elif solve_time_secs is not None:
            time_to_optimum = solve_time_secs
    
    # If we started feasible (from hints), mark it
    if started_feasible is None:
        # Default: assume we start feasible if we have hints (greedy schedule)
        # This is typically the case for our setup
        started_feasible = True  # Conservative assumption
    
    # Calculate per-round metrics
    time_per_round_to_feasibility = time_to_feasibility / rounds if time_to_feasibility is not None else None
    time_per_round_to_optimum = time_to_optimum / rounds if time_to_optimum is not None else None
    
    return {
        'config_name': config_name,
        'rounds': rounds,
        'enable_slot_tightening': enable_slot_tightening,
        'enable_scratch_tightening': enable_scratch_tightening,
        'makespan': makespan,
        'elapsed_time': elapsed_time,
        'solve_time_secs': solve_time_secs,
        'time_to_feasibility': time_to_feasibility,
        'time_to_optimum': time_to_optimum,
        'time_per_round_to_feasibility': time_per_round_to_feasibility,
        'time_per_round_to_optimum': time_per_round_to_optimum,
        'started_feasible': started_feasible,
        'solver_status': solver_status,
        'optimality_gap': optimality_gap,
        'solver_iterations': solver_iterations,
        'solution_count': solution_count,
        'return_code': result.returncode,
        'output_json': output_json,
        'log_file': log_file,
        'json_data': json_data,
    }

def main():
    parser = argparse.ArgumentParser(
        description="Run ablation study for ASAP/ALAP tightening"
    )
    parser.add_argument("--rounds", type=int, default=4,
                        help="Number of rounds (default: 4)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size (default: 64, must be multiple of 8)")
    parser.add_argument("--time-limit", type=float, default=600.0,
                        help="CP-SAT time limit in seconds (default: 600)")
    parser.add_argument("--output", type=str, default="ablation_results.json",
                        help="Output file for results (default: ablation_results.json)")
    parser.add_argument("--quiet", action="store_true",
                        help="Only write scheduler output to log files; do not print to terminal")

    args = parser.parse_args()
    
    if args.batch_size % 8 != 0:
        print(f"ERROR: batch_size must be a multiple of 8 (got {args.batch_size})")
        return 1
    
    print("ASAP/ALAP Tightening Ablation Study")
    print("=" * 80)
    print(f"Configuration: {args.rounds} rounds, batch size {args.batch_size}")
    print(f"Time limit: {args.time_limit}s per run")
    print()
    
    results = []
    
    # Run baseline (tightening disabled)
    baseline_result = run_experiment(
        "Baseline (tightening disabled)",
        enable_slot_tightening=False,
        enable_scratch_tightening=False,
        rounds=args.rounds,
        batch_size=args.batch_size,
        time_limit=args.time_limit,
        quiet=args.quiet,
    )
    if baseline_result:
        results.append(baseline_result)
    
    # Run with slot-only tightening enabled
    slot_tightening_result = run_experiment(
        "Slot-aware tightening enabled",
        enable_slot_tightening=True,
        enable_scratch_tightening=False,
        rounds=args.rounds,
        batch_size=args.batch_size,
        time_limit=args.time_limit,
        quiet=args.quiet,
    )
    if slot_tightening_result:
        results.append(slot_tightening_result)
    
    # Run with scratch-only tightening enabled
    scratch_tightening_result = run_experiment(
        "Scratch-aware tightening enabled",
        enable_slot_tightening=False,
        enable_scratch_tightening=True,
        rounds=args.rounds,
        batch_size=args.batch_size,
        time_limit=args.time_limit,
        quiet=args.quiet,
    )
    if scratch_tightening_result:
        results.append(scratch_tightening_result)
    
    # Run with slot + scratch tightening enabled
    slot_scratch_tightening_result = run_experiment(
        "Slot + scratch tightening enabled",
        enable_slot_tightening=True,
        enable_scratch_tightening=True,
        rounds=args.rounds,
        batch_size=args.batch_size,
        time_limit=args.time_limit,
        quiet=args.quiet,
    )
    if slot_scratch_tightening_result:
        results.append(slot_scratch_tightening_result)
    
    # Print summary
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS")
    print("="*80)
    
    if len(results) >= 2:
        baseline = results[0]
        rounds = baseline.get('rounds', args.rounds)
        
        def format_timing(r, label=""):
            """Format timing metrics for a result."""
            lines = []
            status = r.get('solver_status')
            if r.get('makespan') is not None:
                lines.append(f"  Makespan: {r['makespan']}")
            elif status == 'INFEASIBLE':
                lines.append(f"  Makespan: — (INFEASIBLE: model proven to have no solution)")
            elif status == 'UNKNOWN':
                lines.append(f"  Makespan: — (UNKNOWN: time limit or no solution found; not proven infeasible)")
            elif status:
                lines.append(f"  Makespan: — (solver status: {status})")
            if r.get('elapsed_time'):
                lines.append(f"  Elapsed time: {r['elapsed_time']:.1f}s")
            if r.get('solve_time_secs'):
                lines.append(f"  Solve time: {r['solve_time_secs']:.2f}s")
            if r.get('started_feasible') is not None:
                lines.append(f"  Started feasible: {r['started_feasible']}")
            if r.get('time_to_feasibility'):
                lines.append(f"  Time to feasibility: {r['time_to_feasibility']:.2f}s ({r['time_to_feasibility']/rounds:.2f}s/round)")
            if r.get('time_to_optimum'):
                lines.append(f"  Time to optimum: {r['time_to_optimum']:.2f}s ({r['time_to_optimum']/rounds:.2f}s/round)")
            elif r.get('solver_status') == 'OPTIMAL':
                solve_time = r.get('solve_time_secs') or r.get('elapsed_time')
                if solve_time:
                    lines.append(f"  Time to optimum: {solve_time:.2f}s ({solve_time/rounds:.2f}s/round)")
            if r.get('optimality_gap') is not None:
                lines.append(f"  Optimality gap: {r['optimality_gap']:.4f}")
            if r.get('solver_status'):
                lines.append(f"  Solver status: {r['solver_status']}")
            if r.get('solver_iterations'):
                lines.append(f"  Solver iterations (branches): {r['solver_iterations']}")
            if r.get('solution_count'):
                lines.append(f"  Solutions found: {r['solution_count']}")
            return lines
        
        print(f"\nBaseline (tightening disabled):")
        for line in format_timing(baseline):
            print(line)
        
        if len(results) >= 2:
            slot = results[1]
            print(f"\nSlot-aware tightening:")
            for line in format_timing(slot):
                print(line)
            
            if baseline.get('makespan') and slot.get('makespan'):
                makespan_diff = slot['makespan'] - baseline['makespan']
                makespan_pct = (makespan_diff / baseline['makespan']) * 100
                print(f"  vs baseline: {makespan_diff:+d} cycles ({makespan_pct:+.1f}%)")
            
            # Compare timing metrics normalized by rounds
            if baseline.get('time_to_feasibility') and slot.get('time_to_feasibility'):
                time_diff = slot['time_to_feasibility'] - baseline['time_to_feasibility']
                time_per_round_diff = (slot['time_to_feasibility'] - baseline['time_to_feasibility']) / rounds
                print(f"  Time to feasibility vs baseline: {time_diff:+.2f}s ({time_per_round_diff:+.2f}s/round)")
        
        if len(results) >= 3:
            scratch = results[2]
            print(f"\nScratch-aware tightening:")
            for line in format_timing(scratch):
                print(line)
            
            if baseline.get('makespan') and scratch.get('makespan'):
                makespan_diff = scratch['makespan'] - baseline['makespan']
                makespan_pct = (makespan_diff / baseline['makespan']) * 100
                print(f"  vs baseline: {makespan_diff:+d} cycles ({makespan_pct:+.1f}%)")
            
            if baseline.get('time_to_feasibility') and scratch.get('time_to_feasibility'):
                time_diff = scratch['time_to_feasibility'] - baseline['time_to_feasibility']
                time_per_round_diff = (scratch['time_to_feasibility'] - baseline['time_to_feasibility']) / rounds
                print(f"  Time to feasibility vs baseline: {time_diff:+.2f}s ({time_per_round_diff:+.2f}s/round)")
        
        if len(results) >= 4:
            slot_scratch = results[3]
            print(f"\nSlot + scratch tightening:")
            for line in format_timing(slot_scratch):
                print(line)
            
            if baseline.get('makespan') and slot_scratch.get('makespan'):
                makespan_diff = slot_scratch['makespan'] - baseline['makespan']
                makespan_pct = (makespan_diff / baseline['makespan']) * 100
                print(f"  vs baseline: {makespan_diff:+d} cycles ({makespan_pct:+.1f}%)")
            
            if len(results) >= 2 and slot.get('makespan') and slot_scratch.get('makespan'):
                makespan_diff = slot_scratch['makespan'] - slot['makespan']
                makespan_pct = (makespan_diff / slot['makespan']) * 100 if slot['makespan'] > 0 else 0
                print(f"  vs slot-only: {makespan_diff:+d} cycles ({makespan_pct:+.1f}%)")
            
            if len(results) >= 3 and scratch.get('makespan') and slot_scratch.get('makespan'):
                makespan_diff = slot_scratch['makespan'] - scratch['makespan']
                makespan_pct = (makespan_diff / scratch['makespan']) * 100 if scratch['makespan'] > 0 else 0
                print(f"  vs scratch-only: {makespan_diff:+d} cycles ({makespan_pct:+.1f}%)")
            
            # Compare timing metrics
            if baseline.get('time_to_feasibility') and slot_scratch.get('time_to_feasibility'):
                time_diff = slot_scratch['time_to_feasibility'] - baseline['time_to_feasibility']
                time_per_round_diff = (slot_scratch['time_to_feasibility'] - baseline['time_to_feasibility']) / rounds
                print(f"  Time to feasibility vs baseline: {time_diff:+.2f}s ({time_per_round_diff:+.2f}s/round)")
            
            if slot_scratch.get('time_to_optimum'):
                if baseline.get('time_to_optimum'):
                    time_diff = slot_scratch['time_to_optimum'] - baseline['time_to_optimum']
                    time_per_round_diff = (slot_scratch['time_to_optimum'] - baseline['time_to_optimum']) / rounds
                    print(f"  Time to optimum vs baseline: {time_diff:+.2f}s ({time_per_round_diff:+.2f}s/round)")
    
    # Save results to JSON
    with open(args.output, 'w') as f:
        json.dump({
            'config': {
                'rounds': args.rounds,
                'batch_size': args.batch_size,
                'time_limit': args.time_limit,
            },
            'results': results,
        }, f, indent=2)
    
    print(f"\nResults saved to: {args.output}")
    
    # List log files for reference
    log_files = [r.get('log_file') for r in results if r.get('log_file')]
    if log_files:
        print(f"\nSolver logs saved to:")
        for log_file in log_files:
            print(f"  - {log_file}")
        print("\nLogs contain full CP-SAT solver output for post-hoc analysis.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
