#!/usr/bin/env python3
"""Run ablation study for ASAP/ALAP tightening.

Tests baseline (tightening disabled) vs tightening enabled.
Uses 4 rounds, batch size 64 for fast iteration.

Measures:
- Time to first feasible solution (from CP-SAT logs)
- Makespan
- Propagation strength (from verbose output)
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

def run_experiment(config_name, enable_slot_tightening, enable_scratch_tightening, rounds=4, batch_size=64, time_limit=600):
    """Run a single experiment configuration."""
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
    
    # Run and capture output
    start_time = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    elapsed_time = time.time() - start_time
    
    # Parse results
    if result.returncode != 0:
        print(f"ERROR: Command failed with return code {result.returncode}")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return None
    
    # Extract makespan from output
    makespan = None
    for line in result.stdout.split('\n'):
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
    
    # Extract time to feasibility from CP-SAT logs (if available)
    time_to_feasibility = None
    if json_data and 'solver_stats' in json_data:
        solver_stats = json_data['solver_stats']
        # Look for first feasible solution time
        # CP-SAT logs show "best: X" when first feasible solution is found
        # We'd need to parse the actual solver logs for this, but for now
        # we can use the total solve time as a proxy
    
    return {
        'config_name': config_name,
        'enable_slot_tightening': enable_slot_tightening,
        'enable_scratch_tightening': enable_scratch_tightening,
        'makespan': makespan,
        'elapsed_time': elapsed_time,
        'return_code': result.returncode,
        'output_json': output_json,
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
    )
    if slot_scratch_tightening_result:
        results.append(slot_scratch_tightening_result)
    
    # Print summary
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS")
    print("="*80)
    
    if len(results) >= 2:
        baseline = results[0]
        
        print(f"\nBaseline (tightening disabled):")
        print(f"  Makespan: {baseline['makespan']}")
        print(f"  Elapsed time: {baseline['elapsed_time']:.1f}s")
        
        if len(results) >= 2:
            slot = results[1]
            print(f"\nSlot-aware tightening:")
            print(f"  Makespan: {slot['makespan']}")
            print(f"  Elapsed time: {slot['elapsed_time']:.1f}s")
            
            if baseline['makespan'] and slot['makespan']:
                makespan_diff = slot['makespan'] - baseline['makespan']
                makespan_pct = (makespan_diff / baseline['makespan']) * 100
                print(f"  vs baseline: {makespan_diff:+d} cycles ({makespan_pct:+.1f}%)")
        
        if len(results) >= 3:
            scratch = results[2]
            print(f"\nScratch-aware tightening:")
            print(f"  Makespan: {scratch['makespan']}")
            print(f"  Elapsed time: {scratch['elapsed_time']:.1f}s")
            
            if baseline['makespan'] and scratch['makespan']:
                makespan_diff = scratch['makespan'] - baseline['makespan']
                makespan_pct = (makespan_diff / baseline['makespan']) * 100
                print(f"  vs baseline: {makespan_diff:+d} cycles ({makespan_pct:+.1f}%)")
        
        if len(results) >= 4:
            slot_scratch = results[3]
            print(f"\nSlot + scratch tightening:")
            print(f"  Makespan: {slot_scratch['makespan']}")
            print(f"  Elapsed time: {slot_scratch['elapsed_time']:.1f}s")
            
            if baseline['makespan'] and slot_scratch['makespan']:
                makespan_diff = slot_scratch['makespan'] - baseline['makespan']
                makespan_pct = (makespan_diff / baseline['makespan']) * 100
                print(f"  vs baseline: {makespan_diff:+d} cycles ({makespan_pct:+.1f}%)")
            
            if len(results) >= 2 and slot['makespan'] and slot_scratch['makespan']:
                makespan_diff = slot_scratch['makespan'] - slot['makespan']
                makespan_pct = (makespan_diff / slot['makespan']) * 100 if slot['makespan'] > 0 else 0
                print(f"  vs slot-only: {makespan_diff:+d} cycles ({makespan_pct:+.1f}%)")
            
            if len(results) >= 3 and scratch['makespan'] and slot_scratch['makespan']:
                makespan_diff = slot_scratch['makespan'] - scratch['makespan']
                makespan_pct = (makespan_diff / scratch['makespan']) * 100 if scratch['makespan'] > 0 else 0
                print(f"  vs scratch-only: {makespan_diff:+d} cycles ({makespan_pct:+.1f}%)")
    
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
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
