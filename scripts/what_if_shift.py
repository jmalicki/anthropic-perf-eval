#!/usr/bin/env python3
"""
What-if analysis tool: Force a specific VALU op to shift and re-optimize.

Usage:
    uv run python scripts/what_if_shift.py --op-id 244 --cpsat-time-limit 600
    uv run python scripts/what_if_shift.py --op-id 244 --force-shift false  # Force NOT to shift
    uv run python scripts/what_if_shift.py --op-id 244 --compare-with scheduler_output_fixed.json
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import scheduler_rs


def load_baseline_json(json_path: str):
    """Load baseline JSON for comparison."""
    with open(json_path) as f:
        return json.load(f)


def compare_schedules(baseline_ops, whatif_ops, baseline_stats, whatif_stats):
    """Compare two schedules and show differences."""
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    
    # Cycles comparison
    baseline_cycles = baseline_stats.get('cycles', 0)
    whatif_cycles = whatif_stats.get('cycles', 0)
    cycle_diff = whatif_cycles - baseline_cycles
    print(f"\nCycles:")
    print(f"  Baseline: {baseline_cycles}")
    print(f"  What-if:  {whatif_cycles}")
    print(f"  Difference: {cycle_diff:+d} ({cycle_diff/baseline_cycles*100:+.2f}%)")
    
    # Build op ID -> cycle mapping
    baseline_cycles_map = {op['id']: op.get('cycle', -1) for op in baseline_ops}
    whatif_cycles_map = {op['id']: op.get('cycle', -1) for op in whatif_ops}
    
    # Find ops that changed cycles
    changed_ops = []
    for op_id in set(baseline_cycles_map.keys()) | set(whatif_cycles_map.keys()):
        baseline_cycle = baseline_cycles_map.get(op_id, -1)
        whatif_cycle = whatif_cycles_map.get(op_id, -1)
        if baseline_cycle != whatif_cycle:
            changed_ops.append((op_id, baseline_cycle, whatif_cycle))
    
    if changed_ops:
        print(f"\nOps with changed cycles: {len(changed_ops)}")
        # Show first 20
        for op_id, baseline_cycle, whatif_cycle in sorted(changed_ops)[:20]:
            print(f"  Op {op_id}: {baseline_cycle} -> {whatif_cycle} ({whatif_cycle - baseline_cycle:+d})")
        if len(changed_ops) > 20:
            print(f"  ... and {len(changed_ops) - 20} more")
    
    # ALU/VALU utilization comparison
    baseline_alu = baseline_stats.get('usage_accounting', {}).get('alu_ops', 0)
    baseline_valu = baseline_stats.get('usage_accounting', {}).get('valu_ops', 0)
    whatif_alu = whatif_stats.get('usage_accounting', {}).get('alu_ops', 0)
    whatif_valu = whatif_stats.get('usage_accounting', {}).get('valu_ops', 0)
    
    print(f"\nALU/VALU ops:")
    print(f"  Baseline: ALU={baseline_alu}, VALU={baseline_valu}")
    print(f"  What-if:  ALU={whatif_alu}, VALU={whatif_valu}")
    print(f"  Difference: ALU={whatif_alu - baseline_alu:+d}, VALU={whatif_valu - baseline_valu:+d}")
    
    # Shifted ops comparison
    baseline_shifted = baseline_stats.get('valu_shifted', 0)
    whatif_shifted = whatif_stats.get('valu_shifted', 0)
    print(f"\nShifted VALU ops:")
    print(f"  Baseline: {baseline_shifted}")
    print(f"  What-if:  {whatif_shifted}")
    print(f"  Difference: {whatif_shifted - baseline_shifted:+d}")


def main():
    parser = argparse.ArgumentParser(
        description="What-if analysis: Force a shift decision and re-optimize"
    )
    parser.add_argument(
        "--op-id",
        type=int,
        required=True,
        help="VALU op ID to force shift decision for",
    )
    parser.add_argument(
        "--force-shift",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Force shift (True) or force NOT shift (False). Default: True",
    )
    parser.add_argument(
        "--cpsat-time-limit",
        type=float,
        default=600.0,
        help="CP-SAT time limit in seconds (default: 600)",
    )
    parser.add_argument(
        "--enable-valu-shift",
        action="store_true",
        default=True,
        help="Enable VALU shift optimization (default: True)",
    )
    parser.add_argument(
        "--no-valu-shift",
        action="store_false",
        dest="enable_valu_shift",
        help="Disable VALU shift optimization",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Number of rounds (default: 1)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size (default: 256)",
    )
    parser.add_argument(
        "--forest-height",
        type=int,
        default=10,
        help="Forest height (default: 10)",
    )
    parser.add_argument(
        "--n-nodes",
        type=int,
        default=2047,
        help="Number of nodes (default: 2047)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="scheduler_output_whatif.json",
        help="Output JSON file path (default: scheduler_output_whatif.json)",
    )
    parser.add_argument(
        "--compare-with",
        type=str,
        help="Compare with this JSON file (baseline solution)",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()

    # Build kernel with forced shift
    print(f"What-if analysis: Force Op {args.op_id} to shift={args.force_shift}")
    print("=" * 80)
    print(f"Parameters:")
    print(f"  forest_height: {args.forest_height}")
    print(f"  n_nodes: {args.n_nodes}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  rounds: {args.rounds}")
    print(f"  CP-SAT time limit: {args.cpsat_time_limit}s")
    print(f"  enable_valu_shift: {args.enable_valu_shift}")
    
    # Prepare forced_shifts parameter
    forced_shifts = None
    if args.force_shift:
        forced_shifts = [args.op_id]
        print(f"  Forcing shift: Op {args.op_id} MUST shift to ALU")
    else:
        print(f"  Note: Forcing NOT to shift requires different constraint (not yet implemented)")
        print(f"  This will run with normal optimization (Op {args.op_id} may or may not shift)")
    
    print()
    
    # Load baseline if provided
    baseline_data = None
    if args.compare_with:
        try:
            baseline_data = load_baseline_json(args.compare_with)
            print(f"Loaded baseline from: {args.compare_with}")
            baseline_cycles = baseline_data.get('total_cycles', 0)
            print(f"Baseline cycles: {baseline_cycles}")
        except Exception as e:
            print(f"Warning: Could not load baseline: {e}")
            baseline_data = None
    
    # Build kernel
    try:
        instrs, stats = scheduler_rs.build_tick_tock_kernel_ssa(
            forest_height=args.forest_height,
            n_nodes=args.n_nodes,
            batch_size=args.batch_size,
            rounds=args.rounds,
            enable_valu_shift=args.enable_valu_shift,
            verbose=not args.quiet,
            use_cpsat=True,
            cpsat_time_limit=args.cpsat_time_limit,
            per_section_optimize_secs=None,
            output_json=args.output_json,
            tree_reduce_max_depth=2,
            liveness_detail="summary",
            forced_shifts=forced_shifts,
        )
        
        cycles = stats.get('cycles', 0)
        print(f"\nWhat-if result: {cycles} cycles")
        
        # Verify forced shift was applied
        if forced_shifts:
            # Check if op was actually shifted by looking at the output JSON
            try:
                with open(args.output_json) as f:
                    whatif_data = json.load(f)
                    whatif_ops = whatif_data.get('ops', [])
                    
                    # Find the op in the output
                    op_found = False
                    for op in whatif_ops:
                        if op.get('id') == args.op_id:
                            op_found = True
                            engine = op.get('engine')
                            if engine == 'alu':
                                print(f"✓ Op {args.op_id} was shifted to ALU (as forced)")
                            elif engine == 'valu':
                                print(f"✗ Op {args.op_id} is still VALU (forced shift may have failed)")
                            break
                    
                    if not op_found:
                        print(f"⚠ Op {args.op_id} not found in output (may have been eliminated)")
                    
                    # Compare with baseline if provided
                    if baseline_data:
                        baseline_ops = baseline_data.get('ops', [])
                        baseline_stats = {
                            'cycles': baseline_data.get('total_cycles', 0),
                            'valu_shifted': baseline_data.get('valu_shifted', 0),
                            'usage_accounting': baseline_data.get('usage_accounting', {}),
                        }
                        whatif_stats = {
                            'cycles': whatif_data.get('total_cycles', 0),
                            'valu_shifted': whatif_data.get('valu_shifted', 0),
                            'usage_accounting': whatif_data.get('usage_accounting', {}),
                        }
                        compare_schedules(baseline_ops, whatif_ops, baseline_stats, whatif_stats)
            except Exception as e:
                print(f"Warning: Could not analyze output JSON: {e}")
        
        print(f"\nOutput written to: {args.output_json}")
        return 0
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
