#!/usr/bin/env python3
"""Compute theoretical lower bounds for kernel makespan.

Analyzes scheduler output JSON to compute capacity-based lower bounds,
accounting for:
- Engine slot capacities (ALU 12, VALU 6, Load 2, Store 2, Flow 1)
- Non-shiftable VALU ops (FMA/multiply_add and vbroadcast)
- Optimal VALU→ALU shifting
- SelectChoice and ConstChoice optimizations

Theoretical minimum assumes perfect scheduling with no dependencies or
scratch constraints - real schedules will be higher due to precedence
and scratch pressure.
"""

import argparse
import json
import sys
from typing import Dict, List, Tuple


def count_ops_by_engine(ops: List[Dict]) -> Dict[str, int]:
    """Count ops by engine type."""
    counts = {}
    for op in ops:
        engine = op.get("engine", "").lower()
        counts[engine] = counts.get(engine, 0) + 1
    return counts


def analyze_valu_ops(ops: List[Dict]) -> Tuple[int, int, Dict[str, int]]:
    """Analyze VALU ops to identify shiftable vs non-shiftable.
    
    Returns:
        (shiftable_count, non_shiftable_count, opcode_counts)
    """
    opcode_counts = {}
    shiftable = 0
    non_shiftable = 0
    
    for op in ops:
        if op.get("engine", "").lower() == "valu":
            opcode = op.get("opcode", "").lower()
            opcode_counts[opcode] = opcode_counts.get(opcode, 0) + 1
            
            # FMA (multiply_add) and vbroadcast cannot be shifted
            if opcode in ["multiply_add", "vbroadcast"]:
                non_shiftable += 1
            else:
                shiftable += 1
    
    return shiftable, non_shiftable, opcode_counts


def compute_capacity_bound(count: int, capacity: int) -> int:
    """Compute ceil(count / capacity)."""
    if count <= 0:
        return 0
    return (count + capacity - 1) // capacity


def find_optimal_shift(
    shiftable_valu: int,
    non_shiftable_valu: int,
    alu_count: int,
    load_count: int,
    store_count: int,
    flow_count: int,
    capacities: Dict[str, int],
) -> Tuple[int, int, Dict[str, int]]:
    """Find optimal VALU→ALU shift amount to minimize makespan.
    
    Returns:
        (best_shift, best_makespan, best_bounds)
    """
    # Approximate optimal shift: balance VALU and ALU bounds
    # Solve: ceil((valu - shift)/6) ≈ ceil((alu + 8*shift)/12)
    # Approximate: shift ≈ (2*shiftable_valu - alu_count) / 10
    optimal_shift_approx = max(0, (2 * shiftable_valu - alu_count) // 10)
    optimal_shift_approx = min(optimal_shift_approx, shiftable_valu)
    
    best_makespan = float("inf")
    best_shift = 0
    best_bounds = {}
    
    # Search around approximate optimal point
    search_range = 50
    for shift in range(
        max(0, optimal_shift_approx - search_range),
        min(shiftable_valu + 1, optimal_shift_approx + search_range),
    ):
        # VALU: non-shiftable + (shiftable - shifted)
        valu_remaining = non_shiftable_valu + (shiftable_valu - shift)
        # ALU: original + 8*shifted (each shifted VALU becomes 8 ALU ops)
        alu_total = alu_count + 8 * shift
        
        valu_lb = compute_capacity_bound(valu_remaining, capacities["valu"])
        alu_lb = compute_capacity_bound(alu_total, capacities["alu"])
        load_lb = compute_capacity_bound(load_count, capacities["load"])
        store_lb = compute_capacity_bound(store_count, capacities["store"])
        flow_lb = flow_count  # Flow capacity is 1
        
        makespan = max(valu_lb, alu_lb, load_lb, store_lb, flow_lb)
        if makespan < best_makespan:
            best_makespan = makespan
            best_shift = shift
            best_bounds = {
                "VALU": valu_lb,
                "ALU": alu_lb,
                "Load": load_lb,
                "Store": store_lb,
                "Flow": flow_lb,
            }
    
    return best_shift, best_makespan, best_bounds


def main():
    parser = argparse.ArgumentParser(
        description="Compute theoretical lower bounds for kernel makespan"
    )
    parser.add_argument(
        "json_file",
        nargs="?",
        default="scheduler_output.json",
        help="Scheduler output JSON file (default: scheduler_output.json)",
    )
    args = parser.parse_args()
    
    try:
        with open(args.json_file) as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {args.json_file}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Get ops
    ops = data.get("ssa_ops", [])
    if not ops:
        ops = data.get("ops", [])
    
    if not ops:
        print("Error: No ops found in JSON file", file=sys.stderr)
        sys.exit(1)
    
    # Count ops by engine
    engine_counts = count_ops_by_engine(ops)
    valu_count = engine_counts.get("valu", 0)
    alu_count = engine_counts.get("alu", 0)
    load_count = engine_counts.get("load", 0)
    store_count = engine_counts.get("store", 0)
    flow_count = engine_counts.get("flow", 0)
    
    # Count ConstChoice and SelectChoice ops from annotations
    # First try to use annotation fields from lowered ops
    ops_with_const_choice = [op for op in ops if op.get("from_const_choice") is not None]
    ops_with_select_choice = [op for op in ops if op.get("from_select_choice") is not None]
    
    # Get actual decisions from schedule (fallback if annotations not available)
    const_add_imm_ops = set(data.get("const_add_imm_ops", []))
    select_fma_ops = set(data.get("select_fma_ops", []))
    
    # Count by encoding: use annotations if available, otherwise fall back to SSA op counts
    if ops_with_const_choice:
        # Count unique ConstChoice source IDs
        const_choice_ids = set(op.get("from_const_choice") for op in ops_with_const_choice if op.get("from_const_choice") is not None)
        const_choice_count = len(const_choice_ids)
        
        # Count AddImm vs Load/ALU-derived from lowered ops
        const_add_imm_count = sum(1 for op in ops_with_const_choice if op.get("engine", "").lower() == "flow" and op.get("opcode", "").lower() == "add_imm")
        const_load_count = sum(1 for op in ops_with_const_choice if op.get("engine", "").lower() == "load" and op.get("opcode", "").lower() == "const")
        const_alu_derive_count = sum(1 for op in ops_with_const_choice if op.get("engine", "").lower() == "alu" and op.get("from_const_choice") is not None)
    else:
        # Fallback: count SSA ops
        const_choice_count = sum(1 for op in ops if op.get("opcode", "").lower() == "constchoice")
        const_add_imm_count = len(const_add_imm_ops)
        const_load_count = const_choice_count - const_add_imm_count
        const_alu_derive_count = 0
    
    if ops_with_select_choice:
        # Count unique SelectChoice source IDs
        select_choice_ids = set(op.get("from_select_choice") for op in ops_with_select_choice if op.get("from_select_choice") is not None)
        select_choice_count = len(select_choice_ids)
        
        # Count FMA vs vselect from lowered ops
        # FMA path produces both sub (-) and multiply_add ops, but we count unique SelectChoice sources
        select_fma_count = len(select_choice_ids) - sum(1 for op in ops_with_select_choice if op.get("opcode", "").lower() == "vselect")
        select_vselect_count = sum(1 for op in ops_with_select_choice if op.get("opcode", "").lower() == "vselect")
    else:
        # Fallback: count SSA ops
        select_choice_count = sum(1 for op in ops if op.get("opcode", "").lower() == "selectchoice")
        select_fma_count = len(select_fma_ops)
        select_vselect_count = select_choice_count - select_fma_count
    
    # Analyze VALU ops
    shiftable_valu, non_shiftable_valu, valu_opcodes = analyze_valu_ops(ops)
    
    # Get actual cycles
    actual_cycles = (
        data.get("total_cycles")
        or data.get("cycles")
        or data.get("makespan")
        or data.get("final_metrics", {}).get("cycles")
    )
    
    # Capacities
    capacities = {
        "valu": 6,
        "alu": 12,
        "load": 2,
        "store": 2,
        "flow": 1,
    }
    
    # Find optimal shift
    best_shift, best_makespan, best_bounds = find_optimal_shift(
        shiftable_valu,
        non_shiftable_valu,
        alu_count,
        load_count,
        store_count,
        flow_count,
        capacities,
    )
    
    # Print results
    print("=" * 70)
    print("Theoretical Lower Bounds Analysis")
    print("=" * 70)
    
    if actual_cycles:
        print(f"\nActual schedule: {actual_cycles} cycles")
    
    print(f"\nOp counts by engine:")
    for engine in ["valu", "alu", "load", "store", "flow"]:
        count = engine_counts.get(engine, 0)
        if count > 0:
            print(f"  {engine.upper()}: {count}")
    
    if const_choice_count > 0 or select_choice_count > 0:
        print(f"\nChoice decisions:")
        if const_choice_count > 0:
            print(f"  ConstChoice: {const_choice_count} total")
            print(f"    → AddImm (Flow): {const_add_imm_count}")
            print(f"    → Load: {const_load_count}")
            if const_alu_derive_count > 0:
                print(f"    → ALU-derived: {const_alu_derive_count}")
        if select_choice_count > 0:
            print(f"  SelectChoice: {select_choice_count} total")
            print(f"    → FMA path (VALU sub+fma): {select_fma_count}")
            print(f"    → vselect (Flow): {select_vselect_count}")
    
    print(f"\nVALU ops breakdown:")
    print(f"  Total: {valu_count}")
    print(f"  Shiftable: {shiftable_valu}")
    print(f"  Non-shiftable: {non_shiftable_valu} (FMA + vbroadcast)")
    
    if valu_opcodes:
        print(f"\nVALU ops by opcode:")
        for opcode, count in sorted(valu_opcodes.items()):
            shiftable = " (shiftable)" if opcode not in ["multiply_add", "vbroadcast"] else " (non-shiftable)"
            print(f"  {opcode}: {count}{shiftable}")
    
    # Current state (no shift)
    no_shift_valu_lb = compute_capacity_bound(valu_count, capacities["valu"])
    no_shift_alu_lb = compute_capacity_bound(alu_count, capacities["alu"])
    no_shift_load_lb = compute_capacity_bound(load_count, capacities["load"])
    no_shift_store_lb = compute_capacity_bound(store_count, capacities["store"])
    no_shift_makespan = max(
        no_shift_valu_lb,
        no_shift_alu_lb,
        no_shift_load_lb,
        no_shift_store_lb,
        flow_count,
    )
    
    print(f"\n{'=' * 70}")
    print("Current State (No VALU Shift)")
    print(f"{'=' * 70}")
    print(f"  VALU: ceil({valu_count}/6) = {no_shift_valu_lb} cycles")
    print(f"  ALU: ceil({alu_count}/12) = {no_shift_alu_lb} cycles")
    print(f"  Load: ceil({load_count}/2) = {no_shift_load_lb} cycles")
    print(f"  Store: ceil({store_count}/2) = {no_shift_store_lb} cycles")
    print(f"  Flow: {flow_count} cycles (capacity 1)")
    print(f"\n  Makespan: {no_shift_makespan} cycles")
    print(f"  Bottleneck: {[k for k, v in {'VALU': no_shift_valu_lb, 'ALU': no_shift_alu_lb, 'Load': no_shift_load_lb, 'Store': no_shift_store_lb, 'Flow': flow_count}.items() if v == no_shift_makespan]}")
    
    # Optimal shift
    valu_remaining = non_shiftable_valu + (shiftable_valu - best_shift)
    alu_total = alu_count + 8 * best_shift
    
    print(f"\n{'=' * 70}")
    print(f"Theoretical Minimum (Optimal VALU Shift: {best_shift} ops)")
    print(f"{'=' * 70}")
    print(f"  VALU: {valu_remaining} ops ({non_shiftable_valu} non-shiftable + {shiftable_valu - best_shift} shiftable remaining)")
    print(f"        → ceil({valu_remaining}/6) = {best_bounds['VALU']} cycles")
    print(f"  ALU: {alu_total} ops ({alu_count} original + 8×{best_shift} shifted)")
    print(f"       → ceil({alu_total}/12) = {best_bounds['ALU']} cycles")
    print(f"  Load: {load_count} ops → ceil({load_count}/2) = {best_bounds['Load']} cycles")
    print(f"  Store: {store_count} ops → ceil({store_count}/2) = {best_bounds['Store']} cycles")
    print(f"  Flow: {flow_count} ops → {best_bounds['Flow']} cycles")
    
    print(f"\n  Theoretical minimum makespan: {best_makespan} cycles")
    bottlenecks = [k for k, v in best_bounds.items() if v == best_makespan]
    print(f"  Bottleneck: {bottlenecks}")
    
    # Calculate slot utilization
    print(f"\n{'=' * 70}")
    print("Slot Utilization (Theoretical Minimum)")
    print(f"{'=' * 70}")
    
    valu_remaining = non_shiftable_valu + (shiftable_valu - best_shift)
    alu_total = alu_count + 8 * best_shift
    
    engines_info = [
        ("VALU", valu_remaining, capacities["valu"], best_makespan),
        ("ALU", alu_total, capacities["alu"], best_makespan),
        ("Load", load_count, capacities["load"], best_makespan),
        ("Store", store_count, capacities["store"], best_makespan),
        ("Flow", flow_count, capacities["flow"], best_makespan),
    ]
    
    print(f"{'Engine':<10} {'Ops':<8} {'Capacity':<10} {'Makespan':<10} {'Total Slots':<12} {'Used Slots':<12} {'Free Slots':<12} {'Utilization':<12}")
    print("-" * 90)
    
    for engine, ops_count, capacity, makespan in engines_info:
        total_slots = capacity * makespan
        used_slots = ops_count
        free_slots = total_slots - used_slots
        utilization_pct = 100.0 * used_slots / total_slots if total_slots > 0 else 0.0
        
        print(
            f"{engine:<10} {ops_count:<8} {capacity:<10} {makespan:<10} "
            f"{total_slots:<12} {used_slots:<12} {free_slots:<12} {utilization_pct:>6.1f}%"
        )
    
    if actual_cycles:
        gap = actual_cycles - best_makespan
        overhead_pct = 100 * gap / actual_cycles
        print(f"\n{'=' * 70}")
        print("Comparison")
        print(f"{'=' * 70}")
        print(f"  Actual schedule: {actual_cycles} cycles")
        print(f"  Theoretical minimum: {best_makespan} cycles")
        print(f"  Gap: {gap} cycles ({overhead_pct:.1f}% overhead)")
        print(f"\n  Note: Gap accounts for:")
        print(f"    - Precedence constraints (dependencies)")
        print(f"    - Scratch memory pressure")
        print(f"    - Scheduling inefficiencies/gaps")
        print(f"    - Other constraints preventing perfect packing")


if __name__ == "__main__":
    main()
