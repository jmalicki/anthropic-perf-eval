#!/usr/bin/env python3
"""
Greedily optimize schedule by:
1. Moving Load const ops to add_imm (Flow) when Flow is free
2. Moving other Load ops earlier to fill freed Load slots
3. Moving vbroadcasts earlier when their consts move earlier
4. Moving other VALU ops earlier

Goal: Free a whole cycle in the first decile (cycles 0-19).
"""

import json
import sys
from collections import defaultdict

# Slot limits
SLOT_LIMITS = {
    "alu": 12,
    "valu": 6,
    "load": 2,
    "store": 2,
    "flow": 1,
}


def load_schedule(json_path):
    """Load scheduler output JSON."""
    with open(json_path) as f:
        return json.load(f)


def build_indexes(ops):
    """Build indexes for fast lookup."""
    id_to_op = {}
    cycle_to_ops = defaultdict(list)
    for op in ops:
        op_id = op.get("id", op.get("op_id"))
        cycle = op.get("cycle", -1)
        id_to_op[op_id] = op
        if cycle >= 0:
            cycle_to_ops[cycle].append(op)
    return id_to_op, cycle_to_ops


def get_slot_usage(ops_at_cycle, engine):
    """Get slot usage for an engine at a cycle."""
    return len([o for o in ops_at_cycle if o.get("engine", "").lower() == engine.lower()])


def can_move_op_earlier(op, target_cycle, id_to_op, cycle_to_ops, slot_limits):
    """Check if an op can move to target_cycle."""
    if target_cycle < 0:
        return False
    
    # Check dependencies
    deps = op.get("dep_ids", [])
    for dep_id in deps:
        dep_op = id_to_op.get(dep_id)
        if dep_op:
            dep_cycle = dep_op.get("cycle", -1)
            if dep_cycle >= target_cycle:
                return False  # Dependency not satisfied
    
    # Check slot availability
    eng = op.get("engine", "").lower()
    cap = slot_limits.get(eng, 0)
    if cap == 0:
        return False
    
    ops_at_target = cycle_to_ops.get(target_cycle, [])
    used = get_slot_usage(ops_at_target, eng)
    
    return used < cap


def move_op(op, new_cycle, cycle_to_ops):
    """Move an op to a new cycle."""
    old_cycle = op.get("cycle", -1)
    if old_cycle >= 0:
        cycle_to_ops[old_cycle].remove(op)
    op["cycle"] = new_cycle
    cycle_to_ops[new_cycle].append(op)


def move_consumers_earlier(op_id, id_to_op, cycle_to_ops, first_decile_end, changes):
    """Move all consumers of an op earlier if possible."""
    consumers = []
    for other_op in id_to_op.values():
        deps = other_op.get("dep_ids", [])
        if op_id in deps:
            consumers.append(other_op)
    
    moved_count = 0
    for consumer in consumers:
        current_cycle = consumer.get("cycle", -1)
        if current_cycle < 0 or current_cycle > first_decile_end:
            continue
        
        # Find earliest possible cycle based on all deps
        deps = consumer.get("dep_ids", [])
        earliest_dep_cycle = -1
        for dep_id in deps:
            dep_op = id_to_op.get(dep_id)
            if dep_op:
                dep_cycle = dep_op.get("cycle", -1)
                earliest_dep_cycle = max(earliest_dep_cycle, dep_cycle)
        
        earliest_possible = max(0, earliest_dep_cycle + 1)
        
        # Try to move earlier
        for target_cycle in range(earliest_possible, current_cycle):
            if can_move_op_earlier(
                consumer, target_cycle, id_to_op, cycle_to_ops, SLOT_LIMITS
            ):
                move_op(consumer, target_cycle, cycle_to_ops)
                changes.append(
                    (
                        "consumer earlier",
                        consumer.get("id"),
                        current_cycle,
                        target_cycle,
                    )
                )
                moved_count += 1
                eng = consumer.get("engine", "").lower()
                opcode = (consumer.get("opcode") or "").strip()
                print(
                    f"    -> Moved consumer {eng} {opcode} id={consumer.get('id')} from cycle {current_cycle} to cycle {target_cycle}"
                )
                break
    
    return moved_count


def greedy_optimize(ops, id_to_op, cycle_to_ops, first_decile_end=19):
    """Greedily optimize by moving const→add_imm and shifting ops earlier."""
    changes = []
    max_cycle = max((op.get("cycle", -1) for op in ops), default=0)
    
    print(f"Initial makespan: {max_cycle + 1} cycles")
    print(f"First decile: cycles 0-{first_decile_end}")
    print()
    
    # Multiple passes to cascade optimizations
    max_passes = 5
    for pass_num in range(max_passes):
        if pass_num > 0:
            print(f"\n{'='*70}")
            print(f"PASS {pass_num + 1}")
            print(f"{'='*70}\n")
        
        pass_changes = len(changes)
        
            # Process in reverse order so we move later consts first
        print("Step 1: Moving const ops to add_imm (Flow)...")
        const_moved = 0
    for cycle in range(first_decile_end, -1, -1):  # Reverse order
        ops_at_cycle = cycle_to_ops.get(cycle, [])
        const_ops = [
            o for o in ops_at_cycle
            if o.get("engine", "").lower() == "load"
            and (o.get("opcode") or "").strip() == "const"
        ]
        
        for const_op in const_ops:
            # Try to move to earlier cycle via add_imm (Flow)
            # Find earliest cycle where Flow is free and const has no deps blocking it
            const_deps = const_op.get("dep_ids", [])
            earliest_dep_cycle = -1
            for dep_id in const_deps:
                dep_op = id_to_op.get(dep_id)
                if dep_op:
                    dep_cycle = dep_op.get("cycle", -1)
                    earliest_dep_cycle = max(earliest_dep_cycle, dep_cycle)
            
            # Const can run at earliest_dep_cycle + 1, or cycle 0 if no deps
            earliest_possible = max(0, earliest_dep_cycle + 1)
            
            # Try cycles from earliest_possible to current cycle
            for target_cycle in range(earliest_possible, cycle):
                ops_at_target = cycle_to_ops.get(target_cycle, [])
                flow_used = get_slot_usage(ops_at_target, "flow")
                
                if flow_used < SLOT_LIMITS["flow"]:
                    # Move const to add_imm at target_cycle
                    const_op["engine"] = "Flow"
                    const_op["opcode"] = "add_imm"
                    move_op(const_op, target_cycle, cycle_to_ops)
                    changes.append(
                        (
                            "const→add_imm",
                            const_op.get("id"),
                            cycle,
                            target_cycle,
                        )
                    )
                    const_moved += 1
                    const_id = const_op.get("id")
                    print(
                        f"  Moved const id={const_id} from cycle {cycle} (Load) to cycle {target_cycle} (Flow/add_imm)"
                    )
                    # Immediately try to move consumers earlier
                    moved_consumers = move_consumers_earlier(
                        const_id, id_to_op, cycle_to_ops, first_decile_end, changes
                    )
                    if moved_consumers > 0:
                        print(f"    Moved {moved_consumers} consumers earlier")
                    
                    # Check if we freed a cycle
                    current_max = max((op.get("cycle", -1) for op in ops), default=0)
                    if current_max < max_cycle:
                        print(f"\n✓ FREED A CYCLE! Makespan: {max_cycle + 1} → {current_max + 1}")
                        return changes, max_cycle - current_max
                    break
    
        print(f"  Moved {const_moved} const ops to add_imm")
        print()
        
            # Check makespan after Step 1
        current_max = max((op.get("cycle", -1) for op in ops), default=0)
        if current_max < max_cycle:
            print(f"\n✓ FREED A CYCLE after Step 1! Makespan: {max_cycle + 1} → {current_max + 1}")
            return changes, max_cycle - current_max
        
            # Process in reverse order
        print("Step 2: Moving other Load ops earlier...")
        load_moved = 0
    for cycle in range(max_cycle, first_decile_end, -1):  # Reverse order
        ops_at_cycle = cycle_to_ops.get(cycle, [])
        load_ops = [
            o
            for o in ops_at_cycle
            if o.get("engine", "").lower() == "load"
            and (o.get("opcode") or "").strip() != "const"
        ]
        
        for load_op in load_ops:
            # Try to move to earliest possible cycle in first decile
            deps = load_op.get("dep_ids", [])
            earliest_dep_cycle = -1
            for dep_id in deps:
                dep_op = id_to_op.get(dep_id)
                if dep_op:
                    dep_cycle = dep_op.get("cycle", -1)
                    earliest_dep_cycle = max(earliest_dep_cycle, dep_cycle)
            
            earliest_possible = max(0, earliest_dep_cycle + 1)
            
            # Try to move into first decile
            for target_cycle in range(earliest_possible, min(cycle, first_decile_end + 1)):
                if can_move_op_earlier(
                    load_op, target_cycle, id_to_op, cycle_to_ops, SLOT_LIMITS
                ):
                    move_op(load_op, target_cycle, cycle_to_ops)
                    changes.append(
                        ("load earlier", load_op.get("id"), cycle, target_cycle)
                    )
                    load_moved += 1
                    print(
                        f"  Moved load id={load_op.get('id')} from cycle {cycle} to cycle {target_cycle}"
                    )
                    # Check if we freed a cycle
                    current_max = max((op.get("cycle", -1) for op in ops), default=0)
                    if current_max < max_cycle:
                        print(f"\n✓ FREED A CYCLE! Makespan: {max_cycle + 1} → {current_max + 1}")
                        return changes, max_cycle - current_max
                    break
    
        print(f"  Moved {load_moved} Load ops earlier")
        print()
        
            # Process in reverse order
        print("Step 3: Moving vbroadcasts earlier...")
        vbroadcast_moved = 0
    for cycle in range(max_cycle, first_decile_end, -1):  # Reverse order
        ops_at_cycle = cycle_to_ops.get(cycle, [])
        vbroadcast_ops = [
            o
            for o in ops_at_cycle
            if o.get("engine", "").lower() == "valu"
            and (o.get("opcode") or "").strip() == "vbroadcast"
        ]
        
        for vbroadcast_op in vbroadcast_ops:
            deps = vbroadcast_op.get("dep_ids", [])
            earliest_dep_cycle = -1
            for dep_id in deps:
                dep_op = id_to_op.get(dep_id)
                if dep_op:
                    dep_cycle = dep_op.get("cycle", -1)
                    earliest_dep_cycle = max(earliest_dep_cycle, dep_cycle)
            
            earliest_possible = max(0, earliest_dep_cycle + 1)
            
            # Try to move into first decile
            for target_cycle in range(earliest_possible, min(cycle, first_decile_end + 1)):
                if can_move_op_earlier(
                    vbroadcast_op, target_cycle, id_to_op, cycle_to_ops, SLOT_LIMITS
                ):
                    move_op(vbroadcast_op, target_cycle, cycle_to_ops)
                    changes.append(
                        (
                            "vbroadcast earlier",
                            vbroadcast_op.get("id"),
                            cycle,
                            target_cycle,
                        )
                    )
                    vbroadcast_moved += 1
                    print(
                        f"  Moved vbroadcast id={vbroadcast_op.get('id')} from cycle {cycle} to cycle {target_cycle}"
                    )
                    # Check if we freed a cycle
                    current_max = max((op.get("cycle", -1) for op in ops), default=0)
                    if current_max < max_cycle:
                        print(f"\n✓ FREED A CYCLE! Makespan: {max_cycle + 1} → {current_max + 1}")
                        return changes, max_cycle - current_max
                    break
    
        print(f"  Moved {vbroadcast_moved} vbroadcast ops earlier")
        print()
        
            # Process in reverse order
        print("Step 4: Moving other VALU ops earlier...")
        valu_moved = 0
    for cycle in range(max_cycle, first_decile_end, -1):  # Reverse order
        ops_at_cycle = cycle_to_ops.get(cycle, [])
        valu_ops = [
            o
            for o in ops_at_cycle
            if o.get("engine", "").lower() == "valu"
            and (o.get("opcode") or "").strip() not in ("vbroadcast", "multiply_add", "-")
        ]
        
        for valu_op in valu_ops:
            deps = valu_op.get("dep_ids", [])
            earliest_dep_cycle = -1
            for dep_id in deps:
                dep_op = id_to_op.get(dep_id)
                if dep_op:
                    dep_cycle = dep_op.get("cycle", -1)
                    earliest_dep_cycle = max(earliest_dep_cycle, dep_cycle)
            
            earliest_possible = max(0, earliest_dep_cycle + 1)
            
            # Try to move into first decile
            for target_cycle in range(earliest_possible, min(cycle, first_decile_end + 1)):
                if can_move_op_earlier(
                    valu_op, target_cycle, id_to_op, cycle_to_ops, SLOT_LIMITS
                ):
                    move_op(valu_op, target_cycle, cycle_to_ops)
                    changes.append(
                        ("valu earlier", valu_op.get("id"), cycle, target_cycle)
                    )
                    valu_moved += 1
                    print(
                        f"  Moved VALU id={valu_op.get('id')} from cycle {cycle} to cycle {target_cycle}"
                    )
                    # Check if we freed a cycle
                    current_max = max((op.get("cycle", -1) for op in ops), default=0)
                    if current_max < max_cycle:
                        print(f"\n✓ FREED A CYCLE! Makespan: {max_cycle + 1} → {current_max + 1}")
                        return changes, max_cycle - current_max
                    break
    
        print(f"  Moved {valu_moved} other VALU ops earlier")
        print()
        
        # Check if we made progress this pass
        if len(changes) == pass_changes:
            print("No more changes possible in this pass")
            break
        
        # Check if we freed a cycle
        current_max = max((op.get("cycle", -1) for op in ops), default=0)
        if current_max < max_cycle:
            print(f"\n✓ FREED A CYCLE! Makespan: {max_cycle + 1} → {current_max + 1}")
            return changes, max_cycle - current_max
        
        # Check if we emptied first decile end
        ops_at_decile_end = cycle_to_ops.get(first_decile_end, [])
        if len(ops_at_decile_end) == 0:
            print(f"\n✓ EMPTIED CYCLE {first_decile_end} (freed a cycle in first decile)!")
            return changes, 1
    
    # Final check if we freed a cycle
    new_max_cycle = max((op.get("cycle", -1) for op in ops), default=0)
    cycles_saved = max_cycle - new_max_cycle
    
    # Also check if we emptied the last cycle of first decile
    ops_at_decile_end = cycle_to_ops.get(first_decile_end, [])
    decile_end_emptied = len(ops_at_decile_end) == 0
    
    print("=" * 70)
    print(f"Final makespan: {new_max_cycle + 1} cycles")
    print(f"Cycles saved: {cycles_saved}")
    print(f"Ops at cycle {first_decile_end} (first decile end): {len(ops_at_decile_end)}")
    print(f"Total changes: {len(changes)}")
    
    if cycles_saved > 0:
        print(f"\n✓ SUCCESS: Freed {cycles_saved} cycle(s)!")
        return changes, cycles_saved
    elif decile_end_emptied:
        print(f"\n✓ SUCCESS: Emptied cycle {first_decile_end} (freed a cycle in first decile)!")
        return changes, 1  # Consider it a success
    else:
        print("\n✗ No cycles freed (but may have improved load balance)")
        return changes, 0


def main():
    if len(sys.argv) < 2:
        print("Usage: python greedy_const_to_add_imm.py <scheduler_output.json>")
        sys.exit(1)
    
    json_path = sys.argv[1]
    print(f"Loading schedule from {json_path}...")
    data = load_schedule(json_path)
    
    ops = data.get("ops", [])
    print(f"Loaded {len(ops)} ops")
    print()
    
    id_to_op, cycle_to_ops = build_indexes(ops)
    
    changes, cycles_saved = greedy_optimize(ops, id_to_op, cycle_to_ops)
    
    if cycles_saved > 0:
        # Save optimized schedule
        output_path = json_path.replace(".json", "_optimized.json")
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nOptimized schedule saved to {output_path}")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
