//! Core scheduling algorithms.
//!
//! This module contains the greedy list scheduler with multiple priority modes
//! and support for VALU→ALU shifting when VALU slots are saturated.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

use crate::diagnostics::SchedulingDiagnostics;
use crate::types::{
    can_convert_valu_to_alu, expand_valu_to_alu_smart, get_slot_limit, Op, SlotVal, VLEN,
};

/// Result of scheduling
pub struct ScheduleResult {
    pub cycles: usize,
    #[allow(dead_code)]
    pub assignments: Vec<(usize, usize)>,
    pub diagnostics: Option<SchedulingDiagnostics>,
    /// Expanded ALU ops from VALU→ALU shifting (these replace shifted VALU ops)
    pub expanded_ops: Vec<Op>,
}

/// Priority mode for scheduling
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PriorityMode {
    CriticalPath,     // Prioritize ops on critical path
    CriticalPathValu, // Prioritize VALU first, then critical path
    EngineBalance,    // Balance engine utilization
    Slack,            // Prioritize low-slack ops (standard compiler scheduling)
    LoadEarly,        // Prioritize loads
    LoadEarlySlack,   // Combine load-early with slack
    OriginalOrder,    // Schedule in original emission order
}

pub const ALL_PRIORITY_MODES: [PriorityMode; 7] = [
    PriorityMode::CriticalPath,
    PriorityMode::CriticalPathValu,
    PriorityMode::EngineBalance,
    PriorityMode::Slack,
    PriorityMode::LoadEarly,
    PriorityMode::LoadEarlySlack,
    PriorityMode::OriginalOrder,
];

/// Compute critical path length for each op (longest path to any sink)
pub fn compute_critical_path(ops: &[Op]) -> HashMap<usize, usize> {
    let mut critical_path: HashMap<usize, usize> = HashMap::new();

    // Build successor lists
    let mut successors: HashMap<usize, Vec<usize>> = HashMap::new();
    for op in ops {
        successors.insert(op.id, Vec::new());
    }
    for op in ops {
        for &dep_id in &op.dep_ids {
            if let Some(succ) = successors.get_mut(&dep_id) {
                succ.push(op.id);
            }
        }
    }

    // Start from sinks (ops with no successors)
    let mut work_queue: VecDeque<usize> = VecDeque::new();
    for op in ops {
        if successors.get(&op.id).map(|s| s.is_empty()).unwrap_or(true) {
            work_queue.push_back(op.id);
            critical_path.insert(op.id, 1);
        }
    }

    // Build out-degree for reverse topo processing
    let mut out_degree: HashMap<usize, usize> = HashMap::new();
    for op in ops {
        out_degree.insert(
            op.id,
            successors.get(&op.id).map(|s| s.len()).unwrap_or(0),
        );
    }

    // Build op lookup
    let op_by_id: HashMap<usize, &Op> = ops.iter().map(|op| (op.id, op)).collect();

    // Process in reverse order
    let mut remaining_out = out_degree.clone();
    while let Some(op_id) = work_queue.pop_front() {
        let my_cp = *critical_path.get(&op_id).unwrap_or(&1);

        if let Some(op) = op_by_id.get(&op_id) {
            for &dep_id in &op.dep_ids {
                let new_cp = my_cp + 1;
                let current = critical_path.entry(dep_id).or_insert(0);
                if new_cp > *current {
                    *current = new_cp;
                }

                if let Some(rem) = remaining_out.get_mut(&dep_id) {
                    *rem -= 1;
                    if *rem == 0 {
                        work_queue.push_back(dep_id);
                    }
                }
            }
        }
    }

    critical_path
}

/// Priority for scheduling (lower tuple = higher priority in min-heap with Reverse)
fn compute_priority(
    op_id: usize,
    op_idx: usize,
    engine: &str,
    critical_path: &HashMap<usize, usize>,
    slack_map: &HashMap<usize, i32>,
    mode: PriorityMode,
) -> (i32, i32, usize) {
    let cp = *critical_path.get(&op_id).unwrap_or(&1) as i32;
    let slack = *slack_map.get(&op_id).unwrap_or(&0);
    let is_valu = if engine == "valu" { 0 } else { 1 };
    let is_load = if engine == "load" { 0 } else { 1 };

    match mode {
        PriorityMode::CriticalPath => (-cp, is_valu, op_id),
        PriorityMode::CriticalPathValu => (is_valu, -cp, op_id),
        PriorityMode::EngineBalance => {
            // Matches Python: -cp * load_factor where load_factor is 0.6, 0.8, 0.5, 1.0
            // Use float conversion to match Python's rounding behavior
            let load_factor: f64 = match engine {
                "valu" => 0.6,
                "load" | "store" => 0.8,
                "flow" => 0.5,
                _ => 1.0, // ALU and others
            };
            let adjusted_cp = (-(cp as f64) * load_factor) as i32;
            (adjusted_cp, op_id as i32, op_id)
        }
        PriorityMode::Slack => (slack, -cp, op_id),
        PriorityMode::LoadEarly => (is_load, -cp, op_id),
        PriorityMode::LoadEarlySlack => (is_load, slack, op_id),
        PriorityMode::OriginalOrder => (op_idx as i32, is_valu, op_id),
    }
}

/// Compute ASAP (earliest possible cycle) for each op
fn compute_asap(ops: &[Op]) -> HashMap<usize, i32> {
    let mut asap: HashMap<usize, i32> = HashMap::new();

    // Build in-degree for topological sort
    let mut in_degree: HashMap<usize, usize> = HashMap::new();
    for op in ops {
        in_degree.insert(op.id, op.dep_ids.len());
    }

    let _op_by_id: HashMap<usize, &Op> = ops.iter().map(|op| (op.id, op)).collect();

    // Start with ops that have no dependencies
    let mut queue: VecDeque<usize> = VecDeque::new();
    for op in ops {
        if op.dep_ids.is_empty() {
            queue.push_back(op.id);
            asap.insert(op.id, 0);
        }
    }

    // Process in topological order
    while let Some(op_id) = queue.pop_front() {
        let my_asap = *asap.get(&op_id).unwrap_or(&0);

        // Find successors (ops that depend on this one)
        for other in ops {
            if other.dep_ids.contains(&op_id) {
                // Update successor's ASAP
                let new_asap = my_asap + 1;
                let current = asap.entry(other.id).or_insert(0);
                if new_asap > *current {
                    *current = new_asap;
                }

                // Decrement in-degree
                if let Some(deg) = in_degree.get_mut(&other.id) {
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push_back(other.id);
                    }
                }
            }
        }
    }

    asap
}

/// Compute ALAP (latest possible cycle) for each op given total cycles
fn compute_alap(ops: &[Op], total_cycles: i32) -> HashMap<usize, i32> {
    let mut alap: HashMap<usize, i32> = HashMap::new();

    // Build consumer map (ops that depend on each op)
    let mut consumers: HashMap<usize, Vec<usize>> = HashMap::new();
    for op in ops {
        consumers.insert(op.id, Vec::new());
    }
    for op in ops {
        for &dep_id in &op.dep_ids {
            if let Some(c) = consumers.get_mut(&dep_id) {
                c.push(op.id);
            }
        }
    }

    // Build out-degree
    let mut out_degree: HashMap<usize, usize> = HashMap::new();
    for op in ops {
        out_degree.insert(
            op.id,
            consumers.get(&op.id).map(|c| c.len()).unwrap_or(0),
        );
    }

    // Start with sinks (ops with no consumers)
    let mut queue: VecDeque<usize> = VecDeque::new();
    for op in ops {
        if consumers.get(&op.id).map(|c| c.is_empty()).unwrap_or(true) {
            queue.push_back(op.id);
            alap.insert(op.id, total_cycles - 1);
        }
    }

    let op_by_id: HashMap<usize, &Op> = ops.iter().map(|op| (op.id, op)).collect();

    // Process in reverse topological order
    while let Some(op_id) = queue.pop_front() {
        let my_alap = *alap.get(&op_id).unwrap_or(&(total_cycles - 1));

        // Update predecessors (ops we depend on)
        if let Some(op) = op_by_id.get(&op_id) {
            for &dep_id in &op.dep_ids {
                let new_alap = my_alap - 1;
                let current = alap.entry(dep_id).or_insert(total_cycles - 1);
                if new_alap < *current {
                    *current = new_alap;
                }

                if let Some(deg) = out_degree.get_mut(&dep_id) {
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push_back(dep_id);
                    }
                }
            }
        }
    }

    alap
}

/// Compute slack = ALAP - ASAP for each op
fn compute_slack(ops: &[Op], critical_path: &HashMap<usize, usize>) -> HashMap<usize, i32> {
    let total_cycles = critical_path.values().max().copied().unwrap_or(1) as i32;
    let asap = compute_asap(ops);
    let alap = compute_alap(ops, total_cycles);

    ops.iter()
        .map(|op| {
            let a = *asap.get(&op.id).unwrap_or(&0);
            let l = *alap.get(&op.id).unwrap_or(&(total_cycles - 1));
            (op.id, l - a)
        })
        .collect()
}

/// Greedy list scheduler with configurable priority mode and VALU→ALU shifting
pub fn schedule_greedy(
    ops: &mut [Op],
    critical_path: &HashMap<usize, usize>,
    enable_valu_shift: bool,
    priority_mode: PriorityMode,
) -> ScheduleResult {
    schedule_greedy_with_diagnostics(ops, critical_path, enable_valu_shift, priority_mode, false)
}

/// Greedy list scheduler with optional diagnostics collection
pub fn schedule_greedy_with_diagnostics(
    ops: &mut [Op],
    critical_path: &HashMap<usize, usize>,
    enable_valu_shift: bool,
    priority_mode: PriorityMode,
    collect_diagnostics: bool,
) -> ScheduleResult {
    let n = ops.len();
    if n == 0 {
        return ScheduleResult {
            cycles: 0,
            assignments: vec![],
            diagnostics: None,
            expanded_ops: vec![],
        };
    }

    // Initialize diagnostics if collecting
    let mut diagnostics = if collect_diagnostics {
        Some(SchedulingDiagnostics::new())
    } else {
        None
    };

    // Compute slack for slack-based priority modes
    let slack_map: HashMap<usize, i32> =
        if matches!(priority_mode, PriorityMode::Slack | PriorityMode::LoadEarlySlack) {
            compute_slack(ops, critical_path)
        } else {
            HashMap::new()
        };

    // Build op lookup (by index for mutation)
    let op_id_to_idx: HashMap<usize, usize> = ops
        .iter()
        .enumerate()
        .map(|(i, op)| (op.id, i))
        .collect();

    // Build consumer map (reverse of dep_ids) - O(1) lookup of who depends on each op
    let mut consumers: HashMap<usize, Vec<usize>> = HashMap::new();
    for op in ops.iter() {
        consumers.insert(op.id, Vec::new());
    }
    for op in ops.iter() {
        for &dep_id in &op.dep_ids {
            if let Some(c) = consumers.get_mut(&dep_id) {
                c.push(op.id);
            }
        }
    }

    // Compute in-degree
    let mut in_degree: HashMap<usize, usize> = HashMap::new();
    for op in ops.iter() {
        in_degree.insert(op.id, op.dep_ids.len());
    }

    // Ready queue: ops with in_degree == 0, using priority tuples
    // Format: (priority_tuple, op_id) - lower tuple = higher priority
    let mut ready: BinaryHeap<Reverse<((i32, i32, usize), usize)>> = BinaryHeap::new();
    for (idx, op) in ops.iter().enumerate() {
        if op.dep_ids.is_empty() {
            let prio = compute_priority(
                op.id,
                idx,
                &op.engine,
                critical_path,
                &slack_map,
                priority_mode,
            );
            ready.push(Reverse((prio, op.id)));
        }
    }

    // Track slot usage per cycle
    let mut slot_usage: HashMap<(usize, String), usize> = HashMap::new();

    // For VALU→ALU shifting: track expanded ops and their schedules
    let mut expanded_ops: Vec<Op> = Vec::new();
    let mut expanded_op_cycles: HashMap<usize, i32> = HashMap::new();
    let mut next_expanded_id = n + 1000000; // Start IDs high to avoid collision
    let mut shifted_valu_ops: HashSet<usize> = HashSet::new();

    // Smart Broadcast Forwarding: Track vbroadcast sources
    // Maps broadcast dest base address -> original scalar source address
    let mut broadcast_sources: HashMap<i64, i64> = HashMap::new();
    // Maps broadcast dest base address -> op ID of the broadcast
    let mut broadcast_op_ids: HashMap<i64, usize> = HashMap::new();

    // Pre-scan for vbroadcast ops to build broadcast_sources mapping
    for op in ops.iter() {
        if op.engine == "valu" && op.slot.len() >= 3 {
            if let SlotVal::Str(op_type) = &op.slot[0] {
                if op_type == "vbroadcast" {
                    if let (SlotVal::Int(dest), SlotVal::Int(scalar_src)) =
                        (&op.slot[1], &op.slot[2])
                    {
                        broadcast_sources.insert(*dest, *scalar_src);
                        broadcast_op_ids.insert(*dest, op.id);
                    }
                }
            }
        }
    }

    let get_slot_usage =
        |slot_usage: &HashMap<(usize, String), usize>, cycle: usize, engine: &str| -> usize {
            *slot_usage
                .get(&(cycle, engine.to_string()))
                .unwrap_or(&0)
        };

    let _can_schedule_n = |slot_usage: &HashMap<(usize, String), usize>,
                          cycle: usize,
                          engine: &str,
                          count: usize|
     -> bool {
        let usage = get_slot_usage(slot_usage, cycle, engine);
        let limit = get_slot_limit(engine);
        usage + count <= limit
    };

    let earliest_cycle_fn = |ops: &[Op],
                             op_id: usize,
                             op_id_to_idx: &HashMap<usize, usize>,
                             expanded_cycles: &HashMap<usize, i32>|
     -> usize {
        let op = &ops[*op_id_to_idx.get(&op_id).unwrap()];
        if op.dep_ids.is_empty() {
            return 0;
        }
        let mut max_cycle = 0;
        for &dep_id in &op.dep_ids {
            // Check if dependency was shifted (use expanded op cycle)
            if let Some(&exp_cycle) = expanded_cycles.get(&dep_id) {
                if exp_cycle >= 0 {
                    max_cycle = max_cycle.max(exp_cycle as usize + 1);
                }
            } else if let Some(&dep_idx) = op_id_to_idx.get(&dep_id) {
                let dep_cycle = ops[dep_idx].cycle;
                if dep_cycle >= 0 {
                    max_cycle = max_cycle.max(dep_cycle as usize + 1);
                }
            }
        }
        max_cycle
    };

    let mut assignments: Vec<(usize, usize)> = Vec::with_capacity(n);

    while let Some(Reverse((_prio, op_id))) = ready.pop() {
        let op_idx = *op_id_to_idx.get(&op_id).unwrap();
        let engine = ops[op_idx].engine.clone();
        let mut cycle = earliest_cycle_fn(ops, op_id, &op_id_to_idx, &expanded_op_cycles);

        // DIAGNOSTICS: Track slack and blocked ops
        if let Some(ref mut diag) = diagnostics {
            let op_slack = *slack_map.get(&op_id).unwrap_or(&0);
            diag.add_slack_sample(op_slack);

            // Check for blocked ops in ready queue
            let mut blocked_this_cycle = false;
            for other in ready.iter() {
                let other_id = other.0 .1;
                let other_idx = *op_id_to_idx.get(&other_id).unwrap();
                let other_engine = &ops[other_idx].engine;
                let other_cycle = earliest_cycle_fn(ops, other_id, &op_id_to_idx, &expanded_op_cycles);

                if get_slot_usage(&slot_usage, other_cycle, other_engine)
                    >= get_slot_limit(other_engine)
                {
                    diag.record_blocked(other_engine);
                    blocked_this_cycle = true;

                    // Check for priority inversion
                    let other_slack = *slack_map.get(&other_id).unwrap_or(&0);
                    if other_slack < op_slack {
                        diag.priority_inversions += 1;
                        diag.inversion_slack_delta_total += op_slack - other_slack;
                    }
                }
            }
            if blocked_this_cycle {
                diag.cycles_with_blocked_ops += 1;
            }

            // Check for "almost ready" ops (waiting on 1-2 dependencies)
            // Only count unique ops, not per-cycle
            // This is sampled periodically (every 100 cycles) to avoid O(n²) overhead
            if assignments.len() % 100 == 0 {
                let mut almost_ready_count = 0;
                for other in ops.iter() {
                    if other.cycle == -1 && !ready.iter().any(|r| r.0 .1 == other.id) {
                        if let Some(&remaining) = in_degree.get(&other.id) {
                            if remaining >= 1 && remaining <= 2 {
                                almost_ready_count += 1;
                            }
                        }
                    }
                }
                if almost_ready_count > 0 {
                    diag.cycles_with_almost_ready += 1;
                    // Record by engine for the first few to get distribution
                    for other in ops.iter().take(1000) {
                        if other.cycle == -1 && !ready.iter().any(|r| r.0 .1 == other.id) {
                            if let Some(&remaining) = in_degree.get(&other.id) {
                                if remaining >= 1 && remaining <= 2 {
                                    diag.record_almost_ready(&other.engine);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Broadcast elimination: if this is a vbroadcast and VALU is saturated,
        // check if all consumers are shiftable. If so, skip the broadcast entirely.
        let is_vbroadcast = engine == "valu"
            && ops[op_idx].slot.len() >= 3
            && matches!(&ops[op_idx].slot[0], SlotVal::Str(s) if s == "vbroadcast");

        if enable_valu_shift && is_vbroadcast {
            let valu_usage = get_slot_usage(&slot_usage, cycle, "valu");
            let alu_usage = get_slot_usage(&slot_usage, cycle, "alu");

            // Check if VALU is saturated and ALU has capacity
            if valu_usage >= get_slot_limit("valu") && alu_usage + VLEN <= get_slot_limit("alu") {
                // Check if all consumers are shiftable VALU ops OR already shifted
                let consumer_ids = consumers.get(&op_id).cloned().unwrap_or_default();
                let all_consumers_can_shift = consumer_ids.iter().all(|&cid| {
                    // Consumer already shifted - OK
                    if shifted_valu_ops.contains(&cid) {
                        return true;
                    }
                    // Consumer is shiftable VALU op - OK
                    let cidx = *op_id_to_idx.get(&cid).unwrap();
                    ops[cidx].engine == "valu" && can_convert_valu_to_alu(&ops[cidx])
                });

                if all_consumers_can_shift && !consumer_ids.is_empty() {
                    // ELIMINATE: Don't emit anything, just mark as done
                    if let (SlotVal::Int(dest), SlotVal::Int(scalar)) =
                        (&ops[op_idx].slot[1], &ops[op_idx].slot[2])
                    {
                        broadcast_sources.insert(*dest, *scalar);
                        broadcast_op_ids.insert(*dest, op_id);
                    }

                    // Mark original vbroadcast as eliminated
                    ops[op_idx].engine = "eliminated_broadcast".to_string();
                    ops[op_idx].cycle = cycle as i32;
                    assignments.push((op_id, cycle));

                    if let Some(ref mut diag) = diagnostics {
                        diag.broadcasts_eliminated += 1;
                    }

                    // Update consumers - they're now ready once their other deps are met
                    for &consumer_id in &consumer_ids {
                        if let Some(deg) = in_degree.get_mut(&consumer_id) {
                            *deg -= 1;
                            if *deg == 0 {
                                let consumer_idx = *op_id_to_idx.get(&consumer_id).unwrap();
                                let prio = compute_priority(
                                    consumer_id,
                                    consumer_idx,
                                    &ops[consumer_idx].engine,
                                    critical_path,
                                    &slack_map,
                                    priority_mode,
                                );
                                ready.push(Reverse((prio, consumer_id)));
                            }
                        }
                    }
                    continue;
                }
            }
        }

        // Normal scheduling with VALU→ALU shifting integrated into the loop
        // This matches Python's behavior: check for shifting at each cycle where VALU is full
        let is_shiftable_valu =
            enable_valu_shift && engine == "valu" && can_convert_valu_to_alu(&ops[op_idx]);
        let mut shifted = false;

        while get_slot_usage(&slot_usage, cycle, &engine) >= get_slot_limit(&engine) {
            // VALU→ALU shifting: if VALU is full but ALU has capacity
            if is_shiftable_valu {
                let alu_usage = get_slot_usage(&slot_usage, cycle, "alu");

                if let Some(ref mut diag) = diagnostics {
                    diag.cycles_valu_saturated += 1;
                }

                if alu_usage + VLEN <= get_slot_limit("alu") {
                    // Expand VALU op to 8 ALU ops with smart broadcast forwarding
                    let (mut alu_ops, forward_count) = expand_valu_to_alu_smart(
                        &ops[op_idx],
                        next_expanded_id,
                        &broadcast_sources,
                        &broadcast_op_ids,
                    );
                    next_expanded_id += VLEN;

                    if let Some(ref mut diag) = diagnostics {
                        diag.broadcast_forwards += forward_count;
                    }

                    // Schedule all 8 ALU ops at this cycle - SET THEIR CYCLE!
                    for alu_op in &mut alu_ops {
                        alu_op.cycle = cycle as i32;
                        *slot_usage.entry((cycle, "alu".to_string())).or_insert(0) += 1;
                    }

                    // Mark original VALU op as shifted - change engine to prevent emission
                    ops[op_idx].engine = "shifted_valu".to_string();
                    ops[op_idx].cycle = cycle as i32;
                    expanded_op_cycles.insert(op_id, cycle as i32);
                    shifted_valu_ops.insert(op_id);
                    expanded_ops.extend(alu_ops);
                    assignments.push((op_id, cycle));

                    if let Some(ref mut diag) = diagnostics {
                        diag.valu_shifted += 1;
                    }

                    // Update consumers
                    if let Some(consumer_ids) = consumers.get(&op_id) {
                        for &consumer_id in consumer_ids {
                            if let Some(deg) = in_degree.get_mut(&consumer_id) {
                                *deg -= 1;
                                if *deg == 0 {
                                    let consumer_idx = *op_id_to_idx.get(&consumer_id).unwrap();
                                    let prio = compute_priority(
                                        consumer_id,
                                        consumer_idx,
                                        &ops[consumer_idx].engine,
                                        critical_path,
                                        &slack_map,
                                        priority_mode,
                                    );
                                    ready.push(Reverse((prio, consumer_id)));
                                }
                            }
                        }
                    }
                    shifted = true;
                    break;
                } else {
                    if let Some(ref mut diag) = diagnostics {
                        diag.valu_alu_no_capacity += 1;
                    }
                }
            }
            cycle += 1;
        }

        if shifted {
            continue;
        }

        // Track diagnostics for non-shifted VALU ops
        if enable_valu_shift && engine == "valu" {
            if let Some(ref mut diag) = diagnostics {
                if !can_convert_valu_to_alu(&ops[op_idx]) {
                    diag.valu_not_shiftable += 1;
                } else {
                    diag.valu_slot_available += 1;
                }
            }
        }

        // Schedule the op
        ops[op_idx].cycle = cycle as i32;
        *slot_usage.entry((cycle, engine.clone())).or_insert(0) += 1;
        assignments.push((op_id, cycle));

        // Update ready queue for dependent ops
        if let Some(consumer_ids) = consumers.get(&op_id) {
            for &consumer_id in consumer_ids {
                if let Some(deg) = in_degree.get_mut(&consumer_id) {
                    *deg -= 1;
                    if *deg == 0 {
                        let consumer_idx = *op_id_to_idx.get(&consumer_id).unwrap();
                        let prio = compute_priority(
                            consumer_id,
                            consumer_idx,
                            &ops[consumer_idx].engine,
                            critical_path,
                            &slack_map,
                            priority_mode,
                        );
                        ready.push(Reverse((prio, consumer_id)));
                    }
                }
            }
        }
    }

    let max_cycle = ops
        .iter()
        .filter(|op| op.cycle >= 0)
        .map(|op| op.cycle as usize)
        .max()
        .unwrap_or(0);

    // Finalize diagnostics
    if let Some(ref mut diag) = diagnostics {
        diag.total_cycles = max_cycle + 1;
        diag.total_ops_scheduled = assignments.len();
    }

    ScheduleResult {
        cycles: max_cycle + 1,
        assignments,
        diagnostics,
        expanded_ops,
    }
}
