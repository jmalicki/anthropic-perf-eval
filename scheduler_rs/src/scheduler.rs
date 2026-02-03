//! Core scheduling algorithms.
//!
//! This module contains the greedy list scheduler with multiple priority modes
//! and support for VALU→ALU shifting when VALU slots are saturated.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

use crate::diagnostics::SchedulingDiagnostics;
use arrayvec::ArrayVec;
use crate::types::{
    can_convert_valu_to_alu, expand_valu_to_alu_smart, Engine, Op, SlotVal, VLEN,
    ConstEncoding, PseudoConst, resolve_pseudo_const, MAX_SLOT_SIZE,
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
    ScratchPressure,  // Prioritize ops that free scratch (high LUC = Last Use Count)
}

pub const ALL_PRIORITY_MODES: [PriorityMode; 8] = [
    PriorityMode::CriticalPath,
    PriorityMode::CriticalPathValu,
    PriorityMode::EngineBalance,
    PriorityMode::Slack,
    PriorityMode::LoadEarly,
    PriorityMode::LoadEarlySlack,
    PriorityMode::OriginalOrder,
    PriorityMode::ScratchPressure,
];

// ============================================================================
// Scratch Pressure Tracking
// ============================================================================

/// Scratch memory budget (matches config.rs)
pub const SCRATCH_BUDGET: usize = 1536;

/// Compute which reads are "last uses" for each op.
///
/// An address's last use is the op with the highest ID that reads it.
/// When that op executes, the address can be freed from scratch because
/// no later op will read from it.
///
/// This is used for scratch pressure tracking - we can compute how many
/// addresses are "live" at any point by tracking allocations (writes)
/// and deallocations (last uses).
pub fn compute_last_uses(ops: &mut [Op]) {
    // Map: address -> op_id with latest use (highest ID that reads this address)
    let mut last_reader: HashMap<i64, usize> = HashMap::new();

    // First pass: find the latest reader for each address
    for op in ops.iter() {
        for &addr in &op.reads {
            last_reader
                .entry(addr)
                .and_modify(|existing| {
                    if op.id > *existing {
                        *existing = op.id;
                    }
                })
                .or_insert(op.id);
        }
    }

    // Second pass: mark last uses on each op
    for op in ops.iter_mut() {
        op.last_use_reads.clear();
        for &addr in &op.reads {
            if last_reader.get(&addr) == Some(&op.id) {
                op.last_use_reads.insert(addr);
            }
        }
    }
}

/// Track live scratch addresses during scheduling.
///
/// This implements register pressure tracking adapted for scratch memory.
/// Each address is "live" from when it's written until its last read.
///
/// Key concepts:
/// - **Pressure**: Count of currently live addresses
/// - **Peak**: Maximum pressure seen during scheduling
/// - **Budget**: Maximum allowed pressure (1536 for this architecture)
#[derive(Debug)]
pub struct ScratchPressureTracker {
    /// Currently live addresses (written but not yet last-used)
    live: HashSet<i64>,
    /// Peak pressure observed
    peak: usize,
    /// Budget limit
    limit: usize,
    /// Count of times pressure exceeded budget
    exceeded_count: usize,
}

impl ScratchPressureTracker {
    /// Create a new pressure tracker with the given limit
    pub fn new(limit: usize) -> Self {
        Self {
            live: HashSet::new(),
            peak: 0,
            limit,
            exceeded_count: 0,
        }
    }

    /// Get current pressure (number of live addresses)
    pub fn current_pressure(&self) -> usize {
        self.live.len()
    }

    /// Get peak pressure observed so far
    pub fn peak_pressure(&self) -> usize {
        self.peak
    }

    /// Compute the pressure change if we schedule this op.
    ///
    /// Returns (addresses_added, addresses_freed, net_change)
    /// - Positive net = pressure increases
    /// - Negative net = pressure decreases (good!)
    pub fn pressure_delta(&self, op: &Op) -> (usize, usize, i64) {
        let added = op.writes.len();
        let freed = op.last_use_reads.len();
        let net = added as i64 - freed as i64;
        (added, freed, net)
    }

    /// Check if scheduling this op would exceed the budget.
    ///
    /// Note: This is a soft check - we allow scheduling even if exceeded,
    /// but track how often this happens for diagnostics.
    pub fn would_exceed(&self, op: &Op) -> bool {
        let (added, freed, _) = self.pressure_delta(op);
        let new_pressure = self.live.len() + added - freed.min(self.live.len());
        new_pressure > self.limit
    }

    /// Apply the pressure changes from scheduling an op.
    /// Call this after the op is scheduled.
    pub fn apply(&mut self, op: &Op) {
        // Add written addresses (they become live)
        for &addr in &op.writes {
            self.live.insert(addr);
        }

        // Remove last-used addresses (they are no longer needed)
        for &addr in &op.last_use_reads {
            self.live.remove(&addr);
        }

        // Track peak
        if self.live.len() > self.peak {
            self.peak = self.live.len();
        }

        // Track exceeded count
        if self.live.len() > self.limit {
            self.exceeded_count += 1;
        }
    }

    /// Get count of times pressure exceeded budget
    pub fn exceeded_count(&self) -> usize {
        self.exceeded_count
    }
}

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

/// Priority for scheduling.
///
/// Returns (primary, secondary, luc_tiebreaker, op_id) where lower = higher priority.
///
/// The LUC (Last Use Count) is used as a tie-breaker for ALL modes to help
/// reduce scratch pressure. When ops have equal priority, we prefer those
/// that free more scratch addresses.
fn compute_priority(
    op_id: usize,
    op_idx: usize,
    engine: Engine,
    critical_path: &HashMap<usize, usize>,
    slack_map: &HashMap<usize, i32>,
    luc_map: &HashMap<usize, usize>,
    mode: PriorityMode,
) -> (i32, i32, i32, usize) {
    let cp = *critical_path.get(&op_id).unwrap_or(&1) as i32;
    let slack = *slack_map.get(&op_id).unwrap_or(&0);
    let luc = *luc_map.get(&op_id).unwrap_or(&0) as i32;
    let is_valu = if engine == Engine::Valu { 0 } else { 1 };
    let is_load = if engine == Engine::Load { 0 } else { 1 };

    // LUC as tie-breaker: negative because higher LUC = higher priority = lower sort key
    let luc_tiebreaker = -luc;

    match mode {
        PriorityMode::CriticalPath => (-cp, is_valu, luc_tiebreaker, op_id),
        PriorityMode::CriticalPathValu => (is_valu, -cp, luc_tiebreaker, op_id),
        PriorityMode::EngineBalance => {
            // Matches Python: -cp * load_factor where load_factor is 0.6, 0.8, 0.5, 1.0
            // Scale by 10000 to preserve float precision (Python uses actual floats)
            let load_factor: f64 = match engine {
                Engine::Valu => 0.6,
                Engine::Load | Engine::Store => 0.8,
                Engine::Flow => 0.5,
                _ => 1.0, // ALU and others
            };
            // Scale to preserve decimal precision - Python compares floats directly
            let adjusted_cp = (-(cp as f64) * load_factor * 10000.0) as i32;
            (adjusted_cp, luc_tiebreaker, 0, op_id)
        }
        PriorityMode::Slack => (slack, -cp, luc_tiebreaker, op_id),
        PriorityMode::LoadEarly => (is_load, -cp, luc_tiebreaker, op_id),
        PriorityMode::LoadEarlySlack => (is_load, slack, luc_tiebreaker, op_id),
        PriorityMode::OriginalOrder => (op_idx as i32, is_valu, luc_tiebreaker, op_id),
        PriorityMode::ScratchPressure => {
            // LUC is the PRIMARY criterion - schedule ops that free the most scratch first
            // Then critical path, then op_id for stability
            (-luc, -cp, 0, op_id)
        }
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

/// Proactive lookahead analysis to identify broadcasts that can be eliminated
/// before scheduling begins. This predicts which VALU ops will shift to ALU
/// due to VALU saturation, and eliminates broadcasts whose ALL consumers
/// are predicted to shift.
pub fn compute_proactive_eliminations(ops: &[Op]) -> HashSet<usize> {
    const VALU_SLOTS: usize = 6;

    // Identify shiftable ops and broadcasts
    let mut shiftable_ops: HashSet<usize> = HashSet::new();
    let mut broadcast_ops: HashMap<usize, i64> = HashMap::new(); // bc_id -> scalar source
    let mut broadcast_dest_to_id: HashMap<i64, usize> = HashMap::new(); // dest -> bc_id

    for op in ops {
        if can_convert_valu_to_alu(op) {
            shiftable_ops.insert(op.id);
        }
        // Check for vbroadcast
        if op.engine == Engine::Valu && op.slot.len() >= 3 {
            if let SlotVal::Str(op_type) = &op.slot[0] {
                if op_type == "vbroadcast" {
                    if let (SlotVal::Int(dest), SlotVal::Int(scalar)) =
                        (&op.slot[1], &op.slot[2])
                    {
                        broadcast_ops.insert(op.id, *scalar);
                        broadcast_dest_to_id.insert(*dest, op.id);
                    }
                }
            }
        }
    }

    // Find consumers of each broadcast (ops that depend on broadcast AND read its dest)
    let mut broadcast_consumers: HashMap<usize, HashSet<usize>> = HashMap::new();
    for &bc_id in broadcast_ops.keys() {
        broadcast_consumers.insert(bc_id, HashSet::new());
    }

    // Build op_by_id for broadcast dest lookup
    let op_by_id: HashMap<usize, &Op> = ops.iter().map(|op| (op.id, op)).collect();

    for op in ops {
        for &bc_id in broadcast_ops.keys() {
            // Must depend on this broadcast
            if !op.dep_ids.contains(&bc_id) {
                continue;
            }
            // Must read from broadcast dest range
            let bc_op = op_by_id.get(&bc_id).unwrap();
            if let SlotVal::Int(dest) = &bc_op.slot[1] {
                let bc_dest_range: HashSet<i64> = (0..VLEN as i64).map(|i| dest + i).collect();
                if op.reads.iter().any(|r| bc_dest_range.contains(r)) {
                    broadcast_consumers.get_mut(&bc_id).unwrap().insert(op.id);
                }
            }
        }
    }

    // Identify eliminable broadcasts (all consumers are shiftable)
    let eliminable_broadcasts: HashSet<usize> = broadcast_consumers
        .iter()
        .filter(|(_, consumers)| {
            !consumers.is_empty() && consumers.iter().all(|c| shiftable_ops.contains(c))
        })
        .map(|(&bc_id, _)| bc_id)
        .collect();

    // Predict which ops will shift due to VALU pressure
    // Count VALU ops by category
    let valu_ops: Vec<&Op> = ops.iter().filter(|op| op.engine == Engine::Valu).collect();
    let shiftable_valu: Vec<&Op> = valu_ops
        .iter()
        .filter(|op| shiftable_ops.contains(&op.id))
        .copied()
        .collect();
    let non_shiftable_valu: Vec<&Op> = valu_ops
        .iter()
        .filter(|op| {
            !shiftable_ops.contains(&op.id)
                && !matches!(&op.slot[0], SlotVal::Str(s) if s == "vbroadcast")
        })
        .copied()
        .collect();

    // Estimate minimum cycles for non-shiftable (they MUST use VALU)
    let min_cycles_for_non_shiftable =
        (non_shiftable_valu.len() + VALU_SLOTS - 1) / VALU_SLOTS;

    // Total VALU capacity for shiftable ops
    let total_valu_capacity = min_cycles_for_non_shiftable * VALU_SLOTS;
    let valu_slots_for_shiftable = total_valu_capacity.saturating_sub(non_shiftable_valu.len());

    // Excess shiftable ops that won't fit in VALU
    let excess_shiftable = shiftable_valu.len().saturating_sub(valu_slots_for_shiftable);

    if excess_shiftable == 0 || shiftable_valu.is_empty() {
        return HashSet::new(); // No shifting needed
    }

    let shift_ratio = excess_shiftable as f64 / shiftable_valu.len() as f64;

    if shift_ratio < 0.3 {
        return HashSet::new(); // Too uncertain to predict
    }

    // Predict which ops will shift: last N by emission order (op.id)
    let mut sorted_shiftable: Vec<&Op> = shiftable_valu.clone();
    sorted_shiftable.sort_by_key(|op| op.id);

    // 80% confidence margin
    let confident_excess = (excess_shiftable as f64 * 0.8) as usize;
    if confident_excess == 0 {
        return HashSet::new();
    }

    let predicted_shifts: HashSet<usize> = sorted_shiftable
        .iter()
        .rev()
        .take(confident_excess)
        .map(|op| op.id)
        .collect();

    // Find broadcasts where ALL consumers are predicted to shift
    let mut proactive_eliminations: HashSet<usize> = HashSet::new();

    for &bc_id in &eliminable_broadcasts {
        let consumers = broadcast_consumers.get(&bc_id).unwrap();
        if !consumers.is_empty() && consumers.iter().all(|c| predicted_shifts.contains(c)) {
            proactive_eliminations.insert(bc_id);
        }
    }

    proactive_eliminations
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

    // Compute last-use information for scratch pressure tracking
    compute_last_uses(ops);

    // Build LUC (Last Use Count) map for ScratchPressure priority mode
    let luc_map: HashMap<usize, usize> = ops
        .iter()
        .map(|op| (op.id, op.last_use_reads.len()))
        .collect();

    // Initialize scratch pressure tracker
    let mut pressure_tracker = ScratchPressureTracker::new(SCRATCH_BUDGET);

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

    // Precompute priorities ONCE - they're static (all inputs computed upfront)
    // Priority tuple: (i32, i32, i32, usize) - lower values = higher priority
    let priority_cache: HashMap<usize, (i32, i32, i32, usize)> = ops
        .iter()
        .map(|op| {
            let op_idx = *op_id_to_idx.get(&op.id).unwrap();
            let priority = compute_priority(
                op.id,
                op_idx,
                op.engine,
                critical_path,
                &slack_map,
                &luc_map,
                priority_mode,
            );
            (op.id, priority)
        })
        .collect();

    // Ready queue: ops with in_degree == 0
    // Use BinaryHeap with Reverse for min-heap (lower priority value = higher priority)
    let mut ready_heap: BinaryHeap<Reverse<((i32, i32, i32, usize), usize)>> = BinaryHeap::new();
    // Maintain a HashSet for O(1) membership checks in diagnostics
    let mut ready_set: HashSet<usize> = HashSet::new();
    for op in ops.iter() {
        if op.dep_ids.is_empty() {
            let priority = priority_cache[&op.id];
            ready_heap.push(Reverse((priority, op.id)));
            ready_set.insert(op.id);
        }
    }


    // Track slot usage per cycle
    let mut slot_usage: HashMap<(usize, Engine), usize> = HashMap::new();

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
        if op.engine == Engine::Valu && op.slot.len() >= 3 {
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

    // Proactive broadcast elimination is disabled by default (matches Python).
    // Reactive elimination (when VALU is saturated) is more effective.
    let proactive_eliminations: HashSet<usize> = HashSet::new();

    // Track which broadcasts have been eliminated (their consumers MUST shift)
    let mut eliminated_broadcast_ids: HashSet<usize> = HashSet::new();

    let get_slot_usage =
        |slot_usage: &HashMap<(usize, Engine), usize>, cycle: usize, engine: Engine| -> usize {
            *slot_usage
                .get(&(cycle, engine))
                .unwrap_or(&0)
        };

    let _can_schedule_n = |slot_usage: &HashMap<(usize, Engine), usize>,
                          cycle: usize,
                          engine: Engine,
                          count: usize|
     -> bool {
        let usage = get_slot_usage(slot_usage, cycle, engine);
        let limit = engine.slot_limit();
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

    while let Some(Reverse((_, op_id))) = ready_heap.pop() {
        // Remove from ready_set for O(1) membership checks
        ready_set.remove(&op_id);
        let op_idx = *op_id_to_idx.get(&op_id).unwrap();
        let engine = ops[op_idx].engine;
        let mut cycle = earliest_cycle_fn(ops, op_id, &op_id_to_idx, &expanded_op_cycles);

        // DIAGNOSTICS: Track slack and blocked ops
        if let Some(ref mut diag) = diagnostics {
            let op_slack = *slack_map.get(&op_id).unwrap_or(&0);
            diag.add_slack_sample(op_slack);

            // Check for blocked ops in ready queue
            let mut blocked_this_cycle = false;
            for &other_id in ready_set.iter() {
                let other_idx = *op_id_to_idx.get(&other_id).unwrap();
                let other_engine = ops[other_idx].engine;
                let other_cycle = earliest_cycle_fn(ops, other_id, &op_id_to_idx, &expanded_op_cycles);

                if get_slot_usage(&slot_usage, other_cycle, other_engine)
                    >= other_engine.slot_limit()
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
                    if other.cycle == -1 && !ready_set.contains(&other.id) {
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
                        if other.cycle == -1 && !ready_set.contains(&other.id) {
                            if let Some(&remaining) = in_degree.get(&other.id) {
                                if remaining >= 1 && remaining <= 2 {
                                    diag.record_almost_ready(other.engine);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Broadcast elimination: eliminate broadcasts when EITHER:
        // 1. PROACTIVE: lookahead analysis predicted all consumers will shift
        // 2. REACTIVE: VALU is saturated and all consumers are shiftable
        let is_vbroadcast = engine == Engine::Valu
            && ops[op_idx].slot.len() >= 3
            && matches!(&ops[op_idx].slot[0], SlotVal::Str(s) if s == "vbroadcast");

        if enable_valu_shift && is_vbroadcast {
            // Check proactive elimination (computed upfront via lookahead)
            let is_proactive = proactive_eliminations.contains(&op_id);

            // Check reactive elimination (VALU saturated at this cycle)
            let valu_usage = get_slot_usage(&slot_usage, cycle, Engine::Valu);
            let alu_usage = get_slot_usage(&slot_usage, cycle, Engine::Alu);
            let is_reactive = valu_usage >= Engine::Valu.slot_limit()
                && alu_usage + VLEN <= Engine::Alu.slot_limit()
                && {
                    let consumer_ids = consumers.get(&op_id).cloned().unwrap_or_default();
                    let all_consumers_can_shift = consumer_ids.iter().all(|&cid| {
                        shifted_valu_ops.contains(&cid) || {
                            let cidx = *op_id_to_idx.get(&cid).unwrap();
                            ops[cidx].engine == Engine::Valu && can_convert_valu_to_alu(&ops[cidx])
                        }
                    });
                    all_consumers_can_shift && !consumer_ids.is_empty()
                };

            if is_proactive || is_reactive {
                // ELIMINATE: Don't emit anything, just mark as done
                if let (SlotVal::Int(dest), SlotVal::Int(scalar)) =
                    (&ops[op_idx].slot[1], &ops[op_idx].slot[2])
                {
                    broadcast_sources.insert(*dest, *scalar);
                    broadcast_op_ids.insert(*dest, op_id);
                }

                // Mark original vbroadcast as eliminated
                ops[op_idx].engine = Engine::EliminatedBroadcast;
                ops[op_idx].cycle = cycle as i32;
                eliminated_broadcast_ids.insert(op_id);
                assignments.push((op_id, cycle));

                if let Some(ref mut diag) = diagnostics {
                    diag.broadcasts_eliminated += 1;
                }

                // Update consumers - they're now ready once their other deps are met
                let consumer_ids = consumers.get(&op_id).cloned().unwrap_or_default();
                for &consumer_id in &consumer_ids {
                    if let Some(deg) = in_degree.get_mut(&consumer_id) {
                        *deg -= 1;
                        if *deg == 0 {
                            let priority = priority_cache[&consumer_id];
                            ready_heap.push(Reverse((priority, consumer_id)));
                            ready_set.insert(consumer_id);
                        }
                    }
                }
                continue;
            }
        }

        // Normal scheduling with VALU→ALU shifting integrated into the loop
        // This matches Python's behavior: check for shifting at each cycle where VALU is full
        let is_shiftable_valu =
            enable_valu_shift && engine == Engine::Valu && can_convert_valu_to_alu(&ops[op_idx]);

        // Check if this op MUST shift because it depends on an eliminated broadcast
        // (eliminated broadcasts don't produce vector values, so consumers must use scalar)
        let must_shift_for_eliminated_bc = is_shiftable_valu
            && ops[op_idx]
                .dep_ids
                .iter()
                .any(|d| eliminated_broadcast_ids.contains(d));

        let mut shifted = false;

        // Force shift if consumer of eliminated broadcast (must use scalar forwarding)
        if must_shift_for_eliminated_bc {
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

            // Schedule each ALU op individually at earliest available cycle
            let mut max_alu_cycle = cycle;
            for alu_op in &mut alu_ops {
                let mut alu_cycle = cycle;
                while get_slot_usage(&slot_usage, alu_cycle, Engine::Alu) >= Engine::Alu.slot_limit() {
                    alu_cycle += 1;
                }
                alu_op.cycle = alu_cycle as i32;
                *slot_usage.entry((alu_cycle, Engine::Alu)).or_insert(0) += 1;
                max_alu_cycle = max_alu_cycle.max(alu_cycle);
            }

            ops[op_idx].engine = Engine::ShiftedValu;
            ops[op_idx].cycle = max_alu_cycle as i32;
            expanded_op_cycles.insert(op_id, max_alu_cycle as i32);
            shifted_valu_ops.insert(op_id);
            expanded_ops.extend(alu_ops);
            assignments.push((op_id, max_alu_cycle));

            // Track scratch pressure (use original op's reads/writes)
            pressure_tracker.apply(&ops[op_idx]);

            if let Some(ref mut diag) = diagnostics {
                diag.valu_shifted += 1;
            }

            // Update consumers
            if let Some(consumer_ids) = consumers.get(&op_id) {
                for &consumer_id in consumer_ids {
                    if let Some(deg) = in_degree.get_mut(&consumer_id) {
                        *deg -= 1;
                        if *deg == 0 {
                            let priority = priority_cache[&consumer_id];
                            ready_heap.push(Reverse((priority, consumer_id)));
                            ready_set.insert(consumer_id);
                        }
                    }
                }
            }
            continue;
        }

        while get_slot_usage(&slot_usage, cycle, engine) >= engine.slot_limit() {
            // VALU→ALU shifting: if VALU is full but ALU has capacity
            if is_shiftable_valu {
                let alu_usage = get_slot_usage(&slot_usage, cycle, Engine::Alu);

                if let Some(ref mut diag) = diagnostics {
                    diag.cycles_valu_saturated += 1;
                }

                // Only need at least 1 free ALU slot to start shifting
                // (ALU ops spread across cycles like Python does)
                if alu_usage < Engine::Alu.slot_limit() {
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

                    // Schedule each ALU op individually at earliest available cycle
                    // This matches Python's behavior and allows ops to spread across cycles
                    let mut max_alu_cycle = cycle;
                    for alu_op in &mut alu_ops {
                        let mut alu_cycle = cycle;
                        while get_slot_usage(&slot_usage, alu_cycle, Engine::Alu)
                            >= Engine::Alu.slot_limit()
                        {
                            alu_cycle += 1;
                        }
                        alu_op.cycle = alu_cycle as i32;
                        *slot_usage.entry((alu_cycle, Engine::Alu)).or_insert(0) += 1;
                        max_alu_cycle = max_alu_cycle.max(alu_cycle);
                    }

                    // Mark original VALU op as shifted - cycle is max of all ALU ops
                    ops[op_idx].engine = Engine::ShiftedValu;
                    ops[op_idx].cycle = max_alu_cycle as i32;
                    expanded_op_cycles.insert(op_id, max_alu_cycle as i32);
                    shifted_valu_ops.insert(op_id);
                    expanded_ops.extend(alu_ops);
                    assignments.push((op_id, max_alu_cycle));

                    // Track scratch pressure (use original op's reads/writes)
                    pressure_tracker.apply(&ops[op_idx]);

                    if let Some(ref mut diag) = diagnostics {
                        diag.valu_shifted += 1;
                    }

                    // Update consumers
                    if let Some(consumer_ids) = consumers.get(&op_id) {
                        for &consumer_id in consumer_ids {
                            if let Some(deg) = in_degree.get_mut(&consumer_id) {
                                *deg -= 1;
                                if *deg == 0 {
                                    let priority = priority_cache[&consumer_id];
                                    ready_heap.push(Reverse((priority, consumer_id)));
                                    ready_set.insert(consumer_id);
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
        if enable_valu_shift && engine == Engine::Valu {
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
        *slot_usage.entry((cycle, engine)).or_insert(0) += 1;
        assignments.push((op_id, cycle));

        // Track scratch pressure
        pressure_tracker.apply(&ops[op_idx]);

        // Update ready queue for dependent ops
        if let Some(consumer_ids) = consumers.get(&op_id) {
            for &consumer_id in consumer_ids {
                if let Some(deg) = in_degree.get_mut(&consumer_id) {
                    *deg -= 1;
                    if *deg == 0 {
                        let priority = priority_cache[&consumer_id];
                        ready_heap.push(Reverse((priority, consumer_id)));
                        ready_set.insert(consumer_id);
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

        // Scratch pressure stats
        diag.peak_scratch_pressure = pressure_tracker.peak_pressure();
        diag.scratch_budget = SCRATCH_BUDGET;
        diag.pressure_exceeded_count = pressure_tracker.exceeded_count();
    }

    ScheduleResult {
        cycles: max_cycle + 1,
        assignments,
        diagnostics,
        expanded_ops,
    }
}

// ============================================================================
// Pseudo-Const Scheduling
// ============================================================================

/// State for tracking constant values during scheduling
#[derive(Debug, Default)]
pub struct ConstValueTracker {
    /// Values that are currently available (have been produced)
    pub available_values: HashSet<i64>,

    /// Map from constant value to the register holding it
    pub value_to_register: HashMap<i64, i64>,

    /// Map from register to the constant value it holds (if known)
    pub register_to_value: HashMap<i64, i64>,

    /// Next register to allocate for constants (if needed)
    pub next_const_register: i64,
}

impl ConstValueTracker {
    /// Create a new tracker with a starting register for constants
    pub fn new(start_register: i64) -> Self {
        Self {
            available_values: HashSet::new(),
            value_to_register: HashMap::new(),
            register_to_value: HashMap::new(),
            next_const_register: start_register,
        }
    }

    /// Record that a value is now available in a register
    pub fn record_value(&mut self, value: i64, register: i64) {
        self.available_values.insert(value);
        self.value_to_register.insert(value, register);
        self.register_to_value.insert(register, value);
    }

    /// Allocate a register for a constant value
    pub fn allocate_register(&mut self) -> i64 {
        let reg = self.next_const_register;
        self.next_const_register += 1;
        reg
    }

    /// Check if a value is available
    pub fn is_available(&self, value: i64) -> bool {
        self.available_values.contains(&value)
    }

    /// Get the register holding a value (if available)
    pub fn get_register(&self, value: i64) -> Option<i64> {
        self.value_to_register.get(&value).copied()
    }
}

/// Result of trying to resolve a pseudo-const
pub enum PseudoConstResolution {
    /// Successfully resolved to a concrete op
    Resolved(Op, ConstEncoding),
    /// Cannot resolve this cycle (dependencies not ready or no slots)
    NotReady,
    /// No valid encoding found (should not happen)
    NoEncoding,
}

/// Try to resolve a pseudo-const at the given cycle.
///
/// Returns the resolved Op if successful, along with the encoding used.
pub fn try_resolve_pseudo_const(
    pc: &PseudoConst,
    cycle: usize,
    tracker: &ConstValueTracker,
    slot_usage: &HashMap<(usize, Engine), usize>,
    next_op_id: usize,
) -> PseudoConstResolution {
    // Helper to check slot availability
    let slots_available = |engine: Engine| -> bool {
        let used = slot_usage.get(&(cycle, engine)).copied().unwrap_or(0);
        used < engine.slot_limit()
    };

    // Try each encoding in order of preference
    for encoding in &pc.encodings {
        // Check if dependencies are satisfied
        if !encoding.is_ready(&tracker.available_values) {
            continue;
        }

        // Check if slots are available
        let engine = encoding.engine();
        if !slots_available(engine) {
            continue;
        }

        // Resolve to concrete op
        if let Some(op) = resolve_pseudo_const(pc, encoding, &tracker.value_to_register, next_op_id) {
            return PseudoConstResolution::Resolved(op, encoding.clone());
        }
    }

    // Check if ANY encoding could work with more dependencies
    let has_possible_encoding = pc.encodings.iter().any(|enc| {
        let engine = enc.engine();
        slots_available(engine)
    });

    if has_possible_encoding {
        PseudoConstResolution::NotReady
    } else {
        // All encodings blocked by slots - also NotReady
        PseudoConstResolution::NotReady
    }
}

/// Schedule ops with dynamic pseudo-const resolution.
///
/// This function wraps the regular scheduler but adds:
/// 1. Dynamic resolution of pseudo-consts based on available values and slots
/// 2. Tracking of which constant values are available
/// 3. Integration of resolved pseudo-const ops into the regular schedule
///
/// # Arguments
/// * `ops` - Regular operations to schedule (may depend on pseudo-consts)
/// * `pseudo_consts` - Pseudo-const operations to resolve dynamically
/// * `critical_path` - Critical path information for priority
/// * `enable_valu_shift` - Whether to enable VALU→ALU shifting
/// * `priority_mode` - Priority mode for scheduling
/// * `const_register_start` - Starting register number for constants
///
/// # Returns
/// * Extended ScheduleResult with resolved pseudo-consts included
pub fn schedule_with_pseudo_consts(
    ops: &mut Vec<Op>,
    pseudo_consts: Vec<PseudoConst>,
    critical_path: &HashMap<usize, usize>,
    enable_valu_shift: bool,
    priority_mode: PriorityMode,
    const_register_start: i64,
) -> (ScheduleResult, Vec<Op>) {
    // Initialize constant value tracker
    let mut tracker = ConstValueTracker::new(const_register_start);

    // Pre-populate with value 0 (always available via ZERO_REG or scratch initialization)
    // ZERO_REG is register 63 in this architecture
    tracker.record_value(0, 63);

    // Assign registers to all pseudo-consts first (but don't mark as available yet)
    let mut pending_consts: Vec<PseudoConst> = pseudo_consts
        .into_iter()
        .map(|mut pc| {
            if pc.dest_register.is_none() {
                pc.dest_register = Some(tracker.allocate_register());
            }
            pc
        })
        .collect();

    // Build value-to-register mapping for resolution (needed by resolve_pseudo_const)
    let value_to_reg: HashMap<i64, i64> = pending_consts
        .iter()
        .filter_map(|pc| pc.dest_register.map(|dest| (pc.value, dest)))
        .chain(std::iter::once((0, 63))) // ZERO_REG
        .collect();

    // Ops generated from resolved pseudo-consts
    let mut const_ops: Vec<Op> = Vec::new();
    let mut next_const_op_id = ops.iter().map(|op| op.id).max().unwrap_or(0) + 10000;

    // TOPOLOGICAL RESOLUTION: Resolve constants in dependency order.
    // Bootstrap constants (those that use Load or AddImm) can be resolved immediately.
    // Derived constants (Add, Mul, etc.) must wait for their dependencies.
    //
    // This ensures the main scheduler sees proper dependencies between const ops.
    let max_iterations = 100; // Safety limit
    let mut iteration = 0;

    while !pending_consts.is_empty() && iteration < max_iterations {
        let mut resolved_this_iteration = Vec::new();

        for (idx, pc) in pending_consts.iter().enumerate() {
            // Find first encoding whose dependencies are satisfied
            let encoding = pc.encodings.iter()
                .find(|enc| enc.is_ready(&tracker.available_values))
                .cloned();

            if let Some(enc) = encoding {
                if let Some(op) = resolve_pseudo_const(pc, &enc, &value_to_reg, next_const_op_id) {
                    const_ops.push(op);
                    next_const_op_id += 1;

                    // Mark this value as available for dependent constants
                    if let Some(dest) = pc.dest_register {
                        tracker.record_value(pc.value, dest);
                    }

                    resolved_this_iteration.push(idx);
                }
            }
        }

        if resolved_this_iteration.is_empty() {
            // No progress - remaining constants have unsatisfiable dependencies
            // Fall back to Load encoding for all remaining
            for pc in &pending_consts {
                let fallback_enc = ConstEncoding::Load;
                if let Some(op) = resolve_pseudo_const(pc, &fallback_enc, &value_to_reg, next_const_op_id) {
                    const_ops.push(op);
                    next_const_op_id += 1;

                    if let Some(dest) = pc.dest_register {
                        tracker.record_value(pc.value, dest);
                    }
                }
            }
            break;
        }

        // Remove resolved pseudo-consts (in reverse order to preserve indices)
        for idx in resolved_this_iteration.into_iter().rev() {
            pending_consts.remove(idx);
        }

        iteration += 1;
    }

    // Prepend const_ops to ops (not append!) so dependency computation works correctly.
    // The compute_dependencies function uses array index to determine order, so const ops
    // must come BEFORE the regular ops that depend on them.
    let mut all_ops: Vec<Op> = const_ops.clone();
    all_ops.extend(std::mem::take(ops));
    *ops = all_ops;

    // Recompute dependencies to properly link const ops with their consumers
    compute_dependencies(ops);

    // Run regular scheduler - it will schedule everything optimally
    let result = schedule_greedy(ops, critical_path, enable_valu_shift, priority_mode);

    (result, const_ops)
}

/// Compute dependencies between ops based on read/write sets.
///
/// This should be called after adding new ops (like resolved pseudo-consts)
/// to ensure dependency information is up to date.
///
/// Properly tracks:
/// - RAW (Read-After-Write): Op reads register another op wrote
/// - WAR (Write-After-Read): Op writes register another op read
/// - WAW (Write-After-Write): Op writes register another op wrote
pub fn compute_dependencies(ops: &mut [Op]) {
    // Map: address -> last op that wrote to it
    let mut last_writer: HashMap<i64, usize> = HashMap::new();
    // Map: address -> ops that read since last write
    let mut readers_since_write: HashMap<i64, Vec<usize>> = HashMap::new();

    for i in 0..ops.len() {
        let reads = ops[i].reads.clone();
        let writes = ops[i].writes.clone();
        let op_id = ops[i].id;

        ops[i].dep_ids.clear();

        // RAW: depend on writers of values we read
        for addr in &reads {
            if let Some(&writer_id) = last_writer.get(addr) {
                ops[i].dep_ids.insert(writer_id);
            }
            readers_since_write
                .entry(*addr)
                .or_default()
                .push(op_id);
        }

        // WAR + WAW: depend on ops for values we write
        for addr in &writes {
            // WAW: depend on previous writer
            if let Some(&writer_id) = last_writer.get(addr) {
                ops[i].dep_ids.insert(writer_id);
            }
            // WAR: depend on all readers since last write
            if let Some(readers) = readers_since_write.get(addr) {
                for &reader_id in readers {
                    if reader_id != op_id {
                        ops[i].dep_ids.insert(reader_id);
                    }
                }
            }
            readers_since_write.remove(addr);
            last_writer.insert(*addr, op_id);
        }
    }
}

// ============================================================================
// Post-Scheduling Slot Filling
// ============================================================================

/// Result of slot filling
#[derive(Debug)]
pub struct SlotFillResult {
    /// The scheduled ops (with const ops filled in)
    pub ops: Vec<Op>,
    /// Number of cycles
    pub cycles: usize,
    /// Expanded ops from VALU→ALU shifting
    pub expanded_ops: Vec<Op>,
    /// Statistics about slot filling
    pub stats: SlotFillStats,
}

/// Statistics about slot filling
#[derive(Debug, Default)]
pub struct SlotFillStats {
    /// Number of consts placed via Flow (add_imm)
    pub flow_placements: usize,
    /// Number of consts placed via Load
    pub load_placements: usize,
    /// Number of consts placed via ALU derivation
    pub alu_placements: usize,
    /// Total constants placed
    pub total_placed: usize,
}

/// Schedule with post-pass optimization to use add_imm for constants.
///
/// This approach:
/// 1. Schedules normally (including const ops)
/// 2. Post-processes: converts const loads to add_imm where flow slots are empty at the same cycle
///
/// This preserves the dependency structure while utilizing empty flow slots.
pub fn schedule_with_slot_filling(
    ops: &mut [Op],
    critical_path: &HashMap<usize, usize>,
    enable_valu_shift: bool,
    mode: PriorityMode,
) -> SlotFillResult {
    // Step 1: Schedule normally with all ops
    let result = schedule_greedy(ops, critical_path, enable_valu_shift, mode);
    let max_cycle = result.cycles;

    // Step 2: Build slot usage map and identify const loads
    let mut flow_usage: HashMap<usize, usize> = HashMap::new();
    let mut const_op_indices: Vec<usize> = Vec::new();

    for (idx, op) in ops.iter().enumerate() {
        if op.cycle >= 0 {
            if op.engine == Engine::Flow {
                *flow_usage.entry(op.cycle as usize).or_insert(0) += 1;
            }
            // Check if this is a const load
            if op.engine == Engine::Load {
                if let Some(SlotVal::Str(ref opcode)) = op.slot.get(0) {
                    if opcode == "const" {
                        const_op_indices.push(idx);
                    }
                }
            }
        }
    }

    // Also count flow usage from expanded ops
    for op in &result.expanded_ops {
        if op.cycle >= 0 && op.engine == Engine::Flow {
            *flow_usage.entry(op.cycle as usize).or_insert(0) += 1;
        }
    }

    // Step 3: Convert const loads to add_imm where flow slot is empty
    let mut stats = SlotFillStats::default();

    for &idx in &const_op_indices {
        let op = &ops[idx];
        let cycle = op.cycle as usize;

        // Check if flow slot is available at this cycle
        let flow_used = flow_usage.get(&cycle).copied().unwrap_or(0);
        if flow_used >= Engine::Flow.slot_limit() {
            stats.load_placements += 1;
            continue;
        }

        // Get const value and dest
        if let (Some(SlotVal::Int(dest)), Some(SlotVal::Int(value))) =
            (op.slot.get(1), op.slot.get(2))
        {
            // Convert to add_imm: add_imm dest, 0, value
            // Note: ZERO_REG is always register 0
            let mut new_slot: ArrayVec<SlotVal, MAX_SLOT_SIZE> = ArrayVec::new();
            new_slot.push(SlotVal::Str("add_imm".to_string()));
            new_slot.push(SlotVal::Int(*dest));
            new_slot.push(SlotVal::Int(0)); // ZERO_REG
            new_slot.push(SlotVal::Int(*value));

            // Update the op in place
            let op_mut = &mut ops[idx];
            op_mut.engine = Engine::Flow;
            op_mut.slot = new_slot;
            op_mut.reads.insert(0); // Now reads from ZERO_REG

            // Update flow usage
            *flow_usage.entry(cycle).or_insert(0) += 1;
            stats.flow_placements += 1;
        } else {
            stats.load_placements += 1;
        }
    }

    stats.total_placed = stats.flow_placements + stats.load_placements;

    SlotFillResult {
        ops: ops.to_vec(),
        cycles: max_cycle,
        expanded_ops: result.expanded_ops,
        stats,
    }
}

// ============================================================================
// SSA Scheduling Support
// ============================================================================

use crate::types::{Operand, SsaOp, VReg};

/// Result of SSA scheduling
#[derive(Debug)]
pub struct SsaScheduleResult {
    /// Number of cycles used
    pub cycles: usize,
    /// Peak live words (from liveness analysis)
    pub peak_live: usize,
}

/// Compute RAW-only dependencies for SSA ops.
///
/// Unlike `compute_dependencies` which tracks WAW and WAR dependencies based on
/// concrete addresses, SSA ops only have true data dependencies (RAW) since each
/// VReg is defined exactly once.
///
/// `zero_vreg`: Optional reserved zero VReg. Reads from this VReg have no dependency
/// because scratch[0] is always 0 by hardware initialization (no defining op).
pub fn compute_ssa_dependencies(
    ops: &mut [SsaOp],
    vreg_def: &HashMap<VReg, usize>,
    zero_vreg: Option<VReg>,
) {
    for op in ops.iter_mut() {
        op.dep_ids.clear();

        // RAW: depend on producers of values we read
        for src in &op.srcs {
            if let Operand::VReg(vreg) = src {
                // Skip zero_vreg - it has no defining op (hardware-initialized)
                if Some(*vreg) == zero_vreg {
                    continue;
                }
                if let Some(&def_id) = vreg_def.get(vreg) {
                    op.dep_ids.insert(def_id);
                }
            }
        }
    }
}

/// Liveness tracker for SSA scheduling.
///
/// Tracks which VRegs are live at each cycle during incremental scheduling.
/// A VReg is live from its definition cycle until all its consumers are scheduled.
struct SsaLivenessTracker {
    /// VReg -> (def_cycle, size, remaining_consumer_count)
    vreg_info: HashMap<VReg, (i32, usize, usize)>,
    /// Current live words (sum of sizes of live VRegs)
    live_words: usize,
    /// Peak live words seen so far
    peak_live: usize,
    /// Scratch budget
    budget: usize,
}

impl SsaLivenessTracker {
    fn new(budget: usize) -> Self {
        Self {
            vreg_info: HashMap::new(),
            live_words: 0,
            peak_live: 0,
            budget,
        }
    }

    /// Initialize consumer counts for all VRegs.
    fn init_consumer_counts(&mut self, ops: &[SsaOp]) {
        // Count how many ops consume each VReg
        let mut consumer_counts: HashMap<VReg, usize> = HashMap::new();
        for op in ops {
            for src in &op.srcs {
                if let Operand::VReg(vreg) = src {
                    *consumer_counts.entry(*vreg).or_insert(0) += 1;
                }
            }
        }

        // Initialize vreg_info with consumer counts
        for op in ops {
            if let Some(dest) = op.dest {
                let count = consumer_counts.get(&dest).copied().unwrap_or(0);
                // def_cycle = -1 means not yet scheduled
                self.vreg_info.insert(dest, (-1, dest.size(), count));
            }
        }
    }

    /// Check if scheduling an op at the given cycle would exceed the budget.
    /// Returns the projected live words after scheduling.
    ///
    /// Uses same-cycle reuse model: VRegs that are last consumed by this op will be
    /// freed, reducing pressure. This correctly models that reads happen before writes
    /// within a cycle, so dying VRegs' storage can be reused for new VRegs.
    fn projected_live_after_schedule(&self, op: &SsaOp, _cycle: i32) -> usize {
        let mut projected = self.live_words;

        // The op's destination VReg becomes live (if it has one)
        if let Some(dest) = op.dest {
            projected += dest.size();
        }

        // Count how many times each VReg is used by this op.
        // A VReg dies if uses_by_this_op >= remaining consumers.
        let mut uses_by_op: HashMap<VReg, usize> = HashMap::new();
        for src in &op.srcs {
            if let Operand::VReg(vreg) = src {
                *uses_by_op.entry(*vreg).or_insert(0) += 1;
            }
        }

        // VRegs consumed by this op that have no other consumers will die.
        // With same-cycle reuse, we subtract their space since reads happen before writes.
        for (vreg, uses) in &uses_by_op {
            if let Some((_def_cycle, size, remaining)) = self.vreg_info.get(vreg) {
                // If this op uses all remaining consumers, the VReg dies
                if *uses >= *remaining {
                    projected = projected.saturating_sub(*size);
                }
            }
        }

        projected
    }

    /// Record that an op has been scheduled at the given cycle.
    fn schedule_op(&mut self, op: &SsaOp, cycle: i32) {
        // First, handle consumed VRegs - decrement consumer counts
        // VRegs die when their last consumer is scheduled
        for src in &op.srcs {
            if let Operand::VReg(vreg) = src {
                if let Some((def_cycle, size, ref mut remaining)) = self.vreg_info.get_mut(vreg) {
                    if *def_cycle >= 0 && *remaining > 0 {
                        *remaining -= 1;
                        if *remaining == 0 {
                            // This VReg just died - free its space
                            self.live_words = self.live_words.saturating_sub(*size);
                        }
                    }
                }
            }
        }

        // Then, the op's destination VReg becomes live
        if let Some(dest) = op.dest {
            if let Some((ref mut def_cycle, size, remaining)) = self.vreg_info.get_mut(&dest) {
                *def_cycle = cycle;
                // Only add to live words if this VReg has consumers.
                // Dead stores (remaining == 0) are filtered out by ssa_greedy before
                // scheduling, but even if they weren't, they shouldn't consume scratch
                // budget since no one reads them. This check provides defense-in-depth.
                if *remaining > 0 {
                    self.live_words += *size;
                }
            }
        }

        self.peak_live = self.peak_live.max(self.live_words);
    }

    /// Check if scheduling would exceed the budget.
    fn would_exceed_budget(&self, op: &SsaOp, cycle: i32) -> bool {
        self.projected_live_after_schedule(op, cycle) > self.budget
    }
}

/// Schedule SSA ops using a greedy list scheduler with liveness constraints.
///
/// This scheduler:
/// 1. Uses only RAW dependencies (no false WAW/WAR deps)
/// 2. Respects slot limits per engine per cycle
/// 3. Enforces scratch budget constraints via liveness tracking
/// 4. Delays ops if scheduling them would exceed the scratch budget
///
/// Returns the scheduled ops (with cycle assignments) and scheduling statistics.
pub fn schedule_ssa_greedy(ops: &mut [SsaOp]) -> SsaScheduleResult {
    schedule_ssa_greedy_with_budget(ops, SCRATCH_BUDGET)
}

/// Schedule SSA ops with a custom scratch budget.
pub fn schedule_ssa_greedy_with_budget(ops: &mut [SsaOp], budget: usize) -> SsaScheduleResult {
    if ops.is_empty() {
        return SsaScheduleResult { cycles: 0, peak_live: 0 };
    }

    // Build VReg -> defining op ID map
    let mut vreg_def: HashMap<VReg, usize> = HashMap::new();
    for op in ops.iter() {
        if let Some(dest) = op.dest {
            vreg_def.insert(dest, op.id);
        }
    }

    // Compute dependencies
    // Note: zero_vreg is None here - this standalone scheduler doesn't track it.
    // The `if let Some` in compute_ssa_dependencies handles missing vreg_def entries gracefully.
    compute_ssa_dependencies(ops, &vreg_def, None);

    // Initialize liveness tracker
    let mut liveness = SsaLivenessTracker::new(budget);
    liveness.init_consumer_counts(ops);

    // Build adjacency lists for dependents
    let mut dependents: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut in_degree: HashMap<usize, usize> = HashMap::new();

    for op in ops.iter() {
        in_degree.insert(op.id, op.dep_ids.len());
        for &dep_id in &op.dep_ids {
            dependents.entry(dep_id).or_default().push(op.id);
        }
    }

    // Initialize ready queue with ops that have no dependencies
    // Priority: (earliest_ready, op_id) - schedule earlier-ready ops first
    let mut ready: BinaryHeap<Reverse<(i32, usize)>> = BinaryHeap::new();
    let mut op_index: HashMap<usize, usize> = HashMap::new(); // op_id -> index in ops

    for (idx, op) in ops.iter().enumerate() {
        op_index.insert(op.id, idx);
        if op.dep_ids.is_empty() {
            ready.push(Reverse((0, op.id)));
        }
    }

    // Track slot usage per cycle per engine
    let mut slot_usage: HashMap<(i32, Engine), usize> = HashMap::new();

    // Track earliest ready cycle per op (based on when deps complete)
    let mut earliest_ready: HashMap<usize, i32> = HashMap::new();
    for op in ops.iter() {
        earliest_ready.insert(op.id, 0);
    }

    // Track ops that were deferred due to liveness constraints
    let mut deferred: VecDeque<usize> = VecDeque::new();

    let mut max_cycle: i32 = 0;
    let max_attempts = ops.len() * 100; // Prevent infinite loops
    let mut attempts = 0;

    // Schedule ops
    while !ready.is_empty() || !deferred.is_empty() {
        attempts += 1;
        if attempts > max_attempts {
            // Fallback: schedule remaining ops ignoring liveness constraints
            // This shouldn't happen in practice with reasonable programs
            break;
        }

        // Try to schedule from ready queue first, then deferred
        let op_id = if let Some(Reverse((_, id))) = ready.pop() {
            id
        } else if let Some(id) = deferred.pop_front() {
            id
        } else {
            break;
        };

        let idx = *op_index.get(&op_id).unwrap();
        let op = &ops[idx];
        let engine = op.engine;
        let earliest = *earliest_ready.get(&op_id).unwrap_or(&0);

        // Find earliest cycle where:
        // 1. This engine has a free slot
        // 2. Scheduling wouldn't exceed liveness budget
        let slot_limit = engine.slot_limit();
        let mut cycle = earliest;
        let mut found = false;
        let max_search_cycles = max_cycle + 100; // Don't search too far ahead

        while cycle <= max_search_cycles {
            let used = *slot_usage.get(&(cycle, engine)).unwrap_or(&0);
            if used < slot_limit {
                // Check liveness constraint
                if !liveness.would_exceed_budget(op, cycle) {
                    found = true;
                    break;
                }
            }
            cycle += 1;
        }

        if !found {
            // Couldn't find a valid cycle - defer this op
            // It might become schedulable after other ops free up liveness
            deferred.push_back(op_id);
            continue;
        }

        // Schedule the op
        ops[idx].cycle = cycle;
        *slot_usage.entry((cycle, engine)).or_insert(0) += 1;
        max_cycle = max_cycle.max(cycle);

        // Update liveness tracker
        liveness.schedule_op(&ops[idx], cycle);

        // Update dependents
        if let Some(deps) = dependents.get(&op_id) {
            for &dep_id in deps {
                // Update earliest ready time (result available next cycle)
                let ready_time = earliest_ready.entry(dep_id).or_insert(0);
                *ready_time = (*ready_time).max(cycle + 1);

                // Decrement in-degree
                let deg = in_degree.get_mut(&dep_id).unwrap();
                *deg -= 1;
                if *deg == 0 {
                    let dep_earliest = *earliest_ready.get(&dep_id).unwrap_or(&0);
                    ready.push(Reverse((dep_earliest, dep_id)));
                }
            }
        }

        // Re-add deferred ops to ready queue since liveness may have decreased
        while let Some(deferred_id) = deferred.pop_front() {
            let deferred_earliest = *earliest_ready.get(&deferred_id).unwrap_or(&0);
            ready.push(Reverse((deferred_earliest, deferred_id)));
        }
    }

    // Handle any remaining ops that couldn't be scheduled within constraints
    // Schedule them anyway (program correctness over optimality)
    for &deferred_id in &deferred {
        let idx = *op_index.get(&deferred_id).unwrap();
        if ops[idx].cycle < 0 {
            let engine = ops[idx].engine;
            let earliest = *earliest_ready.get(&deferred_id).unwrap_or(&0);
            let slot_limit = engine.slot_limit();

            let mut cycle = earliest;
            loop {
                let used = *slot_usage.get(&(cycle, engine)).unwrap_or(&0);
                if used < slot_limit {
                    break;
                }
                cycle += 1;
            }

            ops[idx].cycle = cycle;
            *slot_usage.entry((cycle, engine)).or_insert(0) += 1;
            max_cycle = max_cycle.max(cycle);
            liveness.schedule_op(&ops[idx], cycle);
        }
    }

    SsaScheduleResult {
        cycles: (max_cycle + 1) as usize,
        peak_live: liveness.peak_live,
    }
}

/// Full result of SSA scheduling pipeline
#[derive(Debug)]
pub struct SsaFullScheduleResult {
    /// Lowered concrete ops with physical addresses
    pub ops: Vec<Op>,
    /// Number of cycles
    pub cycles: usize,
    /// Peak live words
    pub peak_live: usize,
    /// VReg to address mapping
    pub vreg_to_addr: HashMap<VReg, i64>,
    /// Maximum address used
    pub max_addr: i64,
}

/// Full SSA scheduling pipeline: schedule -> assign addresses -> lower to concrete Ops.
///
/// This is the main entry point for SSA-based scheduling:
/// 1. Compute RAW dependencies from VReg definitions
/// 2. Schedule ops using greedy list scheduler
/// 3. Assign physical addresses via linear scan
/// 4. Lower SsaOps to concrete Ops
///
/// Returns concrete Ops ready for execution.
pub fn schedule_ssa_full(ssa_ops: &mut [SsaOp]) -> Result<SsaFullScheduleResult, String> {
    use crate::address_assign::{assign_addresses, lower_ssa_ops, AddressAssignConfig};

    if ssa_ops.is_empty() {
        return Ok(SsaFullScheduleResult {
            ops: Vec::new(),
            cycles: 0,
            peak_live: 0,
            vreg_to_addr: HashMap::new(),
            max_addr: 0,
        });
    }

    // Step 1: Schedule
    let schedule_result = schedule_ssa_greedy(ssa_ops);

    // Step 2: Assign addresses
    let config = AddressAssignConfig::default();
    let assign_result = assign_addresses(ssa_ops, &config);

    if assign_result.budget_exceeded {
        return Err(format!(
            "Scratch budget exceeded: peak {} > budget {}",
            assign_result.peak_live, assign_result.budget
        ));
    }

    // Step 3: Lower to concrete Ops
    let ops = lower_ssa_ops(ssa_ops, &assign_result.vreg_to_addr);

    Ok(SsaFullScheduleResult {
        ops,
        cycles: schedule_result.cycles,
        peak_live: schedule_result.peak_live,
        vreg_to_addr: assign_result.vreg_to_addr,
        max_addr: assign_result.max_addr,
    })
}

#[cfg(test)]
mod ssa_tests {
    use super::*;
    use crate::kernel_builder::emit_ssa::SsaEmitContext;
    use crate::types::Opcode;

    #[test]
    fn test_schedule_ssa_simple_chain() {
        let mut emit = SsaEmitContext::new();
        let v1 = emit.emit_const(5);
        let v2 = emit.emit_alu(Opcode::Add, v1, v1);
        let _v3 = emit.emit_alu(Opcode::Mul, v2, v2);

        let result = schedule_ssa_greedy(&mut emit.ops);

        // Chain of 3 ops should take 3 cycles
        assert_eq!(result.cycles, 3);

        // Verify cycle assignments: 0 -> 1 -> 2
        assert_eq!(emit.ops[0].cycle, 0);
        assert_eq!(emit.ops[1].cycle, 1);
        assert_eq!(emit.ops[2].cycle, 2);
    }

    #[test]
    fn test_schedule_ssa_parallel_consts() {
        let mut emit = SsaEmitContext::new();
        let _v1 = emit.emit_const(1);
        let _v2 = emit.emit_const(2);

        let result = schedule_ssa_greedy(&mut emit.ops);

        // Two independent consts can both be at cycle 0 (2 load slots)
        assert_eq!(result.cycles, 1);
        assert_eq!(emit.ops[0].cycle, 0);
        assert_eq!(emit.ops[1].cycle, 0);
    }

    #[test]
    fn test_schedule_ssa_slot_limit() {
        let mut emit = SsaEmitContext::new();
        // Create 3 independent const loads - Load engine has limit of 2
        let _v1 = emit.emit_const(1);
        let _v2 = emit.emit_const(2);
        let _v3 = emit.emit_const(3);

        let result = schedule_ssa_greedy(&mut emit.ops);

        // 2 can fit in cycle 0, 1 must go to cycle 1
        assert_eq!(result.cycles, 2);
    }

    #[test]
    fn test_schedule_ssa_no_false_deps() {
        let mut emit = SsaEmitContext::new();
        // In non-SSA, these would have WAW/WAR deps
        // In SSA, they're independent
        let v1 = emit.emit_const(1);
        let v2 = emit.emit_const(2);
        let _sum1 = emit.emit_alu(Opcode::Add, v1, v1);
        let _sum2 = emit.emit_alu(Opcode::Add, v2, v2);

        let result = schedule_ssa_greedy(&mut emit.ops);

        // v1, v2 at cycle 0 (2 loads)
        // sum1, sum2 at cycle 1 (parallel since independent, 2 ALU ops)
        assert_eq!(result.cycles, 2);
        assert_eq!(emit.ops[0].cycle, 0); // v1
        assert_eq!(emit.ops[1].cycle, 0); // v2
        assert_eq!(emit.ops[2].cycle, 1); // sum1
        assert_eq!(emit.ops[3].cycle, 1); // sum2
    }

    #[test]
    fn test_schedule_ssa_full_pipeline() {
        let mut emit = SsaEmitContext::new();
        let v1 = emit.emit_const(5);
        let v2 = emit.emit_const(10);
        let sum = emit.emit_alu(Opcode::Add, v1, v2);
        let _result = emit.emit_alu(Opcode::Mul, sum, sum);

        let result = schedule_ssa_full(&mut emit.ops).expect("Should succeed");

        // Verify scheduling
        assert_eq!(result.cycles, 3); // const, const -> add -> mul

        // Verify we got concrete ops
        assert_eq!(result.ops.len(), 4);

        // Verify addresses were assigned
        assert!(!result.vreg_to_addr.is_empty());

        // Verify concrete ops have valid addresses in reads/writes
        for op in &result.ops {
            assert!(op.cycle >= 0);
            // At least one address in reads or writes (except for loads with imm)
            // Ops should have proper engine assignments
            assert!(op.engine.is_hardware());
        }
    }

    #[test]
    fn test_schedule_ssa_full_with_vectors() {
        let mut emit = SsaEmitContext::new();
        let s1 = emit.emit_const(1);
        let s2 = emit.emit_const(2);
        let v1 = emit.emit_vbroadcast(s1);
        let v2 = emit.emit_vbroadcast(s2);
        let _vsum = emit.emit_valu(Opcode::Add, v1, v2);

        let result = schedule_ssa_full(&mut emit.ops).expect("Should succeed");

        // Verify scheduling happened
        assert!(result.cycles > 0);

        // Verify vector ops got 8-word allocations
        for (vreg, _addr) in &result.vreg_to_addr {
            if vreg.is_vector() {
                // Vector VRegs should exist in the mapping
                assert!(vreg.size() == 8);
            }
        }
    }

    #[test]
    fn test_schedule_ssa_respects_liveness_budget() {
        // Create overlapping vectors that would exceed a small budget
        // The scheduler should delay some ops to reduce peak liveness
        let mut emit = SsaEmitContext::new();
        let s1 = emit.emit_const(1);
        let v1 = emit.emit_vbroadcast_force(s1); // 8 words
        let v2 = emit.emit_vbroadcast_force(s1); // 8 words
        let v3 = emit.emit_vbroadcast_force(s1); // 8 words
        // Consumer that uses all vectors - forces them to be live simultaneously
        // if scheduled naively
        let _ = emit.emit_valu(Opcode::Add, v1, v2);
        let _ = emit.emit_valu(Opcode::Add, v2, v3);

        // With a budget of 20, we can't have all 3 vectors (24 words) live at once
        // The scheduler should sequence them
        let result = schedule_ssa_greedy_with_budget(&mut emit.ops, 20);

        // Peak liveness should respect the budget (or be close to it)
        // The exact value depends on scheduling order, but it shouldn't be 24+
        assert!(
            result.peak_live <= 20,
            "Peak liveness {} should not exceed budget 20",
            result.peak_live
        );
    }

    #[test]
    fn test_schedule_ssa_liveness_decreases_after_last_use() {
        let mut emit = SsaEmitContext::new();
        let s1 = emit.emit_const(1);
        let v1 = emit.emit_vbroadcast(s1);
        // v1 is consumed here and dies
        let v2 = emit.emit_valu(Opcode::Add, v1, v1);
        // v2 is consumed here and dies
        let _v3 = emit.emit_valu(Opcode::Mul, v2, v2);

        let result = schedule_ssa_greedy(&mut emit.ops);

        // After each use, liveness decreases, so peak should be reasonable
        // s1 (1) + v1 (8) = 9 at peak
        // Then v1 dies, v2 (8) is created: 1 + 8 = 9
        // Then v2 dies, v3 (8) is created: 1 + 8 = 9
        assert!(result.peak_live <= 10, "Peak liveness should be around 9-10");
    }

    /// Test SSA full pipeline produces valid concrete Ops.
    /// This verifies that:
    /// 1. SSA scheduling assigns valid cycle numbers
    /// 2. Address assignment produces valid physical addresses
    /// 3. Lowered Ops have correct reads/writes sets
    #[test]
    fn test_schedule_ssa_full_produces_valid_ops() {
        let mut emit = SsaEmitContext::new();
        let v1 = emit.emit_const(5);
        let v2 = emit.emit_const(10);
        let sum = emit.emit_alu(Opcode::Add, v1, v2);
        let product = emit.emit_alu(Opcode::Mul, sum, v1);
        let _final_result = emit.emit_alu(Opcode::Sub, product, v2);

        let result = schedule_ssa_full(&mut emit.ops).expect("Should succeed");

        // Verify all ops have valid cycles
        for op in &result.ops {
            assert!(op.cycle >= 0, "All ops should have valid cycle >= 0");
        }

        // Verify precedence constraints are respected
        // Find ops by their produces_constant or check cycle ordering
        let cycles: Vec<i32> = result.ops.iter().map(|op| op.cycle).collect();

        // v1, v2 can be parallel (cycle 0)
        // sum depends on v1, v2 (cycle >= 1)
        // product depends on sum, v1 (cycle >= 2)
        // final depends on product, v2 (cycle >= 3)
        assert!(cycles[2] > cycles[0], "sum must come after v1");
        assert!(cycles[2] > cycles[1], "sum must come after v2");
        assert!(cycles[3] > cycles[2], "product must come after sum");
        assert!(cycles[4] > cycles[3], "final must come after product");

        // Verify addresses are non-negative and within bounds
        for (&vreg, &addr) in &result.vreg_to_addr {
            assert!(addr >= 0, "Address should be non-negative");
            if vreg.is_vector() {
                assert!(addr + 7 <= result.max_addr, "Vector should fit within allocation");
            }
        }

        // Verify lowered ops have proper reads/writes
        for op in &result.ops {
            // Loads have no reads (or immediate), everything else should have reads
            if op.engine != Engine::Load {
                assert!(!op.reads.is_empty(), "Non-load ops should have reads: {:?}", op);
            }
            // All ops except stores should have writes
            if op.engine != Engine::Store {
                assert!(!op.writes.is_empty(), "Non-store ops should have writes: {:?}", op);
            }
        }
    }

    /// Test SSA pipeline with vectors verifies correct 8-word allocations.
    #[test]
    fn test_schedule_ssa_full_vector_allocation() {
        let mut emit = SsaEmitContext::new();
        let s1 = emit.emit_const(1);
        let s2 = emit.emit_const(2);
        let v1 = emit.emit_vbroadcast(s1);
        let v2 = emit.emit_vbroadcast(s2);
        let vsum = emit.emit_valu(Opcode::Add, v1, v2);
        let _vresult = emit.emit_valu(Opcode::Mul, vsum, v1);

        let result = schedule_ssa_full(&mut emit.ops).expect("Should succeed");

        // Find vector VRegs and verify they got 8-word allocations
        let vector_vregs: Vec<_> = result.vreg_to_addr.iter()
            .filter(|(vreg, _)| vreg.is_vector())
            .collect();

        assert!(!vector_vregs.is_empty(), "Should have vector VRegs");

        // Verify vector ops have 8 addresses in writes
        for op in &result.ops {
            if op.engine == Engine::Valu && !op.writes.is_empty() {
                assert_eq!(op.writes.len(), 8, "VALU op should write 8 addresses");
            }
        }
    }

    /// Test that SSA scheduling + lowering produces ops that can be re-scheduled.
    /// This is a "round trip" test: SSA -> lower -> verify schedulable.
    #[test]
    fn test_schedule_ssa_lowered_ops_are_schedulable() {
        let mut emit = SsaEmitContext::new();
        let v1 = emit.emit_const(5);
        let v2 = emit.emit_const(10);
        let sum = emit.emit_alu(Opcode::Add, v1, v2);
        let _result = emit.emit_alu(Opcode::Mul, sum, sum);

        // SSA schedule + lower
        let ssa_result = schedule_ssa_full(&mut emit.ops).expect("Should succeed");
        let mut lowered_ops = ssa_result.ops;

        // Clear cycle assignments to re-schedule
        for op in &mut lowered_ops {
            op.cycle = -1;
        }

        // Recompute dependencies for lowered ops (based on reads/writes)
        recompute_dependencies(&mut lowered_ops);

        // Schedule with regular greedy scheduler
        let critical_path = compute_critical_path(&lowered_ops);
        let greedy_result = schedule_greedy(&mut lowered_ops, &critical_path, false, PriorityMode::CriticalPath);

        // Should produce valid schedule
        assert!(greedy_result.cycles > 0, "Should produce valid schedule");

        // All ops should have valid cycles
        for op in &lowered_ops {
            assert!(op.cycle >= 0, "All ops should be scheduled");
        }

        // Cycle count should be similar to SSA result
        // (may differ slightly due to different scheduling heuristics)
        assert!(
            greedy_result.cycles <= ssa_result.cycles + 2,
            "Greedy on lowered ops ({}) should be close to SSA result ({})",
            greedy_result.cycles, ssa_result.cycles
        );
    }

    /// Test that SSA scheduling can work with a minimal 1-word scratch budget.
    ///
    /// This verifies same-cycle reuse: when v0's last use is in the same cycle
    /// that v1 is defined, v1 can reuse v0's address.
    ///
    /// Program:
    ///   v0 = const(5)        -- scalar, 1 word, born cycle 0
    ///   v1 = add(v0, v0)     -- scalar, 1 word, born cycle 1, v0 dies here
    ///   store(0, v1)         -- v1 dies here (cycle 2)
    /// Test same-cycle reuse with a minimal chain: const -> alu -> store.
    ///
    /// See types.rs "AUTHORITATIVE LIVENESS MODEL DOCUMENTATION" for full details.
    ///
    /// With CLOSED intervals and same-cycle reuse:
    ///   - v0: [0, 1] — defined at 0, last read at 1
    ///   - v1: [1, 2] — defined at 1, last read at 2 (by store)
    ///   - Overlap check: 0 < 2 AND 1 < 1? → 1 < 1 is FALSE → NO overlap!
    ///   - Same-cycle reuse: v0 read at cycle 1, v1 written at cycle 1
    ///   - They can share the same address (only 1 address needed).
    #[test]
    fn test_ssa_greedy_minimal_budget_2_words() {
        use crate::address_assign::{assign_addresses, AddressAssignConfig};

        let mut emit = SsaEmitContext::new();
        let v0 = emit.emit_const(5);
        let v1 = emit.emit_alu(Opcode::Add, v0, v0);
        emit.emit_store_imm(0, v1);  // Store to address 0, consumes v1

        // Schedule with budget of 1 (sufficient with same-cycle reuse)
        let result = schedule_ssa_greedy_with_budget(&mut emit.ops, 1);

        // With closed intervals and same-cycle reuse:
        // v0 [0,1] and v1 [1,2] do NOT overlap (1 < 1 is FALSE).
        // Peak = 1 word (never more than 1 live at a time due to reuse).
        assert!(
            result.peak_live <= 1,
            "Peak should be at most 1 with same-cycle reuse, got {}",
            result.peak_live
        );

        // Verify all ops are scheduled
        for op in &emit.ops {
            assert!(op.cycle >= 0, "Op {} should be scheduled", op.id);
        }

        // Address assignment needs budget of 1 with same-cycle reuse
        let config = AddressAssignConfig {
            budget: 1,
            start_addr: 1,  // Reserve 0 for ZERO_REG
            ..Default::default()
        };
        let assign_result = assign_addresses(&emit.ops, &config);

        // max_addr should be 1 (one address used: 1)
        // v0 gets addr 1, v1 reuses addr 1 (same-cycle reuse at cycle 1)
        assert_eq!(
            assign_result.max_addr, 1,
            "With closed intervals and same-cycle reuse, only 1 address needed, got max_addr={}",
            assign_result.max_addr
        );

        // peak_live should be 1 with closed intervals and same-cycle reuse:
        // v0 [0,1] expires before v1 [1,2] is allocated (1 <= 1).
        assert_eq!(
            assign_result.peak_live, 1,
            "With closed intervals, peak_live should be 1, got {}",
            assign_result.peak_live
        );

        // Should not exceed budget of 1
        assert!(
            !assign_result.budget_exceeded,
            "Should fit in budget of 1: peak_live={}, budget={}",
            assign_result.peak_live, assign_result.budget
        );
    }
}

/// Recompute dependencies for ops based on reads/writes (for testing lowered ops).
/// This creates RAW dependencies: if op B reads address X and op A wrote X, B depends on A.
#[cfg(test)]
fn recompute_dependencies(ops: &mut [Op]) {
    use std::collections::HashMap;

    // Build write map: address -> op_id (which op writes this address)
    // We need the FIRST writer for each address
    let mut write_map: HashMap<i64, usize> = HashMap::new();

    // Record which op writes each address (use original ordering)
    for op in ops.iter() {
        for &addr in &op.writes {
            // Only record first writer
            write_map.entry(addr).or_insert(op.id);
        }
    }

    // Compute dependencies based on reads
    for op in ops.iter_mut() {
        op.dep_ids.clear();
        for &addr in &op.reads {
            if let Some(&writer_id) = write_map.get(&addr) {
                if writer_id != op.id {
                    op.dep_ids.insert(writer_id);
                }
            }
        }
    }
}

#[cfg(test)]
mod pseudo_const_tests {
    use super::*;
    use crate::types::PseudoConst;

    #[test]
    fn test_const_value_tracker() {
        let mut tracker = ConstValueTracker::new(1000);

        assert!(!tracker.is_available(42));

        tracker.record_value(42, 1000);
        assert!(tracker.is_available(42));
        assert_eq!(tracker.get_register(42), Some(1000));
    }

    #[test]
    fn test_try_resolve_load() {
        let pc = PseudoConst::new(0, 42, Some(100));
        let tracker = ConstValueTracker::new(1000);
        let slot_usage: HashMap<(usize, Engine), usize> = HashMap::new();

        // With empty slot_usage and no deps, try_resolve tries encodings in order:
        // ALU (need deps), then AddImm (no deps), then Load. So we get AddImm first.
        match try_resolve_pseudo_const(&pc, 0, &tracker, &slot_usage, 1) {
            PseudoConstResolution::Resolved(op, encoding) => {
                assert_eq!(encoding, ConstEncoding::AddImm { value: 42 });
                assert_eq!(op.engine, Engine::Flow);
            }
            _ => panic!("Should resolve to AddImm or Load"),
        }
    }

    #[test]
    fn test_try_resolve_with_deps() {
        // Create pseudo-const for 4 = 2 + 2
        let pc = PseudoConst::new(0, 4, Some(100));

        // Initially, AddImm and Load are both available (no deps needed). Encodings order
        // tries AddImm before Load, so we get AddImm.
        let mut tracker = ConstValueTracker::new(1000);
        let slot_usage: HashMap<(usize, Engine), usize> = HashMap::new();

        match try_resolve_pseudo_const(&pc, 0, &tracker, &slot_usage, 1) {
            PseudoConstResolution::Resolved(_, encoding) => {
                assert_eq!(encoding, ConstEncoding::AddImm { value: 4 });
            }
            _ => panic!("Should resolve to AddImm or Load"),
        }

        // Now make value 2 available
        tracker.record_value(2, 50);

        // With load slot full but ALU available, should use Add
        let mut slot_usage_full_load: HashMap<(usize, Engine), usize> = HashMap::new();
        slot_usage_full_load.insert((0, Engine::Load), 2); // Full

        match try_resolve_pseudo_const(&pc, 0, &tracker, &slot_usage_full_load, 1) {
            PseudoConstResolution::Resolved(op, encoding) => {
                assert_eq!(encoding, ConstEncoding::Add { a_value: 2, b_value: 2 });
                assert_eq!(op.engine, Engine::Alu);
            }
            _ => panic!("Should resolve to Add"),
        }
    }
}
