//! CP-SAT based scheduler using Google OR-Tools.
//!
//! This module provides an optimal scheduler using constraint programming.
//!
//! # Problem Model
//!
//! We model the VLIW scheduling problem as a resource-constrained scheduling problem:
//!
//! ## Decision Variables
//!
//! - **Cycle variable per op**: `cycle[i]` ∈ [ASAP[i], ALAP[i]] is the cycle when op i executes
//! - **Makespan**: `makespan` = max(cycle[i]) + 1 is the total number of cycles
//!
//! ## Constraints
//!
//! 1. **Precedence constraints**: If op j depends on op i, then `cycle[j] >= cycle[i] + 1`
//!    (results are available the cycle after execution)
//!
//! 2. **Resource constraints (cumulative)**: Each engine has a limited number of slots per cycle:
//!    - ALU: 12 slots/cycle
//!    - VALU: 6 slots/cycle
//!    - Load: 2 slots/cycle
//!    - Store: 2 slots/cycle
//!    - Flow: 1 slot/cycle
//!
//!    We model this with cumulative constraints: the sum of ops executing in any cycle
//!    must not exceed the engine's capacity.
//!
//! 3. **Optional scratch memory constraint**: At most `scratch_budget` addresses can be live
//!    simultaneously (live = written but not yet consumed).
//!
//! ## Lower Bound Strengthening
//!
//! CP-SAT's cumulative constraint has a weak LP relaxation (the LP doesn't "see" that
//! ops can't share the same slot). We strengthen bounds explicitly:
//!
//! ### Capacity Bounds
//!
//! For each engine: `makespan >= ceil(num_ops / capacity)`
//!
//! Example: 9838 VALU ops with 6 slots/cycle requires at least ceil(9838/6) = 1640 cycles.
//! This is often tighter than the critical path bound.
//!
//! ### Profile Bounds (Energy-Based)
//!
//! For each deadline d, count ops that must complete by d (i.e., ALAP <= d).
//! These ops require `ceil(count / capacity)` cycles, so:
//!   - If `ceil(count / capacity) > d + 1`, the schedule is infeasible
//!   - The tightest such bound across all deadlines is a valid lower bound
//!
//! This catches situations where ops are "bunched up" in tight ASAP/ALAP windows.
//!
//! ### Suffix Bounds
//!
//! For each starting cycle t, count ops with ASAP >= t (ops that can't start before t).
//! These ops need `ceil(count / capacity)` cycles starting from t, so:
//!   - `makespan >= t + ceil(count / capacity)`
//!
//! Example: If 9837 VALU ops have ASAP >= 1, they need ceil(9837/6) = 1640 cycles
//! starting from cycle 1, giving makespan >= 1641.
//!
//! ### Sparse Window Bounds
//!
//! For each cycle t, count ops that CAN run at t (ASAP <= t <= ALAP).
//! Cycles with fewer than `capacity` available ops have "wasted" slots.
//! The bound accounts for this wasted capacity to find the minimum makespan.
//!
//! ## Objective
//!
//! Minimize makespan (total cycles).
//!
//! ## Warm Start (Hints)
//!
//! We provide the greedy schedule as hints to help CP-SAT find good solutions quickly.
//! The greedy schedule is feasible, so CP-SAT can start from a known good point and
//! focus on proving optimality or finding improvements.
//!
//! # Build Requirements
//!
//! This module requires the `cpsat` feature and OR-Tools to be installed:
//! ```bash
//! brew install or-tools
//! export ORTOOLS_PREFIX=$(brew --prefix or-tools)
//! export RUSTFLAGS='-Clink-arg=-lprotobuf'
//! maturin develop --features cpsat
//! ```

#[cfg(feature = "cpsat")]
use cp_sat::builder::{CpModelBuilder, IntVar, IntervalVar, LinearExpr};
#[cfg(feature = "cpsat")]
use cp_sat::proto::{CpSolverStatus, SatParameters};

use std::collections::{HashMap, VecDeque};

use crate::types::{Engine, Op};

/// Result from CP-SAT scheduling
#[derive(Debug, Clone)]
pub struct CpsatResult {
    /// Cycle assignment for each op (indexed by op position in input slice)
    pub cycle_assignments: Vec<i32>,
    /// Total makespan (number of cycles)
    pub makespan: usize,
    /// Solver status
    pub status: CpsatStatus,
    /// Time spent solving (seconds)
    pub solve_time_secs: f64,
    /// Optimality gap (0.0 = optimal)
    pub gap: f64,
}

/// CP-SAT solver status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpsatStatus {
    Unknown,
    ModelInvalid,
    Feasible,
    Infeasible,
    Optimal,
}

#[cfg(feature = "cpsat")]
impl From<CpSolverStatus> for CpsatStatus {
    fn from(status: CpSolverStatus) -> Self {
        match status {
            CpSolverStatus::Unknown => CpsatStatus::Unknown,
            CpSolverStatus::ModelInvalid => CpsatStatus::ModelInvalid,
            CpSolverStatus::Feasible => CpsatStatus::Feasible,
            CpSolverStatus::Infeasible => CpsatStatus::Infeasible,
            CpSolverStatus::Optimal => CpsatStatus::Optimal,
        }
    }
}

impl std::fmt::Display for CpsatStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CpsatStatus::Unknown => write!(f, "unknown"),
            CpsatStatus::ModelInvalid => write!(f, "model_invalid"),
            CpsatStatus::Feasible => write!(f, "feasible"),
            CpsatStatus::Infeasible => write!(f, "infeasible"),
            CpsatStatus::Optimal => write!(f, "optimal"),
        }
    }
}

// ============================================================================
// ASAP/ALAP Computation
// ============================================================================

/// Compute ASAP (As Soon As Possible) cycle for each op.
/// Returns a map from op_id to earliest possible cycle.
pub fn compute_asap(ops: &[Op]) -> HashMap<usize, i32> {
    let mut asap: HashMap<usize, i32> = HashMap::new();
    let mut in_degree: HashMap<usize, usize> = HashMap::new();

    for op in ops {
        in_degree.insert(op.id, op.dep_ids.len());
    }

    // Start with ops that have no dependencies
    let mut queue: VecDeque<usize> = VecDeque::new();
    for op in ops {
        if op.dep_ids.is_empty() {
            queue.push_back(op.id);
            asap.insert(op.id, 0);
        }
    }

    // Build successor map for efficient lookup
    let mut successors: HashMap<usize, Vec<usize>> = HashMap::new();
    for op in ops {
        for &dep_id in &op.dep_ids {
            successors.entry(dep_id).or_default().push(op.id);
        }
    }

    // Process in topological order
    while let Some(op_id) = queue.pop_front() {
        let my_asap = *asap.get(&op_id).unwrap_or(&0);

        if let Some(succs) = successors.get(&op_id) {
            for &succ_id in succs {
                let new_asap = my_asap + 1;
                let current = asap.entry(succ_id).or_insert(0);
                if new_asap > *current {
                    *current = new_asap;
                }

                if let Some(deg) = in_degree.get_mut(&succ_id) {
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push_back(succ_id);
                    }
                }
            }
        }
    }

    asap
}

/// Compute ALAP (As Late As Possible) cycle for each op.
/// Returns a map from op_id to latest possible cycle.
pub fn compute_alap(ops: &[Op], horizon: i32) -> HashMap<usize, i32> {
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
            alap.insert(op.id, horizon - 1);
        }
    }

    let op_by_id: HashMap<usize, &Op> = ops.iter().map(|op| (op.id, op)).collect();

    // Process in reverse topological order
    while let Some(op_id) = queue.pop_front() {
        let my_alap = *alap.get(&op_id).unwrap_or(&(horizon - 1));

        if let Some(op) = op_by_id.get(&op_id) {
            for &dep_id in &op.dep_ids {
                let new_alap = my_alap - 1;
                let current = alap.entry(dep_id).or_insert(horizon - 1);
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

// ============================================================================
// Live Interval Computation (for scratch memory tracking)
// ============================================================================

/// Live interval for scratch memory tracking
#[derive(Debug, Clone)]
pub struct LiveInterval {
    /// The scratch memory address
    pub address: i64,
    /// Op index that writes this address
    pub def_op_idx: usize,
    /// Op index that last reads this address
    pub last_use_op_idx: usize,
    /// Size (1 for scalar, 8 for vector)
    pub size: i64,
}

/// Compute live intervals for all scratch addresses.
pub fn compute_live_intervals(ops: &[Op]) -> Vec<LiveInterval> {
    // Map: address -> (def_op_idx, last_use_op_idx)
    let mut addr_info: HashMap<i64, (usize, usize)> = HashMap::new();

    // First pass: find def and last use for each address
    for (idx, op) in ops.iter().enumerate() {
        // Record writes (definitions)
        for &addr in &op.writes {
            addr_info.entry(addr).or_insert((idx, idx)).0 = idx;
        }

        // Record reads (uses) - update last use if this is later
        for &addr in &op.reads {
            if let Some(info) = addr_info.get_mut(&addr) {
                if idx > info.1 {
                    info.1 = idx;
                }
            }
        }
    }

    // Convert to LiveInterval structs
    addr_info
        .into_iter()
        .filter(|(_, (def, last_use))| def < last_use) // Only intervals that span cycles
        .map(|(address, (def_op_idx, last_use_op_idx))| LiveInterval {
            address,
            def_op_idx,
            last_use_op_idx,
            size: 1, // Assume scalar; could be extended for vectors
        })
        .collect()
}

// ============================================================================
// CP-SAT Scheduling (feature-gated)
// ============================================================================

/// Result including VALU-shift decisions
#[derive(Debug, Clone)]
pub struct CpsatResultWithShift {
    /// Base result
    pub result: CpsatResult,
    /// For each op index, true if the VALU op was shifted to ALU
    /// (empty if op is not VALU or VALU-shift not enabled)
    pub valu_shifted: Vec<bool>,
}

/// Configuration for CP-SAT scheduling
#[derive(Debug, Clone)]
pub struct CpsatConfig<'a> {
    /// Maximum scratch memory budget (0 = no constraint)
    pub scratch_budget: usize,
    /// Maximum solving time in seconds (default: 60.0)
    pub time_limit_secs: f64,
    /// Hints from greedy scheduler (op_idx -> cycle)
    pub greedy_hints: Option<&'a [i32]>,
    /// Hints for VALU-shift decisions (op_idx -> should_shift)
    pub shift_hints: Option<&'a [bool]>,
    /// If true, model VALU->ALU shifting in CP-SAT
    pub enable_valu_shift: bool,
    /// Print solver progress
    pub verbose: bool,
    /// If true, stop after finding the first feasible solution (useful for testing hints)
    pub stop_after_first_solution: bool,
}

impl<'a> Default for CpsatConfig<'a> {
    fn default() -> Self {
        Self {
            scratch_budget: 0,
            time_limit_secs: 60.0, // Sensible default: 60 seconds
            greedy_hints: None,
            shift_hints: None,
            enable_valu_shift: false,
            verbose: false,
            stop_after_first_solution: false,
        }
    }
}

impl<'a> CpsatConfig<'a> {
    /// Create a new config with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set scratch budget
    pub fn with_scratch_budget(mut self, budget: usize) -> Self {
        self.scratch_budget = budget;
        self
    }

    /// Set time limit
    pub fn with_time_limit(mut self, secs: f64) -> Self {
        self.time_limit_secs = secs;
        self
    }

    /// Set greedy hints
    pub fn with_hints(mut self, hints: &'a [i32]) -> Self {
        self.greedy_hints = Some(hints);
        self
    }

    /// Set verbose output
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Stop after finding the first feasible solution (useful for testing hints)
    pub fn with_stop_after_first_solution(mut self, stop: bool) -> Self {
        self.stop_after_first_solution = stop;
        self
    }
}

/// Schedule operations using CP-SAT solver.
///
/// # Arguments
/// * `ops` - Operations to schedule (must have dependencies computed)
/// * `config` - Configuration for the CP-SAT scheduler
///
/// # Returns
/// * `Some(CpsatResultWithShift)` if a solution was found
/// * `None` if no solution found or feature not enabled
#[cfg(feature = "cpsat")]
pub fn schedule_cpsat(ops: &[Op], config: CpsatConfig<'_>) -> Option<CpsatResultWithShift> {
    let CpsatConfig {
        scratch_budget,
        time_limit_secs,
        greedy_hints,
        shift_hints: _,
        enable_valu_shift: _,
        verbose,
        stop_after_first_solution,
    } = config;
    let start_time = std::time::Instant::now();
    let n = ops.len();

    if n == 0 {
        return Some(CpsatResultWithShift {
            result: CpsatResult {
                cycle_assignments: vec![],
                makespan: 0,
                status: CpsatStatus::Optimal,
                solve_time_secs: 0.0,
                gap: 0.0,
            },
            valu_shifted: vec![],
        });
    }

    // 1. Compute ASAP/ALAP bounds
    let asap = compute_asap(ops);
    let critical_path_len = asap.values().max().copied().unwrap_or(0) + 1;

    // Use a reasonable horizon (2x critical path or greedy hint max)
    let greedy_max = greedy_hints
        .map(|h| h.iter().max().copied().unwrap_or(0) + 1)
        .unwrap_or(0);
    let horizon = (critical_path_len * 2).max(greedy_max).max(n as i32) as i64;

    let alap = compute_alap(ops, horizon as i32);

    if verbose {
        eprintln!(
            "CP-SAT: {} ops, critical path {}, horizon {}",
            n, critical_path_len, horizon
        );
    }

    // 2. Build the CP model
    let mut model = CpModelBuilder::default();

    // Build op_id to index mapping
    let op_id_to_idx: HashMap<usize, usize> = ops
        .iter()
        .enumerate()
        .map(|(idx, op)| (op.id, idx))
        .collect();

    // 3. Create cycle variable for each op with [asap, alap] domain
    let cycle_vars: Vec<IntVar> = ops
        .iter()
        .enumerate()
        .map(|(idx, op)| {
            let lb = *asap.get(&op.id).unwrap_or(&0) as i64;
            let ub = *alap.get(&op.id).unwrap_or(&(horizon as i32 - 1)) as i64;
            model.new_int_var_with_name([(lb, ub)], format!("c{}", idx))
        })
        .collect();

    // 4. Add precedence constraints from dependencies
    for (idx, op) in ops.iter().enumerate() {
        for &dep_id in &op.dep_ids {
            if let Some(&dep_idx) = op_id_to_idx.get(&dep_id) {
                // cycle[idx] >= cycle[dep_idx] + 1
                model.add_ge(cycle_vars[idx], LinearExpr::from(cycle_vars[dep_idx]) + 1);
            }
        }
    }

    // 5. Create interval variables for slot consumption (each op occupies 1 slot for 1 cycle)
    let intervals: Vec<IntervalVar> = cycle_vars
        .iter()
        .map(|&cv| model.new_fixed_size_interval_var(cv, 1))
        .collect();

    // 6. Group ops by engine and add cumulative constraints
    let mut ops_by_engine: HashMap<Engine, Vec<usize>> = HashMap::new();
    for (idx, op) in ops.iter().enumerate() {
        if op.engine.is_hardware() {
            ops_by_engine.entry(op.engine).or_default().push(idx);
        }
    }

    for (engine, op_indices) in &ops_by_engine {
        let capacity = engine.slot_limit() as i64;
        if capacity > 0 && !op_indices.is_empty() {
            let engine_intervals: Vec<IntervalVar> =
                op_indices.iter().map(|&idx| intervals[idx]).collect();
            let demands: Vec<i64> = vec![1; op_indices.len()];
            model.add_cumulative(engine_intervals, demands, capacity);
        }
    }

    // 6b. Compute capacity-based lower bounds (used when creating makespan)
    // The cumulative constraint's LP relaxation is weak; we compute tight bounds here
    let mut capacity_lower_bound = critical_path_len as i64;
    for (engine, op_indices) in &ops_by_engine {
        let capacity = engine.slot_limit() as i64;
        if capacity > 0 && !op_indices.is_empty() {
            let num_ops = op_indices.len() as i64;
            // ceil(num_ops / capacity) is a lower bound on makespan
            let min_cycles = (num_ops + capacity - 1) / capacity;
            if min_cycles > capacity_lower_bound {
                if verbose {
                    eprintln!(
                        "  Capacity bound: {:?} has {} ops / {} slots = {} min cycles",
                        engine, num_ops, capacity, min_cycles
                    );
                }
                capacity_lower_bound = min_cycles;
            }
        }
    }

    // 6c. Compute profile-based bounds (mandatory ops in time windows)
    // For each engine, analyze ASAP/ALAP windows to find tight spots
    let mut profile_lower_bound = capacity_lower_bound;
    for (engine, op_indices) in &ops_by_engine {
        let capacity = engine.slot_limit() as i64;
        if capacity <= 0 || op_indices.is_empty() {
            continue;
        }

        // Collect (asap, alap) for each op on this engine
        let bounds: Vec<(i64, i64)> = op_indices
            .iter()
            .map(|&idx| {
                let op = &ops[idx];
                let lb = *asap.get(&op.id).unwrap_or(&0) as i64;
                let ub = *alap.get(&op.id).unwrap_or(&(horizon as i32 - 1)) as i64;
                (lb, ub)
            })
            .collect();

        // Find the max ALAP to define the window range
        let max_alap = bounds.iter().map(|(_, a)| *a).max().unwrap_or(horizon);

        // For each deadline d, count how many ops must complete by cycle d
        // (i.e., have ALAP <= d). These must fit in (d+1) * capacity slots.
        // So we need makespan >= ceil(mandatory_count / capacity) for all such deadlines.
        // Also, ops with ALAP = d contribute "energy" that must finish by d+1.
        for d in 0..=max_alap.min(horizon) {
            // Count ops that must complete by cycle d (ALAP <= d)
            let mandatory_count: i64 = bounds
                .iter()
                .filter(|(_, alap_val)| *alap_val <= d)
                .count() as i64;

            if mandatory_count > 0 {
                // mandatory_count ops must fit in (d+1) cycles with capacity slots each
                let available_slots = (d + 1) * capacity;
                if mandatory_count > available_slots {
                    // Infeasibility - more ops than slots available by deadline
                    let needed_cycles = (mandatory_count + capacity - 1) / capacity;
                    if needed_cycles > profile_lower_bound {
                        if verbose {
                            eprintln!(
                                "  Profile bound: {:?} has {} ops with ALAP <= {} needing {} cycles",
                                engine, mandatory_count, d, needed_cycles
                            );
                        }
                        profile_lower_bound = needed_cycles;
                    }
                }
            }
        }
    }

    // 6d. Compute suffix-based bounds
    // For each starting cycle t, count ops that must start at or after t (ASAP >= t).
    // These ops need ceil(count/capacity) more cycles, so makespan >= t + ceil(count/capacity).
    let mut suffix_lower_bound = profile_lower_bound;
    for (engine, op_indices) in &ops_by_engine {
        let capacity = engine.slot_limit() as i64;
        if capacity <= 0 || op_indices.is_empty() {
            continue;
        }

        // Collect ASAP values for ops on this engine
        let asap_values: Vec<i64> = op_indices
            .iter()
            .map(|&idx| {
                let op = &ops[idx];
                *asap.get(&op.id).unwrap_or(&0) as i64
            })
            .collect();

        let max_asap = asap_values.iter().max().copied().unwrap_or(0);

        // For each possible starting cycle, count suffix ops
        for t in 0..=max_asap {
            let suffix_count: i64 = asap_values.iter().filter(|&&a| a >= t).count() as i64;
            if suffix_count > 0 {
                // These ops must execute in cycles [t, makespan-1], so:
                // makespan >= t + ceil(suffix_count / capacity)
                let required_cycles_after_t = (suffix_count + capacity - 1) / capacity;
                let min_makespan = t + required_cycles_after_t;
                if min_makespan > suffix_lower_bound {
                    if verbose {
                        eprintln!(
                            "  Suffix bound: {:?} has {} ops with ASAP >= {} needing {} more cycles = {} total",
                            engine, suffix_count, t, required_cycles_after_t, min_makespan
                        );
                    }
                    suffix_lower_bound = min_makespan;
                }
            }
        }
    }

    // 6e. Compute sparse window bounds
    // For each cycle, count how many ops CAN run there (ASAP <= t <= ALAP).
    // Cycles with fewer than `capacity` available ops have "wasted" slots.
    // We find the minimum makespan needed to complete all ops given this constraint.
    let mut sparse_lower_bound = suffix_lower_bound;
    for (engine, op_indices) in &ops_by_engine {
        let capacity = engine.slot_limit() as i64;
        if capacity <= 0 || op_indices.is_empty() {
            continue;
        }

        let num_ops = op_indices.len() as i64;

        // Collect (asap, alap) for each op on this engine
        let bounds: Vec<(i64, i64)> = op_indices
            .iter()
            .map(|&idx| {
                let op = &ops[idx];
                let lb = *asap.get(&op.id).unwrap_or(&0) as i64;
                let ub = *alap.get(&op.id).unwrap_or(&(horizon as i32 - 1)) as i64;
                (lb, ub)
            })
            .collect();

        let min_asap = bounds.iter().map(|(a, _)| *a).min().unwrap_or(0);
        let max_alap = bounds.iter().map(|(_, a)| *a).max().unwrap_or(horizon);

        // Compute availability profile: for each cycle, how many ops can run there
        // Then find minimum makespan to fit all ops
        let window_size = (max_alap - min_asap + 1) as usize;
        if window_size > 10000 {
            // Skip very large windows to avoid O(n*m) computation
            continue;
        }

        // Count ops available at each cycle in the window
        let mut available: Vec<i64> = vec![0; window_size];
        for (op_asap, op_alap) in &bounds {
            let start = (*op_asap - min_asap) as usize;
            let end = (*op_alap - min_asap) as usize;
            for t in start..=end.min(window_size - 1) {
                available[t] += 1;
            }
        }

        // Compute cumulative capacity: total ops we can schedule in cycles [0, m-1]
        // At each cycle t, we can schedule min(available[t], capacity) ops
        let mut cumulative_capacity = 0i64;
        for (m, &avail) in available.iter().enumerate() {
            cumulative_capacity += avail.min(capacity);
            // Check if we can fit all ops by cycle m (inclusive)
            // makespan = min_asap + m + 1
            if cumulative_capacity >= num_ops {
                let min_makespan = min_asap + (m as i64) + 1;
                if min_makespan > sparse_lower_bound {
                    if verbose {
                        eprintln!(
                            "  Sparse window bound: {:?} needs makespan >= {} (min_asap={}, window_cycles={})",
                            engine, min_makespan, min_asap, m + 1
                        );
                    }
                    sparse_lower_bound = min_makespan;
                }
                break;
            }
        }
    }

    let computed_lower_bound = sparse_lower_bound;
    if verbose && computed_lower_bound > critical_path_len as i64 {
        eprintln!(
            "  Lower bound improved: critical_path={}, capacity={}, profile={}, suffix={}, sparse={}",
            critical_path_len, capacity_lower_bound, profile_lower_bound, suffix_lower_bound, sparse_lower_bound
        );
    }

    // 7. Optional: Add scratch budget constraint
    if scratch_budget > 0 {
        let live_intervals = compute_live_intervals(ops);

        if !live_intervals.is_empty() {
            if verbose {
                eprintln!(
                    "CP-SAT: {} live intervals for scratch tracking",
                    live_intervals.len()
                );
            }

            let mut scratch_intervals: Vec<IntervalVar> = Vec::new();
            let mut scratch_demands: Vec<i64> = Vec::new();

            for li in &live_intervals {
                let def_var = cycle_vars[li.def_op_idx];
                let last_use_var = cycle_vars[li.last_use_op_idx];

                // Size = last_use_cycle - def_cycle + 1
                let size_var = model.new_int_var([(1, horizon)]);
                model.add_eq(
                    size_var,
                    LinearExpr::from(last_use_var) - LinearExpr::from(def_var) + 1,
                );

                // End = last_use + 1
                let end_var = model.new_int_var([(0, horizon + 1)]);
                model.add_eq(end_var, LinearExpr::from(last_use_var) + 1);

                let interval = model.new_interval_var(def_var, size_var, end_var);
                scratch_intervals.push(interval);
                scratch_demands.push(li.size);
            }

            if !scratch_intervals.is_empty() {
                model.add_cumulative(scratch_intervals, scratch_demands, scratch_budget as i64);
            }
        }
    }

    // 8. Add hints from greedy scheduler
    if let Some(hints) = greedy_hints {
        for (idx, &cycle) in hints.iter().enumerate() {
            if idx < cycle_vars.len() && cycle >= 0 {
                model.add_hint(cycle_vars[idx], cycle as i64);
            }
        }
    }

    // 9. Create makespan variable and minimize it
    // Use tight bounds: computed_lower_bound from capacity/profile analysis,
    // greedy hint provides good upper bound
    let greedy_ub = greedy_hints
        .map(|h| h.iter().max().copied().unwrap_or(0) as i64 + 1)
        .unwrap_or(horizon);
    let makespan = model.new_int_var([(computed_lower_bound, greedy_ub)]);
    for &cv in &cycle_vars {
        model.add_le(cv, LinearExpr::from(makespan) - 1);
    }
    model.minimize(makespan);

    // 10. Add makespan hint if we have greedy solution
    if let Some(hints) = greedy_hints {
        let greedy_makespan = hints.iter().max().copied().unwrap_or(0) as i64 + 1;
        model.add_hint(makespan, greedy_makespan);

        // Debug: check for hints outside variable domains
        if verbose {
            let mut violations = 0;
            for (idx, &hint_cycle) in hints.iter().enumerate() {
                if idx < ops.len() {
                    let op = &ops[idx];
                    let lb = *asap.get(&op.id).unwrap_or(&0);
                    let ub = *alap.get(&op.id).unwrap_or(&(horizon as i32 - 1));
                    if hint_cycle < lb || hint_cycle > ub {
                        violations += 1;
                        if violations <= 10 {
                            eprintln!(
                                "  Hint violation: op {} (idx={}) hint={} but domain=[{}, {}]",
                                op.id, idx, hint_cycle, lb, ub
                            );
                        }
                    }
                }
            }
            if violations > 0 {
                eprintln!("  Total hint violations: {} / {}", violations, hints.len());
            } else {
                eprintln!("  All {} hints are within variable domains", hints.len());
            }
        }
    }

    // 11. Configure solver parameters
    let mut params = SatParameters::default();
    params.max_time_in_seconds = Some(time_limit_secs);

    // Let CP-SAT use its default LP behavior (linearization_level=1)
    // LP relaxation helps with bounds and reduced-cost fixing for scheduling

    // Stop early if we find a good solution (within 2% of greedy)
    params.relative_gap_limit = Some(0.02);

    // Option to stop after finding first feasible solution (useful for testing hints)
    if stop_after_first_solution {
        params.stop_after_first_solution = Some(true);
    }

    if verbose {
        params.log_search_progress = Some(true);
    }

    // 11. Solve
    let response = model.solve_with_parameters(&params);
    let solve_time = start_time.elapsed().as_secs_f64();

    let status = CpsatStatus::from(response.status());

    if verbose {
        eprintln!("CP-SAT: status={:?}, time={:.2}s", status, solve_time);
    }

    // 12. Extract solution
    match status {
        CpsatStatus::Optimal | CpsatStatus::Feasible => {
            let cycle_assignments: Vec<i32> = cycle_vars
                .iter()
                .map(|&cv| cv.solution_value(&response) as i32)
                .collect();

            let actual_makespan = cycle_assignments.iter().max().copied().unwrap_or(0) + 1;

            let gap = if status == CpsatStatus::Optimal {
                0.0
            } else {
                let best = response.objective_value;
                let bound = response.best_objective_bound;
                if best > 0.0 {
                    (best - bound).abs() / best
                } else {
                    0.0
                }
            };

            // TODO: When enable_valu_shift is true, extract shift decisions from solver
            // For now, return empty valu_shifted (no shifting in CP-SAT)
            let valu_shifted = vec![false; n];

            Some(CpsatResultWithShift {
                result: CpsatResult {
                    cycle_assignments,
                    makespan: actual_makespan as usize,
                    status,
                    solve_time_secs: solve_time,
                    gap,
                },
                valu_shifted,
            })
        }
        _ => None,
    }
}

/// Stub implementation when cpsat feature is not enabled
#[cfg(not(feature = "cpsat"))]
pub fn schedule_cpsat(_ops: &[Op], _config: CpsatConfig<'_>) -> Option<CpsatResultWithShift> {
    eprintln!("CP-SAT scheduler not available: compile with --features cpsat");
    None
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Schedule with CP-SAT and fall back to greedy if it fails.
///
/// This function tries CP-SAT first, and if it fails or times out,
/// falls back to the greedy scheduler.
pub fn schedule_optimal(
    ops: &mut [Op],
    scratch_budget: usize,
    cpsat_time_limit: f64,
    verbose: bool,
) -> CpsatResult {
    use crate::scheduler::{
        compute_critical_path, schedule_greedy_with_diagnostics, PriorityMode,
    };

    // Get greedy solution first (for hints and fallback)
    let critical_path = compute_critical_path(ops);
    let greedy_result = schedule_greedy_with_diagnostics(
        ops,
        &critical_path,
        true, // enable_valu_shift
        PriorityMode::CriticalPath,
        false, // collect_diagnostics
    );

    let greedy_cycles: Vec<i32> = ops.iter().map(|op| op.cycle).collect();

    if verbose {
        eprintln!("Greedy scheduler: {} cycles", greedy_result.cycles);
    }

    // Reset cycles before CP-SAT
    for op in ops.iter_mut() {
        op.cycle = -1;
    }

    // Try CP-SAT with greedy hints (no VALU-shift in model, will apply as post-pass)
    let cpsat_result = schedule_cpsat(
        ops,
        CpsatConfig {
            scratch_budget,
            time_limit_secs: cpsat_time_limit,
            greedy_hints: Some(&greedy_cycles),
            shift_hints: None,
            enable_valu_shift: false,
            verbose,
            stop_after_first_solution: false,
        },
    );

    if let Some(result_with_shift) = cpsat_result {
        // Apply CP-SAT solution to ops
        for (idx, op) in ops.iter_mut().enumerate() {
            if idx < result_with_shift.result.cycle_assignments.len() {
                op.cycle = result_with_shift.result.cycle_assignments[idx];
            }
        }

        if verbose {
            eprintln!(
                "CP-SAT result: {} cycles (greedy was {})",
                result_with_shift.result.makespan, greedy_result.cycles
            );
        }

        result_with_shift.result
    } else {
        // Fall back to greedy
        if verbose {
            eprintln!("CP-SAT failed, using greedy result");
        }
        for (idx, &cycle) in greedy_cycles.iter().enumerate() {
            if idx < ops.len() {
                ops[idx].cycle = cycle;
            }
        }
        CpsatResult {
            cycle_assignments: greedy_cycles,
            makespan: greedy_result.cycles,
            status: CpsatStatus::Feasible,
            solve_time_secs: 0.0,
            gap: 0.0,
        }
    }
}

/// Check if CP-SAT feature is available
pub fn cpsat_available() -> bool {
    cfg!(feature = "cpsat")
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Engine, Op, SlotVal, MAX_SLOT_SIZE};
    use arrayvec::ArrayVec;
    use std::collections::HashSet;

    fn make_op(id: usize, engine: Engine, deps: Vec<usize>) -> Op {
        Op {
            id,
            engine,
            slot: {
                let mut s: ArrayVec<SlotVal, MAX_SLOT_SIZE> = ArrayVec::new();
                s.push(SlotVal::Str("+".to_string()));
                s.push(SlotVal::Int(id as i64 * 10));
                s.push(SlotVal::Int(0));
                s.push(SlotVal::Int(1));
                s
            },
            cycle: -1,
            dep_ids: deps.into_iter().collect(),
            reads: [0, 1].into_iter().collect(),
            writes: [id as i64 * 10].into_iter().collect(),
            last_use_reads: HashSet::new(),
            produces_constant: None,
        }
    }

    #[test]
    fn test_compute_asap_chain() {
        let ops = vec![
            make_op(0, Engine::Alu, vec![]),
            make_op(1, Engine::Alu, vec![0]),
            make_op(2, Engine::Alu, vec![1]),
        ];

        let asap = compute_asap(&ops);
        assert_eq!(asap.get(&0), Some(&0));
        assert_eq!(asap.get(&1), Some(&1));
        assert_eq!(asap.get(&2), Some(&2));
    }

    #[test]
    fn test_compute_asap_parallel() {
        let ops = vec![
            make_op(0, Engine::Alu, vec![]),
            make_op(1, Engine::Alu, vec![]),
            make_op(2, Engine::Alu, vec![0, 1]),
        ];

        let asap = compute_asap(&ops);
        assert_eq!(asap.get(&0), Some(&0));
        assert_eq!(asap.get(&1), Some(&0));
        assert_eq!(asap.get(&2), Some(&1));
    }

    #[test]
    fn test_compute_alap_chain() {
        let ops = vec![
            make_op(0, Engine::Alu, vec![]),
            make_op(1, Engine::Alu, vec![0]),
            make_op(2, Engine::Alu, vec![1]),
        ];

        let alap = compute_alap(&ops, 5);
        assert_eq!(alap.get(&2), Some(&4)); // Sink at horizon-1
        assert_eq!(alap.get(&1), Some(&3)); // One before sink
        assert_eq!(alap.get(&0), Some(&2)); // Two before sink
    }

    #[test]
    fn test_compute_live_intervals() {
        let mut ops = vec![
            make_op(0, Engine::Load, vec![]),
            make_op(1, Engine::Alu, vec![0]),
            make_op(2, Engine::Alu, vec![1]),
        ];
        // Op 0 writes to address 100
        ops[0].writes = [100].into_iter().collect();
        ops[0].reads.clear();
        // Op 1 reads 100, writes 110
        ops[1].reads = [100].into_iter().collect();
        ops[1].writes = [110].into_iter().collect();
        // Op 2 reads 110
        ops[2].reads = [110].into_iter().collect();
        ops[2].writes = [120].into_iter().collect();

        let intervals = compute_live_intervals(&ops);
        assert!(!intervals.is_empty());

        // Address 100 should have def=0, last_use=1
        let int_100 = intervals.iter().find(|i| i.address == 100);
        assert!(int_100.is_some());
        let int_100 = int_100.unwrap();
        assert_eq!(int_100.def_op_idx, 0);
        assert_eq!(int_100.last_use_op_idx, 1);
    }

    #[cfg(feature = "cpsat")]
    #[test]
    fn test_schedule_cpsat_empty() {
        let ops: Vec<Op> = vec![];
        let result = schedule_cpsat(&ops, CpsatConfig::new().with_time_limit(10.0));
        assert!(result.is_some());
        let r = result.unwrap();
        assert_eq!(r.result.makespan, 0);
        assert_eq!(r.result.status, CpsatStatus::Optimal);
    }

    #[cfg(feature = "cpsat")]
    #[test]
    fn test_schedule_cpsat_simple_chain() {
        let ops = vec![
            make_op(0, Engine::Alu, vec![]),
            make_op(1, Engine::Alu, vec![0]),
            make_op(2, Engine::Alu, vec![1]),
        ];

        let result = schedule_cpsat(&ops, CpsatConfig::new().with_time_limit(10.0));
        assert!(result.is_some());

        let r = result.unwrap();
        assert!(r.result.makespan >= 3); // At least 3 cycles due to dependencies
        assert_eq!(r.result.cycle_assignments.len(), 3);

        // Verify precedence: cycle[1] > cycle[0], cycle[2] > cycle[1]
        assert!(r.result.cycle_assignments[1] > r.result.cycle_assignments[0]);
        assert!(r.result.cycle_assignments[2] > r.result.cycle_assignments[1]);
    }

    #[cfg(feature = "cpsat")]
    #[test]
    fn test_schedule_cpsat_parallel() {
        // Two independent chains that can run in parallel
        let ops = vec![
            make_op(0, Engine::Alu, vec![]),
            make_op(1, Engine::Alu, vec![]),
            make_op(2, Engine::Alu, vec![0]),
            make_op(3, Engine::Alu, vec![1]),
        ];

        let result = schedule_cpsat(&ops, CpsatConfig::new().with_time_limit(10.0));
        assert!(result.is_some());

        let r = result.unwrap();
        // Should be able to schedule in 2 cycles since chains are independent
        assert!(r.result.makespan <= 2);
    }

    #[cfg(feature = "cpsat")]
    #[test]
    fn test_schedule_cpsat_slot_limits() {
        // 15 ALU ops with no dependencies - should be limited by ALU slots (12)
        let ops: Vec<Op> = (0..15)
            .map(|i| {
                let mut op = make_op(i, Engine::Alu, vec![]);
                op.writes = [(i * 10) as i64].into_iter().collect();
                op
            })
            .collect();

        let result = schedule_cpsat(&ops, CpsatConfig::new().with_time_limit(10.0));
        assert!(result.is_some());

        let r = result.unwrap();
        // 15 ops with 12 ALU slots = at least 2 cycles
        assert!(r.result.makespan >= 2);
    }

    #[cfg(feature = "cpsat")]
    #[test]
    fn test_schedule_cpsat_with_hints() {
        let ops = vec![
            make_op(0, Engine::Alu, vec![]),
            make_op(1, Engine::Alu, vec![0]),
            make_op(2, Engine::Alu, vec![1]),
        ];

        // Provide hints
        let hints = vec![0, 1, 2];

        let result = schedule_cpsat(
            &ops,
            CpsatConfig::new().with_time_limit(10.0).with_hints(&hints),
        );
        assert!(result.is_some());

        let r = result.unwrap();
        assert_eq!(r.result.makespan, 3);
        // With hints, should find the optimal solution quickly
        assert!(r.result.status == CpsatStatus::Optimal);
    }

    #[test]
    fn test_cpsat_available() {
        // This just tests that the function compiles and returns a bool
        let available = cpsat_available();
        // When compiled with --features cpsat, this should be true
        // When compiled without, this should be false
        assert!(available == cfg!(feature = "cpsat"));
    }

    /// Test that a valid hint round-trips correctly.
    /// This verifies our hint encoding matches the constraint encoding.
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_hint_roundtrip() {
        // Create a simple chain: op0 -> op1 -> op2
        let ops = vec![
            make_op(0, Engine::Alu, vec![]),
            make_op(1, Engine::Alu, vec![0]),
            make_op(2, Engine::Alu, vec![1]),
        ];

        // Valid schedule: op0 at cycle 0, op1 at cycle 1, op2 at cycle 2
        let hints = vec![0, 1, 2];

        let result = schedule_cpsat(
            &ops,
            CpsatConfig::new()
                .with_time_limit(10.0)
                .with_hints(&hints)
                .with_verbose(true),
        );
        assert!(result.is_some(), "CP-SAT should accept valid hints");

        let r = result.unwrap();
        // With a valid hint and sufficient time, should return the hinted solution or better
        assert!(
            r.result.status == CpsatStatus::Optimal || r.result.status == CpsatStatus::Feasible,
            "Should find feasible solution with valid hints"
        );
        assert_eq!(r.result.makespan, 3, "Makespan should be 3 for this chain");

        // Verify the cycle assignments respect dependencies
        assert!(
            r.result.cycle_assignments[1] > r.result.cycle_assignments[0],
            "op1 must come after op0"
        );
        assert!(
            r.result.cycle_assignments[2] > r.result.cycle_assignments[1],
            "op2 must come after op1"
        );
    }

    /// Test that hints within ASAP/ALAP bounds are accepted.
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_hint_bounds_check() {
        // Create ops and compute ASAP/ALAP
        let ops = vec![
            make_op(0, Engine::Alu, vec![]),
            make_op(1, Engine::Alu, vec![0]),
            make_op(2, Engine::Alu, vec![1]),
        ];

        let asap = compute_asap(&ops);
        let alap = compute_alap(&ops, 10);

        // Verify our understanding of ASAP/ALAP
        assert_eq!(*asap.get(&0).unwrap(), 0, "op0 ASAP should be 0");
        assert_eq!(*asap.get(&1).unwrap(), 1, "op1 ASAP should be 1");
        assert_eq!(*asap.get(&2).unwrap(), 2, "op2 ASAP should be 2");

        // Create hints that are exactly at ASAP (should be valid)
        let hints_at_asap = vec![0, 1, 2];

        let result = schedule_cpsat(
            &ops,
            CpsatConfig {
                greedy_hints: Some(&hints_at_asap),
                ..Default::default()
            },
        );
        assert!(result.is_some(), "Hints at ASAP should be valid");

        // Create hints that are within bounds but not at ASAP
        let hints_later = vec![2, 3, 4];

        let result = schedule_cpsat(
            &ops,
            CpsatConfig {
                greedy_hints: Some(&hints_later),
                ..Default::default()
            },
        );
        assert!(result.is_some(), "Hints within ALAP should be valid");
    }

    /// Test that hint indices match cycle_vars indices.
    /// This catches the bug where op.id != slice index.
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_hint_index_mapping() {
        // Create ops with non-sequential IDs to test index mapping
        let ops = vec![
            make_op(100, Engine::Alu, vec![]),      // slice idx 0, op.id 100
            make_op(200, Engine::Alu, vec![100]),   // slice idx 1, op.id 200
            make_op(300, Engine::Alu, vec![200]),   // slice idx 2, op.id 300
        ];

        // Verify the ops have the expected structure
        assert_eq!(ops[0].id, 100);
        assert_eq!(ops[1].id, 200);
        assert_eq!(ops[2].id, 300);

        // Hints are indexed by slice position, not op.id
        // So hints[0] is for ops[0] (which has id=100)
        let hints = vec![0, 1, 2];

        let result = schedule_cpsat(
            &ops,
            CpsatConfig::new()
                .with_time_limit(10.0)
                .with_hints(&hints)
                .with_verbose(true),
        );
        assert!(
            result.is_some(),
            "Should handle non-sequential op.id correctly"
        );

        let r = result.unwrap();
        assert_eq!(r.result.cycle_assignments.len(), 3);
        // The assignments should respect the dependency chain
        assert!(r.result.cycle_assignments[1] > r.result.cycle_assignments[0]);
        assert!(r.result.cycle_assignments[2] > r.result.cycle_assignments[1]);
    }

    /// Test that stop_after_first_solution returns exactly the hint.
    /// This ensures hint encoding/decoding is correct.
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_hint_roundtrip_stop_after_first() {
        // Create a simple chain: op0 -> op1 -> op2
        let ops = vec![
            make_op(0, Engine::Alu, vec![]),
            make_op(1, Engine::Alu, vec![0]),
            make_op(2, Engine::Alu, vec![1]),
        ];

        // Valid schedule: op0 at cycle 0, op1 at cycle 1, op2 at cycle 2
        let hints = vec![0, 1, 2];

        // Run solver with stop_after_first_solution - should return hint immediately
        let result = schedule_cpsat(
            &ops,
            CpsatConfig::new()
                .with_hints(&hints)
                .with_stop_after_first_solution(true)
                .with_verbose(true),
        );

        assert!(result.is_some(), "Should return a solution with hint");

        let r = result.unwrap();
        // With stop_after_first_solution, we get exactly the hint back
        assert_eq!(
            r.result.makespan, 3,
            "Makespan should be 3 (the hinted value)"
        );
        // May be optimal if proved during presolve, or feasible if stopped early
        assert!(
            r.result.status == CpsatStatus::Feasible || r.result.status == CpsatStatus::Optimal,
            "Should return feasible or optimal"
        );
        // The cycle assignments should equal the hints exactly
        assert_eq!(
            r.result.cycle_assignments, hints,
            "Should return exactly the hinted solution"
        );
    }
}
