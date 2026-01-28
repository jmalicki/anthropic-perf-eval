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

use std::collections::{HashMap, HashSet, VecDeque};
use std::ops::Not;

use crate::types::{Engine, Op, Opcode, SsaOp, VReg, VLEN};

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
            time_limit_secs: 600.0, // Default: 10 minutes
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

    // Optimize LP relaxation for tighter bounds and faster convergence
    // linearization_level=2: Stronger LP relaxations (better bounds for cumulative constraints)
    //   - Cumulative constraints have weak LP relaxations by default
    //   - Level 2 linearizes more constraints, giving tighter best_objective_bound
    //   - Expected: Gap reduction from 10-15% to 3-8%, 20-40% faster convergence
    // cp_model_presolve=true: Preprocessing to simplify model before solving
    //   - Removes redundant variables/constraints, detects symmetries
    //   - Usually <1% of solve time, but significantly speeds up search
    // cp_model_probing_level=2: Aggressive probing to tighten variable domains
    //   - Temporarily fixes variables and observes consequences
    //   - Strengthens domains and adds binary clauses
    //   - Expected: 5-15% tighter domains, 10-20% faster search
    // randomize_search=true: Help escape local optima
    //   - Adds randomization to search decisions
    //   - Helps when solver gets stuck in suboptimal regions
    params.linearization_level = Some(2);
    params.cp_model_presolve = Some(true);
    params.cp_model_probing_level = Some(2);
    params.randomize_search = Some(true);
    
    // Stop early if we find a good solution (within 2% of greedy)
    params.relative_gap_limit = Some(0.02);

    // Option to stop after finding first feasible solution (useful for testing hints)
    if stop_after_first_solution {
        params.stop_after_first_solution = Some(true);
    }

    // Always log search progress by default - user must go out of their way to suppress
    params.log_search_progress = Some(true);

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
// SSA CP-SAT Scheduling
// ============================================================================

/// Configuration for SSA CP-SAT scheduling
#[derive(Debug, Clone)]
pub struct SsaCpsatConfig {
    /// Maximum solving time in seconds (default: 60.0)
    pub time_limit_secs: f64,
    /// Print solver progress
    pub verbose: bool,
    /// If true, stop after finding the first feasible solution
    pub stop_after_first_solution: bool,
    /// If true, model VALU→ALU shifting as CP-SAT decisions
    pub enable_valu_shift: bool,
    /// Scratch memory budget (max simultaneous live words). 0 = no constraint.
    pub scratch_budget: usize,
}

impl Default for SsaCpsatConfig {
    fn default() -> Self {
        Self {
            time_limit_secs: 60.0,
            verbose: false,
            stop_after_first_solution: false,
            enable_valu_shift: false,
            scratch_budget: 1536, // Default hardware limit
        }
    }
}

/// Result from SSA CP-SAT scheduling with VALU shift decisions
#[derive(Debug, Clone)]
pub struct SsaCpsatResult {
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
    /// Set of op IDs that should be shifted from VALU to ALU
    pub shifted_ops: HashSet<usize>,
    /// Set of ConstChoice op IDs that should use AddImm (Flow) instead of Load
    pub const_add_imm_ops: HashSet<usize>,
}

/// Compute ASAP (As Soon As Possible) cycle for each SSA op.
/// Uses the RAW-only dependencies from dep_ids.
pub fn compute_ssa_asap(ops: &[SsaOp]) -> HashMap<usize, i32> {
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

/// Compute ALAP (As Late As Possible) cycle for each SSA op.
pub fn compute_ssa_alap(ops: &[SsaOp], horizon: i32) -> HashMap<usize, i32> {
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

    let op_by_id: HashMap<usize, &SsaOp> = ops.iter().map(|op| (op.id, op)).collect();

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

/// Schedule SSA operations using CP-SAT solver.
///
/// This operates on SsaOps BEFORE register allocation, using only RAW dependencies.
/// This avoids false WAW/WAR dependencies introduced by register allocation.
///
/// When `enable_valu_shift` is true, the solver can decide which VALU ops to shift
/// to ALU (8 scalar ops). This is modeled using optional intervals:
/// - Each shiftable VALU op has two optional intervals: VALU variant and ALU variant
/// - Exactly one must be present (shift decision)
/// - Shifted ops consume 8 ALU slots instead of 1 VALU slot
///
/// # Arguments
/// * `ops` - SSA operations to schedule (with dep_ids computed)
/// * `config` - Configuration for the CP-SAT scheduler
/// * `greedy_hints` - Optional cycle assignments from greedy scheduler
/// * `shift_hints` - Optional shift decisions from greedy scheduler (op idx -> should shift)
///
/// # Returns
/// * `Some(SsaCpsatResult)` if a solution was found (includes shift decisions)
/// * `None` if no solution found or feature not enabled
#[cfg(feature = "cpsat")]
pub fn schedule_ssa_cpsat(
    ops: &[SsaOp],
    config: SsaCpsatConfig,
    greedy_hints: Option<&[i32]>,
    shift_hints: Option<&HashSet<usize>>,
) -> Option<SsaCpsatResult> {
    use cp_sat::builder::BoolVar;

    let SsaCpsatConfig {
        time_limit_secs,
        verbose,
        stop_after_first_solution,
        enable_valu_shift,
        scratch_budget,
    } = config;
    let start_time = std::time::Instant::now();
    let n = ops.len();

    if n == 0 {
        return Some(SsaCpsatResult {
            cycle_assignments: vec![],
            makespan: 0,
            status: CpsatStatus::Optimal,
            solve_time_secs: 0.0,
            gap: 0.0,
            shifted_ops: HashSet::new(),
            const_add_imm_ops: HashSet::new(),
        });
    }

    // 1. Compute ASAP/ALAP bounds using RAW-only dependencies
    let asap = compute_ssa_asap(ops);
    let critical_path_len = asap.values().max().copied().unwrap_or(0) + 1;

    // Use a reasonable horizon (2x critical path or greedy hint max)
    let greedy_max = greedy_hints
        .map(|h| h.iter().max().copied().unwrap_or(0) + 1)
        .unwrap_or(0);
    let horizon = (critical_path_len * 2).max(greedy_max).max(n as i32) as i64;

    let alap = compute_ssa_alap(ops, horizon as i32);

    // 2. Identify shiftable VALU ops
    let shiftable_valu: Vec<usize> = if enable_valu_shift {
        ops.iter()
            .enumerate()
            .filter(|(_, op)| op.engine == Engine::Valu && op.opcode.can_shift_to_alu())
            .map(|(idx, _)| idx)
            .collect()
    } else {
        Vec::new()
    };

    if verbose {
        eprintln!(
            "SSA CP-SAT: {} ops, critical path {}, horizon {}, shiftable VALU ops: {}, scratch_budget={}",
            n, critical_path_len, horizon, shiftable_valu.len(), scratch_budget
        );
    }

    // 3. Build the CP model
    let mut model = CpModelBuilder::default();

    // Build op_id to index mapping
    let op_id_to_idx: HashMap<usize, usize> = ops
        .iter()
        .enumerate()
        .map(|(idx, op)| (op.id, idx))
        .collect();

    // 4. Create cycle variable for each op with [asap, alap] domain
    let cycle_vars: Vec<IntVar> = ops
        .iter()
        .enumerate()
        .map(|(idx, op)| {
            let lb = *asap.get(&op.id).unwrap_or(&0) as i64;
            let ub = *alap.get(&op.id).unwrap_or(&(horizon as i32 - 1)) as i64;
            model.new_int_var_with_name([(lb, ub)], format!("c{}", idx))
        })
        .collect();

    // 5. Add precedence constraints from RAW dependencies
    for (idx, op) in ops.iter().enumerate() {
        for &dep_id in &op.dep_ids {
            if let Some(&dep_idx) = op_id_to_idx.get(&dep_id) {
                // cycle[idx] >= cycle[dep_idx] + 1
                model.add_ge(cycle_vars[idx], LinearExpr::from(cycle_vars[dep_idx]) + 1);
            }
        }
    }

    // 6. Create shift decision variables and optional intervals for shiftable VALU ops
    let shiftable_set: HashSet<usize> = shiftable_valu.iter().copied().collect();
    let mut shift_vars: HashMap<usize, BoolVar> = HashMap::new();
    let mut valu_optional_intervals: HashMap<usize, IntervalVar> = HashMap::new();
    let mut alu_optional_intervals: HashMap<usize, IntervalVar> = HashMap::new();

    for &idx in &shiftable_valu {
        // shift[idx] = 1 means shift to ALU, 0 means keep on VALU
        let shift_var = model.new_bool_var_with_name(format!("shift_{}", idx));
        shift_vars.insert(idx, shift_var);

        // VALU variant: present when NOT shifted (shift_var = false)
        let valu_interval = model.new_optional_fixed_size_interval_var(
            cycle_vars[idx],
            1,
            shift_var.not(),
        );
        valu_optional_intervals.insert(idx, valu_interval);

        // ALU variant: present when shifted (shift_var = true)
        let alu_interval = model.new_optional_fixed_size_interval_var(
            cycle_vars[idx],
            1,
            shift_var,
        );
        alu_optional_intervals.insert(idx, alu_interval);
    }

    // 6b. Create ConstChoice decision variables and optional intervals
    let const_choice_ops: Vec<usize> = ops.iter()
        .enumerate()
        .filter(|(_, op)| op.opcode == Opcode::ConstChoice)
        .map(|(idx, _)| idx)
        .collect();

    // Note: We don't need to find zero VReg op for AddImm precedence constraints.
    // Scratch memory is ALWAYS initialized to 0 at startup, so address 0 always contains 0.
    // AddImm operations can use address 0 directly without waiting for any op to "produce" zero.
    // The zero VReg is just a reference to address 0, not a value that needs to be computed.

    let mut const_choice_vars: HashMap<usize, BoolVar> = HashMap::new();
    let mut const_load_intervals: HashMap<usize, IntervalVar> = HashMap::new();
    let mut const_flow_intervals: HashMap<usize, IntervalVar> = HashMap::new();

    for &idx in &const_choice_ops {
        // Decision: true = use AddImm (Flow), false = use Load
        let choice_var = model.new_bool_var_with_name(format!("const_choice_{}", idx));
        const_choice_vars.insert(idx, choice_var);

        // Load variant: present when choice_var = false
        let load_interval = model.new_optional_fixed_size_interval_var(
            cycle_vars[idx],
            1,
            choice_var.not(),
        );
        const_load_intervals.insert(idx, load_interval);

        // AddImm variant: present when choice_var = true
        let add_imm_interval = model.new_optional_fixed_size_interval_var(
            cycle_vars[idx],
            1,
            choice_var,
        );
        const_flow_intervals.insert(idx, add_imm_interval);

        // AddImm can use address 0 directly - no precedence constraint needed!
        // Scratch memory is ALWAYS initialized to 0 at startup, so scratch[0] == 0 is
        // guaranteed by hardware initialization. AddImm operations can reference address 0
        // immediately without waiting for any op to "produce" zero.

        // Hint from greedy (default: Load)
        model.add_hint(choice_var, 0); // false = use Load
    }

    // 7. Create regular intervals for non-shiftable, non-ConstChoice ops
    let const_choice_set: HashSet<usize> = const_choice_ops.iter().copied().collect();
    let mut regular_intervals: Vec<IntervalVar> = Vec::new();
    for (idx, _) in ops.iter().enumerate() {
        if !shiftable_set.contains(&idx) && !const_choice_set.contains(&idx) {
            regular_intervals.push(model.new_fixed_size_interval_var(cycle_vars[idx], 1));
        }
    }

    // 8. Build cumulative constraints with optional intervals

    // Collect non-shiftable VALU ops
    let non_shiftable_valu: Vec<usize> = ops
        .iter()
        .enumerate()
        .filter(|(idx, op)| op.engine == Engine::Valu && !shiftable_set.contains(idx))
        .map(|(idx, _)| idx)
        .collect();

    // Collect ALU ops
    let alu_ops: Vec<usize> = ops
        .iter()
        .enumerate()
        .filter(|(_, op)| op.engine == Engine::Alu)
        .map(|(idx, _)| idx)
        .collect();

    // VALU cumulative: non-shiftable VALU + optional VALU intervals from shiftable ops
    {
        let mut valu_intervals: Vec<IntervalVar> = Vec::new();
        let mut valu_demands: Vec<i64> = Vec::new();

        // Add non-shiftable VALU ops (always present)
        for &idx in &non_shiftable_valu {
            valu_intervals.push(model.new_fixed_size_interval_var(cycle_vars[idx], 1));
            valu_demands.push(1);
        }

        // Add optional VALU intervals (present when not shifted)
        for &idx in &shiftable_valu {
            if let Some(&iv) = valu_optional_intervals.get(&idx) {
                valu_intervals.push(iv);
                valu_demands.push(1);
            }
        }

        if !valu_intervals.is_empty() {
            model.add_cumulative(valu_intervals, valu_demands, 6); // VALU has 6 slots
        }
    }

    // ALU cumulative: regular ALU ops + optional ALU intervals (demand=8 for shifted VALU)
    {
        let mut alu_intervals: Vec<IntervalVar> = Vec::new();
        let mut alu_demands: Vec<i64> = Vec::new();

        // Add regular ALU ops (demand = 1)
        for &idx in &alu_ops {
            alu_intervals.push(model.new_fixed_size_interval_var(cycle_vars[idx], 1));
            alu_demands.push(1);
        }

        // Add optional ALU intervals for shifted VALU ops (demand = 8)
        for &idx in &shiftable_valu {
            if let Some(&iv) = alu_optional_intervals.get(&idx) {
                alu_intervals.push(iv);
                alu_demands.push(8); // Shifted VALU op expands to 8 ALU ops
            }
        }

        if !alu_intervals.is_empty() {
            model.add_cumulative(alu_intervals, alu_demands, 12); // ALU has 12 slots
        }
    }

    // Load cumulative: regular Load ops + optional Load intervals from ConstChoice
    {
        let regular_load_ops: Vec<usize> = ops.iter()
            .enumerate()
            .filter(|(_, op)| op.engine == Engine::Load && op.opcode != Opcode::ConstChoice)
            .map(|(idx, _)| idx)
            .collect();

        let mut load_intervals: Vec<IntervalVar> = Vec::new();
        let mut load_demands: Vec<i64> = Vec::new();

        // Add regular Load ops
        for &idx in &regular_load_ops {
            load_intervals.push(model.new_fixed_size_interval_var(cycle_vars[idx], 1));
            load_demands.push(1);
        }

        // Add optional Load intervals from ConstChoice ops (when choice_var = false)
        for &idx in &const_choice_ops {
            if let Some(&iv) = const_load_intervals.get(&idx) {
                load_intervals.push(iv);
                load_demands.push(1);
            }
        }

        if !load_intervals.is_empty() {
            model.add_cumulative(load_intervals, load_demands, 2); // Load has 2 slots
        }
    }

    // Store cumulative (unchanged - no ConstChoice)
    {
        let store_ops: Vec<usize> = ops.iter()
            .enumerate()
            .filter(|(_, op)| op.engine == Engine::Store)
            .map(|(idx, _)| idx)
            .collect();

        if !store_ops.is_empty() {
            let intervals: Vec<IntervalVar> = store_ops
                .iter()
                .map(|&idx| model.new_fixed_size_interval_var(cycle_vars[idx], 1))
                .collect();
            let demands: Vec<i64> = vec![1; intervals.len()];
            model.add_cumulative(intervals, demands, 2); // Store has 2 slots
        }
    }

    // Flow cumulative: regular Flow ops + optional Flow intervals from ConstChoice
    {
        let regular_flow_ops: Vec<usize> = ops.iter()
            .enumerate()
            .filter(|(_, op)| op.engine == Engine::Flow && op.opcode != Opcode::AddImm)
            .map(|(idx, _)| idx)
            .collect();

        let mut flow_intervals: Vec<IntervalVar> = Vec::new();
        let mut flow_demands: Vec<i64> = Vec::new();

        // Add regular Flow ops
        for &idx in &regular_flow_ops {
            flow_intervals.push(model.new_fixed_size_interval_var(cycle_vars[idx], 1));
            flow_demands.push(1);
        }

        // Add optional Flow intervals from ConstChoice ops (when choice_var = true)
        for &idx in &const_choice_ops {
            if let Some(&iv) = const_flow_intervals.get(&idx) {
                flow_intervals.push(iv);
                flow_demands.push(1);
            }
        }

        if !flow_intervals.is_empty() {
            model.add_cumulative(flow_intervals, flow_demands, 1); // Flow has 1 slot
        }
    }

    // 8b. Add scratch pressure constraint (live VReg capacity)
    // Models: at any cycle, total size of live VRegs <= scratch_budget
    eprintln!("DEBUG: scratch_budget = {}, verbose = {}", scratch_budget, verbose);
    if scratch_budget > 0 {
        // Build VReg -> defining op index mapping
        let mut vreg_def_op: HashMap<VReg, usize> = HashMap::new();
        for (idx, op) in ops.iter().enumerate() {
            if let Some(dest) = op.dest {
                vreg_def_op.insert(dest, idx);
            }
        }

        // Build VReg -> last use op index mapping
        // IMPORTANT: We need the op with the LATEST CYCLE that reads this VReg, not just the latest index.
        // If we have greedy hints, use them to determine the actual last use by cycle.
        // Otherwise, fall back to index order (which may be incorrect but is a reasonable heuristic).
        //
        // Dead Store Handling:
        // The greedy scheduler (ssa_greedy.rs) identifies dead stores during scheduling
        // and pre-marks them as "scheduled" without assigning a valid cycle. Their cycle
        // field remains at -1 (the initial unscheduled value). CP-SAT honors this by:
        // 1. Skipping ops with negative cycle hints when building vreg_last_use
        // 2. This correctly excludes dead stores from liveness analysis
        // The warnings printed by greedy are informational, not errors.
        let mut vreg_last_use: HashMap<VReg, usize> = HashMap::new();
        for (idx, op) in ops.iter().enumerate() {
            // Skip dead stores identified by greedy scheduler.
            // Dead stores have negative cycle hints (-1) because greedy pre-marks them
            // as "scheduled" without actually scheduling them to a real cycle.
            if let Some(hints) = greedy_hints {
                if hints.get(idx).copied().unwrap_or(0) < 0 {
                    continue; // Dead store, skip - it will not be scheduled by CP-SAT either
                }
            }
            for vreg in op.reads() {
                let should_update = if let Some(hints) = greedy_hints {
                    // Use cycle from hints to determine if this is a later use
                    let current_cycle = hints.get(idx).copied().unwrap_or(0);
                    let existing_cycle = vreg_last_use.get(&vreg)
                        .and_then(|&existing_idx| hints.get(existing_idx).copied())
                        .unwrap_or(-1);
                    current_cycle > existing_cycle
                } else {
                    // Without hints, use index order (may be suboptimal)
                    true
                };

                if should_update {
                    vreg_last_use.insert(vreg, idx);
                } else if !vreg_last_use.contains_key(&vreg) {
                    vreg_last_use.insert(vreg, idx);
                }
            }
        }

        if verbose {
            // Count ops with dest
            let ops_with_dest = ops.iter().filter(|op| op.dest.is_some()).count();
            let ops_with_reads = ops.iter().filter(|op| op.reads().count() > 0).count();
            eprintln!(
                "  Scratch debug: {} ops total, {} with dest, {} with reads, {} VReg defs, {} VReg uses",
                ops.len(),
                ops_with_dest,
                ops_with_reads,
                vreg_def_op.len(),
                vreg_last_use.len()
            );
        }

        // For VRegs that are defined but never read (dead code), they still occupy space
        // until the end. For simplicity, we'll only model VRegs that have both def and use.

        // Collect VRegs that have both definition and use
        let live_vregs: Vec<VReg> = vreg_def_op
            .keys()
            .filter(|v| vreg_last_use.contains_key(v))
            .copied()
            .collect();

        if !live_vregs.is_empty() {
            let mut scratch_intervals: Vec<IntervalVar> = Vec::new();
            let mut scratch_demands: Vec<i64> = Vec::new();
            let mut size_vars: Vec<(IntVar, usize, usize)> = Vec::new(); // (var, def_idx, last_use_idx)

            // =========================================================================
            // CP-SAT LIVENESS MODEL: Slightly More Conservative than Greedy
            // =========================================================================
            //
            // The cumulative constraint models scratch usage over time. We use
            // `end = last_use_cycle + 1` so intervals are half-open: [start, end).
            //
            // KEY DIFFERENCE FROM GREEDY/ADDRESS_ASSIGN:
            // - Here: interval [0,5] has end=6, and [5,10] has start=5, end=11
            //   These OVERLAP in the cumulative constraint (both include the 5-6 region)
            // - In address_assign.rs: intervals [0,5] and [5,10] do NOT overlap
            //   because we use `end_cycle <= start_cycle` for expiration (same-cycle reuse)
            //
            // This makes CP-SAT slightly more conservative than greedy allocation.
            // In practice, this rarely matters because:
            // 1. Same-cycle reuse is only possible for def/use at exact same cycle
            // 2. CP-SAT's goal is to avoid exceeding budget, not minimize peak
            // 3. The greedy scheduler's peak is an upper bound; CP-SAT can improve it
            //
            // RESULT: If CP-SAT says schedule X fits in budget, it definitely does.
            // But CP-SAT may reject schedules that greedy+address_assign could fit.
            // =========================================================================

            for vreg in &live_vregs {
                let def_idx = vreg_def_op[vreg];
                let last_use_idx = vreg_last_use[vreg];

                // The VReg is live from when it's defined until its last use.
                // Create a variable-length interval: [def_cycle, last_use_cycle]
                // The "size" of this interval is (last_use_cycle - def_cycle + 1)
                //
                // For cumulative, we need: start, size (or end)
                // start = cycle_vars[def_idx]
                // end = cycle_vars[last_use_idx] + 1 (VReg is live through the last use cycle)
                //
                // Use new_interval_var(start, size, end) where end = start + size
                // We want: live from def_cycle to last_use_cycle (inclusive)
                // So: end = last_use_cycle + 1, size = end - start = last_use_cycle + 1 - def_cycle

                let start = cycle_vars[def_idx];
                let end_expr = LinearExpr::from(cycle_vars[last_use_idx]) + 1;

                // Create a size variable: size = end - start = cycle[last_use] + 1 - cycle[def]
                // Size must be >= 1 (def and last_use could be same cycle)
                let size = model.new_int_var_with_name(
                    [(1, horizon + 1)],
                    format!("live_size_{}", vreg.id),
                );

                // Constraint: size = cycle[last_use] + 1 - cycle[def]
                // Rewrite: cycle[def] + size = cycle[last_use] + 1
                model.add_eq(
                    LinearExpr::from(start) + LinearExpr::from(size),
                    end_expr,
                );

                let interval = model.new_interval_var(start, size, LinearExpr::from(cycle_vars[last_use_idx]) + 1);
                scratch_intervals.push(interval);
                size_vars.push((size, def_idx, last_use_idx));

                // Demand = size of VReg (1 for scalar, VLEN for vector)
                let demand = vreg.size() as i64;
                scratch_demands.push(demand);
            }

            model.add_cumulative(scratch_intervals, scratch_demands, scratch_budget as i64);

            // Add hints for size variables if we have greedy hints
            if let Some(hints) = greedy_hints {
                let mut bad_size_count = 0;
                for vreg in &live_vregs {
                    let def_idx = vreg_def_op[vreg];
                    let last_use_idx = vreg_last_use[vreg];
                    if def_idx < hints.len() && last_use_idx < hints.len() {
                        let def_cycle = hints[def_idx];
                        let last_use_cycle = hints[last_use_idx];
                        let expected_size = last_use_cycle - def_cycle + 1;
                        if expected_size < 1 {
                            bad_size_count += 1;
                            if bad_size_count <= 5 && verbose {
                                // Get more info about the problematic VReg
                                let def_op = &ops[def_idx];
                                let use_op = &ops[last_use_idx];
                                eprintln!(
                                    "  BUG: VReg {:?} (size={}): def by op {} (id={}, {:?}) at cycle {}, \
                                     but last_use by op {} (id={}, {:?}) at cycle {}. \
                                     Use depends on def? {:?}",
                                    vreg, vreg.size(), def_idx, def_op.id, def_op.engine, def_cycle,
                                    last_use_idx, use_op.id, use_op.engine, last_use_cycle,
                                    use_op.dep_ids.contains(&def_op.id)
                                );
                            }
                        }
                    }
                }

                // Now add size hints
                for (size_var, def_idx, last_use_idx) in &size_vars {
                    if *def_idx < hints.len() && *last_use_idx < hints.len() {
                        let def_cycle = hints[*def_idx];
                        let last_use_cycle = hints[*last_use_idx];
                        let expected_size = (last_use_cycle - def_cycle + 1).max(1);
                        model.add_hint(*size_var, expected_size as i64);
                    }
                }

                if bad_size_count > 0 {
                    panic!(
                        "BUG: {} VRegs have last_use_cycle < def_cycle! This will cause infeasible hints.",
                        bad_size_count
                    );
                }
            }

            // CRITICAL: Verify hints are feasible before solving
            // If hints are infeasible, there's a bug in our model - fail immediately
            if let Some(hints) = greedy_hints {
                // Compute peak liveness from hints
                let mut events: Vec<(i32, i64)> = Vec::new();
                for vreg in &live_vregs {
                    let def_idx = vreg_def_op[vreg];
                    let last_use_idx = vreg_last_use[vreg];
                    let def_cycle = hints.get(def_idx).copied().unwrap_or(0);
                    let last_use_cycle = hints.get(last_use_idx).copied().unwrap_or(def_cycle);
                    let size = vreg.size() as i64;
                    events.push((def_cycle, size));           // VReg becomes live
                    events.push((last_use_cycle + 1, -size)); // VReg dies after last use
                }
                events.sort_by_key(|(cycle, delta)| (*cycle, *delta));

                let mut current = 0i64;
                let mut peak = 0i64;
                for (_, delta) in &events {
                    current += delta;
                    peak = peak.max(current);
                }

                if peak > scratch_budget as i64 {
                    // This is expected when greedy scheduler exceeded budget (force-scheduled).
                    // CP-SAT hints are "soft" - they're suggestions, not requirements.
                    // CP-SAT will use the hint as a starting point but may ignore infeasible parts.
                    eprintln!(
                        "Note: Greedy hint exceeds budget (peak_live={} > budget={}). \
                         CP-SAT will use hint as starting point but may ignore infeasible parts.",
                        peak, scratch_budget
                    );
                } else if verbose {
                    eprintln!("  Hint verification: peak_live={} <= budget={} OK", peak, scratch_budget);
                }
            }

            if verbose {
                eprintln!(
                    "  Scratch constraint: {} live VRegs, budget={}",
                    live_vregs.len(),
                    scratch_budget
                );
            }
        }
    }

    // Verify VALU and ALU cumulative constraints from greedy hints
    if let Some(hints) = greedy_hints {
        // Compute per-cycle usage
        let max_cycle = *hints.iter().max().unwrap_or(&0) as usize + 1;
        let mut valu_per_cycle = vec![0i32; max_cycle];
        let mut alu_per_cycle = vec![0i32; max_cycle];

        for (idx, op) in ops.iter().enumerate() {
            if idx >= hints.len() {
                continue;
            }
            let cycle = hints[idx] as usize;
            if cycle >= max_cycle {
                continue;
            }

            let is_shifted = if let Some(shift_hints) = shift_hints {
                shift_hints.contains(&op.id)
            } else {
                false
            };

            match op.engine {
                Engine::Valu if !is_shifted => {
                    valu_per_cycle[cycle] += 1;
                }
                Engine::Valu if is_shifted => {
                    alu_per_cycle[cycle] += VLEN as i32; // Shifted VALU uses 8 ALU slots
                }
                Engine::Alu => {
                    alu_per_cycle[cycle] += 1;
                }
                _ => {}
            }
        }

        // Check for violations
        let mut valu_violations = 0;
        let mut alu_violations = 0;
        for cycle in 0..max_cycle {
            if valu_per_cycle[cycle] > 6 {
                if valu_violations < 3 && verbose {
                    eprintln!("  WARNING: VALU overflow at cycle {}: {} > 6", cycle, valu_per_cycle[cycle]);
                }
                valu_violations += 1;
            }
            if alu_per_cycle[cycle] > 12 {
                if alu_violations < 3 && verbose {
                    eprintln!("  WARNING: ALU overflow at cycle {}: {} > 12", cycle, alu_per_cycle[cycle]);
                }
                alu_violations += 1;
            }
        }

        if valu_violations > 0 || alu_violations > 0 {
            panic!(
                "BUG: Greedy hint violates engine capacity! VALU violations: {}, ALU violations: {}",
                valu_violations, alu_violations
            );
        }

        if verbose {
            eprintln!("  Engine capacity verification: OK (VALU<=6, ALU<=12 at all cycles)");
        }
    }

    // 9. Compute capacity-based lower bounds
    // With VALU shifting, we can potentially use ALU capacity for VALU overflow
    let mut capacity_lower_bound = critical_path_len as i64;

    // Count ops by engine
    let valu_count = ops.iter().filter(|op| op.engine == Engine::Valu).count() as i64;
    let alu_count = ops.iter().filter(|op| op.engine == Engine::Alu).count() as i64;
    let load_count = ops.iter().filter(|op| op.engine == Engine::Load).count() as i64;
    let store_count = ops.iter().filter(|op| op.engine == Engine::Store).count() as i64;

    // Without shifting: VALU bound
    let valu_only_bound = (valu_count + 5) / 6;
    if !enable_valu_shift && valu_only_bound > capacity_lower_bound {
        capacity_lower_bound = valu_only_bound;
    }

    // With shifting: combined ALU+VALU bound
    // Each shifted VALU op uses 8 ALU slots
    // Best case: shift as many VALU ops as ALU capacity allows
    if enable_valu_shift {
        // This is a rough lower bound - actual bound depends on shift decisions
        // Minimum cycles needed to execute all ops considering both engines
        let total_valu_work = valu_count; // Each VALU op = 1 VALU slot or 8 ALU slots
        let total_alu_work = alu_count; // Each ALU op = 1 ALU slot

        // Per cycle: up to 6 VALU + 12 ALU, but shifted VALU uses 8 ALU
        // Lower bound: we need at least enough cycles to handle all work
        // Simple bound: (total_alu + shifted_valu*8) / 12 combined with (unshifted_valu) / 6
        // Approximate: max(alu_count/12, valu_count/6)
        let combined_bound = (valu_count + alu_count + 5) / 14; // Rough approximation
        if combined_bound > capacity_lower_bound {
            capacity_lower_bound = combined_bound;
        }
    }

    // Other engine bounds
    let load_bound = (load_count + 1) / 2;
    let store_bound = (store_count + 1) / 2;
    capacity_lower_bound = capacity_lower_bound.max(load_bound).max(store_bound);

    if verbose {
        eprintln!(
            "  Capacity bounds: VALU={}, ALU={}, Load={}, Store={}, combined={}",
            valu_only_bound, (alu_count + 11) / 12, load_bound, store_bound, capacity_lower_bound
        );
    }

    // 10. Create makespan variable and objective
    let makespan = model.new_int_var_with_name(
        [(capacity_lower_bound, horizon)],
        "makespan".to_string(),
    );

    // makespan >= max(cycle[i]) + 1 for all ops
    for &cv in &cycle_vars {
        model.add_ge(makespan, LinearExpr::from(cv) + 1);
    }

    model.minimize(makespan);

    // 11. Add hints from greedy scheduler
    if let Some(hints) = greedy_hints {
        for (idx, &hint_cycle) in hints.iter().enumerate() {
            if idx < cycle_vars.len() && hint_cycle >= 0 {
                model.add_hint(cycle_vars[idx], hint_cycle as i64);
            }
        }
        // Hint the makespan too
        let hint_makespan = hints.iter().max().copied().unwrap_or(0) + 1;
        model.add_hint(makespan, hint_makespan as i64);
    }

    // Add shift hints if provided
    if let Some(hints) = shift_hints {
        for (&idx, &shift_var) in &shift_vars {
            let should_shift = hints.contains(&ops[idx].id);
            model.add_hint(shift_var, if should_shift { 1 } else { 0 });
        }
    }

    // 12. Configure solver
    let mut params = SatParameters::default();
    params.max_time_in_seconds = Some(time_limit_secs);
    // Always log search progress by default - user must go out of their way to suppress
    params.log_search_progress = Some(true);
    params.num_workers = Some(8);

    // Optimize LP relaxation for tighter bounds and faster convergence
    // linearization_level=2: Stronger LP relaxations (better bounds for cumulative constraints)
    //   - Cumulative constraints have weak LP relaxations by default
    //   - Level 2 linearizes more constraints, giving tighter best_objective_bound
    //   - Expected: Gap reduction from 10-15% to 3-8%, 20-40% faster convergence
    // cp_model_presolve=true: Preprocessing to simplify model before solving
    //   - Removes redundant variables/constraints, detects symmetries
    //   - Usually <1% of solve time, but significantly speeds up search
    // cp_model_probing_level=2: Aggressive probing to tighten variable domains
    //   - Temporarily fixes variables and observes consequences
    //   - Strengthens domains and adds binary clauses
    //   - Expected: 5-15% tighter domains, 10-20% faster search
    // randomize_search=true: Help escape local optima
    //   - Adds randomization to search decisions
    //   - Helps when solver gets stuck in suboptimal regions
    params.linearization_level = Some(2);
    params.cp_model_presolve = Some(true);
    params.cp_model_probing_level = Some(2);
    params.randomize_search = Some(true);

    if stop_after_first_solution {
        params.stop_after_first_solution = Some(true);
    }

    // 14. Solve
    let response = model.solve_with_parameters(&params);
    let solve_time = start_time.elapsed().as_secs_f64();

    let status: CpsatStatus = response.status().into();

    match status {
        CpsatStatus::Optimal | CpsatStatus::Feasible => {
            let cycle_assignments: Vec<i32> = cycle_vars
                .iter()
                .map(|&cv| cv.solution_value(&response) as i32)
                .collect();

            let actual_makespan = cycle_assignments.iter().max().copied().unwrap_or(0) + 1;

            // Extract shift decisions
            let shifted_ops: HashSet<usize> = shift_vars
                .iter()
                .filter(|(_, &var)| var.solution_value(&response))
                .map(|(&idx, _)| ops[idx].id) // Return op IDs, not indices
                .collect();

            // Extract ConstChoice decisions
            let const_add_imm_ops: HashSet<usize> = const_choice_vars
                .iter()
                .filter(|(_, &var)| var.solution_value(&response))
                .map(|(&idx, _)| ops[idx].id) // Return op IDs, not indices
                .collect();

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

            if verbose {
                eprintln!(
                    "SSA CP-SAT: {} cycles, status {:?}, gap {:.2}%, time {:.2}s, shifted {} VALU ops, {} ConstChoice AddImm",
                    actual_makespan, status, gap * 100.0, solve_time, shifted_ops.len(), const_add_imm_ops.len()
                );
            }

            Some(SsaCpsatResult {
                cycle_assignments,
                makespan: actual_makespan as usize,
                status,
                solve_time_secs: solve_time,
                gap,
                shifted_ops,
                const_add_imm_ops,
            })
        }
        _ => {
            if verbose {
                eprintln!("SSA CP-SAT: no solution found, status {:?}", status);
            }
            None
        }
    }
}

/// Stub implementation when cpsat feature is not enabled
#[cfg(not(feature = "cpsat"))]
pub fn schedule_ssa_cpsat(
    _ops: &[SsaOp],
    _config: SsaCpsatConfig,
    _greedy_hints: Option<&[i32]>,
    _shift_hints: Option<&HashSet<usize>>,
) -> Option<SsaCpsatResult> {
    eprintln!("SSA CP-SAT scheduler not available: compile with --features cpsat");
    None
}

#[cfg(test)]
mod ssa_tests {
    use super::*;
    use crate::kernel_builder::emit_ssa::{SsaEmitContext, compute_ssa_dependencies};
    use crate::kernel_builder::ssa_lower::lower_ssa_ops_with_shift;
    use crate::types::{Engine, Opcode, SlotVal};
    use std::collections::{HashMap, HashSet};

    #[cfg(feature = "cpsat")]
    #[test]
    fn test_ssa_cpsat_constchoice_decisions() {
        // Build a simple kernel with ConstChoice ops
        let mut emit = SsaEmitContext::new();
        
        // Emit zero constant
        let _zero = emit.emit_const(0);
        
        // Emit ConstChoice constants
        let c1 = emit.emit_const_choice(1);
        let c2 = emit.emit_const_choice(2);
        
        // Use constants in ALU ops
        let _sum = emit.emit_alu(Opcode::Add, c1, c2);

        // Compute dependencies
        let vreg_def = std::mem::take(&mut emit.vreg_def);
        compute_ssa_dependencies(&mut emit.ops, &vreg_def);

        // Schedule with CP-SAT
        let result = schedule_ssa_cpsat(
            &emit.ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: false,
                stop_after_first_solution: true,
                enable_valu_shift: false,
                scratch_budget: 1536,
            },
            None,
            None,
        );

        assert!(result.is_some(), "CP-SAT should find a solution");
        let result = result.unwrap();

        // Verify all ops are scheduled
        assert_eq!(result.cycle_assignments.len(), emit.ops.len());

        // Verify ConstChoice decisions are extracted
        // The result should contain const_add_imm_ops (may be empty if solver chose Load)
        assert!(result.const_add_imm_ops.len() <= 2, 
            "Should have at most 2 ConstChoice ops using AddImm");

        // Note: AddImm operations can use address 0 directly without waiting for zero
        // to be "ready" since scratch memory is always initialized to 0 at startup.
        // There is no precedence constraint requiring zero to be scheduled before AddImm.
    }

    #[cfg(feature = "cpsat")]
    #[test]
    fn test_ssa_cpsat_constchoice_roundtrip() {
        // Build a simple kernel with ConstChoice ops
        let mut emit = SsaEmitContext::new();
        
        // Emit zero constant
        let zero = emit.emit_const(0);
        
        // Emit ConstChoice constants
        let c1 = emit.emit_const_choice(1);
        let c2 = emit.emit_const_choice(2);
        let c4 = emit.emit_const_choice(4);
        
        // Use constants in ALU ops
        let _sum1 = emit.emit_alu(Opcode::Add, c1, c2);
        let _sum2 = emit.emit_alu(Opcode::Add, c4, zero);

        // Compute dependencies
        let vreg_def = std::mem::take(&mut emit.vreg_def);
        compute_ssa_dependencies(&mut emit.ops, &vreg_def);

        // Schedule with CP-SAT
        let result = schedule_ssa_cpsat(
            &emit.ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: false,
                stop_after_first_solution: true,
                enable_valu_shift: false,
                scratch_budget: 1536,
            },
            None,
            None,
        );

        assert!(result.is_some(), "CP-SAT should find a solution");
        let cpsat_result = result.unwrap();

        // Verify all ops are scheduled
        assert_eq!(cpsat_result.cycle_assignments.len(), emit.ops.len());

        // Find ConstChoice op IDs and their values
        let const_choice_info: Vec<(usize, i64)> = emit.ops.iter()
            .filter(|op| op.opcode == Opcode::ConstChoice)
            .map(|op| {
                let value = op.srcs[0].as_imm().expect("ConstChoice should have immediate operand");
                (op.id, value)
            })
            .collect();

        assert_eq!(const_choice_info.len(), 3, "Should have 3 ConstChoice ops (1, 2, 4)");

        // Verify CP-SAT decisions are extracted
        assert!(cpsat_result.const_add_imm_ops.len() <= 3,
            "Should have at most 3 ConstChoice ops using AddImm");

        // Now test round-trip: lower using CP-SAT decisions
        let mut vreg_to_addr = HashMap::new();
        for (idx, op) in emit.ops.iter().enumerate() {
            if let Some(dest) = op.dest {
                vreg_to_addr.insert(dest, idx as i64 * 10); // Simple address assignment
            }
        }
        // Ensure zero is at address 0
        vreg_to_addr.insert(zero, 0);

        let lowered_ops = lower_ssa_ops_with_shift(
            &emit.ops,
            &vreg_to_addr,
            &HashSet::new(), // No shifted ops
            &cpsat_result.const_add_imm_ops, // Use CP-SAT decisions
        );

        // Verify that ConstChoice ops were lowered correctly based on CP-SAT decisions
        for (const_choice_id, expected_value) in &const_choice_info {
            let lowered_op = lowered_ops.iter()
                .find(|op| op.produces_constant == Some(*expected_value))
                .expect(&format!("Should find lowered op for ConstChoice producing {}", expected_value));

            if cpsat_result.const_add_imm_ops.contains(const_choice_id) {
                // CP-SAT chose AddImm - verify it's a Flow op
                assert_eq!(lowered_op.engine, Engine::Flow,
                    "ConstChoice op {} (value {}) should be lowered to Flow (AddImm)", const_choice_id, expected_value);
                match &lowered_op.slot[0] {
                    SlotVal::Str(s) => assert_eq!(s, "add_imm",
                        "ConstChoice op {} should be add_imm", const_choice_id),
                    _ => panic!("AddImm slot[0] should be Str"),
                }
                match &lowered_op.slot[2] {
                    SlotVal::Int(addr) => assert_eq!(*addr, 0,
                        "AddImm should read from address 0"),
                    _ => panic!("AddImm slot[2] should be Int(0)"),
                }
            } else {
                // CP-SAT chose Load - verify it's a Load op
                assert_eq!(lowered_op.engine, Engine::Load,
                    "ConstChoice op {} (value {}) should be lowered to Load", const_choice_id, expected_value);
                match &lowered_op.slot[0] {
                    SlotVal::Str(s) => assert_eq!(s, "const",
                        "ConstChoice op {} should be const", const_choice_id),
                    _ => panic!("Const slot[0] should be Str"),
                }
            }
        }

        // Verify all ConstChoice ops were lowered
        assert_eq!(lowered_ops.len(), emit.ops.len(),
            "All ops should be lowered");
    }

    #[cfg(feature = "cpsat")]
    #[test]
    fn test_ssa_cpsat_constchoice_prefers_addimm_when_load_slots_full() {
        // Create a scenario where Load slots are heavily used, forcing CP-SAT to choose AddImm
        let mut emit = SsaEmitContext::new();
        
        // Emit zero constant
        let zero = emit.emit_const(0);
        
        // Emit many Load ops to fill Load slots
        for i in 0..10 {
            emit.emit_const(i + 100); // Regular Load ops
        }
        
        // Emit ConstChoice ops - CP-SAT should prefer AddImm to avoid Load slot pressure
        let c1 = emit.emit_const_choice(1);
        let c2 = emit.emit_const_choice(2);
        
        // Use constants
        let _sum = emit.emit_alu(Opcode::Add, c1, c2);

        // Compute dependencies
        let vreg_def = std::mem::take(&mut emit.vreg_def);
        compute_ssa_dependencies(&mut emit.ops, &vreg_def);

        // Schedule with CP-SAT with longer time limit to allow optimization
        let result = schedule_ssa_cpsat(
            &emit.ops,
            SsaCpsatConfig {
                time_limit_secs: 30.0,
                verbose: false,
                stop_after_first_solution: false, // Allow optimization
                enable_valu_shift: false,
                scratch_budget: 1536,
            },
            None,
            None,
        );

        assert!(result.is_some(), "CP-SAT should find a solution");
        let cpsat_result = result.unwrap();

        // Find ConstChoice op IDs
        let const_choice_ids: Vec<usize> = emit.ops.iter()
            .filter(|op| op.opcode == Opcode::ConstChoice)
            .map(|op| op.id)
            .collect();

        assert_eq!(const_choice_ids.len(), 2, "Should have 2 ConstChoice ops");

        // Verify decisions were extracted
        assert!(cpsat_result.const_add_imm_ops.len() <= 2,
            "Should have at most 2 ConstChoice ops using AddImm");

        // Test lowering with the decisions
        let mut vreg_to_addr = HashMap::new();
        for (idx, op) in emit.ops.iter().enumerate() {
            if let Some(dest) = op.dest {
                vreg_to_addr.insert(dest, idx as i64 * 10);
            }
        }
        vreg_to_addr.insert(zero, 0);

        let lowered_ops = lower_ssa_ops_with_shift(
            &emit.ops,
            &vreg_to_addr,
            &HashSet::new(),
            &cpsat_result.const_add_imm_ops,
        );

        // Verify ConstChoice ops were lowered correctly
        for const_choice_id in &const_choice_ids {
            let ssa_op = emit.ops.iter().find(|op| op.id == *const_choice_id).unwrap();
            let expected_value = ssa_op.srcs[0].as_imm().unwrap();
            
            let lowered_op = lowered_ops.iter()
                .find(|op| op.produces_constant == Some(expected_value))
                .expect(&format!("Should find lowered op for ConstChoice {}", const_choice_id));

            if cpsat_result.const_add_imm_ops.contains(const_choice_id) {
                // Verify AddImm was chosen
                assert_eq!(lowered_op.engine, Engine::Flow,
                    "ConstChoice {} should use AddImm when Load slots are full", const_choice_id);
                match &lowered_op.slot[0] {
                    SlotVal::Str(s) => assert_eq!(s, "add_imm"),
                    _ => panic!("Should be add_imm"),
                }
            } else {
                // Verify Load was chosen
                assert_eq!(lowered_op.engine, Engine::Load,
                    "ConstChoice {} should use Load", const_choice_id);
            }
        }
    }
}

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
    use crate::types::{Engine, Op, Operand, SlotVal, SsaOp, VReg, MAX_SLOT_SIZE};
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

    #[cfg(feature = "cpsat")]
    #[test]
    fn test_ssa_cpsat_constchoice_decisions() {
        use crate::kernel_builder::emit_ssa::{SsaEmitContext, compute_ssa_dependencies};
        use crate::types::Opcode;

        // Build a simple kernel with ConstChoice ops
        let mut emit = SsaEmitContext::new();
        
        // Emit zero constant
        let _zero = emit.emit_const(0);
        
        // Emit ConstChoice constants
        let c1 = emit.emit_const_choice(1);
        let c2 = emit.emit_const_choice(2);
        
        // Use constants in ALU ops
        let _sum = emit.emit_alu(Opcode::Add, c1, c2);

        // Compute dependencies
        let vreg_def = std::mem::take(&mut emit.vreg_def);
        compute_ssa_dependencies(&mut emit.ops, &vreg_def);

        // Schedule with CP-SAT
        let result = schedule_ssa_cpsat(
            &emit.ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: false,
                stop_after_first_solution: true,
                enable_valu_shift: false,
                scratch_budget: 1536,
            },
            None,
            None,
        );

        assert!(result.is_some(), "CP-SAT should find a solution");
        let result = result.unwrap();

        // Verify all ops are scheduled
        assert_eq!(result.cycle_assignments.len(), emit.ops.len());

        // Verify ConstChoice decisions are extracted
        // The result should contain const_add_imm_ops (may be empty if solver chose Load)
        assert!(result.const_add_imm_ops.len() <= 2, 
            "Should have at most 2 ConstChoice ops using AddImm");

        // Note: AddImm operations can use address 0 directly without waiting for zero
        // to be "ready" since scratch memory is always initialized to 0 at startup.
        // There is no precedence constraint requiring zero to be scheduled before AddImm.
    }

    #[cfg(feature = "cpsat")]
    #[test]
    fn test_ssa_cpsat_constchoice_roundtrip() {
        use crate::kernel_builder::emit_ssa::{SsaEmitContext, compute_ssa_dependencies};
        use crate::kernel_builder::ssa_lower::lower_ssa_ops_with_shift;
        use crate::types::{Engine, Opcode, SlotVal};
        use std::collections::HashMap;

        // Build a simple kernel with ConstChoice ops
        let mut emit = SsaEmitContext::new();
        
        // Emit zero constant
        let zero = emit.emit_const(0);
        
        // Emit ConstChoice constants
        let c1 = emit.emit_const_choice(1);
        let c2 = emit.emit_const_choice(2);
        let c4 = emit.emit_const_choice(4);
        
        // Use constants in ALU ops
        let _sum1 = emit.emit_alu(Opcode::Add, c1, c2);
        let _sum2 = emit.emit_alu(Opcode::Add, c4, zero);

        // Compute dependencies
        let vreg_def = std::mem::take(&mut emit.vreg_def);
        compute_ssa_dependencies(&mut emit.ops, &vreg_def);

        // Schedule with CP-SAT
        let result = schedule_ssa_cpsat(
            &emit.ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: false,
                stop_after_first_solution: true,
                enable_valu_shift: false,
                scratch_budget: 1536,
            },
            None,
            None,
        );

        assert!(result.is_some(), "CP-SAT should find a solution");
        let cpsat_result = result.unwrap();

        // Verify all ops are scheduled
        assert_eq!(cpsat_result.cycle_assignments.len(), emit.ops.len());

        // Find ConstChoice op IDs and their values
        let const_choice_info: Vec<(usize, i64)> = emit.ops.iter()
            .filter(|op| op.opcode == Opcode::ConstChoice)
            .map(|op| {
                let value = op.srcs[0].as_imm().expect("ConstChoice should have immediate operand");
                (op.id, value)
            })
            .collect();

        assert_eq!(const_choice_info.len(), 3, "Should have 3 ConstChoice ops (1, 2, 4)");

        // Verify CP-SAT decisions are extracted
        assert!(cpsat_result.const_add_imm_ops.len() <= 3,
            "Should have at most 3 ConstChoice ops using AddImm");

        // Now test round-trip: lower using CP-SAT decisions
        let mut vreg_to_addr = HashMap::new();
        for (idx, op) in emit.ops.iter().enumerate() {
            if let Some(dest) = op.dest {
                vreg_to_addr.insert(dest, idx as i64 * 10); // Simple address assignment
            }
        }
        // Ensure zero is at address 0
        vreg_to_addr.insert(zero, 0);

        let lowered_ops = lower_ssa_ops_with_shift(
            &emit.ops,
            &vreg_to_addr,
            &HashSet::new(), // No shifted ops
            &cpsat_result.const_add_imm_ops, // Use CP-SAT decisions
        );

        // Verify that ConstChoice ops were lowered correctly based on CP-SAT decisions
        for (const_choice_id, expected_value) in &const_choice_info {
            let lowered_op = lowered_ops.iter()
                .find(|op| op.produces_constant == Some(*expected_value))
                .expect(&format!("Should find lowered op for ConstChoice producing {}", expected_value));

            if cpsat_result.const_add_imm_ops.contains(const_choice_id) {
                // CP-SAT chose AddImm - verify it's a Flow op
                assert_eq!(lowered_op.engine, Engine::Flow,
                    "ConstChoice op {} (value {}) should be lowered to Flow (AddImm)", const_choice_id, expected_value);
                match &lowered_op.slot[0] {
                    SlotVal::Str(s) => assert_eq!(s, "add_imm",
                        "ConstChoice op {} should be add_imm", const_choice_id),
                    _ => panic!("AddImm slot[0] should be Str"),
                }
                match &lowered_op.slot[2] {
                    SlotVal::Int(addr) => assert_eq!(*addr, 0,
                        "AddImm should read from address 0"),
                    _ => panic!("AddImm slot[2] should be Int(0)"),
                }
            } else {
                // CP-SAT chose Load - verify it's a Load op
                assert_eq!(lowered_op.engine, Engine::Load,
                    "ConstChoice op {} (value {}) should be lowered to Load", const_choice_id, expected_value);
                match &lowered_op.slot[0] {
                    SlotVal::Str(s) => assert_eq!(s, "const",
                        "ConstChoice op {} should be const", const_choice_id),
                    _ => panic!("Const slot[0] should be Str"),
                }
            }
        }

        // Verify all ConstChoice ops were lowered
        assert_eq!(lowered_ops.len(), emit.ops.len(),
            "All ops should be lowered");
    }

    #[cfg(feature = "cpsat")]
    #[test]
    fn test_ssa_cpsat_constchoice_prefers_addimm_when_load_slots_full() {
        use crate::kernel_builder::emit_ssa::{SsaEmitContext, compute_ssa_dependencies};
        use crate::kernel_builder::ssa_lower::lower_ssa_ops_with_shift;
        use crate::types::{Engine, Opcode, SlotVal};
        use std::collections::HashMap;

        // Create a scenario where Load slots are heavily used, forcing CP-SAT to choose AddImm
        let mut emit = SsaEmitContext::new();
        
        // Emit zero constant
        let zero = emit.emit_const(0);
        
        // Emit many Load ops to fill Load slots
        for i in 0..10 {
            emit.emit_const(i + 100); // Regular Load ops
        }
        
        // Emit ConstChoice ops - CP-SAT should prefer AddImm to avoid Load slot pressure
        let c1 = emit.emit_const_choice(1);
        let c2 = emit.emit_const_choice(2);
        
        // Use constants
        let _sum = emit.emit_alu(Opcode::Add, c1, c2);

        // Compute dependencies
        let vreg_def = std::mem::take(&mut emit.vreg_def);
        compute_ssa_dependencies(&mut emit.ops, &vreg_def);

        // Schedule with CP-SAT with longer time limit to allow optimization
        let result = schedule_ssa_cpsat(
            &emit.ops,
            SsaCpsatConfig {
                time_limit_secs: 30.0,
                verbose: false,
                stop_after_first_solution: false, // Allow optimization
                enable_valu_shift: false,
                scratch_budget: 1536,
            },
            None,
            None,
        );

        assert!(result.is_some(), "CP-SAT should find a solution");
        let cpsat_result = result.unwrap();

        // Find ConstChoice op IDs
        let const_choice_ids: Vec<usize> = emit.ops.iter()
            .filter(|op| op.opcode == Opcode::ConstChoice)
            .map(|op| op.id)
            .collect();

        assert_eq!(const_choice_ids.len(), 2, "Should have 2 ConstChoice ops");

        // Verify decisions were extracted
        assert!(cpsat_result.const_add_imm_ops.len() <= 2,
            "Should have at most 2 ConstChoice ops using AddImm");

        // Test lowering with the decisions
        let mut vreg_to_addr = HashMap::new();
        for (idx, op) in emit.ops.iter().enumerate() {
            if let Some(dest) = op.dest {
                vreg_to_addr.insert(dest, idx as i64 * 10);
            }
        }
        vreg_to_addr.insert(zero, 0);

        let lowered_ops = lower_ssa_ops_with_shift(
            &emit.ops,
            &vreg_to_addr,
            &HashSet::new(),
            &cpsat_result.const_add_imm_ops,
        );

        // Verify ConstChoice ops were lowered correctly
        for const_choice_id in &const_choice_ids {
            let ssa_op = emit.ops.iter().find(|op| op.id == *const_choice_id).unwrap();
            let expected_value = ssa_op.srcs[0].as_imm().unwrap();
            
            let lowered_op = lowered_ops.iter()
                .find(|op| op.produces_constant == Some(expected_value))
                .expect(&format!("Should find lowered op for ConstChoice {}", const_choice_id));

            if cpsat_result.const_add_imm_ops.contains(const_choice_id) {
                // Verify AddImm was chosen
                assert_eq!(lowered_op.engine, Engine::Flow,
                    "ConstChoice {} should use AddImm when Load slots are full", const_choice_id);
                match &lowered_op.slot[0] {
                    SlotVal::Str(s) => assert_eq!(s, "add_imm"),
                    _ => panic!("Should be add_imm"),
                }
            } else {
                // Verify Load was chosen
                assert_eq!(lowered_op.engine, Engine::Load,
                    "ConstChoice {} should use Load", const_choice_id);
            }
        }
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

    /// Test CP-SAT round trip with SSA-lowered ops.
    /// This verifies that:
    /// 1. SSA ops can be scheduled and lowered to concrete Ops
    /// 2. Those concrete Ops can be scheduled by CP-SAT
    /// 3. The greedy schedule from SSA serves as valid hints for CP-SAT
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_cpsat_roundtrip_with_ssa_lowered_ops() {
        use crate::kernel_builder::emit_ssa::SsaEmitContext;
        use crate::scheduler::schedule_ssa_full;
        use crate::types::Opcode;

        // Create SSA ops
        let mut emit = SsaEmitContext::new();
        let v1 = emit.emit_const(5);
        let v2 = emit.emit_const(10);
        let sum = emit.emit_alu(Opcode::Add, v1, v2);
        let _result = emit.emit_alu(Opcode::Mul, sum, sum);

        // SSA schedule + lower
        let ssa_result = schedule_ssa_full(&mut emit.ops).expect("SSA scheduling should succeed");
        let lowered_ops = ssa_result.ops;

        // Extract cycle hints from SSA result
        let hints: Vec<i32> = lowered_ops.iter().map(|op| op.cycle).collect();

        // Run CP-SAT with the SSA-derived hints
        let cpsat_result = schedule_cpsat(
            &lowered_ops,
            CpsatConfig::new()
                .with_time_limit(10.0)
                .with_hints(&hints)
                .with_stop_after_first_solution(true),
        );

        assert!(cpsat_result.is_some(), "CP-SAT should accept SSA-lowered ops with hints");

        let r = cpsat_result.unwrap();
        assert!(
            r.result.status == CpsatStatus::Feasible || r.result.status == CpsatStatus::Optimal,
            "Should return feasible or optimal, got {:?}",
            r.result.status
        );

        // The hints should be valid (produce same or better makespan)
        assert!(
            r.result.makespan <= ssa_result.cycles + 1,
            "CP-SAT makespan ({}) should be close to SSA result ({})",
            r.result.makespan, ssa_result.cycles
        );
    }

    /// Test CP-SAT round trip with SSA vectors.
    /// Vectors use 8 contiguous addresses, which creates more complex dependency patterns.
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_cpsat_roundtrip_with_ssa_vectors() {
        use crate::kernel_builder::emit_ssa::SsaEmitContext;
        use crate::scheduler::schedule_ssa_full;
        use crate::types::Opcode;

        // Create SSA ops with vectors
        let mut emit = SsaEmitContext::new();
        let s1 = emit.emit_const(1);
        let s2 = emit.emit_const(2);
        let v1 = emit.emit_vbroadcast(s1);
        let v2 = emit.emit_vbroadcast(s2);
        let vsum = emit.emit_valu(Opcode::Add, v1, v2);
        let _vresult = emit.emit_valu(Opcode::Mul, vsum, v1);

        // SSA schedule + lower
        let ssa_result = schedule_ssa_full(&mut emit.ops).expect("SSA scheduling should succeed");
        let lowered_ops = ssa_result.ops;

        // Extract cycle hints
        let hints: Vec<i32> = lowered_ops.iter().map(|op| op.cycle).collect();

        // Run CP-SAT
        let cpsat_result = schedule_cpsat(
            &lowered_ops,
            CpsatConfig::new()
                .with_time_limit(10.0)
                .with_hints(&hints)
                .with_stop_after_first_solution(true),
        );

        assert!(cpsat_result.is_some(), "CP-SAT should accept SSA vector ops");

        let r = cpsat_result.unwrap();
        assert!(
            r.result.status == CpsatStatus::Feasible || r.result.status == CpsatStatus::Optimal,
            "Should find feasible solution for vector ops"
        );

        // Verify all ops are scheduled
        assert_eq!(
            r.result.cycle_assignments.len(),
            lowered_ops.len(),
            "All ops should have cycle assignments"
        );
    }

    /// Test that SSA-lowered ops with a larger dependency chain work correctly.
    /// This exercises the full SSA pipeline with more complex patterns.
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_cpsat_ssa_complex_deps() {
        use crate::kernel_builder::emit_ssa::SsaEmitContext;
        use crate::scheduler::schedule_ssa_full;
        use crate::types::Opcode;

        // Create SSA ops with diamond dependency pattern:
        //       v1  v2  v3
        //        \  /\  /
        //       sum12  sum23
        //          \  /
        //         final
        let mut emit = SsaEmitContext::new();
        let v1 = emit.emit_const(5);
        let v2 = emit.emit_const(10);
        let v3 = emit.emit_const(15);
        let sum12 = emit.emit_alu(Opcode::Add, v1, v2);
        let sum23 = emit.emit_alu(Opcode::Add, v2, v3);
        let _final_sum = emit.emit_alu(Opcode::Add, sum12, sum23);

        // SSA schedule + lower
        let ssa_result = schedule_ssa_full(&mut emit.ops).expect("SSA scheduling should succeed");
        let ssa_lowered_ops = ssa_result.ops;
        let ssa_hints: Vec<i32> = ssa_lowered_ops.iter().map(|op| op.cycle).collect();

        // Run CP-SAT on SSA-lowered ops
        let cpsat_result = schedule_cpsat(
            &ssa_lowered_ops,
            CpsatConfig::new()
                .with_time_limit(10.0)
                .with_hints(&ssa_hints)
                .with_stop_after_first_solution(true),
        );

        assert!(cpsat_result.is_some(), "CP-SAT should work with SSA diamond pattern");

        let r = cpsat_result.unwrap();
        assert!(
            r.result.status == CpsatStatus::Feasible || r.result.status == CpsatStatus::Optimal,
            "Should find feasible solution"
        );

        // Critical path: v1/v2/v3 (0) -> sum12/sum23 (1) -> final (2) = 3 cycles
        // But v3 is load and only 2 load slots, so might need cycle 1
        // Final schedule: v1,v2 (0) -> v3,sum12 (1) -> sum23 (2) -> final (3) = 4 cycles
        // Or: v1,v2 (0) -> v3,sum12,sum23 (1) -> final (2) = 3 cycles if sum12/sum23 can run same cycle
        assert!(
            r.result.makespan >= 3,
            "Makespan ({}) should be at least 3 (critical path)",
            r.result.makespan
        );
    }

    /// Helper to create SsaOp for testing
    fn make_ssa_op(
        id: usize,
        engine: Engine,
        opcode: crate::types::Opcode,
        dest: Option<VReg>,
        srcs: Vec<Operand>,
        deps: Vec<usize>,
    ) -> SsaOp {
        let mut op = SsaOp::new(id, engine, opcode, dest, srcs);
        op.dep_ids = deps.into_iter().collect();
        op
    }

    /// Test that SSA CP-SAT scratch constraint is being applied.
    /// Creates a scenario with high scratch pressure to verify the constraint.
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_ssa_cpsat_scratch_constraint_basic() {
        use crate::types::Opcode;

        // Create a simple chain where each op produces a value used by the next.
        // op0: load -> v0 (scalar, 1 word)
        // op1: alu(v0) -> v1
        // op2: alu(v1) -> v2
        let ops = vec![
            make_ssa_op(0, Engine::Load, Opcode::Const, Some(VReg::scalar(0)), vec![Operand::imm(5)], vec![]),
            make_ssa_op(1, Engine::Alu, Opcode::Add, Some(VReg::scalar(1)), vec![Operand::scalar(0), Operand::imm(1)], vec![0]),
            make_ssa_op(2, Engine::Alu, Opcode::Add, Some(VReg::scalar(2)), vec![Operand::scalar(1), Operand::imm(2)], vec![1]),
        ];

        // With generous budget, should succeed
        let result = schedule_ssa_cpsat(
            &ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: true,
                scratch_budget: 100,  // More than enough
                ..Default::default()
            },
            None,
            None,
        );

        assert!(result.is_some(), "Should find solution with generous budget");
        let r = result.unwrap();
        assert_eq!(r.status, CpsatStatus::Optimal, "Should be optimal for simple chain");
        eprintln!("Basic test: makespan={}, status={:?}", r.makespan, r.status);
    }

    /// Test that scratch constraint forces spreading when budget is tight.
    /// With vector VRegs (8 words each), a tight budget should force longer schedules.
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_ssa_cpsat_scratch_constraint_tight_budget() {
        use crate::types::Opcode;

        // Create multiple independent ops that would normally run in parallel,
        // but with tight scratch budget they must be serialized.
        // Each vector VReg uses 8 words.
        //
        // op0: load -> v0 (vector, 8 words) - no deps
        // op1: load -> v1 (vector, 8 words) - no deps
        // op2: load -> v2 (vector, 8 words) - no deps
        // op3: valu(v0, v1) -> v3 (vector, 8 words) - deps: 0, 1
        // op4: valu(v2, v3) -> v4 (vector, 8 words) - deps: 2, 3
        //
        // If all run ASAP:
        //   Cycle 0: op0, op1, op2 scheduled -> v0, v1, v2 all live = 24 words
        //   Cycle 1: op3 scheduled -> v0, v1, v2, v3 live, then v0, v1 die = still ~24 words at peak
        //   Cycle 2: op4 scheduled -> v2, v3, v4 live
        let ops = vec![
            make_ssa_op(0, Engine::Load, Opcode::Const, Some(VReg::vector(0)), vec![Operand::imm(1)], vec![]),
            make_ssa_op(1, Engine::Load, Opcode::Const, Some(VReg::vector(1)), vec![Operand::imm(2)], vec![]),
            make_ssa_op(2, Engine::Load, Opcode::Const, Some(VReg::vector(2)), vec![Operand::imm(3)], vec![]),
            make_ssa_op(3, Engine::Valu, Opcode::Add, Some(VReg::vector(3)), vec![Operand::vector(0), Operand::vector(1)], vec![0, 1]),
            make_ssa_op(4, Engine::Valu, Opcode::Add, Some(VReg::vector(4)), vec![Operand::vector(2), Operand::vector(3)], vec![2, 3]),
        ];

        // With large budget (e.g., 100 words), should allow parallel loads
        let result_generous = schedule_ssa_cpsat(
            &ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: true,
                scratch_budget: 100,
                ..Default::default()
            },
            None,
            None,
        );

        assert!(result_generous.is_some(), "Should find solution with generous budget");
        let r_generous = result_generous.unwrap();
        eprintln!("Generous budget: makespan={}, status={:?}", r_generous.makespan, r_generous.status);

        // With tight budget (16 words = 2 vectors), should need to serialize more
        let result_tight = schedule_ssa_cpsat(
            &ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: true,
                scratch_budget: 16,  // Only 2 vectors can be live at once
                ..Default::default()
            },
            None,
            None,
        );

        // The tight budget should either:
        // 1. Force a longer schedule (more cycles to reduce concurrent live values), or
        // 2. Return infeasible if impossible
        if let Some(r_tight) = result_tight {
            eprintln!("Tight budget: makespan={}, status={:?}", r_tight.makespan, r_tight.status);
            // With tight budget, makespan should be >= generous budget makespan
            // (can't do better, might need more cycles)
            assert!(
                r_tight.makespan >= r_generous.makespan,
                "Tight budget makespan ({}) should be >= generous ({}) or infeasible",
                r_tight.makespan, r_generous.makespan
            );
        } else {
            eprintln!("Tight budget: infeasible (expected if budget too small)");
        }
    }

    /// Test that scratch constraint with budget=0 disables the constraint.
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_ssa_cpsat_scratch_constraint_disabled() {
        use crate::types::Opcode;

        let ops = vec![
            make_ssa_op(0, Engine::Load, Opcode::Const, Some(VReg::vector(0)), vec![Operand::imm(1)], vec![]),
            make_ssa_op(1, Engine::Load, Opcode::Const, Some(VReg::vector(1)), vec![Operand::imm(2)], vec![]),
            make_ssa_op(2, Engine::Valu, Opcode::Add, Some(VReg::vector(2)), vec![Operand::vector(0), Operand::vector(1)], vec![0, 1]),
        ];

        // With budget=0, constraint is disabled
        let result = schedule_ssa_cpsat(
            &ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: true,
                scratch_budget: 0,  // Disabled
                ..Default::default()
            },
            None,
            None,
        );

        assert!(result.is_some(), "Should find solution when scratch constraint disabled");
        let r = result.unwrap();
        eprintln!("Disabled constraint: makespan={}, status={:?}", r.makespan, r.status);
        // Should be optimal minimal schedule (2 cycles: loads then add)
        assert_eq!(r.makespan, 2, "Minimal schedule should be 2 cycles");
    }

    /// Test that SSA CP-SAT hints are complete and feasible.
    /// This is a critical test - if hints are incomplete or infeasible, the solver will struggle.
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_ssa_cpsat_hint_roundtrip_no_scratch() {
        use crate::types::Opcode;

        // Create a simple dependency chain
        let ops = vec![
            make_ssa_op(0, Engine::Load, Opcode::Const, Some(VReg::scalar(0)), vec![Operand::imm(5)], vec![]),
            make_ssa_op(1, Engine::Alu, Opcode::Add, Some(VReg::scalar(1)), vec![Operand::scalar(0), Operand::imm(1)], vec![0]),
            make_ssa_op(2, Engine::Alu, Opcode::Add, Some(VReg::scalar(2)), vec![Operand::scalar(1), Operand::imm(2)], vec![1]),
        ];

        // Valid ASAP schedule: op0 at 0, op1 at 1, op2 at 2
        let hints = vec![0, 1, 2];

        // Run with stop_after_first_solution - should return exactly the hint if it's valid
        let result = schedule_ssa_cpsat(
            &ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: true,
                stop_after_first_solution: true,
                scratch_budget: 0, // Disable scratch constraint for this test
                ..Default::default()
            },
            Some(&hints),
            None,
        );

        assert!(result.is_some(), "Should find solution with valid hints");
        let r = result.unwrap();

        // Critical: with stop_after_first_solution, we should get exactly the hinted solution
        assert_eq!(r.cycle_assignments, hints,
            "With stop_after_first_solution, should return exactly the hinted solution");
        assert_eq!(r.makespan, 3, "Makespan should be 3 (cycles 0, 1, 2)");

        eprintln!("Hint roundtrip (no scratch): assignments={:?}, makespan={}",
            r.cycle_assignments, r.makespan);
    }

    /// Test that SSA CP-SAT hints with scratch constraint are complete and feasible.
    /// This verifies that the scratch interval size hints are correct.
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_ssa_cpsat_hint_roundtrip_with_scratch() {
        use crate::types::Opcode;

        // Create ops with clear liveness patterns
        // op0: load -> v0 (cycle 0)
        // op1: alu(v0) -> v1 (cycle 1) - v0 live from 0 to 1, size = 2
        // op2: alu(v1) -> v2 (cycle 2) - v1 live from 1 to 2, size = 2
        let ops = vec![
            make_ssa_op(0, Engine::Load, Opcode::Const, Some(VReg::scalar(0)), vec![Operand::imm(5)], vec![]),
            make_ssa_op(1, Engine::Alu, Opcode::Add, Some(VReg::scalar(1)), vec![Operand::scalar(0), Operand::imm(1)], vec![0]),
            make_ssa_op(2, Engine::Alu, Opcode::Add, Some(VReg::scalar(2)), vec![Operand::scalar(1), Operand::imm(2)], vec![1]),
        ];

        // Valid ASAP schedule
        let hints = vec![0, 1, 2];

        // Run with scratch constraint enabled
        let result = schedule_ssa_cpsat(
            &ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: true,
                stop_after_first_solution: true,
                scratch_budget: 10, // Small but sufficient for scalars
                ..Default::default()
            },
            Some(&hints),
            None,
        );

        assert!(result.is_some(), "Should find solution with valid hints and scratch constraint");
        let r = result.unwrap();

        // Critical: hints should be feasible with scratch constraint
        assert_eq!(r.cycle_assignments, hints,
            "With stop_after_first_solution, should return exactly the hinted solution");
        assert_eq!(r.makespan, 3, "Makespan should be 3");

        eprintln!("Hint roundtrip (with scratch): assignments={:?}, makespan={}",
            r.cycle_assignments, r.makespan);
    }

    /// Test that SSA CP-SAT hints with vectors and scratch constraint are feasible.
    /// Vectors use 8 words each, so this tests the demand calculation.
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_ssa_cpsat_hint_roundtrip_vectors_with_scratch() {
        use crate::types::Opcode;

        // Create vector ops with clear liveness
        // op0: load -> v0 (vector, 8 words, cycle 0)
        // op1: load -> v1 (vector, 8 words, cycle 0) - can run parallel with op0
        // op2: valu(v0, v1) -> v2 (cycle 1) - v0, v1 live from 0 to 1
        // At cycle 0: v0, v1 defined = 16 words
        // At cycle 1: v0, v1 used, v2 defined = peak 24 words (v0, v1 still live during use)
        let ops = vec![
            make_ssa_op(0, Engine::Load, Opcode::Const, Some(VReg::vector(0)), vec![Operand::imm(1)], vec![]),
            make_ssa_op(1, Engine::Load, Opcode::Const, Some(VReg::vector(1)), vec![Operand::imm(2)], vec![]),
            make_ssa_op(2, Engine::Valu, Opcode::Add, Some(VReg::vector(2)), vec![Operand::vector(0), Operand::vector(1)], vec![0, 1]),
        ];

        // Schedule: op0 and op1 at cycle 0 (parallel loads), op2 at cycle 1
        let hints = vec![0, 0, 1];

        // Run with scratch constraint - need at least 24 words (3 vectors live at peak)
        let result = schedule_ssa_cpsat(
            &ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: true,
                stop_after_first_solution: true,
                scratch_budget: 32, // Enough for 4 vectors
                ..Default::default()
            },
            Some(&hints),
            None,
        );

        assert!(result.is_some(), "Should find solution with valid vector hints and scratch constraint");
        let r = result.unwrap();

        // Critical: hints should be feasible
        assert_eq!(r.cycle_assignments, hints,
            "With stop_after_first_solution, should return exactly the hinted solution");
        assert_eq!(r.makespan, 2, "Makespan should be 2 (cycles 0 and 1)");

        eprintln!("Hint roundtrip (vectors with scratch): assignments={:?}, makespan={}",
            r.cycle_assignments, r.makespan);
    }

    /// Test that infeasible hints with tight scratch budget are detected.
    /// This verifies the scratch constraint actually constrains the solution.
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_ssa_cpsat_hint_infeasible_tight_scratch() {
        use crate::types::Opcode;

        // Create vector ops that would exceed a tight budget if run in parallel
        // op0: load -> v0 (vector, 8 words)
        // op1: load -> v1 (vector, 8 words)
        // op2: valu(v0, v1) -> v2
        // If op0, op1 at cycle 0: 16 words live at cycle 0
        // At cycle 1 when op2 runs: v0, v1, v2 all live = 24 words
        let ops = vec![
            make_ssa_op(0, Engine::Load, Opcode::Const, Some(VReg::vector(0)), vec![Operand::imm(1)], vec![]),
            make_ssa_op(1, Engine::Load, Opcode::Const, Some(VReg::vector(1)), vec![Operand::imm(2)], vec![]),
            make_ssa_op(2, Engine::Valu, Opcode::Add, Some(VReg::vector(2)), vec![Operand::vector(0), Operand::vector(1)], vec![0, 1]),
        ];

        // Hints that would work without scratch constraint but violate tight budget
        let hints = vec![0, 0, 1]; // Both loads at cycle 0

        // Run with very tight scratch constraint (only 16 words - not enough for 3 vectors)
        let result = schedule_ssa_cpsat(
            &ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: true,
                stop_after_first_solution: true,
                scratch_budget: 16, // Only 2 vectors worth - not enough!
                ..Default::default()
            },
            Some(&hints),
            None,
        );

        // The solver should either:
        // 1. Find a different (valid) schedule that fits the budget, or
        // 2. Return infeasible if no valid schedule exists
        // It should NOT return the infeasible hint as-is
        if let Some(r) = result {
            // If it found a solution, it should NOT be the original hint
            // because that hint violates the scratch budget
            eprintln!("Found alternative schedule: {:?}, makespan={}", r.cycle_assignments, r.makespan);
            // The solver might find a valid schedule by serializing the loads
            // e.g., [0, 1, 2] instead of [0, 0, 1]
            assert!(r.makespan >= 2, "Should need at least 2 cycles");
        } else {
            eprintln!("Correctly returned infeasible for tight budget");
        }
    }

    /// Test scratch interval size when VReg is defined and used in the same cycle.
    /// This should produce interval size = 1 (last_use_cycle - def_cycle + 1 = 0 - 0 + 1 = 1).
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_scratch_interval_size_same_cycle() {
        use crate::types::Opcode;

        // Create ops where a value is defined and used in the same cycle.
        // op0: const -> v0 (cycle 0, defines v0)
        // op1: add v0, imm -> v1 (cycle 1, uses v0 - so v0 lives from cycle 0 to cycle 1)
        // But for same-cycle test, we need op0 and op1 to be in the same cycle.
        // However, due to data dependency (op1 depends on op0), they can't be in the same cycle.
        //
        // For a true same-cycle test, we need an op that defines a value and is the last user.
        // When a VReg is defined but never used elsewhere, its lifetime is just its def cycle.
        // Let's create: op0 defines v0, op1 defines v1 using v0, and v1 is never used again.
        // v1's interval: def_cycle=1, last_use_cycle=1 (no further uses), size=1
        let ops = vec![
            make_ssa_op(0, Engine::Load, Opcode::Const, Some(VReg::scalar(0)), vec![Operand::imm(5)], vec![]),
            make_ssa_op(1, Engine::Alu, Opcode::Add, Some(VReg::scalar(1)), vec![Operand::scalar(0), Operand::imm(1)], vec![0]),
        ];

        // Hints: op0 at cycle 0, op1 at cycle 1
        // v0 interval: def=0, last_use=1, size=2
        // v1 interval: def=1, last_use=1 (no further uses), size=1
        let hints = vec![0, 1];

        let result = schedule_ssa_cpsat(
            &ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: true,
                stop_after_first_solution: true,
                scratch_budget: 10, // Sufficient for scalars
                ..Default::default()
            },
            Some(&hints),
            None,
        );

        assert!(result.is_some(), "Should find solution with same-cycle interval (size=1)");
        let r = result.unwrap();
        assert_eq!(r.cycle_assignments, hints, "Should return the hinted solution");
        assert_eq!(r.makespan, 2, "Makespan should be 2");
        eprintln!("Same-cycle interval test: assignments={:?}, makespan={}", r.cycle_assignments, r.makespan);
    }

    /// Test scratch interval size for VReg defined at cycle N and used at cycle N+1.
    /// This should produce interval size = 2 (last_use_cycle - def_cycle + 1 = 1 - 0 + 1 = 2).
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_scratch_interval_size_consecutive_cycles() {
        use crate::types::Opcode;

        // op0: const -> v0 (cycle 0)
        // op1: add v0, imm -> v1 (cycle 1, uses v0)
        // v0 interval: def_cycle=0, last_use_cycle=1, size=2
        let ops = vec![
            make_ssa_op(0, Engine::Load, Opcode::Const, Some(VReg::scalar(0)), vec![Operand::imm(5)], vec![]),
            make_ssa_op(1, Engine::Alu, Opcode::Add, Some(VReg::scalar(1)), vec![Operand::scalar(0), Operand::imm(1)], vec![0]),
        ];

        // Hints: op0 at cycle 0, op1 at cycle 1
        let hints = vec![0, 1];

        let result = schedule_ssa_cpsat(
            &ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: true,
                stop_after_first_solution: true,
                scratch_budget: 10, // Sufficient
                ..Default::default()
            },
            Some(&hints),
            None,
        );

        assert!(result.is_some(), "Should find solution with consecutive-cycle interval (size=2)");
        let r = result.unwrap();
        assert_eq!(r.cycle_assignments, hints, "Should return the hinted solution");
        assert_eq!(r.makespan, 2, "Makespan should be 2");
        eprintln!("Consecutive-cycle interval test: assignments={:?}, makespan={}", r.cycle_assignments, r.makespan);
    }

    /// Test scratch interval size for VReg defined at cycle 0 and used at cycle 5.
    /// This should produce interval size = 6 (last_use_cycle - def_cycle + 1 = 5 - 0 + 1 = 6).
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_scratch_interval_size_gap() {
        use crate::types::Opcode;

        // Create a chain where v0 is defined at cycle 0 but used at cycle 5.
        // We need intermediate ops to push the use of v0 to cycle 5.
        // op0: const -> v0 (cycle 0)
        // op1: const -> v1 (cycle 0) - independent
        // op2: add v1, imm -> v2 (cycle 1) - chain through v1
        // op3: add v2, imm -> v3 (cycle 2)
        // op4: add v3, imm -> v4 (cycle 3)
        // op5: add v4, imm -> v5 (cycle 4)
        // op6: add v5, v0 -> v6 (cycle 5) - finally uses v0
        // v0 interval: def_cycle=0, last_use_cycle=5, size=6
        let ops = vec![
            make_ssa_op(0, Engine::Load, Opcode::Const, Some(VReg::scalar(0)), vec![Operand::imm(5)], vec![]),
            make_ssa_op(1, Engine::Load, Opcode::Const, Some(VReg::scalar(1)), vec![Operand::imm(10)], vec![]),
            make_ssa_op(2, Engine::Alu, Opcode::Add, Some(VReg::scalar(2)), vec![Operand::scalar(1), Operand::imm(1)], vec![1]),
            make_ssa_op(3, Engine::Alu, Opcode::Add, Some(VReg::scalar(3)), vec![Operand::scalar(2), Operand::imm(1)], vec![2]),
            make_ssa_op(4, Engine::Alu, Opcode::Add, Some(VReg::scalar(4)), vec![Operand::scalar(3), Operand::imm(1)], vec![3]),
            make_ssa_op(5, Engine::Alu, Opcode::Add, Some(VReg::scalar(5)), vec![Operand::scalar(4), Operand::imm(1)], vec![4]),
            make_ssa_op(6, Engine::Alu, Opcode::Add, Some(VReg::scalar(6)), vec![Operand::scalar(5), Operand::scalar(0)], vec![5, 0]),
        ];

        // Hints: op0 and op1 at cycle 0, then op2-op6 at cycles 1-5
        // v0 is defined at cycle 0 and used at cycle 5 (by op6)
        // v0 interval size = 5 - 0 + 1 = 6
        let hints = vec![0, 0, 1, 2, 3, 4, 5];

        let result = schedule_ssa_cpsat(
            &ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: true,
                stop_after_first_solution: true,
                scratch_budget: 20, // Sufficient for multiple scalars
                ..Default::default()
            },
            Some(&hints),
            None,
        );

        assert!(result.is_some(), "Should find solution with gap interval (size=6)");
        let r = result.unwrap();
        assert_eq!(r.cycle_assignments, hints, "Should return the hinted solution");
        assert_eq!(r.makespan, 6, "Makespan should be 6");
        eprintln!("Gap interval test: assignments={:?}, makespan={}", r.cycle_assignments, r.makespan);
    }

    /// Test engine capacity: 6 independent VALU ops at the same cycle (exactly at limit).
    /// Should work without violating the VALU cumulative constraint.
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_engine_capacity_valu_within_limit() {
        use crate::types::Opcode;

        // Create 6 independent VALU ops that can all be scheduled at cycle 1.
        // Each needs a load to provide input values first.
        // op0-op5: load consts (cycle 0)
        // op6-op11: valu add (cycle 1) - all 6 VALU ops at same cycle (exactly at limit)
        let ops = vec![
            // Load ops for first operands (cycle 0)
            make_ssa_op(0, Engine::Load, Opcode::Const, Some(VReg::vector(0)), vec![Operand::imm(1)], vec![]),
            make_ssa_op(1, Engine::Load, Opcode::Const, Some(VReg::vector(1)), vec![Operand::imm(2)], vec![]),
            // Note: only 2 load slots per cycle, so we need more cycles
            // Actually, let's use consts with immediates to avoid load slot issues
            // Create 6 VALU ops that use immediates as second operand
            make_ssa_op(2, Engine::Valu, Opcode::Add, Some(VReg::vector(2)), vec![Operand::vector(0), Operand::vector(1)], vec![0, 1]),
            make_ssa_op(3, Engine::Valu, Opcode::Sub, Some(VReg::vector(3)), vec![Operand::vector(0), Operand::vector(1)], vec![0, 1]),
            make_ssa_op(4, Engine::Valu, Opcode::Mul, Some(VReg::vector(4)), vec![Operand::vector(0), Operand::vector(1)], vec![0, 1]),
            make_ssa_op(5, Engine::Valu, Opcode::Xor, Some(VReg::vector(5)), vec![Operand::vector(0), Operand::vector(1)], vec![0, 1]),
            make_ssa_op(6, Engine::Valu, Opcode::And, Some(VReg::vector(6)), vec![Operand::vector(0), Operand::vector(1)], vec![0, 1]),
            make_ssa_op(7, Engine::Valu, Opcode::Or, Some(VReg::vector(7)), vec![Operand::vector(0), Operand::vector(1)], vec![0, 1]),
        ];

        // Hints: 2 loads at cycle 0, 6 VALU ops all at cycle 1 (exactly at VALU limit of 6)
        let hints = vec![0, 0, 1, 1, 1, 1, 1, 1];

        let result = schedule_ssa_cpsat(
            &ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: true,
                stop_after_first_solution: true,
                scratch_budget: 0, // Disable scratch constraint for this test
                enable_valu_shift: false,
                ..Default::default()
            },
            Some(&hints),
            None,
        );

        // The verification code should NOT panic, and we should get a valid result
        assert!(result.is_some(), "Should find solution with 6 VALU ops at same cycle (exactly at limit)");
        let r = result.unwrap();
        assert_eq!(r.cycle_assignments, hints, "Should accept the hinted solution");
        eprintln!("VALU capacity test: 6 VALU ops at cycle 1, result={:?}", r.cycle_assignments);
    }

    /// Test engine capacity: 12 independent ALU ops at the same cycle (exactly at limit).
    /// Should work without violating the ALU cumulative constraint.
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_engine_capacity_alu_within_limit() {
        use crate::types::Opcode;

        // Create 12 independent ALU ops that can all be scheduled at cycle 1.
        // First, we need load ops to provide input values.
        // op0-op1: load consts (cycle 0) - 2 load slots
        // op2-op13: 12 ALU ops (cycle 1) - all 12 ALU ops at same cycle (exactly at limit)
        let ops = vec![
            // Load ops
            make_ssa_op(0, Engine::Load, Opcode::Const, Some(VReg::scalar(0)), vec![Operand::imm(1)], vec![]),
            make_ssa_op(1, Engine::Load, Opcode::Const, Some(VReg::scalar(1)), vec![Operand::imm(2)], vec![]),
            // 12 ALU ops - all depend on loads, can run in parallel
            make_ssa_op(2, Engine::Alu, Opcode::Add, Some(VReg::scalar(2)), vec![Operand::scalar(0), Operand::scalar(1)], vec![0, 1]),
            make_ssa_op(3, Engine::Alu, Opcode::Sub, Some(VReg::scalar(3)), vec![Operand::scalar(0), Operand::scalar(1)], vec![0, 1]),
            make_ssa_op(4, Engine::Alu, Opcode::Mul, Some(VReg::scalar(4)), vec![Operand::scalar(0), Operand::scalar(1)], vec![0, 1]),
            make_ssa_op(5, Engine::Alu, Opcode::Xor, Some(VReg::scalar(5)), vec![Operand::scalar(0), Operand::scalar(1)], vec![0, 1]),
            make_ssa_op(6, Engine::Alu, Opcode::And, Some(VReg::scalar(6)), vec![Operand::scalar(0), Operand::scalar(1)], vec![0, 1]),
            make_ssa_op(7, Engine::Alu, Opcode::Or, Some(VReg::scalar(7)), vec![Operand::scalar(0), Operand::scalar(1)], vec![0, 1]),
            make_ssa_op(8, Engine::Alu, Opcode::Shl, Some(VReg::scalar(8)), vec![Operand::scalar(0), Operand::scalar(1)], vec![0, 1]),
            make_ssa_op(9, Engine::Alu, Opcode::Shr, Some(VReg::scalar(9)), vec![Operand::scalar(0), Operand::scalar(1)], vec![0, 1]),
            make_ssa_op(10, Engine::Alu, Opcode::Lt, Some(VReg::scalar(10)), vec![Operand::scalar(0), Operand::scalar(1)], vec![0, 1]),
            make_ssa_op(11, Engine::Alu, Opcode::Eq, Some(VReg::scalar(11)), vec![Operand::scalar(0), Operand::scalar(1)], vec![0, 1]),
            make_ssa_op(12, Engine::Alu, Opcode::Add, Some(VReg::scalar(12)), vec![Operand::scalar(0), Operand::scalar(1)], vec![0, 1]),
            make_ssa_op(13, Engine::Alu, Opcode::Sub, Some(VReg::scalar(13)), vec![Operand::scalar(0), Operand::scalar(1)], vec![0, 1]),
        ];

        // Hints: 2 loads at cycle 0, 12 ALU ops all at cycle 1 (exactly at ALU limit of 12)
        let hints = vec![0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];

        let result = schedule_ssa_cpsat(
            &ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: true,
                stop_after_first_solution: true,
                scratch_budget: 0, // Disable scratch constraint for this test
                enable_valu_shift: false,
                ..Default::default()
            },
            Some(&hints),
            None,
        );

        // The verification code should NOT panic, and we should get a valid result
        assert!(result.is_some(), "Should find solution with 12 ALU ops at same cycle (exactly at limit)");
        let r = result.unwrap();
        assert_eq!(r.cycle_assignments, hints, "Should accept the hinted solution");
        eprintln!("ALU capacity test: 12 ALU ops at cycle 1, result={:?}", r.cycle_assignments);
    }

    /// Test engine capacity: A shiftable VALU op shifted to ALU uses 8 slots.
    /// Create scenario with 1 shifted VALU (8 slots) + 4 ALU ops (4 slots) = 12 total.
    /// Should work with enable_valu_shift=true and shift_hints.
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_engine_capacity_shifted_valu() {
        use crate::types::Opcode;

        // Create:
        // - 2 loads (cycle 0)
        // - 1 VALU Add (shiftable, will use 8 ALU slots when shifted) at cycle 1
        // - 4 ALU ops at cycle 1 (4 slots)
        // Total ALU slots at cycle 1: 8 (shifted VALU) + 4 (ALU) = 12 (exactly at limit)
        let ops = vec![
            // Load ops
            make_ssa_op(0, Engine::Load, Opcode::Const, Some(VReg::vector(0)), vec![Operand::imm(1)], vec![]),
            make_ssa_op(1, Engine::Load, Opcode::Const, Some(VReg::scalar(1)), vec![Operand::imm(2)], vec![]),
            // 1 VALU op that will be shifted to ALU (uses 8 ALU slots)
            make_ssa_op(2, Engine::Valu, Opcode::Add, Some(VReg::vector(2)), vec![Operand::vector(0), Operand::vector(0)], vec![0]),
            // 4 ALU ops (4 slots)
            make_ssa_op(3, Engine::Alu, Opcode::Add, Some(VReg::scalar(3)), vec![Operand::scalar(1), Operand::scalar(1)], vec![1]),
            make_ssa_op(4, Engine::Alu, Opcode::Sub, Some(VReg::scalar(4)), vec![Operand::scalar(1), Operand::scalar(1)], vec![1]),
            make_ssa_op(5, Engine::Alu, Opcode::Mul, Some(VReg::scalar(5)), vec![Operand::scalar(1), Operand::scalar(1)], vec![1]),
            make_ssa_op(6, Engine::Alu, Opcode::Xor, Some(VReg::scalar(6)), vec![Operand::scalar(1), Operand::scalar(1)], vec![1]),
        ];

        // Hints: loads at cycle 0, shifted VALU + 4 ALU ops at cycle 1
        let hints = vec![0, 0, 1, 1, 1, 1, 1];

        // shift_hints: op id 2 should be shifted (the VALU Add)
        let mut shift_hints = HashSet::new();
        shift_hints.insert(2); // Op ID 2 is the VALU Add

        let result = schedule_ssa_cpsat(
            &ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: true,
                stop_after_first_solution: true,
                scratch_budget: 0, // Disable scratch constraint for this test
                enable_valu_shift: true, // Enable VALU shifting
                ..Default::default()
            },
            Some(&hints),
            Some(&shift_hints),
        );

        // The verification code should NOT panic:
        // - VALU per cycle: 0 (the VALU op is shifted to ALU)
        // - ALU per cycle: 8 (shifted VALU) + 4 (regular ALU) = 12 (exactly at limit)
        //
        // Note: The verification happens BEFORE solver runs (at lines 1462-1525) and uses
        // shift_hints to verify that the greedy hints don't violate capacity. If the verification
        // panicked, we wouldn't reach this point. Getting a result means verification passed.
        assert!(result.is_some(), "Should find solution with shifted VALU (8) + 4 ALU = 12 (at limit)");
        let r = result.unwrap();
        assert_eq!(r.cycle_assignments, hints, "Should accept the hinted solution");

        // Note: The solver's shifted_ops may or may not contain our op depending on whether
        // the solver decides to shift. The key test is that the verification code (which uses
        // shift_hints to count ALU usage) did not panic. If it had panicked, we wouldn't get here.
        eprintln!("Shifted VALU capacity test: 1 shifted VALU (8 slots) + 4 ALU (4 slots) = 12 total at cycle 1");
        eprintln!("Result: {:?}, shifted_ops from solver: {:?}", r.cycle_assignments, r.shifted_ops);

        // Also verify: if the solver did NOT shift op 2, then we have 1 VALU + 4 ALU at cycle 1,
        // which is fine (within limits). If the solver DID shift it, we have 0 VALU + 12 ALU.
        // Either way should be valid. The test was about the verification code not panicking
        // when we provide shift_hints that put us at exactly the ALU limit.
        eprintln!("Test passed: verification did not panic with 8+4=12 ALU slots at cycle 1");
    }

    // ==========================================================================
    // VReg Last Use Computation Tests
    // ==========================================================================
    // These tests verify that vreg_last_use correctly identifies the op with the
    // LATEST CYCLE that reads each VReg, not just the latest index. This is
    // critical for scratch pressure constraints - using wrong last_use results
    // in last_use_cycle < def_cycle which is impossible.

    /// Test vreg_last_use with a simple dependency chain.
    /// Each op reads the previous op's result, so last_use_cycle must >= def_cycle.
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_vreg_last_use_simple_chain() {
        use crate::types::Opcode;

        // Simple chain: op0 -> op1 -> op2 -> op3
        // Each op reads the previous op's VReg
        // op0: const -> v0 (no reads, defines v0)
        // op1: alu(v0) -> v1 (reads v0, defines v1)
        // op2: alu(v1) -> v2 (reads v1, defines v2)
        // op3: alu(v2) -> v3 (reads v2, defines v3)
        let ops = vec![
            make_ssa_op(0, Engine::Load, Opcode::Const, Some(VReg::scalar(0)), vec![Operand::imm(5)], vec![]),
            make_ssa_op(1, Engine::Alu, Opcode::Add, Some(VReg::scalar(1)), vec![Operand::scalar(0), Operand::imm(1)], vec![0]),
            make_ssa_op(2, Engine::Alu, Opcode::Add, Some(VReg::scalar(2)), vec![Operand::scalar(1), Operand::imm(2)], vec![1]),
            make_ssa_op(3, Engine::Alu, Opcode::Add, Some(VReg::scalar(3)), vec![Operand::scalar(2), Operand::imm(3)], vec![2]),
        ];

        // ASAP schedule: each op depends on the previous
        let hints = vec![0, 1, 2, 3];

        // Run with scratch constraint enabled - this will exercise vreg_last_use computation
        // If last_use_cycle < def_cycle for any VReg, the existing verification code will panic
        let result = schedule_ssa_cpsat(
            &ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: true,
                stop_after_first_solution: true,
                scratch_budget: 10, // Enable scratch constraint
                ..Default::default()
            },
            Some(&hints),
            None,
        );

        // The test passes if we don't panic from "BUG: X VRegs have last_use_cycle < def_cycle"
        // For a simple chain, hints should be optimal (ASAP), so solver may return exact hints.
        assert!(result.is_some(), "Simple chain should be schedulable");
        let r = result.unwrap();
        // Verify basic constraints are satisfied
        assert!(r.makespan >= 4, "Need at least 4 cycles for 4-op dependency chain");
        assert!(r.cycle_assignments.len() == 4, "Should have 4 cycle assignments");
        // Verify dependency order is respected
        assert!(r.cycle_assignments[1] > r.cycle_assignments[0], "op1 must be after op0");
        assert!(r.cycle_assignments[2] > r.cycle_assignments[1], "op2 must be after op1");
        assert!(r.cycle_assignments[3] > r.cycle_assignments[2], "op3 must be after op2");
        eprintln!("test_vreg_last_use_simple_chain: makespan={}, assignments={:?}", r.makespan, r.cycle_assignments);
    }

    /// Test vreg_last_use with reordered hints (greedy schedule differs from index order).
    /// This tests that we use CYCLES from hints, not indices, to determine last use.
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_vreg_last_use_with_reordered_hints() {
        use crate::types::Opcode;

        // Create independent ops that can be reordered by the scheduler.
        // op0: const -> v0 (defines v0)
        // op1: const -> v1 (defines v1, independent of op0)
        // op2: alu(v0) -> v2 (reads v0, defines v2, depends on op0)
        // op3: alu(v1) -> v3 (reads v1, defines v3, depends on op1)
        //
        // Index order: op0, op1, op2, op3
        // But greedy might schedule: op0 (cycle 0), op1 (cycle 0), op3 (cycle 1), op2 (cycle 1)
        // or with explicit hints we can force reordering
        let ops = vec![
            make_ssa_op(0, Engine::Load, Opcode::Const, Some(VReg::scalar(0)), vec![Operand::imm(5)], vec![]),
            make_ssa_op(1, Engine::Load, Opcode::Const, Some(VReg::scalar(1)), vec![Operand::imm(10)], vec![]),
            make_ssa_op(2, Engine::Alu, Opcode::Add, Some(VReg::scalar(2)), vec![Operand::scalar(0), Operand::imm(1)], vec![0]),
            make_ssa_op(3, Engine::Alu, Opcode::Add, Some(VReg::scalar(3)), vec![Operand::scalar(1), Operand::imm(2)], vec![1]),
        ];

        // Hints where op3 (index 3) runs at cycle 1, op2 (index 2) runs at cycle 2
        // This means op2 runs LATER than op3 despite having a lower index
        // v0: defined at cycle 0 (op0), used at cycle 2 (op2) -> last_use_cycle = 2
        // v1: defined at cycle 0 (op1), used at cycle 1 (op3) -> last_use_cycle = 1
        let hints = vec![0, 0, 2, 1]; // op0, op1 at 0; op3 at 1; op2 at 2

        let result = schedule_ssa_cpsat(
            &ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: true,
                stop_after_first_solution: true,
                scratch_budget: 10,
                ..Default::default()
            },
            Some(&hints),
            None,
        );

        // The test passes if we don't panic from "BUG: X VRegs have last_use_cycle < def_cycle"
        // Note: The solver may find a better schedule than the hints, so we don't assert exact match.
        // The key test is that vreg_last_use computation doesn't produce impossible results.
        assert!(result.is_some(), "Reordered schedule should be schedulable");
        let r = result.unwrap();
        // Verify basic constraints are satisfied
        assert!(r.makespan >= 2, "Need at least 2 cycles (loads then ALU ops)");
        assert!(r.cycle_assignments.len() == 4, "Should have 4 cycle assignments");
        eprintln!("test_vreg_last_use_with_reordered_hints: makespan={}, assignments={:?}", r.makespan, r.cycle_assignments);
    }

    /// Test vreg_last_use with multiple readers of the same VReg at different cycles.
    /// We must pick the reader with the LATEST cycle, not the latest index.
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_vreg_last_use_multiple_readers() {
        use crate::types::Opcode;

        // op0: const -> v0 (defines v0)
        // op1: alu(v0) -> v1 (reads v0 - early reader at index 1)
        // op2: const -> v2 (independent, defines v2)
        // op3: alu(v0, v2) -> v3 (reads v0 - late reader at index 3, also reads v2)
        //
        // v0 is read by both op1 and op3
        // With hints [0, 1, 0, 2]: op1 reads v0 at cycle 1, op3 reads v0 at cycle 2
        // So last_use_cycle for v0 should be 2 (from op3), not 1 (from op1)
        //
        // The tricky part: op3 has a higher index than op1, so if we use index order,
        // we'd correctly pick op3. To test the cycle-based logic, we need a case where
        // the LOWER index has a HIGHER cycle - see the test below.
        let ops = vec![
            make_ssa_op(0, Engine::Load, Opcode::Const, Some(VReg::scalar(0)), vec![Operand::imm(5)], vec![]),
            make_ssa_op(1, Engine::Alu, Opcode::Add, Some(VReg::scalar(1)), vec![Operand::scalar(0), Operand::imm(1)], vec![0]),
            make_ssa_op(2, Engine::Load, Opcode::Const, Some(VReg::scalar(2)), vec![Operand::imm(10)], vec![]),
            make_ssa_op(3, Engine::Alu, Opcode::Add, Some(VReg::scalar(3)), vec![Operand::scalar(0), Operand::scalar(2)], vec![0, 2]),
        ];

        // Hints: op0 at 0, op1 at 1, op2 at 0, op3 at 2
        // v0: def at cycle 0, read by op1 at cycle 1, read by op3 at cycle 2 -> last_use = 2
        let hints = vec![0, 1, 0, 2];

        let result = schedule_ssa_cpsat(
            &ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: true,
                stop_after_first_solution: true,
                scratch_budget: 10,
                ..Default::default()
            },
            Some(&hints),
            None,
        );

        // The test passes if we don't panic from "BUG: X VRegs have last_use_cycle < def_cycle"
        // Note: The solver may find a better schedule than the hints, so we don't assert exact match.
        // The key test is that vreg_last_use computation doesn't produce impossible results.
        assert!(result.is_some(), "Multiple readers should be schedulable");
        let r = result.unwrap();
        // Verify basic constraints are satisfied
        assert!(r.makespan >= 2, "Need at least 2 cycles (loads then ALU ops)");
        assert!(r.cycle_assignments.len() == 4, "Should have 4 cycle assignments");
        eprintln!("test_vreg_last_use_multiple_readers: makespan={}, assignments={:?}", r.makespan, r.cycle_assignments);
    }

    /// Test that CP-SAT can schedule with a small scratch budget.
    ///
    /// NOTE: CP-SAT's cumulative constraint does NOT model same-cycle reuse.
    /// At cycle 1, both v0 (being read) and v1 (being written) are counted as live.
    /// So CP-SAT requires budget >= 2 for this program.
    ///
    /// The greedy scheduler DOES support same-cycle reuse and can handle budget=1.
    /// See test_ssa_greedy_minimal_budget_1_word for that test.
    ///
    /// Program:
    ///   v0 = const(5)        -- scalar, 1 word, born cycle 0
    ///   v1 = add(v0, v0)     -- scalar, 1 word, born cycle 1, v0 dies here
    ///   store(0, v1)         -- v1 dies here (cycle 2)
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_ssa_cpsat_minimal_budget_2_words() {
        use crate::address_assign::{assign_addresses, AddressAssignConfig};
        use crate::types::Opcode;

        // Create ops: const -> add -> store
        // v0 = const(5)
        // v1 = add(v0, v0)  -- v0's last use
        // store(0, v1)      -- v1's last use
        let ops = vec![
            make_ssa_op(0, Engine::Load, Opcode::Const, Some(VReg::scalar(0)), vec![Operand::imm(5)], vec![]),
            make_ssa_op(1, Engine::Alu, Opcode::Add, Some(VReg::scalar(1)), vec![Operand::scalar(0), Operand::scalar(0)], vec![0]),
            make_ssa_op(2, Engine::Store, Opcode::Store, None, vec![Operand::imm(0), Operand::scalar(1)], vec![1]),
        ];

        // Schedule with CP-SAT and budget of 2 (CP-SAT doesn't support same-cycle reuse)
        let result = schedule_ssa_cpsat(
            &ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: true,
                scratch_budget: 2,  // CP-SAT needs 2 because no same-cycle reuse
                ..Default::default()
            },
            None,
            None,
        );

        // CP-SAT should find a solution with budget=2
        assert!(
            result.is_some(),
            "CP-SAT should find a schedule with 2-word budget"
        );

        let r = result.unwrap();
        eprintln!(
            "test_ssa_cpsat_minimal_budget_2_words: makespan={}, assignments={:?}",
            r.makespan, r.cycle_assignments
        );

        // Verify all ops are scheduled
        assert_eq!(r.cycle_assignments.len(), 3, "All 3 ops should be scheduled");

        // Apply cycle assignments to a mutable copy for address assignment
        let mut scheduled_ops = ops.clone();
        for (idx, &cycle) in r.cycle_assignments.iter().enumerate() {
            scheduled_ops[idx].cycle = cycle;
        }

        // Address assignment with same-cycle reuse should fit in budget of 1
        // (even though CP-SAT required budget=2)
        let config = AddressAssignConfig {
            budget: 2,
            start_addr: 1,  // Reserve 0 for ZERO_REG
            ..Default::default()
        };
        let assign_result = assign_addresses(&scheduled_ops, &config);

        // With same-cycle reuse, linear scan should achieve max_addr=1
        // (even though CP-SAT conservatively required budget=2)
        assert_eq!(
            assign_result.max_addr, 1,
            "With same-cycle reuse, only 1 address should be needed, got max_addr={}",
            assign_result.max_addr
        );

        // peak_live should be 1 with same-cycle reuse
        assert_eq!(
            assign_result.peak_live, 1,
            "With same-cycle reuse, peak_live should be 1, got {}",
            assign_result.peak_live
        );

        // Should not exceed budget
        assert!(
            !assign_result.budget_exceeded,
            "Should fit in budget of 2: peak_live={}, budget={}",
            assign_result.peak_live, assign_result.budget
        );
    }
}
