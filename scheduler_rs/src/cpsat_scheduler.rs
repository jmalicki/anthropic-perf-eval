//! CP-SAT based scheduler using Google OR-Tools.
//!
//! This module provides an optimal scheduler using constraint programming.
//! It models the scheduling problem with:
//! - Cycle variables for each op with [ASAP, ALAP] bounds
//! - Precedence constraints from dependencies
//! - Cumulative constraints for engine slot limits
//! - Optional allocation feasibility constraints for scratch memory
//!
//! The CP-SAT scheduler can be used as an alternative to the greedy scheduler
//! when optimal solutions are needed, though it may be slower for large problems.
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

/// Schedule operations using CP-SAT solver.
///
/// # Arguments
/// * `ops` - Operations to schedule (must have dependencies computed)
/// * `scratch_budget` - Maximum scratch memory budget (0 = no constraint)
/// * `time_limit_secs` - Maximum solving time in seconds
/// * `greedy_hints` - Optional hints from greedy scheduler (op_idx -> cycle)
/// * `verbose` - Print solver progress
///
/// # Returns
/// * `Some(CpsatResult)` if a solution was found
/// * `None` if no solution found or feature not enabled
#[cfg(feature = "cpsat")]
pub fn schedule_cpsat(
    ops: &[Op],
    scratch_budget: usize,
    time_limit_secs: f64,
    greedy_hints: Option<&[i32]>,
    verbose: bool,
) -> Option<CpsatResult> {
    let start_time = std::time::Instant::now();
    let n = ops.len();

    if n == 0 {
        return Some(CpsatResult {
            cycle_assignments: vec![],
            makespan: 0,
            status: CpsatStatus::Optimal,
            solve_time_secs: 0.0,
            gap: 0.0,
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
    let makespan = model.new_int_var([(critical_path_len as i64, horizon)]);
    for &cv in &cycle_vars {
        model.add_le(cv, LinearExpr::from(makespan) - 1);
    }
    model.minimize(makespan);

    // 10. Configure solver parameters
    let mut params = SatParameters::default();
    params.max_time_in_seconds = Some(time_limit_secs);
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

            Some(CpsatResult {
                cycle_assignments,
                makespan: actual_makespan as usize,
                status,
                solve_time_secs: solve_time,
                gap,
            })
        }
        _ => None,
    }
}

/// Stub implementation when cpsat feature is not enabled
#[cfg(not(feature = "cpsat"))]
pub fn schedule_cpsat(
    _ops: &[Op],
    _scratch_budget: usize,
    _time_limit_secs: f64,
    _greedy_hints: Option<&[i32]>,
    _verbose: bool,
) -> Option<CpsatResult> {
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

    // Try CP-SAT with greedy hints
    let cpsat_result = schedule_cpsat(
        ops,
        scratch_budget,
        cpsat_time_limit,
        Some(&greedy_cycles),
        verbose,
    );

    if let Some(result) = cpsat_result {
        // Apply CP-SAT solution to ops
        for (idx, op) in ops.iter_mut().enumerate() {
            if idx < result.cycle_assignments.len() {
                op.cycle = result.cycle_assignments[idx];
            }
        }

        if verbose {
            eprintln!(
                "CP-SAT result: {} cycles (greedy was {})",
                result.makespan, greedy_result.cycles
            );
        }

        result
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
        let result = schedule_cpsat(&ops, 0, 10.0, None, false);
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.makespan, 0);
        assert_eq!(result.status, CpsatStatus::Optimal);
    }

    #[cfg(feature = "cpsat")]
    #[test]
    fn test_schedule_cpsat_simple_chain() {
        let ops = vec![
            make_op(0, Engine::Alu, vec![]),
            make_op(1, Engine::Alu, vec![0]),
            make_op(2, Engine::Alu, vec![1]),
        ];

        let result = schedule_cpsat(&ops, 0, 10.0, None, false);
        assert!(result.is_some());

        let result = result.unwrap();
        assert!(result.makespan >= 3); // At least 3 cycles due to dependencies
        assert_eq!(result.cycle_assignments.len(), 3);

        // Verify precedence: cycle[1] > cycle[0], cycle[2] > cycle[1]
        assert!(result.cycle_assignments[1] > result.cycle_assignments[0]);
        assert!(result.cycle_assignments[2] > result.cycle_assignments[1]);
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

        let result = schedule_cpsat(&ops, 0, 10.0, None, false);
        assert!(result.is_some());

        let result = result.unwrap();
        // Should be able to schedule in 2 cycles since chains are independent
        assert!(result.makespan <= 2);
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

        let result = schedule_cpsat(&ops, 0, 10.0, None, false);
        assert!(result.is_some());

        let result = result.unwrap();
        // 15 ops with 12 ALU slots = at least 2 cycles
        assert!(result.makespan >= 2);
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

        let result = schedule_cpsat(&ops, 0, 10.0, Some(&hints), false);
        assert!(result.is_some());

        let result = result.unwrap();
        assert_eq!(result.makespan, 3);
        // With hints, should find the optimal solution quickly
        assert!(result.status == CpsatStatus::Optimal);
    }

    #[test]
    fn test_cpsat_available() {
        // This just tests that the function compiles and returns a bool
        let available = cpsat_available();
        // When compiled with --features cpsat, this should be true
        // When compiled without, this should be false
        assert!(available == cfg!(feature = "cpsat"));
    }
}
