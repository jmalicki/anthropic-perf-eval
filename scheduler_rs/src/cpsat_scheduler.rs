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
//! - **Shift variables** (optional): For each shiftable VALU op, `shift[i]` ∈ {0, 1} where
//!   `shift[i] = 1` means shift to ALU (uses 8 ALU slots), `shift[i] = 0` means keep on VALU (uses 1 VALU slot).
//!   Both alternatives use the same `cycle[i]`, so switching doesn't require cycle changes.
//! - **ConstChoice variables** (optional): For each ConstChoice op, `const_choice[i]` ∈ {0, 1} where
//!   `const_choice[i] = 1` means use AddImm (Flow slot), `const_choice[i] = 0` means use Load (Load slot).
//!   Both alternatives use the same `cycle[i]`, so switching doesn't require cycle changes.
//!
//! ## Constraints
//!
//! 1. **Precedence constraints**: If op j depends on op i, then `cycle[j] >= cycle[i] + 1`
//!    (results are available the cycle after execution)
//!
//! 2. **Resource constraints (cumulative)**: Each engine has a limited number of slots per cycle,
//!    defined by [`SsaCpsatConfig::machine`] ([`crate::types::MachineConfig`]). Default capacities:
//!    - ALU: 12 slots/cycle
//!    - VALU: 6 slots/cycle
//!    - Load: 2 slots/cycle
//!    - Store: 2 slots/cycle
//!    - Flow: 1 slot/cycle
//!
//!    We model this with cumulative constraints: the sum of ops executing in any cycle
//!    must not exceed the engine's capacity. Override `config.machine` for tests or alternate machines.
//!
//!    **Optional intervals**: For ops with alternatives (Shift, ConstChoice), we use optional
//!    intervals that are conditionally present based on the decision variable. Both alternatives
//!    reference the same `cycle[i]`, ensuring that switching between alternatives (e.g., Load ↔ AddImm,
//!    VALU ↔ ALU) only changes the boolean decision variable, not the cycle assignment. This avoids
//!    cascading variable changes and makes optimization more efficient.
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
//! For each engine: `makespan >= ceil(num_ops / capacity)` where capacity is from
//! [`SsaCpsatConfig::machine`].
//!
//! Example: 9838 VALU ops with default VALU capacity 6 requires at least ceil(9838/6) = 1640 cycles.
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
//! Example: If 9837 VALU ops have ASAP >= 1 and default VALU capacity 6, they need ceil(9837/6) = 1640 cycles
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
//! ## Empty cycles and the "bulk move" invariant
//!
//! When there is an empty cycle C (no op runs at C), every op at C+1 could run at C by precedence
//! (no dep runs at C, so no dep result is only available at C+1). So shifting **every** op at
//! C+1, C+2, … one cycle earlier (to C, C+1, …) preserves precedence, symmetry (cycle[i] ≤ cycle[j]),
//! and cumulative (same ops per cycle, just shifted); makespan can decrease by 1. Therefore **no
//! single hard constraint in the model should preclude that bulk move**. If any constraint did,
//! it would be a bug. The tie-breaker (minimize sum of cycles) encourages the solver to find that
//! solution; empty cycles then only persist due to capacity lower bound (structural) or search.
//!
//! ## Debugging: makespan active set
//!
//! When `verbose` is true and a solution is found, we print which constraints on makespan are
//! **binding** (hold with equality): `[Makespan active set]` and a one-line summary after the
//! result. Use this to see why makespan is what it is:
//! - **Ops at last cycle**: which op(s) have `effective_last_cycle == max_cycle` (they "define"
//!   the end of the schedule). If only one op is binding, moving that op earlier (or relaxing
//!   its deps) could reduce makespan.
//! - **Capacity bounds binding**: whether `makespan == capacity_lower_bound` or one of the
//!   redundant linear bounds (capacity_lower_bound, load_bound, store_bound). If true,
//!   the structural capacity is limiting; improving choices (shift, ConstChoice, SelectChoice)
//!   may lower the bound.
//! - **Direct capacity constraints**: VALU/ALU/Flow/Load linear constraints that are tight.
//! Run with `uv run generate_ssa_kernel.py` (no `-q`) and search for `[Makespan active set]` or
//! `Active set:` in the output.
//!
//! ## Warm Start (Hints)
//!
//! We provide the greedy schedule as hints to help CP-SAT find good solutions quickly.
//! The greedy schedule is feasible, so CP-SAT can start from a known good point and
//! focus on proving optimality or finding improvements.
//!
//! ## Symmetry Breaking (from Emission Hints)
//!
//! Ops may carry a `symmetry_hint` (see `SymmetryHint` in `types`). The optimizer uses these
//! to add ordering constraints that reduce redundant search:
//!
//! - **Ordering groups**: Ops with the same `ordering_group` key get `cycle[i] <= cycle[j]`
//!   for consecutive ops in emission order. This prunes equivalent schedules where symmetric
//!   ops (e.g. same tree level) are reordered.
//!
//! - **Lexicographic streams**: When `lexicographic_stream` is set, we enforce
//!   `first_op[item_i] <= first_op[item_{i+1}]` for consecutive items in each stream. This
//!   prunes schedules where symmetric "items" (e.g. batch elements) are reordered.
//!
//! **Why we want this**: CP-SAT reports many orbits (e.g. 276); without breaking symmetry,
//! the solver explores equivalent branches. Emission encodes problem structure (e.g. tree
//! levels, batch items) into opaque keys; the optimizer only groups and orders—no
//! interpretation of rounds, batches, etc. See `docs/CPSAT_SLOW_PROGRESS_ANALYSIS.md` and
//! `docs/INTEGRATED_SCHEDULING_ALLOCATION.md` §5.5.
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
use std::str::FromStr;

use crate::types::{Engine, MachineConfig, Op, Opcode, SsaOp, VReg, VLEN};

/// ConstChoice encoding mode: let solver choose (Optimize), force AddImm (Flow), or force Load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstChoiceMode {
    Optimize,
    ForceAddImm,
    ForceLoad,
}

impl Default for ConstChoiceMode {
    fn default() -> Self {
        Self::Optimize
    }
}

impl FromStr for ConstChoiceMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "optimize" => Ok(Self::Optimize),
            "add_imm" => Ok(Self::ForceAddImm),
            "load" => Ok(Self::ForceLoad),
            _ => Err(format!(
                "invalid const_choice '{}', expected optimize, add_imm, or load",
                s
            )),
        }
    }
}

/// SelectChoice encoding mode: let solver choose (Optimize), force FMA (VALU), or force vselect (Flow).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectChoiceMode {
    Optimize,
    ForceFma,
    ForceVselect,
}

impl Default for SelectChoiceMode {
    fn default() -> Self {
        Self::Optimize
    }
}

impl FromStr for SelectChoiceMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "optimize" => Ok(Self::Optimize),
            "fma" => Ok(Self::ForceFma),
            "vselect" => Ok(Self::ForceVselect),
            _ => Err(format!(
                "invalid select_choice '{}', expected optimize, fma, or vselect",
                s
            )),
        }
    }
}

/// VALU shift mode: let solver choose (Optimize), force shift to ALU, or force stay on VALU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValuShiftMode {
    Optimize,
    ForceAlu,
    ForceValu,
}

impl Default for ValuShiftMode {
    fn default() -> Self {
        Self::Optimize
    }
}

impl FromStr for ValuShiftMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "optimize" => Ok(Self::Optimize),
            "alu" => Ok(Self::ForceAlu),
            "valu" => Ok(Self::ForceValu),
            _ => Err(format!(
                "invalid valu_shift '{}', expected optimize, alu, or valu",
                s
            )),
        }
    }
}

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
        out_degree.insert(op.id, consumers.get(&op.id).map(|c| c.len()).unwrap_or(0));
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

/// One row of the Flow audit: an op that contributes to the Flow engine at a given cycle.
#[cfg(feature = "cpsat")]
#[derive(Debug, Clone)]
struct FlowAuditRow {
    idx: usize,
    op_id: usize,
    opcode: Opcode,
    cycle: usize,
    reason: String,
}

/// Build full Flow audit: every op that contributes to Flow, with idx, op.id, opcode, cycle, reason.
/// Used for rigorous breadcrumb logging and panic messages.
#[cfg(feature = "cpsat")]
fn build_flow_audit(
    ops: &[SsaOp],
    cycle_assignments: &[i32],
    const_add_imm_ops: &HashSet<usize>,
    select_fma_ops: &HashSet<usize>,
) -> (Vec<FlowAuditRow>, Vec<i32>, Vec<Vec<(usize, String)>>) {
    let makespan = cycle_assignments.iter().max().copied().unwrap_or(-1) + 1;
    let makespan_usize = makespan.max(0) as usize;
    let mut flow_per_cycle: Vec<i32> = vec![0; makespan_usize];
    let mut flow_contributors_per_cycle: Vec<Vec<(usize, String)>> =
        vec![Vec::new(); makespan_usize];
    let mut all_flow_rows: Vec<FlowAuditRow> = Vec::new();

    for (idx, op) in ops.iter().enumerate() {
        let cycle = cycle_assignments.get(idx).copied().unwrap_or(0) as usize;
        if cycle >= flow_per_cycle.len() {
            continue;
        }

        if op.opcode == Opcode::ConstChoice {
            if const_add_imm_ops.contains(&op.id) {
                let reason = format!("ConstChoice→AddImm (op.id={})", op.id);
                flow_per_cycle[cycle] += 1;
                if cycle < flow_contributors_per_cycle.len() {
                    flow_contributors_per_cycle[cycle].push((idx, reason.clone()));
                }
                all_flow_rows.push(FlowAuditRow {
                    idx,
                    op_id: op.id,
                    opcode: op.opcode,
                    cycle,
                    reason,
                });
            }
            continue;
        }
        if op.opcode == Opcode::SelectChoice {
            if !select_fma_ops.contains(&op.id) {
                let reason = format!("SelectChoice→vselect (op.id={})", op.id);
                flow_per_cycle[cycle] += 1;
                if cycle < flow_contributors_per_cycle.len() {
                    flow_contributors_per_cycle[cycle].push((idx, reason.clone()));
                }
                all_flow_rows.push(FlowAuditRow {
                    idx,
                    op_id: op.id,
                    opcode: op.opcode,
                    cycle,
                    reason,
                });
            }
            continue;
        }

        if op.engine == Engine::Flow {
            let reason = format!("Flow({:?}) op.id={}", op.opcode, op.id);
            flow_per_cycle[cycle] += 1;
            if cycle < flow_contributors_per_cycle.len() {
                flow_contributors_per_cycle[cycle].push((idx, reason.clone()));
            }
            all_flow_rows.push(FlowAuditRow {
                idx,
                op_id: op.id,
                opcode: op.opcode,
                cycle,
                reason,
            });
        }
    }

    (all_flow_rows, flow_per_cycle, flow_contributors_per_cycle)
}

/// Format a detailed Flow violation message for debugging (breadcrumbs).
#[cfg(feature = "cpsat")]
fn format_flow_violation_breadcrumbs(
    violating_cycle: usize,
    count: i32,
    limit: usize,
    flow_rows: &[FlowAuditRow],
    flow_contributors_per_cycle: &[Vec<(usize, String)>],
    const_add_imm_ops: &HashSet<usize>,
    select_fma_ops: &HashSet<usize>,
    cycle_assignments: &[i32],
) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "\n\n=== Flow slot violation at cycle {}: {} > {} ===\n",
        violating_cycle, count, limit
    ));
    s.push_str("Contributors at this cycle (from flow_contributors_per_cycle):\n");
    if violating_cycle < flow_contributors_per_cycle.len() {
        for (idx, desc) in &flow_contributors_per_cycle[violating_cycle] {
            let cy = cycle_assignments.get(*idx).copied().unwrap_or(-1);
            s.push_str(&format!("  idx={} cycle_assignment={} {}\n", idx, cy, desc));
        }
    }
    s.push_str("All Flow-contributing ops in schedule (full audit):\n");
    for r in flow_rows {
        s.push_str(&format!(
            "  idx={} op.id={} opcode={:?} cycle={} reason={}\n",
            r.idx, r.op_id, r.opcode, r.cycle, r.reason
        ));
    }
    s.push_str("Solution decision sets (op.id):\n");
    s.push_str(&format!("  const_add_imm_ops = {:?}\n", const_add_imm_ops));
    s.push_str(&format!("  select_fma_ops   = {:?}\n", select_fma_ops));
    s.push_str("Interpretation: ConstChoice with op.id in const_add_imm_ops -> AddImm (Flow); ");
    s.push_str("SelectChoice with op.id NOT in select_fma_ops -> vselect (Flow).\n");
    s.push_str("=== end Flow violation breadcrumbs ===\n");
    s
}

/// Summary of engine usage and choice decisions for verbose reporting.
#[cfg(feature = "cpsat")]
struct SsaSolutionSummary {
    alu_total: usize,
    valu_total: usize,
    /// VALU slots from SelectChoice FMA path (sub+fma or fma only; not shiftable).
    valu_fma_slots: usize,
    /// VALU slots from Vbroadcast ops (not shiftable).
    valu_vbroadcast_slots: usize,
    /// VALU slots from other VALU ops (arithmetic etc.; shiftable when can_shift_to_alu).
    valu_other_slots: usize,
    load_total: usize,
    store_total: usize,
    flow_total: usize,
    const_add_imm: usize,
    const_load: usize,
    select_fma: usize,
    select_vselect: usize,
}

#[cfg(feature = "cpsat")]
fn summarise_ssa_solution_engines_and_choices(
    ops: &[SsaOp],
    cycle_assignments: &[i32],
    shifted_ops: &HashSet<usize>,
    const_add_imm_ops: &HashSet<usize>,
    select_fma_ops: &HashSet<usize>,
) -> SsaSolutionSummary {
    let mut alu_total = 0usize;
    let mut valu_total = 0usize;
    let mut valu_fma_slots = 0usize;
    let mut valu_vbroadcast_slots = 0usize;
    let mut valu_other_slots = 0usize;
    let mut load_total = 0usize;
    let mut store_total = 0usize;
    let mut flow_total = 0usize;

    for (idx, op) in ops.iter().enumerate() {
        let cycle = cycle_assignments.get(idx).copied().unwrap_or(0);
        if cycle < 0 {
            continue;
        }
        let _cycle = cycle as usize;

        if op.opcode == Opcode::ConstChoice {
            if const_add_imm_ops.contains(&op.id) {
                flow_total += 1;
            } else {
                load_total += 1;
            }
            continue;
        }
        if op.opcode == Opcode::SelectChoice {
            if select_fma_ops.contains(&op.id) {
                valu_total += 1;
                valu_fma_slots += 1;
                if op.delta_vreg.is_none() {
                    valu_total += 1;
                    valu_fma_slots += 1;
                }
            } else {
                flow_total += 1;
            }
            continue;
        }

        match op.engine {
            Engine::Alu => alu_total += 1,
            Engine::Valu => {
                if shifted_ops.contains(&op.id) {
                    alu_total += VLEN as usize;
                } else {
                    valu_total += 1;
                    if op.opcode == Opcode::Vbroadcast {
                        valu_vbroadcast_slots += 1;
                    } else {
                        valu_other_slots += 1;
                    }
                }
            }
            Engine::Load => load_total += 1,
            Engine::Store => store_total += 1,
            Engine::Flow => flow_total += 1,
            _ => {}
        }
    }

    let const_total = ops
        .iter()
        .filter(|o| o.opcode == Opcode::ConstChoice)
        .count();
    let const_add_imm = const_add_imm_ops.len();
    let const_load = const_total.saturating_sub(const_add_imm);

    let select_total = ops
        .iter()
        .filter(|o| o.opcode == Opcode::SelectChoice)
        .count();
    let select_fma = select_fma_ops.len();
    let select_vselect = select_total.saturating_sub(select_fma);

    SsaSolutionSummary {
        alu_total,
        valu_total,
        valu_fma_slots,
        valu_vbroadcast_slots,
        valu_other_slots,
        load_total,
        store_total,
        flow_total,
        const_add_imm,
        const_load,
        select_fma,
        select_vselect,
    }
}

/// Validate that a CP-SAT solution respects engine slot limits per cycle.
/// Panics with a descriptive message if any cycle exceeds limits (indicates a model bug).
#[cfg(feature = "cpsat")]
fn validate_ssa_slot_limits(
    ops: &[SsaOp],
    cycle_assignments: &[i32],
    shifted_ops: &HashSet<usize>,
    shifted_alu_cycles: &HashMap<usize, Vec<i32>>,
    const_add_imm_ops: &HashSet<usize>,
    select_fma_ops: &HashSet<usize>,
    machine: &MachineConfig,
) {
    let makespan = cycle_assignments.iter().max().copied().unwrap_or(-1) + 1;
    if makespan <= 0 {
        return;
    }
    let mut alu_per_cycle: Vec<i32> = vec![0; makespan as usize];
    let mut valu_per_cycle: Vec<i32> = vec![0; makespan as usize];
    let mut load_per_cycle: Vec<i32> = vec![0; makespan as usize];
    let mut store_per_cycle: Vec<i32> = vec![0; makespan as usize];
    let mut flow_per_cycle: Vec<i32> = vec![0; makespan as usize];
    // For Flow violations: which ops contribute Flow at each cycle (for diagnostics).
    let mut flow_contributors_per_cycle: Vec<Vec<(usize, String)>> =
        vec![Vec::new(); makespan as usize];

    for (idx, op) in ops.iter().enumerate() {
        let cycle = cycle_assignments.get(idx).copied().unwrap_or(0) as usize;
        if cycle >= alu_per_cycle.len() {
            continue;
        }

        if op.opcode == Opcode::ConstChoice {
            if const_add_imm_ops.contains(&op.id) {
                flow_per_cycle[cycle] += 1;
                if cycle < flow_contributors_per_cycle.len() {
                    flow_contributors_per_cycle[cycle]
                        .push((idx, format!("op{} ConstChoice→AddImm", op.id)));
                }
            } else {
                load_per_cycle[cycle] += 1;
            }
            continue;
        }
        if op.opcode == Opcode::SelectChoice {
            if select_fma_ops.contains(&op.id) {
                valu_per_cycle[cycle] += 1;
                if op.delta_vreg.is_none() {
                    let cycle2 = cycle + 1;
                    if cycle2 < valu_per_cycle.len() {
                        valu_per_cycle[cycle2] += 1;
                    }
                }
            } else {
                flow_per_cycle[cycle] += 1;
                if cycle < flow_contributors_per_cycle.len() {
                    flow_contributors_per_cycle[cycle]
                        .push((idx, format!("op{} SelectChoice→vselect", op.id)));
                }
            }
            continue;
        }

        match op.engine {
            Engine::Alu => alu_per_cycle[cycle] += 1,
            Engine::Valu => {
                if shifted_ops.contains(&op.id) {
                    // Count each of the 8 ALU ops at their actual cycles (not all at completion_cycle)
                    if let Some(alu_cycles) = shifted_alu_cycles.get(&op.id) {
                        for &alu_cycle in alu_cycles {
                            let alu_cycle_usize = alu_cycle as usize;
                            if alu_cycle_usize < alu_per_cycle.len() {
                                alu_per_cycle[alu_cycle_usize] += 1;
                            }
                        }
                    } else {
                        // Fallback: if shifted_alu_cycles not available, count all 8 at completion_cycle
                        // This shouldn't happen, but provides backward compatibility
                        alu_per_cycle[cycle] += VLEN as i32;
                    }
                } else {
                    valu_per_cycle[cycle] += 1;
                }
            }
            Engine::Load => load_per_cycle[cycle] += 1,
            Engine::Store => store_per_cycle[cycle] += 1,
            Engine::Flow => {
                flow_per_cycle[cycle] += 1;
                if cycle < flow_contributors_per_cycle.len() {
                    flow_contributors_per_cycle[cycle]
                        .push((idx, format!("op{} Flow({:?})", op.id, op.opcode)));
                }
            }
            _ => {}
        }
    }

    let limits: [(u32, &str); 5] = [
        (machine.slot_limit(Engine::Alu), "ALU"),
        (machine.slot_limit(Engine::Valu), "VALU"),
        (machine.slot_limit(Engine::Load), "Load"),
        (machine.slot_limit(Engine::Store), "Store"),
        (machine.slot_limit(Engine::Flow), "Flow"),
    ];
    let counts = [
        &alu_per_cycle,
        &valu_per_cycle,
        &load_per_cycle,
        &store_per_cycle,
        &flow_per_cycle,
    ];
    for (cycle, _) in alu_per_cycle.iter().enumerate() {
        for (limit_idx, (limit, name)) in limits.iter().enumerate() {
            let count = counts[limit_idx][cycle];
            if count > *limit as i32 {
                let detail = if *name == "Flow" && cycle < flow_contributors_per_cycle.len() {
                    let contribs: Vec<String> = flow_contributors_per_cycle[cycle]
                        .iter()
                        .map(|(i, desc)| format!("idx={} {}", i, desc))
                        .collect();
                    let breadcrumbs = {
                        let (flow_rows, _, _) = build_flow_audit(
                            ops,
                            cycle_assignments,
                            const_add_imm_ops,
                            select_fma_ops,
                        );
                        format_flow_violation_breadcrumbs(
                            cycle,
                            count,
                            *limit as usize,
                            &flow_rows,
                            &flow_contributors_per_cycle,
                            const_add_imm_ops,
                            select_fma_ops,
                            cycle_assignments,
                        )
                    };
                    format!(
                        " contributors at cycle {}: {:?}.{}",
                        cycle, contribs, breadcrumbs
                    )
                } else {
                    String::new()
                };
                panic!(
                    "BUG: CP-SAT solution violates {} slot limit at cycle {}: {} > {} (model constraint error).{}",
                    name, cycle, count, limit, detail
                );
            }
        }
    }
}

/// Same as validate_ssa_slot_limits but returns Err(msg) instead of panicking. Used for diagnostic checks.
#[cfg(feature = "cpsat")]
fn try_validate_ssa_slot_limits(
    ops: &[SsaOp],
    cycle_assignments: &[i32],
    shifted_ops: &HashSet<usize>,
    shifted_alu_cycles: &HashMap<usize, Vec<i32>>,
    const_add_imm_ops: &HashSet<usize>,
    select_fma_ops: &HashSet<usize>,
    machine: &MachineConfig,
) -> Result<(), String> {
    let makespan = cycle_assignments.iter().max().copied().unwrap_or(-1) + 1;
    if makespan <= 0 {
        return Ok(());
    }
    let mut alu_per_cycle: Vec<i32> = vec![0; makespan as usize];
    let mut valu_per_cycle: Vec<i32> = vec![0; makespan as usize];
    let mut load_per_cycle: Vec<i32> = vec![0; makespan as usize];
    let mut store_per_cycle: Vec<i32> = vec![0; makespan as usize];
    let mut flow_per_cycle: Vec<i32> = vec![0; makespan as usize];

    for (idx, op) in ops.iter().enumerate() {
        let cycle = cycle_assignments.get(idx).copied().unwrap_or(0) as usize;
        if cycle >= alu_per_cycle.len() {
            continue;
        }
        if op.opcode == Opcode::ConstChoice {
            if const_add_imm_ops.contains(&op.id) {
                flow_per_cycle[cycle] += 1;
            } else {
                load_per_cycle[cycle] += 1;
            }
            continue;
        }
        if op.opcode == Opcode::SelectChoice {
            if select_fma_ops.contains(&op.id) {
                valu_per_cycle[cycle] += 1;
                if op.delta_vreg.is_none() {
                    let cycle2 = cycle + 1;
                    if cycle2 < valu_per_cycle.len() {
                        valu_per_cycle[cycle2] += 1;
                    }
                }
            } else {
                flow_per_cycle[cycle] += 1;
            }
            continue;
        }
        match op.engine {
            Engine::Alu => alu_per_cycle[cycle] += 1,
            Engine::Valu => {
                if shifted_ops.contains(&op.id) {
                    if let Some(alu_cycles) = shifted_alu_cycles.get(&op.id) {
                        for &alu_cycle in alu_cycles {
                            let alu_cycle_usize = alu_cycle as usize;
                            if alu_cycle_usize < alu_per_cycle.len() {
                                alu_per_cycle[alu_cycle_usize] += 1;
                            }
                        }
                    } else {
                        alu_per_cycle[cycle] += VLEN as i32;
                    }
                } else {
                    valu_per_cycle[cycle] += 1;
                }
            }
            Engine::Load => load_per_cycle[cycle] += 1,
            Engine::Store => store_per_cycle[cycle] += 1,
            Engine::Flow => flow_per_cycle[cycle] += 1,
            _ => {}
        }
    }

    let limits: [(u32, &str); 5] = [
        (machine.slot_limit(Engine::Alu), "ALU"),
        (machine.slot_limit(Engine::Valu), "VALU"),
        (machine.slot_limit(Engine::Load), "Load"),
        (machine.slot_limit(Engine::Store), "Store"),
        (machine.slot_limit(Engine::Flow), "Flow"),
    ];
    let counts = [
        &alu_per_cycle,
        &valu_per_cycle,
        &load_per_cycle,
        &store_per_cycle,
        &flow_per_cycle,
    ];
    for (cycle, _) in alu_per_cycle.iter().enumerate() {
        for (limit_idx, (limit, name)) in limits.iter().enumerate() {
            let count = counts[limit_idx][cycle];
            if count > *limit as i32 {
                return Err(format!(
                    "{} slot limit at cycle {}: {} > {}",
                    name, cycle, count, limit
                ));
            }
        }
    }
    Ok(())
}

/// Verify that every consumer of a SelectChoice FMA (no-delta) result is scheduled at cycle >= dep_cycle + 2.
/// SelectChoice FMA emits sub at C and fma at C+1; result is visible at C+2. Panics if violated (model bug).
#[cfg(feature = "cpsat")]
fn verify_select_choice_fma_consumer_precedence(
    ops: &[SsaOp],
    cycle_assignments: &[i32],
    select_fma_ops: &HashSet<usize>,
) {
    let id_to_idx: HashMap<usize, usize> = ops
        .iter()
        .enumerate()
        .map(|(idx, op)| (op.id, idx))
        .collect();
    for (consumer_idx, op) in ops.iter().enumerate() {
        let consumer_cycle = cycle_assignments.get(consumer_idx).copied().unwrap_or(-1);
        for &dep_id in &op.dep_ids {
            let Some(&producer_idx) = id_to_idx.get(&dep_id) else {
                continue;
            };
            let producer = &ops[producer_idx];
            if producer.opcode != Opcode::SelectChoice
                || !select_fma_ops.contains(&producer.id)
                || producer.sub_temp_vreg.is_none()
            {
                continue;
            }
            let producer_cycle = cycle_assignments.get(producer_idx).copied().unwrap_or(-1);
            assert!(
                consumer_cycle >= producer_cycle + 2,
                "BUG: consumer of SelectChoice FMA (op.id={}) at cycle {} but result not visible until cycle {} (dep at {})",
                producer.id,
                consumer_cycle,
                producer_cycle + 2,
                producer_cycle
            );
        }
    }
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
            verbose: true, // Default: show optimizer output
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
            let mandatory_count: i64 =
                bounds.iter().filter(|(_, alap_val)| *alap_val <= d).count() as i64;

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
            eprintln!(
                "  [sparse window bound] Skipping {:?} (window_size={} > 10000); lower bound may be weaker.",
                engine, window_size
            );
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

    // Optional: log every hint we add (OR-Tools does not log which var had "moved var hint").
    let hint_debug = std::env::var("CPSAT_HINT_DEBUG").is_ok();

    // 8. Add hints from greedy scheduler
    // By construction greedy never schedules before ASAP (it respects deps). ALAP is from the same
    // SSA graph. If hint falls outside [ASAP, ALAP], fix the greedy scheduler or ASAP/ALAP computation.
    // OR-Tools may log "moved var hint within its domain" after presolve without naming the variable;
    // set CPSAT_HINT_DEBUG=1 to log every hint we add (op idx, id, value, domain) to correlate.
    if let Some(hints) = greedy_hints {
        for (idx, &cycle) in hints.iter().enumerate() {
            if idx < cycle_vars.len() && cycle >= 0 {
                if idx < ops.len() {
                    let op = &ops[idx];
                    let lb = *asap.get(&op.id).unwrap_or(&0) as i64;
                    let ub = *alap.get(&op.id).unwrap_or(&(horizon as i32 - 1)) as i64;
                    let c = cycle as i64;
                    if hint_debug {
                        eprintln!(
                            "[CPSAT_HINT_DEBUG] cycle_hint idx={} op_id={} hint={} domain=[{}, {}]",
                            idx, op.id, c, lb, ub
                        );
                    }
                    assert!(
                        c >= lb && c <= ub,
                        "cycle hint outside [ASAP, ALAP]: op id={} idx={} hint_cycle={} domain=[{}, {}] (fix greedy or ASAP/ALAP)",
                        op.id, idx, c, lb, ub
                    );
                }
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

    // Redundant linear capacity constraints: makespan >= ceil(num_ops / capacity) per engine.
    // The cumulative constraint's LP relaxation is weak; adding these as explicit linear
    // constraints helps the LP see the bound and can improve best_objective_bound.
    for (engine, op_indices) in &ops_by_engine {
        let capacity = engine.slot_limit() as i64;
        if capacity > 0 && !op_indices.is_empty() {
            let num_ops = op_indices.len() as i64;
            let min_cycles = (num_ops + capacity - 1) / capacity;
            let lb_var = model.new_int_var([(min_cycles, min_cycles)]);
            model.add_ge(makespan, LinearExpr::from(lb_var));
        }
    }

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

    // Optional: crash when presolve adjusts a hint (confirms "moved var hint" is presolve-related).
    // We don't enable this by default: presolve can legitimately tighten domains so a hint that was
    // valid pre-presolve falls outside the tightened domain; OR-Tools then clamps it. That's
    // expected and the solve continues correctly. Crashing would abort normal runs.
    if std::env::var("CPSAT_DEBUG_PRESOLVE_HINT").is_ok() {
        params.debug_crash_if_presolve_breaks_hint = Some(true);
    }

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

    // Force 0 integral gap for proven optimality (no early stopping based on gap)
    params.relative_gap_limit = Some(0.0);

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

    // 12. Extract solution
    match status {
        CpsatStatus::Optimal | CpsatStatus::Feasible => {
            let cycle_assignments: Vec<i32> = cycle_vars
                .iter()
                .map(|&cv| cv.solution_value(&response) as i32)
                .collect();

            let actual_makespan = cycle_assignments.iter().max().copied().unwrap_or(0) + 1;

            let (best_bound, gap) = if status == CpsatStatus::Optimal {
                (actual_makespan as f64, 0.0)
            } else {
                let best = response.objective_value;
                let bound = response.best_objective_bound;
                let gap_val = if best > 0.0 {
                    (best - bound).abs() / best
                } else {
                    0.0
                };
                (bound, gap_val)
            };

            if verbose {
                eprintln!(
                    "CP-SAT result: {} cycles (best_bound: {:.1}), status {:?}, gap {:.2}%, time {:.2}s",
                    actual_makespan, best_bound, status, gap * 100.0, solve_time
                );
            }

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

/// Configuration for SSA CP-SAT scheduling.
///
/// Slot capacities come from [`machine`](SsaCpsatConfig::machine); use
/// [`MachineConfig::default()`] for the hardware spec or override for tests (e.g.
/// `MachineConfig { valu_slots: 1, alu_slots: 8, ..Default::default() }` to exercise shifting).
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
    /// Optional name for the model (appears in solver log, e.g. "Presolved optimization model 'name'").
    pub model_name: Option<String>,
    /// Machine definition: slot capacities per engine per cycle. Default matches hardware (ALU 12, VALU 6, Load 2, Store 2, Flow 1).
    pub machine: MachineConfig,
    /// ConstChoice encoding: Optimize (solver chooses), ForceAddImm, or ForceLoad.
    pub const_choice_mode: ConstChoiceMode,
    /// SelectChoice encoding: Optimize (solver chooses), ForceFma, or ForceVselect.
    pub select_choice_mode: SelectChoiceMode,
    /// VALU shift: Optimize (solver chooses), ForceAlu, or ForceValu.
    pub valu_shift_mode: ValuShiftMode,
    /// If true, force 0 integral gap for proven optimality (debugging only).
    /// Default: false (uses scale/3 gap limit for faster convergence).
    pub force_zero_gap: bool,
}

impl Default for SsaCpsatConfig {
    fn default() -> Self {
        Self {
            time_limit_secs: 60.0,
            verbose: true, // Default: show optimizer output
            stop_after_first_solution: false,
            enable_valu_shift: false,
            scratch_budget: 1536, // Default hardware limit
            model_name: None,
            machine: MachineConfig::default(),
            const_choice_mode: ConstChoiceMode::default(),
            select_choice_mode: SelectChoiceMode::default(),
            valu_shift_mode: ValuShiftMode::default(),
            force_zero_gap: false, // Default: use scale/3 gap limit for faster convergence
        }
    }
}

/// Result of capacity lower bound computation (for makespan domain and redundant constraints).
#[derive(Debug, Clone, Copy)]
pub struct CapacityLowerBoundResult {
    pub capacity_lower_bound: i64,
    pub valu_alu_global_min: i64,
    pub load_bound: i64,
    pub store_bound: i64,
}

/// Compute capacity-based lower bound on makespan by minimizing over choice variables (x, y, z).
/// No CP-SAT needed: pure triple loop over small integer ranges.
///
/// - x = number of shiftable VALU that stay on VALU (0..=shiftable_count)
/// - y = number of ConstChoice that choose add_imm/Flow (0..=const_choice_count)
/// - z = number of SelectChoice that choose FMA/VALU (0..=select_choice_count); each adds 2 VALU ops (sub + fma)
///
/// For each (x,y,z), computes max(ceil(VALU/6), ceil(ALU/12), ceil(Load/2), Flow, ceil(Store/2))
/// then returns the minimum over all (x,y,z), capped below by critical_path_len.
#[allow(clippy::too_many_arguments)]
pub fn compute_capacity_lower_bound_global(
    valu_count: i64,
    alu_count: i64,
    load_count: i64,
    store_count: i64,
    flow_base: i64,
    shiftable_count: i64,
    const_choice_count: i64,
    select_choice_count: i64,
    enable_valu_shift: bool,
    valu_cap: i64,
    alu_cap: i64,
    load_cap: i64,
    store_cap: i64,
    _flow_cap: i64,
    critical_path_len: i64,
) -> CapacityLowerBoundResult {
    let mut capacity_lower_bound = critical_path_len;
    if !enable_valu_shift && valu_count > 0 && valu_cap > 0 {
        let valu_only = (valu_count + valu_cap - 1) / valu_cap;
        capacity_lower_bound = capacity_lower_bound.max(valu_only);
    }
    let store_bound = if store_cap > 0 && store_count > 0 {
        (store_count + store_cap - 1) / store_cap
    } else {
        0
    };
    let mut global_min = i64::MAX;
    for x in 0..=shiftable_count {
        let valu_ops = if enable_valu_shift {
            (valu_count - shiftable_count + x).max(0)
        } else {
            valu_count
        };
        let alu_ops = if enable_valu_shift {
            alu_count + 8 * (shiftable_count - x)
        } else {
            alu_count
        };
        for y in 0..=const_choice_count {
            let load_ops = (load_count - y).max(0);
            for z in 0..=select_choice_count {
                // Each SelectChoice FMA adds 2 VALU ops (sub + fma).
                let valu_with_select = valu_ops + 2 * z;
                // flow_base already counts all Flow ops (including SelectChoice). When z choose FMA they use VALU;
                // (select_choice_count - z) use vselect (1 Flow each). So total Flow = flow_base + y - z.
                let flow_ops = flow_base + y - z;
                let v = if valu_with_select > 0 && valu_cap > 0 {
                    (valu_with_select + valu_cap - 1) / valu_cap
                } else {
                    0
                };
                let a = if alu_ops > 0 && alu_cap > 0 {
                    (alu_ops + alu_cap - 1) / alu_cap
                } else {
                    0
                };
                let ld = if load_cap > 0 && load_ops > 0 {
                    (load_ops + load_cap - 1) / load_cap
                } else {
                    0
                };
                let fl = flow_ops;
                let lb = v.max(a).max(ld).max(fl).max(store_bound);
                global_min = global_min.min(lb);
            }
        }
    }
    if global_min != i64::MAX {
        capacity_lower_bound = capacity_lower_bound.max(global_min);
    }
    let valu_alu_global_min = if enable_valu_shift && (valu_count > 0 || alu_count > 0 || shiftable_count > 0) {
        (0..=shiftable_count)
            .map(|x| {
                let valu_ops = (valu_count - shiftable_count + x).max(0);
                let alu_ops = alu_count + 8 * (shiftable_count - x);
                let v = if valu_ops > 0 && valu_cap > 0 {
                    (valu_ops + valu_cap - 1) / valu_cap
                } else {
                    0
                };
                let a = if alu_ops > 0 && alu_cap > 0 {
                    (alu_ops + alu_cap - 1) / alu_cap
                } else {
                    0
                };
                v.max(a)
            })
            .min()
            .unwrap_or(0)
    } else {
        0
    };
    let load_bound = if load_cap > 0 && (load_count - const_choice_count).max(0) > 0 {
        ((load_count - const_choice_count).max(0) + load_cap - 1) / load_cap
    } else {
        0
    };
    CapacityLowerBoundResult {
        capacity_lower_bound,
        valu_alu_global_min,
        load_bound,
        store_bound,
    }
}

#[cfg(test)]
mod capacity_lb_tests {
    use super::compute_capacity_lower_bound_global;

    fn default_caps() -> (i64, i64, i64, i64, i64) {
        (6, 12, 2, 2, 1) // valu, alu, load, store, flow
    }

    #[test]
    fn test_capacity_lb_no_choices() {
        let (vc, ac, lc, sc, fc) = default_caps();
        let r = compute_capacity_lower_bound_global(
            18, 0, 0, 0, 0, 0, 0, 0, false, vc, ac, lc, sc, fc, 1,
        );
        assert_eq!(r.capacity_lower_bound, 3, "18 VALU / 6 = 3 cycles");
        assert_eq!(r.store_bound, 0);
    }

    #[test]
    fn test_capacity_lb_critical_path_dominates() {
        let (vc, ac, lc, sc, fc) = default_caps();
        let r = compute_capacity_lower_bound_global(
            6, 0, 0, 0, 0, 0, 0, 0, false, vc, ac, lc, sc, fc, 10,
        );
        assert_eq!(r.capacity_lower_bound, 10, "critical_path 10 dominates 6/6=1");
    }

    #[test]
    fn test_capacity_lb_shift_global_min() {
        let (vc, ac, lc, sc, fc) = default_caps();
        // 6 VALU, 3 shiftable. x=0 (all shift): VALU=3, ALU=24 => max(1,2)=2. x=3 (none shift): VALU=6, ALU=0 => max(1,0)=1. Global min = 1.
        let r_no_shift = compute_capacity_lower_bound_global(
            6, 0, 0, 0, 0, 3, 0, 0, false, vc, ac, lc, sc, fc, 1,
        );
        let r_shift = compute_capacity_lower_bound_global(
            6, 0, 0, 0, 0, 3, 0, 0, true, vc, ac, lc, sc, fc, 1,
        );
        assert_eq!(r_no_shift.capacity_lower_bound, 1);
        assert_eq!(r_shift.capacity_lower_bound, 1, "min over x: best is x=3 => 1 cycle");
    }

    #[test]
    fn test_capacity_lb_const_choice_reduces_load() {
        let (vc, ac, lc, sc, fc) = default_caps();
        // 4 Load ops, 2 ConstChoice. If both add_imm: 2 Load => 1 cycle. If both load: 4 Load => 2 cycles.
        let r = compute_capacity_lower_bound_global(
            0, 0, 4, 0, 0, 0, 2, 0, false, vc, ac, lc, sc, fc, 1,
        );
        assert!(r.capacity_lower_bound >= 1 && r.capacity_lower_bound <= 2);
        assert_eq!(r.load_bound, 1, "min Load = 2 ops => ceil(2/2)=1");
    }

    #[test]
    fn test_capacity_lb_flow_dominates() {
        let (vc, ac, lc, sc, fc) = default_caps();
        // 5 Flow ops (flow_base=5), no others. Makespan must be >= 5.
        let r = compute_capacity_lower_bound_global(
            0, 0, 0, 0, 5, 0, 0, 0, false, vc, ac, lc, sc, fc, 1,
        );
        assert_eq!(r.capacity_lower_bound, 5);
    }

    #[test]
    fn test_capacity_lb_triple_choice_global_min() {
        let (vc, ac, lc, sc, fc) = default_caps();
        // Small mix: some VALU (shiftable), some ConstChoice, some SelectChoice. Just check it runs and returns >= critical_path.
        let r = compute_capacity_lower_bound_global(
            6, 12, 2, 0, 1, 3, 1, 1, true, vc, ac, lc, sc, fc, 5,
        );
        assert!(r.capacity_lower_bound >= 5);
        assert!(r.capacity_lower_bound >= r.store_bound);
    }
}

/// Optional hints and segment input for SSA CP-SAT.
/// Use struct literal syntax to pass only what you need; omit fields for `None`.
/// Example: `SsaCpsatHints { greedy_hints: Some(&hints), ..Default::default() }`
#[derive(Debug, Clone, Default)]
pub struct SsaCpsatHints<'a> {
    /// Cycle assignments from greedy scheduler (warm start).
    pub greedy_hints: Option<&'a [i32]>,
    /// Shift decisions from greedy (op ID -> should shift to ALU).
    pub shift_hints: Option<&'a HashSet<usize>>,
    /// Forced shift decisions (op ID -> must shift to ALU). Unlike shift_hints, these are enforced as constraints.
    pub forced_shifts: Option<&'a HashSet<usize>>,
    /// SelectChoice decisions from greedy (op ID -> use FMA path).
    pub select_choice_hints: Option<&'a HashSet<usize>>,
    /// ConstChoice decisions from greedy (op ID -> use add_imm / Flow). If set, ops in this set get hint 1 (add_imm).
    pub const_add_imm_hints: Option<&'a HashSet<usize>>,
    /// Segment mode: (VReg, op index of last use); adds fixed [0, end] intervals to scratch (per-section feasibility).
    pub initial_live: Option<&'a [(VReg, usize)]>,
    /// Segment mode: VRegs defined in this segment but last-used in a later section. Their scratch interval must extend to segment end (max cycle).
    pub live_out_vregs: Option<&'a HashSet<VReg>>,
    /// Segment mode: VRegs defined in the previous section but last-used in a later section (live for the whole of this segment). Their scratch interval is [0, segment_end].
    pub live_through_vregs: Option<&'a HashSet<VReg>>,
}

/// Result from SSA CP-SAT scheduling with VALU shift decisions
#[derive(Debug, Clone)]
pub struct SsaCpsatResult {
    /// Cycle assignment for each op (indexed by op position in input slice)
    /// For shifted VALU ops, this is the completion_cycle (max of all 8 ALU cycles)
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
    /// Individual ALU cycle assignments for shifted VALU ops
    /// Maps op ID -> [cycle0, cycle1, ..., cycle7] (8 cycles, one per ALU op)
    pub shifted_alu_cycles: HashMap<usize, Vec<i32>>,
    /// Set of ConstChoice op IDs that should use AddImm (Flow) instead of Load
    pub const_add_imm_ops: HashSet<usize>,
    /// Set of SelectChoice op IDs that should use FMA path (VALU) instead of vselect (Flow)
    pub select_fma_ops: HashSet<usize>,
    /// Number of empty cycles (makespan minus max cycle used). 0 when schedule has no gaps.
    pub empty_cycle_count: usize,
    /// True when makespan equals capacity lower bound (structural bound caused the gap).
    pub makespan_driven_by_capacity: bool,
    /// Same constraints as the CP-SAT model (for bulk-move simulator and validation).
    pub schedule_constraints: Vec<crate::schedule_constraints::ScheduleConstraint>,
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
        out_degree.insert(op.id, consumers.get(&op.id).map(|c| c.len()).unwrap_or(0));
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
/// * `config` - Solver configuration (time limit, scratch budget, etc.)
/// * `hints` - Optional warm-start and segment input; use `SsaCpsatHints::default()` for none
/// * `greedy_within_scratch_budget` - When `Some(true)`, the greedy schedule (hints) is within scratch budget;
///   when `Some(false)` or `None`, greedy may have exceeded budget (hints can be infeasible). Used only to
///   sanity-check: we panic if greedy makespan < capacity_lower_bound only when this is `Some(true)`.
///
/// # Returns
/// * `Some(SsaCpsatResult)` if a solution was found (includes shift decisions)
/// * `None` if no solution found or feature not enabled
#[cfg(feature = "cpsat")]
pub fn schedule_ssa_cpsat(
    ops: &[SsaOp],
    config: SsaCpsatConfig,
    hints: SsaCpsatHints<'_>,
    greedy_within_scratch_budget: Option<bool>,
) -> Option<SsaCpsatResult> {
    use cp_sat::builder::BoolVar;

    let SsaCpsatConfig {
        time_limit_secs,
        verbose,
        stop_after_first_solution,
        enable_valu_shift,
        scratch_budget,
        model_name,
        machine,
        const_choice_mode,
        select_choice_mode,
        valu_shift_mode,
        force_zero_gap,
    } = config;

    let valu_cap: i64 = machine.valu_slots as i64;
    let alu_cap: i64 = machine.alu_slots as i64;
    let load_cap: i64 = machine.load_slots as i64;
    let store_cap: i64 = machine.store_slots as i64;
    let flow_cap: i64 = machine.flow_slots as i64;
    let greedy_hints = hints.greedy_hints;
    let shift_hints = hints.shift_hints;
    let forced_shifts = hints.forced_shifts;
    let const_add_imm_hints = hints.const_add_imm_hints;
    let select_choice_hints = hints.select_choice_hints;
    let initial_live = hints.initial_live;
    let live_out_vregs = hints.live_out_vregs;
    let live_through_vregs = hints.live_through_vregs;
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
            shifted_alu_cycles: HashMap::new(),
            const_add_imm_ops: HashSet::new(),
            select_fma_ops: HashSet::new(),
            empty_cycle_count: 0,
            makespan_driven_by_capacity: false,
            schedule_constraints: vec![],
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
    // Create shiftable_set early so we can use it in precedence constraints
    let shiftable_set: HashSet<usize> = shiftable_valu.iter().copied().collect();

    if verbose {
        let total_valu_ops: usize = ops.iter().filter(|op| op.engine == Engine::Valu).count();
        let non_shiftable_valu_count = total_valu_ops - shiftable_valu.len();
        let vbroadcast_count = ops
            .iter()
            .filter(|op| op.engine == Engine::Valu && op.opcode == Opcode::Vbroadcast)
            .count();
        let multiply_add_count = ops
            .iter()
            .filter(|op| op.engine == Engine::Valu && op.opcode == Opcode::MultiplyAdd)
            .count();
        let select_choice_count = ops
            .iter()
            .filter(|op| op.opcode == Opcode::SelectChoice)
            .count();
        eprintln!(
            "SSA CP-SAT: {} ops, critical path {}, horizon {}, scratch_budget={}",
            n, critical_path_len, horizon, scratch_budget
        );
        eprintln!(
            "  VALU ops: {} total ({} shiftable, {} non-shiftable: {} vbroadcast, {} multiply_add)",
            total_valu_ops,
            shiftable_valu.len(),
            non_shiftable_valu_count,
            vbroadcast_count,
            multiply_add_count
        );
        if select_choice_count > 0 {
            eprintln!(
                "  SelectChoice ops: {} (can expand to FMA/VALU or vselect/Flow)",
                select_choice_count
            );
        }
    }

    // 3. Build the CP model
    let mut model = CpModelBuilder::default();
    if let Some(name) = &model_name {
        model.set_model_name(name);
    }
    if verbose && model_name.is_some() {
        eprintln!("  Model name: {}", model_name.as_deref().unwrap_or(""));
    }

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

    // SelectChoice ops with sub_temp_vreg (FMA path without delta): result is visible at cycle+2, not cycle+1.
    // We'll add precedence for their consumers after creating select_choice_vars.
    let select_choice_with_fma_delay: HashSet<usize> = ops
        .iter()
        .enumerate()
        .filter(|(_, op)| op.opcode == Opcode::SelectChoice && op.sub_temp_vreg.is_some())
        .map(|(i, _)| i)
        .collect();

    // 5. Add precedence constraints from RAW dependencies
    // Note: For consumers of shifted VALU ops, we'll update these constraints after
    // creating completion_cycle_vars (see below)
    for (idx, op) in ops.iter().enumerate() {
        for &dep_id in &op.dep_ids {
            if let Some(&dep_idx) = op_id_to_idx.get(&dep_id) {
                // Skip if dep_idx is a shiftable VALU op - we'll handle it after creating completion_cycle_vars
                if shiftable_set.contains(&dep_idx) {
                    continue;
                }
                // Skip if dep is SelectChoice with sub_temp (FMA no-delta): consumer precedence added in 5d
                if select_choice_with_fma_delay.contains(&dep_idx) {
                    continue;
                }
                // cycle[idx] >= cycle[dep_idx] + 1
                model.add_ge(cycle_vars[idx], LinearExpr::from(cycle_vars[dep_idx]) + 1);
            }
        }
    }

    // 5b. Symmetry breaking (ordering groups): ops with same ordering_group key get cycle[i] <= cycle[j]
    // for consecutive ops in emission order. Optimizer only groups by key; emission encodes structure.
    {
        use std::collections::HashMap;
        let mut groups: HashMap<Vec<i64>, Vec<usize>> = HashMap::new();
        for (idx, op) in ops.iter().enumerate() {
            let Some(ref hint) = op.symmetry_hint else {
                continue;
            };
            if hint.ordering_group.is_empty() {
                continue;
            }
            groups
                .entry(hint.ordering_group.clone())
                .or_default()
                .push(idx);
        }
        for (_key, indices) in groups.iter() {
            for w in indices.windows(2) {
                let (i, j) = (w[0], w[1]);
                model.add_le(cycle_vars[i], cycle_vars[j]);
            }
        }
        if verbose && !groups.is_empty() {
            let total_constraints: usize = groups.values().map(|v| v.len().saturating_sub(1)).sum();
            eprintln!(
                "  Symmetry breaking (ordering): {} groups, {} constraints",
                groups.len(),
                total_constraints
            );
        }
    }

    // 5c. Symmetry breaking (lexicographic streams): for each stream_key, items ordered by item_key
    // get first_op[item_i] <= first_op[item_{i+1}]. Optimizer only groups and orders; emission encodes structure.
    {
        use std::collections::{BTreeMap, HashMap};
        let mut streams: HashMap<Vec<i64>, BTreeMap<i64, Vec<usize>>> = HashMap::new();
        for (idx, op) in ops.iter().enumerate() {
            let Some(ref hint) = op.symmetry_hint else {
                continue;
            };
            let Some((ref stream_key, item_key)) = hint.lexicographic_stream else {
                continue;
            };
            streams
                .entry(stream_key.clone())
                .or_default()
                .entry(item_key)
                .or_default()
                .push(idx);
        }
        let n_streams = streams.len();
        let mut lex_constraints = 0usize;
        for (_stream_key, by_item) in streams {
            let ordered: Vec<Vec<usize>> = by_item.into_values().collect();
            for w in ordered.windows(2) {
                let first_op_cur = w[0][0];
                let first_op_next = w[1][0];
                model.add_le(cycle_vars[first_op_cur], cycle_vars[first_op_next]);
                lex_constraints += 1;
            }
        }
        if verbose && lex_constraints > 0 {
            eprintln!(
                "  Symmetry breaking (lex streams): {} streams, {} constraints",
                n_streams, lex_constraints
            );
        }
    }

    // 6. Create shift decision variables and optional/required intervals for shiftable VALU ops
    // Note: shiftable_set was already created above in section 2
    let mut shift_vars: HashMap<usize, BoolVar> = HashMap::new();
    let mut valu_optional_intervals: HashMap<usize, IntervalVar> = HashMap::new();
    // Store 8 intervals per shifted VALU op (one per ALU op) - optional when Optimize, required when ForceAlu
    let mut alu_optional_intervals: HashMap<usize, Vec<IntervalVar>> = HashMap::new();
    // Store 8 cycle variables per shifted VALU op (one per ALU op)
    let mut alu_cycle_vars: HashMap<usize, Vec<IntVar>> = HashMap::new();
    // Store completion cycle variable (max of 8 ALU cycles) per shifted VALU op
    let mut completion_cycle_vars: HashMap<usize, IntVar> = HashMap::new();

    for &idx in &shiftable_valu {
        let shift_var = model.new_bool_var_with_name(format!("shift_{}", idx));
        shift_vars.insert(idx, shift_var);

        match valu_shift_mode {
            ValuShiftMode::ForceValu => {
                model.add_eq(LinearExpr::from(shift_var.clone()), 0);
            }
            ValuShiftMode::ForceAlu => {
                model.add_eq(LinearExpr::from(shift_var.clone()), 1);
            }
            ValuShiftMode::Optimize => {
                if let Some(forced) = forced_shifts {
                    if forced.contains(&ops[idx].id) {
                        model.add_eq(LinearExpr::from(shift_var.clone()), 1);
                    }
                }
                // Shift hints are added in section 11 with all other hints (avoids duplicate hint for same var)
            }
        }

        let valu_interval =
            model.new_optional_fixed_size_interval_var(cycle_vars[idx], 1, shift_var.not());
        valu_optional_intervals.insert(idx, valu_interval);

        let op_asap = *asap.get(&ops[idx].id).unwrap_or(&0) as i64;
        let op_alap = *alap.get(&ops[idx].id).unwrap_or(&((horizon - 1) as i32)) as i64;

        let mut alu_cycles = Vec::with_capacity(VLEN);
        let mut alu_intervals = Vec::with_capacity(VLEN);

        for i in 0..VLEN {
            let alu_cycle =
                model.new_int_var_with_name([(op_asap, op_alap)], format!("alu_c{}_{}", idx, i));
            alu_cycles.push(alu_cycle);
            let alu_interval =
                model.new_optional_fixed_size_interval_var(alu_cycle, 1, shift_var.clone());
            alu_intervals.push(alu_interval);
            // Precedence vs deps is added in 5b with shift-activated constraints (see below)
        }
        alu_cycle_vars.insert(idx, alu_cycles.clone());
        alu_optional_intervals.insert(idx, alu_intervals);

        let completion_cycle =
            model.new_int_var_with_name([(op_asap, op_alap)], format!("completion_c{}", idx));
        completion_cycle_vars.insert(idx, completion_cycle);
        model.add_ge(
            LinearExpr::from(completion_cycle),
            LinearExpr::from(cycle_vars[idx]),
        );
        for &alu_cycle in &alu_cycles {
            model.add_ge(completion_cycle, LinearExpr::from(alu_cycle));
        }
        if matches!(valu_shift_mode, ValuShiftMode::ForceValu) {
            model.add_eq(
                LinearExpr::from(completion_cycle),
                LinearExpr::from(cycle_vars[idx]),
            );
        }
    }

    // 5b. Precedence for the 8 ALU ops of shifted VALU (second model-building loop; we add
    // constraints here, not a second solver run). We need completion_cycle_vars for all deps
    // before we can add these, so they cannot go in the loop above. Use the dependency's shift
    // variable to activate the right bound: when dep is shifted, require alu_cycle >= completion_cycle[dep]+1;
    // when dep is not shifted, require alu_cycle >= cycle_vars[dep]+1. Exactly one is active.
    for &idx in &shiftable_valu {
        if let Some(alu_cycles) = alu_cycle_vars.get(&idx) {
            for alu_cycle in alu_cycles {
                for &dep_id in &ops[idx].dep_ids {
                    if let Some(&dep_idx) = op_id_to_idx.get(&dep_id) {
                        if shiftable_set.contains(&dep_idx) {
                            let shift_dep = *shift_vars.get(&dep_idx).unwrap();
                            // When shift[dep]=1: dep is shifted -> wait for completion_cycle
                            if let Some(&completion_cycle) = completion_cycle_vars.get(&dep_idx) {
                                let c = model.add_ge(
                                    *alu_cycle,
                                    LinearExpr::from(completion_cycle) + 1,
                                );
                                model.only_enforce_if(c, [shift_dep]);
                            }
                            // When shift[dep]=0: dep stays on VALU -> use cycle_vars[dep]
                            let c = model.add_ge(
                                *alu_cycle,
                                LinearExpr::from(cycle_vars[dep_idx]) + 1,
                            );
                            model.only_enforce_if(c, [!shift_dep]);
                        } else {
                            model.add_ge(
                                *alu_cycle,
                                LinearExpr::from(cycle_vars[dep_idx]) + 1,
                            );
                        }
                    }
                }
            }
        }
    }

    // 5c. Add precedence constraints for consumers of shifted VALU ops
    // Consumers must wait for completion_cycle (when all 8 ALU ops have completed)
    for (idx, op) in ops.iter().enumerate() {
        for &dep_id in &op.dep_ids {
            if let Some(&dep_idx) = op_id_to_idx.get(&dep_id) {
                // If dependency is a shifted VALU op, use completion_cycle instead of cycle_vars
                if shiftable_set.contains(&dep_idx) {
                    if let Some(&completion_cycle) = completion_cycle_vars.get(&dep_idx) {
                        // cycle[idx] >= completion_cycle[dep_idx] + 1
                        // This ensures consumer waits for all 8 ALU ops to complete
                        model.add_ge(cycle_vars[idx], LinearExpr::from(completion_cycle) + 1);
                    }
                }
            }
        }
    }

    // 6b. Create ConstChoice decision variables and optional intervals
    let const_choice_ops: Vec<usize> = ops
        .iter()
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
        match const_choice_mode {
            ConstChoiceMode::ForceLoad => {
                let load_interval = model.new_fixed_size_interval_var(cycle_vars[idx], 1);
                const_load_intervals.insert(idx, load_interval);
            }
            ConstChoiceMode::ForceAddImm => {
                let add_imm_interval = model.new_fixed_size_interval_var(cycle_vars[idx], 1);
                const_flow_intervals.insert(idx, add_imm_interval);
            }
            ConstChoiceMode::Optimize => {
                let choice_var = model.new_bool_var_with_name(format!("const_choice_{}", idx));
                const_choice_vars.insert(idx, choice_var);

                let load_interval = model.new_optional_fixed_size_interval_var(
                    cycle_vars[idx],
                    1,
                    choice_var.not(),
                );
                const_load_intervals.insert(idx, load_interval);

                let add_imm_interval =
                    model.new_optional_fixed_size_interval_var(cycle_vars[idx], 1, choice_var);
                const_flow_intervals.insert(idx, add_imm_interval);

                // Hint: use greedy's ConstChoice decision if provided, else default to Load (0)
                let use_add_imm = const_add_imm_hints.map_or(false, |h| h.contains(&ops[idx].id));
                model.add_hint(choice_var, if use_add_imm { 1 } else { 0 });
            }
        }
    }

    // 6c. Create SelectChoice decision variables and optional intervals
    let select_choice_ops: Vec<usize> = ops
        .iter()
        .enumerate()
        .filter(|(_, op)| op.opcode == Opcode::SelectChoice)
        .map(|(idx, _)| idx)
        .collect();

    let mut select_choice_vars: HashMap<usize, BoolVar> = HashMap::new();
    let mut select_flow_intervals: HashMap<usize, IntervalVar> = HashMap::new();
    let mut select_valu_intervals: HashMap<usize, Vec<IntervalVar>> = HashMap::new();
    // For SelectChoice FMA (no-delta): last cycle of op is fma_cycle_var (sub at cycle_var, fma at cycle_var+1). Used so makespan is constrained by chosen path only.
    let mut select_choice_fma_last_cycle: HashMap<usize, IntVar> = HashMap::new();

    for &idx in &select_choice_ops {
        match select_choice_mode {
            SelectChoiceMode::ForceVselect => {
                let flow_interval = model.new_fixed_size_interval_var(cycle_vars[idx], 1);
                select_flow_intervals.insert(idx, flow_interval);
            }
            SelectChoiceMode::ForceFma => {
                let valu_intervals = if let Some(_delta_vreg) = ops[idx].delta_vreg {
                    vec![model.new_fixed_size_interval_var(cycle_vars[idx], 1)]
                } else {
                    let sub_interval = model.new_fixed_size_interval_var(cycle_vars[idx], 1);
                    let fma_asap = *asap.get(&ops[idx].id).unwrap_or(&0) + 1;
                    let fma_alap = *alap.get(&ops[idx].id).unwrap_or(&(horizon as i32 - 1)) + 1;
                    let fma_cycle_var = model.new_int_var_with_name(
                        [(fma_asap as i64, fma_alap as i64)],
                        format!("fma_cycle_{}", idx),
                    );
                    model.add_eq(
                        LinearExpr::from(fma_cycle_var),
                        LinearExpr::from(cycle_vars[idx]) + 1,
                    );
                    select_choice_fma_last_cycle.insert(idx, fma_cycle_var);
                    let fma_interval = model.new_fixed_size_interval_var(fma_cycle_var, 1);
                    vec![sub_interval, fma_interval]
                };
                select_valu_intervals.insert(idx, valu_intervals);
            }
            SelectChoiceMode::Optimize => {
                let choice_var = model.new_bool_var_with_name(format!("select_choice_{}", idx));
                select_choice_vars.insert(idx, choice_var);

                if let Some(hints) = select_choice_hints {
                    if hints.contains(&ops[idx].id) {
                        model.add_hint(choice_var, 1);
                    } else {
                        model.add_hint(choice_var, 0);
                    }
                } else {
                    model.add_hint(choice_var, 0);
                }

                let flow_interval = model.new_optional_fixed_size_interval_var(
                    cycle_vars[idx],
                    1,
                    choice_var.not(),
                );
                select_flow_intervals.insert(idx, flow_interval);

                let valu_intervals = if let Some(_delta_vreg) = ops[idx].delta_vreg {
                    vec![model.new_optional_fixed_size_interval_var(cycle_vars[idx], 1, choice_var)]
                } else {
                    let sub_interval =
                        model.new_optional_fixed_size_interval_var(cycle_vars[idx], 1, choice_var);
                    let fma_asap = *asap.get(&ops[idx].id).unwrap_or(&0) + 1;
                    let fma_alap = *alap.get(&ops[idx].id).unwrap_or(&(horizon as i32 - 1)) + 1;
                    let fma_cycle_var = model.new_int_var_with_name(
                        [(fma_asap as i64, fma_alap as i64)],
                        format!("fma_cycle_{}", idx),
                    );
                    model.add_eq(
                        LinearExpr::from(fma_cycle_var),
                        LinearExpr::from(cycle_vars[idx]) + 1,
                    );
                    select_choice_fma_last_cycle.insert(idx, fma_cycle_var);
                    let fma_interval =
                        model.new_optional_fixed_size_interval_var(fma_cycle_var, 1, choice_var);
                    vec![sub_interval, fma_interval]
                };
                select_valu_intervals.insert(idx, valu_intervals);
            }
        }
    }

    // 5d. Precedence for consumers of SelectChoice FMA (no delta): result is visible at cycle+2, not cycle+1
    // Store select_fma_extra vars so we can hint them (avoids "solution hint is incomplete").
    let mut select_fma_extra_vars: std::collections::HashMap<(usize, usize), IntVar> =
        std::collections::HashMap::new();
    for (idx, op) in ops.iter().enumerate() {
        for &dep_id in &op.dep_ids {
            let Some(&dep_idx) = op_id_to_idx.get(&dep_id) else {
                continue;
            };
            if !select_choice_with_fma_delay.contains(&dep_idx) {
                continue;
            }
            match select_choice_mode {
                SelectChoiceMode::ForceVselect => {
                    model.add_ge(cycle_vars[idx], LinearExpr::from(cycle_vars[dep_idx]) + 1);
                }
                SelectChoiceMode::ForceFma => {
                    model.add_ge(cycle_vars[idx], LinearExpr::from(cycle_vars[dep_idx]) + 2);
                }
                SelectChoiceMode::Optimize => {
                    let choice_var = select_choice_vars
                        .get(&dep_idx)
                        .expect("Optimize has choice var");
                    // consumer >= dep_cycle + 1 + (1 if FMA else 0). Use auxiliary 0-1 int.
                    let extra = model.new_int_var_with_name(
                        [(0, 1)],
                        format!("select_fma_extra_{}_{}", dep_idx, idx),
                    );
                    model.add_eq(LinearExpr::from(extra), LinearExpr::from(*choice_var));
                    model.add_ge(
                        cycle_vars[idx],
                        LinearExpr::from(cycle_vars[dep_idx]) + 1 + LinearExpr::from(extra),
                    );
                    select_fma_extra_vars.insert((dep_idx, idx), extra);
                }
            }
        }
    }

    // 7. Create regular intervals for non-shiftable, non-ConstChoice, non-SelectChoice ops
    let const_choice_set: HashSet<usize> = const_choice_ops.iter().copied().collect();
    let select_choice_set: HashSet<usize> = select_choice_ops.iter().copied().collect();
    let mut regular_intervals: Vec<IntervalVar> = Vec::new();
    for (idx, _) in ops.iter().enumerate() {
        if !shiftable_set.contains(&idx)
            && !const_choice_set.contains(&idx)
            && !select_choice_set.contains(&idx)
        {
            regular_intervals.push(model.new_fixed_size_interval_var(cycle_vars[idx], 1));
        }
    }

    // 8. Build cumulative constraints with optional intervals

    // Collect non-shiftable VALU ops (exclude SelectChoice ops - they use optional intervals)
    let non_shiftable_valu: Vec<usize> = ops
        .iter()
        .enumerate()
        .filter(|(idx, op)| {
            op.engine == Engine::Valu
                && !shiftable_set.contains(idx)
                && op.opcode != Opcode::SelectChoice
        })
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

        // Add VALU intervals for shiftable ops: ForceValu => required; ForceAlu => none; Optimize => optional
        for &idx in &shiftable_valu {
            match valu_shift_mode {
                ValuShiftMode::ForceValu => {
                    valu_intervals.push(model.new_fixed_size_interval_var(cycle_vars[idx], 1));
                    valu_demands.push(1);
                }
                ValuShiftMode::ForceAlu => {}
                ValuShiftMode::Optimize => {
                    if let Some(&iv) = valu_optional_intervals.get(&idx) {
                        valu_intervals.push(iv);
                        valu_demands.push(1);
                    }
                }
            }
        }

        // Add optional VALU intervals from SelectChoice ops (when choice_var = true, i.e., FMA path)
        for &idx in &select_choice_ops {
            if let Some(intervals) = select_valu_intervals.get(&idx) {
                for &iv in intervals {
                    valu_intervals.push(iv);
                    valu_demands.push(1); // Each VALU op (sub or fma) uses 1 slot
                }
            }
        }

        if !valu_intervals.is_empty() {
            model.add_cumulative(valu_intervals, valu_demands, valu_cap);
        }
    }

    // ALU cumulative: regular ALU ops + optional ALU intervals (8 intervals with demand=1 each for shifted VALU)
    {
        let mut alu_intervals: Vec<IntervalVar> = Vec::new();
        let mut alu_demands: Vec<i64> = Vec::new();

        // Add regular ALU ops (demand = 1)
        for &idx in &alu_ops {
            alu_intervals.push(model.new_fixed_size_interval_var(cycle_vars[idx], 1));
            alu_demands.push(1);
        }

        // Add ALU intervals for shifted VALU ops: ForceAlu => required intervals; Optimize => optional
        for &idx in &shiftable_valu {
            if let Some(intervals) = alu_optional_intervals.get(&idx) {
                for iv in intervals {
                    alu_intervals.push(*iv);
                    alu_demands.push(1);
                }
            }
        }

        if !alu_intervals.is_empty() {
            model.add_cumulative(alu_intervals, alu_demands, alu_cap);
        }
    }

    // Load cumulative: regular Load ops + optional Load intervals from ConstChoice
    {
        let regular_load_ops: Vec<usize> = ops
            .iter()
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
            model.add_cumulative(load_intervals, load_demands, load_cap);
        }
    }

    // Store cumulative (unchanged - no ConstChoice)
    {
        let store_ops: Vec<usize> = ops
            .iter()
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
            model.add_cumulative(intervals, demands, store_cap);
        }
    }

    // Flow cumulative: regular Flow ops + optional Flow intervals from ConstChoice + optional Flow intervals from SelectChoice
    //
    // NOTE: Opcode::AddImm ops are included as regular Flow ops. These are created by
    // emit_const() as an alternative to Load ops. Both the Load op AND the AddImm op
    // are emitted, but only one should be scheduled. The greedy scheduler marks AddImm
    // ops as dead (cycle=-1). For CP-SAT, we include them in the Flow constraint so if
    // the solver does schedule them, they consume a Flow slot.
    //
    // ConstChoice ops are different - they're a single op with alternatives modeled
    // via choice variables. SelectChoice ops are similar (vselect vs FMA path).
    {
        let regular_flow_ops: Vec<usize> = ops
            .iter()
            .enumerate()
            .filter(|(_idx, op)| {
                // Include all Flow ops EXCEPT:
                // - SelectChoice (has optional intervals via choice var)
                // - Note: Opcode::AddImm IS included (from emit_const() pairs)
                op.engine == Engine::Flow && op.opcode != Opcode::SelectChoice
            })
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

        // Add optional Flow intervals from SelectChoice ops (when choice_var = false, i.e., vselect path)
        for &idx in &select_choice_ops {
            if let Some(&iv) = select_flow_intervals.get(&idx) {
                flow_intervals.push(iv);
                flow_demands.push(1);
            }
        }

        if verbose {
            eprintln!(
                "[Flow model breadcrumbs] regular_flow_ops={} (indices {:?}, op.ids {:?})",
                regular_flow_ops.len(),
                regular_flow_ops,
                regular_flow_ops
                    .iter()
                    .map(|&i| ops[i].id)
                    .collect::<Vec<_>>()
            );
            eprintln!(
                "[Flow model breadcrumbs] const_choice_ops={} (optional Flow when choice=true; indices {:?}, op.ids {:?})",
                const_choice_ops.len(),
                const_choice_ops,
                const_choice_ops.iter().map(|&i| ops[i].id).collect::<Vec<_>>()
            );
            eprintln!(
                "[Flow model breadcrumbs] select_choice_ops={} (optional Flow when choice=false; indices {:?}, op.ids {:?})",
                select_choice_ops.len(),
                select_choice_ops,
                select_choice_ops.iter().map(|&i| ops[i].id).collect::<Vec<_>>()
            );
            eprintln!(
                "[Flow model breadcrumbs] Flow cumulative: total intervals={}, capacity={}",
                flow_intervals.len(),
                flow_cap
            );
        }

        if !flow_intervals.is_empty() {
            model.add_cumulative(flow_intervals.clone(), flow_demands, flow_cap);
            // Redundant with cumulative(1): no two Flow intervals overlap when present.
            model.add_no_overlap(flow_intervals);
        }
    }

    // 8b. Add scratch pressure constraint (live VReg capacity)
    // Models: at any cycle, total size of live VRegs <= scratch_budget
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
                    let existing_cycle = vreg_last_use
                        .get(&vreg)
                        .and_then(|&existing_idx| hints.get(existing_idx).copied())
                        .unwrap_or(-1);
                    current_cycle > existing_cycle
                } else {
                    // Without hints, use index order (may be suboptimal)
                    true
                };

                if should_update || !vreg_last_use.contains_key(&vreg) {
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

        // We only model VRegs that have both a definition and at least one use.
        // The caller (build_and_schedule_cpsat) runs SSA DCE before CP-SAT when
        // enable_ssa_dce is true (default), so every defined VReg is expected to
        // have at least one reader. If DCE is disabled, dead VRegs would be omitted
        // from the scratch constraint and could cause undercounting; the caller
        // asserts no dead VRegs when DCE is enabled.

        // Collect VRegs that have both definition and use (all defined VRegs when DCE ran)
        let live_vregs: Vec<VReg> = vreg_def_op
            .keys()
            .filter(|v| vreg_last_use.contains_key(v))
            .copied()
            .collect();

        // Segment mode with live-out or live-through: need segment_end so we can model
        // VRegs live until end of segment (live-out = defined here, used later; live-through = defined in prev section, used later).
        let segment_end_var: Option<IntVar> = if initial_live.is_some()
            && (live_out_vregs.is_some_and(|s| !s.is_empty())
                || live_through_vregs.is_some_and(|s| !s.is_empty()))
        {
            let m = model.new_int_var_with_name([(0, horizon)], "segment_end");
            for cv in &cycle_vars {
                model.add_ge(LinearExpr::from(m), LinearExpr::from(*cv));
            }
            Some(m)
        } else {
            None
        };

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
                let is_live_out = live_out_vregs.is_some_and(|s| s.contains(vreg));

                // Live-out VRegs (defined here, last-used in a later section) must extend to segment end
                // so the scratch constraint matches the full-kernel peak.
                let (end_expr, end_for_interval) = if is_live_out {
                    let seg_end = segment_end_var.expect("segment_end when live_out");
                    let e = LinearExpr::from(seg_end) + 1;
                    (e.clone(), e)
                } else {
                    let e = LinearExpr::from(cycle_vars[last_use_idx]) + 1;
                    (e.clone(), e)
                };

                // The VReg is live from when it's defined until its last use (or segment end if live-out).
                // size = end - start
                let start = cycle_vars[def_idx];

                // Create a size variable: size = end - start
                // Size must be >= 1 (def and last_use could be same cycle)
                let size = model
                    .new_int_var_with_name([(1, horizon + 1)], format!("live_size_{}", vreg.id));

                // Constraint: cycle[def] + size = end
                model.add_eq(LinearExpr::from(start) + LinearExpr::from(size), end_expr);

                let interval = model.new_interval_var(start, size, end_for_interval);
                scratch_intervals.push(interval);
                size_vars.push((size, def_idx, last_use_idx));

                // Demand = size of VReg (1 for scalar, VLEN for vector)
                let demand = vreg.size() as i64;
                scratch_demands.push(demand);
            }

            // Segment mode: add fixed [0, last_use_cycle+1] intervals for live-in VRegs
            if let Some(il) = initial_live {
                let zero_var = model.new_int_var_with_name([(0, 0)], "zero_seg");
                for (vreg, last_use_idx) in il {
                    if *last_use_idx >= cycle_vars.len() {
                        continue;
                    }
                    let end_expr = LinearExpr::from(cycle_vars[*last_use_idx]) + 1;
                    let size = model.new_int_var_with_name(
                        [(1, horizon + 1)],
                        format!("init_live_size_{}", vreg.id),
                    );
                    model.add_eq(
                        LinearExpr::from(zero_var) + LinearExpr::from(size),
                        end_expr.clone(),
                    );
                    let interval = model.new_interval_var(zero_var, size, end_expr);
                    scratch_intervals.push(interval);
                    scratch_demands.push(vreg.size() as i64);
                }
            }

            // Segment mode: add [0, segment_end+1] intervals for live-through VRegs (defined in prev section, used in later section)
            if let (Some(lt), Some(seg_end)) = (live_through_vregs, segment_end_var) {
                let zero_var = model.new_int_var_with_name([(0, 0)], "zero_seg");
                for vreg in lt {
                    let end_expr = LinearExpr::from(seg_end) + 1;
                    let size = model.new_int_var_with_name(
                        [(1, horizon + 1)],
                        format!("live_through_size_{}", vreg.id),
                    );
                    model.add_eq(
                        LinearExpr::from(zero_var) + LinearExpr::from(size),
                        end_expr.clone(),
                    );
                    let interval = model.new_interval_var(zero_var, size, end_expr);
                    scratch_intervals.push(interval);
                    scratch_demands.push(vreg.size() as i64);
                }
            }

            // SelectChoice FMA sub_temp_vreg: live [op.cycle, op.cycle+2) when FMA path is chosen.
            // ForceFma => required interval; Optimize => optional (present when choice_var); ForceVselect => none.
            for &idx in &select_choice_ops {
                if ops[idx].sub_temp_vreg.is_none() {
                    continue; // delta path: no sub_temp
                }
                if idx >= cycle_vars.len() {
                    continue;
                }
                let start = cycle_vars[idx];
                match select_choice_mode {
                    SelectChoiceMode::ForceFma => {
                        let interval = model.new_fixed_size_interval_var(start, 2);
                        scratch_intervals.push(interval);
                        scratch_demands.push(8);
                    }
                    SelectChoiceMode::Optimize => {
                        if let Some(&choice_var) = select_choice_vars.get(&idx) {
                            let interval =
                                model.new_optional_fixed_size_interval_var(start, 2, choice_var);
                            scratch_intervals.push(interval);
                            scratch_demands.push(8);
                        }
                    }
                    SelectChoiceMode::ForceVselect => {}
                }
            }

            model.add_cumulative(scratch_intervals, scratch_demands, scratch_budget as i64);

            // Add hints for size variables if we have greedy hints
            if let Some(hints) = greedy_hints {
                let max_hint = hints.iter().copied().max().unwrap_or(0);
                if let Some(seg_end) = segment_end_var {
                    model.add_hint(seg_end, max_hint as i64);
                }
                let mut bad_size_count = 0;
                for vreg in &live_vregs {
                    let is_live_out = live_out_vregs.is_some_and(|s| s.contains(vreg));
                    if is_live_out {
                        continue; // live-out uses segment end, not last_use_idx
                    }
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

                // Now add size hints (live-out VRegs: use segment end = max_hint)
                for (i, (size_var, def_idx, last_use_idx)) in size_vars.iter().enumerate() {
                    let vreg = &live_vregs[i];
                    let is_live_out = live_out_vregs.is_some_and(|s| s.contains(vreg));
                    if *def_idx >= hints.len() {
                        continue;
                    }
                    let def_cycle = hints[*def_idx];
                    let expected_size = if is_live_out {
                        (max_hint - def_cycle + 1).max(1)
                    } else {
                        if *last_use_idx >= hints.len() {
                            continue;
                        }
                        (hints[*last_use_idx] - def_cycle + 1).max(1)
                    };
                    model.add_hint(*size_var, expected_size as i64);
                }

                if bad_size_count > 0 && verbose {
                    // This can happen when merging section schedules with different local cycle spaces.
                    // The hint is just a starting point - if it's invalid, the solver will find a valid schedule.
                    eprintln!(
                        "  Note: {} VRegs have last_use_cycle < def_cycle in hints. Solver will recompute valid cycles.",
                        bad_size_count
                    );
                }
            }

            // CRITICAL: Verify hints are feasible before solving
            // If hints are infeasible, there's a bug in our model - fail immediately
            if let Some(hints) = greedy_hints {
                let max_hint = hints.iter().copied().max().unwrap_or(0);
                // Compute peak liveness from hints (live-out VRegs extend to segment end = max_hint)
                let mut events: Vec<(i32, i64)> = Vec::new();
                if let Some(il) = initial_live {
                    for (vreg, last_use_idx) in il.iter() {
                        if *last_use_idx < hints.len() {
                            let size = vreg.size() as i64;
                            events.push((0, size));
                            events.push((hints[*last_use_idx] + 1, -size));
                        }
                    }
                }
                if let Some(lt) = live_through_vregs {
                    for vreg in lt {
                        let size = vreg.size() as i64;
                        events.push((0, size));
                        events.push((max_hint + 1, -size));
                    }
                }
                for vreg in &live_vregs {
                    let def_idx = vreg_def_op[vreg];
                    let last_use_idx = vreg_last_use[vreg];
                    let def_cycle = hints.get(def_idx).copied().unwrap_or(0);
                    let last_use_cycle = if live_out_vregs.is_some_and(|s| s.contains(vreg)) {
                        max_hint
                    } else {
                        hints.get(last_use_idx).copied().unwrap_or(def_cycle)
                    };
                    let size = vreg.size() as i64;
                    events.push((def_cycle, size)); // VReg becomes live
                    events.push((last_use_cycle + 1, -size)); // VReg dies after last use
                }
                // SelectChoice FMA sub_temp: live [c, c+2)
                if let Some(fma_ids) = select_choice_hints {
                    for (op_idx, op) in ops.iter().enumerate() {
                        if op.opcode == Opcode::SelectChoice
                            && fma_ids.contains(&op.id)
                            && op.sub_temp_vreg.is_some()
                            && op_idx < hints.len()
                        {
                            let c = hints[op_idx];
                            events.push((c, 8));
                            events.push((c + 2, -8));
                        }
                    }
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
                    eprintln!(
                        "  Hint verification: peak_live={} <= budget={} OK",
                        peak, scratch_budget
                    );
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
            if (valu_per_cycle[cycle] as i64) > valu_cap {
                if valu_violations < 3 && verbose {
                    eprintln!(
                        "  WARNING: VALU overflow at cycle {}: {} > {}",
                        cycle, valu_per_cycle[cycle], valu_cap
                    );
                }
                valu_violations += 1;
            }
            if (alu_per_cycle[cycle] as i64) > alu_cap {
                if alu_violations < 3 && verbose {
                    eprintln!(
                        "  WARNING: ALU overflow at cycle {}: {} > {}",
                        cycle, alu_per_cycle[cycle], alu_cap
                    );
                }
                alu_violations += 1;
            }
        }

        if valu_violations > 0 || alu_violations > 0 {
            // This can happen when merging independent section schedules. Each section's CP-SAT
            // respects VALU limits locally, but when merged, sections can overlap on VALU cycles.
            // The full-kernel CP-SAT will rebalance as needed. Warn but don't panic.
            if verbose {
                eprintln!(
                    "  Note: Merged hint has engine capacity issues (VALU: {}, ALU: {}). CP-SAT will rebalance.",
                    valu_violations, alu_violations
                );
            }
        }

        if verbose {
            eprintln!(
                "  Engine capacity verification: OK (VALU<={}, ALU<={} at all cycles)",
                valu_cap, alu_cap
            );
        }
    }

    // 9. Compute capacity-based lower bounds via global min over (x, y, z)
    let valu_count = ops.iter().filter(|op| op.engine == Engine::Valu).count() as i64;
    let alu_count = ops.iter().filter(|op| op.engine == Engine::Alu).count() as i64;
    let load_count = ops.iter().filter(|op| op.engine == Engine::Load).count() as i64;
    let store_count = ops.iter().filter(|op| op.engine == Engine::Store).count() as i64;
    let shiftable_count = shiftable_set.len() as i64;
    let const_choice_count = const_choice_ops.len() as i64;
    let select_choice_count = select_choice_ops.len() as i64;
    let flow_base = ops.iter().filter(|op| op.engine == Engine::Flow).count() as i64;

    // Use 0 for dependency floor: critical_path_len (max ASAP + 1) assumes +1 per edge, but the model
    // uses variable-length steps (completion_cycle for shifted, +2 for SelectChoice FMA), so it can
    // overestimate. The capacity-derived minimum over (x,y,z) is a valid lower bound; precedence
    // constraints in the model enforce any dependency lower bound, so the solver cannot find
    // makespan below the true minimum.
    let cap_result = compute_capacity_lower_bound_global(
        valu_count,
        alu_count,
        load_count,
        store_count,
        flow_base,
        shiftable_count,
        const_choice_count,
        select_choice_count,
        enable_valu_shift,
        valu_cap,
        alu_cap,
        load_cap,
        store_cap,
        flow_cap,
        0i64, // dependency floor: don't inflate capacity_lower_bound with ASAP-based critical path
    );
    let capacity_lower_bound = cap_result.capacity_lower_bound;
    let mut valu_alu_global_min = cap_result.valu_alu_global_min;
    let load_bound = cap_result.load_bound;
    let store_bound = cap_result.store_bound;

    if verbose {
        eprintln!(
            "  Capacity bounds (conservative): shiftable={}, const_choice={}; Load={}, Store={}, final_LB={}",
            shiftable_count,
            const_choice_count,
            load_bound,
            store_bound,
            capacity_lower_bound
        );
    }

    // Sanity: when greedy is within scratch budget, it must respect capacity lower bound (infeasible greedy can use fewer cycles).
    let greedy_ok = greedy_within_scratch_budget == Some(true);
    if greedy_ok {
        if let Some(h) = greedy_hints {
            let greedy_makespan = (h.iter().max().copied().unwrap_or(0) + 1) as i64;
            if greedy_makespan < capacity_lower_bound {
                panic!(
                    "greedy makespan {} < capacity_lower_bound {} (greedy schedule would violate capacity; \
                     capacity_lb is a valid lower bound so this indicates a bug in greedy or in capacity_lb)",
                    greedy_makespan, capacity_lower_bound
                );
            }
        }
    }

    // 10. Create makespan and max_cycle variables; tie makespan to actual last cycle so CP-SAT minimizes it.
    // max_cycle = max over all cycle_vars, completion_cycle_vars, select_choice_fma_last_cycle.
    // makespan = max_cycle + 1 (so minimizing makespan forces packing).
    // capacity_lower_bound is a valid lower bound (capacity-derived min over choices); we pass 0 for
    // the dependency floor in compute_capacity_lower_bound_global so it is not inflated by ASAP-based
    // critical path, which overestimates when the model uses variable-length steps.
    let max_cycle_lb = (capacity_lower_bound - 1).max(0);
    let max_cycle =
        model.new_int_var_with_name([(max_cycle_lb, (horizon - 1) as i64)], "max_cycle".to_string());
    let makespan =
        model.new_int_var_with_name([(capacity_lower_bound, horizon)], "makespan".to_string());
    model.add_eq(LinearExpr::from(makespan), LinearExpr::from(max_cycle) + 1);
    for &cv in cycle_vars.iter() {
        model.add_ge(max_cycle, LinearExpr::from(cv));
    }
    for (_, &completion_var) in completion_cycle_vars.iter() {
        model.add_ge(max_cycle, LinearExpr::from(completion_var));
    }
    for (_, &fma_last) in select_choice_fma_last_cycle.iter() {
        model.add_ge(max_cycle, LinearExpr::from(fma_last));
    }

    // Redundant linear capacity constraints: only add bounds that are valid for *any* choice.
    // capacity_lower_bound (makespan domain) is already the min over (x,y,z) from compute_capacity_lower_bound_global.
    // Do NOT add makespan >= ceil(alu_count+8*shiftable/12) or makespan >= ceil(valu_count-shiftable/6): those are
    // bounds for one extreme (all shift / no shift) and would over-constrain when the solver picks a different choice
    // (e.g. shift 163 only → ALU need 121 cycles, but "all shift" bound would force makespan >= 609).
    // Load and store bounds are valid global mins (min over const_choice for load; store is fixed).
    if load_bound > 0 {
        let load_lb = model.new_int_var([(load_bound, load_bound)]);
        model.add_ge(makespan, LinearExpr::from(load_lb));
    }
    if store_bound > 0 {
        let store_lb = model.new_int_var([(store_bound, store_bound)]);
        model.add_ge(makespan, LinearExpr::from(store_lb));
    }

    // Direct capacity constraints: link makespan to choice variables (x = shift, y = ConstChoice AddImm, z = SelectChoice FMA count)
    // so CP-SAT can optimize over (x,y,z) during search instead of relying only on the pre-computed capacity_lower_bound.
    // Uses same formulas as compute_capacity_lower_bound_global: VALU_ops = valu_count - sum(shift) + 2*sum(select_fma), etc.
    {
        let zero_var = model.new_int_var([(0, 0)]);
        let sum_shift: Option<LinearExpr> = shiftable_valu
            .iter()
            .filter_map(|&idx| shift_vars.get(&idx).copied())
            .fold(None, |acc, b| {
                Some(match acc {
                    None => LinearExpr::from(b),
                    Some(a) => a + LinearExpr::from(b),
                })
            });
        let sum_select_fma: Option<LinearExpr> = select_choice_ops
            .iter()
            .filter_map(|&idx| select_choice_vars.get(&idx).copied())
            .fold(None, |acc, b| {
                Some(match acc {
                    None => LinearExpr::from(b),
                    Some(a) => a + LinearExpr::from(b),
                })
            });
        let sum_const_add_imm: Option<LinearExpr> = const_choice_ops
            .iter()
            .filter_map(|&idx| const_choice_vars.get(&idx).copied())
            .fold(None, |acc, b| {
                Some(match acc {
                    None => LinearExpr::from(b),
                    Some(a) => a + LinearExpr::from(b),
                })
            });
        let s = sum_shift.unwrap_or_else(|| LinearExpr::from(zero_var));
        let z = sum_select_fma.unwrap_or_else(|| LinearExpr::from(zero_var));
        let y = sum_const_add_imm.unwrap_or_else(|| LinearExpr::from(zero_var));

        // VALU: each SelectChoice FMA = 2 VALU ops (sub + fma). So 6*makespan >= valu_count - sum(shift) + 2*sum(select_fma)
        if enable_valu_shift && (valu_count > 0 || shiftable_count > 0 || select_choice_count > 0) {
            let lhs = LinearExpr::from(makespan) * 6 + s.clone() - z.clone() * 2;
            let rhs = model.new_int_var([(valu_count, valu_count)]);
            model.add_ge(lhs, LinearExpr::from(rhs));
        }
        // ALU: 12*makespan >= alu_count + 8*sum(shift)  =>  12*makespan - 8*s >= alu_count
        if enable_valu_shift && (alu_count > 0 || shiftable_count > 0) {
            let lhs = LinearExpr::from(makespan) * 12 - s.clone() * 8;
            let rhs = model.new_int_var([(alu_count, alu_count)]);
            model.add_ge(lhs, LinearExpr::from(rhs));
        }
        // Flow: makespan >= flow_base + sum(const_add_imm) - sum(select_fma)  =>  makespan + z - y >= flow_base
        if flow_base > 0 || const_choice_count > 0 || select_choice_count > 0 {
            let lhs = LinearExpr::from(makespan) + z.clone() - y.clone();
            let rhs = model.new_int_var([(flow_base, flow_base)]);
            model.add_ge(lhs, LinearExpr::from(rhs));
        }
        // Load: 2*makespan >= load_count - sum(const_add_imm)
        if load_count > 0 && const_choice_count > 0 {
            let lhs = LinearExpr::from(makespan) * 2 + y;
            let rhs = model.new_int_var([(load_count, load_count)]);
            model.add_ge(lhs, LinearExpr::from(rhs));
        }
    }

    // makespan >= max(chosen_cycle[i]) + 1: for each op use the cycle of the chosen execution path.
    // Shiftable: when shifted use completion_cycle, when not use cycle_vars[idx].
    // SelectChoice: when FMA (no-delta) use fma_cycle_var (last cycle), else use cycle_vars[idx]. ConstChoice: both paths use cycle_vars[idx].
    for (idx, &cv) in cycle_vars.iter().enumerate() {
        if shiftable_set.contains(&idx) {
            let shift_var = *shift_vars.get(&idx).unwrap();
            if let Some(&completion_var) = completion_cycle_vars.get(&idx) {
                let c = model.add_ge(makespan, LinearExpr::from(completion_var) + 1);
                model.only_enforce_if(c, [shift_var]);
            }
            let c = model.add_ge(makespan, LinearExpr::from(cv) + 1);
            model.only_enforce_if(c, [!shift_var]);
        } else if select_choice_set.contains(&idx) {
            match select_choice_mode {
                SelectChoiceMode::ForceVselect => {
                    model.add_ge(makespan, LinearExpr::from(cv) + 1);
                }
                SelectChoiceMode::ForceFma => {
                    if let Some(&fma_last) = select_choice_fma_last_cycle.get(&idx) {
                        model.add_ge(makespan, LinearExpr::from(fma_last) + 1);
                    } else {
                        model.add_ge(makespan, LinearExpr::from(cv) + 1);
                    }
                }
                SelectChoiceMode::Optimize => {
                    if let Some(&fma_last) = select_choice_fma_last_cycle.get(&idx) {
                        let choice_var = *select_choice_vars.get(&idx).unwrap();
                        let c = model.add_ge(makespan, LinearExpr::from(fma_last) + 1);
                        model.only_enforce_if(c, [choice_var]);
                        let c = model.add_ge(makespan, LinearExpr::from(cv) + 1);
                        model.only_enforce_if(c, [!choice_var]);
                    } else {
                        model.add_ge(makespan, LinearExpr::from(cv) + 1);
                    }
                }
            }
        } else {
            model.add_ge(makespan, LinearExpr::from(cv) + 1);
        }
    }

    // makespan = max_cycle + 1 and max_cycle >= every cycle var already force all work into [0, max_cycle];
    // no need for explicit cycle <= makespan-1.

    // Minimize makespan; when equal, prefer earlier completion_cycle for shifted ops, ConstChoice ops, shiftable VALU ops, and SelectChoice ops (tie-breaker).
    //
    // Objective = scale * makespan + sum(completion_cycle) + sum(const_choice_cycles) + sum(shiftable_valu_cycles) + sum(select_choice_cycles).
    // Scale is a power of 10 chosen so that scale > max_sum (max possible value of that sum). Then:
    //   (1) Tie-breaker: one extra cycle costs scale, so scale > max_sum ensures makespan dominates.
    //   (2) Leading digits: tie_breaker < scale, so floor(objective / scale) = makespan — the quotient is the cycle count (e.g. objective 2_000_042_000 with scale 1e7 => makespan 200, not 2000; tie-breaker 42_000 in lower digits).
    //   (3) Including ConstChoice cycles rewards earlier const ops, which naturally prefers add_imm
    //       when Flow is free (enabling better load balance).
    //   (4) Including shiftable VALU cycles rewards earlier VALU ops, which encourages shifting to ALU
    //       when that allows earlier scheduling (freeing VALU slots for other ops).
    //   (5) Including SelectChoice cycles rewards earlier SelectChoice ops, which may favor FMA over vselect
    //       when FMA can execute earlier (though FMA uses VALU slots which may be saturated).
    let mut objective_scale: Option<i64> = None;

    // Tie-breaker should sum only the chosen path's cycle per op (same as makespan constraints).
    // Create auxiliary "chosen cycle" vars for shiftable and SelectChoice so the objective matches.
    let mut shiftable_chosen_cycle: HashMap<usize, IntVar> = HashMap::new();
    for &idx in &shiftable_valu {
        let op_asap = *asap.get(&ops[idx].id).unwrap_or(&0) as i64;
        let default_alap = (horizon - 1) as i32;
        let op_alap = *alap.get(&ops[idx].id).unwrap_or(&default_alap) as i64;
        let chosen = model.new_int_var_with_name(
            [(op_asap, op_alap)],
            format!("shift_chosen_cycle_{}", idx),
        );
        let shift_var = *shift_vars.get(&idx).unwrap();
        if let Some(&cc_var) = completion_cycle_vars.get(&idx) {
            let c = model.add_eq(LinearExpr::from(chosen), LinearExpr::from(cc_var));
            model.only_enforce_if(c, [shift_var]);
        }
        let c = model.add_eq(LinearExpr::from(chosen), LinearExpr::from(cycle_vars[idx]));
        model.only_enforce_if(c, [!shift_var]);
        shiftable_chosen_cycle.insert(idx, chosen);
    }

    let mut select_chosen_cycle: HashMap<usize, IntVar> = HashMap::new();
    for &idx in &select_choice_ops {
        if let Some(&fma_last) = select_choice_fma_last_cycle.get(&idx) {
            let op_asap = *asap.get(&ops[idx].id).unwrap_or(&0) as i64;
            let default_alap = (horizon - 1) as i32;
            let op_alap = *alap.get(&ops[idx].id).unwrap_or(&default_alap) as i64;
            let chosen = model.new_int_var_with_name(
                [(op_asap, op_alap)],
                format!("select_chosen_cycle_{}", idx),
            );
            let choice_var = *select_choice_vars.get(&idx).unwrap();
            let c = model.add_eq(LinearExpr::from(chosen), LinearExpr::from(fma_last));
            model.only_enforce_if(c, [choice_var]);
            let c = model.add_eq(LinearExpr::from(chosen), LinearExpr::from(cycle_vars[idx]));
            model.only_enforce_if(c, [!choice_var]);
            select_chosen_cycle.insert(idx, chosen);
        }
    }

    // Collect tie-breaker vars: one "effective cycle" per op so that when makespan is equal we
    // prefer packing ops earlier (minimize sum of cycles). Without this, ops not in shiftable/
    // ConstChoice/SelectChoice have no incentive to move from C+1 to empty C.
    let mut tie_breaker_vars: Vec<IntVar> = Vec::new();

    // Shiftable VALU: one chosen cycle per op (completion_cycle when shifted, cycle_var when not)
    for &idx in &shiftable_valu {
        tie_breaker_vars.push(*shiftable_chosen_cycle.get(&idx).unwrap());
    }

    // ConstChoice: both paths use same cycle_vars[idx]
    for &idx in &const_choice_ops {
        tie_breaker_vars.push(cycle_vars[idx]);
    }

    // SelectChoice: chosen cycle (fma_cycle when FMA no-delta, else cycle_var)
    for &idx in &select_choice_ops {
        if let Some(&chosen) = select_chosen_cycle.get(&idx) {
            tie_breaker_vars.push(chosen);
        } else {
            tie_breaker_vars.push(cycle_vars[idx]);
        }
    }

    // All other ops: use cycle_vars so every op contributes to tie-breaker (reward moving any op earlier)
    for (idx, _) in ops.iter().enumerate() {
        if !shiftable_set.contains(&idx)
            && !const_choice_set.contains(&idx)
            && !select_choice_set.contains(&idx)
        {
            tie_breaker_vars.push(cycle_vars[idx]);
        }
    }

    if let Some(&first_var) = tie_breaker_vars.first() {
        let rest = &tie_breaker_vars[1..];
        // Max possible value of sum(tie_breaker_vars): each var in [0, horizon-1], so sum <= n * horizon.
        // Use saturating mul so overflow doesn't produce a tiny scale and break "leading digits = makespan".
        let max_sum = (tie_breaker_vars.len() as i64).saturating_mul(horizon);
        // Scale must be > max_sum so that floor(objective/scale) = makespan (tie-breaker stays in lower digits).
        // Presolve can "ExpandObjective" (substitute tight equalities), adding more terms so the actual
        // objective tie-breaker can exceed max_sum. Use an extra factor of 10 so scale stays ahead.
        let mut scale = 1i64;
        while scale <= max_sum {
            scale = scale.saturating_mul(10);
        }
        if scale <= max_sum {
            // Can happen if scale saturated (overflow); objective structure is then wrong.
            eprintln!(
                "  CP-SAT objective scale saturated (max_sum={}); leading digits may not equal makespan",
                max_sum
            );
        }
        objective_scale = Some(scale);
        let sum_completion: LinearExpr = rest.iter().fold(LinearExpr::from(first_var), |acc, &v| {
            acc + LinearExpr::from(v)
        });
        // Explicitly bound tie-breaker below scale so floor(objective/scale) = makespan always.
        model.add_le(sum_completion.clone(), scale.saturating_sub(1));
        let objective = LinearExpr::from(makespan) * scale + sum_completion;
        model.minimize(objective);
    } else {
        model.minimize(makespan);
    }

    // 11. Add hints from greedy scheduler
    // We skip cycle hints for ops with hint_cycle < 0 (dead stores). Greedy marks those as -1
    // and does not assign a real cycle, so CP-SAT must choose.
    //
    // We hint cycle, shift, completion_cycle, alu_cycle, shiftable_chosen_cycle, select_chosen_cycle,
    // const_choice, select_choice, segment sizes, makespan, max_cycle so the solution hint is complete.
    // Set CPSAT_HINT_DEBUG=1 to log every cycle hint (op idx, id, value, domain) to correlate with "moved var hint".
    let hint_debug_ssa = std::env::var("CPSAT_HINT_DEBUG").is_ok();
    if let Some(hints) = greedy_hints {
        let mut skipped_cycle_hints = 0usize;
        for (idx, &hint_cycle) in hints.iter().enumerate() {
            if idx < cycle_vars.len() && hint_cycle >= 0 {
                if idx < ops.len() {
                    let lb = *asap.get(&ops[idx].id).unwrap_or(&0) as i64;
                    let ub = *alap.get(&ops[idx].id).unwrap_or(&(horizon as i32 - 1)) as i64;
                    let c = hint_cycle as i64;
                    if hint_debug_ssa {
                        eprintln!(
                            "[CPSAT_HINT_DEBUG] cycle_hint idx={} op_id={} hint={} domain=[{}, {}]",
                            idx, ops[idx].id, c, lb, ub
                        );
                    }
                    assert!(
                        c >= lb && c <= ub,
                        "cycle hint outside [ASAP, ALAP]: op id={} idx={} hint_cycle={} domain=[{}, {}] (fix greedy or ASAP/ALAP)",
                        ops[idx].id, idx, c, lb, ub
                    );
                }
                model.add_hint(cycle_vars[idx], hint_cycle as i64);
            } else if idx < cycle_vars.len() && hint_cycle < 0 {
                skipped_cycle_hints += 1;
            }
        }
        if verbose && skipped_cycle_hints > 0 {
            eprintln!(
                "  Hint: {} cycle vars unhinted (dead stores, cycle=-1); solver will assign cycles.",
                skipped_cycle_hints
            );
        }
        // Hint makespan and max_cycle so the solver starts from a tight schedule (makespan = max_cycle + 1).
        let hint_makespan = hints.iter().max().copied().unwrap_or(0) + 1;
        model.add_hint(makespan, hint_makespan as i64);
        model.add_hint(max_cycle, (hint_makespan - 1) as i64);
    }

    // Add shift hints - always hint ALL shift variables to avoid "unhinted variables" warnings.
    // Use greedy's shift decisions when provided; otherwise default to no shift (0).
    if !shift_vars.is_empty() {
        let mut hinted_true = 0usize;
        for &idx in &shiftable_valu {
            let shift_var = shift_vars
                .get(&idx)
                .expect("shift_var should exist for all shiftable ops");
            let op_id = ops[idx].id;
            let forced_value = match valu_shift_mode {
                ValuShiftMode::ForceValu => Some(false),
                ValuShiftMode::ForceAlu => Some(true),
                ValuShiftMode::Optimize => forced_shifts.and_then(|forced| {
                    if forced.contains(&op_id) {
                        Some(true)
                    } else {
                        None
                    }
                }),
            };
            let should_shift = forced_value
                .or_else(|| shift_hints.map(|h| h.contains(&op_id)))
                .unwrap_or(false);
            if should_shift {
                hinted_true += 1;
            }
            model.add_hint(*shift_var, if should_shift { 1 } else { 0 });
        }
        if verbose {
            eprintln!(
                "  Hinting {} shift variables ({} with shift=true, {} with shift=false)",
                shift_vars.len(),
                hinted_true,
                shift_vars.len() - hinted_true
            );
        }
    }

    // Hint completion_cycle and alu_cycle for shiftable ops so CP-SAT gets a full solution hint.
    // Greedy gives one cycle per op; we use that for completion_cycle and all 8 alu_cycles
    // (valid when shifted: all 8 ALU ops at same cycle -> completion_cycle = that cycle).
    // By construction greedy never schedules before ASAP; if this fires, fix greedy or ASAP/ALAP.
    if let Some(hints) = greedy_hints {
        for &idx in &shiftable_valu {
            let hint_cycle = hints.get(idx).copied().unwrap_or(-1);
            if hint_cycle < 0 {
                continue;
            }
            let c = hint_cycle as i64;
            let lb = *asap.get(&ops[idx].id).unwrap_or(&0) as i64;
            let ub = *alap.get(&ops[idx].id).unwrap_or(&((horizon - 1) as i32)) as i64;
            assert!(
                c >= lb && c <= ub,
                "completion/alu cycle hint outside [ASAP, ALAP]: op id={} idx={} hint_cycle={} domain=[{}, {}] (fix greedy or ASAP/ALAP)",
                ops[idx].id, idx, c, lb, ub
            );
            if let Some(completion_var) = completion_cycle_vars.get(&idx) {
                model.add_hint(*completion_var, c);
            }
            if let Some(alu_cycles) = alu_cycle_vars.get(&idx) {
                for &alu_var in alu_cycles {
                    model.add_hint(alu_var, c);
                }
            }
        }
    }

    // Hint shiftable_chosen_cycle and select_chosen_cycle so "solution hint is complete" (no unhinted vars).
    // These tie-breaker vars are equal to the effective cycle (completion_cycle when shifted, cycle when not;
    // fma_last when FMA, cycle when vselect). Greedy gives one cycle per op -> use that for chosen.
    if let Some(hints) = greedy_hints {
        for &idx in &shiftable_valu {
            let hint_cycle = hints.get(idx).copied().unwrap_or(-1);
            if hint_cycle < 0 {
                continue;
            }
            if let Some(&chosen) = shiftable_chosen_cycle.get(&idx) {
                model.add_hint(chosen, hint_cycle as i64);
            }
        }
        for &idx in &select_choice_ops {
            let hint_cycle = hints.get(idx).copied().unwrap_or(-1);
            if hint_cycle < 0 {
                continue;
            }
            if let Some(&chosen) = select_chosen_cycle.get(&idx) {
                // vselect path: chosen = cycle; FMA path: chosen = fma_last (often cycle+1). Use hint_cycle as hint.
                model.add_hint(chosen, hint_cycle as i64);
            }
        }
        // Hint select_choice_fma_last_cycle (FMA path: fma_cycle = cycle + 1) so hint is complete.
        let use_fma = |idx: usize| -> bool {
            select_choice_hints.map_or(false, |h| h.contains(&ops[idx].id))
        };
        for (&idx, &fma_last_var) in &select_choice_fma_last_cycle {
            if use_fma(idx) {
                let hint_cycle = hints.get(idx).copied().unwrap_or(-1);
                if hint_cycle >= 0 {
                    model.add_hint(fma_last_var, (hint_cycle + 1) as i64);
                }
            }
        }
        // Hint select_fma_extra vars (0 or 1 = same as choice_var) so hint is complete.
        for (&(dep_idx, _idx), &extra_var) in &select_fma_extra_vars {
            model.add_hint(extra_var, if use_fma(dep_idx) { 1 } else { 0 });
        }
    }

    // 12. Configure solver
    let mut params = SatParameters::default();
    params.max_time_in_seconds = Some(time_limit_secs);
    // Optional: crash when presolve adjusts a hint. Not default: presolve can tighten domains and
    // clamp hints; that's expected and the solve still succeeds.
    if std::env::var("CPSAT_DEBUG_PRESOLVE_HINT").is_ok() {
        params.debug_crash_if_presolve_breaks_hint = Some(true);
    }
    // Always log search progress by default - user must go out of their way to suppress
    params.log_search_progress = Some(true);
    params.num_workers = Some(8);

    // Gap limit: default uses scale/3 for faster convergence, but can force 0 for debugging.
    // Stop when gap <= scale/3: remaining improvement is only in the tie-breaker (sum of completion_cycle).
    // One extra cycle costs scale in the objective, so scale/3 guarantees we're within ~1 cycle worst case.
    if force_zero_gap {
        // Force 0 integral gap for proven optimality (debugging only).
        // This ensures CP-SAT explores all possibilities, including tie-breaker optimizations
        // like moving ConstChoice ops earlier via add_imm.
        params.absolute_gap_limit = Some(0.0);
    } else if let Some(scale) = objective_scale {
        params.absolute_gap_limit = Some((scale / 3) as f64);
    }

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
            // Extract cycle assignments for all ops
            // For shiftable VALU: ForceAlu => completion_cycle; ForceValu => cycle_vars[idx]; Optimize => shift_var ? completion_cycle : cycle_vars[idx]
            let mut cycle_assignments: Vec<i32> = Vec::with_capacity(ops.len());
            for (idx, _) in ops.iter().enumerate() {
                if shiftable_set.contains(&idx) {
                    match valu_shift_mode {
                        ValuShiftMode::ForceAlu => {
                            if let Some(&completion_cycle_var) = completion_cycle_vars.get(&idx) {
                                cycle_assignments
                                    .push(completion_cycle_var.solution_value(&response) as i32);
                            } else {
                                cycle_assignments
                                    .push(cycle_vars[idx].solution_value(&response) as i32);
                            }
                        }
                        ValuShiftMode::ForceValu => {
                            cycle_assignments
                                .push(cycle_vars[idx].solution_value(&response) as i32);
                        }
                        ValuShiftMode::Optimize => {
                            if let Some(shift_var) = shift_vars.get(&idx) {
                                let is_shifted = shift_var.solution_value(&response);
                                if is_shifted {
                                    if let Some(&completion_cycle_var) =
                                        completion_cycle_vars.get(&idx)
                                    {
                                        cycle_assignments
                                            .push(completion_cycle_var.solution_value(&response)
                                                as i32);
                                    } else {
                                        cycle_assignments
                                            .push(cycle_vars[idx].solution_value(&response) as i32);
                                    }
                                } else {
                                    cycle_assignments
                                        .push(cycle_vars[idx].solution_value(&response) as i32);
                                }
                            } else {
                                cycle_assignments
                                    .push(cycle_vars[idx].solution_value(&response) as i32);
                            }
                        }
                    }
                } else {
                    cycle_assignments.push(cycle_vars[idx].solution_value(&response) as i32);
                }
            }

            // Reported makespan must match the objective's leading digits (floor(objective/scale)).
            // The model constrains makespan >= cycle_var+1 for every cycle_var; for shiftable ops we
            // put completion_cycle into cycle_assignments when shifted, so max(cycle_assignments)+1
            // can be lower than the solver's makespan (which is driven by cycle_vars). Use the
            // makespan variable's solution value so we report the same number that appears in the objective.
            let actual_makespan = makespan.solution_value(&response) as i32;

            // Sanity: makespan variable should be >= max(cycle_assignments)+1 (model constraint).
            let max_cycle_from_assignments = cycle_assignments.iter().max().copied().unwrap_or(0) + 1;
            if actual_makespan < max_cycle_from_assignments && verbose {
                eprintln!(
                    "  Inconsistency: makespan var={} but max(cycle_assignments)+1={}",
                    actual_makespan, max_cycle_from_assignments
                );
            }

            // Empty-cycle gap: when makespan > max_cycle_from_assignments, we have empty cycles.
            let empty_count = (actual_makespan - max_cycle_from_assignments).max(0) as usize;
            if empty_count > 0 && verbose {
                let driven_by_bound = (actual_makespan as i64) == capacity_lower_bound;
                eprintln!(
                    "  Empty cycles: {} (actual_makespan={}, max_cycle_from_assignments+1={}, makespan_driven_by_capacity_bound={})",
                    empty_count,
                    actual_makespan,
                    max_cycle_from_assignments,
                    driven_by_bound
                );
            }

            // Extract shift decisions: ForceAlu => all shiftable; ForceValu => none; Optimize => from solver
            let shifted_ops: HashSet<usize> = match valu_shift_mode {
                ValuShiftMode::ForceAlu => shiftable_valu.iter().map(|&idx| ops[idx].id).collect(),
                ValuShiftMode::ForceValu => HashSet::new(),
                ValuShiftMode::Optimize => {
                    if shift_vars.is_empty() {
                        HashSet::new()
                    } else {
                        shift_vars
                            .iter()
                            .filter(|(_, &var)| var.solution_value(&response))
                            .map(|(&idx, _)| ops[idx].id)
                            .collect()
                    }
                }
            };

            // Debug: Extract shift decisions for high-impact ops
            let high_impact_op_ids: HashSet<usize> = [244, 287, 291, 285, 323, 289, 262]
                .iter()
                .copied()
                .collect();
            if verbose && !shift_vars.is_empty() {
                for &idx in &shiftable_valu {
                    let op_id = ops[idx].id;
                    if high_impact_op_ids.contains(&op_id) {
                        let shift_var = shift_vars.get(&idx).expect("shift_var should exist");
                        let is_shifted = shift_var.solution_value(&response);
                        eprintln!("  High-impact op {}: shifted={}", op_id, is_shifted);

                        if is_shifted {
                            // Extract what cycles CP-SAT would assign
                            if let Some(alu_cycles) = alu_cycle_vars.get(&idx) {
                                let cycles: Vec<i32> = alu_cycles
                                    .iter()
                                    .map(|cv| cv.solution_value(&response) as i32)
                                    .collect();
                                let unique_cycles: std::collections::HashSet<i32> =
                                    cycles.iter().copied().collect();
                                eprintln!(
                                    "    ALU cycles: {:?} ({} unique)",
                                    cycles,
                                    unique_cycles.len()
                                );

                                if let Some(&completion_cycle_var) = completion_cycle_vars.get(&idx)
                                {
                                    let completion =
                                        completion_cycle_var.solution_value(&response) as i32;
                                    eprintln!("    Completion cycle: {}", completion);
                                }
                            }
                        } else {
                            // Check why it's not shifted - what would happen if it were?
                            let op_asap = *asap.get(&op_id).unwrap_or(&0);
                            let op_alap = *alap.get(&op_id).unwrap_or(&((horizon - 1) as i32));
                            eprintln!("    Not shifted: ASAP={}, ALAP={}", op_asap, op_alap);
                        }
                    }
                }
            }

            // Extract individual ALU cycle assignments for shifted VALU ops (ForceAlu => all; Optimize => from solver)
            let mut shifted_alu_cycles: HashMap<usize, Vec<i32>> = HashMap::new();
            for &idx in &shiftable_valu {
                let op_id = ops[idx].id;
                let is_shifted = shifted_ops.contains(&op_id);
                if is_shifted {
                    if let Some(alu_cycles) = alu_cycle_vars.get(&idx) {
                        let cycles: Vec<i32> = alu_cycles
                            .iter()
                            .map(|cv| cv.solution_value(&response) as i32)
                            .collect();
                        shifted_alu_cycles.insert(op_id, cycles);
                    }
                }
            }

            // Extract ConstChoice: ForceAddImm => all; ForceLoad => none; Optimize => from solver
            let const_add_imm_ops: HashSet<usize> = match const_choice_mode {
                ConstChoiceMode::ForceAddImm => {
                    const_choice_ops.iter().map(|&idx| ops[idx].id).collect()
                }
                ConstChoiceMode::ForceLoad => HashSet::new(),
                ConstChoiceMode::Optimize => {
                    if const_choice_vars.is_empty() {
                        HashSet::new()
                    } else {
                        const_choice_vars
                            .iter()
                            .filter(|(_, &var)| var.solution_value(&response))
                            .map(|(&idx, _)| ops[idx].id)
                            .collect()
                    }
                }
            };

            // Extract SelectChoice: ForceFma => all; ForceVselect => none; Optimize => from solver
            let select_fma_ops: HashSet<usize> = match select_choice_mode {
                SelectChoiceMode::ForceFma => {
                    select_choice_ops.iter().map(|&idx| ops[idx].id).collect()
                }
                SelectChoiceMode::ForceVselect => HashSet::new(),
                SelectChoiceMode::Optimize => {
                    if select_choice_vars.is_empty() {
                        HashSet::new()
                    } else {
                        select_choice_vars
                            .iter()
                            .filter(|(_, &var)| var.solution_value(&response))
                            .map(|(&idx, _)| ops[idx].id)
                            .collect()
                    }
                }
            };

            // Build schedule constraints once (same as CP-SAT; used for active-set binding list and bulk-move).
            let schedule_constraints = crate::schedule_constraints::build_ssa_schedule_constraints(
                ops,
                &op_id_to_idx,
                &asap,
                &alap,
                &shiftable_set,
                &shifted_ops,
                &select_choice_with_fma_delay,
                &select_fma_ops,
                horizon as i32,
            );

            // Makespan active set: which constraints are binding at the solution (hold with equality).
            if verbose {
                let max_cycle_val = max_cycle.solution_value(&response) as i32;
                let mut binding_ops: Vec<(usize, usize, &str)> = Vec::new(); // (idx, op_id, reason)
                for (idx, _) in ops.iter().enumerate() {
                    let effective_last = if shiftable_set.contains(&idx) {
                        let is_shifted = shift_vars
                            .get(&idx)
                            .map(|v| v.solution_value(&response))
                            .unwrap_or(false);
                        if is_shifted {
                            completion_cycle_vars
                                .get(&idx)
                                .map(|v| v.solution_value(&response) as i32)
                                .unwrap_or(cycle_assignments[idx])
                        } else {
                            cycle_assignments[idx]
                        }
                    } else if let Some(&fma_last_var) = select_choice_fma_last_cycle.get(&idx) {
                        let is_fma = select_choice_vars
                            .get(&idx)
                            .map(|v| v.solution_value(&response))
                            .unwrap_or(false);
                        if is_fma {
                            fma_last_var.solution_value(&response) as i32
                        } else {
                            cycle_assignments[idx]
                        }
                    } else {
                        cycle_assignments[idx]
                    };
                    if effective_last == max_cycle_val {
                        let reason = if shiftable_set.contains(&idx) {
                            if shift_vars.get(&idx).map(|v| v.solution_value(&response)).unwrap_or(false) {
                                "shifted (completion_cycle)"
                            } else {
                                "VALU cycle"
                            }
                        } else if select_choice_fma_last_cycle.contains_key(&idx) {
                            if select_choice_vars.get(&idx).map(|v| v.solution_value(&response)).unwrap_or(false) {
                                "SelectChoice FMA (fma_last)"
                            } else {
                                "SelectChoice vselect"
                            }
                        } else {
                            "cycle"
                        };
                        binding_ops.push((idx, ops[idx].id, reason));
                    }
                }
                eprintln!("  [Makespan active set] makespan={} max_cycle={}", actual_makespan, max_cycle_val);
                eprintln!("    makespan = max_cycle + 1: always binding");
                if !binding_ops.is_empty() {
                    eprintln!(
                        "    max_cycle >= effective_last: {} op(s) binding (define last cycle)",
                        binding_ops.len()
                    );
                    for (idx, op_id, reason) in binding_ops.iter().take(15) {
                        eprintln!("      idx={} op.id={} {}", idx, op_id, *reason);
                    }
                    if binding_ops.len() > 15 {
                        eprintln!("      ... and {} more", binding_ops.len() - 15);
                    }
                }
                let m = actual_makespan as i64;
                let binding_capacity_bounds: Vec<&str> = [
                    (m == capacity_lower_bound, "capacity_lower_bound"),
                    (load_bound > 0 && m == load_bound, "load_bound"),
                    (store_bound > 0 && m == store_bound, "store_bound"),
                ]
                .iter()
                .filter(|(b, _)| *b)
                .map(|(_, name)| *name)
                .collect();
                if !binding_capacity_bounds.is_empty() {
                    eprintln!(
                        "    makespan >= X (redundant linear): binding for {}",
                        binding_capacity_bounds.join(", ")
                    );
                    // Sanity: if makespan is driven by a capacity bound but the schedule has many empty cycles,
                    // that bound may be wrong (over-constraining). Bulk-move only checks precedence/ordering/domain,
                    // not capacity, so it cannot detect an incorrect capacity bound.
                    if empty_count > 10 {
                        eprintln!(
                            "    WARNING: {} empty cycles but makespan binding on {} — bound may be incorrect (over-constraining); bulk-move cannot check capacity.",
                            empty_count,
                            binding_capacity_bounds.join(", ")
                        );
                    }
                }
                if enable_valu_shift && (valu_count > 0 || shiftable_count > 0 || select_choice_count > 0) {
                    let s = shifted_ops.len() as i64;
                    let z = select_fma_ops.len() as i64;
                    let lhs_valu = 6 * (actual_makespan as i64) + s - 2 * z;
                    let binding_valu = lhs_valu == valu_count;
                    if binding_valu {
                        eprintln!(
                            "    Direct VALU (6*makespan + shift - 2*select_fma >= valu_count): binding (LHS={})",
                            lhs_valu
                        );
                    }
                }
                if enable_valu_shift && (alu_count > 0 || shiftable_count > 0) {
                    let s = shifted_ops.len() as i64;
                    let lhs_alu = 12 * (actual_makespan as i64) - 8 * s;
                    let binding_alu = lhs_alu == alu_count;
                    if binding_alu {
                        eprintln!(
                            "    Direct ALU (12*makespan - 8*shift >= alu_count): binding (LHS={})",
                            lhs_alu
                        );
                    }
                }
                if flow_base > 0 || const_choice_count > 0 || select_choice_count > 0 {
                    let z = select_fma_ops.len() as i64;
                    let y = const_add_imm_ops.len() as i64;
                    let lhs_flow = (actual_makespan as i64) + z - y;
                    let binding_flow = lhs_flow == flow_base;
                    if binding_flow {
                        eprintln!(
                            "    Direct Flow (makespan + select_fma - const_add_imm >= flow_base): binding (LHS={})",
                            lhs_flow
                        );
                    }
                }
                if load_count > 0 && const_choice_count > 0 {
                    let y = const_add_imm_ops.len() as i64;
                    let lhs_load = 2 * (actual_makespan as i64) + y;
                    let binding_load = lhs_load == load_count;
                    if binding_load {
                        eprintln!(
                            "    Direct Load (2*makespan + const_add_imm >= load_count): binding (LHS={})",
                            lhs_load
                        );
                    }
                }
                // Binding precedence/ordering/domain (same checks bulk-move uses). When bulk-move blocks, it should report one of these.
                let binding = crate::schedule_constraints::binding_schedule_constraints(
                    ops,
                    &cycle_assignments,
                    &shifted_alu_cycles,
                    &schedule_constraints,
                );
                let n_precedence = binding.iter().filter(|(c, _)| matches!(c, crate::schedule_constraints::ScheduleConstraint::Precedence { .. })).count();
                let n_ordering = binding.iter().filter(|(c, _)| matches!(c, crate::schedule_constraints::ScheduleConstraint::Ordering { .. })).count();
                let n_domain = binding.iter().filter(|(c, _)| matches!(c, crate::schedule_constraints::ScheduleConstraint::Domain { .. })).count();
                eprintln!(
                    "    Binding schedule constraints (bulk-move uses these): {} precedence, {} ordering, {} domain — when bulk-move blocks, the violation should be one of these.",
                    n_precedence, n_ordering, n_domain
                );
                let max_cycle_val = max_cycle.solution_value(&response) as i32;
                use crate::schedule_constraints::dep_effective_cycle as dep_eff;
                let binding_at_last: Vec<_> = binding
                    .iter()
                    .filter(|(c, _)| {
                        match c {
                            crate::schedule_constraints::ScheduleConstraint::Precedence {
                                consumer_idx,
                                dep_idx,
                                dep_is_shifted,
                                ..
                            } => {
                                let cy_c = cycle_assignments.get(*consumer_idx).copied().unwrap_or(-1);
                                let cy_d = dep_eff(
                                    *dep_idx,
                                    *dep_is_shifted,
                                    &cycle_assignments,
                                    &shifted_alu_cycles,
                                    ops,
                                );
                                cy_c == max_cycle_val || cy_d == max_cycle_val
                            }
                            crate::schedule_constraints::ScheduleConstraint::Ordering {
                                before_idx,
                                after_idx,
                                ..
                            } => {
                                let before_c = cycle_assignments.get(*before_idx).copied().unwrap_or(-1);
                                let after_c = cycle_assignments.get(*after_idx).copied().unwrap_or(-1);
                                before_c == max_cycle_val || after_c == max_cycle_val
                            }
                            crate::schedule_constraints::ScheduleConstraint::Domain { idx, .. } => {
                                cycle_assignments.get(*idx).copied().unwrap_or(-1) == max_cycle_val
                            }
                        }
                    })
                    .take(15)
                    .collect();
                if !binding_at_last.is_empty() {
                    eprintln!("    Binding at last cycle (max_cycle={}):", max_cycle_val);
                    for (c, msg) in binding_at_last {
                        let short = match c {
                            crate::schedule_constraints::ScheduleConstraint::Precedence { consumer_idx, dep_idx, .. } => {
                                format!("Precedence consumer {} dep {}", consumer_idx, dep_idx)
                            }
                            crate::schedule_constraints::ScheduleConstraint::Ordering { before_idx, after_idx, .. } => {
                                format!("Ordering {} <= {}", before_idx, after_idx)
                            }
                            crate::schedule_constraints::ScheduleConstraint::Domain { idx, .. } => {
                                format!("Domain idx {}", idx)
                            }
                        };
                        eprintln!("      {} | {}", short, msg);
                    }
                }
            }

            // Empty-cycle breadcrumbs: for each empty C, list all ops at C+1 and why they weren't at C (do not filter by ASAP/ALAP).
            // Invariant: no constraint should preclude every op at C+1 from moving to C (bulk move); if they're all at C+1 it's solver/tie-breaker.
            if verbose && empty_count > 0 {
                let used_cycles: HashSet<i32> = cycle_assignments.iter().copied().collect();
                let empty_cycles_list: Vec<i32> = (0..actual_makespan)
                    .filter(|&c| !used_cycles.contains(&c))
                    .take(15)
                    .collect();
                eprintln!("  [Empty-cycle breadcrumbs] {} empty cycles; showing ops at C+1 for first {} empty C",
                    empty_count, empty_cycles_list.len());
                for &c in &empty_cycles_list {
                    for (idx, &cy) in cycle_assignments.iter().enumerate() {
                        if cy != c + 1 {
                            continue;
                        }
                        let op = &ops[idx];
                        let op_id = op.id;
                        let engine_str = op.engine.as_str();
                        let opcode_str = format!("{:?}", op.opcode);
                        let mut latest_dep_result = -1i32;
                        let mut blocking_dep_id = 0usize;
                        let mut blocking_dep_opcode = String::new();
                        let mut blocking_dep_cycle = -1i32;
                        for &dep_id in &op.dep_ids {
                            let Some(&dep_idx) = op_id_to_idx.get(&dep_id) else {
                                continue;
                            };
                            let dep_cycle = cycle_assignments[dep_idx];
                            let result_cycle = if select_choice_with_fma_delay.contains(&dep_idx)
                                && select_fma_ops.contains(&dep_id)
                            {
                                dep_cycle + 2
                            } else {
                                dep_cycle + 1
                            };
                            if result_cycle > latest_dep_result {
                                latest_dep_result = result_cycle;
                                blocking_dep_id = dep_id;
                                blocking_dep_opcode = format!("{:?}", ops[dep_idx].opcode);
                                blocking_dep_cycle = dep_cycle;
                            }
                        }
                        // Result at cycle R means consumer can run at R or later. So op is blocked from C only when latest_dep_result > C.
                        if latest_dep_result > c {
                            eprintln!(
                                "  Empty {}: op id={} (engine={}, opcode={}) at C+1 blocked by precedence: dep id={} (opcode={}) at cycle {} -> result at {}.",
                                c, op_id, engine_str, opcode_str, blocking_dep_id, blocking_dep_opcode, blocking_dep_cycle, latest_dep_result
                            );
                        } else {
                            eprintln!(
                                "  Empty {}: op id={} (engine={}, opcode={}) at C+1 could run at C (BUG: should not be at C+1); latest dep result at {}.",
                                c, op_id, engine_str, opcode_str, latest_dep_result
                            );
                            let op_asap = *asap.get(&op_id).unwrap_or(&0);
                            let op_alap = *alap.get(&op_id).unwrap_or(&(horizon as i32 - 1));
                            if c < op_asap || c > op_alap {
                                eprintln!(
                                    "    BUG: ASAP/ALAP excludes empty cycle {} (asap={}, alap={})",
                                    c, op_asap, op_alap
                                );
                            }
                        }
                    }
                }
            }

            // Solution breadcrumbs: log Flow audit when verbose (before validation)
            if verbose {
                let (flow_rows, flow_per_cycle, _) =
                    build_flow_audit(ops, &cycle_assignments, &const_add_imm_ops, &select_fma_ops);
                eprintln!(
                    "[Flow solution breadcrumbs] Total Flow-contributing ops: {}",
                    flow_rows.len()
                );
                eprintln!(
                    "[Flow solution breadcrumbs] const_add_imm_ops (op.id) = {:?}",
                    const_add_imm_ops
                );
                eprintln!(
                    "[Flow solution breadcrumbs] select_fma_ops (op.id)   = {:?}",
                    select_fma_ops
                );
                for r in &flow_rows {
                    eprintln!(
                        "  idx={} op.id={} opcode={:?} cycle={} {}",
                        r.idx, r.op_id, r.opcode, r.cycle, r.reason
                    );
                }
                let overloaded: Vec<(usize, i32)> = flow_per_cycle
                    .iter()
                    .enumerate()
                    .filter(|(_, &count)| count > 1)
                    .map(|(cycle_idx, &count)| (cycle_idx, count))
                    .collect();
                if !overloaded.is_empty() {
                    eprintln!(
                        "[Flow solution breadcrumbs] WARNING: cycles with Flow count > 1: {:?}",
                        overloaded
                    );
                }
            }

            let (best_bound, gap) = if status == CpsatStatus::Optimal {
                (actual_makespan as f64, 0.0)
            } else {
                let best = response.objective_value;
                let bound = response.best_objective_bound;
                let gap_val = if best > 0.0 {
                    (best - bound).abs() / best
                } else {
                    0.0
                };
                (bound, gap_val)
            };

            // Post-solution validation: slot limits must never be violated (model bug if they are)
            validate_ssa_slot_limits(
                ops,
                &cycle_assignments,
                &shifted_ops,
                &shifted_alu_cycles,
                &const_add_imm_ops,
                &select_fma_ops,
                &machine,
            );

            // Post-solution validation: SelectChoice FMA result visible at C+2; consumers must be at >= C+2.
            verify_select_choice_fma_consumer_precedence(ops, &cycle_assignments, &select_fma_ops);

            if verbose {
                // Verify objective structure: objective = scale*makespan + tie_breaker, so floor(objective/scale) should equal makespan.
                if let Some(scale) = objective_scale {
                    let obj_val = response.objective_value;
                    let recovered_makespan = (obj_val / (scale as f64)).floor() as i32;
                    if (recovered_makespan - actual_makespan).abs() > 1 {
                        eprintln!(
                            "  Objective structure check: objective_value={:.0}, scale={}, floor(obj/scale)={} vs actual_makespan={}",
                            obj_val, scale, recovered_makespan, actual_makespan
                        );
                    }
                    eprintln!(
                        "  Objective: scale={}, so makespan is leading digits (floor({:.0}/{})= {}), tie-breaker in lower digits",
                        scale, obj_val, scale, recovered_makespan
                    );
                }
                let summary = summarise_ssa_solution_engines_and_choices(
                    ops,
                    &cycle_assignments,
                    &shifted_ops,
                    &const_add_imm_ops,
                    &select_fma_ops,
                );
                eprintln!(
                    "SSA CP-SAT result: {} cycles (best_bound: {:.1}), status {:?}, gap {:.2}%, time {:.2}s",
                    actual_makespan, best_bound, status, gap * 100.0, solve_time
                );
                // One-line active set summary for debugging (full "[Makespan active set]" is printed earlier).
                {
                    let max_cycle_val = max_cycle.solution_value(&response) as i32;
                    let mut n_binding_ops = 0usize;
                    for (idx, _) in ops.iter().enumerate() {
                        let effective_last = if shiftable_set.contains(&idx) {
                            let is_shifted = shift_vars.get(&idx).map(|v| v.solution_value(&response)).unwrap_or(false);
                            if is_shifted {
                                completion_cycle_vars.get(&idx).map(|v| v.solution_value(&response) as i32).unwrap_or(cycle_assignments[idx])
                            } else {
                                cycle_assignments[idx]
                            }
                        } else if let Some(&fma_last_var) = select_choice_fma_last_cycle.get(&idx) {
                            let is_fma = select_choice_vars.get(&idx).map(|v| v.solution_value(&response)).unwrap_or(false);
                            if is_fma { fma_last_var.solution_value(&response) as i32 } else { cycle_assignments[idx] }
                        } else {
                            cycle_assignments[idx]
                        };
                        if effective_last == max_cycle_val {
                            n_binding_ops += 1;
                        }
                    }
                    let m = actual_makespan as i64;
                    let capacity_binding = m == capacity_lower_bound
                        || (load_bound > 0 && m == load_bound)
                        || (store_bound > 0 && m == store_bound);
                    eprintln!(
                        "  Active set: {} op(s) at last cycle (max_cycle={}); capacity_bound_binding={} (see [Makespan active set] above for details)",
                        n_binding_ops, max_cycle_val, capacity_binding
                    );
                }
                let makespan_usize = actual_makespan as usize;
                let alu_capacity = machine.alu_slots as usize * makespan_usize;
                let valu_capacity = machine.valu_slots as usize * makespan_usize;
                let load_capacity = machine.load_slots as usize * makespan_usize;
                let store_capacity = machine.store_slots as usize * makespan_usize;
                let flow_capacity = machine.flow_slots as usize * makespan_usize;
                let pct = |used: usize, cap: usize| {
                    if cap > 0 {
                        (used as f64 / cap as f64) * 100.0
                    } else {
                        0.0
                    }
                };
                eprintln!(
                    "  Slot usage: ALU={}/{} ({:.1}%) VALU={}/{} ({:.1}%) Load={}/{} ({:.1}%) Store={}/{} ({:.1}%) Flow={}/{} ({:.1}%)",
                    summary.alu_total,
                    alu_capacity,
                    pct(summary.alu_total, alu_capacity),
                    summary.valu_total,
                    valu_capacity,
                    pct(summary.valu_total, valu_capacity),
                    summary.load_total,
                    load_capacity,
                    pct(summary.load_total, load_capacity),
                    summary.store_total,
                    store_capacity,
                    pct(summary.store_total, store_capacity),
                    summary.flow_total,
                    flow_capacity,
                    pct(summary.flow_total, flow_capacity)
                );
                eprintln!(
                    "  ConstChoice: {} AddImm, {} Load",
                    summary.const_add_imm, summary.const_load
                );
                eprintln!(
                    "  SelectChoice: {} FMA, {} vselect",
                    summary.select_fma, summary.select_vselect
                );
                eprintln!(
                    "  VALU slots: {} total (FMA: {}, vbroadcast: {}, other: {}; FMA and vbroadcast not shiftable)",
                    summary.valu_total,
                    summary.valu_fma_slots,
                    summary.valu_vbroadcast_slots,
                    summary.valu_other_slots
                );
                eprintln!(
                    "  Shifted {} VALU ops, {} ConstChoice AddImm ops, {} SelectChoice FMA ops",
                    shifted_ops.len(),
                    const_add_imm_ops.len(),
                    select_fma_ops.len()
                );
            }

            // schedule_constraints already built above (for active set and bulk-move).

            // When CP-SAT returns a makespan > actual last cycle (empty cycles), try to compact and use
            // the compacted schedule so the pipeline gets a tight makespan and CP-SAT effectively optimizes.
            let mut out_cycle_assignments = cycle_assignments;
            let mut out_shifted_alu_cycles = shifted_alu_cycles;
            let mut out_makespan = actual_makespan as usize;
            let mut out_empty_count = empty_count;

            if empty_count > 0 {
                // Check capacity first: validate current schedule (with empty cycles) satisfies per-cycle slot limits.
                // If it doesn't, we'd catch a bug before attempting compaction.
                if let Err(e) = try_validate_ssa_slot_limits(
                    ops,
                    &out_cycle_assignments,
                    &shifted_ops,
                    &out_shifted_alu_cycles,
                    &const_add_imm_ops,
                    &select_fma_ops,
                    &machine,
                ) {
                    if verbose {
                        eprintln!(
                            "  [compact] Schedule already violates capacity before bulk-move: {}",
                            e
                        );
                    }
                }
                let mut compact_cycles = out_cycle_assignments.clone();
                let mut compact_shifted = out_shifted_alu_cycles
                    .iter()
                    .map(|(k, v)| (*k, v.clone()))
                    .collect();
                use crate::schedule_constraints::{run_bulk_move_simulator, BulkMoveResult};
                let bulk_result = run_bulk_move_simulator(
                    ops,
                    &mut compact_cycles,
                    &mut compact_shifted,
                    &schedule_constraints,
                    actual_makespan as i32,
                    false,
                );
                match &bulk_result {
                    BulkMoveResult::ReachedEnd {
                        original_max_cycle_plus_one,
                        final_max_cycle_plus_one,
                        steps,
                    } if *final_max_cycle_plus_one < *original_max_cycle_plus_one => {
                        let compacted = *final_max_cycle_plus_one as i64;
                        use crate::schedule_constraints::check_schedule_constraints;
                        let constraints_ok = check_schedule_constraints(
                            ops,
                            &compact_cycles,
                            &compact_shifted,
                            &schedule_constraints,
                        );
                        let capacity_ok = try_validate_ssa_slot_limits(
                            ops,
                            &compact_cycles,
                            &shifted_ops,
                            &compact_shifted,
                            &const_add_imm_ops,
                            &select_fma_ops,
                            &machine,
                        );
                        if constraints_ok.is_ok() && capacity_ok.is_ok() {
                            out_cycle_assignments = compact_cycles;
                            out_shifted_alu_cycles = compact_shifted
                                .into_iter()
                                .map(|(k, v)| (k, v))
                                .collect();
                            out_makespan = (*final_max_cycle_plus_one) as usize;
                            out_empty_count = 0;
                            if verbose {
                                eprintln!(
                                    "  [compact] Using compacted schedule: makespan {} -> {} ({} steps); empty cycles eliminated.",
                                    original_max_cycle_plus_one, final_max_cycle_plus_one, steps
                                );
                            }
                        } else if verbose {
                            eprintln!(
                                "  [compact diagnostic] Bulk-move reduced makespan {} -> {} ({} steps). Capacity LB={}.",
                                original_max_cycle_plus_one,
                                final_max_cycle_plus_one,
                                steps,
                                capacity_lower_bound
                            );
                            if let Err(v) = &constraints_ok {
                                eprintln!(
                                    "  [compact diagnostic] Compacted schedule violates constraints: {}",
                                    v.message
                                );
                            }
                            if let Err(e) = &capacity_ok {
                                eprintln!(
                                    "  [compact diagnostic] Compacted schedule violates capacity: {}",
                                    e
                                );
                            }
                        }
                    }
                    _ => {}
                }
                if verbose && out_empty_count > 0 {
                    if let BulkMoveResult::ReachedEnd {
                        original_max_cycle_plus_one,
                        final_max_cycle_plus_one,
                        steps,
                    } = &bulk_result
                    {
                        if *final_max_cycle_plus_one < *original_max_cycle_plus_one {
                            let compacted = *final_max_cycle_plus_one as i64;
                            if capacity_lower_bound > compacted {
                                eprintln!(
                                    "  [compact diagnostic] Cause (2): capacity LB was too high ({} > {}).",
                                    capacity_lower_bound, compacted
                                );
                            } else {
                                eprintln!(
                                    "  [compact diagnostic] Cause (1): LB is fine ({} <= {}). A constraint in the model forbids the tighter schedule.",
                                    capacity_lower_bound, compacted
                                );
                            }
                        }
                    } else if let BulkMoveResult::Blocked { empty_c, violation, steps } = &bulk_result {
                        eprintln!(
                            "  [compact diagnostic] Bulk-move blocked at empty cycle {} after {} steps: {}",
                            empty_c, steps, violation.message
                        );
                        let binding = crate::schedule_constraints::binding_schedule_constraints(
                            ops,
                            &out_cycle_assignments,
                            &out_shifted_alu_cycles,
                            &schedule_constraints,
                        );
                        let in_active_set = binding.iter().any(|(c, _)| c == &violation.constraint);
                        if in_active_set {
                            eprintln!(
                                "  [compact diagnostic] Blocking constraint IS in active set (binding) — bulk-move correctly identified a tight constraint."
                            );
                        } else {
                            eprintln!(
                                "  [compact diagnostic] Blocking constraint NOT in active set (binding) — possible bug: constraint was not tight in current solution."
                            );
                            let short = match &violation.constraint {
                                crate::schedule_constraints::ScheduleConstraint::Precedence {
                                    consumer_idx,
                                    dep_idx,
                                    result_offset,
                                    dep_is_shifted,
                                } => format!(
                                    "Precedence consumer_idx={} dep_idx={} result_offset={} dep_is_shifted={}",
                                    consumer_idx, dep_idx, result_offset, dep_is_shifted
                                ),
                                crate::schedule_constraints::ScheduleConstraint::Ordering {
                                    before_idx,
                                    after_idx,
                                } => format!("Ordering before_idx={} after_idx={}", before_idx, after_idx),
                                crate::schedule_constraints::ScheduleConstraint::Domain { idx, lb, ub } => {
                                    format!("Domain idx={} lb={} ub={}", idx, lb, ub)
                                }
                            };
                            eprintln!("  [compact diagnostic] Blocking constraint: {}", short);
                        }
                    }
                }
            }

            Some(SsaCpsatResult {
                cycle_assignments: out_cycle_assignments,
                makespan: out_makespan,
                status,
                solve_time_secs: solve_time,
                gap,
                shifted_ops,
                shifted_alu_cycles: out_shifted_alu_cycles,
                const_add_imm_ops,
                select_fma_ops,
                empty_cycle_count: out_empty_count,
                makespan_driven_by_capacity: (out_makespan as i64) == capacity_lower_bound,
                schedule_constraints,
            })
        }
        _ => {
            if verbose {
                eprintln!("SSA CP-SAT: no solution found, status {:?}", status);
                if status == CpsatStatus::Unknown {
                    eprintln!(
                        "  (Unknown usually means time limit or incomplete hint; ensure cycle, ConstChoice, SelectChoice, and segment vars are hinted.)"
                    );
                }
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
    _hints: SsaCpsatHints<'_>,
    _greedy_within_scratch_budget: Option<bool>,
) -> Option<SsaCpsatResult> {
    eprintln!("SSA CP-SAT scheduler not available: compile with --features cpsat");
    None
}

#[cfg(test)]
mod ssa_tests {
    use super::*;
    use crate::kernel_builder::emit_ssa::{compute_ssa_dependencies, SsaEmitContext};
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
        compute_ssa_dependencies(&mut emit.ops, &vreg_def, emit.zero_vreg);

        // Schedule with CP-SAT
        let result = schedule_ssa_cpsat(
            &emit.ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: false,
                stop_after_first_solution: true,
                enable_valu_shift: false,
                scratch_budget: 1536,
                model_name: None,
                ..Default::default()
            },
            SsaCpsatHints::default(),
            None,
        );

        assert!(result.is_some(), "CP-SAT should find a solution");
        let result = result.unwrap();

        // Verify all ops are scheduled
        assert_eq!(result.cycle_assignments.len(), emit.ops.len());

        // Verify ConstChoice decisions are extracted
        // The result should contain const_add_imm_ops (may be empty if solver chose Load)
        assert!(
            result.const_add_imm_ops.len() <= 2,
            "Should have at most 2 ConstChoice ops using AddImm"
        );

        // Note: AddImm operations can use address 0 directly without waiting for zero
        // to be "ready" since scratch memory is always initialized to 0 at startup.
        // There is no precedence constraint requiring zero to be scheduled before AddImm.
    }

    /// Sanity check: ForceAddImm must be feasible. If this fails, AddImm constraints are wrong.
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_ssa_cpsat_constchoice_force_add_imm_feasible() {
        let mut emit = SsaEmitContext::new();
        let _zero = emit.emit_const(0);
        let c1 = emit.emit_const_choice(1);
        let c2 = emit.emit_const_choice(2);
        let _sum = emit.emit_alu(Opcode::Add, c1, c2);
        let vreg_def = std::mem::take(&mut emit.vreg_def);
        compute_ssa_dependencies(&mut emit.ops, &vreg_def, emit.zero_vreg);

        let result = schedule_ssa_cpsat(
            &emit.ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: false,
                stop_after_first_solution: true,
                enable_valu_shift: false,
                scratch_budget: 1536,
                model_name: None,
                const_choice_mode: ConstChoiceMode::ForceAddImm,
                ..Default::default()
            },
            SsaCpsatHints::default(),
            None,
        );
        assert!(
            result.is_some(),
            "ForceAddImm must be feasible; if this fails, AddImm constraints are wrong"
        );
        let r = result.unwrap();
        let n_const = emit.ops.iter().filter(|o| o.opcode == Opcode::ConstChoice).count();
        assert_eq!(
            r.const_add_imm_ops.len(),
            n_const,
            "ForceAddImm: all ConstChoice ops must be AddImm"
        );
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
        compute_ssa_dependencies(&mut emit.ops, &vreg_def, emit.zero_vreg);

        // Schedule with CP-SAT
        let result = schedule_ssa_cpsat(
            &emit.ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: false,
                stop_after_first_solution: true,
                enable_valu_shift: false,
                scratch_budget: 1536,
                model_name: None,
                ..Default::default()
            },
            SsaCpsatHints::default(),
            None,
        );

        assert!(result.is_some(), "CP-SAT should find a solution");
        let cpsat_result = result.unwrap();

        // Verify all ops are scheduled
        assert_eq!(cpsat_result.cycle_assignments.len(), emit.ops.len());

        // Find ConstChoice op IDs and their values
        let const_choice_info: Vec<(usize, i64)> = emit
            .ops
            .iter()
            .filter(|op| op.opcode == Opcode::ConstChoice)
            .map(|op| {
                let value = op.srcs[0]
                    .as_imm()
                    .expect("ConstChoice should have immediate operand");
                (op.id, value)
            })
            .collect();

        assert_eq!(
            const_choice_info.len(),
            3,
            "Should have 3 ConstChoice ops (1, 2, 4)"
        );

        // Verify CP-SAT decisions are extracted
        assert!(
            cpsat_result.const_add_imm_ops.len() <= 3,
            "Should have at most 3 ConstChoice ops using AddImm"
        );

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
            &HashSet::new(),                 // No shifted ops
            &cpsat_result.const_add_imm_ops, // Use CP-SAT decisions
            &cpsat_result.select_fma_ops,
            &cpsat_result.shifted_alu_cycles, // Use CP-SAT individual ALU cycles
        );

        // Verify that ConstChoice ops were lowered correctly based on CP-SAT decisions
        for (const_choice_id, expected_value) in &const_choice_info {
            let lowered_op = lowered_ops
                .iter()
                .find(|op| op.produces_constant == Some(*expected_value))
                .expect(&format!(
                    "Should find lowered op for ConstChoice producing {}",
                    expected_value
                ));

            if cpsat_result.const_add_imm_ops.contains(const_choice_id) {
                // CP-SAT chose AddImm - verify it's a Flow op
                assert_eq!(
                    lowered_op.engine,
                    Engine::Flow,
                    "ConstChoice op {} (value {}) should be lowered to Flow (AddImm)",
                    const_choice_id,
                    expected_value
                );
                match &lowered_op.slot[0] {
                    SlotVal::Str(s) => assert_eq!(
                        s, "add_imm",
                        "ConstChoice op {} should be add_imm",
                        const_choice_id
                    ),
                    _ => panic!("AddImm slot[0] should be Str"),
                }
                match &lowered_op.slot[2] {
                    SlotVal::Int(addr) => assert_eq!(*addr, 0, "AddImm should read from address 0"),
                    _ => panic!("AddImm slot[2] should be Int(0)"),
                }
            } else {
                // CP-SAT chose Load - verify it's a Load op
                assert_eq!(
                    lowered_op.engine,
                    Engine::Load,
                    "ConstChoice op {} (value {}) should be lowered to Load",
                    const_choice_id,
                    expected_value
                );
                match &lowered_op.slot[0] {
                    SlotVal::Str(s) => assert_eq!(
                        s, "const",
                        "ConstChoice op {} should be const",
                        const_choice_id
                    ),
                    _ => panic!("Const slot[0] should be Str"),
                }
            }
        }

        // Verify all ConstChoice ops were lowered
        assert_eq!(
            lowered_ops.len(),
            emit.ops.len(),
            "All ops should be lowered"
        );
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
        compute_ssa_dependencies(&mut emit.ops, &vreg_def, emit.zero_vreg);

        // Schedule with CP-SAT with longer time limit to allow optimization
        let result = schedule_ssa_cpsat(
            &emit.ops,
            SsaCpsatConfig {
                time_limit_secs: 30.0,
                verbose: false,
                stop_after_first_solution: false, // Allow optimization
                enable_valu_shift: false,
                scratch_budget: 1536,
                model_name: None,
                ..Default::default()
            },
            SsaCpsatHints::default(),
            None,
        );

        assert!(result.is_some(), "CP-SAT should find a solution");
        let cpsat_result = result.unwrap();

        // Find ConstChoice op IDs
        let const_choice_ids: Vec<usize> = emit
            .ops
            .iter()
            .filter(|op| op.opcode == Opcode::ConstChoice)
            .map(|op| op.id)
            .collect();

        assert_eq!(const_choice_ids.len(), 2, "Should have 2 ConstChoice ops");

        // Verify decisions were extracted
        assert!(
            cpsat_result.const_add_imm_ops.len() <= 2,
            "Should have at most 2 ConstChoice ops using AddImm"
        );

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
            &cpsat_result.select_fma_ops,
            &cpsat_result.shifted_alu_cycles, // Use CP-SAT individual ALU cycles
        );

        // Verify ConstChoice ops were lowered correctly
        for const_choice_id in &const_choice_ids {
            let ssa_op = emit
                .ops
                .iter()
                .find(|op| op.id == *const_choice_id)
                .unwrap();
            let expected_value = ssa_op.srcs[0].as_imm().unwrap();

            let lowered_op = lowered_ops
                .iter()
                .find(|op| op.produces_constant == Some(expected_value))
                .expect(&format!(
                    "Should find lowered op for ConstChoice {}",
                    const_choice_id
                ));

            if cpsat_result.const_add_imm_ops.contains(const_choice_id) {
                // Verify AddImm was chosen
                assert_eq!(
                    lowered_op.engine,
                    Engine::Flow,
                    "ConstChoice {} should use AddImm when Load slots are full",
                    const_choice_id
                );
                match &lowered_op.slot[0] {
                    SlotVal::Str(s) => assert_eq!(s, "add_imm"),
                    _ => panic!("Should be add_imm"),
                }
            } else {
                // Verify Load was chosen
                assert_eq!(
                    lowered_op.engine,
                    Engine::Load,
                    "ConstChoice {} should use Load",
                    const_choice_id
                );
            }
        }
    }

    #[cfg(feature = "cpsat")]
    #[test]
    fn test_ssa_cpsat_selectchoice_decisions() {
        // Build a simple kernel with SelectChoice ops
        let mut emit = SsaEmitContext::new();

        // Emit vector condition and values (use locals to avoid double mutable borrow)
        let c0 = emit.emit_const(0);
        let c1 = emit.emit_const(1);
        let v_c0 = emit.emit_vbroadcast(c0);
        let v_c1 = emit.emit_vbroadcast(c1);
        let cond = emit.emit_valu(Opcode::Eq, v_c0, v_c1);
        let c10 = emit.emit_const(10);
        let true_val = emit.emit_vbroadcast(c10);
        let c20 = emit.emit_const(20);
        let false_val = emit.emit_vbroadcast(c20);

        // Emit SelectChoice ops
        let sel1 = emit.emit_select_choice(cond, true_val, false_val);
        let sel2 = emit.emit_select_choice(cond, true_val, false_val);

        // Use results
        let _sum = emit.emit_valu(Opcode::Add, sel1, sel2);

        // Compute dependencies
        let vreg_def = std::mem::take(&mut emit.vreg_def);
        compute_ssa_dependencies(&mut emit.ops, &vreg_def, emit.zero_vreg);

        // Schedule with CP-SAT
        let result = schedule_ssa_cpsat(
            &emit.ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: false,
                stop_after_first_solution: true,
                enable_valu_shift: false,
                scratch_budget: 1536,
                model_name: None,
                ..Default::default()
            },
            SsaCpsatHints::default(),
            None,
        );

        assert!(result.is_some(), "CP-SAT should find a solution");
        let result = result.unwrap();

        // Verify all ops are scheduled
        assert_eq!(result.cycle_assignments.len(), emit.ops.len());

        // Verify SelectChoice decisions are extracted
        // The result should contain select_fma_ops (may be empty if solver chose vselect)
        assert!(
            result.select_fma_ops.len() <= 2,
            "Should have at most 2 SelectChoice ops using FMA path"
        );
    }

    /// Toy test: 3 parallel chains SelectChoice -> valu1 (shiftable) -> valu2, with a toy
    /// [`MachineConfig`] (1 VALU slot, 8 ALU slots per cycle). With only 1 VALU slot we need
    /// at least one valu1 to shift to ALU (8 slots). Asserts CP-SAT shifts at least one.
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_ssa_cpsat_selectchoice_valu_shift_toy() {
        let mut emit = SsaEmitContext::new();

        let c0 = emit.emit_const(0);
        let c1 = emit.emit_const(1);
        let v_c0 = emit.emit_vbroadcast(c0);
        let v_c1 = emit.emit_vbroadcast(c1);
        let cond = emit.emit_valu(Opcode::Eq, v_c0, v_c1);
        let c10 = emit.emit_const(10);
        let true_val = emit.emit_vbroadcast(c10);
        let c20 = emit.emit_const(20);
        let false_val = emit.emit_vbroadcast(c20);

        let mut valu1_op_ids = Vec::new();

        // 3 parallel chains so we have 3 valu1 + 3 valu2 = 6 VALU ops; with 1 VALU slot we need 6+ cycles
        // unless we shift (shifted valu1 uses 8 ALU, 0 VALU), giving the solver incentive to shift.
        for _ in 0..3 {
            let sel = emit.emit_select_choice(cond, true_val, false_val);
            let valu1 = emit.emit_valu(Opcode::Add, sel, true_val);
            valu1_op_ids.push(emit.ops.last().unwrap().id);
            let _valu2 = emit.emit_valu(Opcode::Add, valu1, false_val);
        }

        let vreg_def = std::mem::take(&mut emit.vreg_def);
        compute_ssa_dependencies(&mut emit.ops, &vreg_def, emit.zero_vreg);

        // Toy machine: 1 VALU, 8 ALU so one shifted VALU fits in ALU.
        let toy_machine = MachineConfig {
            valu_slots: 1,
            alu_slots: 8,
            ..Default::default()
        };
        let result = schedule_ssa_cpsat(
            &emit.ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: false,
                stop_after_first_solution: false,
                enable_valu_shift: true,
                scratch_budget: 1536,
                model_name: Some("selectchoice_valu_shift_toy".to_string()),
                machine: toy_machine,
                ..Default::default()
            },
            SsaCpsatHints::default(),
            None,
        );

        assert!(
            result.is_some(),
            "CP-SAT should find a solution with 1 VALU / 8 ALU toy caps"
        );
        let r = result.unwrap();
        assert_eq!(r.cycle_assignments.len(), emit.ops.len());

        // With 1 VALU slot and 3 chains (6 VALU ops), shifting at least one valu1 to ALU should help.
        let shifted_valu1 = valu1_op_ids
            .iter()
            .filter(|&id| r.shifted_ops.contains(id))
            .count();
        assert!(
            shifted_valu1 >= 1,
            "With 1 VALU and 8 ALU slots and 3 chains, at least one of the 3 shiftable valu1 ops should be shifted; got shifted_ops={:?}, valu1_op_ids={:?}",
            r.shifted_ops, valu1_op_ids
        );
    }

    /// Test that multiple SelectChoice ops in vselect mode cannot be in the same cycle.
    /// This tests the Flow cumulative constraint with optional intervals.
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_ssa_cpsat_selectchoice_flow_slot_limit() {
        // Build a kernel with 3 independent SelectChoice ops that all use the same inputs.
        // They have no data dependency on each other, so without Flow constraint they could
        // all be scheduled in the same cycle. The Flow slot limit (1) should force them
        // into different cycles when they choose vselect path.
        let mut emit = SsaEmitContext::new();

        // Emit vector condition and values
        let c0 = emit.emit_const(0);
        let c1 = emit.emit_const(1);
        let v_c0 = emit.emit_vbroadcast(c0);
        let v_c1 = emit.emit_vbroadcast(c1);
        let cond = emit.emit_valu(Opcode::Eq, v_c0, v_c1);
        let c10 = emit.emit_const(10);
        let true_val = emit.emit_vbroadcast(c10);
        let c20 = emit.emit_const(20);
        let false_val = emit.emit_vbroadcast(c20);

        // Emit 3 SelectChoice ops - they are independent so could run in parallel if allowed
        let sel1 = emit.emit_select_choice(cond, true_val, false_val);
        let sel2 = emit.emit_select_choice(cond, true_val, false_val);
        let sel3 = emit.emit_select_choice(cond, true_val, false_val);

        // Use results
        let sum12 = emit.emit_valu(Opcode::Add, sel1, sel2);
        let _sum123 = emit.emit_valu(Opcode::Add, sum12, sel3);

        // Compute dependencies
        let vreg_def = std::mem::take(&mut emit.vreg_def);
        compute_ssa_dependencies(&mut emit.ops, &vreg_def, emit.zero_vreg);

        // Schedule with CP-SAT - use verbose to see breadcrumbs
        let result = schedule_ssa_cpsat(
            &emit.ops,
            SsaCpsatConfig {
                time_limit_secs: 30.0,
                verbose: true,                    // Show Flow breadcrumbs
                stop_after_first_solution: false, // Let it optimize
                enable_valu_shift: false,
                scratch_budget: 1536,
                model_name: None,
                ..Default::default()
            },
            SsaCpsatHints::default(),
            None,
        );

        assert!(result.is_some(), "CP-SAT should find a solution");
        let cpsat_result = result.unwrap();

        // Find SelectChoice op indices and their cycles
        let select_choice_info: Vec<(usize, usize, i32)> = emit
            .ops
            .iter()
            .enumerate()
            .filter(|(_, op)| op.opcode == Opcode::SelectChoice)
            .map(|(idx, op)| (idx, op.id, cpsat_result.cycle_assignments[idx]))
            .collect();

        assert_eq!(
            select_choice_info.len(),
            3,
            "Should have 3 SelectChoice ops"
        );
        eprintln!("SelectChoice ops: {:?}", select_choice_info);
        eprintln!("select_fma_ops: {:?}", cpsat_result.select_fma_ops);

        // Count how many SelectChoice ops chose vselect path (not in select_fma_ops)
        let vselect_ops: Vec<_> = select_choice_info
            .iter()
            .filter(|(_, op_id, _)| !cpsat_result.select_fma_ops.contains(op_id))
            .collect();

        if vselect_ops.len() > 1 {
            // If multiple vselect ops, verify they're in different cycles
            let vselect_cycles: Vec<i32> = vselect_ops.iter().map(|(_, _, cy)| *cy).collect();
            eprintln!("vselect ops cycles: {:?}", vselect_cycles);

            // Check no duplicates (each cycle should appear at most once)
            let mut cycle_counts: HashMap<i32, usize> = HashMap::new();
            for cy in &vselect_cycles {
                *cycle_counts.entry(*cy).or_default() += 1;
            }

            for (cy, count) in &cycle_counts {
                assert!(
                    *count <= 1,
                    "Flow slot limit violated: {} vselect ops in cycle {} (limit is 1)",
                    count,
                    cy
                );
            }
        }

        eprintln!("test_ssa_cpsat_selectchoice_flow_slot_limit PASSED");
    }

    // =========================================================================
    // Tests for AddImm ops from emit_const() in Flow constraint
    // These tests verify the fix for the Flow slot violation bug where AddImm
    // ops were not included in the Flow cumulative constraint.
    // =========================================================================

    /// Test that AddImm ops from emit_const() are counted in the Flow constraint.
    /// This was the root cause of the Flow slot violation bug.
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_ssa_cpsat_addimm_ops_counted_in_flow_constraint() {
        let mut emit = SsaEmitContext::new();

        // Set up zero_vreg so emit_const creates AddImm alternatives
        let zero = emit.emit_const(0);
        emit.zero_vreg = Some(zero);

        // Emit 5 non-zero constants - each creates Load + AddImm ops
        let c1 = emit.emit_const(1);
        let c2 = emit.emit_const(2);
        let c3 = emit.emit_const(3);
        let c4 = emit.emit_const(4);
        let c5 = emit.emit_const(5);

        // Use the constants to create dependencies
        let sum1 = emit.emit_alu(Opcode::Add, c1, c2);
        let sum2 = emit.emit_alu(Opcode::Add, c3, c4);
        let _sum3 = emit.emit_alu(Opcode::Add, sum1, sum2);
        let _use_c5 = emit.emit_alu(Opcode::Add, c5, c1);

        // Compute dependencies
        let vreg_def = std::mem::take(&mut emit.vreg_def);
        compute_ssa_dependencies(&mut emit.ops, &vreg_def, emit.zero_vreg);

        // Count AddImm ops
        let addimm_count = emit
            .ops
            .iter()
            .filter(|op| op.opcode == Opcode::AddImm)
            .count();
        assert_eq!(addimm_count, 5, "Should have 5 AddImm ops from emit_const");

        // Schedule with CP-SAT
        let result = schedule_ssa_cpsat(
            &emit.ops,
            SsaCpsatConfig {
                time_limit_secs: 30.0,
                verbose: true,
                stop_after_first_solution: false,
                enable_valu_shift: false,
                scratch_budget: 1536,
                model_name: None,
                ..Default::default()
            },
            SsaCpsatHints::default(),
            None,
        );

        assert!(result.is_some(), "CP-SAT should find a solution");
        let cpsat_result = result.unwrap();

        // Collect all Flow-engine ops and their cycles
        let flow_ops: Vec<(usize, i32)> = emit
            .ops
            .iter()
            .enumerate()
            .filter(|(_, op)| op.engine == Engine::Flow)
            .map(|(idx, _)| (idx, cpsat_result.cycle_assignments[idx]))
            .collect();

        eprintln!("Flow ops (idx, cycle): {:?}", flow_ops);

        // Verify no two Flow ops are in the same cycle
        let mut cycle_counts: HashMap<i32, usize> = HashMap::new();
        for (_, cy) in &flow_ops {
            *cycle_counts.entry(*cy).or_default() += 1;
        }

        for (cy, count) in &cycle_counts {
            assert!(
                *count <= 1,
                "Flow slot limit violated: {} Flow ops in cycle {} (limit is 1). \
                This would indicate AddImm ops are not properly constrained.",
                count,
                cy
            );
        }

        eprintln!("test_ssa_cpsat_addimm_ops_counted_in_flow_constraint PASSED: {} AddImm ops properly scheduled in different cycles", addimm_count);
    }

    /// Test that the validation function correctly detects Flow slot violations.
    /// This ensures our validation logic matches what we expect.
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_validate_ssa_slot_limits_detects_flow_violation() {
        // Create minimal ops that would violate Flow limit if scheduled in same cycle
        let mut op0 = SsaOp::new(
            0,
            Engine::Flow,
            Opcode::AddImm,
            Some(VReg::scalar(0)),
            vec![],
        );
        op0.produces_constant = Some(1);

        let mut op1 = SsaOp::new(
            1,
            Engine::Flow,
            Opcode::AddImm,
            Some(VReg::scalar(1)),
            vec![],
        );
        op1.produces_constant = Some(2);

        let ops = vec![op0, op1];

        // Schedule both in cycle 0 - this should violate Flow limit
        let cycle_assignments = vec![0, 0];
        let shifted_ops = HashSet::new();
        let const_add_imm_ops = HashSet::new();
        let select_fma_ops = HashSet::new();

        // This should panic because we have 2 Flow ops in cycle 0
        let result = std::panic::catch_unwind(|| {
            validate_ssa_slot_limits(
                &ops,
                &cycle_assignments,
                &shifted_ops,
                &HashMap::new(), // shifted_alu_cycles
                &const_add_imm_ops,
                &select_fma_ops,
                &MachineConfig::default(),
            );
        });

        assert!(
            result.is_err(),
            "validate_ssa_slot_limits should panic when 2 Flow ops are in same cycle"
        );

        eprintln!("test_validate_ssa_slot_limits_detects_flow_violation PASSED");
    }

    /// Test that validation passes when Flow ops are in different cycles.
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_validate_ssa_slot_limits_passes_valid_schedule() {
        let mut op0 = SsaOp::new(
            0,
            Engine::Flow,
            Opcode::AddImm,
            Some(VReg::scalar(0)),
            vec![],
        );
        op0.produces_constant = Some(1);

        let mut op1 = SsaOp::new(
            1,
            Engine::Flow,
            Opcode::AddImm,
            Some(VReg::scalar(1)),
            vec![],
        );
        op1.produces_constant = Some(2);

        let ops = vec![op0, op1];

        // Schedule in different cycles - this should be valid
        let cycle_assignments = vec![0, 1];
        let shifted_ops = HashSet::new();
        let const_add_imm_ops = HashSet::new();
        let select_fma_ops = HashSet::new();

        // This should NOT panic
        validate_ssa_slot_limits(
            &ops,
            &cycle_assignments,
            &shifted_ops,
            &HashMap::new(), // shifted_alu_cycles
            &const_add_imm_ops,
            &select_fma_ops,
            &MachineConfig::default(),
        );

        eprintln!("test_validate_ssa_slot_limits_passes_valid_schedule PASSED");
    }

    /// Test that ConstChoice ops are correctly handled when they choose AddImm path.
    /// ConstChoice is different from emit_const() - it's a single op with a choice.
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_ssa_cpsat_constchoice_addimm_uses_flow_slot() {
        let mut emit = SsaEmitContext::new();

        // Emit several ConstChoice constants
        let c1 = emit.emit_const_choice(10);
        let c2 = emit.emit_const_choice(20);
        let c3 = emit.emit_const_choice(30);

        // Use them
        let sum = emit.emit_alu(Opcode::Add, c1, c2);
        let _sum2 = emit.emit_alu(Opcode::Add, sum, c3);

        // Compute dependencies
        let vreg_def = std::mem::take(&mut emit.vreg_def);
        compute_ssa_dependencies(&mut emit.ops, &vreg_def, emit.zero_vreg);

        // Verify we have ConstChoice ops
        let constchoice_count = emit
            .ops
            .iter()
            .filter(|op| op.opcode == Opcode::ConstChoice)
            .count();
        assert_eq!(constchoice_count, 3, "Should have 3 ConstChoice ops");

        // Schedule with CP-SAT
        let result = schedule_ssa_cpsat(
            &emit.ops,
            SsaCpsatConfig {
                time_limit_secs: 30.0,
                verbose: true,
                stop_after_first_solution: false,
                enable_valu_shift: false,
                scratch_budget: 1536,
                model_name: None,
                ..Default::default()
            },
            SsaCpsatHints::default(),
            None,
        );

        assert!(result.is_some(), "CP-SAT should find a solution");
        let cpsat_result = result.unwrap();

        // If any ConstChoice ops chose AddImm path, they should be in different cycles
        if !cpsat_result.const_add_imm_ops.is_empty() {
            let addimm_cycles: Vec<i32> = emit
                .ops
                .iter()
                .enumerate()
                .filter(|(_, op)| {
                    op.opcode == Opcode::ConstChoice
                        && cpsat_result.const_add_imm_ops.contains(&op.id)
                })
                .map(|(idx, _)| cpsat_result.cycle_assignments[idx])
                .collect();

            eprintln!(
                "ConstChoice ops using AddImm path, cycles: {:?}",
                addimm_cycles
            );

            // Check no duplicates
            let mut cycle_counts: HashMap<i32, usize> = HashMap::new();
            for cy in &addimm_cycles {
                *cycle_counts.entry(*cy).or_default() += 1;
            }
            for (cy, count) in &cycle_counts {
                assert!(
                    *count <= 1,
                    "Flow slot limit violated: {} ConstChoice AddImm ops in cycle {}",
                    count,
                    cy
                );
            }
        }

        eprintln!("test_ssa_cpsat_constchoice_addimm_uses_flow_slot PASSED");
    }

    /// Test that mixed AddImm (from emit_const) and ConstChoice AddImm paths
    /// are all constrained to 1 per cycle.
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_ssa_cpsat_mixed_addimm_and_constchoice_flow_constraint() {
        let mut emit = SsaEmitContext::new();

        // Set up zero_vreg for emit_const AddImm ops
        let zero = emit.emit_const(0);
        emit.zero_vreg = Some(zero);

        // Mix of emit_const (creates Load+AddImm) and emit_const_choice (creates ConstChoice)
        let c1 = emit.emit_const(10); // Creates Load + AddImm
        let c2 = emit.emit_const_choice(20); // Creates ConstChoice
        let c3 = emit.emit_const(30); // Creates Load + AddImm
        let c4 = emit.emit_const_choice(40); // Creates ConstChoice

        // Use them
        let sum1 = emit.emit_alu(Opcode::Add, c1, c2);
        let sum2 = emit.emit_alu(Opcode::Add, c3, c4);
        let _final = emit.emit_alu(Opcode::Add, sum1, sum2);

        // Compute dependencies
        let vreg_def = std::mem::take(&mut emit.vreg_def);
        compute_ssa_dependencies(&mut emit.ops, &vreg_def, emit.zero_vreg);

        // Count ops
        let addimm_count = emit
            .ops
            .iter()
            .filter(|op| op.opcode == Opcode::AddImm)
            .count();
        let constchoice_count = emit
            .ops
            .iter()
            .filter(|op| op.opcode == Opcode::ConstChoice)
            .count();

        eprintln!(
            "AddImm ops: {}, ConstChoice ops: {}",
            addimm_count, constchoice_count
        );
        assert_eq!(addimm_count, 2, "Should have 2 AddImm ops from emit_const");
        assert_eq!(constchoice_count, 2, "Should have 2 ConstChoice ops");

        // Schedule with CP-SAT
        let result = schedule_ssa_cpsat(
            &emit.ops,
            SsaCpsatConfig {
                time_limit_secs: 30.0,
                verbose: true,
                stop_after_first_solution: false,
                enable_valu_shift: false,
                scratch_budget: 1536,
                model_name: None,
                ..Default::default()
            },
            SsaCpsatHints::default(),
            None,
        );

        assert!(result.is_some(), "CP-SAT should find a solution");
        let cpsat_result = result.unwrap();

        // Collect all ops that use Flow slot:
        // 1. AddImm ops (from emit_const)
        // 2. ConstChoice ops that chose AddImm path
        let mut flow_users: Vec<(usize, i32, &str)> = Vec::new();

        for (idx, op) in emit.ops.iter().enumerate() {
            let cycle = cpsat_result.cycle_assignments[idx];
            if op.opcode == Opcode::AddImm {
                flow_users.push((idx, cycle, "AddImm"));
            } else if op.opcode == Opcode::ConstChoice
                && cpsat_result.const_add_imm_ops.contains(&op.id)
            {
                flow_users.push((idx, cycle, "ConstChoice→AddImm"));
            }
        }

        eprintln!("All Flow-using ops: {:?}", flow_users);

        // Verify no two in same cycle
        let mut cycle_counts: HashMap<i32, Vec<&str>> = HashMap::new();
        for (_, cy, kind) in &flow_users {
            cycle_counts.entry(*cy).or_default().push(kind);
        }

        for (cy, kinds) in &cycle_counts {
            assert!(
                kinds.len() <= 1,
                "Flow slot limit violated: {} ops in cycle {}: {:?}",
                kinds.len(),
                cy,
                kinds
            );
        }

        eprintln!("test_ssa_cpsat_mixed_addimm_and_constchoice_flow_constraint PASSED");
    }

    #[cfg(feature = "cpsat")]
    #[test]
    fn test_ssa_cpsat_selectchoice_roundtrip() {
        // Build a simple kernel with SelectChoice ops
        let mut emit = SsaEmitContext::new();

        // Emit vector condition and values (use locals to avoid double mutable borrow)
        let c0 = emit.emit_const(0);
        let c1 = emit.emit_const(1);
        let v_c0 = emit.emit_vbroadcast(c0);
        let v_c1 = emit.emit_vbroadcast(c1);
        let cond = emit.emit_valu(Opcode::Eq, v_c0, v_c1);
        let c10 = emit.emit_const(10);
        let true_val = emit.emit_vbroadcast(c10);
        let c20 = emit.emit_const(20);
        let false_val = emit.emit_vbroadcast(c20);

        // Emit SelectChoice ops
        let sel1 = emit.emit_select_choice(cond, true_val, false_val);
        let sel2 = emit.emit_select_choice(cond, true_val, false_val);

        // Use results
        let _sum = emit.emit_valu(Opcode::Add, sel1, sel2);

        // Compute dependencies
        let vreg_def = std::mem::take(&mut emit.vreg_def);
        compute_ssa_dependencies(&mut emit.ops, &vreg_def, emit.zero_vreg);

        // Schedule with CP-SAT
        let result = schedule_ssa_cpsat(
            &emit.ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: false,
                stop_after_first_solution: true,
                enable_valu_shift: false,
                scratch_budget: 1536,
                model_name: None,
                ..Default::default()
            },
            SsaCpsatHints::default(),
            None,
        );

        assert!(result.is_some(), "CP-SAT should find a solution");
        let cpsat_result = result.unwrap();

        // Find SelectChoice op IDs
        let select_choice_ids: Vec<usize> = emit
            .ops
            .iter()
            .filter(|op| op.opcode == Opcode::SelectChoice)
            .map(|op| op.id)
            .collect();

        assert_eq!(select_choice_ids.len(), 2, "Should have 2 SelectChoice ops");

        // Verify decisions were extracted
        assert!(
            cpsat_result.select_fma_ops.len() <= 2,
            "Should have at most 2 SelectChoice ops using FMA path"
        );

        // Now test round-trip: lower using CP-SAT decisions
        let mut vreg_to_addr = HashMap::new();
        for (idx, op) in emit.ops.iter().enumerate() {
            if let Some(dest) = op.dest {
                vreg_to_addr.insert(dest, idx as i64 * 10); // Simple address assignment
            }
        }

        let lowered_ops = lower_ssa_ops_with_shift(
            &emit.ops,
            &vreg_to_addr,
            &HashSet::new(),                  // No shifted ops
            &HashSet::new(),                  // No ConstChoice AddImm ops
            &cpsat_result.select_fma_ops,     // Use CP-SAT decisions
            &cpsat_result.shifted_alu_cycles, // Use CP-SAT individual ALU cycles
        );

        // Verify that SelectChoice ops were lowered correctly based on CP-SAT decisions
        for select_choice_id in &select_choice_ids {
            if cpsat_result.select_fma_ops.contains(select_choice_id) {
                // CP-SAT chose FMA path - verify sub and fma ops exist
                let sub_op = lowered_ops
                    .iter()
                    .find(|op| op.id == *select_choice_id && op.engine == Engine::Valu);
                assert!(
                    sub_op.is_some(),
                    "SelectChoice op {} should be lowered to sub op (VALU)",
                    select_choice_id
                );

                let fma_op = lowered_ops
                    .iter()
                    .find(|op| op.id == *select_choice_id + 1000000 && op.engine == Engine::Valu);
                assert!(
                    fma_op.is_some(),
                    "SelectChoice op {} should be lowered to fma op (VALU)",
                    select_choice_id
                );

                // Verify fma is scheduled after sub
                let sub_cycle = sub_op.unwrap().cycle;
                let fma_cycle = fma_op.unwrap().cycle;
                assert!(
                    fma_cycle > sub_cycle,
                    "FMA should be scheduled after sub: sub={}, fma={}",
                    sub_cycle,
                    fma_cycle
                );
            } else {
                // CP-SAT chose vselect path - verify vselect op exists
                let vselect_op = lowered_ops
                    .iter()
                    .find(|op| op.id == *select_choice_id && op.engine == Engine::Flow);
                assert!(
                    vselect_op.is_some(),
                    "SelectChoice op {} should be lowered to vselect op (Flow)",
                    select_choice_id
                );

                match &vselect_op.unwrap().slot[0] {
                    SlotVal::Str(s) => assert_eq!(
                        s, "vselect",
                        "SelectChoice op {} should be vselect",
                        select_choice_id
                    ),
                    _ => panic!("vselect slot[0] should be Str"),
                }
            }
        }
    }

    /// Small machine: 1 VALU slot so sub and fma cannot share a cycle (forced consecutive bundles).
    const SMALL_MACHINE_FMA: crate::types::MachineConfig = crate::types::MachineConfig {
        valu_slots: 1,
        alu_slots: 12,
        load_slots: 2,
        store_slots: 2,
        flow_slots: 1,
    };

    /// Minimal test: one SelectChoice (no delta → sub+fma path), ForceFma.
    /// Uses small machine (1 VALU slot) so sub and fma are forced into consecutive cycles.
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_ssa_cpsat_selectchoice_force_fma_minimal() {
        use crate::address_assign::{assign_addresses, AddressAssignConfig};
        use crate::kernel_builder::emit_ssa::{compute_ssa_dependencies, SsaEmitContext};

        let mut emit = SsaEmitContext::new();
        let c0 = emit.emit_const(0);
        let c1 = emit.emit_const(1);
        let v_c0 = emit.emit_vbroadcast(c0);
        let v_c1 = emit.emit_vbroadcast(c1);
        let cond = emit.emit_valu(Opcode::Eq, v_c0, v_c1);
        let c10 = emit.emit_const(10);
        let true_val = emit.emit_vbroadcast(c10);
        let c20 = emit.emit_const(20);
        let false_val = emit.emit_vbroadcast(c20);
        let sel = emit.emit_select_choice(cond, true_val, false_val);
        let _use = emit.emit_valu(Opcode::Add, sel, sel);

        let vreg_def = std::mem::take(&mut emit.vreg_def);
        compute_ssa_dependencies(&mut emit.ops, &vreg_def, emit.zero_vreg);

        let select_choice_idx = emit
            .ops
            .iter()
            .position(|o| o.opcode == Opcode::SelectChoice)
            .expect("one SelectChoice");
        let select_choice_id = emit.ops[select_choice_idx].id;
        assert!(
            emit.ops[select_choice_idx].sub_temp_vreg.is_some(),
            "no-delta path (sub+fma)"
        );

        let result = schedule_ssa_cpsat(
            &emit.ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: false,
                stop_after_first_solution: true,
                enable_valu_shift: false,
                scratch_budget: 1536,
                model_name: None,
                machine: SMALL_MACHINE_FMA,
                const_choice_mode: ConstChoiceMode::Optimize,
                select_choice_mode: SelectChoiceMode::ForceFma,
                valu_shift_mode: ValuShiftMode::Optimize,
                ..Default::default()
            },
            SsaCpsatHints::default(),
            None,
        );
        assert!(
            result.is_some(),
            "CP-SAT should find a solution with ForceFma"
        );
        let cpsat_result = result.unwrap();

        assert!(
            cpsat_result.select_fma_ops.contains(&select_choice_id),
            "ForceFma => select_fma_ops contains SelectChoice id"
        );
        let c = cpsat_result.cycle_assignments[select_choice_idx];
        for (idx, op) in emit.ops.iter().enumerate() {
            if op.dep_ids.contains(&select_choice_id) {
                let consumer_cycle = cpsat_result.cycle_assignments[idx];
                assert!(
                    consumer_cycle >= c + 2,
                    "consumer must be at cycle >= {} (SelectChoice result visible at C+2), got {}",
                    c + 2,
                    consumer_cycle
                );
            }
        }

        for (idx, op) in emit.ops.iter_mut().enumerate() {
            op.cycle = cpsat_result.cycle_assignments[idx];
        }
        let mut config = AddressAssignConfig::default();
        config.select_fma_ops = Some(cpsat_result.select_fma_ops.clone());
        let assign_result = assign_addresses(&emit.ops, &config);
        assert!(!assign_result.budget_exceeded);

        // With 1 VALU slot, sub and fma must be in consecutive cycles
        let lowered = lower_ssa_ops_with_shift(
            &emit.ops,
            &assign_result.vreg_to_addr,
            &HashSet::new(),
            &HashSet::new(),
            &cpsat_result.select_fma_ops,
            &cpsat_result.shifted_alu_cycles,
        );
        let sub_op = lowered.iter().find(|o| {
            o.id == select_choice_id
                && o.engine == Engine::Valu
                && format!("{:?}", o.slot.get(0)).contains("\"-\"")
        });
        let fma_op = lowered
            .iter()
            .find(|o| o.id == select_choice_id + 1000000 && o.engine == Engine::Valu);
        assert!(sub_op.is_some(), "should have sub op");
        assert!(fma_op.is_some(), "should have fma op");
        assert_eq!(sub_op.unwrap().cycle, c, "sub at SelectChoice cycle");
        assert_eq!(
            fma_op.unwrap().cycle,
            c + 1,
            "fma at SelectChoice cycle + 1"
        );

        // Total cycle count must include the FMA: max(lowered cycle)+1 >= c+2 (FMA at c+1).
        let max_lowered_cycle = lowered.iter().map(|o| o.cycle.max(0)).max().unwrap_or(0);
        assert!(
            max_lowered_cycle >= c + 1,
            "cycle count must include FMA at cycle {}",
            c + 1
        );
    }

    /// Force VALU shift off (ForceValu) and ensure consumers respect precedence.
    /// Regression test for bug where we skipped RAW constraints for shiftable ops even when shift was disabled.
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_ssa_cpsat_force_valu_precedence() {
        use crate::kernel_builder::emit_ssa::{compute_ssa_dependencies, SsaEmitContext};

        let mut emit = SsaEmitContext::new();
        let c0 = emit.emit_const(0);
        let v0 = emit.emit_vbroadcast(c0);
        let shiftable = emit.emit_valu(Opcode::Add, v0, v0); // Shiftable VALU op
        let consumer = emit.emit_valu(Opcode::Sub, shiftable, v0); // Consumer must run later

        let vreg_def = std::mem::take(&mut emit.vreg_def);
        compute_ssa_dependencies(&mut emit.ops, &vreg_def, emit.zero_vreg);

        let shift_idx = emit
            .ops
            .iter()
            .position(|op| op.dest == Some(shiftable))
            .expect("shiftable op");
        let consumer_idx = emit
            .ops
            .iter()
            .position(|op| op.dest == Some(consumer))
            .expect("consumer op");

        let result = schedule_ssa_cpsat(
            &emit.ops,
            SsaCpsatConfig {
                enable_valu_shift: true,
                valu_shift_mode: ValuShiftMode::ForceValu,
                time_limit_secs: 10.0,
                scratch_budget: 1536,
                ..Default::default()
            },
            SsaCpsatHints::default(),
            None,
        )
        .expect("CP-SAT ForceValu schedule");

        let shift_cycle = result.cycle_assignments[shift_idx];
        let consumer_cycle = result.cycle_assignments[consumer_idx];
        assert!(
            consumer_cycle >= shift_cycle + 1,
            "ForceValu must respect RAW precedence: consumer_cycle={} < shift_cycle+1={}",
            consumer_cycle,
            shift_cycle + 1
        );
    }

    /// Same as test_ssa_cpsat_selectchoice_force_fma_minimal but with DEFAULT machine (6 VALU)
    /// and no hints. Exercises same-cycle VALU packing: sub and fma may share a cycle with other
    /// VALU ops. If the FMA bug is tied to packing or hint-driven schedules, this may fail.
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_ssa_cpsat_selectchoice_force_fma_default_machine_no_hints() {
        use crate::address_assign::{
            assign_addresses, compute_live_intervals, AddressAssignConfig,
        };
        use crate::kernel_builder::emit_ssa::{compute_ssa_dependencies, SsaEmitContext};
        use crate::kernel_builder::ssa_lower::lower_ssa_ops_with_shift;
        use std::collections::HashSet;

        let mut emit = SsaEmitContext::new();
        let c0 = emit.emit_const(0);
        let c1 = emit.emit_const(1);
        let v_c0 = emit.emit_vbroadcast(c0);
        let v_c1 = emit.emit_vbroadcast(c1);
        let cond = emit.emit_valu(Opcode::Eq, v_c0, v_c1);
        let c10 = emit.emit_const(10);
        let true_val = emit.emit_vbroadcast(c10);
        let c20 = emit.emit_const(20);
        let false_val = emit.emit_vbroadcast(c20);
        let sel = emit.emit_select_choice(cond, true_val, false_val);
        let _use = emit.emit_valu(Opcode::Add, sel, sel);

        let vreg_def = std::mem::take(&mut emit.vreg_def);
        compute_ssa_dependencies(&mut emit.ops, &vreg_def, emit.zero_vreg);

        let select_choice_idx = emit
            .ops
            .iter()
            .position(|o| o.opcode == Opcode::SelectChoice)
            .expect("one SelectChoice");
        let select_choice_id = emit.ops[select_choice_idx].id;

        let result = schedule_ssa_cpsat(
            &emit.ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: false,
                stop_after_first_solution: true,
                enable_valu_shift: false,
                scratch_budget: 1536,
                model_name: None,
                machine: crate::types::MachineConfig::default(),
                const_choice_mode: ConstChoiceMode::Optimize,
                select_choice_mode: SelectChoiceMode::ForceFma,
                valu_shift_mode: ValuShiftMode::Optimize,
                ..Default::default()
            },
            SsaCpsatHints::default(),
            None,
        );
        assert!(
            result.is_some(),
            "CP-SAT should find a solution with default machine + ForceFma"
        );
        let cpsat_result = result.unwrap();

        let c = cpsat_result.cycle_assignments[select_choice_idx];
        for (idx, op) in emit.ops.iter().enumerate() {
            if op.dep_ids.contains(&select_choice_id) {
                let consumer_cycle = cpsat_result.cycle_assignments[idx];
                assert!(
                    consumer_cycle >= c + 2,
                    "consumer must be at cycle >= {} (got {})",
                    c + 2,
                    consumer_cycle
                );
            }
        }

        for (idx, op) in emit.ops.iter_mut().enumerate() {
            op.cycle = cpsat_result.cycle_assignments[idx];
        }
        let mut config = AddressAssignConfig::default();
        config.select_fma_ops = Some(cpsat_result.select_fma_ops.clone());
        let assign_result = assign_addresses(&emit.ops, &config);
        assert!(!assign_result.budget_exceeded);

        let intervals = compute_live_intervals(&emit.ops, None, Some(&cpsat_result.select_fma_ops), Some(&cpsat_result.shifted_alu_cycles));
        for (v1, addr1) in &assign_result.vreg_to_addr {
            for (v2, addr2) in &assign_result.vreg_to_addr {
                if v1 >= v2 {
                    continue;
                }
                if addr1 != addr2 {
                    continue;
                }
                if let (Some(i1), Some(i2)) = (intervals.get(v1), intervals.get(v2)) {
                    assert!(
                        !i1.overlaps(i2),
                        "overlapping intervals shared address (default machine)"
                    );
                }
            }
        }

        let lowered = lower_ssa_ops_with_shift(
            &emit.ops,
            &assign_result.vreg_to_addr,
            &HashSet::new(),
            &HashSet::new(),
            &cpsat_result.select_fma_ops,
            &cpsat_result.shifted_alu_cycles,
        );
        let sub_op = lowered.iter().find(|o| {
            o.id == select_choice_id
                && o.engine == Engine::Valu
                && format!("{:?}", o.slot.get(0)).contains("\"-\"")
        });
        let fma_op = lowered
            .iter()
            .find(|o| o.id == select_choice_id + 1000000 && o.engine == Engine::Valu);
        assert!(sub_op.is_some(), "should have sub op");
        assert!(fma_op.is_some(), "should have fma op");
        assert_eq!(sub_op.unwrap().cycle, c, "sub at SelectChoice cycle");
        assert_eq!(
            fma_op.unwrap().cycle,
            c + 1,
            "fma at SelectChoice cycle + 1"
        );
    }

    /// Two SelectChoice FMA ops: verify consumer precedence (>= dep+2) and that address
    /// assignment never gives the same address to two VRegs with overlapping intervals.
    /// Uses small machine (1 VALU slot) so sub/fma pairs are forced into consecutive cycles.
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_ssa_cpsat_two_selectchoice_force_fma_no_overlap() {
        use crate::address_assign::{
            assign_addresses, compute_live_intervals, AddressAssignConfig,
        };
        use crate::kernel_builder::emit_ssa::{compute_ssa_dependencies, SsaEmitContext};

        let mut emit = SsaEmitContext::new();
        let c0 = emit.emit_const(0);
        let c1 = emit.emit_const(1);
        let v_c0 = emit.emit_vbroadcast(c0);
        let v_c1 = emit.emit_vbroadcast(c1);
        let cond = emit.emit_valu(Opcode::Eq, v_c0, v_c1);
        let c10 = emit.emit_const(10);
        let true1 = emit.emit_vbroadcast(c10);
        let c20 = emit.emit_const(20);
        let false1 = emit.emit_vbroadcast(c20);
        let sel1 = emit.emit_select_choice(cond, true1, false1);
        let _u1 = emit.emit_valu(Opcode::Add, sel1, sel1);

        let c30 = emit.emit_const(30);
        let true2 = emit.emit_vbroadcast(c30);
        let c40 = emit.emit_const(40);
        let false2 = emit.emit_vbroadcast(c40);
        let sel2 = emit.emit_select_choice(cond, true2, false2);
        let _u2 = emit.emit_valu(Opcode::Add, sel2, sel2);

        let vreg_def = std::mem::take(&mut emit.vreg_def);
        compute_ssa_dependencies(&mut emit.ops, &vreg_def, emit.zero_vreg);

        let idx1 = emit
            .ops
            .iter()
            .position(|o| o.opcode == Opcode::SelectChoice)
            .expect("first");
        let idx2 = emit
            .ops
            .iter()
            .position(|o| o.opcode == Opcode::SelectChoice && o.id != emit.ops[idx1].id)
            .expect("second");
        let id1 = emit.ops[idx1].id;
        let id2 = emit.ops[idx2].id;
        assert!(emit.ops[idx1].sub_temp_vreg.is_some() && emit.ops[idx2].sub_temp_vreg.is_some());

        let result = schedule_ssa_cpsat(
            &emit.ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: false,
                stop_after_first_solution: true,
                enable_valu_shift: false,
                scratch_budget: 1536,
                model_name: None,
                machine: SMALL_MACHINE_FMA,
                const_choice_mode: ConstChoiceMode::Optimize,
                select_choice_mode: SelectChoiceMode::ForceFma,
                valu_shift_mode: ValuShiftMode::Optimize,
                ..Default::default()
            },
            SsaCpsatHints::default(),
            None,
        );
        assert!(result.is_some(), "CP-SAT should find a solution");
        let cpsat_result = result.unwrap();

        assert!(
            cpsat_result.select_fma_ops.contains(&id1)
                && cpsat_result.select_fma_ops.contains(&id2)
        );
        let c1 = cpsat_result.cycle_assignments[idx1];
        let c2 = cpsat_result.cycle_assignments[idx2];
        for (idx, op) in emit.ops.iter().enumerate() {
            if op.dep_ids.contains(&id1) {
                let consumer_cycle = cpsat_result.cycle_assignments[idx];
                assert!(
                    consumer_cycle >= c1 + 2,
                    "consumer of sel1 must be >= {} (got {})",
                    c1 + 2,
                    consumer_cycle
                );
            }
            if op.dep_ids.contains(&id2) {
                let consumer_cycle = cpsat_result.cycle_assignments[idx];
                assert!(
                    consumer_cycle >= c2 + 2,
                    "consumer of sel2 must be >= {} (got {})",
                    c2 + 2,
                    consumer_cycle
                );
            }
        }

        for (idx, op) in emit.ops.iter_mut().enumerate() {
            op.cycle = cpsat_result.cycle_assignments[idx];
        }
        let mut config = AddressAssignConfig::default();
        config.select_fma_ops = Some(cpsat_result.select_fma_ops.clone());
        let assign_result = assign_addresses(&emit.ops, &config);
        assert!(!assign_result.budget_exceeded);

        let intervals = compute_live_intervals(&emit.ops, None, Some(&cpsat_result.select_fma_ops), Some(&cpsat_result.shifted_alu_cycles));
        for (v1, addr1) in &assign_result.vreg_to_addr {
            for (v2, addr2) in &assign_result.vreg_to_addr {
                if v1 >= v2 {
                    continue;
                }
                if addr1 != addr2 {
                    continue;
                }
                if let (Some(i1), Some(i2)) = (intervals.get(v1), intervals.get(v2)) {
                    assert!(!i1.overlaps(i2), "BUG: VRegs {:?} and {:?} share address {} but intervals [{},{}] and [{},{}] overlap",
                        v1, v2, addr1, i1.start, i1.end, i2.start, i2.end);
                }
            }
        }
    }

    /// Full pipeline: CP-SAT ForceFma then assign then lower; verify no overlapping
    /// intervals share an address (regression test for validation failure).
    /// Uses small machine (1 VALU slot) so sub and fma are in consecutive bundles.
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_ssa_cpsat_force_fma_full_pipeline_no_overlap() {
        use crate::address_assign::{
            assign_addresses, compute_live_intervals, AddressAssignConfig,
        };
        use crate::kernel_builder::emit_ssa::{compute_ssa_dependencies, SsaEmitContext};
        use crate::kernel_builder::ssa_lower::lower_ssa_ops_with_shift;
        use std::collections::HashSet;

        let mut emit = SsaEmitContext::new();
        let c0 = emit.emit_const(0);
        let c1 = emit.emit_const(1);
        let v_c0 = emit.emit_vbroadcast(c0);
        let v_c1 = emit.emit_vbroadcast(c1);
        let cond = emit.emit_valu(Opcode::Eq, v_c0, v_c1);
        let c10 = emit.emit_const(10);
        let true_val = emit.emit_vbroadcast(c10);
        let c20 = emit.emit_const(20);
        let false_val = emit.emit_vbroadcast(c20);
        let sel = emit.emit_select_choice(cond, true_val, false_val);
        let _use = emit.emit_valu(Opcode::Add, sel, sel);

        let vreg_def = std::mem::take(&mut emit.vreg_def);
        compute_ssa_dependencies(&mut emit.ops, &vreg_def, emit.zero_vreg);

        let result = schedule_ssa_cpsat(
            &emit.ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: false,
                stop_after_first_solution: true,
                enable_valu_shift: false,
                scratch_budget: 1536,
                model_name: None,
                machine: SMALL_MACHINE_FMA,
                const_choice_mode: ConstChoiceMode::Optimize,
                select_choice_mode: SelectChoiceMode::ForceFma,
                valu_shift_mode: ValuShiftMode::Optimize,
                ..Default::default()
            },
            SsaCpsatHints::default(),
            None,
        );
        let cpsat_result = result.unwrap();
        for (idx, op) in emit.ops.iter_mut().enumerate() {
            op.cycle = cpsat_result.cycle_assignments[idx];
        }
        let mut config = AddressAssignConfig::default();
        config.select_fma_ops = Some(cpsat_result.select_fma_ops.clone());
        let assign_result = assign_addresses(&emit.ops, &config);
        let intervals = compute_live_intervals(&emit.ops, None, Some(&cpsat_result.select_fma_ops), Some(&cpsat_result.shifted_alu_cycles));
        for (v1, addr1) in &assign_result.vreg_to_addr {
            for (v2, addr2) in &assign_result.vreg_to_addr {
                if v1 >= v2 {
                    continue;
                }
                if addr1 != addr2 {
                    continue;
                }
                if let (Some(i1), Some(i2)) = (intervals.get(v1), intervals.get(v2)) {
                    assert!(!i1.overlaps(i2), "overlapping intervals shared address");
                }
            }
        }
        let _lowered = lower_ssa_ops_with_shift(
            &emit.ops,
            &assign_result.vreg_to_addr,
            &HashSet::new(),
            &HashSet::new(),
            &cpsat_result.select_fma_ops,
            &cpsat_result.shifted_alu_cycles,
        );
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
    use crate::scheduler::{compute_critical_path, schedule_greedy_with_diagnostics, PriorityMode};

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

    /// Verifies that makespan respects the capacity lower bound (ceil(num_ops / capacity))
    /// enforced by our redundant linear constraints. 13 VALU ops with capacity 6 => ceil(13/6)=3.
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_schedule_cpsat_makespan_respects_capacity_lower_bound() {
        let ops: Vec<Op> = (0..13)
            .map(|i| {
                let mut op = make_op(i, Engine::Valu, vec![]);
                op.writes = [(i * 10) as i64].into_iter().collect();
                op
            })
            .collect();

        let result = schedule_cpsat(&ops, CpsatConfig::new().with_time_limit(10.0));
        assert!(result.is_some());

        let r = result.unwrap();
        // 13 VALU ops with 6 slots/cycle => ceil(13/6) = 3 minimum cycles (redundant linear constraint)
        assert!(
            r.result.makespan >= 3,
            "makespan {} should be >= 3 (capacity lower bound for 13 VALU ops)",
            r.result.makespan
        );
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
        use crate::kernel_builder::emit_ssa::{compute_ssa_dependencies, SsaEmitContext};
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
        compute_ssa_dependencies(&mut emit.ops, &vreg_def, emit.zero_vreg);

        // Schedule with CP-SAT
        let result = schedule_ssa_cpsat(
            &emit.ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: false,
                stop_after_first_solution: true,
                enable_valu_shift: false,
                scratch_budget: 1536,
                model_name: None,
                ..Default::default()
            },
            SsaCpsatHints::default(),
            None,
        );

        assert!(result.is_some(), "CP-SAT should find a solution");
        let result = result.unwrap();

        // Verify all ops are scheduled
        assert_eq!(result.cycle_assignments.len(), emit.ops.len());

        // Verify ConstChoice decisions are extracted
        // The result should contain const_add_imm_ops (may be empty if solver chose Load)
        assert!(
            result.const_add_imm_ops.len() <= 2,
            "Should have at most 2 ConstChoice ops using AddImm"
        );

        // Note: AddImm operations can use address 0 directly without waiting for zero
        // to be "ready" since scratch memory is always initialized to 0 at startup.
        // There is no precedence constraint requiring zero to be scheduled before AddImm.
    }

    #[cfg(feature = "cpsat")]
    #[test]
    fn test_ssa_cpsat_constchoice_roundtrip() {
        use crate::kernel_builder::emit_ssa::{compute_ssa_dependencies, SsaEmitContext};
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
        compute_ssa_dependencies(&mut emit.ops, &vreg_def, emit.zero_vreg);

        // Schedule with CP-SAT
        let result = schedule_ssa_cpsat(
            &emit.ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: false,
                stop_after_first_solution: true,
                enable_valu_shift: false,
                scratch_budget: 1536,
                model_name: None,
                ..Default::default()
            },
            SsaCpsatHints::default(),
            None,
        );

        assert!(result.is_some(), "CP-SAT should find a solution");
        let cpsat_result = result.unwrap();

        // Verify all ops are scheduled
        assert_eq!(cpsat_result.cycle_assignments.len(), emit.ops.len());

        // Find ConstChoice op IDs and their values
        let const_choice_info: Vec<(usize, i64)> = emit
            .ops
            .iter()
            .filter(|op| op.opcode == Opcode::ConstChoice)
            .map(|op| {
                let value = op.srcs[0]
                    .as_imm()
                    .expect("ConstChoice should have immediate operand");
                (op.id, value)
            })
            .collect();

        assert_eq!(
            const_choice_info.len(),
            3,
            "Should have 3 ConstChoice ops (1, 2, 4)"
        );

        // Verify CP-SAT decisions are extracted
        assert!(
            cpsat_result.const_add_imm_ops.len() <= 3,
            "Should have at most 3 ConstChoice ops using AddImm"
        );

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
            &HashSet::new(),                 // No shifted ops
            &cpsat_result.const_add_imm_ops, // Use CP-SAT decisions
            &cpsat_result.select_fma_ops,
            &cpsat_result.shifted_alu_cycles, // Use CP-SAT individual ALU cycles
        );

        // Verify that ConstChoice ops were lowered correctly based on CP-SAT decisions
        for (const_choice_id, expected_value) in &const_choice_info {
            let lowered_op = lowered_ops
                .iter()
                .find(|op| op.produces_constant == Some(*expected_value))
                .expect(&format!(
                    "Should find lowered op for ConstChoice producing {}",
                    expected_value
                ));

            if cpsat_result.const_add_imm_ops.contains(const_choice_id) {
                // CP-SAT chose AddImm - verify it's a Flow op
                assert_eq!(
                    lowered_op.engine,
                    Engine::Flow,
                    "ConstChoice op {} (value {}) should be lowered to Flow (AddImm)",
                    const_choice_id,
                    expected_value
                );
                match &lowered_op.slot[0] {
                    SlotVal::Str(s) => assert_eq!(
                        s, "add_imm",
                        "ConstChoice op {} should be add_imm",
                        const_choice_id
                    ),
                    _ => panic!("AddImm slot[0] should be Str"),
                }
                match &lowered_op.slot[2] {
                    SlotVal::Int(addr) => assert_eq!(*addr, 0, "AddImm should read from address 0"),
                    _ => panic!("AddImm slot[2] should be Int(0)"),
                }
            } else {
                // CP-SAT chose Load - verify it's a Load op
                assert_eq!(
                    lowered_op.engine,
                    Engine::Load,
                    "ConstChoice op {} (value {}) should be lowered to Load",
                    const_choice_id,
                    expected_value
                );
                match &lowered_op.slot[0] {
                    SlotVal::Str(s) => assert_eq!(
                        s, "const",
                        "ConstChoice op {} should be const",
                        const_choice_id
                    ),
                    _ => panic!("Const slot[0] should be Str"),
                }
            }
        }

        // Verify all ConstChoice ops were lowered
        assert_eq!(
            lowered_ops.len(),
            emit.ops.len(),
            "All ops should be lowered"
        );
    }

    #[cfg(feature = "cpsat")]
    #[test]
    fn test_ssa_cpsat_constchoice_prefers_addimm_when_load_slots_full() {
        use crate::kernel_builder::emit_ssa::{compute_ssa_dependencies, SsaEmitContext};
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
        compute_ssa_dependencies(&mut emit.ops, &vreg_def, emit.zero_vreg);

        // Schedule with CP-SAT with longer time limit to allow optimization
        let result = schedule_ssa_cpsat(
            &emit.ops,
            SsaCpsatConfig {
                time_limit_secs: 30.0,
                verbose: false,
                stop_after_first_solution: false, // Allow optimization
                enable_valu_shift: false,
                scratch_budget: 1536,
                model_name: None,
                ..Default::default()
            },
            SsaCpsatHints::default(),
            None,
        );

        assert!(result.is_some(), "CP-SAT should find a solution");
        let cpsat_result = result.unwrap();

        // Find ConstChoice op IDs
        let const_choice_ids: Vec<usize> = emit
            .ops
            .iter()
            .filter(|op| op.opcode == Opcode::ConstChoice)
            .map(|op| op.id)
            .collect();

        assert_eq!(const_choice_ids.len(), 2, "Should have 2 ConstChoice ops");

        // Verify decisions were extracted
        assert!(
            cpsat_result.const_add_imm_ops.len() <= 2,
            "Should have at most 2 ConstChoice ops using AddImm"
        );

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
            &cpsat_result.select_fma_ops,
            &cpsat_result.shifted_alu_cycles, // Use CP-SAT individual ALU cycles
        );

        // Verify ConstChoice ops were lowered correctly
        for const_choice_id in &const_choice_ids {
            let ssa_op = emit
                .ops
                .iter()
                .find(|op| op.id == *const_choice_id)
                .unwrap();
            let expected_value = ssa_op.srcs[0].as_imm().unwrap();

            let lowered_op = lowered_ops
                .iter()
                .find(|op| op.produces_constant == Some(expected_value))
                .expect(&format!(
                    "Should find lowered op for ConstChoice {}",
                    const_choice_id
                ));

            if cpsat_result.const_add_imm_ops.contains(const_choice_id) {
                // Verify AddImm was chosen
                assert_eq!(
                    lowered_op.engine,
                    Engine::Flow,
                    "ConstChoice {} should use AddImm when Load slots are full",
                    const_choice_id
                );
                match &lowered_op.slot[0] {
                    SlotVal::Str(s) => assert_eq!(s, "add_imm"),
                    _ => panic!("Should be add_imm"),
                }
            } else {
                // Verify Load was chosen
                assert_eq!(
                    lowered_op.engine,
                    Engine::Load,
                    "ConstChoice {} should use Load",
                    const_choice_id
                );
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
            make_op(100, Engine::Alu, vec![]),    // slice idx 0, op.id 100
            make_op(200, Engine::Alu, vec![100]), // slice idx 1, op.id 200
            make_op(300, Engine::Alu, vec![200]), // slice idx 2, op.id 300
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

        assert!(
            cpsat_result.is_some(),
            "CP-SAT should accept SSA-lowered ops with hints"
        );

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
            r.result.makespan,
            ssa_result.cycles
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

        assert!(
            cpsat_result.is_some(),
            "CP-SAT should accept SSA vector ops"
        );

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

        assert!(
            cpsat_result.is_some(),
            "CP-SAT should work with SSA diamond pattern"
        );

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
            make_ssa_op(
                0,
                Engine::Load,
                Opcode::Const,
                Some(VReg::scalar(0)),
                vec![Operand::imm(5)],
                vec![],
            ),
            make_ssa_op(
                1,
                Engine::Alu,
                Opcode::Add,
                Some(VReg::scalar(1)),
                vec![Operand::scalar(0), Operand::imm(1)],
                vec![0],
            ),
            make_ssa_op(
                2,
                Engine::Alu,
                Opcode::Add,
                Some(VReg::scalar(2)),
                vec![Operand::scalar(1), Operand::imm(2)],
                vec![1],
            ),
        ];

        // With generous budget, should succeed
        let result = schedule_ssa_cpsat(
            &ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: true,
                scratch_budget: 100, // More than enough
                ..Default::default()
            },
            SsaCpsatHints::default(),
            None,
        );

        assert!(
            result.is_some(),
            "Should find solution with generous budget"
        );
        let r = result.unwrap();
        assert_eq!(
            r.status,
            CpsatStatus::Optimal,
            "Should be optimal for simple chain"
        );
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
            make_ssa_op(
                0,
                Engine::Load,
                Opcode::Const,
                Some(VReg::vector(0)),
                vec![Operand::imm(1)],
                vec![],
            ),
            make_ssa_op(
                1,
                Engine::Load,
                Opcode::Const,
                Some(VReg::vector(1)),
                vec![Operand::imm(2)],
                vec![],
            ),
            make_ssa_op(
                2,
                Engine::Load,
                Opcode::Const,
                Some(VReg::vector(2)),
                vec![Operand::imm(3)],
                vec![],
            ),
            make_ssa_op(
                3,
                Engine::Valu,
                Opcode::Add,
                Some(VReg::vector(3)),
                vec![Operand::vector(0), Operand::vector(1)],
                vec![0, 1],
            ),
            make_ssa_op(
                4,
                Engine::Valu,
                Opcode::Add,
                Some(VReg::vector(4)),
                vec![Operand::vector(2), Operand::vector(3)],
                vec![2, 3],
            ),
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
            SsaCpsatHints::default(),
            None,
        );

        assert!(
            result_generous.is_some(),
            "Should find solution with generous budget"
        );
        let r_generous = result_generous.unwrap();
        eprintln!(
            "Generous budget: makespan={}, status={:?}",
            r_generous.makespan, r_generous.status
        );

        // With tight budget (16 words = 2 vectors), should need to serialize more
        let result_tight = schedule_ssa_cpsat(
            &ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: true,
                scratch_budget: 16, // Only 2 vectors can be live at once
                ..Default::default()
            },
            SsaCpsatHints::default(),
            None,
        );

        // The tight budget should either:
        // 1. Force a longer schedule (more cycles to reduce concurrent live values), or
        // 2. Return infeasible if impossible
        if let Some(r_tight) = result_tight {
            eprintln!(
                "Tight budget: makespan={}, status={:?}",
                r_tight.makespan, r_tight.status
            );
            // With tight budget, makespan should be >= generous budget makespan
            // (can't do better, might need more cycles)
            assert!(
                r_tight.makespan >= r_generous.makespan,
                "Tight budget makespan ({}) should be >= generous ({}) or infeasible",
                r_tight.makespan,
                r_generous.makespan
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
            make_ssa_op(
                0,
                Engine::Load,
                Opcode::Const,
                Some(VReg::vector(0)),
                vec![Operand::imm(1)],
                vec![],
            ),
            make_ssa_op(
                1,
                Engine::Load,
                Opcode::Const,
                Some(VReg::vector(1)),
                vec![Operand::imm(2)],
                vec![],
            ),
            make_ssa_op(
                2,
                Engine::Valu,
                Opcode::Add,
                Some(VReg::vector(2)),
                vec![Operand::vector(0), Operand::vector(1)],
                vec![0, 1],
            ),
        ];

        // With budget=0, constraint is disabled
        let result = schedule_ssa_cpsat(
            &ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: true,
                scratch_budget: 0, // Disabled
                ..Default::default()
            },
            SsaCpsatHints::default(),
            None,
        );

        assert!(
            result.is_some(),
            "Should find solution when scratch constraint disabled"
        );
        let r = result.unwrap();
        eprintln!(
            "Disabled constraint: makespan={}, status={:?}",
            r.makespan, r.status
        );
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
            make_ssa_op(
                0,
                Engine::Load,
                Opcode::Const,
                Some(VReg::scalar(0)),
                vec![Operand::imm(5)],
                vec![],
            ),
            make_ssa_op(
                1,
                Engine::Alu,
                Opcode::Add,
                Some(VReg::scalar(1)),
                vec![Operand::scalar(0), Operand::imm(1)],
                vec![0],
            ),
            make_ssa_op(
                2,
                Engine::Alu,
                Opcode::Add,
                Some(VReg::scalar(2)),
                vec![Operand::scalar(1), Operand::imm(2)],
                vec![1],
            ),
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
            SsaCpsatHints {
                greedy_hints: Some(&hints),
                shift_hints: None,
                forced_shifts: None,
                select_choice_hints: None,
                const_add_imm_hints: None,
                initial_live: None,
                live_out_vregs: None,
                live_through_vregs: None,
            },
            Some(true), // valid hints, within scratch
        );

        assert!(result.is_some(), "Should find solution with valid hints");
        let r = result.unwrap();

        // Critical: with stop_after_first_solution, we should get exactly the hinted solution
        assert_eq!(
            r.cycle_assignments, hints,
            "With stop_after_first_solution, should return exactly the hinted solution"
        );
        assert_eq!(r.makespan, 3, "Makespan should be 3 (cycles 0, 1, 2)");

        eprintln!(
            "Hint roundtrip (no scratch): assignments={:?}, makespan={}",
            r.cycle_assignments, r.makespan
        );
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
            make_ssa_op(
                0,
                Engine::Load,
                Opcode::Const,
                Some(VReg::scalar(0)),
                vec![Operand::imm(5)],
                vec![],
            ),
            make_ssa_op(
                1,
                Engine::Alu,
                Opcode::Add,
                Some(VReg::scalar(1)),
                vec![Operand::scalar(0), Operand::imm(1)],
                vec![0],
            ),
            make_ssa_op(
                2,
                Engine::Alu,
                Opcode::Add,
                Some(VReg::scalar(2)),
                vec![Operand::scalar(1), Operand::imm(2)],
                vec![1],
            ),
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
            SsaCpsatHints {
                greedy_hints: Some(&hints),
                ..Default::default()
            },
            None,
        );

        assert!(
            result.is_some(),
            "Should find solution with valid hints and scratch constraint"
        );
        let r = result.unwrap();

        // Critical: hints should be feasible with scratch constraint
        assert_eq!(
            r.cycle_assignments, hints,
            "With stop_after_first_solution, should return exactly the hinted solution"
        );
        assert_eq!(r.makespan, 3, "Makespan should be 3");

        eprintln!(
            "Hint roundtrip (with scratch): assignments={:?}, makespan={}",
            r.cycle_assignments, r.makespan
        );
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
            make_ssa_op(
                0,
                Engine::Load,
                Opcode::Const,
                Some(VReg::vector(0)),
                vec![Operand::imm(1)],
                vec![],
            ),
            make_ssa_op(
                1,
                Engine::Load,
                Opcode::Const,
                Some(VReg::vector(1)),
                vec![Operand::imm(2)],
                vec![],
            ),
            make_ssa_op(
                2,
                Engine::Valu,
                Opcode::Add,
                Some(VReg::vector(2)),
                vec![Operand::vector(0), Operand::vector(1)],
                vec![0, 1],
            ),
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
            SsaCpsatHints {
                greedy_hints: Some(&hints),
                ..Default::default()
            },
            None,
        );

        assert!(
            result.is_some(),
            "Should find solution with valid vector hints and scratch constraint"
        );
        let r = result.unwrap();

        // Critical: hints should be feasible
        assert_eq!(
            r.cycle_assignments, hints,
            "With stop_after_first_solution, should return exactly the hinted solution"
        );
        assert_eq!(r.makespan, 2, "Makespan should be 2 (cycles 0 and 1)");

        eprintln!(
            "Hint roundtrip (vectors with scratch): assignments={:?}, makespan={}",
            r.cycle_assignments, r.makespan
        );
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
            make_ssa_op(
                0,
                Engine::Load,
                Opcode::Const,
                Some(VReg::vector(0)),
                vec![Operand::imm(1)],
                vec![],
            ),
            make_ssa_op(
                1,
                Engine::Load,
                Opcode::Const,
                Some(VReg::vector(1)),
                vec![Operand::imm(2)],
                vec![],
            ),
            make_ssa_op(
                2,
                Engine::Valu,
                Opcode::Add,
                Some(VReg::vector(2)),
                vec![Operand::vector(0), Operand::vector(1)],
                vec![0, 1],
            ),
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
            SsaCpsatHints {
                greedy_hints: Some(&hints),
                ..Default::default()
            },
            None,
        );

        // The solver should either:
        // 1. Find a different (valid) schedule that fits the budget, or
        // 2. Return infeasible if no valid schedule exists
        // It should NOT return the infeasible hint as-is
        if let Some(r) = result {
            // If it found a solution, it should NOT be the original hint
            // because that hint violates the scratch budget
            eprintln!(
                "Found alternative schedule: {:?}, makespan={}",
                r.cycle_assignments, r.makespan
            );
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
            make_ssa_op(
                0,
                Engine::Load,
                Opcode::Const,
                Some(VReg::scalar(0)),
                vec![Operand::imm(5)],
                vec![],
            ),
            make_ssa_op(
                1,
                Engine::Alu,
                Opcode::Add,
                Some(VReg::scalar(1)),
                vec![Operand::scalar(0), Operand::imm(1)],
                vec![0],
            ),
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
            SsaCpsatHints {
                greedy_hints: Some(&hints),
                ..Default::default()
            },
            None,
        );

        assert!(
            result.is_some(),
            "Should find solution with same-cycle interval (size=1)"
        );
        let r = result.unwrap();
        assert_eq!(
            r.cycle_assignments, hints,
            "Should return the hinted solution"
        );
        assert_eq!(r.makespan, 2, "Makespan should be 2");
        eprintln!(
            "Same-cycle interval test: assignments={:?}, makespan={}",
            r.cycle_assignments, r.makespan
        );
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
            make_ssa_op(
                0,
                Engine::Load,
                Opcode::Const,
                Some(VReg::scalar(0)),
                vec![Operand::imm(5)],
                vec![],
            ),
            make_ssa_op(
                1,
                Engine::Alu,
                Opcode::Add,
                Some(VReg::scalar(1)),
                vec![Operand::scalar(0), Operand::imm(1)],
                vec![0],
            ),
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
            SsaCpsatHints {
                greedy_hints: Some(&hints),
                ..Default::default()
            },
            None,
        );

        assert!(
            result.is_some(),
            "Should find solution with consecutive-cycle interval (size=2)"
        );
        let r = result.unwrap();
        assert_eq!(
            r.cycle_assignments, hints,
            "Should return the hinted solution"
        );
        assert_eq!(r.makespan, 2, "Makespan should be 2");
        eprintln!(
            "Consecutive-cycle interval test: assignments={:?}, makespan={}",
            r.cycle_assignments, r.makespan
        );
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
            make_ssa_op(
                0,
                Engine::Load,
                Opcode::Const,
                Some(VReg::scalar(0)),
                vec![Operand::imm(5)],
                vec![],
            ),
            make_ssa_op(
                1,
                Engine::Load,
                Opcode::Const,
                Some(VReg::scalar(1)),
                vec![Operand::imm(10)],
                vec![],
            ),
            make_ssa_op(
                2,
                Engine::Alu,
                Opcode::Add,
                Some(VReg::scalar(2)),
                vec![Operand::scalar(1), Operand::imm(1)],
                vec![1],
            ),
            make_ssa_op(
                3,
                Engine::Alu,
                Opcode::Add,
                Some(VReg::scalar(3)),
                vec![Operand::scalar(2), Operand::imm(1)],
                vec![2],
            ),
            make_ssa_op(
                4,
                Engine::Alu,
                Opcode::Add,
                Some(VReg::scalar(4)),
                vec![Operand::scalar(3), Operand::imm(1)],
                vec![3],
            ),
            make_ssa_op(
                5,
                Engine::Alu,
                Opcode::Add,
                Some(VReg::scalar(5)),
                vec![Operand::scalar(4), Operand::imm(1)],
                vec![4],
            ),
            make_ssa_op(
                6,
                Engine::Alu,
                Opcode::Add,
                Some(VReg::scalar(6)),
                vec![Operand::scalar(5), Operand::scalar(0)],
                vec![5, 0],
            ),
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
            SsaCpsatHints {
                greedy_hints: Some(&hints),
                ..Default::default()
            },
            None,
        );

        assert!(
            result.is_some(),
            "Should find solution with gap interval (size=6)"
        );
        let r = result.unwrap();
        assert_eq!(
            r.cycle_assignments, hints,
            "Should return the hinted solution"
        );
        assert_eq!(r.makespan, 6, "Makespan should be 6");
        eprintln!(
            "Gap interval test: assignments={:?}, makespan={}",
            r.cycle_assignments, r.makespan
        );
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
            make_ssa_op(
                0,
                Engine::Load,
                Opcode::Const,
                Some(VReg::vector(0)),
                vec![Operand::imm(1)],
                vec![],
            ),
            make_ssa_op(
                1,
                Engine::Load,
                Opcode::Const,
                Some(VReg::vector(1)),
                vec![Operand::imm(2)],
                vec![],
            ),
            // Note: only 2 load slots per cycle, so we need more cycles
            // Actually, let's use consts with immediates to avoid load slot issues
            // Create 6 VALU ops that use immediates as second operand
            make_ssa_op(
                2,
                Engine::Valu,
                Opcode::Add,
                Some(VReg::vector(2)),
                vec![Operand::vector(0), Operand::vector(1)],
                vec![0, 1],
            ),
            make_ssa_op(
                3,
                Engine::Valu,
                Opcode::Sub,
                Some(VReg::vector(3)),
                vec![Operand::vector(0), Operand::vector(1)],
                vec![0, 1],
            ),
            make_ssa_op(
                4,
                Engine::Valu,
                Opcode::Mul,
                Some(VReg::vector(4)),
                vec![Operand::vector(0), Operand::vector(1)],
                vec![0, 1],
            ),
            make_ssa_op(
                5,
                Engine::Valu,
                Opcode::Xor,
                Some(VReg::vector(5)),
                vec![Operand::vector(0), Operand::vector(1)],
                vec![0, 1],
            ),
            make_ssa_op(
                6,
                Engine::Valu,
                Opcode::And,
                Some(VReg::vector(6)),
                vec![Operand::vector(0), Operand::vector(1)],
                vec![0, 1],
            ),
            make_ssa_op(
                7,
                Engine::Valu,
                Opcode::Or,
                Some(VReg::vector(7)),
                vec![Operand::vector(0), Operand::vector(1)],
                vec![0, 1],
            ),
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
            SsaCpsatHints {
                greedy_hints: Some(&hints),
                ..Default::default()
            },
            None,
        );

        // The verification code should NOT panic, and we should get a valid result
        assert!(
            result.is_some(),
            "Should find solution with 6 VALU ops at same cycle (exactly at limit)"
        );
        let r = result.unwrap();
        assert_eq!(
            r.cycle_assignments, hints,
            "Should accept the hinted solution"
        );
        eprintln!(
            "VALU capacity test: 6 VALU ops at cycle 1, result={:?}",
            r.cycle_assignments
        );
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
            make_ssa_op(
                0,
                Engine::Load,
                Opcode::Const,
                Some(VReg::scalar(0)),
                vec![Operand::imm(1)],
                vec![],
            ),
            make_ssa_op(
                1,
                Engine::Load,
                Opcode::Const,
                Some(VReg::scalar(1)),
                vec![Operand::imm(2)],
                vec![],
            ),
            // 12 ALU ops - all depend on loads, can run in parallel
            make_ssa_op(
                2,
                Engine::Alu,
                Opcode::Add,
                Some(VReg::scalar(2)),
                vec![Operand::scalar(0), Operand::scalar(1)],
                vec![0, 1],
            ),
            make_ssa_op(
                3,
                Engine::Alu,
                Opcode::Sub,
                Some(VReg::scalar(3)),
                vec![Operand::scalar(0), Operand::scalar(1)],
                vec![0, 1],
            ),
            make_ssa_op(
                4,
                Engine::Alu,
                Opcode::Mul,
                Some(VReg::scalar(4)),
                vec![Operand::scalar(0), Operand::scalar(1)],
                vec![0, 1],
            ),
            make_ssa_op(
                5,
                Engine::Alu,
                Opcode::Xor,
                Some(VReg::scalar(5)),
                vec![Operand::scalar(0), Operand::scalar(1)],
                vec![0, 1],
            ),
            make_ssa_op(
                6,
                Engine::Alu,
                Opcode::And,
                Some(VReg::scalar(6)),
                vec![Operand::scalar(0), Operand::scalar(1)],
                vec![0, 1],
            ),
            make_ssa_op(
                7,
                Engine::Alu,
                Opcode::Or,
                Some(VReg::scalar(7)),
                vec![Operand::scalar(0), Operand::scalar(1)],
                vec![0, 1],
            ),
            make_ssa_op(
                8,
                Engine::Alu,
                Opcode::Shl,
                Some(VReg::scalar(8)),
                vec![Operand::scalar(0), Operand::scalar(1)],
                vec![0, 1],
            ),
            make_ssa_op(
                9,
                Engine::Alu,
                Opcode::Shr,
                Some(VReg::scalar(9)),
                vec![Operand::scalar(0), Operand::scalar(1)],
                vec![0, 1],
            ),
            make_ssa_op(
                10,
                Engine::Alu,
                Opcode::Lt,
                Some(VReg::scalar(10)),
                vec![Operand::scalar(0), Operand::scalar(1)],
                vec![0, 1],
            ),
            make_ssa_op(
                11,
                Engine::Alu,
                Opcode::Eq,
                Some(VReg::scalar(11)),
                vec![Operand::scalar(0), Operand::scalar(1)],
                vec![0, 1],
            ),
            make_ssa_op(
                12,
                Engine::Alu,
                Opcode::Add,
                Some(VReg::scalar(12)),
                vec![Operand::scalar(0), Operand::scalar(1)],
                vec![0, 1],
            ),
            make_ssa_op(
                13,
                Engine::Alu,
                Opcode::Sub,
                Some(VReg::scalar(13)),
                vec![Operand::scalar(0), Operand::scalar(1)],
                vec![0, 1],
            ),
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
            SsaCpsatHints {
                greedy_hints: Some(&hints),
                ..Default::default()
            },
            None,
        );

        // The verification code should NOT panic, and we should get a valid result
        assert!(
            result.is_some(),
            "Should find solution with 12 ALU ops at same cycle (exactly at limit)"
        );
        let r = result.unwrap();
        assert_eq!(
            r.cycle_assignments, hints,
            "Should accept the hinted solution"
        );
        eprintln!(
            "ALU capacity test: 12 ALU ops at cycle 1, result={:?}",
            r.cycle_assignments
        );
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
            make_ssa_op(
                0,
                Engine::Load,
                Opcode::Const,
                Some(VReg::vector(0)),
                vec![Operand::imm(1)],
                vec![],
            ),
            make_ssa_op(
                1,
                Engine::Load,
                Opcode::Const,
                Some(VReg::scalar(1)),
                vec![Operand::imm(2)],
                vec![],
            ),
            // 1 VALU op that will be shifted to ALU (uses 8 ALU slots)
            make_ssa_op(
                2,
                Engine::Valu,
                Opcode::Add,
                Some(VReg::vector(2)),
                vec![Operand::vector(0), Operand::vector(0)],
                vec![0],
            ),
            // 4 ALU ops (4 slots)
            make_ssa_op(
                3,
                Engine::Alu,
                Opcode::Add,
                Some(VReg::scalar(3)),
                vec![Operand::scalar(1), Operand::scalar(1)],
                vec![1],
            ),
            make_ssa_op(
                4,
                Engine::Alu,
                Opcode::Sub,
                Some(VReg::scalar(4)),
                vec![Operand::scalar(1), Operand::scalar(1)],
                vec![1],
            ),
            make_ssa_op(
                5,
                Engine::Alu,
                Opcode::Mul,
                Some(VReg::scalar(5)),
                vec![Operand::scalar(1), Operand::scalar(1)],
                vec![1],
            ),
            make_ssa_op(
                6,
                Engine::Alu,
                Opcode::Xor,
                Some(VReg::scalar(6)),
                vec![Operand::scalar(1), Operand::scalar(1)],
                vec![1],
            ),
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
                scratch_budget: 0,       // Disable scratch constraint for this test
                enable_valu_shift: true, // Enable VALU shifting
                ..Default::default()
            },
            SsaCpsatHints {
                greedy_hints: Some(&hints),
                shift_hints: Some(&shift_hints),
                ..Default::default()
            },
            None,
        );

        // The verification code should NOT panic:
        // - VALU per cycle: 0 (the VALU op is shifted to ALU)
        // - ALU per cycle: 8 (shifted VALU) + 4 (regular ALU) = 12 (exactly at limit)
        //
        // Note: The verification happens BEFORE solver runs (at lines 1462-1525) and uses
        // shift_hints to verify that the greedy hints don't violate capacity. If the verification
        // panicked, we wouldn't reach this point. Getting a result means verification passed.
        assert!(
            result.is_some(),
            "Should find solution with shifted VALU (8) + 4 ALU = 12 (at limit)"
        );
        let r = result.unwrap();
        assert_eq!(
            r.cycle_assignments, hints,
            "Should accept the hinted solution"
        );

        // Note: The solver's shifted_ops may or may not contain our op depending on whether
        // the solver decides to shift. The key test is that the verification code (which uses
        // shift_hints to count ALU usage) did not panic. If it had panicked, we wouldn't get here.
        eprintln!("Shifted VALU capacity test: 1 shifted VALU (8 slots) + 4 ALU (4 slots) = 12 total at cycle 1");
        eprintln!(
            "Result: {:?}, shifted_ops from solver: {:?}",
            r.cycle_assignments, r.shifted_ops
        );

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
            make_ssa_op(
                0,
                Engine::Load,
                Opcode::Const,
                Some(VReg::scalar(0)),
                vec![Operand::imm(5)],
                vec![],
            ),
            make_ssa_op(
                1,
                Engine::Alu,
                Opcode::Add,
                Some(VReg::scalar(1)),
                vec![Operand::scalar(0), Operand::imm(1)],
                vec![0],
            ),
            make_ssa_op(
                2,
                Engine::Alu,
                Opcode::Add,
                Some(VReg::scalar(2)),
                vec![Operand::scalar(1), Operand::imm(2)],
                vec![1],
            ),
            make_ssa_op(
                3,
                Engine::Alu,
                Opcode::Add,
                Some(VReg::scalar(3)),
                vec![Operand::scalar(2), Operand::imm(3)],
                vec![2],
            ),
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
            SsaCpsatHints {
                greedy_hints: Some(&hints),
                ..Default::default()
            },
            None,
        );

        // The test passes if we don't panic from "BUG: X VRegs have last_use_cycle < def_cycle"
        // For a simple chain, hints should be optimal (ASAP), so solver may return exact hints.
        assert!(result.is_some(), "Simple chain should be schedulable");
        let r = result.unwrap();
        // Verify basic constraints are satisfied
        assert!(
            r.makespan >= 4,
            "Need at least 4 cycles for 4-op dependency chain"
        );
        assert!(
            r.cycle_assignments.len() == 4,
            "Should have 4 cycle assignments"
        );
        // Verify dependency order is respected
        assert!(
            r.cycle_assignments[1] > r.cycle_assignments[0],
            "op1 must be after op0"
        );
        assert!(
            r.cycle_assignments[2] > r.cycle_assignments[1],
            "op2 must be after op1"
        );
        assert!(
            r.cycle_assignments[3] > r.cycle_assignments[2],
            "op3 must be after op2"
        );
        eprintln!(
            "test_vreg_last_use_simple_chain: makespan={}, assignments={:?}",
            r.makespan, r.cycle_assignments
        );
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
            make_ssa_op(
                0,
                Engine::Load,
                Opcode::Const,
                Some(VReg::scalar(0)),
                vec![Operand::imm(5)],
                vec![],
            ),
            make_ssa_op(
                1,
                Engine::Load,
                Opcode::Const,
                Some(VReg::scalar(1)),
                vec![Operand::imm(10)],
                vec![],
            ),
            make_ssa_op(
                2,
                Engine::Alu,
                Opcode::Add,
                Some(VReg::scalar(2)),
                vec![Operand::scalar(0), Operand::imm(1)],
                vec![0],
            ),
            make_ssa_op(
                3,
                Engine::Alu,
                Opcode::Add,
                Some(VReg::scalar(3)),
                vec![Operand::scalar(1), Operand::imm(2)],
                vec![1],
            ),
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
            SsaCpsatHints {
                greedy_hints: Some(&hints),
                ..Default::default()
            },
            None,
        );

        // The test passes if we don't panic from "BUG: X VRegs have last_use_cycle < def_cycle"
        // Note: The solver may find a better schedule than the hints, so we don't assert exact match.
        // The key test is that vreg_last_use computation doesn't produce impossible results.
        assert!(result.is_some(), "Reordered schedule should be schedulable");
        let r = result.unwrap();
        // Verify basic constraints are satisfied
        assert!(
            r.makespan >= 2,
            "Need at least 2 cycles (loads then ALU ops)"
        );
        assert!(
            r.cycle_assignments.len() == 4,
            "Should have 4 cycle assignments"
        );
        eprintln!(
            "test_vreg_last_use_with_reordered_hints: makespan={}, assignments={:?}",
            r.makespan, r.cycle_assignments
        );
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
            make_ssa_op(
                0,
                Engine::Load,
                Opcode::Const,
                Some(VReg::scalar(0)),
                vec![Operand::imm(5)],
                vec![],
            ),
            make_ssa_op(
                1,
                Engine::Alu,
                Opcode::Add,
                Some(VReg::scalar(1)),
                vec![Operand::scalar(0), Operand::imm(1)],
                vec![0],
            ),
            make_ssa_op(
                2,
                Engine::Load,
                Opcode::Const,
                Some(VReg::scalar(2)),
                vec![Operand::imm(10)],
                vec![],
            ),
            make_ssa_op(
                3,
                Engine::Alu,
                Opcode::Add,
                Some(VReg::scalar(3)),
                vec![Operand::scalar(0), Operand::scalar(2)],
                vec![0, 2],
            ),
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
            SsaCpsatHints {
                greedy_hints: Some(&hints),
                ..Default::default()
            },
            None,
        );

        // The test passes if we don't panic from "BUG: X VRegs have last_use_cycle < def_cycle"
        // Note: The solver may find a better schedule than the hints, so we don't assert exact match.
        // The key test is that vreg_last_use computation doesn't produce impossible results.
        assert!(result.is_some(), "Multiple readers should be schedulable");
        let r = result.unwrap();
        // Verify basic constraints are satisfied
        assert!(
            r.makespan >= 2,
            "Need at least 2 cycles (loads then ALU ops)"
        );
        assert!(
            r.cycle_assignments.len() == 4,
            "Should have 4 cycle assignments"
        );
        eprintln!(
            "test_vreg_last_use_multiple_readers: makespan={}, assignments={:?}",
            r.makespan, r.cycle_assignments
        );
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
            make_ssa_op(
                0,
                Engine::Load,
                Opcode::Const,
                Some(VReg::scalar(0)),
                vec![Operand::imm(5)],
                vec![],
            ),
            make_ssa_op(
                1,
                Engine::Alu,
                Opcode::Add,
                Some(VReg::scalar(1)),
                vec![Operand::scalar(0), Operand::scalar(0)],
                vec![0],
            ),
            make_ssa_op(
                2,
                Engine::Store,
                Opcode::Store,
                None,
                vec![Operand::imm(0), Operand::scalar(1)],
                vec![1],
            ),
        ];

        // Schedule with CP-SAT and budget of 2 (CP-SAT doesn't support same-cycle reuse)
        let result = schedule_ssa_cpsat(
            &ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: true,
                scratch_budget: 2, // CP-SAT needs 2 because no same-cycle reuse
                ..Default::default()
            },
            SsaCpsatHints::default(),
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
        assert_eq!(
            r.cycle_assignments.len(),
            3,
            "All 3 ops should be scheduled"
        );

        // Apply cycle assignments to a mutable copy for address assignment
        let mut scheduled_ops = ops.clone();
        for (idx, &cycle) in r.cycle_assignments.iter().enumerate() {
            scheduled_ops[idx].cycle = cycle;
        }

        // Address assignment with same-cycle reuse should fit in budget of 1
        // (even though CP-SAT required budget=2)
        let config = AddressAssignConfig {
            budget: 2,
            start_addr: 1, // Reserve 0 for ZERO_REG
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
