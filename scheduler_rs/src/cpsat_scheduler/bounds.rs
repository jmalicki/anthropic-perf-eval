use std::collections::{HashMap, HashSet, VecDeque};

use crate::types::{Engine, MachineConfig, Op, Opcode, PrecedenceOffset, SsaOp, VReg};

use super::cse_groups::CseGroup;

use super::config::{
    ConstChoiceMode, CpsatResult, CpsatStatus, OptionalConstraintGroup, SelectChoiceMode,
    ValuShiftMode,
};

// ============================================================================
// Precedence Offset Diagnostics
// ============================================================================

/// Diagnose precedence bound violations in ASAP/ALAP values.
///
/// Checks that for every RAW edge (pred → succ):
/// - Forward (ASAP): `asap[succ] >= asap[pred] + pred.offset_for_asap(pred_is_shiftable)`
/// - Backward (ALAP): `alap[pred] <= alap[succ] - pred.offset_for_alap(pred_is_shiftable)`
///
/// Also checks domain consistency: `asap[op] <= alap[op]` for all ops.
///
/// # Returns
/// `(violation_count, sample_violations, empty_domain_count)` where:
/// - `violation_count` = number of edges with ASAP or ALAP violations
/// - `sample_violations` = up to 3 human-readable violation descriptions
/// - `empty_domain_count` = number of ops where asap > alap
pub(crate) fn diagnose_precedence_bounds(
    ops: &[SsaOp],
    asap: &HashMap<usize, i32>,
    alap: &HashMap<usize, i32>,
    shiftable_set: &HashSet<usize>,
) -> (usize, Vec<String>, usize) {
    let id_to_idx: HashMap<usize, usize> = ops
        .iter()
        .enumerate()
        .map(|(idx, op)| (op.id, idx))
        .collect();

    let mut violation_count = 0usize;
    let mut samples: Vec<String> = Vec::new();

    // Check each RAW edge
    for (succ_idx, succ_op) in ops.iter().enumerate() {
        for &dep_id in &succ_op.dep_ids {
            let pred_idx = match id_to_idx.get(&dep_id) {
                Some(&idx) => idx,
                None => continue,
            };
            let pred_op = &ops[pred_idx];
            let pred_is_shiftable = shiftable_set.contains(&pred_idx);
            let asap_offset = pred_op.precedence_offset_for_asap(pred_is_shiftable);
            // Use min_delay (asap offset) for both ASAP and ALAP domain checking.
            // Domain bounds use conservative (wide) offsets; the CP-SAT model
            // handles actual shifted latency via completion_cycle constraints.
            let alap_offset = pred_op.precedence_offset_for_asap(pred_is_shiftable);

            let pred_asap = *asap.get(&pred_op.id).unwrap_or(&0);
            let succ_asap = *asap.get(&succ_op.id).unwrap_or(&0);
            let pred_alap = *alap.get(&pred_op.id).unwrap_or(&0);
            let succ_alap = *alap.get(&succ_op.id).unwrap_or(&0);

            // Forward check: succ_asap >= pred_asap + asap_offset
            if succ_asap < pred_asap + asap_offset {
                violation_count += 1;
                if samples.len() < 3 {
                    samples.push(format!(
                        "ASAP violation: op[{}](asap={}) → op[{}](asap={}), expected >= {} (offset={})",
                        pred_op.id, pred_asap, succ_op.id, succ_asap,
                        pred_asap + asap_offset, asap_offset
                    ));
                }
            }

            // Backward check: pred_alap <= succ_alap - alap_offset
            if pred_alap > succ_alap - alap_offset {
                violation_count += 1;
                if samples.len() < 3 {
                    samples.push(format!(
                        "ALAP violation: op[{}](alap={}) → op[{}](alap={}), expected <= {} (offset={})",
                        pred_op.id, pred_alap, succ_op.id, succ_alap,
                        succ_alap - alap_offset, alap_offset
                    ));
                }
            }
        }
    }

    // Check domain consistency
    let mut empty_domain_count = 0usize;
    for op in ops {
        let op_asap = *asap.get(&op.id).unwrap_or(&0);
        let op_alap = *alap.get(&op.id).unwrap_or(&0);
        if op_asap > op_alap {
            empty_domain_count += 1;
            if samples.len() < 3 {
                samples.push(format!(
                    "Empty domain: op[{}] asap={} > alap={}",
                    op.id, op_asap, op_alap
                ));
            }
        }
    }

    (violation_count, samples, empty_domain_count)
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
    /// If true, allow eliminating vbroadcasts when all consumers are shifted to ALU (single scalar producer).
    /// Only applies when enable_valu_shift is true. Default: true when enable_valu_shift is true.
    pub enable_vbroadcast_elimination: bool,
    /// If true, force 0 integral gap for proven optimality (debugging only).
    /// Default: false (uses scale/3 gap limit for faster convergence).
    pub force_zero_gap: bool,
    /// If true, tighten ASAP/ALAP bounds by considering resource capacity.
    /// Uses energetic reasoning to tighten domains before creating cycle variables.
    /// Default: false (test first, validate carefully).
    pub enable_asap_alap_tightening: bool,
    /// If true, consider scratch constraints when tightening ASAP/ALAP bounds.
    /// Can be used independently of enable_asap_alap_tightening. Computes maximum scratch usage
    /// over VReg lifetimes and tightens bounds to avoid cycles where scratch exceeds budget.
    /// Default: false (test first, validate carefully).
    pub enable_scratch_tightening: bool,
    /// If true, enable verbose logging of the precedence fixpoint repair pass.
    /// The fixpoint always runs after slot/scratch tightening to restore precedence consistency.
    /// This flag just controls whether to log the number of ASAP/ALAP changes.
    pub debug_fix_precedence_bounds: bool,
    /// Optional constraint groups to disable for debugging. When a group is in this set, the
    /// corresponding constraints are not added (they are redundant/helper constraints that help
    /// propagation but are not required for correctness). Empty by default (all optional groups on).
    pub disable_optional_constraint_groups: HashSet<OptionalConstraintGroup>,
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
            enable_vbroadcast_elimination: false, // Set to true by caller when enable_valu_shift is true if desired
            force_zero_gap: false, // Default: use scale/3 gap limit for faster convergence
            enable_asap_alap_tightening: true, // Enabled by default (validated on 4r/depth-3). Use --no-enable-asap-alap-tightening to disable.
            enable_scratch_tightening: false, // Default: disabled for ablation studies
            debug_fix_precedence_bounds: false, // Default: disabled
            disable_optional_constraint_groups: HashSet::new(),
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

/// Minimum VALU ops from SelectChoice when `z` consumers choose FMA and CSE groups exist:
/// one SUB per group that has at least one FMA + z FMAs. Minimized by concentrating z in as few groups as possible.
fn min_valu_from_select_cse(z: i64, group_sizes: &[usize]) -> i64 {
    if z <= 0 {
        return 0;
    }
    let z = z as usize;
    let mut sizes: Vec<usize> = group_sizes.to_vec();
    sizes.sort_by(|a, b| b.cmp(a));
    let mut sum = 0usize;
    for (k, &s) in sizes.iter().enumerate() {
        sum += s;
        if sum >= z {
            return (k + 1) as i64 + z as i64;
        }
    }
    (sizes.len() + z) as i64
}

/// Compute capacity-based lower bound on makespan by minimizing over choice variables (x, y, z).
/// No CP-SAT needed: pure triple loop over small integer ranges.
///
/// - x = number of shiftable VALU that stay on VALU (0..=shiftable_count)
/// - y = number of ConstChoice that choose add_imm/Flow (0..=const_choice_count)
/// - z = number of SelectChoice that choose FMA/VALU (0..=select_choice_count).
///   Without CSE: each FMA adds 2 VALU ops (sub + fma). With CSE: (1 SUB per active group) + z FMAs.
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
    cse_groups: Option<&[CseGroup]>,
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
    let cse_group_sizes: Option<Vec<usize>> = cse_groups.map(|g| g.iter().map(|grp| grp.consumer_indices.len()).collect());
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
                let valu_select = match &cse_group_sizes {
                    Some(sizes) if !sizes.is_empty() => min_valu_from_select_cse(z, sizes),
                    _ => 2 * z,
                };
                let valu_with_select = valu_ops + valu_select;
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
    /// Op IDs of vbroadcasts to hint for elimination (try eliminate_vb = true).
    pub eliminate_vbroadcast_hints: Option<&'a HashSet<usize>>,
    /// CSE groups for shared SUB. When set and non-empty, CP-SAT uses shared-SUB model for those groups.
    pub cse_groups: Option<&'a [super::cse_groups::CseGroup]>,
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
    /// ConstChoice ops that chose ALU-derived encoding: op_id -> AluDeriveInfo
    pub const_alu_derive_ops: HashMap<usize, crate::types::AluDeriveInfo>,
    /// Set of SelectChoice op IDs that should use FMA path (VALU) instead of vselect (Flow)
    pub select_fma_ops: HashSet<usize>,
    /// Number of empty cycles (makespan minus max cycle used). 0 when schedule has no gaps.
    pub empty_cycle_count: usize,
    /// True when makespan equals capacity lower bound (structural bound caused the gap).
    pub makespan_driven_by_capacity: bool,
    /// Proven lower bound on makespan (cycles). When status is Optimal, equals makespan; when Feasible, solver's best_bound in cycles.
    pub lb_cycles: Option<usize>,
    /// Op IDs of vbroadcasts that were eliminated (not emitted; consumers use scalar producer).
    pub eliminated_vbroadcast_ops: HashSet<usize>,
    /// For each eliminated vb op ID, the op ID of its scalar producer (for lowering).
    pub eliminated_vb_to_scalar_producer: HashMap<usize, usize>,
    /// CSE group results (Phase A: sub_active, sub_cycle). Indexed by group index. Empty when no CSE groups.
    pub cse_group_results: Vec<super::cse_groups::CseGroupResult>,
    /// Same constraints as the CP-SAT model (for bulk-move simulator and validation).
    pub schedule_constraints: Vec<crate::schedule_constraints::ScheduleConstraint>,
}

/// Compute ASAP (As Soon As Possible) cycle for each SSA op.
/// Uses the RAW-only dependencies from dep_ids, with per-edge precedence offsets.
///
/// # Arguments
/// * `ops` - SSA operations to compute bounds for
/// * `shiftable_set` - Set of op *indices* that are shiftable VALU ops
pub fn compute_ssa_asap(ops: &[SsaOp], shiftable_set: &HashSet<usize>) -> HashMap<usize, i32> {
    let mut asap: HashMap<usize, i32> = HashMap::new();
    let mut in_degree: HashMap<usize, usize> = HashMap::new();

    // Build id → index mapping for looking up ops by their id
    let id_to_idx: HashMap<usize, usize> = ops
        .iter()
        .enumerate()
        .map(|(idx, op)| (op.id, idx))
        .collect();

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
        let pred_idx = id_to_idx[&op_id];
        let pred_is_shiftable = shiftable_set.contains(&pred_idx);
        // Offset is determined by the predecessor (this op): how many cycles until its result is visible
        let offset = ops[pred_idx].precedence_offset_for_asap(pred_is_shiftable);

        if let Some(succs) = successors.get(&op_id) {
            for &succ_id in succs {
                let new_asap = my_asap + offset;
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
///
/// # Arguments
/// * `ops` - SSA operations to compute bounds for
/// * `horizon` - Maximum cycle (ALAP for sinks = horizon - 1)
/// * `shiftable_set` - Set of op *indices* that are shiftable VALU ops
pub fn compute_ssa_alap(ops: &[SsaOp], horizon: i32, shiftable_set: &HashSet<usize>) -> HashMap<usize, i32> {
    let mut alap: HashMap<usize, i32> = HashMap::new();

    // Build id → index mapping for looking up ops by their id
    let id_to_idx: HashMap<usize, usize> = ops
        .iter()
        .enumerate()
        .map(|(idx, op)| (op.id, idx))
        .collect();

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
    // In ALAP, we propagate from consumer (op_id) backward to its predecessors (dep_id).
    // The predecessor (dep_id) determines the offset (its result latency).
    while let Some(op_id) = queue.pop_front() {
        let my_alap = *alap.get(&op_id).unwrap_or(&(horizon - 1));

        if let Some(op) = op_by_id.get(&op_id) {
            for &dep_id in &op.dep_ids {
                // dep_id is the predecessor — offset is determined by the predecessor
                let pred_idx = id_to_idx[&dep_id];
                let pred_is_shiftable = shiftable_set.contains(&pred_idx);
                // Use min_delay (asap offset) for domain bounding, not max_delay.
                // The CP-SAT model's completion_cycle constraints (section 5c) handle
                // the actual shifted latency; wide domains let the solver decide.
                let offset = ops[pred_idx].precedence_offset_for_asap(pred_is_shiftable);
                let new_alap = my_alap - offset;
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

/// Compute the maximum-latency critical path length using ALAP offsets.
///
/// This does a forward ASAP-like pass but uses `precedence_offset_for_alap` (max offsets)
/// instead of `precedence_offset_for_asap` (min offsets). The result determines the minimum
/// horizon needed so that ALAP backward propagation doesn't produce negative values.
pub fn compute_max_latency_critical_path(ops: &[SsaOp], shiftable_set: &HashSet<usize>) -> i32 {
    let mut max_asap: HashMap<usize, i32> = HashMap::new();
    let mut in_degree: HashMap<usize, usize> = HashMap::new();
    let id_to_idx: HashMap<usize, usize> = ops.iter().enumerate().map(|(i, op)| (op.id, i)).collect();
    for op in ops {
        in_degree.insert(op.id, op.dep_ids.len());
    }
    let mut queue: VecDeque<usize> = VecDeque::new();
    for op in ops {
        if op.dep_ids.is_empty() {
            queue.push_back(op.id);
            max_asap.insert(op.id, 0);
        }
    }
    let mut successors: HashMap<usize, Vec<usize>> = HashMap::new();
    for op in ops {
        for &dep_id in &op.dep_ids {
            successors.entry(dep_id).or_default().push(op.id);
        }
    }
    while let Some(op_id) = queue.pop_front() {
        let my_asap = *max_asap.get(&op_id).unwrap_or(&0);
        let pred_idx = id_to_idx[&op_id];
        let pred_is_shiftable = shiftable_set.contains(&pred_idx);
        let offset = ops[pred_idx].precedence_offset_for_alap(pred_is_shiftable);
        if let Some(succs) = successors.get(&op_id) {
            for &succ_id in succs {
                let new_asap = my_asap + offset;
                let current = max_asap.entry(succ_id).or_insert(0);
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
    max_asap.values().max().copied().unwrap_or(0) + 1
}

/// Tighten ASAP/ALAP bounds using slot capacity constraints (energetic reasoning).
/// Uses difference arrays and prefix sums for O(n + horizon) complexity.
pub(crate) fn tighten_asap_alap_by_slots(
    ops: &[SsaOp],
    asap: &HashMap<usize, i32>,
    alap: &HashMap<usize, i32>,
    horizon: i32,
    valu_cap: i64,
    alu_cap: i64,
    load_cap: i64,
    store_cap: i64,
    flow_cap: i64,
    verbose: bool,
) -> Option<(HashMap<usize, i32>, HashMap<usize, i32>)> {
    use std::collections::HashMap as StdHashMap;

    let n = ops.len();
    let horizon_usize = horizon as usize;

    // Build engine capacity map
    let mut engine_caps: StdHashMap<Engine, i64> = StdHashMap::new();
    engine_caps.insert(Engine::Valu, valu_cap);
    engine_caps.insert(Engine::Alu, alu_cap);
    engine_caps.insert(Engine::Load, load_cap);
    engine_caps.insert(Engine::Store, store_cap);
    engine_caps.insert(Engine::Flow, flow_cap);

    // Track which ops belong to which engine
    let mut ops_by_engine: StdHashMap<Engine, Vec<usize>> = StdHashMap::new();
    for (idx, op) in ops.iter().enumerate() {
        if engine_caps.contains_key(&op.engine) {
            ops_by_engine.entry(op.engine).or_default().push(idx);
        }
    }

    let mut new_asap = asap.clone();
    let mut new_alap = alap.clone();
    let mut changed = true;
    let mut iter = 0;
    const MAX_ITER: usize = 3; // Limit iterations

    while changed && iter < MAX_ITER {
        changed = false;
        iter += 1;

        // For each engine, build resource profile and tighten bounds
        for (engine, cap) in &engine_caps {
            let engine_ops = match ops_by_engine.get(engine) {
                Some(ops) => ops,
                None => continue,
            };
            if engine_ops.is_empty() {
                continue;
            }

            // Build resource profile using difference array
            // profile[t] = number of ops on this engine that CAN start at cycle t
            let mut diff = vec![0i64; horizon_usize + 2];
            for &idx in engine_ops {
                let op_id = ops[idx].id;
                let asap_t = *new_asap.get(&op_id)? as usize;
                let alap_t = *new_alap.get(&op_id)? as usize;
                if asap_t <= alap_t && alap_t < horizon_usize {
                    diff[asap_t] += 1;
                    if alap_t + 1 < diff.len() {
                        diff[alap_t + 1] -= 1;
                    }
                }
            }

            // Compute prefix sums to get actual profile
            let mut profile = vec![0i64; horizon_usize + 1];
            let mut sum = 0i64;
            for t in 0..=horizon_usize {
                sum += diff[t];
                profile[t] = sum;
            }

            // Build prefix sums of profile for O(1) window queries
            let mut prefix = vec![0i64; horizon_usize + 2];
            for t in 0..=horizon_usize {
                prefix[t + 1] = prefix[t] + profile[t];
            }

            // For each op on this engine, check if window has enough capacity
            for &idx in engine_ops {
                let op_id = ops[idx].id;
                let old_asap = *new_asap.get(&op_id)?;
                let old_alap = *new_alap.get(&op_id)?;
                let window_size = (old_alap - old_asap + 1) as i64;

                if window_size <= 0 {
                    continue;
                }

                // Count total ops that CAN start in this window
                let window_start = old_asap as usize;
                let window_end = old_alap as usize;
                if window_end >= prefix.len() - 1 {
                    continue;
                }
                let total_ops_in_window = prefix[window_end + 1] - prefix[window_start];

                // Check if window is too small (energetic reasoning)
                // If total ops > capacity * window_size, we need a larger window
                if total_ops_in_window > *cap * window_size {
                    // Tighten ASAP forward: find earliest cycle where there's enough capacity
                    let mut new_asap_val = old_asap;
                    for t in (old_asap as usize)..=(old_alap as usize) {
                        if t >= prefix.len() - 1 {
                            break;
                        }
                        // Check if window [t, old_alap] has enough capacity
                        let remaining_window = (old_alap - t as i32 + 1) as i64;
                        if remaining_window <= 0 {
                            break;
                        }
                        let ops_in_remaining = prefix[old_alap as usize + 1] - prefix[t];
                        if ops_in_remaining <= *cap * remaining_window {
                            new_asap_val = t as i32;
                            break;
                        }
                    }

                    // Tighten ALAP backward: find latest cycle where there's enough capacity
                    let mut new_alap_val = old_alap;
                    for t in ((old_asap as usize)..=(old_alap as usize)).rev() {
                        if t >= prefix.len() - 1 {
                            continue;
                        }
                        // Check if window [old_asap, t] has enough capacity
                        let remaining_window = (t as i32 - old_asap + 1) as i64;
                        if remaining_window <= 0 {
                            break;
                        }
                        let ops_in_remaining = prefix[t + 1] - prefix[old_asap as usize];
                        if ops_in_remaining <= *cap * remaining_window {
                            new_alap_val = t as i32;
                            break;
                        }
                    }

                    // Update bounds if tightened
                    if new_asap_val > old_asap || new_alap_val < old_alap {
                        if new_asap_val > old_asap {
                            new_asap.insert(op_id, new_asap_val);
                            changed = true;
                        }
                        if new_alap_val < old_alap {
                            new_alap.insert(op_id, new_alap_val);
                            changed = true;
                        }
                    }
                }
            }
        }
    }

    if verbose && iter > 0 {
        eprintln!(
            "  Slot-aware tightening: {} iterations, tightened {} ops",
            iter,
            new_asap.len().max(new_alap.len())
        );
    }

    Some((new_asap, new_alap))
}

/// Tighten ASAP/ALAP bounds using scratch memory constraints.
pub(crate) fn tighten_asap_alap_by_scratch(
    ops: &[SsaOp],
    asap: &HashMap<usize, i32>,
    alap: &HashMap<usize, i32>,
    horizon: i32,
    scratch_budget: i32,
    verbose: bool,
) -> Option<(HashMap<usize, i32>, HashMap<usize, i32>)> {
    use std::collections::HashMap as StdHashMap;

    // Build VReg -> defining op index mapping
    let mut vreg_def_op: StdHashMap<VReg, usize> = StdHashMap::new();
    for (idx, op) in ops.iter().enumerate() {
        if let Some(dest) = op.dest {
            vreg_def_op.insert(dest, idx);
        }
    }

    // Build VReg -> last use op index mapping (conservative: use latest index)
    let mut vreg_last_use: StdHashMap<VReg, usize> = StdHashMap::new();
    for (idx, op) in ops.iter().enumerate() {
        for vreg in op.reads() {
            vreg_last_use.insert(vreg, idx);
        }
    }

    // Collect VRegs that have both definition and use
    let live_vregs: Vec<VReg> = vreg_def_op
        .keys()
        .filter(|v| vreg_last_use.contains_key(v))
        .copied()
        .collect();

    if live_vregs.is_empty() {
        return Some((asap.clone(), alap.clone()));
    }

    // Build maximum scratch usage profile over cycles
    // For each cycle t, compute maximum scratch usage assuming worst-case scheduling
    // (VRegs live for their longest possible lifetime: def at ASAP, use at ALAP)
    let horizon_usize = horizon as usize;
    let mut max_scratch = vec![0i32; horizon_usize + 1];

    // For each VReg, determine its maximum lifetime based on op windows
    for vreg in &live_vregs {
        let def_idx = *vreg_def_op.get(vreg)?;
        let use_idx = *vreg_last_use.get(vreg)?;
        let def_op_id = ops[def_idx].id;
        let use_op_id = ops[use_idx].id;

        let def_asap = *asap.get(&def_op_id)? as usize;
        let def_alap = *alap.get(&def_op_id)? as usize;
        let use_asap = *asap.get(&use_op_id)? as usize;
        let use_alap = *alap.get(&use_op_id)? as usize;

        // Maximum lifetime: def at ASAP, use at ALAP (longest possible)
        // This gives us the worst-case scratch usage
        let max_def = def_asap;
        let max_use = use_alap.min(horizon_usize);

        // VReg is live from def cycle to use cycle (inclusive)
        // Update scratch profile for this maximum lifetime
        for t in max_def..=max_use {
            if t <= horizon_usize {
                max_scratch[t] += 1;
            }
        }
    }

    // Find cycles where max scratch exceeds budget
    let mut new_asap = asap.clone();
    let mut new_alap = alap.clone();
    let mut changed = false;

    // For each op, if its window includes cycles where max scratch > budget,
    // tighten bounds to avoid those cycles
    for (idx, op) in ops.iter().enumerate() {
        let op_id = op.id;
        let old_asap = *new_asap.get(&op_id)?;
        let old_alap = *new_alap.get(&op_id)?;

        // Find earliest cycle in window where scratch is within budget
        let mut new_asap_val = old_asap;
        for t in (old_asap as usize)..=(old_alap as usize) {
            if t > horizon_usize {
                break;
            }
            if max_scratch[t] <= scratch_budget {
                new_asap_val = t as i32;
                break;
            }
        }

        // Find latest cycle in window where scratch is within budget
        let mut new_alap_val = old_alap;
        for t in ((old_asap as usize)..=(old_alap as usize)).rev() {
            if t > horizon_usize {
                continue;
            }
            if max_scratch[t] <= scratch_budget {
                new_alap_val = t as i32;
                break;
            }
        }

        // Update bounds if tightened
        if new_asap_val > old_asap || new_alap_val < old_alap {
            if new_asap_val > old_asap {
                new_asap.insert(op_id, new_asap_val);
                changed = true;
            }
            if new_alap_val < old_alap {
                new_alap.insert(op_id, new_alap_val);
                changed = true;
            }
        }
    }

    if verbose && changed {
        eprintln!(
            "  Scratch-aware tightening: tightened {} ops to avoid scratch budget violations",
            new_asap.len().max(new_alap.len())
        );
    }

    Some((new_asap, new_alap))
}

/// Build ordering pairs (i, j) for symmetry breaking: cycle[i] <= cycle[j].
///
/// Sort each group by greedy cycle (primary, when available) then ALAP (tiebreaker).
/// After sorting, verify that ASAP[i] <= ALAP[j] for each consecutive pair — this is
/// required for the constraint `cycle[i] <= cycle[j]` to be satisfiable. If greedy and
/// ASAP/ALAP are inconsistent, panic (indicates a bug in greedy or bounds computation).
pub(crate) fn build_ordering_pairs_from_bounds(
    ops: &[SsaOp],
    asap: &HashMap<usize, i32>,
    alap: &HashMap<usize, i32>,
    greedy_hints: Option<&[i32]>,
) -> Vec<(usize, usize)> {
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
    let mut pairs = Vec::new();
    for (group_key, indices) in &groups {
        let mut sorted: Vec<usize> = indices.clone();
        if let Some(hints) = greedy_hints {
            // Primary: greedy cycle. Tiebreaker: ALAP (for determinism and domain consistency).
            sorted.sort_by_key(|&idx| {
                let greedy = hints.get(idx).copied().unwrap_or(0);
                let alap_val = *alap.get(&ops[idx].id).unwrap_or(&0);
                (greedy, alap_val)
            });
        } else {
            sorted.sort_by_key(|&idx| *alap.get(&ops[idx].id).unwrap_or(&0));
        }
        // Verify domain consistency: ASAP[i] <= ALAP[j] for each consecutive pair.
        for w in sorted.windows(2) {
            let i = w[0];
            let j = w[1];
            let asap_i = *asap.get(&ops[i].id).unwrap_or(&0);
            let alap_j = *alap.get(&ops[j].id).unwrap_or(&0);
            if asap_i > alap_j {
                panic!(
                    "Symmetry ordering inconsistency: ordering group {:?} requires cycle[{}] <= cycle[{}], \
                     but ASAP[{}]={} > ALAP[{}]={}. The constraint cycle[i] <= cycle[j] is unsatisfiable. \
                     Greedy hints: i={}, j={}. Fix: greedy or ASAP/ALAP bounds are inconsistent.",
                    group_key, i, j, i, asap_i, j, alap_j,
                    greedy_hints.and_then(|h| h.get(i).copied()).unwrap_or(-1),
                    greedy_hints.and_then(|h| h.get(j).copied()).unwrap_or(-1),
                );
            }
            pairs.push((i, j));
        }
    }
    pairs
}

/// Fixpoint pass to repair precedence violations in ASAP/ALAP bounds.
///
/// After tightening passes (slot, scratch) may have narrowed bounds without respecting
/// precedence offsets, this pass does bidirectional propagation:
/// - Forward: increase ASAP[succ] to satisfy `asap[succ] >= asap[pred] + pred.offset_for_asap()`
/// - Backward: decrease ALAP[pred] to satisfy `alap[pred] <= alap[succ] - pred.offset_for_alap()`
///
/// Iterates until convergence (bounded by ops.len() iterations).
///
/// # Returns
/// New (asap, alap) maps with all precedence constraints satisfied.
pub(crate) fn fix_precedence_bounds_fixpoint(
    ops: &[SsaOp],
    asap: &HashMap<usize, i32>,
    alap: &HashMap<usize, i32>,
    shiftable_set: &HashSet<usize>,
) -> (HashMap<usize, i32>, HashMap<usize, i32>) {
    let mut current_asap = asap.clone();
    let mut current_alap = alap.clone();

    let id_to_idx: HashMap<usize, usize> = ops
        .iter()
        .enumerate()
        .map(|(idx, op)| (op.id, idx))
        .collect();

    // Build successor map: pred_id → [succ_id]
    let mut successors: HashMap<usize, Vec<usize>> = HashMap::new();
    for op in ops {
        for &dep_id in &op.dep_ids {
            successors.entry(dep_id).or_default().push(op.id);
        }
    }

    let max_iters = ops.len();
    for _ in 0..max_iters {
        let mut changed = false;

        // Forward pass: propagate ASAP increases
        for op in ops {
            let pred_idx = id_to_idx[&op.id];
            let pred_is_shiftable = shiftable_set.contains(&pred_idx);
            let offset = op.precedence_offset_for_asap(pred_is_shiftable);
            let pred_asap = *current_asap.get(&op.id).unwrap_or(&0);

            if let Some(succs) = successors.get(&op.id) {
                for &succ_id in succs {
                    let required = pred_asap + offset;
                    let current = current_asap.entry(succ_id).or_insert(0);
                    if required > *current {
                        *current = required;
                        changed = true;
                    }
                }
            }
        }

        // Backward pass: propagate ALAP decreases (reverse iteration)
        for op in ops.iter().rev() {
            for &dep_id in &op.dep_ids {
                let pred_idx = id_to_idx[&dep_id];
                let pred_is_shiftable = shiftable_set.contains(&pred_idx);
                // Use min_delay for domain bounding (same as compute_ssa_alap)
                let offset = ops[pred_idx].precedence_offset_for_asap(pred_is_shiftable);
                let succ_alap = *current_alap.get(&op.id).unwrap_or(&0);
                let required = succ_alap - offset;
                let current = current_alap.entry(dep_id).or_insert(required);
                if required < *current {
                    *current = required;
                    changed = true;
                }
            }
        }

        if !changed {
            break;
        }
    }

    (current_asap, current_alap)
}

/// Compute ASAP/ALAP bounds (with optional slot/scratch tightening) for use before greedy scheduling.
/// Call this before the greedy pass so greedy can respect the same domains CP-SAT will use.
/// Horizon is computed without greedy hints (2 * critical path, or n).
#[cfg(feature = "cpsat")]
pub fn compute_ssa_asap_alap_bounds(
    ops: &[SsaOp],
    config: &SsaCpsatConfig,
) -> (HashMap<usize, i32>, HashMap<usize, i32>) {
    // Compute shiftable set from config (same logic as schedule_ssa_cpsat)
    let shiftable_set: HashSet<usize> = if config.enable_valu_shift {
        ops.iter()
            .enumerate()
            .filter(|(_, op)| op.engine == Engine::Valu && op.opcode.can_shift_to_alu())
            .map(|(idx, _)| idx)
            .collect()
    } else {
        HashSet::new()
    };

    let asap = compute_ssa_asap(ops, &shiftable_set);
    let n = ops.len();
    let critical_path_len = asap.values().max().copied().unwrap_or(0) + 1;
    let max_latency_critical_path = compute_max_latency_critical_path(ops, &shiftable_set);
    let horizon = (critical_path_len * 2).max(max_latency_critical_path * 2).max(n as i32);
    let alap = compute_ssa_alap(ops, horizon, &shiftable_set);
    let mut current_asap = asap;
    let mut current_alap = alap;
    let machine = &config.machine;
    let valu_cap = machine.valu_slots as i64;
    let alu_cap = machine.alu_slots as i64;
    let load_cap = machine.load_slots as i64;
    let store_cap = machine.store_slots as i64;
    let flow_cap = machine.flow_slots as i64;
    let scratch_budget = config.scratch_budget as i32;
    let verbose = false;

    if config.enable_asap_alap_tightening {
        if let Some((slot_asap, slot_alap)) = tighten_asap_alap_by_slots(
            ops,
            &current_asap,
            &current_alap,
            horizon,
            valu_cap,
            alu_cap,
            load_cap,
            store_cap,
            flow_cap,
            verbose,
        ) {
            current_asap = slot_asap;
            current_alap = slot_alap;
        }
    }

    if config.enable_scratch_tightening {
        if let Some((scratch_asap, scratch_alap)) = tighten_asap_alap_by_scratch(
            ops,
            &current_asap,
            &current_alap,
            horizon,
            scratch_budget,
            verbose,
        ) {
            current_asap = scratch_asap;
            current_alap = scratch_alap;
        }
    }

    // After tightening, re-propagate precedence constraints to restore consistency.
    // Slot/scratch tightening is precedence-unaware, so it can violate precedence edges.
    if config.enable_asap_alap_tightening || config.enable_scratch_tightening {
        let (fixed_asap, fixed_alap) =
            fix_precedence_bounds_fixpoint(ops, &current_asap, &current_alap, &shiftable_set);
        current_asap = fixed_asap;
        current_alap = fixed_alap;
    }

    (current_asap, current_alap)
}
