use std::str::FromStr;

#[cfg(feature = "cpsat")]
use cp_sat::proto::CpSolverStatus;

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

/// Optional constraint group: redundant/helper constraints that can be disabled for debugging.
/// Each variant corresponds to a named group in the CP-SAT model (see model.rs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OptionalConstraintGroup {
    /// makespan_ge_load_lb, makespan_ge_store_lb (redundant with domain + cumulatives).
    RedundantLinearBounds,
    /// Linear constraint: 6*makespan >= valu_count - shift + 2*select_fma.
    CapacityValu,
    /// Linear constraint: 12*makespan - 8*shift >= alu_count.
    CapacityAlu,
    /// Linear constraint: makespan + z - y >= flow_base.
    CapacityFlow,
    /// Linear constraint: 2*makespan + y >= load_count.
    CapacityLoad,
}

impl FromStr for OptionalConstraintGroup {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "redundant_linear_bounds" => Ok(Self::RedundantLinearBounds),
            "capacity_valu" => Ok(Self::CapacityValu),
            "capacity_alu" => Ok(Self::CapacityAlu),
            "capacity_flow" => Ok(Self::CapacityFlow),
            "capacity_load" => Ok(Self::CapacityLoad),
            _ => Err(format!(
                "invalid optional constraint group '{}', expected: redundant_linear_bounds, capacity_valu, capacity_alu, capacity_flow, capacity_load",
                s
            )),
        }
    }
}

impl std::fmt::Display for OptionalConstraintGroup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::RedundantLinearBounds => "redundant_linear_bounds",
            Self::CapacityValu => "capacity_valu",
            Self::CapacityAlu => "capacity_alu",
            Self::CapacityFlow => "capacity_flow",
            Self::CapacityLoad => "capacity_load",
        };
        write!(f, "{}", s)
    }
}

// ============================================================================
// Solver tuning (ablation switches)
// ============================================================================

/// Full-problem subsolver names OR-Tools 9.15 defines (`GetNamedParameters` in
/// `ortools/sat/cp_model_search.cc`). Naming any other one in `subsolvers` or
/// `extra_subsolvers` makes the solver abort.
pub const KNOWN_FULL_SUBSOLVERS: [&str; 41] = [
    "auto", "core", "core_default_lp", "core_max_lp", "default", "default_lp", "fixed", "fixed_max_lp",
    "fixed_no_lp", "lb_tree_search", "lb_tree_search_no_lp", "less_encoding", "lns", "lns_base",
    "lns_routing", "lns_stalling", "max_hs", "max_lp", "max_lp_sym", "no_lp", "objective_lb_search",
    "objective_lb_search_max_lp", "objective_lb_search_no_lp", "objective_shaving",
    "objective_shaving_max_lp", "objective_shaving_no_lp", "portfolio", "portfolio_max_lp",
    "portfolio_no_lp", "probing", "probing_max_lp", "probing_no_lp", "pseudo_costs", "quick_restart",
    "quick_restart_max_lp", "quick_restart_no_lp", "reduced_costs", "shared_tree", "variables_shaving",
    "variables_shaving_max_lp", "variables_shaving_no_lp",
];

/// Switchable CP-SAT configuration and modelling choices for the full-kernel solve.
///
/// Every field defaults to what the scheduler did before the switch existed, so
/// `CpsatTuning::default()` builds the same model and the same solver parameters. The
/// per-section feasibility solves do not read this struct.
///
/// Carried across the Python binding as JSON (`cpsat_tuning_json`); unknown keys are
/// rejected and missing keys take the default.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CpsatTuning {
    /// `num_workers` (default 8). Ignored (forced to 1) when the solve stops at the first solution.
    pub workers: i32,
    /// `interleave_search` (default true): deterministic round-robin of subsolvers. OR-Tools
    /// 9.15 does not start the probing, shaving, `lb_tree_search` and `objective_lb_search`
    /// workers in this mode. With false the parallel search is not reproducible.
    pub interleave_search: bool,
    /// `subsolvers`: replaces CP-SAT's default list of full-problem subsolvers (empty = default list).
    pub subsolvers: Vec<String>,
    /// `extra_subsolvers`: appended to the list of full-problem subsolvers.
    pub extra_subsolvers: Vec<String>,
    /// `ignore_subsolvers`: removed from the list of subsolvers (full-problem and LNS).
    pub ignore_subsolvers: Vec<String>,
}

impl Default for CpsatTuning {
    fn default() -> Self {
        Self {
            workers: 8,
            interleave_search: true,
            subsolvers: Vec::new(),
            extra_subsolvers: Vec::new(),
            ignore_subsolvers: Vec::new(),
        }
    }
}

impl CpsatTuning {
    /// Parse a tuning from JSON, filling missing keys with the defaults.
    ///
    /// # Arguments
    /// * `json` - A JSON object whose keys are field names of [`CpsatTuning`]; `None` or a
    ///   blank string means "all defaults".
    ///
    /// # Returns
    /// The tuning, or a message naming the unknown key, wrong type or out-of-range value.
    pub fn from_json(json: Option<&str>) -> Result<Self, String> {
        let tuning: Self = match json.map(str::trim).filter(|s| !s.is_empty()) {
            None => Self::default(),
            Some(text) => {
                // serde would also accept a positional array for a struct; only objects are meant.
                let value: serde_json::Value = serde_json::from_str(text)
                    .map_err(|e| format!("invalid cpsat_tuning_json: {}", e))?;
                if !value.is_object() {
                    return Err(format!("invalid cpsat_tuning_json: expected a JSON object, got {}", value));
                }
                serde_json::from_value(value).map_err(|e| format!("invalid cpsat_tuning_json: {}", e))?
            }
        };
        tuning.validate()?;
        Ok(tuning)
    }

    /// Check value ranges that the types do not express.
    ///
    /// # Returns
    /// `Err` naming the offending field when a value cannot be passed to the solver.
    pub fn validate(&self) -> Result<(), String> {
        if self.workers < 1 {
            return Err(format!("cpsat tuning: workers must be at least 1, got {}", self.workers));
        }
        for (field, names) in [
            ("subsolvers", &self.subsolvers),
            ("extra_subsolvers", &self.extra_subsolvers),
            ("ignore_subsolvers", &self.ignore_subsolvers),
        ] {
            if let Some(bad) = names.iter().find(|n| n.is_empty() || n.trim() != n.as_str()) {
                return Err(format!("cpsat tuning: {} has an empty or padded name {:?}", field, bad));
            }
        }
        // CP-SAT aborts the process on an unknown full-problem subsolver name, so check here.
        // (ignore_subsolvers is a filter and may name LNS and other subsolvers too.)
        for (field, names) in [("subsolvers", &self.subsolvers), ("extra_subsolvers", &self.extra_subsolvers)] {
            if let Some(bad) = names.iter().find(|n| !KNOWN_FULL_SUBSOLVERS.contains(&n.as_str())) {
                return Err(format!(
                    "cpsat tuning: {} names {:?}, which is not a full-problem subsolver of OR-Tools 9.15; known: {}",
                    field,
                    bad,
                    KNOWN_FULL_SUBSOLVERS.join(", ")
                ));
            }
        }
        Ok(())
    }

    /// The tuning as a JSON object with every field present (recorded in `run_options`).
    ///
    /// # Returns
    /// A JSON string that [`CpsatTuning::from_json`] parses back to `self`.
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).expect("CpsatTuning serialises to JSON")
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
