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

/// Timeline of the scratch cumulative (see `model::scratch_interval`).
///
/// Within a cycle every read happens before every write, so a word last read at cycle C can
/// be rewritten at cycle C (address assignment's same-cycle reuse).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScratchTime {
    /// Half-open `[write, last read + 1)` per value: at cycle C it counts both the values
    /// last read at C and the values written at C, an overcount of the real load (kept for
    /// ablation).
    Cycle,
    /// Same-cycle reuse: `[write, last read)` per value, whose peak is exactly the peak of
    /// address assignment's read and write phases (default).
    #[default]
    Phase,
}

impl FromStr for ScratchTime {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "cycle" => Ok(Self::Cycle),
            "phase" => Ok(Self::Phase),
            _ => Err(format!("invalid cpsat_scratch_time '{}', expected cycle or phase", s)),
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
    /// Per-op rows `makespan >= cycle + 1`, enforced by the op's mode literal where it has one
    /// (`makespan_ge_c_*`, `makespan_ge_valu_*`, `makespan_ge_shifted_*`, `makespan_ge_select_*`).
    /// Implied by `makespan = max_cycle + 1` and the unconditional `max_cycle >= ...` rows.
    ConditionalMakespan,
    /// The capacity-1 Flow cumulative (`cumul_flow`), when the machine has one Flow slot:
    /// `no_overlap_flow` over the same intervals states the same thing.
    FlowCumulative,
}

impl OptionalConstraintGroup {
    /// Every group, in the order the names are listed to users.
    pub const ALL: [OptionalConstraintGroup; 7] = [
        Self::RedundantLinearBounds,
        Self::CapacityValu,
        Self::CapacityAlu,
        Self::CapacityFlow,
        Self::CapacityLoad,
        Self::ConditionalMakespan,
        Self::FlowCumulative,
    ];
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
            "conditional_makespan" => Ok(Self::ConditionalMakespan),
            "flow_cumulative" => Ok(Self::FlowCumulative),
            _ => Err(format!(
                "invalid optional constraint group '{}', expected: redundant_linear_bounds, capacity_valu, capacity_alu, capacity_flow, capacity_load, conditional_makespan, flow_cumulative",
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
            Self::ConditionalMakespan => "conditional_makespan",
            Self::FlowCumulative => "flow_cumulative",
        };
        write!(f, "{}", s)
    }
}

// ============================================================================
// Solver tuning (ablation switches)
// ============================================================================

/// What the full-kernel CP-SAT solve minimises.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum CpsatObjective {
    /// `scale * makespan + sum(effective completion cycle of every op)` (default). The sum
    /// is a tie-breaker that prefers schedules whose ops finish early.
    #[default]
    #[serde(rename = "makespan+tiebreak")]
    MakespanTiebreak,
    /// `makespan` alone: no tie-breaker term and none of its auxiliary "chosen cycle"
    /// variables. The solve can stop as soon as the makespan is proved.
    #[serde(rename = "makespan")]
    Makespan,
}

/// How the eight lane variables of a shiftable vector op's scalar alternative are modelled
/// (`alu_c{op}_{lane}`: the cycle of each scalar op when the vector op is shifted).
///
/// Both restrictions keep an optimal schedule of the free model: every feasible schedule
/// maps to one of the restricted model with the same cycle variables, mode literals,
/// completions and per-cycle slot usage (Z3: `verification/lane_model.py`; the shapes of the
/// rows that mention lanes are checked on exported models by `tests/test_cpsat_lane_model.py`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum CpsatLaneModel {
    /// Lanes are free variables in the op's domain (default): lane precedences are enforced
    /// by the producer's shift literal, and an unshifted op's lanes take any value its rows
    /// allow.
    #[default]
    #[serde(rename = "free")]
    Free,
    /// `lane_0 <= lane_1 <= ... <= lane_7` for every shiftable op: removes the 8! lane
    /// permutations of each op. Valid because every row that mentions lanes either uses the
    /// same lane index on both sides or bounds all lanes of an op alike.
    #[serde(rename = "sorted")]
    Sorted,
    /// `!shift => lane_i = cycle` for every lane, and each lane precedence is one
    /// unconditional row `lane_i(op) >= lane_i(dep) + 1` instead of a pair enforced by the
    /// producer's shift literal.
    #[serde(rename = "pinned")]
    Pinned,
    /// Both of the above.
    #[serde(rename = "sorted-pinned")]
    SortedPinned,
}

impl CpsatLaneModel {
    /// Whether the lanes of each op are ordered.
    pub fn sorted(self) -> bool {
        matches!(self, Self::Sorted | Self::SortedPinned)
    }

    /// Whether an unshifted op's lanes equal its cycle (and lane precedences are unconditional).
    pub fn pinned(self) -> bool {
        matches!(self, Self::Pinned | Self::SortedPinned)
    }
}

/// Extra makespan lower-bound rows (`CpsatTuning::makespan_rows`). See `makespan_rows.rs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MakespanRows {
    /// No extra rows (default).
    #[default]
    None,
    /// Head/tail energy rows from the static longest-path heads and tails: per engine, the
    /// work of the ops that cannot start before cycle t (or must be followed by q cycles)
    /// fits in [t, makespan - q).
    Static,
    /// `Static`, plus prefix/suffix variables per bucket of static heads that follow the
    /// solution: the ops of the first k buckets end by `prefix_end_k`, the ops of the last
    /// buckets start no earlier than `suffix_start_k`, and each set's work fits in its span.
    Bucketed,
}

/// Which symmetry-breaking ordering rows the model gets (`CpsatTuning::symmetry_rows`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SymmetryRows {
    /// The rows emission declares (`sym_ord_*` from `symmetry_hint.ordering_group`,
    /// `sym_lex_*` from `symmetry_hint.lexicographic_stream`), unchecked (default).
    #[default]
    Declared,
    /// No ordering rows.
    None,
    /// Only rows justified by a swap of structurally identical groups (`group_symmetry.rs`):
    /// one anchor op per group, groups of an orbit in a chain. Replaces the declared rows and
    /// the declared ordering in symmetry-aware tightening, and gives corresponding ops of an
    /// orbit the same domain.
    Checked,
}

/// Extra cumulative propagators CP-SAT can be asked to run (all off in CP-SAT by default).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CumulativePropagator {
    /// `use_overload_checker_in_cumulative`: energy-based overload checking.
    OverloadChecker,
    /// `use_timetable_edge_finding_in_cumulative`: energy plus mandatory parts. CP-SAT only
    /// runs it on cumulatives with at most `max_num_intervals_for_timetable_edge_finding`
    /// intervals (see `CpsatTuning::timetable_edge_finding_max_intervals`).
    TimetableEdgeFinding,
    /// `use_hard_precedences_in_cumulative` together with `exploit_all_precedences`:
    /// constraints for variables that are after a set of intervals of the same cumulative.
    /// Both parameters are set because OR-Tools 9.15 builds the precedence list only under
    /// `exploit_all_precedences` (`cumulative.cc`): the first parameter alone adds nothing.
    /// Experimental in CP-SAT; needs the precedence graph to be a DAG, which ours is.
    HardPrecedences,
}

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

/// Largest `num_workers` OR-Tools 9.15 accepts (`kMaxReasonableParallelism`).
pub const MAX_CPSAT_WORKERS: i32 = 10_000;

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
    /// Objective: makespan with the sum-of-cycles tie-breaker (default) or makespan alone.
    pub objective: CpsatObjective,
    /// Cap every op's ALAP at `greedy cycle + slack` (default `Some(15)`). The cap is a search
    /// restriction, not a valid bound: a schedule that runs some op more than `slack` cycles
    /// later than the greedy did is outside the model. `None` (JSON `null`) removes the cap;
    /// ALAP then comes from the horizon alone.
    pub greedy_alap_slack: Option<i32>,
    /// With a greedy schedule the horizon is also raised to the max-latency critical path
    /// (every shiftable op taking its worst-case latency) when this is true (default). That
    /// path is several times the makespan (183 against 43 at 2 rounds / batch 16). ALAP is
    /// computed with minimum latencies, so a feasible greedy schedule already fits below
    /// `greedy makespan + 20`; false drops the guard.
    pub horizon_at_least_max_latency_path: bool,
    /// With a greedy schedule the horizon is `greedy makespan + 20`, raised to the number of
    /// ops when this is true (default). The op count is far above any makespan (12,356 against
    /// about 2,000 at 16 rounds) and only widens the domains of auxiliary variables and of
    /// ALAP when the greedy cap is off. False drops it; the horizon then relies on the greedy
    /// schedule being feasible.
    pub horizon_at_least_num_ops: bool,
    /// Add an interval `[makespan, horizon)` with demand = capacity to each engine cumulative
    /// (the pattern of OR-Tools' `rcpsp_sat.py`). CP-SAT's linear relaxation then recognises
    /// the makespan of each cumulative (`DetectMakespan` in `linear_relaxation.cc`) and uses
    /// it in the energy relaxation and cuts. Adds no restriction: every task ends by `makespan`.
    pub makespan_interval: bool,
    /// `push_all_tasks_toward_start` ("Experimental code: specify if the objective pushes all
    /// tasks toward the start of the schedule"; used by `rcpsp_sat.py`).
    pub push_all_tasks_toward_start: bool,
    /// Extra cumulative propagators to switch on for every worker (default: none).
    pub cumulative_propagators: Vec<CumulativePropagator>,
    /// `max_num_intervals_for_timetable_edge_finding` (CP-SAT default 100; `None` leaves it).
    /// Timetable edge finding is skipped on larger cumulatives, and the engine cumulatives
    /// here have hundreds to tens of thousands of intervals, so the propagator does nothing
    /// unless this is raised.
    pub timetable_edge_finding_max_intervals: Option<i32>,
    /// Lane variables of shiftable ops: free (default), sorted, pinned or both.
    pub lane_model: CpsatLaneModel,
    /// `linearization_level` (default 2; CP-SAT's own default is 1). 0 = no LP, 1 = the
    /// linear constraints and some relaxations, 2 = also the relaxations and cuts of
    /// cumulatives, enforced constraints and others. Measured in docs/CPSAT_CONFIG_ABLATION.md.
    pub linearization_level: i32,
    /// `cp_model_probing_level` (default 2, CP-SAT's default): presolve probing effort, 0 to 3.
    pub probing_level: i32,
    /// Extra makespan lower-bound rows (default: none). See [`MakespanRows`].
    pub makespan_rows: MakespanRows,
    /// Symmetry-breaking ordering rows (default: the ones emission declares). See [`SymmetryRows`].
    pub symmetry_rows: SymmetryRows,
    /// Seed for CP-SAT (default: `None`, CP-SAT's own default).
    /// Reproducible parallel search requires deterministic interleaving and a solve that
    /// finishes before the wall-clock limit.
    pub random_seed: Option<i32>,
}

impl Default for CpsatTuning {
    fn default() -> Self {
        Self {
            workers: 8,
            interleave_search: true,
            subsolvers: Vec::new(),
            extra_subsolvers: Vec::new(),
            ignore_subsolvers: Vec::new(),
            objective: CpsatObjective::default(),
            greedy_alap_slack: Some(15),
            horizon_at_least_max_latency_path: true,
            horizon_at_least_num_ops: true,
            makespan_interval: false,
            push_all_tasks_toward_start: false,
            cumulative_propagators: Vec::new(),
            timetable_edge_finding_max_intervals: None,
            lane_model: CpsatLaneModel::default(),
            linearization_level: 2,
            probing_level: 2,
            makespan_rows: MakespanRows::None,
            symmetry_rows: SymmetryRows::Declared,
            random_seed: None,
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
        if let Some(slack) = self.greedy_alap_slack.filter(|s| *s < 0) {
            return Err(format!("cpsat tuning: greedy_alap_slack must be 0 or more (or null), got {}", slack));
        }
        if let Some(max) = self.timetable_edge_finding_max_intervals.filter(|m| *m < 0) {
            return Err(format!(
                "cpsat tuning: timetable_edge_finding_max_intervals must be 0 or more, got {}",
                max
            ));
        }
        if !(0..=2).contains(&self.linearization_level) {
            return Err(format!(
                "cpsat tuning: linearization_level must be 0, 1 or 2, got {}",
                self.linearization_level
            ));
        }
        if !(0..=3).contains(&self.probing_level) {
            return Err(format!("cpsat tuning: probing_level must be 0 to 3, got {}", self.probing_level));
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
        // CP-SAT rejects num_workers above 10,000 with MODEL_INVALID (parameters_validation.cc,
        // kMaxReasonableParallelism), which would surface as "no solution"; say so here instead.
        if self.workers > MAX_CPSAT_WORKERS {
            return Err(format!(
                "cpsat tuning: workers must be at most {}, got {}",
                MAX_CPSAT_WORKERS, self.workers
            ));
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
