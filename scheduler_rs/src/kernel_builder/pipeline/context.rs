//! Pipeline context sub-types.
//!
//! Defines the structured data types that flow through the pipeline:
//! - `SchedulingBounds` — ASAP/ALAP bounds for CP-SAT domains
//! - `SchedulingDecisions` — choice/shift decisions from scheduling passes
//! - `SchedulingMetrics` — cycle count, peak liveness, budget status
//! - `PipelineConfig` — unified configuration for the pipeline
//! - `PassReport` — timing and diagnostics for each pass

use std::collections::{HashMap, HashSet};
use std::time::Duration;

use crate::types::{ConstChoiceEncoding, MachineConfig, SelectChoiceEncoding};

use super::super::config::CompilerConfig;

use crate::cpsat_scheduler::AsapAlapTightening;

#[cfg(feature = "cpsat")]
use crate::cpsat_scheduler::{
    CseGroup, CseGroupResult, ConstChoiceMode, OptionalConstraintGroup,
    SelectChoiceMode, SsaCpsatConfig, ValuShiftMode,
};

// ============================================================================
// Scheduling Bounds (CP-SAT domain hints)
// ============================================================================

/// ASAP/ALAP bounds computed from the dependency graph.
///
/// These are domain hints for the CP-SAT solver, not correctness guarantees.
/// Uses min_delay (best-case) for widest valid domains.
#[derive(Debug, Clone, Default)]
pub struct SchedulingBounds {
    /// ASAP (as-soon-as-possible) cycle for each op, keyed by op index.
    pub asap: HashMap<usize, i32>,
    /// ALAP (as-late-as-possible) cycle for each op, keyed by op index.
    pub alap: HashMap<usize, i32>,
}

// ============================================================================
// Scheduling Decisions
// ============================================================================

/// Decisions made by a scheduling pass (greedy or CP-SAT).
///
/// These track which ops were shifted, which encoding was chosen for
/// ConstChoice/SelectChoice, and other scheduling decisions.
#[derive(Debug, Clone, Default)]
pub struct SchedulingDecisions {
    /// VALU ops shifted to ALU (by op ID).
    pub shifted_ops: HashSet<usize>,
    /// SelectChoice encoding per op ID (exactly one variant per SelectChoice op).
    pub select_choice_encoding: HashMap<usize, SelectChoiceEncoding>,
    /// ConstChoice encoding per op ID (exactly one variant per ConstChoice op).
    pub const_choice_encoding: HashMap<usize, ConstChoiceEncoding>,
    /// Individual ALU cycle assignments for shifted VALU ops: op_id → [c0..c7].
    pub shifted_alu_cycles: HashMap<usize, Vec<i32>>,
    /// Op IDs of vbroadcasts eliminated (not emitted; consumers use scalar producer).
    pub eliminated_vbroadcast_ops: HashSet<usize>,
    /// Eliminated vbroadcast op_id → scalar producer op_id mapping.
    pub eliminated_vb_to_scalar_producer: HashMap<usize, usize>,
    /// SelectChoice sub cycles from greedy: op_id → cycle for the sub instruction.
    pub select_sub_cycles: HashMap<usize, i32>,
    /// SelectChoice FMA cycles from greedy: op_id → cycle for the FMA instruction.
    pub select_fma_cycles: HashMap<usize, i32>,
    /// CSE groups for shared SUB (from build_cse_groups). Empty when not used.
    #[cfg(feature = "cpsat")]
    pub cse_groups: Vec<CseGroup>,
    /// CSE group results from CP-SAT (sub_active, sub_cycle per group). Indexed by group.
    #[cfg(feature = "cpsat")]
    pub cse_group_results: Vec<CseGroupResult>,
}

// ============================================================================
// Scheduling Metrics
// ============================================================================

/// Metrics from a scheduling pass.
#[derive(Debug, Clone, Default)]
pub struct SchedulingMetrics {
    /// Number of cycles in the schedule.
    pub cycles: usize,
    /// Peak simultaneously-live words.
    pub peak_live: usize,
    /// Whether the scratch budget was exceeded.
    pub budget_exceeded: bool,
}

// ============================================================================
// Pipeline Configuration
// ============================================================================

/// Configuration for ASAP/ALAP bounds computation.
///
/// Always available (not feature-gated). Used by both greedy and CP-SAT
/// pipelines. The greedy pipeline uses bounds for ASAP inflation; the CP-SAT
/// pipeline uses them as domain hints for cycle variables.
#[derive(Debug, Clone)]
pub struct BoundsConfig {
    /// ASAP/ALAP tightening mode: none, simple, symmetry-aware, or energetic (default).
    pub asap_alap_tightening: AsapAlapTightening,
    /// Enable scratch-aware domain tightening.
    pub enable_scratch_tightening: bool,
    /// When true, symmetry ordering considers precedence in tie-breaking (experimental).
    pub symmetry_precedence_integrated: bool,
    /// When true, apply ALAP pulls in energetic loop (experimental; T7-two-step shows composition risk).
    pub energetic_alap_pulls: bool,
    /// Cap ASAP push / ALAP pull magnitude to at most k cycles per iteration (None = unbounded).
    pub energetic_bounded_push_k: Option<u32>,
    /// When true, use precedence-neutral horizon (2x critical path) instead of greedy-based.
    pub precedence_neutral_horizon: bool,
    /// When true (default), use capacity-aware deferral in greedy when Full bounds. Uses energetic logic.
    pub enable_capacity_deferral: bool,
    /// When true with greedy-only pipeline, use Full bounds (ASAP deferral, clamping, capacity deferral).
    /// Used for investigating feasibility: compare PolicyOnly vs Full greedy schedules.
    pub greedy_use_full_bounds: bool,
}

impl Default for BoundsConfig {
    fn default() -> Self {
        Self {
            asap_alap_tightening: AsapAlapTightening::SymmetryAware,
            enable_scratch_tightening: false,
            symmetry_precedence_integrated: false,
            energetic_alap_pulls: false,
            energetic_bounded_push_k: None,
            precedence_neutral_horizon: false,
            enable_capacity_deferral: true,
            greedy_use_full_bounds: false,
        }
    }
}

/// CP-SAT-specific pipeline configuration.
#[cfg(feature = "cpsat")]
#[derive(Debug, Clone)]
pub struct CpSatPipelineConfig {
    /// Time limit for the full-kernel CP-SAT solve.
    pub time_limit_secs: f64,
    /// Whether to print verbose solver output.
    pub verbose: bool,
    /// Enable vbroadcast elimination optimization.
    pub enable_vbroadcast_elimination: bool,
    /// Per-section optimization time (None or Some(0) = feasibility only).
    pub per_section_optimize_secs: Option<f64>,
    /// Forced shift set (ops that must be shifted to ALU).
    pub forced_shifts: Option<HashSet<usize>>,
    /// ConstChoice handling mode.
    pub const_choice_mode: ConstChoiceMode,
    /// SelectChoice handling mode.
    pub select_choice_mode: SelectChoiceMode,
    /// VALU shift handling mode.
    pub valu_shift_mode: ValuShiftMode,
    /// Force zero optimality gap (proven optimal).
    pub force_zero_gap: bool,
    /// ASAP/ALAP tightening mode: none, simple, symmetry-aware, or energetic (default).
    pub asap_alap_tightening: AsapAlapTightening,
    /// Enable scratch-aware domain tightening.
    pub enable_scratch_tightening: bool,
    /// Symmetry ordering: integrate precedence in tie-breaking (experimental).
    pub symmetry_precedence_integrated: bool,
    /// Energetic loop: enable ALAP pulls (experimental).
    pub energetic_alap_pulls: bool,
    /// Energetic loop: cap push/pull magnitude to k cycles (None = unbounded).
    pub energetic_bounded_push_k: Option<u32>,
    /// Use precedence-neutral horizon instead of greedy-based.
    pub precedence_neutral_horizon: bool,
    /// Optional constraint groups to disable (redundant/helper constraints). See SsaCpsatConfig.
    pub disable_optional_constraint_groups: HashSet<OptionalConstraintGroup>,
    /// When true (default), build CSE groups for shared SelectChoice SUB. When false, disable for debugging.
    pub enable_cse_groups: bool,
    /// When true, run bulk-move compaction after CP-SAT to eliminate empty cycles.
    /// Default false: CP-SAT should do it; bulk-move masks bugs and is buggy.
    /// Use --bulk-move only for experimenting, not production.
    pub enable_bulk_move: bool,
}

#[cfg(feature = "cpsat")]
impl Default for CpSatPipelineConfig {
    fn default() -> Self {
        CpSatPipelineConfig {
            time_limit_secs: 3600.0,
            verbose: true,
            enable_vbroadcast_elimination: false,
            per_section_optimize_secs: None,
            forced_shifts: None,
            const_choice_mode: ConstChoiceMode::Optimize,
            select_choice_mode: SelectChoiceMode::Optimize,
            valu_shift_mode: ValuShiftMode::Optimize,
            force_zero_gap: false,
            asap_alap_tightening: AsapAlapTightening::SymmetryAware,
            enable_scratch_tightening: false,
            symmetry_precedence_integrated: false,
            energetic_alap_pulls: false,
            energetic_bounded_push_k: None,
            precedence_neutral_horizon: false,
            disable_optional_constraint_groups: HashSet::new(),
            enable_cse_groups: true,
            enable_bulk_move: false, // off by default: buggy (same-cycle scratch conflicts)
        }
    }
}

#[cfg(feature = "cpsat")]
impl CpSatPipelineConfig {
    /// Build an `SsaCpsatConfig` for the full-kernel CP-SAT solve.
    pub fn to_full_kernel_config(&self, scratch_budget: usize) -> SsaCpsatConfig {
        SsaCpsatConfig {
            time_limit_secs: self.time_limit_secs,
            verbose: self.verbose,
            stop_after_first_solution: false,
            enable_valu_shift: false, // set by caller from pipeline config
            enable_vbroadcast_elimination: self.enable_vbroadcast_elimination,
            scratch_budget,
            model_name: Some("full_kernel_optimization".to_string()),
            machine: MachineConfig::default(),
            const_choice_mode: self.const_choice_mode,
            select_choice_mode: self.select_choice_mode,
            valu_shift_mode: self.valu_shift_mode,
            force_zero_gap: self.force_zero_gap,
            asap_alap_tightening: self.asap_alap_tightening,
            enable_scratch_tightening: self.enable_scratch_tightening,
            symmetry_precedence_integrated: self.symmetry_precedence_integrated,
            energetic_alap_pulls: self.energetic_alap_pulls,
            energetic_bounded_push_k: self.energetic_bounded_push_k,
            precedence_neutral_horizon: self.precedence_neutral_horizon,
            disable_optional_constraint_groups: self.disable_optional_constraint_groups.clone(),
            ..Default::default()
        }
    }

}

/// Unified pipeline configuration.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Compilation config (DCE, VALU shift, etc.).
    pub compiler: CompilerConfig,
    /// Machine slot limits.
    pub machine: MachineConfig,
    /// Hardware scratch budget in words.
    pub scratch_budget: usize,
    /// Whether to print verbose output.
    pub verbose: bool,
    /// Bounds computation configuration (used by both greedy and CP-SAT).
    pub bounds: BoundsConfig,
    /// CP-SAT specific configuration (None for greedy-only pipeline).
    #[cfg(feature = "cpsat")]
    pub cpsat: Option<CpSatPipelineConfig>,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        PipelineConfig {
            compiler: CompilerConfig::default(),
            machine: MachineConfig::default(),
            scratch_budget: 1536,
            verbose: true,
            bounds: BoundsConfig::default(),
            #[cfg(feature = "cpsat")]
            cpsat: None,
        }
    }
}

// ============================================================================
// Pass Report
// ============================================================================

/// Report from a single pass execution.
#[derive(Debug, Clone)]
pub struct PassReport {
    /// Pass name.
    pub name: &'static str,
    /// How long the pass took.
    pub duration: Duration,
    /// Optional detail string (e.g. "213 cycles, peak 1530 words").
    pub detail: String,
}
