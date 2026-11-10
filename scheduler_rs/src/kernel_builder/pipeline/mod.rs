//! Pipeline abstraction for the SSA scheduling pipeline.
//!
//! This module formalizes the scheduling pipeline as a sequence of passes
//! operating on shared `PipelineContext`. Each pass implements the `Pass` trait,
//! reads from the context, performs its work, and writes results back.
//!
//! ## Pipeline Variants
//!
//! - **Greedy pipeline**: Dependencies → DCE → Greedy → Barriers → Filter → Addresses → Lower
//! - **CP-SAT pipeline**: Dependencies → DCE → Bounds → Greedy (hints) → Per-section feasibility
//!   → Barriers → CP-SAT → Filter → Addresses → Lower
//!
//! ## Design
//!
//! The `PipelineContext` struct consolidates the 20+ local variables that previously
//! flowed through `build_and_schedule_with_options` and `build_and_schedule_cpsat`.
//! Each pass is a unit struct that delegates to the existing function implementations.

pub mod context;
pub mod passes;

use std::collections::HashMap;
use std::time::Instant;

use crate::address_assign::{build_vreg_metadata, AddressAssignmentResult};
use crate::schedule_constraints::ScheduleConstraint;
use crate::types::{Op, SsaOp, UsageAccounting, VReg, VarType};

use super::emit_ssa::ConstAlternative;
use super::ssa_program::SsaProgram;
use super::tick_tock_ssa::SsaScheduleResult;

use passes::{
    AssignAddressesPass, ComputeBoundsPass, ComputeDependenciesPass, DcePass, ElideBarriersPass,
    FilterDeadStoresPass, GreedySchedulePass, LowerToOpsPass,
};
use passes::{
    ConstChoiceHintPass, ScheduleWithFixedChoicesPass, SelectChoiceHintPass, ValuShiftHintPass,
};
#[cfg(feature = "cpsat")]
use passes::{BulkMoveCompactionPass, CpSatSchedulePass};

pub use context::{
    BoundsConfig, PassReport, PipelineConfig, SchedulingBounds, SchedulingDecisions,
    SchedulingMetrics,
};
#[cfg(feature = "cpsat")]
pub use context::CpSatPipelineConfig;

// ============================================================================
// Pass Trait
// ============================================================================

/// A single optimization or transformation pass in the pipeline.
///
/// Each pass reads from and writes to `PipelineContext`. The `run` method
/// returns a short summary string on success (e.g. "213 cycles") or an
/// error message on failure.
pub trait Pass {
    /// Human-readable name for this pass (used in reports and logging).
    fn name(&self) -> &'static str;

    /// Execute this pass, reading from and writing to the pipeline context.
    ///
    /// Returns a short summary string on success, or an error string on failure.
    fn run(&self, ctx: &mut PipelineContext) -> Result<String, String>;
}

// ============================================================================
// CP-SAT Extra Results
// ============================================================================

/// Additional CP-SAT-specific results that aren't part of the core pipeline.
///
/// These are only populated when the CP-SAT path is used and are forwarded
/// into `SsaScheduleResult` for bulk-move simulation and diagnostics.
#[derive(Debug, Clone, Default)]
pub struct CpSatExtra {
    /// Empty cycle count in the schedule.
    pub empty_cycle_count: usize,
    /// Whether makespan was driven by capacity lower bound.
    pub makespan_driven_by_capacity: bool,
    /// Schedule constraints for bulk-move simulator.
    pub schedule_constraints: Vec<ScheduleConstraint>,
    /// SSA-level cycle assignments (same index space as schedule_constraints).
    pub ssa_cycle_assignments: Vec<i32>,
    /// SSA-level (id, dep_ids) per op for bulk-move.
    pub ssa_ops_bulk_move: Vec<(usize, Vec<usize>)>,
    /// Proven lower bound on cycles.
    pub lb_cycles: Option<usize>,
}

// ============================================================================
// Pipeline Context
// ============================================================================

/// Shared mutable state flowing through the scheduling pipeline.
///
/// Consolidates the 20+ local variables from the old orchestration functions
/// into a single struct that each pass reads from and writes to.
pub struct PipelineContext {
    // --- Program IR ---
    /// SSA operations (modified in-place by passes).
    pub ops: Vec<SsaOp>,
    /// VReg → defining op index.
    pub vreg_def: HashMap<VReg, usize>,
    /// Total VRegs allocated during emission.
    pub vreg_count: u32,
    /// Constant encoding alternatives (Load vs AddImm vs AluDerive).
    pub const_alternatives: Vec<ConstAlternative>,
    /// VReg representing constant 0 (address 0), if reserved.
    pub zero_vreg: Option<VReg>,
    /// Maps vector group alias VRegs to their 8 scalar lane VRegs (for scatter loads).
    pub vector_group_lanes: HashMap<VReg, Vec<VReg>>,

    // --- Scheduling state ---
    /// ASAP/ALAP bounds (populated by ComputeBoundsPass, consumed by greedy + CP-SAT).
    pub bounds: Option<SchedulingBounds>,
    /// Decisions from the greedy scheduling pass (used as CP-SAT hints).
    pub greedy_decisions: SchedulingDecisions,
    /// Final scheduling decisions (greedy or CP-SAT).
    pub final_decisions: SchedulingDecisions,
    /// Metrics from the greedy scheduling pass.
    pub greedy_metrics: SchedulingMetrics,
    /// Metrics from the final scheduling pass.
    pub final_metrics: SchedulingMetrics,

    // --- Post-scheduling state ---
    /// Address assignment result (populated by AssignAddressesPass).
    pub address_result: Option<AddressAssignmentResult>,
    /// Lowered concrete ops (populated by LowerToOpsPass).
    pub lowered_ops: Option<Vec<Op>>,
    /// Usage accounting (computed after address assignment, before lowering).
    pub usage_accounting: UsageAccounting,
    /// VReg metadata for liveness debugging (computed after address assignment, before lowering).
    pub vreg_metadata:
        HashMap<VReg, (Option<VarType>, Option<String>, Option<usize>)>,
    /// CP-SAT-specific extra results (None for greedy path).
    pub cpsat_extra: Option<CpSatExtra>,

    // --- Configuration ---
    /// Pipeline configuration.
    pub config: PipelineConfig,

    // --- Diagnostics ---
    /// Reports from each pass (timing, summaries).
    pub reports: Vec<PassReport>,
}

impl PipelineContext {
    /// Create a new pipeline context from an SsaProgram and configuration.
    pub fn new(program: SsaProgram, config: PipelineConfig) -> Self {
        let SsaProgram {
            ops,
            vreg_def,
            vreg_count,
            const_alternatives,
            zero_vreg,
            vector_group_lanes,
        } = program;

        PipelineContext {
            ops,
            vreg_def,
            vreg_count,
            const_alternatives,
            zero_vreg,
            vector_group_lanes,
            bounds: None,
            greedy_decisions: SchedulingDecisions::default(),
            final_decisions: SchedulingDecisions::default(),
            greedy_metrics: SchedulingMetrics::default(),
            final_metrics: SchedulingMetrics::default(),
            address_result: None,
            lowered_ops: None,
            usage_accounting: UsageAccounting::default(),
            vreg_metadata: HashMap::new(),
            cpsat_extra: None,
            config,
            reports: Vec::new(),
        }
    }

    /// Run a pass with timing and report recording.
    pub fn run_pass(&mut self, pass: &impl Pass) -> Result<(), String> {
        let name = pass.name();
        if self.config.verbose {
            eprintln!("[pipeline] Running pass: {}", name);
        }
        let start = Instant::now();
        let detail = pass.run(self)?;
        let duration = start.elapsed();
        if self.config.verbose {
            eprintln!(
                "[pipeline] Pass {} completed in {:.1}ms: {}",
                name,
                duration.as_secs_f64() * 1000.0,
                detail
            );
        }
        self.reports.push(PassReport {
            name,
            duration,
            detail,
        });
        Ok(())
    }

    /// Effective scratch budget accounting for zero_vreg reservation.
    pub fn effective_scratch_budget(&self) -> usize {
        super::tick_tock_ssa::effective_scratch_budget(self.zero_vreg.is_some())
    }

    // ========================================================================
    // Pipeline orchestration
    // ========================================================================

    /// Run the greedy-only scheduling pipeline.
    ///
    /// Pipeline: Dependencies → [DCE → Dependencies] → Bounds → Greedy(PolicyOnly) → Barriers → Filter → Addresses → Lower
    ///
    /// Bounds are computed so ASAP inflation can work in greedy-only mode.
    /// The greedy scheduler receives `GreedyBoundsUsage::PolicyOnly`, which enables
    /// inflation (policy decisions) without activating ASAP deferral or cycle
    /// clamping (core loop mechanics).
    pub fn run_greedy_pipeline(&mut self, enable_dce: bool) -> Result<(), String> {
        self.run_pass(&ComputeDependenciesPass)?;
        if enable_dce {
            self.run_pass(&DcePass)?;
            self.run_pass(&ComputeDependenciesPass)?;
        }
        self.run_pass(&ComputeBoundsPass)?;
        self.run_pass(&GreedySchedulePass)?;

        // Hint passes (overwrite greedy_decisions only; run when enabled)
        if self.config.compiler.enable_const_choice_hint_pass {
            self.run_pass(&ConstChoiceHintPass)?;
        }
        if self.config.compiler.enable_select_choice_hint_pass {
            self.run_pass(&SelectChoiceHintPass)?;
        }
        if self.config.compiler.enable_valu_shift_hint_pass {
            self.run_pass(&ValuShiftHintPass)?;
        }
        // When any hint pass ran, re-schedule with fixed encodings so op.cycle matches greedy_decisions
        // (avoids inconsistent hints when CP-SAT or greedy-only uses cycle + encoding).
        let any_hint_pass = self.config.compiler.enable_const_choice_hint_pass
            || self.config.compiler.enable_select_choice_hint_pass
            || self.config.compiler.enable_valu_shift_hint_pass;
        if any_hint_pass || self.config.compiler.enable_schedule_with_fixed_choices {
            self.run_pass(&ScheduleWithFixedChoicesPass)?;
        }

        // In greedy-only mode, final decisions = greedy decisions
        self.final_decisions = self.greedy_decisions.clone();
        self.final_metrics = self.greedy_metrics.clone();

        self.run_pass(&ElideBarriersPass)?;
        self.run_pass(&FilterDeadStoresPass)?;

        // Compute accounting before address assignment (needs SSA ops with metadata)
        self.compute_accounting();

        self.run_pass(&AssignAddressesPass)?;
        self.run_pass(&LowerToOpsPass)?;
        Ok(())
    }

    /// Run the CP-SAT scheduling pipeline.
    ///
    /// Pipeline: Dependencies → [DCE → Dependencies] → Bounds → Greedy (hints)
    ///   → [Per-section feasibility] → Barriers → CP-SAT → Filter → Addresses → Lower
    #[cfg(feature = "cpsat")]
    pub fn run_cpsat_pipeline(&mut self, enable_dce: bool) -> Result<(), String> {
        self.run_pass(&ComputeDependenciesPass)?;
        if enable_dce {
            self.run_pass(&DcePass)?;
            self.run_pass(&ComputeDependenciesPass)?;
        }
        self.run_pass(&ComputeBoundsPass)?;
        self.run_pass(&GreedySchedulePass)?;

        // Hint passes (overwrite greedy_decisions; CP-SAT will use as hints)
        if self.config.compiler.enable_const_choice_hint_pass {
            self.run_pass(&ConstChoiceHintPass)?;
        }
        if self.config.compiler.enable_select_choice_hint_pass {
            self.run_pass(&SelectChoiceHintPass)?;
        }
        if self.config.compiler.enable_valu_shift_hint_pass {
            self.run_pass(&ValuShiftHintPass)?;
        }
        // When any hint pass ran, re-schedule with fixed encodings so cycle hints match greedy_decisions
        // (avoids "Merged hint violates engine capacity" in CP-SAT verification).
        let any_hint_pass = self.config.compiler.enable_const_choice_hint_pass
            || self.config.compiler.enable_select_choice_hint_pass
            || self.config.compiler.enable_valu_shift_hint_pass;
        if any_hint_pass || self.config.compiler.enable_schedule_with_fixed_choices {
            self.run_pass(&ScheduleWithFixedChoicesPass)?;
        }

        // Per-section feasibility stays in tick_tock_ssa orchestration (not a pass)
        // — it involves conditional looping with CP-SAT sub-calls per section.

        self.run_pass(&ElideBarriersPass)?;
        self.run_pass(&CpSatSchedulePass)?;
        self.run_pass(&BulkMoveCompactionPass)?;
        self.run_pass(&FilterDeadStoresPass)?;

        // Compute accounting before address assignment (needs SSA ops with metadata)
        self.compute_accounting();

        self.run_pass(&AssignAddressesPass)?;
        self.run_pass(&LowerToOpsPass)?;
        Ok(())
    }

    // ========================================================================
    // Accounting
    // ========================================================================

    /// Compute usage accounting and VReg metadata from the current SSA ops.
    ///
    /// Must be called after filtering dead stores and before address assignment,
    /// while SSA ops still have their metadata.
    pub fn compute_accounting(&mut self) {
        self.usage_accounting = UsageAccounting::from_ssa_ops(&self.ops);
        self.vreg_metadata = build_vreg_metadata(&self.ops);
    }

    // ========================================================================
    // Result conversion
    // ========================================================================

    /// Convert the pipeline result into an `SsaScheduleResult`.
    ///
    /// This is called after the pipeline completes to produce the return value
    /// expected by the existing callers.
    pub fn into_result(self) -> Result<SsaScheduleResult, String> {
        let address_result = self
            .address_result
            .ok_or_else(|| "Pipeline incomplete: no address assignment result".to_string())?;
        let ops = self
            .lowered_ops
            .ok_or_else(|| "Pipeline incomplete: no lowered ops".to_string())?;

        let budget_exceeded = address_result.budget_exceeded;

        // Update usage accounting with final peak_live
        let mut usage_accounting = self.usage_accounting;
        usage_accounting.set_peak_live(address_result.peak_live);

        // Cycle count from lowered ops (SelectChoice FMA emits sub+FMA across 2 cycles).
        let base_cycles = self.final_metrics.cycles;
        let cycles = ops
            .iter()
            .map(|op| op.cycle.max(0) as usize)
            .max()
            .map(|m| m + 1)
            .unwrap_or(base_cycles)
            .max(base_cycles);

        Ok(SsaScheduleResult {
            ops,
            cycles,
            address_result,
            vreg_count: self.vreg_count,
            valu_shifted: self.final_decisions.shifted_ops.len(),
            budget_exceeded,
            usage_accounting,
            vreg_metadata: self.vreg_metadata,
            shifted_alu_cycles: self.final_decisions.shifted_alu_cycles,
            empty_cycle_count: self
                .cpsat_extra
                .as_ref()
                .map(|e| e.empty_cycle_count),
            makespan_driven_by_capacity: self
                .cpsat_extra
                .as_ref()
                .map(|e| e.makespan_driven_by_capacity),
            schedule_constraints: self
                .cpsat_extra
                .as_ref()
                .map(|e| e.schedule_constraints.clone()),
            ssa_cycle_assignments: self
                .cpsat_extra
                .as_ref()
                .map(|e| e.ssa_cycle_assignments.clone()),
            ssa_ops_bulk_move: self
                .cpsat_extra
                .as_ref()
                .map(|e| e.ssa_ops_bulk_move.clone()),
            ssa_ops: Some(self.ops.clone()), // Include SSA ops for debugging/inspection
            eliminated_vbroadcast_ops: self
                .final_decisions
                .eliminated_vbroadcast_ops
                .into_iter()
                .collect(),
            lb_cycles: self.cpsat_extra.as_ref().and_then(|e| e.lb_cycles),
        })
    }
}
