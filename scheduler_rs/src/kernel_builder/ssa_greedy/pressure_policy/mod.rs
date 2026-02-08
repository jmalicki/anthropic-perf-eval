//! Scheduling pressure policies for the greedy scheduler.
//!
//! Inspired by LLVM's `GenericScheduler` (llvm/lib/CodeGen/MachineScheduler.cpp),
//! which uses hierarchical register pressure heuristics:
//! - `RegExcess`: soft priority bias — prefer pressure-reducing ops when over limit
//! - `RegCritical`: soft priority bias — prefer pressure-reducing ops when
//!   approaching the critical threshold
//!
//! We adapt these concepts into five pressure policies for the scratch-budget–
//! constrained greedy scheduler:
//!
//! # Policies
//!
//! - **Greedy**: Schedule by ASAP+LUC, defer only at hard budget, shift
//!   VALU→ALU whenever VALU is full. No pressure awareness.
//! - **PressureThrottled**: Re-sort ready ops by net pressure when live words
//!   exceed a soft threshold (85%). Suppress VALU shift when ALU is heavy.
//! - **PressureGated**: Above soft threshold, graduated composite sort blending
//!   LUC and net pressure (weight increases with pressure). Suppress shifts
//!   above critical threshold or when ALU is heavy.
//! - **Adaptive** (default): Same graduated composite sort as PressureGated,
//!   with additional shift suppression zones (greedy/throttled/gated).
//! - **Cascade**: LLVM-style strict 5-level priority cascade (engine saturation
//!   → slack → LUC → net pressure → ASAP tiebreak). First distinguishing level
//!   wins — no blending. Opt-in via `--scheduling-pressure-policy cascade`.
//!
//! # Design: Sort + ASAP Inflation
//!
//! All policies use **sort-based** pressure awareness rather than binary deferral.
//! The `should_defer_for_pressure()` method exists on the trait for extensibility
//! but returns `false` for all built-in policies. This avoids **cascading deferral**
//! — a deadlock where all ready ops are deferred because they all have positive net
//! pressure (common on unrolled VLIW kernels with vbroadcasts and intermediates).
//!
//! **ASAP inflation** complements sort-based awareness: when pressure exceeds 80%,
//! high-slack producers have their effective ASAP inflated, deferring them to later
//! cycles. This is NOT binary deferral — it targets only ops with sufficient slack
//! and is limited by `MIN_PASS_THROUGH` (at least 3 ops always pass through).
//! See `compute_asap_inflation()` on the trait.
//!
//! See `docs/SCHEDULING_PRESSURE_CASCADING_DEFERRAL.md` for full design rationale.

mod greedy;
mod pressure_throttled;
mod pressure_gated;
mod adaptive;
mod cascade;

pub use greedy::GreedyPolicy;
pub use pressure_throttled::PressureThrottledPolicy;
pub use pressure_gated::PressureGatedPolicy;
pub use adaptive::AdaptivePolicy;
pub use cascade::CascadePolicy;

use crate::types::Engine;

/// Rich scheduling candidate replacing raw (idx, luc, words_added, words_freed) tuples.
///
/// Exposes ASAP/ALAP bounds, engine type, shiftable status, and consumer count
/// so policies can make fully-informed scheduling decisions.
#[derive(Debug, Clone)]
pub struct SchedulingCandidate {
    /// Index into the ops array
    pub op_index: usize,
    /// Last-Use-Count: how many source VRegs have this op as their last reader
    pub luc: usize,
    /// Words this op's destination will add to scratch
    pub words_added: usize,
    /// Words freed by consuming last-use VRegs
    pub words_freed: usize,
    /// ASAP bound (earliest feasible cycle), if bounds are available
    pub asap: Option<i32>,
    /// ALAP bound (latest feasible cycle), if bounds are available
    pub alap: Option<i32>,
    /// Which engine this op executes on
    pub engine: Engine,
    /// Whether this VALU op can be shifted to ALU
    pub is_shiftable: bool,
    /// Number of unscheduled consumers of this op's output
    pub unscheduled_consumer_count: usize,
}

impl SchedulingCandidate {
    /// Scheduling slack (ALAP - ASAP). None if bounds unavailable.
    pub fn slack(&self) -> Option<i32> {
        match (self.asap, self.alap) {
            (Some(a), Some(b)) => Some(b - a),
            _ => None,
        }
    }

    /// Net scratch pressure: words_added - words_freed. Positive = increases pressure.
    pub fn net_pressure(&self) -> i64 {
        self.words_added as i64 - self.words_freed as i64
    }
}

/// Read-only snapshot of current slot usage, for policies that need engine awareness.
#[derive(Debug, Clone, Copy)]
pub struct SlotTrackerView {
    pub alu_used: usize,
    pub valu_used: usize,
    pub load_used: usize,
    pub store_used: usize,
    pub flow_used: usize,
}

/// Context for the current scheduling cycle.
#[derive(Debug)]
pub struct SchedulingContext<'a> {
    pub cycle: usize,
    pub live_words: usize,
    pub scratch_budget: usize,
    pub slots: &'a SlotTrackerView,
}

/// Pressure policy trait for the greedy scheduler.
///
/// Controls three scheduling decisions at each cycle:
/// 1. How to sort the ready queue (priority heuristic)
/// 2. Whether to defer an op based on register pressure (deprecated — returns
///    `false` for all built-in policies to avoid cascading deferral)
/// 3. Whether to allow a VALU→ALU shift
///
/// Modeled after LLVM's `GenericScheduler` hierarchical heuristics
/// (see `RegExcess` and `RegCritical` in `MachineScheduler.cpp`).
pub trait PressurePolicy: std::fmt::Debug {
    /// Sort the ready-op list for this cycle.
    ///
    /// Called once per cycle after the ready queue is drained.
    /// `candidates` contains rich scheduling info (LUC, ASAP/ALAP, engine, etc.).
    /// The implementation should sort in-place; ops are tried in order.
    fn sort_ready(
        &self,
        candidates: &mut [SchedulingCandidate],
        ctx: &SchedulingContext,
    );

    /// Whether to defer this op due to pressure.
    ///
    /// Called after the hard budget check passes. Returns `true` to defer
    /// the op to the next cycle (it will re-enter the ready queue).
    ///
    /// **Note:** All built-in policies return `false`. Binary deferral causes
    /// cascading deferral on unrolled kernels where all ready ops have positive
    /// net pressure. Pressure awareness should be handled by `sort_ready()`.
    /// This method is retained for extensibility only.
    fn should_defer_for_pressure(
        &self,
        live_words: usize,
        words_added: usize,
        words_freed: usize,
        scratch_budget: usize,
    ) -> bool;

    /// Whether to allow a VALU→ALU shift for this op.
    ///
    /// Called when VALU is full (6/6) and the op is shiftable.
    /// Returns `false` to suppress the shift (the op will be deferred to
    /// the next cycle instead of consuming 8 ALU slots).
    fn allow_shift(&self, live_words: usize, alu_used: usize, scratch_budget: usize) -> bool;

    /// Compute ASAP inflation for a candidate when pressure is high.
    ///
    /// Called during the ASAP-check loop for each ready op. Returns `Some(n)`
    /// to inflate the op's ASAP by `n` cycles (delaying it), or `None` to
    /// schedule normally. The caller clamps the inflated ASAP to the op's
    /// ALAP to prevent infeasibility.
    ///
    /// # Design
    ///
    /// ASAP inflation defers high-slack producers when scratch pressure is
    /// high, creating empty cycles that allow consumers to run and free
    /// registers. This is a greedy-only heuristic — CP-SAT uses original
    /// ASAP/ALAP bounds.
    ///
    /// See: Touati & Eisenbeis (2004), "Register Saturation in ILP" —
    /// when DAG parallelism exceeds register capacity, some ops must be
    /// delayed; inflation implements this proactively rather than reactively.
    fn compute_asap_inflation(
        &self,
        _candidate: &SchedulingCandidate,
        _ctx: &SchedulingContext,
    ) -> Option<i32> {
        None // Default: no inflation (backward-compatible)
    }
}
