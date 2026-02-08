//! Scheduling pressure policies for the greedy scheduler.
//!
//! Inspired by LLVM's `GenericScheduler` (llvm/lib/CodeGen/MachineScheduler.cpp),
//! which uses hierarchical register pressure heuristics:
//! - `RegExcess`: soft priority bias — prefer pressure-reducing ops when over limit
//! - `RegCritical`: soft priority bias — prefer pressure-reducing ops when
//!   approaching the critical threshold
//!
//! We adapt these concepts into four pressure policies for the scratch-budget–
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
//!
//! # Design: Sort-Only Pressure Awareness
//!
//! All policies use **sort-based** pressure awareness rather than binary deferral.
//! The `should_defer_for_pressure()` method exists on the trait for extensibility
//! but returns `false` for all built-in policies. This avoids **cascading deferral**
//! — a deadlock where all ready ops are deferred because they all have positive net
//! pressure (common on unrolled VLIW kernels with vbroadcasts and intermediates).
//!
//! See `docs/SCHEDULING_PRESSURE_CASCADING_DEFERRAL.md` for full design rationale.

mod greedy;
mod pressure_throttled;
mod pressure_gated;
mod adaptive;

pub use greedy::GreedyPolicy;
pub use pressure_throttled::PressureThrottledPolicy;
pub use pressure_gated::PressureGatedPolicy;
pub use adaptive::AdaptivePolicy;

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
    /// `ready_ops` contains `(op_index, luc, words_added, words_freed)` tuples.
    /// The implementation should sort in-place; ops are tried in order.
    fn sort_ready(
        &self,
        ready_ops: &mut [(usize, usize, usize, usize)],
        live_words: usize,
        scratch_budget: usize,
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
}
