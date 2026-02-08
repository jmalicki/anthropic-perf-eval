//! Pressure-throttled policy — soft re-sorting by net pressure.
//!
//! Above a soft threshold (default 85% of budget), re-sorts the ready queue
//! by net pressure (prefer ops that free scratch) instead of pure LUC.
//! Suppresses VALU→ALU shift when ALU is heavily loaded (>8/12 slots).

use super::{PressurePolicy, SchedulingCandidate, SchedulingContext};

/// Soft throttling: re-sort by net pressure above soft threshold, suppress
/// shift when ALU >8/12 used.
#[derive(Debug, Clone, Copy)]
pub struct PressureThrottledPolicy {
    /// Fraction of scratch budget above which we re-sort by net pressure.
    pub soft_threshold_fraction: f64,
    /// Maximum ALU slots used before suppressing VALU→ALU shift.
    pub alu_shift_suppress_threshold: usize,
}

impl Default for PressureThrottledPolicy {
    fn default() -> Self {
        PressureThrottledPolicy {
            soft_threshold_fraction: 0.85,
            alu_shift_suppress_threshold: 8,
        }
    }
}

impl PressurePolicy for PressureThrottledPolicy {
    fn sort_ready(
        &self,
        candidates: &mut [SchedulingCandidate],
        ctx: &SchedulingContext,
    ) {
        let soft_threshold = (ctx.scratch_budget as f64 * self.soft_threshold_fraction) as usize;
        if ctx.live_words > soft_threshold {
            // Above soft threshold: sort by net pressure ascending (prefer ops
            // that free the most scratch), with LUC descending as tiebreaker.
            candidates.sort_by(|a, b| {
                let net_a = a.net_pressure();
                let net_b = b.net_pressure();
                net_a.cmp(&net_b).then_with(|| b.luc.cmp(&a.luc))
            });
        } else {
            candidates.sort_by(|a, b| b.luc.cmp(&a.luc));
        }
    }

    fn should_defer_for_pressure(
        &self,
        _live_words: usize,
        _words_added: usize,
        _words_freed: usize,
        _scratch_budget: usize,
    ) -> bool {
        false // Throttled only re-sorts, does not gate
    }

    fn allow_shift(&self, _live_words: usize, alu_used: usize, _scratch_budget: usize) -> bool {
        alu_used <= self.alu_shift_suppress_threshold
    }
}
