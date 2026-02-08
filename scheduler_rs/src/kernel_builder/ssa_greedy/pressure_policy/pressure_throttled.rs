//! Pressure-throttled policy — soft re-sorting by net pressure.
//!
//! Above a soft threshold (default 85% of budget), re-sorts the ready queue
//! by net pressure (prefer ops that free scratch) instead of pure LUC.
//! Suppresses VALU→ALU shift when ALU is heavily loaded (>8/12 slots).

use super::PressurePolicy;

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
        ready_ops: &mut [(usize, usize, usize, usize)],
        live_words: usize,
        scratch_budget: usize,
    ) {
        let soft_threshold = (scratch_budget as f64 * self.soft_threshold_fraction) as usize;
        if live_words > soft_threshold {
            // Above soft threshold: sort by net pressure ascending (prefer ops
            // that free the most scratch), with LUC descending as tiebreaker.
            ready_ops.sort_by(|a, b| {
                let net_a = a.2 as i64 - a.3 as i64;
                let net_b = b.2 as i64 - b.3 as i64;
                net_a.cmp(&net_b).then_with(|| b.1.cmp(&a.1))
            });
        } else {
            ready_ops.sort_by(|a, b| b.1.cmp(&a.1));
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
