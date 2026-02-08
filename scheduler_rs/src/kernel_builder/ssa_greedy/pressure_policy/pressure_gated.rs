//! Pressure-gated policy — graduated sort above critical threshold.
//!
//! Above a soft threshold (default 85% of budget), re-sorts the ready queue
//! with a graduated composite score that shifts from LUC-dominant to
//! pressure-dominant as pressure rises. Below the threshold, uses pure LUC sort.
//!
//! Unlike the original implementation which hard-gated (deferred) positive-pressure
//! ops above the critical threshold, this version uses sort-only pressure awareness
//! to avoid cascading deferral. See `docs/SCHEDULING_PRESSURE_CASCADING_DEFERRAL.md`.
//!
//! VALU shift is fully suppressed above critical threshold or when ALU is heavy.

use super::{PressurePolicy, SchedulingCandidate, SchedulingContext};

/// Graduated gating: above soft threshold, sort with increasing pressure weight.
/// Suppress shift above critical or when ALU >8/12.
#[derive(Debug, Clone, Copy)]
pub struct PressureGatedPolicy {
    /// Fraction of scratch budget above which we blend pressure into sort.
    pub soft_threshold_fraction: f64,
    /// Fraction of scratch budget above which we suppress VALU→ALU shift.
    pub critical_threshold_fraction: f64,
    /// Maximum ALU slots used before suppressing VALU→ALU shift.
    pub alu_shift_suppress_threshold: usize,
}

impl Default for PressureGatedPolicy {
    fn default() -> Self {
        PressureGatedPolicy {
            soft_threshold_fraction: 0.85,
            critical_threshold_fraction: 0.95,
            alu_shift_suppress_threshold: 8,
        }
    }
}

impl PressurePolicy for PressureGatedPolicy {
    fn sort_ready(
        &self,
        candidates: &mut [SchedulingCandidate],
        ctx: &SchedulingContext,
    ) {
        let pressure_fraction = ctx.live_words as f64 / ctx.scratch_budget as f64;

        if pressure_fraction <= self.soft_threshold_fraction {
            // Below soft threshold: pure LUC descending
            candidates.sort_by(|a, b| b.luc.cmp(&a.luc));
        } else {
            // Above soft threshold: graduated composite score
            let pressure_weight = ((pressure_fraction - self.soft_threshold_fraction)
                / (1.0 - self.soft_threshold_fraction))
                .clamp(0.0, 1.0);

            candidates.sort_by(|a, b| {
                let net_a = a.words_added as f64 - a.words_freed as f64;
                let net_b = b.words_added as f64 - b.words_freed as f64;
                let score_a = pressure_weight * net_a - (1.0 - pressure_weight) * a.luc as f64;
                let score_b = pressure_weight * net_b - (1.0 - pressure_weight) * b.luc as f64;
                score_a
                    .partial_cmp(&score_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
    }

    fn should_defer_for_pressure(
        &self,
        _live_words: usize,
        _words_added: usize,
        _words_freed: usize,
        _scratch_budget: usize,
    ) -> bool {
        // Never defer — pressure awareness is handled entirely by sort_ready().
        // See docs/SCHEDULING_PRESSURE_CASCADING_DEFERRAL.md for why binary
        // gating causes cascading deferral on unrolled kernels.
        false
    }

    fn allow_shift(&self, live_words: usize, alu_used: usize, scratch_budget: usize) -> bool {
        let critical_threshold =
            (scratch_budget as f64 * self.critical_threshold_fraction) as usize;
        live_words <= critical_threshold && alu_used <= self.alu_shift_suppress_threshold
    }
}
