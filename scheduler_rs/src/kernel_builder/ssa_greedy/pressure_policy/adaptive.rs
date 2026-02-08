//! Adaptive pressure policy — graduated composite score based on pressure.
//!
//! Automatically adjusts scheduling priority based on current pressure:
//! - Below soft threshold (85%): Greedy — pure LUC sort, always shift
//! - Above soft threshold: graduated blend of LUC and net pressure, with
//!   pressure weight increasing linearly from 0.0 to 1.0 as pressure rises
//!   from the soft threshold to the budget limit
//!
//! Unlike the original implementation which used a binary gate above the
//! critical threshold (causing cascading deferral on unrolled kernels),
//! this version uses sort-only pressure awareness inspired by LLVM's
//! `GenericScheduler` — the scheduler always schedules something, it just
//! changes what it prefers.
//!
//! See `docs/SCHEDULING_PRESSURE_CASCADING_DEFERRAL.md` for design rationale.

use super::{PressurePolicy, SchedulingCandidate, SchedulingContext};

/// Adaptive policy: graduated composite score shifts from LUC-dominant to
/// pressure-dominant as scratch pressure rises.
#[derive(Debug, Clone, Copy)]
pub struct AdaptivePolicy {
    /// Fraction of scratch budget above which we blend pressure into sort.
    pub soft_threshold_fraction: f64,
    /// Fraction of scratch budget above which we suppress VALU→ALU shifts.
    /// Retained for shift suppression (shift decisions are separate from sort).
    pub critical_threshold_fraction: f64,
    /// Maximum ALU slots used before suppressing VALU→ALU shift (throttled+).
    pub alu_shift_suppress_threshold: usize,
}

impl Default for AdaptivePolicy {
    fn default() -> Self {
        AdaptivePolicy {
            soft_threshold_fraction: 0.85,
            critical_threshold_fraction: 0.95,
            alu_shift_suppress_threshold: 8,
        }
    }
}

impl PressurePolicy for AdaptivePolicy {
    fn sort_ready(
        &self,
        candidates: &mut [SchedulingCandidate],
        ctx: &SchedulingContext,
    ) {
        let pressure_fraction = ctx.live_words as f64 / ctx.scratch_budget as f64;

        if pressure_fraction <= self.soft_threshold_fraction {
            // Greedy zone: pure LUC descending
            candidates.sort_by(|a, b| b.luc.cmp(&a.luc));
        } else {
            // Pressure-aware zone: graduated composite score
            // Weight shifts from LUC-dominant (0.0) to pressure-dominant (1.0)
            // as pressure rises from soft_threshold to 100%.
            let pressure_weight = ((pressure_fraction - self.soft_threshold_fraction)
                / (1.0 - self.soft_threshold_fraction))
                .clamp(0.0, 1.0);

            candidates.sort_by(|a, b| {
                let net_a = a.words_added as f64 - a.words_freed as f64;
                let net_b = b.words_added as f64 - b.words_freed as f64;
                // Composite: lower score = better
                // pressure_weight=0 → pure LUC (higher LUC = lower score = better)
                // pressure_weight=1 → pure net pressure (lower net = better)
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
        if live_words > critical_threshold {
            // Above critical: suppress all shifts
            false
        } else {
            let soft_threshold =
                (scratch_budget as f64 * self.soft_threshold_fraction) as usize;
            if live_words > soft_threshold {
                // Throttled zone: suppress when ALU is heavy
                alu_used <= self.alu_shift_suppress_threshold
            } else {
                // Greedy zone: always allow
                true
            }
        }
    }
}
