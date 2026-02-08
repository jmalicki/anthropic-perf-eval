//! Cascade pressure policy — LLVM-style strict priority cascade.
//!
//! Uses a 5-level strict priority system where the first distinguishing
//! criterion wins. No blending or composite scores — this makes scheduling
//! decisions more predictable and debuggable than the graduated composite
//! used by AdaptivePolicy.
//!
//! # Priority Levels (first distinguishing level wins)
//!
//! 1. **Engine saturation**: Ops on nearly-full engines go first (avoids spillover)
//! 2. **Slack**: Critical-path ops (low ALAP-ASAP) go first
//! 3. **LUC**: Ops that free more scratch go first
//! 4. **Net pressure**: Only above soft threshold — prefer ops that reduce pressure
//! 5. **ASAP tiebreak**: Lower ASAP, then lower op_index (deterministic)

use super::{PressurePolicy, SchedulingCandidate, SchedulingContext};

/// LLVM-style strict priority cascade with 5 levels.
#[derive(Debug, Clone, Copy)]
pub struct CascadePolicy {
    /// Fraction of scratch budget above which net pressure becomes a tiebreaker.
    pub soft_threshold_fraction: f64,
    /// Fraction of scratch budget above which we suppress VALU→ALU shifts.
    pub critical_threshold_fraction: f64,
    /// Maximum ALU slots used before suppressing VALU→ALU shift.
    pub alu_shift_suppress_threshold: usize,
    /// Fraction of scratch budget above which ASAP inflation activates.
    pub inflation_threshold_fraction: f64,
    /// Minimum slack (ALAP - ASAP) required before an op can be inflated.
    /// Ops on the critical path (slack 0-2) are never inflated.
    pub min_inflation_slack: i32,
    /// Maximum number of cycles to inflate ASAP by.
    pub max_inflation_cycles: i32,
}

impl Default for CascadePolicy {
    fn default() -> Self {
        CascadePolicy {
            soft_threshold_fraction: 0.85,
            critical_threshold_fraction: 0.95,
            alu_shift_suppress_threshold: 8,
            inflation_threshold_fraction: 0.80,
            min_inflation_slack: 3,
            max_inflation_cycles: 10,
        }
    }
}

impl CascadePolicy {
    /// Compute engine saturation score based on how full the op's engine is.
    ///
    /// Higher score = engine more saturated = schedule this op sooner to avoid
    /// it being pushed to a later cycle where the engine might be completely full.
    fn engine_saturation_score(candidate: &SchedulingCandidate, ctx: &SchedulingContext) -> u8 {
        match candidate.engine {
            crate::types::Engine::Valu => {
                if ctx.slots.valu_used >= 5 { 3 }  // 5/6 = critical
                else if ctx.slots.valu_used >= 4 { 2 }
                else { 0 }
            }
            crate::types::Engine::Load => {
                if ctx.slots.load_used >= 1 { 3 } else { 0 }  // 1/2 = half full
            }
            crate::types::Engine::Store => {
                if ctx.slots.store_used >= 1 { 3 } else { 0 }
            }
            crate::types::Engine::Flow => {
                if ctx.slots.flow_used >= 1 { 3 } else { 0 }  // 1/1 = full (shouldn't happen if fits)
            }
            crate::types::Engine::Alu => {
                if ctx.slots.alu_used >= 10 { 2 }  // 10/12
                else { 0 }
            }
            _ => 0,
        }
    }
}

impl PressurePolicy for CascadePolicy {
    fn sort_ready(
        &self,
        candidates: &mut [SchedulingCandidate],
        ctx: &SchedulingContext,
    ) {
        let pressure_above_soft = {
            let soft_threshold = (ctx.scratch_budget as f64 * self.soft_threshold_fraction) as usize;
            ctx.live_words > soft_threshold
        };

        candidates.sort_by(|a, b| {
            // Level 1: Engine saturation (higher = first)
            let sat_a = Self::engine_saturation_score(a, ctx);
            let sat_b = Self::engine_saturation_score(b, ctx);
            let cmp = sat_b.cmp(&sat_a);
            if cmp != std::cmp::Ordering::Equal { return cmp; }

            // Level 2: Slack (lower = first, critical path wins)
            // Ops without bounds get max slack (deferred)
            let slack_a = a.slack().unwrap_or(i32::MAX);
            let slack_b = b.slack().unwrap_or(i32::MAX);
            let cmp = slack_a.cmp(&slack_b);
            if cmp != std::cmp::Ordering::Equal { return cmp; }

            // Level 3: LUC (higher = first, frees more scratch)
            let cmp = b.luc.cmp(&a.luc);
            if cmp != std::cmp::Ordering::Equal { return cmp; }

            // Level 4: Net pressure (only when above soft threshold)
            if pressure_above_soft {
                let cmp = a.net_pressure().cmp(&b.net_pressure());
                if cmp != std::cmp::Ordering::Equal { return cmp; }
            }

            // Level 5: ASAP tiebreak (lower = first), then op_index
            let asap_a = a.asap.unwrap_or(0);
            let asap_b = b.asap.unwrap_or(0);
            asap_a.cmp(&asap_b).then_with(|| a.op_index.cmp(&b.op_index))
        });
    }

    fn should_defer_for_pressure(
        &self,
        _live_words: usize,
        _words_added: usize,
        _words_freed: usize,
        _scratch_budget: usize,
    ) -> bool {
        false // Never defer — cascade handles priority through sort_ready()
    }

    fn allow_shift(&self, live_words: usize, alu_used: usize, scratch_budget: usize) -> bool {
        // Same graduated zones as AdaptivePolicy
        let critical_threshold =
            (scratch_budget as f64 * self.critical_threshold_fraction) as usize;
        if live_words > critical_threshold {
            false
        } else {
            let soft_threshold =
                (scratch_budget as f64 * self.soft_threshold_fraction) as usize;
            if live_words > soft_threshold {
                alu_used <= self.alu_shift_suppress_threshold
            } else {
                true
            }
        }
    }

    fn compute_asap_inflation(
        &self,
        candidate: &SchedulingCandidate,
        ctx: &SchedulingContext,
    ) -> Option<i32> {
        let threshold = (ctx.scratch_budget as f64 * self.inflation_threshold_fraction) as usize;
        if ctx.live_words <= threshold {
            return None; // Pressure is fine, no inflation needed
        }

        let slack = candidate.slack().unwrap_or(0);
        if slack < self.min_inflation_slack {
            return None; // Critical-path op, never inflate
        }

        // Graduated inflation: more pressure → more inflation
        // pressure_ratio goes from 0.0 (at threshold) to 1.0 (at budget)
        let budget_range = ctx.scratch_budget as f64 - threshold as f64;
        if budget_range <= 0.0 {
            return None;
        }
        let pressure_ratio =
            ((ctx.live_words as f64 - threshold as f64) / budget_range).clamp(0.0, 1.0);

        // Inflation scales with both pressure and available slack
        // slack_fraction saturates toward 1.0 as slack grows (diminishing returns)
        let slack_fraction = (slack as f64 / (slack as f64 + 5.0)).min(1.0);
        let inflation =
            (pressure_ratio * slack_fraction * self.max_inflation_cycles as f64).ceil() as i32;

        if inflation > 0 {
            Some(inflation.min(self.max_inflation_cycles))
        } else {
            None
        }
    }
}
