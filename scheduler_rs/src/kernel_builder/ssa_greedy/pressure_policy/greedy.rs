//! Greedy pressure policy — no pressure awareness.
//!
//! Schedules purely by LUC priority, defers only at the hard budget wall,
//! and always allows VALU→ALU shifts. This is the original heuristic.

use super::{PressurePolicy, SchedulingCandidate, SchedulingContext};

/// No pressure awareness: LUC sort, defer at hard budget only, always shift.
#[derive(Debug, Clone, Copy)]
pub struct GreedyPolicy;

impl PressurePolicy for GreedyPolicy {
    fn sort_ready(
        &self,
        candidates: &mut [SchedulingCandidate],
        _ctx: &SchedulingContext,
    ) {
        // Sort by LUC descending
        candidates.sort_by(|a, b| b.luc.cmp(&a.luc));
    }

    fn should_defer_for_pressure(
        &self,
        _live_words: usize,
        _words_added: usize,
        _words_freed: usize,
        _scratch_budget: usize,
    ) -> bool {
        false
    }

    fn allow_shift(&self, _live_words: usize, _alu_used: usize, _scratch_budget: usize) -> bool {
        true
    }
}
