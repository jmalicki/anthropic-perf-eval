//! Barrier elision pass.
//!
//! Removes barrier pseudo-ops before CP-SAT (which has proper liveness
//! constraints) or before final output (barriers aren't real hardware
//! instructions).

use crate::kernel_builder::barrier_elision::elide_barriers;
use crate::kernel_builder::pipeline::{Pass, PipelineContext};

/// Removes barrier pseudo-ops from the op list.
///
/// Barriers are scheduling-only constraints used by the greedy scheduler.
/// They must be removed before CP-SAT or lowering.
pub struct ElideBarriersPass;

impl Pass for ElideBarriersPass {
    fn name(&self) -> &'static str {
        "ElideBarriers"
    }

    fn run(&self, ctx: &mut PipelineContext) -> Result<String, String> {
        let before = ctx.ops.len();
        ctx.ops = elide_barriers(std::mem::take(&mut ctx.ops));
        let after = ctx.ops.len();
        Ok(format!(
            "{} → {} ops (removed {} barriers)",
            before,
            after,
            before - after
        ))
    }
}
