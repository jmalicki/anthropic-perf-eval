//! Dead store filtering pass.
//!
//! Removes ops with cycle=-1 (unscheduled) before address assignment.
//! These are ops whose dest VReg has no users — including them would
//! corrupt live interval computation.

use crate::kernel_builder::pipeline::{Pass, PipelineContext};

/// Filters out ops with `cycle < 0` (unscheduled / dead stores).
///
/// Dead stores are ops detected by the scheduler but left with cycle=-1.
/// Including them in address assignment would corrupt live intervals.
pub struct FilterDeadStoresPass;

impl Pass for FilterDeadStoresPass {
    fn name(&self) -> &'static str {
        "FilterDeadStores"
    }

    fn run(&self, ctx: &mut PipelineContext) -> Result<String, String> {
        let before = ctx.ops.len();
        ctx.ops.retain(|op| op.cycle >= 0);
        let after = ctx.ops.len();
        Ok(format!(
            "{} → {} ops (removed {} dead stores)",
            before,
            after,
            before - after
        ))
    }
}
