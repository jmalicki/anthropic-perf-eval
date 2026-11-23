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
        // A memory edge from a removed op (a load whose result nobody reads) orders nothing
        // any more: drop it so no edge names an op that is not in the list.
        if ctx.ops.iter().any(|op| !op.mem_deps.is_empty()) {
            let kept_ids: std::collections::HashSet<usize> = ctx.ops.iter().map(|op| op.id).collect();
            for op in ctx.ops.iter_mut() {
                op.mem_deps.retain(|(id, _)| kept_ids.contains(id));
            }
        }
        let after = ctx.ops.len();
        Ok(format!(
            "{} → {} ops (removed {} dead stores)",
            before,
            after,
            before - after
        ))
    }
}
