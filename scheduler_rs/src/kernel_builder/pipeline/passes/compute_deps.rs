//! Dependency computation pass.
//!
//! Computes RAW (read-after-write) dependencies with barrier constraints.

use crate::kernel_builder::emit_ssa::compute_ssa_dependencies_with_barriers;
use crate::kernel_builder::pipeline::{Pass, PipelineContext};

/// Computes SSA dependencies with barrier awareness.
///
/// Sets `op.dep_ids` on each SSA op based on RAW data dependencies
/// and barrier ordering constraints.
pub struct ComputeDependenciesPass;

impl Pass for ComputeDependenciesPass {
    fn name(&self) -> &'static str {
        "ComputeDependencies"
    }

    fn run(&self, ctx: &mut PipelineContext) -> Result<String, String> {
        compute_ssa_dependencies_with_barriers(&mut ctx.ops, &ctx.vreg_def, ctx.zero_vreg);
        Ok(format!("{} ops", ctx.ops.len()))
    }
}
