//! Dependency computation pass.
//!
//! Computes RAW (read-after-write) dependencies, with or without round barrier constraints
//! (`CompilerConfig::greedy_barriers`).

use crate::kernel_builder::barrier_elision::{count_barriers, elide_barriers};
use crate::kernel_builder::config::GreedyBarrierMode;
use crate::kernel_builder::emit_ssa::{
    compute_ssa_dependencies, compute_ssa_dependencies_with_barriers,
};
use crate::kernel_builder::pipeline::{Pass, PipelineContext};

/// Computes SSA dependencies.
///
/// Sets `op.dep_ids` on each SSA op from RAW data dependencies and, in
/// `GreedyBarrierMode::Round`, the barrier ordering constraints. In
/// `GreedyBarrierMode::None` the barrier pseudo-ops are removed here, so every later pass
/// (bounds, greedy, per-section budget check, CP-SAT) sees one section ordered by data
/// dependencies only.
pub struct ComputeDependenciesPass;

impl Pass for ComputeDependenciesPass {
    fn name(&self) -> &'static str {
        "ComputeDependencies"
    }

    fn run(&self, ctx: &mut PipelineContext) -> Result<String, String> {
        let mut removed = 0;
        if ctx.config.compiler.greedy_barriers == GreedyBarrierMode::None {
            removed = count_barriers(&ctx.ops);
            if removed > 0 {
                ctx.ops = elide_barriers(std::mem::take(&mut ctx.ops));
            }
        }
        let vgl = if ctx.vector_group_lanes.is_empty() {
            None
        } else {
            Some(&ctx.vector_group_lanes)
        };
        match ctx.config.compiler.greedy_barriers {
            GreedyBarrierMode::Round => compute_ssa_dependencies_with_barriers(
                &mut ctx.ops,
                &ctx.vreg_def,
                ctx.zero_vreg,
                vgl,
            ),
            GreedyBarrierMode::None => {
                compute_ssa_dependencies(&mut ctx.ops, &ctx.vreg_def, ctx.zero_vreg, vgl)
            }
        }
        Ok(if removed > 0 {
            format!("{} ops ({} round barriers dropped)", ctx.ops.len(), removed)
        } else {
            format!("{} ops", ctx.ops.len())
        })
    }
}
