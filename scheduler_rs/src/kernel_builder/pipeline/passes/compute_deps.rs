//! Dependency computation pass.
//!
//! Computes RAW (read-after-write) dependencies on VRegs, the main-memory edges they do not
//! already imply (`memory_deps`), and, with `CompilerConfig::greedy_barriers` = round, the
//! round barrier constraints.

use crate::kernel_builder::barrier_elision::{count_barriers, elide_barriers};
use crate::kernel_builder::config::GreedyBarrierMode;
use crate::kernel_builder::emit_ssa::compute_ssa_dependencies;
use crate::kernel_builder::memory_deps::{add_barrier_dependencies, compute_memory_dependencies};
use crate::kernel_builder::pipeline::{Pass, PipelineContext};

/// Computes SSA dependencies.
///
/// Sets `op.dep_ids` on each SSA op from RAW data dependencies and, in
/// `GreedyBarrierMode::Round`, the barrier ordering constraints, and `op.mem_deps` from the
/// main-memory edges that the data dependencies do not imply (empty for today's emissions).
/// Fails if a memory op has no memory location. In
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
        compute_ssa_dependencies(&mut ctx.ops, &ctx.vreg_def, ctx.zero_vreg, vgl);
        // Memory edges are filtered against data paths only, before the barrier edges are
        // added: barriers are elided before CP-SAT, so they cannot be what implies an edge.
        let mem = compute_memory_dependencies(&mut ctx.ops)?;
        // Every kernel emitted today implies all of its memory edges through data flow (see
        // `memory_deps`). An emission that stores and later reads the same words (scatter
        // stores, partition stores) keeps edges and must remove this assertion.
        debug_assert_eq!(
            mem.kept, 0,
            "memory edges not implied by data dependencies in an emission that should have none: {}",
            mem.summary()
        );
        if ctx.config.compiler.greedy_barriers == GreedyBarrierMode::Round {
            add_barrier_dependencies(&mut ctx.ops);
        }
        Ok(if removed > 0 {
            format!("{} ops ({} round barriers dropped); {}", ctx.ops.len(), removed, mem.summary())
        } else {
            format!("{} ops; {}", ctx.ops.len(), mem.summary())
        })
    }
}
