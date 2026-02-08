//! ASAP/ALAP bounds computation pass (CP-SAT only).
//!
//! Computes scheduling bounds that define the domain for CP-SAT cycle variables.
//! Uses min_delay (best-case) for widest valid domains.

use crate::cpsat_scheduler::compute_ssa_asap_alap_bounds;
use crate::kernel_builder::pipeline::{Pass, PipelineContext, SchedulingBounds};

/// Computes ASAP/ALAP bounds from the dependency graph.
///
/// These bounds are domain hints for the CP-SAT solver. They are also
/// passed to the greedy scheduler so its hints stay within CP-SAT domains.
pub struct ComputeBoundsPass;

impl Pass for ComputeBoundsPass {
    fn name(&self) -> &'static str {
        "ComputeBounds"
    }

    fn run(&self, ctx: &mut PipelineContext) -> Result<String, String> {
        let cpsat_config = ctx
            .config
            .cpsat
            .as_ref()
            .ok_or_else(|| "ComputeBoundsPass requires CpSat config".to_string())?;

        let bounds_config = cpsat_config.to_bounds_config(
            ctx.config.compiler.enable_valu_shift,
            ctx.effective_scratch_budget(),
        );

        let (asap, alap) = compute_ssa_asap_alap_bounds(&ctx.ops, &bounds_config);

        let horizon = alap.values().copied().max().unwrap_or(0);

        ctx.bounds = Some(SchedulingBounds { asap, alap });

        Ok(format!("horizon = {}", horizon))
    }
}
