//! Greedy scheduling pass.
//!
//! Runs the greedy scheduler to produce an initial schedule (used as
//! CP-SAT warm-start hints or as the final schedule in greedy-only mode).

use crate::kernel_builder::config::PressurePolicyKind;
use crate::kernel_builder::pipeline::{Pass, PipelineContext};
use crate::kernel_builder::ssa_greedy;
use crate::kernel_builder::ssa_greedy::GreedyBoundsUsage;
use crate::kernel_builder::ssa_greedy::pressure_policy::{
    AdaptivePolicy, CascadePolicy, GreedyPolicy, PressureGatedPolicy, PressurePolicy,
    PressureThrottledPolicy,
};

/// Runs the greedy scheduler with VALU→ALU shifting and choice tracking.
///
/// Populates `ctx.greedy_decisions` with shift/choice decisions and
/// `ctx.greedy_metrics` with cycle count and peak liveness. The greedy
/// schedule is written directly into `op.cycle` on each SSA op.
///
/// Bounds usage mode is determined by the pipeline configuration:
/// - **CP-SAT pipeline** (`cpsat` config present): `Full` — core loop uses ASAP for
///   priority, deferral, and clamping so greedy hints stay within solver domains.
/// - **Greedy-only pipeline** (no `cpsat` config): `PolicyOnly` — inflation is active
///   but core loop mechanics are unaffected, avoiding ASAP deferral regression.
/// - **No bounds computed**: `None` — pure greedy scheduling.
pub struct GreedySchedulePass;

impl Pass for GreedySchedulePass {
    fn name(&self) -> &'static str {
        "GreedySchedule"
    }

    fn run(&self, ctx: &mut PipelineContext) -> Result<String, String> {
        let enable_valu_shift = ctx.config.compiler.enable_valu_shift;

        // Construct pressure policy from config
        let strategy: Box<dyn PressurePolicy> =
            match ctx.config.compiler.scheduling_pressure_policy {
                PressurePolicyKind::Greedy => Box::new(GreedyPolicy),
                PressurePolicyKind::PressureThrottled => Box::new(PressureThrottledPolicy::default()),
                PressurePolicyKind::PressureGated => Box::new(PressureGatedPolicy::default()),
                PressurePolicyKind::Adaptive => Box::new(AdaptivePolicy::default()),
                PressurePolicyKind::Cascade => Box::new(CascadePolicy::default()),
            };

        // Determine bounds usage mode based on pipeline configuration.
        // CP-SAT pipeline → Full (deferral + inflation + clamping for solver domain compliance).
        // Greedy-only pipeline → PolicyOnly (inflation only, no deferral regression).
        // No bounds → None (pure greedy, backward-compatible).
        #[cfg(feature = "cpsat")]
        let is_cpsat_pipeline = ctx.config.cpsat.is_some();
        #[cfg(not(feature = "cpsat"))]
        let is_cpsat_pipeline = false;

        let bounds_usage = match ctx.bounds.as_ref() {
            Some(b) if is_cpsat_pipeline => GreedyBoundsUsage::Full {
                asap: &b.asap,
                alap: &b.alap,
            },
            Some(b) => GreedyBoundsUsage::PolicyOnly {
                asap: &b.asap,
                alap: &b.alap,
            },
            None => GreedyBoundsUsage::None,
        };

        let (cycles, budget_exceeded, peak_live) =
            ssa_greedy::schedule_ssa_greedy_with_shift_budget_and_selectchoice(
                &mut ctx.ops,
                enable_valu_shift,
                &mut ctx.greedy_decisions.shifted_ops,
                ctx.config.scratch_budget,
                Some(&mut ctx.greedy_decisions.select_fma_ops),
                Some(&mut ctx.greedy_decisions.select_sub_cycles),
                Some(&mut ctx.greedy_decisions.select_fma_cycles),
                Some(&mut ctx.greedy_decisions.const_add_imm_ops),
                &bounds_usage,
                &*strategy,
            )?;

        ctx.greedy_metrics = super::super::SchedulingMetrics {
            cycles,
            peak_live,
            budget_exceeded,
        };

        Ok(format!(
            "{} cycles, peak {} words{}{}",
            cycles,
            peak_live,
            if enable_valu_shift {
                format!(", {} VALU shifted", ctx.greedy_decisions.shifted_ops.len())
            } else {
                String::new()
            },
            if budget_exceeded {
                " [BUDGET EXCEEDED]"
            } else {
                ""
            }
        ))
    }
}
