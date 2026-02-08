//! Greedy scheduling pass.
//!
//! Runs the greedy scheduler to produce an initial schedule (used as
//! CP-SAT warm-start hints or as the final schedule in greedy-only mode).

use crate::kernel_builder::config::PressurePolicyKind;
use crate::kernel_builder::pipeline::{Pass, PipelineContext};
use crate::kernel_builder::ssa_greedy;
use crate::kernel_builder::ssa_greedy::pressure_policy::{
    AdaptivePolicy, GreedyPolicy, PressureGatedPolicy, PressurePolicy, PressureThrottledPolicy,
};

/// Runs the greedy scheduler with VALU→ALU shifting and choice tracking.
///
/// Populates `ctx.greedy_decisions` with shift/choice decisions and
/// `ctx.greedy_metrics` with cycle count and peak liveness. The greedy
/// schedule is written directly into `op.cycle` on each SSA op.
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
            };

        // Prepare ASAP/ALAP bounds parameter
        let asap_alap_param = ctx.bounds.as_ref().map(|b| (&b.asap, &b.alap));

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
                asap_alap_param,
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
