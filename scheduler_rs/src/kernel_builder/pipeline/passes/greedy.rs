//! Greedy scheduling pass.
//!
//! Runs the greedy scheduler to produce an initial schedule (used as
//! CP-SAT warm-start hints or as the final schedule in greedy-only mode).

use std::collections::{HashMap, HashSet};

use crate::kernel_builder::pipeline::{Pass, PipelineContext};
use crate::kernel_builder::ssa_greedy;
use crate::kernel_builder::ssa_greedy::GreedyBoundsUsage;
use crate::types::{ConstChoiceEncoding, Opcode, SelectChoiceEncoding};

/// Runs the greedy scheduler with VALU→ALU shifting and choice tracking.
///
/// Populates `ctx.greedy_decisions` with shift/choice decisions and
/// `ctx.greedy_metrics` with cycle count and peak liveness. The greedy
/// schedule is written directly into `op.cycle` on each SSA op.
///
/// When a hint pass is enabled (enable_*_hint_pass), greedy uses a fixed default
/// for that choice so the hint pass owns the decision: ConstChoice→Load,
/// SelectChoice→vselect, VALU shift→none.
///
/// Bounds usage mode is determined by the pipeline configuration:
/// - **CP-SAT pipeline** (`cpsat` config present): `Full` — core loop uses ASAP for
///   priority, deferral, and clamping so greedy hints stay within solver domains.
/// - **Greedy-only pipeline** (no `cpsat` config): `PolicyOnly` — inflation is active
///   but core loop mechanics are unaffected, avoiding ASAP deferral regression.
/// - **No bounds computed**: `None` — pure greedy scheduling.
pub struct GreedySchedulePass;

/// Encodings the greedy must use because CP-SAT will force them. `None` = greedy decides.
#[derive(Default)]
struct ForcedEncodings {
    const_choice: Option<HashMap<usize, ConstChoiceEncoding>>,
    shifted_ops: Option<HashSet<usize>>,
    select_choice: Option<HashMap<usize, SelectChoiceEncoding>>,
}

impl ForcedEncodings {
    #[cfg(feature = "cpsat")]
    fn from_context(ctx: &PipelineContext) -> Self {
        use crate::cpsat_scheduler::{ConstChoiceMode, SelectChoiceMode, ValuShiftMode};
        use crate::types::Engine;

        let Some(cpsat) = ctx.config.cpsat.as_ref() else {
            return Self::default();
        };
        let all = |opcode: Opcode| ctx.ops.iter().filter(move |op| op.opcode == opcode).map(|op| op.id);
        let const_choice = match cpsat.const_choice_mode {
            ConstChoiceMode::Optimize => None,
            ConstChoiceMode::ForceAddImm => {
                Some(all(Opcode::ConstChoice).map(|id| (id, ConstChoiceEncoding::AddImm)).collect())
            }
            ConstChoiceMode::ForceLoad => {
                Some(all(Opcode::ConstChoice).map(|id| (id, ConstChoiceEncoding::Load)).collect())
            }
        };
        let select_choice = match cpsat.select_choice_mode {
            SelectChoiceMode::Optimize => None,
            SelectChoiceMode::ForceFma => {
                Some(all(Opcode::SelectChoice).map(|id| (id, SelectChoiceEncoding::Fma)).collect())
            }
            SelectChoiceMode::ForceVselect => {
                Some(all(Opcode::SelectChoice).map(|id| (id, SelectChoiceEncoding::Vselect)).collect())
            }
        };
        // Same shiftable set as the CP-SAT model (cpsat_scheduler/model.rs, section 0).
        let shifted_ops = match cpsat.valu_shift_mode {
            ValuShiftMode::Optimize => None,
            ValuShiftMode::ForceValu => Some(HashSet::new()),
            ValuShiftMode::ForceAlu if ctx.config.compiler.enable_valu_shift => Some(
                ctx.ops
                    .iter()
                    .filter(|op| op.engine == Engine::Valu && op.opcode.can_shift_to_alu())
                    .map(|op| op.id)
                    .collect(),
            ),
            ValuShiftMode::ForceAlu => None,
        };
        Self { const_choice, shifted_ops, select_choice }
    }

    #[cfg(not(feature = "cpsat"))]
    fn from_context(_ctx: &PipelineContext) -> Self {
        Self::default()
    }
}

impl Pass for GreedySchedulePass {
    fn name(&self) -> &'static str {
        "GreedySchedule"
    }

    fn run(&self, ctx: &mut PipelineContext) -> Result<String, String> {
        let enable_valu_shift = ctx.config.compiler.enable_valu_shift;

        // Construct pressure policy from config
        let strategy = ctx.config.compiler.pressure_policy();

        // Determine bounds usage mode based on pipeline configuration.
        // CP-SAT pipeline → Full (deferral + inflation + clamping for solver domain compliance).
        // Greedy-only pipeline → PolicyOnly unless greedy_use_full_bounds (investigation mode).
        // No bounds → None (pure greedy, backward-compatible).
        #[cfg(feature = "cpsat")]
        let is_cpsat_pipeline = ctx.config.cpsat.is_some();
        #[cfg(not(feature = "cpsat"))]
        let is_cpsat_pipeline = false;
        let use_full_bounds = is_cpsat_pipeline || ctx.config.bounds.greedy_use_full_bounds;

        let bounds_usage = match ctx.bounds.as_ref() {
            Some(b) if use_full_bounds => GreedyBoundsUsage::Full {
                asap: &b.asap,
                alap: &b.alap,
            },
            Some(b) => GreedyBoundsUsage::PolicyOnly {
                asap: &b.asap,
                alap: &b.alap,
            },
            None => GreedyBoundsUsage::None,
        };

        let vgl = if ctx.vector_group_lanes.is_empty() {
            None
        } else {
            Some(&ctx.vector_group_lanes)
        };
        let mut add_imm_ops = HashSet::new();
        let mut select_fma_ops = HashSet::new();
        let enable_capacity_deferral = ctx.config.bounds.enable_capacity_deferral;
        // When a hint pass is enabled, greedy uses fixed default so the hint pass owns the decision.
        let const_choice_fixed_to_load = ctx.config.compiler.enable_const_choice_hint_pass;
        let select_choice_fixed_to_vselect = ctx.config.compiler.enable_select_choice_hint_pass;
        let valu_shift_fixed_to_none = ctx.config.compiler.enable_valu_shift_hint_pass;
        // Forced CP-SAT encodings (--valu-shift alu, --const-choice add_imm, --select-choice fma,
        // and their opposites) must be honoured here too: CP-SAT bounds its search with this
        // schedule, which is only valid if it uses the encodings the model will force.
        let forced = ForcedEncodings::from_context(ctx);
        let mut diagnostics = ssa_greedy::GreedyDiagnostics::new(ctx.config.greedy_verbose);
        let mut shape = ctx.config.compiler.greedy_shape();
        // CP-SAT hints carry one cycle per op, so a split shift cannot be hinted.
        if is_cpsat_pipeline {
            shape.split_shift = false;
        }
        let mut shifted_lane_cycles = std::collections::HashMap::new();
        let ((cycles, budget_exceeded, peak_live), _agreement) =
            ssa_greedy::schedule_ssa_greedy_shaped(
                &mut ctx.ops,
                enable_valu_shift,
                &mut ctx.greedy_decisions.shifted_ops,
                ctx.config.compiler.greedy_scratch_budget(ctx.config.scratch_budget),
                Some(&mut select_fma_ops),
                Some(&mut ctx.greedy_decisions.select_sub_cycles),
                Some(&mut ctx.greedy_decisions.select_fma_cycles),
                Some(&mut add_imm_ops),
                &bounds_usage,
                &*strategy,
                vgl,
                enable_capacity_deferral,
                const_choice_fixed_to_load,
                select_choice_fixed_to_vselect,
                valu_shift_fixed_to_none,
                forced.const_choice.as_ref(),
                forced.shifted_ops.as_ref(),
                forced.select_choice.as_ref(),
                &mut diagnostics,
                &shape,
                &mut shifted_lane_cycles,
            )?;
        ctx.greedy_decisions.shifted_alu_cycles = shifted_lane_cycles;
        // With a fixed encoding the scheduler does not record its (non-)decisions; record them
        // here so hints and lowering see what was scheduled.
        if let Some(shifted) = forced.shifted_ops {
            ctx.greedy_decisions.shifted_ops = shifted;
        }
        if let Some(fixed) = &forced.const_choice {
            add_imm_ops = fixed
                .iter()
                .filter(|(_, enc)| matches!(enc, ConstChoiceEncoding::AddImm))
                .map(|(id, _)| *id)
                .collect();
        }
        if let Some(fixed) = &forced.select_choice {
            select_fma_ops = fixed
                .iter()
                .filter(|(_, enc)| matches!(enc, SelectChoiceEncoding::Fma))
                .map(|(id, _)| *id)
                .collect();
        }
        for op in &ctx.ops {
            if op.opcode == Opcode::ConstChoice {
                let enc = if add_imm_ops.contains(&op.id) {
                    ConstChoiceEncoding::AddImm
                } else {
                    ConstChoiceEncoding::Load
                };
                ctx.greedy_decisions.const_choice_encoding.insert(op.id, enc);
            }
            if op.opcode == Opcode::SelectChoice {
                let enc = if select_fma_ops.contains(&op.id) {
                    SelectChoiceEncoding::Fma
                } else {
                    SelectChoiceEncoding::Vselect
                };
                ctx.greedy_decisions.select_choice_encoding.insert(op.id, enc);
            }
        }

        ctx.greedy_metrics = super::super::SchedulingMetrics {
            cycles,
            peak_live,
            budget_exceeded,
            budget: ctx.config.scratch_budget,
            over_budget_cycles: diagnostics.over_budget_cycles,
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
