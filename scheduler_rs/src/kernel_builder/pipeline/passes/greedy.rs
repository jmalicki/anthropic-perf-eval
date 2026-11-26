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
            return Self::from_greedy_forced(ctx);
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
    fn from_context(ctx: &PipelineContext) -> Self {
        Self::from_greedy_forced(ctx)
    }

    /// Greedy-only build: the encodings `--const-choice`, `--select-choice` and `--valu-shift`
    /// force (`CompilerConfig::greedy_forced`). Before this they were ignored without CP-SAT.
    fn from_greedy_forced(ctx: &PipelineContext) -> Self {
        let forced = &ctx.config.compiler.greedy_forced;
        let all = |opcode: Opcode| ctx.ops.iter().filter(move |op| op.opcode == opcode).map(|op| op.id);
        let const_choice = forced
            .const_choice
            .as_ref()
            .map(|enc| all(Opcode::ConstChoice).map(|id| (id, enc.clone())).collect());
        let select_choice = forced
            .select_choice
            .as_ref()
            .map(|enc| all(Opcode::SelectChoice).map(|id| (id, enc.clone())).collect());
        let shifted_ops = match forced.valu_shift {
            None => None,
            Some(false) => Some(HashSet::new()),
            Some(true) if ctx.config.compiler.enable_valu_shift => Some(
                ctx.ops
                    .iter()
                    .filter(|op| op.engine == crate::types::Engine::Valu && op.opcode.can_shift_to_alu())
                    .map(|op| op.id)
                    .collect(),
            ),
            Some(true) => None,
        };
        Self { const_choice, shifted_ops, select_choice }
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
        let use_full_bounds = (is_cpsat_pipeline && !ctx.config.compiler.cpsat_greedy_policy_bounds)
            || ctx.config.bounds.greedy_use_full_bounds;

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
        // A split shift reaches CP-SAT only as per-lane hints (cpsat_split_shift_hint);
        // without them the hint has one cycle per op and cannot describe it.
        if is_cpsat_pipeline && !ctx.config.compiler.cpsat_split_shift_hint {
            shape.split_shift = false;
        }
        // ALU-derived constants are not hinted to CP-SAT (its hint names Load or AddImm).
        if is_cpsat_pipeline {
            shape.const_policy = ssa_greedy::ConstPolicy::Default;
        }
        let mut shifted_lane_cycles = std::collections::HashMap::new();
        let greedy_budget = ctx.greedy_budget();
        let ((cycles, budget_exceeded, peak_live), _agreement) =
            ssa_greedy::schedule_ssa_greedy_shaped(
                &mut ctx.ops,
                enable_valu_shift,
                &mut ctx.greedy_decisions.shifted_ops,
                greedy_budget,
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
                let enc = if let Some(info) = diagnostics.const_alu_derive.get(&op.id) {
                    ConstChoiceEncoding::AluDerive(info.clone())
                } else if add_imm_ops.contains(&op.id) {
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
            // The budget the scheduler ran against (headroom already subtracted).
            budget: greedy_budget,
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

// ============================================================================
// GreedyJustifyPass: forward/backward justification of the greedy schedule
// ============================================================================

/// Shortens the greedy schedule by forward/backward justification
/// (`ssa_greedy::justify`), keeping the greedy's ConstChoice / SelectChoice encodings.
///
/// Runs after `GreedySchedulePass` and the hint passes, only when
/// `compiler.greedy_justify_iterations > 0`. It rewrites `op.cycle`,
/// `greedy_decisions.shifted_ops` and `greedy_decisions.shifted_alu_cycles` and updates
/// `greedy_metrics`, or leaves all of them untouched when no shorter schedule passes its
/// checks (slot capacities, precedences, memory edges, scratch).
///
/// Shift decisions change only when the greedy owned them (shifting enabled, no forced
/// shift encoding, no hint pass, no fixed-choices reschedule); split shifts are used
/// under the same condition as in `GreedySchedulePass`. When CP-SAT follows, an accepted
/// schedule must also keep every op inside its ASAP/ALAP domain, since it becomes the hint.
/// With individually forced shifts (what-if runs) the pass does not run at all.
pub struct GreedyJustifyPass;

impl Pass for GreedyJustifyPass {
    fn name(&self) -> &'static str {
        "GreedyJustify"
    }

    fn run(&self, ctx: &mut PipelineContext) -> Result<String, String> {
        use crate::address_assign::AddressAssignConfig;
        use crate::kernel_builder::ssa_greedy::justify::{
            justify_schedule, JustifyConfig, DEFAULT_MAX_STALLED_ITERATIONS,
        };

        let compiler = &ctx.config.compiler;
        if compiler.greedy_justify_iterations == 0 {
            return Ok("off".to_string());
        }
        let announce = |ctx: &PipelineContext, summary: &str| {
            // The pipeline prints pass summaries only when verbose; the greedy-only path
            // runs it non-verbose, so print the line here in that case.
            if ctx.config.greedy_verbose && !ctx.config.verbose {
                eprintln!("Greedy justify: {}", summary);
            }
        };
        if ctx.greedy_metrics.budget_exceeded {
            let summary = "skipped: the greedy schedule is over the scratch budget".to_string();
            announce(ctx, &summary);
            return Ok(summary);
        }
        if !ctx.greedy_decisions.eliminated_vbroadcast_ops.is_empty() {
            let summary = "skipped: eliminated vbroadcasts are not modelled".to_string();
            announce(ctx, &summary);
            return Ok(summary);
        }
        // `--greedy-const-policy` derives constants on the scalar ALU from producers that are
        // still live where the greedy placed the derivation. Justification would move both
        // without modelling the derivation's reads (precedence on its producers, their
        // lifetimes in the scratch check and in address assignment), so it does not run.
        let alu_derived = ctx
            .greedy_decisions
            .const_choice_encoding
            .values()
            .filter(|e| matches!(e, ConstChoiceEncoding::AluDerive(_)))
            .count();
        if alu_derived > 0 {
            let summary = format!(
                "skipped: {} ALU-derived constant(s) (--greedy-const-policy) are not modelled",
                alu_derived
            );
            announce(ctx, &summary);
            return Ok(summary);
        }
        #[cfg(feature = "cpsat")]
        let is_cpsat_pipeline = ctx.config.cpsat.is_some();
        #[cfg(not(feature = "cpsat"))]
        let is_cpsat_pipeline = false;

        // Individually forced shifts (what-if runs) are hard constraints of the CP-SAT model
        // that the greedy does not see: its schedule may not be a schedule of that model, yet
        // its length still sets the model's horizon and latest cycles. A shorter schedule
        // would tighten those around encodings the model forbids, so the pass does not run.
        #[cfg(feature = "cpsat")]
        let has_forced_shifts =
            ctx.config.cpsat.as_ref().is_some_and(|c| c.forced_shifts.as_ref().is_some_and(|ids| !ids.is_empty()));
        #[cfg(not(feature = "cpsat"))]
        let has_forced_shifts = false;
        if has_forced_shifts {
            let summary = "skipped: individually forced shifts are set".to_string();
            announce(ctx, &summary);
            return Ok(summary);
        }

        let forced = ForcedEncodings::from_context(ctx);
        let allow_shift_changes = compiler.enable_valu_shift
            && forced.shifted_ops.is_none()
            && !compiler.enable_valu_shift_hint_pass
            && !compiler.enable_const_choice_hint_pass
            && !compiler.enable_select_choice_hint_pass
            && !compiler.enable_schedule_with_fixed_choices;
        let split_shift = allow_shift_changes
            && compiler.greedy_split_shift
            && (!is_cpsat_pipeline || compiler.cpsat_split_shift_hint);

        // Address assignment as `AssignAddressesPass` will configure it for these decisions.
        let mut allocator = AddressAssignConfig {
            budget: compiler.scratch_size,
            scratch_pools: compiler.scratch_model.is_pools(),
            vector_group_lanes: ctx.vector_group_lanes.clone(),
            vreg_def: ctx.vreg_def.clone(),
            select_fma_ops: ctx
                .greedy_decisions
                .select_choice_encoding
                .iter()
                .filter(|(_, e)| matches!(e, SelectChoiceEncoding::Fma))
                .map(|(id, _)| *id)
                .collect(),
            ..Default::default()
        };
        if let Some(zero) = ctx.zero_vreg {
            allocator.fixed_vregs.insert(zero, 0);
            allocator.zero_vreg = Some(zero);
        }
        let config = JustifyConfig {
            max_iterations: compiler.greedy_justify_iterations,
            max_stalled_iterations: DEFAULT_MAX_STALLED_ITERATIONS,
            allow_shift_changes,
            split_shift,
            scratch_budget: ctx.greedy_budget(),
            scratch_pools: compiler.scratch_model.is_pools(),
            allocator: Some(allocator),
        };
        let use_full_bounds = is_cpsat_pipeline || ctx.config.bounds.greedy_use_full_bounds;
        let domains = ctx.bounds.as_ref().filter(|_| use_full_bounds).map(|b| (&b.asap, &b.alap));
        let vgl = if ctx.vector_group_lanes.is_empty() { None } else { Some(&ctx.vector_group_lanes) };
        let enable_valu_shift = ctx.config.compiler.enable_valu_shift;
        let outcome = justify_schedule(
            &mut ctx.ops,
            enable_valu_shift,
            &mut ctx.greedy_decisions.shifted_ops,
            &mut ctx.greedy_decisions.shifted_alu_cycles,
            &ctx.greedy_decisions.const_choice_encoding,
            &ctx.greedy_decisions.select_choice_encoding,
            vgl,
            domains,
            &config,
        )?;
        if outcome.cycles_after < outcome.cycles_before {
            ctx.greedy_metrics.cycles = outcome.cycles_after;
            ctx.greedy_metrics.peak_live = outcome.peak_live_after;
        }
        let summary = outcome.summary();
        announce(ctx, &summary);
        Ok(summary)
    }
}
