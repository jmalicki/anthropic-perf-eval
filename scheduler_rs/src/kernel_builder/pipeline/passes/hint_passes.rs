//! Hint passes: overwrite greedy_decisions (ConstChoice, SelectChoice, VALU shift) based on per-cycle slack.
//!
//! Run after GreedySchedulePass. Only modify `ctx.greedy_decisions`; do not change op.cycle.
//! When a hint pass is enabled, greedy runs with a fixed default so the pass "owns" the decision.

use std::collections::HashMap;

use crate::kernel_builder::pipeline::{Pass, PipelineContext};
use crate::types::{ConstChoiceEncoding, Engine, Opcode, SelectChoiceEncoding};

/// ConstChoice hint pass: set AddImm where Flow has slack and Load is tight.
///
/// For each ConstChoice op at cycle C, count load_used and flow_used at C (from other ops and
/// current ConstChoice encodings). If flow_used < 1 and load_used >= 2, set encoding to AddImm.
pub struct ConstChoiceHintPass;

impl Pass for ConstChoiceHintPass {
    fn name(&self) -> &'static str {
        "ConstChoiceHint"
    }

    fn run(&self, ctx: &mut PipelineContext) -> Result<String, String> {
        let decisions = &ctx.greedy_decisions;
        let load_limit = 2;
        let flow_limit = 1;

        // Per-cycle (load_used, flow_used) using current encodings
        let mut cycle_load: HashMap<i32, usize> = HashMap::new();
        let mut cycle_flow: HashMap<i32, usize> = HashMap::new();
        for op in &ctx.ops {
            let c = op.cycle;
            if c < 0 {
                continue;
            }
            let (eng_load, eng_flow) = effective_load_flow(op.opcode, op.engine, decisions, op.id);
            if eng_load {
                *cycle_load.entry(c).or_insert(0) += 1;
            }
            if eng_flow {
                *cycle_flow.entry(c).or_insert(0) += 1;
            }
        }

        let mut flipped = 0usize;
        for op in &ctx.ops {
            if op.opcode != Opcode::ConstChoice || op.cycle < 0 {
                continue;
            }
            let c = op.cycle;
            let load_used = *cycle_load.get(&c).unwrap_or(&0);
            let flow_used = *cycle_flow.get(&c).unwrap_or(&0);
            // Flow has slack and Load is tight → use AddImm (Flow). At most one per cycle (Flow limit 1).
            if flow_used < flow_limit && load_used >= load_limit {
                ctx.greedy_decisions
                    .const_choice_encoding
                    .insert(op.id, ConstChoiceEncoding::AddImm);
                flipped += 1;
                if let Some(v) = cycle_load.get_mut(&c) {
                    *v = v.saturating_sub(1);
                }
                *cycle_flow.entry(c).or_insert(0) += 1;
            }
        }

        Ok(format!("{} ConstChoice → AddImm", flipped))
    }
}

fn effective_load_flow(
    opcode: Opcode,
    engine: Engine,
    decisions: &crate::kernel_builder::pipeline::SchedulingDecisions,
    op_id: usize,
) -> (bool, bool) {
    if opcode == Opcode::ConstChoice {
        let enc = decisions.const_choice_encoding.get(&op_id);
        let use_load = enc.map_or(true, |e| matches!(e, ConstChoiceEncoding::Load));
        let use_flow = enc.map_or(false, |e| matches!(e, ConstChoiceEncoding::AddImm));
        return (use_load, use_flow);
    }
    let load = engine == Engine::Load;
    let flow = engine == Engine::Flow;
    (load, flow)
}

/// SelectChoice hint pass: prefer FMA (Flow) when VALU is saturated and Flow has slack.
///
/// For each SelectChoice op at cycle C, count valu_used and flow_used at C. If valu_used >= 6
/// and flow_used < 1, set encoding to Fma (frees VALU, uses Flow for sub).
pub struct SelectChoiceHintPass;

impl Pass for SelectChoiceHintPass {
    fn name(&self) -> &'static str {
        "SelectChoiceHint"
    }

    fn run(&self, ctx: &mut PipelineContext) -> Result<String, String> {
        let decisions = &ctx.greedy_decisions;
        let valu_limit = 6;
        let flow_limit = 1;

        let mut cycle_valu: HashMap<i32, usize> = HashMap::new();
        let mut cycle_flow: HashMap<i32, usize> = HashMap::new();
        for op in &ctx.ops {
            let c = op.cycle;
            if c < 0 {
                continue;
            }
            let (use_valu, use_flow) = effective_valu_flow_select(op.opcode, op.engine, decisions, op.id);
            if use_valu {
                *cycle_valu.entry(c).or_insert(0) += 1;
            }
            if use_flow {
                *cycle_flow.entry(c).or_insert(0) += 1;
            }
        }

        let mut flipped = 0usize;
        for op in &ctx.ops {
            if op.opcode != Opcode::SelectChoice || op.cycle < 0 {
                continue;
            }
            let c = op.cycle;
            let valu_used = *cycle_valu.get(&c).unwrap_or(&0);
            let flow_used = *cycle_flow.get(&c).unwrap_or(&0);
            if valu_used >= valu_limit && flow_used < flow_limit {
                ctx.greedy_decisions
                    .select_choice_encoding
                    .insert(op.id, SelectChoiceEncoding::Fma);
                flipped += 1;
            }
        }

        Ok(format!("{} SelectChoice → FMA", flipped))
    }
}

fn effective_valu_flow_select(
    opcode: Opcode,
    engine: Engine,
    decisions: &crate::kernel_builder::pipeline::SchedulingDecisions,
    op_id: usize,
) -> (bool, bool) {
    if opcode == Opcode::SelectChoice {
        let enc = decisions.select_choice_encoding.get(&op_id);
        let use_valu = enc.map_or(true, |e| matches!(e, SelectChoiceEncoding::Vselect));
        let use_flow = enc.map_or(false, |e| matches!(e, SelectChoiceEncoding::Fma));
        return (use_valu, use_flow);
    }
    let valu = engine == Engine::Valu;
    let flow = engine == Engine::Flow;
    (valu, flow)
}

/// VALU shift hint pass: add ops to shifted_ops where ALU has slack and VALU is tight.
///
/// For each shiftable VALU op at cycle C, if valu_used >= 6 and alu_used + 8 <= 12, add to shifted_ops.
pub struct ValuShiftHintPass;

impl Pass for ValuShiftHintPass {
    fn name(&self) -> &'static str {
        "ValuShiftHint"
    }

    fn run(&self, ctx: &mut PipelineContext) -> Result<String, String> {
        use crate::kernel_builder::ssa_lower::can_ssa_valu_shift_to_alu;

        let valu_limit = 6;
        let alu_limit = 12;
        let alu_per_shift = 8;

        let mut cycle_valu: HashMap<i32, usize> = HashMap::new();
        let mut cycle_alu: HashMap<i32, usize> = HashMap::new();
        for op in &ctx.ops {
            let c = op.cycle;
            if c < 0 {
                continue;
            }
            if op.engine == Engine::Valu {
                *cycle_valu.entry(c).or_insert(0) += 1;
            }
            if op.engine == Engine::Alu {
                *cycle_alu.entry(c).or_insert(0) += 1;
            }
            if ctx.greedy_decisions.shifted_ops.contains(&op.id) {
                *cycle_alu.entry(c).or_insert(0) += alu_per_shift;
            }
        }

        let mut added = 0usize;
        for op in ctx.ops.iter() {
            if op.engine != Engine::Valu || op.cycle < 0 {
                continue;
            }
            if !can_ssa_valu_shift_to_alu(op) {
                continue;
            }
            let c = op.cycle;
            let valu_used = *cycle_valu.get(&c).unwrap_or(&0);
            let alu_used = *cycle_alu.get(&c).unwrap_or(&0);
            if valu_used >= valu_limit && alu_used + alu_per_shift <= alu_limit {
                if ctx.greedy_decisions.shifted_ops.insert(op.id) {
                    added += 1;
                    if let Some(v) = cycle_valu.get_mut(&c) {
                        *v = v.saturating_sub(1);
                    }
                    *cycle_alu.entry(c).or_insert(0) += alu_per_shift;
                }
            }
        }

        Ok(format!("{} VALU → ALU shift", added))
    }
}

// ============================================================================
// ScheduleWithFixedChoicesPass
// ============================================================================

/// Re-run greedy with encodings fixed from greedy_decisions (from hint passes or initial greedy).
///
/// Uses existing bounds; assigns new op.cycle to fill freed slots. Does not change greedy_decisions.
pub struct ScheduleWithFixedChoicesPass;

impl Pass for ScheduleWithFixedChoicesPass {
    fn name(&self) -> &'static str {
        "ScheduleWithFixedChoices"
    }

    fn run(&self, ctx: &mut PipelineContext) -> Result<String, String> {
        use crate::kernel_builder::config::PressurePolicyKind;
        use crate::kernel_builder::ssa_greedy;
        use crate::kernel_builder::ssa_greedy::GreedyBoundsUsage;
        use crate::kernel_builder::ssa_greedy::pressure_policy::{
            AdaptivePolicy, CascadePolicy, GreedyPolicy, PressureGatedPolicy, PressurePolicy,
            PressureThrottledPolicy,
        };

        let enable_valu_shift = ctx.config.compiler.enable_valu_shift;
        let strategy: Box<dyn PressurePolicy> =
            match ctx.config.compiler.scheduling_pressure_policy {
                PressurePolicyKind::Greedy => Box::new(GreedyPolicy),
                PressurePolicyKind::PressureThrottled => Box::new(PressureThrottledPolicy::default()),
                PressurePolicyKind::PressureGated => Box::new(PressureGatedPolicy::default()),
                PressurePolicyKind::Adaptive => Box::new(AdaptivePolicy::default()),
                PressurePolicyKind::Cascade => Box::new(CascadePolicy::default()),
            };

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
        let enable_capacity_deferral = ctx.config.bounds.enable_capacity_deferral;

        // Clone so we can pass refs while scheduler may read; scheduler must not overwrite greedy_decisions
        let fixed_const = ctx.greedy_decisions.const_choice_encoding.clone();
        let fixed_shifted = ctx.greedy_decisions.shifted_ops.clone();
        let fixed_select = ctx.greedy_decisions.select_choice_encoding.clone();
        let mut dummy_add_imm = std::collections::HashSet::new();
        let mut dummy_select_fma = std::collections::HashSet::new();
        let mut dummy_shifted = std::collections::HashSet::new();

        let (cycles, budget_exceeded, peak_live) =
            ssa_greedy::schedule_ssa_greedy_with_shift_budget_and_selectchoice(
                &mut ctx.ops,
                enable_valu_shift,
                &mut dummy_shifted,
                ctx.config.scratch_budget,
                Some(&mut dummy_select_fma),
                Some(&mut ctx.greedy_decisions.select_sub_cycles),
                Some(&mut ctx.greedy_decisions.select_fma_cycles),
                Some(&mut dummy_add_imm),
                &bounds_usage,
                &*strategy,
                vgl,
                enable_capacity_deferral,
                false, // not fixed to load; we use the encoding map
                false,
                false,
                Some(&fixed_const),
                Some(&fixed_shifted),
                Some(&fixed_select),
            )
            .map_err(|e| e)?;

        ctx.greedy_metrics = crate::kernel_builder::pipeline::SchedulingMetrics {
            cycles,
            peak_live,
            budget_exceeded,
        };

        Ok(format!("{} cycles (fixed choices)", cycles))
    }
}
