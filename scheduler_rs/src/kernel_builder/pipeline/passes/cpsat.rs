//! CP-SAT scheduling pass.
//!
//! Runs the full-kernel CP-SAT solver to find an optimal schedule,
//! using greedy decisions as warm-start hints.

use std::collections::HashSet;

use crate::cpsat_scheduler::{build_cse_groups, schedule_ssa_cpsat, SsaCpsatHints};
use crate::kernel_builder::pipeline::context::SchedulingDecisions;
use crate::kernel_builder::pipeline::{CpSatExtra, Pass, PipelineContext, SchedulingMetrics};
use crate::types::Opcode;

/// Runs the CP-SAT solver on the full kernel.
///
/// Reads greedy decisions from `ctx.greedy_decisions` as hints, runs the
/// CP-SAT solver, and populates `ctx.final_decisions` and `ctx.final_metrics`.
pub struct CpSatSchedulePass;

impl Pass for CpSatSchedulePass {
    fn name(&self) -> &'static str {
        "CpSatSchedule"
    }

    fn run(&self, ctx: &mut PipelineContext) -> Result<String, String> {
        let cpsat_pipeline_config = ctx
            .config
            .cpsat
            .as_ref()
            .ok_or_else(|| "CpSatSchedulePass requires cpsat config".to_string())?;

        let enable_valu_shift = ctx.config.compiler.enable_valu_shift;
        let scratch_budget = ctx.effective_scratch_budget();

        // Build CP-SAT config
        let mut cpsat_config = cpsat_pipeline_config.to_full_kernel_config(scratch_budget);
        cpsat_config.enable_valu_shift = enable_valu_shift;

        // Collect hints from greedy decisions
        let greedy_hints: Vec<i32> = ctx.ops.iter().map(|op| op.cycle).collect();

        // Build shift hints (from greedy)
        let shift_hints_set = if enable_valu_shift && !ctx.greedy_decisions.shifted_ops.is_empty() {
            Some(ctx.greedy_decisions.shifted_ops.clone())
        } else {
            None
        };
        let shift_hints = shift_hints_set.as_ref();

        // Build SelectChoice hints (Fma op ids from greedy)
        let op_ids: HashSet<usize> = ctx.ops.iter().map(|op| op.id).collect();
        let select_hints_set: HashSet<usize> = ctx
            .greedy_decisions
            .select_choice_encoding
            .iter()
            .filter(|(_, e)| matches!(e, crate::types::SelectChoiceEncoding::Fma))
            .map(|(id, _)| *id)
            .filter(|id| op_ids.contains(id))
            .collect();
        let select_hints = if select_hints_set.is_empty() {
            None
        } else {
            Some(&select_hints_set)
        };

        // Forced shifts
        let forced_shifts = cpsat_pipeline_config.forced_shifts.as_ref();

        let greedy_budget_exceeded = ctx.greedy_metrics.budget_exceeded;

        // Reset cycles before CP-SAT
        for op in &mut ctx.ops {
            op.cycle = -1;
        }

        let enable_cse = cpsat_pipeline_config.enable_cse_groups;
        let cse_groups = if enable_cse {
            build_cse_groups(&ctx.ops)
        } else {
            vec![]
        };
        let cse_groups_ref = if cse_groups.is_empty() {
            None
        } else {
            Some(cse_groups.as_slice())
        };
        let const_add_imm_hints: std::collections::HashSet<usize> = ctx
            .greedy_decisions
            .const_choice_encoding
            .iter()
            .filter(|(_, e)| matches!(e, crate::types::ConstChoiceEncoding::AddImm))
            .map(|(id, _)| *id)
            .collect();
        let cpsat_result = schedule_ssa_cpsat(
            &ctx.ops,
            cpsat_config,
            SsaCpsatHints {
                greedy_hints: Some(&greedy_hints),
                shift_hints,
                forced_shifts,
                select_choice_hints: select_hints,
                const_add_imm_hints: Some(&const_add_imm_hints),
                eliminate_vbroadcast_hints: None,
                initial_live: None,
                live_out_vregs: None,
                live_through_vregs: None,
                cse_groups: cse_groups_ref,
                vector_group_lanes: Some(&ctx.vector_group_lanes),
            },
            Some(!greedy_budget_exceeded),
        )
        .and_then(|o| o.ok_or_else(|| "CP-SAT scheduling failed".to_string()))?;

        // Apply cycle assignments
        for (idx, op) in ctx.ops.iter_mut().enumerate() {
            if idx < cpsat_result.cycle_assignments.len() {
                op.cycle = cpsat_result.cycle_assignments[idx];
            }
        }

        let cycles = cpsat_result.makespan;

        // Populate final decisions from CP-SAT result
        ctx.final_decisions = SchedulingDecisions {
            shifted_ops: cpsat_result.shifted_ops,
            select_choice_encoding: cpsat_result.select_choice_encoding,
            const_choice_encoding: cpsat_result.const_choice_encoding,
            shifted_alu_cycles: cpsat_result.shifted_alu_cycles.clone(),
            eliminated_vbroadcast_ops: cpsat_result.eliminated_vbroadcast_ops.clone(),
            eliminated_vb_to_scalar_producer: cpsat_result
                .eliminated_vb_to_scalar_producer
                .clone(),
            select_sub_cycles: Default::default(),
            select_fma_cycles: Default::default(),
            cse_groups,
            cse_group_results: cpsat_result.cse_group_results.clone(),
            ..Default::default()
        };

        ctx.final_metrics = SchedulingMetrics {
            cycles,
            peak_live: 0, // Will be set by address assignment
            budget_exceeded: false,
            ..Default::default()
        };

        // Capture CP-SAT-specific extra results
        let ssa_cycle_assignments = cpsat_result.cycle_assignments.clone();
        let ssa_ops_bulk_move: Vec<(usize, Vec<usize>)> = ctx
            .ops
            .iter()
            .map(|o| (o.id, o.dep_ids.iter().copied().collect()))
            .collect();

        ctx.cpsat_extra = Some(CpSatExtra {
            empty_cycle_count: cpsat_result.empty_cycle_count,
            makespan_driven_by_capacity: cpsat_result.makespan_driven_by_capacity,
            schedule_constraints: cpsat_result.schedule_constraints.clone(),
            ssa_cycle_assignments,
            ssa_ops_bulk_move,
            lb_cycles: cpsat_result.lb_cycles,
        });

        // Report choices
        let const_total = ctx
            .ops
            .iter()
            .filter(|o| o.opcode == Opcode::ConstChoice)
            .count();
        let const_add_imm = ctx
            .final_decisions
            .const_choice_encoding
            .values()
            .filter(|e| matches!(e, crate::types::ConstChoiceEncoding::AddImm))
            .count();
        let select_total = ctx
            .ops
            .iter()
            .filter(|o| o.opcode == Opcode::SelectChoice)
            .count();
        let select_fma = ctx
            .final_decisions
            .select_choice_encoding
            .values()
            .filter(|e| matches!(e, crate::types::SelectChoiceEncoding::Fma))
            .count();

        Ok(format!(
            "{} cycles (greedy was {}), {} VALU shifted | ConstChoice: {} AddImm, {} Load | SelectChoice: {} FMA, {} vselect",
            cycles,
            ctx.greedy_metrics.cycles,
            ctx.final_decisions.shifted_ops.len(),
            const_add_imm,
            const_total.saturating_sub(const_add_imm),
            select_fma,
            select_total.saturating_sub(select_fma),
        ))
    }
}
