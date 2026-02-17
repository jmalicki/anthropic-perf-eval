//! Bulk-move compaction pass.
//!
//! Runs after CpSatSchedulePass when there are empty cycles. Tries to compact the
//! schedule by moving ops earlier (same constraints as CP-SAT). When disabled or
//! when there are no empty cycles, this pass is a no-op.
//!
//! **Off by default:** CP-SAT should produce schedules without empty cycles;
//! bulk-move masks solver/model bugs rather than fixing them, and is buggy
//! (same-cycle scratch conflicts, e.g. 3r/32). Use --bulk-move only for
//! experimenting, not production.

use std::collections::HashMap;

use crate::cpsat_scheduler::validation::try_validate_ssa_slot_limits;
use crate::kernel_builder::pipeline::{Pass, PipelineContext};
use crate::schedule_constraints::{check_schedule_constraints, run_bulk_move_simulator, BulkMoveResult};

/// Compacts the CP-SAT schedule by filling empty cycles (bulk-move).
///
/// Runs only when `ctx.config.cpsat.enable_bulk_move` is true and
/// `ctx.cpsat_extra.empty_cycle_count > 0`. Uses the same constraints as the
/// CP-SAT model; on success, overwrites `ctx.ops` cycles and
/// `ctx.final_decisions.shifted_alu_cycles` with the compacted schedule.
pub struct BulkMoveCompactionPass;

impl Pass for BulkMoveCompactionPass {
    fn name(&self) -> &'static str {
        "BulkMoveCompaction"
    }

    fn run(&self, ctx: &mut PipelineContext) -> Result<String, String> {
        let cpsat = match &ctx.config.cpsat {
            Some(c) => c,
            None => return Ok("skipped (not CP-SAT)".to_string()),
        };
        let extra = match &ctx.cpsat_extra {
            Some(e) => e,
            None => return Ok("skipped (no cpsat_extra)".to_string()),
        };

        if !cpsat.enable_bulk_move {
            return Ok("skipped (disabled)".to_string());
        }
        if extra.empty_cycle_count == 0 {
            return Ok("skipped (no empty cycles)".to_string());
        }

        let verbose = ctx.config.verbose;
        let total_cycles = ctx.final_metrics.cycles as i32;
        let mut cycle_assignments: Vec<i32> = ctx.ops.iter().map(|op| op.cycle).collect();
        let mut shifted_alu_cycles: HashMap<usize, Vec<i32>> = ctx
            .final_decisions
            .shifted_alu_cycles
            .iter()
            .map(|(k, v)| (*k, v.clone()))
            .collect();
        let constraints = &extra.schedule_constraints;

        let bulk_result = run_bulk_move_simulator(
            &ctx.ops,
            &mut cycle_assignments,
            &mut shifted_alu_cycles,
            constraints,
            total_cycles,
            verbose,
        );

        match &bulk_result {
            BulkMoveResult::ReachedEnd {
                original_max_cycle_plus_one,
                final_max_cycle_plus_one,
                steps,
            } if *final_max_cycle_plus_one < *original_max_cycle_plus_one => {
                let constraints_ok = check_schedule_constraints(
                    &ctx.ops,
                    &cycle_assignments,
                    &shifted_alu_cycles,
                    constraints,
                );
                let capacity_ok = try_validate_ssa_slot_limits(
                    &ctx.ops,
                    &cycle_assignments,
                    &ctx.final_decisions.shifted_ops,
                    &shifted_alu_cycles,
                    &ctx.final_decisions.const_choice_encoding,
                    &ctx.final_decisions.select_choice_encoding,
                    &ctx.final_decisions.eliminated_vbroadcast_ops,
                    &ctx.config.machine,
                );
                if constraints_ok.is_ok() && capacity_ok.is_ok() {
                    for (i, op) in ctx.ops.iter_mut().enumerate() {
                        if i < cycle_assignments.len() {
                            op.cycle = cycle_assignments[i];
                        }
                    }
                    ctx.final_decisions.shifted_alu_cycles = shifted_alu_cycles;
                    ctx.final_metrics.cycles = *final_max_cycle_plus_one as usize;
                    if let Some(extra_mut) = &mut ctx.cpsat_extra {
                        extra_mut.empty_cycle_count = 0;
                        extra_mut.ssa_cycle_assignments = cycle_assignments;
                    }
                    if verbose {
                        eprintln!(
                            "  [BulkMoveCompaction] Compacted schedule: makespan {} -> {} ({} steps); empty cycles eliminated.",
                            original_max_cycle_plus_one, final_max_cycle_plus_one, steps
                        );
                    }
                    return Ok(format!(
                        "compacted {} steps (makespan {} -> {})",
                        steps, original_max_cycle_plus_one, final_max_cycle_plus_one
                    ));
                }
                if verbose {
                    if let Err(v) = &constraints_ok {
                        eprintln!(
                            "  [BulkMoveCompaction] Compacted schedule violates constraints: {}",
                            v.message
                        );
                    }
                    if let Err(e) = &capacity_ok {
                        eprintln!(
                            "  [BulkMoveCompaction] Compacted schedule violates capacity: {}",
                            e
                        );
                    }
                }
                return Ok(format!(
                    "compaction rejected (constraints/capacity); bulk-move reduced {} -> {} ({} steps)",
                    original_max_cycle_plus_one, final_max_cycle_plus_one, steps
                ));
            }
            _ => {}
        }

        Ok("no compaction (ReachedEnd unchanged or Blocked/NoEmptyCycles)".to_string())
    }
}
