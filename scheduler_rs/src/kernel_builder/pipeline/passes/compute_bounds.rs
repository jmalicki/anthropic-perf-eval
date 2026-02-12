//! ASAP/ALAP bounds computation pass.
//!
//! Computes scheduling bounds for both greedy and CP-SAT pipelines.
//! Uses min_delay (best-case) for widest valid domains.
//!
//! For greedy: bounds enable ASAP inflation (deferring high-slack producers
//! under pressure). For CP-SAT: bounds define cycle variable domains.

use crate::cpsat_scheduler::bounds::{
    compute_max_latency_critical_path, compute_ssa_alap, compute_ssa_asap,
};
use crate::kernel_builder::pipeline::{Pass, PipelineContext, SchedulingBounds};
use crate::types::Engine;
use std::collections::HashSet;

/// Computes ASAP/ALAP bounds from the dependency graph.
///
/// These bounds serve dual purposes:
/// - **Greedy**: Enable ASAP inflation to defer high-slack producers under pressure
/// - **CP-SAT**: Domain hints for cycle variables
///
/// Always runs (not feature-gated). Tightening options are controlled by
/// `PipelineConfig.bounds`.
pub struct ComputeBoundsPass;

impl Pass for ComputeBoundsPass {
    fn name(&self) -> &'static str {
        "ComputeBounds"
    }

    fn run(&self, ctx: &mut PipelineContext) -> Result<String, String> {
        let enable_valu_shift = ctx.config.compiler.enable_valu_shift;

        // Compute shiftable set
        let shiftable_set: HashSet<usize> = if enable_valu_shift {
            ctx.ops
                .iter()
                .enumerate()
                .filter(|(_, op)| op.engine == Engine::Valu && op.opcode.can_shift_to_alu())
                .map(|(idx, _)| idx)
                .collect()
        } else {
            HashSet::new()
        };

        // Compute basic ASAP/ALAP bounds
        let asap = compute_ssa_asap(&ctx.ops, &shiftable_set);
        let n = ctx.ops.len();
        let critical_path_len = asap.values().max().copied().unwrap_or(0) + 1;
        let max_latency_critical_path =
            compute_max_latency_critical_path(&ctx.ops, &shiftable_set);
        let horizon = (critical_path_len * 2)
            .max(max_latency_critical_path * 2)
            .max(n as i32);
        let alap = compute_ssa_alap(&ctx.ops, horizon, &shiftable_set);

        let mut current_asap = asap;
        let mut current_alap = alap;

        // Optional tightening (controlled by BoundsConfig)
        let bounds_cfg = &ctx.config.bounds;

        if bounds_cfg.enable_asap_alap_tightening {
            let machine = &ctx.config.machine;
            if let Some((slot_asap, slot_alap)) =
                crate::cpsat_scheduler::bounds::tighten_asap_alap_by_slots(
                    &ctx.ops,
                    &current_asap,
                    &current_alap,
                    horizon,
                    machine.valu_slots as i64,
                    machine.alu_slots as i64,
                    machine.load_slots as i64,
                    machine.store_slots as i64,
                    machine.flow_slots as i64,
                    false, // verbose
                    Some(&shiftable_set),
                )
            {
                current_asap = slot_asap;
                current_alap = slot_alap;
            }
        }

        if bounds_cfg.enable_scratch_tightening {
            let scratch_budget = ctx.effective_scratch_budget() as i32;
            if let Some((scratch_asap, scratch_alap)) =
                crate::cpsat_scheduler::bounds::tighten_asap_alap_by_scratch(
                    &ctx.ops,
                    &current_asap,
                    &current_alap,
                    horizon,
                    scratch_budget,
                    false, // verbose
                )
            {
                current_asap = scratch_asap;
                current_alap = scratch_alap;
            }
        }

        // After tightening, re-propagate precedence constraints
        if bounds_cfg.enable_asap_alap_tightening || bounds_cfg.enable_scratch_tightening {
            let (fixed_asap, fixed_alap) =
                crate::cpsat_scheduler::bounds::fix_precedence_bounds_fixpoint(
                    &ctx.ops,
                    &current_asap,
                    &current_alap,
                    &shiftable_set,
                );
            current_asap = fixed_asap;
            current_alap = fixed_alap;
            // Enforce non-empty domains: for each op, alap >= asap (avoids clamp(lb, ub) panic in greedy)
            for op in &ctx.ops {
                let a = *current_asap.get(&op.id).unwrap_or(&0);
                let b = *current_alap.get(&op.id).unwrap_or(&i32::MAX);
                if a > b {
                    current_alap.insert(op.id, a);
                }
            }
        }

        let final_horizon = current_alap.values().copied().max().unwrap_or(0);

        ctx.bounds = Some(SchedulingBounds {
            asap: current_asap,
            alap: current_alap,
        });

        Ok(format!("horizon = {}", final_horizon))
    }
}
