//! Address assignment pass.
//!
//! Assigns concrete scratch addresses to VRegs via linear scan allocation.

use std::collections::HashMap;

use crate::address_assign::{assign_addresses, verify_no_address_conflicts_fast, AddressAssignConfig};
use crate::kernel_builder::pipeline::{Pass, PipelineContext};
use crate::kernel_builder::tick_tock_ssa::SCRATCH_BUDGET_HW;

/// Assigns physical scratch addresses to VRegs using linear scan allocation.
///
/// Reads the scheduled ops and scheduling decisions to build proper live
/// intervals, then allocates addresses within the scratch budget.
///
/// Does NOT return Err on budget exceeded — the caller (pipeline orchestration)
/// decides how to handle that based on path (greedy vs CP-SAT).
pub struct AssignAddressesPass;

impl Pass for AssignAddressesPass {
    fn name(&self) -> &'static str {
        "AssignAddresses"
    }

    fn run(&self, ctx: &mut PipelineContext) -> Result<String, String> {
        let mut assign_config = AddressAssignConfig {
            budget: SCRATCH_BUDGET_HW,
            ..Default::default()
        };

        // Reserve address 0 for zero_vreg
        if let Some(zero) = ctx.zero_vreg {
            assign_config.fixed_vregs.insert(zero, 0);
            assign_config.zero_vreg = Some(zero);
        }

        // Pass scheduling decisions to address assignment
        let select_fma_op_ids: std::collections::HashSet<usize> = ctx
            .final_decisions
            .select_choice_encoding
            .iter()
            .filter(|(_, e)| matches!(e, crate::types::SelectChoiceEncoding::Fma))
            .map(|(id, _)| *id)
            .collect();
        if !select_fma_op_ids.is_empty() {
            assign_config.select_fma_ops = select_fma_op_ids.clone();
        }
        #[cfg(feature = "cpsat")]
        {
            let groups = &ctx.final_decisions.cse_groups;
            let results = &ctx.final_decisions.cse_group_results;
            let select_fma = &select_fma_op_ids;
            if !groups.is_empty() && results.len() != groups.len() {
                eprintln!(
                    "AssignAddresses: CSE block skipped — groups.len() ({}) != results.len() ({}); \
                     skipping CSE shared-sub-temp intervals to avoid data corruption",
                    groups.len(),
                    results.len(),
                );
            }
            if !groups.is_empty() && results.len() == groups.len() {
                let mut cse_consumer_op_ids = std::collections::HashSet::new();
                let mut cse_shared = Vec::new();
                for (g, group) in groups.iter().enumerate() {
                    let res = &results[g];
                    if !res.sub_active {
                        continue;
                    }
                    for &idx in &group.consumer_indices {
                        if idx < ctx.ops.len() {
                            cse_consumer_op_ids.insert(ctx.ops[idx].id);
                        }
                    }
                    let max_fma = group
                        .consumer_indices
                        .iter()
                        .filter(|&&idx| idx < ctx.ops.len() && select_fma.contains(&ctx.ops[idx].id))
                        .map(|&idx| ctx.ops[idx].cycle + 1)
                        .max()
                        .unwrap_or(res.sub_cycle + 1);
                    cse_shared.push((group.sub_temp_vreg, res.sub_cycle, max_fma + 1));
                }
                if !cse_consumer_op_ids.is_empty() {
                    assign_config.cse_consumer_op_ids = cse_consumer_op_ids;
                }
                if !cse_shared.is_empty() {
                    assign_config.cse_shared_sub_temp_intervals = cse_shared;
                }
            }
        }
        if !ctx.final_decisions.shifted_alu_cycles.is_empty() {
            assign_config.shifted_alu_cycles =
                ctx.final_decisions.shifted_alu_cycles.clone();
        }
        {
            let alu_derive: HashMap<usize, _> = ctx
                .final_decisions
                .const_choice_encoding
                .iter()
                .filter_map(|(id, enc)| {
                    if let crate::types::ConstChoiceEncoding::AluDerive(info) = enc {
                        Some((*id, info.clone()))
                    } else {
                        None
                    }
                })
                .collect();
            if !alu_derive.is_empty() {
                assign_config.const_alu_derive_ops = alu_derive;
            }
        }
        if !ctx.vector_group_lanes.is_empty() {
            assign_config.vector_group_lanes = ctx.vector_group_lanes.clone();
        }
        // vreg_def is required by address assignment and may be empty by design; always copy.
        assign_config.vreg_def = ctx.vreg_def.clone();

        let address_result = assign_addresses(&ctx.ops, &assign_config)?;

        // Verify no address conflicts between simultaneously-live VRegs (T2).
        // O(n log n) check; catches bugs in live interval computation or address allocation.
        let conflicts = verify_no_address_conflicts_fast(
            &address_result.live_intervals,
            &address_result.vreg_to_addr,
        );
        if !conflicts.is_empty() {
            eprintln!(
                "ADDRESS CONFLICT DETECTED: {} conflicts found!",
                conflicts.len()
            );
            for (i, c) in conflicts.iter().enumerate().take(20) {
                eprintln!(
                    "  conflict {}: {:?} (addr={}, size={}, interval=[{},{}]) vs {:?} (addr={}, size={}, interval=[{},{}])",
                    i, c.vreg1, c.addr1, c.vreg1.size(), c.interval1.start, c.interval1.end,
                    c.vreg2, c.addr2, c.vreg2.size(), c.interval2.start, c.interval2.end,
                );
            }
            if conflicts.len() > 20 {
                eprintln!("  ... and {} more", conflicts.len() - 20);
            }
            return Err(format!(
                "Address assignment produced {} conflicts between simultaneously-live VRegs",
                conflicts.len()
            ));
        }

        let peak_live = address_result.peak_live;
        let budget_exceeded = address_result.budget_exceeded;

        ctx.final_metrics.peak_live = peak_live;
        ctx.final_metrics.budget_exceeded =
            ctx.final_metrics.budget_exceeded || budget_exceeded;
        ctx.address_result = Some(address_result);

        Ok(format!(
            "peak_live = {} words{}",
            peak_live,
            if budget_exceeded {
                " [BUDGET EXCEEDED]"
            } else {
                ""
            }
        ))
    }
}
