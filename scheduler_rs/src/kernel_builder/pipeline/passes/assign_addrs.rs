//! Address assignment pass.
//!
//! Assigns concrete scratch addresses to VRegs via linear scan allocation.

use crate::address_assign::{assign_addresses, verify_no_address_conflicts, AddressAssignConfig};
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
        if !ctx.final_decisions.select_fma_ops.is_empty() {
            assign_config.select_fma_ops = Some(ctx.final_decisions.select_fma_ops.clone());
        }
        if !ctx.final_decisions.shifted_alu_cycles.is_empty() {
            assign_config.shifted_alu_cycles =
                Some(ctx.final_decisions.shifted_alu_cycles.clone());
        }
        if !ctx.final_decisions.const_alu_derive_ops.is_empty() {
            assign_config.const_alu_derive_ops =
                Some(ctx.final_decisions.const_alu_derive_ops.clone());
        }
        if !ctx.vector_group_lanes.is_empty() {
            assign_config.vector_group_lanes = Some(ctx.vector_group_lanes.clone());
        }
        assign_config.vreg_def = Some(ctx.vreg_def.clone());

        let address_result = assign_addresses(&ctx.ops, &assign_config);

        // Verify no address conflicts between simultaneously-live VRegs.
        // This catches bugs in live interval computation or address allocation.
        let conflicts = verify_no_address_conflicts(
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
