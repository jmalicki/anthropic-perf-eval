//! Address assignment pass.
//!
//! Assigns concrete scratch addresses to VRegs via linear scan allocation.

use crate::address_assign::{assign_addresses, AddressAssignConfig};
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

        let address_result = assign_addresses(&ctx.ops, &assign_config);

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
