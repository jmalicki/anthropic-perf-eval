//! SSA lowering pass.
//!
//! Converts SSA ops to concrete `Op` instructions, expanding shifted
//! VALU ops to 8 scalar ALU ops and resolving ConstChoice/SelectChoice
//! encoding decisions.

use crate::kernel_builder::pipeline::{Pass, PipelineContext};
use crate::kernel_builder::ssa_lower::{lower_ssa_ops_with_shift, LoweringContext};

/// Lowers SSA ops to concrete `Op` instructions.
///
/// Reads scheduling decisions to determine how each op is lowered:
/// - Shifted VALU → 8 ALU ops
/// - ConstChoice AddImm → flow slot add_imm
/// - ConstChoice Load → load slot const
/// - ConstChoice AluDerive → ALU derivation
/// - SelectChoice FMA → sub + multiply_add
/// - SelectChoice vselect → flow slot vselect
pub struct LowerToOpsPass;

impl Pass for LowerToOpsPass {
    fn name(&self) -> &'static str {
        "LowerToOps"
    }

    fn run(&self, ctx: &mut PipelineContext) -> Result<String, String> {
        let address_result = ctx
            .address_result
            .as_ref()
            .ok_or_else(|| "LowerToOpsPass requires address assignment result".to_string())?;

        let decisions = &ctx.final_decisions;

        let ctx_lowering = LoweringContext {
            ssa_ops: &ctx.ops,
            vreg_to_addr: &address_result.vreg_to_addr,
            shifted_ops: &decisions.shifted_ops,
            const_add_imm_ops: &decisions.const_add_imm_ops,
            const_alu_derive_ops: &decisions.const_alu_derive_ops,
            select_fma_ops: &decisions.select_fma_ops,
            shifted_alu_cycles: &decisions.shifted_alu_cycles,
            eliminated_vbroadcast_ops: &decisions.eliminated_vbroadcast_ops,
            eliminated_vb_to_scalar_producer: &decisions.eliminated_vb_to_scalar_producer,
            #[cfg(feature = "cpsat")]
            cse_groups: (!decisions.cse_groups.is_empty()).then(|| decisions.cse_groups.as_slice()),
            #[cfg(not(feature = "cpsat"))]
            cse_groups: None,
            #[cfg(feature = "cpsat")]
            cse_group_results: (!decisions.cse_group_results.is_empty())
                .then(|| decisions.cse_group_results.as_slice()),
            #[cfg(not(feature = "cpsat"))]
            cse_group_results: None,
        };
        let ops = lower_ssa_ops_with_shift(&ctx_lowering)?;

        let n_ops = ops.len();
        ctx.lowered_ops = Some(ops);

        Ok(format!("{} concrete ops", n_ops))
    }
}
