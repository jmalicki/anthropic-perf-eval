//! Dead code elimination pass.
//!
//! Removes SSA ops whose results have no users, saving cycles,
//! optimizer time, and scratch space.

use crate::kernel_builder::emit_ssa::eliminate_ssa_dead_code;
use crate::kernel_builder::pipeline::{Pass, PipelineContext};

/// Eliminates dead SSA ops (ops whose results are never read).
///
/// After DCE, dependencies are recomputed (the caller should run
/// `ComputeDependenciesPass` again).
pub struct DcePass;

impl Pass for DcePass {
    fn name(&self) -> &'static str {
        "DCE"
    }

    fn run(&self, ctx: &mut PipelineContext) -> Result<String, String> {
        let before = ctx.ops.len();
        let (new_ops, new_vreg_def, new_const_alternatives) = eliminate_ssa_dead_code(
            std::mem::take(&mut ctx.ops),
            &mut ctx.vreg_def,
            &mut ctx.const_alternatives,
            ctx.zero_vreg,
        );
        ctx.ops = new_ops;
        ctx.vreg_def = new_vreg_def;
        ctx.const_alternatives = new_const_alternatives;
        let after = ctx.ops.len();
        Ok(format!("{} → {} ops (removed {})", before, after, before - after))
    }
}
