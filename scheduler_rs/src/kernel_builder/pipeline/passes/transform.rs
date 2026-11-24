//! Transform pass: apply rewrite rules to the finished SSA program.
//!
//! Runs first, before `ComputeDependenciesPass`, on the program exactly as emitted (before
//! dead code elimination). Each rule in `CompilerConfig::transforms` reads only the ops, never
//! emitter state (see `kernel_builder::transform`). With no transforms configured it does
//! nothing, so default output is unchanged.

use crate::kernel_builder::config::TransformKind;
use crate::kernel_builder::pipeline::{Pass, PipelineContext};
use crate::kernel_builder::transform::{apply_hash_merge, HashMergeRule};
use crate::kernel_builder::SsaProgram;

/// Applies `CompilerConfig::transforms` in order.
pub struct TransformPass;

impl Pass for TransformPass {
    fn name(&self) -> &'static str {
        "Transform"
    }

    fn run(&self, ctx: &mut PipelineContext) -> Result<String, String> {
        let transforms = ctx.config.compiler.transforms.clone();
        if transforms.is_empty() {
            return Ok("none".into());
        }
        let mut summaries = Vec::new();
        for t in transforms {
            let program = SsaProgram {
                ops: std::mem::take(&mut ctx.ops),
                vreg_def: std::mem::take(&mut ctx.vreg_def),
                vreg_count: ctx.vreg_count,
                const_alternatives: std::mem::take(&mut ctx.const_alternatives),
                zero_vreg: ctx.zero_vreg,
                vector_group_lanes: std::mem::take(&mut ctx.vector_group_lanes),
            };
            let (program, summary) = match t {
                TransformKind::HashMerge => {
                    let (p, report) = apply_hash_merge(program, &HashMergeRule::default(), |_| true)?;
                    (p, report.summary())
                }
            };
            ctx.ops = program.ops;
            ctx.vreg_def = program.vreg_def;
            ctx.vreg_count = program.vreg_count;
            ctx.const_alternatives = program.const_alternatives;
            ctx.zero_vreg = program.zero_vreg;
            ctx.vector_group_lanes = program.vector_group_lanes;
            if ctx.config.verbose || ctx.config.greedy_verbose {
                eprintln!("[transform] {}", summary);
            }
            summaries.push(summary);
        }
        Ok(summaries.join("; "))
    }
}
