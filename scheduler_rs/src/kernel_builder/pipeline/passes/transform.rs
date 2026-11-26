//! Transform pass: apply rewrite rules to the finished SSA program.
//!
//! Runs first, before `ComputeDependenciesPass`, on the program exactly as emitted (before
//! dead code elimination). Each rule in `CompilerConfig::transforms` reads only the ops, never
//! emitter state (see `kernel_builder::transform`). With no transforms configured it does
//! nothing, so default output is unchanged.
//!
//! Rules apply in the order given. `prune-index` and `select-bits-path` declare the input
//! precondition `InitialIndicesZero` and `values-only` the output contract `IndicesNotObserved`
//! (`transform::values::Assumption`).
//!
//! Op ids are renumbered by a transform, so ids that refer to the program (for example the
//! CP-SAT `forced_shifts`) mean ops of the transformed program, as the scheduler JSON of a
//! run with the same transforms lists them; `build_and_schedule_cpsat` checks they name
//! vector ops of it.

use crate::kernel_builder::config::TransformKind;
use crate::kernel_builder::pipeline::{Pass, PipelineContext};
use crate::kernel_builder::transform::{
    apply_hash_merge, apply_prune_index, apply_select_bits, apply_values_only, HashMergeRule, PruneIndexRule,
    SelectBitsRule, ValuesOnlyRule,
};
use crate::kernel_builder::transform::MemoryLayout;
use crate::kernel_builder::SsaProgram;

/// What one transform did.
pub struct TransformOutcome {
    /// One line for the log.
    pub summary: String,
    /// Sites rewritten.
    pub sites: usize,
    /// Why the program was left unchanged although sites matched, if it was.
    pub refused: Option<String>,
}

/// Apply one transform to a program.
///
/// # Arguments
/// * `program` - The program.
/// * `t` - The transform.
/// * `batch_size` - Batch size for the memory layout (`prune-index`, `values-only`).
///
/// # Returns
/// The rewritten program (unchanged when nothing matched or the surgery refused it) and the
/// outcome; `Err` only for a missing batch size or an inconsistency the surgery found.
pub fn apply_transform(
    program: SsaProgram,
    t: TransformKind,
    batch_size: Option<usize>,
) -> Result<(SsaProgram, TransformOutcome), String> {
    apply_transform_with(program, t, batch_size, None)
}

/// `apply_transform` with the tree's size known (`MemoryLayout::n_nodes`), as the pass runs it.
///
/// # Arguments
/// * `program`, `t`, `batch_size` - As `apply_transform`.
/// * `n_nodes` - Words in the tree, if known.
pub fn apply_transform_with(
    program: SsaProgram,
    t: TransformKind,
    batch_size: Option<usize>,
    n_nodes: Option<usize>,
) -> Result<(SsaProgram, TransformOutcome), String> {
    let layout = || -> Result<MemoryLayout, String> {
        let b = batch_size.ok_or_else(|| {
            format!("transform {} needs the batch size (CompilerConfig::transform_batch_size)", t.as_str())
        })?;
        Ok(match n_nodes {
            Some(n) => MemoryLayout::problem_with_tree(n, b),
            None => MemoryLayout::problem(b),
        })
    };
    Ok(match t {
        TransformKind::HashMerge => {
            let (p, r) = apply_hash_merge(program, &HashMergeRule::default(), |_| true)?;
            (p, TransformOutcome { summary: r.summary(), sites: r.sites, refused: r.refused.clone() })
        }
        TransformKind::PruneIndex => {
            let (p, r) = apply_prune_index(program, &PruneIndexRule { layout: layout()?, ..PruneIndexRule::for_problem(0) }, |_| true)?;
            let sites = if r.refused.is_some() { 0 } else { r.n_sites() };
            (p, TransformOutcome { summary: r.summary(), sites, refused: r.refused.clone() })
        }
        TransformKind::ValuesOnly => {
            let (p, r) = apply_values_only(program, &ValuesOnlyRule { layout: layout()?, ..ValuesOnlyRule::for_problem(0) }, |_| true)?;
            let sites = if r.refused.is_some() { 0 } else { r.n_sites() };
            (p, TransformOutcome { summary: r.summary(), sites, refused: r.refused.clone() })
        }
        TransformKind::SelectBitsPath => {
            let (p, r) = apply_select_bits(program, &SelectBitsRule { layout: layout()?, ..SelectBitsRule::for_problem(0) }, |_| true)?;
            let sites = if r.refused.is_some() { 0 } else { r.n_sites() };
            (p, TransformOutcome { summary: r.summary(), sites, refused: r.refused.clone() })
        }
    })
}

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
        let batch_size = ctx.config.compiler.transform_batch_size;
        let n_nodes = ctx.config.compiler.transform_n_nodes;
        let mut summaries = Vec::new();
        for t in transforms {
            // A copy: on an error the context keeps the program as it was before this
            // transform (the earlier transforms' result), for the error export.
            let program = SsaProgram {
                ops: ctx.ops.clone(),
                vreg_def: ctx.vreg_def.clone(),
                vreg_count: ctx.vreg_count,
                const_alternatives: ctx.const_alternatives.clone(),
                zero_vreg: ctx.zero_vreg,
                vector_group_lanes: ctx.vector_group_lanes.clone(),
            };
            let (program, outcome) = apply_transform_with(program, t, batch_size, n_nodes)?;
            if outcome.sites == 0 {
                // Always shown: a requested rewrite that does nothing is usually a configuration
                // mistake (the flag it stands for is also set, or, for prune-index, the mirrored
                // index form, which has no zero-based update to prune).
                eprintln!(
                    "warning: --transform {} rewrote 0 sites{}",
                    t.as_str(),
                    match (&outcome.refused, t) {
                        (Some(r), _) => format!(" (program refused: {})", r),
                        (None, TransformKind::HashMerge) => " (already merged, e.g. with --merge-hash-stages?)".into(),
                        (None, TransformKind::PruneIndex) => " (mirrored index or --prune-dead-index?)".into(),
                        (None, TransformKind::ValuesOnly) => " (--values-only already set?)".into(),
                        (None, TransformKind::SelectBitsPath) => {
                            " (no bit-driven access at depth 3 or more, or --select-bit-source path already set?)".into()
                        }
                    }
                );
            }
            ctx.transform_sites.push((t.as_str().to_string(), outcome.sites));
            let summary = outcome.summary;
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
