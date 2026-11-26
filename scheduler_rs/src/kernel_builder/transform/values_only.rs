//! Rule `values-only`: the output-contract assumption "the final indices are not observed"
//! (catalogue rule 9), applied to a finished program. Not an identity: the program's output
//! changes, so no proof applies; the obligation is that the run declares
//! `Assumption::IndicesNotObserved` and the validator compares values only.
//!
//! The rule does one thing: the stores that write the indices array stop being roots. A site
//! is a `store` / `vstore` whose address is `mem[layout.indices_word] + k` with the stored
//! words inside the batch (from the ops: a load of the header word plus a constant), and no
//! later load may read those words back (another observer, inside the program). Everything
//! else, which ops die, is dead code elimination's (`Surgery::finish` removes exactly what
//! the removal made dead): the index address add; the mirrored form's final `L_d - n` SUB
//! and its `L_d` constant (C4); the index chain after the last op that reads it, which
//! depends on whether the last rounds' tree access reads the index (C3); never a branch bit
//! that a later bit-driven access reads (C5).
//!
//! The saving is not additive per site: an op that several groups' index chains share (the
//! header pointer load, a shared constant) dies only when every store is gone. The report
//! gives each site's delta when it alone is removed, and the shared remainder.

use std::collections::BTreeMap;

use crate::types::{Engine, Opcode, SsaOp};

use super::super::emit_ssa::eliminate_ssa_dead_code;
use super::super::SsaProgram;
use super::graph::SsaGraph;
use super::report::{counts_of, dce_counts, sub, RuleReport};
use super::rule::{Fact, FactSource, Rejection};
use super::surgery::{Finished, Surgery};
use super::values::{Address, Assumption, MemoryLayout, ValueFacts};

/// Rule id.
pub const VALUES_ONLY_RULE: &str = "values-only";

/// The values-only rule.
#[derive(Clone, Debug)]
pub struct ValuesOnlyRule {
    /// Where the indices are.
    pub layout: MemoryLayout,
    /// Declared assumptions; without `IndicesNotObserved` every site is refused.
    pub assumptions: Vec<Assumption>,
}

impl ValuesOnlyRule {
    /// The rule for the problem's layout, with the output contract declared.
    ///
    /// # Arguments
    /// * `batch_size` - Lanes of the problem.
    ///
    /// # Returns
    /// The rule, assuming `IndicesNotObserved`.
    pub fn for_problem(batch_size: usize) -> Self {
        ValuesOnlyRule { layout: MemoryLayout::problem(batch_size), assumptions: vec![Assumption::IndicesNotObserved] }
    }
}

/// A matched index store.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexStoreSite {
    /// The store.
    pub store: usize,
    /// Its address.
    pub address: Address,
    /// Words stored.
    pub width: i64,
    /// Facts the match relies on.
    pub facts: Vec<Fact>,
}

impl IndexStoreSite {
    /// The group of lanes it stores (`offset / 8`), from the address.
    pub fn group(&self) -> usize {
        (self.address.offset / 8) as usize
    }
}

fn width(op: &SsaOp) -> i64 {
    if matches!(op.opcode, Opcode::Vstore | Opcode::Vload) {
        8
    } else {
        1
    }
}

impl ValuesOnlyRule {
    /// Match a site at op `i`.
    ///
    /// # Arguments
    /// * `graph` - The program.
    /// * `values` - Its value facts (for constant offsets).
    /// * `i` - Candidate op (a store).
    ///
    /// # Returns
    /// The site, or why op `i` is not one.
    pub fn match_at(&self, graph: &SsaGraph, values: &ValueFacts, i: usize) -> Result<IndexStoreSite, Rejection> {
        let reject = |s: String| Err(Rejection(s));
        let op = &graph.ops[i];
        if !op.writes_memory() {
            return reject("not a store".into());
        }
        if op.keep_live.is_some() {
            return reject("store keeps other values live (keep_live)".into());
        }
        let w = width(op);
        let Some(address) = op.srcs.first().and_then(|s| s.as_vreg()).and_then(|a| values.address(graph, a)) else {
            return reject("store address is not a header pointer plus a constant".into());
        };
        if !self.layout.in_indices(address, w) {
            return reject(format!("store writes mem[mem[{}] + {}], not the indices", address.word, address.offset));
        }
        if !self.assumptions.contains(&Assumption::IndicesNotObserved) {
            return reject("output contract: the indices are observed (indices-not-observed is not declared)".into());
        }
        // Another observer: a later load that may read the stored words back.
        for (j, later) in graph.ops.iter().enumerate().skip(i + 1) {
            if !later.reads_memory() {
                continue;
            }
            // Disjoint only if both accesses are proved inside their own arrays
            // (`MemoryLayout::may_overlap`); a different header word alone proves nothing.
            // A load_offset reads at an address plus an offset the analysis does not model:
            // unknown, so it may alias.
            let a = if later.opcode == Opcode::LoadOffset {
                None
            } else {
                later.srcs.first().and_then(|s| s.as_vreg()).and_then(|a| values.address(graph, a))
            };
            if self.layout.may_overlap(Some(address), w, a, width(later)) {
                return reject(format!("op {} ({:?}) may read the stored indices back", j, later.opcode));
            }
        }
        Ok(IndexStoreSite {
            store: i,
            address,
            width: w,
            facts: vec![
                Fact {
                    name: "index_store",
                    value: format!("stores mem[mem[{}] + {}..+{}]", address.word, address.offset, w),
                    source: FactSource::Structure,
                },
                Fact { name: "no_reader", value: "no later load reads those words".into(), source: FactSource::Structure },
                self.layout.fact(),
                Assumption::IndicesNotObserved.fact(),
            ],
        })
    }

    /// Every site of `graph` in op order, and rejected stores counted by reason.
    ///
    /// # Arguments
    /// * `graph` - The program.
    /// * `zero` - The program's zero register.
    ///
    /// # Returns
    /// `(sites, rejections)`.
    pub fn find_sites(&self, graph: &SsaGraph, zero: Option<crate::types::VReg>) -> (Vec<IndexStoreSite>, BTreeMap<String, usize>) {
        let values = ValueFacts::compute(graph, zero, &self.layout, &[]);
        let mut sites = Vec::new();
        let mut rejected: BTreeMap<String, usize> = BTreeMap::new();
        for i in 0..graph.ops.len() {
            if !graph.ops[i].writes_memory() {
                continue;
            }
            match self.match_at(graph, &values, i) {
                Ok(s) => sites.push(s),
                Err(e) => *rejected.entry(e.0).or_default() += 1,
            }
        }
        (sites, rejected)
    }
}

/// Counts of the ops DCE keeps when the stores `removed` are gone.
fn dce_counts_without(program: &SsaProgram, removed: &[usize]) -> Result<crate::kernel_builder::transform::cost::Counts, String> {
    let ops: Vec<SsaOp> =
        program.ops.iter().enumerate().filter(|(i, _)| !removed.contains(i)).map(|(_, o)| o.clone()).collect();
    let mut vreg_def = std::collections::HashMap::new();
    for op in &ops {
        if let Some(d) = op.dest {
            vreg_def.entry(d).or_insert(op.id);
        }
    }
    let mut legacy = Vec::new();
    let (kept, _, _) = eliminate_ssa_dead_code(ops, &mut vreg_def, &mut legacy, program.zero_vreg, Some(&program.vector_group_lanes));
    counts_of(&kept)
}

/// Declare the indices unobserved: remove every index store of `program` chosen by `select`
/// and what that made dead.
///
/// # Arguments
/// * `program` - A finished program (before dependency computation).
/// * `rule` - The rule (layout and declared assumptions).
/// * `select` - Which matched stores to remove.
///
/// # Returns
/// The rewritten program and the report. `site_classes` holds, per site, the delta when that
/// store alone is removed (the store and the ops only its chain needs), keyed `"index-store"`;
/// `cascade_delta` is what dies only when all selected stores are gone (shared ops).
pub fn apply_values_only(
    program: SsaProgram,
    rule: &ValuesOnlyRule,
    select: impl Fn(&IndexStoreSite) -> bool,
) -> Result<(SsaProgram, RuleReport), String> {
    let (sites, rejections) = {
        let graph = SsaGraph::new(&program.ops, &program.vector_group_lanes);
        rule.find_sites(&graph, program.zero_vreg)
    };
    let sites: Vec<IndexStoreSite> = sites.into_iter().filter(|s| select(s)).collect();
    let mut report = RuleReport { rule: VALUES_ONLY_RULE, rejections, ..Default::default() };
    if sites.is_empty() {
        return Ok((program, report));
    }
    // A program the surgery cannot edit safely is left as it is, with the reason.
    if let Err(e) = Surgery::check(&program) {
        report.refused = Some(e);
        return Ok((program, report));
    }
    // Per-site deltas, each store alone (exact for that site; not additive). Cost accounting
    // never blocks the rewrite.
    let before = dce_counts(&program).map_err(|e| report.cost_error = Some(e)).ok();
    for s in &sites {
        *report.sites.entry("index-store").or_default() += 1;
        report.add_facts(&s.facts);
        if let Some(before) = &before {
            match dce_counts_without(&program, &[s.store]) {
                Ok(alone) => *report.site_classes.entry(("index-store", sub(&alone, before))).or_default() += 1,
                Err(e) => report.cost_error = Some(e),
            }
        }
    }
    let mut surgery = Surgery::new(program)?;
    for s in &sites {
        debug_assert_eq!(surgery.ops()[s.store].engine, Engine::Store);
        surgery.remove(s.store);
    }
    let (program, surgery_report) = match surgery.finish_or_refuse()? {
        Finished::Done(program, report) => (program, report),
        Finished::Refused(program, why) => {
            report.refused = Some(why);
            // Nothing was rewritten, so nothing relied on the sites' facts (assumptions).
            report.facts.clear();
            report.sites = Default::default();
            report.site_classes = Default::default();
            return Ok((program, report));
        }
    };
    report.surgery = surgery_report;
    if let (Some(before), None) = (before, &report.cost_error) {
        match dce_counts(&program) {
            Ok(after) => report.close(&before, &after),
            Err(e) => report.cost_error = Some(e),
        }
    }
    Ok((program, report))
}
