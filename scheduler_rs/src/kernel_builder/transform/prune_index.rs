//! Rule `prune-index`: what `--prune-dead-index` emits, as rewrites of the finished program
//! (catalogue rules 5, 6 and 7). Reads no emitter state.
//!
//! * Rule 5, `shared-constant`: a `vbroadcast(s)` of a scalar `s` that holds a known constant,
//!   when an earlier `vbroadcast(s)` of the same scalar exists, is replaced by that earlier
//!   one (CSE of identical ops on identical operands; `kernel_dataflow` hash-conses them into
//!   one term). In the emitted programs these are the wrap round's per-group
//!   `vbroadcast(zero)`. Broadcasts of non-constant scalars (the depth-0 root, kept private by
//!   the emitter on purpose) are not this rule's sites.
//! * Rule 6, `zero-update`: `y = multiply_add(a, m, c)` where a factor is provably 0 in every
//!   lane becomes `y = c` (readers of `y` read `c`; the op is removed). Identity
//!   `0 * m + c = c` for all `m, c`: `R12-fma-zero-factor`; its instance on the index update
//!   is `R12-index-update-from-root`, and `R12-index-update-from-root-needs-zero` shows the
//!   fact is needed. "Provably 0" comes from `values::ValueFacts`: after the wrap round the
//!   index is a broadcast of the zero register (constant propagation alone); in round 0 it is
//!   the load of the input indices, which is 0 only under the declared input precondition
//!   `Assumption::InitialIndicesZero` (refused without it, with the reason named).
//! * Rule 7: the initial index load and its address add are not rewritten by anything. They
//!   go when the surgery removes what the rewrite made dead, i.e. exactly when no reader is
//!   left (coupling C6: a round-0 access that reads the index keeps them).
//!
//! Under the mirrored index (`--index-form mirrored`) there is no multiply_add with a zero
//! index and no wrap broadcast, so the rule finds no sites (coupling C2).

use std::collections::BTreeMap;

use crate::types::{Engine, Opcode, RegionKind, VReg};

use super::super::SsaProgram;
use super::graph::SsaGraph;
use super::hash_merge::ProposalSource;
use super::report::{counts_of, dce_counts, sub, RuleReport};
use super::rule::{AlternativeSpec, Fact, FactSource, Proposal, Rejection, Rule};
use super::surgery::Surgery;
use super::values::{Assumption, MemoryLayout, ValueFacts};

/// Rule id.
pub const PRUNE_INDEX_RULE: &str = "prune-index";

/// Alternatives of a site (rewritten or not). A rule-5 site (`shared-constant`) is plain CSE
/// and takes `shared`, justified by `CSE-identical-broadcast`; only rule-6 sites
/// (`zero-update`) take `pruned` and the R12 proofs.
pub const PRUNE_INDEX_ALTERNATIVES: [AlternativeSpec; 3] = [
    AlternativeSpec { name: "kept", proofs: &[] },
    AlternativeSpec {
        name: "pruned",
        proofs: &["R12-fma-zero-factor", "R12-index-update-from-root", "R12-index-update-from-root-needs-zero"],
    },
    AlternativeSpec { name: "shared", proofs: &["CSE-identical-broadcast"] },
];

/// The prune-index rule.
#[derive(Clone, Debug)]
pub struct PruneIndexRule {
    /// Where to look: every broadcast and multiply_add, or only those in `index_step` regions.
    pub source: ProposalSource,
    /// Where the input indices are.
    pub layout: MemoryLayout,
    /// Declared assumptions (`InitialIndicesZero` lets round 0's update fold).
    pub assumptions: Vec<Assumption>,
    /// The program's zero register, for `Rule::match_site` (which gets only the graph);
    /// `find_sites` takes it as an argument and falls back to this.
    pub zero: Option<VReg>,
}

impl PruneIndexRule {
    /// The same rule for a program whose zero register is `zero`, so `match_site` sees the
    /// zeros `find_sites` sees (the wrap round's broadcasts of it).
    pub fn with_zero(mut self, zero: Option<VReg>) -> Self {
        self.zero = zero;
        self
    }

    /// The value facts of `graph`, as every entry point computes them.
    fn analyses(&self, graph: &SsaGraph, zero: Option<VReg>) -> ValueFacts {
        ValueFacts::compute(graph, zero, &self.layout, &self.assumptions)
    }

    /// The rule for the problem's layout, with the input precondition declared.
    ///
    /// # Arguments
    /// * `batch_size` - Lanes of the problem.
    ///
    /// # Returns
    /// The rule proposing from the ops, assuming `InitialIndicesZero`.
    pub fn for_problem(batch_size: usize) -> Self {
        PruneIndexRule {
            source: ProposalSource::Ops,
            layout: MemoryLayout::problem(batch_size),
            assumptions: vec![Assumption::InitialIndicesZero],
            zero: None,
        }
    }
}

/// Kind of a site.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum PruneKind {
    /// Rule 5: a duplicate broadcast of a constant scalar.
    SharedConstant,
    /// Rule 6: a multiply_add with a zero factor.
    ZeroUpdate,
}

impl PruneKind {
    /// The alternative a rewritten site of this kind takes (`PRUNE_INDEX_ALTERNATIVES`), with
    /// its proofs.
    pub fn alternative(&self) -> &'static AlternativeSpec {
        match self {
            PruneKind::SharedConstant => &PRUNE_INDEX_ALTERNATIVES[2],
            PruneKind::ZeroUpdate => &PRUNE_INDEX_ALTERNATIVES[1],
        }
    }

    /// Name in reports.
    pub fn as_str(&self) -> &'static str {
        match self {
            PruneKind::SharedConstant => "shared-constant",
            PruneKind::ZeroUpdate => "zero-update",
        }
    }
}

/// A matched site: one op to remove, and the value its readers read instead.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PruneIndexSite {
    /// Which rule.
    pub kind: PruneKind,
    /// The op removed.
    pub op: usize,
    /// Its result.
    pub output: VReg,
    /// What readers of `output` read instead (the earlier broadcast; the addend).
    pub replacement: VReg,
    /// Rule 6: the zero factor.
    pub zero: Option<VReg>,
    /// Round label (tag), if any.
    pub round: Option<u32>,
    /// Group label (tag), if any.
    pub group: Option<u32>,
    /// Facts the match relies on.
    pub facts: Vec<Fact>,
}

impl PruneIndexRule {
    /// Match a site at op `i`.
    ///
    /// # Arguments
    /// * `graph` - The program.
    /// * `values` - Its value facts (`ValueFacts::compute` with this rule's layout and
    ///   assumptions).
    /// * `i` - Candidate op (a `vbroadcast` or a vector `multiply_add`).
    ///
    /// # Returns
    /// The site, or why op `i` is not one.
    pub fn match_at(&self, graph: &SsaGraph, values: &ValueFacts, i: usize) -> Result<PruneIndexSite, Rejection> {
        let reject = |s: String| Err(Rejection(s));
        let op = &graph.ops[i];
        let Some(output) = op.dest else { return reject("op without a dest".into()) };
        if graph.group_lanes(output).is_some() || graph.element_of(output).is_some() {
            return reject("result is a vector group (surgery cannot rewire lanes)".into());
        }
        if op.trace_id.is_some() {
            return reject("op carries a trace id".into());
        }
        if op.keep_live.is_some() {
            return reject("op keeps other values live (keep_live)".into());
        }
        if graph.is_redefined(output) {
            return reject("result is defined more than once".into());
        }
        let label = |f: fn(&crate::types::RegionTag) -> Option<u32>| op.region.and_then(|t| f(&t));
        let (round, group) = (label(|t| t.round), label(|t| t.group));
        match (op.engine, op.opcode) {
            (Engine::Valu, Opcode::Vbroadcast) => {
                let Some(s) = op.srcs.first().and_then(|s| s.as_vreg()) else {
                    return reject("broadcast of an immediate".into());
                };
                let Some(value) = values.value(graph, s) else {
                    return reject("broadcast of a non-constant scalar (not this rule)".into());
                };
                let first = graph.ops[..i].iter().position(|o| {
                    o.engine == Engine::Valu && o.opcode == Opcode::Vbroadcast && o.srcs.first().and_then(|x| x.as_vreg()) == Some(s)
                });
                let Some(first) = first else { return reject("first broadcast of its scalar".into()) };
                let replacement = graph.ops[first].dest.expect("broadcast has a dest");
                Ok(PruneIndexSite {
                    kind: PruneKind::SharedConstant,
                    op: i,
                    output,
                    replacement,
                    zero: None,
                    round,
                    group,
                    facts: vec![
                        Fact {
                            name: "same_op",
                            value: format!("vbroadcast of the same scalar (constant {}) as op {}", value, first),
                            source: FactSource::Structure,
                        },
                        Fact { name: "proofs", value: PruneKind::SharedConstant.alternative().proofs.join(", "), source: FactSource::Structure },
                    ],
                })
            }
            (Engine::Valu, Opcode::MultiplyAdd) => {
                let srcs: Option<Vec<VReg>> = op.srcs.iter().map(|s| s.as_vreg()).collect();
                let Some(srcs) = srcs.filter(|s| s.len() == 3) else {
                    return reject("multiply_add operands are not three VRegs".into());
                };
                let zero = [srcs[0], srcs[1]].into_iter().find(|&f| values.value(graph, f) == Some(0));
                let Some(zero) = zero else {
                    return reject(format!(
                        "no factor is provably 0 ({})",
                        values.why_not_zero(graph, srcs[0], &self.layout, &self.assumptions)
                    ));
                };
                let mut facts = vec![Fact {
                    name: "zero_factor",
                    value: if values.relies_on_assumption(graph, zero) {
                        "factor is the load of the input indices".into()
                    } else {
                        "factor is a constant 0 by propagation".into()
                    },
                    source: FactSource::Structure,
                }];
                facts.push(Fact { name: "proofs", value: PruneKind::ZeroUpdate.alternative().proofs.join(", "), source: FactSource::Structure });
                if values.relies_on_assumption(graph, zero) {
                    facts.push(self.layout.fact());
                    facts.push(Assumption::InitialIndicesZero.fact());
                }
                Ok(PruneIndexSite { kind: PruneKind::ZeroUpdate, op: i, output, replacement: srcs[2], zero: Some(zero), round, group, facts })
            }
            _ => reject("not a vbroadcast or a vector multiply_add".into()),
        }
    }

    /// Every site of `graph` in op order, and rejected candidates counted by reason.
    ///
    /// # Arguments
    /// * `graph` - The program.
    /// * `zero` - The program's zero register.
    ///
    /// # Returns
    /// `(sites, rejections)`.
    pub fn find_sites(&self, graph: &SsaGraph, zero: Option<VReg>) -> (Vec<PruneIndexSite>, BTreeMap<String, usize>) {
        let values = self.analyses(graph, zero.or(self.zero));
        let mut sites = Vec::new();
        let mut rejected: BTreeMap<String, usize> = BTreeMap::new();
        for p in self.propose(graph) {
            for &i in &p.ops {
                match self.match_at(graph, &values, i) {
                    Ok(s) => sites.push(s),
                    Err(e) => *rejected.entry(e.0).or_default() += 1,
                }
            }
        }
        sites.sort_by_key(|s| s.op);
        sites.dedup_by_key(|s| s.op);
        (sites, rejected)
    }
}

impl Rule for PruneIndexRule {
    type Site = PruneIndexSite;

    fn id(&self) -> &'static str {
        PRUNE_INDEX_RULE
    }

    fn alternatives(&self) -> &'static [AlternativeSpec] {
        &PRUNE_INDEX_ALTERNATIVES
    }

    /// One proposal per vector `multiply_add` and per `vbroadcast` (only inside `index_step`
    /// regions with `ProposalSource::Tags`), in op order.
    fn propose(&self, graph: &SsaGraph) -> Vec<Proposal> {
        graph
            .ops
            .iter()
            .enumerate()
            .filter(|(_, op)| op.engine == Engine::Valu && matches!(op.opcode, Opcode::MultiplyAdd | Opcode::Vbroadcast))
            .filter(|(_, op)| match self.source {
                ProposalSource::Ops => true,
                ProposalSource::Tags => op.region.map_or(false, |t| t.kind == RegionKind::IndexStep),
            })
            .map(|(i, op)| Proposal {
                region_id: op.region.map(|t| t.id),
                round: op.region.and_then(|t| t.round),
                group: op.region.and_then(|t| t.group),
                depth: op.region.and_then(|t| t.depth),
                ops: vec![i],
            })
            .collect()
    }

    /// Computes the value facts for every call (with `self.zero`, as `find_sites` does); use
    /// `find_sites` for a whole program.
    fn match_site(&self, graph: &SsaGraph, proposal: &Proposal) -> Result<PruneIndexSite, Rejection> {
        let values = self.analyses(graph, self.zero);
        let mut last = Rejection("empty proposal".into());
        for &i in &proposal.ops {
            match self.match_at(graph, &values, i) {
                Ok(s) => return Ok(s),
                Err(e) => last = e,
            }
        }
        Err(last)
    }
}

/// Rewrite every prune-index site of `program` chosen by `select`.
///
/// # Arguments
/// * `program` - A finished program (before dependency computation).
/// * `rule` - The rule.
/// * `select` - Which matched sites to rewrite.
///
/// # Returns
/// The rewritten program (what the rewrite made dead is removed, e.g. the initial index
/// load) and the report: per-site deltas by kind, and the cascade (rule 7 and the shared
/// zero broadcast) paid once.
pub fn apply_prune_index(
    program: SsaProgram,
    rule: &PruneIndexRule,
    select: impl Fn(&PruneIndexSite) -> bool,
) -> Result<(SsaProgram, RuleReport), String> {
    let (sites, rejections) = {
        let graph = SsaGraph::new(&program.ops, &program.vector_group_lanes);
        rule.find_sites(&graph, program.zero_vreg)
    };
    let sites: Vec<PruneIndexSite> = sites.into_iter().filter(|s| select(s)).collect();
    let mut report = RuleReport { rule: PRUNE_INDEX_RULE, rejections, ..Default::default() };
    if sites.is_empty() {
        return Ok((program, report));
    }
    // A program the surgery cannot edit safely is left as it is, with the reason.
    if let Err(e) = Surgery::check(&program) {
        report.refused = Some(e);
        return Ok((program, report));
    }
    // Cost accounting never blocks the rewrite.
    let before = dce_counts(&program).map_err(|e| report.cost_error = Some(e)).ok();
    let mut surgery = Surgery::new(program)?;
    for s in &sites {
        *report.sites.entry(s.kind.as_str()).or_default() += 1;
        report.add_facts(&s.facts);
        match counts_of([&surgery.ops()[s.op]]) {
            Ok(removed) => *report.site_classes.entry((s.kind.as_str(), sub(&[0; 10], &removed))).or_default() += 1,
            Err(e) => report.cost_error = Some(e),
        }
        surgery.remove(s.op);
        surgery.replace_uses(s.output, s.replacement);
    }
    let (program, surgery_report) = surgery.finish()?;
    report.surgery = surgery_report;
    if let (Some(before), None) = (before, &report.cost_error) {
        match dce_counts(&program) {
            Ok(after) => report.close(&before, &after),
            Err(e) => report.cost_error = Some(e),
        }
    }
    Ok((program, report))
}
