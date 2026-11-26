//! Cost reports of rewrites, computed from the rewrite itself: per-site deltas in the ten
//! categories of `cost::CATEGORIES`, and the whole-program remainder (what dead code
//! elimination removes or keeps because of the rewrite, shared setup), paid once.

use std::collections::BTreeMap;

use crate::types::SsaOp;

use super::super::emit_ssa::eliminate_ssa_dead_code;
use super::super::SsaProgram;
use super::cost::{category, Counts, CATEGORIES};
use super::rule::{Fact, FactSource};
use super::surgery::SurgeryReport;

/// Counts per category of some ops.
///
/// # Arguments
/// * `ops` - The ops.
///
/// # Returns
/// Counts in `CATEGORIES` order (barriers are not counted).
pub fn counts_of<'a>(ops: impl IntoIterator<Item = &'a SsaOp>) -> Result<Counts, String> {
    let mut c = [0i64; 10];
    for op in ops {
        if let Some(k) = category(op)? {
            c[k] += 1;
        }
    }
    Ok(c)
}

/// Counts of the ops dead code elimination keeps.
///
/// # Arguments
/// * `program` - The program (not modified).
///
/// # Returns
/// Counts in `CATEGORIES` order.
pub fn dce_counts(program: &SsaProgram) -> Result<Counts, String> {
    let mut vreg_def = program.vreg_def.clone();
    let mut legacy = program.const_alternatives.clone();
    let (ops, _, _) = eliminate_ssa_dead_code(
        program.ops.clone(),
        &mut vreg_def,
        &mut legacy,
        program.zero_vreg,
        Some(&program.vector_group_lanes),
    );
    counts_of(&ops)
}

/// `a - b` per category.
pub fn sub(a: &Counts, b: &Counts) -> Counts {
    let mut d = [0i64; 10];
    for k in 0..10 {
        d[k] = a[k] - b[k];
    }
    d
}

/// Nonzero entries by category name, e.g. `vec_fixed -1, load -1`.
pub fn format_counts(c: &Counts) -> String {
    let parts: Vec<String> =
        (0..10).filter(|&k| c[k] != 0).map(|k| format!("{} {:+}", CATEGORIES[k], c[k])).collect();
    if parts.is_empty() {
        "0".into()
    } else {
        parts.join(", ")
    }
}

/// What a rewrite did and what it costs.
#[derive(Clone, Debug, Default)]
pub struct RuleReport {
    /// Rule id.
    pub rule: &'static str,
    /// Sites rewritten, by site kind.
    pub sites: BTreeMap<&'static str, usize>,
    /// Rejected candidates by reason.
    pub rejections: BTreeMap<String, usize>,
    /// Per-site delta (after - before) -> number of sites with it, keyed by site kind.
    pub site_classes: BTreeMap<(&'static str, Counts), usize>,
    /// Whole-program delta after dead code elimination less the sites' deltas: what the
    /// rewrite made dead (or kept) beyond its own ops, paid once.
    pub cascade_delta: Counts,
    /// Whole-program delta after dead code elimination.
    pub total_delta: Counts,
    /// The surgery's report.
    pub surgery: SurgeryReport,
    /// Why the program was left unchanged although sites matched (`Surgery::check` refused
    /// it, e.g. not single assignment).
    pub refused: Option<String>,
    /// Why the cost could not be computed (an op the categories do not know); the rewrite is
    /// applied regardless and the deltas are left empty.
    pub cost_error: Option<String>,
    /// The whole-program facts and declared assumptions (`FactSource::Program` and
    /// `FactSource::Assumption`) the rewritten sites relied on, each once, in the order first
    /// seen: what a run must state for the rewrite to hold.
    pub facts: Vec<Fact>,
}

impl RuleReport {
    /// Record the facts a rewritten site relied on (`facts` keeps the whole-program ones).
    pub fn add_facts(&mut self, facts: &[Fact]) {
        for f in facts {
            if matches!(f.source, FactSource::Program | FactSource::Assumption) && !self.facts.contains(f) {
                self.facts.push(f.clone());
            }
        }
    }

    /// Names of the declared assumptions the rewrite relied on.
    pub fn assumptions(&self) -> Vec<&'static str> {
        self.facts.iter().filter(|f| f.source == FactSource::Assumption).map(|f| f.name).collect()
    }

    /// Total number of sites rewritten.
    pub fn n_sites(&self) -> usize {
        self.sites.values().sum()
    }

    /// Fill `cascade_delta` and `total_delta` from the program's counts before and after.
    pub fn close(&mut self, before: &Counts, after: &Counts) {
        self.total_delta = sub(after, before);
        let mut sites = [0i64; 10];
        for ((_, d), n) in &self.site_classes {
            for k in 0..10 {
                sites[k] += d[k] * *n as i64;
            }
        }
        self.cascade_delta = sub(&self.total_delta, &sites);
    }

    /// One line for the pipeline log.
    pub fn summary(&self) -> String {
        if let Some(r) = &self.refused {
            return format!("{}: program left unchanged: {}", self.rule, r);
        }
        let sites: Vec<String> = self.sites.iter().map(|(k, n)| format!("{} {}", n, k)).collect();
        let classes: Vec<String> =
            self.site_classes.iter().map(|((k, d), n)| format!("{} x {} [{}]", n, k, format_counts(d))).collect();
        let assumed = self.assumptions();
        format!(
            "{}: sites {}; per-site {}; cascade [{}]; total [{}]; {} removed, {} cascaded, {} ConstChoice alternatives changed{}",
            self.rule,
            if sites.is_empty() { "none".into() } else { sites.join(", ") },
            if classes.is_empty() { "none".into() } else { classes.join(", ") },
            format_counts(&self.cascade_delta),
            format_counts(&self.total_delta),
            self.surgery.removed,
            self.surgery.cascaded.len(),
            self.surgery.alternatives.len(),
            if assumed.is_empty() { String::new() } else { format!("; assumes {}", assumed.join(", ")) }
        )
    }
}
