//! Rule `hash-merge`: two adjacent hash stages `x = a*m + c; r = (x + d) ^ (x << k)` become
//! `r = (a*m + (c+d)) ^ (a*(m<<k) + (c<<k))` (all mod 2^32), as a peephole on a finished SSA
//! program. Catalogue rule 3; the emitter's `--merge-hash-stages` writes the same program
//! directly.
//!
//! The rule depends on the ops only, never on emitter state. What "works on untagged SSA"
//! covers today: any program that is single assignment, in def-before-use order, before
//! dependency computation, whose stage pair is vector ops on broadcast constants
//! (`ConstChoice` / `Const` whose immediate agrees with `produces_constant`). Not covered:
//! scalar (ALU) hashes and the reference 18-op form (they need SLP vectorisation and R1
//! fusion first), constants computed by other ops, and programs whose dependencies or
//! memory edges were already computed (refused).
//!
//! * Propose: every `multiply_add` is a candidate anchor (`ProposalSource::Ops`, the default,
//!   which also works on untagged or lifted SSA), or only those inside `hash` regions
//!   (`ProposalSource::Tags`, tags narrow the candidates and add nothing else).
//! * Match: `x = multiply_add(a, M, C)` with `M`, `C` broadcast constants; `x` is read by
//!   exactly two ops, `t = x + D` and `s = x << K` with `D`, `K` broadcast constants,
//!   `0 <= K < 32`; `t` and `s` are read only by `r = t ^ s`. Nothing keeps `x`, `t` or `s`
//!   live and none of them carries a trace id. A right shift is rejected
//!   (`R8-hash-stage-merge-needs-left-shift`: it does not distribute over `a*m + c`).
//! * Replace: `l = multiply_add(a, M, C1)`, `h = multiply_add(a, M2, C2)`, `r' = l ^ h`
//!   inserted where `x` was; readers of `r` read `r'`; `x, t, s, r` are removed. The new
//!   vector constants go through `Surgery::vector_constant`, i.e. reuse what exists and are
//!   otherwise new `ConstChoice` + `vbroadcast` ops, inserted after the last definition of
//!   any constant operand of the rewritten sites (the anchor). An old operand that other ops
//!   still read (the stage-3 shift 9 is also the stage-4 multiplier, shared by CSE) moves to
//!   the anchor too, so the new form's setup is contiguous. Constants only the old sites read
//!   die with them; every `ConstChoice`'s alternatives are recomputed for the new constant
//!   set and order (C13, see `surgery`).
//!
//! The identity holds for any `m, c, d` and `0 <= k < 32` (`x << k = x * 2^k`, which
//! distributes over `a*m + c` modulo 2^32): `R8-hash-stage-merge-general` in
//! `verification/rewrites.py`. `R8-hash-stage-merge` proves the instance of hash stages 2 and
//! 3; the matcher names on each site whether its constants are that instance (`Fact`
//! `proof_instance`).

use std::collections::{BTreeMap, BTreeSet};

use crate::types::{Engine, Opcode, Operand, RegionKind, SsaOp, VReg};

use super::super::config::{merged_hash_stage_constants, HASH_STAGES, MERGED_HASH_STAGE};
use super::super::emit_ssa::eliminate_ssa_dead_code;
use super::super::SsaProgram;
use super::cost::{category, Counts};
use super::graph::SsaGraph;
use super::rule::{AlternativeSpec, Fact, FactSource, Proposal, Rejection, Rule};
use super::surgery::{Finished, Stamp, Surgery, SurgeryReport};

/// Rule id.
pub const HASH_MERGE_RULE: &str = "hash-merge";

/// Alternatives of a hash-merge site.
pub const HASH_MERGE_ALTERNATIVES: [AlternativeSpec; 2] = [
    AlternativeSpec { name: "unmerged", proofs: &["R1-hash-fma-fusion"] },
    AlternativeSpec {
        name: "merged",
        proofs: &["R8-hash-stage-merge-general", "R8-hash-stage-merge", "R8-hash-stage-merge-needs-left-shift"],
    },
];

const MASK: i64 = 0xFFFF_FFFF;

/// Where the rule looks for sites.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ProposalSource {
    /// Every `multiply_add` (complete; needs no tags).
    #[default]
    Ops,
    /// Only `multiply_add`s inside `hash` regions (tags narrow the search).
    Tags,
}

/// The hash-merge rule.
#[derive(Clone, Copy, Debug, Default)]
pub struct HashMergeRule {
    /// Where to look.
    pub source: ProposalSource,
}

/// A matched site: the four ops and everything the replacement needs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HashMergeSite {
    /// `x = multiply_add(a, M, C)`.
    pub fma: usize,
    /// `t = x + D`.
    pub add: usize,
    /// `s = x << K`.
    pub shl: usize,
    /// `r = t ^ s`.
    pub xor: usize,
    /// The varying input `a`.
    pub input: VReg,
    /// The output `r`.
    pub output: VReg,
    /// Vector operands `M`, `C`, `D`, `K`.
    pub operands: [VReg; 4],
    /// Their values `m, c, d, k`.
    pub values: [i64; 4],
    /// `(m1, c1, m2, c2)` of the merged form.
    pub merged: (i64, i64, i64, i64),
    /// Round label (tag), if any.
    pub round: Option<u32>,
    /// Group label (tag), if any.
    pub group: Option<u32>,
    /// Facts the match relies on.
    pub facts: Vec<Fact>,
}

impl HashMergeSite {
    /// Indices of the four ops the rewrite removes.
    pub fn ops(&self) -> [usize; 4] {
        [self.fma, self.add, self.shl, self.xor]
    }
}

/// Constants of the merged form for `x = a*m + c; (x + d) ^ (x << k)`, each in `0..2^32`.
///
/// # Arguments
/// * `m`, `c`, `d`, `k` - The site's constants (`k < 32`).
///
/// # Returns
/// `(m1, c1, m2, c2)` with `m1 = m`, `c1 = c + d`, `m2 = m << k`, `c2 = c << k`, mod 2^32.
pub fn merged_constants(m: i64, c: i64, d: i64, k: i64) -> (i64, i64, i64, i64) {
    // The inputs are whatever the constants' annotations say (any i64 that agrees with the
    // immediate): reduce them first and wrap, so nothing overflows.
    let (m, c, d) = (m & MASK, c & MASK, d & MASK);
    let k = (k & 31) as u32;
    (m, c.wrapping_add(d) & MASK, m.wrapping_shl(k) & MASK, c.wrapping_shl(k) & MASK)
}

#[cfg(test)]
mod merged_constants_tests {
    use super::merged_constants;

    #[test]
    fn test_merged_constants_wrap_mod_2_32() {
        // Annotations may hold any i64 equal to the immediate: no overflow, results in 0..2^32.
        assert_eq!(merged_constants(i64::MAX, i64::MAX, i64::MAX, 31), (0xFFFF_FFFF, 0xFFFF_FFFE, 0x8000_0000, 0x8000_0000));
        assert_eq!(merged_constants(-1, -1, 1, 0), (0xFFFF_FFFF, 0, 0xFFFF_FFFF, 0xFFFF_FFFF));
        assert_eq!(merged_constants(3, 0xFFFF_FFFF, 2, 4), (3, 1, 48, 0xFFFF_FFF0));
    }
}

fn vreg_srcs(op: &SsaOp) -> Option<Vec<VReg>> {
    op.srcs.iter().map(|s| s.as_vreg()).collect()
}

impl HashMergeRule {
    /// Match a site anchored at the `multiply_add` op `i`.
    ///
    /// # Arguments
    /// * `graph` - The program.
    /// * `i` - Index of the candidate `multiply_add`.
    ///
    /// # Returns
    /// The site, or why the ops around `i` do not have the rule's shape.
    pub fn match_at(&self, graph: &SsaGraph, i: usize) -> Result<HashMergeSite, Rejection> {
        let reject = |s: String| Err(Rejection(s));
        let ops = graph.ops;
        let fma = &ops[i];
        if fma.engine != Engine::Valu || fma.opcode != Opcode::MultiplyAdd {
            return reject("anchor is not a vector multiply_add".into());
        }
        let Some(x) = fma.dest else { return reject("multiply_add without a dest".into()) };
        if graph.is_redefined(x) {
            return reject("x is defined more than once (not single assignment)".into());
        }
        let Some(s) = vreg_srcs(fma).filter(|s| s.len() == 3) else {
            return reject("multiply_add operands are not three VRegs".into());
        };
        // Multiplication commutes: one factor is the broadcast constant M, the other varies.
        let (a, vm, m) = match (graph.broadcast_const(s[0]), graph.broadcast_const(s[1])) {
            (None, Some(m)) => (s[0], s[1], m),
            (Some(m), None) => (s[1], s[0], m),
            (Some(_), Some(_)) => return reject("both factors are constants".into()),
            (None, None) => return reject("multiplier is not a broadcast constant".into()),
        };
        let vc = s[2];
        let Some(c) = graph.broadcast_const(vc) else { return reject("addend is not a broadcast constant".into()) };

        let readers = graph.readers(i);
        if readers.len() != 2 {
            return reject(format!("intermediate has {} readers (needs exactly the add and the shift)", readers.len()));
        }
        let (mut add, mut shl) = (None, None);
        for &u in readers {
            let op = &ops[u];
            let Some(us) = vreg_srcs(op) else { return reject("reader with an immediate operand".into()) };
            if op.engine != Engine::Valu || us.len() != 2 {
                return reject(format!("reader {:?} on {:?} is not a binary vector op", op.opcode, op.engine));
            }
            match op.opcode {
                Opcode::Add => {
                    let other = if us[0] == x { us[1] } else if us[1] == x { us[0] } else { return reject("add does not read x".into()) };
                    let Some(d) = graph.broadcast_const(other) else { return reject("add operand is not a broadcast constant".into()) };
                    add = Some((u, other, d));
                }
                Opcode::Shl => {
                    if us[0] != x {
                        return reject("x is the shift amount, not the shifted value".into());
                    }
                    let Some(k) = graph.broadcast_const(us[1]) else { return reject("shift amount is not a broadcast constant".into()) };
                    if !(0..32).contains(&k) {
                        return reject(format!("shift amount {} is not in 0..32", k));
                    }
                    shl = Some((u, us[1], k));
                }
                Opcode::Shr => return reject("right shift does not distribute over a*m + c (R8-hash-stage-merge-needs-left-shift)".into()),
                other => return reject(format!("reader {:?} is neither the add nor the left shift", other)),
            }
        }
        let (Some((add, vd, d)), Some((shl, vk, k))) = (add, shl) else {
            return reject("readers are not one add and one left shift".into());
        };
        let (rt, rs) = (graph.readers(add), graph.readers(shl));
        if rt.len() != 1 || rs.len() != 1 || rt[0] != rs[0] {
            return reject("the add and the shift are not read by exactly one common op".into());
        }
        let xor = rt[0];
        let xop = &ops[xor];
        let (t, sh) = (ops[add].dest.unwrap(), ops[shl].dest.unwrap());
        let xs = vreg_srcs(xop).unwrap_or_default();
        if xop.engine != Engine::Valu || xop.opcode != Opcode::Xor || !(xs == [t, sh] || xs == [sh, t]) {
            return reject(format!("the combining op is {:?}, not t ^ s", xop.opcode));
        }
        let Some(r) = xop.dest else { return reject("xor without a dest".into()) };
        for (v, name) in [(a, "a"), (t, "t"), (sh, "s"), (r, "r")] {
            if graph.is_redefined(v) {
                return reject(format!("{} is defined more than once (not single assignment)", name));
            }
        }
        for (v, name) in [(x, "x"), (t, "t"), (sh, "s")] {
            if graph.is_kept_live(v) {
                return reject(format!("{} (or a lane of it) is kept live by another op", name));
            }
        }
        // The removed ops' own keep_live lists would be lost with them.
        if [i, add, shl, xor].iter().any(|&j| ops[j].keep_live.as_ref().map_or(false, |k| !k.is_empty())) {
            return reject("a site op carries keep_live".into());
        }
        // Vector groups name VRegs outside the ops (lanes, aliases); a rename would orphan them.
        for (v, name) in [(x, "x"), (t, "t"), (sh, "s"), (r, "r")] {
            if graph.group_lanes(v).is_some() || graph.element_of(v).is_some() {
                return reject(format!("{} belongs to a vector group", name));
            }
        }
        if [i, add, shl].iter().any(|&j| ops[j].trace_id.is_some()) {
            return reject("an intermediate carries a trace id".into());
        }

        let merged = merged_constants(m, c, d, k);
        let proved = merged_hash_stage_constants(MERGED_HASH_STAGE).map_or(false, |p| {
            let (_, c2, _, _, _) = HASH_STAGES[MERGED_HASH_STAGE];
            let (_, d3, _, _, k3) = HASH_STAGES[MERGED_HASH_STAGE + 1];
            p == merged && c == c2 && d == d3 && k == k3
        });
        let label = |f: fn(&crate::types::RegionTag) -> Option<u32>| fma.region.and_then(|t| f(&t));
        Ok(HashMergeSite {
            fma: i,
            add,
            shl,
            xor,
            input: a,
            output: r,
            operands: [vm, vc, vd, vk],
            values: [m, c, d, k],
            merged,
            round: label(|t| t.round),
            group: label(|t| t.group),
            facts: vec![
                Fact { name: "constants", value: format!("m={:#x} c={:#x} d={:#x} k={}", m, c, d, k), source: FactSource::Structure },
                Fact { name: "single_use", value: "x read by the add and the shift only; both read by the xor only".into(), source: FactSource::Structure },
                Fact {
                    name: "proof_instance",
                    value: if proved { "R8-hash-stage-merge (stages 2+3)".into() } else { "R8-hash-stage-merge-general".into() },
                    source: FactSource::Structure,
                },
            ],
        })
    }

    /// Every site of `graph`, in op order, and the number of rejected candidates per reason.
    ///
    /// # Arguments
    /// * `graph` - The program.
    ///
    /// # Returns
    /// `(sites, rejections)`.
    pub fn find_sites(&self, graph: &SsaGraph) -> (Vec<HashMergeSite>, BTreeMap<String, usize>) {
        let mut sites = Vec::new();
        let mut rejected: BTreeMap<String, usize> = BTreeMap::new();
        for p in self.propose(graph) {
            match self.match_site(graph, &p) {
                Ok(site) => sites.push(site),
                Err(e) => *rejected.entry(e.0).or_default() += 1,
            }
        }
        sites.sort_by_key(|s| s.fma);
        sites.dedup_by_key(|s| s.fma);
        (sites, rejected)
    }
}

impl Rule for HashMergeRule {
    type Site = HashMergeSite;

    fn id(&self) -> &'static str {
        HASH_MERGE_RULE
    }

    fn alternatives(&self) -> &'static [AlternativeSpec] {
        &HASH_MERGE_ALTERNATIVES
    }

    /// One proposal per candidate `multiply_add` (inside a `hash` region with
    /// `ProposalSource::Tags`), in op order.
    fn propose(&self, graph: &SsaGraph) -> Vec<Proposal> {
        graph
            .ops
            .iter()
            .enumerate()
            .filter(|(_, op)| op.engine == Engine::Valu && op.opcode == Opcode::MultiplyAdd)
            .filter(|(_, op)| match self.source {
                ProposalSource::Ops => true,
                ProposalSource::Tags => op.region.map_or(false, |t| t.kind == RegionKind::Hash),
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

    fn match_site(&self, graph: &SsaGraph, proposal: &Proposal) -> Result<HashMergeSite, Rejection> {
        let mut last = Rejection("proposal has no multiply_add".into());
        for &i in &proposal.ops {
            match self.match_at(graph, i) {
                Ok(site) => return Ok(site),
                Err(e) => last = e,
            }
        }
        Err(last)
    }
}

/// What `apply_hash_merge` did, with the rule's cost.
#[derive(Clone, Debug, Default)]
pub struct HashMergeReport {
    /// Sites matched.
    pub sites: usize,
    /// Rejected candidates by reason.
    pub rejections: BTreeMap<String, usize>,
    /// Per-site cost delta (after - before, `CATEGORIES` order) -> number of sites with it.
    /// Counted on each site's own ops: the removed `x, t, s, r` and the inserted `l, h, r'`.
    pub site_classes: BTreeMap<Counts, usize>,
    /// Whole-program delta after dead code elimination, less the sites' deltas: the shared
    /// setup the rewrite adds and frees (new and dead constants, broadcasts, C13 category
    /// changes), paid once.
    pub resource_delta: Counts,
    /// The surgery's report (cascaded removals, constants whose alternatives changed).
    pub surgery: SurgeryReport,
    /// Why the program was left unchanged although sites matched (`Surgery::new` refused it,
    /// e.g. a VReg defined twice).
    pub refused: Option<String>,
    /// Why the cost could not be counted (an op the categories do not know); the rewrite is
    /// done regardless and the cost fields are then empty.
    pub cost_error: Option<String>,
    /// Why moving the old operands that stay to the new setup was skipped (they then stay
    /// where they are).
    pub regroup_skipped: Option<String>,
}

impl HashMergeReport {
    /// One line for the pipeline log.
    pub fn summary(&self) -> String {
        if let Some(r) = &self.refused {
            return format!("hash-merge: program left unchanged: {}", r);
        }
        let classes: Vec<String> = self.site_classes.iter().map(|(d, n)| format!("{} x {:?}", n, d)).collect();
        let mut notes = String::new();
        if let Some(e) = &self.cost_error {
            notes.push_str(&format!("; cost not counted: {}", e));
        }
        if let Some(e) = &self.regroup_skipped {
            notes.push_str(&format!("; old operands left in place: {}", e));
        }
        if self.surgery.foreign_alternatives > 0 {
            notes.push_str(&format!("; {} ConstChoice with non-emitter alternatives kept", self.surgery.foreign_alternatives));
        }
        format!(
            "hash-merge: {} sites; per-site delta {}; resources {:?}; {} ops inserted, {} removed, {} moved, {} cascaded, {} ConstChoice alternatives changed",
            self.sites,
            classes.join(", "),
            self.resource_delta,
            self.surgery.inserted,
            self.surgery.removed,
            self.surgery.moved,
            self.surgery.cascaded.len(),
            self.surgery.alternatives.len()
        ) + &notes
    }
}

fn counts_of<'a>(ops: impl IntoIterator<Item = &'a SsaOp>) -> Result<Counts, String> {
    let mut c = [0i64; 10];
    for op in ops {
        if let Some(k) = category(op)? {
            c[k] += 1;
        }
    }
    Ok(c)
}

fn dce_counts(program: &SsaProgram) -> Result<Counts, String> {
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

/// Rewrite every hash-merge site of `program` chosen by `select`.
///
/// # Arguments
/// * `program` - A finished program (before dependency computation).
/// * `rule` - The rule (where it looks).
/// * `select` - Which matched sites to rewrite (`|_| true` for all).
///
/// # Returns
/// The rewritten program (dead code the rewrite made is removed; other dead code is left for
/// the pipeline's DCE) and the report with per-site and shared cost deltas. Sites the rule
/// cannot rewrite safely are rejected one by one (`rejections`); a program that is not
/// single assignment or not in def-before-use order is returned unchanged (`refused`); an
/// op the cost categories do not know leaves the cost empty (`cost_error`) and does not stop
/// the rewrite. An `Err` means the surgery itself found an inconsistency (a bug).
pub fn apply_hash_merge(
    program: SsaProgram,
    rule: &HashMergeRule,
    select: impl Fn(&HashMergeSite) -> bool,
) -> Result<(SsaProgram, HashMergeReport), String> {
    let (sites, rejections) = {
        let graph = SsaGraph::new(&program.ops, &program.vector_group_lanes);
        rule.find_sites(&graph)
    };
    let sites: Vec<HashMergeSite> = sites.into_iter().filter(|s| select(s)).collect();
    let mut report = HashMergeReport { sites: sites.len(), rejections, ..Default::default() };
    if sites.is_empty() {
        return Ok((program, report));
    }
    // A program the surgery cannot edit safely is left as it is, with the reason.
    if let Err(e) = Surgery::check(&program) {
        report.refused = Some(e);
        report.sites = 0;
        return Ok((program, report));
    }
    // Cost accounting never blocks the rewrite: on an op it cannot categorise it is skipped.
    let before_counts = match dce_counts(&program) {
        Ok(c) => Some(c),
        Err(e) => {
            report.cost_error = Some(e);
            None
        }
    };
    // Insertion point of new constants: after the last definition of any constant operand of
    // the rewritten sites (all of them are visible there), unless that comes after the first
    // site (operands defined next to each site); then right before the first site, where new
    // constants (which read nothing) are visible to every site.
    let (anchor, roles, regroup) = {
        let graph = SsaGraph::new(&program.ops, &program.vector_group_lanes);
        let mut last_def = 0;
        for s in &sites {
            for &v in &s.operands {
                last_def = last_def.max(graph.def_index(v).expect("constant operand has a definer"));
            }
        }
        let first_site = sites.iter().map(|s| s.fma).min().unwrap();
        // first_site > 0: its constant operands are defined before it.
        let anchor = if last_def < first_site { last_def } else { first_site.saturating_sub(1) };
        // Metadata of new scalar and vector constants: the anchor's labels (no region or
        // symmetry hint when the anchor is not a setup op), the type of the operand they
        // stand in for (the multiplier for M2, the addend for C1 and C2).
        let first = &sites[0];
        let mut at = Stamp::of(&program.ops[anchor]);
        if anchor != last_def {
            at.region = None;
            at.symmetry_hint = None;
        }
        let role = |v: VReg| {
            let b = graph.def_op(v).unwrap();
            let s = b.srcs[0].as_vreg().and_then(|s| graph.def_op(s));
            (at.clone().with_var_type(s.and_then(|o| o.var_type)), at.clone().with_var_type(b.var_type))
        };
        // Old operands that other ops keep reading (the shift 9 is also the stage-4
        // multiplier, shared by CSE) move to the anchor with their scalar, so the merged
        // form's setup is contiguous, in the order `--merge-hash-stages` emits it.
        let removed: BTreeSet<usize> = sites.iter().flat_map(|s| s.ops()).collect();
        let mut regroup: BTreeSet<usize> = BTreeSet::new();
        for s in &sites {
            for &v in &s.operands {
                let d = graph.def_index(v).unwrap();
                if d >= anchor || graph.readers(d).iter().all(|r| removed.contains(r)) {
                    continue;
                }
                if let Some(sd) = graph.def_op(v).and_then(|b| b.srcs[0].as_vreg()).and_then(|sv| graph.def_index(sv)) {
                    if graph.readers(sd).iter().all(|&r| r == d || r > anchor) {
                        regroup.insert(sd);
                    }
                }
                regroup.insert(d);
            }
        }
        (anchor, [role(first.operands[0]), role(first.operands[1])], regroup)
    };
    let [(mult_s, mult_v), (add_s, add_v)] = roles;

    let mut surgery = Surgery::new(program)?;
    // Left in place (and reported) if something that stays reads them before the anchor.
    if let Err(e) = surgery.move_after(&regroup.into_iter().collect::<Vec<_>>(), anchor) {
        report.regroup_skipped = Some(e);
    }
    for s in &sites {
        let (m1, c1, m2, c2) = s.merged;
        let v_m1 = surgery.vector_constant(m1, anchor, &mult_s, &mult_v);
        let v_c1 = surgery.vector_constant(c1, anchor, &add_s, &add_v);
        let v_m2 = surgery.vector_constant(m2, anchor, &mult_s, &mult_v);
        let v_c2 = surgery.vector_constant(c2, anchor, &add_s, &add_v);
        // l = a*M + C1; h = a*M2 + C2; r' = l ^ h, where x was, with x's labels; r' keeps
        // r's trace id.
        let stamp = Stamp::of(&surgery.ops()[s.fma]);
        let vec_op = |surgery: &mut Surgery, opcode: Opcode, srcs: [VReg; 3]| {
            let dest = surgery.fresh(true);
            let n = if opcode == Opcode::MultiplyAdd { 3 } else { 2 };
            let op = SsaOp::new(0, Engine::Valu, opcode, Some(dest), srcs[..n].iter().map(|&v| Operand::VReg(v)).collect());
            (op, dest)
        };
        let (l_op, l) = vec_op(&mut surgery, Opcode::MultiplyAdd, [s.input, v_m1, v_c1]);
        let (h_op, h) = vec_op(&mut surgery, Opcode::MultiplyAdd, [s.input, v_m2, v_c2]);
        let (mut r_op, r) = vec_op(&mut surgery, Opcode::Xor, [l, h, h]);
        r_op.trace_id = surgery.ops()[s.xor].trace_id.clone();
        let inserted = [l_op, h_op, r_op];

        // The site's own delta: the ops it inserts less the ops it removes.
        let delta = counts_of(s.ops().iter().map(|&i| &surgery.ops()[i])).and_then(|removed| {
            let added = counts_of(inserted.iter())?;
            Ok(std::array::from_fn::<i64, 10, _>(|k| added[k] - removed[k]))
        });
        match delta {
            Ok(d) => *report.site_classes.entry(d).or_default() += 1,
            Err(e) => {
                report.cost_error.get_or_insert(e);
            }
        }
        for op in inserted {
            surgery.insert_before(s.fma, op, &stamp);
        }
        for idx in s.ops() {
            surgery.remove(idx);
        }
        surgery.replace_uses(s.output, r);
    }
    let (program, surgery_report) = match surgery.finish_or_refuse()? {
        Finished::Done(program, report) => (program, report),
        Finished::Refused(program, why) => {
            report.refused = Some(why);
            report.sites = 0;
            report.site_classes = Default::default();
            return Ok((program, report));
        }
    };
    report.surgery = surgery_report;

    if report.cost_error.is_some() {
        report.site_classes.clear();
        return Ok((program, report));
    }
    match (before_counts, dce_counts(&program)) {
        (Some(before), Ok(after)) => {
            for k in 0..10 {
                let sites_total: i64 = report.site_classes.iter().map(|(d, n)| d[k] * *n as i64).sum();
                report.resource_delta[k] = after[k] - before[k] - sites_total;
            }
        }
        (_, Err(e)) => report.cost_error = Some(e),
        (None, _) => {}
    }
    Ok((program, report))
}
