//! Tests of the index rules: `prune-index` and `values-only` reproduce the emissions of
//! `--prune-dead-index` and `--values-only`, alone, together and with `hash-merge`, in any
//! order; facts are derived, not read from tags; near misses are refused with named reasons;
//! the values-only saving follows from dead code elimination alone (C3, C4, C5).

use std::collections::BTreeMap;

use super::graph::SsaGraph;
use super::hash_merge::ProposalSource;
use super::prune_index::{PruneIndexRule, PruneKind};
use super::report::{format_counts, RuleReport};
use super::surgery::recompute_alternatives;
use super::tests::{canonical_ops, clone_program, first_difference, flow90_plan, raw_program};
use super::values_only::{apply_values_only, ValuesOnlyRule};
use crate::kernel_builder::config::{
    AccessEncoding, CachedSelect, IndexForm, ProblemSpec, SelectBitSource, TransformKind,
};
use crate::kernel_builder::emit_ssa::eliminate_ssa_dead_code;
use crate::kernel_builder::pipeline::passes::transform::apply_transform;
use crate::kernel_builder::SsaProgram;
use crate::types::{Opcode, RegionKind, SsaOp};

fn dce(p: &SsaProgram) -> SsaProgram {
    let mut q = clone_program(p);
    let (ops, vreg_def, const_alternatives) = eliminate_ssa_dead_code(
        std::mem::take(&mut q.ops),
        &mut q.vreg_def,
        &mut q.const_alternatives,
        q.zero_vreg,
        Some(&q.vector_group_lanes),
    );
    SsaProgram { ops, vreg_def, const_alternatives, ..q }
}

/// Recompute every `ConstChoice`'s ALU-derived alternatives of both (dead-code-eliminated)
/// programs for their live constants, and list the constants whose alternatives changed.
///
/// # Returns
/// `(value, alternatives in a, alternatives in b)` for each constant whose alternatives
/// differed between `a` and `b` before recomputing and agree after.
fn stale_alternatives(a: &mut SsaProgram, b: &mut SsaProgram) -> Vec<(i64, String, String)> {
    let before: Vec<_> = a.ops.iter().zip(&b.ops).map(|(x, y)| (x.const_alternatives.clone(), y.const_alternatives.clone())).collect();
    for p in [&mut *a, &mut *b] {
        let alts = recompute_alternatives(&p.ops, p.zero_vreg.is_some(), |_| false);
        for (op, alt) in p.ops.iter_mut().zip(alts) {
            if op.opcode == Opcode::ConstChoice && op.const_alternatives.is_some() {
                op.const_alternatives = alt;
            }
        }
    }
    let mut out = Vec::new();
    for ((x, y), (bx, by)) in a.ops.iter().zip(&b.ops).zip(before) {
        if bx != by && x.const_alternatives == y.const_alternatives {
            out.push((x.produces_constant.unwrap_or(0), format!("{:?}", bx), format!("{:?}", by)));
        }
    }
    out
}

/// The flags a list of transforms stands for.
fn with_flags(spec: &ProblemSpec, transforms: &[TransformKind]) -> ProblemSpec {
    let mut s = spec.clone();
    for t in transforms {
        match t {
            TransformKind::HashMerge => s.merge_hash_stages = true,
            TransformKind::PruneIndex => s.prune_dead_index = true,
            TransformKind::ValuesOnly => s.values_only = true,
            TransformKind::SelectBitsPath => s.select_bit_source = SelectBitSource::Path,
            TransformKind::MirrorIndex => s.index_form = IndexForm::Mirrored,
        }
    }
    s
}

/// Outcome of a parity check.
#[derive(Debug)]
pub(super) struct Parity {
    /// The raw programs (before DCE) are equal op for op.
    pub raw_exact: bool,
    /// Ops differing between the raw programs (first few, printed).
    pub raw_diff: Option<String>,
    /// Ops that differ only by the emitter's leaked `HashState` label (after DCE).
    pub leaked: usize,
    /// Constants whose ALU-derived alternatives differ after DCE before recomputing them:
    /// `(value, rewritten, emitted)`. The rewrite recomputes them for the live constants
    /// (C13); DCE only drops the ones that name a constant it removed (#79), so the emitted
    /// program can list fewer.
    pub stale_alternatives: Vec<(i64, String, String)>,
    /// Summaries of each transform.
    pub summaries: Vec<String>,
    /// The rewritten raw program, printed by `canonical_ops`.
    pub program: Vec<String>,
}

/// Gate 1: apply `transforms` in order to the emission of `spec` without the flags, and
/// compare with the emission with the flags, op for op (VRegs renamed by first appearance,
/// region ids dropped). After dead code elimination the programs must be equal (but for the
/// one leaked label of BUGS.md); the raw programs are compared too and the difference
/// reported (the emitter writes ops it knows are dead, such as the last round's index update
/// under `--values-only`, which the rewrite removes with what it made dead).
pub(super) fn check_parity(spec: &ProblemSpec, transforms: &[TransformKind], label: &str) -> Parity {
    let base = ProblemSpec {
        merge_hash_stages: false,
        prune_dead_index: false,
        values_only: false,
        select_bit_source: SelectBitSource::Index,
        index_form: if transforms.contains(&TransformKind::MirrorIndex) { IndexForm::ZeroBased } else { spec.index_form },
        ..spec.clone()
    };
    let mut p = raw_program(&base);
    let mut summaries = Vec::new();
    for &t in transforms {
        let (q, s) = apply_transform(p, t, Some(base.batch_size)).unwrap_or_else(|e| panic!("{}: {}", label, e));
        p = q;
        summaries.push(s.summary);
    }
    let emitted = raw_program(&with_flags(&base, transforms));
    let (raw_diff, _) = first_difference(&canonical_ops(&p), &canonical_ops(&emitted));
    let (mut a, mut b) = (dce(&p), dce(&emitted));
    let stale = stale_alternatives(&mut a, &mut b);
    let (diff, leaked) = first_difference(&canonical_ops(&a), &canonical_ops(&b));
    if let Some(d) = diff {
        panic!("{} {:?}: after DCE: {}\n{}", label, transforms, d, summaries.join("\n"));
    }
    let parity = Parity {
        raw_exact: raw_diff.is_none(),
        raw_diff,
        leaked: leaked.len(),
        stale_alternatives: stale,
        summaries,
        program: canonical_ops(&p),
    };
    eprintln!(
        "{} {:?}: DCE parity, raw {}, {} leaked label(s), stale alternatives in the emission: {:?}\n  {}",
        label,
        transforms.iter().map(|t| t.as_str()).collect::<Vec<_>>(),
        if parity.raw_exact { "exact".to_string() } else { format!("differs: {}", parity.raw_diff.as_deref().unwrap_or("")) },
        parity.leaked,
        parity.stale_alternatives,
        parity.summaries.join("\n  ")
    );
    parity
}

pub(super) fn configs(rounds: usize) -> Vec<(String, ProblemSpec)> {
    let d = ProblemSpec { rounds, ..Default::default() };
    let m = ProblemSpec { index_form: IndexForm::Mirrored, ..d.clone() };
    let mut out = vec![
        (format!("{}r default", rounds), d.clone()),
        (format!("{}r mirrored", rounds), m.clone()),
        (format!("{}r bits", rounds), ProblemSpec { cached_select: CachedSelect::Bits, ..d.clone() }),
        (format!("{}r mirrored bits", rounds), ProblemSpec { cached_select: CachedSelect::Bits, ..m.clone() }),
    ];
    if rounds == 16 {
        out.push(("16r flow90 plan".into(), ProblemSpec { access_plan: Some(flow90_plan(&m)), ..m }));
    }
    out
}

fn single(rounds: usize, t: TransformKind) {
    for (label, spec) in configs(rounds) {
        check_parity(&spec, &[t], &label);
    }
}

#[test]
fn test_prune_index_parity_2_rounds() {
    single(2, TransformKind::PruneIndex);
}

#[test]
fn test_prune_index_parity_16_rounds() {
    single(16, TransformKind::PruneIndex);
}

#[test]
fn test_values_only_parity_2_rounds() {
    single(2, TransformKind::ValuesOnly);
}

#[test]
fn test_values_only_parity_16_rounds() {
    single(16, TransformKind::ValuesOnly);
}

fn permutations(ts: &[TransformKind]) -> Vec<Vec<TransformKind>> {
    if ts.len() <= 1 {
        return vec![ts.to_vec()];
    }
    let mut out = Vec::new();
    for i in 0..ts.len() {
        let mut rest = ts.to_vec();
        let first = rest.remove(i);
        for mut p in permutations(&rest) {
            p.insert(0, first);
            out.push(p);
        }
    }
    out
}

fn combined(rounds: usize) {
    use TransformKind::*;
    for (label, spec) in configs(rounds) {
        for set in [vec![PruneIndex, ValuesOnly], vec![HashMerge, PruneIndex, ValuesOnly]] {
            let mut programs = Vec::new();
            for order in permutations(&set) {
                // Every order gives the same raw program.
                let parity = check_parity(&spec, &order, &label);
                programs.push((order, parity.program));
            }
            for (order, ops) in &programs[1..] {
                let (d, _) = first_difference(ops, &programs[0].1);
                assert!(d.is_none(), "{}: order {:?} differs from {:?}: {:?}", label, order, programs[0].0, d);
            }
        }
    }
}

#[test]
fn test_index_transforms_compose_in_any_order_2_rounds() {
    combined(2);
}

#[test]
fn test_index_transforms_compose_in_any_order_16_rounds() {
    combined(16);
}

// ----------------------------------------------------------------------------
// Tags are hints: the same sites without them
// ----------------------------------------------------------------------------

fn stripped(p: &SsaProgram) -> SsaProgram {
    let mut q = clone_program(p);
    for op in &mut q.ops {
        op.region = None;
    }
    q
}

#[test]
fn test_index_rules_find_the_same_sites_without_tags() {
    let rule = |source| PruneIndexRule { source, ..PruneIndexRule::for_problem(256) };
    let shape = |p: &SsaProgram, source| {
        let g = SsaGraph::new(&p.ops, &p.vector_group_lanes);
        rule(source).find_sites(&g, p.zero_vreg).0.iter().map(|s| (s.kind, s.op, s.output, s.replacement)).collect::<Vec<_>>()
    };
    let tagged = raw_program(&ProblemSpec::default());
    let bare = stripped(&tagged);
    let by_ops = shape(&bare, ProposalSource::Ops);
    assert_eq!(by_ops.iter().filter(|s| s.0 == PruneKind::SharedConstant).count(), 31);
    assert_eq!(by_ops.iter().filter(|s| s.0 == PruneKind::ZeroUpdate).count(), 64);
    assert_eq!(shape(&tagged, ProposalSource::Ops), by_ops);
    assert_eq!(shape(&tagged, ProposalSource::Tags), by_ops);
    assert!(shape(&bare, ProposalSource::Tags).is_empty());
    // values-only reads no tags at all.
    let stores = |p: &SsaProgram| {
        let g = SsaGraph::new(&p.ops, &p.vector_group_lanes);
        ValuesOnlyRule::for_problem(256).find_sites(&g, p.zero_vreg).0.iter().map(|s| (s.store, s.address)).collect::<Vec<_>>()
    };
    assert_eq!(stores(&bare).len(), 32);
    assert_eq!(stores(&bare), stores(&tagged));
    // The rewrites of the stripped program are the tagged rewrites without tags.
    for t in [TransformKind::PruneIndex, TransformKind::ValuesOnly] {
        let (a, _) = apply_transform(clone_program(&bare), t, Some(256)).unwrap();
        let (b, _) = apply_transform(clone_program(&tagged), t, Some(256)).unwrap();
        assert_eq!(canonical_ops(&a), canonical_ops(&stripped(&b)), "{:?}", t);
    }
}

// ----------------------------------------------------------------------------
// Near misses, refused with named reasons
// ----------------------------------------------------------------------------

fn prune_sites(p: &SsaProgram, rule: &PruneIndexRule) -> (BTreeMap<PruneKind, usize>, BTreeMap<String, usize>) {
    let g = SsaGraph::new(&p.ops, &p.vector_group_lanes);
    let (sites, rejected) = rule.find_sites(&g, p.zero_vreg);
    let mut kinds = BTreeMap::new();
    for s in sites {
        *kinds.entry(s.kind).or_insert(0) += 1;
    }
    (kinds, rejected)
}

fn reason_count(rejected: &BTreeMap<String, usize>, text: &str) -> usize {
    rejected.iter().filter(|(r, _)| r.contains(text)).map(|(_, n)| n).sum()
}

/// Index of the op defining `v`.
fn def(p: &SsaProgram, v: crate::types::VReg) -> usize {
    p.ops.iter().position(|o| o.dest == Some(v)).unwrap()
}

#[test]
fn test_prune_index_refuses_an_index_it_cannot_show_is_zero() {
    use crate::types::{Engine, Operand, VReg};
    let p16 = raw_program(&ProblemSpec::default());
    // Without the input precondition, round 0's update is refused with its reason; round 11's
    // (after the wrap, constant propagation alone) and the wrap's shared zero still match.
    let unassumed = PruneIndexRule { assumptions: vec![], ..PruneIndexRule::for_problem(256) };
    let (kinds, rejected) = prune_sites(&p16, &unassumed);
    assert_eq!(kinds.get(&PruneKind::ZeroUpdate), Some(&32), "{:?}", rejected);
    assert_eq!(kinds.get(&PruneKind::SharedConstant), Some(&31));
    assert_eq!(reason_count(&rejected, "load of the input indices and initial-indices-zero is not declared"), 32, "{:?}", rejected);

    let p2 = raw_program(&ProblemSpec { rounds: 2, ..Default::default() });
    let rule = PruneIndexRule::for_problem(256);
    let first = {
        let g = SsaGraph::new(&p2.ops, &p2.vector_group_lanes);
        rule.find_sites(&g, p2.zero_vreg).0.remove(0)
    };
    let vload = def(&p2, first.zero.unwrap());
    let addr = def(&p2, p2.ops[vload].srcs[0].as_vreg().unwrap());
    // The "index" is loaded from the values array: not provably 0.
    {
        let mut p = clone_program(&p2);
        let values_p = p.ops.iter().position(|o| o.opcode == Opcode::Load && o.srcs.first().and_then(|s| s.as_vreg()).map_or(false, |w| p.ops[def(&p2, w)].produces_constant == Some(6))).unwrap();
        let vp = p.ops[values_p].dest.unwrap();
        p.ops[addr].srcs[0] = Operand::VReg(vp);
        let (kinds, rejected) = prune_sites(&p, &rule);
        assert_eq!(kinds.get(&PruneKind::ZeroUpdate), Some(&31));
        assert_eq!(reason_count(&rejected, "factor is a load of mem[mem[6] + 0], not of the input indices"), 1, "{:?}", rejected);
    }
    // A store before the loads: the indices may have been overwritten.
    {
        // Right before the first index load (group 0's), store a broadcast of 0 to group 0's
        // index address: any store before the loads voids the input precondition.
        let mut p = clone_program(&p2);
        assert_eq!(p.ops.iter().position(|o| o.opcode == Opcode::Vload), Some(vload));
        let z = VReg::vector(p.vreg_count);
        p.vreg_count += 1;
        let b = SsaOp::new(0, Engine::Valu, Opcode::Vbroadcast, Some(z), vec![Operand::VReg(p.zero_vreg.unwrap())]);
        let st = SsaOp::new(0, Engine::Store, Opcode::Vstore, None, vec![Operand::VReg(p.ops[addr].dest.unwrap()), Operand::VReg(z)]);
        p.ops.insert(vload, st);
        p.ops.insert(vload, b);
        let (kinds, rejected) = prune_sites(&p, &rule);
        assert_eq!(kinds.get(&PruneKind::ZeroUpdate), None, "{:?}", rejected);
        assert_eq!(reason_count(&rejected, "load of the indices after a store"), 32, "{:?}", rejected);
    }
    // After the wrap the index is a broadcast of 1, not 0: round 11 is refused (the
    // broadcasts of the same constant are still shared).
    {
        let mut p = clone_program(&p16);
        let one = p.ops.iter().find(|o| o.opcode == Opcode::ConstChoice && o.produces_constant == Some(1)).unwrap().dest.unwrap();
        let z = p.zero_vreg.unwrap();
        for op in &mut p.ops {
            if op.opcode == Opcode::Vbroadcast && op.srcs[0].as_vreg() == Some(z) {
                op.srcs[0] = Operand::VReg(one);
                op.produces_constant = Some(1);
            }
        }
        let (kinds, rejected) = prune_sites(&p, &rule);
        assert_eq!(kinds.get(&PruneKind::ZeroUpdate), Some(&32), "{:?}", rejected);
        assert_eq!(reason_count(&rejected, "factor is the constant 1, not 0"), 32, "{:?}", rejected);
    }
}

#[test]
fn test_values_only_refuses_observed_or_unknown_stores() {
    use crate::types::{Engine, Operand, VReg};
    let p = raw_program(&ProblemSpec { rounds: 2, ..Default::default() });
    let sites = |p: &SsaProgram, rule: &ValuesOnlyRule| {
        let g = SsaGraph::new(&p.ops, &p.vector_group_lanes);
        rule.find_sites(&g, p.zero_vreg)
    };
    let rule = ValuesOnlyRule::for_problem(256);
    let (found, rejected) = sites(&p, &rule);
    assert_eq!(found.len(), 32);
    assert_eq!(found.iter().map(|s| s.group()).collect::<Vec<_>>(), (0..32).collect::<Vec<_>>());
    // The value stores are not index stores.
    assert_eq!(reason_count(&rejected, "store writes mem[mem[6] +"), 32, "{:?}", rejected);
    // Without the declared contract every index store is refused.
    let undeclared = ValuesOnlyRule { assumptions: vec![], ..rule.clone() };
    let (found, rejected) = sites(&p, &undeclared);
    assert!(found.is_empty());
    assert_eq!(reason_count(&rejected, "indices-not-observed is not declared"), 32);
    // Another observer: a later load reads group 1's stored indices back.
    let mut q = clone_program(&p);
    let addr = q.ops[found_store(&p, 1)].srcs[0].clone();
    let v = VReg::vector(q.vreg_count);
    q.vreg_count += 1;
    q.ops.push(SsaOp::new(0, Engine::Load, Opcode::Vload, Some(v), vec![addr]));
    let (found, rejected) = sites(&q, &rule);
    assert_eq!(found.len(), 31);
    assert!(found.iter().all(|s| s.group() != 1));
    assert_eq!(reason_count(&rejected, "may read the stored indices back"), 1, "{:?}", rejected);
    // A load from an address that is not a header pointer plus a constant may alias: every
    // index store before it is refused.
    let mut q = clone_program(&p);
    let input_value = q.ops.iter().filter(|o| o.opcode == Opcode::Vload).nth(1).unwrap().dest.unwrap();
    let s = VReg::scalar(q.vreg_count);
    q.vreg_count += 1;
    // Address = a word of the input values: unknown, so it may alias the indices.
    q.ops.push(SsaOp::new(0, Engine::Load, Opcode::Load, Some(s), vec![Operand::VReg(input_value)]));
    let (found, rejected) = sites(&q, &rule);
    assert!(found.is_empty());
    assert_eq!(reason_count(&rejected, "may read the stored indices back"), 32, "{:?}", rejected);
}

/// Index of the index store of `group` in `p`.
fn found_store(p: &SsaProgram, group: usize) -> usize {
    let g = SsaGraph::new(&p.ops, &p.vector_group_lanes);
    ValuesOnlyRule::for_problem(256).find_sites(&g, p.zero_vreg).0.into_iter().find(|s| s.group() == group).unwrap().store
}

// ----------------------------------------------------------------------------
// values-only: what dies is dead code elimination's (C3, C4, C5)
// ----------------------------------------------------------------------------

/// Ops values-only made dead, by (round tag, opcode); ops outside index steps under
/// `(None, opcode)`. Tags label the ops here; the rule never reads them.
fn values_only_dead(spec: &ProblemSpec) -> (BTreeMap<(Option<u32>, String), usize>, RuleReport) {
    let p = raw_program(spec);
    let (_, report) = apply_values_only(clone_program(&p), &ValuesOnlyRule::for_problem(spec.batch_size), |_| true).unwrap();
    let mut out = BTreeMap::new();
    for &(i, opcode, _) in &report.surgery.cascaded {
        let round = p.ops[i].region.filter(|t| t.kind == RegionKind::IndexStep).and_then(|t| t.round);
        *out.entry((round, format!("{:?}", opcode))).or_insert(0) += 1;
    }
    (out, report)
}

fn dead(map: &BTreeMap<(Option<u32>, String), usize>, round: u32, opcode: Opcode) -> usize {
    map.get(&(Some(round), format!("{:?}", opcode))).copied().unwrap_or(0)
}

#[test]
fn test_values_only_saving_comes_from_dce() {
    use Opcode::{And, MultiplyAdd, Sub, Vbroadcast};
    let d2 = ProblemSpec { rounds: 2, ..Default::default() };
    // Zero-based, standard access at depth 1 reads the index: only round 1's step dies.
    let (m, _) = values_only_dead(&d2);
    eprintln!("2r default: {:?}", m);
    assert_eq!((dead(&m, 1, And), dead(&m, 1, Sub), dead(&m, 1, MultiplyAdd)), (32, 32, 32));
    assert_eq!(dead(&m, 0, MultiplyAdd), 0);
    // C3: bit-driven at depth 1 does not read the index, so round 0's update dies too, and with
    // it the initial index load (rule 7 without prune); C5: round 0's branch bit survives (the
    // round-1 access reads it).
    let (m, _) = values_only_dead(&ProblemSpec { cached_select: CachedSelect::Bits, ..d2.clone() });
    eprintln!("2r bits: {:?}", m);
    assert_eq!((dead(&m, 0, Sub), dead(&m, 0, MultiplyAdd), dead(&m, 0, And)), (32, 32, 0));
    assert_eq!(m.get(&(None, "Vload".to_string())), Some(&32));
    // C4: mirrored, the final L_d - n SUB dies (with its L_d vector), and round 1's step.
    let (m, _) = values_only_dead(&ProblemSpec { index_form: IndexForm::Mirrored, ..d2.clone() });
    eprintln!("2r mirrored: {:?}", m);
    assert_eq!(m.get(&(None, "Sub".to_string())), Some(&32));
    assert!(m.get(&(None, "Vbroadcast".to_string())).copied().unwrap_or(0) >= 1);
    assert_eq!((dead(&m, 1, And), dead(&m, 1, MultiplyAdd)), (32, 32));

    let d16 = ProblemSpec::default();
    // 16 rounds, default flags: round 15's access (depth 4, gathered) reads the index, so
    // only round 15's step dies.
    let (m, _) = values_only_dead(&d16);
    eprintln!("16r default: {:?}", m);
    assert_eq!((dead(&m, 15, And), dead(&m, 15, Sub), dead(&m, 15, MultiplyAdd)), (32, 32, 32));
    assert_eq!((dead(&m, 14, Sub), dead(&m, 14, MultiplyAdd)), (0, 0));
    // Bit-driven at depths 0-4 with the index source: depth 4 takes its bits from the index
    // before round 14's update plus round 14's branch bit, so round 14's update dies too but
    // its branch bit survives (C3, C5); round 13's step lives.
    let (m, _) = values_only_dead(&uniform_plan(&d16, AccessEncoding::Bits));
    eprintln!("16r bits (index source): {:?}", m);
    assert_eq!((dead(&m, 15, And), dead(&m, 15, Sub), dead(&m, 15, MultiplyAdd)), (32, 32, 32));
    assert_eq!((dead(&m, 14, And), dead(&m, 14, Sub), dead(&m, 14, MultiplyAdd)), (0, 32, 32));
    assert_eq!((dead(&m, 13, Sub), dead(&m, 13, MultiplyAdd)), (0, 0));
    // Path bits: no access after the wrap reads the index, so rounds 11-15's updates and the
    // wrap's zeros die; the branch bits of rounds 11-14 survive (C5), round 15's dies.
    let path = ProblemSpec { select_bit_source: SelectBitSource::Path, ..d16 };
    let (m, _) = values_only_dead(&uniform_plan(&path, AccessEncoding::Bits));
    eprintln!("16r bits (path source): {:?}", m);
    for r in 11..=15 {
        assert_eq!(dead(&m, r, MultiplyAdd), 32, "round {}", r);
        assert_eq!(dead(&m, r, Sub), 32, "round {}", r);
        assert_eq!(dead(&m, r, And), if r == 15 { 32 } else { 0 }, "round {}", r);
    }
    assert_eq!(dead(&m, 10, Vbroadcast), 32);
    assert_eq!(dead(&m, 10, MultiplyAdd), 0);
}

/// `base` with a plan: `Bits` at depth 0, `e` at depths 1-4, gather below.
fn uniform_plan(base: &ProblemSpec, e: AccessEncoding) -> ProblemSpec {
    let plan: Vec<Vec<AccessEncoding>> = (0..base.rounds)
        .map(|r| {
            let d = base.round_depth(r);
            let enc = if d == 0 { AccessEncoding::Bits } else if d <= 4 { e } else { AccessEncoding::Gather };
            vec![enc; base.n_groups()]
        })
        .collect();
    ProblemSpec { access_plan: Some(plan), ..base.clone() }
}

/// Delta table: values-only per index form, last-round encoding and bit source, from the
/// rewrite (per-site class when that store alone is removed, the shared remainder, total).
#[test]
fn test_values_only_delta_table() {
    use AccessEncoding::*;
    let mut rows = Vec::new();
    for rounds in [2usize, 16] {
        for form in [IndexForm::ZeroBased, IndexForm::Mirrored] {
            for source in [SelectBitSource::Index, SelectBitSource::Path] {
                let base = ProblemSpec { rounds, index_form: form, select_bit_source: source, ..Default::default() };
                let mut specs: Vec<(String, ProblemSpec)> = vec![("standard".into(), base.clone())];
                for e in [Gather, Bits, BitsFlow, Mobius] {
                    specs.push((format!("{:?}", e), uniform_plan(&base, e)));
                }
                for (name, spec) in specs {
                    let (_, report) = values_only_dead(&spec);
                    let classes: Vec<String> =
                        report.site_classes.iter().map(|((_, d), n)| format!("{} x [{}]", n, format_counts(d))).collect();
                    rows.push(format!(
                        "| {} | {:?} | {:?} | {} | {} | {} | {} |",
                        rounds,
                        form,
                        source,
                        name,
                        classes.join("; "),
                        format_counts(&report.cascade_delta),
                        format_counts(&report.total_delta)
                    ));
                    // Each site's own delta is uniform over groups when every group uses one
                    // encoding.
                    assert_eq!(report.site_classes.len(), 1, "{} {:?}", name, report.site_classes);
                }
            }
        }
    }
    // Pinned rows (the measured table, not a formula): the per-site, shared and total deltas.
    // Since #79 DCE drops alternatives that name a constant it removed: where values-only kills
    // the pointer load's address constant 5, 6 loses `1 + 5` and is no longer ALU-derivable,
    // so one more constant moves from const_alu to const (+1 / -2 instead of -1).
    for row in [
        "| 2 | ZeroBased | Index | standard | 32 x [vec_fixed -1, vec_shiftable -2, alu -1, store -1] | 0 | vec_fixed -32, vec_shiftable -64, alu -32, store -32 |",
        "| 2 | ZeroBased | Index | Bits | 32 x [vec_fixed -2, vec_shiftable -3, alu -2, load -1, store -1] | vec_fixed -1, load -1, const +1, const_alu -2 | vec_fixed -65, vec_shiftable -96, alu -64, load -33, store -32, const +1, const_alu -2 |",
        "| 2 | Mirrored | Index | standard | 32 x [vec_fixed -1, vec_shiftable -2, alu -1, store -1] | vec_fixed -2, alu -1, load -1, const +1, const_alu -2 | vec_fixed -34, vec_shiftable -64, alu -33, load -1, store -32, const +1, const_alu -2 |",
        "| 16 | ZeroBased | Index | Bits | 32 x [vec_fixed -2, vec_shiftable -3, alu -1, store -1] | 0 | vec_fixed -64, vec_shiftable -96, alu -32, store -32 |",
        "| 16 | ZeroBased | Path | Bits | 32 x [vec_fixed -6, vec_shiftable -6, alu -1, store -1] | 0 | vec_fixed -192, vec_shiftable -192, alu -32, store -32 |",
        "| 16 | Mirrored | Index | Bits | 32 x [vec_fixed -2, vec_shiftable -2, alu -1, store -1] | vec_fixed -1, load -1, const +1, const_alu -2 | vec_fixed -65, vec_shiftable -64, alu -32, load -1, store -32, const +1, const_alu -2 |",
    ] {
        assert!(rows.iter().any(|r| r == row), "missing or changed row: {}", row);
    }
    eprintln!("| rounds | form | bit source | depths 1-4 | per site (alone) | shared | total |");
    eprintln!("|---|---|---|---|---|---|---|");
    for r in rows {
        eprintln!("{}", r);
    }
}

// ----------------------------------------------------------------------------
// Scheduling inputs: peak live scratch, budget, lane map (32 groups)
// ----------------------------------------------------------------------------

/// Run the greedy pipeline on `program`, applying `transforms` in the pass.
///
/// # Returns
/// `(cycles, lowered ops, peak live words, budget exceeded, lane-map entries, lane-map entries
/// whose key or a lane is still read or written by a scheduled op)`.
fn greedy(program: SsaProgram, transforms: Vec<TransformKind>) -> (usize, usize, usize, bool, usize, usize) {
    use crate::kernel_builder::config::CompilerConfig;
    use crate::kernel_builder::pipeline::{PipelineConfig, PipelineContext};
    let config = PipelineConfig {
        compiler: CompilerConfig { transforms, transform_batch_size: Some(256), ..Default::default() },
        verbose: false,
        ..Default::default()
    };
    let mut ctx = PipelineContext::new(program, config);
    ctx.run_greedy_pipeline(true).unwrap();
    let used: std::collections::HashSet<crate::types::VReg> =
        ctx.ops.iter().flat_map(|o| o.reads().chain(o.dest)).collect();
    let lanes = ctx.vector_group_lanes.len();
    let live_lanes = ctx
        .vector_group_lanes
        .iter()
        .filter(|(k, ls)| used.contains(k) || ls.iter().any(|l| used.contains(l)))
        .count();
    let r = ctx.into_result().unwrap();
    (r.cycles, r.ops.len(), r.address_result.peak_live, r.address_result.budget_exceeded, lanes, live_lanes)
}

#[test]
fn test_index_transforms_schedule_like_the_flags_at_32_groups() {
    use TransformKind::*;
    let specs = [
        ("16r default", ProblemSpec::default()),
        ("16r mirrored bits", ProblemSpec { index_form: IndexForm::Mirrored, cached_select: CachedSelect::Bits, ..Default::default() }),
        ("16r bits path", ProblemSpec { cached_select: CachedSelect::Bits, select_bit_source: SelectBitSource::Path, ..Default::default() }),
    ];
    for (label, spec) in specs {
        for ts in [vec![PruneIndex], vec![ValuesOnly], vec![PruneIndex, ValuesOnly]] {
            let rewritten = greedy(raw_program(&spec), ts.clone());
            let flagged = greedy(raw_program(&with_flags(&spec, &ts)), vec![]);
            eprintln!(
                "{} {:?}: (cycles, ops, peak, over budget, lane entries, live lane entries) rewritten {:?} flags {:?}",
                label, ts, rewritten, flagged
            );
            assert_eq!(rewritten.0, flagged.0, "{} {:?}: cycles", label, ts);
            assert_eq!(rewritten.1, flagged.1, "{} {:?}: lowered ops", label, ts);
            assert_eq!(rewritten.2, flagged.2, "{} {:?}: peak live words", label, ts);
            assert_eq!(rewritten.3, flagged.3, "{} {:?}: budget", label, ts);
            assert_eq!(rewritten.5, flagged.5, "{} {:?}: live lane entries", label, ts);
            assert_eq!(rewritten.4 - rewritten.5, flagged.4 - flagged.5, "{} {:?}: stale lane entries", label, ts);
        }
    }
}

// ----------------------------------------------------------------------------
// Near misses of prune_index.rs's match_at and values-only's keep_live check (review of #77)
// ----------------------------------------------------------------------------

/// Apply `mutate` to the first site of `kind` in the 16-round default program and return that
/// op's rejection.
fn prune_near_miss(kind: PruneKind, mutate: impl Fn(&mut SsaProgram, usize)) -> String {
    let mut p = raw_program(&ProblemSpec::default());
    let rule = PruneIndexRule::for_problem(256);
    let op = {
        let g = SsaGraph::new(&p.ops, &p.vector_group_lanes);
        rule.find_sites(&g, p.zero_vreg).0.into_iter().find(|s| s.kind == kind).unwrap().op
    };
    mutate(&mut p, op);
    let g = SsaGraph::new(&p.ops, &p.vector_group_lanes);
    let values = super::values::ValueFacts::compute(&g, p.zero_vreg, &rule.layout, &rule.assumptions);
    match rule.match_at(&g, &values, op) {
        Ok(s) => panic!("{:?} site at op {} still matched: {:?}", kind, op, s.facts),
        Err(e) => e.0,
    }
}

#[test]
fn test_prune_index_near_misses_are_refused() {
    for kind in [PruneKind::ZeroUpdate, PruneKind::SharedConstant] {
        // keep_live on the op itself: the list would be lost with the op.
        let why = prune_near_miss(kind, |p, op| {
            let other = p.ops[op - 1].dest.or(p.ops[0].dest).unwrap();
            p.ops[op].keep_live = Some(vec![other]);
        });
        assert!(why.contains("keep_live"), "{:?}: {}", kind, why);
        // The result is a vector group key: renaming it would orphan its lanes.
        let why = prune_near_miss(kind, |p, op| {
            let out = p.ops[op].dest.unwrap();
            let lanes: Vec<crate::types::VReg> = (0..8).map(|k| crate::types::VReg::scalar(p.vreg_count + k)).collect();
            p.vreg_count += 8;
            p.vector_group_lanes.insert(out, lanes);
        });
        assert!(why.contains("vector group"), "{:?}: {}", kind, why);
        // The result is defined again later: not single assignment.
        let why = prune_near_miss(kind, |p, op| {
            let out = p.ops[op].dest.unwrap();
            let mut again = p.ops[op].clone();
            again.id = p.ops.len();
            again.dest = Some(out);
            p.ops.push(again);
        });
        assert!(why.contains("more than once"), "{:?}: {}", kind, why);
        // A trace id: the traced value would disappear.
        let why = prune_near_miss(kind, |p, op| p.ops[op].trace_id = Some("t".into()));
        assert!(why.contains("trace id"), "{:?}: {}", kind, why);
    }
}

#[test]
fn test_values_only_refuses_a_store_carrying_keep_live() {
    let mut p = raw_program(&ProblemSpec { rounds: 2, ..Default::default() });
    let store = found_store(&p, 3);
    let v = p.ops[0].dest.unwrap();
    p.ops[store].keep_live = Some(vec![v]);
    let g = SsaGraph::new(&p.ops, &p.vector_group_lanes);
    let (sites, rejected) = ValuesOnlyRule::for_problem(256).find_sites(&g, p.zero_vreg);
    assert_eq!(sites.len(), 31);
    assert!(!sites.iter().any(|s| s.store == store));
    assert_eq!(reason_count(&rejected, "keep_live"), 1, "{:?}", rejected);
}

#[test]
fn test_prune_index_match_site_agrees_with_find_sites() {
    // 16 rounds: round 11's updates and the wrap's shared zeros are sites only with the zero
    // register known; match_site (rule.with_zero) must see them as find_sites does.
    use super::rule::{Proposal, Rule};
    let p = raw_program(&ProblemSpec::default());
    let rule = PruneIndexRule::for_problem(256).with_zero(p.zero_vreg);
    let g = SsaGraph::new(&p.ops, &p.vector_group_lanes);
    let (sites, _) = rule.find_sites(&g, None);
    assert_eq!(sites.len(), 31 + 64);
    for s in &sites {
        let proposal = Proposal { region_id: None, round: None, group: None, depth: None, ops: vec![s.op] };
        assert_eq!(rule.match_site(&g, &proposal).as_ref(), Ok(s), "op {}", s.op);
    }
}

#[test]
fn test_rule_reports_carry_the_assumptions_used() {
    // prune-index relies on InitialIndicesZero only for round 0's update; values-only always
    // relies on IndicesNotObserved. The reports list them (and the layout), each once.
    let p = raw_program(&ProblemSpec { rounds: 2, ..Default::default() });
    let (_, prune) = super::prune_index::apply_prune_index(clone_program(&p), &PruneIndexRule::for_problem(256), |_| true).unwrap();
    assert_eq!(prune.assumptions(), vec!["initial-indices-zero"]);
    assert!(prune.facts.iter().any(|f| f.name == "memory_layout"));
    assert!(prune.summary().contains("assumes initial-indices-zero"));
    let (_, vo) = apply_values_only(clone_program(&p), &ValuesOnlyRule::for_problem(256), |_| true).unwrap();
    assert_eq!(vo.assumptions(), vec!["indices-not-observed"]);
    // A prune of a program whose zeros come from constants only (after the wrap, no round 0
    // site selected) assumes nothing.
    let p16 = raw_program(&ProblemSpec::default());
    let rule = PruneIndexRule::for_problem(256);
    let (_, post_wrap) = super::prune_index::apply_prune_index(clone_program(&p16), &rule, |s| s.round.map_or(false, |r| r >= 10)).unwrap();
    assert!(post_wrap.n_sites() > 0);
    assert!(post_wrap.assumptions().is_empty(), "{:?}", post_wrap.facts);
}
