//! Tests of rule `select-bits-path`: it reproduces the `--select-bit-source path` emission
//! from the index emission (alone and with the other transforms), finds its sites without
//! tags, needs `InitialIndicesZero` only where a zero-based chain starts at the loaded index,
//! and refuses perturbed input with named reasons; rewritten programs keep their outputs.

use std::collections::BTreeMap;

use super::graph::SsaGraph;
use super::select_bits::{apply_select_bits, SelectBitsRule, SelectBitsSite};
use super::tests::{canonical_ops, clone_program, flow90_plan, raw_program};
use super::tests_index::check_parity;
use super::tests_interp::outputs;
use crate::kernel_builder::config::{CachedSelect, IndexForm, ProblemSpec, SelectBitSource, TransformKind};
use crate::kernel_builder::pipeline::passes::transform::apply_transform;
use crate::kernel_builder::SsaProgram;
use crate::types::{Engine, Opcode, Operand, SsaOp, VReg};

const BATCH: usize = 256;

/// Bit-driven configurations that select at depth 3 (and 4 with `depth4`).
fn bit_configs(rounds: usize) -> Vec<(String, ProblemSpec)> {
    let mut out = Vec::new();
    for (form, fname) in [(IndexForm::ZeroBased, "zero-based"), (IndexForm::Mirrored, "mirrored")] {
        for (mode, mname) in [(CachedSelect::Bits, "bits"), (CachedSelect::BitsFlow, "bits-flow"), (CachedSelect::Mobius, "mobius")] {
            for depth in [3u32, 4] {
                let spec = ProblemSpec { rounds, index_form: form, cached_select: mode, tree_reduce_max_depth: depth, ..Default::default() };
                out.push((format!("{}r {} {} d{}", rounds, fname, mname, depth), spec));
            }
        }
    }
    if rounds == 16 {
        let m = ProblemSpec { rounds, index_form: IndexForm::Mirrored, ..Default::default() };
        out.push(("16r flow90 plan".into(), ProblemSpec { access_plan: Some(flow90_plan(&m)), ..m }));
    }
    out
}

fn sites(p: &SsaProgram, rule: &SelectBitsRule) -> (Vec<SelectBitsSite>, BTreeMap<String, usize>) {
    let g = SsaGraph::new(&p.ops, &p.vector_group_lanes);
    rule.find_sites(&g, p.zero_vreg)
}

fn reason_count(rejected: &BTreeMap<String, usize>, text: &str) -> usize {
    rejected.iter().filter(|(r, _)| r.contains(text)).map(|(_, n)| n).sum()
}

fn rule() -> SelectBitsRule {
    SelectBitsRule::for_problem(BATCH)
}

// ----------------------------------------------------------------------------
// Parity with the flag emission
// ----------------------------------------------------------------------------

#[test]
fn test_select_bits_parity_5_rounds() {
    // Rounds 3 and 4 select at depths 3 and 4 on the first pass (zero-based: the chain starts
    // at the loaded index, under InitialIndicesZero).
    for (label, spec) in bit_configs(5) {
        let parity = check_parity(&spec, &[TransformKind::SelectBitsPath], &label);
        assert!(parity.summaries[0].contains("bit-read"), "{}: {}", label, parity.summaries[0]);
    }
}

#[test]
fn test_select_bits_parity_16_rounds() {
    for (label, spec) in bit_configs(16) {
        check_parity(&spec, &[TransformKind::SelectBitsPath], &label);
    }
}

#[test]
fn test_select_bits_composes_with_the_other_transforms_16_rounds() {
    use TransformKind::*;
    let specs: Vec<(String, ProblemSpec)> = bit_configs(16)
        .into_iter()
        .filter(|(l, _)| l.contains("bits d4") || l.contains("flow90"))
        .collect();
    for (label, spec) in specs {
        for set in [vec![SelectBitsPath, ValuesOnly], vec![PruneIndex, SelectBitsPath], vec![HashMerge, PruneIndex, ValuesOnly, SelectBitsPath], vec![SelectBitsPath, HashMerge, ValuesOnly, PruneIndex]] {
            check_parity(&spec, &set, &label);
        }
    }
}

#[test]
fn test_select_bits_site_counts() {
    // Depth 3: 2 older bits (AND, SHR); depth 4: 3 (AND, AND, SHR). Rounds 3, 4, 14, 15.
    for form in [IndexForm::ZeroBased, IndexForm::Mirrored] {
        let spec = ProblemSpec { index_form: form, cached_select: CachedSelect::Bits, tree_reduce_max_depth: 4, ..Default::default() };
        let p = raw_program(&spec);
        let (found, rejected) = sites(&p, &rule());
        assert_eq!(found.len(), 32 * (2 + 3 + 2 + 3), "{:?}: {:?}", form, rejected);
        let mut by_round: BTreeMap<u32, usize> = BTreeMap::new();
        for s in &found {
            *by_round.entry(s.round.unwrap()).or_default() += 1;
        }
        assert_eq!(by_round, BTreeMap::from([(3, 64), (4, 96), (14, 64), (15, 96)]), "{:?}", form);
        // The first pass of a zero-based chain relies on the declared input precondition;
        // after the wrap, and in the mirrored form, nothing is assumed.
        let assumed = found.iter().filter(|s| s.facts.iter().any(|f| f.name == "initial-indices-zero")).count();
        assert_eq!(assumed, if form == IndexForm::ZeroBased { 160 } else { 0 }, "{:?}", form);
        // Every site equals a bit that is `x & 1` of a non-index value.
        for s in &found {
            assert_eq!(p.ops[s.bit_def].opcode, Opcode::And);
        }
    }
}

#[test]
fn test_select_bits_finds_nothing_without_bit_extraction() {
    // Standard selection, path bits, and depth <= 2 bit-driven access extract nothing.
    let none = [
        ProblemSpec::default(),
        ProblemSpec { index_form: IndexForm::Mirrored, tree_reduce_max_depth: 4, ..Default::default() },
        ProblemSpec { cached_select: CachedSelect::Bits, tree_reduce_max_depth: 2, ..Default::default() },
        ProblemSpec { cached_select: CachedSelect::Bits, tree_reduce_max_depth: 4, select_bit_source: SelectBitSource::Path, ..Default::default() },
        ProblemSpec { index_form: IndexForm::Mirrored, cached_select: CachedSelect::Mobius, tree_reduce_max_depth: 4, select_bit_source: SelectBitSource::Path, ..Default::default() },
    ];
    for spec in none {
        let p = raw_program(&spec);
        let (found, _) = sites(&p, &rule());
        assert!(found.is_empty(), "{:?} {:?} {:?}: {:?}", spec.index_form, spec.cached_select, spec.select_bit_source, &found[..1]);
        let (q, report) = apply_select_bits(clone_program(&p), &rule(), |_| true).unwrap();
        assert_eq!(report.n_sites(), 0);
        assert_eq!(canonical_ops(&q), canonical_ops(&p));
    }
}

#[test]
fn test_select_bits_finds_the_same_sites_without_tags() {
    let spec = ProblemSpec { cached_select: CachedSelect::Bits, tree_reduce_max_depth: 4, ..Default::default() };
    let tagged = raw_program(&spec);
    let mut bare = clone_program(&tagged);
    for op in &mut bare.ops {
        op.region = None;
    }
    let shape = |p: &SsaProgram| sites(p, &rule()).0.iter().map(|s| (s.op, s.output, s.bit)).collect::<Vec<_>>();
    assert_eq!(shape(&bare).len(), 320);
    assert_eq!(shape(&bare), shape(&tagged));
    let (a, _) = apply_transform(clone_program(&bare), TransformKind::SelectBitsPath, Some(BATCH)).unwrap();
    let (mut b, _) = apply_transform(tagged, TransformKind::SelectBitsPath, Some(BATCH)).unwrap();
    for op in &mut b.ops {
        op.region = None;
    }
    assert_eq!(canonical_ops(&a), canonical_ops(&b));
}

// ----------------------------------------------------------------------------
// Values: the rewrite keeps the outputs (full memory image, against the reference)
// ----------------------------------------------------------------------------

/// `after` computes the same final indices and values as `before`, which agrees with the
/// reference kernel.
fn assert_same_outputs(before: &SsaProgram, after: &SsaProgram, spec: &ProblemSpec, label: &str) {
    let seeds = [3u64, 99, 2024];
    let b = outputs(before, spec, &seeds);
    let a = outputs(after, spec, &seeds);
    for (k, ((bo, expected), (ao, _))) in b.iter().zip(&a).enumerate() {
        assert_eq!(bo, expected, "{}: the input program disagrees with the reference (seed {})", label, seeds[k]);
        assert_eq!(ao, bo, "{}: outputs differ after the rewrite (seed {})", label, seeds[k]);
    }
}

#[test]
fn test_select_bits_preserves_outputs() {
    for rounds in [5, 16] {
        for (label, spec) in bit_configs(rounds) {
            let p = raw_program(&spec);
            let (q, report) = apply_transform(clone_program(&p), TransformKind::SelectBitsPath, Some(BATCH)).unwrap();
            assert!(report.sites > 0, "{}", label);
            assert_same_outputs(&p, &q, &spec, &label);
        }
    }
}

// ----------------------------------------------------------------------------
// Perturbed input: refused with named reasons, never miscompiled
// ----------------------------------------------------------------------------

/// The 5-round program the perturbations start from.
fn spec5(form: IndexForm) -> ProblemSpec {
    ProblemSpec { rounds: 5, index_form: form, cached_select: CachedSelect::Bits, tree_reduce_max_depth: 4, ..Default::default() }
}

/// Index of the op defining `v`.
fn def(p: &SsaProgram, v: VReg) -> usize {
    p.ops.iter().position(|o| o.dest == Some(v)).unwrap()
}

/// A vector VReg holding the broadcast constant `k` (the first one).
fn vconst(p: &SsaProgram, k: i64) -> VReg {
    let g = SsaGraph::new(&p.ops, &p.vector_group_lanes);
    p.ops.iter().filter_map(|o| o.dest).find(|&d| d.is_vector() && g.broadcast_const(d) == Some(k)).unwrap()
}

/// Round 3 group 0's sites (depth 3): the AND (newest older bit) and the SHR (oldest).
fn first_sites(p: &SsaProgram) -> (SelectBitsSite, SelectBitsSite) {
    let found = sites(p, &rule()).0;
    let r3: Vec<_> = found.into_iter().filter(|s| s.round == Some(3) && s.group == Some(0)).collect();
    assert_eq!(r3.len(), 2);
    let and = r3.iter().find(|s| p.ops[s.op].opcode == Opcode::And).unwrap().clone();
    let shr = r3.iter().find(|s| p.ops[s.op].opcode == Opcode::Shr).unwrap().clone();
    (and, shr)
}

/// Apply the rule to `p` (perturbed from `spec`) and check the outputs did not change.
fn rewrite_and_check(before: &SsaProgram, spec: &ProblemSpec, label: &str) -> (SsaProgram, super::report::RuleReport) {
    let (after, report) = apply_select_bits(clone_program(before), &rule(), |_| true).unwrap();
    assert_same_outputs(before, &after, spec, label);
    (after, report)
}

#[test]
fn test_select_bits_refuses_a_wrong_shift_amount() {
    let spec = spec5(IndexForm::Mirrored);
    let mut p = raw_program(&spec);
    let (_, shr) = first_sites(&p);
    let v2 = vconst(&p, 2);
    // x >> 2 is not the oldest bit: refused, the AND site of the same group still matches.
    p.ops[shr.op].srcs[1] = Operand::VReg(v2);
    let (found, rejected) = sites(&p, &rule());
    assert_eq!(found.len(), 160 - 1);
    assert!(!found.iter().any(|s| s.op == shr.op));
    assert_eq!(reason_count(&rejected, "shift right by Some(2), not by the constant 1"), 1, "{:?}", rejected);
}

#[test]
fn test_select_bits_refuses_a_wrong_and_mask() {
    let spec = spec5(IndexForm::Mirrored);
    let mut p = raw_program(&spec);
    let (and, _) = first_sites(&p);
    let v2 = vconst(&p, 2);
    p.ops[and.op].srcs[1] = Operand::VReg(v2);
    let (found, rejected) = sites(&p, &rule());
    assert_eq!(found.len(), 159);
    assert_eq!(reason_count(&rejected, "AND with neither operand the constant 1"), 1, "{:?}", rejected);
}

#[test]
fn test_select_bits_refuses_a_wrong_index_multiplier() {
    // Round 2's index step of group 0 multiplies by 1 instead of 2: the position it builds
    // is not the bits in binary, so group 0's extractions in rounds 3 and 4 are refused. The
    // program is changed (no longer the kernel), but the rewrite must still keep its values.
    let spec = spec5(IndexForm::Mirrored);
    let mut p = raw_program(&spec);
    let (and, _) = first_sites(&p);
    let prev = p.ops[and.op].srcs[0].as_vreg().unwrap();
    let step = def(&p, prev);
    assert_eq!(p.ops[step].opcode, Opcode::MultiplyAdd);
    let v1 = vconst(&p, 1);
    let k = p.ops[step].srcs.iter().position(|s| s.as_vreg() == Some(vconst(&p, 2))).unwrap();
    p.ops[step].srcs[k] = Operand::VReg(v1);
    let (found, rejected) = sites(&p, &rule());
    assert_eq!(found.len(), 160 - 5, "{:?}", rejected);
    assert!(reason_count(&rejected, "multiply_add with no factor the constant 2") >= 2, "{:?}", rejected);
    let (_, report) = apply_select_bits(clone_program(&p), &rule(), |_| true).unwrap();
    assert_eq!(report.n_sites(), 155);
}

#[test]
fn test_select_bits_refuses_a_bit_that_is_not_boolean() {
    // Group 0's round-1 branch bit becomes v & 2 (0 or 2, not a bit): every position built
    // on it is refused (rounds 3 and 4 of group 0).
    let spec = spec5(IndexForm::Mirrored);
    let mut p = raw_program(&spec);
    let (_, shr) = first_sites(&p);
    let b0 = shr.bit;
    let (and_b1, _) = first_sites(&p);
    let b1_def = and_b1.bit_def;
    assert_ne!(def(&p, b0), b1_def);
    let v2 = vconst(&p, 2);
    let k = p.ops[b1_def].srcs.iter().position(|s| s.as_vreg() == Some(vconst(&p, 1))).unwrap();
    p.ops[b1_def].srcs[k] = Operand::VReg(v2);
    let (found, rejected) = sites(&p, &rule());
    assert_eq!(found.len(), 155, "{:?}", rejected);
    assert!(reason_count(&rejected, "AND with neither operand the constant 1") >= 1, "{:?}", rejected);
}

#[test]
fn test_select_bits_checks_the_constant_annotation() {
    // The scalar constant 1 says 1 in its annotation and 3 in its immediate: the matcher reads
    // the immediate (through `const_value`), so no AND with it is a bit and nothing matches.
    let spec = spec5(IndexForm::Mirrored);
    let mut p = raw_program(&spec);
    let v1 = vconst(&p, 1);
    let s1 = p.ops[def(&p, v1)].srcs[0].as_vreg().unwrap();
    let c = def(&p, s1);
    p.ops[c].srcs[0] = Operand::Imm(3);
    let (found, rejected) = sites(&p, &rule());
    assert!(found.is_empty(), "{:?}", &found[..1]);
    assert!(reason_count(&rejected, "AND with neither operand the constant 1") > 0, "{:?}", rejected);
}

#[test]
fn test_select_bits_refuses_a_wrong_mirroring_constant() {
    // Zero-based: the SUB before the extraction uses 7 instead of L_2 = 6.
    let spec = spec5(IndexForm::ZeroBased);
    let mut p = raw_program(&spec);
    let (and, _) = first_sites(&p);
    let mirrored = p.ops[and.op].srcs[0].as_vreg().unwrap();
    let sub = def(&p, mirrored);
    assert_eq!(p.ops[sub].opcode, Opcode::Sub);
    {
        let g = SsaGraph::new(&p.ops, &p.vector_group_lanes);
        let r = rule();
        let values = super::values::ValueFacts::compute(&g, p.zero_vreg, &r.layout, &r.assumptions);
        assert_eq!(values.value(&g, p.ops[sub].srcs[0].as_vreg().unwrap()), Some(6));
    }
    let (s7, v7) = (VReg::scalar(p.vreg_count), VReg::vector(p.vreg_count + 1));
    p.vreg_count += 2;
    let mut c7 = SsaOp::new(0, Engine::Load, Opcode::ConstChoice, Some(s7), vec![Operand::Imm(7)]);
    c7.produces_constant = Some(7);
    let mut b7 = SsaOp::new(0, Engine::Valu, Opcode::Vbroadcast, Some(v7), vec![Operand::VReg(s7)]);
    b7.produces_constant = Some(7);
    p.ops[sub].srcs[0] = Operand::VReg(v7);
    p.ops.insert(sub, b7);
    p.ops.insert(sub, c7);
    for (i, op) in p.ops.iter_mut().enumerate() {
        op.id = i;
    }
    p.vreg_def = p.ops.iter().filter_map(|o| o.dest.map(|d| (d, o.id))).collect();
    let (found, rejected) = sites(&p, &rule());
    assert_eq!(found.len(), 158, "{:?}", rejected);
    assert!(reason_count(&rejected, "SUB constant 7 is not L_2 = 6") >= 2, "{:?}", rejected);
}

#[test]
fn test_select_bits_needs_initial_indices_zero_for_the_first_zero_based_pass() {
    let spec = ProblemSpec { cached_select: CachedSelect::Bits, tree_reduce_max_depth: 4, ..Default::default() };
    let p = raw_program(&spec);
    let unassumed = SelectBitsRule { assumptions: vec![], ..rule() };
    let (found, rejected) = sites(&p, &unassumed);
    // After the wrap the chain starts at a broadcast of the zero register: still matched.
    assert_eq!(found.len(), 160);
    assert!(found.iter().all(|s| s.round.unwrap() >= 14));
    assert!(reason_count(&rejected, "initial-indices-zero is not declared") >= 32, "{:?}", rejected);
    // Mirrored needs nothing: its chain starts at the bit.
    let m = raw_program(&ProblemSpec { index_form: IndexForm::Mirrored, ..spec });
    assert_eq!(sites(&m, &unassumed).0.len(), 320);
}

#[test]
fn test_select_bits_with_an_extra_reader_of_an_intermediate() {
    // At depth 4 the extraction keeps an intermediate position (x >> 1) that is not a site.
    // Give it an extra, value-neutral reader feeding the hash input (a ^ (y ^ y)): the rewrite
    // must keep it (and what it reads) and the outputs must not change.
    let spec = spec5(IndexForm::Mirrored);
    let mut p = raw_program(&spec);
    let found = sites(&p, &rule()).0;
    let r4: Vec<_> = found.iter().filter(|s| s.round == Some(4) && s.group == Some(0)).collect();
    assert_eq!(r4.len(), 3);
    // The intermediate: a SHR whose result is read by a site AND.
    let inter = p
        .ops
        .iter()
        .position(|o| o.opcode == Opcode::Shr && r4.iter().any(|s| p.ops[s.op].srcs.first().and_then(|x| x.as_vreg()) == o.dest))
        .expect("depth-4 intermediate shift");
    let y = p.ops[inter].dest.unwrap();
    let xor = (inter + 1..p.ops.len()).find(|&i| p.ops[i].opcode == Opcode::Xor && p.ops[i].engine == Engine::Valu).unwrap();
    let a = p.ops[xor].srcs[0].as_vreg().unwrap();
    let (z, w) = (VReg::vector(p.vreg_count), VReg::vector(p.vreg_count + 1));
    p.vreg_count += 2;
    let zop = SsaOp::new(0, Engine::Valu, Opcode::Xor, Some(z), vec![Operand::VReg(y), Operand::VReg(y)]);
    let wop = SsaOp::new(0, Engine::Valu, Opcode::Xor, Some(w), vec![Operand::VReg(a), Operand::VReg(z)]);
    p.ops[xor].srcs[0] = Operand::VReg(w);
    p.ops.insert(xor, wop);
    p.ops.insert(xor, zop);
    for (i, op) in p.ops.iter_mut().enumerate() {
        op.id = i;
    }
    p.vreg_def = p.ops.iter().filter_map(|o| o.dest.map(|d| (d, o.id))).collect();
    let v1 = vconst(&p, 1);
    let shifts_by_one = |q: &SsaProgram| q.ops.iter().filter(|o| o.opcode == Opcode::Shr && o.srcs[1].as_vreg() == Some(v1)).count();
    assert_eq!(shifts_by_one(&p), 32 * 3);
    let (after, report) = rewrite_and_check(&p, &spec, "extra reader");
    assert_eq!(report.n_sites(), 160);
    // The intermediate survives, read by the extra reader; every other extraction shift died.
    assert!(after.ops.iter().any(|o| o.dest == Some(y)));
    assert_eq!(shifts_by_one(&after), 1);
}

#[test]
fn test_select_bits_rejects_kept_live_and_vector_group_results() {
    let spec = spec5(IndexForm::Mirrored);
    let p = raw_program(&spec);
    let (and, shr) = first_sites(&p);
    // Kept live by the last op.
    let mut q = clone_program(&p);
    q.ops.last_mut().unwrap().keep_live = Some(vec![and.output]);
    let (found, rejected) = sites(&q, &rule());
    assert_eq!(found.len(), 159);
    assert_eq!(reason_count(&rejected, "kept live"), 1, "{:?}", rejected);
    // The key of a vector group.
    let mut q = clone_program(&p);
    let lanes: Vec<VReg> = (0..8).map(|k| VReg::scalar(q.vreg_count + k)).collect();
    q.vreg_count += 8;
    q.vector_group_lanes.insert(shr.output, lanes);
    let (found, rejected) = sites(&q, &rule());
    assert_eq!(found.len(), 159);
    assert_eq!(reason_count(&rejected, "vector group"), 1, "{:?}", rejected);
}

#[test]
fn test_select_bits_refuses_non_ssa_input() {
    // A second definition of group 0's oldest bit after its uses: the bit has no single value,
    // so nothing built on it matches, and the pass leaves the program unchanged.
    let spec = spec5(IndexForm::Mirrored);
    let mut p = raw_program(&spec);
    let (_, shr) = first_sites(&p);
    let b0 = shr.bit;
    let mut op = SsaOp::new(0, Engine::Valu, Opcode::Xor, Some(b0), vec![Operand::VReg(b0), Operand::VReg(b0)]);
    op.id = p.ops.len();
    p.ops.push(op);
    let (found, rejected) = sites(&p, &rule());
    assert!(!found.iter().any(|s| s.bit == b0), "a site still uses the redefined bit");
    assert!(reason_count(&rejected, "defined more than once") >= 1, "{:?}", rejected);
    let before = canonical_ops(&p);
    let (after, report) = apply_select_bits(p, &rule(), |_| true).unwrap();
    assert!(report.refused.as_deref().map_or(false, |r| r.contains("more than once")), "{:?}", report.refused);
    assert_eq!(canonical_ops(&after), before);
}

#[test]
fn test_select_bits_on_some_sites_keeps_the_extraction() {
    // Only round 3's sites: round 4's extraction (and the zero-based SUBs it needs) stays.
    let spec = spec5(IndexForm::ZeroBased);
    let p = raw_program(&spec);
    let (after, report) = {
        let (q, r) = apply_select_bits(clone_program(&p), &rule(), |s| s.round == Some(3)).unwrap();
        assert_same_outputs(&p, &q, &spec, "round 3 only");
        (q, r)
    };
    assert_eq!(report.n_sites(), 64);
    let left = sites(&after, &rule()).0;
    assert_eq!(left.len(), 96);
    assert!(left.iter().all(|s| s.round == Some(4)));
}
