//! Tests of rule `mirror-index`: it reproduces the `--index-form mirrored` emission from the
//! zero-based one (alone, with the other transforms, in any order), finds the same index
//! chains without tags, keeps the outputs, and refuses a program as a whole, unchanged and
//! with a named reason, when some index use is not one of the proved forms.

use super::graph::SsaGraph;
use super::mirror_index::{apply_mirror_index, IndexUse, MirrorIndexRule};
use super::tests::{canonical_ops, clone_program, first_difference, flow90_plan, raw_program};
use super::tests_index::check_parity;
use super::tests_interp::outputs;
use crate::kernel_builder::config::{CachedSelect, IndexForm, ProblemSpec, TransformKind};
use crate::kernel_builder::pipeline::passes::transform::apply_transform;
use crate::kernel_builder::SsaProgram;
use crate::types::{Engine, Opcode, Operand, SsaOp, VReg};

const BATCH: usize = 256;

fn mirror_configs(rounds: usize) -> Vec<(String, ProblemSpec)> {
    let d = ProblemSpec { rounds, ..Default::default() };
    let mut out = vec![
        (format!("{}r standard d3", rounds), d.clone()),
        (format!("{}r standard d4", rounds), ProblemSpec { tree_reduce_max_depth: 4, ..d.clone() }),
        (format!("{}r standard d2", rounds), ProblemSpec { tree_reduce_max_depth: 2, ..d.clone() }),
    ];
    for (mode, name) in [(CachedSelect::Bits, "bits"), (CachedSelect::BitsFlow, "bits-flow"), (CachedSelect::Mobius, "mobius")] {
        for depth in [3u32, 4] {
            out.push((format!("{}r {} d{}", rounds, name, depth), ProblemSpec { cached_select: mode, tree_reduce_max_depth: depth, ..d.clone() }));
        }
    }
    if rounds == 16 {
        let m = ProblemSpec { index_form: IndexForm::Mirrored, ..d.clone() };
        out.push(("16r flow90 plan".into(), ProblemSpec { access_plan: Some(flow90_plan(&m)), ..d }));
    }
    out
}

fn rule() -> MirrorIndexRule {
    MirrorIndexRule::for_problem(BATCH)
}

// ----------------------------------------------------------------------------
// Parity with the flag emission
// ----------------------------------------------------------------------------

#[test]
fn test_mirror_index_parity_small() {
    // 1 round: final depth 1; 3: depths 0 to 2; 5: first gathers at depth 4 with d3.
    for rounds in [1, 2, 3, 5] {
        for (label, spec) in mirror_configs(rounds) {
            check_parity(&spec, &[TransformKind::MirrorIndex], &label);
        }
    }
}

#[test]
fn test_mirror_index_parity_around_the_wrap() {
    // 11 rounds end on the wrap round (every final index is 0); 12 and 13 end one and two
    // rounds after it.
    for rounds in [11, 12, 13] {
        for (label, spec) in mirror_configs(rounds).into_iter().filter(|(l, _)| l.contains("standard d3") || l.contains("bits d4")) {
            check_parity(&spec, &[TransformKind::MirrorIndex], &label);
        }
    }
}

#[test]
fn test_mirror_index_parity_16_rounds() {
    for (label, spec) in mirror_configs(16) {
        check_parity(&spec, &[TransformKind::MirrorIndex], &label);
    }
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

#[test]
fn test_mirror_index_composes_in_any_order() {
    // With prune-index first, the depth-0 index is `2 - b` and the wrap zero is shared; with
    // it after, prune-index finds nothing to do (no zero-based update is left, C2).
    use TransformKind::*;
    for rounds in [11, 16] {
        for (label, spec) in mirror_configs(rounds).into_iter().filter(|(l, _)| l.contains("standard d3") || l.contains("bits d4") || l.contains("flow90")) {
            for set in [vec![MirrorIndex, PruneIndex], vec![MirrorIndex, ValuesOnly], vec![MirrorIndex, PruneIndex, ValuesOnly]] {
                for order in permutations(&set) {
                    check_parity(&spec, &order, &label);
                }
            }
            let mut all = vec![HashMerge, PruneIndex, ValuesOnly, MirrorIndex];
            if label.contains("bits") || label.contains("flow90") {
                all.push(SelectBitsPath);
            }
            check_parity(&spec, &all, &label);
            all.reverse();
            check_parity(&spec, &all, &label);
        }
    }
}

// ----------------------------------------------------------------------------
// Values, tags
// ----------------------------------------------------------------------------

fn assert_same_outputs(before: &SsaProgram, after: &SsaProgram, spec: &ProblemSpec, label: &str) {
    let seeds = [5u64, 77, 31337];
    let b = outputs(before, spec, &seeds);
    let a = outputs(after, spec, &seeds);
    for (k, ((bo, expected), (ao, _))) in b.iter().zip(&a).enumerate() {
        assert_eq!(bo, expected, "{}: the input program disagrees with the reference (seed {})", label, seeds[k]);
        assert_eq!(ao, bo, "{}: outputs differ after the rewrite (seed {})", label, seeds[k]);
    }
}

#[test]
fn test_mirror_index_preserves_outputs() {
    for rounds in [3, 11, 16] {
        for (label, spec) in mirror_configs(rounds) {
            let p = raw_program(&spec);
            let (q, outcome) = apply_transform(clone_program(&p), TransformKind::MirrorIndex, Some(BATCH)).unwrap();
            assert!(outcome.sites > 0 && outcome.refused.is_none(), "{}: {}", label, outcome.summary);
            assert_same_outputs(&p, &q, &spec, &label);
        }
    }
}

#[test]
fn test_mirror_index_without_tags() {
    let spec = ProblemSpec { cached_select: CachedSelect::Bits, tree_reduce_max_depth: 4, ..Default::default() };
    let tagged = raw_program(&spec);
    let mut bare = clone_program(&tagged);
    for op in &mut bare.ops {
        op.region = None;
        op.source_info = None;
    }
    let (a, ra) = apply_mirror_index(clone_program(&bare), &rule()).unwrap();
    let (mut b, rb) = apply_mirror_index(tagged, &rule()).unwrap();
    assert_eq!(ra.sites, rb.sites);
    assert_eq!(ra.sites.get("step"), Some(&(15 * 32)));
    for op in &mut b.ops {
        op.region = None;
        op.source_info = None;
    }
    assert_eq!(canonical_ops(&a), canonical_ops(&b));
}

#[test]
fn test_mirror_index_plan_reads_depths_from_the_ops() {
    // 16 rounds: 15 steps per group (the wrap round has none); every step's depth (steps since
    // the zero it starts from, read from the ops) equals the emitter's depth tag.
    let spec = ProblemSpec::default();
    let p = raw_program(&spec);
    let g = SsaGraph::new(&p.ops, &p.vector_group_lanes);
    let plan = rule().plan(&g, p.zero_vreg).unwrap();
    assert_eq!(plan.steps.len(), 15 * 32);
    assert!(plan.assumed);
    for s in &plan.steps {
        let tag_depth = p.ops[s.op].source_info.as_ref().and_then(|i| i.tree_depth).unwrap();
        assert_eq!(s.depth as u64, tag_depth as u64, "op {}: depth from the ops vs the emitter's tag", s.op);
    }
    let stores = plan.uses.iter().filter(|(_, _, u)| *u == IndexUse::Store).count();
    assert_eq!(stores, 32);
}

#[test]
fn test_mirror_index_leaves_a_mirrored_program_alone() {
    let spec = ProblemSpec { index_form: IndexForm::Mirrored, cached_select: CachedSelect::Bits, ..Default::default() };
    let p = raw_program(&spec);
    let (q, report) = apply_mirror_index(clone_program(&p), &rule()).unwrap();
    assert_eq!(report.n_sites(), 0);
    assert!(report.refused.is_none());
    assert_eq!(canonical_ops(&q), canonical_ops(&p));
}

// ----------------------------------------------------------------------------
// Refused as a whole, with the reason
// ----------------------------------------------------------------------------

fn spec5() -> ProblemSpec {
    ProblemSpec { rounds: 5, cached_select: CachedSelect::Bits, tree_reduce_max_depth: 3, ..Default::default() }
}

/// Apply the rule and require it refused the program unchanged with a reason containing `text`.
fn assert_refused(p: SsaProgram, rule: &MirrorIndexRule, text: &str) {
    let before = canonical_ops(&p);
    let (after, report) = apply_mirror_index(p, rule).unwrap();
    let why = report.refused.clone().unwrap_or_else(|| panic!("expected a refusal containing {:?}: {}", text, report.summary()));
    assert!(why.contains(text), "{:?} does not contain {:?}", why, text);
    assert_eq!(report.n_sites(), 0);
    // A refused rewrite relied on nothing: no assumption is reported.
    assert!(report.assumptions().is_empty(), "refused, yet reports {:?}", report.assumptions());
    let (d, _) = first_difference(&canonical_ops(&after), &before);
    assert!(d.is_none(), "a refused program must be left unchanged: {:?}", d);
}

/// Op of the first index step of round `round`, group 0.
fn step_of(p: &SsaProgram, round: u32) -> usize {
    let g = SsaGraph::new(&p.ops, &p.vector_group_lanes);
    let plan = rule().plan(&g, p.zero_vreg).unwrap();
    *plan
        .steps
        .iter()
        .map(|s| &s.op)
        .find(|&&o| p.ops[o].source_info.as_ref().and_then(|i| i.round).map(|r| r as u32) == Some(round))
        .unwrap()
}

fn renumber(p: &mut SsaProgram) {
    for (i, op) in p.ops.iter_mut().enumerate() {
        op.id = i;
    }
    p.vreg_def = p.ops.iter().filter_map(|o| o.dest.map(|d| (d, o.id))).collect();
}

#[test]
fn test_mirror_index_needs_initial_indices_zero() {
    // Without the assumption the first pass is not shown to start at 0: rewriting only the
    // chains after the wrap would leave a mix, so the program is refused.
    let p = raw_program(&ProblemSpec { rounds: 16, ..Default::default() });
    let unassumed = MirrorIndexRule { assumptions: vec![], ..rule() };
    assert_refused(p, &unassumed, "initial-indices-zero is not declared");
}

#[test]
fn test_mirror_index_refuses_an_extra_reader_of_an_index() {
    // An extra reader `z = idx ^ a` (a: the hash input), whose result feeds the hash input:
    // not an R10 form. `idx ^ idx` (read twice) is refused too.
    extra_reader(false, "not an R10 form");
    extra_reader(true, "reads the index more than once");
}

fn extra_reader(twice: bool, text: &str) {
    let mut p = raw_program(&spec5());
    let step = step_of(&p, 2);
    let idx = p.ops[step].dest.unwrap();
    let xor = (step + 1..p.ops.len()).find(|&i| p.ops[i].opcode == Opcode::Xor && p.ops[i].engine == Engine::Valu).unwrap();
    let a = p.ops[xor].srcs[0].as_vreg().unwrap();
    let (z, w) = (VReg::vector(p.vreg_count), VReg::vector(p.vreg_count + 1));
    p.vreg_count += 2;
    p.ops[xor].srcs[0] = Operand::VReg(w);
    p.ops.insert(xor, SsaOp::new(0, Engine::Valu, Opcode::Xor, Some(w), vec![Operand::VReg(a), Operand::VReg(z)]));
    let second = if twice { idx } else { a };
    p.ops.insert(xor, SsaOp::new(0, Engine::Valu, Opcode::Xor, Some(z), vec![Operand::VReg(idx), Operand::VReg(second)]));
    renumber(&mut p);
    assert_refused(p, &rule(), text);
}

#[test]
fn test_mirror_index_refuses_a_wrong_step_multiplier() {
    // Round 2's step multiplies by 3: the index it reads is read by something that is not a
    // step, and the next step's index has no form.
    let mut p = raw_program(&spec5());
    let step = step_of(&p, 2);
    let (s3, v3) = (VReg::scalar(p.vreg_count), VReg::vector(p.vreg_count + 1));
    p.vreg_count += 2;
    let mut c3 = SsaOp::new(0, Engine::Load, Opcode::ConstChoice, Some(s3), vec![Operand::Imm(3)]);
    c3.produces_constant = Some(3);
    let b3 = SsaOp::new(0, Engine::Valu, Opcode::Vbroadcast, Some(v3), vec![Operand::VReg(s3)]);
    let two = p.ops[step].srcs.iter().position(|s| {
        let g = SsaGraph::new(&p.ops, &p.vector_group_lanes);
        s.as_vreg().and_then(|v| g.broadcast_const(v)) == Some(2)
    });
    p.ops[step].srcs[two.unwrap()] = Operand::VReg(v3);
    p.ops.insert(step, b3);
    p.ops.insert(step, c3);
    renumber(&mut p);
    assert_refused(p, &rule(), "is not shown to be a zero-based index");
}

#[test]
fn test_mirror_index_refuses_a_gather_from_a_non_broadcast_base() {
    // A gather address idx + v with v not a broadcast (here the group's value vector).
    let spec = ProblemSpec { rounds: 6, ..Default::default() };
    let mut p = raw_program(&spec);
    let g = SsaGraph::new(&p.ops, &p.vector_group_lanes);
    let plan = rule().plan(&g, p.zero_vreg).unwrap();
    let (r, v, _) = plan.uses.iter().find(|(_, _, u)| matches!(u, IndexUse::Gather { .. })).unwrap().clone();
    let other = p.ops[r].srcs.iter().position(|s| s.as_vreg() != Some(v)).unwrap();
    let val = p.ops.iter().take(r).rev().find(|o| o.opcode == Opcode::Xor && o.dest.map_or(false, |d| d.is_vector())).unwrap().dest.unwrap();
    drop(g);
    p.ops[r].srcs[other] = Operand::VReg(val);
    assert_refused(p, &rule(), "added to something that is not a vbroadcast");
}

#[test]
fn test_mirror_index_refuses_a_compare_with_a_non_constant() {
    // Standard depth 1: idx == 1 becomes idx == (a value vector).
    let spec = ProblemSpec { rounds: 3, ..Default::default() };
    let mut p = raw_program(&spec);
    let g = SsaGraph::new(&p.ops, &p.vector_group_lanes);
    let plan = rule().plan(&g, p.zero_vreg).unwrap();
    let (r, v, _) = plan.uses.iter().find(|(_, _, u)| matches!(u, IndexUse::Compare { .. })).unwrap().clone();
    let other = p.ops[r].srcs.iter().position(|s| s.as_vreg() != Some(v)).unwrap();
    let val = p.ops.iter().take(r).rev().find(|o| o.opcode == Opcode::Xor && o.dest.map_or(false, |d| d.is_vector())).unwrap().dest.unwrap();
    drop(g);
    p.ops[r].srcs[other] = Operand::VReg(val);
    assert_refused(p, &rule(), "compared with a value that is not a constant");
}

#[test]
fn test_mirror_index_refuses_an_index_kept_live() {
    let mut p = raw_program(&spec5());
    let idx = p.ops[step_of(&p, 2)].dest.unwrap();
    p.ops.last_mut().unwrap().keep_live = Some(vec![idx]);
    assert_refused(p, &rule(), "kept live");
}

#[test]
fn test_mirror_index_refuses_non_ssa_input() {
    let mut p = raw_program(&spec5());
    let idx = p.ops[step_of(&p, 2)].dest.unwrap();
    let mut op = SsaOp::new(0, Engine::Valu, Opcode::Xor, Some(idx), vec![Operand::VReg(idx), Operand::VReg(idx)]);
    op.id = p.ops.len();
    p.ops.push(op);
    let before = canonical_ops(&p);
    let (after, report) = apply_mirror_index(p, &rule()).unwrap();
    assert!(report.refused.as_deref().map_or(false, |r| r.contains("more than once")), "{:?}", report.refused);
    assert!(report.assumptions().is_empty(), "{:?}", report.assumptions());
    assert_eq!(canonical_ops(&after), before);
}

#[test]
fn test_mirror_index_assumes_initial_indices_zero_only_when_it_rewrites() {
    // spec5 is zero-based: the rewrite relies on InitialIndicesZero. A trace id on a step
    // refuses it, and then nothing is assumed.
    let p = raw_program(&spec5());
    let (_, done) = apply_mirror_index(clone_program(&p), &rule()).unwrap();
    assert!(done.refused.is_none() && done.n_sites() > 0, "{:?}", done.refused);
    assert_eq!(done.assumptions(), vec!["initial-indices-zero"]);
    let mut q = clone_program(&p);
    let st = step_of(&q, 1);
    q.ops[st].trace_id = Some("t".into());
    assert_refused(q, &rule(), "carries a trace id");
}


// ----------------------------------------------------------------------------
// Review of #81
// ----------------------------------------------------------------------------

/// `after` computes the same final indices and values as `before` (not required to be the
/// kernel: perturbed programs compute something else, the rewrite must keep it).
fn assert_unchanged_outputs(before: &SsaProgram, after: &SsaProgram, spec: &ProblemSpec, label: &str) {
    let seeds = [21u64, 99];
    let (b, a) = (outputs(before, spec, &seeds), outputs(after, spec, &seeds));
    for (k, ((bo, _), (ao, _))) in b.iter().zip(&a).enumerate() {
        assert_eq!(ao, bo, "{}: outputs differ after the rewrite (seed {})", label, seeds[k]);
    }
}

/// Apply the rule (must rewrite, not refuse) and compare outputs before and after.
fn rewrite_and_compare(p: &SsaProgram, spec: &ProblemSpec, label: &str) -> super::report::RuleReport {
    let (after, report) = apply_mirror_index(clone_program(p), &rule()).unwrap_or_else(|e| panic!("{}: {}", label, e));
    assert!(report.refused.is_none(), "{}: refused: {:?}", label, report.refused);
    assert!(report.sites.get("step").copied().unwrap_or(0) > 0, "{}", label);
    assert_unchanged_outputs(p, &after, spec, label);
    report
}

/// A vector VReg holding the constant `k` (the first one, by propagation).
fn vconst(p: &SsaProgram, k: i64) -> VReg {
    let g = SsaGraph::new(&p.ops, &p.vector_group_lanes);
    let values = super::values::ValueFacts::compute(&g, p.zero_vreg, &rule().layout, &[]);
    p.ops.iter().filter_map(|o| o.dest).find(|&d| d.is_vector() && values.value(&g, d) == Some(k)).unwrap()
}

/// A vector of zeros: a new broadcast of the zero register appended to `p`.
fn zero_vec(p: &mut SsaProgram) -> VReg {
    let z = p.zero_vreg.unwrap();
    let d = fresh(p, true);
    let mut op = SsaOp::new(0, Engine::Valu, Opcode::Vbroadcast, Some(d), vec![Operand::VReg(z)]);
    op.id = p.ops.len();
    p.ops.push(op);
    p.vreg_def.insert(d, p.ops.len() - 1);
    d
}

/// A new vector constant `k` (ConstChoice and broadcast) appended to `p`.
fn new_vconst(p: &mut SsaProgram, k: i64) -> VReg {
    let s = fresh(p, false);
    let mut c = SsaOp::new(0, Engine::Load, Opcode::ConstChoice, Some(s), vec![Operand::Imm(k)]);
    c.produces_constant = Some(k);
    c.id = p.ops.len();
    p.ops.push(c);
    p.vreg_def.insert(s, p.ops.len() - 1);
    let d = fresh(p, true);
    let mut b = SsaOp::new(0, Engine::Valu, Opcode::Vbroadcast, Some(d), vec![Operand::VReg(s)]);
    b.id = p.ops.len();
    p.ops.push(b);
    p.vreg_def.insert(d, p.ops.len() - 1);
    d
}

/// Fresh VReg.
fn fresh(p: &mut SsaProgram, vector: bool) -> VReg {
    p.vreg_count += 1;
    if vector { VReg::vector(p.vreg_count - 1) } else { VReg::scalar(p.vreg_count - 1) }
}

/// Append a vector op.
fn push_valu(p: &mut SsaProgram, opcode: Opcode, srcs: &[VReg]) -> VReg {
    let d = fresh(p, true);
    let mut op = SsaOp::new(0, Engine::Valu, opcode, Some(d), srcs.iter().map(|&v| Operand::VReg(v)).collect());
    op.id = p.ops.len();
    p.ops.push(op);
    p.vreg_def.insert(d, p.ops.len() - 1);
    d
}

/// Insert `ops` right before op `at` and renumber.
fn insert_at(p: &mut SsaProgram, at: usize, ops: Vec<SsaOp>) {
    for (k, op) in ops.into_iter().enumerate() {
        p.ops.insert(at + k, op);
    }
    renumber(p);
}

/// Indices of the vstores to the values array.
fn value_stores(p: &SsaProgram) -> Vec<usize> {
    let g = SsaGraph::new(&p.ops, &p.vector_group_lanes);
    let values = super::values::ValueFacts::compute(&g, p.zero_vreg, &rule().layout, &[]);
    (0..p.ops.len())
        .filter(|&i| p.ops[i].opcode == Opcode::Vstore)
        .filter(|&i| p.ops[i].srcs[0].as_vreg().and_then(|a| values.address(&g, a)).map_or(false, |a| a.word == 6))
        .collect()
}

// Finding 1: the final L_k - n must come after n, wherever the stored values are computed.

#[test]
fn test_mirror_index_final_index_after_n_when_values_are_stored_early() {
    // Every value store stores the group's initial value (defined before any index step): the
    // last stored value is computed before n exists.
    let spec = spec5();
    let mut p = raw_program(&spec);
    let loads: Vec<VReg> = p.ops.iter().filter(|o| o.opcode == Opcode::Vload).filter_map(|o| o.dest).collect();
    let stores = value_stores(&p);
    assert_eq!(stores.len(), 32);
    for (k, &st) in stores.iter().enumerate() {
        p.ops[st].srcs[1] = Operand::VReg(loads[k]);
    }
    rewrite_and_compare(&p, &spec, "values stored early");
}

#[test]
fn test_mirror_index_without_value_stores() {
    let spec = spec5();
    let mut p = raw_program(&spec);
    for st in value_stores(&p).into_iter().rev() {
        p.ops.remove(st);
    }
    renumber(&mut p);
    rewrite_and_compare(&p, &spec, "no value stores");
}

// Finding 2: metadata the rewrite would drop.

#[test]
fn test_mirror_index_refuses_a_step_with_a_trace_id() {
    let mut p = raw_program(&spec5());
    let st = step_of(&p, 1);
    p.ops[st].trace_id = Some("t".into());
    assert_refused(p, &rule(), "carries a trace id");
}

#[test]
fn test_mirror_index_refuses_a_mirror_op_with_keep_live() {
    // Zero-based bits at depth 3: the extraction's `L_2 - idx` is a Mirror use (removed).
    let mut p = raw_program(&spec5());
    let g = SsaGraph::new(&p.ops, &p.vector_group_lanes);
    let plan = rule().plan(&g, p.zero_vreg).unwrap();
    let (r, _, _) = *plan.uses.iter().find(|(_, _, u)| *u == IndexUse::Mirror).unwrap();
    drop(g);
    let other = p.ops[0].dest.unwrap();
    p.ops[r].keep_live = Some(vec![other]);
    assert_refused(p, &rule(), "keep_live");
}

// Finding 3: one op, two index values.

#[test]
fn test_mirror_index_refuses_an_op_reading_two_index_values() {
    // Two index chains whose bits are constants (x & 1 of a constant), so both depth-1 indices
    // fold to constants and `idxA == idxB` classifies as a compare for each.
    let mut p = raw_program(&spec5());
    let two = vconst(&p, 2);
    let (three, five) = (new_vconst(&mut p, 3), new_vconst(&mut p, 5));
    let one = vconst(&p, 1);
    let zero = zero_vec(&mut p);
    let chain = |p: &mut SsaProgram, c: VReg| {
        let b = push_valu(p, Opcode::And, &[c, one]);
        let off = push_valu(p, Opcode::Sub, &[two, b]);
        push_valu(p, Opcode::MultiplyAdd, &[zero, two, off])
    };
    let a = chain(&mut p, three);
    let b = chain(&mut p, five);
    push_valu(&mut p, Opcode::Eq, &[a, b]);
    renumber(&mut p);
    assert_refused(p, &rule(), "reads two index values");
}

#[test]
fn test_surgery_finish_rechecks_single_assignment() {
    use super::surgery::{Stamp, Surgery};
    let p = raw_program(&spec5());
    let (at, dup) = p.ops.iter().enumerate().find(|(_, o)| o.opcode == Opcode::Vload).map(|(i, o)| (i, o.clone())).unwrap();
    // The copy is inserted before the original: whichever definer liveness follows, one of
    // the two is a second definer of a live VReg.
    let mut s = Surgery::new(p).unwrap();
    s.insert_before(at, dup, &Stamp::default());
    let err = s.finish().err().expect("two definers of one VReg");
    assert!(err.contains("more than once"), "{}", err);
}

// Finding 4: a constant reused from the program must be a constant expression.

#[test]
fn test_mirror_index_does_not_reuse_an_assumed_constant() {
    // `t = loaded_idx + 1` (1 only under InitialIndicesZero) at the program start: the depth-1
    // compare constant L_1 - 1 = 1 must come from a real constant, not from t.
    let spec = ProblemSpec { rounds: 3, ..Default::default() };
    let mut p = raw_program(&spec);
    let (c5, ptr, idx, c1, t) = (fresh(&mut p, false), fresh(&mut p, false), fresh(&mut p, false), fresh(&mut p, false), fresh(&mut p, false));
    let mut k5 = SsaOp::new(0, Engine::Load, Opcode::ConstChoice, Some(c5), vec![Operand::Imm(5)]);
    k5.produces_constant = Some(5);
    let mut k1 = SsaOp::new(0, Engine::Load, Opcode::ConstChoice, Some(c1), vec![Operand::Imm(1)]);
    k1.produces_constant = Some(1);
    let front = vec![
        k5,
        SsaOp::new(0, Engine::Load, Opcode::Load, Some(ptr), vec![Operand::VReg(c5)]),
        SsaOp::new(0, Engine::Load, Opcode::Load, Some(idx), vec![Operand::VReg(ptr)]),
        k1,
        SsaOp::new(0, Engine::Alu, Opcode::Add, Some(t), vec![Operand::VReg(idx), Operand::VReg(c1)]),
    ];
    insert_at(&mut p, 0, front);
    let (after, report) = apply_mirror_index(p, &rule()).unwrap();
    assert!(report.refused.is_none(), "{:?}", report.refused);
    assert!(report.sites.get("compare").copied().unwrap_or(0) > 0);
    assert!(!after.ops.iter().any(|o| o.srcs.iter().any(|s| s.as_vreg() == Some(t))), "the assumed constant was reused");
}

// Finding 5: refusal branches.

#[test]
fn test_mirror_index_refuses_an_index_in_a_vector_group() {
    let mut p = raw_program(&spec5());
    let idx = p.ops[step_of(&p, 1)].dest.unwrap();
    let lanes: Vec<VReg> = (0..8).map(|_| fresh(&mut p, false)).collect();
    p.vector_group_lanes.insert(idx, lanes);
    assert_refused(p, &rule(), "vector group");
}

#[test]
fn test_mirror_index_refuses_an_index_as_a_select_delta() {
    let mut p = raw_program(&spec5());
    let idx = p.ops[step_of(&p, 1)].dest.unwrap();
    let v = push_valu(&mut p, Opcode::SelectChoice, &[idx, idx, idx]);
    let last = p.ops.len() - 1;
    p.ops[last].srcs = vec![Operand::VReg(vconst(&p, 1)), Operand::VReg(vconst(&p, 2)), Operand::VReg(vconst(&p, 1))];
    p.ops[last].delta_vreg = Some(idx);
    let _ = v;
    assert_refused(p, &rule(), "delta of select");
}

#[test]
fn test_mirror_index_refuses_an_index_as_a_step_addend() {
    // A depth-1 index (2 - b) used as the offset of another step.
    let mut p = raw_program(&spec5());
    let idx1 = p.ops[step_of(&p, 0)].dest.unwrap();
    let two = vconst(&p, 2);
    let zero = zero_vec(&mut p);
    push_valu(&mut p, Opcode::MultiplyAdd, &[zero, two, idx1]);
    assert_refused(p, &rule(), "addend");
}

#[test]
fn test_mirror_index_refuses_a_wrong_mirror_constant() {
    let mut p = raw_program(&spec5());
    let idx2 = p.ops[step_of(&p, 1)].dest.unwrap();
    let five = new_vconst(&mut p, 5);
    push_valu(&mut p, Opcode::Sub, &[five, idx2]);
    assert_refused(p, &rule(), "!= L_2 = 6");
}

#[test]
fn test_mirror_index_refuses_index_minus_a_non_constant() {
    let mut p = raw_program(&spec5());
    let idx2 = p.ops[step_of(&p, 1)].dest.unwrap();
    // The value group 0 stores (a hash result, not a constant; the first vload would be the
    // index load, which is 0 under the assumption).
    let val = p.ops[value_stores(&p)[0]].srcs[1].as_vreg().unwrap();
    push_valu(&mut p, Opcode::Sub, &[idx2, val]);
    assert_refused(p, &rule(), "minus a value that is not a constant");
}

#[test]
fn test_mirror_index_refuses_a_deep_index_that_is_not_a_step_result() {
    // x = 6 - (2 b0 + b1) is Zero([b0, b1]) (depth 2) but not built by a step; a step reading
    // it has no n to start from.
    let mut p = raw_program(&spec5());
    let g = SsaGraph::new(&p.ops, &p.vector_group_lanes);
    let plan = rule().plan(&g, p.zero_vreg).unwrap();
    let (b0, b1, off) = (plan.steps[0].bit, plan.steps[40].bit, plan.steps[0].offset);
    drop(g);
    let (two, six) = (vconst(&p, 2), vconst(&p, 6));
    let pos = push_valu(&mut p, Opcode::MultiplyAdd, &[b0, two, b1]);
    let x = push_valu(&mut p, Opcode::Sub, &[six, pos]);
    push_valu(&mut p, Opcode::MultiplyAdd, &[x, two, off]);
    assert_refused(p, &rule(), "not the result of an index step");
}

#[test]
fn test_mirror_index_refuses_an_index_stored_elsewhere() {
    let mut p = raw_program(&spec5());
    let idx2 = p.ops[step_of(&p, 1)].dest.unwrap();
    let st = value_stores(&p)[0];
    let mut store = p.ops[st].clone();
    store.srcs[1] = Operand::VReg(idx2);
    store.id = p.ops.len();
    p.ops.push(store);
    assert_refused(p, &rule(), "not an R10 form");
}

// Finding 5: positive tests on perturbed input, checked before vs after.

fn standard3() -> ProblemSpec {
    ProblemSpec { rounds: 3, ..Default::default() }
}

/// First use of `kind` in the plan of `p`: (reader op, index value).
fn first_use(p: &SsaProgram, pick: impl Fn(&IndexUse) -> bool) -> (usize, VReg) {
    let g = SsaGraph::new(&p.ops, &p.vector_group_lanes);
    let plan = rule().plan(&g, p.zero_vreg).unwrap();
    let (r, v, _) = plan.uses.iter().find(|(_, _, u)| pick(u)).cloned().unwrap();
    (r, v)
}

#[test]
fn test_mirror_index_rewrites_perturbed_compares() {
    let spec = standard3();
    // A different compare constant: idx == 2 instead of idx == 1.
    let mut p = raw_program(&spec);
    let (r, v) = first_use(&p, |u| matches!(u, IndexUse::Compare { .. }));
    let other = p.ops[r].srcs.iter().position(|s| s.as_vreg() != Some(v)).unwrap();
    p.ops[r].srcs[other] = Operand::VReg(vconst(&p, 2));
    let rep = rewrite_and_compare(&p, &spec, "compare constant 2");
    assert!(rep.sites.get("compare").copied().unwrap_or(0) > 0);
    // Swapped operands: c == idx.
    let mut p = raw_program(&spec);
    let (r, _) = first_use(&p, |u| matches!(u, IndexUse::Compare { .. }));
    p.ops[r].srcs.swap(0, 1);
    rewrite_and_compare(&p, &spec, "c == idx");
}

#[test]
fn test_mirror_index_rewrites_perturbed_gathers() {
    let spec = ProblemSpec { rounds: 6, ..Default::default() };
    // Swapped operands: vbroadcast + idx.
    let mut p = raw_program(&spec);
    let (r, _) = first_use(&p, |u| matches!(u, IndexUse::Gather { .. }));
    p.ops[r].srcs.swap(0, 1);
    rewrite_and_compare(&p, &spec, "base + idx");
    // A base from another scalar: (tree pointer + 1), broadcast just before the gather.
    let mut p = raw_program(&spec);
    let (r, v) = first_use(&p, |u| matches!(u, IndexUse::Gather { .. }));
    let other = p.ops[r].srcs.iter().position(|s| s.as_vreg() != Some(v)).unwrap();
    let b = p.ops[r].srcs[other].as_vreg().unwrap();
    let fp = p.ops.iter().find(|o| o.dest == Some(b)).unwrap().srcs[0].as_vreg().unwrap();
    let c1 = p.ops.iter().find(|o| o.opcode == Opcode::ConstChoice && o.produces_constant == Some(1)).unwrap().dest.unwrap();
    let (s2, b2) = (fresh(&mut p, false), fresh(&mut p, true));
    p.ops[r].srcs[other] = Operand::VReg(b2);
    insert_at(
        &mut p,
        r,
        vec![
            SsaOp::new(0, Engine::Alu, Opcode::Add, Some(s2), vec![Operand::VReg(fp), Operand::VReg(c1)]),
            SsaOp::new(0, Engine::Valu, Opcode::Vbroadcast, Some(b2), vec![Operand::VReg(s2)]),
        ],
    );
    let rep = rewrite_and_compare(&p, &spec, "base from another scalar");
    assert!(rep.sites.get("gather").copied().unwrap_or(0) > 0);
}

#[test]
fn test_mirror_index_rewrites_two_independent_index_families() {
    // A second index chain for group 0, built from group 1's offsets, compared with a constant
    // and folded into group 0's stored value: both families are rewritten.
    let spec = standard3();
    let mut p = raw_program(&spec);
    let g = SsaGraph::new(&p.ops, &p.vector_group_lanes);
    let plan = rule().plan(&g, p.zero_vreg).unwrap();
    let steps0: Vec<_> = plan.steps.iter().filter(|s| p.ops[s.op].region.and_then(|t| t.group) == Some(1)).cloned().collect();
    assert_eq!(steps0.len(), 3);
    drop(g);
    let (two, four) = (vconst(&p, 2), vconst(&p, 4));
    let zero = zero_vec(&mut p);
    let st = value_stores(&p)[0];
    let val = p.ops[st].srcs[1].as_vreg().unwrap();
    let a = push_valu(&mut p, Opcode::MultiplyAdd, &[zero, two, steps0[0].offset]);
    let b = push_valu(&mut p, Opcode::MultiplyAdd, &[a, two, steps0[1].offset]);
    let e = push_valu(&mut p, Opcode::Eq, &[b, four]);
    let w = push_valu(&mut p, Opcode::Xor, &[val, e]);
    let mut store = p.ops[st].clone();
    store.srcs[1] = Operand::VReg(w);
    store.id = p.ops.len();
    p.ops.remove(st);
    p.ops.push(store);
    renumber(&mut p);
    let rep = rewrite_and_compare(&p, &spec, "two families");
    assert!(rep.sites.get("step").copied().unwrap_or(0) >= 3 * 2 + 2);
}

// Finding 6: are the differing kernel bytes only VReg numbering?

/// The program after dead code elimination with VRegs renumbered by first appearance (dest or
/// operand, in op order; vector group lanes right after their key).
fn renumbered(p: &SsaProgram) -> SsaProgram {
    use std::collections::HashMap;
    let mut q = clone_program(p);
    let (ops, vreg_def, alts) = crate::kernel_builder::emit_ssa::eliminate_ssa_dead_code(
        std::mem::take(&mut q.ops),
        &mut q.vreg_def,
        &mut q.const_alternatives,
        q.zero_vreg,
        Some(&q.vector_group_lanes),
    );
    q.ops = ops;
    q.vreg_def = vreg_def;
    q.const_alternatives = alts;
    let mut map: HashMap<VReg, VReg> = HashMap::new();
    let mut next = 0u32;
    let mut name = |v: VReg, map: &mut HashMap<VReg, VReg>, lanes: &HashMap<VReg, Vec<VReg>>| {
        if !map.contains_key(&v) {
            let nv = if v.is_vector() { VReg::vector(next) } else { VReg::scalar(next) };
            next += 1;
            map.insert(v, nv);
            if let Some(ls) = lanes.get(&v) {
                for &l in ls {
                    if !map.contains_key(&l) {
                        map.insert(l, VReg::scalar(next));
                        next += 1;
                    }
                }
            }
        }
        map[&v]
    };
    if let Some(z) = q.zero_vreg {
        q.zero_vreg = Some(name(z, &mut map, &q.vector_group_lanes));
    }
    let lanes = q.vector_group_lanes.clone();
    for op in &mut q.ops {
        for s in &mut op.srcs {
            if let Operand::VReg(v) = s {
                *v = name(*v, &mut map, &lanes);
            }
        }
        if let Some(d) = op.delta_vreg.as_mut() {
            *d = name(*d, &mut map, &lanes);
        }
        if let Some(k) = op.keep_live.as_mut() {
            for v in k.iter_mut() {
                *v = name(*v, &mut map, &lanes);
            }
        }
        if let Some(d) = op.dest.as_mut() {
            *d = name(*d, &mut map, &lanes);
        }
        // A SelectChoice's temporary (its multiply_add form's SUB) is a VReg too: address
        // assignment and liveness read it.
        if let Some(t) = op.sub_temp_vreg.as_mut() {
            *t = name(*t, &mut map, &lanes);
        }
    }
    q.vector_group_lanes = lanes
        .iter()
        .filter(|(k, _)| map.contains_key(k))
        .map(|(k, ls)| (map[k], ls.iter().map(|l| *map.get(l).unwrap_or(l)).collect()))
        .collect();
    for (i, op) in q.ops.iter_mut().enumerate() {
        op.id = i;
    }
    q.vreg_def = q.ops.iter().filter_map(|o| o.dest.map(|d| (d, o.id))).collect();
    q.vreg_count = next;
    q
}

#[test]
fn test_mirror_index_kernel_equals_the_flag_kernel() {
    use crate::kernel_builder::config::CompilerConfig;
    use crate::kernel_builder::pipeline::{PipelineConfig, PipelineContext};
    use crate::kernel_builder::config::{GreedyBarrierMode, GreedyPriorityKind, PressureThresholds};
    let shaped = CompilerConfig {
        greedy_barriers: GreedyBarrierMode::from_str("none").unwrap(),
        greedy_priority: GreedyPriorityKind::from_str("wavefront").unwrap(),
        greedy_split_shift: true,
        greedy_groups_in_flight: 32,
        greedy_stagger_cycles: 9,
        pressure_thresholds: PressureThresholds { soft: None, critical: None, inflation: Some(0.95) },
        ..Default::default()
    };
    for (label, compiler) in [("default greedy", CompilerConfig::default()), ("best shape flags", shaped)] {
    let schedule = |p: SsaProgram| {
        let config = PipelineConfig { compiler: compiler.clone(), verbose: false, ..Default::default() };
        let mut ctx = PipelineContext::new(p, config);
        ctx.run_greedy_pipeline(true).unwrap();
        // What the kernel is: per lowered op its cycle, engine and slot (HashSet fields such as
        // dep_ids print in iteration order and are not part of the kernel).
        ctx.into_result().unwrap().ops.iter().map(|o| format!("{} {:?} {:?}", o.cycle, o.engine, o.slot)).collect::<Vec<String>>()
    };
    let mirrored = ProblemSpec { merge_hash_stages: true, index_form: IndexForm::Mirrored, ..Default::default() };
    let spec = ProblemSpec { access_plan: Some(flow90_plan(&mirrored)), ..mirrored };
    let flag = raw_program(&spec);
    let zero_based = raw_program(&ProblemSpec { index_form: IndexForm::ZeroBased, ..spec.clone() });
    let (rewritten, _) = apply_mirror_index(zero_based, &rule()).unwrap();
    let as_is = (schedule(clone_program(&flag)), schedule(clone_program(&rewritten)));
    let renum = (schedule(renumbered(&flag)), schedule(renumbered(&rewritten)));
    eprintln!("{}: kernels equal as built: {}; equal after renumbering VRegs: {}", label, as_is.0 == as_is.1, renum.0 == renum.1);
    for (name, (x, y)) in [("as built", &as_is), ("renumbered", &renum)] {
        let diffs: Vec<usize> = (0..x.len().min(y.len())).filter(|&i| x[i] != y[i]).collect();
        eprintln!("{}: {} vs {} lowered ops, {} differ", name, x.len(), y.len(), diffs.len());
        if let Some(&i) = diffs.first() {
            eprintln!("  first at {}:\n  flag      {}\n  rewritten {}", i, x[i], y[i]);
        }
    }
    assert_eq!(renum.0, renum.1, "{}: kernels differ beyond VReg numbering", label);
    // The default greedy builds the same kernel as is; the best-shape configuration may differ
    // in scratch addresses only (VReg numbering), which the renumbered comparison covers.
    if label == "default greedy" {
        assert_eq!(as_is.0, as_is.1, "{}: kernels differ as built", label);
    }
    }
}

// ----------------------------------------------------------------------------
// Final-gate review of #81
// ----------------------------------------------------------------------------

#[test]
fn test_mirror_index_refuses_a_mirror_result_read_as_a_vector_group() {
    // Zero-based bits at depth 3: the extraction's `L_2 - idx` is a mirror op. Its result as a
    // vector group key (lanes read by scalar ops) cannot be rewired by replace_uses.
    let mut p = raw_program(&spec5());
    let g = SsaGraph::new(&p.ops, &p.vector_group_lanes);
    let plan = rule().plan(&g, p.zero_vreg).unwrap();
    let (r, _, _) = *plan.uses.iter().find(|(_, _, u)| *u == IndexUse::Mirror).unwrap();
    drop(g);
    let d = p.ops[r].dest.unwrap();
    let lanes: Vec<VReg> = (0..8).map(|_| fresh(&mut p, false)).collect();
    p.vector_group_lanes.insert(d, lanes);
    assert_refused(p, &rule(), "read as a vector group");
}

#[test]
fn test_mirror_index_records_an_assumption_from_a_step_constant() {
    // With --prune-dead-index every chain starts at `2 - b` (no loaded index): nothing is
    // assumed. Make one step's constant 2 `vbroadcast(loaded_idx + 2)` (2 only when the input
    // indices are 0): the rewrite now relies on InitialIndicesZero and says so.
    let spec = ProblemSpec { rounds: 3, prune_dead_index: true, ..Default::default() };
    let p = raw_program(&spec);
    let (_, report) = apply_mirror_index(clone_program(&p), &rule()).unwrap();
    assert!(report.n_sites() > 0 && report.assumptions().is_empty(), "{:?}", report.facts);

    let mut p = p;
    let st = step_of(&p, 2);
    let two = vconst(&p, 2);
    let k = p.ops[st].srcs.iter().position(|s| s.as_vreg() == Some(two)).unwrap();
    let (c5, ptr, idx, c2, sum, b2) = (fresh(&mut p, false), fresh(&mut p, false), fresh(&mut p, false), fresh(&mut p, false), fresh(&mut p, false), fresh(&mut p, true));
    let mut k5 = SsaOp::new(0, Engine::Load, Opcode::ConstChoice, Some(c5), vec![Operand::Imm(5)]);
    k5.produces_constant = Some(5);
    let mut k2 = SsaOp::new(0, Engine::Load, Opcode::ConstChoice, Some(c2), vec![Operand::Imm(2)]);
    k2.produces_constant = Some(2);
    p.ops[st].srcs[k] = Operand::VReg(b2);
    insert_at(&mut p, 0, vec![
        k5,
        SsaOp::new(0, Engine::Load, Opcode::Load, Some(ptr), vec![Operand::VReg(c5)]),
        SsaOp::new(0, Engine::Load, Opcode::Load, Some(idx), vec![Operand::VReg(ptr)]),
        k2,
        SsaOp::new(0, Engine::Alu, Opcode::Add, Some(sum), vec![Operand::VReg(idx), Operand::VReg(c2)]),
        SsaOp::new(0, Engine::Valu, Opcode::Vbroadcast, Some(b2), vec![Operand::VReg(sum)]),
    ]);
    let (_, report) = apply_mirror_index(p, &rule()).unwrap();
    assert!(report.refused.is_none(), "{:?}", report.refused);
    assert_eq!(report.assumptions(), vec!["initial-indices-zero"]);
}

#[test]
fn test_mirror_index_index_store_before_the_last_value_definer() {
    // Group 0's index store moved to right after its operands are defined, i.e. before the
    // values the value stores store are computed: the final L_k - n goes before the store.
    let spec = spec5();
    let mut p = raw_program(&spec);
    let g = SsaGraph::new(&p.ops, &p.vector_group_lanes);
    let plan = rule().plan(&g, p.zero_vreg).unwrap();
    let (r, _, _) = *plan.uses.iter().find(|(_, _, u)| *u == IndexUse::Store).unwrap();
    drop(g);
    let store = p.ops.remove(r);
    let ready = store.srcs.iter().filter_map(|s| s.as_vreg()).map(|v| p.ops.iter().position(|o| o.dest == Some(v)).unwrap()).max().unwrap();
    let moved_srcs = store.srcs.clone();
    p.ops.insert(ready + 1, store);
    renumber(&mut p);
    // The emitter computes the stored values before the index chain ends; move the latest
    // value definer (and its value store, its only reader) to the end, so that a value is
    // computed after the index store.
    let latest = value_stores(&p)
        .into_iter()
        .map(|s| (p.ops.iter().position(|o| o.dest == p.ops[s].srcs[1].as_vreg()).unwrap(), s))
        .max()
        .unwrap();
    let (def_op, store_op) = (p.ops[latest.0].clone(), p.ops[latest.1].clone());
    let (hi, lo) = (latest.0.max(latest.1), latest.0.min(latest.1));
    p.ops.remove(hi);
    p.ops.remove(lo);
    p.ops.push(def_op);
    p.ops.push(store_op);
    renumber(&mut p);
    // The latest definer of a value the value stores write (the stores' data operands), in
    // the reordered program.
    let last_value_def = value_stores(&p)
        .iter()
        .map(|&s| {
            let v = p.ops[s].srcs[1].as_vreg().unwrap();
            p.ops.iter().position(|o| o.dest == Some(v)).unwrap()
        })
        .max()
        .unwrap();
    let at = p.ops.iter().position(|o| o.opcode == Opcode::Vstore && o.srcs == moved_srcs).unwrap();
    assert!(at < last_value_def, "the index store ({}) must precede the last value definer ({})", at, last_value_def);
    rewrite_and_compare(&p, &spec, "index store early");
}

/// Index stores of `p` (vector stores into the indices array).
fn index_stores(p: &SsaProgram) -> Vec<usize> {
    let g = SsaGraph::new(&p.ops, &p.vector_group_lanes);
    let values = super::values::ValueFacts::compute(&g, p.zero_vreg, &rule().layout, &[]);
    (0..p.ops.len())
        .filter(|&i| p.ops[i].opcode == Opcode::Vstore)
        .filter(|&i| p.ops[i].srcs[0].as_vreg().and_then(|a| values.address(&g, a)).map_or(false, |a| a.word == 5))
        .collect()
}

/// Append a vector store of `value` to the address of index store `like`.
fn push_index_store(p: &mut SsaProgram, like: usize, value: VReg) {
    let mut st = p.ops[like].clone();
    st.srcs[1] = Operand::VReg(value);
    st.id = p.ops.len();
    p.ops.push(st);
}

#[test]
fn test_mirror_index_records_the_assumption_of_a_standalone_depth_1_index() {
    // A pruned program (no step relies on InitialIndicesZero) plus an index store of a
    // standalone depth-1 index. Built as `(loaded index + 2) - b` it is `2 - b` only because
    // the loaded index is 0: the assumption is reported. Built from a literal 2, nothing is.
    let spec = standard3();
    let pruned = apply_transform(raw_program(&spec), TransformKind::PruneIndex, Some(BATCH)).unwrap().0;
    let (_, base) = apply_mirror_index(clone_program(&pruned), &rule()).unwrap();
    assert!(base.refused.is_none() && base.n_sites() > 0, "{:?}", base.refused);
    assert!(base.assumptions().is_empty(), "the pruned program relies on nothing: {:?}", base.assumptions());
    let g = SsaGraph::new(&pruned.ops, &pruned.vector_group_lanes);
    let bit = rule().plan(&g, pruned.zero_vreg).unwrap().steps[0].bit;
    drop(g);
    let st = index_stores(&pruned)[0];
    let addr = pruned.ops[st].srcs[0].as_vreg().unwrap();
    for assumed in [true, false] {
        let mut p = clone_program(&pruned);
        let two = vconst(&p, 2);
        let base_value = if assumed {
            // The initial index, loaded right after its address is defined (before any store).
            let ld = fresh(&mut p, true);
            let at = p.ops.iter().position(|o| o.dest == Some(addr)).unwrap() + 1;
            p.ops.insert(at, SsaOp::new(0, Engine::Load, Opcode::Vload, Some(ld), vec![Operand::VReg(addr)]));
            renumber(&mut p);
            push_valu(&mut p, Opcode::Add, &[ld, two])
        } else {
            two
        };
        let z = push_valu(&mut p, Opcode::Sub, &[base_value, bit]);
        let st = index_stores(&p)[0];
        push_index_store(&mut p, st, z);
        renumber(&mut p);
        let (_, report) = apply_mirror_index(p, &rule()).unwrap();
        assert!(report.refused.is_none() && report.n_sites() > 0, "assumed {}: {:?}", assumed, report.refused);
        let expected: Vec<&str> = if assumed { vec!["initial-indices-zero"] } else { vec![] };
        assert_eq!(report.assumptions(), expected, "assumed {}", assumed);
    }
}

#[test]
fn test_mirror_index_refuses_a_stored_index_it_does_not_build() {
    // A second index family stored for group 0: `6 - (2*b0 + b1)`, a depth-2 zero-based index
    // that is no step's result. Rewriting the other chains would leave it zero-based: the
    // program is refused as a whole.
    let spec = standard3();
    let mut p = raw_program(&spec);
    let g = SsaGraph::new(&p.ops, &p.vector_group_lanes);
    let plan = rule().plan(&g, p.zero_vreg).unwrap();
    let bits: Vec<VReg> = plan.steps.iter().filter(|s| p.ops[s.op].region.and_then(|t| t.group) == Some(1)).map(|s| s.bit).collect();
    drop(g);
    let (two, six) = (vconst(&p, 2), vconst(&p, 6));
    let pos = push_valu(&mut p, Opcode::MultiplyAdd, &[bits[0], two, bits[1]]);
    let z = push_valu(&mut p, Opcode::Sub, &[six, pos]);
    let st = index_stores(&p)[0];
    push_index_store(&mut p, st, z);
    renumber(&mut p);
    assert_refused(p, &rule(), "no index value the rewrite builds");
}
