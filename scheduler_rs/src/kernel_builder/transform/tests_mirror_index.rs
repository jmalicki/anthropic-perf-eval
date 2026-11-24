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
    }
    let (a, ra) = apply_mirror_index(clone_program(&bare), &rule()).unwrap();
    let (mut b, rb) = apply_mirror_index(tagged, &rule()).unwrap();
    assert_eq!(ra.sites, rb.sites);
    assert_eq!(ra.sites.get("step"), Some(&(15 * 32)));
    for op in &mut b.ops {
        op.region = None;
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
    assert_eq!(canonical_ops(&after), before);
}
