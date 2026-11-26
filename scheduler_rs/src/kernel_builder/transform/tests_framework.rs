//! Framework checks from the final-gate review of #75 and #77: the surgery refuses what it
//! cannot edit safely (a too-small vreg_count; an edit dropping a kept-live value, a traced op
//! or a lane read through a group key) for every rule; the value facts read addresses as
//! lowering does; the one interpreter follows lowering; the semantic check also runs with the
//! layout the pass uses (tree extent known).

use super::graph::SsaGraph;
use super::hash_merge::{apply_hash_merge, HashMergeRule, ProposalSource};
use super::interp;
use super::surgery::Surgery;
use super::tests::{canonical_ops, clone_program, raw_program};
use super::tests_interp::outputs;
use super::values::{MemoryLayout, ValueFacts};
use super::values_only::{apply_values_only, ValuesOnlyRule};
use crate::kernel_builder::config::{ProblemSpec, TransformKind};
use crate::kernel_builder::pipeline::passes::transform::{apply_transform, apply_transform_with};
use crate::kernel_builder::SsaProgram;
use crate::types::{Engine, Opcode, Operand, SsaOp, VReg};

fn two_rounds() -> SsaProgram {
    raw_program(&ProblemSpec { rounds: 2, ..Default::default() })
}

fn first_hash_site(p: &SsaProgram) -> super::hash_merge::HashMergeSite {
    let g = SsaGraph::new(&p.ops, &p.vector_group_lanes);
    HashMergeRule { source: ProposalSource::Ops }.find_sites(&g).0.remove(0)
}

#[test]
fn test_surgery_refuses_a_vreg_count_not_above_the_ids_in_use() {
    let mut p = two_rounds();
    p.vreg_count = 10;
    let err = Surgery::check(&p).unwrap_err();
    assert!(err.contains("vreg_count"), "{}", err);
    // The rules leave such a program unchanged with the reason.
    let before = canonical_ops(&p);
    let (after, outcome) = apply_transform(p, TransformKind::PruneIndex, Some(256)).unwrap();
    assert!(outcome.refused.as_deref().map_or(false, |r| r.contains("vreg_count")), "{:?}", outcome.refused);
    assert_eq!(canonical_ops(&after), before);
}

#[test]
fn test_hash_merge_keeps_a_kept_live_constant_it_would_drop() {
    // D (stage 3's constant broadcast) dies when every site is merged. Named in keep_live it
    // is kept (the surgery's liveness counts keep_live), and the sites are still merged.
    let mut p = two_rounds();
    let d = first_hash_site(&p).operands[2];
    p.ops.last_mut().unwrap().keep_live = Some(vec![d]);
    let (after, report) = apply_hash_merge(p, &HashMergeRule::default(), |_| true).unwrap();
    assert!(report.refused.is_none(), "{:?}", report.refused);
    assert_eq!(report.sites, 64);
    assert!(after.ops.iter().any(|o| o.dest == Some(d)), "the kept-live constant was dropped");
    super::surgery::Surgery::check(&after).unwrap();
}

#[test]
fn test_hash_merge_refuses_dropping_a_lane_of_a_write_group() {
    // The first site's result r is listed as a lane of a write-group alias (no definer) that a
    // store reads. The matcher only sees lanes of defined vectors; the rename of r would leave
    // the alias's lane undefined.
    let mut p = two_rounds();
    let site = first_hash_site(&p);
    let r = p.ops[site.xor].dest.unwrap();
    let alias = VReg::vector(p.vreg_count);
    p.vreg_count += 1;
    let mut lanes = vec![r];
    lanes.extend(p.ops.iter().filter_map(|o| o.dest).filter(|d| d.is_vector() && *d != r).take(7));
    p.vector_group_lanes.insert(alias, lanes);
    let st = p.ops.iter().rposition(|o| o.opcode == Opcode::Vstore).unwrap();
    let mut store = p.ops[st].clone();
    store.srcs[1] = Operand::VReg(alias);
    store.id = p.ops.len();
    p.ops.push(store);
    let before = canonical_ops(&p);
    let (after, report) = apply_hash_merge(p, &HashMergeRule::default(), |_| true).unwrap();
    assert!(report.refused.as_deref().map_or(false, |m| m.contains("vector group")), "{:?}", report.refused);
    assert_eq!(canonical_ops(&after), before);
}

#[test]
fn test_values_only_refuses_removing_a_traced_index_store() {
    let mut p = two_rounds();
    let g = SsaGraph::new(&p.ops, &p.vector_group_lanes);
    let store = ValuesOnlyRule::for_problem(256).find_sites(&g, p.zero_vreg).0[0].store;
    p.ops[store].trace_id = Some("idx".into());
    let before = canonical_ops(&p);
    let (after, report) = apply_values_only(p, &ValuesOnlyRule::for_problem(256), |_| true).unwrap();
    assert!(report.refused.as_deref().map_or(false, |r| r.contains("trace id")), "{:?}", report.refused);
    assert_eq!(canonical_ops(&after), before);
}

#[test]
fn test_a_load_with_an_immediate_address_operand_is_not_a_header_pointer() {
    // Lowering reads load(Imm k) as mem[scratch[k]]: only k = 0 is known (scratch[0] = 0).
    let mut p = two_rounds();
    let (a, b) = (VReg::scalar(p.vreg_count), VReg::scalar(p.vreg_count + 1));
    p.vreg_count += 2;
    for (d, k) in [(a, 5), (b, 0)] {
        let mut op = SsaOp::new(0, Engine::Load, Opcode::Load, Some(d), vec![Operand::Imm(k)]);
        op.id = p.ops.len();
        p.ops.insert(0, op);
    }
    for (i, op) in p.ops.iter_mut().enumerate() {
        op.id = i;
    }
    let g = SsaGraph::new(&p.ops, &p.vector_group_lanes);
    let values = ValueFacts::compute(&g, p.zero_vreg, &MemoryLayout::problem(256), &[]);
    assert_eq!(values.address(&g, a), None);
    assert_eq!(values.address(&g, b).map(|x| (x.word, x.offset)), Some((0, 0)));
}

#[test]
fn test_values_only_treats_a_later_load_offset_as_unknown() {
    let mut p = two_rounds();
    let addr = p.ops.iter().find(|o| o.opcode == Opcode::Vstore).unwrap().srcs[0].as_vreg().unwrap();
    let d = VReg::scalar(p.vreg_count);
    p.vreg_count += 1;
    let mut op = SsaOp::new(0, Engine::Load, Opcode::LoadOffset, Some(d), vec![Operand::VReg(addr), Operand::Imm(3)]);
    op.id = p.ops.len();
    p.ops.push(op);
    let g = SsaGraph::new(&p.ops, &p.vector_group_lanes);
    let (sites, rejected) = ValuesOnlyRule::for_problem(256).find_sites(&g, p.zero_vreg);
    assert!(sites.is_empty(), "{:?}", rejected);
}

#[test]
fn test_index_transforms_keep_outputs_with_the_tree_extent_known() {
    // TransformPass passes n_nodes (tree accesses can be located); check_semantics in
    // tests_interp runs without it. Same check with it.
    for rounds in [2, 12, 16] {
        for (label, spec) in super::tests_index::configs(rounds) {
            for ts in [vec![TransformKind::PruneIndex], vec![TransformKind::ValuesOnly], vec![TransformKind::PruneIndex, TransformKind::ValuesOnly]] {
                let p = raw_program(&spec);
                let mut q = clone_program(&p);
                for &t in &ts {
                    let (next, outcome) = apply_transform_with(q, t, Some(spec.batch_size), Some(spec.n_nodes)).unwrap();
                    assert!(outcome.refused.is_none(), "{} {:?}: refused: {:?}", label, t, outcome.refused);
                    // values-only always has index stores to drop; prune-index has nothing to
                    // prune on a mirrored index.
                    let expect_sites = t == TransformKind::ValuesOnly || spec.index_form != crate::kernel_builder::config::IndexForm::Mirrored;
                    assert_eq!(outcome.sites > 0, expect_sites, "{} {:?}: {}", label, t, outcome.summary);
                    q = next;
                }
                let seeds = [4u64, 40];
                let (b, a) = (outputs(&p, &spec, &seeds), outputs(&q, &spec, &seeds));
                for ((bo, _), (ao, _)) in b.iter().zip(&a) {
                    assert_eq!(ao.1, bo.1, "{} {:?}: values", label, ts);
                    if ts.contains(&TransformKind::ValuesOnly) {
                        // The index stores are gone: the index memory keeps its initial zeros.
                        assert!(ao.0.iter().all(|&i| i == 0), "{} {:?}: values-only must leave the index memory untouched", label, ts);
                    } else {
                        assert_eq!(ao.0, bo.0, "{} {:?}: indices", label, ts);
                    }
                }
            }
        }
    }
}

// The interpreter follows lowering: it refuses what lowering would read differently.

fn tiny(ops: Vec<SsaOp>, n: u32) -> SsaProgram {
    let vreg_def = ops.iter().filter_map(|o| o.dest.map(|d| (d, o.id))).collect();
    SsaProgram { ops, vreg_def, vreg_count: n, const_alternatives: Vec::new(), zero_vreg: None, vector_group_lanes: Default::default() }
}

fn konst(id: usize, d: VReg, k: i64) -> SsaOp {
    let mut op = SsaOp::new(id, Engine::Load, Opcode::ConstChoice, Some(d), vec![Operand::Imm(k)]);
    op.produces_constant = Some(k);
    op
}

#[test]
fn test_interp_refuses_a_scalar_operand_of_a_vector_op() {
    let (s, v, w) = (VReg::scalar(0), VReg::vector(1), VReg::vector(2));
    let p = tiny(vec![
        konst(0, s, 3),
        SsaOp::new(1, Engine::Valu, Opcode::Vbroadcast, Some(v), vec![Operand::VReg(s)]),
        SsaOp::new(2, Engine::Valu, Opcode::Add, Some(w), vec![Operand::VReg(v), Operand::VReg(s)]),
    ], 3);
    let err = interp::run(&p, &mut vec![0; 16]).unwrap_err();
    assert!(err.contains("mixes widths"), "{}", err);
}

#[test]
fn test_interp_refuses_an_immediate_address() {
    let d = VReg::scalar(0);
    let p = tiny(vec![SsaOp::new(0, Engine::Load, Opcode::Load, Some(d), vec![Operand::Imm(3)])], 1);
    let err = interp::run(&p, &mut vec![0; 16]).unwrap_err();
    assert!(err.contains("immediate operand"), "{}", err);
}

#[test]
fn test_interp_checks_both_select_lowerings() {
    // cond = 2 (not 0/1): vselect gives t, the multiply_add form 2*(t - f) + f; they disagree.
    let (c, t, f, cv, tv, fv, out) =
        (VReg::scalar(0), VReg::scalar(1), VReg::scalar(2), VReg::vector(3), VReg::vector(4), VReg::vector(5), VReg::vector(6));
    let mut ops = vec![konst(0, c, 2), konst(1, t, 10), konst(2, f, 3)];
    for (i, (s, v)) in [(c, cv), (t, tv), (f, fv)].into_iter().enumerate() {
        ops.push(SsaOp::new(3 + i, Engine::Valu, Opcode::Vbroadcast, Some(v), vec![Operand::VReg(s)]));
    }
    ops.push(SsaOp::new(6, Engine::Valu, Opcode::SelectChoice, Some(out), vec![Operand::VReg(cv), Operand::VReg(tv), Operand::VReg(fv)]));
    let err = interp::run(&tiny(ops.clone(), 7), &mut vec![0; 16]).unwrap_err();
    assert!(err.contains("lowerings disagree"), "{}", err);
    // cond = 1 with a delta that is t - f: they agree.
    ops[0] = konst(0, c, 1);
    assert!(interp::run(&tiny(ops, 7), &mut vec![0; 16]).is_ok());
}
