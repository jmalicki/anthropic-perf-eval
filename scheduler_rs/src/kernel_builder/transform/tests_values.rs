//! Tests of the value and address facts the index rules rely on (`values.rs`) and of the
//! values-only alias check (review of PR #77): later loads are disjoint from an index store
//! only when proved inside their own array; addresses wrap mod 2^32; constant folding follows
//! `problem.py` (results mod 2^32, no overflow); header pointers are not trusted after a store
//! that may have rewritten the header; rule-5 (CSE) sites carry their own justification.

use std::collections::BTreeMap;

use super::graph::SsaGraph;
use super::prune_index::{PruneIndexRule, PruneKind};
use super::tests::{clone_program, raw_program};
use super::values::{Assumption, MemoryLayout, ValueFacts};
use super::values_only::ValuesOnlyRule;
use crate::kernel_builder::config::ProblemSpec;
use crate::kernel_builder::SsaProgram;
use crate::types::{Engine, Opcode, Operand, SsaOp, VReg};

const MASK: u64 = 0xFFFF_FFFF;

/// Small program builder: appends ops to `p` with fresh VRegs.
struct Append<'a>(&'a mut SsaProgram);

impl Append<'_> {
    fn fresh(&mut self, vector: bool) -> VReg {
        let id = self.0.vreg_count;
        self.0.vreg_count += 1;
        if vector { VReg::vector(id) } else { VReg::scalar(id) }
    }
    fn push(&mut self, engine: Engine, opcode: Opcode, dest: Option<VReg>, srcs: Vec<Operand>) -> Option<VReg> {
        let mut op = SsaOp::new(self.0.ops.len(), engine, opcode, dest, srcs);
        op.id = self.0.ops.len();
        if let (Opcode::ConstChoice, Some(Operand::Imm(k))) = (opcode, op.srcs.first()) {
            op.produces_constant = Some(*k);
        }
        if let Some(d) = dest {
            self.0.vreg_def.insert(d, op.id);
        }
        self.0.ops.push(op);
        dest
    }
    fn konst(&mut self, k: i64) -> VReg {
        let d = self.fresh(false);
        self.push(Engine::Load, Opcode::ConstChoice, Some(d), vec![Operand::Imm(k)]).unwrap()
    }
    /// `mem[word]`, a header pointer load.
    fn header(&mut self, word: i64) -> VReg {
        let w = self.konst(word);
        let d = self.fresh(false);
        self.push(Engine::Load, Opcode::Load, Some(d), vec![Operand::VReg(w)]).unwrap()
    }
    fn add(&mut self, a: VReg, b: VReg) -> VReg {
        let d = self.fresh(false);
        self.push(Engine::Alu, Opcode::Add, Some(d), vec![Operand::VReg(a), Operand::VReg(b)]).unwrap()
    }
    /// `vload(mem[word] + k)`, all ops appended.
    fn vload_at(&mut self, word: i64, k: i64) -> VReg {
        let p = self.header(word);
        let c = self.konst(k);
        let a = self.add(p, c);
        self.vload(a)
    }
    fn vload(&mut self, a: VReg) -> VReg {
        let d = self.fresh(true);
        self.push(Engine::Load, Opcode::Vload, Some(d), vec![Operand::VReg(a)]).unwrap()
    }
}

/// Renumber op ids in order and rebuild `vreg_def`.
fn renumber(p: &mut SsaProgram) {
    for (i, op) in p.ops.iter_mut().enumerate() {
        op.id = i;
    }
    p.vreg_def = p.ops.iter().filter_map(|o| o.dest.map(|d| (d, o.id))).collect();
}

/// The address `mem[word] + k1 (+ k2 ...)`, computed at the start of the program (before any
/// store, as a kernel computes its pointers), then `vload` of it appended after the index
/// stores.
fn late_load_of_early_address(p: &mut SsaProgram, word: i64, offsets: &[i64]) -> VReg {
    let mut front = SsaProgram { ops: Vec::new(), vreg_count: p.vreg_count, ..clone_program(p) };
    let addr = {
        let mut a = Append(&mut front);
        let mut x = a.header(word);
        for &k in offsets.iter() {
            let c = a.konst(k);
            x = a.add(x, c);
        }
        x
    };
    p.vreg_count = front.vreg_count;
    let mut ops = front.ops;
    ops.append(&mut p.ops);
    p.ops = ops;
    renumber(p);
    Append(p).vload(addr);
    renumber(p);
    addr
}

fn values_only_sites(p: &SsaProgram, rule: &ValuesOnlyRule) -> (usize, BTreeMap<String, usize>) {
    let g = SsaGraph::new(&p.ops, &p.vector_group_lanes);
    let (sites, rejected) = rule.find_sites(&g, p.zero_vreg);
    (sites.len(), rejected)
}

fn read_back(rejected: &BTreeMap<String, usize>) -> usize {
    rejected.iter().filter(|(r, _)| r.contains("may read the stored indices back")).map(|(_, n)| n).sum()
}

fn base() -> SsaProgram {
    raw_program(&ProblemSpec { rounds: 2, ..Default::default() })
}

// ----------------------------------------------------------------------------
// Finding 1: a different header word is not a proof of disjointness
// ----------------------------------------------------------------------------

#[test]
fn test_values_only_refuses_a_tree_load_past_the_tree() {
    // problem.py: indices_p = tree_p + n_nodes, so mem[mem[4]] + n_nodes + 8 is indices word 8,
    // which group 1's index store writes. The load is not proved inside the tree: every index
    // store may be read back and is refused.
    let mut p = base();
    late_load_of_early_address(&mut p, 4, &[2047 + 8]);
    let (sites, rejected) = values_only_sites(&p, &ValuesOnlyRule::for_problem(256));
    assert_eq!(sites, 0, "{:?}", rejected);
    assert_eq!(read_back(&rejected), 32, "{:?}", rejected);
}

#[test]
fn test_values_only_refuses_a_values_load_before_the_values() {
    // mem[mem[6]] - batch + 8: indices word 8 again, reached from the values pointer.
    let mut p = base();
    late_load_of_early_address(&mut p, 6, &[(-(256i64) + 8) & MASK as i64]);
    let (sites, rejected) = values_only_sites(&p, &ValuesOnlyRule::for_problem(256));
    assert_eq!(sites, 0, "{:?}", rejected);
    assert_eq!(read_back(&rejected), 32);
}

#[test]
fn test_values_only_refuses_a_load_from_a_non_pointer_header_word() {
    // mem[0] is the round count, not a pointer: nothing is known about mem[mem[0]] + 8.
    let mut p = base();
    late_load_of_early_address(&mut p, 0, &[8]);
    let (sites, rejected) = values_only_sites(&p, &ValuesOnlyRule::for_problem(256));
    assert_eq!(sites, 0, "{:?}", rejected);
}

#[test]
fn test_values_only_keeps_disjoint_located_loads() {
    // A later load inside the values array, or inside the indices but away from a store, is
    // proved disjoint (precision is kept where the proof exists).
    let mut p = base();
    late_load_of_early_address(&mut p, 6, &[16]);
    let (sites, _) = values_only_sites(&p, &ValuesOnlyRule::for_problem(256));
    assert_eq!(sites, 32);
    let mut p = base();
    late_load_of_early_address(&mut p, 5, &[16]);
    let (sites, rejected) = values_only_sites(&p, &ValuesOnlyRule::for_problem(256));
    assert_eq!((sites, read_back(&rejected)), (31, 1));
}

// ----------------------------------------------------------------------------
// Finding 2: offsets wrap mod 2^32
// ----------------------------------------------------------------------------

#[test]
fn test_addresses_wrap_mod_2_32() {
    // (p5 + 0xFFFFFFF8) + 16 is p5 + 8 on the machine: group 1's index words.
    let mut p = base();
    let y = late_load_of_early_address(&mut p, 5, &[0xFFFF_FFF8, 16]);
    {
        let g = SsaGraph::new(&p.ops, &p.vector_group_lanes);
        let values = ValueFacts::compute(&g, p.zero_vreg, &MemoryLayout::problem(256), &[]);
        let addr = values.address(&g, y).unwrap();
        assert_eq!((addr.word, addr.offset), (5, 8));
    }
    let (sites, rejected) = values_only_sites(&p, &ValuesOnlyRule::for_problem(256));
    assert_eq!((sites, read_back(&rejected)), (31, 1), "{:?}", rejected);
}

// ----------------------------------------------------------------------------
// Finding 6 (part): header words may be rewritten by a store whose address is unknown
// ----------------------------------------------------------------------------

#[test]
fn test_header_pointer_not_trusted_after_an_unknown_store() {
    // A store through an address that is not located (here mem[mem[0]]) may rewrite mem[5]:
    // a later mem[5] + 8 is no longer an address of the indices.
    let mut p = base();
    let mut a = Append(&mut p);
    let p0 = a.header(0);
    let v = a.fresh(true);
    let z = a.konst(0);
    a.push(Engine::Valu, Opcode::Vbroadcast, Some(v), vec![Operand::VReg(z)]);
    a.push(Engine::Store, Opcode::Vstore, None, vec![Operand::VReg(p0), Operand::VReg(v)]);
    let p5 = a.header(5);
    let c = a.konst(8);
    let x = a.add(p5, c);
    let g = SsaGraph::new(&p.ops, &p.vector_group_lanes);
    let values = ValueFacts::compute(&g, p.zero_vreg, &MemoryLayout::problem(256), &[]);
    assert_eq!(values.address(&g, x), None);
    // Before such a store, the same shape is an address.
    let early = p.ops.iter().find(|o| o.opcode == Opcode::Vstore).and_then(|o| o.srcs[0].as_vreg()).unwrap();
    assert!(values.address(&g, early).is_some());
}

#[test]
fn test_address_of_a_long_add_chain_is_linear() {
    // x_(i+1) = x_i + x_i: each step reads a non-constant twice. A recursive lookup without a
    // cache tries both operand orders and doubles per level (2^200 here); the memoised one is
    // linear.
    let mut p = base();
    let mut a = Append(&mut p);
    let mut x = a.header(5);
    for _ in 0..200 {
        x = a.add(x, x);
    }
    let g = SsaGraph::new(&p.ops, &p.vector_group_lanes);
    let values = ValueFacts::compute(&g, p.zero_vreg, &MemoryLayout::problem(256), &[]);
    assert_eq!(values.address(&g, x), None);
}

// ----------------------------------------------------------------------------
// Finding 3: folding matches problem.py (mod 2^32) and never overflows
// ----------------------------------------------------------------------------

/// problem.py `alu` on 32-bit words: Python integers, then `% 2**32`.
fn reference(op: Opcode, a: u64, b: u64) -> Option<u64> {
    let (a, b) = (a as u128, b as u128);
    let r: u128 = match op {
        Opcode::Add => a + b,
        Opcode::Sub => a.wrapping_sub(b) & MASK as u128,
        Opcode::Mul => a * b,
        Opcode::Div => a.checked_div(b)?,
        Opcode::Mod => a.checked_rem(b)?,
        Opcode::Xor => a ^ b,
        Opcode::And => a & b,
        Opcode::Or => a | b,
        Opcode::Shl => {
            if b >= 64 {
                0
            } else {
                (a << b) & MASK as u128
            }
        }
        Opcode::Shr => {
            if b >= 64 {
                0
            } else {
                a >> b
            }
        }
        Opcode::Lt => (a < b) as u128,
        Opcode::Eq => (a == b) as u128,
        _ => return None,
    };
    Some((r & MASK as u128) as u64)
}

/// Fold `op` on constants through `ValueFacts` (vector ops on broadcasts).
fn folded(op: Opcode, a: u64, b: u64) -> Option<u64> {
    let mut p = base();
    let mut ap = Append(&mut p);
    let (sa, sb) = (ap.konst(a as i64), ap.konst(b as i64));
    let (va, vb) = (ap.fresh(true), ap.fresh(true));
    ap.push(Engine::Valu, Opcode::Vbroadcast, Some(va), vec![Operand::VReg(sa)]);
    ap.push(Engine::Valu, Opcode::Vbroadcast, Some(vb), vec![Operand::VReg(sb)]);
    let d = ap.fresh(true);
    ap.push(Engine::Valu, op, Some(d), vec![Operand::VReg(va), Operand::VReg(vb)]);
    let g = SsaGraph::new(&p.ops, &p.vector_group_lanes);
    let values = ValueFacts::compute(&g, p.zero_vreg, &MemoryLayout::problem(256), &[]);
    values.value(&g, d).map(|v| v as u64)
}

#[test]
fn test_fold_matches_problem_py() {
    use Opcode::*;
    let edges = [0u64, 1, 2, 31, 32, 33, 0x7FFF_FFFF, 0x8000_0000, 0xFFFF_FFFE, 0xFFFF_FFFF];
    for op in [Add, Sub, Mul, Div, Mod, Xor, And, Or, Shl, Shr, Lt, Eq] {
        for &a in &edges {
            for &b in &edges {
                assert_eq!(folded(op, a, b), reference(op, a, b), "{:?}({:#x}, {:#x})", op, a, b);
            }
        }
    }
}

#[test]
fn test_fold_absorbs_zero_and_folds_multiply_add() {
    // x * 0 and x & 0 are 0 with x unknown (a load); multiply_add(0, x, c) = c; all-constant
    // multiply_add wraps mod 2^32 without overflowing (0x80000000 * 0xFFFFFFFF + 0xFFFFFFFF
    // exceeds i64 before the mask).
    let mut p = base();
    let mut a = Append(&mut p);
    let x = a.vload_at(6, 0);
    let z = a.konst(0);
    let vz = a.fresh(true);
    a.push(Engine::Valu, Opcode::Vbroadcast, Some(vz), vec![Operand::VReg(z)]);
    let mul = a.fresh(true);
    a.push(Engine::Valu, Opcode::Mul, Some(mul), vec![Operand::VReg(x), Operand::VReg(vz)]);
    let and = a.fresh(true);
    a.push(Engine::Valu, Opcode::And, Some(and), vec![Operand::VReg(vz), Operand::VReg(x)]);
    let c = a.konst(12345);
    let vc = a.fresh(true);
    a.push(Engine::Valu, Opcode::Vbroadcast, Some(vc), vec![Operand::VReg(c)]);
    let fz = a.fresh(true);
    a.push(Engine::Valu, Opcode::MultiplyAdd, Some(fz), vec![Operand::VReg(vz), Operand::VReg(x), Operand::VReg(vc)]);
    let mut vk = Vec::new();
    for k in [0x8000_0000i64, 0xFFFF_FFFF, 0xFFFF_FFFF] {
        let s = a.konst(k);
        let v = a.fresh(true);
        a.push(Engine::Valu, Opcode::Vbroadcast, Some(v), vec![Operand::VReg(s)]);
        vk.push(v);
    }
    let fk = a.fresh(true);
    a.push(Engine::Valu, Opcode::MultiplyAdd, Some(fk), vk.iter().map(|&v| Operand::VReg(v)).collect());
    let g = SsaGraph::new(&p.ops, &p.vector_group_lanes);
    let values = ValueFacts::compute(&g, p.zero_vreg, &MemoryLayout::problem(256), &[]);
    assert_eq!(values.value(&g, x), None);
    assert_eq!(values.value(&g, mul), Some(0));
    assert_eq!(values.value(&g, and), Some(0));
    assert_eq!(values.value(&g, fz), Some(12345));
    let expect = ((0x8000_0000u128 * 0xFFFF_FFFF + 0xFFFF_FFFF) & MASK as u128) as i64;
    assert_eq!(values.value(&g, fk), Some(expect));
}

// ----------------------------------------------------------------------------
// Finding 5: rule-5 sites are CSE, not R12
// ----------------------------------------------------------------------------

#[test]
fn test_shared_constant_sites_cite_cse_not_r12() {
    let p = raw_program(&ProblemSpec::default());
    let g = SsaGraph::new(&p.ops, &p.vector_group_lanes);
    let (sites, _) = PruneIndexRule::for_problem(256).find_sites(&g, p.zero_vreg);
    let proofs = |kind: PruneKind| -> Vec<String> {
        sites.iter().filter(|s| s.kind == kind).flat_map(|s| s.facts.iter().filter(|f| f.name == "proofs").map(|f| f.value.clone())).collect()
    };
    let shared = proofs(PruneKind::SharedConstant);
    let zero = proofs(PruneKind::ZeroUpdate);
    assert_eq!(shared.len(), 31);
    assert!(shared.iter().all(|p| p == "CSE-identical-broadcast"), "{:?}", &shared[..1]);
    assert_eq!(zero.len(), 64);
    assert!(zero.iter().all(|p| p.contains("R12-fma-zero-factor") && !p.contains("CSE")), "{:?}", &zero[..1]);
    let _ = (clone_program, Assumption::InitialIndicesZero);
}
