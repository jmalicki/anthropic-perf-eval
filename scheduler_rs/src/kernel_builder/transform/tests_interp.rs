//! A concrete SSA interpreter (test only) and the semantic check of the index transforms:
//! the program before and after each transform computes the same observable outputs on
//! random inputs, and agrees with the reference kernel.
//!
//! Not `transform::interp`: that one treats loads as inputs keyed by destination VReg and
//! stores as an ordered output list, which cannot express what these rules are about. The
//! index rules remove loads (the initial index load) and stores (the index stores), rely on
//! the contents of memory (initial indices 0), and the tree access gathers from addresses
//! computed from the index. This interpreter runs on a full memory image of `problem.py`
//! (`build_mem_image`) and compares the arrays the harness reads; it also supports
//! `SelectChoice` / `vselect`, `<`, `==` and vector groups, which the emitted kernels use.

use std::collections::HashMap;

use super::tests::{clone_program, raw_program};
use super::tests_index::configs;
use crate::kernel_builder::config::{ProblemSpec, TransformKind, HASH_STAGES};
use crate::kernel_builder::pipeline::passes::transform::apply_transform;
use crate::kernel_builder::SsaProgram;
use crate::types::{Opcode, Operand, VReg};

/// Deterministic pseudo-random words.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
}

/// The memory image of `problem.py` `build_mem_image`, with random tree and input values and
/// every index 0.
///
/// # Returns
/// `(mem, indices_p, values_p)`.
pub(super) fn memory(spec: &ProblemSpec, seed: u64) -> (Vec<u32>, usize, usize) {
    let mut rng = Rng(seed);
    let (n_nodes, batch) = (spec.n_nodes, spec.batch_size);
    let header = 8;
    let extra = n_nodes + batch * 2 + 16 + 32;
    let mut mem = vec![0u32; header + n_nodes + 2 * batch + extra];
    let (tree_p, idx_p) = (header, header + n_nodes);
    let val_p = idx_p + batch;
    mem[0] = spec.rounds as u32;
    mem[1] = n_nodes as u32;
    mem[2] = batch as u32;
    mem[3] = spec.forest_height as u32;
    mem[4] = tree_p as u32;
    mem[5] = idx_p as u32;
    mem[6] = val_p as u32;
    mem[7] = (val_p + batch) as u32;
    for i in 0..n_nodes {
        mem[tree_p + i] = rng.next();
    }
    for i in 0..batch {
        mem[val_p + i] = rng.next();
    }
    (mem, idx_p, val_p)
}

fn hash(mut a: u32) -> u32 {
    let f = |op: char, x: u32, y: u32| match op {
        '+' => x.wrapping_add(y),
        '^' => x ^ y,
        '<' => x << y,
        '>' => x >> y,
        _ => unreachable!(),
    };
    for &(op1, v1, op2, op3, v3) in HASH_STAGES.iter() {
        a = f(op2, f(op1, a, v1 as u32), f(op3, a, v3 as u32));
    }
    a
}

/// `problem.py` `reference_kernel` on the image.
///
/// # Returns
/// Final `(indices, values)`.
fn reference(mem: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let (rounds, n_nodes, batch) = (mem[0] as usize, mem[1], mem[2] as usize);
    let (tree_p, idx_p, val_p) = (mem[4] as usize, mem[5] as usize, mem[6] as usize);
    let mut idx: Vec<u32> = mem[idx_p..idx_p + batch].to_vec();
    let mut val: Vec<u32> = mem[val_p..val_p + batch].to_vec();
    for _ in 0..rounds {
        for i in 0..batch {
            val[i] = hash(val[i] ^ mem[tree_p + idx[i] as usize]);
            let next = 2 * idx[i] + if val[i] % 2 == 0 { 1 } else { 2 };
            idx[i] = if next >= n_nodes { 0 } else { next };
        }
    }
    (idx, val)
}

fn alu(op: Opcode, a: u32, b: u32) -> u32 {
    match op {
        Opcode::Add => a.wrapping_add(b),
        Opcode::Sub => a.wrapping_sub(b),
        Opcode::Mul => a.wrapping_mul(b),
        Opcode::Div => a / b,
        Opcode::Mod => a % b,
        Opcode::Xor => a ^ b,
        Opcode::And => a & b,
        Opcode::Or => a | b,
        Opcode::Shl => a.checked_shl(b).unwrap_or(0),
        Opcode::Shr => a.checked_shr(b).unwrap_or(0),
        Opcode::Lt => (a < b) as u32,
        Opcode::Eq => (a == b) as u32,
        other => panic!("interp: no ALU semantics for {:?}", other),
    }
}

/// Run `p` on `mem` (modified in place). Semantics follow `problem.py` (`alu`, `valu`, loads
/// and stores; results mod 2^32). `SelectChoice`, `vselect` and `select` are
/// `cond != 0 ? t : f` (the multiply-add encoding agrees for 0/1 conditions, which the
/// emitter guarantees). A decomposed vector defines its 8 lanes; reading a group alias
/// assembles its lanes. Barriers and control pseudo-ops are skipped.
///
/// # Arguments
/// * `p` - The program.
/// * `mem` - Main memory.
pub(super) fn run(p: &SsaProgram, mem: &mut [u32]) {
    let mut regs: HashMap<VReg, Vec<u32>> = HashMap::new();
    if let Some(z) = p.zero_vreg {
        regs.insert(z, vec![0]);
    }
    let read = |regs: &HashMap<VReg, Vec<u32>>, v: VReg| -> Vec<u32> {
        if let Some(x) = regs.get(&v) {
            return x.clone();
        }
        if let Some(lanes) = p.vector_group_lanes.get(&v) {
            return lanes
                .iter()
                .map(|l| regs.get(l).unwrap_or_else(|| panic!("interp: lane {:?} of {:?} unset", l, v))[0])
                .collect();
        }
        panic!("interp: {:?} read before its definition", v)
    };
    for op in &p.ops {
        let src = |k: usize, regs: &HashMap<VReg, Vec<u32>>| -> Vec<u32> {
            match op.srcs[k] {
                Operand::VReg(v) => read(regs, v),
                Operand::Imm(i) => vec![i as u32],
            }
        };
        let widen = |x: Vec<u32>, n: usize| -> Vec<u32> {
            if x.len() == 1 {
                vec![x[0]; n]
            } else {
                x
            }
        };
        let out: Option<Vec<u32>> = match op.opcode {
            Opcode::Barrier | Opcode::Halt | Opcode::Jump | Opcode::CondJump | Opcode::DebugPrint => None,
            Opcode::ConstChoice | Opcode::Const | Opcode::AddImm => {
                Some(vec![op.produces_constant.expect("constant op without its value") as u32])
            }
            Opcode::Vbroadcast => Some(vec![src(0, &regs)[0]; 8]),
            Opcode::Load => Some(vec![mem[src(0, &regs)[0] as usize]]),
            Opcode::Vload => {
                let a = src(0, &regs)[0] as usize;
                Some(mem[a..a + 8].to_vec())
            }
            Opcode::Store => {
                mem[src(0, &regs)[0] as usize] = src(1, &regs)[0];
                None
            }
            Opcode::Vstore => {
                let a = src(0, &regs)[0] as usize;
                mem[a..a + 8].copy_from_slice(&src(1, &regs));
                None
            }
            Opcode::MultiplyAdd => {
                let (a, b, c) = (src(0, &regs), src(1, &regs), src(2, &regs));
                Some((0..8).map(|i| a[i].wrapping_mul(b[i]).wrapping_add(c[i])).collect())
            }
            Opcode::SelectChoice | Opcode::Vselect | Opcode::Select => {
                let (c, t, f) = (src(0, &regs), src(1, &regs), src(2, &regs));
                let n = c.len().max(t.len()).max(f.len());
                let (c, t, f) = (widen(c, n), widen(t, n), widen(f, n));
                Some((0..n).map(|i| if c[i] != 0 { t[i] } else { f[i] }).collect())
            }
            _ => {
                let (a, b) = (src(0, &regs), src(1, &regs));
                assert_eq!(a.len(), b.len(), "interp: operand widths of {:?}", op.opcode);
                Some(a.iter().zip(&b).map(|(&x, &y)| alu(op.opcode, x, y)).collect())
            }
        };
        if let (Some(d), Some(v)) = (op.dest, out) {
            if let Some(ls) = p.vector_group_lanes.get(&d) {
                for (k, l) in ls.iter().enumerate() {
                    regs.insert(*l, vec![v[k]]);
                }
            }
            regs.insert(d, v);
        }
    }
}

pub(super) type Outputs = (Vec<u32>, Vec<u32>);

/// Run `p` on random inputs.
///
/// # Returns
/// Per seed, the program's final `(indices, values)` and the reference's.
pub(super) fn outputs(p: &SsaProgram, spec: &ProblemSpec, seeds: &[u64]) -> Vec<(Outputs, Outputs)> {
    seeds
        .iter()
        .map(|&seed| {
            let (mut mem, ip, vp) = memory(spec, seed);
            let expected = reference(&mem);
            run(p, &mut mem);
            let b = spec.batch_size;
            ((mem[ip..ip + b].to_vec(), mem[vp..vp + b].to_vec()), expected)
        })
        .collect()
}

/// The program before and after `transforms` computes the same observable outputs on random
/// inputs: the values always, the indices unless `values-only` is applied (then the index
/// memory must be left as it was). The program before agrees with the reference kernel, so
/// the interpreter is checked too.
fn check_semantics(spec: &ProblemSpec, transforms: &[TransformKind], label: &str) {
    let seeds = [1u64, 7, 12345];
    let base = raw_program(spec);
    let before = outputs(&base, spec, &seeds);
    let mut p = clone_program(&base);
    for &t in transforms {
        p = apply_transform(p, t, Some(spec.batch_size)).unwrap().0;
    }
    let after = outputs(&p, spec, &seeds);
    let values_only = transforms.contains(&TransformKind::ValuesOnly);
    for (k, ((b, expected), (a, _))) in before.iter().zip(&after).enumerate() {
        assert_eq!(b, expected, "{}: the emitted program disagrees with the reference (seed {})", label, seeds[k]);
        assert_eq!(a.1, b.1, "{} {:?}: values differ (seed {})", label, transforms, seeds[k]);
        if values_only {
            assert!(a.0.iter().all(|&i| i == 0), "{}: values-only must leave the index memory untouched", label);
        } else {
            assert_eq!(a.0, b.0, "{} {:?}: indices differ (seed {})", label, transforms, seeds[k]);
        }
    }
    eprintln!("{} {:?}: same outputs on {} inputs", label, transforms, seeds.len());
}

#[test]
fn test_index_transforms_preserve_outputs() {
    use TransformKind::*;
    for rounds in [2, 12, 16] {
        for (label, spec) in configs(rounds) {
            for ts in [vec![PruneIndex], vec![ValuesOnly], vec![PruneIndex, ValuesOnly], vec![HashMerge, PruneIndex, ValuesOnly]] {
                check_semantics(&spec, &ts, &label);
            }
        }
    }
}
