//! A concrete SSA interpreter (test only) and the semantic check of the index transforms:
//! the program before and after each transform computes the same observable outputs on
//! random inputs, and agrees with the reference kernel.
//!
//! The interpreter is `transform::interp` (one for every rule; it follows lowering and models
//! memory). This module runs it on a full memory image of `problem.py` (`build_mem_image`) and
//! compares the arrays the harness reads.

use super::tests::{clone_program, raw_program};
use super::tests_index::configs;
use crate::kernel_builder::config::{ProblemSpec, TransformKind, HASH_STAGES};
use crate::kernel_builder::pipeline::passes::transform::apply_transform;
use crate::kernel_builder::SsaProgram;

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

/// Run `p` on `mem` (modified in place) with `transform::interp::run`; an error (an op it
/// refuses, SelectChoice lowerings that disagree, ...) fails the test.
///
/// # Arguments
/// * `p` - The program.
/// * `mem` - Main memory.
pub(super) fn run(p: &SsaProgram, mem: &mut [u32]) {
    super::interp::run(p, mem).unwrap_or_else(|e| panic!("{}", e));
}

/// `run` following one `SelectChoice` lowering (see `interp::SelectLowering`).
pub(super) fn run_with(p: &SsaProgram, mem: &mut [u32], select: super::interp::SelectLowering) {
    super::interp::run_with(p, mem, select).unwrap_or_else(|e| panic!("{}", e));
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

/// `outputs` following one `SelectChoice` lowering (see `interp::SelectLowering`).
pub(super) fn outputs_with(p: &SsaProgram, spec: &ProblemSpec, seeds: &[u64], select: super::interp::SelectLowering) -> Vec<(Outputs, Outputs)> {
    seeds
        .iter()
        .map(|&seed| {
            let (mut mem, ip, vp) = memory(spec, seed);
            let expected = reference(&mem);
            run_with(p, &mut mem, select);
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
