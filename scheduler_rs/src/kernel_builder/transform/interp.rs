//! A small concrete interpreter for SSA programs, to test that a rewrite preserves values.
//!
//! It runs the ops in order on 32-bit lanes (8 for a vector VReg, 1 for a scalar), with the
//! machine's semantics (wrapping arithmetic, shifts of 32 or more give 0). Loads are the
//! program's inputs: a load's lanes come from a caller-supplied function of its destination
//! VReg (loads are never renamed by a rewrite, so both programs see the same inputs). Stores
//! are the outputs, recorded in program order. A rewrite rule's test builds a program,
//! rewrites it, runs both and compares the stores (see `tests.rs`,
//! `test_hash_merge_preserves_values_*`). Later rules extend `step` with the opcodes they
//! need; an opcode it does not know is an error, never a guess.
//!
//! Supported: `ConstChoice` / `Const` (the immediate), `vbroadcast`, `+ - * ^ & | << >>` on
//! the ALU and vector engines, `multiply_add`, `load` / `vload` (inputs), `store` / `vstore`
//! (outputs), barriers (no effect).

use std::collections::HashMap;

use crate::types::{Opcode, Operand, SsaOp, VReg, VRegWidth};

/// Lanes of a VReg.
fn width(v: VReg) -> usize {
    match v.width {
        VRegWidth::Scalar => 1,
        VRegWidth::Vector => 8,
    }
}

/// Result of a run.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Run {
    /// The value of every VReg defined, lane by lane.
    pub values: HashMap<VReg, Vec<u32>>,
    /// The value each store writes, in program order.
    pub stores: Vec<Vec<u32>>,
}

fn binop(op: Opcode, a: u32, b: u32) -> Result<u32, String> {
    Ok(match op {
        Opcode::Add => a.wrapping_add(b),
        Opcode::Sub => a.wrapping_sub(b),
        Opcode::Mul => a.wrapping_mul(b),
        Opcode::Xor => a ^ b,
        Opcode::And => a & b,
        Opcode::Or => a | b,
        Opcode::Shl => if b >= 32 { 0 } else { a << b },
        Opcode::Shr => if b >= 32 { 0 } else { a >> b },
        other => return Err(format!("interp: binary {:?} not supported", other)),
    })
}

/// Run `ops` in order.
///
/// # Arguments
/// * `ops` - The program (def before use).
/// * `input` - Lanes of a load's destination: `input(dest, lane)`.
///
/// # Returns
/// Every VReg's value and the stores in order, or an error for an unsupported op or a read
/// of a value not yet computed.
pub fn run(ops: &[SsaOp], input: impl Fn(VReg, usize) -> u32) -> Result<Run, String> {
    let mut r = Run::default();
    for (i, op) in ops.iter().enumerate() {
        let get = |r: &Run, s: &Operand| -> Result<Vec<u32>, String> {
            match s {
                Operand::VReg(v) => r.values.get(v).cloned().ok_or_else(|| format!("interp: op {} reads {:?} before it is computed", i, v)),
                Operand::Imm(k) => Ok(vec![*k as u32]),
            }
        };
        let lane = |x: &[u32], k: usize| if x.len() == 1 { x[0] } else { x[k] };
        let out: Option<Vec<u32>> = match op.opcode {
            Opcode::Barrier => None,
            Opcode::ConstChoice | Opcode::Const => match op.srcs.first() {
                Some(Operand::Imm(k)) => Some(vec![*k as u32]),
                _ => return Err(format!("interp: op {} constant without an immediate", i)),
            },
            Opcode::Vbroadcast => Some(vec![get(&r, &op.srcs[0])?[0]; 8]),
            Opcode::Load | Opcode::Vload => {
                let d = op.dest.ok_or("interp: load without a dest")?;
                Some((0..width(d)).map(|k| input(d, k)).collect())
            }
            Opcode::Store | Opcode::Vstore => {
                let v = get(&r, op.srcs.last().ok_or("interp: store without a value")?)?;
                r.stores.push(v);
                None
            }
            Opcode::MultiplyAdd => {
                let (a, b, c) = (get(&r, &op.srcs[0])?, get(&r, &op.srcs[1])?, get(&r, &op.srcs[2])?);
                let n = width(op.dest.ok_or("interp: multiply_add without a dest")?);
                Some((0..n).map(|k| lane(&a, k).wrapping_mul(lane(&b, k)).wrapping_add(lane(&c, k))).collect())
            }
            opc => {
                if op.srcs.len() != 2 {
                    return Err(format!("interp: op {} {:?} not supported", i, opc));
                }
                let (a, b) = (get(&r, &op.srcs[0])?, get(&r, &op.srcs[1])?);
                let n = width(op.dest.ok_or_else(|| format!("interp: op {} without a dest", i))?);
                Some((0..n).map(|k| binop(opc, lane(&a, k), lane(&b, k))).collect::<Result<_, _>>()?)
            }
        };
        if let (Some(v), Some(d)) = (out, op.dest) {
            r.values.insert(d, v);
        }
    }
    Ok(r)
}
