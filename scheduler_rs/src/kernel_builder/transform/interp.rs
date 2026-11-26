//! The concrete interpreter of SSA programs that tests use to check that a rewrite keeps a
//! program's outputs. One interpreter for every rule, following lowering
//! (`address_assign::lower_ssa_op`, `ssa_lower.rs`) and the machine (`problem.py`):
//!
//! * Values are 32-bit lanes: 8 for a vector VReg, 1 for a scalar. Arithmetic wraps; shifts
//!   of 32 or more give 0; division by zero is an error.
//! * Memory is a word array the caller supplies and reads back: `load` / `vload` read it at
//!   the address a scalar operand holds, `store` / `vstore` write it. An address outside the
//!   array is an error.
//! * Constants (`ConstChoice`, `Const`) take `produces_constant`, the value lowering emits.
//!   `add_imm` is its source plus the immediate. Every other operand must be a VReg: lowering
//!   puts an immediate where a scratch address goes, so `load(Imm k)` would read
//!   `mem[scratch[k]]`, not `mem[k]`, and is refused instead of guessed.
//! * Operand widths follow lowering. A vector op reads 8 consecutive words per operand, so a
//!   scalar operand of a vector op is an error (except the source of `vbroadcast`, and the
//!   address operands of `vload` / `vstore`).
//! * `SelectChoice` has two lowerings: `vselect` (`cond != 0 ? t : f`) and the multiply_add
//!   form `cond * delta + f` with `delta = delta_vreg` if set, else `t - f`. The scheduler may
//!   pick either, so both are computed and must agree (they do when `cond` is 0/1 and the
//!   delta is `t - f`); a disagreement is an error.
//! * Vector groups: a decomposed vector (the key has a definer) gives its lanes the key's
//!   lane values; reading a write-group key (no definer) assembles its lanes.
//! * The reserved zero VReg reads 0. Barriers, debug and control pseudo-ops do nothing.
//!
//! An opcode it does not know is an error, never a guess.

use std::collections::HashMap;

use crate::types::{Opcode, Operand, VReg, VRegWidth};

use super::super::SsaProgram;

const VLEN: usize = 8;

/// Lanes of a VReg.
fn width(v: VReg) -> usize {
    match v.width {
        VRegWidth::Scalar => 1,
        VRegWidth::Vector => VLEN,
    }
}

/// `problem.py` alu on 32-bit words.
fn alu(op: Opcode, a: u32, b: u32) -> Result<u32, String> {
    Ok(match op {
        Opcode::Add => a.wrapping_add(b),
        Opcode::Sub => a.wrapping_sub(b),
        Opcode::Mul => a.wrapping_mul(b),
        Opcode::Div => a.checked_div(b).ok_or("interp: division by zero")?,
        Opcode::Mod => a.checked_rem(b).ok_or("interp: remainder by zero")?,
        Opcode::Cdiv => {
            if b == 0 {
                return Err("interp: cdiv by zero".into());
            }
            ((a as u64 + b as u64 - 1) / b as u64) as u32
        }
        Opcode::Xor => a ^ b,
        Opcode::And => a & b,
        Opcode::Or => a | b,
        Opcode::Shl => a.checked_shl(b).unwrap_or(0),
        Opcode::Shr => a.checked_shr(b).unwrap_or(0),
        Opcode::Lt => (a < b) as u32,
        Opcode::Eq => (a == b) as u32,
        other => return Err(format!("interp: binary {:?} not supported", other)),
    })
}

/// Run `p` on `mem` (modified in place).
///
/// # Arguments
/// * `p` - The program (def before use).
/// * `mem` - Main memory; read by loads, written by stores.
///
/// # Returns
/// Every VReg's value, or an error for an unsupported or ill-formed op, a read of a value not
/// yet computed, an address outside memory, a division by zero, or SelectChoice encodings
/// that disagree.
pub fn run(p: &SsaProgram, mem: &mut [u32]) -> Result<HashMap<VReg, Vec<u32>>, String> {
    run_with(p, mem, SelectLowering::Both)
}

/// Which lowering of `SelectChoice` the interpreter follows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectLowering {
    /// Both, and they must agree (the program means one thing whatever the scheduler picks).
    Both,
    /// `vselect`: `cond != 0 ? t : f`.
    Vselect,
    /// The multiply_add form `cond * delta + f`.
    MultiplyAdd,
}

/// `run` following one `SelectChoice` lowering: for programs whose select conditions may be
/// other than 0/1 (perturbed inputs, or inputs outside the precondition), where the two
/// lowerings differ and a rewrite must keep each.
pub fn run_with(p: &SsaProgram, mem: &mut [u32], select: SelectLowering) -> Result<HashMap<VReg, Vec<u32>>, String> {
    let mut regs: HashMap<VReg, Vec<u32>> = HashMap::new();
    if let Some(z) = p.zero_vreg {
        regs.insert(z, vec![0]);
    }
    let read = |regs: &HashMap<VReg, Vec<u32>>, v: VReg| -> Result<Vec<u32>, String> {
        if let Some(x) = regs.get(&v) {
            return Ok(x.clone());
        }
        if let Some(lanes) = p.vector_group_lanes.get(&v) {
            // A group is a vector key and VLEN scalar lanes; anything else is malformed.
            if !v.is_vector() || lanes.len() != VLEN || lanes.iter().any(|l| l.is_vector()) {
                return Err(format!("interp: vector group {:?} is malformed ({} lanes)", v, lanes.len()));
            }
            return lanes
                .iter()
                .map(|l| regs.get(l).map(|x| x[0]).ok_or_else(|| format!("interp: lane {:?} of {:?} unset", l, v)))
                .collect();
        }
        Err(format!("interp: {:?} read before its definition", v))
    };
    for (i, op) in p.ops.iter().enumerate() {
        let vreg = |k: usize| -> Result<VReg, String> {
            match op.srcs.get(k) {
                Some(Operand::VReg(v)) => Ok(*v),
                Some(Operand::Imm(x)) => Err(format!(
                    "interp: op {} ({:?}) has an immediate operand {} where lowering expects a scratch address",
                    i, op.opcode, x
                )),
                None => Err(format!("interp: op {} ({:?}) lacks operand {}", i, op.opcode, k)),
            }
        };
        let src = |k: usize, regs: &HashMap<VReg, Vec<u32>>| -> Result<Vec<u32>, String> { read(regs, vreg(k)?) };
        let addr = |k: usize, regs: &HashMap<VReg, Vec<u32>>, n: usize| -> Result<usize, String> {
            let v = vreg(k)?;
            if v.is_vector() {
                return Err(format!("interp: op {} ({:?}) takes a vector as an address", i, op.opcode));
            }
            let a = read(regs, v)?[0] as usize;
            if a + n > mem.len() {
                return Err(format!("interp: op {} ({:?}) addresses {}..{}, outside memory", i, op.opcode, a, a + n));
            }
            Ok(a)
        };
        // A vector op's operands are vectors (lowering reads 8 consecutive words).
        let lanes = |k: usize, regs: &HashMap<VReg, Vec<u32>>, n: usize| -> Result<Vec<u32>, String> {
            let v = vreg(k)?;
            if width(v) != n {
                return Err(format!("interp: op {} ({:?}) mixes widths: operand {:?} for a {}-lane op", i, op.opcode, v, n));
            }
            read(regs, v)
        };
        let n = op.dest.map(width).unwrap_or(1);
        let out: Option<Vec<u32>> = match op.opcode {
            Opcode::Barrier | Opcode::DebugPrint => None,
            Opcode::Halt | Opcode::Jump | Opcode::CondJump => {
                return Err(format!("interp: op {} is control flow ({:?}); programs are straight-line", i, op.opcode))
            }
            Opcode::ConstChoice | Opcode::Const => {
                let k = match (op.produces_constant, op.srcs.first()) {
                    (Some(k), _) => k,
                    (None, Some(Operand::Imm(k))) => *k,
                    _ => return Err(format!("interp: op {} constant without its value", i)),
                };
                Some(vec![k as u32; n])
            }
            Opcode::AddImm => match op.srcs.get(1) {
                Some(Operand::Imm(k)) => Some(vec![src(0, &regs)?[0].wrapping_add(*k as u32)]),
                _ => return Err(format!("interp: op {} add_imm without an immediate", i)),
            },
            Opcode::Vbroadcast => {
                let s = src(0, &regs)?;
                if s.len() != 1 {
                    return Err(format!("interp: op {} broadcasts a vector", i));
                }
                Some(vec![s[0]; VLEN])
            }
            Opcode::Load => Some(vec![mem[addr(0, &regs, 1)?]]),
            Opcode::Vload => {
                let a = addr(0, &regs, VLEN)?;
                Some(mem[a..a + VLEN].to_vec())
            }
            Opcode::Store | Opcode::Vstore => {
                let w = if op.opcode == Opcode::Vstore { VLEN } else { 1 };
                let a = addr(0, &regs, w)?;
                let v = lanes(1, &regs, w)?;
                mem[a..a + w].copy_from_slice(&v);
                None
            }
            Opcode::MultiplyAdd => {
                let (a, b, c) = (lanes(0, &regs, n)?, lanes(1, &regs, n)?, lanes(2, &regs, n)?);
                Some((0..n).map(|k| a[k].wrapping_mul(b[k]).wrapping_add(c[k])).collect())
            }
            Opcode::Select | Opcode::Vselect => {
                let (c, t, f) = (lanes(0, &regs, n)?, lanes(1, &regs, n)?, lanes(2, &regs, n)?);
                Some((0..n).map(|k| if c[k] != 0 { t[k] } else { f[k] }).collect())
            }
            Opcode::SelectChoice => {
                let (c, t, f) = (lanes(0, &regs, n)?, lanes(1, &regs, n)?, lanes(2, &regs, n)?);
                let vselect: Vec<u32> = (0..n).map(|k| if c[k] != 0 { t[k] } else { f[k] }).collect();
                if select == SelectLowering::Vselect {
                    // The vselect lowering does not read the delta.
                    Some(vselect)
                } else {
                let delta = match op.delta_vreg {
                    Some(d) => {
                        if width(d) != n {
                            return Err(format!("interp: op {} select delta has the wrong width", i));
                        }
                        read(&regs, d)?
                    }
                    None => (0..n).map(|k| t[k].wrapping_sub(f[k])).collect(),
                };
                let fma: Vec<u32> = (0..n).map(|k| c[k].wrapping_mul(delta[k]).wrapping_add(f[k])).collect();
                match select {
                    SelectLowering::Vselect => Some(vselect),
                    SelectLowering::MultiplyAdd => Some(fma),
                    SelectLowering::Both if vselect == fma => Some(vselect),
                    SelectLowering::Both => {
                        return Err(format!(
                            "interp: op {} SelectChoice lowerings disagree (vselect {:?}, multiply_add {:?})",
                            i, vselect, fma
                        ))
                    }
                }
                }
            }
            opc if op.srcs.len() == 2 => {
                let (a, b) = (lanes(0, &regs, n)?, lanes(1, &regs, n)?);
                Some(a.iter().zip(&b).map(|(&x, &y)| alu(opc, x, y)).collect::<Result<_, _>>()?)
            }
            other => return Err(format!("interp: op {} ({:?}) not supported", i, other)),
        };
        if let (Some(d), Some(v)) = (op.dest, out) {
            if v.len() != width(d) {
                return Err(format!("interp: op {} writes {} lanes to {:?}", i, v.len(), d));
            }
            if let Some(ls) = p.vector_group_lanes.get(&d) {
                if !d.is_vector() || ls.len() != VLEN || ls.iter().any(|l| l.is_vector()) {
                    return Err(format!("interp: vector group {:?} is malformed ({} lanes)", d, ls.len()));
                }
                for (k, l) in ls.iter().enumerate() {
                    regs.insert(*l, vec![v[k]]);
                }
            }
            regs.insert(d, v);
        }
    }
    Ok(regs)
}
