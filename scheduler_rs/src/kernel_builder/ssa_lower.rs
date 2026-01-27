//! SSA lowering helpers.
//!
//! This module handles lowering SSA ops to concrete ops, including
//! VALU→ALU shift expansion.

use std::collections::{HashMap, HashSet};

use arrayvec::ArrayVec;

use crate::address_assign::lower_ssa_op;
use crate::types::{Engine, Op, SlotVal, SsaOp, VReg, VLEN};

/// Check if an SSA VALU op can be shifted to ALU.
///
/// Shiftable ops are binary arithmetic operations that can be expanded
/// to 8 scalar ALU ops (one per vector lane).
pub fn can_ssa_valu_shift_to_alu(op: &SsaOp) -> bool {
    if op.engine != Engine::Valu {
        return false;
    }

    // Check if the opcode supports shifting
    op.opcode.can_shift_to_alu()
}

/// Lower SSA ops to concrete ops (no VALU shifting).
#[allow(dead_code)]
pub fn lower_ssa_ops(ssa_ops: &[SsaOp], vreg_to_addr: &HashMap<VReg, i64>) -> Vec<Op> {
    ssa_ops
        .iter()
        .map(|op| lower_ssa_op(op, vreg_to_addr))
        .collect()
}

/// Lower SSA ops to concrete ops, expanding shifted VALU ops to ALU.
///
/// Shifted VALU ops are expanded to 8 scalar ALU ops, one per lane.
/// Each ALU op reads from src[i] and writes to dest[i].
pub fn lower_ssa_ops_with_shift(
    ssa_ops: &[SsaOp],
    vreg_to_addr: &HashMap<VReg, i64>,
    shifted_ops: &HashSet<usize>,
) -> Vec<Op> {
    let mut result = Vec::new();

    for ssa_op in ssa_ops {
        if shifted_ops.contains(&ssa_op.id) {
            // Expand shifted VALU op to 8 ALU ops
            let expanded = expand_ssa_valu_to_alu(ssa_op, vreg_to_addr);
            result.extend(expanded);
        } else {
            // Normal lowering
            result.push(lower_ssa_op(ssa_op, vreg_to_addr));
        }
    }

    result
}

/// Expand a shifted SSA VALU op into 8 scalar ALU ops.
pub fn expand_ssa_valu_to_alu(ssa_op: &SsaOp, vreg_to_addr: &HashMap<VReg, i64>) -> Vec<Op> {
    use crate::types::Operand;

    let dest = ssa_op.dest.expect("VALU op must have dest");
    let dest_addr = *vreg_to_addr.get(&dest).expect("dest VReg not in mapping");

    // Get source addresses
    let (src1_addr, src2_addr) = match (&ssa_op.srcs[..], ssa_op.opcode) {
        ([Operand::VReg(s1), Operand::VReg(s2)], _) => {
            let a1 = *vreg_to_addr.get(s1).expect("src1 VReg not in mapping");
            let a2 = *vreg_to_addr.get(s2).expect("src2 VReg not in mapping");
            (a1, a2)
        }
        _ => return vec![], // Can't expand non-binary ops
    };

    // Create 8 ALU ops
    let mut ops = Vec::with_capacity(VLEN);
    for i in 0..VLEN {
        let i_offset = i as i64;

        let mut slot: ArrayVec<SlotVal, 6> = ArrayVec::new();
        slot.push(SlotVal::Str(ssa_op.opcode.as_str().to_string()));
        slot.push(SlotVal::Int(dest_addr + i_offset));
        slot.push(SlotVal::Int(src1_addr + i_offset));
        slot.push(SlotVal::Int(src2_addr + i_offset));

        // Build reads/writes sets
        let mut reads = HashSet::new();
        reads.insert(src1_addr + i_offset);
        reads.insert(src2_addr + i_offset);

        let mut writes = HashSet::new();
        writes.insert(dest_addr + i_offset);

        ops.push(Op {
            id: ssa_op.id * 1000 + i, // Unique ID for expanded op
            engine: Engine::Alu,
            slot,
            cycle: ssa_op.cycle,
            dep_ids: ssa_op.dep_ids.clone(),
            reads,
            writes,
            last_use_reads: HashSet::new(),
            produces_constant: None,
        });
    }

    ops
}
