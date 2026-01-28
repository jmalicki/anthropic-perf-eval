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
///
/// ConstChoice ops are lowered based on the scheduler's decision:
/// - If in `const_add_imm_ops`: emit AddImm op (Flow engine)
/// - Otherwise: emit Const op (Load engine)
pub fn lower_ssa_ops_with_shift(
    ssa_ops: &[SsaOp],
    vreg_to_addr: &HashMap<VReg, i64>,
    shifted_ops: &HashSet<usize>,
    const_add_imm_ops: &HashSet<usize>,
) -> Vec<Op> {
    use crate::types::{Opcode, SlotVal};
    use arrayvec::ArrayVec;

    let mut result = Vec::new();

    for ssa_op in ssa_ops {
        if shifted_ops.contains(&ssa_op.id) {
            // Expand shifted VALU op to 8 ALU ops
            let expanded = expand_ssa_valu_to_alu(ssa_op, vreg_to_addr);
            result.extend(expanded);
        } else if ssa_op.opcode == Opcode::ConstChoice {
            // Handle ConstChoice based on scheduler decision
            let dest = ssa_op.dest.expect("ConstChoice op must have dest");
            let dest_addr = *vreg_to_addr.get(&dest).expect("dest VReg not in mapping");
            let value = ssa_op.produces_constant.expect("ConstChoice op must have produces_constant");

            if const_add_imm_ops.contains(&ssa_op.id) {
                // AddImm: dest = scratch[0] + value
                // IMPORTANT: Scratch memory is ALWAYS initialized to 0 at startup, so scratch[0] == 0
                // is guaranteed by hardware initialization. add_imm reads from address 0, but we don't
                // need a precedence constraint because address 0 is always initialized to 0.
                let mut slot: ArrayVec<SlotVal, 6> = ArrayVec::new();
                slot.push(SlotVal::Str("add_imm".to_string()));
                slot.push(SlotVal::Int(dest_addr));
                slot.push(SlotVal::Int(0)); // Address 0 always contains 0 (hardware initialization)
                slot.push(SlotVal::Int(value));

                let mut reads = HashSet::new();
                reads.insert(0); // Read from zero VReg

                let mut writes = HashSet::new();
                writes.insert(dest_addr);

                result.push(Op {
                    id: ssa_op.id,
                    engine: crate::types::Engine::Flow,
                    slot,
                    cycle: ssa_op.cycle,
                    dep_ids: ssa_op.dep_ids.clone(),
                    reads,
                    writes,
                    last_use_reads: HashSet::new(),
                    produces_constant: Some(value),
                });
            } else {
                // Load: dest = const value
                let mut slot: ArrayVec<SlotVal, 6> = ArrayVec::new();
                slot.push(SlotVal::Str("const".to_string()));
                slot.push(SlotVal::Int(dest_addr));
                slot.push(SlotVal::Int(value));

                let mut writes = HashSet::new();
                writes.insert(dest_addr);

                result.push(Op {
                    id: ssa_op.id,
                    engine: crate::types::Engine::Load,
                    slot,
                    cycle: ssa_op.cycle,
                    dep_ids: ssa_op.dep_ids.clone(),
                    reads: HashSet::new(),
                    writes,
                    last_use_reads: HashSet::new(),
                    produces_constant: Some(value),
                });
            }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Operand, Opcode};

    fn make_const_choice_op(id: usize, value: i64, dest: VReg, cycle: i32) -> SsaOp {
        let mut op = SsaOp::new(
            id,
            Engine::Load,
            Opcode::ConstChoice,
            Some(dest),
            vec![Operand::Imm(value)],
        );
        op.produces_constant = Some(value);
        op.const_alternatives = Some(vec![
            crate::types::ConstEncoding::Load,
            crate::types::ConstEncoding::AddImm { value },
        ]);
        op.cycle = cycle;
        op
    }

    #[test]
    fn test_lower_constchoice_load() {
        let mut vreg_to_addr = HashMap::new();
        let dest_vreg = VReg::scalar(1);
        vreg_to_addr.insert(dest_vreg, 100);

        let ssa_op = make_const_choice_op(42, 5, dest_vreg, 10);

        let ssa_ops = vec![ssa_op];
        let shifted_ops = HashSet::new();
        let const_add_imm_ops = HashSet::new(); // Use Load

        let result = lower_ssa_ops_with_shift(&ssa_ops, &vreg_to_addr, &shifted_ops, &const_add_imm_ops);

        assert_eq!(result.len(), 1);
        let op = &result[0];
        assert_eq!(op.engine, Engine::Load);
        assert_eq!(op.id, 42);
        assert_eq!(op.cycle, 10);
        assert_eq!(op.slot[0], SlotVal::Str("const".to_string()));
        assert_eq!(op.slot[1], SlotVal::Int(100)); // dest addr
        assert_eq!(op.slot[2], SlotVal::Int(5)); // value
        assert_eq!(op.produces_constant, Some(5));
        assert!(op.reads.is_empty()); // Const doesn't read anything
        assert!(op.writes.contains(&100));
    }

    #[test]
    fn test_lower_constchoice_addimm() {
        let mut vreg_to_addr = HashMap::new();
        let dest_vreg = VReg::scalar(1);
        vreg_to_addr.insert(dest_vreg, 200);

        let ssa_op = make_const_choice_op(99, 256, dest_vreg, 5);

        let ssa_ops = vec![ssa_op];
        let shifted_ops = HashSet::new();
        let mut const_add_imm_ops = HashSet::new();
        const_add_imm_ops.insert(99); // Use AddImm

        let result = lower_ssa_ops_with_shift(&ssa_ops, &vreg_to_addr, &shifted_ops, &const_add_imm_ops);

        assert_eq!(result.len(), 1);
        let op = &result[0];
        assert_eq!(op.engine, Engine::Flow);
        assert_eq!(op.id, 99);
        assert_eq!(op.cycle, 5);
        assert_eq!(op.slot[0], SlotVal::Str("add_imm".to_string()));
        assert_eq!(op.slot[1], SlotVal::Int(200)); // dest addr
        assert_eq!(op.slot[2], SlotVal::Int(0)); // zero VReg addr
        assert_eq!(op.slot[3], SlotVal::Int(256)); // value
        assert_eq!(op.produces_constant, Some(256));
        assert!(op.reads.contains(&0)); // Reads from zero VReg
        assert!(op.writes.contains(&200));
    }

    #[test]
    fn test_lower_constchoice_mixed() {
        let mut vreg_to_addr = HashMap::new();
        let v1 = VReg::scalar(1);
        let v2 = VReg::scalar(2);
        vreg_to_addr.insert(v1, 100);
        vreg_to_addr.insert(v2, 200);

        let op1 = make_const_choice_op(1, 10, v1, 0);
        let op2 = make_const_choice_op(2, 20, v2, 0);

        let ssa_ops = vec![op1, op2];
        let shifted_ops = HashSet::new();
        let mut const_add_imm_ops = HashSet::new();
        const_add_imm_ops.insert(1); // op1 uses AddImm, op2 uses Load

        let result = lower_ssa_ops_with_shift(&ssa_ops, &vreg_to_addr, &shifted_ops, &const_add_imm_ops);

        assert_eq!(result.len(), 2);
        
        // First op: AddImm
        assert_eq!(result[0].engine, Engine::Flow);
        assert_eq!(result[0].slot[0], SlotVal::Str("add_imm".to_string()));
        
        // Second op: Load
        assert_eq!(result[1].engine, Engine::Load);
        assert_eq!(result[1].slot[0], SlotVal::Str("const".to_string()));
    }
}
