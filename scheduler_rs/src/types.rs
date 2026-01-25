//! Core types and constants for the scheduler.

use std::collections::HashSet;

/// Vector length for SIMD operations
pub const VLEN: usize = 8;

/// Slot limits per engine per cycle
pub const SLOT_LIMITS: [(&str, usize); 6] = [
    ("load", 2),
    ("store", 2),
    ("alu", 12),
    ("valu", 6),
    ("flow", 1),
    ("debug", 64),
];

/// Get the slot limit for an engine
pub fn get_slot_limit(engine: &str) -> usize {
    for (e, limit) in SLOT_LIMITS.iter() {
        if *e == engine {
            return *limit;
        }
    }
    1
}

/// VALU ops that can be converted to scalar ALU ops (binary operations)
/// NOTE: 'multiply_add' is intentionally EXCLUDED because it's a fused op
pub const VALU_TO_ALU_OPS: &[&str] = &[
    "+", "-", "*", "/", "%", "&", "|", "^", "<<", ">>", "==", "!=", "<", "<=", ">", ">=",
];

/// Slot value - can be string or int
#[derive(Clone, Debug)]
pub enum SlotVal {
    Str(String),
    Int(i64),
}

/// Operation with its metadata
#[derive(Clone, Debug)]
pub struct Op {
    pub id: usize,
    pub engine: String,
    pub slot: Vec<SlotVal>,
    pub cycle: i32,
    pub dep_ids: HashSet<usize>,
    pub reads: HashSet<i64>,
    pub writes: HashSet<i64>,
}

/// Check if a VALU op can be converted to 8 scalar ALU ops
pub fn can_convert_valu_to_alu(op: &Op) -> bool {
    if op.engine != "valu" {
        return false;
    }
    if op.slot.is_empty() {
        return false;
    }
    if let SlotVal::Str(op_type) = &op.slot[0] {
        VALU_TO_ALU_OPS.contains(&op_type.as_str())
    } else {
        false
    }
}

/// Expand a VALU op into 8 scalar ALU ops
pub fn expand_valu_to_alu(op: &Op, next_op_id: usize) -> Vec<Op> {
    // slot format: (op_type, dest, src1, src2)
    let op_type = match &op.slot[0] {
        SlotVal::Str(s) => s.clone(),
        _ => return vec![],
    };
    let dest = match &op.slot[1] {
        SlotVal::Int(n) => *n,
        _ => return vec![],
    };
    let src1 = match &op.slot[2] {
        SlotVal::Int(n) => *n,
        _ => return vec![],
    };
    let src2 = match &op.slot[3] {
        SlotVal::Int(n) => *n,
        _ => return vec![],
    };

    let mut new_ops = Vec::with_capacity(VLEN);
    for i in 0..VLEN {
        let i = i as i64;
        let new_op = Op {
            id: next_op_id + i as usize,
            engine: "alu".to_string(),
            slot: vec![
                SlotVal::Str(op_type.clone()),
                SlotVal::Int(dest + i),
                SlotVal::Int(src1 + i),
                SlotVal::Int(src2 + i),
            ],
            cycle: -1,
            dep_ids: op.dep_ids.clone(),
            reads: [src1 + i, src2 + i].into_iter().collect(),
            writes: [dest + i].into_iter().collect(),
        };
        new_ops.push(new_op);
    }
    new_ops
}
