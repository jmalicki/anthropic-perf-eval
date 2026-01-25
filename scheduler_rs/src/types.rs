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
    use std::collections::HashMap;
    expand_valu_to_alu_smart(op, next_op_id, &HashMap::new(), &HashMap::new()).0
}

/// Broadcast source info for smart expansion
#[derive(Clone, Debug, Default)]
pub struct BroadcastInfo {
    /// Maps broadcast dest base address -> original scalar source address
    pub sources: std::collections::HashMap<i64, i64>,
    /// Maps broadcast dest base address -> op ID of the broadcast
    pub op_ids: std::collections::HashMap<i64, usize>,
}

/// Expand a VALU op into 8 scalar ALU ops with broadcast forwarding.
///
/// When a VALU op reads from a vbroadcast destination, instead of reading
/// from the broadcast registers (which would be 8 identical values), we
/// substitute the original scalar source directly. This eliminates 7
/// redundant reads per forwarded operand.
///
/// Returns (expanded_ops, forward_count) where forward_count is the number
/// of broadcast forwards applied (0, 1, or 2).
pub fn expand_valu_to_alu_smart(
    op: &Op,
    next_op_id: usize,
    broadcast_sources: &std::collections::HashMap<i64, i64>,
    broadcast_op_ids: &std::collections::HashMap<i64, usize>,
) -> (Vec<Op>, usize) {
    // slot format: (op_type, dest, src1, src2)
    let op_type = match &op.slot[0] {
        SlotVal::Str(s) => s.clone(),
        _ => return (vec![], 0),
    };
    let dest = match &op.slot[1] {
        SlotVal::Int(n) => *n,
        _ => return (vec![], 0),
    };
    let src1 = match &op.slot[2] {
        SlotVal::Int(n) => *n,
        _ => return (vec![], 0),
    };
    let src2 = match &op.slot[3] {
        SlotVal::Int(n) => *n,
        _ => return (vec![], 0),
    };

    // Check if src1 or src2 came from a broadcast AND we depend on that broadcast
    let src1_is_broadcast = broadcast_sources.contains_key(&src1)
        && broadcast_op_ids
            .get(&src1)
            .map(|&bc_id| op.dep_ids.contains(&bc_id))
            .unwrap_or(false);
    let src2_is_broadcast = broadcast_sources.contains_key(&src2)
        && broadcast_op_ids
            .get(&src2)
            .map(|&bc_id| op.dep_ids.contains(&bc_id))
            .unwrap_or(false);

    let forward_count =
        (if src1_is_broadcast { 1 } else { 0 }) + (if src2_is_broadcast { 1 } else { 0 });

    let mut new_ops = Vec::with_capacity(VLEN);
    for i in 0..VLEN {
        let i_offset = i as i64;

        // If source came from broadcast AND we depend on it, use scalar directly
        let actual_src1 = if src1_is_broadcast {
            *broadcast_sources.get(&src1).unwrap()
        } else {
            src1 + i_offset
        };

        let actual_src2 = if src2_is_broadcast {
            *broadcast_sources.get(&src2).unwrap()
        } else {
            src2 + i_offset
        };

        let new_op = Op {
            id: next_op_id + i,
            engine: "alu".to_string(),
            slot: vec![
                SlotVal::Str(op_type.clone()),
                SlotVal::Int(dest + i_offset),
                SlotVal::Int(actual_src1),
                SlotVal::Int(actual_src2),
            ],
            cycle: -1,
            dep_ids: op.dep_ids.clone(),
            reads: [actual_src1, actual_src2].into_iter().collect(),
            writes: [dest + i_offset].into_iter().collect(),
        };
        new_ops.push(new_op);
    }
    (new_ops, forward_count)
}
