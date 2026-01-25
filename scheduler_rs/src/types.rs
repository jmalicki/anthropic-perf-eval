//! Core types and constants for the scheduler.

use arrayvec::ArrayVec;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Maximum number of elements in an operation's slot (opcode + operands)
/// Most ops have 4 (opcode, dest, src1, src2), multiply_add has 5
pub const MAX_SLOT_SIZE: usize = 6;

/// Vector length for SIMD operations
pub const VLEN: usize = 8;

// ============================================================================
// Engine Enum
// ============================================================================

/// Execution engine type - uses u8 backing for minimal size
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum Engine {
    Load = 0,
    Store = 1,
    Alu = 2,
    Valu = 3,
    Flow = 4,
    Debug = 5,
    // Scheduling-internal states (not real hardware engines)
    EliminatedBroadcast = 6,
    ShiftedValu = 7,
}

impl Engine {
    /// Parse engine from string (used at Python boundary)
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "load" => Some(Engine::Load),
            "store" => Some(Engine::Store),
            "alu" => Some(Engine::Alu),
            "valu" => Some(Engine::Valu),
            "flow" => Some(Engine::Flow),
            "debug" => Some(Engine::Debug),
            "eliminated_broadcast" => Some(Engine::EliminatedBroadcast),
            "shifted_valu" => Some(Engine::ShiftedValu),
            _ => None,
        }
    }

    /// Convert engine to string (used at Python boundary)
    pub fn as_str(&self) -> &'static str {
        match self {
            Engine::Load => "load",
            Engine::Store => "store",
            Engine::Alu => "alu",
            Engine::Valu => "valu",
            Engine::Flow => "flow",
            Engine::Debug => "debug",
            Engine::EliminatedBroadcast => "eliminated_broadcast",
            Engine::ShiftedValu => "shifted_valu",
        }
    }

    /// Get the slot limit for this engine
    pub fn slot_limit(&self) -> usize {
        match self {
            Engine::Load => 2,
            Engine::Store => 2,
            Engine::Alu => 12,
            Engine::Valu => 6,
            Engine::Flow => 1,
            Engine::Debug => 64,
            // Internal states have no slots in bundles
            Engine::EliminatedBroadcast => 0,
            Engine::ShiftedValu => 0,
        }
    }

    /// Check if this is a real hardware engine (vs scheduling-internal)
    pub fn is_hardware(&self) -> bool {
        !matches!(self, Engine::EliminatedBroadcast | Engine::ShiftedValu)
    }
}

// ============================================================================
// Opcode Enum
// ============================================================================

/// Operation opcode - uses u8 backing for minimal size
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum Opcode {
    // Arithmetic operations
    Add = 0,       // "+"
    Sub = 1,       // "-"
    Mul = 2,       // "*"
    Div = 3,       // "/"
    Mod = 4,       // "%"
    Cdiv = 5,      // "cdiv" (ceiling division)

    // Bitwise operations
    And = 6,       // "&"
    Or = 7,        // "|"
    Xor = 8,       // "^"
    Shl = 9,       // "<<"
    Shr = 10,      // ">>"

    // Comparison operations
    Eq = 11,       // "=="
    Ne = 12,       // "!="
    Lt = 13,       // "<"
    Le = 14,       // "<="
    Gt = 15,       // ">"
    Ge = 16,       // ">="

    // Vector-specific operations
    Vbroadcast = 17,
    MultiplyAdd = 18,  // "multiply_add"

    // Load operations
    Load = 19,
    Vload = 20,
    Const = 21,
    LoadOffset = 22,

    // Store operations
    Store = 23,
    Vstore = 24,

    // Flow control
    Select = 25,
    Vselect = 26,
    Jump = 27,
    CondJump = 28,
    Halt = 29,

    // Debug
    DebugPrint = 30,
}

impl Opcode {
    /// Parse opcode from string (used at Python boundary)
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "+" => Some(Opcode::Add),
            "-" => Some(Opcode::Sub),
            "*" => Some(Opcode::Mul),
            "/" => Some(Opcode::Div),
            "%" => Some(Opcode::Mod),
            "cdiv" => Some(Opcode::Cdiv),
            "&" => Some(Opcode::And),
            "|" => Some(Opcode::Or),
            "^" => Some(Opcode::Xor),
            "<<" => Some(Opcode::Shl),
            ">>" => Some(Opcode::Shr),
            "==" => Some(Opcode::Eq),
            "!=" => Some(Opcode::Ne),
            "<" => Some(Opcode::Lt),
            "<=" => Some(Opcode::Le),
            ">" => Some(Opcode::Gt),
            ">=" => Some(Opcode::Ge),
            "vbroadcast" => Some(Opcode::Vbroadcast),
            "multiply_add" => Some(Opcode::MultiplyAdd),
            "load" => Some(Opcode::Load),
            "vload" => Some(Opcode::Vload),
            "const" => Some(Opcode::Const),
            "load_offset" => Some(Opcode::LoadOffset),
            "store" => Some(Opcode::Store),
            "vstore" => Some(Opcode::Vstore),
            "select" => Some(Opcode::Select),
            "vselect" => Some(Opcode::Vselect),
            "jump" => Some(Opcode::Jump),
            "cond_jump" => Some(Opcode::CondJump),
            "halt" => Some(Opcode::Halt),
            "debug_print" => Some(Opcode::DebugPrint),
            _ => None,
        }
    }

    /// Convert opcode to string (used at Python boundary)
    pub fn as_str(&self) -> &'static str {
        match self {
            Opcode::Add => "+",
            Opcode::Sub => "-",
            Opcode::Mul => "*",
            Opcode::Div => "/",
            Opcode::Mod => "%",
            Opcode::Cdiv => "cdiv",
            Opcode::And => "&",
            Opcode::Or => "|",
            Opcode::Xor => "^",
            Opcode::Shl => "<<",
            Opcode::Shr => ">>",
            Opcode::Eq => "==",
            Opcode::Ne => "!=",
            Opcode::Lt => "<",
            Opcode::Le => "<=",
            Opcode::Gt => ">",
            Opcode::Ge => ">=",
            Opcode::Vbroadcast => "vbroadcast",
            Opcode::MultiplyAdd => "multiply_add",
            Opcode::Load => "load",
            Opcode::Vload => "vload",
            Opcode::Const => "const",
            Opcode::LoadOffset => "load_offset",
            Opcode::Store => "store",
            Opcode::Vstore => "vstore",
            Opcode::Select => "select",
            Opcode::Vselect => "vselect",
            Opcode::Jump => "jump",
            Opcode::CondJump => "cond_jump",
            Opcode::Halt => "halt",
            Opcode::DebugPrint => "debug_print",
        }
    }

    /// Check if this VALU opcode can be converted to 8 scalar ALU ops
    /// Note: multiply_add is excluded because it's a fused operation
    pub fn can_shift_to_alu(&self) -> bool {
        matches!(
            self,
            Opcode::Add
                | Opcode::Sub
                | Opcode::Mul
                | Opcode::Div
                | Opcode::Mod
                | Opcode::And
                | Opcode::Or
                | Opcode::Xor
                | Opcode::Shl
                | Opcode::Shr
                | Opcode::Eq
                | Opcode::Ne
                | Opcode::Lt
                | Opcode::Le
                | Opcode::Gt
                | Opcode::Ge
        )
    }

    /// Check if this is an arithmetic/bitwise/comparison op (binary operation)
    pub fn is_binary_op(&self) -> bool {
        (*self as u8) <= (Opcode::Ge as u8)
    }
}

// ============================================================================
// Legacy compatibility (slot limits by string)
// ============================================================================

/// Slot limits per engine per cycle
pub const SLOT_LIMITS: [(&str, usize); 6] = [
    ("load", 2),
    ("store", 2),
    ("alu", 12),
    ("valu", 6),
    ("flow", 1),
    ("debug", 64),
];

/// Get the slot limit for an engine (legacy string-based version)
pub fn get_slot_limit(engine: &str) -> usize {
    Engine::from_str(engine).map(|e| e.slot_limit()).unwrap_or(1)
}

/// VALU ops that can be converted to scalar ALU ops (binary operations)
/// NOTE: 'multiply_add' is intentionally EXCLUDED because it's a fused op
pub const VALU_TO_ALU_OPS: &[&str] = &[
    "+", "-", "*", "/", "%", "&", "|", "^", "<<", ">>", "==", "!=", "<", "<=", ">", ">=",
];

// ============================================================================
// SlotVal and Op
// ============================================================================

/// Slot value - can be opcode or int
#[derive(Clone, Debug)]
pub enum SlotVal {
    Str(String),  // Legacy: will be replaced with Op(Opcode)
    Int(i64),
}

/// Operation with its metadata
#[derive(Clone, Debug)]
pub struct Op {
    pub id: usize,
    pub engine: Engine,
    /// Slot contents: [opcode, dest, src1, src2, ...] - inline array, no heap allocation
    pub slot: ArrayVec<SlotVal, MAX_SLOT_SIZE>,
    pub cycle: i32,
    pub dep_ids: HashSet<usize>,
    pub reads: HashSet<i64>,
    pub writes: HashSet<i64>,
}

/// Check if a VALU op can be converted to 8 scalar ALU ops
pub fn can_convert_valu_to_alu(op: &Op) -> bool {
    if op.engine != Engine::Valu {
        return false;
    }
    if op.slot.is_empty() {
        return false;
    }
    if let SlotVal::Str(op_type) = &op.slot[0] {
        // Check via Opcode enum for better performance
        Opcode::from_str(op_type)
            .map(|opc| opc.can_shift_to_alu())
            .unwrap_or(false)
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
            engine: Engine::Alu,
            slot: [
                SlotVal::Str(op_type.clone()),
                SlotVal::Int(dest + i_offset),
                SlotVal::Int(actual_src1),
                SlotVal::Int(actual_src2),
            ].into_iter().collect(),
            cycle: -1,
            dep_ids: op.dep_ids.clone(),
            reads: [actual_src1, actual_src2].into_iter().collect(),
            writes: [dest + i_offset].into_iter().collect(),
        };
        new_ops.push(new_op);
    }
    (new_ops, forward_count)
}
