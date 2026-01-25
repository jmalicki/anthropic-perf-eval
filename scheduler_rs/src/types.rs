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

    // Immediate operations (flow engine)
    AddImm = 31,
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
            "add_imm" => Some(Opcode::AddImm),
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
            Opcode::AddImm => "add_imm",
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
    /// Which addresses in `reads` are their LAST use across the entire program.
    /// This is computed by `compute_last_uses()` in the scheduler before scheduling.
    /// Used for scratch pressure tracking - when this op executes, these addresses
    /// can be freed from scratch since no later op will read them.
    pub last_use_reads: HashSet<i64>,
    /// If this op produces a known constant value, record it here.
    /// This enables late-binding optimization - any pass can replace
    /// this op with a cheaper method of producing the same constant.
    /// For example, a Load("const", dest, 4) might be replaced with
    /// ALU("+", dest, 2, 2) if ALU slots are available but Load slots are scarce.
    pub produces_constant: Option<i64>,
}

// ============================================================================
// Pseudo-Op Constants
// ============================================================================

/// Possible ways to materialize a constant value.
///
/// The scheduler can choose between these encodings dynamically based on
/// resource availability (load slots, ALU slots, flow slots) and which
/// operand values are already available.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConstEncoding {
    /// Load directly: `const dest value`
    /// Cost: 1 load slot, no dependencies
    Load,

    /// Flow add_imm: `add_imm dest s0 value`
    /// Cost: 1 flow slot, needs ZERO_REG available
    AddImm { value: i64 },

    /// ALU addition: `add dest srcA srcB`
    /// Cost: 1 ALU slot, needs both operands ready
    Add { a_value: i64, b_value: i64 },

    /// ALU subtraction: `sub dest srcA srcB`
    /// Cost: 1 ALU slot, needs both operands ready
    Sub { a_value: i64, b_value: i64 },

    /// ALU multiplication: `mul dest srcA srcB`
    /// Cost: 1 ALU slot, needs both operands ready
    Mul { a_value: i64, b_value: i64 },

    /// ALU left shift: `shl dest src amount`
    /// Cost: 1 ALU slot, needs base value ready + shift amount as constant
    Shift { base_value: i64, shift_amount: u8 },
}

impl ConstEncoding {
    /// Get the engine this encoding uses
    pub fn engine(&self) -> Engine {
        match self {
            ConstEncoding::Load => Engine::Load,
            ConstEncoding::AddImm { .. } => Engine::Flow,
            _ => Engine::Alu,
        }
    }

    /// Get the values this encoding depends on (must be available before scheduling)
    pub fn dependencies(&self) -> Vec<i64> {
        match self {
            ConstEncoding::Load => vec![],
            ConstEncoding::AddImm { .. } => vec![0], // Depends on ZERO_REG
            ConstEncoding::Add { a_value, b_value } => vec![*a_value, *b_value],
            ConstEncoding::Sub { a_value, b_value } => vec![*a_value, *b_value],
            ConstEncoding::Mul { a_value, b_value } => vec![*a_value, *b_value],
            ConstEncoding::Shift { base_value, shift_amount } => {
                // Shift amount must also be available as a constant
                vec![*base_value, *shift_amount as i64]
            }
        }
    }

    /// Check if this encoding's dependencies are satisfied
    pub fn is_ready(&self, available_values: &HashSet<i64>) -> bool {
        match self {
            ConstEncoding::Load => true, // No dependencies
            ConstEncoding::AddImm { .. } => available_values.contains(&0), // ZERO_REG must be available
            ConstEncoding::Add { a_value, b_value }
            | ConstEncoding::Sub { a_value, b_value }
            | ConstEncoding::Mul { a_value, b_value } => {
                available_values.contains(a_value) && available_values.contains(b_value)
            }
            ConstEncoding::Shift { base_value, shift_amount } => {
                available_values.contains(base_value)
                    && available_values.contains(&(*shift_amount as i64))
            }
        }
    }
}

/// A pseudo-operation representing "need constant value V available in a register".
///
/// Unlike concrete ops, a PseudoConst doesn't commit to how the constant is generated.
/// The scheduler resolves it to a concrete encoding (Load, Add, etc.) based on:
/// - Available resources (load slots, ALU slots)
/// - Which operand values are already available
/// - Critical path considerations (urgent consumers = higher priority)
#[derive(Clone, Debug)]
pub struct PseudoConst {
    /// Unique identifier (separate namespace from Op ids)
    pub id: usize,

    /// The constant value to materialize
    pub value: i64,

    /// Destination register (assigned when resolved, or pre-assigned)
    pub dest_register: Option<i64>,

    /// Possible encodings, sorted by preference (index 0 = highest priority)
    pub encodings: Vec<ConstEncoding>,

    /// Operations that depend on this constant being ready
    pub consumer_op_ids: Vec<usize>,

    /// The encoding actually chosen (filled when scheduled)
    pub resolved_encoding: Option<ConstEncoding>,

    /// Cycle when this was scheduled (-1 if not yet scheduled)
    pub scheduled_cycle: i32,

    /// Whether this constant is on the critical path
    pub is_critical: bool,
}

impl PseudoConst {
    /// Create a new pseudo-const with computed encodings
    pub fn new(id: usize, value: i64, dest_register: Option<i64>) -> Self {
        Self::new_with_flags(id, value, dest_register, ConstEncodingFlags::all())
    }

    /// Create a new pseudo-const with configurable encoding flags
    pub fn new_with_flags(id: usize, value: i64, dest_register: Option<i64>, flags: ConstEncodingFlags) -> Self {
        let encodings = compute_const_encodings_with_flags(value, flags);
        PseudoConst {
            id,
            value,
            dest_register,
            encodings,
            consumer_op_ids: Vec::new(),
            resolved_encoding: None,
            scheduled_cycle: -1,
            is_critical: false,
        }
    }

    /// Create a pseudo-const marked as critical (will prefer Load encoding)
    pub fn new_critical(id: usize, value: i64, dest_register: Option<i64>) -> Self {
        Self::new_critical_with_flags(id, value, dest_register, ConstEncodingFlags::all())
    }

    /// Create a critical pseudo-const with configurable encoding flags
    pub fn new_critical_with_flags(id: usize, value: i64, dest_register: Option<i64>, flags: ConstEncodingFlags) -> Self {
        let mut pc = Self::new_with_flags(id, value, dest_register, flags);
        pc.is_critical = true;
        // For critical constants, put Load first if not already (and if enabled)
        if flags.enable_load && !pc.encodings.is_empty() && pc.encodings[0] != ConstEncoding::Load {
            pc.encodings.retain(|e| *e != ConstEncoding::Load);
            pc.encodings.insert(0, ConstEncoding::Load);
        }
        pc
    }

    /// Check if this pseudo-const has been resolved
    pub fn is_resolved(&self) -> bool {
        self.resolved_encoding.is_some()
    }

    /// Get the best encoding that can be scheduled now
    pub fn best_available_encoding(&self, available_values: &HashSet<i64>) -> Option<&ConstEncoding> {
        self.encodings.iter().find(|enc| enc.is_ready(available_values))
    }
}

/// Flags to control which constant encodings are enabled
#[derive(Clone, Copy, Debug, Default)]
pub struct ConstEncodingFlags {
    /// Enable AddImm via flow slot
    pub enable_add_imm: bool,
    /// Enable Load via load slot
    pub enable_load: bool,
    /// Enable ALU-based derivations (Add, Sub, Mul, Shift)
    pub enable_alu: bool,
    /// Maximum derivation depth for ALU-based constant generation.
    /// Values requiring deeper derivation chains will use Load instead.
    ///
    /// Derivation depth is estimated as ceil(log2(max(value, smallest_dep))).
    /// For example:
    /// - Value 1: depth 0 (bootstrap)
    /// - Value 2: depth 1 (1+1)
    /// - Value 4: depth 2 (2+2)
    /// - Value 32: depth 5 (16+16)
    /// - Value 240: depth ~8 (requires 120, 60, 30, ...)
    ///
    /// Setting this to 4-5 prevents long critical paths through constant derivation.
    /// None means unlimited (default).
    pub max_alu_depth: Option<usize>,
}

impl ConstEncodingFlags {
    /// All encodings enabled with no depth limit (default)
    pub fn all() -> Self {
        Self {
            enable_add_imm: true,
            enable_load: true,
            enable_alu: true,
            max_alu_depth: None,
        }
    }

    /// Only load encoding enabled (baseline)
    pub fn load_only() -> Self {
        Self {
            enable_add_imm: false,
            enable_load: true,
            enable_alu: false,
            max_alu_depth: None,
        }
    }

    /// All encodings enabled with a depth limit for ALU derivation
    pub fn with_depth_limit(max_depth: usize) -> Self {
        Self {
            enable_add_imm: true,
            enable_load: true,
            enable_alu: true,
            max_alu_depth: Some(max_depth),
        }
    }

    /// Estimate the minimum derivation depth needed to compute a value via ALU.
    ///
    /// This is a heuristic based on the assumption that we start with 1 and
    /// derive larger values through doubling (addition/shift).
    /// For value V, the minimum depth is approximately ceil(log2(V)).
    ///
    /// Returns None for value 0 or 1 (bootstrap, no derivation needed).
    pub fn estimate_derivation_depth(value: i64) -> Option<usize> {
        let abs_val = value.unsigned_abs();
        if abs_val <= 1 {
            return None; // Bootstrap constant, depth 0
        }
        // Minimum depth is ceil(log2(value))
        // e.g., 2 -> 1, 3 -> 2, 4 -> 2, 5-8 -> 3, 9-16 -> 4, etc.
        Some((64 - abs_val.leading_zeros()) as usize)
    }

    /// Check if ALU derivation should be allowed for a given value based on depth limit.
    pub fn allow_alu_for_value(&self, value: i64) -> bool {
        if !self.enable_alu {
            return false;
        }
        match (self.max_alu_depth, Self::estimate_derivation_depth(value)) {
            (Some(max_depth), Some(est_depth)) => est_depth <= max_depth,
            _ => true, // No limit or no depth estimate
        }
    }
}

/// Compute all valid encodings for a constant value.
///
/// Returns encodings sorted by preference:
/// 1. AddImm (uses flow slot, very high priority since flow slots are rarely used)
/// 2. Load (always available, no deps)
/// 3. Add (various decompositions)
/// 4. Shift (if value is related to power of 2)
/// 5. Mul (various decompositions)
pub fn compute_const_encodings(value: i64) -> Vec<ConstEncoding> {
    compute_const_encodings_with_flags(value, ConstEncodingFlags::all())
}

/// Compute valid encodings for a constant value with configurable flags.
///
/// Priority order:
/// 1. ALU derivations (if dependencies available, saves both load and flow slots)
/// 2. AddImm via flow slot (no dependencies, flow slots rarely used)
/// 3. Load (fallback, uses precious load slots)
pub fn compute_const_encodings_with_flags(value: i64, flags: ConstEncodingFlags) -> Vec<ConstEncoding> {
    let mut encodings = Vec::new();

    // ALU-based derivations (highest priority when dependencies are ready)
    // These save both load AND flow slots
    // Use allow_alu_for_value() which checks both enable_alu and depth limit
    if flags.allow_alu_for_value(value) {
        // For positive values, compute decompositions
        if value > 0 {
            // Add decompositions: value = a + b where a <= b
            // Limit to reasonable operands to avoid combinatorial explosion
            let max_operand = value.min(512);
            for a in 0..=max_operand {
                let b = value - a;
                if b >= a && b >= 0 && b <= 512 {
                    encodings.push(ConstEncoding::Add {
                        a_value: a,
                        b_value: b,
                    });
                }
            }

            // Shift decompositions: value = base << shift
            for shift in 1..=9u8 {
                let divisor = 1i64 << shift;
                if value % divisor == 0 {
                    let base = value / divisor;
                    if base > 0 && base <= 256 {
                        encodings.push(ConstEncoding::Shift {
                            base_value: base,
                            shift_amount: shift,
                        });
                    }
                }
            }

            // Mul decompositions: value = a * b where a <= b
            // Only for small factors to avoid blowup
            for a in 2..=16i64 {
                if value % a == 0 {
                    let b = value / a;
                    if b >= a && b <= 256 {
                        encodings.push(ConstEncoding::Mul {
                            a_value: a,
                            b_value: b,
                        });
                    }
                }
            }
        }

        // For negative values, use subtraction from 0
        if value < 0 {
            encodings.push(ConstEncoding::Sub {
                a_value: 0,
                b_value: -value,
            });
        }
    }

    // AddImm via flow slot (second priority - flow slots are rarely used)
    // Works for any value since add_imm has immediate operand
    // Requires ZERO_REG (value 0) to be available
    if flags.enable_add_imm {
        encodings.push(ConstEncoding::AddImm { value });
    }

    // Load is always possible (lowest priority - uses precious load slots)
    if flags.enable_load {
        encodings.push(ConstEncoding::Load);
    }

    encodings
}

// ============================================================================
// VALU to ALU Conversion
// ============================================================================

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
            last_use_reads: HashSet::new(), // Computed later by compute_last_uses
            produces_constant: None,
        };
        new_ops.push(new_op);
    }
    (new_ops, forward_count)
}

// ============================================================================
// PseudoConst Resolution
// ============================================================================

/// Resolve a PseudoConst to a concrete Op.
///
/// Given the chosen encoding and register mappings, produces the actual
/// instruction to execute.
pub fn resolve_pseudo_const(
    pc: &PseudoConst,
    encoding: &ConstEncoding,
    value_to_register: &std::collections::HashMap<i64, i64>,
    next_op_id: usize,
) -> Option<Op> {
    let dest = pc.dest_register?;

    let (engine, slot, reads) = match encoding {
        ConstEncoding::Load => {
            let slot: ArrayVec<SlotVal, MAX_SLOT_SIZE> = [
                SlotVal::Str("const".to_string()),
                SlotVal::Int(dest),
                SlotVal::Int(pc.value),
            ]
            .into_iter()
            .collect();
            (Engine::Load, slot, HashSet::new())
        }

        ConstEncoding::AddImm { value } => {
            // add_imm dest, ZERO_REG, value
            let zero_reg = *value_to_register.get(&0)
                .expect("ZERO_REG must be available for AddImm");
            let slot: ArrayVec<SlotVal, MAX_SLOT_SIZE> = [
                SlotVal::Str("add_imm".to_string()),
                SlotVal::Int(dest),
                SlotVal::Int(zero_reg),
                SlotVal::Int(*value),
            ]
            .into_iter()
            .collect();
            let reads: HashSet<i64> = [zero_reg].into_iter().collect();
            (Engine::Flow, slot, reads)
        }

        ConstEncoding::Add { a_value, b_value } => {
            let a_reg = *value_to_register.get(a_value)?;
            let b_reg = *value_to_register.get(b_value)?;
            let slot: ArrayVec<SlotVal, MAX_SLOT_SIZE> = [
                SlotVal::Str("+".to_string()),
                SlotVal::Int(dest),
                SlotVal::Int(a_reg),
                SlotVal::Int(b_reg),
            ]
            .into_iter()
            .collect();
            let reads: HashSet<i64> = [a_reg, b_reg].into_iter().collect();
            (Engine::Alu, slot, reads)
        }

        ConstEncoding::Sub { a_value, b_value } => {
            let a_reg = *value_to_register.get(a_value)?;
            let b_reg = *value_to_register.get(b_value)?;
            let slot: ArrayVec<SlotVal, MAX_SLOT_SIZE> = [
                SlotVal::Str("-".to_string()),
                SlotVal::Int(dest),
                SlotVal::Int(a_reg),
                SlotVal::Int(b_reg),
            ]
            .into_iter()
            .collect();
            let reads: HashSet<i64> = [a_reg, b_reg].into_iter().collect();
            (Engine::Alu, slot, reads)
        }

        ConstEncoding::Mul { a_value, b_value } => {
            let a_reg = *value_to_register.get(a_value)?;
            let b_reg = *value_to_register.get(b_value)?;
            let slot: ArrayVec<SlotVal, MAX_SLOT_SIZE> = [
                SlotVal::Str("*".to_string()),
                SlotVal::Int(dest),
                SlotVal::Int(a_reg),
                SlotVal::Int(b_reg),
            ]
            .into_iter()
            .collect();
            let reads: HashSet<i64> = [a_reg, b_reg].into_iter().collect();
            (Engine::Alu, slot, reads)
        }

        ConstEncoding::Shift { base_value, shift_amount } => {
            let base_reg = *value_to_register.get(base_value)?;
            let shift_reg = *value_to_register.get(&(*shift_amount as i64))?;
            let slot: ArrayVec<SlotVal, MAX_SLOT_SIZE> = [
                SlotVal::Str("<<".to_string()),
                SlotVal::Int(dest),
                SlotVal::Int(base_reg),
                SlotVal::Int(shift_reg),
            ]
            .into_iter()
            .collect();
            let reads: HashSet<i64> = [base_reg, shift_reg].into_iter().collect();
            (Engine::Alu, slot, reads)
        }
    };

    let writes: HashSet<i64> = [dest].into_iter().collect();

    Some(Op {
        id: next_op_id,
        engine,
        slot,
        cycle: -1,
        dep_ids: HashSet::new(), // Will be filled by caller based on encoding deps
        reads,
        writes,
        last_use_reads: HashSet::new(),
        produces_constant: Some(pc.value),
    })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_encodings_zero() {
        let encodings = compute_const_encodings(0);
        assert!(encodings.contains(&ConstEncoding::Load));
        assert!(encodings.contains(&ConstEncoding::AddImm { value: 0 }));
    }

    #[test]
    fn test_compute_encodings_small() {
        let encodings = compute_const_encodings(4);
        assert!(encodings.contains(&ConstEncoding::Load));
        assert!(encodings.contains(&ConstEncoding::Add { a_value: 2, b_value: 2 }));
        assert!(encodings.contains(&ConstEncoding::Mul { a_value: 2, b_value: 2 }));
        assert!(encodings.contains(&ConstEncoding::Shift { base_value: 1, shift_amount: 2 }));
        assert!(encodings.contains(&ConstEncoding::Shift { base_value: 2, shift_amount: 1 }));
    }

    #[test]
    fn test_compute_encodings_240() {
        let encodings = compute_const_encodings(240);
        assert!(encodings.contains(&ConstEncoding::Load));
        // 240 = 120 + 120
        assert!(encodings.contains(&ConstEncoding::Add { a_value: 120, b_value: 120 }));
        // 240 = 128 + 112
        assert!(encodings.contains(&ConstEncoding::Add { a_value: 112, b_value: 128 }));
        // 240 = 15 << 4
        assert!(encodings.contains(&ConstEncoding::Shift { base_value: 15, shift_amount: 4 }));
        // 240 = 15 * 16
        assert!(encodings.contains(&ConstEncoding::Mul { a_value: 15, b_value: 16 }));
    }

    #[test]
    fn test_encoding_dependencies() {
        let load = ConstEncoding::Load;
        assert!(load.dependencies().is_empty());

        let add_imm = ConstEncoding::AddImm { value: 42 };
        assert_eq!(add_imm.dependencies(), vec![0]); // Depends on ZERO_REG

        let add = ConstEncoding::Add { a_value: 10, b_value: 20 };
        assert_eq!(add.dependencies(), vec![10, 20]);

        let shift = ConstEncoding::Shift { base_value: 5, shift_amount: 3 };
        assert_eq!(shift.dependencies(), vec![5, 3]);
    }

    #[test]
    fn test_encoding_is_ready() {
        let mut available: HashSet<i64> = HashSet::new();

        let load = ConstEncoding::Load;
        assert!(load.is_ready(&available)); // Load always ready

        let add_imm = ConstEncoding::AddImm { value: 42 };
        assert!(!add_imm.is_ready(&available)); // Needs ZERO_REG

        available.insert(0); // ZERO_REG available
        assert!(add_imm.is_ready(&available));

        let add = ConstEncoding::Add { a_value: 10, b_value: 20 };
        assert!(!add.is_ready(&available));

        available.insert(10);
        assert!(!add.is_ready(&available));

        available.insert(20);
        assert!(add.is_ready(&available));
    }

    #[test]
    fn test_pseudo_const_best_encoding() {
        let pc = PseudoConst::new(0, 240, Some(100));
        let mut available: HashSet<i64> = HashSet::new();

        // Initially only Load is available (ALU needs deps, AddImm needs ZERO_REG)
        let best = pc.best_available_encoding(&available);
        assert_eq!(best, Some(&ConstEncoding::Load));

        // Add ZERO_REG - now AddImm is preferred over Load
        available.insert(0);
        let best = pc.best_available_encoding(&available);
        assert_eq!(best, Some(&ConstEncoding::AddImm { value: 240 }));

        // Add 120 - now ALU Add is preferred (ALU > AddImm > Load)
        available.insert(120);
        let best = pc.best_available_encoding(&available);
        assert_eq!(best, Some(&ConstEncoding::Add { a_value: 120, b_value: 120 }));
    }

    #[test]
    fn test_pseudo_const_critical() {
        let pc = PseudoConst::new_critical(0, 240, Some(100));
        assert!(pc.is_critical);
        // Load should be first encoding for critical constants
        assert_eq!(pc.encodings[0], ConstEncoding::Load);
    }

    #[test]
    fn test_resolve_pseudo_const_load() {
        let pc = PseudoConst::new(0, 42, Some(100));
        let value_to_reg: std::collections::HashMap<i64, i64> = std::collections::HashMap::new();

        let op = resolve_pseudo_const(&pc, &ConstEncoding::Load, &value_to_reg, 1000);
        assert!(op.is_some());

        let op = op.unwrap();
        assert_eq!(op.engine, Engine::Load);
        assert_eq!(op.produces_constant, Some(42));
        assert!(op.writes.contains(&100));
    }

    #[test]
    fn test_resolve_pseudo_const_add() {
        let pc = PseudoConst::new(0, 30, Some(100));
        let mut value_to_reg: std::collections::HashMap<i64, i64> = std::collections::HashMap::new();
        value_to_reg.insert(10, 50);  // value 10 is in register 50
        value_to_reg.insert(20, 60);  // value 20 is in register 60

        let encoding = ConstEncoding::Add { a_value: 10, b_value: 20 };
        let op = resolve_pseudo_const(&pc, &encoding, &value_to_reg, 1000);
        assert!(op.is_some());

        let op = op.unwrap();
        assert_eq!(op.engine, Engine::Alu);
        assert!(op.reads.contains(&50));
        assert!(op.reads.contains(&60));
        assert!(op.writes.contains(&100));
        assert_eq!(op.produces_constant, Some(30));
    }

    #[test]
    fn test_resolve_pseudo_const_add_imm() {
        let pc = PseudoConst::new(0, 42, Some(100));
        let mut value_to_reg: std::collections::HashMap<i64, i64> = std::collections::HashMap::new();
        value_to_reg.insert(0, 0);  // ZERO_REG (value 0 is in register 0)

        let encoding = ConstEncoding::AddImm { value: 42 };
        let op = resolve_pseudo_const(&pc, &encoding, &value_to_reg, 1000);
        assert!(op.is_some());

        let op = op.unwrap();
        assert_eq!(op.engine, Engine::Flow);
        assert!(op.reads.contains(&0));  // Reads from ZERO_REG
        assert!(op.writes.contains(&100));
        assert_eq!(op.produces_constant, Some(42));
    }

    #[test]
    fn test_add_imm_encoding_engine() {
        let add_imm = ConstEncoding::AddImm { value: 123 };
        assert_eq!(add_imm.engine(), Engine::Flow);

        let load = ConstEncoding::Load;
        assert_eq!(load.engine(), Engine::Load);

        let add = ConstEncoding::Add { a_value: 1, b_value: 2 };
        assert_eq!(add.engine(), Engine::Alu);
    }
}
