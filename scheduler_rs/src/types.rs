//! Core types and constants for the scheduler.
//!
//! [`MachineConfig`] defines slot capacities per engine (ALU, VALU, Load, Store, Flow)
//! and is used by the CP-SAT scheduler; [`Engine::slot_limit()`] gives the default hardware limits.

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
    Barrier = 8, // Scheduling-only barrier (not real hardware)
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
            "barrier" => Some(Engine::Barrier),
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
            Engine::Barrier => "barrier",
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
            Engine::Barrier => 0, // Takes no slots
        }
    }

    /// Check if this is a real hardware engine (vs scheduling-internal)
    pub fn is_hardware(&self) -> bool {
        !matches!(
            self,
            Engine::EliminatedBroadcast | Engine::ShiftedValu | Engine::Barrier
        )
    }

    /// Slot limit for this engine from a machine config (for configurable machines).
    /// Returns 0 for non-hardware / internal engines.
    pub fn slot_limit_from(&self, machine: &MachineConfig) -> u32 {
        match self {
            Engine::Load => machine.load_slots,
            Engine::Store => machine.store_slots,
            Engine::Alu => machine.alu_slots,
            Engine::Valu => machine.valu_slots,
            Engine::Flow => machine.flow_slots,
            Engine::Debug => 64,
            Engine::EliminatedBroadcast | Engine::ShiftedValu | Engine::Barrier => 0,
        }
    }
}

// ============================================================================
// MachineConfig
// ============================================================================

/// Machine configuration: slot capacities per engine per cycle.
///
/// Defines the target machine for scheduling. The scheduler uses these limits
/// in cumulative constraints (no more than `N` ops of a given engine per cycle)
/// and in capacity-based lower bounds on makespan.
///
/// **Default** ([`Default::default()`]): Matches the hardware spec (ALU 12, VALU 6,
/// Load 2, Store 2, Flow 1). Override for tests or alternate machines (e.g.
/// toy configs with 1 VALU / 8 ALU to exercise VALU→ALU shifting).
///
/// Used by [`crate::cpsat_scheduler::SsaCpsatConfig::machine`]; the greedy
/// scheduler still uses [`Engine::slot_limit()`] (fixed hardware values).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MachineConfig {
    /// ALU slots per cycle (default 12).
    pub alu_slots: u32,
    /// VALU slots per cycle (default 6). A shifted VALU op uses 8 ALU slots instead.
    pub valu_slots: u32,
    /// Load slots per cycle (default 2).
    pub load_slots: u32,
    /// Store slots per cycle (default 2).
    pub store_slots: u32,
    /// Flow slots per cycle (default 1). vselect, AddImm, jump, etc.
    pub flow_slots: u32,
}

impl Default for MachineConfig {
    fn default() -> Self {
        Self {
            alu_slots: 12,
            valu_slots: 6,
            load_slots: 2,
            store_slots: 2,
            flow_slots: 1,
        }
    }
}

impl MachineConfig {
    /// Slot limit for an engine (for use in cumulative constraints and validation).
    pub fn slot_limit(&self, engine: Engine) -> u32 {
        engine.slot_limit_from(self)
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
    Add = 0,  // "+"
    Sub = 1,  // "-"
    Mul = 2,  // "*"
    Div = 3,  // "/"
    Mod = 4,  // "%"
    Cdiv = 5, // "cdiv" (ceiling division)

    // Bitwise operations
    And = 6,  // "&"
    Or = 7,   // "|"
    Xor = 8,  // "^"
    Shl = 9,  // "<<"
    Shr = 10, // ">>"

    // Comparison operations
    Eq = 11, // "=="
    Ne = 12, // "!="
    Lt = 13, // "<"
    Le = 14, // "<="
    Gt = 15, // ">"
    Ge = 16, // ">="

    // Vector-specific operations
    Vbroadcast = 17,
    MultiplyAdd = 18, // "multiply_add"

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

    // Scheduling-only pseudo-ops (elided from output)
    Barrier = 32,      // Round boundary barrier
    ConstChoice = 33,  // Constant with encoding choice (Load vs AddImm)
    SelectChoice = 34, // Selection with encoding choice (vselect vs FMA: sub + fma)
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
            "barrier" => Some(Opcode::Barrier),
            "const_choice" => Some(Opcode::ConstChoice),
            "select_choice" => Some(Opcode::SelectChoice),
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
            Opcode::Barrier => "barrier",
            Opcode::ConstChoice => "const_choice",
            Opcode::SelectChoice => "select_choice",
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
// AluDeriveInfo
// ============================================================================

/// Info for a ConstChoice op that chose ALU-derived encoding: `dest = left opcode right`.
/// Used in `const_alu_derive_ops: HashMap<op_id, AluDeriveInfo>`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AluDeriveInfo {
    /// Op ID of the left producer (defines a value read by the derived op).
    pub left_op_id: usize,
    /// Op ID of the right producer (defines a value read by the derived op).
    pub right_op_id: usize,
    /// The ALU opcode to use (e.g. Add, Mul).
    pub opcode: Opcode,
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
    Engine::from_str(engine)
        .map(|e| e.slot_limit())
        .unwrap_or(1)
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
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SlotVal {
    Str(String), // Legacy: will be replaced with Op(Opcode)
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

    /// Optional trace ID for debug ops (trace_input/trace_output).
    /// Set when this Op is a debug op; propagated from SsaOp for traced logical ops.
    pub trace_id: Option<String>,
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

    /// Flow add_imm: `add_imm dest scratch[0] value`
    /// Cost: 1 flow slot
    /// IMPORTANT: Scratch memory is ALWAYS initialized to 0 at startup, so scratch[0] == 0
    /// is guaranteed by hardware initialization. AddImm can reference address 0 directly
    /// without any dependency on a zero-loading op.
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
            ConstEncoding::AddImm { .. } => vec![], // No scheduling dependencies - address 0 always contains 0 (hardware initialization)
            ConstEncoding::Add { a_value, b_value } => vec![*a_value, *b_value],
            ConstEncoding::Sub { a_value, b_value } => vec![*a_value, *b_value],
            ConstEncoding::Mul { a_value, b_value } => vec![*a_value, *b_value],
            ConstEncoding::Shift {
                base_value,
                shift_amount,
            } => {
                // Shift amount must also be available as a constant
                vec![*base_value, *shift_amount as i64]
            }
        }
    }

    /// Check if this encoding's dependencies are satisfied
    pub fn is_ready(&self, available_values: &HashSet<i64>) -> bool {
        match self {
            ConstEncoding::Load => true,          // No dependencies
            ConstEncoding::AddImm { .. } => true, // No dependencies - address 0 always contains 0 (hardware initialization)
            ConstEncoding::Add { a_value, b_value }
            | ConstEncoding::Sub { a_value, b_value }
            | ConstEncoding::Mul { a_value, b_value } => {
                available_values.contains(a_value) && available_values.contains(b_value)
            }
            ConstEncoding::Shift {
                base_value,
                shift_amount,
            } => {
                available_values.contains(base_value)
                    && available_values.contains(&(*shift_amount as i64))
            }
        }
    }

    /// Opcode for ALU encodings (Add, Sub, Mul, Shift). None for Load/AddImm.
    pub fn alu_opcode(&self) -> Option<Opcode> {
        match self {
            ConstEncoding::Load | ConstEncoding::AddImm { .. } => None,
            ConstEncoding::Add { .. } => Some(Opcode::Add),
            ConstEncoding::Sub { .. } => Some(Opcode::Sub),
            ConstEncoding::Mul { .. } => Some(Opcode::Mul),
            ConstEncoding::Shift { .. } => Some(Opcode::Shl),
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
    pub fn new_with_flags(
        id: usize,
        value: i64,
        dest_register: Option<i64>,
        flags: ConstEncodingFlags,
    ) -> Self {
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
    pub fn new_critical_with_flags(
        id: usize,
        value: i64,
        dest_register: Option<i64>,
        flags: ConstEncodingFlags,
    ) -> Self {
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
    pub fn best_available_encoding(
        &self,
        available_values: &HashSet<i64>,
    ) -> Option<&ConstEncoding> {
        self.encodings
            .iter()
            .find(|enc| enc.is_ready(available_values))
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
pub fn compute_const_encodings_with_flags(
    value: i64,
    flags: ConstEncodingFlags,
) -> Vec<ConstEncoding> {
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
    // IMPORTANT: Scratch memory is ALWAYS initialized to 0 at startup, so scratch[0] == 0
    // is guaranteed by hardware initialization. AddImm can reference address 0 directly.
    if flags.enable_add_imm {
        encodings.push(ConstEncoding::AddImm { value });
    }

    // Load is always possible (lowest priority - uses precious load slots)
    if flags.enable_load {
        encodings.push(ConstEncoding::Load);
    }

    encodings
}

/// Compute ALU-only encodings (Add, Sub, Mul, Shift) for a constant value
/// whose operands are in `available`. Returns at most `max_count` encodings.
/// Used by emit_const_choice to offer ALU-derived as a third option when
/// operand constants are already materialized earlier in the kernel.
pub fn compute_const_encodings_alu_ready(
    value: i64,
    available: &HashSet<i64>,
    max_count: usize,
) -> Vec<ConstEncoding> {
    if max_count == 0 {
        return vec![];
    }
    let flags = ConstEncodingFlags {
        enable_add_imm: false,
        enable_load: false,
        enable_alu: true,
        max_alu_depth: Some(4), // keep derivation chain short
    };
    let all = compute_const_encodings_with_flags(value, flags);
    all.into_iter()
        .filter(|e| e.is_ready(available))
        .take(max_count)
        .collect()
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
            ]
            .into_iter()
            .collect(),
            cycle: -1,
            dep_ids: op.dep_ids.clone(),
            reads: [actual_src1, actual_src2].into_iter().collect(),
            writes: [dest + i_offset].into_iter().collect(),
            last_use_reads: HashSet::new(), // Computed later by compute_last_uses
            produces_constant: None,
            trace_id: None,
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
            // add_imm dest, scratch[0], value
            // IMPORTANT: Scratch memory is ALWAYS initialized to 0 at startup, so scratch[0] == 0
            // is guaranteed by hardware initialization. add_imm reads from address 0, but we don't
            // need a precedence constraint because address 0 is always initialized to 0.
            let slot: ArrayVec<SlotVal, MAX_SLOT_SIZE> = [
                SlotVal::Str("add_imm".to_string()),
                SlotVal::Int(dest),
                SlotVal::Int(0), // Address 0 always contains 0 (hardware initialization)
                SlotVal::Int(*value),
            ]
            .into_iter()
            .collect();
            let reads: HashSet<i64> = [0].into_iter().collect(); // Reads from address 0 (but no precedence constraint needed)
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

        ConstEncoding::Shift {
            base_value,
            shift_amount,
        } => {
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
        trace_id: None,
    })
}

// ============================================================================
// SSA Virtual Registers
// ============================================================================

/// Virtual register width - scalar (1 word) or vector (VLEN words)
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum VRegWidth {
    Scalar,
    Vector,
}

/// Virtual register - represents a value, not a physical location.
///
/// Unlike concrete scratch addresses, VRegs have SSA semantics:
/// - Each VReg is defined exactly once
/// - Dependencies are only true data dependencies (RAW)
/// - No false WAW or WAR dependencies
///
/// Physical addresses are assigned after scheduling via linear scan allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct VReg {
    /// Unique identifier within the kernel
    pub id: u32,
    /// Width: scalar (1 word) or vector (VLEN=8 words)
    pub width: VRegWidth,
}

impl VReg {
    /// Create a scalar VReg
    pub fn scalar(id: u32) -> Self {
        Self {
            id,
            width: VRegWidth::Scalar,
        }
    }

    /// Create a vector VReg
    pub fn vector(id: u32) -> Self {
        Self {
            id,
            width: VRegWidth::Vector,
        }
    }

    /// Get the size in words (1 for scalar, VLEN for vector)
    pub fn size(&self) -> usize {
        match self.width {
            VRegWidth::Scalar => 1,
            VRegWidth::Vector => VLEN,
        }
    }

    /// Check if this is a scalar VReg
    pub fn is_scalar(&self) -> bool {
        self.width == VRegWidth::Scalar
    }

    /// Check if this is a vector VReg
    pub fn is_vector(&self) -> bool {
        self.width == VRegWidth::Vector
    }
}

/// Operand for SSA operations - either a VReg reference or an immediate value.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Operand {
    /// Reference to a virtual register
    VReg(VReg),
    /// Immediate value (constant embedded in instruction)
    Imm(i64),
}

impl Operand {
    /// Create a VReg operand from a scalar VReg
    pub fn scalar(id: u32) -> Self {
        Self::VReg(VReg::scalar(id))
    }

    /// Create a VReg operand from a vector VReg
    pub fn vector(id: u32) -> Self {
        Self::VReg(VReg::vector(id))
    }

    /// Create an immediate operand
    pub fn imm(value: i64) -> Self {
        Self::Imm(value)
    }

    /// Get the VReg if this is a VReg operand
    pub fn as_vreg(&self) -> Option<VReg> {
        match self {
            Self::VReg(v) => Some(*v),
            Self::Imm(_) => None,
        }
    }

    /// Get the immediate value if this is an immediate operand
    pub fn as_imm(&self) -> Option<i64> {
        match self {
            Self::Imm(v) => Some(*v),
            Self::VReg(_) => None,
        }
    }
}

// ============================================================================
// Variable Type and Source Tracking
// ============================================================================

/// Classification of what a variable represents in the kernel.
///
/// Used for usage accounting to understand scratch space consumption
/// by different categories of values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VarType {
    /// Scalar constants (c[i], hash constants/shifts like hc[i], hs[i])
    Constant,
    /// Memory pointers (forest_p, idx_p, val_p, extra_room_p)
    Pointer,
    /// Cached tree values (tc[i], tc_r3[i], td_* deltas)
    TreeCache,
    /// Hash-related state (v_hc, v_hs, v_hash_mult, v_stage5_xor)
    HashState,
    /// Batch data vectors (all_idx, all_val - the input/output data)
    BatchData,
    /// Per-round intermediate state (v_nv, v_t - tree values and temps)
    RoundState,
    /// True temporaries (intermediate computation values)
    Temporary,
    /// Address computations (for loads/stores)
    Address,
    /// Condition flags from comparisons
    Condition,
}

impl VarType {
    /// Parse from string (for deserialization)
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "constant" | "Constant" => Some(VarType::Constant),
            "pointer" | "Pointer" => Some(VarType::Pointer),
            "tree_cache" | "TreeCache" => Some(VarType::TreeCache),
            "hash_state" | "HashState" => Some(VarType::HashState),
            "batch_data" | "BatchData" => Some(VarType::BatchData),
            "round_state" | "RoundState" => Some(VarType::RoundState),
            "temporary" | "Temporary" => Some(VarType::Temporary),
            "address" | "Address" => Some(VarType::Address),
            "condition" | "Condition" => Some(VarType::Condition),
            _ => None,
        }
    }

    /// Convert to string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            VarType::Constant => "constant",
            VarType::Pointer => "pointer",
            VarType::TreeCache => "tree_cache",
            VarType::HashState => "hash_state",
            VarType::BatchData => "batch_data",
            VarType::RoundState => "round_state",
            VarType::Temporary => "temporary",
            VarType::Address => "address",
            VarType::Condition => "condition",
        }
    }
}

/// Phase of kernel emission.
///
/// Tracks which logical section of the kernel generated a variable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EmissionPhase {
    /// Common initialization (constants, pointers, hash setup)
    Init,
    /// VALU selection setup (depth 3 tree cache)
    ValuSelectionInit,
    /// Loading batch data (idx/val vectors)
    LoadData,
    /// Tree access within a round
    TreeAccess,
    /// XOR step within a round
    Xor,
    /// Hash computation within a round
    Hash,
    /// Branch/index update within a round
    Branch,
    /// Storing results
    StoreData,
}

impl EmissionPhase {
    pub fn as_str(&self) -> &'static str {
        match self {
            EmissionPhase::Init => "init",
            EmissionPhase::ValuSelectionInit => "valu_selection_init",
            EmissionPhase::LoadData => "load_data",
            EmissionPhase::TreeAccess => "tree_access",
            EmissionPhase::Xor => "xor",
            EmissionPhase::Hash => "hash",
            EmissionPhase::Branch => "branch",
            EmissionPhase::StoreData => "store_data",
        }
    }
}

/// Hint from emission for symmetry breaking during optimization.
///
/// **Why we want this**: The scheduler explores many equivalent solutions (CP-SAT reports
/// hundreds of orbits). Symmetry hints let emission mark structurally symmetric ops so the
/// optimizer can add ordering constraints and prune redundant search without losing the
/// optimal schedule.
///
/// **Generic structure**: Optimizer only sees ordering groups and lexicographic streams.
/// Emission encodes problem-specific structure (batch, round, etc.) into opaque keys.
///
/// - **Ordering group**: ops with the same `ordering_group` key form a chain; optimizer adds
///   cycle[i] <= cycle[j] for consecutive ops in emission order.
/// - **Lexicographic stream**: when set, (stream_key, item_key) identifies an item in a stream.
///   Optimizer groups by stream_key, orders by item_key, and enforces first_op[item_i] <=
///   first_op[item_{i+1}] for consecutive items.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SymmetryHint {
    /// Key for ordering group. Ops with same key get cycle[i] <= cycle[j] by emission order.
    /// Empty = no ordering hint for this op.
    pub ordering_group: Vec<i64>,

    /// Optional (stream_key, item_key) for lexicographic ordering across items in a stream.
    /// Optimizer groups by stream_key, sorts by item_key, adds first_op[i] <= first_op[i+1].
    /// None = no lexicographic hint for this op.
    pub lexicographic_stream: Option<(Vec<i64>, i64)>,
}

/// Source information for a variable - tracks where/when it was created.
///
/// This metadata enables usage accounting by round and category.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VarSource {
    /// Which round generated this (0-15), or None for init/finalization
    pub round: Option<usize>,
    /// Tree depth when this was created (0-9)
    pub tree_depth: Option<usize>,
    /// Iteration index within round (for unrolled loops)
    pub iter: Option<usize>,
    /// Group index
    pub group: Option<usize>,
    /// Phase of emission
    pub phase: EmissionPhase,
}

impl VarSource {
    /// Create source info for initialization phase
    pub fn init() -> Self {
        Self {
            round: None,
            tree_depth: None,
            iter: None,
            group: None,
            phase: EmissionPhase::Init,
        }
    }

    /// Create source info for a specific round and phase
    pub fn round(round: usize, tree_depth: usize, phase: EmissionPhase) -> Self {
        Self {
            round: Some(round),
            tree_depth: Some(tree_depth),
            iter: None,
            group: None,
            phase,
        }
    }

    /// Set iteration and group indices
    pub fn with_iter_group(mut self, iter: usize, group: usize) -> Self {
        self.iter = Some(iter);
        self.group = Some(group);
        self
    }
}

/// SSA Operation - uses virtual registers instead of concrete addresses.
///
/// This is the SSA form used during kernel construction and scheduling.
/// After scheduling, VRegs are mapped to physical addresses via linear scan.
///
/// Key differences from `Op`:
/// - `dest`: Optional VReg (SSA: each value defined once)
/// - `srcs`: Vec of Operands (VReg or Imm)
/// - No `reads`/`writes` HashSets (derived from VRegs)
/// - Dependencies are RAW-only (true data flow)
#[derive(Clone, Debug)]
pub struct SsaOp {
    /// Unique identifier
    pub id: usize,

    /// Execution engine
    pub engine: Engine,

    /// Operation opcode
    pub opcode: Opcode,

    /// Destination VReg (if any - stores have no dest)
    pub dest: Option<VReg>,

    /// Source operands (VRegs or immediates)
    pub srcs: Vec<Operand>,

    /// Op IDs this op depends on (RAW dependencies only)
    pub dep_ids: HashSet<usize>,

    /// Scheduled cycle (-1 if unscheduled)
    pub cycle: i32,

    /// If this op produces a known constant value
    pub produces_constant: Option<i64>,

    /// Source location for debugging (file:line where this op was emitted)
    pub source: Option<&'static std::panic::Location<'static>>,

    /// Classification of what this variable represents (for usage accounting)
    pub var_type: Option<VarType>,

    /// Source information: which round/phase/iteration generated this op
    pub source_info: Option<VarSource>,

    /// Optional symmetry hint from emission. Optimizer uses `ordering_group` and
    /// `lexicographic_stream` to add ordering constraints that prune redundant search.
    /// Emission sets this; optimization consumes it. See `SymmetryHint` and
    /// `docs/CPSAT_SLOW_PROGRESS_ANALYSIS.md`.
    pub symmetry_hint: Option<SymmetryHint>,

    /// For ConstChoice ops: alternative encodings the scheduler can choose from.
    /// The greedy scheduler uses Load (default), while CP-SAT models the choice.
    pub const_alternatives: Option<Vec<ConstEncoding>>,

    /// For SelectChoice ops: whether subtraction is needed in FMA path.
    /// If `false_val = 0`, subtraction is not needed (FMA simplifies to multiply only).
    pub needs_subtraction: bool,

    /// For SelectChoice ops: optional precomputed delta VReg.
    /// When set, indicates that `true_val = false_val + delta_vreg`, allowing FMA path
    /// to use `cond * delta_vreg + false_val` directly (1 VALU op) instead of computing
    /// `sub = true_val - false_val` on-the-fly (2 VALU ops).
    /// None when delta is not precomputed (e.g., cascaded SelectChoice where base is result of previous SelectChoice).
    pub delta_vreg: Option<VReg>,

    /// For ops that are part of a SelectChoice expansion (sub or fma):
    /// Points to the original SelectChoice op's ID.
    /// None for regular ops or the SelectChoice op itself.
    pub select_choice_parent: Option<usize>,

    /// For SelectChoice ops WITHOUT delta: VReg to hold the sub temp result.
    /// When FMA path is chosen without a precomputed delta, we emit:
    ///   sub_temp = true_val - false_val
    ///   dest = cond * sub_temp + false_val
    /// This VReg is allocated during emission and assigned an address during address assignment.
    /// None when delta_vreg is set (no sub needed) or for non-SelectChoice ops.
    pub sub_temp_vreg: Option<VReg>,

    /// Optional trace ID for semantic debugging (trace_input/trace_output).
    /// When set, lowering emits debug ops at first/last cycle. Format: "r{round}_g{group}_{phase}".
    pub trace_id: Option<String>,
}

impl SsaOp {
    /// Create a new unscheduled SsaOp
    pub fn new(
        id: usize,
        engine: Engine,
        opcode: Opcode,
        dest: Option<VReg>,
        srcs: Vec<Operand>,
    ) -> Self {
        Self {
            id,
            engine,
            opcode,
            dest,
            srcs,
            dep_ids: HashSet::new(),
            cycle: -1,
            produces_constant: None,
            source: None,
            var_type: None,
            source_info: None,
            symmetry_hint: None,
            const_alternatives: None,
            needs_subtraction: true, // Default: subtraction needed (false_val != 0)
            delta_vreg: None,        // Default: no precomputed delta
            select_choice_parent: None, // Default: not part of SelectChoice expansion
            sub_temp_vreg: None, // Default: no sub temp needed (set for SelectChoice without delta)
            trace_id: None,
        }
    }

    /// Get all VRegs read by this op
    pub fn reads(&self) -> impl Iterator<Item = VReg> + '_ {
        self.srcs.iter().filter_map(|op| op.as_vreg())
    }

    /// Get the destination VReg if any
    pub fn writes(&self) -> Option<VReg> {
        self.dest
    }

    /// Check if this op is part of a SelectChoice expansion (sub or fma)
    pub fn is_select_choice_expansion(&self) -> bool {
        self.select_choice_parent.is_some()
    }

    /// Get the parent SelectChoice op ID if this is part of an expansion
    pub fn select_choice_parent_id(&self) -> Option<usize> {
        self.select_choice_parent
    }

    /// Check if this is a sub op from SelectChoice expansion
    pub fn is_select_choice_sub(&self) -> bool {
        self.select_choice_parent.is_some() && self.opcode == Opcode::Sub
    }

    /// Check if this is an fma op from SelectChoice expansion
    pub fn is_select_choice_fma(&self) -> bool {
        self.select_choice_parent.is_some() && self.opcode == Opcode::MultiplyAdd
    }

    /// Check if this op is scheduled
    pub fn is_scheduled(&self) -> bool {
        self.cycle >= 0
    }

    /// Check if this is a vector operation (dest is a vector VReg)
    pub fn is_vector(&self) -> bool {
        self.dest.map(|v| v.is_vector()).unwrap_or(false)
    }

    /// Get unique VRegs read by this op (deduplicated).
    ///
    /// # Why This Matters: The Duplicate Operand Bug
    ///
    /// Operations can have the same VReg appear multiple times as operands.
    /// For example, `add(v1, v1)` doubles a value using v1 twice. Without
    /// deduplication, several bugs can occur:
    ///
    /// ## Bug 1: LUC (Last-Use-Count) Miscalculation
    ///
    /// When computing how many VRegs an op will free (for scheduling priority),
    /// `add(v1, v1)` would incorrectly report LUC=2 instead of LUC=1, since v1
    /// only needs to be freed once.
    ///
    /// ## Bug 2: `vreg_remaining_users` Double-Decrement
    ///
    /// Each VReg tracks how many ops still need to read it. When scheduling
    /// `add(v1, v1)`, if we decrement the counter for each operand (twice),
    /// we'd prematurely mark v1 as having 0 remaining users, causing:
    /// - Early deallocation of v1's storage
    /// - Potential data corruption if another op still needs v1
    ///
    /// ## Bug 3: `words_freed` Over-Estimation
    ///
    /// When checking if an op fits the scratch budget, we calculate how many
    /// words it will free. `add(v1, v1)` where v1 is a vector (8 words) would
    /// incorrectly report 16 words freed instead of 8.
    ///
    /// # Example
    ///
    /// ```text
    /// Op: add(v1, v1) -> v2   // v1 appears twice
    /// reads():           [v1, v1]  // Returns both occurrences
    /// unique_src_vregs(): [v1]     // Returns v1 only once
    /// ```
    ///
    /// # Returns
    ///
    /// A Vec of unique VRegs in first-seen order. Vec is used instead of HashSet
    /// to maintain deterministic ordering for reproducible scheduling.
    pub fn unique_src_vregs(&self) -> Vec<VReg> {
        let mut seen = HashSet::new();
        let mut result = Vec::new();
        for src in &self.srcs {
            if let Operand::VReg(vreg) = src {
                if seen.insert(*vreg) {
                    result.push(*vreg);
                }
            }
        }
        result
    }
}

// ============================================================================
// Precedence Offset Trait
// ============================================================================
//
// Defines per-op precedence delays for ASAP/ALAP computation.
// Source of truth: docs/ASAP_ALAP_PRECEDENCE_OFFSETS.md

/// Provides per-op precedence offsets for ASAP and ALAP bound computation.
///
/// Called on the **predecessor** op to determine how many cycles until its result
/// is visible to consumers:
///
/// - **Normal ops**: offset = 1 (result visible next cycle)
/// - **SelectChoice FMA ops** (`sub_temp_vreg.is_some()`): offset = 2
///   (sub emitted at cycle+1, FMA result visible at cycle+2)
/// - **Shiftable VALU ops**: ASAP offset = 1 (min delay, stays VALU),
///   ALAP offset = 8 (max delay, shifted to 8 ALU ops that complete later)
///
/// See `docs/ASAP_ALAP_PRECEDENCE_OFFSETS.md` for full semantics.
pub trait PrecedenceOffset {
    /// Compute the precedence offset from this op (predecessor) to its consumers for ASAP (forward) propagation.
    ///
    /// # Arguments
    /// * `self_is_shiftable` - Whether this op itself is a shiftable VALU op
    fn precedence_offset_for_asap(&self, self_is_shiftable: bool) -> i32;

    /// Compute the precedence offset from this op (predecessor) to its consumers for ALAP (backward) propagation.
    ///
    /// # Arguments
    /// * `self_is_shiftable` - Whether this op itself is a shiftable VALU op
    fn precedence_offset_for_alap(&self, self_is_shiftable: bool) -> i32;
}

impl PrecedenceOffset for SsaOp {
    fn precedence_offset_for_asap(&self, self_is_shiftable: bool) -> i32 {
        // SelectChoice with sub_temp_vreg: ASAP uses min delay (vselect path = 1 cycle).
        // The FMA path (2 cycles) is handled by explicit CP-SAT constraints in section 5d,
        // just like shiftable ops use min_delay=1 here and let the model handle ALU shift delay.
        // Using max_delay=2 here would unnecessarily tighten domains and cause chain amplification.
        if self.opcode == Opcode::SelectChoice && self.sub_temp_vreg.is_some() {
            return 1;
        }
        // This op is shiftable VALU: ASAP uses min delay (1 = stays as VALU)
        if self_is_shiftable {
            return 1;
        }
        // Default: 1 cycle latency
        1
    }

    fn precedence_offset_for_alap(&self, self_is_shiftable: bool) -> i32 {
        // SelectChoice with sub_temp_vreg: FMA path takes 2 cycles (sub + fma)
        if self.opcode == Opcode::SelectChoice && self.sub_temp_vreg.is_some() {
            return 2;
        }
        // This op is shiftable VALU: ALAP uses max delay (8 = shifted to 8 ALU ops)
        if self_is_shiftable {
            return 8;
        }
        // Default: 1 cycle latency
        1
    }
}

// ============================================================================
// AUTHORITATIVE LIVENESS MODEL DOCUMENTATION
// ============================================================================
//
// This is the ONE SOURCE OF TRUTH for how we represent VReg liveness.
// All other code should reference this documentation.
//
// ## ISA Hardware Semantics (from problem.py simulator)
//
// Within a single VLIW cycle:
//   1. ALL reads happen FIRST (from previous cycle's state)
//   2. ALL writes happen SECOND (results visible next cycle)
//
// This means:
//   - A VReg last READ at cycle C can share storage with a VReg WRITTEN at C
//     (because the read completes before the write begins)
//   - Two VRegs both WRITTEN at cycle C CANNOT share storage
//     (both writes happen simultaneously, would corrupt each other)
//
// ## Interval Representation: Closed [start, end]
//
// We use CLOSED intervals where `end` is the last cycle the VReg is ACCESSED:
//
//   LiveInterval { start: D, end: R }
//     - start (D) = cycle when VReg is WRITTEN (definition)
//     - end (R)   = cycle when VReg is last READ
//
// For a LIVE VReg (has users): [def_cycle, last_read_cycle]
// For a DEAD VReg (no users):  [def_cycle, def_cycle + 1]  ← NOT [def_cycle, def_cycle]!
//
// ## Why Dead VRegs Get end = def + 1
//
// A dead VReg is written but never read. The "+1 extension" is NOT a hack—
// it precisely models the ISA:
//
//   - Live VReg: "I'm read at cycle R, so my storage is free AFTER R"
//     → Can be reused starting at cycle R (same cycle, after the read)
//
//   - Dead VReg: "I'm written at cycle D but never read"
//     → Storage occupied during cycle D for the write
//     → Can be reused starting at cycle D+1 (after write completes)
//
// Without the +1 extension, two dead VRegs defined at the same cycle would
// appear to be reusable (both [D, D]), but they both WRITE at cycle D!
//
// ## The Single Expiration Rule: `end <= start`
//
// When allocating a new VReg at cycle S, expire old VRegs where `end <= S`:
//
//   Case 1: Same-cycle reuse (A read at C, B written at C)
//     A = [0, 5], B = [5, ...]
//     Check: 5 <= 5 ✓ → A expires, B reuses A's address
//     CORRECT: Hardware allows this (read before write)
//
//   Case 2: Same-cycle write conflict (A written at C, B written at C)
//     A = [5, 6] (dead, +1 extended), B = [5, ...]
//     Check: 6 <= 5 ✗ → A does NOT expire, B gets different address
//     CORRECT: Hardware forbids this (both write simultaneously)
//
// This single rule handles BOTH cases correctly!
//
// ## Overlap Check: `start < other.end && other.start < end`
//
// Two intervals overlap iff they share at least one cycle:
//   [0, 5] and [5, 10]: 0 < 10 ✓ AND 5 < 5 ✗ → NO overlap (adjacent)
//   [0, 5] and [4, 10]: 0 < 10 ✓ AND 4 < 5 ✓ → overlap at cycle 4-5
//
// ============================================================================

/// Live interval representing when a VReg occupies storage.
///
/// # Representation: Closed Interval [start, end]
///
/// - `start`: cycle when the VReg is WRITTEN (definition cycle)
/// - `end`: cycle when the VReg is last ACCESSED
///
/// # Construction (see `address_assign::compute_live_intervals`)
///
/// | VReg Type | Interval | Meaning |
/// |-----------|----------|---------|
/// | Live (has readers) | `[def, last_read]` | Read at `last_read`, reusable at same cycle |
/// | Dead (no readers)  | `[def, def + 1]`   | Written at `def`, reusable at `def + 1` |
///
/// # Why This Representation?
///
/// The hardware executes reads-before-writes within a cycle. Our interval
/// representation + expiration rule `end <= start` precisely models this:
///
/// - Same-cycle reuse: VReg A read at 5, VReg B written at 5
///   → A=[0,5], B=[5,...], check 5<=5 ✓, reuse allowed (CORRECT)
///
/// - Write conflict: VReg A written at 5 (dead), VReg B written at 5
///   → A=[5,6], B=[5,...], check 6<=5 ✗, no reuse (CORRECT)
///
/// See the module-level documentation above for full details.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LiveInterval {
    /// Cycle when the VReg is defined (written).
    pub start: i32,
    /// Cycle when the VReg is last accessed.
    /// For live VRegs: last read cycle.
    /// For dead VRegs: def_cycle + 1 (to prevent same-cycle write conflicts).
    pub end: i32,
}

impl LiveInterval {
    /// Create a new closed interval [start, end].
    ///
    /// IMPORTANT: For dead VRegs (never read), pass `end = start + 1`.
    /// See module documentation for why this is correct.
    pub fn new(start: i32, end: i32) -> Self {
        Self { start, end }
    }

    /// Check if this interval overlaps with another.
    ///
    /// Uses strict inequality on BOTH sides, which means adjacent intervals
    /// like [0,5] and [5,10] do NOT overlap. This enables same-cycle reuse:
    /// a VReg last read at cycle 5 can share with a VReg defined at cycle 5.
    ///
    /// See module documentation: "Overlap Check" section.
    pub fn overlaps(&self, other: &LiveInterval) -> bool {
        // Adjacent intervals [0,5] and [5,10]: 0 < 10 ✓ but 5 < 5 ✗ → no overlap
        self.start < other.end && other.start < self.end
    }

    /// Check if this interval contains a specific cycle (inclusive on both ends).
    pub fn contains(&self, cycle: i32) -> bool {
        self.start <= cycle && cycle <= self.end
    }

    /// Get the length of the interval in cycles.
    ///
    /// For closed interval [start, end]: length = end - start + 1.
    pub fn len(&self) -> i32 {
        self.end - self.start + 1
    }
}

// ============================================================================
// Usage Accounting
// ============================================================================

/// Statistics about scratch space usage for a single category.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct UsageStats {
    /// Total number of VRegs in this category
    pub vreg_count: usize,
    /// Total words allocated (scalars count as 1, vectors as 8)
    pub total_words: usize,
    /// Number of scalar VRegs
    pub scalar_count: usize,
    /// Number of vector VRegs
    pub vector_count: usize,
    /// Peak words live at any cycle
    pub peak_words: usize,
}

impl UsageStats {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a VReg's contribution
    pub fn add_vreg(&mut self, vreg: VReg) {
        self.vreg_count += 1;
        self.total_words += vreg.size();
        if vreg.is_scalar() {
            self.scalar_count += 1;
        } else {
            self.vector_count += 1;
        }
    }
}

/// Usage statistics by round.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RoundUsage {
    /// Round number (or -1 for init/finalization)
    pub round: i32,
    /// Stats for this round
    pub stats: UsageStats,
}

/// Usage statistics by variable type.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct VarTypeUsage {
    /// Variable type
    pub var_type: String,
    /// Stats for this type
    pub stats: UsageStats,
}

/// Bivariate usage statistics (round × var_type).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BivariateUsage {
    /// Round number (or -1 for init)
    pub round: i32,
    /// Variable type
    pub var_type: String,
    /// Stats for this combination
    pub stats: UsageStats,
}

/// Complete usage accounting for a scheduled kernel.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct UsageAccounting {
    /// Total VRegs
    pub total_vregs: usize,
    /// Total words allocated
    pub total_words: usize,
    /// Peak words live at any cycle (from address assignment)
    pub peak_live_words: usize,
    /// Breakdown by round
    pub by_round: Vec<RoundUsage>,
    /// Breakdown by variable type
    pub by_var_type: Vec<VarTypeUsage>,
    /// Bivariate breakdown (round × var_type)
    pub bivariate: Vec<BivariateUsage>,
}

impl UsageAccounting {
    /// Compute usage accounting from scheduled SSA ops.
    pub fn from_ssa_ops(ops: &[SsaOp]) -> Self {
        use std::collections::HashMap;

        let mut total_vregs = 0usize;
        let mut total_words = 0usize;

        // Maps for aggregation
        let mut by_round_map: HashMap<i32, UsageStats> = HashMap::new();
        let mut by_type_map: HashMap<VarType, UsageStats> = HashMap::new();
        let mut bivariate_map: HashMap<(i32, VarType), UsageStats> = HashMap::new();

        // Track seen VRegs to avoid double-counting
        let mut seen_vregs: HashSet<VReg> = HashSet::new();

        for op in ops {
            // Only count ops that define a VReg
            if let Some(dest) = op.dest {
                // Skip if already seen (CSE can create multiple refs to same VReg)
                if seen_vregs.contains(&dest) {
                    continue;
                }
                seen_vregs.insert(dest);

                total_vregs += 1;
                total_words += dest.size();

                // Determine round (-1 for init/finalization)
                let round = op
                    .source_info
                    .as_ref()
                    .and_then(|s| s.round)
                    .map(|r| r as i32)
                    .unwrap_or(-1);

                // Determine var_type (default to Temporary if not set)
                let var_type = op.var_type.unwrap_or(VarType::Temporary);

                // Aggregate by round
                by_round_map
                    .entry(round)
                    .or_insert_with(UsageStats::new)
                    .add_vreg(dest);

                // Aggregate by var_type
                by_type_map
                    .entry(var_type)
                    .or_insert_with(UsageStats::new)
                    .add_vreg(dest);

                // Aggregate bivariate
                bivariate_map
                    .entry((round, var_type))
                    .or_insert_with(UsageStats::new)
                    .add_vreg(dest);
            }
        }

        // Convert maps to sorted vectors
        let mut by_round: Vec<RoundUsage> = by_round_map
            .into_iter()
            .map(|(round, stats)| RoundUsage { round, stats })
            .collect();
        by_round.sort_by_key(|r| r.round);

        let mut by_var_type: Vec<VarTypeUsage> = by_type_map
            .into_iter()
            .map(|(var_type, stats)| VarTypeUsage {
                var_type: var_type.as_str().to_string(),
                stats,
            })
            .collect();
        by_var_type.sort_by(|a, b| a.var_type.cmp(&b.var_type));

        let mut bivariate: Vec<BivariateUsage> = bivariate_map
            .into_iter()
            .map(|((round, var_type), stats)| BivariateUsage {
                round,
                var_type: var_type.as_str().to_string(),
                stats,
            })
            .collect();
        bivariate.sort_by(|a, b| {
            a.round
                .cmp(&b.round)
                .then_with(|| a.var_type.cmp(&b.var_type))
        });

        UsageAccounting {
            total_vregs,
            total_words,
            peak_live_words: 0, // Will be set separately
            by_round,
            by_var_type,
            bivariate,
        }
    }

    /// Set peak live words from address assignment result.
    pub fn set_peak_live(&mut self, peak: usize) {
        self.peak_live_words = peak;
    }
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
        assert!(encodings.contains(&ConstEncoding::Add {
            a_value: 2,
            b_value: 2
        }));
        assert!(encodings.contains(&ConstEncoding::Mul {
            a_value: 2,
            b_value: 2
        }));
        assert!(encodings.contains(&ConstEncoding::Shift {
            base_value: 1,
            shift_amount: 2
        }));
        assert!(encodings.contains(&ConstEncoding::Shift {
            base_value: 2,
            shift_amount: 1
        }));
    }

    #[test]
    fn test_compute_encodings_240() {
        let encodings = compute_const_encodings(240);
        assert!(encodings.contains(&ConstEncoding::Load));
        // 240 = 120 + 120
        assert!(encodings.contains(&ConstEncoding::Add {
            a_value: 120,
            b_value: 120
        }));
        // 240 = 128 + 112
        assert!(encodings.contains(&ConstEncoding::Add {
            a_value: 112,
            b_value: 128
        }));
        // 240 = 15 << 4
        assert!(encodings.contains(&ConstEncoding::Shift {
            base_value: 15,
            shift_amount: 4
        }));
        // 240 = 15 * 16
        assert!(encodings.contains(&ConstEncoding::Mul {
            a_value: 15,
            b_value: 16
        }));
    }

    #[test]
    fn test_encoding_dependencies() {
        let load = ConstEncoding::Load;
        assert!(load.dependencies().is_empty());

        let add_imm = ConstEncoding::AddImm { value: 42 };
        assert_eq!(add_imm.dependencies(), Vec::<i64>::new()); // No dependencies - address 0 always contains 0

        let add = ConstEncoding::Add {
            a_value: 10,
            b_value: 20,
        };
        assert_eq!(add.dependencies(), vec![10, 20]);

        let shift = ConstEncoding::Shift {
            base_value: 5,
            shift_amount: 3,
        };
        assert_eq!(shift.dependencies(), vec![5, 3]);
    }

    #[test]
    fn test_encoding_is_ready() {
        let mut available: HashSet<i64> = HashSet::new();

        let load = ConstEncoding::Load;
        assert!(load.is_ready(&available)); // Load always ready

        let add_imm = ConstEncoding::AddImm { value: 42 };
        assert!(add_imm.is_ready(&available)); // Always ready - address 0 always contains 0

        let add = ConstEncoding::Add {
            a_value: 10,
            b_value: 20,
        };
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

        // Initially Load and AddImm are both available (AddImm can use address 0 directly)
        // AddImm is preferred over Load (flow slot vs load slot)
        let best = pc.best_available_encoding(&available);
        assert_eq!(best, Some(&ConstEncoding::AddImm { value: 240 }));

        // Add 120 - now ALU Add is preferred (ALU > AddImm > Load)
        available.insert(120);
        let best = pc.best_available_encoding(&available);
        assert_eq!(
            best,
            Some(&ConstEncoding::Add {
                a_value: 120,
                b_value: 120
            })
        );
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
        let mut value_to_reg: std::collections::HashMap<i64, i64> =
            std::collections::HashMap::new();
        value_to_reg.insert(10, 50); // value 10 is in register 50
        value_to_reg.insert(20, 60); // value 20 is in register 60

        let encoding = ConstEncoding::Add {
            a_value: 10,
            b_value: 20,
        };
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
        let value_to_reg: std::collections::HashMap<i64, i64> = std::collections::HashMap::new();
        // Note: value_to_reg can be empty - AddImm uses address 0 directly (always contains 0)

        let encoding = ConstEncoding::AddImm { value: 42 };
        let op = resolve_pseudo_const(&pc, &encoding, &value_to_reg, 1000);
        assert!(op.is_some());

        let op = op.unwrap();
        assert_eq!(op.engine, Engine::Flow);
        assert!(op.reads.contains(&0)); // Reads from address 0 (but no precedence constraint needed - always initialized)
        assert!(op.writes.contains(&100));
        assert_eq!(op.produces_constant, Some(42));
    }

    #[test]
    fn test_add_imm_encoding_engine() {
        let add_imm = ConstEncoding::AddImm { value: 123 };
        assert_eq!(add_imm.engine(), Engine::Flow);

        let load = ConstEncoding::Load;
        assert_eq!(load.engine(), Engine::Load);

        let add = ConstEncoding::Add {
            a_value: 1,
            b_value: 2,
        };
        assert_eq!(add.engine(), Engine::Alu);
    }

    // ========================================================================
    // VReg and SSA Tests
    // ========================================================================

    #[test]
    fn test_vreg_size() {
        assert_eq!(VReg::scalar(0).size(), 1);
        assert_eq!(VReg::vector(0).size(), 8);
    }

    #[test]
    fn test_vreg_equality() {
        let v1 = VReg::scalar(5);
        let v2 = VReg::scalar(5);
        let v3 = VReg::scalar(6);
        let v4 = VReg::vector(5);
        assert_eq!(v1, v2);
        assert_ne!(v1, v3);
        assert_ne!(v1, v4); // Same ID but different width
    }

    #[test]
    fn test_vreg_is_scalar_vector() {
        let s = VReg::scalar(0);
        let v = VReg::vector(0);
        assert!(s.is_scalar());
        assert!(!s.is_vector());
        assert!(!v.is_scalar());
        assert!(v.is_vector());
    }

    #[test]
    fn test_operand_vreg() {
        let op = Operand::scalar(5);
        assert!(op.as_vreg().is_some());
        assert!(op.as_imm().is_none());
        assert_eq!(op.as_vreg().unwrap(), VReg::scalar(5));
    }

    #[test]
    fn test_operand_imm() {
        let op = Operand::imm(42);
        assert!(op.as_imm().is_some());
        assert!(op.as_vreg().is_none());
        assert_eq!(op.as_imm().unwrap(), 42);
    }

    #[test]
    fn test_ssa_op_reads() {
        let op = SsaOp::new(
            0,
            Engine::Alu,
            Opcode::Add,
            Some(VReg::scalar(2)),
            vec![Operand::scalar(0), Operand::scalar(1)],
        );
        let reads: Vec<_> = op.reads().collect();
        assert_eq!(reads.len(), 2);
        assert!(reads.contains(&VReg::scalar(0)));
        assert!(reads.contains(&VReg::scalar(1)));
    }

    #[test]
    fn test_ssa_op_writes() {
        let op = SsaOp::new(
            0,
            Engine::Alu,
            Opcode::Add,
            Some(VReg::scalar(2)),
            vec![Operand::scalar(0), Operand::scalar(1)],
        );
        assert_eq!(op.writes(), Some(VReg::scalar(2)));
    }

    #[test]
    fn test_ssa_op_is_vector() {
        let scalar_op = SsaOp::new(0, Engine::Alu, Opcode::Add, Some(VReg::scalar(0)), vec![]);
        let vector_op = SsaOp::new(1, Engine::Valu, Opcode::Add, Some(VReg::vector(0)), vec![]);
        assert!(!scalar_op.is_vector());
        assert!(vector_op.is_vector());
    }

    // ========================================================================
    // LiveInterval Tests - CLOSED INTERVAL [start, end]
    //
    // See module documentation above for the authoritative explanation of
    // why we use closed intervals and how the expiration rule works.
    // ========================================================================

    #[test]
    fn test_live_interval_overlaps() {
        // CLOSED intervals: [start, end] where both endpoints are inclusive.
        // [0, 5] covers cycles 0,1,2,3,4,5
        // [3, 8] covers cycles 3,4,5,6,7,8
        // [6, 10] covers cycles 6,7,8,9,10
        let i1 = LiveInterval::new(0, 5);
        let i2 = LiveInterval::new(3, 8);
        let i3 = LiveInterval::new(6, 10);
        assert!(i1.overlaps(&i2)); // cycles 3,4,5 overlap
        assert!(i2.overlaps(&i3)); // cycles 6,7,8 overlap
        assert!(!i1.overlaps(&i3)); // no overlap: [0,5] and [6,10] are disjoint
    }

    #[test]
    fn test_live_interval_adjacent_no_overlap() {
        // SAME-CYCLE REUSE: This is the critical test case.
        //
        // VReg A: last READ at cycle 5 → interval [0, 5]
        // VReg B: WRITTEN at cycle 5 → interval [5, ...]
        //
        // These are ADJACENT (share endpoint 5) but do NOT overlap because
        // our overlap check uses STRICT inequality: `start < other.end`
        //
        // Check: 0 < 10 ✓ AND 5 < 5 ✗ → NO overlap
        //
        // This correctly models the ISA: reads happen before writes in a cycle,
        // so A's storage is free after the read, before B's write begins.
        let i1 = LiveInterval::new(0, 5); // A: last read at cycle 5
        let i2 = LiveInterval::new(5, 10); // B: defined at cycle 5
        assert!(
            !i1.overlaps(&i2),
            "Adjacent intervals [0,5] and [5,10] should NOT overlap (same-cycle reuse)"
        );
        assert!(!i2.overlaps(&i1), "Overlap check should be symmetric");
    }

    #[test]
    fn test_live_interval_gap_no_overlap() {
        // Intervals with a gap: [0, 4] and [6, 10] have cycle 5 gap
        let i1 = LiveInterval::new(0, 4);
        let i2 = LiveInterval::new(6, 10);
        assert!(!i1.overlaps(&i2));
        assert!(!i2.overlaps(&i1));
    }

    #[test]
    fn test_live_interval_dead_vreg_no_same_cycle_reuse() {
        // DEAD VREG WRITE CONFLICT: Critical test case.
        //
        // VReg A: WRITTEN at cycle 5, never read (dead) → interval [5, 6] (with +1 extension)
        // VReg B: WRITTEN at cycle 5 → interval [5, ...]
        //
        // These MUST overlap because both write at cycle 5!
        // Check: 5 < 10 ✓ AND 5 < 6 ✓ → OVERLAP
        //
        // The +1 extension for dead VRegs ensures this works correctly.
        let i1 = LiveInterval::new(5, 6); // A: dead VReg written at 5 (end = 5+1)
        let i2 = LiveInterval::new(5, 10); // B: written at 5
        assert!(
            i1.overlaps(&i2),
            "Dead VReg [5,6] MUST overlap with [5,10] (both write at cycle 5)"
        );
    }

    #[test]
    fn test_live_interval_dead_vreg_next_cycle_reuse() {
        // Dead VReg at cycle 5 can be reused at cycle 6.
        //
        // VReg A: WRITTEN at cycle 5, never read → [5, 6]
        // VReg B: WRITTEN at cycle 6 → [6, ...]
        //
        // Check: 5 < 10 ✓ AND 6 < 6 ✗ → NO overlap (correct!)
        let i1 = LiveInterval::new(5, 6); // A: dead VReg at cycle 5
        let i2 = LiveInterval::new(6, 10); // B: written at cycle 6
        assert!(
            !i1.overlaps(&i2),
            "Dead VReg [5,6] should NOT overlap with [6,10] (different write cycles)"
        );
    }

    #[test]
    fn test_live_interval_contains() {
        // Closed interval [2, 7] contains cycles 2,3,4,5,6,7 (both endpoints inclusive)
        let interval = LiveInterval::new(2, 7);
        assert!(!interval.contains(1));
        assert!(interval.contains(2)); // start is inclusive
        assert!(interval.contains(5));
        assert!(interval.contains(7)); // end is inclusive
        assert!(!interval.contains(8));
    }

    #[test]
    fn test_live_interval_len() {
        // Closed interval [3, 7] has length = 7 - 3 + 1 = 5 cycles: 3, 4, 5, 6, 7
        let interval = LiveInterval::new(3, 7);
        assert_eq!(interval.len(), 5);
    }

    // ========================================================================
    // unique_src_vregs Tests (Duplicate Operand Handling)
    // ========================================================================

    #[test]
    fn test_unique_src_vregs_distinct_operands() {
        // add(v0, v1) -> dest v2: should return [v0, v1]
        let op = SsaOp::new(
            0,
            Engine::Alu,
            Opcode::Add,
            Some(VReg::scalar(2)),
            vec![Operand::scalar(0), Operand::scalar(1)],
        );
        let unique = op.unique_src_vregs();
        assert_eq!(unique.len(), 2);
        assert!(unique.contains(&VReg::scalar(0)));
        assert!(unique.contains(&VReg::scalar(1)));
    }

    #[test]
    fn test_unique_src_vregs_duplicate_operands() {
        // add(v1, v1) -> dest v2: should return [v1] (not [v1, v1])
        // This is the key bug fix - duplicate operands should be deduplicated
        let op = SsaOp::new(
            0,
            Engine::Alu,
            Opcode::Add,
            Some(VReg::scalar(2)),
            vec![Operand::scalar(1), Operand::scalar(1)],
        );
        let unique = op.unique_src_vregs();
        assert_eq!(unique.len(), 1, "Duplicate operands should be deduplicated");
        assert_eq!(unique[0], VReg::scalar(1));
    }

    #[test]
    fn test_unique_src_vregs_with_immediates() {
        // add(v1, imm 5) -> dest v2: should return [v1]
        let op = SsaOp::new(
            0,
            Engine::Alu,
            Opcode::Add,
            Some(VReg::scalar(2)),
            vec![Operand::scalar(1), Operand::imm(5)],
        );
        let unique = op.unique_src_vregs();
        assert_eq!(unique.len(), 1);
        assert_eq!(unique[0], VReg::scalar(1));
    }

    #[test]
    fn test_unique_src_vregs_all_immediates() {
        // const 5 -> dest v0: no VReg sources
        let op = SsaOp::new(
            0,
            Engine::Load,
            Opcode::Const,
            Some(VReg::scalar(0)),
            vec![Operand::imm(5)],
        );
        let unique = op.unique_src_vregs();
        assert!(unique.is_empty());
    }

    #[test]
    fn test_unique_src_vregs_vector_duplicate() {
        // Vector add(v1, v1) where v1 is a vector VReg
        let op = SsaOp::new(
            0,
            Engine::Valu,
            Opcode::Add,
            Some(VReg::vector(2)),
            vec![Operand::vector(1), Operand::vector(1)],
        );
        let unique = op.unique_src_vregs();
        assert_eq!(
            unique.len(),
            1,
            "Vector duplicate operands should be deduplicated"
        );
        assert_eq!(unique[0], VReg::vector(1));
    }

    #[test]
    fn test_unique_src_vregs_three_operands_with_duplicates() {
        // multiply_add(v1, v1, v2) -> dest v3: should return [v1, v2]
        let op = SsaOp::new(
            0,
            Engine::Valu,
            Opcode::MultiplyAdd,
            Some(VReg::vector(3)),
            vec![Operand::vector(1), Operand::vector(1), Operand::vector(2)],
        );
        let unique = op.unique_src_vregs();
        assert_eq!(unique.len(), 2);
        assert!(unique.contains(&VReg::vector(1)));
        assert!(unique.contains(&VReg::vector(2)));
    }

    #[test]
    fn test_unique_src_vregs_preserves_order() {
        // Verify that unique_src_vregs preserves first-seen order
        let op = SsaOp::new(
            0,
            Engine::Valu,
            Opcode::MultiplyAdd,
            Some(VReg::vector(4)),
            vec![Operand::vector(3), Operand::vector(1), Operand::vector(3)],
        );
        let unique = op.unique_src_vregs();
        assert_eq!(unique.len(), 2);
        assert_eq!(unique[0], VReg::vector(3), "First unique VReg should be v3");
        assert_eq!(
            unique[1],
            VReg::vector(1),
            "Second unique VReg should be v1"
        );
    }

    // ========================================================================
    // PrecedenceOffset trait tests
    // ========================================================================

    #[test]
    fn test_precedence_offset_normal_op() {
        // A normal ALU op: offset is always 1 (self_is_shiftable=false for ALU)
        let op = SsaOp::new(
            0,
            Engine::Alu,
            Opcode::Add,
            Some(VReg::scalar(0)),
            vec![Operand::scalar(1), Operand::scalar(2)],
        );
        assert_eq!(op.precedence_offset_for_asap(false), 1);
        assert_eq!(op.precedence_offset_for_alap(false), 1);
        // Even if self_is_shiftable=true (hypothetical), shiftable path applies
        assert_eq!(op.precedence_offset_for_asap(true), 1);
        assert_eq!(op.precedence_offset_for_alap(true), 8);
    }

    #[test]
    fn test_precedence_offset_select_choice_fma() {
        // SelectChoice with sub_temp_vreg: ASAP uses min_delay=1 (vselect path),
        // ALAP uses max_delay=2 (FMA path). Same pattern as shiftable ops.
        let mut op = SsaOp::new(
            1,
            Engine::Flow,
            Opcode::SelectChoice,
            Some(VReg::vector(10)),
            vec![Operand::vector(1), Operand::vector(2), Operand::vector(3)],
        );
        op.sub_temp_vreg = Some(VReg::scalar(99));

        // ASAP: min_delay = 1 (vselect path); FMA delay enforced by CP-SAT model constraints
        assert_eq!(op.precedence_offset_for_asap(false), 1);
        // ALAP: max_delay = 2 (FMA path: sub + fma)
        assert_eq!(op.precedence_offset_for_alap(false), 2);
        // shiftable flag doesn't change SelectChoice offset
        assert_eq!(op.precedence_offset_for_asap(true), 1);
        assert_eq!(op.precedence_offset_for_alap(true), 2);
    }

    #[test]
    fn test_precedence_offset_shiftable() {
        // A shiftable VALU op: ASAP=1 (min delay), ALAP=8 (max delay when shifted)
        let op = SsaOp::new(
            2,
            Engine::Valu,
            Opcode::Add,
            Some(VReg::vector(5)),
            vec![Operand::vector(3), Operand::vector(4)],
        );
        // When self_is_shiftable=true: ASAP=1, ALAP=8
        assert_eq!(op.precedence_offset_for_asap(true), 1);
        assert_eq!(op.precedence_offset_for_alap(true), 8);
        // When self_is_shiftable=false: default offsets
        assert_eq!(op.precedence_offset_for_asap(false), 1);
        assert_eq!(op.precedence_offset_for_alap(false), 1);
    }
}
