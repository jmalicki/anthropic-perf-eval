//! Op emission helpers for clean instruction creation.
//!
//! This module provides helper functions for creating Op structs with proper
//! reads/writes tracking, eliminating the need for Python-side emission.

use arrayvec::ArrayVec;
use std::collections::{HashMap, HashSet};

use crate::types::{Engine, Op, Opcode, PseudoConst, SlotVal, MAX_SLOT_SIZE, VLEN};

/// Broadcast cache entry - tracks where a scalar value has been broadcast to
#[derive(Debug, Clone)]
pub struct BroadcastEntry {
    /// Destination vector base address
    pub dest: i64,
    /// Op ID of the broadcast operation
    pub op_id: usize,
}

/// Emission context - tracks op IDs, constants, and collects emitted ops
#[derive(Debug, Default)]
pub struct EmitContext {
    /// Next op ID to assign
    pub next_op_id: usize,
    /// Collected ops (unscheduled)
    pub ops: Vec<Op>,
    /// Maps addresses to their known constant values (for derivation)
    /// Updated when emit_const is called
    pub known_constants: HashMap<i64, i64>,
    /// Maps source scalar address -> broadcast entry
    /// Used to detect duplicate broadcasts from the same scalar
    pub broadcast_cache: HashMap<i64, BroadcastEntry>,

    // === Pseudo-constant support ===
    /// Next pseudo-const ID to assign
    pub next_pseudo_const_id: usize,
    /// Pending pseudo-constants (not yet resolved)
    pub pseudo_consts: Vec<PseudoConst>,
    /// Maps constant value -> pseudo-const ID (for linking consumers)
    pub value_to_pseudo_const_id: HashMap<i64, usize>,
    /// Maps constant value -> destination register
    pub value_to_register: HashMap<i64, i64>,
}

impl EmitContext {
    pub fn new() -> Self {
        EmitContext {
            next_op_id: 0,
            ops: Vec::new(),
            known_constants: HashMap::new(),
            broadcast_cache: HashMap::new(),
            next_pseudo_const_id: 0,
            pseudo_consts: Vec::new(),
            value_to_pseudo_const_id: HashMap::new(),
            value_to_register: HashMap::new(),
        }
    }

    /// Take all ops, resetting the context
    pub fn take_ops(&mut self) -> Vec<Op> {
        self.next_op_id = 0;
        self.known_constants.clear();
        self.broadcast_cache.clear();
        self.next_pseudo_const_id = 0;
        self.pseudo_consts.clear();
        self.value_to_pseudo_const_id.clear();
        self.value_to_register.clear();
        std::mem::take(&mut self.ops)
    }

    /// Take all ops and pseudo-consts, resetting the context
    pub fn take_ops_and_pseudo_consts(&mut self) -> (Vec<Op>, Vec<PseudoConst>) {
        self.next_op_id = 0;
        self.known_constants.clear();
        self.broadcast_cache.clear();
        self.next_pseudo_const_id = 0;
        self.value_to_pseudo_const_id.clear();
        self.value_to_register.clear();
        (std::mem::take(&mut self.ops), std::mem::take(&mut self.pseudo_consts))
    }

    /// Declare that a constant value is needed.
    ///
    /// Creates a PseudoConst that the scheduler will resolve to a concrete
    /// encoding (Load, Add, Mul, etc.) based on available resources.
    ///
    /// Returns the destination register where the value will be available.
    /// If the same value was already requested, returns the existing register.
    pub fn need_const(&mut self, value: i64, dest: i64) -> i64 {
        // Check if this value is already requested
        if let Some(&existing_dest) = self.value_to_register.get(&value) {
            return existing_dest;
        }

        // Create a new pseudo-const
        let pc = PseudoConst::new(self.next_pseudo_const_id, value, Some(dest));
        let pc_id = pc.id;
        self.pseudo_consts.push(pc);

        self.value_to_pseudo_const_id.insert(value, pc_id);
        self.value_to_register.insert(value, dest);
        self.next_pseudo_const_id += 1;

        // Also register in known_constants so ops can reference it
        self.known_constants.insert(dest, value);

        dest
    }

    /// Declare that a critical constant value is needed (will prefer Load encoding).
    ///
    /// Use this for constants on the critical path that need to be available ASAP.
    pub fn need_const_critical(&mut self, value: i64, dest: i64) -> i64 {
        // Check if this value is already requested
        if let Some(&existing_dest) = self.value_to_register.get(&value) {
            return existing_dest;
        }

        // Create a new critical pseudo-const
        let pc = PseudoConst::new_critical(self.next_pseudo_const_id, value, Some(dest));
        let pc_id = pc.id;
        self.pseudo_consts.push(pc);

        self.value_to_pseudo_const_id.insert(value, pc_id);
        self.value_to_register.insert(value, dest);
        self.next_pseudo_const_id += 1;

        // Also register in known_constants so ops can reference it
        self.known_constants.insert(dest, value);

        dest
    }

    /// Link an op as a consumer of a pseudo-const value.
    ///
    /// Call this after emitting an op that reads from a constant value
    /// to establish the dependency.
    pub fn link_const_consumer(&mut self, value: i64, op_id: usize) {
        if let Some(&pc_id) = self.value_to_pseudo_const_id.get(&value) {
            if let Some(pc) = self.pseudo_consts.iter_mut().find(|p| p.id == pc_id) {
                if !pc.consumer_op_ids.contains(&op_id) {
                    pc.consumer_op_ids.push(op_id);
                }
            }
        }
    }

    /// Get the register holding a constant value, if it was declared via need_const.
    pub fn get_const_register(&self, value: i64) -> Option<i64> {
        self.value_to_register.get(&value).copied()
    }

    /// Get number of ops
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Check if a scalar address has already been broadcast to a vector
    /// Returns the existing vector destination if found
    pub fn get_broadcast(&self, src: i64) -> Option<i64> {
        self.broadcast_cache.get(&src).map(|e| e.dest)
    }

    /// Get the constant value at an address, if known
    pub fn get_constant(&self, addr: i64) -> Option<i64> {
        self.known_constants.get(&addr).copied()
    }

    /// Emit an ALU operation: dest = src1 op src2
    pub fn emit_alu(&mut self, opcode: Opcode, dest: i64, src1: i64, src2: i64) {
        let op = Op {
            id: self.next_op_id,
            engine: Engine::Alu,
            slot: [
                SlotVal::Str(opcode.as_str().to_string()),
                SlotVal::Int(dest),
                SlotVal::Int(src1),
                SlotVal::Int(src2),
            ]
            .into_iter()
            .collect(),
            cycle: -1,
            dep_ids: HashSet::new(),
            reads: [src1, src2].into_iter().collect(),
            writes: [dest].into_iter().collect(),
            last_use_reads: HashSet::new(),
            produces_constant: None,
        };
        self.next_op_id += 1;
        self.ops.push(op);
    }

    /// Emit a VALU operation: dest[0..8] = src1[0..8] op src2[0..8]
    pub fn emit_valu(&mut self, opcode: Opcode, dest: i64, src1: i64, src2: i64) {
        let mut reads = HashSet::new();
        let mut writes = HashSet::new();
        for i in 0..VLEN as i64 {
            reads.insert(src1 + i);
            reads.insert(src2 + i);
            writes.insert(dest + i);
        }

        let op = Op {
            id: self.next_op_id,
            engine: Engine::Valu,
            slot: [
                SlotVal::Str(opcode.as_str().to_string()),
                SlotVal::Int(dest),
                SlotVal::Int(src1),
                SlotVal::Int(src2),
            ]
            .into_iter()
            .collect(),
            cycle: -1,
            dep_ids: HashSet::new(),
            reads,
            writes,
            last_use_reads: HashSet::new(),
            produces_constant: None,
        };
        self.next_op_id += 1;
        self.ops.push(op);
    }

    /// Emit a vbroadcast: dest[0..8] = scalar_src
    ///
    /// **Returns:** The actual vector destination used. If this scalar was already
    /// broadcast, returns the existing destination (CSE deduplication). Otherwise,
    /// returns the provided `dest`.
    ///
    /// Callers should use the returned address instead of their requested `dest`
    /// when the scalar has already been broadcast - this eliminates duplicate
    /// broadcast operations.
    ///
    /// If the source scalar has a known constant value, propagates that to
    /// `produces_constant`.
    pub fn emit_vbroadcast(&mut self, dest: i64, src: i64) -> i64 {
        // CSE: If this scalar was already broadcast, return existing destination
        if let Some(entry) = self.broadcast_cache.get(&src) {
            return entry.dest;
        }

        let mut writes = HashSet::new();
        for i in 0..VLEN as i64 {
            writes.insert(dest + i);
        }

        // Propagate constant info if source is a known constant
        let produces_constant = self.known_constants.get(&src).copied();

        let op = Op {
            id: self.next_op_id,
            engine: Engine::Valu,
            slot: [
                SlotVal::Str("vbroadcast".to_string()),
                SlotVal::Int(dest),
                SlotVal::Int(src),
            ]
            .into_iter()
            .collect(),
            cycle: -1,
            dep_ids: HashSet::new(),
            reads: [src].into_iter().collect(),
            writes,
            last_use_reads: HashSet::new(),
            produces_constant,
        };

        // Cache this broadcast for deduplication
        self.broadcast_cache.insert(src, BroadcastEntry {
            dest,
            op_id: self.next_op_id,
        });

        self.next_op_id += 1;
        self.ops.push(op);

        dest
    }

    /// Emit a vbroadcast without CSE - always emits to the specified destination.
    ///
    /// Use this when the destination matters (e.g., working temporaries that need
    /// the value in a specific location for subsequent operations to read from).
    ///
    /// Does NOT update the broadcast cache, so subsequent CSE-enabled broadcasts
    /// from the same source will still emit.
    pub fn emit_vbroadcast_force(&mut self, dest: i64, src: i64) {
        let mut writes = HashSet::new();
        for i in 0..VLEN as i64 {
            writes.insert(dest + i);
        }

        // Propagate constant info if source is a known constant
        let produces_constant = self.known_constants.get(&src).copied();

        let op = Op {
            id: self.next_op_id,
            engine: Engine::Valu,
            slot: [
                SlotVal::Str("vbroadcast".to_string()),
                SlotVal::Int(dest),
                SlotVal::Int(src),
            ]
            .into_iter()
            .collect(),
            cycle: -1,
            dep_ids: HashSet::new(),
            reads: [src].into_iter().collect(),
            writes,
            last_use_reads: HashSet::new(),
            produces_constant,
        };

        self.next_op_id += 1;
        self.ops.push(op);
    }

    /// Emit a multiply_add (FMA): dest[0..8] = src1[0..8] * src2[0..8] + src3[0..8]
    pub fn emit_fma(&mut self, dest: i64, src1: i64, src2: i64, src3: i64) {
        let mut reads = HashSet::new();
        let mut writes = HashSet::new();
        for i in 0..VLEN as i64 {
            reads.insert(src1 + i);
            reads.insert(src2 + i);
            reads.insert(src3 + i);
            writes.insert(dest + i);
        }

        let op = Op {
            id: self.next_op_id,
            engine: Engine::Valu,
            slot: [
                SlotVal::Str("multiply_add".to_string()),
                SlotVal::Int(dest),
                SlotVal::Int(src1),
                SlotVal::Int(src2),
                SlotVal::Int(src3),
            ]
            .into_iter()
            .collect(),
            cycle: -1,
            dep_ids: HashSet::new(),
            reads,
            writes,
            last_use_reads: HashSet::new(),
            produces_constant: None,
        };
        self.next_op_id += 1;
        self.ops.push(op);
    }

    /// Emit a const load: dest = immediate_value
    ///
    /// This annotates the op with `produces_constant = Some(value)` to enable
    /// late-binding optimization where a scheduler pass can replace the Load
    /// with an ALU derivation if slots are available.
    ///
    /// Also registers the constant in `known_constants` for derivation tracking.
    pub fn emit_const(&mut self, dest: i64, value: i64) {
        // Register this constant for potential derivation use
        self.known_constants.insert(dest, value);

        let op = Op {
            id: self.next_op_id,
            engine: Engine::Load,
            slot: [
                SlotVal::Str("const".to_string()),
                SlotVal::Int(dest),
                SlotVal::Int(value),
            ]
            .into_iter()
            .collect(),
            cycle: -1,
            dep_ids: HashSet::new(),
            reads: HashSet::new(),
            writes: [dest].into_iter().collect(),
            last_use_reads: HashSet::new(),
            produces_constant: Some(value),
        };
        self.next_op_id += 1;
        self.ops.push(op);
    }

    /// Emit a scalar load: dest = mem[addr]
    pub fn emit_load(&mut self, dest: i64, addr: i64) {
        let op = Op {
            id: self.next_op_id,
            engine: Engine::Load,
            slot: [
                SlotVal::Str("load".to_string()),
                SlotVal::Int(dest),
                SlotVal::Int(addr),
            ]
            .into_iter()
            .collect(),
            cycle: -1,
            dep_ids: HashSet::new(),
            reads: [addr].into_iter().collect(),
            writes: [dest].into_iter().collect(),
            last_use_reads: HashSet::new(),
            produces_constant: None,
        };
        self.next_op_id += 1;
        self.ops.push(op);
    }

    /// Emit a vector load: dest[0..8] = mem[addr..addr+8]
    pub fn emit_vload(&mut self, dest: i64, addr: i64) {
        let mut writes = HashSet::new();
        for i in 0..VLEN as i64 {
            writes.insert(dest + i);
        }

        let op = Op {
            id: self.next_op_id,
            engine: Engine::Load,
            slot: [
                SlotVal::Str("vload".to_string()),
                SlotVal::Int(dest),
                SlotVal::Int(addr),
            ]
            .into_iter()
            .collect(),
            cycle: -1,
            dep_ids: HashSet::new(),
            reads: [addr].into_iter().collect(),
            writes,
            last_use_reads: HashSet::new(),
            produces_constant: None,
        };
        self.next_op_id += 1;
        self.ops.push(op);
    }

    /// Emit a scalar store: mem[addr] = src
    pub fn emit_store(&mut self, addr: i64, src: i64) {
        let op = Op {
            id: self.next_op_id,
            engine: Engine::Store,
            slot: [
                SlotVal::Str("store".to_string()),
                SlotVal::Int(addr),
                SlotVal::Int(src),
            ]
            .into_iter()
            .collect(),
            cycle: -1,
            dep_ids: HashSet::new(),
            reads: [addr, src].into_iter().collect(),
            writes: HashSet::new(),
            last_use_reads: HashSet::new(),
            produces_constant: None,
        };
        self.next_op_id += 1;
        self.ops.push(op);
    }

    /// Emit a vector store: mem[addr..addr+8] = src[0..8]
    pub fn emit_vstore(&mut self, addr: i64, src: i64) {
        let mut reads: HashSet<i64> = [addr].into_iter().collect();
        for i in 0..VLEN as i64 {
            reads.insert(src + i);
        }

        let op = Op {
            id: self.next_op_id,
            engine: Engine::Store,
            slot: [
                SlotVal::Str("vstore".to_string()),
                SlotVal::Int(addr),
                SlotVal::Int(src),
            ]
            .into_iter()
            .collect(),
            cycle: -1,
            dep_ids: HashSet::new(),
            reads,
            writes: HashSet::new(),
            last_use_reads: HashSet::new(),
            produces_constant: None,
        };
        self.next_op_id += 1;
        self.ops.push(op);
    }

    /// Emit a pause instruction
    pub fn emit_pause(&mut self) {
        let op = Op {
            id: self.next_op_id,
            engine: Engine::Flow,
            slot: [SlotVal::Str("pause".to_string())].into_iter().collect(),
            cycle: -1,
            dep_ids: HashSet::new(),
            reads: HashSet::new(),
            writes: HashSet::new(),
            last_use_reads: HashSet::new(),
            produces_constant: None,
        };
        self.next_op_id += 1;
        self.ops.push(op);
    }
}

/// Helper trait for creating ArrayVec from iterator
pub trait IntoSlot {
    fn into_slot(self) -> ArrayVec<SlotVal, MAX_SLOT_SIZE>;
}

impl<I> IntoSlot for I
where
    I: IntoIterator<Item = SlotVal>,
{
    fn into_slot(self) -> ArrayVec<SlotVal, MAX_SLOT_SIZE> {
        self.into_iter().collect()
    }
}
