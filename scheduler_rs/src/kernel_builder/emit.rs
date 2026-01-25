//! Op emission helpers for clean instruction creation.
//!
//! This module provides helper functions for creating Op structs with proper
//! reads/writes tracking, eliminating the need for Python-side emission.

use arrayvec::ArrayVec;
use std::collections::HashSet;

use crate::types::{Engine, Op, Opcode, SlotVal, MAX_SLOT_SIZE, VLEN};

/// Emission context - tracks op IDs and collects emitted ops
#[derive(Debug, Default)]
pub struct EmitContext {
    /// Next op ID to assign
    pub next_op_id: usize,
    /// Collected ops (unscheduled)
    pub ops: Vec<Op>,
}

impl EmitContext {
    pub fn new() -> Self {
        EmitContext {
            next_op_id: 0,
            ops: Vec::new(),
        }
    }

    /// Take all ops, resetting the context
    pub fn take_ops(&mut self) -> Vec<Op> {
        self.next_op_id = 0;
        std::mem::take(&mut self.ops)
    }

    /// Get number of ops
    pub fn len(&self) -> usize {
        self.ops.len()
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
        };
        self.next_op_id += 1;
        self.ops.push(op);
    }

    /// Emit a vbroadcast: dest[0..8] = scalar_src
    pub fn emit_vbroadcast(&mut self, dest: i64, src: i64) {
        let mut writes = HashSet::new();
        for i in 0..VLEN as i64 {
            writes.insert(dest + i);
        }

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
        };
        self.next_op_id += 1;
        self.ops.push(op);
    }

    /// Emit a const load: dest = immediate_value
    pub fn emit_const(&mut self, dest: i64, value: i64) {
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
