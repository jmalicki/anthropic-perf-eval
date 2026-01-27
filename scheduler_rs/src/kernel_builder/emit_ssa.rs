//! SSA-based op emission helpers.
//!
//! This module provides helpers for creating SsaOps using virtual registers (VRegs)
//! instead of concrete scratch addresses. This eliminates false WAW and WAR
//! dependencies, allowing more scheduling flexibility.
//!
//! After scheduling, VRegs are mapped to physical addresses via linear scan allocation.

use std::collections::HashMap;

use crate::types::{EmissionPhase, Engine, LiveInterval, Opcode, Operand, SsaOp, VarSource, VarType, VReg};

/// SSA Emission context - tracks VReg IDs and collects emitted ops.
///
/// Unlike EmitContext which uses concrete addresses, SsaEmitContext uses
/// virtual registers that are assigned physical addresses after scheduling.
#[derive(Debug)]
pub struct SsaEmitContext {
    /// Next op ID to assign
    pub next_op_id: usize,
    /// Next VReg ID to assign
    pub next_vreg_id: u32,
    /// Collected ops (unscheduled)
    pub ops: Vec<SsaOp>,
    /// Maps VReg -> defining op ID (for RAW dependency computation)
    pub vreg_def: HashMap<VReg, usize>,
    /// Maps VReg -> constant value (if known)
    pub vreg_constants: HashMap<VReg, i64>,
    /// Broadcast cache: scalar VReg -> vector VReg (CSE deduplication)
    pub broadcast_cache: HashMap<VReg, VReg>,

    // =========================================================================
    // Context Tracking for Usage Accounting
    // =========================================================================
    /// Current round number (0-15), None during init/finalization
    pub current_round: Option<usize>,
    /// Current tree depth (0-9)
    pub current_tree_depth: Option<usize>,
    /// Current iteration index (for unrolled loops)
    pub current_iter: Option<usize>,
    /// Current group index
    pub current_group: Option<usize>,
    /// Current emission phase
    pub current_phase: EmissionPhase,
    /// VarType to use for the next emitted op (if set)
    pub next_var_type: Option<VarType>,
}

impl Default for SsaEmitContext {
    fn default() -> Self {
        Self::new()
    }
}

impl SsaEmitContext {
    pub fn new() -> Self {
        Self {
            next_op_id: 0,
            next_vreg_id: 0,
            ops: Vec::new(),
            vreg_def: HashMap::new(),
            vreg_constants: HashMap::new(),
            broadcast_cache: HashMap::new(),
            // Context tracking
            current_round: None,
            current_tree_depth: None,
            current_iter: None,
            current_group: None,
            current_phase: EmissionPhase::Init,
            next_var_type: None,
        }
    }

    /// Allocate a fresh scalar VReg
    pub fn fresh_scalar(&mut self) -> VReg {
        let vreg = VReg::scalar(self.next_vreg_id);
        self.next_vreg_id += 1;
        vreg
    }

    /// Allocate a fresh vector VReg
    pub fn fresh_vector(&mut self) -> VReg {
        let vreg = VReg::vector(self.next_vreg_id);
        self.next_vreg_id += 1;
        vreg
    }

    /// Get the next op ID without incrementing
    pub fn peek_op_id(&self) -> usize {
        self.next_op_id
    }

    /// Get number of ops
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Check if context is empty
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Take all ops, resetting the context
    pub fn take_ops(&mut self) -> Vec<SsaOp> {
        self.next_op_id = 0;
        self.next_vreg_id = 0;
        self.vreg_def.clear();
        self.vreg_constants.clear();
        self.broadcast_cache.clear();
        // Reset context tracking
        self.current_round = None;
        self.current_tree_depth = None;
        self.current_iter = None;
        self.current_group = None;
        self.current_phase = EmissionPhase::Init;
        self.next_var_type = None;
        std::mem::take(&mut self.ops)
    }

    // =========================================================================
    // Context Tracking Methods
    // =========================================================================

    /// Set the emission phase
    pub fn set_phase(&mut self, phase: EmissionPhase) {
        self.current_phase = phase;
    }

    /// Set round context for a specific round
    pub fn set_round_context(&mut self, round: usize, tree_depth: usize) {
        self.current_round = Some(round);
        self.current_tree_depth = Some(tree_depth);
    }

    /// Clear round context (for init/finalization)
    pub fn clear_round_context(&mut self) {
        self.current_round = None;
        self.current_tree_depth = None;
        self.current_iter = None;
        self.current_group = None;
    }

    /// Set iteration and group context
    pub fn set_iter_group(&mut self, iter: usize, group: usize) {
        self.current_iter = Some(iter);
        self.current_group = Some(group);
    }

    /// Clear iteration and group context
    pub fn clear_iter_group(&mut self) {
        self.current_iter = None;
        self.current_group = None;
    }

    /// Set the VarType for the next emitted op
    pub fn set_next_var_type(&mut self, var_type: VarType) {
        self.next_var_type = Some(var_type);
    }

    /// Build current VarSource from context
    fn current_var_source(&self) -> VarSource {
        VarSource {
            round: self.current_round,
            tree_depth: self.current_tree_depth,
            iter: self.current_iter,
            group: self.current_group,
            phase: self.current_phase,
        }
    }

    /// Record that a VReg is defined by a specific op
    fn record_def(&mut self, vreg: VReg, op_id: usize) {
        self.vreg_def.insert(vreg, op_id);
    }

    /// Get the op ID that defines a VReg
    pub fn get_def(&self, vreg: VReg) -> Option<usize> {
        self.vreg_def.get(&vreg).copied()
    }

    /// Get the constant value of a VReg, if known
    pub fn get_constant(&self, vreg: VReg) -> Option<i64> {
        self.vreg_constants.get(&vreg).copied()
    }

    /// Emit an op and return its ID
    #[track_caller]
    fn emit(&mut self, mut op: SsaOp) -> usize {
        let op_id = self.next_op_id;
        op.id = op_id;
        op.source = Some(std::panic::Location::caller());

        // Capture current context for usage accounting
        op.source_info = Some(self.current_var_source());
        if let Some(var_type) = self.next_var_type.take() {
            op.var_type = Some(var_type);
        }

        self.next_op_id += 1;

        // Record VReg definition
        if let Some(dest) = op.dest {
            self.record_def(dest, op_id);
        }

        self.ops.push(op);
        op_id
    }

    // ========================================================================
    // ALU Operations
    // ========================================================================

    /// Emit ALU operation: result = src1 op src2
    /// Returns the destination VReg
    #[track_caller]
    pub fn emit_alu(&mut self, opcode: Opcode, src1: VReg, src2: VReg) -> VReg {
        let dest = self.fresh_scalar();
        let op = SsaOp::new(
            0, // will be set by emit()
            Engine::Alu,
            opcode,
            Some(dest),
            vec![Operand::VReg(src1), Operand::VReg(src2)],
        );
        self.emit(op);
        dest
    }

    /// Emit ALU operation with immediate: result = src op imm
    #[track_caller]
    pub fn emit_alu_imm(&mut self, opcode: Opcode, src: VReg, imm: i64) -> VReg {
        let dest = self.fresh_scalar();
        let op = SsaOp::new(
            0,
            Engine::Alu,
            opcode,
            Some(dest),
            vec![Operand::VReg(src), Operand::Imm(imm)],
        );
        self.emit(op);
        dest
    }

    // ========================================================================
    // VALU Operations
    // ========================================================================

    /// Emit VALU operation: result[0..8] = src1[0..8] op src2[0..8]
    /// Returns the destination vector VReg
    #[track_caller]
    pub fn emit_valu(&mut self, opcode: Opcode, src1: VReg, src2: VReg) -> VReg {
        assert!(src1.is_vector(), "VALU src1 must be vector");
        assert!(src2.is_vector(), "VALU src2 must be vector");

        let dest = self.fresh_vector();
        let op = SsaOp::new(
            0,
            Engine::Valu,
            opcode,
            Some(dest),
            vec![Operand::VReg(src1), Operand::VReg(src2)],
        );
        self.emit(op);
        dest
    }

    /// Emit vbroadcast: result[0..8] = scalar
    /// Returns the destination vector VReg.
    ///
    /// Uses CSE deduplication - if the same scalar was already broadcast,
    /// returns the existing vector VReg.
    #[track_caller]
    pub fn emit_vbroadcast(&mut self, src: VReg) -> VReg {
        assert!(src.is_scalar(), "vbroadcast src must be scalar");

        // CSE: check if already broadcast
        if let Some(&existing) = self.broadcast_cache.get(&src) {
            return existing;
        }

        let dest = self.fresh_vector();

        // Propagate constant info
        let produces_constant = self.vreg_constants.get(&src).copied();

        let mut op = SsaOp::new(
            0,
            Engine::Valu,
            Opcode::Vbroadcast,
            Some(dest),
            vec![Operand::VReg(src)],
        );
        op.produces_constant = produces_constant;

        self.emit(op);
        self.broadcast_cache.insert(src, dest);
        dest
    }

    /// Emit vbroadcast without CSE - always creates a new broadcast.
    /// Use when the destination matters for subsequent operations.
    #[track_caller]
    pub fn emit_vbroadcast_force(&mut self, src: VReg) -> VReg {
        assert!(src.is_scalar(), "vbroadcast src must be scalar");

        let dest = self.fresh_vector();
        let produces_constant = self.vreg_constants.get(&src).copied();

        let mut op = SsaOp::new(
            0,
            Engine::Valu,
            Opcode::Vbroadcast,
            Some(dest),
            vec![Operand::VReg(src)],
        );
        op.produces_constant = produces_constant;

        self.emit(op);
        dest
    }

    /// Emit multiply_add (FMA): result[0..8] = src1[0..8] * src2[0..8] + src3[0..8]
    #[track_caller]
    pub fn emit_fma(&mut self, src1: VReg, src2: VReg, src3: VReg) -> VReg {
        assert!(src1.is_vector(), "FMA src1 must be vector");
        assert!(src2.is_vector(), "FMA src2 must be vector");
        assert!(src3.is_vector(), "FMA src3 must be vector");

        let dest = self.fresh_vector();
        let op = SsaOp::new(
            0,
            Engine::Valu,
            Opcode::MultiplyAdd,
            Some(dest),
            vec![
                Operand::VReg(src1),
                Operand::VReg(src2),
                Operand::VReg(src3),
            ],
        );
        self.emit(op);
        dest
    }

    /// Emit vselect: result[i] = cond[i] ? true_val[i] : false_val[i]
    #[track_caller]
    pub fn emit_vselect(&mut self, cond: VReg, true_val: VReg, false_val: VReg) -> VReg {
        assert!(cond.is_vector(), "vselect cond must be vector");
        assert!(true_val.is_vector(), "vselect true_val must be vector");
        assert!(false_val.is_vector(), "vselect false_val must be vector");

        let dest = self.fresh_vector();
        let op = SsaOp::new(
            0,
            Engine::Flow,
            Opcode::Vselect,
            Some(dest),
            vec![
                Operand::VReg(cond),
                Operand::VReg(true_val),
                Operand::VReg(false_val),
            ],
        );
        self.emit(op);
        dest
    }

    // ========================================================================
    // Load Operations
    // ========================================================================

    /// Emit const load: result = immediate_value
    /// Returns the destination scalar VReg.
    #[track_caller]
    pub fn emit_const(&mut self, value: i64) -> VReg {
        let dest = self.fresh_scalar();
        let mut op = SsaOp::new(
            0,
            Engine::Load,
            Opcode::Const,
            Some(dest),
            vec![Operand::Imm(value)],
        );
        op.produces_constant = Some(value);
        self.emit(op);

        // Track that this VReg holds a constant
        self.vreg_constants.insert(dest, value);

        dest
    }

    /// Emit scalar load from memory address VReg: result = mem[addr]
    /// The addr VReg holds the memory address to load from.
    #[track_caller]
    pub fn emit_load(&mut self, addr: VReg) -> VReg {
        assert!(addr.is_scalar(), "load addr must be scalar");

        let dest = self.fresh_scalar();
        let op = SsaOp::new(
            0,
            Engine::Load,
            Opcode::Load,
            Some(dest),
            vec![Operand::VReg(addr)],
        );
        self.emit(op);
        dest
    }

    /// Emit scalar load from immediate address: result = mem[addr]
    #[track_caller]
    pub fn emit_load_imm(&mut self, addr: i64) -> VReg {
        let dest = self.fresh_scalar();
        let op = SsaOp::new(
            0,
            Engine::Load,
            Opcode::Load,
            Some(dest),
            vec![Operand::Imm(addr)],
        );
        self.emit(op);
        dest
    }

    /// Emit vector load (contiguous): result[0..8] = mem[addr..addr+8]
    /// The addr VReg is a scalar holding the base address.
    #[track_caller]
    pub fn emit_vload(&mut self, addr: VReg) -> VReg {
        assert!(addr.is_scalar(), "vload addr must be scalar");

        let dest = self.fresh_vector();
        let op = SsaOp::new(
            0,
            Engine::Load,
            Opcode::Vload,
            Some(dest),
            vec![Operand::VReg(addr)],
        );
        self.emit(op);
        dest
    }

    /// Emit vector load from immediate address: result[0..8] = mem[addr..addr+8]
    #[track_caller]
    pub fn emit_vload_imm(&mut self, addr: i64) -> VReg {
        let dest = self.fresh_vector();
        let op = SsaOp::new(
            0,
            Engine::Load,
            Opcode::Vload,
            Some(dest),
            vec![Operand::Imm(addr)],
        );
        self.emit(op);
        dest
    }

    // ========================================================================
    // Store Operations
    // ========================================================================

    /// Emit scalar store: mem[addr] = src
    /// No return value (stores don't produce VRegs).
    #[track_caller]
    pub fn emit_store(&mut self, addr: VReg, src: VReg) {
        assert!(addr.is_scalar(), "store addr must be scalar");
        assert!(src.is_scalar(), "store src must be scalar");

        let op = SsaOp::new(
            0,
            Engine::Store,
            Opcode::Store,
            None, // stores have no dest
            vec![Operand::VReg(addr), Operand::VReg(src)],
        );
        self.emit(op);
    }

    /// Emit scalar store to immediate address: mem[addr] = src
    #[track_caller]
    pub fn emit_store_imm(&mut self, addr: i64, src: VReg) {
        assert!(src.is_scalar(), "store src must be scalar");

        let op = SsaOp::new(
            0,
            Engine::Store,
            Opcode::Store,
            None,
            vec![Operand::Imm(addr), Operand::VReg(src)],
        );
        self.emit(op);
    }

    /// Emit vector store (contiguous): mem[addr..addr+8] = src[0..8]
    /// The addr VReg is a scalar holding the base address.
    #[track_caller]
    pub fn emit_vstore(&mut self, addr: VReg, src: VReg) {
        assert!(addr.is_scalar(), "vstore addr must be scalar");
        assert!(src.is_vector(), "vstore src must be vector");

        let op = SsaOp::new(
            0,
            Engine::Store,
            Opcode::Vstore,
            None,
            vec![Operand::VReg(addr), Operand::VReg(src)],
        );
        self.emit(op);
    }

    /// Emit vector store to immediate address: mem[addr..addr+8] = src[0..8]
    #[track_caller]
    pub fn emit_vstore_imm(&mut self, addr: i64, src: VReg) {
        assert!(src.is_vector(), "vstore src must be vector");

        let op = SsaOp::new(
            0,
            Engine::Store,
            Opcode::Vstore,
            None,
            vec![Operand::Imm(addr), Operand::VReg(src)],
        );
        self.emit(op);
    }

    // ========================================================================
    // Flow Operations
    // ========================================================================

    /// Emit select: result = cond ? true_val : false_val
    #[track_caller]
    pub fn emit_select(&mut self, cond: VReg, true_val: VReg, false_val: VReg) -> VReg {
        assert!(cond.is_scalar(), "select cond must be scalar");
        assert!(true_val.is_scalar(), "select true_val must be scalar");
        assert!(false_val.is_scalar(), "select false_val must be scalar");

        let dest = self.fresh_scalar();
        let op = SsaOp::new(
            0,
            Engine::Flow,
            Opcode::Select,
            Some(dest),
            vec![
                Operand::VReg(cond),
                Operand::VReg(true_val),
                Operand::VReg(false_val),
            ],
        );
        self.emit(op);
        dest
    }

    /// Emit add_imm via flow slot: result = src + imm
    /// Uses the flow engine's add_imm instruction.
    #[track_caller]
    pub fn emit_add_imm(&mut self, src: VReg, imm: i64) -> VReg {
        assert!(src.is_scalar(), "add_imm src must be scalar");

        let dest = self.fresh_scalar();
        let mut op = SsaOp::new(
            0,
            Engine::Flow,
            Opcode::AddImm,
            Some(dest),
            vec![Operand::VReg(src), Operand::Imm(imm)],
        );

        // If src is constant 0 and we're adding imm, result is constant imm
        if let Some(src_const) = self.vreg_constants.get(&src) {
            op.produces_constant = Some(src_const + imm);
            // Note: we don't cache this since it requires src to be available
        }

        self.emit(op);
        dest
    }
}

/// Compute SSA dependencies: only RAW (read-after-write) dependencies.
///
/// For each op that reads a VReg, add a dependency on the op that defined that VReg.
/// No WAW or WAR dependencies - those don't exist in SSA form.
pub fn compute_ssa_dependencies(ops: &mut [SsaOp], vreg_def: &HashMap<VReg, usize>) {
    for op in ops.iter_mut() {
        op.dep_ids.clear();

        // RAW: depend on producers of values we read
        for src in &op.srcs {
            if let Operand::VReg(vreg) = src {
                if let Some(&def_id) = vreg_def.get(vreg) {
                    op.dep_ids.insert(def_id);
                }
            }
        }
    }
}

/// Compute live intervals for all VRegs based on scheduled cycles.
///
/// For each VReg:
/// - start = cycle when the defining op executes
/// - end = cycle when the last op that reads this VReg executes
///
/// If a VReg is never read, end = start (dies immediately after def).
pub fn compute_live_intervals(ops: &[SsaOp]) -> HashMap<VReg, LiveInterval> {
    let mut intervals: HashMap<VReg, LiveInterval> = HashMap::new();

    // First pass: record definition cycles
    for op in ops {
        if let Some(dest) = op.dest {
            intervals.insert(dest, LiveInterval::new(op.cycle, op.cycle));
        }
    }

    // Second pass: extend end to last use cycle
    for op in ops {
        for vreg in op.reads() {
            if let Some(interval) = intervals.get_mut(&vreg) {
                if op.cycle > interval.end {
                    interval.end = op.cycle;
                }
            }
        }
    }

    intervals
}

/// Compute peak liveness: maximum number of words live at any cycle.
///
/// This is used to verify that the schedule is feasible within scratch limits.
pub fn compute_peak_liveness(ops: &[SsaOp]) -> usize {
    if ops.is_empty() {
        return 0;
    }

    let intervals = compute_live_intervals(ops);
    let max_cycle = ops.iter().map(|op| op.cycle).max().unwrap_or(0);

    let mut peak = 0;
    for cycle in 0..=max_cycle {
        let live_words: usize = intervals
            .iter()
            .filter(|(_, interval)| interval.contains(cycle))
            .map(|(vreg, _)| vreg.size())
            .sum();
        peak = peak.max(live_words);
    }

    peak
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emit_const_returns_fresh_vreg() {
        let mut emit = SsaEmitContext::new();
        let v1 = emit.emit_const(5);
        let v2 = emit.emit_const(6);
        let v3 = emit.emit_const(5); // same value, different VReg

        assert_ne!(v1, v2);
        assert_ne!(v2, v3);
        assert_ne!(v1, v3); // SSA: each def is unique
    }

    #[test]
    fn test_emit_alu_returns_fresh_vreg() {
        let mut emit = SsaEmitContext::new();
        let v1 = emit.emit_const(5);
        let v2 = emit.emit_const(6);
        let v3 = emit.emit_alu(Opcode::Add, v1, v2);

        assert_ne!(v1, v2);
        assert_ne!(v2, v3);
        assert_ne!(v1, v3);
    }

    #[test]
    fn test_emit_vbroadcast_cse() {
        let mut emit = SsaEmitContext::new();
        let scalar = emit.emit_const(42);
        let vec1 = emit.emit_vbroadcast(scalar);
        let vec2 = emit.emit_vbroadcast(scalar); // CSE should return same VReg

        assert_eq!(vec1, vec2);
        assert_eq!(emit.ops.len(), 2); // only 2 ops: const + broadcast
    }

    #[test]
    fn test_emit_vbroadcast_force_no_cse() {
        let mut emit = SsaEmitContext::new();
        let scalar = emit.emit_const(42);
        let vec1 = emit.emit_vbroadcast_force(scalar);
        let vec2 = emit.emit_vbroadcast_force(scalar);

        assert_ne!(vec1, vec2); // force creates new VReg
        assert_eq!(emit.ops.len(), 3); // const + 2 broadcasts
    }

    #[test]
    fn test_emit_tracks_vreg_defs() {
        let mut emit = SsaEmitContext::new();
        let v1 = emit.emit_const(5);
        let v2 = emit.emit_const(6);
        let v3 = emit.emit_alu(Opcode::Add, v1, v2);

        assert_eq!(emit.get_def(v1), Some(0));
        assert_eq!(emit.get_def(v2), Some(1));
        assert_eq!(emit.get_def(v3), Some(2));
    }

    #[test]
    fn test_raw_dependency() {
        let mut emit = SsaEmitContext::new();
        let v1 = emit.emit_const(5);
        let _v2 = emit.emit_alu(Opcode::Add, v1, v1);

        compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def);

        // op1 (v2 = v1 + v1) depends on op0 (v1 = const 5)
        assert!(emit.ops[1].dep_ids.contains(&0));
    }

    #[test]
    fn test_no_false_waw_dependency() {
        // v1 = const 5
        // v2 = const 6  -> NO dependency on v1 (different VRegs, SSA)
        let mut emit = SsaEmitContext::new();
        let _v1 = emit.emit_const(5);
        let _v2 = emit.emit_const(6);

        compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def);

        // op1 should have no dependencies
        assert!(emit.ops[1].dep_ids.is_empty());
    }

    #[test]
    fn test_no_false_war_dependency() {
        // v1 = const 5
        // v2 = v1 + v1
        // v3 = const 7  -> NO dependency on v2 reading v1
        let mut emit = SsaEmitContext::new();
        let v1 = emit.emit_const(5);
        let _v2 = emit.emit_alu(Opcode::Add, v1, v1);
        let _v3 = emit.emit_const(7);

        compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def);

        // op2 (v3) should have no dependencies
        assert!(emit.ops[2].dep_ids.is_empty());
    }

    #[test]
    fn test_live_interval_simple() {
        let mut emit = SsaEmitContext::new();
        let v1 = emit.emit_const(5);
        let _v2 = emit.emit_alu(Opcode::Add, v1, v1);

        // Simulate scheduling
        emit.ops[0].cycle = 0;
        emit.ops[1].cycle = 1;

        let intervals = compute_live_intervals(&emit.ops);

        // v1: defined at cycle 0, last used at cycle 1
        assert_eq!(intervals[&v1], LiveInterval::new(0, 1));
    }

    #[test]
    fn test_live_interval_multiple_uses() {
        let mut emit = SsaEmitContext::new();
        let v1 = emit.emit_const(5);
        let v2 = emit.emit_alu(Opcode::Add, v1, v1);
        let _v3 = emit.emit_alu(Opcode::Mul, v1, v2); // v1 used again

        // Simulate scheduling
        emit.ops[0].cycle = 0;
        emit.ops[1].cycle = 1;
        emit.ops[2].cycle = 5;

        let intervals = compute_live_intervals(&emit.ops);

        // v1: defined at cycle 0, last used at cycle 5
        assert_eq!(intervals[&v1], LiveInterval::new(0, 5));
    }

    #[test]
    fn test_peak_liveness() {
        let mut emit = SsaEmitContext::new();
        let v1 = emit.emit_const(5); // scalar: 1 word
        let _v2 = emit.emit_vbroadcast(v1); // vector: 8 words
        let _v3 = emit.emit_alu(Opcode::Add, v1, v1);

        // Simulate scheduling where v1 and v2 are both live
        emit.ops[0].cycle = 0;
        emit.ops[1].cycle = 1;
        emit.ops[2].cycle = 2;

        let peak = compute_peak_liveness(&emit.ops);
        // At cycle 1: v1 (1 word) + v2 being defined (8 words) = 9 words
        // At cycle 2: v1 (1 word) still live, v2 (8 words) - but v2 not used = dies at cycle 1
        // Actually v2 dies immediately since nothing reads it
        // Peak is at cycle 2 where v1 is used: 1 word (v1)
        // Wait, let me reconsider...
        // v1 live [0, 2], v2 live [1, 1]
        // At cycle 0: 1 word (v1)
        // At cycle 1: 1 word (v1) + 8 words (v2) = 9 words
        // At cycle 2: 1 word (v1)
        assert_eq!(peak, 9);
    }

    #[test]
    fn test_ssa_op_reads() {
        let mut emit = SsaEmitContext::new();
        let v1 = emit.emit_const(5);
        let v2 = emit.emit_const(6);
        let _v3 = emit.emit_alu(Opcode::Add, v1, v2);

        let op = &emit.ops[2]; // the add op
        let reads: Vec<_> = op.reads().collect();
        assert_eq!(reads.len(), 2);
        assert!(reads.contains(&v1));
        assert!(reads.contains(&v2));
    }

    #[test]
    fn test_vector_ops() {
        let mut emit = SsaEmitContext::new();
        let s1 = emit.emit_const(5);
        let s2 = emit.emit_const(6);
        let v1 = emit.emit_vbroadcast(s1);
        let v2 = emit.emit_vbroadcast(s2);
        let v3 = emit.emit_valu(Opcode::Add, v1, v2);

        assert!(v1.is_vector());
        assert!(v2.is_vector());
        assert!(v3.is_vector());
        assert!(s1.is_scalar());
        assert!(s2.is_scalar());
    }

    #[test]
    fn test_fma() {
        let mut emit = SsaEmitContext::new();
        let s1 = emit.emit_const(1);
        let s2 = emit.emit_const(2);
        let s3 = emit.emit_const(3);
        let v1 = emit.emit_vbroadcast(s1);
        let v2 = emit.emit_vbroadcast(s2);
        let v3 = emit.emit_vbroadcast(s3);
        let result = emit.emit_fma(v1, v2, v3);

        assert!(result.is_vector());
        assert_eq!(emit.ops.last().unwrap().opcode, Opcode::MultiplyAdd);
    }

    #[test]
    fn test_store_no_dest() {
        let mut emit = SsaEmitContext::new();
        let addr = emit.emit_const(100);
        let val = emit.emit_const(42);
        emit.emit_store(addr, val);

        let store_op = emit.ops.last().unwrap();
        assert!(store_op.dest.is_none());
        assert!(store_op.writes().is_none());
    }
}
