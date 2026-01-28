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
    /// Const cache: constant value -> scalar VReg (CSE deduplication)
    pub const_cache: HashMap<i64, VReg>,

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
            const_cache: HashMap::new(),
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
        self.const_cache.clear();
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
    ///
    /// Uses CSE deduplication - if the same constant value was already emitted,
    /// returns the existing VReg.
    #[track_caller]
    pub fn emit_const(&mut self, value: i64) -> VReg {
        // CSE: check if already emitted
        if let Some(&existing) = self.const_cache.get(&value) {
            return existing;
        }

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
        self.const_cache.insert(value, dest);

        dest
    }

    /// Emit const load without CSE - always creates a new const op.
    /// Use when the destination matters or you specifically need a separate VReg.
    #[track_caller]
    pub fn emit_const_force(&mut self, value: i64) -> VReg {
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

    // ========================================================================
    // Barrier Operations (Scheduling-only)
    // ========================================================================

    /// Emit a barrier pseudo-op for round boundaries.
    ///
    /// The barrier:
    /// - Uses Engine::Barrier (slot_limit=0, is_hardware=false)
    /// - Gets assigned a cycle during scheduling
    /// - Forces all prior ops to complete before any subsequent ops start
    /// - Is elided from final output (not a real instruction)
    ///
    /// Use `compute_ssa_dependencies_with_barriers` after emission to add
    /// barrier dependency constraints.
    #[track_caller]
    pub fn emit_barrier(&mut self) {
        let op = SsaOp::new(
            0, // set by emit()
            Engine::Barrier,
            Opcode::Barrier,
            None,   // no destination
            vec![], // no operands
        );
        self.emit(op);
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

/// Compute dependencies with barrier constraints for greedy scheduling.
///
/// Barriers create hard scheduling boundaries between rounds:
/// - Each barrier depends on all ops in its preceding section (since last barrier)
/// - Each barrier also depends on the previous barrier (to chain them)
/// - All ops after a barrier depend on that barrier
///
/// This is more efficient than making each barrier depend on ALL prior ops,
/// which would create O(n²) dependencies. Instead, this creates a chain of
/// barriers where each barrier gates its section.
///
/// The effect is the same: all ops before barrier N must complete before any
/// ops after barrier N can start. But the dependency graph is O(n) instead of O(n²).
pub fn compute_ssa_dependencies_with_barriers(ops: &mut [SsaOp], vreg_def: &HashMap<VReg, usize>) {
    // First compute normal RAW deps
    compute_ssa_dependencies(ops, vreg_def);

    // Find barrier ops
    let barrier_indices: Vec<usize> = ops
        .iter()
        .enumerate()
        .filter(|(_, op)| op.opcode == Opcode::Barrier)
        .map(|(idx, _)| idx)
        .collect();

    if barrier_indices.is_empty() {
        return;
    }

    // Process barriers in order
    let mut prev_barrier_idx: Option<usize> = None;

    for &barrier_idx in &barrier_indices {
        let barrier_op_id = ops[barrier_idx].id;

        // Find start of this section (after previous barrier, or start of ops)
        let section_start = prev_barrier_idx.map_or(0, |idx| idx + 1);

        // Barrier depends on all non-barrier ops in its section
        for idx in section_start..barrier_idx {
            if ops[idx].opcode != Opcode::Barrier {
                ops[barrier_idx].dep_ids.insert(ops[idx].id);
            }
        }

        // Barrier also depends on previous barrier (to chain them)
        if let Some(prev_idx) = prev_barrier_idx {
            ops[barrier_idx].dep_ids.insert(ops[prev_idx].id);
        }

        // All ops after this barrier (until next barrier or end) depend on it
        let next_barrier_idx = barrier_indices
            .iter()
            .find(|&&i| i > barrier_idx)
            .copied()
            .unwrap_or(ops.len());

        for idx in (barrier_idx + 1)..next_barrier_idx {
            if ops[idx].opcode != Opcode::Barrier {
                ops[idx].dep_ids.insert(barrier_op_id);
            }
        }

        prev_barrier_idx = Some(barrier_idx);
    }
}

/// Compute live intervals for all VRegs based on scheduled cycles.
///
/// Uses **CLOSED intervals** [start, end] as documented in `types.rs`.
/// See `types.rs` "AUTHORITATIVE LIVENESS MODEL DOCUMENTATION" for full details.
///
/// Returns closed intervals where:
/// - `start` = cycle when the VReg is written (definition cycle)
/// - `end` = last cycle the VReg is ACCESSED (last read, or def+1 for dead VRegs)
///
/// For a VReg defined at cycle D and last read at cycle R:
/// - Interval = [D, R] (closed)
///
/// For a dead VReg (defined at D, never read):
/// - Interval = [D, D+1] — the "+1" ensures write conflicts are detected
pub fn compute_live_intervals(ops: &[SsaOp]) -> HashMap<VReg, LiveInterval> {
    let mut intervals: HashMap<VReg, LiveInterval> = HashMap::new();

    // First pass: record definition cycles.
    // Initialize with [def, def] — will be extended if read.
    for op in ops {
        if let Some(dest) = op.dest {
            intervals.insert(dest, LiveInterval::new(op.cycle, op.cycle));
        }
    }

    // Second pass: extend end to last READ cycle.
    // CLOSED interval: end = last_read_cycle (not last_read + 1).
    for op in ops {
        for vreg in op.reads() {
            if let Some(interval) = intervals.get_mut(&vreg) {
                // Closed interval: end is the read cycle itself
                if op.cycle > interval.end {
                    interval.end = op.cycle;
                }
            }
        }
    }

    // Third pass: extend dead VRegs (start == end) to [start, start+1].
    // This prevents same-cycle write conflicts. See types.rs for explanation.
    for interval in intervals.values_mut() {
        if interval.start == interval.end {
            // Dead VReg: written but never read. Extend by 1 to prevent
            // same-cycle write conflicts with other VRegs.
            interval.end = interval.start + 1;
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
    fn test_emit_const_cse() {
        let mut emit = SsaEmitContext::new();
        let v1 = emit.emit_const(5);
        let v2 = emit.emit_const(6);
        let v3 = emit.emit_const(5); // same value, CSE returns same VReg

        assert_ne!(v1, v2);
        assert_eq!(v1, v3); // CSE: same constant value returns same VReg
        assert_eq!(emit.ops.len(), 2); // only 2 ops: const(5) + const(6)
    }

    #[test]
    fn test_emit_const_force_no_cse() {
        let mut emit = SsaEmitContext::new();
        let v1 = emit.emit_const_force(5);
        let v2 = emit.emit_const_force(5); // same value, but force creates new VReg

        assert_ne!(v1, v2); // force creates new VReg
        assert_eq!(emit.ops.len(), 2); // 2 ops: const(5) + const(5)
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

        // v1: defined at cycle 0, last read at cycle 1
        // CLOSED interval: [0, 1] — end = last_read_cycle = 1
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

        // CLOSED intervals: end = last_read_cycle
        // v1: defined at cycle 0, last read at cycle 5 → [0, 5]
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
        // CLOSED intervals (see types.rs for authoritative docs):
        // v1: defined at 0, last read at 2 → [0, 2]
        // v2: defined at 1, never read (dead) → [1, 2] (+1 extension)
        // v3: defined at 2, never read (dead) → [2, 3] (+1 extension)
        //
        // At cycle 0: v1 live (1 word)
        // At cycle 1: v1 [0,2] live (1) + v2 [1,2] live (8) = 9 words
        // At cycle 2: v1 [0,2] live (1) + v2 [1,2] live (8) + v3 [2,3] live (1) = 10 words
        //
        // Peak = 10 at cycle 2
        assert_eq!(peak, 10);
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

    #[test]
    fn test_emit_barrier() {
        let mut emit = SsaEmitContext::new();
        emit.emit_barrier();

        let barrier_op = emit.ops.last().unwrap();
        assert_eq!(barrier_op.engine, Engine::Barrier);
        assert_eq!(barrier_op.opcode, Opcode::Barrier);
        assert!(barrier_op.dest.is_none());
        assert!(barrier_op.srcs.is_empty());
    }

    #[test]
    fn test_barrier_dependencies() {
        // Test that barrier creates proper dependencies:
        // - Barrier depends on all prior non-barrier ops
        // - All ops after barrier depend on it
        let mut emit = SsaEmitContext::new();

        // "Round 1"
        let v1 = emit.emit_const(5);  // op 0
        let v2 = emit.emit_const(6);  // op 1
        let _v3 = emit.emit_alu(Opcode::Add, v1, v2);  // op 2

        // Barrier
        emit.emit_barrier();  // op 3

        // "Round 2"
        let v4 = emit.emit_const(7);  // op 4
        let _v5 = emit.emit_alu(Opcode::Mul, v4, v1);  // op 5 (uses v1 from round 1)

        compute_ssa_dependencies_with_barriers(&mut emit.ops, &emit.vreg_def);

        // Barrier (op 3) should depend on ops 0, 1, 2
        assert!(emit.ops[3].dep_ids.contains(&0), "Barrier should depend on op 0");
        assert!(emit.ops[3].dep_ids.contains(&1), "Barrier should depend on op 1");
        assert!(emit.ops[3].dep_ids.contains(&2), "Barrier should depend on op 2");

        // Ops after barrier should depend on it
        assert!(emit.ops[4].dep_ids.contains(&3), "Op 4 should depend on barrier");
        assert!(emit.ops[5].dep_ids.contains(&3), "Op 5 should depend on barrier");

        // Op 5 should also have RAW dep on op 0 (v1) and op 4 (v4)
        assert!(emit.ops[5].dep_ids.contains(&0), "Op 5 should have RAW dep on op 0 (v1)");
        assert!(emit.ops[5].dep_ids.contains(&4), "Op 5 should have RAW dep on op 4 (v4)");
    }

    #[test]
    fn test_multiple_barriers() {
        // Test multiple barriers in sequence with chained dependencies
        let mut emit = SsaEmitContext::new();

        let v1 = emit.emit_const(1);  // op 0
        emit.emit_barrier();  // op 1
        let v2 = emit.emit_const(2);  // op 2
        emit.emit_barrier();  // op 3
        let _v3 = emit.emit_alu(Opcode::Add, v1, v2);  // op 4

        compute_ssa_dependencies_with_barriers(&mut emit.ops, &emit.vreg_def);

        // First barrier depends on op 0 (its section)
        assert!(emit.ops[1].dep_ids.contains(&0), "First barrier should depend on op 0");

        // Op 2 depends on first barrier
        assert!(emit.ops[2].dep_ids.contains(&1), "Op 2 should depend on first barrier");

        // Second barrier depends on:
        // - op 2 (its section - ops between barrier 1 and barrier 3)
        // - barrier 1 (previous barrier - chaining)
        assert!(emit.ops[3].dep_ids.contains(&2), "Second barrier should depend on op 2");
        assert!(emit.ops[3].dep_ids.contains(&1), "Second barrier should depend on first barrier (chain)");
        // Second barrier does NOT directly depend on op 0 (that's handled by the chain)
        assert!(!emit.ops[3].dep_ids.contains(&0), "Second barrier should NOT directly depend on op 0 (chained via barrier 1)");

        // Op 4 depends on second barrier (and has RAW deps on v1, v2)
        assert!(emit.ops[4].dep_ids.contains(&3), "Op 4 should depend on second barrier");
        assert!(emit.ops[4].dep_ids.contains(&0), "Op 4 should have RAW dep on v1");
        assert!(emit.ops[4].dep_ids.contains(&2), "Op 4 should have RAW dep on v2");
    }
}
