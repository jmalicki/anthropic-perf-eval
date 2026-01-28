//! Address assignment for SSA operations.
//!
//! After scheduling completes (when cycle assignments are known), this module
//! assigns physical scratch addresses to virtual registers (VRegs) using
//! linear scan allocation.
//!
//! ## Liveness Model: Same-Cycle Reuse
//!
//! The liveness model allows **same-cycle reuse**: a VReg that ends at cycle C
//! can share storage with a VReg that starts at cycle C. This is valid because
//! within a VLIW cycle, reads happen before writes:
//!
//! ```text
//! Cycle 5:  [Read A] ... [Write B]
//!           ^^^^^^^       ^^^^^^^
//!           A is read     B is written
//!
//! Storage can be reused: A dies, then B is born in the same cycle.
//! ```
//!
//! **Key implementation details:**
//!
//! - Live interval: `[def_cycle, last_use_cycle]` (both inclusive)
//! - Intervals `[0,5]` and `[5,10]` do **NOT** overlap (5 is the transition point)
//! - `LiveInterval::overlaps()` uses `<` comparison: `self.start < other.end`
//! - Linear scan expires VRegs where `end_cycle <= current_start` (note: `<=`)
//! - Peak liveness: births before deaths at same cycle (conservative for capacity check)
//!
//! This model must be consistent across:
//! - `LiveInterval::overlaps()` in `types.rs`
//! - `compute_peak_liveness()` event ordering
//! - Linear scan expiration condition
//! - Scheduler budget tracking (VRegs freed mid-cycle)
//!
//! ## Algorithm: Linear Scan Allocation
//!
//! 1. Compute live intervals: [def_cycle, last_use_cycle] for each VReg
//! 2. Sort intervals by def_cycle (start)
//! 3. For each VReg:
//!    - Expire VRegs that ended at or before current def_cycle (same-cycle reuse)
//!    - Return expired addresses to free pool
//!    - Allocate from free pool or extend allocation
//!    - Record VReg -> address mapping
//!
//! ## Vector Handling
//!
//! Vector VRegs occupy 8 contiguous words (VLEN=8). Unlike some architectures,
//! vectors do NOT require alignment - they can start at any address.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::types::{LiveInterval, Operand, SsaOp, VReg, VLEN};

/// Result of address assignment.
#[derive(Debug, Default)]
pub struct AddressAssignmentResult {
    /// Mapping from VReg to physical address
    pub vreg_to_addr: HashMap<VReg, i64>,
    /// Peak simultaneous live words (inherent to schedule)
    pub peak_live: usize,
    /// Highest address used (max_addr + 1 = total scratch used)
    pub max_addr: i64,
    /// Number of VRegs assigned
    pub vreg_count: usize,
    /// Total words assigned (scalars count as 1, vectors as 8)
    pub total_words: usize,
    /// True if assignment exceeded the scratch budget
    pub budget_exceeded: bool,
    /// The scratch budget that was used
    pub budget: usize,
}

/// Configuration for address assignment.
#[derive(Clone, Debug)]
pub struct AddressAssignConfig {
    /// Scratch memory budget (default: 1536)
    pub budget: usize,
    /// Starting address (default: 1 to reserve 0 for ZERO_REG)
    pub start_addr: i64,
    /// Fixed VRegs that must use specific addresses
    pub fixed_vregs: HashMap<VReg, i64>,
}

impl Default for AddressAssignConfig {
    fn default() -> Self {
        Self {
            budget: 1536,
            start_addr: 1, // Reserve address 0 for ZERO_REG
            fixed_vregs: HashMap::new(),
        }
    }
}

/// Compute live intervals for all VRegs based on scheduled cycles.
///
/// For each VReg:
/// - start = cycle when the defining op executes
/// - end = cycle when the last op that reads this VReg executes
///
/// If a VReg is never read (dead definition), end = start.
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
        for src in &op.srcs {
            if let Operand::VReg(vreg) = src {
                if let Some(interval) = intervals.get_mut(vreg) {
                    if op.cycle > interval.end {
                        interval.end = op.cycle;
                    }
                }
            }
        }
    }

    intervals
}

/// Compute peak liveness in words at any cycle using the same-cycle reuse model.
///
/// This uses the same model as linear scan allocation:
/// - At cycle C, VRegs with end_cycle < C are expired (freed) BEFORE new VRegs are allocated
/// - At cycle C, VRegs with end_cycle == C are still alive during C (for reads)
/// - Same-cycle reuse: interval [0,1] and [1,2] can share storage because [0,1] is
///   expired (end_cycle 1 <= start_cycle 1) before [1,2] is allocated
///
/// This correctly counts single-cycle intervals [N, N] as 1 word.
pub fn compute_peak_liveness(intervals: &HashMap<VReg, LiveInterval>) -> usize {
    if intervals.is_empty() {
        return 0;
    }

    // Sort intervals by start cycle (same as linear scan)
    let mut sorted: Vec<(&VReg, &LiveInterval)> = intervals.iter().collect();
    sorted.sort_by_key(|(_, interval)| (interval.start, interval.end));

    // Simulate linear scan to track actual peak
    // Active set: (end_cycle, size)
    let mut active: BinaryHeap<Reverse<(i32, usize)>> = BinaryHeap::new();
    let mut current_words: usize = 0;
    let mut peak_words: usize = 0;

    for (vreg, interval) in sorted {
        // Expire intervals that ended at or before this one starts (same-cycle reuse)
        while let Some(&Reverse((end_cycle, size))) = active.peek() {
            if end_cycle <= interval.start {
                active.pop();
                current_words = current_words.saturating_sub(size);
            } else {
                break;
            }
        }

        // Allocate this VReg
        let size = vreg.size();
        current_words += size;
        peak_words = peak_words.max(current_words);

        // Add to active set
        active.push(Reverse((interval.end, size)));
    }

    peak_words
}

/// Assign physical addresses to VRegs using linear scan allocation.
///
/// This is the main entry point for SSA address assignment.
///
/// # Arguments
/// * `ops` - Scheduled SSA operations (must have cycle assignments)
/// * `config` - Assignment configuration (budget, start address, fixed VRegs)
///
/// # Returns
/// Assignment result with VReg -> address mapping and statistics.
pub fn assign_addresses(ops: &[SsaOp], config: &AddressAssignConfig) -> AddressAssignmentResult {
    // Compute live intervals
    let intervals = compute_live_intervals(ops);

    // Calculate peak liveness
    let peak_live = compute_peak_liveness(&intervals);
    let budget_exceeded = peak_live > config.budget;

    if intervals.is_empty() {
        return AddressAssignmentResult {
            budget: config.budget,
            peak_live,
            budget_exceeded,
            ..Default::default()
        };
    }

    // Collect intervals with their VRegs, sorted by start cycle
    let mut sorted: Vec<(VReg, LiveInterval)> = intervals.into_iter().collect();
    sorted.sort_by_key(|(_, interval)| (interval.start, interval.end));

    // Handle fixed VRegs first
    let mut vreg_to_addr: HashMap<VReg, i64> = HashMap::new();
    let mut used_ranges: Vec<(i64, i64)> = Vec::new(); // (start, end_exclusive)

    for (vreg, addr) in &config.fixed_vregs {
        vreg_to_addr.insert(*vreg, *addr);
        let end = addr + vreg.size() as i64;
        used_ranges.push((*addr, end));
    }

    // Active intervals: (end_cycle, allocated_base_addr, size)
    let mut active: BinaryHeap<Reverse<(i32, i64, usize)>> = BinaryHeap::new();

    // Free ranges: list of (base_addr, size) available for reuse
    // Use BinaryHeap with Reverse to prefer lower addresses
    let mut free_ranges: BinaryHeap<Reverse<(i64, usize)>> = BinaryHeap::new();

    // Next fresh address to allocate
    let mut next_addr = config.start_addr;

    // Adjust next_addr to avoid fixed VReg ranges
    for &(start, end) in &used_ranges {
        if next_addr >= start && next_addr < end {
            next_addr = end;
        }
    }

    for (vreg, interval) in &sorted {
        // Skip fixed VRegs
        if config.fixed_vregs.contains_key(vreg) {
            // Still need to track as active for correct expiry
            let addr = config.fixed_vregs[vreg];
            active.push(Reverse((interval.end, addr, vreg.size())));
            continue;
        }

        let size = vreg.size();

        // Expire intervals that ended at or before this one starts (allowing same-cycle reuse)
        while let Some(&Reverse((end_cycle, freed_addr, freed_size))) = active.peek() {
            if end_cycle <= interval.start {
                active.pop();
                // Return freed range to pool (don't free fixed ranges)
                if !is_in_fixed_range(freed_addr, freed_size, &config.fixed_vregs) {
                    free_ranges.push(Reverse((freed_addr, freed_size)));
                }
            } else {
                break;
            }
        }

        // Try to find a free range that fits this VReg's size
        let mut new_addr: Option<i64> = None;
        let mut temp_ranges: Vec<(i64, usize)> = Vec::new();

        while let Some(Reverse((addr, range_size))) = free_ranges.pop() {
            if range_size >= size {
                // Found a suitable range
                new_addr = Some(addr);
                // If there's leftover space, put it back
                if range_size > size {
                    temp_ranges.push((addr + size as i64, range_size - size));
                }
                break;
            } else {
                temp_ranges.push((addr, range_size));
            }
        }

        // Put back ranges we couldn't use
        for (addr, range_size) in temp_ranges {
            free_ranges.push(Reverse((addr, range_size)));
        }

        // If no free range found, allocate fresh
        let new_addr = new_addr.unwrap_or_else(|| {
            let addr = next_addr;
            next_addr += size as i64;
            addr
        });

        vreg_to_addr.insert(*vreg, new_addr);
        active.push(Reverse((interval.end, new_addr, size)));
    }

    // Calculate statistics
    let max_addr = vreg_to_addr
        .iter()
        .map(|(vreg, &addr)| addr + vreg.size() as i64 - 1)
        .max()
        .unwrap_or(0);

    let vreg_count = vreg_to_addr.len();
    let total_words: usize = sorted.iter().map(|(vreg, _)| vreg.size()).sum();

    AddressAssignmentResult {
        vreg_to_addr,
        peak_live,
        max_addr,
        vreg_count,
        total_words,
        budget_exceeded,
        budget: config.budget,
    }
}

/// Check if an address range overlaps with any fixed VReg
fn is_in_fixed_range(addr: i64, size: usize, fixed: &HashMap<VReg, i64>) -> bool {
    for (vreg, &fixed_addr) in fixed {
        let fixed_end = fixed_addr + vreg.size() as i64;
        let end = addr + size as i64;
        // Check overlap
        if !(end <= fixed_addr || addr >= fixed_end) {
            return true;
        }
    }
    false
}

/// Convert scheduled SsaOps to concrete Ops with physical addresses.
///
/// This is the final step after scheduling and address assignment.
/// It replaces VReg references with physical addresses.
use crate::types::{Op, SlotVal};
use arrayvec::ArrayVec;

/// Lower an SsaOp to a concrete Op using the VReg -> address mapping.
pub fn lower_ssa_op(ssa_op: &SsaOp, vreg_to_addr: &HashMap<VReg, i64>) -> Op {
    // Build the slot array based on opcode
    let mut slot: ArrayVec<SlotVal, 6> = ArrayVec::new();
    slot.push(SlotVal::Str(ssa_op.opcode.as_str().to_string()));

    // Add destination address (if any)
    if let Some(dest) = ssa_op.dest {
        let addr = vreg_to_addr.get(&dest).copied().unwrap_or_else(|| {
            panic!("VReg {:?} not found in address mapping", dest);
        });
        slot.push(SlotVal::Int(addr));
    }

    // Add source addresses/immediates
    for src in &ssa_op.srcs {
        match src {
            Operand::VReg(vreg) => {
                let addr = vreg_to_addr.get(vreg).copied().unwrap_or_else(|| {
                    panic!("VReg {:?} not found in address mapping", vreg);
                });
                slot.push(SlotVal::Int(addr));
            }
            Operand::Imm(value) => {
                slot.push(SlotVal::Int(*value));
            }
        }
    }

    // Build reads set
    let mut reads: HashSet<i64> = HashSet::new();
    for src in &ssa_op.srcs {
        if let Operand::VReg(vreg) = src {
            if let Some(&addr) = vreg_to_addr.get(vreg) {
                if vreg.is_vector() {
                    for i in 0..VLEN as i64 {
                        reads.insert(addr + i);
                    }
                } else {
                    reads.insert(addr);
                }
            }
        }
    }

    // Build writes set
    let mut writes: HashSet<i64> = HashSet::new();
    if let Some(dest) = ssa_op.dest {
        if let Some(&addr) = vreg_to_addr.get(&dest) {
            if dest.is_vector() {
                for i in 0..VLEN as i64 {
                    writes.insert(addr + i);
                }
            } else {
                writes.insert(addr);
            }
        }
    }

    Op {
        id: ssa_op.id,
        engine: ssa_op.engine,
        slot,
        cycle: ssa_op.cycle,
        dep_ids: ssa_op.dep_ids.clone(),
        reads,
        writes,
        last_use_reads: HashSet::new(), // Will be computed by scheduler
        produces_constant: ssa_op.produces_constant,
    }
}

/// Lower all SsaOps to concrete Ops.
pub fn lower_ssa_ops(ssa_ops: &[SsaOp], vreg_to_addr: &HashMap<VReg, i64>) -> Vec<Op> {
    ssa_ops.iter().map(|op| lower_ssa_op(op, vreg_to_addr)).collect()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_builder::emit_ssa::SsaEmitContext;
    use crate::types::{Engine, Opcode};

    #[test]
    fn test_compute_live_intervals_simple() {
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
    fn test_compute_live_intervals_multiple_uses() {
        let mut emit = SsaEmitContext::new();
        let v1 = emit.emit_const(5);
        let v2 = emit.emit_alu(Opcode::Add, v1, v1);
        let _v3 = emit.emit_alu(Opcode::Mul, v1, v2);

        emit.ops[0].cycle = 0;
        emit.ops[1].cycle = 1;
        emit.ops[2].cycle = 5;

        let intervals = compute_live_intervals(&emit.ops);

        // v1: defined at 0, last used at 5
        assert_eq!(intervals[&v1], LiveInterval::new(0, 5));
        // v2: defined at 1, last used at 5
        assert_eq!(intervals[&v2], LiveInterval::new(1, 5));
    }

    #[test]
    fn test_assign_non_overlapping_reuse() {
        let mut emit = SsaEmitContext::new();
        let v1 = emit.emit_const(5);
        let _v2 = emit.emit_alu(Opcode::Add, v1, v1); // uses v1
        let v3 = emit.emit_const(6);
        let _v4 = emit.emit_alu(Opcode::Add, v3, v3); // uses v3

        // Schedule so v1 is dead before v3 is defined
        emit.ops[0].cycle = 0; // v1 defined
        emit.ops[1].cycle = 1; // v1 last used (v2 = v1 + v1)
        emit.ops[2].cycle = 3; // v3 defined (v1 already dead)
        emit.ops[3].cycle = 4; // v3 last used

        let config = AddressAssignConfig::default();
        let result = assign_addresses(&emit.ops, &config);

        // v1 and v3 should share the same address (non-overlapping)
        assert_eq!(result.vreg_to_addr[&v1], result.vreg_to_addr[&v3]);
        assert!(!result.budget_exceeded);
    }

    #[test]
    fn test_assign_overlapping_distinct() {
        let mut emit = SsaEmitContext::new();
        let v1 = emit.emit_const(5);
        let v2 = emit.emit_const(6);
        let _v3 = emit.emit_alu(Opcode::Add, v1, v2); // both live simultaneously

        emit.ops[0].cycle = 0;
        emit.ops[1].cycle = 1;
        emit.ops[2].cycle = 2; // v1 and v2 both live here

        let config = AddressAssignConfig::default();
        let result = assign_addresses(&emit.ops, &config);

        // v1 and v2 must have different addresses
        assert_ne!(result.vreg_to_addr[&v1], result.vreg_to_addr[&v2]);
        assert!(!result.budget_exceeded);
    }

    #[test]
    fn test_assign_vector_contiguous() {
        let mut emit = SsaEmitContext::new();
        let s = emit.emit_const(5);
        let v = emit.emit_vbroadcast(s);

        emit.ops[0].cycle = 0;
        emit.ops[1].cycle = 1;

        let config = AddressAssignConfig::default();
        let result = assign_addresses(&emit.ops, &config);

        // Vector should get 8 contiguous addresses
        let v_addr = result.vreg_to_addr[&v];
        assert!(v.is_vector());
        // Verify the address is assigned (just check it exists)
        assert!(v_addr >= 1);
    }

    #[test]
    fn test_peak_liveness() {
        let mut emit = SsaEmitContext::new();
        let s1 = emit.emit_const(1);
        let _v1 = emit.emit_vbroadcast(s1);
        let _s2 = emit.emit_alu(Opcode::Add, s1, s1);

        emit.ops[0].cycle = 0;
        emit.ops[1].cycle = 1;
        emit.ops[2].cycle = 2;

        let intervals = compute_live_intervals(&emit.ops);
        let peak = compute_peak_liveness(&intervals);

        // At cycle 1: s1 (1 word) + v1 (8 words being defined)
        // s1 live [0, 2], v1 live [1, 1] (never read)
        // At cycle 1: s1 live (1) + v1 defined (8) = 9
        assert_eq!(peak, 9);
    }

    #[test]
    fn test_assign_exceeds_budget() {
        let mut emit = SsaEmitContext::new();
        // Create many overlapping vectors to exceed a small budget
        let s1 = emit.emit_const(1);
        let v1 = emit.emit_vbroadcast_force(s1);
        let v2 = emit.emit_vbroadcast_force(s1);
        let v3 = emit.emit_vbroadcast_force(s1);
        let _sum = emit.emit_valu(Opcode::Add, v1, v2);
        let _sum2 = emit.emit_valu(Opcode::Add, v1, v3);

        // All scheduled in overlapping cycles
        emit.ops[0].cycle = 0;
        emit.ops[1].cycle = 1;
        emit.ops[2].cycle = 1;
        emit.ops[3].cycle = 1;
        emit.ops[4].cycle = 2;
        emit.ops[5].cycle = 2;

        let config = AddressAssignConfig {
            budget: 20, // Small budget
            start_addr: 1,
            fixed_vregs: HashMap::new(),
        };
        let result = assign_addresses(&emit.ops, &config);

        // With 3 vectors (24 words) + 1 scalar, should exceed budget of 20
        assert!(result.budget_exceeded);
    }

    #[test]
    fn test_lower_ssa_op() {
        let mut emit = SsaEmitContext::new();
        let v1 = emit.emit_const(42);
        let v2 = emit.emit_const(10);
        let _v3 = emit.emit_alu(Opcode::Add, v1, v2);

        let mut vreg_to_addr = HashMap::new();
        vreg_to_addr.insert(v1, 100);
        vreg_to_addr.insert(v2, 101);
        let dest = emit.ops[2].dest.unwrap();
        vreg_to_addr.insert(dest, 102);

        let lowered = lower_ssa_op(&emit.ops[2], &vreg_to_addr);

        assert_eq!(lowered.engine, Engine::Alu);
        assert!(lowered.writes.contains(&102));
        assert!(lowered.reads.contains(&100));
        assert!(lowered.reads.contains(&101));

        // Check slot values
        if let SlotVal::Str(op) = &lowered.slot[0] {
            assert_eq!(op, "+");
        }
        if let SlotVal::Int(dest) = &lowered.slot[1] {
            assert_eq!(*dest, 102);
        }
    }

    #[test]
    fn test_fixed_vregs() {
        let mut emit = SsaEmitContext::new();
        let v1 = emit.emit_const(5);
        let v2 = emit.emit_const(6);

        emit.ops[0].cycle = 0;
        emit.ops[1].cycle = 1;

        // Force v1 to use address 500
        let mut config = AddressAssignConfig::default();
        config.fixed_vregs.insert(v1, 500);

        let result = assign_addresses(&emit.ops, &config);

        assert_eq!(result.vreg_to_addr[&v1], 500);
        // v2 should use a different address
        assert_ne!(result.vreg_to_addr[&v2], 500);
    }

    // ========================================================================
    // Liveness Model Boundary Tests
    // ========================================================================

    #[test]
    fn test_same_cycle_reuse_boundary() {
        // LIVENESS MODEL: Same-cycle reuse
        // Intervals [0,5] and [5,10] should NOT overlap and can share storage.
        //
        // Scenario: VReg A is defined at cycle 0, last used at cycle 5.
        //           VReg B is defined at cycle 5, last used at cycle 10.
        //
        // Within cycle 5, hardware executes: read A, then write B.
        // So A's storage is freed AFTER being read but BEFORE B needs it.
        // Linear scan should expire A (end 5 <= start 5) before allocating B.

        let mut emit = SsaEmitContext::new();

        // Create VReg A: defined at cycle 0
        let a = emit.emit_const(100);
        // Create a use of A at cycle 5 (this will be an ALU op with result C)
        let c = emit.emit_alu(Opcode::Add, a, a);

        // Create VReg B: defined at cycle 5 (same cycle A is last used)
        let b = emit.emit_const(200);
        // Create a use of B at cycle 10
        let _d = emit.emit_alu(Opcode::Add, b, b);

        // Schedule: A defined at 0, C (which uses A) at 5, B defined at 5, D at 10
        emit.ops[0].cycle = 0;  // A defined
        emit.ops[1].cycle = 5;  // C = A + A (last use of A)
        emit.ops[2].cycle = 5;  // B defined (same cycle as A's last use)
        emit.ops[3].cycle = 10; // D = B + B (last use of B)

        let config = AddressAssignConfig::default();
        let result = assign_addresses(&emit.ops, &config);

        // A and B should be able to share storage (same address)
        // because A's interval [0,5] does not overlap with B's interval [5,10]
        assert_eq!(
            result.vreg_to_addr[&a], result.vreg_to_addr[&b],
            "VRegs with adjacent intervals [0,5] and [5,10] should share storage (same-cycle reuse)"
        );

        // Verify peak liveness is correct (should be 2 at most: A or B + C or D)
        // At cycle 5: A is being read (still live), B is being written, C is being written
        // But with same-cycle reuse, A expires before B is allocated
        assert!(result.peak_live <= 3, "Peak should be reasonable with same-cycle reuse");
    }

    #[test]
    fn test_same_cycle_reuse_vectors() {
        // Same test but with vectors (8 words each)
        // This is more impactful because vectors are 8x the size

        let mut emit = SsaEmitContext::new();

        // Create scalar to broadcast
        let s = emit.emit_const(1);

        // VReg A: vector defined at cycle 0
        let a = emit.emit_vbroadcast_force(s);
        // Use A at cycle 5
        let c = emit.emit_valu(Opcode::Add, a, a);

        // VReg B: vector defined at cycle 5 (same cycle A is last used)
        let b = emit.emit_vbroadcast_force(s);
        // Use B at cycle 10
        let _d = emit.emit_valu(Opcode::Add, b, b);

        emit.ops[0].cycle = 0;  // s defined
        emit.ops[1].cycle = 0;  // A defined (broadcast)
        emit.ops[2].cycle = 5;  // C = A + A (last use of A)
        emit.ops[3].cycle = 5;  // B defined (same cycle as A's last use)
        emit.ops[4].cycle = 10; // D = B + B

        let config = AddressAssignConfig::default();
        let result = assign_addresses(&emit.ops, &config);

        // Vectors A and B should share storage due to same-cycle reuse
        // This saves 8 words of scratch space
        assert_eq!(
            result.vreg_to_addr[&a], result.vreg_to_addr[&b],
            "Vector VRegs with adjacent intervals should share storage"
        );
    }

    #[test]
    fn test_peak_liveness_same_cycle_boundary() {
        // Verify that compute_peak_liveness correctly handles the boundary case
        //
        // With intervals [0,5] and [5,10]:
        // - At cycle 5, using same-cycle reuse model:
        //   - [0,5] is expiring (end 5 <= current 5)
        //   - [5,10] is being allocated
        //   - Peak should NOT count both simultaneously

        use crate::types::LiveInterval;

        let mut intervals = HashMap::new();
        let v1 = VReg::scalar(1);
        let v2 = VReg::scalar(2);

        intervals.insert(v1, LiveInterval::new(0, 5));
        intervals.insert(v2, LiveInterval::new(5, 10));

        let peak = compute_peak_liveness(&intervals);

        // With same-cycle reuse, peak should be 1 (not 2)
        // Because v1 expires before v2 is allocated at cycle 5
        assert_eq!(peak, 1, "Peak should be 1 with same-cycle reuse (intervals [0,5] and [5,10] don't overlap)");
    }

    // ========================================================================
    // Minimal Reproduction: rounds=2 Memory Write Error
    //
    // With rounds=2 and greedy scheduler, the kernel crashes at cycle 81
    // because scratch[4] contains garbage instead of the partition address.
    //
    // Timeline:
    // - Cycle 6: VReg A (partition addr) written to scratch[4]
    // - Cycle 10: VReg B (computation) written to scratch[4], OVERWRITES A
    // - Cycle 81: vstore uses scratch[4] as address → CRASH
    //
    // The bug: VReg A has interval [6, 81], VReg B has interval [10, 17]
    // These overlap, but linear scan assigns both to scratch[4].
    // ========================================================================

    /// Minimal reproduction of the rounds=2 register allocation bug.
    ///
    /// Two VRegs with overlapping intervals must get different addresses.
    #[test]
    fn test_rounds2_overlapping_intervals_bug() {
        let mut emit = SsaEmitContext::new();

        // VReg A: partition address, interval [6, 81]
        let partition_addr = emit.emit_const_force(2055);  // op 0

        // VReg B: computation result, interval [10, 17]
        let x = emit.emit_const_force(256);  // op 1
        let y = emit.emit_const_force(256);  // op 2 (force to avoid CSE)
        let computation = emit.emit_alu(Opcode::Add, x, y);  // op 3

        // VReg B used at cycle 17
        let _use_b = emit.emit_alu(Opcode::Add, computation, y);  // op 4

        // VReg A used by vstore at cycle 81
        let data = emit.emit_vbroadcast_force(x);  // op 5 (force to avoid CSE)
        emit.emit_vstore(partition_addr, data);  // op 6

        // Schedule all ops
        let n_ops = emit.ops.len();
        assert_eq!(n_ops, 7, "Expected 7 ops");

        emit.ops[0].cycle = 6;   // partition_addr
        emit.ops[1].cycle = 8;   // x
        emit.ops[2].cycle = 8;   // y
        emit.ops[3].cycle = 10;  // computation
        emit.ops[4].cycle = 17;  // use of computation
        emit.ops[5].cycle = 80;  // data
        emit.ops[6].cycle = 81;  // vstore

        let intervals = compute_live_intervals(&emit.ops);

        // Verify intervals
        assert_eq!(intervals[&partition_addr], LiveInterval::new(6, 81));
        assert_eq!(intervals[&computation], LiveInterval::new(10, 17));
        assert!(intervals[&partition_addr].overlaps(&intervals[&computation]));

        // Assign addresses - this is where the bug would manifest
        let config = AddressAssignConfig::default();
        let result = assign_addresses(&emit.ops, &config);

        // These MUST have different addresses (the bug is if they're the same)
        assert_ne!(
            result.vreg_to_addr[&partition_addr],
            result.vreg_to_addr[&computation],
            "BUG: Overlapping VRegs got the same address"
        );
    }
}
