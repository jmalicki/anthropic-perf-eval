//! Register allocator for scratch memory.
//!
//! # DEPRECATED
//!
//! **This module is deprecated.** It is used only by the legacy non-SSA kernel builder
//! (`build_tick_tock_kernel`). New code should use the SSA path with
//! [`crate::address_assign`] instead, which:
//!
//! - Works with virtual registers (VRegs) instead of pre-assigned addresses
//! - Correctly implements same-cycle reuse semantics
//! - Is used by the recommended `build_tick_tock_kernel_ssa` function
//!
//! ## Known Issues
//!
//! This module has inconsistent liveness semantics compared to `address_assign`:
//!
//! - **Overlap check uses `>=`** instead of `<`, preventing same-cycle reuse
//! - **Peak calculation uses +1 offset** for death events, which is more conservative
//!
//! These issues mean intervals `[0,5]` and `[5,10]` are considered overlapping here,
//! but non-overlapping in `address_assign`. The hardware actually allows same-cycle
//! reuse (reads before writes within a cycle), so `address_assign` is correct.
//!
//! ---
//!
//! This module implements post-scheduling address rewriting based on variable liveness.
//! After scheduling completes (when cycle assignments are known), we can compute exact
//! live ranges and reuse addresses when their live ranges don't overlap.
//!
//! ## Algorithm: Linear Scan Allocation
//!
//! 1. Build live intervals from scheduled ops (def_cycle from writes, last_use_cycle from reads)
//! 2. Sort intervals by def_cycle
//! 3. For each interval:
//!    - Expire intervals that ended before current def_cycle
//!    - Add expired addresses back to free pool
//!    - Allocate from free pool (8-aligned for vectors)
//!    - Record mapping: old_addr -> new_addr
//! 4. Rewrite all addresses in ops using mapping

// Allow deprecated usage within this module (internal consistency)
#![allow(deprecated)]

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::types::{Op, SlotVal, VLEN};

/// A live interval representing the lifetime of a scratch address.
#[deprecated(
    since = "0.1.0",
    note = "Use types::LiveInterval with address_assign module instead"
)]
#[derive(Clone, Debug)]
pub struct LiveInterval {
    /// Original address assigned during kernel building
    pub addr: i64,
    /// Cycle when first written (defines the value)
    pub def_cycle: i32,
    /// Cycle of last read (value no longer needed after this)
    pub last_use_cycle: i32,
    /// Size: 1 for scalar, 8 for vector
    pub size: usize,
    /// True for fixed addresses (ZERO_REG, inputs, outputs) that cannot be remapped
    pub is_fixed: bool,
}

impl LiveInterval {
    /// Check if this interval is still live at the given cycle
    pub fn is_live_at(&self, cycle: i32) -> bool {
        cycle >= self.def_cycle && cycle <= self.last_use_cycle
    }

    /// Check if this interval overlaps with another
    pub fn overlaps(&self, other: &LiveInterval) -> bool {
        // Intervals overlap if neither ends before the other starts
        !(self.last_use_cycle < other.def_cycle || other.last_use_cycle < self.def_cycle)
    }
}

/// Result of register allocation
#[derive(Debug, Default)]
pub struct AllocationResult {
    /// Mapping from old address to new address
    pub addr_mapping: HashMap<i64, i64>,
    /// Peak simultaneous live addresses (inherent to schedule, cannot be reduced)
    pub peak_live: usize,
    /// Highest original address used (before allocation)
    pub max_addr_before: i64,
    /// Highest new address used (after allocation)
    pub max_addr_after: i64,
    /// Number of addresses that were remapped
    pub remapped_count: usize,
    /// Number of fixed addresses (not remapped)
    pub fixed_count: usize,
    /// True if allocation exceeded the scratch budget
    pub budget_exceeded: bool,
    /// The scratch budget that was used
    pub budget: usize,
}

/// Scratch register allocator using linear scan algorithm.
#[deprecated(since = "0.1.0", note = "Use address_assign::assign_addresses instead")]
pub struct ScratchRegisterAllocator {
    /// All live intervals
    intervals: Vec<LiveInterval>,
    /// Address mapping (old -> new)
    addr_mapping: HashMap<i64, i64>,
    /// Fixed addresses that should not be remapped
    fixed_addrs: HashSet<i64>,
}

impl ScratchRegisterAllocator {
    /// Create a new allocator
    pub fn new() -> Self {
        Self {
            intervals: Vec::new(),
            addr_mapping: HashMap::new(),
            fixed_addrs: HashSet::new(),
        }
    }

    /// Mark an address as fixed (will not be remapped)
    pub fn mark_fixed(&mut self, addr: i64) {
        self.fixed_addrs.insert(addr);
    }

    /// Mark a range of addresses as fixed (for vectors)
    pub fn mark_fixed_range(&mut self, base: i64, size: usize) {
        for i in 0..size as i64 {
            self.fixed_addrs.insert(base + i);
        }
    }

    /// Build live intervals from scheduled ops.
    ///
    /// This extracts def_cycle (when written) and last_use_cycle (when last read)
    /// for each scratch address used by the operations.
    ///
    /// Addresses that are read before being written are considered "input" addresses
    /// (initialized by the machine before kernel execution) and are marked as fixed.
    pub fn build_live_intervals(&mut self, ops: &[Op]) {
        // Sort ops by cycle for correct input detection
        let mut sorted_ops: Vec<&Op> = ops.iter().filter(|op| op.cycle >= 0).collect();
        sorted_ops.sort_by_key(|op| op.cycle);

        // Track first write and last read for each address
        let mut first_write: HashMap<i64, i32> = HashMap::new();
        let mut last_read: HashMap<i64, i32> = HashMap::new();

        // Track which addresses are vector bases
        let mut vector_bases: HashSet<i64> = HashSet::new();

        // Track which addresses are read before being written (input addresses)
        let mut input_addresses: HashSet<i64> = HashSet::new();

        for op in sorted_ops {
            let cycle = op.cycle;

            // Check for reads of addresses that haven't been written yet
            for &addr in &op.reads {
                if !first_write.contains_key(&addr) {
                    input_addresses.insert(addr);
                }
                last_read
                    .entry(addr)
                    .and_modify(|c| *c = (*c).max(cycle))
                    .or_insert(cycle);
            }

            // Track writes (defines)
            for &addr in &op.writes {
                first_write.entry(addr).or_insert(cycle);

                // If this is a vector op and this is the dest address, mark as vector base
                if is_vector_address(op, addr) {
                    vector_bases.insert(addr);
                }
            }
        }

        // Mark input addresses as fixed (they're initialized externally)
        for addr in input_addresses {
            self.fixed_addrs.insert(addr);
        }

        // Build intervals
        self.intervals.clear();
        let mut processed_vector_bases: HashSet<i64> = HashSet::new();
        let mut processed_vector_members: HashSet<i64> = HashSet::new();

        // First, process all vector bases
        for &base in &vector_bases {
            if processed_vector_bases.contains(&base) {
                continue;
            }
            processed_vector_bases.insert(base);

            // Mark all 8 addresses as members of this vector
            for i in 0..VLEN as i64 {
                processed_vector_members.insert(base + i);
            }

            // Find earliest write across all 8 addresses
            let def_cycle = (0..VLEN as i64)
                .filter_map(|i| first_write.get(&(base + i)))
                .min()
                .copied()
                .unwrap_or(0);

            // Find latest read across all 8 addresses
            let last_use = (0..VLEN as i64)
                .filter_map(|i| last_read.get(&(base + i)))
                .max()
                .copied()
                .unwrap_or(def_cycle);

            let is_fixed = (0..VLEN as i64).any(|i| self.fixed_addrs.contains(&(base + i)));

            self.intervals.push(LiveInterval {
                addr: base,
                def_cycle,
                last_use_cycle: last_use,
                size: VLEN,
                is_fixed,
            });
        }

        // Then, process scalar addresses (excluding vector members)
        for (&addr, &def_cycle) in &first_write {
            // Skip if this address is part of a vector
            if processed_vector_members.contains(&addr) {
                continue;
            }

            let last_use = last_read.get(&addr).copied().unwrap_or(def_cycle);
            let is_fixed = self.fixed_addrs.contains(&addr);

            self.intervals.push(LiveInterval {
                addr,
                def_cycle,
                last_use_cycle: last_use,
                size: 1,
                is_fixed,
            });
        }

        // Sort by def_cycle for linear scan
        self.intervals.sort_by_key(|iv| (iv.def_cycle, iv.addr));
    }

    /// Allocate physical addresses using linear scan algorithm.
    ///
    /// This uses a unified allocation approach where all intervals (scalar and vector)
    /// are allocated from a single address space starting at 1. Vector allocations
    /// must be 8-aligned.
    ///
    /// Returns the mapping from old addresses to new addresses.
    pub fn allocate(&mut self, budget: usize) -> AllocationResult {
        self.addr_mapping.clear();

        // Calculate peak before allocation
        let peak_before = self.calculate_peak(&self.intervals);
        let budget_exceeded = peak_before > budget;

        let fixed_count = self.intervals.iter().filter(|iv| iv.is_fixed).count();

        // Handle fixed addresses first
        for iv in &self.intervals {
            if iv.is_fixed {
                if iv.size == VLEN {
                    for i in 0..VLEN as i64 {
                        self.addr_mapping.insert(iv.addr + i, iv.addr + i);
                    }
                } else {
                    self.addr_mapping.insert(iv.addr, iv.addr);
                }
            }
        }

        // Collect non-fixed intervals sorted by def_cycle
        let mut intervals: Vec<LiveInterval> = self
            .intervals
            .iter()
            .filter(|iv| !iv.is_fixed)
            .cloned()
            .collect();
        intervals.sort_by_key(|iv| (iv.def_cycle, iv.addr));

        // Track active intervals: (end_cycle, allocated_base_addr, size)
        let mut active: BinaryHeap<Reverse<(i32, i64, usize)>> = BinaryHeap::new();

        // Free address ranges: (base_addr, size)
        // Use BinaryHeap with Reverse to prefer lower addresses
        let mut free_ranges: BinaryHeap<Reverse<(i64, usize)>> = BinaryHeap::new();

        // Next fresh address to allocate
        let mut next_addr: i64 = 1; // Start after ZERO_REG

        for iv in &intervals {
            // Expire intervals that ended before this one starts
            while let Some(&Reverse((end_cycle, freed_addr, freed_size))) = active.peek() {
                if end_cycle < iv.def_cycle {
                    active.pop();
                    free_ranges.push(Reverse((freed_addr, freed_size)));
                } else {
                    break;
                }
            }

            // Try to find a free range that fits this interval's size
            let mut new_addr: Option<i64> = None;
            let mut temp_ranges: Vec<(i64, usize)> = Vec::new();

            while let Some(Reverse((addr, size))) = free_ranges.pop() {
                if size >= iv.size {
                    // Found a suitable range
                    new_addr = Some(addr);
                    // If there's leftover space, put it back
                    if size > iv.size {
                        temp_ranges.push((addr + iv.size as i64, size - iv.size));
                    }
                    break;
                } else {
                    temp_ranges.push((addr, size));
                }
            }

            // Put back ranges we couldn't use
            for (addr, size) in temp_ranges {
                free_ranges.push(Reverse((addr, size)));
            }

            // If no free range found, allocate fresh
            let new_addr = new_addr.unwrap_or_else(|| {
                let addr = next_addr;
                next_addr += iv.size as i64;
                addr
            });

            // Record mapping for all addresses in this interval
            for i in 0..iv.size as i64 {
                self.addr_mapping.insert(iv.addr + i, new_addr + i);
            }

            // Add to active set
            active.push(Reverse((iv.last_use_cycle, new_addr, iv.size)));
        }

        // Calculate highest addresses used
        let max_addr_before = self
            .intervals
            .iter()
            .map(|iv| iv.addr + (iv.size as i64 - 1))
            .max()
            .unwrap_or(0);

        let max_addr_after = self.addr_mapping.values().copied().max().unwrap_or(0);

        let remapped_count = self
            .addr_mapping
            .iter()
            .filter(|(old, new)| old != new)
            .count();

        AllocationResult {
            addr_mapping: self.addr_mapping.clone(),
            peak_live: peak_before,
            max_addr_before,
            max_addr_after,
            remapped_count,
            fixed_count,
            budget_exceeded,
            budget,
        }
    }

    /// Calculate peak scratch usage from intervals
    fn calculate_peak(&self, intervals: &[LiveInterval]) -> usize {
        if intervals.is_empty() {
            return 0;
        }

        // Collect all events (start, end)
        let mut events: Vec<(i32, i64)> = Vec::new(); // (cycle, delta)

        for iv in intervals {
            events.push((iv.def_cycle, iv.size as i64));
            events.push((iv.last_use_cycle + 1, -(iv.size as i64)));
        }

        events.sort_by_key(|(cycle, _)| *cycle);

        let mut current = 0i64;
        let mut peak = 0i64;

        for (_, delta) in events {
            current += delta;
            peak = peak.max(current);
        }

        peak as usize
    }

    /// Get the address mapping
    pub fn get_mapping(&self) -> &HashMap<i64, i64> {
        &self.addr_mapping
    }
}

/// Check if an address is used as part of a vector operation
fn is_vector_address(op: &Op, addr: i64) -> bool {
    use crate::types::Engine;

    // VALU ops write to 8 consecutive addresses
    if op.engine == Engine::Valu || op.engine == Engine::ShiftedValu {
        // Check if this is a base address (the dest in slot[1])
        if let Some(SlotVal::Int(dest)) = op.slot.get(1) {
            if addr == *dest {
                return true;
            }
        }
    }

    // Check for vload/vstore
    if let Some(SlotVal::Str(opcode)) = op.slot.get(0) {
        if opcode == "vload" || opcode == "vstore" {
            if let Some(SlotVal::Int(dest)) = op.slot.get(1) {
                if addr == *dest {
                    return true;
                }
            }
        }
    }

    false
}

/// Rewrite addresses in operations using the mapping.
///
/// This updates all address references in the ops:
/// - writes set
/// - reads set
/// - last_use_reads set
/// - slot values (only ADDRESS operands, not literal values)
#[deprecated(
    since = "0.1.0",
    note = "Use the SSA path with address_assign::lower_ssa_ops instead"
)]
pub fn rewrite_addresses(ops: &mut [Op], mapping: &HashMap<i64, i64>) {
    for op in ops {
        // Rewrite writes
        let new_writes: HashSet<i64> = op
            .writes
            .iter()
            .map(|&addr| *mapping.get(&addr).unwrap_or(&addr))
            .collect();
        op.writes = new_writes;

        // Rewrite reads
        let new_reads: HashSet<i64> = op
            .reads
            .iter()
            .map(|&addr| *mapping.get(&addr).unwrap_or(&addr))
            .collect();
        op.reads = new_reads;

        // Rewrite last_use_reads
        let new_last_use_reads: HashSet<i64> = op
            .last_use_reads
            .iter()
            .map(|&addr| *mapping.get(&addr).unwrap_or(&addr))
            .collect();
        op.last_use_reads = new_last_use_reads;

        // Rewrite slot values - only address operands, not literal values
        // Get opcode to determine which slots are addresses vs literals
        let opcode: String = match op.slot.get(0) {
            Some(SlotVal::Str(s)) => s.clone(),
            _ => String::new(),
        };

        // Skip slot 0 (opcode), process remaining slots
        for idx in 1..op.slot.len() {
            if let SlotVal::Int(val) = &mut op.slot[idx] {
                // Check if this slot should be treated as a literal (not remapped)
                let is_literal = match opcode.as_str() {
                    // const: slot[2] is literal value, not address
                    "const" => idx == 2,
                    // add_imm: slot[3] is immediate literal
                    "add_imm" => idx == 3,
                    // pause, halt, jump, cond_jump have literal arguments
                    "pause" | "halt" | "jump" | "cond_jump" => true,
                    // Everything else: all slots are addresses
                    _ => false,
                };

                if !is_literal {
                    if let Some(&new_addr) = mapping.get(val) {
                        *val = new_addr;
                    }
                }
            }
        }
    }
}

/// Run register allocation on scheduled ops.
///
/// This is the main entry point for post-scheduling address optimization.
/// It builds live intervals, allocates physical addresses, and rewrites the ops.
///
/// # Arguments
/// * `ops` - Scheduled operations (must have cycle assignments)
/// * `budget` - Scratch memory budget (e.g., 1536). If peak_live exceeds this, budget_exceeded is set.
/// * `fixed_addrs` - Addresses that should not be remapped (e.g., ZERO_REG, inputs, outputs)
///
/// # Returns
/// Allocation result with statistics. Check `budget_exceeded` to see if allocation is valid.
#[deprecated(
    since = "0.1.0",
    note = "Use the SSA path with address_assign::assign_addresses instead"
)]
pub fn allocate_registers(
    ops: &mut [Op],
    budget: usize,
    fixed_addrs: &HashSet<i64>,
) -> AllocationResult {
    let mut allocator = ScratchRegisterAllocator::new();

    // Mark fixed addresses
    for &addr in fixed_addrs {
        allocator.mark_fixed(addr);
    }

    // Build intervals and allocate
    allocator.build_live_intervals(ops);
    let result = allocator.allocate(budget);

    // Rewrite addresses
    rewrite_addresses(ops, &result.addr_mapping);

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Engine, MAX_SLOT_SIZE};
    use arrayvec::ArrayVec;

    fn make_op(id: usize, cycle: i32, reads: &[i64], writes: &[i64]) -> Op {
        Op {
            id,
            engine: Engine::Alu,
            slot: ArrayVec::new(),
            cycle,
            dep_ids: HashSet::new(),
            reads: reads.iter().copied().collect(),
            writes: writes.iter().copied().collect(),
            last_use_reads: HashSet::new(),
            produces_constant: None,
        }
    }

    #[test]
    fn test_live_interval_overlap() {
        let iv1 = LiveInterval {
            addr: 1,
            def_cycle: 0,
            last_use_cycle: 5,
            size: 1,
            is_fixed: false,
        };
        let iv2 = LiveInterval {
            addr: 2,
            def_cycle: 3,
            last_use_cycle: 8,
            size: 1,
            is_fixed: false,
        };
        let iv3 = LiveInterval {
            addr: 3,
            def_cycle: 6,
            last_use_cycle: 10,
            size: 1,
            is_fixed: false,
        };

        assert!(iv1.overlaps(&iv2)); // 3-5 overlap
        assert!(!iv1.overlaps(&iv3)); // iv1 ends at 5, iv3 starts at 6
        assert!(iv2.overlaps(&iv3)); // 6-8 overlap
    }

    #[test]
    fn test_non_overlapping_reuse() {
        // Two variables with non-overlapping lifetimes should share an address
        let mut ops = vec![
            make_op(0, 0, &[], &[100]), // write to 100 at cycle 0
            make_op(1, 1, &[100], &[]), // read 100 at cycle 1 (last use)
            make_op(2, 2, &[], &[200]), // write to 200 at cycle 2
            make_op(3, 3, &[200], &[]), // read 200 at cycle 3 (last use)
        ];
        ops[1].last_use_reads.insert(100);
        ops[3].last_use_reads.insert(200);

        let fixed: HashSet<i64> = HashSet::new();
        let result = allocate_registers(&mut ops, 1536, &fixed);

        // Both should be mapped to the same address since they don't overlap
        let addr_100 = *result.addr_mapping.get(&100).unwrap();
        let addr_200 = *result.addr_mapping.get(&200).unwrap();
        assert_eq!(
            addr_100, addr_200,
            "Non-overlapping intervals should reuse addresses"
        );
        assert!(!result.budget_exceeded);
    }

    #[test]
    fn test_overlapping_no_reuse() {
        // Two variables with overlapping lifetimes cannot share an address
        let mut ops = vec![
            make_op(0, 0, &[], &[100]),      // write to 100 at cycle 0
            make_op(1, 1, &[], &[200]),      // write to 200 at cycle 1
            make_op(2, 2, &[100, 200], &[]), // read both at cycle 2
        ];
        ops[2].last_use_reads.insert(100);
        ops[2].last_use_reads.insert(200);

        let fixed: HashSet<i64> = HashSet::new();
        let result = allocate_registers(&mut ops, 1536, &fixed);

        let addr_100 = *result.addr_mapping.get(&100).unwrap();
        let addr_200 = *result.addr_mapping.get(&200).unwrap();
        assert_ne!(
            addr_100, addr_200,
            "Overlapping intervals must use different addresses"
        );
        assert!(!result.budget_exceeded);
    }

    #[test]
    fn test_fixed_addresses() {
        let mut ops = vec![
            make_op(0, 0, &[0], &[100]), // read ZERO_REG, write 100
            make_op(1, 1, &[100], &[]),  // read 100
        ];
        ops[1].last_use_reads.insert(100);

        let mut fixed: HashSet<i64> = HashSet::new();
        fixed.insert(0); // ZERO_REG is fixed

        let result = allocate_registers(&mut ops, 1536, &fixed);

        // ZERO_REG should map to itself (input address detected automatically)
        // Note: fixed_count will be 1 because address 0 is read before written
        assert_eq!(*result.addr_mapping.get(&0).unwrap_or(&0), 0);
        assert!(!result.budget_exceeded);
    }

    #[test]
    fn test_address_rewrite() {
        let mut slot: ArrayVec<SlotVal, MAX_SLOT_SIZE> = ArrayVec::new();
        slot.push(SlotVal::Str("+".to_string()));
        slot.push(SlotVal::Int(100)); // dest
        slot.push(SlotVal::Int(50)); // src1
        slot.push(SlotVal::Int(60)); // src2

        let mut ops = vec![Op {
            id: 0,
            engine: Engine::Alu,
            slot,
            cycle: 0,
            dep_ids: HashSet::new(),
            reads: [50, 60].into_iter().collect(),
            writes: [100].into_iter().collect(),
            last_use_reads: HashSet::new(),
            produces_constant: None,
        }];

        let mut mapping: HashMap<i64, i64> = HashMap::new();
        mapping.insert(100, 1);
        mapping.insert(50, 2);
        mapping.insert(60, 3);

        rewrite_addresses(&mut ops, &mapping);

        assert!(ops[0].writes.contains(&1));
        assert!(ops[0].reads.contains(&2));
        assert!(ops[0].reads.contains(&3));

        // Check slot values were rewritten
        if let SlotVal::Int(dest) = &ops[0].slot[1] {
            assert_eq!(*dest, 1);
        }
        if let SlotVal::Int(src1) = &ops[0].slot[2] {
            assert_eq!(*src1, 2);
        }
        if let SlotVal::Int(src2) = &ops[0].slot[3] {
            assert_eq!(*src2, 3);
        }
    }
}
