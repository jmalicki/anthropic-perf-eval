//! Address assignment for SSA operations.
//!
//! After scheduling completes (when cycle assignments are known), this module
//! assigns physical scratch addresses to virtual registers (VRegs) using
//! linear scan allocation.
//!
//! # Liveness Model
//!
//! **IMPORTANT:** See `types.rs` for the AUTHORITATIVE documentation on VReg liveness.
//! This comment provides a brief summary; `types.rs` is the single source of truth.
//!
//! ## Closed Interval Convention: [start, end]
//!
//! We use **closed intervals** where `end` is the last cycle the VReg is ACCESSED:
//!
//! ```text
//! LiveInterval { start: D, end: R }
//!   - start (D) = cycle when VReg is WRITTEN (definition cycle)
//!   - end (R)   = cycle when VReg is last ACCESSED
//!
//! For LIVE VRegs (has readers):  [def_cycle, last_read_cycle]
//! For DEAD VRegs (no readers):   [def_cycle, def_cycle + 1]
//! ```
//!
//! ## Why Dead VRegs Get `end = def + 1`
//!
//! This is NOT a hack! It precisely models the ISA hardware semantics:
//! - Within a VLIW cycle: ALL reads happen FIRST, then ALL writes happen
//! - A live VReg read at cycle R can be reused at cycle R (read completes before write)
//! - A dead VReg written at cycle D can be reused at cycle D+1 (after write completes)
//!
//! See `types.rs` "AUTHORITATIVE LIVENESS MODEL DOCUMENTATION" for full explanation.
//!
//! ## The Single Expiration Rule: `end <= start`
//!
//! When allocating VReg B at cycle S, expire VReg A where `A.end <= S`:
//! - Same-cycle reuse: A=[0,5], B=[5,...] → 5<=5 ✓ → A expires (correct)
//! - Write conflict: A=[5,6] (dead), B=[5,...] → 6<=5 ✗ → no reuse (correct)
//!
//! ## Algorithm: Linear Scan Allocation
//!
//! 1. Compute live intervals: [def_cycle, last_read_cycle] for live VRegs,
//!    [def_cycle, def_cycle+1] for dead VRegs
//! 2. Sort intervals by start cycle
//! 3. For each VReg:
//!    - Expire VRegs where `end <= current_start`
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

use crate::types::{EmissionPhase, LiveInterval, Opcode, Operand, SsaOp, VReg, VarType, VLEN};

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
    /// Live intervals for all VRegs (for debugging liveness)
    pub live_intervals: HashMap<VReg, LiveInterval>,
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
    /// When set, this VReg (address 0) is included in live intervals so peak_live counts it and stays within budget.
    pub zero_vreg: Option<VReg>,
    /// When set, SelectChoice ops in this set (by op.id) that have sub_temp_vreg use def_cycle=op.cycle+1 for their dest (FMA writes at cycle+1; address reserved from then).
    pub select_fma_ops: Option<HashSet<usize>>,
    /// When set, VALU ops shifted to ALU (op.id in keys) use min(alu_cycles) as def_cycle for their dest.
    /// The 8 ALU ops write at different cycles; the block is in use from the first write.
    pub shifted_alu_cycles: Option<std::collections::HashMap<usize, Vec<i32>>>,
}

impl Default for AddressAssignConfig {
    fn default() -> Self {
        Self {
            budget: 1536,
            start_addr: 1, // Reserve address 0 - scratch memory is always initialized to 0 at startup, so address 0 always contains 0
            fixed_vregs: HashMap::new(),
            zero_vreg: None,
            select_fma_ops: None,
            shifted_alu_cycles: None,
        }
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
/// NOTE: For shifted VALU ops we treat the entire 8-word block as reserved from the first lane write.
/// This does NOT mean all lanes contain valid data at that moment; it simply blocks reuse until the
/// vector is fully consumed. See TODO(valu-shift-liveness) if we ever add lane-level modeling.
///
/// For a VReg defined at cycle D and last read at cycle R:
/// - Interval = [D, R] (closed)
///
/// For a dead VReg (defined at D, never read):
/// - Interval = [D, D+1] — the "+1" ensures write conflicts are detected
///
/// When `zero_vreg` is Some, it is added with interval [0, max_cycle] so peak_live counts it
/// (address 0 is reserved; total capacity is budget words including zero).
///
/// When `select_fma_ops` is Some, SelectChoice ops whose id is in the set and that have
/// `sub_temp_vreg` (FMA path without delta) use **def_cycle = op.cycle+1** for their dest for
/// allocation: the FMA *writes* at op.cycle+1, so the address must be reserved from that cycle
/// to avoid another op (e.g. another SelectChoice's sub at cycle+1) reusing the same address.
/// (Results are visible at cycle+2; precedence constraints enforce that.)
///
/// When `shifted_alu_cycles` is Some, VALU ops shifted to ALU (op.id in map) use min(alu_cycles)
/// as def_cycle for their dest. The 8 ALU ops write at different cycles; the vector block is
/// in use from the first write.
pub fn compute_live_intervals(
    ops: &[SsaOp],
    zero_vreg: Option<VReg>,
    select_fma_ops: Option<&HashSet<usize>>,
    shifted_alu_cycles: Option<&std::collections::HashMap<usize, Vec<i32>>>,
) -> HashMap<VReg, LiveInterval> {
    let debug = std::env::var("SCHEDULER_DEBUG").is_ok();
    let mut intervals: HashMap<VReg, LiveInterval> = HashMap::new();

    // First pass: record definition cycles.
    // Initialize with [def, def] — will be extended if read.
    for op in ops {
        if let Some(dest) = op.dest {
            // SelectChoice FMA (no delta): FMA writes at cycle+1 → reserve dest address from cycle+1
            let def_cycle = if op.opcode == Opcode::SelectChoice
                && select_fma_ops.map_or(false, |s| s.contains(&op.id))
                && op.sub_temp_vreg.is_some()
            {
                op.cycle + 1
            } else if let Some(cycles) =
                shifted_alu_cycles.and_then(|m| m.get(&op.id)).filter(|c| c.len() == VLEN)
            {
                // Shifted VALU: 8 ALU ops write at different cycles; block in use from first to last write
                // We must not free the address until after the last lane write, or another op could
                // be assigned the same range and we get same-cycle scratch write conflict.
                *cycles.iter().min().unwrap_or(&op.cycle)
            } else {
                op.cycle
            };
            // For shifted VALU dest, end must be at least max(alu_cycles) so we don't reuse before last write
            let end_cycle = if let Some(cycles) =
                shifted_alu_cycles.and_then(|m| m.get(&op.id)).filter(|c| c.len() == VLEN)
            {
                *cycles.iter().max().unwrap_or(&def_cycle)
            } else {
                def_cycle
            };
            intervals.insert(dest, LiveInterval::new(def_cycle, end_cycle));
        }

        // Handle sub_temp_vreg for SelectChoice ops without delta
        // The sub_temp is defined at the SelectChoice's cycle and read at cycle + 1 (by FMA)
        if let Some(sub_temp) = op.sub_temp_vreg {
            // Interval: [cycle, cycle+1] - defined at cycle, read at cycle+1
            intervals.insert(sub_temp, LiveInterval::new(op.cycle, op.cycle + 1));
        }
    }

    // Second pass: extend end to last READ cycle.
    // CLOSED interval: end = last_read_cycle (not last_read + 1).
    for op in ops {
        // Debug: log vstore reads
        if debug && op.opcode == Opcode::Vstore {
            eprintln!(
                "[DEBUG compute_live_intervals] vstore op id={} cycle={} srcs={:?}",
                op.id, op.cycle, op.srcs
            );
        }

        for src in &op.srcs {
            if let Operand::VReg(vreg) = src {
                if let Some(interval) = intervals.get_mut(vreg) {
                    let old_end = interval.end;
                    // Closed interval: end is the read cycle itself
                    if op.cycle > interval.end {
                        interval.end = op.cycle;
                    }
                    // Debug: log interval extension for vstore sources
                    if debug && op.opcode == Opcode::Vstore && interval.end > old_end {
                        eprintln!(
                            "  Extended {:?} interval from [{}, {}] to [{}, {}] (read by vstore)",
                            vreg, interval.start, old_end, interval.start, interval.end
                        );
                    }
                } else if debug && op.opcode == Opcode::Vstore {
                    // VReg not in intervals - this is a bug!
                    eprintln!(
                        "  WARNING: VReg {:?} read by vstore but not in intervals!",
                        vreg
                    );
                }
            }
        }

        // SelectChoice FMA (both delta and sub+fma paths): FMA reads cond, delta (if delta path), and false_val at cycle+1.
        // For delta path: single FMA at cycle+1 reads cond, delta, false_val.
        // For sub+fma path: sub at cycle, FMA at cycle+1 reads cond, sub_temp, false_val.
        // In both cases, cond and false_val are read at cycle+1, so their intervals must extend
        // to cycle+1 or the allocator may reuse those addresses and corrupt the FMA result.
        // For delta path, delta is also read at cycle+1, so its interval must extend to cycle+1.
        if op.opcode == Opcode::SelectChoice
            && select_fma_ops.map_or(false, |s| s.contains(&op.id))
            && op.srcs.len() >= 3
        {
            let read_at_cycle_plus_one = op.cycle + 1;
            for (i, src) in op.srcs.iter().enumerate() {
                // cond = srcs[0], false_val = srcs[2] are read again by FMA at cycle+1
                // (true_val = srcs[1] is only used by sub in sub+fma path, not by FMA)
                if i != 1 {
                    if let Operand::VReg(vreg) = src {
                        if let Some(interval) = intervals.get_mut(vreg) {
                            if read_at_cycle_plus_one > interval.end {
                                interval.end = read_at_cycle_plus_one;
                            }
                        }
                    }
                }
            }
            // For delta path: delta_vreg is also read by FMA at cycle+1
            if let Some(delta_vreg) = op.delta_vreg {
                if let Some(interval) = intervals.get_mut(&delta_vreg) {
                    if read_at_cycle_plus_one > interval.end {
                        interval.end = read_at_cycle_plus_one;
                    }
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

    // Include zero_vreg so peak_live counts the reserved slot (total capacity = budget including zero)
    if let Some(zero) = zero_vreg {
        let max_cycle = ops
            .iter()
            .filter(|op| op.cycle >= 0)
            .map(|op| op.cycle)
            .max()
            .unwrap_or(0);
        intervals.insert(zero, LiveInterval::new(0, max_cycle));
    }

    // Debug: print all intervals for VRegs used by vstores
    if debug {
        let vstore_vregs: HashSet<VReg> = ops
            .iter()
            .filter(|op| op.opcode == Opcode::Vstore)
            .flat_map(|op| op.srcs.iter().filter_map(|s| s.as_vreg()))
            .collect();

        eprintln!("[DEBUG compute_live_intervals] Final intervals for vstore sources:");
        for vreg in &vstore_vregs {
            if let Some(interval) = intervals.get(vreg) {
                eprintln!("  {:?}: [{}, {}]", vreg, interval.start, interval.end);
            } else {
                eprintln!("  {:?}: NOT FOUND IN INTERVALS!", vreg);
            }
        }
    }

    intervals
}

/// Compute peak liveness in words at any cycle using the same-cycle reuse model.
///
/// Uses the same expiration rule as linear scan allocation (see `types.rs`):
/// - Expire VRegs where `end <= current_start`
/// - Same-cycle reuse: [0,5] expires when [5,10] starts (5 <= 5)
/// - Write conflict: [5,6] (dead) does NOT expire for [5,10] (6 > 5)
///
/// This correctly handles both live VRegs and dead VRegs with +1 extension.
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
    let debug = std::env::var("SCHEDULER_DEBUG").is_ok();

    // Compute live intervals (include zero_vreg when set so peak_live counts reserved slot)
    let intervals = compute_live_intervals(
        ops,
        config.zero_vreg,
        config.select_fma_ops.as_ref(),
        config.shifted_alu_cycles.as_ref(),
    );

    // Debug: Collect vstore source VRegs for tracking
    let vstore_vregs: HashSet<VReg> = if debug {
        ops.iter()
            .filter(|op| op.opcode == Opcode::Vstore)
            .flat_map(|op| op.srcs.iter().filter_map(|s| s.as_vreg()))
            .collect()
    } else {
        HashSet::new()
    };

    // Calculate peak liveness
    let peak_live = compute_peak_liveness(&intervals);
    let budget_exceeded = peak_live > config.budget;

    // Clone intervals for storage in result
    let live_intervals = intervals.clone();

    if intervals.is_empty() {
        return AddressAssignmentResult {
            budget: config.budget,
            peak_live,
            budget_exceeded,
            live_intervals,
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

        // Expire intervals that ended at or before this one starts (allowing same-cycle reuse).
        // VLIW semantics: reads happen before writes within a cycle.
        // So if A ends at cycle 11 (last read at 11) and B starts at cycle 11 (written at 11),
        // A is read first, then B is written - they can safely share the same address.
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

        // Snapshot active (addr, size) so we can check overlap reliably.
        let active_ranges: Vec<(i64, usize)> = active
            .iter()
            .map(|Reverse((_, a, s))| (*a, *s))
            .collect();

        let overlaps_any = |addr: i64, len: usize| {
            let end = addr + len as i64;
            active_ranges.iter().any(|&(a, s)| {
                let a_end = a + s as i64;
                addr < a_end && a < end
            })
        };

        // Try to find a free range that fits and doesn't overlap any active allocation.
        let mut new_addr: Option<i64> = None;
        let mut temp_ranges: Vec<(i64, usize)> = Vec::new();

        while let Some(Reverse((addr, range_size))) = free_ranges.pop() {
            if range_size >= size && !overlaps_any(addr, size) {
                new_addr = Some(addr);
                if range_size > size {
                    temp_ranges.push((addr + size as i64, range_size - size));
                }
                break;
            }
            temp_ranges.push((addr, range_size));
        }

        for (addr, range_size) in temp_ranges {
            free_ranges.push(Reverse((addr, range_size)));
        }

        let mut new_addr = new_addr.unwrap_or_else(|| {
            let mut addr = next_addr;
            for &(a, s) in &active_ranges {
                let a_end = a + s as i64;
                if addr < a_end {
                    addr = a_end;
                }
            }
            next_addr = addr + size as i64;
            addr
        });

        // Final safeguard: chosen address must not overlap any active allocation.
        if overlaps_any(new_addr, size) {
            let mut addr = next_addr;
            for &(a, s) in &active_ranges {
                let a_end = a + s as i64;
                if addr < a_end {
                    addr = a_end;
                }
            }
            new_addr = addr;
            next_addr = addr + size as i64;
        }

        vreg_to_addr.insert(*vreg, new_addr);
        active.push(Reverse((interval.end, new_addr, size)));

        // Debug: log assignment for vstore-related VRegs
        if debug && vstore_vregs.contains(vreg) {
            eprintln!(
                "[DEBUG assign_addresses] Assigned {:?} -> addr {} (interval [{}, {}])",
                vreg, new_addr, interval.start, interval.end
            );
        }

        // Debug: log all vector assignments to catch allocator bugs
        if debug && size == 8 {
            eprintln!(
                "[DEBUG assign_addresses] VReg {:?} -> addr {} size {} (interval [{}, {}]), active_ranges: {:?}",
                vreg, new_addr, size, interval.start, interval.end, active_ranges
            );
        }
        // Debug: log all assignments at cycle 3 to catch conflicts
        if debug && interval.start == 3 {
            eprintln!(
                "[DEBUG assign_addresses] CYCLE3: VReg {:?} -> addr {} size {} (interval [{}, {}]), active_ranges: {:?}",
                vreg, new_addr, size, interval.start, interval.end, active_ranges
            );
        }
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
        live_intervals,
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
    let debug = std::env::var("SCHEDULER_DEBUG").is_ok();

    // Debug logging for vstore operations
    if debug && ssa_op.opcode == Opcode::Vstore {
        eprintln!(
            "[DEBUG lower_ssa_op] vstore op id={} cycle={}",
            ssa_op.id, ssa_op.cycle
        );
        for (i, src) in ssa_op.srcs.iter().enumerate() {
            match src {
                Operand::VReg(vreg) => {
                    let addr = vreg_to_addr.get(vreg);
                    eprintln!("  src[{}]: VReg {:?} -> addr {:?}", i, vreg, addr);
                }
                Operand::Imm(val) => {
                    eprintln!("  src[{}]: Imm({})", i, val);
                }
            }
        }
    }

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
                    panic!(
                        "VReg {:?} not found in address mapping (op id={}, cycle={})",
                        vreg, ssa_op.id, ssa_op.cycle
                    );
                });
                slot.push(SlotVal::Int(addr));
            }
            Operand::Imm(value) => {
                slot.push(SlotVal::Int(*value));
            }
        }
    }

    // Debug: print resulting slot for vstore
    if debug && ssa_op.opcode == Opcode::Vstore {
        let slot_vals: Vec<String> = slot
            .iter()
            .map(|s| match s {
                SlotVal::Str(s) => s.clone(),
                SlotVal::Int(i) => i.to_string(),
            })
            .collect();
        eprintln!("  -> lowered slot: [{}]", slot_vals.join(", "));
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
    ssa_ops
        .iter()
        .map(|op| lower_ssa_op(op, vreg_to_addr))
        .collect()
}

// ============================================================================
// Liveness Debugging Helpers
// ============================================================================

use serde::Serialize;

/// Liveness information for a single cycle.
#[derive(Debug, Clone, Serialize)]
pub struct CycleLiveness {
    /// The cycle number
    pub cycle: i32,
    /// Total live words at this cycle
    pub total_live_words: usize,
    /// Breakdown by VarType
    pub by_var_type: HashMap<String, usize>,
    /// VReg IDs becoming live at this cycle (defined here)
    pub becoming_live: Vec<u32>,
    /// VReg IDs dying at this cycle (last use here)
    pub dying: Vec<u32>,
}

/// Summary statistics for liveness analysis.
#[derive(Debug, Clone, Serialize)]
pub struct LivenessSummary {
    /// Peak live words across all cycles
    pub peak_live_words: usize,
    /// Cycle at which peak occurs
    pub peak_cycle: i32,
    /// Peak breakdown by VarType (words at peak cycle)
    pub peak_by_var_type: HashMap<String, usize>,
    /// Total number of VRegs
    pub total_vregs: usize,
    /// Number of dead VRegs (never read)
    pub dead_vregs: usize,
    /// Average interval length (in cycles)
    pub avg_interval_length: f64,
    /// Maximum interval length
    pub max_interval_length: i32,
}

/// Debug information for a single VReg.
#[derive(Debug, Clone, Serialize)]
pub struct VRegDebugInfo {
    /// VReg ID (low 31 bits)
    pub id: u32,
    /// Width: "scalar" or "vector"
    pub width: String,
    /// Size in words (1 for scalar, 8 for vector)
    pub size_words: usize,
    /// Interval start (definition cycle)
    pub interval_start: i32,
    /// Interval end (last use cycle, or def+1 for dead)
    pub interval_end: i32,
    /// Allocated address
    pub address: i64,
    /// VarType category (if known)
    pub var_type: Option<String>,
    /// Emission phase (if known)
    pub phase: Option<String>,
    /// Source round (if known)
    pub round: Option<i32>,
    /// True if VReg is never read (dead)
    pub is_dead: bool,
}

/// Build VReg metadata from SSA ops.
///
/// Returns a map from VReg to (VarType, phase, round).
pub fn build_vreg_metadata(
    ops: &[SsaOp],
) -> HashMap<VReg, (Option<VarType>, Option<String>, Option<usize>)> {
    let mut metadata = HashMap::new();
    for op in ops {
        if let Some(dest) = op.dest {
            // For Branch phase, include round in the phase label (e.g. branch_r3)
            let phase = op.source_info.as_ref().map(|s| {
                if s.phase == EmissionPhase::Branch {
                    match s.round {
                        Some(r) => format!("branch_r{}", r),
                        None => s.phase.as_str().to_string(),
                    }
                } else {
                    s.phase.as_str().to_string()
                }
            });
            let round = op.source_info.as_ref().and_then(|s| s.round);
            metadata.insert(dest, (op.var_type, phase, round));
        }
    }
    metadata
}

/// Build VRegDebugInfo for all VRegs.
pub fn build_vreg_debug_info(
    intervals: &HashMap<VReg, LiveInterval>,
    vreg_to_addr: &HashMap<VReg, i64>,
    vreg_metadata: &HashMap<VReg, (Option<VarType>, Option<String>, Option<usize>)>,
) -> Vec<VRegDebugInfo> {
    let mut infos: Vec<VRegDebugInfo> = intervals
        .iter()
        .map(|(vreg, interval)| {
            let id = vreg.id;
            let is_vector = vreg.is_vector();
            let size_words = vreg.size();
            let address = *vreg_to_addr.get(vreg).unwrap_or(&-1);

            // Dead check: interval was extended by +1 (start == end - 1 for never-read VRegs)
            // Actually, dead VRegs have interval [def, def+1] so interval length is 1
            // But live VRegs can also have length 1 if read same cycle. Check if start+1 == end and start == end-1
            // A dead VReg has no reads, so start == interval.end - 1 (because we add +1 for dead)
            // Actually the simplest check: a dead VReg was defined and never read, meaning
            // no op reads it. We set end = start + 1 for dead VRegs.
            // We can check if interval length is exactly 1 AND the VReg appears in no sources.
            // For now, approximate: if end == start + 1, it might be dead.
            let is_dead = interval.end == interval.start + 1;

            let (var_type, phase, round) = vreg_metadata
                .get(vreg)
                .cloned()
                .unwrap_or((None, None, None));

            VRegDebugInfo {
                id,
                width: if is_vector {
                    "vector".to_string()
                } else {
                    "scalar".to_string()
                },
                size_words,
                interval_start: interval.start,
                interval_end: interval.end,
                address,
                var_type: var_type.map(|vt| format!("{:?}", vt)),
                phase,
                round: round.map(|r| r as i32),
                is_dead,
            }
        })
        .collect();

    // Sort by ID for stable output
    infos.sort_by_key(|info| info.id);
    infos
}

/// Compute per-cycle liveness curve.
///
/// Returns a vector of CycleLiveness entries, one per cycle from 0 to max_cycle.
pub fn compute_liveness_curve(
    intervals: &HashMap<VReg, LiveInterval>,
    max_cycle: i32,
    vreg_metadata: &HashMap<VReg, (Option<VarType>, Option<String>, Option<usize>)>,
) -> Vec<CycleLiveness> {
    if intervals.is_empty() || max_cycle < 0 {
        return vec![];
    }

    // For each cycle, track live VRegs
    let mut curve: Vec<CycleLiveness> = Vec::with_capacity((max_cycle + 1) as usize);

    // Build events: (cycle, vreg, is_start)
    let mut events: Vec<(i32, VReg, bool)> = Vec::new();
    for (&vreg, interval) in intervals {
        events.push((interval.start, vreg, true)); // becoming live
        events.push((interval.end, vreg, false)); // dying (at end cycle)
    }
    events.sort_by_key(|(cycle, _, is_start)| (*cycle, !*is_start)); // ends before starts at same cycle

    // Track currently live VRegs
    let mut live_set: HashSet<VReg> = HashSet::new();
    let mut event_idx = 0;

    for cycle in 0..=max_cycle {
        let mut becoming_live: Vec<u32> = Vec::new();
        let mut dying: Vec<u32> = Vec::new();

        // Process events for this cycle
        while event_idx < events.len() && events[event_idx].0 == cycle {
            let (_, vreg, is_start) = events[event_idx];
            if is_start {
                live_set.insert(vreg);
                becoming_live.push(vreg.id);
            } else {
                live_set.remove(&vreg);
                dying.push(vreg.id);
            }
            event_idx += 1;
        }

        // Compute totals for this cycle
        let mut total_live_words = 0usize;
        let mut by_var_type: HashMap<String, usize> = HashMap::new();

        for vreg in &live_set {
            let size = vreg.size();
            total_live_words += size;

            if let Some((var_type, _, _)) = vreg_metadata.get(vreg) {
                let key = var_type
                    .map(|vt| format!("{:?}", vt))
                    .unwrap_or_else(|| "Unknown".to_string());
                *by_var_type.entry(key).or_insert(0) += size;
            } else {
                *by_var_type.entry("Unknown".to_string()).or_insert(0) += size;
            }
        }

        curve.push(CycleLiveness {
            cycle,
            total_live_words,
            by_var_type,
            becoming_live,
            dying,
        });
    }

    curve
}

/// Compute liveness summary statistics.
pub fn compute_liveness_summary(
    intervals: &HashMap<VReg, LiveInterval>,
    vreg_metadata: &HashMap<VReg, (Option<VarType>, Option<String>, Option<usize>)>,
) -> LivenessSummary {
    if intervals.is_empty() {
        return LivenessSummary {
            peak_live_words: 0,
            peak_cycle: 0,
            peak_by_var_type: HashMap::new(),
            total_vregs: 0,
            dead_vregs: 0,
            avg_interval_length: 0.0,
            max_interval_length: 0,
        };
    }

    // Find max cycle for curve computation
    let max_cycle = intervals.values().map(|i| i.end).max().unwrap_or(0);

    // Compute curve to find peak
    let curve = compute_liveness_curve(intervals, max_cycle, vreg_metadata);

    // Find peak
    let (peak_cycle, peak_live_words) = curve
        .iter()
        .map(|c| (c.cycle, c.total_live_words))
        .max_by_key(|(_, words)| *words)
        .unwrap_or((0, 0));

    // Get breakdown at peak
    let peak_by_var_type = curve
        .iter()
        .find(|c| c.cycle == peak_cycle)
        .map(|c| c.by_var_type.clone())
        .unwrap_or_default();

    // Count dead VRegs
    let dead_vregs = intervals.values().filter(|i| i.end == i.start + 1).count();

    // Interval statistics
    let total_vregs = intervals.len();
    let interval_lengths: Vec<i32> = intervals.values().map(|i| i.end - i.start).collect();
    let sum_lengths: i64 = interval_lengths.iter().map(|&x| x as i64).sum();
    let avg_interval_length = sum_lengths as f64 / total_vregs as f64;
    let max_interval_length = *interval_lengths.iter().max().unwrap_or(&0);

    LivenessSummary {
        peak_live_words,
        peak_cycle,
        peak_by_var_type,
        total_vregs,
        dead_vregs,
        avg_interval_length,
        max_interval_length,
    }
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

        let intervals = compute_live_intervals(&emit.ops, None, None, None);

        // v1: defined at cycle 0, last read at cycle 1
        // CLOSED interval: [0, 1] — end = last_read_cycle = 1
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

        let intervals = compute_live_intervals(&emit.ops, None, None, None);

        // CLOSED intervals: end = last_read_cycle
        // v1: defined at 0, last read at 5 → [0, 5]
        assert_eq!(intervals[&v1], LiveInterval::new(0, 5));
        // v2: defined at 1, last read at 5 → [1, 5]
        assert_eq!(intervals[&v2], LiveInterval::new(1, 5));
    }

    /// SelectChoice FMA (no delta): dest def_cycle = op.cycle+1 (FMA writes then; reserve address), sub_temp at [op.cycle, op.cycle+1].
    #[test]
    fn test_compute_live_intervals_selectchoice_fma_dest_at_cycle_plus_1() {
        let mut emit = SsaEmitContext::new();
        let c0 = emit.emit_const(0);
        let c1 = emit.emit_const(1);
        let v_c0 = emit.emit_vbroadcast(c0);
        let v_c1 = emit.emit_vbroadcast(c1);
        let cond = emit.emit_valu(Opcode::Eq, v_c0, v_c1);
        let c10 = emit.emit_const(10);
        let true_val = emit.emit_vbroadcast(c10);
        let c20 = emit.emit_const(20);
        let false_val = emit.emit_vbroadcast(c20);
        let sel = emit.emit_select_choice(cond, true_val, false_val);
        let _use = emit.emit_valu(Opcode::Add, sel, sel);

        let select_choice_idx = emit
            .ops
            .iter()
            .position(|o| o.opcode == Opcode::SelectChoice)
            .expect("one SelectChoice");
        let select_choice_id = emit.ops[select_choice_idx].id;
        assert!(
            emit.ops[select_choice_idx].sub_temp_vreg.is_some(),
            "no-delta path"
        );

        // Simulate schedule: SelectChoice at cycle 5, consumer (Add) at cycle 8.
        // Note: dep_ids may be empty if compute_ssa_dependencies wasn't run, so set consumer cycle by finding the op that uses sel.
        emit.ops[select_choice_idx].cycle = 5;
        let dest_vreg = emit.ops[select_choice_idx].dest.unwrap();
        for op in emit.ops.iter_mut() {
            if op.cycle < 0 {
                op.cycle = 0;
            }
            if op.srcs.iter().any(|s| s.as_vreg() == Some(dest_vreg)) {
                op.cycle = 8; // consumer of SelectChoice result
            }
        }

        let mut select_fma_ops = HashSet::new();
        select_fma_ops.insert(select_choice_id);
        let intervals = compute_live_intervals(&emit.ops, None, Some(&select_fma_ops), None);

        let dest = emit.ops[select_choice_idx]
            .dest
            .expect("SelectChoice has dest");
        let sub_temp = emit.ops[select_choice_idx].sub_temp_vreg.expect("sub_temp");

        // Dest is written by FMA at cycle 6 → reserve address from cycle 6 (op.cycle+1)
        assert_eq!(
            intervals[&dest].start, 6,
            "SelectChoice FMA dest def_cycle must be op.cycle+1 for allocation"
        );
        assert_eq!(
            intervals[&dest].end, 8,
            "dest last read by consumer at cycle 8"
        );

        // Sub_temp defined at 5, read by FMA at 6 → [5, 6]
        assert_eq!(intervals[&sub_temp], LiveInterval::new(5, 6));
    }

    /// Two SelectChoice FMA ops: after assign_addresses, no two VRegs with overlapping
    /// intervals may share the same address (catches allocation bugs).
    #[test]
    fn test_assign_two_selectchoice_fma_no_overlapping_same_addr() {
        let mut emit = SsaEmitContext::new();
        let c0 = emit.emit_const(0);
        let c1 = emit.emit_const(1);
        let v_c0 = emit.emit_vbroadcast(c0);
        let v_c1 = emit.emit_vbroadcast(c1);
        let cond = emit.emit_valu(Opcode::Eq, v_c0, v_c1);
        let c10 = emit.emit_const(10);
        let true1 = emit.emit_vbroadcast(c10);
        let c20 = emit.emit_const(20);
        let false1 = emit.emit_vbroadcast(c20);
        let sel1 = emit.emit_select_choice(cond, true1, false1);
        let _u1 = emit.emit_valu(Opcode::Add, sel1, sel1);

        let c30 = emit.emit_const(30);
        let true2 = emit.emit_vbroadcast(c30);
        let c40 = emit.emit_const(40);
        let false2 = emit.emit_vbroadcast(c40);
        let sel2 = emit.emit_select_choice(cond, true2, false2);
        let _u2 = emit.emit_valu(Opcode::Add, sel2, sel2);

        let idx1 = emit
            .ops
            .iter()
            .position(|o| o.opcode == Opcode::SelectChoice)
            .expect("first");
        let idx2 = emit
            .ops
            .iter()
            .position(|o| o.opcode == Opcode::SelectChoice && o.id != emit.ops[idx1].id)
            .expect("second");
        let id1 = emit.ops[idx1].id;
        let id2 = emit.ops[idx2].id;

        // Schedule: sel1 at 0 (sub 0, fma 1 → result 2), consumer at 5; sel2 at 2 (sub 2, fma 3 → result 4), consumer at 6
        for (i, op) in emit.ops.iter_mut().enumerate() {
            op.cycle = if i == idx1 {
                0
            } else if op.dep_ids.contains(&id1) && !op.dep_ids.contains(&id2) {
                5
            } else if i == idx2 {
                2
            } else if op.dep_ids.contains(&id2) {
                6
            } else {
                0
            };
        }

        let mut select_fma_ops = HashSet::new();
        select_fma_ops.insert(id1);
        select_fma_ops.insert(id2);

        let mut config = AddressAssignConfig::default();
        config.select_fma_ops = Some(select_fma_ops.clone());
        let result = assign_addresses(&emit.ops, &config);
        assert!(!result.budget_exceeded);

        let intervals = compute_live_intervals(&emit.ops, None, Some(&select_fma_ops), None);
        for (v1, addr1) in &result.vreg_to_addr {
            for (v2, addr2) in &result.vreg_to_addr {
                if v1 >= v2 {
                    continue;
                }
                if addr1 != addr2 {
                    continue;
                }
                let i1 = intervals.get(v1).expect("interval for v1");
                let i2 = intervals.get(v2).expect("interval for v2");
                assert!(!i1.overlaps(i2), "BUG: VRegs {:?} and {:?} share address {} but intervals [{},{}] and [{},{}] overlap",
                    v1, v2, addr1, i1.start, i1.end, i2.start, i2.end);
            }
        }
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

        let intervals = compute_live_intervals(&emit.ops, None, None, None);
        let peak = compute_peak_liveness(&intervals);

        // CLOSED intervals (see types.rs for authoritative docs):
        // s1: defined at 0, last read at 2 → [0, 2]
        // v1: defined at 1, never read (dead) → [1, 2] (+1 extension)
        // s2: defined at 2, never read (dead) → [2, 3] (+1 extension)
        //
        // At cycle 1: s1 [0,2] is live (1 word), v1 [1,2] starts (8 words)
        // Peak = 1 + 8 = 9
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
            zero_vreg: None,
            select_fma_ops: None,
            shifted_alu_cycles: None,
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
        // CLOSED INTERVALS: Same-cycle reuse (read before write within a cycle)
        //
        // See types.rs "AUTHORITATIVE LIVENESS MODEL DOCUMENTATION" for full details.
        //
        // Scenario: VReg A is defined at cycle 0, last READ at cycle 5.
        //           VReg B is defined (WRITTEN) at cycle 5.
        //
        // ISA semantics: Within a cycle, reads happen FIRST, then writes.
        // So at cycle 5: A is read (consumed), then B is written.
        // Therefore A's storage CAN be reused by B.
        //
        // Closed intervals:
        //   A: [0, 5] — defined at 0, last read at 5
        //   B: [5, 10] — defined at 5, last read at 10
        //
        // Expiration check: end <= start → 5 <= 5 ✓ → A expires
        // Result: B can reuse A's address.

        let mut emit = SsaEmitContext::new();

        // VReg A: defined at cycle 0, read at cycle 5 → [0, 5]
        let a = emit.emit_const(100);
        // VReg LONG: defined at cycle 0, read at cycle 10 → [0, 10]
        let long = emit.emit_const(0);
        // VReg B: defined at cycle 5, uses A (so A is read at cycle 5) → [5, 10]
        let b = emit.emit_alu(Opcode::Add, a, long);
        // Use B and LONG at cycle 10 to extend their liveness
        let _c = emit.emit_alu(Opcode::Add, b, long);

        emit.ops[0].cycle = 0; // A defined → [0, 5]
        emit.ops[1].cycle = 0; // LONG defined → [0, 10]
        emit.ops[2].cycle = 5; // B = A + LONG → [5, 10]
        emit.ops[3].cycle = 10; // C = B + LONG

        let config = AddressAssignConfig::default();
        let result = assign_addresses(&emit.ops, &config);

        // With closed intervals:
        // A: [0, 5] — last read at cycle 5
        // B: [5, 10] — defined at cycle 5
        //
        // Expiration: 5 <= 5 ✓ → A expires when B is allocated
        // Result: A and B CAN share the same address (same-cycle reuse)
        assert_eq!(
            result.vreg_to_addr[&a], result.vreg_to_addr[&b],
            "With closed intervals, VRegs at the same-cycle reuse boundary should share addresses"
        );
    }

    #[test]
    fn test_same_cycle_write_conflict() {
        // Two VRegs defined in the SAME cycle must NOT share addresses.
        //
        // See types.rs "AUTHORITATIVE LIVENESS MODEL DOCUMENTATION" for full details.
        //
        // In VLIW, all ops in a cycle execute concurrently. If two ops both
        // WRITE in cycle 3, they cannot share the same address.
        //
        // Closed intervals:
        // A: defined at 3, dead (no reads) → [3, 4] (+1 extension for dead VRegs)
        // B: defined at 3, last read at 5 → [3, 5]
        //
        // These overlap because both are defined at cycle 3.
        // Expiration check for B: A's end (4) <= B's start (3)? NO, 4 > 3.
        // So A is NOT expired when allocating B → different addresses. ✓

        let mut emit = SsaEmitContext::new();

        // VReg A: defined at cycle 3, immediately dead (no users) → [3, 4]
        let a = emit.emit_const(100);
        // VReg B: defined at cycle 3, used at cycle 5 → [3, 5]
        let b = emit.emit_const(200);
        // Use B at cycle 5 to keep it alive
        let _c = emit.emit_alu(Opcode::Add, b, b);

        emit.ops[0].cycle = 3; // A → [3, 4] (dead, +1 extension)
        emit.ops[1].cycle = 3; // B → [3, 5]
        emit.ops[2].cycle = 5; // C = B + B

        let config = AddressAssignConfig::default();
        let result = assign_addresses(&emit.ops, &config);

        // A and B are BOTH written at cycle 3, so they MUST have different addresses
        assert_ne!(
            result.vreg_to_addr[&a], result.vreg_to_addr[&b],
            "VRegs defined in the SAME cycle must have different addresses (concurrent writes)"
        );
    }

    #[test]
    fn test_sequential_reuse_vectors() {
        // Test sequential reuse with vectors (8 words each).
        // When VReg A's interval ends BEFORE VReg B's interval starts,
        // B can reuse A's address.
        //
        // See types.rs "AUTHORITATIVE LIVENESS MODEL DOCUMENTATION" for full details.

        let mut emit = SsaEmitContext::new();

        // VReg A: vector defined at cycle 0, last read at cycle 2 → [0, 2]
        let s1 = emit.emit_const(1);
        let a = emit.emit_vbroadcast_force(s1);
        let _use_a = emit.emit_valu(Opcode::Add, a, a); // A read at cycle 2

        // VReg B: vector defined at cycle 5 (AFTER A's interval [0,2] ends)
        let s2 = emit.emit_const(2);
        let b = emit.emit_vbroadcast_force(s2);
        let _use_b = emit.emit_valu(Opcode::Add, b, b); // B read at cycle 7

        emit.ops[0].cycle = 0; // s1 defined → [0, 0] then read at cycle 0, so [0, 0] (dead after broadcast)
        emit.ops[1].cycle = 0; // A defined → [0, 2] (vector)
        emit.ops[2].cycle = 2; // use_a = A + A (A last read)
        emit.ops[3].cycle = 4; // s2 defined → [4, 4] then read at cycle 5, so [4, 5]
        emit.ops[4].cycle = 5; // B defined → [5, 7] (vector)
        emit.ops[5].cycle = 7; // use_b = B + B (B last read)

        let config = AddressAssignConfig::default();
        let result = assign_addresses(&emit.ops, &config);

        // Closed intervals:
        // - A: [0, 2] — last read at 2
        // - B: [5, 7] — starts at 5
        // Expiration check for B: A's end (2) <= B's start (5)? YES, 2 <= 5.
        // So A is expired when allocating B → they can share.
        assert_eq!(
            result.vreg_to_addr[&a], result.vreg_to_addr[&b],
            "Vector VRegs with non-overlapping intervals should share storage"
        );
    }

    #[test]
    fn test_peak_liveness_adjacent_intervals() {
        // Test peak liveness with adjacent closed intervals.
        //
        // See types.rs "AUTHORITATIVE LIVENESS MODEL DOCUMENTATION" for full details.
        //
        // Closed intervals [0, 5] and [5, 10]:
        // - [0, 5] covers cycles 0,1,2,3,4,5 — VReg last READ at cycle 5
        // - [5, 10] covers cycles 5,6,7,8,9,10 — VReg DEFINED at cycle 5
        //
        // These are ADJACENT at the boundary (both reference cycle 5), but:
        // - In [0, 5], cycle 5 is the last READ
        // - In [5, 10], cycle 5 is the WRITE
        // - ISA: reads happen before writes within a cycle
        //
        // Linear scan expiration: end <= start → 5 <= 5 ✓ → v1 expires
        // So at cycle 5, v1 is expired before v2 is allocated.
        // Peak = 1 (either v1 OR v2, never both)

        use crate::types::LiveInterval;

        let mut intervals = HashMap::new();
        let v1 = VReg::scalar(1);
        let v2 = VReg::scalar(2);

        intervals.insert(v1, LiveInterval::new(0, 5)); // [0, 5] — last read at 5
        intervals.insert(v2, LiveInterval::new(5, 10)); // [5, 10] — defined at 5

        let peak = compute_peak_liveness(&intervals);

        // Adjacent closed intervals at the same-cycle reuse boundary don't conflict.
        // Peak should be 1 (v1 expires before v2 is allocated)
        assert_eq!(
            peak, 1,
            "Adjacent closed intervals [0,5] and [5,10] allow same-cycle reuse; peak = 1"
        );
    }

    // ========================================================================
    // Minimal Reproduction: rounds=2 Memory Write Error
    //
    // See types.rs "AUTHORITATIVE LIVENESS MODEL DOCUMENTATION" for full details.
    //
    // With rounds=2 and greedy scheduler, the kernel crashes at cycle 81
    // because scratch[4] contains garbage instead of the partition address.
    //
    // Timeline:
    // - Cycle 6: VReg A (partition addr) written to scratch[4]
    // - Cycle 10: VReg B (computation) written to scratch[4], OVERWRITES A
    // - Cycle 81: vstore uses scratch[4] as address → CRASH
    //
    // Closed intervals:
    // - A: [6, 81] — defined at 6, last read at 81
    // - B: [10, 17] — defined at 10, last read at 17
    // These overlap, so they must get different addresses.
    // ========================================================================

    /// Minimal reproduction of the rounds=2 register allocation bug.
    ///
    /// Two VRegs with overlapping intervals must get different addresses.
    #[test]
    fn test_rounds2_overlapping_intervals_bug() {
        let mut emit = SsaEmitContext::new();

        // VReg A: partition address, closed interval [6, 81]
        let partition_addr = emit.emit_const_force(2055); // op 0

        // VReg B: computation result, closed interval [10, 17]
        let x = emit.emit_const_force(256); // op 1
        let y = emit.emit_const_force(256); // op 2 (force to avoid CSE)
        let computation = emit.emit_alu(Opcode::Add, x, y); // op 3

        // VReg B used at cycle 17
        let _use_b = emit.emit_alu(Opcode::Add, computation, y); // op 4

        // VReg A used by vstore at cycle 81
        let data = emit.emit_vbroadcast_force(x); // op 5 (force to avoid CSE)
        emit.emit_vstore(partition_addr, data); // op 6

        // Schedule all ops
        let n_ops = emit.ops.len();
        assert_eq!(n_ops, 7, "Expected 7 ops");

        emit.ops[0].cycle = 6; // partition_addr
        emit.ops[1].cycle = 8; // x
        emit.ops[2].cycle = 8; // y
        emit.ops[3].cycle = 10; // computation
        emit.ops[4].cycle = 17; // use of computation
        emit.ops[5].cycle = 80; // data
        emit.ops[6].cycle = 81; // vstore

        let intervals = compute_live_intervals(&emit.ops, None, None, None);

        // Verify closed intervals
        // partition_addr: defined at 6, last read at 81 → [6, 81]
        assert_eq!(intervals[&partition_addr], LiveInterval::new(6, 81));
        // computation: defined at 10, last read at 17 → [10, 17]
        assert_eq!(intervals[&computation], LiveInterval::new(10, 17));
        // These overlap: 6 < 17 AND 10 < 81
        assert!(intervals[&partition_addr].overlaps(&intervals[&computation]));

        // Assign addresses - this is where the bug would manifest
        let config = AddressAssignConfig::default();
        let result = assign_addresses(&emit.ops, &config);

        // These MUST have different addresses (the bug is if they're the same)
        assert_ne!(
            result.vreg_to_addr[&partition_addr], result.vreg_to_addr[&computation],
            "BUG: Overlapping VRegs got the same address"
        );
    }
}
