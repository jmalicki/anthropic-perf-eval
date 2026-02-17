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
//! ## The Expiration Rule
//!
//! When allocating VReg B at cycle S, expire VReg A when:
//! - `A.end < S`, or
//! - `A.end == S` and B is not a "point" interval [S, S], or
//! - `A.end == S` and A was not a point interval (so A was read then B can write; same-cycle reuse).
//! Point intervals [C, C] (defined and read at C) are never expired when allocating another VReg
//! that starts at C, so two VRegs both read at cycle C get different addresses.
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
//!
//! ## Liveness special cases (effective write/read cycles)
//!
//! The general rule is: **interval = [first_write_cycle, last_read_cycle]** where write/read
//! cycles are the actual machine cycles. Some ops have effective writes or reads at cycles
//! that differ from `op.cycle` or from `op.srcs`; those are handled explicitly below. Each
//! special case is justified by "effective write/read at cycle X" that isn't obvious from
//! the SSA graph alone.
//!
//! | Scenario | Where | Effective behavior | Reason |
//! |----------|-------|--------------------|--------|
//! | **SelectChoice FMA** | First pass | Dest: `def_cycle = op.cycle + 1`; sub_temp: `[op.cycle, op.cycle+1]` | FMA writes at cycle+1; sub is read by FMA at cycle+1. Not in srcs as a normal consumer. |
//! | **Shifted VALU** | First pass | Dest: `start = min(alu_cycles)`, `end = max(alu_cycles)` | One SSA op expands to 8 ALU ops at (possibly) 8 cycles; block in use from first to last write. |
//! | **SelectChoice FMA reads** | Second pass | Extend cond, **true_val**, false_val, delta to `op.cycle + 1` | Sub (at cycle+1) reads true_val; FMA (at cycle+1) reads cond, sub result, false_val. All src VRegs must be extended so interval includes the cycle they are read. |
//! | **AluDerive** | Separate pass | Extend left/right producer's interval to derive op's cycle | ConstChoice AluDerive has no VReg srcs; the actual reads are implicit at the derive op's cycle. |
//! | **Dead VRegs** | Third pass | Extend `[def, def]` to `[def, def+1]` | Prevents same-cycle write conflict (see types.rs). |
//! | **Vector group** | Separate pass | Synthesize group interval from lane intervals and consumer reads; remove lane VRegs | Structural: lanes share one block; not a single "op" in the usual sense. |
//!
//! Vstore value: The value (srcs[1]) is read at the vstore's cycle C. So "last read" = C is
//! correct; no extra extension is applied. Offset–FMA ordering is enforced by the scheduler
//! (multiply_add must run after its offset producer), not by extending the value's interval.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

use rangemap::RangeSet;
use crate::types::{EmissionPhase, LiveInterval, Opcode, Operand, SsaOp, VReg, VarType, VLEN};

/// Result of address assignment.
#[derive(Clone, Debug, Default)]
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
    /// SelectChoice ops in this set (by op.id) that have sub_temp_vreg use def_cycle=op.cycle+1 for their dest (FMA writes at cycle+1; address reserved from then).
    pub select_fma_ops: HashSet<usize>,
    /// VALU ops shifted to ALU (op.id in keys) use min(alu_cycles) as def_cycle for their dest.
    /// The 8 ALU ops write at different cycles; the block is in use from the first write.
    /// Presence in map = shifted; require len==VLEN or panic. Absence = not shifted.
    pub shifted_alu_cycles: std::collections::HashMap<usize, Vec<i32>>,
    /// ConstChoice ops using ALU-derive: op_id → AluDeriveInfo (left_op_id, right_op_id, opcode).
    /// The AluDerive op reads the dest VRegs of its left/right producers at its cycle,
    /// so those producers' live intervals must extend to at least the AluDerive op's cycle.
    pub const_alu_derive_ops: HashMap<usize, crate::types::AluDeriveInfo>,
    /// Maps vector group alias VRegs to their 8 scalar lane VRegs (for scatter loads).
    /// The group is allocated as an 8-word block; lane VRegs derive addresses as base+i.
    pub vector_group_lanes: HashMap<VReg, Vec<VReg>>,
    /// Used when detecting precedence violations (interval end < start) to log producer op id.
    pub vreg_def: HashMap<VReg, usize>,
    /// Op IDs that are CSE group consumers (use shared sub_temp; skip per-op sub_temp).
    pub cse_consumer_op_ids: HashSet<usize>,
    /// CSE shared sub_temp intervals: (sub_temp_vreg, sub_cycle, max_fma_cycle+1) per active group.
    pub cse_shared_sub_temp_intervals: Vec<(VReg, i32, i32)>,
}

impl Default for AddressAssignConfig {
    fn default() -> Self {
        Self {
            budget: 1536,
            start_addr: 1, // Reserve address 0 - scratch memory is always initialized to 0 at startup
            fixed_vregs: HashMap::new(),
            zero_vreg: None,
            select_fma_ops: HashSet::new(),
            shifted_alu_cycles: HashMap::new(),
            const_alu_derive_ops: HashMap::new(),
            vector_group_lanes: HashMap::new(),
            vreg_def: HashMap::new(),
            cse_consumer_op_ids: HashSet::new(),
            cse_shared_sub_temp_intervals: Vec::new(),
        }
    }
}

impl AddressAssignConfig {
    /// Computes live intervals for all VRegs from the given SSA ops using this config's
    /// zero_vreg, select_fma_ops, shifted_alu_cycles, const_alu_derive_ops, vector_group_lanes,
    /// vreg_def, cse_consumer_op_ids, and cse_shared_sub_temp_intervals.
    pub fn compute_live_intervals(&self, ops: &[SsaOp]) -> Result<HashMap<VReg, LiveInterval>, String> {
        compute_live_intervals(self, ops)
    }
}

// ---------------------------------------------------------------------------
// Liveness traits: effective write/read cycles per op (see plan "Generalize address-assign liveness")
// ---------------------------------------------------------------------------

/// Context passed to liveness behavior; used to resolve effective def/use cycles.
pub struct LivenessContext<'a> {
    pub select_fma_ops: &'a HashSet<usize>,
    pub shifted_alu_cycles: &'a std::collections::HashMap<usize, Vec<i32>>,
    pub const_alu_derive_ops: &'a HashMap<usize, crate::types::AluDeriveInfo>,
    /// Op IDs in CSE groups (use shared sub_temp; skip per-op sub_temp in aux_defs).
    pub cse_consumer_op_ids: &'a HashSet<usize>,
    pub ops: &'a [SsaOp],
}

/// When does this op write its dest (and any aux VRegs like sub_temp)?
pub trait LivenessDef {
    /// (first_write_cycle, last_write_cycle) for the op's dest, if any.
    fn dest_write_interval(&self, op: &SsaOp, ctx: &LivenessContext<'_>) -> Option<(i32, i32)>;
    /// Auxiliary defs (e.g. sub_temp for SelectChoice FMA): (VReg, start, end).
    fn aux_defs(&self, op: &SsaOp, ctx: &LivenessContext<'_>) -> Vec<(VReg, i32, i32)>;
}

/// At which cycles does this op read each VReg (including implicit reads)?
///
/// **Contract:** Implementations must report every VReg that this op (or its lowering) reads,
/// at the exact cycle where that read happens. For ops that lower to multiple instructions
/// (e.g. SelectChoice FMA → sub at cycle+1, then FMA at cycle+1), every read by any of those
/// instructions must be included. Missing a read leaves that VReg's interval too short and
/// causes use-def failures when the op is scheduled far from the producer.
pub trait LivenessUse {
    /// (VReg, read_cycle) for each use. Must include every read at the cycle it occurs.
    fn use_cycles(&self, op: &SsaOp, ctx: &LivenessContext<'_>) -> Vec<(VReg, i32)>;
}

/// Dispatcher: implements LivenessDef and LivenessUse by branching on opcode and context.
pub struct OpLiveness<'a> {
    pub op: &'a SsaOp,
    pub ctx: &'a LivenessContext<'a>,
}

impl LivenessDef for OpLiveness<'_> {
    fn dest_write_interval(&self, op: &SsaOp, ctx: &LivenessContext<'_>) -> Option<(i32, i32)> {
        op.dest?; // only called when dest is some; return type allows None for consistency
        // SelectChoice FMA (no delta): FMA writes at cycle+1
        let (first, last) = if op.opcode == Opcode::SelectChoice
            && ctx.select_fma_ops.contains(&op.id)
            && op.sub_temp_vreg.is_some()
        {
            (op.cycle + 1, op.cycle + 1)
        } else if let Some(cycles) = ctx.shifted_alu_cycles.get(&op.id) {
            // Validated at start of compute_live_intervals; debug_assert for sanity.
            debug_assert_eq!(cycles.len(), VLEN, "shifted VALU op {} shifted_alu_cycles len {}", op.id, cycles.len());
            let min_c = *cycles.iter().min().unwrap_or(&op.cycle);
            let max_c = *cycles.iter().max().unwrap_or(&op.cycle);
            (min_c, max_c)
        } else {
            (op.cycle, op.cycle)
        };
        Some((first, last))
    }

    fn aux_defs(&self, op: &SsaOp, ctx: &LivenessContext<'_>) -> Vec<(VReg, i32, i32)> {
        let mut out = Vec::new();
        if let Some(sub_temp) = op.sub_temp_vreg {
            if op.opcode == Opcode::SelectChoice
                && ctx.select_fma_ops.contains(&op.id)
                && !ctx.cse_consumer_op_ids.contains(&op.id)
            {
                out.push((sub_temp, op.cycle, op.cycle + 1));
            }
        }
        out
    }
}

impl LivenessUse for OpLiveness<'_> {
    fn use_cycles(&self, op: &SsaOp, ctx: &LivenessContext<'_>) -> Vec<(VReg, i32)> {
        let mut out = Vec::new();
        // SelectChoice FMA: cond, true_val, false_val, delta (and sub_temp) read at cycle+1.
        // The lowered FMA path emits sub = true_val - false_val at cycle+1, so true_val is read
        // and its live interval must extend to cycle+1. Previously we skipped index 1 (true_val),
        // which caused use-def violations when the schedule placed SelectChoice at a cycle not
        // covered by true_val's interval (e.g. interval [8,8] vs read at cycle 20).
        if op.opcode == Opcode::SelectChoice
            && ctx.select_fma_ops.contains(&op.id)
            && op.srcs.len() >= 3
        {
            let read_at = op.cycle + 1;
            for src in &op.srcs {
                if let Operand::VReg(v) = src {
                    out.push((*v, read_at));
                }
            }
            if let Some(delta_vreg) = op.delta_vreg {
                out.push((delta_vreg, read_at));
            }
            if let Some(sub_temp) = op.sub_temp_vreg {
                if !ctx.cse_consumer_op_ids.contains(&op.id) {
                    out.push((sub_temp, read_at));
                }
            }
            return out;
        }
        // Default: each src VReg read at op.cycle
        for src in &op.srcs {
            if let Operand::VReg(v) = src {
                out.push((*v, op.cycle));
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------

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
/// When `select_fma_ops` contains an op id, SelectChoice ops whose id is in the set and that have
/// `sub_temp_vreg` (FMA path without delta) use **def_cycle = op.cycle+1** for their dest for
/// allocation: the FMA *writes* at op.cycle+1, so the address must be reserved from that cycle
/// to avoid another op (e.g. another SelectChoice's sub at cycle+1) reusing the same address.
/// (Results are visible at cycle+2; precedence constraints enforce that.)
///
/// When `shifted_alu_cycles` contains an op id, VALU ops shifted to ALU use min(alu_cycles)
/// as def_cycle for their dest. The 8 ALU ops write at different cycles; the vector block is
/// in use from the first write.
pub fn compute_live_intervals(
    config: &AddressAssignConfig,
    ops: &[SsaOp],
) -> Result<HashMap<VReg, LiveInterval>, String> {
    let debug = std::env::var("SCHEDULER_DEBUG").is_ok();

    // Validate shifted_alu_cycles before we use it in LivenessDef (avoids panic in dest_write_interval).
    for (op_id, cycles) in &config.shifted_alu_cycles {
        if cycles.len() != VLEN {
            return Err(format!(
                "BUG: shifted VALU op {} has shifted_alu_cycles len {} (expected {})",
                op_id, cycles.len(), VLEN
            ));
        }
    }

    let mut intervals: HashMap<VReg, LiveInterval> = HashMap::new();

    let ctx = LivenessContext {
        select_fma_ops: &config.select_fma_ops,
        shifted_alu_cycles: &config.shifted_alu_cycles,
        const_alu_derive_ops: &config.const_alu_derive_ops,
        cse_consumer_op_ids: &config.cse_consumer_op_ids,
        ops,
    };

    // First pass: record definition cycles (LivenessDef).
    for op in ops {
        let op_liv = OpLiveness { op, ctx: &ctx };
        if let Some(dest) = op.dest {
            if let Some((first, last)) = op_liv.dest_write_interval(op, &ctx) {
                intervals.insert(dest, LiveInterval::new(first, last));
            }
        }
        for (vreg, s, e) in op_liv.aux_defs(op, &ctx) {
            intervals.insert(vreg, LiveInterval::new(s, e));
        }
    }

    // Second pass: extend end to last READ cycle (LivenessUse).
    for op in ops {
        if debug && op.opcode == Opcode::Vstore {
            eprintln!(
                "[DEBUG compute_live_intervals] vstore op id={} cycle={} srcs={:?}",
                op.id, op.cycle, op.srcs
            );
        }

        let op_liv = OpLiveness { op, ctx: &ctx };
        for (vreg, read_cycle) in op_liv.use_cycles(op, &ctx) {
            if let Some(interval) = intervals.get_mut(&vreg) {
                let old_end = interval.end;
                if read_cycle > interval.end {
                    interval.end = read_cycle;
                }
                if interval.end < interval.start {
                    let producer_op_id = config.vreg_def.get(&vreg).copied();
                    eprintln!(
                        "PRECEDENCE VIOLATION (interval end < start): op id={} opcode={:?} cycle={} \
                         reads value VReg {:?} whose interval is [{}, {}] (start=producer cycle). \
                         Producer op id: {:?}. Schedule has consumer before producer.",
                        op.id, op.opcode, op.cycle, vreg, interval.start, interval.end, producer_op_id
                    );
                    return Err(format!(
                        "BUG: Schedule has op {} (consumer, cycle {}) before producer of {:?} (interval start {}). \
                         Fix the scheduler so cycle[consumer] >= cycle[producer] + 1.",
                        op.id, op.cycle, vreg, interval.start
                    ));
                }
                if debug && op.opcode == Opcode::Vstore && interval.end > old_end {
                    eprintln!(
                        "  Extended {:?} interval from [{}, {}] to [{}, {}] (read by vstore)",
                        vreg, interval.start, old_end, interval.start, interval.end
                    );
                }
            } else if debug && op.opcode == Opcode::Vstore {
                eprintln!(
                    "  WARNING: VReg {:?} read by vstore but not in intervals!",
                    vreg
                );
            }
        }
    }

    // AluDerive pass: extend producer VRegs' intervals to cover AluDerive reads.
    // When a ConstChoice op uses AluDerive (e.g., C = A + B), the lowered instruction
    // reads from A's and B's addresses at C's cycle. These reads aren't in the SSA graph
    // (ConstChoice has no VReg srcs), so we must extend A's and B's intervals explicitly.
    let id_to_info: HashMap<usize, (usize, Option<VReg>)> = ops
        .iter()
        .enumerate()
        .map(|(i, op)| (op.id, (i, op.dest)))
        .collect();

    for (&derive_op_id, info) in &config.const_alu_derive_ops {
        let left_op_id = info.left_op_id;
        let right_op_id = info.right_op_id;
        let derive_cycle = id_to_info
            .get(&derive_op_id)
            .map(|&(i, _)| ops[i].cycle)
            .unwrap_or(-1);
        if derive_cycle < 0 {
            continue;
        }

        // Extend left producer's dest VReg interval to cover the AluDerive read
        if let Some(&(_, Some(left_vreg))) = id_to_info.get(&left_op_id) {
            if let Some(interval) = intervals.get_mut(&left_vreg) {
                if derive_cycle > interval.end {
                    if debug {
                        eprintln!(
                            "[AluDerive] Extending {:?} (producer op {}) interval end {} -> {} for AluDerive op {}",
                            left_vreg, left_op_id, interval.end, derive_cycle, derive_op_id
                        );
                    }
                    interval.end = derive_cycle;
                }
            }
        }

        // Extend right producer's dest VReg interval to cover the AluDerive read
        if let Some(&(_, Some(right_vreg))) = id_to_info.get(&right_op_id) {
            if let Some(interval) = intervals.get_mut(&right_vreg) {
                if derive_cycle > interval.end {
                    if debug {
                        eprintln!(
                            "[AluDerive] Extending {:?} (producer op {}) interval end {} -> {} for AluDerive op {}",
                            right_vreg, right_op_id, interval.end, derive_cycle, derive_op_id
                        );
                    }
                    interval.end = derive_cycle;
                }
            }
        }
    }

    // Third pass: extend dead VRegs (start == end, never read) to [start, start+1].
    // This prevents same-cycle write conflicts. See types.rs for explanation.
    // Only extend when the VReg is never read: if it is read at the same cycle as def ([C, C]),
    // we must not extend, so that two such VRegs (e.g. two vstore value sources at cycle C)
    // remain distinct for allocation and get different addresses.
    let used_vregs: HashSet<VReg> = ops
        .iter()
        .flat_map(|op| op.srcs.iter().filter_map(|s| s.as_vreg()))
        .collect();
    for (vreg, interval) in intervals.iter_mut() {
        if interval.start == interval.end && !used_vregs.contains(vreg) {
            // Dead VReg: written but never read. Extend by 1 to prevent
            // same-cycle write conflicts with other VRegs.
            interval.end = interval.start + 1;
        }
    }

    // CSE shared sub_temp intervals: add synthetic intervals for shared SUBs.
    for &(vreg, start, end) in &config.cse_shared_sub_temp_intervals {
        intervals.insert(vreg, LiveInterval::new(start, end));
    }

    // Vector group pass: synthesize group interval from lane writes and consumer reads.
    // The group alias VReg occupies 8 contiguous words; lanes share this allocation.
    // After this pass, lane VRegs are removed from intervals (they get addresses derived
    // as base+i from the group's allocation in assign_addresses).
    //
    // Two kinds of groups:
    //   Write group (fresh_vector_group): lanes ARE op destinations (have intervals),
    //     group key is NOT an op destination (no existing interval).
    //   Read group (decompose_vector): lanes are NOT op destinations (no intervals),
    //     group key IS an op destination (has existing interval from its definer).
    //
    // For read groups, we must preserve the existing interval and extend it to cover
    // all ops that read lane VRegs. Lane VRegs lack intervals so we scan ops directly.
    //
    // TODO (future): Have decompose_vector end the vector VReg's liveness and assign
    // independent intervals to each new scalar (lane) VReg, so we can more tightly
    // pack chains of ALU ops that consume lanes at different times. See emit_ssa.rs
    // decompose_vector doc comment and docs/UNIFY_SCRATCH_LIVENESS_PLAN.md.
    for (group_vreg, lane_vregs) in &config.vector_group_lanes {
        // Existing interval for the group key (present for read groups / decomposition)
        let existing_start = intervals.get(group_vreg).map(|i| i.start);
        let existing_end = intervals.get(group_vreg).map(|i| i.end);

        // Lane intervals (present for write groups, absent for read groups)
        let lane_start = lane_vregs
                .iter()
                .filter_map(|lv| intervals.get(lv).map(|i| i.start))
            .min();
        let lane_end = lane_vregs
                .iter()
                .filter_map(|lv| intervals.get(lv).map(|i| i.end))
            .max();

        // Consumer reads of the group alias VReg (ops that use the vector group)
        let consumer_end = ops
                .iter()
                .filter(|op| {
                    op.srcs
                        .iter()
                        .any(|s| matches!(s, Operand::VReg(v) if *v == *group_vreg))
                })
                .map(|op| op.cycle)
            .max();

        // Lane VReg uses: ops that read any lane VReg directly.
        // Critical for read groups where lane VRegs have no intervals.
        let lane_vreg_set: HashSet<VReg> = lane_vregs.iter().copied().collect();
        let lane_use_end = ops
                .iter()
                .filter(|op| {
                    op.srcs
                        .iter()
                        .any(|s| matches!(s, Operand::VReg(v) if lane_vreg_set.contains(v)))
                })
                .map(|op| op.cycle)
            .max();

        // Combine all sources for start and end
        let start = [lane_start, existing_start]
                .iter()
                .flatten()
                .copied()
                .min()
            .unwrap_or(0);

        let end = [lane_end, consumer_end, lane_use_end, existing_end]
                .iter()
                .flatten()
                .copied()
                .max()
                .unwrap_or(start)
            .max(start + 1);

        intervals.insert(*group_vreg, LiveInterval::new(start, end));

        // Remove lane VRegs — they share the group's 8-word allocation
        for lv in lane_vregs {
            intervals.remove(lv);
        }

        if debug {
            eprintln!(
                "[DEBUG compute_live_intervals] Vector group {:?}: [{}, {}] (8 lanes, existing=[{:?}, {:?}])",
                group_vreg, start, end, existing_start, existing_end
            );
        }
    }

    // Include zero_vreg so peak_live counts the reserved slot (total capacity = budget including zero)
    if let Some(zero) = config.zero_vreg {
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

    // A4 sanity check: for VRegs with at least one consumer, end >= max_consumer_cycle.
    let mut max_consumer_cycle: HashMap<VReg, i32> = HashMap::new();
    for op in ops {
        let op_liv = OpLiveness { op, ctx: &ctx };
        for (vreg, read_cycle) in op_liv.use_cycles(op, &ctx) {
            max_consumer_cycle
                .entry(vreg)
                .and_modify(|m| *m = (*m).max(read_cycle))
                .or_insert(read_cycle);
        }
    }
    for (vreg, interval) in &intervals {
        if let Some(&max_c) = max_consumer_cycle.get(vreg) {
            assert!(
                interval.end >= max_c,
                "A4 sanity: VReg {:?} has consumer at cycle {} but interval end is {}",
                vreg,
                max_c,
                interval.end
            );
        }
    }

    Ok(intervals)
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
/// Assignment result with VReg -> address mapping and statistics, or an error string.
pub fn assign_addresses(ops: &[SsaOp], config: &AddressAssignConfig) -> Result<AddressAssignmentResult, String> {
    let debug = std::env::var("SCHEDULER_DEBUG").is_ok();

    // Compute live intervals (include zero_vreg when set so peak_live counts reserved slot)
    let intervals = config.compute_live_intervals(ops)?;

    // Verify use-def: every (vreg, read_cycle) reported by use_cycles must have an interval that
    // includes read_cycle. This uses the same source of truth as interval computation, so we
    // validate the actual read cycles (e.g. op.cycle+1 for SelectChoice FMA), not just op.cycle.
    let ctx = LivenessContext {
        select_fma_ops: &config.select_fma_ops,
        shifted_alu_cycles: &config.shifted_alu_cycles,
        const_alu_derive_ops: &config.const_alu_derive_ops,
        cse_consumer_op_ids: &config.cse_consumer_op_ids,
        ops,
    };
    for op in ops {
        let op_liv = OpLiveness { op, ctx: &ctx };
        for (vreg, read_cycle) in op_liv.use_cycles(op, &ctx) {
            let interval = intervals.get(&vreg).or_else(|| {
                config
                    .vector_group_lanes
                    .iter()
                    .find(|(_, lanes)| lanes.contains(&vreg))
                    .and_then(|(group, _)| intervals.get(group))
            });
            if let Some(interval) = interval {
                if interval.start > read_cycle || read_cycle > interval.end {
                    return Err(format!(
                        "Use-def violation: VReg {:?} read at cycle {} but interval is [{}, {}] (op id={}, opcode={:?}). \
                         The VReg's live interval does not include the read cycle - this indicates a bug in interval computation or use_cycles.",
                        vreg, read_cycle, interval.start, interval.end, op.id, op.opcode
                    ));
                }
            } else {
                return Err(format!(
                    "Use-def violation: VReg {:?} read at cycle {} but has no interval (op id={}, opcode={:?}). \
                     This VReg was never defined or its interval was not computed - indicates an SSA violation or interval computation bug.",
                    vreg, read_cycle, op.id, op.opcode
                ));
            }
        }
    }

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
        return Ok(AddressAssignmentResult {
            budget: config.budget,
            peak_live,
            budget_exceeded,
            live_intervals,
            ..Default::default()
        });
    }

    // Collect intervals with their VRegs, sorted by start cycle
    let mut sorted: Vec<(VReg, LiveInterval)> = intervals.into_iter().collect();
    sorted.sort_by_key(|(_, interval)| (interval.start, interval.end));

    // Handle fixed VRegs and build the set of used address ranges (for O(V²) gap-based allocation).
    let mut vreg_to_addr: HashMap<VReg, i64> = HashMap::new();
    let mut used_range_set: RangeSet<i64> = RangeSet::new();

    for (vreg, addr) in &config.fixed_vregs {
        vreg_to_addr.insert(*vreg, *addr);
        used_range_set.insert(*addr..addr + vreg.size() as i64);
    }

    // Active intervals: (end_cycle, allocated_base_addr, size, was_point_interval)
    // was_point_interval: true if the allocated VReg had interval [C, C]. We need this to avoid
    // expiring a point interval when allocating a non-point [C, C+1] that starts at the same cycle
    // (otherwise we'd assign the same address to both).
    let mut active: BinaryHeap<Reverse<(i32, i64, usize, bool)>> = BinaryHeap::new();

    // Freed ranges (addr, size) available for reuse; try these before gap search so we reuse slots.
    let mut free_ranges: BinaryHeap<Reverse<(i64, usize)>> = BinaryHeap::new();

    // Scratch addresses (every word) we've assigned to intervals that start at the current cycle.
    // When we need a range for another interval that starts at this cycle, no address in that range
    // may be in this set (same-cycle writers must not overlap).
    let mut current_cycle: Option<i32> = None;
    let mut addrs_written_this_cycle: HashSet<i64> = HashSet::new();

    // Next fresh address to allocate (advanced when we skip past used ranges).
    let mut next_addr = config.start_addr;
    while let Some(range) = used_range_set.get(&next_addr) {
        next_addr = range.end;
    }

    for (vreg, interval) in &sorted {
        // When we move to a new start cycle, clear the set of addresses written this cycle.
        if current_cycle != Some(interval.start) {
            current_cycle = Some(interval.start);
            addrs_written_this_cycle.clear();
        }

        // Fixed VRegs: track as active and add their range to same-cycle set so we don't assign that range to another writer at this cycle.
        if config.fixed_vregs.contains_key(vreg) {
            let addr = config.fixed_vregs[vreg];
            let size = vreg.size();
            for a in addr..addr + size as i64 {
                addrs_written_this_cycle.insert(a);
            }
            let point = interval.start == interval.end;
            active.push(Reverse((interval.end, addr, size, point)));
            continue;
        }

        let size = vreg.size();

        // Expire intervals that ended at or before this one starts (allowing same-cycle reuse).
        // VLIW semantics: reads happen before writes within a cycle.
        // So if A ends at cycle 11 (last read at 11) and B starts at cycle 11 (written at 11),
        // A is read first, then B is written - they can safely share the same address.
        // Exception: when THIS interval is a "point" [C, C], only expire if freed ended before C.
        // When the FREED allocation was a point interval ending at C, do NOT expire when
        // allocating a non-point [C, C+k] — otherwise we'd give the new VReg the same address
        // as the point (both live at cycle C).
        while let Some(&Reverse((end_cycle, freed_addr, freed_size, freed_was_point))) = active.peek() {
            let expire = if interval.start == interval.end {
                end_cycle < interval.start
            } else {
                end_cycle < interval.start
                    || (end_cycle == interval.start && !freed_was_point)
            };
            if expire {
                assert!(
                    end_cycle < interval.start
                        || (end_cycle == interval.start && !freed_was_point),
                    "T1 violation: expired end={} for alloc start={}, freed_was_point={}",
                    end_cycle,
                    interval.start,
                    freed_was_point
                );
                active.pop();
                if !is_in_fixed_range(freed_addr, freed_size, &config.fixed_vregs) {
                    used_range_set.remove(freed_addr..freed_addr + freed_size as i64);
                    free_ranges.push(Reverse((freed_addr, freed_size)));
                }
            } else {
                break;
            }
        }

        // Build temp_used = used_range_set ∪ addrs_written_this_cycle for overlap checks and gap search.
        let mut temp_used = used_range_set.clone();
        for &a in &addrs_written_this_cycle {
            temp_used.insert(a..a + 1);
        }
        let budget_i64 = config.budget as i64;
        let end_addr = config.start_addr + budget_i64; // valid range is [start_addr, end_addr)
        let size_i64 = size as i64;

        // Try freed ranges first (same order as old allocator: reuse freed slots).
        let mut new_addr = None;
        let mut temp_ranges = Vec::new();
        while let Some(Reverse((addr, range_size))) = free_ranges.pop() {
            if range_size >= size && !temp_used.overlaps(&(addr..addr + size_i64)) {
                new_addr = Some(addr);
                if range_size > size {
                    temp_ranges.push((addr + size_i64, range_size - size));
                }
                break;
            }
            temp_ranges.push((addr, range_size));
        }
        for (addr, range_size) in temp_ranges {
            free_ranges.push(Reverse((addr, range_size)));
        }

        // Else first-fit from gaps (rangemap: O(V) per allocation).
        let new_addr = new_addr.unwrap_or_else(|| {
            let gap_outer = config.start_addr..end_addr;
            temp_used
                .gaps(&gap_outer)
                .find(|gap| gap.end - gap.start >= size_i64)
                .map(|gap| gap.start)
                .unwrap_or_else(|| {
                    // Fallback: no gap large enough; build sorted used list and find first fit.
                    let mut active_ranges: Vec<(i64, usize)> = used_range_set
                        .iter()
                        .map(|r| (r.start, (r.end - r.start) as usize))
                        .collect();
                    // Sentinel so we never allocate past end_addr (first_gap_of_length sees end_addr as used).
                    active_ranges.push((end_addr, 1));
                    let used = used_addresses_sorted(&active_ranges, &addrs_written_this_cycle);
                    first_gap_of_length(&used, config.start_addr, size)
                })
        });

        next_addr = new_addr + size as i64;
        used_range_set.insert(new_addr..new_addr + size as i64);

        for a in new_addr..new_addr + size as i64 {
            addrs_written_this_cycle.insert(a);
        }
        vreg_to_addr.insert(*vreg, new_addr);
        let point = interval.start == interval.end;
        active.push(Reverse((interval.end, new_addr, size, point)));

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
                "[DEBUG assign_addresses] VReg {:?} -> addr {} size {} (interval [{}, {}])",
                vreg, new_addr, size, interval.start, interval.end
            );
        }
        // Debug: log all assignments at cycle 3 to catch conflicts
        if debug && interval.start == 3 {
            eprintln!(
                "[DEBUG assign_addresses] CYCLE3: VReg {:?} -> addr {} size {} (interval [{}, {}])",
                vreg, new_addr, size, interval.start, interval.end
            );
        }
        // Debug: log assignments at cycle 18 for scratch conflict diagnosis
        if debug && interval.start == 18 {
            eprintln!(
                "[DEBUG assign_addresses] CYCLE18: VReg {:?} -> addr {} size {} (interval [{}, {}])",
                vreg, new_addr, size, interval.start, interval.end
            );
        }
    }

    // Post-allocation: derive lane addresses from vector group base addresses.
    // Each lane VReg gets address = group_base + lane_index.
    for (group_vreg, lane_vregs) in &config.vector_group_lanes {
        if let Some(&base_addr) = vreg_to_addr.get(group_vreg) {
            for (i, lane_vreg) in lane_vregs.iter().enumerate() {
                vreg_to_addr.insert(*lane_vreg, base_addr + i as i64);
            }
            if debug {
                eprintln!(
                    "[DEBUG assign_addresses] Vector group {:?} base={}, lanes: {:?}",
                    group_vreg,
                    base_addr,
                    lane_vregs.iter().enumerate().map(|(i, lv)| format!("{:?}={}", lv, base_addr + i as i64)).collect::<Vec<_>>()
                );
            }
        }
    }

    // Calculate statistics
    let max_addr = vreg_to_addr
        .iter()
        .map(|(vreg, &addr)| addr + vreg.size() as i64 - 1)
        .max()
        .unwrap_or(0);

    // Mark budget exceeded if we use any address at or above start_addr + budget (caller may still use result).
    let budget_limit = config.start_addr + config.budget as i64;
    let budget_exceeded = budget_exceeded || (max_addr >= budget_limit);

    let vreg_count = vreg_to_addr.len();
    let total_words: usize = sorted.iter().map(|(vreg, _)| vreg.size()).sum();

    Ok(AddressAssignmentResult {
        vreg_to_addr,
        peak_live,
        max_addr,
        vreg_count,
        total_words,
        budget_exceeded,
        budget: config.budget,
        live_intervals,
    })
}

/// An address conflict between two simultaneously-live VRegs.
#[derive(Debug, Clone)]
pub struct AddressConflict {
    pub vreg1: VReg,
    pub vreg2: VReg,
    pub addr1: i64,
    pub addr2: i64,
    pub interval1: LiveInterval,
    pub interval2: LiveInterval,
}

/// Verify that no two simultaneously-live VRegs have overlapping address ranges.
///
/// O(n²) worst case; sorts by address so we only check pairs whose address
/// ranges can overlap (fewer comparisons when addresses are spread out).
///
/// # Arguments
/// * `intervals` - Live intervals for all VRegs (from compute_live_intervals)
/// * `vreg_to_addr` - Assigned addresses (from assign_addresses)
///
/// # Returns
/// Vec of conflicts found. Empty means no conflicts.
pub fn verify_no_address_conflicts(
    intervals: &HashMap<VReg, LiveInterval>,
    vreg_to_addr: &HashMap<VReg, i64>,
) -> Vec<AddressConflict> {
    let mut conflicts = Vec::new();

    // Build entries: (VReg, LiveInterval, address, end_exclusive) for all VRegs in both maps
    let mut entries: Vec<(VReg, &LiveInterval, i64, i64)> = intervals
        .iter()
        .filter_map(|(v, iv)| {
            vreg_to_addr.get(v).map(|&a| (*v, iv, a, a + v.size() as i64))
        })
        .collect();

    // Sort by start address so we only check j where a_j < end_i (address overlap possible).
    entries.sort_by_key(|(_, _, a, _)| *a);

    for i in 0..entries.len() {
        let (v1, iv1, a1, end1) = entries[i];
        for j in (i + 1)..entries.len() {
            let (v2, iv2, a2, end2) = entries[j];
            if a2 >= end1 {
                break; // no later entry can overlap i's range
            }
            if end2 <= a1 {
                continue;
            }
            if iv1.overlaps(iv2) {
                conflicts.push(AddressConflict {
                    vreg1: v1,
                    vreg2: v2,
                    addr1: a1,
                    addr2: a2,
                    interval1: iv1.clone(),
                    interval2: iv2.clone(),
                });
            }
        }
    }
    conflicts
}

/// O(n log n) verification that no two simultaneously-live VRegs have overlapping
/// address ranges (T2: overlapping intervals must receive disjoint addresses).
///
/// Algorithm: (1) Sort by address and verify disjointness; (2) Group by (addr, size)
/// and within each group verify time intervals do not overlap.
///
/// # Arguments
/// * `intervals` - Live intervals for all VRegs (from compute_live_intervals)
/// * `vreg_to_addr` - Assigned addresses (from assign_addresses)
///
/// # Returns
/// Vec of conflicts found. Empty means no conflicts.
pub fn verify_no_address_conflicts_fast(
    intervals: &HashMap<VReg, LiveInterval>,
    vreg_to_addr: &HashMap<VReg, i64>,
) -> Vec<AddressConflict> {
    let mut conflicts = Vec::new();

    // Build entries: (VReg, LiveInterval, addr, size) for all VRegs in both maps
    let mut entries: Vec<(VReg, LiveInterval, i64, usize)> = intervals
        .iter()
        .filter_map(|(v, iv)| {
            vreg_to_addr.get(v).map(|&a| (*v, iv.clone(), a, v.size()))
        })
        .collect();

    if entries.is_empty() {
        return conflicts;
    }

    // Step 1: Sort by addr and verify no overlapping address ranges (allocator sanity).
    // Only report when ranges overlap but differ (same (addr, size) is handled in step 2).
    entries.sort_by_key(|(_, _, a, _)| *a);
    for i in 0..entries.len() - 1 {
        let (v1, ref iv1, a1, size1) = entries[i];
        let (v2, ref iv2, a2, size2) = entries[i + 1];
        let end1 = a1 + size1 as i64;
        if a2 < end1 && (a1 != a2 || size1 != size2) {
            // Overlapping but distinct address ranges; conflict iff time intervals overlap (T2)
            if iv1.overlaps(&iv2) {
                conflicts.push(AddressConflict {
                    vreg1: v1,
                    vreg2: v2,
                    addr1: a1,
                    addr2: a2,
                    interval1: iv1.clone(),
                    interval2: iv2.clone(),
                });
            }
        }
    }

    // Step 2: Group by (addr, size); within each group, sort by interval.start and verify
    // interval[i].end <= interval[i+1].start (no time overlap for same-address reuse).
    let mut by_addr_size: HashMap<(i64, usize), Vec<(VReg, LiveInterval)>> = HashMap::new();
    for (vreg, interval, addr, size) in entries {
        by_addr_size
            .entry((addr, size))
            .or_default()
            .push((vreg, interval));
    }
    for (&(addr, _), group) in by_addr_size.iter_mut() {
        if group.len() < 2 {
            continue;
        }
        group.sort_by_key(|(_, iv)| iv.start);
        for i in 0..group.len() - 1 {
            let (v1, ref iv1) = group[i];
            let (v2, ref iv2) = group[i + 1];
            if iv1.overlaps(iv2) {
                conflicts.push(AddressConflict {
                    vreg1: v1,
                    vreg2: v2,
                    addr1: addr,
                    addr2: addr,
                    interval1: iv1.clone(),
                    interval2: iv2.clone(),
                });
            }
        }
    }

    conflicts
}

// ---------------------------------------------------------------------------
// Next-free block: sorted used addresses (no interval merge)
// Sizes are only 1 or 8, so we expand to individual addresses and find first
// gap of length >= size in the sorted list. No generalized interval trees.
// ---------------------------------------------------------------------------

/// Build a sorted, deduplicated list of every used word address.
fn used_addresses_sorted(
    active_ranges: &[(i64, usize)],
    written_this_cycle: &HashSet<i64>,
) -> Vec<i64> {
    let mut addrs: Vec<i64> = active_ranges
        .iter()
        .flat_map(|&(a, s)| (a..a + s as i64))
        .chain(written_this_cycle.iter().copied())
        .collect();
    addrs.sort_unstable();
    addrs.dedup();
    addrs
}

/// First address >= min_start such that [addr, addr+length) is free.
/// Used addresses are sorted and deduplicated; we scan for the first gap of length >= length.
fn first_gap_of_length(sorted_used: &[i64], min_start: i64, length: usize) -> i64 {
    let len_i64 = length as i64;
    let mut candidate = min_start;
    for &u in sorted_used {
        if u > candidate {
            if (u - candidate) >= len_i64 {
                return candidate;
            }
        }
        if u >= candidate {
            candidate = u + 1;
        }
    }
    candidate
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
pub fn lower_ssa_op(ssa_op: &SsaOp, vreg_to_addr: &HashMap<VReg, i64>) -> Result<Op, String> {
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
        let addr = vreg_to_addr.get(&dest).copied().ok_or_else(|| {
            format!("VReg {:?} not found in address mapping", dest)
        })?;
        slot.push(SlotVal::Int(addr));
    }

    // Add source addresses/immediates
    for src in &ssa_op.srcs {
        match src {
            Operand::VReg(vreg) => {
                let addr = vreg_to_addr.get(vreg).copied().ok_or_else(|| {
                    format!(
                        "VReg {:?} not found in address mapping (op id={}, cycle={})",
                        vreg, ssa_op.id, ssa_op.cycle
                    )
                })?;
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

    Ok(Op {
        id: ssa_op.id,
        engine: ssa_op.engine,
        slot,
        cycle: ssa_op.cycle,
        dep_ids: ssa_op.dep_ids.clone(),
        reads,
        writes,
        last_use_reads: HashSet::new(), // Will be computed by scheduler
        trace_id: ssa_op.trace_id.clone(),
        produces_constant: ssa_op.produces_constant,
        from_const_choice: None, // This function doesn't handle ConstChoice/SelectChoice
        from_select_choice: None,
    })
}

/// Lower all SsaOps to concrete Ops.
pub fn lower_ssa_ops(ssa_ops: &[SsaOp], vreg_to_addr: &HashMap<VReg, i64>) -> Result<Vec<Op>, String> {
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
    use crate::types::{Engine, Operand, Opcode};

    #[test]
    fn test_compute_live_intervals_simple() {
        let mut emit = SsaEmitContext::new();
        let v1 = emit.emit_const(5);
        let _v2 = emit.emit_alu(Opcode::Add, v1, v1);

        // Simulate scheduling
        emit.ops[0].cycle = 0;
        emit.ops[1].cycle = 1;

        let config = AddressAssignConfig::default();
        let intervals = config.compute_live_intervals(&emit.ops).unwrap();

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

        let config = AddressAssignConfig::default();
        let intervals = config.compute_live_intervals(&emit.ops).unwrap();

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
        let config = AddressAssignConfig {
            select_fma_ops,
            ..Default::default()
        };
        let intervals = config.compute_live_intervals(&emit.ops).unwrap();

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

        // Regression: FMA path reads cond, true_val, false_val at cycle+1 (sub/FMA). use_cycles must
        // report all three so their intervals extend to at least select_cycle+1; otherwise use-def
        // fails when SelectChoice is scheduled far from producers (e.g. zero-gap + no tightening).
        let select_cycle = 5i32;
        let read_at = select_cycle + 1;
        assert!(
            intervals[&cond].end >= read_at,
            "cond must extend to cycle+1 (FMA path reads at cycle+1), got end {}",
            intervals[&cond].end
        );
        assert!(
            intervals[&true_val].end >= read_at,
            "true_val must extend to cycle+1 (sub reads it at cycle+1), got end {}",
            intervals[&true_val].end
        );
        assert!(
            intervals[&false_val].end >= read_at,
            "false_val must extend to cycle+1 (FMA path reads at cycle+1), got end {}",
            intervals[&false_val].end
        );
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

        // Schedule: both SelectChoices at 0 (so shared inputs have interval [0,0] or extended by consumers),
        // sel1 consumer at 5, sel2 consumer at 6. Use srcs to set consumer cycles.
        let sel1_vreg = emit.ops[idx1].dest.unwrap();
        let sel2_vreg = emit.ops[idx2].dest.unwrap();
        for (i, op) in emit.ops.iter_mut().enumerate() {
            let reads_sel1 = op.srcs.iter().any(|s| s.as_vreg() == Some(sel1_vreg));
            let reads_sel2 = op.srcs.iter().any(|s| s.as_vreg() == Some(sel2_vreg));
            op.cycle = if i == idx1 || i == idx2 {
                0
            } else if reads_sel1 && !reads_sel2 {
                5
            } else if reads_sel2 {
                6
            } else {
                0
            };
        }

        let mut select_fma_ops = HashSet::new();
        select_fma_ops.insert(id1);
        select_fma_ops.insert(id2);

        let mut config = AddressAssignConfig::default();
        config.select_fma_ops = select_fma_ops.clone();
        let result = assign_addresses(&emit.ops, &config).unwrap();
        assert!(!result.budget_exceeded);

        let intervals = config.compute_live_intervals(&emit.ops).unwrap();
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
        let result = assign_addresses(&emit.ops, &config).unwrap();

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
        let result = assign_addresses(&emit.ops, &config).unwrap();

        // v1 and v2 must have different addresses
        assert_ne!(result.vreg_to_addr[&v1], result.vreg_to_addr[&v2]);
        assert!(!result.budget_exceeded);
    }

    /// Hash-as-offset scenario: value VReg (consumed by vstore at T+1) and offset VReg
    /// (consumed by multiply_add at T+2) must get different addresses. Value is extended
    /// to op.cycle+1 by the vstore value pass; both intervals overlap so allocator must
    /// assign distinct addresses.
    #[test]
    fn test_value_and_offset_vregs_different_addresses() {
        use crate::types::Operand;
        let value_vreg = VReg::vector(100);
        let offset_vreg = VReg::vector(101);
        let addr_vreg = VReg::scalar(0);
        let idx_vreg = VReg::vector(1);
        let x_vreg = VReg::vector(2);
        let a_vreg = VReg::vector(3);
        let b_vreg = VReg::vector(4);

        let mut op_value = SsaOp::new(
            0,
            Engine::Valu,
            Opcode::Xor,
            Some(value_vreg),
            vec![Operand::VReg(a_vreg), Operand::VReg(b_vreg)],
        );
        op_value.cycle = 10;

        let mut op_offset = SsaOp::new(
            1,
            Engine::Valu,
            Opcode::Sub,
            Some(offset_vreg),
            vec![Operand::VReg(a_vreg), Operand::VReg(b_vreg)],
        );
        op_offset.cycle = 9;

        let mut op_vstore = SsaOp::new(
            2,
            Engine::Store,
            Opcode::Vstore,
            None,
            vec![Operand::VReg(addr_vreg), Operand::VReg(value_vreg)],
        );
        op_vstore.cycle = 11;

        let mut op_fma = SsaOp::new(
            3,
            Engine::Valu,
            Opcode::MultiplyAdd,
            Some(VReg::vector(102)),
            vec![Operand::VReg(idx_vreg), Operand::VReg(x_vreg), Operand::VReg(offset_vreg)],
        );
        op_fma.cycle = 12;

        // Defs for addr, idx, x, a, b at cycle 0 so they don't overlap with value/offset
        let mut ops = vec![
            SsaOp::new(10, Engine::Load, Opcode::Const, Some(addr_vreg), vec![]),
            SsaOp::new(11, Engine::Valu, Opcode::Xor, Some(idx_vreg), vec![]),
            SsaOp::new(12, Engine::Valu, Opcode::Xor, Some(x_vreg), vec![]),
            SsaOp::new(13, Engine::Load, Opcode::Const, Some(a_vreg), vec![]),
            SsaOp::new(14, Engine::Load, Opcode::Const, Some(b_vreg), vec![]),
        ];
        for op in &mut ops {
            op.cycle = 0;
        }
        ops.push(op_offset);
        ops.push(op_value);
        ops.push(op_vstore);
        ops.push(op_fma);

        let config = AddressAssignConfig::default();
        let result = assign_addresses(&ops, &config).unwrap();

        assert!(!result.budget_exceeded);
        assert_ne!(
            result.vreg_to_addr[&value_vreg],
            result.vreg_to_addr[&offset_vreg],
            "value (vstore at T+1) and offset (multiply_add at T+2) must get different addresses"
        );
    }

    #[test]
    fn test_assign_vector_contiguous() {
        let mut emit = SsaEmitContext::new();
        let s = emit.emit_const(5);
        let v = emit.emit_vbroadcast(s);

        emit.ops[0].cycle = 0;
        emit.ops[1].cycle = 1;

        let config = AddressAssignConfig::default();
        let result = assign_addresses(&emit.ops, &config).unwrap();

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

        let config = AddressAssignConfig::default();
        let intervals = config.compute_live_intervals(&emit.ops).unwrap();
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
            ..Default::default()
        };
        let result = assign_addresses(&emit.ops, &config).unwrap();

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

        let lowered = lower_ssa_op(&emit.ops[2], &vreg_to_addr).unwrap();

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

        let result = assign_addresses(&emit.ops, &config).unwrap();

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
        let result = assign_addresses(&emit.ops, &config).unwrap();

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
        let result = assign_addresses(&emit.ops, &config).unwrap();

        // A and B are BOTH written at cycle 3, so they MUST have different addresses
        assert_ne!(
            result.vreg_to_addr[&a], result.vreg_to_addr[&b],
            "VRegs defined in the SAME cycle must have different addresses (concurrent writes)"
        );
    }

    #[test]
    fn test_same_cycle_vector_and_scalar_ranges_non_overlapping() {
        // Two VRegs written at the SAME cycle where one is a vector (8 words) and one is a scalar (1 word).
        // Their assigned ranges must not overlap (the bug was tracking only base address, so the scalar
        // could get an address inside the vector's block).
        let mut emit = SsaEmitContext::new();
        let s = emit.emit_const(0);
        let v = emit.emit_vbroadcast(s); // vector 8 words
        emit.ops[0].cycle = 5;
        emit.ops[1].cycle = 5; // s and v both written at cycle 5
        let config = AddressAssignConfig::default();
        let result = assign_addresses(&emit.ops, &config).unwrap();
        let s_addr = result.vreg_to_addr[&s];
        let v_addr = result.vreg_to_addr[&v];
        let v_end = v_addr + 8;
        assert!(
            s_addr + 1 <= v_addr || s_addr >= v_end,
            "Scalar (addr {}) must not overlap vector [{}, {}); same-cycle writers",
            s_addr, v_addr, v_end
        );
    }

    #[test]
    fn test_fixed_vreg_same_cycle_other_gets_different_address() {
        // Fixed VReg (e.g. zero at 0) and another VReg both written at cycle 0.
        // The allocator must not assign address 0 (or any in the fixed range) to the other.
        let mut emit = SsaEmitContext::new();
        let zero = emit.emit_const(0);
        let a = emit.emit_const(1);
        emit.ops[0].cycle = 0;
        emit.ops[1].cycle = 0;
        let mut config = AddressAssignConfig::default();
        config.fixed_vregs.insert(zero, 0);
        config.zero_vreg = Some(zero);
        let result = assign_addresses(&emit.ops, &config).unwrap();
        assert_eq!(result.vreg_to_addr[&zero], 0);
        let a_addr = result.vreg_to_addr[&a];
        assert!(a_addr >= 1, "VReg written at same cycle as fixed zero must not get address 0 (got {})", a_addr);
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
        let result = assign_addresses(&emit.ops, &config).unwrap();

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

        let config = AddressAssignConfig::default();
        let intervals = config.compute_live_intervals(&emit.ops).unwrap();

        // Verify closed intervals
        // partition_addr: defined at 6, last read at 81 → [6, 81]
        assert_eq!(intervals[&partition_addr], LiveInterval::new(6, 81));
        // computation: defined at 10, last read at 17 → [10, 17]
        assert_eq!(intervals[&computation], LiveInterval::new(10, 17));
        // These overlap: 6 < 17 AND 10 < 81
        assert!(intervals[&partition_addr].overlaps(&intervals[&computation]));

        // Assign addresses - this is where the bug would manifest
        let config = AddressAssignConfig::default();
        let result = assign_addresses(&emit.ops, &config).unwrap();

        // These MUST have different addresses (the bug is if they're the same)
        assert_ne!(
            result.vreg_to_addr[&partition_addr], result.vreg_to_addr[&computation],
            "BUG: Overlapping VRegs got the same address"
        );

        // Fast verifier should also report no conflicts for this correct allocation
        let fast_conflicts = verify_no_address_conflicts_fast(
            &result.live_intervals,
            &result.vreg_to_addr,
        );
        assert!(fast_conflicts.is_empty(), "Correct allocation should have no conflicts");
    }

    /// Use-def check: assign_addresses returns Err when an op reads a VReg that has no interval
    /// (SSA violation or interval computation bug).
    #[test]
    fn test_use_def_check_rejects_read_with_no_interval() {
        use crate::types::Operand;
        let c1 = VReg::scalar(1);
        let phantom = VReg::scalar(99); // never defined
        let v2 = VReg::scalar(2);
        let mut ops: Vec<SsaOp> = vec![
            SsaOp::new(0, Engine::Load, Opcode::Const, Some(c1), vec![]),
            SsaOp::new(
                1,
                Engine::Alu,
                Opcode::Add,
                Some(v2),
                vec![Operand::VReg(c1), Operand::VReg(phantom)],
            ),
        ];
        ops[0].cycle = 0;
        ops[1].cycle = 1;
        let config = AddressAssignConfig::default();
        let result = assign_addresses(&ops, &config);
        assert!(result.is_err(), "assign_addresses should fail when an op reads a VReg with no interval");
        let err = result.unwrap_err();
        assert!(err.contains("Use-def violation"), "error message should mention use-def violation");
        assert!(err.contains("no interval"), "error message should mention no interval");
    }

    /// Overlapping intervals assigned the same address must be reported as conflicts
    /// by both verify_no_address_conflicts and verify_no_address_conflicts_fast.
    #[test]
    fn test_verify_fast_detects_overlapping_same_addr() {
        use crate::types::LiveInterval;
        let v1 = VReg::scalar(1);
        let v2 = VReg::scalar(2);
        let mut intervals = HashMap::new();
        intervals.insert(v1, LiveInterval::new(0, 10));
        intervals.insert(v2, LiveInterval::new(5, 15)); // overlaps [0,10]
        let mut vreg_to_addr = HashMap::new();
        vreg_to_addr.insert(v1, 100);
        vreg_to_addr.insert(v2, 100); // bug: same address

        let conflicts_slow = verify_no_address_conflicts(&intervals, &vreg_to_addr);
        let conflicts_fast = verify_no_address_conflicts_fast(&intervals, &vreg_to_addr);
        assert_eq!(conflicts_slow.len(), 1, "Slow verifier should report one conflict");
        assert_eq!(conflicts_fast.len(), 1, "Fast verifier should report one conflict");
    }

    // ========================================================================
    // Multi-group / last-round store speculations (3r/64, group 4 bug)
    //
    // When multiple groups' final values are vstored, value VRegs can have
    // identical or adjacent intervals. The allocator must never assign the
    // same address to two VRegs that are both read in the same cycle (two
    // vstore value sources at cycle C), even if one interval "ends" at C and
    // the next "starts" at C (end <= start would expire and reuse).
    // ========================================================================

    /// Two VRegs with identical closed interval [C, C] (both defined and
    /// last-used at cycle C) must get different addresses.
    ///
    /// Scenario: two vstore value sources at the same cycle (e.g. two groups'
    /// hash outputs stored at cycle 85). If the scheduler incorrectly places
    /// the producer at the same cycle as the vstore, we get interval [C, C].
    /// The allocator sorts by (start, end); when allocating the second [C, C]
    /// VReg it would expire the first (end <= start) and reuse the same address,
    /// causing one vstore to read the wrong value. Defensive: require distinct
    /// addresses whenever two intervals both include the same cycle.
    #[test]
    fn test_two_vregs_same_interval_both_read_at_same_cycle_distinct_addresses() {
        use crate::types::Operand;
        let a_vreg = VReg::vector(10);
        let b_vreg = VReg::vector(11);
        let c1 = VReg::scalar(1);
        let c2 = VReg::scalar(2);
        let addr1 = VReg::scalar(3);
        let addr2 = VReg::scalar(4);

        let mut ops: Vec<SsaOp> = vec![
            SsaOp::new(0, Engine::Load, Opcode::Const, Some(c1), vec![]),
            SsaOp::new(1, Engine::Load, Opcode::Const, Some(c2), vec![]),
            SsaOp::new(2, Engine::Load, Opcode::Const, Some(addr1), vec![]),
            SsaOp::new(3, Engine::Load, Opcode::Const, Some(addr2), vec![]),
        ];
        for op in &mut ops {
            op.cycle = 0;
        }
        // Producers at cycle 11 (so A and B get def at 11)
        ops.push(SsaOp::new(
            4,
            Engine::Valu,
            Opcode::Xor,
            Some(a_vreg),
            vec![Operand::VReg(c1), Operand::VReg(c2)],
        ));
        ops.last_mut().unwrap().cycle = 11;
        ops.push(SsaOp::new(
            5,
            Engine::Valu,
            Opcode::Xor,
            Some(b_vreg),
            vec![Operand::VReg(c1), Operand::VReg(c2)],
        ));
        ops.last_mut().unwrap().cycle = 11;
        // Vstores at cycle 11 (so A and B get last use at 11) → intervals [11, 11]
        ops.push(SsaOp::new(
            6,
            Engine::Store,
            Opcode::Vstore,
            None,
            vec![Operand::VReg(addr1), Operand::VReg(a_vreg)],
        ));
        ops.last_mut().unwrap().cycle = 11;
        ops.push(SsaOp::new(
            7,
            Engine::Store,
            Opcode::Vstore,
            None,
            vec![Operand::VReg(addr2), Operand::VReg(b_vreg)],
        ));
        ops.last_mut().unwrap().cycle = 11;

        let config = AddressAssignConfig::default();
        let intervals = config.compute_live_intervals(&ops).unwrap();
        assert_eq!(intervals[&a_vreg], LiveInterval::new(11, 11));
        assert_eq!(intervals[&b_vreg], LiveInterval::new(11, 11));

        let result = assign_addresses(&ops, &config).unwrap();
        assert_ne!(
            result.vreg_to_addr[&a_vreg],
            result.vreg_to_addr[&b_vreg],
            "Two VRegs with interval [C,C] (both read at same cycle) must get different addresses \
             so two vstores at the same cycle do not read the same scratch"
        );
    }

    /// Multiple (e.g. 8) value VRegs all last-used at the same cycle by vstores
    /// must get distinct addresses when their intervals overlap.
    ///
    /// Simulates last-round emit_store_data: 8 groups, each with a final hash
    /// value consumed by a vstore. If all 8 vstores are scheduled at cycle 85
    /// and all 8 values are produced by cycle 84, we have 8 intervals [84, 85].
    /// They must get 8 distinct addresses.
    #[test]
    fn test_multiple_vstore_value_vregs_same_last_use_cycle_distinct_addresses() {
        use crate::types::Operand;
        const N: usize = 8;
        let cycle_produce = 84;
        let cycle_vstore = 85;

        let mut ops: Vec<SsaOp> = vec![];
        let c1 = VReg::scalar(100);
        let c2 = VReg::scalar(101);
        ops.push(SsaOp::new(0, Engine::Load, Opcode::Const, Some(c1), vec![]));
        ops.push(SsaOp::new(1, Engine::Load, Opcode::Const, Some(c2), vec![]));
        ops[0].cycle = 0;
        ops[1].cycle = 0;

        let value_vregs: Vec<VReg> = (0..N).map(|i| VReg::vector(200 + i as u32)).collect();
        let addr_vregs: Vec<VReg> = (0..N).map(|i| VReg::scalar(300 + i as u32)).collect();

        for i in 0..N {
            ops.push(SsaOp::new(
                10 + i * 2,
                Engine::Load,
                Opcode::Const,
                Some(addr_vregs[i]),
                vec![],
            ));
            ops.last_mut().unwrap().cycle = 0;
            ops.push(SsaOp::new(
                10 + i * 2 + 1,
                Engine::Valu,
                Opcode::Xor,
                Some(value_vregs[i]),
                vec![Operand::VReg(c1), Operand::VReg(c2)],
            ));
            ops.last_mut().unwrap().cycle = cycle_produce;
        }
        for i in 0..N {
            ops.push(SsaOp::new(
                100 + i,
                Engine::Store,
                Opcode::Vstore,
                None,
                vec![
                    Operand::VReg(addr_vregs[i]),
                    Operand::VReg(value_vregs[i]),
                ],
            ));
            ops.last_mut().unwrap().cycle = cycle_vstore;
        }

        let config = AddressAssignConfig::default();
        let result = assign_addresses(&ops, &config).unwrap();

        let addrs: std::collections::HashSet<i64> =
            value_vregs.iter().map(|v| result.vreg_to_addr[v]).collect();
        assert_eq!(
            addrs.len(),
            N,
            "{} vstore value VRegs (last use at same cycle) must get {} distinct addresses",
            N,
            N
        );
    }
}
