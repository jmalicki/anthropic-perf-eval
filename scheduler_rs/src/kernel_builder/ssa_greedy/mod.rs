//! SSA greedy scheduler with budget and VALU→ALU shifting support.
//!
//! This module contains the greedy scheduling algorithm that schedules
//! SSA ops while respecting slot limits and scratch budget constraints.
//!
//! Key features:
//! - LUC (Last-Use-Count) heuristic for prioritizing ops that free scratch
//! - Same-cycle reuse model for liveness tracking, checked against interval liveness
//!   after every run (docs/LIVENESS_SPECIFICATION.md, T16)
//! - Emergency force-scheduling to break deadlocks
//! - VALU→ALU shifting when VALU slots are saturated
//! - Dead store elimination (see below)
//!
//! # Dead Store Detection & Elimination
//!
//! Dead stores are operations that define a VReg with zero consumers.
//! These can arise from:
//! - Kernel builder generating unused intermediate values
//! - CSE eliminating consumers but not producers
//! - Conditional code paths where some outputs are unused
//!
//! **Strategy**: Detect dead stores before scheduling, pre-mark them as
//! "already scheduled" so they never enter the ready queue. This:
//! - Avoids wasting cycles on useless work
//! - Doesn't count them toward scratch budget (liveness tracking skips them)
//! - Logs warnings for debugging (these are informational, not errors)
//!
//! The CP-SAT scheduler (`cpsat_scheduler.rs`) also honors dead stores by
//! checking for negative cycle hints (-1) from the greedy scheduler.

pub mod pressure_policy;

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::types::{
    ConstChoiceEncoding, EmissionPhase, Engine, LiveInterval, Opcode, Operand, PrecedenceOffset,
    SelectChoiceEncoding, SsaOp, VReg, VLEN,
};

use crate::address_assign::{compute_live_intervals, compute_peak_liveness, AddressAssignConfig};

use self::pressure_policy::{GreedyPolicy, PressurePolicy};
use super::ssa_lower::can_ssa_valu_shift_to_alu;

// ============================================================================
// GreedyBoundsUsage: explicit control over how ASAP/ALAP bounds affect scheduling
// ============================================================================

/// How the greedy scheduler uses ASAP/ALAP bounds.
///
/// Separates two concerns that were previously coupled via `Option`:
/// - **Policy decisions** (sort_ready, ASAP inflation): improve scheduling quality
/// - **Core loop mechanics** (heap priority, ASAP deferral, cycle clamping): change
///   scheduling behavior to keep hints within CP-SAT solver domains
///
/// Computing bounds should not implicitly activate deferral behavior. This enum
/// lets the caller explicitly choose what they want.
pub enum GreedyBoundsUsage<'a> {
    /// No bounds data. Pure greedy scheduling (backward-compatible with old `None`).
    None,
    /// Bounds available for policy decisions only (sort_ready + inflation).
    /// Core loop mechanics (priority, deferral, clamping) are unaffected.
    /// Used in greedy-only mode where we want inflation without deferral.
    PolicyOnly {
        asap: &'a HashMap<usize, i32>,
        alap: &'a HashMap<usize, i32>,
    },
    /// Full bounds: core loop uses ASAP for priority, deferral, and clamping.
    /// Used for CP-SAT warm-start to ensure hints stay within solver domains.
    Full {
        asap: &'a HashMap<usize, i32>,
        alap: &'a HashMap<usize, i32>,
    },
}

impl<'a> GreedyBoundsUsage<'a> {
    /// Bounds for policy decisions (sort_ready, inflation). Available in PolicyOnly + Full.
    fn policy_bounds(&self) -> Option<(&HashMap<usize, i32>, &HashMap<usize, i32>)> {
        match self {
            GreedyBoundsUsage::None => Option::None,
            GreedyBoundsUsage::PolicyOnly { asap, alap } => Some((asap, alap)),
            GreedyBoundsUsage::Full { asap, alap } => Some((asap, alap)),
        }
    }

    /// Bounds for core loop mechanics (priority, deferral, clamping). Only in Full.
    fn core_bounds(&self) -> Option<(&HashMap<usize, i32>, &HashMap<usize, i32>)> {
        match self {
            GreedyBoundsUsage::Full { asap, alap } => Some((asap, alap)),
            _ => Option::None,
        }
    }
}

// ============================================================================
// IncrementalDemandProfile: capacity-aware deferral (energetic logic)
// ============================================================================

/// Per-engine demand profile for capacity-aware deferral.
///
/// Tracks `demand[engine][t]` = count of unscheduled ops with engine E and ASAP==t.
/// Matches energetic tightening's "contributing-only" policy: we only count ops
/// that contribute to demand at their ASAP cycle. Optional ops (ConstChoice,
/// SelectChoice, shiftable VALU) are excluded.
struct IncrementalDemandProfile {
    /// demand[engine_index][cycle] = count of unscheduled ops with that engine and ASAP==cycle
    demand: [Vec<i64>; 5],
}

impl IncrementalDemandProfile {
    fn new(
        ops: &[SsaOp],
        asap: &HashMap<usize, i32>,
        shiftable_set: &HashSet<usize>,
        dead_store_indices: &HashSet<usize>,
        horizon: usize,
    ) -> Self {
        let mut demand: [Vec<i64>; 5] = [
            vec![0; horizon + 1],
            vec![0; horizon + 1],
            vec![0; horizon + 1],
            vec![0; horizon + 1],
            vec![0; horizon + 1],
        ];
        for (idx, op) in ops.iter().enumerate() {
            if dead_store_indices.contains(&idx) {
                continue;
            }
            if !Self::is_optional_on_engine(idx, op.engine, shiftable_set, op.opcode) {
                let asap_t = *asap.get(&op.id).unwrap_or(&0);
                if asap_t >= 0 && (asap_t as usize) <= horizon {
                    let ei = engine_index(op.engine);
                    demand[ei][asap_t as usize] += 1;
                }
            }
        }
        Self { demand }
    }

    fn is_optional_on_engine(idx: usize, engine: Engine, shiftable_set: &HashSet<usize>, opcode: Opcode) -> bool {
        match engine {
            Engine::Flow => opcode == Opcode::SelectChoice,
            Engine::Load => opcode == Opcode::ConstChoice,
            Engine::Valu => shiftable_set.contains(&idx),
            _ => false,
        }
    }

    fn min_demand_at_cycle(&self, cycle: i32, engine: Engine) -> i64 {
        if cycle < 0 {
            return 0;
        }
        let ei = engine_index(engine);
        self.demand[ei].get(cycle as usize).copied().unwrap_or(0)
    }

    fn schedule_op(&mut self, idx: usize, op: &SsaOp, asap_t: i32, shiftable_set: &HashSet<usize>) {
        if Self::is_optional_on_engine(idx, op.engine, shiftable_set, op.opcode) {
            return;
        }
        let ei = engine_index(op.engine);
        if asap_t >= 0 && (asap_t as usize) < self.demand[ei].len() {
            self.demand[ei][asap_t as usize] = self.demand[ei][asap_t as usize].saturating_sub(1);
        }
    }
}

fn engine_index(e: Engine) -> usize {
    match e {
        Engine::Load => 0,
        Engine::Store => 1,
        Engine::Alu => 2,
        Engine::Valu => 3,
        Engine::Flow => 4,
        _ => 2, // Default to ALU for non-slot engines (shouldn't be used)
    }
}

fn engine_capacity(e: Engine) -> i64 {
    e.slot_limit() as i64
}

// ============================================================================
// InflationStats: lightweight counters for ASAP inflation diagnostics
// ============================================================================

/// Tracks ASAP inflation activity during a greedy scheduling run.
/// Printed as a summary after scheduling completes (via eprintln, verbose only).
#[derive(Default)]
struct InflationStats {
    /// Ops deferred by inflation
    total_deferred: usize,
    /// Sum of all inflation amounts (cycles)
    total_inflation_cycles: i64,
    /// Largest single inflation applied
    max_inflation: i32,
    /// Times MIN_PASS_THROUGH prevented inflation (capacity limit)
    pass_through_blocks: usize,
    /// Times consecutive_empty breaker disabled inflation
    breaker_activations: usize,
    /// Times the net-liveness guard blocked inflation (words_freed >= words_added)
    net_liveness_blocks: usize,
}

// ============================================================================
// SlotTracker: single source of truth for per-cycle engine slot accounting
// ============================================================================

/// Result of attempting to place an op on an engine slot this cycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SlotResult {
    /// Op placed on its natural engine (VALU as VALU, ALU as ALU, etc.)
    Placed,
    /// VALU op shifted to ALU (occupies VLEN=8 ALU slots)
    Shifted,
    /// No room on this cycle — caller should defer or skip
    NoRoom,
}

/// Tracks per-cycle engine slot usage. Enforces capacity invariants in one place
/// so all scheduling paths (normal, force-schedule, simple) share the same logic.
///
/// Capacity limits: ALU=12, VALU=6, Load=2, Store=2, Flow=1.
#[derive(Debug, Default)]
struct SlotTracker {
    load_used: usize,
    store_used: usize,
    alu_used: usize,
    valu_used: usize,
    flow_used: usize,
}

impl SlotTracker {
    fn new() -> Self {
        Self::default()
    }

    /// Check whether an op with the given engine could fit (read-only, no mutation).
    fn fits(&self, engine: Engine, can_shift: bool) -> bool {
        match engine {
            Engine::Load => self.load_used < 2,
            Engine::Store => self.store_used < 2,
            Engine::Alu => self.alu_used < 12,
            Engine::Valu => self.valu_used < 6 || (can_shift && self.alu_used + VLEN <= 12),
            Engine::Flow => self.flow_used < 1,
            _ => true,
        }
    }

    /// Create a read-only snapshot for policy inspection.
    fn view(&self) -> pressure_policy::SlotTrackerView {
        pressure_policy::SlotTrackerView {
            alu_used: self.alu_used,
            valu_used: self.valu_used,
            load_used: self.load_used,
            store_used: self.store_used,
            flow_used: self.flow_used,
        }
    }

    /// Atomically try to place an op. Updates counters on success, returns
    /// `NoRoom` without side effects on failure.
    ///
    /// `allow_shift`: whether the caller permits VALU→ALU shifting (strategy
    /// may block it). When `false` and VALU is full, returns `NoRoom` even if
    /// ALU has room — the caller can retry with `allow_shift=true` to force it.
    fn try_place(&mut self, engine: Engine, can_shift: bool, allow_shift: bool) -> SlotResult {
        match engine {
            Engine::Load if self.load_used < 2 => {
                self.load_used += 1;
                SlotResult::Placed
            }
            Engine::Store if self.store_used < 2 => {
                self.store_used += 1;
                SlotResult::Placed
            }
            Engine::Alu if self.alu_used < 12 => {
                self.alu_used += 1;
                SlotResult::Placed
            }
            Engine::Valu if self.valu_used < 6 => {
                self.valu_used += 1;
                SlotResult::Placed
            }
            Engine::Valu if can_shift && allow_shift && self.alu_used + VLEN <= 12 => {
                self.alu_used += VLEN;
                SlotResult::Shifted
            }
            Engine::Flow if self.flow_used < 1 => {
                self.flow_used += 1;
                SlotResult::Placed
            }
            // Barrier and other non-slot engines always fit (no slot consumed)
            Engine::Barrier => SlotResult::Placed,
            _ => SlotResult::NoRoom,
        }
    }

    /// Try to place a VALU op only as shifted (ALU). Used when the op is in
    /// fixed_shifted_ops so the hint and schedule must agree (shifted => 8 ALU slots).
    /// Returns NoRoom if ALU cannot fit VLEN slots.
    fn try_place_shifted_only(&mut self) -> SlotResult {
        if self.alu_used + VLEN <= 12 {
            self.alu_used += VLEN;
            SlotResult::Shifted
        } else {
            SlotResult::NoRoom
        }
    }
}

// ============================================================================
// GreedyDiagnostics: what the greedy scheduler reports about budget overflow
// ============================================================================

/// Diagnostics for one greedy scheduling run.
///
/// The caller sets `verbose`; the scheduler fills in the rest. A run that is
/// force-scheduled past the scratch budget can stay over budget for thousands
/// of cycles, so the detailed liveness breakdown is emitted for the first
/// over-budget cycle and (if different) the worst one only, and never when
/// `verbose` is false. The one-line summary from [`GreedyDiagnostics::summary`]
/// is always printed.
#[derive(Debug, Default)]
pub(crate) struct GreedyDiagnostics {
    /// Print detailed diagnostics (liveness breakdown dumps, ASAP inflation stats) to stderr.
    pub verbose: bool,
    /// Number of distinct cycles in which an op was force-scheduled past the budget.
    pub over_budget_cycles: usize,
    /// `(cycle, live_words)` of the first over-budget cycle.
    pub first_over_budget: Option<(usize, usize)>,
    /// `(cycle, live_words)` of the over-budget cycle with the most live words.
    pub worst_over_budget: Option<(usize, usize)>,
    /// Number of detailed breakdown dumps written to stderr (0 unless `verbose`; at most 2).
    pub dumps_emitted: usize,
    /// Breakdown for the worst cycle so far, printed by `finish` (only kept when `verbose`).
    worst_dump: Option<String>,
    /// Last cycle recorded, so several force-schedules in one cycle count once.
    last_cycle: Option<usize>,
}

impl GreedyDiagnostics {
    /// Diagnostics with the given verbosity and nothing recorded yet.
    pub(crate) fn new(verbose: bool) -> Self {
        GreedyDiagnostics {
            verbose,
            ..Default::default()
        }
    }

    /// Record that an op was force-scheduled past the budget at `cycle`.
    ///
    /// # Arguments
    /// * `cycle` - Cycle of the force-schedule
    /// * `live_words` - Words live at that point
    /// * `render` - Produces the detailed breakdown; only called when `verbose` and this
    ///   is the first over-budget cycle or a new worst one
    fn record_over_budget(
        &mut self,
        cycle: usize,
        live_words: usize,
        render: impl FnOnce() -> String,
    ) {
        if self.last_cycle == Some(cycle) {
            return;
        }
        self.last_cycle = Some(cycle);
        self.over_budget_cycles += 1;
        if self.first_over_budget.is_none() {
            self.first_over_budget = Some((cycle, live_words));
            self.worst_over_budget = Some((cycle, live_words));
            if self.verbose {
                eprint!("{}", render());
                self.dumps_emitted += 1;
            }
        } else if self.worst_over_budget.map_or(true, |(_, w)| live_words > w) {
            self.worst_over_budget = Some((cycle, live_words));
            if self.verbose {
                self.worst_dump = Some(render());
            }
        }
    }

    /// Print the breakdown of the worst over-budget cycle, if it is not the first one
    /// (which was already printed). Does nothing unless `verbose`.
    fn finish(&mut self) {
        if let Some(dump) = self.worst_dump.take() {
            eprint!("{}", dump);
            self.dumps_emitted += 1;
        }
    }

    /// End-of-run output, for both the success and the failure exit of the scheduler: the
    /// worst-cycle breakdown (verbose only), then the one-line summary (always, if the run
    /// went over budget).
    ///
    /// # Arguments
    /// * `peak_live` - Peak live words over the run (the scheduler's own counter)
    /// * `scratch_budget` - Scratch budget in words
    fn emit(&mut self, peak_live: usize, scratch_budget: usize) {
        self.finish();
        if let Some(summary) = self.summary(peak_live, scratch_budget) {
            eprintln!("{}", summary);
        }
    }

    /// One-line summary of the budget overflow.
    ///
    /// # Arguments
    /// * `peak_live` - Peak live words over the whole run (the scheduler's own counter)
    /// * `scratch_budget` - Scratch budget in words
    ///
    /// # Returns
    /// `None` if the run never went over budget, otherwise a single line (no newline).
    pub(crate) fn summary(&self, peak_live: usize, scratch_budget: usize) -> Option<String> {
        let (first_cycle, first_words) = self.first_over_budget?;
        let (worst_cycle, worst_words) = self.worst_over_budget.unwrap_or((first_cycle, first_words));
        Some(format!(
            "Greedy scheduler force-scheduled past the scratch budget in {} cycle(s) to break deadlock: \
             budget={} words, first at cycle {} ({} words live), worst at cycle {} ({} words live), \
             greedy peak_live={}",
            self.over_budget_cycles,
            scratch_budget,
            first_cycle,
            first_words,
            worst_cycle,
            worst_words,
            peak_live
        ))
    }
}

/// Format a usage breakdown of the currently live VRegs when the budget is exceeded.
///
/// This helps diagnose which rounds/var_types are consuming the most scratch
/// when the scheduler is forced to exceed the budget.
///
/// # Arguments
/// * `ops` - All SSA ops (used to find the definer of each live VReg)
/// * `live_vregs` - VRegs live at `cycle`
/// * `cycle` - Cycle at which an op was force-scheduled past the budget
/// * `live_words` - Words live at `cycle`
/// * `scratch_budget` - Scratch budget in words
///
/// # Returns
/// The multi-line breakdown text (by round, var_type and phase), newline-terminated.
pub(super) fn format_live_usage_breakdown(
    ops: &[SsaOp],
    live_vregs: &HashSet<VReg>,
    cycle: usize,
    live_words: usize,
    scratch_budget: usize,
) -> String {
    use std::collections::BTreeMap;
    use std::fmt::Write as _;

    let mut out = String::new();

    let _ = writeln!(out, "\n=== Budget Exceeded at Cycle {} ===", cycle);
    let _ = writeln!(out, 
        "Live: {} words / {} budget ({} VRegs)",
        live_words,
        scratch_budget,
        live_vregs.len()
    );

    // Find the ops that define each live VReg
    let mut by_round: BTreeMap<i32, (usize, usize)> = BTreeMap::new(); // round -> (count, words)
    let mut by_type: BTreeMap<&str, (usize, usize)> = BTreeMap::new(); // type -> (count, words)
    let mut by_phase: BTreeMap<String, (usize, usize)> = BTreeMap::new(); // phase (branch includes round) -> (count, words)

    for op in ops {
        if let Some(dest) = op.dest {
            if live_vregs.contains(&dest) {
                let words = dest.size();

                // By round
                let round = op
                    .source_info
                    .as_ref()
                    .and_then(|s| s.round)
                    .map(|r| r as i32)
                    .unwrap_or(-1);
                let entry = by_round.entry(round).or_insert((0, 0));
                entry.0 += 1;
                entry.1 += words;

                // By var_type
                let var_type = op.var_type.map(|t| t.as_str()).unwrap_or("temporary");
                let entry = by_type.entry(var_type).or_insert((0, 0));
                entry.0 += 1;
                entry.1 += words;

                // By phase: for Branch, include round (e.g. branch_r3) so we see which round generated the allocation
                let phase_label = op
                    .source_info
                    .as_ref()
                    .map(|s| {
                        if s.phase == EmissionPhase::Branch {
                            match s.round {
                                Some(r) => format!("branch_r{}", r),
                                None => s.phase.as_str().to_string(),
                            }
                        } else {
                            s.phase.as_str().to_string()
                        }
                    })
                    .unwrap_or_else(|| "unknown".to_string());
                let entry = by_phase.entry(phase_label).or_insert((0, 0));
                entry.0 += 1;
                entry.1 += words;
            }
        }
    }

    // Print by round
    let _ = writeln!(out, "\nLive by Round:");
    for (round, (count, words)) in &by_round {
        let round_str = if *round < 0 {
            "init".to_string()
        } else {
            format!("{}", round)
        };
        let _ = writeln!(out, 
            "  Round {:>4}: {:>4} vregs, {:>5} words",
            round_str, count, words
        );
    }

    // Print by var_type
    let _ = writeln!(out, "\nLive by VarType:");
    let mut type_vec: Vec<_> = by_type.iter().collect();
    type_vec.sort_by(|a, b| b.1 .1.cmp(&a.1 .1)); // Sort by words descending
    for (var_type, (count, words)) in type_vec {
        let _ = writeln!(out, "  {:>15}: {:>4} vregs, {:>5} words", var_type, count, words);
    }

    // Print by phase
    let _ = writeln!(out, "\nLive by Phase:");
    let mut phase_vec: Vec<_> = by_phase.iter().collect();
    phase_vec.sort_by(|a, b| b.1 .1.cmp(&a.1 .1)); // Sort by words descending
    for (phase, (count, words)) in phase_vec {
        let _ = writeln!(out, "  {:>20}: {:>4} vregs, {:>5} words", phase, count, words);
    }

    let _ = writeln!(out, "=== End Budget Exceeded Dump ===\n");
    out
}

/// VRegs whose remaining-user count drops when an op is scheduled.
///
/// The count is of reader ops, so each VReg appears once however many operands name it.
/// Readers name only one side of a vector group (a lane or the group VReg), so the other
/// side is released through `liveness_aliases`; without it that side stays live forever.
///
/// # Arguments
/// * `op` - The op being scheduled (reads its `srcs` and holds its `keep_live` values)
/// * `liveness_aliases` - Lane → group VReg (both group kinds); write-group VReg → lanes
///
/// # Returns
/// The VRegs the op reads or keeps live, then their aliases, each exactly once.
fn liveness_release_vregs(op: &SsaOp, liveness_aliases: &HashMap<VReg, Vec<VReg>>) -> Vec<VReg> {
    let mut released: Vec<VReg> = Vec::with_capacity(op.srcs.len());
    for vreg in liveness_read_vregs(op) {
        if !released.contains(&vreg) {
            released.push(vreg);
        }
    }
    let direct = released.len();
    for i in 0..direct {
        if let Some(extra) = liveness_aliases.get(&released[i]) {
            for alias in extra {
                if !released.contains(alias) {
                    released.push(*alias);
                }
            }
        }
    }
    released
}

/// Write groups as the live-word counter sees them.
///
/// A write group (`fresh_vector_group`) is one block of VLEN words, held from its first
/// lane write until its last read, and at least until the cycle after a lane write that
/// nothing reads (as in address assignment). The counter therefore tracks the group VReg,
/// never the lanes.
#[derive(Default)]
struct WriteGroups {
    /// Lane → group VReg.
    of: HashMap<VReg, VReg>,
    /// Lane writes still to be scheduled, per group.
    pending_writes: HashMap<VReg, usize>,
    /// First cycle at which the group may be released, set by writes of unread lanes.
    hold_until: HashMap<VReg, i32>,
}

impl WriteGroups {
    /// The VReg the counter tracks for a written VReg: its group for a lane, else itself.
    fn counted(&self, dest: VReg) -> VReg {
        self.of.get(&dest).copied().unwrap_or(dest)
    }

    /// Whether `vreg` can leave the counter at `cycle` once its last reader has run.
    /// Always true for a VReg that is not a write group.
    fn releasable_at(&self, vreg: &VReg, cycle: usize) -> bool {
        self.pending_writes.get(vreg).map_or(true, |&p| p == 0)
            && self.hold_until.get(vreg).map_or(true, |&c| c <= cycle as i32)
    }
}

/// Add the value written by a just-scheduled op to the live-word counter.
///
/// A write-group lane counts as its group: the whole block is added at the first lane
/// write. A value no scheduled op will read is still held for the cycle of its write and
/// released at the next cycle (spec A4, case 2).
fn count_written_dest(
    dest: VReg,
    cycle: usize,
    write_groups: &mut WriteGroups,
    vreg_remaining_users: &HashMap<VReg, usize>,
    live_vregs: &mut HashSet<VReg>,
    live_words: &mut usize,
    vreg_expiry_heap: &mut BinaryHeap<Reverse<(i32, VReg)>>,
    trace: &mut CounterTrace,
) {
    let unread = |vreg: &VReg| vreg_remaining_users.get(vreg).map_or(true, |&n| n == 0);
    let counted = write_groups.counted(dest);
    if counted != dest {
        if let Some(pending) = write_groups.pending_writes.get_mut(&counted) {
            *pending = pending.saturating_sub(1);
        }
        if unread(&dest) {
            write_groups.hold_until.insert(counted, (cycle + 1) as i32);
        }
    }
    if live_vregs.insert(counted) {
        *live_words += counted.size();
        trace.spans.insert(counted, (cycle as i32, None));
    }
    let writes_done = write_groups.pending_writes.get(&counted).map_or(true, |&p| p == 0);
    if writes_done && unread(&counted) {
        vreg_expiry_heap.push(Reverse(((cycle + 1) as i32, counted)));
    }
}

/// Decrement the remaining-reader count of everything a just-scheduled op read, and
/// release from the live-word counter each value whose last reader it was.
///
/// Release is immediate (same-cycle reuse: reads happen before writes within a cycle),
/// except for a write group that is still being written or holds an unread lane.
fn release_read_values(
    op: &SsaOp,
    cycle: usize,
    liveness_aliases: &HashMap<VReg, Vec<VReg>>,
    write_groups: &WriteGroups,
    vreg_remaining_users: &mut HashMap<VReg, usize>,
    live_vregs: &mut HashSet<VReg>,
    live_words: &mut usize,
    vreg_expiry_heap: &mut BinaryHeap<Reverse<(i32, VReg)>>,
    trace: &mut CounterTrace,
) {
    for vreg in &liveness_release_vregs(op, liveness_aliases) {
        let Some(remaining) = vreg_remaining_users.get_mut(vreg) else {
            continue;
        };
        *remaining = remaining.saturating_sub(1);
        if *remaining != 0 || !live_vregs.contains(vreg) {
            continue;
        }
        if write_groups.releasable_at(vreg, cycle) {
            live_vregs.remove(vreg);
            *live_words = live_words.saturating_sub(vreg.size());
            trace.release(vreg, cycle);
        } else if let Some(&hold_until) = write_groups.hold_until.get(vreg) {
            // Written out, but an unread lane holds the block one more cycle.
            if write_groups.pending_writes.get(vreg).map_or(true, |&p| p == 0) {
                vreg_expiry_heap.push(Reverse((hold_until, *vreg)));
            }
        }
        // Otherwise lane writes are still pending; the last one releases the group.
    }
}

/// VRegs an op holds until it runs: its VReg operands and its `keep_live` list
/// (the same reads `address_assign`'s `LivenessUse` reports). May contain duplicates.
fn liveness_read_vregs(op: &SsaOp) -> impl Iterator<Item = VReg> + '_ {
    op.srcs
        .iter()
        .filter_map(|src| src.as_vreg())
        .chain(op.keep_live.iter().flatten().copied())
}

// ============================================================================
// Liveness agreement: the scheduler's counter against interval liveness
// ============================================================================

/// Comparison of the greedy scheduler's live-word counter with interval liveness
/// (`address_assign::compute_live_intervals`) of the schedule the same run produced.
///
/// Interval liveness at cycle `t` is the words of every value with `start <= t < end`:
/// what is still held once cycle `t`'s writes are done, with a value last read at `t`
/// already released (docs/LIVENESS_SPECIFICATION.md, A2 and T1).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LivenessAgreement {
    /// Peak the scheduler reports: the counter's maximum after any single op.
    pub counter_peak: usize,
    /// Maximum of the counter sampled after the last op of each cycle.
    pub counter_cycle_end_peak: usize,
    /// Peak of interval liveness; equals `address_assign::compute_peak_liveness`.
    pub interval_peak: usize,
    /// Cycles where the cycle-end counter differs from interval liveness.
    pub mismatched_cycles: usize,
    /// Largest amount by which the cycle-end counter is below interval liveness.
    pub max_undercount: usize,
    /// Largest amount by which the cycle-end counter is above interval liveness.
    pub max_overcount: usize,
    /// First differing cycle as `(cycle, counter, interval)`.
    pub first_mismatch: Option<(usize, usize, usize)>,
    /// Which kinds of value differ, one line per kind (empty when every cycle agrees).
    pub value_mismatches: Vec<String>,
}

impl LivenessAgreement {
    /// True when the counter equals interval liveness at the end of every cycle.
    pub fn agrees(&self) -> bool {
        self.mismatched_cycles == 0
    }
}

impl std::fmt::Display for LivenessAgreement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "counter peak {} (cycle-end {}), interval peak {}, mismatched cycles {}, \
             max undercount {}, max overcount {}",
            self.counter_peak,
            self.counter_cycle_end_peak,
            self.interval_peak,
            self.mismatched_cycles,
            self.max_undercount,
            self.max_overcount
        )?;
        if let Some((cycle, counter, interval)) = self.first_mismatch {
            write!(f, ", first at cycle {} (counter {}, interval {})", cycle, counter, interval)?;
        }
        Ok(())
    }
}

/// What the scheduler's live-word counter did during one run.
#[derive(Default)]
struct CounterTrace {
    /// The counter after the last op of each cycle (index = cycle).
    cycle_end_live: Vec<usize>,
    /// Per value: the cycle it was added to the counter and the cycle it was released.
    spans: HashMap<VReg, (i32, Option<i32>)>,
}

impl CounterTrace {
    fn release(&mut self, vreg: &VReg, cycle: usize) {
        if let Some(span) = self.spans.get_mut(vreg) {
            span.1 = Some(cycle as i32);
        }
    }
}

/// Interval liveness per cycle for a schedule: words with `start <= t < end`.
///
/// # Arguments
/// * `ops` - Ops with assigned cycles; ops with a negative cycle (skipped dead stores) are ignored
/// * `vector_group_lanes` - Vector groups, as passed to the scheduler
///
/// # Returns
/// `(curve, peak, intervals)`: `curve[t]` is the live words at cycle `t`; `peak` is
/// `address_assign::compute_peak_liveness` of `intervals`.
pub(super) fn interval_liveness_curve(
    ops: &[SsaOp],
    vector_group_lanes: Option<&HashMap<VReg, Vec<VReg>>>,
) -> Result<(Vec<usize>, usize, HashMap<VReg, LiveInterval>), String> {
    let scheduled: Vec<SsaOp> = ops.iter().filter(|op| op.cycle >= 0).cloned().collect();
    // A group none of whose VRegs is written by a scheduled op occupies nothing.
    // (`compute_live_intervals` would give it a phantom [0, 1] interval.)
    let written: HashSet<VReg> = scheduled.iter().filter_map(|op| op.dest).collect();
    let groups: HashMap<VReg, Vec<VReg>> = vector_group_lanes
        .into_iter()
        .flatten()
        .filter(|(key, lanes)| written.contains(*key) || lanes.iter().any(|l| written.contains(l)))
        .map(|(key, lanes)| (*key, lanes.clone()))
        .collect();
    let config = AddressAssignConfig { vector_group_lanes: groups, ..Default::default() };
    let intervals = compute_live_intervals(&config, &scheduled)?;
    let n_cycles = scheduled.iter().map(|op| op.cycle + 1).max().unwrap_or(0) as usize;
    // Difference array: +size at start, -size at end, prefix-summed.
    let mut delta: Vec<i64> = vec![0; n_cycles + 2];
    for (vreg, interval) in &intervals {
        if interval.end <= interval.start {
            continue;
        }
        delta[interval.start as usize] += vreg.size() as i64;
        delta[(interval.end as usize).min(n_cycles + 1)] -= vreg.size() as i64;
    }
    let mut curve: Vec<usize> = Vec::with_capacity(n_cycles);
    let mut live: i64 = 0;
    for d in delta.iter().take(n_cycles) {
        live += d;
        curve.push(live as usize);
    }
    Ok((curve, compute_peak_liveness(&intervals), intervals))
}

/// Compare the scheduler's counter with interval liveness of the finished schedule.
///
/// # Arguments
/// * `ops` - Ops as scheduled by the greedy
/// * `vector_group_lanes` - Vector groups, as passed to the scheduler
/// * `trace` - What the counter did during the run
/// * `counter_peak` - The peak the scheduler reports
///
/// # Returns
/// The comparison, or `Err` if interval liveness cannot be computed for the schedule.
fn compare_counter_with_intervals(
    ops: &[SsaOp],
    vector_group_lanes: Option<&HashMap<VReg, Vec<VReg>>>,
    trace: &CounterTrace,
    counter_peak: usize,
) -> Result<LivenessAgreement, String> {
    let cycle_end_live = &trace.cycle_end_live;
    let (curve, interval_peak, intervals) = interval_liveness_curve(ops, vector_group_lanes)?;
    let mut agreement = LivenessAgreement {
        counter_peak,
        counter_cycle_end_peak: cycle_end_live.iter().copied().max().unwrap_or(0),
        interval_peak,
        ..Default::default()
    };
    for cycle in 0..curve.len().max(cycle_end_live.len()) {
        let counter = cycle_end_live.get(cycle).copied().unwrap_or(0);
        let interval = curve.get(cycle).copied().unwrap_or(0);
        if counter == interval {
            continue;
        }
        agreement.mismatched_cycles += 1;
        agreement.max_undercount = agreement.max_undercount.max(interval.saturating_sub(counter));
        agreement.max_overcount = agreement.max_overcount.max(counter.saturating_sub(interval));
        if agreement.first_mismatch.is_none() {
            agreement.first_mismatch = Some((cycle, counter, interval));
        }
    }
    if agreement.mismatched_cycles > 0 {
        agreement.value_mismatches = describe_value_mismatches(ops, vector_group_lanes, trace, &intervals);
    }
    Ok(agreement)
}

/// Per-value differences between the counter's spans and the intervals, grouped by kind.
///
/// # Returns
/// One line per kind of value (producer opcode, or "write group") whose counter span
/// `[added, released)` differs from its interval `[start, end)`, with a count and an example.
fn describe_value_mismatches(
    ops: &[SsaOp],
    vector_group_lanes: Option<&HashMap<VReg, Vec<VReg>>>,
    trace: &CounterTrace,
    intervals: &HashMap<VReg, LiveInterval>,
) -> Vec<String> {
    let producer: HashMap<VReg, &SsaOp> =
        ops.iter().filter(|op| op.cycle >= 0).filter_map(|op| op.dest.map(|d| (d, op))).collect();
    let empty = HashMap::new();
    let groups = vector_group_lanes.unwrap_or(&empty);
    let mut by_kind: HashMap<String, (usize, String)> = HashMap::new();
    let mut note = |kind: String, example: String| {
        let entry = by_kind.entry(kind).or_insert((0, example));
        entry.0 += 1;
    };
    let mut seen: HashSet<VReg> = HashSet::new();
    for (vreg, interval) in intervals {
        seen.insert(*vreg);
        let kind = match producer.get(vreg) {
            Some(op) => format!("{:?}", op.opcode),
            None if groups.contains_key(vreg) => "write group".to_string(),
            None => "no producer".to_string(),
        };
        // A value still counted when the schedule ends is released at the cycle after it.
        let n_cycles = trace.cycle_end_live.len() as i32;
        match trace.spans.get(vreg) {
            Some(&(added, released))
                if added == interval.start && released.unwrap_or(n_cycles) == interval.end => {}
            Some(&(added, released)) => note(
                kind,
                format!(
                    "{:?} interval [{}, {}) counter [{}, {:?})",
                    vreg, interval.start, interval.end, added, released
                ),
            ),
            None => note(
                format!("{} (never counted)", kind),
                format!("{:?} interval [{}, {})", vreg, interval.start, interval.end),
            ),
        }
    }
    for (vreg, span) in &trace.spans {
        if !seen.contains(vreg) {
            note("counted, no interval".to_string(), format!("{:?} counter {:?}", vreg, span));
        }
    }
    let mut lines: Vec<String> = by_kind
        .into_iter()
        .map(|(kind, (count, example))| format!("{}: {} values, e.g. {}", kind, count, example))
        .collect();
    lines.sort();
    lines
}

/// Build a lightweight `SchedulingCandidate` for the ASAP inflation check.
///
/// Computes the same fields as the sort-ready candidate builder (LUC, words_freed,
/// ASAP/ALAP, etc.) for a single op. Used in the inflation loop where we need
/// per-op candidates without building the full sorted list.
fn build_inflation_candidate(
    ops: &[SsaOp],
    vreg_remaining_users: &HashMap<VReg, usize>,
    live_vregs: &HashSet<VReg>,
    consumers: &[Vec<usize>],
    scheduled: &HashSet<usize>,
    shiftable_set: &HashSet<usize>,
    asap_map: &HashMap<usize, i32>,
    alap_map: &HashMap<usize, i32>,
    idx: usize,
) -> pressure_policy::SchedulingCandidate {
    let luc = ops[idx]
        .srcs
        .iter()
        .filter(|src| {
            if let Operand::VReg(vreg) = src {
                vreg_remaining_users.get(vreg).map_or(false, |&n| n == 1)
            } else {
                false
            }
        })
        .count();

    let words_added = ops[idx].dest.map_or(0, |d| d.size());
    let words_freed: usize = ops[idx]
        .srcs
        .iter()
        .filter(|src| {
            if let Operand::VReg(vreg) = src {
                vreg_remaining_users.get(vreg).map_or(false, |&n| n == 1)
                    && live_vregs.contains(vreg)
            } else {
                false
            }
        })
        .map(|src| {
            if let Operand::VReg(vreg) = src {
                vreg.size()
            } else {
                0
            }
        })
        .sum();

    let unscheduled_consumers = consumers[idx]
        .iter()
        .filter(|&&c| !scheduled.contains(&c))
        .count();

    pressure_policy::SchedulingCandidate {
        op_index: idx,
        luc,
        words_added,
        words_freed,
        asap: asap_map.get(&ops[idx].id).copied(),
        alap: alap_map.get(&ops[idx].id).copied(),
        engine: ops[idx].engine,
        is_shiftable: shiftable_set.contains(&idx),
        unscheduled_consumer_count: unscheduled_consumers,
    }
}

/// Schedule SSA ops using a greedy list scheduler with VALU→ALU shifting
/// and scratch budget constraint.
///
/// When VALU slots are saturated but ALU has capacity, shiftable VALU ops
/// (arithmetic ops that can be expanded to 8 scalar ALU ops) are shifted.
///
/// The scratch_budget parameter limits the maximum number of words that can
/// be live simultaneously. If scheduling an op would exceed this budget,
/// the op is delayed to a later cycle.
///
/// Returns `(cycles, budget_exceeded, peak_live)` where:
/// - `cycles`: Total number of cycles in the schedule
/// - `budget_exceeded`: True if force-scheduling was needed (budget violated)
/// - `peak_live`: Peak number of live words observed during scheduling
pub(super) fn schedule_ssa_greedy_with_shift_budget(
    ops: &mut [SsaOp],
    enable_valu_shift: bool,
    shifted_ops: &mut HashSet<usize>,
    scratch_budget: usize,
) -> Result<(usize, bool, usize), String> {
    schedule_ssa_greedy_with_shift_budget_and_selectchoice(
        ops,
        enable_valu_shift,
        shifted_ops,
        scratch_budget,
        None,
        None,
        None,
        None,
        &GreedyBoundsUsage::None,
        &GreedyPolicy,
        None,
        false,
        false,
        false,
        false,
        None,
        None,
        None,
    )
}

/// Schedule SSA ops with SelectChoice expansion support, without detailed diagnostics.
///
/// Same as [`schedule_ssa_greedy_with_diagnostics`] with a non-verbose, discarded
/// [`GreedyDiagnostics`]. Returns (cycles, budget_exceeded, peak_live).
#[allow(dead_code)]
pub(super) fn schedule_ssa_greedy_with_shift_budget_and_selectchoice(
    ops: &mut [SsaOp],
    enable_valu_shift: bool,
    shifted_ops: &mut HashSet<usize>,
    scratch_budget: usize,
    select_choice_fma_ops: Option<&mut HashSet<usize>>,
    select_sub_cycles: Option<&mut HashMap<usize, i32>>,
    select_fma_cycles: Option<&mut HashMap<usize, i32>>,
    const_add_imm_ops: Option<&mut HashSet<usize>>,
    bounds_usage: &GreedyBoundsUsage,
    strategy: &dyn PressurePolicy,
    vector_group_lanes: Option<&HashMap<VReg, Vec<VReg>>>,
    enable_capacity_deferral: bool,
    const_choice_fixed_to_load: bool,
    select_choice_fixed_to_vselect: bool,
    valu_shift_fixed_to_none: bool,
    // When Some, use these encodings for ConstChoice (engine from map); do not write to const_add_imm_ops.
    fixed_const_choice_encoding: Option<&HashMap<usize, ConstChoiceEncoding>>,
    // When Some, use this set for shift decisions (op in set => place as shifted); do not add to shifted_ops.
    fixed_shifted_ops: Option<&HashSet<usize>>,
    // When Some (e.g. ScheduleWithFixedChoices), use these encodings for SelectChoice (Fma => VALU, Vselect => Flow).
    fixed_select_choice_encoding: Option<&HashMap<usize, SelectChoiceEncoding>>,
) -> Result<(usize, bool, usize), String> {
    schedule_ssa_greedy_with_diagnostics(
        ops,
        enable_valu_shift,
        shifted_ops,
        scratch_budget,
        select_choice_fma_ops,
        select_sub_cycles,
        select_fma_cycles,
        const_add_imm_ops,
        bounds_usage,
        strategy,
        vector_group_lanes,
        enable_capacity_deferral,
        const_choice_fixed_to_load,
        select_choice_fixed_to_vselect,
        valu_shift_fixed_to_none,
        fixed_const_choice_encoding,
        fixed_shifted_ops,
        fixed_select_choice_encoding,
        &mut GreedyDiagnostics::new(false),
    )
}

/// Schedule SSA ops with SelectChoice expansion support.
/// Returns (cycles, budget_exceeded, peak_live); `diagnostics` records how often and how far
/// the run went over budget (see [`GreedyDiagnostics`]).
///
/// **VALU→ALU shifting (design choice):** We do not prefer shifting in greedy. We only shift
/// when VALU slots are full (6/6); otherwise we keep the op on VALU. Shift decisions are left
/// to CP-SAT, which optimizes them globally. Greedy provides cycle hints; CP-SAT decides which
/// ops to shift.
///
/// SelectChoice tracking is self-contained: ops report their relationship via methods.
/// The sub and fma ops know they belong to a SelectChoice op through `select_choice_parent`.
///
/// When hint passes are enabled, the pipeline passes fixed-default flags so the hint pass
/// "owns" the decision: const_choice_fixed_to_load (all ConstChoice use Load), 
/// select_choice_fixed_to_vselect (no FMA), valu_shift_fixed_to_none (no shifts).
pub(super) fn schedule_ssa_greedy_with_diagnostics(
    ops: &mut [SsaOp],
    enable_valu_shift: bool,
    shifted_ops: &mut HashSet<usize>,
    scratch_budget: usize,
    select_choice_fma_ops: Option<&mut HashSet<usize>>,
    select_sub_cycles: Option<&mut HashMap<usize, i32>>,
    select_fma_cycles: Option<&mut HashMap<usize, i32>>,
    const_add_imm_ops: Option<&mut HashSet<usize>>,
    bounds_usage: &GreedyBoundsUsage,
    strategy: &dyn PressurePolicy,
    vector_group_lanes: Option<&HashMap<VReg, Vec<VReg>>>,
    enable_capacity_deferral: bool,
    const_choice_fixed_to_load: bool,
    select_choice_fixed_to_vselect: bool,
    valu_shift_fixed_to_none: bool,
    // When Some, use these encodings for ConstChoice (engine from map); do not write to const_add_imm_ops.
    fixed_const_choice_encoding: Option<&HashMap<usize, ConstChoiceEncoding>>,
    // When Some, use this set for shift decisions (op in set => place as shifted); do not add to shifted_ops.
    fixed_shifted_ops: Option<&HashSet<usize>>,
    // When Some (e.g. ScheduleWithFixedChoices), use these encodings for SelectChoice (Fma => VALU, Vselect => Flow).
    fixed_select_choice_encoding: Option<&HashMap<usize, SelectChoiceEncoding>>,
    // Verbosity in, budget-overflow record out. Detailed dumps go to stderr only when `verbose`.
    diagnostics: &mut GreedyDiagnostics,
) -> Result<(usize, bool, usize), String> {
    schedule_ssa_greedy_with_liveness_agreement(
        ops,
        enable_valu_shift,
        shifted_ops,
        scratch_budget,
        select_choice_fma_ops,
        select_sub_cycles,
        select_fma_cycles,
        const_add_imm_ops,
        bounds_usage,
        strategy,
        vector_group_lanes,
        enable_capacity_deferral,
        const_choice_fixed_to_load,
        select_choice_fixed_to_vselect,
        valu_shift_fixed_to_none,
        fixed_const_choice_encoding,
        fixed_shifted_ops,
        fixed_select_choice_encoding,
        diagnostics,
    )
    .map(|(result, _agreement)| result)
}

/// Same as [`schedule_ssa_greedy_with_diagnostics`], and also returns the
/// comparison of the scheduler's live-word counter with interval liveness of the schedule it
/// produced (see [`LivenessAgreement`]).
///
/// # Returns
/// `((cycles, budget_exceeded, peak_live), agreement)`. `Err` if scheduling fails or the
/// counter disagrees with the intervals (a bug in the counter, never a property of the input;
/// set `SCHEDULER_LIVENESS_AGREEMENT=warn` to continue with a warning).
pub(super) fn schedule_ssa_greedy_with_liveness_agreement(
    ops: &mut [SsaOp],
    enable_valu_shift: bool,
    shifted_ops: &mut HashSet<usize>,
    scratch_budget: usize,
    select_choice_fma_ops: Option<&mut HashSet<usize>>,
    select_sub_cycles: Option<&mut HashMap<usize, i32>>,
    select_fma_cycles: Option<&mut HashMap<usize, i32>>,
    const_add_imm_ops: Option<&mut HashSet<usize>>,
    bounds_usage: &GreedyBoundsUsage,
    strategy: &dyn PressurePolicy,
    vector_group_lanes: Option<&HashMap<VReg, Vec<VReg>>>,
    enable_capacity_deferral: bool,
    const_choice_fixed_to_load: bool,
    select_choice_fixed_to_vselect: bool,
    valu_shift_fixed_to_none: bool,
    // When Some, use these encodings for ConstChoice (engine from map); do not write to const_add_imm_ops.
    fixed_const_choice_encoding: Option<&HashMap<usize, ConstChoiceEncoding>>,
    // When Some, use this set for shift decisions (op in set => place as shifted); do not add to shifted_ops.
    fixed_shifted_ops: Option<&HashSet<usize>>,
    // When Some (e.g. ScheduleWithFixedChoices), use these encodings for SelectChoice (Fma => VALU, Vselect => Flow).
    fixed_select_choice_encoding: Option<&HashMap<usize, SelectChoiceEncoding>>,
    // Verbosity in, budget-overflow record out. Detailed dumps go to stderr only when `verbose`.
    diagnostics: &mut GreedyDiagnostics,
) -> Result<((usize, bool, usize), LivenessAgreement), String> {
    let mut trace = CounterTrace::default();
    let result = schedule_ssa_greedy_core(
        ops,
        enable_valu_shift,
        shifted_ops,
        scratch_budget,
        select_choice_fma_ops,
        select_sub_cycles,
        select_fma_cycles,
        const_add_imm_ops,
        bounds_usage,
        strategy,
        vector_group_lanes,
        enable_capacity_deferral,
        const_choice_fixed_to_load,
        select_choice_fixed_to_vselect,
        valu_shift_fixed_to_none,
        fixed_const_choice_encoding,
        fixed_shifted_ops,
        fixed_select_choice_encoding,
        &mut trace,
        diagnostics,
    )?;
    let agreement = compare_counter_with_intervals(ops, vector_group_lanes, &trace, result.2)?;
    // SCHEDULER_LIVENESS_AGREEMENT: any value prints the comparison; `warn` also turns a
    // disagreement from an error into a warning.
    let mode = std::env::var("SCHEDULER_LIVENESS_AGREEMENT").ok();
    if mode.is_some() {
        eprintln!("Greedy liveness agreement: {}", agreement);
    }
    if !agreement.agrees() {
        let message = format!(
            "BUG: greedy live-word counter disagrees with interval liveness of its own schedule: {}\n  {}",
            agreement,
            agreement.value_mismatches.join("\n  ")
        );
        if mode.as_deref() == Some("warn") {
            eprintln!("Warning: {}", message);
        } else {
            return Err(message);
        }
    }
    Ok((result, agreement))
}

fn schedule_ssa_greedy_core(
    ops: &mut [SsaOp],
    enable_valu_shift: bool,
    shifted_ops: &mut HashSet<usize>,
    scratch_budget: usize,
    select_choice_fma_ops: Option<&mut HashSet<usize>>,
    select_sub_cycles: Option<&mut HashMap<usize, i32>>,
    select_fma_cycles: Option<&mut HashMap<usize, i32>>,
    mut const_add_imm_ops: Option<&mut HashSet<usize>>,
    bounds_usage: &GreedyBoundsUsage,
    strategy: &dyn PressurePolicy,
    vector_group_lanes: Option<&HashMap<VReg, Vec<VReg>>>,
    enable_capacity_deferral: bool,
    const_choice_fixed_to_load: bool,
    select_choice_fixed_to_vselect: bool,
    valu_shift_fixed_to_none: bool,
    // When Some, use these encodings for ConstChoice (engine from map); do not write to const_add_imm_ops.
    fixed_const_choice_encoding: Option<&HashMap<usize, ConstChoiceEncoding>>,
    // When Some, use this set for shift decisions (op in set => place as shifted); do not add to shifted_ops.
    fixed_shifted_ops: Option<&HashSet<usize>>,
    // When Some (e.g. ScheduleWithFixedChoices), use these encodings for SelectChoice (Fma => VALU, Vselect => Flow).
    fixed_select_choice_encoding: Option<&HashMap<usize, SelectChoiceEncoding>>,
    // Out: what the live-word counter did, for the comparison with interval liveness.
    trace: &mut CounterTrace,
    // Verbosity in, budget-overflow record out. Detailed dumps go to stderr only when `verbose`.
    diagnostics: &mut GreedyDiagnostics,
) -> Result<(usize, bool, usize), String> {
    if ops.is_empty() {
        return Ok((0, false, 0));
    }

    // Build op_id -> index mapping
    let id_to_idx: HashMap<usize, usize> = ops
        .iter()
        .enumerate()
        .map(|(idx, op)| (op.id, idx))
        .collect();

    // Build shiftable set for PrecedenceOffset computation
    let shiftable_set: HashSet<usize> = if enable_valu_shift {
        ops.iter()
            .enumerate()
            .filter(|(_, op)| op.engine == Engine::Valu && can_ssa_valu_shift_to_alu(op))
            .map(|(idx, _)| idx)
            .collect()
    } else {
        HashSet::new()
    };

    // =========================================================================
    // Constant Alternative Detection
    //
    // Identify AddImm ops that are alternatives to Const (Load) ops for the
    // same constant value. Pattern: Const op followed by AddImm op with
    // same destination VReg. The greedy scheduler uses Load; AddImm
    // alternatives are only considered by CP-SAT.
    //
    // We mark AddImm alternatives as dead so they're not scheduled by greedy.
    // =========================================================================
    let mut const_add_imm_alternatives: HashSet<usize> = HashSet::new();
    for (idx, op) in ops.iter().enumerate() {
        // Check if this is an AddImm op (Engine::Flow, Opcode::AddImm)
        if op.engine == Engine::Flow && op.opcode == Opcode::AddImm {
            if let Some(dest) = op.dest {
                // Check if there's an earlier Const op with the same dest
                // Look backwards to find the Load alternative
                for prev_idx in (0..idx).rev() {
                    let prev_op = &ops[prev_idx];
                    if prev_op.dest == Some(dest) {
                        if prev_op.engine == Engine::Load && prev_op.opcode == Opcode::Const {
                            // Found the Load alternative - this AddImm is an alternative
                            const_add_imm_alternatives.insert(idx);
                        }
                        // Stop searching after finding the first op with same dest
                        break;
                    }
                }
            }
        }
    }

    // Pre-compute VReg information for scratch tracking
    // Skip AddImm alternatives when building vreg_def (use Load's definition)
    let mut vreg_def: HashMap<VReg, usize> = HashMap::new();
    let mut vreg_users: HashMap<VReg, HashSet<usize>> = HashMap::new();

    for (idx, op) in ops.iter().enumerate() {
        if let Some(dest) = op.dest {
            // Don't overwrite Load's entry with AddImm alternative
            if !const_add_imm_alternatives.contains(&idx) {
                vreg_def.insert(dest, idx);
            }
        }
        // Still track users from AddImm alternatives (they read zero_vreg).
        // `keep_live` values count as read: address assignment holds them until this op.
        for vreg in liveness_read_vregs(op) {
            vreg_users.entry(vreg).or_default().insert(idx);
        }
    }

    // Vector group awareness: bidirectional user propagation between group aliases
    // and their lane VRegs. This prevents dead store detection from incorrectly
    // eliminating ops that produce values consumed through vector group indirection.
    //
    // Write groups (fresh_vector_group): alias → lanes.
    //   If the group alias has users, all lane VRegs are implicitly used.
    // Read groups (decompose_vector): lanes → group key.
    //   If any lane VReg has users, the group key's definer is implicitly used.
    let mut liveness_aliases: HashMap<VReg, Vec<VReg>> = HashMap::new();
    // Write-group lane → group VReg. A write group is one block of VLEN words held from
    // its first lane write to its last read (as in address assignment), so the counter
    // tracks the group VReg and not the individual lanes.
    let mut write_group_of: HashMap<VReg, VReg> = HashMap::new();
    if let Some(groups) = vector_group_lanes {
        for (group_vreg, lanes) in groups {
            // Liveness release mirrors the user propagation below: every reader of a lane
            // releases the group VReg (whose users are all lane readers), and a write
            // group's lanes are released by the readers of the group VReg.
            for lane_vreg in lanes {
                liveness_aliases.entry(*lane_vreg).or_default().push(*group_vreg);
            }
            if !vreg_def.contains_key(group_vreg) {
                liveness_aliases.entry(*group_vreg).or_default().extend(lanes);
                for lane_vreg in lanes {
                    write_group_of.insert(*lane_vreg, *group_vreg);
                }
            }
            // Write group direction: alias users → lane VRegs
            if let Some(group_users) = vreg_users.get(group_vreg).cloned() {
                for lane_vreg in lanes {
                    vreg_users.entry(*lane_vreg).or_default().extend(&group_users);
                }
            }
            // Read group direction: lane users → group key
            let lane_users: HashSet<usize> = lanes
                .iter()
                .filter_map(|lv| vreg_users.get(lv))
                .flat_map(|s| s.iter().copied())
                .collect();
            if !lane_users.is_empty() {
                vreg_users.entry(*group_vreg).or_default().extend(&lane_users);
            }
        }
    }

    // =========================================================================
    // Dead Store Detection
    //
    // Identify ops with destinations that have zero consumers. These are
    // "dead stores" - they compute values that no one reads. We detect
    // them here and will pre-mark them as scheduled (see below) so they
    // never enter the ready queue. The warnings are informational only.
    // =========================================================================
    let mut dead_store_indices: HashSet<usize> = HashSet::new();
    for (idx, op) in ops.iter().enumerate() {
        // Skip AddImm alternatives - they're handled separately below
        if const_add_imm_alternatives.contains(&idx) {
            continue;
        }
        if let Some(dest) = op.dest {
            let user_count = vreg_users.get(&dest).map_or(0, |s| s.len());
            if user_count == 0 {
                dead_store_indices.insert(idx);
                // Dead store warnings disabled by default - enable with SCHEDULER_DEBUG=1
                if std::env::var("SCHEDULER_DEBUG").is_ok() {
                    eprintln!(
                        "Warning: Dead store detected (will skip): op[{}] id={} defines {:?} with 0 users.\n\
                         Source: {:?}\n\
                         Engine: {:?}, Opcode: {:?}, srcs={:?}",
                        idx, op.id, dest, op.source, op.engine, op.opcode, op.srcs
                    );
                }
            }
        }
    }

    // Add AddImm constant alternatives to dead stores - greedy always uses Load
    // (Note: These are not truly "dead", they're just unused by greedy.
    //  CP-SAT may choose to use them instead of the Load alternative.)
    dead_store_indices.extend(&const_add_imm_alternatives);

    // Summary warnings disabled by default - enable with SCHEDULER_DEBUG=1
    if std::env::var("SCHEDULER_DEBUG").is_ok() && !dead_store_indices.is_empty() {
        let true_dead_stores = dead_store_indices.len() - const_add_imm_alternatives.len();
        if true_dead_stores > 0 {
            eprintln!(
                "Warning: {} dead stores detected and will be skipped",
                true_dead_stores
            );
        }
        if !const_add_imm_alternatives.is_empty() {
            eprintln!(
                "Info: {} constant AddImm alternatives skipped (greedy uses Load)",
                const_add_imm_alternatives.len()
            );
        }
    }

    // Lane writes still to come per write group. The block is held until the last of them
    // even when every reader of the group has already run (or none ever will).
    let mut write_groups = WriteGroups { of: write_group_of, ..Default::default() };
    for (lane, group) in &write_groups.of {
        if vreg_def.get(lane).map_or(false, |def_idx| !dead_store_indices.contains(def_idx)) {
            *write_groups.pending_writes.entry(*group).or_default() += 1;
        }
    }

    // Build reverse dependency map: op_idx -> list of consumer indices
    // This enables O(1) lookup of which ops depend on a given op
    let mut consumers: Vec<Vec<usize>> = vec![vec![]; ops.len()];
    for (idx, op) in ops.iter().enumerate() {
        for &dep_id in &op.dep_ids {
            if let Some(&dep_idx) = id_to_idx.get(&dep_id) {
                consumers[dep_idx].push(idx);
            }
        }
    }

    // Track remaining unscheduled dependencies per op
    // Only count deps that exist in the ops list and are not dead stores (dead stores are pre-satisfied).
    // Deps that don't exist in id_to_idx (e.g. external/phantom deps) are treated as satisfied.
    let mut remaining_deps: Vec<usize> = ops
        .iter()
        .enumerate()
        .map(|(_op_idx, op)| {
            op.dep_ids
                .iter()
                .filter(|dep_id| {
                    let Some(&dep_idx) = id_to_idx.get(dep_id) else {
                        return false; // external/phantom dep - treat as satisfied
                    };
                    !dead_store_indices.contains(&dep_idx)
                })
                .count()
        })
        .collect();

    // Track the earliest cycle each consumer can execute, accounting for
    // per-predecessor precedence offsets. Updated when each predecessor is scheduled.
    let mut max_ready_cycle: Vec<i32> = vec![0; ops.len()];

    // Track scheduling state
    // =========================================================================
    // Dead Store Elimination
    //
    // Pre-mark dead stores as "already scheduled" - this is how we eliminate
    // them. They won't enter the ready heap and won't be counted in
    // ops_to_schedule. Their cycle field remains at -1 (unscheduled), which
    // signals to CP-SAT that these ops should be skipped entirely.
    // =========================================================================
    let mut scheduled: HashSet<usize> = dead_store_indices.clone();
    let ops_to_schedule = ops.len() - dead_store_indices.len();

    // Readers that will be scheduled. Skipped dead stores never run, so counting them
    // would keep their operands live forever.
    let mut vreg_remaining_users: HashMap<VReg, usize> = vreg_users
        .iter()
        .map(|(vreg, users)| {
            (*vreg, users.iter().filter(|u| !dead_store_indices.contains(u)).count())
        })
        .collect();
    let mut live_words: usize = 0;
    let mut live_vregs: HashSet<VReg> = HashSet::new();

    // Budget tracking
    let mut peak_live_words: usize = 0;
    let mut budget_exceeded = false;

    // Track when each VReg was defined (used for debugging/validation)
    let mut vreg_def_cycle: HashMap<VReg, i32> = HashMap::new();

    // Ready heap: (priority, op_idx) - min-heap via Reverse
    // Priority is the earliest cycle this op can be scheduled (ASAP)
    let mut ready_heap: BinaryHeap<Reverse<(i32, usize)>> = BinaryHeap::new();

    // Initialize with ops that have no dependencies (excluding dead stores)
    for (idx, op) in ops.iter().enumerate() {
        if remaining_deps[idx] == 0 && !dead_store_indices.contains(&idx) {
            let priority = bounds_usage
                .core_bounds()
                .and_then(|(a, _)| a.get(&op.id).copied())
                .unwrap_or(0);
            ready_heap.push(Reverse((priority, idx)));
        }
    }

    // VReg expiry heap: (expiry_cycle, vreg) - min-heap via Reverse
    // VRegs expire (can be freed) at the cycle AFTER their last use
    let mut vreg_expiry_heap: BinaryHeap<Reverse<(i32, VReg)>> = BinaryHeap::new();

    let mut cycle = 0;
    let max_cycles = 50000;
    // Track consecutive cycles with no ops scheduled (for ASAP inflation breaker)
    let mut consecutive_empty: usize = 0;

    // ASAP inflation diagnostics
    let mut inflation_stats = InflationStats::default();

    // Capacity-aware deferral: per-engine demand profile (only when Full bounds + enabled)
    let mut demand_profile: Option<IncrementalDemandProfile> = None;
    if enable_capacity_deferral {
        if let Some((asap_map, alap_map)) = bounds_usage.core_bounds() {
            let horizon = alap_map.values().copied().max().unwrap_or(5000) as usize;
            demand_profile = Some(IncrementalDemandProfile::new(
                ops,
                asap_map,
                &shiftable_set,
                &dead_store_indices,
                horizon,
            ));
        }
    }

    while scheduled.len() < ops.len() && cycle < max_cycles {
        // Note: scheduled includes dead stores (pre-marked), so we check against ops.len()
        // Free expired VRegs (O(log n) per VReg)
        // VRegs expire at the cycle AFTER their last use
        while let Some(&Reverse((expiry_cycle, vreg))) = vreg_expiry_heap.peek() {
            if expiry_cycle <= cycle as i32 {
                vreg_expiry_heap.pop();
                if live_vregs.remove(&vreg) {
                    live_words = live_words.saturating_sub(vreg.size());
                    trace.release(&vreg, cycle);
                }
            } else {
                break;
            }
        }

        let mut slots = SlotTracker::new();

        // Collect ready ops from heap
        // Ops in the heap may need to be deferred if their deps aren't truly ready
        // (scheduled in previous cycle) or if slots/budget are exhausted
        let mut ready_this_cycle: Vec<usize> = Vec::new();
        let mut reinsert: Vec<(i32, usize)> = Vec::new();

        while let Some(Reverse((priority, idx))) = ready_heap.pop() {
            if scheduled.contains(&idx) {
                continue; // Already scheduled (shouldn't happen but defensive)
            }

            // Check if deps are truly satisfied (predecessor scheduled early enough
            // that result is visible by this cycle, accounting for precedence offsets).
            // Dead stores are treated as already satisfied (they're skipped entirely).
            // Deps not in id_to_idx (external/phantom) are treated as satisfied.
            let deps_ready = ops[idx].dep_ids.iter().all(|dep_id| {
                let Some(&dep_idx) = id_to_idx.get(dep_id) else {
                    return true; // external/phantom dep - treat as satisfied
                };
                dead_store_indices.contains(&dep_idx)
                    || (ops[dep_idx].cycle >= 0 && {
                        let offset = ops[dep_idx]
                            .precedence_offset_for_asap(shiftable_set.contains(&dep_idx));
                        ops[dep_idx].cycle + offset <= cycle as i32
                    })
            });

            if deps_ready {
                ready_this_cycle.push(idx);
            } else {
                // Not ready yet - reinsert with updated priority
                let new_priority = bounds_usage
                    .core_bounds()
                    .and_then(|(a, _)| a.get(&ops[idx].id).copied())
                    .map(|lb| priority.max(cycle as i32).max(lb))
                    .unwrap_or_else(|| priority.max(cycle as i32));
                reinsert.push((new_priority, idx));
            }
        }

        // Put deferred ops back into heap
        for (priority, idx) in reinsert {
            ready_heap.push(Reverse((priority, idx)));
        }

        // Try to schedule ready ops with retry for budget-constrained ops.
        // When an op frees VRegs, previously budget-failed ops may now fit.
        let mut budget_deferred: Vec<usize> = Vec::new();
        let mut made_progress = true;
        let mut scheduled_this_cycle: usize = 0;

        // Helper: compute LUC (Last-Use-Count) for an op - count of source VRegs
        // where this op is their last reader. Higher LUC = frees more scratch space.
        fn compute_luc(
            ops: &[SsaOp],
            vreg_remaining_users: &HashMap<VReg, usize>,
            idx: usize,
        ) -> usize {
            ops[idx]
                .srcs
                .iter()
                .filter(|src| {
                    if let Operand::VReg(vreg) = src {
                        vreg_remaining_users.get(vreg).map_or(false, |&n| n == 1)
                    } else {
                        false
                    }
                })
                .count()
        }

        // Build SchedulingCandidate structs for strategy-driven sort.
        let mut ready_candidates: Vec<pressure_policy::SchedulingCandidate> = ready_this_cycle
            .iter()
            .map(|&idx| {
                let luc = compute_luc(ops, &vreg_remaining_users, idx);
                let words_added = ops[idx].dest.map_or(0, |d| d.size());
                let words_freed_est = ops[idx]
                    .srcs
                    .iter()
                    .filter(|src| {
                        if let Operand::VReg(vreg) = src {
                            vreg_remaining_users.get(vreg).map_or(false, |&n| n == 1)
                                && live_vregs.contains(vreg)
                        } else {
                            false
                        }
                    })
                    .map(|src| {
                        if let Operand::VReg(vreg) = src {
                            vreg.size()
                        } else {
                            0
                        }
                    })
                    .sum();
                let (asap_val, alap_val) = if let Some((asap_map, alap_map)) = bounds_usage.policy_bounds() {
                    (
                        asap_map.get(&ops[idx].id).copied(),
                        alap_map.get(&ops[idx].id).copied(),
                    )
                } else {
                    (None, None)
                };
                let unscheduled_consumers = consumers[idx]
                    .iter()
                    .filter(|&&c| !scheduled.contains(&c))
                    .count();
                pressure_policy::SchedulingCandidate {
                    op_index: idx,
                    luc,
                    words_added,
                    words_freed: words_freed_est,
                    asap: asap_val,
                    alap: alap_val,
                    engine: ops[idx].engine,
                    is_shiftable: shiftable_set.contains(&idx),
                    unscheduled_consumer_count: unscheduled_consumers,
                }
            })
            .collect();
        let sched_ctx = pressure_policy::SchedulingContext {
            cycle,
            live_words,
            scratch_budget,
            slots: &slots.view(),
        };
        strategy.sort_ready(&mut ready_candidates, &sched_ctx);
        ready_this_cycle = ready_candidates.into_iter().map(|c| c.op_index).collect();

        // Keep retrying as long as we make progress (schedule something or free VRegs)
        while made_progress {
            made_progress = false;

            // Sort budget_deferred by LUC (descending) to prioritize ops that free VRegs
            budget_deferred.sort_by(|&a, &b| {
                compute_luc(ops, &vreg_remaining_users, b).cmp(&compute_luc(
                    ops,
                    &vreg_remaining_users,
                    a,
                ))
            });

            // Defer ops that are ready by deps but before their ASAP when bounds are provided.
            // ASAP inflation: when pressure is high, defer high-slack producers by inflating
            // their effective ASAP, creating breathing room for consumers to free VRegs.
            let mut deferred_by_bounds: Vec<(i32, usize)> = Vec::new();
            let mut ready_for_try: Vec<usize> = Vec::new();

            // ASAP inflation safeguards
            let inflation_disabled = consecutive_empty >= 3;
            let ready_this_cycle_len = ready_this_cycle.len();
            let mut inflation_deferred_count: usize = 0;
            // At least MIN_PASS_THROUGH ops must pass without inflation per cycle
            const MIN_PASS_THROUGH: usize = 3;

            // Build context for inflation decisions
            let inflation_ctx = pressure_policy::SchedulingContext {
                cycle,
                live_words,
                scratch_budget,
                slots: &slots.view(),
            };

            // Capacity deferral: min pass-through safeguard (same pattern as ASAP inflation)
            const MIN_CAPACITY_PASS_THROUGH: usize = 3;
            let max_capacity_deferrals = ready_this_cycle_len.saturating_sub(MIN_CAPACITY_PASS_THROUGH);
            let mut capacity_deferred_count: usize = 0;
            let capacity_deferral_disabled = consecutive_empty >= 3;

            for idx in ready_this_cycle.drain(..) {
                // ASAP deferral: only in Full mode (core loop mechanics).
                // Defers ops whose ASAP cycle hasn't been reached yet.
                if let Some((asap_map, _)) = bounds_usage.core_bounds() {
                    let lb = *asap_map.get(&ops[idx].id).unwrap_or(&0);
                    if (cycle as i32) < lb {
                        deferred_by_bounds.push((lb, idx));
                        continue;
                    }
                }
                // Capacity-aware deferral: only in Full mode when enabled (contributing-only).
                // Uses energetic logic: defer if demand[engine][cycle] + slots > capacity.
                // Skip optional ops (ConstChoice, SelectChoice, shiftable) - they don't contribute.
                if let Some(ref mut profile) = demand_profile {
                    if let Some((asap_map, alap_map)) = bounds_usage.core_bounds() {
                        let op_asap = *asap_map.get(&ops[idx].id).unwrap_or(&0);
                        let o_contributes = op_asap == cycle as i32;
                        let is_optional = IncrementalDemandProfile::is_optional_on_engine(
                            idx,
                            ops[idx].engine,
                            &shiftable_set,
                            ops[idx].opcode,
                        );
                        if o_contributes
                            && !is_optional
                            && !capacity_deferral_disabled
                            && capacity_deferred_count < max_capacity_deferrals
                        {
                            let engine = ops[idx].engine;
                            if engine.is_hardware() {
                                let cap = engine_capacity(engine);
                                // min_demand already includes the current op (we haven't scheduled it yet)
                                let min_demand = profile.min_demand_at_cycle(cycle as i32, engine);
                                if min_demand > cap {
                                    let ub = *alap_map.get(&ops[idx].id).unwrap_or(&i32::MAX);
                                    let next_feasible = (cycle as i32 + 1).min(ub);
                                    deferred_by_bounds.push((next_feasible, idx));
                                    capacity_deferred_count += 1;
                                    continue;
                                }
                            }
                        }
                    }
                }
                // ASAP inflation: available in PolicyOnly + Full mode (policy decisions).
                // Defers high-slack producers when pressure is high to reduce concurrent liveness.
                if let Some((asap_map, alap_map)) = bounds_usage.policy_bounds() {
                    let max_inflation_deferrals =
                        ready_this_cycle_len.saturating_sub(MIN_PASS_THROUGH);
                    if inflation_disabled {
                        inflation_stats.breaker_activations += 1;
                    } else if inflation_deferred_count >= max_inflation_deferrals {
                        inflation_stats.pass_through_blocks += 1;
                    }
                    if !inflation_disabled && inflation_deferred_count < max_inflation_deferrals {
                        let ub = *alap_map.get(&ops[idx].id).unwrap_or(&i32::MAX);
                        let candidate = build_inflation_candidate(
                            ops,
                            &vreg_remaining_users,
                            &live_vregs,
                            &consumers,
                            &scheduled,
                            &shiftable_set,
                            asap_map,
                            alap_map,
                            idx,
                        );
                        if let Some(inflation) =
                            strategy.compute_asap_inflation(&candidate, &inflation_ctx)
                        {
                            // Net-liveness guard: only defer if benefit > cost.
                            // Benefit: P's output (words_added) won't be produced for k
                            // more cycles. Cost: P's input VRegs that are last-used by P
                            // stay live k additional cycles (can't be freed until P runs).
                            // When words_freed >= words_added, deferring extends more live
                            // data than it saves — causes the "deferred avalanche" blowup.
                            if candidate.words_freed >= candidate.words_added {
                                inflation_stats.net_liveness_blocks += 1;
                                ready_for_try.push(idx);
                                continue;
                            }

                            let inflated = ((cycle as i32) + inflation).min(ub);
                            if inflated > cycle as i32 {
                                deferred_by_bounds.push((inflated, idx));
                                inflation_deferred_count += 1;
                                inflation_stats.total_deferred += 1;
                                inflation_stats.total_inflation_cycles += inflation as i64;
                                if inflation > inflation_stats.max_inflation {
                                    inflation_stats.max_inflation = inflation;
                                }
                                continue;
                            }
                        }
                    }
                }
                ready_for_try.push(idx);
            }
            for (priority, idx) in deferred_by_bounds {
                ready_heap.push(Reverse((priority, idx)));
            }

            // Process ready_this_cycle first, then budget_deferred
            let ops_to_try: Vec<usize> = ready_for_try
                .into_iter()
                .chain(budget_deferred.drain(..))
                .collect();

            // Count ready Flow ops that are not ConstChoice (vselect, jump, halt, etc.).
            // We schedule ConstChoice as add_imm only when Flow is free and no other Flow op is ready.
            let flow_other_ready = ops_to_try
                .iter()
                .filter(|&&idx| {
                    !scheduled.contains(&idx)
                        && ops[idx].engine == Engine::Flow
                        && ops[idx].opcode != Opcode::ConstChoice
                })
                .count();

            for idx in ops_to_try {
                if scheduled.contains(&idx) {
                    continue;
                }

                // ConstChoice: use add_imm (Flow) only if there is a free Flow slot and no
                // other schedulable Flow op (vselect, jump, etc.); otherwise use Load.
                // When const_choice_fixed_to_load (hint pass enabled), always use Load so the hint pass owns the decision.
                // When fixed_const_choice_encoding is Some (schedule-with-fixed-choices), use the map.
                // SelectChoice: when fixed_select_choice_encoding is Some, use map (Fma => VALU, Vselect => Flow).
                let op_id = ops[idx].id;
                let engine = if ops[idx].opcode == Opcode::ConstChoice {
                    if let Some(enc) = fixed_const_choice_encoding.and_then(|m| m.get(&op_id)) {
                        if matches!(enc, ConstChoiceEncoding::AddImm) {
                            Engine::Flow
                        } else {
                            Engine::Load
                        }
                    } else if const_choice_fixed_to_load {
                        Engine::Load
                    } else if slots.flow_used < 1 && flow_other_ready == 0 {
                        Engine::Flow
                    } else {
                        Engine::Load
                    }
                } else if ops[idx].opcode == Opcode::SelectChoice {
                    if let Some(enc) = fixed_select_choice_encoding.and_then(|m| m.get(&op_id)) {
                        if matches!(enc, SelectChoiceEncoding::Fma) {
                            Engine::Valu
                        } else {
                            Engine::Flow
                        }
                    } else if select_choice_fixed_to_vselect {
                        Engine::Flow
                    } else {
                        ops[idx].engine // default Flow (vselect)
                    }
                } else {
                    ops[idx].engine
                };
                let dest = ops[idx].dest;

                // When fixed_shifted_ops is Some, can_shift = op is in the set (we place as shifted only those).
                // When in fixed_shifted_ops we must place as shifted only (never VALU), so we need ALU room.
                let can_shift = if let Some(fixed) = fixed_shifted_ops {
                    fixed.contains(&op_id)
                } else {
                    enable_valu_shift
                        && engine == Engine::Valu
                        && can_ssa_valu_shift_to_alu(&ops[idx])
                };
                let must_place_shifted = fixed_shifted_ops.map_or(false, |s| s.contains(&op_id));

                // Check slot availability
                if must_place_shifted {
                    if slots.alu_used + VLEN > 12 {
                        let next_priority = bounds_usage
                            .core_bounds()
                            .and_then(|(a, _)| a.get(&op_id).copied())
                            .map(|lb| ((cycle + 1) as i32).max(lb))
                            .unwrap_or_else(|| (cycle + 1) as i32);
                        ready_heap.push(Reverse((next_priority, idx)));
                        continue;
                    }
                } else if !slots.fits(engine, can_shift) {
                    // Reinsert to heap for next cycle
                    let next_priority = bounds_usage
                        .core_bounds()
                        .and_then(|(a, _)| a.get(&op_id).copied())
                        .map(|lb| ((cycle + 1) as i32).max(lb))
                        .unwrap_or_else(|| (cycle + 1) as i32);
                    ready_heap.push(Reverse((next_priority, idx)));
                    continue;
                }

                // Check scratch budget with same-cycle reuse model:
                // VRegs that will be freed when this op runs (their last use) reduce pressure.
                let words_added = match dest.map(|d| write_groups.counted(d)) {
                    // Later lanes of a write group land in the block its first lane reserved.
                    Some(counted) if live_vregs.contains(&counted) => 0,
                    Some(counted) => counted.size(),
                    None => 0,
                };
                // Exactly what the release loop below frees: every value (operand, keep_live
                // or vector-group alias) for which this op is the last remaining reader.
                let words_freed: usize = liveness_release_vregs(&ops[idx], &liveness_aliases)
                    .iter()
                    .filter(|vreg| {
                        vreg_remaining_users.get(vreg).map_or(false, |&n| n == 1)
                            && live_vregs.contains(vreg)
                            && write_groups.releasable_at(vreg, cycle)
                    })
                    .map(|vreg| vreg.size())
                    .sum();
                // Net pressure change: add new VReg, subtract freed VRegs (same-cycle reuse)
                let projected_live = live_words + words_added - words_freed;
                if projected_live > scratch_budget {
                    // Defer for retry - liveness may decrease after other ops scheduled
                    budget_deferred.push(idx);
                    continue;
                }
                // Strategy-driven soft deferral (pressure awareness)
                if strategy.should_defer_for_pressure(
                    live_words,
                    words_added,
                    words_freed,
                    scratch_budget,
                ) {
                    budget_deferred.push(idx);
                    continue;
                }

                // Schedule the op via SlotTracker.
                // When must_place_shifted (op in fixed_shifted_ops), use shift-only path so hint matches (ALU count).
                let allow_shift = strategy.allow_shift(live_words, slots.alu_used, scratch_budget);
                let place_result = if must_place_shifted {
                    slots.try_place_shifted_only()
                } else {
                    slots.try_place(engine, can_shift, allow_shift)
                };
                match place_result {
                    SlotResult::Placed => {
                        // Record ConstChoice scheduled as add_imm for lowering (skip when hint pass or fixed encoding owns)
                        if fixed_const_choice_encoding.is_none()
                            && !const_choice_fixed_to_load
                            && engine == Engine::Flow
                            && ops[idx].opcode == Opcode::ConstChoice
                        {
                            if let Some(ref mut set) = const_add_imm_ops {
                                set.insert(op_id);
                            }
                        }
                    }
                    SlotResult::Shifted => {
                        if fixed_shifted_ops.is_none() && !valu_shift_fixed_to_none {
                            shifted_ops.insert(op_id);
                        }
                    }
                    SlotResult::NoRoom => {
                        // VALU full, shift suppressed by strategy — defer to next cycle
                        let next_priority = bounds_usage
                            .core_bounds()
                            .and_then(|(a, _)| a.get(&op_id).copied())
                            .map(|lb| ((cycle + 1) as i32).max(lb))
                            .unwrap_or_else(|| (cycle + 1) as i32);
                        ready_heap.push(Reverse((next_priority, idx)));
                        continue;
                    }
                }

                // Record the cycle we actually placed in so slot counts match the hint.
                // Clamping to [asap, alap] would record a different cycle and cause
                // "Merged hint violates engine capacity" when multiple ops get clamped
                // into the same cycle.
                let assigned_cycle = cycle as i32;
                ops[idx].cycle = assigned_cycle;
                scheduled.insert(idx);
                made_progress = true;
                scheduled_this_cycle += 1;

                // Update capacity demand profile: op no longer contributes at its ASAP cycle
                if let (Some(ref mut profile), Some((asap_map, _))) =
                    (demand_profile.as_mut(), bounds_usage.core_bounds())
                {
                    let asap_t = *asap_map.get(&ops[idx].id).unwrap_or(&0);
                    profile.schedule_op(idx, &ops[idx], asap_t, &shiftable_set);
                }

                // Update live tracking for destination VReg
                if let Some(d) = dest {
                    count_written_dest(
                        d,
                        cycle,
                        &mut write_groups,
                        &vreg_remaining_users,
                        &mut live_vregs,
                        &mut live_words,
                        &mut vreg_expiry_heap,
                        trace,
                    );
                    // Track when this VReg was defined
                    vreg_def_cycle.insert(d, cycle as i32);
                }

                // Update remaining users and last use cycle for source VRegs
                // With same-cycle reuse model: VRegs can be freed immediately after their
                // last use is scheduled, allowing other ops in the same cycle to reuse the space.
                release_read_values(
                    &ops[idx],
                    cycle,
                    &liveness_aliases,
                    &write_groups,
                    &mut vreg_remaining_users,
                    &mut live_vregs,
                    &mut live_words,
                    &mut vreg_expiry_heap,
                    trace,
                );

                // Record peak after both adds and frees, matching the same-cycle
                // reuse model used by address assignment's interval-based liveness.
                peak_live_words = peak_live_words.max(live_words);

                // Add newly-ready consumers to heap (O(log n) per consumer)
                for &consumer_idx in &consumers[idx] {
                    // Update max ready cycle from this predecessor's offset
                    let offset = ops[idx]
                        .precedence_offset_for_asap(shiftable_set.contains(&idx));
                    max_ready_cycle[consumer_idx] =
                        max_ready_cycle[consumer_idx].max((cycle as i32) + offset);

                    remaining_deps[consumer_idx] -= 1;
                    if remaining_deps[consumer_idx] == 0 {
                        let priority = bounds_usage
                            .core_bounds()
                            .and_then(|(a, _)| a.get(&ops[consumer_idx].id).copied())
                            .map(|lb| max_ready_cycle[consumer_idx].max(lb))
                            .unwrap_or(max_ready_cycle[consumer_idx]);
                        ready_heap.push(Reverse((priority, consumer_idx)));
                    }
                }
            }
        }

        // ============================================================================
        // EMERGENCY FALLBACK: FORCE-SCHEDULE MECHANISM
        // ============================================================================
        // IMPORTANT: DO NOT REMOVE THIS CODE DURING DEBUGGING UNLESS EXPLICITLY
        // INSTRUCTED BY THE USER. This force-schedule mechanism is intentional and
        // necessary to prevent deadlock when the greedy scheduler cannot make progress
        // due to budget constraints. While it may exceed the scratch budget, it ensures
        // the scheduler completes and produces a valid (though potentially suboptimal)
        // schedule.
        //
        // Emergency fallback: if we made no progress and have budget-deferred ops,
        // force-schedule the highest-LUC op to break the deadlock.
        // This temporarily exceeds the budget but allows progress.
        // ============================================================================
        if !budget_deferred.is_empty() {
            // Sort by LUC and pick the best one
            budget_deferred.sort_by(|&a, &b| {
                compute_luc(ops, &vreg_remaining_users, b).cmp(&compute_luc(
                    ops,
                    &vreg_remaining_users,
                    a,
                ))
            });

            let force_idx = budget_deferred.remove(0);
            let op_id = ops[force_idx].id;
            let engine = if ops[force_idx].opcode == Opcode::ConstChoice {
                if let Some(enc) = fixed_const_choice_encoding.and_then(|m| m.get(&op_id)) {
                    if matches!(enc, ConstChoiceEncoding::AddImm) {
                        Engine::Flow
                    } else {
                        Engine::Load
                    }
                } else {
                    ops[force_idx].engine
                }
            } else if ops[force_idx].opcode == Opcode::SelectChoice {
                if let Some(enc) = fixed_select_choice_encoding.and_then(|m| m.get(&op_id)) {
                    if matches!(enc, SelectChoiceEncoding::Fma) {
                        Engine::Valu
                    } else {
                        Engine::Flow
                    }
                } else {
                    ops[force_idx].engine
                }
            } else {
                ops[force_idx].engine
            };
            let dest = ops[force_idx].dest;

            // Check slot availability - we still respect slot limits
            let can_shift = if let Some(fixed) = fixed_shifted_ops {
                fixed.contains(&op_id)
            } else {
                enable_valu_shift
                    && engine == Engine::Valu
                    && can_ssa_valu_shift_to_alu(&ops[force_idx])
            };
            let must_place_shifted = fixed_shifted_ops.map_or(false, |s| s.contains(&op_id));

            // When must_place_shifted we need ALU room for all lanes.
            let slots_ok = if must_place_shifted {
                slots.alu_used + VLEN <= 12
            } else {
                slots.fits(engine, can_shift)
            };

            if slots_ok {
                // CRITICAL: Force schedule, ignoring budget constraint - this exceeds budget!
                // This is intentional behavior - DO NOT REMOVE unless explicitly instructed.
                // The force-schedule allows progress when retry loops fail to free enough
                // VRegs. The budget_exceeded flag tracks this for reporting.
                //
                // Count the over-budget cycle. The detailed breakdown is rendered only for
                // the first such cycle and for a new worst one, and only when verbose: a run
                // that stays over budget for thousands of cycles used to dump every cycle.
                diagnostics.record_over_budget(cycle, live_words, || {
                    format_live_usage_breakdown(ops, &live_vregs, cycle, live_words, scratch_budget)
                });
                budget_exceeded = true;
                // When must_place_shifted use shift-only path; else try with strategy first.
                let allow_shift = strategy.allow_shift(live_words, slots.alu_used, scratch_budget);
                #[allow(unused_assignments)]
                let force_place_result = if must_place_shifted {
                    slots.try_place_shifted_only()
                } else {
                    slots.try_place(engine, can_shift, allow_shift)
                };
                #[allow(unused_assignments)]
                match force_place_result {
                    SlotResult::Placed => {}
                    SlotResult::Shifted => {
                        if fixed_shifted_ops.is_none() {
                            shifted_ops.insert(op_id);
                        }
                    }
                    SlotResult::NoRoom => {
                        if fixed_shifted_ops.is_none() && !valu_shift_fixed_to_none {
                            if let SlotResult::Shifted = slots.try_place(engine, can_shift, true) {
                                shifted_ops.insert(op_id);
                            }
                        }
                    }
                }

                // Record the cycle we actually placed in (same as normal path).
                let force_cycle = cycle as i32;
                ops[force_idx].cycle = force_cycle;
                scheduled.insert(force_idx);
                scheduled_this_cycle += 1;

                // Update capacity demand profile (same as normal path)
                if let (Some(ref mut profile), Some((asap_map, _))) =
                    (demand_profile.as_mut(), bounds_usage.core_bounds())
                {
                    let asap_t = *asap_map.get(&ops[force_idx].id).unwrap_or(&0);
                    profile.schedule_op(force_idx, &ops[force_idx], asap_t, &shiftable_set);
                }

                // Update liveness (same as normal scheduling path)
                if let Some(dest_vreg) = dest {
                    count_written_dest(
                        dest_vreg,
                        cycle,
                        &mut write_groups,
                        &vreg_remaining_users,
                        &mut live_vregs,
                        &mut live_words,
                        &mut vreg_expiry_heap,
                        trace,
                    );
                    vreg_def_cycle.insert(dest_vreg, force_cycle);
                }

                release_read_values(
                    &ops[force_idx],
                    cycle,
                    &liveness_aliases,
                    &write_groups,
                    &mut vreg_remaining_users,
                    &mut live_vregs,
                    &mut live_words,
                    &mut vreg_expiry_heap,
                    trace,
                );

                // Record peak after both adds and frees
                peak_live_words = peak_live_words.max(live_words);

                for &consumer_idx in &consumers[force_idx] {
                    let offset = ops[force_idx]
                        .precedence_offset_for_asap(shiftable_set.contains(&force_idx));
                    max_ready_cycle[consumer_idx] =
                        max_ready_cycle[consumer_idx].max((cycle as i32) + offset);

                    remaining_deps[consumer_idx] -= 1;
                    if remaining_deps[consumer_idx] == 0 {
                        let priority = bounds_usage
                            .core_bounds()
                            .and_then(|(a, _)| a.get(&ops[consumer_idx].id).copied())
                            .map(|lb| max_ready_cycle[consumer_idx].max(lb))
                            .unwrap_or(max_ready_cycle[consumer_idx]);
                        ready_heap.push(Reverse((priority, consumer_idx)));
                    }
                }
            } else {
                // No slot this cycle: back with the other deferred ops for the next one.
                // (Dropping it here left the op unscheduled forever.)
                budget_deferred.push(force_idx);
            }
        }

        // Push remaining budget-deferred ops back to heap for next cycle
        for idx in budget_deferred {
            let priority = bounds_usage
                .core_bounds()
                .and_then(|(a, _)| a.get(&ops[idx].id).copied())
                .map(|lb| ((cycle + 1) as i32).max(lb))
                .unwrap_or_else(|| (cycle + 1) as i32);
            ready_heap.push(Reverse((priority, idx)));
        }

        // Track consecutive empty cycles for ASAP inflation breaker
        if scheduled_this_cycle == 0 {
            consecutive_empty += 1;
        } else {
            consecutive_empty = 0;
        }

        // Counter value once every op of this cycle is placed (compared with interval
        // liveness after the run; see `compare_counter_with_intervals`).
        if trace.cycle_end_live.len() <= cycle {
            trace.cycle_end_live.resize(cycle + 1, 0);
        }
        trace.cycle_end_live[cycle] = live_words;

        cycle += 1;
    }

    // Suppress unused warning for vreg_def_cycle - it's used for debugging
    let _ = vreg_def_cycle;

    if scheduled.len() < ops.len() {
        // Diagnose why scheduling failed
        let unscheduled: Vec<usize> =
            (0..ops.len()).filter(|i| !scheduled.contains(i)).collect();
        let ready_count = ready_heap.len();
        let blocked_count = unscheduled
            .iter()
            .filter(|&&i| remaining_deps[i] > 0)
            .count();

        // Find a sample blocked op and trace its dependency chain
        let sample_blocked = unscheduled
            .iter()
            .find(|&&i| remaining_deps[i] > 0)
            .map(|&i| {
                let op = &ops[i];
                let missing_deps: Vec<_> = op
                    .dep_ids
                    .iter()
                    .filter_map(|dep_id| {
                        id_to_idx
                            .get(dep_id)
                            .filter(|&dep_idx| !scheduled.contains(dep_idx))
                            .map(|&dep_idx| {
                                let dep_remaining = remaining_deps[dep_idx];
                                format!(
                                    "dep_id={} (idx={}, remaining_deps={})",
                                    dep_id, dep_idx, dep_remaining
                                )
                            })
                    })
                    .collect();
                format!(
                    "op[{}] id={} has {} remaining deps, missing: {:?}",
                    i, op.id, remaining_deps[i], missing_deps
                )
            });

        // Also show first few ready (remaining_deps==0) but unscheduled ops
        let ready_unscheduled: Vec<_> = unscheduled
            .iter()
            .filter(|&&i| remaining_deps[i] == 0)
            .take(5)
            .map(|&i| format!("op[{}] id={}", i, ops[i].id))
            .collect();

        let actually_scheduled = scheduled.len() - dead_store_indices.len();
        // A run that went over budget and then failed still reports the overflow.
        diagnostics.emit(peak_live_words, scratch_budget);
        return Err(format!(
            "Scheduling failed: only scheduled {}/{} ops in {} cycles (skipped {} dead stores). \
            Ready heap: {}, Blocked (deps>0): {}, Ready but unscheduled: {:?}, Sample blocked: {:?}, \
            live_words: {}/{}, live_vregs: {}",
            actually_scheduled,
            ops_to_schedule,
            max_cycles,
            dead_store_indices.len(),
            ready_count,
            blocked_count,
            ready_unscheduled,
            sample_blocked,
            live_words,
            scratch_budget,
            live_vregs.len()
        ));
    }

    // Suppress unused warning for vreg_def
    let _ = vreg_def;

    // =========================================================================
    // Collect SelectChoice tracking information
    //
    // Note: Since SelectChoice ops are no longer expanded in greedy scheduler,
    // these collections will be empty. The tracking code remains for compatibility
    // but will not find any expanded SelectChoice ops.
    // =========================================================================
    if let Some(fma_ops) = select_choice_fma_ops {
        // When select_choice_fixed_to_vselect (hint pass enabled), leave set empty so hint pass owns the decision.
        if !select_choice_fixed_to_vselect {
            // Find all SelectChoice ops that were expanded (have sub/fma children)
            // Since we no longer expand SelectChoice ops, this will be empty
            let mut selectchoice_to_fma: HashSet<usize> = HashSet::new();
            for op in ops.iter() {
                if let Some(parent_id) = op.select_choice_parent_id() {
                    if op.is_select_choice_sub() || op.is_select_choice_fma() {
                        selectchoice_to_fma.insert(parent_id);
                    }
                }
            }
            fma_ops.extend(selectchoice_to_fma);
        }
    }

    if let Some(sub_cycles) = select_sub_cycles {
        // Collect cycles for sub ops
        // Since we no longer expand SelectChoice ops, this will be empty
        for op in ops.iter() {
            if op.is_select_choice_sub() && op.cycle >= 0 {
                if let Some(parent_id) = op.select_choice_parent_id() {
                    sub_cycles.insert(parent_id, op.cycle);
                }
            }
        }
    }

    if let Some(fma_cycles) = select_fma_cycles {
        // Collect cycles for fma ops
        // Since we no longer expand SelectChoice ops, this will be empty
        for op in ops.iter() {
            if op.is_select_choice_fma() && op.cycle >= 0 {
                if let Some(parent_id) = op.select_choice_parent_id() {
                    fma_cycles.insert(parent_id, op.cycle);
                }
            }
        }
    }

    // Print inflation stats if any inflation activity occurred (verbose only)
    if diagnostics.verbose
        && (inflation_stats.total_deferred > 0
            || inflation_stats.net_liveness_blocks > 0
            || inflation_stats.breaker_activations > 0)
    {
        eprintln!(
            "ASAP inflation: deferred={}, total_cycles={}, max={}, net_liveness_blocks={}, \
             pass_through_blocks={}, breaker_activations={}",
            inflation_stats.total_deferred,
            inflation_stats.total_inflation_cycles,
            inflation_stats.max_inflation,
            inflation_stats.net_liveness_blocks,
            inflation_stats.pass_through_blocks,
            inflation_stats.breaker_activations,
        );
    }

    diagnostics.emit(peak_live_words, scratch_budget);

    Ok((cycle, budget_exceeded, peak_live_words))
}

/// Schedule SSA ops using a greedy list scheduler with VALU→ALU shifting.
/// Uses default scratch budget of 1536 (hardware limit).
///
/// Returns `(cycles, budget_exceeded, peak_live)`. If budget was exceeded,
/// logs a warning but returns success - caller can decide whether to use CP-SAT.
pub(super) fn schedule_ssa_greedy_with_shift(
    ops: &mut [SsaOp],
    enable_valu_shift: bool,
    shifted_ops: &mut HashSet<usize>,
) -> Result<(usize, bool, usize), String> {
    // Use hardware scratch budget
    schedule_ssa_greedy_with_shift_budget(ops, enable_valu_shift, shifted_ops, 1536)
}

/// Internal implementation of greedy scheduling (simpler, no budget tracking)
#[allow(dead_code)]
pub(super) fn schedule_ssa_greedy_with_shift_impl(
    ops: &mut [SsaOp],
    enable_valu_shift: bool,
    shifted_ops: &mut HashSet<usize>,
) -> Result<usize, String> {
    if ops.is_empty() {
        return Ok(0);
    }

    // Build op_id -> index mapping
    let id_to_idx: HashMap<usize, usize> = ops
        .iter()
        .enumerate()
        .map(|(idx, op)| (op.id, idx))
        .collect();

    // Track scheduled ops
    let mut scheduled: HashSet<usize> = HashSet::new();
    let mut cycle = 0;
    let max_cycles = 50000; // Safety limit

    while scheduled.len() < ops.len() && cycle < max_cycles {
        // Slot usage this cycle
        let mut slots = SlotTracker::new();

        // Find ready ops (all deps scheduled at earlier cycles)
        let ready: Vec<usize> = ops
            .iter()
            .enumerate()
            .filter(|(idx, op)| {
                // Not yet scheduled
                !scheduled.contains(idx)
                    // All dependencies scheduled in earlier cycles (missing deps treated as satisfied)
                    && op.dep_ids.iter().all(|dep_id| {
                        let Some(&dep_idx) = id_to_idx.get(dep_id) else {
                            return true; // external/phantom dep - treat as satisfied
                        };
                        let dep_cycle = ops[dep_idx].cycle;
                        dep_cycle >= 0 && dep_cycle < cycle as i32
                    })
            })
            .map(|(idx, _)| idx)
            .collect();

        // Schedule ready ops that fit in slots
        for idx in ready {
            let op = &ops[idx];
            let engine = op.engine;

            // Check if this VALU op can be shifted to ALU
            let can_shift =
                enable_valu_shift && engine == Engine::Valu && can_ssa_valu_shift_to_alu(op);

            // No strategy in this simple path — always allow shifting.
            let result = slots.try_place(engine, can_shift, true);
            if result == SlotResult::Shifted {
                shifted_ops.insert(op.id);
            }

            if result != SlotResult::NoRoom {
                ops[idx].cycle = cycle as i32;
                scheduled.insert(idx);
            }
        }

        cycle += 1;
    }

    if scheduled.len() < ops.len() {
        return Err(format!(
            "Scheduling failed: only scheduled {}/{} ops in {} cycles",
            scheduled.len(),
            ops.len(),
            max_cycles
        ));
    }

    Ok(cycle)
}

#[cfg(test)]
mod tests;
