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

pub mod justify;
pub mod pressure_policy;
pub mod wavefront;
pub use wavefront::EarlyStagger;

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::kernel_builder::memory_deps::{check_memory_edges, memory_predecessors, memory_successors};

use crate::types::{
    ConstChoiceEncoding, EmissionPhase, Engine, LiveInterval, Opcode, Operand, PrecedenceOffset,
    SelectChoiceEncoding, SsaOp, VReg, VLEN,
};

use crate::address_assign::{compute_live_intervals, compute_peak_liveness, AddressAssignConfig};

use self::pressure_policy::{GreedyPolicy, PressurePolicy};
pub use self::wavefront::GreedyPriority;
use self::wavefront::Wavefront;
use super::ssa_lower::can_ssa_valu_shift_to_alu;

// ============================================================================
// GreedyShape: switches that change the shape of the greedy schedule
// ============================================================================

/// Options that change the shape of the schedule the greedy scheduler produces.
///
/// The default is the historical behaviour; every field is an ablation switch
/// (docs/GREEDY_SHAPE_ABLATION.md).
#[derive(Debug, Clone, Default)]
pub struct GreedyShape {
    /// Order in which ready ops are tried each cycle.
    pub priority: GreedyPriority,
    /// Let a shifted vector op straddle two cycles (see [`SplitShift`]). Without it the
    /// greedy runs at most one shifted op per cycle (8 of the 12 scalar slots).
    pub split_shift: bool,
    /// Scratch model `pools`: the schedule must also keep
    /// `8 * (most vectors live at once) + (most scalars live at once) <= budget`, the
    /// condition under which address assignment always fits. See
    /// [`schedule_ssa_greedy_shaped`] for how the greedy meets it.
    pub scratch_pools: bool,
    /// Pool capacities `(vectors, scalars)` the run must keep (set by the pools retry in
    /// [`schedule_ssa_greedy_shaped`]); `None` = live words against the budget.
    pub pool_caps: Option<(usize, usize)>,
    /// When a shiftable vector op is run as 8 scalar ops (see [`ShiftPolicy`]).
    pub shift_policy: ShiftPolicy,
    /// Wavefront only: which ready loads get the load slots (see [`LoadPriority`]).
    pub load_priority: LoadPriority,
    /// Wavefront only: which ready flow ops get the flow slot (see [`FlowPriority`]).
    pub flow_priority: FlowPriority,
    /// Where a SelectChoice that has a one-op `multiply_add` form may run (see
    /// [`SelectOverflow`]).
    pub select_overflow: SelectOverflow,
    /// Wavefront only: native scalar ops before vector ops (see [`AluPriority`]).
    pub alu_priority: AluPriority,
    /// Where a ConstChoice is materialised (see [`ConstPolicy`]).
    pub const_policy: ConstPolicy,
    /// Wavefront only: a different stagger for the first groups (see [`EarlyStagger`]).
    pub early_stagger: Option<EarlyStagger>,
    /// When an op the scratch budget check deferred is forced past the budget (see
    /// [`BudgetForce`]). Scratch model `words` only; `pools` always stalls.
    pub budget_force: BudgetForce,
}

/// Scratch model `words`: when the greedy forces an op that the budget check deferred.
///
/// The budget check defers an op whose destination would take the live words over the
/// budget. With `SameCycle` (default, the historical behaviour) the deferred op with the
/// highest last-use count is then forced in the same cycle whenever it has a slot, even if
/// other ops were placed in that cycle, so the check holds one op per cycle at most and
/// any run that reaches the budget overshoots it (address assignment then rejects the
/// kernel). The pressure policy's inflation and shift suppression are what keep a default
/// run below the budget, which is why passing runs stop short of it.
/// `Stall` forces only in a cycle where nothing else was placed (the greedy is stuck), as the
/// `pools` model does: otherwise the deferred op waits for the next cycle, when ops placed
/// in this one may have freed words.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BudgetForce {
    /// Force one deferred op per cycle whenever it has a slot (default).
    #[default]
    SameCycle,
    /// Force only when nothing else was placed in the cycle. Not necessarily a deadlock: a
    /// cycle can be empty because ASAP inflation deferred every op that would free words for a
    /// few cycles (docs/STALL_DEADLOCK.md).
    Stall,
}

/// Wavefront only: the order in which ready flow ops are offered the flow slot. Other
/// engines keep their order.
///
/// The selects of the first rounds (before the mid-kernel gathers) and those of the last
/// rounds (after them) share the one flow slot in the middle of the schedule: early groups
/// are in their last select rounds while late groups are still in their first ones. Under
/// the wavefront key the early groups' late selects go first, and the late groups reach the
/// load stream late (docs/GREEDY_SHAPE_ABLATION.md section 12).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FlowPriority {
    /// Wavefront key (default).
    #[default]
    Key,
    /// Flow ops that feed a non-final load (a load that some later load depends on, e.g. a
    /// mid-kernel gather) before the other flow ops, then key.
    FeedsLoadFirst,
    /// Flow ops by chain (group) rank first, then key; shared ops first. The first groups'
    /// selects of the first rounds are not overtaken by later groups' earlier selects, so they
    /// reach their first gather sooner (docs/STARTUP_AND_STALL_LOOP.md).
    ChainFirst,
}

/// Where the greedy materialises a ConstChoice (a constant with alternative encodings).
///
/// `Default` (the historical rule): `add_imm` on the flow slot when it is free and no other
/// flow op is ready in the cycle, else a load. The other policies may also derive the constant
/// on the scalar ALU from constants already materialised: the `ConstEncoding::{Add, Sub, Mul,
/// Shift}` alternatives the emitter offers, else `a + b` for any two constants a, b > 0 with
/// a + b = value < 2^31 (e.g. the batch address offsets 8g); only from producers scheduled in an earlier cycle
/// whose values are still live in the scheduler's counter, so the read needs no lifetime
/// extension (address assignment's AluDerive pass then changes nothing). Greedy-only: the
/// CP-SAT pipeline keeps `Default`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConstPolicy {
    /// add_imm when flow is free and idle, else load (default).
    #[default]
    Default,
    /// ALU derivation when an alternative's operands are live and a scalar slot is free; else
    /// the default rule.
    AluFirst,
    /// A load while a load slot is free; else ALU derivation; else the default rule.
    AluWhenLoadFull,
}

/// Wavefront only: the order in which ready scalar-ALU work is offered the 12 scalar slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AluPriority {
    /// Wavefront key (default): native scalar ops and shiftable vector ops in one order.
    #[default]
    Key,
    /// Native scalar-ALU ops (e.g. address adds) before every vector op of the cycle, so a
    /// vector op shifted to the scalar ALU never takes a slot a ready native op needs.
    NativeFirst,
}

/// Where a SelectChoice with a precomputed delta (one `multiply_add` on the vector ALU, or
/// one `vselect` on flow; both visible one cycle later) runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SelectOverflow {
    /// Always flow (default): the greedy never takes the multiply_add form.
    #[default]
    Never,
    /// Flow when the flow slot is free in this cycle; otherwise the vector ALU when it has a
    /// free slot, for at most `limit` selects over the run (`None` = no limit).
    Valu { limit: Option<usize> },
}

impl SelectOverflow {
    /// Whether one more select may take the multiply_add form after `taken` have.
    fn allows(&self, taken: usize) -> bool {
        match *self {
            SelectOverflow::Never => false,
            SelectOverflow::Valu { limit } => limit.map_or(true, |l| taken < l),
        }
    }
}

/// Wavefront only: the order in which ready loads are offered the two load slots. Other
/// engines always keep the wavefront order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LoadPriority {
    /// Wavefront key (default).
    #[default]
    Key,
    /// Longest tail first (longest dependency path to the end of the program), then key.
    Tail,
    /// Final loads (no later load depends on them, e.g. the last gather round) after all
    /// other loads, then key: a final load has a short tail, while a load that feeds later
    /// gathers holds up the load engine's future work.
    ///
    /// Exactly what it does: once per cycle, before the first placement pass, the ops whose
    /// own engine is `Load` are re-sorted among the positions they hold in the wavefront order
    /// by `(final, key)`. Limitations, all measured with this behaviour (the
    /// docs/GREEDY_SHAPE_ABLATION.md section 11 numbers):
    /// - Constants (`Const`, `ConstChoice`) are `Load`-engine ops and are sorted too. A
    ///   constant no load depends on counts as final, even when it is then lowered to a flow
    ///   `add_imm`.
    /// - Only the first pass is ordered. A feeder load deferred for the scratch budget in that
    ///   pass is retried after the final loads of the same pass, so a final load can take a
    ///   load slot ahead of a budget-deferred feeder
    ///   (`test_load_priority_final_last_budget_deferral`).
    FinalLast,
}

// ============================================================================
// ShiftPolicy: when the greedy runs a shiftable vector op on the scalar ALU
// ============================================================================

/// When the greedy runs a shiftable vector op as 8 scalar ops.
///
/// The historical rule ([`ShiftPolicy::WhenFull`]) shifts only an op that finds the six
/// vector slots full. Ready ops are tried in priority order, so shiftable ops early in the
/// order take vector slots and `multiply_add`s (which cannot shift) later in the order wait,
/// while the scalar ALU idles (docs/GREEDY_SHAPE_ABLATION.md section 11).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ShiftPolicy {
    /// Shift only when the vector slots are full (default).
    #[default]
    WhenFull,
    /// Also shift when the vector slots left in this cycle are wanted by non-shiftable
    /// vector ops later in this cycle's ready order.
    Reserve,
    /// Also shift whenever the run is behind a pace: `shifted / vector ops placed` below
    /// `target / vector ops`. `target = None` uses the balanced split of the emitted counts,
    /// `(2 V - A) / 10` (vector ALU and scalar ALU floors equal).
    Pace { target: Option<usize> },
    /// Either `Reserve` or `Pace`.
    ReservePace { target: Option<usize> },
}

impl ShiftPolicy {
    fn reserve(&self) -> bool {
        matches!(self, ShiftPolicy::Reserve | ShiftPolicy::ReservePace { .. })
    }

    /// `Some(explicit target or None for the balanced split)` when the policy paces.
    fn pace(&self) -> Option<Option<usize>> {
        match *self {
            ShiftPolicy::Pace { target } | ShiftPolicy::ReservePace { target } => Some(target),
            _ => None,
        }
    }
}

/// Shifted vector ops that balance the vector and scalar ALU floors:
/// `(V - s) / 6 = (A + 8 s) / 12`, i.e. `s = (2 V - A) / 10`, rounded up and clamped to
/// `[0, shiftable]`.
///
/// # Arguments
/// * `vector_ops` - Vector ops the greedy places on the vector ALU unless shifted
/// * `shiftable` - How many of them may run as 8 scalar ops
/// * `scalar_ops` - Native scalar ALU ops
///
/// # Returns
/// The balanced number of shifts.
pub(crate) fn balanced_shift_target(vector_ops: usize, shiftable: usize, scalar_ops: usize) -> usize {
    let num = 2 * vector_ops as i64 - scalar_ops as i64;
    if num <= 0 {
        return 0;
    }
    (((num + 9) / 10) as usize).min(shiftable)
}

// ============================================================================
// SplitShift: a shifted vector op spread over two cycles
// ============================================================================

/// A shifted vector op whose scalar lanes are spread over two consecutive cycles.
///
/// A shifted op is VLEN = 8 scalar ops and the scalar ALU has 12 slots, so whole shifted
/// ops use at most 8 slots per cycle. With split shifts the leftover slots take the first
/// lanes of one more op and the next cycle takes the rest: three shifted ops per two cycles.
///
/// The op's `cycle` is the cycle of its last lane (its completion, as for CP-SAT's shifted
/// ops). Its destination block is reserved from the first lane and its operands stay live
/// until the last lane, exactly as `address_assign` models `shifted_alu_cycles`.
#[derive(Debug, Clone, Copy)]
struct SplitShift {
    /// Index of the op.
    idx: usize,
    /// Cycle of the first lanes.
    first_cycle: i32,
    /// Lanes placed in `first_cycle` (lanes `0..first_lanes`); the rest run the cycle after.
    first_lanes: usize,
}

impl SplitShift {
    /// Fewest scalar slots worth starting a split for. With fewer, the op pays a cycle of
    /// latency and holds its destination block early for little gain.
    const MIN_FIRST_LANES: usize = 2;

    /// Per-lane cycles in `shifted_alu_cycles` form, completing at `last_cycle`.
    fn lane_cycles(&self, last_cycle: i32) -> Vec<i32> {
        (0..VLEN)
            .map(|lane| if lane < self.first_lanes { self.first_cycle } else { last_cycle })
            .collect()
    }
}

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
    /// Shifts (whole or split) made by a proactive shift policy while a vector slot was free.
    pub proactive_shifts: usize,
    /// Breakdown for the worst cycle so far, printed by `finish` (only kept when `verbose`).
    worst_dump: Option<String>,
    /// Last cycle recorded, so several force-schedules in one cycle count once.
    last_cycle: Option<usize>,
    /// Out: ConstChoices the scheduler derived on the scalar ALU (op id -> operands), under a
    /// [`ConstPolicy`] other than `Default`. Lowering and address assignment need them.
    pub const_alu_derive: HashMap<usize, crate::types::AluDeriveInfo>,
    /// Times the hard budget check (live words after the op > budget) deferred an op; an op
    /// retried in the same cycle counts once per rejection.
    pub budget_check_deferrals: usize,
    /// Ops forced past the budget by the force-schedule path.
    pub forced_past_budget: usize,
    /// Cycles whose only placement was the first part of a split shift.
    pub split_only_cycles: usize,
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
/// * `delta_read` - The op is a SelectChoice that may take its one-op multiply_add form, so
///   its precomputed delta counts as read (see [`liveness_read_vregs`])
///
/// # Returns
/// The VRegs the op reads or keeps live, then their aliases, each exactly once.
fn liveness_release_vregs(
    op: &SsaOp,
    liveness_aliases: &HashMap<VReg, Vec<VReg>>,
    delta_read: bool,
) -> Vec<VReg> {
    let mut released: Vec<VReg> = Vec::with_capacity(op.srcs.len());
    for vreg in liveness_read_vregs(op, delta_read) {
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

/// Live words and live vectors (8-word values, write groups included) held by the counter.
/// Live scalars are `words - VLEN * vectors`.
#[derive(Debug, Default, Clone, Copy)]
struct LiveCount {
    words: usize,
    vectors: usize,
}

impl LiveCount {
    fn add(&mut self, vreg: VReg) {
        self.words += vreg.size();
        if vreg.is_vector() {
            self.vectors += 1;
        }
    }

    fn remove(&mut self, vreg: &VReg) {
        self.words = self.words.saturating_sub(vreg.size());
        if vreg.is_vector() {
            self.vectors = self.vectors.saturating_sub(1);
        }
    }

    fn scalars(&self) -> usize {
        self.words - VLEN * self.vectors
    }

    /// The count after adding and removing the given values (sizes 1 or VLEN).
    fn after(&self, added: &[VReg], removed: &[VReg]) -> LiveCount {
        let mut c = *self;
        for v in added {
            c.add(*v);
        }
        for v in removed {
            c.remove(v);
        }
        c
    }
}

/// Scratch budget check of the greedy: live words against the budget, or fixed pool
/// capacities `(N_v, N_s)` with `8 * N_v + N_s <= budget`: at most `N_v` vectors and `N_s`
/// scalars live at once. A schedule within such capacities gets addresses within the budget
/// (`address_assign::assign_two_pass` needs at most `8 * max_v + max_s` words).
#[derive(Debug, Default)]
struct PoolBudget {
    caps: Option<(usize, usize)>,
}

impl PoolBudget {
    fn new(caps: Option<(usize, usize)>) -> Self {
        PoolBudget { caps }
    }

    /// The live state expressed as words of the budget, for the pressure policies: live words,
    /// or with capacities the fuller pool's fill fraction times the budget.
    fn words_needed(&self, live: &LiveCount, budget: usize) -> usize {
        match self.caps {
            None => live.words,
            Some((n_v, n_s)) => {
                let frac = |used: usize, cap: usize| {
                    if cap == 0 {
                        if used == 0 { 0 } else { usize::MAX / 2 }
                    } else {
                        (used * budget).div_ceil(cap)
                    }
                };
                frac(live.vectors, n_v).max(frac(live.scalars(), n_s))
            }
        }
    }

    /// Whether reaching `live` keeps the schedule within the budget (or the capacities).
    fn fits(&self, live: &LiveCount, budget: usize) -> bool {
        match self.caps {
            None => live.words <= budget,
            Some((n_v, n_s)) => live.vectors <= n_v && live.scalars() <= n_s,
        }
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
    live: &mut LiveCount,
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
        live.add(counted);
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
    live: &mut LiveCount,
    vreg_expiry_heap: &mut BinaryHeap<Reverse<(i32, VReg)>>,
    trace: &mut CounterTrace,
) {
    let delta_read = trace.delta_readers.contains(&op.id);
    let fma = trace.fma_selects.contains(&op.id);
    for vreg in &liveness_release_vregs(op, liveness_aliases, delta_read) {
        let Some(remaining) = vreg_remaining_users.get_mut(vreg) else {
            continue;
        };
        *remaining = remaining.saturating_sub(1);
        // A vselect does not read the delta: no read, only one holder fewer.
        if !(delta_read && !fma && op.delta_vreg == Some(*vreg)) {
            let end = cycle as i32 + i32::from(fma);
            let e = trace.read_end.entry(*vreg).or_insert(end);
            *e = (*e).max(end);
        }
        if *remaining != 0 || !live_vregs.contains(vreg) {
            continue;
        }
        if write_groups.releasable_at(vreg, cycle) {
            match trace.read_end.get(vreg).copied() {
                // A multiply_add select read it this cycle: held through this cycle.
                Some(end) if end > cycle as i32 => vreg_expiry_heap.push(Reverse((end, *vreg))),
                // Held only for selects that took the vselect form: its interval ended at
                // its last real read. Give the words back from then on (the counter was
                // conservative in between).
                Some(end) if end < cycle as i32 => {
                    live_vregs.remove(vreg);
                    live.remove(vreg);
                    let start = trace.spans.get(vreg).map_or(end, |s| s.0);
                    let end = end.max(start);
                    for c in end.max(0) as usize..cycle.min(trace.cycle_end_live.len()) {
                        trace.cycle_end_live[c] -= vreg.size();
                    }
                    trace.release(vreg, end as usize);
                }
                _ => {
                    live_vregs.remove(vreg);
                    live.remove(vreg);
                    trace.release(vreg, cycle);
                }
            }
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
/// (the same reads `address_assign`'s `LivenessUse` reports), and with `delta_read` the
/// precomputed delta of a SelectChoice. Address assignment holds the delta (and every
/// operand) of a select in `select_fma_ops` to the select's cycle + 1; the greedy decides
/// the select's form only when it places it, so a select that may take the multiply_add form
/// holds its delta until then (released early again if it takes the vselect form, see
/// [`release_read_values`]). May contain duplicates.
fn liveness_read_vregs(op: &SsaOp, delta_read: bool) -> impl Iterator<Item = VReg> + '_ {
    op.srcs
        .iter()
        .filter_map(|src| src.as_vreg())
        .chain(op.keep_live.iter().flatten().copied())
        .chain(op.delta_vreg.filter(|_| delta_read))
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
    /// Op ids of the SelectChoice ops whose precomputed delta the counter treats as read
    /// (those that may take the one-op multiply_add form in this run).
    delta_readers: HashSet<usize>,
    /// Op ids of the SelectChoice ops placed in the one-op multiply_add form. Address
    /// assignment (`select_fma_ops`) holds their operands and delta to cycle + 1.
    fma_selects: HashSet<usize>,
    /// Per value: the cycle its interval ends at given the reads so far (a read at cycle c
    /// ends it at c, a multiply_add select's read at c + 1).
    read_end: HashMap<VReg, i32>,
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
/// * `shifted_lane_cycles` - Per-lane cycles of shifted ops whose lanes are not all in the
///   op's cycle (split shifts); empty otherwise
/// * `select_fma_ops` - SelectChoice ops lowered as multiply_adds (address assignment's
///   `select_fma_ops`: their reads are held to cycle + 1)
///
/// # Returns
/// `(curve, peak, intervals)`: `curve[t]` is the live words at cycle `t`; `peak` is
/// `address_assign::compute_peak_liveness` of `intervals`.
pub(super) fn interval_liveness_curve(
    ops: &[SsaOp],
    vector_group_lanes: Option<&HashMap<VReg, Vec<VReg>>>,
    shifted_lane_cycles: &HashMap<usize, Vec<i32>>,
    select_fma_ops: &HashSet<usize>,
) -> Result<(Vec<usize>, usize, HashMap<VReg, LiveInterval>), String> {
    let (curve, peak, intervals, _, _) =
        interval_liveness_inputs(ops, vector_group_lanes, shifted_lane_cycles, select_fma_ops)?;
    Ok((curve, peak, intervals))
}

/// [`interval_liveness_curve`], also returning the address-assignment config and the
/// scheduled ops it used (for pool peaks with the same inputs).
#[allow(clippy::type_complexity)]
fn interval_liveness_inputs(
    ops: &[SsaOp],
    vector_group_lanes: Option<&HashMap<VReg, Vec<VReg>>>,
    shifted_lane_cycles: &HashMap<usize, Vec<i32>>,
    select_fma_ops: &HashSet<usize>,
) -> Result<(Vec<usize>, usize, HashMap<VReg, LiveInterval>, AddressAssignConfig, Vec<SsaOp>), String> {
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
    let config = AddressAssignConfig {
        vector_group_lanes: groups,
        shifted_alu_cycles: shifted_lane_cycles.clone(),
        select_fma_ops: select_fma_ops.clone(),
        ..Default::default()
    };
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
    Ok((curve, compute_peak_liveness(&intervals), intervals, config, scheduled))
}

/// Compare the scheduler's counter with interval liveness of the finished schedule.
///
/// # Arguments
/// * `ops` - Ops as scheduled by the greedy
/// * `vector_group_lanes` - Vector groups, as passed to the scheduler
/// * `shifted_lane_cycles` - Per-lane cycles of split shifts (empty if there are none)
/// * `trace` - What the counter did during the run
/// * `counter_peak` - The peak the scheduler reports
///
/// # Returns
/// The comparison, or `Err` if interval liveness cannot be computed for the schedule.
fn compare_counter_with_intervals(
    ops: &[SsaOp],
    vector_group_lanes: Option<&HashMap<VReg, Vec<VReg>>>,
    shifted_lane_cycles: &HashMap<usize, Vec<i32>>,
    trace: &CounterTrace,
    counter_peak: usize,
) -> Result<LivenessAgreement, String> {
    let cycle_end_live = &trace.cycle_end_live;
    let (curve, interval_peak, intervals) =
        interval_liveness_curve(ops, vector_group_lanes, shifted_lane_cycles, &trace.fma_selects)?;
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
    schedule_ssa_greedy_shaped(
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
        &GreedyShape::default(),
        &mut HashMap::new(),
    )
    .map(|(result, _agreement)| result)
}

/// Same as [`schedule_ssa_greedy_with_diagnostics`] with explicit schedule-shape options,
/// and also returns the liveness agreement (see
/// [`schedule_ssa_greedy_with_liveness_agreement`]).
///
/// # Arguments
/// * `shape` - Schedule-shape switches; `GreedyShape::default()` is the historical behaviour
/// * `shifted_lane_cycles` - Out: per-lane cycles (`shifted_alu_cycles` form) of every
///   shifted op whose lanes are not all in the op's cycle. Only split shifts
///   (`shape.split_shift`) produce entries; lowering and address assignment need them.
///
/// # Returns
/// `((cycles, budget_exceeded, peak_live), agreement)`, or `Err` as for
/// [`schedule_ssa_greedy_with_liveness_agreement`].
/// After a greedy rerun with fixed ConstChoice encodings: every fixed `AluDerive` the rerun
/// did not place as a derivation (`derived`, the rerun's `GreedyDiagnostics::const_alu_derive`)
/// was scheduled as a load, so record it as `Load`; lowering then emits what was scheduled.
///
/// # Returns
/// How many derivations became loads.
pub(crate) fn reconcile_const_choice_encodings(
    encodings: &mut HashMap<usize, ConstChoiceEncoding>,
    derived: &HashMap<usize, crate::types::AluDeriveInfo>,
) -> usize {
    let mut demoted = 0;
    for (id, enc) in encodings.iter_mut() {
        if matches!(enc, ConstChoiceEncoding::AluDerive(_)) && !derived.contains_key(id) {
            *enc = ConstChoiceEncoding::Load;
            demoted += 1;
        }
    }
    demoted
}

pub(crate) fn schedule_ssa_greedy_shaped(
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
    fixed_const_choice_encoding: Option<&HashMap<usize, ConstChoiceEncoding>>,
    fixed_shifted_ops: Option<&HashSet<usize>>,
    fixed_select_choice_encoding: Option<&HashMap<usize, SelectChoiceEncoding>>,
    diagnostics: &mut GreedyDiagnostics,
    shape: &GreedyShape,
    shifted_lane_cycles: &mut HashMap<usize, Vec<i32>>,
) -> Result<((usize, bool, usize), LivenessAgreement), String> {
    if !shape.scratch_pools || shape.pool_caps.is_some() {
        return schedule_ssa_greedy_shaped_once(
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
            shape,
            shifted_lane_cycles,
        );
    }
    // Scratch model `pools`. First the usual run (live words against the budget). If its
    // schedule already keeps 8 * max_v + max_s within the budget (measured with address
    // assignment's own liveness, `scratch_pool_peaks`), it is the result: wherever the pool
    // condition does not bind, the schedule is the same as with `words`. Otherwise run again
    // with fixed capacities: the first run's scalar maximum N_s and N_v = (budget - N_s) / 8
    // vectors. Fixed capacities rather than running maxima, because with running maxima the
    // greedy fills one pool early (all vectors are free up to the current maximum) and then
    // deadlocks when the other pool has to grow.
    let mut select_choice_fma_ops = select_choice_fma_ops;
    let mut select_sub_cycles = select_sub_cycles;
    let mut select_fma_cycles = select_fma_cycles;
    let mut const_add_imm_ops = const_add_imm_ops;
    let saved_cycles: Vec<i32> = ops.iter().map(|op| op.cycle).collect();
    let saved_shifted = shifted_ops.clone();
    let saved_fma = select_choice_fma_ops.as_deref().cloned();
    let saved_sub = select_sub_cycles.as_deref().cloned();
    let saved_fma_cycles = select_fma_cycles.as_deref().cloned();
    let saved_add_imm = const_add_imm_ops.as_deref().cloned();
    let words_shape = GreedyShape { scratch_pools: false, pool_caps: None, ..shape.clone() };
    let first = schedule_ssa_greedy_shaped_once(
        ops,
        enable_valu_shift,
        shifted_ops,
        scratch_budget,
        select_choice_fma_ops.as_deref_mut(),
        select_sub_cycles.as_deref_mut(),
        select_fma_cycles.as_deref_mut(),
        const_add_imm_ops.as_deref_mut(),
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
        &words_shape,
        shifted_lane_cycles,
    )?;
    let fma_now: HashSet<usize> = select_choice_fma_ops.as_deref().cloned().unwrap_or_default();
    let peaks = scratch_pool_peaks(ops, vector_group_lanes, shifted_lane_cycles, &fma_now)?;
    if first.0 .1 || peaks.split_words() <= scratch_budget {
        // Within the pools already, or over budget even on words (reported as such).
        return Ok(first);
    }
    let n_s = peaks.scalars.min(scratch_budget);
    let n_v = (scratch_budget - n_s) / VLEN;
    if diagnostics.verbose {
        eprintln!(
            "Greedy (--scratch-model pools): words schedule needs 8*{} + {} = {} > {} words; \
             rescheduling with at most {} vectors and {} scalars live",
            peaks.vectors,
            peaks.scalars,
            peaks.split_words(),
            scratch_budget,
            n_v,
            n_s
        );
    }
    for (op, &c) in ops.iter_mut().zip(&saved_cycles) {
        op.cycle = c;
    }
    *shifted_ops = saved_shifted;
    if let (Some(m), Some(v)) = (select_choice_fma_ops.as_deref_mut(), saved_fma) {
        *m = v;
    }
    if let (Some(m), Some(v)) = (select_sub_cycles.as_deref_mut(), saved_sub) {
        *m = v;
    }
    if let (Some(m), Some(v)) = (select_fma_cycles.as_deref_mut(), saved_fma_cycles) {
        *m = v;
    }
    if let (Some(m), Some(v)) = (const_add_imm_ops.as_deref_mut(), saved_add_imm) {
        *m = v;
    }
    *diagnostics = GreedyDiagnostics::new(diagnostics.verbose);
    let pools_shape = GreedyShape { pool_caps: Some((n_v, n_s)), ..shape.clone() };
    schedule_ssa_greedy_shaped_once(
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
        &pools_shape,
        shifted_lane_cycles,
    )
}

/// Peak vectors and scalars of a greedy schedule under address assignment's own liveness
/// (`address_assign::compute_value_spans` / `compute_pool_peaks`), with the same interval
/// inputs as the T16 agreement check (`interval_liveness_curve`).
///
/// # Arguments
/// * `ops` - Ops with assigned cycles; negative cycles (skipped dead stores) are ignored
/// * `vector_group_lanes` - Vector groups, as passed to the scheduler
/// * `shifted_lane_cycles` - Per-lane cycles of split shifts
/// * `select_fma_ops` - SelectChoice ops lowered as multiply_adds
///
/// # Returns
/// The pool peaks, or the interval computation's error.
pub(super) fn scratch_pool_peaks(
    ops: &[SsaOp],
    vector_group_lanes: Option<&HashMap<VReg, Vec<VReg>>>,
    shifted_lane_cycles: &HashMap<usize, Vec<i32>>,
    select_fma_ops: &HashSet<usize>,
) -> Result<crate::address_assign::PoolPeaks, String> {
    let (_, _, intervals, config, scheduled) =
        interval_liveness_inputs(ops, vector_group_lanes, shifted_lane_cycles, select_fma_ops)?;
    let spans = crate::address_assign::compute_value_spans(
        &intervals,
        &crate::address_assign::last_write_cycles(&scheduled, &config),
    );
    Ok(crate::address_assign::compute_pool_peaks(&spans, &config.fixed_vregs))
}

/// One greedy run (see [`schedule_ssa_greedy_shaped`]).
fn schedule_ssa_greedy_shaped_once(
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
    fixed_const_choice_encoding: Option<&HashMap<usize, ConstChoiceEncoding>>,
    fixed_shifted_ops: Option<&HashSet<usize>>,
    fixed_select_choice_encoding: Option<&HashMap<usize, SelectChoiceEncoding>>,
    diagnostics: &mut GreedyDiagnostics,
    shape: &GreedyShape,
    shifted_lane_cycles: &mut HashMap<usize, Vec<i32>>,
) -> Result<((usize, bool, usize), LivenessAgreement), String> {
    shifted_lane_cycles.clear();
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
        shape,
        shifted_lane_cycles,
    )?;
    // Memory edges are enforced by readiness; check the result (free when there are none).
    check_memory_edges(ops, |idx| ops[idx].cycle).map_err(|e| format!("greedy schedule: {}", e))?;
    check_liveness_agreement(ops, vector_group_lanes, shifted_lane_cycles, &trace, result)
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
    schedule_ssa_greedy_shaped(
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
        &GreedyShape::default(),
        &mut HashMap::new(),
    )
}

/// Compare the counter of a finished greedy run with interval liveness of its schedule.
///
/// # Returns
/// `(result, agreement)`, or `Err` on disagreement (a bug in the counter, never a property
/// of the input; set `SCHEDULER_LIVENESS_AGREEMENT=warn` to continue with a warning).
fn check_liveness_agreement(
    ops: &[SsaOp],
    vector_group_lanes: Option<&HashMap<VReg, Vec<VReg>>>,
    shifted_lane_cycles: &HashMap<usize, Vec<i32>>,
    trace: &CounterTrace,
    result: (usize, bool, usize),
) -> Result<((usize, bool, usize), LivenessAgreement), String> {
    let agreement =
        compare_counter_with_intervals(ops, vector_group_lanes, shifted_lane_cycles, trace, result.2)?;
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

/// Record that op `idx` was placed at `cycle` for its memory-edge successors: raise each
/// one's ready cycle by the edge latency, and push it onto the ready heap once nothing else
/// holds it back. The memory-edge counterpart of the `consumers` loops in
/// `schedule_ssa_greedy_core`.
fn release_memory_successors(
    successors: &[(usize, i32)],
    cycle: i32,
    ops: &[SsaOp],
    bounds_usage: &GreedyBoundsUsage,
    max_ready_cycle: &mut [i32],
    remaining_deps: &mut [usize],
    ready_heap: &mut BinaryHeap<Reverse<(i32, usize)>>,
) {
    for &(target, latency) in successors {
        max_ready_cycle[target] = max_ready_cycle[target].max(cycle + latency);
        remaining_deps[target] -= 1;
        if remaining_deps[target] == 0 {
            let priority = bounds_usage
                .core_bounds()
                .and_then(|(a, _)| a.get(&ops[target].id).copied())
                .map(|lb| max_ready_cycle[target].max(lb))
                .unwrap_or(max_ready_cycle[target]);
            ready_heap.push(Reverse((priority, target)));
        }
    }
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
    // Schedule-shape switches (ready-queue priority, split shifts).
    shape: &GreedyShape,
    // Out: per-lane cycles of split shifts.
    shifted_lane_cycles: &mut HashMap<usize, Vec<i32>>,
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

    // ConstPolicy: per ConstChoice, its ALU alternatives as (left value, right value, opcode),
    // and per constant value the ops that produce it.
    let mut const_derivations: HashMap<usize, Vec<(i64, i64, Opcode)>> = HashMap::new();
    let mut value_producers: HashMap<i64, Vec<usize>> = HashMap::new();
    if shape.const_policy != ConstPolicy::Default {
        for (idx, op) in ops.iter().enumerate() {
            if let (Some(v), Some(d)) = (op.produces_constant, op.dest) {
                if d.is_scalar() {
                    value_producers.entry(v).or_default().push(idx);
                }
            }
            if op.opcode != Opcode::ConstChoice {
                continue;
            }
            let alts: Vec<(i64, i64, Opcode)> = op
                .const_alternatives
                .iter()
                .flatten()
                .filter_map(|enc| {
                    let deps = enc.dependencies();
                    Some((*deps.first()?, *deps.get(1)?, enc.alu_opcode()?))
                })
                .collect();
            if !alts.is_empty() {
                const_derivations.insert(idx, alts);
            }
        }
    }

    // A SelectChoice fixed to the multiply-add path without a precomputed delta (it has a
    // sub_temp: subtract at its cycle, multiply_add one cycle later) makes its result visible
    // one cycle later than the vselect path. The CP-SAT model requires that extra cycle
    // (select_fma_extra_delay); without it the greedy hint violates the model's precedences.
    let fma_extra_latency: HashSet<usize> = fixed_select_choice_encoding
        .map(|fixed| {
            ops.iter()
                .enumerate()
                .filter(|(_, op)| {
                    op.opcode == Opcode::SelectChoice
                        && op.sub_temp_vreg.is_some()
                        && matches!(fixed.get(&op.id), Some(SelectChoiceEncoding::Fma))
                })
                .map(|(idx, _)| idx)
                .collect()
        })
        .unwrap_or_default();
    // Cycles after a producer at which its result can be read.
    let result_latency = |ops: &[SsaOp], idx: usize| -> i32 {
        ops[idx].precedence_offset_for_asap(shiftable_set.contains(&idx))
            + i32::from(fma_extra_latency.contains(&idx))
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
    // Selects that may take the one-op multiply_add form hold their delta (address
    // assignment reads it at cycle + 1 when they do): with select overflow, any select with a
    // delta the hint pass does not own; with fixed encodings, those fixed to the FMA path.
    trace.delta_readers = ops
        .iter()
        .filter(|op| op.opcode == Opcode::SelectChoice && op.delta_vreg.is_some())
        .filter(|op| match fixed_select_choice_encoding {
            Some(fixed) => matches!(fixed.get(&op.id), Some(SelectChoiceEncoding::Fma)),
            None => !select_choice_fixed_to_vselect && shape.select_overflow != SelectOverflow::Never,
        })
        .map(|op| op.id)
        .collect();

    for (idx, op) in ops.iter().enumerate() {
        if let Some(dest) = op.dest {
            // Don't overwrite Load's entry with AddImm alternative
            if !const_add_imm_alternatives.contains(&idx) {
                vreg_def.insert(dest, idx);
            }
        }
        // Still track users from AddImm alternatives (they read zero_vreg).
        // `keep_live` values count as read: address assignment holds them until this op.
        for vreg in liveness_read_vregs(op, trace.delta_readers.contains(&op.id)) {
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
    // Memory ordering edges (`mem_deps`): (target index, latency) per source, and
    // (source index, latency) per target. Kept apart from `consumers`, which is VReg data
    // flow. They gate readiness and link wavefront chains that cannot be admitted independently.
    // A latency-0 (WAR) target is never placed in its source's cycle here: ready ops are
    // collected before any op of the cycle is placed, so it lands one cycle later or more.
    let mem_successors = memory_successors(ops, &id_to_idx);
    let mem_predecessors = memory_predecessors(ops, &id_to_idx);

    // Wavefront priority: per-op keys and the window of chains allowed to run.
    let mut wavefront: Option<Wavefront> = match shape.priority {
        GreedyPriority::Fifo => None,
        GreedyPriority::Wavefront { stagger_cycles, groups_in_flight } => Some(Wavefront::new(
            ops,
            &consumers,
            &id_to_idx,
            &shiftable_set,
            &dead_store_indices,
            stagger_cycles,
            groups_in_flight,
            shape.early_stagger,
        )?),
    };
    if diagnostics.verbose {
        if let Some(w) = wavefront.as_ref() {
            eprintln!("Greedy wavefront priority: {} chains", w.chain_count());
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
                + mem_predecessors[_op_idx]
                    .iter()
                    .filter(|(src, _)| !dead_store_indices.contains(src))
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
    let mut live = LiveCount::default();
    let pools = PoolBudget::new(shape.pool_caps);
    let mut live_vregs: HashSet<VReg> = HashSet::new();

    // Budget tracking
    let mut peak_live_words: usize = 0;
    let mut budget_exceeded = false;

    // Track when each VReg was defined (used for debugging/validation)
    let mut vreg_def_cycle: HashMap<VReg, i32> = HashMap::new();

    // Ready heap: (priority, op_idx) - min-heap via Reverse
    // Priority is the earliest cycle this op can be scheduled (ASAP)
    let mut ready_heap: BinaryHeap<Reverse<(i32, usize)>> = BinaryHeap::new();
    // SCHEDULER_DEADLOCK_DUMP=path: write the state at the first cycle `stall` forces (see
    // `dump_deadlock_state`). A debugging aid; off unless the variable is set.
    let deadlock_dump_path = std::env::var("SCHEDULER_DEADLOCK_DUMP").ok();
    let mut deadlock_dumped = false;

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

    // Split shifts: only when the scheduler itself decides shifts (not with fixed encodings).
    let split_shift_enabled = shape.split_shift
        && enable_valu_shift
        && fixed_shifted_ops.is_none()
        && !valu_shift_fixed_to_none;
    // Shift policy: proactive shifts (before the vector slots are full) only when the
    // scheduler itself decides shifts.
    let proactive_shift_enabled = shape.shift_policy != ShiftPolicy::WhenFull
        && enable_valu_shift
        && fixed_shifted_ops.is_none()
        && !valu_shift_fixed_to_none;
    // Pace accounting. "Vector ops" are ops whose own engine is `Valu` (not dead stores): the
    // same set in the total and in the placed count, so a SelectChoice placed on the vector
    // path counts in neither. `pace_vector_placed` and `pace_shifted` count placements on
    // every path (normal, split start, force-schedule). The pace is a floor on shifting, not
    // a cap: a pace policy shifts proactively while `shifted / placed < target / total`, and
    // the `when-full` shifts (vector slots full) still happen once the target is reached.
    let pace_total_vector = ops
        .iter()
        .enumerate()
        .filter(|(i, op)| op.engine == Engine::Valu && !dead_store_indices.contains(i))
        .count();
    let pace_target: Option<usize> = if proactive_shift_enabled {
        shape.shift_policy.pace().map(|explicit| {
            explicit.unwrap_or_else(|| {
                let shiftable = (0..ops.len())
                    .filter(|i| shiftable_set.contains(i) && !dead_store_indices.contains(i))
                    .count();
                let scalar = ops
                    .iter()
                    .enumerate()
                    .filter(|(i, op)| op.engine == Engine::Alu && !dead_store_indices.contains(i))
                    .count();
                balanced_shift_target(pace_total_vector, shiftable, scalar)
            })
        })
    } else {
        None
    };
    if diagnostics.verbose {
        if let Some(t) = pace_target {
            eprintln!(
                "Greedy shift pace: target {} of {} vector ops shifted",
                t, pace_total_vector
            );
        }
    }
    // Vector ops that cannot shift (multiply_add, vbroadcast, SelectChoice on the vector path).
    let fixed_vector: Vec<bool> = (0..ops.len())
        .map(|i| ops[i].engine == Engine::Valu && !shiftable_set.contains(&i))
        .collect();
    if shape.load_priority != LoadPriority::Key && wavefront.is_none() {
        eprintln!(
            "Warning: greedy load priority {:?} applies only with the wavefront priority; ignored",
            shape.load_priority
        );
    }
    let mut pace_vector_placed: usize = 0;
    let mut pace_shifted: usize = 0;
    // The op whose first lanes ran last cycle and whose remaining lanes run this cycle.
    // It is not in `scheduled` until it completes, so the loop below keeps running for it.
    let mut split_carry: Option<SplitShift> = None;
    // Vector slots reserved in the next cycle for the multiply_add of a select placed on the
    // multiply-add path in this cycle (see fma_extra_latency): the subtract runs at the
    // select's cycle, the multiply_add one cycle later, one vector slot each.
    let mut fma_carry: usize = 0;
    // SelectChoice ops this run placed on the vector ALU in their one-op multiply_add form
    // (`SelectOverflow::Valu`); reported with the other multiply-add selects at the end.
    let mut overflow_select_fma: HashSet<usize> = HashSet::new();

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
                    live.remove(&vreg);
                    trace.release(&vreg, cycle);
                }
            } else {
                break;
            }
        }

        let mut slots = SlotTracker::new();
        slots.valu_used += std::mem::take(&mut fma_carry);
        debug_assert!(slots.valu_used <= 6);

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
                        let offset = result_latency(ops, dep_idx);
                        ops[dep_idx].cycle + offset <= cycle as i32
                    })
            }) && mem_predecessors[idx].iter().all(|&(src, latency)| {
                dead_store_indices.contains(&src)
                    || (ops[src].cycle >= 0 && ops[src].cycle + latency <= cycle as i32)
            });

            // Wavefront window: a chain beyond the in-flight window waits for an earlier chain
            // to finish.
            let in_window = wavefront.as_ref().map_or(true, |w| w.in_window(idx));
            if deps_ready && in_window {
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
        // A split shift started this cycle: its destination block is reserved and scalar
        // slots are taken, so the cycle is not a deadlock for `BudgetForce::Stall`.
        let mut split_started_this_cycle = false;

        // Finish last cycle's split shift before anything else takes scalar slots: its
        // remaining lanes run now and the op completes in this cycle.
        if let Some(split) = split_carry.take() {
            let idx = split.idx;
            slots.alu_used += VLEN - split.first_lanes;
            debug_assert!(slots.alu_used <= 12);
            shifted_lane_cycles.insert(ops[idx].id, split.lane_cycles(cycle as i32));
            shifted_ops.insert(ops[idx].id);
            ops[idx].cycle = cycle as i32;
            scheduled.insert(idx);
            scheduled_this_cycle += 1;
            if let Some(w) = wavefront.as_mut() {
                w.on_scheduled(idx);
            }
            // Operands are read by every lane, so they are released only now.
            release_read_values(
                &ops[idx],
                cycle,
                &liveness_aliases,
                &write_groups,
                &mut vreg_remaining_users,
                &mut live_vregs,
                &mut live,
                &mut vreg_expiry_heap,
                trace,
            );
            peak_live_words = peak_live_words.max(live.words);
            for &consumer_idx in &consumers[idx] {
                let offset = ops[idx].precedence_offset_for_asap(shiftable_set.contains(&idx));
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
            release_memory_successors(
                &mem_successors[idx],
                cycle as i32,
                ops,
                bounds_usage,
                &mut max_ready_cycle,
                &mut remaining_deps,
                &mut ready_heap,
            );
        }

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
            live_words: pools.words_needed(&live, scratch_budget),
            scratch_budget,
            slots: &slots.view(),
        };
        strategy.sort_ready(&mut ready_candidates, &sched_ctx);
        ready_this_cycle = ready_candidates.into_iter().map(|c| c.op_index).collect();
        // Wavefront priority overrides the queue order; the policy's order breaks key ties.
        if let Some(w) = wavefront.as_ref() {
            ready_this_cycle.sort_by_key(|&idx| w.key(idx));
            if shape.load_priority != LoadPriority::Key {
                // Reorder the loads among the positions loads hold; every other op keeps its
                // place in the wavefront order.
                let positions: Vec<usize> = (0..ready_this_cycle.len())
                    .filter(|&p| ops[ready_this_cycle[p]].engine == Engine::Load)
                    .collect();
                let mut loads: Vec<usize> = positions.iter().map(|&p| ready_this_cycle[p]).collect();
                match shape.load_priority {
                    LoadPriority::Tail => loads.sort_by_key(|&idx| (-w.height(idx), w.key(idx), idx)),
                    _ => loads.sort_by_key(|&idx| (!w.feeds_load(idx), w.key(idx), idx)),
                }
                for (&p, idx) in positions.iter().zip(loads) {
                    ready_this_cycle[p] = idx;
                }
            }
            if shape.flow_priority == FlowPriority::ChainFirst {
                let positions: Vec<usize> = (0..ready_this_cycle.len())
                    .filter(|&p| ops[ready_this_cycle[p]].engine == Engine::Flow)
                    .collect();
                let mut flows: Vec<usize> = positions.iter().map(|&p| ready_this_cycle[p]).collect();
                flows.sort_by_key(|&idx| (w.chain(idx).map_or(0, |r| r + 1), w.key(idx), idx));
                for (&p, idx) in positions.iter().zip(flows) {
                    ready_this_cycle[p] = idx;
                }
            }
            if shape.alu_priority == AluPriority::NativeFirst {
                // Stable: natives keep their order among themselves, as does everything else.
                ready_this_cycle.sort_by_key(|&idx| ops[idx].engine != Engine::Alu);
            }
            if shape.flow_priority == FlowPriority::FeedsLoadFirst {
                // Same for the flow ops (ConstChoice is a load-engine op in the SSA).
                let positions: Vec<usize> = (0..ready_this_cycle.len())
                    .filter(|&p| ops[ready_this_cycle[p]].engine == Engine::Flow)
                    .collect();
                let mut flows: Vec<usize> = positions.iter().map(|&p| ready_this_cycle[p]).collect();
                flows.sort_by_key(|&idx| (!w.feeds_nonfinal_load(idx), w.key(idx), idx));
                for (&p, idx) in positions.iter().zip(flows) {
                    ready_this_cycle[p] = idx;
                }
            }
        }

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
                live_words: pools.words_needed(&live, scratch_budget),
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

            // Shift policy `reserve`: non-shiftable vector ops still to be tried this pass.
            let mut pending_fixed_vector = if proactive_shift_enabled && shape.shift_policy.reserve() {
                ops_to_try
                    .iter()
                    .filter(|&&idx| !scheduled.contains(&idx) && fixed_vector[idx])
                    .count()
            } else {
                0
            };

            for idx in ops_to_try {
                if scheduled.contains(&idx) {
                    continue;
                }
                if pending_fixed_vector > 0 && fixed_vector[idx] {
                    pending_fixed_vector -= 1;
                }

                // ConstChoice: use add_imm (Flow) only if there is a free Flow slot and no
                // other schedulable Flow op (vselect, jump, etc.); otherwise use Load.
                // When const_choice_fixed_to_load (hint pass enabled), always use Load so the hint pass owns the decision.
                // When fixed_const_choice_encoding is Some (schedule-with-fixed-choices), use the map.
                // SelectChoice: when fixed_select_choice_encoding is Some, use map (Fma => VALU, Vselect => Flow).
                let op_id = ops[idx].id;
                let mut const_derive: Option<crate::types::AluDeriveInfo> = None;
                let engine = if ops[idx].opcode == Opcode::ConstChoice {
                    let default_rule = |slots: &SlotTracker| {
                        if slots.flow_used < 1 && flow_other_ready == 0 {
                            Engine::Flow
                        } else {
                            Engine::Load
                        }
                    };
                    if let Some(enc) = fixed_const_choice_encoding.and_then(|m| m.get(&op_id)) {
                        match enc {
                            ConstChoiceEncoding::AddImm => Engine::Flow,
                            ConstChoiceEncoding::AluDerive(info) => {
                                // A derivation of the first run: honour it only where it is
                                // sound as the policy path makes it (both producers scheduled
                                // in an earlier cycle and still live, ALU room); otherwise
                                // the rerun loads the constant and the caller records Load
                                // (`reconcile_const_choice_encodings`). Never an ALU op the
                                // slot count and the precedences do not see.
                                let producer_ok = |id: usize| {
                                    id_to_idx.get(&id).is_some_and(|&p| {
                                        ops[p].cycle >= 0
                                            && (ops[p].cycle as usize) < cycle
                                            && ops[p].dest.is_some_and(|d| live_vregs.contains(&d))
                                    })
                                };
                                if slots.alu_used < 12 && producer_ok(info.left_op_id) && producer_ok(info.right_op_id) {
                                    const_derive = Some(info.clone());
                                    Engine::Alu
                                } else {
                                    Engine::Load
                                }
                            }
                            _ => Engine::Load,
                        }
                    } else if const_choice_fixed_to_load {
                        Engine::Load
                    } else if shape.const_policy == ConstPolicy::Default {
                        default_rule(&slots)
                    } else if shape.const_policy == ConstPolicy::AluWhenLoadFull && slots.load_used < 2 {
                        Engine::Load
                    } else {
                        const_derive = if slots.alu_used < 12 {
                            // A producer of `value` scheduled in an earlier cycle whose value
                            // is still live (so the read needs no lifetime extension).
                            let live_producer = |value: i64| {
                                value_producers.get(&value).and_then(|ps| {
                                    ps.iter().copied().find(|&p| {
                                        ops[p].cycle >= 0
                                            && (ops[p].cycle as usize) < cycle
                                            && ops[p].dest.is_some_and(|d| live_vregs.contains(&d))
                                    })
                                })
                            };
                            let derive = |a: i64, b: i64, opcode: Opcode| {
                                Some(crate::types::AluDeriveInfo {
                                    left_op_id: ops[live_producer(a)?].id,
                                    right_op_id: ops[live_producer(b)?].id,
                                    opcode,
                                })
                            };
                            // The emitter's alternatives first, then any two live constants
                            // that add up to the value (non-negative, below 2^31: no wrap).
                            let target = ops[idx].produces_constant;
                            const_derivations
                                .get(&idx)
                                .and_then(|alts| alts.iter().find_map(|&(a, b, opcode)| derive(a, b, opcode)))
                                .or_else(|| {
                                    let v = target.filter(|&v| v > 0 && v < (1i64 << 31))?;
                                    let mut values: Vec<i64> =
                                        value_producers.keys().copied().filter(|&a| a > 0 && a < v).collect();
                                    values.sort_unstable_by(|x, y| y.cmp(x));
                                    values.into_iter().find_map(|a| derive(a, v - a, Opcode::Add))
                                })
                        } else {
                            None
                        };
                        if const_derive.is_some() {
                            Engine::Alu
                        } else {
                            default_rule(&slots)
                        }
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
                    } else if shape.select_overflow.allows(overflow_select_fma.len())
                        && ops[idx].delta_vreg.is_some()
                        && slots.flow_used >= 1
                        && slots.valu_used < 6
                    {
                        // Flow slot taken this cycle: the one-op multiply_add form.
                        Engine::Valu
                    } else {
                        ops[idx].engine // default Flow (vselect)
                    }
                } else {
                    ops[idx].engine
                };
                let dest = ops[idx].dest;

                // When fixed_shifted_ops is Some, can_shift = op is in the set (we place as shifted only those).
                // When in fixed_shifted_ops we must place as shifted only (never VALU), so we need ALU room.
                // With valu_shift_fixed_to_none the hint pass owns the shift decision: never shift
                // here (a shift the slot tracker counted but nobody recorded would lower as a
                // seventh vector op).
                let can_shift = if let Some(fixed) = fixed_shifted_ops {
                    fixed.contains(&op_id)
                } else {
                    enable_valu_shift
                        && !valu_shift_fixed_to_none
                        && engine == Engine::Valu
                        && can_ssa_valu_shift_to_alu(&ops[idx])
                };
                let must_place_shifted = fixed_shifted_ops.map_or(false, |s| s.contains(&op_id));

                // Proactive shift (shift policy other than `when-full`): run this op on the
                // scalar ALU although a vector slot is free.
                let prefer_shift = proactive_shift_enabled
                    && can_shift
                    && engine == Engine::Valu
                    && slots.valu_used < 6
                    && ((shape.shift_policy.reserve()
                        && pending_fixed_vector > 0
                        && pending_fixed_vector >= 6 - slots.valu_used)
                        || pace_target.is_some_and(|t| {
                            (pace_shifted as u128) * (pace_total_vector as u128)
                                < (t as u128) * ((pace_vector_placed + 1) as u128)
                        }))
                    && strategy.allow_shift(
                        pools.words_needed(&live, scratch_budget),
                        slots.alu_used,
                        scratch_budget,
                    );
                // Whole shift does not fit: try a split shift, else run on the vector ALU.
                let prefer_split = prefer_shift && split_shift_enabled && slots.alu_used + VLEN > 12;

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
                } else if !slots.fits(engine, can_shift) || prefer_split {
                    // Split shift: the vector slots are full (or the shift policy prefers the
                    // scalar ALU) and a whole shifted op does not fit, but scalar slots are
                    // left. Run the first lanes now and the rest next cycle. The destination
                    // block is reserved now; operands are released when the op completes, so
                    // nothing is freed in this cycle.
                    let free_alu = 12 - slots.alu_used;
                    if split_shift_enabled
                        && can_shift
                        && engine == Engine::Valu
                        && split_carry.is_none()
                        && free_alu >= SplitShift::MIN_FIRST_LANES
                        && free_alu < VLEN
                        && strategy.allow_shift(pools.words_needed(&live, scratch_budget), slots.alu_used, scratch_budget)
                    {
                        if let Some(d) = dest {
                            let counted = write_groups.counted(d);
                            let added: Vec<VReg> =
                                if live_vregs.contains(&counted) { vec![] } else { vec![counted] };
                            // A destination no scheduled op reads is released the cycle
                            // after its write, which for a split would be before its last
                            // lanes are written: such an op is only placed whole.
                            let has_reader =
                                vreg_remaining_users.get(&counted).is_some_and(|&n| n > 0);
                            if has_reader && pools.fits(&live.after(&added, &[]), scratch_budget) {
                                slots.alu_used += free_alu;
                                split_started_this_cycle = true;
                                split_carry = Some(SplitShift {
                                    idx,
                                    first_cycle: cycle as i32,
                                    first_lanes: free_alu,
                                });
                                made_progress = true;
                                count_written_dest(
                                    d,
                                    cycle,
                                    &mut write_groups,
                                    &vreg_remaining_users,
                                    &mut live_vregs,
                                    &mut live,
                                    &mut vreg_expiry_heap,
                                    trace,
                                );
                                vreg_def_cycle.insert(d, cycle as i32);
                                peak_live_words = peak_live_words.max(live.words);
                                pace_vector_placed += 1;
                                pace_shifted += 1;
                                if prefer_split {
                                    diagnostics.proactive_shifts += 1;
                                }
                                continue;
                            }
                        }
                    }
                    // A preferred split that could not start: the op still fits a vector slot.
                    if !prefer_split {
                        // Reinsert to heap for next cycle
                        let next_priority = bounds_usage
                            .core_bounds()
                            .and_then(|(a, _)| a.get(&op_id).copied())
                            .map(|lb| ((cycle + 1) as i32).max(lb))
                            .unwrap_or_else(|| (cycle + 1) as i32);
                        ready_heap.push(Reverse((next_priority, idx)));
                        continue;
                    }
                }

                // Check scratch budget with same-cycle reuse model:
                // VRegs that will be freed when this op runs (their last use) reduce pressure.
                let added: Vec<VReg> = match dest.map(|d| write_groups.counted(d)) {
                    // Later lanes of a write group land in the block its first lane reserved.
                    Some(counted) if live_vregs.contains(&counted) => vec![],
                    Some(counted) => vec![counted],
                    None => vec![],
                };
                let words_added: usize = added.iter().map(|v| v.size()).sum();
                // Exactly what the release loop below frees: every value (operand, keep_live
                // or vector-group alias) for which this op is the last remaining reader.
                // A select taking the multiply_add form holds what it reads to cycle + 1.
                let holds_reads = engine == Engine::Valu && ops[idx].opcode == Opcode::SelectChoice;
                let freed: Vec<VReg> = liveness_release_vregs(
                    &ops[idx],
                    &liveness_aliases,
                    trace.delta_readers.contains(&op_id),
                )
                    .into_iter()
                    .filter(|_| !holds_reads)
                    .filter(|vreg| {
                        vreg_remaining_users.get(vreg).map_or(false, |&n| n == 1)
                            && live_vregs.contains(vreg)
                            && write_groups.releasable_at(vreg, cycle)
                    })
                    .collect();
                let words_freed: usize = freed.iter().map(|v| v.size()).sum();
                // Net pressure change: add new VReg, subtract freed VRegs (same-cycle reuse)
                if !pools.fits(&live.after(&added, &freed), scratch_budget) {
                    // Defer for retry - liveness may decrease after other ops scheduled
                    diagnostics.budget_check_deferrals += 1;
                    budget_deferred.push(idx);
                    continue;
                }
                // Strategy-driven soft deferral (pressure awareness)
                if strategy.should_defer_for_pressure(
                    pools.words_needed(&live, scratch_budget),
                    words_added,
                    words_freed,
                    scratch_budget,
                ) {
                    budget_deferred.push(idx);
                    continue;
                }

                // Schedule the op via SlotTracker.
                // When must_place_shifted (op in fixed_shifted_ops), use shift-only path so hint matches (ALU count).
                let allow_shift = strategy.allow_shift(pools.words_needed(&live, scratch_budget), slots.alu_used, scratch_budget);
                let place_result = if must_place_shifted {
                    slots.try_place_shifted_only()
                } else if prefer_shift && slots.alu_used + VLEN <= 12 {
                    slots.try_place_shifted_only()
                } else {
                    slots.try_place(engine, can_shift, allow_shift)
                };
                if ops[idx].engine == Engine::Valu && place_result != SlotResult::NoRoom {
                    pace_vector_placed += 1;
                    if place_result == SlotResult::Shifted {
                        pace_shifted += 1;
                        if prefer_shift {
                            diagnostics.proactive_shifts += 1;
                        }
                    }
                }
                match place_result {
                    SlotResult::Placed => {
                        if engine == Engine::Valu && fma_extra_latency.contains(&idx) {
                            fma_carry += 1;
                        }
                        if engine == Engine::Valu && ops[idx].opcode == Opcode::SelectChoice
                            && fixed_select_choice_encoding.is_none()
                        {
                            overflow_select_fma.insert(op_id);
                        }
                        if let Some(info) = const_derive.take() {
                            debug_assert_eq!(engine, Engine::Alu);
                            diagnostics.const_alu_derive.insert(op_id, info);
                        }
                        if engine == Engine::Valu && ops[idx].opcode == Opcode::SelectChoice
                            && ops[idx].delta_vreg.is_some()
                        {
                            trace.fma_selects.insert(op_id);
                        }
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
                        // With fixed shifts the caller records them (the op is in the set).
                        if fixed_shifted_ops.is_none() {
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
                if let Some(w) = wavefront.as_mut() {
                    w.on_scheduled(idx);
                }
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
                        &mut live,
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
                    &mut live,
                    &mut vreg_expiry_heap,
                    trace,
                );

                // Record peak after both adds and frees, matching the same-cycle
                // reuse model used by address assignment's interval-based liveness.
                peak_live_words = peak_live_words.max(live.words);

                // Add newly-ready consumers to heap (O(log n) per consumer)
                for &consumer_idx in &consumers[idx] {
                    // Update max ready cycle from this predecessor's offset
                    let offset = result_latency(ops, idx);
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
                release_memory_successors(
                    &mem_successors[idx],
                    cycle as i32,
                    ops,
                    bounds_usage,
                    &mut max_ready_cycle,
                    &mut remaining_deps,
                    &mut ready_heap,
                );
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
        // With pool capacities (scratch model `pools`) a budget deferral is a real deferral:
        // the op waits for a later cycle, and the force path runs only when nothing at all
        // was scheduled this cycle. (Without capacities the historical behaviour is kept: a
        // deferred op that has a slot is forced in the same cycle.)
        //
        // `BudgetForce::Stall` gives the words model the same rule: an op the budget check
        // deferred waits while anything else is placed, and is forced only in a cycle where
        // nothing was (docs/SCRATCH_CEILING_DIAGNOSIS.md).
        let force_same_cycle = pools.caps.is_none() && shape.budget_force == BudgetForce::SameCycle;
        // With `Stall`, a cycle whose only placement was a split-shift start is not a
        // deadlock either (`pools` keeps its historical rule). Defensive: a split starts only
        // with 1 to 7 scalar slots free, which takes a counted placement on the scalar ALU in
        // the same cycle (`split_only_cycles` stays 0; checked in the tests).
        let stall_progress = shape.budget_force == BudgetForce::Stall && split_started_this_cycle;
        if split_started_this_cycle && scheduled_this_cycle == 0 {
            diagnostics.split_only_cycles += 1;
        }
        if !budget_deferred.is_empty() && (force_same_cycle || (scheduled_this_cycle == 0 && !stall_progress)) {
            if let (Some(path), false, false) = (deadlock_dump_path.as_ref(), deadlock_dumped, force_same_cycle) {
                deadlock_dumped = true;
                let state = serde_json::json!({
                    "cycle": cycle,
                    "live_words": live.words,
                    "budget": scratch_budget,
                    "live": live_vregs.iter().map(|v| serde_json::json!({
                        "vreg": v.id, "words": v.size(),
                        "remaining_users": vreg_remaining_users.get(v).copied().unwrap_or(0),
                    })).collect::<Vec<_>>(),
                    "budget_deferred": budget_deferred.iter().map(|&i| ops[i].id).collect::<Vec<_>>(),
                    "heap": ready_heap.iter().map(|Reverse((pri, i))| serde_json::json!({
                        "id": ops[*i].id, "ready_at": pri,
                        "in_window": wavefront.as_ref().map_or(true, |w| w.in_window(*i)),
                    })).collect::<Vec<_>>(),
                    "slots": {"valu": slots.valu_used, "alu": slots.alu_used, "load": slots.load_used,
                              "flow": slots.flow_used, "store": slots.store_used},
                });
                // One dump per greedy run; a build that runs the greedy again writes the n-th
                // dump to `<path>.<n>` instead of overwriting the first.
                static DUMPS: std::sync::Mutex<std::collections::BTreeMap<String, usize>> =
                    std::sync::Mutex::new(std::collections::BTreeMap::new());
                let n = {
                    let mut dumps = DUMPS.lock().unwrap_or_else(|e| e.into_inner());
                    let n = dumps.entry(path.clone()).or_insert(0);
                    *n += 1;
                    *n - 1
                };
                let path = if n == 0 { path.clone() } else { format!("{}.{}", path, n) };
                if let Err(e) = std::fs::write(&path, state.to_string()) {
                    eprintln!("SCHEDULER_DEADLOCK_DUMP: cannot write {}: {}", path, e);
                }
            }
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
            // Same rule as the normal path: no shifts when the hint pass owns them.
            let can_shift = if let Some(fixed) = fixed_shifted_ops {
                fixed.contains(&op_id)
            } else {
                enable_valu_shift
                    && !valu_shift_fixed_to_none
                    && engine == Engine::Valu
                    && can_ssa_valu_shift_to_alu(&ops[force_idx])
            };
            let must_place_shifted = fixed_shifted_ops.map_or(false, |s| s.contains(&op_id));

            // Take the slot first; budget accounting below only runs for an op that has one.
            // When must_place_shifted use the shift-only path; otherwise try with the strategy
            // first and, if it blocks the shift, force the shift (the op is over budget anyway).
            // `None` = no slot this cycle: the op goes back to the deferred list and takes a
            // later cycle (marking it scheduled without a slot overfilled the cycle).
            let force_place_result = if must_place_shifted {
                slots.try_place_shifted_only()
            } else {
                let allow_shift = strategy.allow_shift(pools.words_needed(&live, scratch_budget), slots.alu_used, scratch_budget);
                match slots.try_place(engine, can_shift, allow_shift) {
                    SlotResult::NoRoom => slots.try_place(engine, can_shift, true),
                    placed => placed,
                }
            };

            if force_place_result != SlotResult::NoRoom {
                // CRITICAL: Force schedule, ignoring budget constraint - this exceeds budget!
                // This is intentional behavior - DO NOT REMOVE unless explicitly instructed.
                // The force-schedule allows progress when retry loops fail to free enough
                // VRegs. The budget_exceeded flag tracks this for reporting.
                //
                // Count the over-budget cycle. The detailed breakdown is rendered only for
                // the first such cycle and for a new worst one, and only when verbose: a run
                // that stays over budget for thousands of cycles used to dump every cycle.
                diagnostics.record_over_budget(cycle, pools.words_needed(&live, scratch_budget), || {
                    format_live_usage_breakdown(ops, &live_vregs, cycle, live.words, scratch_budget)
                });
                budget_exceeded = true;
                diagnostics.forced_past_budget += 1;
                let mut force_shifted = false;
                match force_place_result {
                    SlotResult::Placed => {
                        if engine == Engine::Valu && fma_extra_latency.contains(&force_idx) {
                            fma_carry += 1;
                        }
                        if engine == Engine::Valu && ops[force_idx].opcode == Opcode::SelectChoice
                            && ops[force_idx].delta_vreg.is_some()
                        {
                            trace.fma_selects.insert(op_id);
                        }
                    }
                    SlotResult::Shifted => {
                        force_shifted = true;
                        // With fixed shifts the caller records them (the op is in the set).
                        if fixed_shifted_ops.is_none() {
                            shifted_ops.insert(op_id);
                        }
                    }
                    SlotResult::NoRoom => unreachable!("checked above"),
                }
                if ops[force_idx].engine == Engine::Valu {
                    pace_vector_placed += 1;
                    if force_shifted {
                        pace_shifted += 1;
                    }
                }

                // Record the cycle we actually placed in (same as normal path).
                let force_cycle = cycle as i32;
                ops[force_idx].cycle = force_cycle;
                if let Some(w) = wavefront.as_mut() {
                    w.on_scheduled(force_idx);
                }
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
                        &mut live,
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
                    &mut live,
                    &mut vreg_expiry_heap,
                    trace,
                );

                // Record peak after both adds and frees
                peak_live_words = peak_live_words.max(live.words);

                for &consumer_idx in &consumers[force_idx] {
                    let offset = result_latency(ops, force_idx);
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
                release_memory_successors(
                    &mem_successors[force_idx],
                    cycle as i32,
                    ops,
                    bounds_usage,
                    &mut max_ready_cycle,
                    &mut remaining_deps,
                    &mut ready_heap,
                );
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
        trace.cycle_end_live[cycle] = live.words;

        cycle += 1;
    }

    // Suppress unused warning for vreg_def_cycle - it's used for debugging
    let _ = vreg_def_cycle;

    if scheduled.len() < ops.len() {
        // Diagnose why scheduling failed (a split shift still in progress counts as unscheduled)
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
            live.words,
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
    if select_choice_fma_ops.is_none() && !overflow_select_fma.is_empty() {
        // Lowering would emit these as vselects on the flow slot the greedy did not count.
        return Err("select overflow placed SelectChoice ops on the vector ALU, but the caller \
                    does not collect multiply-add selects"
            .to_string());
    }
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
        fma_ops.extend(overflow_select_fma.iter().copied());
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

    if diagnostics.verbose {
        eprintln!(
            "Greedy budget check: {} deferrals, {} ops forced past the budget",
            diagnostics.budget_check_deferrals, diagnostics.forced_past_budget
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
                    // Memory edges: source placed in an earlier cycle (at least the edge
                    // latency, which is at most 1)
                    && op.mem_deps.iter().all(|(src_id, _)| {
                        let Some(&src_idx) = id_to_idx.get(src_id) else {
                            return true;
                        };
                        let src_cycle = ops[src_idx].cycle;
                        src_cycle >= 0 && src_cycle < cycle as i32
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
