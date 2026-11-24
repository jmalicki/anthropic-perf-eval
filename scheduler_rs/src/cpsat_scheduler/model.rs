#[cfg(feature = "cpsat")]
use cp_sat::builder::{BoolVar, CpModelBuilder, IntVar, IntervalVar, LinearExpr};
#[cfg(feature = "cpsat")]
use cp_sat::proto::SatParameters;

use std::collections::{HashMap, HashSet};
use std::ops::Not;

use crate::types::{Engine, Opcode, Operand, SsaOp, VReg, VLEN};

use super::bounds::{
    compute_capacity_lower_bound_global, compute_max_latency_critical_path, compute_ssa_alap,
    compute_ssa_asap, diagnose_precedence_bounds, fix_precedence_bounds_fixpoint,
    tighten_asap_alap_by_scratch, tighten_asap_alap_by_slots,
    tighten_asap_by_symmetry_ordering,
    SsaCpsatConfig, SsaCpsatHints, SsaCpsatResult,
};
use super::symmetry_ordering::SymmetryOrdering;
#[cfg(feature = "cpsat")]
use super::op_times::{create_op_variables, ConstChoiceVars, OpKinds, TimeDomains};
use super::config::{
    ConstChoiceMode, CpsatObjective, CpsatStatus, CpsatTuning, CumulativePropagator, MakespanRows,
    OptionalConstraintGroup, ScratchTime, SelectChoiceMode, SymmetryRows, ValuShiftMode,
};
#[cfg(feature = "cpsat")]
use super::makespan_rows::{add_makespan_rows, static_paths, OpClasses, RowCaps, WorkContext};
#[cfg(feature = "cpsat")]
use super::validation::{
    build_flow_audit, summarise_ssa_solution_engines_and_choices,
    validate_ssa_slot_limits, verify_all_raw_precedence, verify_cse_sub_precedence,
    verify_memory_precedence, verify_scratch_budget_all_readers,
    verify_select_choice_fma_consumer_precedence,
};

/// Whether the greedy schedule's encoding choices are allowed by the forced modes.
///
/// # Arguments
/// * `ops` - SSA ops
/// * `shiftable_valu` - Indices of VALU ops that may be shifted to ALU
/// * `valu_shift_mode`, `const_choice_mode`, `select_choice_mode` - Forced or free encodings
/// * `shift_hints`, `const_add_imm_hints`, `select_choice_hints` - Op IDs for which the greedy
///   chose ALU, add_imm and FMA respectively (None = chose none)
///
/// # Returns
/// `true` if every forced encoding agrees with what the greedy chose, so the greedy cycles
/// are a feasible schedule of the model being built.
#[cfg(feature = "cpsat")]
#[allow(clippy::too_many_arguments)]
fn greedy_schedule_matches_forced_modes(
    ops: &[SsaOp],
    shiftable_valu: &[usize],
    valu_shift_mode: ValuShiftMode,
    const_choice_mode: ConstChoiceMode,
    select_choice_mode: SelectChoiceMode,
    shift_hints: Option<&HashSet<usize>>,
    const_add_imm_hints: Option<&HashSet<usize>>,
    select_choice_hints: Option<&HashSet<usize>>,
) -> bool {
    // `all` = every listed op is in the greedy's set; `none` = the greedy's set is empty.
    let chosen = |set: Option<&HashSet<usize>>, id: usize| set.map_or(false, |s| s.contains(&id));
    let none_chosen = |set: Option<&HashSet<usize>>| set.map_or(true, |s| s.is_empty());
    let ids_of = |opcode: Opcode| ops.iter().filter(move |op| op.opcode == opcode).map(|op| op.id);

    let shift_ok = match valu_shift_mode {
        ValuShiftMode::Optimize => true,
        ValuShiftMode::ForceAlu => shiftable_valu.iter().all(|&idx| chosen(shift_hints, ops[idx].id)),
        ValuShiftMode::ForceValu => none_chosen(shift_hints),
    };
    let const_ok = match const_choice_mode {
        ConstChoiceMode::Optimize => true,
        ConstChoiceMode::ForceAddImm => ids_of(Opcode::ConstChoice).all(|id| chosen(const_add_imm_hints, id)),
        ConstChoiceMode::ForceLoad => none_chosen(const_add_imm_hints),
    };
    let select_ok = match select_choice_mode {
        SelectChoiceMode::Optimize => true,
        SelectChoiceMode::ForceFma => ids_of(Opcode::SelectChoice).all(|id| chosen(select_choice_hints, id)),
        SelectChoiceMode::ForceVselect => none_chosen(select_choice_hints),
    };
    shift_ok && const_ok && select_ok
}

/// Fallback deterministic-time budget for each CP-SAT solve, from the
/// `CPSAT_MAX_DETERMINISTIC_TIME` environment variable. Used only when the config does not
/// set `deterministic_time_limit` (e.g. for tests and tools that call the solver directly).
///
/// # Returns
/// `Some(limit)` when the variable holds a positive number, else `None` (no limit).
#[cfg(feature = "cpsat")]
fn max_deterministic_time_from_env() -> Option<f64> {
    std::env::var("CPSAT_MAX_DETERMINISTIC_TIME")
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|d| d.is_finite() && *d > 0.0)
}

/// Lower bound on the makespan, in cycles, implied by CP-SAT's bound on the objective.
///
/// The objective is `scale * makespan + tie_breaker` with `0 <= tie_breaker <= scale - 1`
/// (the `tie_sum_completion` row), or the makespan itself when there is no tie-breaker
/// (`objective_scale` is `None`, read as scale 1). All terms are integers, so:
///
/// * the objective is at least `ceil(bound)`: a fractional bound (from an LP) is rounded up
///   first. A small epsilon keeps floating-point noise just above an integer from adding one;
/// * `makespan >= (ceil(bound) - (scale - 1)) / scale`, i.e. `makespan >= floor(ceil(bound) / scale)`.
///
/// For an integral bound this is `floor(bound / scale)`, what was computed before; for a
/// fractional one it is tighter by up to one cycle and still valid. Rounding the quotient up
/// instead would not be valid with a tie-breaker: a bound of `12 * scale - 1` allows
/// makespan 11 with a tie-breaker of `scale - 1`.
///
/// # Arguments
/// * `best_objective_bound` - `CpSolverResponse::best_objective_bound` of a minimisation
/// * `objective_scale` - The makespan's coefficient in the objective; `None` when the
///   objective is the makespan alone
///
/// # Returns
/// The smallest makespan the bound still allows (0 for a non-positive or non-finite bound).
pub(crate) fn makespan_lower_bound(best_objective_bound: f64, objective_scale: Option<i64>) -> usize {
    const EPSILON: f64 = 1e-6;
    let scale = objective_scale.unwrap_or(1).max(1);
    let objective_at_least = (best_objective_bound - EPSILON).ceil();
    if !objective_at_least.is_finite() || objective_at_least <= 0.0 {
        return 0;
    }
    ((objective_at_least as i64).div_euclid(scale)) as usize
}

/// Solver parameters for one CP-SAT solve.
///
/// # Arguments
/// * `time_limit_secs` - Wall-clock limit (`max_time_in_seconds`).
/// * `deterministic_budget` - `max_deterministic_time`, when a budget is set.
/// * `stop_after_first_solution` - Feasibility-only solve: one worker, stop at the first solution.
/// * `absolute_gap_limit` - `absolute_gap_limit`, when the objective defines one.
/// * `tuning` - Switchable search configuration; `CpsatTuning::default()` gives the
///   parameters used before the switches existed (8 workers, interleaved search).
///
/// # Returns
/// The `SatParameters` handed to the solver.
#[cfg(feature = "cpsat")]
pub(crate) fn build_sat_parameters(
    time_limit_secs: f64,
    deterministic_budget: Option<f64>,
    stop_after_first_solution: bool,
    absolute_gap_limit: Option<f64>,
    tuning: &CpsatTuning,
) -> SatParameters {
    let mut params = SatParameters::default();
    params.max_time_in_seconds = Some(time_limit_secs);
    params.max_deterministic_time = deterministic_budget;
    // Optional: crash when presolve adjusts a hint. Not default: presolve can tighten domains and
    // clamp hints; that's expected and the solve still succeeds.
    if std::env::var("CPSAT_DEBUG_PRESOLVE_HINT").is_ok() {
        params.debug_crash_if_presolve_breaks_hint = Some(true);
    }
    // Always log search progress by default - user must go out of their way to suppress
    params.log_search_progress = Some(true);
    // With multiple workers, CP-SAT may keep improving after one worker finds a feasible solution;
    // use a single worker when we only need feasibility so stop_after_first_solution is honored.
    params.num_workers = if stop_after_first_solution {
        Some(1)
    } else {
        Some(tuning.workers)
    };
    params.absolute_gap_limit = absolute_gap_limit;

    // linearization_level (tuning, default 2): how much of the model enters the LP relaxation
    // (2 adds the cumulative relaxations and cuts). cp_model_probing_level (tuning, default 2):
    // presolve probing effort. Neither default was chosen by measurement; the measured effect
    // of each level is in docs/CPSAT_CONFIG_ABLATION.md ("Linearization, probing and makespan
    // rows"). randomize_search=true is likewise unmeasured.
    params.linearization_level = Some(tuning.linearization_level);
    params.cp_model_presolve = Some(true);
    params.cp_model_probing_level = Some(tuning.probing_level);
    params.randomize_search = Some(true);
    if let Some(seed) = tuning.random_seed {
        params.random_seed = Some(seed);
    }
    // interleave_search: runs the subsolvers round-robin in deterministic batches, so the
    // result does not depend on thread timing. OR-Tools 9.15 does not start the probing,
    // shaving, lb_tree_search and objective_lb_search workers in this mode; turning it off
    // gets them back at the price of reproducibility.
    params.interleave_search = Some(tuning.interleave_search);
    params.subsolvers = tuning.subsolvers.clone();
    params.extra_subsolvers = tuning.extra_subsolvers.clone();
    params.ignore_subsolvers = tuning.ignore_subsolvers.clone();
    if tuning.push_all_tasks_toward_start {
        params.push_all_tasks_toward_start = Some(true);
    }
    // Extra cumulative propagators: off in CP-SAT by default ("always better propagation, but
    // usually slow"). Set on the top-level parameters, so every worker runs them.
    for propagator in &tuning.cumulative_propagators {
        match propagator {
            CumulativePropagator::OverloadChecker => {
                params.use_overload_checker_in_cumulative = Some(true);
            }
            CumulativePropagator::TimetableEdgeFinding => {
                params.use_timetable_edge_finding_in_cumulative = Some(true);
            }
            CumulativePropagator::HardPrecedences => {
                // 9.15 fills the precedence list only under exploit_all_precedences, so the
                // first parameter on its own would silently do nothing.
                params.use_hard_precedences_in_cumulative = Some(true);
                params.exploit_all_precedences = Some(true);
            }
        }
    }
    if let Some(max_intervals) = tuning.timetable_edge_finding_max_intervals {
        params.max_num_intervals_for_timetable_edge_finding = Some(max_intervals);
    }

    if stop_after_first_solution {
        params.stop_after_first_solution = Some(true);
    }
    params
}

/// Entries of `map` in ascending key order.
///
/// Anything that adds variables, constraints, hints or search-strategy entries must iterate
/// maps through this: `HashMap` order is randomised per process, and a different creation
/// order gives CP-SAT a different model (different fingerprint, different schedule).
#[cfg(feature = "cpsat")]
fn sorted_by_key<K: Ord, V>(map: &HashMap<K, V>) -> Vec<(&K, &V)> {
    let mut entries: Vec<(&K, &V)> = map.iter().collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));
    entries
}

/// Where a scratch interval starts: at its value's write, or at the start of the segment
/// (a value live into a per-section solve).
#[cfg(feature = "cpsat")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ScratchFrom {
    Write,
    SegmentStart,
}

/// Where a scratch interval ends: after its value's last read, or after the segment (a value
/// live out of a per-section solve, still held after the segment's last cycle).
#[cfg(feature = "cpsat")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ScratchTo {
    LastRead,
    SegmentEnd,
}

/// A scratch interval for the cumulative, on the timeline `time` selects.
///
/// `start`, `size` and `end` describe the value on the cycle timeline as the model builds it:
/// half-open `[start, end)` with `end` = last read + 1 (`ScratchTo::LastRead`) or segment end +
/// 1 (`ScratchTo::SegmentEnd`), and `start + size == end` already constrained.
///
/// - `ScratchTime::Cycle`: that interval as is. At cycle C it counts both the values read for
///   the last time at C and the values written at C.
/// - `ScratchTime::Phase`: address assignment's same-cycle reuse (reads happen before writes in
///   a cycle, so a word read for the last time at C can take a value written at C). On the
///   doubled timeline of `address_assign::interval_span` (read phase 2C, write phase 2C + 1) a
///   value written at S and last read at L > S holds its words over phases `2S + 1 ..= 2L`.
///   The set of values holding words in the write phase of C is `{S <= C < L}`, and in the read
///   phase of C it is the write-phase set of C - 1, so the phase peak equals the peak of the
///   half-open `[S, L)` on the cycle timeline: the interval ends at the last read instead of
///   one cycle after it. A value held to the segment end keeps `[S, segment end + 1)`.
///   (Proof: verification/allocation_conjectures.py, A8-phase-interval-is-allocator-span and
///   A8-phase-peak-exhaustive.) Values read at their own write cycle (L == S) would vanish
///   here; the model's precedences never schedule a reader in its producer's cycle.
///   A value live from the segment start (`ScratchFrom::SegmentStart`) also holds its words
///   in the read phase of the first cycle, before anything is written; on this timeline the
///   read phase of cycle t is slot t - 1, so such an interval starts at -1. Without that a
///   live-in last read at cycle 0 would get the empty `[0, 0)`.
///
/// # Returns
/// The interval (optional when `presence` is given).
#[cfg(feature = "cpsat")]
fn scratch_interval(
    model: &mut CpModelBuilder,
    time: ScratchTime,
    start: LinearExpr,
    size: LinearExpr,
    end: LinearExpr,
    from: ScratchFrom,
    to: ScratchTo,
    presence: Option<BoolVar>,
) -> IntervalVar {
    let (start, size, end) = match time {
        ScratchTime::Cycle => (start, size, end),
        ScratchTime::Phase => {
            let s_off = if from == ScratchFrom::SegmentStart { -1 } else { 0 };
            let e_off = if to == ScratchTo::LastRead { -1 } else { 0 };
            (start + s_off, size + (e_off - s_off), end + e_off)
        }
    };
    match presence {
        Some(p) => model.new_optional_interval_var(start, size, end, p),
        None => model.new_interval_var(start, size, end),
    }
}

/// Readers of every defined VReg, with vector-group aliases resolved.
///
/// An op that reads a lane of a decomposed vector (read group) is a reader of that vector;
/// an op that reads the alias of an assembled vector (write group) is a reader of each lane.
/// A SelectChoice also reads its `delta_vreg`, and an op is a reader of every VReg in its
/// `keep_live` list (a liveness-only use; it is timed like the op's operand reads, which is
/// never earlier than the op's cycle).
///
/// Every op counts, including ones the greedy marked as dead stores (negative hint cycle):
/// the solver still gives them a cycle and they read their operands then. With DCE enabled
/// (the default) there are none.
///
/// # Arguments
/// * `ops` - SSA ops
/// * `vreg_def_op` - Defining op index per VReg
/// * `vector_group_lanes` - Group key/alias → lanes (None = no groups)
///
/// # Returns
/// Defined VReg → reader op indices (sorted, deduplicated). VRegs without readers are absent.
#[cfg(feature = "cpsat")]
pub(crate) fn scratch_readers_by_vreg(
    ops: &[SsaOp],
    vreg_def_op: &HashMap<VReg, usize>,
    vector_group_lanes: Option<&HashMap<VReg, Vec<VReg>>>,
) -> HashMap<VReg, Vec<usize>> {
    use crate::kernel_builder::round_budget::ScratchAliases;
    let aliases = vector_group_lanes.map_or_else(ScratchAliases::default, |groups| {
        ScratchAliases::new(groups, |v| vreg_def_op.contains_key(v))
    });
    let mut readers: HashMap<VReg, Vec<usize>> = HashMap::new();
    for (idx, op) in ops.iter().enumerate() {
        for vreg in aliases.reads(op) {
            if vreg_def_op.contains_key(&vreg) {
                readers.entry(vreg).or_default().push(idx);
            }
        }
    }
    for list in readers.values_mut() {
        list.sort_unstable();
        list.dedup();
    }
    readers
}

/// Vbroadcasts that can be eliminated when all their readers are shifted to scalar ALU.
///
/// A vbroadcast qualifies when it has exactly one producer, that producer is not a VALU op
/// (so it is a scalar the shifted readers can read directly), and every reader is shiftable.
///
/// # Returns
/// (vbroadcast index, scalar producer index, shiftable readers in index order) per
/// eliminable vbroadcast, in index order.
#[cfg(feature = "cpsat")]
fn find_eliminable_vbroadcasts(
    ops: &[SsaOp],
    op_id_to_idx: &HashMap<usize, usize>,
    shiftable_set: &HashSet<usize>,
) -> Vec<(usize, usize, Vec<usize>)> {
    let mut consumers_of: HashMap<usize, Vec<usize>> = HashMap::new();
    for (idx, op) in ops.iter().enumerate() {
        for &dep_id in &op.dep_ids {
            consumers_of.entry(dep_id).or_default().push(idx);
        }
    }
    let mut list: Vec<(usize, usize, Vec<usize>)> = Vec::new();
    for (vb_idx, op) in ops.iter().enumerate() {
        if op.engine != Engine::Valu || op.opcode != Opcode::Vbroadcast {
            continue;
        }
        let dep_id = match op.dep_ids.iter().next() {
            Some(&id) if op.dep_ids.len() == 1 => id,
            _ => continue,
        };
        let Some(&scalar_producer_idx) = op_id_to_idx.get(&dep_id) else {
            continue;
        };
        if ops[scalar_producer_idx].engine == Engine::Valu {
            continue;
        }
        let consumers = consumers_of.get(&op.id).map_or(&[] as &[usize], |v| v.as_slice());
        let shiftable_consumers: Vec<usize> = consumers
            .iter()
            .copied()
            .filter(|&c| shiftable_set.contains(&c))
            .collect();
        if shiftable_consumers.is_empty() {
            continue;
        }
        let all_consumers_shiftable = consumers.iter().all(|&c| shiftable_set.contains(&c));
        if !all_consumers_shiftable {
            continue;
        }
        list.push((vb_idx, scalar_producer_idx, shiftable_consumers));
    }
    list
}

/// Schedule SSA operations using CP-SAT solver.
///
/// This operates on SsaOps BEFORE register allocation, using only RAW dependencies.
/// This avoids false WAW/WAR dependencies introduced by register allocation.
///
/// When `enable_valu_shift` is true, the solver can decide which VALU ops to shift
/// to ALU (8 scalar ops). This is modeled using optional intervals:
/// - Each shiftable VALU op has two optional intervals: VALU variant and ALU variant
/// - Exactly one must be present (shift decision)
/// - Shifted ops consume 8 ALU slots instead of 1 VALU slot
///
/// # Arguments
/// * `ops` - SSA operations to schedule (with dep_ids computed)
/// * `config` - Solver configuration (time limit, scratch budget, etc.)
/// * `hints` - Optional warm-start and segment input; use `SsaCpsatHints::default()` for none
/// * `greedy_within_scratch_budget` - When `Some(true)`, the greedy schedule (hints) is within scratch budget;
///   when `Some(false)` or `None`, greedy may have exceeded budget (hints can be infeasible). Used only to
///   sanity-check: we panic if greedy makespan < capacity_lower_bound only when this is `Some(true)`.
///
/// # Returns
/// * `Ok(Some(SsaCpsatResult))` if a solution was found (includes shift decisions)
/// * `Ok(None)` if no solution found or feature not enabled
/// * `Err(msg)` if a solution was found but post-solve validation failed (e.g. slot limit violated)
#[cfg(feature = "cpsat")]
pub fn schedule_ssa_cpsat(
    ops: &[SsaOp],
    config: SsaCpsatConfig,
    hints: SsaCpsatHints<'_>,
    greedy_within_scratch_budget: Option<bool>,
) -> Result<Option<SsaCpsatResult>, String> {

    let SsaCpsatConfig {
        time_limit_secs,
        deterministic_time_limit,
        verbose,
        stop_after_first_solution,
        enable_valu_shift,
        scratch_budget,
        scratch_pools,
        scratch_time,
        ref model_name,
        ref machine,
        const_choice_mode,
        select_choice_mode,
        valu_shift_mode,
        enable_vbroadcast_elimination,
        force_zero_gap,
        asap_alap_tightening,
        enable_scratch_tightening,
        symmetry_precedence_integrated,
        energetic_alap_pulls,
        energetic_bounded_push_k,
        precedence_neutral_horizon,
        debug_fix_precedence_bounds,
        ref disable_optional_constraint_groups,
        hint_select_choice_vselect_by_default,
        ref tuning,
    } = config;

    let valu_cap: i64 = machine.valu_slots as i64;
    let alu_cap: i64 = machine.alu_slots as i64;
    let load_cap: i64 = machine.load_slots as i64;
    let store_cap: i64 = machine.store_slots as i64;
    let flow_cap: i64 = machine.flow_slots as i64;
    let greedy_hints = hints.greedy_hints;
    let shift_hints = hints.shift_hints;
    let forced_shifts = hints.forced_shifts;
    let const_add_imm_hints = hints.const_add_imm_hints;
    let select_choice_hints = hints.select_choice_hints;
    let eliminate_vbroadcast_hints = hints.eliminate_vbroadcast_hints;
    let live_out_vregs = hints.live_out_vregs;
    let live_through_vregs = hints.live_through_vregs;
    // An incoming value read here and again later is already charged for the entire
    // section. Use only the other incoming values for last-reader intervals and hints.
    // Preserve Some(empty) because it still identifies a section solve.
    let entry_reads = hints.initial_live.map(|inputs| {
        inputs.iter().copied()
            .filter(|(vreg, _)| !live_through_vregs.is_some_and(|values| values.contains(vreg)))
            .collect::<Vec<_>>()
    });
    let initial_live = entry_reads.as_deref();
    let cse_groups = hints.cse_groups;
    let alu_lane_hints = hints.alu_lane_hints;
    let window = hints.window;
    let start_time = std::time::Instant::now();
    let n = ops.len();

    if n == 0 {
        return Ok(Some(SsaCpsatResult {
            cycle_assignments: vec![],
            makespan: 0,
            status: CpsatStatus::Optimal,
            solve_time_secs: 0.0,
            gap: 0.0,
            shifted_ops: HashSet::new(),
            shifted_alu_cycles: HashMap::new(),
            const_choice_encoding: HashMap::new(),
            select_choice_encoding: HashMap::new(),
            empty_cycle_count: 0,
            makespan_driven_by_capacity: false,
            lb_cycles: Some(0),
            eliminated_vbroadcast_ops: HashSet::new(),
            eliminated_vb_to_scalar_producer: HashMap::new(),
            cse_group_results: vec![],
            schedule_constraints: vec![],
        }));
    }

    // 0. Identify shiftable VALU ops (needed for precedence offsets in ASAP/ALAP)
    let shiftable_valu: Vec<usize> = if enable_valu_shift {
        ops.iter()
            .enumerate()
            .filter(|(_, op)| op.engine == Engine::Valu && op.opcode.can_shift_to_alu())
            .map(|(idx, _)| idx)
            .collect()
    } else {
        Vec::new()
    };
    let shiftable_set: HashSet<usize> = shiftable_valu.iter().copied().collect();

    // The greedy schedule bounds the search (horizon, ALAP warm start, greedy-cap tightening,
    // domain widening) only if it is a feasible schedule of *this* model. With a forced
    // encoding the greedy did not use, its cycles prove nothing and bounds derived from them
    // can exclude every feasible schedule. It is still passed to the solver as a hint.
    let greedy_matches_forced_modes = greedy_schedule_matches_forced_modes(
        ops,
        &shiftable_valu,
        valu_shift_mode,
        const_choice_mode,
        select_choice_mode,
        shift_hints,
        const_add_imm_hints,
        select_choice_hints,
    );
    if verbose && greedy_hints.is_some() && !greedy_matches_forced_modes {
        eprintln!(
            "  Greedy schedule uses encodings that the forced modes exclude; not using it to bound domains or horizon"
        );
    }
    let bound_hints = greedy_hints.filter(|_| greedy_matches_forced_modes);

    // 1. Compute ASAP/ALAP bounds using RAW-only dependencies (with precedence offsets)
    let asap = compute_ssa_asap(ops, &shiftable_set);
    let critical_path_len = asap.values().max().copied().unwrap_or(0) + 1;

    // Compute max-latency critical path for horizon sizing.
    // ALAP uses max offsets (e.g. 8 for shiftable VALU), so the horizon must be large enough
    // that backward propagation from horizon-1 doesn't produce negative ALAP values.
    let max_latency_critical_path = compute_max_latency_critical_path(ops, &shiftable_set);

    // Use a reasonable horizon: when greedy hints exist, greedy_max is a proven upper bound
    // on makespan, so we only need a small buffer beyond it. Without greedy, use conservative 2x.
    // precedence_neutral_horizon: always use 2x formula, ignoring greedy (for ablation/debugging).
    let greedy_max = bound_hints
        .map(|h| h.iter().max().copied().unwrap_or(0) + 1)
        .unwrap_or(0);
    let mut horizon = if precedence_neutral_horizon {
        (critical_path_len * 2)
            .max(max_latency_critical_path * 2)
            .max(n as i32) as i64
    } else if greedy_max > 0 {
        // Greedy makespan is a proven upper bound — optimal is at most greedy_max.
        // Add small buffer (+20) for solver flexibility, but keep max_latency_critical_path
        // guard so ALAP backward propagation has enough room.
        // Both guards are bounds that need no greedy schedule; they are far above the greedy
        // makespan and only widen domains. ALAP is computed with minimum latencies, so with a
        // feasible greedy schedule every op's ALAP is at least its greedy cycle without them.
        // tuning.horizon_at_least_max_latency_path / horizon_at_least_num_ops = false leave
        // them out.
        let greedy_horizon = if tuning.horizon_at_least_max_latency_path {
            (greedy_max as i64 + 20).max(max_latency_critical_path as i64)
        } else {
            greedy_max as i64 + 20
        };
        if tuning.horizon_at_least_num_ops {
            greedy_horizon.max(n as i64)
        } else {
            greedy_horizon
        }
    } else {
        (critical_path_len * 2)
            .max(max_latency_critical_path * 2)
            .max(n as i32) as i64
    };

    // Ensure horizon >= capacity_lower_bound so max_cycle/makespan domains are non-empty.
    // With toy machines (e.g. 1 VALU slot), capacity_lower_bound can exceed the formula above.
    {
        let valu_count = ops.iter().filter(|op| op.engine == Engine::Valu).count() as i64;
        let alu_count = ops.iter().filter(|op| op.engine == Engine::Alu).count() as i64;
        let load_count = ops.iter().filter(|op| op.engine == Engine::Load).count() as i64;
        let store_count = ops.iter().filter(|op| op.engine == Engine::Store).count() as i64;
        let flow_base = ops.iter().filter(|op| op.engine == Engine::Flow).count() as i64;
        let shiftable_count = shiftable_set.len() as i64;
        let const_choice_count = ops.iter().filter(|op| op.opcode == Opcode::ConstChoice).count() as i64;
        let select_choice_count = ops.iter().filter(|op| op.opcode == Opcode::SelectChoice).count() as i64;
        let cap_result = compute_capacity_lower_bound_global(
            valu_count,
            alu_count,
            load_count,
            store_count,
            flow_base,
            shiftable_count,
            const_choice_count,
            select_choice_count,
            enable_valu_shift,
            valu_cap,
            alu_cap,
            load_cap,
            store_cap,
            flow_cap,
            0i64,
            cse_groups,
        );
        horizon = horizon.max(cap_result.capacity_lower_bound);
    }

    let alap = compute_ssa_alap(ops, horizon as i32, &shiftable_set);

    // 1a. Warm-start ALAP from greedy schedule: intersect each op's ALAP with greedy_cycle + slack.
    // This tightens domains before energetic reasoning for compounding benefit. It is a search
    // restriction rather than a valid bound: only the greedy makespan bounds the optimum, not
    // each op's greedy cycle, so schedules that move an op more than `slack` cycles later are
    // cut off. tuning.greedy_alap_slack sets the slack (default 15); None leaves ALAP alone.
    let alap = if let (Some(hints), Some(slack)) = (bound_hints, tuning.greedy_alap_slack) {
        let mut tighter_alap = alap;
        let mut tightened_count = 0usize;
        for (idx, &hint_cycle) in hints.iter().enumerate() {
            if idx >= ops.len() || hint_cycle < 0 {
                continue;
            }
            let op_id = ops[idx].id;
            let op_asap = *asap.get(&op_id).unwrap_or(&0);
            if let Some(current) = tighter_alap.get_mut(&op_id) {
                let new_alap = hint_cycle.saturating_add(slack).min(*current);
                // Never tighten below ASAP
                if new_alap >= op_asap && new_alap < *current {
                    *current = new_alap;
                    tightened_count += 1;
                }
            }
        }
        if verbose && tightened_count > 0 {
            eprintln!(
                "  Greedy-based ALAP tightening: {} ops tightened (slack={})",
                tightened_count, slack
            );
        }
        tighter_alap
    } else {
        alap
    };

    // 1b. Tighten ASAP/ALAP bounds if enabled. Build canonical symmetry ordering once when
    // symmetry-aware or energetic tightening is used; use it for both passes and model constraints.
    let mut current_asap = asap.clone();
    let mut current_alap = alap.clone();

    // With checked symmetry rows the emission-declared ordering is not trusted, here either.
    let symmetry_ordering = if asap_alap_tightening.use_symmetry_ordering()
        && tuning.symmetry_rows != SymmetryRows::Checked
    {
        Some(SymmetryOrdering::from_bounds(
            ops,
            &current_asap,
            &current_alap,
            bound_hints,
            symmetry_precedence_integrated,
        ))
    } else {
        None
    };

    if let Some(ref ordering) = symmetry_ordering {
        let (sym_asap, sym_alap) = tighten_asap_by_symmetry_ordering(
            ops,
            &current_asap,
            &current_alap,
            ordering,
            valu_cap,
            alu_cap,
            load_cap,
            store_cap,
            flow_cap,
            Some(&shiftable_set),
            verbose,
        );
        current_asap = sym_asap;
        current_alap = sym_alap;
    }

    // Energetic slot loop: symmetry-aware undercounting (sound, never over-tightens).
    if asap_alap_tightening.use_energetic() {
        let (slot_asap, slot_alap) = tighten_asap_alap_by_slots(
            ops,
            &current_asap,
            &current_alap,
            horizon as i32,
            valu_cap,
            alu_cap,
            load_cap,
            store_cap,
            flow_cap,
            verbose,
            Some(&shiftable_set),
            bound_hints,
            symmetry_ordering.as_ref(),
            energetic_alap_pulls,
            energetic_bounded_push_k,
        )
        .unwrap_or_else(|| (current_asap.clone(), current_alap.clone()));
        current_asap = slot_asap;
        current_alap = slot_alap;
    }

    // scratch_budget == 0 means the model has no scratch constraint: nothing to tighten by.
    if enable_scratch_tightening && scratch_budget > 0 {
        let (scratch_asap, scratch_alap) = tighten_asap_alap_by_scratch(
            ops,
            &current_asap,
            &current_alap,
            horizon as i32,
            scratch_budget as i32,
            verbose,
        )
        .unwrap_or_else(|| (current_asap.clone(), current_alap.clone()));
        current_asap = scratch_asap;
        current_alap = scratch_alap;
    }

    let mut final_asap = current_asap;
    let mut final_alap = current_alap;

    // 2. Verbose VALU shift summary
    if verbose {
        let total_valu_ops: usize = ops.iter().filter(|op| op.engine == Engine::Valu).count();
        let non_shiftable_valu_count = total_valu_ops - shiftable_valu.len();
        let vbroadcast_count = ops
            .iter()
            .filter(|op| op.engine == Engine::Valu && op.opcode == Opcode::Vbroadcast)
            .count();
        let multiply_add_count = ops
            .iter()
            .filter(|op| op.engine == Engine::Valu && op.opcode == Opcode::MultiplyAdd)
            .count();
        let select_choice_count = ops
            .iter()
            .filter(|op| op.opcode == Opcode::SelectChoice)
            .count();
        eprintln!(
            "SSA CP-SAT: {} ops, critical path {}, horizon {}, scratch_budget={}",
            n, critical_path_len, horizon, scratch_budget
        );
        eprintln!(
            "  VALU ops: {} total ({} shiftable, {} non-shiftable: {} vbroadcast, {} multiply_add)",
            total_valu_ops,
            shiftable_valu.len(),
            non_shiftable_valu_count,
            vbroadcast_count,
            multiply_add_count
        );
        if select_choice_count > 0 {
            eprintln!(
                "  SelectChoice ops: {} (can expand to FMA/VALU or vselect/Flow)",
                select_choice_count
            );
        }

        // Log greedy hints that fall outside [final_asap, final_alap] domains
        if let Some(hints) = bound_hints {
            let mut rejected_count = 0usize;
            for (idx, &hint_cycle) in hints.iter().enumerate() {
                if idx >= ops.len() {
                    break;
                }
                let lb = *final_asap.get(&ops[idx].id).unwrap_or(&0);
                let ub = *final_alap.get(&ops[idx].id).unwrap_or(&(horizon as i32 - 1));
                if hint_cycle < lb || hint_cycle > ub {
                    rejected_count += 1;
                }
            }
            if rejected_count > 0 {
                eprintln!(
                    "  Greedy hints outside [asap, alap] domain: {} of {} (will be clamped/ignored)",
                    rejected_count, hints.len()
                );
            }
        }

        // Run precedence diagnostics
        let (viol_count, viol_samples, empty_count) =
            diagnose_precedence_bounds(ops, &final_asap, &final_alap, &shiftable_set);
        if viol_count > 0 || empty_count > 0 {
            eprintln!(
                "  Precedence diagnostics: {} edge violations, {} empty domains",
                viol_count, empty_count
            );
            for sample in &viol_samples {
                eprintln!("    {}", sample);
            }
        }
    }

    // 2b. Precedence fixpoint repair: after slot/scratch tightening, re-propagate precedence
    // constraints to restore consistency. Tightening is resource-aware but precedence-unaware,
    // so it can narrow a predecessor's ALAP past what its consumers require.
    {
        let (fixed_asap, fixed_alap) =
            fix_precedence_bounds_fixpoint(ops, &final_asap, &final_alap, &shiftable_set);
        if verbose {
            let changed_asap = ops.iter().filter(|op| {
                fixed_asap.get(&op.id) != final_asap.get(&op.id)
            }).count();
            let changed_alap = ops.iter().filter(|op| {
                fixed_alap.get(&op.id) != final_alap.get(&op.id)
            }).count();
            if changed_asap > 0 || changed_alap > 0 {
                eprintln!(
                    "  Fixpoint repair: {} ASAP changes, {} ALAP changes",
                    changed_asap, changed_alap
                );
            }
        }
        final_asap = fixed_asap;
        final_alap = fixed_alap;
    }

    // 2c. Guarantee greedy hint feasibility: widen final domains to include greedy values.
    // The greedy schedule is a proven feasible solution (satisfies precedence, slot capacity,
    // scratch budget). If domain tightening excluded any greedy values, the tightening was
    // overly conservative. Widening ensures the hint is always within domain, so CP-SAT
    // gets a feasible warm start instead of searching from scratch.
    if let Some(hints) = bound_hints {
        let mut widened_count = 0usize;
        for (idx, &hint_cycle) in hints.iter().enumerate() {
            if idx >= ops.len() || hint_cycle < 0 {
                continue;
            }
            let op_id = ops[idx].id;
            if let Some(a) = final_asap.get_mut(&op_id) {
                if hint_cycle < *a {
                    *a = hint_cycle;
                    widened_count += 1;
                }
            }
            if let Some(a) = final_alap.get_mut(&op_id) {
                if hint_cycle > *a {
                    *a = hint_cycle;
                    widened_count += 1;
                }
            }
        }
        if verbose && widened_count > 0 {
            eprintln!(
                "  Greedy hint domain widening: {} bound(s) widened to include greedy values",
                widened_count
            );
        }
    }

    // 2d. Checked symmetry rows: find the orbits of identical vector groups and give
    // corresponding ops of an orbit one domain (group_symmetry.rs), so that every swap within
    // an orbit is an automorphism of the model.
    // Section liveness, per-op shift restrictions, window pins, and dead-store hints can distinguish
    // otherwise identical groups. The structural check does not model those restrictions.
    let group_orbits = if tuning.symmetry_rows == SymmetryRows::Checked
        && window.is_none()
        && !forced_shifts.is_some_and(|ids| !ids.is_empty())
        && initial_live.is_none()
        && !live_out_vregs.is_some_and(|vregs| !vregs.is_empty())
        && !live_through_vregs.is_some_and(|vregs| !vregs.is_empty())
        && !greedy_hints.is_some_and(|cycles| cycles.iter().any(|&cycle| cycle < 0))
    {
        let orbits = super::group_symmetry::group_orbits(ops, hints.vector_group_lanes, cse_groups.unwrap_or(&[]));
        super::group_symmetry::symmetric_domains(ops, &orbits, &mut [&mut final_asap, &mut final_alap]);
        Some(orbits)
    } else {
        None
    };
    let (asap, alap) = match &group_orbits {
        Some(orbits) => {
            let (mut asap, mut alap) = (asap, alap);
            super::group_symmetry::symmetric_domains(ops, orbits, &mut [&mut asap, &mut alap]);
            (asap, alap)
        }
        None => (asap, alap),
    };

    // 2d. Window probe: the reference must stay inside every domain, and so must each cycle
    // the block after the window can move to. 2c covers ops with a hint; dead stores (hint
    // -1) are pinned at the reference too but got no widening, and tightening computed on a
    // reduced op list need not cover the moved block.
    if let Some(window) = window {
        let mut widened = 0usize;
        for (idx, op) in ops.iter().enumerate() {
            let Some(&cycle) = window.reference.cycles.get(idx) else { continue };
            if cycle < 0 {
                continue;
            }
            let lowest = if window.region(idx, op) == super::window::WindowRegion::After {
                cycle - window.len.max(0)
            } else {
                cycle
            };
            if let Some(a) = final_asap.get_mut(&op.id) {
                if lowest < *a {
                    *a = lowest.max(0);
                    widened += 1;
                }
            }
            if let Some(a) = final_alap.get_mut(&op.id) {
                if cycle > *a {
                    *a = cycle;
                    widened += 1;
                }
            }
        }
        if widened > 0 {
            eprintln!("[window probe]   widened {} domain bound(s) to cover the reference", widened);
        }
    }

    // A window can also supply lane cycles from a previous feasible CP-SAT solution.
    // Lane cycles of a hinted split shift: only for ops hinted as shifted, with one cycle per
    // lane, and only if every lane lies in the lane variables' domain ([asap, alap] above, see
    // op_times::create_op_variables). Otherwise the op is treated as a whole shift everywhere (capacity check, lane,
    // size and start hints), so what is hinted is exactly what the check counted.
    let hinted_lanes = |op_id: usize| -> Option<&Vec<i32>> {
        if !shift_hints.is_some_and(|ids| ids.contains(&op_id)) {
            return None;
        }
        let lo = *asap.get(&op_id).unwrap_or(&0);
        let hi = *alap.get(&op_id).unwrap_or(&((horizon - 1) as i32));
        hints
            .shifted_lane_cycle_hints
            .and_then(|m| m.get(&op_id))
            .filter(|lanes| lanes.len() == VLEN && lanes.iter().all(|&c| c >= lo && c <= hi))
            .or_else(|| alu_lane_hints.and_then(|m| m.get(&op_id)).filter(|lanes| lanes.len() == VLEN))
    };


    // 3. Build the CP model
    let mut model = CpModelBuilder::default();
    if let Some(name) = &model_name {
        model.set_model_name(name);
    }
    if verbose && model_name.is_some() {
        eprintln!("  Model name: {}", model_name.as_deref().unwrap_or(""));
    }

    // Build op_id to index mapping
    let op_id_to_idx: HashMap<usize, usize> = ops
        .iter()
        .enumerate()
        .map(|(idx, op)| (op.id, idx))
        .collect();

    // SelectChoice ops with sub_temp_vreg (FMA path without delta): result is visible at cycle+2, not cycle+1.
    let select_choice_with_fma_delay: HashSet<usize> = ops
        .iter()
        .enumerate()
        .filter(|(_, op)| op.opcode == Opcode::SelectChoice && op.sub_temp_vreg.is_some())
        .map(|(i, _)| i)
        .collect();
    let const_choice_ops: Vec<usize> = ops
        .iter()
        .enumerate()
        .filter(|(_, op)| op.opcode == Opcode::ConstChoice)
        .map(|(idx, _)| idx)
        .collect();
    let select_choice_ops: Vec<usize> = ops
        .iter()
        .enumerate()
        .filter(|(_, op)| op.opcode == Opcode::SelectChoice)
        .map(|(idx, _)| idx)
        .collect();
    let eliminable_vbs: Vec<(usize, usize, Vec<usize>)> =
        if enable_valu_shift && enable_vbroadcast_elimination && !shiftable_valu.is_empty() {
            find_eliminable_vbroadcasts(ops, &op_id_to_idx, &shiftable_set)
        } else {
            Vec::new()
        };

    // 4. Create every per-op variable: issue cycle (domain [asap, alap]), shift literal and
    // lanes, vbroadcast elimination, ConstChoice and SelectChoice literals, multiply_add
    // cycles, CSE shared subtracts. Constraint generators below get times only through `times`
    // (op_times.rs), so each time (start, visible, last read, finish) has one definition.
    let op_vars = create_op_variables(
        &mut model,
        ops,
        &TimeDomains { final_asap: &final_asap, final_alap: &final_alap, asap: &asap, alap: &alap, horizon },
        &OpKinds {
            shiftable_valu: &shiftable_valu,
            eliminable_vbs: &eliminable_vbs,
            const_choice_ops: &const_choice_ops,
            const_choice_mode,
            select_choice_ops: &select_choice_ops,
            select_choice_mode,
            cse_groups,
        },
        verbose,
    );
    let times = &op_vars.times;
    let cse_subs = &op_vars.cse_subs;
    // Decision literals by op index (shift, SelectChoice FMA path chosen).
    let shift_vars: HashMap<usize, BoolVar> =
        times.iter().enumerate().filter_map(|(idx, t)| t.shift().map(|b| (idx, b))).collect();
    let select_choice_vars: HashMap<usize, BoolVar> =
        times.iter().enumerate().filter_map(|(idx, t)| t.fma_chosen().map(|b| (idx, b))).collect();

    // 5. Add precedence constraints from RAW dependencies.
    // These ensure every consumer (including multiply_add) is scheduled at least one cycle after
    // its producers (e.g. offset producer). Results are visible next cycle; without this, an FMA
    // could run in the same cycle as its offset producer and read before the producer writes.
    // Note: consumers of shiftable VALU ops are ordered after the producer's completion
    // cycle in 5c below.
    for (idx, op) in ops.iter().enumerate() {
        for &dep_id in &op.dep_ids {
            if let Some(&dep_idx) = op_id_to_idx.get(&dep_id) {
                // Skip if dep_idx is a shiftable VALU op - handled in 5c (visible = completion + 1)
                if shiftable_set.contains(&dep_idx) {
                    continue;
                }
                // Skip if dep is SelectChoice with sub_temp (FMA no-delta): consumer precedence added in 5d
                if select_choice_with_fma_delay.contains(&dep_idx) {
                    continue;
                }
                // cycle[idx] >= cycle[dep_idx] + 1
                let c = model.add_ge(times[idx].issue(), times[dep_idx].visible());
                model.set_constraint_name(c, &format!("prec_raw_{}_{}", idx, dep_idx));
            }
        }
    }

    // 5-mem. Memory edges (`mem_deps`, sorted per op): the target's memory access is at least
    // the edge latency after the source's. WAR has latency 0 (same cycle allowed: within a
    // cycle the simulator reads memory before it writes); RAW and WAW have 1. Empty for every
    // kernel emitted today, so this adds no constraint to their models.
    for (idx, op) in ops.iter().enumerate() {
        for &(src_id, kind) in &op.mem_deps {
            if let Some(&src_idx) = op_id_to_idx.get(&src_id) {
                let c = model.add_ge(
                    times[idx].memory_access(),
                    times[src_idx].memory_access() + kind.latency() as i64,
                );
                model.set_constraint_name(c, &format!("prec_mem_{}_{}_{}", kind.as_str(), idx, src_idx));
            }
        }
    }

    // 5b. Symmetry breaking (ordering groups): use the canonical ordering built earlier.
    // Validate that final bounds are consistent with the ordering, then add cycle[i] <= cycle[j].
    if tuning.symmetry_rows == SymmetryRows::Declared {
        let ordering_pairs: Vec<(usize, usize)> = if let Some(ref ord) = symmetry_ordering {
            ord.validate_bounds(ops, &final_asap, &final_alap, greedy_hints);
            ord.ordering_pairs()
        } else {
            super::bounds::build_ordering_pairs_from_bounds(ops, &final_asap, &final_alap, greedy_hints)
        };
        let n_ordering = ordering_pairs.len();
        let order_source = if greedy_hints.is_some() { "greedy" } else { "ALAP" };
        for (i, j) in &ordering_pairs {
            let c = model.add_le(times[*i].issue(), times[*j].issue());
            model.set_constraint_name(c, &format!("sym_ord_{}_{}", i, j));
        }
        if verbose && n_ordering > 0 {
            eprintln!(
                "  Symmetry breaking (ordering): {} constraints (order from {})",
                n_ordering, order_source
            );
        }
    }

    // 5c. Symmetry breaking (lexicographic streams): for each stream_key, items ordered by item_key
    // get first_op[item_i] <= first_op[item_{i+1}]. When greedy hints are available, order items
    // by greedy cycle of their first op so constraints match the hint. Verify domain consistency.
    if tuning.symmetry_rows == SymmetryRows::Declared {
        use std::collections::{BTreeMap, HashMap};
        let mut streams: HashMap<Vec<i64>, BTreeMap<i64, Vec<usize>>> = HashMap::new();
        for (idx, op) in ops.iter().enumerate() {
            let Some(ref hint) = op.symmetry_hint else {
                continue;
            };
            let Some((ref stream_key, item_key)) = hint.lexicographic_stream else {
                continue;
            };
            streams
                .entry(stream_key.clone())
                .or_default()
                .entry(item_key)
                .or_default()
                .push(idx);
        }
        let n_streams = streams.len();
        let mut lex_constraints = 0usize;
        // Sorted: HashMap iteration order is randomised per process, and constraint order
        // changes the model CP-SAT sees (and so the schedule it finds).
        let mut stream_keys: Vec<&Vec<i64>> = streams.keys().collect();
        stream_keys.sort();
        for stream_key in stream_keys {
            let by_item = &streams[stream_key];
            let mut ordered: Vec<Vec<usize>> = by_item.values().cloned().collect();
            if let Some(hints) = greedy_hints {
                // Sort items by greedy cycle of their first op so lex constraints match the hint.
                // Tiebreaker: ALAP of first op for determinism and domain consistency.
                ordered.sort_by_key(|group| {
                    let first = group[0];
                    let greedy = hints.get(first).copied().unwrap_or(0);
                    let alap_val = *final_alap.get(&ops[first].id).unwrap_or(&0);
                    (greedy, alap_val)
                });
            }
            // else: already ordered by item_key (BTreeMap preserves key order)
            for w in ordered.windows(2) {
                let first_op_cur = w[0][0];
                let first_op_next = w[1][0];
                // Verify domain consistency
                let asap_cur = *final_asap.get(&ops[first_op_cur].id).unwrap_or(&0);
                let alap_next = *final_alap.get(&ops[first_op_next].id).unwrap_or(&0);
                if asap_cur > alap_next {
                    panic!(
                        "Lex stream ordering inconsistency: stream {:?} requires cycle[{}] <= cycle[{}], \
                         but ASAP[{}]={} > ALAP[{}]={}. Greedy hints: cur={}, next={}.",
                        stream_key, first_op_cur, first_op_next,
                        first_op_cur, asap_cur, first_op_next, alap_next,
                        greedy_hints.and_then(|h| h.get(first_op_cur).copied()).unwrap_or(-1),
                        greedy_hints.and_then(|h| h.get(first_op_next).copied()).unwrap_or(-1),
                    );
                }
                let c = model.add_le(times[first_op_cur].issue(), times[first_op_next].issue());
                model.set_constraint_name(c, &format!("sym_lex_{}_{}", first_op_cur, first_op_next));
                lex_constraints += 1;
            }
        }
        if verbose && lex_constraints > 0 {
            let lex_source = if greedy_hints.is_some() { "greedy" } else { "item_key" };
            eprintln!(
                "  Symmetry breaking (lex streams): {} streams, {} constraints (order from {})",
                n_streams, lex_constraints, lex_source
            );
        }
    }

    // 5e. Checked symmetry rows: in each orbit of identical groups, the anchor ops' issue
    // cycles in a chain, groups ordered by the anchor's greedy cycle so the hint satisfies it.
    if let Some(orbits) = &group_orbits {
        let mut rows = 0usize;
        for o in 0..orbits.orbits.len() {
            let Some(anchor) = super::group_symmetry::anchor_position(ops, orbits, o, &shiftable_set) else {
                continue;
            };
            let mut members: Vec<(i32, usize, usize)> = orbits.orbits[o]
                .iter()
                .map(|&g| {
                    let idx = orbits.groups[g][anchor];
                    (greedy_hints.and_then(|h| h.get(idx).copied()).unwrap_or(0), g, idx)
                })
                .collect();
            members.sort();
            for w in members.windows(2) {
                let c = model.add_le(times[w[0].2].issue(), times[w[1].2].issue());
                model.set_constraint_name(c, &format!("group_sym_{}_{}_{}", o, w[0].1, w[1].1));
                rows += 1;
            }
        }
        // Declared rows, classified against the orbits (for the record; none are added).
        let declared = super::bounds::build_ordering_pairs_from_bounds(ops, &final_asap, &final_alap, greedy_hints);
        let (justified, same_group, other) = super::group_symmetry::classify_pairs(orbits, &declared);
        let sizes: Vec<usize> = orbits.orbits.iter().map(Vec::len).collect();
        eprintln!(
            "  Group symmetry: {} groups, orbits of sizes {:?}, {} chain rows; declared ordering pairs: {} justified by a group swap, {} within one group, {} other",
            orbits.groups.len(),
            sizes,
            rows,
            justified,
            same_group,
            other
        );
        for why in &orbits.rejections {
            eprintln!("    not identical: {}", why);
        }
    }

    // 6. Shift decisions and optional/required intervals for shiftable VALU ops
    // (variables were created with the others in section 4)
    let mut valu_optional_intervals: HashMap<usize, IntervalVar> = HashMap::new();
    // 8 intervals per shifted VALU op (one per ALU op), present when shifted
    let mut alu_optional_intervals: HashMap<usize, Vec<IntervalVar>> = HashMap::new();

    for &idx in &shiftable_valu {
        let shift_var = times[idx].shift().expect("shiftable op has a shift literal");

        match valu_shift_mode {
            ValuShiftMode::ForceValu => {
                let c = model.add_eq(LinearExpr::from(shift_var.clone()), 0);
                model.set_constraint_name(c, &format!("shift_fix_{}_0", idx));
            }
            ValuShiftMode::ForceAlu => {
                let c = model.add_eq(LinearExpr::from(shift_var.clone()), 1);
                model.set_constraint_name(c, &format!("shift_fix_{}_1", idx));
            }
            ValuShiftMode::Optimize => {
                if let Some(forced) = forced_shifts {
                    if forced.contains(&ops[idx].id) {
                        let c = model.add_eq(LinearExpr::from(shift_var.clone()), 1);
                        model.set_constraint_name(c, &format!("shift_fix_{}_1", idx));
                    }
                }
                // Shift hints are added in section 11 with all other hints (avoids duplicate hint for same var)
            }
        }

        let valu_interval =
            model.new_optional_fixed_size_interval_var(times[idx].issue(), 1, shift_var.not());
        valu_optional_intervals.insert(idx, valu_interval);

        let alu_intervals: Vec<IntervalVar> = times[idx]
            .lanes()
            .iter()
            .map(|&lane| model.new_optional_fixed_size_interval_var(lane, 1, shift_var.clone()))
            .collect();
        alu_optional_intervals.insert(idx, alu_intervals);

        // completion >= issue cycle and every lane (op_times.rs states the mode invariants)
        // and, with tuning.lane_model, the lane order and the pinned lanes of an unshifted op
        times[idx].add_shift_invariants(&mut model, idx, valu_shift_mode, tuning.lane_model);
    }

    // 5a. Element-wise dependency map for shifted VALU -> shifted VALU: op lane i depends on dep lane j.
    // Default: same lane (i -> i). Used in 5b so that when both op and dep are shifted we use
    // alu_cycle[op][i] >= alu_cycle[dep][j]+1 instead of completion_cycle[dep].
    let element_dep_map: HashMap<(usize, usize), Vec<usize>> = {
        let mut m = HashMap::new();
        for &idx in &shiftable_valu {
            for &dep_id in &ops[idx].dep_ids {
                if let Some(&dep_idx) = op_id_to_idx.get(&dep_id) {
                    if shiftable_set.contains(&dep_idx) {
                        // Standard: lane i depends on dep's lane i. (Broadcast deps are not shiftable.)
                        m.insert((idx, dep_idx), (0..VLEN).collect::<Vec<_>>());
                    }
                }
            }
        }
        m
    };

    // 5b. Precedence for the 8 ALU ops of shifted VALU. When dep is shifted use element-wise
    // alu_cycle[op][i] >= alu_cycle[dep][j]+1 (j from element_dep_map); when dep not shifted use
    // its issue cycle + 1 (OpTimes::lane_ready).
    for &idx in &shiftable_valu {
        for (lane_i, &alu_cycle) in times[idx].lanes().iter().enumerate() {
            for &dep_id in &ops[idx].dep_ids {
                if let Some(&dep_idx) = op_id_to_idx.get(&dep_id) {
                    let j = element_dep_map
                        .get(&(idx, dep_idx))
                        .map(|v| v[lane_i])
                        .unwrap_or(lane_i);
                    // Shiftable dep: lane j + 1 when shifted ("on"), its cycle + 1 when not
                    // ("off"); other deps: cycle + 1 ("valu"). With pinned lanes a shiftable
                    // dep gives one unconditional row (no label).
                    for ready in times[dep_idx].lane_ready(j, tuning.lane_model.pinned()) {
                        let c = model.add_ge(alu_cycle, ready.value);
                        if let Some(when) = ready.when {
                            model.only_enforce_if(c, [when]);
                        }
                        let name = format!("shift_alu_prec_{}_{}_dep_{}", idx, lane_i, dep_idx);
                        let name = if ready.label.is_empty() { name } else { format!("{}_{}", name, ready.label) };
                        model.set_constraint_name(c, &name);
                    }
                }
            }
        }
    }

    // 5c. Add precedence constraints for consumers of shifted VALU ops
    // Consumers must wait for completion_cycle (when all 8 ALU ops have completed)
    for (idx, op) in ops.iter().enumerate() {
        for &dep_id in &op.dep_ids {
            if let Some(&dep_idx) = op_id_to_idx.get(&dep_id) {
                if shiftable_set.contains(&dep_idx) {
                    // cycle[idx] >= completion_cycle[dep_idx] + 1
                    // This ensures consumer waits for all 8 ALU ops to complete
                    let c = model.add_ge(times[idx].issue(), times[dep_idx].visible());
                    model.set_constraint_name(c, &format!("consumer_after_completion_{}_{}", idx, dep_idx));
                }
            }
        }
    }

    // 5d. Eliminable vbroadcasts: optional VALU interval (presence = !eliminate_vb), precedence when eliminated
    // (the list and the elimination literals were built with the other variables in section 4)
    let (eliminate_vb_vars, valu_optional_vb_intervals): (HashMap<usize, BoolVar>, HashMap<usize, IntervalVar>) = {
        let mut vars: HashMap<usize, BoolVar> = HashMap::new();
        let mut intervals: HashMap<usize, IntervalVar> = HashMap::new();
        for (vb_idx, scalar_producer_idx, shiftable_consumers) in &eliminable_vbs {
            let (vb_idx, scalar_producer_idx) = (*vb_idx, *scalar_producer_idx);
            let op = &ops[vb_idx];
            let eliminate_var = times[vb_idx].eliminate().expect("eliminable vbroadcast has a literal");
            vars.insert(vb_idx, eliminate_var);
            // eliminate_var = AND(shift[C] for C in shiftable_consumers): encode as
            // (1) eliminate_var <= shift_c for each c, (2) eliminate_var >= sum(shift_c) - (k-1)
            for &c in shiftable_consumers {
                let shift_c = *shift_vars.get(&c).unwrap();
                let ct = model.add_ge(LinearExpr::from(shift_c), LinearExpr::from(eliminate_var.clone()));
                model.set_constraint_name(ct, &format!("vb_elim_{}_shift_ge_c{}", vb_idx, c));
            }
            let k = shiftable_consumers.len() as i64;
            let first_shift = *shift_vars.get(&shiftable_consumers[0]).unwrap();
            let sum_shifts: LinearExpr = shiftable_consumers[1..]
                .iter()
                .fold(LinearExpr::from(first_shift), |acc, &c| {
                    acc + LinearExpr::from(*shift_vars.get(&c).unwrap())
                });
            let ct = model.add_ge(
                LinearExpr::from(eliminate_var.clone()),
                sum_shifts - (k - 1),
            );
            model.set_constraint_name(ct, &format!("vb_elim_{}_sum_ge", vb_idx));
            let vb_interval = model.new_optional_fixed_size_interval_var(
                times[vb_idx].issue(),
                1,
                eliminate_var.not(),
            );
            intervals.insert(vb_idx, vb_interval);
            if let Some(hints) = eliminate_vbroadcast_hints {
                if hints.contains(&op.id) {
                    model.add_hint(eliminate_var, 1);
                } else {
                    model.add_hint(eliminate_var, 0);
                }
            } else {
                model.add_hint(eliminate_var, 0);
            }
            // Eliminated: the lanes read the scalar producer; kept: they read the vbroadcast.
            for &c in shiftable_consumers {
                for (lane_i, &alu_cycle) in times[c].lanes().iter().enumerate() {
                    let c_elim = model.add_ge(alu_cycle, times[scalar_producer_idx].visible());
                    model.only_enforce_if(c_elim, [eliminate_var.clone()]);
                    model.set_constraint_name(
                        c_elim,
                        &format!("vb_elim_{}_prec_elim_c{}_lane{}", vb_idx, c, lane_i),
                    );
                    let c_keep = model.add_ge(alu_cycle, times[vb_idx].visible());
                    model.only_enforce_if(c_keep, [eliminate_var.not()]);
                    model.set_constraint_name(
                        c_keep,
                        &format!("vb_elim_{}_prec_keep_c{}_lane{}", vb_idx, c, lane_i),
                    );
                }
            }
        }
        (vars, intervals)
    };

    // 6b. ConstChoice optional intervals (decision variables were created in section 4)

    // Map constant value -> op index (first producer). Used for ALU-derived precedence.
    let value_to_producer_idx: HashMap<i64, usize> = {
        let mut m = HashMap::new();
        for (idx, op) in ops.iter().enumerate() {
            if let Some(v) = op.produces_constant {
                m.entry(v).or_insert(idx);
            }
        }
        m
    };

    // Note: We don't need to find zero VReg op for AddImm precedence constraints.
    // Scratch memory is ALWAYS initialized to 0 at startup, so address 0 always contains 0.
    // AddImm operations can use address 0 directly without waiting for any op to "produce" zero.
    // The zero VReg is just a reference to address 0, not a value that needs to be computed.

    let mut const_choice_vars: HashMap<usize, BoolVar> = HashMap::new();
    let mut const_load_intervals: HashMap<usize, IntervalVar> = HashMap::new();
    let mut const_flow_intervals: HashMap<usize, IntervalVar> = HashMap::new();
    // N-way ConstChoice (Load | AddImm | ALU_0 | ALU_1 | ...): encoding vars and ALU option info for extraction
    let mut const_choice_encoding_vars: HashMap<usize, Vec<BoolVar>> = HashMap::new();
    let mut const_choice_alu_encoding_info: HashMap<usize, Vec<(usize, usize, Opcode)>> =
        HashMap::new();
    let mut const_choice_alu_intervals: HashMap<usize, Vec<IntervalVar>> = HashMap::new();

    for &idx in &const_choice_ops {
        let has_alu_options = ops[idx]
            .const_alternatives
            .as_ref()
            .map(|a| a.len() > 2)
            .unwrap_or(false);
        let alternatives = ops[idx].const_alternatives.as_deref();

        match const_choice_mode {
            ConstChoiceMode::ForceLoad => {
                let load_interval = model.new_fixed_size_interval_var(times[idx].issue(), 1);
                const_load_intervals.insert(idx, load_interval);
            }
            ConstChoiceMode::ForceAddImm => {
                let add_imm_interval = model.new_fixed_size_interval_var(times[idx].issue(), 1);
                const_flow_intervals.insert(idx, add_imm_interval);
            }
            ConstChoiceMode::Optimize => {
                if !has_alu_options {
                    // Binary: Load vs AddImm (current behavior)
                    let Some(ConstChoiceVars::AddImm(choice_var)) = op_vars.const_choice.get(&idx) else {
                        unreachable!("binary ConstChoice has one literal")
                    };
                    let choice_var = *choice_var;
                    const_choice_vars.insert(idx, choice_var);

                    let load_interval = model.new_optional_fixed_size_interval_var(
                        times[idx].issue(),
                        1,
                        choice_var.not(),
                    );
                    const_load_intervals.insert(idx, load_interval);

                    let add_imm_interval =
                        model.new_optional_fixed_size_interval_var(times[idx].issue(), 1, choice_var);
                    const_flow_intervals.insert(idx, add_imm_interval);

                    let use_add_imm =
                        const_add_imm_hints.map_or(false, |h| h.contains(&ops[idx].id));
                    model.add_hint(choice_var, if use_add_imm { 1 } else { 0 });
                } else {
                    // N-way: Load, AddImm, and optionally ALU-derived encodings
                    let alts = alternatives.unwrap();
                    let Some(ConstChoiceVars::Encoding(use_vars)) = op_vars.const_choice.get(&idx) else {
                        unreachable!("N-way ConstChoice has one literal per encoding")
                    };
                    let use_vars: Vec<BoolVar> = use_vars.clone();
                    // Exactly one encoding
                    let sum_expr = use_vars
                        .iter()
                        .skip(1)
                        .fold(LinearExpr::from(use_vars[0]), |acc, &b| {
                            acc + LinearExpr::from(b)
                        });
                    let c = model.add_eq(sum_expr, 1);
                    model.set_constraint_name(c, &format!("const_choice_one_{}", idx));

                    let use_load = use_vars[0];
                    let use_add_imm = use_vars[1];

                    let load_interval =
                        model.new_optional_fixed_size_interval_var(times[idx].issue(), 1, use_load);
                    const_load_intervals.insert(idx, load_interval);

                    let add_imm_interval = model.new_optional_fixed_size_interval_var(
                        times[idx].issue(), 1, use_add_imm,
                    );
                    const_flow_intervals.insert(idx, add_imm_interval);

                    let mut alu_info: Vec<(usize, usize, Opcode)> = Vec::new();
                    for (k, enc) in alts[2..].iter().enumerate() {
                        let use_alu_k = use_vars[2 + k];
                        let alu_interval = model.new_optional_fixed_size_interval_var(
                            times[idx].issue(), 1, use_alu_k,
                        );
                        const_choice_alu_intervals
                            .entry(idx)
                            .or_default()
                            .push(alu_interval);

                        let deps = enc.dependencies();
                        if deps.len() >= 2 {
                            let left_idx = *value_to_producer_idx.get(&deps[0]).unwrap_or(&idx);
                            let right_idx = *value_to_producer_idx.get(&deps[1]).unwrap_or(&idx);
                            if let Some(opcode) = enc.alu_opcode() {
                                alu_info.push((left_idx, right_idx, opcode));
                                let c_left = model.add_ge(times[idx].issue(), times[left_idx].visible());
                                model.only_enforce_if(c_left, [use_alu_k]);
                                model.set_constraint_name(c_left, &format!("const_choice_prec_{}_k_{}_left", idx, k));
                                let c_right = model.add_ge(times[idx].issue(), times[right_idx].visible());
                                model.only_enforce_if(c_right, [use_alu_k]);
                                model.set_constraint_name(c_right, &format!("const_choice_prec_{}_k_{}_right", idx, k));
                            }
                        }
                    }
                    // Hint ALL encoding vars so the solution hint is complete.
                    // Default to Load (index 0); if const_add_imm_hints says AddImm, use index 1.
                    let hint_idx = if const_add_imm_hints
                        .map_or(false, |h| h.contains(&ops[idx].id))
                    {
                        1 // AddImm
                    } else {
                        0 // Load
                    };
                    // Window probe: a reference from an earlier solution may use an ALU-derived
                    // encoding, which the add_imm hint set cannot say.
                    let hint_idx = match window.and_then(|w| w.reference.const_choice_encoding.get(&ops[idx].id)) {
                        Some(crate::types::ConstChoiceEncoding::AluDerive(info)) => alu_info
                            .iter()
                            .position(|&(l, r, opcode)| {
                                ops[l].id == info.left_op_id && ops[r].id == info.right_op_id && opcode == info.opcode
                            })
                            .map_or(hint_idx, |k| k + 2),
                        _ => hint_idx,
                    };
                    for (i, &var) in use_vars.iter().enumerate() {
                        model.add_hint(var, if i == hint_idx { 1 } else { 0 });
                    }
                    const_choice_encoding_vars.insert(idx, use_vars);
                    const_choice_alu_encoding_info.insert(idx, alu_info);
                }
            }
        }
    }

    // 6c. SelectChoice optional intervals (decision literals and multiply_add cycles were
    // created in section 4)

    // CSE groups: op index -> group index for SelectChoice ops in CSE groups (no delta, has sub_temp)
    let (cse_consumer_set, op_idx_to_cse_group): (HashSet<usize>, HashMap<usize, usize>) =
        if let Some(groups) = cse_groups {
            let mut set = HashSet::new();
            let mut map = HashMap::new();
            for (g, group) in groups.iter().enumerate() {
                for &idx in &group.consumer_indices {
                    set.insert(idx);
                    map.insert(idx, g);
                }
            }
            (set, map)
        } else {
            (HashSet::new(), HashMap::new())
        };

    let mut select_flow_intervals: HashMap<usize, IntervalVar> = HashMap::new();
    let mut select_valu_intervals: HashMap<usize, Vec<IntervalVar>> = HashMap::new();

    for &idx in &select_choice_ops {
        let issue = times[idx].issue();
        match select_choice_mode {
            SelectChoiceMode::ForceVselect => {
                let flow_interval = model.new_fixed_size_interval_var(issue, 1);
                select_flow_intervals.insert(idx, flow_interval);
            }
            SelectChoiceMode::ForceFma => {
                let valu_intervals = if let Some(_delta_vreg) = ops[idx].delta_vreg {
                    vec![model.new_fixed_size_interval_var(issue, 1)]
                } else if cse_consumer_set.contains(&idx) {
                    // In CSE group: only FMA interval; shared SUB added per group
                    times[idx].add_fma_cycle_invariant(&mut model, idx);
                    let fma_cycle_var = times[idx].fma_cycle().expect("select without delta has an FMA cycle");
                    let fma_interval = model.new_fixed_size_interval_var(fma_cycle_var, 1);
                    vec![fma_interval]
                } else {
                    let sub_interval = model.new_fixed_size_interval_var(issue, 1);
                    times[idx].add_fma_cycle_invariant(&mut model, idx);
                    let fma_cycle_var = times[idx].fma_cycle().expect("select without delta has an FMA cycle");
                    let fma_interval = model.new_fixed_size_interval_var(fma_cycle_var, 1);
                    vec![sub_interval, fma_interval]
                };
                select_valu_intervals.insert(idx, valu_intervals);
            }
            SelectChoiceMode::Optimize => {
                let choice_var = times[idx].fma_chosen().expect("Optimize select has a choice literal");

                if let Some(hints) = select_choice_hints {
                    if hints.contains(&ops[idx].id) {
                        model.add_hint(choice_var, 1);
                    } else {
                        model.add_hint(choice_var, 0);
                    }
                } else if hint_select_choice_vselect_by_default {
                    model.add_hint(choice_var, 0);
                }

                let flow_interval = model.new_optional_fixed_size_interval_var(
                    issue,
                    1,
                    choice_var.not(),
                );
                select_flow_intervals.insert(idx, flow_interval);

                let valu_intervals = if let Some(_delta_vreg) = ops[idx].delta_vreg {
                    vec![model.new_optional_fixed_size_interval_var(issue, 1, choice_var)]
                } else if cse_consumer_set.contains(&idx) {
                    // In CSE group: only FMA interval; shared SUB added per group
                    times[idx].add_fma_cycle_invariant(&mut model, idx);
                    let fma_cycle_var = times[idx].fma_cycle().expect("select without delta has an FMA cycle");
                    let fma_interval =
                        model.new_optional_fixed_size_interval_var(fma_cycle_var, 1, choice_var);
                    vec![fma_interval]
                } else {
                    let sub_interval =
                        model.new_optional_fixed_size_interval_var(issue, 1, choice_var);
                    times[idx].add_fma_cycle_invariant(&mut model, idx);
                    let fma_cycle_var = times[idx].fma_cycle().expect("select without delta has an FMA cycle");
                    let fma_interval =
                        model.new_optional_fixed_size_interval_var(fma_cycle_var, 1, choice_var);
                    vec![sub_interval, fma_interval]
                };
                select_valu_intervals.insert(idx, valu_intervals);
            }
        }
    }

    // 6c-bis. CSE shared SUB: constraints per group (Phase A: SUB on VALU only); its
    // variables were created in section 4
    let mut cse_shared_sub_valu_intervals: Vec<IntervalVar> = Vec::new();
    if let Some(groups) = cse_groups {
        for (g, group) in groups.iter().enumerate() {
            let sub_active_var = cse_subs[g].active();
            let sub_cycle_var = cse_subs[g].cycle();

            // Producer precedence: when sub is active, sub_cycle >= producer result visible cycle
            // (completion + 1 for a shiftable producer, + 1 more for a select on the
            // multiply-add path).
            // For complex_phi, one or both producers are None; we only add precedence for present producers.
            if let Some(pt) = group.producer_true_idx {
                let c = model.add_ge(LinearExpr::from(sub_cycle_var), times[pt].visible());
                model.set_constraint_name(c, &format!("cse_sub_after_producer_true_g{}", g));
                model.only_enforce_if(c, [sub_active_var]);
            }
            if let Some(pf) = group.producer_false_idx {
                let c = model.add_ge(LinearExpr::from(sub_cycle_var), times[pf].visible());
                model.set_constraint_name(c, &format!("cse_sub_after_producer_false_g{}", g));
                model.only_enforce_if(c, [sub_active_var]);
            }

            // Activity: sub_active <=> (sum choice[c] >= 1)
            match select_choice_mode {
                SelectChoiceMode::ForceVselect => {
                    model.add_eq(sub_active_var, 0);
                }
                SelectChoiceMode::ForceFma => {
                    model.add_eq(sub_active_var, 1);
                }
                SelectChoiceMode::Optimize => {
                    let choice_vars: Vec<BoolVar> = group
                        .consumer_indices
                        .iter()
                        .filter_map(|&idx| select_choice_vars.get(&idx).copied())
                        .collect();
                    if !choice_vars.is_empty() {
                        let sum_choice = choice_vars
                            .iter()
                            .skip(1)
                            .fold(LinearExpr::from(choice_vars[0]), |acc, &b| {
                                acc + LinearExpr::from(b)
                            });
                        let c1 = model.add_ge(sum_choice.clone(), 1);
                        model.only_enforce_if(c1, [sub_active_var]);
                        let c2 = model.add_eq(sum_choice, 0);
                        model.only_enforce_if(c2, [sub_active_var.not()]);
                        // (sum_choice >= 1) => sub_active_var: any selected choice forces sub active
                        for &choice in &choice_vars {
                            model.add_implication(choice, sub_active_var);
                        }
                    }
                }
            }

            // Precedence: for each consumer c: sub_cycle + 1 <= fma_cycle[c] (when FMA path)
            for &idx in &group.consumer_indices {
                if select_choice_mode == SelectChoiceMode::ForceVselect {
                    continue; // No FMA path, no precedence needed
                }
                let Some(fma_cycle_var) = times[idx].fma_cycle() else {
                    continue;
                };
                let c = model.add_ge(LinearExpr::from(fma_cycle_var), cse_subs[g].visible());
                model.set_constraint_name(c, &format!("cse_prec_g{}_c{}", g, idx));
                if select_choice_mode == SelectChoiceMode::Optimize {
                    if let Some(&choice_var) = select_choice_vars.get(&idx) {
                        model.only_enforce_if(c, [choice_var]);
                    }
                }
            }

            // Optional VALU interval for shared SUB at sub_cycle when sub_active
            let sub_interval =
                model.new_optional_fixed_size_interval_var(sub_cycle_var, 1, sub_active_var);
            cse_shared_sub_valu_intervals.push(sub_interval);
        }
    }

    // 5d. Precedence for consumers of SelectChoice FMA (no delta): result is visible at cycle+2, not cycle+1
    // Store select_fma_extra vars so we can hint them (avoids "solution hint is incomplete").
    let mut select_fma_extra_vars: std::collections::HashMap<(usize, usize), IntVar> =
        std::collections::HashMap::new();
    for (idx, op) in ops.iter().enumerate() {
        for &dep_id in &op.dep_ids {
            let Some(&dep_idx) = op_id_to_idx.get(&dep_id) else {
                continue;
            };
            if !select_choice_with_fma_delay.contains(&dep_idx) {
                continue;
            }
            // A shifted consumer executes as 8 scalar ops at its lane cycles, not at its issue
            // cycle, so each lane needs the same FMA delay. Section 5b only gives the lanes
            // dep_cycle + 1 (the vselect delay).
            let consumer_lanes: &[IntVar] = times[idx].lanes();
            match select_choice_mode {
                SelectChoiceMode::ForceVselect => {
                    let c = model.add_ge(times[idx].issue(), times[dep_idx].visible());
                    model.set_constraint_name(c, &format!("select_prec_{}_dep_{}_vselect", idx, dep_idx));
                }
                SelectChoiceMode::ForceFma => {
                    let c = model.add_ge(times[idx].issue(), times[dep_idx].visible());
                    model.set_constraint_name(c, &format!("select_prec_{}_dep_{}_fma", idx, dep_idx));
                    for (lane_i, alu_cycle) in consumer_lanes.iter().enumerate() {
                        let c = model.add_ge(*alu_cycle, times[dep_idx].visible());
                        model.set_constraint_name(c, &format!("select_prec_{}_{}_dep_{}_fma", idx, lane_i, dep_idx));
                    }
                }
                SelectChoiceMode::Optimize => {
                    let choice_var = times[dep_idx].fma_chosen().expect("Optimize has choice var");
                    // consumer >= dep_cycle + 1 + (1 if FMA else 0). Use auxiliary 0-1 int.
                    let extra = model.new_int_var_with_name(
                        [(0, 1)],
                        format!("select_fma_extra_{}_{}", dep_idx, idx),
                    );
                    let c = model.add_eq(LinearExpr::from(extra), LinearExpr::from(choice_var));
                    model.set_constraint_name(c, &format!("select_fma_extra_{}_{}", dep_idx, idx));
                    let c = model.add_ge(
                        times[idx].issue(),
                        times[dep_idx].visible_with_latency(LinearExpr::from(extra)),
                    );
                    model.set_constraint_name(c, &format!("select_prec_{}_dep_{}_opt", idx, dep_idx));
                    for (lane_i, alu_cycle) in consumer_lanes.iter().enumerate() {
                        let c = model.add_ge(
                            *alu_cycle,
                            times[dep_idx].visible_with_latency(LinearExpr::from(extra)),
                        );
                        model.set_constraint_name(c, &format!("select_prec_{}_{}_dep_{}_opt", idx, lane_i, dep_idx));
                    }
                    select_fma_extra_vars.insert((dep_idx, idx), extra);
                }
            }
        }
    }

    // 7. Sets for distinguishing choice/shift ops from regular ops
    let const_choice_set: HashSet<usize> = const_choice_ops.iter().copied().collect();
    let select_choice_set: HashSet<usize> = select_choice_ops.iter().copied().collect();

    // 8. Build cumulative constraints with optional intervals

    // Collect non-shiftable VALU ops (exclude SelectChoice ops - they use optional intervals)
    let non_shiftable_valu: Vec<usize> = ops
        .iter()
        .enumerate()
        .filter(|(idx, op)| {
            op.engine == Engine::Valu
                && !shiftable_set.contains(idx)
                && op.opcode != Opcode::SelectChoice
        })
        .map(|(idx, _)| idx)
        .collect();

    // Collect ALU ops
    let alu_ops: Vec<usize> = ops
        .iter()
        .enumerate()
        .filter(|(_, op)| op.engine == Engine::Alu)
        .map(|(idx, _)| idx)
        .collect();

    // Window probe: ops whose intervals no movable interval can meet are left out of the
    // engine and scratch cumulatives (see CpsatWindow::pruned_ops). Empty otherwise.
    let window_pruned: HashSet<usize> =
        window.map_or_else(HashSet::new, |w| w.pruned_ops(ops, cse_groups.map_or(0, |g| g.len())));
    let in_cumulative = |idx: usize| !window_pruned.contains(&idx);
    if verbose && window.is_some() {
        eprintln!("  Window: {} ops left out of the cumulatives", window_pruned.len());
    }

    // VALU cumulative: non-shiftable VALU (always present except eliminable vb) + optional vb + shiftable + SelectChoice
    {
        let eliminable_vb_set: HashSet<usize> =
            eliminable_vbs.iter().map(|(vb_idx, _, _)| *vb_idx).collect();
        let mut valu_intervals: Vec<IntervalVar> = Vec::new();
        let mut valu_demands: Vec<i64> = Vec::new();

        // Add non-shiftable VALU ops that are not eliminable vbroadcasts (always present)
        for &idx in &non_shiftable_valu {
            if eliminable_vb_set.contains(&idx) || !in_cumulative(idx) {
                continue;
            }
            valu_intervals.push(model.new_fixed_size_interval_var(times[idx].issue(), 1));
            valu_demands.push(1);
        }
        // Add optional VALU intervals for eliminable vbroadcasts (presence = !eliminate_vb)
        for (vb_idx, _, _) in eliminable_vbs.iter().filter(|(i, _, _)| in_cumulative(*i)) {
            if let Some(&iv) = valu_optional_vb_intervals.get(vb_idx) {
                valu_intervals.push(iv);
                valu_demands.push(1);
            }
        }

        // Add VALU intervals for shiftable ops: ForceValu => required; ForceAlu => none; Optimize => optional
        for &idx in shiftable_valu.iter().filter(|&&i| in_cumulative(i)) {
            match valu_shift_mode {
                ValuShiftMode::ForceValu => {
                    valu_intervals.push(model.new_fixed_size_interval_var(times[idx].issue(), 1));
                    valu_demands.push(1);
                }
                ValuShiftMode::ForceAlu => {}
                ValuShiftMode::Optimize => {
                    if let Some(&iv) = valu_optional_intervals.get(&idx) {
                        valu_intervals.push(iv);
                        valu_demands.push(1);
                    }
                }
            }
        }

        // Add optional VALU intervals from SelectChoice ops (when choice_var = true, i.e., FMA path)
        for &idx in select_choice_ops.iter().filter(|&&i| in_cumulative(i)) {
            if let Some(intervals) = select_valu_intervals.get(&idx) {
                for &iv in intervals {
                    valu_intervals.push(iv);
                    valu_demands.push(1); // Each VALU op (sub or fma) uses 1 slot
                }
            }
        }

        // Add shared SUB intervals from CSE groups (Phase A: one per group when sub_active)
        for iv in &cse_shared_sub_valu_intervals {
            valu_intervals.push(*iv);
            valu_demands.push(1);
        }

        if !valu_intervals.is_empty() {
            let max_demand = *valu_demands.iter().max().unwrap_or(&0);
            debug_assert!(
                max_demand <= valu_cap,
                "cumul_valu: max demand {} > capacity {}",
                max_demand,
                valu_cap
            );
            let c = model.add_cumulative(valu_intervals, valu_demands, valu_cap);
            model.set_constraint_name(c, "cumul_valu");
        }
    }

    // ALU cumulative: regular ALU ops + optional ALU intervals (8 intervals with demand=1 each for shifted VALU)
    {
        let mut alu_intervals: Vec<IntervalVar> = Vec::new();
        let mut alu_demands: Vec<i64> = Vec::new();

        // Add regular ALU ops (demand = 1)
        for &idx in alu_ops.iter().filter(|&&i| in_cumulative(i)) {
            alu_intervals.push(model.new_fixed_size_interval_var(times[idx].issue(), 1));
            alu_demands.push(1);
        }

        // Add ALU intervals for shifted VALU ops: ForceAlu => required intervals; Optimize => optional
        for &idx in shiftable_valu.iter().filter(|&&i| in_cumulative(i)) {
            if let Some(intervals) = alu_optional_intervals.get(&idx) {
                for iv in intervals {
                    alu_intervals.push(*iv);
                    alu_demands.push(1);
                }
            }
        }

        // Add optional ALU intervals from ConstChoice ops (when ALU-derived encoding is chosen)
        for &idx in const_choice_ops.iter().filter(|&&i| in_cumulative(i)) {
            if let Some(intervals) = const_choice_alu_intervals.get(&idx) {
                for iv in intervals {
                    alu_intervals.push(*iv);
                    alu_demands.push(1);
                }
            }
        }

        if !alu_intervals.is_empty() {
            let max_demand = *alu_demands.iter().max().unwrap_or(&0);
            debug_assert!(
                max_demand <= alu_cap,
                "cumul_alu: max demand {} > capacity {}",
                max_demand,
                alu_cap
            );
            let c = model.add_cumulative(alu_intervals, alu_demands, alu_cap);
            model.set_constraint_name(c, "cumul_alu");
        }
    }

    // Load cumulative: regular Load ops + optional Load intervals from ConstChoice
    {
        let regular_load_ops: Vec<usize> = ops
            .iter()
            .enumerate()
            .filter(|(_, op)| op.engine == Engine::Load && op.opcode != Opcode::ConstChoice)
            .map(|(idx, _)| idx)
            .collect();

        let mut load_intervals: Vec<IntervalVar> = Vec::new();
        let mut load_demands: Vec<i64> = Vec::new();

        // Add regular Load ops
        for &idx in regular_load_ops.iter().filter(|&&i| in_cumulative(i)) {
            load_intervals.push(model.new_fixed_size_interval_var(times[idx].issue(), 1));
            load_demands.push(1);
        }

        // Add optional Load intervals from ConstChoice ops (when choice_var = false)
        for &idx in const_choice_ops.iter().filter(|&&i| in_cumulative(i)) {
            if let Some(&iv) = const_load_intervals.get(&idx) {
                load_intervals.push(iv);
                load_demands.push(1);
            }
        }

        if !load_intervals.is_empty() {
            let max_demand = *load_demands.iter().max().unwrap_or(&0);
            debug_assert!(
                max_demand <= load_cap,
                "cumul_load: max demand {} > capacity {}",
                max_demand,
                load_cap
            );
            let c = model.add_cumulative(load_intervals, load_demands, load_cap);
            model.set_constraint_name(c, "cumul_load");
        }
    }

    // Store cumulative (unchanged - no ConstChoice)
    {
        let store_ops: Vec<usize> = ops
            .iter()
            .enumerate()
            .filter(|(idx, op)| op.engine == Engine::Store && in_cumulative(*idx))
            .map(|(idx, _)| idx)
            .collect();

        if !store_ops.is_empty() {
            let intervals: Vec<IntervalVar> = store_ops
                .iter()
                .map(|&idx| model.new_fixed_size_interval_var(times[idx].issue(), 1))
                .collect();
            let demands: Vec<i64> = vec![1; intervals.len()];
            let max_demand = *demands.iter().max().unwrap_or(&0);
            debug_assert!(
                max_demand <= store_cap,
                "cumul_store: max demand {} > capacity {}",
                max_demand,
                store_cap
            );
            let c = model.add_cumulative(intervals, demands, store_cap);
            model.set_constraint_name(c, "cumul_store");
        }
    }

    // Flow cumulative: regular Flow ops + optional Flow intervals from ConstChoice + optional Flow intervals from SelectChoice
    //
    // NOTE: Opcode::AddImm ops are included as regular Flow ops. These are created by
    // emit_const() as an alternative to Load ops. Both the Load op AND the AddImm op
    // are emitted, but only one should be scheduled. The greedy scheduler marks AddImm
    // ops as dead (cycle=-1). For CP-SAT, we include them in the Flow constraint so if
    // the solver does schedule them, they consume a Flow slot.
    //
    // ConstChoice ops are different - they're a single op with alternatives modeled
    // via choice variables. SelectChoice ops are similar (vselect vs FMA path).
    {
        let regular_flow_ops: Vec<usize> = ops
            .iter()
            .enumerate()
            .filter(|(_idx, op)| {
                // Include all Flow ops EXCEPT:
                // - SelectChoice (has optional intervals via choice var)
                // - Note: Opcode::AddImm IS included (from emit_const() pairs)
                op.engine == Engine::Flow && op.opcode != Opcode::SelectChoice
            })
            .map(|(idx, _)| idx)
            .collect();

        let mut flow_intervals: Vec<IntervalVar> = Vec::new();
        let mut flow_demands: Vec<i64> = Vec::new();

        // Add regular Flow ops
        for &idx in regular_flow_ops.iter().filter(|&&i| in_cumulative(i)) {
            flow_intervals.push(model.new_fixed_size_interval_var(times[idx].issue(), 1));
            flow_demands.push(1);
        }

        // Add optional Flow intervals from ConstChoice ops (when choice_var = true)
        for &idx in const_choice_ops.iter().filter(|&&i| in_cumulative(i)) {
            if let Some(&iv) = const_flow_intervals.get(&idx) {
                flow_intervals.push(iv);
                flow_demands.push(1);
            }
        }

        // Add optional Flow intervals from SelectChoice ops (when choice_var = false, i.e., vselect path)
        for &idx in select_choice_ops.iter().filter(|&&i| in_cumulative(i)) {
            if let Some(&iv) = select_flow_intervals.get(&idx) {
                flow_intervals.push(iv);
                flow_demands.push(1);
            }
        }

        if verbose {
            eprintln!(
                "[Flow model breadcrumbs] regular_flow_ops={} (indices {:?}, op.ids {:?})",
                regular_flow_ops.len(),
                regular_flow_ops,
                regular_flow_ops
                    .iter()
                    .map(|&i| ops[i].id)
                    .collect::<Vec<_>>()
            );
            eprintln!(
                "[Flow model breadcrumbs] const_choice_ops={} (optional Flow when choice=true; indices {:?}, op.ids {:?})",
                const_choice_ops.len(),
                const_choice_ops,
                const_choice_ops.iter().map(|&i| ops[i].id).collect::<Vec<_>>()
            );
            eprintln!(
                "[Flow model breadcrumbs] select_choice_ops={} (optional Flow when choice=false; indices {:?}, op.ids {:?})",
                select_choice_ops.len(),
                select_choice_ops,
                select_choice_ops.iter().map(|&i| ops[i].id).collect::<Vec<_>>()
            );
            eprintln!(
                "[Flow model breadcrumbs] Flow cumulative: total intervals={}, capacity={}",
                flow_intervals.len(),
                flow_cap
            );
        }

        if !flow_intervals.is_empty() {
            let max_demand = *flow_demands.iter().max().unwrap_or(&0);
            debug_assert!(
                max_demand <= flow_cap,
                "cumul_flow: max demand {} > capacity {}",
                max_demand,
                flow_cap
            );
            // With one Flow slot the cumulative and the no-overlap state the same thing; the
            // optional group FlowCumulative leaves the cumulative out in that case.
            let cumulative_is_duplicate = flow_cap == 1
                && disable_optional_constraint_groups.contains(&OptionalConstraintGroup::FlowCumulative);
            if !cumulative_is_duplicate {
                let c = model.add_cumulative(flow_intervals.clone(), flow_demands, flow_cap);
                model.set_constraint_name(c, "cumul_flow");
            }
            // Redundant with cumulative(1): no two Flow intervals overlap when present.
            let c = model.add_no_overlap(flow_intervals);
            model.set_constraint_name(c, "no_overlap_flow");
        }
    }

    // 8b. Add scratch pressure constraint (live VReg capacity)
    // Models: at any cycle, total size of live VRegs <= scratch_budget
    //
    // CRITICAL: This constraint must match address_assign.rs liveness logic exactly.
    // Two bugs were fixed to ensure consistency:
    // 1. Shifted VALU dests: start = min(alu_cycles), not completion_cycle (max).
    //    Address assignment reserves addresses from the FIRST write (min), not the last.
    // 2. SelectChoice FMA reads: cond/false_val/delta are read at cycle+1 by FMA,
    //    so interval end = cycle+2 (half-open) to match address_assign.rs which extends
    //    these intervals to cycle+1 (closed) = cycle+2 (half-open).
    //
    // Without these fixes, CP-SAT could report FEASIBLE solutions that exceed budget
    // during address assignment (e.g., peak_live=1825 > budget=1536).
    // Track scratch-related IntVars that need hints (select_fma_end, shifted_dest_start).
    // Declared outside scratch_budget block so section 11 can hint them.
    let mut scratch_select_fma_end_vars: Vec<(IntVar, usize)> = Vec::new(); // (var, last_use_idx)
    let mut scratch_shifted_start_vars: Vec<(IntVar, usize)> = Vec::new(); // (var, def_idx)
    // Per CSE group: (end, size) of the shared SUB's scratch interval, for hinting.
    let mut cse_sub_scratch_vars: Vec<(IntVar, IntVar)> = Vec::new();
    // Readers per live VReg (def idx, reader idxs), kept for the post-solve scratch check.
    let mut scratch_values: Vec<(VReg, usize, Vec<usize>)> = Vec::new();

    if scratch_budget > 0 {
        // Build VReg -> defining op index mapping
        let mut vreg_def_op: HashMap<VReg, usize> = HashMap::new();
        for (idx, op) in ops.iter().enumerate() {
            if let Some(dest) = op.dest {
                vreg_def_op.insert(dest, idx);
            }
        }

        // Build VReg -> last use op index mapping
        // IMPORTANT: We need the op with the LATEST CYCLE that reads this VReg, not just the latest index.
        // If we have greedy hints, use them to determine the actual last use by cycle.
        // Otherwise, fall back to index order (which may be incorrect but is a reasonable heuristic).
        //
        // Dead Store Handling:
        // The greedy scheduler (ssa_greedy.rs) identifies dead stores during scheduling
        // and pre-marks them as "scheduled" without assigning a valid cycle. Their cycle
        // field remains at -1 (the initial unscheduled value). CP-SAT honors this by:
        // 1. Skipping ops with negative cycle hints when building vreg_last_use
        // 2. This correctly excludes dead stores from liveness analysis
        // The warnings printed by greedy are informational, not errors.
        let mut vreg_last_use: HashMap<VReg, usize> = HashMap::new();
        for (idx, op) in ops.iter().enumerate() {
            // Skip dead stores identified by greedy scheduler.
            // Dead stores have negative cycle hints (-1) because greedy pre-marks them
            // as "scheduled" without actually scheduling them to a real cycle.
            if let Some(hints) = greedy_hints {
                if hints.get(idx).copied().unwrap_or(0) < 0 {
                    continue; // Dead store, skip - it will not be scheduled by CP-SAT either
                }
            }
            for vreg in op.reads() {
                let should_update = if let Some(hints) = greedy_hints {
                    // Use cycle from hints to determine if this is a later use
                    let current_cycle = hints.get(idx).copied().unwrap_or(0);
                    let existing_cycle = vreg_last_use
                        .get(&vreg)
                        .and_then(|&existing_idx| hints.get(existing_idx).copied())
                        .unwrap_or(-1);
                    current_cycle > existing_cycle
                } else {
                    // Without hints, use index order (may be suboptimal)
                    true
                };

                if should_update || !vreg_last_use.contains_key(&vreg) {
                    vreg_last_use.insert(vreg, idx);
                }
            }
        }

        // SelectChoice FMA reads at cycle+1: cond, false_val, and delta_vreg are read by FMA at cycle+1,
        // not at the SelectChoice op's cycle. We must extend their last_use to account for this.
        // This matches address_assign.rs logic (lines 234-260) which extends intervals to cycle+1.
        for &idx in &select_choice_ops {
            if ops[idx].opcode != Opcode::SelectChoice {
                continue;
            }
            // Check if FMA path is chosen (or forced)
            let is_fma = match select_choice_mode {
                SelectChoiceMode::ForceFma => true,
                SelectChoiceMode::ForceVselect => false,
                SelectChoiceMode::Optimize => {
                    // For Optimize, we can't know the choice yet, but we need to model both possibilities.
                    // We'll handle this conservatively: if the op has sub_temp_vreg (no-delta path),
                    // we model the FMA reads at cycle+1 as optional (only when choice_var=true).
                    // For now, we'll be conservative and always extend to cycle+1 when hints suggest FMA.
                    // The actual constraint will be enforced via optional intervals in the scratch model.
                    if let Some(hints) = select_choice_hints {
                        hints.contains(&ops[idx].id)
                    } else {
                        false // Without hints, be conservative: assume FMA might be chosen
                    }
                }
            };
            if is_fma && ops[idx].srcs.len() >= 3 {
                // FMA reads cond (srcs[0]) and false_val (srcs[2]) at cycle+1
                // true_val (srcs[1]) is only used by sub in sub+fma path, not by FMA
                for (i, src) in ops[idx].srcs.iter().enumerate() {
                    if i != 1 {
                        // cond (i=0) or false_val (i=2)
                        if let Operand::VReg(vreg) = src {
                            // Update last_use to this SelectChoice op (cycle+1 will be handled in scratch interval end)
                            // We use idx as the "last use" op, but the scratch constraint will use the issue cycle + 1
                            if !vreg_last_use.contains_key(vreg) {
                                vreg_last_use.insert(*vreg, idx);
                            }
                        }
                    }
                }
                // For delta path: delta_vreg is also read by FMA at cycle+1
                if let Some(delta_vreg) = ops[idx].delta_vreg {
                    if !vreg_last_use.contains_key(&delta_vreg) {
                        vreg_last_use.insert(delta_vreg, idx);
                    }
                }
            }
        }

        // Every reader of every value. The interval of a value must cover all of them: the
        // reader that is last in the greedy hint need not be last in the solution, and values
        // read only through a vector-group alias have no direct reader at all.
        let scratch_readers =
            scratch_readers_by_vreg(ops, &vreg_def_op, hints.vector_group_lanes);
        let hinted_fma = |idx: usize| {
            ops[idx].opcode == Opcode::SelectChoice
                && match select_choice_mode {
                    SelectChoiceMode::ForceFma => true,
                    SelectChoiceMode::ForceVselect => false,
                    SelectChoiceMode::Optimize => {
                        select_choice_hints.is_some_and(|h| h.contains(&ops[idx].id))
                    }
                }
        };
        let hinted_first_write = |idx: usize| -> Option<i32> {
            let cycle = greedy_hints?.get(idx).copied().filter(|c| *c >= 0)?;
            Some(
                hinted_lanes(ops[idx].id)
                    .and_then(|lanes| lanes.iter().copied().min())
                    .unwrap_or(cycle),
            )
        };
        // Cycle after the last read by reader `idx` under the greedy hint (None without hints).
        let hinted_read_end = |idx: usize| -> Option<i32> {
            // Dead stores have no hint (cycle -1): they do not contribute to the hinted end.
            let cycle = greedy_hints?.get(idx).copied().filter(|c| *c >= 0)?;
            Some(cycle + 1 + i32::from(hinted_fma(idx)))
        };
        // Hinted end of each value's interval: the latest hinted read end over all readers.
        let hinted_live_end: HashMap<VReg, i32> = scratch_readers
            .iter()
            .filter_map(|(vreg, readers)| {
                readers.iter().filter_map(|&r| hinted_read_end(r)).max().map(|e| (*vreg, e))
            })
            .collect();
        // `vreg_last_use` stays as the representative reader (latest in the hint) for segment
        // bookkeeping and single-reader values; make it cover alias-only readers too.
        for (vreg, readers) in &scratch_readers {
            if let Some(&latest) = readers.iter().max_by_key(|&&r| (hinted_read_end(r), r)) {
                vreg_last_use.insert(*vreg, latest);
            }
        }

        if verbose {
            // Count ops with dest
            let ops_with_dest = ops.iter().filter(|op| op.dest.is_some()).count();
            let ops_with_reads = ops.iter().filter(|op| op.reads().count() > 0).count();
            eprintln!(
                "  Scratch debug: {} ops total, {} with dest, {} with reads, {} VReg defs, {} VReg uses",
                ops.len(),
                ops_with_dest,
                ops_with_reads,
                vreg_def_op.len(),
                vreg_last_use.len()
            );
        }

        // We only model VRegs that have both a definition and at least one use.
        // The caller (build_and_schedule_cpsat) runs SSA DCE before CP-SAT when
        // enable_ssa_dce is true (default), so every defined VReg is expected to
        // have at least one reader. If DCE is disabled, dead VRegs would be omitted
        // from the scratch constraint and could cause undercounting; the caller
        // asserts no dead VRegs when DCE is enabled.

        // Collect VRegs that have both definition and use (all defined VRegs when DCE ran)
        let mut live_vregs: Vec<VReg> = vreg_def_op
            .keys()
            .filter(|v| vreg_last_use.contains_key(v))
            .copied()
            .collect();
        // Sorted so scratch intervals are created in the same order on every run.
        live_vregs.sort();

        let hinted_segment_finish = greedy_hints.map(|cycles| {
            cycles
                .iter()
                .zip(times.iter())
                .enumerate()
                .filter(|(_, (&cycle, _))| cycle >= 0)
                .map(|(idx, (&cycle, time))| {
                    cycle + i32::from(
                        scratch_time == ScratchTime::Phase
                            && time.fma_cycle().is_some()
                            && hinted_fma(idx),
                    )
                })
                .max()
                .unwrap_or(0)
        });

        // Segment mode with live-out or live-through: need segment_end so we can model
        // VRegs live until end of segment (live-out = defined here, used later; live-through = defined in prev section, used later).
        let segment_end_var: Option<IntVar> = if initial_live.is_some()
            && (live_out_vregs.is_some_and(|s| !s.is_empty())
                || live_through_vregs.is_some_and(|s| !s.is_empty()))
        {
            let m = model.new_int_var_with_name([(0, horizon)], "segment_end");
            for (idx, time) in times.iter().enumerate() {
                if scratch_time == ScratchTime::Phase {
                    // A live-through value still occupies scratch at the last write slot.
                    for finish in time.finish_alternatives() {
                        let c = model.add_ge(m, finish.value);
                        if let Some(when) = finish.when {
                            model.only_enforce_if(c, [when]);
                        }
                        model.set_constraint_name(c, &format!("segment_end_ge_{}_{}", finish.label, idx));
                    }
                } else {
                    let c = model.add_ge(m, time.issue());
                    model.set_constraint_name(c, &format!("segment_end_ge_c_{}", idx));
                }
            }
            if let Some(finish) = hinted_segment_finish {
                model.add_hint(m, i64::from(finish));
            }
            Some(m)
        } else {
            None
        };

        // Section live-ins, live-through values and temporary intervals can consume scratch
        // even when no locally produced value is read within the section.
        {
            let mut scratch_intervals: Vec<IntervalVar> = Vec::new();
            let mut scratch_demands: Vec<i64> = Vec::new();
            let mut size_vars: Vec<(IntVar, usize, usize)> = Vec::new(); // (var, def_idx, last_use_idx)
            // The value of each size_vars entry (not every live VReg gets one: the window
            // probe leaves some out).
            let mut size_var_vregs: Vec<VReg> = Vec::new();

            // Scratch model `pools`: a write group (alias with no definer, lanes written one by
            // one) is one 8-word value for address assignment, so it is one vector here, live
            // from its first lane write to its last lane's end. Lanes then contribute to the
            // group's interval instead of having their own. (lane -> group; group -> lane
            // (start, end, def_idx, end hint))
            let mut pool_group_of_lane: HashMap<VReg, VReg> = HashMap::new();
            let mut scratch_pool_vars: Option<(IntVar, IntVar)> = None;
            if scratch_pools {
                if let Some(groups) = hints.vector_group_lanes {
                    for (group, lanes) in sorted_by_key(groups) {
                        if vreg_def_op.contains_key(group) {
                            continue; // read group: the group value itself is defined (a vector)
                        }
                        for lane in lanes {
                            pool_group_of_lane.insert(*lane, *group);
                        }
                    }
                }
            }
            let mut pool_group_lanes: std::collections::BTreeMap<VReg, Vec<(IntVar, LinearExpr, usize, Option<i32>)>> =
                std::collections::BTreeMap::new();

            // Group ends are expressed as last-read ends for scratch_interval. In phase
            // mode it subtracts one from such ends; a lane held through the section keeps
            // the final write slot, so compensate before taking the maximum over lanes.
            let group_live_out_offset = i32::from(scratch_time == ScratchTime::Phase);
            let hinted_segment_end = hinted_segment_finish.map(|finish| finish + 1);

            // =========================================================================
            // CP-SAT LIVENESS MODEL: Matches Address Assignment Logic
            // =========================================================================
            //
            // Each value is built on the cycle timeline as [start, end) with
            // `end = last_use_cycle + 1`, then turned into the cumulative's interval by
            // `scratch_interval` on the timeline `scratch_time` selects.
            //
            // Within a cycle every read happens before every write, so a word whose value is
            // last read at cycle C can be rewritten at cycle C. Address assignment and the
            // greedy counter use that (same-cycle reuse). `ScratchTime::Cycle` does not: at
            // cycle C it counts both the values dying at C and the values born at C. That is
            // not rare: every op whose input dies as its output is written is this case, so it
            // happens along every dependency chain (16-round greedy schedules: 46-64 words at
            // the peak, up to 80 in one cycle; scripts/scratch_time_overcount.py).
            // `ScratchTime::Phase` (default) ends each value at its last read, `[def, last_read)`,
            // which is exactly the allocator's per-cycle peak on its read/write phases
            // (verification/allocation_conjectures.py: A8-phase-interval-is-allocator-span,
            // A8-phase-peak-exhaustive).
            //
            // CRITICAL FIXES (to match address_assign.rs logic):
            // 1. Shifted VALU dests: start = min(alu_cycles), not completion_cycle (max).
            //    Address assignment reserves the address from the FIRST write (min), not the last.
            // 2. SelectChoice FMA reads: cond/false_val/delta are read at cycle+1 by FMA,
            //    so interval end = cycle+2 (not cycle+1). This matches address_assign.rs
            //    which extends these intervals to cycle+1 (closed) = cycle+2 (half-open).
            //
            // RESULT: CP-SAT scratch constraint now matches address assignment logic.
            // If CP-SAT says schedule X fits in budget, address assignment will also fit.
            // =========================================================================

            for vreg in &live_vregs {
                let def_idx = vreg_def_op[vreg];
                let last_use_idx = vreg_last_use[vreg];
                let is_live_out = live_out_vregs.is_some_and(|s| s.contains(vreg));

                // Live-out VRegs (defined here, last-used in a later section) must extend to segment end
                // so the scratch constraint matches the full-kernel peak.
                let readers: &[usize] = scratch_readers.get(vreg).map_or(&[], |r| r.as_slice());
                if initial_live.is_none() {
                    scratch_values.push((*vreg, def_idx, readers.to_vec()));
                }
                // Window probe: a value defined and read only by pruned ops is live only at
                // cycles nothing can move to (the post-solve check still covers it).
                if !in_cumulative(def_idx) && readers.iter().all(|&r| !in_cumulative(r)) {
                    continue;
                }
                let (end_expr, end_for_interval) = if is_live_out {
                    let seg_end = segment_end_var.expect("segment_end when live_out");
                    let e = LinearExpr::from(seg_end) + 1;
                    (e.clone(), e)
                } else if readers.len() > 1 {
                    // Several readers: the value is live until the last of them has read it.
                    // A shifted reader reads at its lane cycles, so use its completion cycle.
                    let end_var = model.new_int_var_with_name(
                        [(0, horizon + 2)],
                        format!("live_end_{}", vreg.id),
                    );
                    let read_ends: Vec<LinearExpr> =
                        readers.iter().map(|&r| times[r].last_read_end()).collect();
                    let c = model.add_max_eq(end_var, read_ends);
                    model.set_constraint_name(c, &format!("scratch_live_end_{}", vreg.id));
                    if let Some(&end_hint) = hinted_live_end.get(vreg) {
                        model.add_hint(end_var, end_hint as i64);
                    }
                    (LinearExpr::from(end_var), LinearExpr::from(end_var))
                } else {
                    // Normal read at cycle C: interval end = C+1 (half-open: [start, end))
                    // SelectChoice FMA reads at cycle+1: if last_use is a SelectChoice op that chose FMA,
                    // the actual read happens at cycle+1, so end should be cycle+2.
                    // A shifted reader reads at its lane cycles: completion is the last of them
                    // (and equals the op cycle when it is not shifted).
                    let reader = &times[last_use_idx];
                    if select_choice_set.contains(&last_use_idx) {
                        match select_choice_mode {
                            SelectChoiceMode::ForceFma => {
                                // FMA forced: reads at cycle+1, end = cycle+2
                                let e = reader.last_read_end_on_path(true);
                                (e.clone(), e)
                            }
                            SelectChoiceMode::ForceVselect => {
                                // vselect forced: reads at cycle, end = cycle+1
                                let e = reader.last_read_end_on_path(false);
                                (e.clone(), e)
                            }
                            SelectChoiceMode::Optimize => {
                                // Conditional: if FMA chosen, end = cycle+2; if vselect, end = cycle+1
                                if let Some(choice_var) = reader.fma_chosen() {
                                    // Create conditional end: choice_var ? (cycle+2) : (cycle+1)
                                    let end_fma = reader.last_read_end_on_path(true);
                                    let end_vselect = reader.last_read_end_on_path(false);
                                    // Use a variable for the end, constrained by choice
                                    let end_var = model.new_int_var_with_name(
                                        [(0, horizon + 2)],
                                        format!("select_fma_end_{}", last_use_idx),
                                    );
                                    let c_fma = model.add_eq(LinearExpr::from(end_var), end_fma.clone());
                                    model.only_enforce_if(c_fma, [choice_var]);
                                    model.set_constraint_name(c_fma, &format!("scratch_end_fma_{}", last_use_idx));
                                    let c_vselect = model.add_eq(LinearExpr::from(end_var), end_vselect.clone());
                                    model.only_enforce_if(c_vselect, [!choice_var]);
                                    model.set_constraint_name(c_vselect, &format!("scratch_end_vselect_{}", last_use_idx));
                                    scratch_select_fma_end_vars.push((end_var, last_use_idx));
                                    (LinearExpr::from(end_var), LinearExpr::from(end_var))
                                } else {
                                    // No choice_var (shouldn't happen in Optimize mode)
                                    let e = reader.last_read_end_on_path(false);
                                    (e.clone(), e)
                                }
                            }
                        }
                    } else {
                        // Normal read at cycle C: end = C+1
                        let e = reader.last_read_end();
                        (e.clone(), e)
                    }
                };

                // The VReg is live from when it's defined until its last use (or segment end if live-out).
                // size = end - start
                //
                // Shifted VALU dests: 8 ALU ops write at different cycles. The address is reserved from
                // the FIRST write (min of the lane cycles), not the completion cycle (max). This matches
                // address_assign.rs logic (def_cycle = min(alu_cycles)). `first_write` creates a start
                // variable no later than any lane and equal to the issue cycle when not shifted; the
                // objective pushes it up to the minimum.
                let (start, start_aux) = times[def_idx].first_write(&mut model, def_idx, horizon);
                if let Some(start_var) = start_aux {
                    scratch_shifted_start_vars.push((start_var, def_idx));
                }

                // Create a size variable: size = end - start
                // Size must be >= 1 (def and last_use could be same cycle)
                let size = model
                    .new_int_var_with_name([(1, horizon + 1)], format!("live_size_{}", vreg.id));

                // Constraint: cycle[def] + size = end
                let c = model.add_eq(LinearExpr::from(start) + LinearExpr::from(size), end_expr);
                model.set_constraint_name(c, &format!("scratch_live_size_{}", vreg.id));

                size_vars.push((size, def_idx, last_use_idx));
                size_var_vregs.push(*vreg);
                if let Some(group) = pool_group_of_lane.get(vreg) {
                    let (group_end, end_hint) = if is_live_out {
                        (end_for_interval + i64::from(group_live_out_offset),
                         hinted_segment_end.map(|e| e + group_live_out_offset))
                    } else {
                        (end_for_interval, hinted_live_end.get(vreg).copied())
                    };
                    pool_group_lanes.entry(*group).or_default().push((start, group_end, def_idx, end_hint));
                    continue;
                }
                let interval = scratch_interval(
                    &mut model,
                    scratch_time,
                    LinearExpr::from(start),
                    LinearExpr::from(size),
                    end_for_interval,
                    ScratchFrom::Write,
                    if is_live_out { ScratchTo::SegmentEnd } else { ScratchTo::LastRead },
                    None,
                );
                scratch_intervals.push(interval);

                // Demand = size of VReg (1 for scalar, VLEN for vector)
                let demand = vreg.size() as i64;
                scratch_demands.push(demand);
            }

            // Segment mode: add live_out VRegs that are defined in segment but not used within segment.
            // These VRegs are not in live_vregs (because vreg_last_use only includes uses within the segment),
            // but they must be accounted for in scratch since they're live until segment end.
            if let Some(lo) = live_out_vregs {
                if let Some(seg_end) = segment_end_var {
                    let mut lo: Vec<&VReg> = lo.iter().collect();
                    lo.sort();
                    for vreg in lo {
                        // Only add if not already processed (not in live_vregs)
                        if !live_vregs.contains(vreg) {
                            // This VReg is defined in segment but only used after
                            if let Some(&def_idx) = vreg_def_op.get(vreg) {
                                let (start, start_aux) =
                                    times[def_idx].first_write(&mut model, def_idx, horizon);
                                if let Some(start_var) = start_aux {
                                    scratch_shifted_start_vars.push((start_var, def_idx));
                                }
                                let end_expr = LinearExpr::from(seg_end) + 1;
                                if let Some(group) = pool_group_of_lane.get(vreg) {
                                    pool_group_lanes.entry(*group).or_default().push((
                                        start,
                                        end_expr + i64::from(group_live_out_offset),
                                        def_idx,
                                        hinted_segment_end.map(|e| e + group_live_out_offset),
                                    ));
                                    continue;
                                }
                                let size = model.new_int_var_with_name(
                                    [(1, horizon + 1)],
                                    format!("live_out_size_{}", vreg.id),
                                );
                                if let (Some(start), Some(end)) =
                                    (hinted_first_write(def_idx), hinted_segment_end)
                                {
                                    model.add_hint(size, i64::from((end - start).max(1)));
                                }
                                let c = model.add_eq(
                                    LinearExpr::from(start) + LinearExpr::from(size),
                                    end_expr.clone(),
                                );
                                model.set_constraint_name(c, &format!("scratch_live_out_size_{}", vreg.id));
                                let interval = scratch_interval(
                                    &mut model,
                                    scratch_time,
                                    LinearExpr::from(start),
                                    LinearExpr::from(size),
                                    end_expr,
                                    ScratchFrom::Write,
                                    ScratchTo::SegmentEnd,
                                    None,
                                );
                                scratch_intervals.push(interval);
                                scratch_demands.push(vreg.size() as i64);
                            }
                        }
                    }
                }
            }

            // Pools: one vector interval per write group, [min lane start, max lane end).
            // (group, start hint, end hint) for the pool-peak hints below.
            let mut pool_group_hints: Vec<(i32, i32)> = Vec::new();
            for (group, lanes) in &pool_group_lanes {
                let g_start = model.new_int_var_with_name(
                    [(0, horizon)],
                    format!("pool_group_start_{}", group.id),
                );
                let g_end = model.new_int_var_with_name(
                    [(0, horizon + 2)],
                    format!("pool_group_end_{}", group.id),
                );
                let g_size = model.new_int_var_with_name(
                    [(1, horizon + 2)],
                    format!("pool_group_size_{}", group.id),
                );
                let c = model.add_min_eq(g_start, lanes.iter().map(|(s, _, _, _)| LinearExpr::from(*s)));
                model.set_constraint_name(c, &format!("pool_group_start_min_{}", group.id));
                let c = model.add_max_eq(g_end, lanes.iter().map(|(_, e, _, _)| e.clone()));
                model.set_constraint_name(c, &format!("pool_group_end_max_{}", group.id));
                let c = model.add_eq(
                    LinearExpr::from(g_start) + LinearExpr::from(g_size),
                    LinearExpr::from(g_end),
                );
                model.set_constraint_name(c, &format!("pool_group_size_{}", group.id));
                let interval = scratch_interval(
                    &mut model,
                    scratch_time,
                    LinearExpr::from(g_start),
                    LinearExpr::from(g_size),
                    LinearExpr::from(g_end),
                    ScratchFrom::Write,
                    ScratchTo::LastRead,
                    None,
                );
                scratch_intervals.push(interval);
                scratch_demands.push(VLEN as i64);
                if greedy_hints.is_some() {
                    let start_hint = lanes
                        .iter()
                        .map(|(_, _, d, _)| hinted_first_write(*d).unwrap_or(0))
                        .min()
                        .unwrap_or(0);
                    let end_hint = lanes
                        .iter()
                        .filter_map(|(_, _, _, e)| *e)
                        .max()
                        .unwrap_or(start_hint + 1)
                        .max(start_hint + 1);
                    model.add_hint(g_start, start_hint as i64);
                    model.add_hint(g_end, end_hint as i64);
                    model.add_hint(g_size, (end_hint - start_hint) as i64);
                    pool_group_hints.push((start_hint, end_hint));
                }
            }

            // Segment mode: live-in VRegs are live from the segment start until the last of
            // their readers in the segment has read them: [0, max over readers of read + 1),
            // with the same per-reader rules as values defined here (a shifted reader's
            // completion cycle, +1 for a SelectChoice on the multiply-add path). A single reader
            // picked from the greedy schedule (`last_use_idx`) is not enough: the solve can move
            // another reader later. The cycle timeline's same-cycle overcount used to hide this.
            let mut incoming_hinted_ends = Vec::new();
            if let Some(il) = initial_live {
                let zero_var = model.new_int_var_with_name([(0, 0)], "zero_seg");
                // Include external owners when resolving aliases. An incoming assembled
                // lane may have several readers of the vector alias, while an incoming
                // decomposed vector may have several independent scalar lane readers.
                let mut known_defs = vreg_def_op.clone();
                known_defs.extend(il.iter().copied());
                let incoming_readers = scratch_readers_by_vreg(ops, &known_defs, hints.vector_group_lanes);
                for (vreg, last_use_idx) in il {
                    if *last_use_idx >= times.len() {
                        continue;
                    }
                    let fallback = [*last_use_idx];
                    let readers = incoming_readers.get(vreg).map_or(fallback.as_slice(), Vec::as_slice);
                    let read_end = |r: usize| times[r].last_read_end();
                    // OR-Tools interval endpoints must be affine in a single variable.
                    // An optional FMA read adds its path literal to the issue cycle.
                    let end = model.new_int_var_with_name(
                        [(1, horizon + 2)],
                        format!("init_live_end_{}", vreg.id),
                    );
                    let end_expr = LinearExpr::from(end);
                    let c = if readers.len() == 1 {
                        model.add_eq(end_expr.clone(), read_end(readers[0]))
                    } else {
                        model.add_max_eq(end, readers.iter().map(|&r| read_end(r)))
                    };
                    model.set_constraint_name(c, &format!("scratch_init_live_end_{}", vreg.id));
                    let hinted_end = greedy_hints.map(|_| {
                        readers.iter().filter_map(|&r| hinted_read_end(r)).max().unwrap_or(1)
                    });
                    if let Some(hint) = hinted_end {
                        model.add_hint(end, i64::from(hint));
                        incoming_hinted_ends.push((*vreg, hint));
                    }
                    let size = model.new_int_var_with_name(
                        [(1, horizon + 2)],
                        format!("init_live_size_{}", vreg.id),
                    );
                    // Hint: size = end (start is 0)
                    if let Some(e) = hinted_end {
                        model.add_hint(size, e.max(1) as i64);
                    }
                    let c = model.add_eq(
                        LinearExpr::from(zero_var) + LinearExpr::from(size),
                        end_expr.clone(),
                    );
                    model.set_constraint_name(c, &format!("scratch_init_live_size_{}", vreg.id));
                    let interval = scratch_interval(
                        &mut model,
                        scratch_time,
                        LinearExpr::from(zero_var),
                        LinearExpr::from(size),
                        end_expr,
                        ScratchFrom::SegmentStart,
                        ScratchTo::LastRead,
                        None,
                    );
                    scratch_intervals.push(interval);
                    scratch_demands.push(vreg.size() as i64);
                }
            }

            // Segment mode: add [0, segment_end+1] intervals for live-through VRegs (defined in prev section, used in later section)
            if let (Some(lt), Some(seg_end)) = (live_through_vregs, segment_end_var) {
                let zero_var = model.new_int_var_with_name([(0, 0)], "zero_seg");
                let mut lt: Vec<&VReg> = lt.iter().collect();
                lt.sort();
                for vreg in lt {
                    let end_expr = LinearExpr::from(seg_end) + 1;
                    let size = model.new_int_var_with_name(
                        [(1, horizon + 1)],
                        format!("live_through_size_{}", vreg.id),
                    );
                    // Hint: size = seg_end + 1 (start is 0, end is seg_end + 1)
                    if let Some(end) = hinted_segment_end {
                        model.add_hint(size, i64::from(end));
                    }
                    let c = model.add_eq(
                        LinearExpr::from(zero_var) + LinearExpr::from(size),
                        end_expr.clone(),
                    );
                    model.set_constraint_name(c, &format!("scratch_live_through_size_{}", vreg.id));
                    let interval = scratch_interval(
                        &mut model,
                        scratch_time,
                        LinearExpr::from(zero_var),
                        LinearExpr::from(size),
                        end_expr,
                        ScratchFrom::SegmentStart,
                        ScratchTo::SegmentEnd,
                        None,
                    );
                    scratch_intervals.push(interval);
                    scratch_demands.push(vreg.size() as i64);
                }
            }

            // SelectChoice FMA sub_temp_vreg: live [op.cycle, op.cycle+2) when FMA path is chosen.
            // ForceFma => required interval; Optimize => optional (present when choice_var); ForceVselect => none.
            // CSE consumers skip individual sub_temp (they use shared sub_temp instead).
            for &idx in &select_choice_ops {
                if ops[idx].sub_temp_vreg.is_none() {
                    continue; // delta path: no sub_temp
                }
                if idx >= times.len() {
                    continue;
                }
                // Skip CSE consumers - they use shared sub_temp instead of individual sub_temp
                if cse_consumer_set.contains(&idx) || !in_cumulative(idx) {
                    continue;
                }
                let start = times[idx].issue();
                match select_choice_mode {
                    // Written by the subtract at C, read by the multiply_add at C + 1.
                    SelectChoiceMode::ForceFma => {
                        let interval = scratch_interval(
                            &mut model,
                            scratch_time,
                            LinearExpr::from(start),
                            LinearExpr::from(2),
                            LinearExpr::from(start) + 2,
                            ScratchFrom::Write,
                            ScratchTo::LastRead,
                            None,
                        );
                        scratch_intervals.push(interval);
                        scratch_demands.push(8);
                    }
                    SelectChoiceMode::Optimize => {
                        if let Some(&choice_var) = select_choice_vars.get(&idx) {
                            let interval = scratch_interval(
                                &mut model,
                                scratch_time,
                                LinearExpr::from(start),
                                LinearExpr::from(2),
                                LinearExpr::from(start) + 2,
                                ScratchFrom::Write,
                                ScratchTo::LastRead,
                                Some(choice_var),
                            );
                            scratch_intervals.push(interval);
                            scratch_demands.push(8);
                        }
                    }
                    SelectChoiceMode::ForceVselect => {}
                }
            }

            // CSE shared sub_temp intervals: one per active group, live [sub_cycle, max_fma_cycle+1]
            // where max_fma_cycle is the maximum FMA cycle among consumers (FMA writes at cycle+1).
            // Model end conservatively as horizon+1 (will be tightened by actual FMA cycles in solution).
            if let Some(groups) = cse_groups {
                for (g, group) in groups.iter().enumerate() {
                    let sub_active_var = cse_subs[g].active();
                    let sub_cycle_var = cse_subs[g].cycle();
                    
                    // End cycle: max FMA cycle + 1 among consumers that chose FMA.
                    // FMA cycle for consumer idx is its issue cycle + 1 when FMA is chosen.
                    // Model conservatively: use horizon+1 as upper bound; actual end will be <= this.
                    let end_cycle_var = model.new_int_var_with_name(
                        [(0, horizon + 2)],
                        format!("cse_sub_end_{}", g),
                    );
                    // Constraint: end >= sub_cycle + 1 (at least 1 cycle lifetime)
                    let c_min = model.add_ge(
                        LinearExpr::from(end_cycle_var),
                        LinearExpr::from(sub_cycle_var) + 1,
                    );
                    model.set_constraint_name(c_min, &format!("cse_sub_end_ge_sub_plus_1_{}", g));
                    // Constraint: end >= max(issue[consumer] + 2) for consumers that chose FMA
                    // (FMA writes at cycle+1, so last read is at cycle+1, interval end is cycle+2)
                    for &consumer_idx in &group.consumer_indices {
                        if consumer_idx >= times.len() {
                            continue;
                        }
                        match select_choice_mode {
                            SelectChoiceMode::ForceFma => {
                                let fma_end = times[consumer_idx].last_read_end_on_path(true);
                                let c = model.add_ge(
                                    LinearExpr::from(end_cycle_var),
                                    fma_end,
                                );
                                model.set_constraint_name(c, &format!("cse_sub_end_ge_fma_{}_c{}", g, consumer_idx));
                            }
                            SelectChoiceMode::Optimize => {
                                if let Some(&choice_var) = select_choice_vars.get(&consumer_idx) {
                                    let fma_end = times[consumer_idx].last_read_end_on_path(true);
                                    // Only enforce when FMA is chosen (sub_active implies at least one FMA)
                                    let c = model.add_ge(
                                        LinearExpr::from(end_cycle_var),
                                        fma_end,
                                    );
                                    model.only_enforce_if(c, [sub_active_var, choice_var]);
                                    model.set_constraint_name(c, &format!("cse_sub_end_ge_fma_opt_{}_c{}", g, consumer_idx));
                                }
                            }
                            SelectChoiceMode::ForceVselect => {}
                        }
                    }
                    
                    // Add scratch interval: [sub_cycle, end_cycle) when sub_active
                    let size_var = model.new_int_var_with_name(
                        [(1, horizon + 2)],
                        format!("cse_sub_size_{}", g),
                    );
                    let c_size = model.add_eq(
                        LinearExpr::from(sub_cycle_var) + LinearExpr::from(size_var),
                        LinearExpr::from(end_cycle_var),
                    );
                    model.set_constraint_name(c_size, &format!("cse_sub_size_eq_{}", g));
                    cse_sub_scratch_vars.push((end_cycle_var, size_var));
                    // Only enforce size constraint when sub_active
                    model.only_enforce_if(c_size, [sub_active_var]);
                    
                    let interval = scratch_interval(
                        &mut model,
                        scratch_time,
                        LinearExpr::from(sub_cycle_var),
                        LinearExpr::from(size_var),
                        LinearExpr::from(end_cycle_var),
                        ScratchFrom::Write,
                        ScratchTo::LastRead,
                        Some(sub_active_var),
                    );
                    scratch_intervals.push(interval);
                    scratch_demands.push(8); // Vector VReg (8 words)
                }
            }

            // The cumulative rejects mandatory intervals wider than the scratch budget.
            if scratch_pools {
                // Two pools: vectors (demand VLEN) and scalars (demand 1), with capacities
                // N_v and N_s and 8 * N_v + N_s <= budget. Address assignment needs at most
                // 8 * (most live vectors) + (most live scalars) words.
                let mut vec_intervals: Vec<IntervalVar> = Vec::new();
                let mut scalar_intervals: Vec<IntervalVar> = Vec::new();
                for (iv, d) in scratch_intervals.iter().zip(scratch_demands.iter()) {
                    if *d == VLEN as i64 {
                        vec_intervals.push(*iv);
                    } else {
                        debug_assert_eq!(*d, 1, "pools: demand {} is neither 1 nor {}", d, VLEN);
                        scalar_intervals.push(*iv);
                    }
                }
                let budget = scratch_budget as i64;
                let n_v = model.new_int_var_with_name([(0, budget / VLEN as i64)], "scratch_pool_vectors");
                let n_s = model.new_int_var_with_name([(0, budget)], "scratch_pool_scalars");
                let ones_v = vec![1i64; vec_intervals.len()];
                let ones_s = vec![1i64; scalar_intervals.len()];
                let c = model.add_cumulative(vec_intervals, ones_v, n_v);
                model.set_constraint_name(c, "cumul_scratch_vectors");
                let c = model.add_cumulative(scalar_intervals, ones_s, n_s);
                model.set_constraint_name(c, "cumul_scratch_scalars");
                let c = model.add_le(
                    LinearExpr::from(n_v) * VLEN as i64 + LinearExpr::from(n_s),
                    LinearExpr::from(budget),
                );
                model.set_constraint_name(c, "scratch_pools_capacity");
                scratch_pool_vars = Some((n_v, n_s));
            } else {
                let c = model.add_cumulative(scratch_intervals, scratch_demands, scratch_budget as i64);
                model.set_constraint_name(c, "cumul_scratch");
            }

            // Add hints for size variables if we have greedy hints
            if let Some(hints) = greedy_hints {
                let max_hint = hinted_segment_finish.unwrap_or(0);
                // segment_end_var is already hinted where it's created (line ~3633); don't duplicate.
                let mut bad_size_count = 0;
                for vreg in &live_vregs {
                    let is_live_out = live_out_vregs.is_some_and(|s| s.contains(vreg));
                    if is_live_out {
                        continue; // live-out uses segment end, not last_use_idx
                    }
                    let def_idx = vreg_def_op[vreg];
                    let last_use_idx = vreg_last_use[vreg];
                    if def_idx < hints.len() && last_use_idx < hints.len() {
                        let def_cycle = hints[def_idx];
                        let last_use_cycle = hints[last_use_idx];
                        let expected_size = last_use_cycle - def_cycle + 1;
                        if expected_size < 1 {
                            bad_size_count += 1;
                            if bad_size_count <= 5 && verbose {
                                // Get more info about the problematic VReg
                                let def_op = &ops[def_idx];
                                let use_op = &ops[last_use_idx];
                                eprintln!(
                                    "  BUG: VReg {:?} (size={}): def by op {} (id={}, {:?}) at cycle {}, \
                                     but last_use by op {} (id={}, {:?}) at cycle {}. \
                                     Use depends on def? {:?}",
                                    vreg, vreg.size(), def_idx, def_op.id, def_op.engine, def_cycle,
                                    last_use_idx, use_op.id, use_op.engine, last_use_cycle,
                                    use_op.dep_ids.contains(&def_op.id)
                                );
                            }
                        }
                    }
                }

                // Now add size hints (live-out VRegs: use segment end = max_hint)
                for (i, (size_var, def_idx, last_use_idx)) in size_vars.iter().enumerate() {
                    let vreg = &size_var_vregs[i];
                    let is_live_out = live_out_vregs.is_some_and(|s| s.contains(vreg));
                    if *def_idx >= hints.len() {
                        continue;
                    }
                    // A split shift reserves its destination from its first lane.
                    let def_cycle = hinted_first_write(*def_idx).unwrap_or(hints[*def_idx]);
                    let expected_size = if is_live_out {
                        (max_hint - def_cycle + 1).max(1)
                    } else {
                        if *last_use_idx >= hints.len() {
                            continue;
                        }
                        let end = hinted_live_end
                            .get(vreg)
                            .copied()
                            .unwrap_or(hints[*last_use_idx] + 1);
                        (end - def_cycle).max(1)
                    };
                    model.add_hint(*size_var, expected_size as i64);
                }

                if bad_size_count > 0 && verbose {
                    // This can happen when merging section schedules with different local cycle spaces.
                    // The hint is just a starting point - if it's invalid, the solver will find a valid schedule.
                    eprintln!(
                        "  Note: {} VRegs have last_use_cycle < def_cycle in hints. Solver will recompute valid cycles.",
                        bad_size_count
                    );
                }
            }

            // CRITICAL: Verify hints are feasible before solving
            // If hints are infeasible, there's a bug in our model - fail immediately
            if let Some(hints) = greedy_hints {
                let max_hint = hinted_segment_finish.unwrap_or(0);
                // Compute peak liveness from hints (live-out VRegs extend to segment end = max_hint)
                let mut events: Vec<(i32, i64)> = Vec::new();
                // Same timeline as the cumulative (`scratch_interval`): times of a start at a
                // write / the segment start and of an end after a last read / the segment end.
                let (at_write, at_seg_start, after_read, after_seg) = match scratch_time {
                    ScratchTime::Cycle => (
                        (|c: i32| c) as fn(i32) -> i32,
                        (|c: i32| c) as fn(i32) -> i32,
                        (|e: i32| e) as fn(i32) -> i32,
                        (|e: i32| e) as fn(i32) -> i32,
                    ),
                    ScratchTime::Phase => (
                        (|c: i32| c) as fn(i32) -> i32,
                        (|c: i32| c - 1) as fn(i32) -> i32,
                        (|e: i32| e - 1) as fn(i32) -> i32,
                        (|e: i32| e) as fn(i32) -> i32,
                    ),
                };
                for &(vreg, end) in &incoming_hinted_ends {
                    let size = vreg.size() as i64;
                    events.push((at_seg_start(0), size));
                    events.push((after_read(end), -size));
                }
                if let Some(lt) = live_through_vregs {
                    for vreg in lt {
                        let size = vreg.size() as i64;
                        events.push((at_seg_start(0), size));
                        events.push((after_seg(max_hint + 1), -size));
                    }
                }
                for vreg in &live_vregs {
                    let def_idx = vreg_def_op[vreg];
                    let last_use_idx = vreg_last_use[vreg];
                    let def_cycle = hinted_first_write(def_idx).unwrap_or(0);
                    let live_end = if live_out_vregs.is_some_and(|s| s.contains(vreg)) {
                        after_seg(max_hint + 1)
                    } else {
                        after_read(
                            hinted_live_end
                                .get(vreg)
                                .copied()
                                .unwrap_or(hints.get(last_use_idx).copied().unwrap_or(def_cycle) + 1),
                        )
                    };
                    let size = vreg.size() as i64;
                    events.push((at_write(def_cycle), size)); // VReg becomes live
                    events.push((live_end, -size)); // VReg dies after its last read
                }
                // Add live_out VRegs that are defined in segment but not used within segment
                if let Some(lo) = live_out_vregs {
                    for vreg in lo {
                        if !live_vregs.contains(vreg) {
                            // This VReg is defined in segment but only used after
                            if let Some(&def_idx) = vreg_def_op.get(vreg) {
                                let def_cycle = hinted_first_write(def_idx).unwrap_or(0);
                                let size = vreg.size() as i64;
                                events.push((at_write(def_cycle), size)); // VReg becomes live
                                events.push((after_seg(max_hint + 1), -size)); // VReg dies at segment end
                            }
                        }
                    }
                }
                // SelectChoice FMA sub_temp: live [c, c+2)
                if let Some(fma_ids) = select_choice_hints {
                    for (op_idx, op) in ops.iter().enumerate() {
                        if op.opcode == Opcode::SelectChoice
                            && fma_ids.contains(&op.id)
                            && op.sub_temp_vreg.is_some()
                            && op_idx < hints.len()
                        {
                            let c = hints[op_idx];
                            events.push((at_write(c), 8));
                            events.push((after_read(c + 2), -8));
                        }
                    }
                }
                events.sort_by_key(|(cycle, delta)| (*cycle, *delta));

                let mut current = 0i64;
                let mut peak = 0i64;
                for (_, delta) in &events {
                    current += delta;
                    peak = peak.max(current);
                }

                if let Some((n_v, n_s)) = scratch_pool_vars {
                    // Hint N_v, N_s with the hint schedule's pool maxima: the same intervals,
                    // write-group lanes replaced by their group.
                    let mut pool_events: Vec<(i32, i64, i64)> = Vec::new(); // (cycle, dv, ds)
                    let mut push = |start: i32, end: i32, vector: bool| {
                        let (dv, ds) = if vector { (1, 0) } else { (0, 1) };
                        pool_events.push((start, dv, ds));
                        pool_events.push((end, -dv, -ds));
                    };
                    for &(vreg, end) in &incoming_hinted_ends {
                        push(at_seg_start(0), after_read(end), vreg.is_vector());
                    }
                    if let Some(lt) = live_through_vregs {
                        for vreg in lt {
                            push(at_seg_start(0), after_seg(max_hint + 1), vreg.is_vector());
                        }
                    }
                    for vreg in &live_vregs {
                        if pool_group_of_lane.contains_key(vreg) {
                            continue;
                        }
                        let def_idx = vreg_def_op[vreg];
                        let last_use_idx = vreg_last_use[vreg];
                        let def_cycle = hinted_first_write(def_idx).unwrap_or(0);
                        let live_end = if live_out_vregs.is_some_and(|s| s.contains(vreg)) {
                            after_seg(max_hint + 1)
                        } else {
                            after_read(
                                hinted_live_end
                                    .get(vreg)
                                    .copied()
                                    .unwrap_or(hints.get(last_use_idx).copied().unwrap_or(def_cycle) + 1),
                            )
                        };
                        push(at_write(def_cycle), live_end, vreg.is_vector());
                    }
                    if let Some(lo) = live_out_vregs {
                        for vreg in lo {
                            if !pool_group_of_lane.contains_key(vreg) && !live_vregs.contains(vreg) {
                                if let Some(&def_idx) = vreg_def_op.get(vreg) {
                                    push(
                                        at_write(hinted_first_write(def_idx).unwrap_or(0)),
                                        after_seg(max_hint + 1),
                                        vreg.is_vector(),
                                    );
                                }
                            }
                        }
                    }
                    for &(s, e) in &pool_group_hints {
                        push(at_write(s), after_read(e), true);
                    }
                    if let Some(fma_ids) = select_choice_hints {
                        for (op_idx, op) in ops.iter().enumerate() {
                            if op.opcode == Opcode::SelectChoice
                                && fma_ids.contains(&op.id)
                                && op.sub_temp_vreg.is_some()
                                && op_idx < hints.len()
                            {
                                push(at_write(hints[op_idx]), after_read(hints[op_idx] + 2), true);
                            }
                        }
                    }
                    pool_events.sort_by_key(|&(c, dv, ds)| (c, dv + ds));
                    let (mut v, mut s, mut max_v, mut max_s) = (0i64, 0i64, 0i64, 0i64);
                    for (_, dv, ds) in &pool_events {
                        v += dv;
                        s += ds;
                        max_v = max_v.max(v);
                        max_s = max_s.max(s);
                    }
                    let budget = scratch_budget as i64;
                    model.add_hint(n_v, max_v.min(budget / VLEN as i64));
                    model.add_hint(n_s, max_s.min(budget));
                    if verbose || VLEN as i64 * max_v + max_s > budget {
                        eprintln!(
                            "  Scratch pools in the hint: 8*{} + {} = {} words (budget {})",
                            max_v,
                            max_s,
                            VLEN as i64 * max_v + max_s,
                            budget
                        );
                    }
                }
                if peak > scratch_budget as i64 {
                    // This is expected when greedy scheduler exceeded budget (force-scheduled).
                    // CP-SAT hints are "soft" - they're suggestions, not requirements.
                    // CP-SAT will use the hint as a starting point but may ignore infeasible parts.
                    eprintln!(
                        "Note: Greedy hint exceeds budget (peak_live={} > budget={}). \
                         CP-SAT will use hint as starting point but may ignore infeasible parts.",
                        peak, scratch_budget
                    );
                } else if verbose {
                    eprintln!(
                        "  Hint verification: peak_live={} <= budget={} OK",
                        peak, scratch_budget
                    );
                }
            }

            if verbose {
                eprintln!(
                    "  Scratch constraint: {} live VRegs, budget={}",
                    live_vregs.len(),
                    scratch_budget
                );
            }
        }
    }

    // Verify engine capacity constraints from greedy hints (VALU, ALU, Load, Store, Flow)
    if let Some(hints) = greedy_hints {
        // Compute per-cycle usage
        let max_cycle = *hints.iter().max().unwrap_or(&0) as usize + 1;
        let mut valu_per_cycle = vec![0i32; max_cycle];
        let mut alu_per_cycle = vec![0i32; max_cycle];
        let mut load_per_cycle = vec![0i32; max_cycle];
        let mut store_per_cycle = vec![0i32; max_cycle];
        let mut flow_per_cycle = vec![0i32; max_cycle];

        for (idx, op) in ops.iter().enumerate() {
            if idx >= hints.len() {
                continue;
            }
            let cycle = hints[idx] as usize;
            if cycle >= max_cycle {
                continue;
            }

            let is_shifted = if let Some(shift_hints) = shift_hints {
                shift_hints.contains(&op.id)
            } else {
                false
            };

            // ConstChoice: hinted as Load by default (greedy treats them as Load)
            let is_const_add_imm = const_add_imm_hints
                .map_or(false, |h| h.contains(&op.id));
            // SelectChoice: hinted as vselect (Flow) by default
            let is_select_fma = select_choice_hints
                .map_or(false, |h| h.contains(&op.id));

            if op.opcode == Opcode::ConstChoice {
                if is_const_add_imm {
                    flow_per_cycle[cycle] += 1; // AddImm uses Flow
                } else {
                    load_per_cycle[cycle] += 1; // Load uses Load
                }
            } else if op.opcode == Opcode::SelectChoice {
                if is_select_fma {
                    valu_per_cycle[cycle] += 1; // FMA uses VALU
                } else {
                    flow_per_cycle[cycle] += 1; // vselect uses Flow
                }
            } else {
                match op.engine {
                    Engine::Valu if !is_shifted => {
                        valu_per_cycle[cycle] += 1;
                    }
                    Engine::Valu if is_shifted => match hinted_lanes(op.id) {
                        // Split shift: each lane takes one ALU slot in its own cycle.
                        Some(lanes) => {
                            for &lane_cycle in lanes {
                                if let Some(used) = alu_per_cycle.get_mut(lane_cycle.max(0) as usize) {
                                    *used += 1;
                                }
                            }
                        }
                        // Whole shift: 8 ALU slots in the op's cycle.
                        None => alu_per_cycle[cycle] += VLEN as i32,
                    },
                    Engine::Alu => {
                        alu_per_cycle[cycle] += 1;
                    }
                    Engine::Load => {
                        load_per_cycle[cycle] += 1;
                    }
                    Engine::Store => {
                        store_per_cycle[cycle] += 1;
                    }
                    Engine::Flow => {
                        flow_per_cycle[cycle] += 1;
                    }
                    _ => {}
                }
            }
        }

        // Check for violations
        let mut valu_violations = 0;
        let mut alu_violations = 0;
        let mut load_violations = 0;
        let mut store_violations = 0;
        let mut flow_violations = 0;
        for cycle in 0..max_cycle {
            if (valu_per_cycle[cycle] as i64) > valu_cap {
                if valu_violations < 3 && verbose {
                    eprintln!(
                        "  WARNING: VALU overflow at cycle {}: {} > {}",
                        cycle, valu_per_cycle[cycle], valu_cap
                    );
                }
                valu_violations += 1;
            }
            if (alu_per_cycle[cycle] as i64) > alu_cap {
                if alu_violations < 3 && verbose {
                    eprintln!(
                        "  WARNING: ALU overflow at cycle {}: {} > {}",
                        cycle, alu_per_cycle[cycle], alu_cap
                    );
                }
                alu_violations += 1;
            }
            if (load_per_cycle[cycle] as i64) > load_cap {
                if load_violations < 3 && verbose {
                    eprintln!(
                        "  WARNING: Load overflow at cycle {}: {} > {}",
                        cycle, load_per_cycle[cycle], load_cap
                    );
                }
                load_violations += 1;
            }
            if (store_per_cycle[cycle] as i64) > store_cap {
                if store_violations < 3 && verbose {
                    eprintln!(
                        "  WARNING: Store overflow at cycle {}: {} > {}",
                        cycle, store_per_cycle[cycle], store_cap
                    );
                }
                store_violations += 1;
            }
            if (flow_per_cycle[cycle] as i64) > flow_cap {
                if flow_violations < 3 && verbose {
                    eprintln!(
                        "  WARNING: Flow overflow at cycle {}: {} > {}",
                        cycle, flow_per_cycle[cycle], flow_cap
                    );
                }
                flow_violations += 1;
            }
        }

        let total_violations = valu_violations + alu_violations + load_violations + store_violations + flow_violations;
        if total_violations > 0 {
            if valu_violations > 0 || alu_violations > 0 {
                // VALU/ALU violations are bugs in merge/rebase — panic.
                panic!(
                    "Merged hint violates engine capacity (VALU: {} cycles, ALU: {} cycles). \
                     Slot limits must never be exceeded; scratch may be exceeded in greedy, but not VALU/ALU.",
                    valu_violations, alu_violations
                );
            }
            // Load/Store/Flow violations are less critical — warn but don't panic.
            // The CP-SAT model will detect infeasible hints and search from scratch.
            eprintln!(
                "  WARNING: Hint violates engine capacity (Load: {} cycles, Store: {} cycles, Flow: {} cycles). \
                 Hint will be infeasible in CP-SAT model.",
                load_violations, store_violations, flow_violations
            );
        }

        if verbose {
            eprintln!(
                "  Engine capacity verification: OK (VALU<={}, ALU<={}, Load<={}, Store<={}, Flow<={} at all cycles)",
                valu_cap, alu_cap, load_cap, store_cap, flow_cap
            );
        }
    }

    // Check precedence constraints against hints
    if let Some(hints) = greedy_hints {
        let mut prec_violations = 0;
        for (idx, op) in ops.iter().enumerate() {
            if idx >= hints.len() || hints[idx] < 0 {
                continue;
            }
            let consumer_cycle = hints[idx];
            for &dep_id in &op.dep_ids {
                if let Some(&dep_idx) = op_id_to_idx.get(&dep_id) {
                    if dep_idx >= hints.len() || hints[dep_idx] < 0 {
                        continue;
                    }
                    let producer_cycle = hints[dep_idx];
                    // Basic precedence: consumer >= producer + 1
                    if consumer_cycle < producer_cycle + 1 {
                        prec_violations += 1;
                        if prec_violations <= 5 && verbose {
                            eprintln!(
                                "  WARNING: Precedence violation: op {} (id={}) at cycle {} depends on op {} (id={}) at cycle {} (need >= {})",
                                idx, op.id, consumer_cycle, dep_idx, ops[dep_idx].id, producer_cycle, producer_cycle + 1
                            );
                        }
                    }
                }
            }
            // Memory edges: consumer >= source + latency (0 for WAR)
            for &(src_id, kind) in &op.mem_deps {
                let Some(&src_idx) = op_id_to_idx.get(&src_id) else {
                    continue;
                };
                if src_idx >= hints.len() || hints[src_idx] < 0 {
                    continue;
                }
                if consumer_cycle < hints[src_idx] + kind.latency() {
                    prec_violations += 1;
                    if prec_violations <= 5 && verbose {
                        eprintln!(
                            "  WARNING: Memory {} edge violation: op {} (id={}) at cycle {} after op {} (id={}) at cycle {} (need >= {})",
                            kind.as_str(), idx, op.id, consumer_cycle, src_idx, src_id, hints[src_idx],
                            hints[src_idx] + kind.latency()
                        );
                    }
                }
            }
        }
        if prec_violations > 0 && verbose {
            eprintln!("  Total precedence violations: {}", prec_violations);
        } else if verbose {
            eprintln!("  Precedence verification: OK (all consumer >= producer + 1)");
        }
    }

    // 9. Compute capacity-based lower bounds via global min over (x, y, z)
    let valu_count = ops.iter().filter(|op| op.engine == Engine::Valu).count() as i64;
    let alu_count = ops.iter().filter(|op| op.engine == Engine::Alu).count() as i64;
    let load_count = ops.iter().filter(|op| op.engine == Engine::Load).count() as i64;
    let store_count = ops.iter().filter(|op| op.engine == Engine::Store).count() as i64;
    let shiftable_count = shiftable_set.len() as i64;
    let const_choice_count = const_choice_ops.len() as i64;
    let select_choice_count = select_choice_ops.len() as i64;
    let flow_base = ops.iter().filter(|op| op.engine == Engine::Flow).count() as i64;

    // Dependency floor = 0 on purpose: critical_path_len (max ASAP + 1) assumes +1 per edge, but the
    // model uses variable-length steps (completion_cycle for shifted VALU, +2 for SelectChoice FMA),
    // so critical_path_len can overestimate. If we used it as the makespan domain LB we would
    // exclude feasible solutions (make them infeasible). The capacity-derived minimum over (x,y,z) is
    // a valid lower bound; precedence constraints enforce the true dependency lower bound, so the
    // solver cannot find makespan below the actual minimum.
    let cap_result = compute_capacity_lower_bound_global(
        valu_count,
        alu_count,
        load_count,
        store_count,
        flow_base,
        shiftable_count,
        const_choice_count,
        select_choice_count,
        enable_valu_shift,
        valu_cap,
        alu_cap,
        load_cap,
        store_cap,
        flow_cap,
        0i64, // dependency floor: don't inflate capacity_lower_bound with ASAP-based critical path
        cse_groups,
    );
    let capacity_lower_bound = cap_result.capacity_lower_bound;
    let mut valu_alu_global_min = cap_result.valu_alu_global_min;
    let load_bound = cap_result.load_bound;
    let store_bound = cap_result.store_bound;

    if verbose {
        eprintln!(
            "  Capacity bounds (conservative): shiftable={}, const_choice={}; Load={}, Store={}, final_LB={}",
            shiftable_count,
            const_choice_count,
            load_bound,
            store_bound,
            capacity_lower_bound
        );
    }

    // Sanity: when greedy is within scratch budget, it must respect capacity lower bound (infeasible greedy can use fewer cycles).
    let greedy_ok = greedy_within_scratch_budget == Some(true);
    if greedy_ok {
        if let Some(h) = greedy_hints {
            let greedy_makespan = (h.iter().max().copied().unwrap_or(0) + 1) as i64;
            if greedy_makespan < capacity_lower_bound {
                panic!(
                    "greedy makespan {} < capacity_lower_bound {} (greedy schedule would violate capacity; \
                     capacity_lb is a valid lower bound so this indicates a bug in greedy or in capacity_lb)",
                    greedy_makespan, capacity_lower_bound
                );
            }
        }
    }

    // 10. Create makespan and max_cycle variables; tie makespan to actual last cycle so CP-SAT minimizes it.
    // max_cycle = max over every issue cycle, completion cycle and multiply_add cycle.
    // makespan = max_cycle + 1 (so minimizing makespan forces packing).
    // capacity_lower_bound is a valid lower bound (min over (x,y,z) of per-choice capacity bound),
    // so it never overestimates the true minimum and does not exclude feasible makespans. We pass 0
    // for the dependency floor so the domain is not inflated by ASAP-based critical path (that
    // would overestimate and incorrectly exclude feasible solutions).
    let max_cycle_lb = (capacity_lower_bound - 1).max(0);
    let max_cycle =
        model.new_int_var_with_name([(max_cycle_lb, (horizon - 1) as i64)], "max_cycle".to_string());
    let makespan =
        model.new_int_var_with_name([(capacity_lower_bound, horizon)], "makespan".to_string());
    let c = model.add_eq(LinearExpr::from(makespan), LinearExpr::from(max_cycle) + 1);
    model.set_constraint_name(c, "makespan_eq_max_cycle");
    for (cv_idx, t) in times.iter().enumerate() {
        let c = model.add_ge(max_cycle, LinearExpr::from(t.issue()));
        model.set_constraint_name(c, &format!("max_cycle_ge_c_{}", cv_idx));
    }
    for (idx, completion_var) in times.iter().enumerate().filter_map(|(i, t)| t.completion().map(|v| (i, v))) {
        let c = model.add_ge(max_cycle, LinearExpr::from(completion_var));
        model.set_constraint_name(c, &format!("max_cycle_ge_completion_{}", idx));
    }
    for (idx, fma_last) in times.iter().enumerate().filter_map(|(i, t)| t.fma_cycle().map(|v| (i, v))) {
        let c = model.add_ge(max_cycle, LinearExpr::from(fma_last));
        model.set_constraint_name(c, &format!("max_cycle_ge_fma_last_{}", idx));
    }

    // 10a. Window optimality probe (analysis only, hints.window): hold every op outside the
    // window at its reference cycle (ops after it move as one block) and keep the ops inside
    // it in the window. See cpsat_scheduler::window.
    if let Some(window) = window {
        let counts = super::window::apply_window_constraints(
            &mut model,
            ops,
            window,
            &op_vars,
            &const_choice_alu_encoding_info,
        );
        if verbose {
            eprintln!(
                "  Window [{}, {}): {} ops before (fixed), {} inside (free{}), {} after (move as one block), {} free",
                window.start,
                window.start + window.len,
                counts.before,
                counts.inside,
                if window.fix_window_encodings { ", encodings fixed" } else { "" },
                counts.after,
                counts.free
            );
        }
    }

    // 10b. Makespan interval (tuning.makespan_interval): one interval [makespan, horizon) that
    // takes the whole capacity of every engine cumulative, as in OR-Tools' rcpsp_sat.py. It
    // excludes nothing (every interval of those cumulatives is present only for work that runs
    // at a cycle <= max_cycle, so it ends by makespan); it lets CP-SAT's linear relaxation find
    // the makespan of each cumulative and tie the energy rows and cuts to it. The scratch
    // cumulative is left alone: live intervals may end after the makespan.
    let makespan_tail_size: Option<IntVar> = if tuning.makespan_interval {
        let tail_size = model.new_int_var_with_name(
            [(0, horizon - capacity_lower_bound)],
            "makespan_tail_size".to_string(),
        );
        let tail = model.new_interval_var(makespan, tail_size, horizon);
        let mut attached = 0usize;
        for (name, cap) in [
            ("cumul_valu", valu_cap),
            ("cumul_alu", alu_cap),
            ("cumul_load", load_cap),
            ("cumul_store", store_cap),
            ("cumul_flow", flow_cap),
        ] {
            for constraint in model.proto_mut().constraints.iter_mut().filter(|c| c.name == name) {
                if let Some(cp_sat::proto::constraint_proto::Constraint::Cumulative(cumulative)) =
                    constraint.constraint.as_mut()
                {
                    cumulative.intervals.push(tail.0);
                    cumulative.demands.push(cp_sat::proto::LinearExpressionProto {
                        vars: vec![],
                        coeffs: vec![],
                        offset: cap,
                    });
                    attached += 1;
                }
            }
        }
        if verbose {
            eprintln!("  Makespan interval [makespan, {}) added to {} engine cumulatives", horizon, attached);
        }
        Some(tail_size)
    } else {
        None
    };

    // Redundant linear capacity constraints: only add bounds that are valid for *any* choice.
    // capacity_lower_bound (makespan domain) is already the min over (x,y,z) from compute_capacity_lower_bound_global.
    // Do NOT add makespan >= ceil(alu_count+8*shiftable/12) or makespan >= ceil(valu_count-shiftable/6): those are
    // bounds for one extreme (all shift / no shift) and would over-constrain when the solver picks a different choice
    // (e.g. shift 163 only → ALU need 121 cycles, but "all shift" bound would force makespan >= 609).
    // Load and store bounds are valid global mins (min over const_choice for load; store is fixed).
    // Optional group RedundantLinearBounds: redundant with makespan domain (T15; propagation-only).
    if !disable_optional_constraint_groups.contains(&OptionalConstraintGroup::RedundantLinearBounds) {
        if load_bound > 0 {
            let load_lb = model.new_int_var([(load_bound, load_bound)]);
            let c = model.add_ge(makespan, LinearExpr::from(load_lb));
            model.set_constraint_name(c, "makespan_ge_load_lb");
        }
        if store_bound > 0 {
            let store_lb = model.new_int_var([(store_bound, store_bound)]);
            let c = model.add_ge(makespan, LinearExpr::from(store_lb));
            model.set_constraint_name(c, "makespan_ge_store_lb");
        }
    }

    // Direct capacity constraints: link makespan to choice variables (x = shift, y = ConstChoice AddImm, z = SelectChoice FMA count)
    // so CP-SAT can optimize over (x,y,z) during search instead of relying only on the pre-computed capacity_lower_bound.
    // VALU uses CSE-aware formula when cse_groups present; else 2*sum(select_fma). Flow/Load/ALU unchanged.
    {
        let zero_var = model.new_int_var([(0, 0)]);
        let sum_shift: Option<LinearExpr> = shiftable_valu
            .iter()
            .filter_map(|&idx| shift_vars.get(&idx).copied())
            .fold(None, |acc, b| {
                Some(match acc {
                    None => LinearExpr::from(b),
                    Some(a) => a + LinearExpr::from(b),
                })
            });
        let sum_select_fma: Option<LinearExpr> = select_choice_ops
            .iter()
            .filter_map(|&idx| select_choice_vars.get(&idx).copied())
            .fold(None, |acc, b| {
                Some(match acc {
                    None => LinearExpr::from(b),
                    Some(a) => a + LinearExpr::from(b),
                })
            });
        let sum_const_add_imm: Option<LinearExpr> = const_choice_ops
            .iter()
            .filter_map(|&idx| const_choice_vars.get(&idx).copied())
            .fold(None, |acc, b| {
                Some(match acc {
                    None => LinearExpr::from(b),
                    Some(a) => a + LinearExpr::from(b),
                })
            });
        let s = sum_shift.unwrap_or_else(|| LinearExpr::from(zero_var));
        let z = sum_select_fma.unwrap_or_else(|| LinearExpr::from(zero_var));
        let y = sum_const_add_imm.unwrap_or_else(|| LinearExpr::from(zero_var));

        // Direct capacity linear constraints: optional groups per engine. Redundancy is PROVEN in
        // LIVENESS_SPECIFICATION.md T15: cumulative implies these inequalities (proof uses T14).
        // So every solution feasible without them is feasible with them; propagation-only.
        // Formulas must match the cumulative: VALU ops = valu_count - s + select_valu_contrib (s = number shifted to ALU);
        // ALU ops = alu_count + 8*(shiftable_count - s). So 6*makespan >= VALU_ops and 12*makespan >= ALU_ops.
        if !disable_optional_constraint_groups.contains(&OptionalConstraintGroup::CapacityValu)
            && enable_valu_shift
            && (valu_count > 0 || shiftable_count > 0 || select_choice_count > 0)
        {
            let select_valu_contrib: LinearExpr = if let Some(groups) = cse_groups {
                if groups.is_empty() {
                    z.clone() * 2
                } else {
                    let mut c = LinearExpr::from(zero_var);
                    for (g, group) in groups.iter().enumerate() {
                        c = c + LinearExpr::from(cse_subs[g].active());
                        for &idx in &group.consumer_indices {
                            if let Some(&cv) = select_choice_vars.get(&idx) {
                                c = c + LinearExpr::from(cv);
                            }
                        }
                    }
                    for &idx in &select_choice_ops {
                        if !cse_consumer_set.contains(&idx) {
                            if let Some(&cv) = select_choice_vars.get(&idx) {
                                c = c + LinearExpr::from(cv) * 2;
                            }
                        }
                    }
                    c
                }
            } else {
                z.clone() * 2
            };
            // 6*makespan >= valu_count - s + select_valu_contrib (s = number shifted to ALU; redundant with cumul_valu per T15)
            let lhs = LinearExpr::from(makespan) * 6 + s.clone() - select_valu_contrib;
            let rhs = model.new_int_var([(valu_count, valu_count)]);
            let c = model.add_ge(lhs, LinearExpr::from(rhs));
            model.set_constraint_name(c, "capacity_valu");
        }
        // ALU: 12*makespan >= alu_count + 8*s (s = number shifted to ALU; redundant with cumul_alu per T15)
        if !disable_optional_constraint_groups.contains(&OptionalConstraintGroup::CapacityAlu)
            && enable_valu_shift
            && (alu_count > 0 || shiftable_count > 0)
        {
            let alu_const = model.new_int_var([(alu_count, alu_count)]);
            let lhs = LinearExpr::from(makespan) * 12 - s.clone() * 8;
            let c = model.add_ge(lhs, LinearExpr::from(alu_const));
            model.set_constraint_name(c, "capacity_alu");
        }
        // Flow: makespan >= flow_base + sum(const_add_imm) - sum(select_fma)  =>  makespan + z - y >= flow_base
        if !disable_optional_constraint_groups.contains(&OptionalConstraintGroup::CapacityFlow)
            && (flow_base > 0 || const_choice_count > 0 || select_choice_count > 0)
        {
            let lhs = LinearExpr::from(makespan) + z.clone() - y.clone();
            let rhs = model.new_int_var([(flow_base, flow_base)]);
            let c = model.add_ge(lhs, LinearExpr::from(rhs));
            model.set_constraint_name(c, "capacity_flow");
        }
        // Load: 2*makespan >= load_count - sum(const_add_imm)
        if !disable_optional_constraint_groups.contains(&OptionalConstraintGroup::CapacityLoad)
            && load_count > 0
            && const_choice_count > 0
        {
            let lhs = LinearExpr::from(makespan) * 2 + y;
            let rhs = model.new_int_var([(load_count, load_count)]);
            let c = model.add_ge(lhs, LinearExpr::from(rhs));
            model.set_constraint_name(c, "capacity_load");
        }
    }

    // makespan >= max(chosen_cycle[i]) + 1: for each op use the cycle of the chosen execution path.
    // Shiftable: when shifted use completion_cycle, when not the issue cycle (OpTimes::finish_alternatives).
    // SelectChoice: when FMA (no-delta) use fma_cycle_var (last cycle), else the issue cycle. ConstChoice: every path ends at the issue cycle.
    // Optional group ConditionalMakespan: every one of these rows is implied by
    // makespan = max_cycle + 1 with max_cycle >= issue, completion and multiply_add cycles above
    // (those are unconditional, so they are at least as strong as the enforced rows here).
    let conditional_makespan_rows =
        !disable_optional_constraint_groups.contains(&OptionalConstraintGroup::ConditionalMakespan);
    for (idx, t) in times.iter().enumerate().filter(|_| conditional_makespan_rows) {
        // One row per finish alternative (op_times.rs), enforced by the literal selecting it.
        for alternative in t.finish_alternatives() {
            let c = model.add_ge(makespan, alternative.value + 1);
            if let Some(when) = alternative.when {
                model.only_enforce_if(c, [when]);
            }
            model.set_constraint_name(c, &format!("makespan_ge_{}_{}", alternative.label, idx));
        }
    }

    // makespan = max_cycle + 1 and max_cycle >= every cycle var already force all work into [0, max_cycle];
    // no need for explicit cycle <= makespan-1.
    // We do NOT add makespan >= critical_path_len: dependency is enforced only by precedence
    // (consumer_cycle >= producer_completion + offset), so the true dependency bound is implied and does not overestimate.

    // CANONICAL DESCRIPTION (docs should reference this block, not duplicate the formula):
    // Minimize makespan; when equal, minimize sum of effective completion cycles (tie-breaker).
    // Philosophy: "early completion of computation is better"—a smoother objective that steers
    // search toward packings that can lead to cycle reduction, not only for breaking ties.
    // The solver thus fills earlier slots even when that does not reduce makespan.
    //
    // Design: every op contributes exactly one "effective completion cycle"; for choice vars we
    // use the chosen path's completion so the objective fully accounts for the choice.
    //
    // Objective = scale * makespan + sum(effective_completion_cycle over all ops).
    // Scale is a power of 10 chosen so that scale > max_sum (max possible value of that sum). Then:
    //   (1) Tie-breaker: one extra cycle costs scale, so scale > max_sum ensures makespan dominates.
    //   (2) Leading digits: tie_breaker < scale, so floor(objective / scale) = makespan.
    //   (3) Shiftable VALU: chosen path's completion (completion_cycle when shifted, cycle_var when not).
    //   (4) ConstChoice: all encodings (Load, AddImm, ALU-derived) use the issue cycle as scheduled cycle.
    //   (5) SelectChoice: chosen path's last cycle (fma_last when FMA, cycle_var when vselect).
    //   (6) All other ops: the issue cycle.
    //
    // tuning.objective = Makespan drops the tie-breaker: the objective is `makespan` alone and
    // the "chosen cycle" variables below are not created. Nothing else in the model reads them.
    // The schedule returned is then any schedule of that makespan, not the left-packed one.
    let mut objective_scale: Option<i64> = None;
    let with_tie_breaker = tuning.objective == CpsatObjective::MakespanTiebreak;
    let tie_shiftable: &[usize] = if with_tie_breaker { &shiftable_valu } else { &[] };
    let tie_select: &[usize] = if with_tie_breaker { &select_choice_ops } else { &[] };

    // Tie-breaker should sum only the chosen path's cycle per op (same as makespan constraints).
    // Create auxiliary "chosen cycle" vars for shiftable and SelectChoice so the objective matches:
    // each equals the finish alternative (op_times.rs) its literal selects.
    let add_chosen_cycle_rows = |model: &mut CpModelBuilder, chosen: IntVar, idx: usize, rename_c: &str| {
        for alternative in times[idx].finish_alternatives() {
            let c = model.add_eq(LinearExpr::from(chosen), alternative.value);
            if let Some(when) = alternative.when {
                model.only_enforce_if(c, [when]);
            }
            let label = if alternative.label == "c" { rename_c } else { alternative.label };
            model.set_constraint_name(c, &format!("chosen_cycle_{}_{}", label, idx));
        }
    };
    let mut shiftable_chosen_cycle: HashMap<usize, IntVar> = HashMap::new();
    for &idx in tie_shiftable {
        let op_asap = *asap.get(&ops[idx].id).unwrap_or(&0) as i64;
        let default_alap = (horizon - 1) as i32;
        let op_alap = *alap.get(&ops[idx].id).unwrap_or(&default_alap) as i64;
        let chosen = model.new_int_var_with_name(
            [(op_asap, op_alap)],
            format!("shift_chosen_cycle_{}", idx),
        );
        // completion when shifted, issue cycle when not
        add_chosen_cycle_rows(&mut model, chosen, idx, "c");
        shiftable_chosen_cycle.insert(idx, chosen);
    }

    // Tie-breaker "chosen cycle" for SelectChoice: always the *last* cycle of the op for the path
    // that was chosen (FMA path → multiply_add cycle; vselect path → issue cycle). So the objective
    // tie-breaker uses the actual completion cycle of the chosen variant.
    let mut select_chosen_cycle: HashMap<usize, IntVar> = HashMap::new();
    for &idx in tie_select {
        let op_asap = *asap.get(&ops[idx].id).unwrap_or(&0) as i64;
        let default_alap = (horizon - 1) as i32;
        let op_alap = *alap.get(&ops[idx].id).unwrap_or(&default_alap) as i64;
        if times[idx].fma_cycle().is_some() {
            // The FMA path ends one cycle after the op's own cycle, so the domain must reach
            // alap + 1. With alap as the upper bound a select scheduled at its ALAP could not
            // take the FMA path: an auxiliary of the tie-breaker was restricting the schedule.
            let chosen = model.new_int_var_with_name(
                [(op_asap, op_alap + 1)],
                format!("select_chosen_cycle_{}", idx),
            );
            add_chosen_cycle_rows(&mut model, chosen, idx, "select");
            select_chosen_cycle.insert(idx, chosen);
        } else if times[idx].fma_chosen().is_some() {
            // Optimize but no multiply_add cycle (e.g. FMA has delta_vreg → single cycle). Both
            // paths end at the issue cycle.
            let chosen = model.new_int_var_with_name(
                [(op_asap, op_alap)],
                format!("select_chosen_cycle_{}", idx),
            );
            add_chosen_cycle_rows(&mut model, chosen, idx, "select");
            select_chosen_cycle.insert(idx, chosen);
        }
    }

    // Collect tie-breaker vars: one effective completion cycle per op (chosen path for choice ops).
    // Minimizing this sum implements "early completion of computation is better" when makespan is equal.
    let mut tie_breaker_vars: Vec<IntVar> = Vec::new();

    // Shiftable VALU: chosen path completion (completion_cycle when shifted, cycle_var when not)
    for &idx in tie_shiftable {
        tie_breaker_vars.push(*shiftable_chosen_cycle.get(&idx).unwrap());
    }

    // ConstChoice: all encodings (Load, AddImm, ALU-derived) schedule at the issue cycle
    for &idx in const_choice_ops.iter().filter(|_| with_tie_breaker) {
        tie_breaker_vars.push(times[idx].issue());
    }

    // SelectChoice: chosen path last cycle (fma_last when FMA, cycle_var when vselect)
    for &idx in tie_select {
        if let Some(&chosen) = select_chosen_cycle.get(&idx) {
            tie_breaker_vars.push(chosen);
        } else {
            tie_breaker_vars.push(times[idx].issue());
        }
    }

    // All other ops: execution cycle so every op contributes (reward earlier completion)
    for (idx, _) in ops.iter().enumerate().filter(|_| with_tie_breaker) {
        if !shiftable_set.contains(&idx)
            && !const_choice_set.contains(&idx)
            && !select_choice_set.contains(&idx)
        {
            tie_breaker_vars.push(times[idx].issue());
        }
    }

    if let Some(&first_var) = tie_breaker_vars.first() {
        let rest = &tie_breaker_vars[1..];
        // Max possible value of sum(tie_breaker_vars): each var in [0, horizon-1], so sum <= n * horizon.
        // Use saturating mul so overflow doesn't produce a tiny scale and break "leading digits = makespan".
        let max_sum = (tie_breaker_vars.len() as i64).saturating_mul(horizon);
        // Scale must be > max_sum so that floor(objective/scale) = makespan (tie-breaker stays in lower digits).
        // Presolve can "ExpandObjective" (substitute tight equalities), adding more terms so the actual
        // objective tie-breaker can exceed max_sum. Use an extra factor of 10 so scale stays ahead.
        let mut scale = 1i64;
        while scale <= max_sum {
            scale = scale.saturating_mul(10);
        }
        if scale <= max_sum {
            // Can happen if scale saturated (overflow); objective structure is then wrong.
            eprintln!(
                "  CP-SAT objective scale saturated (max_sum={}); leading digits may not equal makespan",
                max_sum
            );
        }
        objective_scale = Some(scale);
        let sum_completion: LinearExpr = rest.iter().fold(LinearExpr::from(first_var), |acc, &v| {
            acc + LinearExpr::from(v)
        });
        // Explicitly bound tie-breaker below scale so floor(objective/scale) = makespan always.
        let c = model.add_le(sum_completion.clone(), scale.saturating_sub(1));
        model.set_constraint_name(c, "tie_sum_completion");
        let objective = LinearExpr::from(makespan) * scale + sum_completion;
        model.minimize(objective);
    } else {
        // Makespan-only objective (or no ops): no scale, the objective value is the makespan.
        model.minimize(makespan);
    }

    // 11. Add hints from greedy scheduler
    // We skip cycle hints for ops with hint_cycle < 0 (dead stores). Greedy marks those as -1
    // and does not assign a real cycle, so CP-SAT must choose.
    //
    // We hint cycle, shift, completion_cycle, alu_cycle, shiftable_chosen_cycle, select_chosen_cycle,
    // const_choice, select_choice, segment sizes, makespan, max_cycle so the solution hint is complete.
    // Set CPSAT_HINT_DEBUG=1 to log every cycle hint (op idx, id, value, domain) to correlate with "moved var hint".
    let hint_debug_ssa = std::env::var("CPSAT_HINT_DEBUG").is_ok();
    if let Some(hints) = greedy_hints {
        let mut skipped_cycle_hints = 0usize;
        // Under ForceLoad, a greedy schedule that used add_imm for some ConstChoice ops would pack
        // too many Load ops into a cycle once they are forced to Load (Load capacity 2), so their
        // cycle hints are skipped then. A greedy that already honoured the forced encodings
        // (`greedy_matches_forced_modes`) is a feasible schedule and is hinted in full.
        let skip_const_cycle_hints = matches!(const_choice_mode, ConstChoiceMode::ForceLoad)
            && !greedy_matches_forced_modes;
        if skip_const_cycle_hints && verbose {
            eprintln!(
                "  Skipping cycle hints for {} ConstChoice ops (ForceLoad: greedy hint would violate Load capacity)",
                const_choice_ops.len()
            );
        }
        for (idx, &hint_cycle) in hints.iter().enumerate() {
            if idx < times.len() && hint_cycle >= 0 {
                if skip_const_cycle_hints && const_choice_set.contains(&idx) {
                    skipped_cycle_hints += 1;
                    continue;
                }
                if idx < ops.len() {
                    let lb = *final_asap.get(&ops[idx].id).unwrap_or(&0) as i64;
                    let ub = *final_alap.get(&ops[idx].id).unwrap_or(&(horizon as i32 - 1)) as i64;
                    let c = hint_cycle as i64;
                    if hint_debug_ssa {
                        eprintln!(
                            "[CPSAT_HINT_DEBUG] cycle_hint idx={} op_id={} hint={} domain=[{}, {}]",
                            idx, ops[idx].id, c, lb, ub
                        );
                    }
                    // Clamp hint to [final_asap, final_alap] domain. After domain widening (step 2c),
                    // greedy hints should always be within domain. This clamping is defensive for any
                    // future hint sources that might not come from greedy.
                    if c < lb || c > ub {
                        let clamped = c.max(lb).min(ub);
                        if verbose {
                            eprintln!(
                                "  Hint clamped: op id={} idx={} hint={} -> {} domain=[{}, {}]",
                                ops[idx].id, idx, c, clamped, lb, ub
                            );
                        }
                        model.add_hint(times[idx].issue(), clamped);
                    } else {
                        model.add_hint(times[idx].issue(), hint_cycle as i64);
                    }
                } else {
                    model.add_hint(times[idx].issue(), hint_cycle as i64);
                }
            } else if idx < times.len() && hint_cycle < 0 {
                skipped_cycle_hints += 1;
            }
        }
        if verbose && skipped_cycle_hints > 0 {
            eprintln!(
                "  Hint: {} cycle vars unhinted (dead stores, cycle=-1); solver will assign cycles.",
                skipped_cycle_hints
            );
        }
        // Hint makespan and max_cycle so the solver starts from a tight schedule (makespan = max_cycle + 1).
        let hint_makespan = hints.iter().max().copied().unwrap_or(0) + 1;
        model.add_hint(makespan, hint_makespan as i64);
        model.add_hint(max_cycle, (hint_makespan - 1) as i64);
        if let Some(tail_size) = makespan_tail_size {
            model.add_hint(tail_size, (horizon - hint_makespan as i64).max(0));
        }
    }

    // Add shift hints - always hint ALL shift variables to avoid "unhinted variables" warnings.
    // Use greedy's shift decisions when provided; otherwise default to no shift (0).
    if !shift_vars.is_empty() {
        let mut hinted_true = 0usize;
        for &idx in &shiftable_valu {
            let shift_var = shift_vars
                .get(&idx)
                .expect("shift_var should exist for all shiftable ops");
            let op_id = ops[idx].id;
            let forced_value = match valu_shift_mode {
                ValuShiftMode::ForceValu => Some(false),
                ValuShiftMode::ForceAlu => Some(true),
                ValuShiftMode::Optimize => forced_shifts.and_then(|forced| {
                    if forced.contains(&op_id) {
                        Some(true)
                    } else {
                        None
                    }
                }),
            };
            let should_shift = forced_value
                .or_else(|| shift_hints.map(|h| h.contains(&op_id)))
                .unwrap_or(false);
            if should_shift {
                hinted_true += 1;
            }
            model.add_hint(*shift_var, if should_shift { 1 } else { 0 });
        }
        if verbose {
            eprintln!(
                "  Hinting {} shift variables ({} with shift=true, {} with shift=false)",
                shift_vars.len(),
                hinted_true,
                shift_vars.len() - hinted_true
            );
        }
    }

    // Hint completion_cycle and alu_cycle for shiftable ops so CP-SAT gets a full solution hint.
    // Greedy gives one cycle per op; we use that for completion_cycle and all 8 alu_cycles
    // (valid when shifted: all 8 ALU ops at same cycle -> completion_cycle = that cycle).
    // If the greedy scheduled these with encodings the forced modes exclude (e.g. as VALU under
    // ForceAlu), hinting all 8 ALU ops at its cycle would exceed ALU capacity (12/cycle). Skip
    // these hints then so the solver can find a feasible assignment.
    if let Some(hints) = greedy_hints {
        let skip_shifted_hints = !greedy_matches_forced_modes;
        if skip_shifted_hints && verbose {
            eprintln!("  Skipping completion/alu_cycle hints for shiftable ops (greedy schedule does not use the forced encodings)");
        }
        for &idx in &shiftable_valu {
            if skip_shifted_hints {
                continue;
            }
            let hint_cycle = hints.get(idx).copied().unwrap_or(-1);
            if hint_cycle < 0 {
                continue;
            }
            let c = hint_cycle as i64;
            let lb = *final_asap.get(&ops[idx].id).unwrap_or(&0) as i64;
            let ub = *final_alap.get(&ops[idx].id).unwrap_or(&((horizon - 1) as i32)) as i64;
            // Clamp to valid domain (same rationale as cycle hint clamping above)
            let clamped = c.max(lb).min(ub);
            if c != clamped && verbose {
                eprintln!(
                    "  Completion hint clamped: op id={} idx={} hint={} -> {} domain=[{}, {}]",
                    ops[idx].id, idx, c, clamped, lb, ub
                );
            }
            let lanes = hinted_lanes(ops[idx].id);
            if let Some(completion_var) = times[idx].completion() {
                let done = lanes.map_or(clamped, |lanes| lanes.iter().copied().max().unwrap_or(0) as i64);
                model.add_hint(completion_var, done);
            }
            for (lane, &alu_var) in times[idx].lanes().iter().enumerate() {
                // Match the lane cycles used in the capacity check. A whole shift
                // uses the same cycle for every lane.
                let lane_hint = lanes.map_or(clamped, |l| l[lane] as i64);
                model.add_hint(alu_var, lane_hint);
            }
        }
    }

    // Hint shiftable_chosen_cycle and select_chosen_cycle so "solution hint is complete" (no unhinted vars).
    // These tie-breaker vars are equal to the effective cycle (completion_cycle when shifted, cycle when not;
    // fma_last when FMA, cycle when vselect). Greedy gives one cycle per op -> use that for chosen.
    // When ForceAlu and the greedy did not shift every shiftable op, skip shiftable_chosen_cycle
    // hints (same reason as completion/alu_cycle above).
    if let Some(hints) = greedy_hints {
        let skip_shifted_chosen =
            matches!(valu_shift_mode, ValuShiftMode::ForceAlu) && !greedy_matches_forced_modes;
        for &idx in &shiftable_valu {
            if skip_shifted_chosen {
                continue;
            }
            let hint_cycle = hints.get(idx).copied().unwrap_or(-1);
            if hint_cycle < 0 {
                continue;
            }
            if let Some(&chosen) = shiftable_chosen_cycle.get(&idx) {
                // Clamp to domain (defensive: after domain widening, greedy values should be in domain)
                let lb = *final_asap.get(&ops[idx].id).unwrap_or(&0) as i64;
                let ub = *final_alap.get(&ops[idx].id).unwrap_or(&((horizon - 1) as i32)) as i64;
                let clamped = (hint_cycle as i64).max(lb).min(ub);
                model.add_hint(chosen, clamped);
            }
        }
        for &idx in &select_choice_ops {
            let hint_cycle = hints.get(idx).copied().unwrap_or(-1);
            if hint_cycle < 0 {
                continue;
            }
            if let Some(&chosen) = select_chosen_cycle.get(&idx) {
                // vselect path: chosen = cycle. FMA path without a delta: chosen = fma_last =
                // cycle + 1 (chosen_cycle_select_fma_*), so the hint must say so or the hint is
                // infeasible. The domain reaches alap + 1 in that case.
                let hinted_fma = times[idx].fma_cycle().is_some()
                    && match select_choice_mode {
                        SelectChoiceMode::ForceFma => true,
                        SelectChoiceMode::ForceVselect => false,
                        SelectChoiceMode::Optimize => select_choice_hints
                            .map_or(false, |h| h.contains(&ops[idx].id)),
                    };
                let extra = i64::from(hinted_fma);
                // Clamp to domain (defensive: after domain widening, greedy values should be in domain)
                let lb = *final_asap.get(&ops[idx].id).unwrap_or(&0) as i64;
                let ub = *final_alap.get(&ops[idx].id).unwrap_or(&((horizon - 1) as i32)) as i64;
                let clamped = (hint_cycle as i64).max(lb).min(ub) + extra;
                model.add_hint(chosen, clamped);
            }
        }
        // Hint the multiply_add cycle for ALL SelectChoice ops that have one (not just FMA path).
        // fma_cycle = cycle + 1 by constraint, so hint it to hint_cycle + 1 regardless of path.
        // Previously only hinted when use_fma was true, leaving vars unhinted on the vselect path.
        let use_fma = |idx: usize| -> bool {
            select_choice_hints.map_or(false, |h| h.contains(&ops[idx].id))
        };
        for (idx, fma_last_var) in times.iter().enumerate().filter_map(|(i, t)| t.fma_cycle().map(|v| (i, v))) {
            let hint_cycle = hints.get(idx).copied().unwrap_or(-1);
            if hint_cycle >= 0 {
                model.add_hint(fma_last_var, (hint_cycle + 1) as i64);
            }
        }
        // Hint select_fma_extra vars (0 or 1 = same as choice_var) so hint is complete.
        for (&(dep_idx, _idx), &extra_var) in sorted_by_key(&select_fma_extra_vars) {
            model.add_hint(extra_var, if use_fma(dep_idx) { 1 } else { 0 });
        }

        // Hint scratch select_fma_end vars: end = last_use_cycle + 2 (FMA) or + 1 (vselect).
        for &(end_var, last_use_idx) in &scratch_select_fma_end_vars {
            let hint_cycle = hints.get(last_use_idx).copied().unwrap_or(-1);
            if hint_cycle >= 0 {
                let fma = use_fma(last_use_idx);
                let end_hint = if fma { hint_cycle + 2 } else { hint_cycle + 1 };
                model.add_hint(end_var, end_hint as i64);
            }
        }

        // Hint scratch shifted_dest_start vars: when not shifted, start = def_cycle.
        // Greedy defaults to not shifting, so hint = def_cycle.
        // A hinted split shift starts at its first lane.
        for &(start_var, def_idx) in &scratch_shifted_start_vars {
            let hint_cycle = hints.get(def_idx).copied().unwrap_or(-1);
            if hint_cycle >= 0 {
                let start_hint = hinted_lanes(ops[def_idx].id)
                    .and_then(|lanes| lanes.iter().copied().min())
                    .unwrap_or(hint_cycle);
                model.add_hint(start_var, start_hint as i64);
            }
        }
    }

    // Hint CSE shared SUB vars so the solution hint is complete (active literal and cycle;
    // cse_shared_sub_valu_intervals are fully determined by these).
    if let Some(groups) = cse_groups {
        for (g, group) in groups.iter().enumerate() {
            // sub_active = 1 if any consumer in the group has select_choice hint indicating FMA
            let sub_active = select_choice_hints.map_or(false, |h| {
                group
                    .consumer_indices
                    .iter()
                    .any(|&c| h.contains(&ops[c].id))
            });
            model.add_hint(cse_subs[g].active(), if sub_active { 1 } else { 0 });
        }
        let horizon_i64 = horizon as i64;
        for (g, group) in groups.iter().enumerate() {
            // sub_cycle hint: first cycle at which both producers' results are visible: producer
            // cycle + 1, + 1 more for a producer that is itself a select on the multiply-add path
            // (the same delay as the cse_sub_after_producer constraints). Do not unwrap
            // producer_true_idx/producer_false_idx; for complex_phi groups we use only present
            // producer hints and fall back to 0 when none exist.
            let producer_on_fma_path = |idx: usize| -> bool {
                select_choice_with_fma_delay.contains(&idx)
                    && match select_choice_mode {
                        SelectChoiceMode::ForceFma => true,
                        SelectChoiceMode::ForceVselect => false,
                        SelectChoiceMode::Optimize => {
                            select_choice_hints.is_some_and(|h| h.contains(&ops[idx].id))
                        }
                    }
            };
            let sub_cycle_hint = greedy_hints.map_or(0, |hints| {
                let visible_cycles = [group.producer_true_idx, group.producer_false_idx]
                    .into_iter()
                    .flatten()
                    .filter_map(|idx| hints.get(idx).copied().map(|t| (idx, t)))
                    .filter(|&(_, t)| t >= 0)
                    .map(|(idx, t)| t + 1 + i32::from(producer_on_fma_path(idx)));
                let after_producers = visible_cycles.max().unwrap_or(0);
                // When the greedy put consumers on the multiply-add path, it reserved a vector
                // slot for each one's subtract at the consumer's own cycle. The shared subtract
                // takes the earliest of those slots: it is free in the hint, it is after the
                // producers (the consumer depends on them too), and the consumers' multiply_adds
                // run one cycle later.
                let earliest_fma_consumer = group
                    .consumer_indices
                    .iter()
                    .filter(|&&c| select_choice_hints.is_some_and(|h| h.contains(&ops[c].id)))
                    .filter_map(|&c| hints.get(c).copied())
                    .filter(|&t| t >= 0)
                    .min();
                earliest_fma_consumer.map_or(after_producers, |t| t.max(after_producers))
            });
            // Window probe: the reference gives the SUB's cycle (it is pinned there).
            let sub_cycle_hint = window
                .and_then(|w| w.reference.cse_sub_cycles.get(g).copied())
                .filter(|_| window.is_some_and(|w| w.reference.cse_sub_cycles.len() == groups.len()))
                .unwrap_or(sub_cycle_hint);
            let clamped = (sub_cycle_hint as i64).max(0).min(horizon_i64 - 1);
            model.add_hint(cse_subs[g].cycle(), clamped);
            // The shared SUB's scratch interval: it ends one cycle after the SUB, or two
            // cycles after the last consumer hinted as FMA (which reads it at cycle + 1).
            if let Some(&(end_var, size_var)) = cse_sub_scratch_vars.get(g) {
                let last_fma_read_end = group
                    .consumer_indices
                    .iter()
                    .filter(|&&c| select_choice_hints.is_some_and(|h| h.contains(&ops[c].id)))
                    .filter_map(|&c| greedy_hints.and_then(|hints| hints.get(c).copied()))
                    .map(|cycle| cycle as i64 + 2)
                    .max();
                let end_hint = last_fma_read_end.map_or(clamped + 1, |e| e.max(clamped + 1));
                model.add_hint(end_var, end_hint);
                model.add_hint(size_var, end_hint - clamped);
            }
        }
    }

    // 11b. Makespan rows (tuning.makespan_rows, makespan_rows.rs): head/tail rows from static
    // longest-path heads and tails, and with Bucketed prefix/suffix variables that follow the
    // solution. Added after the hints so their auxiliaries are hinted from them.
    if tuning.makespan_rows != MakespanRows::None {
        let eliminable_set: HashSet<usize> = eliminable_vbs.iter().map(|(vb, _, _)| *vb).collect();
        let paths = static_paths(ops, &eliminable_set);
        let hint_map: HashMap<i32, i64> = model
            .proto()
            .solution_hint
            .as_ref()
            .map(|h| h.vars.iter().copied().zip(h.values.iter().copied()).collect())
            .unwrap_or_default();
        let hint = |v: IntVar| hint_map.get(&v.index()).copied();
        let ctx = WorkContext {
            ops,
            times,
            classes: OpClasses { shiftable: &shiftable_set, eliminable_vbs: &eliminable_set },
            const_choice: &op_vars.const_choice,
            const_choice_mode,
        };
        let caps = RowCaps { valu: valu_cap, alu: alu_cap, load: load_cap, store: store_cap, flow: flow_cap };
        let added = add_makespan_rows(
            &mut model,
            &ctx,
            &paths,
            &caps,
            makespan,
            horizon,
            tuning.makespan_rows == MakespanRows::Bucketed,
            &hint,
        );
        for &(var, value) in &added.hints {
            model.add_hint(var, value);
        }
        if let Some(sb) = added.static_bound {
            eprintln!(
                "  Static head/tail bound: makespan >= {} ({} row, head >= {}, tail >= {}; capacity floor {})",
                sb.bound,
                sb.engine.name(),
                sb.t,
                sb.q,
                capacity_lower_bound
            );
        }
        if verbose {
            eprintln!(
                "  Makespan rows: {} rows ({} literal head/tail rows), {} buckets, {} new variables",
                added.rows.len(),
                added.literal_rows,
                added.buckets,
                added.new_vars
            );
        }
        // Every row holds in every schedule (verification/makespan_rows.py), so a greedy
        // schedule within the slot limits satisfies them; a violation is a bug in the rows.
        let violations = if greedy_hints.is_some() { added.hint_violations(&hint) } else { Vec::new() };
        if !violations.is_empty() {
            let message = format!(
                "makespan rows cut off the hint: {} row(s), first {:?}",
                violations.len(),
                &violations[..violations.len().min(5)]
            );
            // A greedy schedule that does not use the forced encodings is not a schedule of
            // this model (its shift hints say ALU while its cycles were packed for VALU).
            // Likewise for ops whose shift is forced (what-if runs) but which the greedy kept
            // on VALU. (Checked here rather than in greedy_schedule_matches_forced_modes, which
            // also decides the domains: changing it would change those models.)
            let forced_shifts_honoured = forced_shifts.map_or(true, |forced| {
                shiftable_valu.iter().all(|&idx| {
                    !forced.contains(&ops[idx].id) || shift_hints.is_some_and(|h| h.contains(&ops[idx].id))
                })
            });
            if cfg!(debug_assertions)
                && greedy_within_scratch_budget == Some(true)
                && greedy_matches_forced_modes
                && forced_shifts_honoured
            {
                panic!("{}", message);
            }
            eprintln!("  WARNING: {}", message);
        }
    }

    // Diagnostic: report unhinted non-fixed variables AND hint values outside domain.
    if verbose {
        let proto = model.proto();
        let num_vars = proto.variables.len();
        let hint_map: std::collections::HashMap<i32, i64> = proto
            .solution_hint
            .as_ref()
            .map(|h| h.vars.iter().copied().zip(h.values.iter().copied()).collect())
            .unwrap_or_default();
        let hinted_set: std::collections::HashSet<i32> = hint_map.keys().copied().collect();
        let mut unhinted_names: Vec<String> = Vec::new();
        let mut out_of_domain: Vec<String> = Vec::new();
        for i in 0..num_vars {
            let var_proto = &proto.variables[i];
            let name = if var_proto.name.is_empty() {
                format!("var_{}", i)
            } else {
                var_proto.name.clone()
            };
            // Skip fixed variables (single-value domain: domain = [v, v])
            let is_fixed = var_proto.domain.len() == 2 && var_proto.domain[0] == var_proto.domain[1];
            if !is_fixed && !hinted_set.contains(&(i as i32)) {
                unhinted_names.push(name.clone());
            }
            // Check if hint value is within domain
            if let Some(&hint_val) = hint_map.get(&(i as i32)) {
                let domain = &var_proto.domain;
                // Domain is a list of [lb, ub] pairs: domain = [lb1, ub1, lb2, ub2, ...]
                let mut in_domain = false;
                for chunk in domain.chunks(2) {
                    if chunk.len() == 2 && hint_val >= chunk[0] && hint_val <= chunk[1] {
                        in_domain = true;
                        break;
                    }
                }
                if !in_domain {
                    out_of_domain.push(format!(
                        "{} hint={} domain={:?}", name, hint_val, domain
                    ));
                }
            }
        }
        if !unhinted_names.is_empty() {
            eprintln!(
                "  CP-SAT hint diagnostic: {} non-fixed variables still unhinted:",
                unhinted_names.len()
            );
            for (i, name) in unhinted_names.iter().enumerate() {
                if i < 20 {
                    eprintln!("    - {}", name);
                } else {
                    eprintln!("    ... and {} more", unhinted_names.len() - 20);
                    break;
                }
            }
        }
        if !out_of_domain.is_empty() {
            eprintln!(
                "  CP-SAT hint diagnostic: {} variables with hint OUTSIDE domain:",
                out_of_domain.len()
            );
            for (i, entry) in out_of_domain.iter().enumerate() {
                if i < 20 {
                    eprintln!("    - {}", entry);
                } else {
                    eprintln!("    ... and {} more", out_of_domain.len() - 20);
                    break;
                }
            }
        } else {
            eprintln!("  CP-SAT hint diagnostic: all hint values within domain bounds");
        }
    }

    // 12. Configure solver
    // Wall-clock limits make the result depend on machine load: the same command explores
    // less on a busy machine and returns a different schedule. A deterministic-time budget
    // (CP-SAT's own work counter) gives the same schedule every time. The wall-clock limit
    // stays as a backstop.
    let deterministic_budget = deterministic_time_limit
        .filter(|d| d.is_finite() && *d > 0.0)
        .or_else(max_deterministic_time_from_env);
    if let Some(dtime) = deterministic_budget {
        if verbose {
            eprintln!("  CP-SAT deterministic time limit: {}", dtime);
        }
    }
    // Gap limit: default uses scale/3 for faster convergence, but can force 0 for debugging.
    // Stop when gap <= scale/3: remaining improvement is only in the tie-breaker (sum of completion_cycle).
    // One extra cycle costs scale in the objective, so scale/3 guarantees we're within ~1 cycle worst case.
    let absolute_gap_limit = if force_zero_gap {
        // Force 0 integral gap for proven optimality (debugging only).
        // This ensures CP-SAT explores all possibilities, including tie-breaker optimizations
        // like moving ConstChoice ops earlier via add_imm.
        Some(0.0)
    } else {
        objective_scale.map(|scale| (scale / 3) as f64)
    };
    let params = build_sat_parameters(
        time_limit_secs,
        deterministic_budget,
        stop_after_first_solution,
        absolute_gap_limit,
        tuning,
    );
    if verbose && *tuning != CpsatTuning::default() {
        eprintln!("  CP-SAT tuning: {}", tuning.to_json());
    }

    // 13b. Decision strategy: branch on structural choices first, then bottleneck engines.
    // This decomposes search into "pick a configuration, then schedule within it."
    // Decision strategies are advisory — CP-SAT can override them, so this cannot make things worse.
    {
        use cp_sat::proto::DecisionStrategyProto;
        use cp_sat::proto::decision_strategy_proto::{
            VariableSelectionStrategy, DomainReductionStrategy,
        };

        // Phase 1: Structural decisions (shift? FMA? Load?) — branch on these first
        let mut choice_var_indices: Vec<i32> = Vec::new();
        for (_, &bv) in sorted_by_key(&shift_vars) {
            choice_var_indices.push(bv.index());
        }
        for (_, &bv) in sorted_by_key(&const_choice_vars) {
            choice_var_indices.push(bv.index());
        }
        for (_, &bv) in sorted_by_key(&select_choice_vars) {
            choice_var_indices.push(bv.index());
        }
        if !choice_var_indices.is_empty() {
            model.proto_mut().search_strategy.push(DecisionStrategyProto {
                variables: choice_var_indices.clone(),
                exprs: vec![],
                variable_selection_strategy: VariableSelectionStrategy::ChooseFirst as i32,
                domain_reduction_strategy: DomainReductionStrategy::SelectMinValue as i32,
            });
        }

        // Phase 2: Bottleneck engine scheduling (VALU and Flow — tightest capacity)
        let mut bottleneck_var_indices: Vec<i32> = Vec::new();
        for (idx, op) in ops.iter().enumerate() {
            if op.engine == Engine::Valu || op.engine == Engine::Flow {
                bottleneck_var_indices.push(times[idx].issue().index());
            }
        }
        if !bottleneck_var_indices.is_empty() {
            model.proto_mut().search_strategy.push(DecisionStrategyProto {
                variables: bottleneck_var_indices,
                exprs: vec![],
                variable_selection_strategy: VariableSelectionStrategy::ChooseLowestMin as i32,
                domain_reduction_strategy: DomainReductionStrategy::SelectMinValue as i32,
            });
        }

        // Phase 3: Remaining engines (ALU, Load, Store — more capacity headroom)
        let mut remaining_var_indices: Vec<i32> = Vec::new();
        for (idx, op) in ops.iter().enumerate() {
            if op.engine != Engine::Valu && op.engine != Engine::Flow {
                remaining_var_indices.push(times[idx].issue().index());
            }
        }
        if !remaining_var_indices.is_empty() {
            model.proto_mut().search_strategy.push(DecisionStrategyProto {
                variables: remaining_var_indices,
                exprs: vec![],
                variable_selection_strategy: VariableSelectionStrategy::ChooseLowestMin as i32,
                domain_reduction_strategy: DomainReductionStrategy::SelectMinValue as i32,
            });
        }

        if verbose {
            let n_strategies = model.proto().search_strategy.len();
            let n_choice = choice_var_indices.len();
            eprintln!(
                "  Decision strategy: {} phases ({} choice vars first, then VALU/Flow, then rest)",
                n_strategies, n_choice
            );
        }
    }

    // Optional export of the exact model handed to the solver (CPSAT_EXPORT_MODEL=path), e.g.
    // to diff two builds with scripts/diff_cpsat_models.py. Later solves in the same process
    // (per-section runs come first) overwrite the file, so it holds the last model built.
    if let Ok(path) = std::env::var("CPSAT_EXPORT_MODEL") {
        let bytes = prost::Message::encode_to_vec(model.proto());
        match std::fs::write(&path, &bytes) {
            Ok(()) => eprintln!("  CP-SAT model exported to {}", path),
            Err(e) => eprintln!("  CP-SAT model export to {} failed: {}", path, e),
        }
    }
    // CPSAT_EXPORT_MODEL_DIR=dir keeps every model built in this process, per-section solves
    // included, as dir/model_000.pb, dir/model_001.pb, ... in build order.
    if let Ok(dir) = std::env::var("CPSAT_EXPORT_MODEL_DIR") {
        static EXPORTED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = EXPORTED.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let path = std::path::Path::new(&dir).join(format!("model_{:03}.pb", n));
        let bytes = prost::Message::encode_to_vec(model.proto());
        match std::fs::create_dir_all(&dir).and_then(|()| std::fs::write(&path, &bytes)) {
            Ok(()) => eprintln!("  CP-SAT model exported to {}", path.display()),
            Err(e) => eprintln!("  CP-SAT model export to {} failed: {}", path.display(), e),
        }
    }

    // 14. Solve
    if window.is_some() {
        eprintln!(
            "[window probe]   model built in {:.1}s: {} variables, {} constraints",
            start_time.elapsed().as_secs_f64(),
            model.proto().variables.len(),
            model.proto().constraints.len()
        );
    }
    let response = model.solve_with_parameters(&params);
    if window.is_some() {
        eprintln!(
            "[window probe]   solve: {:.1}s wall, {:.1} deterministic time",
            response.wall_time, response.deterministic_time
        );
    }
    let solve_time = start_time.elapsed().as_secs_f64();
    // A deterministic budget only makes the run reproducible if the solve ended on it (or on
    // optimality). If the wall-clock backstop fired first, how far the search got depended on
    // machine load, so say so.
    if let Some(dtime) = deterministic_budget {
        if response.wall_time >= time_limit_secs * 0.999 && response.deterministic_time < dtime {
            eprintln!(
                "  WARNING: CP-SAT stopped on the wall-clock limit ({:.0}s) after {:.1} of {} deterministic time; \
                 this result is not reproducible. Raise --cpsat-time-limit.",
                time_limit_secs, response.deterministic_time, dtime
            );
        }
    }

    let status: CpsatStatus = response.status().into();

    match status {
        CpsatStatus::Optimal | CpsatStatus::Feasible => {
            // Extract cycle assignments for all ops: the completion cycle of a shifted op, the
            // issue cycle otherwise (ForceAlu fixes every shift literal to 1, ForceValu to 0).
            let cycle_assignments: Vec<i32> = times.iter().map(|t| t.reported_cycle(&response)).collect();

            // Reported makespan must match the objective's leading digits (floor(objective/scale)).
            // The model constrains makespan >= cycle_var+1 for every cycle_var; for shiftable ops we
            // put completion_cycle into cycle_assignments when shifted, so max(cycle_assignments)+1
            // can be lower than the solver's makespan (which is driven by the issue cycles). Use the
            // makespan variable's solution value so we report the same number that appears in the objective.
            let actual_makespan = makespan.solution_value(&response) as i32;

            // Sanity: makespan variable should be >= max(cycle_assignments)+1 (model constraint).
            let max_cycle_from_assignments = cycle_assignments.iter().max().copied().unwrap_or(0) + 1;
            if actual_makespan < max_cycle_from_assignments && verbose {
                eprintln!(
                    "  Inconsistency: makespan var={} but max(cycle_assignments)+1={}",
                    actual_makespan, max_cycle_from_assignments
                );
            }

            // Empty-cycle gap: when makespan > max_cycle_from_assignments, we have empty cycles.
            let empty_count = (actual_makespan - max_cycle_from_assignments).max(0) as usize;
            if empty_count > 0 && verbose {
                let driven_by_bound = (actual_makespan as i64) == capacity_lower_bound;
                eprintln!(
                    "  Empty cycles: {} (actual_makespan={}, max_cycle_from_assignments+1={}, makespan_driven_by_capacity_bound={})",
                    empty_count,
                    actual_makespan,
                    max_cycle_from_assignments,
                    driven_by_bound
                );
            }

            // Extract shift decisions: ForceAlu => all shiftable; ForceValu => none; Optimize => from solver
            let shifted_ops: HashSet<usize> = match valu_shift_mode {
                ValuShiftMode::ForceAlu => shiftable_valu.iter().map(|&idx| ops[idx].id).collect(),
                ValuShiftMode::ForceValu => HashSet::new(),
                ValuShiftMode::Optimize => {
                    if shift_vars.is_empty() {
                        HashSet::new()
                    } else {
                        shift_vars
                            .iter()
                            .filter(|(_, &var)| var.solution_value(&response))
                            .map(|(&idx, _)| ops[idx].id)
                            .collect()
                    }
                }
            };

            // Debug: Extract shift decisions for high-impact ops
            let high_impact_op_ids: HashSet<usize> = [244, 287, 291, 285, 323, 289, 262]
                .iter()
                .copied()
                .collect();
            if verbose && !shift_vars.is_empty() {
                for &idx in &shiftable_valu {
                    let op_id = ops[idx].id;
                    if high_impact_op_ids.contains(&op_id) {
                        let shift_var = shift_vars.get(&idx).expect("shift_var should exist");
                        let is_shifted = shift_var.solution_value(&response);
                        eprintln!("  High-impact op {}: shifted={}", op_id, is_shifted);

                        if is_shifted {
                            // Extract what cycles CP-SAT would assign
                            {
                                let cycles: Vec<i32> = times[idx].lane_values(&response);
                                let unique_cycles: std::collections::HashSet<i32> =
                                    cycles.iter().copied().collect();
                                eprintln!(
                                    "    ALU cycles: {:?} ({} unique)",
                                    cycles,
                                    unique_cycles.len()
                                );

                                if let Some(completion_cycle_var) = times[idx].completion() {
                                    let completion =
                                        completion_cycle_var.solution_value(&response) as i32;
                                    eprintln!("    Completion cycle: {}", completion);
                                }
                            }
                        } else {
                            // Check why it's not shifted - what would happen if it were?
                            let op_asap = *asap.get(&op_id).unwrap_or(&0);
                            let op_alap = *alap.get(&op_id).unwrap_or(&((horizon - 1) as i32));
                            eprintln!("    Not shifted: ASAP={}, ALAP={}", op_asap, op_alap);
                        }
                    }
                }
            }

            // Extract individual ALU cycle assignments for shifted VALU ops (ForceAlu => all; Optimize => from solver)
            let mut shifted_alu_cycles: HashMap<usize, Vec<i32>> = HashMap::new();
            for &idx in &shiftable_valu {
                let op_id = ops[idx].id;
                let is_shifted = shifted_ops.contains(&op_id);
                if is_shifted {
                    shifted_alu_cycles.insert(op_id, times[idx].lane_values(&response));
                }
            }

            // Extract ConstChoice: one encoding per op (Load, AddImm, or AluDerive).
            let const_choice_encoding: HashMap<usize, crate::types::ConstChoiceEncoding> =
                match const_choice_mode {
                    ConstChoiceMode::ForceAddImm => const_choice_ops
                        .iter()
                        .map(|&idx| (ops[idx].id, crate::types::ConstChoiceEncoding::AddImm))
                        .collect(),
                    ConstChoiceMode::ForceLoad => const_choice_ops
                        .iter()
                        .map(|&idx| (ops[idx].id, crate::types::ConstChoiceEncoding::Load))
                        .collect(),
                    ConstChoiceMode::Optimize => {
                        let mut enc = HashMap::new();
                        for &idx in &const_choice_ops {
                            let op_id = ops[idx].id;
                            let choice = if let Some(use_vars) = const_choice_encoding_vars.get(&idx) {
                                let chosen = use_vars
                                    .iter()
                                    .position(|&b| b.solution_value(&response));
                                match chosen {
                                    Some(1) => crate::types::ConstChoiceEncoding::AddImm,
                                    Some(k) if k >= 2 => {
                                        if let Some(info) = const_choice_alu_encoding_info.get(&idx) {
                                            let i = k - 2;
                                            if i < info.len() {
                                                let (li, ri, opcode) = info[i];
                                                crate::types::ConstChoiceEncoding::AluDerive(
                                                    crate::types::AluDeriveInfo {
                                                        left_op_id: ops[li].id,
                                                        right_op_id: ops[ri].id,
                                                        opcode,
                                                    },
                                                )
                                            } else {
                                                crate::types::ConstChoiceEncoding::Load
                                            }
                                        } else {
                                            crate::types::ConstChoiceEncoding::Load
                                        }
                                    }
                                    _ => crate::types::ConstChoiceEncoding::Load,
                                }
                            } else if let Some(&var) = const_choice_vars.get(&idx) {
                                if var.solution_value(&response) {
                                    crate::types::ConstChoiceEncoding::AddImm
                                } else {
                                    crate::types::ConstChoiceEncoding::Load
                                }
                            } else {
                                crate::types::ConstChoiceEncoding::Load
                            };
                            enc.insert(op_id, choice);
                        }
                        enc
                    }
                };

            // Extract SelectChoice: one encoding per op (Fma or Vselect).
            let select_choice_encoding: HashMap<usize, crate::types::SelectChoiceEncoding> =
                match select_choice_mode {
                    SelectChoiceMode::ForceFma => select_choice_ops
                        .iter()
                        .map(|&idx| (ops[idx].id, crate::types::SelectChoiceEncoding::Fma))
                        .collect(),
                    SelectChoiceMode::ForceVselect => select_choice_ops
                        .iter()
                        .map(|&idx| (ops[idx].id, crate::types::SelectChoiceEncoding::Vselect))
                        .collect(),
                    SelectChoiceMode::Optimize => {
                        let mut enc = HashMap::new();
                        for &idx in &select_choice_ops {
                            let op_id = ops[idx].id;
                            let e = if select_choice_vars
                                .get(&idx)
                                .map(|v| v.solution_value(&response))
                                .unwrap_or(false)
                            {
                                crate::types::SelectChoiceEncoding::Fma
                            } else {
                                crate::types::SelectChoiceEncoding::Vselect
                            };
                            enc.insert(op_id, e);
                        }
                        enc
                    }
                };
            let select_fma_op_ids: HashSet<usize> = select_choice_encoding
                .iter()
                .filter(|(_, e)| matches!(e, crate::types::SelectChoiceEncoding::Fma))
                .map(|(id, _)| *id)
                .collect();

            // Extract CSE group results (sub_active, sub_cycle per group)
            let cse_group_results: Vec<super::cse_groups::CseGroupResult> =
                cse_subs
                    .iter()
                    .map(|sub| super::cse_groups::CseGroupResult {
                        sub_active: sub.active().solution_value(&response),
                        sub_cycle: sub.cycle().solution_value(&response) as i32,
                    })
                    .collect();

            // Extract eliminated vbroadcasts and vb -> scalar producer map
            let (eliminated_vbroadcast_ops, eliminated_vb_to_scalar_producer) = {
                let mut eliminated = HashSet::new();
                let mut vb_to_scalar = HashMap::new();
                for (vb_idx, scalar_producer_idx, _) in &eliminable_vbs {
                    if let Some(&ev) = eliminate_vb_vars.get(vb_idx) {
                        if ev.solution_value(&response) {
                            eliminated.insert(ops[*vb_idx].id);
                            vb_to_scalar.insert(ops[*vb_idx].id, ops[*scalar_producer_idx].id);
                        }
                    }
                }
                (eliminated, vb_to_scalar)
            };

            // Build schedule constraints once (same as CP-SAT; used for active-set binding list and bulk-move).
            // Use final_asap/final_alap so exported Domain bounds match the cycle var domains the solver used.
            let schedule_constraints = crate::schedule_constraints::build_ssa_schedule_constraints(
                ops,
                &op_id_to_idx,
                &final_asap,
                &final_alap,
                &shiftable_set,
                &shifted_ops,
                &select_choice_with_fma_delay,
                &select_fma_op_ids,
                horizon as i32,
            );

            // Makespan active set: which constraints are binding at the solution (hold with equality).
            if verbose {
                let max_cycle_val = max_cycle.solution_value(&response) as i32;
                let mut binding_ops: Vec<(usize, usize, &str)> = Vec::new(); // (idx, op_id, reason)
                for (idx, _) in ops.iter().enumerate() {
                    let (effective_last, finish) = times[idx].finish_value(&response);
                    if effective_last == max_cycle_val {
                        let reason = match finish {
                            "shifted" => "shifted (completion_cycle)",
                            "valu" => "VALU cycle",
                            "select_fma" => "SelectChoice FMA (fma_last)",
                            "select_vselect" => "SelectChoice vselect",
                            _ => "cycle",
                        };
                        binding_ops.push((idx, ops[idx].id, reason));
                    }
                }
                eprintln!("  [Makespan active set] makespan={} max_cycle={}", actual_makespan, max_cycle_val);
                eprintln!("    makespan = max_cycle + 1: always binding");
                if !binding_ops.is_empty() {
                    eprintln!(
                        "    max_cycle >= effective_last: {} op(s) binding (define last cycle)",
                        binding_ops.len()
                    );
                    for (idx, op_id, reason) in binding_ops.iter().take(15) {
                        let cse_info = op_idx_to_cse_group
                            .get(idx)
                            .map(|g| format!(" CSE_group={}", g))
                            .unwrap_or_default();
                        eprintln!("      idx={} op.id={} {}{}", idx, op_id, *reason, cse_info);
                    }
                    if binding_ops.len() > 15 {
                        eprintln!("      ... and {} more", binding_ops.len() - 15);
                    }
                }
                let m = actual_makespan as i64;
                let binding_capacity_bounds: Vec<&str> = [
                    (m == capacity_lower_bound, "capacity_lower_bound"),
                    (load_bound > 0 && m == load_bound, "load_bound"),
                    (store_bound > 0 && m == store_bound, "store_bound"),
                ]
                .iter()
                .filter(|(b, _)| *b)
                .map(|(_, name)| *name)
                .collect();
                if !binding_capacity_bounds.is_empty() {
                    eprintln!(
                        "    makespan >= X (redundant linear): binding for {}",
                        binding_capacity_bounds.join(", ")
                    );
                    // Sanity: if makespan is driven by a capacity bound but the schedule has many empty cycles,
                    // that bound may be wrong (over-constraining). Bulk-move only checks precedence/ordering/domain,
                    // not capacity, so it cannot detect an incorrect capacity bound.
                    if empty_count > 10 {
                        eprintln!(
                            "    WARNING: {} empty cycles but makespan binding on {} — bound may be incorrect (over-constraining); bulk-move cannot check capacity.",
                            empty_count,
                            binding_capacity_bounds.join(", ")
                        );
                    }
                }
                if enable_valu_shift && (valu_count > 0 || shiftable_count > 0 || select_choice_count > 0) {
                    let s = shifted_ops.len() as i64; // number shifted to ALU (model s = sum(shift_vars))
                    let z = select_fma_op_ids.len() as i64;
                    let select_valu_contrib_solution: i64 = if !cse_group_results.is_empty() {
                        let num_active = cse_group_results.iter().filter(|r| r.sub_active).count() as i64;
                        num_active + z
                    } else {
                        2 * z
                    };
                    let lhs_valu = 6 * (actual_makespan as i64) + s - select_valu_contrib_solution;
                    let binding_valu = lhs_valu == valu_count;
                    if binding_valu {
                        eprintln!(
                            "    Direct VALU (6*makespan + shift - select_valu_contrib >= valu_count): binding (LHS={})",
                            lhs_valu
                        );
                    }
                }
                if enable_valu_shift && (alu_count > 0 || shiftable_count > 0) {
                    let s = shifted_ops.len() as i64; // number shifted to ALU (same s as in model)
                    let lhs_alu = 12 * (actual_makespan as i64) - 8 * s;
                    let binding_alu = lhs_alu == alu_count;
                    if binding_alu {
                        eprintln!(
                            "    Direct ALU (12*makespan - 8*shift >= alu_count): binding (LHS={})",
                            lhs_alu
                        );
                    }
                }
                if flow_base > 0 || const_choice_count > 0 || select_choice_count > 0 {
                    let z = select_fma_op_ids.len() as i64;
                    let y = const_choice_encoding
                        .values()
                        .filter(|e| matches!(e, crate::types::ConstChoiceEncoding::AddImm))
                        .count() as i64;
                    let lhs_flow = (actual_makespan as i64) + z - y;
                    let binding_flow = lhs_flow == flow_base;
                    if binding_flow {
                        eprintln!(
                            "    Direct Flow (makespan + select_fma - const_add_imm >= flow_base): binding (LHS={})",
                            lhs_flow
                        );
                    }
                }
                if load_count > 0 && const_choice_count > 0 {
                    let y = const_choice_encoding
                        .values()
                        .filter(|e| matches!(e, crate::types::ConstChoiceEncoding::AddImm))
                        .count() as i64;
                    let lhs_load = 2 * (actual_makespan as i64) + y;
                    let binding_load = lhs_load == load_count;
                    if binding_load {
                        eprintln!(
                            "    Direct Load (2*makespan + const_add_imm >= load_count): binding (LHS={})",
                            lhs_load
                        );
                    }
                }
                // Binding precedence/ordering/domain (same checks bulk-move uses). When bulk-move blocks, it should report one of these.
                let binding = crate::schedule_constraints::binding_schedule_constraints(
                    ops,
                    &cycle_assignments,
                    &shifted_alu_cycles,
                    &schedule_constraints,
                );
                let n_precedence = binding.iter().filter(|(c, _)| matches!(c, crate::schedule_constraints::ScheduleConstraint::Precedence { .. })).count();
                let n_ordering = binding.iter().filter(|(c, _)| matches!(c, crate::schedule_constraints::ScheduleConstraint::Ordering { .. })).count();
                let n_domain = binding.iter().filter(|(c, _)| matches!(c, crate::schedule_constraints::ScheduleConstraint::Domain { .. })).count();
                eprintln!(
                    "    Binding schedule constraints (bulk-move uses these): {} precedence, {} ordering, {} domain — when bulk-move blocks, the violation should be one of these.",
                    n_precedence, n_ordering, n_domain
                );
                let max_cycle_val = max_cycle.solution_value(&response) as i32;
                use crate::schedule_constraints::dep_effective_cycle as dep_eff;
                let binding_at_last: Vec<_> = binding
                    .iter()
                    .filter(|(c, _)| {
                        match c {
                            crate::schedule_constraints::ScheduleConstraint::Precedence {
                                consumer_idx,
                                dep_idx,
                                dep_is_shifted,
                                ..
                            } => {
                                let cy_c = cycle_assignments.get(*consumer_idx).copied().unwrap_or(-1);
                                let cy_d = dep_eff(
                                    *dep_idx,
                                    *dep_is_shifted,
                                    &cycle_assignments,
                                    &shifted_alu_cycles,
                                    ops,
                                );
                                cy_c == max_cycle_val || cy_d == max_cycle_val
                            }
                            crate::schedule_constraints::ScheduleConstraint::Ordering {
                                before_idx,
                                after_idx,
                                ..
                            } => {
                                let before_c = cycle_assignments.get(*before_idx).copied().unwrap_or(-1);
                                let after_c = cycle_assignments.get(*after_idx).copied().unwrap_or(-1);
                                before_c == max_cycle_val || after_c == max_cycle_val
                            }
                            crate::schedule_constraints::ScheduleConstraint::Domain { idx, .. } => {
                                cycle_assignments.get(*idx).copied().unwrap_or(-1) == max_cycle_val
                            }
                        }
                    })
                    .take(15)
                    .collect();
                if !binding_at_last.is_empty() {
                    eprintln!("    Binding at last cycle (max_cycle={}):", max_cycle_val);
                    for (c, msg) in binding_at_last {
                        let short = match c {
                            crate::schedule_constraints::ScheduleConstraint::Precedence { consumer_idx, dep_idx, .. } => {
                                format!("Precedence consumer {} dep {}", consumer_idx, dep_idx)
                            }
                            crate::schedule_constraints::ScheduleConstraint::Ordering { before_idx, after_idx, .. } => {
                                format!("Ordering {} <= {}", before_idx, after_idx)
                            }
                            crate::schedule_constraints::ScheduleConstraint::Domain { idx, .. } => {
                                format!("Domain idx {}", idx)
                            }
                        };
                        eprintln!("      {} | {}", short, msg);
                    }
                }
            }

            // Empty-cycle breadcrumbs: for each empty C, list all ops at C+1 and why they weren't at C (do not filter by ASAP/ALAP).
            // Invariant: no constraint should preclude every op at C+1 from moving to C (bulk move); if they're all at C+1 it's solver/tie-breaker.
            if verbose && empty_count > 0 {
                let used_cycles: HashSet<i32> = cycle_assignments.iter().copied().collect();
                let empty_cycles_list: Vec<i32> = (0..actual_makespan)
                    .filter(|&c| !used_cycles.contains(&c))
                    .take(15)
                    .collect();
                eprintln!("  [Empty-cycle breadcrumbs] {} empty cycles; showing ops at C+1 for first {} empty C",
                    empty_count, empty_cycles_list.len());
                for &c in &empty_cycles_list {
                    for (idx, &cy) in cycle_assignments.iter().enumerate() {
                        if cy != c + 1 {
                            continue;
                        }
                        let op = &ops[idx];
                        let op_id = op.id;
                        let engine_str = op.engine.as_str();
                        let opcode_str = format!("{:?}", op.opcode);
                        let mut latest_dep_result = -1i32;
                        let mut blocking_dep_id = 0usize;
                        let mut blocking_dep_opcode = String::new();
                        let mut blocking_dep_cycle = -1i32;
                        for &dep_id in &op.dep_ids {
                            let Some(&dep_idx) = op_id_to_idx.get(&dep_id) else {
                                continue;
                            };
                            let dep_cycle = cycle_assignments[dep_idx];
                            let result_cycle = if select_choice_with_fma_delay.contains(&dep_idx)
                                && select_fma_op_ids.contains(&dep_id)
                            {
                                dep_cycle + 2
                            } else {
                                dep_cycle + 1
                            };
                            if result_cycle > latest_dep_result {
                                latest_dep_result = result_cycle;
                                blocking_dep_id = dep_id;
                                blocking_dep_opcode = format!("{:?}", ops[dep_idx].opcode);
                                blocking_dep_cycle = dep_cycle;
                            }
                        }
                        // Result at cycle R means consumer can run at R or later. So op is blocked from C only when latest_dep_result > C.
                        if latest_dep_result > c {
                            eprintln!(
                                "  Empty {}: op id={} (engine={}, opcode={}) at C+1 blocked by precedence: dep id={} (opcode={}) at cycle {} -> result at {}.",
                                c, op_id, engine_str, opcode_str, blocking_dep_id, blocking_dep_opcode, blocking_dep_cycle, latest_dep_result
                            );
                        } else {
                            eprintln!(
                                "  Empty {}: op id={} (engine={}, opcode={}) at C+1 could run at C (BUG: should not be at C+1); latest dep result at {}.",
                                c, op_id, engine_str, opcode_str, latest_dep_result
                            );
                            let op_asap = *asap.get(&op_id).unwrap_or(&0);
                            let op_alap = *alap.get(&op_id).unwrap_or(&(horizon as i32 - 1));
                            if c < op_asap || c > op_alap {
                                eprintln!(
                                    "    BUG: ASAP/ALAP excludes empty cycle {} (asap={}, alap={})",
                                    c, op_asap, op_alap
                                );
                            }
                        }
                    }
                }
            }

            // Solution breadcrumbs: log Flow audit when verbose (before validation)
            if verbose {
                let (flow_rows, flow_per_cycle, _) =
                    build_flow_audit(ops, &cycle_assignments, &const_choice_encoding, &select_choice_encoding);
                eprintln!(
                    "[Flow solution breadcrumbs] Total Flow-contributing ops: {}",
                    flow_rows.len()
                );
                let add_imm_ids: Vec<usize> = const_choice_encoding
                    .iter()
                    .filter(|(_, e)| matches!(e, crate::types::ConstChoiceEncoding::AddImm))
                    .map(|(&id, _)| id)
                    .collect();
                eprintln!(
                    "[Flow solution breadcrumbs] ConstChoice AddImm (op.id) = {:?}",
                    add_imm_ids
                );
                eprintln!(
                    "[Flow solution breadcrumbs] SelectChoice Fma (op.id) = {:?}",
                    select_fma_op_ids
                );
                for r in &flow_rows {
                    eprintln!(
                        "  idx={} op.id={} opcode={:?} cycle={} {}",
                        r.idx, r.op_id, r.opcode, r.cycle, r.reason
                    );
                }
                let overloaded: Vec<(usize, i32)> = flow_per_cycle
                    .iter()
                    .enumerate()
                    .filter(|(_, &count)| count > 1)
                    .map(|(cycle_idx, &count)| (cycle_idx, count))
                    .collect();
                if !overloaded.is_empty() {
                    eprintln!(
                        "[Flow solution breadcrumbs] WARNING: cycles with Flow count > 1: {:?}",
                        overloaded
                    );
                }
            }

            let (best_bound, gap) = if status == CpsatStatus::Optimal {
                (actual_makespan as f64, 0.0)
            } else {
                let best = response.objective_value;
                let bound = response.best_objective_bound;
                let gap_val = if best > 0.0 {
                    (best - bound).abs() / best
                } else {
                    0.0
                };
                (bound, gap_val)
            };

            // Post-solution validation: slot limits must never be violated (model bug if they are)
            validate_ssa_slot_limits(
                ops,
                &cycle_assignments,
                &shifted_ops,
                &shifted_alu_cycles,
                &const_choice_encoding,
                &select_choice_encoding,
                &eliminated_vbroadcast_ops,
                &machine,
                &cse_group_results,
                &op_idx_to_cse_group,
            )?;

            // Post-solution validation: every RAW dependency satisfies precedence (consumer >= producer visible cycle).
            verify_all_raw_precedence(
                ops,
                &cycle_assignments,
                &shifted_ops,
                &shifted_alu_cycles,
                &select_choice_encoding,
            );
            // SelectChoice FMA result visible at C+2 (redundant with verify_all_raw_precedence but kept for clarity).
            verify_select_choice_fma_consumer_precedence(ops, &cycle_assignments, &select_choice_encoding);
            // Memory edges (latency 0 for WAR).
            verify_memory_precedence(ops, &cycle_assignments);
            // The scratch cumulative must have covered every reader of every value.
            if scratch_budget > 0 && !scratch_values.is_empty() {
                verify_scratch_budget_all_readers(
                    ops,
                    &scratch_values,
                    &cycle_assignments,
                    &shifted_ops,
                    &shifted_alu_cycles,
                    &select_choice_encoding,
                    scratch_budget,
                    scratch_time,
                );
            }
            // Shared CSE subs read true_val/false_val at sub_cycle; those must be visible by then.
            if let Some(groups) = cse_groups {
                verify_cse_sub_precedence(
                    ops,
                    groups,
                    &cse_group_results,
                    &cycle_assignments,
                    &shifted_ops,
                    &shifted_alu_cycles,
                    &select_choice_encoding,
                );
            }

            if verbose {
                // Verify objective structure: objective = scale*makespan + tie_breaker, so floor(objective/scale) should equal makespan.
                if let Some(scale) = objective_scale {
                    let obj_val = response.objective_value;
                    let recovered_makespan = (obj_val / (scale as f64)).floor() as i32;
                    if (recovered_makespan - actual_makespan).abs() > 1 {
                        eprintln!(
                            "  Objective structure check: objective_value={:.0}, scale={}, floor(obj/scale)={} vs actual_makespan={}",
                            obj_val, scale, recovered_makespan, actual_makespan
                        );
                    }
                    eprintln!(
                        "  Objective: scale={}, so makespan is leading digits (floor({:.0}/{})= {}), tie-breaker in lower digits",
                        scale, obj_val, scale, recovered_makespan
                    );
                }
                let summary = summarise_ssa_solution_engines_and_choices(
                    ops,
                    &cycle_assignments,
                    &shifted_ops,
                    &const_choice_encoding,
                    &select_choice_encoding,
                );
                eprintln!(
                    "SSA CP-SAT result: {} cycles (best_bound: {:.1}), status {:?}, gap {:.2}%, time {:.2}s",
                    actual_makespan, best_bound, status, gap * 100.0, solve_time
                );
                // One-line active set summary for debugging (full "[Makespan active set]" is printed earlier).
                {
                    let max_cycle_val = max_cycle.solution_value(&response) as i32;
                    let mut n_binding_ops = 0usize;
                    for (idx, _) in ops.iter().enumerate() {
                        let (effective_last, _) = times[idx].finish_value(&response);
                        if effective_last == max_cycle_val {
                            n_binding_ops += 1;
                        }
                    }
                    let m = actual_makespan as i64;
                    let capacity_binding = m == capacity_lower_bound
                        || (load_bound > 0 && m == load_bound)
                        || (store_bound > 0 && m == store_bound);
                    eprintln!(
                        "  Active set: {} op(s) at last cycle (max_cycle={}); capacity_bound_binding={} (see [Makespan active set] above for details)",
                        n_binding_ops, max_cycle_val, capacity_binding
                    );
                }
                let makespan_usize = actual_makespan as usize;
                let alu_capacity = machine.alu_slots as usize * makespan_usize;
                let valu_capacity = machine.valu_slots as usize * makespan_usize;
                let load_capacity = machine.load_slots as usize * makespan_usize;
                let store_capacity = machine.store_slots as usize * makespan_usize;
                let flow_capacity = machine.flow_slots as usize * makespan_usize;
                let pct = |used: usize, cap: usize| {
                    if cap > 0 {
                        (used as f64 / cap as f64) * 100.0
                    } else {
                        0.0
                    }
                };
                eprintln!(
                    "  Slot usage: ALU={}/{} ({:.1}%) VALU={}/{} ({:.1}%) Load={}/{} ({:.1}%) Store={}/{} ({:.1}%) Flow={}/{} ({:.1}%)",
                    summary.alu_total,
                    alu_capacity,
                    pct(summary.alu_total, alu_capacity),
                    summary.valu_total,
                    valu_capacity,
                    pct(summary.valu_total, valu_capacity),
                    summary.load_total,
                    load_capacity,
                    pct(summary.load_total, load_capacity),
                    summary.store_total,
                    store_capacity,
                    pct(summary.store_total, store_capacity),
                    summary.flow_total,
                    flow_capacity,
                    pct(summary.flow_total, flow_capacity)
                );
                eprintln!(
                    "  ConstChoice: {} AddImm, {} Load, {} AluDerive",
                    summary.const_add_imm, summary.const_load, summary.const_alu_derive
                );
                eprintln!(
                    "  SelectChoice: {} FMA, {} vselect",
                    summary.select_fma, summary.select_vselect
                );
                eprintln!(
                    "  VALU slots: {} total (FMA: {}, vbroadcast: {}, other: {}; FMA and vbroadcast not shiftable)",
                    summary.valu_total,
                    summary.valu_fma_slots,
                    summary.valu_vbroadcast_slots,
                    summary.valu_other_slots
                );
                let n_add_imm = const_choice_encoding
                    .values()
                    .filter(|e| matches!(e, crate::types::ConstChoiceEncoding::AddImm))
                    .count();
                let n_alu_derive = const_choice_encoding
                    .values()
                    .filter(|e| matches!(e, crate::types::ConstChoiceEncoding::AluDerive(_)))
                    .count();
                eprintln!(
                    "  Shifted {} VALU ops, {} ConstChoice AddImm ops, {} ConstChoice AluDerive ops, {} SelectChoice FMA ops",
                    shifted_ops.len(),
                    n_add_imm,
                    n_alu_derive,
                    select_fma_op_ids.len()
                );
            }

            // schedule_constraints already built above (for active set and for BulkMoveCompactionPass).
            // CP-SAT returns the raw solution; compaction (bulk-move) is a separate pipeline pass.

            let out_cycle_assignments = cycle_assignments;
            let out_shifted_alu_cycles = shifted_alu_cycles;
            let out_makespan = actual_makespan as usize;
            let out_empty_count = empty_count;

            let lb_cycles = if status == CpsatStatus::Optimal {
                Some(out_makespan)
            } else {
                Some(makespan_lower_bound(response.best_objective_bound, objective_scale))
            };

            Ok(Some(SsaCpsatResult {
                cycle_assignments: out_cycle_assignments,
                makespan: out_makespan,
                status,
                solve_time_secs: solve_time,
                gap,
                shifted_ops,
                shifted_alu_cycles: out_shifted_alu_cycles,
                const_choice_encoding,
                select_choice_encoding,
                empty_cycle_count: out_empty_count,
                makespan_driven_by_capacity: (out_makespan as i64) == capacity_lower_bound,
                lb_cycles,
                eliminated_vbroadcast_ops,
                eliminated_vb_to_scalar_producer,
                cse_group_results,
                schedule_constraints,
            }))
        }
        _ => {
            if verbose {
                eprintln!("SSA CP-SAT: no solution found, status {:?}", status);
                if status == CpsatStatus::Unknown {
                    eprintln!(
                        "  (Unknown usually means time limit or incomplete hint; ensure cycle, ConstChoice, SelectChoice, and segment vars are hinted.)"
                    );
                }
            }
            // Export model on INFEASIBLE for debugging (CPSAT_EXPORT_INFEASIBLE=1)
            if status == CpsatStatus::Infeasible {
                if std::env::var("CPSAT_EXPORT_INFEASIBLE").is_ok() {
                    let path = std::env::var("CPSAT_EXPORT_INFEASIBLE_PATH")
                        .unwrap_or_else(|_| "cpsat_infeasible_model.pb".to_string());
                    let proto = model.proto();
                    let bytes = prost::Message::encode_to_vec(proto);
                    if std::fs::write(&path, &bytes).is_ok() {
                        eprintln!("  [INFEASIBLE] Model exported to {}", path);
                    }
                }
            }
            Ok(None)
        }
    }
}

/// Stub implementation when cpsat feature is not enabled
#[cfg(not(feature = "cpsat"))]
pub fn schedule_ssa_cpsat(
    _ops: &[SsaOp],
    _config: SsaCpsatConfig,
    _hints: SsaCpsatHints<'_>,
    _greedy_within_scratch_budget: Option<bool>,
) -> Result<Option<SsaCpsatResult>, String> {
    eprintln!("SSA CP-SAT scheduler not available: compile with --features cpsat");
    Ok(None)
}
