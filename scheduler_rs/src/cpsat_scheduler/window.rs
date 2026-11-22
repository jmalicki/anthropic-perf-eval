//! Window optimality probe: re-optimise one window of cycles of a known schedule with the
//! full CP-SAT model, everything outside the window held at the reference schedule.
//!
//! This is an analysis tool (off by default), not a pipeline phase. It answers "is the
//! reference schedule locally near-optimal, or does a smaller model find improvements that
//! the whole-program solve cannot reach?". See `docs/WINDOW_OPTIMALITY_PROBE.md`.
//!
//! # How the window is imposed on the full model
//!
//! [`schedule_ssa_cpsat`](super::schedule_ssa_cpsat) builds its usual model (every op, every
//! precedence offset, VALU shift, ConstChoice/SelectChoice encodings, the scratch cumulative)
//! and then, when [`SsaCpsatHints::window`](super::SsaCpsatHints) is set, adds the
//! constraints of [`apply_window_constraints`]. Each op of the reference schedule falls in one
//! [`WindowRegion`] by the cycles its slots occupy:
//!
//! - **Before** (some slot before the window start): every cycle variable fixed to the
//!   reference cycle, encodings fixed.
//! - **Inside** (every slot in `[start, start + len)`): free, but kept inside the window;
//!   encodings free (or fixed with [`CpsatWindow::fix_window_encodings`]).
//! - **After** (every slot at or after the start, some at or after the end): fixed relative to
//!   each other, encodings fixed, and moved as a block by one shared variable `delta` in
//!   `[0, len]`: `cycle = reference cycle - delta`.
//! - **Free** (no reference cycle, i.e. dead stores the greedy did not place): unconstrained.
//!
//! With the objective `makespan` alone, minimising the makespan maximises `delta` whenever the
//! last op is after the window: the cycles saved are exactly the cycles by which the rest of
//! the schedule can start earlier, which is the real makespan gain. A window that reaches the
//! end of the schedule has no After ops and the objective is the window's own end. Moving the
//! After block keeps every constraint between After ops (precedence, slot counts, liveness
//! shape); the model still checks everything that crosses the boundaries: precedence from
//! Before and Inside ops, slot capacity where Inside and After ops now share cycles, and the
//! scratch cumulative over the whole schedule (values defined before and read after the window
//! are counted at every cycle they are live).
//!
//! Not pinned: the cycle variable of an op that runs shifted (its eight ALU lanes and its
//! completion are pinned; the cycle variable only orders it after its producers), and the
//! shared SUB of a CSE group (`cse_sub_cycle_*`), which has no op of its own. Both are
//! constrained by everything the full model says about them, so any solution is a schedule
//! of the full model.

use std::collections::{HashMap, HashSet};

use crate::types::{ConstChoiceEncoding, Opcode, SsaOp, VLEN};

/// A complete schedule to hold fixed outside the window (op indices are positions in the
/// slice handed to the solver).
#[derive(Debug, Clone, Default)]
pub struct WindowReference {
    /// Per op index: the cycle in the sense of `SsaCpsatResult::cycle_assignments` (the
    /// completion cycle, i.e. the last lane, for an op that runs shifted). Negative = no cycle
    /// (dead store); such ops are left free.
    pub cycles: Vec<i32>,
    /// Op id -> the eight lane cycles of an op that runs shifted. An op in `shifted_ops` that
    /// is missing here runs all eight lanes at its cycle (the greedy's convention).
    pub shifted_alu_cycles: HashMap<usize, Vec<i32>>,
    /// Op ids that run shifted to the ALU.
    pub shifted_ops: HashSet<usize>,
    /// Op ids of SelectChoice ops lowered as multiply-add (the rest are vselect).
    pub select_fma_ops: HashSet<usize>,
    /// Op id -> ConstChoice encoding. Missing = Load.
    pub const_choice_encoding: HashMap<usize, ConstChoiceEncoding>,
    /// Op ids of vbroadcasts that are eliminated.
    pub eliminated_vbroadcast_ops: HashSet<usize>,
}

impl WindowReference {
    /// Lane cycles of op `idx` when it runs shifted in the reference.
    fn lanes(&self, idx: usize, op: &SsaOp) -> Option<Vec<i32>> {
        if !self.shifted_ops.contains(&op.id) {
            return None;
        }
        let cycle = *self.cycles.get(idx)?;
        Some(
            self.shifted_alu_cycles
                .get(&op.id)
                .cloned()
                .unwrap_or_else(|| vec![cycle; VLEN]),
        )
    }

    /// First and last cycle at which op `idx` occupies a slot in the reference schedule.
    ///
    /// # Returns
    /// `None` when the op has no cycle (dead store).
    pub fn slot_span(&self, idx: usize, op: &SsaOp) -> Option<(i32, i32)> {
        let cycle = *self.cycles.get(idx)?;
        if cycle < 0 {
            return None;
        }
        if let Some(lanes) = self.lanes(idx, op) {
            let lo = lanes.iter().copied().min().unwrap_or(cycle);
            let hi = lanes.iter().copied().max().unwrap_or(cycle);
            return Some((lo, hi));
        }
        // SelectChoice on the multiply-add path without a delta: SUB at cycle, FMA at cycle + 1.
        if op.opcode == Opcode::SelectChoice
            && op.sub_temp_vreg.is_some()
            && op.delta_vreg.is_none()
            && self.select_fma_ops.contains(&op.id)
        {
            return Some((cycle, cycle + 1));
        }
        Some((cycle, cycle))
    }

    /// Last cycle any op occupies, plus one (the reference makespan).
    pub fn makespan(&self, ops: &[SsaOp]) -> i32 {
        ops.iter()
            .enumerate()
            .filter_map(|(idx, op)| self.slot_span(idx, op).map(|(_, hi)| hi + 1))
            .max()
            .unwrap_or(0)
    }
}

/// Where an op sits relative to the window in the reference schedule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowRegion {
    /// Some slot before the window start: fixed at its reference cycles.
    Before,
    /// Every slot inside the window: free to move within it.
    Inside,
    /// Every slot at or after the window start and some at or after its end: moves with the
    /// shared `delta`.
    After,
    /// No reference cycle (dead store): unconstrained.
    Free,
}

/// One window of the probe.
#[derive(Debug, Clone, Default)]
pub struct CpsatWindow {
    /// First cycle of the window.
    pub start: i32,
    /// Number of cycles in the window (`[start, start + len)`). 0 fixes the whole schedule.
    pub len: i32,
    /// Fix the encodings (shift, ConstChoice, SelectChoice, vbroadcast elimination) of ops
    /// inside the window to the reference's as well. False = encodings inside are free.
    pub fix_window_encodings: bool,
    /// The schedule held fixed outside the window.
    pub reference: WindowReference,
}

impl CpsatWindow {
    /// Region of op `idx` for this window.
    pub fn region(&self, idx: usize, op: &SsaOp) -> WindowRegion {
        let Some((lo, hi)) = self.reference.slot_span(idx, op) else {
            return WindowRegion::Free;
        };
        let end = self.start + self.len;
        if lo < self.start {
            WindowRegion::Before
        } else if hi < end {
            WindowRegion::Inside
        } else {
            WindowRegion::After
        }
    }
}

/// The model variables the window constraints act on (all keyed by op index).
#[cfg(feature = "cpsat")]
pub(crate) struct WindowModelVars<'a> {
    pub cycle_vars: &'a [cp_sat::builder::IntVar],
    pub alu_cycle_vars: &'a HashMap<usize, Vec<cp_sat::builder::IntVar>>,
    pub completion_cycle_vars: &'a HashMap<usize, cp_sat::builder::IntVar>,
    pub select_choice_fma_last_cycle: &'a HashMap<usize, cp_sat::builder::IntVar>,
    pub shift_vars: &'a HashMap<usize, cp_sat::builder::BoolVar>,
    pub const_choice_vars: &'a HashMap<usize, cp_sat::builder::BoolVar>,
    pub const_choice_encoding_vars: &'a HashMap<usize, Vec<cp_sat::builder::BoolVar>>,
    pub const_choice_alu_encoding_info: &'a HashMap<usize, Vec<(usize, usize, Opcode)>>,
    pub select_choice_vars: &'a HashMap<usize, cp_sat::builder::BoolVar>,
    pub eliminate_vb_vars: &'a HashMap<usize, cp_sat::builder::BoolVar>,
}

/// Counts of ops per region, for logging and the probe report.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WindowRegionCounts {
    pub before: usize,
    pub inside: usize,
    pub after: usize,
    pub free: usize,
}

/// Add the window constraints to a full model (see the module docs).
///
/// # Arguments
/// * `model` - The model built by `schedule_ssa_cpsat`, with every variable created.
/// * `ops` - The ops of that model.
/// * `window` - The window and its reference schedule.
/// * `vars` - The model's variables.
///
/// # Returns
/// The number of ops in each region.
#[cfg(feature = "cpsat")]
pub(crate) fn apply_window_constraints(
    model: &mut cp_sat::builder::CpModelBuilder,
    ops: &[SsaOp],
    window: &CpsatWindow,
    vars: &WindowModelVars<'_>,
) -> WindowRegionCounts {
    use cp_sat::builder::{BoolVar, IntVar, LinearExpr};

    let reference = &window.reference;
    let start = window.start as i64;
    let last = (window.start + window.len) as i64 - 1;
    let delta = model.new_int_var_with_name([(0, window.len.max(0) as i64)], "window_delta".to_string());
    model.add_hint(delta, 0);

    let id_to_idx: HashMap<usize, usize> = ops.iter().enumerate().map(|(i, op)| (op.id, i)).collect();

    // var == value (Before) or var + delta == value (After).
    let pin = |model: &mut cp_sat::builder::CpModelBuilder, var: IntVar, value: i32, shifted: bool, name: String| {
        let c = if shifted {
            model.add_eq(LinearExpr::from(var) + LinearExpr::from(delta), value as i64)
        } else {
            model.add_eq(LinearExpr::from(var), value as i64)
        };
        model.set_constraint_name(c, &name);
    };
    let fix_bool = |model: &mut cp_sat::builder::CpModelBuilder, var: BoolVar, value: bool, name: String| {
        let c = model.add_eq(LinearExpr::from(var), i64::from(value));
        model.set_constraint_name(c, &name);
    };
    let keep_inside = |model: &mut cp_sat::builder::CpModelBuilder, var: IntVar, name: &str| {
        let c = model.add_ge(LinearExpr::from(var), start);
        model.set_constraint_name(c, &format!("{}_ge_start", name));
        let c = model.add_le(LinearExpr::from(var), last);
        model.set_constraint_name(c, &format!("{}_le_end", name));
    };

    let mut counts = WindowRegionCounts::default();
    for (idx, op) in ops.iter().enumerate() {
        let region = window.region(idx, op);
        match region {
            WindowRegion::Before => counts.before += 1,
            WindowRegion::Inside => counts.inside += 1,
            WindowRegion::After => counts.after += 1,
            WindowRegion::Free => counts.free += 1,
        }
        if region == WindowRegion::Free {
            continue;
        }
        let fix_encodings = region != WindowRegion::Inside || window.fix_window_encodings;
        if fix_encodings {
            if let Some(&v) = vars.shift_vars.get(&idx) {
                fix_bool(model, v, reference.shifted_ops.contains(&op.id), format!("window_shift_{}", idx));
            }
            if let Some(&v) = vars.select_choice_vars.get(&idx) {
                fix_bool(model, v, reference.select_fma_ops.contains(&op.id), format!("window_select_{}", idx));
            }
            if let Some(&v) = vars.eliminate_vb_vars.get(&idx) {
                fix_bool(
                    model,
                    v,
                    reference.eliminated_vbroadcast_ops.contains(&op.id),
                    format!("window_elim_vb_{}", idx),
                );
            }
            let encoding = reference.const_choice_encoding.get(&op.id);
            if let Some(&v) = vars.const_choice_vars.get(&idx) {
                fix_bool(
                    model,
                    v,
                    matches!(encoding, Some(ConstChoiceEncoding::AddImm)),
                    format!("window_const_{}", idx),
                );
            }
            if let Some(use_vars) = vars.const_choice_encoding_vars.get(&idx) {
                let chosen = match encoding {
                    None | Some(ConstChoiceEncoding::Load) => Some(0),
                    Some(ConstChoiceEncoding::AddImm) => Some(1),
                    Some(ConstChoiceEncoding::AluDerive(info)) => vars
                        .const_choice_alu_encoding_info
                        .get(&idx)
                        .and_then(|alts| {
                            alts.iter().position(|&(l, r, opcode)| {
                                Some(&l) == id_to_idx.get(&info.left_op_id)
                                    && Some(&r) == id_to_idx.get(&info.right_op_id)
                                    && opcode == info.opcode
                            })
                        })
                        .map(|k| k + 2),
                };
                if let Some(chosen) = chosen {
                    let c = model.add_eq(LinearExpr::from(use_vars[chosen]), 1);
                    model.set_constraint_name(c, &format!("window_const_enc_{}_{}", idx, chosen));
                }
            }
        }

        match region {
            WindowRegion::Inside => {
                keep_inside(model, vars.cycle_vars[idx], &format!("window_c{}", idx));
                if let Some(&v) = vars.completion_cycle_vars.get(&idx) {
                    keep_inside(model, v, &format!("window_completion_c{}", idx));
                }
                if let Some(lanes) = vars.alu_cycle_vars.get(&idx) {
                    for (lane, &v) in lanes.iter().enumerate() {
                        keep_inside(model, v, &format!("window_alu_c{}_{}", idx, lane));
                    }
                }
                if let Some(&v) = vars.select_choice_fma_last_cycle.get(&idx) {
                    let c = model.add_le(LinearExpr::from(v), last);
                    model.set_constraint_name(c, &format!("window_fma_last_{}_le_end", idx));
                }
            }
            WindowRegion::Before | WindowRegion::After => {
                let moves = region == WindowRegion::After;
                let cycle = reference.cycles[idx];
                let tag = if moves { "after" } else { "before" };
                match reference.lanes(idx, op) {
                    Some(lanes) => {
                        // Runs shifted: pin the eight lanes and the completion (the last lane).
                        if let Some(alu) = vars.alu_cycle_vars.get(&idx) {
                            for (lane, (&v, &c)) in alu.iter().zip(lanes.iter()).enumerate() {
                                pin(model, v, c, moves, format!("window_{}_alu_c{}_{}", tag, idx, lane));
                            }
                        }
                        if let Some(&v) = vars.completion_cycle_vars.get(&idx) {
                            let done = lanes.iter().copied().max().unwrap_or(cycle);
                            pin(model, v, done, moves, format!("window_{}_completion_c{}", tag, idx));
                        }
                    }
                    None => {
                        pin(model, vars.cycle_vars[idx], cycle, moves, format!("window_{}_c{}", tag, idx));
                        if let Some(&v) = vars.completion_cycle_vars.get(&idx) {
                            pin(model, v, cycle, moves, format!("window_{}_completion_c{}", tag, idx));
                        }
                    }
                }
            }
            WindowRegion::Free => {}
        }
    }
    counts
}

// ============================================================================
// Probe driver: sweep windows over a schedule, then roll non-overlapping windows
// ============================================================================

/// What to probe. Read from the JSON file named by `CPSAT_WINDOW_PROBE` (see
/// [`WindowProbeSpec::from_env`]); unknown keys are rejected, missing keys take the default.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct WindowProbeSpec {
    /// Window sizes W in cycles (default `[50]`).
    pub window_sizes: Vec<i32>,
    /// Sweep stride as a fraction of W (default 0.5 = W/2; 1.0 = back to back).
    pub stride_fraction: f64,
    /// Encoding modes for the sweep: "free" (encodings inside the window free) and/or "fixed"
    /// (held at the reference's). Default both.
    pub encodings: Vec<String>,
    /// Run the independent sweep (every window against the unchanged reference). Default true.
    pub sweep: bool,
    /// Run the rolling pass for each W: non-overlapping windows from the start, each solved
    /// against the schedule the previous ones improved (encodings free). Default true.
    pub rolling: bool,
    /// Only windows whose start lies in `[lo, hi)` (default: the whole schedule).
    pub cycle_range: Option<(i32, i32)>,
    /// CP-SAT deterministic-time budget per window (default 30).
    pub deterministic_time: f64,
    /// Wall-clock backstop per window in seconds (default 900).
    pub time_limit_secs: f64,
    /// CP-SAT workers per window (default: the full-kernel tuning's).
    pub workers: Option<i32>,
    /// Report path (JSON, rewritten after every window). Required.
    pub output: String,
}

impl Default for WindowProbeSpec {
    fn default() -> Self {
        Self {
            window_sizes: vec![50],
            stride_fraction: 0.5,
            encodings: vec!["free".to_string(), "fixed".to_string()],
            sweep: true,
            rolling: true,
            cycle_range: None,
            deterministic_time: 30.0,
            time_limit_secs: 900.0,
            workers: None,
            output: String::new(),
        }
    }
}

impl WindowProbeSpec {
    /// Parse and check a spec.
    ///
    /// # Arguments
    /// * `text` - JSON object with any subset of the fields; `output` is required.
    ///
    /// # Returns
    /// The spec, or a message naming the bad key or value.
    pub fn from_json(text: &str) -> Result<Self, String> {
        let spec: Self = serde_json::from_str(text).map_err(|e| format!("window probe spec: {}", e))?;
        if spec.output.is_empty() {
            return Err("window probe spec: `output` (report path) is required".to_string());
        }
        if spec.window_sizes.is_empty() || spec.window_sizes.iter().any(|&w| w < 1) {
            return Err(format!("window probe spec: window_sizes must be positive, got {:?}", spec.window_sizes));
        }
        if !(spec.stride_fraction > 0.0 && spec.stride_fraction.is_finite()) {
            return Err(format!("window probe spec: stride_fraction must be > 0, got {}", spec.stride_fraction));
        }
        if let Some(bad) = spec.encodings.iter().find(|e| *e != "free" && *e != "fixed") {
            return Err(format!("window probe spec: encodings must be \"free\" or \"fixed\", got {:?}", bad));
        }
        if !(spec.deterministic_time > 0.0 && spec.time_limit_secs > 0.0) {
            return Err("window probe spec: deterministic_time and time_limit_secs must be > 0".to_string());
        }
        Ok(spec)
    }

    /// The spec named by the `CPSAT_WINDOW_PROBE` environment variable (a JSON file path).
    ///
    /// # Returns
    /// `Ok(None)` when the variable is unset (the normal pipeline), the spec otherwise.
    pub fn from_env() -> Result<Option<Self>, String> {
        let Ok(path) = std::env::var("CPSAT_WINDOW_PROBE") else {
            return Ok(None);
        };
        let text = std::fs::read_to_string(&path).map_err(|e| format!("CPSAT_WINDOW_PROBE={}: {}", path, e))?;
        Self::from_json(&text).map(Some)
    }

    /// Sweep stride for window size `w` (at least 1).
    pub fn stride(&self, w: i32) -> i32 {
        ((w as f64 * self.stride_fraction).round() as i32).max(1)
    }
}

/// One window solve in the probe report.
#[derive(Debug, Clone, serde::Serialize)]
pub struct WindowProbeEntry {
    /// "sweep" or "rolling".
    pub phase: String,
    pub w: i32,
    pub start: i32,
    /// "free" or "fixed" (encodings inside the window).
    pub encodings: String,
    /// CP-SAT status ("optimal", "feasible", "no solution", or "error: ...").
    pub status: String,
    /// Makespan of the reference this window was solved against.
    pub reference_makespan: i32,
    /// Makespan of the window solution (None when no solution).
    pub makespan: Option<i32>,
    /// Cycles saved: reference makespan minus the solution's.
    pub saved: i32,
    /// Proven lower bound on the makespan with this window (None when unknown).
    pub lb: Option<i32>,
    /// Most cycles this window could save (reference makespan minus `lb`).
    pub saved_bound: Option<i32>,
    /// Wall-clock seconds for model build plus solve (depends on machine load).
    pub wall_secs: f64,
    pub before: usize,
    pub inside: usize,
    pub after: usize,
    /// Ops inside the window whose encoding differs from the reference in the solution.
    pub shift_changes: usize,
    pub select_changes: usize,
    pub const_changes: usize,
}

/// The probe report written to [`WindowProbeSpec::output`].
#[derive(Debug, Clone, serde::Serialize)]
pub struct WindowProbeReport {
    pub spec: WindowProbeSpec,
    pub num_ops: usize,
    /// Makespan of the reference schedule (the warm start the pipeline built).
    pub base_makespan: i32,
    /// Outcome of the all-fixed check (a window of length 0): the reference must be a solution
    /// of the model, or no window result means anything.
    pub base_check_status: String,
    pub entries: Vec<WindowProbeEntry>,
    /// Per W: (W, final makespan of the rolling pass).
    pub rolling_final: Vec<(i32, i32)>,
}

impl WindowProbeReport {
    fn write(&self) -> Result<(), String> {
        let text = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(&self.spec.output, text).map_err(|e| format!("{}: {}", self.spec.output, e))
    }
}

/// The parts of the full-kernel hints that do not change between windows.
#[derive(Debug, Clone, Copy, Default)]
pub struct WindowProbeContext<'a> {
    pub forced_shifts: Option<&'a HashSet<usize>>,
    pub cse_groups: Option<&'a [super::cse_groups::CseGroup]>,
    pub vector_group_lanes: Option<&'a HashMap<crate::types::VReg, Vec<crate::types::VReg>>>,
}

impl WindowReference {
    /// The reference given by a CP-SAT solution.
    ///
    /// # Arguments
    /// * `result` - The solution.
    /// * `dead` - Op indices that have no reference cycle in the base (dead stores the greedy
    ///   did not place); they stay free, so the model's liveness bookkeeping, which skips
    ///   ops with a negative hint, is the same in every window.
    pub fn from_result(result: &super::SsaCpsatResult, dead: &[usize]) -> Self {
        let mut cycles = result.cycle_assignments.clone();
        for &idx in dead {
            if let Some(c) = cycles.get_mut(idx) {
                *c = -1;
            }
        }
        Self {
            cycles,
            shifted_alu_cycles: result.shifted_alu_cycles.clone(),
            shifted_ops: result.shifted_ops.clone(),
            select_fma_ops: result
                .select_choice_encoding
                .iter()
                .filter(|(_, e)| matches!(e, crate::types::SelectChoiceEncoding::Fma))
                .map(|(&id, _)| id)
                .collect(),
            const_choice_encoding: result.const_choice_encoding.clone(),
            eliminated_vbroadcast_ops: result.eliminated_vbroadcast_ops.clone(),
        }
    }
}

/// Solve one window against its reference with the full model.
#[cfg(feature = "cpsat")]
pub fn solve_window(
    ops: &[SsaOp],
    config: &super::SsaCpsatConfig,
    ctx: &WindowProbeContext<'_>,
    window: &CpsatWindow,
) -> Result<Option<super::SsaCpsatResult>, String> {
    let reference = &window.reference;
    let const_add_imm: HashSet<usize> = reference
        .const_choice_encoding
        .iter()
        .filter(|(_, e)| matches!(e, ConstChoiceEncoding::AddImm))
        .map(|(&id, _)| id)
        .collect();
    let hints = super::SsaCpsatHints {
        greedy_hints: Some(&reference.cycles),
        shift_hints: Some(&reference.shifted_ops),
        forced_shifts: ctx.forced_shifts,
        select_choice_hints: Some(&reference.select_fma_ops),
        const_add_imm_hints: Some(&const_add_imm),
        eliminate_vbroadcast_hints: Some(&reference.eliminated_vbroadcast_ops),
        cse_groups: ctx.cse_groups,
        vector_group_lanes: ctx.vector_group_lanes,
        alu_lane_hints: Some(&reference.shifted_alu_cycles),
        window: Some(window),
        ..Default::default()
    };
    super::schedule_ssa_cpsat(ops, config.clone(), hints, Some(true))
}

/// Encoding changes among the ops inside the window: (shift, select, const).
fn encoding_changes(ops: &[SsaOp], window: &CpsatWindow, result: &super::SsaCpsatResult) -> (usize, usize, usize) {
    let reference = &window.reference;
    let mut changes = (0, 0, 0);
    for (idx, op) in ops.iter().enumerate() {
        if window.region(idx, op) != WindowRegion::Inside {
            continue;
        }
        if reference.shifted_ops.contains(&op.id) != result.shifted_ops.contains(&op.id) {
            changes.0 += 1;
        }
        if op.opcode == Opcode::SelectChoice {
            let fma = matches!(
                result.select_choice_encoding.get(&op.id),
                Some(crate::types::SelectChoiceEncoding::Fma)
            );
            if fma != reference.select_fma_ops.contains(&op.id) {
                changes.1 += 1;
            }
        }
        if op.opcode == Opcode::ConstChoice {
            let before = reference.const_choice_encoding.get(&op.id).unwrap_or(&ConstChoiceEncoding::Load);
            let after = result.const_choice_encoding.get(&op.id).unwrap_or(&ConstChoiceEncoding::Load);
            if before != after {
                changes.2 += 1;
            }
        }
    }
    changes
}

/// Run the window optimality probe on the full-kernel model.
///
/// # Arguments
/// * `ops` - The ops the full-kernel CP-SAT would schedule.
/// * `base_config` - The full-kernel CP-SAT configuration. The probe keeps the model options
///   and sets the objective to makespan alone, `verbose` off and the per-window budgets from
///   `spec`.
/// * `ctx` - Hints that stay the same for every window.
/// * `base` - The reference schedule (the pipeline's warm start).
/// * `spec` - What to sweep.
///
/// # Returns
/// The report (also written to `spec.output` after every window) and a full-kernel
/// solution: the best rolling-pass schedule when one improved on the reference, otherwise the
/// reference itself as solved by the all-fixed check.
#[cfg(feature = "cpsat")]
pub fn run_window_probe(
    ops: &[SsaOp],
    base_config: &super::SsaCpsatConfig,
    ctx: &WindowProbeContext<'_>,
    base: WindowReference,
    spec: &WindowProbeSpec,
) -> Result<(WindowProbeReport, super::SsaCpsatResult), String> {
    let mut config = base_config.clone();
    config.verbose = false;
    config.stop_after_first_solution = false;
    config.force_zero_gap = false;
    config.deterministic_time_limit = Some(spec.deterministic_time);
    config.time_limit_secs = spec.time_limit_secs;
    config.tuning.objective = super::CpsatObjective::Makespan;
    if let Some(workers) = spec.workers {
        config.tuning.workers = workers;
    }
    let dead: Vec<usize> = base.cycles.iter().enumerate().filter(|(_, &c)| c < 0).map(|(i, _)| i).collect();
    let base_makespan = base.makespan(ops);

    let mut report = WindowProbeReport {
        spec: spec.clone(),
        num_ops: ops.len(),
        base_makespan,
        base_check_status: String::new(),
        entries: Vec::new(),
        rolling_final: Vec::new(),
    };

    // All-fixed check: with an empty window every op is pinned, so the only solution is the
    // reference itself. If the model rejects it, no window result means anything.
    let started = std::time::Instant::now();
    let all_fixed = CpsatWindow { start: 0, len: 0, fix_window_encodings: true, reference: base.clone() };
    let base_result = match solve_window(ops, &config, ctx, &all_fixed) {
        Ok(Some(r)) => r,
        Ok(None) => {
            report.base_check_status = "no solution".to_string();
            report.write()?;
            return Err("window probe: the reference schedule is not a solution of the CP-SAT model".to_string());
        }
        Err(e) => {
            report.base_check_status = format!("error: {}", e);
            report.write()?;
            return Err(format!("window probe: all-fixed check failed: {}", e));
        }
    };
    report.base_check_status = format!(
        "{} (makespan {}, {:.1}s)",
        base_result.status,
        base_result.makespan,
        started.elapsed().as_secs_f64()
    );
    eprintln!(
        "[window probe] {} ops, reference makespan {}, all-fixed check: {}",
        ops.len(),
        base_makespan,
        report.base_check_status
    );
    report.write()?;

    let in_range = |start: i32| spec.cycle_range.map_or(true, |(lo, hi)| start >= lo && start < hi);
    let range_lo = spec.cycle_range.map_or(0, |(lo, _)| lo.max(0));

    let record = |report: &mut WindowProbeReport,
                  phase: &str,
                  window: &CpsatWindow,
                  reference_makespan: i32,
                  outcome: &Result<Option<super::SsaCpsatResult>, String>,
                  wall_secs: f64|
     -> Result<(), String> {
        let mut counts = WindowRegionCounts::default();
        for (idx, op) in ops.iter().enumerate() {
            match window.region(idx, op) {
                WindowRegion::Before => counts.before += 1,
                WindowRegion::Inside => counts.inside += 1,
                WindowRegion::After => counts.after += 1,
                WindowRegion::Free => counts.free += 1,
            }
        }
        let (status, makespan, lb, changes) = match outcome {
            Ok(Some(r)) => (
                r.status.to_string(),
                Some(r.makespan as i32),
                r.lb_cycles.map(|l| l as i32),
                encoding_changes(ops, window, r),
            ),
            Ok(None) => ("no solution".to_string(), None, None, (0, 0, 0)),
            Err(e) => (format!("error: {}", e), None, None, (0, 0, 0)),
        };
        let entry = WindowProbeEntry {
            phase: phase.to_string(),
            w: window.len,
            start: window.start,
            encodings: if window.fix_window_encodings { "fixed" } else { "free" }.to_string(),
            status,
            reference_makespan,
            makespan,
            saved: makespan.map_or(0, |m| reference_makespan - m),
            lb,
            saved_bound: lb.map(|l| reference_makespan - l),
            wall_secs,
            before: counts.before,
            inside: counts.inside,
            after: counts.after,
            shift_changes: changes.0,
            select_changes: changes.1,
            const_changes: changes.2,
        };
        eprintln!(
            "[window probe] {} W={} [{}, {}) {}: {}, makespan {:?}, saved {} (bound {:?}), {} ops inside, {:.1}s; encoding changes: shift {}, select {}, const {}",
            entry.phase,
            entry.w,
            entry.start,
            entry.start + entry.w,
            entry.encodings,
            entry.status,
            entry.makespan,
            entry.saved,
            entry.saved_bound,
            entry.inside,
            entry.wall_secs,
            entry.shift_changes,
            entry.select_changes,
            entry.const_changes
        );
        report.entries.push(entry);
        report.write()
    };

    if spec.sweep {
        for &w in &spec.window_sizes {
            let stride = spec.stride(w);
            for encodings in &spec.encodings {
                let mut start = range_lo;
                while start < base_makespan && in_range(start) {
                    let window = CpsatWindow {
                        start,
                        len: w,
                        fix_window_encodings: encodings == "fixed",
                        reference: base.clone(),
                    };
                    let t = std::time::Instant::now();
                    let outcome = solve_window(ops, &config, ctx, &window);
                    record(&mut report, "sweep", &window, base_makespan, &outcome, t.elapsed().as_secs_f64())?;
                    if start + w >= base_makespan {
                        break;
                    }
                    start += stride;
                }
            }
        }
    }

    let mut best: Option<super::SsaCpsatResult> = None;
    if spec.rolling {
        for &w in &spec.window_sizes {
            let mut reference = base.clone();
            let mut makespan = base_makespan;
            let mut improved: Option<super::SsaCpsatResult> = None;
            let mut start = range_lo;
            while start < makespan && in_range(start) {
                let window = CpsatWindow { start, len: w, fix_window_encodings: false, reference: reference.clone() };
                let t = std::time::Instant::now();
                let outcome = solve_window(ops, &config, ctx, &window);
                record(&mut report, "rolling", &window, makespan, &outcome, t.elapsed().as_secs_f64())?;
                let mut saved = 0;
                if let Ok(Some(r)) = &outcome {
                    if (r.makespan as i32) < makespan {
                        saved = makespan - r.makespan as i32;
                        makespan = r.makespan as i32;
                        reference = WindowReference::from_result(r, &dead);
                        improved = Some(r.clone());
                    }
                }
                // The next window starts where this one now ends.
                start += (w - saved).max(1);
            }
            eprintln!("[window probe] rolling W={}: {} -> {} cycles", w, base_makespan, makespan);
            report.rolling_final.push((w, makespan));
            report.write()?;
            if let Some(r) = improved {
                if best.as_ref().map_or(true, |b| r.makespan < b.makespan) {
                    best = Some(r);
                }
            }
        }
    }
    Ok((report, best.unwrap_or(base_result)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Engine, Opcode, SsaOp};

    fn op(id: usize, opcode: Opcode, engine: Engine) -> SsaOp {
        SsaOp::new(id, engine, opcode, None, vec![])
    }

    #[test]
    fn regions_follow_slot_spans() {
        let ops = vec![
            op(0, Opcode::Add, Engine::Alu),
            op(1, Opcode::Add, Engine::Alu),
            op(2, Opcode::Add, Engine::Alu),
            op(3, Opcode::Add, Engine::Alu),
            op(4, Opcode::Xor, Engine::Valu),
        ];
        let mut reference = WindowReference {
            cycles: vec![1, 5, 9, -1, 9],
            ..Default::default()
        };
        // Op 4 runs shifted with lanes 4..=9: it starts before the window [5, 9).
        reference.shifted_ops.insert(4);
        reference.shifted_alu_cycles.insert(4, vec![4, 5, 6, 7, 8, 9, 9, 9]);
        let window = CpsatWindow { start: 5, len: 4, reference, ..Default::default() };
        let regions: Vec<_> = ops.iter().enumerate().map(|(i, o)| window.region(i, o)).collect();
        assert_eq!(
            regions,
            vec![
                WindowRegion::Before,
                WindowRegion::Inside,
                WindowRegion::After,
                WindowRegion::Free,
                WindowRegion::Before,
            ]
        );
        assert_eq!(window.reference.makespan(&ops), 10);
    }

    // ------------------------------------------------------------------------
    // Solver tests on a small program with every kind of choice
    // ------------------------------------------------------------------------

    /// Ops of a small program: ConstChoice constants, a vbroadcast, two shiftable VALU
    /// chains, a SelectChoice (FMA path without delta) and its consumer, and a vstore.
    #[cfg(feature = "cpsat")]
    fn small_program() -> Vec<SsaOp> {
        use crate::kernel_builder::emit_ssa::{compute_ssa_dependencies, SsaEmitContext};

        let mut emit = SsaEmitContext::new();
        let c3 = emit.emit_const_choice(3);
        let c5 = emit.emit_const_choice(5);
        let v3 = emit.emit_vbroadcast(c3);
        let v5 = emit.emit_vbroadcast(c5);
        let mut a = emit.emit_valu(Opcode::Add, v3, v5);
        let mut b = emit.emit_valu(Opcode::Xor, v5, v3);
        for _ in 0..3 {
            a = emit.emit_valu(Opcode::Mul, a, v3);
            b = emit.emit_valu(Opcode::Add, b, v5);
        }
        let cond = emit.emit_valu(Opcode::And, a, v3);
        let sel = emit.emit_select_choice(cond, a, b);
        let out = emit.emit_valu(Opcode::Add, sel, b);
        emit.emit_vstore_imm(64, out);
        let vreg_def = std::mem::take(&mut emit.vreg_def);
        compute_ssa_dependencies(&mut emit.ops, &vreg_def, emit.zero_vreg, None);
        emit.ops
    }

    /// A feasible but loose reference: op `i` at cycle `2 * i`, every choice at its default
    /// (Load, vselect, no shift).
    #[cfg(feature = "cpsat")]
    fn loose_reference(ops: &[SsaOp]) -> WindowReference {
        WindowReference { cycles: (0..ops.len() as i32).map(|i| 2 * i).collect(), ..Default::default() }
    }

    #[cfg(feature = "cpsat")]
    fn probe_config() -> crate::cpsat_scheduler::SsaCpsatConfig {
        let mut config = crate::cpsat_scheduler::SsaCpsatConfig {
            enable_valu_shift: true,
            time_limit_secs: 60.0,
            deterministic_time_limit: Some(20.0),
            scratch_budget: 1536,
            ..Default::default()
        };
        config.tuning.objective = crate::cpsat_scheduler::CpsatObjective::Makespan;
        config.tuning.workers = 1;
        config
    }

    /// A window that covers the whole schedule leaves every op free, so its optimum is the
    /// full model's optimum from the same warm start.
    #[cfg(feature = "cpsat")]
    #[test]
    fn whole_schedule_window_matches_full_model_optimum() {
        let ops = small_program();
        let reference = loose_reference(&ops);
        let config = probe_config();
        let ctx = WindowProbeContext::default();

        let full = crate::cpsat_scheduler::schedule_ssa_cpsat(
            &ops,
            config.clone(),
            crate::cpsat_scheduler::SsaCpsatHints { greedy_hints: Some(&reference.cycles), ..Default::default() },
            Some(true),
        )
        .expect("full model")
        .expect("full model solution");
        assert_eq!(full.status, crate::cpsat_scheduler::CpsatStatus::Optimal);

        let window = CpsatWindow { start: 0, len: reference.makespan(&ops), fix_window_encodings: false, reference };
        let windowed = solve_window(&ops, &config, &ctx, &window).expect("window model").expect("window solution");
        assert_eq!(windowed.status, crate::cpsat_scheduler::CpsatStatus::Optimal);
        assert_eq!(windowed.makespan, full.makespan, "whole-schedule window must reach the full model's optimum");
        assert!(full.makespan < window.reference.makespan(&ops) as usize);
    }

    /// With an empty window every op is pinned: the only solution is the reference.
    #[cfg(feature = "cpsat")]
    #[test]
    fn empty_window_reproduces_reference() {
        let ops = small_program();
        let reference = loose_reference(&ops);
        let window = CpsatWindow { start: 0, len: 0, fix_window_encodings: true, reference: reference.clone() };
        let result = solve_window(&ops, &probe_config(), &WindowProbeContext::default(), &window)
            .expect("model")
            .expect("solution");
        assert_eq!(result.cycle_assignments, reference.cycles);
        assert_eq!(result.makespan as i32, reference.makespan(&ops));
        assert!(result.shifted_ops.is_empty());
    }

    /// A reference from an earlier CP-SAT solution can run a shifted op's lanes at different
    /// cycles; pinning and hinting must use those lanes (the hint capacity check once counted
    /// all eight at the completion cycle and panicked).
    #[cfg(feature = "cpsat")]
    #[test]
    fn shifted_lanes_at_different_cycles_are_pinned() {
        let ops = small_program();
        let mut reference = loose_reference(&ops);
        let idx = ops
            .iter()
            .position(|op| op.engine == crate::types::Engine::Valu && op.opcode == Opcode::Add)
            .expect("shiftable add");
        let c = reference.cycles[idx];
        let lanes = vec![c - 1, c - 1, c - 1, c - 1, c, c, c, c];
        reference.shifted_ops.insert(ops[idx].id);
        reference.shifted_alu_cycles.insert(ops[idx].id, lanes.clone());
        let window = CpsatWindow { start: 0, len: 0, fix_window_encodings: true, reference: reference.clone() };
        let result = solve_window(&ops, &probe_config(), &WindowProbeContext::default(), &window)
            .expect("model")
            .expect("solution");
        assert_eq!(result.shifted_alu_cycles.get(&ops[idx].id), Some(&lanes));
        assert_eq!(result.cycle_assignments[idx], c);
    }

    /// A window in the middle compacts its ops; ops before it keep their cycles and ops after
    /// it all move earlier by the cycles saved.
    #[cfg(feature = "cpsat")]
    #[test]
    fn middle_window_moves_after_block_by_saved_cycles() {
        let ops = small_program();
        let reference = loose_reference(&ops);
        let base = reference.makespan(&ops);
        let window = CpsatWindow { start: 6, len: 10, fix_window_encodings: false, reference: reference.clone() };
        let result = solve_window(&ops, &probe_config(), &WindowProbeContext::default(), &window)
            .expect("model")
            .expect("solution");
        let saved = base - result.makespan as i32;
        assert!(saved > 0, "a window over a loose schedule must save cycles");
        assert!(saved <= window.len);
        for (idx, op) in ops.iter().enumerate() {
            let got = result.cycle_assignments[idx];
            match window.region(idx, op) {
                WindowRegion::Before => assert_eq!(got, reference.cycles[idx], "op {} before the window moved", idx),
                WindowRegion::After => {
                    assert_eq!(got, reference.cycles[idx] - saved, "op {} after the window must move by {}", idx, saved)
                }
                WindowRegion::Inside => assert!((6..16).contains(&got), "op {} left the window: {}", idx, got),
                WindowRegion::Free => {}
            }
        }
    }

    /// With fixed encodings the ops inside the window keep the reference's choices.
    #[cfg(feature = "cpsat")]
    #[test]
    fn fixed_encodings_keep_reference_choices() {
        let ops = small_program();
        let reference = loose_reference(&ops);
        let len = reference.makespan(&ops);
        let window = CpsatWindow { start: 0, len, fix_window_encodings: true, reference };
        let result = solve_window(&ops, &probe_config(), &WindowProbeContext::default(), &window)
            .expect("model")
            .expect("solution");
        assert!(result.shifted_ops.is_empty(), "no op may shift when encodings are fixed to 'no shift'");
        assert!(result
            .select_choice_encoding
            .values()
            .all(|e| matches!(e, crate::types::SelectChoiceEncoding::Vselect)));
        assert!(result.const_choice_encoding.values().all(|e| matches!(e, ConstChoiceEncoding::Load)));
        assert_eq!(encoding_changes(&ops, &window, &result), (0, 0, 0));
    }

    /// The rolling pass, re-fixing at each improved schedule, gives a valid schedule no
    /// longer than the reference, and the all-fixed check reports the reference's makespan.
    #[cfg(feature = "cpsat")]
    #[test]
    fn rolling_pass_improves_loose_reference() {
        let ops = small_program();
        let reference = loose_reference(&ops);
        let base = reference.makespan(&ops);
        let dir = std::env::temp_dir().join(format!("window_probe_test_{}.json", std::process::id()));
        let spec = WindowProbeSpec {
            window_sizes: vec![8],
            sweep: false,
            output: dir.to_string_lossy().to_string(),
            deterministic_time: 20.0,
            time_limit_secs: 60.0,
            workers: Some(1),
            ..Default::default()
        };
        let (report, result) =
            run_window_probe(&ops, &probe_config(), &WindowProbeContext::default(), reference, &spec).expect("probe");
        let _ = std::fs::remove_file(&dir);
        assert_eq!(report.base_makespan, base);
        assert!(report.base_check_status.starts_with("optimal"), "{}", report.base_check_status);
        assert_eq!(report.rolling_final.len(), 1);
        assert_eq!(report.rolling_final[0].1, result.makespan as i32);
        assert!((result.makespan as i32) < base);
        let saved: i32 = report.entries.iter().map(|e| e.saved).sum();
        assert_eq!(saved, base - result.makespan as i32, "rolling savings must add up");
    }

    #[test]
    fn spec_rejects_bad_input() {
        assert!(WindowProbeSpec::from_json("{}").is_err(), "output is required");
        assert!(WindowProbeSpec::from_json(r#"{"output":"x","window_sizes":[0]}"#).is_err());
        assert!(WindowProbeSpec::from_json(r#"{"output":"x","encodings":["maybe"]}"#).is_err());
        assert!(WindowProbeSpec::from_json(r#"{"output":"x","bogus":1}"#).is_err());
        let spec = WindowProbeSpec::from_json(r#"{"output":"x","window_sizes":[25,50]}"#).unwrap();
        assert_eq!(spec.stride(25), 13);
        assert_eq!(spec.stride(50), 25);
    }
}
