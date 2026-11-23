//! When each op runs and when its result can be read: one definition for the CP-SAT model
//! and one for checking a solution.
//!
//! An op of the model has several representations of its timing: the issue cycle (`c{idx}`),
//! eight lane cycles and a completion cycle when it may be shifted from VALU to scalar ALU,
//! the multiply_add cycle of a select on the multiply-add path. Every past timing bug in the
//! model (BUGS.md: shifted consumer of an FMA select, CSE subtract after an FMA select,
//! forced-FMA hint) was a constraint written against one representation while another was
//! the one in force. This module is the only place that knows the representations:
//! constraint generators in `model.rs` ask an [`OpTimes`] for the times they need, and the
//! post-solve validators ask a [`ScheduledTimes`] for the same quantities computed from the
//! solution.
//!
//! # Definitions
//!
//! For op `o` (model side, as linear expressions; solution side, as cycles):
//!
//! | Name | Model ([`OpTimes`]) | Solution ([`ScheduledTimes`]) |
//! |---|---|---|
//! | issue | `c_o` | `cycle_assignments[o]` (completion when shifted) |
//! | exec end | `completion_o` if shiftable, else `c_o` | last lane if shifted, else the cycle |
//! | result latency | 1 on the multiply-add path of a select with a `sub_temp` (sub at c, multiply_add at c+1), else 0 | same |
//! | visible | exec end + 1 + result latency | same |
//! | lane visible | lane `i` + 1 when shifted, issue + 1 when not | lane `i` + 1 when shifted, else visible |
//! | read latency | 1 on the multiply-add path of any select (it reads its operands at c+1), else 0 | same |
//! | last read end | exec end + 1 + read latency | same |
//! | finish | completion when shifted, multiply_add cycle on the FMA path, else `c_o` | |
//! | memory access | `c_o` (memory ops are never shifted and never selects) | the cycle |
//!
//! # Mode invariants
//!
//! Stated once, where the variables get their meaning:
//!
//! * shiftable op: `completion >= c` and `completion >= lane_i` for every lane; with
//!   `--valu-shift valu`, `completion = c` ([`OpTimes::add_shift_invariants`]).
//! * select with a multiply_add cycle: `fma_cycle = c + 1` ([`OpTimes::add_fma_cycle_invariant`]).
//!
//! By default (`--cpsat-lane-model free`) the lanes of an op that is not shifted, and `c` of
//! an op that is, are not pinned: the model leaves the inactive representation free
//! (CPSAT_VARIABLE_MODELLING_REVIEW.md 4.2). `sorted` orders the lanes and `pinned` sets an
//! unshifted op's lanes to `c`, both in `add_shift_invariants`; under `pinned`,
//! [`OpTimes::lane_ready`] gives one unconditional per-lane precedence (docs/CPSAT_LANE_MODEL.md).
//!
//! # Where the model is weaker than the definitions
//!
//! Kept as they were when this module was introduced (it did not change the model); each is
//! a place where a generator uses a weaker or different time than the definition:
//!
//! * per-lane precedence from a select on the multiply-add path uses `c + 1`
//!   ([`OpTimes::lane_ready`]); separate rows add the latency for the same edge.
//! * the main cycle of a reader is ordered after `completion` of a shiftable producer and
//!   is itself free when the reader is shifted, so these rows only bind the lanes.
//! * symmetry orderings, `segment_end`, live-in interval ends and live-out interval starts
//!   use the issue cycle even for shifted ops.

use std::collections::{HashMap, HashSet};

#[cfg(feature = "cpsat")]
use cp_sat::builder::{BoolVar, CpModelBuilder, IntVar, LinearExpr};
#[cfg(feature = "cpsat")]
use cp_sat::proto::CpSolverResponse;

#[cfg(feature = "cpsat")]
use super::config::{ConstChoiceMode, CpsatLaneModel, SelectChoiceMode, ValuShiftMode};
#[cfg(feature = "cpsat")]
use super::cse_groups::CseGroup;
use crate::types::{Opcode, SelectChoiceEncoding, SsaOp};
#[cfg(feature = "cpsat")]
use crate::types::VLEN;

// ============================================================================
// Model side
// ============================================================================

/// One way an op's time can be, and the literal that selects it.
#[cfg(feature = "cpsat")]
#[derive(Clone)]
pub(crate) struct Alternative {
    /// The time (a linear expression over model variables)
    pub(crate) value: LinearExpr,
    /// Literal under which this alternative holds (`None`: always)
    pub(crate) when: Option<BoolVar>,
    /// Short name used in constraint names
    pub(crate) label: &'static str,
}

/// Whether a SelectChoice takes the multiply-add path.
#[cfg(feature = "cpsat")]
#[derive(Clone, Copy)]
pub(crate) enum FmaPath {
    /// `--select-choice vselect`
    Never,
    /// `--select-choice fma`
    Always,
    /// Chosen by the solver: the literal is true on the multiply-add path
    Chosen(BoolVar),
}

#[cfg(feature = "cpsat")]
impl FmaPath {
    /// 1 on the multiply-add path, 0 otherwise.
    fn latency(self) -> LinearExpr {
        let none = LinearExpr::default();
        match self {
            FmaPath::Never => none,
            FmaPath::Always => none + 1,
            FmaPath::Chosen(fma) => LinearExpr::from(fma),
        }
    }
}

/// Variables of a VALU op that may be shifted to eight scalar ALU ops.
#[cfg(feature = "cpsat")]
struct ShiftTimes {
    shift: BoolVar,
    lanes: Vec<IntVar>,
    completion: IntVar,
}

/// Timing of a SelectChoice.
#[cfg(feature = "cpsat")]
struct SelectTimes {
    path: FmaPath,
    /// Cycle of the multiply_add (sub at the issue cycle, multiply_add one later) for a select
    /// without a delta operand; `None` for a delta select or `--select-choice vselect`.
    fma_cycle: Option<IntVar>,
    /// The result is visible one cycle later on the multiply-add path (the op has a sub_temp).
    delayed_result: bool,
}

/// Times of one op of the CP-SAT model.
#[cfg(feature = "cpsat")]
pub(crate) struct OpTimes {
    issue: IntVar,
    shift: Option<ShiftTimes>,
    select: Option<SelectTimes>,
    /// For an eliminable vbroadcast: true when it is eliminated (its shifted readers read the
    /// scalar producer instead).
    eliminate: Option<BoolVar>,
}

/// The shared subtract of a CSE group (an op that is not in the SSA list).
#[cfg(feature = "cpsat")]
pub(crate) struct CseSubTimes {
    active: BoolVar,
    cycle: IntVar,
}

#[cfg(feature = "cpsat")]
impl CseSubTimes {
    /// True when at least one consumer takes the multiply-add path, so the subtract runs.
    pub(crate) fn active(&self) -> BoolVar {
        self.active
    }
    /// Cycle of the subtract.
    pub(crate) fn cycle(&self) -> IntVar {
        self.cycle
    }
    /// First cycle its result can be read.
    pub(crate) fn visible(&self) -> LinearExpr {
        LinearExpr::from(self.cycle) + 1
    }
}

/// Decision variables of a ConstChoice in `--const-choice optimize`.
#[cfg(feature = "cpsat")]
pub(crate) enum ConstChoiceVars {
    /// Load or add_imm: true = add_imm
    AddImm(BoolVar),
    /// One literal per alternative: Load, AddImm, then each ALU derivation
    Encoding(Vec<BoolVar>),
}

/// Every per-op variable of the model, created in one place.
#[cfg(feature = "cpsat")]
pub(crate) struct OpVariables {
    /// Times of each op, indexed like `ops`
    pub(crate) times: Vec<OpTimes>,
    /// ConstChoice decision variables by op index
    pub(crate) const_choice: HashMap<usize, ConstChoiceVars>,
    /// Shared subtract of each CSE group, indexed like the groups
    pub(crate) cse_subs: Vec<CseSubTimes>,
}

/// Domains for the time variables.
#[cfg(feature = "cpsat")]
pub(crate) struct TimeDomains<'a> {
    /// Issue-cycle domain lower bounds (after tightening), by op id
    pub(crate) final_asap: &'a HashMap<usize, i32>,
    /// Issue-cycle domain upper bounds (after tightening), by op id
    pub(crate) final_alap: &'a HashMap<usize, i32>,
    /// Untightened ASAP, by op id (lanes, completion, multiply_add, CSE subtract)
    pub(crate) asap: &'a HashMap<usize, i32>,
    /// Untightened ALAP, by op id
    pub(crate) alap: &'a HashMap<usize, i32>,
    pub(crate) horizon: i64,
}

/// What kinds of op the model contains and how their encodings are decided.
#[cfg(feature = "cpsat")]
pub(crate) struct OpKinds<'a> {
    pub(crate) shiftable_valu: &'a [usize],
    /// (vbroadcast index, scalar producer index, shiftable consumers)
    pub(crate) eliminable_vbs: &'a [(usize, usize, Vec<usize>)],
    pub(crate) const_choice_ops: &'a [usize],
    pub(crate) const_choice_mode: ConstChoiceMode,
    pub(crate) select_choice_ops: &'a [usize],
    pub(crate) select_choice_mode: SelectChoiceMode,
    pub(crate) cse_groups: Option<&'a [CseGroup]>,
}

/// Create every per-op variable of the model: issue cycles, shift literals and lanes,
/// vbroadcast elimination literals, ConstChoice and SelectChoice literals, multiply_add
/// cycles and CSE shared subtracts, in that order.
///
/// Only variables are created here; the constraints that give them meaning are added by the
/// sections of `schedule_ssa_cpsat` (and [`OpTimes::add_shift_invariants`],
/// [`OpTimes::add_fma_cycle_invariant`]). Variables and constraints are separate lists in the
/// model, so creating all variables first leaves both lists in the order they had when
/// each section created its own.
///
/// # Arguments
/// * `model` - Model under construction
/// * `ops` - SSA ops
/// * `domains` - Cycle domains
/// * `kinds` - Shiftable, eliminable, ConstChoice and SelectChoice ops, CSE groups and modes
/// * `verbose` - Log CSE groups whose operands have several producers
///
/// # Returns
/// The variables, with an [`OpTimes`] per op.
#[cfg(feature = "cpsat")]
pub(crate) fn create_op_variables(
    model: &mut CpModelBuilder,
    ops: &[SsaOp],
    domains: &TimeDomains<'_>,
    kinds: &OpKinds<'_>,
    verbose: bool,
) -> OpVariables {
    let horizon = domains.horizon;
    let asap_of = |idx: usize| *domains.asap.get(&ops[idx].id).unwrap_or(&0);
    let alap_of = |idx: usize| *domains.alap.get(&ops[idx].id).unwrap_or(&(horizon as i32 - 1));

    // Issue cycle of every op.
    let issue: Vec<IntVar> = ops
        .iter()
        .enumerate()
        .map(|(idx, op)| {
            let lb = *domains.final_asap.get(&op.id).unwrap_or(&0) as i64;
            let ub = *domains.final_alap.get(&op.id).unwrap_or(&(horizon as i32 - 1)) as i64;
            model.new_int_var_with_name([(lb, ub)], format!("c{}", idx))
        })
        .collect();

    // Shift literal, eight lane cycles and the completion cycle of each shiftable op.
    let mut shift: HashMap<usize, ShiftTimes> = HashMap::new();
    for &idx in kinds.shiftable_valu {
        let shift_var = model.new_bool_var_with_name(format!("shift_{}", idx));
        let op_asap = asap_of(idx) as i64;
        let op_alap = *domains.alap.get(&ops[idx].id).unwrap_or(&((horizon - 1) as i32)) as i64;
        let lanes: Vec<IntVar> = (0..VLEN)
            .map(|i| model.new_int_var_with_name([(op_asap, op_alap)], format!("alu_c{}_{}", idx, i)))
            .collect();
        let completion = model.new_int_var_with_name([(op_asap, op_alap)], format!("completion_c{}", idx));
        shift.insert(idx, ShiftTimes { shift: shift_var, lanes, completion });
    }

    // Elimination literal of each eliminable vbroadcast.
    let mut eliminate: HashMap<usize, BoolVar> = HashMap::new();
    for (vb_idx, _, _) in kinds.eliminable_vbs {
        eliminate.insert(*vb_idx, model.new_bool_var_with_name(format!("eliminate_vb_{}", vb_idx)));
    }

    // ConstChoice literals.
    let mut const_choice: HashMap<usize, ConstChoiceVars> = HashMap::new();
    if kinds.const_choice_mode == ConstChoiceMode::Optimize {
        for &idx in kinds.const_choice_ops {
            let alternatives = ops[idx].const_alternatives.as_deref();
            let has_alu_options = alternatives.map(|a| a.len() > 2).unwrap_or(false);
            let vars = if !has_alu_options {
                ConstChoiceVars::AddImm(model.new_bool_var_with_name(format!("const_choice_{}", idx)))
            } else {
                let n = alternatives.unwrap().len();
                ConstChoiceVars::Encoding(
                    (0..n)
                        .map(|i| model.new_bool_var_with_name(format!("const_choice_enc_{}_{}", idx, i)))
                        .collect(),
                )
            };
            const_choice.insert(idx, vars);
        }
    }

    // SelectChoice literal and multiply_add cycle.
    let mut select: HashMap<usize, SelectTimes> = HashMap::new();
    for &idx in kinds.select_choice_ops {
        let path = match kinds.select_choice_mode {
            SelectChoiceMode::ForceVselect => FmaPath::Never,
            SelectChoiceMode::ForceFma => FmaPath::Always,
            SelectChoiceMode::Optimize => {
                FmaPath::Chosen(model.new_bool_var_with_name(format!("select_choice_{}", idx)))
            }
        };
        let fma_cycle = if matches!(path, FmaPath::Never) || ops[idx].delta_vreg.is_some() {
            None
        } else {
            let fma_asap = asap_of(idx) + 1;
            let fma_alap = alap_of(idx) + 1;
            Some(model.new_int_var_with_name(
                [(fma_asap as i64, fma_alap as i64)],
                format!("fma_cycle_{}", idx),
            ))
        };
        let delayed_result = ops[idx].sub_temp_vreg.is_some();
        select.insert(idx, SelectTimes { path, fma_cycle, delayed_result });
    }

    // Shared subtract of each CSE group.
    let mut cse_subs: Vec<CseSubTimes> = Vec::new();
    if let Some(groups) = kinds.cse_groups {
        for (g, group) in groups.iter().enumerate() {
            let active = model.new_bool_var_with_name(format!("cse_sub_active_{}", g));
            // sub_asap = max(asap[producer_true], asap[producer_false]) + 1 when both producers
            // are unique. For complex_phi (multi-def/phi), use conservative bound 0; do not
            // unwrap producer_true_idx/producer_false_idx.
            let sub_asap = if group.complex_phi {
                if verbose {
                    eprintln!(
                        "  CSE group {}: complex-phi (operands have multiple producers); using conservative sub_asap=0",
                        g
                    );
                }
                0
            } else if let (Some(pt), Some(pf)) = (group.producer_true_idx, group.producer_false_idx) {
                asap_of(pt).max(asap_of(pf)) + 1
            } else {
                0
            };
            let sub_alap = horizon as i32 - 1;
            let cycle = model.new_int_var_with_name(
                [(sub_asap as i64, sub_alap as i64)],
                format!("cse_sub_cycle_{}", g),
            );
            cse_subs.push(CseSubTimes { active, cycle });
        }
    }

    let times = issue
        .into_iter()
        .enumerate()
        .map(|(idx, issue)| OpTimes {
            issue,
            shift: shift.remove(&idx),
            select: select.remove(&idx),
            eliminate: eliminate.remove(&idx),
        })
        .collect();
    OpVariables { times, const_choice, cse_subs }
}

#[cfg(feature = "cpsat")]
impl OpTimes {
    // ------------------------------------------------------------------
    // Variables (for intervals, hints, the search strategy and extraction)
    // ------------------------------------------------------------------

    /// Issue cycle: the cycle of the op's slot when it is not shifted. Every path of a
    /// ConstChoice runs here, and so does the vselect or the sub of a SelectChoice.
    pub(crate) fn issue(&self) -> IntVar {
        self.issue
    }

    /// Shift literal of a shiftable op (true = eight scalar ALU ops).
    pub(crate) fn shift(&self) -> Option<BoolVar> {
        self.shift.as_ref().map(|s| s.shift)
    }

    /// Lane cycles of a shiftable op (the cycles of its scalar ops when shifted); empty
    /// otherwise.
    pub(crate) fn lanes(&self) -> &[IntVar] {
        self.shift.as_ref().map_or(&[], |s| s.lanes.as_slice())
    }

    /// Completion cycle of a shiftable op: at least the issue cycle and every lane.
    pub(crate) fn completion(&self) -> Option<IntVar> {
        self.shift.as_ref().map(|s| s.completion)
    }

    /// Multiply-add path of a SelectChoice.
    pub(crate) fn fma_path(&self) -> Option<FmaPath> {
        self.select.as_ref().map(|s| s.path)
    }

    /// The literal that chooses the multiply-add path (Optimize mode only).
    pub(crate) fn fma_chosen(&self) -> Option<BoolVar> {
        match self.fma_path() {
            Some(FmaPath::Chosen(b)) => Some(b),
            _ => None,
        }
    }

    /// Cycle of the multiply_add of a select without a delta operand.
    pub(crate) fn fma_cycle(&self) -> Option<IntVar> {
        self.select.as_ref().and_then(|s| s.fma_cycle)
    }

    /// Elimination literal of an eliminable vbroadcast.
    pub(crate) fn eliminate(&self) -> Option<BoolVar> {
        self.eliminate
    }

    // ------------------------------------------------------------------
    // Definitions
    // ------------------------------------------------------------------

    /// Last cycle in which the op itself executes, before any multiply_add second cycle:
    /// the completion cycle of a shiftable op (when it is not shifted the completion is only
    /// bounded below by the issue cycle), the issue cycle otherwise.
    pub(crate) fn exec_end(&self) -> LinearExpr {
        LinearExpr::from(self.completion().unwrap_or(self.issue))
    }

    /// Extra cycles before the result can be read: 1 on the multiply-add path of a select
    /// with a sub_temp, 0 otherwise.
    pub(crate) fn result_latency(&self) -> LinearExpr {
        match &self.select {
            Some(s) if s.delayed_result => s.path.latency(),
            _ => LinearExpr::default(),
        }
    }

    /// Extra cycles during which the op still reads its operands: 1 on the multiply-add path
    /// of any select (the multiply_add reads at the issue cycle + 1), 0 otherwise.
    pub(crate) fn read_latency(&self) -> LinearExpr {
        match &self.select {
            Some(s) => s.path.latency(),
            None => LinearExpr::default(),
        }
    }

    /// First cycle the result can be read: exec end + 1 + result latency.
    pub(crate) fn visible(&self) -> LinearExpr {
        self.exec_end() + 1 + self.result_latency()
    }

    /// Cycle in which a memory op reads or writes main memory: its issue cycle. Memory ops
    /// (load, vload, store, vstore) run on the load and store engines, are never shifted and
    /// are never selects, so the issue cycle is their only representation.
    pub(crate) fn memory_access(&self) -> LinearExpr {
        debug_assert!(self.shift.is_none() && self.select.is_none());
        LinearExpr::from(self.issue)
    }

    /// [`Self::visible`] with the result latency given as an equal expression (an auxiliary
    /// variable fixed to the multiply-add literal).
    pub(crate) fn visible_with_latency(&self, latency: LinearExpr) -> LinearExpr {
        self.exec_end() + 1 + latency
    }

    /// First cycle the result of lane `lane` can be read, as the per-lane precedence rows
    /// state it: lane `lane` + 1 when shifted, issue + 1 when not. For a select on the
    /// multiply-add path the extra cycle is not included here; its readers get separate rows
    /// with [`Self::visible`].
    ///
    /// With `pinned` lanes (`CpsatLaneModel::pinned`) an unshifted op's lanes equal its issue
    /// cycle, so a shiftable op gives one unconditional alternative, lane + 1 (empty label).
    pub(crate) fn lane_ready(&self, lane: usize, pinned: bool) -> Vec<Alternative> {
        match &self.shift {
            Some(s) if pinned => {
                vec![Alternative { value: LinearExpr::from(s.lanes[lane]) + 1, when: None, label: "" }]
            }
            Some(s) => vec![
                Alternative { value: LinearExpr::from(s.lanes[lane]) + 1, when: Some(s.shift), label: "on" },
                Alternative { value: LinearExpr::from(self.issue) + 1, when: Some(!s.shift), label: "off" },
            ],
            None => vec![Alternative { value: LinearExpr::from(self.issue) + 1, when: None, label: "valu" }],
        }
    }

    /// Cycle after the op's last read of its operands: exec end + 1 + read latency.
    pub(crate) fn last_read_end(&self) -> LinearExpr {
        self.exec_end() + 1 + self.read_latency()
    }

    /// [`Self::last_read_end`] for one path of a select: + 2 on the multiply-add path, + 1 on
    /// the vselect path.
    pub(crate) fn last_read_end_on_path(&self, fma: bool) -> LinearExpr {
        self.exec_end() + if fma { 2 } else { 1 }
    }

    /// Last cycle the op occupies, one alternative per mode: the completion cycle when shifted,
    /// the multiply_add cycle on the multiply-add path, the issue cycle otherwise.
    pub(crate) fn finish_alternatives(&self) -> Vec<Alternative> {
        let issue = || LinearExpr::from(self.issue);
        if let Some(s) = &self.shift {
            return vec![
                Alternative { value: LinearExpr::from(s.completion), when: Some(s.shift), label: "shifted" },
                Alternative { value: issue(), when: Some(!s.shift), label: "valu" },
            ];
        }
        match &self.select {
            Some(SelectTimes { path: FmaPath::Never, .. }) => {
                vec![Alternative { value: issue(), when: None, label: "select_vselect" }]
            }
            Some(SelectTimes { path, fma_cycle: Some(fma), .. }) => match path {
                FmaPath::Chosen(b) => vec![
                    Alternative { value: LinearExpr::from(*fma), when: Some(*b), label: "select_fma" },
                    Alternative { value: issue(), when: Some(!*b), label: "select_vselect" },
                ],
                _ => vec![Alternative { value: LinearExpr::from(*fma), when: None, label: "select_fma" }],
            },
            _ => vec![Alternative { value: issue(), when: None, label: "c" }],
        }
    }

    // ------------------------------------------------------------------
    // Mode invariants
    // ------------------------------------------------------------------

    /// Constraints that tie a shiftable op's completion cycle to its issue cycle and lanes.
    ///
    /// # Arguments
    /// * `model` - Model under construction
    /// * `idx` - Op index (for constraint names)
    /// * `valu_shift_mode` - With `ForceValu`, the completion equals the issue cycle
    pub(crate) fn add_shift_invariants(
        &self,
        model: &mut CpModelBuilder,
        idx: usize,
        valu_shift_mode: ValuShiftMode,
        lane_model: CpsatLaneModel,
    ) {
        let Some(s) = &self.shift else {
            return;
        };
        let c = model.add_ge(LinearExpr::from(s.completion), LinearExpr::from(self.issue));
        model.set_constraint_name(c, &format!("completion_ge_start_{}", idx));
        for (lane_i, &lane) in s.lanes.iter().enumerate() {
            let c = model.add_ge(s.completion, LinearExpr::from(lane));
            model.set_constraint_name(c, &format!("completion_ge_alu_{}_{}", idx, lane_i));
        }
        if matches!(valu_shift_mode, ValuShiftMode::ForceValu) {
            let c = model.add_eq(LinearExpr::from(s.completion), LinearExpr::from(self.issue));
            model.set_constraint_name(c, &format!("completion_eq_cycle_{}", idx));
        }
        // Lane model (free adds nothing). Sorted: the lanes of one op are interchangeable, so
        // order them. Pinned: an unshifted op's lanes are its issue cycle. Neither loses a
        // schedule (verification/lane_model.py, docs/CPSAT_LANE_MODEL.md).
        if lane_model.sorted() {
            for lane_i in 0..s.lanes.len() - 1 {
                let c = model.add_le(s.lanes[lane_i], s.lanes[lane_i + 1]);
                model.set_constraint_name(c, &format!("lane_order_{}_{}", idx, lane_i));
            }
        }
        if lane_model.pinned() {
            for (lane_i, &lane) in s.lanes.iter().enumerate() {
                let c = model.add_eq(LinearExpr::from(lane), LinearExpr::from(self.issue));
                model.only_enforce_if(c, [!s.shift]);
                model.set_constraint_name(c, &format!("lane_pin_{}_{}", idx, lane_i));
            }
        }
    }

    /// The multiply_add of a select runs one cycle after its issue cycle.
    pub(crate) fn add_fma_cycle_invariant(&self, model: &mut CpModelBuilder, idx: usize) {
        if let Some(fma) = self.fma_cycle() {
            let c = model.add_eq(LinearExpr::from(fma), LinearExpr::from(self.issue) + 1);
            model.set_constraint_name(c, &format!("select_fma_end_{}", idx));
        }
    }

    /// Start of the op's result in scratch: its first write.
    ///
    /// For a shiftable op this is an auxiliary `shifted_dest_start_{idx}` no later than any
    /// lane and equal to the issue cycle when not shifted; otherwise the issue cycle.
    ///
    /// # Returns
    /// The start variable, and the auxiliary when one was created (it needs a hint).
    pub(crate) fn first_write(&self, model: &mut CpModelBuilder, idx: usize, horizon: i64) -> (IntVar, Option<IntVar>) {
        let Some(s) = &self.shift else {
            return (self.issue, None);
        };
        let start_var = model.new_int_var_with_name([(0, horizon)], format!("shifted_dest_start_{}", idx));
        for (lane_i, &lane) in s.lanes.iter().enumerate() {
            let c = model.add_ge(LinearExpr::from(lane), LinearExpr::from(start_var));
            model.set_constraint_name(c, &format!("scratch_shifted_start_ge_alu_{}_{}", idx, lane_i));
        }
        let c = model.add_eq(LinearExpr::from(start_var), LinearExpr::from(self.issue));
        model.only_enforce_if(c, [!s.shift]);
        model.set_constraint_name(c, &format!("scratch_shifted_start_eq_cycle_{}", idx));
        (start_var, Some(start_var))
    }

    // ------------------------------------------------------------------
    // Solution values
    // ------------------------------------------------------------------

    /// Whether the solution puts this select on the multiply-add path.
    pub(crate) fn on_fma_path(&self, response: &CpSolverResponse) -> bool {
        match self.fma_path() {
            Some(FmaPath::Always) => true,
            Some(FmaPath::Chosen(b)) => b.solution_value(response),
            _ => false,
        }
    }

    /// The op's cycle as reported to the caller: the completion cycle when shifted, the issue
    /// cycle otherwise.
    pub(crate) fn reported_cycle(&self, response: &CpSolverResponse) -> i32 {
        match &self.shift {
            Some(s) if s.shift.solution_value(response) => s.completion.solution_value(response) as i32,
            _ => self.issue.solution_value(response) as i32,
        }
    }

    /// Lane cycles in the solution (for a shiftable op).
    pub(crate) fn lane_values(&self, response: &CpSolverResponse) -> Vec<i32> {
        self.lanes().iter().map(|v| v.solution_value(response) as i32).collect()
    }

    /// Last cycle the op occupies in the solution ([`Self::finish_alternatives`] evaluated),
    /// and which alternative that is.
    pub(crate) fn finish_value(&self, response: &CpSolverResponse) -> (i32, &'static str) {
        if let Some(s) = &self.shift {
            return if s.shift.solution_value(response) {
                (s.completion.solution_value(response) as i32, "shifted")
            } else {
                (self.issue.solution_value(response) as i32, "valu")
            };
        }
        let on_fma = self.on_fma_path(response);
        if let (Some(fma), true) = (self.fma_cycle(), on_fma) {
            return (fma.solution_value(response) as i32, "select_fma");
        }
        // A delta select on the multiply-add path is one VALU op at its issue cycle: the
        // cycle is the issue cycle, the label still names the path taken.
        let label = match (self.select.is_some(), on_fma) {
            (true, true) => "select_fma",
            (true, false) => "select_vselect",
            (false, _) => "c",
        };
        (self.issue.solution_value(response) as i32, label)
    }
}

// ============================================================================
// Solution side
// ============================================================================

/// The definitions of the module computed from a solution: the post-solve validators use
/// these, so they check the same quantities the model constrains.
pub(crate) struct ScheduledTimes<'a> {
    ops: &'a [SsaOp],
    cycle_assignments: &'a [i32],
    shifted_ops: &'a HashSet<usize>,
    shifted_alu_cycles: &'a HashMap<usize, Vec<i32>>,
    select_choice_encoding: &'a HashMap<usize, SelectChoiceEncoding>,
}

impl<'a> ScheduledTimes<'a> {
    /// # Arguments
    /// * `ops` - SSA ops
    /// * `cycle_assignments` - Cycle per op index (completion cycle for shifted ops)
    /// * `shifted_ops` - Ids of shifted ops
    /// * `shifted_alu_cycles` - Lane cycles by op id
    /// * `select_choice_encoding` - SelectChoice path by op id
    pub(crate) fn new(
        ops: &'a [SsaOp],
        cycle_assignments: &'a [i32],
        shifted_ops: &'a HashSet<usize>,
        shifted_alu_cycles: &'a HashMap<usize, Vec<i32>>,
        select_choice_encoding: &'a HashMap<usize, SelectChoiceEncoding>,
    ) -> Self {
        Self { ops, cycle_assignments, shifted_ops, shifted_alu_cycles, select_choice_encoding }
    }

    /// Cycle of op `idx` (-1 when missing).
    pub(crate) fn cycle(&self, idx: usize) -> i32 {
        self.cycle_assignments.get(idx).copied().unwrap_or(-1)
    }

    /// Lane cycles of op `idx` when it is shifted.
    pub(crate) fn lanes(&self, idx: usize) -> Option<&'a [i32]> {
        let id = self.ops[idx].id;
        if !self.shifted_ops.contains(&id) {
            return None;
        }
        self.shifted_alu_cycles.get(&id).map(|v| v.as_slice())
    }

    /// First write of the result: earliest lane when shifted, else the cycle.
    pub(crate) fn first_write(&self, idx: usize) -> i32 {
        self.lanes(idx).and_then(|l| l.iter().min().copied()).unwrap_or(self.cycle(idx))
    }

    /// Last cycle the op executes: latest lane when shifted, else the cycle.
    pub(crate) fn exec_end(&self, idx: usize) -> i32 {
        self.lanes(idx).and_then(|l| l.iter().max().copied()).unwrap_or(self.cycle(idx))
    }

    /// Whether op `idx` is a select on the multiply-add path.
    pub(crate) fn on_fma_path(&self, idx: usize) -> bool {
        let op = &self.ops[idx];
        op.opcode == Opcode::SelectChoice
            && matches!(self.select_choice_encoding.get(&op.id), Some(SelectChoiceEncoding::Fma))
    }

    /// 1 for a select with a sub_temp on the multiply-add path, else 0.
    pub(crate) fn result_latency(&self, idx: usize) -> i32 {
        i32::from(self.on_fma_path(idx) && self.ops[idx].sub_temp_vreg.is_some())
    }

    /// First cycle the result can be read.
    pub(crate) fn visible(&self, idx: usize) -> i32 {
        self.exec_end(idx) + 1 + self.result_latency(idx)
    }

    /// First cycle lane `lane` of the result can be read: that lane + 1 when shifted, else
    /// [`Self::visible`].
    pub(crate) fn lane_visible(&self, idx: usize, lane: usize) -> i32 {
        self.lanes(idx).and_then(|l| l.get(lane)).map_or(self.visible(idx), |&c| c + 1)
    }

    /// Cycle after the last read of the op's operands.
    pub(crate) fn last_read_end(&self, idx: usize) -> i32 {
        self.exec_end(idx) + 1 + i32::from(self.on_fma_path(idx))
    }

    /// Cycle in which memory op `idx` reads or writes main memory (its cycle).
    pub(crate) fn memory_access(&self, idx: usize) -> i32 {
        self.cycle(idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Engine, Operand, VReg};

    fn op(id: usize, engine: Engine, opcode: Opcode) -> SsaOp {
        SsaOp::new(id, engine, opcode, Some(VReg::vector(id as u32)), vec![Operand::VReg(VReg::vector(100))])
    }

    /// The solution-side definitions on the three kinds of op whose timing differs: a shifted
    /// op (lanes), a select with a sub_temp on the multiply-add path (result one cycle
    /// later, reads one cycle later) and a delta select on that path (reads later only).
    #[test]
    fn test_scheduled_times_definitions() {
        let mut fma_select = op(1, Engine::Valu, Opcode::SelectChoice);
        fma_select.sub_temp_vreg = Some(VReg::vector(200));
        let delta_select = op(2, Engine::Valu, Opcode::SelectChoice);
        let ops = vec![op(0, Engine::Valu, Opcode::Add), fma_select, delta_select, op(3, Engine::Valu, Opcode::Xor)];
        let cycles = vec![9, 4, 6, 3];
        let shifted: HashSet<usize> = [0].into_iter().collect();
        let lanes: HashMap<usize, Vec<i32>> = [(0, vec![5, 5, 6, 6, 7, 7, 8, 9])].into_iter().collect();
        let encoding: HashMap<usize, SelectChoiceEncoding> =
            [(1, SelectChoiceEncoding::Fma), (2, SelectChoiceEncoding::Fma)].into_iter().collect();
        let t = ScheduledTimes::new(&ops, &cycles, &shifted, &lanes, &encoding);

        // Shifted: first write is the earliest lane, results lane by lane.
        assert_eq!((t.first_write(0), t.exec_end(0), t.visible(0)), (5, 9, 10));
        assert_eq!((t.lane_visible(0, 0), t.lane_visible(0, 7)), (6, 10));
        assert_eq!(t.last_read_end(0), 10);
        // FMA select with sub_temp: sub at 4, multiply_add at 5, result readable at 6.
        assert_eq!((t.result_latency(1), t.visible(1), t.lane_visible(1, 3)), (1, 6, 6));
        assert_eq!(t.last_read_end(1), 6);
        // Delta select on the FMA path: one VALU op, result at cycle + 1, read latency kept.
        assert_eq!((t.result_latency(2), t.visible(2), t.last_read_end(2)), (0, 7, 8));
        // Plain op.
        assert_eq!((t.first_write(3), t.visible(3), t.lane_visible(3, 2), t.last_read_end(3)), (3, 4, 4, 4));
    }
}
