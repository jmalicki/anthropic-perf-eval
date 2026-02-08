//! SSA-based TickTock Kernel Builder.
//!
//! This builder generates optimized instruction sequences using SSA form,
//! eliminating false WAW/WAR dependencies for better scheduling.
//!
//! ## Flow
//!
//! 1. `build()` - Emit SsaOps using VRegs (no concrete addresses)
//! 2. `compute_ssa_dependencies()` - Only RAW dependencies (true data flow)
//! 3. `schedule_ssa_greedy()` - Schedule with VALU→ALU shifting
//! 4. `assign_addresses()` - Linear scan allocation based on live ranges
//! 5. `lower_ssa_ops()` - Convert to concrete Ops (expanding shifted VALU ops)

use std::collections::{HashMap, HashSet};

use crate::address_assign::AddressAssignmentResult;
use crate::types::{Engine, MachineConfig, Op, SsaOp, UsageAccounting, VReg, VarType};
use crate::types::SlotVal;

use super::config::{fma_multiplier, KernelConfig};
use super::context_ssa::SsaScratchContext;
use super::emit_ssa::SsaEmitContext;
use super::ssa_program::SsaProgram;

/// Unroll factor - process 8 groups per iteration
pub(crate) const UNROLL: usize = 8;

/// Depths where TICK stores happen (first half of groups)
pub(crate) const TICK_STORE_DEPTHS: [usize; 6] = [0, 1, 2, 4, 5, 6];

/// Depths where TOCK stores happen (second half of groups)
pub(crate) const TOCK_STORE_DEPTHS: [usize; 4] = [3, 7, 8, 9];

/// Hardware scratch size in words (addresses 0..SCRATCH_BUDGET_HW-1).
pub(crate) const SCRATCH_BUDGET_HW: usize = 1536;

/// Minimum time (seconds) for per-section feasibility CP-SAT. We require a feasible schedule
/// for each section that exceeds scratch budget; give the solver enough time to find one.
const PER_SECTION_FEASIBILITY_MIN_SECS: f64 = 600.0;
/// Cap per-section feasibility time so one section cannot consume the whole run.
const PER_SECTION_FEASIBILITY_MAX_SECS: f64 = 3600.0;

/// When address 0 is reserved for zero, only 1535 words are assignable; use this as the scheduler
/// scratch budget so we never schedule more than 1535 assignable words (peak_live then includes zero = 1536).
pub(crate) fn effective_scratch_budget(zero_vreg_reserved: bool) -> usize {
    if zero_vreg_reserved {
        SCRATCH_BUDGET_HW - 1
    } else {
        SCRATCH_BUDGET_HW
    }
}

/// Result of SSA scheduling and address assignment.
#[derive(Debug)]
pub struct SsaScheduleResult {
    /// Scheduled concrete ops
    pub ops: Vec<Op>,
    /// Number of cycles
    pub cycles: usize,
    /// Address assignment result
    pub address_result: AddressAssignmentResult,
    /// Number of VRegs used
    pub vreg_count: u32,
    /// Number of VALU ops shifted to ALU
    pub valu_shifted: usize,
    /// True if scratch budget was exceeded (force-scheduling or address assignment)
    pub budget_exceeded: bool,
    /// Usage accounting: breakdown by round and var_type
    pub usage_accounting: UsageAccounting,
    /// VReg metadata for liveness debugging: (var_type, phase, round)
    pub vreg_metadata: HashMap<VReg, (Option<VarType>, Option<String>, Option<usize>)>,
    /// Individual ALU cycle assignments for shifted VALU ops (from CP-SAT)
    /// Maps op_id -> [cycle0, cycle1, ..., cycle7] (8 cycles, one per ALU op)
    pub shifted_alu_cycles: HashMap<usize, Vec<i32>>,
    /// Empty cycle count from CP-SAT (None for greedy path). When > 0, makespan has gaps.
    pub empty_cycle_count: Option<usize>,
    /// When true, makespan was driven by capacity lower bound (structural). None for greedy.
    pub makespan_driven_by_capacity: Option<bool>,
    /// Same constraints as CP-SAT model (for bulk-move simulator). None for greedy path.
    pub schedule_constraints: Option<Vec<crate::schedule_constraints::ScheduleConstraint>>,
    /// SSA-level cycle for each op (same order/length as schedule_constraints index space). For bulk-move from JSON.
    pub ssa_cycle_assignments: Option<Vec<i32>>,
    /// SSA-level (id, dep_ids) per op in same order. For bulk-move from JSON.
    pub ssa_ops_bulk_move: Option<Vec<(usize, Vec<usize>)>>,
    /// Op IDs of vbroadcasts eliminated by CP-SAT (not emitted; consumers use scalar producer). Empty for greedy.
    pub eliminated_vbroadcast_ops: Vec<usize>,
    /// Proven lower bound on cycles (CP-SAT only). When Optimal, equals cycles; when Feasible, solver best_bound in cycles.
    pub lb_cycles: Option<usize>,
}

/// SSA-based TickTock Kernel Builder.
///
/// Unlike TickTockKernelBuilder which uses pre-allocated addresses,
/// this builder emits SsaOps with VRegs that are assigned addresses
/// after scheduling.
#[derive(Debug)]
pub struct SsaTickTockKernelBuilder {
    /// Kernel configuration
    pub config: KernelConfig,
    /// SSA scratch context (holds VRegs instead of addresses)
    pub ctx: SsaScratchContext,
    /// SSA emission context
    pub emit: SsaEmitContext,
    /// Whether depth 3 broadcasts have been initialized
    pub(crate) depth3_broadcasts_initialized: bool,
    /// Whether any round will use depth 3
    pub(crate) uses_depth3: bool,
    /// VReg representing constant 0 (loaded at address 0)
    pub(crate) zero_vreg: Option<VReg>,
}

/// Scratch write size in words for an op (1 for scalar, 8 for vector). Store writes to mem, not scratch.
fn scratch_write_size(op: &Op) -> usize {
    match op.engine {
        Engine::Valu => 8,
        Engine::Alu => 1,
        Engine::Flow => 1,
        Engine::Load => {
            let first = op.slot.first();
            match first {
                Some(SlotVal::Str(s)) if s == "vload" => 8,
                _ => 1,
            }
        }
        _ => 0,
    }
}

/// Compute contiguous (base, len) ranges from a set of addresses.
/// Used so we only treat an op as having one scratch write when it's exactly one contiguous block.
fn contiguous_ranges_from_addresses(addrs: &HashSet<i64>) -> Vec<(i64, usize)> {
    if addrs.is_empty() {
        return vec![];
    }
    let mut sorted: Vec<i64> = addrs.iter().copied().collect();
    sorted.sort_unstable();
    let mut ranges = Vec::new();
    let mut base = sorted[0];
    let mut len = 1usize;
    for &a in &sorted[1..] {
        if a == base + len as i64 {
            len += 1;
        } else {
            ranges.push((base, len));
            base = a;
            len = 1;
        }
    }
    ranges.push((base, len));
    ranges
}

/// Panics if two distinct ops in the same cycle write overlapping scratch ranges.
/// Each op's scratch writes are represented as one or more contiguous (base, size) blocks;
/// no two such blocks (from any ops in the same cycle) may overlap.
fn verify_no_same_cycle_scratch_write_conflict(ops: &[Op]) {
    use std::collections::BTreeMap;
    // Store (base, len, op_id) to identify which ops conflict
    let by_cycle: BTreeMap<i32, Vec<(i64, usize, usize)>> = {
        let mut m: BTreeMap<i32, Vec<(i64, usize, usize)>> = BTreeMap::new();
        for op in ops {
            if op.cycle < 0 {
                continue;
            }
            if scratch_write_size(op) == 0 {
                continue;
            }
            if op.writes.is_empty() {
                continue;
            }
            let ranges = contiguous_ranges_from_addresses(&op.writes);
            for (base, len) in ranges {
                m.entry(op.cycle).or_default().push((base, len, op.id));
            }
        }
        m
    };
    for (_cycle, ranges) in &by_cycle {
        for (i, &(a1, l1, id1)) in ranges.iter().enumerate() {
            for &(a2, l2, id2) in ranges.iter().skip(i + 1) {
                let overlap = a1 < a2 + (l2 as i64) && a2 < a1 + (l1 as i64);
                if overlap {
                    // Find the actual ops for better error message
                    let op1 = ops.iter().find(|o| o.id == id1);
                    let op2 = ops.iter().find(|o| o.id == id2);
                    let op1_info = op1.map(|o| format!("{:?} engine={:?}", o.slot.first(), o.engine)).unwrap_or_else(|| "unknown".to_string());
                    let op2_info = op2.map(|o| format!("{:?} engine={:?}", o.slot.first(), o.engine)).unwrap_or_else(|| "unknown".to_string());
                    panic!(
                        "BUG: same-cycle scratch write conflict at cycle {}: op id={} ({}) writes [{}, {}) len {} vs op id={} ({}) writes [{}, {}) len {} (allocator must not assign overlapping scratch to ops in the same cycle)",
                        _cycle,
                        id1, op1_info,
                        a1, a1 + l1 as i64, l1,
                        id2, op2_info,
                        a2, a2 + l2 as i64, l2
                    );
                }
            }
        }
    }
}

impl SsaTickTockKernelBuilder {
    /// Create a new SSA kernel builder with the given configuration.
    pub fn new(config: KernelConfig) -> Self {
        let n_groups = config.n_groups();
        let n_iters = config.n_iters();
        let tree_height = config.spec.forest_height;
        let uses_depth3 = (0..config.spec.rounds).any(|r| {
            let tree_depth = r % tree_height;
            tree_depth == 3
        });

        // Pre-compute FMA multipliers in config-compatible format
        let mut hash_fma_mult = [None; 6];
        for i in 0..6 {
            hash_fma_mult[i] = fma_multiplier(i);
        }

        let mut ctx = SsaScratchContext::new(n_groups, n_iters);
        ctx.hash_fma_mult = hash_fma_mult;

        SsaTickTockKernelBuilder {
            config,
            ctx,
            emit: SsaEmitContext::new(),
            depth3_broadcasts_initialized: false,
            uses_depth3,
            zero_vreg: None,
        }
    }

    /// Build the kernel, returning an `SsaProgram` with all emission artifacts.
    pub fn build(&mut self) -> SsaProgram {
        self.build_internal();
        SsaProgram {
            vreg_def: std::mem::take(&mut self.emit.vreg_def),
            const_alternatives: self.emit.take_const_alternatives(),
            vreg_count: self.emit.next_vreg_id,
            ops: self.emit.take_ops(),
            zero_vreg: self.zero_vreg,
        }
    }

    /// Build the kernel, returning just the ops (convenience for tests).
    pub fn build_ops(&mut self) -> Vec<SsaOp> {
        self.build().ops
    }

    /// Build and schedule the kernel, returning the full result.
    ///
    /// This performs:
    /// 1. SSA op emission
    /// 2. SSA dependency computation (RAW only)
    /// 3. Greedy scheduling with VALU→ALU shifting
    /// 4. Address assignment via linear scan
    /// 5. Lowering to concrete ops (expanding shifted VALU ops)
    ///
    /// Returns `Err` if scratch budget was exceeded (peak_live > budget); the caller must not use the kernel in that case.
    pub fn build_and_schedule(&mut self) -> Result<SsaScheduleResult, String> {
        self.build_and_schedule_with_options(self.config.compiler.enable_valu_shift, None)
    }

    /// Build and schedule with explicit VALU shift option.
    ///
    /// `scratch_budget_override`: If `Some(b)`, use `b` as scratch budget for the greedy
    /// scheduler (for tests that need to trigger force-schedule / budget_exceeded).
    /// If `None`, use the hardware limit 1536 (peak_live includes zero_vreg when reserved).
    ///
    /// Returns `Err` if scratch budget was exceeded (peak_live > budget); the caller must not use the kernel in that case.
    pub fn build_and_schedule_with_options(
        &mut self,
        enable_valu_shift: bool,
        scratch_budget_override: Option<usize>,
    ) -> Result<SsaScheduleResult, String> {
        use super::pipeline::{PipelineConfig, PipelineContext};

        let program = self.build();

        let config = PipelineConfig {
            compiler: super::config::CompilerConfig {
                enable_valu_shift,
                ..self.config.compiler.clone()
            },
            scratch_budget: scratch_budget_override
                .unwrap_or_else(|| effective_scratch_budget(program.zero_vreg.is_some())),
            verbose: false, // greedy path doesn't need pipeline-level verbosity
            ..Default::default()
        };

        let mut ctx = PipelineContext::new(program, config);
        ctx.run_greedy_pipeline(self.config.compiler.enable_ssa_dce)?;

        // Check budget: address assignment is the ground truth for peak liveness
        // (it accounts for sub_temp_vregs, shifted VALU multi-cycle writes, and
        // zero_vreg permanent reservation that the greedy scheduler doesn't track).
        let sched_exceeded = ctx.greedy_metrics.budget_exceeded;
        let addr_exceeded = ctx
            .address_result
            .as_ref()
            .map(|ar| ar.budget_exceeded)
            .unwrap_or(false);

        if sched_exceeded || addr_exceeded {
            let peak = ctx
                .address_result
                .as_ref()
                .map(|ar| ar.peak_live)
                .unwrap_or(0);
            let budget = ctx
                .address_result
                .as_ref()
                .map(|ar| ar.budget)
                .unwrap_or(SCRATCH_BUDGET_HW);
            return Err(format!(
                "scratch budget exceeded: peak_live={} > budget={}",
                peak, budget
            ));
        }

        ctx.into_result()
    }

    /// Build and schedule using CP-SAT on SSA ops (before register allocation).
    ///
    /// Uses `PipelineContext` for the early passes (deps, DCE, bounds, greedy)
    /// and the final passes (filter, assign, lower). Per-section feasibility and
    /// the full-kernel CP-SAT call remain as direct orchestration logic because
    /// they involve conditional looping and complex hint management.
    ///
    /// Returns `Err` if scratch budget was exceeded or CP-SAT failed.
    pub fn build_and_schedule_cpsat(
        &mut self,
        time_limit_secs: f64,
        verbose: bool,
        enable_valu_shift: bool,
        enable_vbroadcast_elimination: bool,
        per_section_optimize_secs: Option<f64>,
        forced_shifts: Option<HashSet<usize>>,
        const_choice_mode: crate::cpsat_scheduler::ConstChoiceMode,
        select_choice_mode: crate::cpsat_scheduler::SelectChoiceMode,
        valu_shift_mode: crate::cpsat_scheduler::ValuShiftMode,
        force_zero_gap: bool,
        enable_asap_alap_tightening: bool,
        enable_scratch_tightening: bool,
    ) -> Result<SsaScheduleResult, String> {
        use super::barrier_elision::{count_barriers, section_boundaries};
        use super::round_budget::{
            live_in_for_section_filtered, live_out_from_section, live_through_for_section,
            sections_exceeding_budget, vreg_def_section,
        };
        use super::pipeline::{
            CpSatExtra, CpSatPipelineConfig, PipelineConfig, PipelineContext,
            SchedulingDecisions,
            passes::{ComputeBoundsPass, ComputeDependenciesPass, DcePass,
                     ElideBarriersPass, FilterDeadStoresPass, AssignAddressesPass,
                     LowerToOpsPass},
        };
        use crate::cpsat_scheduler::{schedule_ssa_cpsat, SsaCpsatConfig, SsaCpsatHints};
        use crate::types::Opcode;

        // Build SSA program and create pipeline context
        let program = self.build();
        let config = PipelineConfig {
            compiler: super::config::CompilerConfig {
                enable_valu_shift,
                ..self.config.compiler.clone()
            },
            scratch_budget: SCRATCH_BUDGET_HW,
            verbose,
            cpsat: Some(CpSatPipelineConfig {
                time_limit_secs,
                verbose,
                enable_vbroadcast_elimination,
                per_section_optimize_secs,
                forced_shifts: forced_shifts.clone(),
                const_choice_mode,
                select_choice_mode,
                valu_shift_mode,
                force_zero_gap,
                enable_asap_alap_tightening,
                enable_scratch_tightening,
            }),
            ..Default::default()
        };
        let mut ctx = PipelineContext::new(program, config);

        // Phase 1: Dependencies + DCE + Bounds + Greedy (via pipeline passes)
        ctx.run_pass(&ComputeDependenciesPass)?;
        if self.config.compiler.enable_ssa_dce {
            ctx.run_pass(&DcePass)?;
            ctx.run_pass(&ComputeDependenciesPass)?;
        }

        // DCE assertion: every defined VReg must have a use
        if self.config.compiler.enable_ssa_dce {
            let defined: HashSet<_> = ctx.ops.iter().filter_map(|op| op.dest).collect();
            let used: HashSet<_> = ctx.ops.iter().flat_map(|op| op.reads()).collect();
            let dead: Vec<_> = defined.difference(&used).copied().collect();
            assert!(
                dead.is_empty(),
                "After DCE every defined VReg must have a use (CP-SAT scratch model expects this). Dead: {:?}",
                dead
            );
        }

        let barrier_count = count_barriers(&ctx.ops);

        ctx.run_pass(&ComputeBoundsPass)?;

        // Greedy scheduling for hints
        ctx.run_pass(&super::pipeline::passes::GreedySchedulePass)?;

        let greedy_cycles = ctx.greedy_metrics.cycles;
        let greedy_budget_exceeded = ctx.greedy_metrics.budget_exceeded;
        let greedy_peak_live = ctx.greedy_metrics.peak_live;

        if verbose {
            eprintln!(
                "SSA Greedy hint: {} cycles ({} barriers){}{}",
                greedy_cycles,
                barrier_count,
                if enable_valu_shift {
                    format!(" ({} VALU shifted)", ctx.greedy_decisions.shifted_ops.len())
                } else {
                    String::new()
                },
                if greedy_budget_exceeded {
                    format!(" [BUDGET EXCEEDED: peak={}]", greedy_peak_live)
                } else {
                    String::new()
                }
            );
        }

        // Clone decisions so we can mutate ctx freely in later phases
        let greedy_shifted = ctx.greedy_decisions.shifted_ops.clone();
        let greedy_select_fma_ops = ctx.greedy_decisions.select_fma_ops.clone();
        let greedy_const_add_imm_ops = ctx.greedy_decisions.const_add_imm_ops.clone();

        // ====================================================================
        // Phase 2: Per-section feasibility (stays as orchestration, not a pass)
        // ====================================================================
        let boundaries = section_boundaries(&ctx.ops);
        let effective_budget = effective_scratch_budget(ctx.zero_vreg.is_some());
        let budget_result = sections_exceeding_budget(
            &ctx.ops,
            effective_budget,
            Some(&greedy_select_fma_ops),
            None,
        );
        let n_exceeding: usize = budget_result
            .section_exceeds_budget
            .iter()
            .filter(|&&b| b)
            .count();
        if verbose {
            eprintln!();
            eprintln!(
                "================================================================================"
            );
            eprintln!(
                "  CHECKING EACH ROUND FOR SCRATCH BUDGET (budget = {} words)",
                SCRATCH_BUDGET_HW
            );
            eprintln!(
                "  Sections that exceed budget will get a per-section CP-SAT run (feasibility)."
            );
            eprintln!(
                "================================================================================"
            );
            eprintln!();
            for (k, &peak) in budget_result.section_peaks.iter().enumerate() {
                if budget_result.section_exceeds_budget[k] {
                    eprintln!(
                        "  Section {:2}:  peak {} words  > {}  -->  RUN PER-SECTION CP-SAT (feasibility)",
                        k, peak, effective_budget
                    );
                } else {
                    eprintln!(
                        "  Section {:2}:  peak {} words  <= {}  -->  SKIP (within budget)",
                        k, peak, effective_budget
                    );
                }
            }
            eprintln!();
            if n_exceeding > 0 {
                eprintln!(
                    "  SUMMARY: {} section(s) exceed budget; running per-section CP-SAT for those.",
                    n_exceeding
                );
            } else {
                eprintln!("  SUMMARY: All sections within budget; no per-section CP-SAT.");
            }
            eprintln!();
        }
        let vreg_def_sec = vreg_def_section(&ctx.ops, &boundaries);
        let mut section_offset: i32 = 0;
        let mut section_select_fma_overrides: Vec<Option<HashSet<usize>>> =
            vec![None; boundaries.len()];
        let mut current_select_fma: HashSet<usize> = greedy_select_fma_ops.clone();
        let mut current_const_add_imm: HashSet<usize> = greedy_const_add_imm_ops.clone();

        for k in 0..boundaries.len() {
            let mut this_section_makespan: Option<i32> = None;
            let (start, end) = boundaries[k];
            let section_span_after_rebase: i32 = {
                let mut min_c = i32::MAX;
                let mut max_c = i32::MIN;
                for op in &ctx.ops[start..end] {
                    if op.opcode != Opcode::Barrier && op.cycle >= 0 {
                        min_c = min_c.min(op.cycle);
                        max_c = max_c.max(op.cycle);
                    }
                }
                let span = if min_c <= max_c {
                    max_c - min_c + 1
                } else {
                    0
                };
                if span > 0 {
                    for op in &mut ctx.ops[start..end] {
                        if op.opcode != Opcode::Barrier && op.cycle >= 0 {
                            op.cycle = section_offset + (op.cycle - min_c);
                        }
                    }
                }
                span
            };
            if budget_result.section_exceeds_budget[k] {
                let global_indices: Vec<usize> = (start..end)
                    .filter(|&i| ctx.ops[i].opcode != Opcode::Barrier)
                    .collect();
                if !global_indices.is_empty() {
                let (greedy_min_c, greedy_max_c) = global_indices
                    .iter()
                    .filter_map(|&i| {
                        let c = ctx.ops[i].cycle;
                        if c >= 0 { Some(c) } else { None }
                    })
                    .fold((i32::MAX, i32::MIN), |(lo, hi), c| (lo.min(c), hi.max(c)));
                let greedy_section_makespan = if greedy_min_c <= greedy_max_c {
                    greedy_max_c - greedy_min_c + 1
                } else {
                    0
                };
                let greedy_section_peak = budget_result.section_peaks[k];

                let section_optimize = per_section_optimize_secs.map_or(false, |t| t > 0.0);
                if verbose {
                    eprintln!("--------------------------------------------------------------------------------");
                    eprintln!(
                        "  PER-SECTION CP-SAT:  SECTION {}  ({})",
                        k,
                        if section_optimize {
                            format!(
                                "optimize up to {:.0}s, no VALU shift",
                                per_section_optimize_secs.unwrap_or(0.0)
                            )
                        } else {
                            "feasibility — find any valid schedule, no VALU shift".to_string()
                        }
                    );
                    eprintln!(
                        "  {} ops in this section. Goal: stay within scratch budget.",
                        global_indices.len()
                    );
                    eprintln!("--------------------------------------------------------------------------------");
                    eprintln!();
                }
                let segment_op_ids: HashSet<usize> =
                    global_indices.iter().map(|&i| ctx.ops[i].id).collect();
                let segment_ops_raw: Vec<SsaOp> = global_indices
                    .iter()
                    .map(|&i| {
                        let mut op = ctx.ops[i].clone();
                        op.dep_ids.retain(|id| segment_op_ids.contains(id));
                        op
                    })
                    .collect();

                let mut seg_addimm_alternatives: HashSet<usize> = HashSet::new();
                for (idx, op) in segment_ops_raw.iter().enumerate() {
                    if op.engine == Engine::Flow && op.opcode == Opcode::AddImm {
                        if let Some(dest) = op.dest {
                            for prev_idx in (0..idx).rev() {
                                let prev_op = &segment_ops_raw[prev_idx];
                                if prev_op.dest == Some(dest) {
                                    if prev_op.engine == Engine::Load
                                        && prev_op.opcode == Opcode::Const
                                    {
                                        seg_addimm_alternatives.insert(idx);
                                    }
                                    break;
                                }
                            }
                        }
                    }
                }
                let filtered_segment: Vec<(SsaOp, i32)> = segment_ops_raw
                    .into_iter()
                    .zip(global_indices.iter().map(|&i| {
                        let c = ctx.ops[i].cycle;
                        if c >= 0 { (c - section_offset).max(0) } else { c }
                    }))
                    .enumerate()
                    .filter(|(idx, _)| !seg_addimm_alternatives.contains(idx))
                    .map(|(_, pair)| pair)
                    .collect();
                let (segment_ops, segment_hints): (Vec<SsaOp>, Vec<i32>) =
                    filtered_segment.into_iter().unzip();

                let filtered_global_indices: Vec<usize> = global_indices
                    .iter()
                    .enumerate()
                    .filter(|(idx, _)| !seg_addimm_alternatives.contains(idx))
                    .map(|(_, &g)| g)
                    .collect();

                let initial_live = live_in_for_section_filtered(
                    &ctx.ops, &boundaries, k, &vreg_def_sec, &filtered_global_indices,
                );
                let live_out = live_out_from_section(&ctx.ops, &boundaries, k, &vreg_def_sec);
                let live_through =
                    live_through_for_section(&ctx.ops, &boundaries, k, &vreg_def_sec);
                let (section_time_limit, section_stop_first) = match per_section_optimize_secs {
                    Some(t) if t > 0.0 => (t, false),
                    _ => {
                        let limit = (time_limit_secs * 0.5)
                            .max(PER_SECTION_FEASIBILITY_MIN_SECS)
                            .min(PER_SECTION_FEASIBILITY_MAX_SECS);
                        if verbose {
                            eprintln!(
                                "  Per-section feasibility time limit: {:.0}s (min {}, max {}).",
                                limit, PER_SECTION_FEASIBILITY_MIN_SECS, PER_SECTION_FEASIBILITY_MAX_SECS
                            );
                        }
                        (limit, true)
                    }
                };
                let config_feasibility = SsaCpsatConfig {
                    time_limit_secs: section_time_limit,
                    verbose,
                    stop_after_first_solution: section_stop_first,
                    enable_valu_shift: false,
                    scratch_budget: effective_budget,
                    model_name: Some(format!("section_{}_feasibility", k)),
                    ..Default::default()
                };
                let seg_shift_set: HashSet<usize> = greedy_shifted
                    .intersection(&segment_op_ids)
                    .copied()
                    .collect();
                let seg_shift: Option<&HashSet<usize>> =
                    if seg_shift_set.is_empty() { None } else { Some(&seg_shift_set) };
                let seg_select_set: HashSet<usize> =
                    current_select_fma.intersection(&segment_op_ids).copied().collect();
                let seg_select = if seg_select_set.is_empty() { None } else { Some(&seg_select_set) };
                let seg_const_add_imm: HashSet<usize> = current_const_add_imm
                    .intersection(&segment_op_ids)
                    .copied()
                    .collect();
                let seg_const_add_imm_hints = if seg_const_add_imm.is_empty() {
                    None
                } else {
                    Some(&seg_const_add_imm)
                };
                let seg_result = match schedule_ssa_cpsat(
                    &segment_ops,
                    config_feasibility,
                    SsaCpsatHints {
                        greedy_hints: Some(&segment_hints),
                        shift_hints: seg_shift,
                        forced_shifts: None,
                        select_choice_hints: seg_select,
                        const_add_imm_hints: seg_const_add_imm_hints,
                        eliminate_vbroadcast_hints: None,
                        initial_live: Some(&initial_live),
                        live_out_vregs: if live_out.is_empty() { None } else { Some(&live_out) },
                        live_through_vregs: if live_through.is_empty() { None } else { Some(&live_through) },
                    },
                    Some(false),
                ) {
                    Some(r) => r,
                    None => panic!(
                        "Per-section CP-SAT must find a feasible schedule for section {} (scratch budget {}). \
                         No solution found within {:.0}s. Either the segment is infeasible (bug in model/liveness) \
                         or the time limit was too short; per-section feasibility uses at least {}s.",
                        k, effective_budget, section_time_limit, PER_SECTION_FEASIBILITY_MIN_SECS
                    ),
                };
                section_select_fma_overrides[k] = Some(seg_result.select_fma_ops.clone());
                for &i in &global_indices {
                    current_select_fma.remove(&ctx.ops[i].id);
                }
                for &id in &seg_result.select_fma_ops {
                    current_select_fma.insert(id);
                }
                let (start, end) = boundaries[k];
                for op in &ctx.ops[start..end] {
                    if op.opcode == Opcode::ConstChoice {
                        current_const_add_imm.remove(&op.id);
                    }
                }
                for &id in &seg_result.const_add_imm_ops {
                    current_const_add_imm.insert(id);
                }
                for (seg_idx, &cycle) in seg_result.cycle_assignments.iter().enumerate() {
                    if seg_idx < filtered_global_indices.len() && cycle >= 0 {
                        ctx.ops[filtered_global_indices[seg_idx]].cycle = section_offset + cycle;
                    }
                }
                let filtered_set: HashSet<usize> =
                    filtered_global_indices.iter().copied().collect();
                for &i in &global_indices {
                    if !filtered_set.contains(&i) && ctx.ops[i].cycle >= 0 {
                        ctx.ops[i].cycle =
                            section_offset + (ctx.ops[i].cycle - greedy_min_c);
                    }
                }
                this_section_makespan = Some(
                    (seg_result.makespan as i32).max(greedy_section_makespan),
                );
                if verbose {
                    let after_budget = sections_exceeding_budget(
                        &ctx.ops,
                        effective_scratch_budget(ctx.zero_vreg.is_some()),
                        Some(&greedy_select_fma_ops),
                        Some(&section_select_fma_overrides),
                    );
                    let after_peak = after_budget.section_peaks[k];
                    let delta_cycles = seg_result.makespan as i32 - greedy_section_makespan;
                    let delta_peak = after_peak as i32 - greedy_section_peak as i32;
                    eprintln!();
                    eprintln!(
                        "  --> Section {} done: {} cycles (merged into global schedule)",
                        k, seg_result.makespan
                    );
                    eprintln!(
                        "      Greedy: {} cycles, peak {} words  →  After feasibility: {} cycles, peak {} words  (Δ cycles: {:+}, Δ peak: {:+})",
                        greedy_section_makespan,
                        greedy_section_peak,
                        seg_result.makespan,
                        after_peak,
                        delta_cycles,
                        delta_peak
                    );
                    eprintln!();
                }
                } // end if !global_indices.is_empty()
            }
            let advance = this_section_makespan.unwrap_or(section_span_after_rebase);
            section_offset += advance;
        }

        // Presolve guarantee: merged schedule must be within scratch budget
        let post_merge_budget = sections_exceeding_budget(
            &ctx.ops, SCRATCH_BUDGET_HW,
            Some(&greedy_select_fma_ops),
            Some(&section_select_fma_overrides),
        );
        if let Some((sec, peak)) = post_merge_budget
            .section_peaks.iter().enumerate()
            .find(|(_, &p)| p > SCRATCH_BUDGET_HW)
        {
            return Err(format!(
                "Per-section CP-SAT produced a merged schedule that exceeds scratch budget: section {} has peak {} words (budget {}). \
                 This indicates a bug in the segment scratch model (e.g. live-through or live-out).",
                sec, *peak, SCRATCH_BUDGET_HW
            ));
        }

        let rescheduled_by_section_cpsat: HashSet<usize> = boundaries
            .iter().enumerate()
            .filter(|(k, _)| budget_result.section_exceeds_budget[*k])
            .flat_map(|(_, &(start, end))| ctx.ops[start..end].iter().map(|op| op.id))
            .collect();

        // ====================================================================
        // Phase 3: Elide barriers → Full-kernel CP-SAT → Filter → Assign → Lower
        // ====================================================================
        ctx.run_pass(&ElideBarriersPass)?;

        // Collect hints for full-kernel CP-SAT
        let greedy_hints: Vec<i32> = ctx.ops.iter().map(|op| op.cycle).collect();
        let hint_makespan = greedy_hints.iter().copied().max().map(|c| c + 1).unwrap_or(0);
        let effective_budget_hint = effective_scratch_budget(ctx.zero_vreg.is_some());
        let hint_peak = sections_exceeding_budget(
            &ctx.ops, effective_budget_hint, Some(&current_select_fma), None,
        ).section_peaks.iter().copied().max().unwrap_or(0);

        if hint_peak > SCRATCH_BUDGET_HW {
            return Err(format!(
                "Merged hint schedule exceeds scratch budget: peak {} words (budget {}). \
                 Presolves must guarantee a feasible hint; this indicates a bug in the segment scratch model.",
                hint_peak, SCRATCH_BUDGET_HW
            ));
        }

        // Reset cycles before CP-SAT
        for op in &mut ctx.ops {
            op.cycle = -1;
        }

        // Build full-kernel CP-SAT config
        let cpsat_config = SsaCpsatConfig {
            time_limit_secs,
            verbose,
            stop_after_first_solution: false,
            enable_valu_shift,
            enable_vbroadcast_elimination,
            scratch_budget: effective_scratch_budget(ctx.zero_vreg.is_some()),
            model_name: Some("full_kernel_optimization".to_string()),
            machine: MachineConfig::default(),
            const_choice_mode,
            select_choice_mode,
            valu_shift_mode,
            force_zero_gap,
            enable_asap_alap_tightening,
            enable_scratch_tightening,
            ..Default::default()
        };

        // Shift hints: exclude ops rescheduled by per-section CP-SAT
        let full_kernel_shift_set: HashSet<usize> = if enable_valu_shift {
            greedy_shifted
                .difference(&rescheduled_by_section_cpsat)
                .copied()
                .collect()
        } else {
            HashSet::new()
        };
        let shift_hints = if full_kernel_shift_set.is_empty() { None } else { Some(&full_kernel_shift_set) };

        // SelectChoice hints
        let op_ids_after_elision: HashSet<usize> = ctx.ops.iter().map(|op| op.id).collect();
        let select_choice_hints_set: HashSet<usize> = current_select_fma
            .iter()
            .filter(|id| op_ids_after_elision.contains(id))
            .copied()
            .collect();
        let select_choice_hints = if select_choice_hints_set.is_empty() { None } else { Some(&select_choice_hints_set) };

        if verbose {
            eprintln!();
            eprintln!(
                "================================================================================"
            );
            eprintln!(
                "  FULL KERNEL CP-SAT  (optimization — minimize makespan over entire kernel)"
            );
            eprintln!(
                "  All rounds merged; solving for minimum cycles. Scratch budget = {}.",
                cpsat_config.scratch_budget
            );
            eprintln!(
                "  Before optimization: {} cycles, peak {} words (hint schedule).",
                hint_makespan, hint_peak
            );
            if greedy_budget_exceeded {
                eprintln!(
                    "  (Greedy exceeded budget; hints may be infeasible — CP-SAT will search.)"
                );
            }
            eprintln!(
                "================================================================================"
            );
            eprintln!();
        }

        let forced_shifts_ref = forced_shifts.as_ref();
        let cpsat_result = schedule_ssa_cpsat(
            &ctx.ops,
            cpsat_config,
            SsaCpsatHints {
                greedy_hints: Some(&greedy_hints),
                shift_hints,
                forced_shifts: forced_shifts_ref,
                select_choice_hints,
                const_add_imm_hints: Some(&current_const_add_imm),
                eliminate_vbroadcast_hints: None,
                initial_live: None,
                live_out_vregs: None,
                live_through_vregs: None,
            },
            Some(!greedy_budget_exceeded),
        )
        .ok_or_else(|| "CP-SAT scheduling failed".to_string())?;

        // Apply cycle assignments
        for (idx, op) in ctx.ops.iter_mut().enumerate() {
            if idx < cpsat_result.cycle_assignments.len() {
                op.cycle = cpsat_result.cycle_assignments[idx];
            }
        }

        let cycles = cpsat_result.makespan;

        if std::env::var("SCHEDULER_DEBUG_DECISIONS").is_ok() {
            let mut s: Vec<_> = cpsat_result.shifted_ops.iter().copied().collect();
            s.sort_unstable();
            let mut c: Vec<_> = cpsat_result.const_add_imm_ops.iter().copied().collect();
            c.sort_unstable();
            let mut f: Vec<_> = cpsat_result.select_fma_ops.iter().copied().collect();
            f.sort_unstable();
            eprintln!(
                "[CPSAT_DECISIONS] shifted_ops={:?} const_add_imm_ops={:?} select_fma_ops={:?}",
                s, c, f
            );
        }

        let shifted_ops = &cpsat_result.shifted_ops;

        if verbose {
            eprintln!(
                "SSA CP-SAT result: {} cycles (greedy was {}), {} VALU ops shifted",
                cycles, greedy_cycles, shifted_ops.len()
            );
            let const_total = ctx.ops.iter().filter(|o| o.opcode == Opcode::ConstChoice).count();
            let const_add_imm = cpsat_result.const_add_imm_ops.len();
            let select_total = ctx.ops.iter().filter(|o| o.opcode == Opcode::SelectChoice).count();
            let select_fma = cpsat_result.select_fma_ops.len();
            eprintln!(
                "  ConstChoice: {} AddImm, {} Load | SelectChoice: {} FMA, {} vselect",
                const_add_imm, const_total.saturating_sub(const_add_imm),
                select_fma, select_total.saturating_sub(select_fma)
            );
        }

        // Capture CP-SAT extra results (bulk-move, constraints, etc.)
        let ssa_cycle_assignments = cpsat_result.cycle_assignments.clone();
        let ssa_ops_bulk_move: Vec<(usize, Vec<usize>)> = ctx.ops
            .iter()
            .map(|o| (o.id, o.dep_ids.iter().copied().collect()))
            .collect();

        // Populate final decisions and metrics from CP-SAT result
        ctx.final_decisions = SchedulingDecisions {
            shifted_ops: cpsat_result.shifted_ops.clone(),
            select_fma_ops: cpsat_result.select_fma_ops.clone(),
            const_add_imm_ops: cpsat_result.const_add_imm_ops.clone(),
            const_alu_derive_ops: cpsat_result.const_alu_derive_ops.clone(),
            shifted_alu_cycles: cpsat_result.shifted_alu_cycles.clone(),
            eliminated_vbroadcast_ops: cpsat_result.eliminated_vbroadcast_ops.clone(),
            eliminated_vb_to_scalar_producer: cpsat_result.eliminated_vb_to_scalar_producer.clone(),
            ..Default::default()
        };
        ctx.final_metrics.cycles = cycles;
        ctx.cpsat_extra = Some(CpSatExtra {
            empty_cycle_count: cpsat_result.empty_cycle_count,
            makespan_driven_by_capacity: cpsat_result.makespan_driven_by_capacity,
            schedule_constraints: cpsat_result.schedule_constraints.clone(),
            ssa_cycle_assignments,
            ssa_ops_bulk_move,
            lb_cycles: cpsat_result.lb_cycles,
        });

        // Final passes: Filter → Assign → Lower
        ctx.run_pass(&FilterDeadStoresPass)?;
        ctx.compute_accounting();
        ctx.run_pass(&AssignAddressesPass)?;

        // Budget exceeded check for CP-SAT path
        if let Some(ref ar) = ctx.address_result {
            if ar.budget_exceeded {
                eprintln!(
                    "Warning: CP-SAT address assignment exceeded budget: peak_live={} > budget={}",
                    ar.peak_live, ar.budget
                );
                return Err(format!(
                    "scratch budget exceeded: peak_live={} > budget={}",
                    ar.peak_live, ar.budget
                ));
            }

            // Assert no two VRegs with overlapping live intervals share the same address
            let intervals = &ar.live_intervals;
            for (v1, addr1) in &ar.vreg_to_addr {
                for (v2, addr2) in &ar.vreg_to_addr {
                    if v1 >= v2 || addr1 != addr2 { continue; }
                    if let (Some(i1), Some(i2)) = (intervals.get(v1), intervals.get(v2)) {
                        assert!(
                            !i1.overlaps(i2),
                            "BUG: VRegs {:?} and {:?} share address {} but intervals [{},{}] and [{},{}] overlap",
                            v1, v2, addr1, i1.start, i1.end, i2.start, i2.end
                        );
                    }
                }
            }
        }

        ctx.run_pass(&LowerToOpsPass)?;

        // Debug: assert no two ops in the same cycle write overlapping scratch addresses
        if cfg!(debug_assertions) {
            if let Some(ref ops) = ctx.lowered_ops {
                verify_no_same_cycle_scratch_write_conflict(ops);
            }
        }

        ctx.into_result()
    }

    // Note: Greedy scheduling methods are implemented in ssa_greedy.rs
    // Note: Emission methods are implemented in ssa_emit.rs
}

// ============================================================================
// Debug Helpers
// ============================================================================

/// Verify that all source VRegs in ops have addresses assigned.
/// This helps diagnose bugs where VRegs are read but not defined.
fn verify_vreg_address_consistency(
    ops: &[SsaOp],
    vreg_to_addr: &std::collections::HashMap<VReg, i64>,
) {
    use crate::types::Operand;

    eprintln!(
        "[DEBUG verify_vreg_address_consistency] Called with {} ops",
        ops.len()
    );

    let mut missing_vregs = Vec::new();

    for op in ops {
        for src in &op.srcs {
            if let Operand::VReg(vreg) = src {
                if !vreg_to_addr.contains_key(vreg) {
                    missing_vregs.push((op.id, op.cycle, op.opcode, *vreg));
                }
            }
        }
    }

    if !missing_vregs.is_empty() {
        eprintln!("[DEBUG verify_vreg_address_consistency] ERRORS FOUND:");
        for (op_id, cycle, opcode, vreg) in &missing_vregs {
            eprintln!(
                "  Op id={} cycle={} opcode={:?} reads {:?} but no address assigned!",
                op_id, cycle, opcode, vreg
            );
        }
    } else {
        eprintln!("[DEBUG verify_vreg_address_consistency] All source VRegs have addresses.");
    }

    // Also check for vstores specifically
    let vstore_ops: Vec<_> = ops
        .iter()
        .filter(|op| op.opcode == crate::types::Opcode::Vstore)
        .collect();

    if !vstore_ops.is_empty() {
        eprintln!(
            "[DEBUG] {} vstore ops found. Checking sources:",
            vstore_ops.len()
        );
        for op in vstore_ops {
            eprintln!("  vstore id={} cycle={}", op.id, op.cycle);
            for (i, src) in op.srcs.iter().enumerate() {
                match src {
                    Operand::VReg(vreg) => {
                        let addr = vreg_to_addr.get(vreg);
                        eprintln!("    src[{}]: {:?} -> addr {:?}", i, vreg, addr);
                    }
                    Operand::Imm(val) => {
                        eprintln!("    src[{}]: Imm({})", i, val);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
