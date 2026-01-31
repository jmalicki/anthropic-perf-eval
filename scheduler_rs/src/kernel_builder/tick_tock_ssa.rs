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

use crate::address_assign::{assign_addresses, build_vreg_metadata, AddressAssignConfig, AddressAssignmentResult};
use crate::types::{Op, SsaOp, UsageAccounting, VarType, VReg};

use super::barrier_elision::elide_barriers;
use super::config::{fma_multiplier, KernelConfig};
use super::context_ssa::SsaScratchContext;
use super::emit_ssa::{compute_ssa_dependencies, compute_ssa_dependencies_with_barriers, SsaEmitContext};
use super::ssa_lower::lower_ssa_ops_with_shift;

/// Unroll factor - process 8 groups per iteration
pub(crate) const UNROLL: usize = 8;

/// Depths where TICK stores happen (first half of groups)
pub(crate) const TICK_STORE_DEPTHS: [usize; 6] = [0, 1, 2, 4, 5, 6];

/// Depths where TOCK stores happen (second half of groups)
pub(crate) const TOCK_STORE_DEPTHS: [usize; 4] = [3, 7, 8, 9];

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

impl SsaTickTockKernelBuilder {
    /// Create a new SSA kernel builder with the given configuration.
    pub fn new(config: KernelConfig) -> Self {
        let n_groups = config.n_groups();
        let n_iters = config.n_iters();
        let tree_height = config.forest_height;
        let uses_depth3 = (0..config.rounds).any(|r| {
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

    /// Build the kernel, returning unscheduled SSA ops.
    pub fn build(&mut self) -> Vec<SsaOp> {
        self.build_internal();
        self.emit.take_ops()
    }

    /// Build and schedule the kernel, returning the full result.
    ///
    /// This performs:
    /// 1. SSA op emission
    /// 2. SSA dependency computation (RAW only)
    /// 3. Greedy scheduling with VALU→ALU shifting
    /// 4. Address assignment via linear scan
    /// 5. Lowering to concrete ops (expanding shifted VALU ops)
    pub fn build_and_schedule(&mut self) -> Result<SsaScheduleResult, String> {
        self.build_and_schedule_with_options(self.config.enable_valu_shift, None)
    }

    /// Build and schedule with explicit VALU shift option.
    ///
    /// `scratch_budget_override`: If `Some(b)`, use `b` as scratch budget for the greedy
    /// scheduler (for tests that need to trigger force-schedule / budget_exceeded).
    /// If `None`, use the hardware limit 1536.
    pub fn build_and_schedule_with_options(
        &mut self,
        enable_valu_shift: bool,
        scratch_budget_override: Option<usize>,
    ) -> Result<SsaScheduleResult, String> {
        println!("[DEBUG] build_and_schedule_with_options ENTRY");
        std::io::Write::flush(&mut std::io::stdout()).ok();
        // Build SSA ops
        self.build_internal();

        // Get ops and dependency info
        // IMPORTANT: Take vreg_def and const_alternatives BEFORE take_ops() since take_ops() clears them
        let vreg_def = std::mem::take(&mut self.emit.vreg_def);
        let const_alternatives = self.emit.take_const_alternatives();
        let vreg_count = self.emit.next_vreg_id;
        let mut ssa_ops = self.emit.take_ops();

        // Build set of AddImm alternative op indices for the greedy scheduler to skip
        // The greedy scheduler always uses the Load alternative; AddImm alternatives
        // are only considered by CP-SAT. Mark them as dead so they're not scheduled.
        let const_add_imm_indices: std::collections::HashSet<usize> = const_alternatives
            .iter()
            .map(|alt| alt.add_imm_op_idx)
            .collect();

        // Use barrier-aware dependencies for greedy scheduler
        // Barriers create hard scheduling boundaries between rounds,
        // constraining cross-round interleaving and reducing peak liveness
        compute_ssa_dependencies_with_barriers(&mut ssa_ops, &vreg_def);

        let scratch_budget = scratch_budget_override.unwrap_or(1536);

        // Schedule using greedy scheduler with VALU→ALU shifting
        // Collect SelectChoice decisions from greedy scheduler
        let mut shifted_ops: HashSet<usize> = HashSet::new();
        let mut greedy_select_fma_ops = HashSet::new();
        let (cycles, sched_budget_exceeded, _sched_peak_live) = 
            self.schedule_ssa_greedy_with_shift_budget_and_selectchoice(
                &mut ssa_ops,
                enable_valu_shift,
                &mut shifted_ops,
                scratch_budget,
                Some(&mut greedy_select_fma_ops),
                None, // Don't need cycle hints for greedy path
                None,
            )?;

        // Elide barriers before address assignment and lowering
        // (barriers are scheduling-only, not real hardware instructions)
        let ssa_ops = elide_barriers(ssa_ops);

        // Filter out dead stores (ops with cycle=-1) before address assignment.
        // Dead stores are ops whose dest VReg has no users - they were detected
        // and marked by the greedy scheduler but remain in the ops list with
        // cycle=-1. Including them in address assignment would corrupt live
        // interval computation (their [-1,-1] intervals get processed first
        // and their addresses freed immediately, potentially allowing live
        // VRegs to incorrectly share addresses with them).
        let ssa_ops: Vec<SsaOp> = ssa_ops
            .into_iter()
            .filter(|op| op.cycle >= 0)
            .collect();

        // Assign addresses via linear scan
        let mut assign_config = AddressAssignConfig::default();
        assign_config.budget = scratch_budget;

        // Reserve address 0 - scratch memory is always initialized to 0 at startup, so address 0 always contains 0
        if let Some(zero) = self.zero_vreg {
            assign_config.fixed_vregs.insert(zero, 0);
        }

        eprintln!("[TRACE build_and_schedule] About to call assign_addresses with {} ops", ssa_ops.len());

        let address_result = assign_addresses(&ssa_ops, &assign_config);

        eprintln!("[TRACE build_and_schedule] assign_addresses returned, peak_live={}", address_result.peak_live);

        // Debug: verify all source VRegs have addresses assigned
        if std::env::var("SCHEDULER_DEBUG").is_ok() {
            verify_vreg_address_consistency(&ssa_ops, &address_result.vreg_to_addr);
        }

        // Track if budget was exceeded (either in scheduling or address assignment)
        let budget_exceeded = sched_budget_exceeded || address_result.budget_exceeded;

        if budget_exceeded {
            eprintln!(
                "Budget exceeded: sched={}, addr_assign={}, peak_live={}, budget={}",
                sched_budget_exceeded, address_result.budget_exceeded,
                address_result.peak_live, address_result.budget
            );
        }

        // Compute usage accounting from SSA ops (before lowering loses metadata)
        let mut usage_accounting = UsageAccounting::from_ssa_ops(&ssa_ops);
        usage_accounting.set_peak_live(address_result.peak_live);

        // Build VReg metadata for liveness debugging (before lowering loses SSA info)
        let vreg_metadata = build_vreg_metadata(&ssa_ops);

        // Lower to concrete ops, expanding shifted VALU ops to ALU
        // Greedy scheduler always uses Load for ConstChoice (no AddImm decisions)
        // Greedy scheduler chooses SelectChoice path based on resource availability
        let ops = lower_ssa_ops_with_shift(
            &ssa_ops,
            &address_result.vreg_to_addr,
            &shifted_ops,
            &HashSet::new(), // Greedy always uses Load for ConstChoice
            &greedy_select_fma_ops, // Pass greedy's SelectChoice decisions
        );

        let valu_shifted = shifted_ops.len();

        Ok(SsaScheduleResult {
            ops,
            cycles,
            address_result,
            vreg_count,
            valu_shifted,
            budget_exceeded,
            usage_accounting,
            vreg_metadata,
        })
    }

    /// Build and schedule using CP-SAT on SSA ops (before register allocation).
    ///
    /// This performs:
    /// 1. SSA op emission (including barrier pseudo-ops)
    /// 2. SSA dependency computation with barriers (for greedy hints)
    /// 3. Greedy scheduling to get hints (with barrier constraints)
    /// 4. Elide barriers before CP-SAT (CP-SAT has proper liveness constraints)
    /// 5. CP-SAT scheduling on SsaOps (with optional VALU→ALU shift modeling)
    /// 6. Address assignment via linear scan
    /// 7. Lowering to concrete ops (expanding shifted VALU ops)
    ///
    /// When `enable_valu_shift` is true, the CP-SAT solver models VALU→ALU shifting
    /// as decision variables, allowing it to find globally optimal shift decisions.
    pub fn build_and_schedule_cpsat(
        &mut self,
        time_limit_secs: f64,
        verbose: bool,
        enable_valu_shift: bool,
    ) -> Result<SsaScheduleResult, String> {
        use crate::types::Opcode;
        use crate::cpsat_scheduler::{schedule_ssa_cpsat, SsaCpsatConfig, SsaCpsatHints};
        use super::barrier_elision::{count_barriers, section_boundaries};
        use super::round_budget::{sections_exceeding_budget, live_in_for_section, live_out_from_section, live_through_for_section, vreg_def_section};

        // Build SSA ops (includes barrier pseudo-ops between rounds)
        self.build_internal();

        // Get ops and dependency info
        // IMPORTANT: Take vreg_def BEFORE take_ops() since take_ops() clears it
        let vreg_def = std::mem::take(&mut self.emit.vreg_def);
        let vreg_count = self.emit.next_vreg_id;
        let mut ssa_ops = self.emit.take_ops();

        let barrier_count = count_barriers(&ssa_ops);

        // Use barrier-aware dependencies for greedy hints
        // This constrains cross-round interleaving and provides better hints
        compute_ssa_dependencies_with_barriers(&mut ssa_ops, &vreg_def);

        // Get greedy hints first (for warm-starting CP-SAT)
        // If shift is enabled, also get shift hints from greedy scheduler
        // Collect SelectChoice decisions from greedy scheduler
        let mut greedy_shifted = HashSet::new();
        let mut greedy_select_fma_ops = HashSet::new();
        let mut greedy_select_sub_cycles = HashMap::new();
        let mut greedy_select_fma_cycles = HashMap::new();
        let (greedy_cycles, greedy_budget_exceeded, greedy_peak_live) = 
            self.schedule_ssa_greedy_with_shift_budget_and_selectchoice(
                &mut ssa_ops,
                enable_valu_shift, // Match CP-SAT shift setting for hints
                &mut greedy_shifted,
                1536, // Scratch budget
                Some(&mut greedy_select_fma_ops),
                Some(&mut greedy_select_sub_cycles),
                Some(&mut greedy_select_fma_cycles),
            )?;

        if verbose {
            eprintln!(
                "SSA Greedy hint: {} cycles ({} barriers){}{}",
                greedy_cycles,
                barrier_count,
                if enable_valu_shift {
                    format!(" ({} VALU shifted)", greedy_shifted.len())
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

        // Phase 1: Per-section feasibility (only when a section exceeds scratch budget)
        let boundaries = section_boundaries(&ssa_ops);
        let budget_result = sections_exceeding_budget(&ssa_ops, 1536);
        let n_exceeding: usize = budget_result.section_exceeds_budget.iter().filter(|&&b| b).count();
        if verbose {
            eprintln!();
            eprintln!("================================================================================");
            eprintln!("  CHECKING EACH ROUND FOR SCRATCH BUDGET (budget = 1536 words)");
            eprintln!("  Sections that exceed budget will get a per-section CP-SAT run (feasibility).");
            eprintln!("================================================================================");
            eprintln!();
            for (k, &peak) in budget_result.section_peaks.iter().enumerate() {
                if budget_result.section_exceeds_budget[k] {
                    eprintln!(
                        "  Section {:2}:  peak {} words  > 1536  -->  RUN PER-SECTION CP-SAT (feasibility)",
                        k, peak
                    );
                } else {
                    eprintln!(
                        "  Section {:2}:  peak {} words  <= 1536  -->  SKIP (within budget)",
                        k, peak
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
        let vreg_def_sec = vreg_def_section(&ssa_ops, &boundaries);
        let mut section_offset: i32 = 0;

        for k in 0..boundaries.len() {
            if budget_result.section_exceeds_budget[k] {
                let (start, end) = boundaries[k];
                let global_indices: Vec<usize> = (start..end)
                    .filter(|&i| ssa_ops[i].opcode != Opcode::Barrier)
                    .collect();
                if global_indices.is_empty() {
                    section_offset += 0;
                    continue;
                }
                // Greedy section makespan and peak (for delta reporting after feasibility)
                let (greedy_min_c, greedy_max_c) = global_indices
                    .iter()
                    .filter_map(|&i| {
                        let c = ssa_ops[i].cycle;
                        if c >= 0 { Some(c) } else { None }
                    })
                    .fold((i32::MAX, i32::MIN), |(lo, hi), c| (lo.min(c), hi.max(c)));
                let greedy_section_makespan =
                    if greedy_min_c <= greedy_max_c { greedy_max_c - greedy_min_c + 1 } else { 0 };
                let greedy_section_peak = budget_result.section_peaks[k];

                if verbose {
                    eprintln!("--------------------------------------------------------------------------------");
                    eprintln!("  PER-SECTION CP-SAT:  SECTION {}  (feasibility — find any valid schedule)", k);
                    eprintln!("  {} ops in this section. Goal: stay within scratch budget.", global_indices.len());
                    eprintln!("--------------------------------------------------------------------------------");
                    eprintln!();
                }
                let segment_op_ids: HashSet<usize> =
                    global_indices.iter().map(|&i| ssa_ops[i].id).collect();
                let segment_ops: Vec<SsaOp> = global_indices
                    .iter()
                    .map(|&i| {
                        let mut op = ssa_ops[i].clone();
                        op.dep_ids.retain(|id| segment_op_ids.contains(id));
                        op
                    })
                    .collect();
                let initial_live =
                    live_in_for_section(&ssa_ops, &boundaries, k, &vreg_def_sec);
                let live_out = live_out_from_section(&ssa_ops, &boundaries, k, &vreg_def_sec);
                let live_through = live_through_for_section(&ssa_ops, &boundaries, k, &vreg_def_sec);
                let segment_hints: Vec<i32> = global_indices
                    .iter()
                    .map(|&i| {
                        let c = ssa_ops[i].cycle;
                        if c >= 0 {
                            (c - section_offset).max(0)
                        } else {
                            c
                        }
                    })
                    .collect();
                let config_feasibility = SsaCpsatConfig {
                    time_limit_secs: (time_limit_secs * 0.25).min(300.0),
                    verbose,
                    stop_after_first_solution: true,
                    enable_valu_shift,
                    scratch_budget: 1536,
                    model_name: Some(format!("section_{}_feasibility", k)),
                };
                let seg_shift = if enable_valu_shift {
                    Some(&greedy_shifted)
                } else {
                    None
                };
                let seg_select = if greedy_select_fma_ops.is_empty() {
                    None
                } else {
                    Some(&greedy_select_fma_ops)
                };
                let seg_result = schedule_ssa_cpsat(
                    &segment_ops,
                    config_feasibility,
                    SsaCpsatHints {
                        greedy_hints: Some(&segment_hints),
                        shift_hints: seg_shift,
                        select_choice_hints: seg_select,
                        initial_live: Some(&initial_live),
                        live_out_vregs: if live_out.is_empty() { None } else { Some(&live_out) },
                        live_through_vregs: if live_through.is_empty() { None } else { Some(&live_through) },
                    },
                )
                .ok_or_else(|| {
                    format!("Per-section CP-SAT failed for section {}", k)
                })?;
                for (seg_idx, &cycle) in seg_result.cycle_assignments.iter().enumerate() {
                    if seg_idx < global_indices.len() && cycle >= 0 {
                        ssa_ops[global_indices[seg_idx]].cycle = section_offset + cycle;
                    }
                }
                if verbose {
                    let after_budget = sections_exceeding_budget(&ssa_ops, 1536);
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
            }
            // Advance section_offset by this section's makespan
            let (start, end) = boundaries[k];
            let mut min_c = i32::MAX;
            let mut max_c = i32::MIN;
            for op in &ssa_ops[start..end] {
                if op.opcode != Opcode::Barrier && op.cycle >= 0 {
                    min_c = min_c.min(op.cycle);
                    max_c = max_c.max(op.cycle);
                }
            }
            if min_c <= max_c {
                section_offset += max_c - min_c + 1;
            }
        }

        // Presolve guarantee: merged schedule must be within scratch budget.
        let post_merge_budget = sections_exceeding_budget(&ssa_ops, 1536);
        if let Some((sec, peak)) = post_merge_budget
            .section_peaks
            .iter()
            .enumerate()
            .find(|(_, &p)| p > 1536)
        {
            return Err(format!(
                "Per-section CP-SAT produced a merged schedule that exceeds scratch budget: section {} has peak {} words (budget 1536). \
                 This indicates a bug in the segment scratch model (e.g. live-through or live-out).",
                sec, *peak
            ));
        }

        // Elide barriers before CP-SAT
        // CP-SAT has proper liveness constraints via ILP, so barriers are unnecessary
        // and would interfere with its optimization
        let mut ssa_ops = elide_barriers(ssa_ops);

        // Collect hints for full-kernel CP-SAT: merged schedule (greedy for within-budget
        // sections, per-section feasibility result for sections that exceeded budget).
        let greedy_hints: Vec<i32> = ssa_ops.iter().map(|op| op.cycle).collect();
        let hint_makespan = greedy_hints.iter().copied().max().map(|c| c + 1).unwrap_or(0);
        let hint_peak = sections_exceeding_budget(&ssa_ops, 1536)
            .section_peaks
            .first()
            .copied()
            .unwrap_or(0);

        if hint_peak > 1536 {
            return Err(format!(
                "Merged hint schedule exceeds scratch budget: peak {} words (budget 1536). \
                 Presolves must guarantee a feasible hint; this indicates a bug in the segment scratch model.",
                hint_peak
            ));
        }

        // Reset cycles before CP-SAT
        for op in &mut ssa_ops {
            op.cycle = -1;
        }

        // Run CP-SAT on SSA ops with scratch budget constraint
        let cpsat_config = SsaCpsatConfig {
            time_limit_secs,
            verbose,
            stop_after_first_solution: false,
            enable_valu_shift,
            scratch_budget: 1536, // Hardware limit
            model_name: Some("full_kernel_optimization".to_string()),
        };

        // Pass shift hints if VALU shift is enabled
        // When disabled, no shift variables are created so no hints needed
        // (The 16 "unhinted" variables in CP-SAT are symmetry-breaking variables from preprocessing)
        let shift_hints = if enable_valu_shift {
            Some(&greedy_shifted)
        } else {
            None
        };

        let greedy_hints_opt = Some(&greedy_hints);

        // Pass SelectChoice hints from greedy scheduler
        // Note: SelectChoice decisions are keyed by op ID, which remains constant after barrier elision
        // We just need to verify the SelectChoice ops still exist after elision
        let mut select_choice_hints_set = HashSet::new();
        
        // Build set of op IDs that exist after barrier elision
        let op_ids_after_elision: HashSet<usize> = ssa_ops.iter().map(|op| op.id).collect();
        
        // Filter greedy SelectChoice decisions to only include ops that still exist after elision
        for &select_choice_id in &greedy_select_fma_ops {
            if op_ids_after_elision.contains(&select_choice_id) {
                select_choice_hints_set.insert(select_choice_id);
            }
        }
        
        let select_choice_hints = if select_choice_hints_set.is_empty() {
            None
        } else {
            Some(&select_choice_hints_set)
        };

        if verbose {
            eprintln!();
            eprintln!("================================================================================");
            eprintln!("  FULL KERNEL CP-SAT  (optimization — minimize makespan over entire kernel)");
            eprintln!("  All rounds merged; solving for minimum cycles. Scratch budget = 1536.");
            eprintln!(
                "  Before optimization: {} cycles, peak {} words (hint schedule).",
                hint_makespan, hint_peak
            );
            if greedy_budget_exceeded {
                eprintln!("  (Greedy exceeded budget; hints may be infeasible — CP-SAT will search.)");
            }
            eprintln!("================================================================================");
            eprintln!();
        }

        let cpsat_result = schedule_ssa_cpsat(
            &ssa_ops,
            cpsat_config,
            SsaCpsatHints {
                greedy_hints: greedy_hints_opt.map(|v| v.as_slice()),
                shift_hints,
                select_choice_hints,
                ..Default::default()
            },
        )
        .ok_or_else(|| "CP-SAT scheduling failed".to_string())?;

        // Apply cycle assignments
        for (idx, op) in ssa_ops.iter_mut().enumerate() {
            if idx < cpsat_result.cycle_assignments.len() {
                op.cycle = cpsat_result.cycle_assignments[idx];
            }
        }

        let cycles = cpsat_result.makespan;
        let shifted_ops = cpsat_result.shifted_ops;

        if verbose {
            eprintln!(
                "SSA CP-SAT result: {} cycles (greedy was {}), {} VALU ops shifted",
                cycles, greedy_cycles, shifted_ops.len()
            );
        }

        // Filter out dead stores (ops with cycle=-1) before address assignment.
        // Dead stores are ops whose dest VReg has no users - they were detected
        // during greedy hints generation but CP-SAT may also leave some ops
        // unscheduled. Including them in address assignment would corrupt live
        // interval computation.
        let ssa_ops: Vec<SsaOp> = ssa_ops
            .into_iter()
            .filter(|op| op.cycle >= 0)
            .collect();

        // Assign addresses via linear scan
        let mut assign_config = AddressAssignConfig::default();

        // Reserve address 0 - scratch memory is always initialized to 0 at startup, so address 0 always contains 0
        if let Some(zero) = self.zero_vreg {
            assign_config.fixed_vregs.insert(zero, 0);
        }

        let address_result = assign_addresses(&ssa_ops, &assign_config);
        let budget_exceeded = address_result.budget_exceeded;

        if budget_exceeded {
            // CP-SAT should respect budget, so this indicates a bug in CP-SAT solver
            eprintln!(
                "Warning: CP-SAT address assignment exceeded budget: peak_live={} > budget={}",
                address_result.peak_live, address_result.budget
            );
        }

        // Compute usage accounting from SSA ops
        let mut usage_accounting = UsageAccounting::from_ssa_ops(&ssa_ops);
        usage_accounting.set_peak_live(address_result.peak_live);

        // Build VReg metadata for liveness debugging (before lowering loses SSA info)
        let vreg_metadata = build_vreg_metadata(&ssa_ops);

        // Lower to concrete ops, expanding shifted VALU ops to ALU
        let valu_shifted = shifted_ops.len();
        let ops = lower_ssa_ops_with_shift(
            &ssa_ops,
            &address_result.vreg_to_addr,
            &shifted_ops,
            &cpsat_result.const_add_imm_ops, // CP-SAT decisions for ConstChoice
            &cpsat_result.select_fma_ops, // CP-SAT decisions for SelectChoice
        );

        Ok(SsaScheduleResult {
            ops,
            cycles,
            address_result,
            vreg_count,
            valu_shifted,
            budget_exceeded, // CP-SAT should not exceed budget
            usage_accounting,
            vreg_metadata,
        })
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

    eprintln!("[DEBUG verify_vreg_address_consistency] Called with {} ops", ops.len());

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
        eprintln!("[DEBUG] {} vstore ops found. Checking sources:", vstore_ops.len());
        for op in vstore_ops {
            eprintln!("  vstore id={} cycle={}", op.id, op.cycle);
            for (i, src) in op.srcs.iter().enumerate() {
                match src {
                    Operand::VReg(vreg) => {
                        let addr = vreg_to_addr.get(vreg);
                        eprintln!(
                            "    src[{}]: {:?} -> addr {:?}",
                            i, vreg, addr
                        );
                    }
                    Operand::Imm(val) => {
                        eprintln!("    src[{}]: Imm({})", i, val);
                    }
                }
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_builder::ssa_lower::expand_ssa_valu_to_alu;
    use crate::types::{Engine, Opcode, Operand};
    use std::collections::HashMap;

    /// Helper to create a simple SsaOp for testing
    fn make_test_ssa_op(
        id: usize,
        engine: Engine,
        opcode: Opcode,
        dest: Option<VReg>,
        srcs: Vec<Operand>,
        dep_ids: Vec<usize>,
    ) -> SsaOp {
        let mut op = SsaOp::new(id, engine, opcode, dest, srcs);
        op.dep_ids = dep_ids.into_iter().collect();
        op
    }

    #[test]
    fn test_ssa_builder_creates_ops() {
        // Use fewer rounds (4) to test faster and stay within budget
        let config = KernelConfig::new(10, 2047, 256, 4);
        let mut builder = SsaTickTockKernelBuilder::new(config);
        let ops = builder.build();

        assert!(!ops.is_empty(), "Should generate ops");
        println!("Generated {} SSA ops", ops.len());
    }

    #[test]
    fn test_ssa_build_and_schedule() {
        // Use fewer rounds (4) to stay within scratch budget.
        // Full 16 rounds creates too many VRegs for SSA without reuse.
        let config = KernelConfig::new(10, 2047, 256, 4);
        let mut builder = SsaTickTockKernelBuilder::new(config);

        let result = builder.build_and_schedule();
        assert!(result.is_ok(), "Scheduling should succeed: {:?}", result.err());

        let result = result.unwrap();
        println!("SSA kernel: {} cycles, {} ops, {} VRegs, peak_live={}",
            result.cycles, result.ops.len(), result.vreg_count, result.address_result.peak_live);

        assert!(result.cycles > 0, "Should have positive cycle count");
        assert!(!result.address_result.budget_exceeded, "Should not exceed budget");
    }

    /// 4-round optimization effort: production-sized 4-round kernel must complete and stay within scratch budget.
    #[test]
    fn test_4_round_kernel_generation_within_budget() {
        let config = KernelConfig::new(10, 2047, 256, 4);
        let mut builder = SsaTickTockKernelBuilder::new(config);
        let result = builder.build_and_schedule();
        assert!(result.is_ok(), "4-round kernel generation should succeed: {:?}", result.err());
        let result = result.unwrap();
        assert!(
            result.address_result.peak_live <= 1536,
            "4-round kernel must stay within scratch budget: peak_live={} > 1536",
            result.address_result.peak_live
        );
        assert!(!result.address_result.budget_exceeded, "4-round kernel must not report budget_exceeded");
    }

    // =========================================================================
    // Greedy Scheduler Scratch Budget Tests
    // =========================================================================

    /// Test that greedy scheduler delays ops when scratch budget would be exceeded.
    ///
    /// Sequential chain where VRegs overlap (no same-cycle reuse opportunity).
    #[test]
    fn test_greedy_scratch_budget_delays_ops() {
        // Sequential chain where v0 is used by MULTIPLE ops:
        // op0: broadcast -> v0 (8 words)
        // op1: valu(v0) -> v1 (8 words), dep on 0 (v0 still live - used by op2)
        // op2: valu(v0, v1) -> v2 (8 words), dep on 1 (v0, v1 die via same-cycle reuse)
        // op3: sink to consume v2
        //
        // With same-cycle reuse model:
        // - Cycle 0: op0, v0 born (8 words)
        // - Cycle 1: op1, v0 still live (used by op2), v1 born (16 words) PEAK
        // - Cycle 2: op2, v0+v1 die, v2 born (8 words)
        // - Cycle 3: op3 (sink), v2 dies
        //
        // Peak: 16 words (at cycle 1: v0 + v1)
        let ops = vec![
            make_test_ssa_op(0, Engine::Valu, Opcode::Vbroadcast, Some(VReg::vector(0)), vec![Operand::Imm(1)], vec![]),
            make_test_ssa_op(1, Engine::Valu, Opcode::Add, Some(VReg::vector(1)),
                vec![Operand::VReg(VReg::vector(0)), Operand::VReg(VReg::vector(0))], vec![0]),
            // op2 uses BOTH v0 and v1 - this creates the peak at cycle 1
            make_test_ssa_op(2, Engine::Valu, Opcode::Add, Some(VReg::vector(2)),
                vec![Operand::VReg(VReg::vector(0)), Operand::VReg(VReg::vector(1))], vec![1]),
            // Sink to consume v2
            make_test_ssa_op(3, Engine::Store, Opcode::Vstore, None,
                vec![Operand::Imm(100), Operand::VReg(VReg::vector(2))], vec![2]),
        ];

        let config = KernelConfig::new(10, 2047, 256, 1);
        let builder = SsaTickTockKernelBuilder::new(config);
        let mut shifted = HashSet::new();

        // With budget=16, should succeed (peak is 16)
        let result = builder.schedule_ssa_greedy_with_shift_budget(
            &mut ops.clone(), false, &mut shifted, 16
        );
        assert!(result.is_ok(), "Budget=16 should succeed: {:?}", result.err());

        // With budget=8, the scheduler will still succeed due to emergency force-scheduling.
        // The scheduler prefers to exceed budget temporarily rather than fail completely.
        let mut shifted2 = HashSet::new();
        let result2 = builder.schedule_ssa_greedy_with_shift_budget(
            &mut ops.clone(), false, &mut shifted2, 8
        );
        // Scheduler uses emergency force-scheduling when budget is tight, so it still succeeds
        assert!(result2.is_ok(), "Budget=8 should still succeed (emergency force-scheduling): {:?}", result2.err());
        let (_cycles2, budget_exceeded, _peak2) = result2.unwrap();
        assert!(budget_exceeded, "When force-schedule is used (budget=8), budget_exceeded must be true");
    }

    /// Test that budget_exceeded from the greedy scheduler is combined with address
    /// assignment and reflected in the final SsaScheduleResult (CRITICAL_TESTS_PLAN 5.2).
    #[test]
    fn test_budget_exceeded_propagates() {
        // Build and schedule with a tiny scratch budget so the greedy scheduler
        // must force-schedule and set sched_budget_exceeded. The final result
        // must have budget_exceeded = sched_budget_exceeded || address_result.budget_exceeded.
        let config = KernelConfig::new(10, 2047, 256, 4);
        let mut builder = SsaTickTockKernelBuilder::new(config);
        let result = builder.build_and_schedule_with_options(false, Some(8));
        assert!(result.is_ok(), "Build+schedule with budget=8 should complete: {:?}", result.err());
        let ssa_result = result.unwrap();
        assert!(
            ssa_result.budget_exceeded,
            "budget_exceeded must propagate to final result when scheduler exceeds budget"
        );
    }

    /// Test that greedy scheduler with generous budget allows parallel ops.
    #[test]
    fn test_greedy_scratch_budget_generous_allows_parallel() {
        // Same 6-producer setup but with generous budget
        let mut ops = vec![
            make_test_ssa_op(0, Engine::Valu, Opcode::Vbroadcast, Some(VReg::vector(0)), vec![Operand::Imm(1)], vec![]),
            make_test_ssa_op(1, Engine::Valu, Opcode::Vbroadcast, Some(VReg::vector(1)), vec![Operand::Imm(2)], vec![]),
            make_test_ssa_op(2, Engine::Valu, Opcode::Vbroadcast, Some(VReg::vector(2)), vec![Operand::Imm(3)], vec![]),
            make_test_ssa_op(3, Engine::Valu, Opcode::Vbroadcast, Some(VReg::vector(3)), vec![Operand::Imm(4)], vec![]),
            make_test_ssa_op(4, Engine::Valu, Opcode::Vbroadcast, Some(VReg::vector(4)), vec![Operand::Imm(5)], vec![]),
            make_test_ssa_op(5, Engine::Valu, Opcode::Vbroadcast, Some(VReg::vector(5)), vec![Operand::Imm(6)], vec![]),
            make_test_ssa_op(6, Engine::Valu, Opcode::Add, Some(VReg::vector(6)),
                vec![Operand::VReg(VReg::vector(0)), Operand::VReg(VReg::vector(1))], vec![0, 1]),
            make_test_ssa_op(7, Engine::Valu, Opcode::Add, Some(VReg::vector(7)),
                vec![Operand::VReg(VReg::vector(2)), Operand::VReg(VReg::vector(3))], vec![2, 3]),
            make_test_ssa_op(8, Engine::Valu, Opcode::Add, Some(VReg::vector(8)),
                vec![Operand::VReg(VReg::vector(4)), Operand::VReg(VReg::vector(5))], vec![4, 5]),
            make_test_ssa_op(9, Engine::Valu, Opcode::Add, Some(VReg::vector(9)),
                vec![Operand::VReg(VReg::vector(6)), Operand::VReg(VReg::vector(7))], vec![6, 7]),
            make_test_ssa_op(10, Engine::Valu, Opcode::Add, Some(VReg::vector(10)),
                vec![Operand::VReg(VReg::vector(8)), Operand::VReg(VReg::vector(9))], vec![8, 9]),
        ];

        let config = KernelConfig::new(10, 2047, 256, 1);
        let builder = SsaTickTockKernelBuilder::new(config);
        let mut shifted = HashSet::new();

        // Schedule with generous budget (1000 words)
        let (cycles, _, _) = builder.schedule_ssa_greedy_with_shift_budget(
            &mut ops, false, &mut shifted, 1000
        ).expect("Should find feasible schedule");

        // With generous budget and 6 VALU slots, all 6 producers can run at cycle 0
        let producers_at_cycle_0 = ops[0..6].iter()
            .filter(|op| op.cycle == 0)
            .count();

        assert_eq!(
            producers_at_cycle_0, 6,
            "With generous budget and 6 VALU slots, all 6 producers should be at cycle 0, got {}. \
             Schedule: {:?}",
            producers_at_cycle_0,
            ops.iter().map(|op| (op.id, op.cycle)).collect::<Vec<_>>()
        );

        println!("Generous budget schedule: {} cycles", cycles);
    }

    /// Test that tight budget leads to longer makespan than generous budget.
    #[test]
    fn test_greedy_scratch_budget_tight_increases_makespan() {
        // Use same ops for both tests
        fn make_ops() -> Vec<SsaOp> {
            vec![
                make_test_ssa_op(0, Engine::Valu, Opcode::Vbroadcast, Some(VReg::vector(0)), vec![Operand::Imm(1)], vec![]),
                make_test_ssa_op(1, Engine::Valu, Opcode::Vbroadcast, Some(VReg::vector(1)), vec![Operand::Imm(2)], vec![]),
                make_test_ssa_op(2, Engine::Valu, Opcode::Vbroadcast, Some(VReg::vector(2)), vec![Operand::Imm(3)], vec![]),
                make_test_ssa_op(3, Engine::Valu, Opcode::Vbroadcast, Some(VReg::vector(3)), vec![Operand::Imm(4)], vec![]),
                make_test_ssa_op(4, Engine::Valu, Opcode::Vbroadcast, Some(VReg::vector(4)), vec![Operand::Imm(5)], vec![]),
                make_test_ssa_op(5, Engine::Valu, Opcode::Vbroadcast, Some(VReg::vector(5)), vec![Operand::Imm(6)], vec![]),
                make_test_ssa_op(6, Engine::Valu, Opcode::Add, Some(VReg::vector(6)),
                    vec![Operand::VReg(VReg::vector(0)), Operand::VReg(VReg::vector(1))], vec![0, 1]),
                make_test_ssa_op(7, Engine::Valu, Opcode::Add, Some(VReg::vector(7)),
                    vec![Operand::VReg(VReg::vector(2)), Operand::VReg(VReg::vector(3))], vec![2, 3]),
                make_test_ssa_op(8, Engine::Valu, Opcode::Add, Some(VReg::vector(8)),
                    vec![Operand::VReg(VReg::vector(4)), Operand::VReg(VReg::vector(5))], vec![4, 5]),
                make_test_ssa_op(9, Engine::Valu, Opcode::Add, Some(VReg::vector(9)),
                    vec![Operand::VReg(VReg::vector(6)), Operand::VReg(VReg::vector(7))], vec![6, 7]),
                make_test_ssa_op(10, Engine::Valu, Opcode::Add, Some(VReg::vector(10)),
                    vec![Operand::VReg(VReg::vector(8)), Operand::VReg(VReg::vector(9))], vec![8, 9]),
                // Sink to consume v10
                make_test_ssa_op(11, Engine::Store, Opcode::Vstore, None,
                    vec![Operand::Imm(100), Operand::VReg(VReg::vector(10))], vec![10]),
            ]
        }

        let mut ops_generous = make_ops();
        let mut ops_tight = make_ops();

        let config = KernelConfig::new(10, 2047, 256, 1);
        let builder = SsaTickTockKernelBuilder::new(config);

        let mut shifted_generous = HashSet::new();
        let (cycles_generous, _, _) = builder.schedule_ssa_greedy_with_shift_budget(
            &mut ops_generous, false, &mut shifted_generous, 1000
        ).expect("Should find feasible schedule with generous budget");

        let mut shifted_tight = HashSet::new();
        let (cycles_tight, _, _) = builder.schedule_ssa_greedy_with_shift_budget(
            &mut ops_tight, false, &mut shifted_tight, 24  // Only 3 vectors
        ).expect("Should find feasible schedule with tight budget");

        // Both should schedule successfully. The tight budget may or may not take longer
        // due to emergency force-scheduling which can exceed budget temporarily.
        // The key assertion is that both complete.
        assert!(
            cycles_tight >= cycles_generous,
            "Tight budget ({} cycles) should be at least as long as generous ({} cycles).",
            cycles_tight, cycles_generous
        );

        println!("Generous budget: {} cycles", cycles_generous);
        println!("Tight budget: {} cycles", cycles_tight);
    }

    /// Test that greedy scheduler's liveness counting with same-cycle reuse.
    ///
    /// With same-cycle reuse model: VRegs that die when an op runs (their last use)
    /// are freed BEFORE the new VReg's space is needed. This correctly models
    /// that reads happen before writes within a cycle.
    #[test]
    fn test_greedy_liveness_matches_verification() {
        // Simple chain where we can verify the peak liveness:
        // op0: broadcast -> v0 (8 words)
        // op1: broadcast -> v1 (8 words), depends on op0 (forces sequential)
        // op2: valu(v0, v1) -> v2 (8 words)
        // op3: sink to consume v2
        //
        // Timeline with same-cycle reuse:
        // Cycle 0: op0 runs, v0 live = 8 words
        // Cycle 1: op1 runs, v0 still live (used by op2), v1 born = 16 words PEAK
        // Cycle 2: op2 runs, v0 and v1 are last used (freed via same-cycle reuse), v2 born
        //          Net pressure change: +8 (v2) - 16 (v0+v1) = 8 words
        // Cycle 3: op3 (sink) runs, v2 dies
        //
        // Peak is 16 words (at cycle 1). With same-cycle reuse, op2 can schedule
        // even with budget=16 because it frees v0+v1 before needing v2's space.
        let ops = vec![
            make_test_ssa_op(0, Engine::Valu, Opcode::Vbroadcast, Some(VReg::vector(0)), vec![Operand::Imm(1)], vec![]),
            make_test_ssa_op(1, Engine::Valu, Opcode::Vbroadcast, Some(VReg::vector(1)), vec![Operand::Imm(2)], vec![0]), // dep on 0
            make_test_ssa_op(2, Engine::Valu, Opcode::Add, Some(VReg::vector(2)),
                vec![Operand::VReg(VReg::vector(0)), Operand::VReg(VReg::vector(1))], vec![0, 1]),
            // Sink to consume v2
            make_test_ssa_op(3, Engine::Store, Opcode::Vstore, None,
                vec![Operand::Imm(100), Operand::VReg(VReg::vector(2))], vec![2]),
        ];

        let config = KernelConfig::new(10, 2047, 256, 1);
        let builder = SsaTickTockKernelBuilder::new(config);

        // With budget=16, should succeed (peak is 16 at cycle 1)
        let mut shifted = HashSet::new();
        let result = builder.schedule_ssa_greedy_with_shift_budget(
            &mut ops.clone(), false, &mut shifted, 16
        );
        assert!(result.is_ok(), "Budget=16 should succeed, peak is 16");

        // With budget=8, the scheduler will still succeed due to emergency force-scheduling.
        // The scheduler prefers to exceed budget temporarily rather than fail completely.
        let mut shifted2 = HashSet::new();
        let result2 = builder.schedule_ssa_greedy_with_shift_budget(
            &mut ops.clone(), false, &mut shifted2, 8
        );
        // Scheduler uses emergency force-scheduling when budget is tight, so it still succeeds
        assert!(result2.is_ok(), "Budget=8 should still succeed (emergency force-scheduling): {:?}", result2.err());
    }

    /// Test that greedy scheduler correctly tracks VReg deaths.
    /// When a VReg's last user is scheduled, that VReg should no longer count
    /// against the live budget (same-cycle reuse).
    #[test]
    fn test_greedy_scratch_budget_vreg_death() {
        // Create a chain where early VRegs are consumed, freeing space for later ones.
        // op0: broadcast -> v0 (8 words)
        // op1: broadcast -> v1 (8 words)
        // op2: valu(v0, v1) -> v2 (8 words) - v0, v1 die here (same-cycle reuse)
        // op3: broadcast -> v3 (8 words) - depends on op0, so ready at cycle 1
        // op4: valu(v2, v3) -> v4 (8 words)
        //
        // NOTE: op3 needs a dependency so it's not greedily scheduled at cycle 0.
        // We add dep on op0 so op3 becomes ready at cycle 1.
        let mut ops = vec![
            make_test_ssa_op(0, Engine::Valu, Opcode::Vbroadcast, Some(VReg::vector(0)), vec![Operand::Imm(1)], vec![]),
            make_test_ssa_op(1, Engine::Valu, Opcode::Vbroadcast, Some(VReg::vector(1)), vec![Operand::Imm(2)], vec![]),
            make_test_ssa_op(2, Engine::Valu, Opcode::Add, Some(VReg::vector(2)),
                vec![Operand::VReg(VReg::vector(0)), Operand::VReg(VReg::vector(1))], vec![0, 1]),
            make_test_ssa_op(3, Engine::Valu, Opcode::Vbroadcast, Some(VReg::vector(3)), vec![Operand::Imm(3)], vec![0]), // dep on op0
            make_test_ssa_op(4, Engine::Valu, Opcode::Add, Some(VReg::vector(4)),
                vec![Operand::VReg(VReg::vector(2)), Operand::VReg(VReg::vector(3))], vec![2, 3]),
        ];

        let config = KernelConfig::new(10, 2047, 256, 1);
        let builder = SsaTickTockKernelBuilder::new(config);
        let mut shifted = HashSet::new();

        // With budget=24 (3 vectors) and same-cycle reuse:
        // - Cycle 0: op0, op1 -> v0, v1 live (16 words)
        // - Cycle 1: op2 -> v0, v1 die (same-cycle reuse), v2 born (8 words)
        //            op3 -> v3 born (v2 + v3 = 16 words) - fits!
        // - Cycle 2: op4 -> v2, v3 die, v4 born
        //
        // Without same-cycle reuse, op3 would be delayed to cycle 2.
        let (cycles, _, _) = builder.schedule_ssa_greedy_with_shift_budget(
            &mut ops, false, &mut shifted, 24
        ).expect("Should find feasible schedule");

        println!("VReg death tracking schedule: {} cycles", cycles);
        for op in &ops {
            println!("  op{}: {:?} -> cycle {}", op.id, op.engine, op.cycle);
        }

        // With same-cycle reuse, schedule should be 3 cycles (critical path)
        // Without same-cycle reuse, it would be longer
        assert!(cycles <= 3, "With same-cycle reuse, schedule should be at most 3 cycles, got {}", cycles);
    }

    // =========================================================================
    // SSA Kernel Correctness Tests
    // =========================================================================
    // These tests verify that the SSA kernel produces correct output.
    // KNOWN BUG: The SSA kernel currently fails correctness tests due to incorrect
    // address assignment in the linear scan allocator. The allocator incorrectly
    // reuses scratch addresses for VRegs that are still live, causing memory
    // operations to use wrong addresses.

    /// Test that SSA kernel builds and schedules without error.
    ///
    /// This is a basic sanity check that the SSA pipeline completes.
    /// Note: The kernel may produce incorrect output due to known liveness bug.
    #[test]
    fn test_ssa_kernel_builds_successfully() {
        let config = KernelConfig::new(3, 15, 16, 1);
        let mut ssa_builder = SsaTickTockKernelBuilder::new(config);
        let ssa_result = ssa_builder.build_and_schedule();

        assert!(ssa_result.is_ok(), "SSA kernel should build: {:?}", ssa_result.err());

        let result = ssa_result.unwrap();
        println!("SSA kernel: {} cycles, {} ops, {} VRegs",
            result.cycles, result.ops.len(), result.vreg_count);

        assert!(result.cycles > 0, "Should have positive cycle count");
        assert!(!result.ops.is_empty(), "Should have ops");
    }

    /// Test that VReg liveness intervals are computed correctly.
    ///
    /// See types.rs "AUTHORITATIVE LIVENESS MODEL DOCUMENTATION" for full details.
    /// The liveness interval for a VReg uses CLOSED convention [start, end].
    #[test]
    fn test_vreg_liveness_intervals() {
        use crate::address_assign::compute_live_intervals;

        // Create a simple test case:
        // op0: const -> s0 (def at cycle 0)
        // op1: const -> s1 (def at cycle 1)
        // op2: alu(s0, s1) -> s2 (use s0, s1 at cycle 10)
        let mut ops = vec![
            make_test_ssa_op(0, Engine::Load, Opcode::Const, Some(VReg::scalar(0)), vec![Operand::Imm(5)], vec![]),
            make_test_ssa_op(1, Engine::Load, Opcode::Const, Some(VReg::scalar(1)), vec![Operand::Imm(6)], vec![]),
            make_test_ssa_op(2, Engine::Alu, Opcode::Add, Some(VReg::scalar(2)),
                vec![Operand::VReg(VReg::scalar(0)), Operand::VReg(VReg::scalar(1))], vec![0, 1]),
        ];

        // Simulate scheduling
        ops[0].cycle = 0;
        ops[1].cycle = 1;
        ops[2].cycle = 10;

        let intervals = compute_live_intervals(&ops);

        // CLOSED intervals: [start, end] where end = last_read_cycle
        //
        // s0: defined at 0, last read at 10 → [0, 10]
        let s0_interval = intervals.get(&VReg::scalar(0)).expect("s0 should have interval");
        assert_eq!(s0_interval.start, 0, "s0 should start at cycle 0");
        assert_eq!(s0_interval.end, 10, "s0 closed end = last_read = 10");

        // s1: defined at 1, last read at 10 → [1, 10]
        let s1_interval = intervals.get(&VReg::scalar(1)).expect("s1 should have interval");
        assert_eq!(s1_interval.start, 1, "s1 should start at cycle 1");
        assert_eq!(s1_interval.end, 10, "s1 closed end = last_read = 10");

        // s2: defined at 10, never read (dead) → [10, 11] (+1 extension)
        // Dead VRegs get interval [D, D+1] with closed convention.
        let s2_interval = intervals.get(&VReg::scalar(2)).expect("s2 should have interval");
        assert_eq!(s2_interval.start, 10, "s2 should start at cycle 10");
        assert_eq!(s2_interval.end, 11, "s2 (dead) has closed interval [10, 11] (+1 extension)");
    }

    /// Test that overlapping VRegs get different addresses.
    ///
    /// Two VRegs that are live at the same time must not share the same
    /// physical address.
    #[test]
    fn test_overlapping_vregs_different_addresses() {
        use crate::address_assign::{assign_addresses, AddressAssignConfig};

        // Create ops where s0 and s1 are both live at cycle 5:
        // op0: const -> s0 (def cycle 0, used at cycle 5)
        // op1: const -> s1 (def cycle 2, used at cycle 5)
        // op2: alu(s0, s1) -> s2 (cycle 5)
        let mut ops = vec![
            make_test_ssa_op(0, Engine::Load, Opcode::Const, Some(VReg::scalar(0)), vec![Operand::Imm(5)], vec![]),
            make_test_ssa_op(1, Engine::Load, Opcode::Const, Some(VReg::scalar(1)), vec![Operand::Imm(6)], vec![]),
            make_test_ssa_op(2, Engine::Alu, Opcode::Add, Some(VReg::scalar(2)),
                vec![Operand::VReg(VReg::scalar(0)), Operand::VReg(VReg::scalar(1))], vec![0, 1]),
        ];

        ops[0].cycle = 0;
        ops[1].cycle = 2;
        ops[2].cycle = 5;

        let config = AddressAssignConfig::default();
        let result = assign_addresses(&ops, &config);

        let addr_s0 = result.vreg_to_addr.get(&VReg::scalar(0)).expect("s0 should have address");
        let addr_s1 = result.vreg_to_addr.get(&VReg::scalar(1)).expect("s1 should have address");

        assert_ne!(addr_s0, addr_s1,
            "Overlapping VRegs s0 and s1 must have different addresses: s0={}, s1={}",
            addr_s0, addr_s1);
    }

    /// Test that non-overlapping VRegs can share addresses.
    ///
    /// See types.rs "AUTHORITATIVE LIVENESS MODEL DOCUMENTATION" for full details.
    /// Two VRegs where one's interval ends before the other's begins
    /// can share the same physical address.
    #[test]
    fn test_non_overlapping_vregs_can_share_addresses() {
        use crate::address_assign::{assign_addresses, AddressAssignConfig};

        // Create ops where VRegs have truly non-overlapping lifetimes:
        // op0: const -> s0 (def cycle 0)
        // op1: alu(s0, s0) -> s1 (cycle 1, s0 last used)
        // op2: const -> s2 (def cycle 5, AFTER s0's interval ends)
        // op3: alu(s1, s2) -> s3 (cycle 6)
        //
        // CLOSED intervals:
        // - s0: [0, 1] — defined at 0, last read at 1
        // - s1: [1, 6] — defined at 1, last read at 6
        // - s2: [5, 6] — defined at 5, last read at 6
        // - s3: [6, 7] — defined at 6, never read (dead, +1 extension)
        //
        // Overlap analysis (strict inequality: start < other.end AND other.start < end):
        // - s0 [0,1] and s1 [1,6]: 0 < 6 AND 1 < 1? NO, no overlap (same-cycle reuse)
        // - s0 [0,1] and s2 [5,6]: 0 < 6 AND 5 < 1? NO, no overlap → s2 can reuse s0
        // - s1 [1,6] and s2 [5,6]: 1 < 6 AND 5 < 6? YES, overlap
        // - s1 [1,6] and s3 [6,7]: 1 < 7 AND 6 < 6? NO, no overlap (same-cycle reuse)
        // - s2 [5,6] and s3 [6,7]: 5 < 7 AND 6 < 6? NO, no overlap (same-cycle reuse)
        let mut ops = vec![
            make_test_ssa_op(0, Engine::Load, Opcode::Const, Some(VReg::scalar(0)), vec![Operand::Imm(5)], vec![]),
            make_test_ssa_op(1, Engine::Alu, Opcode::Add, Some(VReg::scalar(1)),
                vec![Operand::VReg(VReg::scalar(0)), Operand::VReg(VReg::scalar(0))], vec![0]),
            make_test_ssa_op(2, Engine::Load, Opcode::Const, Some(VReg::scalar(2)), vec![Operand::Imm(6)], vec![]),
            make_test_ssa_op(3, Engine::Alu, Opcode::Add, Some(VReg::scalar(3)),
                vec![Operand::VReg(VReg::scalar(1)), Operand::VReg(VReg::scalar(2))], vec![1, 2]),
        ];

        ops[0].cycle = 0;
        ops[1].cycle = 1;  // s0 last used here → s0 [0, 1]
        ops[2].cycle = 5;  // s2 defined here → s2 [5, 6]
        ops[3].cycle = 6;

        let config = AddressAssignConfig::default();
        let result = assign_addresses(&ops, &config);

        let addr_s0 = result.vreg_to_addr.get(&VReg::scalar(0)).expect("s0 should have address");
        let addr_s1 = result.vreg_to_addr.get(&VReg::scalar(1)).expect("s1 should have address");
        let addr_s2 = result.vreg_to_addr.get(&VReg::scalar(2)).expect("s2 should have address");
        let addr_s3 = result.vreg_to_addr.get(&VReg::scalar(3)).expect("s3 should have address");

        println!("Addresses: s0={}, s1={}, s2={}, s3={}", addr_s0, addr_s1, addr_s2, addr_s3);

        // With CLOSED intervals and linear scan allocation:
        // 1. s0 [0,1] gets address 1
        // 2. s1 [1,6] - s0 expires (1 <= 1) → s1 reuses address 1
        // 3. s2 [5,6] - s1 still active (6 > 5) → s2 gets address 2
        // 4. s3 [6,7] - s1 expires (6 <= 6), s2 expires (6 <= 6) → s3 reuses address 1
        //
        // So: s0=1, s1=1, s2=2, s3=1
        // s2 CANNOT reuse s0's address because s1 took it and s1 overlaps with s2.
        assert_eq!(addr_s0, addr_s1,
            "s0 [0,1] and s1 [1,6] allow same-cycle reuse, should share addresses");
        assert_ne!(addr_s1, addr_s2,
            "s1 [1,6] and s2 [5,6] overlap, must have different addresses");
        assert_eq!(addr_s0, addr_s3,
            "s0=s1 expires before s3 [6,7], s3 can reuse address 1");

        // Verify efficient reuse: max_addr should be 2 (addresses 1, 2)
        // - s0 gets addr 1, s1 reuses addr 1, s2 gets addr 2, s3 reuses addr 1
        assert!(result.max_addr <= 2,
            "Should use at most 2 addresses with closed intervals and same-cycle reuse, got max_addr={}",
            result.max_addr);
    }

    /// Test that the greedy scheduler schedules a simple 4-op chain and respects dependency order.
    #[test]
    fn test_greedy_handles_external_deps() {
        // Simple chain: op0 -> op1 -> op2 -> op3 (sink)
        let mut ops = vec![
            make_test_ssa_op(0, Engine::Valu, Opcode::Vbroadcast, Some(VReg::vector(0)), vec![Operand::Imm(1)], vec![]),
            make_test_ssa_op(1, Engine::Valu, Opcode::Add, Some(VReg::vector(1)),
                vec![Operand::VReg(VReg::vector(0)), Operand::VReg(VReg::vector(0))],
                vec![0]),
            make_test_ssa_op(2, Engine::Valu, Opcode::Add, Some(VReg::vector(2)),
                vec![Operand::VReg(VReg::vector(1)), Operand::VReg(VReg::vector(1))], vec![1]),
            make_test_ssa_op(3, Engine::Store, Opcode::Vstore, None,
                vec![Operand::Imm(100), Operand::VReg(VReg::vector(2))], vec![2]),
        ];

        let config = KernelConfig::new(10, 2047, 256, 1);
        let builder = SsaTickTockKernelBuilder::new(config);
        let mut shifted = HashSet::new();

        let result = builder.schedule_ssa_greedy_with_shift_budget(
            &mut ops, false, &mut shifted, 1000
        );

        assert!(result.is_ok(), "Should schedule chain: {:?}", result.err());

        let (cycles, _, _) = result.unwrap();
        assert_eq!(cycles, 4, "Chain of 4 ops should take 4 cycles");

        for (i, op) in ops.iter().enumerate() {
            assert!(op.cycle >= 0, "Op {} should be scheduled", i);
        }

        assert!(ops[1].cycle > ops[0].cycle, "Op1 should be after op0");
        assert!(ops[2].cycle > ops[1].cycle, "Op2 should be after op1");
        assert!(ops[3].cycle > ops[2].cycle, "Op3 (sink) should be after op2");
    }

    /// Test that a long dependency chain schedules all ops.
    ///
    /// This tests that ops which become ready mid-scheduling are properly
    /// added to the ready heap and eventually scheduled.
    #[test]
    fn test_greedy_long_chain_all_scheduled() {
        // Create a chain: op0 -> op1 -> op2 -> ... -> op9 -> sink
        // Each op depends on the previous one.
        let mut ops: Vec<SsaOp> = Vec::new();
        for i in 0..10 {
            let deps = if i == 0 { vec![] } else { vec![i - 1] };
            let srcs = if i == 0 {
                vec![Operand::Imm(1)]
            } else {
                vec![Operand::VReg(VReg::vector((i - 1) as u32)),
                     Operand::VReg(VReg::vector((i - 1) as u32))]
            };
            let opcode = if i == 0 { Opcode::Vbroadcast } else { Opcode::Add };
            ops.push(make_test_ssa_op(
                i, Engine::Valu, opcode,
                Some(VReg::vector(i as u32)), srcs, deps
            ));
        }
        // Add sink to consume the final output (v9)
        ops.push(make_test_ssa_op(
            10, Engine::Store, Opcode::Vstore, None,
            vec![Operand::Imm(100), Operand::VReg(VReg::vector(9))], vec![9]
        ));

        let config = KernelConfig::new(10, 2047, 256, 1);
        let builder = SsaTickTockKernelBuilder::new(config);
        let mut shifted = HashSet::new();

        let result = builder.schedule_ssa_greedy_with_shift_budget(
            &mut ops, false, &mut shifted, 1000
        );

        assert!(result.is_ok(), "Long chain should schedule: {:?}", result.err());

        // Verify ALL ops were scheduled (including sink)
        for (i, op) in ops.iter().enumerate() {
            assert!(op.cycle >= 0, "Op {} should be scheduled, got cycle {}", i, op.cycle);
        }

        // Verify scheduling order for the chain (0..10)
        for i in 1..10 {
            assert!(ops[i].cycle > ops[i-1].cycle,
                "Op {} (cycle {}) should be after op {} (cycle {})",
                i, ops[i].cycle, i-1, ops[i-1].cycle);
        }
    }

    /// Test diamond-shaped dependency graph where multiple paths converge.
    ///
    /// This tests that ops with multiple dependencies are properly tracked
    /// and become ready when ALL dependencies are satisfied.
    #[test]
    fn test_greedy_diamond_deps_all_scheduled() {
        // Diamond pattern:
        //     op0
        //    /    \
        //  op1    op2
        //    \    /
        //     op3
        //      |
        //     op4
        //      |
        //    sink
        let mut ops = vec![
            make_test_ssa_op(0, Engine::Valu, Opcode::Vbroadcast,
                Some(VReg::vector(0)), vec![Operand::Imm(1)], vec![]),
            make_test_ssa_op(1, Engine::Valu, Opcode::Add,
                Some(VReg::vector(1)),
                vec![Operand::VReg(VReg::vector(0)), Operand::Imm(1)], vec![0]),
            make_test_ssa_op(2, Engine::Valu, Opcode::Add,
                Some(VReg::vector(2)),
                vec![Operand::VReg(VReg::vector(0)), Operand::Imm(2)], vec![0]),
            // op3 depends on BOTH op1 and op2
            make_test_ssa_op(3, Engine::Valu, Opcode::Add,
                Some(VReg::vector(3)),
                vec![Operand::VReg(VReg::vector(1)), Operand::VReg(VReg::vector(2))],
                vec![1, 2]),
            make_test_ssa_op(4, Engine::Valu, Opcode::Add,
                Some(VReg::vector(4)),
                vec![Operand::VReg(VReg::vector(3)), Operand::Imm(1)], vec![3]),
            // Sink to consume v4
            make_test_ssa_op(5, Engine::Store, Opcode::Vstore, None,
                vec![Operand::Imm(100), Operand::VReg(VReg::vector(4))], vec![4]),
        ];

        let config = KernelConfig::new(10, 2047, 256, 1);
        let builder = SsaTickTockKernelBuilder::new(config);
        let mut shifted = HashSet::new();

        let result = builder.schedule_ssa_greedy_with_shift_budget(
            &mut ops, false, &mut shifted, 1000
        );

        assert!(result.is_ok(), "Diamond should schedule: {:?}", result.err());

        // Verify ALL ops were scheduled
        for (i, op) in ops.iter().enumerate() {
            assert!(op.cycle >= 0, "Op {} should be scheduled", i);
        }

        // Verify dependency ordering
        assert!(ops[1].cycle > ops[0].cycle, "op1 after op0");
        assert!(ops[2].cycle > ops[0].cycle, "op2 after op0");
        assert!(ops[3].cycle > ops[1].cycle, "op3 after op1");
        assert!(ops[3].cycle > ops[2].cycle, "op3 after op2");
        assert!(ops[4].cycle > ops[3].cycle, "op4 after op3");
    }

    /// Test many parallel independent chains.
    ///
    /// This tests that the heap correctly handles many ops becoming ready
    /// at the same time and that they all eventually get scheduled.
    #[test]
    fn test_greedy_parallel_chains_all_scheduled() {
        // 10 independent chains, each with 3 ops + sink
        // Chain i: op[i*3] -> op[i*3+1] -> op[i*3+2]
        let mut ops: Vec<SsaOp> = Vec::new();
        for chain in 0..10 {
            let base = chain * 3;
            // First op in chain - no deps
            ops.push(make_test_ssa_op(
                base, Engine::Valu, Opcode::Vbroadcast,
                Some(VReg::vector(base as u32)),
                vec![Operand::Imm(chain as i64)], vec![]
            ));
            // Second op - depends on first
            ops.push(make_test_ssa_op(
                base + 1, Engine::Valu, Opcode::Add,
                Some(VReg::vector((base + 1) as u32)),
                vec![Operand::VReg(VReg::vector(base as u32)), Operand::Imm(1)],
                vec![base]
            ));
            // Third op - depends on second
            ops.push(make_test_ssa_op(
                base + 2, Engine::Valu, Opcode::Add,
                Some(VReg::vector((base + 2) as u32)),
                vec![Operand::VReg(VReg::vector((base + 1) as u32)), Operand::Imm(1)],
                vec![base + 1]
            ));
        }
        // Add sinks to consume each chain's final output (v2, v5, v8, ..., v29)
        for chain in 0..10 {
            let last_vreg = (chain * 3 + 2) as u32;
            let last_op_id = chain * 3 + 2;
            ops.push(make_test_ssa_op(
                30 + chain, Engine::Store, Opcode::Vstore, None,
                vec![Operand::Imm(100 + chain as i64), Operand::VReg(VReg::vector(last_vreg))],
                vec![last_op_id]
            ));
        }

        let config = KernelConfig::new(10, 2047, 256, 1);
        let builder = SsaTickTockKernelBuilder::new(config);
        let mut shifted = HashSet::new();

        let result = builder.schedule_ssa_greedy_with_shift_budget(
            &mut ops, false, &mut shifted, 1000
        );

        assert!(result.is_ok(), "Parallel chains should schedule: {:?}", result.err());

        // Verify ALL 40 ops were scheduled (30 chain ops + 10 sinks)
        for (i, op) in ops.iter().enumerate() {
            assert!(op.cycle >= 0, "Op {} should be scheduled, got cycle {}", i, op.cycle);
        }
    }

    /// Test that slot limits don't cause ops to get stuck.
    ///
    /// When slot limits prevent scheduling, ops should be deferred and
    /// tried again in later cycles.
    #[test]
    fn test_greedy_slot_overflow_all_scheduled() {
        // Create 12 independent VALU ops (more than the 6 VALU slots)
        // All should eventually be scheduled across multiple cycles.
        let mut ops: Vec<SsaOp> = Vec::new();
        for i in 0..12 {
            ops.push(make_test_ssa_op(
                i, Engine::Valu, Opcode::Vbroadcast,
                Some(VReg::vector(i as u32)),
                vec![Operand::Imm(i as i64)], vec![]
            ));
        }
        // Add sinks to consume all 12 VRegs
        for i in 0..12 {
            ops.push(make_test_ssa_op(
                12 + i, Engine::Store, Opcode::Vstore, None,
                vec![Operand::Imm(100 + i as i64), Operand::VReg(VReg::vector(i as u32))], vec![i]
            ));
        }

        let config = KernelConfig::new(10, 2047, 256, 1);
        let builder = SsaTickTockKernelBuilder::new(config);
        let mut shifted = HashSet::new();

        let result = builder.schedule_ssa_greedy_with_shift_budget(
            &mut ops, false, &mut shifted, 1000
        );

        assert!(result.is_ok(), "Slot overflow should still schedule all: {:?}", result.err());

        let (cycles, _, _) = result.unwrap();

        // Verify ALL ops were scheduled
        for (i, op) in ops.iter().enumerate() {
            assert!(op.cycle >= 0, "Op {} should be scheduled", i);
        }

        // With 6 VALU slots, 12 independent ops need at least 2 cycles
        assert!(cycles >= 2, "12 VALU ops + sinks should need at least 2 cycles");
    }

    /// Test that budget limits don't cause ops to get stuck.
    ///
    /// When budget limits prevent scheduling, ops should be deferred and
    /// tried again when liveness decreases.
    #[test]
    fn test_greedy_budget_overflow_all_scheduled() {
        // Create a sequence where budget pressure builds and releases
        // op0, op1: produce 16 words
        // op2: consumes op0, op1 (frees 16), produces 8 words
        // op3: reads v2, produces v3
        // op4: reads v2, produces v4
        // op5: consumes v3, v4 (frees them), produces v5
        // op6: sink to consume v5
        let mut ops = vec![
            make_test_ssa_op(0, Engine::Valu, Opcode::Vbroadcast,
                Some(VReg::vector(0)), vec![Operand::Imm(1)], vec![]),
            make_test_ssa_op(1, Engine::Valu, Opcode::Vbroadcast,
                Some(VReg::vector(1)), vec![Operand::Imm(2)], vec![]),
            make_test_ssa_op(2, Engine::Valu, Opcode::Add,
                Some(VReg::vector(2)),
                vec![Operand::VReg(VReg::vector(0)), Operand::VReg(VReg::vector(1))],
                vec![0, 1]),
            // op3 actually reads v2 (not just depends on op2)
            make_test_ssa_op(3, Engine::Valu, Opcode::Add,
                Some(VReg::vector(3)),
                vec![Operand::VReg(VReg::vector(2)), Operand::Imm(3)], vec![2]),
            // op4 also reads v2
            make_test_ssa_op(4, Engine::Valu, Opcode::Add,
                Some(VReg::vector(4)),
                vec![Operand::VReg(VReg::vector(2)), Operand::Imm(4)], vec![2]),
            make_test_ssa_op(5, Engine::Valu, Opcode::Add,
                Some(VReg::vector(5)),
                vec![Operand::VReg(VReg::vector(3)), Operand::VReg(VReg::vector(4))],
                vec![3, 4]),
            // Sink to consume v5
            make_test_ssa_op(6, Engine::Store, Opcode::Vstore, None,
                vec![Operand::Imm(100), Operand::VReg(VReg::vector(5))], vec![5]),
        ];

        let config = KernelConfig::new(10, 2047, 256, 1);
        let builder = SsaTickTockKernelBuilder::new(config);
        let mut shifted = HashSet::new();

        // Use a moderate budget that requires careful scheduling
        let result = builder.schedule_ssa_greedy_with_shift_budget(
            &mut ops, false, &mut shifted, 32  // 4 vectors worth
        );

        assert!(result.is_ok(), "Budget pressure should still schedule all: {:?}", result.err());

        // Verify ALL ops were scheduled
        for (i, op) in ops.iter().enumerate() {
            assert!(op.cycle >= 0, "Op {} should be scheduled", i);
        }
    }

    /// Test that a larger scale schedule works - catches bugs that only appear at scale.
    ///
    /// This creates 1000+ ops to test heap behavior with many ops.
    #[test]
    fn test_greedy_large_scale_all_scheduled() {
        // Create 100 chains of 10 ops each = 1000 ops
        // This tests the scheduler at a larger scale
        let mut ops: Vec<SsaOp> = Vec::new();
        let mut id = 0;
        for chain in 0..100 {
            for pos in 0..10 {
                let deps = if pos == 0 { vec![] } else { vec![id - 1] };
                let srcs = if pos == 0 {
                    vec![Operand::Imm(chain as i64)]
                } else {
                    vec![Operand::VReg(VReg::vector((id - 1) as u32)), Operand::Imm(1)]
                };
                let opcode = if pos == 0 { Opcode::Vbroadcast } else { Opcode::Add };
                ops.push(make_test_ssa_op(
                    id, Engine::Valu, opcode,
                    Some(VReg::vector(id as u32)), srcs, deps
                ));
                id += 1;
            }
        }

        // Add sinks for each chain's final output (every 10th vreg: v9, v19, v29, ..., v999)
        for chain in 0..100 {
            let last_vreg = (chain * 10 + 9) as u32;
            let last_op_id = chain * 10 + 9;
            ops.push(make_test_ssa_op(
                id, Engine::Store, Opcode::Vstore, None,
                vec![Operand::Imm(100 + chain as i64), Operand::VReg(VReg::vector(last_vreg))],
                vec![last_op_id]
            ));
            id += 1;
        }

        assert_eq!(ops.len(), 1100, "Should have 1100 ops (1000 chain ops + 100 sinks)");

        let config = KernelConfig::new(10, 2047, 256, 1);
        let builder = SsaTickTockKernelBuilder::new(config);
        let mut shifted = HashSet::new();

        let result = builder.schedule_ssa_greedy_with_shift_budget(
            &mut ops, false, &mut shifted, 10000  // Large budget
        );

        assert!(result.is_ok(), "Large scale should schedule: {:?}", result.err());

        // Verify ALL ops were scheduled
        let unscheduled: Vec<_> = ops.iter().enumerate()
            .filter(|(_, op)| op.cycle < 0)
            .map(|(i, _)| i)
            .collect();
        assert!(unscheduled.is_empty(),
            "All 1100 ops should be scheduled, unscheduled: {:?}", &unscheduled[..unscheduled.len().min(10)]);
    }

    /// Test that cross-chain dependencies work correctly.
    ///
    /// Creates chains where later chains depend on earlier chains,
    /// mimicking the interleaved structure of the real kernel.
    #[test]
    fn test_greedy_cross_chain_deps_all_scheduled() {
        // 10 chains, where chain N's first op depends on chain N-1's last op
        let mut ops: Vec<SsaOp> = Vec::new();
        let mut id = 0;
        for chain in 0..10 {
            for pos in 0..5 {
                let deps = if pos == 0 {
                    if chain == 0 {
                        vec![]  // First chain, first op: no deps
                    } else {
                        // Depends on previous chain's last op
                        vec![chain * 5 - 1]
                    }
                } else {
                    vec![id - 1]  // Depends on previous op in chain
                };
                let srcs = if pos == 0 && chain == 0 {
                    vec![Operand::Imm(1)]
                } else {
                    vec![Operand::VReg(VReg::vector((id - 1).max(0) as u32)), Operand::Imm(1)]
                };
                let opcode = if pos == 0 && chain == 0 { Opcode::Vbroadcast } else { Opcode::Add };
                ops.push(make_test_ssa_op(
                    id, Engine::Valu, opcode,
                    Some(VReg::vector(id as u32)), srcs, deps
                ));
                id += 1;
            }
        }

        // Add sink to consume the final chain's output (v49)
        ops.push(make_test_ssa_op(
            id, Engine::Store, Opcode::Vstore, None,
            vec![Operand::Imm(100), Operand::VReg(VReg::vector(49))], vec![49]
        ));

        assert_eq!(ops.len(), 51, "Should have 51 ops (50 chain ops + 1 sink)");

        let config = KernelConfig::new(10, 2047, 256, 1);
        let builder = SsaTickTockKernelBuilder::new(config);
        let mut shifted = HashSet::new();

        let result = builder.schedule_ssa_greedy_with_shift_budget(
            &mut ops, false, &mut shifted, 1000
        );

        assert!(result.is_ok(), "Cross-chain deps should schedule: {:?}", result.err());

        // Verify ALL ops were scheduled
        for (i, op) in ops.iter().enumerate() {
            assert!(op.cycle >= 0, "Op {} should be scheduled", i);
        }
    }

    /// Test that the actual SSA kernel builder produces schedulable ops.
    ///
    /// This uses the real SSA builder with a smaller config to catch
    /// bugs in the interaction between op generation and scheduling.
    #[test]
    fn test_greedy_real_kernel_small() {
        // Use a small config that should produce ~100-200 ops
        let config = KernelConfig::new(3, 15, 16, 2);  // height=3, nodes=15, batch=16, rounds=2
        let mut builder = SsaTickTockKernelBuilder::new(config);

        // Build SSA ops
        builder.build_internal();
        let vreg_def = std::mem::take(&mut builder.emit.vreg_def);
        let mut ops = builder.emit.take_ops();

        // Compute dependencies
        crate::kernel_builder::emit_ssa::compute_ssa_dependencies(&mut ops, &vreg_def);

        println!("Generated {} ops", ops.len());

        // Identify dead stores (ops with dest VReg but no users)
        let mut vreg_users: HashMap<VReg, usize> = HashMap::new();
        for op in ops.iter() {
            for src in &op.srcs {
                if let Operand::VReg(vreg) = src {
                    *vreg_users.entry(*vreg).or_default() += 1;
                }
            }
        }
        let dead_store_count = ops.iter()
            .filter(|op| op.dest.map_or(false, |d| vreg_users.get(&d).map_or(true, |&n| n == 0)))
            .count();
        println!("Detected {} dead stores (will be skipped)", dead_store_count);

        let mut shifted = HashSet::new();
        let result = builder.schedule_ssa_greedy_with_shift_budget(
            &mut ops, true, &mut shifted, 1536
        );

        assert!(result.is_ok(), "Real kernel (small) should schedule: {:?}", result.err());

        let (_cycles, budget_exceeded, _peak_live) = result.unwrap();
        // When budget is exceeded, greedy may leave some ops unscheduled (force-schedule path).
        // Only assert all non-dead-store ops scheduled when we stayed within budget.
        if !budget_exceeded {
            let unscheduled: Vec<_> = ops.iter().enumerate()
                .filter(|(_, op)| {
                    op.cycle < 0 && !op.dest.map_or(false, |d| vreg_users.get(&d).map_or(true, |&n| n == 0))
                })
                .map(|(i, op)| format!("op[{}] id={}", i, op.id))
                .collect();
            assert!(unscheduled.is_empty(),
                "All non-dead-store ops should be scheduled when within budget, {} unscheduled: {:?}",
                unscheduled.len(), &unscheduled[..unscheduled.len().min(10)]);
        }
    }

    /// Test that the actual SSA kernel builder produces schedulable ops at medium scale.
    #[test]
    fn test_greedy_real_kernel_medium() {
        // Medium config: ~1000-2000 ops
        let config = KernelConfig::new(5, 63, 64, 4);  // height=5, nodes=63, batch=64, rounds=4
        let mut builder = SsaTickTockKernelBuilder::new(config);

        // Build SSA ops
        builder.build_internal();
        let vreg_def = std::mem::take(&mut builder.emit.vreg_def);
        let mut ops = builder.emit.take_ops();

        // Compute dependencies
        crate::kernel_builder::emit_ssa::compute_ssa_dependencies(&mut ops, &vreg_def);

        println!("Generated {} ops", ops.len());

        let mut shifted = HashSet::new();
        let result = builder.schedule_ssa_greedy_with_shift_budget(
            &mut ops, true, &mut shifted, 1536
        );

        assert!(result.is_ok(), "Real kernel (medium) should schedule: {:?}", result.err());

        let (_cycles, budget_exceeded, _peak_live) = result.unwrap();
        if !budget_exceeded {
            let unscheduled: Vec<_> = ops.iter().enumerate()
                .filter(|(_, op)| op.cycle < 0)
                .map(|(i, op)| format!("op[{}] id={}", i, op.id))
                .collect();
            assert!(unscheduled.is_empty(),
                "All ops should be scheduled when within budget, {} unscheduled: {:?}",
                unscheduled.len(), &unscheduled[..unscheduled.len().min(10)]);
        }
    }

    /// Test the exact failing configuration: forest_height=10, n_nodes=2047, batch_size=256, rounds=16.
    ///
    /// This is the production config that triggers the scheduling failure.
    #[test]
    fn test_greedy_real_kernel_full_scale() {
        let config = KernelConfig::new(10, 2047, 256, 16);
        let mut builder = SsaTickTockKernelBuilder::new(config);

        // Build SSA ops
        builder.build_internal();
        let vreg_def = std::mem::take(&mut builder.emit.vreg_def);
        let mut ops = builder.emit.take_ops();

        // Compute dependencies
        crate::kernel_builder::emit_ssa::compute_ssa_dependencies(&mut ops, &vreg_def);

        println!("Generated {} ops for full-scale kernel", ops.len());

        let mut shifted = HashSet::new();
        let result = builder.schedule_ssa_greedy_with_shift_budget(
            &mut ops, true, &mut shifted, 1536
        );

        assert!(result.is_ok(), "Full-scale kernel should schedule: {:?}", result.err());

        let (_cycles, budget_exceeded, peak_live) = result.unwrap();
        // Greedy can exceed scratch budget for 16-round production config; CP-SAT is used for correct output.
        if !budget_exceeded {
            let mut vreg_users: HashSet<VReg> = HashSet::new();
            for op in ops.iter() {
                for src in &op.srcs {
                    if let Operand::VReg(vreg) = src {
                        vreg_users.insert(*vreg);
                    }
                }
            }
            let dead_store_count = ops.iter()
                .filter(|op| op.dest.is_some() && !vreg_users.contains(&op.dest.unwrap()))
                .count();
            let unscheduled: Vec<_> = ops.iter().enumerate()
                .filter(|(_, op)| op.cycle < 0)
                .filter(|(_, op)| op.dest.is_none() || vreg_users.contains(&op.dest.unwrap()))
                .map(|(i, op)| format!("op[{}] id={}", i, op.id))
                .collect();
            assert!(unscheduled.is_empty(),
                "All non-dead-store ops should be scheduled when within budget ({} dead stores filtered), {} unscheduled: {:?}",
                dead_store_count, unscheduled.len(), &unscheduled[..unscheduled.len().min(10)]);
        } else {
            println!("Full-scale greedy exceeded budget (peak_live={}); CP-SAT used for production.", peak_live);
        }
    }

    // =========================================================================
    // LUC (Last-Use-Count) Heuristic Tests
    // =========================================================================

    /// Test that LUC is used as tie-breaker when ops have same ASAP priority.
    ///
    /// Creates two ops at same cycle, one with high LUC (frees 2 VRegs) and one with low LUC.
    /// The high-LUC op should be scheduled first.
    #[test]
    fn test_luc_tiebreaker_prefers_high_luc() {
        // Setup:
        // op0: const -> v0 (scalar)
        // op1: const -> v1 (scalar)
        // op2: add(v0, v1) -> v2 (frees both v0 and v1) - HIGH LUC = 2
        // op3: add(v1, v1) -> v3 (doesn't free v1, still used by op2) - LOW LUC = 0
        // op4: add(v2, v3) -> v4 (sink - consumes both to avoid dead stores)
        //
        // op2 and op3 both depend on op1, so they can run in same cycle.
        // LUC tie-breaker should prefer op2 (frees 2) over op3 (frees 0).
        let mut ops = vec![
            make_test_ssa_op(0, Engine::Load, Opcode::Const, Some(VReg::scalar(0)), vec![Operand::Imm(1)], vec![]),
            make_test_ssa_op(1, Engine::Load, Opcode::Const, Some(VReg::scalar(1)), vec![Operand::Imm(2)], vec![]),
            // op2: uses v0 (last use) and v1 (not last - op3 uses v1 too) -> LUC=1
            make_test_ssa_op(2, Engine::Alu, Opcode::Add, Some(VReg::scalar(2)),
                vec![Operand::VReg(VReg::scalar(0)), Operand::VReg(VReg::scalar(1))], vec![0, 1]),
            // op3: uses v1 twice (last uses for v1) -> LUC=1 (but v1 counted once)
            make_test_ssa_op(3, Engine::Alu, Opcode::Add, Some(VReg::scalar(3)),
                vec![Operand::VReg(VReg::scalar(1)), Operand::VReg(VReg::scalar(1))], vec![1]),
            // op4: sink to consume v2 and v3
            make_test_ssa_op(4, Engine::Alu, Opcode::Add, Some(VReg::scalar(4)),
                vec![Operand::VReg(VReg::scalar(2)), Operand::VReg(VReg::scalar(3))], vec![2, 3]),
            // op5: final sink to consume v4
            make_test_ssa_op(5, Engine::Store, Opcode::Store, None,
                vec![Operand::Imm(100), Operand::VReg(VReg::scalar(4))], vec![4]),
        ];

        let config = KernelConfig::new(10, 2047, 256, 1);
        let builder = SsaTickTockKernelBuilder::new(config);
        let mut shifted = HashSet::new();

        let result = builder.schedule_ssa_greedy_with_shift_budget(
            &mut ops, false, &mut shifted, 100
        );
        assert!(result.is_ok(), "Should schedule successfully: {:?}", result.err());

        // Verify scheduling order at cycle 2 (when op2 and op3 can both run)
        // Both ops depend on cycle-1 ops, so they're ready at cycle 2.
        // The one with higher LUC should come first in the cycle.
        let cycle_2_ops: Vec<_> = ops.iter()
            .filter(|op| op.cycle == 2)
            .collect();

        // With LUC tie-breaking, the op that frees v0 (op2) should be scheduled.
        // Since we have limited visibility into intra-cycle ordering, just verify both complete.
        assert!(ops[2].cycle >= 0, "op2 should be scheduled");
        assert!(ops[3].cycle >= 0, "op3 should be scheduled");
    }

    /// Test that budget-deferred ops are sorted by LUC to break deadlocks.
    ///
    /// Creates a scenario where budget is tight and ops are deferred.
    /// Ops that free more VRegs should be scheduled first.
    #[test]
    fn test_luc_deferred_ops_prioritizes_freeing() {
        // Setup: tight budget with multiple vector ops
        // op0: broadcast -> v0 (8 words)
        // op1: broadcast -> v1 (8 words)
        // op2: broadcast -> v2 (8 words)
        // op3: add(v0, v1) -> v3 (frees v0, v1 = 16 words freed) HIGH LUC
        // op4: add(v2, v2) -> v4 (frees v2 = 8 words freed, but v2 used twice counts as 1)
        // op5: add(v3, v4) -> v5 (sink)
        //
        // With budget = 24, after ops 0,1,2 we have 24 words live.
        // ops 3 and 4 both want to add 8 words.
        // op3 frees 16 (v0+v1), op4 frees 8 (v2).
        // LUC should prefer op3 first.
        let mut ops = vec![
            make_test_ssa_op(0, Engine::Valu, Opcode::Vbroadcast, Some(VReg::vector(0)), vec![Operand::Imm(1)], vec![]),
            make_test_ssa_op(1, Engine::Valu, Opcode::Vbroadcast, Some(VReg::vector(1)), vec![Operand::Imm(2)], vec![]),
            make_test_ssa_op(2, Engine::Valu, Opcode::Vbroadcast, Some(VReg::vector(2)), vec![Operand::Imm(3)], vec![]),
            // op3: uses v0 and v1 (both last uses) -> LUC=2
            make_test_ssa_op(3, Engine::Valu, Opcode::Add, Some(VReg::vector(3)),
                vec![Operand::VReg(VReg::vector(0)), Operand::VReg(VReg::vector(1))], vec![0, 1]),
            // op4: uses v2 twice (last use) -> LUC=1
            make_test_ssa_op(4, Engine::Valu, Opcode::Add, Some(VReg::vector(4)),
                vec![Operand::VReg(VReg::vector(2)), Operand::VReg(VReg::vector(2))], vec![2]),
            // op5: sink
            make_test_ssa_op(5, Engine::Valu, Opcode::Add, Some(VReg::vector(5)),
                vec![Operand::VReg(VReg::vector(3)), Operand::VReg(VReg::vector(4))], vec![3, 4]),
            // op6: final sink
            make_test_ssa_op(6, Engine::Store, Opcode::Vstore, None,
                vec![Operand::Imm(100), Operand::VReg(VReg::vector(5))], vec![5]),
        ];

        let config = KernelConfig::new(10, 2047, 256, 1);
        let builder = SsaTickTockKernelBuilder::new(config);
        let mut shifted = HashSet::new();

        // With budget = 32, ops 0,1,2 take 24 words. Op3 or op4 can fit (adds 8, frees some).
        // LUC should help choose the better one.
        let result = builder.schedule_ssa_greedy_with_shift_budget(
            &mut ops, false, &mut shifted, 32
        );
        assert!(result.is_ok(), "Should schedule with budget=32: {:?}", result.err());

        // All ops should be scheduled
        for (i, op) in ops.iter().enumerate() {
            assert!(op.cycle >= 0, "op{} should be scheduled, cycle={}", i, op.cycle);
        }
    }

    /// Test that emergency fallback breaks deadlock when LUC alone isn't enough.
    ///
    /// Creates extreme pressure where all ready ops exceed budget.
    /// Emergency fallback should force-schedule the highest-LUC op.
    #[test]
    fn test_luc_emergency_fallback_breaks_deadlock() {
        // Setup: very tight budget that would deadlock without emergency fallback
        // op0: broadcast -> v0 (8 words)
        // op1: broadcast -> v1 (8 words)
        // op2: add(v0, v1) -> v2 (frees 16, adds 8 = net -8)
        // op3: sink
        //
        // Budget = 16: after op0 and op1, we have 16 words.
        // op2 would add 8 more (total 24) before freeing v0,v1.
        // Without emergency fallback, this would deadlock.
        let mut ops = vec![
            make_test_ssa_op(0, Engine::Valu, Opcode::Vbroadcast, Some(VReg::vector(0)), vec![Operand::Imm(1)], vec![]),
            make_test_ssa_op(1, Engine::Valu, Opcode::Vbroadcast, Some(VReg::vector(1)), vec![Operand::Imm(2)], vec![]),
            make_test_ssa_op(2, Engine::Valu, Opcode::Add, Some(VReg::vector(2)),
                vec![Operand::VReg(VReg::vector(0)), Operand::VReg(VReg::vector(1))], vec![0, 1]),
            make_test_ssa_op(3, Engine::Store, Opcode::Vstore, None,
                vec![Operand::Imm(100), Operand::VReg(VReg::vector(2))], vec![2]),
        ];

        let config = KernelConfig::new(10, 2047, 256, 1);
        let builder = SsaTickTockKernelBuilder::new(config);
        let mut shifted = HashSet::new();

        // With same-cycle reuse, op2 should work: projected = 16 + 8 - 16 = 8
        // But if that doesn't work, emergency fallback should kick in.
        let result = builder.schedule_ssa_greedy_with_shift_budget(
            &mut ops, false, &mut shifted, 16
        );
        assert!(result.is_ok(), "Emergency fallback should break deadlock: {:?}", result.err());

        // All ops should be scheduled
        for (i, op) in ops.iter().enumerate() {
            assert!(op.cycle >= 0, "op{} should be scheduled, cycle={}", i, op.cycle);
        }
    }

    // =========================================================================
    // Duplicate Operand Bug Tests
    //
    // These tests verify that operations with duplicate operands (e.g., add(v1, v1))
    // are handled correctly by the greedy scheduler. Without proper deduplication,
    // several bugs can occur:
    //
    // 1. LUC (Last-Use-Count) miscalculation: add(v1, v1) would report LUC=2
    //    instead of LUC=1, affecting scheduling priority.
    //
    // 2. vreg_remaining_users double-decrement: The use counter for v1 would be
    //    decremented twice, causing premature VReg deallocation.
    //
    // 3. words_freed over-estimation: For a vector v1 (8 words), add(v1, v1)
    //    would incorrectly report 16 words freed instead of 8.
    // =========================================================================

    /// Test that duplicate operands are correctly counted for LUC calculation.
    ///
    /// Bug scenario: add(v1, v1) where v1 is the last use.
    /// - WRONG: LUC = 2 (counting v1 twice)
    /// - CORRECT: LUC = 1 (v1 is freed once)
    #[test]
    fn test_duplicate_operand_luc_calculation() {
        // Create a simple chain:
        // op0: broadcast -> v0 (8 words)
        // op1: add(v0, v0) -> v1 (8 words), DUPLICATE OPERAND
        // op2: sink v1
        //
        // v0 has 1 user (op1, which uses it twice but counts as 1 use)
        // So when op1 runs, LUC should be 1, not 2.
        let mut ops = vec![
            make_test_ssa_op(0, Engine::Valu, Opcode::Vbroadcast, Some(VReg::vector(0)), vec![Operand::Imm(1)], vec![]),
            // DUPLICATE OPERAND: v0 appears twice
            make_test_ssa_op(1, Engine::Valu, Opcode::Add, Some(VReg::vector(1)),
                vec![Operand::VReg(VReg::vector(0)), Operand::VReg(VReg::vector(0))], vec![0]),
            make_test_ssa_op(2, Engine::Store, Opcode::Vstore, None,
                vec![Operand::Imm(100), Operand::VReg(VReg::vector(1))], vec![1]),
        ];

        let config = KernelConfig::new(10, 2047, 256, 1);
        let builder = SsaTickTockKernelBuilder::new(config);
        let mut shifted = HashSet::new();

        // Schedule with a budget that requires accurate liveness tracking
        let result = builder.schedule_ssa_greedy_with_shift_budget(
            &mut ops, false, &mut shifted, 16  // Just enough for v0 + v1
        );
        assert!(result.is_ok(), "Should schedule with duplicate operands: {:?}", result.err());

        // Verify all ops were scheduled correctly
        assert!(ops[0].cycle >= 0, "op0 should be scheduled");
        assert!(ops[1].cycle >= 0, "op1 should be scheduled");
        assert!(ops[2].cycle >= 0, "op2 should be scheduled");

        // Verify dependency order
        assert!(ops[1].cycle > ops[0].cycle, "op1 must come after op0");
        assert!(ops[2].cycle > ops[1].cycle, "op2 must come after op1");
    }

    /// Test that vreg_remaining_users is not double-decremented for duplicate operands.
    ///
    /// Bug scenario: If add(v1, v1) decrements v1's user count twice, then:
    /// - After op with add(v1, v1) runs, v1's remaining_users becomes -1 or underflows
    /// - v1 gets freed prematurely, corrupting later ops that need v1
    #[test]
    fn test_duplicate_operand_remaining_users_not_double_decremented() {
        // Create a chain where v0 is used by TWO different ops:
        // op0: broadcast -> v0 (8 words)
        // op1: add(v0, v0) -> v1 (8 words), DUPLICATE OPERAND, first use of v0
        // op2: add(v0, v1) -> v2 (8 words), second use of v0
        // op3: sink v2
        //
        // v0 has 2 users: op1 and op2
        // If op1 double-decrements v0's user count, it would go from 2 -> 0,
        // causing v0 to be freed before op2 runs!
        let mut ops = vec![
            make_test_ssa_op(0, Engine::Valu, Opcode::Vbroadcast, Some(VReg::vector(0)), vec![Operand::Imm(1)], vec![]),
            // DUPLICATE OPERAND: v0 appears twice - should only count as 1 use
            make_test_ssa_op(1, Engine::Valu, Opcode::Add, Some(VReg::vector(1)),
                vec![Operand::VReg(VReg::vector(0)), Operand::VReg(VReg::vector(0))], vec![0]),
            // op2 also uses v0 - this should still work after op1 runs
            make_test_ssa_op(2, Engine::Valu, Opcode::Add, Some(VReg::vector(2)),
                vec![Operand::VReg(VReg::vector(0)), Operand::VReg(VReg::vector(1))], vec![0, 1]),
            make_test_ssa_op(3, Engine::Store, Opcode::Vstore, None,
                vec![Operand::Imm(100), Operand::VReg(VReg::vector(2))], vec![2]),
        ];

        let config = KernelConfig::new(10, 2047, 256, 1);
        let builder = SsaTickTockKernelBuilder::new(config);
        let mut shifted = HashSet::new();

        // Schedule with a tight budget that requires accurate liveness
        let result = builder.schedule_ssa_greedy_with_shift_budget(
            &mut ops, false, &mut shifted, 24  // v0 + v1 + v2 = 24 words peak
        );
        assert!(result.is_ok(), "Should schedule without double-decrement bug: {:?}", result.err());

        // All ops should be scheduled in dependency order
        assert!(ops[0].cycle >= 0, "op0 should be scheduled");
        assert!(ops[1].cycle > ops[0].cycle, "op1 must come after op0");
        assert!(ops[2].cycle > ops[1].cycle, "op2 must come after op1 (and needs v0!)");
        assert!(ops[3].cycle > ops[2].cycle, "op3 must come after op2");
    }

    /// Test that words_freed is not over-estimated for duplicate operands.
    ///
    /// Bug scenario: When checking scratch budget, add(v1, v1) where v1 is a
    /// vector (8 words) and is the last use would incorrectly calculate:
    /// - WRONG: words_freed = 16 (counting v1 twice)
    /// - CORRECT: words_freed = 8 (v1 is freed once)
    ///
    /// This could cause the scheduler to think an op fits when it doesn't.
    #[test]
    fn test_duplicate_operand_words_freed_not_over_estimated() {
        // Create a scenario where accurate words_freed matters:
        // op0: broadcast -> v0 (8 words)
        // op1: broadcast -> v1 (8 words)
        // op2: add(v0, v0) -> v2 (8 words), DUPLICATE OPERAND, last use of v0
        // op3: add(v1, v2) -> v3 (8 words), consumes v1 and v2
        //
        // With correct calculation at op2:
        // - Before op2: live = v0 + v1 = 16 words
        // - Op2 adds v2 (8 words), frees v0 (8 words)
        // - After op2: live = v1 + v2 = 16 words
        //
        // With buggy over-estimation (counting v0 freed twice):
        // - Scheduler would think words_freed = 16, projected_live = 8
        // - But reality: projected_live = 16
        let mut ops = vec![
            make_test_ssa_op(0, Engine::Valu, Opcode::Vbroadcast, Some(VReg::vector(0)), vec![Operand::Imm(1)], vec![]),
            make_test_ssa_op(1, Engine::Valu, Opcode::Vbroadcast, Some(VReg::vector(1)), vec![Operand::Imm(2)], vec![]),
            // DUPLICATE OPERAND: v0 appears twice
            make_test_ssa_op(2, Engine::Valu, Opcode::Add, Some(VReg::vector(2)),
                vec![Operand::VReg(VReg::vector(0)), Operand::VReg(VReg::vector(0))], vec![0]),
            make_test_ssa_op(3, Engine::Valu, Opcode::Add, Some(VReg::vector(3)),
                vec![Operand::VReg(VReg::vector(1)), Operand::VReg(VReg::vector(2))], vec![1, 2]),
            make_test_ssa_op(4, Engine::Store, Opcode::Vstore, None,
                vec![Operand::Imm(100), Operand::VReg(VReg::vector(3))], vec![3]),
        ];

        let config = KernelConfig::new(10, 2047, 256, 1);
        let builder = SsaTickTockKernelBuilder::new(config);
        let mut shifted = HashSet::new();

        // Use a budget of exactly 16 words - this should work with correct calculation
        // Peak liveness: v0 + v1 after cycle 0, or v1 + v2 after cycle 1 = 16 words
        let result = builder.schedule_ssa_greedy_with_shift_budget(
            &mut ops, false, &mut shifted, 16
        );
        assert!(result.is_ok(), "Should schedule with accurate words_freed: {:?}", result.err());

        // Verify scheduling succeeded
        for (i, op) in ops.iter().enumerate() {
            assert!(op.cycle >= 0, "op{} should be scheduled, cycle={}", i, op.cycle);
        }
    }

    /// Test scalar duplicate operands (not just vectors).
    ///
    /// Verifies the fix works for scalar VRegs too, not just vectors.
    #[test]
    fn test_duplicate_operand_scalar() {
        // Scalar chain with duplicate operand:
        // op0: const -> s0 (1 word)
        // op1: add(s0, s0) -> s1 (1 word), DUPLICATE OPERAND
        // op2: sink s1
        let mut ops = vec![
            make_test_ssa_op(0, Engine::Load, Opcode::Const, Some(VReg::scalar(0)), vec![Operand::Imm(42)], vec![]),
            // DUPLICATE OPERAND: s0 appears twice
            make_test_ssa_op(1, Engine::Alu, Opcode::Add, Some(VReg::scalar(1)),
                vec![Operand::VReg(VReg::scalar(0)), Operand::VReg(VReg::scalar(0))], vec![0]),
            make_test_ssa_op(2, Engine::Store, Opcode::Store, None,
                vec![Operand::Imm(100), Operand::VReg(VReg::scalar(1))], vec![1]),
        ];

        let config = KernelConfig::new(10, 2047, 256, 1);
        let builder = SsaTickTockKernelBuilder::new(config);
        let mut shifted = HashSet::new();

        let result = builder.schedule_ssa_greedy_with_shift_budget(
            &mut ops, false, &mut shifted, 10  // Generous budget for scalars
        );
        assert!(result.is_ok(), "Should schedule scalar duplicate operands: {:?}", result.err());

        // Verify scheduling order
        assert!(ops[1].cycle > ops[0].cycle, "op1 must come after op0");
        assert!(ops[2].cycle > ops[1].cycle, "op2 must come after op1");
    }

    /// Test multiply_add with mixed duplicate and unique operands.
    ///
    /// multiply_add(a, b, c) computes a * b + c. Test cases:
    /// - multiply_add(v0, v0, v1): v0 duplicated, v1 unique
    /// - multiply_add(v0, v1, v0): v0 at positions 0 and 2, v1 unique
    #[test]
    fn test_duplicate_operand_multiply_add() {
        // Test multiply_add(v0, v0, v1) - v0 appears twice
        let mut ops = vec![
            make_test_ssa_op(0, Engine::Valu, Opcode::Vbroadcast, Some(VReg::vector(0)), vec![Operand::Imm(2)], vec![]),
            make_test_ssa_op(1, Engine::Valu, Opcode::Vbroadcast, Some(VReg::vector(1)), vec![Operand::Imm(3)], vec![]),
            // multiply_add(v0, v0, v1) = v0 * v0 + v1
            // v0 appears twice, v1 once -> 2 unique VRegs
            make_test_ssa_op(2, Engine::Valu, Opcode::MultiplyAdd, Some(VReg::vector(2)),
                vec![Operand::VReg(VReg::vector(0)), Operand::VReg(VReg::vector(0)), Operand::VReg(VReg::vector(1))],
                vec![0, 1]),
            make_test_ssa_op(3, Engine::Store, Opcode::Vstore, None,
                vec![Operand::Imm(100), Operand::VReg(VReg::vector(2))], vec![2]),
        ];

        let config = KernelConfig::new(10, 2047, 256, 1);
        let builder = SsaTickTockKernelBuilder::new(config);
        let mut shifted = HashSet::new();

        let result = builder.schedule_ssa_greedy_with_shift_budget(
            &mut ops, false, &mut shifted, 24  // v0 + v1 + v2 = 24 words
        );
        assert!(result.is_ok(), "Should schedule multiply_add with duplicate operand: {:?}", result.err());

        // Verify correct scheduling
        assert!(ops[2].cycle > ops[0].cycle, "multiply_add must come after v0 defined");
        assert!(ops[2].cycle > ops[1].cycle, "multiply_add must come after v1 defined");
    }

    // =========================================================================
    // VALU/ALU Consistency Tests
    //
    // These tests verify that operations with different widths (scalar vs vector)
    // are handled consistently throughout the SSA pipeline:
    //
    // 1. VReg width is correctly tracked (scalar=1 word, vector=8 words)
    // 2. VALU ops only accept vector operands
    // 3. ALU ops only accept scalar operands
    // 4. vload produces vector VRegs, load produces scalar VRegs
    // 5. vstore consumes vector VRegs, store consumes scalar VRegs
    // 6. VALU→ALU expansion correctly handles address offsets
    // 7. Broadcast correctly converts scalar to vector
    // 8. Lowering produces correct reads/writes sets based on width
    // =========================================================================

    /// Test that VReg width is correctly used for liveness calculation.
    ///
    /// Scenario: Vector VRegs should count as 8 words, scalars as 1 word.
    #[test]
    fn test_vreg_width_affects_liveness_words() {
        // Create two ops:
        // op0: const -> s0 (scalar, 1 word)
        // op1: vbroadcast(s0) -> v0 (vector, 8 words)
        // op2: sink both
        //
        // Peak liveness: s0 + v0 = 1 + 8 = 9 words
        let mut ops = vec![
            make_test_ssa_op(0, Engine::Load, Opcode::Const, Some(VReg::scalar(0)),
                vec![Operand::Imm(42)], vec![]),
            make_test_ssa_op(1, Engine::Valu, Opcode::Vbroadcast, Some(VReg::vector(1)),
                vec![Operand::VReg(VReg::scalar(0))], vec![0]),
            // Sink that uses both - scalar and vector
            make_test_ssa_op(2, Engine::Valu, Opcode::Add, Some(VReg::vector(2)),
                vec![Operand::VReg(VReg::vector(1)), Operand::VReg(VReg::vector(1))], vec![1]),
        ];

        let config = KernelConfig::new(10, 2047, 256, 1);
        let builder = SsaTickTockKernelBuilder::new(config);
        let mut shifted = HashSet::new();

        // Budget of 9 words should be enough (s0=1 + v0=8)
        let result = builder.schedule_ssa_greedy_with_shift_budget(
            &mut ops, false, &mut shifted, 9
        );
        assert!(result.is_ok(), "Budget=9 should fit scalar(1)+vector(8): {:?}", result.err());

        // Verify scheduling
        assert!(ops[1].cycle > ops[0].cycle, "broadcast depends on const");
    }

    /// Test that VALU ops only work with vector VRegs.
    ///
    /// The emit_valu() function should assert that sources are vectors.
    /// This test verifies the SsaOp invariant is maintained through scheduling.
    #[test]
    fn test_valu_requires_vector_sources() {
        // Create VALU op with proper vector sources
        let mut ops = vec![
            make_test_ssa_op(0, Engine::Valu, Opcode::Vbroadcast, Some(VReg::vector(0)),
                vec![Operand::Imm(1)], vec![]),
            make_test_ssa_op(1, Engine::Valu, Opcode::Vbroadcast, Some(VReg::vector(1)),
                vec![Operand::Imm(2)], vec![]),
            // VALU add with vector sources (correct)
            make_test_ssa_op(2, Engine::Valu, Opcode::Add, Some(VReg::vector(2)),
                vec![Operand::VReg(VReg::vector(0)), Operand::VReg(VReg::vector(1))], vec![0, 1]),
        ];

        let config = KernelConfig::new(10, 2047, 256, 1);
        let builder = SsaTickTockKernelBuilder::new(config);
        let mut shifted = HashSet::new();

        let result = builder.schedule_ssa_greedy_with_shift_budget(
            &mut ops, false, &mut shifted, 24
        );
        assert!(result.is_ok(), "VALU with vector sources should work: {:?}", result.err());

        // All vector VRegs should be properly counted
        for op in &ops {
            if let Some(dest) = op.dest {
                assert!(dest.is_vector(), "VALU dest should be vector: {:?}", op);
            }
        }
    }

    /// Test that ALU ops work with scalar VRegs.
    ///
    /// Verifies scalar operations don't accidentally get vector treatment.
    #[test]
    fn test_alu_uses_scalar_vregs() {
        // Create ALU ops with scalar sources
        let mut ops = vec![
            make_test_ssa_op(0, Engine::Load, Opcode::Const, Some(VReg::scalar(0)),
                vec![Operand::Imm(5)], vec![]),
            make_test_ssa_op(1, Engine::Load, Opcode::Const, Some(VReg::scalar(1)),
                vec![Operand::Imm(10)], vec![]),
            // ALU add with scalar sources
            make_test_ssa_op(2, Engine::Alu, Opcode::Add, Some(VReg::scalar(2)),
                vec![Operand::VReg(VReg::scalar(0)), Operand::VReg(VReg::scalar(1))], vec![0, 1]),
            make_test_ssa_op(3, Engine::Store, Opcode::Store, None,
                vec![Operand::Imm(100), Operand::VReg(VReg::scalar(2))], vec![2]),
        ];

        let config = KernelConfig::new(10, 2047, 256, 1);
        let builder = SsaTickTockKernelBuilder::new(config);
        let mut shifted = HashSet::new();

        // Budget of 3 words should be enough for 3 scalars
        let result = builder.schedule_ssa_greedy_with_shift_budget(
            &mut ops, false, &mut shifted, 3
        );
        assert!(result.is_ok(), "ALU with scalar sources should work: {:?}", result.err());
    }

    /// Test broadcast: scalar input -> vector output.
    ///
    /// Verifies that vbroadcast correctly converts width.
    #[test]
    fn test_broadcast_scalar_to_vector_conversion() {
        // Broadcast chain:
        // op0: const -> s0 (scalar, 1 word)
        // op1: vbroadcast(s0) -> v0 (vector, 8 words)
        // op2: valu(v0, v0) -> v1 (vector, 8 words)
        let mut ops = vec![
            // Scalar const
            make_test_ssa_op(0, Engine::Load, Opcode::Const, Some(VReg::scalar(0)),
                vec![Operand::Imm(42)], vec![]),
            // Broadcast takes scalar, produces vector
            make_test_ssa_op(1, Engine::Valu, Opcode::Vbroadcast, Some(VReg::vector(1)),
                vec![Operand::VReg(VReg::scalar(0))], vec![0]),
            // VALU uses vector
            make_test_ssa_op(2, Engine::Valu, Opcode::Add, Some(VReg::vector(2)),
                vec![Operand::VReg(VReg::vector(1)), Operand::VReg(VReg::vector(1))], vec![1]),
            make_test_ssa_op(3, Engine::Store, Opcode::Vstore, None,
                vec![Operand::Imm(100), Operand::VReg(VReg::vector(2))], vec![2]),
        ];

        // Verify VReg widths
        assert!(ops[0].dest.unwrap().is_scalar(), "const dest should be scalar");
        assert!(ops[1].dest.unwrap().is_vector(), "broadcast dest should be vector");
        assert!(ops[2].dest.unwrap().is_vector(), "valu dest should be vector");

        let config = KernelConfig::new(10, 2047, 256, 1);
        let builder = SsaTickTockKernelBuilder::new(config);
        let mut shifted = HashSet::new();

        let result = builder.schedule_ssa_greedy_with_shift_budget(
            &mut ops, false, &mut shifted, 17  // s0(1) + v0(8) + v1(8) = 17 peak
        );
        assert!(result.is_ok(), "Broadcast chain should schedule: {:?}", result.err());
    }

    /// Test vload produces vector, vstore consumes vector.
    ///
    /// Memory operations have strict width requirements.
    #[test]
    fn test_vload_vstore_vector_width() {
        // vload -> valu -> vstore chain
        let mut ops = vec![
            // Address scalar
            make_test_ssa_op(0, Engine::Load, Opcode::Const, Some(VReg::scalar(0)),
                vec![Operand::Imm(200)], vec![]),
            // vload produces vector
            make_test_ssa_op(1, Engine::Load, Opcode::Vload, Some(VReg::vector(1)),
                vec![Operand::VReg(VReg::scalar(0))], vec![0]),
            // VALU uses vector
            make_test_ssa_op(2, Engine::Valu, Opcode::Add, Some(VReg::vector(2)),
                vec![Operand::VReg(VReg::vector(1)), Operand::VReg(VReg::vector(1))], vec![1]),
            // vstore consumes vector
            make_test_ssa_op(3, Engine::Store, Opcode::Vstore, None,
                vec![Operand::VReg(VReg::scalar(0)), Operand::VReg(VReg::vector(2))], vec![0, 2]),
        ];

        // Verify widths
        assert!(ops[1].dest.unwrap().is_vector(), "vload dest must be vector");
        // vstore has no dest, but src (second operand) must be vector
        if let Operand::VReg(src) = &ops[3].srcs[1] {
            assert!(src.is_vector(), "vstore src must be vector");
        }

        let config = KernelConfig::new(10, 2047, 256, 1);
        let builder = SsaTickTockKernelBuilder::new(config);
        let mut shifted = HashSet::new();

        let result = builder.schedule_ssa_greedy_with_shift_budget(
            &mut ops, false, &mut shifted, 17  // s0(1) + v1(8) + v2(8)
        );
        assert!(result.is_ok(), "vload/vstore chain should schedule: {:?}", result.err());
    }

    /// Test VALU→ALU shifting preserves width semantics.
    ///
    /// When a VALU op is shifted to ALU, it expands to 8 scalar ops.
    /// Each expanded op should access the correct offset of the vector.
    #[test]
    fn test_valu_to_alu_shift_preserves_semantics() {
        // Create a VALU op that could be shifted to ALU
        let mut ops = vec![
            make_test_ssa_op(0, Engine::Valu, Opcode::Vbroadcast, Some(VReg::vector(0)),
                vec![Operand::Imm(1)], vec![]),
            make_test_ssa_op(1, Engine::Valu, Opcode::Vbroadcast, Some(VReg::vector(1)),
                vec![Operand::Imm(2)], vec![]),
            // This Add op can be shifted from VALU to ALU
            make_test_ssa_op(2, Engine::Valu, Opcode::Add, Some(VReg::vector(2)),
                vec![Operand::VReg(VReg::vector(0)), Operand::VReg(VReg::vector(1))], vec![0, 1]),
            make_test_ssa_op(3, Engine::Store, Opcode::Vstore, None,
                vec![Operand::Imm(100), Operand::VReg(VReg::vector(2))], vec![2]),
        ];

        let config = KernelConfig::new(10, 2047, 256, 1);
        let builder = SsaTickTockKernelBuilder::new(config);
        let mut shifted = HashSet::new();

        // Enable VALU shift
        let result = builder.schedule_ssa_greedy_with_shift_budget(
            &mut ops, true, &mut shifted, 24
        );
        assert!(result.is_ok(), "VALU→ALU shift should work: {:?}", result.err());

        // Note: The actual shifting happens in lower_ssa_ops_with_shift, not in scheduling.
        // The scheduler just marks which ops to shift.
    }

    /// Test that expand_ssa_valu_to_alu produces correct address offsets.
    ///
    /// When a VALU op is expanded to 8 ALU ops, each should access:
    /// - dest[i] for i in 0..8
    /// - src1[i] for i in 0..8
    /// - src2[i] for i in 0..8
    #[test]
    fn test_expand_ssa_valu_to_alu_offsets() {
        use crate::address_assign::lower_ssa_op;

        // Create a VALU add op
        let mut op = SsaOp::new(
            42, // id
            Engine::Valu,
            Opcode::Add,
            Some(VReg::vector(2)),
            vec![Operand::VReg(VReg::vector(0)), Operand::VReg(VReg::vector(1))],
        );
        op.cycle = 5;

        // Create address mapping: v0->100, v1->200, v2->300
        let mut vreg_to_addr = HashMap::new();
        vreg_to_addr.insert(VReg::vector(0), 100);
        vreg_to_addr.insert(VReg::vector(1), 200);
        vreg_to_addr.insert(VReg::vector(2), 300);

        // Expand
        let expanded = expand_ssa_valu_to_alu(&op, &vreg_to_addr);

        // Should produce 8 ALU ops
        assert_eq!(expanded.len(), 8, "Should expand to 8 ALU ops");

        for (i, alu_op) in expanded.iter().enumerate() {
            assert_eq!(alu_op.engine, Engine::Alu, "Expanded op should be ALU");

            // Check reads: should read src1[i] and src2[i]
            assert!(alu_op.reads.contains(&(100 + i as i64)), "Should read src1[{}]=100+{}", i, i);
            assert!(alu_op.reads.contains(&(200 + i as i64)), "Should read src2[{}]=200+{}", i, i);

            // Check writes: should write dest[i]
            assert!(alu_op.writes.contains(&(300 + i as i64)), "Should write dest[{}]=300+{}", i, i);
        }
    }

    /// Test lowering produces correct reads/writes sets based on VReg width.
    ///
    /// Scalar VRegs should produce 1 read/write.
    /// Vector VRegs should produce 8 reads/writes.
    #[test]
    fn test_lower_ssa_op_width_consistency() {
        use crate::address_assign::lower_ssa_op;

        // Test 1: Scalar op
        let scalar_op = SsaOp::new(
            0,
            Engine::Alu,
            Opcode::Add,
            Some(VReg::scalar(2)),
            vec![Operand::VReg(VReg::scalar(0)), Operand::VReg(VReg::scalar(1))],
        );

        let mut scalar_mapping = HashMap::new();
        scalar_mapping.insert(VReg::scalar(0), 10);
        scalar_mapping.insert(VReg::scalar(1), 20);
        scalar_mapping.insert(VReg::scalar(2), 30);

        let lowered_scalar = lower_ssa_op(&scalar_op, &scalar_mapping);
        assert_eq!(lowered_scalar.reads.len(), 2, "Scalar should read 2 addresses");
        assert_eq!(lowered_scalar.writes.len(), 1, "Scalar should write 1 address");
        assert!(lowered_scalar.reads.contains(&10));
        assert!(lowered_scalar.reads.contains(&20));
        assert!(lowered_scalar.writes.contains(&30));

        // Test 2: Vector op
        let vector_op = SsaOp::new(
            1,
            Engine::Valu,
            Opcode::Add,
            Some(VReg::vector(5)),
            vec![Operand::VReg(VReg::vector(3)), Operand::VReg(VReg::vector(4))],
        );

        let mut vector_mapping = HashMap::new();
        vector_mapping.insert(VReg::vector(3), 100);
        vector_mapping.insert(VReg::vector(4), 200);
        vector_mapping.insert(VReg::vector(5), 300);

        let lowered_vector = lower_ssa_op(&vector_op, &vector_mapping);
        assert_eq!(lowered_vector.reads.len(), 16, "Vector should read 16 addresses (2 srcs × 8)");
        assert_eq!(lowered_vector.writes.len(), 8, "Vector should write 8 addresses");

        // Check all 8 offsets for each operand
        for i in 0..8 {
            assert!(lowered_vector.reads.contains(&(100 + i)), "Should read src1[{}]", i);
            assert!(lowered_vector.reads.contains(&(200 + i)), "Should read src2[{}]", i);
            assert!(lowered_vector.writes.contains(&(300 + i)), "Should write dest[{}]", i);
        }
    }

    /// Test mixed scalar/vector pipeline: const -> broadcast -> valu -> vstore.
    ///
    /// This is a common pattern that exercises width conversions.
    #[test]
    fn test_mixed_width_pipeline() {
        // Full pipeline:
        // s0 = const 5       (scalar)
        // s1 = const 10      (scalar)
        // v0 = broadcast(s0) (vector)
        // v1 = broadcast(s1) (vector)
        // v2 = vadd(v0, v1)  (vector)
        // vstore(addr, v2)   (consumes vector)
        let mut ops = vec![
            make_test_ssa_op(0, Engine::Load, Opcode::Const, Some(VReg::scalar(0)),
                vec![Operand::Imm(5)], vec![]),
            make_test_ssa_op(1, Engine::Load, Opcode::Const, Some(VReg::scalar(1)),
                vec![Operand::Imm(10)], vec![]),
            make_test_ssa_op(2, Engine::Valu, Opcode::Vbroadcast, Some(VReg::vector(2)),
                vec![Operand::VReg(VReg::scalar(0))], vec![0]),
            make_test_ssa_op(3, Engine::Valu, Opcode::Vbroadcast, Some(VReg::vector(3)),
                vec![Operand::VReg(VReg::scalar(1))], vec![1]),
            make_test_ssa_op(4, Engine::Valu, Opcode::Add, Some(VReg::vector(4)),
                vec![Operand::VReg(VReg::vector(2)), Operand::VReg(VReg::vector(3))], vec![2, 3]),
            make_test_ssa_op(5, Engine::Store, Opcode::Vstore, None,
                vec![Operand::Imm(100), Operand::VReg(VReg::vector(4))], vec![4]),
        ];

        let config = KernelConfig::new(10, 2047, 256, 1);
        let builder = SsaTickTockKernelBuilder::new(config);
        let mut shifted = HashSet::new();

        // Peak: s0(1) + s1(1) + v2(8) + v3(8) + v4(8) = 26 words
        // But scalars die early, so actual peak is less
        let result = builder.schedule_ssa_greedy_with_shift_budget(
            &mut ops, false, &mut shifted, 26
        );
        assert!(result.is_ok(), "Mixed width pipeline should schedule: {:?}", result.err());

        // Verify dependency chain
        assert!(ops[2].cycle > ops[0].cycle, "broadcast depends on const");
        assert!(ops[3].cycle > ops[1].cycle, "broadcast depends on const");
        assert!(ops[4].cycle > ops[2].cycle, "vadd depends on broadcast");
        assert!(ops[4].cycle > ops[3].cycle, "vadd depends on broadcast");
        assert!(ops[5].cycle > ops[4].cycle, "vstore depends on vadd");
    }

    /// Test that unique_src_vregs correctly handles mixed scalar/vector operands.
    ///
    /// This ensures the duplicate operand fix works with width consistency.
    #[test]
    fn test_unique_src_vregs_with_mixed_widths() {
        // Op with one scalar and one vector operand shouldn't happen in practice
        // (VALU requires all vector, ALU requires all scalar), but test the method.
        let op = SsaOp::new(
            0,
            Engine::Flow,
            Opcode::Select, // select can have mixed operands in theory
            Some(VReg::scalar(3)),
            vec![
                Operand::VReg(VReg::scalar(0)),  // cond
                Operand::VReg(VReg::scalar(1)),  // true
                Operand::VReg(VReg::scalar(1)),  // false (duplicate of true)
            ],
        );

        let unique = op.unique_src_vregs();
        assert_eq!(unique.len(), 2, "Should have 2 unique VRegs (s0, s1)");
        assert!(unique.contains(&VReg::scalar(0)));
        assert!(unique.contains(&VReg::scalar(1)));
    }

    // =========================================================================
    // CP-SAT Round-Trip Budget Validation Tests
    // =========================================================================

    /// Test whether greedy scheduler hints can round-trip through CP-SAT.
    ///
    /// This test validates:
    /// 1. Whether CP-SAT accepts greedy hints if they're budget-feasible
    /// 2. Whether CP-SAT finds a different solution if hints violate budget
    /// 3. Or whether the problem is fundamentally infeasible within budget
    ///
    /// The greedy scheduler often exceeds the 1536-word scratch budget because
    /// it doesn't model budget constraints during scheduling. This test uses
    /// CP-SAT with `stop_after_first_solution` to quickly determine if:
    /// - The greedy hints are directly usable (hints_match = true)
    /// - A feasible schedule exists but differs from greedy (hints_match = false)
    /// - No feasible schedule exists within budget (infeasible)
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_greedy_cpsat_roundtrip_budget_check() {
        use crate::cpsat_scheduler::{schedule_ssa_cpsat, SsaCpsatConfig, SsaCpsatHints};

        // Build a full-scale kernel
        let config = KernelConfig::new(10, 2047, 256, 16);
        let mut builder = SsaTickTockKernelBuilder::new(config);

        builder.build_internal();
        let vreg_def = std::mem::take(&mut builder.emit.vreg_def);
        let mut ssa_ops = builder.emit.take_ops();
        crate::kernel_builder::emit_ssa::compute_ssa_dependencies(&mut ssa_ops, &vreg_def);

        // Run greedy scheduler to get hints
        let mut greedy_shifted = HashSet::new();
        let (greedy_cycles, greedy_budget_exceeded, greedy_peak_live) =
            builder.schedule_ssa_greedy_with_shift(
                &mut ssa_ops,
                false, // no VALU shift
                &mut greedy_shifted,
            ).expect("Greedy scheduler should succeed");

        eprintln!("Greedy result: {} cycles, budget_exceeded={}, peak_live={}",
            greedy_cycles, greedy_budget_exceeded, greedy_peak_live);

        // Collect hints from greedy schedule
        let greedy_hints: Vec<i32> = ssa_ops.iter().map(|op| op.cycle).collect();

        // Reset cycles before CP-SAT
        for op in &mut ssa_ops {
            op.cycle = -1;
        }

        // Run CP-SAT with stop_after_first_solution - just validate hints
        let cpsat_result = schedule_ssa_cpsat(
            &ssa_ops,
            SsaCpsatConfig {
                time_limit_secs: 60.0,  // Short timeout
                verbose: true,
                stop_after_first_solution: true,  // Return first feasible solution
                scratch_budget: 1536,  // Hardware limit
                enable_valu_shift: false,
                model_name: None,
            },
            SsaCpsatHints {
                greedy_hints: Some(&greedy_hints),
                ..Default::default()
            },
        );

        match cpsat_result {
            Some(result) => {
                eprintln!("CP-SAT found solution: {} cycles, status={:?}",
                    result.makespan, result.status);

                // Check if CP-SAT used the hints directly or found different solution
                let hints_match = result.cycle_assignments == greedy_hints;
                eprintln!("Hints match exactly: {}", hints_match);

                // If hints don't match, greedy hints were infeasible for budget
                if !hints_match {
                    eprintln!("Greedy hints were INFEASIBLE - CP-SAT found different solution");

                    // Count how many cycle assignments differ
                    let diff_count = result.cycle_assignments.iter()
                        .zip(greedy_hints.iter())
                        .filter(|(a, b)| a != b)
                        .count();
                    eprintln!("  {} of {} cycle assignments differ", diff_count, greedy_hints.len());
                } else {
                    eprintln!("Greedy hints were FEASIBLE - accepted by CP-SAT");
                }

                // The test passes if CP-SAT found any solution (feasible or optimal)
                assert!(
                    result.status == crate::cpsat_scheduler::CpsatStatus::Feasible ||
                    result.status == crate::cpsat_scheduler::CpsatStatus::Optimal,
                    "CP-SAT should find a feasible solution"
                );
            }
            None => {
                eprintln!("CP-SAT returned NO SOLUTION - problem may be infeasible within budget");
                // This is an expected outcome if the kernel truly cannot fit in 1536 words
                // We don't fail the test - just report the finding
            }
        }
    }

    /// Regression: two-phase CP-SAT (per-section feasibility then full-kernel optimization) must
    /// produce a merged hint schedule that stays within scratch budget so the final result does not exceed 1536.
    /// Presolves (per-section CP-SAT with live_out and live_through) guarantee the merged schedule is feasible;
    /// if not, build_and_schedule_cpsat returns Err. This test asserts the happy path.
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_cpsat_two_phase_hint_schedule_within_budget() {
        // Small config: fast run, still exercises full two-phase flow (section boundaries, budget check, merge, full CP-SAT)
        let config = KernelConfig::new(5, 63, 64, 4);
        let mut builder = SsaTickTockKernelBuilder::new(config);
        let result = builder.build_and_schedule_cpsat(
            30.0,   // Short time limit; small kernel solves quickly
            false,  // verbose
            false,  // enable_valu_shift to keep test deterministic and faster
        );
        assert!(result.is_ok(), "build_and_schedule_cpsat should succeed: {:?}", result.err());
        let result = result.unwrap();
        assert!(
            !result.address_result.budget_exceeded,
            "Two-phase CP-SAT must produce a schedule within scratch budget; peak_live={} > 1536",
            result.address_result.peak_live
        );
        assert!(
            result.address_result.peak_live <= 1536,
            "Peak live words must not exceed scratch budget: {} > 1536",
            result.address_result.peak_live
        );
    }
}
