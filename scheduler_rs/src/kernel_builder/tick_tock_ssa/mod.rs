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

use crate::address_assign::{
    assign_addresses, build_vreg_metadata, AddressAssignConfig, AddressAssignmentResult,
};
use crate::types::{Engine, MachineConfig, Op, SsaOp, UsageAccounting, VReg, VarType};
use crate::types::SlotVal;

use super::barrier_elision::elide_barriers;
use super::config::{fma_multiplier, KernelConfig};
use super::context_ssa::SsaScratchContext;
use super::emit_ssa::{
    compute_ssa_dependencies_with_barriers, eliminate_ssa_dead_code,
    SsaEmitContext,
};
use super::ssa_greedy;
use super::ssa_lower::lower_ssa_ops_with_shift;
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
        println!("[DEBUG] build_and_schedule_with_options ENTRY");
        std::io::Write::flush(&mut std::io::stdout()).ok();
        // Build SSA program (emission output)
        let program = self.build();
        let SsaProgram { ops: mut ssa_ops, mut vreg_def, vreg_count, mut const_alternatives, zero_vreg } = program;

        // Use barrier-aware dependencies for greedy scheduler
        compute_ssa_dependencies_with_barriers(&mut ssa_ops, &vreg_def, zero_vreg);

        // Optional SSA DCE: remove ops whose result has no users (saves cycles, optimizer time, scratch)
        if self.config.compiler.enable_ssa_dce {
            let (new_ops, new_vreg_def, new_const_alternatives) = eliminate_ssa_dead_code(
                ssa_ops,
                &mut vreg_def,
                &mut const_alternatives,
                zero_vreg,
            );
            ssa_ops = new_ops;
            vreg_def = new_vreg_def;
            const_alternatives = new_const_alternatives;
            compute_ssa_dependencies_with_barriers(&mut ssa_ops, &vreg_def, zero_vreg);
        }

        // Build set of AddImm alternative op indices for the greedy scheduler to skip
        // The greedy scheduler always uses the Load alternative; AddImm alternatives
        // are only considered by CP-SAT. Mark them as dead so they're not scheduled.
        let const_add_imm_indices: std::collections::HashSet<usize> = const_alternatives
            .iter()
            .map(|alt| alt.add_imm_op_idx)
            .collect();

        let scratch_budget = scratch_budget_override.unwrap_or(SCRATCH_BUDGET_HW);

        // Schedule using greedy scheduler with VALU→ALU shifting
        // Collect SelectChoice and ConstChoice (add_imm) decisions from greedy scheduler
        let mut shifted_ops: HashSet<usize> = HashSet::new();
        let mut greedy_select_fma_ops = HashSet::new();
        let mut greedy_const_add_imm_ops = HashSet::new();
        let (cycles, sched_budget_exceeded, _sched_peak_live) =
            ssa_greedy::schedule_ssa_greedy_with_shift_budget_and_selectchoice(
                &mut ssa_ops,
                enable_valu_shift,
                &mut shifted_ops,
                scratch_budget,
                Some(&mut greedy_select_fma_ops),
                None, // Don't need cycle hints for greedy path
                None,
                Some(&mut greedy_const_add_imm_ops),
                None, // Greedy-only path: no CP-SAT bounds
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
        let ssa_ops: Vec<SsaOp> = ssa_ops.into_iter().filter(|op| op.cycle >= 0).collect();

        // Assign addresses via linear scan (budget = hardware limit; zero_vreg included in peak so we stay within capacity)
        let mut assign_config = AddressAssignConfig::default();
        assign_config.budget = SCRATCH_BUDGET_HW;

        // Reserve address 0 and include zero in peak_live so scheduler budget 1536 matches assignable+zero
        if let Some(zero) = zero_vreg {
            assign_config.fixed_vregs.insert(zero, 0);
            assign_config.zero_vreg = Some(zero);
        }

        eprintln!(
            "[TRACE build_and_schedule] About to call assign_addresses with {} ops",
            ssa_ops.len()
        );

        let address_result = assign_addresses(&ssa_ops, &assign_config);

        eprintln!(
            "[TRACE build_and_schedule] assign_addresses returned, peak_live={}",
            address_result.peak_live
        );

        // Debug: verify all source VRegs have addresses assigned
        if std::env::var("SCHEDULER_DEBUG").is_ok() {
            verify_vreg_address_consistency(&ssa_ops, &address_result.vreg_to_addr);
        }

        // Track if budget was exceeded (either in scheduling or address assignment)
        let budget_exceeded = sched_budget_exceeded || address_result.budget_exceeded;

        if budget_exceeded {
            eprintln!(
                "Budget exceeded: sched={}, addr_assign={}, peak_live={}, budget={}",
                sched_budget_exceeded,
                address_result.budget_exceeded,
                address_result.peak_live,
                address_result.budget
            );
            return Err(format!(
                "scratch budget exceeded: peak_live={} > budget={}",
                address_result.peak_live, address_result.budget
            ));
        }

        // Compute usage accounting from SSA ops (before lowering loses metadata)
        let mut usage_accounting = UsageAccounting::from_ssa_ops(&ssa_ops);
        usage_accounting.set_peak_live(address_result.peak_live);

        // Build VReg metadata for liveness debugging (before lowering loses SSA info)
        let vreg_metadata = build_vreg_metadata(&ssa_ops);

        // Debug: dump scheduler decisions for comparison (e.g. CP-SAT vs greedy)
        if std::env::var("SCHEDULER_DEBUG_DECISIONS").is_ok() {
            let mut s: Vec<_> = shifted_ops.iter().copied().collect();
            s.sort_unstable();
            let mut c: Vec<_> = greedy_const_add_imm_ops.iter().copied().collect();
            c.sort_unstable();
            let mut f: Vec<_> = greedy_select_fma_ops.iter().copied().collect();
            f.sort_unstable();
            eprintln!(
                "[GREEDY_DECISIONS] shifted_ops={:?} const_add_imm_ops={:?} select_fma_ops={:?}",
                s, c, f
            );
        }

        // Lower to concrete ops, expanding shifted VALU ops to ALU
        // Greedy schedules ConstChoice as add_imm only when Flow is free and no other Flow op is ready
        let ops = lower_ssa_ops_with_shift(
            &ssa_ops,
            &address_result.vreg_to_addr,
            &shifted_ops,
            &greedy_const_add_imm_ops,
            &HashMap::new(),        // Greedy doesn't use ALU-derived const
            &greedy_select_fma_ops, // Pass greedy's SelectChoice decisions
            &HashMap::new(),        // Greedy doesn't provide individual ALU cycles
            &HashSet::new(),        // Greedy doesn't eliminate vbroadcasts
            &HashMap::new(),
        );

        let valu_shifted = shifted_ops.len();

        // Cycle count must be derived from lowered ops: SelectChoice FMA (no delta) emits
        // sub at C and FMA at C+1, so we need max(lowered op cycle)+1.
        let cycles = ops
            .iter()
            .map(|op| op.cycle.max(0) as usize)
            .max()
            .map(|m| m + 1)
            .unwrap_or(cycles)
            .max(cycles);

        Ok(SsaScheduleResult {
            shifted_alu_cycles: HashMap::new(), // Greedy doesn't provide individual ALU cycles
            ops,
            cycles,
            address_result,
            vreg_count,
            valu_shifted,
            budget_exceeded,
            usage_accounting,
            vreg_metadata,
            empty_cycle_count: None,
            makespan_driven_by_capacity: None,
            schedule_constraints: None,
            ssa_cycle_assignments: None,
            ssa_ops_bulk_move: None,
            eliminated_vbroadcast_ops: vec![],
            lb_cycles: None,
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
    ///
    /// When `per_section_optimize_secs` is `Some(t)` with t > 0, each per-section CP-SAT
    /// run uses that time limit and optimizes for cycles (stop_after_first_solution: false).
    /// Per-section never uses VALU shift. When `None`, per-section runs in feasibility-only
    /// mode (stop at first feasible solution) with the usual time cap.
    ///
    /// Returns `Err` if scratch budget was exceeded (peak_live > budget); the caller must not use the kernel in that case.
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
        use crate::cpsat_scheduler::{schedule_ssa_cpsat, SsaCpsatConfig, SsaCpsatHints};
        #[cfg(feature = "cpsat")]
        use crate::cpsat_scheduler::compute_ssa_asap_alap_bounds;
        use crate::types::Opcode;

        // Build SSA program (emission output)
        let program = self.build();
        let SsaProgram { ops: mut ssa_ops, mut vreg_def, vreg_count, mut const_alternatives, zero_vreg } = program;

        // Use barrier-aware dependencies for greedy hints
        compute_ssa_dependencies_with_barriers(&mut ssa_ops, &vreg_def, zero_vreg);

        // Optional SSA DCE: remove ops whose result has no users (saves cycles, optimizer time, scratch)
        if self.config.compiler.enable_ssa_dce {
            let (new_ops, new_vreg_def, new_const_alternatives) = eliminate_ssa_dead_code(
                ssa_ops,
                &mut vreg_def,
                &mut const_alternatives,
                zero_vreg,
            );
            ssa_ops = new_ops;
            vreg_def = new_vreg_def;
            const_alternatives = new_const_alternatives;
            compute_ssa_dependencies_with_barriers(&mut ssa_ops, &vreg_def, zero_vreg);
        }
        let _ = &const_alternatives; // used in greedy path; CP-SAT discovers alternatives from ops

        // CP-SAT scratch liveness only models VRegs with at least one use. When DCE ran,
        // every op.dest must have a reader in the SSA op list. (sub_temp_vreg is not
        // included: it is only read by the FMA that appears after lowering, not by any SsaOp.)
        if self.config.compiler.enable_ssa_dce {
            let defined: HashSet<_> = ssa_ops.iter().filter_map(|op| op.dest).collect();
            let used: HashSet<_> = ssa_ops.iter().flat_map(|op| op.reads()).collect();
            let dead: Vec<_> = defined.difference(&used).copied().collect();
            assert!(
                dead.is_empty(),
                "After DCE every defined VReg must have a use (CP-SAT scratch model expects this). Dead: {:?}",
                dead
            );
        }

        let barrier_count = count_barriers(&ssa_ops);

        // Compute ASAP/ALAP bounds (with optional tightening) before greedy so hints stay inside CP-SAT domains.
        #[cfg(feature = "cpsat")]
        let (asap_bounds, alap_bounds) = {
            let bounds_config = SsaCpsatConfig {
                enable_asap_alap_tightening,
                enable_scratch_tightening,
                enable_valu_shift,
                scratch_budget: effective_scratch_budget(zero_vreg.is_some()),
                ..Default::default()
            };
            compute_ssa_asap_alap_bounds(&ssa_ops, &bounds_config)
        };
        #[cfg(feature = "cpsat")]
        let asap_alap_param = Some((&asap_bounds, &alap_bounds));
        #[cfg(not(feature = "cpsat"))]
        let asap_alap_param = None;

        // Get greedy hints first (for warm-starting CP-SAT)
        // If shift is enabled, also get shift hints from greedy scheduler
        // Collect SelectChoice and ConstChoice (add_imm) decisions from greedy scheduler
        let mut greedy_shifted = HashSet::new();
        let mut greedy_select_fma_ops = HashSet::new();
        let mut greedy_select_sub_cycles = HashMap::new();
        let mut greedy_select_fma_cycles = HashMap::new();
        let mut greedy_const_add_imm_ops = HashSet::new();
        let (greedy_cycles, greedy_budget_exceeded, greedy_peak_live) =
            ssa_greedy::schedule_ssa_greedy_with_shift_budget_and_selectchoice(
                &mut ssa_ops,
                enable_valu_shift, // Match CP-SAT shift setting for hints
                &mut greedy_shifted,
                1536, // Scratch budget
                Some(&mut greedy_select_fma_ops),
                Some(&mut greedy_select_sub_cycles),
                Some(&mut greedy_select_fma_cycles),
                Some(&mut greedy_const_add_imm_ops), // Hint ConstChoice as add_imm when greedy used Flow
                asap_alap_param,
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
        let effective_budget = effective_scratch_budget(zero_vreg.is_some());
        let budget_result =
            sections_exceeding_budget(
                &ssa_ops,
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
        let vreg_def_sec = vreg_def_section(&ssa_ops, &boundaries);
        let mut section_offset: i32 = 0;
        // Per-section SelectChoice FMA sets from segment CP-SAT; used when measuring peak so we match segment decisions.
        let mut section_select_fma_overrides: Vec<Option<HashSet<usize>>> =
            vec![None; boundaries.len()];
        // Merged hint state: cycles live on ssa_ops; choice hints maintained here.
        // Before any per-section run, the hint for every section is greedy. When we run per-section
        // CP-SAT for a section, we pass the current hint for that section into the segment; after
        // it runs we update cycles and these choice sets so the final model gets greedy-or-CP-SAT
        // per section consistently.
        let mut current_select_fma: HashSet<usize> = greedy_select_fma_ops.clone();
        let mut current_const_add_imm: HashSet<usize> = greedy_const_add_imm_ops.clone();

        for k in 0..boundaries.len() {
            let mut this_section_makespan: Option<i32> = None;
            let (start, end) = boundaries[k];
            // Rebase this section into [section_offset, section_offset + span) so the merged
            // schedule is a strict concatenation. Do this every iteration (whether we run
            // per-section CP-SAT or not) so segment_hints (c - section_offset) are correct.
            let section_span_after_rebase: i32 = {
                let mut min_c = i32::MAX;
                let mut max_c = i32::MIN;
                for op in &ssa_ops[start..end] {
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
                    for op in &mut ssa_ops[start..end] {
                        if op.opcode != Opcode::Barrier && op.cycle >= 0 {
                            op.cycle = section_offset + (op.cycle - min_c);
                        }
                    }
                }
                span
            };
            if budget_result.section_exceeds_budget[k] {
                let global_indices: Vec<usize> = (start..end)
                    .filter(|&i| ssa_ops[i].opcode != Opcode::Barrier)
                    .collect();
                if !global_indices.is_empty() {
                // Section's hint is current merged state (greedy for this section until we run CP-SAT)
                // Greedy section makespan and peak (for delta reporting after feasibility)
                let (greedy_min_c, greedy_max_c) = global_indices
                    .iter()
                    .filter_map(|&i| {
                        let c = ssa_ops[i].cycle;
                        if c >= 0 {
                            Some(c)
                        } else {
                            None
                        }
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
                    global_indices.iter().map(|&i| ssa_ops[i].id).collect();
                let segment_ops_raw: Vec<SsaOp> = global_indices
                    .iter()
                    .map(|&i| {
                        let mut op = ssa_ops[i].clone();
                        op.dep_ids.retain(|id| segment_op_ids.contains(id));
                        op
                    })
                    .collect();

                // Filter out AddImm alternatives from segment_ops (same logic as full-kernel CP-SAT)
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
                // Filter segment_ops and corresponding hints together
                let filtered_segment: Vec<(SsaOp, i32)> = segment_ops_raw
                    .into_iter()
                    .zip(global_indices.iter().map(|&i| {
                        let c = ssa_ops[i].cycle;
                        if c >= 0 {
                            (c - section_offset).max(0)
                        } else {
                            c
                        }
                    }))
                    .enumerate()
                    .filter(|(idx, _)| !seg_addimm_alternatives.contains(idx))
                    .map(|(_, pair)| pair)
                    .collect();
                let (segment_ops, segment_hints): (Vec<SsaOp>, Vec<i32>) =
                    filtered_segment.into_iter().unzip();

                // Indices into segment_ops must match what live_in_for_section_filtered returns:
                // pass the filtered global indices so last_use_idx is in [0, segment_ops.len()).
                let filtered_global_indices: Vec<usize> = global_indices
                    .iter()
                    .enumerate()
                    .filter(|(idx, _)| !seg_addimm_alternatives.contains(idx))
                    .map(|(_, &g)| g)
                    .collect();

                let initial_live = live_in_for_section_filtered(
                    &ssa_ops,
                    &boundaries,
                    k,
                    &vreg_def_sec,
                    &filtered_global_indices,
                );
                let live_out = live_out_from_section(&ssa_ops, &boundaries, k, &vreg_def_sec);
                let live_through =
                    live_through_for_section(&ssa_ops, &boundaries, k, &vreg_def_sec);
                // Per-section never uses VALU shift (avoids extra scratch pressure in segment).
                // Feasibility: we require a feasible solution; give enough time and stop as soon as we find one.
                // SSA CP-SAT uses num_workers=1 when stop_after_first_solution is true so the solver
                // actually stops at the first feasible (multi-worker would keep improving past it).
                let (section_time_limit, section_stop_first) = match per_section_optimize_secs {
                    Some(t) if t > 0.0 => (t, false), // Short optimization run
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
                        (limit, true) // Feasibility: stop after first solution
                    }
                };
                let config_feasibility = SsaCpsatConfig {
                    time_limit_secs: section_time_limit,
                    verbose,
                    stop_after_first_solution: section_stop_first,
                    enable_valu_shift: false, // Per-section: no VALU shift
                    scratch_budget: effective_budget,
                    model_name: Some(format!("section_{}_feasibility", k)),
                    ..Default::default()
                };
                // Pass which segment ops were shifted in the full-kernel greedy so hint verification
                // counts them as ALU (merged hint must not violate slot capacity).
                let seg_shift_set: HashSet<usize> = greedy_shifted
                    .intersection(&segment_op_ids)
                    .copied()
                    .collect();
                let seg_shift: Option<&HashSet<usize>> =
                    if seg_shift_set.is_empty() {
                        None
                    } else {
                        Some(&seg_shift_set)
                    };
                // Hint this section with current merged choices (greedy for this section until we run and update).
                let seg_select_set: HashSet<usize> =
                    current_select_fma.intersection(&segment_op_ids).copied().collect();
                let seg_select = if seg_select_set.is_empty() {
                    None
                } else {
                    Some(&seg_select_set)
                };
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
                        live_out_vregs: if live_out.is_empty() {
                            None
                        } else {
                            Some(&live_out)
                        },
                        live_through_vregs: if live_through.is_empty() {
                            None
                        } else {
                            Some(&live_through)
                        },
                    },
                    Some(false), // section exceeded budget, so greedy was infeasible
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
                // Update merged hint with this section's solution so the final model gets it as hint.
                for &i in &global_indices {
                    current_select_fma.remove(&ssa_ops[i].id);
                }
                for &id in &seg_result.select_fma_ops {
                    current_select_fma.insert(id);
                }
                let (start, end) = boundaries[k];
                for op in &ssa_ops[start..end] {
                    if op.opcode == Opcode::ConstChoice {
                        current_const_add_imm.remove(&op.id);
                    }
                }
                for &id in &seg_result.const_add_imm_ops {
                    current_const_add_imm.insert(id);
                }
                // Write back cycles: this section's solution (segment result) is now the hint for this section.
                for (seg_idx, &cycle) in seg_result.cycle_assignments.iter().enumerate() {
                    if seg_idx < filtered_global_indices.len() && cycle >= 0 {
                        ssa_ops[filtered_global_indices[seg_idx]].cycle = section_offset + cycle;
                    }
                }
                // Rebase ops not in segment (e.g. AddImm alternatives) so section is strictly [section_offset, section_offset + span)
                let filtered_set: HashSet<usize> =
                    filtered_global_indices.iter().copied().collect();
                for &i in &global_indices {
                    if !filtered_set.contains(&i) && ssa_ops[i].cycle >= 0 {
                        ssa_ops[i].cycle =
                            section_offset + (ssa_ops[i].cycle - greedy_min_c);
                    }
                }
                this_section_makespan = Some(
                    (seg_result.makespan as i32).max(greedy_section_makespan),
                );
                if verbose {
                    let after_budget = sections_exceeding_budget(
                        &ssa_ops,
                        effective_scratch_budget(zero_vreg.is_some()),
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
            // Advance section_offset by this section's span. We already rebased at start of
            // iteration; if we ran per-section CP-SAT the writeback may have changed the span.
            let advance = this_section_makespan.unwrap_or(section_span_after_rebase);
            section_offset += advance;
        }

        // Presolve guarantee: merged schedule must be within scratch budget.
        // Use per-section SelectChoice (segment result) so peak matches segment scratch model.
        let post_merge_budget = sections_exceeding_budget(
            &ssa_ops,
            SCRATCH_BUDGET_HW,
            Some(&greedy_select_fma_ops),
            Some(&section_select_fma_overrides),
        );
        if let Some((sec, peak)) = post_merge_budget
            .section_peaks
            .iter()
            .enumerate()
            .find(|(_, &p)| p > SCRATCH_BUDGET_HW)
        {
            return Err(format!(
                "Per-section CP-SAT produced a merged schedule that exceeds scratch budget: section {} has peak {} words (budget {}). \
                 This indicates a bug in the segment scratch model (e.g. live-through or live-out).",
                sec, *peak, SCRATCH_BUDGET_HW
            ));
        }

        // Op IDs in sections that ran per-section CP-SAT: their cycles come from the segment (no shift).
        // Full-kernel hint verification must not count them as shifted even if they were in greedy_shifted.
        let rescheduled_by_section_cpsat: HashSet<usize> = boundaries
            .iter()
            .enumerate()
            .filter(|(k, _)| budget_result.section_exceeds_budget[*k])
            .flat_map(|(_, &(start, end))| ssa_ops[start..end].iter().map(|op| op.id))
            .collect();

        // Elide barriers before CP-SAT
        // CP-SAT has proper liveness constraints via ILP, so barriers are unnecessary
        // and would interfere with its optimization
        let mut ssa_ops = elide_barriers(ssa_ops);

        // Collect hints for full-kernel CP-SAT: merged schedule (greedy for within-budget
        // sections, per-section feasibility result for sections that exceeded budget).
        let greedy_hints: Vec<i32> = ssa_ops.iter().map(|op| op.cycle).collect();
        let hint_makespan = greedy_hints
            .iter()
            .copied()
            .max()
            .map(|c| c + 1)
            .unwrap_or(0);
        let effective_budget_hint = effective_scratch_budget(zero_vreg.is_some());
        // After elision there is one section; use current choice hints (greedy + per-section updates).
        let hint_peak = sections_exceeding_budget(
            &ssa_ops,
            effective_budget_hint,
            Some(&current_select_fma),
            None,
        )
        .section_peaks
        .iter()
        .copied()
        .max()
        .unwrap_or(0);

        if hint_peak > SCRATCH_BUDGET_HW {
            return Err(format!(
                "Merged hint schedule exceeds scratch budget: peak {} words (budget {}). \
                 Presolves must guarantee a feasible hint; this indicates a bug in the segment scratch model.",
                hint_peak, SCRATCH_BUDGET_HW
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
            enable_vbroadcast_elimination,
            scratch_budget: effective_scratch_budget(zero_vreg.is_some()),
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

        // Pass shift hints if VALU shift is enabled. Exclude ops that were re-scheduled by per-section
        // CP-SAT (their cycles come from the segment, which doesn't use shift; counting them as shifted
        // would overcount ALU in hint verification).
        let full_kernel_shift_set: HashSet<usize> = if enable_valu_shift {
            greedy_shifted
                .difference(&rescheduled_by_section_cpsat)
                .copied()
                .collect()
        } else {
            HashSet::new()
        };
        let shift_hints = if full_kernel_shift_set.is_empty() {
            None
        } else {
            Some(&full_kernel_shift_set)
        };

        let greedy_hints_opt = Some(&greedy_hints);

        // Pass SelectChoice hints: use current choices (maintained from greedy + per-section CP-SAT).
        let op_ids_after_elision: HashSet<usize> = ssa_ops.iter().map(|op| op.id).collect();
        let mut select_choice_hints_set = HashSet::new();
        for &select_choice_id in &current_select_fma {
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
            &ssa_ops,
            cpsat_config,
            SsaCpsatHints {
                greedy_hints: greedy_hints_opt.map(|v| v.as_slice()),
                shift_hints,
                forced_shifts: forced_shifts_ref,
                select_choice_hints,
                const_add_imm_hints: Some(&current_const_add_imm),
                eliminate_vbroadcast_hints: None,
                initial_live: None,
                live_out_vregs: None,
                live_through_vregs: None,
            },
            Some(!greedy_budget_exceeded), // same as used for "Greedy exceeded budget" report above
        )
        .ok_or_else(|| "CP-SAT scheduling failed".to_string())?;

        // Apply cycle assignments
        for (idx, op) in ssa_ops.iter_mut().enumerate() {
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

        let shifted_ops = cpsat_result.shifted_ops;

        if verbose {
            eprintln!(
                "SSA CP-SAT result: {} cycles (greedy was {}), {} VALU ops shifted",
                cycles,
                greedy_cycles,
                shifted_ops.len()
            );
            let const_total = ssa_ops
                .iter()
                .filter(|o| o.opcode == Opcode::ConstChoice)
                .count();
            let const_add_imm = cpsat_result.const_add_imm_ops.len();
            let const_load = const_total.saturating_sub(const_add_imm);
            let select_total = ssa_ops
                .iter()
                .filter(|o| o.opcode == Opcode::SelectChoice)
                .count();
            let select_fma = cpsat_result.select_fma_ops.len();
            let select_vselect = select_total.saturating_sub(select_fma);
            eprintln!(
                "  ConstChoice: {} AddImm, {} Load | SelectChoice: {} FMA, {} vselect",
                const_add_imm, const_load, select_fma, select_vselect
            );
        }

        // Capture SSA-level schedule for bulk-move simulator (same index space as schedule_constraints).
        let ssa_cycle_assignments = cpsat_result.cycle_assignments.clone();
        let ssa_ops_bulk_move: Vec<(usize, Vec<usize>)> = ssa_ops
            .iter()
            .map(|o| (o.id, o.dep_ids.iter().copied().collect()))
            .collect();

        // Filter out dead stores (ops with cycle=-1) before address assignment.
        // Dead stores are ops whose dest VReg has no users - they were detected
        // during greedy hints generation but CP-SAT may also leave some ops
        // unscheduled. Including them in address assignment would corrupt live
        // interval computation.
        let ssa_ops: Vec<SsaOp> = ssa_ops.into_iter().filter(|op| op.cycle >= 0).collect();

        // Assign addresses via linear scan (zero_vreg included in peak so we stay within 1536)
        let mut assign_config = AddressAssignConfig::default();
        assign_config.budget = SCRATCH_BUDGET_HW;

        if let Some(zero) = zero_vreg {
            assign_config.fixed_vregs.insert(zero, 0);
            assign_config.zero_vreg = Some(zero);
        }
        // SelectChoice FMA (no delta): sub at C, FMA at C+1 → result at C+2; address assigner uses def_cycle+2 for dest
        assign_config.select_fma_ops = Some(cpsat_result.select_fma_ops.clone());
        // Shifted VALU: 8 ALU ops write at different cycles; def_cycle = min(alu_cycles) for correct allocation
        assign_config.shifted_alu_cycles = Some(cpsat_result.shifted_alu_cycles.clone());
        // AluDerive: producer VRegs must stay alive until the AluDerive op reads them
        assign_config.const_alu_derive_ops = Some(cpsat_result.const_alu_derive_ops.clone());

        let address_result = assign_addresses(&ssa_ops, &assign_config);
        let budget_exceeded = address_result.budget_exceeded;

        // Assert no two VRegs with overlapping live intervals share the same address.
        // Uses intervals already computed in assign_addresses (no extra cost).
        let intervals = &address_result.live_intervals;
        for (v1, addr1) in &address_result.vreg_to_addr {
            for (v2, addr2) in &address_result.vreg_to_addr {
                if v1 >= v2 {
                    continue;
                }
                if addr1 != addr2 {
                    continue;
                }
                if let (Some(i1), Some(i2)) = (intervals.get(v1), intervals.get(v2)) {
                    assert!(
                        !i1.overlaps(i2),
                        "BUG: VRegs {:?} and {:?} share address {} but intervals [{},{}] and [{},{}] overlap",
                        v1, v2, addr1, i1.start, i1.end, i2.start, i2.end
                    );
                }
            }
        }

        if budget_exceeded {
            // CP-SAT should respect budget, so this indicates a bug in CP-SAT solver
            eprintln!(
                "Warning: CP-SAT address assignment exceeded budget: peak_live={} > budget={}",
                address_result.peak_live, address_result.budget
            );
            return Err(format!(
                "scratch budget exceeded: peak_live={} > budget={}",
                address_result.peak_live, address_result.budget
            ));
        }

        // Compute usage accounting from SSA ops
        let mut usage_accounting = UsageAccounting::from_ssa_ops(&ssa_ops);
        usage_accounting.set_peak_live(address_result.peak_live);

        // Build VReg metadata for liveness debugging (before lowering loses SSA info)
        let vreg_metadata = build_vreg_metadata(&ssa_ops);

        // Lower to concrete ops, expanding shifted VALU ops to ALU
        // Optional: force all ConstChoice to Load (for debugging CP-SAT validation vs greedy)
        let empty_const: HashSet<usize> = HashSet::new();
        let const_add_imm_ops = if std::env::var("SCHEDULER_FORCE_CONST_LOAD").is_ok() {
            &empty_const
        } else {
            &cpsat_result.const_add_imm_ops
        };
        let valu_shifted = shifted_ops.len();
        let ops = lower_ssa_ops_with_shift(
            &ssa_ops,
            &address_result.vreg_to_addr,
            &shifted_ops,
            const_add_imm_ops,
            &cpsat_result.const_alu_derive_ops,
            &cpsat_result.select_fma_ops, // CP-SAT decisions for SelectChoice
            &cpsat_result.shifted_alu_cycles, // Individual ALU cycles from CP-SAT
            &cpsat_result.eliminated_vbroadcast_ops,
            &cpsat_result.eliminated_vb_to_scalar_producer,
        );

        // Debug: assert no two ops in the same cycle write overlapping scratch addresses.
        if cfg!(debug_assertions) {
            verify_no_same_cycle_scratch_write_conflict(&ops);
        }

        // Cycle count must be derived from lowered ops: SelectChoice FMA (no delta) emits
        // sub at C and FMA at C+1, so if the last SSA op was a SelectChoice FMA at C,
        // CP-SAT makespan is C+1 but we need C+2 cycles for the FMA to be emitted.
        let cycles = ops
            .iter()
            .map(|op| op.cycle.max(0) as usize)
            .max()
            .map(|m| m + 1)
            .unwrap_or(cycles)
            .max(cycles);

        Ok(SsaScheduleResult {
            ops,
            cycles,
            address_result,
            vreg_count,
            valu_shifted,
            budget_exceeded, // CP-SAT should not exceed budget
            usage_accounting,
            vreg_metadata,
            shifted_alu_cycles: cpsat_result.shifted_alu_cycles.clone(),
            empty_cycle_count: Some(cpsat_result.empty_cycle_count),
            makespan_driven_by_capacity: Some(cpsat_result.makespan_driven_by_capacity),
            schedule_constraints: Some(cpsat_result.schedule_constraints.clone()),
            ssa_cycle_assignments: Some(ssa_cycle_assignments),
            ssa_ops_bulk_move: Some(ssa_ops_bulk_move),
            eliminated_vbroadcast_ops: cpsat_result.eliminated_vbroadcast_ops.iter().copied().collect(),
            lb_cycles: cpsat_result.lb_cycles,
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
