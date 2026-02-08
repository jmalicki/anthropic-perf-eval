//! SSA greedy scheduler with budget and VALU→ALU shifting support.
//!
//! This module contains the greedy scheduling algorithm that schedules
//! SSA ops while respecting slot limits and scratch budget constraints.
//!
//! Key features:
//! - LUC (Last-Use-Count) heuristic for prioritizing ops that free scratch
//! - Same-cycle reuse model for accurate liveness tracking
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

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::types::{EmissionPhase, Engine, Opcode, Operand, PrecedenceOffset, SsaOp, VReg, VLEN};

use super::ssa_lower::can_ssa_valu_shift_to_alu;

/// Dump usage breakdown for currently live VRegs when budget is exceeded.
///
/// This helps diagnose which rounds/var_types are consuming the most scratch
/// when the scheduler is forced to exceed the budget.
pub(super) fn dump_live_usage_breakdown(
    ops: &[SsaOp],
    live_vregs: &HashSet<VReg>,
    cycle: usize,
    live_words: usize,
    scratch_budget: usize,
) {
    use std::collections::BTreeMap;

    eprintln!("\n=== Budget Exceeded at Cycle {} ===", cycle);
    eprintln!(
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
    eprintln!("\nLive by Round:");
    for (round, (count, words)) in &by_round {
        let round_str = if *round < 0 {
            "init".to_string()
        } else {
            format!("{}", round)
        };
        eprintln!(
            "  Round {:>4}: {:>4} vregs, {:>5} words",
            round_str, count, words
        );
    }

    // Print by var_type
    eprintln!("\nLive by VarType:");
    let mut type_vec: Vec<_> = by_type.iter().collect();
    type_vec.sort_by(|a, b| b.1 .1.cmp(&a.1 .1)); // Sort by words descending
    for (var_type, (count, words)) in type_vec {
        eprintln!("  {:>15}: {:>4} vregs, {:>5} words", var_type, count, words);
    }

    // Print by phase
    eprintln!("\nLive by Phase:");
    let mut phase_vec: Vec<_> = by_phase.iter().collect();
    phase_vec.sort_by(|a, b| b.1 .1.cmp(&a.1 .1)); // Sort by words descending
    for (phase, (count, words)) in phase_vec {
        eprintln!("  {:>20}: {:>4} vregs, {:>5} words", phase, count, words);
    }

    eprintln!("=== End Budget Exceeded Dump ===\n");
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
        None,
    )
}

/// Schedule SSA ops with SelectChoice expansion support.
/// Returns (cycles, budget_exceeded, peak_live)
///
/// **VALU→ALU shifting (design choice):** We do not prefer shifting in greedy. We only shift
/// when VALU slots are full (6/6); otherwise we keep the op on VALU. Shift decisions are left
/// to CP-SAT, which optimizes them globally. Greedy provides cycle hints; CP-SAT decides which
/// ops to shift.
///
/// SelectChoice tracking is self-contained: ops report their relationship via methods.
/// The sub and fma ops know they belong to a SelectChoice op through `select_choice_parent`.
pub(super) fn schedule_ssa_greedy_with_shift_budget_and_selectchoice(
    ops: &mut [SsaOp],
    enable_valu_shift: bool,
    shifted_ops: &mut HashSet<usize>,
    scratch_budget: usize,
    select_choice_fma_ops: Option<&mut HashSet<usize>>,
    select_sub_cycles: Option<&mut HashMap<usize, i32>>,
    select_fma_cycles: Option<&mut HashMap<usize, i32>>,
    mut const_add_imm_ops: Option<&mut HashSet<usize>>,
    asap_alap: Option<(&HashMap<usize, i32>, &HashMap<usize, i32>)>,
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
        // Still track users from AddImm alternatives (they read zero_vreg)
        for src in &op.srcs {
            if let Operand::VReg(vreg) = src {
                vreg_users.entry(*vreg).or_default().insert(idx);
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

    let mut vreg_remaining_users: HashMap<VReg, usize> = vreg_users
        .iter()
        .map(|(vreg, users)| (*vreg, users.len()))
        .collect();
    let mut live_words: usize = 0;
    let mut live_vregs: HashSet<VReg> = HashSet::new();

    // Budget tracking
    let mut peak_live_words: usize = 0;
    let mut budget_exceeded = false;
    let mut last_budget_dump_cycle: Option<usize> = None;

    // Track when each VReg was defined (used for debugging/validation)
    let mut vreg_def_cycle: HashMap<VReg, i32> = HashMap::new();

    // Ready heap: (priority, op_idx) - min-heap via Reverse
    // Priority is the earliest cycle this op can be scheduled (ASAP)
    let mut ready_heap: BinaryHeap<Reverse<(i32, usize)>> = BinaryHeap::new();

    // Initialize with ops that have no dependencies (excluding dead stores)
    for (idx, op) in ops.iter().enumerate() {
        if remaining_deps[idx] == 0 && !dead_store_indices.contains(&idx) {
            let priority = asap_alap
                .as_ref()
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

    while scheduled.len() < ops.len() && cycle < max_cycles {
        // Note: scheduled includes dead stores (pre-marked), so we check against ops.len()
        // Free expired VRegs (O(log n) per VReg)
        // VRegs expire at the cycle AFTER their last use
        while let Some(&Reverse((expiry_cycle, vreg))) = vreg_expiry_heap.peek() {
            if expiry_cycle <= cycle as i32 {
                vreg_expiry_heap.pop();
                if live_vregs.remove(&vreg) {
                    live_words = live_words.saturating_sub(vreg.size());
                }
            } else {
                break;
            }
        }

        let mut load_used = 0;
        let mut store_used = 0;
        let mut alu_used = 0;
        let mut valu_used = 0;
        let mut flow_used = 0;

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
                let new_priority = asap_alap
                    .as_ref()
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

        // Sort ready_this_cycle by LUC (descending) as tie-breaker for ASAP.
        // Since heap already sorted by ASAP, stable sort preserves that ordering.
        ready_this_cycle.sort_by(|&a, &b| {
            compute_luc(ops, &vreg_remaining_users, b).cmp(&compute_luc(
                ops,
                &vreg_remaining_users,
                a,
            ))
        });

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

            // Defer ops that are ready by deps but before their ASAP when bounds are provided
            let mut deferred_by_bounds: Vec<(i32, usize)> = Vec::new();
            let mut ready_for_try: Vec<usize> = Vec::new();
            for idx in ready_this_cycle.drain(..) {
                if let Some((asap, _)) = asap_alap {
                    let lb = *asap.get(&ops[idx].id).unwrap_or(&0);
                    if (cycle as i32) < lb {
                        deferred_by_bounds.push((lb, idx));
                        continue;
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
                let engine = if ops[idx].opcode == Opcode::ConstChoice {
                    if flow_used < 1 && flow_other_ready == 0 {
                        Engine::Flow
                    } else {
                        Engine::Load
                    }
                } else {
                    ops[idx].engine
                };
                let dest = ops[idx].dest;
                let op_id = ops[idx].id;
                let srcs: Vec<Operand> = ops[idx].srcs.clone();

                let can_shift = enable_valu_shift
                    && engine == Engine::Valu
                    && can_ssa_valu_shift_to_alu(&ops[idx]);

                // Check slot availability
                let fits_slots = match engine {
                    Engine::Load => load_used < 2,
                    Engine::Store => store_used < 2,
                    Engine::Alu => alu_used < 12,
                    Engine::Valu => valu_used < 6 || (can_shift && alu_used + VLEN <= 12),
                    Engine::Flow => flow_used < 1,
                    _ => true,
                };

                if !fits_slots {
                    // Reinsert to heap for next cycle
                    let next_priority = asap_alap
                        .as_ref()
                        .and_then(|(a, _)| a.get(&op_id).copied())
                        .map(|lb| ((cycle + 1) as i32).max(lb))
                        .unwrap_or_else(|| (cycle + 1) as i32);
                    ready_heap.push(Reverse((next_priority, idx)));
                    continue;
                }

                // Check scratch budget with same-cycle reuse model:
                // VRegs that will be freed when this op runs (their last use) reduce pressure.
                let words_added = dest.map_or(0, |d| d.size());
                let mut words_freed = 0usize;
                for src in &srcs {
                    if let Operand::VReg(vreg) = src {
                        // If this is the last use of vreg, it will be freed after this op
                        if let Some(&remaining) = vreg_remaining_users.get(vreg) {
                            if remaining == 1 && live_vregs.contains(vreg) {
                                words_freed += vreg.size();
                            }
                        }
                    }
                }
                // Net pressure change: add new VReg, subtract freed VRegs (same-cycle reuse)
                let projected_live = live_words + words_added - words_freed;
                if projected_live > scratch_budget {
                    // Defer for retry - liveness may decrease after other ops scheduled
                    budget_deferred.push(idx);
                    continue;
                }

                // Schedule the op
                //
                // Design choice: we do NOT prefer VALU→ALU shifting in greedy. We only shift when
                // VALU is full (valu_used >= 6). Shift decisions are left to CP-SAT, which can
                // optimize them globally; greedy just provides cycle hints and shifts only when
                // necessary to make progress.
                match engine {
                    Engine::Load => load_used += 1,
                    Engine::Store => store_used += 1,
                    Engine::Alu => alu_used += 1,
                    Engine::Valu => {
                        if can_shift && valu_used >= 6 {
                            alu_used += VLEN;
                            shifted_ops.insert(op_id);
                        } else {
                            valu_used += 1;
                        }
                    }
                    Engine::Flow => {
                        flow_used += 1;
                        // Record ConstChoice scheduled as add_imm for lowering
                        if ops[idx].opcode == Opcode::ConstChoice {
                            if let Some(ref mut set) = const_add_imm_ops {
                                set.insert(op_id);
                            }
                        }
                    }
                    _ => {}
                }

                let assigned_cycle = if let Some((asap, alap)) = asap_alap {
                    let lb = *asap.get(&op_id).unwrap_or(&0);
                    let ub = *alap.get(&op_id).unwrap_or(&i32::MAX);
                    (cycle as i32).clamp(lb, ub)
                } else {
                    cycle as i32
                };
                ops[idx].cycle = assigned_cycle;
                scheduled.insert(idx);
                made_progress = true;

                // Update live tracking for destination VReg
                if let Some(d) = dest {
                    if vreg_remaining_users.get(&d).map_or(false, |&n| n > 0) {
                        live_vregs.insert(d);
                        live_words += d.size();
                        peak_live_words = peak_live_words.max(live_words);
                    }
                    // Track when this VReg was defined
                    vreg_def_cycle.insert(d, cycle as i32);
                }

                // Update remaining users and last use cycle for source VRegs
                // With same-cycle reuse model: VRegs can be freed immediately after their
                // last use is scheduled, allowing other ops in the same cycle to reuse the space.
                for src in &srcs {
                    if let Operand::VReg(vreg) = src {
                        // Decrement remaining users count
                        if let Some(remaining) = vreg_remaining_users.get_mut(vreg) {
                            *remaining = remaining.saturating_sub(1);
                            // Free VReg immediately when last user is scheduled (same-cycle reuse)
                            // This is safe because reads happen before writes within a cycle.
                            if *remaining == 0 && live_vregs.contains(vreg) {
                                live_vregs.remove(vreg);
                                live_words = live_words.saturating_sub(vreg.size());
                                // Add to expiry heap for cleanup (in case it gets re-added)
                                // Expiry is at next cycle
                                vreg_expiry_heap.push(Reverse(((cycle + 1) as i32, *vreg)));
                            }
                        }
                    }
                }

                // Add newly-ready consumers to heap (O(log n) per consumer)
                for &consumer_idx in &consumers[idx] {
                    // Update max ready cycle from this predecessor's offset
                    let offset = ops[idx]
                        .precedence_offset_for_asap(shiftable_set.contains(&idx));
                    max_ready_cycle[consumer_idx] =
                        max_ready_cycle[consumer_idx].max((cycle as i32) + offset);

                    remaining_deps[consumer_idx] -= 1;
                    if remaining_deps[consumer_idx] == 0 {
                        let priority = asap_alap
                            .as_ref()
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
            let engine = ops[force_idx].engine;
            let dest = ops[force_idx].dest;
            let op_id = ops[force_idx].id;
            let srcs: Vec<Operand> = ops[force_idx].srcs.clone();

            // Check slot availability - we still respect slot limits
            let can_shift = enable_valu_shift
                && engine == Engine::Valu
                && can_ssa_valu_shift_to_alu(&ops[force_idx]);

            let fits_slots = match engine {
                Engine::Load => load_used < 2,
                Engine::Store => store_used < 2,
                Engine::Alu => alu_used < 12,
                Engine::Valu => valu_used < 6 || (can_shift && alu_used + VLEN <= 12),
                Engine::Flow => flow_used < 1,
                _ => true,
            };

            if fits_slots {
                // CRITICAL: Force schedule, ignoring budget constraint - this exceeds budget!
                // This is intentional behavior - DO NOT REMOVE unless explicitly instructed.
                // The force-schedule allows progress when retry loops fail to free enough
                // VRegs. The budget_exceeded flag tracks this for reporting.
                //
                // Dump usage breakdown once per cycle with budget violation
                if last_budget_dump_cycle != Some(cycle) {
                    dump_live_usage_breakdown(
                        ops,
                        &live_vregs,
                        cycle,
                        live_words,
                        scratch_budget,
                    );
                    last_budget_dump_cycle = Some(cycle);
                }
                budget_exceeded = true;
                #[allow(unused_assignments)]
                // Same design as normal path: shift only when VALU full, not by preference
                match engine {
                    Engine::Load => load_used += 1,
                    Engine::Store => store_used += 1,
                    Engine::Alu => alu_used += 1,
                    Engine::Valu => {
                        if can_shift && valu_used >= 6 {
                            alu_used += VLEN;
                            shifted_ops.insert(op_id);
                        } else {
                            valu_used += 1;
                        }
                    }
                    Engine::Flow => flow_used += 1,
                    _ => {}
                }

                let force_cycle = if let Some((asap, alap)) = asap_alap {
                    let lb = *asap.get(&op_id).unwrap_or(&0);
                    let ub = *alap.get(&op_id).unwrap_or(&i32::MAX);
                    (cycle as i32).clamp(lb, ub)
                } else {
                    cycle as i32
                };
                ops[force_idx].cycle = force_cycle;
                scheduled.insert(force_idx);

                // Update liveness (same as normal scheduling path)
                if let Some(dest_vreg) = dest {
                    live_vregs.insert(dest_vreg);
                    live_words += dest_vreg.size();
                    peak_live_words = peak_live_words.max(live_words);
                    vreg_def_cycle.insert(dest_vreg, force_cycle);
                }

                for src in &srcs {
                    if let Operand::VReg(vreg) = src {
                        if let Some(remaining) = vreg_remaining_users.get_mut(vreg) {
                            *remaining = remaining.saturating_sub(1);
                            if *remaining == 0 && live_vregs.contains(vreg) {
                                live_vregs.remove(vreg);
                                live_words = live_words.saturating_sub(vreg.size());
                                vreg_expiry_heap.push(Reverse(((cycle + 1) as i32, *vreg)));
                            }
                        }
                    }
                }

                for &consumer_idx in &consumers[force_idx] {
                    let offset = ops[force_idx]
                        .precedence_offset_for_asap(shiftable_set.contains(&force_idx));
                    max_ready_cycle[consumer_idx] =
                        max_ready_cycle[consumer_idx].max((cycle as i32) + offset);

                    remaining_deps[consumer_idx] -= 1;
                    if remaining_deps[consumer_idx] == 0 {
                        let priority = asap_alap
                            .as_ref()
                            .and_then(|(a, _)| a.get(&ops[consumer_idx].id).copied())
                            .map(|lb| max_ready_cycle[consumer_idx].max(lb))
                            .unwrap_or(max_ready_cycle[consumer_idx]);
                        ready_heap.push(Reverse((priority, consumer_idx)));
                    }
                }
            }
        }

        // Push remaining budget-deferred ops back to heap for next cycle
        for idx in budget_deferred {
            let priority = asap_alap
                .as_ref()
                .and_then(|(a, _)| a.get(&ops[idx].id).copied())
                .map(|lb| ((cycle + 1) as i32).max(lb))
                .unwrap_or_else(|| (cycle + 1) as i32);
            ready_heap.push(Reverse((priority, idx)));
        }

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

    if budget_exceeded {
        eprintln!(
            "Greedy scheduler exceeded budget: peak_live={} > budget={} (force-scheduled to break deadlock)",
            peak_live_words, scratch_budget
        );
    }

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
        let mut load_used = 0;
        let mut store_used = 0;
        let mut alu_used = 0;
        let mut valu_used = 0;
        let mut flow_used = 0;

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

            let fits = match engine {
                Engine::Load => {
                    if load_used < 2 {
                        load_used += 1;
                        true
                    } else {
                        false
                    }
                }
                Engine::Store => {
                    if store_used < 2 {
                        store_used += 1;
                        true
                    } else {
                        false
                    }
                }
                Engine::Alu => {
                    if alu_used < 12 {
                        alu_used += 1;
                        true
                    } else {
                        false
                    }
                }
                Engine::Valu => {
                    if valu_used < 6 {
                        valu_used += 1;
                        true
                    } else if can_shift && alu_used + VLEN <= 12 {
                        // Design choice: shift only when VALU full; we do not prefer shifting in greedy (CP-SAT decides).
                        alu_used += VLEN;
                        shifted_ops.insert(op.id);
                        true
                    } else {
                        false
                    }
                }
                Engine::Flow => {
                    if flow_used < 1 {
                        flow_used += 1;
                        true
                    } else {
                        false
                    }
                }
                _ => true,
            };

            if fits {
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
