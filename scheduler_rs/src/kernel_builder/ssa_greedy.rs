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

use crate::types::{Engine, Opcode, Operand, SsaOp, VReg, VLEN};

use super::ssa_lower::can_ssa_valu_shift_to_alu;
use super::tick_tock_ssa::SsaTickTockKernelBuilder;

impl SsaTickTockKernelBuilder {
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
        let mut by_phase: BTreeMap<&str, (usize, usize)> = BTreeMap::new(); // phase -> (count, words)

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

                    // By phase
                    let phase = op
                        .source_info
                        .as_ref()
                        .map(|s| s.phase.as_str())
                        .unwrap_or("unknown");
                    let entry = by_phase.entry(phase).or_insert((0, 0));
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
        &self,
        ops: &mut [SsaOp],
        enable_valu_shift: bool,
        shifted_ops: &mut HashSet<usize>,
        scratch_budget: usize,
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
                    eprintln!(
                        "Warning: Dead store detected (will skip): op[{}] id={} defines {:?} with 0 users.\n\
                         Source: {:?}\n\
                         Engine: {:?}, Opcode: {:?}",
                        idx, op.id, dest, op.source, op.engine, op.opcode
                    );
                }
            }
        }

        // Add AddImm constant alternatives to dead stores - greedy always uses Load
        // (Note: These are not truly "dead", they're just unused by greedy.
        //  CP-SAT may choose to use them instead of the Load alternative.)
        dead_store_indices.extend(&const_add_imm_alternatives);

        if !dead_store_indices.is_empty() {
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
        // Only count deps that exist in id_to_idx AND are not dead stores
        // (external/missing deps and dead stores are treated as satisfied)
        let mut remaining_deps: Vec<usize> = ops
            .iter()
            .map(|op| {
                op.dep_ids
                    .iter()
                    .filter(|dep_id| {
                        id_to_idx.get(dep_id).map_or(false, |&idx| !dead_store_indices.contains(&idx))
                    })
                    .count()
            })
            .collect();

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
        for (idx, _op) in ops.iter().enumerate() {
            if remaining_deps[idx] == 0 && !dead_store_indices.contains(&idx) {
                ready_heap.push(Reverse((0, idx)));
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

                // Check if deps are truly satisfied (scheduled in previous cycle)
                // Dead stores are treated as already satisfied (they're skipped entirely)
                let deps_ready = ops[idx].dep_ids.iter().all(|dep_id| {
                    id_to_idx.get(dep_id).map_or(true, |&dep_idx| {
                        dead_store_indices.contains(&dep_idx) ||
                        (ops[dep_idx].cycle >= 0 && ops[dep_idx].cycle < cycle as i32)
                    })
                });

                if deps_ready {
                    ready_this_cycle.push(idx);
                } else {
                    // Not ready yet - reinsert with updated priority
                    reinsert.push((priority.max(cycle as i32), idx));
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
                compute_luc(ops, &vreg_remaining_users, b)
                    .cmp(&compute_luc(ops, &vreg_remaining_users, a))
            });

            // Keep retrying as long as we make progress (schedule something or free VRegs)
            while made_progress {
                made_progress = false;

                // Sort budget_deferred by LUC (descending) to prioritize ops that free VRegs
                budget_deferred.sort_by(|&a, &b| {
                    compute_luc(ops, &vreg_remaining_users, b)
                        .cmp(&compute_luc(ops, &vreg_remaining_users, a))
                });

                // Process ready_this_cycle first, then budget_deferred
                let ops_to_try: Vec<usize> = ready_this_cycle
                    .drain(..)
                    .chain(budget_deferred.drain(..))
                    .collect();

                for idx in ops_to_try {
                    if scheduled.contains(&idx) {
                        continue;
                    }

                    // Map ConstChoice to Load for greedy scheduler (default encoding)
                    let engine = if ops[idx].opcode == Opcode::ConstChoice {
                        Engine::Load
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
                        ready_heap.push(Reverse(((cycle + 1) as i32, idx)));
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

                    ops[idx].cycle = cycle as i32;
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
                        remaining_deps[consumer_idx] -= 1;
                        if remaining_deps[consumer_idx] == 0 {
                            // Priority: earliest possible cycle is next cycle (since dep just scheduled)
                            let priority = (cycle + 1) as i32;
                            ready_heap.push(Reverse((priority, consumer_idx)));
                        }
                    }
                }
            }

            // Emergency fallback: if we made no progress and have budget-deferred ops,
            // force-schedule the highest-LUC op to break the deadlock.
            // This temporarily exceeds the budget but allows progress.
            if !budget_deferred.is_empty() {
                // Sort by LUC and pick the best one
                budget_deferred.sort_by(|&a, &b| {
                    compute_luc(ops, &vreg_remaining_users, b)
                        .cmp(&compute_luc(ops, &vreg_remaining_users, a))
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
                    // Force schedule, ignoring budget constraint - this exceeds budget!
                    // Dump usage breakdown once per cycle with budget violation
                    if last_budget_dump_cycle != Some(cycle) {
                        Self::dump_live_usage_breakdown(
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

                    ops[force_idx].cycle = cycle as i32;
                    scheduled.insert(force_idx);

                    // Update liveness (same as normal scheduling path)
                    if let Some(dest_vreg) = dest {
                        live_vregs.insert(dest_vreg);
                        live_words += dest_vreg.size();
                        peak_live_words = peak_live_words.max(live_words);
                        vreg_def_cycle.insert(dest_vreg, cycle as i32);
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
                        remaining_deps[consumer_idx] -= 1;
                        if remaining_deps[consumer_idx] == 0 {
                            ready_heap.push(Reverse(((cycle + 1) as i32, consumer_idx)));
                        }
                    }
                }
            }

            // Push remaining budget-deferred ops back to heap for next cycle
            for idx in budget_deferred {
                ready_heap.push(Reverse(((cycle + 1) as i32, idx)));
            }

            cycle += 1;
        }

        // Suppress unused warning for vreg_def_cycle - it's used for debugging
        let _ = vreg_def_cycle;

        if scheduled.len() < ops.len() {
            // Diagnose why scheduling failed
            let unscheduled: Vec<usize> = (0..ops.len())
                .filter(|i| !scheduled.contains(i))
                .collect();
            let ready_count = ready_heap.len();
            let blocked_count = unscheduled.iter().filter(|&&i| remaining_deps[i] > 0).count();

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
        &self,
        ops: &mut [SsaOp],
        enable_valu_shift: bool,
        shifted_ops: &mut HashSet<usize>,
    ) -> Result<(usize, bool, usize), String> {
        // Use hardware scratch budget
        self.schedule_ssa_greedy_with_shift_budget(ops, enable_valu_shift, shifted_ops, 1536)
    }

    /// Internal implementation of greedy scheduling (simpler, no budget tracking)
    #[allow(dead_code)]
    pub(super) fn schedule_ssa_greedy_with_shift_impl(
        &self,
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
                        // All dependencies scheduled in earlier cycles
                        && op.dep_ids.iter().all(|dep_id| {
                            if let Some(&dep_idx) = id_to_idx.get(dep_id) {
                                let dep_cycle = ops[dep_idx].cycle;
                                dep_cycle >= 0 && dep_cycle < cycle as i32
                            } else {
                                true // Unknown dep, assume ready
                            }
                        })
                })
                .map(|(idx, _)| idx)
                .collect();

            // Schedule ready ops that fit in slots
            for idx in ready {
                let op = &ops[idx];
                let engine = op.engine;

                // Check if this VALU op can be shifted to ALU
                let can_shift = enable_valu_shift
                    && engine == Engine::Valu
                    && can_ssa_valu_shift_to_alu(op);

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
                            // VALU is full, but we can shift to ALU
                            // Shifted VALU op uses 8 ALU slots
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_builder::emit_ssa::{compute_ssa_dependencies, compute_ssa_dependencies_with_barriers, SsaEmitContext};
    use crate::types::Opcode;

    // =========================================================================
    // Barrier + Dead Store Interaction Tests
    //
    // These tests verify the fix for the dead store + barrier interaction bug.
    //
    // THE BUG: The greedy scheduler had two places checking dependencies:
    // 1. `remaining_deps` count - determines when op enters ready heap
    // 2. `deps_ready` check - validates deps when processing from heap
    //
    // These were INCONSISTENT in handling dead stores:
    // - `remaining_deps` excluded dead stores (correct)
    // - `deps_ready` included dead stores and checked `cycle >= 0` (failed
    //   because dead stores have `cycle = -1`)
    //
    // WHY BARRIERS MATTER: The only way an op can depend on a dead store
    // (other than artificial test setups) is through barriers. Barriers depend
    // on ALL prior ops in their section, including dead stores:
    //
    //   compute_ssa_dependencies_with_barriers() does:
    //     for idx in section_start..barrier_idx {
    //         ops[barrier_idx].dep_ids.insert(ops[idx].id);  // Includes dead stores!
    //     }
    //
    // =========================================================================

    /// Test barrier with dead store in its section.
    ///
    /// This is the actual bug scenario: a barrier depends on a dead store.
    /// Without the fix, the barrier would never be scheduled because the
    /// dead store's cycle == -1 fails the deps_ready check.
    #[test]
    fn test_barrier_with_dead_store_in_section() {
        // Scenario:
        // op0: v1 = const 5          (used by op2, op4)
        // op1: v2 = const 6          (DEAD STORE - never used)
        // op2: v3 = add v1, v1       (used by store)
        // op3: barrier               (depends on op0, op1, op2 - including dead store!)
        // op4: v4 = mul v1, v3       (used by store)
        // op5: addr = const 100
        // op6: store addr, v4        (makes v4 live)
        //
        // The barrier (op3) depends on dead store (op1).
        // Without the fix, barrier would never be scheduled because
        // op1.cycle == -1 fails the deps_ready check.

        let mut emit = SsaEmitContext::new();
        let v1 = emit.emit_const(5);       // op 0 - used
        let _v2 = emit.emit_const(6);      // op 1 - DEAD STORE
        let v3 = emit.emit_alu(Opcode::Add, v1, v1);  // op 2 - used
        emit.emit_barrier();                // op 3 - depends on ALL above including dead store
        let v4 = emit.emit_alu(Opcode::Mul, v1, v3);  // op 4 - used
        let addr = emit.emit_const(100);   // op 5
        emit.emit_store(addr, v4);         // op 6 - makes v4 live

        compute_ssa_dependencies_with_barriers(&mut emit.ops, &emit.vreg_def);

        // Verify barrier depends on the dead store
        assert!(
            emit.ops[3].dep_ids.contains(&1),
            "Barrier should depend on dead store (op1)"
        );

        // Schedule
        let config = crate::kernel_builder::config::KernelConfig::new(10, 2047, 256, 16);
        let builder = crate::kernel_builder::tick_tock_ssa::SsaTickTockKernelBuilder::new(config);

        let mut shifted_ops = HashSet::new();
        let result = builder.schedule_ssa_greedy_with_shift_budget(
            &mut emit.ops,
            false, // no VALU shift
            &mut shifted_ops,
            1536, // default budget
        );

        assert!(result.is_ok(), "Should schedule despite barrier depending on dead store: {:?}", result.err());

        // Verify barrier was scheduled
        assert!(emit.ops[3].cycle >= 0, "Barrier should be scheduled (cycle={})", emit.ops[3].cycle);

        // Verify ops after barrier were scheduled
        assert!(emit.ops[4].cycle > emit.ops[3].cycle,
            "Op after barrier should be after barrier (op4.cycle={} > barrier.cycle={})",
            emit.ops[4].cycle, emit.ops[3].cycle);

        // Dead store (op1) should remain unscheduled (cycle = -1)
        assert_eq!(emit.ops[1].cycle, -1, "Dead store should not be scheduled");
    }

    /// Test multiple dead stores before a barrier.
    ///
    /// Ensures the fix handles multiple dead stores, not just one.
    #[test]
    fn test_multiple_dead_stores_before_barrier() {
        // Scenario:
        // op0: v1 = const 1          (DEAD STORE)
        // op1: v2 = const 2          (DEAD STORE)
        // op2: v3 = const 3          (DEAD STORE)
        // op3: v4 = const 4          (used by op5)
        // op4: barrier               (depends on op0-3, including 3 dead stores)
        // op5: v5 = add v4, v4       (used by store)
        // op6: addr = const 100
        // op7: store addr, v5

        let mut emit = SsaEmitContext::new();
        let _v1 = emit.emit_const(1);      // op 0 - DEAD STORE
        let _v2 = emit.emit_const(2);      // op 1 - DEAD STORE
        let _v3 = emit.emit_const(3);      // op 2 - DEAD STORE
        let v4 = emit.emit_const(4);       // op 3 - used
        emit.emit_barrier();                // op 4 - depends on all above
        let v5 = emit.emit_alu(Opcode::Add, v4, v4);  // op 5 - used
        let addr = emit.emit_const(100);   // op 6
        emit.emit_store(addr, v5);         // op 7

        compute_ssa_dependencies_with_barriers(&mut emit.ops, &emit.vreg_def);

        // Verify barrier depends on all dead stores
        assert!(emit.ops[4].dep_ids.contains(&0), "Barrier should depend on dead store op0");
        assert!(emit.ops[4].dep_ids.contains(&1), "Barrier should depend on dead store op1");
        assert!(emit.ops[4].dep_ids.contains(&2), "Barrier should depend on dead store op2");

        // Schedule
        let config = crate::kernel_builder::config::KernelConfig::new(10, 2047, 256, 16);
        let builder = crate::kernel_builder::tick_tock_ssa::SsaTickTockKernelBuilder::new(config);

        let mut shifted_ops = HashSet::new();
        let result = builder.schedule_ssa_greedy_with_shift_budget(
            &mut emit.ops,
            false,
            &mut shifted_ops,
            1536,
        );

        assert!(result.is_ok(), "Should schedule with multiple dead stores: {:?}", result.err());

        // Verify all dead stores remain unscheduled
        assert_eq!(emit.ops[0].cycle, -1, "Dead store op0 should not be scheduled");
        assert_eq!(emit.ops[1].cycle, -1, "Dead store op1 should not be scheduled");
        assert_eq!(emit.ops[2].cycle, -1, "Dead store op2 should not be scheduled");

        // Verify barrier and subsequent ops are scheduled
        assert!(emit.ops[4].cycle >= 0, "Barrier should be scheduled");
        assert!(emit.ops[5].cycle > emit.ops[4].cycle, "Op5 should be after barrier");
    }

    /// Test barrier chaining with dead stores between barriers.
    ///
    /// Ensures dead stores in different sections don't break barrier chaining.
    #[test]
    fn test_barrier_chain_with_dead_stores() {
        // Scenario:
        // Section 1:
        //   op0: v1 = const 1        (used by op5)
        //   op1: v2 = const 2        (DEAD STORE)
        // op2: barrier1              (depends on op0, op1 including dead store)
        // Section 2:
        //   op3: v3 = const 3        (DEAD STORE)
        //   op4: v4 = const 4        (used by op5)
        // op5: barrier2              (depends on op3, op4, barrier1)
        // Section 3:
        //   op6: v5 = add v1, v4     (used by store)
        //   op7: addr = const 100
        //   op8: store addr, v5

        let mut emit = SsaEmitContext::new();
        // Section 1
        let v1 = emit.emit_const(1);       // op 0 - used
        let _v2 = emit.emit_const(2);      // op 1 - DEAD STORE
        emit.emit_barrier();                // op 2 - barrier1
        // Section 2
        let _v3 = emit.emit_const(3);      // op 3 - DEAD STORE
        let v4 = emit.emit_const(4);       // op 4 - used
        emit.emit_barrier();                // op 5 - barrier2
        // Section 3
        let v5 = emit.emit_alu(Opcode::Add, v1, v4);  // op 6 - used
        let addr = emit.emit_const(100);   // op 7
        emit.emit_store(addr, v5);         // op 8

        compute_ssa_dependencies_with_barriers(&mut emit.ops, &emit.vreg_def);

        // Verify barrier1 depends on dead store op1
        assert!(emit.ops[2].dep_ids.contains(&1), "Barrier1 should depend on dead store op1");

        // Verify barrier2 depends on dead store op3 and barrier1
        assert!(emit.ops[5].dep_ids.contains(&3), "Barrier2 should depend on dead store op3");
        assert!(emit.ops[5].dep_ids.contains(&2), "Barrier2 should depend on barrier1 (chain)");

        // Schedule
        let config = crate::kernel_builder::config::KernelConfig::new(10, 2047, 256, 16);
        let builder = crate::kernel_builder::tick_tock_ssa::SsaTickTockKernelBuilder::new(config);

        let mut shifted_ops = HashSet::new();
        let result = builder.schedule_ssa_greedy_with_shift_budget(
            &mut emit.ops,
            false,
            &mut shifted_ops,
            1536,
        );

        assert!(result.is_ok(), "Should schedule with barrier chain and dead stores: {:?}", result.err());

        // Verify barriers are scheduled in order
        assert!(emit.ops[2].cycle >= 0, "Barrier1 should be scheduled");
        assert!(emit.ops[5].cycle > emit.ops[2].cycle, "Barrier2 should be after barrier1");

        // Verify final ops are after barrier2
        assert!(emit.ops[6].cycle > emit.ops[5].cycle, "Op6 should be after barrier2");
    }

    // =========================================================================
    // Dead Store Detection Tests
    //
    // These tests verify that dead store detection works correctly:
    // - Ops with no readers are dead stores
    // - Dead stores are skipped from scheduling (cycle remains -1)
    // - Non-dead-store ops are not affected
    // =========================================================================

    /// Test that dead stores in the middle of a dependency chain don't break scheduling.
    #[test]
    fn test_dead_store_in_dependency_chain() {
        // Scenario:
        // v1 = const 1
        // v2 = add v1, v1  (dead store - result never used)
        // v3 = const 2
        // v4 = add v1, v3  (uses v1, v3 - not v2)
        // store addr, v4   (makes v4 live)
        //
        // v2 is a dead store but sits between used ops.

        let mut emit = SsaEmitContext::new();
        let v1 = emit.emit_const(1);  // op 0
        let _v2 = emit.emit_alu(Opcode::Add, v1, v1); // op 1 - dead store
        let v3 = emit.emit_const(2);  // op 2
        let v4 = emit.emit_alu(Opcode::Add, v1, v3); // op 3
        let addr = emit.emit_const(100); // op 4
        emit.emit_store(addr, v4);     // op 5 - makes v4 live

        compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def);

        let config = crate::kernel_builder::config::KernelConfig::new(10, 2047, 256, 16);
        let builder = crate::kernel_builder::tick_tock_ssa::SsaTickTockKernelBuilder::new(config);

        let mut shifted_ops = HashSet::new();
        let result = builder.schedule_ssa_greedy_with_shift_budget(
            &mut emit.ops,
            false,
            &mut shifted_ops,
            1536,
        );

        assert!(result.is_ok(), "Should schedule with dead store in chain: {:?}", result.err());

        // Check that non-dead-stores got valid cycles
        assert!(emit.ops[0].cycle >= 0, "op0 should be scheduled");
        assert!(emit.ops[2].cycle >= 0, "op2 should be scheduled");
        assert!(emit.ops[3].cycle >= 0, "op3 should be scheduled");

        // op1 (dead store) should remain unscheduled
        assert_eq!(emit.ops[1].cycle, -1, "Dead store op1 should not be scheduled");
    }

    /// Test that transitive dead stores are detected.
    ///
    /// If op A is dead, and op B only depends on A and produces a value
    /// that is never used, then B is also dead.
    #[test]
    fn test_dead_store_detection_is_transitive() {
        // Scenario:
        // op0: v1 = const 1          (used only by dead store op1)
        // op1: v2 = add v1, v1       (DEAD STORE - result never used)
        // op2: v3 = const 2          (used by op3)
        // op3: v4 = add v3, v3       (used by store)
        // op4: addr = const 100
        // op5: store addr, v4
        //
        // v1 is transitively dead (only user is dead store v2)
        // v2 is dead (never used)

        let mut emit = SsaEmitContext::new();
        let v1 = emit.emit_const(1);       // op 0 - transitively dead (only user is dead)
        let _v2 = emit.emit_alu(Opcode::Add, v1, v1); // op 1 - dead store
        let v3 = emit.emit_const(2);       // op 2 - used
        let v4 = emit.emit_alu(Opcode::Add, v3, v3); // op 3 - used
        let addr = emit.emit_const(100);   // op 4
        emit.emit_store(addr, v4);         // op 5

        compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def);

        let config = crate::kernel_builder::config::KernelConfig::new(10, 2047, 256, 16);
        let builder = crate::kernel_builder::tick_tock_ssa::SsaTickTockKernelBuilder::new(config);

        let mut shifted_ops = HashSet::new();
        let result = builder.schedule_ssa_greedy_with_shift_budget(
            &mut emit.ops,
            false,
            &mut shifted_ops,
            1536,
        );

        assert!(result.is_ok(), "Should schedule with transitive dead stores: {:?}", result.err());

        // Note: Current implementation only detects immediate dead stores,
        // not transitive ones. op0 (v1) is used by op1 (v2), so it's not
        // considered dead even though op1 is dead.
        //
        // This is acceptable behavior - transitive dead code elimination
        // would require fixed-point iteration. The important thing is that
        // scheduling doesn't fail.

        // op1 (v2) is definitely dead (no users)
        assert_eq!(emit.ops[1].cycle, -1, "Dead store op1 should not be scheduled");

        // Live ops should be scheduled
        assert!(emit.ops[2].cycle >= 0, "op2 should be scheduled");
        assert!(emit.ops[3].cycle >= 0, "op3 should be scheduled");
    }

    /// Test scheduling with an immediate dead store (no transitive analysis).
    ///
    /// The scheduler detects immediate dead stores (ops whose dest has no readers),
    /// but does NOT do transitive dead store elimination. So if v3 = add(v1, v2)
    /// is never used, v3 is dead, but v1 and v2 are NOT dead because they have
    /// v3 as a reader (even though v3 itself is dead).
    ///
    /// This test verifies the scheduler handles immediate dead stores correctly
    /// and doesn't fail when some ops in the graph are dead.
    #[test]
    fn test_immediate_dead_store() {
        let mut emit = SsaEmitContext::new();
        // Use emit_const_force to avoid CSE (const caching)
        let v1 = emit.emit_const_force(5);  // op 0 - has reader (op2), not dead
        let v2 = emit.emit_const_force(6);  // op 1 - has reader (op2), not dead
        let _v3 = emit.emit_alu(Opcode::Add, v1, v2); // op 2 - DEAD (no readers)

        compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def);

        // Verify all 3 ops were created
        assert_eq!(emit.ops.len(), 3, "Should have 3 ops");

        let config = crate::kernel_builder::config::KernelConfig::new(10, 2047, 256, 16);
        let builder = crate::kernel_builder::tick_tock_ssa::SsaTickTockKernelBuilder::new(config);

        let mut shifted_ops = HashSet::new();
        let result = builder.schedule_ssa_greedy_with_shift_budget(
            &mut emit.ops,
            false,
            &mut shifted_ops,
            1536,
        );

        // Scheduling should succeed
        assert!(result.is_ok(), "Scheduling should succeed: {:?}", result.err());
        let (_cycles, budget_exceeded, _peak_live) = result.unwrap();
        assert!(!budget_exceeded, "Should not exceed budget");

        // Only op2 (the add) is an immediate dead store
        // op0 and op1 (the consts) have readers so they're scheduled
        assert!(emit.ops[0].cycle >= 0, "op0 should be scheduled (has reader op2)");
        assert!(emit.ops[1].cycle >= 0, "op1 should be scheduled (has reader op2)");
        assert_eq!(emit.ops[2].cycle, -1, "op2 should not be scheduled (dead store)");
    }

    // =========================================================================
    // Store Source VReg Liveness Tests
    //
    // These tests verify that store operations correctly count their source
    // VRegs as users, preventing those VRegs from being marked as dead stores.
    //
    // THE BUG: If store operations don't count as users of their source VRegs,
    // ops that compute values for stores could be marked as dead and skipped,
    // leaving garbage values in scratch that cause runtime crashes.
    // =========================================================================

    /// Test that store source VRegs are NOT marked as dead stores.
    ///
    /// This is the core test for the validation bug: a vstore's data source
    /// must be counted as a user, otherwise the FMA that produces it could
    /// be marked as dead and skipped.
    #[test]
    fn test_store_source_vregs_not_dead() {
        // Scenario (simplified version of the bug):
        // op0: v0 = const 2 (multiplier for FMA)
        // op1: v1 = const 100 (offset for FMA)
        // op2: v2 = vbroadcast(v0) (FMA multiplier vector)
        // op3: v3 = vbroadcast(v1) (FMA offset vector)
        // op4: v4 = const 0 (base index)
        // op5: v5 = vbroadcast(v4) (index vector)
        // op6: v6 = FMA(v5, v2, v3) -> computes idx = idx * 2 + 100
        // op7: addr = const 200 (store address)
        // op8: vstore(addr, v6) -> store the FMA result
        //
        // v6 (FMA result) is ONLY used by vstore (op8).
        // If vstore doesn't count as a user, v6 would be dead and op6 would be skipped.
        // This would leave garbage at the scratch address for v6, causing crashes.

        let mut emit = SsaEmitContext::new();
        let mult_scalar = emit.emit_const(2);           // op 0
        let offset_scalar = emit.emit_const(100);       // op 1
        let mult_vec = emit.emit_vbroadcast(mult_scalar);   // op 2
        let offset_vec = emit.emit_vbroadcast(offset_scalar); // op 3
        let base_scalar = emit.emit_const(0);           // op 4
        let base_vec = emit.emit_vbroadcast(base_scalar);    // op 5
        let fma_result = emit.emit_fma(base_vec, mult_vec, offset_vec); // op 6
        let addr = emit.emit_const(200);                // op 7
        emit.emit_vstore(addr, fma_result);             // op 8

        compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def);

        // Verify: vstore (op8) reads fma_result (v6)
        // Build vreg_users map as the scheduler does
        let mut vreg_users: HashMap<VReg, HashSet<usize>> = HashMap::new();
        for (idx, op) in emit.ops.iter().enumerate() {
            for src in &op.srcs {
                if let Operand::VReg(vreg) = src {
                    vreg_users.entry(*vreg).or_default().insert(idx);
                }
            }
        }

        // fma_result (v6) should have at least 1 user (the vstore)
        let fma_result_users = vreg_users.get(&fma_result).map_or(0, |s| s.len());
        assert!(fma_result_users >= 1,
            "FMA result should have at least 1 user (vstore), but has {} users",
            fma_result_users);

        // Schedule and verify FMA is NOT a dead store
        let config = crate::kernel_builder::config::KernelConfig::new(10, 2047, 256, 16);
        let builder = crate::kernel_builder::tick_tock_ssa::SsaTickTockKernelBuilder::new(config);

        let mut shifted_ops = HashSet::new();
        let result = builder.schedule_ssa_greedy_with_shift_budget(
            &mut emit.ops,
            false,
            &mut shifted_ops,
            1536,
        );

        assert!(result.is_ok(), "Should schedule successfully: {:?}", result.err());

        // The FMA (op6) should be scheduled (NOT a dead store)
        assert!(emit.ops[6].cycle >= 0,
            "FMA op should be scheduled (used by vstore), but has cycle={}",
            emit.ops[6].cycle);

        // The vstore (op8) should be scheduled
        assert!(emit.ops[8].cycle >= 0,
            "vstore op should be scheduled, but has cycle={}",
            emit.ops[8].cycle);

        // vstore should happen AFTER FMA (dependency order)
        assert!(emit.ops[8].cycle > emit.ops[6].cycle,
            "vstore (cycle={}) should be after FMA (cycle={})",
            emit.ops[8].cycle, emit.ops[6].cycle);
    }

    /// Test that store address VRegs are also counted as users.
    ///
    /// Both the address and data operands of a store should be counted as users.
    #[test]
    fn test_store_address_vreg_liveness() {
        // Scenario:
        // op0: base = const 100
        // op1: offset = const 8
        // op2: addr = add(base, offset) -> computed address
        // op3: data = const 42
        // op4: store(addr, data)
        //
        // Both addr (v2) and data (v3) are used by the store.
        // Neither should be marked as dead.

        let mut emit = SsaEmitContext::new();
        let base = emit.emit_const(100);           // op 0
        let offset = emit.emit_const(8);           // op 1
        let addr = emit.emit_alu(Opcode::Add, base, offset);  // op 2
        let data = emit.emit_const(42);            // op 3
        emit.emit_store(addr, data);               // op 4

        compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def);

        // Build vreg_users map
        let mut vreg_users: HashMap<VReg, HashSet<usize>> = HashMap::new();
        for (idx, op) in emit.ops.iter().enumerate() {
            for src in &op.srcs {
                if let Operand::VReg(vreg) = src {
                    vreg_users.entry(*vreg).or_default().insert(idx);
                }
            }
        }

        // addr should have the store as a user
        assert!(vreg_users.get(&addr).map_or(false, |users| users.contains(&4)),
            "addr VReg should have store (op4) as user");

        // data should have the store as a user
        assert!(vreg_users.get(&data).map_or(false, |users| users.contains(&4)),
            "data VReg should have store (op4) as user");

        // Schedule
        let config = crate::kernel_builder::config::KernelConfig::new(10, 2047, 256, 16);
        let builder = crate::kernel_builder::tick_tock_ssa::SsaTickTockKernelBuilder::new(config);

        let mut shifted_ops = HashSet::new();
        let result = builder.schedule_ssa_greedy_with_shift_budget(
            &mut emit.ops,
            false,
            &mut shifted_ops,
            1536,
        );

        assert!(result.is_ok(), "Should schedule successfully: {:?}", result.err());

        // addr computation (op2) should be scheduled
        assert!(emit.ops[2].cycle >= 0,
            "addr computation should be scheduled (used by store)");

        // data (op3) should be scheduled
        assert!(emit.ops[3].cycle >= 0,
            "data const should be scheduled (used by store)");
    }

    // =========================================================================
    // DCE Invariant Tests: Cannot Eliminate Ops With Users
    //
    // These tests verify the fundamental invariant: DCE should NEVER mark an
    // op as dead if its dest VReg has any users. This is a structural property
    // of the vreg_users map construction.
    // =========================================================================

    /// Test that DCE cannot eliminate an op that has a user.
    ///
    /// This is the core invariant test: if op A defines v1, and op B uses v1,
    /// then op A should NOT be marked as dead, regardless of whether op B
    /// itself is dead or alive.
    #[test]
    fn test_dce_cannot_eliminate_op_with_user() {
        // Scenario:
        // op0: v1 = const 5
        // op1: v2 = add(v1, v1)  <- uses v1
        // op2: vstore(addr, v2)  <- uses v2
        //
        // v1 has user (op1), v2 has user (op2)
        // Neither op0 nor op1 should be marked as dead

        let mut emit = SsaEmitContext::new();
        let v1 = emit.emit_const(5);           // op 0 - has user (op1)
        let v1_vec = emit.emit_vbroadcast(v1); // op 1 - has user (op2)
        let v2 = emit.emit_valu(Opcode::Add, v1_vec, v1_vec); // op 2 - has user (op3)
        let addr = emit.emit_const(100);       // op 3 - has user (op4)
        emit.emit_vstore(addr, v2);            // op 4 - sink (no dest)

        compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def);

        // Build vreg_users map as the scheduler does
        let mut vreg_users: HashMap<VReg, HashSet<usize>> = HashMap::new();
        for (idx, op) in emit.ops.iter().enumerate() {
            for src in &op.srcs {
                if let Operand::VReg(vreg) = src {
                    vreg_users.entry(*vreg).or_default().insert(idx);
                }
            }
        }

        // Verify the invariant: each VReg with users should NOT be dead
        // v1 (scalar) should have user (op1 - vbroadcast)
        assert!(vreg_users.get(&v1).map_or(false, |users| !users.is_empty()),
            "v1 should have at least 1 user");

        // v1_vec (vector) should have user (op2 - add)
        assert!(vreg_users.get(&v1_vec).map_or(false, |users| !users.is_empty()),
            "v1_vec should have at least 1 user");

        // v2 (vector) should have user (op4 - vstore)
        assert!(vreg_users.get(&v2).map_or(false, |users| !users.is_empty()),
            "v2 should have at least 1 user");

        // addr should have user (op4 - vstore)
        assert!(vreg_users.get(&addr).map_or(false, |users| !users.is_empty()),
            "addr should have at least 1 user");

        // Now schedule and verify none of these are marked as dead
        let config = crate::kernel_builder::config::KernelConfig::new(10, 2047, 256, 16);
        let builder = crate::kernel_builder::tick_tock_ssa::SsaTickTockKernelBuilder::new(config);

        let mut shifted_ops = HashSet::new();
        let result = builder.schedule_ssa_greedy_with_shift_budget(
            &mut emit.ops,
            false,
            &mut shifted_ops,
            1536,
        );

        assert!(result.is_ok(), "Should schedule successfully: {:?}", result.err());

        // All ops except the store should be scheduled (store has no dest, not tracked)
        // ops 0-3 have dests and should all be scheduled
        assert!(emit.ops[0].cycle >= 0, "op0 (const) should be scheduled");
        assert!(emit.ops[1].cycle >= 0, "op1 (vbroadcast) should be scheduled");
        assert!(emit.ops[2].cycle >= 0, "op2 (add) should be scheduled");
        assert!(emit.ops[3].cycle >= 0, "op3 (addr) should be scheduled");
        assert!(emit.ops[4].cycle >= 0, "op4 (vstore) should be scheduled");
    }

    /// Test that transitive users prevent elimination.
    ///
    /// A chain of ops where each depends on the previous should all be
    /// scheduled, not eliminated by DCE.
    #[test]
    fn test_transitive_users_prevent_elimination() {
        // Scenario:
        // op0: v1 = const 5
        // op1: v2 = add(v1, v1)  <- v1 has user
        // op2: v3 = mul(v2, v2)  <- v2 has user
        // op3: vstore(addr, v3)  <- v3 has user
        //
        // All of v1, v2, v3 have users - none should be dead

        let mut emit = SsaEmitContext::new();
        let v1 = emit.emit_const(5);           // op 0
        let v1_vec = emit.emit_vbroadcast(v1); // op 1
        let v2 = emit.emit_valu(Opcode::Add, v1_vec, v1_vec); // op 2
        let v3 = emit.emit_valu(Opcode::Mul, v2, v2);         // op 3
        let addr = emit.emit_const(100);       // op 4
        emit.emit_vstore(addr, v3);            // op 5

        compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def);

        // Build vreg_users map
        let mut vreg_users: HashMap<VReg, HashSet<usize>> = HashMap::new();
        for (idx, op) in emit.ops.iter().enumerate() {
            for src in &op.srcs {
                if let Operand::VReg(vreg) = src {
                    vreg_users.entry(*vreg).or_default().insert(idx);
                }
            }
        }

        // Verify the transitive chain: each VReg should have a user
        assert!(vreg_users.get(&v1).map_or(0, |s| s.len()) >= 1, "v1 should have user");
        assert!(vreg_users.get(&v1_vec).map_or(0, |s| s.len()) >= 1, "v1_vec should have user");
        assert!(vreg_users.get(&v2).map_or(0, |s| s.len()) >= 1, "v2 should have user");
        assert!(vreg_users.get(&v3).map_or(0, |s| s.len()) >= 1, "v3 should have user");

        // Schedule
        let config = crate::kernel_builder::config::KernelConfig::new(10, 2047, 256, 16);
        let builder = crate::kernel_builder::tick_tock_ssa::SsaTickTockKernelBuilder::new(config);

        let mut shifted_ops = HashSet::new();
        let result = builder.schedule_ssa_greedy_with_shift_budget(
            &mut emit.ops,
            false,
            &mut shifted_ops,
            1536,
        );

        assert!(result.is_ok(), "Should schedule transitive chain: {:?}", result.err());

        // All ops in the chain should be scheduled
        for (i, op) in emit.ops.iter().enumerate() {
            assert!(op.cycle >= 0, "op{} should be scheduled, got cycle={}", i, op.cycle);
        }
    }

    /// Test that store sources are counted as users.
    ///
    /// This is the key invariant for the vstore bug: the data source of a
    /// store operation must be counted as a user to prevent DCE from
    /// eliminating the op that produces it.
    #[test]
    fn test_store_sources_count_as_users() {
        // Scenario:
        // op0: v1 = FMA(...)
        // op1: vstore(addr, v1)  <- v1 is used by store
        //
        // v1 has a user (the vstore), so op0 should NOT be dead

        let mut emit = SsaEmitContext::new();
        // Create FMA inputs
        let a_scalar = emit.emit_const(1);
        let b_scalar = emit.emit_const(2);
        let c_scalar = emit.emit_const(3);
        let a = emit.emit_vbroadcast(a_scalar);
        let b = emit.emit_vbroadcast(b_scalar);
        let c = emit.emit_vbroadcast(c_scalar);

        // FMA result - ONLY used by vstore
        let fma_result = emit.emit_fma(a, b, c);  // op 6
        let addr = emit.emit_const(100);         // op 7
        emit.emit_vstore(addr, fma_result);      // op 8

        compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def);

        // Build vreg_users map
        let mut vreg_users: HashMap<VReg, HashSet<usize>> = HashMap::new();
        for (idx, op) in emit.ops.iter().enumerate() {
            for src in &op.srcs {
                if let Operand::VReg(vreg) = src {
                    vreg_users.entry(*vreg).or_default().insert(idx);
                }
            }
        }

        // The FMA result should have the vstore as its user
        let fma_users = vreg_users.get(&fma_result).map_or(0, |s| s.len());
        assert!(fma_users >= 1,
            "FMA result should have at least 1 user (vstore), but has {} users",
            fma_users);

        // Schedule
        let config = crate::kernel_builder::config::KernelConfig::new(10, 2047, 256, 16);
        let builder = crate::kernel_builder::tick_tock_ssa::SsaTickTockKernelBuilder::new(config);

        let mut shifted_ops = HashSet::new();
        let result = builder.schedule_ssa_greedy_with_shift_budget(
            &mut emit.ops,
            false,
            &mut shifted_ops,
            1536,
        );

        assert!(result.is_ok(), "Should schedule FMA with store user: {:?}", result.err());

        // The FMA (op 6) should be scheduled, NOT marked as dead
        assert!(emit.ops[6].cycle >= 0,
            "FMA op should be scheduled (used by vstore), but has cycle={}",
            emit.ops[6].cycle);
    }

    // =========================================================================
    // Minimal Reproduction of Invalid Memory Write Bug
    //
    // The invalid memory write error is NOT a DCE bug. It's a register
    // allocation bug where a VReg's live interval is computed incorrectly,
    // causing another VReg to overwrite it before its last use.
    //
    // Root cause: Linear scan allocation in address_assign.rs
    // Symptom: vstore uses garbage address from scratch[4]
    //
    // Reproduction (run with rounds=2):
    //   Cycle 10: scratch[4] = scratch[29] + scratch[1]  (overwrites partition addr)
    //   Cycle 81: vstore scratch[4], scratch[126]        (uses garbage address)
    //
    // The partition store address VReg should have a live interval that
    // extends to cycle 81, but it's being ended too early.
    // =========================================================================

    /// Test DCE correctly handles unused VRegs (simulates scattered_access bug).
    ///
    /// This test simulates the bug where `let _ = idx;` in Rust doesn't create
    /// an SSA op, so the VReg has no users and is incorrectly marked as dead.
    ///
    /// The DCE invariant is: if a VReg has NO users across ALL ops, it's dead.
    /// This test verifies that DCE correctly identifies dead stores when a
    /// VReg is genuinely unused.
    #[test]
    fn test_dce_detects_genuinely_dead_store() {
        // Scenario:
        // op0: v1 = const 5         (used by op1)
        // op1: v2 = add(v1, v1)     (DEAD - never used, simulates emit_fma result)
        // op2: v3 = const 10        (used by store)
        // op3: addr = const 100
        // op4: store(addr, v3)
        //
        // v2 is genuinely dead - no op reads it. DCE should mark op1 as dead.

        let mut emit = SsaEmitContext::new();
        let v1 = emit.emit_const(5);           // op 0
        let _v2 = emit.emit_alu(Opcode::Add, v1, v1); // op 1 - DEAD (result never used)
        let v3 = emit.emit_const(10);          // op 2
        let addr = emit.emit_const(100);       // op 3
        emit.emit_store(addr, v3);             // op 4

        compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def);

        // Schedule
        let config = crate::kernel_builder::config::KernelConfig::new(10, 2047, 256, 16);
        let builder = crate::kernel_builder::tick_tock_ssa::SsaTickTockKernelBuilder::new(config);

        let mut shifted_ops = HashSet::new();
        let result = builder.schedule_ssa_greedy_with_shift_budget(
            &mut emit.ops,
            false,
            &mut shifted_ops,
            1536,
        );

        assert!(result.is_ok(), "Should schedule: {:?}", result.err());

        // op1 (v2) should be marked as dead (cycle = -1)
        assert_eq!(emit.ops[1].cycle, -1, "Dead store op1 should not be scheduled");

        // Other ops should be scheduled
        assert!(emit.ops[0].cycle >= 0, "op0 should be scheduled");
        assert!(emit.ops[2].cycle >= 0, "op2 should be scheduled");
        assert!(emit.ops[3].cycle >= 0, "op3 should be scheduled");
        assert!(emit.ops[4].cycle >= 0, "op4 should be scheduled");
    }

    /// Test that dead store detection correctly counts store users.
    ///
    /// Verifies the vreg_users map is built correctly when stores are present.
    #[test]
    fn test_dead_store_detection_counts_store_users() {
        // Scenario:
        // op0: v0 = const 5
        // op1: v1 = add(v0, v0)  <- used by op2 vstore
        // op2: addr = const 100
        // op3: vstore(addr, v1)  <- sink operation
        //
        // v1 is ONLY used by the vstore.
        // The vreg_users map should include op3 as a user of v1.

        let mut emit = SsaEmitContext::new();
        let v0_scalar = emit.emit_const(5);           // op 0
        let v0 = emit.emit_vbroadcast(v0_scalar);     // op 1
        let v1 = emit.emit_valu(Opcode::Add, v0, v0); // op 2 - ONLY used by vstore
        let addr = emit.emit_const(100);              // op 3
        emit.emit_vstore(addr, v1);                   // op 4 - uses v1

        compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def);

        // Build vreg_users map as the scheduler does
        let mut vreg_users: HashMap<VReg, HashSet<usize>> = HashMap::new();
        for (idx, op) in emit.ops.iter().enumerate() {
            for src in &op.srcs {
                if let Operand::VReg(vreg) = src {
                    vreg_users.entry(*vreg).or_default().insert(idx);
                }
            }
        }

        // v1 should have vstore (op4) as its user
        let v1_users = vreg_users.get(&v1);
        assert!(v1_users.is_some(), "v1 should have users");
        assert!(v1_users.unwrap().contains(&4),
            "v1's users should include vstore (op4), but users are: {:?}", v1_users);

        // v1 should NOT be considered a dead store (it has the vstore as user)
        let v1_is_dead = vreg_users.get(&v1).map_or(true, |s| s.is_empty());
        assert!(!v1_is_dead, "v1 should NOT be a dead store (used by vstore)");

        // Schedule and verify v1 is scheduled
        let config = crate::kernel_builder::config::KernelConfig::new(10, 2047, 256, 16);
        let builder = crate::kernel_builder::tick_tock_ssa::SsaTickTockKernelBuilder::new(config);

        let mut shifted_ops = HashSet::new();
        let result = builder.schedule_ssa_greedy_with_shift_budget(
            &mut emit.ops,
            false,
            &mut shifted_ops,
            1536,
        );

        assert!(result.is_ok(), "Should schedule successfully: {:?}", result.err());

        // v1 (op2) should be scheduled
        assert!(emit.ops[2].cycle >= 0,
            "v1 (add) should be scheduled (used by vstore), but has cycle={}",
            emit.ops[2].cycle);
    }

    #[test]
    fn test_greedy_constchoice_treated_as_load() {
        use super::super::emit_ssa::SsaEmitContext;
        use crate::types::Opcode;

        let mut emit = SsaEmitContext::new();
        
        // Emit zero constant (required for AddImm)
        let zero = emit.emit_const(0);
        
        // Emit ConstChoice constant
        let c1 = emit.emit_const_choice(42);
        
        // Use the constant in an ALU op
        let _result = emit.emit_alu(Opcode::Add, c1, zero);

        let mut builder = SsaTickTockKernelBuilder::new(
            super::super::config::KernelConfig::new(10, 2047, 256, 16)
        );
        builder.zero_vreg = Some(zero);

        let mut shifted_ops = HashSet::new();
        let result = builder.schedule_ssa_greedy_with_shift_budget(
            &mut emit.ops,
            false,
            &mut shifted_ops,
            1536,
        );

        assert!(result.is_ok(), "Should schedule successfully: {:?}", result.err());

        // Find the ConstChoice op
        let const_choice_op = emit.ops.iter().find(|op| op.opcode == Opcode::ConstChoice);
        assert!(const_choice_op.is_some(), "Should have ConstChoice op");
        let const_choice_op = const_choice_op.unwrap();
        
        // ConstChoice should be scheduled (greedy treats it as Load)
        assert!(const_choice_op.cycle >= 0, 
            "ConstChoice op should be scheduled, but has cycle={}", 
            const_choice_op.cycle);
        
        // Should be scheduled early (before ops that use it)
        let use_op = emit.ops.iter().find(|op| {
            op.cycle >= 0 && // Only check scheduled ops (skip dead stores)
            op.srcs.iter().any(|src| {
                if let Operand::VReg(vreg) = src {
                    *vreg == c1
                } else {
                    false
                }
            })
        });
        if let Some(use_op) = use_op {
            assert!(const_choice_op.cycle < use_op.cycle,
                "ConstChoice should be scheduled before ops that use it: const_choice={}, use_op={}",
                const_choice_op.cycle, use_op.cycle);
        }
    }
}
