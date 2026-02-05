//! Schedule constraints: same rules as CP-SAT model, for checking and bulk-move simulation.
//!
//! We build the exact precedence, ordering, and domain constraints that the CP-SAT
//! model uses, so the bulk-move simulator can check them without duplicating logic.
//!
//! **Eliminated vbroadcasts:** When CP-SAT eliminates a vbroadcast (optional VALU interval
//! absent), that op still has a cycle assignment and appears in `ssa_cycle_assignments` and
//! in the constraint list (Domain/Precedence). It is not emitted in lowering. Bulk-move
//! and constraint checking use the same index space, so no special handling is needed.

use std::collections::{HashMap, HashSet};

use crate::types::SsaOp;

/// One constraint that must hold for a valid schedule (same as in CP-SAT).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ScheduleConstraint {
    /// consumer_idx must run at cycle >= dep_cycle + result_offset.
    /// If dep_is_shifted, dep_cycle is the completion_cycle (max of 8 ALU cycles).
    Precedence {
        consumer_idx: usize,
        dep_idx: usize,
        result_offset: i32,
        dep_is_shifted: bool,
    },
    /// before_idx must run at cycle <= after_idx (ordering group or lex stream).
    Ordering {
        before_idx: usize,
        after_idx: usize,
    },
    /// op at idx must run in [lb, ub] (domain).
    Domain {
        idx: usize,
        lb: i32,
        ub: i32,
    },
}

/// Why a schedule or a bulk-move step is invalid.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ConstraintViolation {
    pub constraint: ScheduleConstraint,
    pub message: String,
}

/// Build the same constraints the CP-SAT model uses (precedence, ordering, domain).
/// Uses concrete decisions: shifted_ops (op ids), select_fma_ops (op ids).
#[allow(clippy::too_many_arguments)]
pub fn build_ssa_schedule_constraints(
    ops: &[SsaOp],
    op_id_to_idx: &HashMap<usize, usize>,
    asap: &HashMap<usize, i32>,
    alap: &HashMap<usize, i32>,
    shiftable_set: &HashSet<usize>,
    shifted_ops: &HashSet<usize>, // op ids that are shifted (from result)
    select_choice_with_fma_delay: &HashSet<usize>, // indices
    select_fma_ops: &HashSet<usize>,              // op ids that chose FMA (from result)
    horizon: i32,
) -> Vec<ScheduleConstraint> {
    let mut out = Vec::new();
    let default_alap = horizon - 1;

    // Domain: [ASAP, ALAP] per op (same as cycle var domain)
    for (idx, op) in ops.iter().enumerate() {
        let lb = *asap.get(&op.id).unwrap_or(&0);
        let ub = *alap.get(&op.id).unwrap_or(&default_alap);
        out.push(ScheduleConstraint::Domain { idx, lb, ub });
    }

    // 5. Precedence: normal deps (skip shiftable and SelectChoice FMA-delay; those added below)
    for (idx, op) in ops.iter().enumerate() {
        for &dep_id in &op.dep_ids {
            let Some(&dep_idx) = op_id_to_idx.get(&dep_id) else {
                continue;
            };
            if shiftable_set.contains(&dep_idx) {
                continue;
            }
            if select_choice_with_fma_delay.contains(&dep_idx) {
                continue;
            }
            out.push(ScheduleConstraint::Precedence {
                consumer_idx: idx,
                dep_idx,
                result_offset: 1,
                dep_is_shifted: false,
            });
        }
    }

    // 5c. Precedence: consumers of shifted VALU (use completion_cycle)
    for (idx, op) in ops.iter().enumerate() {
        for &dep_id in &op.dep_ids {
            let Some(&dep_idx) = op_id_to_idx.get(&dep_id) else {
                continue;
            };
            if !shiftable_set.contains(&dep_idx) {
                continue;
            }
            out.push(ScheduleConstraint::Precedence {
                consumer_idx: idx,
                dep_idx,
                result_offset: 1,
                dep_is_shifted: true,
            });
        }
    }

    // 5d. Precedence: consumers of SelectChoice FMA (no delta): result at cycle+2
    for (idx, op) in ops.iter().enumerate() {
        for &dep_id in &op.dep_ids {
            let Some(&dep_idx) = op_id_to_idx.get(&dep_id) else {
                continue;
            };
            if !select_choice_with_fma_delay.contains(&dep_idx) {
                continue;
            }
            let dep_op = &ops[dep_idx];
            let result_offset = if select_fma_ops.contains(&dep_op.id) {
                2
            } else {
                1
            };
            out.push(ScheduleConstraint::Precedence {
                consumer_idx: idx,
                dep_idx,
                result_offset,
                dep_is_shifted: false,
            });
        }
    }

    // 5b. Ordering groups: cycle[i] <= cycle[j] for consecutive in group
    let mut groups: HashMap<Vec<i64>, Vec<usize>> = HashMap::new();
    for (idx, op) in ops.iter().enumerate() {
        let Some(ref hint) = op.symmetry_hint else {
            continue;
        };
        if hint.ordering_group.is_empty() {
            continue;
        }
        groups
            .entry(hint.ordering_group.clone())
            .or_default()
            .push(idx);
    }
    for indices in groups.values() {
        for w in indices.windows(2) {
            let (i, j) = (w[0], w[1]);
            out.push(ScheduleConstraint::Ordering {
                before_idx: i,
                after_idx: j,
            });
        }
    }

    // 5c. Lexicographic streams: first_op[cur] <= first_op[next]
    let mut streams: HashMap<Vec<i64>, std::collections::BTreeMap<i64, Vec<usize>>> = HashMap::new();
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
    for by_item in streams.values() {
        let ordered: Vec<Vec<usize>> = by_item.values().cloned().collect();
        for w in ordered.windows(2) {
            let first_cur = w[0][0];
            let first_next = w[1][0];
            out.push(ScheduleConstraint::Ordering {
                before_idx: first_cur,
                after_idx: first_next,
            });
        }
    }

    out
}

/// Effective cycle for a dep: completion_cycle (max of 8) if shifted, else cycle_assignments[dep_idx].
pub(crate) fn dep_effective_cycle(
    dep_idx: usize,
    dep_is_shifted: bool,
    cycle_assignments: &[i32],
    shifted_alu_cycles: &HashMap<usize, Vec<i32>>,
    ops: &[SsaOp],
) -> i32 {
    if dep_is_shifted {
        let op_id = ops[dep_idx].id;
        shifted_alu_cycles
            .get(&op_id)
            .and_then(|v| v.iter().max().copied())
            .unwrap_or(cycle_assignments[dep_idx])
    } else {
        cycle_assignments[dep_idx]
    }
}

/// Check a cycle assignment against all constraints. Uses current cycle_assignments and
/// shifted_alu_cycles (for completion_cycle of shifted deps).
pub fn check_schedule_constraints(
    ops: &[SsaOp],
    cycle_assignments: &[i32],
    shifted_alu_cycles: &HashMap<usize, Vec<i32>>,
    constraints: &[ScheduleConstraint],
) -> Result<(), ConstraintViolation> {
    for c in constraints {
        match c {
            ScheduleConstraint::Precedence {
                consumer_idx,
                dep_idx,
                result_offset,
                dep_is_shifted,
            } => {
                let consumer_cycle = cycle_assignments.get(*consumer_idx).copied().unwrap_or(-1);
                let dep_cycle = dep_effective_cycle(
                    *dep_idx,
                    *dep_is_shifted,
                    cycle_assignments,
                    shifted_alu_cycles,
                    ops,
                );
                if consumer_cycle < dep_cycle + result_offset {
                    return Err(ConstraintViolation {
                        constraint: c.clone(),
                        message: format!(
                            "Precedence: op {} cycle {} < dep {} result_at {} (dep_cycle {} + {})",
                            consumer_idx,
                            consumer_cycle,
                            dep_idx,
                            dep_cycle + result_offset,
                            dep_cycle,
                            result_offset
                        ),
                    });
                }
            }
            ScheduleConstraint::Ordering {
                before_idx,
                after_idx,
            } => {
                let before_c = cycle_assignments.get(*before_idx).copied().unwrap_or(-1);
                let after_c = cycle_assignments.get(*after_idx).copied().unwrap_or(-1);
                if before_c > after_c {
                    return Err(ConstraintViolation {
                        constraint: c.clone(),
                        message: format!(
                            "Ordering: op {} cycle {} > op {} cycle {}",
                            before_idx, before_c, after_idx, after_c
                        ),
                    });
                }
            }
            ScheduleConstraint::Domain { idx, lb, ub } => {
                let cy = cycle_assignments.get(*idx).copied().unwrap_or(-1);
                if cy < *lb || cy > *ub {
                    return Err(ConstraintViolation {
                        constraint: c.clone(),
                        message: format!(
                            "Domain: op {} cycle {} not in [{}, {}]",
                            idx, cy, lb, ub
                        ),
                    });
                }
            }
        }
    }
    Ok(())
}

/// Constraints that are binding at the current solution (hold with equality).
/// Used to cross-check with bulk-move: when bulk-move blocks, the violation should be one of these.
pub fn binding_schedule_constraints(
    ops: &[SsaOp],
    cycle_assignments: &[i32],
    shifted_alu_cycles: &HashMap<usize, Vec<i32>>,
    constraints: &[ScheduleConstraint],
) -> Vec<(ScheduleConstraint, String)> {
    let mut out = Vec::new();
    for c in constraints {
        match c {
            ScheduleConstraint::Precedence {
                consumer_idx,
                dep_idx,
                result_offset,
                dep_is_shifted,
            } => {
                let consumer_cycle = cycle_assignments.get(*consumer_idx).copied().unwrap_or(-1);
                let dep_cycle = dep_effective_cycle(
                    *dep_idx,
                    *dep_is_shifted,
                    cycle_assignments,
                    shifted_alu_cycles,
                    ops,
                );
                if consumer_cycle == dep_cycle + result_offset {
                    out.push((
                        c.clone(),
                        format!(
                            "consumer {} cycle {} = dep {} effective {} + {}",
                            consumer_idx, consumer_cycle, dep_idx, dep_cycle, result_offset
                        ),
                    ));
                }
            }
            ScheduleConstraint::Ordering {
                before_idx,
                after_idx,
            } => {
                let before_c = cycle_assignments.get(*before_idx).copied().unwrap_or(-1);
                let after_c = cycle_assignments.get(*after_idx).copied().unwrap_or(-1);
                if before_c == after_c {
                    out.push((
                        c.clone(),
                        format!("before {} cycle {} = after {} cycle {}", before_idx, before_c, after_idx, after_c),
                    ));
                }
            }
            ScheduleConstraint::Domain { idx, lb, ub } => {
                let cy = cycle_assignments.get(*idx).copied().unwrap_or(-1);
                if cy == *lb || cy == *ub {
                    out.push((
                        c.clone(),
                        format!("op {} cycle {} at bound (lb={}, ub={})", idx, cy, lb, ub),
                    ));
                }
            }
        }
    }
    out
}

/// Result of one bulk-move simulation step or full run.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub enum BulkMoveResult {
    /// Reached end: empty cycles pushed to end; makespan could have been reduced.
    ReachedEnd {
        original_max_cycle_plus_one: i32,
        final_max_cycle_plus_one: i32,
        steps: usize,
    },
    /// A constraint blocked the move at empty C.
    Blocked {
        empty_c: i32,
        violation: ConstraintViolation,
        steps: usize,
    },
    /// No empty cycles.
    NoEmptyCycles,
}

/// Run the bulk-move simulator: repeatedly fill empty C by moving ops from C+1 to C,
/// checking constraints after each move. Uses the same constraints as CP-SAT.
///
/// Compacting by one cycle cannot worsen per-engine slot capacity: the same set of ops
/// that was at C+1 (feasible) is moved to C, so counts at C after the move equal counts
/// at C+1 before. We do not check capacity here; if a move ever violated capacity, that
/// would indicate a bug in our constraint or counting formulation, not invalid compaction.
///
/// **Bulk-move does not validate capacity lower bounds.** It only checks precedence,
/// ordering, and domain. So an incorrect redundant constraint (e.g. makespan >= wrong_bound)
/// will not be "found" by bulk-move — the schedule will compact successfully while the
/// solver's makespan stays wrong. Use the active-set diagnostic and the "empty cycles +
/// binding capacity" warning in the CP-SAT path to catch over-constraining bounds.
pub fn run_bulk_move_simulator(
    ops: &[SsaOp],
    cycle_assignments: &mut [i32],
    shifted_alu_cycles: &mut HashMap<usize, Vec<i32>>,
    constraints: &[ScheduleConstraint],
    total_cycles: i32,
    verbose: bool,
) -> BulkMoveResult {
    let used: HashSet<i32> = cycle_assignments.iter().copied().filter(|&c| c >= 0).collect();
    let empty_cycles: Vec<i32> = (0..total_cycles).filter(|c| !used.contains(c)).collect();
    if empty_cycles.is_empty() {
        return BulkMoveResult::NoEmptyCycles;
    }

    let mut c = empty_cycles[0];
    let mut steps = 0usize;
    let original_max = cycle_assignments.iter().copied().filter(|&x| x >= 0).max().unwrap_or(-1);

    loop {
        let max_cycle = cycle_assignments
            .iter()
            .copied()
            .filter(|&x| x >= 0)
            .max()
            .unwrap_or(-1);
        let ops_at_next: Vec<usize> = cycle_assignments
            .iter()
            .enumerate()
            .filter(|(_, &cy)| cy == c + 1)
            .map(|(i, _)| i)
            .collect();

        if ops_at_next.is_empty() {
            c += 1;
            if c + 1 > max_cycle {
                let final_max = cycle_assignments
                    .iter()
                    .copied()
                    .filter(|&x| x >= 0)
                    .max()
                    .unwrap_or(-1);
                return BulkMoveResult::ReachedEnd {
                    original_max_cycle_plus_one: original_max + 1,
                    final_max_cycle_plus_one: final_max + 1,
                    steps,
                };
            }
            continue;
        }

        // Hypothetical move: set all ops at C+1 to C. For shifted ops, decrement each ALU cycle >= c+1.
        for &idx in &ops_at_next {
            cycle_assignments[idx] = c;
            let op_id = ops[idx].id;
            if let Some(alu_cycles) = shifted_alu_cycles.get_mut(&op_id) {
                for cy in alu_cycles.iter_mut() {
                    if *cy >= c + 1 {
                        *cy -= 1;
                    }
                }
            }
        }

        if let Err(violation) = check_schedule_constraints(
            ops,
            cycle_assignments,
            shifted_alu_cycles,
            constraints,
        ) {
            // Revert the move
            for &idx in &ops_at_next {
                cycle_assignments[idx] = c + 1;
                let op_id = ops[idx].id;
                if let Some(alu_cycles) = shifted_alu_cycles.get_mut(&op_id) {
                    for cy in alu_cycles.iter_mut() {
                        if *cy >= c {
                            *cy += 1;
                        }
                    }
                }
            }
            return BulkMoveResult::Blocked {
                empty_c: c,
                violation,
                steps,
            };
        }

        steps += 1;
        if verbose {
            eprintln!(
                "  Bulk move: C={} moved {} ops from C+1 to C",
                c,
                ops_at_next.len()
            );
        }
        c += 1;
    }
}
