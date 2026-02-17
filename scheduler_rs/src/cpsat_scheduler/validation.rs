use std::collections::{HashMap, HashSet};

use super::cse_groups::CseGroupResult;
use crate::types::{ConstChoiceEncoding, Engine, MachineConfig, Opcode, SelectChoiceEncoding, SsaOp, VLEN};

/// One row of the Flow audit: an op that contributes to the Flow engine at a given cycle.
#[cfg(feature = "cpsat")]
#[derive(Debug, Clone)]
pub(crate) struct FlowAuditRow {
    pub(crate) idx: usize,
    pub(crate) op_id: usize,
    pub(crate) opcode: Opcode,
    pub(crate) cycle: usize,
    pub(crate) reason: String,
}

/// Build full Flow audit: every op that contributes to Flow, with idx, op.id, opcode, cycle, reason.
/// Used for rigorous breadcrumb logging and panic messages.
#[cfg(feature = "cpsat")]
pub(crate) fn build_flow_audit(
    ops: &[SsaOp],
    cycle_assignments: &[i32],
    const_choice_encoding: &HashMap<usize, ConstChoiceEncoding>,
    select_choice_encoding: &HashMap<usize, SelectChoiceEncoding>,
) -> (Vec<FlowAuditRow>, Vec<i32>, Vec<Vec<(usize, String)>>) {
    let makespan = cycle_assignments.iter().max().copied().unwrap_or(-1) + 1;
    let makespan_usize = makespan.max(0) as usize;
    let mut flow_per_cycle: Vec<i32> = vec![0; makespan_usize];
    let mut flow_contributors_per_cycle: Vec<Vec<(usize, String)>> =
        vec![Vec::new(); makespan_usize];
    let mut all_flow_rows: Vec<FlowAuditRow> = Vec::new();

    for (idx, op) in ops.iter().enumerate() {
        let cycle = cycle_assignments.get(idx).copied().unwrap_or(0) as usize;
        if cycle >= flow_per_cycle.len() {
            continue;
        }

        if op.opcode == Opcode::ConstChoice {
            if matches!(const_choice_encoding.get(&op.id), Some(ConstChoiceEncoding::AddImm)) {
                let reason = format!("ConstChoice→AddImm (op.id={})", op.id);
                flow_per_cycle[cycle] += 1;
                if cycle < flow_contributors_per_cycle.len() {
                    flow_contributors_per_cycle[cycle].push((idx, reason.clone()));
                }
                all_flow_rows.push(FlowAuditRow {
                    idx,
                    op_id: op.id,
                    opcode: op.opcode,
                    cycle,
                    reason,
                });
            }
            continue;
        }
        if op.opcode == Opcode::SelectChoice {
            if !matches!(select_choice_encoding.get(&op.id), Some(SelectChoiceEncoding::Fma)) {
                let reason = format!("SelectChoice→vselect (op.id={})", op.id);
                flow_per_cycle[cycle] += 1;
                if cycle < flow_contributors_per_cycle.len() {
                    flow_contributors_per_cycle[cycle].push((idx, reason.clone()));
                }
                all_flow_rows.push(FlowAuditRow {
                    idx,
                    op_id: op.id,
                    opcode: op.opcode,
                    cycle,
                    reason,
                });
            }
            continue;
        }

        if op.engine == Engine::Flow {
            let reason = format!("Flow({:?}) op.id={}", op.opcode, op.id);
            flow_per_cycle[cycle] += 1;
            if cycle < flow_contributors_per_cycle.len() {
                flow_contributors_per_cycle[cycle].push((idx, reason.clone()));
            }
            all_flow_rows.push(FlowAuditRow {
                idx,
                op_id: op.id,
                opcode: op.opcode,
                cycle,
                reason,
            });
        }
    }

    (all_flow_rows, flow_per_cycle, flow_contributors_per_cycle)
}

/// Format a detailed Flow violation message for debugging (breadcrumbs).
#[cfg(feature = "cpsat")]
pub(crate) fn format_flow_violation_breadcrumbs(
    violating_cycle: usize,
    count: i32,
    limit: usize,
    flow_rows: &[FlowAuditRow],
    flow_contributors_per_cycle: &[Vec<(usize, String)>],
    const_choice_encoding: &HashMap<usize, ConstChoiceEncoding>,
    select_choice_encoding: &HashMap<usize, SelectChoiceEncoding>,
    cycle_assignments: &[i32],
) -> String {
    let add_imm_op_ids: Vec<usize> = const_choice_encoding
        .iter()
        .filter(|(_, enc)| matches!(enc, ConstChoiceEncoding::AddImm))
        .map(|(&id, _)| id)
        .collect();
    let select_fma_op_ids: Vec<usize> = select_choice_encoding
        .iter()
        .filter(|(_, enc)| matches!(enc, SelectChoiceEncoding::Fma))
        .map(|(&id, _)| id)
        .collect();
    let mut s = String::new();
    s.push_str(&format!(
        "\n\n=== Flow slot violation at cycle {}: {} > {} ===\n",
        violating_cycle, count, limit
    ));
    s.push_str("Contributors at this cycle (from flow_contributors_per_cycle):\n");
    if violating_cycle < flow_contributors_per_cycle.len() {
        for (idx, desc) in &flow_contributors_per_cycle[violating_cycle] {
            let cy = cycle_assignments.get(*idx).copied().unwrap_or(-1);
            s.push_str(&format!("  idx={} cycle_assignment={} {}\n", idx, cy, desc));
        }
    }
    s.push_str("All Flow-contributing ops in schedule (full audit):\n");
    for r in flow_rows {
        s.push_str(&format!(
            "  idx={} op.id={} opcode={:?} cycle={} reason={}\n",
            r.idx, r.op_id, r.opcode, r.cycle, r.reason
        ));
    }
    s.push_str(&format!(
        "Solution: ConstChoice AddImm op.ids = {:?}; SelectChoice Fma op.ids = {:?}\n",
        add_imm_op_ids, select_fma_op_ids
    ));
    s.push_str("=== end Flow violation breadcrumbs ===\n");
    s
}

/// Summary of engine usage and choice decisions for verbose reporting.
#[cfg(feature = "cpsat")]
pub(crate) struct SsaSolutionSummary {
    pub(crate) alu_total: usize,
    pub(crate) valu_total: usize,
    /// VALU slots from SelectChoice FMA path (sub+fma or fma only; not shiftable).
    pub(crate) valu_fma_slots: usize,
    /// VALU slots from Vbroadcast ops (not shiftable).
    pub(crate) valu_vbroadcast_slots: usize,
    /// VALU slots from other VALU ops (arithmetic etc.; shiftable when can_shift_to_alu).
    pub(crate) valu_other_slots: usize,
    pub(crate) load_total: usize,
    pub(crate) store_total: usize,
    pub(crate) flow_total: usize,
    pub(crate) const_add_imm: usize,
    pub(crate) const_load: usize,
    pub(crate) const_alu_derive: usize,
    pub(crate) select_fma: usize,
    pub(crate) select_vselect: usize,
}

#[cfg(feature = "cpsat")]
pub(crate) fn summarise_ssa_solution_engines_and_choices(
    ops: &[SsaOp],
    cycle_assignments: &[i32],
    shifted_ops: &HashSet<usize>,
    const_choice_encoding: &HashMap<usize, ConstChoiceEncoding>,
    select_choice_encoding: &HashMap<usize, SelectChoiceEncoding>,
) -> SsaSolutionSummary {
    let mut alu_total = 0usize;
    let mut valu_total = 0usize;
    let mut valu_fma_slots = 0usize;
    let mut valu_vbroadcast_slots = 0usize;
    let mut valu_other_slots = 0usize;
    let mut load_total = 0usize;
    let mut store_total = 0usize;
    let mut flow_total = 0usize;

    for (idx, op) in ops.iter().enumerate() {
        let cycle = cycle_assignments.get(idx).copied().unwrap_or(0);
        if cycle < 0 {
            continue;
        }
        let _cycle = cycle as usize;

        if op.opcode == Opcode::ConstChoice {
            match const_choice_encoding.get(&op.id) {
                Some(ConstChoiceEncoding::AluDerive(_)) => alu_total += 1,
                Some(ConstChoiceEncoding::AddImm) => flow_total += 1,
                Some(ConstChoiceEncoding::Load) | None => load_total += 1,
            }
            continue;
        }
        if op.opcode == Opcode::SelectChoice {
            if matches!(select_choice_encoding.get(&op.id), Some(SelectChoiceEncoding::Fma)) {
                valu_total += 1;
                valu_fma_slots += 1;
                if op.delta_vreg.is_none() {
                    valu_total += 1;
                    valu_fma_slots += 1;
                }
            } else {
                flow_total += 1;
            }
            continue;
        }

        match op.engine {
            Engine::Alu => alu_total += 1,
            Engine::Valu => {
                if shifted_ops.contains(&op.id) {
                    alu_total += VLEN as usize;
                } else {
                    valu_total += 1;
                    if op.opcode == Opcode::Vbroadcast {
                        valu_vbroadcast_slots += 1;
                    } else {
                        valu_other_slots += 1;
                    }
                }
            }
            Engine::Load => load_total += 1,
            Engine::Store => store_total += 1,
            Engine::Flow => flow_total += 1,
            _ => {}
        }
    }

    let const_add_imm = const_choice_encoding
        .values()
        .filter(|enc| matches!(enc, ConstChoiceEncoding::AddImm))
        .count();
    let const_alu_derive = const_choice_encoding
        .values()
        .filter(|enc| matches!(enc, ConstChoiceEncoding::AluDerive(_)))
        .count();
    let const_load = const_choice_encoding
        .values()
        .filter(|enc| matches!(enc, ConstChoiceEncoding::Load))
        .count();

    let select_total = ops
        .iter()
        .filter(|o| o.opcode == Opcode::SelectChoice)
        .count();
    let select_fma = select_choice_encoding
        .values()
        .filter(|e| matches!(e, SelectChoiceEncoding::Fma))
        .count();
    let select_vselect = select_total.saturating_sub(select_fma);

    SsaSolutionSummary {
        alu_total,
        valu_total,
        valu_fma_slots,
        valu_vbroadcast_slots,
        valu_other_slots,
        load_total,
        store_total,
        flow_total,
        const_add_imm,
        const_load,
        const_alu_derive,
        select_fma,
        select_vselect,
    }
}

/// Validate that shifted_alu_cycles has exactly VLEN entries per shifted op and each cycle is in [0, makespan).
#[cfg(feature = "cpsat")]
pub(crate) fn validate_shifted_alu_cycles(
    shifted_ops: &HashSet<usize>,
    shifted_alu_cycles: &HashMap<usize, Vec<i32>>,
    makespan: i32,
) {
    let makespan_usize = makespan.max(0) as usize;
    for &op_id in shifted_ops {
        if let Some(cycles) = shifted_alu_cycles.get(&op_id) {
            assert_eq!(
                cycles.len(),
                VLEN,
                "BUG: shifted op id={} has {} ALU cycles, expected {} (VLEN)",
                op_id,
                cycles.len(),
                VLEN
            );
            for (lane, &c) in cycles.iter().enumerate() {
                assert!(
                    c >= 0 && (c as usize) < makespan_usize,
                    "BUG: shifted op id={} lane {} has cycle {} out of range [0, {})",
                    op_id,
                    lane,
                    c,
                    makespan_usize
                );
            }
        }
    }
}

/// Validate that a CP-SAT solution respects engine slot limits per cycle.
/// Returns `Err(msg)` if any cycle exceeds limits (indicates a model bug).
/// Eliminated vbroadcast ops (when vb elimination is used) do not consume VALU slots and are not counted.
///
/// When `cse_group_results` and `op_idx_to_cse_group` are provided and non-empty, validation uses
/// CSE-aware counting: for SelectChoice FMA (no-delta) ops in CSE groups, only the FMA at cycle+1
/// is counted per consumer; the shared SUB is counted once per active group at its sub_cycle.
/// Without CSE info, each SelectChoice FMA (no-delta) is counted as sub at cycle + fma at cycle+1
/// (which would over-count when CSE is used, causing false validation failures).
#[cfg(feature = "cpsat")]
pub(crate) fn validate_ssa_slot_limits(
    ops: &[SsaOp],
    cycle_assignments: &[i32],
    shifted_ops: &HashSet<usize>,
    shifted_alu_cycles: &HashMap<usize, Vec<i32>>,
    const_choice_encoding: &HashMap<usize, ConstChoiceEncoding>,
    select_choice_encoding: &HashMap<usize, SelectChoiceEncoding>,
    eliminated_vbroadcast_ops: &HashSet<usize>,
    machine: &MachineConfig,
    cse_group_results: &[CseGroupResult],
    op_idx_to_cse_group: &HashMap<usize, usize>,
) -> Result<(), String> {
    let makespan = cycle_assignments.iter().max().copied().unwrap_or(-1) + 1;
    if makespan <= 0 {
        return Ok(());
    }
    validate_shifted_alu_cycles(shifted_ops, shifted_alu_cycles, makespan);
    let mut alu_per_cycle: Vec<i32> = vec![0; makespan as usize];
    let mut valu_per_cycle: Vec<i32> = vec![0; makespan as usize];
    let mut load_per_cycle: Vec<i32> = vec![0; makespan as usize];
    let mut store_per_cycle: Vec<i32> = vec![0; makespan as usize];
    let mut flow_per_cycle: Vec<i32> = vec![0; makespan as usize];
    // For Flow violations: which ops contribute Flow at each cycle (for diagnostics).
    let mut flow_contributors_per_cycle: Vec<Vec<(usize, String)>> =
        vec![Vec::new(); makespan as usize];

    for (idx, op) in ops.iter().enumerate() {
        let cycle = cycle_assignments.get(idx).copied().unwrap_or(0) as usize;
        if cycle >= alu_per_cycle.len() {
            continue;
        }

        if op.opcode == Opcode::ConstChoice {
            match const_choice_encoding.get(&op.id) {
                Some(ConstChoiceEncoding::AluDerive(_)) => {
                    alu_per_cycle[cycle] += 1;
                }
                Some(ConstChoiceEncoding::AddImm) => {
                    flow_per_cycle[cycle] += 1;
                    if cycle < flow_contributors_per_cycle.len() {
                        flow_contributors_per_cycle[cycle]
                            .push((idx, format!("op{} ConstChoice→AddImm", op.id)));
                    }
                }
                Some(ConstChoiceEncoding::Load) | None => {
                    load_per_cycle[cycle] += 1;
                }
            }
            continue;
        }
        if op.opcode == Opcode::SelectChoice {
            if matches!(select_choice_encoding.get(&op.id), Some(SelectChoiceEncoding::Fma)) {
                let in_cse_group = op_idx_to_cse_group.contains_key(&idx);
                if op.delta_vreg.is_none() {
                    // No-delta: sub at cycle C, fma at C+1. CSE groups share one SUB.
                    if in_cse_group {
                        // Only count FMA at cycle+1; shared SUB counted separately
                        let cycle2 = cycle + 1;
                        if cycle2 < valu_per_cycle.len() {
                            valu_per_cycle[cycle2] += 1;
                        }
                    } else {
                        valu_per_cycle[cycle] += 1;
                        let cycle2 = cycle + 1;
                        if cycle2 < valu_per_cycle.len() {
                            valu_per_cycle[cycle2] += 1;
                        }
                    }
                } else {
                    // Delta: single VALU at cycle
                    valu_per_cycle[cycle] += 1;
                }
            } else {
                flow_per_cycle[cycle] += 1;
                if cycle < flow_contributors_per_cycle.len() {
                    flow_contributors_per_cycle[cycle]
                        .push((idx, format!("op{} SelectChoice→vselect", op.id)));
                }
            }
            continue;
        }

        match op.engine {
            Engine::Alu => alu_per_cycle[cycle] += 1,
            Engine::Valu => {
                if shifted_ops.contains(&op.id) {
                    // Count each of the 8 ALU ops at their actual cycles (not all at completion_cycle)
                    if let Some(alu_cycles) = shifted_alu_cycles.get(&op.id) {
                        for &alu_cycle in alu_cycles {
                            let alu_cycle_usize = alu_cycle as usize;
                            if alu_cycle_usize < alu_per_cycle.len() {
                                alu_per_cycle[alu_cycle_usize] += 1;
                            }
                        }
                    } else {
                        // Fallback: if shifted_alu_cycles not available, count all 8 at completion_cycle
                        // This shouldn't happen, but provides backward compatibility
                        alu_per_cycle[cycle] += VLEN as i32;
                    }
                } else {
                    // Eliminated vbroadcasts do not use a VALU slot (optional interval absent)
                    if !eliminated_vbroadcast_ops.contains(&op.id) {
                        valu_per_cycle[cycle] += 1;
                    }
                }
            }
            Engine::Load => load_per_cycle[cycle] += 1,
            Engine::Store => store_per_cycle[cycle] += 1,
            Engine::Flow => {
                flow_per_cycle[cycle] += 1;
                if cycle < flow_contributors_per_cycle.len() {
                    flow_contributors_per_cycle[cycle]
                        .push((idx, format!("op{} Flow({:?})", op.id, op.opcode)));
                }
            }
            _ => {}
        }
    }

    // CSE shared SUB: one VALU slot per active group at sub_cycle (model has one interval per group)
    for result in cse_group_results {
        if result.sub_active {
            let sub_cycle = result.sub_cycle as usize;
            if sub_cycle < valu_per_cycle.len() {
                valu_per_cycle[sub_cycle] += 1;
            }
        }
    }

    let limits: [(u32, &str); 5] = [
        (machine.slot_limit(Engine::Alu), "ALU"),
        (machine.slot_limit(Engine::Valu), "VALU"),
        (machine.slot_limit(Engine::Load), "Load"),
        (machine.slot_limit(Engine::Store), "Store"),
        (machine.slot_limit(Engine::Flow), "Flow"),
    ];
    let counts = [
        &alu_per_cycle,
        &valu_per_cycle,
        &load_per_cycle,
        &store_per_cycle,
        &flow_per_cycle,
    ];
    for (cycle, _) in alu_per_cycle.iter().enumerate() {
        for (limit_idx, (limit, name)) in limits.iter().enumerate() {
            let count = counts[limit_idx][cycle];
            if count > *limit as i32 {
                let detail = if *name == "Flow" && cycle < flow_contributors_per_cycle.len() {
                    let contribs: Vec<String> = flow_contributors_per_cycle[cycle]
                        .iter()
                        .map(|(i, desc)| format!("idx={} {}", i, desc))
                        .collect();
                    let breadcrumbs = {
                        let (flow_rows, _, _) = build_flow_audit(
                            ops,
                            cycle_assignments,
                            const_choice_encoding,
                            select_choice_encoding,
                        );
                        format_flow_violation_breadcrumbs(
                            cycle,
                            count,
                            *limit as usize,
                            &flow_rows,
                            &flow_contributors_per_cycle,
                            const_choice_encoding,
                            select_choice_encoding,
                            cycle_assignments,
                        )
                    };
                    format!(
                        " contributors at cycle {}: {:?}.{}",
                        cycle, contribs, breadcrumbs
                    )
                } else if *name == "VALU" {
                    // List ops contributing to VALU at this cycle for debugging
                    let contribs: Vec<String> = ops
                        .iter()
                        .enumerate()
                        .filter(|(idx, op)| {
                            let c = cycle_assignments.get(*idx).copied().unwrap_or(-1) as i32;
                            if c < 0 {
                                return false;
                            }
                            let contribs_to_cycle = |target: usize| -> bool {
                                if op.engine == Engine::Valu
                                    && !shifted_ops.contains(&op.id)
                                    && !eliminated_vbroadcast_ops.contains(&op.id)
                                {
                                    c as usize == target
                                } else if op.opcode == Opcode::SelectChoice
                                    && matches!(select_choice_encoding.get(&op.id), Some(SelectChoiceEncoding::Fma))
                                {
                                    c as usize == target
                                        || (op.delta_vreg.is_none() && c as usize + 1 == target)
                                } else {
                                    false
                                }
                            };
                            contribs_to_cycle(cycle)
                        })
                        .map(|(idx, op)| format!("idx={} op.id={} {:?}", idx, op.id, op.opcode))
                        .collect();
                    format!(" contributors at cycle {}: {:?}", cycle, contribs)
                } else {
                    String::new()
                };
                return Err(format!(
                    "CP-SAT solution violates {} slot limit at cycle {}: {} > {} (model constraint error).{}",
                    name, cycle, count, limit, detail
                ));
            }
        }
    }
    Ok(())
}

/// Same as validate_ssa_slot_limits but returns Err(msg) instead of panicking. Used for diagnostic checks.
#[cfg(feature = "cpsat")]
pub(crate) fn try_validate_ssa_slot_limits(
    ops: &[SsaOp],
    cycle_assignments: &[i32],
    shifted_ops: &HashSet<usize>,
    shifted_alu_cycles: &HashMap<usize, Vec<i32>>,
    const_choice_encoding: &HashMap<usize, ConstChoiceEncoding>,
    select_choice_encoding: &HashMap<usize, SelectChoiceEncoding>,
    eliminated_vbroadcast_ops: &HashSet<usize>,
    machine: &MachineConfig,
    cse_group_results: &[CseGroupResult],
    op_idx_to_cse_group: &HashMap<usize, usize>,
) -> Result<(), String> {
    let makespan = cycle_assignments.iter().max().copied().unwrap_or(-1) + 1;
    if makespan <= 0 {
        return Ok(());
    }
    let mut alu_per_cycle: Vec<i32> = vec![0; makespan as usize];
    let mut valu_per_cycle: Vec<i32> = vec![0; makespan as usize];
    let mut load_per_cycle: Vec<i32> = vec![0; makespan as usize];
    let mut store_per_cycle: Vec<i32> = vec![0; makespan as usize];
    let mut flow_per_cycle: Vec<i32> = vec![0; makespan as usize];

    for (idx, op) in ops.iter().enumerate() {
        let cycle = cycle_assignments.get(idx).copied().unwrap_or(0) as usize;
        if cycle >= alu_per_cycle.len() {
            continue;
        }
        if op.opcode == Opcode::ConstChoice {
            match const_choice_encoding.get(&op.id) {
                Some(ConstChoiceEncoding::AluDerive(_)) => alu_per_cycle[cycle] += 1,
                Some(ConstChoiceEncoding::AddImm) => flow_per_cycle[cycle] += 1,
                Some(ConstChoiceEncoding::Load) | None => load_per_cycle[cycle] += 1,
            }
            continue;
        }
        if op.opcode == Opcode::SelectChoice {
            if matches!(select_choice_encoding.get(&op.id), Some(SelectChoiceEncoding::Fma)) {
                let in_cse_group = op_idx_to_cse_group.contains_key(&idx);
                if op.delta_vreg.is_none() {
                    if in_cse_group {
                        let cycle2 = cycle + 1;
                        if cycle2 < valu_per_cycle.len() {
                            valu_per_cycle[cycle2] += 1;
                        }
                    } else {
                        valu_per_cycle[cycle] += 1;
                        let cycle2 = cycle + 1;
                        if cycle2 < valu_per_cycle.len() {
                            valu_per_cycle[cycle2] += 1;
                        }
                    }
                } else {
                    valu_per_cycle[cycle] += 1;
                }
            } else {
                flow_per_cycle[cycle] += 1;
            }
            continue;
        }
        match op.engine {
            Engine::Alu => alu_per_cycle[cycle] += 1,
            Engine::Valu => {
                if shifted_ops.contains(&op.id) {
                    if let Some(alu_cycles) = shifted_alu_cycles.get(&op.id) {
                        for &alu_cycle in alu_cycles {
                            let alu_cycle_usize = alu_cycle as usize;
                            if alu_cycle_usize < alu_per_cycle.len() {
                                alu_per_cycle[alu_cycle_usize] += 1;
                            }
                        }
                    } else {
                        alu_per_cycle[cycle] += VLEN as i32;
                    }
                } else {
                    if !eliminated_vbroadcast_ops.contains(&op.id) {
                        valu_per_cycle[cycle] += 1;
                    }
                }
            }
            Engine::Load => load_per_cycle[cycle] += 1,
            Engine::Store => store_per_cycle[cycle] += 1,
            Engine::Flow => flow_per_cycle[cycle] += 1,
            _ => {}
        }
    }

    for result in cse_group_results {
        if result.sub_active {
            let sub_cycle = result.sub_cycle as usize;
            if sub_cycle < valu_per_cycle.len() {
                valu_per_cycle[sub_cycle] += 1;
            }
        }
    }

    let limits: [(u32, &str); 5] = [
        (machine.slot_limit(Engine::Alu), "ALU"),
        (machine.slot_limit(Engine::Valu), "VALU"),
        (machine.slot_limit(Engine::Load), "Load"),
        (machine.slot_limit(Engine::Store), "Store"),
        (machine.slot_limit(Engine::Flow), "Flow"),
    ];
    let counts = [
        &alu_per_cycle,
        &valu_per_cycle,
        &load_per_cycle,
        &store_per_cycle,
        &flow_per_cycle,
    ];
    for (cycle, _) in alu_per_cycle.iter().enumerate() {
        for (limit_idx, (limit, name)) in limits.iter().enumerate() {
            let count = counts[limit_idx][cycle];
            if count > *limit as i32 {
                return Err(format!(
                    "{} slot limit at cycle {}: {} > {}",
                    name, cycle, count, limit
                ));
            }
        }
    }
    Ok(())
}

/// Verify that every RAW dependency satisfies cycle[consumer] >= producer_visible_cycle.
/// Producer visible cycle: SelectChoice FMA => producer_cycle+2; shifted => completion_cycle+1; else producer_cycle+1.
#[cfg(feature = "cpsat")]
pub(crate) fn verify_all_raw_precedence(
    ops: &[SsaOp],
    cycle_assignments: &[i32],
    shifted_ops: &HashSet<usize>,
    shifted_alu_cycles: &HashMap<usize, Vec<i32>>,
    select_choice_encoding: &HashMap<usize, SelectChoiceEncoding>,
) {
    let id_to_idx: HashMap<usize, usize> = ops
        .iter()
        .enumerate()
        .map(|(idx, op)| (op.id, idx))
        .collect();
    for (consumer_idx, op) in ops.iter().enumerate() {
        let consumer_cycle = cycle_assignments.get(consumer_idx).copied().unwrap_or(-1);
        for &dep_id in &op.dep_ids {
            let Some(&producer_idx) = id_to_idx.get(&dep_id) else {
                continue;
            };
            let producer = &ops[producer_idx];
            let producer_cycle = cycle_assignments.get(producer_idx).copied().unwrap_or(-1);
            let producer_visible_cycle = if producer.opcode == Opcode::SelectChoice
                && producer.sub_temp_vreg.is_some()
                && matches!(select_choice_encoding.get(&producer.id), Some(SelectChoiceEncoding::Fma))
            {
                producer_cycle + 2
            } else if shifted_ops.contains(&producer.id) {
                let completion = shifted_alu_cycles
                    .get(&producer.id)
                    .and_then(|v| v.iter().max().copied())
                    .unwrap_or(producer_cycle);
                completion + 1
            } else {
                producer_cycle + 1
            };
            assert!(
                consumer_cycle >= producer_visible_cycle,
                "BUG: RAW precedence violation: op id={} (consumer) at cycle {} depends on op id={} (producer), result visible at cycle {}",
                op.id,
                consumer_cycle,
                producer.id,
                producer_visible_cycle
            );
        }
    }
}

/// Verify that every consumer of a SelectChoice FMA (no-delta) result is scheduled at cycle >= dep_cycle + 2.
/// SelectChoice FMA emits sub at C and fma at C+1; result is visible at C+2. Panics if violated (model bug).
#[cfg(feature = "cpsat")]
pub(crate) fn verify_select_choice_fma_consumer_precedence(
    ops: &[SsaOp],
    cycle_assignments: &[i32],
    select_choice_encoding: &HashMap<usize, SelectChoiceEncoding>,
) {
    let id_to_idx: HashMap<usize, usize> = ops
        .iter()
        .enumerate()
        .map(|(idx, op)| (op.id, idx))
        .collect();
    for (consumer_idx, op) in ops.iter().enumerate() {
        let consumer_cycle = cycle_assignments.get(consumer_idx).copied().unwrap_or(-1);
        for &dep_id in &op.dep_ids {
            let Some(&producer_idx) = id_to_idx.get(&dep_id) else {
                continue;
            };
            let producer = &ops[producer_idx];
            if producer.opcode != Opcode::SelectChoice
                || !matches!(select_choice_encoding.get(&producer.id), Some(SelectChoiceEncoding::Fma))
                || producer.sub_temp_vreg.is_none()
            {
                continue;
            }
            let producer_cycle = cycle_assignments.get(producer_idx).copied().unwrap_or(-1);
            assert!(
                consumer_cycle >= producer_cycle + 2,
                "BUG: consumer of SelectChoice FMA (op.id={}) at cycle {} but result not visible until cycle {} (dep at {})",
                producer.id,
                consumer_cycle,
                producer_cycle + 2,
                producer_cycle
            );
        }
    }
}

