use std::collections::{HashMap, HashSet};

use crate::types::{AluDeriveInfo, Engine, MachineConfig, Opcode, SsaOp, VLEN};

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
    const_add_imm_ops: &HashSet<usize>,
    select_fma_ops: &HashSet<usize>,
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
            if const_add_imm_ops.contains(&op.id) {
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
            if !select_fma_ops.contains(&op.id) {
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
    const_add_imm_ops: &HashSet<usize>,
    select_fma_ops: &HashSet<usize>,
    cycle_assignments: &[i32],
) -> String {
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
    s.push_str("Solution decision sets (op.id):\n");
    s.push_str(&format!("  const_add_imm_ops = {:?}\n", const_add_imm_ops));
    s.push_str(&format!("  select_fma_ops   = {:?}\n", select_fma_ops));
    s.push_str("Interpretation: ConstChoice with op.id in const_add_imm_ops -> AddImm (Flow); ");
    s.push_str("SelectChoice with op.id NOT in select_fma_ops -> vselect (Flow).\n");
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
    const_add_imm_ops: &HashSet<usize>,
    const_alu_derive_ops: &HashMap<usize, AluDeriveInfo>,
    select_fma_ops: &HashSet<usize>,
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
            if const_alu_derive_ops.contains_key(&op.id) {
                alu_total += 1;
            } else if const_add_imm_ops.contains(&op.id) {
                flow_total += 1;
            } else {
                load_total += 1;
            }
            continue;
        }
        if op.opcode == Opcode::SelectChoice {
            if select_fma_ops.contains(&op.id) {
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

    let const_total = ops
        .iter()
        .filter(|o| o.opcode == Opcode::ConstChoice)
        .count();
    let const_add_imm = const_add_imm_ops.len();
    let const_alu_derive = const_alu_derive_ops.len();
    let const_load = const_total
        .saturating_sub(const_add_imm)
        .saturating_sub(const_alu_derive);

    let select_total = ops
        .iter()
        .filter(|o| o.opcode == Opcode::SelectChoice)
        .count();
    let select_fma = select_fma_ops.len();
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

/// Validate that a CP-SAT solution respects engine slot limits per cycle.
/// Panics with a descriptive message if any cycle exceeds limits (indicates a model bug).
/// Eliminated vbroadcast ops (when vb elimination is used) do not consume VALU slots and are not counted.
#[cfg(feature = "cpsat")]
pub(crate) fn validate_ssa_slot_limits(
    ops: &[SsaOp],
    cycle_assignments: &[i32],
    shifted_ops: &HashSet<usize>,
    shifted_alu_cycles: &HashMap<usize, Vec<i32>>,
    const_add_imm_ops: &HashSet<usize>,
    const_alu_derive_ops: &HashMap<usize, AluDeriveInfo>,
    select_fma_ops: &HashSet<usize>,
    eliminated_vbroadcast_ops: &HashSet<usize>,
    machine: &MachineConfig,
) {
    let makespan = cycle_assignments.iter().max().copied().unwrap_or(-1) + 1;
    if makespan <= 0 {
        return;
    }
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
            if const_alu_derive_ops.contains_key(&op.id) {
                alu_per_cycle[cycle] += 1;
            } else if const_add_imm_ops.contains(&op.id) {
                flow_per_cycle[cycle] += 1;
                if cycle < flow_contributors_per_cycle.len() {
                    flow_contributors_per_cycle[cycle]
                        .push((idx, format!("op{} ConstChoice→AddImm", op.id)));
                }
            } else {
                load_per_cycle[cycle] += 1;
            }
            continue;
        }
        if op.opcode == Opcode::SelectChoice {
            if select_fma_ops.contains(&op.id) {
                valu_per_cycle[cycle] += 1;
                if op.delta_vreg.is_none() {
                    let cycle2 = cycle + 1;
                    if cycle2 < valu_per_cycle.len() {
                        valu_per_cycle[cycle2] += 1;
                    }
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
                            const_add_imm_ops,
                            select_fma_ops,
                        );
                        format_flow_violation_breadcrumbs(
                            cycle,
                            count,
                            *limit as usize,
                            &flow_rows,
                            &flow_contributors_per_cycle,
                            const_add_imm_ops,
                            select_fma_ops,
                            cycle_assignments,
                        )
                    };
                    format!(
                        " contributors at cycle {}: {:?}.{}",
                        cycle, contribs, breadcrumbs
                    )
                } else {
                    String::new()
                };
                panic!(
                    "BUG: CP-SAT solution violates {} slot limit at cycle {}: {} > {} (model constraint error).{}",
                    name, cycle, count, limit, detail
                );
            }
        }
    }
}

/// Same as validate_ssa_slot_limits but returns Err(msg) instead of panicking. Used for diagnostic checks.
#[cfg(feature = "cpsat")]
pub(crate) fn try_validate_ssa_slot_limits(
    ops: &[SsaOp],
    cycle_assignments: &[i32],
    shifted_ops: &HashSet<usize>,
    shifted_alu_cycles: &HashMap<usize, Vec<i32>>,
    const_add_imm_ops: &HashSet<usize>,
    const_alu_derive_ops: &HashMap<usize, AluDeriveInfo>,
    select_fma_ops: &HashSet<usize>,
    eliminated_vbroadcast_ops: &HashSet<usize>,
    machine: &MachineConfig,
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
            if const_alu_derive_ops.contains_key(&op.id) {
                alu_per_cycle[cycle] += 1;
            } else if const_add_imm_ops.contains(&op.id) {
                flow_per_cycle[cycle] += 1;
            } else {
                load_per_cycle[cycle] += 1;
            }
            continue;
        }
        if op.opcode == Opcode::SelectChoice {
            if select_fma_ops.contains(&op.id) {
                valu_per_cycle[cycle] += 1;
                if op.delta_vreg.is_none() {
                    let cycle2 = cycle + 1;
                    if cycle2 < valu_per_cycle.len() {
                        valu_per_cycle[cycle2] += 1;
                    }
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

/// Verify that every consumer of a SelectChoice FMA (no-delta) result is scheduled at cycle >= dep_cycle + 2.
/// SelectChoice FMA emits sub at C and fma at C+1; result is visible at C+2. Panics if violated (model bug).
#[cfg(feature = "cpsat")]
pub(crate) fn verify_select_choice_fma_consumer_precedence(
    ops: &[SsaOp],
    cycle_assignments: &[i32],
    select_fma_ops: &HashSet<usize>,
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
                || !select_fma_ops.contains(&producer.id)
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

