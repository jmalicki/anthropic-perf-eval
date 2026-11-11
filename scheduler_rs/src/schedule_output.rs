//! Schedule output formatting and JSON serialization.
//!
//! Pure Rust logic for building schedule_output.json and SSA schedule JSON.
//! No PyO3 types; used by the Python FFI layer.

use serde::Serialize;
use std::collections::{BTreeSet, HashMap, HashSet};

use crate::address_assign::{
    build_vreg_debug_info, compute_liveness_curve, compute_liveness_summary, CycleLiveness,
    LivenessSummary, VRegDebugInfo,
};
use crate::scheduler::PriorityMode;
use crate::types::{Engine, Op, SlotVal};

// =============================================================================
// Non-SSA schedule output (schedule_to_json)
// =============================================================================

/// Format an address with prefix for readability
pub fn format_addr(val: i64, context: &str, slot_idx: usize) -> String {
    match context {
        "const" => {
            if slot_idx == 1 {
                format!("s{}", val)
            } else {
                format!("i{}", val)
            }
        }
        "load" | "vload" | "store" | "vstore" => format!("s{}", val),
        "vbroadcast" => {
            if slot_idx == 1 {
                format!("v{}", val)
            } else {
                format!("s{}", val)
            }
        }
        "vselect" | "multiply_add" | "vmultiply_add" => format!("v{}", val),
        "+" | "-" | "*" | "/" | "%" | "&" | "|" | "^" | "<<" | ">>" | "==" | "!=" | "<" | "<="
        | ">" | ">=" => format!("s{}", val),
        "add_imm" => {
            if slot_idx == 3 {
                format!("i{}", val)
            } else {
                format!("s{}", val)
            }
        }
        "pause" | "jump" | "branch" => format!("p{}", val),
        _ => val.to_string(),
    }
}

/// Format an op's compact string with an optional comment (deps, dependents, const)
pub fn format_op_with_comment(
    op: &Op,
    compact: String,
    op_id_to_cycle: &HashMap<usize, i32>,
    dependents_map: &HashMap<usize, Vec<usize>>,
) -> String {
    let mut comment_parts: Vec<String> = Vec::new();

    if let Some(c) = op.produces_constant {
        comment_parts.push(format!("const={}", c));
    }

    let dep_cycles: BTreeSet<i32> = op
        .dep_ids
        .iter()
        .filter_map(|id| op_id_to_cycle.get(id).copied())
        .collect();
    if !dep_cycles.is_empty() {
        let cycles_vec: Vec<i32> = dep_cycles.into_iter().collect();
        comment_parts.push(format!("deps={:?}", cycles_vec));
    }

    if let Some(deps) = dependents_map.get(&op.id) {
        let dependent_cycles: BTreeSet<i32> = deps
            .iter()
            .filter_map(|id| op_id_to_cycle.get(id).copied())
            .collect();
        if !dependent_cycles.is_empty() {
            let cycles_vec: Vec<i32> = dependent_cycles.into_iter().collect();
            comment_parts.push(format!("dependents={:?}", cycles_vec));
        }
    }

    if comment_parts.is_empty() {
        compact
    } else {
        format!("{} // {}", compact, comment_parts.join(" "))
    }
}

#[derive(Serialize)]
pub(crate) struct CycleOutput {
    cycle: usize,
    load: Vec<String>,
    load_empty: usize,
    store: Vec<String>,
    store_empty: usize,
    alu: Vec<String>,
    alu_empty: usize,
    valu: Vec<String>,
    valu_empty: usize,
    flow: Vec<String>,
    flow_empty: usize,
}

#[derive(Serialize)]
pub(crate) struct ScheduleOutput {
    total_cycles: usize,
    total_ops: usize,
    best_mode: String,
    slot_limits: HashMap<String, usize>,
    cycles: Vec<CycleOutput>,
    summary: ScheduleSummary,
}

#[derive(Serialize)]
pub(crate) struct ScheduleSummary {
    load_utilization_pct: f64,
    store_utilization_pct: f64,
    alu_utilization_pct: f64,
    valu_utilization_pct: f64,
    total_empty_load: usize,
    total_empty_store: usize,
    total_empty_alu: usize,
    total_empty_valu: usize,
}

/// Build schedule output struct from scheduled ops (for schedule_to_json).
pub(crate) fn build_schedule_output(
    ops: &[Op],
    expanded_ops: &[Op],
    max_cycles: usize,
    best_mode: PriorityMode,
) -> ScheduleOutput {
    let mut op_id_to_cycle: HashMap<usize, i32> = HashMap::new();
    for op in ops {
        op_id_to_cycle.insert(op.id, op.cycle);
    }
    for op in expanded_ops {
        op_id_to_cycle.insert(op.id, op.cycle);
    }

    let mut dependents_map: HashMap<usize, Vec<usize>> = HashMap::new();
    for op in ops.iter().chain(expanded_ops.iter()) {
        for &dep_id in &op.dep_ids {
            dependents_map.entry(dep_id).or_default().push(op.id);
        }
    }

    let mut cycles_output: Vec<CycleOutput> = Vec::new();
    let mut total_empty_load = 0usize;
    let mut total_empty_store = 0usize;
    let mut total_empty_alu = 0usize;
    let mut total_empty_valu = 0usize;
    let mut total_load = 0usize;
    let mut total_store = 0usize;
    let mut total_alu = 0usize;
    let mut total_valu = 0usize;

    for cycle in 0..max_cycles {
        let mut load_ops: Vec<String> = Vec::new();
        let mut store_ops: Vec<String> = Vec::new();
        let mut alu_ops: Vec<String> = Vec::new();
        let mut valu_ops: Vec<String> = Vec::new();
        let mut flow_ops: Vec<String> = Vec::new();

        for op in ops {
            if op.cycle == cycle as i32
                && op.engine != Engine::ShiftedValu
                && op.engine != Engine::EliminatedBroadcast
            {
                let opcode = op
                    .slot
                    .get(0)
                    .map(|v| match v {
                        SlotVal::Str(s) => s.as_str(),
                        SlotVal::Int(_) => "",
                    })
                    .unwrap_or("");

                let parts: Vec<String> = op
                    .slot
                    .iter()
                    .enumerate()
                    .map(|(idx, v)| match v {
                        SlotVal::Str(s) => s.clone(),
                        SlotVal::Int(n) => format_addr(*n, opcode, idx),
                    })
                    .collect();
                let compact = parts.join(" ");
                let formatted =
                    format_op_with_comment(op, compact, &op_id_to_cycle, &dependents_map);

                match op.engine {
                    Engine::Load => load_ops.push(formatted),
                    Engine::Store => store_ops.push(formatted),
                    Engine::Alu => alu_ops.push(formatted),
                    Engine::Valu => valu_ops.push(formatted),
                    Engine::Flow => flow_ops.push(formatted),
                    _ => {}
                }
            }
        }

        for op in expanded_ops {
            if op.cycle == cycle as i32 {
                let opcode = op
                    .slot
                    .get(0)
                    .map(|v| match v {
                        SlotVal::Str(s) => s.as_str(),
                        SlotVal::Int(_) => "",
                    })
                    .unwrap_or("");

                let parts: Vec<String> = op
                    .slot
                    .iter()
                    .enumerate()
                    .map(|(idx, v)| match v {
                        SlotVal::Str(s) => s.clone(),
                        SlotVal::Int(n) => format_addr(*n, opcode, idx),
                    })
                    .collect();
                let compact = parts.join(" ");
                let formatted =
                    format_op_with_comment(op, compact, &op_id_to_cycle, &dependents_map);

                alu_ops.push(formatted);
            }
        }

        let load_empty = 2 - load_ops.len().min(2);
        let store_empty = 2 - store_ops.len().min(2);
        let alu_empty = 12 - alu_ops.len().min(12);
        let valu_empty = 6 - valu_ops.len().min(6);
        let flow_empty = 1 - flow_ops.len().min(1);

        total_load += load_ops.len();
        total_store += store_ops.len();
        total_alu += alu_ops.len();
        total_valu += valu_ops.len();
        total_empty_load += load_empty;
        total_empty_store += store_empty;
        total_empty_alu += alu_empty;
        total_empty_valu += valu_empty;

        cycles_output.push(CycleOutput {
            cycle,
            load: load_ops,
            load_empty,
            store: store_ops,
            store_empty,
            alu: alu_ops,
            alu_empty,
            valu: valu_ops,
            valu_empty,
            flow: flow_ops,
            flow_empty,
        });
    }

    let total_load_slots = max_cycles * 2;
    let total_store_slots = max_cycles * 2;
    let total_alu_slots = max_cycles * 12;
    let total_valu_slots = max_cycles * 6;

    let summary = ScheduleSummary {
        load_utilization_pct: (total_load as f64 / total_load_slots as f64) * 100.0,
        store_utilization_pct: (total_store as f64 / total_store_slots as f64) * 100.0,
        alu_utilization_pct: (total_alu as f64 / total_alu_slots as f64) * 100.0,
        valu_utilization_pct: (total_valu as f64 / total_valu_slots as f64) * 100.0,
        total_empty_load,
        total_empty_store,
        total_empty_alu,
        total_empty_valu,
    };

    let mut slot_limits = HashMap::new();
    slot_limits.insert("load".to_string(), 2);
    slot_limits.insert("store".to_string(), 2);
    slot_limits.insert("alu".to_string(), 12);
    slot_limits.insert("valu".to_string(), 6);
    slot_limits.insert("flow".to_string(), 1);

    ScheduleOutput {
        total_cycles: max_cycles,
        total_ops: ops.len() + expanded_ops.len(),
        best_mode: format!("{:?}", best_mode),
        slot_limits,
        cycles: cycles_output,
        summary,
    }
}

/// Build schedule output and return JSON string.
pub fn build_schedule_output_json(
    ops: &[Op],
    expanded_ops: &[Op],
    max_cycles: usize,
    best_mode: PriorityMode,
) -> String {
    let output = build_schedule_output(ops, expanded_ops, max_cycles, best_mode);
    serde_json::to_string_pretty(&output).unwrap_or_else(|_| "{}".to_string())
}

/// Build schedule output and write JSON to path.
pub fn write_schedule_output_to_path(
    ops: &[Op],
    expanded_ops: &[Op],
    max_cycles: usize,
    best_mode: PriorityMode,
    path: &str,
) -> std::io::Result<()> {
    let output = build_schedule_output(ops, expanded_ops, max_cycles, best_mode);
    let json = serde_json::to_string_pretty(&output).unwrap_or_else(|_| "{}".to_string());
    std::fs::write(path, &json)
}

// =============================================================================
// SSA schedule output (build_tick_tock_kernel_ssa output_json)
// =============================================================================

#[derive(Serialize)]
struct SsaScheduleOutput {
    total_cycles: usize,
    total_ops: usize,
    vreg_count: u32,
    peak_live_words: usize,
    max_addr: i64,
    budget_exceeded: bool,
    valu_shifted: usize,
    usage_accounting: crate::types::UsageAccounting,
    ops: Vec<OpDebugInfo>,
    vreg_to_addr: std::collections::HashMap<String, i64>,
    liveness_summary: LivenessSummary,
    #[serde(skip_serializing_if = "Option::is_none")]
    vregs: Option<Vec<VRegDebugInfo>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    liveness_curve: Option<Vec<CycleLiveness>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    valu_shift_debug: Option<ValuShiftDebug>,
    #[serde(skip_serializing_if = "Option::is_none")]
    schedule_constraints: Option<Vec<crate::schedule_constraints::ScheduleConstraint>>,
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    shifted_alu_cycles: std::collections::HashMap<usize, Vec<i32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ssa_cycle_assignments: Option<Vec<i32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ssa_ops_bulk_move: Option<Vec<SsaOpBulkMoveOut>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ssa_ops: Option<Vec<SsaOpDebugOut>>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    eliminated_vbroadcast_ops: Vec<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    run_options: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    git_sha: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    comment: Option<String>,
    /// Solver lower bound on cycles for THIS emission (CP-SAT only): the fewest cycles in which
    /// the ops this build emitted, under the encoding choices the model offers, can be scheduled.
    /// When optimal, equals total_cycles; when feasible, solver best_bound in cycles. It is not a
    /// bound on the problem: an emission with fewer ops has a lower bound (see `lb_scope`).
    #[serde(skip_serializing_if = "Option::is_none")]
    lb_cycles: Option<usize>,
    /// What `lb_cycles` is a bound on. Always "emission" (written whenever `lb_cycles` is):
    /// the emitted op set and its encoding choices, not the kernel problem itself.
    #[serde(skip_serializing_if = "Option::is_none")]
    lb_scope: Option<&'static str>,
    /// Vector group key/alias VReg id → lane VReg ids, in lane order. Always written (even
    /// when empty) so checkers can tell "no groups" from "JSON predates this field".
    vector_group_lanes: std::collections::BTreeMap<u32, Vec<u32>>,
}

#[derive(Serialize)]
struct SsaOpBulkMoveOut {
    id: usize,
    dep_ids: Vec<usize>,
}

#[derive(Serialize)]
struct SsaOpDebugOut {
    id: usize,
    engine: String,
    opcode: String,
    dest: Option<String>, // VReg as string
    srcs: Vec<SsaOperandOut>,
    dep_ids: Vec<usize>,
    cycle: i32,
    var_type: Option<String>,
    source_info: Option<SsaSourceInfoOut>,
}

#[derive(Serialize)]
enum SsaOperandOut {
    VReg { id: u32, width: String }, // "scalar" or "vector"
    Imm(i64),
}

#[derive(Serialize)]
struct SsaSourceInfoOut {
    round: Option<usize>,
    phase: Option<String>,
    group: Option<usize>,
    tree_depth: Option<usize>,
    iter: Option<usize>,
}

#[derive(Serialize)]
struct ValuShiftDebug {
    total_shifted: usize,
    spread_count: usize,
    clustered_count: usize,
    samples: Vec<ShiftedValuSample>,
}

#[derive(Serialize)]
struct ShiftedValuSample {
    op_id: usize,
    alu_cycles: Vec<i32>,
    unique_cycles: usize,
    spread: bool,
}

#[derive(Serialize)]
struct OpDebugInfo {
    id: usize,
    cycle: i32,
    engine: String,
    opcode: String,
    slot: Vec<String>,
    dep_ids: Vec<usize>,
    reads: Vec<i64>,
    writes: Vec<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    from_const_choice: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    from_select_choice: Option<usize>,
}

/// Liveness detail level for SSA schedule JSON output.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LivenessDetail {
    None,
    Summary,
    Full,
}

impl LivenessDetail {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "none" => LivenessDetail::None,
            "summary" => LivenessDetail::Summary,
            "full" => LivenessDetail::Full,
            _ => LivenessDetail::Full,
        }
    }
}

/// Write SSA schedule with usage accounting to JSON file.
pub fn write_ssa_schedule_json(
    result: &crate::kernel_builder::SsaScheduleResult,
    output_path: &str,
    liveness_detail: LivenessDetail,
    run_options: Option<serde_json::Value>,
    git_sha: Option<String>,
    comment: Option<String>,
) -> Result<(), String> {
    use crate::types::SlotVal;

    let ops_debug: Vec<OpDebugInfo> = result
        .ops
        .iter()
        .map(|op| {
            let slot: Vec<String> = op
                .slot
                .iter()
                .map(|sv| match sv {
                    SlotVal::Str(s) => s.clone(),
                    SlotVal::Int(i) => i.to_string(),
                })
                .collect();

            OpDebugInfo {
                id: op.id,
                cycle: op.cycle,
                engine: op.engine.as_str().to_string(),
                opcode: slot.first().cloned().unwrap_or_default(),
                slot,
                dep_ids: op.dep_ids.iter().copied().collect(),
                reads: op.reads.iter().copied().collect(),
                writes: op.writes.iter().copied().collect(),
                from_const_choice: op.from_const_choice,
                from_select_choice: op.from_select_choice,
            }
        })
        .collect();

    let vreg_to_addr: std::collections::HashMap<String, i64> = result
        .address_result
        .vreg_to_addr
        .iter()
        .map(|(vreg, addr)| (format!("{:?}", vreg), *addr))
        .collect();

    let (liveness_summary, vregs, liveness_curve) = if liveness_detail == LivenessDetail::None {
        (
            LivenessSummary {
                peak_live_words: result.address_result.peak_live,
                peak_cycle: 0,
                peak_by_var_type: HashMap::new(),
                total_vregs: result.vreg_count as usize,
                dead_vregs: 0,
                avg_interval_length: 0.0,
                max_interval_length: 0,
            },
            None,
            None,
        )
    } else {
        let summary =
            compute_liveness_summary(&result.address_result.live_intervals, &result.vreg_metadata);

        if liveness_detail == LivenessDetail::Full {
            let vreg_infos = build_vreg_debug_info(
                &result.address_result.live_intervals,
                &result.address_result.vreg_to_addr,
                &result.vreg_metadata,
            );

            let max_cycle = result.cycles as i32;
            let curve = compute_liveness_curve(
                &result.address_result.live_intervals,
                max_cycle,
                &result.vreg_metadata,
            );

            (summary, Some(vreg_infos), Some(curve))
        } else {
            (summary, None, None)
        }
    };

    use crate::types::{Operand, VRegWidth};

    // Convert SSA ops to debug format
    let ssa_ops_debug: Option<Vec<SsaOpDebugOut>> = result.ssa_ops.as_ref().map(|ssa_ops| {
        ssa_ops
            .iter()
            .map(|op| {
                let srcs: Vec<SsaOperandOut> = op
                    .srcs
                    .iter()
                    .map(|src| match src {
                        Operand::VReg(vreg) => SsaOperandOut::VReg {
                            id: vreg.id,
                            width: match vreg.width {
                                VRegWidth::Scalar => "scalar".to_string(),
                                VRegWidth::Vector => "vector".to_string(),
                            },
                        },
                        Operand::Imm(val) => SsaOperandOut::Imm(*val),
                    })
                    .collect();

                let source_info = op.source_info.as_ref().map(|info| SsaSourceInfoOut {
                    round: info.round,
                    phase: Some(info.phase.as_str().to_string()),
                    group: info.group,
                    tree_depth: info.tree_depth,
                    iter: info.iter,
                });

                SsaOpDebugOut {
                    id: op.id,
                    engine: op.engine.as_str().to_string(),
                    opcode: op.opcode.as_str().to_string(),
                    dest: op.dest.as_ref().map(|v| {
                        format!(
                            "v{}_{}",
                            v.id,
                            match v.width {
                                VRegWidth::Scalar => "s",
                                VRegWidth::Vector => "v",
                            }
                        )
                    }),
                    srcs,
                    dep_ids: op.dep_ids.iter().copied().collect(),
                    cycle: op.cycle,
                    var_type: op.var_type.as_ref().map(|vt| vt.as_str().to_string()),
                    source_info,
                }
            })
            .collect()
    });

    let valu_shift_debug = if !result.shifted_alu_cycles.is_empty() {
        let mut spread_count = 0;
        let mut clustered_count = 0;
        let mut spread_samples = Vec::new();
        let mut clustered_samples = Vec::new();

        for (op_id, cycles) in &result.shifted_alu_cycles {
            let unique: HashSet<i32> = cycles.iter().copied().collect();
            let is_spread = unique.len() > 1;

            if is_spread {
                spread_count += 1;
                if spread_samples.len() < 10 {
                    spread_samples.push(ShiftedValuSample {
                        op_id: *op_id,
                        alu_cycles: cycles.clone(),
                        unique_cycles: unique.len(),
                        spread: true,
                    });
                }
            } else {
                clustered_count += 1;
                if clustered_samples.len() < 10 {
                    clustered_samples.push(ShiftedValuSample {
                        op_id: *op_id,
                        alu_cycles: cycles.clone(),
                        unique_cycles: 1,
                        spread: false,
                    });
                }
            }
        }

        let mut samples = spread_samples;
        samples.extend(clustered_samples);

        Some(ValuShiftDebug {
            total_shifted: result.shifted_alu_cycles.len(),
            spread_count,
            clustered_count,
            samples,
        })
    } else {
        None
    };

    let output = SsaScheduleOutput {
        total_cycles: result.cycles,
        total_ops: result.ops.len(),
        vreg_count: result.vreg_count,
        peak_live_words: result.address_result.peak_live,
        max_addr: result.address_result.max_addr,
        budget_exceeded: result.budget_exceeded,
        valu_shifted: result.valu_shifted,
        usage_accounting: result.usage_accounting.clone(),
        ops: ops_debug,
        vreg_to_addr,
        liveness_summary,
        vregs,
        liveness_curve,
        valu_shift_debug,
        schedule_constraints: result.schedule_constraints.clone(),
        shifted_alu_cycles: result.shifted_alu_cycles.clone(),
        ssa_cycle_assignments: result.ssa_cycle_assignments.clone(),
        ssa_ops_bulk_move: result.ssa_ops_bulk_move.as_ref().map(|v| {
            v.iter()
                .map(|(id, dep_ids)| SsaOpBulkMoveOut {
                    id: *id,
                    dep_ids: dep_ids.clone(),
                })
                .collect()
        }),
        ssa_ops: ssa_ops_debug,
        eliminated_vbroadcast_ops: result.eliminated_vbroadcast_ops.clone(),
        run_options,
        git_sha,
        comment,
        lb_cycles: result.lb_cycles,
        lb_scope: result.lb_cycles.map(|_| "emission"),
        vector_group_lanes: result
            .vector_group_lanes
            .iter()
            .map(|(group, lanes)| (group.id, lanes.iter().map(|lane| lane.id).collect()))
            .collect(),
    };

    let json = serde_json::to_string_pretty(&output)
        .map_err(|e| format!("JSON serialization failed: {}", e))?;

    std::fs::write(output_path, &json)
        .map_err(|e| format!("Failed to write {}: {}", output_path, e))?;

    Ok(())
}
