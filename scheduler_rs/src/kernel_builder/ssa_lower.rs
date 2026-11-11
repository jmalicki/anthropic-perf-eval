//! SSA lowering helpers.
//!
//! This module handles lowering SSA ops to concrete ops, including
//! VALU→ALU shift expansion.

use std::collections::{HashMap, HashSet};

use arrayvec::ArrayVec;

/// Prefix for op ids of CSE shared SUB ops (one per active CSE group).
/// Ops use ids CSE_SHARED_SUB_PREFIX + g for group index g.
const CSE_SHARED_SUB_PREFIX: usize = 0xC5E0_0000;

/// Prefix for trace debug op ids (trace_input/trace_output).
const TRACE_DEBUG_OP_PREFIX: usize = 0xD3B0_0000;

use crate::address_assign::lower_ssa_op;
use crate::types::{ConstChoiceEncoding, Engine, Op, SelectChoiceEncoding, SlotVal, SsaId, SsaOp, VReg, VLEN};

/// Check if an SSA VALU op can be shifted to ALU.
///
/// Shiftable ops are binary arithmetic operations that can be expanded
/// to 8 scalar ALU ops (one per vector lane).
pub fn can_ssa_valu_shift_to_alu(op: &SsaOp) -> bool {
    if op.engine != Engine::Valu {
        return false;
    }

    // Check if the opcode supports shifting
    op.opcode.can_shift_to_alu()
}

/// Lower SSA ops to concrete ops (no VALU shifting).
#[allow(dead_code)]
pub fn lower_ssa_ops(ssa_ops: &[SsaOp], vreg_to_addr: &HashMap<VReg, i64>) -> Result<Vec<Op>, String> {
    ssa_ops
        .iter()
        .map(|op| lower_ssa_op(op, vreg_to_addr))
        .collect()
}

/// Context for SSA lowering with VALU shift, ConstChoice, SelectChoice, and optional CSE.
///
/// Groups all inputs to `lower_ssa_ops_with_shift` in one struct to reduce parameter count.
#[derive(Debug)]
pub struct LoweringContext<'a> {
    pub ssa_ops: &'a [SsaOp],
    pub vreg_to_addr: &'a HashMap<VReg, i64>,
    pub shifted_ops: &'a HashSet<usize>,
    pub const_choice_encoding: &'a HashMap<usize, ConstChoiceEncoding>,
    pub select_choice_encoding: &'a HashMap<usize, SelectChoiceEncoding>,
    /// op_id -> [cycle0, ..., cycle7]
    pub shifted_alu_cycles: &'a HashMap<usize, Vec<i32>>,
    pub eliminated_vbroadcast_ops: &'a HashSet<usize>,
    /// vb op_id -> scalar producer op_id
    pub eliminated_vb_to_scalar_producer: &'a HashMap<usize, usize>,
    pub cse_groups: Option<&'a [crate::cpsat_scheduler::CseGroup]>,
    pub cse_group_results: Option<&'a [crate::cpsat_scheduler::CseGroupResult]>,
}

/// Lower SSA ops to concrete ops, expanding shifted VALU ops to ALU.
///
/// Shifted VALU ops are expanded to 8 scalar ALU ops, one per lane.
/// Each ALU op reads from src[i] and writes to dest[i].
///
/// ConstChoice ops (single op from emit_const_choice) are lowered based on the scheduler's decision:
/// - `ConstChoiceEncoding::AddImm`: emit AddImm op (Flow engine)
/// - `ConstChoiceEncoding::Load`: emit Const op (Load engine)
/// - `ConstChoiceEncoding::AluDerive(info)`: emit ALU derive op
///
/// The SSA kernel path uses only emit_const_choice(); const_choice_encoding comes from CP-SAT
/// or greedy. The legacy emit_const() dual-op path is used only in tests.
///
/// SelectChoice ops are lowered based on the scheduler's decision:
/// - `SelectChoiceEncoding::Fma`: emit sub + fma ops (both VALU, sequential)
/// - `SelectChoiceEncoding::Vselect` or missing: emit vselect op (Flow engine)
///
/// When `cse_groups` and `cse_group_results` are provided and non-empty, SelectChoice FMA
/// consumers in CSE groups use a shared SUB: one SUB per active group, multiple FMAs per group.

/// Verifies that every VReg that lowering will look up in `vreg_to_addr` is present.
/// Returns `Err(msg)` on first missing VReg so the pipeline can report a clear error.
pub fn verify_vreg_mapping_complete(ctx: &LoweringContext<'_>) -> Result<(), String> {
    use crate::types::{Operand, Opcode};
    for (idx, op) in ctx.ssa_ops.iter().enumerate() {
        if let Some(dest) = op.dest {
            if !ctx.vreg_to_addr.contains_key(&dest) {
                return Err(format!(
                    "lowering op[{}] id={}: dest VReg {:?} not in vreg_to_addr",
                    idx, op.id, dest
                ));
            }
        }
        for src in &op.srcs {
            if let Some(vreg) = src.as_vreg() {
                if !ctx.vreg_to_addr.contains_key(&vreg) {
                    return Err(format!(
                        "lowering op[{}] id={}: src VReg {:?} not in vreg_to_addr",
                        idx, op.id, vreg
                    ));
                }
            }
        }
        if op.opcode == Opcode::ConstChoice {
            if let Some(ConstChoiceEncoding::AluDerive(info)) = ctx.const_choice_encoding.get(&op.id) {
                let left_dest = ctx
                    .ssa_ops
                    .iter()
                    .find(|o| o.id == info.left_op_id)
                    .and_then(|o| o.dest)
                    .ok_or_else(|| format!("ConstChoice AluDerive op[{}]: left producer op id={} not found or no dest", idx, info.left_op_id))?;
                let right_dest = ctx
                    .ssa_ops
                    .iter()
                    .find(|o| o.id == info.right_op_id)
                    .and_then(|o| o.dest)
                    .ok_or_else(|| format!("ConstChoice AluDerive op[{}]: right producer op id={} not found or no dest", idx, info.right_op_id))?;
                if !ctx.vreg_to_addr.contains_key(&left_dest) {
                    return Err(format!(
                        "lowering op[{}] id={}: AluDerive left producer dest {:?} not in vreg_to_addr",
                        idx, op.id, left_dest
                    ));
                }
                if !ctx.vreg_to_addr.contains_key(&right_dest) {
                    return Err(format!(
                        "lowering op[{}] id={}: AluDerive right producer dest {:?} not in vreg_to_addr",
                        idx, op.id, right_dest
                    ));
                }
            }
        }
        if op.opcode == Opcode::SelectChoice {
            if let Some(delta_vreg) = op.delta_vreg {
                if !ctx.vreg_to_addr.contains_key(&delta_vreg) {
                    return Err(format!(
                        "lowering op[{}] id={}: SelectChoice delta_vreg {:?} not in vreg_to_addr",
                        idx, op.id, delta_vreg
                    ));
                }
            }
            if matches!(ctx.select_choice_encoding.get(&op.id), Some(SelectChoiceEncoding::Fma))
                && op.delta_vreg.is_none()
            {
                let uses_shared_sub = match (ctx.cse_groups, ctx.cse_group_results) {
                    (Some(groups), Some(results)) if groups.len() == results.len() => groups
                        .iter()
                        .zip(results.iter())
                        .any(|(g, r)| r.sub_active && g.consumer_indices.contains(&idx)),
                    _ => false,
                };
                if !uses_shared_sub {
                    if let Some(sub_temp_vreg) = op.sub_temp_vreg {
                        if !ctx.vreg_to_addr.contains_key(&sub_temp_vreg) {
                            return Err(format!(
                                "lowering op[{}] id={}: SelectChoice FMA sub_temp_vreg {:?} not in vreg_to_addr",
                                idx, op.id, sub_temp_vreg
                            ));
                        }
                    }
                }
            }
        }
    }
    if let (Some(groups), Some(results)) = (ctx.cse_groups, ctx.cse_group_results) {
        if groups.len() == results.len() {
            for (group, res) in groups.iter().zip(results.iter()) {
                if res.sub_active && !ctx.vreg_to_addr.contains_key(&group.sub_temp_vreg) {
                    return Err(format!(
                        "CSE group sub_temp_vreg {:?} (sub_active) not in vreg_to_addr",
                        group.sub_temp_vreg
                    ));
                }
            }
        }
    }
    Ok(())
}

/// When `cse_groups` and `cse_group_results` are provided and non-empty, SelectChoice FMA
/// consumers in CSE groups use a shared SUB: one SUB per active group, multiple FMAs per group.
///
/// Before lowering, verifies that every VReg that will be looked up in `vreg_to_addr` is present
/// (see `verify_vreg_mapping_complete`). Returns `Err` on first missing VReg.
pub fn lower_ssa_ops_with_shift(ctx: &LoweringContext<'_>) -> Result<Vec<Op>, String> {
    use crate::types::{Operand, Opcode, SlotVal};
    use arrayvec::ArrayVec;

    verify_vreg_mapping_complete(ctx)?;

    let mut result = Vec::new();

    // Build op_id -> shared sub_temp_addr for CSE group FMA consumers
    let cse_op_to_sub_temp_addr: HashMap<usize, i64> = {
        let mut m = HashMap::new();
        if let (Some(groups), Some(results)) = (ctx.cse_groups, ctx.cse_group_results) {
            if groups.len() == results.len() {
                for (group, res) in groups.iter().zip(results.iter()) {
                    if !res.sub_active {
                        continue;
                    }
                    let Some(&sub_temp_addr) = ctx.vreg_to_addr.get(&group.sub_temp_vreg) else {
                        continue;
                    };
                    for &idx in &group.consumer_indices {
                        if idx < ctx.ssa_ops.len() && matches!(ctx.select_choice_encoding.get(&ctx.ssa_ops[idx].id), Some(SelectChoiceEncoding::Fma)) {
                            m.insert(ctx.ssa_ops[idx].id, sub_temp_addr);
                        }
                    }
                }
            } else {
                eprintln!(
                    "[ssa_lower] CSE groups/results length mismatch: cse_groups.len()={} cse_group_results.len()={}; skipping CSE mapping in lower_ssa_ops_with_shift (caller bug?)",
                    groups.len(),
                    results.len()
                );
            }
        }
        m
    };

    // Emit shared SUBs first (one per active CSE group)
    if let (Some(groups), Some(results)) = (ctx.cse_groups, ctx.cse_group_results) {
        if groups.len() == results.len() {
            for (g, (group, res)) in groups.iter().zip(results.iter()).enumerate() {
                if !res.sub_active {
                    continue;
                }
                let sub_temp_addr = *ctx.vreg_to_addr
                    .get(&group.sub_temp_vreg)
                    .expect("CSE group sub_temp_vreg VReg not in mapping");
                let true_val_addr = *ctx.vreg_to_addr
                    .get(&group.true_val)
                    .expect("CSE group true_val VReg not in mapping");
                let false_val_addr = *ctx.vreg_to_addr
                    .get(&group.false_val)
                    .expect("CSE group false_val VReg not in mapping");

                let mut sub_slot: ArrayVec<SlotVal, 6> = ArrayVec::new();
                sub_slot.push(SlotVal::Str("-".to_string()));
                sub_slot.push(SlotVal::Int(sub_temp_addr));
                sub_slot.push(SlotVal::Int(true_val_addr));
                sub_slot.push(SlotVal::Int(false_val_addr));

                let mut sub_reads = HashSet::new();
                for i in 0..VLEN as i64 {
                    sub_reads.insert(true_val_addr + i);
                    sub_reads.insert(false_val_addr + i);
                }
                let mut sub_writes = HashSet::new();
                for i in 0..VLEN as i64 {
                    sub_writes.insert(sub_temp_addr + i);
                }

                result.push(Op {
                    id: CSE_SHARED_SUB_PREFIX + g,
                    engine: crate::types::Engine::Valu,
                    slot: sub_slot,
                    cycle: res.sub_cycle,
                    dep_ids: HashSet::new(),
                    reads: sub_reads,
                    writes: sub_writes,
                    last_use_reads: HashSet::new(),
                    produces_constant: None,
                    trace_id: None,
                    from_const_choice: None,
                    from_select_choice: None, // CSE shared sub is not from a single SelectChoice
                });
            }
        }
    }

    // Build map: eliminated vb dest VReg -> scalar producer address (for shifted consumers reading from vb)
    let eliminated_vb_vreg_to_scalar_addr: HashMap<VReg, i64> = {
        let mut m = HashMap::new();
        for (&vb_id, &scalar_id) in ctx.eliminated_vb_to_scalar_producer {
            if !ctx.eliminated_vbroadcast_ops.contains(&vb_id) {
                continue;
            }
            let vb_dest = ctx.ssa_ops
                .iter()
                .find(|o| o.id == vb_id)
                .and_then(|o| o.dest);
            let scalar_dest = ctx.ssa_ops
                .iter()
                .find(|o| o.id == scalar_id)
                .and_then(|o| o.dest);
            if let (Some(vreg), Some(svreg)) = (vb_dest, scalar_dest) {
                if let Some(&addr) = ctx.vreg_to_addr.get(&svreg) {
                    m.insert(vreg, addr);
                }
            }
        }
        m
    };

    for ssa_op in ctx.ssa_ops {
        if ctx.eliminated_vbroadcast_ops.contains(&ssa_op.id) {
            continue;
        }
        if ctx.shifted_ops.contains(&ssa_op.id) {
            // Expand shifted VALU op to 8 ALU ops
            // Use individual cycles if available (from CP-SAT), otherwise use completion_cycle
            let alu_cycles = ctx.shifted_alu_cycles.get(&ssa_op.id);
            let expanded = expand_ssa_valu_to_alu(
                ssa_op,
                ctx.vreg_to_addr,
                alu_cycles.map(|v| v.as_slice()),
                if eliminated_vb_vreg_to_scalar_addr.is_empty() {
                    None
                } else {
                    Some(&eliminated_vb_vreg_to_scalar_addr)
                },
            );
            result.extend(expanded);
        } else if ssa_op.opcode == Opcode::ConstChoice {
            // Handle ConstChoice based on scheduler decision
            let dest = ssa_op.dest.expect("ConstChoice op must have dest");
            let dest_addr = *ctx.vreg_to_addr.get(&dest).expect("dest VReg not in mapping");
            let value = ssa_op
                .produces_constant
                .expect("ConstChoice op must have produces_constant");

            match ctx.const_choice_encoding.get(&ssa_op.id) {
            Some(ConstChoiceEncoding::AluDerive(info)) => {
                let left_op_id = info.left_op_id;
                let right_op_id = info.right_op_id;
                let opcode = info.opcode;
                // ALU-derived: dest = left opcode right (e.g. add, mul)
                let left_dest = ctx.ssa_ops
                    .iter()
                    .find(|o| o.id == left_op_id)
                    .and_then(|o| o.dest)
                    .expect("ALU derive left producer must exist");
                let right_dest = ctx.ssa_ops
                    .iter()
                    .find(|o| o.id == right_op_id)
                    .and_then(|o| o.dest)
                    .expect("ALU derive right producer must exist");
                let left_addr = *ctx.vreg_to_addr.get(&left_dest).expect("left producer VReg in mapping");
                let right_addr =
                    *ctx.vreg_to_addr.get(&right_dest).expect("right producer VReg in mapping");

                let mut slot: ArrayVec<SlotVal, 6> = ArrayVec::new();
                slot.push(SlotVal::Str(opcode.as_str().to_string()));
                slot.push(SlotVal::Int(dest_addr));
                slot.push(SlotVal::Int(left_addr));
                slot.push(SlotVal::Int(right_addr));

                let mut reads = HashSet::new();
                reads.insert(left_addr);
                reads.insert(right_addr);
                let mut writes = HashSet::new();
                writes.insert(dest_addr);

                result.push(Op {
                    id: ssa_op.id,
                    engine: crate::types::Engine::Alu,
                    slot,
                    cycle: ssa_op.cycle,
                    dep_ids: ssa_op.dep_ids.iter().copied().collect(),
                    reads,
                    writes,
                    last_use_reads: HashSet::new(),
                    produces_constant: Some(value),
                    trace_id: ssa_op.trace_id.clone(),
                    from_const_choice: Some(ssa_op.id),
                    from_select_choice: None,
                });
            }
            Some(ConstChoiceEncoding::AddImm) => {
                // AddImm: dest = scratch[0] + value
                // IMPORTANT: Scratch memory is ALWAYS initialized to 0 at startup, so scratch[0] == 0
                // is guaranteed by hardware initialization. add_imm reads from address 0, but we don't
                // need a precedence constraint because address 0 is always initialized to 0.
                let mut slot: ArrayVec<SlotVal, 6> = ArrayVec::new();
                slot.push(SlotVal::Str("add_imm".to_string()));
                slot.push(SlotVal::Int(dest_addr));
                slot.push(SlotVal::Int(0)); // Address 0 always contains 0 (hardware initialization)
                slot.push(SlotVal::Int(value));

                let mut reads = HashSet::new();
                reads.insert(0); // Read from zero VReg

                let mut writes = HashSet::new();
                writes.insert(dest_addr);

                result.push(Op {
                    id: ssa_op.id,
                    engine: crate::types::Engine::Flow,
                    slot,
                    cycle: ssa_op.cycle,
                    dep_ids: ssa_op.dep_ids.iter().copied().collect(),
                    reads,
                    writes,
                    last_use_reads: HashSet::new(),
                    produces_constant: Some(value),
                    trace_id: ssa_op.trace_id.clone(),
                    from_const_choice: Some(ssa_op.id),
                    from_select_choice: None,
                });
            }
            Some(ConstChoiceEncoding::Load) | None => {
                // Load: dest = const value
                let mut slot: ArrayVec<SlotVal, 6> = ArrayVec::new();
                slot.push(SlotVal::Str("const".to_string()));
                slot.push(SlotVal::Int(dest_addr));
                slot.push(SlotVal::Int(value));

                let mut writes = HashSet::new();
                writes.insert(dest_addr);

                result.push(Op {
                    id: ssa_op.id,
                    engine: crate::types::Engine::Load,
                    slot,
                    cycle: ssa_op.cycle,
                    dep_ids: ssa_op.dep_ids.iter().copied().collect(),
                    reads: HashSet::new(),
                    writes,
                    last_use_reads: HashSet::new(),
                    produces_constant: Some(value),
                    trace_id: ssa_op.trace_id.clone(),
                    from_const_choice: Some(ssa_op.id),
                    from_select_choice: None,
                });
            }
            }
        } else if ssa_op.opcode == Opcode::SelectChoice {
            // Handle SelectChoice based on scheduler decision
            let dest = ssa_op.dest.expect("SelectChoice op must have dest");
            let dest_addr = *ctx.vreg_to_addr.get(&dest).expect("dest VReg not in mapping");
            if std::env::var("SCHEDULER_DEBUG").is_ok() {
                eprintln!(
                    "[DEBUG lower_ssa_ops_with_shift] SelectChoice op.id={} cycle={} select_choice=Fma={} sub_temp_vreg={:?}",
                    ssa_op.id, ssa_op.cycle, matches!(ctx.select_choice_encoding.get(&ssa_op.id), Some(SelectChoiceEncoding::Fma)), ssa_op.sub_temp_vreg
                );
            }

            // Extract operands: cond, true_val, false_val
            if ssa_op.srcs.len() < 3 {
                panic!("SelectChoice op must have 3 operands: cond, true_val, false_val");
            }

            use crate::types::Operand;
            let cond_vreg = ssa_op.srcs[0]
                .as_vreg()
                .expect("SelectChoice cond must be VReg");
            let true_val_vreg = ssa_op.srcs[1]
                .as_vreg()
                .expect("SelectChoice true_val must be VReg");
            let false_val_vreg = ssa_op.srcs[2]
                .as_vreg()
                .expect("SelectChoice false_val must be VReg");

            let cond_addr = *ctx.vreg_to_addr
                .get(&cond_vreg)
                .expect("cond VReg not in mapping");
            let true_val_addr = *ctx.vreg_to_addr
                .get(&true_val_vreg)
                .expect("true_val VReg not in mapping");
            let false_val_addr = *ctx.vreg_to_addr
                .get(&false_val_vreg)
                .expect("false_val VReg not in mapping");

            if matches!(ctx.select_choice_encoding.get(&ssa_op.id), Some(SelectChoiceEncoding::Fma)) {
                // FMA path: check if we have a precomputed delta
                if std::env::var("SCHEDULER_DEBUG").is_ok() {
                    eprintln!(
                        "[DEBUG lower_ssa_ops_with_shift] SelectChoice FMA path: op.id={} cycle={} delta_vreg={:?} sub_temp_vreg={:?}",
                        ssa_op.id, ssa_op.cycle, ssa_op.delta_vreg, ssa_op.sub_temp_vreg
                    );
                }
                if let Some(delta_vreg) = ssa_op.delta_vreg {
                    // Precomputed delta available: emit single FMA op
                    // fma = cond * delta + base (where base = false_val)
                    let delta_addr = *ctx.vreg_to_addr
                        .get(&delta_vreg)
                        .expect("delta VReg not in mapping");
                    if std::env::var("SCHEDULER_DEBUG").is_ok() {
                        eprintln!(
                            "[DEBUG lower_ssa_ops_with_shift] SelectChoice FMA delta path: op.id={} delta_vreg={:?} -> delta_addr={}, dest={:?} -> {}, cond={:?} -> {}, false_val={:?} -> {}",
                            ssa_op.id, delta_vreg, delta_addr,
                            ssa_op.dest.unwrap(), dest_addr,
                            cond_vreg, cond_addr,
                            false_val_vreg, false_val_addr
                        );
                    }

                    let mut fma_slot: ArrayVec<SlotVal, 6> = ArrayVec::new();
                    fma_slot.push(SlotVal::Str("multiply_add".to_string()));
                    fma_slot.push(SlotVal::Int(dest_addr));
                    fma_slot.push(SlotVal::Int(cond_addr));
                    fma_slot.push(SlotVal::Int(delta_addr));
                    fma_slot.push(SlotVal::Int(false_val_addr));

                    let mut fma_reads = HashSet::new();
                    // Expand vector addresses: all SelectChoice operands are vectors
                    for i in 0..VLEN as i64 {
                        fma_reads.insert(cond_addr + i);
                        fma_reads.insert(delta_addr + i);
                        fma_reads.insert(false_val_addr + i);
                    }

                    let mut fma_writes = HashSet::new();
                    // dest is a vector VReg
                    for i in 0..VLEN as i64 {
                        fma_writes.insert(dest_addr + i);
                    }

                    result.push(Op {
                        id: ssa_op.id,
                        engine: crate::types::Engine::Valu,
                        slot: fma_slot,
                        cycle: ssa_op.cycle,
                        dep_ids: ssa_op.dep_ids.iter().copied().collect(),
                        reads: fma_reads,
                        writes: fma_writes,
                        last_use_reads: HashSet::new(),
                        trace_id: ssa_op.trace_id.clone(),
                        produces_constant: None,
                        from_const_choice: None,
                        from_select_choice: Some(ssa_op.id),
                    });
                } else {
                    // No precomputed delta: emit sub + fma ops (both VALU, sequential)
                    // sub = true_val - false_val
                    // fma = cond * sub + false_val
                    // When in CSE group: shared SUB already emitted; use shared sub_temp_addr, emit only FMA
                    let use_shared_sub = cse_op_to_sub_temp_addr.contains_key(&ssa_op.id);
                    let sub_temp_addr = if use_shared_sub {
                        *cse_op_to_sub_temp_addr.get(&ssa_op.id).unwrap()
                    } else {
                        let sub_temp_vreg = ssa_op.sub_temp_vreg.expect(
                            "SelectChoice without delta must have sub_temp_vreg allocated during emission"
                        );
                        *ctx.vreg_to_addr.get(&sub_temp_vreg).expect(
                            "sub_temp_vreg not in address mapping - was it included in address assignment?"
                        )
                    };
                    if std::env::var("SCHEDULER_DEBUG").is_ok() {
                        let sub_src = if use_shared_sub {
                            format!("shared->{}", sub_temp_addr)
                        } else {
                            format!("vreg{:?}->{}", ssa_op.sub_temp_vreg, sub_temp_addr)
                        };
                        eprintln!(
                            "[DEBUG lower_ssa_ops_with_shift] SelectChoice FMA (op.id={} cycle={}): sub_temp {} dest={:?}->{} cond={:?}->{} true_val={:?}->{} false_val={:?}->{}",
                            ssa_op.id, ssa_op.cycle, sub_src,
                            ssa_op.dest.unwrap(), dest_addr,
                            cond_vreg, cond_addr,
                            true_val_vreg, true_val_addr,
                            false_val_vreg, false_val_addr
                        );
                    }

                    // Sub op: only when NOT using shared SUB (CSE group)
                    if !use_shared_sub {
                        let mut sub_slot: ArrayVec<SlotVal, 6> = ArrayVec::new();
                        sub_slot.push(SlotVal::Str("-".to_string()));
                        sub_slot.push(SlotVal::Int(sub_temp_addr));
                        sub_slot.push(SlotVal::Int(true_val_addr));
                        sub_slot.push(SlotVal::Int(false_val_addr));

                        let mut sub_reads = HashSet::new();
                        for i in 0..VLEN as i64 {
                            sub_reads.insert(true_val_addr + i);
                            sub_reads.insert(false_val_addr + i);
                        }
                        let mut sub_writes = HashSet::new();
                        for i in 0..VLEN as i64 {
                            sub_writes.insert(sub_temp_addr + i);
                        }

                        result.push(Op {
                            id: ssa_op.id,
                            engine: crate::types::Engine::Valu,
                            slot: sub_slot,
                            cycle: ssa_op.cycle,
                            dep_ids: ssa_op.dep_ids.iter().copied().collect(),
                            reads: sub_reads,
                            writes: sub_writes,
                            last_use_reads: HashSet::new(),
                            produces_constant: None,
                            trace_id: ssa_op.trace_id.clone(),
                            from_const_choice: None,
                            from_select_choice: Some(ssa_op.id),
                        });
                    }

                    // FMA op: dest = cond * sub_temp + false_val
                    let mut fma_slot: ArrayVec<SlotVal, 6> = ArrayVec::new();
                    fma_slot.push(SlotVal::Str("multiply_add".to_string()));
                    fma_slot.push(SlotVal::Int(dest_addr));
                    fma_slot.push(SlotVal::Int(cond_addr));
                    fma_slot.push(SlotVal::Int(sub_temp_addr));
                    fma_slot.push(SlotVal::Int(false_val_addr));

                    let mut fma_reads = HashSet::new();
                    // Expand vector addresses: all SelectChoice operands are vectors
                    for i in 0..VLEN as i64 {
                        fma_reads.insert(cond_addr + i);
                        fma_reads.insert(sub_temp_addr + i);
                        fma_reads.insert(false_val_addr + i);
                    }

                    let mut fma_writes = HashSet::new();
                    // dest is a vector VReg
                    for i in 0..VLEN as i64 {
                        fma_writes.insert(dest_addr + i);
                    }

                    let fma_op = Op {
                        id: ssa_op.id + 1000000, // Different ID for fma op
                        engine: crate::types::Engine::Valu,
                        slot: fma_slot,
                        cycle: ssa_op.cycle + 1, // FMA is at cycle + 1 (adjacent to sub)
                        dep_ids: {
                            let mut deps: HashSet<usize> = ssa_op.dep_ids.iter().copied().collect();
                            deps.insert(ssa_op.id); // FMA depends on sub
                            deps
                        },
                        reads: fma_reads,
                        writes: fma_writes,
                        last_use_reads: HashSet::new(),
                        produces_constant: None,
                        trace_id: ssa_op.trace_id.clone(),
                        from_const_choice: None,
                        from_select_choice: Some(ssa_op.id),
                    };
                    if std::env::var("SCHEDULER_DEBUG").is_ok() {
                        eprintln!(
                            "[DEBUG lower_ssa_ops_with_shift] SelectChoice FMA fma: op.id={} cycle={} writes={:?} reads={:?}",
                            fma_op.id, fma_op.cycle, fma_op.writes, fma_op.reads
                        );
                    }
                    result.push(fma_op);
                }
            } else {
                // vselect path: emit vselect op (Flow)
                // If delta_vreg is set, true_val was already computed during emission
                // (via emit_select_choice_with_delta), so we can use it directly
                let mut slot: ArrayVec<SlotVal, 6> = ArrayVec::new();
                slot.push(SlotVal::Str("vselect".to_string()));
                slot.push(SlotVal::Int(dest_addr));
                slot.push(SlotVal::Int(cond_addr));
                slot.push(SlotVal::Int(true_val_addr));
                slot.push(SlotVal::Int(false_val_addr));

                let mut reads = HashSet::new();
                reads.insert(cond_addr);
                reads.insert(true_val_addr);
                reads.insert(false_val_addr);

                let mut writes = HashSet::new();
                writes.insert(dest_addr);

                result.push(Op {
                    id: ssa_op.id,
                    engine: crate::types::Engine::Flow,
                    slot,
                    cycle: ssa_op.cycle,
                    dep_ids: ssa_op.dep_ids.iter().copied().collect(),
                    reads,
                    writes,
                    last_use_reads: HashSet::new(),
                    produces_constant: None,
                    trace_id: ssa_op.trace_id.clone(),
                    from_const_choice: None,
                    from_select_choice: Some(ssa_op.id),
                });
            }
        } else {
            // Normal lowering
            result.push(lower_ssa_op(ssa_op, ctx.vreg_to_addr)?);
        }
    }

    // Post-pass: emit trace_input/trace_output debug ops for ops with trace_id
    emit_trace_debug_ops(&mut result, ctx);

    Ok(result)
}

/// Emit trace_input and trace_output debug ops for ops that have trace_id.
/// Groups ops by trace_id, computes first/last cycle, and creates debug ops.
fn emit_trace_debug_ops(result: &mut Vec<Op>, ctx: &LoweringContext<'_>) {
    use crate::types::Operand;
    use std::collections::HashMap;

    // Build trace_id -> (first_cycle, last_cycle, ssa_op)
    let mut trace_info: HashMap<String, (i32, i32, &SsaOp)> = HashMap::new();
    for op in result.iter() {
        if let Some(ref tid) = op.trace_id {
            match trace_info.get_mut(tid) {
                Some((first, last, _)) => {
                    *first = (*first).min(op.cycle);
                    *last = (*last).max(op.cycle);
                }
                None => {
                    let ssa_op = ctx
                        .ssa_ops
                        .iter()
                        .find(|s| s.trace_id.as_deref() == Some(tid.as_str()))
                        .expect("trace_id on op but not found in ssa_ops");
                    trace_info.insert(tid.clone(), (op.cycle, op.cycle, ssa_op));
                }
            }
        }
    }

    let mut debug_op_id = TRACE_DEBUG_OP_PREFIX;
    for (trace_id, (first_cycle, last_cycle, ssa_op)) in trace_info {
        let input_addrs: Vec<i64> = ssa_op
            .srcs
            .iter()
            .filter_map(|o| o.as_vreg())
            .filter_map(|v| ctx.vreg_to_addr.get(&v).copied())
            .collect();
        let dest_addr = ssa_op
            .dest
            .and_then(|d| ctx.vreg_to_addr.get(&d).copied());
        let n_elems = ssa_op
            .dest
            .map(|d| if d.is_vector() { VLEN as i64 } else { 1 })
            .unwrap_or(1);

        // trace_input at first_cycle
        let mut trace_input_slot: ArrayVec<SlotVal, 6> = ArrayVec::new();
        trace_input_slot.push(SlotVal::Str("trace_input".to_string()));
        trace_input_slot.push(SlotVal::Str(trace_id.clone()));
        trace_input_slot.push(SlotVal::Int(n_elems));
        for &a in input_addrs.iter().take(3) {
            trace_input_slot.push(SlotVal::Int(a));
        }
        result.push(Op {
            id: debug_op_id,
            engine: Engine::Debug,
            slot: trace_input_slot,
            cycle: first_cycle,
            dep_ids: HashSet::new(),
            reads: HashSet::new(),
            writes: HashSet::new(),
            last_use_reads: HashSet::new(),
            produces_constant: None,
            trace_id: Some(trace_id.clone()),
            from_const_choice: None,
            from_select_choice: None,
        });
        debug_op_id += 1;

        // trace_output at last_cycle (only if we have a dest)
        if let Some(dest) = dest_addr {
            let mut trace_output_slot: ArrayVec<SlotVal, 6> = ArrayVec::new();
            trace_output_slot.push(SlotVal::Str("trace_output".to_string()));
            trace_output_slot.push(SlotVal::Str(trace_id));
            trace_output_slot.push(SlotVal::Int(n_elems));
            trace_output_slot.push(SlotVal::Int(dest));
            result.push(Op {
                id: debug_op_id,
                engine: Engine::Debug,
                slot: trace_output_slot,
                cycle: last_cycle,
                dep_ids: HashSet::new(),
                reads: HashSet::new(),
                writes: HashSet::new(),
                last_use_reads: HashSet::new(),
                produces_constant: None,
                trace_id: None, // Don't nest
                from_const_choice: None,
                from_select_choice: None,
            });
            debug_op_id += 1;
        }
    }
}

/// Expand a shifted SSA VALU op into 8 scalar ALU ops.
/// When `eliminated_vb_vreg_to_scalar_addr` is set, a source VReg that was produced by an
/// eliminated vbroadcast uses the given scalar address for all 8 lanes (broadcast semantics).
pub fn expand_ssa_valu_to_alu(
    ssa_op: &SsaOp,
    vreg_to_addr: &HashMap<VReg, i64>,
    alu_cycles: Option<&[i32]>, // Individual cycles for 8 ALU ops (if available from CP-SAT)
    eliminated_vb_vreg_to_scalar_addr: Option<&HashMap<VReg, i64>>,
) -> Vec<Op> {
    use crate::types::Operand;

    let dest = ssa_op.dest.expect("VALU op must have dest");
    let dest_addr = *vreg_to_addr.get(&dest).expect("dest VReg not in mapping");

    let resolve_addr = |vreg: &VReg| -> i64 {
        eliminated_vb_vreg_to_scalar_addr
            .and_then(|m| m.get(vreg).copied())
            .unwrap_or_else(|| *vreg_to_addr.get(vreg).expect("VReg not in mapping"))
    };
    let is_eliminated_vb_src = |vreg: &VReg| -> bool {
        eliminated_vb_vreg_to_scalar_addr.map_or(false, |m| m.contains_key(vreg))
    };

    // Get base source addresses and whether each is scalar (eliminated vb -> same addr for all lanes)
    let (src1_base, src2_base, src1_scalar, src2_scalar) = match (&ssa_op.srcs[..], ssa_op.opcode) {
        ([Operand::VReg(s1), Operand::VReg(s2)], _) => {
            let a1 = resolve_addr(s1);
            let a2 = resolve_addr(s2);
            (
                a1,
                a2,
                is_eliminated_vb_src(s1),
                is_eliminated_vb_src(s2),
            )
        }
        _ => return vec![], // Can't expand non-binary ops
    };

    // Determine cycles for each ALU op
    // If individual cycles are provided (from CP-SAT), use them; otherwise use ssa_op.cycle for all
    let use_individual_cycles = alu_cycles.is_some() && alu_cycles.unwrap().len() == VLEN;

    // Create 8 ALU ops
    let mut ops = Vec::with_capacity(VLEN);
    for i in 0..VLEN {
        let i_offset = i as i64;
        let src1_addr_i = if src1_scalar {
            src1_base
        } else {
            src1_base + i_offset
        };
        let src2_addr_i = if src2_scalar {
            src2_base
        } else {
            src2_base + i_offset
        };

        let mut slot: ArrayVec<SlotVal, 6> = ArrayVec::new();
        slot.push(SlotVal::Str(ssa_op.opcode.as_str().to_string()));
        slot.push(SlotVal::Int(dest_addr + i_offset));
        slot.push(SlotVal::Int(src1_addr_i));
        slot.push(SlotVal::Int(src2_addr_i));

        // Build reads/writes sets
        let mut reads = HashSet::new();
        reads.insert(src1_addr_i);
        reads.insert(src2_addr_i);

        let mut writes = HashSet::new();
        writes.insert(dest_addr + i_offset);

        // Use individual cycle if available, otherwise use ssa_op.cycle (completion_cycle)
        let cycle = if use_individual_cycles {
            alu_cycles.unwrap()[i]
        } else {
            ssa_op.cycle
        };

        ops.push(Op {
            id: SsaId::encode_with_sub_index(ssa_op.id, i).as_usize(), // Unique ID for expanded op
            engine: Engine::Alu,
            slot,
            cycle,
            dep_ids: ssa_op.dep_ids.iter().copied().collect(),
            reads,
            writes,
            last_use_reads: HashSet::new(),
            produces_constant: None,
            trace_id: ssa_op.trace_id.clone(),
            from_const_choice: None,
            from_select_choice: None, // Shifted VALU ops are not from SelectChoice
        });
    }

    ops
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpsat_scheduler::{CseGroup, CseGroupResult};
    use crate::types::{Opcode, Operand};

    fn make_const_choice_op(id: usize, value: i64, dest: VReg, cycle: i32) -> SsaOp {
        let mut op = SsaOp::new(
            id,
            Engine::Load,
            Opcode::ConstChoice,
            Some(dest),
            vec![Operand::Imm(value)],
        );
        op.produces_constant = Some(value);
        op.const_alternatives = Some(vec![
            crate::types::ConstEncoding::Load,
            crate::types::ConstEncoding::AddImm { value },
        ]);
        op.cycle = cycle;
        op
    }

    #[test]
    fn test_lower_constchoice_load() {
        let mut vreg_to_addr = HashMap::new();
        let dest_vreg = VReg::scalar(1);
        vreg_to_addr.insert(dest_vreg, 100);

        let ssa_op = make_const_choice_op(42, 5, dest_vreg, 10);

        let ssa_ops = vec![ssa_op];
        let shifted_ops = HashSet::new();
        let const_choice_encoding = HashMap::new(); // Use Load (None => Load)

        let result = lower_ssa_ops_with_shift(&LoweringContext {
            ssa_ops: &ssa_ops,
            vreg_to_addr: &vreg_to_addr,
            shifted_ops: &shifted_ops,
            const_choice_encoding: &const_choice_encoding,
            select_choice_encoding: &HashMap::new(),
            shifted_alu_cycles: &HashMap::new(),
            eliminated_vbroadcast_ops: &HashSet::new(),
            eliminated_vb_to_scalar_producer: &HashMap::new(),
            cse_groups: None,
            cse_group_results: None,
        }).unwrap();

        assert_eq!(result.len(), 1);
        let op = &result[0];
        assert_eq!(op.engine, Engine::Load);
        assert_eq!(op.id, 42);
        assert_eq!(op.cycle, 10);
        assert_eq!(op.slot[0], SlotVal::Str("const".to_string()));
        assert_eq!(op.slot[1], SlotVal::Int(100)); // dest addr
        assert_eq!(op.slot[2], SlotVal::Int(5)); // value
        assert_eq!(op.produces_constant, Some(5));
        assert!(op.reads.is_empty()); // Const doesn't read anything
    }

    #[test]
    fn test_lower_emits_trace_debug_ops_when_trace_id_set() {
        let mut vreg_to_addr = HashMap::new();
        let dest_vreg = VReg::scalar(1);
        vreg_to_addr.insert(dest_vreg, 100);

        let mut ssa_op = make_const_choice_op(42, 5, dest_vreg, 10);
        ssa_op.trace_id = Some("r0_g0_hashed_val".to_string());

        let ssa_ops = vec![ssa_op];
        let shifted_ops = HashSet::new();
        let const_choice_encoding = HashMap::new();

        let result = lower_ssa_ops_with_shift(&LoweringContext {
            ssa_ops: &ssa_ops,
            vreg_to_addr: &vreg_to_addr,
            shifted_ops: &shifted_ops,
            const_choice_encoding: &const_choice_encoding,
            select_choice_encoding: &HashMap::new(),
            shifted_alu_cycles: &HashMap::new(),
            eliminated_vbroadcast_ops: &HashSet::new(),
            eliminated_vb_to_scalar_producer: &HashMap::new(),
            cse_groups: None,
            cse_group_results: None,
        }).unwrap();

        let debug_ops: Vec<_> = result.iter().filter(|o| o.engine == Engine::Debug).collect();
        assert!(
            debug_ops.len() >= 2,
            "Expected at least 2 debug ops (trace_input + trace_output), got {}",
            debug_ops.len()
        );
        let trace_input = debug_ops
            .iter()
            .find(|o| matches!(&o.slot[0], SlotVal::Str(s) if s == "trace_input"));
        let trace_output = debug_ops
            .iter()
            .find(|o| matches!(&o.slot[0], SlotVal::Str(s) if s == "trace_output"));
        assert!(trace_input.is_some(), "Expected trace_input debug op");
        assert!(trace_output.is_some(), "Expected trace_output debug op");
    }

    #[test]
    fn test_lower_constchoice_alu_derive() {
        use crate::types::Opcode;
        let mut vreg_to_addr = HashMap::new();
        let v1 = VReg::scalar(1);
        let v2 = VReg::scalar(2);
        let v3 = VReg::scalar(3);
        vreg_to_addr.insert(v1, 10);
        vreg_to_addr.insert(v2, 20);
        vreg_to_addr.insert(v3, 30);

        let op1 = make_const_choice_op(1, 1, v1, 0);
        let op2 = make_const_choice_op(2, 2, v2, 0);
        let mut op3 = SsaOp::new(
            3,
            Engine::Load,
            Opcode::ConstChoice,
            Some(v3),
            vec![Operand::Imm(3)],
        );
        op3.produces_constant = Some(3);
        op3.const_alternatives = Some(vec![
            crate::types::ConstEncoding::Load,
            crate::types::ConstEncoding::AddImm { value: 3 },
            crate::types::ConstEncoding::Add {
                a_value: 1,
                b_value: 2,
            },
        ]);
        op3.cycle = 1;

        let ssa_ops = vec![op1, op2, op3];
        let shifted_ops = HashSet::new();
        let mut const_choice_encoding = HashMap::new();
        const_choice_encoding.insert(1, ConstChoiceEncoding::Load);
        const_choice_encoding.insert(2, ConstChoiceEncoding::Load);
        const_choice_encoding.insert(
            3,
            ConstChoiceEncoding::AluDerive(crate::types::AluDeriveInfo {
                left_op_id: 1,
                right_op_id: 2,
                opcode: Opcode::Add,
            }),
        ); // op 3 = op1 + op2

        let result = lower_ssa_ops_with_shift(&LoweringContext {
            ssa_ops: &ssa_ops,
            vreg_to_addr: &vreg_to_addr,
            shifted_ops: &shifted_ops,
            const_choice_encoding: &const_choice_encoding,
            select_choice_encoding: &HashMap::new(),
            shifted_alu_cycles: &HashMap::new(),
            eliminated_vbroadcast_ops: &HashSet::new(),
            eliminated_vb_to_scalar_producer: &HashMap::new(),
            cse_groups: None,
            cse_group_results: None,
        }).unwrap();

        assert_eq!(result.len(), 3); // 3 ConstChoice ops -> 3 lowered ops
        let alu_op = result
            .iter()
            .find(|o| o.produces_constant == Some(3))
            .expect("should find op producing 3");
        assert_eq!(alu_op.engine, Engine::Alu);
        assert_eq!(alu_op.slot[0], SlotVal::Str("+".to_string()));
        assert_eq!(alu_op.slot[1], SlotVal::Int(30)); // dest
        assert_eq!(alu_op.slot[2], SlotVal::Int(10)); // left (op1)
        assert_eq!(alu_op.slot[3], SlotVal::Int(20)); // right (op2)
        assert!(alu_op.reads.contains(&10));
        assert!(alu_op.reads.contains(&20));
        assert_eq!(alu_op.writes.len(), 1);
        assert!(alu_op.writes.contains(&30));
    }

    #[test]
    fn test_lower_constchoice_addimm() {
        let mut vreg_to_addr = HashMap::new();
        let dest_vreg = VReg::scalar(1);
        vreg_to_addr.insert(dest_vreg, 200);

        let ssa_op = make_const_choice_op(99, 256, dest_vreg, 5);

        let ssa_ops = vec![ssa_op];
        let shifted_ops = HashSet::new();
        let mut const_choice_encoding = HashMap::new();
        const_choice_encoding.insert(99, ConstChoiceEncoding::AddImm);

        let result = lower_ssa_ops_with_shift(&LoweringContext {
            ssa_ops: &ssa_ops,
            vreg_to_addr: &vreg_to_addr,
            shifted_ops: &shifted_ops,
            const_choice_encoding: &const_choice_encoding,
            select_choice_encoding: &HashMap::new(),
            shifted_alu_cycles: &HashMap::new(),
            eliminated_vbroadcast_ops: &HashSet::new(),
            eliminated_vb_to_scalar_producer: &HashMap::new(),
            cse_groups: None,
            cse_group_results: None,
        }).unwrap();

        assert_eq!(result.len(), 1);
        let op = &result[0];
        assert_eq!(op.engine, Engine::Flow);
        assert_eq!(op.id, 99);
        assert_eq!(op.cycle, 5);
        assert_eq!(op.slot[0], SlotVal::Str("add_imm".to_string()));
        assert_eq!(op.slot[1], SlotVal::Int(200)); // dest addr
        assert_eq!(op.slot[2], SlotVal::Int(0)); // zero VReg addr
        assert_eq!(op.slot[3], SlotVal::Int(256)); // value
        assert_eq!(op.produces_constant, Some(256));
        assert!(op.reads.contains(&0)); // Reads from zero VReg
        assert!(op.writes.contains(&200));
    }

    #[test]
    fn test_lower_constchoice_mixed() {
        let mut vreg_to_addr = HashMap::new();
        let v1 = VReg::scalar(1);
        let v2 = VReg::scalar(2);
        vreg_to_addr.insert(v1, 100);
        vreg_to_addr.insert(v2, 200);

        let op1 = make_const_choice_op(1, 10, v1, 0);
        let op2 = make_const_choice_op(2, 20, v2, 0);

        let ssa_ops = vec![op1, op2];
        let shifted_ops = HashSet::new();
        let mut const_choice_encoding = HashMap::new();
        const_choice_encoding.insert(1, ConstChoiceEncoding::AddImm);
        const_choice_encoding.insert(2, ConstChoiceEncoding::Load);

        let result = lower_ssa_ops_with_shift(&LoweringContext {
            ssa_ops: &ssa_ops,
            vreg_to_addr: &vreg_to_addr,
            shifted_ops: &shifted_ops,
            const_choice_encoding: &const_choice_encoding,
            select_choice_encoding: &HashMap::new(),
            shifted_alu_cycles: &HashMap::new(),
            eliminated_vbroadcast_ops: &HashSet::new(),
            eliminated_vb_to_scalar_producer: &HashMap::new(),
            cse_groups: None,
            cse_group_results: None,
        }).unwrap();

        assert_eq!(result.len(), 2);

        // First op: AddImm
        assert_eq!(result[0].engine, Engine::Flow);
        assert_eq!(result[0].slot[0], SlotVal::Str("add_imm".to_string()));

        // Second op: Load
        assert_eq!(result[1].engine, Engine::Load);
        assert_eq!(result[1].slot[0], SlotVal::Str("const".to_string()));
    }

    // =========================================================================
    // SelectChoice lowering tests
    // =========================================================================

    fn make_select_choice_op(
        id: usize,
        cond: VReg,
        true_val: VReg,
        false_val: VReg,
        delta: Option<VReg>,
        cycle: i32,
    ) -> SsaOp {
        let mut op = SsaOp::new(
            id,
            Engine::Flow, // Default engine
            Opcode::SelectChoice,
            Some(VReg::vector(100 + id as u32)), // dest
            vec![
                Operand::VReg(cond),
                Operand::VReg(true_val),
                Operand::VReg(false_val),
            ],
        );
        op.delta_vreg = delta;
        op.cycle = cycle;
        op
    }

    /// Test that SelectChoice vselect path creates valid addresses
    #[test]
    fn test_lower_selectchoice_vselect_path() {
        let mut vreg_to_addr = HashMap::new();
        let cond = VReg::vector(1);
        let true_val = VReg::vector(2);
        let false_val = VReg::vector(3);
        let dest = VReg::vector(100);

        vreg_to_addr.insert(cond, 100);
        vreg_to_addr.insert(true_val, 200);
        vreg_to_addr.insert(false_val, 300);
        vreg_to_addr.insert(dest, 400);

        let mut op = make_select_choice_op(42, cond, true_val, false_val, None, 10);
        op.dest = Some(dest);

        let ssa_ops = vec![op];
        let shifted_ops = HashSet::new();
        let const_choice_encoding = HashMap::new();
        let select_choice_enc = HashMap::new(); // vselect path

        let result = lower_ssa_ops_with_shift(&LoweringContext {
            ssa_ops: &ssa_ops,
            vreg_to_addr: &vreg_to_addr,
            shifted_ops: &shifted_ops,
            const_choice_encoding: &const_choice_encoding,
            select_choice_encoding: &select_choice_enc,
            shifted_alu_cycles: &HashMap::new(),
            eliminated_vbroadcast_ops: &HashSet::new(),
            eliminated_vb_to_scalar_producer: &HashMap::new(),
            cse_groups: None,
            cse_group_results: None,
        }).unwrap();

        assert_eq!(result.len(), 1, "vselect should produce 1 op");
        let op = &result[0];
        assert_eq!(op.slot[0], SlotVal::Str("vselect".to_string()));

        // Verify all addresses are valid (< scratch budget)
        for addr in &op.reads {
            assert!(*addr < 1536, "Read address {} exceeds scratch budget", addr);
        }
        for addr in &op.writes {
            assert!(
                *addr < 1536,
                "Write address {} exceeds scratch budget",
                addr
            );
        }
    }

    /// Test that SelectChoice FMA path WITH delta creates valid addresses
    #[test]
    fn test_lower_selectchoice_fma_with_delta() {
        let mut vreg_to_addr = HashMap::new();
        let cond = VReg::vector(1);
        let true_val = VReg::vector(2);
        let false_val = VReg::vector(3);
        let delta = VReg::vector(4);
        let dest = VReg::vector(100);

        vreg_to_addr.insert(cond, 100);
        vreg_to_addr.insert(true_val, 200);
        vreg_to_addr.insert(false_val, 300);
        vreg_to_addr.insert(delta, 500);
        vreg_to_addr.insert(dest, 400);

        let mut op = make_select_choice_op(42, cond, true_val, false_val, Some(delta), 10);
        op.dest = Some(dest);

        let ssa_ops = vec![op];
        let shifted_ops = HashSet::new();
        let const_choice_encoding = HashMap::new();
        let mut select_choice_enc = HashMap::new();
        select_choice_enc.insert(42, SelectChoiceEncoding::Fma);

        let result = lower_ssa_ops_with_shift(&LoweringContext {
            ssa_ops: &ssa_ops,
            vreg_to_addr: &vreg_to_addr,
            shifted_ops: &shifted_ops,
            const_choice_encoding: &const_choice_encoding,
            select_choice_encoding: &select_choice_enc,
            shifted_alu_cycles: &HashMap::new(),
            eliminated_vbroadcast_ops: &HashSet::new(),
            eliminated_vb_to_scalar_producer: &HashMap::new(),
            cse_groups: None,
            cse_group_results: None,
        }).unwrap();

        assert_eq!(result.len(), 1, "FMA with delta should produce 1 op");
        let op = &result[0];
        assert_eq!(op.slot[0], SlotVal::Str("multiply_add".to_string()));

        // Verify all addresses are valid (< scratch budget)
        for addr in &op.reads {
            assert!(*addr < 1536, "Read address {} exceeds scratch budget", addr);
        }
        for addr in &op.writes {
            assert!(
                *addr < 1536,
                "Write address {} exceeds scratch budget",
                addr
            );
        }
    }

    /// Test that SelectChoice FMA path WITHOUT delta creates valid addresses
    /// Previously: Created addresses like 10000 + op_id which are invalid!
    /// Now: Uses sub_temp_vreg allocated during emission
    #[test]
    fn test_lower_selectchoice_fma_without_delta_addresses_valid() {
        let mut vreg_to_addr = HashMap::new();
        let cond = VReg::vector(1);
        let true_val = VReg::vector(2);
        let false_val = VReg::vector(3);
        let dest = VReg::vector(100);
        let sub_temp = VReg::vector(101); // Allocated for sub temp

        vreg_to_addr.insert(cond, 100);
        vreg_to_addr.insert(true_val, 200);
        vreg_to_addr.insert(false_val, 300);
        vreg_to_addr.insert(dest, 400);
        vreg_to_addr.insert(sub_temp, 500); // Valid address for sub temp

        let mut op = make_select_choice_op(42, cond, true_val, false_val, None, 10);
        op.dest = Some(dest);
        op.sub_temp_vreg = Some(sub_temp); // Set the sub_temp_vreg

        let ssa_ops = vec![op];
        let shifted_ops = HashSet::new();
        let const_choice_encoding = HashMap::new();
        let mut select_choice_encoding = HashMap::new();
        select_choice_encoding.insert(42, SelectChoiceEncoding::Fma);

        let result = lower_ssa_ops_with_shift(&LoweringContext {
            ssa_ops: &ssa_ops,
            vreg_to_addr: &vreg_to_addr,
            shifted_ops: &shifted_ops,
            const_choice_encoding: &const_choice_encoding,
            select_choice_encoding: &select_choice_encoding,
            shifted_alu_cycles: &HashMap::new(),
            eliminated_vbroadcast_ops: &HashSet::new(),
            eliminated_vb_to_scalar_producer: &HashMap::new(),
            cse_groups: None,
            cse_group_results: None,
        }).unwrap();

        // FMA without delta should produce 2 ops: sub + fma
        assert_eq!(
            result.len(),
            2,
            "FMA without delta should produce 2 ops (sub + fma)"
        );

        let sub_op = &result[0];
        let fma_op = &result[1];

        assert_eq!(sub_op.slot[0], SlotVal::Str("-".to_string()));
        assert_eq!(fma_op.slot[0], SlotVal::Str("multiply_add".to_string()));

        // CRITICAL: Verify all addresses are valid (< scratch budget)
        // This test should FAIL currently because sub_temp_addr = 10000 + 42 = 10042
        for addr in &sub_op.reads {
            assert!(
                *addr < 1536,
                "Sub read address {} exceeds scratch budget",
                addr
            );
        }
        for addr in &sub_op.writes {
            assert!(
                *addr < 1536,
                "Sub write address {} exceeds scratch budget (BUG: using 10000 + op_id)",
                addr
            );
        }
        for addr in &fma_op.reads {
            assert!(
                *addr < 1536,
                "FMA read address {} exceeds scratch budget (BUG: reading from sub temp)",
                addr
            );
        }
        for addr in &fma_op.writes {
            assert!(
                *addr < 1536,
                "FMA write address {} exceeds scratch budget",
                addr
            );
        }
    }

    /// Test addresses in slot values are also valid
    #[test]
    fn test_lower_selectchoice_fma_without_delta_slot_addresses_valid() {
        let mut vreg_to_addr = HashMap::new();
        let cond = VReg::vector(1);
        let true_val = VReg::vector(2);
        let false_val = VReg::vector(3);
        let dest = VReg::vector(100);
        let sub_temp = VReg::vector(101);

        vreg_to_addr.insert(cond, 100);
        vreg_to_addr.insert(true_val, 200);
        vreg_to_addr.insert(false_val, 300);
        vreg_to_addr.insert(dest, 400);
        vreg_to_addr.insert(sub_temp, 500);

        let mut op = make_select_choice_op(42, cond, true_val, false_val, None, 10);
        op.dest = Some(dest);
        op.sub_temp_vreg = Some(sub_temp);

        let ssa_ops = vec![op];
        let shifted_ops = HashSet::new();
        let const_choice_encoding = HashMap::new();
        let mut select_choice_encoding = HashMap::new();
        select_choice_encoding.insert(42, SelectChoiceEncoding::Fma);

        let result = lower_ssa_ops_with_shift(&LoweringContext {
            ssa_ops: &ssa_ops,
            vreg_to_addr: &vreg_to_addr,
            shifted_ops: &shifted_ops,
            const_choice_encoding: &const_choice_encoding,
            select_choice_encoding: &select_choice_encoding,
            shifted_alu_cycles: &HashMap::new(),
            eliminated_vbroadcast_ops: &HashSet::new(),
            eliminated_vb_to_scalar_producer: &HashMap::new(),
            cse_groups: None,
            cse_group_results: None,
        }).unwrap();

        // Check slot values for invalid addresses
        for op in &result {
            for slot_val in op.slot.iter() {
                if let SlotVal::Int(addr) = slot_val {
                    assert!(
                        *addr < 1536,
                        "Slot contains invalid address {}: {} (op: {:?})",
                        addr,
                        if *addr >= 10000 {
                            "BUG: using fake temp address 10000 + op_id"
                        } else {
                            ""
                        },
                        op.slot[0]
                    );
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // SelectChoice CSE group lowering tests
    // -------------------------------------------------------------------------

    /// With non-None cse_groups and cse_group_results, shared SUBs are emitted for
    /// active CSE groups (one SUB per active group), and FMA consumers in that group
    /// do not emit their own SUB.
    #[test]
    fn test_lower_selectchoice_cse_shared_sub_emitted() {
        let mut vreg_to_addr = HashMap::new();
        let cond = VReg::vector(1);
        let true_val = VReg::vector(2);
        let false_val = VReg::vector(3);
        let dest = VReg::vector(100);
        let sub_temp_cse = VReg::vector(201); // CSE group's shared sub_temp

        vreg_to_addr.insert(cond, 100);
        vreg_to_addr.insert(true_val, 200);
        vreg_to_addr.insert(false_val, 300);
        vreg_to_addr.insert(dest, 400);
        vreg_to_addr.insert(sub_temp_cse, 500);

        let mut op = make_select_choice_op(42, cond, true_val, false_val, None, 10);
        op.dest = Some(dest);
        op.sub_temp_vreg = Some(VReg::vector(101)); // per-op temp (ignored when CSE active)

        let ssa_ops = vec![op];
        let shifted_ops = HashSet::new();
        let const_choice_encoding = HashMap::new();
        let mut select_choice_enc = HashMap::new();
        select_choice_enc.insert(42, SelectChoiceEncoding::Fma);

        let cse_groups: Vec<CseGroup> = vec![CseGroup {
            true_val,
            false_val,
            consumer_indices: vec![0],
            producer_true_idx: None,
            producer_false_idx: None,
            complex_phi: true,
            sub_temp_vreg: sub_temp_cse,
        }];
        let cse_group_results: Vec<CseGroupResult> = vec![CseGroupResult {
            sub_active: true,
            sub_cycle: 5,
        }];

        let result = lower_ssa_ops_with_shift(&LoweringContext {
            ssa_ops: &ssa_ops,
            vreg_to_addr: &vreg_to_addr,
            shifted_ops: &shifted_ops,
            const_choice_encoding: &const_choice_encoding,
            select_choice_encoding: &select_choice_enc,
            shifted_alu_cycles: &HashMap::new(),
            eliminated_vbroadcast_ops: &HashSet::new(),
            eliminated_vb_to_scalar_producer: &HashMap::new(),
            cse_groups: Some(&cse_groups),
            cse_group_results: Some(&cse_group_results),
        }).unwrap();

        // One shared SUB + one FMA (no per-op SUB)
        assert_eq!(result.len(), 2, "CSE path: 1 shared SUB + 1 FMA");
        let sub_op = &result[0];
        let fma_op = &result[1];
        assert_eq!(sub_op.slot[0], SlotVal::Str("-".to_string()));
        assert_eq!(sub_op.engine, Engine::Valu);
        assert_eq!(sub_op.slot[1], SlotVal::Int(500)); // sub_temp_cse addr
        assert_eq!(fma_op.slot[0], SlotVal::Str("multiply_add".to_string()));
    }

    /// FMA consumer in a CSE group must use the shared SUB address in its reads
    /// (sub_temp), not a per-op temp.
    #[test]
    fn test_lower_selectchoice_cse_fma_uses_shared_sub_address() {
        let mut vreg_to_addr = HashMap::new();
        let cond = VReg::vector(1);
        let true_val = VReg::vector(2);
        let false_val = VReg::vector(3);
        let dest = VReg::vector(100);
        let shared_sub_addr = 500i64;
        let sub_temp_cse = VReg::vector(201);

        vreg_to_addr.insert(cond, 100);
        vreg_to_addr.insert(true_val, 200);
        vreg_to_addr.insert(false_val, 300);
        vreg_to_addr.insert(dest, 400);
        vreg_to_addr.insert(sub_temp_cse, shared_sub_addr);

        let mut op = make_select_choice_op(42, cond, true_val, false_val, None, 10);
        op.dest = Some(dest);
        op.sub_temp_vreg = Some(VReg::vector(101));

        let ssa_ops = vec![op];
        let shifted_ops = HashSet::new();
        let const_choice_encoding = HashMap::new();
        let mut select_choice_enc = HashMap::new();
        select_choice_enc.insert(42, SelectChoiceEncoding::Fma);

        let cse_groups: Vec<CseGroup> = vec![CseGroup {
            true_val,
            false_val,
            consumer_indices: vec![0],
            producer_true_idx: None,
            producer_false_idx: None,
            complex_phi: true,
            sub_temp_vreg: sub_temp_cse,
        }];
        let cse_group_results: Vec<CseGroupResult> = vec![CseGroupResult {
            sub_active: true,
            sub_cycle: 5,
        }];

        let result = lower_ssa_ops_with_shift(&LoweringContext {
            ssa_ops: &ssa_ops,
            vreg_to_addr: &vreg_to_addr,
            shifted_ops: &shifted_ops,
            const_choice_encoding: &const_choice_encoding,
            select_choice_encoding: &select_choice_enc,
            shifted_alu_cycles: &HashMap::new(),
            eliminated_vbroadcast_ops: &HashSet::new(),
            eliminated_vb_to_scalar_producer: &HashMap::new(),
            cse_groups: Some(&cse_groups),
            cse_group_results: Some(&cse_group_results),
        }).unwrap();

        let fma_op = &result[1];
        // FMA reads: cond, sub_temp, false_val (vector lanes)
        assert!(
            fma_op.reads.contains(&shared_sub_addr),
            "FMA must read from shared SUB address {}",
            shared_sub_addr
        );
        // Slot operands: multiply_add, dest, cond, sub_temp, false_val
        assert_eq!(fma_op.slot[3], SlotVal::Int(shared_sub_addr));
    }

    /// Mismatched cse_groups vs cse_group_results lengths: lowering skips CSE mapping,
    /// logs a diagnostic, and each FMA consumer gets its own SUB (no shared SUBs).
    #[test]
    fn test_lower_selectchoice_cse_mismatched_lengths_fallback() {
        let mut vreg_to_addr = HashMap::new();
        let cond = VReg::vector(1);
        let true_val = VReg::vector(2);
        let false_val = VReg::vector(3);
        let dest1 = VReg::vector(100);
        let dest2 = VReg::vector(101);
        let sub_temp1 = VReg::vector(102);
        let sub_temp2 = VReg::vector(103);

        vreg_to_addr.insert(cond, 100);
        vreg_to_addr.insert(true_val, 200);
        vreg_to_addr.insert(false_val, 300);
        vreg_to_addr.insert(dest1, 400);
        vreg_to_addr.insert(dest2, 401);
        vreg_to_addr.insert(sub_temp1, 500);
        vreg_to_addr.insert(sub_temp2, 501);

        let mut op1 = make_select_choice_op(42, cond, true_val, false_val, None, 10);
        op1.dest = Some(dest1);
        op1.sub_temp_vreg = Some(sub_temp1);
        let mut op2 = make_select_choice_op(43, cond, true_val, false_val, None, 10);
        op2.dest = Some(dest2);
        op2.sub_temp_vreg = Some(sub_temp2);

        let ssa_ops = vec![op1, op2];
        let shifted_ops = HashSet::new();
        let const_choice_encoding = HashMap::new();
        let mut select_choice_encoding = HashMap::new();
        select_choice_encoding.insert(42, SelectChoiceEncoding::Fma);
        select_choice_encoding.insert(43, SelectChoiceEncoding::Fma);

        // Mismatch: one group, two results
        let cse_groups: Vec<CseGroup> = vec![CseGroup {
            true_val,
            false_val,
            consumer_indices: vec![0, 1],
            producer_true_idx: None,
            producer_false_idx: None,
            complex_phi: true,
            sub_temp_vreg: VReg::vector(200),
        }];
        let cse_group_results: Vec<CseGroupResult> = vec![
            CseGroupResult {
                sub_active: true,
                sub_cycle: 5,
            },
            CseGroupResult {
                sub_active: true,
                sub_cycle: 6,
            },
        ];

        let result = lower_ssa_ops_with_shift(&LoweringContext {
            ssa_ops: &ssa_ops,
            vreg_to_addr: &vreg_to_addr,
            shifted_ops: &shifted_ops,
            const_choice_encoding: &const_choice_encoding,
            select_choice_encoding: &select_choice_encoding,
            shifted_alu_cycles: &HashMap::new(),
            eliminated_vbroadcast_ops: &HashSet::new(),
            eliminated_vb_to_scalar_producer: &HashMap::new(),
            cse_groups: Some(&cse_groups),
            cse_group_results: Some(&cse_group_results),
        }).unwrap();

        // CSE mapping skipped -> each consumer emits SUB + FMA -> 4 ops
        assert_eq!(
            result.len(),
            4,
            "Mismatched lengths: CSE skipped, expect 2 SUBs + 2 FMAs"
        );
        let sub_count = result
            .iter()
            .filter(|o| o.slot.get(0) == Some(&SlotVal::Str("-".to_string())))
            .count();
        let fma_count = result
            .iter()
            .filter(|o| o.slot.get(0) == Some(&SlotVal::Str("multiply_add".to_string())))
            .count();
        assert_eq!(sub_count, 2, "Each FMA consumer gets its own SUB when CSE skipped");
        assert_eq!(fma_count, 2);
    }
}
