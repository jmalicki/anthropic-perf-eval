//! SSA lowering helpers.
//!
//! This module handles lowering SSA ops to concrete ops, including
//! VALU→ALU shift expansion.

use std::collections::{HashMap, HashSet};

use arrayvec::ArrayVec;

use crate::address_assign::lower_ssa_op;
use crate::types::{Engine, Op, SlotVal, SsaOp, VReg, VLEN};

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
pub fn lower_ssa_ops(ssa_ops: &[SsaOp], vreg_to_addr: &HashMap<VReg, i64>) -> Vec<Op> {
    ssa_ops
        .iter()
        .map(|op| lower_ssa_op(op, vreg_to_addr))
        .collect()
}

/// Lower SSA ops to concrete ops, expanding shifted VALU ops to ALU.
///
/// Shifted VALU ops are expanded to 8 scalar ALU ops, one per lane.
/// Each ALU op reads from src[i] and writes to dest[i].
///
/// ConstChoice ops (single op from emit_const_choice) are lowered based on the scheduler's decision:
/// - If in `const_add_imm_ops`: emit AddImm op (Flow engine)
/// - Otherwise: emit Const op (Load engine)
///
/// The SSA kernel path uses only emit_const_choice(); const_add_imm_ops comes from CP-SAT
/// (or an empty set for greedy). The legacy emit_const() dual-op path is used only in tests;
/// there, AddImm ops are filtered before scheduling, so lowering always sees only the Load op.
///
/// SelectChoice ops are lowered based on the scheduler's decision:
/// - If in `select_fma_ops`: emit sub + fma ops (both VALU, sequential)
/// - Otherwise: emit vselect op (Flow engine)
pub fn lower_ssa_ops_with_shift(
    ssa_ops: &[SsaOp],
    vreg_to_addr: &HashMap<VReg, i64>,
    shifted_ops: &HashSet<usize>,
    const_add_imm_ops: &HashSet<usize>,
    select_fma_ops: &HashSet<usize>,
    shifted_alu_cycles: &HashMap<usize, Vec<i32>>, // op_id -> [cycle0, ..., cycle7]
    eliminated_vbroadcast_ops: &HashSet<usize>,
    eliminated_vb_to_scalar_producer: &HashMap<usize, usize>, // vb op_id -> scalar producer op_id
) -> Vec<Op> {
    use crate::types::{Opcode, SlotVal};
    use arrayvec::ArrayVec;

    let mut result = Vec::new();

    // Build map: eliminated vb dest VReg -> scalar producer address (for shifted consumers reading from vb)
    let eliminated_vb_vreg_to_scalar_addr: HashMap<VReg, i64> = {
        let mut m = HashMap::new();
        for (&vb_id, &scalar_id) in eliminated_vb_to_scalar_producer {
            if !eliminated_vbroadcast_ops.contains(&vb_id) {
                continue;
            }
            let vb_dest = ssa_ops
                .iter()
                .find(|o| o.id == vb_id)
                .and_then(|o| o.dest);
            let scalar_dest = ssa_ops
                .iter()
                .find(|o| o.id == scalar_id)
                .and_then(|o| o.dest);
            if let (Some(vreg), Some(svreg)) = (vb_dest, scalar_dest) {
                if let Some(&addr) = vreg_to_addr.get(&svreg) {
                    m.insert(vreg, addr);
                }
            }
        }
        m
    };

    for ssa_op in ssa_ops {
        if eliminated_vbroadcast_ops.contains(&ssa_op.id) {
            continue;
        }
        if shifted_ops.contains(&ssa_op.id) {
            // Expand shifted VALU op to 8 ALU ops
            // Use individual cycles if available (from CP-SAT), otherwise use completion_cycle
            let alu_cycles = shifted_alu_cycles.get(&ssa_op.id);
            let expanded = expand_ssa_valu_to_alu(
                ssa_op,
                vreg_to_addr,
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
            let dest_addr = *vreg_to_addr.get(&dest).expect("dest VReg not in mapping");
            let value = ssa_op
                .produces_constant
                .expect("ConstChoice op must have produces_constant");

            if const_add_imm_ops.contains(&ssa_op.id) {
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
                    dep_ids: ssa_op.dep_ids.clone(),
                    reads,
                    writes,
                    last_use_reads: HashSet::new(),
                    produces_constant: Some(value),
                });
            } else {
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
                    dep_ids: ssa_op.dep_ids.clone(),
                    reads: HashSet::new(),
                    writes,
                    last_use_reads: HashSet::new(),
                    produces_constant: Some(value),
                });
            }
        } else if ssa_op.opcode == Opcode::SelectChoice {
            // Handle SelectChoice based on scheduler decision
            let dest = ssa_op.dest.expect("SelectChoice op must have dest");
            let dest_addr = *vreg_to_addr.get(&dest).expect("dest VReg not in mapping");
            if std::env::var("SCHEDULER_DEBUG").is_ok() {
                eprintln!(
                    "[DEBUG lower_ssa_ops_with_shift] SelectChoice op.id={} cycle={} select_fma_ops.contains={} sub_temp_vreg={:?}",
                    ssa_op.id, ssa_op.cycle, select_fma_ops.contains(&ssa_op.id), ssa_op.sub_temp_vreg
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

            let cond_addr = *vreg_to_addr
                .get(&cond_vreg)
                .expect("cond VReg not in mapping");
            let true_val_addr = *vreg_to_addr
                .get(&true_val_vreg)
                .expect("true_val VReg not in mapping");
            let false_val_addr = *vreg_to_addr
                .get(&false_val_vreg)
                .expect("false_val VReg not in mapping");

            if select_fma_ops.contains(&ssa_op.id) {
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
                    let delta_addr = *vreg_to_addr
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
                        dep_ids: ssa_op.dep_ids.clone(),
                        reads: fma_reads,
                        writes: fma_writes,
                        last_use_reads: HashSet::new(),
                        produces_constant: None,
                    });
                } else {
                    // No precomputed delta: emit sub + fma ops (both VALU, sequential)
                    // sub = true_val - false_val
                    // fma = cond * sub + false_val

                    // Get the sub_temp VReg that was allocated during emission
                    let sub_temp_vreg = ssa_op.sub_temp_vreg.expect(
                        "SelectChoice without delta must have sub_temp_vreg allocated during emission"
                    );
                    let sub_temp_addr = *vreg_to_addr.get(&sub_temp_vreg).expect(
                        "sub_temp_vreg not in address mapping - was it included in address assignment?"
                    );
                    if std::env::var("SCHEDULER_DEBUG").is_ok() {
                        eprintln!(
                            "[DEBUG lower_ssa_ops_with_shift] SelectChoice FMA (op.id={} cycle={}): sub_temp_vreg={:?} -> addr {}, dest={:?} -> addr {}, cond={:?} -> addr {}, true_val={:?} -> addr {}, false_val={:?} -> addr {}",
                            ssa_op.id, ssa_op.cycle,
                            sub_temp_vreg, sub_temp_addr,
                            ssa_op.dest.unwrap(), dest_addr,
                            cond_vreg, cond_addr,
                            true_val_vreg, true_val_addr,
                            false_val_vreg, false_val_addr
                        );
                    }

                    // Sub op: sub_temp = true_val - false_val
                    let mut sub_slot: ArrayVec<SlotVal, 6> = ArrayVec::new();
                    sub_slot.push(SlotVal::Str("-".to_string()));
                    sub_slot.push(SlotVal::Int(sub_temp_addr));
                    sub_slot.push(SlotVal::Int(true_val_addr));
                    sub_slot.push(SlotVal::Int(false_val_addr));

                    let mut sub_reads = HashSet::new();
                    // Expand vector addresses: all SelectChoice operands are vectors
                    for i in 0..VLEN as i64 {
                        sub_reads.insert(true_val_addr + i);
                        sub_reads.insert(false_val_addr + i);
                    }

                    let mut sub_writes = HashSet::new();
                    // sub_temp is a vector VReg
                    for i in 0..VLEN as i64 {
                        sub_writes.insert(sub_temp_addr + i);
                    }

                    let sub_op = Op {
                        id: ssa_op.id, // Use same ID for sub (will be different for fma)
                        engine: crate::types::Engine::Valu,
                        slot: sub_slot,
                        cycle: ssa_op.cycle, // Sub is at the same cycle as SelectChoice
                        dep_ids: ssa_op.dep_ids.clone(),
                        reads: sub_reads,
                        writes: sub_writes,
                        last_use_reads: HashSet::new(),
                        produces_constant: None,
                    };
                    if std::env::var("SCHEDULER_DEBUG").is_ok() {
                        eprintln!(
                            "[DEBUG lower_ssa_ops_with_shift] SelectChoice FMA sub: op.id={} cycle={} writes={:?} reads={:?}",
                            sub_op.id, sub_op.cycle, sub_op.writes, sub_op.reads
                        );
                    }
                    result.push(sub_op);

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
                            let mut deps = ssa_op.dep_ids.clone();
                            deps.insert(ssa_op.id); // FMA depends on sub
                            deps
                        },
                        reads: fma_reads,
                        writes: fma_writes,
                        last_use_reads: HashSet::new(),
                        produces_constant: None,
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
                    dep_ids: ssa_op.dep_ids.clone(),
                    reads,
                    writes,
                    last_use_reads: HashSet::new(),
                    produces_constant: None,
                });
            }
        } else {
            // Normal lowering
            result.push(lower_ssa_op(ssa_op, vreg_to_addr));
        }
    }

    result
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
            id: ssa_op.id * 1000 + i, // Unique ID for expanded op
            engine: Engine::Alu,
            slot,
            cycle,
            dep_ids: ssa_op.dep_ids.clone(),
            reads,
            writes,
            last_use_reads: HashSet::new(),
            produces_constant: None,
        });
    }

    ops
}

#[cfg(test)]
mod tests {
    use super::*;
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
        let const_add_imm_ops = HashSet::new(); // Use Load

        let result = lower_ssa_ops_with_shift(
            &ssa_ops,
            &vreg_to_addr,
            &shifted_ops,
            &const_add_imm_ops,
            &HashSet::new(),
            &HashMap::new(),
            &HashSet::new(),
            &HashMap::new(),
        );

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
        assert!(op.writes.contains(&100));
    }

    #[test]
    fn test_lower_constchoice_addimm() {
        let mut vreg_to_addr = HashMap::new();
        let dest_vreg = VReg::scalar(1);
        vreg_to_addr.insert(dest_vreg, 200);

        let ssa_op = make_const_choice_op(99, 256, dest_vreg, 5);

        let ssa_ops = vec![ssa_op];
        let shifted_ops = HashSet::new();
        let mut const_add_imm_ops = HashSet::new();
        const_add_imm_ops.insert(99); // Use AddImm

        let result = lower_ssa_ops_with_shift(
            &ssa_ops,
            &vreg_to_addr,
            &shifted_ops,
            &const_add_imm_ops,
            &HashSet::new(),
            &HashMap::new(),
            &HashSet::new(),
            &HashMap::new(),
        );

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
        let mut const_add_imm_ops = HashSet::new();
        const_add_imm_ops.insert(1); // op1 uses AddImm, op2 uses Load

        let result = lower_ssa_ops_with_shift(
            &ssa_ops,
            &vreg_to_addr,
            &shifted_ops,
            &const_add_imm_ops,
            &HashSet::new(),
            &HashMap::new(),
            &HashSet::new(),
            &HashMap::new(),
        );

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
        let const_add_imm_ops = HashSet::new();
        let select_fma_ops = HashSet::new(); // vselect path

        let result = lower_ssa_ops_with_shift(
            &ssa_ops,
            &vreg_to_addr,
            &shifted_ops,
            &const_add_imm_ops,
            &select_fma_ops,
            &HashMap::new(),
            &HashSet::new(),
            &HashMap::new(),
        );

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
        let const_add_imm_ops = HashSet::new();
        let mut select_fma_ops = HashSet::new();
        select_fma_ops.insert(42); // FMA path

        let result = lower_ssa_ops_with_shift(
            &ssa_ops,
            &vreg_to_addr,
            &shifted_ops,
            &const_add_imm_ops,
            &select_fma_ops,
            &HashMap::new(),
            &HashSet::new(),
            &HashMap::new(),
        );

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
        let const_add_imm_ops = HashSet::new();
        let mut select_fma_ops = HashSet::new();
        select_fma_ops.insert(42); // FMA path WITHOUT delta

        let result = lower_ssa_ops_with_shift(
            &ssa_ops,
            &vreg_to_addr,
            &shifted_ops,
            &const_add_imm_ops,
            &select_fma_ops,
            &HashMap::new(),
            &HashSet::new(),
            &HashMap::new(),
        );

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
        let const_add_imm_ops = HashSet::new();
        let mut select_fma_ops = HashSet::new();
        select_fma_ops.insert(42); // FMA path WITHOUT delta

        let result = lower_ssa_ops_with_shift(
            &ssa_ops,
            &vreg_to_addr,
            &shifted_ops,
            &const_add_imm_ops,
            &select_fma_ops,
            &HashMap::new(),
            &HashSet::new(),
            &HashMap::new(),
        );

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
}
