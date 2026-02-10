//! CSE (Common Subexpression Elimination) groups for SelectChoice shared SUB.
//!
//! Multiple SelectChoice ops with the same (true_val, false_val) can share one SUB
//! that computes sub_temp = true_val - false_val. This module identifies such groups
//! and provides the structure for the CP-SAT shared-SUB model.

use std::collections::HashMap;

use crate::types::{Opcode, SsaOp, VReg, VRegWidth};

/// Base ID for CSE group sub_temp VRegs. Chosen to avoid collision with normal emission IDs.
const CSE_SUB_TEMP_BASE_ID: u32 = 0x1000000;

/// One CSE group: SelectChoice ops sharing the same (true_val, false_val).
#[derive(Debug, Clone)]
pub struct CseGroup {
    /// true_val operand (srcs[1] of SelectChoice)
    pub true_val: VReg,
    /// false_val operand (srcs[2] of SelectChoice)
    pub false_val: VReg,
    /// Op indices in the ops slice (consumers)
    pub consumer_indices: Vec<usize>,
    /// Op index that defines true_val, if unique
    pub producer_true_idx: Option<usize>,
    /// Op index that defines false_val, if unique
    pub producer_false_idx: Option<usize>,
    /// Allocated VReg for the shared sub_temp (sub_temp = true_val - false_val)
    pub sub_temp_vreg: VReg,
}

/// Result from CP-SAT for one CSE group (Phase A: no sub_shift).
#[derive(Debug, Clone, Default)]
pub struct CseGroupResult {
    /// At least one consumer chose FMA
    pub sub_active: bool,
    /// Cycle when the shared SUB runs (only meaningful when sub_active)
    pub sub_cycle: i32,
}

/// Build CSE groups from SelectChoice ops.
///
/// Groups ops by (true_val, false_val) for SelectChoice ops that have sub_temp_vreg
/// and no delta_vreg (FMA path without precomputed delta). Excludes delta-path SelectChoices.
///
/// Returns groups in deterministic order. Each group gets a unique sub_temp_vreg.
pub fn build_cse_groups(ops: &[SsaOp]) -> Vec<CseGroup> {
    // Build vreg -> op index (defining op)
    let vreg_def: HashMap<VReg, usize> = ops
        .iter()
        .enumerate()
        .filter_map(|(idx, op)| op.dest.map(|d| (d, idx)))
        .collect();

    // Group SelectChoice ops by (true_val, false_val)
    // Only include: Opcode::SelectChoice, sub_temp_vreg.is_some(), delta_vreg.is_none()
    let mut key_to_indices: HashMap<(VReg, VReg), Vec<usize>> = HashMap::new();

    for (idx, op) in ops.iter().enumerate() {
        if op.opcode != Opcode::SelectChoice {
            continue;
        }
        if op.sub_temp_vreg.is_none() || op.delta_vreg.is_some() {
            continue;
        }
        if op.srcs.len() < 3 {
            continue;
        }
        let true_val = match op.srcs[1].as_vreg() {
            Some(v) => v,
            None => continue,
        };
        let false_val = match op.srcs[2].as_vreg() {
            Some(v) => v,
            None => continue,
        };
        key_to_indices
            .entry((true_val, false_val))
            .or_default()
            .push(idx);
    }

    // Convert to Vec<CseGroup> with allocated sub_temp_vreg per group
    let mut groups = Vec::with_capacity(key_to_indices.len());
    for (g, ((true_val, false_val), consumer_indices)) in key_to_indices.into_iter().enumerate() {
        let producer_true_idx = vreg_def.get(&true_val).copied();
        let producer_false_idx = vreg_def.get(&false_val).copied();
        let sub_temp_vreg = VReg {
            id: CSE_SUB_TEMP_BASE_ID + g as u32,
            width: VRegWidth::Vector,
        };
        groups.push(CseGroup {
            true_val,
            false_val,
            consumer_indices,
            producer_true_idx,
            producer_false_idx,
            sub_temp_vreg,
        });
    }
    groups
}
