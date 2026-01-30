//! Barrier elision for CP-SAT and final output.
//!
//! Barriers are scheduling-only pseudo-ops that create hard scheduling boundaries.
//! They are used by the greedy scheduler to constrain cross-round interleaving
//! and reduce peak liveness.
//!
//! However, barriers should be elided (removed) before:
//! - CP-SAT scheduling (which has proper liveness constraints via ILP)
//! - Final output (barriers aren't real hardware instructions)

use std::collections::HashSet;

use crate::types::{Opcode, SsaOp};

/// Remove barrier ops from the ops list.
///
/// Barriers are scheduling-only pseudo-ops. They should be removed:
/// - Before CP-SAT (which has proper liveness constraints)
/// - Before lowering to concrete ops (barriers aren't real instructions)
///
/// This function:
/// 1. Removes all barrier ops from the list
/// 2. Removes barrier op IDs from dependency lists of remaining ops
pub fn elide_barriers(ops: Vec<SsaOp>) -> Vec<SsaOp> {
    // Collect IDs of barrier ops
    let barrier_ids: HashSet<usize> = ops
        .iter()
        .filter(|op| op.opcode == Opcode::Barrier)
        .map(|op| op.id)
        .collect();

    if barrier_ids.is_empty() {
        return ops;
    }

    // Filter out barriers and clean up deps
    ops.into_iter()
        .filter(|op| op.opcode != Opcode::Barrier)
        .map(|mut op| {
            // Remove barrier deps from remaining ops
            op.dep_ids.retain(|id| !barrier_ids.contains(id));
            op
        })
        .collect()
}

/// Check if the ops list contains any barrier ops.
pub fn has_barriers(ops: &[SsaOp]) -> bool {
    ops.iter().any(|op| op.opcode == Opcode::Barrier)
}

/// Count the number of barrier ops in the list.
pub fn count_barriers(ops: &[SsaOp]) -> usize {
    ops.iter().filter(|op| op.opcode == Opcode::Barrier).count()
}

/// Return section boundaries for ops that contain barriers.
///
/// Sections are the ranges of ops between consecutive barriers (same logic as
/// `compute_ssa_dependencies_with_barriers`). Section k = `ops[start..end]` with
/// `(start, end)` from the k-th element of the returned vec; end is exclusive.
/// - Section 0: from start of ops to (but not including) first barrier index.
/// - Section k for k > 0: from the op after barrier k-1 to (but not including) barrier k.
/// - Last section: from the op after the last barrier to end of ops.
///
/// If there are no barriers, returns a single section `[(0, ops.len())]`.
pub fn section_boundaries(ops: &[SsaOp]) -> Vec<(usize, usize)> {
    let barrier_indices: Vec<usize> = ops
        .iter()
        .enumerate()
        .filter(|(_, op)| op.opcode == Opcode::Barrier)
        .map(|(idx, _)| idx)
        .collect();

    if barrier_indices.is_empty() {
        return vec![(0, ops.len())];
    }

    let mut boundaries = Vec::with_capacity(barrier_indices.len() + 1);
    let mut prev = 0usize;
    for &bi in &barrier_indices {
        // Section from prev (inclusive) to barrier (exclusive) = ops in this section before the barrier
        boundaries.push((prev, bi));
        prev = bi + 1;
    }
    boundaries.push((prev, ops.len()));
    boundaries
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Engine, Operand, VReg};
    use std::collections::HashSet;

    fn make_op(id: usize, engine: Engine, opcode: Opcode, deps: Vec<usize>) -> SsaOp {
        let mut op = SsaOp::new(
            id,
            engine,
            opcode,
            Some(VReg::scalar(id as u32)),
            vec![],
        );
        op.dep_ids = deps.into_iter().collect();
        op
    }

    fn make_barrier(id: usize, deps: Vec<usize>) -> SsaOp {
        let mut op = SsaOp::new(id, Engine::Barrier, Opcode::Barrier, None, vec![]);
        op.dep_ids = deps.into_iter().collect();
        op
    }

    #[test]
    fn test_elide_barriers_removes_barriers() {
        let ops = vec![
            make_op(0, Engine::Load, Opcode::Const, vec![]),
            make_op(1, Engine::Load, Opcode::Const, vec![]),
            make_barrier(2, vec![0, 1]),
            make_op(3, Engine::Alu, Opcode::Add, vec![0, 1, 2]),
        ];

        let result = elide_barriers(ops);

        assert_eq!(result.len(), 3, "Should have 3 ops (barrier removed)");
        assert!(!result.iter().any(|op| op.opcode == Opcode::Barrier));
    }

    #[test]
    fn test_elide_barriers_removes_barrier_deps() {
        let ops = vec![
            make_op(0, Engine::Load, Opcode::Const, vec![]),
            make_barrier(1, vec![0]),
            make_op(2, Engine::Alu, Opcode::Add, vec![0, 1]), // depends on barrier
        ];

        let result = elide_barriers(ops);

        assert_eq!(result.len(), 2);
        // Op 2 should no longer depend on barrier (op 1)
        let op2 = result.iter().find(|op| op.id == 2).unwrap();
        assert!(!op2.dep_ids.contains(&1), "Should not depend on elided barrier");
        assert!(op2.dep_ids.contains(&0), "Should still depend on op 0");
    }

    #[test]
    fn test_elide_barriers_no_barriers() {
        let ops = vec![
            make_op(0, Engine::Load, Opcode::Const, vec![]),
            make_op(1, Engine::Alu, Opcode::Add, vec![0]),
        ];

        let result = elide_barriers(ops.clone());

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].id, 0);
        assert_eq!(result[1].id, 1);
    }

    #[test]
    fn test_has_barriers() {
        let without = vec![
            make_op(0, Engine::Load, Opcode::Const, vec![]),
        ];
        let with = vec![
            make_op(0, Engine::Load, Opcode::Const, vec![]),
            make_barrier(1, vec![0]),
        ];

        assert!(!has_barriers(&without));
        assert!(has_barriers(&with));
    }

    #[test]
    fn test_count_barriers() {
        let ops = vec![
            make_op(0, Engine::Load, Opcode::Const, vec![]),
            make_barrier(1, vec![0]),
            make_op(2, Engine::Alu, Opcode::Add, vec![1]),
            make_barrier(3, vec![2]),
            make_op(4, Engine::Alu, Opcode::Add, vec![3]),
        ];

        assert_eq!(count_barriers(&ops), 2);
    }

    #[test]
    fn test_section_boundaries() {
        // ops: [op0, op1, barrier2, op3, barrier4, op5]
        let ops = vec![
            make_op(0, Engine::Load, Opcode::Const, vec![]),
            make_op(1, Engine::Load, Opcode::Const, vec![]),
            make_barrier(2, vec![0, 1]),
            make_op(3, Engine::Alu, Opcode::Add, vec![0, 1, 2]),
            make_barrier(4, vec![3]),
            make_op(5, Engine::Alu, Opcode::Add, vec![3, 4]),
        ];
        let boundaries = section_boundaries(&ops);
        assert_eq!(boundaries.len(), 3);
        assert_eq!(boundaries[0], (0, 2)); // section 0: indices 0, 1 (before first barrier)
        assert_eq!(boundaries[1], (3, 4)); // section 1: index 3 (between barrier 2 and 4)
        assert_eq!(boundaries[2], (5, 6)); // section 2: index 5 (after last barrier)
    }

    #[test]
    fn test_section_boundaries_no_barriers() {
        let ops = vec![
            make_op(0, Engine::Load, Opcode::Const, vec![]),
            make_op(1, Engine::Alu, Opcode::Add, vec![0]),
        ];
        let boundaries = section_boundaries(&ops);
        assert_eq!(boundaries.len(), 1);
        assert_eq!(boundaries[0], (0, 2));
    }
}
