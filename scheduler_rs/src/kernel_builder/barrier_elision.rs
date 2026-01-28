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
}
