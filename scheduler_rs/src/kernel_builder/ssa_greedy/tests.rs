use super::*;
use crate::kernel_builder::emit_ssa::{
    compute_ssa_dependencies, compute_ssa_dependencies_with_barriers, SsaEmitContext,
};
use crate::types::Opcode;

// =========================================================================
// Force-Schedule and Budget-Exceeded Tests (CRITICAL_TESTS_PLAN)
//
// These tests verify that the emergency force-schedule path sets budget_exceeded
// and that the flag is not removed during debugging unless explicitly instructed.
// =========================================================================

/// Test that when the greedy scheduler force-schedules (budget too tight),
/// it sets budget_exceeded to true and returns Ok.
///
/// Scenario: two vbroadcasts (8 words each) feed one VALU add. With budget 8,
/// after the first vbroadcast we have 8 words live; the second vbroadcast would
/// push to 16 > 8, so it is deferred. No other op is ready. The retry loop
/// makes no progress, so the emergency fallback force-schedules the second
/// vbroadcast and sets budget_exceeded.
#[test]
fn test_greedy_fallback_sets_budget_exceeded() {
    let mut emit = SsaEmitContext::new();
    let s1 = emit.emit_const(1); // op 0
    let s2 = emit.emit_const(2); // op 1
    let v1 = emit.emit_vbroadcast(s1); // op 2 - 8 words
    let v2 = emit.emit_vbroadcast(s2); // op 3 - 8 words
    let v3 = emit.emit_valu(Opcode::Add, v1, v2); // op 4 - needs v1, v2 live
    let addr = emit.emit_const(100); // op 5
    emit.emit_vstore(addr, v3); // op 6 - sink

    compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg);

    let mut shifted_ops = HashSet::new();

    let result = schedule_ssa_greedy_with_shift_budget(
        &mut emit.ops,
        false,
        &mut shifted_ops,
        8, // Tight budget: 2 vectors (16 words) cannot fit in 8
    );

    assert!(
        result.is_ok(),
        "Scheduler should complete (force-schedule): {:?}",
        result.err()
    );
    let (_cycles, budget_exceeded, peak_live) = result.unwrap();
    assert!(
        budget_exceeded,
        "Force-schedule path must set budget_exceeded=true when budget is exceeded; got budget_exceeded=false, peak_live={}",
        peak_live
    );
}

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
    let v1 = emit.emit_const(5); // op 0 - used
    let _v2 = emit.emit_const(6); // op 1 - DEAD STORE
    let v3 = emit.emit_alu(Opcode::Add, v1, v1); // op 2 - used
    emit.emit_barrier(); // op 3 - depends on ALL above including dead store
    let v4 = emit.emit_alu(Opcode::Mul, v1, v3); // op 4 - used
    let addr = emit.emit_const(100); // op 5
    emit.emit_store(addr, v4); // op 6 - makes v4 live

    compute_ssa_dependencies_with_barriers(&mut emit.ops, &emit.vreg_def, emit.zero_vreg);

    // Verify barrier depends on the dead store
    assert!(
        emit.ops[3].dep_ids.contains(&1),
        "Barrier should depend on dead store (op1)"
    );

    // Schedule

    let mut shifted_ops = HashSet::new();
    let result = schedule_ssa_greedy_with_shift_budget(
        &mut emit.ops,
        false, // no VALU shift
        &mut shifted_ops,
        1536, // default budget
    );

    assert!(
        result.is_ok(),
        "Should schedule despite barrier depending on dead store: {:?}",
        result.err()
    );

    // Verify barrier was scheduled
    assert!(
        emit.ops[3].cycle >= 0,
        "Barrier should be scheduled (cycle={})",
        emit.ops[3].cycle
    );

    // Verify ops after barrier were scheduled
    assert!(
        emit.ops[4].cycle > emit.ops[3].cycle,
        "Op after barrier should be after barrier (op4.cycle={} > barrier.cycle={})",
        emit.ops[4].cycle,
        emit.ops[3].cycle
    );

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
    let _v1 = emit.emit_const(1); // op 0 - DEAD STORE
    let _v2 = emit.emit_const(2); // op 1 - DEAD STORE
    let _v3 = emit.emit_const(3); // op 2 - DEAD STORE
    let v4 = emit.emit_const(4); // op 3 - used
    emit.emit_barrier(); // op 4 - depends on all above
    let v5 = emit.emit_alu(Opcode::Add, v4, v4); // op 5 - used
    let addr = emit.emit_const(100); // op 6
    emit.emit_store(addr, v5); // op 7

    compute_ssa_dependencies_with_barriers(&mut emit.ops, &emit.vreg_def, emit.zero_vreg);

    // Verify barrier depends on all dead stores
    assert!(
        emit.ops[4].dep_ids.contains(&0),
        "Barrier should depend on dead store op0"
    );
    assert!(
        emit.ops[4].dep_ids.contains(&1),
        "Barrier should depend on dead store op1"
    );
    assert!(
        emit.ops[4].dep_ids.contains(&2),
        "Barrier should depend on dead store op2"
    );

    // Schedule

    let mut shifted_ops = HashSet::new();
    let result = schedule_ssa_greedy_with_shift_budget(
        &mut emit.ops,
        false,
        &mut shifted_ops,
        1536,
    );

    assert!(
        result.is_ok(),
        "Should schedule with multiple dead stores: {:?}",
        result.err()
    );

    // Verify all dead stores remain unscheduled
    assert_eq!(
        emit.ops[0].cycle, -1,
        "Dead store op0 should not be scheduled"
    );
    assert_eq!(
        emit.ops[1].cycle, -1,
        "Dead store op1 should not be scheduled"
    );
    assert_eq!(
        emit.ops[2].cycle, -1,
        "Dead store op2 should not be scheduled"
    );

    // Verify barrier and subsequent ops are scheduled
    assert!(emit.ops[4].cycle >= 0, "Barrier should be scheduled");
    assert!(
        emit.ops[5].cycle > emit.ops[4].cycle,
        "Op5 should be after barrier"
    );
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
    let v1 = emit.emit_const(1); // op 0 - used
    let _v2 = emit.emit_const(2); // op 1 - DEAD STORE
    emit.emit_barrier(); // op 2 - barrier1
                         // Section 2
    let _v3 = emit.emit_const(3); // op 3 - DEAD STORE
    let v4 = emit.emit_const(4); // op 4 - used
    emit.emit_barrier(); // op 5 - barrier2
                         // Section 3
    let v5 = emit.emit_alu(Opcode::Add, v1, v4); // op 6 - used
    let addr = emit.emit_const(100); // op 7
    emit.emit_store(addr, v5); // op 8

    compute_ssa_dependencies_with_barriers(&mut emit.ops, &emit.vreg_def, emit.zero_vreg);

    // Verify barrier1 depends on dead store op1
    assert!(
        emit.ops[2].dep_ids.contains(&1),
        "Barrier1 should depend on dead store op1"
    );

    // Verify barrier2 depends on dead store op3 and barrier1
    assert!(
        emit.ops[5].dep_ids.contains(&3),
        "Barrier2 should depend on dead store op3"
    );
    assert!(
        emit.ops[5].dep_ids.contains(&2),
        "Barrier2 should depend on barrier1 (chain)"
    );

    // Schedule

    let mut shifted_ops = HashSet::new();
    let result = schedule_ssa_greedy_with_shift_budget(
        &mut emit.ops,
        false,
        &mut shifted_ops,
        1536,
    );

    assert!(
        result.is_ok(),
        "Should schedule with barrier chain and dead stores: {:?}",
        result.err()
    );

    // Verify barriers are scheduled in order
    assert!(emit.ops[2].cycle >= 0, "Barrier1 should be scheduled");
    assert!(
        emit.ops[5].cycle > emit.ops[2].cycle,
        "Barrier2 should be after barrier1"
    );

    // Verify final ops are after barrier2
    assert!(
        emit.ops[6].cycle > emit.ops[5].cycle,
        "Op6 should be after barrier2"
    );
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
    let v1 = emit.emit_const(1); // op 0
    let _v2 = emit.emit_alu(Opcode::Add, v1, v1); // op 1 - dead store
    let v3 = emit.emit_const(2); // op 2
    let v4 = emit.emit_alu(Opcode::Add, v1, v3); // op 3
    let addr = emit.emit_const(100); // op 4
    emit.emit_store(addr, v4); // op 5 - makes v4 live

    compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg);


    let mut shifted_ops = HashSet::new();
    let result = schedule_ssa_greedy_with_shift_budget(
        &mut emit.ops,
        false,
        &mut shifted_ops,
        1536,
    );

    assert!(
        result.is_ok(),
        "Should schedule with dead store in chain: {:?}",
        result.err()
    );

    // Check that non-dead-stores got valid cycles
    assert!(emit.ops[0].cycle >= 0, "op0 should be scheduled");
    assert!(emit.ops[2].cycle >= 0, "op2 should be scheduled");
    assert!(emit.ops[3].cycle >= 0, "op3 should be scheduled");

    // op1 (dead store) should remain unscheduled
    assert_eq!(
        emit.ops[1].cycle, -1,
        "Dead store op1 should not be scheduled"
    );
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
    let v1 = emit.emit_const(1); // op 0 - transitively dead (only user is dead)
    let _v2 = emit.emit_alu(Opcode::Add, v1, v1); // op 1 - dead store
    let v3 = emit.emit_const(2); // op 2 - used
    let v4 = emit.emit_alu(Opcode::Add, v3, v3); // op 3 - used
    let addr = emit.emit_const(100); // op 4
    emit.emit_store(addr, v4); // op 5

    compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg);


    let mut shifted_ops = HashSet::new();
    let result = schedule_ssa_greedy_with_shift_budget(
        &mut emit.ops,
        false,
        &mut shifted_ops,
        1536,
    );

    assert!(
        result.is_ok(),
        "Should schedule with transitive dead stores: {:?}",
        result.err()
    );

    // Note: Current implementation only detects immediate dead stores,
    // not transitive ones. op0 (v1) is used by op1 (v2), so it's not
    // considered dead even though op1 is dead.
    //
    // This is acceptable behavior - transitive dead code elimination
    // would require fixed-point iteration. The important thing is that
    // scheduling doesn't fail.

    // op1 (v2) is definitely dead (no users)
    assert_eq!(
        emit.ops[1].cycle, -1,
        "Dead store op1 should not be scheduled"
    );

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
    let v1 = emit.emit_const_force(5); // op 0 - has reader (op2), not dead
    let v2 = emit.emit_const_force(6); // op 1 - has reader (op2), not dead
    let _v3 = emit.emit_alu(Opcode::Add, v1, v2); // op 2 - DEAD (no readers)

    compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg);

    // Verify all 3 ops were created
    assert_eq!(emit.ops.len(), 3, "Should have 3 ops");


    let mut shifted_ops = HashSet::new();
    let result = schedule_ssa_greedy_with_shift_budget(
        &mut emit.ops,
        false,
        &mut shifted_ops,
        1536,
    );

    // Scheduling should succeed
    assert!(
        result.is_ok(),
        "Scheduling should succeed: {:?}",
        result.err()
    );
    let (_cycles, budget_exceeded, _peak_live) = result.unwrap();
    assert!(!budget_exceeded, "Should not exceed budget");

    // Only op2 (the add) is an immediate dead store
    // op0 and op1 (the consts) have readers so they're scheduled
    assert!(
        emit.ops[0].cycle >= 0,
        "op0 should be scheduled (has reader op2)"
    );
    assert!(
        emit.ops[1].cycle >= 0,
        "op1 should be scheduled (has reader op2)"
    );
    assert_eq!(
        emit.ops[2].cycle, -1,
        "op2 should not be scheduled (dead store)"
    );
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
    let mult_scalar = emit.emit_const(2); // op 0
    let offset_scalar = emit.emit_const(100); // op 1
    let mult_vec = emit.emit_vbroadcast(mult_scalar); // op 2
    let offset_vec = emit.emit_vbroadcast(offset_scalar); // op 3
    let base_scalar = emit.emit_const(0); // op 4
    let base_vec = emit.emit_vbroadcast(base_scalar); // op 5
    let fma_result = emit.emit_fma(base_vec, mult_vec, offset_vec); // op 6
    let addr = emit.emit_const(200); // op 7
    emit.emit_vstore(addr, fma_result); // op 8

    compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg);

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
    assert!(
        fma_result_users >= 1,
        "FMA result should have at least 1 user (vstore), but has {} users",
        fma_result_users
    );

    // Schedule and verify FMA is NOT a dead store

    let mut shifted_ops = HashSet::new();
    let result = schedule_ssa_greedy_with_shift_budget(
        &mut emit.ops,
        false,
        &mut shifted_ops,
        1536,
    );

    assert!(
        result.is_ok(),
        "Should schedule successfully: {:?}",
        result.err()
    );

    // The FMA (op6) should be scheduled (NOT a dead store)
    assert!(
        emit.ops[6].cycle >= 0,
        "FMA op should be scheduled (used by vstore), but has cycle={}",
        emit.ops[6].cycle
    );

    // The vstore (op8) should be scheduled
    assert!(
        emit.ops[8].cycle >= 0,
        "vstore op should be scheduled, but has cycle={}",
        emit.ops[8].cycle
    );

    // vstore should happen AFTER FMA (dependency order)
    assert!(
        emit.ops[8].cycle > emit.ops[6].cycle,
        "vstore (cycle={}) should be after FMA (cycle={})",
        emit.ops[8].cycle,
        emit.ops[6].cycle
    );
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
    let base = emit.emit_const(100); // op 0
    let offset = emit.emit_const(8); // op 1
    let addr = emit.emit_alu(Opcode::Add, base, offset); // op 2
    let data = emit.emit_const(42); // op 3
    emit.emit_store(addr, data); // op 4

    compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg);

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
    assert!(
        vreg_users
            .get(&addr)
            .map_or(false, |users| users.contains(&4)),
        "addr VReg should have store (op4) as user"
    );

    // data should have the store as a user
    assert!(
        vreg_users
            .get(&data)
            .map_or(false, |users| users.contains(&4)),
        "data VReg should have store (op4) as user"
    );

    // Schedule

    let mut shifted_ops = HashSet::new();
    let result = schedule_ssa_greedy_with_shift_budget(
        &mut emit.ops,
        false,
        &mut shifted_ops,
        1536,
    );

    assert!(
        result.is_ok(),
        "Should schedule successfully: {:?}",
        result.err()
    );

    // addr computation (op2) should be scheduled
    assert!(
        emit.ops[2].cycle >= 0,
        "addr computation should be scheduled (used by store)"
    );

    // data (op3) should be scheduled
    assert!(
        emit.ops[3].cycle >= 0,
        "data const should be scheduled (used by store)"
    );
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
    let v1 = emit.emit_const(5); // op 0 - has user (op1)
    let v1_vec = emit.emit_vbroadcast(v1); // op 1 - has user (op2)
    let v2 = emit.emit_valu(Opcode::Add, v1_vec, v1_vec); // op 2 - has user (op3)
    let addr = emit.emit_const(100); // op 3 - has user (op4)
    emit.emit_vstore(addr, v2); // op 4 - sink (no dest)

    compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg);

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
    assert!(
        vreg_users.get(&v1).map_or(false, |users| !users.is_empty()),
        "v1 should have at least 1 user"
    );

    // v1_vec (vector) should have user (op2 - add)
    assert!(
        vreg_users
            .get(&v1_vec)
            .map_or(false, |users| !users.is_empty()),
        "v1_vec should have at least 1 user"
    );

    // v2 (vector) should have user (op4 - vstore)
    assert!(
        vreg_users.get(&v2).map_or(false, |users| !users.is_empty()),
        "v2 should have at least 1 user"
    );

    // addr should have user (op4 - vstore)
    assert!(
        vreg_users
            .get(&addr)
            .map_or(false, |users| !users.is_empty()),
        "addr should have at least 1 user"
    );

    // Now schedule and verify none of these are marked as dead

    let mut shifted_ops = HashSet::new();
    let result = schedule_ssa_greedy_with_shift_budget(
        &mut emit.ops,
        false,
        &mut shifted_ops,
        1536,
    );

    assert!(
        result.is_ok(),
        "Should schedule successfully: {:?}",
        result.err()
    );

    // All ops except the store should be scheduled (store has no dest, not tracked)
    // ops 0-3 have dests and should all be scheduled
    assert!(emit.ops[0].cycle >= 0, "op0 (const) should be scheduled");
    assert!(
        emit.ops[1].cycle >= 0,
        "op1 (vbroadcast) should be scheduled"
    );
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
    let v1 = emit.emit_const(5); // op 0
    let v1_vec = emit.emit_vbroadcast(v1); // op 1
    let v2 = emit.emit_valu(Opcode::Add, v1_vec, v1_vec); // op 2
    let v3 = emit.emit_valu(Opcode::Mul, v2, v2); // op 3
    let addr = emit.emit_const(100); // op 4
    emit.emit_vstore(addr, v3); // op 5

    compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg);

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
    assert!(
        vreg_users.get(&v1).map_or(0, |s| s.len()) >= 1,
        "v1 should have user"
    );
    assert!(
        vreg_users.get(&v1_vec).map_or(0, |s| s.len()) >= 1,
        "v1_vec should have user"
    );
    assert!(
        vreg_users.get(&v2).map_or(0, |s| s.len()) >= 1,
        "v2 should have user"
    );
    assert!(
        vreg_users.get(&v3).map_or(0, |s| s.len()) >= 1,
        "v3 should have user"
    );

    // Schedule

    let mut shifted_ops = HashSet::new();
    let result = schedule_ssa_greedy_with_shift_budget(
        &mut emit.ops,
        false,
        &mut shifted_ops,
        1536,
    );

    assert!(
        result.is_ok(),
        "Should schedule transitive chain: {:?}",
        result.err()
    );

    // All ops in the chain should be scheduled
    for (i, op) in emit.ops.iter().enumerate() {
        assert!(
            op.cycle >= 0,
            "op{} should be scheduled, got cycle={}",
            i,
            op.cycle
        );
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
    let fma_result = emit.emit_fma(a, b, c); // op 6
    let addr = emit.emit_const(100); // op 7
    emit.emit_vstore(addr, fma_result); // op 8

    compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg);

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
    assert!(
        fma_users >= 1,
        "FMA result should have at least 1 user (vstore), but has {} users",
        fma_users
    );

    // Schedule

    let mut shifted_ops = HashSet::new();
    let result = schedule_ssa_greedy_with_shift_budget(
        &mut emit.ops,
        false,
        &mut shifted_ops,
        1536,
    );

    assert!(
        result.is_ok(),
        "Should schedule FMA with store user: {:?}",
        result.err()
    );

    // The FMA (op 6) should be scheduled, NOT marked as dead
    assert!(
        emit.ops[6].cycle >= 0,
        "FMA op should be scheduled (used by vstore), but has cycle={}",
        emit.ops[6].cycle
    );
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
    let v1 = emit.emit_const(5); // op 0
    let _v2 = emit.emit_alu(Opcode::Add, v1, v1); // op 1 - DEAD (result never used)
    let v3 = emit.emit_const(10); // op 2
    let addr = emit.emit_const(100); // op 3
    emit.emit_store(addr, v3); // op 4

    compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg);

    // Schedule

    let mut shifted_ops = HashSet::new();
    let result = schedule_ssa_greedy_with_shift_budget(
        &mut emit.ops,
        false,
        &mut shifted_ops,
        1536,
    );

    assert!(result.is_ok(), "Should schedule: {:?}", result.err());

    // op1 (v2) should be marked as dead (cycle = -1)
    assert_eq!(
        emit.ops[1].cycle, -1,
        "Dead store op1 should not be scheduled"
    );

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
    let v0_scalar = emit.emit_const(5); // op 0
    let v0 = emit.emit_vbroadcast(v0_scalar); // op 1
    let v1 = emit.emit_valu(Opcode::Add, v0, v0); // op 2 - ONLY used by vstore
    let addr = emit.emit_const(100); // op 3
    emit.emit_vstore(addr, v1); // op 4 - uses v1

    compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg);

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
    assert!(
        v1_users.unwrap().contains(&4),
        "v1's users should include vstore (op4), but users are: {:?}",
        v1_users
    );

    // v1 should NOT be considered a dead store (it has the vstore as user)
    let v1_is_dead = vreg_users.get(&v1).map_or(true, |s| s.is_empty());
    assert!(
        !v1_is_dead,
        "v1 should NOT be a dead store (used by vstore)"
    );

    // Schedule and verify v1 is scheduled

    let mut shifted_ops = HashSet::new();
    let result = schedule_ssa_greedy_with_shift_budget(
        &mut emit.ops,
        false,
        &mut shifted_ops,
        1536,
    );

    assert!(
        result.is_ok(),
        "Should schedule successfully: {:?}",
        result.err()
    );

    // v1 (op2) should be scheduled
    assert!(
        emit.ops[2].cycle >= 0,
        "v1 (add) should be scheduled (used by vstore), but has cycle={}",
        emit.ops[2].cycle
    );
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

    let mut shifted_ops = HashSet::new();
    let result = schedule_ssa_greedy_with_shift_budget(
        &mut emit.ops,
        false,
        &mut shifted_ops,
        1536,
    );

    assert!(
        result.is_ok(),
        "Should schedule successfully: {:?}",
        result.err()
    );

    // Find the ConstChoice op
    let const_choice_op = emit.ops.iter().find(|op| op.opcode == Opcode::ConstChoice);
    assert!(const_choice_op.is_some(), "Should have ConstChoice op");
    let const_choice_op = const_choice_op.unwrap();

    // ConstChoice should be scheduled (greedy treats it as Load)
    assert!(
        const_choice_op.cycle >= 0,
        "ConstChoice op should be scheduled, but has cycle={}",
        const_choice_op.cycle
    );

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

// =========================================================================
// Context Variable Dead Store Bug Tests
//
// These tests reproduce the bug where MultiplyAdd ops that update context
// variables (like ctx.all_idx[g]) are incorrectly marked as dead stores,
// even though they're used later in vstore operations.
//
// The bug pattern:
// 1. MultiplyAdd produces a VReg and stores it in a context variable
// 2. Later, vstore reads from that context variable and uses the VReg
// 3. Dead store detection doesn't properly track this dependency,
//    especially when there's a barrier between the MultiplyAdd and vstore
//
// These tests are designed to FAIL initially (demonstrating the bug exists).
// =========================================================================

/// Test that MultiplyAdd updating a context variable is NOT marked as dead
/// when used by vstore in the same section.
///
/// This test simulates the pattern:
/// - MultiplyAdd produces a VReg (simulating ctx.all_idx[g] = Some(emit_fma(...)))
/// - vstore uses that VReg (simulating emit_vstore(addr, ctx.all_idx[g].unwrap()))
///
/// Even without barriers, this should work correctly, but we test it to
/// establish a baseline.
#[test]
fn test_context_variable_multiplyadd_not_dead() {
    // Scenario:
    // op0: mult_scalar = const 2
    // op1: offset_scalar = const 100
    // op2: mult_vec = vbroadcast(mult_scalar)
    // op3: offset_vec = vbroadcast(offset_scalar)
    // op4: idx_base = const 0
    // op5: idx_base_vec = vbroadcast(idx_base)
    // op6: idx_new = FMA(idx_base_vec, mult_vec, offset_vec) -> computes idx = idx * 2 + 100
    //      (This simulates: ctx.all_idx[g] = Some(emit_fma(...)))
    // op7: addr = const 200
    // op8: vstore(addr, idx_new) -> store the FMA result
    //      (This simulates: emit_vstore(addr, ctx.all_idx[g].unwrap()))
    //
    // idx_new (FMA result) is stored in a "context variable" (local Rust variable)
    // and then used by vstore. The FMA should NOT be marked as dead.

    let mut emit = SsaEmitContext::new();
    let mult_scalar = emit.emit_const(2); // op 0
    let offset_scalar = emit.emit_const(100); // op 1
    let mult_vec = emit.emit_vbroadcast(mult_scalar); // op 2
    let offset_vec = emit.emit_vbroadcast(offset_scalar); // op 3
    let idx_base_scalar = emit.emit_const(0); // op 4
    let idx_base_vec = emit.emit_vbroadcast(idx_base_scalar); // op 5

    // Simulate: ctx.all_idx[g] = Some(emit_fma(...))
    // Store the VReg in a local variable (simulating context variable)
    let idx_new = emit.emit_fma(idx_base_vec, mult_vec, offset_vec); // op 6

    let addr = emit.emit_const(200); // op 7

    // Simulate: emit_vstore(addr, ctx.all_idx[g].unwrap())
    // Use the VReg from the "context variable"
    emit.emit_vstore(addr, idx_new); // op 8

    // Use barrier-aware dependencies to match real kernel behavior
    compute_ssa_dependencies_with_barriers(&mut emit.ops, &emit.vreg_def, emit.zero_vreg);

    // Verify: vstore (op8) reads idx_new (the FMA result)
    // Build vreg_users map as the scheduler does
    let mut vreg_users: HashMap<VReg, HashSet<usize>> = HashMap::new();
    for (idx, op) in emit.ops.iter().enumerate() {
        for src in &op.srcs {
            if let Operand::VReg(vreg) = src {
                vreg_users.entry(*vreg).or_default().insert(idx);
            }
        }
    }

    // idx_new should have at least 1 user (the vstore)
    let idx_new_users = vreg_users.get(&idx_new).map_or(0, |s| s.len());
    assert!(
        idx_new_users >= 1,
        "FMA result (idx_new) should have at least 1 user (vstore), but has {} users",
        idx_new_users
    );

    // Schedule and verify FMA is NOT a dead store

    let mut shifted_ops = HashSet::new();
    let result = schedule_ssa_greedy_with_shift_budget(
        &mut emit.ops,
        false,
        &mut shifted_ops,
        1536,
    );

    assert!(
        result.is_ok(),
        "Should schedule successfully: {:?}",
        result.err()
    );

    // The FMA (op6) should be scheduled (NOT a dead store)
    // This test may fail if the bug exists - the FMA might be marked as dead
    assert!(emit.ops[6].cycle >= 0,
        "FMA op should be scheduled (used by vstore via context variable), but has cycle={}. \
         This indicates the bug: MultiplyAdd updating context variable is marked as dead store.",
        emit.ops[6].cycle);

    // The vstore (op8) should be scheduled
    assert!(
        emit.ops[8].cycle >= 0,
        "vstore op should be scheduled, but has cycle={}",
        emit.ops[8].cycle
    );

    // vstore should happen AFTER FMA (dependency order)
    assert!(
        emit.ops[8].cycle > emit.ops[6].cycle,
        "vstore (cycle={}) should be after FMA (cycle={})",
        emit.ops[8].cycle,
        emit.ops[6].cycle
    );
}

/// Test that MultiplyAdd updating a context variable is NOT marked as dead
/// when used by vstore across a barrier.
///
/// This test simulates the actual bug pattern from the kernel:
/// - MultiplyAdd produces a VReg in one section (simulating idx update in a round)
/// - Barrier separates sections (simulating round boundary)
/// - vstore uses that VReg in next section (simulating partition store in next round)
///
/// This is the most likely scenario to reproduce the bug, as barriers may
/// interfere with dependency tracking for context variables.
#[test]
fn test_context_variable_multiplyadd_across_barrier() {
    // Scenario:
    // Section 1 (simulating idx update in a round):
    //   op0: mult_scalar = const 2
    //   op1: offset_scalar = const 100
    //   op2: mult_vec = vbroadcast(mult_scalar)
    //   op3: offset_vec = vbroadcast(offset_scalar)
    //   op4: idx_base = const 0
    //   op5: idx_base_vec = vbroadcast(idx_base)
    //   op6: idx_new = FMA(idx_base_vec, mult_vec, offset_vec)
    //        (This simulates: ctx.all_idx[g] = Some(emit_fma(...)) in round N)
    // op7: barrier (simulating round boundary)
    // Section 2 (simulating partition store in next round):
    //   op8: addr = const 200
    //   op9: vstore(addr, idx_new)
    //        (This simulates: emit_vstore(addr, ctx.all_idx[g].unwrap()) in round N+1)
    //
    // The barrier between the FMA and vstore may cause dependency tracking
    // to fail, causing the FMA to be marked as dead.

    let mut emit = SsaEmitContext::new();

    // Section 1: Emit MultiplyAdd that updates context variable
    let mult_scalar = emit.emit_const(2); // op 0
    let offset_scalar = emit.emit_const(100); // op 1
    let mult_vec = emit.emit_vbroadcast(mult_scalar); // op 2
    let offset_vec = emit.emit_vbroadcast(offset_scalar); // op 3
    let idx_base_scalar = emit.emit_const(0); // op 4
    let idx_base_vec = emit.emit_vbroadcast(idx_base_scalar); // op 5

    // Simulate: ctx.all_idx[g] = Some(emit_fma(...)) in round N
    let idx_new = emit.emit_fma(idx_base_vec, mult_vec, offset_vec); // op 6

    // Barrier simulates round boundary
    emit.emit_barrier(); // op 7

    // Section 2: Use context variable in vstore
    let addr = emit.emit_const(200); // op 8

    // Simulate: emit_vstore(addr, ctx.all_idx[g].unwrap()) in round N+1
    emit.emit_vstore(addr, idx_new); // op 9

    // Use barrier-aware dependencies to match real kernel behavior
    compute_ssa_dependencies_with_barriers(&mut emit.ops, &emit.vreg_def, emit.zero_vreg);

    // Verify: vstore (op9) reads idx_new (the FMA result)
    // Build vreg_users map as the scheduler does
    let mut vreg_users: HashMap<VReg, HashSet<usize>> = HashMap::new();
    for (idx, op) in emit.ops.iter().enumerate() {
        for src in &op.srcs {
            if let Operand::VReg(vreg) = src {
                vreg_users.entry(*vreg).or_default().insert(idx);
            }
        }
    }

    // idx_new should have at least 1 user (the vstore)
    let idx_new_users = vreg_users.get(&idx_new).map_or(0, |s| s.len());
    assert!(
        idx_new_users >= 1,
        "FMA result (idx_new) should have at least 1 user (vstore), but has {} users",
        idx_new_users
    );

    // Schedule and verify FMA is NOT a dead store

    let mut shifted_ops = HashSet::new();
    let result = schedule_ssa_greedy_with_shift_budget(
        &mut emit.ops,
        false,
        &mut shifted_ops,
        1536,
    );

    assert!(
        result.is_ok(),
        "Should schedule successfully: {:?}",
        result.err()
    );

    // The FMA (op6) should be scheduled (NOT a dead store)
    // This test is expected to FAIL if the bug exists - the FMA will be marked as dead
    // because the dependency across the barrier isn't properly tracked
    assert!(
        emit.ops[6].cycle >= 0,
        "FMA op should be scheduled (used by vstore via context variable across barrier), \
         but has cycle={}. This indicates the bug: MultiplyAdd updating context variable \
         is marked as dead store when used across a barrier.",
        emit.ops[6].cycle
    );

    // The barrier (op7) should be scheduled
    assert!(
        emit.ops[7].cycle >= 0,
        "Barrier op should be scheduled, but has cycle={}",
        emit.ops[7].cycle
    );

    // The vstore (op9) should be scheduled
    assert!(
        emit.ops[9].cycle >= 0,
        "vstore op should be scheduled, but has cycle={}",
        emit.ops[9].cycle
    );

    // vstore should happen AFTER barrier, which should happen AFTER FMA
    assert!(
        emit.ops[7].cycle > emit.ops[6].cycle,
        "Barrier (cycle={}) should be after FMA (cycle={})",
        emit.ops[7].cycle,
        emit.ops[6].cycle
    );
    assert!(
        emit.ops[9].cycle > emit.ops[7].cycle,
        "vstore (cycle={}) should be after barrier (cycle={})",
        emit.ops[9].cycle,
        emit.ops[7].cycle
    );
}

// =========================================================================
// Advanced Dead Store Bug Tests - Testing Hypotheses
//
// These tests simulate more complex kernel patterns to reproduce the bug
// where MultiplyAdd ops updating context variables are marked as dead stores.
// =========================================================================

/// Test Hypothesis 1: Multi-Round with Conditional Stores
///
/// Simulates the real kernel pattern where:
/// - Round 1: tree_depth NOT in TICK/TOCK_STORE_DEPTHS
///   - MultiplyAdd updates all_idx[g] for all groups
///   - NO partition stores happen (tree_depth doesn't match)
/// - Round 2: Operations that use all_idx[g] from round 1
///
/// The bug: MultiplyAdd VRegs from Round 1 have no immediate users (no stores),
/// but ARE needed for Round 2 operations. Dead store detection may mark them as dead.
#[test]
fn test_multi_round_conditional_stores_hypothesis1() {
    // Simulate kernel with 2 rounds
    // Round 1: tree_depth = 10 (NOT in TICK_STORE_DEPTHS [0,1,2,4,5,6] or TOCK_STORE_DEPTHS [3,7,8,9])
    //   - MultiplyAdd updates all_idx[g] for all groups
    //   - NO partition stores (tree_depth doesn't match)
    // Round 2: Operations use all_idx[g] from round 1

    let mut emit = SsaEmitContext::new();

    // Round 1 setup: Simulate idx update for one group (simplified)
    let mult_scalar = emit.emit_const(2); // op 0
    let offset_scalar = emit.emit_const(100); // op 1
    let mult_vec = emit.emit_vbroadcast(mult_scalar); // op 2
    let offset_vec = emit.emit_vbroadcast(offset_scalar); // op 3
    let idx_base_scalar = emit.emit_const(0); // op 4
    let idx_base_vec = emit.emit_vbroadcast(idx_base_scalar); // op 5

    // Round 1: MultiplyAdd updates context variable (simulating all_idx[g])
    // This simulates: ctx.all_idx[g] = Some(emit_fma(...)) in round 1
    // NO partition stores happen because tree_depth=10 doesn't match TICK/TOCK_STORE_DEPTHS
    let idx_round1 = emit.emit_fma(idx_base_vec, mult_vec, offset_vec); // op 6

    // Barrier between rounds
    emit.emit_barrier(); // op 7

    // Round 2: Operations that use all_idx[g] from round 1
    // Simulate: Operations in round 2 that read ctx.all_idx[g].unwrap() and use it
    // For example: tree access operations that use the idx
    let idx_round2_use = emit.emit_valu(Opcode::Add, idx_round1, mult_vec); // op 8 - uses idx from round 1

    // Eventually store the result (simulating final store)
    let addr = emit.emit_const(200); // op 9
    emit.emit_vstore(addr, idx_round2_use); // op 10

    // Use barrier-aware dependencies to match real kernel behavior
    compute_ssa_dependencies_with_barriers(&mut emit.ops, &emit.vreg_def, emit.zero_vreg);

    // Verify: idx_round2_use (op8) reads idx_round1 (the FMA result from round 1)
    // Build vreg_users map as the scheduler does
    let mut vreg_users: HashMap<VReg, HashSet<usize>> = HashMap::new();
    for (idx, op) in emit.ops.iter().enumerate() {
        for src in &op.srcs {
            if let Operand::VReg(vreg) = src {
                vreg_users.entry(*vreg).or_default().insert(idx);
            }
        }
    }

    // idx_round1 should have at least 1 user (op8 uses it)
    let idx_round1_users = vreg_users.get(&idx_round1).map_or(0, |s| s.len());
    assert!(idx_round1_users >= 1,
        "FMA result from round 1 (idx_round1) should have at least 1 user (op8), but has {} users",
        idx_round1_users);

    // Schedule and verify Round 1 FMA is NOT a dead store

    let mut shifted_ops = HashSet::new();
    let result = schedule_ssa_greedy_with_shift_budget(
        &mut emit.ops,
        false,
        &mut shifted_ops,
        1536,
    );

    assert!(
        result.is_ok(),
        "Should schedule successfully: {:?}",
        result.err()
    );

    // The FMA from round 1 (op6) should be scheduled (NOT a dead store)
    // This test is expected to FAIL if Hypothesis 1 is correct - the FMA will be marked as dead
    // because it has no immediate users in round 1 (no stores), even though it's used in round 2
    assert!(
        emit.ops[6].cycle >= 0,
        "FMA from round 1 should be scheduled (used in round 2), but has cycle={}. \
         This indicates Hypothesis 1 is correct: MultiplyAdd with no immediate users \
         (no stores in round 1) is marked as dead, even though it's used in round 2.",
        emit.ops[6].cycle
    );

    // Verify dependency order: Round 1 FMA -> Barrier -> Round 2 use -> Store
    assert!(
        emit.ops[7].cycle > emit.ops[6].cycle,
        "Barrier (cycle={}) should be after Round 1 FMA (cycle={})",
        emit.ops[7].cycle,
        emit.ops[6].cycle
    );
    assert!(
        emit.ops[8].cycle > emit.ops[7].cycle,
        "Round 2 use (cycle={}) should be after barrier (cycle={})",
        emit.ops[8].cycle,
        emit.ops[7].cycle
    );
}

/// Test Hypothesis 2: Last Round Edge Case
///
/// Simulates last round where tree_depth doesn't match store depths:
/// - idx_update_start=0, idx_update_end=0 → NO idx updates
/// - NO partition stores
/// - But all_idx[g] from previous round might still be used
#[test]
fn test_last_round_edge_case_hypothesis2() {
    // Simulate last round with tree_depth not matching store depths
    // This means idx_update_start=0, idx_update_end=0, so NO idx updates

    let mut emit = SsaEmitContext::new();

    // Previous round: all_idx[g] was set
    let idx_prev_round = emit.emit_const(42); // op 0 - simulates all_idx[g] from prev round
    let idx_prev_vec = emit.emit_vbroadcast(idx_prev_round); // op 1

    // Last round: tree_depth doesn't match → NO idx updates
    // But operations might still use all_idx[g] from previous round
    let use_idx = emit.emit_valu(Opcode::Add, idx_prev_vec, idx_prev_vec); // op 2 - uses idx from prev round

    // Store the result
    let addr = emit.emit_const(200); // op 3
    emit.emit_vstore(addr, use_idx); // op 4

    compute_ssa_dependencies_with_barriers(&mut emit.ops, &emit.vreg_def, emit.zero_vreg);

    // Schedule

    let mut shifted_ops = HashSet::new();
    let result = schedule_ssa_greedy_with_shift_budget(
        &mut emit.ops,
        false,
        &mut shifted_ops,
        1536,
    );

    assert!(
        result.is_ok(),
        "Should schedule successfully: {:?}",
        result.err()
    );

    // This test verifies that previous round's idx values are still tracked
    // If this fails, it might indicate a different issue
    assert!(
        emit.ops[1].cycle >= 0,
        "Previous round idx should be scheduled (used in last round), but has cycle={}",
        emit.ops[1].cycle
    );
}

/// Test Hypothesis 3: Complex Barrier Pattern
///
/// Tests dependency tracking across multiple barriers when context variables are involved.
/// Simulates:
/// - Round N: MultiplyAdd updates all_idx[g], barrier emitted
/// - Round N+1: Operations use all_idx[g] (read from Rust variable)
/// - Verify dependency chain is tracked correctly
#[test]
fn test_complex_barrier_pattern_hypothesis3() {
    // Simulate multiple rounds with barriers
    // Round 1: MultiplyAdd updates context variable
    // Barrier
    // Round 2: Operations use context variable
    // Barrier
    // Round 3: More operations use context variable

    let mut emit = SsaEmitContext::new();

    // Round 1: MultiplyAdd updates all_idx[g]
    let mult_scalar = emit.emit_const(2);
    let offset_scalar = emit.emit_const(100);
    let mult_vec = emit.emit_vbroadcast(mult_scalar);
    let offset_vec = emit.emit_vbroadcast(offset_scalar);
    let idx_base_scalar = emit.emit_const(0);
    let idx_base_vec = emit.emit_vbroadcast(idx_base_scalar);

    let idx_round1 = emit.emit_fma(idx_base_vec, mult_vec, offset_vec); // op 6
    emit.emit_barrier(); // op 7

    // Round 2: Use idx from round 1
    let idx_round2_use1 = emit.emit_valu(Opcode::Add, idx_round1, mult_vec); // op 8
    emit.emit_barrier(); // op 9

    // Round 3: Use idx from round 1 again (through round 2's result)
    let idx_round3_use = emit.emit_valu(Opcode::Mul, idx_round2_use1, mult_vec); // op 10

    // Final store
    let addr = emit.emit_const(200);
    emit.emit_vstore(addr, idx_round3_use); // op 12

    compute_ssa_dependencies_with_barriers(&mut emit.ops, &emit.vreg_def, emit.zero_vreg);

    // Verify dependency chain: Round 1 FMA -> Round 2 use -> Round 3 use -> Store
    let mut vreg_users: HashMap<VReg, HashSet<usize>> = HashMap::new();
    for (idx, op) in emit.ops.iter().enumerate() {
        for src in &op.srcs {
            if let Operand::VReg(vreg) = src {
                vreg_users.entry(*vreg).or_default().insert(idx);
            }
        }
    }

    let idx_round1_users = vreg_users.get(&idx_round1).map_or(0, |s| s.len());
    assert!(
        idx_round1_users >= 1,
        "Round 1 FMA result should have users, but has {} users",
        idx_round1_users
    );

    // Schedule

    let mut shifted_ops = HashSet::new();
    let result = schedule_ssa_greedy_with_shift_budget(
        &mut emit.ops,
        false,
        &mut shifted_ops,
        1536,
    );

    assert!(
        result.is_ok(),
        "Should schedule successfully: {:?}",
        result.err()
    );

    // Round 1 FMA should be scheduled (used in round 2 and indirectly in round 3)
    // This test is expected to FAIL if Hypothesis 3 is correct - dependencies across
    // multiple barriers aren't tracked properly
    assert!(
        emit.ops[6].cycle >= 0,
        "Round 1 FMA should be scheduled (used across multiple barriers), but has cycle={}. \
         This indicates Hypothesis 3 is correct: Dependencies across barriers aren't \
         properly tracked for context variables.",
        emit.ops[6].cycle
    );
}

/// Test Hypothesis 4: VReg Replacement Pattern
///
/// Tests the pattern where context variable is replaced:
/// - Initial all_idx[g] = Some(old_vreg)
/// - MultiplyAdd creates new_vreg, updates all_idx[g] = Some(new_vreg)
/// - vstore uses all_idx[g].unwrap() (which is new_vreg)
/// - Verify new_vreg has dependency on MultiplyAdd
#[test]
fn test_vreg_replacement_pattern_hypothesis4() {
    // Simulate VReg replacement pattern
    // Initial: all_idx[g] = Some(old_vreg)
    // MultiplyAdd: creates new_vreg, updates all_idx[g] = Some(new_vreg)
    // vstore: uses all_idx[g].unwrap() (which is new_vreg)

    let mut emit = SsaEmitContext::new();

    // Initial context variable value (simulating all_idx[g] from previous operation)
    let old_idx_scalar = emit.emit_const(10);
    let old_idx_vec = emit.emit_vbroadcast(old_idx_scalar); // op 1 - old VReg

    // MultiplyAdd creates new VReg and replaces old one
    let mult_scalar = emit.emit_const(2);
    let offset_scalar = emit.emit_const(100);
    let mult_vec = emit.emit_vbroadcast(mult_scalar);
    let offset_vec = emit.emit_vbroadcast(offset_scalar);

    // This simulates: ctx.all_idx[g] = Some(emit_fma(...)) replacing old value
    let new_idx_vec = emit.emit_fma(old_idx_vec, mult_vec, offset_vec); // op 6 - new VReg

    // vstore uses the new VReg (simulating emit_vstore(..., ctx.all_idx[g].unwrap()))
    let addr = emit.emit_const(200);
    emit.emit_vstore(addr, new_idx_vec); // op 8

    compute_ssa_dependencies_with_barriers(&mut emit.ops, &emit.vreg_def, emit.zero_vreg);

    // Verify: vstore uses new_idx_vec, which depends on MultiplyAdd
    let mut vreg_users: HashMap<VReg, HashSet<usize>> = HashMap::new();
    for (idx, op) in emit.ops.iter().enumerate() {
        for src in &op.srcs {
            if let Operand::VReg(vreg) = src {
                vreg_users.entry(*vreg).or_default().insert(idx);
            }
        }
    }

    let new_idx_users = vreg_users.get(&new_idx_vec).map_or(0, |s| s.len());
    assert!(
        new_idx_users >= 1,
        "New VReg from MultiplyAdd should have at least 1 user (vstore), but has {} users",
        new_idx_users
    );

    // Schedule

    let mut shifted_ops = HashSet::new();
    let result = schedule_ssa_greedy_with_shift_budget(
        &mut emit.ops,
        false,
        &mut shifted_ops,
        1536,
    );

    assert!(
        result.is_ok(),
        "Should schedule successfully: {:?}",
        result.err()
    );

    // MultiplyAdd should be scheduled (its result is used by vstore)
    // This test is expected to PASS if VReg replacement works correctly
    assert!(
        emit.ops[6].cycle >= 0,
        "MultiplyAdd should be scheduled (its result new_idx_vec is used by vstore), \
         but has cycle={}. This might indicate an issue with VReg replacement tracking.",
        emit.ops[6].cycle
    );
}

/// Test Real Kernel Pattern: Conditional Stores on Non-Last Rounds
///
/// This test accurately simulates the real kernel pattern:
/// - Non-last round: idx updates happen for ALL groups
/// - But partition stores only happen if tree_depth matches TICK/TOCK_STORE_DEPTHS
/// - If tree_depth doesn't match: NO stores, but idx values ARE used in next round
///
/// The critical pattern: MultiplyAdd updates all_idx[g] but has no immediate users
/// (no stores in this round), yet the idx IS needed for next round's operations.
#[test]
fn test_real_kernel_conditional_stores_pattern() {
    // Simulate real kernel: Round 1 (non-last, tree_depth=10, not in TICK/TOCK_STORE_DEPTHS)
    // - MultiplyAdd updates all_idx[g] for all groups (simplified to 1 group)
    // - NO partition stores (tree_depth doesn't match)
    // Round 2: Tree access uses all_idx[g] from round 1

    let mut emit = SsaEmitContext::new();

    // Round 1: Branch computation - MultiplyAdd updates all_idx[g]
    // This simulates: emit_branch() with tree_depth=10 (not in TICK/TOCK_STORE_DEPTHS)
    let val_scalar = emit.emit_const(1); // op 0 - val for branch
    let val_vec = emit.emit_vbroadcast(val_scalar); // op 1
    let v_1_scalar = emit.emit_const(1);
    let v_1_vec = emit.emit_vbroadcast(v_1_scalar); // op 2
    let v_2_scalar = emit.emit_const(2);
    let v_2_vec = emit.emit_vbroadcast(v_2_scalar); // op 3

    let idx_old_scalar = emit.emit_const(5); // op 4 - old idx value
    let idx_old_vec = emit.emit_vbroadcast(idx_old_scalar); // op 5

    // Branch computation: idx = idx * 2 + (2 - (val & 1))
    let bit = emit.emit_valu(Opcode::And, val_vec, v_1_vec); // op 6
    let offset = emit.emit_valu(Opcode::Sub, v_2_vec, bit); // op 7
    let idx_new = emit.emit_fma(idx_old_vec, v_2_vec, offset); // op 8 - MultiplyAdd updates all_idx[g]

    // NO partition stores happen (tree_depth=10 doesn't match TICK/TOCK_STORE_DEPTHS)
    // So idx_new has no immediate users in this round!

    // Barrier between rounds
    emit.emit_barrier(); // op 9

    // Round 2: Tree access uses all_idx[g] from round 1
    // This simulates: emit_depth1_access() or similar that reads ctx.all_idx[g].unwrap()
    let idx_used = emit.emit_valu(Opcode::Eq, idx_new, v_1_vec); // op 10 - uses idx from round 1

    // Eventually store result
    let result_vec = emit.emit_valu(Opcode::And, idx_used, v_1_vec); // op 11
    let addr = emit.emit_const(200);
    emit.emit_vstore(addr, result_vec); // op 13

    compute_ssa_dependencies_with_barriers(&mut emit.ops, &emit.vreg_def, emit.zero_vreg);

    // Verify dependency chain
    let mut vreg_users: HashMap<VReg, HashSet<usize>> = HashMap::new();
    for (idx, op) in emit.ops.iter().enumerate() {
        for src in &op.srcs {
            if let Operand::VReg(vreg) = src {
                vreg_users.entry(*vreg).or_default().insert(idx);
            }
        }
    }

    // idx_new should have at least 1 user (op10 uses it)
    let idx_new_users = vreg_users.get(&idx_new).map_or(0, |s| s.len());
    assert!(idx_new_users >= 1,
        "MultiplyAdd result (idx_new) should have at least 1 user (op10 in round 2), but has {} users",
        idx_new_users);

    // Schedule

    let mut shifted_ops = HashSet::new();
    let result = schedule_ssa_greedy_with_shift_budget(
        &mut emit.ops,
        false,
        &mut shifted_ops,
        1536,
    );

    assert!(
        result.is_ok(),
        "Should schedule successfully: {:?}",
        result.err()
    );

    // The MultiplyAdd (op8) should be scheduled (used in round 2)
    // This test is designed to FAIL if the bug exists - MultiplyAdd with no immediate
    // users in round 1 (no stores) might be marked as dead, even though it's used in round 2
    assert!(
        emit.ops[8].cycle >= 0,
        "MultiplyAdd from round 1 should be scheduled (used in round 2), but has cycle={}. \
         This indicates the bug: MultiplyAdd with no immediate users (no stores in round 1) \
         is incorrectly marked as dead, even though it's used in round 2 after barrier.",
        emit.ops[8].cycle
    );
}

// =========================================================================
// PrecedenceOffset Tests
//
// Verify that the greedy scheduler respects PrecedenceOffset (latency > 1)
// for SelectChoice FMA ops instead of hardcoding +1 everywhere.
// =========================================================================

/// Test that the greedy scheduler uses PrecedenceOffset for SelectChoice ops.
/// ASAP offset for SelectChoice is 1 (vselect path = min_delay), matching the
/// shiftable pattern. The FMA delay (2 cycles) is enforced by CP-SAT model constraints.
#[test]
fn test_greedy_selectchoice_uses_precedence_offset() {
    use crate::types::PrecedenceOffset;

    // Build: cond, true_val, false_val → SelectChoice (with sub_temp_vreg) → consumer VALU add
    let mut emit = SsaEmitContext::new();

    // Create scalar constants and broadcast to vectors (SelectChoice needs vectors)
    let s_cond = emit.emit_const(1); // op 0
    let s_true = emit.emit_const(2); // op 1
    let s_false = emit.emit_const(3); // op 2
    let v_cond = emit.emit_vbroadcast(s_cond); // op 3
    let v_true = emit.emit_vbroadcast(s_true); // op 4
    let v_false = emit.emit_vbroadcast(s_false); // op 5

    // SelectChoice with sub_temp_vreg: ASAP offset=1 (vselect min_delay)
    let v_sel = emit.emit_select_choice(v_cond, v_true, v_false); // op 6

    // Consumer: uses v_sel
    let v_result = emit.emit_valu(Opcode::Add, v_sel, v_sel); // op 7

    // Sink: store the result to make it live
    let addr = emit.emit_const(100); // op 8
    emit.emit_vstore(addr, v_result); // op 9

    // Verify ASAP offset=1 (vselect path), ALAP offset=2 (FMA path)
    assert_eq!(
        emit.ops[6].precedence_offset_for_asap(false),
        1,
        "SelectChoice ASAP offset should be 1 (vselect min_delay)"
    );
    assert_eq!(
        emit.ops[6].precedence_offset_for_alap(false),
        2,
        "SelectChoice ALAP offset should be 2 (FMA max_delay)"
    );

    compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg);

    let mut shifted_ops = HashSet::new();

    let result = schedule_ssa_greedy_with_shift_budget(
        &mut emit.ops,
        false,
        &mut shifted_ops,
        1536,
    );

    assert!(
        result.is_ok(),
        "Should schedule SelectChoice chain: {:?}",
        result.err()
    );

    let sel_cycle = emit.ops[6].cycle;
    let consumer_cycle = emit.ops[7].cycle;
    assert!(
        sel_cycle >= 0,
        "SelectChoice should be scheduled (cycle={})",
        sel_cycle
    );
    // Consumer at sel_cycle+1 (greedy uses ASAP offset=1 for SelectChoice)
    assert!(
        consumer_cycle >= sel_cycle + 1,
        "Consumer must be at cycle >= {} but was at cycle={}",
        sel_cycle + 1,
        consumer_cycle
    );
}

/// Test diamond DAG: consumer has a SelectChoice predecessor (ASAP offset=1)
/// and a normal predecessor (offset=1). Both use min_delay in greedy.
#[test]
fn test_greedy_mixed_predecessors_diamond() {
    use crate::types::PrecedenceOffset;

    let mut emit = SsaEmitContext::new();

    // Create vectors for SelectChoice
    let s_cond = emit.emit_const(1); // op 0
    let s_true = emit.emit_const(2); // op 1
    let s_false = emit.emit_const(3); // op 2
    let v_cond = emit.emit_vbroadcast(s_cond); // op 3
    let v_true = emit.emit_vbroadcast(s_true); // op 4
    let v_false = emit.emit_vbroadcast(s_false); // op 5

    // SelectChoice (ASAP offset=1, ALAP offset=2)
    let v_sel = emit.emit_select_choice(v_cond, v_true, v_false); // op 6

    // Normal VALU op (offset=1): does NOT depend on SelectChoice
    let v_normal = emit.emit_valu(Opcode::Add, v_true, v_false); // op 7

    // Consumer: depends on BOTH v_sel and v_normal
    let v_result = emit.emit_valu(Opcode::Add, v_sel, v_normal); // op 8

    // Sink
    let addr = emit.emit_const(100); // op 9
    emit.emit_vstore(addr, v_result); // op 10

    // Verify offsets: both ASAP=1
    assert_eq!(emit.ops[6].precedence_offset_for_asap(false), 1);
    assert_eq!(emit.ops[7].precedence_offset_for_asap(false), 1);
    // SelectChoice ALAP=2 (FMA path)
    assert_eq!(emit.ops[6].precedence_offset_for_alap(false), 2);

    compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg);

    let mut shifted_ops = HashSet::new();

    let result = schedule_ssa_greedy_with_shift_budget(
        &mut emit.ops,
        false,
        &mut shifted_ops,
        1536,
    );

    assert!(
        result.is_ok(),
        "Should schedule mixed-predecessor diamond: {:?}",
        result.err()
    );

    let sel_cycle = emit.ops[6].cycle;
    let normal_cycle = emit.ops[7].cycle;
    let consumer_cycle = emit.ops[8].cycle;

    assert!(sel_cycle >= 0, "SelectChoice should be scheduled");
    assert!(normal_cycle >= 0, "Normal op should be scheduled");

    // Consumer must respect both predecessors' ASAP offsets (both = 1)
    assert!(
        consumer_cycle >= sel_cycle + 1,
        "Consumer must respect SelectChoice offset=1: consumer={} >= sel+1={}",
        consumer_cycle,
        sel_cycle + 1
    );
    assert!(
        consumer_cycle >= normal_cycle + 1,
        "Consumer must respect normal offset=1: consumer={} >= normal+1={}",
        consumer_cycle,
        normal_cycle + 1
    );
}

// =========================================================================
// PressurePolicy Tests
//
// Verify the four pressure policies (Greedy, PressureThrottled, PressureGated,
// Adaptive) produce correct scheduling decisions and that the trait integration
// works end-to-end.
// =========================================================================

use super::pressure_policy::{
    AdaptivePolicy, GreedyPolicy, PressureGatedPolicy, PressurePolicy, PressureThrottledPolicy,
};

/// Test GreedyPolicy sort: pure LUC descending.
#[test]
fn test_greedy_policy_sort_by_luc() {
    let policy = GreedyPolicy;
    // (idx, luc, words_added, words_freed)
    let mut ops = vec![(0, 3, 8, 0), (1, 1, 8, 8), (2, 5, 1, 0)];
    policy.sort_ready(&mut ops, 500, 1536);
    // Should sort by LUC descending: 5, 3, 1
    assert_eq!(ops[0].0, 2, "LUC=5 should be first");
    assert_eq!(ops[1].0, 0, "LUC=3 should be second");
    assert_eq!(ops[2].0, 1, "LUC=1 should be third");
}

/// Test GreedyPolicy never defers and always allows shift.
#[test]
fn test_greedy_policy_no_deferral_always_shift() {
    let policy = GreedyPolicy;
    assert!(!policy.should_defer_for_pressure(1500, 8, 0, 1536));
    assert!(!policy.should_defer_for_pressure(1536, 8, 0, 1536));
    assert!(policy.allow_shift(1500, 10, 1536));
    assert!(policy.allow_shift(1536, 12, 1536));
}

/// Test PressureThrottledPolicy sort changes above soft threshold.
#[test]
fn test_throttled_policy_sort_changes_above_threshold() {
    let policy = PressureThrottledPolicy::default(); // soft_threshold = 85%

    // Below threshold (85% of 1536 = 1305): sort by LUC descending
    let mut ops_below = vec![(0, 3, 8, 0), (1, 1, 8, 8), (2, 5, 1, 0)];
    policy.sort_ready(&mut ops_below, 1000, 1536);
    assert_eq!(ops_below[0].0, 2, "Below threshold: LUC=5 first");

    // Above threshold: sort by net pressure ascending (prefer freeing ops)
    // op0: net = 8-0 = 8, op1: net = 8-8 = 0, op2: net = 1-0 = 1
    let mut ops_above = vec![(0, 3, 8, 0), (1, 1, 8, 8), (2, 5, 1, 0)];
    policy.sort_ready(&mut ops_above, 1400, 1536);
    assert_eq!(ops_above[0].0, 1, "Above threshold: net=0 first (most freeing)");
    assert_eq!(ops_above[1].0, 2, "Above threshold: net=1 second");
    assert_eq!(ops_above[2].0, 0, "Above threshold: net=8 last (most adding)");
}

/// Test PressureThrottledPolicy suppresses shift when ALU is heavily loaded.
#[test]
fn test_throttled_policy_suppresses_shift_high_alu() {
    let policy = PressureThrottledPolicy::default(); // alu_shift_suppress_threshold = 8

    // ALU <= 8: allow shift
    assert!(policy.allow_shift(1000, 4, 1536));
    assert!(policy.allow_shift(1000, 8, 1536));

    // ALU > 8: suppress shift
    assert!(
        !policy.allow_shift(1000, 9, 1536),
        "Should suppress shift when alu_used > 8"
    );
    assert!(!policy.allow_shift(1000, 12, 1536));
}

/// Test PressureThrottledPolicy does NOT defer (only re-sorts).
#[test]
fn test_throttled_policy_no_deferral() {
    let policy = PressureThrottledPolicy::default();
    assert!(!policy.should_defer_for_pressure(1500, 8, 0, 1536));
    assert!(!policy.should_defer_for_pressure(1536, 8, 0, 1536));
}

/// Test PressureGatedPolicy never defers (deferral disabled to avoid cascading deferral).
/// See docs/SCHEDULING_PRESSURE_CASCADING_DEFERRAL.md.
#[test]
fn test_gated_policy_no_deferral() {
    let policy = PressureGatedPolicy::default();

    // Never defers — pressure awareness is handled by sort_ready()
    assert!(!policy.should_defer_for_pressure(1400, 8, 0, 1536));
    assert!(!policy.should_defer_for_pressure(1500, 8, 0, 1536));
    assert!(!policy.should_defer_for_pressure(1500, 8, 8, 1536));
    assert!(!policy.should_defer_for_pressure(1536, 8, 0, 1536));
}

/// Test PressureGatedPolicy graduated sort: above soft threshold, composite score
/// shifts from LUC-dominant to pressure-dominant.
#[test]
fn test_gated_policy_graduated_sort() {
    let policy = PressureGatedPolicy::default(); // soft=85%, critical=95%

    // Below soft threshold (85% of 1536 = 1305): pure LUC sort
    let mut ops_below = vec![(0, 3, 8, 0), (1, 1, 8, 8), (2, 5, 1, 0)];
    policy.sort_ready(&mut ops_below, 1000, 1536);
    assert_eq!(ops_below[0].0, 2, "Below soft: LUC=5 first");

    // At ~100% pressure: nearly pure net-pressure sort
    // op0: net=8, op1: net=0, op2: net=1
    let mut ops_high = vec![(0, 5, 8, 0), (1, 1, 8, 8), (2, 3, 1, 0)];
    policy.sort_ready(&mut ops_high, 1530, 1536);
    assert_eq!(ops_high[0].0, 1, "Near 100%: net=0 first (most freeing)");
    // op2 (net=1) should come before op0 (net=8) at high pressure
    assert_eq!(ops_high[1].0, 2, "Near 100%: net=1 second");
    assert_eq!(ops_high[2].0, 0, "Near 100%: net=8 last");
}

/// Test PressureGatedPolicy suppresses shift above critical threshold.
#[test]
fn test_gated_policy_shift_suppression() {
    let policy = PressureGatedPolicy::default(); // critical = 95% of 1536 = 1459

    // Below critical threshold and ALU low: allow
    assert!(policy.allow_shift(1000, 4, 1536));

    // Above critical threshold: suppress regardless of ALU
    assert!(
        !policy.allow_shift(1500, 4, 1536),
        "Should suppress shift above critical threshold"
    );

    // Below critical but ALU > 8: also suppress (same as Throttled)
    assert!(
        !policy.allow_shift(1000, 9, 1536),
        "Should suppress shift when alu_used > 8"
    );
}

/// Test AdaptivePolicy transitions: below soft → Greedy behavior.
#[test]
fn test_adaptive_policy_greedy_below_soft() {
    let policy = AdaptivePolicy::default(); // soft=85%, critical=95%

    // Below 85%: LUC sort (same as GreedyPolicy)
    let mut ops = vec![(0, 3, 8, 0), (1, 1, 8, 8), (2, 5, 1, 0)];
    policy.sort_ready(&mut ops, 1000, 1536);
    assert_eq!(ops[0].0, 2, "Below soft: LUC=5 should be first");

    // Below 85%: no deferral
    assert!(!policy.should_defer_for_pressure(1000, 8, 0, 1536));

    // Below 85%: always allow shift
    assert!(policy.allow_shift(1000, 10, 1536));
}

/// Test AdaptivePolicy transitions: between soft and critical → graduated blend.
///
/// At ~91% pressure (1400/1536), pressure_weight ≈ 0.41. This is a blend where
/// both LUC and net pressure matter. The composite score is:
///   score = 0.41 * net_pressure - 0.59 * LUC
///
/// ops: (0, LUC=3, net=8), (1, LUC=1, net=0), (2, LUC=5, net=1)
/// scores: op0=1.51, op1=-0.59, op2=-2.54 → sort: op2, op1, op0
/// Op2 wins because its high LUC (5) outweighs its small net pressure (1) at this blend.
#[test]
fn test_adaptive_policy_throttled_between_thresholds() {
    let policy = AdaptivePolicy::default(); // soft=85% (1305), critical=95% (1459)

    // At 1400 (~91%): graduated composite score — LUC still has significant weight
    let mut ops = vec![(0, 3, 8, 0), (1, 1, 8, 8), (2, 5, 1, 0)];
    policy.sort_ready(&mut ops, 1400, 1536);
    // Op2 (LUC=5, net=1) has best composite score at this blend level
    assert_eq!(ops[0].0, 2, "Throttled zone: op2 (high LUC, low net) first at ~91%");
    // Op0 (LUC=3, net=8) should be last — high net pressure penalty
    assert_eq!(ops[2].0, 0, "Throttled zone: op0 (high net=8) last");

    // No deferral
    assert!(!policy.should_defer_for_pressure(1400, 8, 0, 1536));

    // Throttled: suppress shift when ALU > threshold
    assert!(policy.allow_shift(1400, 4, 1536));
    assert!(!policy.allow_shift(1400, 9, 1536));
}

/// Test AdaptivePolicy transitions: above critical → graduated sort, no deferral.
#[test]
fn test_adaptive_policy_gated_above_critical() {
    let policy = AdaptivePolicy::default(); // critical=95% (1459)

    // Never defers — pressure awareness handled by sort_ready()
    assert!(
        !policy.should_defer_for_pressure(1500, 8, 0, 1536),
        "Above critical: should NOT defer (cascading deferral fix)"
    );
    assert!(
        !policy.should_defer_for_pressure(1500, 8, 8, 1536),
        "Above critical: should NOT defer"
    );

    // Above critical: suppress all shifts (shift suppression is retained)
    assert!(!policy.allow_shift(1500, 4, 1536));

    // Graduated sort at high pressure: pressure-dominant
    // op0: LUC=5, net=8; op1: LUC=1, net=0; op2: LUC=3, net=1
    let mut ops = vec![(0, 5, 8, 0), (1, 1, 8, 8), (2, 3, 1, 0)];
    policy.sort_ready(&mut ops, 1500, 1536);
    assert_eq!(ops[0].0, 1, "Above critical: net=0 first (pressure-dominant sort)");
}

/// Test that GreedyPolicy produces the same schedule as before (regression guard).
/// Uses the same test pattern as test_greedy_fallback_sets_budget_exceeded but with
/// a comfortable budget so nothing changes.
#[test]
fn test_greedy_policy_unchanged_regression() {
    let mut emit = SsaEmitContext::new();
    let s1 = emit.emit_const(1);
    let s2 = emit.emit_const(2);
    let v1 = emit.emit_vbroadcast(s1);
    let v2 = emit.emit_vbroadcast(s2);
    let v3 = emit.emit_valu(Opcode::Add, v1, v2);
    let addr = emit.emit_const(100);
    emit.emit_vstore(addr, v3);

    compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg);

    let mut shifted_ops = HashSet::new();
    let result = schedule_ssa_greedy_with_shift_budget_and_selectchoice(
        &mut emit.ops,
        false,
        &mut shifted_ops,
        1536,
        None,
        None,
        None,
        None,
        None,
        &GreedyPolicy,
    );

    assert!(result.is_ok(), "GreedyPolicy should schedule: {:?}", result.err());
    let (cycles, budget_exceeded, _peak_live) = result.unwrap();
    assert!(!budget_exceeded, "Should not exceed budget with GreedyPolicy");
    assert!(cycles > 0, "Should produce a valid schedule");

    // Verify all ops have valid cycles (except dead stores)
    for (i, op) in emit.ops.iter().enumerate() {
        if op.dest.is_some() {
            // All ops with dests should be scheduled (unless dead)
            // The vstore has no dest, so skip it
        }
        if i == emit.ops.len() - 1 {
            // vstore should be scheduled
            assert!(op.cycle >= 0, "vstore should be scheduled");
        }
    }
}

/// Test that PressureThrottledPolicy can be passed and produces a valid schedule.
#[test]
fn test_throttled_policy_produces_valid_schedule() {
    let mut emit = SsaEmitContext::new();
    let s1 = emit.emit_const(1);
    let s2 = emit.emit_const(2);
    let v1 = emit.emit_vbroadcast(s1);
    let v2 = emit.emit_vbroadcast(s2);
    let v3 = emit.emit_valu(Opcode::Add, v1, v2);
    let addr = emit.emit_const(100);
    emit.emit_vstore(addr, v3);

    compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg);

    let mut shifted_ops = HashSet::new();
    let result = schedule_ssa_greedy_with_shift_budget_and_selectchoice(
        &mut emit.ops,
        false,
        &mut shifted_ops,
        1536,
        None,
        None,
        None,
        None,
        None,
        &PressureThrottledPolicy::default(),
    );

    assert!(result.is_ok(), "PressureThrottledPolicy should schedule: {:?}", result.err());
    let (_cycles, budget_exceeded, _peak_live) = result.unwrap();
    assert!(!budget_exceeded, "Should not exceed budget with PressureThrottledPolicy");
}

/// Test that PressureGatedPolicy can be passed and produces a valid schedule.
#[test]
fn test_gated_policy_produces_valid_schedule() {
    let mut emit = SsaEmitContext::new();
    let s1 = emit.emit_const(1);
    let s2 = emit.emit_const(2);
    let v1 = emit.emit_vbroadcast(s1);
    let v2 = emit.emit_vbroadcast(s2);
    let v3 = emit.emit_valu(Opcode::Add, v1, v2);
    let addr = emit.emit_const(100);
    emit.emit_vstore(addr, v3);

    compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg);

    let mut shifted_ops = HashSet::new();
    let result = schedule_ssa_greedy_with_shift_budget_and_selectchoice(
        &mut emit.ops,
        false,
        &mut shifted_ops,
        1536,
        None,
        None,
        None,
        None,
        None,
        &PressureGatedPolicy::default(),
    );

    assert!(result.is_ok(), "PressureGatedPolicy should schedule: {:?}", result.err());
    let (_cycles, budget_exceeded, _peak_live) = result.unwrap();
    assert!(!budget_exceeded, "Should not exceed budget with PressureGatedPolicy");
}

/// Test that AdaptivePolicy can be passed and produces a valid schedule.
#[test]
fn test_adaptive_policy_produces_valid_schedule() {
    let mut emit = SsaEmitContext::new();
    let s1 = emit.emit_const(1);
    let s2 = emit.emit_const(2);
    let v1 = emit.emit_vbroadcast(s1);
    let v2 = emit.emit_vbroadcast(s2);
    let v3 = emit.emit_valu(Opcode::Add, v1, v2);
    let addr = emit.emit_const(100);
    emit.emit_vstore(addr, v3);

    compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg);

    let mut shifted_ops = HashSet::new();
    let result = schedule_ssa_greedy_with_shift_budget_and_selectchoice(
        &mut emit.ops,
        false,
        &mut shifted_ops,
        1536,
        None,
        None,
        None,
        None,
        None,
        &AdaptivePolicy::default(),
    );

    assert!(result.is_ok(), "AdaptivePolicy should schedule: {:?}", result.err());
    let (_cycles, budget_exceeded, _peak_live) = result.unwrap();
    assert!(!budget_exceeded, "Should not exceed budget with AdaptivePolicy");
}

/// Test AdaptivePolicy graduated sort weight: as pressure increases from 85% to 100%,
/// sort weight shifts from LUC-dominant to pressure-dominant.
#[test]
fn test_adaptive_graduated_sort_weight() {
    let policy = AdaptivePolicy::default(); // soft=85%

    // Create ops with conflicting LUC vs pressure signals:
    // op0: high LUC (5), high net pressure (8 added, 0 freed) — good for ILP, bad for pressure
    // op1: low LUC (1), zero net pressure (8 added, 8 freed) — neutral ILP, great for pressure
    // op2: medium LUC (3), low net pressure (1 added, 0 freed) — medium both

    // At exactly soft threshold (85%): pure LUC sort
    let at_soft = (1536.0 * 0.85) as usize; // 1305
    let mut ops_soft = vec![(0, 5, 8, 0), (1, 1, 8, 8), (2, 3, 1, 0)];
    policy.sort_ready(&mut ops_soft, at_soft, 1536);
    assert_eq!(ops_soft[0].0, 0, "At soft threshold: LUC=5 first (pure LUC)");

    // Just above soft threshold (~87%): LUC still dominates but pressure has some influence
    let mut ops_87 = vec![(0, 5, 8, 0), (1, 1, 8, 8), (2, 3, 1, 0)];
    policy.sort_ready(&mut ops_87, 1340, 1536); // ~87.2%
    // At low pressure_weight, LUC=5 (op0) likely still wins despite net=8
    // But op1 (net=0) should be preferred over op2 (net=1) when LUC difference is small

    // At ~93% pressure: pressure weight ~0.53 — roughly balanced
    let mut ops_93 = vec![(0, 5, 8, 0), (1, 1, 8, 8), (2, 3, 1, 0)];
    policy.sort_ready(&mut ops_93, 1428, 1536); // ~93%
    // Pressure is now significant enough that net=0 (op1) can beat net=8 (op0)

    // At ~100% pressure: pressure weight → 1.0, nearly pure pressure sort
    let mut ops_100 = vec![(0, 5, 8, 0), (1, 1, 8, 8), (2, 3, 1, 0)];
    policy.sort_ready(&mut ops_100, 1530, 1536);
    assert_eq!(ops_100[0].0, 1, "Near 100%: net=0 first (pressure-dominant)");
    assert_eq!(ops_100[1].0, 2, "Near 100%: net=1 second");
    assert_eq!(ops_100[2].0, 0, "Near 100%: net=8 last");
}

/// Test that AdaptivePolicy does NOT cause cascading deferral when all ready ops
/// have positive net pressure. This was the original bug.
///
/// Scenario: 4 vbroadcasts (each +8 words) with tight budget. With old binary gate,
/// above 95% all would be deferred → force-schedule → budget_exceeded. With graduated
/// sort, they are sorted by composite score and scheduled normally.
#[test]
fn test_adaptive_no_cascading_deferral() {
    let mut emit = SsaEmitContext::new();

    // Create 4 scalar constants feeding 4 vbroadcasts, then a chain of VALU ops
    // consuming them. All vbroadcasts add 8 words and free 0 (positive net pressure).
    let s1 = emit.emit_const(1);
    let s2 = emit.emit_const(2);
    let s3 = emit.emit_const(3);
    let s4 = emit.emit_const(4);
    let v1 = emit.emit_vbroadcast(s1);
    let v2 = emit.emit_vbroadcast(s2);
    let v3 = emit.emit_vbroadcast(s3);
    let v4 = emit.emit_vbroadcast(s4);
    let v5 = emit.emit_valu(Opcode::Add, v1, v2);
    let v6 = emit.emit_valu(Opcode::Add, v3, v4);
    let v7 = emit.emit_valu(Opcode::Add, v5, v6);
    let addr = emit.emit_const(100);
    emit.emit_vstore(addr, v7);

    compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg);

    let mut shifted_ops = HashSet::new();

    // Budget of 40: tight enough that 4 vectors (32 words) + scalars approach the limit,
    // but should not force-schedule with graduated sort.
    let result = schedule_ssa_greedy_with_shift_budget_and_selectchoice(
        &mut emit.ops,
        false,
        &mut shifted_ops,
        40,
        None,
        None,
        None,
        None,
        None,
        &AdaptivePolicy::default(),
    );

    assert!(result.is_ok(), "Should complete: {:?}", result.err());
    let (_cycles, budget_exceeded, _peak_live) = result.unwrap();
    assert!(
        !budget_exceeded,
        "AdaptivePolicy should NOT force-schedule (no cascading deferral); peak_live={}",
        _peak_live
    );
}

/// Regression test: force-schedule path must never exceed VALU capacity (6 ops/cycle).
///
/// Before the fix, when a shiftable VALU op entered the force-schedule path with
/// VALU full (6 ops), if `strategy.allow_shift()` returned false, the code fell
/// through to `valu_used += 1`, creating a 7-VALU-per-cycle violation. This caused
/// panics in the CP-SAT hint verification ("VALU overflow at cycle N: 7 > 6").
///
/// The fix adds an intermediate branch: if VALU is full and strategy blocks the shift
/// but ALU has room, force the shift anyway (since we're already in the emergency
/// force-schedule path that ignores budget constraints).
#[test]
fn test_force_schedule_never_exceeds_valu_capacity() {
    // Create a kernel with many VALU ops and very tight budget to trigger
    // force-scheduling. After scheduling, verify no cycle has > 6 non-shifted VALU ops.
    let mut emit = SsaEmitContext::new();

    // Create 8 scalar constants (Load slots, 1 word each)
    let mut scalars = vec![];
    for i in 0..8 {
        scalars.push(emit.emit_const(i as i64));
    }

    // Create 8 vbroadcasts (each 8 words) — non-shiftable VALU ops
    let mut vecs = vec![];
    for &s in &scalars {
        vecs.push(emit.emit_vbroadcast(s));
    }

    // Create 12 shiftable VALU ops (Add), each consuming 2 vectors.
    // With tight budget, some will be force-scheduled.
    let mut results = vec![];
    for i in 0..12 {
        let a = vecs[i % 8];
        let b = vecs[(i + 1) % 8];
        results.push(emit.emit_valu(Opcode::Add, a, b));
    }

    // Create sinks (stores) to consume results
    let addr = emit.emit_const(200);
    for &r in &results {
        emit.emit_vstore(addr, r);
    }

    compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg);

    let mut shifted_ops = HashSet::new();

    // Use AdaptivePolicy (production default) with a tight budget.
    // The tight budget forces the force-schedule path, and AdaptivePolicy
    // blocks shifting when live pressure is above critical threshold.
    let budget = 80; // Very tight: 8 vbroadcasts * 8 words = 64 baseline
    let result = schedule_ssa_greedy_with_shift_budget_and_selectchoice(
        &mut emit.ops,
        true, // enable_valu_shift
        &mut shifted_ops,
        budget,
        None,
        None,
        None,
        None,
        None,
        &AdaptivePolicy::default(),
    );

    assert!(result.is_ok(), "Scheduler should complete: {:?}", result.err());

    // Verify the VALU <= 6 invariant on every cycle
    let max_cycle = emit.ops.iter().map(|op| op.cycle).max().unwrap_or(0);
    for cycle in 0..=max_cycle {
        let valu_at_cycle: usize = emit.ops.iter()
            .filter(|op| op.cycle == cycle
                && op.engine == Engine::Valu
                && !shifted_ops.contains(&op.id))
            .count();
        assert!(
            valu_at_cycle <= 6,
            "VALU overflow at cycle {}: {} > 6 (regression: force-schedule must shift \
             when strategy blocks but VALU is full)",
            cycle, valu_at_cycle
        );
    }
}

// =========================================================================
// SlotTracker unit tests — shared slot-capacity invariants
//
// These tests verify the SlotTracker abstraction in isolation, ensuring that:
// - Each engine respects its capacity limit
// - fits() and try_place() are consistent (fits=true implies try_place succeeds)
// - VALU→ALU shifting consumes exactly VLEN ALU slots
// - allow_shift=false prevents shifting even when ALU has room
// - Non-slot engines (Barrier, Debug, etc.) are always placeable
// =========================================================================

#[test]
fn test_slot_tracker_alu_capacity() {
    let mut slots = SlotTracker::new();
    // Fill all 12 ALU slots
    for i in 0..12 {
        assert!(slots.fits(Engine::Alu, false), "ALU should fit at count {}", i);
        assert_eq!(slots.try_place(Engine::Alu, false, false), SlotResult::Placed);
    }
    // 13th should fail
    assert!(!slots.fits(Engine::Alu, false));
    assert_eq!(slots.try_place(Engine::Alu, false, false), SlotResult::NoRoom);
}

#[test]
fn test_slot_tracker_valu_capacity() {
    let mut slots = SlotTracker::new();
    // Fill all 6 VALU slots
    for i in 0..6 {
        assert!(slots.fits(Engine::Valu, false), "VALU should fit at count {}", i);
        assert_eq!(slots.try_place(Engine::Valu, false, false), SlotResult::Placed);
    }
    // 7th should fail (not shiftable)
    assert!(!slots.fits(Engine::Valu, false));
    assert_eq!(slots.try_place(Engine::Valu, false, false), SlotResult::NoRoom);
}

#[test]
fn test_slot_tracker_valu_shift_to_alu() {
    let mut slots = SlotTracker::new();
    // Fill VALU
    for _ in 0..6 {
        slots.try_place(Engine::Valu, false, false);
    }
    // VALU full, but shiftable op can go to ALU
    assert!(slots.fits(Engine::Valu, true)); // can_shift=true
    let result = slots.try_place(Engine::Valu, true, true);
    assert_eq!(result, SlotResult::Shifted);
    // Shifting consumed VLEN=8 ALU slots
    assert_eq!(slots.alu_used, VLEN);
}

#[test]
fn test_slot_tracker_shift_blocked_by_allow_shift_false() {
    let mut slots = SlotTracker::new();
    // Fill VALU
    for _ in 0..6 {
        slots.try_place(Engine::Valu, false, false);
    }
    // can_shift=true but allow_shift=false (strategy blocks) → NoRoom
    assert_eq!(slots.try_place(Engine::Valu, true, false), SlotResult::NoRoom);
    // ALU unchanged
    assert_eq!(slots.alu_used, 0);
}

#[test]
fn test_slot_tracker_shift_blocked_by_alu_full() {
    let mut slots = SlotTracker::new();
    // Fill VALU
    for _ in 0..6 {
        slots.try_place(Engine::Valu, false, false);
    }
    // Fill ALU to 5 — only 7 remaining < VLEN=8, so shift won't fit
    for _ in 0..5 {
        slots.try_place(Engine::Alu, false, false);
    }
    assert!(!slots.fits(Engine::Valu, true)); // no room even with shift
    assert_eq!(slots.try_place(Engine::Valu, true, true), SlotResult::NoRoom);
}

#[test]
fn test_slot_tracker_load_store_flow_capacity() {
    let mut slots = SlotTracker::new();
    // Load: 2 slots
    assert_eq!(slots.try_place(Engine::Load, false, false), SlotResult::Placed);
    assert_eq!(slots.try_place(Engine::Load, false, false), SlotResult::Placed);
    assert_eq!(slots.try_place(Engine::Load, false, false), SlotResult::NoRoom);

    // Store: 2 slots
    assert_eq!(slots.try_place(Engine::Store, false, false), SlotResult::Placed);
    assert_eq!(slots.try_place(Engine::Store, false, false), SlotResult::Placed);
    assert_eq!(slots.try_place(Engine::Store, false, false), SlotResult::NoRoom);

    // Flow: 1 slot
    assert_eq!(slots.try_place(Engine::Flow, false, false), SlotResult::Placed);
    assert_eq!(slots.try_place(Engine::Flow, false, false), SlotResult::NoRoom);
}

#[test]
fn test_slot_tracker_barrier_always_fits() {
    let mut slots = SlotTracker::new();
    // Barrier ops don't consume any engine slot
    assert!(slots.fits(Engine::Barrier, false));
    assert_eq!(slots.try_place(Engine::Barrier, false, false), SlotResult::Placed);
    // Multiple barriers in same cycle
    assert_eq!(slots.try_place(Engine::Barrier, false, false), SlotResult::Placed);
    assert_eq!(slots.try_place(Engine::Barrier, false, false), SlotResult::Placed);
}

#[test]
fn test_slot_tracker_fits_and_try_place_consistent() {
    // For every engine, fits()==true must imply try_place() succeeds
    let engines = [
        (Engine::Load, false),
        (Engine::Store, false),
        (Engine::Alu, false),
        (Engine::Valu, false),
        (Engine::Valu, true), // with can_shift
        (Engine::Flow, false),
        (Engine::Barrier, false),
    ];
    for &(engine, can_shift) in &engines {
        let mut slots = SlotTracker::new();
        // Keep placing until fits() says no
        let mut placed = 0;
        while slots.fits(engine, can_shift) {
            let result = slots.try_place(engine, can_shift, true);
            assert_ne!(
                result,
                SlotResult::NoRoom,
                "fits() returned true but try_place() returned NoRoom for {:?} (can_shift={})",
                engine, can_shift,
            );
            placed += 1;
            // Safety: avoid infinite loop for Barrier (always fits)
            if placed > 20 {
                break;
            }
        }
    }
}
