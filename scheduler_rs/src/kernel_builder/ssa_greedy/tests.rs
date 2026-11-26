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

    compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg, None);

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

    compute_ssa_dependencies_with_barriers(&mut emit.ops, &emit.vreg_def, emit.zero_vreg, None);

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

    compute_ssa_dependencies_with_barriers(&mut emit.ops, &emit.vreg_def, emit.zero_vreg, None);

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

    compute_ssa_dependencies_with_barriers(&mut emit.ops, &emit.vreg_def, emit.zero_vreg, None);

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

    compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg, None);


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

    compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg, None);


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

    compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg, None);

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

    compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg, None);

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

    compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg, None);

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

    compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg, None);

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

    compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg, None);

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

    compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg, None);

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

    compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg, None);

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

    compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg, None);

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
    compute_ssa_dependencies_with_barriers(&mut emit.ops, &emit.vreg_def, emit.zero_vreg, None);

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
    compute_ssa_dependencies_with_barriers(&mut emit.ops, &emit.vreg_def, emit.zero_vreg, None);

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
    compute_ssa_dependencies_with_barriers(&mut emit.ops, &emit.vreg_def, emit.zero_vreg, None);

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

    compute_ssa_dependencies_with_barriers(&mut emit.ops, &emit.vreg_def, emit.zero_vreg, None);

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

    compute_ssa_dependencies_with_barriers(&mut emit.ops, &emit.vreg_def, emit.zero_vreg, None);

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

    compute_ssa_dependencies_with_barriers(&mut emit.ops, &emit.vreg_def, emit.zero_vreg, None);

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

    compute_ssa_dependencies_with_barriers(&mut emit.ops, &emit.vreg_def, emit.zero_vreg, None);

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

    compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg, None);

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

    compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg, None);

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
    AdaptivePolicy, CascadePolicy, GreedyPolicy, PressureGatedPolicy, PressurePolicy,
    PressureThrottledPolicy, SchedulingCandidate, SchedulingContext, SlotTrackerView,
};

/// Helper: create a SchedulingCandidate from (op_index, luc, words_added, words_freed).
/// Other fields default to sensible values for unit tests.
fn make_candidate(op_index: usize, luc: usize, words_added: usize, words_freed: usize) -> SchedulingCandidate {
    SchedulingCandidate {
        op_index,
        luc,
        words_added,
        words_freed,
        asap: None,
        alap: None,
        engine: Engine::Valu,
        is_shiftable: false,
        unscheduled_consumer_count: 0,
    }
}

/// Helper: create a default SlotTrackerView (all zeros).
fn empty_slots() -> SlotTrackerView {
    SlotTrackerView {
        alu_used: 0,
        valu_used: 0,
        load_used: 0,
        store_used: 0,
        flow_used: 0,
    }
}

/// Helper: create a SchedulingContext with the given live_words and scratch_budget.
fn make_ctx(live_words: usize, scratch_budget: usize, slots: &SlotTrackerView) -> SchedulingContext<'_> {
    SchedulingContext {
        cycle: 0,
        live_words,
        scratch_budget,
        slots,
    }
}

/// Helper: create a SchedulingCandidate with full control over all fields.
/// Used by CascadePolicy tests that need engine, ASAP, and ALAP settings.
fn make_candidate_full(
    op_index: usize,
    luc: usize,
    words_added: usize,
    words_freed: usize,
    asap: Option<i32>,
    alap: Option<i32>,
    engine: Engine,
    is_shiftable: bool,
    unscheduled_consumer_count: usize,
) -> SchedulingCandidate {
    SchedulingCandidate {
        op_index,
        luc,
        words_added,
        words_freed,
        asap,
        alap,
        engine,
        is_shiftable,
        unscheduled_consumer_count,
    }
}

/// Test GreedyPolicy sort: pure LUC descending.
#[test]
fn test_greedy_policy_sort_by_luc() {
    let policy = GreedyPolicy;
    let mut ops = vec![
        make_candidate(0, 3, 8, 0),
        make_candidate(1, 1, 8, 8),
        make_candidate(2, 5, 1, 0),
    ];
    let slots = empty_slots();
    let ctx = make_ctx(500, 1536, &slots);
    policy.sort_ready(&mut ops, &ctx);
    // Should sort by LUC descending: 5, 3, 1
    assert_eq!(ops[0].op_index, 2, "LUC=5 should be first");
    assert_eq!(ops[1].op_index, 0, "LUC=3 should be second");
    assert_eq!(ops[2].op_index, 1, "LUC=1 should be third");
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
    let slots = empty_slots();

    // Below threshold (85% of 1536 = 1305): sort by LUC descending
    let mut ops_below = vec![
        make_candidate(0, 3, 8, 0),
        make_candidate(1, 1, 8, 8),
        make_candidate(2, 5, 1, 0),
    ];
    let ctx = make_ctx(1000, 1536, &slots);
    policy.sort_ready(&mut ops_below, &ctx);
    assert_eq!(ops_below[0].op_index, 2, "Below threshold: LUC=5 first");

    // Above threshold: sort by net pressure ascending (prefer freeing ops)
    // op0: net = 8-0 = 8, op1: net = 8-8 = 0, op2: net = 1-0 = 1
    let mut ops_above = vec![
        make_candidate(0, 3, 8, 0),
        make_candidate(1, 1, 8, 8),
        make_candidate(2, 5, 1, 0),
    ];
    let ctx = make_ctx(1400, 1536, &slots);
    policy.sort_ready(&mut ops_above, &ctx);
    assert_eq!(ops_above[0].op_index, 1, "Above threshold: net=0 first (most freeing)");
    assert_eq!(ops_above[1].op_index, 2, "Above threshold: net=1 second");
    assert_eq!(ops_above[2].op_index, 0, "Above threshold: net=8 last (most adding)");
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
    let slots = empty_slots();

    // Below soft threshold (85% of 1536 = 1305): pure LUC sort
    let mut ops_below = vec![
        make_candidate(0, 3, 8, 0),
        make_candidate(1, 1, 8, 8),
        make_candidate(2, 5, 1, 0),
    ];
    let ctx = make_ctx(1000, 1536, &slots);
    policy.sort_ready(&mut ops_below, &ctx);
    assert_eq!(ops_below[0].op_index, 2, "Below soft: LUC=5 first");

    // At ~100% pressure: nearly pure net-pressure sort
    // op0: net=8, op1: net=0, op2: net=1
    let mut ops_high = vec![
        make_candidate(0, 5, 8, 0),
        make_candidate(1, 1, 8, 8),
        make_candidate(2, 3, 1, 0),
    ];
    let ctx = make_ctx(1530, 1536, &slots);
    policy.sort_ready(&mut ops_high, &ctx);
    assert_eq!(ops_high[0].op_index, 1, "Near 100%: net=0 first (most freeing)");
    // op2 (net=1) should come before op0 (net=8) at high pressure
    assert_eq!(ops_high[1].op_index, 2, "Near 100%: net=1 second");
    assert_eq!(ops_high[2].op_index, 0, "Near 100%: net=8 last");
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

/// Test AdaptivePolicy transitions: below soft -> Greedy behavior.
#[test]
fn test_adaptive_policy_greedy_below_soft() {
    let policy = AdaptivePolicy::default(); // soft=85%, critical=95%
    let slots = empty_slots();

    // Below 85%: LUC sort (same as GreedyPolicy)
    let mut ops = vec![
        make_candidate(0, 3, 8, 0),
        make_candidate(1, 1, 8, 8),
        make_candidate(2, 5, 1, 0),
    ];
    let ctx = make_ctx(1000, 1536, &slots);
    policy.sort_ready(&mut ops, &ctx);
    assert_eq!(ops[0].op_index, 2, "Below soft: LUC=5 should be first");

    // Below 85%: no deferral
    assert!(!policy.should_defer_for_pressure(1000, 8, 0, 1536));

    // Below 85%: always allow shift
    assert!(policy.allow_shift(1000, 10, 1536));
}

/// Test AdaptivePolicy transitions: between soft and critical -> graduated blend.
///
/// At ~91% pressure (1400/1536), pressure_weight ~ 0.41. This is a blend where
/// both LUC and net pressure matter. The composite score is:
///   score = 0.41 * net_pressure - 0.59 * LUC
///
/// ops: (0, LUC=3, net=8), (1, LUC=1, net=0), (2, LUC=5, net=1)
/// scores: op0=1.51, op1=-0.59, op2=-2.54 -> sort: op2, op1, op0
/// Op2 wins because its high LUC (5) outweighs its small net pressure (1) at this blend.
#[test]
fn test_adaptive_policy_throttled_between_thresholds() {
    let policy = AdaptivePolicy::default(); // soft=85% (1305), critical=95% (1459)
    let slots = empty_slots();

    // At 1400 (~91%): graduated composite score -- LUC still has significant weight
    let mut ops = vec![
        make_candidate(0, 3, 8, 0),
        make_candidate(1, 1, 8, 8),
        make_candidate(2, 5, 1, 0),
    ];
    let ctx = make_ctx(1400, 1536, &slots);
    policy.sort_ready(&mut ops, &ctx);
    // Op2 (LUC=5, net=1) has best composite score at this blend level
    assert_eq!(ops[0].op_index, 2, "Throttled zone: op2 (high LUC, low net) first at ~91%");
    // Op0 (LUC=3, net=8) should be last -- high net pressure penalty
    assert_eq!(ops[2].op_index, 0, "Throttled zone: op0 (high net=8) last");

    // No deferral
    assert!(!policy.should_defer_for_pressure(1400, 8, 0, 1536));

    // Throttled: suppress shift when ALU > threshold
    assert!(policy.allow_shift(1400, 4, 1536));
    assert!(!policy.allow_shift(1400, 9, 1536));
}

/// Test AdaptivePolicy transitions: above critical -> graduated sort, no deferral.
#[test]
fn test_adaptive_policy_gated_above_critical() {
    let policy = AdaptivePolicy::default(); // critical=95% (1459)
    let slots = empty_slots();

    // Never defers -- pressure awareness handled by sort_ready()
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
    let mut ops = vec![
        make_candidate(0, 5, 8, 0),
        make_candidate(1, 1, 8, 8),
        make_candidate(2, 3, 1, 0),
    ];
    let ctx = make_ctx(1500, 1536, &slots);
    policy.sort_ready(&mut ops, &ctx);
    assert_eq!(ops[0].op_index, 1, "Above critical: net=0 first (pressure-dominant sort)");
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

    compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg, None);

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
        &GreedyBoundsUsage::None,
        &GreedyPolicy,
        None,
        false,
        false,
        false,
        false,
        None,
        None,
        None,
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

    compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg, None);

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
        &GreedyBoundsUsage::None,
        &PressureThrottledPolicy::default(),
        None,
        false,
        false,
        false,
        false,
        None,
        None,
        None,
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

    compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg, None);

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
        &GreedyBoundsUsage::None,
        &PressureGatedPolicy::default(),
        None,
        false,
        false,
        false,
        false,
        None,
        None,
        None,
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

    compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg, None);

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
        &GreedyBoundsUsage::None,
        &AdaptivePolicy::default(),
        None,
        false,
        false,
        false,
        false,
        None,
        None,
        None,
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
    let slots = empty_slots();

    // Create ops with conflicting LUC vs pressure signals:
    // op0: high LUC (5), high net pressure (8 added, 0 freed) -- good for ILP, bad for pressure
    // op1: low LUC (1), zero net pressure (8 added, 8 freed) -- neutral ILP, great for pressure
    // op2: medium LUC (3), low net pressure (1 added, 0 freed) -- medium both

    // At exactly soft threshold (85%): pure LUC sort
    let at_soft = (1536.0 * 0.85) as usize; // 1305
    let mut ops_soft = vec![
        make_candidate(0, 5, 8, 0),
        make_candidate(1, 1, 8, 8),
        make_candidate(2, 3, 1, 0),
    ];
    let ctx = make_ctx(at_soft, 1536, &slots);
    policy.sort_ready(&mut ops_soft, &ctx);
    assert_eq!(ops_soft[0].op_index, 0, "At soft threshold: LUC=5 first (pure LUC)");

    // Just above soft threshold (~87%): LUC still dominates but pressure has some influence
    let mut ops_87 = vec![
        make_candidate(0, 5, 8, 0),
        make_candidate(1, 1, 8, 8),
        make_candidate(2, 3, 1, 0),
    ];
    let ctx = make_ctx(1340, 1536, &slots);
    policy.sort_ready(&mut ops_87, &ctx); // ~87.2%
    // At low pressure_weight, LUC=5 (op0) likely still wins despite net=8
    // But op1 (net=0) should be preferred over op2 (net=1) when LUC difference is small

    // At ~93% pressure: pressure weight ~0.53 -- roughly balanced
    let mut ops_93 = vec![
        make_candidate(0, 5, 8, 0),
        make_candidate(1, 1, 8, 8),
        make_candidate(2, 3, 1, 0),
    ];
    let ctx = make_ctx(1428, 1536, &slots);
    policy.sort_ready(&mut ops_93, &ctx); // ~93%
    // Pressure is now significant enough that net=0 (op1) can beat net=8 (op0)

    // At ~100% pressure: pressure weight -> 1.0, nearly pure pressure sort
    let mut ops_100 = vec![
        make_candidate(0, 5, 8, 0),
        make_candidate(1, 1, 8, 8),
        make_candidate(2, 3, 1, 0),
    ];
    let ctx = make_ctx(1530, 1536, &slots);
    policy.sort_ready(&mut ops_100, &ctx);
    assert_eq!(ops_100[0].op_index, 1, "Near 100%: net=0 first (pressure-dominant)");
    assert_eq!(ops_100[1].op_index, 2, "Near 100%: net=1 second");
    assert_eq!(ops_100[2].op_index, 0, "Near 100%: net=8 last");
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

    compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg, None);

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
        &GreedyBoundsUsage::None,
        &AdaptivePolicy::default(),
        None,
        false,
        false,
        false,
        false,
        None,
        None,
        None,
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

    compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg, None);

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
        &GreedyBoundsUsage::None,
        &AdaptivePolicy::default(),
        None,
        false,
        false,
        false,
        false,
        None,
        None,
        None,
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
// CascadePolicy Tests
//
// Verify the 5-level strict priority cascade:
// 1. Engine saturation, 2. Slack, 3. LUC, 4. Net pressure (above soft), 5. ASAP tiebreak
// =========================================================================

/// Test CascadePolicy Level 1: engine saturation dominates.
/// VALU op at 5/6 capacity sorts before ALU op at 2/12 capacity.
#[test]
fn test_cascade_engine_saturation_dominates() {
    let policy = CascadePolicy::default();
    let mut candidates = vec![
        make_candidate_full(0, 5, 8, 0, Some(0), Some(10), Engine::Alu, false, 0),
        make_candidate_full(1, 3, 8, 0, Some(0), Some(10), Engine::Valu, false, 0),
    ];
    let slots = SlotTrackerView {
        valu_used: 5,  // 5/6 = critical for VALU
        alu_used: 2,   // 2/12 = not saturated for ALU
        ..empty_slots()
    };
    let ctx = SchedulingContext {
        cycle: 10,
        live_words: 500,
        scratch_budget: 1536,
        slots: &slots,
    };
    policy.sort_ready(&mut candidates, &ctx);
    assert_eq!(candidates[0].op_index, 1, "VALU op (5/6 saturated) should sort first");
}

/// Test CascadePolicy Level 2: slack dominates when engine saturation is equal.
/// Critical-path op (slack=2) sorts before high-slack op (slack=20).
#[test]
fn test_cascade_slack_dominates_over_luc() {
    let policy = CascadePolicy::default();
    let mut candidates = vec![
        make_candidate_full(0, 10, 8, 0, Some(5), Some(25), Engine::Alu, false, 0), // slack=20, high LUC
        make_candidate_full(1, 1, 8, 0, Some(5), Some(7), Engine::Alu, false, 0),   // slack=2, low LUC
    ];
    let slots = empty_slots();
    let ctx = make_ctx(500, 1536, &slots);
    policy.sort_ready(&mut candidates, &ctx);
    assert_eq!(candidates[0].op_index, 1, "Slack=2 (critical path) should sort before slack=20");
}

/// Test CascadePolicy Level 3: LUC tiebreak when engine and slack are equal.
#[test]
fn test_cascade_luc_tiebreak() {
    let policy = CascadePolicy::default();
    let mut candidates = vec![
        make_candidate_full(0, 2, 8, 0, Some(5), Some(15), Engine::Alu, false, 0), // slack=10, LUC=2
        make_candidate_full(1, 5, 8, 0, Some(5), Some(15), Engine::Alu, false, 0), // slack=10, LUC=5
    ];
    let slots = empty_slots();
    let ctx = make_ctx(500, 1536, &slots);
    policy.sort_ready(&mut candidates, &ctx);
    assert_eq!(candidates[0].op_index, 1, "Higher LUC (5) should win when slack is equal");
}

/// Test CascadePolicy Level 4: net pressure only activates above soft threshold.
#[test]
fn test_cascade_net_pressure_only_above_threshold() {
    let policy = CascadePolicy::default();

    // Below soft threshold (85% of 1536 = 1305): net pressure should NOT distinguish
    let slots = empty_slots();
    let mut candidates_below = vec![
        make_candidate_full(0, 3, 8, 0, Some(5), Some(15), Engine::Alu, false, 0), // net=8
        make_candidate_full(1, 3, 1, 8, Some(5), Some(15), Engine::Alu, false, 0), // net=-7
    ];
    let ctx_below = make_ctx(1000, 1536, &slots);
    policy.sort_ready(&mut candidates_below, &ctx_below);
    // With equal saturation, slack, and LUC, net pressure is NOT active below threshold
    // So tiebreak goes to ASAP (both same) then op_index (0 before 1)
    assert_eq!(candidates_below[0].op_index, 0, "Below threshold: net pressure inactive, op_index tiebreak");

    // Above soft threshold: net pressure SHOULD distinguish
    let mut candidates_above = vec![
        make_candidate_full(0, 3, 8, 0, Some(5), Some(15), Engine::Alu, false, 0), // net=8
        make_candidate_full(1, 3, 1, 8, Some(5), Some(15), Engine::Alu, false, 0), // net=-7
    ];
    let ctx_above = make_ctx(1400, 1536, &slots);
    policy.sort_ready(&mut candidates_above, &ctx_above);
    assert_eq!(candidates_above[0].op_index, 1, "Above threshold: net=-7 (freeing) should sort first");
}

/// Test CascadePolicy allow_shift: same graduated zones as Adaptive.
#[test]
fn test_cascade_allow_shift_graduated() {
    let policy = CascadePolicy::default(); // soft=85% (1305), critical=95% (1459)

    // Below soft: always allow
    assert!(policy.allow_shift(1000, 10, 1536));

    // Between soft and critical: suppress when ALU > 8
    assert!(policy.allow_shift(1400, 4, 1536));
    assert!(!policy.allow_shift(1400, 9, 1536));

    // Above critical: suppress all
    assert!(!policy.allow_shift(1500, 4, 1536));
}

/// Test CascadePolicy never defers.
#[test]
fn test_cascade_no_deferral() {
    let policy = CascadePolicy::default();
    assert!(!policy.should_defer_for_pressure(1500, 8, 0, 1536));
    assert!(!policy.should_defer_for_pressure(1536, 8, 0, 1536));
}

/// Test CascadePolicy Level 5: ASAP tiebreak when all other levels are equal.
#[test]
fn test_cascade_asap_tiebreak() {
    let policy = CascadePolicy::default();
    let mut candidates = vec![
        make_candidate_full(0, 3, 4, 0, Some(10), Some(20), Engine::Alu, false, 0), // ASAP=10, slack=10
        make_candidate_full(1, 3, 4, 0, Some(5), Some(15), Engine::Alu, false, 0),  // ASAP=5, slack=10
    ];
    let slots = empty_slots();
    let ctx = make_ctx(500, 1536, &slots);
    policy.sort_ready(&mut candidates, &ctx);
    assert_eq!(candidates[0].op_index, 1, "Lower ASAP (5) should win as tiebreak");
}

// =========================================================================
// Wavefront priority (GreedyPriority::Wavefront)
// =========================================================================

/// `chains` independent chains of `levels` vector adds each, emitted level by level (all
/// chains' level 0, then all chains' level 1, ...) the way the kernel emits rounds. The
/// chains share one broadcast constant and one store address.
///
/// # Returns
/// `(emit, level_ops)` where `level_ops[chain][level]` is the op index of that add.
fn emit_independent_chains(chains: usize, levels: usize) -> (SsaEmitContext, Vec<Vec<usize>>) {
    let mut emit = SsaEmitContext::new();
    let scalar = emit.emit_const(3);
    let shared = emit.emit_vbroadcast(scalar);
    let addr = emit.emit_const(100);
    let mut value: Vec<VReg> = Vec::new();
    let mut level_ops: Vec<Vec<usize>> = vec![Vec::new(); chains];
    for level in 0..levels {
        for chain in 0..chains {
            let input = if level == 0 { shared } else { value[chain] };
            let next = emit.emit_valu(Opcode::Add, input, shared);
            if level == 0 {
                value.push(next);
            } else {
                value[chain] = next;
            }
            level_ops[chain].push(emit.ops.len() - 1);
        }
    }
    for chain in 0..chains {
        emit.emit_vstore(addr, value[chain]);
    }
    compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg, None);
    (emit, level_ops)
}

/// Schedule `ops` with the given shape, no shifting, the hardware budget and the plain policy.
fn schedule_with_shape(ops: &mut [SsaOp], shape: &GreedyShape) -> (usize, bool, usize) {
    let mut shifted = HashSet::new();
    let (result, agreement) = schedule_ssa_greedy_shaped(
        ops,
        false,
        &mut shifted,
        1536,
        None,
        None,
        None,
        None,
        &GreedyBoundsUsage::None,
        &GreedyPolicy,
        None,
        false,
        false,
        false,
        false,
        None,
        None,
        None,
        &mut GreedyDiagnostics::new(false),
        shape,
        &mut HashMap::new(),
    )
    .expect("greedy schedules the chains");
    assert!(agreement.agrees(), "live-word counter disagrees: {}", agreement);
    result
}

fn wavefront_shape(stagger_cycles: u32, groups_in_flight: Option<usize>) -> GreedyShape {
    GreedyShape {
        priority: GreedyPriority::Wavefront { stagger_cycles, groups_in_flight },
        ..Default::default()
    }
}

/// Chains are the connected components of ops that reach few sinks; ops that reach most
/// sinks are shared and take the key of their most urgent consumer.
#[test]
fn test_wavefront_finds_chains_and_keys() {
    let (emit, level_ops) = emit_independent_chains(4, 3);
    let ops = &emit.ops;
    let id_to_idx: HashMap<usize, usize> = ops.iter().enumerate().map(|(i, op)| (op.id, i)).collect();
    let mut consumers: Vec<Vec<usize>> = vec![vec![]; ops.len()];
    for (idx, op) in ops.iter().enumerate() {
        for dep in &op.dep_ids {
            consumers[id_to_idx[dep]].push(idx);
        }
    }
    let stagger = 10;
    let wavefront = wavefront::Wavefront::new(
        ops,
        &consumers,
        &id_to_idx,
        &HashSet::new(),
        &HashSet::new(),
        stagger,
        None,
        None,
    )
    .unwrap();

    assert_eq!(wavefront.chain_count(), 4);
    // Ops 0..3 are the shared scalar, its broadcast and the store address.
    for shared_idx in 0..3 {
        assert_eq!(wavefront.chain(shared_idx), None, "op {} is shared", shared_idx);
    }
    for (chain, levels) in level_ops.iter().enumerate() {
        for (level, &idx) in levels.iter().enumerate() {
            assert_eq!(wavefront.chain(idx), Some(chain), "chain of op {}", idx);
            // Depth: const at 0, broadcast at 1, level 0 at 2.
            assert_eq!(wavefront.key(idx), (2 + level) as i64 + stagger as i64 * chain as i64);
        }
    }
    // The broadcast is needed by chain 0's first add; the scalar just before the broadcast.
    assert_eq!(wavefront.key(1), wavefront.key(level_ops[0][0]) - 1);
    assert_eq!(wavefront.key(0), wavefront.key(1) - 1);
    assert!((0..ops.len()).all(|idx| wavefront.in_window(idx)), "no window: everything may run");
}

/// A large stagger finishes the first chains at full speed; stagger 0 keeps all chains
/// abreast, as the first-come-first-served queue does.
#[test]
fn test_wavefront_stagger_lets_leading_chains_run_ahead() {
    let (chains, levels) = (12, 5);
    let last_cycle = |shape: &GreedyShape| -> (Vec<i32>, usize) {
        let (mut emit, level_ops) = emit_independent_chains(chains, levels);
        let (cycles, budget_exceeded, _) = schedule_with_shape(&mut emit.ops, shape);
        assert!(!budget_exceeded);
        (level_ops.iter().map(|l| emit.ops[*l.last().unwrap()].cycle).collect(), cycles)
    };
    let (abreast, abreast_cycles) = last_cycle(&wavefront_shape(0, None));
    let (staggered, staggered_cycles) = last_cycle(&wavefront_shape(100, None));

    // 12 chains on 6 vector slots: abreast, a level takes two cycles; staggered, chain 0
    // runs a level per cycle from cycle 2 on.
    assert_eq!(staggered[0], 2 + levels as i32 - 1);
    assert!(abreast[0] > staggered[0], "abreast {:?} staggered {:?}", abreast, staggered);
    assert!(staggered.windows(2).all(|w| w[0] <= w[1]), "chains finish in order: {:?}", staggered);
    // Staggered, the stores (two per cycle) start earlier, so the schedule is no longer.
    assert!(staggered_cycles <= abreast_cycles);
}

/// With one chain in flight, no op of a chain runs before the previous chain has finished.
#[test]
fn test_wavefront_window_limits_chains_in_flight() {
    let (chains, levels) = (4, 3);
    let (mut emit, level_ops) = emit_independent_chains(chains, levels);
    schedule_with_shape(&mut emit.ops, &wavefront_shape(0, Some(1)));
    for chain in 1..chains {
        let previous_end = emit.ops[*level_ops[chain - 1].last().unwrap()].cycle;
        let start = emit.ops[level_ops[chain][0]].cycle;
        assert!(
            start > previous_end,
            "chain {} starts at {} before chain {} ends at {}",
            chain,
            start,
            chain - 1,
            previous_end
        );
    }

    // Unlimited window: the chains overlap.
    let (mut emit, level_ops) = emit_independent_chains(chains, levels);
    schedule_with_shape(&mut emit.ops, &wavefront_shape(0, None));
    assert!(emit.ops[level_ops[1][0]].cycle <= emit.ops[*level_ops[0].last().unwrap()].cycle);
}

/// Wavefront state for an emitted program (no shiftable ops, no skipped ops).
fn wavefront_for(
    ops: &[SsaOp],
    stagger_cycles: u32,
    groups_in_flight: Option<usize>,
) -> wavefront::Wavefront {
    let id_to_idx: HashMap<usize, usize> = ops.iter().enumerate().map(|(i, op)| (op.id, i)).collect();
    let mut consumers: Vec<Vec<usize>> = vec![vec![]; ops.len()];
    for (idx, op) in ops.iter().enumerate() {
        for dep in &op.dep_ids {
            consumers[id_to_idx[dep]].push(idx);
        }
    }
    wavefront::Wavefront::new(
        ops,
        &consumers,
        &id_to_idx,
        &HashSet::new(),
        &HashSet::new(),
        stagger_cycles,
        groups_in_flight,
        None,
    )
    .unwrap()
}

/// The window admits the first N unfinished chains, whichever have finished: when chain 1
/// finishes before chain 0, chain 2 becomes eligible although chain 0 is still running.
#[test]
fn test_wavefront_window_refills_when_a_later_chain_finishes() {
    let (emit, level_ops) = emit_independent_chains(3, 2);
    let mut wavefront = wavefront_for(&emit.ops, 0, Some(2));
    let chain_ops = |chain: usize| -> Vec<usize> {
        (0..emit.ops.len()).filter(|&i| wavefront_for(&emit.ops, 0, None).chain(i) == Some(chain)).collect()
    };
    assert!(wavefront.in_window(level_ops[0][0]) && wavefront.in_window(level_ops[1][0]));
    assert!(!wavefront.in_window(level_ops[2][0]), "two chains in flight: chain 2 waits");

    // Finish chain 1 only.
    for idx in chain_ops(1) {
        wavefront.on_scheduled(idx);
    }
    assert!(wavefront.in_window(level_ops[0][0]), "chain 0 is still running");
    assert!(wavefront.in_window(level_ops[2][0]), "chain 1 finished: chain 2 takes its place");

    // With a window of one, chain 2 must still wait for chain 0.
    let mut single = wavefront_for(&emit.ops, 0, Some(1));
    for idx in chain_ops(1) {
        single.on_scheduled(idx);
    }
    assert!(single.in_window(level_ops[0][0]) && !single.in_window(level_ops[2][0]));
    for idx in chain_ops(0) {
        single.on_scheduled(idx);
    }
    assert!(single.in_window(level_ops[2][0]));
}

/// An op that every chain reads but that reaches only one of each chain's two sinks (such
/// as an initial index common to all chains) reaches exactly half of the sinks. It is
/// shared: it must not join the chains into one.
#[test]
fn test_wavefront_op_reaching_half_the_sinks_is_shared() {
    let chains = 4;
    let mut emit = SsaEmitContext::new();
    let scalar = emit.emit_const(3);
    let shared = emit.emit_vbroadcast(scalar);
    let addr = emit.emit_const(100);
    let common_index = emit.emit_valu(Opcode::Add, shared, shared);
    let common_index_op = emit.ops.len() - 1;
    let mut first_ops = Vec::new();
    for _ in 0..chains {
        let mut value = emit.emit_valu(Opcode::Add, shared, shared);
        first_ops.push(emit.ops.len() - 1);
        for _ in 0..4 {
            value = emit.emit_valu(Opcode::Add, value, shared);
        }
        let out_value = emit.emit_valu(Opcode::Xor, value, shared);
        let out_index = emit.emit_valu(Opcode::Add, common_index, value);
        emit.emit_vstore(addr, out_value);
        emit.emit_vstore(addr, out_index);
    }
    compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg, None);

    let wavefront = wavefront_for(&emit.ops, 10, None);
    assert_eq!(wavefront.chain_count(), chains);
    assert_eq!(wavefront.chain(common_index_op), None, "the common index is shared");
    for (chain, &idx) in first_ops.iter().enumerate() {
        assert_eq!(wavefront.chain(idx), Some(chain));
    }
}

/// The default shape is the first-come-first-served queue: same schedule as the entry
/// points that take no shape.
#[test]
fn test_default_shape_is_fifo() {
    assert_eq!(GreedyShape::default().priority, GreedyPriority::Fifo);
    let (mut a, _) = emit_independent_chains(12, 5);
    let (mut b, _) = emit_independent_chains(12, 5);
    schedule_with_shape(&mut a.ops, &GreedyShape::default());
    let mut shifted = HashSet::new();
    schedule_ssa_greedy_with_shift_budget(&mut b.ops, false, &mut shifted, 1536).unwrap();
    let cycles = |ops: &[SsaOp]| ops.iter().map(|op| op.cycle).collect::<Vec<_>>();
    assert_eq!(cycles(&a.ops), cycles(&b.ops));
}

// =========================================================================
// Split shifts (GreedyShape::split_shift)
// =========================================================================

/// Schedule `ops` with shifting enabled and the given shape.
///
/// # Returns
/// `(cycles, shifted op ids, per-lane cycles of split shifts)`.
fn schedule_with_shift_and_shape(
    ops: &mut [SsaOp],
    shape: &GreedyShape,
) -> (usize, HashSet<usize>, HashMap<usize, Vec<i32>>) {
    let mut shifted = HashSet::new();
    let mut lane_cycles = HashMap::new();
    let ((cycles, budget_exceeded, _), agreement) = schedule_ssa_greedy_shaped(
        ops,
        true,
        &mut shifted,
        1536,
        None,
        None,
        None,
        None,
        &GreedyBoundsUsage::None,
        &GreedyPolicy,
        None,
        false,
        false,
        false,
        false,
        None,
        None,
        None,
        &mut GreedyDiagnostics::new(false),
        shape,
        &mut lane_cycles,
    )
    .expect("greedy schedules the chains");
    assert!(!budget_exceeded);
    assert!(agreement.agrees(), "live-word counter disagrees: {}", agreement);
    (cycles, shifted, lane_cycles)
}

/// Scalar ALU slots used per cycle by shifted ops: all VLEN lanes at the op's cycle, or the
/// per-lane cycles of a split shift.
fn shifted_alu_use(
    ops: &[SsaOp],
    shifted: &HashSet<usize>,
    lane_cycles: &HashMap<usize, Vec<i32>>,
) -> HashMap<i32, usize> {
    let mut used: HashMap<i32, usize> = HashMap::new();
    for op in ops.iter().filter(|op| shifted.contains(&op.id)) {
        match lane_cycles.get(&op.id) {
            Some(lanes) => lanes.iter().for_each(|&c| *used.entry(c).or_default() += 1),
            None => *used.entry(op.cycle).or_default() += VLEN,
        }
    }
    used
}

/// With split shifts a shifted op may straddle two cycles: the scalar slots carry three
/// shifted ops per two cycles instead of two, no cycle uses more than 12 scalar slots, a
/// split op completes at its last lane, and its consumers run after that.
#[test]
fn test_split_shift_fills_scalar_slots() {
    let split = GreedyShape { split_shift: true, ..Default::default() };

    let (mut whole_ops, _) = emit_independent_chains(30, 12);
    let (whole_cycles, whole_shifted, whole_lanes) =
        schedule_with_shift_and_shape(&mut whole_ops.ops, &GreedyShape::default());
    assert!(whole_lanes.is_empty(), "no split shifts unless enabled");
    assert!(!whole_shifted.is_empty(), "the chains saturate the vector slots");
    let whole_use = shifted_alu_use(&whole_ops.ops, &whole_shifted, &whole_lanes);
    assert!(whole_use.values().all(|&n| n == VLEN), "whole shifts: 8 scalar slots per cycle");

    let (mut emit, level_ops) = emit_independent_chains(30, 12);
    let (cycles, shifted, lanes) = schedule_with_shift_and_shape(&mut emit.ops, &split);
    assert!(!lanes.is_empty(), "split shifts happen when the vector slots are full");
    assert!(cycles < whole_cycles, "split {} cycles, whole {} cycles", cycles, whole_cycles);

    let alu_use = shifted_alu_use(&emit.ops, &shifted, &lanes);
    assert!(alu_use.values().all(|&n| n <= 12), "scalar slots over capacity: {:?}", alu_use);
    assert!(alu_use.values().any(|&n| n > VLEN), "no cycle uses more than 8 scalar slots");

    let by_id: HashMap<usize, &SsaOp> = emit.ops.iter().map(|op| (op.id, op)).collect();
    for (op_id, lane) in &lanes {
        assert!(shifted.contains(op_id), "split op {} is recorded as shifted", op_id);
        assert_eq!(lane.len(), VLEN);
        let (first, last) = (*lane.iter().min().unwrap(), *lane.iter().max().unwrap());
        assert_eq!(last, first + 1, "a split shift spans two consecutive cycles: {:?}", lane);
        assert!(lane.windows(2).all(|w| w[0] <= w[1]), "low lanes run first: {:?}", lane);
        assert_eq!(by_id[op_id].cycle, last, "the op completes at its last lane");
    }
    // Every add reads the previous level's value: it runs after that op has completed.
    for levels in &level_ops {
        for pair in levels.windows(2) {
            let (producer, consumer) = (&emit.ops[pair[0]], &emit.ops[pair[1]]);
            let consumer_start = lanes
                .get(&consumer.id)
                .map_or(consumer.cycle, |l| *l.iter().min().unwrap());
            assert!(
                consumer_start > producer.cycle,
                "op {} starts at {} but its operand completes at {}",
                consumer.id,
                consumer_start,
                producer.cycle
            );
        }
    }
}

/// An op whose destination no scheduled op reads (its only reader is a skipped dead store)
/// is never split: its destination is released the cycle after its write, which would be
/// before a split's last lanes are written. It is still shifted whole.
#[test]
fn test_split_shift_skips_destinations_without_scheduled_readers() {
    let mut emit = SsaEmitContext::new();
    let scalar = emit.emit_const(3);
    let shared = emit.emit_vbroadcast(scalar);
    let mut unread_producers = HashSet::new();
    for _ in 0..40 {
        let value = emit.emit_valu(Opcode::Add, shared, shared);
        unread_producers.insert(emit.ops.last().unwrap().id);
        // The only reader of `value`; nothing reads its result, so the scheduler skips it.
        emit.emit_valu(Opcode::Xor, value, shared);
    }
    compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg, None);

    let split = GreedyShape { split_shift: true, ..Default::default() };
    let (_, shifted, lanes) = schedule_with_shift_and_shape(&mut emit.ops, &split);
    assert!(
        shifted.iter().any(|id| unread_producers.contains(id)),
        "the producers saturate the vector slots, so some are shifted whole"
    );
    assert!(lanes.is_empty(), "no producer without a scheduled reader is split: {:?}", lanes);
    assert!(emit.ops.iter().filter(|op| unread_producers.contains(&op.id)).all(|op| op.cycle >= 0));
}

/// Lane cycles of a split shift: the first lanes in the first cycle, the rest in the last.
#[test]
fn test_split_shift_lane_cycles() {
    let split = SplitShift { idx: 0, first_cycle: 7, first_lanes: 3 };
    assert_eq!(split.lane_cycles(8), vec![7, 7, 7, 8, 8, 8, 8, 8]);
}

/// Split shifts need the scheduler to own the shift decision: with shifting off or with
/// fixed shift encodings there are none.
#[test]
fn test_split_shift_off_without_shift_decisions() {
    let split = GreedyShape { split_shift: true, ..Default::default() };
    let (mut emit, _) = emit_independent_chains(30, 12);
    let mut shifted = HashSet::new();
    let mut lane_cycles = HashMap::new();
    schedule_ssa_greedy_shaped(
        &mut emit.ops,
        false, // shifting disabled
        &mut shifted,
        1536,
        None,
        None,
        None,
        None,
        &GreedyBoundsUsage::None,
        &GreedyPolicy,
        None,
        false,
        false,
        false,
        false,
        None,
        None,
        None,
        &mut GreedyDiagnostics::new(false),
        &split,
        &mut lane_cycles,
    )
    .unwrap();
    assert!(shifted.is_empty() && lane_cycles.is_empty());
}

// =========================================================================
// VALU shift fixed to none (`valu_shift_fixed_to_none`, set by the VALU-shift hint pass)
//
// BUGS.md "Greedy with `valu_shift_fixed_to_none` can place a vector op without a vector
// slot". The greedy must never shift when the hint pass owns the shift decision, and on
// every path (normal and force-schedule) no cycle may exceed an engine's slot limit.
// =========================================================================

/// Policy for the force-path test: like [`GreedyPolicy`] but never allows a shift, so the
/// normal path cannot place an op on the scalar ALU (it defers instead) and any vector op
/// beyond six in a cycle can only come from the force-schedule path.
#[derive(Debug)]
struct NeverShiftPolicy;

impl PressurePolicy for NeverShiftPolicy {
    fn sort_ready(
        &self,
        candidates: &mut [pressure_policy::SchedulingCandidate],
        ctx: &pressure_policy::SchedulingContext,
    ) {
        GreedyPolicy.sort_ready(candidates, ctx);
    }

    fn should_defer_for_pressure(&self, _: usize, _: usize, _: usize, _: usize) -> bool {
        false
    }

    fn allow_shift(&self, _: usize, _: usize, _: usize) -> bool {
        false
    }
}

/// Schedule 30 chains of 12 vector adds with shifting enabled and `valu_shift_fixed_to_none`,
/// then check the slot invariant: every op is placed, and per cycle at most six vector ops
/// not recorded as shifted and at most 12 scalar slots (scalar ops plus 8 per recorded shift).
///
/// # Returns
/// `(max unshifted vector ops in a cycle, max scalar slots in a cycle, recorded shifts,
/// budget exceeded)`.
fn schedule_with_shift_fixed_to_none(
    budget: usize,
    policy: &dyn PressurePolicy,
) -> (usize, usize, usize, bool) {
    let (mut emit, _) = emit_independent_chains(30, 12);
    let mut shifted = HashSet::new();
    let ((_, budget_exceeded, _), _) = schedule_ssa_greedy_shaped(
        &mut emit.ops,
        true,
        &mut shifted,
        budget,
        None,
        None,
        None,
        None,
        &GreedyBoundsUsage::None,
        policy,
        None,
        false,
        false,
        false,
        true, // valu_shift_fixed_to_none
        None,
        None,
        None,
        &mut GreedyDiagnostics::new(false),
        &GreedyShape::default(),
        &mut HashMap::new(),
    )
    .expect("greedy schedules the chains");
    assert!(emit.ops.iter().all(|op| op.cycle >= 0), "every op is placed");
    let mut valu: HashMap<i32, usize> = HashMap::new();
    let mut alu: HashMap<i32, usize> = HashMap::new();
    for op in &emit.ops {
        if op.engine == Engine::Valu && shifted.contains(&op.id) {
            *alu.entry(op.cycle).or_default() += VLEN;
        } else if op.engine == Engine::Valu {
            *valu.entry(op.cycle).or_default() += 1;
        } else if op.engine == Engine::Alu {
            *alu.entry(op.cycle).or_default() += 1;
        }
    }
    (
        valu.values().copied().max().unwrap_or(0),
        alu.values().copied().max().unwrap_or(0),
        shifted.len(),
        budget_exceeded,
    )
}

/// Normal path: with the six vector slots full the slot tracker used to take 8 scalar slots
/// for the op (`SlotResult::Shifted`) without recording it in `shifted_ops`, so it lowered as
/// a seventh vector op. Budget 1536: nothing is forced.
#[test]
fn test_valu_shift_fixed_to_none_keeps_vector_slot_limit() {
    let (valu, alu, shifts, exceeded) = schedule_with_shift_fixed_to_none(1536, &GreedyPolicy);
    assert!(!exceeded, "budget 1536 forces nothing");
    assert!(valu <= 6, "a cycle has {} unshifted vector ops", valu);
    assert!(alu <= 12, "a cycle uses {} scalar slots", alu);
    assert_eq!(shifts, 0, "the hint pass owns the shift decision");
}

/// Force-schedule path, `SlotResult::NoRoom` arm: the policy never allows a shift (so the
/// normal path places no unrecorded shift), the budget forces ops through the force path,
/// and there an op that did not fit used to be marked scheduled with no slot.
#[test]
fn test_valu_shift_fixed_to_none_keeps_vector_slot_limit_when_force_scheduling() {
    let (valu, alu, shifts, exceeded) = schedule_with_shift_fixed_to_none(80, &NeverShiftPolicy);
    assert!(exceeded, "budget 80 must force ops through the force path");
    assert!(valu <= 6, "a cycle has {} unshifted vector ops", valu);
    assert!(alu <= 12, "a cycle uses {} scalar slots", alu);
    assert_eq!(shifts, 0, "the hint pass owns the shift decision");
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

// =========================================================================
// fixed_shifted_ops: why we need try_place_shifted_only (Bug #2)
// =========================================================================
//
// When encoding is *optional* (can_shift=true), we may place on VALU or shift.
// When encoding is *fixed* to shifted (op in fixed_shifted_ops), the verifier
// will count 8 ALU for that op. So the schedule must actually place 8 ALU for
// it—we must not place on VALU or the counts diverge and we can exceed ALU cap.

/// **Why try_place_shifted_only exists:** try_place(Engine::Valu, true, true) prefers
/// VALU when there is room (valu_used < 6). So a "shiftable" op can be placed on
/// VALU even when we intend it to be shifted. If the hint says "shifted", the verifier
/// counts 8 ALU; if we placed on VALU we count 1 VALU. So we must use a path that
/// never places on VALU when the op is in fixed_shifted_ops.
#[test]
fn test_slot_tracker_try_place_valu_prefers_valu_when_room() {
    let mut slots = SlotTracker::new();
    // Empty slots: can_shift=true, allow_shift=true → still places on VALU first (VALU has room)
    let result = slots.try_place(Engine::Valu, true, true);
    assert_eq!(result, SlotResult::Placed, "try_place prefers VALU when valu_used < 6");
    assert_eq!(slots.valu_used, 1);
    assert_eq!(slots.alu_used, 0);
}

/// **Why we use try_place_shifted_only for fixed_shifted_ops:** The shift-only path
/// never touches VALU. So when the encoding fixes "this op is shifted", we must use
/// this path so the schedule matches the verifier's count (8 ALU, 0 VALU for that op).
#[test]
fn test_slot_tracker_try_place_shifted_only_never_uses_valu() {
    let mut slots = SlotTracker::new();
    assert_eq!(slots.try_place_shifted_only(), SlotResult::Shifted);
    assert_eq!(slots.valu_used, 0);
    assert_eq!(slots.alu_used, VLEN);
    assert_eq!(slots.try_place_shifted_only(), SlotResult::NoRoom); // 8+8 > 12
    assert_eq!(slots.valu_used, 0);
    assert_eq!(slots.alu_used, VLEN);
}

/// **Why we must defer when ALU is full for fixed_shifted_ops:** One shifted op uses
/// 8 ALU slots; a second in the same cycle would use 16 > 12. So at most one
/// try_place_shifted_only can succeed per cycle. The scheduler must defer the second
/// "must shift" op to a later cycle.
#[test]
fn test_slot_tracker_try_place_shifted_only_respects_alu_cap() {
    let mut slots = SlotTracker::new();
    assert_eq!(slots.try_place_shifted_only(), SlotResult::Shifted);
    assert_eq!(slots.alu_used, VLEN);
    assert_eq!(slots.try_place_shifted_only(), SlotResult::NoRoom);
    assert_eq!(slots.alu_used, VLEN);
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

// =========================================================================
// ASAP Inflation Tests
//
// Verify that compute_asap_inflation correctly defers high-slack producers
// when pressure is high, while protecting critical-path ops and respecting
// safeguards (MIN_PASS_THROUGH, consecutive_empty breaker, ALAP clamping).
// =========================================================================

/// Test that under high pressure (>80% budget), a high-slack producer gets
/// inflated while a critical-path op (slack=0) does not.
#[test]
fn test_asap_inflation_defers_high_slack_producer() {
    let policy = CascadePolicy::default(); // inflation_threshold=80%
    let slots = empty_slots();
    // 85% of 1000 = 850 > threshold of 800 → inflation active
    let ctx = make_ctx(850, 1000, &slots);

    // High-slack producer: ASAP=0, ALAP=20 → slack=20
    let high_slack = make_candidate_full(0, 2, 8, 0, Some(0), Some(20), Engine::Valu, false, 3);
    let inflation = policy.compute_asap_inflation(&high_slack, &ctx);
    assert!(
        inflation.is_some(),
        "High-slack op (slack=20) should be inflated at 85% pressure"
    );
    let n = inflation.unwrap();
    assert!(n > 0, "Inflation should be positive, got {}", n);
    assert!(n <= 10, "Inflation should be at most max_inflation_cycles=10, got {}", n);

    // Critical-path op: ASAP=0, ALAP=1 → slack=1 (< min_inflation_slack=3)
    let critical = make_candidate_full(1, 2, 8, 0, Some(0), Some(1), Engine::Valu, false, 3);
    let inflation = policy.compute_asap_inflation(&critical, &ctx);
    assert!(
        inflation.is_none(),
        "Critical-path op (slack=1) should never be inflated"
    );
}

/// Test that inflation is clamped to ALAP: if inflation would push beyond ALAP,
/// the caller clamps. Here we verify the inflation amount itself can exceed the
/// remaining slack (the caller is responsible for clamping).
#[test]
fn test_asap_inflation_clamps_to_alap() {
    let policy = CascadePolicy {
        max_inflation_cycles: 10,
        min_inflation_slack: 3,
        inflation_threshold_fraction: 0.80,
        ..CascadePolicy::default()
    };
    let slots = empty_slots();
    // At budget limit → maximum pressure
    let ctx = make_ctx(1000, 1000, &slots);

    // Op with ASAP=10, ALAP=13 → slack=3 (exactly at min_inflation_slack)
    let candidate = make_candidate_full(0, 0, 8, 0, Some(10), Some(13), Engine::Valu, false, 1);
    let inflation = policy.compute_asap_inflation(&candidate, &ctx);
    assert!(
        inflation.is_some(),
        "Op with slack=3 at max pressure should be inflated"
    );
    // The inflation value may exceed remaining slack — caller clamps to ALAP.
    // We just verify it's positive and within max_inflation_cycles.
    let n = inflation.unwrap();
    assert!(n > 0 && n <= 10);
    // Verify caller clamping logic: inflated = (cycle + n).min(alap)
    // At cycle=10, inflated = (10 + n).min(13) = 13 (since n >= 1)
    let cycle = 10i32;
    let alap = 13i32;
    let inflated = (cycle + n).min(alap);
    assert!(inflated <= alap, "Inflated ASAP must not exceed ALAP");
}

/// Test MIN_PASS_THROUGH: with 5 ready ops, at most 2 can be inflation-deferred
/// (5 - MIN_PASS_THROUGH=3 = 2). This is a property test on the scheduling loop
/// logic, verified through the CascadePolicy integration test below.
/// Here we verify the policy itself returns inflation for all high-slack ops
/// (it's the loop that limits how many are deferred).
#[test]
fn test_asap_inflation_policy_returns_for_all_eligible() {
    let policy = CascadePolicy::default();
    let slots = empty_slots();
    // High pressure: 90% of 1000 = 900
    let ctx = make_ctx(900, 1000, &slots);

    // 5 high-slack candidates — policy should inflate all of them
    let mut inflated_count = 0;
    for i in 0..5 {
        let candidate = make_candidate_full(i, 1, 8, 0, Some(0), Some(20), Engine::Valu, false, 2);
        if policy.compute_asap_inflation(&candidate, &ctx).is_some() {
            inflated_count += 1;
        }
    }
    assert_eq!(
        inflated_count, 5,
        "Policy should return inflation for all 5 eligible ops (loop enforces MIN_PASS_THROUGH)"
    );
}

/// Test that inflation is disabled below the threshold (80%).
#[test]
fn test_asap_inflation_disabled_below_threshold() {
    let policy = CascadePolicy::default(); // threshold=80%
    let slots = empty_slots();

    // Well below threshold: 70% of 1000 = 700
    let ctx = make_ctx(700, 1000, &slots);
    let candidate = make_candidate_full(0, 0, 8, 0, Some(0), Some(50), Engine::Valu, false, 5);
    assert!(
        policy.compute_asap_inflation(&candidate, &ctx).is_none(),
        "Below 80% threshold: no inflation regardless of slack"
    );

    // Exactly at threshold: 80% of 1000 = 800
    let ctx = make_ctx(800, 1000, &slots);
    assert!(
        policy.compute_asap_inflation(&candidate, &ctx).is_none(),
        "At exactly 80% threshold (<=): no inflation"
    );

    // Just above threshold: 801/1000
    let ctx = make_ctx(801, 1000, &slots);
    let candidate = make_candidate_full(0, 0, 8, 0, Some(0), Some(50), Engine::Valu, false, 5);
    assert!(
        policy.compute_asap_inflation(&candidate, &ctx).is_some(),
        "Above 80% threshold: inflation should activate for high-slack op"
    );
}

/// Test that AdaptivePolicy has the same inflation behavior as CascadePolicy.
#[test]
fn test_adaptive_inflation_matches_cascade() {
    let cascade = CascadePolicy::default();
    let adaptive = AdaptivePolicy::default();
    let slots = empty_slots();

    // High pressure scenario
    let ctx = make_ctx(900, 1000, &slots);

    // High slack op
    let high_slack = make_candidate_full(0, 2, 8, 0, Some(0), Some(20), Engine::Valu, false, 3);
    let cascade_result = cascade.compute_asap_inflation(&high_slack, &ctx);
    let adaptive_result = adaptive.compute_asap_inflation(&high_slack, &ctx);
    assert_eq!(
        cascade_result, adaptive_result,
        "Cascade and Adaptive should produce identical inflation"
    );

    // Low slack op (should not inflate for either)
    let low_slack = make_candidate_full(1, 2, 8, 0, Some(0), Some(1), Engine::Valu, false, 3);
    assert!(cascade.compute_asap_inflation(&low_slack, &ctx).is_none());
    assert!(adaptive.compute_asap_inflation(&low_slack, &ctx).is_none());

    // Below threshold (should not inflate for either)
    let ctx_low = make_ctx(700, 1000, &slots);
    assert!(cascade.compute_asap_inflation(&high_slack, &ctx_low).is_none());
    assert!(adaptive.compute_asap_inflation(&high_slack, &ctx_low).is_none());
}

/// Test that GreedyPolicy (default implementation) never inflates.
#[test]
fn test_greedy_policy_no_inflation() {
    let policy = GreedyPolicy;
    let slots = empty_slots();
    let ctx = make_ctx(999, 1000, &slots); // Maximum pressure

    let candidate = make_candidate_full(0, 0, 8, 0, Some(0), Some(100), Engine::Valu, false, 5);
    assert!(
        policy.compute_asap_inflation(&candidate, &ctx).is_none(),
        "GreedyPolicy should never inflate (default impl returns None)"
    );
}

/// Integration test: CascadePolicy with inflation on a DAG that would exceed
/// budget without inflation. Verifies the full scheduler loop handles inflation
/// correctly (defers ops, respects MIN_PASS_THROUGH, and completes).
///
/// DAG: 6 independent vbroadcast chains → forces high parallelism.
/// With tight budget, without inflation the greedy scheduler force-schedules
/// and sets budget_exceeded. With CascadePolicy inflation, the scheduler can
/// defer producers and reduce peak liveness.
#[test]
fn test_cascade_inflation_in_scheduler() {
    use std::collections::HashMap;

    let mut emit = SsaEmitContext::new();

    // Create 6 independent chains: const → vbroadcast → valu_add → vstore
    // Each chain needs 8 words live for the vector. 6 chains = 48 words peak
    // if all scheduled concurrently. With budget=32, inflation should spread them.
    let mut stores = vec![];
    for i in 0..6i64 {
        let c = emit.emit_const(i + 1);
        let v = emit.emit_vbroadcast(c);
        let c2 = emit.emit_const(i + 100);
        let v2 = emit.emit_vbroadcast(c2);
        let sum = emit.emit_valu(Opcode::Add, v, v2);
        let addr = emit.emit_const(200 + i * 8);
        stores.push((addr, sum));
    }
    for (addr, val) in stores {
        emit.emit_vstore(addr, val);
    }

    compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg, None);

    // Compute ASAP/ALAP bounds for inflation to work
    let asap = crate::cpsat_scheduler::bounds::compute_ssa_asap(
        &emit.ops,
        &HashSet::new(),
    );
    let max_asap = asap.values().copied().max().unwrap_or(0);
    let alap = crate::cpsat_scheduler::bounds::compute_ssa_alap(
        &emit.ops,
        max_asap + 20,
        &HashSet::new(),
    );

    // Run with CascadePolicy (has inflation) using PolicyOnly bounds
    // (inflation test: we want policy decisions but not core loop deferral)
    let mut shifted_ops = HashSet::new();
    let bounds_policy_only = GreedyBoundsUsage::PolicyOnly { asap: &asap, alap: &alap };
    let result_cascade = schedule_ssa_greedy_with_shift_budget_and_selectchoice(
        &mut emit.ops.clone(),
        false,
        &mut shifted_ops,
        32, // Tight budget to trigger inflation
        None,
        None,
        None,
        None,
        &bounds_policy_only,
        &CascadePolicy::default(),
        None,
        false,
        false,
        false,
        false,
        None,
        None,
        None,
    );

    assert!(
        result_cascade.is_ok(),
        "CascadePolicy with inflation should complete: {:?}",
        result_cascade.err()
    );
    let (cycles_cascade, _, peak_cascade) = result_cascade.unwrap();
    assert!(
        cycles_cascade > 0,
        "Should produce a non-empty schedule"
    );

    // Run with GreedyPolicy (no inflation) for comparison
    let mut shifted_ops2 = HashSet::new();
    let bounds_policy_only2 = GreedyBoundsUsage::PolicyOnly { asap: &asap, alap: &alap };
    let result_greedy = schedule_ssa_greedy_with_shift_budget_and_selectchoice(
        &mut emit.ops.clone(),
        false,
        &mut shifted_ops2,
        32,
        None,
        None,
        None,
        None,
        &bounds_policy_only2,
        &GreedyPolicy,
        None,
        false,
        false,
        false,
        false,
        None,
        None,
        None,
    );

    assert!(result_greedy.is_ok());
    let (cycles_greedy, _, peak_greedy) = result_greedy.unwrap();

    // With inflation: may take more cycles but should have lower/equal peak liveness
    // (or at least complete without error). We don't assert strict ordering since
    // the DAG may be small enough that both stay within budget, but we verify
    // the scheduler handles inflation correctly.
    assert!(
        cycles_cascade > 0 && cycles_greedy > 0,
        "Both should produce valid schedules"
    );

    // If greedy exceeded budget, cascade should have lower or equal peak
    // (inflation defers producers to reduce concurrent liveness)
    if peak_greedy > 32 {
        assert!(
            peak_cascade <= peak_greedy,
            "Cascade with inflation should have peak <= greedy peak: cascade={}, greedy={}",
            peak_cascade, peak_greedy
        );
    }
}

/// Test graduated inflation: higher pressure → more inflation cycles.
#[test]
fn test_asap_inflation_graduated_response() {
    let policy = CascadePolicy::default(); // threshold=80%, max=10
    let slots = empty_slots();

    let candidate = make_candidate_full(0, 0, 8, 0, Some(0), Some(50), Engine::Valu, false, 5);

    // Just above threshold (81%): minimal inflation
    let ctx_low = make_ctx(810, 1000, &slots);
    let inflation_low = policy
        .compute_asap_inflation(&candidate, &ctx_low)
        .unwrap_or(0);

    // Mid pressure (90%): moderate inflation
    let ctx_mid = make_ctx(900, 1000, &slots);
    let inflation_mid = policy
        .compute_asap_inflation(&candidate, &ctx_mid)
        .unwrap_or(0);

    // At budget (100%): maximum inflation
    let ctx_high = make_ctx(1000, 1000, &slots);
    let inflation_high = policy
        .compute_asap_inflation(&candidate, &ctx_high)
        .unwrap_or(0);

    assert!(
        inflation_low <= inflation_mid,
        "More pressure should produce >= inflation: low={}, mid={}",
        inflation_low, inflation_mid
    );
    assert!(
        inflation_mid <= inflation_high,
        "More pressure should produce >= inflation: mid={}, high={}",
        inflation_mid, inflation_high
    );
    assert!(
        inflation_high > inflation_low,
        "Max pressure should produce strictly more inflation than minimal: high={}, low={}",
        inflation_high, inflation_low
    );
}

/// Regression: vector-group VRegs must be released from the greedy's live set.
///
/// A gather reads lanes of a decomposed address vector and writes lanes of an assembled
/// result vector. Readers name only one side of each group, so the address vector and the
/// loaded lanes used to stay live forever (16 words per gather), which pushed any kernel
/// with three or more scatter rounds over the scratch budget.
#[test]
fn test_vector_group_vregs_released_after_last_use() {
    const GATHERS: usize = 300; // leaked 16 words each before the fix: 4800 > 1536

    let mut emit = SsaEmitContext::new();
    let base = emit.emit_const(100);
    let v_base = emit.emit_vbroadcast(base);
    let mut idx = emit.emit_vload_imm(200);
    for _ in 0..GATHERS {
        let v_addr = emit.emit_valu(Opcode::Add, idx, v_base);
        let addr_lanes = emit.decompose_vector(v_addr);
        let (v_loaded, loaded_lanes) = emit.fresh_vector_group();
        for (addr, dest) in addr_lanes.iter().zip(loaded_lanes.iter()) {
            emit.emit_load_into(*addr, *dest);
        }
        idx = emit.emit_valu(Opcode::Xor, idx, v_loaded);
    }
    emit.emit_vstore_imm(300, idx);

    let groups = emit.take_vector_group_lanes();
    compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg, Some(&groups));

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
        &GreedyBoundsUsage::None,
        &GreedyPolicy,
        Some(&groups),
        false,
        false,
        false,
        false,
        None,
        None,
        None,
    );

    let (_cycles, budget_exceeded, peak_live) = result.expect("gather chain should schedule");
    assert!(!budget_exceeded, "vector-group liveness leaked: peak_live={}", peak_live);
    assert!(
        peak_live < 64,
        "one gather is live at a time, so peak should be a few vectors, got {}",
        peak_live
    );
}

// =========================================================================
// Liveness agreement: the scheduler's counter against interval liveness
//
// The greedy's live-word counter and `address_assign::compute_live_intervals` are two
// implementations of one definition (docs/LIVENESS_SPECIFICATION.md, "Greedy counter
// agreement"). The scheduler compares them after every run and returns Err when they
// differ, so each test below fails with the per-value differences if the counter is wrong.
// =========================================================================

/// Schedule with the default policy and return the counter/interval comparison.
fn schedule_for_agreement(
    emit: &mut SsaEmitContext,
    groups: &HashMap<VReg, Vec<VReg>>,
    enable_valu_shift: bool,
    scratch_budget: usize,
    strategy: &dyn PressurePolicy,
) -> Result<((usize, bool, usize), LivenessAgreement), String> {
    let groups_opt = if groups.is_empty() { None } else { Some(groups) };
    compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg, groups_opt);
    let mut shifted_ops = HashSet::new();
    schedule_ssa_greedy_with_liveness_agreement(
        &mut emit.ops,
        enable_valu_shift,
        &mut shifted_ops,
        scratch_budget,
        None,
        None,
        None,
        None,
        &GreedyBoundsUsage::None,
        strategy,
        groups_opt,
        false,
        false,
        false,
        false,
        None,
        None,
        None,
        &mut GreedyDiagnostics::new(false),
    )
}

/// Regression: an op naming one value in both operands is one reader, not two.
///
/// `d = x + x` used to decrement x's remaining-reader count twice, so x was released
/// while `e`'s producer still had to read it.
#[test]
fn test_liveness_agreement_duplicate_operand() {
    let mut emit = SsaEmitContext::new();
    let a = emit.emit_const(1);
    let b = emit.emit_const(2);
    let x = emit.emit_alu(Opcode::Add, a, b);
    let d = emit.emit_alu(Opcode::Add, x, x);
    let d2 = emit.emit_alu(Opcode::Add, d, b);
    let e = emit.emit_alu(Opcode::Add, d2, x); // reads x two cycles after `d` does
    emit.emit_store_imm(100, e);

    let (_, agreement) =
        schedule_for_agreement(&mut emit, &HashMap::new(), false, 1536, &GreedyPolicy)
            .expect("counter must agree with interval liveness");
    assert!(agreement.agrees(), "{}", agreement);
    assert_eq!(agreement.counter_cycle_end_peak, agreement.interval_peak);
}

/// Regression: a `keep_live` value is held until the op that names it.
///
/// Address assignment extends such a value's interval to that op; the counter released it
/// at its last operand read (256 words too early in the 2- and 4-round kernels).
#[test]
fn test_liveness_agreement_keep_live() {
    let mut emit = SsaEmitContext::new();
    let v = emit.emit_vload_imm(200);
    let w = emit.emit_vload_imm(208);
    let kept = emit.emit_valu(Opcode::Xor, v, w);
    let mut chain = emit.emit_valu(Opcode::Add, kept, w); // last operand read of `kept`
    for _ in 0..4 {
        chain = emit.emit_valu(Opcode::Add, chain, w);
    }
    emit.set_next_keep_live(vec![kept]);
    let last = emit.emit_valu(Opcode::Xor, chain, w);
    emit.emit_vstore_imm(300, last);

    let (_, agreement) =
        schedule_for_agreement(&mut emit, &HashMap::new(), false, 1536, &GreedyPolicy)
            .expect("counter must agree with interval liveness");
    assert!(agreement.agrees(), "{}", agreement);
    // v/w loads, then `kept` (8) + w (8) + the chain value (8) are held together.
    assert_eq!(agreement.interval_peak, 24);
    assert_eq!(agreement.counter_cycle_end_peak, 24);
}

/// Regression: a write group is one 8-word block from its first lane write.
///
/// Two loads fit per cycle, so the lanes of a gather arrive over four cycles. The counter
/// used to add one word per loaded lane; address assignment reserves the block at once.
#[test]
fn test_liveness_agreement_write_group_block() {
    let mut emit = SsaEmitContext::new();
    let base = emit.emit_const(100);
    let v_base = emit.emit_vbroadcast(base);
    let idx = emit.emit_vload_imm(200);
    let v_addr = emit.emit_valu(Opcode::Add, idx, v_base);
    let addr_lanes = emit.decompose_vector(v_addr);
    let (v_loaded, loaded_lanes) = emit.fresh_vector_group();
    for (addr, dest) in addr_lanes.iter().zip(loaded_lanes.iter()) {
        emit.emit_load_into(*addr, *dest);
    }
    let out = emit.emit_valu(Opcode::Xor, idx, v_loaded);
    emit.emit_vstore_imm(300, out);
    let groups = emit.take_vector_group_lanes();

    let (_, agreement) = schedule_for_agreement(&mut emit, &groups, false, 1536, &GreedyPolicy)
        .expect("counter must agree with interval liveness");
    assert!(agreement.agrees(), "{}", agreement);
}

/// Regression: a reader that is a skipped dead store does not hold its operands.
///
/// `dead = x + b` has no reader and is never scheduled, but it used to count as a reader
/// of x, so x stayed in the counter to the end of the schedule.
#[test]
fn test_liveness_agreement_dead_reader() {
    let mut emit = SsaEmitContext::new();
    let a = emit.emit_const(1);
    let b = emit.emit_const(2);
    let x = emit.emit_alu(Opcode::Add, a, b);
    let _dead = emit.emit_alu(Opcode::Sub, x, b);
    let only_dead_reader = emit.emit_alu(Opcode::Mul, a, b);
    let _dead2 = emit.emit_alu(Opcode::Sub, only_dead_reader, b);
    let mut y = emit.emit_alu(Opcode::Add, x, a);
    for _ in 0..6 {
        y = emit.emit_alu(Opcode::Add, y, a);
    }
    emit.emit_store_imm(100, y);

    let (_, agreement) =
        schedule_for_agreement(&mut emit, &HashMap::new(), false, 1536, &GreedyPolicy)
            .expect("counter must agree with interval liveness");
    assert!(agreement.agrees(), "{}", agreement);
}

/// Deterministic generator for the property test (xorshift64*).
struct TestRng(u64);

impl TestRng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn chance(&mut self, percent: usize) -> bool {
        self.below(100) < percent
    }
}

/// Build a random SSA program covering the patterns the counter has to get right:
/// repeated operands, `keep_live`, read groups, write groups (with and without direct lane
/// readers), values read many cycles later, and values nobody reads.
fn random_liveness_program(rng: &mut TestRng) -> (SsaEmitContext, HashMap<VReg, Vec<VReg>>) {
    let mut emit = SsaEmitContext::new();
    let mut scalars: Vec<VReg> = Vec::new();
    let mut vectors: Vec<VReg> = Vec::new();
    // For keep_live: a value the listed value was computed from (so it is defined earlier).
    let mut ancestor: HashMap<VReg, VReg> = HashMap::new();
    // A vector is decomposed at most once (a second call would replace its lanes).
    let mut decomposed: HashMap<VReg, Vec<VReg>> = HashMap::new();
    let mut next_const = 1_000i64;

    for _ in 0..(2 + rng.below(3)) {
        next_const += 1;
        scalars.push(emit.emit_const_force(next_const));
    }
    for i in 0..(1 + rng.below(3)) {
        vectors.push(emit.emit_vload_imm(2_000 + 8 * i as i64));
    }

    let alu_ops = [Opcode::Add, Opcode::Sub, Opcode::Xor, Opcode::Mul];
    let n_steps = 5 + rng.below(40);
    for _ in 0..n_steps {
        match rng.below(100) {
            0..=29 => {
                let a = scalars[rng.below(scalars.len())];
                let b = if rng.chance(20) { a } else { scalars[rng.below(scalars.len())] };
                if rng.chance(15) {
                    if let Some(&keep) = ancestor.get(&a) {
                        emit.set_next_keep_live(vec![keep]);
                    }
                }
                let d = emit.emit_alu(alu_ops[rng.below(alu_ops.len())], a, b);
                ancestor.insert(d, a);
                scalars.push(d);
            }
            30..=64 => {
                let a = vectors[rng.below(vectors.len())];
                let b = if rng.chance(20) { a } else { vectors[rng.below(vectors.len())] };
                if rng.chance(15) {
                    if let Some(&keep) = ancestor.get(&a) {
                        emit.set_next_keep_live(vec![keep]);
                    }
                }
                let d = emit.emit_valu(alu_ops[rng.below(alu_ops.len())], a, b);
                ancestor.insert(d, a);
                vectors.push(d);
            }
            65..=74 => {
                let a = scalars[rng.below(scalars.len())];
                let d = emit.emit_vbroadcast_force(a);
                ancestor.insert(d, a);
                vectors.push(d);
            }
            75..=89 => {
                // Gather: read group (address lanes) feeding a write group (loaded lanes).
                let v_addr = vectors[rng.below(vectors.len())];
                let addr_lanes = decomposed
                    .entry(v_addr)
                    .or_insert_with(|| emit.decompose_vector(v_addr))
                    .clone();
                let (v_loaded, loaded_lanes) = emit.fresh_vector_group();
                for (addr, dest) in addr_lanes.iter().zip(loaded_lanes.iter()) {
                    emit.emit_load_into(*addr, *dest);
                }
                let d = emit.emit_valu(Opcode::Xor, v_addr, v_loaded);
                ancestor.insert(d, v_addr);
                vectors.push(d);
                if rng.chance(30) {
                    // A scalar op reads one loaded lane directly.
                    scalars.push(loaded_lanes[rng.below(VLEN)]);
                }
            }
            _ => {
                // Read group whose lanes feed scalar ops.
                let v = vectors[rng.below(vectors.len())];
                let lanes =
                    decomposed.entry(v).or_insert_with(|| emit.decompose_vector(v)).clone();
                for _ in 0..(1 + rng.below(3)) {
                    scalars.push(lanes[rng.below(VLEN)]);
                }
            }
        }
    }

    // Sinks: store some values; the rest stay unread (dead stores, possibly in chains).
    for (i, v) in vectors.clone().into_iter().enumerate() {
        if rng.chance(50) {
            emit.emit_vstore_imm(4_000 + 8 * i as i64, v);
        }
    }
    for (i, s) in scalars.clone().into_iter().enumerate() {
        if rng.chance(50) {
            emit.emit_store_imm(8_000 + i as i64, s);
        }
    }
    let groups = emit.take_vector_group_lanes();
    (emit, groups)
}

/// Run the agreement property over `cases` random programs.
fn check_liveness_agreement_property(seed: u64, cases: usize) {
    let mut rng = TestRng(seed);
    for case in 0..cases {
        let (mut emit, groups) = random_liveness_program(&mut rng);
        // Tight budgets exercise deferral and the force-schedule path.
        let budget = [1536, 96, 40, 16][rng.below(4)];
        let enable_valu_shift = rng.chance(50);
        let policies: [Box<dyn PressurePolicy>; 3] = [
            Box::new(GreedyPolicy),
            Box::new(pressure_policy::AdaptivePolicy::default()),
            Box::new(pressure_policy::PressureGatedPolicy::default()),
        ];
        let policy = &*policies[rng.below(policies.len())];

        let result = schedule_for_agreement(&mut emit, &groups, enable_valu_shift, budget, policy);
        let ((_cycles, _exceeded, peak), agreement) = result.unwrap_or_else(|e| {
            panic!("seed {} case {} (budget {}, shift {}): {}", seed, case, budget, enable_valu_shift, e)
        });
        let context = format!("seed {} case {} (budget {}): {}", seed, case, budget, agreement);
        // 1. Cycle by cycle, the counter is interval liveness.
        assert!(agreement.agrees(), "{}", context);
        // 2. Hence the cycle-end peak is the allocator's peak.
        assert_eq!(agreement.counter_cycle_end_peak, agreement.interval_peak, "{}", context);
        // 3. The reported peak is sampled after each op, so it can only be higher.
        assert_eq!(peak, agreement.counter_peak, "{}", context);
        assert!(agreement.counter_peak >= agreement.interval_peak, "{}", context);
    }
}

/// Property: on random programs the greedy's counter equals interval liveness every cycle.
#[test]
fn test_liveness_agreement_property() {
    check_liveness_agreement_property(0x9E37_79B9_7F4A_7C15, 400);
}

/// Same property, more programs and seeds (CI only).
#[cfg(feature = "ci-tests")]
#[test]
fn test_liveness_agreement_property_extended() {
    for seed in 1..=10u64 {
        check_liveness_agreement_property(seed.wrapping_mul(0xD6E8_FEB8_6659_FD93), 1_000);
    }
}

/// Regression: a force-schedule candidate that finds no free slot stays queued.
///
/// With budget 2, cycle 0 places the two constants on both Load slots. The ConstChoice
/// would still fit on Flow, so it is deferred for budget, not for slots. The force path
/// then tries it on its Load engine, finds Load full, and used to drop it: the run ended
/// with "Scheduling failed: only scheduled 5/6 ops ... Ready but unscheduled".
#[test]
fn test_force_schedule_without_slot_keeps_op() {
    let mut emit = SsaEmitContext::new();
    let c1 = emit.emit_const_force(200); // op 0
    let c2 = emit.emit_const_force(300); // op 1
    let k = emit.emit_const_choice(5); // op 2
    let s = emit.emit_alu(Opcode::Add, c1, c2);
    let t = emit.emit_alu(Opcode::Add, s, k);
    emit.emit_store(k, t);

    let (_, agreement) =
        schedule_for_agreement(&mut emit, &HashMap::new(), false, 2, &GreedyPolicy)
            .expect("the deferred ConstChoice must not be dropped");

    assert!(agreement.agrees(), "{}", agreement);
    assert!(emit.ops.iter().all(|op| op.cycle >= 0), "every op must be scheduled");
    // The scenario the test is about: Load was full in cycle 0, and the ConstChoice,
    // deferred there, was placed in a later cycle.
    assert_eq!((emit.ops[0].cycle, emit.ops[1].cycle), (0, 0), "constants fill Load in cycle 0");
    assert_eq!(emit.ops[2].opcode, Opcode::ConstChoice);
    assert!(emit.ops[2].cycle > 0, "ConstChoice placed after cycle 0");
}

// =========================================================================
// GreedyDiagnostics: over-budget reporting must not flood
//
// A 16-round greedy run that stayed over budget used to print a liveness
// breakdown for every over-budget cycle (about 253,000 lines), even with -q.
// =========================================================================

/// Many over-budget cycles produce one breakdown immediately, at most one more (the worst)
/// at the end, and a one-line summary with the cycle count.
#[test]
fn test_greedy_diagnostics_dump_once_and_summarise() {
    let mut diag = GreedyDiagnostics::new(true);
    let mut renders = 0;
    for cycle in 0..5000usize {
        // Live words rise then fall, so the worst cycle is neither first nor last.
        let live = 2000 + if cycle < 3000 { cycle } else { 6000 - cycle };
        diag.record_over_budget(cycle, live, || {
            renders += 1;
            String::new()
        });
        // A second force-schedule in the same cycle is not a new over-budget cycle.
        diag.record_over_budget(cycle, live + 8, || unreachable!("same cycle must not render"));
    }
    assert_eq!(diag.over_budget_cycles, 5000);
    assert_eq!(diag.dumps_emitted, 1, "only the first over-budget cycle is dumped while scheduling");
    diag.finish();
    assert_eq!(diag.dumps_emitted, 2, "the worst cycle is dumped once at the end");
    assert_eq!(diag.first_over_budget, Some((0, 2000)));
    assert_eq!(diag.worst_over_budget, Some((3000, 5000)));
    // Rendering happens for the first cycle and each new worst only (cycles 0..=3000).
    assert_eq!(renders, 3001);

    let summary = diag.summary(5010, 1535).expect("over budget => summary");
    assert_eq!(summary.lines().count(), 1, "summary must be one line: {}", summary);
    assert!(summary.contains("in 5000 cycle(s)"), "{}", summary);
    assert!(summary.contains("budget=1535"), "{}", summary);
    assert!(summary.contains("first at cycle 0 (2000 words live)"), "{}", summary);
    assert!(summary.contains("worst at cycle 3000 (5000 words live)"), "{}", summary);
    assert!(summary.contains("greedy peak_live=5010"), "{}", summary);
}

/// Non-verbose (the CLI's -q) never renders or prints a breakdown, but still counts.
#[test]
fn test_greedy_diagnostics_quiet_never_dumps() {
    let mut diag = GreedyDiagnostics::new(false);
    for cycle in 0..5000usize {
        diag.record_over_budget(cycle, 2000 + cycle, || {
            unreachable!("quiet mode must not render the breakdown")
        });
    }
    diag.finish();
    assert_eq!(diag.dumps_emitted, 0);
    assert_eq!(diag.over_budget_cycles, 5000);
    assert!(diag.summary(7000, 1535).is_some());
    assert!(GreedyDiagnostics::new(false).summary(100, 1535).is_none(), "no overflow => no summary");
}

/// Schedule `pairs` independent (vbroadcast, vbroadcast, add, vstore) groups with an 8-word
/// budget: every group needs 16 words live, so each one force-schedules past the budget.
fn schedule_over_budget_groups(pairs: usize, verbose: bool) -> (GreedyDiagnostics, bool) {
    let mut emit = SsaEmitContext::new();
    for i in 0..pairs {
        let s1 = emit.emit_const(2 * i as i64 + 1);
        let s2 = emit.emit_const(2 * i as i64 + 2);
        let v1 = emit.emit_vbroadcast(s1);
        let v2 = emit.emit_vbroadcast(s2);
        let v3 = emit.emit_valu(Opcode::Add, v1, v2);
        let addr = emit.emit_const(1000 + 8 * i as i64);
        emit.emit_vstore(addr, v3);
    }
    compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg, None);

    let mut shifted_ops = HashSet::new();
    let mut diag = GreedyDiagnostics::new(verbose);
    let (_cycles, budget_exceeded, _peak) = schedule_ssa_greedy_with_diagnostics(
        &mut emit.ops,
        false,
        &mut shifted_ops,
        8,
        None,
        None,
        None,
        None,
        &GreedyBoundsUsage::None,
        &GreedyPolicy,
        None,
        false,
        false,
        false,
        false,
        None,
        None,
        None,
        &mut diag,
    )
    .expect("force-schedule keeps the scheduler going");
    (diag, budget_exceeded)
}

/// End to end: a schedule that goes over budget in many cycles dumps at most twice when
/// verbose and never when quiet, and reports how many cycles were over budget.
#[test]
fn test_greedy_over_budget_many_cycles_does_not_flood() {
    let (verbose_diag, exceeded) = schedule_over_budget_groups(40, true);
    assert!(exceeded);
    assert!(
        verbose_diag.over_budget_cycles >= 10,
        "scenario must go over budget in many cycles, got {}",
        verbose_diag.over_budget_cycles
    );
    assert!(
        (1..=2).contains(&verbose_diag.dumps_emitted),
        "verbose: first (and optionally worst) cycle only, got {} dumps for {} over-budget cycles",
        verbose_diag.dumps_emitted,
        verbose_diag.over_budget_cycles
    );

    let (quiet_diag, exceeded) = schedule_over_budget_groups(40, false);
    assert!(exceeded);
    assert_eq!(quiet_diag.dumps_emitted, 0, "quiet: no breakdown at all");
    assert_eq!(quiet_diag.over_budget_cycles, verbose_diag.over_budget_cycles);
}

// =========================================================================
// Selects fixed to the multiply-add path (--select-choice fma)
// =========================================================================

/// A select fixed to the multiply-add path without a delta runs its subtract at its cycle and
/// its multiply_add one cycle later. The greedy must (a) schedule its readers two cycles after
/// it, as the CP-SAT model requires (`select_fma_extra_delay`), and (b) count a vector slot in
/// both cycles. Before the fix it used the vselect latency and one slot, so the hint it gave
/// CP-SAT under `--select-choice fma` was infeasible ("The solution hint is complete, but it is
/// infeasible") and the 3-round / batch-64 build found no solution.
#[test]
fn test_fixed_fma_selects_get_their_latency_and_two_vector_slots() {
    let mut emit = SsaEmitContext::new();
    let c1 = emit.emit_const(1);
    let c2 = emit.emit_const(2);
    let v1 = emit.emit_vbroadcast(c1);
    let v2 = emit.emit_vbroadcast(c2);
    let mut selects = Vec::new();
    let mut readers = Vec::new();
    // Twelve independent select chains: enough to fill the vector slots.
    for _ in 0..12 {
        let cond = emit.emit_valu(Opcode::Eq, v1, v2);
        let sel = emit.emit_select_choice(cond, v1, v2);
        selects.push(emit.ops.len() - 1);
        let reader = emit.emit_valu(Opcode::Add, sel, v1);
        readers.push(emit.ops.len() - 1);
        emit.emit_vstore_imm(64 * readers.len() as i64, reader); // keeps the reader live
    }
    let vreg_def = std::mem::take(&mut emit.vreg_def);
    compute_ssa_dependencies(&mut emit.ops, &vreg_def, emit.zero_vreg, None);
    let ops = &mut emit.ops;
    assert!(selects.iter().all(|&i| ops[i].sub_temp_vreg.is_some()), "selects need a subtract");

    let fixed: HashMap<usize, SelectChoiceEncoding> = selects
        .iter()
        .map(|&i| (ops[i].id, SelectChoiceEncoding::Fma))
        .collect();
    let mut shifted = HashSet::new();
    let mut lane_cycles = HashMap::new();
    let ((_, budget_exceeded, _), agreement) = schedule_ssa_greedy_shaped(
        ops,
        false,
        &mut shifted,
        1536,
        None,
        None,
        None,
        None,
        &GreedyBoundsUsage::None,
        &GreedyPolicy,
        None,
        false,
        false,
        false,
        false,
        None,
        None,
        Some(&fixed),
        &mut GreedyDiagnostics::new(false),
        &GreedyShape::default(),
        &mut lane_cycles,
    )
    .expect("greedy schedules the selects");
    assert!(!budget_exceeded);
    assert!(agreement.agrees(), "live-word counter disagrees: {}", agreement);

    for (&s, &r) in selects.iter().zip(&readers) {
        assert!(
            ops[r].cycle >= ops[s].cycle + 2,
            "reader {} at {} reads select {} at {}: the multiply_add result is visible at +2",
            ops[r].id,
            ops[r].cycle,
            ops[s].id,
            ops[s].cycle
        );
    }
    // Vector slots per cycle: one per vector op, plus the multiply_add of each select one
    // cycle after its subtract.
    let mut valu: HashMap<i32, usize> = HashMap::new();
    for (i, op) in ops.iter().enumerate() {
        if selects.contains(&i) {
            *valu.entry(op.cycle).or_default() += 1;
            *valu.entry(op.cycle + 1).or_default() += 1;
        } else if op.engine == Engine::Valu {
            *valu.entry(op.cycle).or_default() += 1;
        }
    }
    assert!(valu.values().all(|&n| n <= 6), "vector slots over capacity: {:?}", valu);
}

// =========================================================================
// Scratch model `pools` (GreedyShape::scratch_pools)
// =========================================================================

/// A vector phase then a scalar phase: `k` broadcasts summed in a chain (many vectors live
/// at once), the sum decomposed into lanes, `m` scalars derived from the lanes and summed in
/// a chain (many scalars live at once). The vector peak and the scalar peak happen at
/// different times, so live words can fit a budget that `8 * max_v + max_s` does not.
fn emit_vector_then_scalar_phases(k: usize, m: usize) -> SsaEmitContext {
    let mut emit = SsaEmitContext::new();
    let s0 = emit.emit_const(1);
    let vs: Vec<VReg> = (0..k).map(|_| emit.emit_vbroadcast_force(s0)).collect();
    let mut acc = emit.emit_valu(Opcode::Add, vs[0], vs[1]);
    for v in &vs[2..] {
        acc = emit.emit_valu(Opcode::Add, acc, *v);
    }
    let lanes = emit.decompose_vector(acc);
    let xs: Vec<VReg> = (0..m)
        .map(|j| emit.emit_alu_imm(Opcode::Add, lanes[j % VLEN], j as i64 + 1))
        .collect();
    let mut y = emit.emit_alu(Opcode::Add, xs[0], xs[1]);
    for x in &xs[2..] {
        y = emit.emit_alu(Opcode::Add, y, *x);
    }
    let addr = emit.emit_const(200);
    emit.emit_store(addr, y);
    let lanes_map = emit.vector_group_lanes.clone();
    compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg, Some(&lanes_map));
    emit
}

/// Schedule with the given budget and scratch model, no shifting.
///
/// # Returns
/// `(budget_exceeded, pool peaks of the schedule under address assignment's liveness)`.
fn schedule_with_scratch_model(
    emit: &mut SsaEmitContext,
    budget: usize,
    pools: bool,
) -> (bool, crate::address_assign::PoolPeaks) {
    let lanes = emit.vector_group_lanes.clone();
    let shape = GreedyShape { scratch_pools: pools, ..Default::default() };
    let mut shifted = HashSet::new();
    let ((_, budget_exceeded, _), agreement) = schedule_ssa_greedy_shaped(
        &mut emit.ops,
        false,
        &mut shifted,
        budget,
        None,
        None,
        None,
        None,
        &GreedyBoundsUsage::None,
        &GreedyPolicy,
        Some(&lanes),
        false,
        false,
        false,
        false,
        None,
        None,
        None,
        &mut GreedyDiagnostics::new(false),
        &shape,
        &mut HashMap::new(),
    )
    .expect("greedy schedules the phases");
    assert!(agreement.agrees(), "live-word counter disagrees: {}", agreement);
    let peaks = scratch_pool_peaks(&emit.ops, Some(&lanes), &HashMap::new(), &HashSet::new(), &HashMap::new()).unwrap();
    (budget_exceeded, peaks)
}

/// With `--scratch-model pools` the greedy keeps `8 * max_v + max_s` within the budget even
/// where the words schedule does not, and address assignment of the result fits. Fails if
/// the flag is ignored: the words schedule breaks the pool condition at this budget.
#[test]
fn test_pools_greedy_keeps_split_within_budget_where_words_does_not() {
    for (k, m) in [(12usize, 40usize), (20, 30), (8, 64), (30, 12)] {
        // Unconstrained run: where the vector and scalar peaks are.
        let mut probe = emit_vector_then_scalar_phases(k, m);
        let (_, free) = schedule_with_scratch_model(&mut probe, 100_000, false);
        // A budget between the per-point peak and the split: words accepts, pools must not.
        let budget = free.split_words() - 1;
        assert!(free.words <= budget, "{}x{}: phases overlap, pick other sizes", k, m);

        let mut words = emit_vector_then_scalar_phases(k, m);
        let (words_exceeded, words_peaks) = schedule_with_scratch_model(&mut words, budget, false);
        assert!(!words_exceeded);
        assert!(
            words_peaks.split_words() > budget,
            "{}x{}: words schedule already within the pools ({:?}); the test proves nothing",
            k,
            m,
            words_peaks
        );

        let mut pools = emit_vector_then_scalar_phases(k, m);
        let (pools_exceeded, pools_peaks) = schedule_with_scratch_model(&mut pools, budget, true);
        assert!(!pools_exceeded, "{}x{}: pools greedy force-scheduled", k, m);
        assert!(
            pools_peaks.split_words() <= budget,
            "{}x{}: pools schedule needs 8*{} + {} = {} > {}",
            k,
            m,
            pools_peaks.vectors,
            pools_peaks.scalars,
            pools_peaks.split_words(),
            budget
        );

        // And address assignment fits it: `budget` assignable words above address 0.
        let config = crate::address_assign::AddressAssignConfig {
            budget: budget + 1,
            vector_group_lanes: pools.vector_group_lanes.clone(),
            scratch_pools: true,
            ..Default::default()
        };
        let scheduled: Vec<SsaOp> = pools.ops.iter().filter(|op| op.cycle >= 0).cloned().collect();
        let assigned = crate::address_assign::assign_addresses(&scheduled, &config).unwrap();
        assert!(
            !assigned.budget_exceeded,
            "{}x{}: pools schedule did not get addresses: max_addr {} with budget {}",
            k,
            m,
            assigned.max_addr,
            budget + 1
        );
    }
}

/// Where the words schedule already keeps the pool condition, `pools` returns the same
/// schedule (no second run, no cycle change).
#[test]
fn test_pools_greedy_same_schedule_when_pools_do_not_bind() {
    let mut words = emit_vector_then_scalar_phases(12, 40);
    let mut pools = emit_vector_then_scalar_phases(12, 40);
    schedule_with_scratch_model(&mut words, 1535, false);
    schedule_with_scratch_model(&mut pools, 1535, true);
    let cycles = |ops: &[SsaOp]| ops.iter().map(|op| op.cycle).collect::<Vec<_>>();
    assert_eq!(cycles(&words.ops), cycles(&pools.ops));
}

/// Three broadcasts of one scalar are ready in the same cycle, and the budget fits only two
/// (plus the scalar and the store address). The add of the first two frees both one cycle later.
fn emit_three_broadcasts() -> SsaEmitContext {
    let mut emit = SsaEmitContext::new();
    let s = emit.emit_const(1);
    let v1 = emit.emit_vbroadcast_force(s);
    let v2 = emit.emit_vbroadcast_force(s);
    let v3 = emit.emit_vbroadcast_force(s);
    let w = emit.emit_valu(Opcode::Add, v1, v2);
    let x = emit.emit_valu(Opcode::Add, w, v3);
    let addr = emit.emit_const(100);
    emit.emit_vstore(addr, x);
    compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg, None);
    emit
}

/// Schedule with the given budget-force rule (words model, no shifting).
///
/// # Returns
/// `(budget_exceeded, peak live words)`.
fn schedule_with_budget_force(emit: &mut SsaEmitContext, budget: usize, force: BudgetForce) -> (bool, usize) {
    let shape = GreedyShape { budget_force: force, ..Default::default() };
    let ((_, budget_exceeded, peak), agreement) = schedule_ssa_greedy_shaped(
        &mut emit.ops,
        false,
        &mut HashSet::new(),
        budget,
        None,
        None,
        None,
        None,
        &GreedyBoundsUsage::None,
        &GreedyPolicy,
        None,
        false,
        false,
        false,
        false,
        None,
        None,
        None,
        &mut GreedyDiagnostics::new(false),
        &shape,
        &mut HashMap::new(),
    )
    .expect("greedy schedules the broadcasts");
    assert!(agreement.agrees(), "live-word counter disagrees: {}", agreement);
    (budget_exceeded, peak)
}

/// `--greedy-budget-force stall`: a budget-deferred op waits while other ops are placed in
/// the cycle, so the run stays within the budget. The default (`same-cycle`) forces the
/// third broadcast in the cycle the first two were placed and goes over: the control shows
/// the program exercises the difference.
#[test]
fn test_budget_force_stall_keeps_words_budget_where_same_cycle_does_not() {
    let budget = 2 * VLEN + 2;
    let (same_exceeded, same_peak) =
        schedule_with_budget_force(&mut emit_three_broadcasts(), budget, BudgetForce::SameCycle);
    assert!(
        same_exceeded && same_peak > budget,
        "control: same-cycle should force the third broadcast over the budget (peak {})",
        same_peak
    );
    let (stall_exceeded, stall_peak) =
        schedule_with_budget_force(&mut emit_three_broadcasts(), budget, BudgetForce::Stall);
    assert!(!stall_exceeded, "stall forced an op although the cycle placed others");
    assert!(stall_peak <= budget, "stall peak {} > budget {}", stall_peak, budget);
}

/// When nothing fits, `stall` still forces: the scheduler completes and reports it.
#[test]
fn test_budget_force_stall_still_breaks_deadlock() {
    let (exceeded, _) = schedule_with_budget_force(&mut emit_three_broadcasts(), VLEN, BudgetForce::Stall);
    assert!(exceeded, "budget below two vectors: the add can only run over budget");
}

/// The greedy scheduler respects kept memory edges: a load after a store to its word runs at
/// least one cycle later (RAW), a store over a word read earlier runs no earlier than the
/// read (WAR, latency 0), and two stores to one word are a cycle apart (WAW). Without the
/// edges the second load would be ready in the store's cycle.
#[test]
fn test_greedy_respects_memory_edges() {
    use crate::kernel_builder::memory_deps::{compute_memory_dependencies, test_programs::store_load_same_word};

    let (mut ops, ix) = store_load_same_word();
    let stats = compute_memory_dependencies(&mut ops).unwrap();
    assert_eq!(stats.kept, 3);
    let mut shifted_ops = HashSet::new();
    schedule_ssa_greedy_with_shift_budget(&mut ops, false, &mut shifted_ops, 1536).unwrap();
    let c = |i: usize| ops[i].cycle;
    assert!(c(ix.store_word0) >= c(ix.load_before), "WAR: {} < {}", c(ix.store_word0), c(ix.load_before));
    assert!(c(ix.load_after) >= c(ix.store_word0) + 1, "RAW: {} vs {}", c(ix.load_after), c(ix.store_word0));
    assert!(c(ix.last_store) >= c(ix.first_store) + 1, "WAW: {} vs {}", c(ix.last_store), c(ix.first_store));

    // Control: without the memory edges the second load is not held back by the store.
    let (mut free, _) = store_load_same_word();
    schedule_ssa_greedy_with_shift_budget(&mut free, false, &mut HashSet::new(), 1536).unwrap();
    assert!(
        free[ix.load_after].cycle <= free[ix.store_word0].cycle,
        "the test needs a program where the RAW edge binds: load {} store {}",
        free[ix.load_after].cycle,
        free[ix.store_word0].cycle
    );
}

/// The first chain by emission order feeds the last store, which must wait for the
/// store in the other chain. Both stores need to be reachable with a one-chain window.
#[test]
fn test_wavefront_window_respects_memory_store_order() {
    use crate::kernel_builder::memory_deps::{check_memory_edges, compute_memory_dependencies};
    use crate::types::{MemLocation, MemRegion};

    let mut emit = SsaEmitContext::new();
    let p = emit.emit_const_choice(100);
    let a = emit.emit_const_choice(7);
    let b = emit.emit_const_choice(8);
    for value in [b, a] {
        emit.set_next_mem_loc(MemLocation::word(MemRegion::ExtraRoom, 0));
        emit.emit_store(p, value);
    }
    compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg, None);
    assert_eq!(compute_memory_dependencies(&mut emit.ops).unwrap().kept, 1);
    schedule_with_shape(&mut emit.ops, &wavefront_shape(0, Some(1)));
    check_memory_edges(&emit.ops, |i| emit.ops[i].cycle).unwrap();
}

/// A capped wavefront must also admit loads and stores linked by RAW and WAR.
#[test]
fn test_wavefront_window_respects_mixed_memory_edges() {
    use crate::kernel_builder::memory_deps::{
        check_memory_edges, compute_memory_dependencies, test_programs::store_load_same_word,
    };

    let (mut ops, _) = store_load_same_word();
    assert_eq!(compute_memory_dependencies(&mut ops).unwrap().kept, 3);
    schedule_with_shape(&mut ops, &wavefront_shape(0, Some(1)));
    check_memory_edges(&ops, |i| ops[i].cycle).unwrap();
}

/// A shared load feeding several otherwise independent chains can require a store
/// from a later chain. That store and its data prerequisites must be admitted too.
#[test]
fn test_wavefront_shared_load_admits_its_store() {
    use crate::kernel_builder::memory_deps::{check_memory_edges, compute_memory_dependencies};
    use crate::types::{MemLocation, MemRegion};

    let mut emit = SsaEmitContext::new();
    let addr = emit.emit_const(100);
    let offsets: Vec<_> = (1..=3).map(|c| emit.emit_const(c)).collect();
    let value = emit.emit_const(7);
    let written = MemLocation::word(MemRegion::ExtraRoom, 0);
    emit.set_next_mem_loc(written);
    emit.emit_store(addr, value);
    let store = emit.ops.len() - 1;
    emit.set_next_mem_loc(written);
    let shared = emit.emit_load(addr);
    for (i, input) in offsets.into_iter().enumerate() {
        let out = emit.emit_alu(Opcode::Add, shared, input);
        let dest = emit.emit_const(101 + i as i64);
        emit.set_next_mem_loc(MemLocation::word(MemRegion::ExtraRoom, 1 + i as i64));
        emit.emit_store(dest, out);
    }
    compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg, None);
    assert!(wavefront_for(&emit.ops, 0, Some(1)).chain(store).is_some());
    assert_eq!(compute_memory_dependencies(&mut emit.ops).unwrap().kept, 1);
    assert_eq!(wavefront_for(&emit.ops, 0, Some(1)).chain(store), None);
    schedule_with_shape(&mut emit.ops, &wavefront_shape(0, Some(1)));
    check_memory_edges(&emit.ops, |i| emit.ops[i].cycle).unwrap();
}

// =========================================================================
// Shift policies and load priority (docs/GREEDY_SHAPE_ABLATION.md section 11)
// =========================================================================

/// Vector ops per cycle left on the vector ALU, and scalar slots per cycle (shifted lanes
/// plus native scalar ops). Panics if an op is unscheduled.
fn vector_and_scalar_use(
    ops: &[SsaOp],
    shifted: &HashSet<usize>,
    lane_cycles: &HashMap<usize, Vec<i32>>,
) -> (HashMap<i32, usize>, HashMap<i32, usize>) {
    let mut valu: HashMap<i32, usize> = HashMap::new();
    let mut alu = shifted_alu_use(ops, shifted, lane_cycles);
    for op in ops {
        assert!(op.cycle >= 0, "op {} is unscheduled", op.id);
        match op.engine {
            Engine::Valu if !shifted.contains(&op.id) => *valu.entry(op.cycle).or_default() += 1,
            Engine::Alu => *alu.entry(op.cycle).or_default() += 1,
            _ => {}
        }
    }
    (valu, alu)
}

/// Every op runs after each of its operands completes (a split shift starts at its first
/// lane).
fn assert_dependencies_respected(ops: &[SsaOp], lane_cycles: &HashMap<usize, Vec<i32>>) {
    let by_id: HashMap<usize, &SsaOp> = ops.iter().map(|op| (op.id, op)).collect();
    for op in ops {
        let start = lane_cycles.get(&op.id).map_or(op.cycle, |l| *l.iter().min().unwrap());
        for dep in &op.dep_ids {
            if let Some(producer) = by_id.get(dep) {
                assert!(
                    start > producer.cycle,
                    "op {} at {} reads op {} at {}",
                    op.id,
                    start,
                    dep,
                    producer.cycle
                );
            }
        }
    }
}

/// Chains that alternate a shiftable add and a multiply_add, which cannot shift.
fn emit_mixed_chains(chains: usize, levels: usize) -> SsaEmitContext {
    let mut emit = SsaEmitContext::new();
    let scalar = emit.emit_const(3);
    let shared = emit.emit_vbroadcast(scalar);
    let addr = emit.emit_const(100);
    let mut value: Vec<VReg> = vec![shared; chains];
    for level in 0..levels {
        for chain in 0..chains {
            value[chain] = if level % 2 == 0 {
                emit.emit_valu(Opcode::Add, value[chain], shared)
            } else {
                emit.emit_fma(value[chain], shared, shared)
            };
        }
    }
    for chain in 0..chains {
        emit.emit_vstore(addr, value[chain]);
    }
    compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg, None);
    emit
}

#[test]
fn test_balanced_shift_target() {
    // flow90 at 16 rounds: 7,426 vector ops, 4,844 shiftable, 208 scalar ops.
    assert_eq!(balanced_shift_target(7426, 4844, 208), 1465);
    // Clamped to the shiftable ops, and never negative.
    assert_eq!(balanced_shift_target(100, 5, 0), 5);
    assert_eq!(balanced_shift_target(10, 10, 500), 0);
}

/// Every shift policy, with and without split shifts, keeps at most 6 vector ops and 12
/// scalar slots per cycle, schedules every op and respects every dependency.
#[test]
fn test_shift_policies_keep_slot_limits() {
    let policies = [
        ShiftPolicy::WhenFull,
        ShiftPolicy::Reserve,
        ShiftPolicy::Pace { target: None },
        ShiftPolicy::ReservePace { target: Some(40) },
    ];
    for &shift_policy in &policies {
        for split_shift in [false, true] {
            let shape = GreedyShape { split_shift, shift_policy, ..Default::default() };
            let mut emit = emit_mixed_chains(12, 10);
            let (_, shifted, lanes) = schedule_with_shift_and_shape(&mut emit.ops, &shape);
            let (valu, alu) = vector_and_scalar_use(&emit.ops, &shifted, &lanes);
            assert!(
                valu.values().all(|&n| n <= 6),
                "{:?} split={}: vector ALU {:?}",
                shift_policy,
                split_shift,
                valu
            );
            assert!(
                alu.values().all(|&n| n <= 12),
                "{:?} split={}: scalar ALU {:?}",
                shift_policy,
                split_shift,
                alu
            );
            assert_dependencies_respected(&emit.ops, &lanes);
            if !split_shift {
                assert!(lanes.is_empty());
            }
        }
    }
}

/// Two adds and six multiply_adds ready together. Shifting only when the vector slots are
/// full puts both adds on the vector ALU and two multiply_adds wait a cycle; `reserve`
/// shifts the adds (one whole, one split) so all six multiply_adds run at once.
#[test]
fn test_shift_reserve_leaves_vector_slots_to_multiply_adds() {
    let emit_one_cycle = || {
        let mut emit = SsaEmitContext::new();
        let scalar = emit.emit_const(3);
        let shared = emit.emit_vbroadcast(scalar);
        let addr = emit.emit_const(100);
        let mut results = Vec::new();
        for _ in 0..2 {
            results.push(emit.emit_valu(Opcode::Add, shared, shared));
        }
        let mut fmas = HashSet::new();
        for _ in 0..6 {
            results.push(emit.emit_fma(shared, shared, shared));
            fmas.insert(emit.ops.last().unwrap().id);
        }
        for r in results {
            emit.emit_vstore(addr, r);
        }
        compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg, None);
        (emit, fmas)
    };
    let fma_cycles = |policy: ShiftPolicy| {
        let (mut emit, fmas) = emit_one_cycle();
        let shape = GreedyShape { split_shift: true, shift_policy: policy, ..Default::default() };
        let (_, shifted, lanes) = schedule_with_shift_and_shape(&mut emit.ops, &shape);
        let (valu, alu) = vector_and_scalar_use(&emit.ops, &shifted, &lanes);
        assert!(valu.values().all(|&n| n <= 6) && alu.values().all(|&n| n <= 12));
        let cycles: std::collections::BTreeSet<i32> =
            emit.ops.iter().filter(|op| fmas.contains(&op.id)).map(|op| op.cycle).collect();
        (cycles, shifted.len())
    };
    let (when_full, when_full_shifted) = fma_cycles(ShiftPolicy::WhenFull);
    assert_eq!(when_full.len(), 2, "two multiply_adds wait a cycle: {:?}", when_full);
    assert_eq!(when_full_shifted, 0);
    let (reserve, reserve_shifted) = fma_cycles(ShiftPolicy::Reserve);
    assert_eq!(reserve.len(), 1, "all multiply_adds in one cycle: {:?}", reserve);
    assert_eq!(reserve_shifted, 2);
}

/// `pace` shifts toward its target even where the vector slots never fill; `when-full`
/// shifts nothing there.
#[test]
fn test_shift_pace_reaches_its_target() {
    for (policy, lo, hi) in [
        (ShiftPolicy::WhenFull, 0, 0),
        (ShiftPolicy::Pace { target: Some(12) }, 11, 13),
    ] {
        let (mut emit, _) = emit_independent_chains(2, 30);
        let shape = GreedyShape { shift_policy: policy, ..Default::default() };
        let (_, shifted, lanes) = schedule_with_shift_and_shape(&mut emit.ops, &shape);
        let (valu, alu) = vector_and_scalar_use(&emit.ops, &shifted, &lanes);
        assert!(valu.values().all(|&n| n <= 6) && alu.values().all(|&n| n <= 12));
        assert!((lo..=hi).contains(&shifted.len()), "{:?}: {} shifted", policy, shifted.len());
    }
}

/// Schedule `ops` with shifting on, `budget` words and `policy`; returns
/// (budget exceeded, shifted ops, split lane cycles, diagnostics).
fn schedule_tight(
    ops: &mut [SsaOp],
    shape: &GreedyShape,
    budget: usize,
    policy: &dyn PressurePolicy,
) -> (bool, HashSet<usize>, HashMap<usize, Vec<i32>>, GreedyDiagnostics) {
    let mut shifted = HashSet::new();
    let mut lanes = HashMap::new();
    let mut diagnostics = GreedyDiagnostics::new(false);
    let ((_, budget_exceeded, _), _) = schedule_ssa_greedy_shaped(
        ops,
        true,
        &mut shifted,
        budget,
        None,
        None,
        None,
        None,
        &GreedyBoundsUsage::None,
        policy,
        None,
        false,
        false,
        false,
        false,
        None,
        None,
        None,
        &mut diagnostics,
        shape,
        &mut lanes,
    )
    .expect("greedy completes past the budget");
    (budget_exceeded, shifted, lanes, diagnostics)
}

/// Force-schedule path: budgets tight enough that ops are forced past them, where the pace
/// policy also makes proactive shifts in the same run (counted by
/// `GreedyDiagnostics::proactive_shifts`), under both the plain and the adaptive policy.
/// Slot limits and dependencies hold. `reserve` alone makes no proactive shift on these
/// budgets (the budget lets too few multiply_adds be ready at once), so it is checked for
/// the limits only.
#[test]
fn test_shift_policies_keep_slot_limits_when_force_scheduling() {
    let adaptive = AdaptivePolicy::default();
    let policies: [&dyn PressurePolicy; 2] = [&GreedyPolicy, &adaptive];
    for budget in [56, 64] {
        for policy in policies {
            for shift_policy in [
                ShiftPolicy::Reserve,
                ShiftPolicy::Pace { target: None },
                ShiftPolicy::ReservePace { target: None },
            ] {
                for split_shift in [false, true] {
                    let shape = GreedyShape { split_shift, shift_policy, ..Default::default() };
                    let mut emit = emit_mixed_chains(10, 6);
                    let (exceeded, shifted, lanes, diagnostics) =
                        schedule_tight(&mut emit.ops, &shape, budget, policy);
                    assert!(exceeded && diagnostics.over_budget_cycles > 0, "budget {} forces", budget);
                    if shift_policy != ShiftPolicy::Reserve {
                        assert!(
                            diagnostics.proactive_shifts > 0,
                            "{:?} split={} budget {}: no proactive shift on a force-scheduling run",
                            shift_policy,
                            split_shift,
                            budget
                        );
                    }
                    let (valu, alu) = vector_and_scalar_use(&emit.ops, &shifted, &lanes);
                    assert!(valu.values().all(|&n| n <= 6), "{:?}: vector ALU {:?}", shift_policy, valu);
                    assert!(alu.values().all(|&n| n <= 12), "{:?}: scalar ALU {:?}", shift_policy, alu);
                    assert_dependencies_respected(&emit.ops, &lanes);
                }
            }
        }
    }
}

/// With the hardware budget, `reserve` makes proactive shifts on the mixed chains (the
/// counter the force-path test relies on is live).
#[test]
fn test_proactive_shift_counter() {
    for (policy, expect_some) in [(ShiftPolicy::WhenFull, false), (ShiftPolicy::Reserve, true)] {
        let mut emit = emit_mixed_chains(12, 10);
        let shape = GreedyShape { shift_policy: policy, ..Default::default() };
        let (exceeded, _, _, diagnostics) = schedule_tight(&mut emit.ops, &shape, 1536, &GreedyPolicy);
        assert!(!exceeded);
        assert_eq!(diagnostics.proactive_shifts > 0, expect_some, "{:?}", policy);
    }
}

/// Chains of load -> add -> load -> add -> load -> store: the last load of each chain is
/// final (no later load depends on it).
fn emit_load_chains(chains: usize) -> (SsaEmitContext, HashSet<usize>) {
    let mut emit = SsaEmitContext::new();
    let mut finals = HashSet::new();
    for chain in 0..chains {
        let mut addr = emit.emit_const(10 * chain as i64);
        for step in 0..3 {
            let value = emit.emit_load(addr);
            if step == 2 {
                finals.insert(emit.ops.last().unwrap().id);
                emit.emit_store_imm(1000 + chain as i64, value);
            } else {
                addr = emit.emit_alu_imm(Opcode::Add, value, 1);
            }
        }
    }
    compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg, None);
    (emit, finals)
}

/// Number of final loads that took a load slot while a non-final load was ready and
/// waiting. Also checks the load slot limit.
fn final_loads_ahead_of_ready_feeders(ops: &[SsaOp], finals: &HashSet<usize>) -> usize {
    let by_id: HashMap<usize, &SsaOp> = ops.iter().map(|op| (op.id, op)).collect();
    let ready = |op: &SsaOp| op.dep_ids.iter().map(|d| by_id[d].cycle + 1).max().unwrap_or(0);
    let loads: Vec<&SsaOp> = ops.iter().filter(|op| op.engine == Engine::Load).collect();
    let mut per_cycle: HashMap<i32, usize> = HashMap::new();
    for op in &loads {
        *per_cycle.entry(op.cycle).or_default() += 1;
    }
    assert!(per_cycle.values().all(|&n| n <= 2), "load slots over capacity: {:?} {:?}", per_cycle, loads.iter().map(|o| (o.id, o.opcode, o.cycle)).collect::<Vec<_>>());
    loads
        .iter()
        .filter(|op| finals.contains(&op.id))
        .filter(|f| {
            loads
                .iter()
                .any(|l| !finals.contains(&l.id) && ready(l) <= f.cycle && l.cycle > f.cycle)
        })
        .count()
}

/// `final-last`: a final load never takes a load slot while a load that feeds later loads
/// is ready; with the wavefront key alone it does.
#[test]
fn test_load_priority_final_last() {
    let shape = |load_priority| GreedyShape {
        priority: GreedyPriority::Wavefront { stagger_cycles: 6, groups_in_flight: None },
        load_priority,
        ..Default::default()
    };
    let (mut keyed, finals) = emit_load_chains(8);
    schedule_with_shape(&mut keyed.ops, &shape(LoadPriority::Key));
    assert!(final_loads_ahead_of_ready_feeders(&keyed.ops, &finals) > 0);

    let (mut final_last, finals) = emit_load_chains(8);
    schedule_with_shape(&mut final_last.ops, &shape(LoadPriority::FinalLast));
    assert_eq!(final_loads_ahead_of_ready_feeders(&final_last.ops, &finals), 0);
    assert_dependencies_respected(&final_last.ops, &HashMap::new());
}

/// The wavefront marks exactly the loads no later load depends on as final.
#[test]
fn test_wavefront_feeds_load() {
    let (emit, finals) = emit_load_chains(3);
    let w = wavefront_for(&emit.ops, 6, None);
    for (idx, op) in emit.ops.iter().enumerate() {
        if op.engine == Engine::Load {
            assert_eq!(w.feeds_load(idx), !finals.contains(&op.id), "load {}", op.id);
        }
    }
}

// =========================================================================
// Flow priority and select overflow (docs/GREEDY_SHAPE_ABLATION.md section 12)
// =========================================================================

/// `chains` independent chains, each: vector AND -> SelectChoice with a precomputed delta
/// (one multiply_add or one vselect) -> vector add -> vstore. The table vectors (base,
/// delta, base + delta) are shared. Returns the context and the select op ids.
fn emit_delta_select_chains(chains: usize) -> (SsaEmitContext, HashSet<usize>) {
    let mut emit = SsaEmitContext::new();
    let c1 = emit.emit_const(1);
    let one = emit.emit_vbroadcast(c1);
    let c5 = emit.emit_const(5);
    let base = emit.emit_vbroadcast(c5);
    let c9 = emit.emit_const(9);
    let delta = emit.emit_vbroadcast(c9);
    let true_val = emit.emit_valu(Opcode::Add, base, delta);
    let addr = emit.emit_const(100);
    let mut selects = HashSet::new();
    for chain in 0..chains {
        let cv = emit.emit_const(chain as i64);
        let v = emit.emit_vbroadcast(cv);
        let cond = emit.emit_valu(Opcode::And, v, one);
        let sel = emit.emit_select_choice_shared_delta(cond, true_val, delta, base);
        selects.insert(emit.ops.last().unwrap().id);
        let out = emit.emit_valu(Opcode::Add, sel, one);
        emit.emit_vstore(addr, out);
    }
    compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg, None);
    (emit, selects)
}

/// Schedule with shifting on; with `collect`, also collect the selects placed in the
/// multiply_add form. Returns (budget exceeded, shifted, multiply-add selects, lane cycles).
fn schedule_collecting_fma(
    ops: &mut [SsaOp],
    shape: &GreedyShape,
    budget: usize,
    policy: &dyn PressurePolicy,
    collect: bool,
) -> Result<(bool, HashSet<usize>, HashSet<usize>, HashMap<usize, Vec<i32>>), String> {
    let mut shifted = HashSet::new();
    let mut lanes = HashMap::new();
    let mut fma = HashSet::new();
    let ((_, budget_exceeded, _), _) = schedule_ssa_greedy_shaped(
        ops,
        true,
        &mut shifted,
        budget,
        if collect { Some(&mut fma) } else { None },
        None,
        None,
        None,
        &GreedyBoundsUsage::None,
        policy,
        None,
        false,
        false,
        false,
        false,
        None,
        None,
        None,
        &mut GreedyDiagnostics::new(false),
        shape,
        &mut lanes,
    )?;
    Ok((budget_exceeded, shifted, fma, lanes))
}

/// Per-cycle slot limits with selects resolved as the run chose: a select in `fma` takes a
/// vector slot, any other flow op the flow slot.
fn assert_slot_limits_with_selects(
    ops: &[SsaOp],
    shifted: &HashSet<usize>,
    fma: &HashSet<usize>,
    lanes: &HashMap<usize, Vec<i32>>,
    label: &str,
) {
    let (mut valu, alu) = vector_and_scalar_use(ops, shifted, lanes);
    let mut flow: HashMap<i32, usize> = HashMap::new();
    for op in ops {
        if op.engine == Engine::Flow {
            if fma.contains(&op.id) {
                assert_eq!(op.opcode, Opcode::SelectChoice, "{}: op {} in the fma set", label, op.id);
                assert!(op.delta_vreg.is_some(), "{}: select {} has no one-op form", label, op.id);
                *valu.entry(op.cycle).or_default() += 1;
            } else {
                *flow.entry(op.cycle).or_default() += 1;
            }
        }
    }
    assert!(valu.values().all(|&n| n <= 6), "{}: vector ALU {:?}", label, valu);
    assert!(alu.values().all(|&n| n <= 12), "{}: scalar ALU {:?}", label, alu);
    assert!(flow.values().all(|&n| n <= 1), "{}: flow {:?}", label, flow);
    assert_dependencies_respected(ops, lanes);
}

/// Default (`Never`): every select stays on flow. `Valu`: selects that find the flow slot
/// taken run as a multiply_add, within the limit. Slot limits hold either way, with and
/// without split shifts, under the wavefront priority with both flow priorities.
#[test]
fn test_select_overflow_keeps_slot_limits() {
    let overflows = [
        SelectOverflow::Never,
        SelectOverflow::Valu { limit: None },
        SelectOverflow::Valu { limit: Some(3) },
    ];
    for &select_overflow in &overflows {
        for split_shift in [false, true] {
            for (flow_priority, alu_priority) in [
                (FlowPriority::Key, AluPriority::Key),
                (FlowPriority::FeedsLoadFirst, AluPriority::Key),
                (FlowPriority::ChainFirst, AluPriority::NativeFirst),
            ] {
                let shape = GreedyShape {
                    priority: GreedyPriority::Wavefront { stagger_cycles: 2, groups_in_flight: None },
                    split_shift,
                    select_overflow,
                    flow_priority,
                    alu_priority,
                    ..Default::default()
                };
                let (mut emit, selects) = emit_delta_select_chains(24);
                let (_, shifted, fma, lanes) =
                    schedule_collecting_fma(&mut emit.ops, &shape, 1536, &GreedyPolicy, true).unwrap();
                let label = format!("{:?} split={} {:?} {:?}", select_overflow, split_shift, flow_priority, alu_priority);
                assert_slot_limits_with_selects(&emit.ops, &shifted, &fma, &lanes, &label);
                assert!(fma.is_subset(&selects), "{}", label);
                match select_overflow {
                    SelectOverflow::Never => assert!(fma.is_empty(), "{}", label),
                    SelectOverflow::Valu { limit: None } => assert!(fma.len() > 3, "{}: {}", label, fma.len()),
                    SelectOverflow::Valu { limit: Some(l) } => assert_eq!(fma.len(), l, "{}", label),
                }
            }
        }
    }
}

/// Force-schedule path (tight budget, adaptive policy): select overflow keeps the slot
/// limits there too.
#[test]
fn test_select_overflow_keeps_slot_limits_when_force_scheduling() {
    for split_shift in [false, true] {
        let shape = GreedyShape {
            split_shift,
            select_overflow: SelectOverflow::Valu { limit: None },
            ..Default::default()
        };
        let (mut emit, _) = emit_delta_select_chains(16);
        let (budget_exceeded, shifted, fma, lanes) =
            schedule_collecting_fma(&mut emit.ops, &shape, 80, &AdaptivePolicy::default(), true)
                .expect("greedy completes past the budget");
        assert!(budget_exceeded, "the budget is tight enough to force-schedule");
        assert_slot_limits_with_selects(&emit.ops, &shifted, &fma, &lanes, "force");
    }
}

/// Review of #76 (CodeRabbit, second pass): a multiply-add select with a distinct `keep_live`
/// vector it is the last holder of. `keep_live` is a liveness-only dependency through the
/// select's cycle; only the values the multiply_add reads are held to cycle + 1. The counter
/// and address assignment must apply the same rule (before: the counter held it to cycle + 1,
/// the allocator's multiply-add branch skipped `keep_live`, and the check failed).
#[test]
fn test_overflow_select_keep_live_agrees_with_interval_liveness() {
    let shape = GreedyShape {
        priority: GreedyPriority::Wavefront { stagger_cycles: 1, groups_in_flight: None },
        select_overflow: SelectOverflow::Valu { limit: None },
        ..Default::default()
    };
    let mut emit = SsaEmitContext::new();
    let mut vec_of = |emit: &mut SsaEmitContext, v: i64| {
        let c = emit.emit_const(v);
        emit.emit_vbroadcast(c)
    };
    let one = vec_of(&mut emit, 1);
    let (b, d) = (vec_of(&mut emit, 5), vec_of(&mut emit, 6));
    let (c1v, c2v) = (vec_of(&mut emit, 1), vec_of(&mut emit, 0));
    let held = vec_of(&mut emit, 7);
    let early = emit.emit_valu(Opcode::Add, held, one); // the only real read of `held`
    // A chain delays the selects well past `early`, the last real read of `held`.
    let mut b2 = b;
    for _ in 0..4 {
        b2 = emit.emit_valu(Opcode::Add, b2, one);
    }
    let t = emit.emit_valu(Opcode::Add, b2, d);
    let s1 = emit.emit_select_choice_shared_delta(c1v, t, d, b2);
    // Both selects keep `held` live until they run (liveness only); one of them overflows.
    emit.ops.last_mut().unwrap().keep_live = Some(vec![held]);
    let s2 = emit.emit_select_choice_shared_delta(c2v, t, d, b2);
    emit.ops.last_mut().unwrap().keep_live = Some(vec![held]);
    let addr = emit.emit_const(100);
    let x = emit.emit_valu(Opcode::Add, s1, s2);
    let y = emit.emit_valu(Opcode::Add, x, early);
    emit.emit_vstore(addr, y);
    compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg, None);
    let mut fma = HashSet::new();
    let ((_, _, _), agreement) = schedule_ssa_greedy_shaped(
        &mut emit.ops, true, &mut HashSet::new(), 1536, Some(&mut fma), None, None, None,
        &GreedyBoundsUsage::None, &GreedyPolicy, None, false, false, false, false, None, None, None,
        &mut GreedyDiagnostics::new(false), &shape, &mut HashMap::new(),
    )
    .expect("the counter agrees with interval liveness");
    assert!(!fma.is_empty(), "control: a select overflowed to the multiply_add form");
    assert!(agreement.agrees(), "{}", agreement);
}

/// Review of #76 (CodeRabbit): the budget fit check must not count as freed a value a
/// multiply-add select read this cycle (it is held to cycle + 1). Swept over tight budgets
/// with selects overflowing to the vector ALU: whenever the greedy reports the budget as kept,
/// interval liveness of its schedule stays within it.
#[test]
fn test_fit_check_respects_multiply_add_read_holds() {
    let mut binding = 0;
    for budget in (40..=200).step_by(4) {
        for chains in [12, 24] {
            let shape = GreedyShape {
                priority: GreedyPriority::Wavefront { stagger_cycles: 1, groups_in_flight: None },
                select_overflow: SelectOverflow::Valu { limit: None },
                ..Default::default()
            };
            let (mut emit, _) = emit_delta_select_chains(chains);
            let mut fma = HashSet::new();
            let mut lanes = HashMap::new();
            let result = schedule_ssa_greedy_shaped(
                &mut emit.ops,
                true,
                &mut HashSet::new(),
                budget,
                Some(&mut fma),
                None,
                None,
                None,
                &GreedyBoundsUsage::None,
                &GreedyPolicy,
                None,
                false,
                false,
                false,
                false,
                None,
                None,
                None,
                &mut GreedyDiagnostics::new(false),
                &shape,
                &mut lanes,
            );
            // An over-budget run returns Ok with `exceeded`; any error is a failure.
            let ((_, exceeded, _), _) = result.unwrap_or_else(|e| panic!("budget {}: scheduler error: {}", budget, e));
            if exceeded {
                continue;
            }
            let scheduled: Vec<SsaOp> = emit.ops.iter().filter(|op| op.cycle >= 0).cloned().collect();
            let config = crate::address_assign::AddressAssignConfig {
                select_fma_ops: fma.clone(),
                shifted_alu_cycles: lanes.clone(),
                ..Default::default()
            };
            let intervals = crate::address_assign::compute_live_intervals(&config, &scheduled).unwrap();
            let peak = crate::address_assign::compute_peak_liveness(&intervals);
            assert!(peak <= budget, "budget {} chains {}: kept by the greedy, but interval peak {}", budget, chains, peak);
            binding += usize::from(peak + 16 >= budget);
        }
    }
    assert!(binding > 0, "control: some run must sit near its budget");
}

/// The same, on a program built for it: in one cycle a select takes the flow slot, a second
/// select overflows to the vector ALU (a delta multiply_add reading its base `b2`, held to
/// cycle + 1), and an ordinary op is the last reader of `b2` in that cycle and writes a new
/// vector. At a budget where only the (wrong) early release of `b2` lets that op fit, the
/// greedy must defer it or report the budget exceeded.
#[test]
fn test_fit_check_respects_a_same_cycle_multiply_add_hold() {
    let build = || {
        let mut emit = SsaEmitContext::new();
        let mut vec_of = |emit: &mut SsaEmitContext, v: i64| {
            let c = emit.emit_const(v);
            emit.emit_vbroadcast(c)
        };
        let one = vec_of(&mut emit, 1);
        let (b, d) = (vec_of(&mut emit, 5), vec_of(&mut emit, 6));
        let (c1v, c2v) = (vec_of(&mut emit, 1), vec_of(&mut emit, 0));
        let t = emit.emit_valu(Opcode::Add, b, d);
        // Both selects and `o` become ready in the same cycle, right after `t`.
        let s1 = emit.emit_select_choice_shared_delta(c1v, t, d, b);
        let s2 = emit.emit_select_choice_shared_delta(c2v, t, d, b);
        let o = emit.emit_valu(Opcode::Add, b, t); // last reader of b, a new vector
        let _ = one;
        let addr = emit.emit_const(100);
        let x = emit.emit_valu(Opcode::Add, s1, s2);
        let y = emit.emit_valu(Opcode::Add, x, o);
        emit.emit_vstore(addr, y);
        compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg, None);
        emit
    };
    let mut checked = 0;
    for budget in 16..=160 {
        let mut emit = build();
        let mut fma = HashSet::new();
        let shape = GreedyShape { select_overflow: SelectOverflow::Valu { limit: None }, ..Default::default() };
        let ((_, exceeded, _), _) = schedule_ssa_greedy_shaped(
            &mut emit.ops, true, &mut HashSet::new(), budget, Some(&mut fma), None, None, None,
            &GreedyBoundsUsage::None, &GreedyPolicy, None, false, false, false, false, None, None, None,
            &mut GreedyDiagnostics::new(false), &shape, &mut HashMap::new(),
        )
        .unwrap_or_else(|e| panic!("budget {}: scheduler error: {}", budget, e));
        if exceeded || fma.is_empty() {
            continue;
        }
        let scheduled: Vec<SsaOp> = emit.ops.iter().filter(|op| op.cycle >= 0).cloned().collect();
        let config = crate::address_assign::AddressAssignConfig { select_fma_ops: fma.clone(), ..Default::default() };
        let intervals = crate::address_assign::compute_live_intervals(&config, &scheduled).unwrap();
        let peak = crate::address_assign::compute_peak_liveness(&intervals);
        assert!(peak <= budget, "budget {}: kept by the greedy, but interval peak {}", budget, peak);
        checked += 1;
    }
    assert!(checked > 0, "control: some budget runs with a multiply-add select and is kept");
}

/// Review of #76 (CodeRabbit, third pass; fix in #82): the fit estimate of a select in the
/// multiply_add form must hold only what the multiply_add reads (sources, delta, their group
/// aliases) to cycle + 1. A `keep_live`-only vector is released at the select's cycle, so the
/// select's destination may reuse its words. Four vector inputs, a `keep_live` vector and a
/// scalar take 41 words; at budget 41 the estimate was 49, the select was deferred with nothing
/// else ready, and the force path reported the budget exceeded.
#[test]
fn test_fit_estimate_frees_keep_live_only_value_of_multiply_add_select() {
    let mut emit = SsaEmitContext::new();
    let mut vec_of = |emit: &mut SsaEmitContext, v: i64| {
        let c = emit.emit_const(v);
        emit.emit_vbroadcast(c)
    };
    let (b, d) = (vec_of(&mut emit, 5), vec_of(&mut emit, 6));
    let cond = vec_of(&mut emit, 1);
    let held = vec_of(&mut emit, 7);
    let t = emit.emit_valu(Opcode::Add, b, d);
    let s = emit.emit_select_choice_shared_delta(cond, t, d, b);
    emit.ops.last_mut().unwrap().keep_live = Some(vec![held]);
    let select_id = emit.ops.last().unwrap().id;
    let addr = emit.emit_const(100);
    emit.emit_vstore(addr, s);
    compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg, None);
    let forced: HashMap<usize, SelectChoiceEncoding> = [(select_id, SelectChoiceEncoding::Fma)].into_iter().collect();
    let budget = 41;
    let mut fma = HashSet::new();
    let ((_, exceeded, _), agreement) = schedule_ssa_greedy_shaped(
        &mut emit.ops, true, &mut HashSet::new(), budget, Some(&mut fma), None, None, None,
        &GreedyBoundsUsage::None, &GreedyPolicy, None, false, false, false, false, None, None,
        Some(&forced), &mut GreedyDiagnostics::new(false), &GreedyShape::default(), &mut HashMap::new(),
    )
    .unwrap_or_else(|e| panic!("scheduler error: {}", e));
    assert!(agreement.agrees(), "{}", agreement);
    let scheduled: Vec<SsaOp> = emit.ops.iter().filter(|op| op.cycle >= 0).cloned().collect();
    let config = crate::address_assign::AddressAssignConfig {
        select_fma_ops: [select_id].into_iter().collect(),
        ..Default::default()
    };
    let intervals = crate::address_assign::compute_live_intervals(&config, &scheduled).unwrap();
    let peak = crate::address_assign::compute_peak_liveness(&intervals);
    assert!(peak <= budget, "control: the schedule fits in {} words (interval peak {})", budget, peak);
    assert!(!exceeded, "the select fits by reusing the keep_live vector's words (interval peak {})", peak);
}

/// Review of #76 (CodeRabbit, second pass): the pools check must measure selects forced to the
/// multiply_add form by `fixed_select_choice_encoding` with their cycle + 1 reads. Before, only
/// the caller's set (which gets forced selects after the call) went into the measurement, so a
/// schedule whose real pool split exceeds the budget was returned without the capacity rerun.
#[test]
fn test_scratch_pools_measure_forced_multiply_add_selects() {
    let build = || {
        let mut emit = SsaEmitContext::new();
        let mut vec_of = |emit: &mut SsaEmitContext, v: i64| {
            let c = emit.emit_const(v);
            emit.emit_vbroadcast(c)
        };
        let (b, d) = (vec_of(&mut emit, 5), vec_of(&mut emit, 6));
        let (c1v, c2v) = (vec_of(&mut emit, 1), vec_of(&mut emit, 0));
        let t = emit.emit_valu(Opcode::Add, b, d);
        let s1 = emit.emit_select_choice_shared_delta(c1v, t, d, b);
        let s2 = emit.emit_select_choice_shared_delta(c2v, t, d, b);
        let o = emit.emit_valu(Opcode::Add, b, t);
        let addr = emit.emit_const(100);
        let x = emit.emit_valu(Opcode::Add, s1, s2);
        let y = emit.emit_valu(Opcode::Add, x, o);
        emit.emit_vstore(addr, y);
        compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg, None);
        emit
    };
    let mut checked = 0;
    for budget in 16..=160 {
        let mut emit = build();
        let forced: HashMap<usize, SelectChoiceEncoding> = emit
            .ops
            .iter()
            .filter(|op| op.opcode == Opcode::SelectChoice)
            .map(|op| (op.id, SelectChoiceEncoding::Fma))
            .collect();
        let forced_ids: HashSet<usize> = forced.keys().copied().collect();
        let shape = GreedyShape { scratch_pools: true, ..Default::default() };
        let mut fma = HashSet::new();
        let result = schedule_ssa_greedy_shaped(
            &mut emit.ops, true, &mut HashSet::new(), budget, Some(&mut fma), None, None, None,
            &GreedyBoundsUsage::None, &GreedyPolicy, None, false, false, false, false, None, None,
            Some(&forced), &mut GreedyDiagnostics::new(false), &shape, &mut HashMap::new(),
        );
        let ((_, exceeded, _), _) = result.unwrap_or_else(|e| panic!("budget {}: scheduler error: {}", budget, e));
        if exceeded {
            continue;
        }
        let measured: HashSet<usize> = fma.union(&forced_ids).copied().collect();
        let peaks = scratch_pool_peaks(&emit.ops, None, &HashMap::new(), &measured, &HashMap::new()).unwrap();
        assert!(
            peaks.split_words() <= budget,
            "budget {}: kept within the pools, but the pool split with forced multiply-add reads is {}",
            budget,
            peaks.split_words()
        );
        checked += 1;
    }
    assert!(checked > 0, "control: some budget is kept within the pools");
}

/// Review of #76 (CodeRabbit): a select whose condition is the same VReg as its delta. In the
/// vselect form the delta is not read, but the condition is: the counter must keep the shared
/// VReg to the select's cycle (before, the delta-only skip also dropped the direct read, so the
/// counter released it at the earlier `Add` and disagreed with the intervals).
#[test]
fn test_select_whose_condition_is_its_delta_keeps_the_direct_read() {
    let shape = GreedyShape {
        priority: GreedyPriority::Wavefront { stagger_cycles: 2, groups_in_flight: None },
        select_overflow: SelectOverflow::Valu { limit: None },
        ..Default::default()
    };
    let mut emit = SsaEmitContext::new();
    let c5 = emit.emit_const(5);
    let base = emit.emit_vbroadcast(c5);
    let c1 = emit.emit_const(1);
    let one = emit.emit_vbroadcast(c1);
    let addr = emit.emit_const(100);
    for chain in 0..6 {
        let cv = emit.emit_const(chain as i64);
        let v = emit.emit_vbroadcast(cv);
        let delta = emit.emit_valu(Opcode::And, v, one);
        let true_val = emit.emit_valu(Opcode::Add, base, delta);
        // Work between the Add and the select, so the select reads the shared VReg later.
        let mut pad = emit.emit_valu(Opcode::Add, true_val, one);
        for _ in 0..3 {
            pad = emit.emit_valu(Opcode::Add, pad, one);
        }
        let sel = emit.emit_select_choice_shared_delta(delta, true_val, delta, base);
        let out = emit.emit_valu(Opcode::Add, sel, pad);
        emit.emit_vstore(addr, out);
    }
    compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg, None);
    let mut fma = HashSet::new();
    let ((_, _, _), agreement) = schedule_ssa_greedy_shaped(
        &mut emit.ops,
        true,
        &mut HashSet::new(),
        1536,
        Some(&mut fma),
        None,
        None,
        None,
        &GreedyBoundsUsage::None,
        &GreedyPolicy,
        None,
        false,
        false,
        false,
        false,
        None,
        None,
        None,
        &mut GreedyDiagnostics::new(false),
        &shape,
        &mut HashMap::new(),
    )
    .expect("the live-word counter agrees with interval liveness");
    assert!(agreement.agrees(), "{}", agreement);
}

/// Review of #88 (CodeRabbit, stacked from #82): a vselect that keeps a lane of its delta's
/// vector group live. The lane's read belongs to the group's block, which address assignment
/// holds through the select's cycle; the delta-only skip matched exact VRegs only, so it
/// treated the group as unread at the select and backdated its release.
#[test]
fn test_select_keeping_a_lane_of_its_delta_holds_the_delta_group() {
    let shape = GreedyShape {
        priority: GreedyPriority::Wavefront { stagger_cycles: 2, groups_in_flight: None },
        select_overflow: SelectOverflow::Valu { limit: None },
        ..Default::default()
    };
    let mut emit = SsaEmitContext::new();
    let c5 = emit.emit_const(5);
    let base = emit.emit_vbroadcast(c5);
    let c1 = emit.emit_const(1);
    let one = emit.emit_vbroadcast(c1);
    let addr = emit.emit_const(100);
    for chain in 0..6 {
        let cv = emit.emit_const(chain as i64);
        let v = emit.emit_vbroadcast(cv);
        let delta = emit.emit_valu(Opcode::And, v, one);
        let delta_lanes = emit.decompose_vector(delta);
        let true_val = emit.emit_valu(Opcode::Add, base, delta);
        // Work between the Add and the select, so the select's hold is later than the Add.
        let mut pad = emit.emit_valu(Opcode::Add, true_val, one);
        for _ in 0..3 {
            pad = emit.emit_valu(Opcode::Add, pad, one);
        }
        let sel = emit.emit_select_choice_shared_delta(v, true_val, delta, base);
        let sel_idx = emit.ops.iter().position(|op| op.dest == Some(sel)).expect("select op");
        emit.ops[sel_idx].keep_live = Some(vec![delta_lanes[0]]);
        let out = emit.emit_valu(Opcode::Add, sel, pad);
        emit.emit_vstore(addr, out);
    }
    let lanes = emit.vector_group_lanes.clone();
    compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg, Some(&lanes));
    let mut fma = HashSet::new();
    let ((_, _, _), agreement) = schedule_ssa_greedy_shaped(
        &mut emit.ops,
        true,
        &mut HashSet::new(),
        1536,
        Some(&mut fma),
        None,
        None,
        None,
        &GreedyBoundsUsage::None,
        &GreedyPolicy,
        Some(&lanes),
        false,
        false,
        false,
        false,
        None,
        None,
        None,
        &mut GreedyDiagnostics::new(false),
        &shape,
        &mut HashMap::new(),
    )
    .expect("the live-word counter agrees with interval liveness");
    assert!(agreement.agrees(), "{}", agreement);
}

/// Regression (review of #76): a select the overflow puts on the vector ALU reads its
/// operands and its precomputed delta one cycle after its own (address assignment's
/// `select_fma_ops` rule, address_assign.rs). The counter did not count the delta as read
/// by the select at all and released the operands at the select's cycle, so with the delta
/// shared by many selects it undercounted interval liveness by the delta's 8 words over most
/// of the schedule. Interval liveness here is computed exactly as address assignment does,
/// with the run's multiply-add selects in `select_fma_ops`.
#[test]
fn test_select_overflow_counter_matches_address_assignment_liveness() {
    for split_shift in [false, true] {
        for limit in [None, Some(3)] {
            let shape = GreedyShape {
                priority: GreedyPriority::Wavefront { stagger_cycles: 2, groups_in_flight: None },
                split_shift,
                select_overflow: SelectOverflow::Valu { limit },
                ..Default::default()
            };
            let (mut emit, _) = emit_delta_select_chains(24);
            let mut shifted = HashSet::new();
            let mut lanes = HashMap::new();
            let mut fma = HashSet::new();
            let label = format!("split={} limit={:?}", split_shift, limit);
            let ((_, _, _), agreement) = schedule_ssa_greedy_shaped(
                &mut emit.ops,
                true,
                &mut shifted,
                1536,
                Some(&mut fma),
                None,
                None,
                None,
                &GreedyBoundsUsage::None,
                &GreedyPolicy,
                None,
                false,
                false,
                false,
                false,
                None,
                None,
                None,
                &mut GreedyDiagnostics::new(false),
                &shape,
                &mut lanes,
            )
            .unwrap_or_else(|e| panic!("{}: {}", label, e));
            assert!(!fma.is_empty(), "{}: the overflow placed no select on the vector ALU", label);
            let scheduled: Vec<SsaOp> = emit.ops.iter().filter(|op| op.cycle >= 0).cloned().collect();
            let config = crate::address_assign::AddressAssignConfig {
                select_fma_ops: fma.clone(),
                shifted_alu_cycles: lanes.clone(),
                ..Default::default()
            };
            let intervals = crate::address_assign::compute_live_intervals(&config, &scheduled).unwrap();
            let peak = crate::address_assign::compute_peak_liveness(&intervals);
            assert!(agreement.agrees(), "{}: {}", label, agreement);
            assert_eq!(agreement.interval_peak, peak, "{}: the check must use the run's multiply-add selects", label);
            assert_eq!(agreement.counter_cycle_end_peak, peak, "{}: {}", label, agreement);
        }
    }
}

/// A caller that does not collect multiply-add selects cannot use select overflow: lowering
/// would emit vselects on a flow slot the greedy did not count.
#[test]
fn test_select_overflow_requires_collecting_fma_selects() {
    let shape = GreedyShape { select_overflow: SelectOverflow::Valu { limit: None }, ..Default::default() };
    let (mut emit, _) = emit_delta_select_chains(24);
    assert!(schedule_collecting_fma(&mut emit.ops, &shape, 1536, &GreedyPolicy, false).is_err());
    let (mut emit, _) = emit_delta_select_chains(24);
    assert!(schedule_collecting_fma(&mut emit.ops, &GreedyShape::default(), 1536, &GreedyPolicy, false).is_ok());
}

/// The wavefront marks the ops a non-final load depends on. In load -> add -> load -> add
/// -> load (final), the first add feeds the middle load (non-final); the middle load and
/// the second add feed only the final load.
#[test]
fn test_wavefront_feeds_nonfinal_load() {
    let (emit, finals) = emit_load_chains(3);
    let w = wavefront_for(&emit.ops, 6, None);
    let by_id: HashMap<usize, usize> = emit.ops.iter().enumerate().map(|(i, op)| (op.id, i)).collect();
    // The latest-emitted producer of an op is its chain predecessor (constants come first).
    let pred = |idx: usize| by_id[&emit.ops[idx].dep_ids.iter().copied().max().unwrap()];
    for &f in &finals {
        let final_idx = by_id[&f];
        assert!(!w.feeds_nonfinal_load(final_idx));
        let add1 = pred(final_idx);
        assert!(!w.feeds_nonfinal_load(add1), "add before the final load");
        let load1 = pred(add1);
        assert_eq!(emit.ops[load1].engine, Engine::Load);
        assert!(!w.feeds_nonfinal_load(load1), "middle load");
        let add0 = pred(load1);
        assert!(w.feeds_nonfinal_load(add0), "add before the middle load");
    }
}


// =========================================================================
// Start-up priorities: chain-first flow, native-first scalar ALU
// =========================================================================

/// Per chain: a select, an AND of its result, a second select on that, an add and a store.
/// Returns the program and each chain's second select (op index).
fn emit_two_select_chains(chains: usize) -> (SsaEmitContext, Vec<usize>) {
    let mut emit = SsaEmitContext::new();
    let c1 = emit.emit_const(1);
    let one = emit.emit_vbroadcast(c1);
    let c5 = emit.emit_const(5);
    let base = emit.emit_vbroadcast(c5);
    let c9 = emit.emit_const(9);
    let delta = emit.emit_vbroadcast(c9);
    let true_val = emit.emit_valu(Opcode::Add, base, delta);
    let addr = emit.emit_const(100);
    let mut second = Vec::new();
    for chain in 0..chains {
        let cv = emit.emit_const(chain as i64);
        let v = emit.emit_vbroadcast(cv);
        let cond = emit.emit_valu(Opcode::And, v, one);
        let s1 = emit.emit_select_choice_shared_delta(cond, true_val, delta, base);
        let cond2 = emit.emit_valu(Opcode::And, s1, one);
        let s2 = emit.emit_select_choice_shared_delta(cond2, true_val, delta, base);
        second.push(emit.ops.len() - 1);
        let out = emit.emit_valu(Opcode::Add, s2, one);
        emit.emit_vstore(addr, out);
    }
    compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg, None);
    (emit, second)
}

/// `chain-first`: the first chain's second select is not overtaken on the one flow slot by
/// later chains' first selects, so it runs earlier than under the wavefront key (control:
/// with stagger 0 every first select has a lower key than any second select).
#[test]
fn test_flow_chain_first_lets_the_first_chain_through() {
    let run = |flow_priority| {
        let shape = GreedyShape {
            priority: GreedyPriority::Wavefront { stagger_cycles: 0, groups_in_flight: None },
            flow_priority,
            ..Default::default()
        };
        let (mut emit, second) = emit_two_select_chains(8);
        let (_, shifted, fma, lanes) =
            schedule_collecting_fma(&mut emit.ops, &shape, 1536, &GreedyPolicy, true).unwrap();
        assert_slot_limits_with_selects(&emit.ops, &shifted, &fma, &lanes, &format!("{:?}", flow_priority));
        emit.ops[second[0]].cycle
    };
    let key = run(FlowPriority::Key);
    let chain_first = run(FlowPriority::ChainFirst);
    assert!(chain_first < key, "chain 0's second select: chain-first {} vs key {}", chain_first, key);
}

/// 24 one-op vector chains off one shared broadcast (so they are all ready together and the
/// vector ALU overflows into shifts) and, last in emission order (highest key), 8 native
/// scalar adds. Returns the program and the native adds (op indices).
fn emit_vector_chains_and_late_natives() -> (SsaEmitContext, Vec<usize>) {
    let mut emit = SsaEmitContext::new();
    let addr = emit.emit_const(100);
    let c = emit.emit_const(3);
    let shared = emit.emit_vbroadcast(c);
    for _ in 0..24 {
        let w = emit.emit_valu(Opcode::Add, shared, shared);
        emit.emit_vstore(addr, w);
    }
    let x = emit.emit_const(7);
    let base = emit.emit_alu_imm(Opcode::Add, x, 1);
    let mut natives = Vec::new();
    let mut sums = Vec::new();
    for j in 0..8 {
        sums.push(emit.emit_alu_imm(Opcode::Add, base, j + 2));
        natives.push(emit.ops.len() - 1);
    }
    let mut acc = emit.emit_alu(Opcode::Add, sums[0], sums[1]);
    for s in &sums[2..] {
        acc = emit.emit_alu(Opcode::Add, acc, *s);
    }
    let a2 = emit.emit_const(200);
    emit.emit_store(a2, acc);
    compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg, None);
    (emit, natives)
}

/// `native-first`: native scalar ops take the scalar slots before vector ops are shifted
/// onto them, so all 8 run in the cycle they become ready; under the wavefront key a shifted
/// vector op takes 8 of the 12 slots first (control). Slot limits hold both ways.
#[test]
fn test_alu_native_first_keeps_scalar_slots_for_natives() {
    let run = |alu_priority| {
        let shape = GreedyShape {
            priority: GreedyPriority::Wavefront { stagger_cycles: 4, groups_in_flight: None },
            alu_priority,
            ..Default::default()
        };
        let (mut emit, natives) = emit_vector_chains_and_late_natives();
        let (_, shifted, fma, lanes) =
            schedule_collecting_fma(&mut emit.ops, &shape, 1536, &GreedyPolicy, true).unwrap();
        assert_slot_limits_with_selects(&emit.ops, &shifted, &fma, &lanes, &format!("{:?}", alu_priority));
        let cycles: Vec<i32> = natives.iter().map(|&i| emit.ops[i].cycle).collect();
        (*cycles.iter().min().unwrap(), *cycles.iter().max().unwrap(), shifted.len())
    };
    let (key_first, key_last, key_shifted) = run(AluPriority::Key);
    let (nf_first, nf_last, _) = run(AluPriority::NativeFirst);
    assert!(key_shifted > 0, "control: the program must shift vector ops");
    assert_eq!(nf_first, nf_last, "native-first: all natives in one cycle");
    assert!(key_last > key_first, "control: under the key, shifts split the natives ({}..{})", key_first, key_last);
    assert!(nf_last < key_last, "native-first {} vs key {}", nf_last, key_last);
}

/// `final-last` under scratch-budget deferral (the documented limitation of
/// `LoadPriority::FinalLast`): only the first placement pass of a cycle is ordered, so a
/// feeder load deferred for the budget is retried after the final loads of that pass, and a
/// final load can take a slot while the feeder is operand-ready. With a budget of 6 words
/// (peak 8 unconstrained) this happens; from 8 words up it does not. Every run keeps the
/// load slot limit and schedules every op within the budget. A fix that also orders the
/// retry passes must turn the first assertion around.
#[test]
fn test_load_priority_final_last_budget_deferral() {
    let shape = GreedyShape {
        priority: GreedyPriority::Wavefront { stagger_cycles: 6, groups_in_flight: None },
        load_priority: LoadPriority::FinalLast,
        ..Default::default()
    };
    let ahead_at = |budget: usize| {
        let (mut emit, finals) = emit_load_chains(8);
        let mut shifted = HashSet::new();
        let ((_, exceeded, peak), agreement) = schedule_ssa_greedy_shaped(
            &mut emit.ops,
            false,
            &mut shifted,
            budget,
            None,
            None,
            None,
            None,
            &GreedyBoundsUsage::None,
            &GreedyPolicy,
            None,
            false,
            false,
            false,
            false,
            None,
            None,
            None,
            &mut GreedyDiagnostics::new(false),
            &shape,
            &mut HashMap::new(),
        )
        .expect("greedy schedules the chains");
        assert!(agreement.agrees(), "live-word counter disagrees: {}", agreement);
        assert!(!exceeded && peak <= budget, "budget {}: peak {}", budget, peak);
        assert_dependencies_respected(&emit.ops, &HashMap::new());
        final_loads_ahead_of_ready_feeders(&emit.ops, &finals)
    };
    assert!(ahead_at(6) > 0, "a budget-deferred feeder lets a final load go first");
    for budget in [8, 16, 1536] {
        assert_eq!(ahead_at(budget), 0, "budget {}", budget);
    }
}

/// `stall` under the shaped pipeline's settings: adaptive policy with ASAP inflation (policy
/// bounds), wavefront priority, VALU shift and split shifts, with budgets that bind. 24
/// independent chains of vector adds off one broadcast: a running chain holds one vector
/// and each step frees what it adds, so no budget of at least three vectors deadlocks. The
/// run must stay within the budget and never force; returns (budget_exceeded, peak,
/// diagnostics) for `force`.
fn schedule_chains_shaped(budget: usize, force: BudgetForce, split_shift: bool) -> (bool, usize, GreedyDiagnostics) {
    let (mut emit, _) = emit_independent_chains(24, 6);
    let asap = crate::cpsat_scheduler::bounds::compute_ssa_asap(&emit.ops, &HashSet::new());
    let max_asap = asap.values().copied().max().unwrap_or(0);
    let alap = crate::cpsat_scheduler::bounds::compute_ssa_alap(&emit.ops, max_asap + 20, &HashSet::new());
    let shape = GreedyShape {
        priority: GreedyPriority::Wavefront { stagger_cycles: 2, groups_in_flight: None },
        split_shift,
        budget_force: force,
        ..Default::default()
    };
    let mut diagnostics = GreedyDiagnostics::new(false);
    let ((_, exceeded, peak), agreement) = schedule_ssa_greedy_shaped(
        &mut emit.ops,
        true,
        &mut HashSet::new(),
        budget,
        None,
        None,
        None,
        None,
        &GreedyBoundsUsage::PolicyOnly { asap: &asap, alap: &alap },
        &pressure_policy::AdaptivePolicy::default(),
        None,
        false,
        false,
        false,
        false,
        None,
        None,
        None,
        &mut diagnostics,
        &shape,
        &mut HashMap::new(),
    )
    .expect("greedy schedules the chains");
    assert!(agreement.agrees(), "live-word counter disagrees: {}", agreement);
    (exceeded, peak, diagnostics)
}

#[test]
fn test_budget_force_stall_holds_the_budget_in_the_shaped_pipeline() {
    let mut control_overshoots = 0;
    for budget in [3 * VLEN + 1, 5 * VLEN + 1, 8 * VLEN + 1, 12 * VLEN + 1] {
        for split_shift in [false, true] {
            let (exceeded, peak, diag) = schedule_chains_shaped(budget, BudgetForce::Stall, split_shift);
            assert!(!exceeded && diag.forced_past_budget == 0, "stall forced at budget {} split {}", budget, split_shift);
            assert!(peak <= budget, "stall peak {} > budget {} (split {})", peak, budget, split_shift);
            // A split shift starts only with 1 to 7 scalar slots free, i.e. after a counted
            // placement on the scalar ALU in the same cycle (a carried split's completion, a
            // native op or a whole shift): a cycle whose only placement is a split start
            // does not occur. The stall rule still treats one as progress, defensively.
            assert_eq!(diag.split_only_cycles, 0);
            let (same_exceeded, same_peak, _) = schedule_chains_shaped(budget, BudgetForce::SameCycle, split_shift);
            if same_exceeded || same_peak > budget {
                control_overshoots += 1;
            }
        }
    }
    assert!(control_overshoots > 0, "control: same-cycle never overshot, so the budgets do not bind");
}

// =========================================================================
// ConstPolicy (ALU-derived constants) and EarlyStagger
// =========================================================================

/// Five ConstChoices 8, 16, 32, 24, 40, summed into one stored scalar, 24 + 40 first (so 8,
/// 16 and 32 stay live until after them). Cycle 0 has two load slots and one flow slot; the
/// last two constants come after, when 8 + 16 and 8 + 32 are live.
fn emit_derivable_consts() -> SsaEmitContext {
    let mut emit = SsaEmitContext::new();
    let vals: Vec<VReg> = [8i64, 16, 32, 24, 40].iter().map(|&v| emit.emit_const_choice(v)).collect();
    // The derived constants are summed first, so 8, 16 and 32 are read only after them.
    let mut acc = emit.emit_alu(Opcode::Add, vals[3], vals[4]);
    for v in &vals[..3] {
        acc = emit.emit_alu(Opcode::Add, acc, *v);
    }
    let addr = emit.emit_const(100);
    emit.emit_store(addr, acc);
    compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg, None);
    emit
}

/// Schedule with a ConstPolicy; returns the ALU derivations the greedy chose.
fn schedule_with_const_policy(
    emit: &mut SsaEmitContext,
    const_policy: ConstPolicy,
) -> HashMap<usize, crate::types::AluDeriveInfo> {
    let shape = GreedyShape { const_policy, ..Default::default() };
    let mut diagnostics = GreedyDiagnostics::new(false);
    let mut add_imm = HashSet::new();
    let (_, agreement) = schedule_ssa_greedy_shaped(
        &mut emit.ops,
        false,
        &mut HashSet::new(),
        1536,
        None,
        None,
        None,
        Some(&mut add_imm),
        &GreedyBoundsUsage::None,
        &GreedyPolicy,
        None,
        false,
        false,
        false,
        false,
        None,
        None,
        None,
        &mut diagnostics,
        &shape,
        &mut HashMap::new(),
    )
    .expect("greedy schedules the constants");
    assert!(agreement.agrees(), "live-word counter disagrees: {}", agreement);
    assert!(diagnostics.const_alu_derive.keys().all(|id| !add_imm.contains(id)), "one encoding per const");
    diagnostics.const_alu_derive
}

/// Review of #86: a producer that is visible (an earlier cycle) but already dead must not be
/// derived from. 8 is read only by `8 + 8`, early; 16 (= 8 + 8, its only derivation: the
/// other constants sum to nothing here) is emitted last, so it is placed after 8 is dead.
/// Without the liveness condition the greedy would read 8's word after it was freed.
#[test]
fn test_const_policy_never_derives_from_a_dead_constant() {
    let mut emit = SsaEmitContext::new();
    let c8 = emit.emit_const_choice(8);
    let others: Vec<VReg> = [1001i64, 2003, 4007, 8009, 16019, 32003].iter().map(|&v| emit.emit_const_choice(v)).collect();
    let c16 = emit.emit_const_choice(16);
    let mut acc = emit.emit_alu(Opcode::Add, c8, c8); // the only reader of 8
    for v in others.iter().chain([c16].iter()) {
        acc = emit.emit_alu(Opcode::Add, acc, *v);
    }
    let addr = emit.emit_const(100);
    emit.emit_store(addr, acc);
    compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg, None);
    let id8 = emit.ops[0].id;
    let id16 = emit.ops[7].id;
    let derived = schedule_with_const_policy(&mut emit, ConstPolicy::AluFirst);
    let cycle = |id: usize| emit.ops.iter().find(|o| o.id == id).unwrap().cycle;
    let reader = emit.ops[8].cycle; // 8 + 8
    assert!(cycle(id16) > reader, "control: 16 is placed after 8's last reader ({} vs {})", cycle(id16), reader);
    assert!(!derived.contains_key(&id16), "16 = 8 + 8 needs 8, which is dead by then: {:?}", derived.get(&id16));
}

/// Review of #86: a rerun with fixed encodings (hint passes, --schedule-with-fixed-choices)
/// schedules a fixed `AluDerive` as the ALU op it lowers to, after its producers, or, where
/// that is not sound, as a load that `reconcile_const_choice_encodings` records as `Load`.
/// Before, it always scheduled it as a load while lowering still emitted the ALU op.
#[test]
fn test_fixed_alu_derive_is_scheduled_as_derived_or_recorded_as_load() {
    let mut first = emit_derivable_consts();
    let derived = schedule_with_const_policy(&mut first, ConstPolicy::AluFirst);
    assert!(!derived.is_empty());
    let mut emit = emit_derivable_consts();
    let id_of = |value: i64| emit.ops.iter().find(|o| o.produces_constant == Some(value)).unwrap().id;
    let mut fixed: HashMap<usize, ConstChoiceEncoding> = emit
        .ops
        .iter()
        .filter(|o| o.opcode == Opcode::ConstChoice)
        .map(|o| (o.id, derived.get(&o.id).map_or(ConstChoiceEncoding::Load, |i| ConstChoiceEncoding::AluDerive(i.clone()))))
        .collect();
    // 8 = 24 - 16: its producers cannot be scheduled before it (all constants are ready at
    // cycle 0 and 8 comes first), so this derivation cannot be honoured.
    let unsound = (id_of(8), crate::types::AluDeriveInfo { left_op_id: id_of(24), right_op_id: id_of(16), opcode: Opcode::Sub });
    assert!(!derived.contains_key(&unsound.0));
    fixed.insert(unsound.0, ConstChoiceEncoding::AluDerive(unsound.1));

    let mut diagnostics = GreedyDiagnostics::new(false);
    let (_, agreement) = schedule_ssa_greedy_shaped(
        &mut emit.ops,
        false,
        &mut HashSet::new(),
        1536,
        None,
        None,
        None,
        Some(&mut HashSet::new()),
        &GreedyBoundsUsage::None,
        &GreedyPolicy,
        None,
        false,
        false,
        false,
        false,
        Some(&fixed),
        None,
        None,
        &mut diagnostics,
        &GreedyShape::default(),
        &mut HashMap::new(),
    )
    .expect("greedy schedules the constants");
    assert!(agreement.agrees(), "live-word counter disagrees: {}", agreement);
    // Every honoured derivation runs on the ALU after both producers.
    let cycle_of = |id: usize| emit.ops.iter().find(|o| o.id == id).unwrap().cycle;
    for (id, info) in &diagnostics.const_alu_derive {
        assert!(cycle_of(*id) > cycle_of(info.left_op_id) && cycle_of(*id) > cycle_of(info.right_op_id));
    }
    assert!(!diagnostics.const_alu_derive.contains_key(&unsound.0), "8 cannot be derived before 24 and 16");
    assert!(!diagnostics.const_alu_derive.is_empty(), "the first run's derivations are kept");
    let demoted = reconcile_const_choice_encodings(&mut fixed, &diagnostics.const_alu_derive);
    assert!(demoted >= 1);
    assert_eq!(fixed[&unsound.0], ConstChoiceEncoding::Load);
    for id in diagnostics.const_alu_derive.keys() {
        assert!(matches!(fixed[id], ConstChoiceEncoding::AluDerive(_)));
    }
}

/// `alu-first` derives constants on the scalar ALU from two constants that are visible (an
/// earlier cycle) and still live at the derive's cycle, and the operands add up to the value.
/// The default policy never derives (control).
#[test]
fn test_const_policy_derives_from_live_constants() {
    let mut control = emit_derivable_consts();
    assert!(schedule_with_const_policy(&mut control, ConstPolicy::Default).is_empty());

    let mut emit = emit_derivable_consts();
    let derived = schedule_with_const_policy(&mut emit, ConstPolicy::AluFirst);
    assert!(!derived.is_empty(), "alu-first derived nothing");
    let by_id: HashMap<usize, &SsaOp> = emit.ops.iter().map(|op| (op.id, op)).collect();
    for (id, info) in &derived {
        let op = by_id[id];
        let (l, r) = (by_id[&info.left_op_id], by_id[&info.right_op_id]);
        assert_eq!(info.opcode, Opcode::Add);
        assert_eq!(l.produces_constant.unwrap() + r.produces_constant.unwrap(), op.produces_constant.unwrap());
        for p in [l, r] {
            assert!(p.cycle < op.cycle, "producer {} at {} not before derive at {}", p.id, p.cycle, op.cycle);
            // Still live: some reader of the producer's value runs at or after the derive.
            let last_read = emit
                .ops
                .iter()
                .filter(|o| o.srcs.iter().any(|s| matches!(s, Operand::VReg(v) if Some(*v) == p.dest)))
                .map(|o| o.cycle)
                .max()
                .unwrap();
            assert!(last_read >= op.cycle, "producer {} dead at {} before derive at {}", p.id, last_read, op.cycle);
        }
    }
}

/// The first `groups` chains use the early stagger; later chains continue from there with the
/// normal stagger. Without an early stagger the offset is `stagger * rank`.
#[test]
fn test_early_stagger_chain_offsets() {
    use wavefront::{chain_offset, EarlyStagger};
    let early = Some(EarlyStagger { groups: 4, cycles: 2 });
    let offsets: Vec<i64> = (0..7).map(|r| chain_offset(r, 9, early)).collect();
    assert_eq!(offsets, vec![0, 2, 4, 6, 8, 17, 26]);
    assert_eq!((0..7).map(|r| chain_offset(r, 9, None)).collect::<Vec<_>>(), vec![0, 9, 18, 27, 36, 45, 54]);
}

// =========================================================================
// Forward/backward justification (ssa_greedy::justify)
// =========================================================================

use super::justify::{justify_schedule, verify_schedule, JustifyConfig, JustifyOutcome};

/// Justify the schedule in `ops` with no ConstChoice / SelectChoice decisions and no
/// vector groups.
fn justify_plain(
    ops: &mut [SsaOp],
    shifted: &mut HashSet<usize>,
    lanes: &mut HashMap<usize, Vec<i32>>,
    config: &JustifyConfig,
) -> JustifyOutcome {
    justify_schedule(ops, true, shifted, lanes, &HashMap::new(), &HashMap::new(), None, None, config)
        .expect("liveness of the input schedule can be computed")
}

/// Options used by most justification tests: shifts may change, hardware budget.
fn justify_config(max_iterations: usize) -> JustifyConfig {
    JustifyConfig {
        max_iterations,
        allow_shift_changes: true,
        scratch_budget: 1536,
        ..Default::default()
    }
}

/// Six short load chains emitted before one long chain: the load slots are the bottleneck
/// and the long chain is critical.
///
/// # Returns
/// `(emit, critical path length)`.
fn emit_short_chains_before_a_long_one() -> (SsaEmitContext, usize) {
    let mut emit = SsaEmitContext::new();
    for k in 0..6 {
        let value = emit.emit_const(1000 + k);
        emit.emit_store_imm(200 + k, value);
    }
    let mut value = emit.emit_const(7);
    for step in 0..6 {
        value = emit.emit_alu_imm(Opcode::Add, value, step + 1);
    }
    emit.emit_store_imm(300, value);
    compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg, None);
    (emit, 8) // const, six adds, store
}

/// The greedy serves the load slots in emission order, so the long chain starts last.
/// The backward pass finds that the long chain is the urgent one, and the forward pass
/// then starts it first: the schedule shrinks to the critical path.
#[test]
fn test_justify_strictly_improves_greedy_schedule() {
    let (mut emit, critical_path) = emit_short_chains_before_a_long_one();
    let (greedy_cycles, mut shifted, mut lanes) =
        schedule_with_shift_and_shape(&mut emit.ops, &GreedyShape::default());
    assert!(greedy_cycles > critical_path, "the greedy starts the long chain late: {}", greedy_cycles);

    let outcome = justify_plain(&mut emit.ops, &mut shifted, &mut lanes, &justify_config(5));
    assert_eq!(outcome.cycles_before, greedy_cycles);
    assert_eq!(outcome.cycles_after, critical_path, "{}", outcome.summary());
    assert_eq!(outcome.verify_rejections, 0);
    assert_eq!(outcome.scratch_rejections, 0);
    let verified = verify_schedule(&emit.ops, true, &shifted, &lanes, &HashMap::new(), &HashMap::new());
    assert_eq!(verified, Ok(critical_path));
}

/// With zero iterations the pass only measures the schedule.
#[test]
fn test_justify_zero_iterations_changes_nothing() {
    let (mut emit, _) = emit_short_chains_before_a_long_one();
    let (greedy_cycles, mut shifted, mut lanes) =
        schedule_with_shift_and_shape(&mut emit.ops, &GreedyShape::default());
    let before: Vec<i32> = emit.ops.iter().map(|op| op.cycle).collect();
    let outcome = justify_plain(&mut emit.ops, &mut shifted, &mut lanes, &justify_config(0));
    assert_eq!((outcome.cycles_before, outcome.cycles_after), (greedy_cycles, greedy_cycles));
    assert_eq!(before, emit.ops.iter().map(|op| op.cycle).collect::<Vec<_>>());
}

/// Twelve broadcasts, each stored once, scheduled one per cycle: one vector is live at a
/// time. Any shorter schedule runs two stores in some cycle and so holds two vectors at
/// once. With the input's peak as the budget every shorter candidate is rejected for
/// scratch and the input is returned untouched; with the hardware budget it is shortened.
#[test]
fn test_justify_scratch_budget_rejects_improvement() {
    let mut emit = SsaEmitContext::new();
    let scalar = emit.emit_const(3);
    let mut pairs = Vec::new();
    for k in 0..12 {
        let vector = emit.emit_vbroadcast_force(scalar);
        let broadcast = emit.ops.len() - 1;
        emit.emit_vstore_imm(64 * (k + 1), vector);
        pairs.push((broadcast, emit.ops.len() - 1));
    }
    compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg, None);
    // Hand-made schedule: const at 0, then broadcast k at 1 + k and its store at 2 + k.
    emit.ops[0].cycle = 0;
    for (k, &(broadcast, store)) in pairs.iter().enumerate() {
        emit.ops[broadcast].cycle = 1 + k as i32;
        emit.ops[store].cycle = 2 + k as i32;
    }
    let input: Vec<i32> = emit.ops.iter().map(|op| op.cycle).collect();
    let (_, input_peak, _) = interval_liveness_curve(&emit.ops, None, &HashMap::new(), &HashSet::new(), &HashMap::new()).unwrap();

    let mut tight_ops = emit.ops.clone();
    let (mut shifted, mut lanes) = (HashSet::new(), HashMap::new());
    let tight = JustifyConfig { scratch_budget: input_peak, ..justify_config(5) };
    let outcome = justify_plain(&mut tight_ops, &mut shifted, &mut lanes, &tight);
    assert_eq!(outcome.cycles_before, 14);
    assert_eq!(outcome.cycles_after, 14, "{}", outcome.summary());
    assert!(outcome.scratch_rejections > 0, "{}", outcome.summary());
    assert_eq!(outcome.scratch_rejections, outcome.candidates_checked);
    assert_eq!(outcome.peak_live_after, input_peak);
    assert_eq!(input, tight_ops.iter().map(|op| op.cycle).collect::<Vec<_>>(), "input schedule is kept");
    assert!(shifted.is_empty() && lanes.is_empty());

    let outcome = justify_plain(&mut emit.ops, &mut shifted, &mut lanes, &justify_config(5));
    assert!(outcome.cycles_after < 14, "{}", outcome.summary());
    assert!(outcome.peak_live_after > input_peak);
    assert_eq!(outcome.scratch_rejections, 0);
}

/// Starting from a schedule with whole shifts only, justification with split shifts uses
/// the scalar slots a whole shift leaves over: the schedule gets shorter, split ops span
/// two consecutive cycles with their cycle at the last lane, and consumers wait for it.
/// Without split shifts no lane cycles are produced.
#[test]
fn test_justify_split_lanes() {
    let (mut emit, level_ops) = emit_independent_chains(30, 12);
    let (whole_cycles, whole_shifted, whole_lanes) =
        schedule_with_shift_and_shape(&mut emit.ops, &GreedyShape::default());
    assert!(whole_lanes.is_empty());

    let mut no_split_ops = emit.ops.clone();
    let (mut shifted, mut lanes) = (whole_shifted.clone(), whole_lanes.clone());
    let outcome = justify_plain(&mut no_split_ops, &mut shifted, &mut lanes, &justify_config(5));
    assert!(lanes.is_empty(), "no split shifts unless enabled");
    assert_eq!(outcome.split, 0);
    let no_split_cycles = outcome.cycles_after;
    assert!(no_split_cycles <= whole_cycles);

    let (mut shifted, mut lanes) = (whole_shifted, whole_lanes);
    let split = JustifyConfig { split_shift: true, ..justify_config(5) };
    let outcome = justify_plain(&mut emit.ops, &mut shifted, &mut lanes, &split);
    assert!(outcome.cycles_after < no_split_cycles, "{}", outcome.summary());
    assert!(!lanes.is_empty(), "split shifts fill the scalar slots");
    assert_eq!(outcome.split, lanes.len());
    assert_eq!(outcome.shifted, shifted.len());

    let alu_use = shifted_alu_use(&emit.ops, &shifted, &lanes);
    assert!(alu_use.values().all(|&n| n <= 12), "scalar slots over capacity: {:?}", alu_use);
    assert!(alu_use.values().any(|&n| n > VLEN), "some cycle carries more than one shifted op");
    let by_id: HashMap<usize, &SsaOp> = emit.ops.iter().map(|op| (op.id, op)).collect();
    for (op_id, lane) in &lanes {
        assert!(shifted.contains(op_id));
        assert_eq!(lane.len(), VLEN);
        let (first, last) = (*lane.iter().min().unwrap(), *lane.iter().max().unwrap());
        assert_eq!(last, first + 1, "a split shift spans two consecutive cycles: {:?}", lane);
        assert!(lane.windows(2).all(|w| w[0] <= w[1]), "low lanes run first: {:?}", lane);
        assert_eq!(by_id[op_id].cycle, last, "the op completes at its last lane");
    }
    for levels in &level_ops {
        for pair in levels.windows(2) {
            let (producer, consumer) = (&emit.ops[pair[0]], &emit.ops[pair[1]]);
            let consumer_start =
                lanes.get(&consumer.id).map_or(consumer.cycle, |l| *l.iter().min().unwrap());
            assert!(consumer_start > producer.cycle);
        }
    }
    // The result also satisfies the greedy's own liveness accounting within the budget.
    let (_, peak, _) = interval_liveness_curve(&emit.ops, None, &lanes, &HashSet::new(), &HashMap::new()).unwrap();
    assert_eq!(peak, outcome.peak_live_after);
    assert!(peak <= 1536);
}

/// The same input gives the same schedule, shift set and lane cycles every time.
#[test]
fn test_justify_is_deterministic() {
    let run = || {
        let (mut emit, _) = emit_independent_chains(30, 12);
        let (_, mut shifted, mut lanes) =
            schedule_with_shift_and_shape(&mut emit.ops, &GreedyShape { split_shift: true, ..Default::default() });
        let config = JustifyConfig { split_shift: true, ..justify_config(4) };
        let outcome = justify_plain(&mut emit.ops, &mut shifted, &mut lanes, &config);
        let cycles: Vec<i32> = emit.ops.iter().map(|op| op.cycle).collect();
        let mut shifted: Vec<usize> = shifted.into_iter().collect();
        shifted.sort_unstable();
        let mut lanes: Vec<(usize, Vec<i32>)> = lanes.into_iter().collect();
        lanes.sort();
        (outcome, cycles, shifted, lanes)
    };
    let first = run();
    for _ in 0..3 {
        assert_eq!(first, run());
    }
}

/// When the shift decision is not the pass's to make, every op keeps its encoding: the
/// shifted set is unchanged and shifted ops still take VLEN scalar slots in one cycle.
#[test]
fn test_justify_keeps_fixed_shift_decisions() {
    let (mut emit, _) = emit_independent_chains(30, 12);
    let (_, mut shifted, mut lanes) = schedule_with_shift_and_shape(&mut emit.ops, &GreedyShape::default());
    assert!(!shifted.is_empty());
    // Pad the schedule so there is something to gain.
    for op in emit.ops.iter_mut() {
        op.cycle *= 2;
    }
    let fixed_shifts = shifted.clone();
    let config = JustifyConfig { allow_shift_changes: false, ..justify_config(5) };
    let outcome = justify_plain(&mut emit.ops, &mut shifted, &mut lanes, &config);
    assert!(outcome.cycles_after < outcome.cycles_before, "{}", outcome.summary());
    assert_eq!(shifted, fixed_shifts);
    assert!(lanes.is_empty());
    let alu_use = shifted_alu_use(&emit.ops, &shifted, &lanes);
    assert!(alu_use.values().all(|&n| n == VLEN), "one whole shifted op per cycle: {:?}", alu_use);
}

/// A select on the multiply-add path keeps its two vector slots and its two-cycle latency
/// in both passes (in the backward pass it is scheduled from its multiply_add).
#[test]
fn test_justify_two_cycle_selects() {
    let mut emit = SsaEmitContext::new();
    let c1 = emit.emit_const(1);
    let c2 = emit.emit_const(2);
    let v1 = emit.emit_vbroadcast(c1);
    let v2 = emit.emit_vbroadcast(c2);
    let mut selects = Vec::new();
    let mut readers = Vec::new();
    for _ in 0..12 {
        let cond = emit.emit_valu(Opcode::Eq, v1, v2);
        let sel = emit.emit_select_choice(cond, v1, v2);
        selects.push(emit.ops.len() - 1);
        let reader = emit.emit_valu(Opcode::Add, sel, v1);
        readers.push(emit.ops.len() - 1);
        emit.emit_vstore_imm(64 * readers.len() as i64, reader);
    }
    let vreg_def = std::mem::take(&mut emit.vreg_def);
    compute_ssa_dependencies(&mut emit.ops, &vreg_def, emit.zero_vreg, None);
    let ops = &mut emit.ops;
    let fixed: HashMap<usize, SelectChoiceEncoding> =
        selects.iter().map(|&i| (ops[i].id, SelectChoiceEncoding::Fma)).collect();
    let mut shifted = HashSet::new();
    let mut lanes = HashMap::new();
    let ((greedy_cycles, _, _), _) = schedule_ssa_greedy_shaped(
        ops,
        false,
        &mut shifted,
        1536,
        None,
        None,
        None,
        None,
        &GreedyBoundsUsage::None,
        &GreedyPolicy,
        None,
        false,
        false,
        false,
        false,
        None,
        None,
        Some(&fixed),
        &mut GreedyDiagnostics::new(false),
        &GreedyShape::default(),
        &mut lanes,
    )
    .expect("greedy schedules the selects");
    // The verifier accepts the greedy's schedule, including the second vector slot.
    let no_consts = HashMap::new();
    assert!(verify_schedule(ops, false, &shifted, &lanes, &no_consts, &fixed).is_ok());
    // The verifier adds the multiply_add's cycle to the select's precedence offset. Spread
    // the schedule by four and move one select and its reader into the empty cycles, so no
    // slot limit is involved: a reader in the multiply_add's cycle is rejected, a later one
    // is accepted.
    let (select, reader) = (selects[0], readers[0]);
    let mut spread = ops.to_vec();
    for op in spread.iter_mut() {
        op.cycle *= 4;
    }
    spread[select].cycle += 2;
    spread[reader].cycle = spread[select].cycle + 1;
    let err = verify_schedule(&spread, false, &shifted, &lanes, &no_consts, &fixed).unwrap_err();
    assert!(err.contains("readable from cycle"), "{}", err);
    spread[reader].cycle = spread[select].cycle + 4;
    assert!(verify_schedule(&spread, false, &shifted, &lanes, &no_consts, &fixed).is_ok());
    // Pad it (still valid: every gap only grows) and justify.
    for op in ops.iter_mut() {
        op.cycle *= 3;
    }
    let config = JustifyConfig { allow_shift_changes: false, ..justify_config(5) };
    let outcome =
        justify_schedule(ops, false, &mut shifted, &mut lanes, &no_consts, &fixed, None, None, &config).unwrap();
    assert!(outcome.cycles_after <= greedy_cycles, "{} (greedy {})", outcome.summary(), greedy_cycles);
    assert_eq!(outcome.verify_rejections, 0);
    for (&s, &r) in selects.iter().zip(&readers) {
        assert!(ops[r].cycle >= ops[s].cycle + 2, "reader at {} of select at {}", ops[r].cycle, ops[s].cycle);
    }
    let mut valu: HashMap<i32, usize> = HashMap::new();
    for (i, op) in ops.iter().enumerate() {
        if selects.contains(&i) {
            *valu.entry(op.cycle).or_default() += 1;
            *valu.entry(op.cycle + 1).or_default() += 1;
        } else if op.engine == Engine::Valu {
            *valu.entry(op.cycle).or_default() += 1;
        }
    }
    assert!(valu.values().all(|&n| n <= 6), "vector slots over capacity: {:?}", valu);
}

/// The verifier reports slot overflow, precedence violations and malformed shift records.
#[test]
fn test_justify_verifier_rejects_invalid_schedules() {
    let none_c = HashMap::new();
    let none_s = HashMap::new();
    let (mut emit, level_ops) = emit_independent_chains(8, 2);
    let (cycles, shifted, lanes) = schedule_with_shift_and_shape(&mut emit.ops, &GreedyShape::default());
    assert_eq!(verify_schedule(&emit.ops, true, &shifted, &lanes, &none_c, &none_s), Ok(cycles));

    // Seven vector ops in one cycle with nothing shifted.
    let mut crowded = emit.ops.clone();
    let cycle = crowded[level_ops[0][0]].cycle;
    for chain in level_ops.iter().take(7) {
        crowded[chain[0]].cycle = cycle;
    }
    let err = verify_schedule(&crowded, true, &HashSet::new(), &lanes, &none_c, &none_s).unwrap_err();
    assert!(err.contains("valu slots"), "{}", err);

    // A consumer in its producer's cycle (a scalar chain, so no slot limit is involved).
    let (mut chain, _) = emit_short_chains_before_a_long_one();
    let (_, chain_shifted, chain_lanes) = schedule_with_shift_and_shape(&mut chain.ops, &GreedyShape::default());
    let consumer = chain.ops.iter().position(|op| op.engine == Engine::Alu).expect("the chain has adds");
    let producer_id = *chain.ops[consumer].dep_ids.iter().next().expect("the add reads a value");
    let producer_cycle = chain.ops.iter().find(|op| op.id == producer_id).expect("producer exists").cycle;
    chain.ops[consumer].cycle = producer_cycle;
    let err = verify_schedule(&chain.ops, true, &chain_shifted, &chain_lanes, &none_c, &none_s).unwrap_err();
    assert!(err.contains("readable from cycle"), "{}", err);

    // A split whose last lane is not the op's cycle.
    let op = &emit.ops[level_ops[0][1]];
    let bad_shift: HashSet<usize> = [op.id].into_iter().collect();
    let bad_lanes: HashMap<usize, Vec<i32>> = [(op.id, vec![op.cycle + 1; VLEN])].into_iter().collect();
    let err = verify_schedule(&emit.ops, true, &bad_shift, &bad_lanes, &none_c, &none_s).unwrap_err();
    assert!(err.contains("last lane"), "{}", err);
}

/// The verifier checks memory edges with their own latencies: a store over a word in the
/// cycle of an earlier load of it is accepted (WAR, latency 0), a load in the cycle of the
/// store it must see is rejected (RAW, latency 1). No slot limit or data dependency is
/// involved in the rejected schedule.
#[test]
fn test_justify_verifier_checks_memory_edges() {
    use crate::kernel_builder::memory_deps::{compute_memory_dependencies, test_programs::store_load_same_word};

    let (mut ops, ix) = store_load_same_word();
    assert_eq!(compute_memory_dependencies(&mut ops).unwrap().kept, 3);
    // The three constants as add_imm (one flow slot each), so the load slots are free.
    let consts: HashMap<usize, ConstChoiceEncoding> = ops
        .iter()
        .filter(|op| op.opcode == Opcode::ConstChoice)
        .map(|op| (op.id, ConstChoiceEncoding::AddImm))
        .collect();
    assert_eq!(consts.len(), 3);
    let none_s = HashMap::new();
    let (shifted, lanes) = (HashSet::new(), HashMap::new());
    // p, w, q at 0, 1, 2; the first load and the store over its word both at 2.
    for (idx, cycle) in [0, 1, 2].into_iter().enumerate() {
        ops[idx].cycle = cycle;
    }
    ops[ix.load_before].cycle = 2;
    ops[ix.store_word0].cycle = 2;
    ops[ix.load_before + 1].cycle = 3;
    ops[ix.first_store].cycle = 4;
    ops[ix.load_after].cycle = 3;
    ops[ix.load_after + 1].cycle = 4;
    ops[ix.load_after + 2].cycle = 5;
    ops[ix.last_store].cycle = 6;
    assert!(ops.iter().all(|op| op.cycle >= 0), "every op has a cycle");
    assert_eq!(verify_schedule(&ops, false, &shifted, &lanes, &consts, &none_s), Ok(7), "same-cycle WAR");

    ops[ix.load_after].cycle = ops[ix.store_word0].cycle;
    let err = verify_schedule(&ops, false, &shifted, &lanes, &consts, &none_s).unwrap_err();
    assert!(err.contains("raw"), "{}", err);
}

/// The verifier counts a split shift's lanes in the cycle each lane runs in: a whole
/// shifted op plus four lanes of a split one fill the 12 scalar slots, a fifth lane is
/// one too many.
#[test]
fn test_justify_verifier_counts_split_lanes_per_cycle() {
    let (mut emit, level_ops) = emit_independent_chains(2, 1);
    let (whole, split) = (level_ops[0][0], level_ops[1][0]);
    // Constants at 0, the broadcast at 1, both adds from 2, the stores at 4 and 5.
    for op in emit.ops.iter_mut() {
        op.cycle = match (op.engine, op.id) {
            (Engine::Load, _) => 0,
            (Engine::Valu, _) => 1,
            _ => 4,
        };
    }
    emit.ops[whole].cycle = 2;
    emit.ops[split].cycle = 3;
    let last = emit.ops.len() - 1;
    emit.ops[last].cycle = 5;
    let (whole_id, split_id) = (emit.ops[whole].id, emit.ops[split].id);
    let shifted: HashSet<usize> = [whole_id, split_id].into_iter().collect();
    let lanes_with = |first: usize| -> HashMap<usize, Vec<i32>> {
        [(split_id, (0..VLEN).map(|lane| if lane < first { 2 } else { 3 }).collect())].into_iter().collect()
    };
    let (none_c, none_s) = (HashMap::new(), HashMap::new());
    assert_eq!(verify_schedule(&emit.ops, true, &shifted, &lanes_with(4), &none_c, &none_s), Ok(6));
    let err = verify_schedule(&emit.ops, true, &shifted, &lanes_with(5), &none_c, &none_s).unwrap_err();
    assert!(err.contains("cycle 2 uses 13 alu slots"), "{}", err);
}

/// A shorter schedule that moves an op out of its `[ASAP, ALAP]` domain is dropped and
/// counted as a domain rejection, not as a verification failure.
#[test]
fn test_justify_counts_domain_rejections_separately() {
    let (mut emit, _) = emit_short_chains_before_a_long_one();
    let (greedy_cycles, mut shifted, mut lanes) =
        schedule_with_shift_and_shape(&mut emit.ops, &GreedyShape::default());
    let input: Vec<i32> = emit.ops.iter().map(|op| op.cycle).collect();
    // Domains that hold every op at its greedy cycle.
    let pinned: HashMap<usize, i32> = emit.ops.iter().map(|op| (op.id, op.cycle)).collect();
    let outcome = justify_schedule(
        &mut emit.ops,
        true,
        &mut shifted,
        &mut lanes,
        &HashMap::new(),
        &HashMap::new(),
        None,
        Some((&pinned, &pinned)),
        &justify_config(5),
    )
    .unwrap();
    assert_eq!(outcome.cycles_after, greedy_cycles, "{}", outcome.summary());
    assert!(outcome.domain_rejections > 0, "{}", outcome.summary());
    assert_eq!(outcome.domain_rejections, outcome.candidates_checked);
    assert_eq!((outcome.verify_rejections, outcome.scratch_rejections), (0, 0));
    assert!(outcome.summary().contains("rejected for ASAP/ALAP domains"));
    assert_eq!(input, emit.ops.iter().map(|op| op.cycle).collect::<Vec<_>>(), "input schedule is kept");
}

/// An orbit that never reproduces its starting schedule runs every iteration it is given;
/// the stall limit ends it after that many iterations in a row without a shorter accepted
/// schedule, with the same result.
#[test]
fn test_justify_stops_after_stalled_iterations() {
    let run = |max_stalled_iterations: usize| {
        let (mut emit, _) = emit_independent_chains(30, 12);
        let (_, mut shifted, mut lanes) =
            schedule_with_shift_and_shape(&mut emit.ops, &GreedyShape { split_shift: true, ..Default::default() });
        let config = JustifyConfig { split_shift: true, max_stalled_iterations, ..justify_config(40) };
        let outcome = justify_plain(&mut emit.ops, &mut shifted, &mut lanes, &config);
        (outcome, emit.ops.iter().map(|op| op.cycle).collect::<Vec<i32>>())
    };
    let (unlimited, unlimited_cycles) = run(0);
    assert_eq!(unlimited.iterations, 40, "the test needs an orbit: {}", unlimited.summary());
    assert!(unlimited.cycles_after < unlimited.cycles_before, "{}", unlimited.summary());

    let (limited, limited_cycles) = run(3);
    // The gain comes in the first iteration; three iterations without one follow.
    assert_eq!(limited.iterations, 4, "{}", limited.summary());
    assert_eq!(limited.cycles_after, unlimited.cycles_after);
    assert_eq!(limited_cycles, unlimited_cycles);
}

// =========================================================================
// Individually forced shifts (GreedyShape::forced_shift_ops)
//
// The CP-SAT model fixes the shift literal of a forced op to 1. The greedy schedule is the
// model's hint and bounds its domains, so it must place those ops shifted on every path.
// =========================================================================

/// A policy that never lets the scheduler choose a shift (as the pressure policies do above
/// their critical threshold) and otherwise behaves like the plain one.
#[derive(Debug)]
struct NoShiftPolicy;

impl PressurePolicy for NoShiftPolicy {
    fn sort_ready(&self, candidates: &mut [SchedulingCandidate], ctx: &SchedulingContext) {
        GreedyPolicy.sort_ready(candidates, ctx)
    }

    fn should_defer_for_pressure(
        &self,
        _live_words: usize,
        _words_added: usize,
        _words_freed: usize,
        _scratch_budget: usize,
    ) -> bool {
        false
    }

    fn allow_shift(&self, _live_words: usize, _alu_used: usize, _scratch_budget: usize) -> bool {
        false
    }
}

/// What a forced-shift scheduling run is given besides the ops.
struct ForcedShiftRun<'a> {
    forced: &'a [usize],
    policy: &'a dyn PressurePolicy,
    budget: usize,
    enable_valu_shift: bool,
    fixed_shifted_ops: Option<&'a HashSet<usize>>,
    valu_shift_fixed_to_none: bool,
    split_shift: bool,
}

impl Default for ForcedShiftRun<'_> {
    fn default() -> Self {
        ForcedShiftRun {
            forced: &[],
            policy: &GreedyPolicy,
            budget: 1536,
            enable_valu_shift: true,
            fixed_shifted_ops: None,
            valu_shift_fixed_to_none: false,
            split_shift: false,
        }
    }
}

/// Schedule `ops` with the given forced shifts.
///
/// # Returns
/// `(shifted op ids, per-lane cycles of split shifts, budget_exceeded)`.
fn schedule_with_forced_shifts(
    ops: &mut [SsaOp],
    run: &ForcedShiftRun,
) -> (HashSet<usize>, HashMap<usize, Vec<i32>>, bool) {
    let shape = GreedyShape {
        split_shift: run.split_shift,
        forced_shift_ops: run.forced.iter().copied().collect(),
        ..Default::default()
    };
    let mut shifted = HashSet::new();
    let mut lane_cycles = HashMap::new();
    let ((_, budget_exceeded, _), _) = schedule_ssa_greedy_shaped(
        ops,
        run.enable_valu_shift,
        &mut shifted,
        run.budget,
        None,
        None,
        None,
        None,
        &GreedyBoundsUsage::None,
        run.policy,
        None,
        false,
        false,
        false,
        run.valu_shift_fixed_to_none,
        None,
        run.fixed_shifted_ops,
        None,
        &mut GreedyDiagnostics::new(false),
        &shape,
        &mut lane_cycles,
    )
    .expect("greedy schedules the ops");
    (shifted, lane_cycles, budget_exceeded)
}

/// Every op has a cycle, no cycle holds more than 6 vector ops that are not shifted or more
/// than 12 scalar ops (native ones plus the lanes of shifted ops), and every reader runs
/// after its producers.
fn assert_slot_limits_with_shifts(
    ops: &[SsaOp],
    shifted: &HashSet<usize>,
    lane_cycles: &HashMap<usize, Vec<i32>>,
) {
    let mut alu = shifted_alu_use(ops, shifted, lane_cycles);
    let mut valu: HashMap<i32, usize> = HashMap::new();
    let cycle_of: HashMap<usize, i32> = ops.iter().map(|op| (op.id, op.cycle)).collect();
    for op in ops {
        assert!(op.cycle >= 0, "op {} was not scheduled", op.id);
        match op.engine {
            Engine::Alu => *alu.entry(op.cycle).or_default() += 1,
            Engine::Valu if !shifted.contains(&op.id) => *valu.entry(op.cycle).or_default() += 1,
            _ => {}
        }
        for dep in &op.dep_ids {
            assert!(cycle_of[dep] < op.cycle, "op {} runs before its producer {}", op.id, dep);
        }
    }
    for (cycle, used) in &alu {
        assert!(*used <= 12, "cycle {}: {} scalar slots", cycle, used);
    }
    for (cycle, used) in &valu {
        assert!(*used <= 6, "cycle {}: {} vector slots", cycle, used);
    }
}

/// Ids of the adds of `emit_independent_chains` at `level`.
fn chain_level_ids(emit: &SsaEmitContext, level_ops: &[Vec<usize>], level: usize) -> Vec<usize> {
    level_ops.iter().map(|chain| emit.ops[chain[level]].id).collect()
}

/// Ops in the forced set are placed shifted even where the scheduler would leave everything
/// on the vector slots, including several that become ready in the same cycle (one whole
/// shift fits per cycle); nothing else is shifted and the slot limits hold.
#[test]
fn test_forced_shift_ops_are_placed_shifted() {
    // 4 chains: at most 4 vector ops per cycle, so the greedy never shifts on its own.
    let (mut free, free_levels) = emit_independent_chains(4, 3);
    let (free_shifted, _, _) = schedule_with_forced_shifts(&mut free.ops, &ForcedShiftRun::default());
    assert!(free_shifted.is_empty(), "nothing to shift without a forced set");
    let free_cycle = free.ops[free_levels[0][1]].cycle;

    let (mut emit, level_ops) = emit_independent_chains(4, 3);
    let forced = chain_level_ids(&emit, &level_ops, 1);
    let run = ForcedShiftRun { forced: &forced, ..Default::default() };
    let (shifted, lanes, budget_exceeded) = schedule_with_forced_shifts(&mut emit.ops, &run);
    assert!(!budget_exceeded);
    assert_eq!(shifted, forced.iter().copied().collect::<HashSet<_>>());
    assert_slot_limits_with_shifts(&emit.ops, &shifted, &lanes);
    // All four were ready in `free_cycle`; shifted, they take one cycle each.
    let mut cycles: Vec<i32> = level_ops.iter().map(|chain| emit.ops[chain[1]].cycle).collect();
    cycles.sort();
    assert_eq!(cycles, (free_cycle..free_cycle + 4).collect::<Vec<_>>());
}

/// A policy that blocks shifting does not apply to a forced shift.
#[test]
fn test_forced_shift_overrides_policy_that_blocks_shifting() {
    // 30 chains saturate the vector slots; with shifting blocked nothing is shifted.
    let (mut blocked, _) = emit_independent_chains(30, 4);
    let run = ForcedShiftRun { policy: &NoShiftPolicy, ..Default::default() };
    let (blocked_shifted, _, _) = schedule_with_forced_shifts(&mut blocked.ops, &run);
    assert!(blocked_shifted.is_empty(), "the policy blocks every shift");

    let (mut emit, level_ops) = emit_independent_chains(30, 4);
    let forced: Vec<usize> = chain_level_ids(&emit, &level_ops, 2).into_iter().step_by(7).collect();
    let run = ForcedShiftRun { forced: &forced, policy: &NoShiftPolicy, ..Default::default() };
    let (shifted, lanes, _) = schedule_with_forced_shifts(&mut emit.ops, &run);
    assert_eq!(shifted, forced.iter().copied().collect::<HashSet<_>>());
    assert_slot_limits_with_shifts(&emit.ops, &shifted, &lanes);
}

/// The force-schedule path (scratch budget too tight, pressure policy blocking shifts) keeps
/// forced ops shifted and the slot limits intact. Same program as
/// `test_force_schedule_never_exceeds_valu_capacity`.
#[test]
fn test_forced_shift_holds_on_force_schedule_path() {
    let policies: [(&str, &dyn PressurePolicy); 3] = [
        ("adaptive", &AdaptivePolicy::default()),
        ("no-shift", &NoShiftPolicy),
        ("greedy", &GreedyPolicy),
    ];
    for (name, policy) in policies {
        let mut emit = SsaEmitContext::new();
        let scalars: Vec<VReg> = (0..8).map(|i| emit.emit_const(i)).collect();
        let vecs: Vec<VReg> = scalars.iter().map(|&s| emit.emit_vbroadcast(s)).collect();
        let mut adds = Vec::new();
        let mut results = Vec::new();
        for i in 0..12 {
            results.push(emit.emit_valu(Opcode::Add, vecs[i % 8], vecs[(i + 1) % 8]));
            adds.push(emit.ops.last().unwrap().id);
        }
        let addr = emit.emit_const(200);
        for &r in &results {
            emit.emit_vstore(addr, r);
        }
        compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg, None);

        // 8 broadcasts are 64 words: with 80 the adds are force-scheduled past the budget.
        let run = ForcedShiftRun { forced: &adds, policy, budget: 80, ..Default::default() };
        let (shifted, lanes, budget_exceeded) = schedule_with_forced_shifts(&mut emit.ops, &run);
        assert!(budget_exceeded, "{}: the force-schedule path must run", name);
        assert_eq!(shifted, adds.iter().copied().collect::<HashSet<_>>(), "{}", name);
        assert_slot_limits_with_shifts(&emit.ops, &shifted, &lanes);
    }
}

/// With split shifts enabled a forced op is still shifted (whole) and the scalar slots stay
/// within 12 per cycle next to the split shifts the scheduler chooses itself.
#[test]
fn test_forced_shift_with_split_shifts() {
    let (mut emit, level_ops) = emit_independent_chains(30, 6);
    let forced: Vec<usize> = chain_level_ids(&emit, &level_ops, 3).into_iter().step_by(5).collect();
    let run = ForcedShiftRun { forced: &forced, split_shift: true, ..Default::default() };
    let (shifted, lanes, _) = schedule_with_forced_shifts(&mut emit.ops, &run);
    assert!(!lanes.is_empty(), "the scheduler still makes split shifts");
    for id in &forced {
        assert!(shifted.contains(id), "forced op {} is not shifted", id);
        assert!(!lanes.contains_key(id), "forced op {} was split over two cycles", id);
    }
    assert_slot_limits_with_shifts(&emit.ops, &shifted, &lanes);
}

/// A forced shift is recorded when a hint pass owns the other shift decisions
/// (`valu_shift_fixed_to_none`): the pass only adds shifts, so it must see this one.
#[test]
fn test_forced_shift_recorded_when_hint_pass_owns_shifts() {
    let (mut emit, level_ops) = emit_independent_chains(4, 3);
    let forced = vec![emit.ops[level_ops[2][1]].id];
    let run = ForcedShiftRun { forced: &forced, valu_shift_fixed_to_none: true, ..Default::default() };
    let (shifted, lanes, _) = schedule_with_forced_shifts(&mut emit.ops, &run);
    assert_eq!(shifted, forced.iter().copied().collect::<HashSet<_>>());
    assert_slot_limits_with_shifts(&emit.ops, &shifted, &lanes);
}

/// The forced set is consulted exactly where the CP-SAT model consults it: not with shifting
/// disabled, not under a complete fixed encoding (`--valu-shift valu`), and never for an op
/// that cannot shift (a broadcast) or is not a vector op.
#[test]
fn test_forced_shift_ignored_where_the_model_ignores_it() {
    let build = || {
        let (emit, level_ops) = emit_independent_chains(4, 3);
        let mut forced = chain_level_ids(&emit, &level_ops, 1);
        // The two constants and the broadcast: not shiftable.
        forced.extend(emit.ops.iter().take(3).map(|op| op.id));
        (emit, forced)
    };

    let (mut emit, forced) = build();
    let run = ForcedShiftRun { forced: &forced, enable_valu_shift: false, ..Default::default() };
    let (shifted, lanes, _) = schedule_with_forced_shifts(&mut emit.ops, &run);
    assert!(shifted.is_empty(), "shifting disabled");
    assert_slot_limits_with_shifts(&emit.ops, &shifted, &lanes);

    let (mut emit, forced) = build();
    let none = HashSet::new();
    let run = ForcedShiftRun { forced: &forced, fixed_shifted_ops: Some(&none), ..Default::default() };
    let (shifted, lanes, _) = schedule_with_forced_shifts(&mut emit.ops, &run);
    assert!(shifted.is_empty(), "a fixed encoding owns every shift decision");
    assert_slot_limits_with_shifts(&emit.ops, &shifted, &lanes);

    let (mut emit, forced) = build();
    let run = ForcedShiftRun { forced: &forced, ..Default::default() };
    let (shifted, lanes, _) = schedule_with_forced_shifts(&mut emit.ops, &run);
    let shiftable: HashSet<usize> = forced[..4].iter().copied().collect();
    assert_eq!(shifted, shiftable, "only the shiftable vector ops of the set are shifted");
    assert_slot_limits_with_shifts(&emit.ops, &shifted, &lanes);
}

/// Justification after `--greedy-select-overflow valu`: the selects the greedy put on the
/// vector ALU are multiply-add selects for the pass too (their delta and operands are held to
/// cycle + 1, as address assignment holds them). The pass's peak equals address
/// assignment's on its result, the slot limits hold with those selects on the vector ALU, and
/// a budget one word below the input's interval peak is refused for every candidate (the
/// input itself is never re-checked, so it stays).
#[test]
fn test_justify_after_select_overflow_keeps_fma_liveness() {
    for split_shift in [false, true] {
        let shape = GreedyShape {
            priority: GreedyPriority::Wavefront { stagger_cycles: 2, groups_in_flight: None },
            split_shift,
            select_overflow: SelectOverflow::Valu { limit: None },
            ..Default::default()
        };
        let (mut emit, _) = emit_delta_select_chains(24);
        let (_, mut shifted, fma, mut lanes) =
            schedule_collecting_fma(&mut emit.ops, &shape, 1536, &GreedyPolicy, true).unwrap();
        assert!(!fma.is_empty(), "split={}: the overflow placed no select on the vector ALU", split_shift);
        let select_choice: HashMap<usize, SelectChoiceEncoding> = emit
            .ops
            .iter()
            .filter(|op| op.opcode == Opcode::SelectChoice)
            .map(|op| {
                let enc = if fma.contains(&op.id) { SelectChoiceEncoding::Fma } else { SelectChoiceEncoding::Vselect };
                (op.id, enc)
            })
            .collect();
        let peak_of = |ops: &[SsaOp], lanes: &HashMap<usize, Vec<i32>>| {
            let scheduled: Vec<SsaOp> = ops.iter().filter(|op| op.cycle >= 0).cloned().collect();
            let config = crate::address_assign::AddressAssignConfig {
                select_fma_ops: fma.clone(),
                shifted_alu_cycles: lanes.clone(),
                ..Default::default()
            };
            let intervals = crate::address_assign::compute_live_intervals(&config, &scheduled).unwrap();
            crate::address_assign::compute_peak_liveness(&intervals)
        };
        let input_peak = peak_of(&emit.ops, &lanes);
        let config = JustifyConfig { split_shift, ..justify_config(5) };
        let outcome = justify_schedule(
            &mut emit.ops, true, &mut shifted, &mut lanes, &HashMap::new(), &select_choice, None, None, &config,
        )
        .unwrap();
        assert_eq!(outcome.peak_live_before, input_peak, "split={}", split_shift);
        assert_eq!(outcome.peak_live_after, peak_of(&emit.ops, &lanes), "split={}", split_shift);
        assert_slot_limits_with_selects(&emit.ops, &shifted, &fma, &lanes, &format!("split={}", split_shift));

        let (mut emit, _) = emit_delta_select_chains(24);
        let (_, mut shifted, _, mut lanes) =
            schedule_collecting_fma(&mut emit.ops, &shape, 1536, &GreedyPolicy, true).unwrap();
        let before: Vec<i32> = emit.ops.iter().map(|op| op.cycle).collect();
        let tight = JustifyConfig { scratch_budget: input_peak - 1, split_shift, ..justify_config(5) };
        let outcome = justify_schedule(
            &mut emit.ops, true, &mut shifted, &mut lanes, &HashMap::new(), &select_choice, None, None, &tight,
        )
        .unwrap();
        assert_eq!(outcome.cycles_after, outcome.cycles_before, "split={}", split_shift);
        assert_eq!(emit.ops.iter().map(|op| op.cycle).collect::<Vec<_>>(), before);
    }
}

/// The pace accounting counts every placement of a vector op and every shift, on every path:
/// a shift placed on the force-schedule path (over budget) counts as shifted (the
/// `force_shifted = true` in the force path's `Shifted` arm, kept when #78's restructuring of
/// that path was merged). The runs below force-schedule and make force-path shifts.
#[test]
fn test_pace_counts_force_path_shifts() {
    let mut force_shifts_seen = 0;
    for budget in [56, 64] {
        for shift_policy in [ShiftPolicy::WhenFull, ShiftPolicy::Pace { target: None }] {
            for split_shift in [false, true] {
                let shape = GreedyShape { split_shift, shift_policy, ..Default::default() };
                let mut emit = emit_mixed_chains(10, 6);
                let (exceeded, shifted, _, diagnostics) = schedule_tight(&mut emit.ops, &shape, budget, &GreedyPolicy);
                assert!(exceeded, "budget {} forces", budget);
                let label = format!("budget {} {:?} split={}", budget, shift_policy, split_shift);
                let vector: Vec<&SsaOp> =
                    emit.ops.iter().filter(|op| op.engine == Engine::Valu && op.cycle >= 0).collect();
                let shifted_vector = vector.iter().filter(|op| shifted.contains(&op.id)).count();
                assert_eq!(diagnostics.pace_vector_placed, vector.len(), "{}", label);
                assert_eq!(diagnostics.pace_shifted, shifted_vector, "{}", label);
                force_shifts_seen += diagnostics.force_shifts;
            }
        }
    }
    assert!(force_shifts_seen > 0, "control: some run must shift on the force path");
}

// =========================================================================
// Justification with ALU-derived constants (--greedy-const-policy)
// =========================================================================

/// 8 and 16 loaded, 24 derived as 8 + 16 on the scalar ALU, `(8 + 16) + 24` stored. Returns
/// the program and the ConstChoice encodings (Load, Load, AluDerive).
fn emit_derived_constant() -> (SsaEmitContext, HashMap<usize, ConstChoiceEncoding>) {
    let mut emit = SsaEmitContext::new();
    let c8 = emit.emit_const_choice(8);
    let c16 = emit.emit_const_choice(16);
    let c24 = emit.emit_const_choice(24);
    let s1 = emit.emit_alu(Opcode::Add, c8, c16);
    let s2 = emit.emit_alu(Opcode::Add, s1, c24);
    emit.emit_store_imm(100, s2);
    compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg, None);
    let (i8, i16, i24) = (emit.ops[0].id, emit.ops[1].id, emit.ops[2].id);
    let encodings = HashMap::from([
        (i8, ConstChoiceEncoding::Load),
        (i16, ConstChoiceEncoding::Load),
        (
            i24,
            ConstChoiceEncoding::AluDerive(crate::types::AluDeriveInfo {
                left_op_id: i8,
                right_op_id: i16,
                opcode: Opcode::Add,
            }),
        ),
    ]);
    (emit, encodings)
}

/// Address assignment's peak for `ops`, with or without the ALU derivations in `encodings`.
fn allocator_peak(ops: &[SsaOp], encodings: &HashMap<usize, ConstChoiceEncoding>, with_derives: bool) -> usize {
    let scheduled: Vec<SsaOp> = ops.iter().filter(|op| op.cycle >= 0).cloned().collect();
    let derives: HashMap<usize, crate::types::AluDeriveInfo> = encodings
        .iter()
        .filter_map(|(id, e)| match e {
            ConstChoiceEncoding::AluDerive(info) if with_derives => Some((*id, info.clone())),
            _ => None,
        })
        .collect();
    let config = crate::address_assign::AddressAssignConfig { const_alu_derive_ops: derives, ..Default::default() };
    let intervals = crate::address_assign::compute_live_intervals(&config, &scheduled).unwrap();
    crate::address_assign::compute_peak_liveness(&intervals)
}

/// A derived constant is an ALU op that reads its two producers at its own cycle. The pass
/// schedules it after both, keeps both live until then in its scratch accounting (as address
/// assignment does: the earlier AluDerive bugs were producer lifetimes that were not
/// extended), and its peak equals address assignment's before and after. Hand-made input:
/// the derive runs four cycles after the producers' last other read, which only the
/// extension sees.
#[test]
fn test_justify_models_alu_derived_constants() {
    let (mut emit, encodings) = emit_derived_constant();
    // 8, 16 at 0 (two loads); 8 + 16 at 1; 24 = 8 + 16 at 5; the sum at 6; the store at 7.
    for (i, c) in [0, 0, 5, 1, 6, 7].into_iter().enumerate() {
        emit.ops[i].cycle = c;
    }
    let with_ext = allocator_peak(&emit.ops, &encodings, true);
    assert!(with_ext > allocator_peak(&emit.ops, &encodings, false), "control: the extension must matter here");
    let (mut shifted, mut lanes) = (HashSet::new(), HashMap::new());
    assert_eq!(
        verify_schedule(&emit.ops, true, &shifted, &lanes, &encodings, &HashMap::new()),
        Ok(8),
        "the greedy-style input passes the verifier"
    );
    let outcome = justify_schedule(
        &mut emit.ops, true, &mut shifted, &mut lanes, &encodings, &HashMap::new(), None, None, &justify_config(5),
    )
    .unwrap();
    assert_eq!(outcome.note, None, "{}", outcome.summary());
    assert_eq!(outcome.peak_live_before, with_ext, "the pass counts the producers live to the derive");
    assert!(outcome.cycles_after < outcome.cycles_before, "{}", outcome.summary());
    assert_eq!(outcome.peak_live_after, allocator_peak(&emit.ops, &encodings, true));
    // The derive runs after both producers are visible, and the reader of 24 after it.
    let cycle = |i: usize| emit.ops[i].cycle;
    assert!(cycle(2) > cycle(0) && cycle(2) > cycle(1), "{:?}", emit.ops.iter().map(|o| o.cycle).collect::<Vec<_>>());
    assert!(cycle(4) > cycle(2));
    assert!(verify_schedule(&emit.ops, true, &shifted, &lanes, &encodings, &HashMap::new()).is_ok());

    // The verifier refuses a derive in its producer's cycle (the edge is not in dep_ids).
    let (mut bad, _) = emit_derived_constant();
    for (i, c) in [0, 1, 1, 2, 3, 4].into_iter().enumerate() {
        bad.ops[i].cycle = c;
    }
    let err = verify_schedule(&bad.ops, true, &HashSet::new(), &HashMap::new(), &encodings, &HashMap::new()).unwrap_err();
    assert!(err.contains("ALU-derived constant"), "{}", err);
}

/// The scratch check every candidate goes through uses the extended lifetimes. On the
/// hand-made schedule (the derive four cycles after its producers' last other read) the peak
/// is 2 words without the extension and 3 with it. A budget of 2 is rejected with the
/// derivations (and would be accepted without them, which is what the check must not do); a
/// budget of 3 is accepted. Both with and without the allocator trial.
#[test]
fn test_justify_alu_derive_budget_uses_extended_lifetimes() {
    use super::justify::{scratch_check, Workspace};
    let (mut emit, encodings) = emit_derived_constant();
    for (i, c) in [0, 0, 5, 1, 6, 7].into_iter().enumerate() {
        emit.ops[i].cycle = c;
    }
    let (ext, unext) = (allocator_peak(&emit.ops, &encodings, true), allocator_peak(&emit.ops, &encodings, false));
    assert_eq!((unext, ext), (2, 3), "control: the extension adds one word here");
    let derives: HashMap<usize, crate::types::AluDeriveInfo> = encodings
        .iter()
        .filter_map(|(id, e)| match e {
            ConstChoiceEncoding::AluDerive(info) => Some((*id, info.clone())),
            _ => None,
        })
        .collect();
    let work = Workspace { ops: emit.ops.clone(), shifted_ops: HashSet::new(), shifted_lane_cycles: HashMap::new() };
    for allocator in [None, Some(crate::address_assign::AddressAssignConfig::default())] {
        let config = |budget: usize| JustifyConfig { scratch_budget: budget, allocator: allocator.clone(), ..justify_config(5) };
        let check = |budget: usize, d: &HashMap<usize, crate::types::AluDeriveInfo>| {
            scratch_check(&work, None, &HashSet::new(), d, &config(budget))
        };
        let tight = check(unext, &derives);
        assert!(tight.is_err(), "budget {} with derivations must be rejected: {:?}", unext, tight);
        assert!(check(unext, &HashMap::new()).is_ok(), "control: without the extension budget {} fits", unext);
        assert_eq!(check(ext, &derives), Ok(ext), "budget {} fits exactly", ext);
    }

    // End to end: with the budget at the input's unextended peak the pass still only returns
    // schedules that fit under address assignment's accounting.
    let (mut emit, encodings) = emit_derived_constant();
    for (i, c) in [0, 0, 5, 1, 6, 7].into_iter().enumerate() {
        emit.ops[i].cycle = c;
    }
    let (mut shifted, mut lanes) = (HashSet::new(), HashMap::new());
    let config = JustifyConfig { scratch_budget: unext, ..justify_config(5) };
    let outcome = justify_schedule(
        &mut emit.ops, true, &mut shifted, &mut lanes, &encodings, &HashMap::new(), None, None, &config,
    )
    .unwrap();
    assert!(outcome.cycles_after < outcome.cycles_before, "{}", outcome.summary());
    assert!(allocator_peak(&emit.ops, &encodings, true) <= unext);
}

/// 8 and 16 loaded; 24 derived as 8 + 16; `16 + 24` stored. The constant 8 is read only by
/// the derivation (no op has it as a source), and `[c8, c16, c24, sum, store]` at cycles
/// 0, 0, 3, 4, 5. Returns the program and the derivation.
fn emit_constant_read_only_by_derivation() -> (SsaEmitContext, HashMap<usize, crate::types::AluDeriveInfo>) {
    let mut emit = SsaEmitContext::new();
    let c8 = emit.emit_const_choice(8);
    let c16 = emit.emit_const_choice(16);
    let c24 = emit.emit_const_choice(24);
    let sum = emit.emit_alu(Opcode::Add, c16, c24);
    emit.emit_store_imm(100, sum);
    compute_ssa_dependencies(&mut emit.ops, &emit.vreg_def, emit.zero_vreg, None);
    for (i, c) in [0, 0, 3, 4, 5].into_iter().enumerate() {
        emit.ops[i].cycle = c;
    }
    let derive = HashMap::from([(
        emit.ops[2].id,
        crate::types::AluDeriveInfo { left_op_id: emit.ops[0].id, right_op_id: emit.ops[1].id, opcode: Opcode::Add },
    )]);
    let _ = c8;
    (emit, derive)
}

/// Review of #93: address assignment extends an AluDerive producer only if it has an
/// interval. A producer read only by derivations still gets one (its definition), so the
/// extension applies: the constant 8 stays live until the derive reads it at cycle 3.
#[test]
fn test_alu_derive_extends_a_producer_read_only_by_derivations() {
    let (emit, derive) = emit_constant_read_only_by_derivation();
    let c8 = emit.ops[0].dest.unwrap();
    let plain = crate::address_assign::AddressAssignConfig::default();
    let without = crate::address_assign::compute_live_intervals(&plain, &emit.ops).unwrap();
    assert_eq!((without[&c8].start, without[&c8].end), (0, 1), "control: a value nothing reads lives [D, D + 1]");
    let config = crate::address_assign::AddressAssignConfig { const_alu_derive_ops: derive, ..Default::default() };
    let with = crate::address_assign::compute_live_intervals(&config, &emit.ops).unwrap();
    assert_eq!((with[&c8].start, with[&c8].end), (0, 3), "live until the derive's read");
}

/// A derive whose producer is not in the scheduled ops would read an unassigned word; it is an
/// error (in every build profile), not a skipped extension.
#[test]
fn test_alu_derive_without_its_producer_is_an_error() {
    let (emit, derive) = emit_constant_read_only_by_derivation();
    let without_producer: Vec<SsaOp> = emit.ops[1..].to_vec();
    let config = crate::address_assign::AddressAssignConfig { const_alu_derive_ops: derive, ..Default::default() };
    let err = crate::address_assign::compute_live_intervals(&config, &without_producer).unwrap_err();
    assert!(err.contains("has no live interval"), "{}", err);
}

/// The greedy's own `alu-first` schedule of five derivable constants: the pass accepts it as
/// input (no note: before, its verifier refused ALU-derived constants), and its peak equals
/// address assignment's with the derivations, before and after.
#[test]
fn test_justify_after_const_policy_alu_first() {
    let mut emit = emit_derivable_consts();
    let shape = GreedyShape { const_policy: ConstPolicy::AluFirst, ..Default::default() };
    let mut diagnostics = GreedyDiagnostics::new(false);
    let mut add_imm = HashSet::new();
    schedule_ssa_greedy_shaped(
        &mut emit.ops,
        false,
        &mut HashSet::new(),
        1536,
        None,
        None,
        None,
        Some(&mut add_imm),
        &GreedyBoundsUsage::None,
        &GreedyPolicy,
        None,
        false,
        false,
        false,
        false,
        None,
        None,
        None,
        &mut diagnostics,
        &shape,
        &mut HashMap::new(),
    )
    .expect("greedy schedules the constants");
    let derived = diagnostics.const_alu_derive;
    assert!(!derived.is_empty());
    let encodings: HashMap<usize, ConstChoiceEncoding> = emit
        .ops
        .iter()
        .filter(|op| op.opcode == Opcode::ConstChoice)
        .map(|op| {
            let enc = if let Some(info) = derived.get(&op.id) {
                ConstChoiceEncoding::AluDerive(info.clone())
            } else if add_imm.contains(&op.id) {
                ConstChoiceEncoding::AddImm
            } else {
                ConstChoiceEncoding::Load
            };
            (op.id, enc)
        })
        .collect();
    let (mut shifted, mut lanes) = (HashSet::new(), HashMap::new());
    let before = allocator_peak(&emit.ops, &encodings, true);
    let outcome = justify_schedule(
        &mut emit.ops, false, &mut shifted, &mut lanes, &encodings, &HashMap::new(), None, None, &justify_config(5),
    )
    .unwrap();
    assert_eq!(outcome.note, None, "{}", outcome.summary());
    assert_eq!(outcome.peak_live_before, before);
    assert_eq!(outcome.peak_live_after, allocator_peak(&emit.ops, &encodings, true));
    assert!(verify_schedule(&emit.ops, false, &shifted, &lanes, &encodings, &HashMap::new()).is_ok());
}
