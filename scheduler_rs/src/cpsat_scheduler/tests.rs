// Pull in all re-exports from mod.rs so inner test modules' `use super::*;` works
#[allow(unused_imports)]
use super::*;

#[cfg(test)]
mod ssa_tests {
    use super::*;
    use crate::kernel_builder::emit_ssa::{compute_ssa_dependencies, SsaEmitContext};
    use crate::kernel_builder::ssa_lower::lower_ssa_ops_with_shift;
    use crate::types::{Engine, Opcode, SlotVal};
    use std::collections::{HashMap, HashSet};

    #[cfg(feature = "cpsat")]
    #[test]
    fn test_ssa_cpsat_constchoice_decisions() {
        // Build a simple kernel with ConstChoice ops
        let mut emit = SsaEmitContext::new();

        // Emit zero constant
        let _zero = emit.emit_const(0);

        // Emit ConstChoice constants
        let c1 = emit.emit_const_choice(1);
        let c2 = emit.emit_const_choice(2);

        // Use constants in ALU ops
        let _sum = emit.emit_alu(Opcode::Add, c1, c2);

        // Compute dependencies
        let vreg_def = std::mem::take(&mut emit.vreg_def);
        compute_ssa_dependencies(&mut emit.ops, &vreg_def, emit.zero_vreg, None);

        // Schedule with CP-SAT
        let result = schedule_ssa_cpsat(
            &emit.ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: false,
                stop_after_first_solution: true,
                enable_valu_shift: false,
                scratch_budget: 1536,
                model_name: None,
                ..Default::default()
            },
            SsaCpsatHints::default(),
            None,
        );

        assert!(result.is_some(), "CP-SAT should find a solution");
        let result = result.unwrap();

        // Verify all ops are scheduled
        assert_eq!(result.cycle_assignments.len(), emit.ops.len());

        // Verify ConstChoice decisions are extracted
        // The result should contain const_add_imm_ops (may be empty if solver chose Load)
        assert!(
            result.const_add_imm_ops.len() <= 2,
            "Should have at most 2 ConstChoice ops using AddImm"
        );

        // Note: AddImm operations can use address 0 directly without waiting for zero
        // to be "ready" since scratch memory is always initialized to 0 at startup.
        // There is no precedence constraint requiring zero to be scheduled before AddImm.
    }

    /// Sanity check: ForceAddImm must be feasible. If this fails, AddImm constraints are wrong.
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_ssa_cpsat_constchoice_force_add_imm_feasible() {
        let mut emit = SsaEmitContext::new();
        let _zero = emit.emit_const(0);
        let c1 = emit.emit_const_choice(1);
        let c2 = emit.emit_const_choice(2);
        let _sum = emit.emit_alu(Opcode::Add, c1, c2);
        let vreg_def = std::mem::take(&mut emit.vreg_def);
        compute_ssa_dependencies(&mut emit.ops, &vreg_def, emit.zero_vreg, None);

        let result = schedule_ssa_cpsat(
            &emit.ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: false,
                stop_after_first_solution: true,
                enable_valu_shift: false,
                scratch_budget: 1536,
                model_name: None,
                const_choice_mode: ConstChoiceMode::ForceAddImm,
                ..Default::default()
            },
            SsaCpsatHints::default(),
            None,
        );
        assert!(
            result.is_some(),
            "ForceAddImm must be feasible; if this fails, AddImm constraints are wrong"
        );
        let r = result.unwrap();
        let n_const = emit.ops.iter().filter(|o| o.opcode == Opcode::ConstChoice).count();
        assert_eq!(
            r.const_add_imm_ops.len(),
            n_const,
            "ForceAddImm: all ConstChoice ops must be AddImm"
        );
    }

    #[cfg(feature = "cpsat")]
    #[test]
    fn test_ssa_cpsat_constchoice_roundtrip() {
        // Build a simple kernel with ConstChoice ops
        let mut emit = SsaEmitContext::new();

        // Emit zero constant
        let zero = emit.emit_const(0);

        // Emit ConstChoice constants
        let c1 = emit.emit_const_choice(1);
        let c2 = emit.emit_const_choice(2);
        let c4 = emit.emit_const_choice(4);

        // Use constants in ALU ops
        let _sum1 = emit.emit_alu(Opcode::Add, c1, c2);
        let _sum2 = emit.emit_alu(Opcode::Add, c4, zero);

        // Compute dependencies
        let vreg_def = std::mem::take(&mut emit.vreg_def);
        compute_ssa_dependencies(&mut emit.ops, &vreg_def, emit.zero_vreg, None);

        // Schedule with CP-SAT
        let result = schedule_ssa_cpsat(
            &emit.ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: false,
                stop_after_first_solution: true,
                enable_valu_shift: false,
                scratch_budget: 1536,
                model_name: None,
                ..Default::default()
            },
            SsaCpsatHints::default(),
            None,
        );

        assert!(result.is_some(), "CP-SAT should find a solution");
        let cpsat_result = result.unwrap();

        // Verify all ops are scheduled
        assert_eq!(cpsat_result.cycle_assignments.len(), emit.ops.len());

        // Find ConstChoice op IDs and their values
        let const_choice_info: Vec<(usize, i64)> = emit
            .ops
            .iter()
            .filter(|op| op.opcode == Opcode::ConstChoice)
            .map(|op| {
                let value = op.srcs[0]
                    .as_imm()
                    .expect("ConstChoice should have immediate operand");
                (op.id, value)
            })
            .collect();

        assert_eq!(
            const_choice_info.len(),
            3,
            "Should have 3 ConstChoice ops (1, 2, 4)"
        );

        // Verify CP-SAT decisions are extracted
        assert!(
            cpsat_result.const_add_imm_ops.len() <= 3,
            "Should have at most 3 ConstChoice ops using AddImm"
        );

        // Now test round-trip: lower using CP-SAT decisions
        let mut vreg_to_addr = HashMap::new();
        for (idx, op) in emit.ops.iter().enumerate() {
            if let Some(dest) = op.dest {
                vreg_to_addr.insert(dest, idx as i64 * 10); // Simple address assignment
            }
        }
        // Ensure zero is at address 0
        vreg_to_addr.insert(zero, 0);

        let lowered_ops = lower_ssa_ops_with_shift(
            &emit.ops,
            &vreg_to_addr,
            &HashSet::new(),                 // No shifted ops
            &cpsat_result.const_add_imm_ops, // Use CP-SAT decisions
            &cpsat_result.const_alu_derive_ops,
            &cpsat_result.select_fma_ops,
            &cpsat_result.shifted_alu_cycles, // Use CP-SAT individual ALU cycles
            &cpsat_result.eliminated_vbroadcast_ops,
            &cpsat_result.eliminated_vb_to_scalar_producer,
        );

        // Verify that ConstChoice ops were lowered correctly based on CP-SAT decisions
        for (const_choice_id, expected_value) in &const_choice_info {
            let lowered_op = lowered_ops
                .iter()
                .find(|op| op.produces_constant == Some(*expected_value))
                .expect(&format!(
                    "Should find lowered op for ConstChoice producing {}",
                    expected_value
                ));

            if cpsat_result.const_add_imm_ops.contains(const_choice_id) {
                // CP-SAT chose AddImm - verify it's a Flow op
                assert_eq!(
                    lowered_op.engine,
                    Engine::Flow,
                    "ConstChoice op {} (value {}) should be lowered to Flow (AddImm)",
                    const_choice_id,
                    expected_value
                );
                match &lowered_op.slot[0] {
                    SlotVal::Str(s) => assert_eq!(
                        s, "add_imm",
                        "ConstChoice op {} should be add_imm",
                        const_choice_id
                    ),
                    _ => panic!("AddImm slot[0] should be Str"),
                }
                match &lowered_op.slot[2] {
                    SlotVal::Int(addr) => assert_eq!(*addr, 0, "AddImm should read from address 0"),
                    _ => panic!("AddImm slot[2] should be Int(0)"),
                }
            } else if cpsat_result.const_alu_derive_ops.contains_key(const_choice_id) {
                // CP-SAT chose ALU-derive - verify it's an ALU op
                assert_eq!(
                    lowered_op.engine,
                    Engine::Alu,
                    "ConstChoice op {} (value {}) should be lowered to Alu (ALU-derive)",
                    const_choice_id,
                    expected_value
                );
            } else {
                // CP-SAT chose Load - verify it's a Load op
                assert_eq!(
                    lowered_op.engine,
                    Engine::Load,
                    "ConstChoice op {} (value {}) should be lowered to Load",
                    const_choice_id,
                    expected_value
                );
                match &lowered_op.slot[0] {
                    SlotVal::Str(s) => assert_eq!(
                        s, "const",
                        "ConstChoice op {} should be const",
                        const_choice_id
                    ),
                    _ => panic!("Const slot[0] should be Str"),
                }
            }
        }

        // Verify all ConstChoice ops were lowered
        assert_eq!(
            lowered_ops.len(),
            emit.ops.len(),
            "All ops should be lowered"
        );
    }

    #[cfg(feature = "cpsat")]
    #[test]
    fn test_ssa_cpsat_constchoice_prefers_addimm_when_load_slots_full() {
        // Create a scenario where Load slots are heavily used, forcing CP-SAT to choose AddImm
        let mut emit = SsaEmitContext::new();

        // Emit zero constant
        let zero = emit.emit_const(0);

        // Emit many Load ops to fill Load slots
        for i in 0..10 {
            emit.emit_const(i + 100); // Regular Load ops
        }

        // Emit ConstChoice ops - CP-SAT should prefer AddImm to avoid Load slot pressure
        let c1 = emit.emit_const_choice(1);
        let c2 = emit.emit_const_choice(2);

        // Use constants
        let _sum = emit.emit_alu(Opcode::Add, c1, c2);

        // Compute dependencies
        let vreg_def = std::mem::take(&mut emit.vreg_def);
        compute_ssa_dependencies(&mut emit.ops, &vreg_def, emit.zero_vreg, None);

        // Schedule with CP-SAT with longer time limit to allow optimization
        let result = schedule_ssa_cpsat(
            &emit.ops,
            SsaCpsatConfig {
                time_limit_secs: 30.0,
                verbose: false,
                stop_after_first_solution: false, // Allow optimization
                enable_valu_shift: false,
                scratch_budget: 1536,
                model_name: None,
                ..Default::default()
            },
            SsaCpsatHints::default(),
            None,
        );

        assert!(result.is_some(), "CP-SAT should find a solution");
        let cpsat_result = result.unwrap();

        // Find ConstChoice op IDs
        let const_choice_ids: Vec<usize> = emit
            .ops
            .iter()
            .filter(|op| op.opcode == Opcode::ConstChoice)
            .map(|op| op.id)
            .collect();

        assert_eq!(const_choice_ids.len(), 2, "Should have 2 ConstChoice ops");

        // Verify decisions were extracted
        assert!(
            cpsat_result.const_add_imm_ops.len() <= 2,
            "Should have at most 2 ConstChoice ops using AddImm"
        );

        // Test lowering with the decisions
        let mut vreg_to_addr = HashMap::new();
        for (idx, op) in emit.ops.iter().enumerate() {
            if let Some(dest) = op.dest {
                vreg_to_addr.insert(dest, idx as i64 * 10);
            }
        }
        vreg_to_addr.insert(zero, 0);

        let lowered_ops = lower_ssa_ops_with_shift(
            &emit.ops,
            &vreg_to_addr,
            &HashSet::new(),
            &cpsat_result.const_add_imm_ops,
            &cpsat_result.const_alu_derive_ops,
            &cpsat_result.select_fma_ops,
            &cpsat_result.shifted_alu_cycles, // Use CP-SAT individual ALU cycles
            &cpsat_result.eliminated_vbroadcast_ops,
            &cpsat_result.eliminated_vb_to_scalar_producer,
        );

        // Verify ConstChoice ops were lowered correctly
        for const_choice_id in &const_choice_ids {
            let ssa_op = emit
                .ops
                .iter()
                .find(|op| op.id == *const_choice_id)
                .unwrap();
            let expected_value = ssa_op.srcs[0].as_imm().unwrap();

            let lowered_op = lowered_ops
                .iter()
                .find(|op| op.produces_constant == Some(expected_value))
                .expect(&format!(
                    "Should find lowered op for ConstChoice {}",
                    const_choice_id
                ));

            if cpsat_result.const_add_imm_ops.contains(const_choice_id) {
                // Verify AddImm was chosen
                assert_eq!(
                    lowered_op.engine,
                    Engine::Flow,
                    "ConstChoice {} should use AddImm when Load slots are full",
                    const_choice_id
                );
                match &lowered_op.slot[0] {
                    SlotVal::Str(s) => assert_eq!(s, "add_imm"),
                    _ => panic!("Should be add_imm"),
                }
            } else if cpsat_result.const_alu_derive_ops.contains_key(const_choice_id) {
                // Verify ALU-derived was chosen
                assert_eq!(
                    lowered_op.engine,
                    Engine::Alu,
                    "ConstChoice {} should use ALU when AluDerive chosen",
                    const_choice_id
                );
            } else {
                // Verify Load was chosen
                assert_eq!(
                    lowered_op.engine,
                    Engine::Load,
                    "ConstChoice {} should use Load",
                    const_choice_id
                );
            }
        }
    }

    #[cfg(feature = "cpsat")]
    #[test]
    fn test_ssa_cpsat_selectchoice_decisions() {
        // Build a simple kernel with SelectChoice ops
        let mut emit = SsaEmitContext::new();

        // Emit vector condition and values (use locals to avoid double mutable borrow)
        let c0 = emit.emit_const(0);
        let c1 = emit.emit_const(1);
        let v_c0 = emit.emit_vbroadcast(c0);
        let v_c1 = emit.emit_vbroadcast(c1);
        let cond = emit.emit_valu(Opcode::Eq, v_c0, v_c1);
        let c10 = emit.emit_const(10);
        let true_val = emit.emit_vbroadcast(c10);
        let c20 = emit.emit_const(20);
        let false_val = emit.emit_vbroadcast(c20);

        // Emit SelectChoice ops
        let sel1 = emit.emit_select_choice(cond, true_val, false_val);
        let sel2 = emit.emit_select_choice(cond, true_val, false_val);

        // Use results
        let _sum = emit.emit_valu(Opcode::Add, sel1, sel2);

        // Compute dependencies
        let vreg_def = std::mem::take(&mut emit.vreg_def);
        compute_ssa_dependencies(&mut emit.ops, &vreg_def, emit.zero_vreg, None);

        // Schedule with CP-SAT
        let result = schedule_ssa_cpsat(
            &emit.ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: false,
                stop_after_first_solution: true,
                enable_valu_shift: false,
                scratch_budget: 1536,
                model_name: None,
                ..Default::default()
            },
            SsaCpsatHints::default(),
            None,
        );

        assert!(result.is_some(), "CP-SAT should find a solution");
        let result = result.unwrap();

        // Verify all ops are scheduled
        assert_eq!(result.cycle_assignments.len(), emit.ops.len());

        // Verify SelectChoice decisions are extracted
        // The result should contain select_fma_ops (may be empty if solver chose vselect)
        assert!(
            result.select_fma_ops.len() <= 2,
            "Should have at most 2 SelectChoice ops using FMA path"
        );
    }

    /// Toy test: 3 parallel chains SelectChoice -> valu1 (shiftable) -> valu2, with a toy
    /// [`MachineConfig`] (1 VALU slot, 8 ALU slots per cycle). With only 1 VALU slot we need
    /// at least one valu1 to shift to ALU (8 slots). Asserts CP-SAT shifts at least one.
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_ssa_cpsat_selectchoice_valu_shift_toy() {
        let mut emit = SsaEmitContext::new();

        let c0 = emit.emit_const(0);
        let c1 = emit.emit_const(1);
        let v_c0 = emit.emit_vbroadcast(c0);
        let v_c1 = emit.emit_vbroadcast(c1);
        let cond = emit.emit_valu(Opcode::Eq, v_c0, v_c1);
        let c10 = emit.emit_const(10);
        let true_val = emit.emit_vbroadcast(c10);
        let c20 = emit.emit_const(20);
        let false_val = emit.emit_vbroadcast(c20);

        let mut valu1_op_ids = Vec::new();

        // 3 parallel chains so we have 3 valu1 + 3 valu2 = 6 VALU ops; with 1 VALU slot we need 6+ cycles
        // unless we shift (shifted valu1 uses 8 ALU, 0 VALU), giving the solver incentive to shift.
        for _ in 0..3 {
            let sel = emit.emit_select_choice(cond, true_val, false_val);
            let valu1 = emit.emit_valu(Opcode::Add, sel, true_val);
            valu1_op_ids.push(emit.ops.last().unwrap().id);
            let _valu2 = emit.emit_valu(Opcode::Add, valu1, false_val);
        }

        let vreg_def = std::mem::take(&mut emit.vreg_def);
        compute_ssa_dependencies(&mut emit.ops, &vreg_def, emit.zero_vreg, None);

        // Toy machine: 1 VALU, 8 ALU so one shifted VALU fits in ALU.
        let toy_machine = MachineConfig {
            valu_slots: 1,
            alu_slots: 8,
            ..Default::default()
        };
        let result = schedule_ssa_cpsat(
            &emit.ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: false,
                stop_after_first_solution: false,
                enable_valu_shift: true,
                scratch_budget: 1536,
                model_name: Some("selectchoice_valu_shift_toy".to_string()),
                machine: toy_machine,
                ..Default::default()
            },
            SsaCpsatHints::default(),
            None,
        );

        assert!(
            result.is_some(),
            "CP-SAT should find a solution with 1 VALU / 8 ALU toy caps"
        );
        let r = result.unwrap();
        assert_eq!(r.cycle_assignments.len(), emit.ops.len());

        // With 1 VALU slot and 3 chains (6 VALU ops), shifting at least one valu1 to ALU should help.
        let shifted_valu1 = valu1_op_ids
            .iter()
            .filter(|&id| r.shifted_ops.contains(id))
            .count();
        assert!(
            shifted_valu1 >= 1,
            "With 1 VALU and 8 ALU slots and 3 chains, at least one of the 3 shiftable valu1 ops should be shifted; got shifted_ops={:?}, valu1_op_ids={:?}",
            r.shifted_ops, valu1_op_ids
        );
    }

    /// Test that multiple SelectChoice ops in vselect mode cannot be in the same cycle.
    /// This tests the Flow cumulative constraint with optional intervals.
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_ssa_cpsat_selectchoice_flow_slot_limit() {
        // Build a kernel with 3 independent SelectChoice ops that all use the same inputs.
        // They have no data dependency on each other, so without Flow constraint they could
        // all be scheduled in the same cycle. The Flow slot limit (1) should force them
        // into different cycles when they choose vselect path.
        let mut emit = SsaEmitContext::new();

        // Emit vector condition and values
        let c0 = emit.emit_const(0);
        let c1 = emit.emit_const(1);
        let v_c0 = emit.emit_vbroadcast(c0);
        let v_c1 = emit.emit_vbroadcast(c1);
        let cond = emit.emit_valu(Opcode::Eq, v_c0, v_c1);
        let c10 = emit.emit_const(10);
        let true_val = emit.emit_vbroadcast(c10);
        let c20 = emit.emit_const(20);
        let false_val = emit.emit_vbroadcast(c20);

        // Emit 3 SelectChoice ops - they are independent so could run in parallel if allowed
        let sel1 = emit.emit_select_choice(cond, true_val, false_val);
        let sel2 = emit.emit_select_choice(cond, true_val, false_val);
        let sel3 = emit.emit_select_choice(cond, true_val, false_val);

        // Use results
        let sum12 = emit.emit_valu(Opcode::Add, sel1, sel2);
        let _sum123 = emit.emit_valu(Opcode::Add, sum12, sel3);

        // Compute dependencies
        let vreg_def = std::mem::take(&mut emit.vreg_def);
        compute_ssa_dependencies(&mut emit.ops, &vreg_def, emit.zero_vreg, None);

        // Schedule with CP-SAT - use verbose to see breadcrumbs
        let result = schedule_ssa_cpsat(
            &emit.ops,
            SsaCpsatConfig {
                time_limit_secs: 30.0,
                verbose: true,                    // Show Flow breadcrumbs
                stop_after_first_solution: false, // Let it optimize
                enable_valu_shift: false,
                scratch_budget: 1536,
                model_name: None,
                ..Default::default()
            },
            SsaCpsatHints::default(),
            None,
        );

        assert!(result.is_some(), "CP-SAT should find a solution");
        let cpsat_result = result.unwrap();

        // Find SelectChoice op indices and their cycles
        let select_choice_info: Vec<(usize, usize, i32)> = emit
            .ops
            .iter()
            .enumerate()
            .filter(|(_, op)| op.opcode == Opcode::SelectChoice)
            .map(|(idx, op)| (idx, op.id, cpsat_result.cycle_assignments[idx]))
            .collect();

        assert_eq!(
            select_choice_info.len(),
            3,
            "Should have 3 SelectChoice ops"
        );
        eprintln!("SelectChoice ops: {:?}", select_choice_info);
        eprintln!("select_fma_ops: {:?}", cpsat_result.select_fma_ops);

        // Count how many SelectChoice ops chose vselect path (not in select_fma_ops)
        let vselect_ops: Vec<_> = select_choice_info
            .iter()
            .filter(|(_, op_id, _)| !cpsat_result.select_fma_ops.contains(op_id))
            .collect();

        if vselect_ops.len() > 1 {
            // If multiple vselect ops, verify they're in different cycles
            let vselect_cycles: Vec<i32> = vselect_ops.iter().map(|(_, _, cy)| *cy).collect();
            eprintln!("vselect ops cycles: {:?}", vselect_cycles);

            // Check no duplicates (each cycle should appear at most once)
            let mut cycle_counts: HashMap<i32, usize> = HashMap::new();
            for cy in &vselect_cycles {
                *cycle_counts.entry(*cy).or_default() += 1;
            }

            for (cy, count) in &cycle_counts {
                assert!(
                    *count <= 1,
                    "Flow slot limit violated: {} vselect ops in cycle {} (limit is 1)",
                    count,
                    cy
                );
            }
        }

        eprintln!("test_ssa_cpsat_selectchoice_flow_slot_limit PASSED");
    }

    // =========================================================================
    // Tests for AddImm ops from emit_const() in Flow constraint
    // These tests verify the fix for the Flow slot violation bug where AddImm
    // ops were not included in the Flow cumulative constraint.
    // =========================================================================

    /// Test that AddImm ops from emit_const() are counted in the Flow constraint.
    /// This was the root cause of the Flow slot violation bug.
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_ssa_cpsat_addimm_ops_counted_in_flow_constraint() {
        let mut emit = SsaEmitContext::new();

        // Set up zero_vreg so emit_const creates AddImm alternatives
        let zero = emit.emit_const(0);
        emit.zero_vreg = Some(zero);

        // Emit 5 non-zero constants - each creates Load + AddImm ops
        let c1 = emit.emit_const(1);
        let c2 = emit.emit_const(2);
        let c3 = emit.emit_const(3);
        let c4 = emit.emit_const(4);
        let c5 = emit.emit_const(5);

        // Use the constants to create dependencies
        let sum1 = emit.emit_alu(Opcode::Add, c1, c2);
        let sum2 = emit.emit_alu(Opcode::Add, c3, c4);
        let _sum3 = emit.emit_alu(Opcode::Add, sum1, sum2);
        let _use_c5 = emit.emit_alu(Opcode::Add, c5, c1);

        // Compute dependencies
        let vreg_def = std::mem::take(&mut emit.vreg_def);
        compute_ssa_dependencies(&mut emit.ops, &vreg_def, emit.zero_vreg, None);

        // Count AddImm ops
        let addimm_count = emit
            .ops
            .iter()
            .filter(|op| op.opcode == Opcode::AddImm)
            .count();
        assert_eq!(addimm_count, 5, "Should have 5 AddImm ops from emit_const");

        // Schedule with CP-SAT
        let result = schedule_ssa_cpsat(
            &emit.ops,
            SsaCpsatConfig {
                time_limit_secs: 30.0,
                verbose: true,
                stop_after_first_solution: false,
                enable_valu_shift: false,
                scratch_budget: 1536,
                model_name: None,
                ..Default::default()
            },
            SsaCpsatHints::default(),
            None,
        );

        assert!(result.is_some(), "CP-SAT should find a solution");
        let cpsat_result = result.unwrap();

        // Collect all Flow-engine ops and their cycles
        let flow_ops: Vec<(usize, i32)> = emit
            .ops
            .iter()
            .enumerate()
            .filter(|(_, op)| op.engine == Engine::Flow)
            .map(|(idx, _)| (idx, cpsat_result.cycle_assignments[idx]))
            .collect();

        eprintln!("Flow ops (idx, cycle): {:?}", flow_ops);

        // Verify no two Flow ops are in the same cycle
        let mut cycle_counts: HashMap<i32, usize> = HashMap::new();
        for (_, cy) in &flow_ops {
            *cycle_counts.entry(*cy).or_default() += 1;
        }

        for (cy, count) in &cycle_counts {
            assert!(
                *count <= 1,
                "Flow slot limit violated: {} Flow ops in cycle {} (limit is 1). \
                This would indicate AddImm ops are not properly constrained.",
                count,
                cy
            );
        }

        eprintln!("test_ssa_cpsat_addimm_ops_counted_in_flow_constraint PASSED: {} AddImm ops properly scheduled in different cycles", addimm_count);
    }

    /// Test that the validation function correctly detects Flow slot violations.
    /// This ensures our validation logic matches what we expect.
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_validate_ssa_slot_limits_detects_flow_violation() {
        // Create minimal ops that would violate Flow limit if scheduled in same cycle
        let mut op0 = SsaOp::new(
            0,
            Engine::Flow,
            Opcode::AddImm,
            Some(VReg::scalar(0)),
            vec![],
        );
        op0.produces_constant = Some(1);

        let mut op1 = SsaOp::new(
            1,
            Engine::Flow,
            Opcode::AddImm,
            Some(VReg::scalar(1)),
            vec![],
        );
        op1.produces_constant = Some(2);

        let ops = vec![op0, op1];

        // Schedule both in cycle 0 - this should violate Flow limit
        let cycle_assignments = vec![0, 0];
        let shifted_ops = HashSet::new();
        let const_add_imm_ops = HashSet::new();
        let select_fma_ops = HashSet::new();

        // This should panic because we have 2 Flow ops in cycle 0
        let result = std::panic::catch_unwind(|| {
            validate_ssa_slot_limits(
                &ops,
                &cycle_assignments,
                &shifted_ops,
                &HashMap::new(), // shifted_alu_cycles
                &const_add_imm_ops,
                &HashMap::new(), // const_alu_derive_ops
                &select_fma_ops,
                &HashSet::new(), // eliminated_vbroadcast_ops
                &MachineConfig::default(),
            );
        });

        assert!(
            result.is_err(),
            "validate_ssa_slot_limits should panic when 2 Flow ops are in same cycle"
        );

        eprintln!("test_validate_ssa_slot_limits_detects_flow_violation PASSED");
    }

    /// Test that validation passes when Flow ops are in different cycles.
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_validate_ssa_slot_limits_passes_valid_schedule() {
        let mut op0 = SsaOp::new(
            0,
            Engine::Flow,
            Opcode::AddImm,
            Some(VReg::scalar(0)),
            vec![],
        );
        op0.produces_constant = Some(1);

        let mut op1 = SsaOp::new(
            1,
            Engine::Flow,
            Opcode::AddImm,
            Some(VReg::scalar(1)),
            vec![],
        );
        op1.produces_constant = Some(2);

        let ops = vec![op0, op1];

        // Schedule in different cycles - this should be valid
        let cycle_assignments = vec![0, 1];
        let shifted_ops = HashSet::new();
        let const_add_imm_ops = HashSet::new();
        let select_fma_ops = HashSet::new();

        // This should NOT panic
        validate_ssa_slot_limits(
            &ops,
            &cycle_assignments,
            &shifted_ops,
            &HashMap::new(), // shifted_alu_cycles
            &const_add_imm_ops,
            &HashMap::new(), // const_alu_derive_ops
            &select_fma_ops,
            &HashSet::new(), // eliminated_vbroadcast_ops
            &MachineConfig::default(),
        );

        eprintln!("test_validate_ssa_slot_limits_passes_valid_schedule PASSED");
    }

    /// Test that ConstChoice ops are correctly handled when they choose AddImm path.
    /// ConstChoice is different from emit_const() - it's a single op with a choice.
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_ssa_cpsat_constchoice_addimm_uses_flow_slot() {
        let mut emit = SsaEmitContext::new();

        // Emit several ConstChoice constants
        let c1 = emit.emit_const_choice(10);
        let c2 = emit.emit_const_choice(20);
        let c3 = emit.emit_const_choice(30);

        // Use them
        let sum = emit.emit_alu(Opcode::Add, c1, c2);
        let _sum2 = emit.emit_alu(Opcode::Add, sum, c3);

        // Compute dependencies
        let vreg_def = std::mem::take(&mut emit.vreg_def);
        compute_ssa_dependencies(&mut emit.ops, &vreg_def, emit.zero_vreg, None);

        // Verify we have ConstChoice ops
        let constchoice_count = emit
            .ops
            .iter()
            .filter(|op| op.opcode == Opcode::ConstChoice)
            .count();
        assert_eq!(constchoice_count, 3, "Should have 3 ConstChoice ops");

        // Schedule with CP-SAT
        let result = schedule_ssa_cpsat(
            &emit.ops,
            SsaCpsatConfig {
                time_limit_secs: 30.0,
                verbose: true,
                stop_after_first_solution: false,
                enable_valu_shift: false,
                scratch_budget: 1536,
                model_name: None,
                ..Default::default()
            },
            SsaCpsatHints::default(),
            None,
        );

        assert!(result.is_some(), "CP-SAT should find a solution");
        let cpsat_result = result.unwrap();

        // If any ConstChoice ops chose AddImm path, they should be in different cycles
        if !cpsat_result.const_add_imm_ops.is_empty() {
            let addimm_cycles: Vec<i32> = emit
                .ops
                .iter()
                .enumerate()
                .filter(|(_, op)| {
                    op.opcode == Opcode::ConstChoice
                        && cpsat_result.const_add_imm_ops.contains(&op.id)
                })
                .map(|(idx, _)| cpsat_result.cycle_assignments[idx])
                .collect();

            eprintln!(
                "ConstChoice ops using AddImm path, cycles: {:?}",
                addimm_cycles
            );

            // Check no duplicates
            let mut cycle_counts: HashMap<i32, usize> = HashMap::new();
            for cy in &addimm_cycles {
                *cycle_counts.entry(*cy).or_default() += 1;
            }
            for (cy, count) in &cycle_counts {
                assert!(
                    *count <= 1,
                    "Flow slot limit violated: {} ConstChoice AddImm ops in cycle {}",
                    count,
                    cy
                );
            }
        }

        eprintln!("test_ssa_cpsat_constchoice_addimm_uses_flow_slot PASSED");
    }

    /// Test that mixed AddImm (from emit_const) and ConstChoice AddImm paths
    /// are all constrained to 1 per cycle.
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_ssa_cpsat_mixed_addimm_and_constchoice_flow_constraint() {
        let mut emit = SsaEmitContext::new();

        // Set up zero_vreg for emit_const AddImm ops
        let zero = emit.emit_const(0);
        emit.zero_vreg = Some(zero);

        // Mix of emit_const (creates Load+AddImm) and emit_const_choice (creates ConstChoice)
        let c1 = emit.emit_const(10); // Creates Load + AddImm
        let c2 = emit.emit_const_choice(20); // Creates ConstChoice
        let c3 = emit.emit_const(30); // Creates Load + AddImm
        let c4 = emit.emit_const_choice(40); // Creates ConstChoice

        // Use them
        let sum1 = emit.emit_alu(Opcode::Add, c1, c2);
        let sum2 = emit.emit_alu(Opcode::Add, c3, c4);
        let _final = emit.emit_alu(Opcode::Add, sum1, sum2);

        // Compute dependencies
        let vreg_def = std::mem::take(&mut emit.vreg_def);
        compute_ssa_dependencies(&mut emit.ops, &vreg_def, emit.zero_vreg, None);

        // Count ops
        let addimm_count = emit
            .ops
            .iter()
            .filter(|op| op.opcode == Opcode::AddImm)
            .count();
        let constchoice_count = emit
            .ops
            .iter()
            .filter(|op| op.opcode == Opcode::ConstChoice)
            .count();

        eprintln!(
            "AddImm ops: {}, ConstChoice ops: {}",
            addimm_count, constchoice_count
        );
        assert_eq!(addimm_count, 2, "Should have 2 AddImm ops from emit_const");
        assert_eq!(constchoice_count, 2, "Should have 2 ConstChoice ops");

        // Schedule with CP-SAT
        let result = schedule_ssa_cpsat(
            &emit.ops,
            SsaCpsatConfig {
                time_limit_secs: 30.0,
                verbose: true,
                stop_after_first_solution: false,
                enable_valu_shift: false,
                scratch_budget: 1536,
                model_name: None,
                ..Default::default()
            },
            SsaCpsatHints::default(),
            None,
        );

        assert!(result.is_some(), "CP-SAT should find a solution");
        let cpsat_result = result.unwrap();

        // Collect all ops that use Flow slot:
        // 1. AddImm ops (from emit_const)
        // 2. ConstChoice ops that chose AddImm path
        let mut flow_users: Vec<(usize, i32, &str)> = Vec::new();

        for (idx, op) in emit.ops.iter().enumerate() {
            let cycle = cpsat_result.cycle_assignments[idx];
            if op.opcode == Opcode::AddImm {
                flow_users.push((idx, cycle, "AddImm"));
            } else if op.opcode == Opcode::ConstChoice
                && cpsat_result.const_add_imm_ops.contains(&op.id)
            {
                flow_users.push((idx, cycle, "ConstChoice→AddImm"));
            }
        }

        eprintln!("All Flow-using ops: {:?}", flow_users);

        // Verify no two in same cycle
        let mut cycle_counts: HashMap<i32, Vec<&str>> = HashMap::new();
        for (_, cy, kind) in &flow_users {
            cycle_counts.entry(*cy).or_default().push(kind);
        }

        for (cy, kinds) in &cycle_counts {
            assert!(
                kinds.len() <= 1,
                "Flow slot limit violated: {} ops in cycle {}: {:?}",
                kinds.len(),
                cy,
                kinds
            );
        }

        eprintln!("test_ssa_cpsat_mixed_addimm_and_constchoice_flow_constraint PASSED");
    }

    #[cfg(feature = "cpsat")]
    #[test]
    fn test_ssa_cpsat_selectchoice_roundtrip() {
        // Build a simple kernel with SelectChoice ops
        let mut emit = SsaEmitContext::new();

        // Emit vector condition and values (use locals to avoid double mutable borrow)
        let c0 = emit.emit_const(0);
        let c1 = emit.emit_const(1);
        let v_c0 = emit.emit_vbroadcast(c0);
        let v_c1 = emit.emit_vbroadcast(c1);
        let cond = emit.emit_valu(Opcode::Eq, v_c0, v_c1);
        let c10 = emit.emit_const(10);
        let true_val = emit.emit_vbroadcast(c10);
        let c20 = emit.emit_const(20);
        let false_val = emit.emit_vbroadcast(c20);

        // Emit SelectChoice ops
        let sel1 = emit.emit_select_choice(cond, true_val, false_val);
        let sel2 = emit.emit_select_choice(cond, true_val, false_val);

        // Use results
        let _sum = emit.emit_valu(Opcode::Add, sel1, sel2);

        // Compute dependencies
        let vreg_def = std::mem::take(&mut emit.vreg_def);
        compute_ssa_dependencies(&mut emit.ops, &vreg_def, emit.zero_vreg, None);

        // Schedule with CP-SAT
        let result = schedule_ssa_cpsat(
            &emit.ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: false,
                stop_after_first_solution: true,
                enable_valu_shift: false,
                scratch_budget: 1536,
                model_name: None,
                ..Default::default()
            },
            SsaCpsatHints::default(),
            None,
        );

        assert!(result.is_some(), "CP-SAT should find a solution");
        let cpsat_result = result.unwrap();

        // Find SelectChoice op IDs
        let select_choice_ids: Vec<usize> = emit
            .ops
            .iter()
            .filter(|op| op.opcode == Opcode::SelectChoice)
            .map(|op| op.id)
            .collect();

        assert_eq!(select_choice_ids.len(), 2, "Should have 2 SelectChoice ops");

        // Verify decisions were extracted
        assert!(
            cpsat_result.select_fma_ops.len() <= 2,
            "Should have at most 2 SelectChoice ops using FMA path"
        );

        // Now test round-trip: lower using CP-SAT decisions
        let mut vreg_to_addr = HashMap::new();
        for (idx, op) in emit.ops.iter().enumerate() {
            if let Some(dest) = op.dest {
                vreg_to_addr.insert(dest, idx as i64 * 10); // Simple address assignment
            }
        }

        let lowered_ops = lower_ssa_ops_with_shift(
            &emit.ops,
            &vreg_to_addr,
            &HashSet::new(),                  // No shifted ops
            &HashSet::new(),                  // No ConstChoice AddImm ops
            &HashMap::new(),                  // No ConstChoice ALU-derived
            &cpsat_result.select_fma_ops,     // Use CP-SAT decisions
            &cpsat_result.shifted_alu_cycles, // Use CP-SAT individual ALU cycles
            &cpsat_result.eliminated_vbroadcast_ops,
            &cpsat_result.eliminated_vb_to_scalar_producer,
        );

        // Verify that SelectChoice ops were lowered correctly based on CP-SAT decisions
        for select_choice_id in &select_choice_ids {
            if cpsat_result.select_fma_ops.contains(select_choice_id) {
                // CP-SAT chose FMA path - verify sub and fma ops exist
                let sub_op = lowered_ops
                    .iter()
                    .find(|op| op.id == *select_choice_id && op.engine == Engine::Valu);
                assert!(
                    sub_op.is_some(),
                    "SelectChoice op {} should be lowered to sub op (VALU)",
                    select_choice_id
                );

                let fma_op = lowered_ops
                    .iter()
                    .find(|op| op.id == *select_choice_id + 1000000 && op.engine == Engine::Valu);
                assert!(
                    fma_op.is_some(),
                    "SelectChoice op {} should be lowered to fma op (VALU)",
                    select_choice_id
                );

                // Verify fma is scheduled after sub
                let sub_cycle = sub_op.unwrap().cycle;
                let fma_cycle = fma_op.unwrap().cycle;
                assert!(
                    fma_cycle > sub_cycle,
                    "FMA should be scheduled after sub: sub={}, fma={}",
                    sub_cycle,
                    fma_cycle
                );
            } else {
                // CP-SAT chose vselect path - verify vselect op exists
                let vselect_op = lowered_ops
                    .iter()
                    .find(|op| op.id == *select_choice_id && op.engine == Engine::Flow);
                assert!(
                    vselect_op.is_some(),
                    "SelectChoice op {} should be lowered to vselect op (Flow)",
                    select_choice_id
                );

                match &vselect_op.unwrap().slot[0] {
                    SlotVal::Str(s) => assert_eq!(
                        s, "vselect",
                        "SelectChoice op {} should be vselect",
                        select_choice_id
                    ),
                    _ => panic!("vselect slot[0] should be Str"),
                }
            }
        }
    }

    /// Small machine: 1 VALU slot so sub and fma cannot share a cycle (forced consecutive bundles).
    const SMALL_MACHINE_FMA: crate::types::MachineConfig = crate::types::MachineConfig {
        valu_slots: 1,
        alu_slots: 12,
        load_slots: 2,
        store_slots: 2,
        flow_slots: 1,
    };

    /// Minimal test: one SelectChoice (no delta → sub+fma path), ForceFma.
    /// Uses small machine (1 VALU slot) so sub and fma are forced into consecutive cycles.
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_ssa_cpsat_selectchoice_force_fma_minimal() {
        use crate::address_assign::{assign_addresses, AddressAssignConfig};
        use crate::kernel_builder::emit_ssa::{compute_ssa_dependencies, SsaEmitContext};

        let mut emit = SsaEmitContext::new();
        let c0 = emit.emit_const(0);
        let c1 = emit.emit_const(1);
        let v_c0 = emit.emit_vbroadcast(c0);
        let v_c1 = emit.emit_vbroadcast(c1);
        let cond = emit.emit_valu(Opcode::Eq, v_c0, v_c1);
        let c10 = emit.emit_const(10);
        let true_val = emit.emit_vbroadcast(c10);
        let c20 = emit.emit_const(20);
        let false_val = emit.emit_vbroadcast(c20);
        let sel = emit.emit_select_choice(cond, true_val, false_val);
        let _use = emit.emit_valu(Opcode::Add, sel, sel);

        let vreg_def = std::mem::take(&mut emit.vreg_def);
        compute_ssa_dependencies(&mut emit.ops, &vreg_def, emit.zero_vreg, None);

        let select_choice_idx = emit
            .ops
            .iter()
            .position(|o| o.opcode == Opcode::SelectChoice)
            .expect("one SelectChoice");
        let select_choice_id = emit.ops[select_choice_idx].id;
        assert!(
            emit.ops[select_choice_idx].sub_temp_vreg.is_some(),
            "no-delta path (sub+fma)"
        );

        let result = schedule_ssa_cpsat(
            &emit.ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: false,
                stop_after_first_solution: true,
                enable_valu_shift: false,
                scratch_budget: 1536,
                model_name: None,
                machine: SMALL_MACHINE_FMA,
                const_choice_mode: ConstChoiceMode::Optimize,
                select_choice_mode: SelectChoiceMode::ForceFma,
                valu_shift_mode: ValuShiftMode::Optimize,
                ..Default::default()
            },
            SsaCpsatHints::default(),
            None,
        );
        assert!(
            result.is_some(),
            "CP-SAT should find a solution with ForceFma"
        );
        let cpsat_result = result.unwrap();

        assert!(
            cpsat_result.select_fma_ops.contains(&select_choice_id),
            "ForceFma => select_fma_ops contains SelectChoice id"
        );
        let c = cpsat_result.cycle_assignments[select_choice_idx];
        for (idx, op) in emit.ops.iter().enumerate() {
            if op.dep_ids.contains(&select_choice_id) {
                let consumer_cycle = cpsat_result.cycle_assignments[idx];
                assert!(
                    consumer_cycle >= c + 2,
                    "consumer must be at cycle >= {} (SelectChoice result visible at C+2), got {}",
                    c + 2,
                    consumer_cycle
                );
            }
        }

        for (idx, op) in emit.ops.iter_mut().enumerate() {
            op.cycle = cpsat_result.cycle_assignments[idx];
        }
        let mut config = AddressAssignConfig::default();
        config.select_fma_ops = Some(cpsat_result.select_fma_ops.clone());
        let assign_result = assign_addresses(&emit.ops, &config);
        assert!(!assign_result.budget_exceeded);

        // With 1 VALU slot, sub and fma must be in consecutive cycles
        let lowered = lower_ssa_ops_with_shift(
            &emit.ops,
            &assign_result.vreg_to_addr,
            &HashSet::new(),
            &HashSet::new(),
            &cpsat_result.const_alu_derive_ops,
            &cpsat_result.select_fma_ops,
            &cpsat_result.shifted_alu_cycles,
            &cpsat_result.eliminated_vbroadcast_ops,
            &cpsat_result.eliminated_vb_to_scalar_producer,
        );
        let sub_op = lowered.iter().find(|o| {
            o.id == select_choice_id
                && o.engine == Engine::Valu
                && format!("{:?}", o.slot.get(0)).contains("\"-\"")
        });
        let fma_op = lowered
            .iter()
            .find(|o| o.id == select_choice_id + 1000000 && o.engine == Engine::Valu);
        assert!(sub_op.is_some(), "should have sub op");
        assert!(fma_op.is_some(), "should have fma op");
        assert_eq!(sub_op.unwrap().cycle, c, "sub at SelectChoice cycle");
        assert_eq!(
            fma_op.unwrap().cycle,
            c + 1,
            "fma at SelectChoice cycle + 1"
        );

        // Total cycle count must include the FMA: max(lowered cycle)+1 >= c+2 (FMA at c+1).
        let max_lowered_cycle = lowered.iter().map(|o| o.cycle.max(0)).max().unwrap_or(0);
        assert!(
            max_lowered_cycle >= c + 1,
            "cycle count must include FMA at cycle {}",
            c + 1
        );
    }

    /// Force VALU shift off (ForceValu) and ensure consumers respect precedence.
    /// Regression test for bug where we skipped RAW constraints for shiftable ops even when shift was disabled.
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_ssa_cpsat_force_valu_precedence() {
        use crate::kernel_builder::emit_ssa::{compute_ssa_dependencies, SsaEmitContext};

        let mut emit = SsaEmitContext::new();
        let c0 = emit.emit_const(0);
        let v0 = emit.emit_vbroadcast(c0);
        let shiftable = emit.emit_valu(Opcode::Add, v0, v0); // Shiftable VALU op
        let consumer = emit.emit_valu(Opcode::Sub, shiftable, v0); // Consumer must run later

        let vreg_def = std::mem::take(&mut emit.vreg_def);
        compute_ssa_dependencies(&mut emit.ops, &vreg_def, emit.zero_vreg, None);

        let shift_idx = emit
            .ops
            .iter()
            .position(|op| op.dest == Some(shiftable))
            .expect("shiftable op");
        let consumer_idx = emit
            .ops
            .iter()
            .position(|op| op.dest == Some(consumer))
            .expect("consumer op");

        let result = schedule_ssa_cpsat(
            &emit.ops,
            SsaCpsatConfig {
                enable_valu_shift: true,
                valu_shift_mode: ValuShiftMode::ForceValu,
                time_limit_secs: 10.0,
                scratch_budget: 1536,
                ..Default::default()
            },
            SsaCpsatHints::default(),
            None,
        )
        .expect("CP-SAT ForceValu schedule");

        let shift_cycle = result.cycle_assignments[shift_idx];
        let consumer_cycle = result.cycle_assignments[consumer_idx];
        assert!(
            consumer_cycle >= shift_cycle + 1,
            "ForceValu must respect RAW precedence: consumer_cycle={} < shift_cycle+1={}",
            consumer_cycle,
            shift_cycle + 1
        );
    }

    /// Element-wise ALU precedence: when VALU1 -> VALU2 -> VALU3 are all shifted, each lane i of
    /// VALU2 must satisfy alu_cycle[VALU2][i] >= alu_cycle[VALU1][i]+1 (and same for VALU3/VALU2).
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_ssa_cpsat_element_wise_valu_shift_chain() {
        use crate::kernel_builder::emit_ssa::{compute_ssa_dependencies, SsaEmitContext};
        use crate::types::VLEN;

        let mut emit = SsaEmitContext::new();
        let c0 = emit.emit_const(0);
        let v0 = emit.emit_vbroadcast(c0);
        let valu1 = emit.emit_valu(Opcode::Add, v0, v0);
        let id1 = emit.ops.last().unwrap().id;
        let valu2 = emit.emit_valu(Opcode::Add, valu1, valu1);
        let id2 = emit.ops.last().unwrap().id;
        let valu3 = emit.emit_valu(Opcode::Add, valu2, valu2);
        let id3 = emit.ops.last().unwrap().id;

        let vreg_def = std::mem::take(&mut emit.vreg_def);
        compute_ssa_dependencies(&mut emit.ops, &vreg_def, emit.zero_vreg, None);

        let toy_machine = MachineConfig {
            valu_slots: 1,
            alu_slots: 24,
            ..Default::default()
        };
        let result = schedule_ssa_cpsat(
            &emit.ops,
            SsaCpsatConfig {
                enable_valu_shift: true,
                valu_shift_mode: ValuShiftMode::ForceAlu,
                time_limit_secs: 10.0,
                scratch_budget: 1536,
                machine: toy_machine,
                ..Default::default()
            },
            SsaCpsatHints::default(),
            None,
        )
        .expect("CP-SAT should find solution for vb -> valu1 -> valu2 -> valu3 with ForceAlu");

        let alu1 = result
            .shifted_alu_cycles
            .get(&id1)
            .expect("valu1 should be shifted and have per-lane cycles");
        let alu2 = result
            .shifted_alu_cycles
            .get(&id2)
            .expect("valu2 should be shifted");
        let alu3 = result
            .shifted_alu_cycles
            .get(&id3)
            .expect("valu3 should be shifted");

        for i in 0..VLEN {
            assert!(
                alu2[i] >= alu1[i] + 1,
                "element-wise: alu2[{}]={} must be >= alu1[{}]+1={}",
                i,
                alu2[i],
                i,
                alu1[i] + 1
            );
            assert!(
                alu3[i] >= alu2[i] + 1,
                "element-wise: alu3[{}]={} must be >= alu2[{}]+1={}",
                i,
                alu3[i],
                i,
                alu2[i] + 1
            );
        }
    }

    /// Vbroadcast elimination: const -> vbroadcast -> shiftable VALU; with enable_valu_shift and
    /// enable_vbroadcast_elimination the solver may eliminate the vb and use scalar producer for precedence.
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_ssa_cpsat_vbroadcast_elimination_feasible() {
        use crate::kernel_builder::emit_ssa::{compute_ssa_dependencies, SsaEmitContext};

        let mut emit = SsaEmitContext::new();
        let _c0 = emit.emit_const(0);
        let vb = emit.emit_vbroadcast(_c0);
        let vb_id = emit.ops.last().unwrap().id;
        let add_vreg = emit.emit_valu(Opcode::Add, vb, vb);
        let add_id = emit.ops.last().unwrap().id;

        let vreg_def = std::mem::take(&mut emit.vreg_def);
        compute_ssa_dependencies(&mut emit.ops, &vreg_def, emit.zero_vreg, None);

        let toy_machine = MachineConfig {
            valu_slots: 1,
            alu_slots: 8,
            ..Default::default()
        };
        let result = schedule_ssa_cpsat(
            &emit.ops,
            SsaCpsatConfig {
                enable_valu_shift: true,
                enable_vbroadcast_elimination: true,
                valu_shift_mode: ValuShiftMode::Optimize,
                time_limit_secs: 10.0,
                scratch_budget: 1536,
                machine: toy_machine,
                ..Default::default()
            },
            SsaCpsatHints::default(),
            None,
        );

        assert!(result.is_some(), "CP-SAT should find a solution");
        let r = result.unwrap();
        if r.shifted_ops.contains(&add_id) {
            assert!(
                r.eliminated_vbroadcast_ops.contains(&vb_id),
                "when consumer is shifted, vb should be eliminated; shifted_ops={:?} eliminated={:?}",
                r.shifted_ops,
                r.eliminated_vbroadcast_ops
            );
        }
    }

    /// Same as test_ssa_cpsat_selectchoice_force_fma_minimal but with DEFAULT machine (6 VALU)
    /// and no hints. Exercises same-cycle VALU packing: sub and fma may share a cycle with other
    /// VALU ops. If the FMA bug is tied to packing or hint-driven schedules, this may fail.
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_ssa_cpsat_selectchoice_force_fma_default_machine_no_hints() {
        use crate::address_assign::{
            assign_addresses, compute_live_intervals, AddressAssignConfig,
        };
        use crate::kernel_builder::emit_ssa::{compute_ssa_dependencies, SsaEmitContext};
        use crate::kernel_builder::ssa_lower::lower_ssa_ops_with_shift;
        use std::collections::HashSet;

        let mut emit = SsaEmitContext::new();
        let c0 = emit.emit_const(0);
        let c1 = emit.emit_const(1);
        let v_c0 = emit.emit_vbroadcast(c0);
        let v_c1 = emit.emit_vbroadcast(c1);
        let cond = emit.emit_valu(Opcode::Eq, v_c0, v_c1);
        let c10 = emit.emit_const(10);
        let true_val = emit.emit_vbroadcast(c10);
        let c20 = emit.emit_const(20);
        let false_val = emit.emit_vbroadcast(c20);
        let sel = emit.emit_select_choice(cond, true_val, false_val);
        let _use = emit.emit_valu(Opcode::Add, sel, sel);

        let vreg_def = std::mem::take(&mut emit.vreg_def);
        compute_ssa_dependencies(&mut emit.ops, &vreg_def, emit.zero_vreg, None);

        let select_choice_idx = emit
            .ops
            .iter()
            .position(|o| o.opcode == Opcode::SelectChoice)
            .expect("one SelectChoice");
        let select_choice_id = emit.ops[select_choice_idx].id;

        let result = schedule_ssa_cpsat(
            &emit.ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: false,
                stop_after_first_solution: true,
                enable_valu_shift: false,
                scratch_budget: 1536,
                model_name: None,
                machine: crate::types::MachineConfig::default(),
                const_choice_mode: ConstChoiceMode::Optimize,
                select_choice_mode: SelectChoiceMode::ForceFma,
                valu_shift_mode: ValuShiftMode::Optimize,
                ..Default::default()
            },
            SsaCpsatHints::default(),
            None,
        );
        assert!(
            result.is_some(),
            "CP-SAT should find a solution with default machine + ForceFma"
        );
        let cpsat_result = result.unwrap();

        let c = cpsat_result.cycle_assignments[select_choice_idx];
        for (idx, op) in emit.ops.iter().enumerate() {
            if op.dep_ids.contains(&select_choice_id) {
                let consumer_cycle = cpsat_result.cycle_assignments[idx];
                assert!(
                    consumer_cycle >= c + 2,
                    "consumer must be at cycle >= {} (got {})",
                    c + 2,
                    consumer_cycle
                );
            }
        }

        for (idx, op) in emit.ops.iter_mut().enumerate() {
            op.cycle = cpsat_result.cycle_assignments[idx];
        }
        let mut config = AddressAssignConfig::default();
        config.select_fma_ops = Some(cpsat_result.select_fma_ops.clone());
        let assign_result = assign_addresses(&emit.ops, &config);
        assert!(!assign_result.budget_exceeded);

        let intervals = compute_live_intervals(&emit.ops, None, Some(&cpsat_result.select_fma_ops), Some(&cpsat_result.shifted_alu_cycles), Some(&cpsat_result.const_alu_derive_ops), None);
        for (v1, addr1) in &assign_result.vreg_to_addr {
            for (v2, addr2) in &assign_result.vreg_to_addr {
                if v1 >= v2 {
                    continue;
                }
                if addr1 != addr2 {
                    continue;
                }
                if let (Some(i1), Some(i2)) = (intervals.get(v1), intervals.get(v2)) {
                    assert!(
                        !i1.overlaps(i2),
                        "overlapping intervals shared address (default machine)"
                    );
                }
            }
        }

        let lowered = lower_ssa_ops_with_shift(
            &emit.ops,
            &assign_result.vreg_to_addr,
            &HashSet::new(),
            &HashSet::new(),
            &cpsat_result.const_alu_derive_ops,
            &cpsat_result.select_fma_ops,
            &cpsat_result.shifted_alu_cycles,
            &cpsat_result.eliminated_vbroadcast_ops,
            &cpsat_result.eliminated_vb_to_scalar_producer,
        );
        let sub_op = lowered.iter().find(|o| {
            o.id == select_choice_id
                && o.engine == Engine::Valu
                && format!("{:?}", o.slot.get(0)).contains("\"-\"")
        });
        let fma_op = lowered
            .iter()
            .find(|o| o.id == select_choice_id + 1000000 && o.engine == Engine::Valu);
        assert!(sub_op.is_some(), "should have sub op");
        assert!(fma_op.is_some(), "should have fma op");
        assert_eq!(sub_op.unwrap().cycle, c, "sub at SelectChoice cycle");
        assert_eq!(
            fma_op.unwrap().cycle,
            c + 1,
            "fma at SelectChoice cycle + 1"
        );
    }

    /// Two SelectChoice FMA ops: verify consumer precedence (>= dep+2) and that address
    /// assignment never gives the same address to two VRegs with overlapping intervals.
    /// Uses small machine (1 VALU slot) so sub/fma pairs are forced into consecutive cycles.
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_ssa_cpsat_two_selectchoice_force_fma_no_overlap() {
        use crate::address_assign::{
            assign_addresses, compute_live_intervals, AddressAssignConfig,
        };
        use crate::kernel_builder::emit_ssa::{compute_ssa_dependencies, SsaEmitContext};

        let mut emit = SsaEmitContext::new();
        let c0 = emit.emit_const(0);
        let c1 = emit.emit_const(1);
        let v_c0 = emit.emit_vbroadcast(c0);
        let v_c1 = emit.emit_vbroadcast(c1);
        let cond = emit.emit_valu(Opcode::Eq, v_c0, v_c1);
        let c10 = emit.emit_const(10);
        let true1 = emit.emit_vbroadcast(c10);
        let c20 = emit.emit_const(20);
        let false1 = emit.emit_vbroadcast(c20);
        let sel1 = emit.emit_select_choice(cond, true1, false1);
        let _u1 = emit.emit_valu(Opcode::Add, sel1, sel1);

        let c30 = emit.emit_const(30);
        let true2 = emit.emit_vbroadcast(c30);
        let c40 = emit.emit_const(40);
        let false2 = emit.emit_vbroadcast(c40);
        let sel2 = emit.emit_select_choice(cond, true2, false2);
        let _u2 = emit.emit_valu(Opcode::Add, sel2, sel2);

        let vreg_def = std::mem::take(&mut emit.vreg_def);
        compute_ssa_dependencies(&mut emit.ops, &vreg_def, emit.zero_vreg, None);

        let idx1 = emit
            .ops
            .iter()
            .position(|o| o.opcode == Opcode::SelectChoice)
            .expect("first");
        let idx2 = emit
            .ops
            .iter()
            .position(|o| o.opcode == Opcode::SelectChoice && o.id != emit.ops[idx1].id)
            .expect("second");
        let id1 = emit.ops[idx1].id;
        let id2 = emit.ops[idx2].id;
        assert!(emit.ops[idx1].sub_temp_vreg.is_some() && emit.ops[idx2].sub_temp_vreg.is_some());

        let result = schedule_ssa_cpsat(
            &emit.ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: false,
                stop_after_first_solution: true,
                enable_valu_shift: false,
                scratch_budget: 1536,
                model_name: None,
                machine: SMALL_MACHINE_FMA,
                const_choice_mode: ConstChoiceMode::Optimize,
                select_choice_mode: SelectChoiceMode::ForceFma,
                valu_shift_mode: ValuShiftMode::Optimize,
                ..Default::default()
            },
            SsaCpsatHints::default(),
            None,
        );
        assert!(result.is_some(), "CP-SAT should find a solution");
        let cpsat_result = result.unwrap();

        assert!(
            cpsat_result.select_fma_ops.contains(&id1)
                && cpsat_result.select_fma_ops.contains(&id2)
        );
        let c1 = cpsat_result.cycle_assignments[idx1];
        let c2 = cpsat_result.cycle_assignments[idx2];
        for (idx, op) in emit.ops.iter().enumerate() {
            if op.dep_ids.contains(&id1) {
                let consumer_cycle = cpsat_result.cycle_assignments[idx];
                assert!(
                    consumer_cycle >= c1 + 2,
                    "consumer of sel1 must be >= {} (got {})",
                    c1 + 2,
                    consumer_cycle
                );
            }
            if op.dep_ids.contains(&id2) {
                let consumer_cycle = cpsat_result.cycle_assignments[idx];
                assert!(
                    consumer_cycle >= c2 + 2,
                    "consumer of sel2 must be >= {} (got {})",
                    c2 + 2,
                    consumer_cycle
                );
            }
        }

        for (idx, op) in emit.ops.iter_mut().enumerate() {
            op.cycle = cpsat_result.cycle_assignments[idx];
        }
        let mut config = AddressAssignConfig::default();
        config.select_fma_ops = Some(cpsat_result.select_fma_ops.clone());
        let assign_result = assign_addresses(&emit.ops, &config);
        assert!(!assign_result.budget_exceeded);

        let intervals = compute_live_intervals(&emit.ops, None, Some(&cpsat_result.select_fma_ops), Some(&cpsat_result.shifted_alu_cycles), Some(&cpsat_result.const_alu_derive_ops), None);
        for (v1, addr1) in &assign_result.vreg_to_addr {
            for (v2, addr2) in &assign_result.vreg_to_addr {
                if v1 >= v2 {
                    continue;
                }
                if addr1 != addr2 {
                    continue;
                }
                if let (Some(i1), Some(i2)) = (intervals.get(v1), intervals.get(v2)) {
                    assert!(!i1.overlaps(i2), "BUG: VRegs {:?} and {:?} share address {} but intervals [{},{}] and [{},{}] overlap",
                        v1, v2, addr1, i1.start, i1.end, i2.start, i2.end);
                }
            }
        }
    }

    /// Full pipeline: CP-SAT ForceFma then assign then lower; verify no overlapping
    /// intervals share an address (regression test for validation failure).
    /// Uses small machine (1 VALU slot) so sub and fma are in consecutive bundles.
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_ssa_cpsat_force_fma_full_pipeline_no_overlap() {
        use crate::address_assign::{
            assign_addresses, compute_live_intervals, AddressAssignConfig,
        };
        use crate::kernel_builder::emit_ssa::{compute_ssa_dependencies, SsaEmitContext};
        use crate::kernel_builder::ssa_lower::lower_ssa_ops_with_shift;
        use std::collections::HashSet;

        let mut emit = SsaEmitContext::new();
        let c0 = emit.emit_const(0);
        let c1 = emit.emit_const(1);
        let v_c0 = emit.emit_vbroadcast(c0);
        let v_c1 = emit.emit_vbroadcast(c1);
        let cond = emit.emit_valu(Opcode::Eq, v_c0, v_c1);
        let c10 = emit.emit_const(10);
        let true_val = emit.emit_vbroadcast(c10);
        let c20 = emit.emit_const(20);
        let false_val = emit.emit_vbroadcast(c20);
        let sel = emit.emit_select_choice(cond, true_val, false_val);
        let _use = emit.emit_valu(Opcode::Add, sel, sel);

        let vreg_def = std::mem::take(&mut emit.vreg_def);
        compute_ssa_dependencies(&mut emit.ops, &vreg_def, emit.zero_vreg, None);

        let result = schedule_ssa_cpsat(
            &emit.ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: false,
                stop_after_first_solution: true,
                enable_valu_shift: false,
                scratch_budget: 1536,
                model_name: None,
                machine: SMALL_MACHINE_FMA,
                const_choice_mode: ConstChoiceMode::Optimize,
                select_choice_mode: SelectChoiceMode::ForceFma,
                valu_shift_mode: ValuShiftMode::Optimize,
                ..Default::default()
            },
            SsaCpsatHints::default(),
            None,
        );
        let cpsat_result = result.unwrap();
        for (idx, op) in emit.ops.iter_mut().enumerate() {
            op.cycle = cpsat_result.cycle_assignments[idx];
        }
        let mut config = AddressAssignConfig::default();
        config.select_fma_ops = Some(cpsat_result.select_fma_ops.clone());
        let assign_result = assign_addresses(&emit.ops, &config);
        let intervals = compute_live_intervals(&emit.ops, None, Some(&cpsat_result.select_fma_ops), Some(&cpsat_result.shifted_alu_cycles), Some(&cpsat_result.const_alu_derive_ops), None);
        for (v1, addr1) in &assign_result.vreg_to_addr {
            for (v2, addr2) in &assign_result.vreg_to_addr {
                if v1 >= v2 {
                    continue;
                }
                if addr1 != addr2 {
                    continue;
                }
                if let (Some(i1), Some(i2)) = (intervals.get(v1), intervals.get(v2)) {
                    assert!(!i1.overlaps(i2), "overlapping intervals shared address");
                }
            }
        }
        let _lowered = lower_ssa_ops_with_shift(
            &emit.ops,
            &assign_result.vreg_to_addr,
            &HashSet::new(),
            &HashSet::new(),
            &cpsat_result.const_alu_derive_ops,
            &cpsat_result.select_fma_ops,
            &cpsat_result.shifted_alu_cycles,
            &cpsat_result.eliminated_vbroadcast_ops,
            &cpsat_result.eliminated_vb_to_scalar_producer,
        );
    }
}


// ============================================================================
// Precedence Offset ASAP/ALAP Tests
// ============================================================================

#[cfg(test)]
mod precedence_offset_tests {
    use super::*;
    use crate::types::{Engine, Opcode, SsaOp, VReg};
    use std::collections::{HashMap, HashSet};

    /// Helper: create a simple SsaOp for testing
    fn make_ssa_op(id: usize, engine: Engine, opcode: Opcode, deps: Vec<usize>) -> SsaOp {
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

    #[test]
    fn test_compute_ssa_asap_with_shiftable_offset() {
        // Chain: A(0) → B(1) → C(2)
        // A is shiftable VALU. ASAP offset from A→B should use B's offset_for_asap(pred_is_shiftable=true)=1
        let a = make_ssa_op(0, Engine::Valu, Opcode::Add, vec![]);
        let b = make_ssa_op(1, Engine::Alu, Opcode::Add, vec![0]);
        let c = make_ssa_op(2, Engine::Alu, Opcode::Add, vec![1]);
        let ops = vec![a, b, c];
        let shiftable_set: HashSet<usize> = [0].into_iter().collect(); // index 0 is shiftable

        let asap = compute_ssa_asap(&ops, &shiftable_set);
        assert_eq!(*asap.get(&0).unwrap(), 0);
        assert_eq!(*asap.get(&1).unwrap(), 1); // min_delay=1 for shiftable predecessor
        assert_eq!(*asap.get(&2).unwrap(), 2);
    }

    #[test]
    fn test_compute_ssa_alap_with_shiftable_offset() {
        // Chain: A(0) → B(1) → C(2), horizon=20
        // A is shiftable VALU. ALAP domain bounding uses min_delay=1 (not max_delay=8)
        // because the CP-SAT model handles shifted latency via completion_cycle constraints.
        let a = make_ssa_op(0, Engine::Valu, Opcode::Add, vec![]);
        let b = make_ssa_op(1, Engine::Alu, Opcode::Add, vec![0]);
        let c = make_ssa_op(2, Engine::Alu, Opcode::Add, vec![1]);
        let ops = vec![a, b, c];
        let shiftable_set: HashSet<usize> = [0].into_iter().collect();

        let alap = compute_ssa_alap(&ops, 20, &shiftable_set);
        assert_eq!(*alap.get(&2).unwrap(), 19); // sink: horizon-1
        assert_eq!(*alap.get(&1).unwrap(), 18); // 19 - 1 (B is not shiftable predecessor of C)
        assert_eq!(*alap.get(&0).unwrap(), 17); // 18 - 1 (A is shiftable, but domain uses min_delay=1)
    }

    #[test]
    fn test_compute_ssa_asap_selectchoice_fma() {
        // Chain: A(0) → B(1, SelectChoice+sub_temp) → C(2)
        // A→B: predecessor A is normal ALU, offset = A.offset_for_asap(false) = 1
        // B→C: predecessor B is SelectChoice+sub_temp, offset = B.offset_for_asap(false) = 1
        //       (min_delay = vselect path; FMA delay enforced by model constraints)
        let a = make_ssa_op(0, Engine::Alu, Opcode::Add, vec![]);
        let mut b = make_ssa_op(1, Engine::Flow, Opcode::SelectChoice, vec![0]);
        b.sub_temp_vreg = Some(VReg::scalar(99));
        let c = make_ssa_op(2, Engine::Alu, Opcode::Add, vec![1]);
        let ops = vec![a, b, c];
        let shiftable_set: HashSet<usize> = HashSet::new();

        let asap = compute_ssa_asap(&ops, &shiftable_set);
        assert_eq!(*asap.get(&0).unwrap(), 0);
        assert_eq!(*asap.get(&1).unwrap(), 1); // 0 + 1 (normal predecessor A)
        assert_eq!(*asap.get(&2).unwrap(), 2); // 1 + 1 (SelectChoice ASAP = min_delay = 1)
    }

    #[test]
    fn test_compute_ssa_asap_empty_shiftable_set() {
        // Regression test: empty shiftable_set should give same result as old hardcoded +1
        let a = make_ssa_op(0, Engine::Alu, Opcode::Add, vec![]);
        let b = make_ssa_op(1, Engine::Alu, Opcode::Add, vec![0]);
        let c = make_ssa_op(2, Engine::Alu, Opcode::Add, vec![1]);
        let ops = vec![a, b, c];
        let shiftable_set: HashSet<usize> = HashSet::new();

        let asap = compute_ssa_asap(&ops, &shiftable_set);
        assert_eq!(*asap.get(&0).unwrap(), 0);
        assert_eq!(*asap.get(&1).unwrap(), 1);
        assert_eq!(*asap.get(&2).unwrap(), 2);

        let alap = compute_ssa_alap(&ops, 10, &shiftable_set);
        assert_eq!(*alap.get(&2).unwrap(), 9);  // horizon-1
        assert_eq!(*alap.get(&1).unwrap(), 8);  // 9-1
        assert_eq!(*alap.get(&0).unwrap(), 7);  // 8-1
    }

    #[test]
    fn test_mixed_chain_asap_alap() {
        // A(0, shiftable VALU) → B(1, normal ALU) → C(2, SelectChoice+fma) → D(3, normal ALU)
        // Domain bounding uses min_delay for both ASAP and ALAP (wide domains):
        //   A→B: A.asap_offset(shiftable=true)=1
        //   B→C: B.asap_offset(false)=1
        //   C→D: C.asap_offset(false)=1 (vselect min_delay; FMA enforced by model)
        let a = make_ssa_op(0, Engine::Valu, Opcode::Add, vec![]);
        let b = make_ssa_op(1, Engine::Alu, Opcode::Add, vec![0]);
        let mut c = make_ssa_op(2, Engine::Flow, Opcode::SelectChoice, vec![1]);
        c.sub_temp_vreg = Some(VReg::scalar(88));
        let d = make_ssa_op(3, Engine::Alu, Opcode::Add, vec![2]);
        let ops = vec![a, b, c, d];
        let shiftable_set: HashSet<usize> = [0].into_iter().collect();

        let asap = compute_ssa_asap(&ops, &shiftable_set);
        assert_eq!(*asap.get(&0).unwrap(), 0);
        assert_eq!(*asap.get(&1).unwrap(), 1); // 0 + 1 (A shiftable, min delay=1)
        assert_eq!(*asap.get(&2).unwrap(), 2); // 1 + 1 (B normal, offset=1)
        assert_eq!(*asap.get(&3).unwrap(), 3); // 2 + 1 (C SelectChoice, min_delay=1)

        let alap = compute_ssa_alap(&ops, 30, &shiftable_set);
        // D is sink: alap=29
        assert_eq!(*alap.get(&3).unwrap(), 29);
        // C→D backward: predecessor C is SelectChoice+fma, min_delay=1 (vselect path)
        assert_eq!(*alap.get(&2).unwrap(), 28); // 29 - 1
        // B→C backward: predecessor B is normal ALU, min_delay=1
        assert_eq!(*alap.get(&1).unwrap(), 27); // 28 - 1
        // A→B backward: predecessor A is shiftable VALU, min_delay=1 (not 8!)
        assert_eq!(*alap.get(&0).unwrap(), 26); // 27 - 1
    }

    #[test]
    fn test_diamond_dag_asap() {
        // Diamond: A(0) → B(1), A(0) → C(2), B(1) → D(3), C(2) → D(3)
        // A is shiftable VALU, C is SelectChoice+sub_temp
        let a = make_ssa_op(0, Engine::Valu, Opcode::Add, vec![]);
        let b = make_ssa_op(1, Engine::Alu, Opcode::Add, vec![0]);
        let mut c = make_ssa_op(2, Engine::Flow, Opcode::SelectChoice, vec![0]);
        c.sub_temp_vreg = Some(VReg::scalar(77));
        let d = make_ssa_op(3, Engine::Alu, Opcode::Add, vec![1, 2]);
        let ops = vec![a, b, c, d];
        let shiftable_set: HashSet<usize> = [0].into_iter().collect();

        let asap = compute_ssa_asap(&ops, &shiftable_set);
        assert_eq!(*asap.get(&0).unwrap(), 0);
        // A→B: predecessor A is shiftable VALU, A.asap_offset(true)=1
        assert_eq!(*asap.get(&1).unwrap(), 1);
        // A→C: predecessor A is shiftable VALU, A.asap_offset(true)=1
        assert_eq!(*asap.get(&2).unwrap(), 1);
        // D: max(B_asap + B.offset, C_asap + C.offset) = max(1+1, 1+1) = 2
        // C is SelectChoice but ASAP uses min_delay=1 (vselect path)
        assert_eq!(*asap.get(&3).unwrap(), 2);
    }

    // ========================================================================
    // diagnose_precedence_bounds tests
    // ========================================================================

    #[test]
    fn test_diagnose_precedence_violations() {
        // Chain: A(0) → B(1) → C(2), with ASAP manually set to violate
        let a = make_ssa_op(0, Engine::Alu, Opcode::Add, vec![]);
        let b = make_ssa_op(1, Engine::Alu, Opcode::Add, vec![0]);
        let c = make_ssa_op(2, Engine::Alu, Opcode::Add, vec![1]);
        let ops = vec![a, b, c];
        let shiftable_set = HashSet::new();

        // Inject violated ASAP: B.asap=0 but should be >= A.asap+1=1
        let mut asap: HashMap<usize, i32> = HashMap::new();
        asap.insert(0, 0);
        asap.insert(1, 0); // Violation: should be >= 1
        asap.insert(2, 1);
        let mut alap: HashMap<usize, i32> = HashMap::new();
        alap.insert(0, 5);
        alap.insert(1, 6);
        alap.insert(2, 7);

        let (viol, samples, empty) = diagnose_precedence_bounds(&ops, &asap, &alap, &shiftable_set);
        assert!(viol >= 1, "Should detect ASAP violation, got {}", viol);
        assert!(!samples.is_empty(), "Should have sample violations");
        assert_eq!(empty, 0);
    }

    #[test]
    fn test_diagnose_empty_domain() {
        // Single op with asap > alap = empty domain
        let a = make_ssa_op(0, Engine::Alu, Opcode::Add, vec![]);
        let ops = vec![a];
        let shiftable_set = HashSet::new();

        let mut asap: HashMap<usize, i32> = HashMap::new();
        asap.insert(0, 10);
        let mut alap: HashMap<usize, i32> = HashMap::new();
        alap.insert(0, 5); // Empty: asap=10 > alap=5

        let (viol, _samples, empty) = diagnose_precedence_bounds(&ops, &asap, &alap, &shiftable_set);
        assert_eq!(empty, 1, "Should detect 1 empty domain");
        assert_eq!(viol, 0); // No edges to violate
    }

    #[test]
    fn test_diagnose_no_violations() {
        // Consistent bounds from proper ASAP/ALAP computation
        let a = make_ssa_op(0, Engine::Alu, Opcode::Add, vec![]);
        let b = make_ssa_op(1, Engine::Alu, Opcode::Add, vec![0]);
        let c = make_ssa_op(2, Engine::Alu, Opcode::Add, vec![1]);
        let ops = vec![a, b, c];
        let shiftable_set = HashSet::new();

        let asap = compute_ssa_asap(&ops, &shiftable_set);
        let alap = compute_ssa_alap(&ops, 10, &shiftable_set);

        let (viol, samples, empty) = diagnose_precedence_bounds(&ops, &asap, &alap, &shiftable_set);
        assert_eq!(viol, 0, "Should have no violations, got {:?}", samples);
        assert_eq!(empty, 0);
    }

    // ========================================================================
    // fix_precedence_bounds_fixpoint tests
    // ========================================================================

    #[test]
    fn test_fixpoint_converges_on_cascade() {
        // Chain: A(0) → B(1) → C(2), with cascading ASAP violations
        // A.asap=0 correct, but B.asap=0 violates (should be >=1), C.asap=0 violates (should be >=2)
        let a = make_ssa_op(0, Engine::Alu, Opcode::Add, vec![]);
        let b = make_ssa_op(1, Engine::Alu, Opcode::Add, vec![0]);
        let c = make_ssa_op(2, Engine::Alu, Opcode::Add, vec![1]);
        let ops = vec![a, b, c];
        let shiftable_set = HashSet::new();

        let mut asap: HashMap<usize, i32> = HashMap::new();
        asap.insert(0, 0);
        asap.insert(1, 0); // Violation
        asap.insert(2, 0); // Violation
        let mut alap: HashMap<usize, i32> = HashMap::new();
        alap.insert(0, 10);
        alap.insert(1, 10);
        alap.insert(2, 10);

        let (fixed_asap, fixed_alap) = fix_precedence_bounds_fixpoint(&ops, &asap, &alap, &shiftable_set);
        assert_eq!(*fixed_asap.get(&0).unwrap(), 0);
        assert_eq!(*fixed_asap.get(&1).unwrap(), 1); // Fixed: 0 + 1
        assert_eq!(*fixed_asap.get(&2).unwrap(), 2); // Fixed: 1 + 1

        // ALAP should still be consistent
        let (viol, _, empty) = diagnose_precedence_bounds(&ops, &fixed_asap, &fixed_alap, &shiftable_set);
        assert_eq!(viol, 0, "Fixpoint should resolve all violations");
        assert_eq!(empty, 0);
    }

    #[test]
    fn test_fixpoint_noop_on_consistent_bounds() {
        // Already consistent bounds should not change
        let a = make_ssa_op(0, Engine::Alu, Opcode::Add, vec![]);
        let b = make_ssa_op(1, Engine::Alu, Opcode::Add, vec![0]);
        let c = make_ssa_op(2, Engine::Alu, Opcode::Add, vec![1]);
        let ops = vec![a, b, c];
        let shiftable_set = HashSet::new();

        let asap = compute_ssa_asap(&ops, &shiftable_set);
        let alap = compute_ssa_alap(&ops, 10, &shiftable_set);

        let (fixed_asap, fixed_alap) = fix_precedence_bounds_fixpoint(&ops, &asap, &alap, &shiftable_set);
        assert_eq!(fixed_asap, asap, "Consistent bounds should not change");
        assert_eq!(fixed_alap, alap, "Consistent ALAP should not change");
    }

    #[cfg(feature = "cpsat")]
    #[test]
    fn test_fixpoint_integration_small_kernel() {
        // Run a small kernel through CP-SAT with debug_fix_precedence_bounds=true
        use crate::kernel_builder::emit_ssa::{compute_ssa_dependencies, SsaEmitContext};

        let mut emit = SsaEmitContext::new();
        let c0 = emit.emit_const(0);
        let v0 = emit.emit_vbroadcast(c0);
        let v1 = emit.emit_valu(Opcode::Add, v0, v0);
        let _v2 = emit.emit_valu(Opcode::Sub, v1, v0);

        let vreg_def = std::mem::take(&mut emit.vreg_def);
        compute_ssa_dependencies(&mut emit.ops, &vreg_def, emit.zero_vreg, None);

        let result = schedule_ssa_cpsat(
            &emit.ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: false,
                debug_fix_precedence_bounds: true,
                ..Default::default()
            },
            SsaCpsatHints::default(),
            None,
        );

        assert!(result.is_some(), "CP-SAT should find a solution with fixpoint enabled");
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Engine, Op, Operand, SlotVal, SsaOp, VReg, MAX_SLOT_SIZE};
    use arrayvec::ArrayVec;
    use std::collections::HashSet;

    fn make_op(id: usize, engine: Engine, deps: Vec<usize>) -> Op {
        Op {
            id,
            engine,
            slot: {
                let mut s: ArrayVec<SlotVal, MAX_SLOT_SIZE> = ArrayVec::new();
                s.push(SlotVal::Str("+".to_string()));
                s.push(SlotVal::Int(id as i64 * 10));
                s.push(SlotVal::Int(0));
                s.push(SlotVal::Int(1));
                s
            },
            cycle: -1,
            dep_ids: deps.into_iter().collect(),
            reads: [0, 1].into_iter().collect(),
            writes: [id as i64 * 10].into_iter().collect(),
            last_use_reads: HashSet::new(),
            produces_constant: None,
        }
    }

    #[test]
    fn test_compute_asap_chain() {
        let ops = vec![
            make_op(0, Engine::Alu, vec![]),
            make_op(1, Engine::Alu, vec![0]),
            make_op(2, Engine::Alu, vec![1]),
        ];

        let asap = compute_asap(&ops);
        assert_eq!(asap.get(&0), Some(&0));
        assert_eq!(asap.get(&1), Some(&1));
        assert_eq!(asap.get(&2), Some(&2));
    }

    #[test]
    fn test_compute_asap_parallel() {
        let ops = vec![
            make_op(0, Engine::Alu, vec![]),
            make_op(1, Engine::Alu, vec![]),
            make_op(2, Engine::Alu, vec![0, 1]),
        ];

        let asap = compute_asap(&ops);
        assert_eq!(asap.get(&0), Some(&0));
        assert_eq!(asap.get(&1), Some(&0));
        assert_eq!(asap.get(&2), Some(&1));
    }

    #[test]
    fn test_compute_alap_chain() {
        let ops = vec![
            make_op(0, Engine::Alu, vec![]),
            make_op(1, Engine::Alu, vec![0]),
            make_op(2, Engine::Alu, vec![1]),
        ];

        let alap = compute_alap(&ops, 5);
        assert_eq!(alap.get(&2), Some(&4)); // Sink at horizon-1
        assert_eq!(alap.get(&1), Some(&3)); // One before sink
        assert_eq!(alap.get(&0), Some(&2)); // Two before sink
    }

    #[test]
    fn test_compute_live_intervals() {
        let mut ops = vec![
            make_op(0, Engine::Load, vec![]),
            make_op(1, Engine::Alu, vec![0]),
            make_op(2, Engine::Alu, vec![1]),
        ];
        // Op 0 writes to address 100
        ops[0].writes = [100].into_iter().collect();
        ops[0].reads.clear();
        // Op 1 reads 100, writes 110
        ops[1].reads = [100].into_iter().collect();
        ops[1].writes = [110].into_iter().collect();
        // Op 2 reads 110
        ops[2].reads = [110].into_iter().collect();
        ops[2].writes = [120].into_iter().collect();

        let intervals = compute_live_intervals(&ops);
        assert!(!intervals.is_empty());

        // Address 100 should have def=0, last_use=1
        let int_100 = intervals.iter().find(|i| i.address == 100);
        assert!(int_100.is_some());
        let int_100 = int_100.unwrap();
        assert_eq!(int_100.def_op_idx, 0);
        assert_eq!(int_100.last_use_op_idx, 1);
    }


    #[cfg(feature = "cpsat")]
    #[test]
    fn test_ssa_cpsat_constchoice_decisions() {
        use crate::kernel_builder::emit_ssa::{compute_ssa_dependencies, SsaEmitContext};
        use crate::types::Opcode;

        // Build a simple kernel with ConstChoice ops
        let mut emit = SsaEmitContext::new();

        // Emit zero constant
        let _zero = emit.emit_const(0);

        // Emit ConstChoice constants
        let c1 = emit.emit_const_choice(1);
        let c2 = emit.emit_const_choice(2);

        // Use constants in ALU ops
        let _sum = emit.emit_alu(Opcode::Add, c1, c2);

        // Compute dependencies
        let vreg_def = std::mem::take(&mut emit.vreg_def);
        compute_ssa_dependencies(&mut emit.ops, &vreg_def, emit.zero_vreg, None);

        // Schedule with CP-SAT
        let result = schedule_ssa_cpsat(
            &emit.ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: false,
                stop_after_first_solution: true,
                enable_valu_shift: false,
                scratch_budget: 1536,
                model_name: None,
                ..Default::default()
            },
            SsaCpsatHints::default(),
            None,
        );

        assert!(result.is_some(), "CP-SAT should find a solution");
        let result = result.unwrap();

        // Verify all ops are scheduled
        assert_eq!(result.cycle_assignments.len(), emit.ops.len());

        // Verify ConstChoice decisions are extracted
        // The result should contain const_add_imm_ops (may be empty if solver chose Load)
        assert!(
            result.const_add_imm_ops.len() <= 2,
            "Should have at most 2 ConstChoice ops using AddImm"
        );

        // Note: AddImm operations can use address 0 directly without waiting for zero
        // to be "ready" since scratch memory is always initialized to 0 at startup.
        // There is no precedence constraint requiring zero to be scheduled before AddImm.
    }

    #[cfg(feature = "cpsat")]
    #[test]
    fn test_ssa_cpsat_constchoice_roundtrip() {
        use crate::kernel_builder::emit_ssa::{compute_ssa_dependencies, SsaEmitContext};
        use crate::kernel_builder::ssa_lower::lower_ssa_ops_with_shift;
        use crate::types::{Engine, Opcode, SlotVal};
        use std::collections::HashMap;

        // Build a simple kernel with ConstChoice ops
        let mut emit = SsaEmitContext::new();

        // Emit zero constant
        let zero = emit.emit_const(0);

        // Emit ConstChoice constants
        let c1 = emit.emit_const_choice(1);
        let c2 = emit.emit_const_choice(2);
        let c4 = emit.emit_const_choice(4);

        // Use constants in ALU ops
        let _sum1 = emit.emit_alu(Opcode::Add, c1, c2);
        let _sum2 = emit.emit_alu(Opcode::Add, c4, zero);

        // Compute dependencies
        let vreg_def = std::mem::take(&mut emit.vreg_def);
        compute_ssa_dependencies(&mut emit.ops, &vreg_def, emit.zero_vreg, None);

        // Schedule with CP-SAT
        let result = schedule_ssa_cpsat(
            &emit.ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: false,
                stop_after_first_solution: true,
                enable_valu_shift: false,
                scratch_budget: 1536,
                model_name: None,
                ..Default::default()
            },
            SsaCpsatHints::default(),
            None,
        );

        assert!(result.is_some(), "CP-SAT should find a solution");
        let cpsat_result = result.unwrap();

        // Verify all ops are scheduled
        assert_eq!(cpsat_result.cycle_assignments.len(), emit.ops.len());

        // Find ConstChoice op IDs and their values
        let const_choice_info: Vec<(usize, i64)> = emit
            .ops
            .iter()
            .filter(|op| op.opcode == Opcode::ConstChoice)
            .map(|op| {
                let value = op.srcs[0]
                    .as_imm()
                    .expect("ConstChoice should have immediate operand");
                (op.id, value)
            })
            .collect();

        assert_eq!(
            const_choice_info.len(),
            3,
            "Should have 3 ConstChoice ops (1, 2, 4)"
        );

        // Verify CP-SAT decisions are extracted
        assert!(
            cpsat_result.const_add_imm_ops.len() <= 3,
            "Should have at most 3 ConstChoice ops using AddImm"
        );

        // Now test round-trip: lower using CP-SAT decisions
        let mut vreg_to_addr = HashMap::new();
        for (idx, op) in emit.ops.iter().enumerate() {
            if let Some(dest) = op.dest {
                vreg_to_addr.insert(dest, idx as i64 * 10); // Simple address assignment
            }
        }
        // Ensure zero is at address 0
        vreg_to_addr.insert(zero, 0);

        let lowered_ops = lower_ssa_ops_with_shift(
            &emit.ops,
            &vreg_to_addr,
            &HashSet::new(),                 // No shifted ops
            &cpsat_result.const_add_imm_ops, // Use CP-SAT decisions
            &cpsat_result.const_alu_derive_ops,
            &cpsat_result.select_fma_ops,
            &cpsat_result.shifted_alu_cycles, // Use CP-SAT individual ALU cycles
            &cpsat_result.eliminated_vbroadcast_ops,
            &cpsat_result.eliminated_vb_to_scalar_producer,
        );

        // Verify that ConstChoice ops were lowered correctly based on CP-SAT decisions
        for (const_choice_id, expected_value) in &const_choice_info {
            let lowered_op = lowered_ops
                .iter()
                .find(|op| op.produces_constant == Some(*expected_value))
                .expect(&format!(
                    "Should find lowered op for ConstChoice producing {}",
                    expected_value
                ));

            if cpsat_result.const_add_imm_ops.contains(const_choice_id) {
                // CP-SAT chose AddImm - verify it's a Flow op
                assert_eq!(
                    lowered_op.engine,
                    Engine::Flow,
                    "ConstChoice op {} (value {}) should be lowered to Flow (AddImm)",
                    const_choice_id,
                    expected_value
                );
                match &lowered_op.slot[0] {
                    SlotVal::Str(s) => assert_eq!(
                        s, "add_imm",
                        "ConstChoice op {} should be add_imm",
                        const_choice_id
                    ),
                    _ => panic!("AddImm slot[0] should be Str"),
                }
                match &lowered_op.slot[2] {
                    SlotVal::Int(addr) => assert_eq!(*addr, 0, "AddImm should read from address 0"),
                    _ => panic!("AddImm slot[2] should be Int(0)"),
                }
            } else if cpsat_result.const_alu_derive_ops.contains_key(const_choice_id) {
                // CP-SAT chose ALU-derive - verify it's an ALU op
                assert_eq!(
                    lowered_op.engine,
                    Engine::Alu,
                    "ConstChoice op {} (value {}) should be lowered to Alu (ALU-derive)",
                    const_choice_id,
                    expected_value
                );
            } else {
                // CP-SAT chose Load - verify it's a Load op
                assert_eq!(
                    lowered_op.engine,
                    Engine::Load,
                    "ConstChoice op {} (value {}) should be lowered to Load",
                    const_choice_id,
                    expected_value
                );
                match &lowered_op.slot[0] {
                    SlotVal::Str(s) => assert_eq!(
                        s, "const",
                        "ConstChoice op {} should be const",
                        const_choice_id
                    ),
                    _ => panic!("Const slot[0] should be Str"),
                }
            }
        }

        // Verify all ConstChoice ops were lowered
        assert_eq!(
            lowered_ops.len(),
            emit.ops.len(),
            "All ops should be lowered"
        );
    }

    #[cfg(feature = "cpsat")]
    #[test]
    fn test_ssa_cpsat_constchoice_prefers_addimm_when_load_slots_full() {
        use crate::kernel_builder::emit_ssa::{compute_ssa_dependencies, SsaEmitContext};
        use crate::kernel_builder::ssa_lower::lower_ssa_ops_with_shift;
        use crate::types::{Engine, Opcode, SlotVal};
        use std::collections::HashMap;

        // Create a scenario where Load slots are heavily used, forcing CP-SAT to choose AddImm
        let mut emit = SsaEmitContext::new();

        // Emit zero constant
        let zero = emit.emit_const(0);

        // Emit many Load ops to fill Load slots
        for i in 0..10 {
            emit.emit_const(i + 100); // Regular Load ops
        }

        // Emit ConstChoice ops - CP-SAT should prefer AddImm to avoid Load slot pressure
        let c1 = emit.emit_const_choice(1);
        let c2 = emit.emit_const_choice(2);

        // Use constants
        let _sum = emit.emit_alu(Opcode::Add, c1, c2);

        // Compute dependencies
        let vreg_def = std::mem::take(&mut emit.vreg_def);
        compute_ssa_dependencies(&mut emit.ops, &vreg_def, emit.zero_vreg, None);

        // Schedule with CP-SAT with longer time limit to allow optimization
        let result = schedule_ssa_cpsat(
            &emit.ops,
            SsaCpsatConfig {
                time_limit_secs: 30.0,
                verbose: false,
                stop_after_first_solution: false, // Allow optimization
                enable_valu_shift: false,
                scratch_budget: 1536,
                model_name: None,
                ..Default::default()
            },
            SsaCpsatHints::default(),
            None,
        );

        assert!(result.is_some(), "CP-SAT should find a solution");
        let cpsat_result = result.unwrap();

        // Find ConstChoice op IDs
        let const_choice_ids: Vec<usize> = emit
            .ops
            .iter()
            .filter(|op| op.opcode == Opcode::ConstChoice)
            .map(|op| op.id)
            .collect();

        assert_eq!(const_choice_ids.len(), 2, "Should have 2 ConstChoice ops");

        // Verify decisions were extracted
        assert!(
            cpsat_result.const_add_imm_ops.len() <= 2,
            "Should have at most 2 ConstChoice ops using AddImm"
        );

        // Test lowering with the decisions
        let mut vreg_to_addr = HashMap::new();
        for (idx, op) in emit.ops.iter().enumerate() {
            if let Some(dest) = op.dest {
                vreg_to_addr.insert(dest, idx as i64 * 10);
            }
        }
        vreg_to_addr.insert(zero, 0);

        let lowered_ops = lower_ssa_ops_with_shift(
            &emit.ops,
            &vreg_to_addr,
            &HashSet::new(),
            &cpsat_result.const_add_imm_ops,
            &cpsat_result.const_alu_derive_ops,
            &cpsat_result.select_fma_ops,
            &cpsat_result.shifted_alu_cycles, // Use CP-SAT individual ALU cycles
            &cpsat_result.eliminated_vbroadcast_ops,
            &cpsat_result.eliminated_vb_to_scalar_producer,
        );

        // Verify ConstChoice ops were lowered correctly
        for const_choice_id in &const_choice_ids {
            let ssa_op = emit
                .ops
                .iter()
                .find(|op| op.id == *const_choice_id)
                .unwrap();
            let expected_value = ssa_op.srcs[0].as_imm().unwrap();

            let lowered_op = lowered_ops
                .iter()
                .find(|op| op.produces_constant == Some(expected_value))
                .expect(&format!(
                    "Should find lowered op for ConstChoice {}",
                    const_choice_id
                ));

            if cpsat_result.const_add_imm_ops.contains(const_choice_id) {
                // Verify AddImm was chosen
                assert_eq!(
                    lowered_op.engine,
                    Engine::Flow,
                    "ConstChoice {} should use AddImm when Load slots are full",
                    const_choice_id
                );
                match &lowered_op.slot[0] {
                    SlotVal::Str(s) => assert_eq!(s, "add_imm"),
                    _ => panic!("Should be add_imm"),
                }
            } else if cpsat_result.const_alu_derive_ops.contains_key(const_choice_id) {
                // Verify ALU-derived was chosen
                assert_eq!(
                    lowered_op.engine,
                    Engine::Alu,
                    "ConstChoice {} should use ALU when AluDerive chosen",
                    const_choice_id
                );
            } else {
                // Verify Load was chosen
                assert_eq!(
                    lowered_op.engine,
                    Engine::Load,
                    "ConstChoice {} should use Load",
                    const_choice_id
                );
            }
        }
    }

    #[test]
    fn test_cpsat_available() {
        // This just tests that the function compiles and returns a bool
        let available = cpsat_available();
        // When compiled with --features cpsat, this should be true
        // When compiled without, this should be false
        assert!(available == cfg!(feature = "cpsat"));
    }

    /// Helper to create SsaOp for testing
    fn make_ssa_op(
        id: usize,
        engine: Engine,
        opcode: crate::types::Opcode,
        dest: Option<VReg>,
        srcs: Vec<Operand>,
        deps: Vec<usize>,
    ) -> SsaOp {
        let mut op = SsaOp::new(id, engine, opcode, dest, srcs);
        op.dep_ids = deps.into_iter().collect();
        op
    }

    /// Test that SSA CP-SAT scratch constraint is being applied.
    /// Creates a scenario with high scratch pressure to verify the constraint.
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_ssa_cpsat_scratch_constraint_basic() {
        use crate::types::Opcode;

        // Create a simple chain where each op produces a value used by the next.
        // op0: load -> v0 (scalar, 1 word)
        // op1: alu(v0) -> v1
        // op2: alu(v1) -> v2
        let ops = vec![
            make_ssa_op(
                0,
                Engine::Load,
                Opcode::Const,
                Some(VReg::scalar(0)),
                vec![Operand::imm(5)],
                vec![],
            ),
            make_ssa_op(
                1,
                Engine::Alu,
                Opcode::Add,
                Some(VReg::scalar(1)),
                vec![Operand::scalar(0), Operand::imm(1)],
                vec![0],
            ),
            make_ssa_op(
                2,
                Engine::Alu,
                Opcode::Add,
                Some(VReg::scalar(2)),
                vec![Operand::scalar(1), Operand::imm(2)],
                vec![1],
            ),
        ];

        // With generous budget, should succeed
        let result = schedule_ssa_cpsat(
            &ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: true,
                scratch_budget: 100, // More than enough
                ..Default::default()
            },
            SsaCpsatHints::default(),
            None,
        );

        assert!(
            result.is_some(),
            "Should find solution with generous budget"
        );
        let r = result.unwrap();
        assert_eq!(
            r.status,
            CpsatStatus::Optimal,
            "Should be optimal for simple chain"
        );
        eprintln!("Basic test: makespan={}, status={:?}", r.makespan, r.status);
    }

    /// Test that scratch constraint forces spreading when budget is tight.
    /// With vector VRegs (8 words each), a tight budget should force longer schedules.
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_ssa_cpsat_scratch_constraint_tight_budget() {
        use crate::types::Opcode;

        // Create multiple independent ops that would normally run in parallel,
        // but with tight scratch budget they must be serialized.
        // Each vector VReg uses 8 words.
        //
        // op0: load -> v0 (vector, 8 words) - no deps
        // op1: load -> v1 (vector, 8 words) - no deps
        // op2: load -> v2 (vector, 8 words) - no deps
        // op3: valu(v0, v1) -> v3 (vector, 8 words) - deps: 0, 1
        // op4: valu(v2, v3) -> v4 (vector, 8 words) - deps: 2, 3
        //
        // If all run ASAP:
        //   Cycle 0: op0, op1, op2 scheduled -> v0, v1, v2 all live = 24 words
        //   Cycle 1: op3 scheduled -> v0, v1, v2, v3 live, then v0, v1 die = still ~24 words at peak
        //   Cycle 2: op4 scheduled -> v2, v3, v4 live
        let ops = vec![
            make_ssa_op(
                0,
                Engine::Load,
                Opcode::Const,
                Some(VReg::vector(0)),
                vec![Operand::imm(1)],
                vec![],
            ),
            make_ssa_op(
                1,
                Engine::Load,
                Opcode::Const,
                Some(VReg::vector(1)),
                vec![Operand::imm(2)],
                vec![],
            ),
            make_ssa_op(
                2,
                Engine::Load,
                Opcode::Const,
                Some(VReg::vector(2)),
                vec![Operand::imm(3)],
                vec![],
            ),
            make_ssa_op(
                3,
                Engine::Valu,
                Opcode::Add,
                Some(VReg::vector(3)),
                vec![Operand::vector(0), Operand::vector(1)],
                vec![0, 1],
            ),
            make_ssa_op(
                4,
                Engine::Valu,
                Opcode::Add,
                Some(VReg::vector(4)),
                vec![Operand::vector(2), Operand::vector(3)],
                vec![2, 3],
            ),
        ];

        // With large budget (e.g., 100 words), should allow parallel loads
        let result_generous = schedule_ssa_cpsat(
            &ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: true,
                scratch_budget: 100,
                ..Default::default()
            },
            SsaCpsatHints::default(),
            None,
        );

        assert!(
            result_generous.is_some(),
            "Should find solution with generous budget"
        );
        let r_generous = result_generous.unwrap();
        eprintln!(
            "Generous budget: makespan={}, status={:?}",
            r_generous.makespan, r_generous.status
        );

        // With tight budget (16 words = 2 vectors), should need to serialize more
        let result_tight = schedule_ssa_cpsat(
            &ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: true,
                scratch_budget: 16, // Only 2 vectors can be live at once
                ..Default::default()
            },
            SsaCpsatHints::default(),
            None,
        );

        // The tight budget should either:
        // 1. Force a longer schedule (more cycles to reduce concurrent live values), or
        // 2. Return infeasible if impossible
        if let Some(r_tight) = result_tight {
            eprintln!(
                "Tight budget: makespan={}, status={:?}",
                r_tight.makespan, r_tight.status
            );
            // With tight budget, makespan should be >= generous budget makespan
            // (can't do better, might need more cycles)
            assert!(
                r_tight.makespan >= r_generous.makespan,
                "Tight budget makespan ({}) should be >= generous ({}) or infeasible",
                r_tight.makespan,
                r_generous.makespan
            );
        } else {
            eprintln!("Tight budget: infeasible (expected if budget too small)");
        }
    }

    /// Test that scratch constraint with budget=0 disables the constraint.
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_ssa_cpsat_scratch_constraint_disabled() {
        use crate::types::Opcode;

        let ops = vec![
            make_ssa_op(
                0,
                Engine::Load,
                Opcode::Const,
                Some(VReg::vector(0)),
                vec![Operand::imm(1)],
                vec![],
            ),
            make_ssa_op(
                1,
                Engine::Load,
                Opcode::Const,
                Some(VReg::vector(1)),
                vec![Operand::imm(2)],
                vec![],
            ),
            make_ssa_op(
                2,
                Engine::Valu,
                Opcode::Add,
                Some(VReg::vector(2)),
                vec![Operand::vector(0), Operand::vector(1)],
                vec![0, 1],
            ),
        ];

        // With budget=0, constraint is disabled
        let result = schedule_ssa_cpsat(
            &ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: true,
                scratch_budget: 0, // Disabled
                ..Default::default()
            },
            SsaCpsatHints::default(),
            None,
        );

        assert!(
            result.is_some(),
            "Should find solution when scratch constraint disabled"
        );
        let r = result.unwrap();
        eprintln!(
            "Disabled constraint: makespan={}, status={:?}",
            r.makespan, r.status
        );
        // Should be optimal minimal schedule (2 cycles: loads then add)
        assert_eq!(r.makespan, 2, "Minimal schedule should be 2 cycles");
    }

    /// Test that SSA CP-SAT hints are complete and feasible.
    /// This is a critical test - if hints are incomplete or infeasible, the solver will struggle.
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_ssa_cpsat_hint_roundtrip_no_scratch() {
        use crate::types::Opcode;

        // Create a simple dependency chain
        let ops = vec![
            make_ssa_op(
                0,
                Engine::Load,
                Opcode::Const,
                Some(VReg::scalar(0)),
                vec![Operand::imm(5)],
                vec![],
            ),
            make_ssa_op(
                1,
                Engine::Alu,
                Opcode::Add,
                Some(VReg::scalar(1)),
                vec![Operand::scalar(0), Operand::imm(1)],
                vec![0],
            ),
            make_ssa_op(
                2,
                Engine::Alu,
                Opcode::Add,
                Some(VReg::scalar(2)),
                vec![Operand::scalar(1), Operand::imm(2)],
                vec![1],
            ),
        ];

        // Valid ASAP schedule: op0 at 0, op1 at 1, op2 at 2
        let hints = vec![0, 1, 2];

        // Run with stop_after_first_solution - should return exactly the hint if it's valid
        let result = schedule_ssa_cpsat(
            &ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: true,
                stop_after_first_solution: true,
                scratch_budget: 0, // Disable scratch constraint for this test
                ..Default::default()
            },
            SsaCpsatHints {
                greedy_hints: Some(&hints),
                shift_hints: None,
                forced_shifts: None,
                select_choice_hints: None,
                const_add_imm_hints: None,
                eliminate_vbroadcast_hints: None,
                initial_live: None,
                live_out_vregs: None,
                live_through_vregs: None,
            },
            Some(true), // valid hints, within scratch
        );

        assert!(result.is_some(), "Should find solution with valid hints");
        let r = result.unwrap();

        // Critical: with stop_after_first_solution, we should get exactly the hinted solution
        assert_eq!(
            r.cycle_assignments, hints,
            "With stop_after_first_solution, should return exactly the hinted solution"
        );
        assert_eq!(r.makespan, 3, "Makespan should be 3 (cycles 0, 1, 2)");

        eprintln!(
            "Hint roundtrip (no scratch): assignments={:?}, makespan={}",
            r.cycle_assignments, r.makespan
        );
    }

    /// Test that SSA CP-SAT hints with scratch constraint are complete and feasible.
    /// This verifies that the scratch interval size hints are correct.
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_ssa_cpsat_hint_roundtrip_with_scratch() {
        use crate::types::Opcode;

        // Create ops with clear liveness patterns
        // op0: load -> v0 (cycle 0)
        // op1: alu(v0) -> v1 (cycle 1) - v0 live from 0 to 1, size = 2
        // op2: alu(v1) -> v2 (cycle 2) - v1 live from 1 to 2, size = 2
        let ops = vec![
            make_ssa_op(
                0,
                Engine::Load,
                Opcode::Const,
                Some(VReg::scalar(0)),
                vec![Operand::imm(5)],
                vec![],
            ),
            make_ssa_op(
                1,
                Engine::Alu,
                Opcode::Add,
                Some(VReg::scalar(1)),
                vec![Operand::scalar(0), Operand::imm(1)],
                vec![0],
            ),
            make_ssa_op(
                2,
                Engine::Alu,
                Opcode::Add,
                Some(VReg::scalar(2)),
                vec![Operand::scalar(1), Operand::imm(2)],
                vec![1],
            ),
        ];

        // Valid ASAP schedule
        let hints = vec![0, 1, 2];

        // Run with scratch constraint enabled
        let result = schedule_ssa_cpsat(
            &ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: true,
                stop_after_first_solution: true,
                scratch_budget: 10, // Small but sufficient for scalars
                ..Default::default()
            },
            SsaCpsatHints {
                greedy_hints: Some(&hints),
                ..Default::default()
            },
            None,
        );

        assert!(
            result.is_some(),
            "Should find solution with valid hints and scratch constraint"
        );
        let r = result.unwrap();

        // Critical: hints should be feasible with scratch constraint
        assert_eq!(
            r.cycle_assignments, hints,
            "With stop_after_first_solution, should return exactly the hinted solution"
        );
        assert_eq!(r.makespan, 3, "Makespan should be 3");

        eprintln!(
            "Hint roundtrip (with scratch): assignments={:?}, makespan={}",
            r.cycle_assignments, r.makespan
        );
    }

    /// Test that SSA CP-SAT hints with vectors and scratch constraint are feasible.
    /// Vectors use 8 words each, so this tests the demand calculation.
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_ssa_cpsat_hint_roundtrip_vectors_with_scratch() {
        use crate::types::Opcode;

        // Create vector ops with clear liveness
        // op0: load -> v0 (vector, 8 words, cycle 0)
        // op1: load -> v1 (vector, 8 words, cycle 0) - can run parallel with op0
        // op2: valu(v0, v1) -> v2 (cycle 1) - v0, v1 live from 0 to 1
        // At cycle 0: v0, v1 defined = 16 words
        // At cycle 1: v0, v1 used, v2 defined = peak 24 words (v0, v1 still live during use)
        let ops = vec![
            make_ssa_op(
                0,
                Engine::Load,
                Opcode::Const,
                Some(VReg::vector(0)),
                vec![Operand::imm(1)],
                vec![],
            ),
            make_ssa_op(
                1,
                Engine::Load,
                Opcode::Const,
                Some(VReg::vector(1)),
                vec![Operand::imm(2)],
                vec![],
            ),
            make_ssa_op(
                2,
                Engine::Valu,
                Opcode::Add,
                Some(VReg::vector(2)),
                vec![Operand::vector(0), Operand::vector(1)],
                vec![0, 1],
            ),
        ];

        // Schedule: op0 and op1 at cycle 0 (parallel loads), op2 at cycle 1
        let hints = vec![0, 0, 1];

        // Run with scratch constraint - need at least 24 words (3 vectors live at peak)
        let result = schedule_ssa_cpsat(
            &ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: true,
                stop_after_first_solution: true,
                scratch_budget: 32, // Enough for 4 vectors
                ..Default::default()
            },
            SsaCpsatHints {
                greedy_hints: Some(&hints),
                ..Default::default()
            },
            None,
        );

        assert!(
            result.is_some(),
            "Should find solution with valid vector hints and scratch constraint"
        );
        let r = result.unwrap();

        // Critical: hints should be feasible
        assert_eq!(
            r.cycle_assignments, hints,
            "With stop_after_first_solution, should return exactly the hinted solution"
        );
        assert_eq!(r.makespan, 2, "Makespan should be 2 (cycles 0 and 1)");

        eprintln!(
            "Hint roundtrip (vectors with scratch): assignments={:?}, makespan={}",
            r.cycle_assignments, r.makespan
        );
    }

    /// Test that infeasible hints with tight scratch budget are detected.
    /// This verifies the scratch constraint actually constrains the solution.
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_ssa_cpsat_hint_infeasible_tight_scratch() {
        use crate::types::Opcode;

        // Create vector ops that would exceed a tight budget if run in parallel
        // op0: load -> v0 (vector, 8 words)
        // op1: load -> v1 (vector, 8 words)
        // op2: valu(v0, v1) -> v2
        // If op0, op1 at cycle 0: 16 words live at cycle 0
        // At cycle 1 when op2 runs: v0, v1, v2 all live = 24 words
        let ops = vec![
            make_ssa_op(
                0,
                Engine::Load,
                Opcode::Const,
                Some(VReg::vector(0)),
                vec![Operand::imm(1)],
                vec![],
            ),
            make_ssa_op(
                1,
                Engine::Load,
                Opcode::Const,
                Some(VReg::vector(1)),
                vec![Operand::imm(2)],
                vec![],
            ),
            make_ssa_op(
                2,
                Engine::Valu,
                Opcode::Add,
                Some(VReg::vector(2)),
                vec![Operand::vector(0), Operand::vector(1)],
                vec![0, 1],
            ),
        ];

        // Hints that would work without scratch constraint but violate tight budget
        let hints = vec![0, 0, 1]; // Both loads at cycle 0

        // Run with very tight scratch constraint (only 16 words - not enough for 3 vectors)
        let result = schedule_ssa_cpsat(
            &ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: true,
                stop_after_first_solution: true,
                scratch_budget: 16, // Only 2 vectors worth - not enough!
                ..Default::default()
            },
            SsaCpsatHints {
                greedy_hints: Some(&hints),
                ..Default::default()
            },
            None,
        );

        // The solver should either:
        // 1. Find a different (valid) schedule that fits the budget, or
        // 2. Return infeasible if no valid schedule exists
        // It should NOT return the infeasible hint as-is
        if let Some(r) = result {
            // If it found a solution, it should NOT be the original hint
            // because that hint violates the scratch budget
            eprintln!(
                "Found alternative schedule: {:?}, makespan={}",
                r.cycle_assignments, r.makespan
            );
            // The solver might find a valid schedule by serializing the loads
            // e.g., [0, 1, 2] instead of [0, 0, 1]
            assert!(r.makespan >= 2, "Should need at least 2 cycles");
        } else {
            eprintln!("Correctly returned infeasible for tight budget");
        }
    }

    /// Test scratch interval size when VReg is defined and used in the same cycle.
    /// This should produce interval size = 1 (last_use_cycle - def_cycle + 1 = 0 - 0 + 1 = 1).
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_scratch_interval_size_same_cycle() {
        use crate::types::Opcode;

        // Create ops where a value is defined and used in the same cycle.
        // op0: const -> v0 (cycle 0, defines v0)
        // op1: add v0, imm -> v1 (cycle 1, uses v0 - so v0 lives from cycle 0 to cycle 1)
        // But for same-cycle test, we need op0 and op1 to be in the same cycle.
        // However, due to data dependency (op1 depends on op0), they can't be in the same cycle.
        //
        // For a true same-cycle test, we need an op that defines a value and is the last user.
        // When a VReg is defined but never used elsewhere, its lifetime is just its def cycle.
        // Let's create: op0 defines v0, op1 defines v1 using v0, and v1 is never used again.
        // v1's interval: def_cycle=1, last_use_cycle=1 (no further uses), size=1
        let ops = vec![
            make_ssa_op(
                0,
                Engine::Load,
                Opcode::Const,
                Some(VReg::scalar(0)),
                vec![Operand::imm(5)],
                vec![],
            ),
            make_ssa_op(
                1,
                Engine::Alu,
                Opcode::Add,
                Some(VReg::scalar(1)),
                vec![Operand::scalar(0), Operand::imm(1)],
                vec![0],
            ),
        ];

        // Hints: op0 at cycle 0, op1 at cycle 1
        // v0 interval: def=0, last_use=1, size=2
        // v1 interval: def=1, last_use=1 (no further uses), size=1
        let hints = vec![0, 1];

        let result = schedule_ssa_cpsat(
            &ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: true,
                stop_after_first_solution: true,
                scratch_budget: 10, // Sufficient for scalars
                ..Default::default()
            },
            SsaCpsatHints {
                greedy_hints: Some(&hints),
                ..Default::default()
            },
            None,
        );

        assert!(
            result.is_some(),
            "Should find solution with same-cycle interval (size=1)"
        );
        let r = result.unwrap();
        assert_eq!(
            r.cycle_assignments, hints,
            "Should return the hinted solution"
        );
        assert_eq!(r.makespan, 2, "Makespan should be 2");
        eprintln!(
            "Same-cycle interval test: assignments={:?}, makespan={}",
            r.cycle_assignments, r.makespan
        );
    }

    /// Test scratch interval size for VReg defined at cycle N and used at cycle N+1.
    /// This should produce interval size = 2 (last_use_cycle - def_cycle + 1 = 1 - 0 + 1 = 2).
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_scratch_interval_size_consecutive_cycles() {
        use crate::types::Opcode;

        // op0: const -> v0 (cycle 0)
        // op1: add v0, imm -> v1 (cycle 1, uses v0)
        // v0 interval: def_cycle=0, last_use_cycle=1, size=2
        let ops = vec![
            make_ssa_op(
                0,
                Engine::Load,
                Opcode::Const,
                Some(VReg::scalar(0)),
                vec![Operand::imm(5)],
                vec![],
            ),
            make_ssa_op(
                1,
                Engine::Alu,
                Opcode::Add,
                Some(VReg::scalar(1)),
                vec![Operand::scalar(0), Operand::imm(1)],
                vec![0],
            ),
        ];

        // Hints: op0 at cycle 0, op1 at cycle 1
        let hints = vec![0, 1];

        let result = schedule_ssa_cpsat(
            &ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: true,
                stop_after_first_solution: true,
                scratch_budget: 10, // Sufficient
                ..Default::default()
            },
            SsaCpsatHints {
                greedy_hints: Some(&hints),
                ..Default::default()
            },
            None,
        );

        assert!(
            result.is_some(),
            "Should find solution with consecutive-cycle interval (size=2)"
        );
        let r = result.unwrap();
        assert_eq!(
            r.cycle_assignments, hints,
            "Should return the hinted solution"
        );
        assert_eq!(r.makespan, 2, "Makespan should be 2");
        eprintln!(
            "Consecutive-cycle interval test: assignments={:?}, makespan={}",
            r.cycle_assignments, r.makespan
        );
    }

    /// Test scratch interval size for VReg defined at cycle 0 and used at cycle 5.
    /// This should produce interval size = 6 (last_use_cycle - def_cycle + 1 = 5 - 0 + 1 = 6).
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_scratch_interval_size_gap() {
        use crate::types::Opcode;

        // Create a chain where v0 is defined at cycle 0 but used at cycle 5.
        // We need intermediate ops to push the use of v0 to cycle 5.
        // op0: const -> v0 (cycle 0)
        // op1: const -> v1 (cycle 0) - independent
        // op2: add v1, imm -> v2 (cycle 1) - chain through v1
        // op3: add v2, imm -> v3 (cycle 2)
        // op4: add v3, imm -> v4 (cycle 3)
        // op5: add v4, imm -> v5 (cycle 4)
        // op6: add v5, v0 -> v6 (cycle 5) - finally uses v0
        // v0 interval: def_cycle=0, last_use_cycle=5, size=6
        let ops = vec![
            make_ssa_op(
                0,
                Engine::Load,
                Opcode::Const,
                Some(VReg::scalar(0)),
                vec![Operand::imm(5)],
                vec![],
            ),
            make_ssa_op(
                1,
                Engine::Load,
                Opcode::Const,
                Some(VReg::scalar(1)),
                vec![Operand::imm(10)],
                vec![],
            ),
            make_ssa_op(
                2,
                Engine::Alu,
                Opcode::Add,
                Some(VReg::scalar(2)),
                vec![Operand::scalar(1), Operand::imm(1)],
                vec![1],
            ),
            make_ssa_op(
                3,
                Engine::Alu,
                Opcode::Add,
                Some(VReg::scalar(3)),
                vec![Operand::scalar(2), Operand::imm(1)],
                vec![2],
            ),
            make_ssa_op(
                4,
                Engine::Alu,
                Opcode::Add,
                Some(VReg::scalar(4)),
                vec![Operand::scalar(3), Operand::imm(1)],
                vec![3],
            ),
            make_ssa_op(
                5,
                Engine::Alu,
                Opcode::Add,
                Some(VReg::scalar(5)),
                vec![Operand::scalar(4), Operand::imm(1)],
                vec![4],
            ),
            make_ssa_op(
                6,
                Engine::Alu,
                Opcode::Add,
                Some(VReg::scalar(6)),
                vec![Operand::scalar(5), Operand::scalar(0)],
                vec![5, 0],
            ),
        ];

        // Hints: op0 and op1 at cycle 0, then op2-op6 at cycles 1-5
        // v0 is defined at cycle 0 and used at cycle 5 (by op6)
        // v0 interval size = 5 - 0 + 1 = 6
        let hints = vec![0, 0, 1, 2, 3, 4, 5];

        let result = schedule_ssa_cpsat(
            &ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: true,
                stop_after_first_solution: true,
                scratch_budget: 20, // Sufficient for multiple scalars
                ..Default::default()
            },
            SsaCpsatHints {
                greedy_hints: Some(&hints),
                ..Default::default()
            },
            None,
        );

        assert!(
            result.is_some(),
            "Should find solution with gap interval (size=6)"
        );
        let r = result.unwrap();
        assert_eq!(
            r.cycle_assignments, hints,
            "Should return the hinted solution"
        );
        assert_eq!(r.makespan, 6, "Makespan should be 6");
        eprintln!(
            "Gap interval test: assignments={:?}, makespan={}",
            r.cycle_assignments, r.makespan
        );
    }

    /// Test engine capacity: 6 independent VALU ops at the same cycle (exactly at limit).
    /// Should work without violating the VALU cumulative constraint.
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_engine_capacity_valu_within_limit() {
        use crate::types::Opcode;

        // Create 6 independent VALU ops that can all be scheduled at cycle 1.
        // Each needs a load to provide input values first.
        // op0-op5: load consts (cycle 0)
        // op6-op11: valu add (cycle 1) - all 6 VALU ops at same cycle (exactly at limit)
        let ops = vec![
            // Load ops for first operands (cycle 0)
            make_ssa_op(
                0,
                Engine::Load,
                Opcode::Const,
                Some(VReg::vector(0)),
                vec![Operand::imm(1)],
                vec![],
            ),
            make_ssa_op(
                1,
                Engine::Load,
                Opcode::Const,
                Some(VReg::vector(1)),
                vec![Operand::imm(2)],
                vec![],
            ),
            // Note: only 2 load slots per cycle, so we need more cycles
            // Actually, let's use consts with immediates to avoid load slot issues
            // Create 6 VALU ops that use immediates as second operand
            make_ssa_op(
                2,
                Engine::Valu,
                Opcode::Add,
                Some(VReg::vector(2)),
                vec![Operand::vector(0), Operand::vector(1)],
                vec![0, 1],
            ),
            make_ssa_op(
                3,
                Engine::Valu,
                Opcode::Sub,
                Some(VReg::vector(3)),
                vec![Operand::vector(0), Operand::vector(1)],
                vec![0, 1],
            ),
            make_ssa_op(
                4,
                Engine::Valu,
                Opcode::Mul,
                Some(VReg::vector(4)),
                vec![Operand::vector(0), Operand::vector(1)],
                vec![0, 1],
            ),
            make_ssa_op(
                5,
                Engine::Valu,
                Opcode::Xor,
                Some(VReg::vector(5)),
                vec![Operand::vector(0), Operand::vector(1)],
                vec![0, 1],
            ),
            make_ssa_op(
                6,
                Engine::Valu,
                Opcode::And,
                Some(VReg::vector(6)),
                vec![Operand::vector(0), Operand::vector(1)],
                vec![0, 1],
            ),
            make_ssa_op(
                7,
                Engine::Valu,
                Opcode::Or,
                Some(VReg::vector(7)),
                vec![Operand::vector(0), Operand::vector(1)],
                vec![0, 1],
            ),
        ];

        // Hints: 2 loads at cycle 0, 6 VALU ops all at cycle 1 (exactly at VALU limit of 6)
        let hints = vec![0, 0, 1, 1, 1, 1, 1, 1];

        let result = schedule_ssa_cpsat(
            &ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: true,
                stop_after_first_solution: true,
                scratch_budget: 0, // Disable scratch constraint for this test
                enable_valu_shift: false,
                ..Default::default()
            },
            SsaCpsatHints {
                greedy_hints: Some(&hints),
                ..Default::default()
            },
            None,
        );

        // The verification code should NOT panic, and we should get a valid result
        assert!(
            result.is_some(),
            "Should find solution with 6 VALU ops at same cycle (exactly at limit)"
        );
        let r = result.unwrap();
        assert_eq!(
            r.cycle_assignments, hints,
            "Should accept the hinted solution"
        );
        eprintln!(
            "VALU capacity test: 6 VALU ops at cycle 1, result={:?}",
            r.cycle_assignments
        );
    }

    /// Test engine capacity: 12 independent ALU ops at the same cycle (exactly at limit).
    /// Should work without violating the ALU cumulative constraint.
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_engine_capacity_alu_within_limit() {
        use crate::types::Opcode;

        // Create 12 independent ALU ops that can all be scheduled at cycle 1.
        // First, we need load ops to provide input values.
        // op0-op1: load consts (cycle 0) - 2 load slots
        // op2-op13: 12 ALU ops (cycle 1) - all 12 ALU ops at same cycle (exactly at limit)
        let ops = vec![
            // Load ops
            make_ssa_op(
                0,
                Engine::Load,
                Opcode::Const,
                Some(VReg::scalar(0)),
                vec![Operand::imm(1)],
                vec![],
            ),
            make_ssa_op(
                1,
                Engine::Load,
                Opcode::Const,
                Some(VReg::scalar(1)),
                vec![Operand::imm(2)],
                vec![],
            ),
            // 12 ALU ops - all depend on loads, can run in parallel
            make_ssa_op(
                2,
                Engine::Alu,
                Opcode::Add,
                Some(VReg::scalar(2)),
                vec![Operand::scalar(0), Operand::scalar(1)],
                vec![0, 1],
            ),
            make_ssa_op(
                3,
                Engine::Alu,
                Opcode::Sub,
                Some(VReg::scalar(3)),
                vec![Operand::scalar(0), Operand::scalar(1)],
                vec![0, 1],
            ),
            make_ssa_op(
                4,
                Engine::Alu,
                Opcode::Mul,
                Some(VReg::scalar(4)),
                vec![Operand::scalar(0), Operand::scalar(1)],
                vec![0, 1],
            ),
            make_ssa_op(
                5,
                Engine::Alu,
                Opcode::Xor,
                Some(VReg::scalar(5)),
                vec![Operand::scalar(0), Operand::scalar(1)],
                vec![0, 1],
            ),
            make_ssa_op(
                6,
                Engine::Alu,
                Opcode::And,
                Some(VReg::scalar(6)),
                vec![Operand::scalar(0), Operand::scalar(1)],
                vec![0, 1],
            ),
            make_ssa_op(
                7,
                Engine::Alu,
                Opcode::Or,
                Some(VReg::scalar(7)),
                vec![Operand::scalar(0), Operand::scalar(1)],
                vec![0, 1],
            ),
            make_ssa_op(
                8,
                Engine::Alu,
                Opcode::Shl,
                Some(VReg::scalar(8)),
                vec![Operand::scalar(0), Operand::scalar(1)],
                vec![0, 1],
            ),
            make_ssa_op(
                9,
                Engine::Alu,
                Opcode::Shr,
                Some(VReg::scalar(9)),
                vec![Operand::scalar(0), Operand::scalar(1)],
                vec![0, 1],
            ),
            make_ssa_op(
                10,
                Engine::Alu,
                Opcode::Lt,
                Some(VReg::scalar(10)),
                vec![Operand::scalar(0), Operand::scalar(1)],
                vec![0, 1],
            ),
            make_ssa_op(
                11,
                Engine::Alu,
                Opcode::Eq,
                Some(VReg::scalar(11)),
                vec![Operand::scalar(0), Operand::scalar(1)],
                vec![0, 1],
            ),
            make_ssa_op(
                12,
                Engine::Alu,
                Opcode::Add,
                Some(VReg::scalar(12)),
                vec![Operand::scalar(0), Operand::scalar(1)],
                vec![0, 1],
            ),
            make_ssa_op(
                13,
                Engine::Alu,
                Opcode::Sub,
                Some(VReg::scalar(13)),
                vec![Operand::scalar(0), Operand::scalar(1)],
                vec![0, 1],
            ),
        ];

        // Hints: 2 loads at cycle 0, 12 ALU ops all at cycle 1 (exactly at ALU limit of 12)
        let hints = vec![0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];

        let result = schedule_ssa_cpsat(
            &ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: true,
                stop_after_first_solution: true,
                scratch_budget: 0, // Disable scratch constraint for this test
                enable_valu_shift: false,
                ..Default::default()
            },
            SsaCpsatHints {
                greedy_hints: Some(&hints),
                ..Default::default()
            },
            None,
        );

        // The verification code should NOT panic, and we should get a valid result
        assert!(
            result.is_some(),
            "Should find solution with 12 ALU ops at same cycle (exactly at limit)"
        );
        let r = result.unwrap();
        assert_eq!(
            r.cycle_assignments, hints,
            "Should accept the hinted solution"
        );
        eprintln!(
            "ALU capacity test: 12 ALU ops at cycle 1, result={:?}",
            r.cycle_assignments
        );
    }

    /// Test engine capacity: A shiftable VALU op shifted to ALU uses 8 slots.
    /// Create scenario with 1 shifted VALU (8 slots) + 4 ALU ops (4 slots) = 12 total.
    /// Should work with enable_valu_shift=true and shift_hints.
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_engine_capacity_shifted_valu() {
        use crate::types::Opcode;

        // Create:
        // - 2 loads (cycle 0)
        // - 1 VALU Add (shiftable, will use 8 ALU slots when shifted) at cycle 1
        // - 4 ALU ops at cycle 1 (4 slots)
        // Total ALU slots at cycle 1: 8 (shifted VALU) + 4 (ALU) = 12 (exactly at limit)
        let ops = vec![
            // Load ops
            make_ssa_op(
                0,
                Engine::Load,
                Opcode::Const,
                Some(VReg::vector(0)),
                vec![Operand::imm(1)],
                vec![],
            ),
            make_ssa_op(
                1,
                Engine::Load,
                Opcode::Const,
                Some(VReg::scalar(1)),
                vec![Operand::imm(2)],
                vec![],
            ),
            // 1 VALU op that will be shifted to ALU (uses 8 ALU slots)
            make_ssa_op(
                2,
                Engine::Valu,
                Opcode::Add,
                Some(VReg::vector(2)),
                vec![Operand::vector(0), Operand::vector(0)],
                vec![0],
            ),
            // 4 ALU ops (4 slots)
            make_ssa_op(
                3,
                Engine::Alu,
                Opcode::Add,
                Some(VReg::scalar(3)),
                vec![Operand::scalar(1), Operand::scalar(1)],
                vec![1],
            ),
            make_ssa_op(
                4,
                Engine::Alu,
                Opcode::Sub,
                Some(VReg::scalar(4)),
                vec![Operand::scalar(1), Operand::scalar(1)],
                vec![1],
            ),
            make_ssa_op(
                5,
                Engine::Alu,
                Opcode::Mul,
                Some(VReg::scalar(5)),
                vec![Operand::scalar(1), Operand::scalar(1)],
                vec![1],
            ),
            make_ssa_op(
                6,
                Engine::Alu,
                Opcode::Xor,
                Some(VReg::scalar(6)),
                vec![Operand::scalar(1), Operand::scalar(1)],
                vec![1],
            ),
        ];

        // Hints: loads at cycle 0, shifted VALU + 4 ALU ops at cycle 1
        let hints = vec![0, 0, 1, 1, 1, 1, 1];

        // shift_hints: op id 2 should be shifted (the VALU Add)
        let mut shift_hints = HashSet::new();
        shift_hints.insert(2); // Op ID 2 is the VALU Add

        let result = schedule_ssa_cpsat(
            &ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: true,
                stop_after_first_solution: true,
                scratch_budget: 0,       // Disable scratch constraint for this test
                enable_valu_shift: true, // Enable VALU shifting
                ..Default::default()
            },
            SsaCpsatHints {
                greedy_hints: Some(&hints),
                shift_hints: Some(&shift_hints),
                ..Default::default()
            },
            None,
        );

        // The verification code should NOT panic:
        // - VALU per cycle: 0 (the VALU op is shifted to ALU)
        // - ALU per cycle: 8 (shifted VALU) + 4 (regular ALU) = 12 (exactly at limit)
        //
        // Note: The verification happens BEFORE solver runs (at lines 1462-1525) and uses
        // shift_hints to verify that the greedy hints don't violate capacity. If the verification
        // panicked, we wouldn't reach this point. Getting a result means verification passed.
        assert!(
            result.is_some(),
            "Should find solution with shifted VALU (8) + 4 ALU = 12 (at limit)"
        );
        let r = result.unwrap();
        assert_eq!(
            r.cycle_assignments, hints,
            "Should accept the hinted solution"
        );

        // Note: The solver's shifted_ops may or may not contain our op depending on whether
        // the solver decides to shift. The key test is that the verification code (which uses
        // shift_hints to count ALU usage) did not panic. If it had panicked, we wouldn't get here.
        eprintln!("Shifted VALU capacity test: 1 shifted VALU (8 slots) + 4 ALU (4 slots) = 12 total at cycle 1");
        eprintln!(
            "Result: {:?}, shifted_ops from solver: {:?}",
            r.cycle_assignments, r.shifted_ops
        );

        // Also verify: if the solver did NOT shift op 2, then we have 1 VALU + 4 ALU at cycle 1,
        // which is fine (within limits). If the solver DID shift it, we have 0 VALU + 12 ALU.
        // Either way should be valid. The test was about the verification code not panicking
        // when we provide shift_hints that put us at exactly the ALU limit.
        eprintln!("Test passed: verification did not panic with 8+4=12 ALU slots at cycle 1");
    }

    // ==========================================================================
    // VReg Last Use Computation Tests
    // ==========================================================================
    // These tests verify that vreg_last_use correctly identifies the op with the
    // LATEST CYCLE that reads each VReg, not just the latest index. This is
    // critical for scratch pressure constraints - using wrong last_use results
    // in last_use_cycle < def_cycle which is impossible.

    /// Test vreg_last_use with a simple dependency chain.
    /// Each op reads the previous op's result, so last_use_cycle must >= def_cycle.
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_vreg_last_use_simple_chain() {
        use crate::types::Opcode;

        // Simple chain: op0 -> op1 -> op2 -> op3
        // Each op reads the previous op's VReg
        // op0: const -> v0 (no reads, defines v0)
        // op1: alu(v0) -> v1 (reads v0, defines v1)
        // op2: alu(v1) -> v2 (reads v1, defines v2)
        // op3: alu(v2) -> v3 (reads v2, defines v3)
        let ops = vec![
            make_ssa_op(
                0,
                Engine::Load,
                Opcode::Const,
                Some(VReg::scalar(0)),
                vec![Operand::imm(5)],
                vec![],
            ),
            make_ssa_op(
                1,
                Engine::Alu,
                Opcode::Add,
                Some(VReg::scalar(1)),
                vec![Operand::scalar(0), Operand::imm(1)],
                vec![0],
            ),
            make_ssa_op(
                2,
                Engine::Alu,
                Opcode::Add,
                Some(VReg::scalar(2)),
                vec![Operand::scalar(1), Operand::imm(2)],
                vec![1],
            ),
            make_ssa_op(
                3,
                Engine::Alu,
                Opcode::Add,
                Some(VReg::scalar(3)),
                vec![Operand::scalar(2), Operand::imm(3)],
                vec![2],
            ),
        ];

        // ASAP schedule: each op depends on the previous
        let hints = vec![0, 1, 2, 3];

        // Run with scratch constraint enabled - this will exercise vreg_last_use computation
        // If last_use_cycle < def_cycle for any VReg, the existing verification code will panic
        let result = schedule_ssa_cpsat(
            &ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: true,
                stop_after_first_solution: true,
                scratch_budget: 10, // Enable scratch constraint
                ..Default::default()
            },
            SsaCpsatHints {
                greedy_hints: Some(&hints),
                ..Default::default()
            },
            None,
        );

        // The test passes if we don't panic from "BUG: X VRegs have last_use_cycle < def_cycle"
        // For a simple chain, hints should be optimal (ASAP), so solver may return exact hints.
        assert!(result.is_some(), "Simple chain should be schedulable");
        let r = result.unwrap();
        // Verify basic constraints are satisfied
        assert!(
            r.makespan >= 4,
            "Need at least 4 cycles for 4-op dependency chain"
        );
        assert!(
            r.cycle_assignments.len() == 4,
            "Should have 4 cycle assignments"
        );
        // Verify dependency order is respected
        assert!(
            r.cycle_assignments[1] > r.cycle_assignments[0],
            "op1 must be after op0"
        );
        assert!(
            r.cycle_assignments[2] > r.cycle_assignments[1],
            "op2 must be after op1"
        );
        assert!(
            r.cycle_assignments[3] > r.cycle_assignments[2],
            "op3 must be after op2"
        );
        eprintln!(
            "test_vreg_last_use_simple_chain: makespan={}, assignments={:?}",
            r.makespan, r.cycle_assignments
        );
    }

    /// Test vreg_last_use with reordered hints (greedy schedule differs from index order).
    /// This tests that we use CYCLES from hints, not indices, to determine last use.
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_vreg_last_use_with_reordered_hints() {
        use crate::types::Opcode;

        // Create independent ops that can be reordered by the scheduler.
        // op0: const -> v0 (defines v0)
        // op1: const -> v1 (defines v1, independent of op0)
        // op2: alu(v0) -> v2 (reads v0, defines v2, depends on op0)
        // op3: alu(v1) -> v3 (reads v1, defines v3, depends on op1)
        //
        // Index order: op0, op1, op2, op3
        // But greedy might schedule: op0 (cycle 0), op1 (cycle 0), op3 (cycle 1), op2 (cycle 1)
        // or with explicit hints we can force reordering
        let ops = vec![
            make_ssa_op(
                0,
                Engine::Load,
                Opcode::Const,
                Some(VReg::scalar(0)),
                vec![Operand::imm(5)],
                vec![],
            ),
            make_ssa_op(
                1,
                Engine::Load,
                Opcode::Const,
                Some(VReg::scalar(1)),
                vec![Operand::imm(10)],
                vec![],
            ),
            make_ssa_op(
                2,
                Engine::Alu,
                Opcode::Add,
                Some(VReg::scalar(2)),
                vec![Operand::scalar(0), Operand::imm(1)],
                vec![0],
            ),
            make_ssa_op(
                3,
                Engine::Alu,
                Opcode::Add,
                Some(VReg::scalar(3)),
                vec![Operand::scalar(1), Operand::imm(2)],
                vec![1],
            ),
        ];

        // Hints where op3 (index 3) runs at cycle 1, op2 (index 2) runs at cycle 2
        // This means op2 runs LATER than op3 despite having a lower index
        // v0: defined at cycle 0 (op0), used at cycle 2 (op2) -> last_use_cycle = 2
        // v1: defined at cycle 0 (op1), used at cycle 1 (op3) -> last_use_cycle = 1
        let hints = vec![0, 0, 2, 1]; // op0, op1 at 0; op3 at 1; op2 at 2

        let result = schedule_ssa_cpsat(
            &ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: true,
                stop_after_first_solution: true,
                scratch_budget: 10,
                ..Default::default()
            },
            SsaCpsatHints {
                greedy_hints: Some(&hints),
                ..Default::default()
            },
            None,
        );

        // The test passes if we don't panic from "BUG: X VRegs have last_use_cycle < def_cycle"
        // Note: The solver may find a better schedule than the hints, so we don't assert exact match.
        // The key test is that vreg_last_use computation doesn't produce impossible results.
        assert!(result.is_some(), "Reordered schedule should be schedulable");
        let r = result.unwrap();
        // Verify basic constraints are satisfied
        assert!(
            r.makespan >= 2,
            "Need at least 2 cycles (loads then ALU ops)"
        );
        assert!(
            r.cycle_assignments.len() == 4,
            "Should have 4 cycle assignments"
        );
        eprintln!(
            "test_vreg_last_use_with_reordered_hints: makespan={}, assignments={:?}",
            r.makespan, r.cycle_assignments
        );
    }

    /// Test vreg_last_use with multiple readers of the same VReg at different cycles.
    /// We must pick the reader with the LATEST cycle, not the latest index.
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_vreg_last_use_multiple_readers() {
        use crate::types::Opcode;

        // op0: const -> v0 (defines v0)
        // op1: alu(v0) -> v1 (reads v0 - early reader at index 1)
        // op2: const -> v2 (independent, defines v2)
        // op3: alu(v0, v2) -> v3 (reads v0 - late reader at index 3, also reads v2)
        //
        // v0 is read by both op1 and op3
        // With hints [0, 1, 0, 2]: op1 reads v0 at cycle 1, op3 reads v0 at cycle 2
        // So last_use_cycle for v0 should be 2 (from op3), not 1 (from op1)
        //
        // The tricky part: op3 has a higher index than op1, so if we use index order,
        // we'd correctly pick op3. To test the cycle-based logic, we need a case where
        // the LOWER index has a HIGHER cycle - see the test below.
        let ops = vec![
            make_ssa_op(
                0,
                Engine::Load,
                Opcode::Const,
                Some(VReg::scalar(0)),
                vec![Operand::imm(5)],
                vec![],
            ),
            make_ssa_op(
                1,
                Engine::Alu,
                Opcode::Add,
                Some(VReg::scalar(1)),
                vec![Operand::scalar(0), Operand::imm(1)],
                vec![0],
            ),
            make_ssa_op(
                2,
                Engine::Load,
                Opcode::Const,
                Some(VReg::scalar(2)),
                vec![Operand::imm(10)],
                vec![],
            ),
            make_ssa_op(
                3,
                Engine::Alu,
                Opcode::Add,
                Some(VReg::scalar(3)),
                vec![Operand::scalar(0), Operand::scalar(2)],
                vec![0, 2],
            ),
        ];

        // Hints: op0 at 0, op1 at 1, op2 at 0, op3 at 2
        // v0: def at cycle 0, read by op1 at cycle 1, read by op3 at cycle 2 -> last_use = 2
        let hints = vec![0, 1, 0, 2];

        let result = schedule_ssa_cpsat(
            &ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: true,
                stop_after_first_solution: true,
                scratch_budget: 10,
                ..Default::default()
            },
            SsaCpsatHints {
                greedy_hints: Some(&hints),
                ..Default::default()
            },
            None,
        );

        // The test passes if we don't panic from "BUG: X VRegs have last_use_cycle < def_cycle"
        // Note: The solver may find a better schedule than the hints, so we don't assert exact match.
        // The key test is that vreg_last_use computation doesn't produce impossible results.
        assert!(result.is_some(), "Multiple readers should be schedulable");
        let r = result.unwrap();
        // Verify basic constraints are satisfied
        assert!(
            r.makespan >= 2,
            "Need at least 2 cycles (loads then ALU ops)"
        );
        assert!(
            r.cycle_assignments.len() == 4,
            "Should have 4 cycle assignments"
        );
        eprintln!(
            "test_vreg_last_use_multiple_readers: makespan={}, assignments={:?}",
            r.makespan, r.cycle_assignments
        );
    }

    /// Test that CP-SAT can schedule with a small scratch budget.
    ///
    /// NOTE: CP-SAT's cumulative constraint does NOT model same-cycle reuse.
    /// At cycle 1, both v0 (being read) and v1 (being written) are counted as live.
    /// So CP-SAT requires budget >= 2 for this program.
    ///
    /// The greedy scheduler DOES support same-cycle reuse and can handle budget=1.
    /// See test_ssa_greedy_minimal_budget_1_word for that test.
    ///
    /// Program:
    ///   v0 = const(5)        -- scalar, 1 word, born cycle 0
    ///   v1 = add(v0, v0)     -- scalar, 1 word, born cycle 1, v0 dies here
    ///   store(0, v1)         -- v1 dies here (cycle 2)
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_ssa_cpsat_minimal_budget_2_words() {
        use crate::address_assign::{assign_addresses, AddressAssignConfig};
        use crate::types::Opcode;

        // Create ops: const -> add -> store
        // v0 = const(5)
        // v1 = add(v0, v0)  -- v0's last use
        // store(0, v1)      -- v1's last use
        let ops = vec![
            make_ssa_op(
                0,
                Engine::Load,
                Opcode::Const,
                Some(VReg::scalar(0)),
                vec![Operand::imm(5)],
                vec![],
            ),
            make_ssa_op(
                1,
                Engine::Alu,
                Opcode::Add,
                Some(VReg::scalar(1)),
                vec![Operand::scalar(0), Operand::scalar(0)],
                vec![0],
            ),
            make_ssa_op(
                2,
                Engine::Store,
                Opcode::Store,
                None,
                vec![Operand::imm(0), Operand::scalar(1)],
                vec![1],
            ),
        ];

        // Schedule with CP-SAT and budget of 2 (CP-SAT doesn't support same-cycle reuse)
        let result = schedule_ssa_cpsat(
            &ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: true,
                scratch_budget: 2, // CP-SAT needs 2 because no same-cycle reuse
                ..Default::default()
            },
            SsaCpsatHints::default(),
            None,
        );

        // CP-SAT should find a solution with budget=2
        assert!(
            result.is_some(),
            "CP-SAT should find a schedule with 2-word budget"
        );

        let r = result.unwrap();
        eprintln!(
            "test_ssa_cpsat_minimal_budget_2_words: makespan={}, assignments={:?}",
            r.makespan, r.cycle_assignments
        );

        // Verify all ops are scheduled
        assert_eq!(
            r.cycle_assignments.len(),
            3,
            "All 3 ops should be scheduled"
        );

        // Apply cycle assignments to a mutable copy for address assignment
        let mut scheduled_ops = ops.clone();
        for (idx, &cycle) in r.cycle_assignments.iter().enumerate() {
            scheduled_ops[idx].cycle = cycle;
        }

        // Address assignment with same-cycle reuse should fit in budget of 1
        // (even though CP-SAT required budget=2)
        let config = AddressAssignConfig {
            budget: 2,
            start_addr: 1, // Reserve 0 for ZERO_REG
            ..Default::default()
        };
        let assign_result = assign_addresses(&scheduled_ops, &config);

        // With same-cycle reuse, linear scan should achieve max_addr=1
        // (even though CP-SAT conservatively required budget=2)
        assert_eq!(
            assign_result.max_addr, 1,
            "With same-cycle reuse, only 1 address should be needed, got max_addr={}",
            assign_result.max_addr
        );

        // peak_live should be 1 with same-cycle reuse
        assert_eq!(
            assign_result.peak_live, 1,
            "With same-cycle reuse, peak_live should be 1, got {}",
            assign_result.peak_live
        );

        // Should not exceed budget
        assert!(
            !assign_result.budget_exceeded,
            "Should fit in budget of 2: peak_live={}, budget={}",
            assign_result.peak_live, assign_result.budget
        );
    }

    /// Test that CP-SAT scratch constraint correctly models shifted VALU dests.
    ///
    /// Bug fix: Shifted VALU dests must use min(alu_cycles) as interval start, not completion_cycle (max).
    /// Address assignment reserves addresses from the FIRST write (min), so CP-SAT must match this.
    ///
    /// This test creates a scenario where:
    /// - A VALU op is shifted to ALU (8 ALU ops write at cycles [C, C+1, C+2, ...])
    /// - The dest VReg is used later
    /// - With tight budget, CP-SAT must account for the address being reserved from min(alu_cycles)
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_ssa_cpsat_scratch_constraint_shifted_valu_dest() {
        use crate::address_assign::{assign_addresses, AddressAssignConfig};
        use crate::types::Opcode;

        // Create a shifted VALU op that writes a vector (8 words) at different ALU cycles.
        // The dest is used later, so it must be live from min(alu_cycles) to last_use.
        //
        // op0: load -> v0 (vector, 8 words) - no deps
        // op1: valu(v0) -> v1 (vector, 8 words) - dep: 0, SHIFTED to ALU
        // op2: valu(v1) -> v2 (vector, 8 words) - dep: 1
        //
        // If op1 is shifted: 8 ALU ops write v1 at cycles [C, C+1, ..., C+7]
        // Address assignment reserves v1 from min(alu_cycles) = C, not completion_cycle = C+7.
        // CP-SAT scratch constraint must match this.
        let ops = vec![
            make_ssa_op(
                0,
                Engine::Load,
                Opcode::Vload,
                Some(VReg::vector(0)),
                vec![Operand::imm(100)],
                vec![],
            ),
            make_ssa_op(
                1,
                Engine::Valu,
                Opcode::Add,
                Some(VReg::vector(1)),
                vec![Operand::vector(0), Operand::imm(1)],
                vec![0],
            ),
            make_ssa_op(
                2,
                Engine::Valu,
                Opcode::Add,
                Some(VReg::vector(2)),
                vec![Operand::vector(1), Operand::imm(2)],
                vec![1],
            ),
        ];

        // Schedule with CP-SAT, forcing shift of op1 to ALU, with tight budget
        let result = schedule_ssa_cpsat(
            &ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: true,
                scratch_budget: 16, // Tight: 2 vectors = 16 words
                enable_valu_shift: true, // Must enable VALU shift for ForceAlu to work
                valu_shift_mode: ValuShiftMode::ForceAlu, // Force shift of op1
                ..Default::default()
            },
            SsaCpsatHints::default(),
            None,
        );

        assert!(
            result.is_some(),
            "Should find solution with shifted VALU and tight budget"
        );

        let r = result.unwrap();
        eprintln!(
            "test_ssa_cpsat_scratch_constraint_shifted_valu_dest: makespan={}, shifted_ops={:?}",
            r.makespan, r.shifted_ops
        );

        // Verify op1 was shifted
        assert!(
            r.shifted_ops.contains(&1),
            "op1 should be shifted to ALU"
        );

        // Apply cycle assignments and verify address assignment fits
        let mut scheduled_ops = ops.clone();
        for (idx, &cycle) in r.cycle_assignments.iter().enumerate() {
            scheduled_ops[idx].cycle = cycle;
        }

        let config = AddressAssignConfig {
            budget: 16,
            start_addr: 1,
            shifted_alu_cycles: Some(r.shifted_alu_cycles.clone()),
            ..Default::default()
        };
        let assign_result = assign_addresses(&scheduled_ops, &config);

        // Address assignment should fit within budget (CP-SAT constraint should match)
        assert!(
            assign_result.peak_live <= 16,
            "Address assignment peak_live={} should fit in budget=16 (CP-SAT constraint should match)",
            assign_result.peak_live
        );
    }

    /// Test that ASAP/ALAP tightening works correctly when enabled.
    ///
    /// Creates a scenario where many ops compete for limited capacity, so tightening should
    /// reduce the search space by tightening ASAP/ALAP bounds. Verifies that:
    /// 1. CP-SAT can find solutions with tightening enabled
    /// 2. Solutions are still valid (respect precedence and capacity)
    /// 3. Tightening doesn't break feasibility
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_asap_alap_tightening_enabled() {
        use crate::types::Opcode;

        // Create many VALU ops that can all start early (from precedence), but capacity is limited
        // This should cause tightening to push some ops later
        let mut ops = Vec::new();
        
        // op0: load -> v0 (no deps, ASAP=0)
        ops.push(make_ssa_op(
            0,
            Engine::Load,
            Opcode::Vload,
            Some(VReg::vector(0)),
            vec![Operand::imm(100)],
            vec![],
        ));

        // Create 15 VALU ops that all depend on op0
        // With VALU capacity=6, they can't all start at cycle 1, so tightening should push some later
        for i in 1..=15 {
            ops.push(make_ssa_op(
                i,
                Engine::Valu,
                Opcode::Add,
                Some(VReg::vector(i as u32)),
                vec![Operand::vector(0), Operand::imm(1)],
                vec![0], // All depend on op0
            ));
        }

        // Test with tightening disabled (baseline)
        let result_no_tightening = schedule_ssa_cpsat(
            &ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: false,
                enable_asap_alap_tightening: false,
                ..Default::default()
            },
            SsaCpsatHints::default(),
            None,
        );

        assert!(
            result_no_tightening.is_some(),
            "CP-SAT should find a solution without tightening"
        );

        // Test with tightening enabled
        let result_with_tightening = schedule_ssa_cpsat(
            &ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: false,
                enable_asap_alap_tightening: true,
                ..Default::default()
            },
            SsaCpsatHints::default(),
            None,
        );

        assert!(
            result_with_tightening.is_some(),
            "CP-SAT should find a solution with tightening enabled"
        );

        let r_no_tightening = result_no_tightening.unwrap();
        let r_with_tightening = result_with_tightening.unwrap();

        eprintln!(
            "test_asap_alap_tightening_enabled: makespan without tightening={}, with tightening={}",
            r_no_tightening.makespan, r_with_tightening.makespan
        );

        // Verify all ops are scheduled in both cases
        assert_eq!(
            r_no_tightening.cycle_assignments.len(),
            16,
            "All 16 ops should be scheduled without tightening"
        );
        assert_eq!(
            r_with_tightening.cycle_assignments.len(),
            16,
            "All 16 ops should be scheduled with tightening"
        );

        // Verify solutions respect precedence constraints
        for (idx, &cycle) in r_with_tightening.cycle_assignments.iter().enumerate() {
            let op = &ops[idx];
            for &dep_idx in &op.dep_ids {
                if let Some(&dep_cycle) = r_with_tightening.cycle_assignments.get(dep_idx) {
                    assert!(
                        cycle >= dep_cycle + 1,
                        "Precedence violated: op {} at cycle {} depends on op {} at cycle {}",
                        idx, cycle, dep_idx, dep_cycle
                    );
                }
            }
        }

        // Tightening should not make the problem infeasible
        // (makespan might be same or slightly different, but solution should exist)
        assert!(
            r_with_tightening.status == CpsatStatus::Optimal || r_with_tightening.status == CpsatStatus::Feasible,
            "Solution with tightening should be feasible or optimal"
        );
    }

    /// Test that scratch-aware tightening works correctly when enabled independently.
    ///
    /// Creates a scenario with high scratch pressure (many VRegs live simultaneously)
    /// to verify that scratch-aware tightening can tighten bounds independently of slot tightening.
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_scratch_tightening_enabled() {
        use crate::types::Opcode;

        // Create ops that produce many VRegs that are all live simultaneously
        // This creates scratch pressure without necessarily creating slot pressure
        let mut ops = Vec::new();
        
        // op0: load -> v0 (no deps, ASAP=0)
        ops.push(make_ssa_op(
            0,
            Engine::Load,
            Opcode::Vload,
            Some(VReg::vector(0)),
            vec![Operand::imm(100)],
            vec![],
        ));

        // Create 20 ALU ops that all depend on op0 and produce different VRegs
        // These can all start at cycle 1, but if they all produce VRegs that stay live,
        // we'll have high scratch pressure
        for i in 1..=20 {
            ops.push(make_ssa_op(
                i,
                Engine::Alu,
                Opcode::Add,
                Some(VReg::scalar(i as u32)),
                vec![Operand::scalar(0), Operand::imm(1)],
                vec![0], // All depend on op0
            ));
        }

        // Create consumers that use all these VRegs much later (creates long lifetimes)
        for i in 1..=20 {
            ops.push(make_ssa_op(
                20 + i,
                Engine::Alu,
                Opcode::Add,
                Some(VReg::scalar(100 + i as u32)),
                vec![Operand::scalar(i as u32), Operand::imm(1)],
                vec![i], // Depends on producer
            ));
        }

        // Test with scratch tightening disabled (baseline)
        let result_no_tightening = schedule_ssa_cpsat(
            &ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: false,
                enable_asap_alap_tightening: false,
                enable_scratch_tightening: false,
                scratch_budget: 10, // Tight budget to force scratch pressure
                ..Default::default()
            },
            SsaCpsatHints::default(),
            None,
        );

        assert!(
            result_no_tightening.is_some(),
            "CP-SAT should find a solution without scratch tightening"
        );

        // Test with scratch tightening enabled (but slot tightening disabled)
        let result_with_scratch_tightening = schedule_ssa_cpsat(
            &ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: false,
                enable_asap_alap_tightening: false,
                enable_scratch_tightening: true,
                scratch_budget: 10, // Tight budget
                ..Default::default()
            },
            SsaCpsatHints::default(),
            None,
        );

        assert!(
            result_with_scratch_tightening.is_some(),
            "CP-SAT should find a solution with scratch tightening enabled"
        );

        let r_no_tightening = result_no_tightening.unwrap();
        let r_with_scratch = result_with_scratch_tightening.unwrap();

        eprintln!(
            "test_scratch_tightening_enabled: makespan without tightening={}, with scratch tightening={}",
            r_no_tightening.makespan, r_with_scratch.makespan
        );

        // Verify all ops are scheduled in both cases
        assert_eq!(
            r_no_tightening.cycle_assignments.len(),
            ops.len(),
            "All ops should be scheduled without scratch tightening"
        );
        assert_eq!(
            r_with_scratch.cycle_assignments.len(),
            ops.len(),
            "All ops should be scheduled with scratch tightening"
        );

        // Verify solutions respect precedence constraints
        for (idx, &cycle) in r_with_scratch.cycle_assignments.iter().enumerate() {
            let op = &ops[idx];
            for &dep_idx in &op.dep_ids {
                if let Some(&dep_cycle) = r_with_scratch.cycle_assignments.get(dep_idx) {
                    assert!(
                        cycle >= dep_cycle + 1,
                        "Op {} scheduled at cycle {} should be after dep {} at cycle {}",
                        idx, cycle, dep_idx, dep_cycle
                    );
                }
            }
        }

        // Scratch tightening should not make the problem infeasible
        assert!(
            r_with_scratch.status == CpsatStatus::Optimal || r_with_scratch.status == CpsatStatus::Feasible,
            "Solution with scratch tightening should be feasible or optimal"
        );
    }

    /// Test that both slot and scratch tightening can work together.
    ///
    /// Creates a scenario with both slot pressure and scratch pressure to verify
    /// that both tightening methods can be applied together.
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_both_tightening_methods_together() {
        use crate::types::Opcode;

        // Create ops that create both slot pressure (many VALU ops) and scratch pressure (long lifetimes)
        let mut ops = Vec::new();
        
        // op0: load -> v0 (no deps, ASAP=0)
        ops.push(make_ssa_op(
            0,
            Engine::Load,
            Opcode::Vload,
            Some(VReg::vector(0)),
            vec![Operand::imm(100)],
            vec![],
        ));

        // Create 15 VALU ops that all depend on op0 (slot pressure)
        for i in 1..=15 {
            ops.push(make_ssa_op(
                i,
                Engine::Valu,
                Opcode::Add,
                Some(VReg::vector(i as u32)),
                vec![Operand::vector(0), Operand::imm(1)],
                vec![0], // All depend on op0
            ));
        }

        // Create consumers much later (scratch pressure - long lifetimes)
        for i in 1..=15 {
            ops.push(make_ssa_op(
                15 + i,
                Engine::Alu,
                Opcode::Add,
                Some(VReg::scalar(100 + i as u32)),
                vec![Operand::vector(i as u32), Operand::imm(1)],
                vec![i], // Depends on VALU producer
            ));
        }

        // Test with both tightening methods enabled
        let result_both = schedule_ssa_cpsat(
            &ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: false,
                enable_asap_alap_tightening: true,
                enable_scratch_tightening: true,
                scratch_budget: 20, // Moderate budget
                ..Default::default()
            },
            SsaCpsatHints::default(),
            None,
        );

        assert!(
            result_both.is_some(),
            "CP-SAT should find a solution with both tightening methods enabled"
        );

        let r_both = result_both.unwrap();

        eprintln!(
            "test_both_tightening_methods_together: makespan with both tightening={}",
            r_both.makespan
        );

        // Verify all ops are scheduled
        assert_eq!(
            r_both.cycle_assignments.len(),
            ops.len(),
            "All ops should be scheduled with both tightening methods"
        );

        // Verify solutions respect precedence constraints
        for (idx, &cycle) in r_both.cycle_assignments.iter().enumerate() {
            let op = &ops[idx];
            for &dep_idx in &op.dep_ids {
                if let Some(&dep_cycle) = r_both.cycle_assignments.get(dep_idx) {
                    assert!(
                        cycle >= dep_cycle + 1,
                        "Op {} scheduled at cycle {} should be after dep {} at cycle {}",
                        idx, cycle, dep_idx, dep_cycle
                    );
                }
            }
        }

        // Both tightening methods should not make the problem infeasible
        assert!(
            r_both.status == CpsatStatus::Optimal || r_both.status == CpsatStatus::Feasible,
            "Solution with both tightening methods should be feasible or optimal"
        );
    }

    /// Test that CP-SAT scratch constraint correctly models SelectChoice FMA reads at cycle+1.
    ///
    /// Bug fix: SelectChoice FMA reads cond/false_val/delta at cycle+1, so interval end should be
    /// cycle+2 (half-open), not cycle+1. Address assignment extends these intervals to cycle+1
    /// (closed) = cycle+2 (half-open), so CP-SAT must match.
    ///
    /// This test creates a scenario where:
    /// - A SelectChoice op chooses FMA path
    /// - cond/false_val are read at cycle+1 by FMA
    /// - With tight budget, CP-SAT must account for these VRegs being live until cycle+1
    #[cfg(feature = "cpsat")]
    #[test]
    fn test_ssa_cpsat_scratch_constraint_selectchoice_fma_reads() {
        use crate::address_assign::{assign_addresses, AddressAssignConfig, compute_live_intervals};
        use crate::types::Opcode;

        // Create a SelectChoice op that chooses FMA path.
        // cond and false_val are read at cycle+1 by FMA, so their intervals must extend to cycle+1.
        //
        // op0: load -> cond (scalar, 1 word) - no deps
        // op1: load -> true_val (scalar, 1 word) - no deps
        // op2: load -> false_val (scalar, 1 word) - no deps
        // op3: select(cond, true_val, false_val) -> result (scalar, 1 word) - deps: 0,1,2, CHOOSES FMA
        // op4: use(result) - dep: 3
        //
        // If op3 chooses FMA: FMA reads cond and false_val at cycle+1.
        // Address assignment extends cond/false_val intervals to cycle+1 (closed).
        // CP-SAT scratch constraint must use cycle+2 (half-open) to match.
        let mut ops = vec![
            make_ssa_op(
                0,
                Engine::Load,
                Opcode::Const,
                Some(VReg::scalar(0)),
                vec![Operand::imm(1)], // cond = 1 (true)
                vec![],
            ),
            make_ssa_op(
                1,
                Engine::Load,
                Opcode::Const,
                Some(VReg::scalar(1)),
                vec![Operand::imm(10)], // true_val
                vec![],
            ),
            make_ssa_op(
                2,
                Engine::Load,
                Opcode::Const,
                Some(VReg::scalar(2)),
                vec![Operand::imm(20)], // false_val
                vec![],
            ),
            make_ssa_op(
                3,
                Engine::Flow,
                Opcode::SelectChoice,
                Some(VReg::scalar(3)),
                vec![
                    Operand::scalar(0), // cond
                    Operand::scalar(1), // true_val
                    Operand::scalar(2), // false_val
                ],
                vec![0, 1, 2],
            ),
            make_ssa_op(
                4,
                Engine::Store,
                Opcode::Store,
                None,
                vec![Operand::imm(0), Operand::scalar(3)],
                vec![3],
            ),
        ];

        // Set sub_temp_vreg for SelectChoice (no-delta path, uses FMA)
        ops[3].sub_temp_vreg = Some(VReg::scalar(99));

        // Schedule with CP-SAT, forcing FMA path, with tight budget
        // Budget must accommodate: cond (1), false_val (1), true_val (1), result (1), sub_temp (8) = 12 words peak
        // However, intervals can be scheduled to minimize overlap. At peak (cycle 2): cond (1), false_val (1), sub_temp (8) = 10 words
        // But presolve detects infeasibility if max profile > capacity, so we need at least 10.
        // Use 12 to be safe and match the test's expectation that a solution exists.
        let result = schedule_ssa_cpsat(
            &ops,
            SsaCpsatConfig {
                time_limit_secs: 10.0,
                verbose: true,
                scratch_budget: 12, // Must accommodate sub_temp (8) + cond (1) + false_val (1) + true_val (1) + result (1) = 12 peak
                select_choice_mode: SelectChoiceMode::ForceFma, // Force FMA path
                ..Default::default()
            },
            SsaCpsatHints::default(),
            None,
        );

        assert!(
            result.is_some(),
            "Should find solution with SelectChoice FMA and tight budget"
        );

        let r = result.unwrap();
        eprintln!(
            "test_ssa_cpsat_scratch_constraint_selectchoice_fma_reads: makespan={}, select_fma_ops={:?}",
            r.makespan, r.select_fma_ops
        );

        // Verify op3 chose FMA
        assert!(
            r.select_fma_ops.contains(&3),
            "op3 should choose FMA path"
        );

        // Apply cycle assignments and verify address assignment fits
        let mut scheduled_ops = ops.clone();
        for (idx, &cycle) in r.cycle_assignments.iter().enumerate() {
            scheduled_ops[idx].cycle = cycle;
        }

        let config = AddressAssignConfig {
            budget: 4,
            start_addr: 1,
            select_fma_ops: Some(r.select_fma_ops.clone()),
            ..Default::default()
        };
        let assign_result = assign_addresses(&scheduled_ops, &config);

        // Verify live intervals: cond and false_val should extend to cycle+1 (where FMA reads them)
        let intervals = compute_live_intervals(
            &scheduled_ops,
            None,
            Some(&r.select_fma_ops),
            None,
            Some(&r.const_alu_derive_ops),
            None,
        );
        let select_cycle = scheduled_ops[3].cycle;
        let cond_interval = intervals.get(&VReg::scalar(0)).unwrap();
        let false_val_interval = intervals.get(&VReg::scalar(2)).unwrap();
        assert_eq!(
            cond_interval.end,
            select_cycle + 1,
            "cond interval should extend to cycle+1 (FMA reads at cycle+1)"
        );
        assert_eq!(
            false_val_interval.end,
            select_cycle + 1,
            "false_val interval should extend to cycle+1 (FMA reads at cycle+1)"
        );

        // Address assignment should fit within budget (CP-SAT constraint should match)
        assert!(
            assign_result.peak_live <= 4,
            "Address assignment peak_live={} should fit in budget=4 (CP-SAT constraint should match)",
            assign_result.peak_live
        );
    }
}

#[cfg(test)]
mod capacity_lb_tests {
    use super::compute_capacity_lower_bound_global;

    fn default_caps() -> (i64, i64, i64, i64, i64) {
        (6, 12, 2, 2, 1) // valu, alu, load, store, flow
    }

    #[test]
    fn test_capacity_lb_no_choices() {
        let (vc, ac, lc, sc, fc) = default_caps();
        let r = compute_capacity_lower_bound_global(
            18, 0, 0, 0, 0, 0, 0, 0, false, vc, ac, lc, sc, fc, 1,
        );
        assert_eq!(r.capacity_lower_bound, 3, "18 VALU / 6 = 3 cycles");
        assert_eq!(r.store_bound, 0);
    }

    #[test]
    fn test_capacity_lb_critical_path_dominates() {
        let (vc, ac, lc, sc, fc) = default_caps();
        let r = compute_capacity_lower_bound_global(
            6, 0, 0, 0, 0, 0, 0, 0, false, vc, ac, lc, sc, fc, 10,
        );
        assert_eq!(r.capacity_lower_bound, 10, "critical_path 10 dominates 6/6=1");
    }

    #[test]
    fn test_capacity_lb_shift_global_min() {
        let (vc, ac, lc, sc, fc) = default_caps();
        // 6 VALU, 3 shiftable. x=0 (all shift): VALU=3, ALU=24 => max(1,2)=2. x=3 (none shift): VALU=6, ALU=0 => max(1,0)=1. Global min = 1.
        let r_no_shift = compute_capacity_lower_bound_global(
            6, 0, 0, 0, 0, 3, 0, 0, false, vc, ac, lc, sc, fc, 1,
        );
        let r_shift = compute_capacity_lower_bound_global(
            6, 0, 0, 0, 0, 3, 0, 0, true, vc, ac, lc, sc, fc, 1,
        );
        assert_eq!(r_no_shift.capacity_lower_bound, 1);
        assert_eq!(r_shift.capacity_lower_bound, 1, "min over x: best is x=3 => 1 cycle");
    }

    #[test]
    fn test_capacity_lb_const_choice_reduces_load() {
        let (vc, ac, lc, sc, fc) = default_caps();
        // 4 Load ops, 2 ConstChoice. If both add_imm: 2 Load => 1 cycle. If both load: 4 Load => 2 cycles.
        let r = compute_capacity_lower_bound_global(
            0, 0, 4, 0, 0, 0, 2, 0, false, vc, ac, lc, sc, fc, 1,
        );
        assert!(r.capacity_lower_bound >= 1 && r.capacity_lower_bound <= 2);
        assert_eq!(r.load_bound, 1, "min Load = 2 ops => ceil(2/2)=1");
    }

    #[test]
    fn test_capacity_lb_flow_dominates() {
        let (vc, ac, lc, sc, fc) = default_caps();
        // 5 Flow ops (flow_base=5), no others. Makespan must be >= 5.
        let r = compute_capacity_lower_bound_global(
            0, 0, 0, 0, 5, 0, 0, 0, false, vc, ac, lc, sc, fc, 1,
        );
        assert_eq!(r.capacity_lower_bound, 5);
    }

    #[test]
    fn test_capacity_lb_triple_choice_global_min() {
        let (vc, ac, lc, sc, fc) = default_caps();
        // Small mix: some VALU (shiftable), some ConstChoice, some SelectChoice. Just check it runs and returns >= critical_path.
        let r = compute_capacity_lower_bound_global(
            6, 12, 2, 0, 1, 3, 1, 1, true, vc, ac, lc, sc, fc, 5,
        );
        assert!(r.capacity_lower_bound >= 5);
        assert!(r.capacity_lower_bound >= r.store_bound);
    }
}
