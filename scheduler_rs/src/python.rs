//! Python interface for the scheduler.
//!
//! Contains all PyO3 bindings, parsing, and emission functions.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::str::FromStr;

use crate::constants::{
    analyze_constant_usage, build_constant_plan, emit_constant_generation_ops,
    print_constant_plan_summary,
};
use crate::diagnostics::{
    analyze_shared_temp_dependencies, SchedulingDiagnostics, SharedTempAnalysis,
};
use crate::schedule_output::{
    build_schedule_output_json, write_schedule_output_to_path, write_ssa_schedule_json,
    LivenessDetail,
};
use crate::scheduler::{
    compute_critical_path, compute_dependencies, empty_cycles, schedule_greedy,
    schedule_greedy_with_diagnostics, PriorityMode, ALL_PRIORITY_MODES,
};
use arrayvec::ArrayVec;

use crate::types::{Engine, Op, SlotVal, MAX_SLOT_SIZE, VLEN};

// =============================================================================
// Parsing: Python -> Rust
// =============================================================================

/// Parse ops from Python list of dicts
pub fn parse_ops(_py: Python<'_>, instrs: &Bound<'_, PyList>) -> PyResult<Vec<Op>> {
    let mut ops = Vec::new();
    let mut op_id = 0usize;

    for (cycle, instr) in instrs.iter().enumerate() {
        let instr_dict = instr.downcast::<PyDict>()?;

        for (engine_key, slots) in instr_dict.iter() {
            let engine: String = engine_key.extract()?;
            let slots_list = slots.downcast::<PyList>()?;

            for slot in slots_list.iter() {
                let slot_tuple = slot.downcast::<PyTuple>()?;
                let mut reads: HashSet<i64> = HashSet::new();
                let mut writes: HashSet<i64> = HashSet::new();

                // Parse reads/writes based on op type
                let op_type: String = slot_tuple.get_item(0)?.extract()?;

                match engine.as_str() {
                    "load" => match op_type.as_str() {
                        "const" => {
                            let dest: i64 = slot_tuple.get_item(1)?.extract()?;
                            writes.insert(dest);
                        }
                        "load" => {
                            let dest: i64 = slot_tuple.get_item(1)?.extract()?;
                            let src: i64 = slot_tuple.get_item(2)?.extract()?;
                            writes.insert(dest);
                            reads.insert(src);
                        }
                        "vload" => {
                            let dest: i64 = slot_tuple.get_item(1)?.extract()?;
                            let src: i64 = slot_tuple.get_item(2)?.extract()?;
                            for i in 0..VLEN as i64 {
                                writes.insert(dest + i);
                            }
                            reads.insert(src);
                        }
                        _ => {}
                    },
                    "store" => match op_type.as_str() {
                        "store" => {
                            let addr: i64 = slot_tuple.get_item(1)?.extract()?;
                            let data: i64 = slot_tuple.get_item(2)?.extract()?;
                            reads.insert(addr);
                            reads.insert(data);
                        }
                        "vstore" => {
                            let addr: i64 = slot_tuple.get_item(1)?.extract()?;
                            let data: i64 = slot_tuple.get_item(2)?.extract()?;
                            reads.insert(addr);
                            for i in 0..VLEN as i64 {
                                reads.insert(data + i);
                            }
                        }
                        _ => {}
                    },
                    "alu" => {
                        let dest: i64 = slot_tuple.get_item(1)?.extract()?;
                        let src1: i64 = slot_tuple.get_item(2)?.extract()?;
                        let src2: i64 = slot_tuple.get_item(3)?.extract()?;
                        writes.insert(dest);
                        reads.insert(src1);
                        reads.insert(src2);
                    }
                    "valu" => {
                        match op_type.as_str() {
                            "vbroadcast" => {
                                let dest: i64 = slot_tuple.get_item(1)?.extract()?;
                                let src: i64 = slot_tuple.get_item(2)?.extract()?;
                                for i in 0..VLEN as i64 {
                                    writes.insert(dest + i);
                                }
                                reads.insert(src);
                            }
                            "vmultiply_add" | "multiply_add" => {
                                let dest: i64 = slot_tuple.get_item(1)?.extract()?;
                                let src1: i64 = slot_tuple.get_item(2)?.extract()?;
                                let src2: i64 = slot_tuple.get_item(3)?.extract()?;
                                let src3: i64 = slot_tuple.get_item(4)?.extract()?;
                                for i in 0..VLEN as i64 {
                                    writes.insert(dest + i);
                                    reads.insert(src1 + i);
                                    reads.insert(src2 + i);
                                    reads.insert(src3 + i);
                                }
                            }
                            _ => {
                                // Binary vector ops
                                let dest: i64 = slot_tuple.get_item(1)?.extract()?;
                                let src1: i64 = slot_tuple.get_item(2)?.extract()?;
                                let src2: i64 = slot_tuple.get_item(3)?.extract()?;
                                for i in 0..VLEN as i64 {
                                    writes.insert(dest + i);
                                    reads.insert(src1 + i);
                                    reads.insert(src2 + i);
                                }
                            }
                        }
                    }
                    "flow" => match op_type.as_str() {
                        "vselect" => {
                            let dest: i64 = slot_tuple.get_item(1)?.extract()?;
                            let src1: i64 = slot_tuple.get_item(2)?.extract()?;
                            let src2: i64 = slot_tuple.get_item(3)?.extract()?;
                            let src3: i64 = slot_tuple.get_item(4)?.extract()?;
                            for i in 0..VLEN as i64 {
                                writes.insert(dest + i);
                                reads.insert(src1 + i);
                                reads.insert(src2 + i);
                                reads.insert(src3 + i);
                            }
                        }
                        "add_imm" => {
                            let dest: i64 = slot_tuple.get_item(1)?.extract()?;
                            let src: i64 = slot_tuple.get_item(2)?.extract()?;
                            writes.insert(dest);
                            reads.insert(src);
                        }
                        "cond_jump" => {
                            let cond: i64 = slot_tuple.get_item(1)?.extract()?;
                            reads.insert(cond);
                        }
                        _ => {}
                    },
                    "debug" => match op_type.as_str() {
                        "vcompare" => {
                            let loc: i64 = slot_tuple.get_item(1)?.extract()?;
                            for i in 0..VLEN as i64 {
                                reads.insert(loc + i);
                            }
                        }
                        "compare" => {
                            let loc: i64 = slot_tuple.get_item(1)?.extract()?;
                            reads.insert(loc);
                        }
                        _ => {}
                    },
                    _ => {}
                }

                // Store slot as native Rust values (inline array, no heap allocation)
                let mut slot_vec: ArrayVec<SlotVal, MAX_SLOT_SIZE> = ArrayVec::new();
                for i in 0..slot_tuple.len() {
                    let item = slot_tuple.get_item(i)?;
                    if let Ok(s) = item.extract::<String>() {
                        slot_vec.push(SlotVal::Str(s));
                    } else if let Ok(n) = item.extract::<i64>() {
                        slot_vec.push(SlotVal::Int(n));
                    } else {
                        // Fallback: convert to string
                        slot_vec.push(SlotVal::Str(item.str()?.to_string()));
                    }
                }

                // Convert engine string to enum
                let engine_enum = Engine::from_str(&engine).unwrap_or(Engine::Alu);

                ops.push(Op {
                    id: op_id,
                    engine: engine_enum,
                    slot: slot_vec,
                    cycle: cycle as i32,
                    dep_ids: HashSet::new(),
                    reads,
                    writes,
                    last_use_reads: HashSet::new(),
                    trace_id: None,
                    produces_constant: None,
                    from_const_choice: None,
                    from_select_choice: None,
                });
                op_id += 1;
            }
        }
    }

    Ok(ops)
}

// =============================================================================
// Emission: Rust -> Python
// =============================================================================

/// Emit scheduled ops back to Python instruction list
pub fn emit_scheduled<'py>(
    py: Python<'py>,
    ops: &[Op],
    max_cycle: usize,
) -> PyResult<Bound<'py, PyList>> {
    emit_scheduled_with_expanded(py, ops, &[], max_cycle)
}

/// Emit scheduled ops including expanded ALU ops from VALU→ALU shifting
/// Skips ops with engine="shifted_valu" (these are replaced by expanded_ops)
pub fn emit_scheduled_with_expanded<'py>(
    py: Python<'py>,
    ops: &[Op],
    expanded_ops: &[Op],
    max_cycle: usize,
) -> PyResult<Bound<'py, PyList>> {
    let instrs = PyList::empty(py);

    for cycle in 0..max_cycle {
        let instr = PyDict::new(py);

        // Emit regular ops, SKIP shifted_valu (these are replaced by expanded ALU ops)
        for op in ops {
            if op.cycle == cycle as i32
                && op.engine != Engine::ShiftedValu
                && op.engine != Engine::EliminatedBroadcast
            {
                emit_op_to_dict(py, op, &instr)?;
            }
        }

        // Emit expanded ALU ops from VALU→ALU shifting
        for op in expanded_ops {
            if op.cycle == cycle as i32 {
                emit_op_to_dict(py, op, &instr)?;
            }
        }

        // Emit one bundle per cycle so program length = makespan (machine runs one cycle per instruction).
        instrs.append(instr)?;
    }

    Ok(instrs)
}

/// Helper to emit a single op to a dict
fn emit_op_to_dict<'py>(py: Python<'py>, op: &Op, instr: &Bound<'py, PyDict>) -> PyResult<()> {
    let engine_str = op.engine.as_str();

    // Get or create the list for this engine
    if !instr.contains(engine_str)? {
        instr.set_item(engine_str, PyList::empty(py))?;
    }

    let slots = instr.get_item(engine_str)?.unwrap();
    let slots_list = slots.downcast::<PyList>()?;

    // Create tuple from slot
    let py_items: Vec<PyObject> = op
        .slot
        .iter()
        .map(|v| match v {
            SlotVal::Str(s) => s.into_pyobject(py).unwrap().into_any().unbind(),
            SlotVal::Int(n) => n.into_pyobject(py).unwrap().into_any().unbind(),
        })
        .collect();
    let slot_tuple = PyTuple::new(py, &py_items)?;
    slots_list.append(slot_tuple)?;

    Ok(())
}

// =============================================================================
// PyO3 Functions
// =============================================================================

/// Main scheduling function exposed to Python
#[pyfunction]
pub fn schedule<'py>(
    py: Python<'py>,
    instrs: &Bound<'py, PyList>,
    enable_valu_shift: bool,
) -> PyResult<Bound<'py, PyList>> {
    let mut ops = parse_ops(py, instrs)?;

    if ops.is_empty() {
        return Ok(PyList::empty(py));
    }

    // Reset cycles
    for op in &mut ops {
        op.cycle = -1;
    }

    // Compute dependencies
    compute_dependencies(&mut ops);

    // Compute critical path
    let critical_path = compute_critical_path(&ops);

    // Schedule with default priority mode
    let result = schedule_greedy(
        &mut ops,
        &critical_path,
        enable_valu_shift,
        PriorityMode::CriticalPath,
    );

    // Emit (include expanded ops from VALU→ALU shifting)
    emit_scheduled_with_expanded(py, &ops, &result.expanded_ops, result.cycles)
}

/// Get just the cycle count without emitting (faster for testing)
#[pyfunction]
pub fn schedule_cycles(
    py: Python<'_>,
    instrs: &Bound<'_, PyList>,
    enable_valu_shift: bool,
) -> PyResult<usize> {
    let mut ops = parse_ops(py, instrs)?;

    if ops.is_empty() {
        return Ok(0);
    }

    for op in &mut ops {
        op.cycle = -1;
    }

    compute_dependencies(&mut ops);
    let critical_path = compute_critical_path(&ops);
    let result = schedule_greedy(
        &mut ops,
        &critical_path,
        enable_valu_shift,
        PriorityMode::CriticalPath,
    );

    Ok(result.cycles)
}

/// Run multiple scheduling passes with different priority modes and return the best
#[pyfunction]
#[pyo3(signature = (instrs, enable_valu_shift, verbose=true))]
pub fn schedule_multipass<'py>(
    py: Python<'py>,
    instrs: &Bound<'py, PyList>,
    enable_valu_shift: bool,
    verbose: bool,
) -> PyResult<Bound<'py, PyList>> {
    let base_ops = parse_ops(py, instrs)?;

    if base_ops.is_empty() {
        return Ok(PyList::empty(py));
    }

    // Compute dependencies once (they don't change between passes)
    let mut ops_for_deps = base_ops.clone();
    compute_dependencies(&mut ops_for_deps);

    // Copy dep_ids to base_ops for reuse
    let dep_ids_map: HashMap<usize, HashSet<usize>> = ops_for_deps
        .iter()
        .map(|op| (op.id, op.dep_ids.clone()))
        .collect();

    let critical_path = compute_critical_path(&ops_for_deps);

    let mut best_cycles = usize::MAX;
    let mut best_ops: Option<Vec<Op>> = None;
    let mut best_expanded_ops: Vec<Op> = Vec::new();
    let mut best_mode: Option<PriorityMode> = None;

    // Try all priority modes and keep the best result
    for &mode in ALL_PRIORITY_MODES.iter() {
        let mut ops = base_ops.clone();
        for op in &mut ops {
            op.cycle = -1;
            op.dep_ids = dep_ids_map.get(&op.id).cloned().unwrap_or_default();
        }

        let result = schedule_greedy(&mut ops, &critical_path, enable_valu_shift, mode);

        if verbose {
            println!("  {:?}: {} cycles", mode, result.cycles);
        }

        if result.cycles < best_cycles {
            best_cycles = result.cycles;
            best_ops = Some(ops);
            best_expanded_ops = result.expanded_ops;
            best_mode = Some(mode);
        }
    }

    if verbose {
        println!("Best: {:?} with {} cycles", best_mode.unwrap(), best_cycles);
    }

    emit_scheduled_with_expanded(py, &best_ops.unwrap(), &best_expanded_ops, best_cycles)
}

/// Count cycles for a program (simulates cycle counting without full memory simulation)
#[pyfunction]
pub fn count_cycles(_py: Python<'_>, instrs: &Bound<'_, PyList>) -> PyResult<usize> {
    let mut cycles = 0;
    for instr in instrs.iter() {
        let instr_dict = instr.downcast::<PyDict>()?;
        let has_non_debug = instr_dict.keys().iter().any(|key| {
            let key_str: String = key.extract().unwrap_or_default();
            key_str != "debug"
        });
        if has_non_debug {
            cycles += 1;
        }
    }
    Ok(cycles)
}

/// Get cycle count from multipass (for testing)
#[pyfunction]
#[pyo3(signature = (instrs, enable_valu_shift, verbose=true))]
pub fn schedule_multipass_cycles(
    py: Python<'_>,
    instrs: &Bound<'_, PyList>,
    enable_valu_shift: bool,
    verbose: bool,
) -> PyResult<usize> {
    let base_ops = parse_ops(py, instrs)?;

    if base_ops.is_empty() {
        return Ok(0);
    }

    let mut ops_for_deps = base_ops.clone();
    compute_dependencies(&mut ops_for_deps);

    let dep_ids_map: HashMap<usize, HashSet<usize>> = ops_for_deps
        .iter()
        .map(|op| (op.id, op.dep_ids.clone()))
        .collect();

    let critical_path = compute_critical_path(&ops_for_deps);

    let mut best_cycles = usize::MAX;
    let mut best_mode: Option<PriorityMode> = None;

    for &mode in ALL_PRIORITY_MODES.iter() {
        let mut ops = base_ops.clone();
        for op in &mut ops {
            op.cycle = -1;
            op.dep_ids = dep_ids_map.get(&op.id).cloned().unwrap_or_default();
        }

        let result = schedule_greedy(&mut ops, &critical_path, enable_valu_shift, mode);

        if verbose {
            println!("  {:?}: {} cycles", mode, result.cycles);
        }

        if result.cycles < best_cycles {
            best_cycles = result.cycles;
            best_mode = Some(mode);
        }
    }

    if verbose {
        println!("Best: {:?} with {} cycles", best_mode.unwrap(), best_cycles);
    }

    Ok(best_cycles)
}

/// Run multipass scheduling with diagnostics collection, returns JSON report
#[pyfunction]
#[pyo3(signature = (instrs, enable_valu_shift, verbose=true))]
pub fn run_diagnostics(
    py: Python<'_>,
    instrs: &Bound<'_, PyList>,
    enable_valu_shift: bool,
    verbose: bool,
) -> PyResult<String> {
    let base_ops = parse_ops(py, instrs)?;

    if base_ops.is_empty() {
        return Ok("{}".to_string());
    }

    let mut ops_for_deps = base_ops.clone();
    compute_dependencies(&mut ops_for_deps);

    let dep_ids_map: HashMap<usize, HashSet<usize>> = ops_for_deps
        .iter()
        .map(|op| (op.id, op.dep_ids.clone()))
        .collect();

    let critical_path = compute_critical_path(&ops_for_deps);

    // First pass: find best mode
    let mut best_cycles = usize::MAX;
    let mut best_mode: Option<PriorityMode> = None;

    for &mode in ALL_PRIORITY_MODES.iter() {
        let mut ops = base_ops.clone();
        for op in &mut ops {
            op.cycle = -1;
            op.dep_ids = dep_ids_map.get(&op.id).cloned().unwrap_or_default();
        }

        let result = schedule_greedy(&mut ops, &critical_path, enable_valu_shift, mode);

        if verbose {
            println!("  {:?}: {} cycles", mode, result.cycles);
        }

        if result.cycles < best_cycles {
            best_cycles = result.cycles;
            best_mode = Some(mode);
        }
    }

    let best_mode = best_mode.unwrap();
    if verbose {
        println!("Best: {:?} with {} cycles", best_mode, best_cycles);
        println!("Collecting diagnostics for best mode...");
    }

    // Second pass: collect diagnostics for best mode
    let mut ops = base_ops.clone();
    for op in &mut ops {
        op.cycle = -1;
        op.dep_ids = dep_ids_map.get(&op.id).cloned().unwrap_or_default();
    }

    let result = schedule_greedy_with_diagnostics(
        &mut ops,
        &critical_path,
        enable_valu_shift,
        best_mode,
        true,
    );

    // Analyze shared temp dependencies
    let shared_temp_analysis = analyze_shared_temp_dependencies(&ops_for_deps);

    // Build combined report
    #[derive(Serialize)]
    struct DiagnosticReport {
        scheduling: SchedulingDiagnostics,
        shared_temps: SharedTempAnalysis,
        best_mode: String,
        cycles: usize,
    }

    let report = DiagnosticReport {
        scheduling: result.diagnostics.unwrap_or_default(),
        shared_temps: shared_temp_analysis,
        best_mode: format!("{:?}", best_mode),
        cycles: result.cycles,
    };

    if verbose {
        println!("\n{}", report.scheduling.summary());
        println!("\n{}", report.shared_temps.summary());
    }

    Ok(serde_json::to_string_pretty(&report).unwrap_or_else(|_| "{}".to_string()))
}

/// Run shared temp analysis only, returns JSON
#[pyfunction]
pub fn analyze_shared_temps(py: Python<'_>, instrs: &Bound<'_, PyList>) -> PyResult<String> {
    let mut ops = parse_ops(py, instrs)?;

    if ops.is_empty() {
        return Ok("{}".to_string());
    }

    compute_dependencies(&mut ops);
    let analysis = analyze_shared_temp_dependencies(&ops);

    Ok(analysis.summary()
        + "\n\n"
        + &serde_json::to_string_pretty(&analysis).unwrap_or_else(|_| "{}".to_string()))
}

/// Print full diagnostic report (convenience function)
#[pyfunction]
#[pyo3(signature = (instrs, enable_valu_shift=true))]
pub fn print_diagnostics(
    py: Python<'_>,
    instrs: &Bound<'_, PyList>,
    enable_valu_shift: bool,
) -> PyResult<()> {
    let json = run_diagnostics(py, instrs, enable_valu_shift, true)?;
    println!("\n=== JSON REPORT ===\n{}", json);
    Ok(())
}

// =============================================================================
// JSON Output (delegates to schedule_output)
// =============================================================================

/// Schedule instructions and output JSON with slot utilization markers
#[pyfunction]
#[pyo3(signature = (instrs, enable_valu_shift, output_path, optimize_consts=true))]
pub fn schedule_to_json(
    py: Python<'_>,
    instrs: &Bound<'_, PyList>,
    enable_valu_shift: bool,
    output_path: String,
    optimize_consts: bool,
) -> PyResult<String> {
    let mut base_ops = parse_ops(py, instrs)?;

    if base_ops.is_empty() {
        return Ok("{}".to_string());
    }

    // Apply constant derivation optimization
    if optimize_consts {
        let usage = analyze_constant_usage(&base_ops);

        if !usage.is_empty() {
            let plan = build_constant_plan(&usage, 3);

            println!(
                "Const derivation: {} unique values, {} bootstrap loads, {} derivable",
                usage.len(),
                plan.bootstrap_loads.len(),
                plan.derivations.len() - plan.bootstrap_loads.len()
            );

            // Find highest used register
            let max_reg = base_ops
                .iter()
                .flat_map(|op| op.reads.iter().chain(op.writes.iter()))
                .max()
                .copied()
                .unwrap_or(0);

            // Allocate fresh registers
            let mut value_to_register: HashMap<i64, i64> = HashMap::new();
            let mut register_rename: HashMap<i64, i64> = HashMap::new();
            let mut next_fresh_reg = max_reg + 1000;

            let mut sorted_values: Vec<_> = usage.keys().copied().collect();
            sorted_values.sort();

            for value in sorted_values {
                let u = &usage[&value];
                let fresh_reg = next_fresh_reg;
                next_fresh_reg += 1;
                value_to_register.insert(value, fresh_reg);

                for &dest in &u.destinations {
                    register_rename.insert(dest, fresh_reg);
                }
            }

            let renames_count = register_rename.len();

            // Remove const loads
            let const_op_ids: HashSet<usize> = base_ops
                .iter()
                .filter(|op| {
                    op.engine == Engine::Load
                        && op
                            .slot
                            .get(0)
                            .map(|s| matches!(s, SlotVal::Str(ref o) if o == "const"))
                            .unwrap_or(false)
                })
                .map(|op| op.id)
                .collect();

            let original_const_count = const_op_ids.len();
            base_ops.retain(|op| !const_op_ids.contains(&op.id));

            // Apply register renaming to reads
            for op in &mut base_ops {
                let opcode: String = op
                    .slot
                    .get(0)
                    .and_then(|v| match v {
                        SlotVal::Str(s) => Some(s.clone()),
                        _ => None,
                    })
                    .unwrap_or_default();

                for (idx, slot_val) in op.slot.iter_mut().enumerate() {
                    if let SlotVal::Int(reg) = slot_val {
                        let is_source = match opcode.as_str() {
                            "+" | "-" | "*" | "/" | "%" | "&" | "|" | "^" | "<<" | ">>" | "=="
                            | "!=" | "<" | "<=" | ">" | ">=" => idx >= 2,
                            "multiply_add" | "vmultiply_add" => idx >= 2,
                            "vbroadcast" => idx == 2,
                            "vselect" => idx >= 2,
                            "load" | "vload" => idx == 2,
                            "store" | "vstore" => idx >= 1,
                            "const" => false,
                            _ => idx >= 2,
                        };

                        if is_source {
                            if let Some(&fresh) = register_rename.get(reg) {
                                *reg = fresh;
                            }
                        }
                    }
                }

                let renamed_reads: HashSet<i64> = op
                    .reads
                    .iter()
                    .map(|r| *register_rename.get(r).unwrap_or(r))
                    .collect();
                op.reads = renamed_reads;
            }

            // Generate constant ops
            let max_id = base_ops.iter().map(|op| op.id).max().unwrap_or(0);
            let const_ops = emit_constant_generation_ops(&plan, &value_to_register, max_id + 1);

            println!(
                "Const optimization: {} loads removed, {} generation ops added, {} registers renamed (saves {})",
                original_const_count,
                const_ops.len(),
                renames_count,
                original_const_count.saturating_sub(const_ops.len())
            );

            let mut new_ops = const_ops;
            new_ops.extend(base_ops);
            base_ops = new_ops;
        }
    }

    // Compute dependencies
    let mut ops_for_deps = base_ops.clone();
    compute_dependencies(&mut ops_for_deps);

    let dep_ids_map: HashMap<usize, HashSet<usize>> = ops_for_deps
        .iter()
        .map(|op| (op.id, op.dep_ids.clone()))
        .collect();

    for op in &mut base_ops {
        if let Some(deps) = dep_ids_map.get(&op.id) {
            op.dep_ids = deps.clone();
        }
    }

    let critical_path = compute_critical_path(&ops_for_deps);

    // Find best mode
    let mut best_cycles = usize::MAX;
    let mut best_ops: Option<Vec<Op>> = None;
    let mut best_expanded_ops: Vec<Op> = Vec::new();
    let mut best_mode: Option<PriorityMode> = None;

    for &mode in ALL_PRIORITY_MODES.iter() {
        let mut ops = base_ops.clone();
        for op in &mut ops {
            op.cycle = -1;
            op.dep_ids = dep_ids_map.get(&op.id).cloned().unwrap_or_default();
        }

        let result = schedule_greedy(&mut ops, &critical_path, enable_valu_shift, mode);

        if result.cycles < best_cycles {
            best_cycles = result.cycles;
            best_ops = Some(ops);
            best_expanded_ops = result.expanded_ops;
            best_mode = Some(mode);
        }
    }

    let best_mode = best_mode.unwrap();
    let ops = best_ops.unwrap();

    let json = build_schedule_output_json(&ops, &best_expanded_ops, best_cycles, best_mode);

    if !output_path.is_empty() {
        write_schedule_output_to_path(
            &ops,
            &best_expanded_ops,
            best_cycles,
            best_mode,
            &output_path,
        )
        .map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to write {}: {}", output_path, e))
        })?;
        println!("Wrote schedule to {}", output_path);
    }

    Ok(json)
}

/// Analyze constants and build derivation plan
#[pyfunction]
#[pyo3(signature = (instrs, hot_threshold=3))]
pub fn analyze_constants(
    py: Python<'_>,
    instrs: &Bound<'_, PyList>,
    hot_threshold: usize,
) -> PyResult<String> {
    let ops = parse_ops(py, instrs)?;

    let usage = analyze_constant_usage(&ops);

    if usage.is_empty() {
        return Ok("No constants found".to_string());
    }

    let plan = build_constant_plan(&usage, hot_threshold);

    print_constant_plan_summary(&plan);

    let summary = serde_json::json!({
        "total_unique_constants": usage.len(),
        "bootstrap_loads": plan.bootstrap_loads,
        "total_derivations": plan.derivations.len(),
        "alu_ops_added": plan.total_alu_ops_added,
        "const_loads_saved": plan.total_const_loads_saved,
        "constant_pool": plan.constant_pool.iter().collect::<Vec<_>>(),
        "usage_by_value": usage.iter()
            .map(|(v, u)| (*v, serde_json::json!({
                "count": u.use_count,
                "destinations": u.destinations.len(),
            })))
            .collect::<HashMap<_, _>>(),
    });

    Ok(serde_json::to_string_pretty(&summary).unwrap_or_else(|_| "{}".to_string()))
}

/// Apply constant derivation optimization and return optimized ops
/// If skip_scheduling is true, returns unscheduled ops (for use with schedule_multipass)
#[pyfunction]
#[pyo3(signature = (instrs, enable_valu_shift, hot_threshold=3, verbose=true, skip_scheduling=false))]
pub fn optimize_constants<'py>(
    py: Python<'py>,
    instrs: &Bound<'py, PyList>,
    enable_valu_shift: bool,
    hot_threshold: usize,
    verbose: bool,
    skip_scheduling: bool,
) -> PyResult<Bound<'py, PyList>> {
    let mut ops = parse_ops(py, instrs)?;

    if ops.is_empty() {
        return Ok(PyList::empty(py));
    }

    let usage = analyze_constant_usage(&ops);

    if usage.is_empty() {
        if skip_scheduling {
            // Return unscheduled ops as-is
            return emit_unscheduled(py, &ops);
        }
        let max_cycle = ops
            .iter()
            .map(|op| op.cycle.max(0) as usize)
            .max()
            .unwrap_or(0);
        return emit_scheduled(py, &ops, max_cycle + 1);
    }

    let plan = build_constant_plan(&usage, hot_threshold);

    if verbose {
        print_constant_plan_summary(&plan);
    }

    // Allocate registers
    let mut value_to_register: HashMap<i64, i64> = HashMap::new();
    for (&value, u) in &usage {
        if let Some(&dest) = u.destinations.iter().next() {
            value_to_register.insert(value, dest);
        }
    }

    // Remove const loads
    let const_op_ids: HashSet<usize> = ops
        .iter()
        .filter(|op| {
            op.engine == Engine::Load
                && op
                    .slot
                    .get(0)
                    .map(|s| matches!(s, SlotVal::Str(ref o) if o == "const"))
                    .unwrap_or(false)
        })
        .map(|op| op.id)
        .collect();

    ops.retain(|op| !const_op_ids.contains(&op.id));

    // Generate constant ops
    let max_id = ops.iter().map(|op| op.id).max().unwrap_or(0);
    let const_ops = emit_constant_generation_ops(&plan, &value_to_register, max_id + 1);

    if verbose {
        println!(
            "Generated {} constant ops (was {} loads)",
            const_ops.len(),
            const_op_ids.len()
        );
    }

    let mut new_ops = const_ops;
    new_ops.extend(ops);

    compute_dependencies(&mut new_ops);

    // If skip_scheduling, return unscheduled ops for caller to schedule with multipass
    if skip_scheduling {
        return emit_unscheduled(py, &new_ops);
    }

    let critical_path = compute_critical_path(&new_ops);
    let result = schedule_greedy(
        &mut new_ops,
        &critical_path,
        enable_valu_shift,
        PriorityMode::CriticalPath,
    );

    if verbose {
        println!("Scheduled to {} cycles", result.cycles);
    }

    emit_scheduled_with_expanded(py, &new_ops, &result.expanded_ops, result.cycles)
}

/// Emit unscheduled ops back to Python (one op per "cycle" bundle)
fn emit_unscheduled<'py>(py: Python<'py>, ops: &[Op]) -> PyResult<Bound<'py, PyList>> {
    let result = PyList::empty(py);

    for op in ops {
        let instr = PyDict::new(py);
        emit_op_to_dict(py, op, &instr)?;
        result.append(instr)?;
    }

    Ok(result)
}

// =============================================================================
// Kernel Building Functions
// =============================================================================

/// Check if CP-SAT scheduler is available (compiled with --features cpsat)
#[pyfunction]
pub fn cpsat_available() -> bool {
    crate::cpsat_scheduler::cpsat_available()
}

// =============================================================================
// Optional constraint group (Python wrapper for Rust enum)
// =============================================================================

use crate::cpsat_scheduler::OptionalConstraintGroup;

/// Python-exposed wrapper for OptionalConstraintGroup so we can pass enum values over the boundary.
#[pyclass(name = "OptionalConstraintGroup", module = "scheduler_rs")]
#[derive(Clone, Copy)]
pub struct OptionalConstraintGroupPy {
    pub inner: OptionalConstraintGroup,
}

#[pymethods]
impl OptionalConstraintGroupPy {
    /// CLI name for this group (e.g. "redundant_linear_bounds").
    #[getter]
    fn name(&self) -> String {
        self.inner.to_string()
    }

    /// Parse a CLI name into an OptionalConstraintGroup. Raises ValueError on invalid name.
    #[classmethod]
    fn from_str(_cls: &Bound<'_, pyo3::types::PyType>, s: &str) -> PyResult<Self> {
        OptionalConstraintGroup::from_str(s)
            .map(|inner| OptionalConstraintGroupPy { inner })
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
    }

    fn __repr__(&self) -> String {
        format!("OptionalConstraintGroup.{}", self.name())
    }
}

/// Resolve CP-SAT tuning options against their defaults.
///
/// # Arguments
/// * `tuning_json` - JSON object with any subset of the `CpsatTuning` keys (None = all defaults).
///
/// # Returns
/// JSON object with every tuning key and its effective value. Raises `ValueError` for an
/// unknown key, a wrong type or an out-of-range value.
#[pyfunction]
#[pyo3(signature = (tuning_json=None))]
pub fn resolve_cpsat_tuning(tuning_json: Option<&str>) -> PyResult<String> {
    crate::cpsat_scheduler::CpsatTuning::from_json(tuning_json)
        .map(|t| t.to_json())
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

/// Returns the list of valid optional constraint group names (for CLI help and validation).
#[pyfunction]
pub fn optional_constraint_group_names() -> Vec<String> {
    use crate::cpsat_scheduler::OptionalConstraintGroup;
    OptionalConstraintGroup::ALL
    .iter()
    .map(|v| v.to_string())
    .collect()
}

// =============================================================================
// SSA Kernel Building Functions
// =============================================================================

use crate::kernel_builder::{KernelConfig, SsaTickTockKernelBuilder};

/// Build a tick-tock kernel using SSA-based emission.
///
/// This builder uses virtual registers (VRegs) instead of pre-allocated
/// addresses, eliminating false WAW/WAR dependencies for better scheduling.
///
/// Returns a tuple (instructions, stats_dict) where:
/// - instructions: List of scheduled instruction dicts
/// - stats_dict: Dictionary with keys:
///   - cycles: Total scheduled cycles
///   - vreg_count: Number of virtual registers used
///   - peak_live: Peak simultaneous live words
///   - max_addr: Highest scratch address used
///   - budget_exceeded: Whether scratch budget was exceeded
///
/// If verbose is true, prints progress information.
/// If enable_valu_shift is true (default), VALU ops are shifted to ALU when VALU is saturated.
/// If use_cpsat is true, uses CP-SAT scheduler instead of greedy (operates on SSA before register allocation).
/// cpsat_time_limit sets the maximum time for CP-SAT solver (default: 600 seconds / 10 minutes).
/// per_section_optimize_secs: when use_cpsat is true and > 0, each per-section CP-SAT run uses this time limit
/// and optimizes for cycles (no VALU shift in per-section). When 0 or None, per-section runs in feasibility-only mode.
/// enable_cse_groups: when use_cpsat is true, enable CSE (common subexpression) groups so SelectChoice ops sharing
/// the same (true_val, false_val) use one shared SUB per group instead of one SUB per FMA, reducing VALU pressure.
/// Defaults to true when not provided. Has no effect when use_cpsat is false.
/// If output_json is provided, writes schedule with usage accounting to the specified path.
/// liveness_detail controls verbosity of liveness data in JSON: "none", "summary", or "full" (default).
#[pyfunction]
#[pyo3(signature = (forest_height, n_nodes, batch_size, rounds, enable_valu_shift=true, verbose=true, use_cpsat=false, cpsat_time_limit=600.0, per_section_optimize_secs=None, output_json=None, tree_reduce_max_depth=3, const_choice="optimize", select_choice="optimize", valu_shift="optimize", *, liveness_detail=None, forced_shifts=None, force_zero_gap=None, enable_vbroadcast_elimination=None, run_options_json=None, git_sha=None, comment=None, asap_alap_tightening="energetic", enable_scratch_tightening=None, symmetry_precedence_integrated=false, energetic_alap_pulls=false, energetic_bounded_push=false, energetic_bounded_push_k=None, precedence_neutral_horizon=false, scheduling_pressure_policy="adaptive", optional_constraint_groups_to_disable=None, enable_cse_groups=None, enable_trace_ids=None, verify_same_cycle_scratch=true, enable_bulk_move=false, greedy_use_full_bounds=false, enable_const_choice_hint_pass=false, enable_select_choice_hint_pass=false, enable_valu_shift_hint_pass=false, enable_schedule_with_fixed_choices=false, cpsat_deterministic_time=None, cpsat_tuning_json=None))]
pub fn build_tick_tock_kernel_ssa<'py>(
    py: Python<'py>,
    forest_height: usize,
    n_nodes: usize,
    batch_size: usize,
    rounds: usize,
    enable_valu_shift: bool,
    verbose: bool,
    use_cpsat: bool,
    cpsat_time_limit: f64,
    per_section_optimize_secs: Option<f64>,
    output_json: Option<String>,
    tree_reduce_max_depth: u32,
    const_choice: &str,
    select_choice: &str,
    valu_shift: &str,
    liveness_detail: Option<String>,
    forced_shifts: Option<Vec<usize>>,
    force_zero_gap: Option<bool>,
    enable_vbroadcast_elimination: Option<bool>,
    run_options_json: Option<String>,
    git_sha: Option<String>,
    comment: Option<String>,
    asap_alap_tightening: &str,
    enable_scratch_tightening: Option<bool>,
    symmetry_precedence_integrated: bool,
    energetic_alap_pulls: bool,
    energetic_bounded_push: bool,
    energetic_bounded_push_k: Option<u32>,
    precedence_neutral_horizon: bool,
    scheduling_pressure_policy: &str,
    optional_constraint_groups_to_disable: Option<Vec<OptionalConstraintGroupPy>>,
    enable_cse_groups: Option<bool>,
    enable_trace_ids: Option<bool>,
    verify_same_cycle_scratch: bool,
    enable_bulk_move: bool,
    greedy_use_full_bounds: bool,
    enable_const_choice_hint_pass: bool,
    enable_select_choice_hint_pass: bool,
    enable_valu_shift_hint_pass: bool,
    enable_schedule_with_fixed_choices: bool,
    cpsat_deterministic_time: Option<f64>,
    cpsat_tuning_json: Option<&str>,
) -> PyResult<(Bound<'py, PyList>, Bound<'py, PyDict>)> {
    let cpsat_tuning = crate::cpsat_scheduler::CpsatTuning::from_json(cpsat_tuning_json)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    if let Some(dtime) = cpsat_deterministic_time {
        if !(dtime.is_finite() && dtime > 0.0) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "cpsat_deterministic_time must be a positive finite number, got {}",
                dtime
            )));
        }
    }
    if batch_size == 0 || batch_size % VLEN != 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "batch_size must be a positive multiple of VLEN ({}) (got {}). Use e.g. 8, 16, 32.",
            VLEN, batch_size
        )));
    }
    use crate::kernel_builder::config::MAX_TREE_REDUCE_DEPTH;
    let tree_reduce_max_depth = tree_reduce_max_depth.clamp(2, MAX_TREE_REDUCE_DEPTH);
    use crate::kernel_builder::config::{ProblemSpec, CompilerConfig, PressurePolicyKind};
    let pressure_policy_kind = match scheduling_pressure_policy {
        "greedy" => PressurePolicyKind::Greedy,
        "throttled" => PressurePolicyKind::PressureThrottled,
        "gated" => PressurePolicyKind::PressureGated,
        "adaptive" => PressurePolicyKind::Adaptive,
        "cascade" => PressurePolicyKind::Cascade,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Invalid scheduling_pressure_policy '{}': must be 'greedy', 'throttled', 'gated', 'adaptive', or 'cascade'",
                other
            )));
        }
    };
    let config = KernelConfig {
        spec: ProblemSpec {
            forest_height,
            n_nodes,
            batch_size,
            rounds,
            enable_debug: false,
            enable_trace_ids: enable_trace_ids.unwrap_or(false),
            enable_lazy_partition: false,
            partition_frequency: 3,
            enable_partition_stores: false,
            enable_pseudo_consts: false,
            enable_add_imm: true,
            enable_const_load: true,
            enable_const_alu: true,
            max_alu_depth: None,
            tree_reduce_max_depth,
        },
        compiler: CompilerConfig {
            enable_valu_shift,
            enable_ssa_dce: true, // Remove dead code before scheduling (saves cycles, optimizer time, scratch)
            scheduling_pressure_policy: pressure_policy_kind,
            verify_same_cycle_scratch,
            greedy_use_full_bounds,
            enable_const_choice_hint_pass,
            enable_select_choice_hint_pass,
            enable_valu_shift_hint_pass,
            enable_schedule_with_fixed_choices,
        },
    };

    let mut builder = SsaTickTockKernelBuilder::new(config);
    builder.cpsat_deterministic_time = cpsat_deterministic_time;
    builder.cpsat_tuning = cpsat_tuning;

    if use_cpsat && !builder.config.compiler.enable_ssa_dce {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "CP-SAT requires DCE; enable_ssa_dce must be true when use_cpsat is true.",
        ));
    }

    if verbose {
        println!(
            "Building SSA kernel{}...",
            if use_cpsat { " (CP-SAT)" } else { "" }
        );
    }

    let const_choice_mode = crate::cpsat_scheduler::ConstChoiceMode::from_str(const_choice)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
    let select_choice_mode = crate::cpsat_scheduler::SelectChoiceMode::from_str(select_choice)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
    let valu_shift_mode = crate::cpsat_scheduler::ValuShiftMode::from_str(valu_shift)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

    let forced_shifts_set: Option<std::collections::HashSet<usize>> =
        forced_shifts.map(|v| v.into_iter().collect());
    let force_zero_gap = force_zero_gap.unwrap_or(false);
    // Off by default: elimination is buggy (scalar liveness not extended when shifted ALU reads it).
    let enable_vbroadcast_elimination = enable_vbroadcast_elimination.unwrap_or(false);
    let asap_alap_tightening = crate::cpsat_scheduler::AsapAlapTightening::from_str(asap_alap_tightening)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
    let enable_scratch_tightening = enable_scratch_tightening.unwrap_or(false);
    let energetic_bounded_push_k = if energetic_bounded_push {
        Some(energetic_bounded_push_k.unwrap_or(2))
    } else {
        None
    };
    let enable_cse_groups = enable_cse_groups.unwrap_or(true);
    let disable_optional_set: Option<HashSet<OptionalConstraintGroup>> =
        optional_constraint_groups_to_disable.map(|v| v.into_iter().map(|p| p.inner).collect());
    let result = if use_cpsat {
        builder
            .build_and_schedule_cpsat(
                cpsat_time_limit,
                verbose,
                enable_valu_shift,
                enable_vbroadcast_elimination,
                per_section_optimize_secs,
                forced_shifts_set,
                const_choice_mode,
                select_choice_mode,
                valu_shift_mode,
                force_zero_gap,
                asap_alap_tightening,
                enable_scratch_tightening,
                symmetry_precedence_integrated,
                energetic_alap_pulls,
                energetic_bounded_push_k,
                precedence_neutral_horizon,
                disable_optional_set,
                enable_cse_groups,
                enable_bulk_move,
            )
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("SSA CP-SAT build failed: {}", e))
            })?
    } else {
        let enable_valu_shift = builder.config.compiler.enable_valu_shift;
        builder.build_and_schedule_greedy(enable_valu_shift, None, verbose).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("SSA build failed: {}", e))
        })?
    };

    if verbose {
        println!(
            "SSA kernel: {} cycles, {} ops, {} VRegs, peak_live={}, max_addr={}, valu_shifted={}",
            result.cycles,
            result.ops.len(),
            result.vreg_count,
            result.address_result.peak_live,
            result.address_result.max_addr,
            result.valu_shifted
        );
        let empty = empty_cycles(&result.ops, &[], result.cycles);
        if !empty.is_empty() {
            const MAX_SHOW: usize = 25;
            let show: Vec<String> = empty
                .iter()
                .take(MAX_SHOW)
                .map(|c| c.to_string())
                .collect();
            let rest = empty.len().saturating_sub(MAX_SHOW);
            if rest > 0 {
                println!(
                    "  Empty cycles: {} at {} (suboptimal; first {} shown)",
                    empty.len(),
                    show.join(", ") + &format!(" ... and {} more", rest),
                    MAX_SHOW
                );
            } else {
                println!("  Empty cycles: {} at {}", empty.len(), show.join(", "));
            }
            if let (Some(ec), Some(true)) = (result.empty_cycle_count, result.makespan_driven_by_capacity) {
                if ec > 0 {
                    let last_op_cycle_plus_one = result.cycles.saturating_sub(ec);
                    println!(
                        "  Makespan is driven by capacity lower bound ({}); last op cycle+1 = {}.",
                        result.cycles, last_op_cycle_plus_one
                    );
                }
            }
        }
    }

    // Emit scheduled ops (one bundle per cycle; do not elide empty cycles)
    let scheduled = emit_scheduled(py, &result.ops, result.cycles)?;

    // Build final list with pause wrappers
    let instrs = PyList::empty(py);
    let pause_dict = PyDict::new(py);
    pause_dict.set_item(
        "flow",
        PyList::new(
            py,
            &[PyTuple::new(
                py,
                &["pause".into_pyobject(py).unwrap().into_any()],
            )?],
        )?,
    )?;
    instrs.append(&pause_dict)?;

    for item in scheduled.iter() {
        instrs.append(item)?;
    }

    instrs.append(&pause_dict)?;

    // Build stats dict
    let stats = PyDict::new(py);
    stats.set_item("cycles", result.cycles)?;
    stats.set_item("vreg_count", result.vreg_count)?;
    stats.set_item("peak_live", result.address_result.peak_live)?;
    stats.set_item("max_addr", result.address_result.max_addr)?;
    stats.set_item("budget_exceeded", result.budget_exceeded)?;
    stats.set_item("valu_shifted", result.valu_shifted)?;
    if let Some(ec) = result.empty_cycle_count {
        stats.set_item("empty_cycle_count", ec)?;
    }
    if let Some(md) = result.makespan_driven_by_capacity {
        stats.set_item("makespan_driven_by_capacity", md)?;
    }

    // Add usage accounting as JSON
    let usage_json =
        serde_json::to_string(&result.usage_accounting).unwrap_or_else(|_| "{}".to_string());
    stats.set_item("usage_accounting_json", usage_json)?;

    // Also add summary stats for easy access
    let usage = &result.usage_accounting;
    stats.set_item("usage_total_vregs", usage.total_vregs)?;
    stats.set_item("usage_total_words", usage.total_words)?;
    stats.set_item("usage_peak_live_words", usage.peak_live_words)?;

    // Write output JSON if requested
    if let Some(output_path) = output_json {
        // Default to "full" liveness detail when output is requested
        let detail = liveness_detail
            .as_ref()
            .map(|s| LivenessDetail::from_str(s))
            .unwrap_or(LivenessDetail::Full);

        let run_options: Option<serde_json::Value> = run_options_json
            .as_ref()
            .and_then(|s| serde_json::from_str(s).ok());
        write_ssa_schedule_json(
            &result,
            &output_path,
            detail,
            run_options,
            git_sha,
            comment,
        )
        .map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to write output JSON: {}", e))
        })?;
        if verbose {
            println!(
                "Wrote schedule output with usage accounting to {} (liveness: {:?})",
                output_path, detail
            );
        }
    }

    Ok((instrs, stats))
}

// =============================================================================
// Bulk-move simulator (load from JSON, run Rust simulator, return result)
// =============================================================================

#[derive(serde::Deserialize)]
struct BulkMoveJsonInput {
    total_cycles: usize,
    #[serde(default)]
    ops: Option<Vec<OpBulkMove>>,
    #[serde(default)]
    schedule_constraints: Option<Vec<crate::schedule_constraints::ScheduleConstraint>>,
    /// JSON object keys are strings (op id as string)
    #[serde(default)]
    shifted_alu_cycles: std::collections::HashMap<String, Vec<i32>>,
    /// SSA-level data (same index space as schedule_constraints). Prefer over ops for bulk-move.
    #[serde(default)]
    ssa_cycle_assignments: Option<Vec<i32>>,
    /// SSA ops: id and dep_ids only; cycle comes from ssa_cycle_assignments by index.
    #[serde(default)]
    ssa_ops_bulk_move: Option<Vec<SsaOpBulkMoveIn>>,
}

#[derive(serde::Deserialize)]
struct SsaOpBulkMoveIn {
    id: usize,
    #[serde(default)]
    dep_ids: Vec<usize>,
}

#[derive(serde::Deserialize)]
struct OpBulkMove {
    id: usize,
    cycle: i32,
    #[serde(default)]
    dep_ids: Vec<usize>,
}

/// Run the bulk-move simulator on an SSA schedule JSON file.
///
/// The file must be the output of `build_tick_tock_kernel_ssa(..., output_json=path)` with
/// CP-SAT (so it contains `schedule_constraints` and `shifted_alu_cycles`). Uses the same
/// constraint logic as the CP-SAT model to validate each move.
///
/// Returns a dict with:
/// - "result": "NoEmptyCycles" | "ReachedEnd" | "Blocked"
/// - If ReachedEnd: "original_makespan", "final_makespan", "steps"
/// - If Blocked: "empty_c", "violation_message", "steps"
#[pyfunction]
#[pyo3(signature = (path, verbose=false))]
pub fn run_bulk_move_simulator_from_json<'py>(
    py: Python<'py>,
    path: &str,
    verbose: bool,
) -> PyResult<Bound<'py, PyDict>> {
    use crate::schedule_constraints::{run_bulk_move_simulator, BulkMoveResult};
    use crate::types::{Engine, Opcode, Operand, SsaOp};

    let json_str = std::fs::read_to_string(path).map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!("Failed to read {}: {}", path, e))
    })?;
    let input: BulkMoveJsonInput = serde_json::from_str(&json_str).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "Invalid SSA schedule JSON (need schedule_constraints and ops with id, cycle, dep_ids): {}",
            e
        ))
    })?;

    let constraints = input.schedule_constraints.as_deref().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(
            "JSON has no schedule_constraints (run with CP-SAT and output_json to generate it)",
        )
    })?;

    // Prefer SSA-level data (same index space as constraints); fall back to lowered ops (may mismatch).
    let (ops, mut cycle_assignments) = if let (Some(ssa_ops), Some(ssa_cy)) = (
        input.ssa_ops_bulk_move.as_ref(),
        input.ssa_cycle_assignments.as_ref(),
    ) {
        if ssa_ops.len() != ssa_cy.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "ssa_ops_bulk_move length {} != ssa_cycle_assignments length {}",
                ssa_ops.len(),
                ssa_cy.len()
            )));
        }
        let ops: Vec<SsaOp> = ssa_ops
            .iter()
            .map(|o| {
                let mut op = SsaOp::new(
                    o.id,
                    Engine::Alu,
                    Opcode::Add,
                    None,
                    Vec::<Operand>::new(),
                );
                op.dep_ids = o.dep_ids.iter().copied().collect();
                op
            })
            .collect();
        let cy: Vec<i32> = ssa_cy.clone();
        (ops, cy)
    } else if let Some(lowered_ops) = &input.ops {
        let ops: Vec<SsaOp> = lowered_ops
            .iter()
            .map(|o| {
                let mut op = SsaOp::new(
                    o.id,
                    Engine::Alu,
                    Opcode::Add,
                    None,
                    Vec::<Operand>::new(),
                );
                op.dep_ids = o.dep_ids.iter().copied().collect();
                op
            })
            .collect();
        let cy: Vec<i32> = lowered_ops.iter().map(|o| o.cycle).collect();
        (ops, cy)
    } else {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "JSON has neither ssa_ops_bulk_move/ssa_cycle_assignments nor ops (re-run with CP-SAT and output_json)",
        ));
    };

    let n = ops.len();
    let total_cycles = input.total_cycles as i32;
    let mut shifted_alu_cycles: HashMap<usize, Vec<i32>> = input
        .shifted_alu_cycles
        .into_iter()
        .filter_map(|(k, v)| k.parse::<usize>().ok().map(|id| (id, v)))
        .collect();

    if cycle_assignments.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "ops/ssa length {} != cycle_assignments length {}",
            n,
            cycle_assignments.len()
        )));
    }

    let result = run_bulk_move_simulator(
        &ops,
        &mut cycle_assignments[..],
        &mut shifted_alu_cycles,
        constraints,
        total_cycles,
        verbose,
    );

    let out = PyDict::new(py);
    match result {
        BulkMoveResult::NoEmptyCycles => {
            out.set_item("result", "NoEmptyCycles")?;
        }
        BulkMoveResult::ReachedEnd {
            original_max_cycle_plus_one,
            final_max_cycle_plus_one,
            steps,
        } => {
            out.set_item("result", "ReachedEnd")?;
            out.set_item("original_makespan", original_max_cycle_plus_one)?;
            out.set_item("final_makespan", final_max_cycle_plus_one)?;
            out.set_item("steps", steps)?;
        }
        BulkMoveResult::Blocked {
            empty_c,
            violation,
            steps,
        } => {
            out.set_item("result", "Blocked")?;
            out.set_item("empty_c", empty_c)?;
            out.set_item("violation_message", violation.message)?;
            out.set_item("steps", steps)?;
        }
    }
    Ok(out)
}
