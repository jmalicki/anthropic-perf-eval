//! Python interface for the scheduler.
//!
//! Contains all PyO3 bindings, parsing, and emission functions.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use serde::Serialize;
use std::collections::{BTreeSet, HashMap, HashSet};

use crate::allocator::{allocate_registers, rewrite_addresses, AllocationResult};
use crate::constants::{
    analyze_constant_usage, build_constant_plan, build_constant_plan_with_depth,
    emit_constant_generation_ops, print_constant_plan_summary,
};
use crate::peephole::{optimize as peephole_optimize, PeepholeConfig};
use crate::diagnostics::{analyze_shared_temp_dependencies, SchedulingDiagnostics, SharedTempAnalysis};
use crate::scheduler::{
    compute_critical_path, compute_last_uses, schedule_greedy, schedule_greedy_with_diagnostics,
    schedule_with_pseudo_consts, schedule_with_slot_filling, PriorityMode, ALL_PRIORITY_MODES,
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
                    "load" => {
                        match op_type.as_str() {
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
                        }
                    }
                    "store" => {
                        match op_type.as_str() {
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
                        }
                    }
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
                    "flow" => {
                        match op_type.as_str() {
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
                        }
                    }
                    "debug" => {
                        match op_type.as_str() {
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
                        }
                    }
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
                    produces_constant: None,
                });
                op_id += 1;
            }
        }
    }

    Ok(ops)
}

// =============================================================================
// Dependencies
// =============================================================================

/// Compute dependencies between ops
pub fn compute_dependencies(ops: &mut [Op]) {
    // Map: address -> last op that wrote to it
    let mut last_writer: HashMap<i64, usize> = HashMap::new();
    // Map: address -> ops that read since last write
    let mut readers_since_write: HashMap<i64, Vec<usize>> = HashMap::new();

    for i in 0..ops.len() {
        let reads = ops[i].reads.clone();
        let writes = ops[i].writes.clone();
        let op_id = ops[i].id;

        // RAW: depend on writers of values we read
        for addr in &reads {
            if let Some(&writer_id) = last_writer.get(addr) {
                ops[i].dep_ids.insert(writer_id);
            }
            readers_since_write
                .entry(*addr)
                .or_default()
                .push(op_id);
        }

        // WAR + WAW: depend on ops for values we write
        for addr in &writes {
            // WAW: depend on previous writer
            if let Some(&writer_id) = last_writer.get(addr) {
                ops[i].dep_ids.insert(writer_id);
            }
            // WAR: depend on all readers since last write
            if let Some(readers) = readers_since_write.get(addr) {
                for &reader_id in readers {
                    if reader_id != op_id {
                        ops[i].dep_ids.insert(reader_id);
                    }
                }
            }
            readers_since_write.remove(addr);
            last_writer.insert(*addr, op_id);
        }
    }
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
            if op.cycle == cycle as i32 && op.engine != Engine::ShiftedValu && op.engine != Engine::EliminatedBroadcast {
                emit_op_to_dict(py, op, &instr)?;
            }
        }

        // Emit expanded ALU ops from VALU→ALU shifting
        for op in expanded_ops {
            if op.cycle == cycle as i32 {
                emit_op_to_dict(py, op, &instr)?;
            }
        }

        if !instr.is_empty() {
            instrs.append(instr)?;
        }
    }

    Ok(instrs)
}

/// Helper to emit a single op to a dict
fn emit_op_to_dict<'py>(
    py: Python<'py>,
    op: &Op,
    instr: &Bound<'py, PyDict>,
) -> PyResult<()> {
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
#[pyo3(signature = (instrs, enable_valu_shift, verbose=false))]
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
#[pyo3(signature = (instrs, enable_valu_shift, verbose=false))]
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
#[pyo3(signature = (instrs, enable_valu_shift, verbose=false))]
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
// JSON Output
// =============================================================================

/// Format an address with prefix for readability
fn format_addr(val: i64, context: &str, slot_idx: usize) -> String {
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
        "+" | "-" | "*" | "/" | "%" | "&" | "|" | "^" | "<<" | ">>" | "==" | "!=" | "<"
        | "<=" | ">" | ">=" => format!("s{}", val),
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

#[derive(Serialize)]
struct CycleOutput {
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
struct ScheduleOutput {
    total_cycles: usize,
    total_ops: usize,
    best_mode: String,
    slot_limits: HashMap<String, usize>,
    cycles: Vec<CycleOutput>,
    summary: ScheduleSummary,
}

#[derive(Serialize)]
struct ScheduleSummary {
    load_utilization_pct: f64,
    store_utilization_pct: f64,
    alu_utilization_pct: f64,
    valu_utilization_pct: f64,
    total_empty_load: usize,
    total_empty_store: usize,
    total_empty_alu: usize,
    total_empty_valu: usize,
}

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
                            "+" | "-" | "*" | "/" | "%" | "&" | "|" | "^" | "<<" | ">>"
                            | "==" | "!=" | "<" | "<=" | ">" | ">=" => idx >= 2,
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

    // Build op_id -> cycle mapping for comments
    let mut op_id_to_cycle: HashMap<usize, i32> = HashMap::new();
    for op in &ops {
        op_id_to_cycle.insert(op.id, op.cycle);
    }
    for op in &best_expanded_ops {
        op_id_to_cycle.insert(op.id, op.cycle);
    }

    // Build reverse dependency map: op_id -> list of dependent op_ids
    let mut dependents_map: HashMap<usize, Vec<usize>> = HashMap::new();
    for op in ops.iter().chain(best_expanded_ops.iter()) {
        for &dep_id in &op.dep_ids {
            dependents_map.entry(dep_id).or_default().push(op.id);
        }
    }

    // Build cycle-by-cycle output
    let mut cycles_output: Vec<CycleOutput> = Vec::new();
    let mut total_empty_load = 0usize;
    let mut total_empty_store = 0usize;
    let mut total_empty_alu = 0usize;
    let mut total_empty_valu = 0usize;
    let mut total_load = 0usize;
    let mut total_store = 0usize;
    let mut total_alu = 0usize;
    let mut total_valu = 0usize;

    for cycle in 0..best_cycles {
        let mut load_ops: Vec<String> = Vec::new();
        let mut store_ops: Vec<String> = Vec::new();
        let mut alu_ops: Vec<String> = Vec::new();
        let mut valu_ops: Vec<String> = Vec::new();
        let mut flow_ops: Vec<String> = Vec::new();

        // Process regular ops
        for op in &ops {
            if op.cycle == cycle as i32 && op.engine != Engine::ShiftedValu && op.engine != Engine::EliminatedBroadcast {
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
                let formatted = format_op_with_comment(op, compact, &op_id_to_cycle, &dependents_map);

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

        // Process expanded ALU ops
        for op in &best_expanded_ops {
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
                let formatted = format_op_with_comment(op, compact, &op_id_to_cycle, &dependents_map);

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

    let total_load_slots = best_cycles * 2;
    let total_store_slots = best_cycles * 2;
    let total_alu_slots = best_cycles * 12;
    let total_valu_slots = best_cycles * 6;

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

    let output = ScheduleOutput {
        total_cycles: best_cycles,
        total_ops: ops.len() + best_expanded_ops.len(),
        best_mode: format!("{:?}", best_mode),
        slot_limits,
        cycles: cycles_output,
        summary,
    };

    let json = serde_json::to_string_pretty(&output).unwrap_or_else(|_| "{}".to_string());

    if !output_path.is_empty() {
        std::fs::write(&output_path, &json).map_err(|e| {
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
#[pyo3(signature = (instrs, enable_valu_shift, hot_threshold=3, verbose=false, skip_scheduling=false))]
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
        let max_cycle = ops.iter().map(|op| op.cycle.max(0) as usize).max().unwrap_or(0);
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

use crate::kernel_builder::{KernelConfig, TickTockKernelBuilder};
use rayon::prelude::*;

/// Format an op's compact string with an optional comment containing:
/// - const=N if produces_constant is set
/// - deps=[cycle1,cycle2,...] showing which cycles this op depends on
/// - dependents=[cycle1,cycle2,...] showing which cycles depend on this op
fn format_op_with_comment(
    op: &Op,
    compact: String,
    op_id_to_cycle: &HashMap<usize, i32>,
    dependents_map: &HashMap<usize, Vec<usize>>,
) -> String {
    let mut comment_parts: Vec<String> = Vec::new();

    // Add constant value if known
    if let Some(c) = op.produces_constant {
        comment_parts.push(format!("const={}", c));
    }

    // Add dependency cycles (sorted for deterministic output)
    let dep_cycles: BTreeSet<i32> = op
        .dep_ids
        .iter()
        .filter_map(|id| op_id_to_cycle.get(id).copied())
        .collect();
    if !dep_cycles.is_empty() {
        let cycles_vec: Vec<i32> = dep_cycles.into_iter().collect();
        comment_parts.push(format!("deps={:?}", cycles_vec));
    }

    // Add dependent cycles (sorted for deterministic output)
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

/// Write schedule output in JSON format (schedule_output.json format)
fn write_schedule_json(
    ops: &[Op],
    expanded_ops: &[Op],
    max_cycles: usize,
    best_mode: PriorityMode,
    output_path: &str,
) -> PyResult<()> {
    // Build op_id -> cycle mapping for all ops
    let mut op_id_to_cycle: HashMap<usize, i32> = HashMap::new();
    for op in ops {
        op_id_to_cycle.insert(op.id, op.cycle);
    }
    for op in expanded_ops {
        op_id_to_cycle.insert(op.id, op.cycle);
    }

    // Build reverse dependency map: op_id -> list of dependent op_ids
    let mut dependents_map: HashMap<usize, Vec<usize>> = HashMap::new();
    for op in ops.iter().chain(expanded_ops.iter()) {
        for &dep_id in &op.dep_ids {
            dependents_map.entry(dep_id).or_default().push(op.id);
        }
    }

    // Build cycle-by-cycle output
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

        // Process regular ops
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
                let formatted = format_op_with_comment(op, compact, &op_id_to_cycle, &dependents_map);

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

        // Process expanded ALU ops
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
                let formatted = format_op_with_comment(op, compact, &op_id_to_cycle, &dependents_map);

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

    let output = ScheduleOutput {
        total_cycles: max_cycles,
        total_ops: ops.len() + expanded_ops.len(),
        best_mode: format!("{:?}", best_mode),
        slot_limits,
        cycles: cycles_output,
        summary,
    };

    let json = serde_json::to_string_pretty(&output).unwrap_or_else(|_| "{}".to_string());

    std::fs::write(output_path, &json).map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!("Failed to write {}: {}", output_path, e))
    })?;

    Ok(())
}

/// Result of scheduling with a specific priority mode
struct ScheduleAttempt {
    mode: PriorityMode,
    cycles: usize,
    ops: Vec<Op>,
    expanded_ops: Vec<Op>,
    allocation_result: Option<AllocationResult>,
}

/// Apply register allocation to scheduled ops if enabled.
/// Returns the allocation result (or None if disabled).
fn apply_register_allocation(
    ops: &mut Vec<Op>,
    expanded_ops: &mut Vec<Op>,
    enable: bool,
    budget: usize,
    fixed_addrs: &HashSet<i64>,
) -> Option<AllocationResult> {
    if !enable {
        return None;
    }

    // Compute last uses for allocation
    compute_last_uses(ops);
    compute_last_uses(expanded_ops);

    // Combine ops for interval building (need to see all reads/writes)
    let mut all_ops: Vec<Op> = ops.iter().cloned().collect();
    all_ops.extend(expanded_ops.iter().cloned());

    // Run register allocation on combined ops
    let alloc_result = allocate_registers(&mut all_ops, budget, fixed_addrs);

    // Only apply mapping if within budget
    if !alloc_result.budget_exceeded {
        rewrite_addresses(ops, &alloc_result.addr_mapping);
        rewrite_addresses(expanded_ops, &alloc_result.addr_mapping);
    }

    Some(alloc_result)
}

/// Build a tick-tock kernel entirely in Rust and return scheduled instructions.
///
/// This eliminates the Python→Rust parsing overhead by generating ops directly
/// as Rust structs, then scheduling them and emitting back to Python.
///
/// Uses parallel multipass scheduling to try all priority modes concurrently.
///
/// Returns a list of scheduled instruction dicts, ready for execution.
///
/// If `output_json` is provided, writes the schedule in schedule_output.json format
/// with detailed slot utilization info per cycle.
///
/// If `enable_pseudo_consts` is true, uses flexible constant generation where the scheduler
/// dynamically chooses between Load, Add, Mul, etc. based on resource availability.
///
/// Encoding flags (only relevant when enable_pseudo_consts=true):
/// - `enable_add_imm`: Use add_imm via flow slot (default: true)
/// - `enable_const_load`: Use const via load slot (default: true)
/// - `enable_const_alu`: Use ALU derivation (Add, Sub, Mul, Shift) (default: true)
///
/// Slot filling mode (`enable_slot_filling=true`):
/// - Schedules non-const ops first, then fills empty slots with const generation
/// - Prefers Flow (add_imm) > Load (const) > ALU (derivation)
/// - More efficient than pre-scheduling pseudo-consts
///
/// Register allocation (`enable_register_allocation=true`):
/// - Post-scheduling optimization that rewrites addresses based on liveness
/// - Compacts scratch usage by reusing addresses when live ranges don't overlap
/// - Uses linear scan algorithm for efficient allocation
#[pyfunction]
#[pyo3(signature = (forest_height, n_nodes, batch_size, rounds, enable_valu_shift=true, verbose=false, output_json=None, enable_lazy_partition=false, partition_frequency=3, enable_pseudo_consts=false, enable_add_imm=true, enable_const_load=true, enable_const_alu=true, enable_slot_filling=false, max_alu_depth=None, enable_register_allocation=false, enable_tree_reduce_depth3=false))]
pub fn build_tick_tock_kernel<'py>(
    py: Python<'py>,
    forest_height: usize,
    n_nodes: usize,
    batch_size: usize,
    rounds: usize,
    enable_valu_shift: bool,
    verbose: bool,
    output_json: Option<String>,
    enable_lazy_partition: bool,
    partition_frequency: usize,
    enable_pseudo_consts: bool,
    enable_add_imm: bool,
    enable_const_load: bool,
    enable_const_alu: bool,
    enable_slot_filling: bool,
    max_alu_depth: Option<usize>,
    enable_register_allocation: bool,
    enable_tree_reduce_depth3: bool,
) -> PyResult<Bound<'py, PyList>> {
    let config = KernelConfig {
        forest_height,
        n_nodes,
        batch_size,
        rounds,
        enable_valu_shift,
        enable_debug: false,
        enable_lazy_partition,
        partition_frequency,
        enable_pseudo_consts,
        enable_add_imm,
        enable_const_load,
        enable_const_alu,
        max_alu_depth,
        enable_tree_reduce_depth3,
    };

    // Build the kernel (generates unscheduled ops)
    let mut builder = TickTockKernelBuilder::new(config.clone());

    // Get ops and optionally pseudo-consts
    let (mut ops, pseudo_consts) = if enable_pseudo_consts {
        builder.build_with_pseudo_consts()
    } else {
        (builder.build(), Vec::new())
    };

    if verbose {
        println!("Generated {} ops", ops.len());
        if !pseudo_consts.is_empty() {
            println!("Generated {} pseudo-consts", pseudo_consts.len());
        }
        println!("Scratch usage: {} / 1536", builder.alloc.usage());
    }

    if ops.is_empty() {
        return Ok(PyList::empty(py));
    }

    // Compute dependencies
    compute_dependencies(&mut ops);

    // Peephole optimization pass
    let peephole_config = PeepholeConfig::default();
    let peephole_stats = peephole_optimize(&mut ops, &peephole_config);

    if verbose && peephole_stats.total_eliminated > 0 {
        println!(
            "Peephole: {} ops eliminated (CSE: {}, algebraic: {}, zero: {}, DCE: {})",
            peephole_stats.total_eliminated,
            peephole_stats.cse_eliminated,
            peephole_stats.algebraic_simplified,
            peephole_stats.zero_ops_eliminated,
            peephole_stats.dce_eliminated
        );
    }

    // Recompute dependencies after peephole transformations
    compute_dependencies(&mut ops);

    // Compute critical path
    let critical_path = compute_critical_path(&ops);

    // Clone dep_ids for reuse
    let dep_ids_map: HashMap<usize, HashSet<usize>> = ops
        .iter()
        .map(|op| (op.id, op.dep_ids.clone()))
        .collect();

    // Fixed addresses that should not be remapped during register allocation
    let fixed_addrs: HashSet<i64> = [0i64].into_iter().collect(); // ZERO_REG

    // Run all priority modes in parallel using Rayon
    let results: Vec<ScheduleAttempt> = if enable_slot_filling {
        // Use post-scheduling slot filling for constants
        ALL_PRIORITY_MODES
            .par_iter()
            .map(|&mode| {
                let mut test_ops = ops.clone();
                for op in &mut test_ops {
                    op.cycle = -1;
                    op.dep_ids = dep_ids_map.get(&op.id).cloned().unwrap_or_default();
                }

                let result = schedule_with_slot_filling(
                    &mut test_ops,
                    &critical_path,
                    enable_valu_shift,
                    mode,
                );

                let mut final_ops = result.ops;
                let mut final_expanded = result.expanded_ops;
                let allocation_result = apply_register_allocation(
                    &mut final_ops,
                    &mut final_expanded,
                    enable_register_allocation,
                    1536,
                    &fixed_addrs,
                );

                ScheduleAttempt {
                    mode,
                    cycles: result.cycles,
                    ops: final_ops,
                    expanded_ops: final_expanded,
                    allocation_result,
                }
            })
            .collect()
    } else if enable_pseudo_consts && !pseudo_consts.is_empty() {
        // Use pseudo-const aware scheduler
        ALL_PRIORITY_MODES
            .par_iter()
            .map(|&mode| {
                let mut test_ops = ops.clone();
                for op in &mut test_ops {
                    op.cycle = -1;
                    op.dep_ids = dep_ids_map.get(&op.id).cloned().unwrap_or_default();
                }

                // Use schedule_with_pseudo_consts - start const registers at 2000 to avoid conflicts
                // After this call, test_ops contains both const ops and regular ops (all scheduled)
                let (result, _const_ops) = schedule_with_pseudo_consts(
                    &mut test_ops,
                    pseudo_consts.clone(),
                    &critical_path,
                    enable_valu_shift,
                    mode,
                    2000,  // const_register_start
                );

                let mut final_expanded = result.expanded_ops;
                let allocation_result = apply_register_allocation(
                    &mut test_ops,
                    &mut final_expanded,
                    enable_register_allocation,
                    1536,
                    &fixed_addrs,
                );

                // test_ops now contains all scheduled ops including const ops
                ScheduleAttempt {
                    mode,
                    cycles: result.cycles,
                    ops: test_ops,
                    expanded_ops: final_expanded,
                    allocation_result,
                }
            })
            .collect()
    } else {
        // Use standard greedy scheduler
        ALL_PRIORITY_MODES
            .par_iter()
            .map(|&mode| {
                let mut test_ops = ops.clone();
                for op in &mut test_ops {
                    op.cycle = -1;
                    op.dep_ids = dep_ids_map.get(&op.id).cloned().unwrap_or_default();
                }

                let result = schedule_greedy(&mut test_ops, &critical_path, enable_valu_shift, mode);

                let mut final_expanded = result.expanded_ops;
                let allocation_result = apply_register_allocation(
                    &mut test_ops,
                    &mut final_expanded,
                    enable_register_allocation,
                    1536,
                    &fixed_addrs,
                );

                ScheduleAttempt {
                    mode,
                    cycles: result.cycles,
                    ops: test_ops,
                    expanded_ops: final_expanded,
                    allocation_result,
                }
            })
            .collect()
    };

    // Find the best result
    let best = results
        .into_iter()
        .min_by_key(|r| r.cycles)
        .expect("No scheduling results");

    if verbose {
        println!("Best: {:?} with {} cycles (parallel multipass)", best.mode, best.cycles);
        if let Some(ref alloc) = best.allocation_result {
            let status = if alloc.budget_exceeded { " EXCEEDED BUDGET!" } else { "" };
            println!(
                "Register allocation: max addr {} -> {} (peak live: {}/{}, {} remapped, {} fixed){}",
                alloc.max_addr_before, alloc.max_addr_after, alloc.peak_live, alloc.budget,
                alloc.remapped_count, alloc.fixed_count, status
            );
        }
    }

    // Write JSON output if requested
    if let Some(ref path) = output_json {
        write_schedule_json(&best.ops, &best.expanded_ops, best.cycles, best.mode, path)?;
        if verbose {
            println!("Wrote schedule to {}", path);
        }
    }

    // Add pause instructions at start and end
    let scheduled = emit_scheduled_with_expanded(py, &best.ops, &best.expanded_ops, best.cycles)?;

    // Build final list with pause wrappers
    let result = PyList::empty(py);
    let pause_dict = PyDict::new(py);
    pause_dict.set_item("flow", PyList::new(py, &[PyTuple::new(py, &["pause".into_pyobject(py).unwrap().into_any()])?])?)?;
    result.append(&pause_dict)?;

    for item in scheduled.iter() {
        result.append(item)?;
    }

    result.append(&pause_dict)?;

    Ok(result)
}

/// Build a tick-tock kernel and return the number of cycles (for benchmarking).
///
/// This is a faster version that doesn't emit back to Python, just returns the cycle count.
/// Uses parallel multipass scheduling.
///
/// If `enable_pseudo_consts` is true, uses flexible constant generation where the scheduler
/// dynamically chooses between Load, Add, Mul, etc. based on resource availability.
///
/// Encoding flags (only relevant when enable_pseudo_consts=true):
/// - `enable_add_imm`: Use add_imm via flow slot (default: true)
/// - `enable_const_load`: Use const via load slot (default: true)
/// - `enable_const_alu`: Use ALU derivation (Add, Sub, Mul, Shift) (default: true)
/// - `max_alu_depth`: Max derivation depth for ALU constants (None = unlimited)
///   Values requiring deeper chains use Load instead. Recommended: 4-6 to prevent
///   long critical paths through constant derivation.
///
/// Slot filling mode (`enable_slot_filling=true`):
/// - Schedules non-const ops first, then fills empty slots with const generation
/// - Prefers Flow (add_imm) > Load (const) > ALU (derivation)
///
/// Register allocation (`enable_register_allocation=true`):
/// - Post-scheduling optimization that rewrites addresses based on liveness
/// - Compacts scratch usage by reusing addresses when live ranges don't overlap
#[pyfunction]
#[pyo3(signature = (forest_height, n_nodes, batch_size, rounds, enable_valu_shift=true, verbose=false, enable_lazy_partition=false, partition_frequency=3, enable_pseudo_consts=false, enable_add_imm=true, enable_const_load=true, enable_const_alu=true, enable_slot_filling=false, max_alu_depth=None, enable_register_allocation=false))]
pub fn build_tick_tock_kernel_ops(
    _py: Python<'_>,
    forest_height: usize,
    n_nodes: usize,
    batch_size: usize,
    rounds: usize,
    enable_valu_shift: bool,
    verbose: bool,
    enable_lazy_partition: bool,
    partition_frequency: usize,
    enable_pseudo_consts: bool,
    enable_add_imm: bool,
    enable_const_load: bool,
    enable_const_alu: bool,
    enable_slot_filling: bool,
    max_alu_depth: Option<usize>,
    enable_register_allocation: bool,
) -> PyResult<usize> {
    let config = KernelConfig {
        forest_height,
        n_nodes,
        batch_size,
        rounds,
        enable_valu_shift,
        enable_debug: false,
        enable_lazy_partition,
        partition_frequency,
        enable_pseudo_consts,
        enable_add_imm,
        enable_const_load,
        enable_const_alu,
        max_alu_depth,
        enable_tree_reduce_depth3: false, // Disabled - exceeds scratch budget
    };

    // Build the kernel (generates unscheduled ops)
    let mut builder = TickTockKernelBuilder::new(config.clone());

    // Get ops and optionally pseudo-consts
    let (mut ops, pseudo_consts) = if enable_pseudo_consts {
        builder.build_with_pseudo_consts()
    } else {
        (builder.build(), Vec::new())
    };

    if verbose {
        println!("Generated {} ops", ops.len());
        if !pseudo_consts.is_empty() {
            println!("Generated {} pseudo-consts", pseudo_consts.len());
        }
        println!("Scratch usage: {} / 1536", builder.alloc.usage());
    }

    if ops.is_empty() {
        return Ok(0);
    }

    // Compute dependencies
    compute_dependencies(&mut ops);

    // Peephole optimization pass
    let peephole_config = PeepholeConfig::default();
    let peephole_stats = peephole_optimize(&mut ops, &peephole_config);

    if verbose && peephole_stats.total_eliminated > 0 {
        println!(
            "Peephole: {} ops eliminated (CSE: {}, algebraic: {}, zero: {}, DCE: {})",
            peephole_stats.total_eliminated,
            peephole_stats.cse_eliminated,
            peephole_stats.algebraic_simplified,
            peephole_stats.zero_ops_eliminated,
            peephole_stats.dce_eliminated
        );
    }

    // Recompute dependencies after peephole transformations
    compute_dependencies(&mut ops);

    // Compute critical path
    let critical_path = compute_critical_path(&ops);

    // Clone dep_ids for reuse
    let dep_ids_map: HashMap<usize, HashSet<usize>> = ops
        .iter()
        .map(|op| (op.id, op.dep_ids.clone()))
        .collect();

    // Run all priority modes in parallel using Rayon
    let results: Vec<(PriorityMode, usize)> = if enable_slot_filling {
        // Use post-scheduling slot filling for constants
        ALL_PRIORITY_MODES
            .par_iter()
            .map(|&mode| {
                let mut test_ops = ops.clone();
                for op in &mut test_ops {
                    op.cycle = -1;
                    op.dep_ids = dep_ids_map.get(&op.id).cloned().unwrap_or_default();
                }

                let result = schedule_with_slot_filling(
                    &mut test_ops,
                    &critical_path,
                    enable_valu_shift,
                    mode,
                );
                (mode, result.cycles)
            })
            .collect()
    } else if enable_pseudo_consts && !pseudo_consts.is_empty() {
        // Use pseudo-const aware scheduler
        ALL_PRIORITY_MODES
            .par_iter()
            .map(|&mode| {
                let mut test_ops = ops.clone();
                for op in &mut test_ops {
                    op.cycle = -1;
                    op.dep_ids = dep_ids_map.get(&op.id).cloned().unwrap_or_default();
                }

                // Use schedule_with_pseudo_consts - start const registers at 2000 to avoid conflicts
                let (result, _const_ops) = schedule_with_pseudo_consts(
                    &mut test_ops,
                    pseudo_consts.clone(),
                    &critical_path,
                    enable_valu_shift,
                    mode,
                    2000,  // const_register_start
                );
                (mode, result.cycles)
            })
            .collect()
    } else {
        // Use standard greedy scheduler
        ALL_PRIORITY_MODES
            .par_iter()
            .map(|&mode| {
                let mut test_ops = ops.clone();
                for op in &mut test_ops {
                    op.cycle = -1;
                    op.dep_ids = dep_ids_map.get(&op.id).cloned().unwrap_or_default();
                }

                let result = schedule_greedy(&mut test_ops, &critical_path, enable_valu_shift, mode);
                (mode, result.cycles)
            })
            .collect()
    };

    // Find the best result
    let (best_mode, best_cycles) = results
        .into_iter()
        .min_by_key(|(_, cycles)| *cycles)
        .expect("No scheduling results");

    if verbose {
        println!("Best: {:?} with {} cycles (parallel multipass)", best_mode, best_cycles);
        if enable_register_allocation {
            println!("Note: Register allocation enabled but only affects address layout, not cycle count");
            println!("Use build_tick_tock_kernel() for full allocation statistics");
        }
    }

    Ok(best_cycles)
}
