//! Python interface for the scheduler.
//!
//! Contains all PyO3 bindings, parsing, and emission functions.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use serde::Serialize;
use std::collections::{HashMap, HashSet};

use crate::constants::{
    analyze_constant_usage, build_constant_plan, emit_constant_generation_ops,
    print_constant_plan_summary,
};
use crate::diagnostics::{analyze_shared_temp_dependencies, SchedulingDiagnostics, SharedTempAnalysis};
use crate::scheduler::{
    compute_critical_path, schedule_greedy, schedule_greedy_with_diagnostics, PriorityMode,
    ALL_PRIORITY_MODES,
};
use crate::types::{Op, SlotVal, VLEN};

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

                // Store slot as native Rust values
                let mut slot_vec: Vec<SlotVal> = Vec::new();
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

                ops.push(Op {
                    id: op_id,
                    engine: engine.clone(),
                    slot: slot_vec,
                    cycle: cycle as i32,
                    dep_ids: HashSet::new(),
                    reads,
                    writes,
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
            if op.cycle == cycle as i32 && op.engine != "shifted_valu" && op.engine != "eliminated_broadcast" {
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
    let engine = &op.engine;

    // Get or create the list for this engine
    if !instr.contains(engine.as_str())? {
        instr.set_item(engine.as_str(), PyList::empty(py))?;
    }

    let slots = instr.get_item(engine.as_str())?.unwrap();
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
                    op.engine == "load"
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
            if op.cycle == cycle as i32 && op.engine != "shifted_valu" && op.engine != "eliminated_broadcast" {
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

                match op.engine.as_str() {
                    "load" => load_ops.push(compact),
                    "store" => store_ops.push(compact),
                    "alu" => alu_ops.push(compact),
                    "valu" => valu_ops.push(compact),
                    "flow" => flow_ops.push(compact),
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

                alu_ops.push(compact);
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
#[pyfunction]
#[pyo3(signature = (instrs, enable_valu_shift, hot_threshold=3, verbose=false))]
pub fn optimize_constants<'py>(
    py: Python<'py>,
    instrs: &Bound<'py, PyList>,
    enable_valu_shift: bool,
    hot_threshold: usize,
    verbose: bool,
) -> PyResult<Bound<'py, PyList>> {
    let mut ops = parse_ops(py, instrs)?;

    if ops.is_empty() {
        return Ok(PyList::empty(py));
    }

    let usage = analyze_constant_usage(&ops);

    if usage.is_empty() {
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
            op.engine == "load"
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
