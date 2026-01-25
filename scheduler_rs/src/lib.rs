//! Rust-based instruction scheduler for the performance evaluation kernel.
//!
//! This crate provides a high-performance greedy list scheduler with:
//! - Multiple priority modes (critical path, slack-based, engine balance, etc.)
//! - VALU→ALU shifting when VALU slots are saturated
//! - Constant derivation optimization
//! - Comprehensive diagnostics
//! - Native kernel building (eliminating Python→Rust parsing overhead)
//!
//! ## Module Organization
//!
//! - `types` - Core types: Op, SlotVal, ScheduleResult, slot limits
//! - `diagnostics` - Scheduling diagnostics and shared temp analysis
//! - `constants` - Constant analysis and derivation planning
//! - `scheduler` - Core scheduling algorithms
//! - `kernel_builder` - Native Rust kernel generation
//! - `python` - PyO3 bindings and Python interface

use pyo3::prelude::*;

pub mod constants;
pub mod diagnostics;
pub mod kernel_builder;
pub mod python;
pub mod scheduler;
pub mod types;

/// Python module registration
#[pymodule]
fn scheduler_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(python::schedule, m)?)?;
    m.add_function(wrap_pyfunction!(python::schedule_cycles, m)?)?;
    m.add_function(wrap_pyfunction!(python::schedule_multipass, m)?)?;
    m.add_function(wrap_pyfunction!(python::schedule_multipass_cycles, m)?)?;
    m.add_function(wrap_pyfunction!(python::count_cycles, m)?)?;
    // Diagnostic functions
    m.add_function(wrap_pyfunction!(python::run_diagnostics, m)?)?;
    m.add_function(wrap_pyfunction!(python::analyze_shared_temps, m)?)?;
    m.add_function(wrap_pyfunction!(python::print_diagnostics, m)?)?;
    m.add_function(wrap_pyfunction!(python::schedule_to_json, m)?)?;
    // Constant optimization functions
    m.add_function(wrap_pyfunction!(python::analyze_constants, m)?)?;
    m.add_function(wrap_pyfunction!(python::optimize_constants, m)?)?;
    // Kernel building functions
    m.add_function(wrap_pyfunction!(python::build_tick_tock_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(python::build_tick_tock_kernel_ops, m)?)?;
    Ok(())
}
