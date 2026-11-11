//! Rust-based instruction scheduler for the performance evaluation kernel.
//!
//! This crate provides a high-performance greedy list scheduler with:
//! - Multiple priority modes (critical path, slack-based, engine balance, etc.)
//! - VALU→ALU shifting when VALU slots are saturated
//! - Constant derivation optimization
//! - Comprehensive diagnostics
//! - Native kernel building (eliminating Python→Rust parsing overhead)
//! - SSA-based virtual registers for false dependency elimination
//!
//! ## Module Organization
//!
//! - `types` - Core types: Op, SsaOp, VReg, SlotVal, ScheduleResult, slot limits
//! - `diagnostics` - Scheduling diagnostics and shared temp analysis
//! - `constants` - Constant analysis and derivation planning
//! - `scheduler` - Core scheduling algorithms
//! - `address_assign` - SSA address assignment via linear scan allocation
//! - `cpsat_scheduler/` - CP-SAT model (config, bounds, validation, model, legacy)
//! - `kernel_builder/` - Native Rust kernel generation (tick_tock_ssa, ssa_greedy, emit)
//! - `python` - PyO3 bindings and Python interface

use pyo3::prelude::*;

pub mod address_assign;
pub mod constants;
pub mod cpsat_scheduler;
pub mod diagnostics;
pub mod kernel_builder;
pub mod peephole;
pub mod python;
pub mod schedule_constraints;
pub mod schedule_output;
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
    m.add_function(wrap_pyfunction!(python::cpsat_available, m)?)?;
    // SSA kernel building functions
    m.add_function(wrap_pyfunction!(python::build_tick_tock_kernel_ssa, m)?)?;
    m.add_function(wrap_pyfunction!(python::optional_constraint_group_names, m)?)?;
    m.add_function(wrap_pyfunction!(python::resolve_cpsat_tuning, m)?)?;
    m.add_class::<python::OptionalConstraintGroupPy>()?;
    // Expose each variant as a class attribute for Python (e.g. OptionalConstraintGroup.redundant_linear_bounds).
    let py = m.py();
    let cls = m.getattr("OptionalConstraintGroup")?;
    use crate::cpsat_scheduler::OptionalConstraintGroup;
    let set_attr = |name: &str, inner: OptionalConstraintGroup| -> PyResult<()> {
        let val = python::OptionalConstraintGroupPy { inner };
        cls.setattr(name, Py::new(py, val)?)
    };
    for group in OptionalConstraintGroup::ALL {
        set_attr(&group.to_string(), group)?;
    }
    m.add_function(wrap_pyfunction!(python::run_bulk_move_simulator_from_json, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    /// Optional/slow tests run only when `ci-tests` feature is enabled.
    /// Local (fast): `cargo test --features cpsat`
    /// Full (CI): `cargo test --features "cpsat ci-tests"`
    #[cfg(feature = "ci-tests")]
    #[test]
    fn test_ci_tests_feature_included() {
        assert!(true);
    }
}
