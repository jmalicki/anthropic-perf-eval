# CP-SAT Scheduler Implementation Plan

This document provides a concrete implementation plan for adding an optional CP-SAT based scheduler to scheduler_rs. The CP-SAT scheduler will use Google's OR-Tools CP-SAT solver via the `jmalicki/cp_sat` Rust crate fork.

## Overview

The CP-SAT scheduler models instruction scheduling as a constraint satisfaction problem:
- **Decision variables**: Cycle assignment for each operation
- **Constraints**: Dependencies (precedence), slot limits (cumulative), optionally scratch budget
- **Objective**: Minimize makespan (total cycles)

## Why CP-SAT is Necessary

Scheduling with scratch budget constraints is equivalent to the **black pebbling game** on DAGs, which is **PSPACE-complete** (Gilbert, Lengauer, Tarjan 1980). This means:

1. **No polynomial-time greedy algorithm can guarantee finding a valid schedule** when one exists
2. Greedy heuristics may get stuck even when valid schedules exist
3. Only complete solvers (with backtracking) can guarantee finding solutions

Our approach:
- **Fast path**: Greedy scheduler with "lowest net_change first" heuristic
- **Fallback**: CP-SAT solver when greedy exceeds budget

CP-SAT is essentially "backtracking done right" - it uses conflict-driven clause learning (CDCL) and constraint propagation to efficiently search the exponential space.

See `docs/SCRATCH_PRESSURE.md` section "Theoretical Background: Pebbling Games" for details.

## Dependencies

### Rust Crate
```toml
# In scheduler_rs/Cargo.toml
[dependencies]
cp_sat = { git = "https://github.com/jmalicki/cp_sat.git", optional = true }

[features]
default = []
cpsat = ["cp_sat"]
```

### Build Environment
```bash
# OR-Tools must be installed via Homebrew
brew install or-tools

# Build with CP-SAT support
export ORTOOLS_PREFIX=$(brew --prefix or-tools)
export RUSTFLAGS='-Clink-arg=-lprotobuf'
cd scheduler_rs
maturin develop --features cpsat
```

## cp_sat Crate API Reference

The `jmalicki/cp_sat` fork provides these key APIs:

### Variables
```rust
// Integer variable with domain
let cycle = model.new_int_var([(lb, ub)]);
let cycle = model.new_int_var_with_name([(lb, ub)], "name");

// Boolean variable
let present = model.new_bool_var();
```

### Interval Variables (for cumulative constraints)
```rust
// Fixed-size interval: occupies [start, start+size)
let interval = model.new_fixed_size_interval_var(start_var, size_i64);

// Variable-size interval
let interval = model.new_interval_var(start, size, end);

// Optional interval (only enforced when literal is true)
let interval = model.new_optional_fixed_size_interval_var(start, size, is_present);
```

### Constraints
```rust
// Linear: lhs >= rhs, lhs == rhs, etc.
model.add_ge(cycle_consumer, LinearExpr::from(cycle_producer) + 1);
model.add_eq(expr1, expr2);
model.add_le(cycle, makespan);

// Cumulative: sum of demands for overlapping intervals <= capacity
model.add_cumulative(intervals, demands, capacity);

// No-overlap: intervals must be disjoint
model.add_no_overlap(intervals);
```

### Hints and Objective
```rust
// Provide warm-start hint
model.add_hint(var, value);

// Set objective
model.minimize(makespan);
model.maximize(expr);
```

### Solving
```rust
// Basic solve
let response = model.solve();

// Solve with parameters
let mut params = SatParameters::default();
params.max_time_in_seconds = Some(60.0);
params.log_search_progress = Some(true);
let response = model.solve_with_parameters(&params);

// Check status and extract solution
match response.status() {
    CpSolverStatus::Optimal | CpSolverStatus::Feasible => {
        let value = var.solution_value(&response);
    }
    _ => { /* no solution */ }
}
```

## Implementation Structure

### File: `scheduler_rs/src/cpsat_scheduler.rs`

```rust
//! CP-SAT based scheduler using Google OR-Tools.

#[cfg(feature = "cpsat")]
use cp_sat::builder::{CpModelBuilder, IntVar, IntervalVar, LinearExpr};
#[cfg(feature = "cpsat")]
use cp_sat::proto::{CpSolverStatus, SatParameters};

use std::collections::HashMap;
use crate::types::{Engine, Op};
```

### Data Structures

```rust
/// Result from CP-SAT scheduling
pub struct CpsatResult {
    /// Cycle assignment for each op (indexed by position in input slice)
    pub cycle_assignments: Vec<i32>,
    /// Total makespan (number of cycles)
    pub makespan: usize,
    /// Solver status
    pub status: CpsatStatus,
    /// Time spent solving (seconds)
    pub solve_time_secs: f64,
    /// Optimality gap (0.0 = proven optimal)
    pub gap: f64,
}

/// Solver status enum (mirrors CpSolverStatus)
pub enum CpsatStatus {
    Unknown,
    ModelInvalid,
    Feasible,
    Infeasible,
    Optimal,
}

/// Live interval for scratch memory tracking
struct LiveInterval {
    address: i64,
    def_op_idx: usize,      // Op index that writes this address
    last_use_op_idx: usize, // Op index that last reads this address
    size: i64,              // 1 for scalar, 8 for vector
}
```

### Core Functions

#### 1. ASAP/ALAP Computation
Reuse logic from `scheduler.rs` or implement standalone:

```rust
/// Compute earliest possible cycle for each op (forward pass)
pub fn compute_asap(ops: &[Op]) -> HashMap<usize, i32> {
    // Topological sort from sources
    // asap[op] = max(asap[dep] + 1 for dep in op.dep_ids)
}

/// Compute latest possible cycle for each op (backward pass)
pub fn compute_alap(ops: &[Op], horizon: i32) -> HashMap<usize, i32> {
    // Reverse topological sort from sinks
    // alap[op] = min(alap[consumer] - 1 for consumer of op)
}
```

#### 2. Live Interval Computation
```rust
/// Compute live intervals for scratch memory addresses
pub fn compute_live_intervals(ops: &[Op]) -> Vec<LiveInterval> {
    // For each address written by some op:
    //   - def_op_idx = index of op that writes it
    //   - last_use_op_idx = index of op that last reads it
    // Filter to intervals where def < last_use
}
```

#### 3. Main Scheduling Function

```rust
#[cfg(feature = "cpsat")]
pub fn schedule_cpsat(
    ops: &[Op],
    scratch_budget: usize,        // 0 = no constraint
    time_limit_secs: f64,
    greedy_hints: Option<&[i32]>, // Warm-start from greedy
    verbose: bool,
) -> Option<CpsatResult> {
    let n = ops.len();
    if n == 0 { return Some(empty_result()); }

    // 1. Compute bounds
    let asap = compute_asap(ops);
    let critical_path_len = asap.values().max().unwrap() + 1;
    let horizon = compute_horizon(critical_path_len, greedy_hints, n);
    let alap = compute_alap(ops, horizon);

    // 2. Build model
    let mut model = CpModelBuilder::default();

    // 3. Create cycle variables with [asap, alap] domains
    let op_id_to_idx: HashMap<usize, usize> = /* build mapping */;
    let cycle_vars: Vec<IntVar> = ops.iter().enumerate()
        .map(|(idx, op)| {
            let lb = asap[&op.id] as i64;
            let ub = alap[&op.id] as i64;
            model.new_int_var([(lb, ub)])
        })
        .collect();

    // 4. Add precedence constraints
    for (idx, op) in ops.iter().enumerate() {
        for &dep_id in &op.dep_ids {
            let dep_idx = op_id_to_idx[&dep_id];
            // cycle[idx] >= cycle[dep_idx] + 1
            model.add_ge(cycle_vars[idx], LinearExpr::from(cycle_vars[dep_idx]) + 1);
        }
    }

    // 5. Create interval variables for slot limits
    let intervals: Vec<IntervalVar> = cycle_vars.iter()
        .map(|&cv| model.new_fixed_size_interval_var(cv, 1))
        .collect();

    // 6. Add cumulative constraints per engine
    let ops_by_engine = group_ops_by_engine(ops);
    for (engine, op_indices) in &ops_by_engine {
        let capacity = engine.slot_limit();
        let engine_intervals: Vec<_> = op_indices.iter()
            .map(|&idx| intervals[idx])
            .collect();
        let demands = vec![1i64; op_indices.len()];
        model.add_cumulative(engine_intervals, demands, capacity as i64);
    }

    // 7. Optional: Add scratch budget constraint
    if scratch_budget > 0 {
        add_scratch_constraints(&mut model, ops, &cycle_vars, scratch_budget);
    }

    // 8. Add hints from greedy scheduler
    if let Some(hints) = greedy_hints {
        for (idx, &cycle) in hints.iter().enumerate() {
            if cycle >= 0 {
                model.add_hint(cycle_vars[idx], cycle as i64);
            }
        }
    }

    // 9. Add makespan variable and objective
    let makespan = model.new_int_var([(critical_path_len as i64, horizon as i64)]);
    for &cv in &cycle_vars {
        model.add_le(cv, LinearExpr::from(makespan) - 1);
    }
    model.minimize(makespan);

    // 10. Configure and solve
    let mut params = SatParameters::default();
    params.max_time_in_seconds = Some(time_limit_secs);
    if verbose {
        params.log_search_progress = Some(true);
    }

    let response = model.solve_with_parameters(&params);

    // 11. Extract solution
    extract_solution(&response, &cycle_vars, solve_time)
}
```

#### 4. Scratch Budget Constraints (Optional)

**Important**: The interval `[def_cycle, last_use_cycle)` models same-cycle reuse.
At `last_use_cycle`, the VReg is being consumed and its space becomes available
for other VRegs defined in the same cycle. See `docs/SCRATCH_PRESSURE.md` for
detailed timing semantics.

```rust
fn add_scratch_constraints(
    model: &mut CpModelBuilder,
    ops: &[Op],
    cycle_vars: &[IntVar],
    budget: usize,
) {
    let live_intervals = compute_live_intervals(ops);

    let mut scratch_intervals = Vec::new();
    let mut scratch_demands = Vec::new();

    for li in &live_intervals {
        let def_var = cycle_vars[li.def_op_idx];
        let last_use_var = cycle_vars[li.last_use_op_idx];

        // Size = last_use - def (same-cycle reuse: VReg freed at last_use, not after)
        let size_var = model.new_int_var([(1, horizon)]);
        model.add_eq(size_var, LinearExpr::from(last_use_var) - def_var);

        // End = last_use (half-open interval [def, last_use))
        let end_var = model.new_int_var([(1, horizon + 1)]);
        model.add_eq(end_var, LinearExpr::from(last_use_var));

        let interval = model.new_interval_var(def_var, size_var, end_var);
        scratch_intervals.push(interval);
        scratch_demands.push(li.size);
    }

    if !scratch_intervals.is_empty() {
        model.add_cumulative(scratch_intervals, scratch_demands, budget as i64);
    }
}
```

### Module Integration

#### File: `scheduler_rs/src/lib.rs`
```rust
#[cfg(feature = "cpsat")]
pub mod cpsat_scheduler;
```

#### File: `scheduler_rs/src/python.rs`
Add Python binding:

```rust
#[cfg(feature = "cpsat")]
#[pyfunction]
#[pyo3(signature = (instrs, scratch_budget=0, time_limit_secs=60.0, use_greedy_hint=true, verbose=false))]
pub fn schedule_cpsat<'py>(
    py: Python<'py>,
    instrs: &Bound<'py, PyList>,
    scratch_budget: usize,
    time_limit_secs: f64,
    use_greedy_hint: bool,
    verbose: bool,
) -> PyResult<(Bound<'py, PyList>, Bound<'py, PyDict>)> {
    // Parse ops, compute dependencies
    // Get greedy hints if use_greedy_hint
    // Call cpsat_scheduler::schedule_cpsat
    // Emit result back to Python
}

#[pyfunction]
pub fn cpsat_available() -> bool {
    cfg!(feature = "cpsat")
}
```

#### File: `scheduler_rs/src/lib.rs` (PyModule)
```rust
m.add_function(wrap_pyfunction!(python::schedule_cpsat, m)?)?;
m.add_function(wrap_pyfunction!(python::cpsat_available, m)?)?;
```

## Usage Examples

### From Rust
```rust
use scheduler_rs::cpsat_scheduler::schedule_cpsat;

let mut ops = build_kernel_ops();
compute_dependencies(&mut ops);

let greedy_cycles: Vec<i32> = ops.iter().map(|op| op.cycle).collect();

let result = schedule_cpsat(
    &ops,
    1536,           // scratch_budget
    60.0,           // time_limit_secs
    Some(&greedy_cycles),
    true,           // verbose
);

if let Some(r) = result {
    println!("CP-SAT: {} cycles ({:?})", r.makespan, r.status);
    for (idx, &cycle) in r.cycle_assignments.iter().enumerate() {
        ops[idx].cycle = cycle;
    }
}
```

### From Python
```python
import scheduler_rs

# Check availability
if scheduler_rs.cpsat_available():
    # Build unscheduled ops
    ops = build_kernel_ops()

    # Schedule with CP-SAT
    scheduled, result = scheduler_rs.schedule_cpsat(
        ops,
        scratch_budget=1536,
        time_limit_secs=60.0,
        use_greedy_hint=True,
        verbose=True
    )

    print(f"Status: {result['status']}, Makespan: {result['makespan']}")
    print(f"Solve time: {result['solve_time_secs']:.2f}s, Gap: {result['gap']:.1%}")
```

## Testing Plan

### Unit Tests (`scheduler_rs/src/cpsat_scheduler.rs`)

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_compute_asap() {
        // Chain: 0 -> 1 -> 2
        // Expected: asap = [0, 1, 2]
    }

    #[test]
    fn test_compute_alap() {
        // Chain: 0 -> 1 -> 2, horizon=5
        // Expected: alap = [2, 3, 4]
    }

    #[cfg(feature = "cpsat")]
    #[test]
    fn test_simple_chain() {
        // 3 ALU ops in chain
        // Expected: makespan = 3, respects precedence
    }

    #[cfg(feature = "cpsat")]
    #[test]
    fn test_parallel_ops() {
        // 4 independent ALU ops
        // Expected: makespan = 1 (all fit in 12 ALU slots)
    }

    #[cfg(feature = "cpsat")]
    #[test]
    fn test_slot_limits() {
        // 15 independent ALU ops
        // Expected: makespan >= 2 (12 slots per cycle)
    }

    #[cfg(feature = "cpsat")]
    #[test]
    fn test_multi_engine() {
        // Mix of Load (2 slots), ALU (12 slots), VALU (6 slots)
        // Verify each engine's cumulative constraint
    }

    #[cfg(feature = "cpsat")]
    #[test]
    fn test_greedy_hints() {
        // Verify hints improve solve time
    }
}
```

### Integration Tests

```python
# test_cpsat_scheduler.py

def test_cpsat_vs_greedy():
    """CP-SAT should match or beat greedy on small instances."""
    ops = build_small_kernel()

    greedy_result = scheduler_rs.schedule_multipass(ops, enable_valu_shift=True)
    greedy_cycles = count_cycles(greedy_result)

    if scheduler_rs.cpsat_available():
        _, cpsat_result = scheduler_rs.schedule_cpsat(ops, time_limit_secs=30.0)
        assert cpsat_result['makespan'] <= greedy_cycles

def test_cpsat_timeout():
    """CP-SAT should return feasible solution before timeout."""
    ops = build_large_kernel()

    if scheduler_rs.cpsat_available():
        _, result = scheduler_rs.schedule_cpsat(
            ops,
            time_limit_secs=5.0,
            use_greedy_hint=True
        )
        assert result['status'] in ['optimal', 'feasible']
```

## Lower Bound Strengthening

CP-SAT's cumulative constraint has a weak LP relaxation. The LP doesn't enforce that at most `capacity` intervals can overlap—it only sees a "fractional" view. This causes the solver to report overly optimistic lower bounds (e.g., critical path = 825 when the true optimum is ~1900).

We strengthen bounds explicitly with two techniques:

### Capacity Bounds

For each engine, we add the constraint:
```
makespan >= ceil(num_ops / capacity)
```

Example calculation:
- VALU: 9838 ops / 6 slots = ceil(9838/6) = 1640 cycles minimum
- Load: 2252 ops / 2 slots = ceil(2252/2) = 1126 cycles minimum
- ALU: 2687 ops / 12 slots = ceil(2687/12) = 224 cycles minimum

The VALU bound (1640) is much tighter than the critical path (825).

### Profile Bounds (Energy-Based)

For each engine and each deadline `d`, we count how many ops must complete by cycle `d` (i.e., have ALAP <= d). These ops must fit in `(d + 1) * capacity` total slots.

If the count exceeds this capacity:
```
mandatory_count > (d + 1) * capacity
```

Then the schedule is infeasible, and we can derive a tighter lower bound:
```
makespan >= ceil(mandatory_count / capacity)
```

This catches situations where ops are "bunched up" in tight ASAP/ALAP windows that the simple capacity bound misses.

### Impact

With these bounds:
- The solver's initial lower bound is much tighter
- Branch-and-bound prunes more aggressively
- Time to prove optimality is reduced

## Scalability Considerations

### Problem Size
- ~15,000 ops for the full kernel
- Each op: 1 cycle variable + 1 interval variable
- 5 cumulative constraints (one per hardware engine)
- ~85,000+ precedence constraints

### Mitigation Strategies

1. **Tight Bounds**: Use ASAP/ALAP to minimize variable domains, plus capacity/profile bounds
2. **Greedy Warm-Start**: Provide greedy solution as hint
3. **Time Limits**: Set reasonable timeout (60-300s)
4. **Incremental Approach**: If timeout, return best feasible solution found
5. **Early Stopping**: Stop when solution is within 2% of lower bound

### Expected Performance
- Small kernels (<1000 ops): Optimal in <10s
- Medium kernels (1000-5000 ops): Optimal or near-optimal in <60s
- Full kernel (~13000 ops): May need 5-10 minutes for optimal, but greedy hint should provide good feasible solution quickly

## Future Extensions

1. **Window Decomposition**: Split schedule into windows, solve each optimally
2. **Rolling Horizon**: Schedule first N cycles optimally, fix, repeat
3. **LNS (Large Neighborhood Search)**: Use CP-SAT's built-in LNS for large instances
4. **Parallel Workers**: Use `num_workers` parameter for multi-threaded search

## Build and Test Commands

```bash
# Build without CP-SAT (default)
cd scheduler_rs
maturin develop

# Build with CP-SAT
export ORTOOLS_PREFIX=$(brew --prefix or-tools)
export RUSTFLAGS='-Clink-arg=-lprotobuf'
maturin develop --features cpsat

# Run Rust tests
cargo test --features cpsat

# Run Python tests
pytest tests/test_cpsat_scheduler.py
```
