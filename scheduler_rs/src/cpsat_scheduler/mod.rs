//! CP-SAT based scheduler using Google OR-Tools.
//!
//! This module provides an optimal scheduler using constraint programming.
//!
//! # Problem Model
//!
//! We model the VLIW scheduling problem as a resource-constrained scheduling problem:
//!
//! ## Decision Variables
//!
//! - **Cycle variable per op**: `cycle[i]` ∈ [ASAP[i], ALAP[i]] is the cycle when op i executes
//! - **Makespan**: `makespan` = max(cycle[i]) + 1 is the total number of cycles
//! - **Shift variables** (optional): For each shiftable VALU op, `shift[i]` ∈ {0, 1} where
//!   `shift[i] = 1` means shift to ALU (uses 8 ALU slots), `shift[i] = 0` means keep on VALU (uses 1 VALU slot).
//!   Both alternatives use the same `cycle[i]`, so switching doesn't require cycle changes.
//! - **ConstChoice variables** (optional): For each ConstChoice op, `const_choice[i]` ∈ {0, 1} where
//!   `const_choice[i] = 1` means use AddImm (Flow slot), `const_choice[i] = 0` means use Load (Load slot).
//!   Both alternatives use the same `cycle[i]`, so switching doesn't require cycle changes.
//!
//! ## Constraints
//!
//! 1. **Precedence constraints**: If op j depends on op i, then `cycle[j] >= cycle[i] + 1`
//!    (results are available the cycle after execution)
//!
//! 2. **Resource constraints (cumulative)**: Each engine has a limited number of slots per cycle,
//!    defined by [`SsaCpsatConfig::machine`] ([`crate::types::MachineConfig`]). Default capacities:
//!    - ALU: 12 slots/cycle
//!    - VALU: 6 slots/cycle
//!    - Load: 2 slots/cycle
//!    - Store: 2 slots/cycle
//!    - Flow: 1 slot/cycle
//!
//!    We model this with cumulative constraints: the sum of ops executing in any cycle
//!    must not exceed the engine's capacity. Override `config.machine` for tests or alternate machines.
//!
//!    **Optional intervals**: For ops with alternatives (Shift, ConstChoice), we use optional
//!    intervals that are conditionally present based on the decision variable. Both alternatives
//!    reference the same `cycle[i]`, ensuring that switching between alternatives (e.g., Load ↔ AddImm,
//!    VALU ↔ ALU) only changes the boolean decision variable, not the cycle assignment. This avoids
//!    cascading variable changes and makes optimization more efficient.
//!
//! 3. **Optional scratch memory constraint**: At most `scratch_budget` addresses can be live
//!    simultaneously (live = written but not yet consumed).
//!
//! ## Lower Bound Strengthening
//!
//! CP-SAT's cumulative constraint has a weak LP relaxation (the LP doesn't "see" that
//! ops can't share the same slot). We strengthen bounds explicitly:
//!
//! ### Capacity Bounds
//!
//! For each engine: `makespan >= ceil(num_ops / capacity)` where capacity is from
//! [`SsaCpsatConfig::machine`].
//!
//! Example: 9838 VALU ops with default VALU capacity 6 requires at least ceil(9838/6) = 1640 cycles.
//! This is often tighter than the critical path bound.
//!
//! ### Profile Bounds (Energy-Based)
//!
//! For each deadline d, count ops that must complete by d (i.e., ALAP <= d).
//! These ops require `ceil(count / capacity)` cycles, so:
//!   - If `ceil(count / capacity) > d + 1`, the schedule is infeasible
//!   - The tightest such bound across all deadlines is a valid lower bound
//!
//! This catches situations where ops are "bunched up" in tight ASAP/ALAP windows.
//!
//! ### Suffix Bounds
//!
//! For each starting cycle t, count ops with ASAP >= t (ops that can't start before t).
//! These ops need `ceil(count / capacity)` cycles starting from t, so:
//!   - `makespan >= t + ceil(count / capacity)`
//!
//! Example: If 9837 VALU ops have ASAP >= 1 and default VALU capacity 6, they need ceil(9837/6) = 1640 cycles
//! starting from cycle 1, giving makespan >= 1641.
//!
//! ### Sparse Window Bounds
//!
//! For each cycle t, count ops that CAN run at t (ASAP <= t <= ALAP).
//! Cycles with fewer than `capacity` available ops have "wasted" slots.
//! The bound accounts for this wasted capacity to find the minimum makespan.
//!
//! ## Objective
//!
//! Minimize makespan (total cycles). When makespan is fixed, we minimize a **tie-breaker**: the sum
//! of effective completion cycles over all ops (one per op; for choice vars we use the chosen path's
//! completion).
//!
//! **Philosophy of the tie-breaker:** "Early completion of computation is better." Even when we
//! cannot reduce cycles yet, preferring schedules that get more results done earlier is a
//! **smoother objective** than raw makespan alone: it steers the solver toward packings that
//! leave room to shrink the schedule (e.g. by removing empty cycles or freeing capacity). So the
//! tie-breaker is not only for breaking ties—it helps find paths that lead to cycle reduction.
//!
//! ## Empty cycles and the "bulk move" invariant
//!
//! When there is an empty cycle C (no op runs at C), every op at C+1 could run at C by precedence
//! (no dep runs at C, so no dep result is only available at C+1). So shifting **every** op at
//! C+1, C+2, … one cycle earlier (to C, C+1, …) preserves precedence, symmetry (cycle[i] ≤ cycle[j]),
//! and cumulative (same ops per cycle, just shifted); makespan can decrease by 1. Therefore **no
//! single hard constraint in the model should preclude that bulk move**. If any constraint did,
//! it would be a bug. The tie-breaker encourages the solver to find that solution; empty cycles
//! then only persist due to capacity lower bound (structural) or search.
//!
//! ## Debugging: makespan active set
//!
//! When `verbose` is true and a solution is found, we print which constraints on makespan are
//! **binding** (hold with equality): `[Makespan active set]` and a one-line summary after the
//! result. Use this to see why makespan is what it is:
//! - **Ops at last cycle**: which op(s) have `effective_last_cycle == max_cycle` (they "define"
//!   the end of the schedule). If only one op is binding, moving that op earlier (or relaxing
//!   its deps) could reduce makespan.
//! - **Capacity bounds binding**: whether `makespan == capacity_lower_bound` or one of the
//!   redundant linear bounds (capacity_lower_bound, load_bound, store_bound). If true,
//!   the structural capacity is limiting; improving choices (shift, ConstChoice, SelectChoice)
//!   may lower the bound.
//! - **Direct capacity constraints**: VALU/ALU/Flow/Load linear constraints that are tight.
//! Run with `uv run generate_ssa_kernel.py` (no `-q`) and search for `[Makespan active set]` or
//! `Active set:` in the output.
//!
//! ## Warm Start (Hints)
//!
//! We provide the greedy schedule as hints to help CP-SAT find good solutions quickly.
//! The greedy schedule is feasible, so CP-SAT can start from a known good point and
//! focus on proving optimality or finding improvements.
//!
//! ## Symmetry Breaking (from Emission Hints)
//!
//! Ops may carry a `symmetry_hint` (see `SymmetryHint` in `types`). The optimizer uses these
//! to add ordering constraints that reduce redundant search:
//!
//! - **Ordering groups**: Ops with the same `ordering_group` key get `cycle[i] <= cycle[j]`
//!   for consecutive ops in emission order. This prunes equivalent schedules where symmetric
//!   ops (e.g. same tree level) are reordered.
//!
//! - **Lexicographic streams**: When `lexicographic_stream` is set, we enforce
//!   `first_op[item_i] <= first_op[item_{i+1}]` for consecutive items in each stream. This
//!   prunes schedules where symmetric "items" (e.g. batch elements) are reordered.
//!
//! **Why we want this**: CP-SAT reports many orbits (e.g. 276); without breaking symmetry,
//! the solver explores equivalent branches. Emission encodes problem structure (e.g. tree
//! levels, batch items) into opaque keys; the optimizer only groups and orders—no
//! interpretation of rounds, batches, etc. See `docs/CPSAT_SLOW_PROGRESS_ANALYSIS.md` and
//! `docs/INTEGRATED_SCHEDULING_ALLOCATION.md` §5.5.
//!
//! # Build Requirements
//!
//! This module requires the `cpsat` feature and OR-Tools to be installed:
//! ```bash
//! brew install or-tools
//! export ORTOOLS_PREFIX=$(brew --prefix or-tools)
//! export RUSTFLAGS='-Clink-arg=-lprotobuf'
//! maturin develop --features cpsat
//! ```

// Re-imports from crate::types — needed so that test modules' `use super::*` works
// (in Rust, `use` items are private by default; `pub use` makes them visible to child modules)
#[allow(unused_imports)]
pub use std::collections::{HashMap, HashSet, VecDeque};
#[allow(unused_imports)]
pub use std::ops::Not;
#[allow(unused_imports)]
pub use std::str::FromStr;
#[allow(unused_imports)]
pub use crate::types::{Engine, MachineConfig, Op, Opcode, Operand, PrecedenceOffset, SsaOp, VReg, VLEN};

pub mod config;
pub mod cse_groups;
pub mod legacy;
pub mod validation;
pub mod bounds;
pub mod model;
pub(crate) mod symmetry_ordering;
#[cfg(test)]
mod tests;

// Re-exports: all public items from sub-modules
pub use config::{
    ConstChoiceMode, CpsatResult, CpsatStatus, OptionalConstraintGroup, SelectChoiceMode,
    ValuShiftMode,
};
pub use legacy::{compute_asap, compute_alap};
pub use cse_groups::{build_cse_groups, CseGroup, CseGroupResult};
pub use bounds::{
    AsapAlapTightening, CapacityLowerBoundResult, CpsatConfig, CpsatResultWithShift, LiveInterval,
    SsaCpsatConfig, SsaCpsatHints, SsaCpsatResult,
    compute_capacity_lower_bound_global, compute_live_intervals, compute_max_latency_critical_path,
    compute_ssa_alap, compute_ssa_asap,
};
#[cfg(feature = "cpsat")]
pub use bounds::compute_ssa_asap_alap_bounds;
pub use model::schedule_ssa_cpsat;

// Re-exports needed by tests (via `use super::*;` in test modules)
#[cfg(test)]
pub(crate) use bounds::{diagnose_precedence_bounds, fix_precedence_bounds_fixpoint};
#[cfg(all(test, feature = "cpsat"))]
pub(crate) use validation::validate_ssa_slot_limits;

/// Check if CP-SAT feature is available
pub fn cpsat_available() -> bool {
    cfg!(feature = "cpsat")
}
