# Integrated Scheduling and Register Allocation

## Executive Summary

This document surveys approaches to simultaneous instruction scheduling and register allocation, with focus on constraint programming formulations suitable for our VLIW kernel optimizer. The key insight is that scheduling and allocation are interdependent: scheduling decisions affect register pressure (how many values are live simultaneously), while allocation constraints limit scheduling freedom.

**Environment Note:** OR-Tools is installed via Homebrew (`brew install or-tools`). For Rust integration, we use the [`cp_sat`](https://crates.io/crates/cp_sat) crate which provides native Rust bindings to the CP-SAT solver.

**Current State:**
- Greedy list scheduler with post-scheduling linear scan allocation
- Tree-reduce optimization exceeds scratch budget due to peak liveness
- Need integrated approach to enable aggressive optimizations

**Chosen Approach:** CP-SAT solver via `cp_sat` Rust crate (OR-Tools installed via Homebrew).

**Key Simplification:** We need *optimal scheduling* but only *feasible allocation* (not optimal). This means:
- CP-SAT minimizes cycles (scheduling objective)
- Cumulative constraint ensures peak liveness ≤ scratch budget (allocation feasibility)
- Post-scheduling linear scan assigns actual addresses (trivial once feasibility is guaranteed)

No diffn constraints, no address variables in the model—just scheduling variables + cumulative for capacity.

---

## 1. Problem Statement

### 1.1 The Phase-Ordering Problem

Traditional compilers separate instruction scheduling and register allocation into distinct phases:

```
Source → ... → Scheduling → Register Allocation → Code Generation
```

This creates a fundamental tension:

1. **Scheduling-first:** Maximizes ILP but may create high register pressure, causing spills
2. **Allocation-first:** Minimizes spills but constrains scheduling, reducing ILP

Neither ordering is universally optimal. The phases interact:
- Scheduling affects **live ranges**: reordering instructions changes when values are defined and last used
- Allocation affects **scheduling freedom**: if registers are scarce, ops may need specific orderings to avoid spills

### 1.2 Our Specific Challenge

In our VLIW kernel optimizer, the tree-reduce optimization (depth 3) fails due to scratch memory overflow:

```
Architecture constraints:
- Scratch space: 1536 words maximum
- Vector operations: 8 consecutive words each
- 32 groups × operations per group = high parallelism potential

Tree-reduce at depth 3:
- Creates additional intermediate values for parallel reduction
- Peak liveness exceeds 1536 words
- Current post-scheduling allocator cannot find valid assignment
```

The fundamental issue: our greedy scheduler optimizes for minimum cycles without considering register pressure. When peak liveness exceeds the scratch budget, no valid allocation exists.

### 1.3 Theoretical Complexity

Scheduling with scratch budget constraints is equivalent to the **black pebbling game** on DAGs. This problem is **PSPACE-complete** (Gilbert, Lengauer, Tarjan 1980), meaning:

- No polynomial-time algorithm can guarantee finding a valid schedule
- Greedy heuristics may fail even when valid schedules exist
- Complete solvers (with backtracking, like CP-SAT) are required for guaranteed solutions

See `docs/SCRATCH_PRESSURE.md` section "Theoretical Background: Pebbling Games" for detailed analysis, references, and our greedy + CP-SAT fallback approach.

### 1.4 Current Architecture

<svg viewBox="0 0 300 200" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="arrow1" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#333"/>
    </marker>
  </defs>

  <rect x="50" y="20" width="200" height="40" rx="5" fill="#e3f2fd" stroke="#1976d2" stroke-width="2"/>
  <text x="150" y="45" text-anchor="middle" font-family="monospace" font-size="12" fill="#333">scheduler.rs</text>
  <text x="150" y="58" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#666">(Greedy List Scheduler)</text>

  <line x1="150" y1="60" x2="150" y2="80" stroke="#333" stroke-width="2" marker-end="url(#arrow1)"/>

  <rect x="50" y="90" width="200" height="40" rx="5" fill="#f3e5f5" stroke="#7b1fa2" stroke-width="2"/>
  <text x="150" y="115" text-anchor="middle" font-family="monospace" font-size="12" fill="#333">allocator.rs</text>
  <text x="150" y="128" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#666">(Linear Scan Allocation)</text>

  <line x1="150" y1="130" x2="150" y2="150" stroke="#333" stroke-width="2" marker-end="url(#arrow1)"/>

  <rect x="50" y="160" width="200" height="40" rx="5" fill="#e8f5e9" stroke="#388e3c" stroke-width="2"/>
  <text x="150" y="185" text-anchor="middle" font-family="monospace" font-size="12" fill="#333">emit.rs</text>
  <text x="150" y="198" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#666">(Code Generation)</text>
</svg>

The scheduler uses 8 priority modes with parallel multipass:
- Critical path priority
- Reverse dependencies
- ASAP/ALAP ordering
- etc.

Post-scheduling allocation uses linear scan with live range analysis. If allocation fails (budget exceeded), the entire optimization is disabled.

---

## 2. Literature Survey

### 2.1 Foundational Work

#### Register Allocation via Graph Coloring
**Chaitin, G.J. et al. (1981). "Register Allocation via Coloring."**

The seminal paper establishing register allocation as graph coloring:
- Build **interference graph**: nodes = temporaries, edges = overlapping live ranges
- **k-coloring** the graph assigns k registers
- NP-complete in general, but effective heuristics exist

Key insight: two temporaries need different registers iff their live ranges overlap.

#### Optimal Allocation via Integer Programming
**Goodwin, D.W. & Wilken, K.D. (1996). "Optimal and Near-Optimal Global Register Allocation Using 0-1 Integer Programming."**

First ILP formulation for optimal register allocation:
- Binary variables for register assignment
- Constraints for interference, spill costs
- Objective: minimize spill code cost

Showed optimal solutions tractable for small-to-medium functions.

#### Optimal Scheduling via Integer Programming
**Wilken, K., Liu, J., Heffernan, M. (PLDI 2000). "Optimal Instruction Scheduling Using Integer Programming."**

ILP formulation for instruction scheduling:
- Variables: `x[i,t]` = 1 iff instruction i scheduled at cycle t
- Precedence constraints from dependencies
- Resource constraints from functional unit limits

Achieved optimal schedules for basic blocks up to ~1000 instructions.

### 2.2 Integrated Approaches

#### Unison: Constraint Programming for Compilers
**Castañeda Lozano, R. et al. (CP 2012, TOPLAS 2019). "Combinatorial Register Allocation and Instruction Scheduling."**

The Unison project is the most comprehensive integrated approach:
- **Constraint programming** model combining scheduling + allocation
- Decision variables for issue cycle, register assignment, instruction selection
- Global constraints: `cumulative` for resources, `diffn` for register packing
- Integrated into LLVM backend

Key contributions:
1. **Unified model**: scheduling and allocation solved together
2. **Copy coalescing**: integrated into the model
3. **Instruction selection**: alternative implementations as choices
4. **Scalability**: decomposition techniques for large functions

#### Survey Paper
**Castañeda Lozano, R. & Schulte, C. (2019). "Survey on Combinatorial Register Allocation and Instruction Scheduling." [arXiv:1409.7628]**

Comprehensive survey covering:
- ILP approaches (Goodwin-Wilken, Wilken et al.)
- Constraint programming (Unison)
- PBQP (Partitioned Boolean Quadratic Programming)
- Enumeration-based methods
- Hybrid approaches

Concludes that CP offers best balance of expressiveness and solver efficiency.

### 2.3 VLIW-Specific Work

#### ILP for VLIW Scheduling
**Eichenberger, A.E. et al. (PLDI 1998). "Minimum Register Requirements for a Modulo Schedule."**

Addresses register pressure in software pipelining:
- Compute minimum registers needed for a given schedule
- Trade-off between II (initiation interval) and register pressure
- Relevant for our loop-like unrolled structure

#### VLIW Compiler Optimization
**Leupers, R. & Marwedel, P. (1997). "Algorithms for Address Assignment in DSP Code Generation."**

Addresses assignment for DSP architectures:
- Similar scratch memory constraints
- Address generation interacts with scheduling
- Techniques for minimizing address arithmetic

### 2.4 Geometric Approaches

#### Register Allocation by Puzzle Solving
**Pereira, F.M.Q. & Palsberg, J. (PLDI 2008). "Register Allocation by Puzzle Solving."**

Novel formulation as 2D puzzle:
- Live ranges as puzzle pieces (time × register space)
- Allocation = fitting pieces without overlap
- Polynomial-time algorithm for certain architectures

Relevant to our diffn-based formulation.

---

## 3. Constraint Model Formulations

This section presents constraint formulations from the literature, then specializes them for our VLIW architecture.

### 3.1 Generic Decision Variables (Unison Model)

```
cycle[o]    : operation o → issue cycle (integer)
reg[t]      : temporary t → register/address (integer)
active[o]   : operation o → is scheduled? (boolean, for alternatives)
live[t,c]   : temporary t live at cycle c? (boolean, derived)
```

### 3.2 Key Constraint Patterns

**Precedence (Dependencies):**
```
cycle[consumer] >= cycle[producer] + latency[producer]
```

**Resource Capacity (Cumulative):**
```
For each resource r with capacity C:
  sum(demand[o] for o using r at cycle c) <= C, for all c
```

In CP, this is expressed via the `cumulative` global constraint with interval variables.

**Register Non-Interference (Diffn):**
```
For temporaries t1, t2 with overlapping live ranges:
  reg[t1] != reg[t2]  (if same size)

Or geometrically: rectangles in (register, cycle) space don't overlap
```

The `diffn` (non-overlapping rectangles) constraint handles this elegantly.

**All-Different for Interfering Temporaries:**
```
For set S of mutually interfering temporaries:
  all_different([reg[t] for t in S])
```

---

## 4. Concrete Formulation for Our VLIW Architecture

### 4.1 Architecture Parameters

```rust
// From our codebase (constants.rs, scheduler.rs)
const SLOT_LIMITS: [(Engine, usize); 5] = [
    (ALU, 12),
    (VALU, 6),
    (Load, 2),
    (Store, 2),
    (Flow, 1),
];

const SCRATCH_BUDGET: usize = 1536;  // Maximum live addresses
const VLEN: usize = 8;               // Vector length
const LATENCY: usize = 1;            // All ops: 1 cycle, result next cycle
```

### 4.2 Scheduling Decision Variables

Using the `cp_sat` Rust crate:

```rust
use cp_sat::builder::CpModelBuilder;
use cp_sat::proto::CpSolverStatus;

let mut model = CpModelBuilder::default();

// For each operation o:
// cycle[o] ∈ [asap[o], alap[o]]
let cycle: Vec<IntVar> = ops.iter().enumerate().map(|(i, op)| {
    model.new_int_var([(asap[i], alap[i])], format!("cycle_{}", i))
}).collect();

// Fixed-size interval for slot consumption (each op uses slot for 1 cycle)
let slot_interval: Vec<IntervalVar> = ops.iter().enumerate().map(|(i, _)| {
    model.new_fixed_size_interval_var(cycle[i], 1, format!("slot_{}", i))
}).collect();
```

**Why fixed-size intervals:** Each operation occupies its engine slot for exactly 1 cycle. Fixed-size intervals enable efficient cumulative propagation.

### 4.3 Slot/Resource Constraints

```rust
// Group operations by engine
let alu_ops: Vec<usize> = ops.iter().enumerate()
    .filter(|(_, op)| op.engine == Engine::ALU)
    .map(|(i, _)| i).collect();

let valu_ops: Vec<usize> = ops.iter().enumerate()
    .filter(|(_, op)| op.engine == Engine::VALU)
    .map(|(i, _)| i).collect();

// ... similarly for Load, Store, Flow

// Cumulative constraint per engine
model.add_cumulative(
    alu_ops.iter().map(|&i| slot_interval[i]).collect(),
    alu_ops.iter().map(|_| 1).collect(),  // unit demands
    12,  // ALU capacity
);

model.add_cumulative(
    valu_ops.iter().map(|&i| slot_interval[i]).collect(),
    valu_ops.iter().map(|_| 1).collect(),
    6,  // VALU capacity
);

// Load(2), Store(2), Flow(1) similarly
```

**Key insight:** Cumulative with fixed-size intervals and unit demands is equivalent to cardinality constraints per cycle, but propagates much more efficiently via specialized algorithms (edge-finding, theta-trees).

### 4.4 Dependency/Precedence Constraints

```rust
// dep_ids[o] contains IDs of ops that must complete before o
for (o, op) in ops.iter().enumerate() {
    for &dep in &dep_ids[o] {
        // Result available next cycle after producer completes
        model.add_ge(cycle[o], cycle[dep] + 1);
    }
}
```

**Critical path pre-computation** tightens variable domains:

```rust
// Forward pass: ASAP (earliest possible cycle)
let mut asap = vec![0; ops.len()];
for o in topo_order.iter() {
    asap[*o] = dep_ids[*o].iter()
        .map(|&d| asap[d] + 1)
        .max()
        .unwrap_or(0);
}

// Backward pass: ALAP (latest possible cycle)
let critical_path = *asap.iter().max().unwrap();
let mut alap = vec![critical_path; ops.len()];
for o in topo_order.iter().rev() {
    let succs: Vec<usize> = (0..ops.len())
        .filter(|&s| dep_ids[s].contains(o))
        .collect();
    if !succs.is_empty() {
        alap[*o] = succs.iter().map(|&s| alap[s] - 1).min().unwrap();
    }
}
```

### 4.5 Live Range Variables (for Feasibility Checking)

Since we only need *feasible* allocation (not optimal), we don't need address assignment variables in the CP model. We only need to track live ranges to check capacity.

**Liveness Model: Same-Cycle Reuse**

The liveness model allows **same-cycle reuse**: a VReg ending at cycle C can share storage with a VReg starting at cycle C. This is valid because within a VLIW cycle, reads happen before writes:

```text
Cycle 5:  [Read A] ... [Write B]
          ^^^^^^^       ^^^^^^^
          A is read     B is written

Storage can be reused: A dies, then B is born in the same cycle.
Intervals [0,5] and [5,10] do NOT overlap at cycle 5.
```

**Key implication for CP-SAT:** Interval end times should be `last_use_cycle` (not `last_use_cycle + 1`), and the cumulative constraint should use **exclusive end** semantics where intervals ending at time T do not contribute to capacity at time T.

**Live Range Intervals:**

```rust
// Each temporary has a live range from definition to last use (inclusive)
// Same-cycle reuse: intervals [0,5] and [5,10] do NOT overlap
let live_interval: Vec<IntervalVar> = temporaries.iter().enumerate().map(|(i, temp)| {
    let def_cycle = cycle[temp.def_op];
    let end_cycle = cycle[temp.last_use_op];  // NOT +1: same-cycle reuse model
    model.new_interval_var(
        def_cycle,
        end_cycle - def_cycle + 1,  // duration includes last use cycle
        end_cycle + 1,              // exclusive end for cumulative constraint
        format!("live_{}", i)
    )
}).collect();
```

**Note:** The cumulative constraint in CP-SAT uses half-open intervals `[start, end)`. A VReg last used at cycle 5 should have `end=6` in the interval variable so it contributes to capacity through cycle 5 but not cycle 6. This allows another VReg starting at cycle 5 to reuse the storage (they briefly coexist at cycle 5, but reads-before-writes makes this safe).

**No Address Variables Needed:** We don't create `addr[t]` variables in the CP model. The cumulative constraint checks that `Σ sizes ≤ budget` at every cycle—this guarantees a valid allocation *exists*. Actual address assignment happens post-scheduling with linear scan.

### 4.6 Register Capacity Constraints

**Our Goal:** Optimal scheduling with feasible allocation. We only need to ensure peak liveness ≤ scratch budget. Actual address assignment is done post-hoc with linear scan.

**The Right Constraint: Cumulative (Capacity Check Only)**

```rust
// Total live addresses at any cycle <= SCRATCH_BUDGET
model.add_cumulative(
    live_interval.clone(),
    temporaries.iter().map(|t| t.size as i64).collect(),  // 1 or 8
    SCRATCH_BUDGET as i64,
);
```

This ensures peak liveness never exceeds budget. The solver will find a schedule where allocation is *feasible*—we don't need it to find the actual allocation.

**Why NOT diffn/rectangle packing:** The `diffn` constraint (2D non-overlapping rectangles) would assign specific addresses to each temporary. This is:
- Unnecessary: linear scan allocation works once we have a feasible schedule
- Expensive: diffn adds O(n²) potential constraints
- Over-constrained: diffn requires non-overlapping even when temporaries could share addresses (one dies before another is born)

**Post-Scheduling Allocation:**
```rust
// After CP-SAT finds optimal schedule with feasible pressure:
let schedule = solver.extract_schedule();

// Simple linear scan assigns actual addresses
let allocation = linear_scan_allocate(&schedule, SCRATCH_BUDGET)
    .expect("CP-SAT ensured this is feasible");
```

### 4.7 Vector Operation Handling

Vector temporaries occupy 8 consecutive addresses:

```rust
for (i, temp) in temporaries.iter().enumerate() {
    if temp.is_vector {
        // addr[i] is base; elements at addr[i]..addr[i]+7
        model.add_le(addr[i] + 8, SCRATCH_BUDGET as i64);
    }
}

// In diffn formulation, vectors have width=8
```

**Vector broadcasts** (scalar → vector) create interesting interference:

```rust
// vbroadcast reads scalar src, writes vector dest
// The vector output interferes with any scalar live at the same time
// This is automatically handled by live range intervals
```

### 4.8 VALU→ALU Shifting

When VALU slots are saturated, ops can expand to 8 scalar ALU ops:

```rust
// Decision: should this VALU op be shifted to ALU?
let shift: Vec<BoolVar> = shiftable_valu_ops.iter().enumerate().map(|(i, _)| {
    model.new_bool_var(format!("shift_{}", i))
}).collect();

// If shifted: consumes 8 ALU slots instead of 1 VALU slot
for (i, &op_idx) in shiftable_valu_ops.iter().enumerate() {
    // VALU interval present only if NOT shifted
    let valu_int = model.new_optional_fixed_size_interval_var(
        cycle[op_idx], 1, shift[i].not(), format!("valu_opt_{}", i)
    );

    // 8 ALU intervals present only if shifted
    let alu_ints: Vec<IntervalVar> = (0..8).map(|j| {
        model.new_optional_fixed_size_interval_var(
            cycle[op_idx], 1, shift[i], format!("alu_shift_{}_{}", i, j)
        )
    }).collect();

    // Add to respective cumulative constraints
}
```

### 4.9 Objective Function

```rust
// Primary: minimize makespan (total cycles)
let makespan = model.new_int_var([(0, max_cycles)], "makespan");
for (i, _) in ops.iter().enumerate() {
    model.add_le(cycle[i] + 1, makespan);
}
model.minimize(makespan);

// Alternative: weighted multi-objective
// model.minimize(makespan * 1000 + num_spills * 100);
```

### 4.10 Implied Constraints (Strengthen Propagation)

```rust
// Lower bound from critical path
let critical_path_length = *asap.iter().max().unwrap();
model.add_ge(makespan, critical_path_length as i64);

// Lower bound from slot limits
let min_cycles_valu = (valu_ops.len() + 5) / 6;  // ceil division
model.add_ge(makespan, min_cycles_valu as i64);

// Similar for other engines
```

### 4.11 Search Strategy

```rust
// Prioritize scheduling variables over allocation
model.add_decision_strategy(
    cycle.clone(),
    VariableOrder::ChooseLowestMin,
    ValueOrder::SelectMinValue,
);

// For addresses: pack low (reuse freed registers)
model.add_decision_strategy(
    addr.clone(),
    VariableOrder::ChooseFirst,
    ValueOrder::SelectMinValue,
);
```

---

## 5. Solver-Friendly Formulations

### 5.1 Domain Initialization and Tightening

**Principle:** Smaller domains = faster solving. Pre-compute bounds before creating variables.

```rust
// GOOD: Tight domains from ASAP/ALAP analysis
let cycle = model.new_int_var([(asap[i], alap[i])], name);

// BAD: Full domain
let cycle = model.new_int_var([(0, max_cycles)], name);
```

**Impact:** For 13,000 ops with average domain reduction from 1000 to 50 cycles, search space reduces by factor of 20^13000.

### 5.2 Normal Forms and Canonical Representations

**Use Interval Variables (not raw integers):**

```rust
// BAD: Manual overlap checking
for c in 0..max_cycles {
    let count = alu_ops.iter()
        .map(|&o| model.new_bool_var_from_eq(cycle[o], c))
        .sum();
    model.add_le(count, 12);
}

// GOOD: Cumulative with intervals
model.add_cumulative(alu_intervals, demands, 12);
```

Cumulative has O(n log n) propagation via edge-finding; manual decomposition is O(n × max_cycles).

**Use OnlyEnforceIf for Conditionals:**

```rust
// BAD: Big-M formulation
model.add_ge(cycle[o], lower_bound - M * (1 - active[o]));

// GOOD: Reified constraint
model.add_ge(cycle[o], lower_bound).only_enforce_if(active[o]);
```

### 5.3 Decomposition Strategies

For our kernel with ~13,000 ops, joint solving may be intractable. CP solvers typically exhibit superlinear (often exponential) time growth with problem size. Decomposition trades optimality guarantees for tractability.

#### 5.3.1 Window-Based (Time) Decomposition

Partition ops into time windows based on ASAP bounds, solve sequentially:

```rust
let windows = partition_by_asap(&ops, window_size: 200);
for window in windows {
    let fixed_cycles = earlier_window_results;
    solve_window(&window, &fixed_cycles);
}
```

| Pros | Cons |
|------|------|
| Simple to implement | **Not optimal**: committing to early windows can preclude global optimum |
| Predictable runtime (bounded per window) | Window boundaries are arbitrary; good ops to group may span windows |
| Scales linearly with total ops | No backtracking across windows if later window fails |
| Works well when dependencies are mostly local | Requires tuning window size (too small = suboptimal, too large = slow) |

**Mitigation:** Use overlapping windows with "frozen" and "flexible" zones. Re-optimize flexible zone when solving next window.

**References:** [Decomposition Strategies for Job-shop Scheduling (arXiv:2205.07537)](https://arxiv.org/html/2205.07537v3)

#### 5.3.2 Phase Decomposition

Solve scheduling first (ignoring pressure), then check allocation feasibility:

```rust
// Phase 1: Optimize for cycles only
let schedule = solve_scheduling_only(&ops);

// Phase 2: Check feasibility
if peak_liveness(&schedule) <= SCRATCH_BUDGET {
    return schedule;  // Lucky: optimal schedule is feasible
}

// Phase 3: Re-solve with pressure constraint
solve_with_pressure_limit(&ops, SCRATCH_BUDGET);
```

| Pros | Cons |
|------|------|
| Fast when optimal schedule happens to be feasible | Phase 3 may be as hard as joint problem |
| Separates concerns cleanly | No information sharing between phases |
| Easy to implement incrementally | May solve the full problem twice |

**When it works well:** When resource constraints (slots) dominate and register pressure is secondary. Our kernel may fit this profile—slot limits are tight (especially VALU=6).

#### 5.3.3 Logic-Based Benders Decomposition (LBBD)

Iterate between master problem (scheduling) and subproblem (allocation feasibility), learning from failures:

```rust
loop {
    // Master: find optimal schedule ignoring detailed allocation
    let schedule = solve_master(&model);

    // Subproblem: check if allocation is feasible
    match check_allocation_feasibility(&schedule) {
        Feasible => return schedule,
        Infeasible { conflict_set } => {
            // Learn: add cut preventing this conflict pattern
            let cut = generate_benders_cut(&conflict_set);
            model.add(cut);
        }
    }
}
```

| Pros | Cons |
|------|------|
| **Provably optimal** under mild conditions | Requires designing good cuts (problem-specific) |
| Can be 2-3 orders of magnitude faster than monolithic | Cut generation logic is non-trivial |
| Master and subproblem can use different solvers | May iterate many times if cuts are weak |
| Naturally handles structure where parts have different difficulty | **Unbalanced decomposition hurts**: works best when master and subproblem are similarly difficult |

**Key insight from literature:** "LBBD is most effective when the planning and scheduling aspects of the problem are roughly balanced in difficulty." ([Hooker & Ottosson, 2003](https://link.springer.com/chapter/10.1007/11564751_58))

**For our problem:** Master = scheduling (hard), Subproblem = allocation feasibility (easy—just a sweep-line check). This imbalance suggests LBBD may not be the best fit. The subproblem is so fast that we might as well include cumulative constraints directly in the master.

**When LBBD shines:**
- Master: assignment/planning decisions (which machine, which route)
- Subproblem: detailed scheduling (when exactly)
- Both are NP-hard but decompose naturally

**References:**
- [Hooker, J.N. "Integrating Benders Decomposition Within Constraint Programming"](https://link.springer.com/chapter/10.1007/11564751_58)
- [Logic-based Benders decomposition for planning and scheduling](https://www.cambridge.org/core/journals/knowledge-engineering-review/article/abs/logicbased-benders-decomposition-for-planning-and-scheduling-a-computational-analysis/70EE77BFAE516C56F3363A90A3786D84)

#### 5.3.4 Recommendation for Our Problem

Given our specific structure:
- **Scheduling is hard** (13,000 ops, tight slot limits)
- **Allocation feasibility is easy** (O(n log n) sweep-line check)

**Best approach: Direct CP-SAT with cumulative for allocation**

Since checking allocation feasibility is cheap (cumulative constraint propagation), include it directly in the model rather than decomposing. The solver handles the interaction automatically.

If the monolithic model is too slow, use **window-based decomposition** as fallback:
1. Try full model with time limit (e.g., 60s)
2. If timeout, switch to window-based with 200-op windows
3. Accept suboptimal solution

```rust
match solve_full_model(&ops, Duration::from_secs(60)) {
    Some(schedule) => schedule,
    None => solve_with_windows(&ops, window_size: 200),  // Suboptimal fallback
}
```

### 5.4 Implied Constraints

Implied (redundant) constraints are logically entailed by existing constraints but help propagation by providing alternative reasoning paths.

```rust
// Example 1: Total slot usage (sum of per-engine limits)
let total_capacity = 12 + 6 + 2 + 2 + 1;  // = 23
model.add_cumulative(all_intervals, all_demands, total_capacity);

// Example 2: Write conflict (if two ops write same address, different cycles)
for (o1, o2) in same_address_writers {
    model.add_ne(cycle[o1], cycle[o2]);
}

// Example 3: Lower bound from engine bottleneck
let min_cycles_valu = (valu_ops.len() + 5) / 6;
model.add_ge(makespan, min_cycles_valu);
```

| Pros | Cons |
|------|------|
| **Faster propagation**: redundant constraints can detect infeasibility earlier via different reasoning | **Model bloat**: too many constraints slow down each propagation step |
| **Tighter bounds**: can prune values that individual constraints wouldn't catch | **Redundant work**: solver may waste time on constraints that rarely help |
| **Help the LP relaxation**: linear constraints improve the lower bound used in branch-and-bound | **Solver-dependent**: what helps one solver may hurt another |
| Cost is one-time at model construction | Can interfere with solver's learned nogoods |

**Guidelines:**
1. **Add implied constraints that span multiple constraint groups** (e.g., total slots combines all engines)
2. **Add lower bounds derived from relaxations** (e.g., min cycles from bottleneck engine)
3. **Avoid O(n²) implied constraints** unless they catch frequent conflicts
4. **Profile with and without** - implied constraints can hurt as much as help

**For our problem:** The total-slots cumulative is cheap and useful. We implement the capacity lower-bound redundant constraints (makespan ≥ ceil(num_ops_engine / capacity) per engine) in `scheduler_rs/src/cpsat_scheduler.rs` to strengthen the LP relaxation. Write-conflict constraints (O(n²) pairs) are probably not worth it since we don't have address variables.

### 5.5 Symmetry Breaking

Symmetry in CP models causes redundant search: the solver explores equivalent solutions that differ only in arbitrary choices (e.g., which of two identical ops gets scheduled first).

```rust
// Example 1: Fix first temporary to address 0 (if using address vars)
model.add_eq(addr[0], 0);

// Example 2: Order symmetric ops arbitrarily
for (o1, o2) in symmetric_pairs {
    if engine[o1] == engine[o2] && asap[o1] == asap[o2] {
        model.add_le(cycle[o1], cycle[o2]);  // Break tie: o1 ≤ o2
    }
}

// Example 3: Lexicographic ordering of interchangeable groups
// If groups 0,1,2 are symmetric, enforce group_start[0] ≤ group_start[1] ≤ group_start[2]
```

| Pros | Cons |
|------|------|
| **Exponential speedup** in highly symmetric problems | **Can break the model** if constraints are too strong (remove valid solutions) |
| Reduces search space without losing optimal solution | **Subtle to get right**: easy to accidentally over-constrain |
| Works well with static symmetries (known at model construction) | **Interferes with dynamic symmetry breaking** that solvers do internally |
| Zero runtime cost (constraints added once) | **Less effective for optimization**: solver may not explore symmetric region anyway if it has a good bound |

**Types of symmetry in our problem:**
1. **Group symmetry**: 32 groups do identical work; interchangeable in principle
2. **Op symmetry within cycle**: ops scheduled in same cycle can be reordered
3. **Register symmetry**: (if using address vars) addresses are interchangeable

**Caution:** CP-SAT has built-in symmetry detection. Adding manual symmetry breaking may conflict with or duplicate the solver's efforts. Test empirically.

**For our problem:** Limited symmetry breaking is safe:
- Groups are NOT symmetric (they operate on different data)
- Ops within a cycle are symmetric only if they have identical structure
- We don't use address variables, so no register symmetry

**Recommendation:** Add symmetry breaking only if profiling shows the solver exploring symmetric branches. Start without it.

### 5.6 Solver Parameters and Diagnostics

CP-SAT provides rich diagnostic output that helps inform formulation decisions on short runs.

#### Enabling Diagnostics

```rust
let mut solver = CpSolver::new();

// CRITICAL: Enable search progress logging
solver.parameters.log_search_progress = true;

// Log to file for analysis
solver.parameters.log_to_stdout = false;
// (In Rust, capture stderr or use log callback)

// Time limit for quick diagnostic runs
solver.set_time_limit(Duration::from_secs(30));

// Parallelism
solver.set_num_workers(8);

let status = solver.solve(&model);

// After solving, extract statistics
let response = solver.response();
println!("Status: {:?}", response.status);
println!("Objective: {}", response.objective_value);
println!("Best bound: {}", response.best_objective_bound);
println!("Conflicts: {}", response.num_conflicts);
println!("Branches: {}", response.num_branches);
println!("Propagations: {}", response.num_propagations);
println!("Wall time: {}s", response.wall_time);
println!("Gap integral: {}", response.gap_integral);
```

#### Understanding the Search Log

The log format is: `EVENT | TIME | BEST_SOLUTION | SEARCH_RANGE | COMMENT`

Example log output:
```
#1       0.02s best:2100  next:[1764,2099]  quick_restart_no_lp
#2       0.15s best:1950  next:[1764,1949]  default_lp
#Bound   0.23s best:1950  next:[1780,1949]  lp_relaxation
#3       0.45s best:1820  next:[1780,1819]  probing
#Bound   0.52s best:1820  next:[1795,1819]  timetable_edge_finding
...
```

#### Key Metrics to Watch

| Metric | What it tells you | What to do if bad |
|--------|-------------------|-------------------|
| **Gap = (best - bound) / best** | How far from proven optimal | If gap stuck high: add implied constraints, try decomposition |
| **Conflicts/sec** | Search intensity | Low = propagation-heavy (good). High = thrashing (bad) |
| **Bound improvement rate** | Is lower bound tightening? | If stuck: enable timetable_edge_finding, add redundant cumulative |
| **Solution improvement rate** | Is solver finding better solutions? | If stuck: try different search strategy, warm-start from greedy |
| **Gap integral** | Area under gap curve over time | Lower = faster convergence. Compare across formulations |
| **Presolve reductions** | How much model simplification | Large reductions = model has redundancy (usually fine) |

#### Diagnostic Questions for Quick Runs

Run for 30-60 seconds and examine the log to answer:

1. **Is the model feasible?**
   - If `INFEASIBLE` quickly: check constraints, may have bug
   - If no solution found: problem may be very constrained

2. **Is propagation effective?**
   - Look for `#Bound` events from `timetable_edge_finding`, `cumulative`
   - If bounds only improve from `lp_relaxation`: cumulative propagation may not be helping

3. **Are solutions improving?**
   - Frequent `#N` events with decreasing objectives = good
   - Long gaps between solutions = search is stuck

4. **Which subsolver is winning?**
   - Log shows which strategy found each solution/bound
   - If one dominates, consider tuning parameters for that strategy

5. **Is the LP relaxation tight?**
   - Compare `lp_relaxation` bound to final objective
   - Loose LP = more search needed, tight LP = faster proving

#### Example: Comparing Formulations

```bash
# Run A: baseline formulation
./scheduler --cp-sat --log=log_baseline.txt --time-limit=60

# Run B: with implied constraints
./scheduler --cp-sat --implied-constraints --log=log_implied.txt --time-limit=60

# Run C: with timetable edge-finding
./scheduler --cp-sat --edge-finding --log=log_ef.txt --time-limit=60

# Compare gap integrals and final gaps
grep "gap_integral" log_*.txt
```

#### Log Analysis Tools

- **[cpsat-logutils](https://pypi.org/project/cpsat-logutils/)**: Python library to parse logs into structured objects
- **[CP-SAT Log Analyzer](https://cpsat-log-analyzer.streamlit.app/)**: Web GUI for visualizing search progress
- Both work with logs from any language (C++/Rust/Python) as long as `log_search_progress=true`

**References:**
- [Understanding the Log - CP-SAT Primer](https://d-krupke.github.io/cpsat-primer/understanding_the_log.html)
- [Parameters - CP-SAT Primer](https://d-krupke.github.io/cpsat-primer/05_parameters.html)

### 5.7 Warm-Starting from Greedy Solutions

CP-SAT supports **solution hints** - initial variable assignments that guide the search. Our fast greedy scheduler can provide these hints, potentially giving CP-SAT a significant head start.

#### The `add_hint()` API

```rust
// After building the model, before solving:
let greedy_schedule = greedy_scheduler.schedule(&ops);

for (op_id, cycle_value) in greedy_schedule.iter() {
    model.add_hint(cycle[op_id], cycle_value as i64);
}

// Now solve - CP-SAT will use hints to guide search
let status = solver.solve(&model);
```

#### How Hints Help

| Benefit | Mechanism |
|---------|-----------|
| **Immediate upper bound** | If hint is feasible, solver knows objective ≤ hint_value from start |
| **LNS warm start** | Large Neighborhood Search workers use hint as starting point |
| **Better branching** | Hint values inform variable/value selection heuristics |
| **Faster first solution** | Solver can verify hint feasibility quickly, then improve |

The log shows `[hint]` when a hint-derived solution is found:
```
#1       0.05s best:1850  next:[1764,1849]  [hint]
```

#### When Hints Help Most

| Scenario | Hint Value |
|----------|------------|
| **Hint is optimal or near-optimal** | Huge win - solver proves optimality quickly |
| **Hint is feasible but suboptimal** | Good - provides starting point for LNS improvement |
| **Hint is infeasible** | Still useful - guides search toward good regions |
| **Hint is far from optimal** | May hurt - can mislead search initially |

#### Our Greedy Scheduler as Hint Source

Our existing greedy scheduler is ideal for generating hints:

```rust
// Fast: O(n log n) greedy schedule
let greedy_result = greedy_schedule(&ops, PriorityMode::CriticalPath);

// Check if it's allocation-feasible
let peak_liveness = compute_peak_liveness(&greedy_result);

if peak_liveness <= SCRATCH_BUDGET {
    // Greedy found a valid solution! Hint it and let CP-SAT try to improve
    for (op_id, cycle) in &greedy_result {
        model.add_hint(cycle_var[*op_id], *cycle as i64);
    }
    // Also hint the objective
    model.add_hint(makespan, greedy_result.makespan() as i64);
} else {
    // Greedy solution violates allocation - hint anyway, CP-SAT will fix
    // The cycle assignments are still a good starting point
    for (op_id, cycle) in &greedy_result {
        model.add_hint(cycle_var[*op_id], *cycle as i64);
    }
}
```

#### Hint Quality vs. Solver Internals

**Question:** Can our greedy scheduler beat CP-SAT's internal heuristics?

**Answer:** Often yes, for domain-specific reasons:

| Our Greedy | CP-SAT's Generic Heuristics |
|------------|----------------------------|
| Knows about VLIW slot semantics | Sees abstract cumulative constraints |
| Uses critical-path priority (domain knowledge) | Uses generic variable/value ordering |
| O(n log n) focused on scheduling | Explores broader search space |
| 2 seconds for 13k ops | May spend minutes finding first solution |

**However:** CP-SAT's LNS is very good at *improving* solutions. The best strategy is:
1. **Greedy provides initial solution** (fast, domain-aware)
2. **CP-SAT improves it** (powerful local search)
3. **CP-SAT proves optimality** (or bounds the gap)

#### Partial Hints

You don't need to hint every variable. Partial hints work too:

```rust
// Only hint the "hard" variables (e.g., VALU ops that are bottleneck)
for op_id in valu_ops {
    model.add_hint(cycle_var[op_id], greedy_result[op_id] as i64);
}
// Let CP-SAT figure out ALU ops (more slack, less constrained)
```

#### Diagnostic: Is the Hint Helping?

Compare runs with and without hints:

```bash
# Without hint
./scheduler --cp-sat --no-hint --time-limit=60 --log=log_no_hint.txt

# With greedy hint
./scheduler --cp-sat --greedy-hint --time-limit=60 --log=log_hint.txt

# Compare time to first solution and gap integral
```

Look for:
- `[hint]` tag in log (hint was used)
- Earlier first solution with hint
- Lower gap integral (faster convergence)

**References:**
- [Solution Hints - CP-SAT Primer](https://d-krupke.github.io/cpsat-primer/)
- [OR-Tools add_hint() API](https://or-tools.github.io/docs/pdoc/ortools/sat/python/cp_model.html)

---

## 6. Propagation Algorithms: Edge-Finding and Theta-Trees

The cumulative constraint admits specialized propagation algorithms that are far more efficient than naive decomposition. This section covers these algorithms, what OR-Tools implements, and whether standalone implementation is tractable.

### 6.1 The Cumulative Constraint

The cumulative constraint ensures that at any point in time, the sum of resource demands does not exceed capacity:

```
∀t: Σ{demand[i] : task i active at time t} ≤ capacity
```

A naive implementation checks this for every time point—O(n × T) where T is the time horizon. For n=13,000 ops and T=2,000 cycles, this is 26 million checks per propagation.

### 6.2 Edge-Finding Algorithm

**Key Insight:** Instead of checking every time point, reason about *sets* of tasks and their aggregate energy requirements.

**Edge-Finding Rule:** If a set Ω of tasks requires so much energy that task i cannot start before all tasks in Ω complete, then we can tighten i's earliest start time.

Formally: If `energy(Ω) + demand[i] × min_duration[i] > capacity × (latest_end(Ω) - earliest_start[i])`, then task i cannot start before some task in Ω finishes.

**Original Complexity:**
- Carlier & Pinson (1989, 1994): O(n³) for unary resources
- Nuijten (1994): O(n²k) for cumulative, where k = distinct demand values
- Mercier & Van Hentenryck (2008): Corrected O(n²k) algorithm fixing invalid dominance rule in Nuijten's version

**References:**
- Carlier, J. & Pinson, E. (1989). "An Algorithm for Solving the Job-Shop Problem." *Management Science*, 35(2):164-176.
- Nuijten, W.P.M. (1994). *Time and Resource Constrained Scheduling*. PhD thesis, Eindhoven University of Technology.
- [Mercier, L. & Van Hentenryck, P. (2008). "Edge Finding for Cumulative Scheduling." *INFORMS Journal on Computing*, 20(1):143-153.](https://dl.acm.org/doi/10.1287/ijoc.1070.0226)

### 6.3 Theta-Trees (Θ-Trees)

**Key Insight:** Use a balanced binary tree to efficiently compute aggregates needed for edge-finding.

The Θ-tree is a data structure introduced by Vilím that stores tasks and supports:
- Insert/remove a task: O(log n)
- Query total energy of stored tasks: O(1)
- Query earliest completion time (ECT) of stored tasks: O(1)

By maintaining a Θ-tree during edge-finding, the algorithm achieves:
- **O(n log n)** for unary resources (Vilím, 2004)
- **O(kn log n)** for cumulative resources (Vilím, 2009), where k = distinct demand values

**The Θ-Λ-Tree Extension:** Adds a secondary "gray" set Λ for tasks whose membership is conditional, enabling not-first/not-last reasoning in the same complexity.

**References:**
- [Vilím, P. (2007). *Global Constraints in Scheduling*. PhD thesis, Charles University in Prague.](http://vilim.eu/petr/disertace.pdf)
- Vilím, P. (2004). "O(n log n) Filtering Algorithms for Unary Resource Constraint." *CPAIOR 2004*, LNCS 3011, pp. 335-347.
- [Vilím, P. (2009). "Edge Finding Filtering Algorithm for Discrete Cumulative Resources in O(kn log n)." *CP 2009*, LNCS 5732, pp. 802-816.](https://link.springer.com/chapter/10.1007/978-3-642-04244-7_62)

### 6.4 Timetable Edge-Finding

Combines mandatory part reasoning (timetabling) with edge-finding:
- **Timetabling:** Some portion of each task *must* execute in a fixed interval (when ASAP end > ALAP start)
- **Timetable Edge-Finding:** Uses mandatory parts to strengthen edge-finding deductions

**Complexity:** O(n²) but stronger propagation than pure edge-finding.

**Reference:**
- [Vilím, P. (2011). "Timetable Edge Finding Filtering Algorithm for Discrete Cumulative Resources." *CPAIOR 2011*, LNCS 6697, pp. 230-245.](https://link.springer.com/chapter/10.1007/978-3-642-21311-3_22)

### 6.5 What OR-Tools CP-SAT Implements

OR-Tools provides several propagation levels for the cumulative constraint, controlled by solver parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_timetable_edge_finding_in_cumulative` | false | Enable timetable edge-finding (stronger but slower) |
| `max_num_intervals_for_timetable_edge_finding` | 100 | Max intervals before disabling (0 = always disabled) |
| `use_overload_checker_in_cumulative` | false | Enable overload checking (energy-based reasoning) |

**Default behavior:** Basic cumulative propagation without edge-finding. The solver relies on its SAT-based search rather than heavy propagation.

**For scheduling problems:** Enabling timetable edge-finding can significantly reduce search but may slow down each propagation. Tune based on problem characteristics.

```rust
// Enable stronger propagation for scheduling-heavy problems
solver.parameters.use_timetable_edge_finding_in_cumulative = true;
solver.parameters.max_num_intervals_for_timetable_edge_finding = 500;
solver.parameters.use_overload_checker_in_cumulative = true;
```

### 6.6 Why CP-SAT (Not Standalone Implementation)

**Question:** Could we implement edge-finding ourselves without OR-Tools?

**Assessment:**

| Aspect | Difficulty | Notes |
|--------|------------|-------|
| Basic edge-finding (O(n²k)) | Moderate | ~500-800 lines, well-documented in papers |
| Theta-tree (O(n log n)) | Hard | ~300-500 lines for the data structure, subtle invariants |
| Full cumulative propagator | Very Hard | Need to combine multiple rules, handle backtracking |
| Integration with search | Very Hard | Need backtracking, nogood learning, variable selection |

**Verdict:** Edge-finding alone only provides *propagation*—it prunes impossible values but doesn't find optimal solutions. For optimal scheduling we need:
1. **Search:** Systematic exploration of the remaining search space
2. **Backtracking:** Undo assignments when reaching dead ends
3. **Learning:** Avoid repeating failed subproblems (nogoods)
4. **Restarts:** Escape from bad search regions

Building a competitive solver from scratch is a multi-year effort. **We use CP-SAT.**

**Our Chosen Approach:**

| Requirement | Solution |
|-------------|----------|
| Optimal scheduling | CP-SAT with cumulative constraints per engine |
| Feasible allocation | Cumulative constraint on live ranges (capacity check) |
| Actual address assignment | Post-hoc linear scan (already implemented) |

### 6.7 Simpler Alternatives for Feasibility Checking

If you only need to check whether a schedule respects resource limits (not find an optimal one), simpler O(n log n) algorithms exist:

**Sweep-Line Algorithm:**
```rust
// Check if schedule respects capacity
fn check_cumulative_feasibility(tasks: &[(start, end, demand)], capacity: i64) -> bool {
    // Create events: +demand at start, -demand at end
    let mut events: Vec<(i64, i64)> = Vec::new();
    for &(start, end, demand) in tasks {
        events.push((start, demand));
        events.push((end, -demand));
    }
    events.sort_by_key(|(time, _)| *time);

    let mut current_load = 0i64;
    for (_, delta) in events {
        current_load += delta;
        if current_load > capacity {
            return false;
        }
    }
    true
}
```

This is O(n log n) for sorting + O(n) sweep = O(n log n) total. No need for edge-finding or theta-trees if you just want feasibility checking.

**For our greedy scheduler:** This sweep-line check is sufficient to validate schedules post-hoc. Edge-finding is only needed if we want the solver to *prune* the search space during optimization.

---

## 7. CP-SAT Integration Architecture

We use CP-SAT (via `cp_sat` Rust crate) for optimal scheduling with allocation feasibility constraints.

### 7.0 Note: cp_sat Crate Limitations

**Important:** The [`cp_sat`](https://crates.io/crates/cp_sat) Rust crate's builder API is **incomplete** for scheduling problems. As of v0.3, it lacks:

- `new_interval_var()` / `new_fixed_size_interval_var()`
- `add_cumulative()`
- `add_no_overlap()`

GitHub issue [#21](https://github.com/KardinalAI/cp_sat/issues/21) has been open since September 2021.

**Our solution:** Fork the crate and add the missing functionality. The proto types exist (`IntervalConstraintProto`, `CumulativeConstraintProto`), so we add builder methods that construct these protos:

```rust
// IntervalConstraintProto fields:
message IntervalConstraintProto {
  LinearExpressionProto start = 4;
  LinearExpressionProto end = 5;
  LinearExpressionProto size = 6;
}

// CumulativeConstraintProto fields:
message CumulativeConstraintProto {
  LinearExpressionProto capacity = 1;
  repeated int32 intervals = 2;
  repeated LinearExpressionProto demands = 3;
}
```

The forked crate will be used via git dependency in `Cargo.toml`.

### 7.1 Core Model

```rust
use cp_sat::builder::CpModelBuilder;
use cp_sat::solver::{CpSolver, CpSolverStatus};

fn solve_schedule(
    ops: &[Op],
    deps: &[Vec<usize>],
    temporaries: &[Temporary],
    scratch_budget: usize,
) -> Option<Schedule> {
    let mut model = CpModelBuilder::default();

    // === Scheduling Variables ===
    // cycle[o] ∈ [asap[o], alap[o]] - when each op executes
    let cycle = create_cycle_vars(&mut model, ops, &asap, &alap);

    // === Slot Intervals (for cumulative) ===
    let slot_intervals = create_slot_intervals(&mut model, ops, &cycle);

    // === Live Range Intervals (for capacity checking) ===
    let live_intervals = create_live_intervals(&mut model, temporaries, &cycle);

    // === Constraints ===

    // 1. Precedence: respect dependencies
    for (o, op) in ops.iter().enumerate() {
        for &dep in &deps[o] {
            model.add_ge(cycle[o], cycle[dep] + 1);
        }
    }

    // 2. Slot limits: cumulative per engine
    for (engine, capacity) in SLOT_LIMITS {
        let engine_intervals: Vec<_> = ops.iter().enumerate()
            .filter(|(_, op)| op.engine == engine)
            .map(|(i, _)| slot_intervals[i].clone())
            .collect();
        model.add_cumulative(engine_intervals, vec![1; n], capacity);
    }

    // 3. Allocation feasibility: peak liveness ≤ budget
    let sizes: Vec<i64> = temporaries.iter().map(|t| t.size as i64).collect();
    model.add_cumulative(live_intervals, sizes, scratch_budget as i64);

    // === Objective: minimize makespan ===
    let makespan = model.new_int_var([(0, max_cycles)], "makespan");
    for (i, _) in ops.iter().enumerate() {
        model.add_le(cycle[i] + 1, makespan);
    }
    model.minimize(makespan);

    // === Solve ===
    let mut solver = CpSolver::new();
    solver.set_time_limit(Duration::from_secs(300));
    solver.set_num_workers(8);

    match solver.solve(&model) {
        CpSolverStatus::Optimal | CpSolverStatus::Feasible => {
            Some(extract_schedule(&solver, &cycle, ops))
        }
        _ => None
    }
}
```

### 7.2 Post-Scheduling Allocation

Once CP-SAT finds an optimal schedule with feasible liveness, address assignment is trivial:

```rust
// CP-SAT guarantees peak liveness ≤ budget, so this always succeeds
let schedule = solve_schedule(&ops, &deps, &temps, SCRATCH_BUDGET)?;
let allocation = linear_scan_allocate(&schedule, SCRATCH_BUDGET)
    .expect("CP-SAT ensured feasibility");
```

### 7.3 Decomposition for Scale (13,000+ ops)

For large models, solve in windows:

```rust
fn solve_with_decomposition(ops: &[Op], window_size: usize) -> Option<Schedule> {
    // Partition ops by ASAP bounds
    let windows = partition_by_asap(ops, window_size);
    let mut fixed_cycles: HashMap<usize, i64> = HashMap::new();

    for window in windows {
        // Solve window with earlier results fixed
        let partial = solve_window(&window, &fixed_cycles, SCRATCH_BUDGET)?;

        // Fix these cycles for subsequent windows
        for (op_id, cycle) in partial {
            fixed_cycles.insert(op_id, cycle);
        }
    }

    Some(Schedule::from(fixed_cycles))
}
```

### 7.4 Architecture Diagram

<svg viewBox="0 0 600 480" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#333"/>
    </marker>
  </defs>

  <!-- Input box -->
  <rect x="100" y="20" width="400" height="40" rx="5" fill="#e3f2fd" stroke="#1976d2" stroke-width="2"/>
  <text x="300" y="45" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#333">Input: Ops + Dependencies</text>

  <!-- Arrow -->
  <line x1="300" y1="60" x2="300" y2="90" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>

  <!-- CP-SAT Model box -->
  <rect x="50" y="100" width="500" height="120" rx="5" fill="#fff3e0" stroke="#f57c00" stroke-width="2"/>
  <text x="300" y="125" text-anchor="middle" font-family="sans-serif" font-size="14" font-weight="bold" fill="#333">CP-SAT Model (cp_sat crate)</text>

  <!-- Inner boxes -->
  <rect x="70" y="140" width="130" height="50" rx="3" fill="#fff" stroke="#666" stroke-width="1"/>
  <text x="135" y="162" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#333">cycle[o] vars</text>
  <text x="135" y="178" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#666">(scheduling)</text>

  <rect x="220" y="140" width="150" height="50" rx="3" fill="#fff" stroke="#666" stroke-width="1"/>
  <text x="295" y="162" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#333">Slot cumulative</text>
  <text x="295" y="178" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#666">(ALU≤12, VALU≤6, ...)</text>

  <rect x="390" y="140" width="140" height="50" rx="3" fill="#fff" stroke="#666" stroke-width="1"/>
  <text x="460" y="162" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#333">Live cumulative</text>
  <text x="460" y="178" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#666">(≤1536 words)</text>

  <text x="300" y="210" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#666">Objective: minimize makespan</text>

  <!-- Arrow -->
  <line x1="300" y1="220" x2="300" y2="250" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>

  <!-- Schedule box -->
  <rect x="100" y="260" width="400" height="40" rx="5" fill="#e8f5e9" stroke="#388e3c" stroke-width="2"/>
  <text x="300" y="285" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#333">Optimal Schedule (cycles assigned)</text>

  <!-- Arrow -->
  <line x1="300" y1="300" x2="300" y2="330" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>

  <!-- Linear scan box -->
  <rect x="100" y="340" width="400" height="50" rx="5" fill="#f3e5f5" stroke="#7b1fa2" stroke-width="2"/>
  <text x="300" y="362" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#333">Linear Scan Allocation (addresses)</text>
  <text x="300" y="380" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#666">Guaranteed to succeed - CP-SAT ensured feasibility</text>

  <!-- Arrow -->
  <line x1="300" y1="390" x2="300" y2="420" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>

  <!-- Output box -->
  <rect x="100" y="430" width="400" height="40" rx="5" fill="#e0f7fa" stroke="#0097a7" stroke-width="2"/>
  <text x="300" y="455" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#333">Output: Scheduled + Allocated Kernel</text>
</svg>

---

## 8. Using cp_sat Rust Crate

### 8.1 Installation

OR-Tools is installed via Homebrew:

```bash
brew install or-tools
```

Add the `cp_sat` crate to `scheduler_rs/Cargo.toml`:

```toml
[dependencies]
cp_sat = "0.1"  # Check crates.io for latest version
```

The crate links against the Homebrew-installed OR-Tools library.

### 8.2 Basic Usage

```rust
use cp_sat::builder::CpModelBuilder;
use cp_sat::solver::{CpSolver, CpSolverStatus};

fn example() {
    let mut model = CpModelBuilder::default();

    // Variables
    let x = model.new_int_var([(0, 10)], "x");
    let y = model.new_int_var([(0, 10)], "y");

    // Constraints
    model.add_le(x + y, 15);
    model.add_ne(x, y);

    // Objective
    model.maximize(x + 2 * y);

    // Solve
    let mut solver = CpSolver::new();
    let status = solver.solve(&model);

    if status == CpSolverStatus::Optimal {
        println!("x = {}, y = {}", solver.value(x), solver.value(y));
    }
}
```

### 8.3 Interval Variables and Cumulative

```rust
// Fixed-size intervals for slot consumption
let intervals: Vec<IntervalVar> = ops.iter().enumerate().map(|(i, _)| {
    model.new_fixed_size_interval_var(
        start: cycle[i],
        size: 1,
        name: format!("int_{}", i)
    )
}).collect();

// Cumulative constraint
model.add_cumulative(
    intervals: intervals.clone(),
    demands: vec![1; ops.len()],
    capacity: 12,
);
```

### 8.4 Optional Intervals (for VALU shifting)

```rust
let shift = model.new_bool_var("shift");

// Interval present only if shift is false
let valu_int = model.new_optional_fixed_size_interval_var(
    start: cycle[op],
    size: 1,
    is_present: shift.not(),
    name: "valu_int"
);

// 8 ALU intervals present only if shift is true
let alu_ints: Vec<_> = (0..8).map(|j| {
    model.new_optional_fixed_size_interval_var(
        start: cycle[op],
        size: 1,
        is_present: shift,
        name: format!("alu_{}", j)
    )
}).collect();
```

### 8.5 No-Overlap-2D (diffn)

```rust
// For register allocation as rectangle packing
let x_intervals: Vec<IntervalVar> = temps.iter().enumerate().map(|(i, t)| {
    model.new_fixed_size_interval_var(addr[i], t.size, format!("x_{}", i))
}).collect();

let y_intervals: Vec<IntervalVar> = temps.iter().enumerate().map(|(i, _)| {
    live_interval[i].clone()
}).collect();

model.add_no_overlap_2d(x_intervals, y_intervals);
```

---

## 9. Implementation Plan

### 9.1 Phase 1: CP-SAT Integration

1. **Add `cp_sat` crate** to `scheduler_rs/Cargo.toml`
   - Verify linking against Homebrew OR-Tools
   - Create basic test to confirm solver works

2. **Implement CP model** (Section 7.1)
   - Cycle variables with ASAP/ALAP bounds
   - Slot intervals + cumulative per engine
   - Live range intervals + cumulative for scratch budget
   - Makespan objective

3. **Integrate with existing pipeline**
   - Replace greedy scheduler call with CP-SAT solve
   - Keep existing linear scan allocator (post-scheduling)
   - Add fallback to greedy if CP-SAT times out

### 9.2 Phase 2: Scalability

1. **Window-based decomposition** for 13,000+ ops
   - Partition by ASAP bounds
   - Solve windows sequentially, fixing earlier results

2. **Tune solver parameters**
   - Time limits per window
   - Enable/disable timetable edge-finding based on problem size
   - Parallel workers

### 9.3 Phase 3: Refinement

1. **Warm-starting** from greedy solution
   - Use greedy schedule as hint to CP-SAT
   - Faster convergence to good solutions

2. **Incremental solving** for optimization exploration
   - Cache model, re-solve with modified constraints
   - E.g., "what if we allow 5% more cycles but reduce pressure?"

---

## 9. References

### Foundational Work

1. **Chaitin, G.J., Auslander, M.A., Chandra, A.K., Cocke, J., Hopkins, M.E., & Markstein, P.W.** (1981). "Register Allocation via Coloring." *Computer Languages*, 6(1):47-57.

2. **Goodwin, D.W. & Wilken, K.D.** (1996). "Optimal and Near-Optimal Global Register Allocation Using 0-1 Integer Programming." *Software: Practice and Experience*, 26(8):929-965.

3. **Wilken, K., Liu, J., & Heffernan, M.** (2000). "Optimal Instruction Scheduling Using Integer Programming." *Proceedings of the ACM SIGPLAN Conference on Programming Language Design and Implementation (PLDI)*, pp. 121-133.

### Integrated Approaches

4. **Castañeda Lozano, R., Carlsson, M., Drejhammar, F., & Schulte, C.** (2012). "Constraint-Based Register Allocation and Instruction Scheduling." *Proceedings of the 18th International Conference on Principles and Practice of Constraint Programming (CP)*, Springer LNCS 7514.

5. **Castañeda Lozano, R., Carlsson, M., Blindell, G.H., & Schulte, C.** (2019). "Combinatorial Register Allocation and Instruction Scheduling." *ACM Transactions on Programming Languages and Systems (TOPLAS)*, 41(3):17:1-17:53. [arXiv:1804.02452]

6. **Castañeda Lozano, R. & Schulte, C.** (2019). "Survey on Combinatorial Register Allocation and Instruction Scheduling." *arXiv:1409.7628*.

### Geometric Approaches

7. **Pereira, F.M.Q. & Palsberg, J.** (2008). "Register Allocation by Puzzle Solving." *Proceedings of the ACM SIGPLAN Conference on Programming Language Design and Implementation (PLDI)*, pp. 216-226.

### Constraint Programming

8. **Baptiste, P., Le Pape, C., & Nuijten, W.** (2001). *Constraint-Based Scheduling: Applying Constraint Programming to Scheduling Problems*. Springer.

9. **Van Hoeve, W.J. & Katriel, I.** (2006). "Global Constraints." In *Handbook of Constraint Programming*, Chapter 6. Elsevier.

### OR-Tools / CP-SAT

10. **Perron, L. & Furnon, V.** (2019). "OR-Tools." Google. https://developers.google.com/optimization

11. **Krupke, D.** (2024). "CP-SAT Primer." https://d-krupke.github.io/cpsat-primer/

12. **cp_sat crate** by KardinalAI. https://crates.io/crates/cp_sat

### VLIW-Specific

13. **Eichenberger, A.E., Davidson, E.S., & Abraham, S.G.** (1996). "Minimum Register Requirements for a Modulo Schedule." *Proceedings of the International Symposium on Computer Architecture (ISCA)*.

14. **Leupers, R. & Marwedel, P.** (1997). "Algorithms for Address Assignment in DSP Code Generation." *Proceedings of the International Conference on Computer-Aided Design (ICCAD)*.

### Edge-Finding and Propagation Algorithms

15. **Carlier, J. & Pinson, E.** (1989). "An Algorithm for Solving the Job-Shop Problem." *Management Science*, 35(2):164-176.

16. **Nuijten, W.P.M.** (1994). *Time and Resource Constrained Scheduling*. PhD thesis, Eindhoven University of Technology.

17. **Vilím, P.** (2007). [*Global Constraints in Scheduling*](http://vilim.eu/petr/disertace.pdf). PhD thesis, Charles University in Prague.

18. **Vilím, P.** (2004). "O(n log n) Filtering Algorithms for Unary Resource Constraint." *CPAIOR 2004*, LNCS 3011, pp. 335-347.

19. **Vilím, P.** (2009). ["Edge Finding Filtering Algorithm for Discrete Cumulative Resources in O(kn log n)."](https://link.springer.com/chapter/10.1007/978-3-642-04244-7_62) *CP 2009*, LNCS 5732, pp. 802-816.

20. **Vilím, P.** (2011). ["Timetable Edge Finding Filtering Algorithm for Discrete Cumulative Resources."](https://link.springer.com/chapter/10.1007/978-3-642-21311-3_22) *CPAIOR 2011*, LNCS 6697, pp. 230-245.

21. **Mercier, L. & Van Hentenryck, P.** (2008). ["Edge Finding for Cumulative Scheduling."](https://dl.acm.org/doi/10.1287/ijoc.1070.0226) *INFORMS Journal on Computing*, 20(1):143-153.

---

## Appendix A: Glossary

**ASAP (As Soon As Possible):** Earliest cycle an operation can be scheduled, considering dependencies.

**ALAP (As Late As Possible):** Latest cycle an operation can be scheduled without increasing makespan.

**CP-SAT:** Google OR-Tools' constraint programming solver that combines CP propagation with SAT-based search. Uses lazy clause generation for efficient solving.

**Cumulative Constraint:** Global constraint ensuring resource capacity is never exceeded. Uses interval variables with demands.

**Diffn (Non-Overlapping Rectangles):** Global constraint ensuring n-dimensional boxes don't overlap. Used for register allocation as 2D packing. *Not needed for our use case* (feasibility-only allocation).

**Edge-Finding:** Propagation algorithm for cumulative constraints. Reasons about sets of tasks to tighten bounds. O(kn log n) with theta-trees.

**Interference Graph:** Graph where nodes are temporaries and edges connect temporaries with overlapping live ranges.

**Live Range:** The span from when a value is defined to when it is last used.

**Makespan:** Total schedule length (number of cycles).

**Register Pressure:** Number of values live simultaneously at a given point.

**Temporary:** A named value that needs storage (register/scratch address).

**Theta-Tree (Θ-Tree):** Balanced binary tree data structure for efficient aggregate queries (total energy, earliest completion time) used in edge-finding algorithms.

---

## Appendix B: cp_sat Crate API Quick Reference

```rust
// Model building
let mut model = CpModelBuilder::default();

// Integer variables
let x = model.new_int_var([(lo, hi)], "name");
let b = model.new_bool_var("bool_name");

// Interval variables
let fixed = model.new_fixed_size_interval_var(start, size, "name");
let var = model.new_interval_var(start, size, end, "name");
let opt = model.new_optional_fixed_size_interval_var(start, size, is_present, "name");

// Constraints
model.add_eq(x, y);           // x == y
model.add_ne(x, y);           // x != y
model.add_lt(x, y);           // x < y
model.add_le(x, y);           // x <= y
model.add_gt(x, y);           // x > y
model.add_ge(x, y);           // x >= y

// Global constraints
model.add_all_different(vars);
model.add_cumulative(intervals, demands, capacity);
model.add_no_overlap(intervals);
model.add_no_overlap_2d(x_intervals, y_intervals);

// Conditional constraints
model.add_le(x, 10).only_enforce_if(b);
model.add_ge(x, 5).only_enforce_if(b.not());

// Objective
model.minimize(expr);
model.maximize(expr);

// Solving
let mut solver = CpSolver::new();
solver.set_time_limit(Duration::from_secs(60));
solver.set_num_workers(8);
let status = solver.solve(&model);

// Accessing solution
let val = solver.value(x);
let obj = solver.objective_value();
```
