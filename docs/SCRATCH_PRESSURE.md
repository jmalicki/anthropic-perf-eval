# Scratch Pressure Tracking

This document describes the scratch memory pressure tracking system in the greedy list scheduler.

## Concepts

### Scratch Memory as a Register File

Scratch memory (1536 words) in this VLIW architecture is functionally equivalent to a **register file**:
- Fixed size limit (1536 words)
- Fast access (1 cycle latency)
- Values must be "live" between definition (write) and last use (read)
- Exceeding capacity = program failure

### Live Ranges

A **live range** is the interval during which a value must be kept in scratch:

```
Cycle 0:  s5 = load(...)       ← s5 DEFINED (becomes live)
Cycle 1:  s6 = s5 + 1          ← s5 read (still live)
Cycle 2:  ...                  ← s5 not used (still live!)
Cycle 3:  s7 = s5 * 2          ← s5 LAST USE (dies)
```

`s5` is "live" during cycles 0-3 (occupies 1 scratch word).

## VLIW Timing Model and Liveness

### Cycle Execution Semantics

From the ISA (docs/ISA_REFERENCE.md):
- "Results written in cycle N are readable in cycle N+1"
- "Within the same cycle, all reads happen before all writes"

This means each cycle has two logical phases:
1. **Read phase**: All ops read their inputs (from values written in previous cycles)
2. **Write phase**: All ops write their outputs

### Implications for Liveness

A VReg's **live range** is the period during which it must occupy scratch space:

- **Definition**: VReg becomes live at the END of its definition cycle (after the write phase)
- **Last use**: VReg dies at the END of its last use cycle (after the read phase consumes it)

Key insight: A VReg is NOT live "through" its last use cycle in the scratch budget sense. At the end of that cycle, after being read, it's dead and its space is freed.

### SSA Guarantees

In SSA form:
- Each VReg is defined exactly once
- A VReg must be defined BEFORE any of its uses (data dependency)
- Therefore: `def_cycle < last_use_cycle` (strictly less, never equal)
- A same-cycle def+use is impossible because the use would read stale data

### Same-Cycle Reuse

"Same-cycle reuse" refers to **aggregate** scratch pressure, not individual VRegs:

At cycle C, if:
- VRegs A, B, C have their last use (being consumed)
- VRegs X, Y are being defined (newly created)

The scratch pressure at the END of cycle C = (previous live - A,B,C + X,Y)

The freed space from A, B, C becomes available for X, Y within the same cycle because:
- A, B, C are read at the START of cycle C
- X, Y are written at the END of cycle C
- By the end, A, B, C are dead and X, Y need storage

### CP-SAT Interval Model

For cumulative constraints, we model liveness as half-open intervals `[start, end)`:
- `start = def_cycle`
- `end = last_use_cycle` (NOT last_use_cycle + 1)

At time point `t`, cumulative counts intervals where `start <= t < end`.

With `[def, last_use)`:
- At `def_cycle`: VReg is live (just defined, needs storage)
- At `last_use_cycle - 1`: VReg is live (will be used next cycle)
- At `last_use_cycle`: VReg is NOT counted (being consumed, storage freed)

This matches greedy's same-cycle reuse semantics.

### Scratch Pressure

**Scratch pressure** at any point = count of currently live addresses.

```
Cycle:     0   1   2   3   4   5   6   7
─────────────────────────────────────────
s5:        [===========]                    live 0-3
s6:            [===============]            live 1-5
s7:                    [===========]        live 3-6
s8:                            [=======]    live 5-7
─────────────────────────────────────────
Pressure:  1   2   2   3   2   3   2   1
                       ↑
                  Peak = 3
```

If peak pressure exceeds the scratch budget (1536), the schedule is invalid.

### LUC (Last Use Count) Heuristic

From LLVM and compiler literature, **LUC** is a simple but effective heuristic for reducing pressure:

> **Prioritize scheduling ops that CLOSE more live ranges.**

When multiple ops are ready to schedule, prefer the one whose reads include more "last uses" - this frees scratch for subsequent ops.

```rust
// LUC = count of addresses where this op is their last reader
let luc = op.last_use_reads.len();

// Higher LUC = higher priority (frees more scratch)
```

## Implementation

### Data Structures

#### `Op.last_use_reads: HashSet<i64>`

Each op tracks which of its read addresses are "last uses" - i.e., no later op will read from them. This is computed by `compute_last_uses()` before scheduling.

#### `ScratchPressureTracker`

Tracks live addresses during scheduling:

```rust
struct ScratchPressureTracker {
    live: HashSet<i64>,      // Currently live addresses
    peak: usize,             // Maximum pressure seen
    limit: usize,            // Budget limit (1536)
    exceeded_count: usize,   // Times we exceeded budget
}
```

Methods:
- `pressure_delta(op)` - compute how scheduling this op would change pressure
- `would_exceed(op)` - check if scheduling would exceed budget
- `apply(op)` - update live set after scheduling (add writes, remove last uses)

### Algorithm

1. **Pre-scheduling**: `compute_last_uses()` analyzes all ops to determine which reads are last uses
2. **Priority**: LUC is used as a **tie-breaker** for all priority modes - when ops have equal priority, prefer those that free more scratch
3. **Tracking**: After each op is scheduled, `pressure_tracker.apply()` updates the live set
4. **Reporting**: Peak pressure and exceeded count are reported in diagnostics

### Priority Modes

LUC is integrated as a tie-breaker for **all** priority modes:

```rust
PriorityMode::CriticalPath => (-cp, is_valu, luc_tiebreaker, op_id)
PriorityMode::Slack => (slack, -cp, luc_tiebreaker, op_id)
// ... etc
```

There's also a dedicated `ScratchPressure` mode where LUC is the **primary** criterion:

```rust
PriorityMode::ScratchPressure => (-luc, -cp, 0, op_id)
```

## Diagnostics

When diagnostics are enabled, the scheduler reports:

```
--- Scratch Pressure ---
  Peak pressure: 1423 / 1536 (92.6%)
```

If pressure ever exceeded the budget:

```
  WARNING: Pressure exceeded budget 5 times!
```

## Duality: Constraint vs Heuristic

Scratch pressure tracking serves two purposes:

1. **Constraint (future)**: Detect when a schedule would exceed the scratch budget
2. **Heuristic (current)**: Use LUC to reduce pressure peaks by preferring ops that free scratch

Currently, we track pressure and report overflows but don't reject schedules that exceed. Future work could add hard constraints or spilling support.

## Scheduling Strategy: Inverted Priority Based on Headroom

### The Problem

Standard pressure-aware scheduling (LUC heuristic) always prioritizes ops that
reduce register pressure. However, this can lead to deadlock: if all pressure-reducing
ops are scheduled early, we may be left with only pressure-increasing ops (like
vbroadcast) when budget is tight.

### Our Solution: Proactive Resource Reservation

We invert the priority based on current budget headroom:

- **High headroom (>50%)**: Schedule pressure-INCREASING ops first
- **Low headroom (<50%)**: Schedule pressure-REDUCING ops first

This ensures pressure-reducing ops are preserved as "escape hatches" for tight situations.

### How It Works

The scheduler computes `net_change` for each ready op:
- `net_change = words_added - words_freed`
- Positive = increases pressure (e.g., vbroadcast adds 8 words)
- Negative = reduces pressure (e.g., op that consumes multiple VRegs)

When headroom is high, we sort ready ops by `net_change DESC` (highest first).
When headroom is low, we sort by `net_change ASC` (lowest/most negative first).

### ALAP Forecasting

We also use ALAP (As-Late-As-Possible) analysis to predict when VRegs will be freed:

1. Compute ALAP for each op (latest cycle it can be scheduled)
2. For each VReg, its death bound = ALAP of its last consumer
3. Build a forecast: guaranteed free words by cycle C

When an op would exceed budget, instead of force-scheduling (which violates budget),
we use the forecast to find the earliest cycle where it safely fits.

### Purpose

The greedy scheduler's primary role is to provide a **feasible hint** for the CP-SAT
optimizer. When greedy succeeds (stays within budget), we use it directly. When it
fails (would need to exceed budget), we pass its result as a hint to CP-SAT, which
can find a valid schedule using backtracking search.

### Limitations (Why Greedy Can Fail)

Due to PSPACE-completeness of the pebbling problem (see "Theoretical Background" below),
**no polynomial-time greedy algorithm can guarantee finding a valid schedule**.

The inverted priority and ALAP forecasting heuristics help in practice but cannot
overcome the fundamental computational complexity. When greedy gets stuck:

1. It force-schedules the best available op (using LUC heuristic)
2. This may violate the budget constraint
3. The scheduler tracks that budget was exceeded
4. The system falls back to CP-SAT with the greedy schedule as a hint

This is the correct behavior - we try our best with fast heuristics, track failures,
and use complete (but slower) solvers when needed.

### Novelty

This strategy appears to be novel. Literature search (January 2026) found related but
distinct approaches:

- **Belady's MIN** (1966): Evict value used furthest in future - future-aware but reactive
- **LUC Heuristic** (Shobaki et al. 2022): Always prefer ops closing live ranges - no headroom adaptation
- **SLIL Cost Function** (Shobaki et al. 2019): Optimize total overlap - different objective

Our approach combines future-awareness (via ALAP forecasting) with adaptive priority
inversion based on current resource state.

## Theoretical Background: Pebbling Games

### The Fundamental Problem

Our scheduling problem is equivalent to the **black pebbling game** on directed acyclic graphs (DAGs):

- **Nodes** = operations (SSA ops)
- **Pebbles** = live scratch words (memory currently in use)
- **Budget** = maximum pebbles allowed simultaneously (1536 words)
- **Goal** = Execute all ops (visit all nodes) without exceeding pebble limit

The pebbling game proceeds as follows:
1. To "execute" a node, all its predecessors must currently have pebbles
2. After execution, we place a pebble on the node (its result is live)
3. Pebbles can be removed when a value is no longer needed
4. At no point can total pebbles exceed the budget

### Complexity: PSPACE-Complete

**Theorem (Gilbert, Lengauer, Tarjan, 1980)**: Deciding whether a DAG can be pebbled with at most k pebbles is **PSPACE-complete**.

This has profound implications for our scheduler:

1. **No polynomial-time algorithm can guarantee finding a solution** (unless P = PSPACE, which is believed false)
2. Simple greedy heuristics like "lowest net_change first" **may fail** even when a valid schedule exists
3. Only **complete solvers** (like CP-SAT with backtracking) can guarantee finding a solution if one exists

### Why This Problem is Hard

Consider a simple example where greedy fails:

```
     A
    / \
   B   C
    \ /
     D
```

If we greedily schedule A, then B (because it has lower net_change), we might paint ourselves into a corner where scheduling C would exceed budget. A smarter order (A, C, B, D) might work, but greedy can't backtrack.

The difficulty arises because:
- Scheduling decisions have **non-local effects** (scheduling X now affects what's possible later)
- The search space is exponentially large
- Optimal solutions may require **counter-intuitive** orderings

### Special Cases with Polynomial Solutions

Some restricted DAG structures admit efficient algorithms:

| DAG Structure | Algorithm | Complexity |
|--------------|-----------|------------|
| Trees | Sethi-Ullman | O(n) optimal |
| Series-parallel | Dynamic programming | O(n) optimal |
| Fixed budget k | FPT algorithm | O(f(k) · n) |
| Bounded pathwidth | DP on tree decomposition | Polynomial |

Unfortunately, our kernel DAGs have **high pathwidth** and complex structure, so these special cases don't apply.

### Our Approach: Greedy Heuristic + CP-SAT Fallback

Given the PSPACE-completeness, we use a pragmatic two-tier approach:

```
1. Try greedy scheduler (fast, O(n²))
   - Sort ready ops by net_change ASC (pressure-reducing first)
   - Schedule first op that fits budget
   - If stuck: force-schedule best available (may exceed budget)
   - Track whether budget was exceeded

2. If greedy stayed within budget → use schedule directly

3. If greedy exceeded budget → fall back to CP-SAT
   - Pass greedy schedule as hint (ordering, structure)
   - CP-SAT finds valid schedule using hint as starting point
   - CP-SAT uses CDCL (conflict-driven clause learning) for efficiency
```

### Why CP-SAT Works Well Here

CP-SAT (Constraint Programming with Boolean Satisfiability) is essentially "backtracking done right":

- **Conflict-driven clause learning (CDCL)**: Learns from failures to prune search space
- **Constraint propagation**: Eagerly infers forced assignments
- **Highly optimized implementation**: Google OR-Tools is state-of-the-art

While CP-SAT has exponential worst-case complexity (unavoidable for PSPACE-complete problems), it typically solves practical instances efficiently by:
1. Using greedy hints to start near a good solution
2. Learning from conflicts to avoid repeating mistakes
3. Aggressive pruning via constraint propagation

### Why We Still Use Greedy

Despite its incompleteness, the greedy scheduler is valuable:

1. **Speed**: O(n²) vs exponential for CP-SAT
2. **Usually works**: For most kernel configurations, greedy finds valid schedules
3. **Provides hints**: Even when greedy fails, its partial solution helps CP-SAT
4. **Deterministic**: Same input always produces same output

The greedy scheduler is our "fast path" - we only invoke the expensive CP-SAT solver when necessary.

### Guarantees and Non-Guarantees

**What greedy DOES guarantee:**
- If it produces a schedule without exceeding budget, that schedule is valid
- Topological order is always respected (dependencies satisfied)
- Slot limits are always respected

**What greedy CANNOT guarantee:**
- Finding a schedule when one exists (PSPACE-complete)
- Optimal cycle count (even if valid)
- Minimum peak scratch usage

**What CP-SAT guarantees:**
- If a valid schedule exists, CP-SAT will find one (given enough time)
- Optimality within time limit (best solution found)

### Literature Search Notes (January 2026)

We conducted a literature search to understand the state of the art:

**Directly relevant:**
- Gilbert, Lengauer, Tarjan (1980) - PSPACE-completeness proof
- Nordström (2013) - Comprehensive pebbling survey: https://arxiv.org/pdf/1307.3913
- Inapproximability results: Even approximating minimum pebbles is hard

**Related but different:**
- Register allocation via graph coloring (NP-complete, not PSPACE)
- Sethi-Ullman for expression trees (polynomial, but only for trees)
- Integrated scheduling + allocation papers (heuristics, no completeness guarantees)

**Our contribution:**
Our "inverted priority based on headroom" heuristic appears novel - we couldn't find prior work that adaptively changes priority direction based on current resource utilization.

## References

### Pebbling Games and Complexity
- Gilbert, J., Lengauer, T., & Tarjan, R. (1980). "The Pebbling Problem is Complete in Polynomial Space" - SIAM Journal on Computing
- Nordström, J. (2013). "Pebble Games, Proof Complexity, and Time-Space Trade-offs" - https://arxiv.org/pdf/1307.3913
- Chan, S.O. (2013). "Approximation Resistance from Pebbling" - https://www.researchgate.net/publication/333867447

### Register Allocation and Scheduling
- LLVM's `RegPressureTracker` class
- "Combinatorial Register Allocation and Instruction Scheduling" (arXiv:1804.02452)
- Berson et al. "Integrated Instruction Scheduling and Register Allocation"
- Hack, S. & Goos, G. (2006). "Optimal Register Allocation for SSA-form Programs in Polynomial Time"

### Pressure-Aware Scheduling Heuristics
- Belady, L.A. (1966). "A Study of Replacement Algorithms for Virtual Storage Computers"
- Shobaki, G. et al. (2022). "Register-Pressure-Aware Instruction Scheduling Using ACO" [ACM TACO]
- Shobaki, G. et al. (2019). "Exploring Alternative Cost Functions for Register-Pressure-Aware Scheduling" [ACM TACO]

### Constraint Programming
- Google OR-Tools CP-SAT Solver: https://developers.google.com/optimization/cp/cp_solver
- Conflict-Driven Clause Learning: https://en.wikipedia.org/wiki/Conflict-driven_clause_learning
