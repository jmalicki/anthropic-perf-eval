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

## References

- LLVM's `RegPressureTracker` class
- "Combinatorial Register Allocation and Instruction Scheduling" (arXiv:1804.02452)
- "Register-Pressure-Aware Instruction Scheduling Using ACO" (ACM 2022)
- Berson et al. "Integrated Instruction Scheduling and Register Allocation"
