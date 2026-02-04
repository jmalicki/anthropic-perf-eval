# completion_cycle vs ALAP: Detailed Explanation

## Key Concepts

### 1. ALAP (As Late As Possible)
- **Definition**: `ALAP = min(consumer cycles) - 1`
- **Meaning**: The latest cycle an operation can complete without delaying its consumers
- **Example**: If consumers are at cycles 29, 30, 31 → `ALAP = 28`
- **Purpose**: Defines the upper bound of when an op can finish

### 2. completion_cycle (for shifted VALU ops)
- **Definition**: `completion_cycle = max(alu_cycles[0..8])`
- **Meaning**: When ALL 8 ALU operations (from a shifted VALU op) have completed
- **Purpose**: Consumers wait for `completion_cycle + 1` before they can start

## Current CP-SAT Model

### Domain Constraint
```rust
completion_cycle domain: [ASAP, ALAP]
```

**Example for Op 244**:
- ASAP = 6 (from dependencies)
- ALAP = 28 (from consumers at cycle 29)
- `completion_cycle ∈ [6, 28]`

### Max Constraint
```rust
completion_cycle >= alu_cycles[i] for all i in [0..8]
```

This ensures `completion_cycle` is the maximum of all 8 ALU cycles.

### Consumer Constraint
```rust
consumer_cycle >= completion_cycle + 1
```

Consumers must wait for all 8 ALU ops to complete.

## How completion_cycle Relates to ALAP

### Scenario 1: Clustered ALU Ops
- All 8 ALU ops scheduled at cycle 4
- `completion_cycle = max(4, 4, 4, 4, 4, 4, 4, 4) = 4`
- Consumer constraint: `consumer_cycle >= 4 + 1 = 5`

### Scenario 2: Spread ALU Ops
- ALU ops spread across cycles 4-10
- `completion_cycle = max(4, 5, 6, 7, 8, 9, 10, 10) = 10`
- Consumer constraint: `consumer_cycle >= 10 + 1 = 11`

### Scenario 3: Spread to ALAP
- ALU ops spread across cycles 4-28
- `completion_cycle = max(4, 8, 12, ..., 28) = 28` (ALAP)
- Consumer constraint: `consumer_cycle >= 28 + 1 = 29`

## The Critical Issue

### CP-SAT's View
CP-SAT sees:
1. **Objective**: Minimize makespan
2. **Makespan** depends on consumer cycles
3. **Consumer cycles** >= `completion_cycle + 1`
4. Therefore: **Minimize `completion_cycle`**
5. **Result**: Clustering is preferred (all 8 ALU ops at ASAP)

### The Reality
Consumers have **MULTIPLE dependencies**:
- Consumer depends on Op 244 (completion_cycle)
- Consumer also depends on Op 287 (cycle 8)
- Consumer constraint: `consumer_cycle >= max(completion_cycle, cycle_287) + 1`

**Example**:
- If `completion_cycle = 4` and `cycle_287 = 8`:
  - Consumer waits for: `max(4, 8) + 1 = 9`
- If `completion_cycle = 28` and `cycle_287 = 8`:
  - Consumer waits for: `max(28, 8) + 1 = 29`

### The Problem
**CP-SAT doesn't see that spreading is neutral!**

- CP-SAT thinks: "If I spread, `completion_cycle = 28`, consumer >= 29 (bad!)"
- Reality: "Consumer is already at 29 due to other dependencies!"
- CP-SAT should think: "Spreading is neutral (consumer already at 29), clustering helps (consumer could be at 9)!"

## Why ALAP Matters

**ALAP represents the theoretical limit**:
- If `completion_cycle <= ALAP`, consumers aren't delayed (by this dependency)
- But CP-SAT minimizes `completion_cycle`, so it avoids spreading
- Even though spreading up to ALAP doesn't hurt consumers!

## The Fix Needed

CP-SAT should recognize that:
1. **Spreading up to ALAP is neutral** (doesn't delay consumers if they're already waiting)
2. **Clustering is beneficial** (allows consumers to start earlier)
3. **Therefore**: Shift and try to cluster, but spreading is acceptable if capacity requires it

This requires CP-SAT to understand that `completion_cycle` up to ALAP doesn't delay consumers when they have other dependencies that are already later.
