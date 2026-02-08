# Greedy Scheduler Design & Research Findings

This document captures the findings from a 6-track parallel investigation into
improving the greedy scheduler's hint quality for CP-SAT (Feb 2026).

## Table of Contents

1. [Overview & Motivation](#overview--motivation)
2. [Empirical Findings (Track 0)](#empirical-findings-track-0)
3. [Current Limitations](#current-limitations-of-all-pressure-policies)
4. [LLVM GenericScheduler Lessons (Track 1)](#llvm-genericscheduler-lessons-track-1)
5. [Problem Structure (Track 2)](#problem-structure-track-2)
6. [Slack & Deferral Analysis (Track 3)](#slack--deferral-analysis-track-3)
7. [Hint Pipeline Inconsistency (Track 4)](#hint-pipeline-inconsistency-track-4)
8. [Proposed Architecture (Track 5)](#proposed-architecture-track-5)
9. [Implementation Roadmap](#implementation-roadmap)

---

## Overview & Motivation

The greedy scheduler produces warm-start hints (~100ms) for the CP-SAT solver.
Better hints reduce the time CP-SAT needs to find optimal (or near-optimal)
schedules, directly accelerating the overall pipeline.

### Three-Phase Pipeline

```
Phase 1: Greedy Scheduler (~100ms)
  └── Produces per-op cycle hints, VALU shift decisions, SelectChoice/ConstChoice
  └── May exceed scratch budget in some sections

Phase 2: Per-Section Feasibility CP-SAT (seconds–minutes)
  └── Re-schedules only sections that exceed scratch budget
  └── Merges results back into global schedule
  └── Changes SelectChoice/ConstChoice for repaired sections

Phase 3: Full-Kernel CP-SAT (minutes–hours)
  └── Uses greedy+repair hints as warm start
  └── Optimizes across all sections simultaneously
  └── Finds minimum-cycle feasible schedule
```

### Why Hint Quality Matters

CP-SAT uses hints for:
- **Initial feasible solution**: if hints are feasible, solver has a valid
  starting point immediately
- **Search guidance**: hint values bias the branching heuristic toward the
  hinted region of the search space
- **Objective bound**: hint makespan provides an initial upper bound

Poor hints (more cycles, higher peak liveness) → solver takes longer to find
and prove optimal solutions.

---

## Empirical Findings (Track 0)

We compared all 4 pressure policies on the standard 4-round kernel
(`forest_height=10, batch_size=256, rounds=4, tree_reduce_max_depth=3,
enable_valu_shift=true`).

### Policy Comparison (4-round, depth-3)

| Metric | Greedy | Throttled | Gated | Adaptive |
|--------|--------|-----------|-------|----------|
| Cycles | 502 | 502 | 503 | 503 |
| Total ops | 6635 | 6635 | 6551 | 6551 |
| Peak live words | 1523 | 1523 | 1515 | 1515 |
| Max addr | 1536 | 1536 | 1520 | 1520 |
| Budget headroom | 13 | 13 | 21 | 21 |
| VALU shifted | 408 | 408 | 396 | 396 |
| Budget exceeded | No | No | No | No |

### Key Finding: Only 2 Distinct Behaviors

Despite 4 policy implementations, there are only **2 distinct clusters**:

- **Cluster A** (Greedy, Throttled): 502 cycles, 1523 peak, 408 shifts
- **Cluster B** (Gated, Adaptive): 503 cycles, 1515 peak, 396 shifts

Greedy and Throttled produce identical schedules because the Throttled policy's
sort re-ordering threshold (85%) is rarely crossed during the 4-round kernel —
or when it is crossed, the re-ordering doesn't change which ops fit in slots.

Gated and Adaptive produce identical schedules because Adaptive's additional
shift suppression zones don't activate on this kernel configuration (shift
suppression is already handled by the graduated composite sort).

### Cluster B Is Better for Hint Quality

Although Cluster B uses 1 more cycle, it has:
- **8 fewer peak live words** (1515 vs 1523) → more scratch headroom
- **21 words of budget headroom** vs 13 → safer margin for CP-SAT
- **12 fewer VALU shifts** → less ALU pressure from shifted 8-wide ops
- **84 fewer total ops** → smaller model for CP-SAT

The extra cycle is a trivial cost compared to the improved headroom, which gives
CP-SAT more room to rearrange ops without hitting scratch limits.

---

## Current Limitations of All Pressure Policies

All 4 policies share fundamental limitations:

### 1. No Cross-Cycle Deferral

Every policy returns `false` from `should_defer_for_pressure()`:

```rust
fn should_defer_for_pressure(...) -> bool {
    // Never defer — pressure awareness is handled entirely by sort_ready().
    false
}
```

This was a deliberate design choice to avoid **cascading deferral** — a deadlock
where all ready ops are deferred because they all have positive net pressure
(common on unrolled VLIW kernels). See
`docs/SCHEDULING_PRESSURE_CASCADING_DEFERRAL.md`.

However, it means the scheduler cannot strategically delay a load to a later
cycle to reduce peak liveness, even when slack allows it.

### 2. Limited Information in sort_ready

The `sort_ready` signature only provides:
```rust
fn sort_ready(
    ready_ops: &mut [(usize, usize, usize, usize)],  // (op_index, luc, words_added, words_freed)
    live_words: usize,
    scratch_budget: usize,
);
```

Missing information:
- **ASAP/ALAP bounds** — no slack awareness
- **Engine type** — can't prioritize by bottleneck engine (VALU vs ALU)
- **Successor fan-out** — can't estimate downstream impact
- **Critical path position** — can't distinguish critical from non-critical ops

### 3. Hardcoded Thresholds

All policies use hardcoded pressure thresholds (e.g. 85% for soft, 95% for
critical) that aren't tuned per kernel or per section. A 4-round kernel with
6551 ops has very different pressure dynamics than a 2-round kernel with ~1600
ops.

---

## LLVM GenericScheduler Lessons (Track 1)

LLVM's `GenericScheduler` (in `llvm/lib/CodeGen/MachineScheduler.cpp`) provides
a well-tested reference for register-pressure–aware instruction scheduling.

### Key Design Principles

1. **Strict priority cascade** — not a blended score. Each level is a hard
   filter; lower levels only break ties.

2. **Bidirectional scheduling** — LLVM schedules in both directions:
   - **Top-down** (forward): maximizes ILP
   - **Bottom-up** (backward): minimizes register pressure
   - The scheduler picks the direction that produces better pressure.

3. **Per-candidate pressure delta** — for each candidate op, LLVM computes the
   precise register pressure change if that op were scheduled, using
   `RegExcess` (over limit) and `CriticalMax` (approaching critical set).

### 8-Level Priority Hierarchy

LLVM's `tryCandidate()` uses this strict cascade:

| Level | Criterion | Purpose |
|-------|-----------|---------|
| 1 | Avoid unschedulable ops | Don't create register spills |
| 2 | Physical register copies | Free fixed registers early |
| 3 | RegExcess (excess pressure) | Reduce pressure when over limit |
| 4 | RegCritical (critical pressure) | Reduce pressure approaching limit |
| 5 | Cluster heuristic | Group related ops for locality |
| 6 | Weak RegExcess/RegCritical | Soft pressure reduction |
| 7 | Resource contention | Balance functional unit usage |
| 8 | Latency/height | Maximize ILP (top-down) or reduce height (bottom-up) |

### Applicable Lessons

For our greedy scheduler:
- **Cascade over blend**: a strict cascade avoids the instability of weighted
  scoring where small threshold changes flip priority order
- **Engine saturation as level 1**: VALU is our critical bottleneck (98%+
  utilization); prioritizing ops for the saturated engine is analogous to
  LLVM's "avoid unschedulable" level
- **Slack as tiebreaker**: ops with zero slack (on critical path) should be
  scheduled before ops with slack, analogous to LLVM's latency/height level

---

## Problem Structure (Track 2)

### Op Mix Analysis

For the standard 4-round kernel:
- **Total ops**: ~6500–6600 (varies by policy)
- **VALU ops**: ~55-60% of total
- **VALU bottleneck**: `ceil(VALU_ops / 6)` = theoretical minimum cycles
  (VALU has 6 slots per cycle)

### Structure by Phase

Each round has a repeating pattern:
1. **Load phase**: vload + const + load_offset (Load slots, 2/cycle)
2. **Compute phase**: VALU ops (xor, add, multiply_add, vbroadcast)
3. **Store phase**: vstore (Store slots, 2/cycle)
4. **Flow phase**: vselect/select (Flow slots, 1/cycle — the scarcest resource)

### Section Independence

Sections (separated by barriers) are independent:
- No data dependencies cross barriers
- Each section can be scheduled independently for feasibility
- Full-kernel CP-SAT optimizes globally after per-section repair

---

## Slack & Deferral Analysis (Track 3)

### Slack Is Never Used

The greedy scheduler computes ASAP/ALAP bounds but **never uses slack**
(ALAP - ASAP) in scheduling decisions. All ops are scheduled as early as
possible (ASAP-ordered), regardless of how much slack they have.

This is a significant missed opportunity: a load with slack=20 could be
deferred to reduce peak liveness without increasing makespan.

### Deferral Direction Matters

- **Deferring producers** (loads, vbroadcasts) — **reduces** liveness pressure
  because consumers haven't been scheduled yet; the loaded values don't need to
  be live as long
- **Deferring consumers** (stores, xor, etc.) — **increases** liveness pressure
  because their operands must remain live longer

This is why a simple "defer when pressure is high" heuristic fails: it defers
*all* ready ops equally, when only deferring *producers* would help.

### Touati's Register Saturation

The theoretical lower bound on peak register pressure for a DAG is computable
(Touati & Eisenbeis, "A Polynomial Algorithm for Register Saturation"). This
could serve as:
- A validation metric: how far is the greedy schedule from optimal pressure?
- A stopping criterion: if pressure equals the lower bound, no improvement is
  possible through reordering alone

---

## Hint Pipeline Inconsistency (Track 4)

### The Problem

When per-section CP-SAT repairs a section, it may change SelectChoice decisions
(FMA vs vselect). This creates an inconsistency:

1. **Greedy** schedules section K with SelectChoice=vselect
2. **Per-section CP-SAT** re-schedules section K, choosing SelectChoice=FMA
3. The FMA encoding has **different latency** (2 cycles vs 1 cycle)
4. **Downstream ops in later sections** still have cycle hints based on the
   old vselect timing
5. These hints are now **stale** — they may violate precedence constraints
   with the new FMA latency

### Code References

The merge happens in `tick_tock_ssa/mod.rs`:
- **Lines 702-708**: SelectChoice overrides are merged into `current_select_fma`
  ```rust
  section_select_fma_overrides[k] = Some(seg_result.select_fma_ops.clone());
  for &i in &global_indices {
      current_select_fma.remove(&ctx.ops[i].id);
  }
  for &id in &seg_result.select_fma_ops {
      current_select_fma.insert(id);
  }
  ```
- **Line 795**: Hints collected from ops whose cycles may be inconsistent
  ```rust
  let greedy_hints: Vec<i32> = ctx.ops.iter().map(|op| op.cycle).collect();
  ```

Between lines 702-708 (merge) and line 795 (hint collection), ops in sections
downstream of a repaired section still carry their original greedy cycle values.
If the repaired section changed timing, those downstream hints are stale.

### Recommended Fix: Option C (BFS Strip)

Three options were considered:

| Option | Approach | Pros | Cons |
|--------|----------|------|------|
| A | Re-run greedy on downstream sections | Correct, thorough | Expensive, complex |
| B | Shift downstream hints by delta | Simple | Doesn't account for changed precedence |
| **C** | **BFS to find affected ops, set hints to -1** | **Targeted, safe** | CP-SAT must re-derive those hints |

**Option C** is recommended:
1. After per-section repair, BFS from the repaired section's outputs
2. Any downstream op whose hint depends on a changed cycle → set hint to -1
3. CP-SAT will freely schedule those ops (no stale hint bias)

This is safe because CP-SAT already handles ops with no hints; we're just
removing hints that are known to be unreliable.

---

## Proposed Architecture (Track 5)

### SchedulingCandidate Struct

Replace the raw tuple `(op_index, luc, words_added, words_freed)` with a
richer struct:

```rust
pub struct SchedulingCandidate {
    pub op_index: usize,
    pub luc: usize,           // live-use count (how many live values this op consumes)
    pub words_added: usize,   // scratch words this op would add
    pub words_freed: usize,   // scratch words freed by scheduling this op
    pub engine: Engine,       // ALU, VALU, Load, Store, Flow
    pub slack: i32,           // ALAP - ASAP (how much this op can move)
    pub asap: i32,            // earliest feasible cycle
    pub alap: i32,            // latest feasible cycle
    pub is_producer: bool,    // true if this op defines a value used later
    pub successor_count: usize, // fan-out
}
```

### SlotSnapshot + SchedulingContext

```rust
pub struct SlotSnapshot {
    pub alu_used: usize,
    pub valu_used: usize,
    pub load_used: usize,
    pub store_used: usize,
    pub flow_used: usize,
}

pub struct SchedulingContext {
    pub live_words: usize,
    pub scratch_budget: usize,
    pub current_cycle: usize,
    pub slots: SlotSnapshot,
}
```

### CascadePolicy: 5-Level Strict Priority

```
Level 1: Engine Saturation
  → If VALU is nearly full and candidate is VALU, schedule it NOW
  → If candidate's engine has 0 remaining slots, skip to next candidate
  → Ensures bottleneck engine ops aren't displaced by non-critical ops

Level 2: Slack-Zero (Critical Path)
  → Ops with slack=0 are on the critical path; delaying them increases makespan
  → Always prefer slack=0 ops over slack>0 ops at the same level

Level 3: Live-Use Count (LUC)
  → Higher LUC = more values freed → schedule these first to reduce pressure
  → This is the current greedy heuristic's primary sort key

Level 4: Net Pressure (words_added - words_freed)
  → When above soft pressure threshold, prefer ops with lower net pressure
  → Only activates when pressure exceeds configurable threshold

Level 5: Tiebreak (ASAP, then op_index)
  → Schedule earlier ASAP ops first (maintains topological order)
  → op_index breaks remaining ties deterministically
```

### ASAP Inflation: Strategic Deferral

Instead of the current "never defer" approach, inflate ASAP bounds for
high-slack producers when pressure exceeds a soft threshold:

```
if live_words > soft_threshold && op.is_producer && op.slack > min_slack:
    inflated_asap = current_cycle + 1  // push to next cycle
```

This is safe because:
- Only affects ops with slack > 0 (not on critical path)
- Only defers producers (which reduce pressure when deferred)
- The inflation is bounded: at most `slack` cycles of delay
- Falls back to normal scheduling when pressure drops

### Estimated Code Changes

- ~260 lines new code (SchedulingCandidate, CascadePolicy, ASAP inflation)
- ~40 lines migration (replace tuple sort_ready signature)
- No changes to CP-SAT model or hint pipeline

---

## Implementation Roadmap

| Priority | Change | Impact | Risk | Effort |
|----------|--------|--------|------|--------|
| 1 | LLVM-style priority cascade (CascadePolicy) | High — structured prioritization replaces ad-hoc blending | Low — sort_ready is already the control point | ~150 lines |
| 2 | ASAP inflation for deferral | High — enables pressure-aware deferral without cascading deadlock | Low — only affects ops with slack, reversible | ~80 lines |
| 3 | Multi-greedy via rayon | Medium — run multiple policies in parallel, pick best | Low — policies are pure functions, trivially parallelizable | ~50 lines |
| 4 | Hint pipeline fix (Option C: BFS strip) | Medium — removes stale hints after per-section repair | Low — only removes hints, doesn't add constraints | ~60 lines |
| 5 | Bidirectional scheduling | High — bottom-up pass for pressure, top-down for ILP | Medium — requires reverse-direction scheduling infrastructure | ~300 lines |

### Dependencies

- Priority 1 (CascadePolicy) is a prerequisite for Priority 2 (ASAP inflation)
  because the cascade provides the structured context for deferral decisions
- Priority 3 (multi-greedy) is independent and can be done at any time
- Priority 4 (hint fix) is independent and should be done soon to improve
  CP-SAT convergence
- Priority 5 (bidirectional) depends on Priority 1 for the scheduling
  infrastructure

### Validation Strategy

Each change should be validated by:
1. Running the 4-round benchmark with CP-SAT (1800s)
2. Comparing hint quality: cycles, peak liveness, budget headroom
3. Comparing CP-SAT convergence: time to first feasible, time to best solution
4. Regression tests for slot invariants and scratch budget

---

## References

- LLVM `GenericScheduler`: `llvm/lib/CodeGen/MachineScheduler.cpp`
- Cascading deferral analysis: `docs/SCHEDULING_PRESSURE_CASCADING_DEFERRAL.md`
- ASAP/ALAP precedence offsets: `docs/ASAP_ALAP_PRECEDENCE_OFFSETS.md`
- Touati & Eisenbeis, "A Polynomial Algorithm for Register Saturation"
- Policy implementations: `scheduler_rs/src/kernel_builder/ssa_greedy/pressure_policy/`
