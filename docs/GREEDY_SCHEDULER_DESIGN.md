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
7. [Hint Pipeline Inconsistency (Track 4) — FIXED](#hint-pipeline-inconsistency-track-4--fixed)
8. [Implemented Architecture (Track 5)](#implemented-architecture-track-5)
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

## Hint Pipeline Inconsistency (Track 4) — FIXED

> **Status**: Three hint pipeline inconsistencies have been fixed in
> `tick_tock_ssa/mod.rs`. The BFS strip (Option C below) for stale downstream
> hints remains a future improvement. See the fix details at the end of this section.

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

### Fixes Implemented

Three additional hint pipeline inconsistencies (beyond the stale downstream
timing issue above) were identified and fixed:

1. **Shift hint over-exclusion** (A1): `rescheduled_by_section_cpsat` collected
   ALL ops in budget-exceeding sections, but per-section CP-SAT only processed
   `filtered_global_indices` (AddImm alternatives excluded). Valid shift hints
   for filtered ops were incorrectly excluded. **Fix**: Track
   `actually_rescheduled_op_ids` — only ops sent to per-section CP-SAT.

2. **ConstChoice hint loss** (A2): Per-section CP-SAT returns its ConstChoice
   decisions, but the code removed ALL ConstChoice ops from the section first,
   then only re-inserted CP-SAT's results. Filtered AddImm ops (never sent to
   CP-SAT) were permanently lost. **Fix**: Re-insert filtered ops that were in
   greedy's original `greedy_const_add_imm_ops`.

3. **Post-elision ConstChoice filtering** (A3): SelectChoice hints were filtered
   against `op_ids_after_elision`, but ConstChoice hints were not — risking
   stale hints for barrier-elided ops. **Fix**: Apply same filter pattern.

A `debug_assertions` invariant now verifies all hint op IDs exist in the
post-elision op set (shift, SelectChoice, and ConstChoice).

---

## Implemented Architecture (Track 5)

> **Status**: SchedulingCandidate, SchedulingContext, and CascadePolicy are
> implemented and tested. The hint pipeline fix (Track 4) is also implemented.
> See `pressure_policy/mod.rs` for types and `pressure_policy/cascade.rs` for
> the CascadePolicy. ASAP inflation is not yet implemented (future work).

### SchedulingCandidate Struct (IMPLEMENTED)

Replaced the raw tuple `(op_index, luc, words_added, words_freed)` with a
richer struct in `pressure_policy/mod.rs`:

```rust
pub struct SchedulingCandidate {
    pub op_index: usize,
    pub luc: usize,                      // last-use count
    pub words_added: usize,              // scratch words this op would add
    pub words_freed: usize,              // scratch words freed by scheduling this op
    pub asap: Option<i32>,               // earliest feasible cycle (None if no bounds)
    pub alap: Option<i32>,               // latest feasible cycle (None if no bounds)
    pub engine: Engine,                  // ALU, VALU, Load, Store, Flow
    pub is_shiftable: bool,              // whether this VALU op can shift to ALU
    pub unscheduled_consumer_count: usize, // fan-out to unscheduled ops
}
```

Helper methods: `slack() -> Option<i32>` (ALAP - ASAP), `net_pressure() -> i64`.

### SlotTrackerView + SchedulingContext (IMPLEMENTED)

```rust
pub struct SlotTrackerView {
    pub alu_used: usize,
    pub valu_used: usize,
    pub load_used: usize,
    pub store_used: usize,
    pub flow_used: usize,
}

pub struct SchedulingContext<'a> {
    pub cycle: usize,
    pub live_words: usize,
    pub scratch_budget: usize,
    pub slots: &'a SlotTrackerView,
}
```

The `PressurePolicy::sort_ready` signature now takes
`(&self, &mut [SchedulingCandidate], &SchedulingContext)` instead of raw tuples.
All 5 policies (Greedy, PressureThrottled, PressureGated, Adaptive, Cascade)
use this interface. The call site in `ssa_greedy/mod.rs` builds candidates with
ASAP/ALAP from bounds, engine from `ops[idx].engine`, shiftable from the
`shiftable_set`, and unscheduled consumer count from `consumers[idx]`.

### CascadePolicy: 5-Level Strict Priority (IMPLEMENTED)

Implemented in `pressure_policy/cascade.rs`. Opt-in via
`--scheduling-pressure-policy cascade` (default remains Adaptive).

```
Level 1: Engine Saturation (higher saturation = first)
  → VALU 5/6 → score 3, VALU 4/6 → score 2
  → Load/Store/Flow at 1/capacity → score 3
  → ALU 10/12 → score 2
  → Below those thresholds → 0

Level 2: Slack (lower = first, critical path wins)
  → Ops without bounds get max slack (deferred)

Level 3: LUC (higher = first, frees more scratch)

Level 4: Net Pressure (only above soft threshold 85%)
  → Lower net pressure = first (prefer ops that reduce pressure)

Level 5: ASAP tiebreak (lower = first), then op_index
```

`allow_shift`: same graduated zones as AdaptivePolicy.
`should_defer_for_pressure`: always false.

### ASAP Inflation: Strategic Deferral (NOT YET IMPLEMENTED)

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

---

## Implementation Roadmap

| Priority | Change | Status | Impact | Risk | Effort |
|----------|--------|--------|--------|------|--------|
| 1 | LLVM-style priority cascade (CascadePolicy) | **DONE** | High — structured prioritization replaces ad-hoc blending | Low — sort_ready is already the control point | ~150 lines |
| 2 | ASAP inflation for deferral | TODO | High — enables pressure-aware deferral without cascading deadlock | Low — only affects ops with slack, reversible | ~80 lines |
| 3 | Multi-greedy via rayon | TODO | Medium — run multiple policies in parallel, pick best | Low — policies are pure functions, trivially parallelizable | ~50 lines |
| 4 | Hint pipeline fix (3 inconsistencies) | **DONE** | Medium — fixes shift over-exclusion, ConstChoice loss, post-elision filtering | Low — correctness fix, no behavioral change for correct hints | ~60 lines |
| 5 | Bidirectional scheduling | TODO | High — bottom-up pass for pressure, top-down for ILP | Medium — requires reverse-direction scheduling infrastructure | ~300 lines |

### Dependencies

- ~~Priority 1 (CascadePolicy) is a prerequisite for Priority 2 (ASAP inflation)~~ **Done** — CascadePolicy provides the structured context
- Priority 2 (ASAP inflation) can now be built on the CascadePolicy infrastructure
- Priority 3 (multi-greedy) is independent and can be done at any time
- ~~Priority 4 (hint fix) is independent and should be done soon~~ **Done** — 3 hint pipeline inconsistencies fixed
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
