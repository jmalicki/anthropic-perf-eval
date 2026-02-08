# Cascading Deferral in Scheduling Pressure Policies

## Problem Statement

The greedy scheduler uses **pressure policies** to manage scratch memory (register
file) pressure. The `AdaptivePolicy` and `PressureGatedPolicy` both implemented a
**binary gate** via `should_defer_for_pressure()`:

```
if live_words > 95% of budget:
    if words_added > words_freed:
        DEFER this op  // do not schedule it this cycle
```

This works when the ready queue contains a mix of pressure-increasing and
pressure-decreasing ops. But on unrolled VLIW kernels (like our tree traversal +
hashing workload), the ready queue is dominated by:

- **vbroadcasts**: +8 words added, 0 freed (vector-width allocation)
- **ALU intermediates**: +1–4 words added, 0 freed (no consumer scheduled yet)

When **every** ready op has positive net pressure, the binary gate defers them
**all**. This triggers:

1. `made_progress = false` (no op was scheduled this cycle)
2. The force-schedule emergency fallback fires
3. `budget_exceeded = true` is set
4. Peak liveness spikes because the force-scheduled op ignores the budget

The force-schedule mechanism is **reactive** (breaks deadlock after it occurs),
not **preventive**. A better design avoids the deadlock entirely.

### Observed Impact

On the 4-round production kernel (forest_height=10, batch_size=256, no VALU shift,
greedy-only), the Adaptive policy in Gated mode caused force-scheduling, setting
`budget_exceeded = true`. Since the greedy schedule provides CP-SAT warm-start
hints, its quality directly affects solver convergence time.

## How Production Compilers Avoid This

### LLVM GenericScheduler — Hierarchical Heuristics, Not Binary Gates

LLVM's `GenericScheduler` (in `MachineScheduler.cpp`) uses `tryCandidate()`, a
cascading comparator that evaluates candidates on multiple criteria:

1. **RegExcess** — prefer ops that reduce excess register pressure (soft priority)
2. **RegCritical** — prefer ops that reduce critical-set pressure (soft priority)
3. **Resource pressure** — balance functional unit utilization
4. **Latency** — prefer ops on the critical path
5. **NodeOrder** — tie-breaker for determinism

Key insight: `RegExcess` and `RegCritical` are **soft priorities in a comparator**,
not absolute blocks. When all ops increase pressure, the scheduler falls through
to secondary heuristics (resource pressure, latency, `NodeOrder`). The scheduler
**always selects something** — it just prefers different things based on pressure.

### GCC `-fsched-pressure` — Proportional Excess Costs

GCC's register-pressure-aware scheduling computes an **excess cost (ECC)** per
instruction: just 1 per potential spill, not a high penalty. Instructions are
ranked by `(lowest ECC + delay)` then `(lowest delay)` then normal priority.

The algorithm is described as "quite timid" but **never completely blocks** all
options. Proportional costs allow graceful trade-offs instead of binary cliffs.

### Hsu's CSP/CSR (1988) — Mode Switching, Not Gating

Goodman and Hsu described two scheduling modes:

- **CSP** (Code Scheduling for Pipelines): optimize for pipeline/ILP
- **CSR** (Code Scheduling for Registers): optimize for register freeing

Below the threshold: CSP mode (maximize ILP). Above the threshold: CSR mode
(always select the **best op for pressure reduction** from the ready queue).

In CSR mode, no op gets indefinitely deferred; scheduling criteria change,
but scheduling continues. This is mode switching, not gating.

### Unison — Constraint Programming Co-optimization

Unison co-optimizes scheduling and register allocation via constraint programming.
Cascading deferral doesn't arise because the solver explores the full search space.
It has been shown to solve functions up to ~946 instructions optimally.

### Academic (SLIL) — Proportional Per-Instruction Cost

Traditional register pressure minimization optimizes **peak** pressure, which
creates a cliff at the threshold. SLIL (Sum of Live Interval Lengths) instead
optimizes the **sum of live interval lengths** across all program points.

This smooth, proportional cost function prevents the "all ops fail at one
threshold" problem because each op contributes a distributed cost rather than
a binary pass/fail at a single pressure point.

## Key Insight

> Production compilers **never** use binary gates for register pressure. They use
> soft priority biases (LLVM), proportional costs (GCC), or mode switching (Hsu).
> The scheduler always schedules something — it just changes what it prefers.

## Our Fix: Graduated Composite Score (LLVM-style)

We replaced the binary gate in `AdaptivePolicy` and `PressureGatedPolicy` with a
**graduated composite score** in `sort_ready()`:

```
pressure_fraction = live_words / scratch_budget

if pressure_fraction <= soft_threshold (85%):
    Sort by LUC descending (pure greedy — maximize ILP)
else:
    pressure_weight = (fraction - soft_threshold) / (1.0 - soft_threshold)
    // Clamp to [0, 1]
    composite_score = pressure_weight * net_pressure
                    - (1 - pressure_weight) * LUC
    Sort by composite_score ascending (lower = better)
```

- At 85% pressure: weight ≈ 0 → pure LUC (greedy behavior)
- At 90% pressure: weight ≈ 0.33 → blend of LUC and pressure
- At 95% pressure: weight ≈ 0.67 → pressure-dominant, LUC secondary
- At 100% pressure: weight = 1.0 → pure net-pressure minimization

`should_defer_for_pressure()` now always returns `false` for both Adaptive and
PressureGated policies. Pressure awareness is entirely handled by sort order.

## References

- **LLVM**: `GenericScheduler` class, `tryCandidate()` in `llvm/lib/CodeGen/MachineScheduler.cpp`
- **GCC**: RFC for alternative `-fsched-pressure` implementation (2011)
- **Goodman & Hsu (1988)**: *Code Scheduling and Register Allocation in Large Basic Blocks*, ICS '88
- **Touati (2005)**: *Early Control of Register Pressure for Software Pipelined Loops*
- **Berson et al. (1998)**: *URSA: A Unified Resource and Schedule Allocator for Registers, Functional Units, and Memory*
- **Shobaki & Gordon (2021)**: *Register-Pressure-Aware Instruction Scheduling Using Ant Colony Optimization*
- **Shobaki et al. (2019)**: *Exploring Alternative Cost Functions for Register-Pressure-Aware Instruction Scheduling*, ACM TACO
