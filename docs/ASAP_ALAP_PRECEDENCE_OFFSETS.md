# Precedence Offsets for ASAP/ALAP

## Purpose: Why We Compute ASAP/ALAP Bounds

ASAP (As Soon As Possible) and ALAP (As Late As Possible) bounds serve a single purpose: **defining the domain of the `cycle_vars[i]` variables** in the CP-SAT model. Each op's cycle variable is created with domain `[ASAP[i], ALAP[i]]`.

This is purely a **search space restriction** for the solver. The actual correctness of the schedule is enforced by the CP-SAT model's constraints (sections 5, 5c, 5d in `cpsat_scheduler.rs`). The ASAP/ALAP domains only need to **contain** the optimal solution — they don't need to exclude every infeasible value.

Crucially:
- **Wider domains** = more freedom for the solver = slower but always finds a solution if one exists
- **Tighter domains** = faster solver (less search space) but risk excluding the optimal solution or making the problem infeasible

When in doubt, **prefer wider domains.** A wider domain that includes some infeasible values is harmless — the solver's constraint propagation will prune them. A domain that's too tight causes INFEASIBLE or UNKNOWN.

## Two Roles of Delay Values

Every RAW dependency edge (predecessor → consumer) has a latency that depends on the predecessor's execution mode. For example, a shiftable VALU op has latency 1 when kept as VALU, but latency up to 8 when shifted to ALU.

This gives us two delay values:
- **min_delay**: Latency in the best case (fastest mode)
- **max_delay**: Latency in the worst case (slowest mode)

These serve **different purposes** in different computations:

### For ASAP (consumer's lower bound)

**Use min_delay.** ASAP asks: "What is the earliest cycle this consumer could possibly execute?" The answer is: the earliest, assuming every predecessor completes as fast as possible (best-case latency).

```
ASAP[consumer] >= ASAP[predecessor] + min_delay
```

### For ALAP domain bounding (predecessor's upper bound)

**Use min_delay** (not max_delay). ALAP for domain bounding asks: "What is the latest cycle this predecessor could possibly be placed?" If the solver chooses NOT to shift the op (keeping latency = min_delay), the predecessor CAN run as late as `ALAP[consumer] - min_delay`. The CP-SAT model constraints handle the shifted case — if the solver shifts the op, the `completion_cycle` variable and section 5c constraints will prevent the consumer from running too early.

```
ALAP[predecessor] <= ALAP[consumer] - min_delay   (for domain bounding)
```

### For horizon sizing (worst-case schedule length)

**Use max_delay.** The horizon must be large enough that backward ALAP propagation doesn't produce negative values. `compute_max_latency_critical_path` does a forward pass using max_delay to find the worst-case critical path length.

```
horizon >= 2 * max_latency_critical_path
```

## Why NOT Use max_delay for ALAP Domain Bounding

This is the critical design decision. Using `ALAP[pred] <= ALAP[consumer] - max_delay` is mathematically correct for the worst case, but **practically harmful** because:

### 1. Chain Amplification

In a chain of N shiftable VALU ops, the ALAP loss compounds:
- With max_delay=8: total loss from chain = `8 * N` cycles
- With min_delay=1: total loss from chain = `1 * N` cycles

For the 2-round kernel (896 shiftable ops), dependency chains of depth 20+ exist. Using max_delay=8 loses **140+ extra cycles** from ALAP domains compared to min_delay=1.

### 2. Slot Tightening Interaction

After ASAP/ALAP computation, `tighten_asap_alap_by_slots` and `tighten_asap_alap_by_scratch` further narrow domains based on resource pressure. These tightening passes are **oblivious to precedence constraints** — they don't know that narrowing an already-tight domain could violate a precedence edge. Starting with narrow domains from max_delay=8 means tightening can easily create empty domains (ASAP > ALAP), making the problem infeasible.

### 3. The CP-SAT Model Already Handles It

The CP-SAT model's constraints (section 5c) explicitly handle shifted VALU latency:

```
cycle[consumer] >= completion_cycle[predecessor] + 1
completion_cycle = max(cycle_var, alu_cycle_0, ..., alu_cycle_7)
```

When the solver shifts an op, `completion_cycle` extends to `cycle + 7`, naturally constraining the consumer. When NOT shifted, `completion_cycle = cycle_var`, giving latency 1. The domain doesn't need to pre-encode this decision — it just needs to contain both possible values.

### 4. Decision Variables vs. Fixed Latencies

The key insight: shiftable VALU's latency is a **decision variable**, not a fixed property. The solver decides whether to shift each op. Using max_delay in ALAP assumes the worst case **for every op simultaneously**, but the solver can (and does) choose a mix of shifted and non-shifted ops.

In contrast, SelectChoice FMA has min_delay = max_delay = 2, because the consumer always sees the FMA result regardless of the SelectChoice decision. And normal ops have min_delay = max_delay = 1. For these, there's no distinction between the two uses.

## Dependency Types and Offsets

| Type | min_delay | max_delay | Domain bounding uses | Horizon sizing uses |
|------|-----------|-----------|---------------------|-------------------|
| **Normal** | 1 | 1 | 1 (both same) | 1 |
| **SelectChoice FMA** | 2 | 2 | 2 (both same) | 2 |
| **Shiftable VALU** | 1 | 8 | **1** (min_delay) | **8** (max_delay) |

## Trait Design (types.rs) — Single Source of Truth

The `PrecedenceOffset` trait in `types.rs` is the **single source of truth** for per-op precedence delays. It provides both min_delay and max_delay; call sites choose which to use based on context.

The trait is called on the **predecessor** (the op producing a result):

- **`precedence_offset_for_asap(&self, self_is_shiftable: bool) -> i32`** — Returns min_delay.
- **`precedence_offset_for_alap(&self, self_is_shiftable: bool) -> i32`** — Returns max_delay.

Where each is used:
- `compute_ssa_asap`: uses `precedence_offset_for_asap` (min_delay) — correct
- `compute_ssa_alap`: uses `precedence_offset_for_asap` (min_delay) — wide domains
- `compute_max_latency_critical_path`: uses `precedence_offset_for_alap` (max_delay) — horizon sizing
- `diagnose_precedence_bounds`: uses `precedence_offset_for_asap` (min_delay) — consistent with domains
- `fix_precedence_bounds_fixpoint`: uses `precedence_offset_for_asap` (min_delay) — consistent with domains

Implementation for `SsaOp`:
- SelectChoice with `sub_temp_vreg.is_some()` → both return 2.
- Else if `self_is_shiftable` → ASAP returns 1, ALAP returns 8.
- Else → both return 1.

## Horizon Sizing

When shiftable VALU ops are present, the worst-case critical path (using max_delay=8) can be much longer than the min-latency critical path (using min_delay=1). The helper `compute_max_latency_critical_path` does a forward pass using max_delay to determine the minimum required horizon. Both `schedule_ssa_cpsat` and `compute_ssa_asap_alap_bounds` use:

```
horizon = max(min_critical * 2, max_critical * 2, n)
```

This ensures the horizon is large enough for all modes of operation.

## Summary: The Design Principle

**ASAP/ALAP bounds are a hint to the solver, not a correctness guarantee.** They restrict the search space to speed up solving. The actual correctness comes from the CP-SAT model's constraints.

For decision variables (shift/no-shift), the domain must be wide enough to contain the solution for **any** choice the solver might make. This means using best-case (min_delay) offsets for domain bounding, while using worst-case (max_delay) offsets only where the absolute worst case matters (horizon sizing).
