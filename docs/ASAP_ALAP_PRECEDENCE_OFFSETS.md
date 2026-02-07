# Precedence Offsets for ASAP/ALAP

This document defines how precedence (RAW dependencies) affects **ASAP** and **ALAP** bounds. The key point: **ASAP uses the minimum delay** (earliest the consumer could run); **ALAP uses the maximum delay** (latest the predecessor can run so the consumer is still valid).

## Semantics of ASAP and ALAP

- **ASAP[op]** = “This operation could not possibly run earlier than this cycle under any circumstances.” So it is the **earliest** cycle the op can run, determined by the **earliest** time any dependency can produce its result.
- **ALAP[op]** = “This operation could not possibly run later than this cycle.” For a predecessor, the latest it can run is constrained by when the consumer must run; that in turn depends on the **latest** time the predecessor’s result can be ready.

So for a RAW edge (pred → succ):

- **ASAP[succ]** (consumer’s lower bound): use **minimum** delay → ASAP[succ] ≥ ASAP[pred] + **min_delay**.
- **ALAP[pred]** (predecessor’s upper bound): use **maximum** delay → ALAP[pred] ≤ ALAP[succ] − **max_delay**.

For normal and SelectChoice FMA, min_delay = max_delay. For shiftable VALU they differ.

## Dependency Types and Offsets

| Type | Min delay (for ASAP) | Max delay (for ALAP) | Meaning |
|------|----------------------|----------------------|---------|
| **Normal** | 1 | 1 | Result visible at pred_cycle + 1. Earliest and latest are the same. |
| **SelectChoice FMA delay** | 2 | 2 | On FMA arm: vsub at +1, FMA at +2; consumer reads FMA result. So earliest and latest are 2. |
| **Consumer of shifted VALU** | 1 | 8 | Earliest: pred not shifted or completes in 1 cycle → result at +1. Latest: when shifted, completion_cycle can be up to pred_cycle + 7, so result at +8. |

**Rule:**

- When **tightening ASAP[succ]** (or checking a lower bound on the consumer): use **min_delay** (1, 2, or 1 for shiftable).
- When **tightening ALAP[pred]** (or checking an upper bound on the predecessor): use **max_delay** (1, 2, or 8 for shiftable).

The CP-SAT model enforces the same constraints (sections 5, 5c, 5d in `cpsat_scheduler.rs`). Bounds must be valid for every resolution (every choice of ConstChoice, SelectChoice, VALU shift).

## Trait Design (types.rs)

Expose two values from the op (as predecessor), so call sites stay free of if/else:

- **For ASAP (consumer bound):** `precedence_offset_for_asap(&self, predecessor_is_shiftable: bool) -> i32`  
  Returns min_delay: 1 (normal), 2 (SelectChoice FMA), 1 (shiftable).
- **For ALAP (predecessor bound):** `precedence_offset_for_alap(&self, predecessor_is_shiftable: bool) -> i32`  
  Returns max_delay: 1 (normal), 2 (SelectChoice FMA), 8 (shiftable).

Implementation for `SsaOp`:

- SelectChoice with `sub_temp_vreg.is_some()` → both return 2.
- Else if `predecessor_is_shiftable` → ASAP returns 1, ALAP returns 8.
- Else → both return 1.

Adding a new op only extends this impl; diagnostic and precedence-consistency pass use the two methods and do not branch on dependency type.
