---
name: use-theoretical-lower-bounds
description: Compute the capacity-only theoretical minimum makespan for a kernel using scripts/theoretical_lower_bounds.py. Use when the user asks for the theoretical lowest bound, tight-pack bound, or "if we could perfectly pack" the schedule.
---

# Theoretical Lower Bounds

Use **scripts/theoretical_lower_bounds.py** to compute the **capacity-only theoretical minimum makespan** for a kernel. This is the minimum number of cycles needed if ops could be perfectly packed by engine capacity alone—ignoring precedence, scratch pressure, and other scheduling constraints. The gap between this bound and the actual schedule shows how much overhead comes from dependencies and packing.

## When to use

- **User asks for "theoretical lowest bound"** or **"if we could tightly pack"** — Run the script on the scheduler output JSON.
- **Understanding schedule overhead** — See how much of the actual makespan is capacity-bound vs precedence/scratch.
- **Bottleneck identification** — Script reports which engine(s) drive the theoretical minimum (VALU, Load, etc.).

## Prerequisite

Scheduler output JSON from `generate_ssa_kernel.py --output-json` (e.g. `scheduler_output.json`). The file must contain `ops` or `ssa_ops`.

## How to run

```bash
# Default: scheduler_output.json
uv run python scripts/theoretical_lower_bounds.py

# Custom path
uv run python scripts/theoretical_lower_bounds.py path/to/schedule.json
```

## What it does

1. **Counts ops by engine** (VALU, ALU, Load, Store, Flow) from the JSON.
2. **VALU breakdown:** Classifies VALU ops as shiftable vs non-shiftable (FMA and vbroadcast cannot shift to ALU).
3. **Current state (no VALU shift):** Computes ceil(ops/capacity) per engine; capacity is 6 (VALU), 12 (ALU), 2 (Load), 2 (Store), 1 (Flow). Makespan = max over engines.
4. **Theoretical minimum:** Sweeps over how many shiftable VALU ops to move to ALU (each becomes 8 ALU ops); picks the shift count that minimizes the capacity makespan. Reports that minimum and the bottleneck(s).
5. **Comparison:** Actual schedule cycles (from JSON) vs theoretical minimum; gap and % overhead.

**Note:** The bound assumes perfect packing with no dependencies or scratch limits. Real schedules are always ≥ this bound; often much higher.

## Output summary

- **Theoretical minimum makespan:** The capacity-only lower bound (cycles).
- **Bottleneck:** Engine(s) that determine that minimum (e.g. VALU and Load both need 14 cycles).
- **Gap:** Actual − theoretical; script explains it as precedence, scratch, and scheduling inefficiencies.

## Reference

- Script: [scripts/theoretical_lower_bounds.py](../../scripts/theoretical_lower_bounds.py)
