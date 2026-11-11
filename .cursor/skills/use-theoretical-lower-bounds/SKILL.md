---
name: use-theoretical-lower-bounds
description: Compute the capacity-only minimum makespan for the ops one kernel emission contains, using scripts/theoretical_lower_bounds.py. Use when the user asks for the tight-pack bound or "if we could perfectly pack" this emission. It is a floor for the emitted ops, not a bound on the problem; for the usual per-run answer prefer use-schedule-report.
---

# Theoretical Lower Bounds

Use **scripts/theoretical_lower_bounds.py** to compute the **capacity-only minimum makespan for the ops this emission contains**. This is the minimum number of cycles needed if those ops could be perfectly packed by engine capacity alone—ignoring precedence, scratch pressure, and other scheduling constraints. The gap between this floor and the actual schedule shows how much overhead comes from dependencies and packing.

**Scope: this emission only.** The number is slot arithmetic over the ops the compiler emitted. A kernel that emits fewer ops has a lower floor, so this is **not a bound on the problem** (our 16-round emission: 1,271; public kernels reach 863; see [docs/BOUND_GAP_ANALYSIS.md](../../../docs/BOUND_GAP_ANALYSIS.md)). Never report it as "the theoretical minimum" without that qualifier.

**Prefer `scripts/schedule_report.py`** ([use-schedule-report](../use-schedule-report/SKILL.md)) for the everyday question "how good is this schedule": it gives the same floor plus per-round spans and utilisation over time. This script adds the breakdown by VALU opcode and the ConstChoice/SelectChoice counts. It counts SSA ops (each ConstChoice as a load), so its load and flow terms differ slightly from the report, which counts the ops as lowered.

## When to use

- **User asks for the capacity floor of an emission** or **"if we could tightly pack"** — Run the script on the scheduler output JSON, and state that the result is for this emission.
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

**Note:** The floor assumes perfect packing with no dependencies or scratch limits. Real schedules of the same ops are always ≥ this floor; often much higher. A different emission (fewer ops) can be below it.

## Output summary

- **Theoretical minimum makespan:** The capacity-only floor for this emission (cycles); not a bound on the problem.
- **Bottleneck:** Engine(s) that determine that minimum (e.g. VALU and Load both need 14 cycles).
- **Gap:** Actual − theoretical; script explains it as precedence, scratch, and scheduling inefficiencies.

## Reference

- Script: [scripts/theoretical_lower_bounds.py](../../scripts/theoretical_lower_bounds.py)
