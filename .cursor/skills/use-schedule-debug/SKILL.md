---
name: use-schedule-debug
description: Use per-decile slot utilization and scheduler output JSON to find schedule optimization opportunities. Use when investigating whether further optimizations are possible, analyzing slot balance over time, or looking for encoding improvements (e.g. Const→add_imm, VALU shift).
---

# Schedule debug: per-decile slot utilization

Use **per-decile slot utilization** to see where each engine (ALU, VALU, Load, Store, Flow) is saturated or idle across the schedule. That helps identify optimization opportunities (e.g. move work to underused engines or change encodings).

## When to use

- **Further optimizations?** — After compaction (bulk-move) or when makespan is still high, check per-decile utilization to find imbalanced deciles.
- **Slot balance** — One engine at 100% and another free in the same decile suggests a possible encoding or shift opportunity.
- **Worked example** — First decile Load saturated + Flow free → Const→add_imm to free Load slots: see [docs/PER_DECILE_SLOT_UTILIZATION.md](../../docs/PER_DECILE_SLOT_UTILIZATION.md) and [docs/first_decile_const_to_add_imm.md](../../docs/first_decile_const_to_add_imm.md).

## How to run

1. **Prerequisite:** Scheduler output JSON (same as bulk-move). Generate with:
   ```bash
   uv run generate_ssa_kernel.py --output-json scheduler_output.json
   ```
   Use `--rounds 2` for quick iteration.

2. **Per-decile slot usage and scratch:**
   ```bash
   uv run scripts/slot_usage_by_decile.py [scheduler_output.json]
   ```
   Default input is `scheduler_output.json` if omitted.

## Full doc

For full description (what to look for, how to use it for optimization, relation to bulk-move), see **[docs/PER_DECILE_SLOT_UTILIZATION.md](../../docs/PER_DECILE_SLOT_UTILIZATION.md)**.
