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

1. **Prerequisite:** Scheduler output JSON. **Kernel generation always writes** a schedule JSON. If you omit `--output-json`, it writes **`scheduler_output.json`** by default (with `run_options`, `git_sha`, and optional `comment`).

   **For this workflow (schedule debug / comparing runs):** each time you run the generator, **specify a different file** with `--output-json` and add `--comment` so runs don’t overwrite each other and each file is identifiable:
   ```bash
   uv run generate_ssa_kernel.py --rounds 2 --output-json scheduler_output_2r_60s.json --comment "2-round 60s CP-SAT"
   # next run, use a different path, e.g.:
   uv run generate_ssa_kernel.py --rounds 4 --output-json scheduler_output_4r_handopt.json --comment "4-round hand-tuned"
   ```
   If you don’t pass `--output-json`, the run still writes to the default `scheduler_output.json`. The JSON produced (default or specified path) is what you pass to the decile script.

2. **Per-decile slot usage and scratch:**
   ```bash
   uv run scripts/slot_usage_by_decile.py [scheduler_output.json]
   uv run scripts/slot_usage_by_decile.py --markdown scheduler_output.json   # markdown table (renders in docs/notebooks)
   ```
   Default input is `scheduler_output.json` if omitted.

   **Note:** The default output is plain text (for the terminal). To get a **markdown table that renders** in docs, notebooks, or chat, use `--markdown` (or `-m`). When pasting or saving the table for rendering, always run with `--markdown`.

## Table columns (subtype breakdown)

The script prints a table with **subtype columns** so you can see what is using each engine:

- **VALU:** total (used/cap %) plus **VALU vb** | **VALU FMA** | **VALU other**. vb = vbroadcast, FMA = multiply_add/sub (not shiftable); other = shiftable VALU ops.
- **Flow:** total (used/cap %) plus **Flow add_imm** | **Flow vselect** | **Flow other**. add_imm = ConstChoice when true; vselect = SelectChoice when false; other = e.g. jump, halt.
- **Load:** total (used/cap %) plus **Load const** | **Load other**. const = ConstChoice when Load chosen; other = vload, load, load_offset.

Use these columns to spot imbalances (e.g. Load const high and Flow add_imm zero → Const→add_imm opportunity).

## Full doc

For full description (what to look for, how to use it for optimization, relation to bulk-move), see **[docs/PER_DECILE_SLOT_UTILIZATION.md](../../docs/PER_DECILE_SLOT_UTILIZATION.md)**.
