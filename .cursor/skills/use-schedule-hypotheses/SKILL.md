---
name: use-schedule-hypotheses
description: Estimate impact of vbroadcast elimination, element-wise chaining, ConstChoice, or VALU shift before changing the solver or encoder; run hypothesis quantification on scheduler output JSON.
---

# Schedule hypotheses: quantify optimization ideas

Use the **hypothesis quantification script** on scheduler output JSON to estimate whether vbroadcast elimination, element-wise ALU chaining, ConstChoice mix, or VALU shift could improve cycles. Use when analyzing an optimal or hand-tuned schedule to see which strategies might help, or before changing the solver/encoder.

## When to use

- **Before changing the solver or encoder** — Estimate whether a strategy could improve cycles (vbroadcast eliminable count, slack distribution, decile 0 add_imm vs const).
- **After producing an optimal or hand-tuned schedule** — Run hypothesis metrics to see eliminable vbroadcasts, element-wise slack, and decile-0 ConstChoice mix.

## How to use

1. **Prerequisite:** Scheduler output JSON from a run you care about (e.g. `uv run generate_ssa_kernel.py --output-json scheduler_output.json`). For vbroadcast and element-wise slack, run with `--enable-valu-shift` so the JSON includes `shifted_alu_cycles`.

2. **Run hypothesis quantification script:**
   ```bash
   uv run scripts/quantify_schedule_hypotheses.py [scheduler_output.json]
   uv run scripts/quantify_schedule_hypotheses.py --markdown scheduler_output.json   # renderable output
   ```
   Interpret: eliminable vbroadcast count (and VALU slot saving), element-wise slack distribution and theoretical cycle bound, decile 0 add_imm vs const.

3. **VALU shift A/B:** Run the generator with and without `--enable-valu-shift` (same `--cpsat-time-limit`, `--rounds`), save to two JSONs. Compare `total_cycles`, `peak_live_words`, `budget_exceeded`. Run `slot_usage_by_decile.py` on both to compare decile utilization.

4. **Precedence / movement:** Use [use-bulk-move](../use-bulk-move/SKILL.md) (`scripts/simulate_bulk_move.py`) to see if empty cycles can be compacted and which constraint blocks. Use [use-whatif-shift](../use-whatif-shift/SKILL.md) (`scripts/what_if_shift.py`) to force shift for specific ops and compare cycles.

## References

- Script: [scripts/quantify_schedule_hypotheses.py](../../scripts/quantify_schedule_hypotheses.py)
- Plan: [docs/HYPOTHESIS_QUANTIFICATION_PLAN.md](../../docs/HYPOTHESIS_QUANTIFICATION_PLAN.md)
- Schedule debug (per-decile): [.cursor/skills/use-schedule-debug/SKILL.md](../use-schedule-debug/SKILL.md)
- Bulk-move: [.cursor/skills/use-bulk-move/SKILL.md](../use-bulk-move/SKILL.md)
- What-if shift: [.cursor/skills/use-whatif-shift/SKILL.md](../use-whatif-shift/SKILL.md)
