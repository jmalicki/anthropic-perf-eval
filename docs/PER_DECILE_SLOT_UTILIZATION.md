# Per-decile slot utilization for optimization

Per-decile slot utilization is slot usage (and optionally scratch) grouped by deciles of the schedule: decile 0 = first 10% of cycles, decile 1 = next 10%, and so on. It helps identify where engines are saturated or idle so you can target further optimizations.

## How to produce it

1. **Prerequisite:** Scheduler output JSON (same as for bulk-move). Generate with:
   ```bash
   uv run generate_ssa_kernel.py --output-json scheduler_output.json
   ```
   Use `--rounds 2` for quick iteration.

2. **Run the script:**
   ```bash
   uv run scripts/slot_usage_by_decile.py [scheduler_output.json]
   ```
   Default input is `scheduler_output.json` if omitted.

The script reports per-decile ALU, VALU (FMA / vbroadcast / other), Load (const vs other), Store, Flow (add_imm / vselect / other), and scratch if `liveness_curve` is present in the JSON.

## How to use it for optimization

- **Look for imbalance:** In a given decile, if one engine is at or near 100% and another is underutilized, that suggests a possible optimization (move work from the saturated engine to the free one, or change encoding so more work uses the free engine).
- **Examples:**
  - **Load saturated, Flow free** → Prefer ConstChoice as add_imm (Flow) instead of Load to free Load slots for data loads. See [first_decile_const_to_add_imm.md](first_decile_const_to_add_imm.md) for a worked example.
  - **VALU saturated, ALU free** → Consider VALU→ALU shift or op mix so some work uses ALU slots.
- **Same JSON as bulk-move:** The same scheduler output JSON feeds both decile analysis and the bulk-move simulator. Use bulk-move to debug compaction (empty cycles, blocking constraints); use per-decile slot utilization to find slot-balance and encoding opportunities. The workflows complement each other.

## References

- Script: [scripts/slot_usage_by_decile.py](../scripts/slot_usage_by_decile.py)
- Worked example (first decile, Const→add_imm): [docs/first_decile_const_to_add_imm.md](first_decile_const_to_add_imm.md)
- Bulk-move (compaction): [.cursor/skills/use-bulk-move/SKILL.md](../.cursor/skills/use-bulk-move/SKILL.md)
