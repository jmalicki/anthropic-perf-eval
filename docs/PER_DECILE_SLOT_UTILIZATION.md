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
- **First decile: “Could we have started more hashes?”** In many 4-round runs, decile 0 has high VALU and Load use but ~35% ALU free. **In principle, yes — round 1 for some groups could start in decile 0.** Dependencies are per-group: round 1 for group *g* only needs group *g*’s round 0 output. In the observed 4-round schedule, round 0 for all groups finishes by cycle ~14, but round 1 does not start until cycle ~91 (a ~77-cycle gap). So we could have started round 1 for groups that finished round 0 early (e.g. groups 0–4 by cycle 12) during decile 0 (cycles 0–44). The current schedule does not exploit that: it effectively batches by round (all round 0, then later all round 1). Better **per-group interleaving** (start round 1 for a group as soon as that group’s round 0 is done) could use the free ALU and VALU slots in cycles 15–44 for round-1 work instead of leaving a long idle gap. Separately, the free ALU in decile 0 can also be targeted by **VALU→ALU shift** for ops that are ready in that window; use `scripts/what_if_shift.py` or hypothesis quantification to test.
- **Same JSON as bulk-move:** The same scheduler output JSON feeds both decile analysis and the bulk-move simulator. Use bulk-move to debug compaction (empty cycles, blocking constraints); use per-decile slot utilization to find slot-balance and encoding opportunities. The workflows complement each other.

## References

- Script: [scripts/slot_usage_by_decile.py](../scripts/slot_usage_by_decile.py)
- Worked example (first decile, Const→add_imm): [docs/first_decile_const_to_add_imm.md](first_decile_const_to_add_imm.md)
- Bulk-move (compaction): [.cursor/skills/use-bulk-move/SKILL.md](../.cursor/skills/use-bulk-move/SKILL.md)
