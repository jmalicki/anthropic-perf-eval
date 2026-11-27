# Round progress tracking (SHA vs cycles)

Record git SHA, exact command, cycles, peak liveness, solver status, and validation status for each round count.

**Important:** Cycles vary significantly with batch size. Lower cycle counts often come from smaller batch sizes (32, 64) rather than batch 256.

Data sources: scheduler output JSON files (`scheduler_output*.json`) and manual runs.

**Solver status:** OPTIMAL (optimal for the ops that run emitted, `lb_cycles == total_cycles`), FEASIBLE (CP-SAT solution found, not shown optimal for that emission: `lb_cycles < total_cycles`, or `lb_cycles` absent with `run_options.use_cpsat` true), or GREEDY (greedy scheduler, no optimality proof). `lb_cycles` and OPTIMAL are statements about one emission and its encoding choices (`lb_scope: "emission"` in the JSON). They are not bounds on the problem: an emission with fewer ops can be faster.

---

## Unknown rounds (from JSON files)

| Date | SHA | Batch | Cycles | Peak live | Status | Notes | Source |
|------|-----|-------|--------|-----------|--------|-------|--------|
| 2026-02-03 | N/A | N/A | 104 | N/A | GREEDY | | scheduler_output_256.json |
| 2026-02-03 | N/A | N/A | 104 | N/A | GREEDY | | scheduler_output_debug.json |
| 2026-02-03 | N/A | N/A | 104 | N/A | GREEDY | | scheduler_output_fixed.json |
| 2026-02-03 | N/A | N/A | 104 | N/A | GREEDY | | scheduler_output_whatif_op244.json |
| 2026-02-04 | N/A | N/A | 192 | N/A | GREEDY | | scheduler_output_10s.json |
| 2026-02-04 | N/A | N/A | 192 | N/A | GREEDY | | scheduler_output_test.json |

---

## 2 rounds

| Date | SHA | Batch | Cycles | Peak live | Status | Validation | Notes | Source |
|------|-----|-------|--------|-----------|--------|------------|-------|--------|
| 2026-02-06 | d400bdd5 | 32 | 37 | N/A | OPTIMAL | N/A | baseline (tightening disabled) | scheduler_output_baseline_(tightening_disabled).json |
| 2026-02-06 | df846227 | 64 | 50 | N/A | FEASIBLE | N/A | slot-aware tightening enabled | scheduler_output_slot-aware_tightening_enabled.json |
| 2026-02-06 | df846227 | 64 | 52 | N/A | FEASIBLE | N/A | slot and scratch tightening enabled | scheduler_output_slot__and__scratch_tightening_enabled.json |
| 2026-02-05 | a38cc8ed | 256 | 176 | N/A | GREEDY | N/A | | scheduler_output_2round.json |
| 2026-02-05 | 79205ca8 | 256 | 192 | N/A | GREEDY | N/A | | scheduler_output_2r.json |

---

## 3 rounds

| Date | SHA | Batch | Command | Cycles | Peak live | Status | Validation | Notes | Source |
|------|-----|-------|---------|--------|-----------|--------|------------|-------|--------|
| 2026-02-06 | df846227 | 64 | N/A | 73 | N/A | FEASIBLE | N/A | scratch-aware tightening enabled | scheduler_output_scratch-aware_tightening_enabled.json |
| 2026-02-06 | df846227 | 64 | N/A | 80 | N/A | FEASIBLE | N/A | baseline tightening disabled | scheduler_output_baseline_tightening_disabled.json |
| 2026-02-06 | d400bdd5 | 64 | N/A | 86 | N/A | FEASIBLE | N/A | ASAP/ALAP tightening enabled | scheduler_output_asap_alap_tightening_enabled.json |
| 2026-02-10 | 1864ae7 | 256 | `uv run generate_ssa_kernel.py --rounds 3 --cpsat-time-limit 60 -q --validate` | 309 | 1269 | OPTIMAL | PASSED | Trait-based liveness refactor (LivenessDef/LivenessUse) | scheduler_output.json |

---

## 4 rounds

See [4-round-phase-results.md](4-round-phase-results.md) for detailed 4-round tracking.

**Ablation (2026-02-10):**
- **Experiment:** Method tree-reduce; depths compared 2 vs 3; 4 rounds, batch 256; 1800s CP-SAT; `--validate`.
- **depth-2:** 435 cycles, OPTIMAL, validation PASSED.
- **depth-3:** 443 cycles, OPTIMAL, validation FAILED.
- **Artifacts:** Results in `ablation_4r_depth2_vs_depth3.json`.
- **Analyzer:** `uv run python scripts/analyze_ablation.py ablation_4r_depth2_vs_depth3.json`.

Quick summary:
- **2a32f71** (Phase 1): 514 cycles, 1489 peak (greedy, batch 256)
- **68b1602** (tree-reduce depth3): 460 cycles, 1522 peak (greedy, valu-shift, batch 256)
- **ae4a158** (current): 513 cycles, 1488 peak (greedy, no valu-shift, batch 256)

Superseded (2026-11): ae4a158 was current in February 2026. Later 4-round figures at batch 256,
schedule / simulator cycles: 423 / 425 from CP-SAT started from the shaped greedy with
`--cpsat-split-shift-hint` ([GREEDY_SHAPE_ABLATION.md](GREEDY_SHAPE_ABLATION.md) section 6),
and 364 / 366 from the shaped greedy with `--greedy-justify 10`, greedy only (section 13 of the
same document). For 16 rounds see [16 rounds](#16-rounds) below.

### From JSON files (all batch 256)

| Date | SHA | Batch | Cycles | Peak live | Status | Validation | Notes | Source |
|------|-----|-------|--------|-----------|--------|------------|-------|--------|
| 2026-02-10 | 614354c1 | 256 | 435 | 1267 | OPTIMAL | PASSED | ablation: tree-reduce depth 2 (run_ablation.py 4r 1800s) | scheduler_output_depth-2.json |
| 2026-02-10 | 614354c1 | 256 | 443 | 1266 | OPTIMAL | FAILED | ablation: tree-reduce depth 3 (validation failed; resolved in 29ad610b) | scheduler_output_depth-3.json |
| 2026-02-10 | 29ad610b | 256 | 435 | 1267 | OPTIMAL | PASSED | depth-2 tree-reduce (validated) | scheduler_output_4r_256b_depth2.json |
| 2026-02-10 | 29ad610b | 256 | 443 | 1266 | OPTIMAL | PASSED | depth-3 tree-reduce (validated) | scheduler_output_4r_256b_depth3.json |
| 2026-02-09 | 16d5766a | 256 | 435 | N/A | OPTIMAL | N/A | depth-2 tree-reduce | scheduler_output_depth2.json |
| 2026-02-09 | 16d5766a | 256 | 443 | N/A | OPTIMAL | N/A | depth-3 tree-reduce | scheduler_output_depth3.json |
| 2026-02-08 | 1a4a1817 | 256 | 435 | N/A | OPTIMAL | N/A | scatter load | scheduler_output_scatter.json |
| 2026-02-08 | 1a4a1817 | 256 | 443 | N/A | OPTIMAL | N/A | treeselect | scheduler_output_treeselect.json |
| 2026-02-07 | d6461724 | 256 | 499 | N/A | OPTIMAL | N/A | depth-3, no valu-shift | scheduler_output_depth3_novs.json |
| 2026-02-07 | d6461724 | 256 | 506 | N/A | OPTIMAL | N/A | depth-2, no valu-shift | scheduler_output_depth2_novs.json |

**Caveat on OPTIMAL in the February rows (added 2026-11):** from 1a4a1817 (2026-02-08) on,
`generate_ssa_kernel.py` ran slot-capacity ASAP/ALAP tightening by default
(`tighten_asap_alap_by_slots`); at d6461724 it was opt-in. PR #84 (2026-11) removed that
function because, in its words, the rule behind it "excluded feasible, better schedules": at
6 rounds / batch 64 "a validated 160-cycle schedule had 24 ops outside the resulting domains".
OPTIMAL here therefore means optimal for the model as it was built then, with those domains.
Whether the February version of the rule excluded any schedule on these instances was not
checked, and the rows have not been re-measured since #79, #84 and #92 changed the CP-SAT
model. The same applies to the OPTIMAL rows of the 3-round and 5-round tables from 1864ae7
and 16d5766a.

**Depth-3 validation note:** The validation failure for tree-reduce depth 3 at 614354c1 was later resolved in 29ad610b—the same configuration (4r, 256, depth 3) passed validation there. So identical configs show different outcomes because of fixes in between (e.g. scheduler/address-assign or validation; see BUGS.md for related scratch/liveness issues).

---

## 5 rounds

| Date | SHA | Batch | Cycles | Peak live | Status | Validation | Notes | Source |
|------|-----|-------|--------|-----------|--------|------------|-------|--------|
| 2026-02-09 | 16d5766a | 256 | 568 | N/A | OPTIMAL | N/A | depth-3 tree-reduce | scheduler_5r_depth3.json |
| 2026-02-09 | 16d5766a | 256 | 704 | N/A | OPTIMAL | N/A | depth-4 tree-reduce | scheduler_5r_depth4.json |

---

## 16 rounds

Added 2026-11 from the documents named in the last column; nothing was re-run for this table.
All rows: batch 256, height 10, PROVED (dataflow replay) and PASSED (`--validate`). Cycles are
schedule / simulator. "Strict" means `./scripts/check_upstream.sh --strict` passed (unmodified
upstream correctness tests and every speed threshold).

**The greedy rows are development figures, not CP-SAT results.** None of these is optimal or a
bound.

"Shaped" = `--greedy-barriers none --greedy-priority wavefront --greedy-split-shift
--pressure-inflation-threshold 0.95`. "Load-tail" = shaped plus `--greedy-load-priority
final-last --greedy-groups-in-flight 32 --greedy-stagger-cycles 9 --merge-hash-stages
--index-form mirrored`.

| Configuration | Cycles (sched / sim) | Scheduler | Strict | Output | Source |
|---|---|---|---|---|---|
| Default emission, default greedy | 2,007 / 1,994 | GREEDY | not stated | values and indices | [GREEDY_SHAPE_ABLATION.md](GREEDY_SHAPE_ABLATION.md) section 1 |
| Default emission, shaped | 1,313 / 1,315 | GREEDY | yes | values and indices | same, section 1 |
| Default emission, shaped + `--greedy-justify 10` | 1,279 / 1,281 | GREEDY | yes | values and indices | same, section 13, PR #85 |
| Shaped + `--merge-hash-stages --index-form mirrored --cached-select bits` | 1,153 / 1,155 | GREEDY | yes | values and indices | [EMISSION_ABLATION.md](EMISSION_ABLATION.md) section 3.5, PR #36 |
| Load-tail + `--emission-plan docs/emission_plans/flow90.json` (index bits) | 1,039 / 1,041 | GREEDY | yes | values and indices | [GREEDY_SHAPE_ABLATION.md](GREEDY_SHAPE_ABLATION.md) section 0.1, PR #73 |
| Same plan + `--reassociate-gather-xor --index-early-terms final --select-xor-pushdown first-pass --greedy-justify 10` (best index-bit kernel) | 1,033 / 1,035 | GREEDY | yes | values and indices | [CRITICAL_PATH.md](CRITICAL_PATH.md), "Justification on the best kernels", PR #93 |
| Load-tail + `path_flow90.json --select-bit-source path --greedy-alu-priority native-first --greedy-flow-priority chain-first` + the same three rewrites and `--greedy-justify 10` (best recorded) | **1,012 / 1,014** | GREEDY | yes | values and indices | same, PR #93 |
| Shaped + `values_only_path_fix15_flow90.json` (no load-tail flags, no justify) | 1,081 / 1,083 | GREEDY | yes | **values only** | [EMISSION_STRUCTURE_MODEL.md](EMISSION_STRUCTURE_MODEL.md) section 6.1, PR #51 |
| Default emission, shaped + `--cpsat-split-shift-hint`, `--cpsat-deterministic-time 60` | 1,314 / 1,316 | CP-SAT, returned its start unchanged | yes | values and indices | [GREEDY_SHAPE_ABLATION.md](GREEDY_SHAPE_ABLATION.md) section 6, PR #57 |
| Default emission, shaped hint, `--cpsat-objective makespan --cpsat-deterministic-time 60` | 1,399 / 1,401, `lb_cycles` 1,271 | CP-SAT, did not improve on its hint | not stated | values and indices | [RELAXED_SCRATCH_REFERENCE.md](RELAXED_SCRATCH_REFERENCE.md), CP-SAT table |
| `flow90.json`, shaped, `--cpsat-deterministic-time 300` | 1,132 / 1,134, bound 1,000 | CP-SAT, stopped on the wall-clock backstop: not reproducible | not stated | values and indices | [EMISSION_STRUCTURE_MODEL.md](EMISSION_STRUCTURE_MODEL.md) section 5.1 |

- The CP-SAT rows were measured before #79 (stale ConstChoice alternatives dropped), #84 (the
  unsound energetic tightening rule replaced) and #92 (per-section scratch check changed), and
  have not been re-measured.
- The 1,012 and 1,033 rows are pinned by `tests/test_justify_on_best_pin.py` (ci_only).
- The 1,271 capacity floor belongs to the default emission only. It is not a bound on the
  problem, and the plan rows emit fewer ops.

---

## Notes

- **Batch size matters:** Cycles scale with batch size. Compare runs with the same batch size.
- **Solver status:** OPTIMAL = optimal for that run's emitted ops (`lb_cycles == total_cycles`), FEASIBLE = CP-SAT solution found but not shown optimal for that emission (`lb_cycles < total_cycles`, or no `lb_cycles` field with `run_options.use_cpsat` true), GREEDY = greedy scheduler (no `lb_cycles` field and `run_options.use_cpsat` false). None of these bounds the problem; `uv run python -m scripts.schedule_report <json>` shows the floor for an emission and the gap to it.
- Commands use default options unless specified (e.g. `--enable-valu-shift`, `--tree-reduce-max-depth`)
- Validation = Python kernel verification against reference kernel
- Peak live = peak simultaneous scratch words (budget = 1536)
- Date = file modification date (when the JSON was generated)
- SHA = git commit SHA (truncated to 8 chars for readability)
