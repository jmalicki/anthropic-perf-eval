# Round progress tracking (SHA vs cycles)

Record git SHA, exact command, cycles, peak liveness, solver status, and validation status for each round count.

**Important:** Cycles vary significantly with batch size. Lower cycle counts often come from smaller batch sizes (32, 64) rather than batch 256.

Data sources: scheduler output JSON files (`scheduler_output*.json`) and manual runs.

**Solver status:** OPTIMAL (proven optimal, `lb_cycles == total_cycles`), FEASIBLE (solution found but not proven optimal), or GREEDY (greedy scheduler, no optimality proof).

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

**Depth-3 validation note:** The validation failure for tree-reduce depth 3 at 614354c1 was later resolved in 29ad610b—the same configuration (4r, 256, depth 3) passed validation there. So identical configs show different outcomes because of fixes in between (e.g. scheduler/address-assign or validation; see BUGS.md for related scratch/liveness issues).

---

## 5 rounds

| Date | SHA | Batch | Cycles | Peak live | Status | Validation | Notes | Source |
|------|-----|-------|--------|-----------|--------|------------|-------|--------|
| 2026-02-09 | 16d5766a | 256 | 568 | N/A | OPTIMAL | N/A | depth-3 tree-reduce | scheduler_5r_depth3.json |
| 2026-02-09 | 16d5766a | 256 | 704 | N/A | OPTIMAL | N/A | depth-4 tree-reduce | scheduler_5r_depth4.json |

---

## Notes

- **Batch size matters:** Cycles scale with batch size. Compare runs with the same batch size.
- **Solver status:** OPTIMAL = proven optimal (`lb_cycles == total_cycles`), FEASIBLE = solution found but not proven optimal (`lb_cycles < total_cycles`), GREEDY = greedy scheduler (no `lb_cycles` field).
- Commands use default options unless specified (e.g. `--enable-valu-shift`, `--tree-reduce-max-depth`)
- Validation = Python kernel verification against reference kernel
- Peak live = peak simultaneous scratch words (budget = 1536)
- Date = file modification date (when the JSON was generated)
- SHA = git commit SHA (truncated to 8 chars for readability)
