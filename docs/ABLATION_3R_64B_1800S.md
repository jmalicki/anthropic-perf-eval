# Ablation Analysis: 3 Rounds, Batch 64, 1800s Time Limit

**Configuration**: 3 rounds, batch size 64, 1800s CP-SAT time limit per run  
**Results file**: `ablation_results_3r_64b_1800s.json`  
**Logs**: `ablation_log_*.txt`

## Completion status

| Arm | Status | Makespan | Notes |
|-----|--------|----------|--------|
| **Baseline** (no tightening) | Completed | **80** | FEASIBLE, 15% gap |
| **Slot-aware only** | **INFEASIBLE** | — | Model **proven** to have no solution (not timeout) |
| **Scratch-aware only** | Completed | **73** | FEASIBLE, 6.85% gap |
| **Slot + scratch** | **INFEASIBLE** | — | Model **proven** to have no solution (not timeout) |

Only 2 of 4 arms produced a feasible schedule. Slot-only and Slot+scratch returned **INFEASIBLE**: the solver proved that no feasible schedule exists under those tightened bounds (over-tightening), as opposed to UNKNOWN (timeout without a solution).

## Results (completed arms)

| Arm | Makespan | Status | Gap | Solve time | Time to first feasible (from logs) | Branches |
|-----|----------|--------|-----|------------|-------------------------------------|----------|
| **Baseline** | **80** | FEASIBLE | 15.00% | 1800.4s | **~6.5s** | 25,222 |
| **Scratch-aware only** | **73** | FEASIBLE | 6.85% | 1800.3s | **~64.7s** | 198,334 |

### Interpretation

- **Makespan**: Scratch-aware tightening **improved** makespan (73 vs 80, −7 cycles, −8.75%) and **reduced** optimality gap (6.85% vs 15%). So for 3r/64b, scratch-aware tightening was beneficial for both solution quality and bound quality.
- **Time to first feasible**: Baseline found a first solution in ~6.5s; scratch-aware took ~64.7s. So scratch tightening delayed feasibility in this run but produced a better final solution.
- **Search effort**: Scratch-aware did many more branches (198k vs 25k) and ran the full 1800s; baseline also ran the full 1800s but with fewer branches. Neither proved optimality.

## Failures (slot tightening) — INFEASIBLE

- **Slot-only**: Solver status **INFEASIBLE** (proof of no solution). Slot-aware tightening likely over-tightened and excluded all feasible schedules.
- **Slot+scratch**: Solver status **INFEASIBLE**. Combining slot and scratch tightening over-constrains the model for this instance.

## Recommendations

1. **Slot-only and slot+scratch are INFEASIBLE** — a longer time limit will not help; the model is proven to have no solution. Fix would be to relax or correct slot (and optionally scratch) tightening so the domains remain feasible.
2. **Treat scratch-only as the best arm** among completed runs for 3r/64b: better makespan and gap than baseline.
3. **Investigate slot tightening**: Slot-aware tightening is over-constraining (proven INFEASIBLE). Compare ASAP/ALAP domains with baseline to see which bounds exclude feasible schedules.

## Log files (post-hoc)

- `ablation_log_baseline_tightening_disabled.txt`
- `ablation_log_slot-aware_tightening_enabled.txt` (failed)
- `ablation_log_scratch-aware_tightening_enabled.txt`
- `ablation_log_slot__and__scratch_tightening_enabled.txt` (failed)

Note: Stored `time_to_feasibility` in the results JSON is currently the solve time (1800s) when the parser does not detect the first-solution time; actual time to first feasible was taken from CP-SAT log lines (`#1 ... 6.47s` and `#1 ... 64.68s`).
