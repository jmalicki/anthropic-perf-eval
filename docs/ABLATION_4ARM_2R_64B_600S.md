# 4-Arm Ablation Study: 2 Rounds, Batch 64, 600s Time Limit

**Date**: 2026-02-06  
**Configuration**: 2 rounds, batch size 64, 600s CP-SAT time limit per run

## Summary

| Arm | Makespan | Status | Gap | Solve time | Notes |
|-----|----------|--------|-----|------------|--------|
| **Baseline** (no tightening) | **50** | FEASIBLE | 14.00% | 600.2s | Best makespan |
| **Slot-aware only** | **50** | FEASIBLE | 14.00% | 600.2s | Tied with baseline |
| **Scratch-aware only** | 51 | FEASIBLE | 15.69% | 600.2s | +1 cycle vs baseline |
| **Slot + scratch** | 52 | FEASIBLE | 17.31% | 600.2s | +2 cycles vs baseline |

All four runs hit the 600s time limit; none proved optimality (all FEASIBLE).  
All runs started feasible from greedy hints (`started_feasible: True`).

## Performance characteristics (optimization speed)

Parsed from CP-SAT solver logs (`#1 …s best:…` = first feasible solution; summary stats at end of run):

| Arm | Time to first feasible | Branches (600s) | Conflicts | Restarts |
|-----|------------------------|-----------------|-----------|----------|
| **Baseline** | **2.23s** | 18,435 | 457 | 8 |
| **Slot-aware only** | 3.76s | 64,364 | 510 | 6 |
| **Scratch-aware only** | 3.11s | 57,540 | 597 | 5 |
| **Slot + scratch** | **1.30s** | **13,769** | **104** | **2** |

### Interpretation

- **Time to first feasible**: Slot+scratch was **fastest** (1.30s); baseline was 2.23s; slot-only was slowest (3.76s). So scratch tightening (with or without slot) can make the solver find *some* feasible solution sooner.
- **Search effort**: Slot+scratch did **far fewer** branches and conflicts (13.7k branches, 104 conflicts) than baseline (18.4k, 457) or slot-only (64.4k, 510). Tighter domains reduced the search space.
- **Regression vs speed**: Slot+scratch found feasibility quickly and explored less, but **ended at worse makespan (52)**. So in this run, the tighter model made optimization *faster* in terms of search effort and time-to-feasibility, but the pruned space did not contain the 50-cycle solution; we correctly caught a **quality regression** (worse makespan), not a pure slowdown.
- **Slot-only**: Slightly slower to first feasible (3.76s) and more branches (64k) than baseline, but **same final makespan (50)** — no regression, and no clear win on speed in this 600s window.

So we care about both: (1) **quality** (makespan), where we caught a regression from scratch tightening, and (2) **optimization speed** (time-to-feasibility, branches), where tightening can make the solver faster but at the cost of solution quality when it over-tightens.

## Findings

### Makespan

- **Baseline and slot-only tightening both reached 50 cycles** — slot-aware tightening did not hurt and did not improve makespan in this run.
- **Scratch-only and slot+scratch produced worse makespans** (51 and 52 cycles). Possible causes:
  - Scratch-aware tightening may be over-tightening (excluding feasible schedules), or
  - Different search paths under the time limit led to different local optima.

### Optimality gap

- Gap increases as we add scratch tightening: 14% → 15.69% → 17.31%.
- With slot+scratch, the solver was furthest from the lower bound when time ran out, so we cannot conclude whether the true optimum is better or worse than 50.

### Time to feasibility

- See the **Performance characteristics** table above for time-to-first-feasible from solver logs.
- Feasibility is found early (1.3–3.8s); most of the 600s is spent improving the objective and closing the gap.

### Solver logs

- Full CP-SAT logs are on disk for post-hoc analysis:
  - `ablation_log_baseline_tightening_disabled.txt`
  - `ablation_log_slot-aware_tightening_enabled.txt`
  - `ablation_log_scratch-aware_tightening_enabled.txt`
  - `ablation_log_slot__and__scratch_tightening_enabled.txt`

## Why might tighter bounds lead to more branches?

### 1) Same number of constraints

Yes. Tightening only changes the **domains** of the cycle variables, not the model structure. We still create one integer variable per op with domain `[lb, ub]`; we just set `lb = final_asap[op]` and `ub = final_alap[op]` instead of the original ASAP/ALAP. Precedence, cumulative, scratch, and optional-interval constraints are unchanged. So: same variables, same constraints; only the initial domain bounds are tighter.

### 2) Hypothesis: less symmetry

With **loose** bounds (baseline), many ops have the same or heavily overlapping windows (e.g. many VALU ops with ASAP=1, ALAP=400). Those ops are **symmetric** from the solver’s perspective: swapping which op gets cycle 5 and which gets cycle 6 doesn’t change the model. CP-SAT can use symmetry-breaking to prune whole families of equivalent branches, so it can do **fewer** branches (e.g. 18k in baseline).

With **tighter** bounds (slot-aware), we’ve differentiated ops: one op might get [5, 10], another [6, 12], another [7, 11]. They’re no longer interchangeable, so there’s **less symmetry**. The solver can’t prune as much by “one representative per equivalence class,” so it may need to explore **more** branches (e.g. 64k for slot-only) even though the feasible set is smaller. So: tighter bounds → smaller search space in principle, but less symmetry → more branches in practice.

Other factors that could play a role: the LP relaxation and value-ordering heuristics can behave differently when domains are tight (e.g. different first solution, different branch order), and propagation strength can change (tighter domains can sometimes propagate more, sometimes lead to more “almost feasible” nodes and thus more backtracking). The symmetry story is a plausible main effect and is consistent with slot-only doing more branches than baseline while still finding the same makespan.

### Ordering constraints and tightening

We add symmetry constraints **cycle[i] <= cycle[j]** for consecutive ops in the same `ordering_group`. For that to be satisfiable we need **ASAP[i] <= ALAP[j]** (so some assignment exists). Rather than a separate “consistency pass,” **tightening respects ordering directly**: we derive the chain order from the bounds: after final ASAP/ALAP we sort each group by ALAP ascending and add cycle[i] <= cycle[j] for consecutive pairs, so ASAP[i] <= ALAP[j] holds by construction. Order within each group is now derived from the bounds (sort by ALAP after tightening); tightening no longer takes ordering as input. Tightening runs before the greedy pass so greedy hints stay inside CP-SAT domains.

## Conclusions

1. **Slot-aware tightening**: In this 2r/64b/600s setup, slot-only tightening matched baseline (50 cycles) and did not degrade performance.
2. **Scratch-aware tightening**: Associated with higher makespans (51, 52) and larger gaps. Worth checking for over-tightening (e.g. correctness of scratch-based bound tightening) or re-running with different seeds/time limits.
3. **No run proved optimal**: All runs are FEASIBLE with 14–17% gap; longer time limits or different parameters might change ranking.
4. **Recommendation**: Treat slot-aware tightening as safe to enable for this configuration; investigate scratch-aware tightening (correctness and impact on search) before enabling it by default.
