# Ablation: Symmetry-Aware vs Energetic Tightening (3 rounds, batch 32, 600s)

**Date:** 2026-02-17  
**Configuration:** 3 rounds, batch size 32, CP-SAT time limit 600s

## Summary

Both tightening modes reached the same optimal makespan (56 cycles), but **energetic tightening converged faster**:

- **Energetic:** 9.63s to optimal (12.2% faster)
- **Symmetry-aware:** 10.97s to optimal

Energetic also required fewer branches (38,335 vs 43,082), indicating tighter domains helped prune the search space more effectively.

## Results

| Metric | Symmetry-Aware | Energetic | Difference |
|--------|----------------|-----------|------------|
| **Makespan** | 56 cycles | 56 cycles | Same |
| **Solve time** | 10.97s | 9.63s | **-1.34s (-12.2%)** |
| **Elapsed time** | 11.6s | 9.8s | -1.8s (-15.2%) |
| **Solver status** | OPTIMAL | OPTIMAL | Both optimal |
| **Branches** | 43,082 | 38,335 | **-4,747 (-11.0%)** |
| **Peak liveness** | 353 words | 353 words | Same |
| **VALU shifted** | 13 ops | 12 ops | -1 op |
| **Optimality gap** | 0.0000 | 0.0000 | Both proven optimal |

## Convergence Milestones

### Objective Improvement Timeline

**Symmetry-aware:**
- First feasible: 0.41s
- Solution #2: 2.38s (obj: 74010757)
- Solution #3: 2.40s (obj: 74010720)
- Solution #4: 2.49s (obj: 66008798, ~44% improvement)
- Solution #5 (optimal): 2.70s (obj: 56007887, 100% improvement)
- **Time to prove optimality:** 10.97s total

**Energetic:**
- First feasible: 0.34s (faster initial feasibility)
- Solution #2: 2.42s (obj: 74010789)
- Solution #3: 2.43s (obj: 74010683)
- Solution #4: 2.53s (obj: 73010677, ~6% improvement)
- Solution #5: 2.58s (obj: 66008798, ~44% improvement)
- Solution #6 (optimal): 2.82s (obj: 56007807, 100% improvement)
- **Time to prove optimality:** 9.63s total

### Key Observations

1. **Energetic found first feasible solution faster** (0.34s vs 0.41s)
2. **Energetic found more intermediate solutions** (6 vs 5), suggesting better search guidance
3. **Energetic proved optimality faster** despite finding the optimal solution slightly later (2.82s vs 2.70s), indicating tighter bounds helped prove optimality more efficiently
4. **Energetic required fewer branches** (38,335 vs 43,082), confirming tighter domains reduced search space

## Time to Reach Improvement Milestones

| Improvement % | Symmetry-Aware | Energetic | Speedup |
|---------------|----------------|-----------|---------|
| 25% | 2.49s | 2.58s | 0.97x |
| 50% | 2.70s | 2.82s | 0.96x |
| 75% | 2.70s | 2.82s | 0.96x |
| 90% | 2.70s | 2.82s | 0.96x |
| 100% (optimal) | 2.70s | 2.82s | 0.96x |

**Note:** These times reflect when the optimal solution was *found*, not when optimality was *proven*. The total solve time (including proof) favors energetic.

## Conclusion

**Energetic tightening provides faster convergence** for this configuration:
- 12.2% faster solve time (9.63s vs 10.97s)
- 11.0% fewer branches (38,335 vs 43,082)
- Same optimal makespan (56 cycles)

The energetic tightening's tighter domains help CP-SAT:
1. Find feasible solutions faster (0.34s vs 0.41s)
2. Explore the search space more efficiently (fewer branches)
3. Prove optimality faster (despite finding the optimal solution slightly later)

This confirms the expected behavior: **energetic tightening converges faster in CP-SAT optimization** by providing tighter bounds that reduce the search space.

## Files

- Results JSON: `ablation_symmetry_vs_energetic_3r_32b_600s.json`
- Solver logs:
  - `ablation_log_symmetry-aware.txt`
  - `ablation_log_energetic.txt`
- Scheduler outputs:
  - `scheduler_output_symmetry-aware.json`
  - `scheduler_output_energetic.json`
