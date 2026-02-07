# ASAP/ALAP Tightening Ablation Study Results

**Date**: 2026-02-06  
**Configuration**: 2 rounds, batch size 32  
**Time limit**: 300s per run

## Summary

| Configuration | Makespan | Solve Time | Notes |
|--------------|----------|------------|-------|
| Baseline (tightening disabled) | 37 cycles | 1.1s | Default configuration |
| ASAP/ALAP tightening enabled | 37 cycles | 0.7s | 40.8% faster solve time |

## Key Findings

1. **Makespan unchanged**: Both configurations achieved the same makespan (37 cycles)
2. **Solve time improved**: Tightening reduced solve time by 0.5s (40.8% faster)
3. **Small problem size**: With only 2 rounds and batch size 32, this is a very small problem (~394 ops vs ~11,776 for full kernel). Results may not be representative of larger problems.

## Analysis

### Why Makespan Didn't Change

For this small problem size:
- The problem is already easy to solve optimally
- Capacity constraints may not be the bottleneck
- The greedy hint may already be near-optimal

### Why Solve Time Improved

Even though makespan didn't change, tightening likely:
- Reduced search space by tightening variable domains
- Helped CP-SAT find the optimal solution faster
- Improved propagation strength, allowing earlier pruning

### Limitations

- **Small problem size**: 2 rounds × 32 batch size = ~394 ops vs full kernel's ~11,776 ops
- **Fast solve**: Both runs completed in <2s, so the improvement may not be significant
- **May not scale**: Results on small problems don't necessarily predict behavior on large problems

## Recommendations

1. **Test on larger problems**: Run with 4 rounds, batch size 64 (or larger) to see if improvements scale
2. **Measure propagation strength**: Check if tightening actually tightened bounds (look for verbose output)
3. **Time to feasibility**: For larger problems, measure time to first feasible solution (`best:inf` → `best:<value>`)
4. **Domain reduction**: Measure average domain size reduction from tightening

## Next Steps

- Run ablation study with 4 rounds, batch size 64
- Measure time to first feasible solution for larger problems
- Check verbose output for tightening statistics (how many ops had bounds tightened)
- Test on full kernel (16 rounds, batch size 256) if smaller tests show promise
