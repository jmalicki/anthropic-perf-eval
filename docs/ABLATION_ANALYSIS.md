# Ablation Study Analysis

**Date**: 2026-02-06  
**Configuration**: 3 rounds, batch size 64  
**Time limit**: 300s per run

## Overview

This document analyzes the ablation study results comparing ASAP/ALAP tightening enabled vs disabled configurations.

## Available Results

### Configuration Tested

- **Rounds**: 3
- **Batch size**: 64
- **Time limit**: 300 seconds
- **Total operations**: 1,136 ops

### Results Summary

| Configuration | Makespan | Peak Liveness | VALU Shifted | Solve Time | Status |
|--------------|----------|---------------|--------------|------------|--------|
| ASAP/ALAP tightening enabled | 86 cycles | 497 words | 74 ops | 301.0s | Completed (hit time limit) |
| Baseline (tightening disabled) | **Missing** | - | - | - | Not found |

**Note**: The baseline run for this configuration (3 rounds, batch size 64) was not found in the results. Only the tightening-enabled configuration completed.

## Detailed Analysis: Tightening Enabled

### Performance Metrics

- **Makespan**: 86 cycles
- **Lower bound**: 68 cycles (gap of 18 cycles, 26.5% above lower bound)
- **Peak liveness**: 497 words (cycle 16)
- **Average liveness**: 325.6 words
- **Budget exceeded**: False
- **VALU shifted**: 74 operations

### Solve Time

- **Total elapsed time**: 301.0 seconds (hit the 300s time limit)
- **Status**: Solver did not converge to optimal solution within time limit

### Liveness Analysis

The liveness curve shows:
- **Peak**: 497 words at cycle 16
- **Average**: 325.6 words across 87 cycles
- **Minimum**: 0 words (some cycles have no live values)

The peak liveness occurs relatively early in the schedule (cycle 16 out of 86), suggesting that the initial rounds create significant register pressure.

### Usage Accounting

#### By Round

| Round | VRegs | Total Words | Scalar | Vector |
|-------|-------|-------------|--------|--------|
| -1 (setup) | 134 | 526 | 78 | 56 |
| 0 | 136 | 1,088 | 0 | 136 |
| 1 | 152 | 1,216 | 0 | 152 |
| 2 | 184 | 1,472 | 0 | 184 |

**Observations**:
- Round -1 (setup) has a mix of scalar and vector registers
- Rounds 0-2 are purely vector operations (no scalar vregs)
- Each round increases register usage, with round 2 using the most

#### By Variable Type

| Type | VRegs | Words | Notes |
|------|-------|-------|-------|
| temporary | 421 | 3,333 | Largest category (69% of vregs) |
| address | 39 | 39 | Scalar addresses |
| batch_data | 32 | 256 | Vector batch data |
| condition | 32 | 256 | Vector conditions |
| round_state | 24 | 192 | Per-round state |
| hash_state | 22 | 99 | Hash computation state |
| tree_cache | 19 | 75 | Tree traversal cache |
| constant | 14 | 49 | Materialized constants |
| pointer | 3 | 3 | Tree pointers |

**Key Insights**:
- **Temporary variables dominate**: 421 vregs (69% of total) accounting for 3,333 words (77% of total words)
- This suggests significant opportunities for register allocation optimization
- Address variables are all scalar (39 words for 39 vregs)
- Batch data and conditions are purely vector (8 words per vreg)

### VALU Shift Analysis

- **Total VALU ops shifted to ALU**: 74 operations
- This represents significant shifting activity, suggesting VALU slots were a bottleneck

## Comparison with Other Configurations

### Baseline from Different Configuration

A baseline run exists for a **different configuration** (2 rounds, batch size 32):
- **Makespan**: 37 cycles
- **Peak liveness**: 250 words
- **VALU shifted**: 24 ops
- **Total ops**: 394 ops

**Note**: This is not directly comparable due to different problem size, but provides context:
- The 3-round, 64-batch configuration is ~2.9x larger (1,136 vs 394 ops)
- Makespan scales roughly proportionally (86 vs 37 cycles = 2.3x)
- Peak liveness scales less (497 vs 250 words = 2.0x), suggesting better register reuse at larger scale

## Key Findings

### 1. Time Limit Hit

The solver hit the 300-second time limit without finding an optimal solution:
- Makespan: 86 cycles
- Lower bound: 68 cycles
- Gap: 18 cycles (26.5% above lower bound)

This suggests the problem is challenging for CP-SAT at this scale, even with tightening enabled.

### 2. Register Pressure

- Peak liveness of 497 words is well below the 1,536-word budget (32% utilization)
- However, the average liveness of 325.6 words suggests consistent pressure
- Temporary variables dominate usage (77% of total words)

### 3. VALU Shift Activity

- 74 VALU operations were shifted to ALU
- This indicates VALU slots were a bottleneck that the solver addressed via shifting

### 4. Missing Baseline

The baseline run (tightening disabled) for this configuration was not found. To complete the ablation study:
- Run baseline with same configuration (3 rounds, batch size 64)
- Compare makespan, solve time, and solver statistics
- Measure impact of tightening on search space reduction

## Recommendations

### 1. Complete the Ablation Study

Run the baseline (tightening disabled) configuration:
```bash
uv run scripts/run_asap_alap_ablation.py --rounds 3 --batch-size 64 --time-limit 300
```

This will enable direct comparison of:
- Makespan improvement/regression
- Solve time impact
- Search space reduction (branches, conflicts)

### 2. Analyze Solve Time vs Problem Size

The current run hit the time limit. Consider:
- Increasing time limit for larger problems
- Measuring time to first feasible solution
- Comparing tightening impact on search efficiency

### 3. Investigate Register Allocation

The dominance of temporary variables (77% of words) suggests:
- Opportunities for better register reuse
- Potential for dead code elimination improvements
- Analysis of register lifetime patterns

### 4. VALU Shift Analysis

With 74 shifted operations:
- Analyze which operation types were shifted
- Compare VALU/ALU slot utilization
- Evaluate if shifting improved makespan

## Next Steps

1. **Run baseline**: Complete the ablation study with baseline configuration
2. **Extend time limit**: For larger problems, consider longer time limits (600s+) to allow convergence
3. **Measure propagation**: Add metrics for how much tightening actually tightened bounds
4. **Compare solver stats**: When baseline is available, compare branches/conflicts/restarts
5. **Scale up**: Test on 4-round, batch size 128 to see if improvements scale
