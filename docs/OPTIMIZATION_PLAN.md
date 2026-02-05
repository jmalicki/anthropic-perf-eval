# Optimization Plan: Current State and Path to 100x

## Current State

| Metric | Value |
|--------|-------|
| Baseline | 147,734 cycles |
| **Current best** | **2,390 cycles** |
| **Current speedup** | **61.8x** |
| Target | 1,487 cycles (100x) |
| Gap | ~900 cycles to go |

## Implemented Optimizations

### 1. Full Unrolling ✅

All 16 rounds × 32 groups are emitted as explicit instructions (no loops).

**Benefits:**
- Eliminates loop overhead (branch, counter update)
- Enables cross-iteration scheduling
- Allows CP-SAT to find optimal global packing

**Implementation:** `interleaved_unrolled_kernel.py`

### 2. Interleaved Emission ✅

Operations from different iterations are emitted interleaved, not sequentially.

**Before (sequential):**
```
iter0_op1, iter0_op2, iter0_op3, iter1_op1, iter1_op2, ...
```

**After (interleaved):**
```
iter0_op1, iter1_op1, iter2_op1, iter0_op2, iter1_op2, ...
```

**Benefits:**
- Independent operations become adjacent
- Scheduler can easily find parallelism
- Better VLIW slot utilization

### 3. Tree Value Caching (Rounds 0-2) ✅

Early rounds access predictable tree indices:
- Round 0: Always index 0 → broadcast tree[0]
- Round 1: Indices 1 or 2 → broadcast + select
- Round 2: Indices 3-6 → broadcast + multiple selects

**Benefits:**
- Eliminates gather loads for first 3 rounds
- Saves ~128 cycles per cached round

### 4. CP-SAT Scheduling ✅

Three-phase scheduling pipeline:

1. **Chunked solve**: Divide into ~1500-op chunks, solve each
2. **Full solve**: Use chunked as warm-start for global optimization
3. **VALU→ALU shifting**: Move ops when ALU has earlier free slots

**Benefits:**
- Near-optimal VLIW packing
- Exploits all available parallelism
- Handles ~10,000 ops in minutes

### 5. Fused multiply_add ✅

Branch computation uses fused multiply-add:
```python
# Old (2 VALU ops):
idx = idx * 2
idx = idx + branch_offset

# New (1 VALU op):
idx = multiply_add(idx, 2, branch_offset)  # idx*2 + offset
```

**Benefits:**
- Saves 1 VALU op per group per round
- 32 groups × 16 rounds = 512 ops saved

### 6. Bounds Check Elimination ✅

Skip bounds checking for early rounds where overflow is impossible:
```python
max_idx_this_round = (1 << (rnd + 2)) - 1
needs_bounds_check = max_idx_this_round >= n_nodes  # 2047
```

For rounds 0-9 (max_idx < 2047), no bounds check needed.

**Benefits:**
- Saves 2 VALU ops per group for ~10 rounds
- ~640 ops eliminated

### 7. Precomputed Kernel ✅

Kernel is built offline and stored compressed:
- Zlib compression: 119KB → 22KB (82% reduction)
- Base64 encoding for embedding in Python
- Runtime decompression is instant

**Benefits:**
- No scheduling overhead at runtime
- Reproducible results

## Current Bottleneck Analysis

With 2,390 cycles for 32 groups × 16 rounds:
- **~4.7 cycles per group per round** (average)

Breakdown per group per round:
| Operation | Cycles | Notes |
|-----------|--------|-------|
| Gather (rounds 3-15) | ~4 | 8 loads at 2/cycle |
| XOR | 0.17 | Packs with others |
| Hash (6 stages) | ~3 | 18 VALU ops at 6/cycle |
| Branch | ~0.5 | 3 VALU ops |
| **Total** | **~7.7** | Theoretical per round |

Rounds 0-2 are cheaper (~2 cycles each, no gather).

**Primary bottleneck: Gather operations** (scalar loads for tree access)

## Remaining Optimization Opportunities

### 1. Deeper Tree Caching

Current: Cache tree[0-6] (7 nodes), cover rounds 0-2
Potential: Cache tree[0-126] (127 nodes), cover rounds 0-6

| Depth | Nodes | Scratch | Rounds Covered | Expected Savings |
|-------|-------|---------|----------------|------------------|
| 3 | 7 | 7 | 0-2 | baseline |
| 4 | 15 | 15 | 0-3 | ~30 cycles |
| 5 | 31 | 31 | 0-4 | ~60 cycles |
| 6 | 63 | 63 | 0-5 | ~90 cycles |
| 7 | 127 | 127 | 0-6 | ~120 cycles |

**Challenge:** Selection logic grows exponentially (2^round vselects needed)

### 2. Speculative Dual-Path Execution

Load BOTH possible tree values before knowing which is needed:
```python
tree_left = tree[2*idx + 1]   # Load both
tree_right = tree[2*idx + 2]
val_left = hash(val ^ tree_left)    # Compute both
val_right = hash(val ^ tree_right)
val = select(condition, val_left, val_right)  # Choose
```

**Benefits:**
- Hides load latency behind computation
- Uses idle VALU slots (computing "wrong" path)
- Load slots used more efficiently (parallel loads)

**Cost:**
- 2× hash computation (but overlapped)
- Extra scratch for second path

### 3. Software Pipelining

Overlap operations across rounds:
- While hashing round R, start gathering for round R+1
- Requires careful scratch management

### 4. Better VALU Packing

Current VALU utilization: ~87% (5.2/6 slots average)

Opportunities:
- Reorder operations to fill gaps
- Use ALU for some ops to relieve VALU pressure
- Already partially addressed by VALU→ALU shifting

## Path to 1,487 Cycles (100x)

| Optimization | Expected Cycles | Delta |
|--------------|-----------------|-------|
| Current | 2,390 | - |
| Extended tree cache (7 levels) | ~2,270 | -120 |
| Speculative dual-path | ~1,900 | -370 |
| Software pipelining | ~1,600 | -300 |
| **Combined** | **~1,600** | Still ~100 cycles short |

**Conclusion:** Reaching 1,487 cycles likely requires:
- Novel algorithmic restructuring
- Custom instruction sequences not yet discovered
- Near-perfect scheduling with no wasted slots

## Files to Modify

| File | Purpose |
|------|---------|
| `interleaved_unrolled_kernel.py` | Kernel generation |
| `round_chunked_scheduler.py` | Multi-phase scheduling |
| `iterative_reschedule.py` | VALU→ALU shifting |
| `generate_ssa_kernel.py` | Kernel generation (SSA + CP-SAT) |

## Running Experiments

```bash
# Generate new kernel with different parameters
uv run generate_ssa_kernel.py --cpsat-time-limit 7200

# Test the kernel
uv run test_optimized.py

# Run full experiment suite with W&B logging
python run_experiment.py
```
