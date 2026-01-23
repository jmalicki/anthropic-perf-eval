# Problem Constraints & Theoretical Analysis

## Problem Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Tree Height | 10 | Levels below root (11 total levels) |
| Tree Nodes | 2047 | 2^(h+1) - 1 = 2^11 - 1 |
| Batch Size | 256 | Number of elements to process |
| Rounds | 16 | Number of tree traversal steps |
| VLEN | 8 | Elements per vector operation |
| N_CORES | 1 | Single core |

## Memory Layout

```
Header (7 words):
  [0] rounds = 16
  [1] n_nodes = 2047
  [2] batch_size = 256
  [3] forest_height = 10
  [4] forest_values_p → pointer to tree values
  [5] inp_indices_p → pointer to input indices
  [6] inp_values_p → pointer to input values
  [7] extra_room_p → workspace

Data:
  tree values: 2047 words
  input indices: 256 words
  input values: 256 words
  extra room: variable

Total memory: ~2600 words minimum
```

## Scratch Space

| Constraint | Value |
|------------|-------|
| SCRATCH_SIZE | 1536 words |
| Vector allocation | VLEN=8 words per vector |
| Effective vectors | 1536/8 = 192 vector registers |

### Scratch Budget for Blocked Kernel
- 32 idx vectors: 256 words
- 32 val vectors: 256 words
- Tree cache: 7 words (nodes 0-6)
- Constants: ~50 words
- Temps: ~40 words
- **Total used**: ~600 words
- **Remaining**: ~900 words (room for speculation!)

## Slot Limits (VLIW Constraints)

| Engine | Slots/Cycle | Notes |
|--------|-------------|-------|
| ALU | 12 | Scalar arithmetic |
| VALU | 6 | Vector arithmetic (8-wide) |
| Load | 2 | Memory loads |
| Store | 2 | Memory stores |
| Flow | 1 | Control flow |
| Debug | 64 | Debug only |

## Work Analysis

### Total Operations Per Round
- 256 elements × 1 tree lookup = 256 loads (gather)
- 256 elements × 1 XOR = 256 XORs (32 vector ops)
- 256 elements × 6 hash stages = 1536 hash ops (192 vector ops)
- 256 elements × 1 branch = 256 branches (32 vector ops)

### Per Round Minimum Cycles

**Without Optimizations (gather-based):**
- Tree gather: 256 loads / 2 loads/cycle = 128 cycles
- Hash: 192 vector ops / 6 ops/cycle = 32 cycles
- Branch: 32 ops / 6 ops/cycle = ~6 cycles
- Overhead: ~10 cycles
- **Per round**: ~176 cycles
- **16 rounds**: ~2816 cycles

**With Tree Preload (rounds 0-2):**
- Round 0: 1 broadcast instead of 256 loads → saves ~127 cycles
- Round 1: 2 broadcasts + select → saves ~120 cycles
- Round 2: 4 broadcasts + 3 selects → saves ~110 cycles
- **Savings**: ~357 cycles

**With Full Blocking:**
- Load all idx/val once: 64 vloads / 2 = 32 cycles
- Store all idx/val once: 64 vstores / 2 = 32 cycles
- Per-round savings: ~32 cycles (no stores between rounds)
- **Savings over 16 rounds**: 15 × 32 = ~480 cycles

## Theoretical Minimum Analysis

### Lower Bounds

1. **Memory Bandwidth Bound:**
   - Load 256 idx + 256 val = 512 loads
   - Store 256 idx + 256 val = 512 stores
   - Tree accesses: 256 × 16 = 4096 (if all cached: 0)
   - At 2 loads/cycle + 2 stores/cycle: **256 cycles minimum** (data I/O only)

2. **Compute Bound:**
   - Hash: 16 rounds × 256 elements × 6 stages × 2 ops = 49,152 scalar ops
   - At 6 VALU × 8 wide = 48 ops/cycle: **~1024 cycles minimum**

3. **Combined Realistic Minimum:**
   - Accounting for dependencies and overhead: **~1500-2000 cycles**

### Current Performance

| Kernel | Cycles | Speedup | Status |
|--------|--------|---------|--------|
| Baseline | 147,734 | 1.0x | Reference |
| Ultra (8x unroll) | 12,373 | 11.9x | Working |
| Combined | 12,256 | 12.0x | Working |
| Target | 1,487 | 99.3x | Goal |

### Gap Analysis

**Current best: 12,256 cycles**
**Target: 1,487 cycles**
**Gap: 10,769 cycles (8.2x improvement needed)**

## Optimization Opportunities

### 1. Full Blocking ✓ IMPLEMENTED
- Keep all 256 idx/val in scratch
- Process all 16 rounds without memory stores
- Result: 13.91x → 10,622 cycles

### 2. Tree Preload ✓ IMPLEMENTED
- Broadcasts for rounds 0-2 instead of gather
- Round 0: 1 broadcast (tree[0])
- Round 1: 2 broadcasts + select (tree[1]/[2])
- Round 2: 4 broadcasts + 3 selects (tree[3-6])
- Result: 14.90x → 9,917 cycles

### 3. Speculative Execution (TODO)
- Pre-compute both branches: tree[2*idx+1] and tree[2*idx+2]
- Overlap tree loads with computation
- Expected: ~1.5x improvement

### 4. Extended Tree Caching
- With 2047 tree nodes and ~800 spare scratch:
  - Can cache top ~7-8 levels (127-255 nodes)
  - Covers more early-round accesses
- Expected: ~1.2x improvement

### 5. Software Pipelining
- Load group N+1 while computing group N
- Better instruction-level parallelism
- Expected: ~1.2x improvement

## MILP Optimization Variables

The MILP solver could optimize these parameters:

### Block Size (UNROLL)
| UNROLL | Groups/Iter | VALU Util | Scratch Cost | Code Size |
|--------|-------------|-----------|--------------|-----------|
| 1 | 32 iters | 16% | 16 vectors | Small |
| 2 | 16 iters | 33% | 32 vectors | Small |
| 4 | 8 iters | 66% | 64 vectors | Medium |
| 8 | 4 iters | 100% | 128 vectors | Large |
| 16 | 2 iters | 100%* | 256 vectors | Very Large |

*Beyond 6-8, limited by VALU slots (max 6/cycle)

### Tree Cache Depth (MILP Variable)
| Depth | Nodes Cached | Scratch | Rounds Covered | Gather Rounds | Saves |
|-------|--------------|---------|----------------|---------------|-------|
| 3 | 7 | 7 words | 0-2 | 13 | baseline |
| 4 | 15 | 15 words | 0-3 | 12 | ~128 cycles |
| 5 | 31 | 31 words | 0-4 | 11 | ~256 cycles |
| 6 | 63 | 63 words | 0-5 | 10 | ~384 cycles |
| 7 | 127 | 127 words | 0-6 | 9 | ~512 cycles |

**Cost-Benefit Analysis (4 vs 5 levels):**
- 4 levels: 15 words scratch, saves 4 gather rounds (0-3)
- 5 levels: 31 words scratch (+16), saves 5 gather rounds (0-4)
- Trade-off: 16 extra scratch words for 1 fewer gather round (~128 cycles saved)
- Per gather round: 64 loads at 2/cycle = 32 cycles minimum (+ compute overlap)
- **Recommendation**: Let MILP optimize based on scratch pressure from other uses

**Round-by-Round Lookup Complexity:**
| Round | Possible Indices | Lookup Method (if cached) |
|-------|------------------|---------------------------|
| 0 | {0} | 1 broadcast |
| 1 | {1,2} | 2 broadcasts + 1 vselect |
| 2 | {3,4,5,6} | 4 broadcasts + 3 vselects |
| 3 | {7..14} | 8 broadcasts + 7 vselects |
| 4 | {15..30} | 16 broadcasts + 15 vselects |
| 5+ | {31+} | Too many selects, use gather |

**Key insight**: Beyond depth 4-5, vselect chains become expensive. MILP should weigh:
- Scratch cost: 2^depth words
- Gather savings: ~32 cycles/round
- vselect cost: ~(2^round) vselects per lookup

### MILP Decision Variables
```
block_size ∈ {1, 2, 4, 8, 16}
tree_cache_depth ∈ {3, 4, 5, 6, 7}
use_speculation ∈ {0, 1}
pipeline_depth ∈ {1, 2, 3}
```

### MILP Constraints
```
scratch_used = 2*batch_size + block_size*VLEN*3 + 2^tree_cache_depth + constants
scratch_used <= SCRATCH_SIZE

cycles = f(block_size, tree_cache_depth, speculation, pipeline)
```

### Combined Target
- Current best: 9,917 cycles (14.90x)
- Speculation: ÷1.5 → 6,611 cycles
- Extended caching: ÷1.2 → 5,509 cycles
- Pipelining: ÷1.2 → 4,591 cycles

**Realistic target: ~4,000-5,000 cycles**

To reach 1,487 cycles (100x), we likely need:
- Deeper algorithmic insights
- Perfect instruction scheduling via MILP
- Novel loop transformations

## Current Best Results

| Kernel | Cycles | Speedup | Key Optimizations |
|--------|--------|---------|-------------------|
| baseline | 147,734 | 1.0x | Sequential scalar |
| ultra_8x | 12,373 | 11.9x | 8-way parallel |
| combined | 12,256 | 12.0x | + tree preload |
| blocked_parallel | 10,622 | 13.9x | + full blocking |
| ultimate | 9,917 | 14.9x | All combined |
| **dual_path** | **9,898** | **14.9x** | Best current |

**Current gap**: 9,898 cycles actual vs 1,487 target = **6.7x to go**

## Key Findings

1. **Blocking saves 16× stores** but adds complexity
2. **Tree preload** saves ~30 cycles for rounds 0-2
3. **8-way parallelism** essential for VALU utilization
4. **Gather bottleneck**: 64 loads/round at 2/cycle = 32 cycles minimum
5. **No native vpermute**: Emulation with vselect is slower than gather
6. **Efficiency**: ~52% (estimate 5168, actual 9898) - room for improvement

## Remaining Optimization Opportunities

1. **Better instruction packing**: Fill all VLIW slots each cycle
2. **Pipeline across iterations**: Load iter N+1 while computing iter N
3. **Extended tree caching**: Cache more tree levels in scratch
4. **MILP-driven scheduling**: Let solver find optimal instruction order

## Hash Function Details

```python
HASH_STAGES = [
    ("+", 0x7ED55D16, "+", "<<", 12),
    ("^", 0xC761C23C, "^", ">>", 19),
    ("+", 0x165667B1, "+", "<<", 5),
    ("+", 0xD3A2646C, "^", "<<", 9),
    ("+", 0xFD7046C5, "+", "<<", 3),
    ("^", 0xB55A4F09, "^", ">>", 16),
]
```

Each stage: `val = op2(op1(val, const), op3(val, shift))`
- 6 stages × 3 ops = 18 vector ops per hash
- 256 elements ÷ 8 VLEN = 32 hashes per round
- 32 × 18 = 576 vector ops per round for hashing
