# Problem Constraints & Theoretical Analysis

## Problem Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Tree Height | 10 | Levels below root (11 total levels) |
| Tree Nodes | 2047 | 2^(h+1) - 1 = 2^11 - 1 |
| Batch Size | 256 | Number of elements to process; **must be a multiple of VLEN (8)** |
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

**Hard Constraint**: The scratch budget is a strict limit that cannot be exceeded. Scheduling with this constraint is equivalent to the "black pebbling game" on DAGs, which is **PSPACE-complete** (Gilbert, Lengauer, Tarjan 1980). Our greedy scheduler uses heuristics to stay within budget, with CP-SAT fallback when needed. See `docs/SCRATCH_PRESSURE.md` for theoretical details.

### Current Scratch Usage

From `interleaved_unrolled_kernel.py`:

| Allocation | Size (words) |
|------------|--------------|
| Scalar temps (st0-st15) | 16 |
| Constants (c0-c15) | 16 |
| Hash constants/shifts | 12 |
| Tree cache (tc0-tc6) | 7 |
| All indices (32 vectors) | 256 |
| All values (32 vectors) | 256 |
| Node values (4×8 vectors) | 256 |
| Per-group temps (4×8 vectors) | 256 |
| Broadcast vectors | ~100 |
| **Total** | **~1,323** |
| **Remaining** | **~213** |

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

| Operation Type | Count | Engine |
|---------------|-------|--------|
| Tree lookups | 256 loads (gather) | Load |
| XOR | 32 vector ops | VALU |
| Hash (6 stages) | 192 vector ops | VALU |
| Branch computation | ~96 vector ops | VALU |

### Operations by Round Type

**Rounds 0-2 (cached tree values):**
- Broadcast/select: ~10-30 VALU ops per round
- XOR: 32 VALU ops
- Hash: 192 VALU ops
- Branch: ~96 VALU ops
- No gather loads

**Rounds 3-15 (gather required):**
- Gather: 256 scalar loads (32 groups × 8 elements)
- Address computation: 256 ALU ops
- XOR: 32 VALU ops
- Hash: 192 VALU ops
- Branch: ~96 VALU ops

### Total Operation Count

| Category | Ops | Notes |
|----------|-----|-------|
| Gather loads (rounds 3-15) | 3,328 | 13 rounds × 256 |
| Address computation | 3,328 | 13 rounds × 256 |
| XOR | 512 | 16 rounds × 32 |
| Hash | 3,072 | 16 rounds × 192 |
| Branch | ~1,536 | 16 rounds × ~96 |
| **Total** | **~11,776** | Estimated |

## Theoretical Minimum Analysis

### Lower Bounds

1. **Memory Bandwidth Bound:**
   - Initial load: 512 words (idx + val)
   - Final store: 512 words (idx + val)
   - Gather: 256 × 13 = 3,328 loads (rounds 3-15)
   - At 2 loads/cycle: ~1,920 cycles for loads alone

2. **Compute Bound (VALU):**
   - Total VALU ops: ~5,120 (hash + XOR + branch)
   - At 6 VALU/cycle: ~853 cycles

3. **Combined Realistic Minimum:**
   - VALU and Load can overlap
   - Dependencies prevent perfect overlap
   - **Estimated minimum: ~1,500-2,000 cycles**

### Current Performance

| Kernel | Cycles | Speedup | Status |
|--------|--------|---------|--------|
| Baseline | 147,734 | 1.0x | Reference |
| **Current (precomputed)** | **2,390** | **61.8x** | Production |
| Target | 1,487 | 99.3x | Goal |

### Efficiency Analysis

| Metric | Value |
|--------|-------|
| Theoretical minimum | ~1,500 cycles |
| Current best | 2,390 cycles |
| Efficiency | ~63% |
| Gap to target | ~900 cycles |

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
- 256 elements ÷ 8 VLEN = 32 hashes per group
- Total per round: 32 groups × 2 VALU = 64 VALU ops (with packing)

## Key Constraints Summary

1. **VALU is the bottleneck**: 6 slots/cycle limits hash throughput
2. **Gather dominates later rounds**: 4 cycles minimum per group (8 loads at 2/cycle)
3. **Scratch is sufficient**: ~213 words remaining for additional optimizations
4. **Dependencies limit parallelism**: Hash depends on tree value, index depends on hash

## Optimization Levers

| Lever | Effect | Implemented |
|-------|--------|-------------|
| Unrolling | Exposes parallelism | ✅ Yes |
| Interleaving | Better scheduling | ✅ Yes |
| Tree caching | Eliminates gathers | ✅ Partial (rounds 0-2) |
| CP-SAT scheduling | Optimal packing | ✅ Yes |
| VALU→ALU shifting | Relieves VALU pressure | ✅ Yes |
| multiply_add fusion | Saves VALU ops | ✅ Yes |
| Bounds check elimination | Saves VALU ops | ✅ Yes |
| Speculation | Hides latency | ❌ Not yet |
| Software pipelining | Overlaps rounds | ❌ Not yet |
