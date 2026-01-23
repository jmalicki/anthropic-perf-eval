# Speculation Strategies for VLIW Tree Traversal

## Problem Structure

The kernel performs tree traversal with hash computation:
```
for each round:
    val = hash(val ^ tree[idx])
    idx = 2*idx + (1 if val%2==0 else 2)
    if idx >= n_nodes: idx = 0
```

This creates a sequential dependency chain:
- `val₁` depends on `tree[idx₀]`
- `idx₁` depends on `val₁ % 2`
- `val₂` depends on `tree[idx₁]`
- ...

## VLIW Resource Availability

| Engine | Slots/Cycle | Typical Usage | Free Slots |
|--------|-------------|---------------|------------|
| ALU | 12 | ~4 for address computation | ~8 |
| VALU | 6 | ~5.2 (87% utilization) | ~0.8 |
| Load | 2 | 2 for gather | 0 |
| Store | 2 | 0 during compute | 2 |
| Flow | 1 | 0-1 for select | 0-1 |

**Key insight**: ALU is heavily underutilized. Speculation that uses ALU is "free".

## Implemented Strategies

### Strategy 1: Tree Preload (Rounds 0-2) ✅

**Status: Implemented in `interleaved_unrolled_kernel.py`**

**Idea**: Early rounds access predictable tree indices. Cache these values and use broadcast + select.

| Round | Output Range | Strategy |
|-------|--------------|----------|
| 0 | {0} | Broadcast tree[0] |
| 1 | {1, 2} | Broadcast tree[1], tree[2] + select |
| 2 | {3..6} | Broadcast tree[3-6] + 3 selects |

**Benefits:**
- Eliminates 256 gather loads per round (3 rounds = 768 loads saved)
- Saves ~384 cycles (768 loads at 2/cycle)

**Cost:**
- Extra VALU ops for selection (~30 per round)
- 7 words scratch for cached values

### Strategy 2: Bounds Check Elimination ✅

**Status: Implemented in `interleaved_unrolled_kernel.py`**

**Idea**: Skip bounds checking when index overflow is impossible.

```python
max_idx_this_round = (1 << (rnd + 2)) - 1
needs_bounds_check = max_idx_this_round >= n_nodes  # 2047
```

For rounds 0-9: max_idx ∈ {3, 7, 15, 31, 63, 127, 255, 511, 1023} < 2047

**Benefits:**
- Saves 2 VALU ops per group per round (compare + multiply)
- 10 rounds × 32 groups × 2 ops = 640 ops eliminated

## Proposed Strategies (Not Yet Implemented)

### Strategy 3: Dual Path Execution ⏳

**Status: Not implemented**

**Idea**: Since next_idx is EITHER 2*idx+1 OR 2*idx+2, compute both paths speculatively.

```python
# Instead of waiting for val%2 to decide which tree node to load:
tree_left = tree[2*idx + 1]
tree_right = tree[2*idx + 2]
val_left = hash(val ^ tree_left)
val_right = hash(val ^ tree_right)

# Then select based on original val:
new_val = (orig_val%2 == 0) ? val_left : val_right
new_idx = (orig_val%2 == 0) ? 2*idx+1 : 2*idx+2
```

**Benefits:**
- Loads BOTH paths in parallel (uses both load slots productively)
- Hides memory latency behind 2× hash computation
- Uses idle ALU slots for address computation

**Cost:**
- 2× hash computation per round
- Extra scratch for second path temporaries
- Extra selection ops

**Analysis:**
- Current per-round: ~150 cycles (gather-bound)
- With speculation: ~100 cycles (parallel loads + overlapped hash)
- Expected savings: ~650 cycles over 13 gather rounds

### Strategy 4: Extended Tree Caching ⏳

**Status: Not implemented**

**Idea**: Cache more tree levels to cover more rounds without gather.

| Depth | Nodes Cached | Scratch | Rounds Covered | Selection Cost |
|-------|--------------|---------|----------------|----------------|
| 3 | 7 | 7 | 0-2 | baseline ✅ |
| 4 | 15 | 15 | 0-3 | 7 vselects |
| 5 | 31 | 31 | 0-4 | 15 vselects |
| 6 | 63 | 63 | 0-5 | 31 vselects |
| 7 | 127 | 127 | 0-6 | 63 vselects |

**Trade-off**: Beyond depth 4-5, vselect chains become expensive (O(2^round)).

**Estimated benefit**: ~30 cycles per additional cached round.

### Strategy 5: Software Pipelining ⏳

**Status: Not implemented**

**Idea**: Overlap operations across consecutive rounds.

```
Round R:
  - Hash val_R
  - Compute idx_R

Overlapped with Round R+1:
  - Start gather for tree[idx_R] (idx_R now known)
```

**Benefits:**
- Hides gather latency behind hash computation
- Better overall throughput

**Challenges:**
- Complex scratch management (need space for 2 rounds of data)
- Dependency tracking across rounds

### Strategy 6: Branch Prediction ⏳

**Status: Not implemented (unlikely to help)**

**Idea**: Predict branch direction and speculatively load only one path.

**Analysis:**
- For cryptographic hashes, val%2 is uniformly random (50/50)
- Static prediction: 50% misprediction rate
- Pattern prediction: Hash destroys patterns

**Conclusion**: Not viable for this hash function. Dual-path execution (Strategy 3) is better.

## Expected Performance Impact

| Strategy | Current Status | Expected Savings |
|----------|---------------|------------------|
| Tree preload (0-2) | ✅ Implemented | baseline |
| Bounds check elimination | ✅ Implemented | ~100 cycles |
| Dual path execution | ⏳ Proposed | ~500 cycles |
| Extended tree cache | ⏳ Proposed | ~120 cycles |
| Software pipelining | ⏳ Proposed | ~200 cycles |

**Total potential**: ~2,390 - 920 = **~1,470 cycles** (approaching 100x target!)

## Implementation Priority

1. ✅ **Tree preload (rounds 0-2)** - Done
2. ✅ **Bounds check elimination** - Done
3. ⏳ **Dual path execution** - Highest potential, should implement next
4. ⏳ **Extended tree cache (rounds 3-4)** - Moderate benefit
5. ⏳ **Software pipelining** - Complex, implement last

## Code Locations

| Strategy | File | Function/Section |
|----------|------|------------------|
| Tree preload | `interleaved_unrolled_kernel.py` | `build_kernel()` rounds 0-2 |
| Bounds check | `interleaved_unrolled_kernel.py` | `build_kernel()` branch phase |
| CP-SAT scheduling | `cp_scheduler.py` | `schedule_cp()` |
| VALU→ALU shifting | `iterative_reschedule.py` | `iterative_reschedule()` |
