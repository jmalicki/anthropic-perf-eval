# Lazy Partition Optimization - Detailed Design

## Executive Summary

Replace 8 scattered loads per group (depths 4+) with vload + selection.

---

## 🎯 Implementation Strategy: SIMPLE FIRST, MEASURE, ITERATE

**Do NOT over-engineer. Implement incrementally, measure at each step.**

| Step | What | Measure Before Proceeding |
|------|------|---------------------------|
| 1 | Fix compilation (missing types) | Baseline: current cycle count |
| 2 | Simplest partition + 8-way select | Did cycles improve? |
| 3 | Add TIK/TOK independent caches | Further improvement? |
| 4 | Multi-pass (only if needed) | Only if Step 2-3 show index spread |

**Each feature below is marked:**
- 🟢 **IMPLEMENT FIRST** - simple version, do now
- 🟡 **MEASURE FIRST** - implement only if data shows need
- 🔴 **FUTURE** - note in design but don't implement yet

| Metric | Baseline | Target |
|--------|----------|--------|
| Cycles | 1,757 | ~1,450 (estimate) |
| Speedup | 84x | ~102x |

---

## ⚠️ CRITICAL GOTCHAS - Read Before Implementation

### GOTCHA 1: TIK/TOK is NOT "all 0s" vs "all 1s"

**WRONG:** TIK has all items with bit=0, TOK has all items with bit=1

**CORRECT:**
- TIK (groups 0-15) and TOK (groups 16-31) BOTH have mixed 0s and 1s
- Key property: TOK indices are HIGHER than TIK indices (monotonically increasing)
- Each half maintains its OWN independent tree cache

### GOTCHA 2: Sub-Partition, Not Global Re-Partition

**WRONG:** Round 4 partitions ALL items by bit 4, destroying bit 0-3 sorting

**CORRECT:**
- Each round SUB-partitions WITHIN existing buckets
- Round 4: Within each 4-bit bucket, split by bit 4 → 5-bit buckets
- Sorting ACCUMULATES across rounds, never destroyed

### GOTCHA 3: Control Flow with Code Duplication (🟡 MEASURE FIRST)

When TIK branches (e.g., span > 8):
- Branch target A: TIK single-pass + TOK code (copy 1)
- Branch target B: TIK multi-pass + TOK code (copy 2, IDENTICAL)

**Only implement if multi-pass is needed!**

### GOTCHA 4: Non-Aligned Tree vloads

Tree vloads can start at ANY offset: `vload tree[25:33]` is valid.

### GOTCHA 5: Bounded Hash Waste (🟡 MEASURE FIRST)

Multi-pass: some items compute hash with wrong tree values, recomputed later.
Extra work is bounded (sparse groups balanced by dense groups).

### GOTCHA 6: Bounded De-Sorting (🔴 FUTURE)

We don't need perfect sorting within each vload. Parallel writes to 2-4
positions instead of sequential counters. Only implement if ALU is bottleneck.

---

## 🟢 STEP 1: Simplest Working Version (IMPLEMENT NOW)

**Goal:** Get partition working with minimal complexity. Measure.

### What to implement:
1. Fix PartitionBuffer type (rename PartitionHalf or add alias)
2. Add `get_read_buffer()` and `should_partition()` helpers
3. Use existing `emit_stable_partition_group()` with global counters
4. Use existing 8-way selection (compare-accumulate)
5. Single tree cache (not TIK/TOK independent yet)

### What NOT to implement yet:
- ❌ Multi-pass windowed access
- ❌ Control flow branching
- ❌ Bounded de-sorting optimization
- ❌ Per-bucket boundary tracking for rounds 4+

### Measurement checkpoint:
```bash
cd scheduler_rs && maturin develop
python test_optimized.py --rust-builder
# Record: cycle count, correctness pass/fail
```

**If cycles IMPROVE:** proceed to Step 2
**If cycles WORSE:** analyze why before adding complexity

---

## 🟡 STEP 2: Add Complexity Only If Needed

### 2a. TIK/TOK Independent Caches
- **When:** If Step 1 shows tree cache thrashing between halves
- **What:** Each half maintains its own 16-value cache

### 2b. Multi-Pass Windowed Access
- **When:** If Step 1 shows groups with index spread > 8
- **What:** Branch on span, process in multiple passes

### 2c. Control Flow with Code Duplication
- **When:** If 2b is needed and fixed pass count doesn't work
- **What:** TIK branches duplicate TOK code, vice versa

### 2d. Bounded De-Sorting
- **When:** If partition store ALU is bottleneck
- **What:** Parallel writes to 2-4 positions instead of sequential counters

---

## The Tree Sub-Partition Property (Background)

This is the fundamental insight that makes the optimization work.

### Binary Tree Traversal Creates Nested Partitions

At each round, `idx = idx * 2 + (val & 1)`. This means:

```
Round 0: All items at idx=0
         After branch: idx ∈ {0, 1}

Round 1: Items with idx=0 → {0, 1}
         Items with idx=1 → {2, 3}   ← NEVER crosses to {0,1}!

Round 2: {0,1} → {0,1,2,3}
         {2,3} → {4,5,6,7}           ← NEVER crosses!

Round 3: {0,1,2,3} → {0..7}
         {4,5,6,7} → {8..15}
```

**Critical property**: Items in partition P can ONLY go to sub-partitions 2P or 2P+1.
They can NEVER jump to a different branch of the tree.

### Why This Enables Efficient Tree Access

After N rounds of stable partitioning, items are sorted by their N-bit branch history:
- Items with prefix `0000` access tree indices 0-1 at round 4
- Items with prefix `0001` access tree indices 2-3 at round 4
- Items with prefix `0010` access tree indices 4-5 at round 4
- ...and so on

**Within each SIMD group of 8 consecutive items**, all indices are contiguous!
This means we can:
1. vload 8 consecutive tree values (1 vload instead of 8 scattered loads)
2. Use simple 2-way select based on the current branch bit

---

## Phased Partition Strategy

### Why We Don't Need Partition for Rounds 0-3

| Round | Depth | Tree indices | Access method |
|-------|-------|--------------|---------------|
| 0 | 0 | {0} | Broadcast tree[0] - all same |
| 1 | 1 | {1, 2} | Delta select (2 values) |
| 2 | 2 | {3..6} | Delta select (4 values) |
| 3 | 3 | {7..14} | 8-way VALU select (preloaded) |
| 4+ | 4+ | {15..} | **Need partition!** |

Rounds 0-3 use fast broadcast/delta/VALU selection - no scattered loads.
Round 4+ has too many possible tree indices - needs partition for efficient access.

### Spreading Partition Stores Across Rounds 0-3

**We have 4 full rounds to complete the partition stores before round 4 needs them!**

Each partition (TIK or TOK) has:
- 16 groups × 8 items = 128 items
- 128 idx stores + 128 val stores = 256 stores total
- At 2 stores/cycle = 128 cycles needed

But we have ~100 cycles of hash per round × 4 rounds = 400 cycles available!

**Store schedule (per partition):**

| Round | Groups stored | Stores | % Complete |
|-------|--------------|--------|------------|
| 0 | 0-3 | 64 | 25% |
| 1 | 4-7 | 64 | 50% |
| 2 | 8-11 | 64 | 75% |
| 3 | 12-15 | 64 | 100% |

By round 4, partition is complete and ready for sorted vload access!

---

## TIK/TOK Interleaving

### TIK/TOK Structure (CORRECTED - See GOTCHA 1)

**TIK and TOK are NOT split by branch bit!** Both halves have mixed values.

The split is by GROUP NUMBER:
- **TIK**: Groups 0-15 (items 0-127) — tend to have LOWER tree indices
- **TOK**: Groups 16-31 (items 128-255) — tend to have HIGHER tree indices

### 🟡 Independent Tree Caches (Step 2a - Measure First)

- TIK maintains its own 16-value cache (2 vloads)
- TOK maintains its own 16-value cache (2 vloads)
- When TIK's indices advance past its cache, TIK vloads new values
- TOK does the same independently
- At most ~1 overlapping vload when transitioning between halves

**Implement in Step 2a only if Step 1 shows cache thrashing!**

### Natural Pipelining

During TIK's hash computation, store slots are IDLE (hash uses VALU only).
We use those idle store slots for TOK's partition stores, and vice versa.

```
┌─────────────────────────────────────────────────────────────────┐
│ TIK: [tree access] [hash ~~~~~~~~~~~~~~~~~~~~~~~~~~~~]           │
│                     ↑ VALU busy, Store slots IDLE                │
│ TOK partition stores: [store][store][store]...                   │
│                        ↑ Use TIK's idle store slots!             │
└─────────────────────────────────────────────────────────────────┘
```

Long dependency chains are absorbed by TIK/TOK pipelining.

### Why Stores Are "Free"

Hash computation uses:
- **VALU**: 6 slots (fully utilized)
- **ALU**: Some slots
- **Store**: 0 slots (IDLE!)
- **Load**: 0 slots (IDLE!)

Partition stores use:
- **Store**: 2 slots
- **Load**: 2 slots (load from scratch)
- **ALU**: Several (address computation)

**These don't conflict!** The scheduler interleaves them automatically.

---

## Memory Layout

### Extra Room Buffer (512 words total)

With TIK/TOK split, we only need half the memory:

```
Offset    Size    Content
──────────────────────────────────────
0         128     TIK idx (16 groups × 8)
128       128     TIK val (16 groups × 8)
256       128     TOK idx (16 groups × 8)
384       128     TOK val (16 groups × 8)
──────────────────────────────────────
Total: 512 words
```

**Note**: We can optionally pack idx (2 per word) to reduce to 384 words,
but unpacked is simpler and 512 words fits comfortably in extra_room.

---

## Detailed Algorithm

### Part 1: Partition Store (during rounds 0-3)

For each group g in the current round's slice:

```
emit_partition_group(g, left_ptr, right_ptr, idx_base, val_base):
    # Step 1: vstore to scratch for scalar access
    vstore scratch_idx, all_idx[g]
    vstore scratch_val, all_val[g]

    # Step 2: For each of 8 items, scatter by branch bit
    for i in 0..8:
        val = load(scratch_val + i)
        idx = load(scratch_idx + i)
        bit = val & 1

        # Compute destination pointer
        # dest = bit ? right_ptr : left_ptr
        # Using: dest = left_ptr + bit * (right_ptr - left_ptr)
        diff = right_ptr - left_ptr
        dest = left_ptr + bit * diff

        # Store to partition buffer
        store(idx_base + dest, idx)
        store(val_base + dest, val)

        # Increment counters
        left_ptr  += (1 - bit)  # Increment if bit=0
        right_ptr += bit        # Increment if bit=1
```

**Operations per item**: 2 loads, 2 stores, ~8 ALU
**Operations per group**: 16 loads, 16 stores, ~64 ALU, 2 vstores

### Part 2: Sorted vload Access (rounds 4+)

**Two independent vload streams, not matched 1:1:**

**Stream 1: Data (from partition buffer)**
```
Per 2 groups (with idx packing):
  1 vload: packed idx (8 words covers 2 groups, 4 words each)
  2 vloads: val (1 per group)

For 32 groups: 16 idx vloads + 32 val vloads = 48 data vloads per depth
```

**Stream 2: Tree (just 2 recent vloads, no cache management)**
```
- Keep only 2 most recent tree vloads (16 values total)
- Sorted data → tree indices increase monotonically
- When indices advance past current 16 values, vload next chunk
- Old values discarded - no cache needed!
- ~16-32 tree vloads per depth (depends on index spread)
```

**The two streams are independent:**
- Data vloads: predictable pattern, ~48 per depth
- Tree vloads: monotonically advancing, ~24 per depth
- Just 2 tree vload registers needed, no complex cache

**VALU operations:**
- Unpack idx from packed words
- 2-way select: `result = tree[0] + (tree[1] - tree[0]) * (idx & 1)`

---

## Cycle Budget Analysis

### Baseline (depths 4-9, no partition)

| Component | Per group | Groups | Total |
|-----------|-----------|--------|-------|
| Tree access (8 scattered loads) | 4 cycles | 32×6 depths | 768 cycles |
| Address computation | ~2 cycles | 32×6 | 384 cycles |
| **Subtotal for depths 4-9** | | | **~1152 cycles** |

### With Lazy Partition (depths 4-9)

| Component | Per group | Groups | Total |
|-----------|-----------|--------|-------|
| Partition load (2 vloads) | 1 cycle | 32×6 | 192 cycles |
| Tree vload | 0.5 cycles | 32×6 | 96 cycles |
| 2-way select (VALU) | ~1 cycle | 32×6 | 192 cycles |
| **Subtotal for depths 4-9** | | | **~480 cycles** |

**Partition store cost**: ~FREE (overlaps with hash)

### Net Savings

```
Baseline depths 4-9:     ~1152 cycles
With partition:           ~480 cycles
──────────────────────────────────────
Savings:                  ~672 cycles
```

But this is optimistic. More realistic estimate accounting for:
- Load port contention (partition loads + tree vloads)
- Additional setup overhead
- Scheduler inefficiencies

**Conservative estimate**: 300-400 cycle savings

### Projected Performance

| Metric | Baseline | Projected |
|--------|----------|-----------|
| Cycles | 1,757 | ~1,400-1,450 |
| Speedup | 84.0x | ~102-105x |
| Target (100x) | 1,487 | **Achievable!** |

---

## Critical Review Findings (and Corrections)

Independent review agents identified issues, but made a critical error in their analysis.

### Review Error: Confused Partition ACCESS with Partition STORE

The reviews incorrectly assumed partition ACCESS needs scalar loads.

**Wrong (review assumption):**
```
Partition access: 256 scalar loads (16 loads/group × 16 groups)
At 2 loads/cycle = 128 cycles
```

**Correct:**
```
Partition access: 64 VLOADS (2 vloads/group × 32 groups)
At 2 vloads/cycle = 32 cycles per depth
```

**The partition buffer is loaded with VLOAD, not scalar loads!**

### Partition STORE Phase (rounds 0-3)

This DOES need scalar loads (to scatter items by branch bit):
```
Per partition (16 groups):
- 32 vstores (write vectors to scratch)
- 256 scalar loads (read individual items)
- 256 scalar stores (write to partition buffer)
```

**But spread across 4 rounds of hash (~400 cycles available), this overlaps!**

With idx packing (2 per word): only 384 stores total.
48 vloads worth of setup for 384 stores.

### Partition ACCESS Phase (rounds 4+)

**Per group:**
- 1 vload: partition idx → all_idx[g]
- 1 vload: partition val → all_val[g]
- 1 vload: tree values
- ~4 VALU: 2-way select

**Total: 3 vloads + 4 VALU per group**

---

## Corrected Cost Analysis

### Baseline (depths 4-9): ~768 cycles
- 32 groups × 6 depths × 8 scattered loads
- 1536 loads ÷ 2 loads/cycle = **768 cycles**

### With Partition (depths 4-9): ~288 cycles + VALU
- 32 groups × 6 depths × 3 vloads = 576 vloads
- 576 vloads ÷ 2 vloads/cycle = **288 cycles**
- Plus VALU for 2-way select: 192 groups × 4 ops = 768 ops ÷ 6 = ~128 cycles

**Total ACCESS: ~288 + 128 = ~416 cycles** (VALU overlaps with loads)

### Partition Store Overhead
- Spread across rounds 0-3 (~400 cycles of hash time)
- Stores use idle store slots during hash
- Loads for partition setup: ~128 cycles but overlapped
- **Net overhead: ~50-100 cycles** (only the portion that doesn't fit in hash)

### Net Savings: ~300-400 cycles!
```
Baseline depths 4-9:     768 cycles
With partition:          ~416 cycles
Partition overhead:      ~50 cycles
─────────────────────────────────────
Net savings:             ~300 cycles
```

**The original estimate was correct!**

---

## Risk Analysis

### Critical Risks

1. **Dependency chain dominates**: The vstore→load→vload→select chain creates
   a bottleneck similar to scattered loads. The optimization may provide
   negligible benefit for significant complexity.

2. **Load port saturation**: Both partition stores (256 loads) and sorted access
   (3 vloads/group × 192 groups) compete for 2 load slots/cycle.

3. **Correctness edge cases**: SIMD groups spanning partition boundaries,
   unbalanced splits, buffer overflow potential.

### Lower Risks

1. **Scratch space**: 512 words fits in 1536 budget
2. **Algorithm correctness**: Tree sub-partition property is mathematically sound

### Verification Plan

1. **Correctness first**: Verify against reference kernel before optimizing
2. **Measure actual cycles**: Don't trust estimates - profile the implementation
3. **Test edge cases**: Unbalanced partitions, boundary-spanning SIMD groups

---

## Implementation Phases (Revised - Incremental)

### 🟢 Phase 1: Fix Compilation Issues (DO NOW)
- [ ] Fix PartitionBuffer type (line 175, 431, 434 in partition.rs)
- [ ] Add `get_read_buffer()` function
- [ ] Add `should_partition()` function
- [ ] Verify code compiles: `cd scheduler_rs && cargo check`

### 🟢 Phase 2: Simplest Partition + 8-Way Select (DO NOW)
- [ ] Use existing `emit_stable_partition_group()` with global counters
- [ ] Use existing 8-way compare-accumulate selection
- [ ] Wire up in `emit_round` for depths 4+
- [ ] Enable with `enable_lazy_partition = true`

### 📊 MEASUREMENT CHECKPOINT
```bash
cd scheduler_rs && maturin develop
python test_optimized.py --rust-builder
# Record: cycle count, correctness
```
**STOP HERE. Analyze results before proceeding.**

### 🟡 Phase 3: Add TIK/TOK Independent Caches (IF NEEDED)
- [ ] Only if Phase 2 shows cache thrashing
- [ ] Each half maintains own 16-value tree cache
- [ ] Measure improvement

### 🟡 Phase 4: Multi-Pass Windowed Access (IF NEEDED)
- [ ] Only if Phase 2/3 shows index spread > 8
- [ ] Branch on span check
- [ ] Code duplication for TIK/TOK branches
- [ ] Measure improvement

### 🔴 Phase 5: Bounded De-Sorting (FUTURE)
- [ ] Only if partition store ALU is measured bottleneck
- [ ] Parallel writes to 2-4 positions
- [ ] Do NOT implement unless data shows need

---

## Appendix: Sub-Partition Visualization

```
Round 0: [──────────────────────────────────────]
         All 256 items, all access tree[0]

Round 1: [────────────────][────────────────────]
         Left (bit=0)      Right (bit=1)
         Access tree[1]    Access tree[2]

Round 2: [────────][────────][────────][────────]
         LL       LR        RL        RR
         tree[3]  tree[4]   tree[5]   tree[6]

Round 3: [────][────][────][────][────][────][────][────]
         LLL  LLR  LRL  LRR  RLL  RLR  RRL  RRR
         t[7] t[8] t[9] t[10] t[11] t[12] t[13] t[14]

Round 4+: Each bucket accesses 2 adjacent tree nodes
          (the current round's bit determines which)
```

The key: Items NEVER cross partition boundaries. Once sorted, they stay sorted
within their partition, just subdividing further.
