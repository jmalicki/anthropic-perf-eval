# vload Optimization for Tree Gather - Understanding Document

## Definitions

### Batch
The 256 input elements being processed in parallel. Each batch element has:
- An **index** into the tree (starts at 0, updates each round)
- A **value** that gets hashed each round

### Group
A set of 8 batch elements processed together as a SIMD vector. With VLEN=8 and batch_size=256:
- 32 groups total (256 / 8 = 32)
- Group g contains batch elements [g*8, g*8+1, ..., g*8+7]

### Iteration
Groups are processed in iterations of UNROLL=8 groups each:
- 4 iterations total (32 / 8 = 4)
- Iteration it processes groups [it*8, it*8+1, ..., it*8+7]

### Round
One traversal step down the tree. With 16 rounds:
- Round 0: All elements at root (index 0)
- Round 1: Elements at level 1 (indices 1-2)
- Round 2: Elements at level 2 (indices 3-6)
- Round 3+: Elements at level 3+ (indices 7+), requires gather

---

## Current Optimizer Design

### Architecture: VLIW with Multiple Engines

Per cycle, we can execute:
- **VALU**: 6 slots (vector arithmetic, each processes 8 elements)
- **ALU**: 12 slots (scalar arithmetic)
- **Load**: 2 slots (memory reads)
- **Store**: 2 slots (memory writes)
- **Flow**: 1 slot (control flow)

### Current Kernel Structure (interleaved_unrolled_kernel.py)

**Fully unrolled**: All 16 rounds × 32 groups × all operations emitted as explicit instructions.

**Per-round structure**:
1. **Gather/Compute phase**: Get tree node value for each batch element
   - Rounds 0-2: Compute from cached tree values (VALU only)
   - Rounds 3-15: Gather from memory (ALU + Load)

2. **XOR phase**: value ^= node_value (VALU)

3. **Hash phase**: 6 hash stages (VALU heavy - 18 ops per element)

4. **Branch phase**: Compute next index (VALU with multiply_add fusion)

**Current gather implementation (rounds 3-15)**:
```python
for vi in range(VLEN):  # 8 vector elements
    for i in range(0, UNROLL, 2):  # pairs of groups
        for it in range(n_iters):  # 4 iterations
            # Address calculation (ALU)
            emit(alu=[("+", st_a, forest_p, all_idx[group] + vi),
                      ("+", st_b, forest_p, all_idx[group+1] + vi)])
            # Memory load (Load)
            emit(load=[("load", v_nv[it][i] + vi, st_a),
                       ("load", v_nv[it][i+1] + vi, st_b)])
```

This emits 8 ALU + 8 Load ops per group, totaling:
- 13 rounds × 32 groups × 8 = 3,328 scalar loads
- 13 rounds × 32 groups × 8 = 3,328 ALU ops

### Scheduling (round_chunked_scheduler.py)

Operations are scheduled using CP-SAT (constraint programming):
1. Extract all operations with their dependencies
2. Split into chunks of ~1500 ops
3. For each chunk, solve optimal slot assignment
4. Stitch chunks together

**Current bottleneck**: VALU at 87% utilization (5.2/6 slots per cycle)

### Multi-Path IR (multi_path_ir.py)

Infrastructure for instruction selection - operations can have multiple implementations:

```python
def arithmetic_flexible(self, dest, src1, src2, op, ...):
    """Scheduler chooses VALU or ALU implementation."""
    valu_impl = Implementation(Engine.VALU, [(op, dest, src1, src2)])
    alu_impl = Implementation(Engine.ALU,
                              [(op, dest+i, src1+i, src2+i) for i in range(8)])
    return self.add_op(name, [valu_impl, alu_impl], deps)
```

The CP-SAT scheduler picks which implementation minimizes total cycles.

---

## Tree Memory Layout (BFS Order)

```
Index 0:      root (level 0) - 1 node
Index 1-2:    level 1 - 2 nodes
Index 3-6:    level 2 - 4 nodes
Index 7-14:   level 3 - 8 nodes    ← exactly 8, perfect for vload!
Index 15-30:  level 4 - 16 nodes
Index 31-62:  level 5 - 32 nodes
...
Level L: indices [2^L - 1, 2^(L+1) - 2], total 2^L nodes
```

---

## My Understanding of Your vload Suggestion

### Key Insight #1: Level 3 Has Exactly 8 Nodes

At round 3, all batch elements need values from indices 7-14.

**ONE vload loads the ENTIRE level:**
```
vload(dest, forest_p + 7) → loads forest[7:15] into scratch
```

This is 1 Load op instead of 256 scalar loads!

### Key Insight #2: Speculative Multi-Level Loading

From any node index `i`, a vload at `forest_p + i` loads:
- The node at `i`
- Plus 7 following nodes (which include descendants in BFS layout)

So from index 0: `vload` gets forest[0:8] = levels 0, 1, 2, and part of level 3.

### Key Insight #3: The Offset Vector

Keep a constant vector in scratch:
```
v_offsets = [0, 1, 2, 3, 4, 5, 6, 7]
```

I'm uncertain how this helps since vload already loads 8 consecutive values from a scalar address. Is it for:
- Computing which loaded element each batch element needs?
- Something else I'm missing?

### The Two Paths for Scheduler to Choose

**Path A: Traditional Scalar Gather**
```
Per group (8 elements):
  - 8 ALU ops: compute addresses
  - 8 Load ops: read from memory
```

**Path B: vload + Selection**
```
Once per level (shared across all 32 groups):
  - 1 vload: load entire level into scratch

Per group:
  - VALU ops: select correct value for each element
```

### My Uncertainty: The Selection Step

After vloading forest[7:14] into scratch at address V:
- Batch element with idx=10 needs V[10-7] = V[3]
- How do we efficiently extract V[local_idx] for each element?

My current understanding requires comparison chains:
```
local_idx = idx - 7
result = 0
for k in 0..7:
    mask = (local_idx == k)
    result += broadcast(V[k]) * mask
```
This is ~31 VALU ops per group.

**Is there a simpler selection method I'm missing?**

---

## ISA Analysis: Why vload + Selection is Challenging

### The Core Problem

After vloading `tree[7:14]` into scratch (say at address V), we need to extract `V[idx-7]` for each batch element. The ISA presents a fundamental challenge:

**Load instruction**: `("load", dest, addr)` loads from `mem[scratch[addr]]`
- It reads a memory address FROM scratch, then loads FROM memory
- It does NOT load from scratch at a computed offset

**There is no**: `("load_from_scratch", dest, base, offset)` that would compute `scratch[base + scratch[offset]]`

### Attempted Workaround

One approach:
1. vload tree[7:14] into scratch at V
2. vstore V back to memory at temp location T
3. Per element: compute `T + (idx - 7)`, load from memory

But step 3 requires:
- 1 ALU: compute `idx - 7`
- 1 ALU: compute `T + (idx - 7)`
- 1 Load: load from memory

That's **2 ALU + 1 Load** per element, vs the current **1 ALU + 1 Load**.

The vload approach is actually WORSE for rounds 3+!

### What the Offset Vector Could Enable

The offset vector `[0,1,2,...,7]` could help if we had a vgather instruction:
```
vgather dest, base_vec, offset_vec
# dest[i] = mem[base_vec[i] + offset_vec[i]]
```

Or a scatter-load from scratch:
```
vload_indirect dest, base, index_vec
# dest[i] = scratch[base + index_vec[i]]
```

Neither exists in this ISA.

---

## Current Performance Analysis

With CP-SAT scheduling:
- **Current**: 2452 cycles (60.20x speedup)
- **Target**: 1487 cycles (100x speedup)
- **Gap**: Need ~40% reduction

### Resource Utilization (approximate)
- VALU: ~87% (5.2/6 slots per cycle) - **bottleneck**
- ALU: ~12% (1.4/12 slots per cycle) - **underutilized**
- Load: ~71% (1.4/2 slots per cycle)

### Key Insight
The bottleneck is VALU, not Load. Moving work FROM VALU TO other engines could help.

---

## Potential Optimization Directions

### 1. Shift VALU Work to ALU
The multi-path IR already supports VALU→ALU alternatives. Key question: is the scheduler actually using ALU when beneficial?

### 2. Reduce Hash VALU Operations
Hash is 6 stages × 3 VALU ops = 18 VALU ops per element per round.
- Can any hash operations use ALU?
- Can we fuse hash operations?

### 3. Reduce Selection VALU in Rounds 0-2
Current rounds 1-2 use VALU comparisons. Alternative: use cached values differently?

### 4. Inter-Round Pipelining
Overlap gather operations of round N+1 with hash operations of round N.

---

## Questions for Clarification

1. **Selection efficiency**: Given the ISA limitation (no gather from scratch), is there a way to do selection that I'm missing?

2. **Multi-level loading**: When you say "load a node, plus two levels deeper" - is this about speculative loading for future rounds?

3. **Offset vector usage**: How does broadcasting an index and adding the offset vector help, given we can't gather from scratch?

4. **Alternative ISA operations**: Are there any instructions I'm overlooking that could enable efficient selection?
