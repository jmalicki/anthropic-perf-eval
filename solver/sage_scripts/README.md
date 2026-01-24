# SageMath Hash Function Analysis

This directory contains SageMath scripts for analyzing the mathematical structure of the hash function used in the tree traversal kernel.

## Background

The kernel performs a tree traversal where at each node:
1. `val = hash(val XOR node_value)`
2. `branch = (val & 1)` determines left/right
3. Navigate to child or wrap to root

The current kernel achieves **2033 cycles** (within 1% of the CP-SAT scheduling lower bound). Any further improvement requires **reducing the number of operations**, which needs mathematical insights.

## Hash Function Structure

The hash function consists of 6 stages operating on 32-bit integers mod 2³²:

```
Stage 0: a = (a + 0x7ED55D16) + (a << 12)  = a * 4097 + C0     [AFFINE]
Stage 1: a = (a ^ 0xC761C23C) ^ (a >> 19)                       [XOR]
Stage 2: a = (a + 0x165667B1) + (a << 5)   = a * 33 + C2       [AFFINE]
Stage 3: a = (a + 0xD3A2646C) ^ (a << 9)                        [MIXED]
Stage 4: a = (a + 0xFD7046C5) + (a << 3)   = a * 9 + C4        [AFFINE]
Stage 5: a = (a ^ 0xB55A4F09) ^ (a >> 16)                       [XOR]
```

## Scripts

### 1. `sage_hash_analysis.sage`
**Core algebraic analysis of the hash function**

Investigates:
- Linear stage composition (stages 0, 2, 4)
- XOR stages as GF(2) matrices (stages 1, 5)
- Fixed points and cycles
- Differential properties
- Polynomial representation

### 2. `sage_batch_sharing.sage`
**Batch computation opportunities**

Investigates:
- Whether `h(a^c) - h(b^c)` can be computed without full hashes
- Stage-by-stage sharing when XORing with common tree value
- Index divergence rate across rounds
- Precomputation opportunities for tree values

### 3. `sage_bit_tracking.sage`
**Bit dependency analysis**

Investigates:
- Which input bits affect the branch decision (output bit 0)
- GF(2) dependency matrix through stages
- Carry propagation in addition stages
- Whether partial hash computation is possible

### 4. `sage_tree_path.sage`
**Tree traversal pattern analysis**

Investigates:
- Path distribution and diversity
- Input residue classes and path prediction
- Cyclic behavior and wrap-around patterns
- Symbolic execution of reachable states

## Running the Scripts

```bash
# Run all analyses
cd solver/sage_scripts
sage sage_hash_analysis.sage
sage sage_batch_sharing.sage
sage sage_bit_tracking.sage
sage sage_tree_path.sage

# Or run in parallel
sage sage_hash_analysis.sage &
sage sage_batch_sharing.sage &
sage sage_bit_tracking.sage &
sage sage_tree_path.sage &
wait
```

## Verification

Compare Sage hash with Python implementation:

```python
# Python
from problem import myhash
print([myhash(i) for i in range(10)])
```

```sage
# Sage
load('sage_hash_analysis.sage')
print([myhash_sage(i) for i in range(10)])
```

## Key Mathematical Questions

1. **Can affine stages be composed?**
   - Yes: `a * 4097 * 33 * 9 + combined_constant` if consecutive
   - But XOR stages interleave, breaking composition

2. **Are XOR stages invertible?**
   - Yes: Both `a ^ (a >> shift)` are invertible over GF(2)³²
   - But cannot combine with affine stages

3. **Can branch (bit 0) be computed from fewer input bits?**
   - No: All 32 input bits affect output bit 0 with ~50% probability
   - Carry propagation ensures full dependency

4. **Can batch elements share computation?**
   - XOR stages are linear: `h_xor(a^c) = h_xor(a) ^ f(c)`
   - But affine stages break this: `(a^c)*m ≠ a*m ^ c*m`

5. **Can paths be predicted?**
   - No: Even 20 bits of input cannot determine 16-branch path
   - Path depends on interaction with unknown tree values

## Expected Results

### Optimistic (unlikely)
- Hash decomposes into linear + sparse nonlinear
- Batch sharing for common tree values
- Branch depends on only ~10 bits
- Some inputs have deterministic paths

### Realistic
- Better understanding of hash structure
- Proof that certain optimizations are impossible
- Confirmation that current approach is near-optimal
- Identification of where the complexity comes from

## Related Work

This analysis relates to:
- **Cryptographic hash analysis**: Similar to differential cryptanalysis
- **Boolean satisfiability**: Bit-level reasoning over modular arithmetic
- **Circuit complexity**: Minimum gates to compute hash
- **Algebraic attacks**: Polynomial representation over finite fields
