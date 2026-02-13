# 4-Round Kernel Deep Debug Results

**Date:** 2026-02-10  
**Kernel:** 4 rounds, 8 batch, depth-3  
**Plan:** `/Users/josephmalicki/.cursor/plans/deep_debug_4-round_kernel_hash_liveness_55c99403.plan.md`

## Executive Summary

After deep investigation of the 4-round kernel, we have **definitively determined**:

1. **Round 2 branch correctly reads Round 2 hash** (not Round 1 hash)
2. **Liveness computation is correct** - Round 1 hash VReg extends to cover Round 1 XOR reads
3. **Dependency chain is correct** - Round 1 hash → Round 1 XOR → Round 2 hash → Round 2 branch
4. **Kernel still fails validation** - indicating a different root cause than originally suspected

## Key Findings

### 1. Round 2 Branch Reads Correct VReg

**Op 159** (cycle 50, Round 2 branch phase):
- Reads VReg 159 (Round 2 hash output) ✓
- Reads from addresses 13-20 (contiguous, correct) ✓
- VReg 159 is live at cycle 50 ✓

**Emission Debug Confirms:**
```
[EMISSION DEBUG] Round 2 branch phase: reading all_val[0] = VReg 159 (group 0)
```

**Conclusion:** Round 2 branch correctly reads Round 2 hash via `all_val[g]`, NOT Round 1 hash.

### 2. Dependency Chain Analysis

**Round 1 Hash → Round 1 XOR → Round 2 Hash:**
- Round 1 hash (v115_v) produced by Op 114 at cycle 15
- Round 1 XOR (Op 123) reads v115 at cycle 22, produces v123_v ✓
- Round 2 hash (Op 124) reads v123 (Round 1 XOR output) at cycle 23 ✓

**Dependency chain is correct** - Round 2 hash reads Round 1 XOR output, not Round 1 hash directly.

### 3. Liveness Computation is Correct

**Round 1 hash VReg (v115):**
- Defined at cycle 15
- Read by Op 115 (Round 0 branch) at cycle 16 → interval extends to [15, 16]
- Read by Op 123 (Round 1 XOR) at cycle 22 → interval extends to [15, 22] ✓
- Interval correctly covers all reads ✓

**Liveness debug output:**
```
[LIVENESS DEBUG] Round 1 hash VReg v115_v defined: op id=114 opcode=Xor cycle=15 round=Some(0) phase=Some("hash") interval=[15, 15]
[LIVENESS DEBUG] Round 1 hash VReg v115_v read: op id=115 opcode=And cycle=16 round=Some(0) phase=Some("branch") read_cycle=16
[LIVENESS DEBUG] Extended Round 1 hash VReg v115_v interval: [15, 15] -> [15, 16] (read by op id=115)
[LIVENESS DEBUG] Round 1 hash VReg v115_v read: op id=123 opcode=Xor cycle=22 round=Some(1) phase=Some("xor") read_cycle=22
[LIVENESS DEBUG] Extended Round 1 hash VReg v115_v interval: [15, 16] -> [15, 22] (read by op id=123)
```

**Conclusion:** Liveness computation correctly extends Round 1 hash VReg to cover Round 1 XOR reads.

### 4. Address Assignment Analysis

**VReg 159 (Round 2 hash):**
- Assigned to address 13
- Should span addresses 13-20 (size 8 words)
- Live at cycle 50 (when Op 159 reads it) ✓

**VReg 115 (Round 1 hash):**
- Assigned to address 94
- Interval [15, 22] - expires before cycle 50 ✓
- Address 94 reused for other VRegs after cycle 22 ✓

**Conclusion:** Address assignment is correct. Round 1 hash and Round 2 hash are at different addresses, so there's no conflict.

### 5. Validation Failure Analysis

**Kernel fails validation:**
- Value mismatch at batch index 1
- Index mismatch at batch index 1
- But Round 2 branch reads correct VReg (Round 2 hash)

**Possible Root Causes:**
1. **Different bug:** The validation failure might be caused by a different issue than originally suspected
2. **Configuration difference:** Original failure used `--no-cse-groups`, current kernel may use different options
3. **Timing issue:** Values might be computed correctly but read at wrong cycles
4. **Address scrambling:** Original failure showed scrambled addresses [23, 17, 22, 28, 19, 16, 13, 20], but current kernel reads contiguous addresses [13-20]

## Comparison with Original Failure

### Original Failure (from TRACE_ANALYSIS_ROOT_CAUSE.md):
- Op 159 reads from addresses `[23, 17, 22, 28, 19, 16, 13, 20]` (scrambled)
- Expected to read Round 1 hash from addresses `[13-20]`
- Round 1 hash VRegs expired before cycle 50

### Current Kernel:
- Op 159 reads from addresses `[13-20]` (contiguous, correct)
- Reads Round 2 hash (VReg 159) - correct behavior
- Round 1 hash VReg correctly expires at cycle 22 (after Round 1 XOR reads it)

**Key Difference:** Original failure had scrambled addresses, suggesting a different bug (possibly in address assignment or lowering). Current kernel has correct addresses but still fails validation.

## Root Cause Hypothesis

The original diagnosis may have been incorrect. The real issue might be:

1. **Address scrambling bug:** Original failure showed non-contiguous addresses, suggesting a bug in how vector ops are lowered or how addresses are assigned
2. **Different failure mode:** Current kernel fails validation but for different reasons than originally suspected
3. **Configuration-dependent:** The bug might only occur with specific options (e.g., `--no-cse-groups`)

## Next Steps

1. **Generate kernel with `--no-cse-groups`** to match original failure configuration
2. **Check if address scrambling occurs** in that configuration
3. **Investigate validation failure** - trace where values diverge from reference
4. **Check if there's a bug in vector address assignment** that causes scrambling

## Tools Used

1. **`scripts/inspect_ssa_dependencies.py`** - Enhanced to check branch ops
2. **Debug logging in `emit_branch()`** - Tracks what `all_val[g]` contains
3. **Debug logging in `compute_live_intervals()`** - Tracks Round 1 hash VReg liveness
4. **Debug logging in `emit_hash()`** - Tracks `all_val[g]` flow between rounds

## Files Modified

1. `scheduler_rs/src/kernel_builder/ssa_emit.rs` - Added debug logging in `emit_branch()`
2. `scripts/inspect_ssa_dependencies.py` - Enhanced to check branch ops

## Conclusion

The investigation shows that:
- **Liveness computation is correct** - Round 1 hash VReg correctly extends to cover Round 1 XOR reads
- **Dependency chain is correct** - Round 1 hash → Round 1 XOR → Round 2 hash → Round 2 branch
- **Round 2 branch reads correct VReg (Round 2 hash)** - not Round 1 hash as originally suspected
- **Address assignment appears correct** - contiguous addresses, no scrambling observed
- **Emission flow is correct** - `all_val[g]` correctly updated between rounds

**Key Discovery:** The originally suspected "liveness extension bug" where Round 1 hash VReg expires before Round 2 branch reads it is **NOT the root cause**. Round 2 branch correctly reads Round 2 hash, and Round 1 hash VReg correctly expires after Round 1 XOR reads it.

The validation failure is likely caused by a **different bug** than the liveness extension issue originally suspected. Possible causes:
1. **Value computation error** - Values computed incorrectly despite correct dependencies
2. **Timing issue** - Values read/written at wrong cycles
3. **Batch-specific bug** - Failure only affects certain batches (e.g., batch index 1)
4. **Configuration-dependent bug** - Only occurs with specific options

**Recommendation:** Investigate the validation failure using trace IDs to see where values diverge from the reference, rather than focusing on liveness intervals.
