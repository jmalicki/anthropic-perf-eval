# 5r/256 Depth-3 Validation Failure - Trace Findings

## Phase 1 Summary

**Failure:** Group 20 (batch 160-167) produces wrong value and index.

**Key cycles:**
- **521:** XOR writes hash to s126: `['^', 126, 142, 462]`
- **522:** SUB overwrites s126: `['-', 126, 27, 142]`; vstore reads s126: `['vstore', 90, 126]`
- **523:** multiply_add reads s126 as offset: `['multiply_add', 182, 622, 27, 126]` (idx = idx*2 + offset)

## Phase 2 Classification: Scratch Reuse / Hash-as-Offset

**Evidence:**
1. Hash (XOR output) written to s126 at cycle 521.
2. multiply_add at 523 uses s126 as the offset operand (hash parity bit).
3. SUB at cycle 522 overwrites s126 with (27 - 142) before multiply_add reads it.
4. multiply_add thus gets wrong offset → wrong idx (e.g. got 37, ref 62 for batch 160).

**Root cause:** Two distinct VRegs—(1) hash output (consumer: vstore at 522) and (2) offset (consumer: multiply_add at 523)—share the same scratch address 126, or the hash VReg's liveness was not extended to cycle 523. Per T1/T2, overlapping intervals require disjoint addresses.

**Bug class:** Scratch reuse / hash-as-offset (matches BUGS.md entry). The hash value's live interval should extend to cycle 523 (last read by multiply_add), but the allocator allowed reuse at 522.

## Recommended Next Steps

- Phase 3: Check scheduler JSON for vreg_to_addr and intervals at address 126.
- Phase 6: Apply UNIFY_SCRATCH_LIVENESS_PLAN; fix address_assign or CP-SAT model.
