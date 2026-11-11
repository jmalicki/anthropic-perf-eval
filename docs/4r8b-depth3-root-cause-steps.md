# Root Cause Investigation: 4r/8 Depth-3 Validation Failure

## Findings so far

### Data from scratch_vs_ref
- **Cycle 71 s[13..20]:** Lanes 4–7 = correct round-3 `hashed_val`; lanes 0–3 = wrong.
- **Cycle 72 vstore** reads from s[21] (hash output): final output reports all 8 lanes as wrong, but debug_validation_trace shows only lanes 0–3 are actually wrong and lanes 4–7 are correct.
- Divergence is **lane-specific** (0–3 wrong, 4–7 correct).

### Ruled out
- **Allocation conflicts:** `addr_conflict_report` shows no problematic overlaps at 13 or 21.
- **Wrong-round store:** Wrong values do not match round-2 `hashed_val`.

### Likely cause
Bug is **before or during** the round-3 hash for lanes 0–3. Most likely:

1. **Tree-selection bug:** Depth-3 tree-reduce chooses wrong `node_val` for lanes 0–3.
2. **val_after_xor wrong:** XOR stage uses wrong input for lanes 0–3.

## Next steps (in order)

### 1. Map tree-reduce output to cycles (DONE)
Schedule JSON shows:
- **tree_access round 3:** cycles 53–60, addrs 51, 76, 94, 102, 110, 142
- **branch_r3:** addr 13 at cycles 70–73 (hash output, bit extraction, offset, multiply_add)

### 2. Compare tree output to ref node_val (DONE — ROOT CAUSE FOUND)
At cycle 60, s[51..58] vs ref node_val:
- **Lanes 0, 4, 5, 6, 7:** OK
- **Lane 1:** got 0x06a4b6a5 (batch 5’s node_val) vs ref 0x2a857444
- **Lane 2:** got 0x2b29ea5e (lane 0’s node_val) vs ref 0x2ab9d716
- **Lane 3:** got 0x2b9f46bb (batch 7’s node_val) vs ref 0x2a857444

**Symptom:** Depth-3 tree-reduce produces wrong node_val for lanes 1–3. **Actual cause:** wrong idx (see section 4).

### 3. Trace val_after_xor
If tree output is correct, the bug is in the XOR or hash. Run `scratch_vs_ref` at the cycle just before the first hash stage, comparing to `val_after_xor`:

```bash
uv run python -m scripts.scratch_vs_ref ... --ref-key val_after_xor --ref-round 3 --cycles <xor_output_cycle>
```

### 4. Root cause: wrong tree index at round-3 tree access (FOUND)

**Trace chain:**
- Wrong node_val at cycle 60 → wrong vselect branch (bit1) at cycle 57
- bit1 (s[110]) = [1,1,1,1,0,0,0,0] but should be [1,0,1,0,0,0,0,0]
- bit1 comes from s[76] & 1 at cycle 56, where s[76] = idx_adj_shr
- idx_adj_shr wrong for lanes 1,3 → idx_adj wrong → **idx (s[43]) wrong**
- At cycle 54 start, **s[43] (idx):** kernel [13, 10, 13, 9, 12, 12, 11, 12] vs ref [13, 8, 14, 8, 12, 12, 11, 12]
- **Upstream:** Round 1 offset wrong [2,2,2,2,1,1,1,1] vs ref [2,1,2,1,1,1,1,1] → wrong hashed_val parity → wrong depth-1 node_val
- **Init:** s[102] (tree[1]) = 0x217f5688 vs ref 0x22433ed9; Add at cycle 10 uses wrong td_1 in s[2]

**Root cause:** Wrong tree index for lanes 1 and 3 when entering round-3 tree access. Lanes 1 and 3 have idx=10,9 instead of 8,8. The bug is **before** the tree access phase—in the **branch step** (round 2→3 transition) that computes the next round’s tree index.

**Fix:** Inspect init phase: loads of tc[1]/tc[2], Sub for td_1, Add for tree[1]. Ensure td_1 = tree[1] - tree[2] is correct. The wrong td_1 causes wrong depth-1 node_val → wrong hash parity → wrong branch offset → wrong idx.

### 5. Cross-check with passing config
Depth-2 at 4 rounds passes. Compare depth-2 vs depth-3 emission to see where lane handling diverges.

## Quick commands reference

```bash
# Conflict check
uv run python -m scripts.addr_conflict_report scheduler_4r_8b_d3.json --addr 13

# Scratch vs ref (multiple keys)
uv run python -m scripts.scratch_vs_ref kernel_4r_8b_d3_no_cse.json --rounds 4 --batch-size 8 \
  --addr 13 21 --cycles 68 69 70 71 72 --ref-key hashed_val --ref-round 3 --group 0

# Who wrote addr
# who_wrote_addr.py was replaced by inspect_schedule.py (needs the scheduler JSON of the same build):
uv run python -m scripts.inspect_schedule sched_4r_8b_d3_no_cse.json addr 13 --cycle 72 --bundle

# Compare depth-3 (use forest_height=10)
uv run python -m scripts.compare_depth3_tree_value --kernel kernel_4r_8b_d3_no_cse.json \
  --rounds 4 --batch-size 8 --forest-height 10 --group 0
```
