# 4-round optimization phase results

Record after each phase: git SHA, exact uv command, liveness peak, cycles, and confirmation that Python kernel verification passes with 4 rounds.

---

## Phase 1

- **Git SHA:** 2a32f71
- **Exact uv command:** `uv run generate_ssa_kernel.py --rounds 4 --use-greedy --validate -q`
- **Liveness peak:** 1489 words
- **Cycles:** 514
- **Verification:** PASSED — `uv run generate_ssa_kernel.py --rounds 4 --use-greedy --validate -q` exited successfully; kernel produces correct output (Python validation against reference kernel with 4 rounds).

---

## Before/after tree-reduce depth3 (treeselect)

- **Git SHA:** 68b1602
- Comparison with **all bells and whistles**: 4 rounds, `--enable-valu-shift`, greedy scheduler (CP-SAT + valu_shift for 4 rounds currently fails merged scratch budget check, so greedy was used to get returned cycles and peak liveness).
- **Exact command:** `uv run python scripts/run_before_after_treeselect.py --use-greedy -q`
- **Options:** rounds=4, enable_valu_shift=True, scheduler=greedy

| Config | Cycles | Peak liveness (words) |
|--------|--------|------------------------|
| **BEFORE** (enable_tree_reduce_depth3=False) | 465 | 1529 |
| **AFTER** (enable_tree_reduce_depth3=True)   | 460 | 1522 |
| **Delta**                                    | **-5** | **-7** |

Tree-reduce depth3 (8-way selection) saves 5 cycles and 7 words peak liveness for this configuration. The `--enable-tree-reduce-depth3` flag is exposed in `generate_ssa_kernel.py` and in the SSA Python binding (`enable_tree_reduce_depth3`).

---

## Current Status (ff9895a)

- **Git SHA:** ff9895a
- **Changes:** Reserved zero VReg without emitting op, deprecated `emit_const()`, added filtering for AddImm alternatives

### Greedy scheduler (no valu-shift)

- **Command:** `uv run python generate_ssa_kernel.py --rounds 4 --use-greedy --validate -q`
- **Cycles:** 513
- **Peak live:** 1488 words
- **Verification:** PASSED

### Known Issues

1. **SelectChoice lowering bug**: When CP-SAT (or greedy with valu-shift) chooses the FMA path for SelectChoice ops without a precomputed delta, the lowering creates invalid temporary addresses (`10000 + op_id`) that don't exist in scratch memory. This causes validation failures.

2. **CP-SAT non-determinism**: Per-section CP-SAT sometimes fails to find feasible solutions within scratch budget, and full-kernel CP-SAT often times out (10 min) without finding a solution.

| Config | Status | Cycles | Peak | Notes |
|--------|--------|--------|------|-------|
| Greedy (no valu-shift) | PASS | 513 | 1488 | Baseline working |
| Greedy (valu-shift) | FAIL | - | - | SelectChoice lowering bug |
| CP-SAT (no valu-shift) | FAIL | 491* | 1458* | *Found but fails validation |
| CP-SAT (valu-shift) | FAIL | - | - | Times out (10 min) |
