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
