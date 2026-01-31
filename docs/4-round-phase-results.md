# 4-round optimization phase results

Record after each phase: git SHA, exact uv command, liveness peak, cycles, and confirmation that Python kernel verification passes with 4 rounds.

---

## Phase 1

- **Git SHA:** 2a32f71
- **Exact uv command:** `uv run generate_ssa_kernel.py --rounds 4 --use-greedy --validate -q`
- **Liveness peak:** 1489 words
- **Cycles:** 514
- **Verification:** PASSED — `uv run generate_ssa_kernel.py --rounds 4 --use-greedy --validate -q` exited successfully; kernel produces correct output (Python validation against reference kernel with 4 rounds).
