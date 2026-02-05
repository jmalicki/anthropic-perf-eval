# Known bugs and limitations

## Open

(none at the moment)

## Fixed

- [x] **CP-SAT loose makespan** – Makespan domain was forced to `[capacity_lower_bound, horizon]` with `capacity_lower_bound` inflated by `critical_path_len` (max ASAP + 1). That overestimates because the model uses variable-length steps (completion_cycle for shifted, +2 for SelectChoice FMA). Solver found 609 with 416 empty cycles; compaction reduced to 192. **Fix:** Pass 0 for the dependency floor when calling `compute_capacity_lower_bound_global` in the SSA path so `capacity_lower_bound` is the actual capacity-derived lower bound (min over x,y,z only). Precedence constraints in the model enforce any dependency lower bound. `scheduler_rs/src/cpsat_scheduler.rs`.

## Fixed (historical)

- [x] **Greedy / merged hint violates engine capacity (VALU, ALU)** – Fix: rebase all sections in `tick_tock_ssa.rs` so merged hint is strict concatenation. See below for details.

**Symptom (historical):** When building the CP-SAT model with greedy hints, you see warnings like:

```
  WARNING: VALU overflow at cycle 175: 7 > 6
  WARNING: ALU overflow at cycle 178: 16 > 12
  ...
  Note: Merged hint has engine capacity issues (VALU: N, ALU: M). CP-SAT will rebalance.
```

**Intent:** Sections are meant to be **concatenated** in time: section 0 at `[0, L0)`, section 1 at `[L0, L0+L1)`, section 2 at `[L0+L1, L0+L1+L2)`, etc. So the merged hint should never have two sections using the same cycle — we bump each section’s cycles by the sum of previous sections’ makespans.

**Cause (fixed):** Only sections that exceeded the scratch budget ran per-section CP-SAT and got their cycles written back as `section_offset + local_cycle`. Sections that stayed within budget were never rebased; they kept the initial greedy schedule’s cycles, which overlap with other sections (greedy schedules the whole kernel in one space). So the “merged” hint was a mix of concatenated sections and overlapping greedy cycles.

**Fix:** In `scheduler_rs/src/kernel_builder/tick_tock_ssa.rs`, after the per-section loop we now **rebase** every section that did *not* run per-section CP-SAT: for each such section, set `op.cycle = section_offset + (op.cycle - min_c)` so that section occupies `[section_offset, section_offset + span)`. Then advance `section_offset` by that span. The merged schedule is thus a strict concatenation of sections.

**Verification:** We are allowed to exceed scratch in the greedy scheduler, but not slot constraints. If the merged hint ever has VALU/ALU capacity violations, `scheduler_rs/src/cpsat_scheduler.rs` panics (merge bug or single-section schedule exceeding capacity).

**Merged hint (cycles + choices):** The final full-kernel CP-SAT receives a single merged hint: for each section, either the greedy solution or the per-section CP-SAT solution. We maintain `current_select_fma` and `current_const_add_imm` (and cycles on `ssa_ops`): before running per-section CP-SAT for a section we pass the current merged choices for that section’s ops into the segment; after it runs we update cycles and those choice sets with the segment result. So the combination of greedy-or-CP-SAT per section is exactly what the final model is hinted with. See `scheduler_rs/src/kernel_builder/tick_tock_ssa.rs` (section loop and “Merged hint state” comment). Post-merge scratch peak uses `sections_exceeding_budget(..., Some(&section_select_fma_overrides))` so each section's peak reflects that section's FMA/vselect choices (greedy or per-section CP-SAT); see `round_budget.rs`.
