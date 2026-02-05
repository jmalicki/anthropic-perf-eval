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

**If overflow still appears:** The hint may still violate VALU/ALU in other cases, e.g. the **initial** greedy schedule (before any per-section CP-SAT) can exceed capacity within a single section. Hint verification and “CP-SAT will rebalance” remain in `scheduler_rs/src/cpsat_scheduler.rs`; only the first 3 VALU and first 3 ALU violations are printed.
