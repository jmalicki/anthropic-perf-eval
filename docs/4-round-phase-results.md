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

## Current Status (ae4a158)

- **Git SHA:** ae4a158
- **Changes:** Fixed SelectChoice lowering bug (allocate real VReg for sub temp), reserved zero VReg

### Greedy scheduler (no valu-shift)

- **Command:** `uv run python generate_ssa_kernel.py --rounds 4 --use-greedy --validate -q`
- **Cycles:** 513
- **Peak live:** 1488 words
- **Verification:** PASSED

### Fixes applied (recent)

- **Effective scratch budget when zero reserved**: Scheduler uses 1535 words when address 0 is reserved for zero; `peak_live` includes zero_vreg so total capacity stays 1536.
- **sub_temp_vreg in segment scratch**: CP-SAT per-section scratch model now adds intervals for SelectChoice FMA `sub_temp_vreg` (def at op cycle, last use at cycle+1); hint verification includes them.
- **Filtered segment indices**: `live_in_for_section_filtered` uses indices into the *filtered* segment (AddImm alternatives removed) so CP-SAT `cycle_vars` and initial_live last-use indices align.

### Known Issues

1. **~~SelectChoice lowering bug~~** (FIXED in ae4a158): Now allocates real sub_temp_vreg during emission.

2. **VALU-shift scratch pressure**: With `--enable-valu-shift`, greedy may still force-schedule when at the limit, so peak_live can reach 1536 assignable + zero → 1537 and exceed budget. Effective budget 1535 is used; greedy does not yet count sub_temp in its budget check.

3. **CP-SAT per-section liveness model**: Per-section CP-SAT may still produce schedules that exceed scratch budget (e.g. segment end vs actual makespan, live sets by index not cycle). The sub_temp and filtered initial_live fixes reduce the gap.

| Config | Status | Cycles | Peak | Notes |
|--------|--------|--------|------|-------|
| Greedy (no valu-shift) | PASS | 513 | 1489 | Baseline working |
| Greedy (valu-shift) | FAIL | - | 1537 | May force-schedule; peak includes zero |
| CP-SAT (no valu-shift) | FAIL | - | - | Per-section liveness bug (improved) |
| CP-SAT (valu-shift) | FAIL | - | - | Times out (10 min) |

---

## CP-SAT per-section liveness

When CP-SAT is used, the flow is:

1. **Greedy with barriers** – Schedule the full kernel with barriers between rounds. Compute peak scratch per *section* (ops between barriers).
2. **Per-section CP-SAT (feasibility)** – For any section whose greedy peak exceeds 1536, run CP-SAT on that section only to find a feasible schedule that fits in the budget.
3. **Merge** – Write per-section cycle assignments back into the full `ssa_ops` (local cycle + `section_offset`).
4. **Check** – Re-run `sections_exceeding_budget(&ssa_ops, 1536)`. If any section peak > 1536, we return the error you see.

So the bug: **per-section CP-SAT says “feasible” for a section, but after merge the full-kernel peak for that section is > 1536** (e.g. section 2 peak 1544).

### How per-section scratch is modeled

For a segment (one section’s ops), CP-SAT gets:

- **Segment ops** – Only ops in that section (by *op index* in the full list; section = `boundaries[k]`).
- **initial_live** – VRegs live at segment start: defined in an earlier section, last use in *this* section. From `live_in_for_section`; each entry is `(VReg, segment_last_use_idx)`.
- **live_out** – VRegs defined in *this* section, last use in a *later* section. Must stay live until segment end.
- **live_through** – VRegs defined in an *earlier* section, last use in a *later* section. Live for the whole segment `[0, segment_end]`.

Section membership is by **op index** (which ops fall in `boundaries[k]`), not by cycle. Liveness uses **cycles** (def_cycle, last_use_cycle). So we mix “section by index” with “live range by cycle”.

The segment model:

- Puts each segment op’s result on `[cycle, cycle+size]` (or for live-out, `[cycle, segment_end]`).
- Puts live_through on `[0, segment_end+1]`.
- Adds a cumulative constraint: total scratch demand ≤ 1536 at every time in the segment.

So in theory the segment scratch model should match “this section’s cycles in global time” once we merge with `section_offset`.

### Likely causes of the bug (detailed)

---

#### 1. Segment end vs. actual makespan

**What "segment end" is.** In the per-section CP-SAT model, the segment runs in *local* cycle space: cycles 0, 1, 2, … up to some maximum. That maximum is `segment_end`: the makespan of the segment (the latest cycle at which any segment op runs). Live_through and live_out VRegs are modeled as occupying scratch for the *entire* segment: from cycle 0 to `segment_end` (or `segment_end + 1` depending on convention). So the cumulative constraint "total scratch ≤ 1536" is enforced over the range `[0, segment_end]`.

**How segment_end is chosen.** The model has a variable (or derived value) that represents the segment's makespan. Hints from the greedy schedule give a *suggested* segment_end (e.g. max of greedy cycle hints for that section). CP-SAT is free to pick a *different* makespan: it can lengthen the segment (schedule some ops later) or shorten it (schedule earlier). So after solving, the *actual* segment_end is whatever the solver chose (max of the cycle variables for segment ops).

**Why a longer makespan can break the merged peak.** When we *merge* back into the full kernel, we set `ssa_ops[global_idx].cycle = section_offset + local_cycle`. So the section's *global* cycle range is `[section_offset, section_offset + actual_segment_end]`. The next section then starts at `section_offset + actual_segment_end + 1`. If CP-SAT *lengthened* the segment (actual_segment_end > greedy_segment_end), then live_through and live_out are still "live for the whole segment" in the segment model — but we're now counting over more cycles. Two ways the full-kernel peak can exceed 1536 even though the segment model was satisfied: (a) **Overlap shape changes** — with more cycles, the *pattern* of which VRegs overlap at which cycle can change; the segment model's intervals might not perfectly match what we compute in `sections_exceeding_budget` at each instant. (b) **segment_end used for hints vs. solution** — if the model fixes or biases segment_end to the *hint* (greedy makespan) but the solution actually uses a *larger* makespan, then live_through/live_out might be modeled only until the hint makespan, while the merged schedule extends further; we'd under-count their lifetime in the segment and the full-kernel check would see more overlap.

---

#### 2. Live sets by index, not by cycle

**How "last use" is defined today.** `live_out_from_section(ops, boundaries, section_k, vreg_def_section)` answers: "Which VRegs are defined in section_k and have their *last use* in a *later* section?" It decides "last use" by **section index**, not by cycle: for each VReg we find the *highest section index* that contains an op that *reads* that VReg (by op index: op in `boundaries[s]`). That section index is "last use section." So "last use" = "last section that contains a reader," not "cycle at which the last read happens." Similarly, `live_through_for_section` uses "defined in a section before k" and "last use section > k," again with "last use" = "max section index of a reader."

**Why this is a problem after CP-SAT reorders.** Section membership is fixed by **op index**: op at index `i` belongs to section `s` iff `boundaries[s].0 <= i < boundaries[s].1`. So "which section contains the op that last uses VReg v?" is fixed by the *ordering of ops in the list*, not by when those ops are *scheduled*. After CP-SAT runs, it only moves *cycles*; op indices don't change. So the *section index* of "last use" for every VReg is unchanged. But the *cycle* at which that last use happens can change. Example: **Greedy** — Section 2 has ops A, B, C at cycles 10, 11, 12; section 3 has op D at cycle 13. So VReg v (defined in section 2) is "last used" in section 3 (by D). So v is in live_out for section 2: "live until end of segment 2." **CP-SAT** — It might schedule D *earlier*, e.g. cycle 9 (still in section 3 by index). So in *global time*, v's last use is now at cycle 9. But we still treat v as "live until end of segment 2." If segment 2's cycles are now 8..15, we're still counting v as live for 8..15, even though the true last use is at 9 — we *over*-count v's lifetime in the segment. **Opposite case** — CP-SAT moves a *reader* from section 3 into *later* cycles (e.g. cycle 20). So v's true last-use cycle is 20, but we only model segment 2 as 8..15. In the segment we treat v as live until 15, but in the full-kernel check v is live until 20. So we *under*-count v's lifetime in the segment; at cycle 20 we might have v plus other VRegs that we didn't account for in the segment, and the full-kernel peak can exceed 1536. So **"live sets by index, not cycle"** means: we commit to "last use section" at hint time (by op index). After CP-SAT, the *actual* last-use *cycle* can move. If it moves so that a VReg is live *longer* than we modeled in the segment, the full-kernel peak can exceed what the segment model allowed.

---

#### 3. ~~sub_temp_vreg not in round_budget~~ (FIXED)

**What sub_temp_vreg is.** For a SelectChoice op that uses the FMA path *without* a precomputed delta, we emit two conceptual steps: (1) `sub_temp = true_val - false_val`, (2) `dest = cond * sub_temp + false_val`. The temporary `sub_temp` is a real VReg (`sub_temp_vreg`) allocated during SSA emission.

**Fix.** `sections_exceeding_budget` now accepts optional `select_fma_op_ids`; when provided, for each SelectChoice op in that set with `sub_temp_vreg`, the sub_temp is counted (def at op.cycle, last use at op.cycle+1). The CP-SAT segment scratch model also adds intervals for these sub_temps when `select_choice_hints` is provided. So section peaks and segment feasibility now include sub_temp when the FMA path is chosen.

---

#### 4. Filtering and indices

Before per-section CP-SAT we filter out AddImm alternatives from the *segment* ops (same logic as full-kernel CP-SAT: remove the AddImm op when a Const op with the same dest exists). Segment hints are then aligned to the *filtered* segment (one hint per filtered op, in order). If any indexing or hint mapping is off — e.g. `segment_last_use_idx` in `initial_live` refers to segment op indices, and the segment is filtered — then the scratch model's notion of "which op last uses this VReg" or "which cycle range an initial_live VReg occupies" could be wrong. That could make the segment model optimistic (under-count overlap) or pessimistic; if optimistic, the merged schedule can exceed 1536. The exact failure mode depends on how `live_in_for_section` and the CP-SAT segment code map segment indices to ops and cycles; any mismatch between filtered segment order and the full-kernel op list can cause inconsistency.


### Where to look in code

- **Section boundaries and peak:** `scheduler_rs/src/kernel_builder/round_budget.rs`: `sections_exceeding_budget`, `live_in_for_section`, `live_out_from_section`, `live_through_for_section`.
- **Segment CP-SAT scratch model:** `scheduler_rs/src/cpsat_scheduler.rs`: segment mode with `initial_live`, `live_through_vregs`, `live_out_vregs`, `segment_end_var`, and cumulative constraint.
- **Merge and check:** `scheduler_rs/src/kernel_builder/tick_tock_ssa.rs`: after the per-section loop, `section_offset` update and `sections_exceeding_budget(&ssa_ops, 1536)` that triggers “section N has peak P words (budget 1536).”
