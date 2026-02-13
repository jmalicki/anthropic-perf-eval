# 2r/16b Zero-Gap vs Default Kernel: Validation Failure Findings

**Running doc** of the comparison between the kernel that **passed** validation (default CP-SAT, 1h) and the one that **failed** (zero-gap, 10 min limit). Same problem: 2 rounds, batch 16. Goal: locate the bug.

---

## Summary

| Artifact | Validation | Notes |
|----------|------------|--------|
| `artifacts/kernel_2r_16b.json` | **PASS** | Default run, no zero-gap |
| `artifacts/scheduler_2r_16b.json` | — | 368 cycles, **valu_shifted: 0** |
| `artifacts/kernel_2r_16b_zero_gap.json` | **FAIL** | Zero-gap run (10 min) |
| `artifacts/scheduler_2r_16b_zero_gap.json` | — | 368 cycles, **valu_shifted: 16** |

**Failure mode:** Value mismatch at batch index 0; index mismatch at batch index 1.

A related **live-interval bug** appeared when slot tightening was disabled; that bug is **fixed** (see below).

---

## What's fixed / Current state

### Fixed (this work)

| Item | What we did |
|------|-------------|
| **Live-interval bug (use-def violation)** | SelectChoice FMA path in `use_cycles()` was skipping **true_val**. The sub at `op.cycle + 1` reads true_val; we now report all srcs (cond, true_val, false_val) at `op.cycle + 1` so intervals extend correctly. **Location:** `scheduler_rs/src/address_assign.rs` (FMA branch of `LivenessUse::use_cycles`). |
| **Use-def check at allocation time** | The check in `assign_addresses` now validates against the same source of truth as interval building: for each op we use **`use_cycles(op)`** and require each (vreg, read_cycle) to lie inside that VReg's interval. So we check the actual read cycles (e.g. op.cycle+1 for FMA), not just op.cycle. Missing a read in `use_cycles` now causes allocation to return an error. |
| **Docs and recurrence** | Module table in `address_assign.rs` corrected to list cond, **true_val**, false_val for SelectChoice FMA reads. `LivenessUse` trait doc states the contract: report every read at the exact cycle. Findings doc (§5b) explains why tests missed the bug and how we prevent recurrence. |
| **Regression test** | `test_compute_live_intervals_selectchoice_fma_dest_at_cycle_plus_1` now asserts that cond, true_val, and false_val all have interval end ≥ `select_cycle + 1`. |
| **BUGS.md** | Entry "Use-def violation when ASAP/ALAP tightening disabled" marked **FIXED** with cause/fix and pointer to this doc. |

### Current state

| Scenario | Result |
|----------|--------|
| **2r/16b, default (1h CP-SAT, slot tightening on)** | **PASS** — kernel validates. |
| **2r/16b, zero-gap (10 min), slot tightening on** | **FAIL** — kernel fails validation (value/index mismatch). Cause is separate from the live-interval bug: zero-gap shifts 16 VALU ops, default shifts 0; kernels diverge from cycle 1. Not yet root-caused. |
| **2r/16b, zero-gap + no slot tightening** | **PASS** — previously failed during build with use-def violation; after the fix, build completes and validation **passes** (36-cycle kernel). |
| **CP-SAT toy test** (`test_ssa_cpsat_selectchoice_valu_shift_toy`) | **FAIL** — known issue: CP-SAT presolve INFEASIBLE with default config (ASAP/ALAP tightening + optional intervals). See BUGS.md; unrelated to the live-interval fix. |

**Summary:** The **live-interval / use-def bug** is fixed. Zero-gap with **no** tightening now works. The **original** 2r/16b zero-gap validation failure (with tightening, 10 min limit) is a different bug (VALU-shift / schedule divergence) and remains open.

**Did our fixes introduce the failures?** No. The 2r/16b zero-gap validation failure and the CP-SAT toy presolve INFEASIBLE were both present before the live-interval/use-def changes. Our changes only affect interval computation and the use-def check in address assignment; they do not affect CP-SAT presolve or the zero-gap schedule/validation path.

---

## 1. Scheduler output differences

### 1.1 Top-level

| Field | OK (default) | Zero-gap |
|-------|-------------|----------|
| `total_cycles` | 368 | 368 |
| `total_ops` (in JSON) | 134 | 244 |
| `valu_shifted` | **0** | **16** |
| `max_addr` | 185 | 189 |

So the zero-gap run chose to **shift 16 VALU ops to ALU**; the default run shifted none. That is the biggest structural difference.

### 1.2 Cycle assignments

- **44 op indices** have different cycle assignments between the two scheduler outputs.
- Examples (op index → cycle_ok, cycle_zerogap):
  - 5: 332 → 1
  - 24: 345 → 1
  - 41: 333 → 2
  - 45: 6 → 2
  - 46: 333 → 3

So zero-gap moves many ops **much earlier** (e.g. from 332–345 down to 1–3). Those ops in the OK run sit in the “tail” cluster; in zero-gap they are scheduled in the first few cycles.

### 1.3 Shifted-ALU (zero-gap only)

Zero-gap scheduler output contains `shifted_alu_cycles`: 16 op_ids, each with 8 ALU cycles (one per lane). Examples:

- op_id 85: all 9
- op_id 91: all 10
- op_id 97: all 12
- op_id 133: all 355
- op_id 136: all 356

So some shifted ops are in the early cluster (9–20) and some in the late cluster (355–356).

### 1.4 ConstChoice / SelectChoice

- Scheduler output JSON does not expose `const_add_imm_ops` / `select_fma_ops` as top-level keys (they live in the Rust result; may be in a nested structure if exported). **Hypothesis:** Choice sets may be the same; the main difference is **cycle assignments + VALU shift**, which then changes lowering (scratch addresses, bundle ordering).

### 1.5 schedule_constraints

- Length same (354) but content differs between the two JSONs (different cycle domains or precedence after repair).

---

## 2. Emitted kernel differences

Bundles **diverge from cycle 1**.

### 2.1 Cycle 1

- **OK:** `add_imm(3,0,1)`, `const(2,2)`, `const(1,4)`
- **Zero-gap:** `add_imm(1,0,5)`, `const(3,1)`, `const(2,4)`

Different destinations and constants: different **assignment of SSA values to scratch** and/or different **order of ConstChoice** (which constant is emitted first and to which dest).

### 2.2 Cycle 2

- **OK:** add_imm, load, const, vbroadcast (no ALU).
- **Zero-gap:** ALU `<<`, `+`, `+` (look like shifted VALU), flow add_imm, load, vbroadcast — i.e. **shifted ops already in cycle 2** with different load/vbroadcast dests (e.g. vbroadcast 8,3 vs 6,2).

So from cycle 2 we have a different schedule (what runs where) and different scratch/dest numbering.

### 2.3 Later cycles (6–21)

- Zero-gap introduces **ALU xor/add/shift** (element-wise, 8 lanes) that correspond to shifted VALU; OK keeps the same work in VALU.
- Different **vbroadcast** and **multiply_add** placement and operands.
- By cycle 10–12 zero-gap has many ALU `^` and `+` (element-wise); OK has the same logic in VALU.

So the **semantic content** is “same algorithm, different placement (VALU vs ALU) and different cycle/scratch layout”. The bug is likely in how **shifted ops** are lowered (completion cycle, liveness, or ordering) or how **cycle reordering** interacts with ConstChoice/address assignment.

---

## 3. Hypotheses (to refine)

1. **VALU-shift lowering bug**  
   When 16 VALU ops are shifted to ALU, `shifted_alu_cycles` (per-lane ALU cycles) or the use of “completion” cycle for liveness/precedence might be wrong, leading to:
   - wrong scratch reuse (e.g. overwrite before read), or
   - wrong instruction ordering in the bundle.

2. **Cycle reordering + address assignment**  
   Zero-gap moves many ops from the tail into the first ~20 cycles. If address assignment or lowering assumes something about “when” an op runs (e.g. for liveness or for ordering), that could produce wrong dest/scratch or wrong bundle order.

3. **ConstChoice emission order**  
   If constants are emitted in an order that depends on schedule (e.g. cycle order of ConstChoice ops), then the different cycle assignment could change which constant lands in which scratch slot and thus wrong values at batch 0.

4. **Same-cycle or boundary bugs**  
   BUGS.md already mentions “Same-cycle scratch write conflict (shifted VALU adjacent intervals)”. Zero-gap has 16 shifted ops; if any of them get wrong intervals or same-cycle reuse, we could see wrong values/indices.

5. **Slot ASAP/ALAP tightening creating precedence violations**  
   The pipeline runs slot-aware tightening (and optionally scratch tightening) *before* a precedence fixpoint. That can leave producer ASAP later than consumer ASAP (e.g. producer 348, consumer 4), which the diagnostic reports as "24 edge violations". The fixpoint repair then pushes consumer ASAP to 349+, which forces the whole tail into the last decile and creates the empty middle. If the same precedence-inconsistent bounds or the order of operations (widen for greedy hint after repair) somehow affects which solution CP-SAT finds—e.g. different VALU-shift or cycle choices—that could change the emitted kernel and expose a lowering bug. So slot tightening could be *indirectly* related to the validation failure (different schedule → different shift decisions → bug in shift path).

---

## 4. Proposed fix (slot/precedence pipeline)

If the issue is related to slot ASAP/ALAP tightening and precedence:

- **Run the precedence fixpoint immediately after slot (and scratch) tightening**, before setting `final_asap` / `final_alap` and before the diagnostic. That way:
  1. Bounds are always precedence-consistent before we build the CP-SAT model and before we log "Precedence diagnostics".
  2. We avoid the confusing "24 edge violations" then "Fixpoint repair: 62 ASAP changes, 119 ALAP changes" and the tail being pushed to the end as a side effect of that repair.
- **Order in code:**  
  `slot tightening → scratch tightening (if enabled) → precedence fixpoint → final_asap/final_alap → diagnose (should be 0 violations) → greedy hint domain widening → build model`.

No change to the model itself; only the order of operations so that we never expose precedence-inconsistent bounds to the rest of the pipeline. The empty middle might still occur if the fixpoint legitimately pushes consumer ASAP to 349+ because of a late producer, but we would no longer be *creating* those violations by tightening without a fixpoint.

---

## 5a. Experiment: zero-gap with slot tightening disabled

**Command:** `uv run generate_ssa_kernel.py --rounds 2 --batch-size 16 --force-zero-gap --no-enable-asap-alap-tightening --cpsat-time-limit 300 --validate -q`

**Outcome:** Build failed before validation with a **use-def violation** in lowering/address assignment:

- **Error:** "VReg 126 read at cycle 20 but interval is [8, 8] (op id=127, opcode=SelectChoice). The VReg's live interval does not include the read cycle - this indicates a bug in interval computation."
- **Context:** With no ASAP/ALAP tightening, cycle domains are wide (e.g. [0,183] instead of tight tail [348,367]). The solver found a **much shorter** schedule (objective 3601726 → ~36 cycles). That schedule places SelectChoice op 127 at cycle 20, but the live interval for VReg 126 (read by that op) was computed as [8, 8], which does not include 20.
- **Implication:** (1) We cannot test "no tightening → validation pass" until this interval bug is fixed. (2) Live-interval computation (for SelectChoice or for consumers of shifted ops) may be wrong when the schedule has ops at cycles far from the interval currently stored. (3) Schedule was exported to `scheduler_output_error.json` for debugging.

---

## 5b. Live-interval bug (use-def violation)

This section describes the bug exposed when running zero-gap with slot tightening disabled.

### What are live intervals?

For scratch (address) assignment, the scheduler computes a **live interval** for each VReg (virtual register): the range of cycles `[def_cycle, last_use_cycle]` during which that VReg must be kept in scratch. The allocator uses these intervals to decide when a scratch address can be reused. A **use-def** check ensures that every read of a VReg happens at a cycle inside that VReg’s live interval; otherwise the program would read a value before it is defined or after it has been “dead” (wrong or overwritten).

**How intervals are built (two passes).** In `address_assign::compute_live_intervals`: (1) **First pass:** for each op that writes a VReg (`dest`), record that VReg's definition cycle (and for special cases like SelectChoice FMA or shifted VALU, the write interval). (2) **Second pass:** for each op, ask "which VRegs does this op read, and at which cycle?" via `use_cycles(op)`; for each `(vreg, read_cycle)` returned, extend that VReg's interval so `last_use_cycle >= read_cycle`. So **every read must be reported** in `use_cycles`; if a consumer op does not report a read, that VReg's interval is never extended and the use-def check can fail.

### What went wrong

- **VReg 126** (vector width) is **read** by **op 127** (SelectChoice) at **cycle 20**.
- The stored **live interval** for VReg 126 is **[8, 8]** (single cycle).
- So the interval does **not** include the read cycle 20 → use-def violation: “read at cycle 20 but interval is [8, 8]”.

So either:

1. **Interval too short:** The interval for VReg 126 should have been extended so that `last_use_cycle >= 20` (at least `[def_cycle, 20]` or wider). The code that extends liveness to consumers (e.g. “last use = max of all consumer cycles”) did not account for op 127 at cycle 20, or used a wrong cycle for op 127.
2. **Wrong cycle for the read:** Op 127’s cycle was recorded as something other than 20 when building intervals (e.g. 8), so the interval was built as [8, 8] and later the same op is interpreted as reading at 20. That would point to inconsistent cycle information between interval computation and the use-def check.

### Likely cause

- **SelectChoice** can be scheduled at a cycle that comes from the CP-SAT solution. With **wide domains** (no tightening), the solver can place SelectChoice at 20 while the **producer** of VReg 126 might be at cycle 8. The code that computes “last use” for VReg 126 may be using the **producer’s** cycle or an old/greedy cycle instead of the **consumer’s** (op 127’s) cycle from the actual schedule.
- Alternatively, SelectChoice has special handling (e.g. FMA path vs vselect path, or optional intervals), and the interval for an **input** VReg to SelectChoice might be computed from one path only or from a wrong cycle variable.

### Where to look

- **Live-interval computation:** Likely in `scheduler_rs` (e.g. `address_assign.rs`, or a shared bounds/liveness module). Search for where `last_use` or “live interval” is set per VReg and where it is extended to include consumer ops’ cycles.
- **Use-def check:** The error is raised when verifying that a read at cycle 20 lies inside the VReg’s interval; the code path that **builds** the interval for VReg 126 (and any SelectChoice inputs) is what needs to use the **actual** cycle of op 127 from the CP-SAT result.
- **SelectChoice and cycles:** Confirm that op 127’s cycle in the schedule (20) is the one used when extending liveness for VReg 126. If interval computation uses a different source (e.g. greedy hint, or a different op id), that would explain [8, 8] vs read at 20.

### Repro and artifacts

- **Repro:** `uv run generate_ssa_kernel.py --rounds 2 --batch-size 16 --force-zero-gap --no-enable-asap-alap-tightening --cpsat-time-limit 300 -q` (pre-fix: failed during build; post-fix: completes and validates).
- **Exported schedule:** `scheduler_output_error.json` (when the error is hit, the pipeline exports the schedule for debugging).
- **BUGS.md:** Entry “Use-def violation when ASAP/ALAP tightening disabled (2r/16b zero-gap)” with pointer to this doc.

### Fix (applied)

**Root cause:** For SelectChoice ops that chose the **FMA** path, `use_cycles()` in `address_assign.rs` skipped **true_val** (srcs index 1) when reporting read cycles. In the lowered FMA path, **only the sub** reads true_val: the code emits `sub = true_val - false_val` at `op.cycle + 1`; the FMA instruction itself then uses cond, the sub result, and false_val—it does **not** read true_val again. The original comment said to skip true_val because "true_val is only used by sub in sub+fma path, not by FMA." That was misleading: the **sub is part of the same SelectChoice op's lowering**, and it runs at cycle+1 and reads true_val there. So we must extend true_val's live interval to at least `op.cycle + 1`. By skipping true_val in `use_cycles`, we never extended its interval; it stayed at the producer's cycle (e.g. [8, 8]), and the use-def check failed when it saw op 127 reading VReg 126 at cycle 20.

**Change:** In `scheduler_rs/src/address_assign.rs`, in the SelectChoice FMA branch of `LivenessUse::use_cycles`, include **all** src VRegs (cond, true_val, false_val) at `read_at = op.cycle + 1`, instead of skipping index 1. That extends true_val's interval so it covers the cycle where the sub runs and the use-def check passes. After the fix, zero-gap + no-tightening completes and validation **passes** (36-cycle kernel).

**Not a section/live-in/live-out bug:** Address assignment runs **once** on the **full** ops list (`assign_addresses(&ctx.ops, ...)`). It does not use section boundaries or per-round partitioning. `compute_live_intervals` iterates over all ops and extends each VReg's interval to the latest read cycle reported by `use_cycles`; there is no separate live-in/live-out pass in address assignment. The **live_out** / **live_in** logic in `round_budget.rs` and the CP-SAT model is used only for **per-section CP-SAT** (when re-solving a section that exceeded scratch budget), so the segment scratch constraint correctly accounts for VRegs that span section boundaries. That path never touches `address_assign::compute_live_intervals`. So the bug was purely the missing true_val in the FMA branch of `use_cycles`, not an issue between round1 and round2 or live-in/live-out handling.

### Why did unit tests miss this?

1. **No assertion on input intervals:** The interval tests for SelectChoice FMA (e.g. `test_compute_live_intervals_selectchoice_fma_dest_at_cycle_plus_1`) only assert on **dest** and **sub_temp** (def_cycle and interval). They never assert that **cond**, **true_val**, or **false_val** have intervals extending to `op.cycle + 1`. So a bug that left true_val's interval short was invisible to those tests.

2. **Use-def only fires when read cycle is outside interval:** The use-def check in `assign_addresses` uses `op.reads()` (all src VRegs) and requires each read's interval to contain **op.cycle**. With the bug, true_val's interval stayed at the producer's cycle (e.g. [8, 8]). The check fails when that interval does **not** contain the op's cycle (e.g. SelectChoice at 20 → 20 not in [8, 8]). So the bug only appears when SelectChoice is scheduled **far from** the producer. In existing unit tests, SelectChoice is always at cycle 0 or a small cycle, with producers also at 0 or nearby, so true_val's interval [0, 0] or [0, 5] still contained the op cycle and the use-def check passed.

3. **Wide domains exposed it:** Disabling ASAP/ALAP tightening gave the solver wide domains, so it produced a much shorter schedule with SelectChoice at cycle 20 and producers at cycle 8. That "spread" made the missing extension of true_val's interval visible.

**Takeaway:** Add a test that either (a) schedules SelectChoice FMA at a cycle far from the producer and runs `assign_addresses`, or (b) explicitly asserts that **true_val** (and cond, false_val) have interval end ≥ `select_cycle + 1` after `compute_live_intervals`, so that omitting any of them from `use_cycles` would fail the test.

### Preventing recurrence (docs and code)

- **Contract for `use_cycles`:** Implementations must report **every** VReg that this op (or its lowering) reads, at the **exact cycle** where that read happens. For ops that lower to multiple instructions (e.g. SelectChoice FMA → sub at cycle+1, then FMA at cycle+1), every read by any of those instructions must be included. Missing a read leaves that VReg's interval too short and causes use-def failures when the op is scheduled far from the producer.
- **Code comments:** In `address_assign.rs`, the module-level table and the `LivenessUse::use_cycles` implementation now state that SelectChoice FMA must extend **cond, true_val, false_val** (and delta, sub_temp) to `op.cycle + 1`, and why (sub reads true_val at cycle+1). The `LivenessUse` trait doc was updated to require reporting every read at the correct cycle so future op lowerings (or new opcodes) don't repeat the mistake.
- **Table fix:** The "SelectChoice FMA reads" row in the module table was wrong (it listed "cond, false_val, delta" and omitted true_val); it is now corrected to "cond, true_val, false_val, delta" so the code and docs stay in sync.
- **Regression test:** `test_compute_live_intervals_selectchoice_fma_dest_at_cycle_plus_1` now asserts that cond, true_val, and false_val all have interval end ≥ `select_cycle + 1`. Omitting true_val from `use_cycles` would fail this test.

---

## 5. Next steps (suggested)

- [ ] Confirm `const_add_imm_ops` and `select_fma_ops` are identical between the two scheduler JSONs.
- [ ] Trace one ConstChoice op that appears in cycle 1 in both: same op_id? same constant value? different dest in emitted kernel?
- [ ] For one shifted op (e.g. op_id 85, cycle 9 in zero-gap): check in zero-gap kernel that the emitted ALU op at cycle 9 uses the correct source addresses and that no consumer reads before the op’s “completion” (e.g. if completion is cycle 10 for a lane, no consumer at 9 should read that lane).
- [ ] Run validation with `--no-valu-shift` and zero-gap (if supported) to see if zero-gap alone (same shifts as default) still fails; if it passes, the bug is in the VALU-shift path.
- [x] Add to BUGS.md: "2r/16b zero-gap (10 min) kernel fails validation; zero-gap run shifts 16 VALU ops, default shifts 0; kernels diverge from cycle 1."
- [x] **Zero-gap with slot tightening disabled:** Run 2r/16b with `--force-zero-gap --no-enable-asap-alap-tightening --cpsat-time-limit 300` and validate. **Result:** Did not reach validation. Solver found a much shorter schedule (objective 3601726 → ~36 cycles) because domains are wide without tightening. Lowering then failed with **use-def violation:** "VReg 126 read at cycle 20 but interval is [8, 8] (op id=127, opcode=SelectChoice). The VReg's live interval does not include the read cycle." So disabling slot tightening (1) changes the solution radically (36 vs 368 cycles) and (2) exposes a bug in live-interval computation when the schedule has SelectChoice at a cycle not covered by the VReg's interval. Conclusion: we cannot test "zero-gap + no tightening → validation pass" until this interval bug is fixed; the interval bug may be related to shifted ops or SelectChoice and wide domains.

---

## 6. Artifacts and commands

- Passed: `artifacts/kernel_2r_16b.json`, `artifacts/scheduler_2r_16b.json`
- Failed: `artifacts/kernel_2r_16b_zero_gap.json`, `artifacts/scheduler_2r_16b_zero_gap.json`
- Validate only:  
  `uv run generate_ssa_kernel.py --validate-only artifacts/kernel_2r_16b_zero_gap.json --rounds 2 --batch-size 16`
- Decile / binding:  
  `uv run scripts/slot_usage_by_decile.py --markdown artifacts/scheduler_2r_16b_zero_gap.json`  
  `uv run python -m scripts.inspect_binding_constraints artifacts/scheduler_2r_16b_zero_gap.json`
- **Zero-gap with no slot tightening (experiment):**  
  `uv run generate_ssa_kernel.py --rounds 2 --batch-size 16 --force-zero-gap --no-enable-asap-alap-tightening --cpsat-time-limit 300 --output-json artifacts/scheduler_2r_16b_zero_gap_no_tightening.json --save-kernel artifacts/kernel_2r_16b_zero_gap_no_tightening.json --validate`

---

*Last updated: 2026-02-13 (What's fixed / Current state summary; §5b: why tests missed it, preventing recurrence; use-def check uses use_cycles).*
