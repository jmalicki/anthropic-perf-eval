# Targeted Unit Tests for Hint-Pass / Fixed-Encoding Logic

These tests are designed to exploit **fundamental logic misconceptions** that could reappear (e.g. during refactors) and to **document why** specific conditions in the code are necessary. Each test is minimal and ties one code path to one invariant.

---

## 1. SlotTracker: why `try_place_shifted_only` exists (Bug #2)

**Misconception:** “If an op *can* shift (can_shift=true), we can use try_place(Engine::Valu, can_shift, allow_shift) and we’ll get a valid placement.”

**Reality:** try_place **prefers VALU** when there is room (it checks `valu_used < 6` first). So we can get `Placed` (on VALU) even when the **encoding** says “shifted.” The verifier then counts 8 ALU for that op; the schedule only used 1 VALU → mismatch and possible ALU overflow when several such ops are in one cycle.

**Why the condition is necessary:** When the op is in `fixed_shifted_ops`, we must **never** place it on VALU. So we use a separate path that only tries the shift: `try_place_shifted_only()`.

**Tests added (ssa_greedy/tests.rs):**

- **`test_slot_tracker_try_place_valu_prefers_valu_when_room`**  
  With empty slots, `try_place(Engine::Valu, true, true)` returns `Placed` and increments `valu_used`. So “can shift” does **not** imply “will be placed as shifted.” Therefore for fixed_shifted_ops we must not use try_place; we must use try_place_shifted_only.

- **`test_slot_tracker_try_place_shifted_only_never_uses_valu`**  
  `try_place_shifted_only()` never increases `valu_used`; it only adds 8 to `alu_used`. So the shift-only path is the one that keeps the schedule consistent with “this op is shifted” (8 ALU, 0 VALU for that op).

- **`test_slot_tracker_try_place_shifted_only_respects_alu_cap`**  
  First `try_place_shifted_only()` succeeds (8 ALU); second in the same cycle returns `NoRoom` (8+8 > 12). So at most one shifted op per cycle; the scheduler must defer extra “must shift” ops to a later cycle.

---

## 2. SelectChoice / ConstChoice: why fixed encoding must be used in re-run (Bug #1)

**Misconception:** “Re-deriving cycles with fixed ConstChoice and fixed shift is enough; SelectChoice can stay as the scheduler’s default.”

**Reality:** If we don’t pass `fixed_select_choice_encoding` into the re-run, the scheduler places SelectChoice ops as Flow (vselect). The hint (from hint passes) can say FMA for some of them. The verifier then counts those as VALU in their cycles. So we get cycles from a schedule that used Flow and encodings that say VALU → e.g. 7 VALU in one cycle.

**Why the condition is necessary:** When we re-derive cycles after hint passes, we must pass **all** authoritative encodings (const, shift, **and** select) into the greedy so the engine choice for SelectChoice matches the hint.

**Targeted test idea (not yet added):**

- The minimal regression would be: run the greedy with a small graph that has at least one SelectChoice op and `fixed_select_choice_encoding = Some(map)` with that op as FMA. Then compute per-cycle counts **using that same map** (as the verifier does) and assert VALU ≤ 6 and ALU ≤ 12. If the code path ignores `fixed_select_choice_encoding`, the scheduler might put 6 VALU + that SelectChoice as Flow in one cycle; counting that op as FMA gives 7 VALU and the assertion fails. The smallest graph that exposes this has 6 “normal” VALU ops plus 1 SelectChoice in the same cycle under the default placement; the test then asserts that with the fixed encoding the schedule has no cycle with 7 VALU. That’s a bit larger than the SlotTracker tests but still a single, focused scenario. Alternatively, the **engine selection** for SelectChoice could be extracted into a small pure function and tested in isolation (given op, fixed map, default_flow → Engine); that would be a 2–3 line test per branch.

---

## 3. Stale cycles: why we re-derive after hint passes (Bug #1 / pipeline)

**Misconception:** “We can run hint passes to update encodings and then keep using the first greedy’s op.cycle for hints.”

**Reality:** Cycles were computed for the **old** encodings. After hint passes, encodings are the new authority; cycles are stale. Merging stale cycles with new encodings can produce an invalid schedule (e.g. 7 VALU).

**Why the condition is necessary:** Whenever we update encodings (hint passes), we must re-derive cycles (ScheduleWithFixedChoices) so the single schedule we pass to CP-SAT has cycles consistent with those encodings.

**Targeted test idea:**

- A **verifier** that, given (ops, cycle_hints, encoding hints), computes per-cycle VALU/ALU/… and returns Ok or Err(cycle, engine). Then: (1) Unit test: feed it a schedule that is valid when cycles and encodings match; assert Ok. (2) Unit test: feed it the **same** cycles but flip one encoding (e.g. one SelectChoice to FMA); assert Err. That illustrates that “same cycles + different encodings” can violate capacity, so we must not mix cycles from before the update with encodings from after.

---

## Summary

| Test | Misconception | Why the condition exists |
|------|----------------|---------------------------|
| try_place prefers VALU when room | “can_shift ⇒ we’ll get shifted” | try_place can return Placed (VALU); for fixed_shifted_ops we need try_place_shifted_only |
| try_place_shifted_only never uses VALU | — | Shift-only path keeps schedule and encoding in sync (8 ALU, 0 VALU for that op) |
| try_place_shifted_only respects ALU cap | — | At most one shifted op per cycle (8+8>12); must defer the rest |
| fixed_select_choice in re-run | “fixed const + fixed shift is enough” | Otherwise SelectChoice stays Flow while hint says FMA → 7 VALU |
| Re-derive after hint passes | “we can keep first greedy’s cycles” | Stale cycles + new encodings → invalid merged schedule |

The three SlotTracker tests are implemented; they are small and directly document the logic that prevents Bug #2 (fixed_shifted_ops ⇒ place only as shifted, and respect ALU cap when doing so).
