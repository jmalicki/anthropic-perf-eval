# Design: VStore / Offset Scratch Collision Bug

## 1. Invariant we need

**Correctness:** Two distinct SSA values (VRegs) that are both *live* at some cycle must not be assigned the same scratch address.

- **Value VReg**: produced by last hash op (e.g. xor), read by the partition **vstore** (store group value to buffer).
- **Offset VReg**: produced by `Sub(v_2, bit)`, read by the **multiply_add** that updates idx.

If both get the same address, then either:
- the vstore overwrites the offset before the FMA reads it, or
- the Sub overwrites the value before the vstore reads it.

In the observed bug: value and offset both got address 10 → Sub at T wrote offset to 10, vstore at T+1 read from 10 (got offset instead of hash), FMA at T+2 read from 10 (got offset). Symptom: “hash used as offset”, wrong indices.

So the **design invariant** is: **liveness intervals used by the allocator must be conservative** — any VReg that is *read* at cycle C must have its interval include C. Then the linear-scan allocator will not reuse that VReg’s address for another VReg that is live at C.

---

## 2. Root cause (two layers)

### Layer A: Scheduler (cycle assignment)

We have RAW precedence: `cycle[consumer] >= cycle[producer] + 1`. So in a correct schedule, the vstore (consumer of the value) must be at least one cycle after the xor (producer).

If the solver or merge ever produced a schedule where **vstore cycle < producer cycle**, that would be a **scheduler bug**. Then when we build liveness from that schedule, the value’s “last read” would be at the vstore’s (wrong) cycle, and we’d get a short or invalid interval.

### Layer B: Allocator (liveness + linear scan)

Liveness is built from the **same** schedule we’re allocating for:

1. **Definition:** each VReg gets `[def_cycle, def_cycle]`.
2. **Extension:** for every op that **reads** a VReg, we set `end = max(end, op.cycle)`.

So *if* the schedule is correct (vstore >= producer + 1), the value’s interval should already extend to the vstore’s cycle. The only way the value’s interval ends *before* the vstore’s cycle is if:

- the schedule is wrong (Layer A), or
- we never extend the value when we process the vstore (e.g. vstore not in `ops`, wrong `srcs`, or bug in the extension loop) (Layer B).

The allocator itself (linear scan, expire when `end <= start`) is standard and correct. The bug is **input** to the allocator: an interval that is too short or invalid.

---

## 3. Design options

### Option 1: Fix the scheduler only

- **Idea:** Ensure we never emit a schedule where `cycle[vstore] < cycle[producer] + 1`. Add precedence constraints and/or post-solve checks; reject or repair bad solutions.
- **Pros:** Fixes root cause at the source; no need to “paper over” in the allocator.
- **Cons:** We don’t yet know *why* the bad schedule appeared (wrong dep_ids? wrong model? merge bug?). Until we know, we can’t be sure we’ve fixed it everywhere. Also, if we only *detect* and reject, we might turn previously “working” runs into failures until the solver is fixed.

### Option 2: Defensive liveness in the allocator only

- **Idea:** Make liveness conservative so that even with a wrong schedule we don’t assign the same address to two values that *should* be live at the same time. Concretely:
  - **Clamp:** Never allow `interval.end < interval.start` (treat as `[start, start]` then let dead-VReg pass extend to `[start, start+1]`).
  - **Vstore guarantee:** For every vstore, explicitly extend the *value* operand (e.g. `srcs[1]`) to the vstore’s cycle (and clamp).
- **Pros:** No dependency on proving/fixing the scheduler; fixes the observed bug; small, localized changes in address_assign.
- **Cons:** If the schedule is wrong, we’re “hiding” it: the value’s interval becomes longer and we use more scratch, but we don’t fix the underlying precedence. So we might want Option 1 as well later.

### Option 3: Both (scheduler + allocator)

- **Idea:** Add scheduler-side precedence checks (and fixes if we find the cause) *and* keep the allocator defensive (clamp + vstore pass).
- **Pros:** Correct schedules and robust allocation; defense in depth.
- **Cons:** More work; we haven’t fully root-caused the scheduler yet.

---

## 4. Recommended “right fix” before changing more

**Minimal design we can agree on:**

1. **Allocator invariant (we own this):**  
   Liveness intervals must never have `end < start`, and the *stored value* of every vstore must be considered live through the vstore’s cycle.  
   So:
   - **Clamp:** Whenever we set `interval.end`, enforce `interval.end >= interval.start` (e.g. if we would get `end < start`, set `end = start`; the existing dead-VReg pass then extends `[s,s]` to `[s,s+1]`).
   - **Vstore pass:** For each vstore, extend the value VReg (e.g. `srcs[1]`) to `op.cycle` and clamp. This guarantees the value is live at the vstore even if the main extension pass missed it or the schedule was wrong.

2. **Scheduler (investigate later):**  
   Treat “why did we get a schedule where value’s last read was before the vstore?” as a separate task: add logging or post-solve checks (e.g. for every vstore, `cycle[vstore] >= cycle[dep] + 1`), and fix the model or merge if we find a violation. That’s independent of the allocator fix.

So the **right fix** for the allocator is: **clamp + explicit vstore value pass**. What we’ve already added matches that. We can either:

- **Keep it as-is** and document it (this doc + BUGS.md), or
- **Simplify:** e.g. keep clamp everywhere we extend; keep one explicit vstore pass; remove any redundant second clamp in the vstore pass if the first clamp already guarantees `end >= start`.

---

## 5. What’s in the tree right now

- **address_assign.rs**
  - In the main “extend reads” loop: after `interval.end = op.cycle`, we clamp `if interval.end < interval.start { interval.end = interval.start }`.
  - A separate “Vstore value pass” that extends `srcs[1]` to the vstore’s cycle and applies the same clamp.

So we have both parts of the recommended allocator fix. No scheduler-side changes yet (no post-solve vstore precedence check in the tree).

---

## 6. Open decisions

1. **Keep vs simplify:** Is the explicit vstore pass redundant given we already extend all `op.srcs` in the main loop?  
   - It’s redundant *when the schedule is correct and the main loop sees the vstore*. It’s **not** redundant when the schedule is wrong (value’s interval might already be invalid before we process the vstore) or when there’s a bug in how we iterate (e.g. op order). So keeping the explicit pass is a **defensive** choice; we can keep it for clarity and safety.

2. **Scheduler follow-up:** Add a post-solve check that for every vstore, `cycle[vstore] >= cycle[producer] + 1`, and panic or log in debug? That would confirm whether the bad schedule came from the solver/merge and guide a future scheduler fix.

3. **Revert anything?** If we want the *minimal* change, we could try only the clamp (no vstore pass) and see if that alone fixes the 4r/256 case; if yes, we might drop the vstore pass. If no, we need the pass. I’d keep both unless we explicitly decide to minimize.

Once we agree on this design (invariant, cause, and “right fix”: clamp + vstore pass, with optional scheduler checks later), we can lock it in and only then consider further code changes or simplifications.
