# ALAP pull initial value: correct fix vs masking

## Summary

- **The c70ca95 fix is correct:** we must not set ALAP to a value we did not derive from a valid cycle. Using `old_alap` when `query_last` returns `None` is the right behavior; using `old_asap - 1` was a bug.
- **Reverting the fix is masking:** it hides that bug and can hide other issues (invalid domains, or different iteration behavior). It does **not** fix over-tightening in a principled way.
- **Validation failure with the fix** points to a bug elsewhere (model, solver choice, or post-schedule pipeline), not to the ALAP init being wrong. We should keep the fix and find that root cause.

---

## 1. What the ALAP pull loop does

In `tighten_asap_alap_by_slots` (energetic ALAP pulls), for each op we want to pull ALAP backward only when we have a **valid** cycle: the last cycle `t` in `[old_asap, old_alap]` where placing this op at `t` would still be within capacity (`total_min_demand(t) + 1 <= cap`). That is exactly what `seg_tree_alap.query_last(start, end, cap_minus_one)` returns.

- If the query returns **`Some(t)`**: we can safely set `new_alap_val = t` (and then apply greedy-cap / bounded-pull).
- If the query returns **`None`**: there is no such cycle in the window, so we must **not** tighten; we should leave ALAP as `old_alap`.

So the correct initial value for `new_alap_val` before the query is: “only update when we find a valid `t`.” That implies we should **not** use a value that we would write when the query returns `None`.

---

## 2. Why `old_asap - 1` is wrong

Original code:

```rust
let mut new_alap_val = old_asap - 1;
if let Some(t) = seg_tree_alap.query_last(start, end, cap_minus_one) {
    new_alap_val = t as i32;
}
// ...
if new_alap_val < old_alap {
    new_alap.insert(op_id, new_alap_val);
}
```

When `query_last` returns **`None`**:

- `new_alap_val` stays `old_asap - 1`.
- We still do `new_alap.insert(op_id, old_asap - 1)` because `old_asap - 1 < old_alap`.
- So we set **ALAP = old_asap - 1** for that op.

But ALAP must be ≥ ASAP. So we have **ALAP &lt; ASAP** (invalid window). That is an over-tightening bug: we excluded every cycle in the window without a proof (we never found a valid `t`). Consequences:

- Downstream (e.g. `new_int_var_with_name([lb, ub)]`) can see empty or invalid domains (ub &lt; lb).
- The precedence fixpoint does **not** repair “ALAP &lt; ASAP”; it only propagates precedence. So the invalid bound can persist and lead to INFEASIBLE or undefined behavior.

So **any** run where `query_last` returns `None` for some op is buggy with the old code. For runs that “work” with the old code, we are either (a) never hitting `None`, or (b) relying on some other step (e.g. domain construction) to effectively mask the invalid value.

---

## 3. Why `old_alap` is the correct fix (c70ca95)

Fix:

```rust
let mut new_alap_val = old_alap;
if let Some(t) = seg_tree_alap.query_last(start, end, cap_minus_one) {
    new_alap_val = t as i32;
}
// ...
if new_alap_val < old_alap {
    new_alap.insert(op_id, new_alap_val);
}
```

When `query_last` returns **`None`**:

- `new_alap_val` stays `old_alap`.
- We do **not** insert, because `new_alap_val < old_alap` is false.
- So we leave ALAP unchanged for that op.

So we only tighten when we actually found a valid `t`. That satisfies the strong guarantee: we only exclude cycles that we have a reason to exclude (the capacity check at `t`). We do **not** exclude the whole window by setting ALAP to an invalid value.

Conclusion: **the c70ca95 fix is the correct behavior.**

---

## 4. Is reverting the fix “just masking”?

Yes.

- Reverting restores the bug: when `query_last` returns `None` we again write `old_asap - 1` and create invalid bounds. So we are not fixing over-tightening in a principled way; we are re‑introducing an invalid ALAP value.
- If with the reverted code we see “more cycles” or “validation passes,” that can be because:
  - We rarely hit `None` for this instance, so old and new behave the same for most ops; or
  - Invalid domains (or downstream handling of them) change solver behavior or iteration in a way that accidentally avoids a bad schedule or hides another bug.

So reverting does **not** fix the tightening logic; it masks the init bug and can mask other issues. The right direction is to **keep the fix** and fix any other bugs (validation, model, or pipeline) that the correct bounds expose.

---

## 5. Why does validation fail with the fix?

With the fix we have **correct** ALAP (no invalid `old_asap - 1`). So:

- Domains are valid and at least as large as with the buggy code (when query returns `None` we keep `old_alap`).
- The solver can find a feasible schedule that satisfies the model.

If that schedule **fails validation** (wrong output), then the bug is not in “we over-tightened” but in one of:

1. **Model bug:** the model allows a schedule that is not actually feasible (e.g. wrong precedence, capacity, or symmetry), and the solver picks it.
2. **Solver / hints:** with larger or different domains, the solver (or hints) choose a different schedule that triggers a bug in lowering, encoding, or address assignment.
3. **Post-schedule bug:** the same cycle assignment is interpreted incorrectly later (e.g. scratch assignment, instruction encoding).

So the fix is not causing over-tightening; it is exposing a bug elsewhere. Next steps should be to **keep the c70ca95 fix** and to **find the root cause** of the validation failure (e.g. validation trace, compare model solution to reference, or bisect within the fix).

---

## 6. Recommended actions

1. **Keep the c70ca95 fix** (`new_alap_val = old_alap` when we don’t find a valid `t`). Do not revert to `old_asap - 1`.
2. **Treat “more cycles without the fix” as a symptom:** tightening should not make a previously feasible schedule infeasible. If we see worse cycles with the reverted (buggy) init, that can be due to invalid domains or different iteration behavior, not a correct design. The correct design is: only exclude cycles we can justify; the fix enforces that.
3. **Debug the validation failure with the fix:** use existing tools (e.g. validation trace, compare kernel scratch, trace IDs) to see which op/cycle/value is wrong and whether the model solution is consistent with the reference. Fix the underlying bug (model, encoding, or pipeline) rather than reverting the bounds fix.
4. **Optional:** Add an assertion or check that every ALAP written by the ALAP pull loop satisfies `ALAP >= ASAP` (or that we never insert when `new_alap_val < old_asap`) so we never reintroduce the invalid init.

---

## 7. References

- Design: [ASAP_ALAP_SLOT_TIGHTENING.md](ASAP_ALAP_SLOT_TIGHTENING.md) (strong guarantee, safe rules).
- Energetic algorithm: [ENERGETIC_SLOT_TIGHTENING_DEEP_DIVE.md](ENERGETIC_SLOT_TIGHTENING_DEEP_DIVE.md).
- Bisect: defaults 4r/16b first bad commit for “defaults” was c70ca95; “with fix backed out” ablation (energetic vs none) both passed validation.
