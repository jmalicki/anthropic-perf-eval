# Hint-Pass VALU/ALU Overflow: Deep Bug Report and Fixes

This document explains the bugs that caused **"Merged hint violates engine capacity"** panics when using `--hint-passes` or `--hint-passes --schedule-with-fixed-choices`, and the fixes that were applied. It goes from high-level impact down to root causes and data-flow invariants.

---

## 1. Design intent: hint passes supersede greedy; they do not coexist

### 1.1 What hint passes are for

The **hint passes** exist to **update** the encoding decisions that the first greedy run produced. They run **after** greedy and **overwrite** the relevant parts of `greedy_decisions` (ConstChoice, SelectChoice, VALU shift). The idea is:

- **Greedy** produces an initial schedule: it assigns cycles and makes encoding choices (const → Load/add_imm, SelectChoice → vselect/FMA, which VALU ops to shift).
- **Hint passes** then **replace** those encoding choices with their own, based on per-cycle slack (e.g. “use AddImm where Flow has slack,” “use FMA where VALU is saturated,” “shift to ALU where ALU has slack”).

So after hint passes, the **authoritative** encoding is what the hint passes wrote. Greedy’s encoding for those dimensions is **superseded**; it should not still be in play. There is no design where “greedy’s encodings and hint-pass encodings both exist and we pick one for cycles and one for encodings.” The hint pass is the update; the state after hint passes is the single source of truth for encodings.

### 1.2 The dependency that was missed

**Cycles** depend on **encodings**: the cycle assignment is only valid for the encoding under which it was computed. If you change “op X is vselect (Flow)” to “op X is FMA (VALU),” the cycle assignment that put 6 VALU + op X as Flow in cycle 50 is no longer valid when op X becomes VALU—you’ve effectively added a 7th VALU in that cycle.

So the intended flow is:

1. Greedy runs → sets cycles and encodings (both in `op.cycle` and `greedy_decisions`).
2. Hint passes run → **update** `greedy_decisions` only; they **supersede** greedy’s encoding. At this point the **encodings** are correct (hint-pass authority), but **cycles** are stale (they were computed for the old encodings).
3. We must **re-derive cycles** under the new encodings so that the only schedule we use (for lowering, for CP-SAT hints) has cycles consistent with the authoritative encodings. That is what **ScheduleWithFixedChoices** does: re-run greedy with encodings **fixed** to the current `greedy_decisions` (the hint-pass output), and write the new `op.cycle`. After that, there is a single coherent schedule: encodings from hint passes, cycles from a scheduler run that respected those encodings.

So hint passes and greedy are not “two schedules that coexist.” Hint passes **update** encodings; we then **recompute** the dependent quantity (cycles) so everything is consistent. The bugs were (a) not re-deriving cycles after the update, or (b) re-deriving only part of the encodings (const and shift but not SelectChoice), or (c) not running this flow at all on the CP-SAT path.

### 1.3 Decisions and who is authoritative

| Decision | Meaning | Set by greedy | **Updated by hint passes** (supersedes greedy) | Must be re-derived when encodings change |
|----------|---------|----------------|-------------------------------------------------|------------------------------------------|
| **Cycle** | Which cycle each op runs in | `op.cycle` | — | **Yes** (ScheduleWithFixedChoices) |
| **ConstChoice** | Load vs AddImm (Flow) | `greedy_decisions.const_choice_encoding` | ConstChoiceHintPass overwrites | — |
| **SelectChoice** | FMA (VALU) vs vselect (Flow) | `greedy_decisions.select_choice_encoding` | SelectChoiceHintPass overwrites | — |
| **VALU shift** | Which VALU ops use ALU (8 slots each) | `greedy_decisions.shifted_ops` | ValuShiftHintPass overwrites | — |

After hint passes, **encodings** are authoritative (hint-pass output). **Cycles** must be recomputed under those encodings; otherwise we are mixing “old cycles” with “new encodings” and the result is not a valid schedule.

### 1.4 How CP-SAT uses the hint schedule

CP-SAT is given:

- **Cycle hints:** `op.cycle` for each op.
- **Encoding hints:** from `greedy_decisions` (after hint passes have run).

It **verifies** that this combined schedule obeys engine capacity every cycle (VALU ≤ 6, ALU ≤ 12, etc.). If the merged schedule violates capacity, it **panics**.

So the invariant is: **the cycle assignment and the encoding decisions must describe one and the same feasible schedule.** When hint passes have updated encodings, that “one schedule” must use those encodings and cycles that were **derived** under those encodings—not cycles left over from before the update.

---

## 2. Bug #1: Stale cycles after hint passes update encodings (VALU overflow 7 > 6)

### 2.1 High-level impact

- **Symptom:** Panic as soon as CP-SAT starts: *"VALU overflow at cycle N: 7 > 6"*.
- **When:** Any run with `--hint-passes` (and no re-schedule), or with `--hint-passes --schedule-with-fixed-choices` before the re-schedule fix.
- **User impact:** Hint-pass ablation and experiments were unusable; the process died in a few hundred ms.

### 2.2 Root cause: we updated encodings but did not re-derive cycles

Hint passes are supposed to **supersede** greedy’s encoding decisions. They run after greedy and **overwrite** `greedy_decisions` (ConstChoice, SelectChoice, shifted_ops). So after hint passes, the **authoritative** encodings are the hint-pass output; greedy’s encodings for those fields are no longer in effect.

The bug: we **updated** the encodings (hint passes wrote to `greedy_decisions`) but we **did not** re-derive the thing that **depends** on those encodings—the cycle assignment. We kept using `op.cycle` from the **first** greedy run, which was computed **under greedy’s original encodings** (e.g. all SelectChoice as vselect/Flow). So we had:

- **Encodings:** authoritative, from hint passes (e.g. “op X is FMA”).
- **Cycles:** stale, from greedy under the **old** encodings (op X was placed at cycle 50 as Flow).

When CP-SAT builds the “hint schedule,” it correctly uses **current** encodings (hint-pass output) and **current** cycles (`op.cycle`). So it counts “op X at cycle 50 as FMA” → 1 VALU. But in the schedule that **produced** that cycle 50, op X was Flow, so that cycle had 6 VALU + 1 Flow. Under the **updated** encoding (FMA), the same cycle now has 7 VALU → violation.

So the issue is not “two schedules coexisting.” It is: **we updated encodings (hint passes supersede greedy) but left cycles as-is, so cycles were computed for a different encoding than the one we now consider authoritative.** The fix is to **re-derive cycles** after the update, so that the only schedule we use has cycles that are consistent with the authoritative encodings.

### 2.3 Why ScheduleWithFixedChoices didn’t fix it fully (before SelectChoice fix)

**ScheduleWithFixedChoicesPass** is exactly “re-derive cycles under the current encodings”: it re-runs greedy with encodings **fixed** to the current `greedy_decisions` (after hint passes) and rewrites `op.cycle`. So once we run it, cycles **should** match the authoritative encodings.

But the re-run only **partially** used the updated encodings. The greedy API had:

- `fixed_const_choice_encoding` and `fixed_shifted_ops` → re-run used the hint-pass updates for ConstChoice and shift.
- **No** `fixed_select_choice_encoding` → for SelectChoice, the re-run still used greedy’s default (vselect/Flow).

So after the re-run, **ConstChoice and shift** were consistent (cycles derived under those encodings), but **SelectChoice** was not: `greedy_decisions.select_choice_encoding` could say “op X is FMA” (hint pass updated it), while the re-run had placed op X at cycle 50 **as Flow** (because it never read the updated SelectChoice). So we still had stale cycles for the SelectChoice dimension—the re-derivation was incomplete.

### 2.4 Fix for Bug #1

- **Greedy:** Add `fixed_select_choice_encoding: Option<&HashMap<usize, SelectChoiceEncoding>>`. When present, the re-run uses the map for SelectChoice (Fma → VALU, Vselect → Flow) in both the main loop and the force-schedule path, so cycles are derived under the **full** authoritative encoding (including hint-pass SelectChoice).
- **ScheduleWithFixedChoicesPass:** Pass `Some(&ctx.greedy_decisions.select_choice_encoding)` so the re-run uses the same SelectChoice as the hint passes.
- **Pipeline / tick_tock_ssa:** When **any** hint pass is enabled, **always** run ScheduleWithFixedChoices after the hint passes. That way, whenever we update encodings (hint passes), we always re-derive cycles; we never hand CP-SAT (or lowering) stale cycles plus new encodings.

**Restored invariant:** After hint passes update encodings, we always re-derive cycles under those encodings. The single schedule we use has encodings from hint passes and cycles from a scheduler run that respected those encodings.

---

## 3. Bug #2: Fixed-shift ops placed on VALU (ALU overflow)

### 3.1 High-level impact

- **Symptom:** After fixing Bug #1, panic changed to *"Merged hint violates engine capacity (VALU: 0 cycles, ALU: 1 cycles)."* So some cycle had ALU count > 12.
- **When:** Same hint-pass flows; appears once the re-run greedy runs with `fixed_shifted_ops` and `fixed_select_choice_encoding`.
- **User impact:** Hint-pass + re-schedule still crashed; the bug was subtle because the greedy *thought* it was respecting slots while the verifier disagreed.

### 3.2 Root cause: re-derivation didn’t treat “must be shifted” as binding

When we re-derive cycles (ScheduleWithFixedChoices), we pass the **authoritative** encodings from `greedy_decisions` (after hint passes). For shift, that means `fixed_shifted_ops`: the set of ops that the hint pass decided should be shifted. So in the re-run, those ops **must** be placed as shifted; the re-derived schedule is only valid if it respects that.

The verifier assumes exactly that: for each op in **cycle** `c`,

- If the op is in **shift_hints** (e.g. `greedy_decisions.shifted_ops`), it counts as **8 ALU slots** at cycle `c`.
- Otherwise, if it’s a VALU op, it counts as **1 VALU** at cycle `c`.

So the **contract** for the re-derived schedule is: *"If an op is in `shifted_ops` (authoritative encoding), the schedule must have that op using 8 ALU slots in its cycle, not 1 VALU slot."*

In the re-run greedy, when `fixed_shifted_ops` is `Some(set)` and the op is in the set, we set `can_shift = true` and called `slots.try_place(Engine::Valu, can_shift, allow_shift)`. The slot tracker’s logic is:

1. If **VALU has room** (`valu_used < 6`), it places the op **on VALU** and returns `Placed`.
2. Else if **can_shift && allow_shift && alu_used + 8 ≤ 12**, it places **on ALU** (shifted) and returns `Shifted`.

So when VALU still had room, the greedy **placed the op on VALU** (1 VALU slot), while the verifier—because the op is in `shifted_ops`—counted it as **8 ALU**. Same cycle could then have, for example:

- 6 VALU (correct) + 1 op that greedy put on VALU but hints say is shifted → verifier counts 6 VALU + 8 ALU for that op.
- Another shifted op in the same cycle → verifier adds another 8 ALU → 16 ALU in one cycle → ALU violation.

So the **invariant** that broke: *When re-deriving cycles under fixed encodings, every op in `fixed_shifted_ops` must be placed only as shifted (ALU), never as VALU.* The implementation allowed “place on VALU when there’s room,” which is correct when the scheduler is **choosing** whether to shift, but wrong when the encoding has already been **set** by the hint pass—the re-derivation must treat that as binding.

### 3.3 Why it’s subtle

- The greedy never exceeded **its own** view of capacity (VALU ≤ 6, ALU ≤ 12) because it was only adding 1 VALU when it placed the op on VALU.
- The mismatch was **semantic**: the same op was "shifted" in the hint and "non-shifted" in the actual placement. So the bug is in the **contract** between the re-run greedy and the hint: when we pass `fixed_shifted_ops`, we are promising "these ops will be placed as shifted"; the code must not place them on VALU.

### 3.4 Fix for Bug #2

- **SlotTracker:** Add `try_place_shifted_only()`: only tries the ALU path (add 8 to `alu_used`); returns `NoRoom` if `alu_used + VLEN > 12`. So it never places on VALU.
- **Greedy main path:** When `fixed_shifted_ops` contains the op (`must_place_shifted`):
  - First check `alu_used + VLEN <= 12`; if not, defer to a later cycle.
  - If room, call `try_place_shifted_only()` instead of `try_place(engine, can_shift, allow_shift)`.
- **Greedy force-schedule path:** Same idea: if `must_place_shifted` and ALU is full, defer (push back to heap for next cycle); otherwise call `try_place_shifted_only()`.

So for any op in `fixed_shifted_ops`, we **only** use the shift path; we never place it on VALU. The verifier’s count (8 ALU per such op) then matches exactly what the greedy did.

**Restored invariant:** *For every op in `fixed_shifted_ops`, the schedule produced by the greedy uses exactly 8 ALU slots in that op’s cycle, and 0 VALU for that op.*

---

## 4. Bug #3: Hint passes and re-schedule not run in CP-SAT path

### 4.1 High-level impact

- **Symptom:** Even after adding "run ScheduleWithFixedChoices when any hint pass is set" in **pipeline/mod.rs**, the **hint-passes** ablation arm still panicked with VALU overflow. Logs showed no "ScheduleWithFixedChoices" between GreedySchedule and the section check.
- **User impact:** The fix in the pipeline had no effect for the actual code path that runs when you call the generator with CP-SAT; only the greedy-only path was fixed.

### 4.2 Root cause: CP-SAT path never ran “update encodings, then re-derive cycles”

The intended flow when hint passes are enabled is: (1) greedy runs and sets initial cycles and encodings; (2) hint passes **update** encodings (supersede greedy for those fields); (3) we **re-derive** cycles under the updated encodings (ScheduleWithFixedChoices). So there is one coherent schedule: encodings from hint passes, cycles from a run that respected those encodings.

Kernel build has two high-level paths:

1. **Greedy-only path:** Implemented via `run_greedy_pipeline()` in **pipeline/mod.rs**, which runs GreedySchedule, then the hint-pass block, then ScheduleWithFixedChoices when any hint pass is set. So it does “update encodings, then re-derive cycles.”
2. **CP-SAT path:** Implemented in **tick_tock_ssa/mod.rs** inside `build_and_schedule_cpsat()`. It does **not** call the pipeline; it manually runs ComputeDependencies, (DCE), ComputeBounds, **GreedySchedule**, then immediately the per-section budget check and later ElideBarriers and CpSatSchedulePass. It **never** ran the hint-pass block or ScheduleWithFixedChoices.

So on the CP-SAT path, even with `--hint-passes`, we never updated encodings (hint passes didn’t run) and never re-derived cycles. We were still using greedy’s encodings and cycles everywhere—except that the **Python** side had set the hint-pass flags, so if anything downstream had read those flags and expected “hint-pass encodings + re-derived cycles,” it would see the wrong state. In practice, the CP-SAT path never applied the hint-pass design at all; it was as if hint passes didn’t exist on that path.

### 4.3 Fix for Bug #3

- In **tick_tock_ssa/mod.rs**, immediately after GreedySchedule, add the **same** block as in the pipeline:
  - If `ctx.config.compiler.enable_const_choice_hint_pass`, run ConstChoiceHintPass.
  - If `ctx.config.compiler.enable_select_choice_hint_pass`, run SelectChoiceHintPass.
  - If `ctx.config.compiler.enable_valu_shift_hint_pass`, run ValuShiftHintPass.
  - If **any** of these is true **or** `enable_schedule_with_fixed_choices`, run ScheduleWithFixedChoicesPass.

After this, the CP-SAT path and the pipeline path both run hint passes and re-schedule when hint passes are enabled, so cycle hints and encodings stay consistent and the verifier no longer sees an impossible merged schedule.

**Restored invariant:** *Whenever hint passes are enabled, the same sequence of passes (hint passes + ScheduleWithFixedChoices) runs regardless of whether we are on the greedy-only or CP-SAT path.*

---

## 5. Design takeaway: update, then re-derive—no coexistence

Hint passes are not a “second schedule” that coexists with greedy. They **update** the encoding decisions that greedy wrote; after that update, greedy’s encoding for those dimensions is superseded. The only correct state is: **encodings = what the hint passes wrote (authoritative), cycles = re-derived under those encodings.** If we ever use cycles that were computed under the **pre-update** encodings, we are mixing stale data with the new authority and the result is not a valid schedule. So the flow must always be: run greedy → run hint passes (update encodings) → re-derive cycles (ScheduleWithFixedChoices). The bugs were failures to do that update/re-derive flow completely or on the right path.

---

## 6. Summary table

| Bug | High-level impact | Root cause (invariant broken) | Fix in one line |
|-----|-------------------|-------------------------------|------------------|
| **#1 Stale cycles after encoding update** | VALU 7 > 6 panic; hint-pass runs unusable | Hint passes **update** encodings (supersede greedy) but we didn’t **re-derive** cycles; re-derivation (ScheduleWithFixedChoices) didn’t include SelectChoice | Re-derive cycles using **all** authoritative encodings (add fixed_select_choice_encoding); always run ScheduleWithFixedChoices when any hint pass runs |
| **#2 Re-derivation didn’t enforce “must shift”** | ALU overflow panic after #1 fixed | When re-deriving under fixed encodings, ops in fixed_shifted_ops must be placed only as shifted; greedy could still place them on VALU when VALU had room | Place ops in fixed_shifted_ops only via try_place_shifted_only() (never on VALU) |
| **#3 CP-SAT path skipped update + re-derive** | Pipeline fix had no effect; still panic on hint-passes | CP-SAT path never ran hint passes or ScheduleWithFixedChoices, so it never did “update encodings, then re-derive cycles” | Run the same hint-pass + ScheduleWithFixedChoices block in build_and_schedule_cpsat() |

Together, these restore the intended design: **hint passes supersede greedy’s encoding decisions; we then re-derive cycles under those encodings so the single schedule we use (for CP-SAT hints and for lowering) is coherent and never violates engine capacity.** Hint passes and greedy do not simultaneously exist as two schedules—the hint passes **update** the decisions, and we **re-derive** the dependent cycles so there is only one consistent schedule.

---

## 7. Why the Z3 proofs didn’t catch these bugs

We have a Z3 verification suite (see [verification/README.md](../verification/README.md) and [docs/LIVENESS_SPECIFICATION.md](LIVENESS_SPECIFICATION.md)) that proves properties about **tightening** and **CP-SAT feasibility**. Those proofs did not (and were not designed to) catch the hint-pass bugs. This section explains the **scope gap**.

### 7.1 What the Z3 proofs actually prove

The Z3 proofs reason about:

- **Domain exclusions (tightening):** When we **exclude** an (op, cycle) pair from an op’s domain (e.g. “op O cannot run at cycle t”) based on slot demand or scratch, do we **preserve feasibility**? I.e. do we only exclude (O,t) that no feasible schedule uses, so that at least one feasible schedule (or the canonical greedy schedule G) still exists?
- **One-step soundness:** One slot-aware or scratch-aware exclusion step is sound (T7, T8).
- **Composition and counterexamples:** Two-step composition can remove all feasible schedules (T7-two-step); fixpoint + tightening can over-remove (slot-tightening-plus-fixpoint-unsat); the **greedy-cap rule** keeps the canonical schedule G feasible (T7-greedy-cap-preserves-G).
- **Strong guarantee in special cases:** Exclusions entailed by precedence preserve all feasible schedules (T7-strong-guarantee-entailed-exclusions).
- **Liveness / precedence / bounds:** T1–T6, T9–T13 (expiration, overlap, ASAP/ALAP, etc.).

So the Z3 layer is about: **given** a set of constraints (precedence, capacity) and **given** a set of domain exclusions produced by our tightening rules, **does a feasible schedule exist?** and **does the canonical schedule G remain feasible?** It does **not** model the pipeline, hint passes, or the greedy’s behavior when given fixed encodings.

### 7.2 What the hint-pass bugs were about

The hint-pass bugs lived in a **different layer**:

| Concern | Z3 verification | Hint-pass bugs |
|--------|------------------|-----------------|
| **What is being checked?** | Tightening rules: “if we exclude (O,t) under condition X, do we preserve feasibility?” | Pipeline and scheduler: “after we **update** encodings (hint passes), do we **re-derive** cycles? Does the greedy **enforce** fixed encodings when re-deriving?” |
| **Object of study** | The **CP-SAT model**: constraints, domains, existence of a feasible assignment. | **Orchestration** (when do we run which passes?), **data consistency** (cycles vs encodings), and **greedy implementation** (when `fixed_shifted_ops` is set, must we place only as shifted?). |
| **Invariant** | “Exclusions don’t remove all feasible schedules” / “G remains feasible.” | “The (cycle, encoding) pair we pass to CP-SAT comes from one coherent schedule: encodings updated by hint passes, cycles re-derived under those encodings.” |

So:

- **Bug #1 (stale cycles):** We **did** pass a (cycle, encoding) pair to CP-SAT that was **infeasible** (7 VALU in one cycle). But that infeasibility was not caused by **tightening** (excluding (O,t) from domains). It was caused by **never re-deriving cycles** after updating encodings—a pipeline / data-consistency bug. Z3 proves that **if** we apply our tightening rules, **then** we don’t over-exclude. Z3 does not prove that we always run “update encodings → re-derive cycles” before building the hint.
- **Bug #2 (fixed-shift on VALU):** The greedy, when given `fixed_shifted_ops`, could place an op on VALU instead of as shifted. So the **schedule** we produced didn’t match the **encoding** we claimed (shifted). Z3 reasons about constraints and domains; it doesn’t model the greedy’s placement logic or the contract “when fixed_shifted_ops contains O, O must be placed only as shifted.”
- **Bug #3 (CP-SAT path):** The CP-SAT path never ran hint passes or ScheduleWithFixedChoices. That’s **which code path runs which passes**—orchestration. Z3 doesn’t model the Rust/Python pipeline or the two entry points (greedy-only vs CP-SAT).

So the Z3 proofs are about **soundness of the tightening rules relative to the CP-SAT model**. They were not designed to verify **pipeline consistency**, **re-derivation after encoding updates**, or **greedy’s behavior under fixed encodings**. Different layer, different invariants.

### 7.3 Could we add Z3 (or other) verification for this layer?

In principle we could try to formalize and verify:

1. **Pipeline invariant:** “Whenever hint passes run, ScheduleWithFixedChoices runs after them (on the same path).” That could be a **static** check (both pipeline and tick_tock_ssa have the same conditional block) or a small model of “pass sequence” and a proof that the sequence always includes re-schedule when hint passes are enabled.
2. **Re-derived schedule consistency:** “The hint schedule (op.cycle + greedy_decisions) passed to CP-SAT is built from a single run: encodings from greedy_decisions, cycles from a scheduler run that used those encodings as fixed.” That is more of a **design/implementation** invariant (we enforce it by always running ScheduleWithFixedChoices when hint passes run and by having the re-run use all fixed encodings).
3. **Greedy contract for fixed_shifted_ops:** “For every op in fixed_shifted_ops, the greedy places it only as shifted (never on VALU).” That could be a **unit test** (run greedy with a small fixed set and assert per-cycle ALU/VALU counts) or, in theory, a more formal model of the greedy’s placement rules—but the greedy is complex (scratch, force-schedule, multiple engines), so Z3 models of it would be non-trivial.

So the existing Z3 work **does** prevent a real class of bugs (over-tightening that makes the CP-SAT problem infeasible or removes the canonical schedule). The hint-pass bugs were **outside that class**: they were about keeping the **hint schedule** itself coherent (update then re-derive, and enforce fixed encodings during re-derivation), which the current verification suite does not model. Making that explicit helps explain why the Z3 proofs didn’t catch these—and where additional checks (tests, static structure, or future formalization) could help.
