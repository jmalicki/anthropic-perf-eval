# Liveness Specification Verification

Z3-based formal verification of [docs/LIVENESS_SPECIFICATION.md](../docs/LIVENESS_SPECIFICATION.md).

## Z3 vs CP-SAT: two notions of feasibility

We use Z3 to reason about **CP-SAT feasibility** (does a feasible schedule exist?). The meaning of Z3's result depends on how we encode the check—it's easy to confuse them.

**Unified framing (soundness proofs):** In most of our Z3 checks we encode a **violation** of the theorem we want to prove (a counterexample). Then:
- **Z3 SAT** = Z3 found an assignment that violates our theorems = **bad**.
- **Z3 UNSAT** = no variable assignment violates our theorems = our theorems hold = **good**.

Example: T7 one-step. We encode the negation ("tightening holds AND feasible with O at t"). If that were satisfiable (SAT), we'd have a counterexample. We get **UNSAT** → no counterexample → theorem holds (good).

**What we want to preserve (strong statement):** We want tightening to preserve that **all** previously feasible schedules are still feasible—i.e. we only ever exclude (O,t) that no feasible schedule uses, so no feasible schedule is removed. That is strictly stronger than "at least one feasible schedule remains." **We have a Z3 proof of the strong guarantee when exclusions are entailed by precedence** (T7-strong-guarantee-entailed-exclusions): we check (precedence ∧ capacity ∧ ¬exclusions); UNSAT means no feasible schedule was removed. The one-step lemma proves it for a single exclusion; T7-two-step shows general composition can fail; witness checks show "at least one exists."

**CP-SAT feasibility (scheduler):** **Infeasible** = no feasible schedule = **bad**. The strong property above implies at least one feasible schedule exists; the converse is false.

**Preservation checks (e.g. T7-asap-only-composition):** We do **not** encode a violation; we ask "does a feasible schedule exist?" by giving Z3 the constraints (precedence + capacity + exclusions) and asking for a **witness**. **Z3 SAT** = witness found = at least one feasible schedule exists = a necessary condition for the strong property = **good**. **Z3 UNSAT** = no witness = we removed all feasible schedules = we violated the strong property = **bad**. So the witness is a **necessary** check: if we get UNSAT, we've definitely broken the strong guarantee. Finding one witness does **not** prove the strong guarantee (all previously feasible still feasible); it only shows that not everything was removed.

So: in **soundness** proofs we encode violations and want **Z3 UNSAT**. In **preservation** checks we ask for a witness and want **Z3 SAT**; that only confirms "at least one exists." Our aim is to use Z3 to show that our tightening procedures preserve the **strong** property (all previously feasible schedules still feasible). See also [SLOT_TIGHTENING_T7_COUNTEREXAMPLE.md](../docs/SLOT_TIGHTENING_T7_COUNTEREXAMPLE.md) and [CLAUDE.md](../CLAUDE.md) § Verification (Z3).

## Setup

```bash
uv pip install -e ".[verify]"
```

## Run

```bash
uv run python -m verification.verify
```

Options:

- `-t T1 -t T2` — Run only specific theorems
- `--timeout 60000` — Timeout in ms (default: 60000 = 1 minute)
- `-q` — Quiet (less output)
- `--links` — Print spec file path with anchors for each theorem (e.g. `path/to/spec.md#t1`)
- `--seed 42` — Random seed for Z3

## Greedy-cap rule (T7-greedy-cap-preserves-G)

**Exact condition (Z3-proven):** Exclude cycle `t` from op `O`'s domain only when (i) the one-step slot-tightening condition holds at `(O, t)` (demand from others at `t` plus `slots(O)` would exceed capacity), and (ii) **t ≠ G(O)** where G is the canonical feasible schedule (e.g. greedy). Under this rule, G remains feasible after any finite set of such exclusions.

**Implementation:** Slot tightening is **on by default** (R_precedence_only: no slot exclusions; see [ASAP_ALAP_SLOT_TIGHTENING.md](../docs/ASAP_ALAP_SLOT_TIGHTENING.md)). Use `--no-enable-asap-alap-tightening` to disable. With `--energetic-slot-tightening`, `tighten_asap_alap_by_slots` runs (ASAP-only + greedy-cap). The scheduler uses the greedy-cap rule so the canonical schedule G remains feasible (T7-greedy-cap-preserves-G). Design, safe/unsafe rules, and implementation are in [ASAP_ALAP_SLOT_TIGHTENING.md](../docs/ASAP_ALAP_SLOT_TIGHTENING.md); investigation (2r/16b, Z3) is in [SLOT_TIGHTENING_T7_COUNTEREXAMPLE.md](../docs/SLOT_TIGHTENING_T7_COUNTEREXAMPLE.md).

**Model used in proof:** 2 ops, 3 cycles, capacity 1, precedence O2 ≥ O1+1; G = (1, 2). Allowed exclusions: O1 from {0, 2}, O2 from {0, 1}. With the maximal allowed set, G is still a solution (SAT). See `check_t7_greedy_cap_preserves_canonical` in `theorems.py`.

**Domain-based refutation (T7-domain-based-removes-G):** If we use domain-based demand (count ops whose domain contains `t`) to decide exclusions, the second exclusion can remove G; Z3 shows that after exclusions c1≠0 and c2≠2, the assignment (1, 2) is not feasible (UNSAT).

## Undercounting energetic tightening (T7-undercount-*)

The energetic slot tightening (`--energetic-slot-tightening`) uses **undercounting**: at cycle t, `min_demand(t)` = count of ops with `asap[op]==t` (per group, engine). We exclude (O,t) only when `total_min_demand(t) > cap` (O contributes) or `total_min_demand(t) >= cap` (O does not contribute). This never over-tightens. See `scheduler_rs/src/cpsat_scheduler/bounds.rs` and [ENERGETIC_SLOT_TIGHTENING_DEEP_DIVE.md](../docs/ENERGETIC_SLOT_TIGHTENING_DEEP_DIVE.md) §7.1.

**Z3 proofs (parallelism + symmetry ordering; single-group, single-engine):**

- **T7-undercount-one-exclusion-sound**: One exclusion is sound. Model: 2 parallel ops, cap 1, ordering c1≤c2. Both asap 0 → min_demand(0)=2. Exclude O2 from 0. Prove (capacity ∧ ordering ∧ c2=0) UNSAT.
- **T7-undercount-chain-preserves-witness**: Exclusions preserve a witness. Model: fan-out O1→O2, O1→O3, cap 1. At t=1, min_demand=2; exclude O2 from 1. Witness (0,2,3) satisfies precedence, ordering, capacity. SAT.
- **T7-undercount-parallel-contributing-sound**: Contributing exclusion with 3 parallel ops, cap 2. All asap 0 → min_demand=3. Exclude O3 from 0. Prove (capacity ∧ ordering ∧ c3=0) UNSAT.
- **T7-undercount-contributing-implies-T7**: When O contributes and we exclude (total>cap), the T7 one-step condition holds. Symbolic link to general T7 soundness lemma.

## Theorems

Each theorem links to its definition in [LIVENESS_SPECIFICATION.md](../docs/LIVENESS_SPECIFICATION.md):

| ID | Description | Spec |
|----|-------------|------|
| T1 | Expiration rule: when `end(v)<=S`, reuse is safe | [#t1](../docs/LIVENESS_SPECIFICATION.md#t1) |
| T2 | Overlap definition: `overlaps([s1,e1],[s2,e2]) iff e1>s2 and e2>s1` | [#t2](../docs/LIVENESS_SPECIFICATION.md#t2) |
| T3 | ASAP bounds: `cycle(succ) >= cycle(pred) + offset` | [#t3](../docs/LIVENESS_SPECIFICATION.md#t3) |
| T4 | ALAP bounds: `cycle(pred) <= cycle(succ) - offset` | [#t4](../docs/LIVENESS_SPECIFICATION.md#t4) |
| T5 | Cycle domain: `ASAP(op) <= cycle(op) <= ALAP(op)` is sound | [#t5](../docs/LIVENESS_SPECIFICATION.md#t5) |
| T6 | Half-open conversion: `end_halfopen = end_closed + 1` | [#t6](../docs/LIVENESS_SPECIFICATION.md#t6) |
| T7 | Slot-aware tightening: one-step lemma | [#t7](../docs/LIVENESS_SPECIFICATION.md#t7) |
| T7-two-step | Two-step composition (counterexample: no feasible schedule) | [SLOT_TIGHTENING_T7_COUNTEREXAMPLE](../docs/SLOT_TIGHTENING_T7_COUNTEREXAMPLE.md) |
| T7-one-exclusion-asap | One ASAP exclusion preserves feasibility | [SLOT_TIGHTENING_T7_COUNTEREXAMPLE](../docs/SLOT_TIGHTENING_T7_COUNTEREXAMPLE.md) |
| T7-one-exclusion-alap | One ALAP exclusion preserves feasibility | [SLOT_TIGHTENING_T7_COUNTEREXAMPLE](../docs/SLOT_TIGHTENING_T7_COUNTEREXAMPLE.md) |
| T7-asap-only-composition | ASAP-only composition (safe-composition candidate) | [SLOT_TIGHTENING_T7_COUNTEREXAMPLE](../docs/SLOT_TIGHTENING_T7_COUNTEREXAMPLE.md) |
| T7-alap-only-composition | ALAP-only composition (safe-composition candidate) | [SLOT_TIGHTENING_T7_COUNTEREXAMPLE](../docs/SLOT_TIGHTENING_T7_COUNTEREXAMPLE.md) |
| T7-asap-plus-fixpoint-unsat | ASAP exclusions + precedence can remove all feasible schedules (fixpoint counterexample) | [SLOT_TIGHTENING_T7_COUNTEREXAMPLE](../docs/SLOT_TIGHTENING_T7_COUNTEREXAMPLE.md) |
| T7-strong-guarantee-entailed-exclusions | **Strong guarantee:** exclusions entailed by precedence preserve all feasible schedules (Z3 UNSAT) | [SLOT_TIGHTENING_T7_COUNTEREXAMPLE](../docs/SLOT_TIGHTENING_T7_COUNTEREXAMPLE.md) |
| T7-strong-guarantee-fails-asap-only | Attempt to prove strong guarantee for ASAP-only; fails (SAT = counterexample); important result preserved | [SLOT_TIGHTENING_T7_COUNTEREXAMPLE](../docs/SLOT_TIGHTENING_T7_COUNTEREXAMPLE.md) |
| T7-greedy-cap-preserves-G | **Greedy-cap rule:** canonical schedule G remains feasible (Z3-proven safe rule) | [SLOT_TIGHTENING_T7_COUNTEREXAMPLE](../docs/SLOT_TIGHTENING_T7_COUNTEREXAMPLE.md) |
| T7-domain-based-removes-G | Domain-based rule refutation: second exclusion removes canonical G | [SLOT_TIGHTENING_T7_COUNTEREXAMPLE](../docs/SLOT_TIGHTENING_T7_COUNTEREXAMPLE.md) |
| T7-undercount-one-exclusion-sound | **Undercounting:** one exclusion sound (2 parallel ops, ordering, cap 1) | [ENERGETIC_SLOT_TIGHTENING_DEEP_DIVE](../docs/ENERGETIC_SLOT_TIGHTENING_DEEP_DIVE.md) §7.1 |
| T7-undercount-chain-preserves-witness | **Undercounting:** fan-out exclusions preserve witness (O1→O2,O3 parallel) | [ENERGETIC_SLOT_TIGHTENING_DEEP_DIVE](../docs/ENERGETIC_SLOT_TIGHTENING_DEEP_DIVE.md) §7.1 |
| T7-undercount-parallel-contributing-sound | **Undercounting:** contributing exclusion sound (3 parallel ops, cap 2) | [ENERGETIC_SLOT_TIGHTENING_DEEP_DIVE](../docs/ENERGETIC_SLOT_TIGHTENING_DEEP_DIVE.md) §7.1 |
| T7-undercount-contributing-implies-T7 | **Undercounting:** contributing exclusion implies T7 one-step condition | [ENERGETIC_SLOT_TIGHTENING_DEEP_DIVE](../docs/ENERGETIC_SLOT_TIGHTENING_DEEP_DIVE.md) §7.1 |
| T8 | Scratch-aware tightening: one-step lemma | [#t8](../docs/LIVENESS_SPECIFICATION.md#t8) |
| T9 | Expiration implies no overlap | [#t9](../docs/LIVENESS_SPECIFICATION.md#t9) |
| T10 | Dead shifted VALU dest: `end(v) = max_cycle(v) + 1` | [#t10](../docs/LIVENESS_SPECIFICATION.md#t10) |
| T11 | Dead single-cycle VReg: `end(v) = D + 1` | [#t11](../docs/LIVENESS_SPECIFICATION.md#t11) |
| T12 | Shifted VALU dest with consumer: `end(v) = R` | [#t12](../docs/LIVENESS_SPECIFICATION.md#t12) |
| T13 | Occupied interval semantics: reusable at `end + 1` | [#t13](../docs/LIVENESS_SPECIFICATION.md#t13) |
