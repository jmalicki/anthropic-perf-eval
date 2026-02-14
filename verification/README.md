# Liveness Specification Verification

Z3-based formal verification of [docs/LIVENESS_SPECIFICATION.md](../docs/LIVENESS_SPECIFICATION.md).

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

**Implementation:** The scheduler follows this rule when building the CP-SAT model: `tighten_asap_alap_by_slots` in `scheduler_rs/src/cpsat_scheduler/bounds.rs` takes an optional `greedy_cycles` (the canonical schedule G); when present, it clamps ASAP/ALAP so the domain for each op always contains its canonical cycle. Thus G is never excluded and remains feasible. Strengths, weaknesses, and possible future work are documented in [SLOT_TIGHTENING_T7_COUNTEREXAMPLE.md](../docs/SLOT_TIGHTENING_T7_COUNTEREXAMPLE.md) § "Greedy-cap fix: strengths, weaknesses, future work".

**Model used in proof:** 2 ops, 3 cycles, capacity 1, precedence O2 ≥ O1+1; G = (1, 2). Allowed exclusions: O1 from {0, 2}, O2 from {0, 1}. With the maximal allowed set, G is still a solution (SAT). See `check_t7_greedy_cap_preserves_canonical` in `theorems.py`.

**Domain-based refutation (T7-domain-based-removes-G):** If we use domain-based demand (count ops whose domain contains `t`) to decide exclusions, the second exclusion can remove G; Z3 shows that after exclusions c1≠0 and c2≠2, the assignment (1, 2) is not feasible (UNSAT).

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
| T7-greedy-cap-preserves-G | **Greedy-cap rule:** canonical schedule G remains feasible (Z3-proven safe rule) | [SLOT_TIGHTENING_T7_COUNTEREXAMPLE](../docs/SLOT_TIGHTENING_T7_COUNTEREXAMPLE.md) |
| T7-domain-based-removes-G | Domain-based rule refutation: second exclusion removes canonical G | [SLOT_TIGHTENING_T7_COUNTEREXAMPLE](../docs/SLOT_TIGHTENING_T7_COUNTEREXAMPLE.md) |
| T8 | Scratch-aware tightening: one-step lemma | [#t8](../docs/LIVENESS_SPECIFICATION.md#t8) |
| T9 | Expiration implies no overlap | [#t9](../docs/LIVENESS_SPECIFICATION.md#t9) |
| T10 | Dead shifted VALU dest: `end(v) = max_cycle(v) + 1` | [#t10](../docs/LIVENESS_SPECIFICATION.md#t10) |
| T11 | Dead single-cycle VReg: `end(v) = D + 1` | [#t11](../docs/LIVENESS_SPECIFICATION.md#t11) |
| T12 | Shifted VALU dest with consumer: `end(v) = R` | [#t12](../docs/LIVENESS_SPECIFICATION.md#t12) |
| T13 | Occupied interval semantics: reusable at `end + 1` | [#t13](../docs/LIVENESS_SPECIFICATION.md#t13) |
