# Precedence Fixpoint and Slot Tightening: Formal Model

This document provides formal Z3/SMT models for the precedence fixpoint algorithm and its interaction with slot tightening. The models prove convergence, preservation properties, and identify counterexamples where the combined pipeline removes feasible schedules.

## Overview

The precedence fixpoint algorithm ensures that ASAP/ALAP bounds remain consistent with precedence constraints. When combined with slot tightening, it can propagate aggressive exclusions across the entire DAG, potentially removing all feasible schedules.

**Pipeline order:**
1. Compute initial ASAP/ALAP from precedence (no capacity)
2. Optional: slot tightening updates ASAP/ALAP (capacity-aware, not precedence-aware)
3. Precedence fixpoint propagates updated bounds along the precedence graph

**Key insight:** Slot tightening can make bounds inconsistent with precedence; the fixpoint restores consistency but may amplify over-tightening.

## Fixpoint Algorithm

### Definition

**Input:** ASAP and ALAP maps (possibly inconsistent with precedence), ops list, precedence graph.

**Algorithm:**
```
Repeat until no change (or max_iters = ops.len()):
  Forward pass: For each edge A → B with offset:
    ASAP[B] = max(ASAP[B], ASAP[A] + offset)
  
  Backward pass: For each edge A → B with offset:
    ALAP[A] = min(ALAP[A], ALAP[B] - offset)
```

**Output:** Precedence-consistent ASAP/ALAP bounds.

### Convergence Proof (Model 1)

**Theorem:** The fixpoint converges within `ops.len()` iterations on any DAG.

**Proof sketch:**
- Each forward pass can only increase ASAP values (non-decreasing)
- Each backward pass can only decrease ALAP values (non-increasing)
- Bounds are bounded (ASAP ≥ 0, ALAP ≤ horizon)
- In a DAG with N ops, the longest path has at most N-1 edges
- After N iterations, all paths have been propagated
- No further changes are possible

**Z3 model:** `model_fixpoint_convergence` in `verification/precedence_fixpoint.py`

**Result:** SAT (convergence is achievable)

### Precedence Preservation (Model 2)

**Theorem:** After fixpoint, bounds satisfy precedence inequalities:
- For each edge A → B: `ASAP[B] >= ASAP[A] + offset`
- For each edge A → B: `ALAP[A] <= ALAP[B] - offset`

**Proof:** By construction. The fixpoint explicitly enforces these inequalities.

**Z3 model:** `model_fixpoint_preserves_precedence`

**Result:** UNSAT (no violation exists → precedence is preserved)

## Slot Tightening + Fixpoint Interaction

### The Cascade Effect

When slot tightening pushes one op's ASAP forward (e.g., from 26 to 357), the fixpoint propagates that change to **all successors**:

```
Slot tightening: ASAP(O1) = 357 (was 26)
Fixpoint forward pass:
  ASAP(O2) >= ASAP(O1) + 1 = 358
  ASAP(O3) >= ASAP(O2) + 1 = 359
  ...
```

This **cascade** can push an entire tail of the DAG into late cycles, excluding compact schedules.

### Counterexample: Removes All Feasible Schedules (Model 3)

**Model:** 3 ops (O1, O2, O3), 5 cycles (0..4), capacity 1, chain precedence.

**Slot tightening exclusions:**
- `c1 >= 3` (O1 pushed from early cycles)
- `c2 >= 4` (O2 pushed from early cycles)

**Fixpoint propagation:**
- `c3 >= c2 + 1 >= 4 + 1 = 5`

**Result:** `c3 >= 5` but domain is `[0,4]` → **no feasible schedule exists** (UNSAT).

**Z3 model:** `model_slot_tightening_plus_fixpoint_unsat`

**Result:** UNSAT (proves no feasible schedule remains)

**Reference:** This extends `T7-asap-plus-fixpoint-unsat` from `verification/theorems.py`.

### Parameterized Cascade Analysis (Model 4)

**Question:** For an N-op chain with makespan M, how large can a push P be before feasibility is lost?

**Model:** 4-op chain, makespan 5, capacity 1.

**Tightening:** Push O1 by P cycles (`c1 >= P`)

**Fixpoint propagation:**
- `c2 >= P + 1`
- `c3 >= P + 2`
- `c4 >= P + 3`

**Feasibility condition:** `c4 <= 4` → `P + 3 <= 4` → `P <= 1`

**Z3 results:**
- `P = 1`: SAT (feasible schedule exists, e.g., (1,2,3,4))
- `P = 2`: UNSAT (c4 >= 5 > 4, no feasible schedule)

**Z3 model:** `model_fixpoint_cascade_analysis`

**Insight:** The maximum safe push magnitude decreases as the chain length increases or makespan decreases.

### Fan-Out DAG (Model 5)

**Model:** Fan-out O1 → O2, O1 → O3 (two parallel branches).

**Slot tightening:** Push O1 forward (`c1 >= 2`)

**Fixpoint propagation:** Both O2 and O3 get pushed:
- `c2 >= c1 + 1 >= 3`
- `c3 >= c1 + 1 >= 3`

**Result:** SAT (feasible schedule exists, e.g., (2,3,4))

**Z3 model:** `model_fixpoint_with_fanout`

**Insight:** Fan-out spreads the push to multiple branches, but each branch can still find a feasible cycle.

## Iteration Boundedness (Model 6)

**Theorem:** Fixpoint converges within `ops.len()` iterations.

**Model:** 3-op chain, explicit iteration encoding.

**Result:** SAT (convergence achieved in 2 iterations)

**Z3 model:** `model_fixpoint_iteration_bounded`

**Proof sketch:**
- Iteration 1: Forward pass propagates ASAP [0,0,0] → [0,1,2]; backward pass propagates ALAP [4,4,4] → [2,3,4]
- Iteration 2: No further changes (converged)
- For N ops, at most N iterations needed (one per op in longest path)

## Order of Operations (Model 7)

**Question:** Does the order matter? Tighten-then-fixpoint vs fixpoint-then-tighten?

**Hypothesis:** Tightening before fixpoint may preserve more schedules because:
- Fixpoint sees tighter initial bounds
- Less aggressive propagation

**Model:** 3-op chain, capacity 1, makespan 5.

**Case A:** Tighten O1 (`c1 >= 2`), then fixpoint → `c2 >= 3`, `c3 >= 4`

**Case B:** Fixpoint first (bounds [0,2], [1,3], [2,4]), then tighten O1 (`c1 >= 2`)

**Z3 results:** Both cases yield SAT (feasible schedule exists)

**Z3 model:** `model_slot_tightening_before_fixpoint_preserves_feasibility`

**Insight:** The order may matter for more complex DAGs or aggressive tightening, but in simple cases both orders preserve feasibility.

## Summary of Results

| Model | Claim | Result | Meaning |
|-------|-------|--------|---------|
| Fixpoint convergence | Converges within N iterations | SAT | Algorithm terminates |
| Precedence preservation | Final bounds satisfy precedence | UNSAT | No violations exist |
| Slot + fixpoint UNSAT | Can remove all feasible schedules | UNSAT | Counterexample exists |
| Cascade analysis P=1 | Push magnitude 1 preserves feasibility | SAT | Safe for small pushes |
| Cascade analysis P=2 | Push magnitude 2 removes feasibility | UNSAT | Unsafe for large pushes |
| Fan-out | Fan-out preserves feasibility (moderate push) | SAT | Fan-out is less dangerous |
| Iteration bounded | Converges in ≤N iterations | SAT | Bounded runtime |
| Order of operations | Tighten-then-fixpoint preserves feasibility | SAT | Order may matter |

## Key Insights

1. **Fixpoint converges:** The algorithm terminates within `ops.len()` iterations.

2. **Fixpoint preserves precedence:** By construction, final bounds satisfy all precedence inequalities.

3. **Cascade can remove feasibility:** Aggressive slot tightening + fixpoint propagation can exclude all feasible schedules. The maximum safe push magnitude depends on chain length and makespan.

4. **Fan-out is safer:** Fan-out DAGs spread pushes across branches, reducing the risk of removing all feasible schedules.

5. **Order may matter:** Tightening before fixpoint may preserve more schedules, but the effect depends on DAG structure and tightening aggressiveness.

## Implementation Implications

1. **Bounded iterations:** The fixpoint loop can safely use `max_iters = ops.len()`.

2. **Validation:** After slot tightening + fixpoint, check that `ASAP[op] <= ALAP[op]` for all ops. If violated, the model is INFEASIBLE.

3. **Conservative tightening:** Limit push magnitudes to avoid cascade failures. For a chain of length L with makespan M, maximum safe push is approximately `M - L`.

4. **Greedy-cap protection:** The `greedy-cap` rule (never exclude G(O) for any op) ensures at least the greedy schedule remains feasible, preventing complete INFEASIBLE.

## References

- `scheduler_rs/src/cpsat_scheduler/bounds.rs`: `fix_precedence_bounds_fixpoint` implementation
- `verification/precedence_fixpoint.py`: Z3 models
- `verification/theorems.py`: T7-asap-plus-fixpoint-unsat (existing counterexample)
- `docs/ENERGETIC_SLOT_TIGHTENING_DEEP_DIVE.md`: §4.2 (fixpoint amplification)
- `docs/ASAP_ALAP_SLOT_TIGHTENING.md`: Canonical design, safe/unsafe rules

## Running the Models

```bash
# Run all precedence fixpoint models
uv run python -m verification.verify -t fixpoint-convergence
uv run python -m verification.verify -t fixpoint-preserves-precedence
uv run python -m verification.verify -t slot-tightening-plus-fixpoint-unsat
uv run python -m verification.verify -t fixpoint-cascade-analysis
uv run python -m verification.verify -t fixpoint-with-fanout
uv run python -m verification.verify -t fixpoint-iteration-bounded
uv run python -m verification.verify -t slot-tightening-before-fixpoint
```
