# Big-O Analysis: Energetic ASAP/ALAP Slot Tightening

This document characterizes the time and space complexity of the **energetic** slot tightening implementation in `scheduler_rs/src/cpsat_scheduler/bounds.rs` (`tighten_asap_alap_by_slots`).

## Algorithm summary

- **Input:** ops, asap/alap maps, horizon, per-engine capacities, optional shiftable set, greedy cycles, and `SymmetryOrdering`.
- **Behavior:** Uses symmetry-aware **undercounting**: for each engine, builds per-group chains of ops that definitely use that engine; at each cycle `t`, `min_demand[group][t]` = count of ops in that group’s chain with `asap[op] == t`. For each op, finds the earliest `t` in its window where `total_min_demand(t) <= cap` (and applies greedy-cap). Only ASAP pushes; no ALAP pulls.
- **Iteration:** Up to `MAX_ITER` (3) iterations; each iteration runs over all 5 engines. Stops early if no bound changed.

## Notation

| Symbol | Meaning |
|--------|--------|
| **N** | Number of SSA ops |
| **G** | Number of ordering groups (from `SymmetryOrdering`) |
| **H** | Horizon (max cycle index + 1); typically `greedy_max + 20` |
| **E** | Number of engines = 5 (constant) |
| **I** | Max iterations = 3 (constant) |
| **W** | Max window size per op = `alap - asap + 1`; worst case **W = O(H)** |

## Time complexity

### Per engine, per iteration

1. **Build phase**
   - Iterate over all groups: **O(G)**.
   - For each group: build `engine_chain` by filtering that group’s sorted op indices by engine and “not optional” → over all groups, each op is visited at most once ⇒ **O(N)** total.
   - For each group: allocate `demand_at_t[0..=horizon]` ⇒ **O(G · H)**.
   - For each group: for each op in that group’s `engine_chain`, one HashMap lookup and one index increment ⇒ over all groups, **O(N)**.
   - **Build total: O(N + G · H).**

2. **Tighten phase**
   - For each group and each op in that group’s `engine_chain`: over all groups, each op appears in at most one chain for this engine ⇒ **O(N)** op visits (for ops that use this engine; still O(N) in worst case).
   - For each such op: scan `t` from `old_asap` to `min(old_alap, horizon)` (window size ≤ **W**, with **W = O(H)**).
   - For each `t`: sum `min_demand[group][t]` over all groups ⇒ **O(G)** per `t`.
   - **Tighten total: O(N · W · G)**; worst case **O(N · H · G)**.

**Per engine per iteration:**  
**O(N + G·H + N·H·G) = O(N · H · G)** when N, H, G ≥ 1.

### Full procedure

- **Iterations:** at most **I** (3).
- **Engines:** **E** (5).

**Time complexity:**

**T = O(I · E · (N · H · G)) = O(N · H · G).**

Constants I and E are fixed, so the **dominant term is N · H · G**.

### Typical scaling

- **N:** Grows with problem size (e.g. rounds × batch × tree work). Often in the thousands to tens of thousands for 16-round kernels.
- **G:** Number of distinct symmetry (ordering) groups; often **O(rounds × n_groups)** (e.g. hundreds).
- **H:** Horizon ≈ greedy makespan + 20; **O(makespan)** (hundreds to thousands of cycles).

So in practice the procedure can be **cubic in problem size** when N, H, and G all scale (e.g. with rounds and batch): **O(N · H · G)**.

## Space complexity

- **Per engine, per iteration:**
  - `group_chains`: **O(N)** (total op indices across groups).
  - `min_demand`: **G** vectors of length **H + 1** ⇒ **O(G · H)**.
- **Persistent state:** `new_asap` / `new_alap`: **O(N)** (copies of input maps).
- **No accumulation across iterations:** we reuse and overwrite `group_chains` and `min_demand` each time.

**Space complexity: O(N + G · H).**

## Summary

| Complexity | Bound | Notes |
|------------|--------|--------|
| **Time** | **O(N · H · G)** | Dominant: for each op, scan up to H cycles and sum over G groups; × I × E is constant. |
| **Space** | **O(N + G · H)** | Maps + per-group demand vectors over horizon. |

So the energetic ASAP/ALAP tightening is **linear in N**, **linear in H**, and **linear in G**, i.e. **O(N · H · G)** time and **O(N + G · H)** space. When N, H, and G all grow with problem size (rounds, batch, makespan), the runtime grows like the product of those three dimensions.
