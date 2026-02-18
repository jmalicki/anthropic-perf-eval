# Big-O Analysis: Energetic ASAP/ALAP Slot Tightening

This document characterizes the time and space complexity of the **energetic** slot tightening implementation in `scheduler_rs/src/cpsat_scheduler/bounds.rs` (`tighten_asap_alap_by_slots`).

## Algorithm summary (current: segment tree)

- **Input:** ops, asap/alap maps, horizon, per-engine capacities, optional shiftable set, greedy cycles, and `SymmetryOrdering`.
- **Behavior:** Uses symmetry-aware **undercounting**: for each engine, builds per-group chains of ops that definitely use that engine; at each cycle `t`, `min_demand[group][t]` = count of ops in that group’s chain with `asap[op] == t`. For each op, finds the earliest `t` in its window where `total_min_demand(t) <= cap` (and applies greedy-cap). Only ASAP pushes; no ALAP pulls.
- **Iteration:** Up to `MAX_ITER` (3) iterations; each iteration runs over all 5 engines. Stops early if no bound changed.
- **Implementation:** Single `total_min_demand[0..=H]` plus a segment tree (`DemandSegmentTree` in `bounds.rs`) for O(log H) "first/last index in [L,R] where value ≤ cap" queries and point updates; see below.

## Notation

| Symbol | Meaning |
|--------|--------|
| **N** | Number of SSA ops |
| **G** | Number of ordering groups (from `SymmetryOrdering`) |
| **H** | Horizon (max cycle index + 1); typically `greedy_max + 20` |
| **E** | Number of engines = 5 (constant) |
| **I** | Max iterations = 3 (constant) |
| **W** | Max window size per op = `alap - asap + 1`; worst case **W = O(H)** |

## Time complexity (with segment tree — current implementation)

### Per engine, per iteration

1. **Build phase**
   - Iterate over all groups and build `engine_chain` per group (filter by engine and not optional) ⇒ **O(N)** over all groups.
   - Build **total_min_demand[0..=H]** in one pass over all ops in `group_chains`: for each op, `total_min_demand[asap[op]] += 1` ⇒ **O(N)**.
   - Build segment tree from `total_min_demand` ⇒ **O(H)**.
   - **Build total: O(N + H).**

2. **Tighten phase (ASAP)**
   - For each op in `group_chains` (same order as before): **O(N)** op visits.
   - Per op: one **query_first(L, R, cap)** ⇒ **O(log H)**; optional **contributing-only** check (read `total_min_demand[op_asap]`, O(1)); if ASAP changed, two **point updates** on the segment tree ⇒ **O(log H)**.
   - **Tighten total: O(N · log H).**

3. **ALAP pulls (optional)**  
   Rebuild `total_min_demand` and segment tree from current `new_asap` ⇒ **O(N + H)**; then for each op one **query_last(L, R, cap-1)** ⇒ **O(N log H)**.

**Per engine per iteration:**  
**O(N + H + N log H) = O(N log H)** when H = Ω(N).

### Full procedure (segment tree)

- **Iterations:** at most **I** (3).
- **Engines:** **E** (5).

**Time complexity (with segment tree):**

**T = O(I · E · (N + N log H)) = O(N log H).**

When **H = O(N)** (horizon scales with makespan and N), this is **O(N log N)**.

### Typical scaling

- **N:** Grows with problem size (e.g. rounds × batch × tree work). Often in the thousands to tens of thousands for 16-round kernels.
- **H:** Horizon ≈ greedy makespan + 20; **O(makespan)**; in practice often **O(N)**.
- **G:** No longer in the dominant term (we use a single total demand array and segment tree).

So in practice the procedure is **O(N log N)** when H = O(N).

---

## Time complexity (previous: naive per-group demand)

### Per engine, per iteration (naive)

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

## Space complexity (with segment tree)

- **Per engine, per iteration:**
  - `group_chains`: **O(N)** (total op indices across groups).
  - `total_min_demand`: one vector of length **H + 1** ⇒ **O(H)**.
  - Segment tree over `total_min_demand`: **O(H)** (e.g. 4× rounded size).
- **Persistent state:** `new_asap` / `new_alap`: **O(N)** (copies of input maps).
- **No accumulation across iterations:** we reuse and overwrite `group_chains`, `total_min_demand`, and the tree each time.

**Space complexity: O(N + H).**

## Summary (current implementation)

| Complexity | Bound | Notes |
|------------|--------|--------|
| **Time** | **O(N log H)** | Build total_min_demand + tree O(N + H); N queries/updates at O(log H) each. With H = O(N) → **O(N log N)**. |
| **Space** | **O(N + H)** | Maps + one demand vector + segment tree over horizon. |

The segment tree supports **point update** (add delta at index `i`) and **query_first** / **query_last** (first/last index in `[L,R]` where value ≤ cap), each in **O(log H)**. See `DemandSegmentTree` in `bounds.rs` and the comment block above `tighten_asap_alap_by_slots`.
