# Scratch Liveness Specification

Formal specification of scratch memory occupancy, liveness intervals, and allocation rules. No handwaving; all definitions are precise and support formal reasoning.

**Structure**: We first state **axioms** (minimal assumptions). Then we list **theorems** we want to prove from those axioms. The theorems include the bounds and constraints we feed to CP-SAT (ASAP/ALAP, cycle domains, scratch cumulative). Proving these theorems ensures our implementation is sound; Z3 can be used to verify them.

---

## Notation

| Symbol | Meaning |
|--------|---------|
| `t`, `C`, `R`, `D`, `L`, `S` | Cycle indices (non-negative integers) |
| `v`, `w` | VRegs (virtual registers / SSA values) |
| `op` | A scheduled operation |
| `[a, a+size)` | Scratch address range (half-open) |
| `start(v)` | First cycle at which `v` occupies its range |
| `end(v)` | Last cycle at which `v` blocks reuse when allocating a writer (see §3) |
| `completion_cycle(op)` | Cycle when op's result is visible (next cycle after last write) |
| `min_cycle(v)`, `max_cycle(v)` | Min/max cycle among lane writes (shifted VALU dest) |
| `[s, e]` | Closed interval of cycles (inclusive) |
| `[s, e)` | Half-open interval (used in CP-SAT cumulative) |
| `last_is_read(v)` | True iff `v` has consumers and last access is a read |
| `last_is_write(v)` | True iff `v` has no consumer or last access is a write |
| `pred → succ` | SSA edge: `succ` consumes a value produced by `pred` |

---

## Part I: Axioms

The following are **axioms**: foundational assumptions. No proofs required; they model the ISA and our allocation model.

### A1. VLIW Order (ISA)

Each cycle has two ordered phases:
1. **Read phase**: All loads from scratch occur. Each read sees the state at the end of the previous cycle.
2. **Write phase**: All stores to scratch occur. Results are visible at the start of the next cycle.

**Axiom (A1)**: For any cycle `t`, every read at `t` completes before any write at `t` begins.

### A2. Reuse Rule

**Axiom (A2)**: Scratch range `[a, a+size)` is reusable at cycle S iff for every prior allocation to that range with last access at cycle L:
- If the last access was a **read** at L: then `S ≥ L` suffices (read completes before write at same cycle).
- If the last access was a **write** at L: then `S > L` is required (writes at the same cycle conflict).

### A3. Consumer Readiness (Precedence)

**Axiom (A3)**: For SSA edge `pred → succ`, a consumer may execute at cycle `t` only if `t ≥ completion_cycle(pred) + 1`. The producer's result is visible one cycle after the last write.

### A4. Definition of `end` by VReg Kind

For each VReg `v`, `end(v)` is the maximum cycle such that we must **not** allow reuse when allocating a **writer** at that cycle. The following definitions satisfy A2:

| # | VReg kind | `end(v)` |
|---|-----------|----------|
| 1 | Normal (has consumers), last read at R | `R` |
| 2 | Dead (no consumers), written at D | `D + 1` |
| 3 | Shifted VALU dest (no consumer) | `max_cycle(v) + 1` |
| 4 | Shifted VALU dest (has consumer at R) | `R` (R ≥ max_cycle+1 by precedence) |
| 5 | FMA dest (writes at cycle+1) | last read or `op.cycle+2` if dead |
| 6 | FMA sub_temp | `op.cycle + 1` (FMA reads at cycle+1) |

### A5. Completion Cycle

**Axiom (A5)**:
- Normal op: `completion_cycle(op) = op.cycle`
- Shifted VALU: `completion_cycle(op) = max(alu_cycle[0], …, alu_cycle[7])`
- FMA: `completion_cycle(op) = op.cycle + 1`

### A6. Slot Capacity

**Axiom (A6)**: Each engine has a fixed slot limit per cycle (ALU 12, VALU 6, Load 2, Store 2, Flow 1). At most that many ops of that engine may execute in any single cycle.

### A7. Scratch Budget

**Axiom (A7)**: Total scratch memory is finite (budget = 1536 words). VRegs must be assigned disjoint address ranges; overlapping liveness intervals require disjoint ranges.

---

## Part II: Theorems to Prove

The following are **theorems**: they should be derivable from axioms A1–A7 alone. We feed these to CP-SAT as explicit constraints; they tighten the search space but add no new semantic assumptions. Proofs (or Z3 verification) should establish: **axioms ⊢ theorem**.

### Constraint scope: unconditional vs conditional (enforce-if)

- **Unconditional constraints** hold for **every** valid schedule. They do not depend on choice variables (ConstChoice, SelectChoice, VALU shift) or on whether an optional structure is present. The axioms A1–A7 and theorems T1–T6, T9–T13 are in this category: they apply to every op/VReg that exists in the schedule. The CP-SAT model enforces them always (e.g. precedence, overlap, scratch cumulative for live VRegs, half-open conversion).

- **Conditional (enforce-if) constraints** apply only when a given optional part is active. The CP-SAT model encodes these with optional interval variables or `only_enforce_if(condition)` so that the constraint is enforced iff the condition holds. Examples:
  - **Shifted VALU**: An op may use VALU (one interval) or ALU (eight intervals); only the chosen alternative contributes to slot demand and scratch. Precedence and liveness (T12, T10) apply to the *chosen* placement.
  - **CSE shared SUB**: The shared sub_temp VReg and its liveness (§3.6, §7) exist only when `sub_active`; scratch and precedence for that SUB are conditional on `sub_active` and on each FMA consumer's choice.
  - **SelectChoice**: Flow (vselect) vs VALU (SUB+FMA) — only the chosen path has intervals and precedence.

When we say a theorem "holds for each fixed assignment" (e.g. T14), we mean: for any assignment to choice variables, the constraint holds for the ops/intervals that are present under that assignment. Tightening (T7, T8) must count only intervals that are *present* for the candidate assignment (optional intervals excluded from demand when the choice is not yet fixed), so the tightened bounds remain valid for both enforce-if and non-enforce-if encodings. The encoding of these conditional constraints (optional intervals, only_enforce_if, exactly-one) is specified in Part III §10 (Choice variable encoding).

<a id="t1"></a>
### T1. Expiration Rule

**Theorem**: When allocating a new VReg `w` that writes at cycle S, expire prior allocation `v` iff `end(v) ≤ S`. This rule never causes a write-write conflict.

*Proof obligation*: Show that whenever we reuse (i.e. `end(v) ≤ S`), the last access of `v` cannot be a write at S. (By A4, if last access is write at L, `end(v) ≥ L+1`; so `end(v) ≤ S` implies S > L.)

<a id="t2"></a>
### T2. Overlap ⇒ Disjoint Allocation

**Theorem**: Two closed intervals `[s₁,e₁]` and `[s₂,e₂]` overlap (conflict) iff `e₁ > s₂ ∧ e₂ > s₁`. Overlapping intervals must receive disjoint scratch ranges.

<a id="t3"></a>
### T3. ASAP Bounds (Precedence)

**Theorem**: For each SSA edge `pred → succ`, any valid schedule satisfies
```
cycle(succ) ≥ cycle(pred) + offset_asap(pred)
```
where `offset_asap(pred)` = minimum delay (1 for normal, 1 for shifted min, etc.). Therefore:
```
ASAP(succ) ≥ ASAP(pred) + offset_asap(pred)
```
for any fixed-point ASAP assignment. *Derives from A3.*

<a id="t4"></a>
### T4. ALAP Bounds (Precedence)

**Theorem**: For each SSA edge `pred → succ`, any valid schedule satisfies
```
cycle(pred) ≤ cycle(succ) - offset_asap(pred)
```
Therefore:
```
ALAP(pred) ≤ ALAP(succ) - offset_asap(pred)
```
for any fixed-point ALAP assignment. *Derives from A3.*

<a id="t5"></a>
### T5. Cycle Domain

**Theorem**: For each op, `ASAP(op) ≤ cycle(op) ≤ ALAP(op)` defines a sound domain (contains all feasible cycles). *Derives from T3, T4, and the definition of ASAP/ALAP as fixpoints.*

<a id="t6"></a>
### T6. Half-Open Conversion

**Theorem**: For CP-SAT cumulative scratch constraint, closed interval `[s, e]` (last occupied cycle e) corresponds to half-open `[s, e+1)`. So `end_halfopen = end_closed + 1`. *Derives from A1, A2, A4.*

<a id="t7"></a>
### T7. Slot-Aware ASAP/ALAP Tightening

**Theorem**: If we push ASAP forward or ALAP backward using slot capacity (A6) so that the demand in each op's window respects the cap, the tightened bounds still contain at least one feasible schedule. *Derives from A6 and the fact that tightening only removes infeasible regions.*

*Scope*: Unconditional in the sense that the theorem holds for any schedule. When the model has optional intervals (choice variables), demand in a window must count only ops that are *present* for the engine in that window; optional-interval ops are excluded from the profile so we do not over-tighten (see `bounds.rs` "optional intervals").

<a id="t8"></a>
### T8. Scratch-Aware ASAP/ALAP Tightening

**Theorem**: If we push ASAP forward or ALAP backward using scratch liveness (A7) so that total scratch usage in each candidate window stays within budget, the tightened bounds still contain at least one feasible schedule. *Derives from A7 and the liveness definitions.*

*Scope*: Same as T7 — when optional structures (e.g. CSE shared sub_temp) exist only under a condition, scratch demand in a window should count only VRegs whose intervals are present for the candidate assignment.

<a id="t9"></a>
### T9. Expiration Implies No Overlap

**Theorem**: The expiration rule (T1) ensures that when we reuse an address, the new interval `[S, E]` does not overlap the expired interval `[start(v), end(v)]`. *Proof*: Overlap requires `S < end(v)`; expiration requires `end(v) ≤ S`; contradiction.

<a id="t10"></a>
### T10. Dead Shifted VALU Dest End

**Theorem**: For a shifted VALU dest `v` with no consumers, `end(v) = max_cycle(v) + 1`. *Derives from A2*: the last access is a write at `max_cycle(v)`; per A2, reuse for a new writer at cycle S requires `S > max_cycle(v)`, hence the minimal `end(v)` that blocks reuse at `max_cycle` is `max_cycle(v) + 1`.

*CP-SAT linkage*: Dead VRegs are excluded from the scratch cumulative (DCE assumes no dead). If DCE is ever disabled, dead shifted VALU dests must use half-open end `max_cycle + 2` per T6 (closed `end = max_cycle + 1` → half-open `[min_cycle, max_cycle + 2)`). `model.rs` would need to add such intervals to `scratch_intervals`.

<a id="t11"></a>
### T11. Dead Single-Cycle VReg End

**Theorem**: For a VReg `v` written at cycle D with no consumers, `end(v) = D + 1`. *Derives from A2*: the last access is a write at D; per A2, reuse for a new writer at cycle S requires `S > D`, hence the minimal `end(v)` that blocks reuse at D is `D + 1`.

*CP-SAT linkage*: Same as T10—dead VRegs are omitted from `live_vregs` and thus from `scratch_intervals` (cumul_scratch). `address_assign.rs` third pass extends `[D, D]` to `[D, D+1]`; expiration logic (T1) uses this for linear-scan allocation.

<a id="t12"></a>
### T12. Shifted VALU Dest with Consumer

**Theorem**: For a shifted VALU dest `v` with a consumer at last read R, `end(v) = R`. *Derives from A2 and A3*: precedence forces `R ≥ max_cycle(v) + 1`; the last access is a read at R. Per A2, reuse for a writer at R is allowed (read-then-write same cycle), so the minimal `end(v)` is R.

*CP-SAT linkage*: `model.rs` scratch constraint uses `last_use_idx` (consumer op) for end. For live shifted VALU dests, end = `cycle_vars[last_use_idx] + 1` (half-open). Per T6, closed `[min_cycle, R]` → half-open `[min_cycle, R+1)`. The start uses `min(alu_cycles)` when shifted (see CRITICAL FIXES in model.rs). *Conditional*: this VReg and its interval exist only when the op is shifted to ALU (optional interval / choice); when the op stays on VALU, a single-cycle interval is used instead.

<a id="t13"></a>
### T13. Occupied Interval Semantics

**Theorem**: For VReg `v` with closed interval `[start(v), end(v)]`, the address is occupied for every cycle `t` with `start(v) ≤ t ≤ end(v)`, and becomes reusable for a writer at cycle `end(v) + 1`. *Derives from A2, A4, and the definition of `end`*.

*CP-SAT linkage*: The scratch cumulative constraint (`add_cumulative(scratch_intervals, scratch_demands, scratch_budget)`) enforces that at any cycle t, the sum of demands for intervals spanning t does not exceed budget. Intervals are half-open `[start, end)` per T6; the CP-SAT solver treats `t` as contained when `start ≤ t < end`, which corresponds to occupied for `start ≤ t ≤ end_closed` where `end = end_closed + 1`. `address_assign.rs` uses the same occupied semantics for overlap detection (T2) and expiration (T1).

<a id="t14"></a>
### T14. Capacity Lower Bound

**Theorem**: For each engine with slot capacity C per cycle (A6), if the total number of op-executions (op-count) for that engine in a schedule is N, then the schedule's makespan satisfies `makespan ≥ ⌈N / C⌉`. Therefore any valid schedule has `makespan ≥ max_engine ⌈N_engine / C_engine⌉`. *Derives from A6*: at most C ops of that engine execute per cycle, so at least ⌈N / C⌉ cycles are required.

*CP-SAT linkage*: The solver uses a capacity-derived lower bound on makespan (e.g. `compute_capacity_lower_bound_global` in `bounds.rs`) to set the makespan domain and redundant constraints. *Unconditional*: the inequality makespan ≥ ⌈N/C⌉ holds for every valid schedule (for each engine). When choice variables exist (VALU shift, ConstChoice, SelectChoice), N depends on the assignment; the bound we use is the minimum over all choice assignments of that max, so it remains a valid lower bound for the model with enforce-if. See also T7 (slot-aware tightening).

---

## Part III: Definitions and Supporting Detail

### 1. Primitives

- **Cycle**: A non-negative integer `t ∈ ℕ`. The machine executes cycles `0, 1, 2, ...`.
- **Op**: A scheduled operation with `op.cycle = t` meaning the op executes in cycle `t`.
- **VReg**: A virtual register (SSA value). Each VReg has at most one definition and zero or more uses.
- **Scratch address**: An integer `a ∈ [0, budget)` (budget = 1536). Vector VReg occupies 8 words; scalar occupies 1.
- **Write event** `W(v, t, a)`: At cycle `t`, an op writes `v` to `[a, a+size)`; result visible at `t+1`.
- **Read event** `R(v, t, a)`: At cycle `t`, an op reads from `[a, a+size)` during the read phase.
- **Occupied**: Range is occupied at cycle t iff there is a write at t or a read at t requiring a prior write.
- **Reusable**: Range is reusable at cycle S iff we may allocate a new VReg whose first write is at S.

#### Cycle semantics and half-open intervals

Within each cycle, **reads come before writes** (A1). So at cycle `t`, the read phase sees the state at the end of the previous cycle; the write phase then commits; results are visible at the start of the next cycle.

It is natural to describe liveness in **half-open** form with integer cycle indices. We say interval **`[s, e)`** is live from the **write at cycle s** up until **cycle e's writes** (i.e. the range is still needed through the read phase of cycle `e-1`, and if the last use is a **read at cycle e**, that read happens before cycle e's writes, so the range is reusable when cycle e's write phase begins). So:
- **`[0, 5)`** = live for cycles 0, 1, 2, 3, 4; first free when cycle 5's write phase starts.
- If the last use is a **read at cycle 5**, we need the range through cycle 5's read phase, so we use **`[0, 6)`** (occupied for `t` with `0 ≤ t < 6`, i.e. cycles 0..5).

So in half-open form, **end** is the first cycle at which a new writer can use the range (the first cycle whose write phase no longer conflicts). Conversion from "last occupied cycle" to half-open end is therefore `end_halfopen = last_occupied_cycle + 1` when the last access was a read at that cycle, and similarly for write-only (dead) as in A4.

**Z3**: Integer cycle indices are sufficient for verification. Z3 works well with integers; we do not need to change the definition (e.g. to reals or to a finer-grained sub-cycle model) to encode read-before-write. The half-open condition `start ≤ t < end` and the reuse rules (A2, T1) already capture the intended semantics.

### 2. Liveness Interval (Closed Form)

We represent liveness as a **closed interval** `[start, end]` of cycles.

**Definition (LiveInterval)**: For VReg `v` with allocation `[a, a+size)`:
- `start(v)` = the minimum cycle at which any word in `[a, a+size)` is written.
- `end(v)` = defined per case below; it is the maximum cycle such that we must **not** allow reuse when allocating a **writer** at that cycle.

**Semantic meaning**: The address is occupied for every cycle `t` with `start(v) ≤ t ≤ end(v)`. It becomes reusable at cycle `end(v) + 1` for a writer, and at cycle `end(v)` for a read-then-write at the same cycle (per A2). *(T13)*

**Expiration rule**: When allocating a new VReg `w` that writes at cycle S, expire prior allocation `v` iff `end(v) ≤ S`. *(See T1, T9.)*

---

### 3. Definition of `end` by VReg Kind (A4)

#### 3.1 Normal VReg (single-cycle write, has consumers)

- **Write**: One op writes `v` at cycle `D`. `start(v) = D`.
- **Last read**: Some consumer reads `v` at cycle `R` (maximum over all consumers).
- **Last access type**: read at `R`.
- **Rule**: We allow reuse when allocating a writer at `R` (read then write). So `end(v) ≤ R`. We need the address through `R` for the read, so the minimal choice is `end(v) = R`.

**Definition**: `end(v) = R` where `R` is the maximum cycle among all consumers of `v`.

#### 3.2 Dead VReg (single-cycle write, no consumers)

- **Write**: One op writes `v` at cycle `D`. `start(v) = D`.
- **Last access type**: write at `D`.
- **Rule**: We must not allow reuse when allocating a writer at `D`. So `end(v) > D`, hence `end(v) ≥ D + 1`. Minimal: `end(v) = D + 1`.

**Definition**: `end(v) = D + 1`. *(T11)*

#### 3.3 Shifted VALU Dest (multi-cycle write)

A shifted VALU op is expanded into 8 ALU ops, each writing one lane. Let the 8 ALU cycles be `c₀, c₁, …, c₇`.

- **min_cycle(v)** = min{c₀, …, c₇}
- **max_cycle(v)** = max{c₀, …, c₇}
- **completion_cycle(v)** = max_cycle(v)  (cycle when the last lane write occurs)

- **Write span**: Lanes are written at cycles `min_cycle(v)` through `max_cycle(v)`. The address is written to (at least one lane) at each of those cycles.
- **start(v)** = min_cycle(v).
- **Last write**: at cycle `max_cycle(v)`.
- **Last access type**: write at `max_cycle(v)` (until a consumer reads; see below).

**Rule**: We must not allow reuse when allocating a writer at `max_cycle(v)`. So `end(v) > max_cycle(v)`, hence `end(v) ≥ max_cycle(v) + 1`.

**If `v` has no consumer** (dead shifted VALU dest):

**Definition**: `end(v) = max_cycle(v) + 1`. *(T10)*

**If `v` has a consumer** at cycle `C`: The consumer must read no earlier than `max_cycle(v) + 1` (results visible next cycle). So the last read `R ≥ max_cycle(v) + 1`. We take `end(v) = max{R, max_cycle(v) + 1}`. Since `R ≥ max_cycle(v) + 1`, we have `end(v) = R`. So the consumer read dominates; the interval is `[min_cycle(v), R]`. *(T12)*

**Definition**: For shifted VALU dest with consumer at last read `R`:
```
end(v) = max(R, max_cycle(v) + 1)
```
Since precedence forces `R ≥ max_cycle(v) + 1`, this simplifies to `end(v) = R`.

#### 3.4 SelectChoice FMA Dest

The FMA writes at cycle `op.cycle + 1`. So:
- **start(v)** = op.cycle + 1
- **end(v)** = last read cycle (as in 3.1), or `op.cycle + 2` if dead (as in 3.2).

#### 3.5 SelectChoice FMA sub_temp

- **start** = op.cycle (sub executes)
- **end** = op.cycle + 1 (FMA reads sub_temp at cycle+1)
- **Last access type**: read at op.cycle + 1. So `end = op.cycle + 1`.

#### 3.6 CSE shared sub_temp

When SelectChoice ops share the same `(true_val, false_val)` operand pair, they form a **CSE group**; at most one shared SUB computes `sub_temp = true_val - false_val`, and every FMA consumer in that group reads that same `sub_temp`. So the shared sub_temp VReg has **multiple consumers** (one per FMA in the group).

- **start** = sub_cycle (the single cycle when the shared SUB runs).
- **Last read**: the maximum cycle among all FMA consumers in the group that read sub_temp. Call that cycle R.
- **Last access type**: read at R.
- **Rule**: Same as 3.1 (normal VReg with consumers): we need the range through R; reuse for a writer at R is allowed. So **end = R**.

*CP-SAT linkage*: When CSE groups exist, the model introduces one sub_temp VReg per group **when sub_active**. Its liveness interval is `[sub_cycle, R]` (closed) where R is the maximum FMA consumer cycle for that group. Scratch cumulative and address assignment treat it like any other VReg with a single definition and multiple uses. *Conditional*: this VReg and its interval exist only when sub_active (enforce-if / optional); constraints on it are only enforced when the shared SUB is present. See §7 and `docs/SUB_CSE_ALU_SHIFT_DESIGN.md`.

---

### 4. Overlap and Allocation Invariant (T2, T9)

**Definition (Allocation conflict)**: Two allocations to the same scratch range **conflict** iff there exists a cycle when both require exclusive use.

**Definition (Overlap)**: Two closed intervals `[s₁, e₁]` and `[s₂, e₂]` **overlap** iff `e₁ > s₂ ∧ e₂ > s₁`. Adjacent intervals (e.g. `[0,5]` and `[5,10]`) do not overlap.

**Allocation invariant (A7)**: Two VRegs with overlapping intervals must receive disjoint scratch ranges.

---

### 5. Completion Cycle (A5)

**Definition (Completion cycle)**:
- For a normal op: `completion_cycle(op) = op.cycle` (result visible at cycle+1).
- For a shifted VALU op with ALU cycles `[c₀,…,c₇]`: `completion_cycle(op) = max(c₀,…,c₇)`.
- For SelectChoice FMA (no delta): `completion_cycle(op) = op.cycle + 1` (FMA writes at cycle+1; result visible at cycle+2).

**Definition (Consumer readiness)**: A consumer may execute at cycle `t` iff `t ≥ completion_cycle(pred) + 1` (A3).

---

### 6. Shift Model (VALU → ALU)

When a VALU op is **shifted** to ALU:
- The op is replaced by 8 ALU ops, one per lane.
- Each ALU op has its own cycle variable `alu_cycle[i]` for `i ∈ {0,…,7}`.
- **completion_cycle** = max over `alu_cycle[i]`.
- Precedence: for each consumer, `consumer.cycle ≥ completion_cycle + 1`.
- The dest VReg's liveness uses `start = min(alu_cycle[i])`, `end` per §3.3.

**Invariant**: For a shifted dest with no consumer, `end = max_cycle + 1`. For a shifted dest with consumer at `R`, `end = R` (since `R ≥ max_cycle + 1` by precedence).

---

### 7. CSE Shared SUB

**CSE group**: A set of SelectChoice ops (with `needs_subtraction`, no `delta_vreg`) that share the same operand pair `(true_val, false_val)`. Built at emission; see `scheduler_rs/src/cpsat_scheduler/cse_groups.rs`.

**Shared SUB**: For each CSE group G, when at least one consumer chooses FMA, one logical op computes `sub_temp = true_val - false_val`. That SUB runs at cycle `sub_cycle` (decision variable when sub_active). All FMA consumers in G read the same sub_temp; their FMA ops run at cycles ≥ sub_cycle + 1 (A3). All constraints on the shared SUB (placement, precedence, scratch, capacity) are **conditional**: enforced only when sub_active (e.g. optional interval or only_enforce_if in the CP-SAT model). Without CSE or when no consumer in G chooses FMA, the shared SUB and its sub_temp do not exist.

**Liveness of shared sub_temp**: As in §3.6 — `start = sub_cycle`, `end = R` where R is the maximum cycle among FMA consumers in G. So the closed interval is `[sub_cycle, R]`; half-open for CP-SAT is `[sub_cycle, R+1)` per T6.

**Capacity**: With CSE, VALU count for SelectChoice is not 2 per FMA but (1 SUB per active group) + (number of FMA consumers). Capacity lower bound (T14) and slot cumulative constraints use this group-aware count; see `bounds.rs` `min_valu_from_select_cse` and `compute_capacity_lower_bound_global`, and `docs/SUB_CSE_ALU_SHIFT_DESIGN.md` §4.5–4.6.

---

### 8. Half-Open Representation (T6)

The CP-SAT cumulative constraint uses **half-open** intervals `[start, end)`:
- Occupied for cycles `t` with `start ≤ t < end`.
- "End" in half-open = first cycle at which the range is free.

**Conversion from closed to half-open**:
- Closed `[s, e]` (last occupied cycle `e`) → Half-open `[s, e+1)`.
- So `end_halfopen = end_closed + 1`.

For shifted VALU dest (no consumer): closed `end = max_cycle + 1` → half-open `end = max_cycle + 2`.

---

### 9. Flow NoOverlap

No two Flow intervals that are **present** may overlap in time. Formally: for any two distinct Flow interval variables that are both present, their half-open intervals `[s_i, e_i)` and `[s_j, e_j)` do not overlap (no shared cycle: either `e_i ≤ s_j` or `e_j ≤ s_i`).

**Implementation**: The CP-SAT model adds `add_no_overlap(flow_intervals)` in addition to `add_cumulative(flow_intervals, flow_demands, flow_cap)` (`scheduler_rs/src/cpsat_scheduler/model.rs` ~1549–1553).

**Redundancy**: When **Flow capacity = 1** and **every Flow interval has size (demand) = 1**, the cumulative constraint already enforces that at most one Flow op runs at any cycle. So at any time point at most one interval is active ⇒ no two intervals overlap. In that case NoOverlap is redundant with cumulative.

**When it matters**: If Flow capacity is ever increased (e.g. 2) or if multi-cycle Flow ops (size > 1) are introduced, cumulative alone would allow multiple intervals per cycle up to capacity; NoOverlap still forbids overlapping intervals. The current ISA has Flow cap 1 and all Flow intervals size 1, so the model keeps NoOverlap for clarity and future-proofing.

---

### 10. Choice variable encoding

The encoding of conditional constraints (Part II "Constraint scope") uses the following CP-SAT primitives and four choice kinds.

**Primitives**:
- **Optional interval**: An interval variable with a **presence literal** (Boolean). The interval is **present** in the schedule iff the literal is true. When present, it contributes to cumulative constraints and obeys precedence; when absent, it does not.
- **only_enforce_if(C, [lit])**: Constraint C is enforced only when `lit` is true. Used so that precedence and other constraints for an alternative apply only when that alternative is chosen.
- **Exactly-one (mutual exclusion)**: For N alternatives, one Boolean per alternative with `sum(literals) = 1`. Exactly one alternative is chosen.

**ConstChoice** (e.g. Load vs AddImm vs ALU-derived): One op index; alternatives are Load, AddImm, and optionally several ALU-derived encodings. **Binary**: One Boolean `choice_var`; Load interval optional with presence `¬choice_var`, AddImm/Flow interval optional with presence `choice_var`. **N-way**: N Booleans `use_vars[0..N]` with `sum(use_vars) = 1`; each alternative has an optional interval with presence `use_vars[k]`. Precedence for ALU option k is added with `only_enforce_if(..., [use_alu_k])`. Ref: `model.rs` ConstChoice block (~826–935).

**SelectChoice** (vselect vs FMA): One Boolean `choice_var`: true ⇒ FMA path (VALU), false ⇒ vselect (Flow). Optional Flow interval with presence `¬choice_var`; optional VALU intervals (SUB + FMA) with presence `choice_var`. Precedence for FMA is conditional on the path. Ref: `model.rs` SelectChoice block (~978–1085).

**VALU shift** (VALU vs ALU): One Boolean per shiftable op: stay on VALU or shift to ALU. Optional VALU interval (presence = ¬shift), optional ALU intervals (8, one per lane) with presence = shift. Precedence for consumers uses completion_cycle when shifted; constraints use `only_enforce_if(..., [shift_var])` or the opposite literal as appropriate. Ref: `model.rs` shiftable VALU / completion_cycle (~540–595, 637–675).

**CSE shared SUB**: Already specified in §7 and §3.6. One Boolean `sub_active` per group; optional SUB interval and optional scratch/liveness constraints with `only_enforce_if(..., [sub_active_var])`. Ref: `model.rs` CSE blocks.

---

### 11. References and Verification

- `scheduler_rs/src/types.rs`: LiveInterval struct, overlaps()
- `scheduler_rs/src/address_assign.rs`: compute_live_intervals, assign_addresses, expiration logic
- `scheduler_rs/src/cpsat_scheduler/model.rs`: Scratch cumulative constraint; Flow cumulative and NoOverlap (§9); choice encoding — optional intervals, only_enforce_if (§10)
- `scheduler_rs/src/cpsat_scheduler/bounds.rs`: ASAP/ALAP computation, tightening, capacity lower bound (T14)
- `scheduler_rs/src/cpsat_scheduler/cse_groups.rs`: CSE group construction
- `docs/ISA_REFERENCE.md`: VLIW timing, read-before-write
- `docs/SUB_CSE_ALU_SHIFT_DESIGN.md`: CSE shared SUB model and capacity

**Z3 verification** (`verification/verify.py`, `verification/theorems.py`): Encode the negation of each theorem (and any axioms used); if UNSAT, the theorem follows. Use **integer** cycle indices and half-open intervals; no need to model within-cycle read/write order explicitly. Scope of what is proven: **T3/T4** use a symbolic `offset ≥ 1`, so the precedence bounds are proven for any minimum delay (normal, FMA, shifted). **T5** uses a 2-op chain with symbolic makespan `K ≥ 2`; soundness for arbitrary DAGs follows from T3, T4, and the ASAP/ALAP fixpoint definition. All of T1–T14 are checked by refutation.

---

## Not Yet Included

The following CP-SAT constraints or model aspects are not yet formalized in this spec:

- **Symmetry breaking** (ordering groups, lexicographic streams) — search heuristics; prune equivalent solutions but don't affect correctness
- **Segment mode** — live-out, live-through, segment_end for per-section scheduling
- **Dead VReg exclusion** — CP-SAT scratch model omits dead VRegs (assumes DCE); address assign allocates them
