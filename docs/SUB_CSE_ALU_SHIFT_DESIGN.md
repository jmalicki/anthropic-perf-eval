# SUB CSE and ALU-Shift: Design Breakdown

A structured breakdown of the problem, tensions, and a unified CP-SAT model for SUB CSE + SUB ALU-shift + vselect/FMA choice.

---

## 1. The Three Optimizations

| Optimization | What | Current state |
|--------------|------|---------------|
| **vselect vs FMA** | Per SelectChoice: vselect (1 Flow) or FMA (1 SUB + 1 FMA = 2 VALU) | CP-SAT chooses; modeled correctly |
| **SUB CSE** | Multiple SelectChoices with same (true_val, false_val) share one SUB | Not done; 120 redundant SUBs |
| **SUB ALU-shift** | SUB on VALU (1 slot) or ALU (8 slots) when VALU saturated | SUB created at lowering; scheduler never sees it |

---

## 2. The Tensions

### Tension A: CSE vs vselect choice

- **CSE at emission:** Emit one shared SUB per (true_val, false_val). Consumers must use FMA (they need the SUB result). We force FMA and lose vselect for those consumers.
- **CSE at lowering:** After CP-SAT, we know who chose FMA. We can CSE among them. But CP-SAT never saw the reduced op set, so it didn't optimize for it.

### Tension B: SUB visibility vs vselect choice

- **Pre-expand SUB at emission:** SUB exists as SsaOp; CP-SAT can schedule and ALU-shift it. But we've committed to FMA (SUB must exist).
- **Keep SUB implicit:** CP-SAT chooses vselect vs FMA. SUB only appears at lowering. Scheduler never sees SUB, so no ALU-shift.

### Tension C: Who decides what, when

```
Emission          CP-SAT              Lowering
--------          ------              --------
emit SelectChoice -> choose vselect   -> emit vselect
                   or FMA             -> or emit SUB+FMA
```

CSE and SUB ALU-shift need the scheduler to see and place SUBs. The scheduler runs during CP-SAT. So SUB must exist (or be modeled) in the CP-SAT phase. But vselect vs FMA is also decided there. We need both in the same model.

---

## 3. Design Goal: Unified Model

**Requirement:** CP-SAT decides vselect vs FMA, SUB CSE, and SUB ALU-shift in one solve. No post-hoc hacks.

---

## 4. Model Structure

### 4.1 Groups (computed at emission)

For each unique `(true_val, false_val)` among SelectChoice ops (with `needs_subtraction`), form a **CSE group**:
- Group G = set of SelectChoice op indices with that pair
- Example: 32 groups share (values[1], values[0]) → one CSE group of size 32

Exclude SelectChoices with `delta_vreg` (they use 1 FMA, no SUB).

### 4.2 Per-group shared SUB

For each CSE group G:
- **Shared SUB:** One op that computes `sub_temp = true_val - false_val`
- **Active when:** At least one consumer in G chooses FMA
- **Consumers:** Each SelectChoice in G; if it chooses FMA, it has an FMA that reads `sub_temp`

### 4.3 Variables (per CSE group G)

| Variable | Type | Meaning |
|----------|------|---------|
| `choice[c]` | Bool | Consumer c chooses FMA (1) or vselect (0). Already exists. |
| `sub_active` | Bool | At least one consumer chose FMA. `sub_active <=> (sum over c in G of choice[c]) >= 1` |
| `sub_cycle` | Int | Cycle when shared SUB runs (only meaningful when sub_active) |
| `sub_shift` | Bool | When sub_active: SUB on ALU (1) or VALU (0). Like existing shift vars. |

### 4.4 Constraints

**Activity:**
```
sub_active == (sum(choice[c] for c in G) >= 1)
```
Use `add_ge` on sum; or `sub_active == max_eq(1, choice[c])` if available. Need one auxiliary bool that is 1 iff any choice[c] is 1.

**Precedence (for each consumer c in G):**
```
If choice[c] == 1:  sub_cycle <= fma_cycle[c] - 1
```
Use `only_enforce_if(choice[c])` so we only enforce when c chose FMA.

**SUB placement:**
- When sub_active and sub_shift=0: SUB uses 1 VALU slot at sub_cycle
- When sub_active and sub_shift=1: SUB uses 8 ALU slots at sub_cycle (element-wise)

Same pattern as existing shiftable VALU ops: optional VALU interval, optional ALU interval, exactly one when active.

**FMA cycles:**
- Each consumer c has `fma_cycle[c]` (already modeled when choice[c]=1). FMA is at `sub_cycle + 1` for the "canonical" consumer, but in fact each FMA has its own cycle. The only constraint is `sub_cycle <= fma_cycle[c] - 1` for each FMA consumer. So the shared SUB can run as late as `min(fma_cycle[c]) - 1` over active consumers. The solver picks sub_cycle anywhere in [ASAP, min(fma_cycle)-1].

### 4.5 Capacity constraints

**VALU:** Count 1 for shared SUB when sub_active and sub_shift=0. Count 2 for each consumer FMA (SUB+FMA in the current per-op model). But wait — with shared SUB we have:
- 1 SUB (when sub_active, not shifted)
- N FMAs (one per consumer with choice=1)

So VALU count = (1 if sub_active and not sub_shift else 0) + 2 * (number of consumers with choice=1). Need to adjust the global VALU capacity formula to account for shared SUBs vs per-op SUBs.

**Current model:** Each SelectChoice FMA contributes 2 VALU (sub + fma). So `valu_count += 2 * num_select_fma`.

**With shared SUBs:** For each CSE group G, we have:
- 1 SUB if sub_active and not shifted
- len(FMA consumers) FMAs
- Total VALU from this group = (1 if sub_active and not sub_shift else 0) + len(FMA consumers)

And we must not double-count: the current model adds 2 per SelectChoice FMA. With CSE we add 1 per group (SUB) + 1 per FMA consumer. So we need to change how we count: instead of 2 per SelectChoice FMA, we use the shared-SUB-aware formula.

### 4.6 Scratch / liveness

The shared SUB writes one `sub_temp` VReg. Its live range is [sub_cycle, max(fma_cycle[c])+1] over consumers with choice[c]=1. Address assignment and scratch constraints must account for this. The current scratch model uses intervals per VReg; we'd add sub_temp with the appropriate interval when sub_active.

---

## 5. SUB ALU-Shift: Detailed Design

SUB ALU-shift is a difficult addition. This section works through the modeling choices, element-wise scheduling, and tradeoffs.

### 5.1 Two Placement Modes

When the shared SUB is active, it can run in two ways:

| Mode | Engine | Slots | Cycles | Notes |
|------|--------|-------|--------|-------|
| **VALU** | VALU | 1 | 1 | Simple: one cycle, one slot. Same as current per-op SUB. |
| **ALU** | ALU | 8 (element-wise) | 1–8 (per lane) | Complex: 8 scalar SUBs, each can be scheduled independently. |

The VALU case is straightforward: one optional interval at `sub_cycle`, enabled when `sub_active & !sub_shift`. The ALU case requires the same treatment as existing shiftable VALU ops: per-lane cycles, completion cycle, and element-wise precedence.

### 5.2 How Existing Shiftable VALU Ops Work

From [scheduler_rs/src/cpsat_scheduler/model.rs](scheduler_rs/src/cpsat_scheduler/model.rs):

- **shift_var** (BoolVar): 1 = on ALU, 0 = on VALU
- **valu_optional_interval**: present when `!shift_var`; uses `cycle_vars[idx]`, size 1
- **alu_cycle_vars**: 8 IntVars, one per lane; bounds `[op_asap, op_alap]`
- **alu_optional_intervals**: 8 optional intervals, one per lane; each present when `shift_var`; start = `alu_cycle_vars[i]`, size 1
- **completion_cycle_var**: `max(alu_cycle_vars[0..8])`; used for makespan and consumer precedence when shifted
- **Per-lane precedence**: For each lane i and each dependency dep: `alu_cycle[op][i] >= alu_cycle[dep][j] + 1` (when dep shifted) or `alu_cycle[op][i] >= cycle_vars[dep] + 1` (when dep not shifted). The mapping `(op, dep) -> j` is in `element_dep_map`; default is same-lane (i -> i).
- **Consumer precedence**: When a consumer depends on a shifted op, it uses `completion_cycle` (not `cycle_vars`) for the offset.

### 5.2a Reuse vs Redesign

**Reuse the existing element-wise shift infrastructure.** The shared SUB is a shiftable op with an extra presence gate (`sub_active`). Do not redesign; parameterize what exists.

**Strategy:**
1. Extract the existing shift constraint-building into a helper: `build_shift_constraints_for_entity(cycle_var, deps, asap, alap, gate, shift_var, ...)` — gate is a BoolVar; when false, no intervals. For normal shiftable ops, gate = always true.
2. For the shared SUB: call the helper with `gate = sub_active[g]`, `shift_var = sub_shift[g]`, `cycle_var = sub_cycle[g]`, deps = producer indices from CseGroup. Same element_dep_map (same-lane from producers).
3. The helper creates: optional VALU interval (present when gate & !shift_var), 8 optional ALU intervals (present when gate & shift_var), completion_cycle, per-lane precedence. The only structural change is the extra gate.

### 5.3 SUB as a Synthetic Shiftable Op

The shared SUB has no SsaOp; it is a synthetic op. To model ALU-shift we treat it like a shiftable VALU op:

- **Inputs:** `true_val`, `false_val` (both vectors, 8 elements each)
- **Output:** `sub_temp` (vector)
- **Dependencies:** The defining ops of `true_val` and `false_val`. We need `vreg_def` or equivalent to map VReg -> producer op index.

**CseGroup must carry dependency info:**
- `true_val`, `false_val`
- `consumer_indices`
- `producer_true_idx: Option<usize>` — op index that defines true_val (if unique)
- `producer_false_idx: Option<usize>` — op index that defines false_val

If true_val or false_val has multiple producers (e.g. phi), the model is more complex; for tree-select they are single-def.

### 5.4 Per-Lane Dependencies for SUB

For lane i: `sub_temp[i] = true_val[i] - false_val[i]`. Lane i depends on:
- Producer of `true_val[i]`: typically the same op for all i (vector op). If that op is shiftable, lane i depends on `alu_cycle[producer_true][i] + 1`. If not, `cycle_vars[producer_true] + 1`.
- Producer of `false_val[i]`: same pattern.

So:
- `sub_alu_cycle[g][i] >= max(ready_true[i], ready_false[i])`

Where `ready_true[i]` = `alu_cycle[producer_true][i] + 1` (if producer shifted) or `cycle_vars[producer_true] + 1` (if not). Same for false.

**Element dep map:** Default same-lane (i -> i). Tree-select operands are full vectors from one op, so lane i of SUB depends on lane i of each producer.

### 5.5 Precedence to Consumers (FMAs)

Each FMA consumer runs as a single VALU op at `fma_cycle[c]`. It reads all 8 elements of `sub_temp`. So:
- When SUB on VALU: `sub_cycle + 1 <= fma_cycle[c]`
- When SUB on ALU: `completion_sub_cycle + 1 <= fma_cycle[c]` (all 8 lanes must be done)

Both are conditional on `choice[c] = 1` and (implicitly) `sub_active`. We need:
- `add_ge(fma_cycle[c], sub_cycle[g] + 1)` with `only_enforce_if([choice[c], !sub_shift[g]])`
- `add_ge(fma_cycle[c], completion_sub_cycle[g] + 1)` with `only_enforce_if([choice[c], sub_shift[g]])`

When `sub_shift=1`, `sub_cycle` is still defined (we can use it for the non-shifted case), but the constraint that matters for FMA precedence is `completion_sub_cycle`. So we need both `sub_cycle` and `completion_sub_cycle` when ALU-shift is enabled.

### 5.6 Optional Gating for ALU Path

When `sub_shift=1`, we create:
- 8 `sub_alu_cycle_vars[g][i]` with bounds `[sub_asap, sub_alap]`
- `completion_sub_cycle[g] >= max(sub_alu_cycle_vars[g])`
- 8 optional ALU intervals, each present when `sub_active & sub_shift`
- Per-lane precedence from producers, enforced only when `sub_shift`

The difficulty: `sub_active & sub_shift` must gate the ALU intervals. OR-Tools optional intervals take a single BoolVar. We need an auxiliary `sub_on_alu[g] = sub_active[g] & sub_shift[g]`.

### 5.7 ASAP/ALAP for SUB

- **sub_asap:** `max(asap[producer_true], asap[producer_false]) + 1`. When producers are shiftable, their ASAP is the earliest cycle any lane is ready; conservatively we use the op-level ASAP.
- **sub_alap:** When SUB is on ALU, it must complete before the earliest FMA. So `sub_alap <= min(fma_cycle[c]) - 1` over consumers with choice[c]=1. But choice is a variable. We can use a generous bound: `horizon - 1`, and let the precedence constraints tighten it.

### 5.8 Scratch / Liveness When Shifted

When SUB is on ALU, each lane writes at a potentially different cycle. The live range of `sub_temp` is:
- Start: `min(sub_alu_cycle[g][0..8])` — when the first lane is written
- End: `max(fma_cycle[c]) + 1` over consumers — when the last FMA reads

For scratch, we need an interval that spans [start, end). The start is variable (min of 8 IntVars). The end is variable (max of FMA cycles). The scratch model uses optional intervals with fixed start/end or variable end. This may require extending the scratch formulation to support variable start for `sub_temp` when shifted.

### 5.9 Fallback: Atomic ALU SUB

If reusing the element-wise helper proves too complex (e.g. scratch/liveness for variable start), we could fall back to **atomic** ALU: one cycle, 8 ALU slots. All 8 scalar SUBs run in the same cycle. No per-lane vars, no completion_cycle. Precedence: `sub_cycle + 1 <= fma_cycle[c]`. Less flexible but simpler. Prefer reuse (5.2a) first.

### 5.10 Implementation Order for ALU-Shift

1. **Phase A (no ALU-shift):** Implement shared SUB CSE only. SUB always on VALU. Validates the CSE model.
2. **Phase B (ALU-shift with reuse):** Extract `build_shift_constraints_for_entity` from the existing shiftable-op loop. Add `sub_shift` and call the helper for each CSE group with `gate = sub_active`. Reuses full element-wise scheduling (no need for atomic-only first).

---

## 6. Edge Cases and Pitfalls

### 6.1 Single-consumer groups

If a CSE group has size 1, there is no CSE benefit. We still need a SUB when that consumer chooses FMA. The model reduces to: one optional SUB (when choice=1), same as current per-SelectChoice SUB. So the shared-SUB model subsumes the current one; single-consumer groups are fine.

### 6.2 ASAP/ALAP for sub_cycle

sub_cycle must satisfy:
- `sub_cycle >= earliest_cycle` where (true_val, false_val) are both defined
- `sub_cycle <= fma_cycle[c] - 1` for each c with choice[c]=1

The second is conditional on choice[c]. For ASAP: we need the earliest cycle at which both operands are ready. For ALAP: when choice[c]=1, sub_cycle <= fma_cycle[c]-1; the tightest bound is min over active consumers. But choice[c] is a variable, so ALAP is dynamic. The solver will enforce the constraints; we may need generous bounds for sub_cycle (e.g. [0, horizon-1]) or a more careful formulation.

### 6.3 Makespan and "completion" for the group

The group's contribution to makespan: the last operation is the last FMA (max of fma_cycle[c] for active c). The SUB runs earlier. The current makespan formula uses `max(cycle[i] + 1)` over ops. For shared SUB, we don't add a separate completion — the FMAs already extend makespan. The SUB just needs to run before them. So no special makespan handling for the SUB itself.

### 6.4 Greedy hints

Greedy does not need to optimize, but add simple hints for sub_cycle and sub_shift: sub_cycle = ASAP for the group; sub_shift = 0 (VALU) by default, or a cheap heuristic (e.g. prefer ALU when VALU pressure estimate is high). Keeps CP-SAT solve time reasonable.

### 6.5 Lowering

After solve:
- For each CSE group G with sub_active=1: emit one SUB at sub_cycle, one FMA per consumer with choice=1 at their fma_cycle.
- For each consumer with choice=0: emit vselect.

The lowering logic must map shared SUB result to the correct sub_temp VReg and pass it to each FMA. Address assignment must assign sub_temp and ensure it's live for [sub_cycle, max_fma_cycle+1].

---

## 7. Data Structures and Constraint Representation

### 7.1 CSE Groups (Distinct Pass)

**Where:** A distinct pass, not inside the CP-SAT model builder. Input: `ops: &[SsaOp]`. Output: `Vec<CseGroup>`. Called from the scheduling orchestration (e.g. tick_tock_ssa) before `schedule_ssa_cpsat`; the result is passed in. Keeps grouping separate from modeling and allows reuse (lowering, validation, analysis).

**Structure:**
- `CseGroup { true_val: VReg, false_val: VReg, consumer_indices: Vec<usize>, producer_true_idx: Option<usize>, producer_false_idx: Option<usize> }` — one struct per unique (true_val, false_val). Producer indices needed for ALU-shift per-lane precedence (5.3).
- `cse_groups: Vec<CseGroup>` — ordered list. Index `g` identifies the group.
- Build via `HashMap<(VReg, VReg), Vec<usize>>`: for each SelectChoice op index `idx` with `Opcode::SelectChoice`, `sub_temp_vreg.is_some()`, `delta_vreg.is_none()`, extract `(true_val, false_val)` from `ops[idx].srcs` (indices 1 and 2). Insert `idx` into the map under that pair. Convert map to `Vec<CseGroup>`.

**Key:** VReg must implement `Hash`/`Eq` for the map. Existing `VReg` (Scalar/Vector with id) is suitable.

### 7.2 Per-Group CP-SAT Variables

**Maps keyed by group index `g`:**

| Variable | Type | Storage | Bounds |
|----------|------|---------|--------|
| `sub_active[g]` | BoolVar | `HashMap<usize, BoolVar>` or `Vec<Option<BoolVar>>` | — |
| `sub_cycle[g]` | IntVar | Same | `[sub_asap_g, horizon-1]` where sub_asap = earliest cycle both operands ready |
| `sub_shift[g]` | BoolVar | Same | — |

**Linking to existing:**
- `select_choice_vars: HashMap<usize, BoolVar>` — op index -> choice (already exists). For consumer `c` in group `g`, `choice_var[c]` = SelectChoice FMA (true) or vselect (false).
- `select_choice_fma_last_cycle: HashMap<usize, IntVar>` — for no-delta SelectChoice, FMA runs at `cycle_vars[c] + 1`. So `fma_cycle[c]` = `cycle_vars[c] + 1` (or we use the existing `fma_cycle_var` when present).

### 7.3 Constraint Representation

**Activity (`sub_active <=> at least one consumer chose FMA`):**
- Sum: `sum_g = sum over c in group g of choice_var[c]`. CP-SAT: `LinearExpr::sum(choice_vars)`.
- `sub_active[g] <=> (sum_g >= 1)`: 
  - `sum_g >= 1 => sub_active[g]`: add_ge(sum_g, 1).only_enforce_if(sub_active[g]). No — that enforces sum>=1 when sub_active. We need the reverse: when sum>=1, sub_active must be true.
  - Standard encoding: `sub_active[g] = 1` iff `sum_g >= 1`. Use `model.add_ge(LinearExpr::from(sum_g), 1)` and `model.add_eq(sub_active[g], max_eq(1, sum_g))` — but max_eq may not exist. Alternative: `sub_active[g] >= sum_g / M` (big-M) or use auxiliary vars. Simpler: `add_ge(sum_g, 1)` only when sub_active; `add_eq(sum_g, 0)` only when !sub_active. That forces the equivalence.
  - Concretely: (a) `model.only_enforce_if(add_ge(sum_g, 1), [sub_active[g]])` — when sub_active, sum>=1. (b) `model.only_enforce_if(add_eq(sum_g, 0), [sub_active[g].not()])` — when !sub_active, sum=0. That gives the bidirectional link.

**Precedence (`sub_cycle <= fma_cycle[c] - 1` when consumer c chose FMA):**
- For each consumer `c` in group `g`: `sub_cycle[g] + 1 <= fma_cycle[c]`, i.e. `fma_cycle[c] >= sub_cycle[g] + 1`.
- `fma_cycle[c]` = `cycle_vars[c] + 1` for no-delta SelectChoice (FMA at cycle+1).
- Constraint: `add_ge(cycle_vars[c] + 1, sub_cycle[g] + 1)` => `add_ge(cycle_vars[c], sub_cycle[g])`.
- Conditional: enforce only when `choice_var[c] = 1`. Use `model.only_enforce_if(constraint, [choice_var[c]])`.

**SUB placement (VALU vs ALU):**
- When `sub_active[g]` and `!sub_shift[g]`: SUB contributes 1 VALU slot at `sub_cycle[g]`. Optional interval: `new_optional_fixed_size_interval_var(sub_cycle[g], 1, sub_active[g] & !sub_shift[g])`. CP-SAT needs "interval present when (A and B)" — use an auxiliary bool `sub_on_valu[g] = sub_active[g] & !sub_shift[g]`, or two optional intervals with disjoint presence.
- When `sub_active[g]` and `sub_shift[g]`: SUB contributes 8 ALU slots (element-wise). Same as shifted VALU: 8 optional ALU intervals, one per lane, each present when `sub_active & sub_shift`. Existing shiftable ops use `alu_cycle_vars` — we could add 8 IntVars for sub_cycle or a single cycle with demand 8. Check how ALU cumulative works: it uses `add_cumulative(intervals, demands, cap)`. For element-wise, each lane gets demand 1; 8 intervals at the same start. So we add 8 optional intervals at `sub_cycle[g]` (or per-lane if we want element-wise flexibility), enabled when `sub_active & sub_shift`.
- Exactly one of VALU or ALU when active: `sub_shift[g]` already encodes that. When sub_active, exactly one of (sub_on_valu, sub_on_alu) is true.

### 7.4 Capacity Formula Changes

**Current:** `valu_count` includes `2 * sum(choice_var[c])` for each SelectChoice FMA (sub + fma). Stored in `valu_count` before choice-dependent terms; then `6*makespan >= valu_count - sum(shift) + 2*sum(select_fma)`.

**With shared SUBs:** For SelectChoice ops that belong to CSE groups, we must not count 2 per FMA. Instead:
- Base `valu_count` excludes the per-SelectChoice FMA contribution (the +2 per FMA).
- For each CSE group `g`: add `(1 if sub_active and !sub_shift else 0) + num_fma_consumers` to VALU. `num_fma_consumers = sum(choice_var[c] for c in g)`.
- The capacity constraint becomes: `6*makespan >= valu_count_base + sum over g of [(1 - sub_shift[g]) * sub_active[g] + sum(choice[c])]` — but sub_active and sub_shift are vars. So we need a linear expression. `sub_contrib = sub_active * (1 - sub_shift)` in reals; for int: `aux_g = sub_active & !sub_shift`, then add `aux_g` to the sum. And add `sum(choice[c])` for consumers in that group.
- Implementation: build a `LinearExpr` for the total VALU from SelectChoice: for each group g, add `sub_active_and_not_shifted[g] + sum(choice[c] for c in consumers of g)`. The first term is one auxiliary BoolVar per group.

### 7.5 Result Extraction

**After solve, we need:**
- `select_fma_ops: HashSet<usize>` — op IDs that chose FMA (existing).
- **New:** `cse_group_results: HashMap<usize, CseGroupResult>` where key = group index, value = `{ sub_active: bool, sub_cycle: i32, sub_shift: bool }`. Or a flat structure: `Vec<(bool, i32, bool)>` indexed by group.
- Lowering uses this: for each group with sub_active, emit one SUB at sub_cycle (on VALU or ALU per sub_shift), and one FMA per consumer with choice=1.

### 7.6 Integration Points in Code

| Location | What to add |
|----------|-------------|
| New pass: `build_cse_groups(ops)` | Compute and return `Vec<CseGroup>`. Called before CP-SAT. |
| Orchestration (tick_tock_ssa) | Call `build_cse_groups`, pass result to `schedule_ssa_cpsat`. |
| `schedule_ssa_cpsat` (model build) | Accept `cse_groups` as input. Skip shared-SUB logic if empty. |
| Same | For each CSE group: create sub_active, sub_cycle, sub_shift; add activity, precedence, placement constraints. |
| Same | Adjust VALU cumulative: remove per-SelectChoice FMA intervals for ops in CSE groups; add shared SUB intervals. Adjust capacity formula. |
| Scratch/liveness | For each group with sub_active, add sub_temp interval [sub_cycle, max_fma_cycle+1]. Need to handle optional presence. |
| Solution extraction | Read sub_active, sub_cycle, sub_shift from solver; store in result. |
| `SsaCpsatResult` | Add `cse_group_results: Option<Vec<CseGroupResult>>` or similar. |
| Lowering | When processing SelectChoice in CSE group: if sub_active, use shared SUB; emit only FMA (no per-op SUB). |

### 7.7 Backward Compatibility

When `cse_groups` is empty (no shared operands), behavior must match the current model. Options:
- Run the new group logic only when `cse_groups.len() > 0`; otherwise use the existing per-SelectChoice SUB model.
- Or always use the shared-SUB model: groups of size 1 are single-consumer groups; the formulation reduces to the current one (one optional SUB per SelectChoice when choice=FMA).

---

## 8. Implementation Phases

| Phase | Scope | Notes |
|-------|-------|------|
| 1 | Build CSE groups (distinct pass) | See 7.1; include producer indices for ALU-shift |
| 2 | Add shared SUB variables and activity constraint | sub_active, sub_cycle; sub_shift deferred to Phase B |
| 3 | Add precedence constraints | sub_cycle <= fma_cycle[c]-1 when choice[c]=1 |
| 4 | Add SUB to VALU cumulative | Adjust capacity formula; no ALU yet |
| 5 | Scratch/liveness for sub_temp | Extend address_assign for shared SUB |
| 6 | Lowering | Emit one SUB per active group, N FMAs |
| 7 | Validation | Correctness and cycle comparison |
| 8 | **Phase B:** Extract `build_shift_constraints_for_entity` | Refactor existing shift loop into reusable helper (gate param) |
| 9 | **Phase B (cont):** Add sub_shift, call helper for CSE groups | Reuse element-wise ALU; see 5.2a |

---

## 9. Open Questions

1. **CP-SAT expressiveness:** Can we cleanly express `sub_active = (sum >= 1)` and conditional precedence? OR-Tools supports `only_enforce_if` and `add_ge`; need to verify the pattern.
2. **Solve time:** More variables and constraints per CSE group. Does it blow up for 37 groups?
3. **Delta-path SelectChoices:** Excluded from CSE (no SUB). Confirm they stay unchanged.
4. **Hints:** Greedy doesn't optimize, but add simple hints: sub_cycle = ASAP for the group; sub_shift = 0 (VALU) by default, or a cheap heuristic (e.g. prefer ALU when VALU pressure estimate is high). Keeps solve time reasonable without extending greedy.
5. **ALU cumulative demand:** Does OR-Tools `add_cumulative` support demand > 1 per interval for atomic ALU SUB? Or do we need 8 intervals at the same start?

---

## 10. Alternatives (Not Chosen)

| Alternative | Why not |
|-------------|---------|
| CSE at lowering | CP-SAT never sees reduced ops |
| Two-phase solve | Extra complexity; want single solve |
| CSE at emission (force FMA) | Loses vselect option for shared operands |
| Pre-expand SUB without CSE | Each op gets its own SUB; no CSE benefit |
