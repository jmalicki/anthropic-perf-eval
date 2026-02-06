# CP-SAT Solving Speed Improvements

This document describes proposed improvements to help CP-SAT find feasible solutions faster and propagate variable domains more effectively. The primary goal is to reduce time to first feasible solution from >1hr to <10 minutes (6-10x improvement).

## Problem Statement

When CP-SAT takes an hour to find a feasible solution (`best:inf`), it indicates weak propagation and domain tightening. The solver isn't effectively pruning infeasible assignments early.

**Quantitative problem description:**
- **Current state**: CP-SAT takes >1hr before finding first feasible solution (`best:inf` for extended periods)
- **Problem scale**: ~11,776 ops, ~5,120 VALU ops, horizon ~2,400 cycles, ~2,000-3,000 precedence constraints, 5 cumulative constraints
- **Root cause**: Optional intervals in cumulative constraints propagate weakly until choice variables are fixed, creating a "propagation dead zone"
- **Search space**: With many choice variables (shift_var, const_choice_var, select_choice_var), search space is exponential (2^k where k = number of choice variables). Without early propagation, solver explores many infeasible branches.
- **Target**: Reduce time to first feasible solution from >1hr to <10 minutes (6-10x improvement)

## Current Constraint Structure

**Key characteristics:**

- **Many optional intervals**: Shift (VALU→ALU), ConstChoice (Load vs AddImm vs ALU-derived), SelectChoice (FMA vs vselect). See optional interval creation in [`scheduler_rs/src/cpsat_scheduler.rs`](scheduler_rs/src/cpsat_scheduler.rs) around lines 2000-2300.
- **5 cumulative constraints**: One per engine (ALU 12, VALU 6, Load 2, Store 2, Flow 1). See [`model.add_cumulative` calls](scheduler_rs/src/cpsat_scheduler.rs) at lines [2411](scheduler_rs/src/cpsat_scheduler.rs#L2411) (VALU), [2447](scheduler_rs/src/cpsat_scheduler.rs#L2447) (ALU), [2478](scheduler_rs/src/cpsat_scheduler.rs#L2478) (Load), [2497](scheduler_rs/src/cpsat_scheduler.rs#L2497) (Store), [2579](scheduler_rs/src/cpsat_scheduler.rs#L2579) (Flow).
- **1 NoOverlap constraint**: For Flow (capacity 1, redundant with cumulative but helps propagation). See [`model.add_no_overlap`](scheduler_rs/src/cpsat_scheduler.rs#L2581) at line 2581.
- **Many boolean choice variables**: `shift_var`, `const_choice_var`, `select_choice_var`. See choice variable creation around lines 1800-2000 in [`scheduler_rs/src/cpsat_scheduler.rs`](scheduler_rs/src/cpsat_scheduler.rs).
- **Progress comes from LNS**: `scheduling_time_window_lns`, `graph_cst_lns`, `rnd_var_lns`, `scheduling_intervals_lns` find most improvements; core MIP subsolvers (`default_lp`, `no_lp`, `max_lp_sym`) mostly improve bounds

**Current parameters:** See [`SatParameters` configuration](scheduler_rs/src/cpsat_scheduler.rs#L3746) around line 3746:

- `linearization_level=2` (stronger LP relaxations)
- `cp_model_presolve=true` (preprocessing)
- `cp_model_probing_level=2` (aggressive domain tightening)
- `randomize_search=true` (escape local optima)
- `num_workers=8` (parallel search)

## Constraint Structure Analysis: Propagation Weaknesses

Based on analysis of [`scheduler_rs/src/cpsat_scheduler.rs`](scheduler_rs/src/cpsat_scheduler.rs), here are identified propagation weaknesses:

### 1. Cumulative Constraints with Many Optional Intervals

**Structure:**
- **VALU cumulative** ([line 2411](scheduler_rs/src/cpsat_scheduler.rs#L2411)): Contains optional intervals from:
  - Shiftable VALU ops (when `shift_var=false`, VALU interval is optional)
  - SelectChoice FMA path (when `select_choice_var=true`, FMA VALU intervals are optional)
  - Eliminable vbroadcasts (when `eliminate_vb=true`, vbroadcast interval is optional)
- **ALU cumulative** ([line 2447](scheduler_rs/src/cpsat_scheduler.rs#L2447)): Contains optional intervals from:
  - Shifted VALU ops (8 intervals per op when `shift_var=true`, all optional)
  - ConstChoice ALU-derived encodings (when ALU encoding chosen)
- **Load cumulative** ([line 2478](scheduler_rs/src/cpsat_scheduler.rs#L2478)): Contains optional intervals from:
  - ConstChoice Load path (when `const_choice_var=false`)
- **Flow cumulative** ([line 2579](scheduler_rs/src/cpsat_scheduler.rs#L2579)): Contains optional intervals from:
  - ConstChoice AddImm path (when `const_choice_var=true`)
  - SelectChoice vselect path (when `select_choice_var=false`)

**Propagation Weakness:**
- **Optional intervals prevent early pruning**: Cumulative constraints with many optional intervals propagate weakly because the solver doesn't know which intervals will be present until choice variables are fixed.
- **Example**: If 100 shiftable VALU ops can start at cycle 10, and VALU capacity is 6, the cumulative constraint can't prune cycle 10 until it knows which ops are shifted (VALU vs ALU). This requires fixing 100 `shift_var` boolean variables.
- **Impact**: With ~5,120 VALU ops total (from [docs/PROBLEM_CONSTRAINTS.md](docs/PROBLEM_CONSTRAINTS.md)), a significant fraction may be shiftable, creating many optional intervals in cumulative constraints.

### 2. Choice Variables Fixed Too Late

**Structure:**
- **Shift variables** ([line 1830](scheduler_rs/src/cpsat_scheduler.rs#L1830)): `shift_var` for each shiftable VALU op
- **ConstChoice variables** ([line 2114](scheduler_rs/src/cpsat_scheduler.rs#L2114)): `const_choice_var` for each ConstChoice op (binary or N-way encoding)
- **SelectChoice variables** ([line 2200+](scheduler_rs/src/cpsat_scheduler.rs#L2200)): `select_choice_var` for each SelectChoice op

**Propagation Weakness:**
- **No explicit search strategy**: Choice variables are boolean (smallest domain), but CP-SAT's default search may not prioritize fixing them early.
- **Cascading effect**: Until choice variables are fixed, optional intervals remain "maybe present", preventing cumulative constraints from propagating effectively.
- **Impact**: The solver may explore many branches where choice variables are unfixed, wasting time on infeasible partial assignments.

### 3. Precedence Constraints Don't Consider Resource Capacity

**Structure:**
- Precedence constraints ([line 1731](scheduler_rs/src/cpsat_scheduler.rs#L1731)): `cycle[idx] >= cycle[dep_idx] + 1`
- Cycle variable domains ([line 1708](scheduler_rs/src/cpsat_scheduler.rs#L1708)): Initialized from ASAP/ALAP bounds

**Propagation Weakness:**
- **ASAP/ALAP ignore capacity**: [`compute_ssa_asap`](scheduler_rs/src/cpsat_scheduler.rs#L1439) and [`compute_ssa_alap`](scheduler_rs/src/cpsat_scheduler.rs#L1490) only consider precedence, not resource capacity.
- **Loose initial domains**: If 100 ops can all start in window [10, 20] from precedence, but capacity is 6, domains remain loose until cumulative constraints propagate (which is weak with optional intervals).
- **Impact**: Cycle variable domains may remain loose for many ops, creating a large search space.

### 4. Scratch Constraint Complexity

**Structure:**
- Scratch constraint ([line 2998](scheduler_rs/src/cpsat_scheduler.rs#L2998)): Cumulative constraint on VReg liveness intervals
- Variable-length intervals: Shifted VALU ops create 8 ALU intervals with different cycles; SelectChoice FMA creates interval of length 2

**Propagation Weakness:**
- **Complex liveness computation**: VReg liveness depends on:
  - Write cycle (varies for shifted ops: min of 8 ALU cycles)
  - Last read cycle (varies for SelectChoice FMA: cycle+1)
  - Optional intervals for SelectChoice FMA sub_temp_vreg
- **Weak propagation**: The cumulative constraint on variable-length intervals with optional presence may propagate weakly, especially when combined with choice variables.

### 5. Overall Constraint Interaction Issues

**Systemic Weaknesses:**
- **Optional intervals + cumulative = weak propagation**: The combination of many optional intervals in cumulative constraints creates a "propagation dead zone" where constraints can't prune until choice variables are fixed.
- **No redundant constraints**: The model relies entirely on cumulative constraints for resource capacity. There are no redundant constraints (like per-cycle bounds) that could help propagate earlier.
- **Choice variables not prioritized**: Boolean choice variables have the smallest domain but may not be prioritized in search, delaying when optional intervals become fixed.

**Why `best:inf` for an hour:**
- **Cumulative constraints with optional intervals can't prune infeasible assignments until choice variables are fixed**: With ~2,000+ shiftable VALU ops (estimated), ~const_choice_ops ConstChoice ops, ~select_choice_ops SelectChoice ops, there are thousands of choice variables. Until these are fixed, optional intervals remain "maybe present", preventing cumulative constraints from propagating.
- **Choice variables may not be fixed early in search (no explicit strategy)**: Boolean choice variables have smallest domain (2 values), but CP-SAT's default search may not prioritize fixing them early. With thousands of choice variables, search space is 2^k where k = number of unfixed choices.
- **Precedence constraints don't tighten domains based on capacity**: ASAP/ALAP bounds are loose (only consider precedence), so cycle variable domains remain large. With ~11,776 ops and average domain size ~100 cycles, search space is 100^11,776 (astronomical).
- **No redundant constraints to provide alternative propagation paths**: Model relies entirely on cumulative constraints for resource capacity. When cumulative constraints can't propagate (due to optional intervals), there's no alternative propagation path.
- **Result**: Solver explores many branches with loose domains and unfixed choices before finding feasibility. With exponential search space and weak propagation, finding feasibility takes >1hr.

## Proposed Improvements

### 1. ASAP/ALAP Tightening via Capacity (Currently Implementing)

**Idea:** Currently ASAP/ALAP only consider precedence. Tighten them by considering resource capacity - if too many ops need to run in a window, some must start earlier/later.

**Addresses propagation weakness #3**: Resource-aware ASAP/ALAP tightening directly addresses the issue that precedence-only bounds create loose initial domains. By tightening domains before creating cycle variables, we reduce the search space from the start.

**Resources Used:** The tightening currently uses **slot constraints only** (ALU, VALU, Load, Store, Flow capacities), but **scratch-aware tightening is also possible** and could provide additional tightening.

**Current Implementation (Slot Constraints Only):**
- Tightening happens before cycle variables are created (before scratch liveness is computed)
- Slot constraints are simpler to reason about (fixed capacity per engine per cycle)
- Slot capacity violations are easier to detect via energetic reasoning (count ops per engine in a window)

**Why Scratch Constraints Could Also Be Used:**
- **Slot constraints also depend on choices**: Shifted VALU ops (shift_var), ConstChoice (Load vs AddImm), SelectChoice (FMA vs vselect) all affect slot usage, just like scratch. So slot constraints have min/max bounds over choices too.
- **Scratch constraints have min/max bounds over choices**: For each cycle t, we can compute:
  - **Minimum scratch usage** (best case): Assume all choices minimize scratch:
    - ConstChoice: Use AddImm instead of Load (avoids temp VRegs)
    - SelectChoice: Use vselect instead of FMA (avoids sub_temp_vreg)
    - Shifted VALU: Assume earliest possible write cycle (min of ASAP bounds for the 8 ALU ops)
  - **Maximum scratch usage** (worst case): Assume all choices maximize scratch:
    - ConstChoice: Use Load instead of AddImm (creates temp VRegs)
    - SelectChoice: Use FMA instead of vselect (creates sub_temp_vreg)
    - Shifted VALU: Assume latest possible write cycle (max of ALAP bounds for the 8 ALU ops)
- **If min scratch > budget, bounds can be tightened**: If even in the best case (minimum scratch), we exceed the scratch budget at cycle t, then some ops MUST be scheduled earlier/later, regardless of choices. This can tighten ASAP/ALAP bounds further.
- **This could tighten bounds further**: Scratch-aware tightening could provide additional tightening beyond slot-aware tightening, especially when scratch is the bottleneck (e.g., when many VRegs are live simultaneously).

**Implementation Approach for Scratch-Aware Tightening:**
1. **After slot-aware tightening**: First tighten using slot constraints (current implementation)
2. **Compute min/max scratch usage**: For each cycle t in [0, horizon-1]:
   - Build liveness intervals assuming minimum scratch choices (best case)
   - Build liveness intervals assuming maximum scratch choices (worst case)
   - Compute scratch usage profile: `scratch[t] = sum of live VReg sizes at cycle t`
3. **Tighten if min scratch > budget**: For each cycle t where `min_scratch[t] > scratch_budget`:
   - Find ops that contribute to scratch at cycle t (VRegs that are live at t)
   - Tighten ASAP forward or ALAP backward to avoid cycle t
4. **Iterate**: Repeat until convergence (typically 1-2 passes)

**Computational Cost:**
- **Single pass cost**: O(n + horizon), same as slot-aware tightening:
  - Build `vreg_def_op` and `vreg_last_use`: O(n) - one pass through ops
  - Build scratch profile using difference arrays: O(n + horizon) - similar to slot profile
  - Check for violations: O(horizon) - scan profile once
  - Tighten bounds: O(n) - update ASAP/ALAP for affected ops
- **Iterative updates are cheap**: Each iteration is O(n + horizon), so multiple passes are still efficient
- **Convergence**: Typically 1-2 passes (similar to slot-aware tightening), so total cost is O(k × (n + horizon)) where k = 1-2
- **Comparison to slot-aware**: Same complexity! Scratch-aware tightening is no more expensive than slot-aware tightening

**Why a Single Pass Could Help a Lot:**
- **Scratch violations are "global"**: When scratch exceeds budget at cycle t, many VRegs are live simultaneously. Tightening bounds to avoid cycle t fixes a whole region, not just one op.
- **Even partial tightening helps**: Even if we don't fully resolve all scratch violations in one pass, tightening bounds reduces search space significantly.
- **No need for perfect convergence**: A single pass that tightens bounds by 10-20% could still provide significant benefit (exponential search space reduction).

**Implementation Challenges:**
- Scratch liveness depends on write cycles (def_cycle) and last read cycles (last_use_cycle), which depend on scheduling decisions
- Need to compute liveness intervals for different choice scenarios (min/max scratch)
- More complex than slot constraints (need to track VReg liveness, not just count ops), but same computational complexity
- Iterative updates are cheap (O(n + horizon) per iteration), so convergence is not a concern

**Future Enhancement:**
- Consider adding scratch-aware tightening as a second pass after slot-aware tightening
- Compute min/max scratch usage over choices for each cycle t
- If min scratch > budget at cycle t, tighten ASAP/ALAP bounds to avoid that cycle
- This could provide additional tightening when scratch is the bottleneck (e.g., when many VRegs are live simultaneously)
- Test on problems where scratch is tight (peak liveness near budget) to see if it helps

**Implementation (O(n + horizon) algorithm):**

1. For each engine (ALU, VALU, Load, Store, Flow), build resource profile using difference array:
   - Initialize `profile[engine][t] = 0` for all cycles t
   - For each op i on engine e: `profile[e][ASAP[i]]++`, `profile[e][ALAP[i]+1]--`
   - Compute prefix sums to get actual counts: `profile[e][t] = sum of ops on engine e that CAN start at t`
2. Build prefix sums of profile for O(1) window queries per engine
3. For each op i, check if window [ASAP[i], ALAP[i]] has enough capacity for its engine:
   - Query prefix sums: `total_capacity_needed = sum(profile[engine[i]][ASAP[i]..ALAP[i]])`
   - If `total_capacity_needed > capacity[engine[i]] * (ALAP[i] - ASAP[i] + 1)`, tighten window
4. Iterate until convergence (typically 1-2 passes)
5. Use tightened ASAP/ALAP to set cycle variable domains

**Tightening Algorithm Details:**

The tightening algorithm uses **energetic reasoning** to determine when a window is too small:

- For each op i with window [ASAP[i], ALAP[i]] on engine e:
  - Count total ops on engine e that CAN start in this window (using prefix sums of resource profile for engine e)
  - If total ops > capacity[e] × window_size, the window is too small
  - Tighten ASAP forward: Find earliest cycle t where there's enough capacity for all ops on engine e that MUST start by t
  - Tighten ALAP backward: Find latest cycle t where there's enough capacity for all ops on engine e that MUST start by t

**Location:** After [`compute_ssa_asap`](scheduler_rs/src/cpsat_scheduler.rs#L1439) and [`compute_ssa_alap`](scheduler_rs/src/cpsat_scheduler.rs#L1490) calls (around [lines 1636-1642](scheduler_rs/src/cpsat_scheduler.rs#L1636)), before creating cycle variables ([line 1708](scheduler_rs/src/cpsat_scheduler.rs#L1708)) in [`scheduler_rs/src/cpsat_scheduler.rs`](scheduler_rs/src/cpsat_scheduler.rs).

**Cost Analysis:**
- **Computation cost**: O(k × (n + horizon)) using efficient resource profile approach, where k = number of iterations (typically 1-2):
  1. Build resource profile: For each cycle t, count ops that CAN start at t (based on ASAP/ALAP). Use difference array for O(n) construction.
  2. Build prefix sums of profile for O(horizon) preprocessing.
  3. For each op i, check if window [ASAP[i], ALAP[i]] has enough capacity using prefix sum queries = O(1) per op = O(n) total.
  4. Tighten ASAP/ALAP iteratively (may need multiple passes, but typically converges in 1-2 iterations).
- **For ~11,776 ops, horizon ~2,400, k=2 iterations**: ~28K operations (2 × (11,776 + 2,400) = 28,352), dominated by n, not O(n²)
- **Preprocessing overhead**: One-time cost before model building, not per-solve. ~28K operations is negligible compared to model building (~O(n²) = ~138M operations for precedence constraints)
- **Memory overhead**: O(horizon) = ~2,400 integers for resource profile, negligible (~9.6KB)

**Risk Analysis:**
- **Current ASAP/ALAP are TRUE bounds**: They're computed from precedence only (see [`compute_ssa_asap`](scheduler_rs/src/cpsat_scheduler.rs#L1439) and [`compute_ssa_alap`](scheduler_rs/src/cpsat_scheduler.rs#L1490) in [`scheduler_rs/src/cpsat_scheduler.rs`](scheduler_rs/src/cpsat_scheduler.rs)), so they're correct bounds that don't consider resource capacity. An op CAN start anywhere in [ASAP, ALAP] from a precedence perspective.
- **Resource-aware tightening makes bounds TIGHTER**: The tightening algorithm would conclude "given resource capacity, this op MUST start earlier/later" and reduce the window.
- **The risk is ONLY algorithm correctness**: If the tightening algorithm is correct (e.g., using proven techniques like energetic reasoning or time-table propagation), it will only tighten bounds when it can PROVE they must be tighter. A correct algorithm will never cut off feasible solutions - it only removes infeasible parts of the domain. **The risk is entirely in the correctness of the algorithm implementation**, not in the concept itself.
- **Example of incorrect tightening (buggy algorithm)**: If we see 100 ops can start in window [10, 20] with capacity 10, a buggy algorithm might incorrectly conclude "impossible, need 100 ops in 11 cycles with capacity 10" and tighten ASAP to 20. But if those ops can actually be scheduled across multiple cycles within [10, 20] (some starting at 10, some at 11, etc.), the original window was correct. A correct algorithm using energetic reasoning would account for the fact that ops can be distributed across cycles.
- **Risk probability**: Low if using proven techniques (energetic reasoning, time-table propagation), Medium if implementing custom heuristics. The risk is entirely in implementation correctness, not in the concept.
- **Risk mitigation**: 
  1. Use proven tightening algorithms (energetic reasoning, time-table propagation) rather than custom heuristics
  2. Validate tightened bounds against greedy schedule (if greedy schedule fits, tightened bounds are at least feasible)
  3. **Test by re-optimizing without tightening**: After finding a solution with tightened bounds, remove tightening and re-optimize starting from that solution as a hint. If bounds were correct, CP-SAT should almost instantly verify the solution is still feasible (and possibly optimal). If bounds were over-tightened, we might find a better solution without tightening, indicating a bug.
  4. Make tightening optional with a flag, default to disabled initially
  5. Test on small problems first before scaling up
- **Conservative approach**: Only tighten when we can PROVE that the window is too small (e.g., using energetic reasoning or time-table propagation), not just when it looks crowded.

**Benefit Analysis:**
- **Search space reduction**: Tighter domains reduce search space exponentially (each op with domain size d contributes factor d to search space). If average domain size reduces from 100 cycles to 50 cycles, search space reduces by factor of 2^n where n = number of ops tightened.
- **Quantitative impact estimate**:
  - If tightening reduces average domain size by 20-30% for ~50% of ops (~5,888 ops), search space reduction = 2^(5,888 × 0.25) = astronomical reduction (but this is theoretical - actual reduction depends on which ops are tightened)
  - More realistic: If tightening helps propagate 100-500 ops earlier, search space reduction = 2^100 to 2^500 (still exponential)
  - Expected time to feasibility improvement: 2-10x faster (from >1hr to 6-30 minutes), depending on how many ops are tightened and how much
- **Propagation improvement**: Tighter domains help constraint propagation prune infeasible assignments earlier. Tighter ASAP/ALAP bounds mean cumulative constraints can propagate earlier, creating positive feedback loop.
- **Resource-aware reasoning**: Similar to "resource-aware ASAP/ALAP" techniques in scheduling literature
- **Addresses weakness #3**: Directly addresses the issue that precedence-only bounds create loose initial domains

**Configuration:**
- Add `enable_asap_alap_tightening: bool` field to `SsaCpsatConfig` struct (default: false)
- When enabled, apply tightening algorithm after ASAP/ALAP computation, before cycle variable creation

**Validation:**
1. Verify that greedy schedule still fits within tightened bounds. If greedy schedule doesn't fit, tightening algorithm is incorrect.
2. **Re-optimize without tightening**: After finding a solution with tightened bounds, remove tightening and re-optimize starting from that solution as a hint. If bounds were correct, CP-SAT should almost instantly verify the solution is still feasible (and possibly optimal). If bounds were over-tightened (buggy algorithm), we might find a better solution without tightening, indicating a bug.

### 2. Worst-Case Redundant Cumulative Constraints (Future)

**Idea:** For cumulative constraints with optional intervals, add redundant constraints assuming the "worst case" (most resource-intensive) branch. This provides propagation even when choice variables aren't fixed.

**Why This Fixes the >1hr Feasibility Problem:**

The core issue is that CP-SAT takes >1hr before finding a feasible solution (`best:inf` for extended periods). This happens because:

1. **Optional intervals propagate weakly**: The main cumulative constraints use optional intervals (present only when choice variables are true/false), which propagate weakly until choice variables are fixed
2. **Solver must explore many combinations**: Without early propagation, the solver must explore many combinations of `shift_var`, `const_choice_var`, `select_choice_var` values before discovering feasibility
3. **Long time to feasibility**: This leads to `best:inf` for extended periods while the solver searches

**How worst-case constraints fix this:**

1. **Early propagation**: Worst-case constraints use REQUIRED intervals (not optional), so they propagate immediately without fixing choice variables
2. **Prune infeasible branches early**: If worst-case VALU constraint (all shiftable use VALU) violates capacity, we know at least some MUST shift to ALU - this propagates `shift_var` variables earlier
3. **Cascade effect**: Once choice variables are propagated/constrained, the main cumulative constraint (with optional intervals) can propagate better
4. **Faster feasibility**: By pruning infeasible branches early, the solver finds feasible solutions faster

**Implementation:**
- **VALU cumulative worst-case**: Add constraint assuming ALL shiftable ops use VALU (worst case for VALU capacity)
  - For each shiftable VALU op, add required VALU interval (not optional)
  - Include non-shiftable VALU ops (e.g., vbroadcast, multiply_add) - they're always required in VALU
  - This constraint: `cumulative(all_valu_intervals_including_shiftable, demands, valu_cap)`
  - If this is infeasible, we know at least some shiftable ops MUST shift to ALU
- **ALU cumulative worst-case**: Add constraint assuming ALL shiftable ops shift to ALU (worst case for ALU capacity)
  - For each shiftable VALU op, add all 8 required ALU intervals
  - This constraint: `cumulative(all_alu_intervals_including_shifted, demands, alu_cap)`
  - If this is infeasible, we know at least some shiftable ops MUST use VALU
- **Load cumulative worst-case**: Add constraint assuming ALL ConstChoice ops use Load path
- **Flow cumulative worst-case**: Add constraint assuming ALL ConstChoice ops use AddImm path, ALL SelectChoice ops use vselect path

**Location:** After creating optional intervals but before (or alongside) the main cumulative constraints. See cumulative constraint creation at [lines 2411-2583](scheduler_rs/src/cpsat_scheduler.rs#L2411) in [`scheduler_rs/src/cpsat_scheduler.rs`](scheduler_rs/src/cpsat_scheduler.rs).

**Cost Analysis:**
- **Constraint count**: Only 5 additional cumulative constraints (one per engine), increasing model size by ~0.1% (5 / ~5,000 existing constraints)
- **Interval count**: 
  - VALU worst-case: ~5,120 VALU intervals (all shiftable ops assumed to use VALU)
  - ALU worst-case: ~5,120 × 8 = ~40,960 ALU intervals (all shiftable ops assumed shifted, 8 ALU ops per VALU op)
  - Load worst-case: ~const_choice_ops count (all ConstChoice use Load)
  - Flow worst-case: ~const_choice_ops + ~select_choice_ops count (all use Flow path)
  - Total: ~46K+ intervals added (but intervals are lightweight, just variable references)
- **Computation cost**: O(n) to create intervals and constraints, where n = ~11,776 ops. Negligible compared to model building (~O(n²) for precedence constraints)
- **Memory overhead**: Minimal - intervals are just variable references, not full copies
- **Solver overhead**: Each cumulative constraint adds propagation work, but worst-case constraints use required intervals which propagate efficiently (unlike optional intervals)
- **Total cost**: ~0.1% model size increase, ~0.01% memory increase, negligible computation cost

**Benefit Analysis:**
- **Early propagation**: Provides propagation even when choice variables aren't fixed. Worst-case constraints use required intervals, which propagate immediately (unlike optional intervals that wait for choice variables)
- **Prunes infeasible branches**: If worst-case is infeasible, we know the opposite branch must be taken (at least partially). This can propagate choice variables earlier, reducing search space exponentially (each choice variable fixed reduces search space by 2x)
- **Quantitative impact estimate**: 
  - If ~2,000 shiftable VALU ops exist, worst-case VALU constraint can immediately show "all VALU" is infeasible
  - This propagates that at least some `shift_var` must be true, potentially fixing hundreds of choice variables early
  - Search space reduction: 2^k where k = number of choice variables fixed early (could be 100-500 variables)
  - Expected time to feasibility improvement: 10-100x faster (from >1hr to 6-60 minutes), depending on how many choices are fixed early
- **Addresses weakness #1**: Directly addresses the issue that optional intervals prevent cumulative propagation by providing required-interval constraints that propagate immediately
- **Cascade effect**: Once choice variables are propagated/constrained, the main cumulative constraint (with optional intervals) can propagate better, creating a positive feedback loop

**Are Worst-Case Constraints Redundant?**

**Logical redundancy: YES**
- Worst-case constraints are logically implied by the main cumulative constraint when all choice variables are set to worst-case
- Example: If we set all `shift_var=false` (use VALU), the main VALU cumulative constraint becomes equivalent to the worst-case VALU constraint
- So worst-case constraints are logically redundant

**Propagation redundancy: MAYBE NOT**
- **Key insight**: Worst-case constraints use REQUIRED intervals (not optional), so they can propagate immediately without fixing choice variables
- **Main constraint**: Uses OPTIONAL intervals, which propagate weakly until choice variables are fixed
- **Difference**: Worst-case constraints can prune infeasible assignments early, while main constraint can't until choices are fixed
- **Example**: If worst-case VALU constraint (all shiftable use VALU) violates capacity at cycle 10, we know at least some MUST shift, even before fixing any `shift_var` variables

**Do They Hurt If Redundant?**

**Risk analysis:**
- **If propagation redundant**: Research shows propagation redundant constraints can slow down solvers ("the extra computation efforts of the additional variables and constraints may outweigh the gain of reduction in search space" [[Propagation Redundancy Paper](https://link.springer.com/chapter/10.1007/978-3-540-45193-8_16)])
- **If NOT propagation redundant**: Even if logically redundant, worst-case constraints can help if they propagate differently/better
- **Our case**: Worst-case constraints use required intervals vs optional intervals, so they likely propagate differently - they're probably NOT propagation redundant

**Potential issues:**
- **If worst-case is feasible**: Doesn't help prune (but also doesn't hurt - just adds ~0.1% overhead). However, worst-case constraints still provide propagation information (they constrain the search space even if they don't prove infeasibility)
- **If worst-case is infeasible**: Helps prune, but this is already implied by main constraint when choices are worst-case. The key benefit is WHEN this propagation happens (immediately vs after fixing many choice variables)
- **Overhead**: 5 additional cumulative constraints add some overhead (~5% more cumulative propagation work), but likely minimal compared to benefit. Worst-case constraints use required intervals which propagate efficiently (O(n) per propagation cycle), unlike optional intervals which require choice variable fixing
- **Memory**: ~46K+ interval references added, but intervals are lightweight (just variable references), so memory overhead is ~0.01% of model size

**Configuration:**
- Add `enable_worst_case_cumulative: bool` field to `SsaCpsatConfig` struct (default: false)
- When enabled, add worst-case cumulative constraints alongside main cumulative constraints

### 3. Per-Cycle Capacity Bounds (Future)

**Idea:** For each cycle `t`, add constraints that limit how many ops can start at `t` based on resource capacity. This helps propagate domains earlier by ruling out cycles where too many ops would conflict.

**Addresses propagation weakness #1 and #5**: Per-cycle bounds provide an alternative propagation path that doesn't depend on optional intervals being fixed. Even when cumulative constraints can't propagate due to optional intervals, per-cycle bounds can still prune infeasible cycles.

**Implementation:**

- For each cycle `t` in `[0, horizon-1]`:
  - Count ops that CAN start at `t`: `ASAP[i] <= t <= ALAP[i]`
  - For each engine, add linear constraint: `sum(boolean_vars[i] where ASAP[i] <= t <= ALAP[i]) <= capacity`
  - Handle optional intervals: Count all ops that CAN start (including optional), but only enforce constraint when intervals are present

**Exact constraint formulation:**
- For each cycle t and engine e:
  - Create boolean variables `can_start[i,t]` for each op i where `ASAP[i] <= t <= ALAP[i]`
  - Add constraint: `sum(can_start[i,t] for all i) <= capacity[e]`
  - Link `can_start[i,t]` to `cycle[i] == t` via reification

**Location:** After creating cycle variables and computing ASAP/ALAP, before cumulative constraints. See cycle variable creation around [line 1708](scheduler_rs/src/cpsat_scheduler.rs#L1708) and cumulative constraints starting at [line 2411](scheduler_rs/src/cpsat_scheduler.rs#L2411) in [`scheduler_rs/src/cpsat_scheduler.rs`](scheduler_rs/src/cpsat_scheduler.rs).

**Cost Analysis:**
- **Constraint count**: With horizon ~2,400 cycles and 5 engines, adding per-cycle constraints = ~12,000 constraints (2,400 cycles × 5 engines)
- **Current model size**: ~11,776 operations estimated for full kernel (16 rounds). Current constraint count: ~3,000-5,000 (precedence ~2,000-3,000 + cumulative 5 + scratch 1 + NoOverlap 1 + other). See constraint creation in [`scheduler_rs/src/cpsat_scheduler.rs`](scheduler_rs/src/cpsat_scheduler.rs): precedence constraints around [line 1800](scheduler_rs/src/cpsat_scheduler.rs#L1800), cumulative constraints at [lines 2411-2579](scheduler_rs/src/cpsat_scheduler.rs#L2411), scratch constraints at [line 2998](scheduler_rs/src/cpsat_scheduler.rs#L2998).
- **Model size increase**: ~12,000 constraints = 2-4x current model size (12K / 3K-5K). This is a significant increase that could impact solver performance.
- **Computation cost**: O(horizon × engines × ops_per_cycle) = O(2,400 × 5 × avg_ops_per_cycle) to build constraints. For ~11,776 ops distributed over 2,400 cycles, avg ~5 ops/cycle, so ~60K operations to build constraints. This is one-time preprocessing cost, negligible compared to solve time.
- **Solver overhead**: Each constraint adds propagation work. With ~12,000 additional constraints, propagation overhead could be significant. Research shows that "propagation redundant constraints" (logically redundant but don't add new propagation info) can slow down solvers: "the extra computation efforts of the additional variables and constraints may outweigh the gain of reduction in search space" [[Propagation Redundancy Paper](https://link.springer.com/chapter/10.1007/978-3-540-45193-8_16)]
- **Memory overhead**: ~12,000 constraint objects, each storing variable references and coefficients. Estimated ~1-2MB additional memory (negligible compared to model size).
- **Key question**: Do per-cycle bounds add NEW propagation information that cumulative constraints don't already provide? If cumulative constraints already enforce capacity per cycle, per-cycle bounds may be propagation redundant. However, cumulative constraints with optional intervals propagate weakly, so per-cycle bounds might provide alternative propagation path.

**Benefit Analysis:**
- **Potential benefit**: If cumulative constraints propagate weakly (especially in LP relaxation), per-cycle bounds could provide explicit linear constraints that strengthen LP bounds. This could help LP-based subsolvers (`default_lp`, `no_lp`, `max_lp_sym`) find better bounds faster.
- **Alternative propagation path**: Per-cycle bounds don't depend on optional intervals being fixed, so they can propagate even when cumulative constraints can't. This addresses weakness #1 (optional intervals preventing propagation).
- **Quantitative impact estimate**:
  - If cumulative constraints propagate weakly due to optional intervals, per-cycle bounds could provide 10-50% faster propagation
  - Expected time to feasibility improvement: 2-5x faster (from >1hr to 12-30 minutes), but only if bounds are NOT propagation redundant
  - Risk: If bounds are propagation redundant, they could slow down solver by 10-50% due to overhead
- **Time-table reasoning**: Similar to "time-table-edge-finding" propagation that considers resource profiles over time [[IJCAI 2016](https://ijcai.org/Proceedings/16/Papers/440.pdf)]
- **Redundant constraints can help**: Academic work shows redundant constraints (cutting planes) can strengthen cumulative propagation when they add new information [[Thesis HAL](https://theses.hal.science/tel-01745888/document)]
- **Conservative approach**: Start with subset of cycles (100-500 cycles, ~500-2,500 constraints) to test if propagation improves before adding all cycles

**Configuration:**
- Add `enable_per_cycle_bounds: bool` field to `SsaCpsatConfig` struct (default: false)
- Add `per_cycle_bounds_subset: Option<PerCycleBoundsSubset>` enum for conservative testing:
  - `FirstNCycles(usize)` - e.g., FirstNCycles(100) = first 100 cycles
  - `EveryNthCycle(usize)` - e.g., EveryNthCycle(10) = every 10th cycle
  - `NearCapacityBound(usize)` - e.g., NearCapacityBound(100) = 100 cycles around capacity_lower_bound
  - `AggregatedWindows(usize)` - e.g., AggregatedWindows(10) = windows of 10 cycles

### 4. Resource-Aware Tie-Breaker Weighting (Future)

**Idea:** Weight ops in the tie-breaker by their resource demand or criticality to guide search toward feasible solutions faster.

**Options:**

**A. Resource demand weighting:**
- Weight shifted VALU ops higher (they use 8 ALU slots, so scheduling them early frees capacity)
- Weight ConstChoice ops that choose Load higher (Load is more constrained than Flow)
- Formula: `tie_breaker = sum(weight[i] * completion_cycle[i])` where `weight[i]` depends on op type

**B. Critical path weighting:**
- Weight ops by number of successors (ops that unblock many others should be scheduled earlier)
- Compute `num_successors[i]` from dependency graph
- Formula: `tie_breaker = sum((1 + num_successors[i]) * completion_cycle[i])`

**Cost Analysis:**
- **Computation cost**: 
  - Option A (resource demand): O(n) = ~11,776 ops to compute weights based on op type
  - Option B (critical path): O(n + m) = ~11,776 ops + ~2,000-3,000 edges to build reverse dependency graph and compute successor counts
  - Building weighted objective: O(n) = ~11,776 ops (already O(n) for unweighted tie-breaker)
  - Total: O(n) for Option A, O(n + m) for Option B, both negligible compared to model building
- **Solver overhead**: Negligible - objective evaluation is already done, just with different coefficients. Weighted sum evaluation is O(n) per evaluation, same as unweighted.
- **Memory overhead**: Minimal - just storing weights (one float per op = ~48KB for 11,776 ops)
- **Risk**: Very low - worst case is neutral (solver ignores weights if they don't help). No constraint overhead, no propagation overhead.

**Benefit Analysis:**
- **Search guidance**: Helps solver prioritize scheduling resource-heavy ops earlier, which may help find feasibility faster. By weighting ops by resource demand (e.g., shifted VALU ops use 8 ALU slots), the solver is guided toward scheduling these ops earlier, which may help fix choice variables (`shift_var`) earlier.
- **No constraint overhead**: Unlike per-cycle bounds, this doesn't add constraints - just changes objective coefficients. This means no propagation overhead, no memory overhead beyond weights.
- **Quantitative impact estimate**:
  - Tie-breaker already exists and helps guide search (see [tie-breaker documentation](scheduler_rs/src/cpsat_scheduler.rs#L84) and [implementation](scheduler_rs/src/cpsat_scheduler.rs#L3451))
  - Weighting may make tie-breaker 10-30% more effective at guiding search toward feasible solutions
  - Expected time to feasibility improvement: 10-30% faster (from >1hr to 42-54 minutes), but this is indirect (search guidance) so impact may be smaller
- **Indirect effect on weakness #2**: While tie-breaker weighting doesn't directly fix choice variables early, it guides search toward solutions where resource-heavy ops (which often involve choices) are scheduled earlier, which may indirectly help fix choice variables sooner

**Implementation:** Modify tie-breaker computation around [lines 3528-3587](scheduler_rs/src/cpsat_scheduler.rs#L3528) in [`scheduler_rs/src/cpsat_scheduler.rs`](scheduler_rs/src/cpsat_scheduler.rs). See also tie-breaker documentation at [line 84](scheduler_rs/src/cpsat_scheduler.rs#L84).

**Configuration:**
- Add `tie_breaker_weighting: TieBreakerWeighting` enum to `SsaCpsatConfig` struct:
  - `Uniform` - No weighting (baseline)
  - `ResourceDemand` - Weight by resource demand (shifted VALU, ConstChoice Load, etc.)
  - `CriticalPath` - Weight by number of successors

### 5. CP-SAT Subsolver/Parameter Options (Future)

**Given our constraint structure, explore these parameter options.** See also [`docs/CP_SAT_SUBSOLVER_PARAMETER_EVIDENCE.md`](docs/CP_SAT_SUBSOLVER_PARAMETER_EVIDENCE.md) for detailed evidence and rationale.

#### A. LNS Tuning (High Priority)

Since progress comes from LNS (`scheduling_time_window_lns`, `graph_cst_lns`, etc.), tune LNS parameters:

**Parameters to explore:**
- `use_lns_only`: If true, focus search on LNS heuristics (may help when core MIP is slow)
- `lns_focus_on_decision_variables`: Prioritize LNS neighborhoods that include choice variables (`shift_var`, `const_choice_var`, `select_choice_var`)
- `use_optional_variables`: Control how optional intervals are handled in search (we have many: shift, ConstChoice, SelectChoice)

**Systematic testing approach:**
- Test one parameter at a time, measure impact, document results
- Avoid combinatorial explosion of parameter combinations
- Create testing matrix: test each parameter independently, measure impact, document results in a table

**Rationale and Evidence:**
- **LNS is a key component**: CP-SAT uses LNS as part of its portfolio of incomplete subsolvers for finding primal solutions. The CP-SAT Primer notes that "CP-SAT uses LNS itself to find better solutions" and that LNS operates using a "destroy and repair" strategy that can explore large neighborhoods efficiently [[CP-SAT Primer LNS](https://d-krupke.github.io/cpsat-primer/09_lns.html)].
- **Portfolio approach**: The CP-SAT-LP solver paper describes how "a portfolio of different workers makes it easier to try new ideas" and that "the continuous sharing of information between workers has produced massive speedups when running multiple workers in parallel" [[Perron et al. 2023](https://drops.dagstuhl.de/storage/00lipics/lipics-vol280-cp2023/LIPIcs.CP.2023.3/LIPIcs.CP.2023.3.pdf)].
- **When to focus on LNS**: If LNS workers (`scheduling_time_window_lns`, `graph_cst_lns`, etc.) are finding solutions but core MIP subsolvers (`default_lp`, `no_lp`) are slow to find feasibility, focusing on LNS might help find feasibility faster. However, the CP-SAT Primer warns that "most solver parameters beyond basic settings like `max_time_in_seconds` are best left at their defaults" [[CP-SAT Primer Parameters](https://d-krupke.github.io/cpsat-primer/05_parameters.html)].

**Caveat**: The CP-SAT Primer notes that `use_lns_only` "needs to be combined with a time limit, as the incomplete subsolvers do not know when to stop" [[CP-SAT Primer LNS](https://d-krupke.github.io/cpsat-primer/09_lns.html)].

**Configuration:**
- Add optional `cpsat_params: Option<CpsatParams>` field to `SsaCpsatConfig` struct for parameter overrides
- Test `use_lns_only` first (simplest, most likely to help)
- Document which parameters help and which don't

### 6. Profile Bounds as Explicit Constraints (Future)

**Idea:** Add profile bounds (from [`compute_capacity_lower_bound_global`](scheduler_rs/src/cpsat_scheduler.rs#L1174)) as explicit linear constraints for critical deadlines, not just makespan domain.

**Implementation:**
- Compute profile bounds: for deadline `d`, count ops with `ALAP <= d`
- Add constraints: `makespan >= d + ceil(count_d / capacity)` for a few critical `d`
- Focus on deadlines near `capacity_lower_bound` where bounds are tightest

**Location:** After computing [`capacity_lower_bound`](scheduler_rs/src/cpsat_scheduler.rs#L3277) (around [line 3277](scheduler_rs/src/cpsat_scheduler.rs#L3277)), alongside existing redundant linear constraints ([lines 3330-3405](scheduler_rs/src/cpsat_scheduler.rs#L3330)) in [`scheduler_rs/src/cpsat_scheduler.rs`](scheduler_rs/src/cpsat_scheduler.rs).

**Cost Analysis:**
- **Constraint count**: Only a few constraints (maybe 10-20 for critical deadlines). This is ~0.5% of current model size (~20 / ~5,000 constraints).
- **Computation cost**: O(n × k) to count ops per deadline, where n = ~11,776 ops and k = number of critical deadlines (~10-20). Total: ~118K-236K operations, negligible compared to model building.
- **Memory overhead**: Minimal - just storing constraint coefficients (~20 constraints × ~10-20 ops per constraint = ~200-400 coefficients, ~1.6-3.2KB)
- **Solver overhead**: Minimal - similar to existing redundant linear constraints already in the model (see [lines 3330-3405](scheduler_rs/src/cpsat_scheduler.rs#L3330))

**Benefit Analysis:**
- **LP relaxation**: Makes the LP relaxation tighter by giving it explicit constraints, not just domain bounds. This could help LP-based subsolvers (`default_lp`, `no_lp`, `max_lp_sym`) find better bounds faster.
- **Propagation**: May help cumulative constraints propagate better by providing explicit capacity bounds at specific deadlines. However, impact may be limited since cumulative constraints already enforce capacity.
- **Quantitative impact estimate**:
  - If LP relaxation improves by 5-10% (tighter bounds), this could help LP-based subsolvers find feasibility 10-20% faster
  - Expected time to feasibility improvement: 5-20% faster (from >1hr to 48-57 minutes), but this is indirect (LP relaxation) so impact may be smaller
- **Low risk**: Small number of constraints (~20), similar to existing redundant constraints, so overhead is minimal

**Configuration:**
- Add `enable_profile_bounds_constraints: bool` field to `SsaCpsatConfig` struct (default: false)
- When enabled, add profile bounds as explicit linear constraints

## Implementation Order

**All improvements must be optional** to enable ablation studies. Implement in this order:

1. **Documentation** (write everything in a checked-in doc) ✅ **COMPLETED**
   - Create comprehensive documentation file (e.g., `docs/CPSAT_IMPROVEMENTS.md`)
   - Document all proposed improvements, rationale, implementation details, testing strategy
   - Include this plan's content plus implementation notes
   - Check in documentation before implementing any code changes

2. **Implement optional ASAP/ALAP tightening** (addresses weakness #3 directly - tightens domains before search starts) 🔄 **IN PROGRESS**
   - Implement with `enable_asap_alap_tightening` flag (default: false)
   - Use proven techniques (energetic reasoning, time-table propagation)
   - Add to `SsaCpsatConfig` struct
   - Implement tightening algorithm after ASAP/ALAP computation, before cycle variable creation
   - Include validation: verify greedy schedule fits within tightened bounds
   - Include re-optimization test: remove tightening and re-optimize from solution to verify correctness

3. **Ablation study** (limited 4-round test with batch size 64) ⏳ **PENDING**
   - Test baseline (all improvements disabled)
   - Test ASAP/ALAP tightening enabled
   - Measure: time to first feasible solution, makespan, propagation strength
   - Document results: which improvements help, which don't, quantitative impact
   - Use 4 rounds, batch size 64 for fast iteration (~2,944 ops vs ~11,776 for full)
   - After identifying helpful improvements, consider testing combinations

**Future implementation order** (after ablation study identifies helpful improvements):

4. **Worst-case redundant cumulative constraints** (if ablation shows promise)
5. **Resource-aware tie-breaker** (if ablation shows promise)
6. **CP-SAT subsolver/parameter exploration** (if ablation shows promise)
7. **Per-cycle capacity bounds** (if ablation shows promise and worst-case doesn't help enough)
8. **Profile bounds as constraints** (if ablation shows promise)

## Configuration Structure

**All improvements must be optional** to enable ablation studies. Add config flags to `SsaCpsatConfig` struct:

```rust
pub struct SsaCpsatConfig {
    // ... existing fields ...
    
    // ASAP/ALAP tightening
    pub enable_asap_alap_tightening: bool,  // default: false (test first, validate carefully)
    
    // Future improvements (to be added after ablation study):
    // pub enable_worst_case_cumulative: bool,  // default: false
    // pub enable_per_cycle_bounds: bool,  // default: false
    // pub per_cycle_bounds_subset: Option<PerCycleBoundsSubset>,  // None = all cycles, Some = subset
    // pub tie_breaker_weighting: TieBreakerWeighting,  // default: Uniform
    // pub enable_profile_bounds_constraints: bool,  // default: false
}
```

**Default configuration (all disabled for ablation):**
- All improvements disabled by default to enable clean baseline comparison
- Enable improvements one at a time for ablation studies
- After identifying helpful improvements, enable them by default

## Testing Strategy

**Ablation Studies:**
- **All improvements must be optional** to enable ablation studies
- Test each improvement independently: baseline vs each improvement alone, then combinations
- Measure impact of each improvement: time to first feasible solution, makespan, propagation strength
- Identify which improvements provide the most benefit and which can be combined effectively

**Shorter Test Runs for Faster Iteration:**

- **Use fewer rounds for development/testing**: Default is 16 rounds, but use 2-4 rounds for faster iteration
  - 2 rounds: ~1,472 ops (vs ~11,776 for 16 rounds) = ~8x smaller problem
  - 4 rounds: ~2,944 ops = ~4x smaller problem
  - Horizon scales roughly linearly with rounds, so 2 rounds = ~300 cycles (vs ~2,400 for 16 rounds)
- **Use smaller batch sizes**: Default is 256, but use 64 or 128 for faster iteration
  - 64 batch size: ~4x fewer ops per round = ~4x smaller problem
  - 128 batch size: ~2x fewer ops per round = ~2x smaller problem
  - Note: batch_size must be a multiple of VLEN (8), so valid sizes are 8, 16, 24, 32, 64, 128, 256
- **Expected speedup**: 2 rounds + 64 batch size = ~32x smaller problem = ~32x faster solve time (from >1hr to ~2 minutes)
- **Validation**: After finding improvements that help on shorter runs, validate on full 16-round, 256 batch size runs

**Testing Protocol:**

1. **Initial testing (short runs)**:
   - Use 2-4 rounds, 64-128 batch size for fast iteration
   - Test each improvement independently (ablation)
   - Measure: time to first feasible solution, makespan, propagation in logs
   - Identify which improvements help most

2. **Combination testing (short runs)**:
   - Test combinations of improvements that showed promise
   - Measure: do improvements combine additively, multiplicatively, or interfere?

3. **Validation (full runs)**:
   - Run on full-kernel (16-round, 256 batch size) with best combination
   - Compare with baseline: does `best:inf` time decrease?
   - Verify solution quality: does makespan improve or stay similar?
   - Check propagation: do variable domains tighten faster in logs?

**Specific metrics to measure:**
- Time to first feasible solution (seconds)
- Number of choice variables fixed early (count)
- Domain reduction percentage (before/after)
- Propagation strength (can measure via CP-SAT logs - number of propagations, domain reductions)

**Special testing for ASAP/ALAP tightening:**
- After finding a solution with tightened bounds, remove tightening and re-optimize starting from that solution as a hint
- If bounds were correct, CP-SAT should almost instantly verify the solution is still feasible (and possibly optimal)
- If bounds were over-tightened (buggy algorithm), we might find a better solution without tightening, indicating a bug
- This provides a strong correctness check: correct tightening should never cut off feasible solutions, only remove infeasible parts of domains

**Success criteria:**
- Time to feasibility improves by X% (e.g., 20%+)
- Makespan doesn't degrade significantly (within 5%)
- Propagation strength improves (measurable via logs)

## Files to Modify

- [`scheduler_rs/src/cpsat_scheduler.rs`](scheduler_rs/src/cpsat_scheduler.rs):
  - Add `enable_asap_alap_tightening` field to `SsaCpsatConfig` struct (around [line 1126](scheduler_rs/src/cpsat_scheduler.rs#L1126))
  - Implement ASAP/ALAP tightening function after ASAP/ALAP computation (around [line 1642](scheduler_rs/src/cpsat_scheduler.rs#L1642)), before creating cycle variables ([line 1708](scheduler_rs/src/cpsat_scheduler.rs#L1708))
  - Future: Add worst-case redundant cumulative constraints alongside main cumulative constraints ([~line 2411-2583](scheduler_rs/src/cpsat_scheduler.rs#L2411))
  - Future: Add per-cycle capacity bounds after cycle variable creation ([~line 1717](scheduler_rs/src/cpsat_scheduler.rs#L1717))
  - Future: Modify tie-breaker computation to use weights ([~line 3528-3587](scheduler_rs/src/cpsat_scheduler.rs#L3528))
  - Future: Add profile bounds as constraints ([~line 3330](scheduler_rs/src/cpsat_scheduler.rs#L3330))

## References

- [CP-SAT Primer](https://d-krupke.github.io/cpsat-primer/) - Comprehensive guide to CP-SAT
- [CP-SAT-LP Paper](https://drops.dagstuhl.de/storage/00lipics/lipics-vol280-cp2023/LIPIcs.CP.2023.3/LIPIcs.CP.2023.3.pdf) - CP-SAT with LP integration
- [Propagation Redundancy Paper](https://link.springer.com/chapter/10.1007/978-3-540-45193-8_16) - When redundant constraints help or hurt
- [Thesis HAL](https://theses.hal.science/tel-01745888/document) - Redundant constraints for cumulative propagation
- [IJCAI 2016](https://ijcai.org/Proceedings/16/Papers/440.pdf) - Time-table-edge-finding propagation
- [`docs/CP_SAT_SUBSOLVER_PARAMETER_EVIDENCE.md`](docs/CP_SAT_SUBSOLVER_PARAMETER_EVIDENCE.md) - Evidence for CP-SAT parameter options
