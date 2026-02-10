# Depth-2 vs Depth-3 Tree-Select Analysis

**Analysis log:** Depth-2 vs depth-3 comparison, per-decile utilization, hypothesis quantification, CSE review, and SUB-in-FMA redundant-SUB findings (with cycle estimate). Data from `scheduler_output_4r_256b_depth2.json` and `scheduler_output_4r_256b_depth3.json` (4 rounds, batch 256).

## Summary

**Question:** Are there inefficiencies holding depth-3 back that, if fixed, could make depth-3 faster than scatter load?

**Answer:** Depth-3 is currently **8 cycles slower** (443 vs 435) due to increased VALU pressure from tree-select operations. The main bottlenecks are:
1. **VALU saturation** in decile 6 (98.9% vs 80.7%) from 128 additional FMA ops
2. **Tighter dependencies** reducing parallelism (less theoretical cycle saving: 1321 vs 3071)
3. **More vselect ops** (96 additional) creating Flow slot pressure in later deciles

However, depth-3 has **potential** to be faster if we can:
- Shift more FMA ops to ALU (currently underutilized in deciles 7-9)
- Optimize SelectChoice decisions (more vselect when VALU saturated)
- Improve dependency chains (reduce wasted slack)

## Operation Counts

| Metric | Depth-2 | Depth-3 | Delta |
|--------|---------|---------|-------|
| **Total cycles** | 435 | 443 | +8 |
| **FMA ops** | 512 | 640 | +128 |
| **vselect ops** | 128 | 224 | +96 |
| **vbroadcast ops** | 54 | 61 | +7 |
| **Peak liveness** | 1267 | 1266 | -1 |

## Per-Decile Slot Utilization

### Key Differences

**Decile 6 (cycles 261-310 depth-2, 266-310 depth-3):**
- Depth-3: **75.6% ALU** (408/540), **98.9% VALU** (267/270), **134 FMA ops**
- Depth-2: **59.1% ALU** (312/528), **80.7% VALU** (213/264), **83 FMA ops**
- **Issue:** Depth-3 has much higher VALU saturation here, with 51 more FMA ops in this decile

**Deciles 7-9 (later cycles):**
- Depth-3: Has **ALU usage** (33-64%) where depth-2 has **0% ALU**
- Depth-3: More **Flow vselect** ops (100% Flow in deciles 5, 7, 8)
- **Opportunity:** Depth-3's ALU usage suggests some ops could be shifted earlier

## Hypothesis Quantification

| Metric | Depth-2 | Depth-3 | Implication |
|--------|---------|---------|-------------|
| **Theoretical cycle saving** (element-wise slack) | 3071 | 1321 | Depth-3 has **less parallelism opportunity** |
| **Vbroadcast elimination chains** | 5840 | 6907 | More chains, but less profitable |
| **Shifted ops with consumers** | 247 | 353 | More shifted ops, but less slack to exploit |

**Key insight:** Depth-3 has **tighter dependencies** - the tree-select operations create sequential chains that reduce parallelism.

## Root Cause Analysis

### Why Depth-3 is Slower

1. **Round 3 tree-select overhead:**
   - Depth-3: 32 groups × (4 FMA + 2 SelectChoice + 1 SelectChoice) = 224 ops
   - Depth-2: 32 groups × 8 scatter loads = 256 loads (but loads are cheaper and parallelize better)

2. **VALU saturation:**
   - Depth-3 adds 128 FMA ops (4 per group × 32 groups)
   - These FMAs compete with hash computation for VALU slots
   - Decile 6 shows 98.9% VALU utilization vs 80.7% for depth-2

3. **Dependency chains:**
   - Tree-select creates sequential dependencies: bit extraction → level 0 FMA → level 1 select → level 2 select
   - This reduces parallelism compared to scatter loads which can be issued independently

### Why Depth-3 Could Be Faster (Potential)

1. **ALU underutilization:**
   - Depth-3 uses ALU in deciles 7-9 (33-64%) where depth-2 has 0%
   - If we shift more FMA ops to ALU, we could reduce VALU pressure

2. **SelectChoice optimization:**
   - Depth-3 has 96 more vselect ops (224 vs 128)
   - If CP-SAT chose more FMA paths when VALU is free, we could reduce Flow pressure
   - Current: vselect saves 2 VALU ops at cost of 1 Flow op (good when VALU saturated)

3. **Better scheduling:**
   - Depth-3 has less theoretical cycle saving (1321 vs 3071)
   - This suggests the current schedule isn't exploiting parallelism well
   - Better CP-SAT optimization or different SelectChoice decisions could help

## Recommendations

### Short-term (Quick Wins)

1. **Force more FMA → ALU shift:**
   - Depth-3's tree-select FMAs could be shifted to ALU when VALU is saturated
   - Check if CP-SAT is shifting these FMAs optimally
   - Use `what_if_shift.py` to test forcing specific FMAs to ALU

2. **Optimize SelectChoice for depth-3:**
   - When VALU is saturated (decile 6), prefer vselect (1 Flow vs 2 VALU)
   - When VALU is free (deciles 7-9), prefer FMA (2 VALU vs 1 Flow)
   - Current CP-SAT may not be making optimal choices per decile

### Medium-term (Architectural)

1. **Hybrid approach:**
   - Use depth-3 tree-select for early rounds (when VALU is less saturated)
   - Fall back to scatter load for later rounds (when hash computation saturates VALU)
   - This would require per-round tree-reduce depth selection

2. **Improve tree-select emission:**
   - Current: 4 FMA + 2 SelectChoice + 1 SelectChoice = 7 ops per group
   - Could we reduce to 4 FMA + 1 SelectChoice + 1 SelectChoice = 6 ops?
   - Or use different encoding that's more parallelizable

3. **Better dependency analysis:**
   - Tree-select creates tight chains: bit0 → FMA → bit1 → select → bit2 → select
   - Could we overlap bit extraction with previous round's hash?
   - Or pipeline tree-select across groups?

### Long-term (Research)

1. **Alternative tree-select encodings:**
   - Current uses bit-tree (logarithmic depth)
   - Could try linear chain (more parallelizable but more ops)
   - Or hybrid: some levels use different encoding

2. **Scratch-aware tree-select:**
   - Depth-3 caches 8 tree values (vs 4 for depth-2)
   - Could we cache more efficiently or reuse scratch better?

## Conclusion

Depth-3 is currently **8 cycles slower** due to VALU saturation from tree-select operations. However, there are clear optimization opportunities:

1. **ALU shift:** Depth-3 has ALU capacity in later deciles that could absorb shifted FMAs
2. **SelectChoice tuning:** Better per-decile SelectChoice decisions could reduce Flow pressure
3. **Dependency optimization:** Current schedule has less parallelism than theoretically possible

**Verdict:** Depth-3 has potential to match or beat scatter load, but requires:
- Better CP-SAT optimization (more aggressive ALU shift, per-decile SelectChoice)
- Possibly hybrid approach (depth-3 for early rounds, scatter for later)
- Or architectural improvements to tree-select emission

The **8-cycle gap** is small enough that these optimizations could close it, but depth-3 is not inherently faster - it requires careful optimization to realize its benefits.

---

## CSE Analysis (Depth-3 Tree-Select)

### What Is Already CSE'd

- **Constants:** `v_const_7`, `v_1`, tree-reduce deltas (`v_td_r3_01`, etc.), bases (`v_tc_r3_0`, etc.) — written once, read many times.
- **Broadcasts:** `broadcast_cache` in the emitter deduplicates vbroadcast by scalar VReg.
- **Group-specific values** (e.g. `idx_adj`, `bit0`, `idx_adj_shr`) cannot be CSE'd; they depend on per-group `idx`.

### SUB in SelectChoice FMA Path

When SelectChoice takes the **FMA path**, we emit:
- `sub_temp = true_val - false_val` (1 VALU SUB)
- `dest = multiply_add(cond, sub_temp, false_val)` (1 VALU FMA)

**Finding (from `scheduler_output_4r_256b_depth3.json`):**

| Metric | Value |
|--------|--------|
| Total SUB ops | 385 |
| Unique SUB input patterns | 265 |
| Repeated SUB patterns | 37 |
| **Redundant SUBs (CSE opportunity)** | **120** |

Many SelectChoice ops share the same `(true_val, false_val)` but each allocates its own `sub_temp_vreg` and emits its own SUB. Those SUBs have **identical inputs** and write different VRegs — they are repeated work.

**Example:** `SUB with reads (27, 28, …, 34, 86, 87, …, 93)` appears **13 times** (13 different result VRegs). All 13 could use a single CSE'd SUB.

### Where CSE Must Happen

**CSE has to be done at emission (before scheduling), not at lowering.**  
If we eliminate a SUB, we change the SSA graph and slot demand, so the scheduler must see the reduced op set and dependencies. Lowering is too late; it only expands SelectChoice into SUB+FMA after scheduling.

### Estimated Cycle Impact of CSE'ing Redundant SUBs

- **120 redundant SUBs** → 120 fewer VALU ops if we CSE.
- VALU has **6 slots/cycle**.
- **Upper bound (if SUBs were the only VALU ops):** 120 / 6 ≈ **20 cycles**.
- **In practice:** SUBs are mixed with FMA and other VALU ops; makespan is set by critical path and slot pressure. Removing 120 SUBs mainly reduces VALU pressure, so:
  - **Conservative:** 2–5 cycles (only where SUBs extend the critical path or force extra cycles).
  - **Optimistic:** up to ~10–15 cycles if VALU is the bottleneck in the regions where these SUBs sit.

**To get a real number:** Implement SUB CSE at emission, re-run the scheduler (same options), and compare `total_cycles` before/after.

---

## Deciles During Depth-3 Tree-Select: ALU Slack and SUB ALU-Shift

Per-decile slot usage (depth-3, 443 cycles) and where SUBs land. Columns: ALU, VALU, Flow, Load, Store (each as used/cap and %), and SUB count in that decile.

| Decile | Cycle range | ALU | VALU | Flow | Load | Store | SUBs |
|--------|-------------|-----|------|------|------|-------|------|
| 0 | 0–44 | 354/540 (65.6%) | 251/270 (93.0%) | 40/45 (89%) | 88/90 (98%) | 0/90 (0%) | 8 |
| 1 | 45–88 | 352/528 (66.7%) | 264/264 (**100%**) | 0/44 (0%) | 0/88 (0%) | 0/88 (0%) | 56 |
| 2 | 89–132 | 336/528 (63.6%) | 260/264 (98.5%) | 32/44 (73%) | 0/88 (0%) | 0/88 (0%) | 4 |
| 3 | 133–177 | 336/540 (62.2%) | 261/270 (**96.7%**) | 2/45 (4%) | 0/90 (0%) | 0/90 (0%) | 60 |
| 4 | 178–221 | 136/528 (25.8%) | 133/264 (50.4%) | **44/44 (100%)** | 0/88 (0%) | 0/88 (0%) | 0 |
| 5 | 222–265 | 336/528 (63.6%) | 257/264 (97.3%) | **44/44 (100%)** | 0/88 (0%) | 0/88 (0%) | 0 |
| 6 | 266–310 | 408/540 (75.6%) | 267/270 (**98.9%**) | 23/45 (51%) | 0/90 (0%) | 0/90 (0%) | **137** |
| 7 | 311–354 | **176/528 (33.3%)** | **176/264 (66.7%)** | **44/44 (100%)** | 0/88 (0%) | 0/88 (0%) | 88 |
| 8 | 355–398 | 336/528 (63.6%) | 261/264 (98.9%) | 35/44 (80%) | 0/88 (0%) | 0/88 (0%) | 0 |
| 9 | 399–442 | 216/528 (40.9%) | 172/264 (65.2%) | 0/44 (0%) | 0/88 (0%) | **64/88 (73%)** | 32 |

Flow breakdown: add_imm = ConstChoice (true path); vselect = SelectChoice (false path). In deciles 4, 5, 7, Flow is 100% and all are **vselect** (44 ops over the decile). Load/Store: 0% in most deciles except decile 0 (Load 98%) and decile 9 (Store 73%).

**Finding:** Where tree-select SUBs run (deciles 1, 3, 6, 7), we have:

- **Deciles 1, 3, 6:** VALU at or near **100%**; ALU has **~24–38% slack**; Flow and Load/Store are 0% or low.
- **Decile 7:** **88 SUBs**, but ALU **33.3%** and VALU **66.7%** → lots of free ALU and VALU. **Flow is 100%** (44 vselects over the decile). So decile 7 is **Flow-bound**: every cycle uses the single Flow slot for vselect, while ALU and VALU have headroom. Load and Store are 0%. The slack in ALU/VALU is not helping because the work in that window is dominated by SelectChoice ops taking the vselect path (1 Flow op per cycle).

So during depth-3 tree-select we often have free ALU slots while VALU is saturated.

**Would ALU-shifting the SUB be beneficial?** Yes, in principle:

- Each SUB today uses 1 VALU slot. Shifting to ALU would use 8 ALU slots (element-wise) and free 1 VALU slot.
- ALU has 12 slots/cycle and slack in these deciles; VALU has 6 and is full in 1, 3, 6.
- Moving SUB off VALU would reduce VALU pressure and could shorten the schedule.

**Catch:** SUBs are not first-class ops at schedule time. They are created in **lowering** when a SelectChoice is expanded to the FMA path (sub_temp = high - low, then multiply_add). The CP-SAT model only sees SelectChoice; it doesn’t see SUB as a separate shiftable VALU op. So today the scheduler cannot “ALU-shift the SUB” because the SUB doesn’t exist yet.

**Ways to get SUB on ALU:**

1. **Pre-expand SelectChoice before scheduling**  
   Emit real SUB + FMA (or vselect) at emission so the scheduler sees them. Then SUB is a normal VALU op and `Opcode::Sub` already has `can_shift_to_alu() == true`, so existing VALU→ALU shift would apply. The “choice” of vselect vs FMA would need to be decided earlier or modeled differently.

2. **Extend the CP-SAT model**  
   When SelectChoice takes the FMA path, treat the implied SUB as an optional shiftable VALU op (e.g. optional interval on VALU vs ALU) so the solver can place that subtraction on ALU.

So: **we do have lots of free ALU during depth-3 select, and shifting the SUB to ALU would likely help; to get that, SUB must be visible to the scheduler (emission or model), not only created at lowering.**

---

## SUB with constant 0 (zero operand)

**Question:** Do the SUBs from the SelectChoice FMA path ever use 0 as a constant argument? If so, that could be special-cased (e.g. skip SUB when `false_val == 0` → use `true_val` as `sub_temp`).

**Emitter behavior (already in code):** In `emit_select_choice` we have:
- `needs_subtraction = self.vreg_constants.get(&false_val) != Some(&0)`
- When **false_val is constant 0**, we set `needs_subtraction = false`, do **not** allocate `sub_temp_vreg`, and the SelectChoice FMA path becomes `multiply_add(cond, true_val, 0)` (i.e. no SUB). So **false_val == 0 is already special-cased** and no SUB is emitted.

**Schedule analysis (depth-3 JSON):**
- Total SUB ops in schedule: 385.
- No SUB op has **0** in its `reads` (checked all SUB ops; min/max read VReg IDs 1–799; 0 never appears).
- So in this kernel we do **not** see any SUB that uses constant 0 as an operand.

**Interpretation:**
1. Any SelectChoice with **false_val == 0** is already optimized at emission (no SUB), so such cases do not appear as SUBs in the schedule.
2. **true_val == 0** would give `sub_temp = 0 - false_val` (negation). The emitter does not special-case that; we’d still emit a SUB. In the depth-3 tree-select, operands are tree-cache values or earlier select results, so true_val is not constant 0 in practice.
3. **Conclusion:** For this kernel there is no additional zero-operand SUB case to special-case. The only useful special case (false_val == 0) is already implemented.
