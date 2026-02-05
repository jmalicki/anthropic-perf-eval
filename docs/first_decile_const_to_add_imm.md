# First-decile hand-optimization: Const → add_imm to free Load slots

## Observation (from scheduler_output.json, 2-round kernel, first decile = cycles 0–19)

- **Load:** Every cycle in the first decile uses both Load slots (cap 2). So Load is 100% utilized.
- **Flow:** Every cycle in the first decile uses **zero** Flow slots (cap 1). Flow is completely free.
- **Const vs data loads:** In decile 0 there are 20 const ops (ConstChoice using Load) and 18 other load ops (vload/load). The consts occupy Load slots in 10 cycles where both slots are const-only.

## Per-cycle picture (first decile)

| Cycle | Load slot 1     | Load slot 2     | Flow  |
|-------|-----------------|-----------------|-------|
| 0     | const (id 0)    | const (id 1)    | free  |
| 1     | const (2)       | const (15)       | free  |
| 2     | load (14)       | load (16)        | free  |
| 3     | const (17)      | const (22)       | free  |
| 4     | load (18)       | load (47)        | full  |
| 5     | load (49)       | load (51)        | full  |
| 6     | vload (78)      | vload (79)       | free  |
| 7     | const (24)      | const (26)       | free  |
| 8     | const (28)      | const (30)       | free  |
| 9     | const (32)      | const (34)       | free  |
| 10    | const (36)      | const (44)       | free  |
| 11    | const (72)      | const (74)       | free  |
| 12    | vload (87)      | vload (88)       | free  |
| 13    | const (80)      | const (89)       | free  |
| 14    | const (94)      | const (99)       | free  |
| 15–18 | data loads      | …                | free  |

## Hand-optimization (what the scheduler should be able to find)

In each of the **10 cycles** where both Load slots are const (0, 1, 3, 7, 8, 9, 10, 11, 13, 14):

- Keep **one** ConstChoice as Load (const).
- Switch the **other** ConstChoice to **add_imm** (Flow), using the free Flow slot.

Effects:

- **10 Load slots** are freed in the first decile (one per cycle).
- Const semantics are unchanged: add_imm in the same cycle produces the same constant for consumers.
- No extra cycles: the same op runs in the same cycle, only the engine choice (Load vs Flow) changes.

Concrete choices (one per cycle; the other const in that cycle stays Load):

| Cycle | Prefer add_imm (op id) | Keep as Load (op id) |
|-------|------------------------|----------------------|
| 0     | 0 or 1                 | the other            |
| 1     | 2 or 15                | the other            |
| 3     | 17 or 22               | the other            |
| 7     | 24 or 26               | the other            |
| 8     | 28 or 30               | the other            |
| 9     | 32 or 34               | the other            |
| 10    | 36 or 44               | the other            |
| 11    | 72 or 74               | the other            |
| 13    | 80 or 89               | the other            |
| 14    | 94 or 99               | the other            |

## Why this helps

- **Get data loads ASAP:** The 10 freed Load slots can be used for **vload** or **load** ops that are currently scheduled in later cycles (e.g. first vloads at 6, 12, 15…), **if** their dependencies allow running earlier. Even if some cannot move, freeing Load in the first decile reduces pressure and can allow a better global schedule.
- **No new constraints:** ConstChoice is already modeled as “either Load or add_imm”; the solver only needs to set `choice_var = true` (add_imm) for one const per cycle in these 10 cycles. Flow capacity in decile 0 is sufficient (0/1 used).

## Why converting const→add_imm SHOULD reduce makespan (but CP-SAT isn't finding it)

**The optimization opportunity:**

- First decile: 19 cycles (0–18)
- Load capacity: 19 × 2 = **38 slots**
- Current Load usage: **38 ops** (20 const + 18 other) = **100% utilization**
- Flow capacity: 19 × 1 = **19 slots**
- Current Flow usage: **0 ops** = **0% utilization**

**The key insight:** Const ops have **no dependencies** — they can run at cycle 0 or earlier. Many const ops are scheduled at cycles 1, 3, 7, 8, etc., and have ALU/VALU consumers that run 1 cycle later.

**Example:** const id=22 at cycle 3 → consumer (valu vbroadcast id=23) at cycle 4
- Const has no deps, Flow at cycle 2 is free (0/1)
- If we convert const to add_imm and schedule it at cycle 2, consumer can run at cycle 3
- **Saves 1 cycle!**

**Why this should work:**
1. Const ops have no dependencies — they can be scheduled at cycle 0 or earlier
2. Flow slots are free in early cycles (0/1 utilization in decile 0)
3. **Early cycles have massive unused capacity:**
   - Cycles 0-4: ALU 0-4/12 (0-33%), VALU 0-2/6 (0-33%), Load 2/2 (100%), Flow 0/1 (0%)
   - Tons of free ALU/VALU slots available!
4. Moving const earlier via add_imm (using free Flow) allows ALU/VALU consumers to run in early cycles where there's plenty of capacity
5. This should reduce makespan by at least 1 cycle

**Example opportunities:**
- const id=24 at cycle 7 → consumer (valu vbroadcast id=25) at cycle 8
  - Could move const to cycle 0-4 via add_imm (Flow free), consumer to cycle 5-7 (VALU has capacity)
  - Saves 1-3 cycles!
- const id=0 at cycle 0 → consumer (valu vbroadcast id=38) at cycle 1
  - Already at cycle 0, but using Load (saturated). Could use Flow instead, freeing Load for data loads

**Why CP-SAT might not be finding it:**
- ConstChoice hint biases toward Load (all consts use Load)
- Search might not explore flipping ConstChoice variables when Load is saturated
- Objective (makespan) might not be sensitive enough to detect this pattern
- May need secondary objective or branching priority to prefer add_imm when Flow is free and Load is saturated

**Conclusion:** This appears to be a **search/exploration issue**, not a constraint issue. The model allows this optimization, but CP-SAT isn't finding it. Adding a secondary objective or branching strategy to prefer add_imm when Load is saturated (or Flow is free) might help the solver discover this pattern.

## Expectation

The CP-SAT model already allows this: ConstChoice has optional Load and optional Flow intervals, and Flow is uncongested in the first decile. So the solver *can* choose add_imm for some of these consts. **If it does not find this optimization, this indicates a bug in the search/exploration** (e.g. hint bias, branching order, or objective not properly rewarding Load balance). Adding a secondary objective or branching priority to prefer add_imm when Load is saturated (or Flow free) could make the solver more likely to find this pattern.

## Third option: ALU-derived constants

ConstChoice also supports a **third encoding**: produce a constant via one ALU op from two previously materialized constants (e.g. 4 = 2+2 or 2×2, 3 = 2+1). When chosen, the op uses an ALU slot and is scheduled after both operand producers (enforced by CP-SAT). This can free Load slots in early cycles where ALU is underused, so more consts or data loads can be scheduled earlier. The solver decides per ConstChoice op whether to use Load, AddImm, or an ALU derivation when operands are available.
