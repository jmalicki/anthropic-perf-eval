# CP-SAT variable modelling review (November 2026)

Review of how `scheduler_rs/src/cpsat_scheduler/model.rs` declares decision and auxiliary
variables, with proposals for remodelling them. No code was changed.

**Line references** are to `model.rs` at commit `bed155e` (4,252 lines). The working tree
was being edited by another session during this review (uncommitted changes add about 50
lines near the top and in section 6c-bis), so current line numbers may be offset.

**Evidence labels** used throughout:

- **[V]** verified: by reading the cited code, by a solver log produced in this review, by
  OR-Tools 9.15 source, or by the toy experiments in `scripts/cpsat_encoding_experiments.py`.
- **[I]** inferred: follows from verified facts but was not observed directly.
- **[S]** speculative: plausible, untested.

The toy experiments are a miniature of the kernel (independent chains of vector ops that
can run as 1 vector slot or 8 scalar slots). They are single runs with one seed and a 20 s
limit; treat differences under a factor of two as noise.

## 1. Summary of top findings

1. **The scalar alternative of a vector op costs about 70% of all variables and most of
   the linear constraints, and its inactive copy is left free.** Each shiftable op declares
   11 variables besides its cycle variable (shift Boolean, 8 lane cycles, completion,
   "chosen cycle") plus a liveness start variable and 9 optional intervals. At 16 rounds
   that is about 78,000 of 111,276 variables for 7,106 shiftable ops, of which the greedy
   hint shifts 724. When an op is not shifted its 8 lane variables and completion variable
   are still free; when it is shifted its main cycle variable is free. CP-SAT's own
   symmetry detection finds the lanes as orbits of size 8 **[V]**.
2. **The liveness (scratch) cumulative under-approximates.** A value's interval ends at
   the cycle of *one* reader chosen before solving (the reader latest in the greedy hint),
   not at the maximum over readers, and it uses that reader's *main* cycle variable even
   when the reader is shifted and actually reads at its lane cycles. The solver may
   therefore return schedules the model considers within budget that address assignment
   then rejects. This is the same class as the bug just fixed (a rule applied to one
   representation and not the other). A model-faithful toy replica returned a schedule
   with true peak 168 words under a 144-word budget **[V toy; I for the Rust model]**.
3. **Three block-level changes fix both and shrink the model**: (a) pin the inactive
   alternative, (b) order an op's lanes `t_0 <= ... <= t_7`, which is a dominance and loses
   nothing under today's precedence structure, (c) optionally use two half-blocks of
   demand 4 instead of eight lanes of demand 1, which keeps the capacity floor (3 x 4 = 12).
   Precedence is then one unconditional statement per block, and "first write" and "last
   read" are `t_0` and `t_last` with no min/max variable. In the toy, two sorted blocks
   reached equal or better makespans than the current encoding in 0.1 to 1 s instead of
   1.4 to 15 s, with 72% fewer variables **[V toy]**. Forcing all eight lanes into one cycle
   (one block of demand 8) proves optimality fast but raises the capacity floor from
   `N/7.5` to `N/7`; it cost 1 to 2 cycles in 29 to 43 on the toy **[V toy]**.
4. **Every ALAP is exactly greedy cycle + 15** (12,356 of 12,356 ops in the 16-round run)
   **[V]**. The model is a neighbourhood of the greedy schedule, not the full problem: an
   "OPTIMAL" status is optimal only within that neighbourhood, and a schedule shape
   different from the round-serialised hint is unreachable. On the ASAP side the windows
   are very wide (median about 870 cycles at 16 rounds; 10.7 million (op, cycle) pairs), so
   a time-indexed encoding is out of reach at full size.
5. **The tie-breaker dominates solve time on small instances.** On 2 rounds / batch 16 the
   optimal makespan (36, equal to the critical path) was found at 0.62 s; the remaining
   6.9 s of the 7.5 s run were spent improving the sum-of-cycles term **[V]**.
6. **Hints**: at 2 rounds / batch 16 CP-SAT reports "The solution hint is complete and is
   feasible" **[V]**. By code reading, several auxiliary variables are hinted with values
   that are wrong when a select is hinted as multiply-add or when an op is hinted as
   shifted, and two CSE variables per group are not hinted at all **[V code; effect I]**.
   Whether the 16-round hint is accepted as complete could not be determined here.
7. **Auxiliary endpoints must be functionally determined.** In the toy, liveness intervals
   whose start or end was only bounded by inequalities (leaving the objective or the
   cumulative to "pull" them) were markedly worse (with two blocks: makespan 42 and 36
   with `>=` ends, 32 and 50 with max-equality ends but a slack start, against 28 and 31
   with exact endpoints) **[V toy]**. The current model has such slack variables:
   `shifted_dest_start` when shifted, `completion` when not shifted, `cse_sub_end`.

Ranked proposals are in section 10; the structural safeguard is section 11; what could
not be determined is section 12.

## 2. Scale: where the variables go

Counts for 16 rounds, batch 256, derived from `scheduler_output_c16.json` (12,356 SSA ops:
7,106 shiftable vector ops, 2,400 other vector ops, 2,176 loads, 448 selects, 162 scalar
ops, 64 stores) and the declarations in `model.rs`. The total matches the measured
111,276 to within rounding of the select-related terms **[I]**.

| Family | Declared at | Per unit | Count at 16 rounds | Share |
|---|---|---|---|---|
| Main cycle `c{idx}` | 455-463 | 1 per op | 12,356 | 11% |
| Shift Boolean | 598 | 1 per shiftable | 7,106 | 6% |
| Lane cycles `alu_c{idx}_{i}` | 632-633 | 8 per shiftable | 56,848 | 51% |
| Completion `completion_c{idx}` | 643-644 | 1 per shiftable | 7,106 | 6% |
| Chosen cycle (tie-breaker) | 2815-2818 | 1 per shiftable | 7,106 | 6% |
| Liveness start `shifted_dest_start_` | 1910-1913 | 1 per shiftable value | about 7,100 | 6% |
| Liveness size `live_size_` | 1938-1939 | 1 per value with a reader | about 12,300 | 11% |
| Select / const / CSE / makespan | 6b, 6c, 5d, 10 | several | about 2,000 | 2% |

Intervals: 9 optional per shiftable op (1 vector, 8 scalar) gives 63,954 of the 65,593
optional intervals. Linear constraints: each precedence edge into a shiftable op from a
shiftable op costs 16 enforced constraints plus 1 (section 5b, 698-710, and 5c, 734);
with 26,833 precedence edges this family is of the order of 200,000 to 300,000 of the
470,000 linear constraints **[I]**.

Two further size effects found while reading:

- The Rust binding adds a linear constraint `start + size == end` for **every** interval,
  including fixed-size ones where it reads `c + 1 == c + 1`
  (`~/.cargo/git/checkouts/cp_sat-*/165da6f/src/builder.rs` 698-721, 733-742, 769-797)
  **[V]**. That is about 80,000 trivially true or duplicate rows that presolve must remove;
  `model.rs` 1942 adds the same equality a second time for liveness intervals. The
  2-round log shows `[DetectDuplicateConstraints] #duplicates=283` **[V]**.
- `horizon` is `max(greedy + 20, max-latency critical path, n)` (151-166). The `n` term is
  12,356 at 16 rounds, about six times the greedy makespan **[I]**; at 2 rounds / batch 16
  the log shows horizon 183 against a greedy makespan of 44 **[V]**. Main cycle variables
  are unaffected (their ALAP is greedy + 15) but `makespan`, `max_cycle`, every
  `live_size`, `shifted_dest_start`, `cse_sub_cycle`, `cse_sub_end` and the objective
  scale take their upper bound from `horizon`.

## 3. Family 1: cycle variables and their domains

### 3.1 Today

- One `IntVar` per op with domain `[final_asap, final_alap]` (455-463).
- ASAP: longest path with +1 per edge (`compute_ssa_asap`, bounds.rs 621).
- ALAP: backward pass from `horizon`, then intersected with `greedy_cycle + 15` (202-233),
  then symmetry and energetic tightening (235-292), precedence fixpoint (381-403), then
  widened to contain the greedy value (405-436).
- Lane, completion and chosen-cycle variables use the **pre-tightening** `asap`/`alap`
  maps, not `final_*` (625-626, 2812-2814), so their domains differ from the main
  variable's.

Measured window widths (ALAP - ASAP + 1) from the exported `Domain` constraints **[V]**:

| Instance | Engine | Ops | p10 | Median | p90 | Sum of widths |
|---|---|---|---|---|---|---|
| 16 rounds, batch 256 | vector, shiftable | 7,106 | 135 | 875 | 1,579 | 6,145,007 |
| | vector, other | 2,400 | 121 | 871 | 1,564 | 2,064,383 |
| | load | 2,176 | 419 | 852 | 1,635 | 1,903,413 |
| | flow | 448 | 170 | 1,331 | 1,554 | 382,842 |
| | all | 12,356 | | | | 10,725,939 |
| 2 rounds, batch 16 | all | 134 | 16 | 22 | 24 | 2,885 |

In the 16-round export `ub - scheduled cycle = 15` for all 12,356 ops **[V]**.

### 3.2 Weaknesses

- **The +15 cap is a search restriction, not a tightening.** The comment at 203-206 calls
  it a valid upper bound; it is valid only for the makespan, not per op. A schedule in
  which some group runs its round later than in the greedy while others run earlier
  (which is what removing round barriers requires) is outside the domains. Bounds and
  "OPTIMAL" are relative to this neighbourhood **[V code; consequence I]**.
- **The ASAP side is honest and wide.** Groups are independent chains, so an op's
  ancestors are only its own group's earlier ops; precedence-closed energy bounds on ASAP
  are therefore weak **[I]**. The real lower bound comes from total capacity, which no
  per-op window can express.
- With median width 870, branching on a start variable removes one value at a time from
  the solver's point of view unless a propagator (timetable) backs it up. CP-SAT encodes
  integer bounds lazily as literals `x >= v`, so a wide domain is not itself expensive,
  but each conflict explanation over a cumulative with 9,500 unit tasks is long **[I]**.

### 3.3 Alternatives

| Option | Formula | Size at 16 rounds | Assessment |
|---|---|---|---|
| A. Keep integer starts (today) | `c in [asap, alap]` | 12,356 vars | Only option that fits at full size |
| B. Time-indexed Booleans for every op | `x[o,t]`, `sum_t x[o,t] = 1`, `sum_o x[o,t] <= cap` | 10.7 M Booleans | Not feasible |
| C. Time-indexed for the vector engine only | as B on vector ops, channel `c = sum t*x[o,t]` | 8.2 M Booleans | Not feasible at 16 rounds; 1,900 at 2 rounds / batch 16 |
| D. Time-indexed inside an LNS window | fix ops outside `[t0, t0+W)`, time-index inside | `ops_in_window x W` (W = 40: about 250 vector ops, 10,000 Booleans) | Feasible; needs the windowed model of the pickup assessment |
| E. Coarse time buckets | Boolean `b[o,k]`: op o in bucket k of width W; per-bucket linear capacity `sum demand <= cap*W` | `ops x windows/W` | Gives the LP per-window demand counters without full time-indexing; see section 9 |

Why a time-indexed form would help where it fits **[I, standard result]**: the LP
relaxation of `sum_o x[o,t] <= cap` is the strongest known for unit-duration tasks, and
mode choice becomes linear (`x_vec[o,t] + sum of scalar x = 1`). CP-SAT's cumulative has
only an aggregate energy row in the LP unless cuts are generated.

Recommendation: keep A for the monolith; make the +15 slack a parameter and report it in
every result; try C on 2 and 4 rounds purely as a bound experiment; use D if and when a
windowed model exists.

### 3.4 Risk and test

- Widening the ALAP slack increases model size but cannot remove solutions. Test on 4
  rounds: slack 15, 40, unbounded; compare best makespan and bound at a fixed time.
- C is additive (channeling constraints only) and can be checked by solving 2 rounds /
  batch 16 both ways and comparing optimal makespans.

## 4. Family 2: alternative modes for a vector op

### 4.1 Today

For each shiftable op `o` (section 6, 597-662):

| Object | Declaration | Lines |
|---|---|---|
| `shift_o` | Boolean | 598 |
| Vector interval | optional, start `c_o`, size 1, present iff `!shift_o` | 621-623 |
| 8 lane cycles `a_o,i` | `IntVar [asap, alap]` | 631-634 |
| 8 scalar intervals | optional, start `a_o,i`, size 1, present iff `shift_o`, demand 1 | 635-637 |
| `completion_o` | `IntVar`; `completion >= c_o`; `completion >= a_o,i` for all i | 643-654 |
| `chosen_o` | `= completion_o` if shift, `= c_o` otherwise (enforced equalities) | 2811-2829 |
| `shifted_dest_start_o` | `<= a_o,i` for all i; `= c_o` if not shifted | 1910-1926 |

Precedence is written per representation:

- Main variable: `c_o >= c_d + 1` for a non-shiftable producer (480-496), and
  `c_o >= completion_d + 1` for a shiftable producer (726-740), both unconditional.
- Lanes: for a shiftable producer, `a_o,i >= a_d,i + 1` enforced by `shift_d` and
  `a_o,i >= c_d + 1` enforced by `!shift_d` (698-710); for other producers
  `a_o,i >= c_d + 1` (712-716).
- Extra latency of a multiply-add select: added to the main variable and, since the recent
  fix, to each lane (1305-1316).
- Eliminable broadcast: lane-only enforced pairs (821-843).
- CSE shared subtract: uses `completion` or `c` of the producer (1180-1203); the uncommitted
  change in the working tree adds the multiply-add latency here too.

### 4.2 Weaknesses

1. **Inactive alternative is free.** Not shifted: the lanes satisfy only
   `a_o,i >= c_o` (via 1917 and 1921-1925) and `completion_o >= a_o,i`; nothing pins them
   to `c_o`. Shifted: `c_o` satisfies only its own precedences and `c_o <= completion_o`.
   Each free variable multiplies the number of solutions that are the same schedule, and
   each must be assigned by search and included in every hint **[V code]**.
2. **Lane symmetry.** The 8 lanes of an op are interchangeable as long as the permutation
   is applied consistently along the lane-wise precedence. The 2-round log reports
   "57 orbits on 450 variables with sizes: 8,8,8,..." and "Found orbitope of size 6 x 8"
   **[V]**. CP-SAT exploits detected symmetry only partially (orbitope fixing, the
   `max_lp_sym` worker).
3. **Meaning depends on the mode.** `c_o` is "the cycle" when not shifted and nothing when
   shifted, yet these constraints use it unconditionally: the symmetry orderings
   `c_i <= c_j` (510, 572), the liveness end of every value whose designated last reader is
   `o` (1837, 1878), `max_cycle >= c_o` (2594-2597), the decision strategy phase 2
   (3357-3360). The first two are wrong when `o` is shifted **[V code]**.
4. **Precedence stated several times.** Five places add a "reader after producer"
   constraint, each choosing between `c`, `completion` and lanes by hand. The bug just
   fixed and the CSE gap being fixed in the working tree are both omissions of one term in
   one of these places.
5. **Enforcement literals on precedence.** About two thirds of lane precedences carry a
   literal. An enforced linear constraint propagates only once its literal is true and
   contributes nothing to the LP until then **[I, from how CP-SAT loads enforced linear
   constraints]**.
6. **`completion >= c_o` when shifted** forces nothing useful, and
   `c_o >= completion_d + 1` on the main variable of a shifted reader is redundant with
   its lanes; they only constrain the meaningless variable.

### 4.3 What optimality is lost by co-scheduling lanes

Scalar capacity is 12 and a shifted op needs 8. With free lanes (or any split whose
block demand divides 12) the scalar engine absorbs 1.5 shifted ops per cycle; with all
eight lanes in one cycle it absorbs 1.

| Encoding | Scalar ops absorbed per cycle | Joint floor, 9,506 vector-op equivalents | Lost |
|---|---|---|---|
| 8 free lanes (today) | 1.5 | 9,506 / 7.5 = 1,268 | reference |
| 2 half-blocks of demand 4 | 1.5 | 1,268 | lane-level pipelining only |
| 1 block of demand 8 | 1 (plus 4 slots for native scalar ops) | 9,506 / 7 = 1,358 | 7% of the floor |

**[V arithmetic]**. Half-blocks reach 1.5 per cycle with a two-cycle pattern: op A all 8
in cycle t, op B 4 in t and 4 in t+1, op C all 8 in t+1. What half-blocks cannot express
is one lane of an op running a cycle ahead of its three block-mates; no evidence was found
that this matters, and the exported `valu_shift_debug.clustered_count` /`spread_count`
fields in the scheduler JSON can quantify how often CP-SAT solutions actually spread lanes
(not analysed here).

### 4.4 Alternatives

**(a) Pin the inactive alternative** (no loss):

```
!shift_o  =>  a_o,i = c_o   for all i          (lanes follow the main variable)
 shift_o  =>  c_o = a_o,0                      (or drop c_o: see (c))
```

**(b) Sort the lanes** (dominance, no loss under today's precedences):

```
a_o,0 <= a_o,1 <= ... <= a_o,7
```

Argument: lane-wise precedence is `a_o,i >= a_d,i + 1` with the same index on both sides
(`element_dep_map` is the identity, 667-680), and every other producer constrains all
lanes equally. Sorting the lane times of every op independently preserves all such
constraints (if `x_i >= y_i + 1` for all i under some pairing then the sorted sequences
satisfy it too) and leaves scalar-slot usage per cycle unchanged. So any feasible schedule
has an equivalent one with sorted lanes. This removes the 8! permutations per op and makes
`first write = a_o,0`, `last write = a_o,7` plain variables. It stops being valid if a
future change makes a reader depend on *different* scalar producers per lane (for example
lane-wise precedence from the scalar loads of a gather); that would need the ordering
dropped for those ops.

**(c) One set of execution-time variables for both modes** ("unified"):

```
t_o,0 .. t_o,K-1                         K = 8, 2 or 1; t_o,0 is today's c_o
vector interval:  start t_o,0, present iff !shift_o, demand 1 (vector engine)
scalar intervals: start t_o,j, present iff  shift_o, demand 8/K (scalar engine)
t_o,0 <= ... <= t_o,K-1
!shift_o => t_o,K-1 = t_o,0
precedence, every mode combination:   t_o,j >= t_d,j + 1 + extra_d      (K rows per edge)
```

No completion variable, no chosen-cycle variable, no start variable; `finish_o = t_o,K-1`.
A non-shiftable op has K = 1 and `t_d,j` means `t_d,0`.

**(d) Single optional interval of demand 8 plus a spread variable**: `t_o` and
`spread_o in [0, S]`, scalar interval of size `1 + spread` and demand 8. This
over-reserves (8 slots for every cycle of the spread) and so is strictly worse than (c)
with K = 2; not recommended.

**(e) Variable demand instead of optional intervals**: one always-present interval per
engine with demands `1 - shift_o` and `8/K * shift_o`. The binding accepts linear demands
(`builder.rs` 850-861) **[V]**. Removes two thirds of the optional intervals; whether
CP-SAT's timetable propagates a variable-demand unit task as well as an optional one was
not tested **[S]**.

Counts per shiftable op:

| Encoding | Extra vars | Intervals | Rows per shiftable-to-shiftable edge | Vars at 16 rounds (total model) |
|---|---|---|---|---|
| Today | 11 (+1 liveness start) | 9 | 17 (16 enforced) | 111,276 |
| (a)+(b), K = 8 | 8 | 9 | 8 | about 90,000 |
| (c), K = 2 | 2 | 3 | 2 | about 43,000 |
| (c), K = 1 | 1 | 2 | 1 | about 36,000 |

### 4.5 Toy results

`scripts/cpsat_encoding_experiments.py`, experiments A and F; project solver parameters;
20 s; no hint. "s to best" is the time at which the final makespan was first reached.

| Instance (ops) | Encoding | Vars | Makespan | Bound | s to best | Status |
|---|---|---|---|---|---|---|
| G=8 S=8 (200) | today (replica) | 2,121 | 29 | 27 | 15.4 | feasible |
| | unified K=8, unsorted | 1,737 | 31 | 27 | 9.8 | feasible |
| | sorted K=8 | 1,737 | 29 | 27 | 7.5 | feasible |
| | unified K=2 | 585 | 28 | 27 | 0.1 | feasible |
| | sorted K=2 | 585 | 28 | 27 | 0.7 | feasible |
| | K=1 | 393 | 29 | 29 | 0.0 | optimal, 3.6 s |
| G=16 S=6 (304) | today (replica) | 3,185 | 41 | 40 | 3.4 | feasible |
| | unified K=8, unsorted | 2,609 | 43 | 40 | 0.4 | feasible |
| | sorted K=8 | 2,609 | 41 | 40 | 19.9 | feasible |
| | unified K=2 | 881 | 41 | 40 | 0.2 | feasible |
| | sorted K=2 | 881 | 41 | 40 | 1.0 | feasible |
| | K=1 | 593 | 43 | 43 | 0.1 | optimal, 11.2 s |
| G=32 S=6 (608) | today (replica) | 6,369 | 81 | 78 | 1.4 | feasible |
| | unified K=8, unsorted | 5,217 | 80 | 78 | 4.4 | feasible |
| | unified K=2 | 1,761 | 79 | 78 | 0.9 | feasible |

Reading **[V toy]**: K = 2 is consistently as good or better and faster. K = 8 with pinning
alone, or with sorting, is not clearly better than today's encoding on these instances;
the gain comes from having fewer lanes, not from pinning by itself. K = 1 is the only
encoding that proves optimality, at a makespan 0 to 2 cycles worse. No encoding closed
the last cycle of gap for K >= 2 in 20 s.

### 4.6 Risk and test

- (c) changes extraction (`shifted_alu_cycles` becomes K values repeated 8/K times) and
  every post-solve validator; the lowering already accepts arbitrary per-lane cycles.
- Test in isolation with the existing 2-round and 4-round CP-SAT tests plus the dataflow
  checker: K = 8 sorted must give the same optimal makespan as today on 2 rounds (36 on
  batch 16); K = 2 must give the same or one more.
- Keep K a config field so the three variants can be A/B-run on 4 rounds.

## 5. Family 3: two-way encodings (ConstChoice, SelectChoice)

### 5.1 Today

**ConstChoice** (6b, 884-1001): binary case has one Boolean and two optional intervals on
the same start `c` (load engine if false, flow engine if true) (904-917). N-way case has
one Boolean per alternative, `sum = 1` as a linear equality (935-942), one optional
interval per alternative, and enforced precedences to the two producer constants for each
ALU-derived alternative (973-980). All alternatives share the start variable.

**SelectChoice** (6c, 1033-1139): one Boolean `choice` (true = subtract + multiply-add).
Flow interval at `c` if false. If true: with a precomputed delta, one vector interval at
`c`; otherwise a subtract interval at `c` and a multiply-add interval at
`fma_cycle = c + 1`, where `fma_cycle` is a separate `IntVar` tied by an unconditional
equality (1122-1131). In a CSE group the subtract interval is replaced by one shared
optional interval per group with its own `cse_sub_cycle` and `cse_sub_active`
(1145-1262).

**Mode-dependent latency** (5d, 1269-1321): per (select, reader) pair a 0/1 `IntVar`
`select_fma_extra` constrained equal to `choice`, used in
`reader >= c_select + 1 + extra` for the reader's main variable and each lane.

**CSE activity** (1213-1235): `sub_active => sum(choice) >= 1`,
`!sub_active => sum(choice) = 0`, and `choice_c => sub_active` for each consumer.

### 5.2 Weaknesses

- `select_fma_extra` duplicates a Boolean as an integer once per reader. Presolve
  substitutes it, so the cost is model size and hint surface, not search **[I]**. The
  working tree's new helper `select_fma_extra_delay` already uses the Boolean directly in
  the linear expression for the CSE case.
- `fma_cycle` is an affine image of `c`; an interval can take `c + 1` as its start
  expression directly (the binding accepts `LinearExpr` starts, `builder.rs` 814-824)
  **[V]**.
- The N-way "exactly one" is a linear row, not `add_exactly_one`. Presolve converts it
  (the 2-round presolved model shows `#kExactlyOne: 6`) **[V]**, so this is cosmetic.
- `max_cycle >= fma_cycle` is unconditional (2602-2605), so a flow-path select in the last
  cycle would add a cycle. Harmless in practice because every select has a reader
  **[I]**, but it makes the conditional `makespan_ge_select_*` rows (2748-2777) redundant.
- `cse_sub_cycle` has domain up to `horizon - 1` and no unconditional upper link to
  `max_cycle`; when the group is inactive it is free. See 4.2 item 1 and section 7.
- `cse_sub_end` (2093-2132) is bounded below only, and only for active consumers; the
  interval size equality is enforced only when active (2139-2145). It is a slack endpoint.
- The CSE producer precedence had no multiply-add latency term until the working-tree
  change: the same omission class again.

### 5.3 Alternatives

1. **Latency as a property of the producer**, not of each edge:
   `visible_d = t_d,last + 1 + extra_d` where `extra_d` is `choice_d` (a literal used
   directly as a 0/1 term), 0, or 1. Every reader constraint is `t_r,j >= visible_d,j`.
   Removes all `select_fma_extra` variables and makes the +2 impossible to forget (see
   section 11).
2. **Drop `fma_cycle`**: use `c + 1` as the interval start and `c + choice` as the op's
   finish.
3. **CSE subtract as a pinned optional**: `!sub_active => sub_cycle = sub_lb` (its domain
   minimum) so the inactive value is fixed; `sub_end = max over consumers` via
   `add_max_eq` over `c_consumer + 2*choice_consumer` is not linear in general, so the
   practical form is: `sub_end >= c_k + 2` enforced by `choice_k` (as today) **plus**
   `sub_end <= max_k(c_k) + 2` and `!sub_active => sub_end = sub_cycle + 1`.
4. **ConstChoice**: nothing structural to gain. With 46 such ops at 16 rounds the family
   is negligible.

Expected effect **[I]**: a few hundred fewer variables and constraints at 16 rounds (448
selects). The value is in correctness and hint completeness, not speed.

### 5.4 Risk and test

Low. Each change is an equivalence. Test: 2-round and 4-round optimal makespans unchanged;
`verify_select_choice_fma_consumer_precedence` and the new `verify_cse_sub_precedence`
still pass; force-FMA and force-vselect modes still solve.

## 6. Family 4: liveness / scratch

### 6.1 Today

For each value with a definition and a reader (8b, 1639-2166):

| Part | Definition | Lines |
|---|---|---|
| Last reader | **one** op index, chosen before solving: the reader with the latest greedy-hint cycle, or the last in index order without hints | 1660-1688 |
| End | `c[last_reader] + 1`; `+2` if that reader is a forced multiply-add select; an `IntVar` with two enforced equalities if the select is a choice | 1829-1881 |
| Start | `c[def]`, or for a shiftable def an `IntVar` `start <= a_i` for all i with `start = c[def]` enforced when not shifted | 1895-1934 |
| Size | `IntVar [1, horizon+1]` with `start + size = end` | 1938-1943 |
| Interval | `new_interval_var(start, size, end)`, always present, demand 1 or 8 | 1945-1951 |
| Segment mode | live-out: end = `segment_end + 1`; live-in: start fixed 0; live-through: `[0, segment_end+1]`; `segment_end >= c` for every op | 1766-1785, 1954-2047 |
| Select temporaries | optional fixed-size-2 interval; CSE group: optional interval `[sub_cycle, sub_end)` | 2052-2156 |

One cumulative with capacity `scratch_budget` (2165).

### 6.2 Weaknesses

1. **Under-approximation, designated reader.** No constraint makes the designated reader
   the last one. 2,770 of 12,339 values at 16 rounds have two or more readers **[V]**. If
   the solver schedules another reader later, the modelled interval ends early
   **[V code]**.
2. **Under-approximation, shifted reader.** The end uses `c[last_reader]`. When that
   reader is shifted it reads at its lane cycles, up to `completion`; `c` is then free
   between its precedence lower bound and `completion`, and the solver is rewarded for
   setting it low **[V code]**.
3. The backstop is address assignment after the solve, which fails the build with
   `BudgetExceeded` (`tick_tock_ssa/mod.rs` 589, 1498). So the effect is rejected solves,
   not wrong kernels **[V code]**. At 16 rounds the hint's peak is 1,268 of 1,536 words,
   so the hole may not have been hit yet; it becomes relevant exactly when the solver
   starts overlapping rounds **[I]**.
4. **Slack start when shifted.** `start <= a_i` with no lower link; the comment at
   1927-1929 relies on "the objective" to push it, but the objective does not contain it.
   A too-small start only over-reserves (sound), yet it is one more free variable per
   shifted op that search must set.
5. **Half-open against closed.** The model keeps a value through the whole cycle of its
   last read, so a write in that cycle cannot reuse the address; address assignment allows
   it (comment 1799-1809). The model is conservative by up to one cycle per value here,
   while being optimistic in items 1 and 2.
6. **Size variables.** CP-SAT intervals need affine start, size and end, so with two
   different variables as endpoints a size variable is unavoidable. It is not a search
   burden if both endpoints are determined: it is then fixed by propagation **[I]**.

Toy evidence (experiment C and F, G=8 S=8, 200 ops) **[V toy]**:

| Budget | Encoding | Makespan | True peak | Sound |
|---|---|---|---|---|
| 144 | today's replica (designated reader, main var) | 31 | **168** | no |
| 144 | today's lanes, end >= every read | 31 | 144 | yes |
| 144 | K=2 unsorted, slack start, end >= | 36 | 136 | yes |
| 144 | K=2 unsorted, slack start, end = max | 50 | 96 | yes |
| 144 | K=2 sorted, exact endpoints | 31 | 144 | yes |
| 144 | K=8 sorted, exact endpoints | 32 | 144 | yes |
| 192 | today's replica | 30 | 192 | (not binding) |
| 192 | K=2 unsorted, slack start, end >= | 42 | 120 | yes |
| 192 | K=2 sorted, exact endpoints | 28 | 192 | yes |
| 192 | K=1, exact endpoints | 29 (optimal) | 176 | yes |

The rows with slack endpoints are the worst in the table. The cumulative does not pull an
endpoint; it only fails when the interval is too large, and the search then has to
discover that shrinking it is free.

### 6.3 Alternatives

1. **Exact endpoints from the unified op times** (needs 4.4 (b) or (c)):

   ```
   start_v = t_def,0                                      (no variable)
   end_v   = t_r,last + 1 + reads_late_r                  if v has one reader (no variable)
   end_v   = max over readers r of (t_r,last + 1 + reads_late_r)   via add_max_eq otherwise
   ```

   `reads_late_r` is 1 for the operands a multiply-add select reads at `c + 1` (as today).
   77% of values have one reader, so about 2,800 end variables at 16 rounds instead of
   about 7,100 start variables, and no slack anywhere.
2. **`>=` constraints instead of `add_max_eq`** for the end: sound, fewer propagators, but
   leaves the end free above. The toy says avoid slack on these endpoints. If used, add
   the upper side `end_v <= makespan` at least.
3. **Lazy / partial scratch**: include only values that can be live when pressure is high.
   A value whose whole window `[asap_def, alap_last_reader]` lies where the hint's
   liveness plus a margin is far below budget can be left out and checked after the solve,
   adding it back on violation (Roorda's lazy register constraints, cited in the pickup
   assessment). At 16 rounds the hint peaks at 1,268 with most cycles well below; the
   reduction could be large but depends on how far CP-SAT moves ops, which the +15 cap
   currently bounds **[S]**.
4. **Dominance for single-reader scalar temporaries**: none found that is valid in
   general; skipping.

### 6.4 Risk and test

- Alternative 1 makes the model strictly tighter than today on liveness, so instances that
  "solved" by exploiting the hole will get worse or infeasible. That is the correct
  outcome; expect the per-section feasibility runs to be the first place it shows.
- Isolated test, no solver change needed: after any CP-SAT solve, recompute peak liveness
  from the solution with the address-assignment rules and assert it equals the peak
  implied by the model's own interval variables. Today that assertion should fail on some
  4-round runs if the hole is ever exercised.
- Unit test: three ops, value `x` read by `r1` (hinted later) and `r2`; budget such that
  `r2` after `r1` overflows. Today's model accepts it.

## 7. Family 5: makespan and objective

### 7.1 Today

- `max_cycle in [cap_lb - 1, horizon - 1]`, `makespan = max_cycle + 1` (2587-2593).
- `max_cycle >= c` for every op, `>= completion` for every shiftable op, `>= fma_cycle`
  for every select; all unconditional (2594-2605).
- Per-op `makespan >= ... + 1`, enforced by the mode literal where relevant (2737-2782).
- Redundant capacity rows: load and store constants (2614-2625); `6*makespan + s - sel >=
  valu_count`, `12*makespan - 8*s >= alu_count`, flow and load with choice sums
  (2668-2731), where `s` is the number of shifted ops.
- Objective `scale * makespan + sum(effective completion of every op)` with `scale` the
  next power of ten above `n * horizon`, and `absolute_gap_limit = scale / 3`
  (2907-2934, 3289-3296).

### 7.2 Weaknesses

- Since `makespan = max_cycle + 1` and `max_cycle >=` everything unconditionally, all the
  per-op rows at 2737-2782 are implied: about 12,356 + 7,106 rows, 14,000 of them
  enforced **[V code]**.
- The unconditional rows also bind inactive variables (`completion` of an unshifted op,
  free lanes via `completion`), so the makespan can be held up by a variable that
  represents nothing until search moves it.
- **Tie-breaker.** 2 rounds / batch 16, this review's run **[V]**:

  | Event | Time |
  |---|---|
  | Hint accepted, makespan 43 | 0.07 s |
  | Makespan 36 (= critical path, the lower bound) | 0.62 s |
  | Last tie-breaker improvement | 1.48 s |
  | Search ends, status OPTIMAL | 7.5 s |

  From 0.62 s on, the log prints "Absolute gap limit of 33333 reached" after every
  solution yet the run continues for 6.9 s. So the reported 7 s is not the time to prove
  the makespan. On the toy (experiment B) the tie-breaker made no clear difference to the
  makespan reached, so the cost is in termination and in the solution stream, not
  necessarily in makespan progress **[V toy]**.
- With 12,356 terms the objective is dense; `core`-based workers work on objective terms
  and the LP objective no longer isolates the makespan **[I]**.
- The capacity rows use fresh fixed `IntVar`s for constants (2616, 2621, 2699, 2708, 2718,
  2728): harmless, but they appear as variables in statistics and hints.
- The floor `cap_lb` is the only bound that moved at 16 rounds (1,271). The rows are
  aggregate over the whole horizon; nothing bounds a prefix or suffix of the schedule.

### 7.3 Comparison with the OR-Tools RCPSP pattern

`examples/python/rcpsp_sat.py` (v9.15) declares `makespan`, an interval
`[makespan, horizon]` with a variable size, adds that interval with demand = capacity to
each renewable-resource cumulative (lines 237-244, 301-306), constrains each task end
`<= makespan`, minimises `makespan`, and sets `push_all_tasks_toward_start = true`
(lines 370-371) **[V source]**.

What the interval buys, from `ortools/sat/linear_relaxation.cc` v9.15 **[V source]**:
`AppendCumulativeRelaxationAndCutGenerator` (770-810) looks for a makespan in two ways:
`DetectMakespan` (650-683) finds an always-present interval with demand equal to capacity
ending at the horizon; otherwise `DetectMakespanFromPrecedences` (687-740) looks for a
variable that is after the end of **every** interval of the cumulative and gives up if any
interval end is not a plain variable or is not covered. The makespan is then passed to the
energy relaxation and to the cumulative cut generators (which are added when
`linearization_level > 1`, as here).

For this model **[I]**: detection from precedences can succeed for the engine cumulatives
because `max_cycle >= c` is unconditional for every start variable; it fails for the
vector cumulative whenever CSE groups exist, because `cse_sub_cycle` has no unconditional
link to `max_cycle`, and it cannot apply to the scratch cumulative. Whether detection
succeeds on the real model was not observed (it is logged only at `VLOG(2)`).

Toy experiment D (304 ops): bound 40 and makespan 41 in all four combinations of
{redundant capacity rows, makespan interval} under both the project parameters and
default parameters **[V toy]**. So on the toy neither the interval nor the redundant rows
changed the bound; CP-SAT derived the capacity bound unaided. No evidence for or against
on the real model.

### 7.4 Alternatives

1. **Makespan-only objective**, with the tie-breaker as an optional second solve that
   fixes `makespan <= M*` and minimises the sum (or `push_all_tasks_toward_start`, whose
   proto comment reads "Experimental code: specify if the objective pushes all tasks
   toward the start of the schedule" **[V proto]**; its effect was not tested).
2. **One definition of makespan**: `makespan >= finish_o + 1` for every op, where
   `finish_o` is the unified finish of section 11; delete `max_cycle` and the conditional
   rows.
3. **Make the makespan detectable**: either add the RCPSP makespan interval to the five
   engine cumulatives, or guarantee `end <= makespan` unconditionally for every interval
   in them (pin inactive optional starts; bound `cse_sub_cycle`).
4. **Tight `horizon`**: `greedy + slack` without the `.max(n)` term when a hint exists;
   the max-latency critical-path guard is only needed for the ALAP pass, not for variable
   domains.
5. **Prefix/suffix capacity rows**: see section 9.

### 7.5 Risk and test

Alternative 1 changes which optimal schedule is returned (less left-packed), which can
change peak liveness and the hint quality for later stages. Test: 2 and 4 rounds, compare
time to OPTIMAL and peak liveness with tie-breaker on, off, and two-phase.

## 8. Families 6 and 7: symmetry and hints

### 8.1 Symmetry today

- Ordering groups (5b, 500-519): ops sharing `symmetry_hint.ordering_group`
  (`[round, iter, level]`, `emit_ssa.rs` 290) are sorted by greedy cycle and chained
  `c_i <= c_j`.
- Lexicographic streams (5c, 524-584): for each stream, the first op of consecutive items
  is chained.
- At 16 rounds the export contains 480 ordering constraints, all on tree-access
  multiply-adds (304) and selects (176) **[V]**. The hash-chain ops of the 32 identical
  groups, which are the bulk of the model, carry no ordering.
- At 2 rounds / batch 16 no ordering constraints were emitted **[V]**.

Weaknesses:

- The constraints are on `c`, which is meaningless for a shifted op (4.2 item 3). None of
  the 480 are on shiftable ops today (multiply-add and select are not shiftable), so this
  is latent **[V]**.
- For selects the order is on the start cycle regardless of mode; two selects in different
  modes are not symmetric, and ordering them is a restriction, not a symmetry break. It is
  consistent with the hint, so it cannot cut off the hint, but it can cut off better
  schedules **[I]**.
- The order is taken from the greedy schedule, which makes the "symmetry breaking" another
  greedy-neighbourhood restriction when the group members are not truly interchangeable
  (different ASAP because of different upstream modes).
- The big symmetries are untouched: lanes within an op (section 4) and whole groups
  (32 identical chains per round).

Alternatives:

1. Lane ordering, section 4.4 (b).
2. **Group ordering on a canonical op**: for two groups that are identical chains with
   identical external inputs, order one designated op per round (for example the first
   hash op) by `start_g <= start_g+1`. Valid as a symmetry break only if the groups are
   interchangeable *as whole chains*, in which case ordering one op per pair is a sound
   lex-leader on the first differing position only when earlier positions are equal; the
   safe form is to order a single op per pair of groups, once, not one per round. The
   pickup assessment's reference (Kovacs and Vancza 2006) gives the stronger progressive
   form. **[S]** for effect.
3. State every ordering on `start_o = t_o,0` of the unified representation.

Test: on 4 rounds compare conflicts and time to a fixed makespan with (i) today's
ordering, (ii) none, (iii) lane ordering only, (iv) lane plus one-op-per-group ordering;
the optimum must not change between (ii) and the others on 2 rounds.

### 8.2 Hints today

| Variable | Hinted? | Value | Lines | Problem |
|---|---|---|---|---|
| `c` | yes, unless greedy cycle is -1 | greedy cycle, clamped | 2959-2996 | dead ops left unhinted |
| `makespan`, `max_cycle` | yes | greedy max + 1, greedy max | 3004-3006 | |
| `shift` | yes | greedy decision | 3011-3036 | |
| lanes, `completion` | yes | greedy cycle for all 8 | 3058-3085 | consistent with all lanes in one cycle |
| `chosen` (shiftable) | yes | greedy cycle | 3094-3109 | |
| `select_chosen_cycle` | yes | greedy cycle | 3110-3123 | must be `cycle + 1` when the select is hinted as multiply-add (constraint 2846-2848); the comment at 3116 acknowledges it |
| `fma_cycle` | yes | cycle + 1 | 3130-3135 | |
| `select_fma_extra` | yes | choice | 3137-3139 | |
| `select_fma_end` (liveness) | yes | cycle + 1 or + 2 | 3142-3149 | |
| `shifted_dest_start` | yes | def cycle | 3153-3158 | |
| `live_size` | yes | `last - def + 1` | 2204-2220 | one too small when the last reader is a multiply-add select (end is `cycle + 2`) |
| `const_choice`, N-way Booleans | yes | load or add-immediate | 921, 993-995 | |
| `select_choice` | yes if hints or default flag | | 1081-1089 | unhinted when neither is given |
| `eliminate_vb` | yes | | 812-820 | |
| `cse_sub_active`, `cse_sub_cycle` | yes | any consumer FMA; max producer cycle + 1 | 3163-3192 | cycle hint ignores whether a vector slot is free there and ignores the producers' own latency |
| `cse_sub_end`, `cse_sub_size` | **no** | | 2093-2138 | two unhinted variables per CSE group |
| segment sizes, `segment_end` | yes | | 1777-1781, 1971-2037 | |

Observations:

- 2 rounds / batch 16: "The solution hint is complete and is feasible", first solution
  `complete_hint` at 0.07 s **[V]**. The hint there has 0 shifted ops and only 2 selects
  (their hinted mode was not inspected), so it exercises few of the problem rows above.
- An older log in the repository (`optimization_run.log`, February 2026, 29,051
  variables) shows "The solution hint is incomplete: 29035 out of 29051 non fixed
  variables hinted" and a first solution only at 50 s from `fs_random_no_lp [hint]`
  **[V, old code]**.
- What CP-SAT does with a hint, from v9.15 **[V source/proto]**: a complete, feasible hint
  is injected as a solution before search (`cp_model_solver.cc` 982-993). Otherwise each
  worker follows the hint as a fixed search until `hint_conflict_limit` (default 10)
  conflicts and then switches to normal search (`sat_parameters.proto` 1054-1060).
- Toy experiment E (304 ops, hint taken from a solved run) **[V toy]**: complete hint gave
  the first solution as `complete_hint` at 1.67 s; the same hint with all auxiliary
  variables removed (593 of 3,473 hinted) still produced a first solution of equal quality
  at 3.31 s through hint-guided search. So at this size an incomplete but consistent hint
  is recovered; the statement "an incomplete hint is nearly useless" is not supported by
  the toy. The risk at 16 rounds is that ten conflicts are not enough to complete 78,000
  auxiliary variables, and that a hint which is *wrong* (not merely missing) on a few
  variables is rejected outright.
- The 193 s to first solution at 16 rounds is consistent with either a rejected hint or
  slow presolve; the log needed to tell them apart was not available.

### 8.3 Making hint completeness checkable

1. **Generate the hint from one function** `assign_all(schedule, modes) -> value for every
   variable`, written next to the variable declarations (section 11), instead of from
   eight separate blocks.
2. **Assert, do not log**: after building, require every non-fixed variable to have a
   hint when greedy hints are supplied. The diagnostic at 3194-3266 already computes the
   list; make a non-empty list an error in tests and a warning in production.
3. **Check feasibility before solving**: evaluate every linear constraint of the proto on
   the hint in Rust (the proto is available through `model.proto()`); report the first
   violated constraint by name. All constraints are already named. Cumulatives can be
   checked with the existing per-cycle counters.
4. **Solver-side confirmation**: fail tests unless the log contains "The solution hint is
   complete and is feasible"; `debug_crash_if_presolve_breaks_hint` is already wired to an
   environment variable (3273-3275).
5. With the unified representation the hint surface shrinks to: op times, mode Booleans,
   multi-reader ends, sizes.

## 9. Family 8: redundant constraints and auxiliary variables worth adding

| Idea | Formula | Why it could help | Evidence | Cost |
|---|---|---|---|---|
| NoOverlap only for the flow slot | keep `add_no_overlap`, drop the capacity-1 cumulative (1613-1617) | the model already has both; disjunctive propagation subsumes cumulative at capacity 1 | **[I]** | removes one cumulative |
| At-most-one cliques for flow | not needed | NoOverlap over unit intervals is the clique | **[I]** | none |
| Mode literals in the LP | already present through the capacity rows (2668-2731) and optional-interval energies | | **[V code]** | none |
| Prefix capacity (head) rows | for a cycle T: `cap_e * T >= sum of demand of ops with alap < T` per engine, with mode sums | forces early ops early; with the +15 cap these are the ops whose window closes before T | **[S]** | one row per chosen T per engine |
| Suffix capacity (tail) rows | `cap_e * (makespan - T) >= demand of ops with asap >= T` | links makespan to late work; this is the row that can lift the bound above the global floor when work cannot start early | **[S]** | as above |
| Per-window demand counters | Boolean `b[o,k]` "o runs in window k" for windows of width W, `sum_k b = 1`, `c in window k` channelled by two enforced bounds, `sum_o demand*b[o,k] <= cap*W` | gives the LP a bucketed time-indexed view; `ops * (window/W)` Booleans, about 12,356 * 870/64 = 170,000 at W = 64 | **[S]** | large; try on 4 rounds first |
| Energy over precedence-closed sets | `start_o >= ceil(E(ancestors of o) / cap)` | standard head bound | weak here: ancestors of an op are its own group's chain only **[I]** | cheap but low value |
| Load-engine tail | every gather round needs 256 loads at 2 per cycle after its addresses exist: `finish_round_r >= start_addr_r + 128` | a structural bound CP-SAT cannot see from unit tasks cheaply | **[S]** | 16 rows |
| Transitive shortcuts along a chain | `c_j >= c_i + dist(i,j)` for ops d apart on one group's chain, with dist counting mode latencies at their minimum | CP-SAT's precedence propagation is already transitive on unconditional edges; gains only across enforced edges, which the unified encoding removes | **[I]** | prefer fixing the encoding |
| Shift count per window | `sum shift_o over ops with window inside [T1,T2] <= 1.5*(T2-T1+1)` (scalar capacity) | prunes mode assignments early | **[S]** | few rows |

The redundant rows that exist today and could be removed are listed in 7.2 and 2.

## 10. Ranked proposals

Effect is the expected direction, not a measured gain on the real model. "Size" is
implementation effort in `model.rs` and its consumers.

| # | Proposal | Expected effect | Size | Confirm or refute on 2 and 4 rounds |
|---|---|---|---|---|
| 1 | **Sound liveness end**: end = max over all readers of their last execution time (lanes when shifted), section 6.3 | removes a soundness hole; fewer post-solve `BudgetExceeded` | small if done on today's variables (`end >= completion_r + 1` or `c_r + 1` per reader, enforced by mode); clean once #2 exists | add the post-solve assertion "model peak = recomputed peak"; it must fail before and pass after. Compare 4-round makespan at a tight budget |
| 2 | **Unified op times with sorted lanes and pinned inactive mode**, K configurable, section 4.4 (b)(c) and 11 | at K = 8: about 20% fewer variables, half the lane precedence rows, no enforcement literals on precedence, no slack endpoints. At K = 2: about 60% fewer variables | large (sections 5, 6, 8b, 10, 11 of `model.rs`, extraction, validators) | 2 rounds / batch 16: optimum must stay 36 for K = 8 and K = 2; time to 36 and to OPTIMAL. 4 rounds: best makespan and bound at 300 s for today / K = 8 / K = 2 |
| 3 | **Makespan-only objective**; tie-breaker as an optional second phase | on 2 rounds the proof ends at about 0.6 s instead of 7.5 s; at 16 rounds a sparse objective for bound workers | small | time to OPTIMAL at 2 rounds; best bound at 4 rounds at fixed time; peak liveness of the returned schedule |
| 4 | **Hint as a checked artefact**: one generator, completeness assertion, constraint-by-constraint feasibility check, fix the three wrong or missing hints in 8.2 | a complete feasible hint is accepted at presolve; turns silent hint rejection into a named constraint | medium | force `--select-choice fma` and a shifted hint on 2 rounds and require "complete and is feasible"; time to first solution at 4 rounds before and after |
| 5 | **Report and parameterise the greedy + 15 ALAP cap**; tight `horizon` (drop `.max(n)`) | honest optimality claims; smaller auxiliary domains; lets the solver leave the greedy's shape | small | 4 rounds with slack 15 / 40 / none: makespan, bound, time; check the 16-round log line `horizon` |
| 6 | **Remove redundant rows and duplicate variables**: conditional makespan rows, `select_fma_extra`, `fma_cycle`, `max_cycle`, constant `IntVar`s, the flow cumulative duplicate, the binding's trivial interval equalities | about 100,000 fewer rows before presolve at 16 rounds; shorter presolve | small (binding change is separate) | presolve time and "#duplicates" in the log at 4 and 16 rounds; results unchanged |
| 7 | **Two half-blocks (K = 2) as the default** once #2 exists | toy: same or better makespan 5 to 100 times sooner; keeps the 7.5 floor | trivial after #2 | 2 rounds: optimum equal to K = 8 or +1; 4 rounds: makespan at 300 s; count lane spread in today's solutions (`valu_shift_debug.spread_count`) to estimate what is lost |
| 8 | **Solver workers** (from the pickup assessment; confirmed here): `interleave_search = true` gives a portfolio of `core, default_lp, fixed, max_lp, max_lp_sym, no_lp, pseudo_costs, quick_restart, quick_restart_no_lp, reduced_costs` with no `objective_shaving`, `objective_lb_search`, `lb_tree_search` or `probing` worker (2-round log; `cp_model_search.cc` 945-979 skips them under interleave) | a bound that can move | trivial | best bound at 4 rounds at 300 s with interleave on/off |
| 9 | **Makespan detectability in the LP**: makespan interval in the engine cumulatives, or unconditional `end <= makespan` for every interval | stronger energy rows and cuts if detection currently fails | small | no effect on the toy; on 4 rounds compare root bound with `linearization_level = 2`, with and without CSE groups |
| 10 | **Prefix/suffix capacity rows and the load-tail row**, section 9 | a bound above the global floor at 16 rounds | small to medium | 4 rounds: bound at 300 s; rows must not cut the known best schedule (check by evaluating them on it) |

Not ranked, with reasons: full time-indexed encoding (does not fit, section 3); single
block of demand 8 (raises the floor 7%); `>=`-only liveness ends (toy says slack endpoints
hurt); N-way exactly-one and other cosmetic forms (presolve already does them).

Decision-strategy notes that did not earn a rank: the three phases at 3336-3385 iterate
`HashMap`s, so the order of the "choose first" phase is arbitrary and differs between
runs; lane and completion variables are in no phase. From `integer_search.cc` v9.15
(1224-1243) the user strategy is the primary policy only in the `fixed` worker and the
integer-completion fallback after the SAT heuristic elsewhere **[V source]**, so the
strategy matters less than the comment at 3328 suggests. Sort the lists, or drop them
once the encoding has no free auxiliaries.

## 11. Structural safeguard: one definition of when an op runs and when its result is visible

Every bug in this class is a constraint written against one representation of an op while
another representation was the one in force. The remedy is that no constraint generator
is allowed to touch `cycle_vars`, lanes, `completion` or `fma_cycle` directly.

### 11.1 The object

One struct per op, built in one function, immediately after variable creation:

```
OpTimes {
    blocks:  Vec<LinearExpr>   // execution time of each lane block, sorted; len K or 1
    extra:   LinearExpr        // additional result latency: 0, 1, or the FMA literal
}

start(o)            = blocks[0]                        first cycle the op occupies / first write
finish(o)           = blocks[last] + extra             last cycle the op occupies
visible(o, j)       = blocks[min(j, last)] + extra + 1 first cycle lane block j of the result can be read
last_read(o, src)   = blocks[last] + reads_late(o, src)   cycle of the last read of operand src
```

with the mode invariants stated once, where the variables are created:

```
blocks[0] <= blocks[1] <= ... <= blocks[last]
!shift_o  =>  blocks[last] = blocks[0]
extra = fma_literal for a no-delta select in Optimize mode; 1 if forced; 0 otherwise
reads_late(o, src) = fma_literal for the operands the multiply-add reads at c + 1; else 0
```

Synthetic ops that are not in the SSA list get an `OpTimes` too: the shared CSE subtract
(`blocks = [sub_cycle]`, readers are the group's consumers) and an eliminated broadcast
(its readers take `visible` of the scalar producer).

### 11.2 The generators

Each takes `OpTimes` only:

| Generator | Formula | Replaces |
|---|---|---|
| `add_precedence(reader, producer)` | for each block j of the reader: `reader.blocks[j] >= visible(producer, j)` | sections 5, 5b, 5c, 5d (both), CSE producer and consumer precedence |
| `live_interval(value)` | `[start(def), max over readers of last_read(r, value) + 1)` | 1822-1952, 2052-2156 |
| `makespan` | `makespan >= finish(o) + 1` for every op | 2594-2605, 2737-2782 |
| `symmetry_order(i, j)` | `start(i) <= start(j)` | 5b, 5c |
| `objective_term(o)` | `finish(o)` | 2811-2905 |
| `hint(o, schedule)` | values for `blocks`, mode literals | section 11 of `model.rs` |
| `extract(o, response)` | per-lane cycles, visible cycle | 3404-3676 |

### 11.3 Making omissions impossible rather than unlikely

- Keep the raw variables private to the module that builds `OpTimes`; the rest of
  `model.rs` receives only the struct.
- Post-solve validators (`verify_all_raw_precedence`,
  `verify_select_choice_fma_consumer_precedence`, the new `verify_cse_sub_precedence`,
  slot limits, liveness) should compute `start/finish/visible/last_read` from the
  **solution** with an independent implementation of the same four definitions (they
  already exist in `address_assign.rs` and the dataflow checker). A mismatch between the
  model's and the validator's values is then a test failure even when the schedule
  happens to be valid.
- A property test over small random DAGs with every mode forced in turn (all vector, all
  scalar, all FMA, all vselect, mixed): solve, then replay. Each of the 2 x 2 x 2
  producer/reader mode combinations for an edge must appear at least once; the bug just
  fixed is the (FMA producer, shifted reader) cell and the working-tree fix is the
  (FMA producer, CSE-subtract reader) cell.
- A Z3 lemma in `verification/` fits the existing style: for one edge, for all mode
  assignments, `add_precedence` implies "every lane of the reader executes strictly after
  the cycle in which that lane of the producer is written".

## 12. What could not be determined

- **Whether the 16-round hint is accepted as complete and feasible**, and what the 193 s
  before the first solution consists of. No 16-round solver log was available; a run was
  not attempted because another session was rebuilding the extension in this directory.
  The log lines to look for are "The solution hint is ..." and the `#1` line's worker
  name.
- **The value of `horizon` at 16 rounds.** Inferred to be at least 12,356 from the code
  (`.max(n)`), not observed. The verbose line `SSA CP-SAT: ... horizon ...` shows it.
- **Whether CP-SAT detects a makespan for each cumulative in the real model**
  (section 7.3). It is logged only at `VLOG(2)`.
- **Whether the liveness hole has ever been hit on a real run.** The mechanism is in the
  code and reproduced on a replica; no Rust run was made to exhibit it. The
  `scheduler_output_error.json` present in the repository (dated today) records
  "ERROR: CP-SAT scheduling failed" with no schedule and does not settle it.
- **How much lane spreading real solutions use**, i.e. what K = 2 would lose. The
  `valu_shift_debug` block in the scheduler JSON has the counts; not analysed.
- **Effect of every proposal on the real model.** All measurements of alternatives are on
  the toy; the toy has no selects, constants, CSE groups, symmetry constraints or hints,
  and its instances are 76 to 608 ops.
- **`push_all_tasks_toward_start`, variable-demand intervals, per-window counters and the
  prefix/suffix rows** were not tested.
- **Exact constraint counts per family at 16 rounds.** The table in section 2 is derived
  from op counts and declarations, not read from a model dump.

## 13. Reproduction

```bash
# Toy experiments (about 12 minutes; -P is needed because scripts/bisect.py shadows stdlib bisect)
uv run python -P scripts/cpsat_encoding_experiments.py
uv run python -P scripts/cpsat_encoding_experiments.py --only F --time-limit 20

# Real-model log used in sections 1, 4.2, 7.2, 8.2 (uses the installed extension; no rebuild)
uv run generate_ssa_kernel.py --rounds 2 --batch-size 16 --cpsat-time-limit 60 \
    --no-validate --output-json /tmp/r2b16.json > /tmp/r2b16.log 2>&1
grep -n "solution hint\|orbits\|orbitope\|subsolvers\|^#Variables\|^#k" /tmp/r2b16.log
```

OR-Tools sources read: tag `v9.15`, files `ortools/sat/linear_relaxation.cc`,
`ortools/sat/integer_search.cc`, `ortools/sat/cp_model_search.cc`,
`ortools/sat/cp_model_solver.cc`, `examples/python/rcpsp_sat.py`; and
`sat_parameters.proto` as vendored in the `cp_sat` fork
(`~/.cargo/git/checkouts/cp_sat-*/165da6f/src/`). The fork exposes: Boolean and integer
variables, `add_or`, `add_and`, `add_implication`, `add_at_most_one`, `add_exactly_one`,
`add_xor`, `add_all_different`, linear constraints with `only_enforce_if`, `add_min_eq`,
`add_max_eq`, intervals (fixed, variable, optional) with linear start/size/end,
`add_cumulative` with linear demands and capacity, `add_no_overlap`, hints, and
`proto_mut()` for anything else (element, table, reservoir, decision strategies). There
is no `add_element` wrapper.
