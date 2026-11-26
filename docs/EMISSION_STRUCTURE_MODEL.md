# Emission structure model: let a solver choose the kernel's structure

A small model decides how every group of 8 lanes fetches its tree node in every round
(gather, or one of three ways to select among cached nodes). It is calibrated against the real
emitter and predicts the op counts of every plan exactly. Its best plan has a capacity floor of
989 cycles against 1,049 for today's best flags. With the shaped greedy scheduler a plan from the
model runs in **1,104 cycles (1,106 on the simulator)** against 1,153 / 1,155 for today's best
flags. Every kernel here is proved equal to its SSA program, passes validation and is correct on
the unmodified upstream tests; every model plan also passes their speed thresholds (`--strict`).
The floor ignores time, so it is a good guide only while the scheduler can approach it. Here
every schedule lands 90 to 120 cycles above its floor, and two plans with the same floor can
differ by 100 cycles (section 5). Cutting vector work (section 6.1) takes it to **1,099** on the
simulator with the output unchanged (branch bits kept live), and **1,083** with `--values-only`
(indices not stored; the unmodified upstream tests pass). Estimated with the model, `vload` at
the 1-based index and subtree packets (phases 2 and 3) are worth only about 35 cycles of floor
on this structure: once loads stop binding, the vector work binds near 975 (section 6).

A closed loop between this model and the shaped greedy (section 10, Benders-style feedback with
no-goods, learned caps, and time-aware rows whose form is proved but whose per-class table is
only observed) **found no plan better than `flow90.json`**.
Four other plans tie it at 1,104 / 1,106, and nothing goes below. No lower bound stage 1 can
compute ranks the near-optimal plans: over 26 loop plans the floor has Spearman rho +0.03 against
scheduled cycles, the floor with time-aware rows -0.41, and the exact static bound of each built
kernel -0.42. Only the capped floor ranks them (rho +0.64), and it is a heuristic, not a bound;
its cap was learned on the same builds it is scored on, so that rho is in-sample (section 10.4
lists the caveats).
So stage 1 cannot yet be trusted to pick among plans within about 20 cycles of each other. The
remaining spread comes from the greedy.

November 2026. Code: `scripts/emission_structure.py` (model, calibration, solve, measurement),
`solver/emission_plan.py` (plan format), `--emission-plan` (emitter). Skill:
`.cursor/skills/use-emission-structure/SKILL.md`. All numbers: 16 rounds, batch 256, height 10,
`--merge-hash-stages --index-form mirrored`, reproduced by the commands in section 9.

## 1. Two levels

| Level | Decides | Over | Tool |
|---|---|---|---|
| Structure (this model) | Which encoding each group uses for the tree access of each round | Op counts per engine; no time | CP-SAT over group counts, exact |
| Schedule (existing pipeline) | Cycle, engine and address of every op of the chosen emission | Dependencies, latency, slots, scratch | Greedy, then optionally CP-SAT |

The structure model's output is a plan file (`solver/emission_plan.py`), which
`generate_ssa_kernel.py --emission-plan PATH` emits and then schedules as usual.

## 2. Encodings

At depth `d` the node a lane needs is one of `2^d` nodes. Today's emitter has four ways to get
it, all plannable per round and per group:

| Encoding | Depths | What a group emits (relative to gathering, from the cost table, section 3) |
|---|---|---|
| `gather` | 1 to 10 | 1 vector address op (`(forest_p + L_d) - n`), 8 scalar loads, 1 XOR with the deferred hash constant |
| `bits` | 0 to 4 | `2^(d-1)` multiply_adds (leaf pairs), `2^(d-1) - 1` SelectChoices with a 2-op FMA form, bit extraction (2 per group at depth 3, 4 at depth 4) |
| `bits-flow` | 0 to 4 | As `bits`, but the leaf selects are SelectChoices with a 1-op FMA form, so all `2^d - 1` selects may go to flow |
| `mobius` | 0 to 4 | `2^d - 1` multiply_adds over Mobius coefficients, no flow |

The three bit-driven encodings come from `--cached-select` (PR #34, proofs R13 to R15).
`standard` (the default compare-and-select emission) is not plannable: each bit-driven encoding
needs fewer ops at every depth. Depth 0 is the shared root vector whatever the encoding, and depths
5 to 10 can only gather, so the choice is made in the 8 rounds at depths 1 to 4 (rounds 1 to 4
and 12 to 15).

Setup is paid once if any group uses it: the level's node loads and the node table XORed with
the hash constant (any bit-driven encoding at that level), the leaf-pair vectors (`bits`,
`bits-flow`), their `high` vectors (`bits-flow`), the Mobius coefficients (`mobius`), the level's
end-pointer constant (any `gather` at that depth) and the round's address base vector (any
`gather` in that round).

Mixing within a round is supported: `emit_round` calls each encoding's emitter once, over the
groups that use it, in the order of the first group using each. A plan that gives every group the
encoding the flags would give emits exactly the flags' SSA program
(`tests/test_emission_plan.py::test_plan_reproduces_flag_emission`), and without a plan nothing
changes (the 16-round default and `bits` builds are identical to before, op for op).

## 3. Counts from the emitter

An emission is summarised by ten counts (`emission_counts`, read from the scheduler JSON's SSA
ops after dead code elimination):

| Category | Ops | Engines it may use |
|---|---|---|
| `vec_fixed` | multiply_add, vbroadcast | VALU |
| `vec_shiftable` | other vector ops | VALU, or 8 scalar ALU slots |
| `alu` | scalar ALU ops | ALU |
| `load`, `store`, `flow` | fixed loads (`load`, `vload`), stores, flow ops other than selects | that engine |
| `select1` | SelectChoice whose FMA form is 1 multiply_add | 1 flow, or 1 VALU |
| `select2` | SelectChoice whose FMA form is SUB + multiply_add | 1 flow, or 1 fixed + 1 shiftable vector op |
| `const` / `const_alu` | ConstChoice without / with an ALU-derived form | load or flow add_imm (or 1 scalar ALU op) |

The scheduler JSON now records the FMA cost of each SelectChoice (`select_fma_vector_ops`) and
whether a ConstChoice has an ALU form (`const_alu_derivable`), so these are read, not inferred.

**Cost model.** Counts are linear in features of the plan (`plan_features`). The structure is
written by hand from reading the emitter. The coefficients are not:

| Feature | Value |
|---|---|
| `base` | 1 |
| `groups:d:e` | groups using bit-driven encoding `e` at depth `d`, over both visits (gathering groups are in `base`: per depth the counts sum to a constant) |
| `table:d`, `pairs:d`, `pairs_high:d`, `mobius:d` | 1 if the level-`d` setup of section 2 is needed |
| `gather_depth:d`, `gather_round:r` | 1 if some group gathers at depth `d` / in round `r` |
| `and:table:2+table:3`, ... (4 products) | the scalar constants 3 to 15 are derived from each other and feed the node addresses of levels 2, 3 and 4, so whether one survives dead code elimination is an OR over those levels; the products give the full multilinear form |

`calibrate` builds 144 design plans (each round and encoding alone, every subset of cached levels
with each encoding, pairs with Mobius, 40 random mixes) and solves for integer coefficients by
exact rational elimination. It raises if no exact fit exists. Without the 4 products there is
none (scalar ALU ops are not additive), which is how the coupling was found. `check` then builds
fresh random plans and compares every count:

| Shape | Design plans (exact) | Unseen random plans predicted exactly |
|---|---|---|
| 16 rounds, 32 groups | 144 | 30 of 30 |
| 16 rounds, 4 groups (the test) | 124 | 12 of 12, every run of `tests/test_emission_structure.py` |
| 16 rounds, 4 groups | 144 | 40 of 40 |

The cost table at 32 groups (`python -m scripts.emission_structure table`; empty = 0):

| Feature | vec_fixed | vec_shiftable | alu | load | select1 | select2 | const | const_alu |
|---|---|---|---|---|---|---|---|---|
| base | 2,486 | 5,056 | 104 | 3,620 | | | 47 | 6 |
| groups:1:bits / bits-flow / mobius | 1 / 0 / 1 | -2 | | -8 | 0 / 1 / 0 | | | |
| groups:2:bits / bits-flow / mobius | 2 / 0 / 3 | -2 | | -8 | 0 / 2 / 0 | 1 / 1 / 0 | | |
| groups:3:bits / bits-flow / mobius | 4 / 0 / 7 | 0 | | -8 | 0 / 4 / 0 | 3 / 3 / 0 | | |
| groups:4:bits / bits-flow / mobius | 8 / 0 / 15 | 2 | | -8 | 0 / 8 / 0 | 7 / 7 / 0 | | |
| table:1 / 2 / 3 / 4 | 1 | | 4 / 11 / 27 / 53 | 2 / 4 / 8 / 16 | | | | |
| pairs:1 / 2 / 3 / 4 | 1 / 3 / 7 / 15 | | 1 / 2 / 4 / 8 | | | | | |
| pairs_high:1 / 2 / 3 / 4 | | 1 / 2 / 4 / 8 | | | | | | |
| mobius:1 / 2 / 3 / 4 | 1 / 3 / 7 / 15 | | 1 / 4 / 12 / 32 | | | | | |
| gather_round:r (each of 8) | 1 | | 1 | | | | | |
| gather_depth:3 / 4 | | | | | | | 0 / 1 | 1 / 0 |
| and:table:2+3, 2+4, 3+4, 2+3+4 | | | -3, -3, -5, +3 | | | | | |

Stores (64) and fixed flow ops (0) are constant. Reading it: a group that selects level 4 with
`bits-flow` instead of gathering saves 8 loads and costs 8 + 7 selects and 2 vector ops (bit
extraction 4, minus the gather's 2); the level-4 table costs 16 loads and 53 + 8 scalar ops once.

### 3.1 The same table by instantiation (no probe builds)

`python -m scripts.emission_structure derive` gets the table from one emission instead of 144
fits (`scheduler_rs/src/kernel_builder/transform/`, binding `derive_tree_access_cost_table`).
The canonical program (root from the shared vector, every other round gathering) is emitted
with every bit-driven alternative of every choice site built next to its gather by the
emitter's own per-group code, on operands a structural matcher binds (rule `tree-access`;
region tags only propose where to look). Then every op gets the condition under which dead
code elimination keeps it, an OR of (round, group, encoding) choices, which reduces to the
features above: shared tables to `table:d` / `pairs:d` / `pairs_high:d` / `mobius:d`, gathers'
base vectors and level-end constants to `gather_round:r` / `gather_depth:d`, and ORs across
levels, by inclusion-exclusion, to the `and:table:*` products (the scalar constants 3 to 15
that address the cached nodes). Per-site differences to gathering give `groups:d:e` and are
checked to be the same at every site. Whether a constant is `const` or `const_alu` depends
on the constants created before it, so on emission order: every constant is checked to keep
its category whatever the lazily created level-end constants `L_d` (and its own position, for
those). Anything that does not reduce to the features is reported, never fitted. At 16 rounds the derived table equals the calibrated one entry for
entry with 4 and with 32 groups (`tests/test_emission_structure.py`, and `derive` against
`calibrate`); the Rust tests also compare its predictions with emitted programs directly.
Contexts it does not cover yet: the zero-based index (bits are matched only on the mirrored
index chain) and `--values-only` (dead index updates are per group, the `tail:*` features).

The apply pass (`transform::apply_tree_access_plan`) takes a plan and rewrites the same
instantiated program into the plan's program: readers of each site's value are rewired to the
chosen candidate, then dead code elimination. Three emitter-order effects are reproduced
explicitly so the result is the `--emission-plan` program op for op (up to VReg and op
numbering; Rust tests over 74 plans at 4 and 32 groups, and the greedy schedule is the same): a mixed round's
encodings in order of their first group; the level-end constant `L_d`, which the emitter
creates at its first live user: the first round that gathers at depth `d`, or, when no round
gathers there, the final mirrored index conversion if the last round leaves the indices at
depth `d` (e.g. 15 rounds at height 10 and `L_4`; tested at that shape), with its ALU-derived
alternatives recomputed there; and the broadcast of a level's first table entry, shared by leaf pairs and
Mobius coefficients, which the emitter creates with the coefficients when no pair is used.

## 4. The objective: the capacity floor with choices resolved

`choice_floor(counts)` is the smallest integer `T` such that some resolution of every choice fits
in `T` cycles:

```
loads + consts left on load                  <= 2T     stores <= 2T
flow + selects on flow + consts on flow      <= T
vector ops - s                               <= 6T     (s shiftable vector ops run as 8 scalar ops)
scalar ops + consts on the ALU + 8s          <= 12T
```

Taking 2-op selects to flow first is never worse, and only the consts that do not fit on the load
engine leave it, so the search is exact; it is checked against brute force over every resolution
(`test_choice_floor_matches_brute_force`). The CP-SAT model in `solve_plan` states the same
constraints over the plan variables (groups per round and encoding, booleans for the setup
features and products) and minimises `T`. The model is small (8 rounds x 4 encodings) and solves
to proven optimality in about a second.

Differences from `scripts/schedule_report.py`'s floor: that one keeps each choice as the scheduler
resolved it. The greedy puts every ConstChoice it can on load (2,124 loads for `bits` against the
2,098 fixed), so its floor is 1,062 where the choice floor is 1,049. Both are reported.

**What the model ignores.** Dependencies and latency (a select tree and the hash chain are
serial), the shape of the schedule (no load can run before a group reaches a gathered level, and
the last rounds are the schedule's tail), scratch (the bit-driven encodings keep tables live) and
the scheduler's own choices (the greedy does not balance selects between flow and the vector ALU).

**Why the floor is still a useful proxy.** Near saturation a schedule cannot be shorter than the
floor, and a good one is not much longer: the public 880-cycle kernel lands 7 above its floor
(docs/BOUND_GAP_ANALYSIS.md section 5). Here the gap is larger (section 5), so two tie-breaks
and an optional cap steer the choice among plans with similar floors:

| Rule | Why |
|---|---|
| Fewest encodings mixed into one round | Simpler plans. Never changed a result here |
| Select work at the earliest visit of a level (`lateness`: rounds weighted by the groups that select instead of gathering) | Round 15 is the schedule's tail: nothing after it can overlap its select trees. Measured: the same floor with level 4 cached in round 15 instead of round 4 costs 30 to 100 greedy cycles |
| `--max-util flow=F` (optional): use at most a fraction F of an engine's capacity | The floor-optimal plan saturates loads and flow together, and the greedy cannot pack both. The cap minimises `T` under the cap and reports the plan's own floor too |

## 5. Results

Shaped greedy (`--greedy-barriers none --greedy-priority wavefront --greedy-split-shift
--pressure-inflation-threshold 0.95`). "Predicted floor" is `choice_floor` of the model's counts,
"emitted floor" the same of the built kernel's counts, "scheduled floor" `schedule_report`'s floor
with the scheduler's own resolution (binding engine in brackets). Every row: counts predicted
exactly, PROVED (dataflow replay), PASSED (values and indices), and correct on the unmodified
upstream tests (`./scripts/check_upstream.sh`). Every row also passes the upstream speed
thresholds (`--strict`) except `bits-flow` with level 4 on both visits, at 1,700 cycles.

| Structure | Predicted floor | Emitted floor | Scheduled floor | Cycles (sched / sim) | Peak live |
|---|---|---|---|---|---|
| flags `--cached-select bits` (today's best) | 1,049 | 1,049 | 1,062 (load) | 1,153 / 1,155 | 771 |
| flags `bits-flow` | 1,049 | 1,049 | 1,067 (load) | 1,160 / 1,162 | 771 |
| flags `mobius` | 1,071 | 1,071 | 1,071 (ALUs) | 1,154 / 1,156 | 771 |
| flags `bits`, level 4 after the wrap | 1,082 | 1,082 | 1,082 (ALUs) | 1,138 / 1,140 | 915 |
| flags `bits-flow`, level 4 after the wrap | 1,013 | 1,013 | 1,204 (flow) | 1,257 / 1,259 | 1,035 |
| flags `bits`, level 4 on both visits | 1,124 | 1,124 | 1,124 (ALUs) | 1,171 / 1,173 | 930 |
| flags `bits-flow`, level 4 on both visits | 1,077 | 1,077 | 1,685 (flow) | 1,698 / 1,700 | 1,043 |
| model, floor-optimal (`docs/emission_plans/floor_optimal.json`) | **989** | 989 | 1,007 (load) | 1,125 / 1,127 | 1,027 |
| model, `--max-util flow=0.9` (`docs/emission_plans/flow90.json`) | 997 | 997 | 1,014 (load) | **1,104 / 1,106** | 1,049 |

The two model plans:

| Plan | Rounds 0 to 3 | Round 4 | Rounds 11 to 14 | Round 15 |
|---|---|---|---|---|
| floor-optimal | bits, then bits-flow x3 | 15 gather + 17 bits-flow | bits, then bits-flow x3 | gather |
| flow 0.9 | bits x2, bits-flow x2 | 17 gather + 15 bits-flow | bits x2, bits-flow x2 | gather |

Both cache level 4 in only some groups of one visit, the per-group mix that design C1 of
docs/BOUND_GAP_ANALYSIS.md section 5 predicted.

Sweep of the flow cap (`sweep`, same scheduler flags; all rows PROVED and PASSED, the
tie-break rows also upstream `--strict`). Simulator cycles:

| Cap | none | flow 0.95 | 0.9 | 0.85 | 0.8 | 0.7 | 0.6 | 0.5 | load 0.97 | load 0.95 + flow 0.8 |
|---|---|---|---|---|---|---|---|---|---|---|
| Plan floor | 989 | 993 | 997 | 1,001 | 1,005 | 1,013 | 1,021 | 1,029 | 996 | 1,018 |
| With the lateness tie-break | 1,127 | 1,118 | **1,106** | 1,119 | 1,122 | 1,125 | 1,130 | | | 1,108 |
| Without it (`--no-lateness`) | 1,127 | | 1,106 | | 1,120 | 1,125 | 1,158 | 1,155 | 1,212 | 1,108 |

Without the tie-break the solver is free to put the level-4 selects in round 15 instead of
round 4 (same floor): the flow 0.6, 0.5 and load 0.97 plans do so and lose 30 to 100 cycles.

Floor against cycles for every build above (`scatter`; dashed: cycles = floor):

![Simulator cycles against the emitted floor](emission_plans/floor_vs_cycles.svg)

The flag configurations spread along the floor axis with cycles barely moving (1,140 to 1,175,
outliers where the greedy puts every select on flow); the model's plans sit 60 floor cycles
lower and 30 to 50 cycles faster, about 100 to 140 above their floors.

What the numbers say:

- **The model is exact on counts and floors**: predicted = emitted in every build (several hundred
  plans across calibration, checks and these tables).
- **It finds better structures than the flags**: the floor falls from 1,049 to 989 (-60), and
  cycles from 1,153 to 1,104 (-49). The flag combination `bits` + level 4 after the wrap, which
  docs/EMISSION_ABLATION.md measured only with the default greedy (2,357 cycles), already gives
  1,138 here. The model's plan is 34 better still.
- **The floor ranks plans only roughly once loads bind**: plans with floors 989 to 1,021 all land
  at 1,104 to 1,128, and the order is not the floor's order. Before the lateness tie-break, plans
  with the same floors put the level-4 selects in round 15 and ran at 1,153 to 1,212.
- **The gap is the scheduler's**: 91 above the floor for `bits`, 107 for the best plan. The greedy
  resolution alone accounts for 13 to 17 (consts on load: "scheduled floor" against "emitted
  floor").

**Where the gap goes** (`scripts.schedule_report` on the 1,104-cycle schedule of `flow90.json`;
see docs/GREEDY_SHAPE_ABLATION.md section 11: shifting toward the floor's split does not shorten
the schedule (PR #76: the shift count follows the schedule's length), and retuning the window and
stagger plus `--greedy-load-priority final-last` takes this plan to 1,039 cycles with the greedy):
the vector ALU is 100% busy from cycle 111 to about 994 and the loads nearly so, but the greedy
ran only 1,039 vector ops as 8 scalar ops, while the floor's balance needs about 1,465. The scalar
ALU is 64% used. The vector ALU's own term is then 6,387 / 6 = 1,065, only 39 below the schedule.
So most of the gap is the greedy's VALU-to-ALU shifting (roughly 50 to 70 cycles), and the rest is
ramp-up and tail (the first and last tenth run at 72% to 93% on the vector ALU, 47% to 75% on loads).
This is recorded to keep the gap measured; scheduler work is out of scope here.

CP-SAT on the best plan: section 5.1.

### 5.1 CP-SAT

One 16-round run on `flow90.json` with the shaped flags and `--cpsat-deterministic-time 300`
(wall-clock backstop 7,200 s, on a shared and heavily loaded machine): **1,132 schedule / 1,134
simulator cycles**, PROVED and PASSED, best bound 1,000. It stopped on the wall-clock backstop
after 172 of the 300 deterministic-time units, so it is not reproducible, and it did not improve
its warm start. That warm start is worse than the greedy's own 1,104 because the greedy split
shift is not used when CP-SAT follows the greedy. CP-SAT at 3 rounds through the same emission
paths is covered by docs/EMISSION_ABLATION.md section 3.4.

## 6. Phases 2 and 3: what the model says they are worth

The model takes new encodings as rows of the cost table once the emitter has them. Before
building one, `--what-if` gives an uncalibrated estimate with hand-adjusted coefficients:

| What-if (not calibrated) | Floor-optimal plan | Binds |
|---|---|---|
| Today's encodings (calibrated) | 989 | loads |
| `gather-without-address-op`: every gather loses its vector address op, as `vload` at the 1-based index would (phase 2) | 971 | loads and ALUs |
| `packets-5`: levels 5+6 from one per-lane `vload` of a 3-word packet (phase 3, one pair) | 981 | ALUs |
| `packets-5-7`: levels 5+6 and 7+8 | 996 | ALUs |
| `packets-5-7-9`: levels 5 to 10 | 1,054 | ALUs |
| phase 2 + `packets-5` (the packet estimate removes the address ops again, so about 8 too low) | 947 | ALUs |

The packet estimate per group and level pair (`what_if_packets`): 8 loads instead of 16; one
multiply_add for the packet address instead of 2 address ops and 2 constant XORs; 8 scalar copies
of the node word into the group vector and 16 scalar ops for the child (`left + bit * diff`).
Setup per pair at level L: `3 * 2^L / 8` vloads and vector XORs, `2^L` scalar subtracts, and
`3 * 2^L` scalar stores with their address adds, on the otherwise idle store engine. The setup
grows as `2^L`, which is why deeper pairs make it worse.

**Why packets pay little here.** As soon as loads stop binding, the ALUs do. The best plan's
7,452 vector-op equivalents are 14.55 per group-round:

| Phase | Vector-op equivalents | Per group-round |
|---|---|---|
| Hash | 5,120 | 10.00 |
| Branch bit and index update | 896 | 1.75 |
| Tree access (addresses, constant XOR, selects not on flow, bit extraction) | 743 | 1.45 |
| Node XOR | 512 | 1.00 |
| Setup, input and output | 181 | 0.35 |

That alone is a floor of about 975 whatever the load count. The public 880-cycle kernel does
all of its non-hash work in 2.58 per group-round (docs/BOUND_GAP_ANALYSIS.md section 3); here it
is 4.55. That difference (about 1,000 equivalents, 135 cycles) is the larger part of its
advantage, so a packet encoding pays only together with cheaper index, address and XOR work.
Phase 2 and phase 3 together are worth about 35 cycles of floor on this structure, less than the
scheduler's 107-cycle gap on the current best plan.

Leads from the leaderboard research, to verify:

| Lead | Mechanism | In this model |
|---|---|---|
| 899 cycles | 3-word two-level packets, 6 loads per lane, no address add | The `packets-*` what-ifs above; with this structure's vector work it does not reach the load floor |
| 913 cycles | Per-lane `jump_indirect` dispatch reading depth-4 nodes from scratch: rounds 4 and 15 need no gathers | Needs control flow in a straight-line scheduled kernel, which the pipeline does not have |
| 982 cycles | Level-4 caching for a searched subset of groups | What this model does (section 5) |
| Estimate | 899-style packets for levels 4 to 7: about -480 loads, load floor about 822 | The load floor falls, but the ALUs bind near 975 first |

### 6.1 Cutting vector work (the direction taken)

The user chose to cut vector work rather than build packets or work on the scheduler. Four
candidates, cheapest to verify first:

| Candidate | Result | Evidence |
|---|---|---|
| Address-free gather by `vload` at a pointer-form index (phase 2) | **Impossible at the same op count.** A `vload` at `p = idx + F` leaves `tree[idx]` in a fixed word for any `F = forest_p - w`, but the true branch bit is `1 - (lazy & 1)`, so the update `p' = 2p + bit + 1 - F` needs one op giving `(2 - F) - (v & 1)`. No single ISA op with a constant does, for any `F` from 1 to 8 and either operand order. The update stays 3 ops, which costs the address op back in every round | Z3, `scripts/analyze_bound_gap.py --z3` ("NO SUCH OP") |
| Fold the node XOR into hash stage 0 | **No fold exists.** It would need `(x ^ y) * 4097 + C0` to be affine in the value `x` and the node `y`; no constants make it so | Z3, same script ("NO SUCH FOLD") |
| Skip index updates nothing reads | Only with `--values-only`: otherwise the final index store reads the whole chain. With it, a group's updates after its last index read are dead, and which ones depends on the group's own gather/select pattern over rounds 12 to 15, so the cost model gains `tail:<pattern>` features (exact: 144 design plans and every unseen plan) | `tests/test_emission_structure.py::test_values_only_tail_patterns_make_the_fit_exact` |
| Branch bits kept live instead of extracted (`--select-bit-source path`) | No bit-extraction ops at depths 3 and 4. Fits the scratch budget under the wavefront for the solved plans; some calibration plans do not build and are skipped | Calibration notes; the rows below |

`--values-only` changes the output (indices not stored) and is allowed in recorded results when
the unmodified upstream tests pass; every values-only number below says so.

Solved and measured (shaped greedy; every row counts predicted exactly, PROVED, PASSED, and
`./scripts/check_upstream.sh --strict` passes; values-only rows are validated on values only):

| Context | Cap | Plan floor | Cycles (sched / sim) | Level-4 selects |
|---|---|---|---|---|
| merge + mirrored (section 5) | flow 0.9 | 997 | 1,104 / 1,106 | round 4, 15 groups |
| + `--select-bit-source path` | none | 973 | 1,107 / 1,109 | round 4, 21 groups |
| + `--select-bit-source path` | **flow 0.9** | 980 | **1,097 / 1,099** | round 4, 20 groups |
| + `--select-bit-source path` | flow 0.8 | 989 | 1,103 / 1,105 | round 4, 17 groups |
| + `--values-only` (values only) | none | 980 | 1,200 / 1,202 | round 15, 20 groups |
| + `--values-only` (values only) | flow 0.9 | 985 | 1,196 / 1,198 | round 15, 18 groups |
| + `--values-only` (values only) | flow 0.8 | 995 | 1,137 / 1,139 | round 15, 16 groups |
| + both (values only) | none | 957 | 1,157 / 1,159 | round 15, 25 groups |
| + both (values only) | flow 0.9 | 965 | 1,107 / 1,109 | round 15, 23 groups |
| + both (values only) | flow 0.8 | 973 | 1,112 / 1,114 | round 15, 21 groups |
| + `--values-only`, `--fix 15=gather` (values only) | none | 981 | 1,120 / 1,122 | round 4, 19 groups |
| + `--values-only`, `--fix 15=gather` (values only) | flow 0.9 | 989 | 1,095 / 1,097 | round 4, 17 groups |
| + `--values-only`, `--fix 15=gather` (values only) | flow 0.8 | 997 | 1,109 / 1,111 | round 4, 15 groups |
| + both, `--fix 15=gather` (values only) | none | 964 | 1,086 / 1,088 | round 4, 24 groups |
| + both, `--fix 15=gather` (values only) | **flow 0.9** | 969 | **1,081 / 1,083** | round 4, 22 groups |
| + both, `--fix 15=gather` (values only) | flow 0.8 | 977 | 1,084 / 1,086 | round 4, 20 groups |

What this says:

- Path bits are worth 7 cycles measured (1,106 to 1,099) and 9 to 16 of floor, with the output
  unchanged.
- Values-only lowers the floor (989 to 980 alone, 957 with path bits) but the measured cycles get
  worse, because it makes level-4 selects in round 15 cheaper in ops: a group that selects there
  needs no index update in rounds 14 and 15. The floor then prefers round 15, the schedule's tail,
  and the lateness tie-break never applies (the floors differ). This is the model's known limit
  (section 7) showing up in the opposite direction: an op-count saving that costs time.
- Keeping the last round free of select work (`--fix 15=gather`, a constraint on the plan rather
  than a tie-break) recovers it: values-only with path bits reaches **1,081 / 1,083** (floor 969),
  values-only alone 1,095 / 1,097. Under every cap, the fixed plans beat the unfixed ones by 15 to
  100 cycles.
- Best plans, both checked in: `docs/emission_plans/path_flow90.json` (output unchanged, 1,099 on
  the simulator) and `docs/emission_plans/values_only_path_fix15_flow90.json` (values only, 1,083).


## 7. Limits

| Limit | Consequence | Mitigation here |
|---|---|---|
| No time: the floor counts slots, not when work can run | Plans with equal floors schedule up to 100 cycles apart; the floor ranks plans only roughly once loads bind | Measure every candidate; `sweep` over caps |
| Symmetric between visits of a level | Cannot tell round 4 from round 15, though round 15 is the tail | Lateness tie-break (select work at the earliest visit), a heuristic justified by the measurements above, not part of the floor |
| No scheduler model | The greedy keeps consts on load and does not balance selects between flow and the vector ALU | `--max-util` headroom on flow; report the scheduled floor beside the emitted one |
| Cost model structure is hand-written | A new encoding with new shared resources needs new features | `calibrate` fails loudly on any inexact fit; `check` on unseen plans |
| Group assignment within a round | Contiguous runs in a fixed order | Not explored |

## 8. What was not verified

| Item | State |
|---|---|
| CP-SAT at 16 rounds on the model's plans | See section 5.1 |
| Which groups get which encoding within a round | The plan gives each encoding a contiguous run of groups in a fixed order; other assignments (e.g. interleaved with the wavefront's stagger) were not tried |
| `--values-only`, the zero-based index, `--select-bit-source path` | Supported as contexts (separate calibration); not measured here. The zero-based form is covered by a PROVED and PASSED test |
| Calibration at other shapes than 16 rounds with 4 and 32 groups | Not run |
| The what-if of section 6 | An estimate by construction; no emitter exists for it yet |

## 9. Reproduction

```bash
./scripts/build_scheduler_rs.sh --release
M="uv run python -m scripts.emission_structure"
$M calibrate                                   # 144 builds, about 2 minutes; exact or an error
$M check --plans 30 --seed 99                  # unseen random plans: predicted == emitted
$M table                                       # section 3
$M predict                                     # floors of the flag configurations
$M solve --out docs/emission_plans/floor_optimal.json
$M solve --max-util flow=0.9 --out docs/emission_plans/flow90.json
$M measure --shaped --upstream --plan bits --plan bits-flow --plan mobius --plan bits+depth4-after-wrap \
  --plan bits-flow+depth4-after-wrap --plan bits+depth4 --plan bits-flow+depth4 \
  --plan docs/emission_plans/floor_optimal.json \
  --plan docs/emission_plans/flow90.json                                       # section 5, first table
$M sweep --shaped --upstream --max-util none --max-util flow=0.95 --max-util flow=0.9 --max-util flow=0.85 \
  --max-util flow=0.8 --max-util flow=0.7 --max-util flow=0.6 --max-util load=0.95,flow=0.8 --json sweep.json
$M sweep --shaped --no-lateness --max-util none --max-util flow=0.9 --max-util flow=0.8 --max-util flow=0.7 \
  --max-util flow=0.6 --max-util flow=0.5 --max-util load=0.97 --max-util load=0.95,flow=0.8 --json nolate.json
$M measure --shaped --json flags.json --plan bits --plan bits-flow --plan mobius --plan bits+depth4-after-wrap \
  --plan bits-flow+depth4-after-wrap --plan bits+depth4 --plan bits-flow+depth4
$M scatter --input "flag configurations=flags.json" --input "model plans=sweep.json" \
  --input "model plans, no lateness tie-break=nolate.json" --out docs/emission_plans/floor_vs_cycles.svg
$M sweep --shaped --upstream --select-bit-source path --max-util none --max-util flow=0.9 \
  --max-util flow=0.8                                                         # section 6.1
$M sweep --shaped --upstream --values-only --select-bit-source path --fix 15=gather --max-util none \
  --max-util flow=0.9 --max-util flow=0.8                                     # (also without --fix,
                                                                              # and without path)
uv run --extra verify python scripts/analyze_bound_gap.py --z3 --json sched.json  # the two Z3 negatives
$M solve --what-if gather-without-address-op                                 # section 6, also packets-5,
                                                                              # packets-5-7, packets-5-7-9

# One plan by hand
uv run generate_ssa_kernel.py --rounds 16 --use-greedy --validate --merge-hash-stages --index-form mirrored \
  --emission-plan docs/emission_plans/flow90.json \
  --greedy-barriers none --greedy-priority wavefront --greedy-split-shift --pressure-inflation-threshold 0.95
```

## 10. Closing the loop: stage 1 driven by measured schedules

**Conclusion (negative).** Feeding greedy measurements back into the structure model did not
beat `docs/emission_plans/flow90.json` (1,104 / 1,106, development greedy measurement). The best
loop found **ties** it: the learned-cap loop reaches 1,104 / 1,106 in 3 builds with no manual
sweep, and five different plans with floors 994 to 997 all land at exactly 1,104 / 1,106.
Nothing in 32 loop-state builds went below. The time-aware rows are proved in form (section
10.3: a Z3 lemma for the engine coupling, a hand argument for the windows, a randomized check of
the whole), but over
the plans near the optimum no lower bound stage 1 can compute ranks them, and the rows rank them
worse than the bare floor. So stage 1 should not be trusted to choose among plans within about
20 cycles of each other. Its floor and the learned caps get within that distance of the best
plan, and the last cycles depend on how the greedy schedules each plan.

All numbers in this section: 16 rounds, 32 groups, `--merge-hash-stages --index-form mirrored`,
the shaped greedy (`--greedy-barriers none --greedy-priority wavefront --greedy-split-shift
--pressure-inflation-threshold 0.95`) for every build. They are development measurements with
the greedy, not recorded benchmark results. Every build: counts predicted exactly, PROVED
(dataflow replay) and PASSED (`--validate`). The flow90 kernel rebuilt here also passes
`./scripts/check_upstream.sh --strict` (1,106 cycles). Budget: 34 sixteen-round builds at 32
groups (2 reference builds + 32 in the loop states), one at a time, greedy only.

### 10.1 The loop

`scripts/emission_loop.py` (`python -m scripts.emission_structure loop run|report`). Each
iteration: (1) stage 1 solves for a plan (`solve_plan`, about 2 s); (2) one shaped-greedy build,
proved and validated; (3) the scheduler JSON is read: cycles, per-engine utilisation in 10
buckets (start-up, steady state, tail), the scheduler's own floor and binding engine, the exact
static head-body-tail bound, and per op class the work, head and tail; (4) feedback; (5) stop on
the iteration budget, `--patience` iterations without improvement, or an infeasible stage 1.
Builds are cached in a state file, so a plan is never built twice and modes can share seeds.
The state records the SHA-256 of the scheduler extension that made its builds and `run` refuses a
state from another build. Stage-1 solves take a deterministic-time budget (`--deterministic-time`,
default 20; at 16 rounds and 32 groups each solve proves optimality in under 0.2 units, so the
budget never binds and the plan does not depend on machine load); the trace records, per
stage-1 call, every CP-SAT solve (main, windows floor, tie-break) with its status, wall and
deterministic time and whether it stopped on the wall clock.

The traces in `docs/emission_plans/loop_report.json` predate three of these changes; none
changes them. They were run with the wall-clock limit only (every solve in them reports
`optimal`, so they ended on optimality, not on the clock). Rows mode cut with `M < 1,104`, which
never closed a run and is far above the M of every plan it picked (995, 996). And the no-goods
of the seeds `bits-flow+depth4` and `mobius+depth4`, which select the root with bits-flow /
mobius and so are not the arrangement stage 1 emits (it keeps the root on bits), cut their model
points unbuilt. Those points have floors 1,077 and 1,219, above every objective any recorded
iteration picked (at most 1,049), so the cuts never changed a pick.

| Feedback | What it adds to stage 1 | Bound or heuristic |
|---|---|---|
| (a) `nogood` | A no-good cut per built model point (`plan_model_key`: the group counts per round and encoding, plus tail patterns with `--values-only`) and `floor < best cycles`. A point is cut only once the arrangement stage 1 emits for it (`canonical_rows`) is built; a seed built in another arrangement leaves the point open | Sound: the floor is a lower bound on every schedule. Weak: the incumbent never prunes (floors 989 to 1,000 against 1,104) |
| (b) `caps` | Learned `max_util` per engine. From the best plan, cap an engine the plan uses at >= 90% at its use minus a step (0.04, halved when nothing is left); keep it if cycles improve. Rule `saturated`: an engine the schedule runs at >= 97% in its steady state. Rule `underfilled`: an engine the schedule fills less than the plan needs, largest shortfall first | Heuristic. It generalises `--max-util flow=0.9` |
| (c) `rows` | Objective M >= T and, per window (t, q), M >= t + q + `window_floor`(work with head >= t and tail >= q), as linear rows over the plan variables; no-goods; `floor < best cycles` (default `--rows-cut sound`) | Row form proved; the per-class table it is fed is observed (section 10.3), so M is not a bound on an unbuilt plan and `M < best cycles` would be a heuristic cut. It is used only with `--rows-cut heuristic`, as `M < best + e` (e the largest overestimate seen so far), and its closing is reported as the heuristic cut closing, never as exhaustion |
| (a) under (b)'s cap | No-good enumeration of plans in capped-floor order (`--nogood-caps flow=0.92`) | Heuristic (a capped floor is not a bound, so no incumbent cut) |

The window rows need work per op class as a function of the plan. SSA ops carry
`source_info` (round, group, phase). The hash, index update and node XOR have a round and no
group, and are plan-independent per round ("round classes"). Tree-access ops have a group and
are charged per group using encoding e in round r ("group classes"). The coefficients in stage 1
are the class counts on the plan's n[r, e], with every class whose minimum head is >= t and
minimum tail is >= q. Windows are all pairs of observed head and tail values (5,028 windows at
32 groups; the solve still takes about 2 s). A window is kept only if plan-independent work
lies in it, so M >= t + q holds for every plan.

### 10.2 Iteration trace

Seeds (built once, shared by every mode):

| Seed | Floor | Window rows (final table) | Cycles (sched / sim) |
|---|---|---|---|
| `gather` everywhere | 1,810 | 1,823 | 1,850 / 1,852 |
| flags `bits`, level 4 on both visits | 1,124 | 1,109 | 1,171 / 1,173 |
| flags `bits-flow`, level 4 on both visits | 1,077 | 1,063 | 1,698 / 1,700 |
| flags `mobius`, level 4 on both visits | 1,219 | 1,203 | 1,263 / 1,265 |
| `floor_optimal.json` | 989 | 1,021 | 1,125 / 1,127 |
| `flow90.json` | 997 | 1,029 | **1,104 / 1,106** |

(a) No-goods, floor objective. It walks through floor-989 neighbours of the floor-optimal plan
and stops on patience:

| Iter | Plan (choice rounds that differ from floor-optimal) | Floor | Cycles |
|---|---|---|---|
| 0 | level 4 selected in round 15 (x17), gathered in round 4 | 989 | 1,210 / 1,212 |
| 1 | round 4: one group bits instead of bits-flow | 989 | 1,125 / 1,127 |
| 2 | round 12: one group bits | 989 | 1,125 / 1,127 |
| 3 | round 4: two groups bits | 989 | 1,127 / 1,129 |
| 4 | round 1: one group bits | 989 | 1,125 / 1,127 |

(b) Learned caps, rule `saturated`. It caps VALU and load, the engines the schedule runs full,
and never flow (steady flow 79% to 90%):

| Iter | Caps solved under | Plan floor / capped T | Cycles | Feedback applied |
|---|---|---|---|---|
| 0 | none | 989 / 989 | 1,125 / 1,127 | cap valu 0.96 |
| 1 | valu 0.96 | 1,017 / 1,017 | 1,121 / 1,123 (best) | cap load 0.96 |
| 2 | valu 0.96, load 0.96 | 991 / 1,022 | 1,128 / 1,130 | revert; cap valu 0.92 |
| 3 | valu 0.92 | 1,049 / 1,049 | 1,160 / 1,162 | revert; cap load 0.98 |
| 4 | valu 0.96, load 0.98 | 997 / 1,018 | 1,121 / 1,123 | cap valu 0.94 |
| 5 | valu 0.94 | 1,033 / 1,033 | 1,135 / 1,137 | cap load 0.99 |
| 6 | valu 0.96, load 0.99 | 1,005 / 1,018 | 1,120 / 1,122 (best) | cap load 0.98 |
| 7 | valu 0.96, load 0.98 | (plan of iter 4, not rebuilt) | 1,121 / 1,123 | cap valu 0.95 |
| 8 | valu 0.95, load 0.99 | 1,013 / 1,026 | **1,118 / 1,120** (best) | cap load 0.98 |
| 9 | valu 0.95, load 0.98 | (plan of iter 6) | 1,120 / 1,122 | budget |

(b) Learned caps, rule `underfilled`. The plan needs flow at 100% of its floor, but the
schedule fills it at 84% to 90%, so the rule caps flow:

| Iter | Caps solved under | Plan | Floor / capped T | Cycles |
|---|---|---|---|---|
| 0 | none | floor-optimal | 989 / 989 | 1,125 / 1,127 |
| 1 | flow 0.96 | r1, r11 bits; r4 gather x16 + bits-flow x16 | 993 / 993 | 1,116 / 1,118 |
| 2 | flow 0.92 | r1, r12 **mobius**; r4 gather x16 + bits-flow x16 | 994 / 996 | **1,104 / 1,106** |
| 3 | flow 0.88 | as 2, r4 gather x17 + bits-flow x15 | 997 / 999 | 1,104 / 1,106 |
| 4, 5 | flow 0.90, 0.91 | `flow90.json` itself (not rebuilt) | 997 / 997 | 1,104 / 1,106 |

(a) under the learned cap `flow=0.92`, next plan by capped floor, no-goods:

| Iter | Plan | Floor | Cycles |
|---|---|---|---|
| 0 | r0-1 bits; r2-3 bits-flow; r4 gather x16 + bits-flow x16; r11-12 bits; r13-14 bits-flow | 994 | 1,104 / 1,106 |
| 1 | r1-3 bits-flow; r4 x16 / x16; r12 bits-flow, r13 bits, r14 bits-flow | 994 | 1,111 / 1,113 |
| 2 | r1-3 bits-flow; r4 x16 / x16; r12-14 bits-flow | 993 | 1,125 / 1,127 |
| 3 | r1 bits-flow, r2 bits, r3 bits-flow; r4 x16 / x16; r12-14 bits-flow | 994 | 1,111 / 1,113 |
| 4 | r1-3 bits-flow; r4 x16 / x16; r12 bits, r13-14 bits-flow | 993 | 1,112 / 1,114 |
| 5 | r0-1 bits; r2-3 bits-flow; r4 x16 / x16; r12 mobius; r13-14 bits-flow | 995 | 1,104 / 1,106 |

(c) Window rows, no-goods, `M < 1,104`. Stage 1 now prefers gathering 2 or 3 groups in round 3
and 8 in round 4: loads can then start about 10 cycles earlier, which lowers the load window
from about 1,021 to 995. The schedules are worse:

| Iter | Plan | Floor | M (rows) | Exact static bound | Cycles |
|---|---|---|---|---|---|
| 0 | r1-2 bits-flow; r3 gather x2 + bits-flow x30; r4 gather x8 + bits-flow x24; r12-14 bits-flow | 996 | 996 | 993 | 1,150 / 1,152 |
| 1 | as 0 with r3 gather x3, r12 bits | 995 | 995 | 995 | 1,127 / 1,129 |
| 2 | as 1 with r12 bits-flow | 995 | 995 | 994 | 1,150 / 1,152 |
| 3 | as 2 with r1 bits | 995 | 995 | 995 | 1,130 / 1,132 |
| 4 | as 2 with r1 mobius | 995 | 995 | 995 | 1,130 / 1,132 |

Plans that tie `flow90.json` at 1,104 / 1,106 (all PROVED and PASSED): the `underfilled` loop's
iterations 2 and 3, and the capped no-good iterations 0 and 5. None is checked in: none beats
flow90.

### 10.3 Which rows are bounds

| Claim | Status |
|---|---|
| Row form: for any op set, makespan >= t + q + `window_floor`(ops with head >= t and tail >= q), with heads and tails as in `scripts/static_bounds.py` (minimum latency over encodings). Selects are charged one flow slot or one fixed VALU slot, shiftable vector ops one VALU slot or 8 ALU slots, consts and eliminable broadcasts nothing | Three parts (`verification/window_rows_proof.py`, registered in `verification/verify.py`). (1) Every slot of a window op lies in [t, M - 1 - q]: **argued by hand** (induction along edges of latency >= 1 out of every slot, for every encoding: vselect, two-op FMA select, one-op delta select, shifted lanes). (2) WR3, the engine coupling: any per-engine split of the window's ops that fits in S cycles passes `_window_fits(S)`, which is monotone in S: **proved by Z3 for all counts and spans** on the real machine and the six scaled machines, with the transcription checked against the Python function. (3) WR1, the composition: a **randomized check**, not a proof, against the reference model of `verification/static_bounds_proof.py`, which now includes the one-op delta select that bits-flow plans emit (multiply_add at x, visible at x + 1; before, it could not represent it): 2,000 random 3-7 op DAGs at VLEN 2, 815 with a delta select, no violation (150 in the proofs suite). WR1c: 400 instances drawn so that the window row is above every per-engine row (the coupling binds), no violation (40 in the suite); before, the coupling bound on 39 of 2,000 instances. WR2 (non-vacuity): adding 1 to every non-trivial window is violated on 1,763 of 2,000 |
| `window_floor` is exact for its counts and monotone | Brute force over every resolution (`tests/test_emission_loop.py`) |
| The class table: every plan's per-group class counts, heads and tails are >= the table's (minimum over built plans) | **Observed, and false in general.** A class's head and tail depend on the encodings of the group's other rounds. The first plan that gathers in a round after selecting in earlier rounds lowers them by 5 to 22 cycles. Over 26 loop plans, 2 contradicted the table they were checked against (9 class fields). For one of them (`rows` iteration 0) the plan's own row bound was overestimated: 995 at solve time, 993 from its own classes. Among the seeds, the floor-optimal plan was overestimated by 9 (1,020 against 1,011). The final table never puts a plan's row bound above its cycles (0 of 32) |
| Floor (whole emission, `choice_floor`) | Bound (section 4) |
| Capped floor | Not a bound: a heuristic objective |

So (c) as built is "row form proved (parts 1 and 2) and checked (part 3), over an observed
table". Making the table a proved bound
would need class heads and tails computed as a minimum over the encodings of the neighbouring
rounds, not as a minimum over plans seen so far.

### 10.4 Surrogate quality

Over every valid build (32 distinct plans across the loop states, seeds included) and over the
26 plans the loops proposed. "Loop plans" are the regime that matters: everything there is
within 110 cycles of the best. Every point and the full trace:
`docs/emission_plans/loop_report.json`.

Caveats, which apply to every coefficient below:

- The plans are not a random sample. Each loop plan was chosen by one of these surrogates
  (the floor, the capped floor or the rows), so the range of each surrogate is restricted
  (floors 989 to 1,049 among loop plans, 20 of 26 within 989 to 997) and ties are heavy (the
  26 loop plans take 12 distinct floors, 5 each at 989 and 995; 15 distinct cycle values, 4 at
  1,104 and 4 at 1,125). Rank correlations on such data are unstable: a few plans move rho a
  lot.
- The learned cap `flow=0.92` was learned by the caps loop on these same builds, and it is
  scored on them: its rho is in-sample. `flow=0.9` was set by hand before the loop, but on
  earlier builds of the same structure, so it is not out-of-sample either.
- "floor + window rows" uses the final row table, learned from all builds including the one
  scored; at solve time each plan saw an earlier, partly different table.
- None of these is evidence that a surrogate will rank plans it has not seen.

| Surrogate | All builds (n = 32): Pearson / Spearman | Loop plans (n = 26): Pearson / Spearman |
|---|---|---|
| floor (`choice_floor`) | +0.82 / +0.30 | +0.18 / +0.03 |
| floor + window rows (final table) | +0.80 / +0.03 | +0.00 / **-0.41** |
| capped floor, learned `flow=0.92` | +0.82 / +0.75 | +0.32 / **+0.64** |
| capped floor, `flow=0.9` | +0.83 / +0.73 | +0.33 / +0.56 |
| exact static bound of the built kernel (`schedule_report`, any encoding) | +0.74 / -0.13 | -0.00 / -0.42 |

Reading:

- Over all builds the Pearson r of about 0.8 is carried by the extreme seeds (1,700 and 1,850
  cycles). Among near-optimal plans the floor carries no ranking information.
- The time-aware rows make the ranking worse. They are true lower bounds on this emission, but
  what they see (when the first gather can issue, when the last can end) is not what limits the
  greedy. Moving gathers earlier lowers the rows and raises the measured cycles. The exact
  static bound of each built kernel behaves the same (-0.42), so this is a limit of
  head-body-tail reasoning here, not of the observed table.
- Only the capped floor ranks plans, because the cap encodes a fact about this greedy (it
  does not fill flow beyond about 90% while loads and the VALU are full). That is a property of
  the scheduler, not of the plan. A different scheduler would need different caps, so a
  learned cap must be re-learned whenever stage 2 changes.

### 10.5 What did not work

| Attempt | Result |
|---|---|
| (a) no-goods on the floor | Enumerates floor-989 neighbours (1,125 to 1,212). The incumbent cut never fires: floors are about 110 below the cycles |
| (b) caps on the engines the schedule saturates | 1,118 at best: capping VALU and load lowers neither |
| (c) window rows in the objective | 1,127 to 1,150: stage 1 moves gathers into round 3, which the rows reward and the greedy does not |
| Caps at 4 groups | At 4 groups every schedule is latency-bound (steady utilisation about 50%), so no rule finds an engine to cap. The small shape is good for testing the mechanics only |
| The plateau at 1,104 | Five plans with floors 994 to 997 and different structures in rounds 1, 4 and 12 all schedule to exactly 1,104. The remaining gap (about 107 above the floor, section 5) is not reachable by choosing the structure with this greedy |

Reproduction (one build at a time, about 0.3 s per greedy build plus proof and validation):

```bash
L="uv run python -m scripts.emission_structure loop"
$L run --state artifacts/benders/seeds.json --modes ""            # the six seeds
for m in nogood caps capsu rows; do cp artifacts/benders/seeds.json artifacts/benders/$m.json; done
$L run --state artifacts/benders/nogood.json --modes nogood --budget 8 --patience 5 --seed-plan gather
$L run --state artifacts/benders/caps.json --modes caps --budget 10 --patience 5 --seed-plan gather
$L run --state artifacts/benders/capsu.json --modes caps-underfilled --budget 10 --patience 5 --seed-plan gather
$L run --state artifacts/benders/rows.json --modes rows --budget 10 --patience 5 --seed-plan gather
cp artifacts/benders/capsu.json artifacts/benders/nogoodcap.json
$L run --state artifacts/benders/nogoodcap.json --modes nogood --nogood-caps flow=0.92 --budget 8 --patience 6 \
  --seed-plan gather
$L report --state artifacts/benders/rows.json --extra-state artifacts/benders/nogood.json \
  --extra-state artifacts/benders/caps.json --extra-state artifacts/benders/capsu.json \
  --extra-state artifacts/benders/nogoodcap.json --caps flow=0.92               # 10.4
uv run --extra verify python -m verification.window_rows_proof --count 2000 --coupled 400   # WR1, WR1c, WR2, WR3
```
