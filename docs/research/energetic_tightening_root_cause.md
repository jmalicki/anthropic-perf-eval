# Root cause: unsound ASAP bounds from `--asap-alap-tightening energetic`

Paths are relative to the worktree `.claude/worktrees/cpsat-stall-investigation`. Scripts and
outputs are in `scratchpad/rc/` (`an1.py`, `an2.py`, `minimal_cex.py`, `dom.json`, `sim.json`).
Tags: **[V-code]** read at file:line, **[V-data]** computed from the exported files,
**[V-run]** small script executed, **[I]** inferred.

## Summary

The lead's guess (shifted ops counted as 8 ALU slots in one cycle) is **refuted**. Shiftable ops
are not in the rule at all. The rule that fires is the "contributing-only" ASAP push of
`tighten_asap_alap_by_slots` (`scheduler_rs/src/cpsat_scheduler/bounds.rs:1187-1265`), applied to
the 32 non-shiftable level-0 `multiply_add` ops of round 3. It counts ops whose ASAP equals `t`,
and when that count exceeds the engine capacity it pushes whichever ops come first in iteration
order. Counting can only say "not all of them fit at t"; it cannot say which ones move. The ops
it picks are the first in the group's sort order, i.e. the ones the greedy schedules earliest /
the ones with the smallest ALAP. It is a real unsoundness, with or without the ordering rows.

## 1. Which rule produced the 24 excluded lower bounds

- [V-data] `none`, `simple`, `symmetry-aware` have identical `c<N>` domains (sum of widths
  138690). `energetic` differs in 529 ASAPs (max +3), 0 ALAPs (sum of widths 137480). `rc/an1.py`.
- [V-data] The 24 excluded variables are 4 roots plus 20 descendants:
  - roots: positions 624, 625, 626, 627 (op ids 641-644), `multiply_add`, engine valu,
    non-shiftable, round 3, phase tree_access, batch group 0, tree_depth 3. ASAP 55 in the three
    other modes, 58 in energetic. Pinned solution has them at 56, 56, 55, 57.
  - descendants (each exactly +3, each one cycle below its energetic lower bound or more):
    628-630 `select_choice` (flow), 631 `^`, then the dependent chain 723, 731, 739, 740, 741,
    763, 771, 772, 773, 795, 803, 804, 819, 820, 821 (xor/hash/branch, `source_info.group` 7) and
    844 (round 4 tree_access `+`). They are a single dependency cone of the 4 roots; they are not
    "the same op position in different groups".
- [V-code] Emission puts every select-tree level under `ordering_group = [round, iter, level]`
  (`kernel_builder/emit_ssa.rs:298-305`); the key has no batch group. For depth 3 the level-0
  ops are 4 direct FMAs per batch group (`kernel_builder/ssa_emit.rs:1258-1267`), so group
  `[3, 0, 0]` holds 8 x 4 = 32 `multiply_add` ops. `multiply_add` is not in `can_shift_to_alu`
  (`types.rs:305-317`), so these are the "definitely VALU" chain (`bounds.rs:1142-1150,
  1169-1181`). SelectChoice and shiftable ops are filtered out of every chain.
- [V-data] All 32 have precedence ASAP 55; VALU capacity is 6.
- [V-code] The rule (`bounds.rs:1187-1265`): `total_min_demand[t]` = number of chain ops with
  `new_asap == t` (1188-1197). For each op in chain order, if `total_min_demand[asap] > cap` the
  op moves to `min(asap+1, first t with total <= cap)` (1216-1233), capped at its greedy cycle
  (1240-1244), and the counts are updated (1261-1264). The outer loop runs `MAX_ITER = 3` times
  (1156-1158).
- [V-run] Re-running exactly that in Python (`rc/an2.py`) on the sym-mode domains reproduces the
  export bit for bit: 26 ops pushed (iteration 1: first 26 of the 32 go 55->56, leaving 6;
  iteration 2: first 20 go 56->57; iteration 3: first 14 go 57->58). Final: 14 ops at 58, 6 at 57,
  6 at 56, 6 at 55. All 26 simulated values equal the exported energetic lower bounds. The +3 on
  group 0 is three +1 pushes, one per iteration; with a larger `MAX_ITER` it would be larger.
- [V-code] Chain order is `(greedy cycle, ALAP, idx)` (`symmetry_ordering.rs:54-63`), so the ops
  pushed furthest are the ones greedy schedules first (batch group 0: greedy 76-78), and without
  hints the ones with the smallest ALAP. The last six in the order (groups 6-7) keep ASAP 55.
- [V-code + V-data] The other 503 changed ASAPs come from `fix_precedence_bounds_fixpoint`
  (`bounds.rs:1675`, called at `model.rs:804`), which propagates the pushed ASAPs to all
  successors. The log says "Fixpoint repair: 503 ASAP changes"; 26 + 503 = 529.
- [V-code] The greedy cap (1240-1244) and the later widening to the greedy hint
  (`model.rs:828-843`) only guarantee the greedy schedule stays in the domains. Greedy has these
  ops at 76+, so neither does anything for cycles 55-57. This is why the bug never showed as
  infeasibility: it only removes schedules that differ from (are better than) the hint.

### The invalid assumption

`count(ops with ASAP == t) > cap` proves that at least `count - cap` of those ops are not at `t`.
The code turns that into "this particular op is not at `t`" for the first `count - cap` ops in
iteration order (`bounds.rs:1219-1233`). That is valid only if those specific ops are known to
be the ones that give way. Nothing establishes that:

1. The demand is summed over all chains of the engine (1187-1197), and ops in different
   ordering groups have no order between them.
2. Within a chain the declared order is `c[first] <= c[later]`. Under that order the ops that
   can be proven late are the last ones; the loop pushes the first ones. The Z3 theorem cited as
   proof, `T7-undercount-one-exclusion-sound` (`verification/theorems.py:612-637`), excludes the
   last op (`c1 <= c2`, exclude `c2 = 0`). The code does the opposite.
   `T7-undercount-chain-preserves-witness` (640-676) only shows one hand-picked witness survives.
3. The comment at `bounds.rs:1075-1078` and `docs/ASAP_ALAP_SLOT_TIGHTENING.md` (status table)
   call this "Z3-proven sound"; the theorems are 2-3 op toys that do not model the loop.

## 2. Real unsoundness or valid under a convention?

- [V-data] With `--cpsat-symmetry-rows none` (this export) it is a real unsoundness:
  `exp/pin_6x64s.sol.json` satisfies every constraint and violates 24 domains.
- [V-code] With `declared` rows it is still unsound, and worse. The rows are
  `c[i] <= c[j]` for consecutive ops of the same sorted order (`model.rs:1049-1061`). The first
  14 ops of the chain have ASAP 58, so the rows force all 32 FMAs to >= 58: VALU cycles 55-57 are
  closed to the whole group. The bound that the order would justify is
  `k-th op >= 55 + ceil(k/6) - 1`, which lets the first six sit at 55. [I] for this instance
  (not solved); [V-run] on the minimal counterexample below, where the excluded optimal schedule
  satisfies the ordering row.
- [V-code] With `checked` rows `symmetry_ordering` is `None` (`model.rs:655-668`) and
  `tighten_asap_alap_by_slots` returns its input unchanged (`bounds.rs:1126-1129`), so
  `--cpsat-symmetry-rows checked` silently disables the rule. `none` does not: the ordering is
  still built and used, as `docs/CPSAT_CONFIG_ABLATION.md` section 10 says.
- [I] The declared order is not a symmetry of the instance anyway. The key `[round, iter, level]`
  mixes 8 batch groups, and sorting by greedy cycle makes the rows say "all four FMAs of batch
  group 0 before all four of group 1, ...". Swapping whole batch groups (if they were
  interchangeable) would justify ordering one representative per group, not all members. The
  lead's colour-refinement result (879 classes / 883 ops at 4x64) says even that does not hold.
  The pinned solution has c626 = 55 < c624 = 56, which the declared rows would forbid.
- [V-code] Separate latent issue, inert here: `tighten_asap_by_symmetry_ordering`
  (`bounds.rs:1488-1587`, `k-th op >= ceil(k/C) - 1`) also runs when rows are `none`
  (`model.rs:669-686`), where nothing enforces the order it relies on. [V-data] It changed no
  domain in this instance (sym == simple), because the bound is at most 5 and ASAPs are 55.

## 3. ALAP

- [V-data] ALAP is identical in all four exports (0 differences).
- [V-code] ASAP pushes never touch ALAP; the fixpoint's backward pass only lowers a
  predecessor's ALAP from its successors' ALAPs (`bounds.rs:1731-1757`). The ALAP pull branch is
  behind `energetic_alap_pulls`, default false (`bounds.rs:425, 1270`).
- [I] If that flag is turned on it is unsound for a different reason: it pulls ALAP to the last
  `t` with `count(ASAP == t) <= cap - 1` (1283-1301). The number of ops whose earliest start is
  `t` says nothing about whether they are at `t`. Example: cap 1, X window [0,3], Y window [3,9]:
  X's ALAP is pulled to 2, yet X=3, Y=4 is feasible.
- [V-code] `tighten_asap_alap_by_scratch` (`bounds.rs:1335`) excludes cycles using a worst-case
  (over-counted) liveness profile, the same shape as the documented-unsafe R_precedence_demand.
  Default off (`bounds.rs:423`, `kernel_build_args.py:646`); not active here.

## 4. Minimal counterexample

[V-run] `rc/minimal_cex.py` (brute force over all schedules, plus a Z3 check; both ran).

Machine: 1 VALU slot, 12 ALU slots. Horizon 5.

| op | engine | deps | ordering group | precedence ASAP | ALAP |
|----|--------|------|----------------|-----------------|------|
| R  | ALU    | -    | -              | 0 | 2 |
| A  | VALU `multiply_add` | R | g | 1 | 3 |
| B  | VALU `multiply_add` | R | g | 1 | 4 |
| C  | ALU    | A    | -              | 2 | 4 |

Chain order by (ALAP, idx) is [A, B]. `total_min_demand[1] = 2 > 1`, A is processed first and
pushed to 2, B stays at 1; the fixpoint gives C >= 3.

- Tightened ASAP: R 0, A 2, B 1, C 3. Best makespan inside these domains: 4.
- Feasible schedule R=0, A=1, B=2, C=2 has makespan 3, satisfies precedence, capacity and the
  ordering row `c_A <= c_B`, and is excluded. Same result with the ordering row off.
- Same result with the greedy cap on and a feasible but late hint (R0 A2 B3 C3).
- The common predecessor R matters: with ASAP 0 for A and B, the symmetry pass has already set
  B >= 1 and the energetic count is 1.

(a) Unit test for `scheduler_rs/src/cpsat_scheduler/tests.rs`: build the four `SsaOp`s
(`SsaOp::new`, `symmetry_hint.ordering_group = vec![0]` on A and B), `asap`/`alap` as in the
table, `SymmetryOrdering::from_bounds(&ops, &asap, &alap, None, false)`, call
`tighten_asap_alap_by_slots(.., valu_cap = 1, alu_cap = 12, .., shiftable_valu = Some(&empty),
greedy_cycles = None, ordering = Some(&ord), false, None)` and assert `new_asap[A] <= 1`
(every returned ASAP must be <= the op's cycle in the feasible schedule {R:0, A:1, B:2, C:2}).
Fails today with `new_asap[A] == 2`. A second case with `greedy_cycles = Some(&[0, 2, 3, 3])`
covers the model path. A property version: for random small DAGs, brute-force all feasible
schedules and assert `tightened_asap[o] <= min feasible c_o` for every op.

(b) Z3 obligation in the style of `verification/` (refutation: UNSAT means sound):

```
0 <= c_R, c_A, c_B, c_C < 5
c_A >= c_R + 1, c_B >= c_R + 1, c_C >= c_A + 1      # precedence
c_A != c_B                                          # 1 VALU slot
c_A <= c_B                                          # declared ordering row (optional)
c_A < 2                                             # negation of the rule's bound
```

Z3 returns SAT with R=0, A=1, B=2, C=2. The existing T7-undercount theorems should be replaced
by this shape: encode the model's constraints plus the negation of a bound the Rust rule actually
produced (export the bounds from Rust for the instance), not a hand-derived exclusion.

## 5. Sound replacement and its cost

- [V-data] Cost of removing the rule on this instance: the energetic loop removes 1210 of 138690
  domain values (0.87%); 529 of 1284 ops change, by at most 3 cycles. `symmetry-aware` equals
  `simple` equals `none` here. So switching the default to `simple` costs under 1% of domain
  width.
- Recommended now: make `simple` (or `none`) the default and stop describing energetic as proven.
  With rows `none`, also skip `tighten_asap_by_symmetry_ordering` (gate both passes on
  `symmetry_rows == Declared`, `model.rs:655-656`).
- [I] A sound capacity-based ASAP needs precedence, because counting alone can never single out
  an op. Ancestor rule: for op O and any `a`, let `S` = ancestors of O with ASAP >= a. All of
  `S` must run in `[a, c_O - 1]`, so
  - VALU only: `c_O >= a + ceil(|non-shiftable VALU ops in S| / 6)`;
  - joint, in ALU-slot units (the lead's suggestion): `c_O >= a + ceil(W / 60)` with
    `W = 8 * (VALU ops in S, shiftable or not) + (ALU ops in S)`, capacity `6*8 + 12 = 60` per
    cycle. A shiftable op costs 8 units whichever engine it uses, and lanes may spread over
    cycles, so this stays valid for the free-lane model.
  Take the max over the distinct ASAP values `a` in the cone. The mirror with descendants gives a
  sound ALAP and a sound per-op tail for the makespan bound. Both pass the Z3 shape in 4(b) by
  construction; add the property test.

  > **Editor's note (2026-11-24).** This proposal was superseded and is not sound as stated
  > for shifted lanes. Counting a shiftable ancestor from its ASAP is wrong because its lanes
  > can run before its issue cycle (Z3 check `RB-issue-bound-for-shifted-ancestor` finds a
  > violating schedule), and the full mirror over descendants is wrong because the lanes of a
  > shifted reader can run before a shiftable producer completes (`RB-full-tail-for-shiftable`).
  > The rule in the code counts shiftable ancestors from a separate lane bound and, in the tail
  > of a shiftable op, only descendants that cannot be shifted: see
  > [ENERGETIC_REASONING_FOR_UNIT_TASKS.md](../ENERGETIC_REASONING_FOR_UNIT_TASKS.md) section 5.
  > The gate on `symmetry_rows == Declared` recommended above is in `model.rs`.
- [I] If declared rows are kept, the only count-based rule they justify is positional:
  `k-th op of the chain >= (min ASAP of the chain) + ceil(k/C) - 1`, pushing the last ops, and
  only when the rows are in the model.

## 6. How far the damage reaches

- [V-code] Only the op start variables `c<N>` take `final_asap`/`final_alap`
  (`op_times.rs:249-250`). Lane variables, `completion_c<N>` and the FMA/sub variables use the
  untightened precedence bounds (`op_times.rs:241-264, 307-341`). Matches the data: all 24
  out-of-domain variables are `c<N>`.
- [V-code] Makespan lower bound: `compute_capacity_lower_bound_global` takes op counts and
  capacities only (`model.rs:592-610, 3384`); not affected. `makespan_rows.rs` has no reference
  to asap/alap (grep: 0); not affected. Horizon is computed before tightening
  (`model.rs:560-611`); not affected. ALAP: not affected (section 3).
- [V-code] Affected besides the domains: the declared ordering and lexicographic-stream rows are
  chosen and filtered with the final bounds (`model.rs:1052-1055, 1106-1118, 1172`); hint
  clamping (`model.rs:3856, 3960, 4002, 4026`); the `Domain` entries exported for bulk-move
  (`model.rs:4688-4693`), so bulk-move can report these false bounds as binding.
- [V-code] The same function runs in the pipeline `ComputeBounds` pass with no greedy cap
  (`pipeline/passes/compute_bounds.rs:98-116`) and in `compute_ssa_asap_alap_bounds`
  (`bounds.rs:1803-1830`). Greedy uses these bounds for policy sorting/inflation by default and
  as hard ASAPs with `greedy_use_full_bounds` (`passes/greedy.rs:111-119`).
- [I] Every CP-SAT result obtained with the default tightening and rows `declared` or `none`
  searched a space missing schedules, including "optimal" and lower-bound claims from those
  runs. Whether the 160-cycle optimum of this instance survives in some other form inside the
  energetic domains was not determined (no solves were run).
