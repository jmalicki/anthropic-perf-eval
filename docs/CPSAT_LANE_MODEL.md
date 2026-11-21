# Lane model of shiftable ops (`--cpsat-lane-model`)

A shiftable vector op `o` has, in the CP-SAT model (`model.rs` section 6), a cycle variable
`c_o`, a shift literal `shift_o`, eight lane variables `alu_c{o}_{i}` (`a_o,i`: the cycle
of each scalar op when `o` is shifted, each the start of an optional size-1 interval in the
scalar cumulative) and `completion_o >= c_o, a_o,i`. Review
[CPSAT_VARIABLE_MODELLING_REVIEW.md](CPSAT_VARIABLE_MODELLING_REVIEW.md) 4.2 (items 1, 2, 3, 5)
and 4.4 (a), (b) proposed two restrictions; both are now switches, default off:

| `--cpsat-lane-model` | Adds | Removes |
|---|---|---|
| `free` (default) | nothing: the exported model is byte-identical to `main` | |
| `sorted` | `lane_order_{o}_{i}`: `a_o,i <= a_o,i+1` (7 rows per shiftable op) | the 8! lane permutations of each op |
| `pinned` | `lane_pin_{o}_{i}`: `!shift_o => a_o,i = c_o` (8 enforced rows per op) | the free lanes of an unshifted op; each lane precedence pair `shift_d => a_o,i >= a_d,i + 1`, `!shift_d => a_o,i >= c_d + 1` becomes one unconditional row `shift_alu_prec_{o}_{i}_dep_{d}` |
| `sorted-pinned` | both | both |

## 1. Why neither loses a schedule

Both are dominance restrictions: every feasible schedule of the free model maps to a
feasible schedule of the restricted model with the same cycle variables, shift literals,
completions, liveness starts and, per op, the same multiset of lane times. So the makespan,
the tie-breaker objective and the per-cycle scalar-slot usage are unchanged, and the
restricted model keeps an optimal schedule of the free one.

The mapping is: sort the lanes of every shiftable op (sorted); set the lanes of every
unshifted op to its cycle (pinned); both, in that order (sorted-pinned).

**Every row of today's model that mentions a lane** has one of these shapes (from
`model.rs`; confirmed on exported models, below):

| Rows | Shape | Under a common lane permutation | Under per-op sorting | Under pinning (op unshifted) |
|---|---|---|---|---|
| `shift_alu_prec_*_on` | `a_o,i >= a_d,i + 1`, enforced by `shift_d`; **same lane index** (`element_dep_map` is the identity) | kept | kept: if `x_i >= y_i + 1` for all i then `sort(x)_k >= sort(y)_k + 1` | `c_o >= completion_d + 1 >= a_d,i + 1` (`consumer_after_completion`, `completion_ge_alu`) |
| `shift_alu_prec_*_off` | `a_o,i >= c_d + 1`, enforced by `!shift_d` | same bound on every lane | kept | `c_o >= completion_d + 1 >= c_d + 1` |
| `shift_alu_prec_*_valu` | `a_o,i >= c_p + 1`, non-shiftable producer `p` (includes the 8 scalar loads of a gather: each bounds every lane) | same bound on every lane | kept | twin row `prec_raw` on `c_o` |
| `select_prec_*_{fma,opt}` | `a_o,i >= c_s + 1 (+ extra)` | same bound on every lane | kept | twin row on `c_o` |
| `vb_elim_*_prec_{elim,keep}` | `a_o,i >= c_q + 1` / `>= c_vb + 1`, enforced by `eliminate_vb` / its negation | same bound on every lane | kept | `eliminate_vb => shift_o`; keep: twin precedence on `c_o` |
| `completion_ge_alu_*` | `completion_o >= a_o,i` | max | kept | `completion_ge_start` |
| `scratch_shifted_start_ge_alu_*` | `start_o <= a_o,i` | min | kept | `!shift_o => start_o = c_o` |
| lane intervals in `cumul_alu` | size 1, presence `shift_o`, demand 1 | same multiset per cycle | kept (a permutation) | absent when unshifted |
| interval binding | `a - a = 0` (the binding's start + size = end row) | trivial | trivial | trivial |

Nothing else mentions a lane: no objective term, no decision-strategy entry, no `lin_max`,
no `no_overlap`. Readers of a shifted op's result (vector or a decomposed lane) wait for
`completion_o`, which is permutation-invariant. The lane domains are `[asap, alap]`; the
cycle variable can only take values in that range in a feasible schedule (ASAP is sound for
precedence, and `c_o <= completion_o <= alap`), so pinning does not cut the cycle's domain
either.

**Where per-op sorting would be invalid.** If a row tied lane i of a reader to a producer
that differs per lane (for example lane-wise precedence from the eight scalar loads of a
gather into the lanes of a shifted reader, or a decomposed vector whose lane-i reader waited
for lane i of a shifted producer), sorting one op without the other would break it
(`LANE-sort-lane-specific-producers` is the Z3 counterexample). No such row exists today.
If one is added, sort only ops whose lanes have no lane-specific producer, or sort
consistently along each chain; `tests/test_cpsat_lane_model.py` fails until then.

### 1.1 Mechanical checks

- **Z3** (`verification/lane_model.py`, run by `./scripts/test.sh proofs`; see
  [verification/README.md](../verification/README.md)). Each check encodes a free schedule
  that satisfies every row and a mapped schedule that breaks one; UNSAT = no schedule lost.
  Unbounded integers, so each holds for all values. Sorting is a fixed compare-exchange
  network (Batcher, 19 comparators) applied to every op at once: one step keeps every row of
  a DAG with all shapes above, for each comparator, in the free and the pinned model
  (`LANE-sort-dag`, `LANE-sorted-pinned-dag`), and the network sorts every input
  (`LANE-network-sorts`); by induction a sorted schedule exists. `LANE-pin-dag` proves the
  pin mapping, `LANE-pin-unconditional-precedence` that the unconditional row equals the
  enforced pair once lanes are pinned.
- **Structure of the real model** (`tests/test_cpsat_lane_model.py`). On exported models
  (3 and 5 rounds, the latter with a gather round; forced multiply-add selects; eliminable
  broadcasts), every row that mentions a lane is classified: a uniform row must exist for all
  eight lanes of the op, a lane-wise row must use the same index on both sides and exist for
  all eight, lane intervals share one presence literal and demand 1. Any other row fails
  the test. The premises of the pinning proof (a twin row on the cycle variable, or
  `consumer_after_completion` for lane-wise rows; the lane domain containing the cycle's)
  are checked on the free model.
- **Optimum unchanged on a toy** (`tuning_lane_model_tests` in `cpsat_scheduler/tests.rs`):
  every lane model proves the same optimal makespan under both objectives.
- **Hint** (`tests/test_cpsat_hint_complete.py`): under every lane model the greedy hint is
  complete and feasible (each hinted variable fixed to its hint, CP-SAT finds the fixed
  model feasible). The greedy hints all eight lanes of an op at the op's cycle, which is
  sorted and pinned. A future per-lane hint (the split-shift hint) must hint lanes in
  non-decreasing order and an unshifted op's lanes at its cycle.

## 2. Measurements

PENDING
