# Start-up window probe: the after-block cannot move earlier

Does any schedule of the same ops do better in the start-up (cycles 0 to ~110) than the
greedy's? Three scheduler-side attempts recovered 3 of the 24 start-up cycles of the path
kernel, and the accounting had misled us three times, so this measures instead: the window
optimality probe (docs/WINDOW_OPTIMALITY_PROBE.md) re-optimises a window of the schedule
with the full CP-SAT model, every op before the window pinned, the ops after it pinned
relative to each other and moving earlier as one block, encodings (VALU shift, SelectChoice,
ConstChoice) free inside the window, on deterministic-time budgets.

## What is proved, and what is not

Every run below asks one question: **can the ops after a window move earlier as one rigid
block** (relative order fixed), with the window re-optimised inside the CP-SAT model and the
rest of the schedule pinned? A "0, optimal" answer is a proof for that model and that
reference only. It does not prove that the start-up has no scheduling slack: a gain that needs
the later schedule to re-interleave is outside the question.

The first runs (table A) used the model as it was then, which also contained:

- the greedy ALAP cap (every op at most its reference cycle + 15);
- the default `declared` symmetry rows (`sym_ord` / `sym_lex`), which BUGS.md records as not
  justified by a proven symmetry: they can raise the optimum, and they fix the cross-group
  order that a start-up re-interleave would change;
- a scratch overcount (a select's delta counted on the vselect path: seven 8-word tables, 56
  words of intervals, a 48-word higher peak (1,544 against the allocator's 1,496) on the path
  kernel; fixed in #92).

Table B re-runs the windows on the fixed model of #92, from the current best references, with
the symmetry rows removed and the ALAP cap lifted.

### A. First runs (model of the time; references from the pipeline greedy)

On that model, every start-up window is optimal with nothing saved, with the ASAP/ALAP
tightening on (the default) and off:

| Kernel (probe reference) | [0, 60) | [30, 90) | [60, 120) | [0, 110) | [55, 165) |
|---|---|---|---|---|---|
| path, `path_flow90.json` + chain-first + native-first (1,032) | 0, optimal | 0, optimal | 0, optimal | 0, optimal | 0, optimal |
| index, `flow90.json` (1,045) | 0, optimal | 0, optimal | 0, optimal | 0, optimal | 0, optimal |

Free encodings; deterministic time 120 per window (every window closed in under 4 s);
reports `docs/window_probe/{path,idx}_full.json` (tightening on) and
`{path,idx}_full_notight.json` (off).

### B. Re-run on the #92 model, best references, relaxed rows

References: path **1,012** and index **1,033**, the greedy-only bests of #93 (all three
critical-path rewrites, `--greedy-justify 10`), carried into the CP-SAT pipeline with
`--cpsat-greedy-bounds policy-only --cpsat-split-shift-hint`; both pass the all-fixed check
(optimal at their makespan). Full model (`--no-prune`), free encodings, deterministic time 60
per window, 900 s backstop, 4 workers. Start-up: sweep W=60 ([0, 60), [30, 90), [60, 120))
and W=110 ([0, 110), [55, 165)); rolling: W=60 non-overlapping over the whole schedule.

| Model setting | Path 1,012 start-up | Path 1,012 rolling | Index 1,033 start-up | Index 1,033 rolling |
|---|---|---|---|---|
| `declared` symmetry rows, ALAP cap 15 (defaults) | 5/5 optimal, 0 saved | 17/17 optimal, 0 saved | 5/5 optimal, 0 saved | 18/18 optimal, 0 saved |
| `--cpsat-symmetry-rows none`, `--cpsat-greedy-alap-slack none` | 5/5 optimal, 0 saved | 17/17 optimal, 0 saved | 5/5 optimal, 0 saved | 18/18 optimal, 0 saved |
| `--cpsat-symmetry-rows checked`, `--cpsat-greedy-alap-slack none` | 5/5 optimal, 0 saved | 17/17 optimal, 0 saved | 5/5 optimal, 0 saved | 18/18 optimal, 0 saved |

Solve time per run 68 to 233 s. With `none` the window models have 1,000 fewer constraints
than with `declared` (the symmetry rows). `checked` builds the same models as `none` here: a
pinned window probe skips the checked anchor rows by design (CLAUDE.md), so the `checked` row
repeats the `none` row and is not a separate test of checked symmetry. Reports:
`docs/window_probe/rescoped/{PATHB,INDEXB}_{declared,none,checked}_{startup,rolling}.json`.

Commands for table B (run from the #92 branch, `section-check-timeline`, at 0d63e67):

```bash
COMMON="--rounds 16 --greedy-barriers none --greedy-priority wavefront --greedy-split-shift \
  --pressure-inflation-threshold 0.95 --greedy-load-priority final-last --greedy-groups-in-flight 32 \
  --greedy-stagger-cycles 9 --merge-hash-stages --index-form mirrored"
REW="--reassociate-gather-xor --index-early-terms final --select-xor-pushdown first-pass --greedy-justify 10"
PATHB="$COMMON --emission-plan docs/emission_plans/path_flow90.json --select-bit-source path \
  --greedy-alu-priority native-first --greedy-flow-priority chain-first $REW"
INDEXB="$COMMON --emission-plan docs/emission_plans/flow90.json --select-bit-source index $REW"
G="--cpsat-split-shift-hint --cpsat-greedy-bounds policy-only"
P="uv run python -m scripts.window_optimality_probe --rounds 16 --encodings free \
  --deterministic-time 60 --time-limit 900 --workers 4 --no-validate --no-prune"
STARTUP="--window-sizes 60 110 --stride-fraction 0.5 --cycle-range 0 61 --no-rolling"
ROLLING="--window-sizes 60 --no-sweep"
# for K in PATHB INDEXB, rows:
#   declared: $P $STARTUP --report ${K}_declared_startup.json -- ${!K} $G --cpsat-symmetry-rows declared
#             $P $ROLLING --report ${K}_declared_rolling.json -- ${!K} $G --cpsat-symmetry-rows declared
#   none:     ... -- ${!K} $G --cpsat-symmetry-rows none --cpsat-greedy-alap-slack none
#   checked:  ... -- ${!K} $G --cpsat-symmetry-rows checked --cpsat-greedy-alap-slack none
# Reduced vs full (path, pipeline greedy 1,032):
#   $P(without --no-prune) --window-sizes 60 --stride-fraction 0.5 --cycle-range 30 61 --no-rolling \
#     -- $COMMON --emission-plan docs/emission_plans/path_flow90.json --select-bit-source path \
#        --greedy-alu-priority native-first --greedy-flow-priority chain-first --cpsat-split-shift-hint
#   and the same with --no-prune.
```

The reports of these runs predate the probe recording its generator arguments: the declared,
none and checked reports differ only in wall times. The probe now writes `generator_args` into
the report's spec and, per window, the model size and the stop reason (`solve`:
`num_variables`, `num_constraints`, `deterministic_time`, `stop` = proved / deterministic
budget / wall-clock backstop); `docs/window_probe/rescoped/PATHB_none_rolling_recorded.json`
is the `none` path rolling run regenerated with it (same tree as above): 17/17 windows proved, 0 saved, 1,012; its spec lists `--cpsat-symmetry-rows none --cpsat-greedy-alap-slack none` and the first window's model has 433,801 constraints.

**What this shows:** neither the symmetry rows, nor the ALAP cap, nor the scratch overcount
hid a gain of this kind. In no window of either best kernel, start-up or later, can the rest
of the schedule move earlier as a block. Whether a re-interleaving of the later schedule could
make room in the start-up is still open; this probe cannot answer it.

What a shorter start-up would take:

- **Emission: less work before the load stream saturates.** The first gather is ready at
  t = 51 (6 setup ops, then rounds 0 to 3: one select, the xor, the 9-op hash chain and the
  branch `&` each; then the index update, the address, the load; docs/STARTUP_AND_STALL_LOOP.md
  B.3). The saturated load stream after it is what pins the end of the schedule, and the
  probe shows it cannot start earlier without shortening that chain or the work per group
  ahead of its first gather (a shorter setup chain, e.g. broadcasting the root before the
  deferred-constant xor; fewer hash ops per round).
- **Or a non-local change** the probe does not try: it moves the rest of the schedule as a
  rigid block, so gains that need the post-start-up schedule to re-interleave (another
  stagger profile, another admission order) are outside its question. The early-stagger and
  admission experiments (docs/GREEDY_CONST_CHOICE.md, docs/STALL_DEADLOCK.md) covered part of
  that and found nothing.

## What the probe could and could not take (table A runs)

This section describes the table A runs; table B's references are the best kernels (1,012 /
1,033), carried with `--cpsat-greedy-bounds policy-only`.

- **Reference.** The probe takes its reference from the CP-SAT pipeline's greedy, which runs
  with full bounds (ASAP deferral, capacity deferral) and gives **1,032** (path) and **1,045**
  (index) cycles, not the greedy-only **1,030** and **1,039**. Both references are proved
  solutions of the model (all-fixed check, optimal) and their kernels validate (path 1,034
  simulated, index 1,047; PROVED, PASSED). The probe therefore answers the question for
  schedules 2 and 6 cycles longer than the best kernels; nothing here says the greedy-only
  start-ups are optimal, but neither probe found any start-up slack at all.
- **Split shifts** are kept: `--cpsat-split-shift-hint` carries the greedy's per-lane cycles
  into the reference. **chain-first / native-first** are greedy orderings and shape the
  reference like any other greedy option. **The greedy const policy** (`--greedy-const-policy`)
  is not available in the CP-SAT pipeline (its hint names only Load or AddImm); the
  references use the default policy, and the window models' encodings are free anyway.
- **The CP-SAT pipeline could not carry the path kernel** as it stood: its per-section
  budget check counts liveness on the cycle timeline (1,597 > 1,535 for a schedule whose
  exact peak is 1,483) and starts a no-shift feasibility repair of the whole kernel, which
  found nothing in 600 s (without shifts the vector ALU alone needs 1,220 cycles).
  The table A runs used `CPSAT_SECTION_CHECK_TIMELINE=model` (an analysis switch, removed in
  #92), which made the check use the model's own timeline; the section then fit (1,533). That
  check had no per-lane shifted cycles and could miss an over-budget section; the probe's
  all-fixed check would then have rejected the reference.

## The reduced and full models disagreed

The probe's default reduced model (`restrict_to_window`) is documented as exact. On the path
kernel the two models disagreed:

| Run (path, reduced model unless noted) | Result |
|---|---|
| sweep W=60, dtime 60: [30, 90), [60, 120) | 1 cycle saved each, "optimal" (14 / 73 shift and 1 / 11 select changes) |
| sweep W=110 [0, 110), dtime 60, then 300 (2,235 s) | no solution better than the reference, bound 8 (open) |
| rolling W=60 over [0, 120) | 1,032 -> 1,031; **the full-model re-check rejects the schedule** ("infeasible during initial copy") |
| **full model** (`--no-prune`), rolling over [60, 120) | 1,032, optimal, 0 saved (2.8 s); kernel PROVED, PASSED (1,034 simulated) |
| **full model** sweep (table above) | every window 0 saved, optimal |

Which model was right is not shown. The dropped constraint was never identified, and the
1,031 schedule was never lowered and validated on its own. The full model of the time was
itself too tight: it carried the delta overcount fixed in #92 (seven 8-word tables, 56 words;
48 words at the peak). So the 1-cycle gains
may have been real and the full model wrong. There is no tool here that lowers a rolled
schedule without one of the two models, so this stays open (BUGS.md).

Re-run on the #92 model, same reference (path, pipeline greedy, 1,032), W=60:

| Window | Reduced model | Full model |
|---|---|---|
| [30, 90) | 0 saved, bound 1, ended on the budget (475 s) | 0 saved, optimal (4.6 s) |
| [60, 120) | 0 saved, bound 1, ended on the budget (403 s) | 0 saved, optimal (4.7 s) |

No disagreement was observed, but that settles nothing: the reduced runs did not close
(bound 1 left open on both windows) while the full model proved 0. #92's fix only removes
demand, so a schedule the old reduced model accepted should still be feasible in the new
reduced model while the new full model proves it impossible; the disagreement most likely
persists and the reduced run did not find that schedule within its budget. These runs used
#92's branch before it had main's #79 change to `restrict_to_window` (producers of ALU-derived
operands kept). Re-run on a tree with both #79 and #92's model (#92 with #89 merged in): the
same, [30, 90) and [60, 120) both 0 saved, bound 1, stopped on the deterministic budget
(60.1 and 60.0 deterministic time, 297 s and 236 s wall), as recorded per window in
`docs/window_probe/rescoped/i10_reduced_79.json`. Reports: `docs/window_probe/rescoped/i10_{reduced,full}.json`.

The rolling pass is protected by its full-model re-check. Sweep rows are not re-checked, so
reduced-model sweep gains should not be trusted without `--no-prune`. On the
index kernel the reduced sweep and rolling pass found no gain (bounds of 1 left open); the
full model closes them.

## Dependency on #92

The runs in table A and the first reduced-model runs used `CPSAT_SECTION_CHECK_TIMELINE=model`,
an analysis switch in this branch's `tick_tock_ssa` that is not recorded in `run_options`. #92
removes it: its section check counts with the allocator's intervals and needs no switch. This
branch must not land on main without #92.

## Commands

```bash
COMMON="--greedy-split-shift --cpsat-split-shift-hint --pressure-inflation-threshold 0.95 \
  --greedy-load-priority final-last --greedy-groups-in-flight 32 --greedy-stagger-cycles 9 \
  --merge-hash-stages --index-form mirrored"
export CPSAT_SECTION_CHECK_TIMELINE=model
uv run python -m scripts.window_optimality_probe --rounds 16 --window-sizes 60 110 \
  --stride-fraction 0.5 --cycle-range 0 61 --encodings free --no-rolling --no-prune \
  --deterministic-time 120 --time-limit 1800 --workers 4 --report path_full.json --no-validate \
  -- $COMMON --emission-plan docs/emission_plans/path_flow90.json --select-bit-source path \
  --greedy-alu-priority native-first --greedy-flow-priority chain-first
# index: --emission-plan docs/emission_plans/flow90.json --select-bit-source index
# tightening off: add --tightening none (window models) and --asap-alap-tightening none (generator)
```

Table B used the #92 branch (`section-check-timeline`) with `--cpsat-greedy-bounds policy-only
--cpsat-split-shift-hint`, the three rewrites and `--greedy-justify 10`, plus the model setting.

All results are probe measurements on CP-SAT models of the emitted ops, not recorded kernel
results. The only kernels built in the table A runs are their references (1,034 and 1,047
simulated), worse than the best kernels and not results; table B builds no kernel
(`--no-validate`).
