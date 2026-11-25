# Start-up window probe on the best kernels

Does any schedule of the same ops do better in the start-up (cycles 0 to ~110) than the
greedy's? Three scheduler-side attempts recovered 3 of the 24 start-up cycles of the path
kernel, and the accounting had misled us three times, so this measures instead: the window
optimality probe (docs/WINDOW_OPTIMALITY_PROBE.md) re-optimises a window of the schedule
with the full CP-SAT model, every op before the window pinned, the ops after it pinned
relative to each other and moving earlier as one block, encodings (VALU shift, SelectChoice,
ConstChoice) free inside the window, on deterministic-time budgets.

## Answer: (a) proven no gain in the start-up, on both kernels

On the full model, every start-up window is **proven optimal with nothing saved**, with
the ASAP/ALAP tightening on (the default) and off:

| Kernel (probe reference) | [0, 60) | [30, 90) | [60, 120) | [0, 110) | [55, 165) |
|---|---|---|---|---|---|
| path, `path_flow90.json` + chain-first + native-first (1,032) | 0, optimal | 0, optimal | 0, optimal | 0, optimal | 0, optimal |
| index, `flow90.json` (1,045) | 0, optimal | 0, optimal | 0, optimal | 0, optimal | 0, optimal |

Free encodings; deterministic time 120 per window (every window closed in under 4 s);
reports `docs/window_probe/{path,idx}_full.json` (tightening on) and
`{path,idx}_full_notight.json` (off).

So, with the rest of the schedule fixed in its greedy order, no start-up schedule of these
ops is shorter: the start-up's loss is not a local scheduling problem. What it would take:

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

## What the probe could and could not take

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
  These runs used an analysis switch (`CPSAT_SECTION_CHECK_TIMELINE=model`, since removed):
  the check now counts with the allocator's intervals and needs no switch (BUGS.md,
  docs/SECTION_CHECK_AND_ROLLING_PROBE.md).

## The reduced model was wrong here

The probe's default reduced model (`restrict_to_window`) is documented as exact. On the path
kernel it is not:

| Run (path, reduced model unless noted) | Result |
|---|---|
| sweep W=60, dtime 60: [30, 90), [60, 120) | 1 cycle saved each, "optimal" (14 / 73 shift and 1 / 11 select changes) |
| sweep W=110 [0, 110), dtime 60, then 300 (2,235 s) | no solution better than the reference, bound 8 (open) |
| rolling W=60 over [0, 120) | 1,032 -> 1,031; **the full-model re-check rejects the schedule** ("infeasible during initial copy") |
| **full model** (`--no-prune`), rolling over [60, 120) | 1,032, optimal, 0 saved (2.8 s); kernel PROVED, PASSED (1,034 simulated) |
| **full model** sweep (table above) | every window 0 saved, optimal |

The 1-cycle "gains" were artifacts of a constraint the reduced model leaves out (logged in
BUGS.md). The rolling pass is protected by its full-model re-check; sweep rows are not
re-checked, so reduced-model sweep gains should not be trusted without `--no-prune`. On the
index kernel the reduced sweep and rolling pass found no gain (bounds of 1 left open); the
full model closes them.

## Commands

```bash
COMMON="--greedy-split-shift --cpsat-split-shift-hint --pressure-inflation-threshold 0.95 \
  --greedy-load-priority final-last --greedy-groups-in-flight 32 --greedy-stagger-cycles 9 \
  --merge-hash-stages --index-form mirrored"
uv run python -m scripts.window_optimality_probe --rounds 16 --window-sizes 60 110 \
  --stride-fraction 0.5 --cycle-range 0 61 --encodings free --no-rolling --no-prune \
  --deterministic-time 120 --time-limit 1800 --workers 4 --report path_full.json --no-validate \
  -- $COMMON --emission-plan docs/emission_plans/path_flow90.json --select-bit-source path \
  --greedy-alu-priority native-first --greedy-flow-priority chain-first
# index: --emission-plan docs/emission_plans/flow90.json --select-bit-source index
# tightening off: add --tightening none (window models) and --asap-alap-tightening none (generator)
```

All results are probe measurements on CP-SAT models of the emitted ops, not recorded kernel
results; the only kernels built here are the references (1,034 and 1,047 simulated), which
are worse than the best kernels and not results.
