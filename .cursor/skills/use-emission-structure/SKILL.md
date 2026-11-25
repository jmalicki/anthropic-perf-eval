---
name: use-emission-structure
description: Choose the kernel's structure with a solver - which tree-access encoding (gather, bits, bits-flow, mobius) each group of 8 lanes uses in each round - by minimising the capacity floor over op counts calibrated exactly from the emitter, then emit the plan with --emission-plan and measure it (PROVED, PASSED, upstream); or run the closed loop (loop) that feeds shaped-greedy schedules back into the choice. Use when asked which emission mix is best, when adding a tree-access encoding, or to refresh docs/EMISSION_STRUCTURE_MODEL.md.
---

# Emission structure model

**Module:** `scripts/emission_structure.py`. Plan format: `solver/emission_plan.py`. Design and
results: [docs/EMISSION_STRUCTURE_MODEL.md](../../../docs/EMISSION_STRUCTURE_MODEL.md).

## How to run

```bash
./scripts/build_scheduler_rs.sh --release      # the script uses the installed extension
M="uv run python -m scripts.emission_structure"

$M calibrate                      # cost table from 144 emitter builds (about 2 min); exact or an error
$M check --plans 30               # build unseen random plans: predicted counts == emitted counts?
$M table                          # the calibrated coefficients
$M predict [--plan plan.json|bits|bits-flow+depth4-after-wrap ...]   # counts and floor, no build
$M solve --out plan.json [--max-util flow=0.9] [--encodings gather,bits]
$M measure --shaped --upstream --plan plan.json --plan bits          # build, prove, validate, upstream --strict
$M sweep --shaped --max-util none --max-util flow=0.9 ...            # solve + measure per cap
$M solve --values-only --select-bit-source path --fix 15=gather --max-util flow=0.9 --out p.json
$M solve --what-if gather-without-address-op                        # uncalibrated estimate for a future encoding
$M solve --greedy-model --out plan.json   # minimise a calibrated HEURISTIC of the load-tail greedy's cycles
```

`--greedy-model` (`solve_plan(greedy_model={})`, `es.GREEDY_MODEL`) is fitted on the recorded
load-tail builds (index bits; leave-one-out MAE 6.2 cycles). It ranks index-bit plans near the
optimum (Spearman +0.71, floor -0.54), not path-bit plans (scratch), and is optimistic by 20 to
40 cycles on the plans it chooses. It is not a bound. To see why a built plan is slow, run
`uv run python -m scripts.shift_limit_diagnosis sched.json --floor N`: the shift count follows
from the cycle count (`V - s = 6T - I`), so look at the static bound, select resolution and flow
waits, and scratch (docs/GREEDY_SHAPE_ABLATION.md section 12).

## Closed loop with the shaped greedy (`loop`)

`scripts/emission_loop.py`, design and results in section 10 of the doc. Each iteration solves
stage 1, builds the plan once with the shaped greedy (proved, validated), reads the scheduler
JSON and feeds back. Builds are cached in `--state`, so later runs and other modes reuse them.

```bash
L="uv run python -m scripts.emission_structure loop"
$L run --state artifacts/benders/s.json --modes ""                       # seeds only (6 builds)
$L run --state artifacts/benders/s.json --modes caps-underfilled --budget 10 --seed-plan gather
$L run --state artifacts/benders/s.json --modes nogood --nogood-caps flow=0.92 --budget 8 --seed-plan gather
$L run --state artifacts/benders/s.json --modes rows --budget 10 --seed-plan gather
$L report --state a.json --extra-state b.json --caps flow=0.92                # trace + surrogate correlations
```

| Mode | Feedback | Bound? |
|---|---|---|
| `nogood` | no-good cut per built plan, floor < incumbent (`--nogood-caps`: enumerate under caps instead) | sound, weak |
| `caps` / `caps-underfilled` | learned `max_util`: cap the engine the schedule saturates / under-fills relative to the plan | heuristic; `caps-underfilled` rediscovers flow 0.9 in 3 builds |
| `rows` | M >= t + q + `window_floor`(work with head >= t, tail >= q) over plan variables | row form: engine coupling proved by Z3 for all counts, window containment argued by hand, composition a randomized check (`verification/window_rows_proof.py`); the per-class table is observed and can be contradicted by new mixes, so the default cut is `floor < best` (`--rows-cut heuristic` for `M < best + e`) |

Use `--load-tail` for the current greedy (adds `--greedy-load-priority final-last
--greedy-groups-in-flight 32 --greedy-stagger-cycles 9`; stored in the state, which then refuses
other flags) and `--select-bit-source path` for path bits. A plan that overflows the scratch
budget is recorded as a failed build with its peak, and `report` lists it; `report --near N`
adds a near-optimal subset to the correlations.

Result to remember (section 10.0, load-tail greedy): nothing beats the hand-made plans under
identical flags (index bits: `flow90.json` 1,039, five loop plans tie; path bits:
`path_flow90.json` 1,033, best loop plan 1,099). The 1,104 plateau is gone, but among plans within
30 cycles of the best the floor and every lower bound do not rank (Spearman -0.54 to -0.29). The
ranking follows the vector work the greedy leaves on the VALU after shifting, which stage 1 does
not model. Do not trust stage 1 to choose between plans within about 35 cycles, nor for the
round-4 encoding with path bits: build them. A run builds at most `--max-builds` (default 60) new plans
in total; builds are about 0.3 s each, but the machine budget is the limit.

The cost table is cached per context (rounds, batch, global emission options) in
`artifacts/emission_structure/costs_<hash>.json` and calibrated on first use. Context flags:
`--rounds --batch-size --forest-height --no-merge-hash-stages --index-form --select-bit-source
--values-only` (defaults: 16 / 256 / 10, merged hash, mirrored index).

Build a plan by hand: `uv run generate_ssa_kernel.py --rounds 16 --merge-hash-stages --index-form
mirrored --emission-plan plan.json ...` (not together with `--cached-select` or
`--depth4-select-rounds`, which it replaces).

## Numbers that are easy to confuse

| Number | Meaning |
|---|---|
| Predicted floor | `choice_floor` of the model's counts: every SelectChoice / ConstChoice / VALU shift resolved in the best way |
| Emitted floor | The same for the built kernel's SSA ops. Must equal the predicted floor; if not, the cost model is wrong |
| Scheduled floor | `scripts.schedule_report`'s floor: choices as this scheduler resolved them (greedy keeps consts on load) |
| Cycles | Schedule length / simulator count |

All floors are for one emission and ignore time; never quote them as bounds on the problem.

## Adding an encoding

1. Emit it in Rust behind `AccessEncoding` (`scheduler_rs/src/kernel_builder/config.rs`), dispatched
   per group in `emit_round` (`ssa_emit.rs`; respect `emits_group`), and add its name to
   `solver/emission_plan.py` (`ENCODINGS`, `allowed_encodings`).
2. Add its features to `feature_names` / `plan_features` (per-group count, setup booleans) and,
   in `solve_plan`, the matching CP-SAT expressions.
3. `calibrate`. If it raises "no exact integer fit", some resource is shared between features:
   find it in the emitter (dead code elimination of shared constants, CSE) and add a product to
   `interactions`. If a feature is reported "not identified", add design plans that vary it alone.
4. `check --plans 30` must report every plan exact; extend `tests/test_emission_structure.py`
   and `tests/test_emission_plan.py` (PROVED and PASSED with the encoding mixed into a round).

## Gotchas

- The pip `ortools` used for the structure solve and the OR-Tools linked into `scheduler_rs`
  crash when loaded into one process. Commands that solve and build use `solve_plan_isolated`
  (a spawned child); do the same in tests.
- With `--values-only` the floor prefers level-4 selects in round 15 (they also kill that group's
  late index updates), which schedules 40 to 100 cycles worse: use `--fix 15=gather`. The Z3
  negatives for an address-free `vload` index and a node-XOR fold are in
  `scripts/analyze_bound_gap.py --z3`; do not re-derive them.
- Plans with the same floor can schedule 100 cycles apart. Select work in the last round is the
  schedule's tail; the solve's tie-break puts it at the earliest visit. Compare plans by measured
  cycles, and use `--max-util` to keep headroom on engines the scheduler packs badly (flow).
