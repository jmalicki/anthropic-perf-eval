# Relaxed-scratch reference

November 2026. How much could any scratch-saving technique buy (spilling to main memory,
a better allocator, fewer groups in flight)? Rerun the pipeline with the 1,536-word scratch
limit raised or removed and compare. This is a different kind of reference from the
capacity floor in [BOUND_GAP_ANALYSIS.md](BOUND_GAP_ANALYSIS.md): the capacity floor drops
the precedence constraints and keeps the slot limits; this drops the scratch limit and
keeps everything else.

## What is a bound and what is not

- **The relaxed run's cycle count is a schedule, not a floor.** It is what the same greedy
  (or the same CP-SAT budget) finds without the scratch limit. A better scheduler could do
  better with or without scratch. Real minus relaxed is "what scratch costs *this* scheduler".
- **The floor is CP-SAT's proven bound (`lb_cycles`) on the relaxed model.** Dropping a
  constraint can only lower the optimum, so it is also a valid floor for the real model,
  with one caveat: CP-SAT's domains are capped by the greedy hint (each op's latest cycle is
  its greedy cycle + 15, and the horizon comes from the greedy makespan). The relaxed run's
  greedy differs from the real run's, so the two bounds are over slightly different domains.
  Both are bounds for this emission and these domains, never for the problem.
- **Sanity rule:** a relaxed bound above the real schedule's cycle count would be a bug (or
  the domain caveat biting) and must be investigated. None was seen.

## How to run it

```bash
# Unbounded: no scratch cumulative in CP-SAT; greedy and allocator get 1,048,576 words.
uv run generate_ssa_kernel.py --rounds 16 --use-greedy --validate --no-scratch-constraint \
  --greedy-barriers none --greedy-priority wavefront --greedy-split-shift --pressure-inflation-threshold 0.95
# A larger finite scratch everywhere (greedy, per-section repair, CP-SAT, allocator):
uv run generate_ssa_kernel.py ... --scratch-budget-override 3072
# One row per run:
uv run python -m scripts.relaxed_scratch_summary real.json relaxed.json
```

A relaxed build (more than 1,536 words, or no constraint) does not run on the real machine.
The generator prints a banner before and after, writes `kernel_relaxed_scratch.json` and
`scheduler_output_relaxed_scratch.json` by default, refuses `kernel_to_validate.json`,
`kernel_upstream_check.json` and `scheduler_output.json`, and marks both files
(`"relaxed_scratch"` in the kernel, `"scratch_relaxed": true` in the scheduler JSON).
`scripts/check_upstream.sh` refuses the options and any marked kernel.

Validation still means something at the relaxed size: `--validate` runs the symbolic
dataflow replay (independent of the scratch size) and then the numeric check on
`problem.Machine(scratch_size=highest address + 1)`, our simulator's existing parameter;
the vendored `upstream/` files are not touched. Every row below is PROVED and PASSED.

## Where the limit enters

| Place | Before | Now |
|---|---|---|
| Greedy-only pipeline budget (`PipelineConfig.scratch_budget` → `GreedySchedulePass`, hint passes, pressure policies, deferral, force-schedule) | `effective_scratch_budget()` = 1,535 | `CompilerConfig::effective_scratch_budget` from `scratch_size` |
| CP-SAT pipeline greedy hint budget | `SCRATCH_BUDGET_HW` | `scratch_size` |
| Per-section repair: which sections exceed (`sections_exceeding_budget`), the section CP-SAT `scratch_budget`, the post-merge and hint-peak checks | 1,535 / 1,536 | from `scratch_size` |
| Full-kernel CP-SAT `SsaCpsatConfig.scratch_budget` (model.rs section 8b cumulative, post-solve `verify_scratch_budget_all_readers`, scratch tightening) | 1,535 | `cpsat_scratch_budget`: from `scratch_size`, or 0 (section 8b skipped) with `--no-scratch-constraint` |
| `CpSatSchedulePass`, `ComputeBoundsPass` scratch tightening | `effective_scratch_budget()` | same, from `scratch_size` (tightening skipped at budget 0) |
| Address assignment `AddressAssignConfig::budget` (`AssignAddressesPass`) | `SCRATCH_BUDGET_HW` | `scratch_size` |
| Not changed: `scheduler.rs::SCRATCH_BUDGET` (legacy non-SSA scheduler), `AddressAssignConfig::default`, test-only asserts in `ssa_lower.rs`, `problem.SCRATCH_SIZE` (now passed explicitly for relaxed kernels) | | |

The wavefront's `--greedy-groups-in-flight` is a separate knob; it exists because of
scratch, so the relaxed runs also try `--greedy-groups-in-flight 0` (no limit).

## Results

`forest_height=10`, `batch_size=256`. "Shaped" is the greedy command of
[GREEDY_SHAPE_ABLATION.md](GREEDY_SHAPE_ABLATION.md) section 1 (`--greedy-barriers none
--greedy-priority wavefront --greedy-split-shift --pressure-inflation-threshold 0.95`; CP-SAT
runs drop `--greedy-split-shift`, which CP-SAT ignores). "Relaxed" is `--no-scratch-constraint`.
Schedule cycles (simulator cycles). Peak = live scratch words from address assignment. Groups =
batch groups in flight at the peak-liveness cycle (a group is in flight from its first to its
last op; with round barriers all 32 always are). No run needed the per-section repair: every
log says "All sections within budget" or is greedy-only. Every row PROVED and PASSED.

### Greedy, 16 rounds

| Configuration | Real (1,536) | Relaxed | Peak real / relaxed | Groups real / relaxed | Capacity floor |
|---|---|---|---|---|---|
| Default shape (round barriers) | 2,007 (1,994) | 1,951 (1,938) | 1,268 / 1,628 | 32 / 32 | 1,271 |
| Shaped | 1,313 (1,315) | 1,313 (1,315) | 1,244 / 1,244 | 17 / 17 | 1,271 |
| Shaped, `--greedy-groups-in-flight 0` | 1,313 (1,315) | 1,313 (1,315) | 1,336 / 1,328 | 22 / 22 | 1,271 |
| Shaped, groups 0, stagger 10 | fails (1,648 > 1,536) | 1,308 (1,310) | - / 1,557 | - / 25 | 1,271 |
| Shaped, groups 0, stagger 8 | fails (1,652) | 1,302 (1,304) | - / 1,827 | - / 32 | 1,271 |
| Shaped, groups 0, stagger 6 | fails (1,650) | **1,293 (1,295)** | - / 2,009 | - / 32 | 1,271 |
| Shaped, groups 0, stagger 4 | fails (1,646) | 1,343 (1,345) | - / 2,097 | - / 32 | 1,271 |
| Shaped, groups 0, stagger 0 | fails (2,618) | 1,912 (1,914) | - / 3,081 | - / 32 | 1,271 |
| `--merge-hash-stages --index-form mirrored` (main's best emission), default shape | 1,931 (1,918) | 1,898 (1,885) | 1,269 / 1,620 | 32 / 32 | 1,130 |
| same, shaped | 1,197 (1,199) | 1,197 (1,199) | 1,118 / 1,118 | 17 / 17 | 1,130 |
| same, shaped, groups 0 | 1,192 (1,194) | 1,192 (1,194) | 1,305 / 1,305 | 24 / 24 | 1,130 |
| PR #51 plan `path_flow90.json` (detached checkout of `emission/vector-work` + this change), shaped | 1,097 (1,099) | 1,097 (1,099) | 1,050 / 1,050 | 15 / 15 | 994 |
| same, groups 0 | **1,087 (1,089)** | 1,087 (1,089) | 1,233 / 1,233 | 16 / 16 | 995 |

`--scratch-budget-override 3072` with shaped and groups 0 gives the same 1,313 as unbounded.

### CP-SAT, `--cpsat-objective makespan --cpsat-deterministic-time 60`

| Configuration | Real: cycles / `lb_cycles` / peak | Relaxed: cycles / `lb_cycles` / peak | Model (vars / constraints) real → relaxed | Capacity floor |
|---|---|---|---|---|
| 4 rounds, shaped hint | 425 (427) / 357 / 812 | 425 (427) / 357 / 828 | 29,919 / 102,223 → 23,211 / 76,104 | 357 |
| 4 rounds, default hint (round barriers) | 455 (457) / 357 / 1,266 | **405 (407)** / 357 / 1,793 | 29,932 / 103,392 → 23,219 / 77,094 | 357 / 358 |
| 16 rounds, shaped hint | 1,399 (1,401) / 1,271 / 1,156 | not measured yet (queued) | 108,171 / 378,621 → 84,201 / 284,023 | 1,271 |
| 16 rounds, default hint | not measured yet (queued) | not measured yet (queued) | | 1,271 |

The queued 16-round solves were stopped for CPU priority (and two run at once were killed for
memory); they run one at a time with the same options. 16 rounds, shaped, real: CP-SAT did not
improve on the 1,399-cycle hint and proved 1,271, the capacity floor.

`lb_cycles` is CP-SAT's proven bound; at 4 rounds it equals the capacity floor in every run,
so removing the scratch constraint did not let CP-SAT prove anything stronger. Every solve
ended on the deterministic budget (no wall-clock warning). With the shaped hint neither run
improved on the hint. With the default hint the relaxed greedy starts at 423 instead of 458
(no inflation throttle) and CP-SAT reaches 405, against 455 with scratch; 405 is also 20
below the best real 4-round result here (425).

## What it says

- **Scratch is not what limits the shaped schedules.** With the wavefront shape the relaxed
  runs give the same cycles as the real ones, for the default emission, main's best emission
  and PR #51's plan. Their peaks (1,050 to 1,336) are below 1,536 already.
- **Scratch does limit the default (round-barrier) shape**, through the pressure policy's
  inflation throttle: relaxed gains 56 cycles (2,007 → 1,951) and 33 with main's best emission,
  with peaks of 1,620 to 1,628. That shape is far from the floor for other reasons (rounds run
  one after another), so this is not where to spend effort.
- **The best observed relaxed greedy schedule saved 15 to 20 cycles at 16 rounds.**
  It reached 1,293 cycles at stagger 6 with a 2,009-word peak. The best real schedules reached
  1,308 at stagger 10 and 20 groups in flight, and 1,313 with the shaped greedy configuration.
  The relaxed schedule was still 22 cycles above the 1,271-cycle floor.
- **CP-SAT does find shorter schedules without scratch from the round-barrier hint** at 4
  rounds: 405 against 455 with scratch, and 20 cycles (5%) faster than the best real run of 425.
  The 16-round comparison is still queued.
- **The CP-SAT bound does not move without scratch** at 4 rounds (357 in every run, the
  capacity floor), and 1,271 (the floor) with scratch at 16 rounds. So the bound gap is not a
  scratch artefact.
- Side result on the real machine: PR #51's plan with `--greedy-groups-in-flight 0` is
  **1,089** simulator cycles (peak 1,233, valid on the 1,536-word machine; PROVED, PASSED and
  `./scripts/check_upstream.sh --strict` passes), 10 better than the 1,099 recorded there.
