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

RESULTS_PLACEHOLDER
