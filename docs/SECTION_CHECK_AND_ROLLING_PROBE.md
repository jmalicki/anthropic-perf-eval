# Section scratch check, model delta overcount, rolling window pass

Three things, measured on the two best 16-round greedy kernels (docs/CRITICAL_PATH.md).

## 1. The per-section scratch check counts what the allocator counts

Before the full-kernel solve the CP-SAT pipeline checks each section's scratch peak and
repairs the sections over budget with a per-section CP-SAT feasibility solve. The check
counted on the cycle timeline with its own liveness, which had no per-lane cycles for split
shifts; on the phase timeline it could therefore miss an over-budget section (3x256 with the
`greedy` pressure policy), so PR #64 left it on the overcounting cycle timeline. On the
1,030-cycle path kernel it counted 1,597 words > 1,535 for a schedule the allocator places in
1,496, and the whole-kernel no-shift repair it started found nothing in 600 s.

`round_budget::section_budget` now builds the allocator's own intervals
(`address_assign::compute_live_intervals` with the greedy's shifted lane cycles, multiply-add
selects, ALU-derived constants and vector groups; spans from `compute_value_spans`) and takes
each section's peak over its half-cycle range, on the model's timeline
(`--cpsat-scratch-time`; `cycle` widens every span to whole cycles).

| Kernel (16 rounds, greedy) | Old check | New check | Allocator (`scratch_pools.words`) |
|---|---|---|---|
| path 1,030 (`path_flow90.json`, chain-first + native-first) | 1,597 | 1,496 | 1,496 |
| path 1,026 (+ `--reassociate-gather-xor --greedy-const-policy alu-first`) | | 1,456 | 1,456 |
| index 1,038 (`flow90.json`, `--index-early-terms final`) | | 1,422 | 1,422 |

The check's peak is reported as `section_check_peak` (stats and scheduler JSON).
Tests: `test_section_check_counts_what_the_allocator_counts` (Rust, 4 rounds, round
barriers / wavefront / wavefront + split shifts) and `tests/test_section_check.py`.

The check bounds peak live words. A section within the budget can still lack an address
layout (the `words` model's known limit: docs/STALL_DEADLOCK.md records `floor10` at 28 groups
in flight with 0 words forced and no layout), so "within the budget" here means "no repair
needed by word count", not "address assignment will succeed". All three checks (which sections
to repair, the post-merge check, the hint check) compare against the same budget, the 1,535
assignable words (the zero word is not counted).

After a per-section solve the check uses that section's new encodings: multiply-add selects
(per-section overrides) and constants (`merge_section_const_encodings`: the solve's AddImm /
AluDerive, the greedy's for ops not sent to the solve). Test `test_merge_section_const_encodings`.

**What moves.** The greedy-only pipeline runs the check only to report `section_check_peak`;
an error there is printed and the peak left out, so it cannot fail or change a greedy-only
build (address assignment stays the judge). No greedy number moves (1,026 and 1,038 rebuilt
unchanged). CP-SAT-pipeline builds can only repair a different set of sections.
Flagged sections old vs new on 40 cells (2x16, 3x64, 4x64, 2x256, 3x256, 4x256, 6x64, 16x256 ×
default, `--no-valu-shift`, `--scheduling-pressure-policy greedy`, shaped, shaped + 0.95): the
same everywhere except 4x256 with the `greedy` policy (sections 2, 3 → 3). 3x256 with the
`greedy` policy is still flagged (1,544 > 1,535 exact); its per-section repair fails within
`--cpsat-deterministic-time 20` before and after this change.

`--cpsat-greedy-bounds policy-only` (new) runs the CP-SAT pipeline's greedy with bounds for the
pressure policy only, as `--use-greedy` does; the default (`full`: ASAP deferral, clamping,
capacity deferral) gives 1,044 (index) and 1,032 (path) instead of 1,038 and 1,030. With
`policy-only` and `--cpsat-split-shift-hint`, CP-SAT starts from exactly the greedy-only
kernels (hint "complete and feasible"). `--greedy-const-policy` is not carried by the CP-SAT
pipeline, so the 1,026 path kernel cannot be its warm start; the path runs below use 1,030.

## 2. The CP-SAT model counted a select's delta on the vselect path

With the check fixed, the model still rejected the 1,030 path reference (all-fixed check
INFEASIBLE; the hint's own count was 1,544 against the allocator's 1,496; accepted with the
model's scratch constraint removed). Diffing the model's hint intervals against the
allocator's at the peak cycle: seven 8-word node-delta tables (`vbroadcast` + `add`,
`valu_selection_init`) were live from cycle ~10 to ~970 in the model and [10, 11] for the
allocator. They are the `delta_vreg` of SelectChoice ops whose other operands do not include
them. Only the multiply-add lowering reads the delta; the vselect lowering does not, and the
allocator counts it only on the multiply-add path (`OpLiveness::use_cycles`). The model
counted it on both.

Fix (`cpsat_scheduler/model.rs`): `delta_only_readers` finds `(value, select)` pairs where
the select reads the value only as its delta; such a reader adds to the value's end-of-life
maximum only when its multiply-add literal is true (a term equal to that path's read end, 0
otherwise; `Always` adds it, `Never` drops it). The hint and the post-solve all-readers check
(`verify_scratch_budget_all_readers`) apply the same rule. Regression test
`test_ssa_cpsat_delta_read_only_on_the_multiply_add_path`: smallest feasible budget 24 words
with the select forced to vselect (32 before the fix), 40 forced to multiply-add, 24 with
`Optimize`. With the fix the 1,030 path reference passes the all-fixed check (optimal).

A second test, `test_ssa_cpsat_delta_kept_live_when_the_solver_chooses_multiply_add`, has two
delta selects compete for the one flow slot under `Optimize`, so the solver takes the
multiply-add path at some budgets, and checks that address assignment fits every solution in
its budget; with the multiply-add row of the fix removed it fails (48 live words at a 40-word
budget). If every read of a value is conditional (not reachable today: `true_val = base +
delta` always reads the delta), the value is kept as a dead value through the write slot of
its last write (`exec_end + 2`), in the model and in the post-solve check.

**What this changes.** The default emission uses delta selects (`ssa_emit.rs`, the cached
tree levels), so every default CP-SAT model changes: one extra variable and two enforced rows
per (value, select) pair where the select reads the value only as its delta, and a max over
readers instead of a single reader for those values. Per-section repair models change the same
way, and `scripts/cpsat_model_identity.py` records made before this change no longer match
(re-record the base). It only removes an overcount (every schedule feasible before is still
feasible), but the search changes, so any documented CP-SAT result can move. At risk: the
4-round 485 (CLAUDE.md), the CP-SAT tables of docs/CPSAT_CONFIG_ABLATION.md,
docs/CPSAT_LANE_MODEL.md, docs/GREEDY_SHAPE_ABLATION.md (CP-SAT from each greedy start) and
docs/4-round-phase-results.md, and the window-probe results of docs/STARTUP_WINDOW_PROBE.md
and docs/WINDOW_OPTIMALITY_PROBE.md. Re-measured: see "Re-measured after the model change"
below; everything else is not re-measured.

## 3. Rolling full-model window pass over the whole schedule

Window optimality probe (docs/WINDOW_OPTIMALITY_PROBE.md), rolling pass only: non-overlapping
windows of W = 60 cycles from cycle 0, each solved against the schedule the earlier ones
produced; full model (`--no-prune`), encodings free, ops after the window moved as one rigid
block; deterministic time 60 per window, 900 s wall-clock backstop, 4 workers. Every window's
result is re-checked on the full model; the final kernel is lowered and validated.

**Answer: no cycle saved anywhere on either kernel.** Index: all 18 windows proven optimal
with 0 saved, 159 s of solving in total. Path: 15 of 18 proven 0; three steady-state/tail
windows ended on the budget with 0 found and an open bound (2, 2, 4), 2,334 s in total. Both
final kernels are the references (1,038 and 1,030), PROVED and PASSED.

| Window | Phase | Index 1,038: saved, status, wall | Path 1,030: saved, status, wall |
|---|---|---|---|
| [0, 60) | start-up | 0, optimal, 3 s | 0, optimal, 3 s |
| [60, 120) | start-up | 0, optimal, 3 s | 0, optimal, 3 s |
| [120, 180) | steady state | 0, optimal, 5 s | 0, optimal, 5 s |
| [180, 240) | steady state | 0, optimal, 5 s | 0, optimal, 6 s |
| [240, 300) | steady state | 0, optimal, 5 s | 0, optimal, 5 s |
| [300, 360) | steady state | 0, optimal, 5 s | 0, optimal, 5 s |
| [360, 420) | steady state | 0, optimal, 5 s | 0, optimal, 5 s |
| [420, 480) | steady state | 0, optimal, 5 s | 0, optimal, 4 s |
| [480, 540) | steady state | 0, optimal, 4 s | 0, optimal, 6 s |
| [540, 600) | steady state | 0, optimal, 3 s | 0, open, bound 2, 673 s |
| [600, 660) | steady state | 0, optimal, 3 s | 0, open, bound 2, 712 s |
| [660, 720) | steady state | 0, optimal, 3 s | 0, optimal, 3 s |
| [720, 780) | steady state | 0, optimal, 3 s | 0, optimal, 3 s |
| [780, 840) | steady state | 0, optimal, 3 s | 0, optimal, 3 s |
| [840, 900) | steady state | 0, optimal, 3 s | 0, optimal, 3 s |
| [900, 960) | steady state | 0, optimal, 3 s | 0, optimal, 2 s |
| [960, 1020) | tail | 0, optimal, 3 s | 0, open, bound 4, 894 s |
| [1020, 1080) | tail | 0, optimal, 95 s | 0, optimal, 2 s |

Mid-schedule, with all 32 groups in flight, windows of ~600 ops close in 3 to 6 s on the index
kernel and on most of the path kernel; the open path windows ([540, 600), [600, 660),
[960, 1020)) spent the whole deterministic budget (700 to 900 s wall) without finding a
better schedule. The rigid after-block means a window can only save by moving all later work
earlier, so an open bound of 4 is an upper limit on that kind of gain, not evidence of one.
Nothing changed encodings in a way that saved cycles, so there is no rule for the greedy to
imitate.

Reports: `docs/window_probe/rolling_index_1038.json`, `docs/window_probe/rolling_path_1030.json`.

## Commands

```bash
COMMON="--rounds 16 --greedy-barriers none --greedy-priority wavefront --greedy-split-shift \
  --pressure-inflation-threshold 0.95 --greedy-load-priority final-last --greedy-groups-in-flight 32 \
  --greedy-stagger-cycles 9 --merge-hash-stages --index-form mirrored"
INDEX="$COMMON --emission-plan docs/emission_plans/flow90.json --select-bit-source index --index-early-terms final"
PATH_1030="$COMMON --emission-plan docs/emission_plans/path_flow90.json --select-bit-source path \
  --greedy-alu-priority native-first --greedy-flow-priority chain-first"
uv run python -m scripts.window_optimality_probe --rounds 16 --window-sizes 60 --encodings free \
  --no-sweep --no-prune --deterministic-time 60 --time-limit 900 --workers 4 \
  --report rolling_index.json -- $INDEX --cpsat-split-shift-hint --cpsat-greedy-bounds policy-only
# path: same with $PATH_1030
```

All window results are probe measurements on CP-SAT models of the emitted ops; the only
kernels built are the references, which are not new results.
