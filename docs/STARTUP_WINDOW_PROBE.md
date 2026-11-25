# Start-up window probe on the best kernels (in progress)

Does any schedule of the same ops do better in the start-up (cycles 0 to ~110) than the greedy?
Measured with the window optimality probe (docs/WINDOW_OPTIMALITY_PROBE.md): every op outside
a window pinned at its warm-start cycles, the ops after the window moving as one block,
encodings (shift, select, const) free inside the window, deterministic-time budgets.

## Warm start: what the probe can and cannot take

- The probe takes its reference from the CP-SAT pipeline's greedy, which differs from a
  greedy-only build: it runs with full bounds (ASAP deferral, capacity deferral). For
  `path_flow90.json` with `chain-first` + `native-first` it gives **1,032** cycles, not the
  greedy-only 1,030. Savings below are against 1,032; absolute makespans are compared with
  1,030. Split shifts are kept (`--cpsat-split-shift-hint`, lane cycles in the reference);
  the greedy const policy is not available there (the CP-SAT pipeline keeps the default).
- The CP-SAT pipeline could not carry this kernel at all as it stood: the per-section budget
  check counts liveness on the cycle timeline (1,597 > 1,535 for this schedule, whose exact
  live peak is 1,483) and starts a no-shift feasibility repair of the whole kernel that finds
  nothing in 600 s (without shifts the vector ALU alone needs 1,220 cycles).
  `CPSAT_SECTION_CHECK_TIMELINE=model` (analysis only) makes the check use the model's own
  timeline; the section then fits (1,533), and the probe's all-fixed check proves the
  reference is a solution of the full model (optimal, makespan 1,032, 1.7 s).

## Path kernel, first sweep (deterministic time 60 per window, 4 workers)

| Window | Status | Saved vs 1,032 | Bound | Ops inside | Encoding changes |
|---|---|---|---|---|---|
| [0, 110) | feasible (budget) | 0 | 8 | 1,203 | - |
| [55, 165) | optimal | 0 | 0 | 1,050 | - |
| [0, 60) | feasible (budget) | 0 | 1 | 739 | - |
| [30, 90) | optimal | 1 | 1 | 645 | 14 shift, 1 select |
| [60, 120) | optimal | 1 | 1 | 563 | 73 shift, 11 select |

Report: `docs/window_probe/path_startup_sweep.json`.
