# Ways to Induce the CP-SAT + FMA Validation Failure

The bug: **Greedy + FMA** validates; **CP-SAT + Fma** produces wrong output (value/index mismatch). Unit tests with minimal SSA (1–2 SelectChoice ops, small machine) pass. This doc lists ways to make the failure more likely or reproducible.

## 1. Full kernel, default machine (current repro)

- **Script:** `uv run python scripts/minimal_fma_repro.py` (no `--greedy`, no `--vselect`)
- **Params:** rounds=2, batch_size=16, forest_height=10, `--select-choice fma`, `--valu-shift valu`
- **Why it fails:** Full SSA kernel has many SelectChoice ops, barriers, and mixed VALU/Flow/Load; CP-SAT finds a different schedule than greedy, and something (address overlap, precedence, or lowering order) goes wrong.

## 2. Bisect all choice combinations

- **Script:** `uv run python scripts/bisect_cpsat_validation.py --rounds 2 --batch-size 16`
- **Observation:** Only `select=vselect` passes (2/8). Any `select=fma` or `select=optimize` fails.
- **Use:** Compare passing vs failing to see which encoding correlates with failure.

## 3. Default machine vs small machine (unit tests)

- **Unit tests** now use `SMALL_MACHINE_FMA` (1 VALU slot) so sub and fma are forced into **consecutive cycles**. That avoids packing multiple VALU ops in the same cycle.
- **To induce failure in tests:** Change tests to use `MachineConfig::default()` (6 VALU slots). Then CP-SAT may schedule sub and fma in the same cycle as other VALU ops; if there is a bug tied to same-cycle packing or hint-driven schedules, it may appear.

## 4. Greedy hints + ForceFma (conflicting hints)

- **Idea:** Production runs pass **greedy cycle hints** and **select_choice_hints** (greedy’s FMA set) to CP-SAT. With **ForceFma**, the solver ignores select_choice_hints (all SelectChoice are forced to FMA). Cycle hints still nudge CP-SAT toward the greedy schedule.
- **To induce:** Run CP-SAT **without** greedy hints (e.g. `greedy_hints: None`, `select_choice_hints: None`). If the failure is tied to hint-driven cycle choices, a different schedule might trigger the bug more often or in a different way.
- **Or:** Run with **wrong** select_choice_hints (e.g. empty set while using ForceFma) to ensure no FMA hint is applied and the model is purely constraint-driven.

## 5. More rounds / larger batch (stress)

- **Script:** `uv run python scripts/minimal_fma_repro.py --rounds 4 --batch-size 64` (or 256)
- **Why:** More SelectChoice ops, more pressure on scratch and VALU slots; CP-SAT may find schedules that stress address assignment or liveness (e.g. more overlapping intervals, or reuse at boundaries).

## 6. SelectChoice Optimize (solver chooses FMA vs vselect)

- **Script:** `uv run python scripts/minimal_fma_repro.py` but pass `--select-choice optimize` and ensure the solver actually picks FMA for some ops (e.g. by time limit or seed).
- **Why:** With **Optimize**, the model has choice variables; extraction and lowering must match. A bug in how we extract or apply `select_fma_ops` when the solver mixes FMA and vselect could show up here.

## 7. Barrier elision and section boundaries

- **Idea:** The full pipeline elides barriers **after** greedy and **before** CP-SAT. Indices and op counts change. If any code assumes op index == position in a pre-elision list, or if section boundaries are computed on one view and applied on another, that could cause wrong cycles or wrong `select_fma_ops` mapping.
- **To induce:** Add a test that runs CP-SAT **with** barriers present (no elision) and ForceFma on a tiny kernel that has one barrier and 1–2 SelectChoice ops; compare cycle assignments and lowered output to the elided case.

## 8. Scratch pressure near budget

- **Idea:** If the bug is address reuse under pressure, run with a **reduced scratch budget** (e.g. 1200 or 1000) so that the allocator is forced to reuse more aggressively. Compare greedy vs CP-SAT at the same budget; if only CP-SAT fails, the schedule may be creating an overlapping-live range that greedy avoids.
- **Caveat:** Reducing budget may make the problem infeasible; use small kernels or fewer rounds.

## 9. Deterministic seed and time limit

- **Script:** `uv run python scripts/minimal_fma_repro.py --seed 42 --cpsat-time-limit 5`
- **Why:** Short time limit may yield a different (worse or more “edge-case”) schedule; fixed seed makes the failure reproducible for debugging.

## 10. Rust integration test (full kernel + ForceFma)

- **Location:** `scheduler_rs` – build the real SSA kernel (e.g. rounds=2, batch=16), run CP-SAT with ForceFma and default machine, then run address assignment + overlap check (no Python validation). If the overlap check fails, the bug is in schedule/assignment; if it passes but Python validation still fails, the bug is in lowering or machine execution.
- **Optional:** Mark test as `#[ignore]` or behind a feature and run only in CI or when debugging.

## Bisect minimal failing config (batch, rounds, forest_height)

**Script:** `scripts/bisect_minimal_fma.py`

- **Machine config (VALU/ALU slots) is NOT exposed via the CLI.** The Rust builder uses `MachineConfig::default()`. To sweep machine size you’d need to add parameters to the Python binding and `generate_ssa_kernel.py`.
- The script sweeps **rounds**, **batch_size**, and **forest_height** and reports which combos pass/fail with CP-SAT + fma.

```bash
uv run python scripts/bisect_minimal_fma.py --rounds 2 --batch-size 16 --forest-height 10
uv run python scripts/bisect_minimal_fma.py --rounds 1 2 --batch-size 8 16 --forest-height 9 10
uv run python scripts/bisect_minimal_fma.py --shrink
```

Observed: In a small grid (rounds 1–2, batch 8–16, height 9–10), all CP-SAT + fma runs failed; smallest in that grid is **rounds=1, batch_size=8, forest_height=9**.

## Summary

| Method                         | What it stresses                          |
|--------------------------------|-------------------------------------------|
| Full kernel, default machine   | Realistic schedule + many SelectChoice   |
| Bisect choices                 | Which encoding (fma/vselect/optimize) fails |
| Default machine in unit tests  | Same-cycle VALU packing                   |
| No greedy hints                | Pure CP-SAT schedule                      |
| More rounds/batch              | Scratch and VALU pressure                  |
| SelectChoice Optimize          | Mixed FMA/vselect extraction               |
| Barriers / no elision          | Index and section mapping                 |
| Reduced scratch budget         | Aggressive address reuse                  |
| Seed + short time limit        | Reproducible, possibly bad schedules      |
| Rust full-kernel + overlap check | Pinpoint schedule vs lowering vs execution |
| **bisect_minimal_fma.py**        | Minimal (rounds, batch, forest_height) failing |
| **Machine config**               | Not in CLI; add Rust + CLI params to sweep VALU/ALU slots |
