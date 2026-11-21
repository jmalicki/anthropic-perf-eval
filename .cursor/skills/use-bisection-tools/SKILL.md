---
name: use-bisection-tools
description: Use bisection scripts to find which choice options (SelectChoice, ConstChoice, valu shift), which constraint groups, or which problem size (rounds/batch/height) cause validation failures. Use when debugging CP-SAT validation failures, isolating scheduler bugs, or minimizing repro configs.
---

# Using the Bisection Tools

Bisection can sweep **choice options**, **constraint groups**, and **problem size**. The main entrypoint is **scripts/bisect.py**; **bisect_cpsat_validation.py** and **bisect_minimal_fma.py** are backward-compatible wrappers.

Run all scripts from **project root**. Ensure **scheduler_rs** is built (`./scripts/build_scheduler_rs.sh`). **batch_size** must be a multiple of 8 (VLEN).

---

## 1. Unified bisection (any axis)

**Script:** `scripts/bisect.py`

Bisect across any combination of:

- **Choice:** const_choice, select_choice, valu_shift (instruction encoding)
- **Constraints:** which optional CP-SAT constraint groups to disable
- **Size:** rounds, batch_size, forest_height

### Sweep choice (which encoding causes failure?)

```bash
# Default 8 combinations (same as bisect_cpsat_validation): const ∈ {optimize, load}, select ∈ {optimize, vselect}, valu ∈ {alu, valu}
uv run python scripts/bisect.py --sweep-choice

# Include fma / add_imm in the grid
uv run python scripts/bisect.py --sweep-choice --const-choice optimize load add_imm --select-choice optimize fma vselect --valu-shift alu valu

# With custom rounds/batch/time
uv run python scripts/bisect.py --sweep-choice --rounds 2 --batch-size 256 --cpsat-time-limit 90
```

### Sweep constraint groups (which disabled group correlates with failure? or infeasibility?)

Use this to find which optional constraint group makes the model **infeasible**: run with `single` (baseline + one run per group disabled). The script reports **solver status** (INFEASIBLE, FEASIBLE, OPTIMAL, UNKNOWN) for each run and, when the baseline is infeasible, which disabled groups made the model feasible.

```bash
# Single run with no disabled groups (baseline)
uv run python scripts/bisect.py --sweep-constraints none

# None + one run per group with that group disabled (1 + 5 = 6 runs); reports solver status for infeasibility debugging
uv run python scripts/bisect.py --sweep-constraints single

# One run with all five groups disabled
uv run python scripts/bisect.py --sweep-constraints all

# Explicit disable sets
uv run python scripts/bisect.py --sweep-constraints capacity_valu capacity_alu "capacity_valu,capacity_alu"
```

Valid constraint group names: `redundant_linear_bounds`, `capacity_valu`, `capacity_alu`, `capacity_flow`, `capacity_load`, `conditional_makespan`, `flow_cumulative` (from `generate_ssa_kernel.py --help` or `scheduler_rs.optional_constraint_group_names()`).

### Sweep size (smallest config that fails?)

```bash
# Grid: rounds × batch_size × forest_height (default single config if not --sweep-size)
uv run python scripts/bisect.py --sweep-size --rounds 1 2 --batch-size 8 16 32 --forest-height 8 9 10 --select-choice fma --valu-shift valu

# Shrink from reference (2, 16, 10): vary one dimension at a time
uv run python scripts/bisect.py --shrink --select-choice fma --valu-shift valu
```

### Fixed overrides (when not sweeping that axis)

- **Choice:** `--const-choice C --select-choice S --valu-shift V` (single value each; default optimize).
- **Constraints:** `--disable-cpsat-constraints G1,G2` for a single fixed disable set.
- **Size:** `--rounds R --batch-size B --forest-height H` (single value when not using `--sweep-size`).

### Other options

- `--use-greedy` — Run greedy only (one run, no CP-SAT).
- `--cpsat-time-limit SEC`, `--seed N` — Per-run settings.

---

## 2. Which choice options cause issues? (wrapper)

**Script:** `scripts/bisect_cpsat_validation.py`

Thin wrapper around `bisect.py --sweep-choice`. Runs **greedy (baseline)** then **8 CP-SAT runs** (const ∈ {optimize, load}, select ∈ {optimize, vselect}, valu ∈ {alu, valu}).

### Usage

```bash
# Default: rounds=2, batch_size=16, CP-SAT 60s
uv run python scripts/bisect_cpsat_validation.py

# Smaller/faster or larger
uv run python scripts/bisect_cpsat_validation.py --rounds 2 --batch-size 8 --cpsat-time-limit 30
uv run python scripts/bisect_cpsat_validation.py --rounds 2 --batch-size 256 --cpsat-time-limit 90
```

### Interpreting results

- **Greedy fails** — Environment or baseline broken.
- **All 8 CP-SAT fail** — Bug likely in core CP-SAT schedule.
- **Some pass, some fail** — Compare rows to see which choice(s) correlate with failure.

**Choice meanings:** ConstChoice `optimize` | `load` | `add_imm`; SelectChoice `optimize` | `fma` | `vselect`; valu shift `optimize` | `alu` | `valu`. To test a single combo by hand: `generate_ssa_kernel.py --const-choice C --select-choice S --valu-shift V --validate`.

---

## 3. Smallest batch size and rounds that reproduce (wrapper)

**Script:** `scripts/bisect_minimal_fma.py`

Uses shared `bisect_common.run_one` with **SelectChoice = fma**, **valu_shift = valu**. Sweeps **rounds**, **batch_size**, **forest_height** (grid or `--shrink`). Same behavior as before; for full axis options use `scripts/bisect.py --sweep-size --select-choice fma --valu-shift valu`.

### Usage

```bash
# Grid sweep: rounds [1,2], batch [8,16,32], forest_height [8,9,10]
uv run python scripts/bisect_minimal_fma.py

# Single combo
uv run python scripts/bisect_minimal_fma.py --rounds 2 --batch-size 16 --forest-height 10

# Custom sweep
uv run python scripts/bisect_minimal_fma.py --rounds 1 2 --batch-size 8 16 --forest-height 9 10

# Shrink from (2, 16, 10)
uv run python scripts/bisect_minimal_fma.py --shrink
```

**Options:** `--rounds`, `--batch-size`, `--forest-height` (one or more values), `--cpsat-time-limit`, `--seed`, `--shrink`.

---

## 4. Running a minimal repro after bisection

Once you have a minimal (rounds, batch_size, forest_height) or a known bad choice/constraint combo:

- **Minimal FMA repro (one config):**  
  `uv run python scripts/minimal_fma_repro.py --rounds 2 --batch-size 16`  
  Options: `--greedy` (expect pass), `--vselect` (CP-SAT + vselect), `--save path`, `--no-validate`.

- **Single config with specific constraints disabled:**  
  `uv run python scripts/bisect.py --rounds 2 --batch-size 16 --disable-cpsat-constraints capacity_valu`  
  (no sweep; one run).

- **2-round SelectChoice FMA repro:**  
  `uv run python scripts/bisect_select_fma_2round.py`  
  Runs rounds=2, batch=8, select=fma, valu_shift=valu.

---

## Workflow summary

| Goal | Script | Typical use |
|------|--------|-------------|
| Bisect any axis (choice / constraints / size) | bisect.py | `--sweep-choice`, `--sweep-constraints single`, `--sweep-size`, `--shrink`. |
| Which choice (Const/Select/valu) causes failure? | bisect_cpsat_validation.py | Wrapper: run with default or pass --rounds/--batch-size/--cpsat-time-limit. |
| Smallest rounds/batch/height that fails? | bisect_minimal_fma.py | Sweep or `--shrink`; use reported smallest failing. |
| Run one minimal failing config | minimal_fma_repro.py | After bisect_minimal_fma; add --greedy or --vselect as needed. |
| Debug 2-round SelectChoice FMA | bisect_select_fma_2round.py | Fixed minimal config for that bug. |

All scripts invoke `generate_ssa_kernel.py` with `--validate` (unless `--no-validate` where supported). Validation compares kernel output to the reference in `problem.py`.
