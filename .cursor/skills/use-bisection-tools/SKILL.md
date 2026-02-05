---
name: use-bisection-tools
description: Use bisection scripts to find which choice options (SelectChoice, ConstChoice, valu shift) cause validation failures and to find the smallest batch size and rounds that reproduce bugs. Use when debugging CP-SAT validation failures, isolating scheduler bugs, or minimizing repro configs.
---

# Using the Bisection Tools

Two bisection scripts help debug CP-SAT validation and minimize repro configs:

1. **bisect_cpsat_validation.py** — Which **choice options** (ConstChoice, SelectChoice, valu shift) correlate with failure?
2. **bisect_minimal_fma.py** — What is the **smallest** (rounds, batch_size, forest_height) that still fails?

Run both from **project root**. Ensure **scheduler_rs** is built (`./scripts/build_scheduler_rs.sh`). **batch_size** must be a multiple of 8 (VLEN).

---

## 1. Which choice options cause issues?

**Script:** `scripts/bisect_cpsat_validation.py`

Runs **greedy (baseline)** then **8 CP-SAT runs**, one per combination of:

- **ConstChoice:** `optimize` | `load` (constant materialization: derive via ALU vs load)
- **SelectChoice:** `optimize` | `vselect` (tree traversal: FMA vs vselect)
- **valu shift:** `alu` | `valu` (where shifted VALU ops go)

Uses forced values (`alu`/`valu`) for valu shift (not `optimize`) so the failing combination is clear. Each run uses `--validate -q`; pass = validation matches reference.

### Usage

```bash
# Default: rounds=2, batch_size=16, CP-SAT 60s
uv run python scripts/bisect_cpsat_validation.py

# Smaller/faster for quick check
uv run python scripts/bisect_cpsat_validation.py --rounds 2 --batch-size 8 --cpsat-time-limit 30

# Larger config (more coverage, slower)
uv run python scripts/bisect_cpsat_validation.py --rounds 2 --batch-size 256 --cpsat-time-limit 90
```

### Interpreting results

- **Greedy fails** — Environment or baseline broken; fix before trusting CP-SAT bisection.
- **All 8 CP-SAT fail** — Bug likely in **core** CP-SAT schedule (cycle assignment, precedence), not in a single choice.
- **Some pass, some fail** — Compare passing vs failing rows. The **difference in flags** (const/select/shift) indicates which choice(s) correlate with failure (e.g. only `select=fma` fails → SelectChoice FMA path).

**Choice meanings (for follow-up):**

- **ConstChoice:** `optimize` (solver picks), `load` (force load), `add_imm` (force add_imm; not swept by this script).
- **SelectChoice:** `optimize` (solver picks), `fma` (force FMA), `vselect` (force vselect).
- **valu shift:** `optimize` (solver picks), `alu` (force ALU), `valu` (force VALU).

To test a single combo by hand, use `generate_ssa_kernel.py` with `--const-choice`, `--select-choice`, `--valu-shift` and `--validate`.

---

## 2. Smallest batch size and rounds that reproduce

**Script:** `scripts/bisect_minimal_fma.py`

Fixes **SelectChoice = fma** and sweeps **rounds**, **batch_size**, **forest_height**. Finds the **smallest config** where CP-SAT + fma still **fails** validation (minimal repro). Uses greedy + fma as baseline (expected to pass).

### Usage

```bash
# Grid sweep: rounds in [1,2], batch in [8,16,32], forest_height in [8,9,10]
uv run python scripts/bisect_minimal_fma.py

# Single combo (quick check)
uv run python scripts/bisect_minimal_fma.py --rounds 2 --batch-size 16 --forest-height 10

# Custom sweep (multiple values per option)
uv run python scripts/bisect_minimal_fma.py --rounds 1 2 --batch-size 8 16 --forest-height 9 10

# Shrink mode: try smaller configs from (2, 16, 10); report pass/fail per dimension
uv run python scripts/bisect_minimal_fma.py --shrink
```

**Options:**

- `--rounds R [R ...]` — Rounds to try (default: 1 2).
- `--batch-size B [B ...]` — Batch sizes (default: 8 16 32); must be multiple of 8.
- `--forest-height H [H ...]` — Forest heights (default: 8 9 10).
- `--cpsat-time-limit SEC` — CP-SAT time per run (default: 90).
- `--seed N` — Random seed (default: 123).
- `--shrink` — From (rounds=2, batch=16, height=10), vary one dimension at a time and print pass/fail; suggests minimal repro.

### Interpreting results

- **Summary** lists passing vs failing configs and reports **smallest failing (lex)** by (rounds, batch_size, forest_height).
- Use that smallest failing triple for **minimal repro** in other scripts or when filing bugs.

**Note:** Machine config (VALU/ALU slots) is not swept; the script uses the default machine.

---

## 3. Running a minimal repro after bisection

Once you have a minimal (rounds, batch_size, forest_height) or a known bad choice combo:

- **Minimal FMA repro (one config):**  
  `uv run python scripts/minimal_fma_repro.py --rounds 2 --batch-size 16`  
  Options: `--greedy` (expect pass), `--vselect` (CP-SAT + vselect), `--save path`, `--no-validate`.

- **2-round SelectChoice FMA repro:**  
  `uv run python scripts/bisect_select_fma_2round.py`  
  Runs rounds=2, batch=8, select=fma, valu_shift=valu; use for debugging the SelectChoice FMA path.

---

## Workflow summary

| Goal | Script | Typical use |
|------|--------|-------------|
| Which choice (Const/Select/valu) causes failure? | bisect_cpsat_validation.py | Run with default or small batch; compare 8 rows. |
| Smallest rounds/batch/height that fails? | bisect_minimal_fma.py | Sweep or `--shrink`; use reported smallest failing. |
| Run one minimal failing config | minimal_fma_repro.py | After bisect_minimal_fma; add --greedy or --vselect as needed. |
| Debug 2-round SelectChoice FMA | bisect_select_fma_2round.py | Fixed minimal config for that bug. |

All scripts invoke `generate_ssa_kernel.py` with `--validate` (unless `--no-validate` where supported). Validation compares kernel output to the reference in `problem.py`.
