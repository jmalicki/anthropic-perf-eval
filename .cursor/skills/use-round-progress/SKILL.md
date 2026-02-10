---
name: use-round-progress
description: Run kernel generation with validation and update the round progress tracking table. Use when benchmarking different round counts, batch sizes, or configurations to track performance over time.
---

# Running Rounds with Validation and Tracking Progress

When running kernel generation for benchmarking or progress tracking, always:
1. **Run with validation** (`--validate`) to ensure correctness
2. **Save both kernel and scheduler JSON** (`--save-kernel` and `--output-json`)
3. **Update [docs/round-progress.md](docs/round-progress.md)** with the results

---

## Running a validated kernel generation

### Basic command

```bash
# Generate kernel with validation, save both kernel and scheduler JSON
uv run generate_ssa_kernel.py --rounds 3 --batch-size 256 \
  --cpsat-time-limit 60 -q \
  --validate \
  --save-kernel kernel_3r_256b.json \
  --output-json scheduler_output_3r_256b.json
```

### Options

- `--rounds N` — Number of rounds (required)
- `--batch-size N` — Batch size, must be multiple of 8 (default: 256)
- `--cpsat-time-limit SECS` — CP-SAT time limit (default: 3600)
- `--validate` — Validate against reference kernel (required for tracking)
- `--save-kernel PATH` — Save kernel JSON (default: `kernel_to_validate.json`)
- `--output-json PATH` — Save scheduler output JSON (default: `scheduler_output.json`)
- `--liveness-detail LEVEL` — Liveness detail: `none`, `summary`, or `full` (default: `full`)

### What gets recorded

The scheduler output JSON includes:
- `git_sha` — Current git commit SHA
- `total_cycles` — Final cycle count
- `peak_live` — Peak scratch usage (words)
- `run_options.rounds` — Number of rounds
- `run_options.batch_size` — Batch size
- `lb_cycles` — Proven lower bound (CP-SAT only). When `lb_cycles == total_cycles`, status is OPTIMAL; otherwise FEASIBLE
- File modification date (for tracking when run occurred)

**Solver status:** The JSON doesn't include an explicit `solver_status` field, but you can infer it:
- **OPTIMAL:** `lb_cycles == total_cycles` (proven optimal)
- **FEASIBLE:** `lb_cycles < total_cycles` or `lb_cycles` is null (found a solution but not proven optimal)
- **Greedy:** No `lb_cycles` field (greedy scheduler doesn't prove optimality)

Alternatively, check the command output for "status: OPTIMAL" or "status: FEASIBLE" in the CP-SAT solver summary.

---

## Updating the progress table

After a validated run, add an entry to [docs/round-progress.md](docs/round-progress.md):

### Extract data from JSON

```bash
# Get SHA, cycles, batch size, peak live, solver status from scheduler JSON
jq -r '.git_sha, .total_cycles, .run_options.batch_size, .peak_live, .lb_cycles' scheduler_output_3r_256b.json

# Determine solver status: if lb_cycles == total_cycles, OPTIMAL; else FEASIBLE
jq -r 'if .lb_cycles == .total_cycles then "OPTIMAL" elif .lb_cycles then "FEASIBLE" else "GREEDY" end' scheduler_output_3r_256b.json

# Get file date
stat -f "%Sm" -t "%Y-%m-%d" scheduler_output_3r_256b.json
```

### Add entry to table

1. Find the appropriate section (2 rounds, 3 rounds, 4 rounds, 5 rounds, or "Unknown rounds")
2. Add a new row with:
   - **Date** — File modification date (`YYYY-MM-DD`)
   - **SHA** — Git commit SHA (8 chars) from JSON or `git rev-parse HEAD`
   - **Batch** — Batch size from `run_options.batch_size`
   - **Command** — Exact command used (if notable config)
   - **Cycles** — `total_cycles` from JSON
   - **Peak live** — `peak_live` from JSON (or "N/A")
   - **Status** — Solver status: "OPTIMAL" (if `lb_cycles == total_cycles`), "FEASIBLE" (if `lb_cycles < total_cycles`), or "GREEDY" (if no `lb_cycles`)
   - **Validation** — "PASSED" if validation succeeded
   - **Notes** — Any notable options or changes (e.g. "trait-based liveness refactor")
   - **Source** — JSON filename

3. **Sort entries** by date (newest first) within each round count, then by batch size

### Example entry

```markdown
| 2026-02-10 | 1864ae7 | 256 | `uv run generate_ssa_kernel.py --rounds 3 --cpsat-time-limit 60 -q --validate` | 309 | 1269 | OPTIMAL | PASSED | Trait-based liveness refactor | scheduler_output.json |
```

Note: The table includes a **Status** column (OPTIMAL/FEASIBLE/GREEDY) to track solver quality.

---

## Important notes

- **Batch size matters:** Cycles scale with batch size. Always compare runs with the same batch size.
- **Validation required:** Only add entries for runs that passed validation (`--validate` succeeded).
- **Same SHA, different configs:** If multiple runs share the same SHA but differ in options (e.g. tree-reduce depth, tightening), add separate entries with notes explaining the difference.
- **File naming:** Use descriptive names like `kernel_3r_256b.json` or `scheduler_output_4r_64b_depth3.json` to avoid confusion.

---

## Quick reference

```bash
# Standard 3-round run (batch 256)
uv run generate_ssa_kernel.py --rounds 3 --batch-size 256 \
  --cpsat-time-limit 60 -q --validate \
  --save-kernel kernel_3r_256b.json \
  --output-json scheduler_output_3r_256b.json

# 2-round run (batch 64, quick dev)
uv run generate_ssa_kernel.py --rounds 2 --batch-size 64 \
  --cpsat-time-limit 30 -q --validate \
  --save-kernel kernel_2r_64b.json \
  --output-json scheduler_output_2r_64b.json

# 4-round run (batch 256, longer optimization)
uv run generate_ssa_kernel.py --rounds 4 --batch-size 256 \
  --cpsat-time-limit 1800 -q --validate \
  --save-kernel kernel_4r_256b.json \
  --output-json scheduler_output_4r_256b.json
```
