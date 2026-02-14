---
name: use-compare-kernel-scratch
description: Compare scratch state cycle-by-cycle between two kernels on the same input to find the first cycle/address where they differ. Use when debugging validation failure between two kernels (e.g. zero-gap vs default, or passing vs failing) to pinpoint where scratch diverges.
---

# Compare Kernel Scratch Cycle-by-Cycle

**Script:** `scripts/compare_kernel_scratch_cycle.py`

Runs two kernels with the **same input** (same mem image, seed) and records end-of-cycle scratch snapshots. Reports the **first cycle and address** where scratch differs, plus a small window of addresses. Use when you have a **passing** and **failing** kernel (e.g. default vs zero-gap) and want to see exactly where execution state diverges.

## When to Use

- Two kernels (e.g. `kernel_2r_16b.json` vs `kernel_2r_16b_zero_gap.json`) that should be equivalent but one fails validation.
- You want the **first** (cycle, address) where scratch differs—before diving into scheduler JSON or find_wrong_scratch_use.

## Prerequisites

- Two kernel JSON files (e.g. from `--save-kernel` or `kernel_to_validate.json`).
- Same `--rounds` and `--batch-size` as the runs that produced the kernels.

## How to Run

```bash
# Compare passing vs failing kernel (match config that produced them)
uv run python scripts/compare_kernel_scratch_cycle.py artifacts/kernel_2r_16b.json artifacts/kernel_2r_16b_zero_gap.json --rounds 2 --batch-size 16

# Custom seed / scratch window / max cycles
uv run python scripts/compare_kernel_scratch_cycle.py kernel_a.json kernel_b.json --rounds 4 --batch-size 32 --seed 456 --scratch-len 256 --max-cycles 400
```

## Output

- First line: `First scratch difference at cycle C, address A: a=<val> b=<val>`.
- Then `s[lo..hi]` for both kernels at that cycle so you can see the divergence window.

Use the reported **(cycle, address)** with `scripts/find_wrong_scratch_use.py` (with `--addrs A`) or `scripts/inspect_cycle_addr.py` on the scheduler JSON to find the incorrect use or allocation.
