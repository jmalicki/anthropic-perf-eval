# Minimal problem size for CP-SAT validation failure

## Summary

- **Minimal failing case (wrong output):** `rounds=2`, `forest_height=10`, `batch_size=208` (n_groups=26).
- **Largest passing case:** `rounds=2`, `forest_height=10`, `batch_size=192` (n_groups=24).
- Failure shows as value/index mismatch; first bad index is often 192 (e.g. batch_size=208, 224).

## Constraints for valid runs

1. **batch_size must be ≥ 64** so that `n_groups = batch_size/8 ≥ 8` and `n_iters = n_groups/8 ≥ 1`. Otherwise no hash/branch iterations run and output is wrong (and with batch_size=8 the builder panics on `all_idx[g+1]`).

2. **n_groups must be even** because load/store data loops use `(0..n_groups).step_by(2)` and index `g+1`. So **batch_size must be a multiple of 16** (e.g. 64, 80, …, 192, 208, …). If not (e.g. batch_size=200 → n_groups=25), the builder panics: `index out of bounds: the len is 25 but the index is 25` in `ssa_emit.rs` at `all_idx[g + 1]`.

## Test matrix (2 rounds, forest_height=10, CP-SAT 90s)

| batch_size | n_groups | n_iters | Result        |
|------------|----------|---------|---------------|
| 64         | 8        | 1       | PASS          |
| 128        | 16       | 2       | PASS          |
| 192        | 24       | 3       | PASS          |
| 200        | 25       | 3       | **PANIC** (n_groups odd) |
| 208        | 26       | 3       | FAIL (first diff at index 192) |
| 224        | 28       | 3       | FAIL (first diff at index 192) |
| 240        | 30       | 3       | FAIL          |
| 256        | 32       | 4       | FAIL (first diff at index 152) |

Greedy passes for all valid (even n_groups, batch≥64) sizes tested. The wrong-output bug is specific to the CP-SAT path and appears when n_groups ≥ 26 (batch_size ≥ 208).

## Reproduce minimal failure

```bash
# Minimal failing (wrong output)
uv run generate_ssa_kernel.py --rounds 2 --forest-height 10 --n-nodes 2047 --batch-size 208 --cpsat-time-limit 90 --validate

# Largest passing
uv run generate_ssa_kernel.py --rounds 2 --forest-height 10 --n-nodes 2047 --batch-size 192 --cpsat-time-limit 90 --validate
```

## Panics to avoid when reducing problem size

- **batch_size=8** (n_groups=1): panic at `all_idx[g+1]` (need n_groups ≥ 2 for step_by(2)).
- **batch_size=200** (n_groups=25): panic at `all_idx[25]` (need n_groups even).
- **batch_size=16,32** (n_iters=0): no panic but wrong output (no hash/branch work done).
