# Minimal reproducible test case for validation failure (non-16-round)

When running the SSA kernel with **rounds < 16** (e.g. 1, 2, or 3), the generated kernel **builds and schedules** but **fails validation**: output does not match the reference kernel. This doc records the smallest configs that reproduce the failure for debugging.

## Minimal (fastest) repro

- **rounds=2, batch_size=16**
- Build: ~0.1s, 7 cycles
- Validation: **FAIL** (value/index mismatch vs reference)

```bash
uv run generate_ssa_kernel.py --rounds 2 --batch-size 16 --cpsat-time-limit 120 --validate --save-kernel kernel_2r_16b.json
```

## Even smaller (1 round, 1 group)

- **rounds=1, batch_size=8** (n_groups=1)
- Requires the `n_groups=1` fix in `ssa_emit.rs` (emit_load_data / emit_store_data) so the single-group case doesn’t panic.
- Build: ~0.03s, 7 cycles
- Validation: **FAIL**

```bash
uv run generate_ssa_kernel.py --rounds 1 --batch-size 8 --cpsat-time-limit 60 --validate --save-kernel kernel_1r_8b.json
```

## Notes

- **batch_size=8** used to panic (index out of bounds in `ssa_emit.rs`) because load/store assumed at least two groups (pairs). The fix handles `n_groups == 1` (and odd group count) by only emitting the second load/store when `g + 1 < n_groups`.
- For **batch_size < 64**, `n_iters = n_groups / 8` is 0 when `n_groups < 8`, so the hash/xor phases may emit no work for small batches; that may be part of the wrong-output cause and would need a separate fix (e.g. partial iters or `n_iters = max(1, n_groups/8)` with bounds checks).
