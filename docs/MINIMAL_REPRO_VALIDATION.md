# Minimal reproducible test case for validation failure (non-16-round)

When running the SSA kernel with **rounds < 16** and **CP-SAT** (not greedy), the generated kernel can **fail validation**: output does not match the reference. With **greedy** scheduling the same configs pass. This doc records the minimum failing config for debugging.

## Minimum failing problem (CP-SAT)

- **rounds=2, batch_size=16**, **use CP-SAT** (do not use `--use-greedy`)
- Build: ~0.04s with short CP-SAT limit, 7 cycles
- Validation: **FAIL** (value/index mismatch; often first at batch index 1)
- **Same config with greedy: PASS**

So the smallest problem that fails is **2 rounds, batch 16, CP-SAT**. Larger batches (32, 64, 128, 256) also fail with CP-SAT; greedy passes for all tested sizes.

```bash
# Fails (CP-SAT)
uv run generate_ssa_kernel.py --rounds 2 --batch-size 16 --cpsat-time-limit 30 --validate -q

# Passes (greedy)
uv run generate_ssa_kernel.py --rounds 2 --batch-size 16 --use-greedy --validate -q
```

## Configs that pass (after fixes)

- **1 round, batch 8** (n_groups=1): passes with both greedy and CP-SAT (fixed via n_iters ceiling and n_groups bounds).
- **2 rounds, batch 16** (and 32, 64, 128, 256): passes with **greedy** only; fails with CP-SAT.

## Notes

- The failure is tied to **CP-SAT’s schedule or choices** (ConstChoice / SelectChoice / shift), not to problem size alone: greedy produces a correct kernel for the same size.
- **batch_size=8** (n_groups=1) used to panic; fixed by single-group load/store and `n_iters = ceil(n_groups/8)` with `g < n_groups` guards in xor/hash/depth.
