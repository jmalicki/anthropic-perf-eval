---
name: use-ssa-equivalence
description: Prove that a transformed SSA program (an emission option, a combination, an emission plan) writes the same final memory as the canonical SSA program for every valid input, or get the first output word where they differ with a concrete input. Use after adding or changing an emission rewrite, before trusting a new emission plan, or when kernel_dataflow proves but numeric validation fails.
---

# SSA equivalence

**Module:** `verification/ssa_equivalence.py`. Inputs are two scheduler JSONs (`--output-json`). `kernel_dataflow` proves kernel = its SSA program; this proves SSA program A = SSA program B. Together: kernel_A = SSA_A = SSA_B = kernel_B.

## When to use

- After adding or changing an emission rewrite (`--merge-hash-stages`, `--index-form`, `--cached-select`, `--select-bit-source`, `--prune-dead-index`, `--tree-reduce-max-depth`, `--emission-plan`): prove the new emission against the default one.
- Before relying on a new emission plan from `scripts/emission_structure.py`.
- When `kernel_dataflow` says PROVED but `--validate` fails: the SSA program itself is wrong. Compare it with a default build to get the first differing word and an input that shows it.

## How to run

```bash
uv run generate_ssa_kernel.py --rounds 16 --batch-size 32 --use-greedy -q --output-json /tmp/a.json
uv run generate_ssa_kernel.py --rounds 16 --batch-size 32 --use-greedy -q --output-json /tmp/b.json \
    --merge-hash-stages --index-form mirrored --cached-select bits
uv run python -m verification.ssa_equivalence /tmp/a.json /tmp/b.json

# A --values-only build stores no indices: compare the values only
uv run python -m verification.ssa_equivalence /tmp/a.json /tmp/vo.json --values-only

# Save the counterexample input (initial memory image) for problem.Machine
uv run python -m verification.ssa_equivalence /tmp/a.json /tmp/bad.json --counterexample-out /tmp/cex.json
```

JSONs written by the binding directly (no `run_options`) need `--shape ROUNDS BATCH HEIGHT`; a JSON that records its shape must match `--shape` and the other JSON. Exit status: 0 PROVED, 1 COUNTEREXAMPLE, 2 NOT PROVED, 3 input error (not single assignment, undefined VReg read (a VReg defined only by a dead op included), a dest-less op that is not a store, non-constant store address, a word stored twice, nothing to compare, shapes disagree) or any checker error.

## Reading the output

| Verdict | Meaning |
|---|---|
| `PROVED` | For every input satisfying the preconditions: no op of either program traps, and every compared word has the same final value |
| `COUNTEREXAMPLE` | The lowest-address word that differs, with both values, on an input confirmed by a plain concrete SSA interpreter. Run both kernels on that image to reproduce |
| `NOT PROVED` | No sample input separates the programs, but some word was not proved (Z3 sat on the abstraction, unknown/timeout, a query over the node budget), a load the memory model cannot order, or an op that may trap. The reasons are listed. Never a pass |

Compared words: every word either program stores (a word not stored keeps its initial value), or with `--values-only` the output values only (no other word, indices included).

## What is assumed

- **Preconditions** (`problem.build_mem_image`): header words fixed by the shape, every input index 0, tree and input values in [0, 2^30), extra room 0.
- **SSA semantics**: ops are pure with machine semantics modulo 2^32; loads read the initial memory; stores write the final memory (same model as `kernel_dataflow`). Checked, not assumed: a constant-address load of a stored word must feed that store; other loads must have an address interval inside memory and away from stored words.
- **No traps** (checked): every op of both programs, including value-dead ones still in the program, must provably not trap: load addresses inside memory and divisors (`//`, `%`, `cdiv`) nonzero, by intervals (R25), and constant store addresses inside memory. Otherwise NOT PROVED with the op named.
- **Well-formed input** (checked, else input error): single assignment over the live ops (two constants of one immediate may share a VReg); dead ops (cycle < 0) are not part of the program; only the reserved zero VReg `v0` may be read without a definition; every dest-less op is a store; store addresses are constants; no word is stored twice; at least one word is compared.

## How it works (for when it does not prove)

1. Both programs are symbolically executed into one term store whose constructors normalise by proved identities: linear forms (R21), XOR sets (R22), equality as a difference (R23), 0/1 multiply_adds and products as selects (R4, R5, R24, with 0/1-ness from intervals R25), and the preconditions. Merged hash stages, mirrored index, pruned index work and the stage-5 deferral normalise to identical terms.
2. Sweeping: terms of A and B with equal values on 32 random valid inputs are paired (no emitter metadata needed); each pair is proved by Z3 with shared subterms cut to interval-constrained variables, case-splitting on 0/1 select conditions (Mobius sums are slow for Z3 otherwise). Proven pairs are merged and later terms rebuilt over them, so queries stay small. Alpha-equivalent queries are answered once (thousands of lanes, about ten distinct queries).
3. A new rewrite that does not prove: look at the `NOT PROVED` terms. Either add a normal form (with a Z3 proof in `verification/rewrites.py`, registered in `REWRITE_CHECKS`, cited in the constructor) or check that the residual query is small enough for Z3 (`--timeout-ms`, `--node-budget`).

## Measured (greedy builds, height 10)

All of: merge-hash, mirrored, prune-dead-index, bits, bits-flow, mobius, bits + path bits, values-only, merge+mirrored+bits, merge+mirrored+mobius, merge+prune+bits-flow, merge+mirrored+bits+depth 4, and the four plans in `docs/emission_plans/`, prove against the default emission: 16 rounds batch 8 or 32 in about 1 s each, 2 rounds in under 0.1 s, 16 rounds batch 256 in 2 to 5 s (about 450 MB).

## Tests

`uv run --extra dev pytest tests/test_ssa_equivalence.py`: each emission option and a combination prove; a wrong hash constant, swapped select arms (default and bits emission) and an off-by-one load address, applied to the SSA program and the kernel alike, are rejected, and the counterexample input makes the two kernels write different words on `problem.Machine`. Small synthetic programs on which every sample agrees check that Z3, not sampling, decides: differences on rare values are NOT PROVED, a difference only outside the precondition is PROVED. Also: Z3 unknown is not a pass; case splitting; the memo keeps queries with different ranges apart; value-dead traps; undefined VRegs, double stores, non-constant store addresses and mismatched shapes are errors; the memory-order check rejects an unordered load.
