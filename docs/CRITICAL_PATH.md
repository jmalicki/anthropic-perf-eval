# Critical-path rewrites: the chains behind t and q

The load-bound static bound is `t + q + ceil(W / 2)`: t is the dependency depth to the first
gather load, q the depth from the last gather load to the end. The start-up window probe
(docs/STARTUP_WINDOW_PROBE.md) found no scheduling slack in the start-up, so this shortens the
chains themselves, with emitter options (each a Z3 identity in `verification/rewrites.py`).

All cycle counts are 16-round greedy development measurements with the best flags
(`--use-greedy --greedy-barriers none --greedy-priority wavefront --greedy-split-shift
--pressure-inflation-threshold 0.95 --greedy-load-priority final-last
--greedy-groups-in-flight 32 --greedy-stagger-cycles 9 --merge-hash-stages --index-form
mirrored`; path plan `path_flow90.json --select-bit-source path --greedy-alu-priority
native-first --greedy-flow-priority chain-first`; index plan `flow90.json`). Every kernel
quoted is PROVED and PASSED. "Measured" = built with and without.

## Answers

- **New bests:**
  - **path: 1,030 -> 1,026** (1,028 simulated; PROVED, PASSED, `check_upstream.sh --strict`)
    with `--reassociate-gather-xor --greedy-const-policy alu-first`. Measured by removal it
    is an **interaction**: R16 alone 1,031, `alu-first` alone 1,030, together 1,026; and it is
    **fragile**: at stagger 8 the same pair gives 1,067 (base 1,030), at stagger 10 1,040 (base
    1,045).
  - **index: 1,039 -> 1,038** (1,040 simulated; PROVED, PASSED, `check_upstream.sh --strict`)
    with `--index-early-terms final`, and consistently 1 to 2 cycles better at staggers 8 and
    10 (1,046 vs 1,048, 1,056 vs 1,058).
- **The bound moves, the greedy mostly does not.** Every rewrite shortens its chain as
  designed (measured from the static bound's heads and tails), but the bound only falls by
  1 to 5 cycles and most combinations run slower than the baseline.
- **t alone cannot lower the bound on these plans.** With t = 51 the head row is
  51 + 14 + ceil(1,888 / 2) = 1,009, and the whole-load row (t = 0) is 14 + ceil(1,989 / 2) =
  1,009 too: about 100 setup loads exactly fill [0, t). Shortening t helps only together with
  fewer setup loads (`--greedy-const-policy alu-first`: 22 fewer) or a shorter q.

## 1. The chains (measured: `scripts/critical_chain.py`)

Path kernel (1,030), same shape on the index kernel (1,039):

| Piece | Emitted depth | Minimum in principle | Where the emission exceeds it |
|---|---|---|---|
| setup (header pointer load, `+`, root load, deferred-constant `^`, `vbroadcast`) | 6 | 5 if the root is broadcast before the deferred-constant XOR, but that costs one XOR per group | - |
| round 0: xor, hash, branch `&` | 10 | 10 | the hash is 8 = 1 + 2 + 2 + 1 + 2, the minimum for the merged form (R8) |
| select round 1 to 3: vselect, xor, hash, `&` | 11 each | 10 | xor after the select: push it below (R18) |
| into the first gather: `&`, index `multiply_add`, address `-`, load | 4 | 3 | precompute the part without the newest bit (R17) |
| tail: gather load, `^ K`, `^ val`, hash, `&`, `multiply_add`, `-`, `vstore` | 14 | 12 | two XORs after the load (R16); the index chain (R17) |

## 2. The rewrites

| Option | Identity | Chain | Ops |
|---|---|---|---|
| `--reassociate-gather-xor` | R16: `val ^ (node ^ K) = (val ^ K) ^ node` | q -1 (one XOR after each gather load) | +0 |
| `--index-early-terms final` | R17: `V - (2n + b) = madd(n, -2, V) - b` | q -1 (final index store) | +1 (the -2 broadcast; `n'` becomes dead) |
| `--index-early-terms first-gather` | R17, also each group's first gather address | t -1 | +65 (path) |
| `--select-xor-pushdown first-pass` | R18: `val ^ sel(c, a, b) = sel(c, val ^ a, val ^ b)` | t -3 (path), -2 (index) | +1 per group per select round (+116 path, +79 index) |

## 3. Measurements (16 rounds)

| Configuration | Path: bound (t, q), cycles | Index: bound (t, q), cycles |
|---|---|---|
| base | 1,009 (51, 14), **1,030** | 1,029 (51, 14), **1,039** |
| R16 | 1,008 (51, 13), 1,031 | 1,028 (51, 13), 1,084 |
| R17 final | 1,008 (51, 13), 1,031 | 1,028 (51, 13), **1,038** |
| R17 first-gather | 1,007, 1,044 | 1,027 (50, 13), 1,043 |
| R16 + R17 final | 1,007 (51, 12), 1,031 | 1,027 (51, 12), 1,074 |
| R16 + R17 first-gather | fails the scratch budget (peak 1,727) | 1,026 (50, 12), 1,045 |
| R18 first-pass | 1,008 (whole-load row; t 48), 1,048 | 1,027 (t 49), 1,046 |
| R18 all | 1,010, 1,059 | 1,027, 1,057 |
| `alu-first` | 1,009, 1,030 | 1,029, 1,039 |
| R16 + `alu-first` | 1,008, **1,026** | 1,028, 1,082 |
| R17 final + `alu-first` | 1,008, 1,029 | 1,028, 1,038 |
| R16 + R17 final + `alu-first` | 1,007 (51, 12), **1,026** | 1,027, 1,049 |
| R18 + R16 + R17 final + `alu-first` | **1,004** (48, 12), 1,043 | **1,025** (49, 12), 1,046 |

Reading:
- The rewrites do what they claim on the chains; the lowest bound (1,004 path, 1,025 index,
  5 and 4 below the baselines' bounds) needs all of them, and that combination runs 13 and 7
  cycles above the best kernels.
- R16 costs the index plan 45 cycles (attributed: its peak scratch rises from 1,431 to
  1,495, above the 0.95 inflation threshold, and the greedy shifts 115 fewer ops).
- R18 adds vector ops where the vector ALU is busiest (the select rounds), which outweighs
  three cycles of chain.

## Not done, and why

- **Setup root broadcast before the deferred-constant XOR**: one cycle off the setup chain,
  but the head row is not the binding one alone (see Answers), and it costs one XOR per group.
- **Transform rules**: the rewrites are emitter options; as transform rules
  (`scheduler_rs/src/kernel_builder/transform/`, not in this stack) they would match:
  R16, a vector XOR whose operand is `gathered_node ^ broadcast(K)` with one reader;
  R17, a SUB `V - madd(n, 2, b)` (or the final store's) where `b` is a branch bit;
  R18, a vector XOR of a value with the result of a top-level select.

## Commands

```bash
uv run python -m scripts.critical_chain sched.json          # the t and q chains, op by op
uv run python -m verification.rewrites                      # R16 to R18
uv run generate_ssa_kernel.py <best flags> --emission-plan docs/emission_plans/path_flow90.json \
  --select-bit-source path --greedy-alu-priority native-first --greedy-flow-priority chain-first \
  --reassociate-gather-xor --greedy-const-policy alu-first --save-kernel k.json   # 1,026
uv run generate_ssa_kernel.py <best flags> --emission-plan docs/emission_plans/flow90.json \
  --index-early-terms final --save-kernel k.json                                  # 1,038
./scripts/check_upstream.sh --kernel k.json --strict
```

Tests: `tests/test_critical_path_rewrites.py` (each rewrite proved, validated, its op-count
delta, and the chain it shortens, at 6 rounds).
