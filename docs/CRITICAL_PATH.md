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

An option that would change nothing in a kernel is an error (`ProblemSpec::check_critical_path_options`,
for the CLI, the binding and Rust callers alike): `--index-early-terms` without `--index-form
mirrored`; `--select-xor-pushdown` when no round it covers has a SelectChoice at the top of its
select tree (`standard` or `mobius` cached select, `bits` only at depth 1); `--reassociate-gather-xor`
when no group gathers. `first-pass` means rounds 0 to `forest_height` (up to the first wrap).
`tests/test_critical_path_rewrites.py` also builds every option after the wrap (height 3, 7 to 9
rounds), proved and validated.

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

## Justification on the best kernels (`--greedy-justify 10`)

Forward/backward justification (docs/GREEDY_SHAPE_ABLATION.md section 13) applied to the
best flags above. Justification now models ALU-derived constants (`--greedy-const-policy`):
a derived constant is scheduled after both producers, and both stay live to its cycle in the
pass's scratch check and in its address-assignment trial. Before, the pass skipped any
schedule with one. All rows: 16 rounds, `--use-greedy --validate`, PROVED and PASSED, no
build over the scratch budget unless marked.

Data: the sweep, rewrites and neighbours tables below are
`docs/emission_plans/justify_on_best_sweep.json` (every row with its command line), rebuilt
and compared by `uv run python -m scripts.justify_on_best_sweep` and by the ci_only test
`tests/test_justify_on_best_sweep.py`. The justify 0 / 10 rows of the removal table are
pinned by `tests/test_justify_on_best_pin.py` (ci_only); the justify 50 column was measured
once and is not in the data.

**What this branch's code change does and does not explain.** The change is that
justification now models ALU-derived constants, so it runs with `--greedy-const-policy
alu-first` (before, it skipped such schedules): that is the `alu-first` rows (1,026 -> 1,016).
The headline 1,012 / 1,033 uses the default const policy (no derived constants) and does not
depend on it: the neighbours table rebuilt on `critical-path` (#91, without this change)
gives the same cycles in every row. It is a measurement recorded here.

**Measured by removal** (justify on/off, every other flag fixed; best flags + path
`--reassociate-gather-xor` + chain-first + native-first, index `--index-early-terms final`):

| Configuration | justify 0 | justify 10 | justify 50 |
|---|---|---|---|
| index | 1,038 / 1,040 | 1,038 / 1,040 | 1,038 / 1,040 |
| path, `alu-first` | 1,026 / 1,028 | **1,016 / 1,018** | 1,016 / 1,018 |
| path, default const policy | 1,031 / 1,033 | **1,016 / 1,018** | 1,016 / 1,018 |

(scheduled / simulator cycles). The 1,026 interaction does not stack with justification:
both const policies end at 1,016. Peak scratch 1,497 words (alu-first) and 1,520 (default).

**Index plan, why nothing changes:** not a guard. With `SCHEDULER_DEBUG=1` every forward
pass comes out at exactly 1,038 and the backward passes at 1,038 to 1,046, so no candidate is
strictly shorter and none is checked. A plateau of the list scheduler, not a proven optimum
(the static bound is 1,028).

### Sweep (all with `--greedy-justify 10`)

Shape, with each plan's rewrites as above (path also `alu-first`); cycles, "fail" = the
greedy itself exceeded the scratch budget (peak 1,728), so justification did not run and the
build failed:

| stagger / in flight | path, prio | path, no prio | index, prio | index, no prio |
|---|---|---|---|---|
| 8 / 24 | 1,030 | 1,032 | 1,061 | 1,057 |
| 8 / 32 | **1,016** | fail | 1,038 | 1,038 |
| 9 / 24 | 1,027 | 1,033 | 1,061 | 1,061 |
| 9 / 32 | **1,016** | fail | 1,038 | 1,038 |
| 10 / 24 | 1,028 | 1,035 | 1,063 | 1,059 |
| 10 / 32 | **1,016** | 1,018 | 1,038 | 1,038 |

Robustness: with 32 groups in flight and the priorities, path is 1,016 at staggers 8, 9 and
10 (before justification the same pair gave 1,067 and 1,040 at 8 and 10); index is 1,038 at
all three. 24 in flight is worse on both plans.

Rewrites (stagger 9, 32 in flight; path with chain-first + native-first, index without);
cycles, with `alu-first` off / on:

| Rewrites | path | index |
|---|---|---|
| none | 1,019 / 1,019 | 1,039 / 1,039 |
| R16 | 1,016 / 1,016 | 1,036 / 1,036 |
| R17 final | 1,018 / 1,018 | 1,038 / 1,038 |
| R16 + R17 final | 1,015 / 1,015 | 1,035 / 1,035 |
| R18 first-pass | 1,016 / 1,016 | 1,037 / 1,037 |
| R18 first-pass + R16 + R17 final | **1,012 / 1,012** | **1,033 / 1,033** |

The rewrites lowered the static bound but the plain greedy did not follow (section 3);
with justification it does: all three together give path **1,012** (1,014 simulated) and
index **1,033** (1,035), the lowest of the sweep, with or without `alu-first`. Peak scratch
1,401 to 1,453 words.

`--greedy-budget-force stall` at inflation 0.95 and 0.98 (best path / index rows above):
1,016 / 1,038 in both, unchanged.

**The new bests and their neighbours** (R18 first-pass + R16 + R17 final, justify 10; path
with chain-first + native-first unless noted):

| Setting | path | index |
|---|---|---|
| stagger 9, 32 in flight (best) | **1,012 / 1,014** | **1,033 / 1,035** |
| stagger 8 | 1,012 | 1,033 |
| stagger 10 | 1,012 | 1,035 |
| 28 in flight | 1,014 | 1,040 |
| 24 in flight | 1,026 | 1,061 |
| path without the priorities / index with them | 1,012 | 1,033 |
| same, **justify 0** (measured by removal) | 1,044 | 1,047 |

Both best kernels pass `./scripts/check_upstream.sh --strict` (path 1,014 simulated, 145.7x;
index 1,035, 142.7x). Path 1,012 holds at staggers 8 to 10 and with or without the
priorities; the gain needs 32 groups in flight (28: +2, 24: +14). Justification is worth 32
cycles on path and 14 on index with these rewrites: without it the rewrites run slower than
the baseline (section 3), with it they give the best kernels. Pinned by
`tests/test_justify_on_best_pin.py` (ci_only; also the `alu-first` pair 1,026 -> 1,016, the
only rows that need this branch's change).

**As a CP-SAT hint (measured on a local merge, not reproducible from this branch).** The
justified schedules are solutions of the CP-SAT model. Checked with the window probe's
all-fixed check (every op pinned) on `section-check-timeline` (#92, 87f4e80, its
scratch-model fix) with this branch merged in locally and never pushed; on this branch alone
the per-section check overcount (BUGS.md) blocks it, so the result cannot be reproduced here
until #92 lands. Best flags plus
`--greedy-justify 10 --cpsat-split-shift-hint`, `--no-sweep --no-rolling`:

| Plan | greedy (CP-SAT pipeline, full bounds) | justified, inside every ASAP/ALAP domain | all-fixed check |
|---|---|---|---|
| path, R16 + R17 final + R18 first-pass | 1,046 | 1,012 (0 domain rejections) | optimal, makespan 1,012 |
| index, the same rewrites | 1,044 | 1,033 | optimal, makespan 1,033 |

Both final kernels PROVED and PASSED. So the justified schedule survives as the hint of a
whole-program or windowed CP-SAT solve with these flags; no CP-SAT improvement was run here.

```bash
# 1,016 (path, R16, alu-first optional)
uv run generate_ssa_kernel.py --rounds 16 --use-greedy --validate \
  --greedy-barriers none --greedy-priority wavefront --greedy-split-shift \
  --pressure-inflation-threshold 0.95 --greedy-load-priority final-last \
  --greedy-groups-in-flight 32 --greedy-stagger-cycles 9 --merge-hash-stages --index-form mirrored \
  --emission-plan docs/emission_plans/path_flow90.json --select-bit-source path \
  --greedy-alu-priority native-first --greedy-flow-priority chain-first \
  --reassociate-gather-xor --greedy-justify 10 --save-kernel k.json
# 1,012 (path) / 1,033 (index, flow90.json, no priorities, no path bits): add
#   --index-early-terms final --select-xor-pushdown first-pass
./scripts/check_upstream.sh --strict --kernel k.json
```
