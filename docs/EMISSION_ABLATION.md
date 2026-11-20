# Emission ablation: fewer ops per kernel

November 2026. Every number here is reproduced by `scripts/emission_ablation.py`
(section 6). All builds are 16 rounds, batch 256, height 10, greedy scheduler, unless a
table says otherwise.

## 1. Short answer

The kernel can emit 18% to 23% fewer vector-op equivalents with its output unchanged, each
reduction behind a switch that is off by default and backed by a Z3 proof.

| | Vector-op equivalents | Loads | Capacity floor | Greedy cycles (schedule / simulator) |
|---|---|---|---|---|
| Default emission | 9,526 | 2,136 | 1,271 | 2,007 / 1,994 |
| `--merge-hash-stages --index-form mirrored --cached-select bits` | 7,769 | 2,114 | 1,057 | 1,860 / 1,847 |
| `--merge-hash-stages --index-form mirrored --cached-select mobius` | 8,026 | 2,114 | 1,071 | 1,837 / 1,824 |
| First row of the two above plus `--values-only` (indices not stored) | 7,668 | 2,112 | 1,056 | 1,855 / 1,842 |

- The floor is the capacity floor for the ops that run emitted, from
  `scripts/schedule_report.py`. It is not a bound on the problem.
- With the three output-preserving options the floor is **bound by loads** (2,114 loads at
  2 per cycle = 1,057), not by the ALUs any more (7,769 / 7.5 = 1,036). Removing further
  vector ops does not lower it; removing loads does.
- With the default greedy scheduler, cycles fall much less than the floor (147 to 170
  cycles against 214), because that schedule is limited by its shape (rounds run one after
  another), not by op count. With the shaped greedy options of the greedy-shape work
  (measured on a local merge, section 3.5) the same emission goes from 1,313 to **1,153**
  schedule cycles, output unchanged.
- `--values-only` changes what the kernel writes (no final indices). It is worth 1 cycle of
  floor once the other options are on, because the floor is then bound by loads. It needs a
  decision from the user and is not part of any recommendation below.

## 2. The options

All are keyword-only arguments of `scheduler_rs.build_tick_tock_kernel_ssa`, flags of
`generate_ssa_kernel.py`, fields of `ProblemSpec` (`scheduler_rs/src/kernel_builder/config.rs`)
and recorded in `run_options`. Proofs are in `verification/rewrites.py` and run under
`./scripts/test.sh proofs`.

| Flag | What it removes | Proof | Output |
|---|---|---|---|
| `--merge-hash-stages` | 1 of 11 hash ops: stages 2 and 3 as `(33a + C2+C3) ^ (16896a + (C2<<9))` | R8 | unchanged |
| `--index-form mirrored` | 1 of 3 index-update ops per round (2 at depth 0, the per-group zero broadcast in the wrap round), the initial index load. Tracks `n = L_d - idx`, `L_d = 2^(d+1) - 2` | R9, R10, R11 | unchanged; initial indices assumed 0, as round 0 already assumed |
| `--prune-dead-index` | For the zero-based form only: the `multiply_add` at depth 0, the initial index load, 31 of 32 wrap broadcasts. No effect with `mirrored` | R12 | unchanged; same assumption |
| `--cached-select bits` | On cached levels: compares, bit extraction from the index, per-group root broadcast, per-group XOR with the deferred hash constant. Selects with the branch bits over a pre-XORed node table; leaf selects are `multiply_add`s, the rest SelectChoice | R13, R15 | unchanged |
| `--cached-select bits-flow` | As `bits`, and the leaf selects are SelectChoice too, so all `2^d - 1` selects per group can go to the flow engine | R13, R15 | unchanged |
| `--cached-select mobius` | As `bits`, with all `2^d - 1` selects as `multiply_add`s over Mobius coefficients; no flow ops | R14, R15 | unchanged |
| `--select-bit-source path` | The bit-extraction ops of the modes above (2 per group at depth 3, 4 at depth 4), by keeping every branch bit live | (none needed) | unchanged; greedy exceeds the scratch budget at 16 rounds |
| `--depth4-select-rounds after-wrap` (with `--tree-reduce-max-depth 4`) | 256 loads: level 4 is selected among 16 cached nodes in round 15 only, gathered in round 4 | R13 to R15 | unchanged |
| `--values-only` | The final index store, the conversion before it, and by dead code elimination the last round's index update | none: justified by the oracle | **changed**: indices not written |

Each `needs-*` check next to these (R8, R9, R12, R13, R14, R15) drops a precondition and
expects a counterexample, to record that the precondition is necessary.

With no option set the emission is identical to what it was before these switches existed:
the 16-round scheduler JSON (SSA ops, operands, cycles, lowered slots) was compared with
`main`'s and is the same.

## 3. Ablation table

`PROVED` is the dataflow replay (kernel equals its SSA program for all inputs). `PASSED` is
numeric validation against `reference_kernel2`: values and indices, or values only where
marked. `Upstream` is `./scripts/check_upstream.sh --kernel` (unmodified upstream simulator
and correctness tests; it compares values only). Every row is PROVED and correct upstream.

"Floor, best split" re-divides the selects between the flow engine and the vector ALU in the
best way (section 5); it differs from "Floor" only where the scheduler's own division is
far from the best one.

### 3.1 One option at a time

| Configuration | Vector-op equiv. | Removed | Loads | Flow | Floor | Floor change | Greedy cycles (sched / sim) | Peak live | PASSED |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 9,526.25 | | 2,136 | 488 | 1,271 | | 2,007 / 1,994 | 1,268 | values and indices |
| `--merge-hash-stages` | 9,015.25 | 511 | 2,138 | 487 | 1,203 | -68 | 1,973 / 1,960 | 1,268 | values and indices |
| `--index-form mirrored` | 8,985.25 | 541, 23 loads | 2,113 | 486 | 1,199 | -72 | 1,969 / 1,956 | 1,269 | values and indices |
| `--prune-dead-index` | 9,426.25 | 100, 23 loads | 2,113 | 479 | 1,257 | -14 | 1,998 / 1,985 | 1,268 | values and indices |
| `--cached-select bits` | 8,884.00 | 642 | 2,136 | 296 | 1,185 | -86 | 2,054 / 2,041 | 1,347 | values and indices |
| `--cached-select bits-flow` | 8,443.00 | 1,083 | 2,136 | 744 | 1,126 | -145 | 2,257 / 2,244 | 1,363 | values and indices |
| `--cached-select mobius` | 9,141.25 | 385 | 2,136 | 40 | 1,219 | -52 | 2,031 / 2,018 | 1,347 | values and indices |
| `--values-only` | 9,426.25 | 100 | 2,136 | 488 | 1,257 | -14 | 2,005 / 1,992 | 1,267 | **values only** |

### 3.2 Combinations with the output unchanged

| Configuration | Vector-op equiv. | Loads | Flow | Floor | Floor, best split | Greedy cycles (sched / sim) | Peak live |
|---|---|---|---|---|---|---|---|
| baseline | 9,526.25 | 2,136 | 488 | 1,271 | 1,271 | 2,007 / 1,994 | 1,268 |
| mirrored + `bits` | 8,280.00 | 2,113 | 294 | 1,104 | 1,104 | 1,879 / 1,866 | 1,331 |
| mirrored + `mobius` | 8,537.25 | 2,113 | 38 | 1,139 | 1,139 | 1,860 / 1,847 | 1,331 |
| **merge + mirrored + `bits`** | 7,769.00 | 2,114 | 294 | **1,057** | 1,057 | 1,860 / 1,847 | 1,339 |
| merge + mirrored + `bits-flow` | 7,328.00 | 2,114 | 742 | 1,057 | 1,057 | 2,243 / 2,230 | 1,347 |
| merge + mirrored + `mobius` | 8,026.25 | 2,114 | 38 | 1,071 | 1,071 | **1,837 / 1,824** | 1,339 |
| `--tree-reduce-max-depth 4` alone | 10,444.25 | 1,637 | 939 | 1,393 | 1,393 | 2,301 / 2,288 | 1,269 |
| ... `--depth4-select-rounds after-wrap` | 9,996.25 | 1,893 | 715 | 1,333 | 1,333 | 2,161 / 2,148 | 1,269 |
| merge + mirrored + `bits` + depth 4 (both visits) | 8,429.75 | 1,612 | 747 | 1,124 | 1,124 | 2,590 / 2,577 | 1,507 |
| merge + mirrored + `bits` + depth 4 after-wrap | 8,110.88 | 1,868 | 524 | 1,082 | 1,082 | 2,357 / 2,344 | 1,467 |
| merge + mirrored + `bits-flow` + depth 4 after-wrap | 7,421.88 | 1,868 | 1,228 | 1,228 | **1,018** | 2,719 / 2,706 | 1,475 |
| merge + mirrored + `mobius` + depth 4 after-wrap | 8,595.12 | 1,868 | 44 | 1,147 | 1,147 | 2,280 / 2,267 | 1,467 |

All rows: PASSED with values and indices.

### 3.3 The same with `--values-only` (output changed: indices not stored)

| Configuration | Vector-op equiv. | Loads | Flow | Floor | Floor, best split | Greedy cycles (sched / sim) | Peak live |
|---|---|---|---|---|---|---|---|
| `--values-only` | 9,426.25 | 2,136 | 488 | 1,257 | 1,257 | 2,005 / 1,992 | 1,267 |
| mirrored + values-only | 8,884.25 | 2,112 | 485 | 1,185 | 1,185 | 1,966 / 1,953 | 1,267 |
| merge + mirrored + `bits` + values-only | 7,668.00 | 2,112 | 294 | 1,056 | 1,056 | 1,855 / 1,842 | 1,338 |
| merge + mirrored + `bits-flow` + values-only | 7,227.00 | 2,112 | 742 | 1,056 | 1,056 | 2,237 / 2,224 | 1,346 |
| merge + mirrored + `mobius` + values-only | 7,925.25 | 2,112 | 38 | 1,057 | 1,057 | 1,835 / 1,822 | 1,338 |
| merge + mirrored + `bits` + depth 4 after-wrap + values-only | 7,977.88 | 1,867 | 523 | 1,064 | 1,064 | 2,232 / 2,219 | 1,466 |
| merge + mirrored + `mobius` + depth 4 after-wrap + values-only | 8,462.12 | 1,867 | 43 | 1,129 | 1,129 | 2,126 / 2,113 | 1,466 |

All rows: PASSED on values only; validation with indices fails for these kernels by
construction (tested).

### 3.4 CP-SAT (small)

3 rounds, batch 64, `--cpsat-deterministic-time 10`. This checks that the options work
through the per-section and full-kernel CP-SAT paths and the forced encodings; it is not a
performance result.

| Configuration | Forced mode | Vector-op equiv. | Loads | Flow | Floor | Cycles (sched / sim) | Peak live | PROVED | PASSED |
|---|---|---|---|---|---|---|---|---|---|
| baseline | | 493.88 | 35 | 36 | 66 | 97 / 99 | 505 | yes | values and indices |
| merge + mirrored + `bits` | | 381.88 | 28 | 19 | 51 | 70 / 72 | 449 | yes | values and indices |
| merge + mirrored + `bits-flow` | | 363.00 | 27 | 40 | 49 | 79 / 81 | 433 | yes | values and indices |
| merge + mirrored + `mobius` | | 388.38 | 26 | 12 | 52 | 71 / 73 | 481 | yes | values and indices |
| merge + mirrored + `bits` + values-only | | 344.75 | 26 | 19 | 46 | 65 / 67 | 361 | yes | values only |
| merge + mirrored + `bits-flow` | `--select-choice fma` | 399.50 | 35 | 0 | 54 | 82 / 83 | 498 | yes | values and indices |
| merge + mirrored + `bits-flow` | `--select-choice vselect` | 359.25 | 27 | 42 | 48 | 78 / 80 | 425 | yes | values and indices |
| merge + mirrored + `bits` | `--valu-shift alu` | 379.75 | 29 | 20 | 51 | 224 / 226 | 457 | yes | values and indices |

No run stopped on the wall-clock backstop. At this size the combined options take CP-SAT
from 97 to 70 cycles (`bits`) or 71 (`mobius`); `bits-flow` is 79, behind `bits`, so CP-SAT
within this budget does not exploit the extra choice either. Vector-op equivalents differ
between rows of the same emission because they count the encodings the solver chose.

### 3.5 Together with the shaped greedy scheduler (measured on a local merge)

The greedy-shape options `--greedy-barriers none --greedy-priority wavefront
--greedy-split-shift --pressure-inflation-threshold 0.95` take the default emission from
2,007 to 1,313 schedule cycles, 42 above its floor. `--greedy-barriers` and
`--greedy-priority` are on `main` (PRs #22, #29); split shift and the pressure thresholds
were still in PRs #30 and #32 when this was measured. **These rows therefore come from a
local, unpushed merge** of `origin/ablation/greedy-shape-4-thresholds` (at `9641198`) into
the tip of the emission stack, and should be re-measured on `main` once those PRs land.

16 rounds, batch 256, all four shape options on. Every row is PROVED, PASSED and correct on
the unmodified upstream tests (`./scripts/check_upstream.sh --kernel`).

| Emission options | Vector-op equiv. | Loads | Flow | Floor | Binds | Cycles (sched / sim) | Above floor | Peak live | PASSED |
|---|---|---|---|---|---|---|---|---|---|
| none (baseline) | 9,526.25 | 2,141 | 483 | 1,271 | ALUs | 1,313 / 1,315 | 42 | 1,244 | values and indices |
| `--merge-hash-stages` | 9,015.25 | 2,155 | 470 | 1,203 | ALUs | 1,251 / 1,253 | 48 | 1,236 | values and indices |
| `--index-form mirrored` | 8,985.25 | 2,117 | 482 | 1,199 | ALUs | 1,247 / 1,249 | 48 | 1,070 | values and indices |
| `--prune-dead-index` | 9,426.25 | 2,111 | 481 | 1,257 | ALUs | 1,301 / 1,303 | 44 | 1,037 | values and indices |
| merge + mirrored | 8,474.25 | 2,119 | 481 | 1,130 | ALUs | 1,197 / 1,199 | 67 | 1,118 | values and indices |
| mirrored + `bits` | 8,280.00 | 2,113 | 294 | 1,104 | ALUs | 1,174 / 1,176 | 70 | 771 | values and indices |
| **merge + mirrored + `bits`** | 7,769.00 | 2,124 | 284 | 1,062 | loads | **1,153 / 1,155** | 91 | 771 | values and indices |
| merge + mirrored + `bits-flow` | 7,328.00 | 2,133 | 723 | 1,067 | loads | 1,160 / 1,162 | 93 | 771 | values and indices |
| merge + mirrored + `mobius` | 8,026.25 | 2,115 | 37 | 1,071 | ALUs | 1,154 / 1,156 | 83 | 771 | values and indices |
| merge + mirrored + `bits` + `--select-bit-source path` | 7,641.00 | 2,119 | 289 | 1,060 | loads | 1,152 / 1,154 | 92 | 811 | values and indices |
| merge + mirrored + `bits` + `--prune-dead-index` | 7,769.00 | 2,124 | 284 | 1,062 | loads | 1,153 / 1,155 | 91 | 771 | values and indices |
| merge + `bits` (zero-based index) | 8,373.00 | 2,139 | 294 | 1,117 | ALUs | 1,183 / 1,185 | 66 | 803 | values and indices |
| merge + `--prune-dead-index` + `bits` | 8,273.00 | 2,109 | 292 | 1,104 | ALUs | 1,173 / 1,175 | 69 | 691 | values and indices |
| merge + `--prune-dead-index` + `bits-flow` | 7,832.00 | 2,127 | 722 | 1,064 | loads | 1,164 / 1,166 | 100 | 715 | values and indices |
| merge + `--prune-dead-index` + `mobius` | 8,530.25 | 2,109 | 36 | 1,138 | ALUs | 1,204 / 1,206 | 66 | 707 | values and indices |
| merge + mirrored + `bits` + `--values-only` | 7,668.00 | 2,125 | 281 | 1,063 | loads | 1,150 / 1,152 | 87 | 738 | **values only** |
| merge + mirrored + `mobius` + `--values-only` | 7,925.25 | 2,113 | 37 | 1,057 | ALUs and loads | 1,147 / 1,149 | 90 | 738 | **values only** |

Floors differ by a few cycles from section 3.2 because this greedy encodes a few more
constants as loads (2,109 to 2,155 loads against 2,113 to 2,138).

What this shows:

- With a scheduler that is not limited by its shape the emission options do become cycles:
  **1,313 to 1,153** schedule cycles with the output unchanged, 160 cycles for 209 cycles of
  floor. The simulator counts 1,155.
- Once loads bind, the remaining options are worth nothing in cycles, as section 4.1
  predicts: `bits` 1,153, `mobius` 1,154, `bits-flow` 1,160, `bits` with `path` 1,152,
  `--values-only` 1,150. Fewer vector ops no longer help. The distance to the floor
  grows as ops are removed: 42 cycles for the default emission, 67 to 70 for merge +
  mirrored and mirrored + `bits`, 83 to 93 for the full combinations. What is left is how
  the loads are scheduled (2 per cycle, and none possible before a group reaches a gathered
  level).
- `--select-bit-source path`, which does not fit the scratch budget with the default
  greedy scheduler, fits here (peak 811 of 1,536) because the wavefront keeps only some
  groups in flight. It saves 128 vector ops and 1 cycle.
- `--prune-dead-index` is a no-op with the mirrored index (identical row) and worth 10
  cycles on the zero-based index with `bits` (1,183 to 1,173).
- Scratch use falls sharply with the bit-driven selects under the wavefront (peak 771
  against 1,244): there is headroom for more groups in flight.
- The wavefront's structural chain detection finds 32 chains with the emission options on
  (read from the scheduler's message for merge + mirrored + `bits`, merge +
  `--prune-dead-index` + `mobius`, and merge + mirrored + `bits-flow` + `--values-only`).
  With the detection as it was before PR #29's review fix it found 1 chain for
  `--index-form mirrored`, because the index pointer and the final-conversion constant
  reach exactly half of the sinks; that is fixed on `main`.

Cross-check with the greedy-shape work's own local merge: it reported 1,199 simulator
cycles for merge + mirrored (reproduced: 1,199) and 1,155 for "mirrored + bits"
(reproduced as merge + mirrored + `bits`: 1,155; without `--merge-hash-stages` it is 1,176).

Reproduce (after merging the greedy-shape branch locally):

```bash
uv run python -m scripts.emission_ablation --upstream \
  --only baseline,merge-hash,mirrored-index,prune-dead-index,merge+mirrored,select-bits+mirrored,all-safe-bits,all-safe-bits-flow,all-safe-mobius,all-safe-bits-path,all-safe-bits+prune,merge+bits,merge+prune+bits,merge+prune+bits-flow,merge+prune+mobius,all-bits+values-only,all-mobius+values-only \
  --build-option greedy_barriers=none --build-option greedy_priority=wavefront \
  --build-option greedy_split_shift=true --build-option pressure_inflation_threshold=0.95
```

## 4. What the measurements say

| Observation | Evidence |
|---|---|
| The three output-preserving options compose almost additively on vector ops: 511 + 541 + 642 = 1,694 separately, 1,757 together | Rows of 3.1 and 3.2. Together is more because `bits` with the mirrored index needs no SUB to mirror it |
| After them the floor is bound by loads | 2,114 loads / 2 = 1,057; ALUs alone would allow 7,769 / 7.5 = 1,036 |
| Of the 2,114 loads, 2,048 are gathers (8 rounds x 256), 32 input `vload`s, 18 tree-cache and header loads, and 16 constants the greedy scheduler encoded as loads | Lowered load ops of the scheduler JSON (`load` 2,066, `vload` 32, `const` 16). Constants as `add_imm` on the idle flow engine would give 2,098 loads, floor 1,049 |
| Greedy cycles do not follow the floor | `--cached-select bits` alone: floor -86, greedy **+47**. `bits-flow`: floor -145, greedy +250. The greedy scheduler puts every SelectChoice on the flow engine (1 slot per cycle) and runs rounds strictly in sequence |
| `mobius` schedules best with greedy today although it has more vector ops than `bits` | 1,837 against 1,860: it needs no flow slot, so select rounds are not serialised |
| Caching level 4 does not pay with the current schedulers | It trades 8 loads per group for 15 selects per group. With leaf selects on the vector ALU (`bits`) the floor rises from 1,057 to 1,082. With every select a choice (`bits-flow`) the best division gives 1,018, but greedy's division gives 1,228 |
| Keeping bits live instead of extracting them (`--select-bit-source path`) does not fit | Greedy at 16 rounds: `peak_live=1571 > budget=1536`, build fails. It saves 128 vector ops (17 cycles of ALU floor) where the floor is already bound by loads |
| The order of bits in the select tree matters for the scheduler | With the newest bit at the leaves, greedy overflowed the scratch budget from 12 rounds on (zero-based index). With the oldest bit at the leaves, which also keeps all but one select off the hash-to-hash critical path, it fits |

### 4.1 Which engine binds, and what it would take to go lower

Per-engine terms of the floor: ALUs = vector-op equivalents / 7.5 rounded up, loads / 2,
flow / 1. The floor is the largest; that engine binds, and removing work from any other
engine does nothing.

| Combination | ALU term | Load term | Flow term | Floor | Binds |
|---|---|---|---|---|---|
| baseline | 1,271 | 1,068 | 488 | 1,271 | ALUs |
| `--merge-hash-stages` | 1,203 | 1,069 | 487 | 1,203 | ALUs |
| `--index-form mirrored` | 1,199 | 1,057 | 486 | 1,199 | ALUs |
| `--cached-select bits` | 1,185 | 1,068 | 296 | 1,185 | ALUs |
| mirrored + `bits` | 1,104 | 1,057 | 294 | 1,104 | ALUs |
| merge + mirrored + `bits` | 1,036 | 1,057 | 294 | 1,057 | **loads** |
| merge + mirrored + `bits-flow` | 978 | 1,057 | 742 | 1,057 | **loads** |
| merge + mirrored + `mobius` | 1,071 | 1,057 | 38 | 1,071 | ALUs |
| merge + mirrored + `bits` + values-only | 1,023 | 1,056 | 294 | 1,056 | **loads** |
| merge + mirrored + `bits-flow` + values-only | 964 | 1,056 | 742 | 1,056 | **loads** |
| merge + mirrored + `mobius` + values-only | 1,057 | 1,056 | 38 | 1,057 | ALUs |
| merge + mirrored + `bits` + depth 4 after-wrap | 1,082 | 934 | 524 | 1,082 | ALUs |
| merge + mirrored + `bits-flow` + depth 4 after-wrap | 990 | 934 | 1,228 | 1,228 | flow as greedy encodes the selects; 1,018 (ALUs and flow together) with the best division |

So with `bits` or `bits-flow` the next lever is loads, not vector ops: `bits-flow` already
has 79 cycles of ALU headroom under its load term that it cannot use.

What the load count would have to be:

| Floor target | Loads allowed (2 per cycle) | To remove from 2,114 | Vector-op equivalents allowed (7.5 per cycle) | Where the combinations stand |
|---|---|---|---|---|
| 1,057 (today) | 2,114 | 0 | 7,927 | |
| 1,049 | 2,098 | 16: the constants greedy encodes as loads, as `add_imm` on flow instead | 7,867 | Reachable by constant encoding alone |
| 1,000 | 2,000 | 114 (98 beyond the constants): the gathers of 13 of the 32 groups in one round | 7,500 | `bits` has 7,769 (269 too many); `bits-flow` has 7,328 |
| 950 | 1,900 | 214 (198 beyond the constants): the gathers of 25 groups in one round, i.e. most of one of the 8 gather rounds | 7,125 | `bits-flow` has 7,328 (203 too many), 7,227 with `--values-only` (102 too many) |

The loads are 2,048 gathers (8 gather rounds x 32 groups x 8 lanes), 32 input `vload`s, 18
tree-cache and header loads and 16 constants. Everything but the gathers is 66 loads, so
both targets need gathers to go. The only switch here that removes gathers is level-4
caching (256 loads in round 15), and it pays for them with vector ops and selects: with
`bits-flow` it lands at 990 for the ALUs and 934 for the loads but needs 1,184 selects
divided well between flow and the vector ALU (floor 1,018), and with `bits` it adds 342
vector-op equivalents (floor 1,082). Caching a level for part of the groups (the
"best mix" of `docs/BOUND_GAP_ANALYSIS.md` section 5) is not implemented. Below about 1,000
the route is the one that document describes: one load per lane serving several rounds
(packets and per-lane `vload`), which is out of scope here.

### 4.2 Against the estimates

Against the estimates of `docs/BOUND_GAP_ANALYSIS.md` section 4.1:

| Estimate | Estimated saving | Measured |
|---|---|---|
| Merge hash stages 2 and 3 | 512 | 511 (three constants changed) |
| 2-op index update | 480 | 541 with the mirrored form, which also drops the depth-0 `multiply_add`s and the wrap broadcasts |
| No index chain where nothing reads it | 160 | 100 as `--prune-dead-index` or `--values-only` on the default form; contained in the 541 for the mirrored form |
| Tree access on cached levels | 1,257, all selects on flow | 1,083 with `bits-flow` (all selects on flow as greedy encodes them); 642 with `bits`. The difference to the estimate is the bit extraction at depth 3 (2 to 3 ops per group) that keeps bits from staying live |
| Design B0 (all of the above, values only): floor 1,044 from 2,088 loads | | 1,056 from 2,112 loads; the 24 extra loads are constants and header loads that the model's 40-load overhead did not include |
| Design B1 (round 15 selects among 16 nodes): floor 954 | | 1,018 at best, see above |

## 5. Two floors

`scripts/schedule_report.py` computes the floor with every select and constant encoded as
the scheduler chose in that run. For most rows that is what matters. Where every select is
a choice (`bits-flow`), greedy's all-on-flow encoding can make the flow engine the binding
term although a better division exists. `best_select_split_floor` in
`scripts/emission_ablation.py` reports the smallest `T` such that loads fit in `2T`, and
with as many selects on the `T` flow slots as fit (the 2-op ones first), the remaining
vector-op equivalents fit in `7.5 T`. It is continuous in the ALU term and keeps constants
as encoded, so it is a floor for the emission under the best select encoding, still not a
bound on the problem.

## 6. Recommendation

| Option | Recommendation | Why |
|---|---|---|
| `--merge-hash-stages` | Turn on by default | 511 ops, no downside measured: output unchanged, scratch unchanged, greedy 34 cycles better alone |
| `--index-form mirrored` | Turn on by default | 541 ops and 23 loads; greedy 38 cycles better alone; makes `--prune-dead-index` unnecessary |
| `--cached-select` | `bits` together with the shaped greedy options; `mobius` with the default greedy scheduler | Shaped greedy (section 3.5): `bits` 1,153, `mobius` 1,154, and `bits` has the lower floor (1,057 against 1,071). Default greedy: `mobius` 1,837, `bits` 1,860 |
| `--cached-select bits-flow` | Leave off | Lowest op count, but only pays with a scheduler that balances flow against the vector ALU; greedy gets 2,243 |
| `--prune-dead-index` | Leave off | Subsumed by `mirrored` |
| `--select-bit-source path` | Leave off | Does not fit the scratch budget at 16 rounds with the default greedy scheduler; fits under the wavefront but is worth 1 cycle there, because loads bind |
| `--tree-reduce-max-depth 4`, `--depth4-select-rounds after-wrap` | Leave off; re-measure when CP-SAT or the greedy scheduler handles `bits-flow` well | Only `bits-flow` with a balanced division gets below the load-bound 1,057 (to 1,018) |
| `--values-only` | **Needs a decision.** | It stops writing the final indices. Upstream tests and the leaderboard compare values only, the repository's own `--validate` compares indices too. Its gain is 1 cycle of floor with the other options on (the floor is then bound by loads), 14 cycles alone |

Defaults were not changed by this work.

What would lower the floor further, in order: fewer loads (the gathers: this is the
packet / per-lane `vload` design of `docs/BOUND_GAP_ANALYSIS.md` section 4.4, out of scope
here); constants on the flow engine instead of loads (8 cycles); then vector ops again.
Section 4.1 gives the load counts needed for 1,000 and 950.

## 7. Reproduction

```bash
./scripts/build_scheduler_rs.sh --release

# Sections 3.1 to 3.3 (about 15 minutes; each kernel is built, proved, validated, and run upstream)
uv run python -m scripts.emission_ablation --upstream --markdown

# Section 3.4
uv run python -m scripts.emission_ablation --rounds 3 --batch-size 64 --cpsat-deterministic-time 10 \
  --only baseline,all-safe-bits,all-safe-bits-flow,all-safe-mobius,all-bits+values-only --markdown
uv run python -m scripts.emission_ablation --rounds 3 --batch-size 64 --cpsat-deterministic-time 10 \
  --only all-safe-bits-flow --select-choice fma --markdown      # also: --select-choice vselect
uv run python -m scripts.emission_ablation --rounds 3 --batch-size 64 --cpsat-deterministic-time 10 \
  --only all-safe-bits --valu-shift alu --markdown

# One configuration by hand
uv run generate_ssa_kernel.py --rounds 16 --use-greedy --validate -q \
  --merge-hash-stages --index-form mirrored --cached-select bits --output-json sched.json
uv run python -m scripts.schedule_report sched.json

# The proofs
uv run --extra verify python -m verification.rewrites
```

## 8. Not verified

| Item | State |
|---|---|
| CP-SAT at 16 rounds with any of these options | Not run. Section 3.4 is 3 rounds, batch 64 |
| `--depth4-select-rounds` and `--select-bit-source path` with CP-SAT | Not run |
| Forest heights other than 10 | Not run: the default emission itself panics at height 3 |
| Initial indices other than 0 | `Input.generate` always produces 0. `mirrored`, `--prune-dead-index` and the bit-driven selects rely on it, as the root broadcast of round 0 always did |
| The whole hash as one Z3 query | Times out; proved stage by stage (R1, R8, R2) |
| Interaction with the greedy-shape switches | Measured on a local merge only (section 3.5); to be repeated on `main` once those PRs land |
| Interaction with the CP-SAT switches (PRs #23 to #28) | Not measured |
