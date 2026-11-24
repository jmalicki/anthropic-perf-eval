# From the baseline kernel to our best: the gap a compiler has to close

**Conclusion.** The north star is a compiler that takes the original, unoptimised kernel and
transforms it into a kernel as good as our best, with no strategy flags. The bar is a top
model working on the kernel by hand for hours.

The original kernel is the scalar baseline builder: 147,734 cycles, 16 rounds, batch 256,
measured. It is straight-line code. One op per bundle. Indices and values round-trip
through main memory in every round.

Our best kernel is about 1,100 cycles. Three kinds of work separate the two:

1. **Already done well.** The scheduler, the scratch allocator, and the dataflow proof of
   kernel against SSA.
2. **Known only inside the emitter.** Today this knowledge is hard-coded in `ssa_emit.rs`
   rather than available as a transform:
   - keeping indices and values in registers;
   - turning the branch into `& 1` arithmetic;
   - bounds-check elimination;
   - hash fusion;
   - vectorisation into 32 groups of 8;
   - every rule in the catalogue.
3. **Missing outright.** SSA recovery from a lifted kernel, memory-to-register promotion,
   dead-store elimination, if-conversion, and SLP vectorisation as transforms. Also an
   SSA-to-SSA equivalence check.

The analyses these need mostly exist already, in `verification/memory_alias.py`:
- exact memory aliasing;
- header folding;
- index value sets, which already show `2^d` values per round.

So the first missing step is SSA recovery plus memory-to-register promotion, not new
analysis.

This document is a map, not a work plan. The near-term plan stays as it is in the catalogue:
first the strategy flags become rules. Reaching the benchmark number does not need anything
in section 3's "goal only" rows. See
[TRANSFORM_PIPELINE_DESIGN.md](TRANSFORM_PIPELINE_DESIGN.md) and
[TRANSFORM_RULE_CATALOGUE.md](TRANSFORM_RULE_CATALOGUE.md).

Each claim below is marked in one of three ways: **read** (file:line), **measured** (the
command is in section 6), or *(inferred)*.

## 1. What the baseline program is

### Where it lives

- The repo's original baseline builder is `KernelBuilder.build_kernel` in `perf_takehome.py`
  at the repo's first commit `dceff11`, at lines 28-147 of that version (**read**: `git show
  dceff11:perf_takehome.py`).
- Today's root `perf_takehome.py` imports our builder instead (`perf_takehome.py:27`,
  `from solver.kernel_builder import KernelBuilder`).
- `upstream/perf_takehome.py` is a shim that loads a saved kernel (`upstream/perf_takehome.py:1-15`).
- The upstream original is not vendored. `upstream/UPSTREAM_COMMIT` records its sha256 as
  `af14cbb2…`. The `dceff11` file hashes to `f9c1de09…` (**measured**), so it is not
  byte-identical to upstream.
- It produces exactly upstream's `BASELINE = 147734` (`upstream/tests/submission_tests.py:63`)
  on the vendored upstream simulator.

### Measured

16 rounds, batch 256, height 10, seed 123, vendored `upstream/tests/frozen_problem.py`
simulator, debug and pause disabled.

| Fact | Value |
|---|---|
| Cycles | **147,734**; values and indices both match `reference_kernel2` |
| Bundles | 196,887, of which 147,734 contain a non-debug slot. Debug-only bundles cost no cycle |
| Slots per bundle | 1 (`build` puts each slot in its own bundle, `dceff11:perf_takehome.py:39-43`) |
| Control flow | Straight-line: no `jump`, `cond_jump` or loop; 2 `pause` |
| Ops per engine | ALU 118,784; load 12,564; store 8,192; flow 8,194; VALU 0; debug 49,153 |
| ALU opcodes | `+` 53,248; `^` 24,576; `<<` 16,384; `>>` 8,192; `%` 4,096; `*` 4,096; `==` 4,096; `<` 4,096 |
| Other opcodes | `load` 12,295; `const` 269; `store` 8,192; `select` 8,192 |
| Scratch words used | 276 of 1,536 |

### Per element per round

Element `i` in round `r`, 4,096 times in all (**read**, `dceff11:perf_takehome.py:112-145`).
Each step is one op:

1. Load `idx` from `inp_indices_p + i`, and `val` from `inp_values_p + i`. Each address is
   an ALU add.
2. Load `node` from `forest_values_p + idx` (ALU add, scalar load).
3. `val ^= node`.
4. Hash: 6 stages × 3 ALU ops, each against a `const`-loaded constant (`build_hash`, lines
   65-73).
5. Branch: `t = val % 2`; `t = t == 0`; `off = select(t, 1, 2)` on flow.
6. Index update: `idx = idx * 2 + off`.
7. Bounds check: `t = idx < n_nodes`; `idx = select(t, idx, 0)` on flow.
8. Store `idx` and `val` back to their addresses (ALU adds, scalar stores).

### How idx and val move through memory

From the lift (**measured**, section 2):
- Every store of round `r` is read back by exactly one load in round `r + 1`, from the same
  constant address.
- The 3,840 value stores of rounds 0–14 are each read once; the round-15 stores are never
  read.
- Index stores follow the same pattern.

### Ops the baseline uses and our emission never does

- `%`, `==` and `<` on the ALU.
- Scalar flow `select`.
- Per-element scalar `load` and `store` of indices and values. We use `vload`/`vstore` once
  per group.
- `const` loads for every element offset 0–255. We have 32 group offsets as ConstChoice.
- A `load` of every header word 0–6. We load 4–7: `ssa_emit.rs:269-280`.
- Scalar `*` and `+` where we use `multiply_add`.
- The 18-op hash. Ours is 11 or 10 ops.

### Its target

The upstream tests set the bar: `< 18,532` ("updated starting point"), `< 2,164` (Opus 4,
many hours), down to `< 1,487` and `< 1,363` (`upstream/tests/submission_tests.py:88-115`).
Our best 16-round kernel is about 1,099 cycles with the output unchanged
(EMISSION_STRUCTURE_MODEL.md, PUBLIC_KERNEL_RESCHEDULE.md "Sources").

## 2. What the lifter and the alias analysis give for it today

`scripts/kernel_lift.py` with `verification/memory_alias.py`, on the baseline kernel saved
with debug slots removed and layout metadata set (**measured**):

| | 2 rounds, batch 8 | 16 rounds, batch 256 |
|---|---|---|
| Runtime | 0.7 s (with 4 concrete runs) | 5.1 s (no concrete runs) |
| Ops | 610 | 147,734 |
| Graph edges | 1,894 | 490,502 |
| Memory edges | 64 | 23,552 |
| Memory ops; ordered pairs with a store | 87; 2,256 | 20,487; 134,270,976 |
| Pairs needing Z3 | 0 (all decided by value sets) | 0 (all decided by value sets) |
| Critical path with the kernel's scratch addresses | 392 | **98,312** |
| Critical path through value dependencies only | 49 | **371** |
| Capacity floor (fixed encoding) | 40 (load) | 9,899 (ALU) |
| Concrete check | PASS, 0 unordered aliasing pairs | not run |

What this says:

- **Supported at full size, and fast.** Nothing stopped or timed out.
- **Scratch reuse serialises the program.** The baseline reuses `tmp1`, `tmp2`, `tmp3`,
  `tmp_idx`, `tmp_val` and `tmp_addr` for all 4,096 element-rounds. Only 276 words are used.
  These are false dependencies: the critical path is 98,312 cycles with the kernel's
  addresses against 371 through values. Renaming into SSA removes them.
- **Memory is exact and simple.** Every address is a constant or `forest_p + idx`. Value sets
  decide every pair, with no Z3 call. The store-to-load flow is one-to-one (section 1). This
  is exactly what memory-to-register promotion needs.
- **The index range analysis is already there in part.**
  - The memory-use report classifies each index store by its number of possible values:
    2, 4, 8, … 1,024 in successive rounds.
  - That is the `2^d` level size of R7, derived by `ValueSets` on the lifted program without
    help from an emitter.
  - Whether the sets are exactly the R7 intervals, and not just of that size, is not checked
    here *(inferred: they are)*.
- **What the lift does not give.**
  - SSA values, because words are reused.
  - Vector groups, because there are none in a scalar kernel.
  - Which ops form a hash, a branch or a round.
  - A program our scheduler can take. It works on `SsaProgram` with `MemLocation`, not on
    a lifted graph (PUBLIC_KERNEL_RESCHEDULE.md "Rescheduling").

## 3. The chain of passes

The passes run in order. "Bench" means the step is needed for the benchmark number. "Goal"
means it is needed only to start from the baseline. Status:

- **have**: exists as a transform or analysis;
- **emitter-only**: the knowledge exists, hard-coded in the emitter;
- **lack**: does not exist.

| # | Pass | Matches | Facts needed | Proof obligation | Status | Bench / goal |
|---|---|---|---|---|---|---|
| B0 | Drop debug and pause | `debug` slots; `pause` | none | none (the simulator ignores debug for cycles; pause is a harness hook) | have in practice: the measurement drops them | goal |
| B1 | SSA recovery | per-word writes in a straight line become values; vector reads become vector values or groups | last writer per word | the recovered program computes what the kernel computes: `kernel_dataflow` run on the original kernel against the recovered program | lack. Per-word RAW edges exist (`scripts/kernel_lift.py:83-115`); renaming and group recovery do not (design doc §7) | goal |
| B2 | Parameter specialisation | header loads → constants (`n_nodes = 2047`, pointers) | the memory-image preconditions | folding is sound under the stated preconditions | analysis has it (`memory_alias.py` docstring §2: header words fold). Emitter-only for sizes: `ProblemSpec` compile-time shape (`config.rs:253-318`). Our emitter still loads the 4 pointers (`ssa_emit.rs:269-280`) | goal |
| B3 | Memory-to-register promotion of idx and val across rounds | `store(a, x)` … `load(a)` with no may-alias store between | exact alias (value sets suffice here, measured) | the forwarded value equals memory's: the `SymbolicKernel` store-forwarding model (`memory_alias.py:592`) | analysis has it; the transform is emitter-only (`ssa_emit.rs:565-597` load once, `615-646` store once) | goal |
| B4 | Dead-store elimination | a store overwritten at the same address with no load between; with `values_only`, stores to unobserved memory | must-alias WAW, no may-alias load between; the output contract | exact alias; contract declared | lack as a transform (`FilterDeadStoresPass` only drops ops left unscheduled, `pipeline/passes/filter_dead.rs:1-13`); emitter-only (same lines as B3) | goal |
| B5 | If-conversion of the branch | `t = v % 2; t = (t == 0); off = select(t, 1, 2)` → `off = 1 + (v & 1)` | none beyond shape | `x % 2 == x & 1` is MA2 (`memory_alias.py:949-965`). `select(x & 1 == 0, 1, 2) = 1 + (x & 1)`: none, proof needed (small) | emitter-only, fused with R2/R3 as `2 - (v' & 1)` (`ssa_emit.rs:1743-1747`) | goal |
| B6 | Bounds-check elimination | `t = idx < n_nodes; idx = select(t, idx, 0)` | the index range at that round | R7-index-range-by-depth (wrap at the last level, never earlier) | emitter-only (`ssa_emit.rs:1674, 1728-1736`). Analysis partly have: value sets give `2^d` values per round (measured) | goal (the canonical program already has it) |
| B7 | Multiply-add formation | `idx * 2 + off` → `multiply_add` after vectorisation | none | definition of `multiply_add` | emitter-only (`ssa_emit.rs:1747`) | goal |
| B8 | Hash FMA fusion | `(a + c) + (a << k)` | constants | R1-hash-fma-fusion | emitter-only (`ssa_emit.rs:1575-1591`) | goal |
| B9 | Stage-5 XOR deferral | the value chain | full use set; `C` odd | R2, R3 (+ needs-odd) | emitter-only (catalogue rule 2) | goal |
| B10 | SLP vectorisation | 256 isomorphic per-element chains per round → 32 chains of 8 lanes. Contiguous `inp_p + i` loads and stores → `vload`/`vstore`. Constants → broadcasts. The node load stays 8 scalar loads from a vector address (decompose) | isomorphism by structural hashing; no cross-element dataflow (def-use); contiguous constant addresses | lane-wise VALU semantics equal per-lane ALU (ISA). The equivalence checker must map 8 scalar chains to one vector chain | emitter-only (all of `ssa_emit.rs`; gather shape `ssa_emit.rs:1426-1502`). Lack as a transform | goal |
| B11 | Constant materialisation and CSE | 269 `const` loads; duplicate constants, broadcasts and addresses | constant propagation | trivial | ConstChoice encoding: have (scheduler, `emit_ssa.rs:696-...`). CSE: emitter-only (`const_cache`, `broadcast_cache`, `emit_ssa.rs:473-500, 696-700`). No CSE pass in `pipeline/passes/` | goal |
| B12 | DCE | unused results (header words, the loaded index after B6/rule 6) | none | none | have (`eliminate_ssa_dead_code`, `emit_ssa.rs:1271`; `DcePass`) | bench and goal |
| — | Canonical SSA (the catalogue's starting point) | | | | | |
| C1 | Catalogue rules 3, 5, 6, 8, 9, 10, 10a, 11; N1, N2 | see the catalogue | see the catalogue | see the catalogue | emitter-only (flags), except N1/N2 (lack) | bench |
| C2 | Stage-1 choice | site counts | the choice floor + greedy feedback | none (a choice among proved alternatives) | have for tree access only (`scripts/emission_structure.py`) | bench |
| S | Scheduling, address assignment, lowering | — | — | `kernel_dataflow` | have (`pipeline/`) | bench and goal |

**Missing from the requested list, found in the code or the measurement:**
- B0 (debug and pause);
- B1 (SSA recovery: needed because of scratch reuse, measured);
- B2 (header specialisation);
- B4 (dead stores, as distinct from forwarding);
- B7 (multiply-add formation);
- B9 (the stage-5 deferral, a representation change and not a local rule);
- B12 (DCE).

**Loop handling is not needed.** The baseline is straight-line (measured).

## 4. Where equivalence is checked, and what is uncovered

| Pass | Checked by | Uncovered |
|---|---|---|
| B1 SSA recovery | `verification/kernel_dataflow.py` (original kernel against recovered SSA), once a lifted program can be written in the scheduler-JSON form it reads | Vector-group recovery on kernels that have groups. The baseline has none |
| B2, B3, B4 memory passes | Z3 alias decisions (`memory_alias.py`); concrete check (`--concrete-runs`) | `kernel_dataflow` reads only initial memory (`Terms.mem`, `kernel_dataflow.py:140`), so it cannot compare a kernel that round-trips through memory with one that does not. It needs `SymbolicKernel` forwarding, or the SSA-to-SSA checker below |
| B5–B9 identities | Z3: MA2, R7, R1, R2, R3, R8 | B5's `select` form and B7 have no named check (small obligations) |
| B10 vectorisation | none yet | Lane mapping, scalar chains against vector chains. The SSA-to-SSA checker another agent is building (`verification/ssa_equivalence.py`, not in this branch) is the natural place |
| B11, B12 | none needed beyond term equality (CSE) and reachability (DCE) | — |
| C1 rules | Z3 R1–R15 + the rewrite certificate (design doc §6) | as the catalogue lists |
| Whole chain | `--validate` against `reference_kernel2` on random inputs; `kernel_dataflow` for the final kernel | No end-to-end proof of "baseline ≡ best". It would be the composition of the per-pass checks above |

## 5. What we deliberately defer

These are goal-only. None is needed for the benchmark number.

- B1 SSA recovery and vector-group recovery from lifted kernels.
- B3/B4 memory-to-register promotion and dead-store elimination as transforms.
- B5 if-conversion and B7 multiply-add formation as transforms.
- B10 SLP vectorisation.
- B2 header specialisation of pointers (our emitter keeps the loads).
- Discovering new rules automatically (superoptimisation or search over identities). Today
  every rule is written by hand and proved.
- Control flow. The baseline and all kernels we lift are straight-line; the 913-cycle public
  kernel's indirect jumps are out of scope (PUBLIC_KERNEL_RESCHEDULE.md).

## 6. Reproduction

```bash
# The baseline builder as of the repo's first commit (not byte-identical to upstream's file)
git show dceff11:perf_takehome.py > $SCRATCH/pt_dceff11.py

# Count ops and run it on the vendored upstream simulator; --save writes the kernel without
# debug slots and with layout metadata for the lifter. The script is a ~60-line harness that
# exec's the builder part of that file with `problem` bound to upstream/tests/frozen_problem.py.
uv run python $SCRATCH/measure_baseline.py $REPO $SCRATCH/pt_dceff11.py 16 256 --save $SCRATCH/baseline_16r256b.json
uv run python $SCRATCH/measure_baseline.py $REPO $SCRATCH/pt_dceff11.py 2 8   --save $SCRATCH/baseline_2r8b.json

# Lift and alias analysis
uv run python -m scripts.kernel_lift baseline_2r8b=$SCRATCH/baseline_2r8b.json --concrete-runs 4
uv run python -m scripts.kernel_lift baseline_16r256b=$SCRATCH/baseline_16r256b.json
```

The measurement harness was a scratch file and is not checked in. If this map turns into
work, check it in under `scripts/`, with a skill.
