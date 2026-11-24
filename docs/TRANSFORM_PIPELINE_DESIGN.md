# Transform pipeline: from an emitter with strategy flags to a compiler that rewrites

**Conclusion.** The kernel generator should emit one canonical SSA program and reach every
current emission option by proved rewrites chosen by a small solver. The current flags turn
out to be four kinds of thing, and only two of them are rewrites in the usual sense:

- **Local rules.** Per-site rewrites, such as the hash stage merge and the depth-0 index
  update. Port them first.
- **Gated local rules.** Per-site alternatives that are only valid, or only priced, under a
  global choice. Tree access is the main one: its alternatives depend on the index form and
  on where the branch bits come from.
- **Global representation choices.** The mirrored index, and the stage-5 XOR deferral. Each
  changes every op of a chain.
- **Output-contract assumptions.** `values_only`, which drops the index stores and leaves the
  rest to dead code elimination.

Some flags are not rewrites at all: `tree_reduce_max_depth`, `depth4_select_rounds`,
`access_plan` and `scripts/emission_structure.py --fix` are restrictions on, or the output
of, the stage-1 choice. ConstChoice, SelectChoice and VALU shift stay in the scheduler.

The catalogue that backs these classes is
[TRANSFORM_RULE_CATALOGUE.md](TRANSFORM_RULE_CATALOGUE.md). It lists every branch with
file:line, every coupling, and the proof ids. This document describes the mechanism.

Status: design. Nothing here is implemented except where a file is cited. Statements about
current code cite file:line. Statements marked *(inferred)* are my reading, not checked by
running anything.

## 1. Pipeline

```
  SsaTickTockKernelBuilder::build()          canonical emission (one program, no strategy flags)
            │  SsaProgram + VarSource tags on every op
            ▼
  [T1] DetectSites        tags narrow candidates; matcher checks the shape structurally
  [T2] Instantiate        for each site, build every alternative's replacement; count ops per
                          engine class; record shared resources the replacement needs
  [T3] Stage-1 select     CP-SAT over class counts, objective = choice floor
                          (EMISSION_STRUCTURE_MODEL.md §4); greedy feedback loop
  [T4] Apply              apply chosen replacements, insert resources at fixed anchors
  [T5] DCE                existing eliminate_ssa_dead_code
            │  SsaProgram (rewritten)
            ▼
  PipelineContext::new(program, config)
  run_greedy_pipeline / build_and_schedule_cpsat    stage 2, unchanged
```

**Where it sits in the code.** Both scheduling entry points build the program and hand it to
`PipelineContext::new`: `build_and_schedule_greedy` at
`scheduler_rs/src/kernel_builder/tick_tock_ssa/mod.rs:782`, and `build_and_schedule_cpsat` at
`tick_tock_ssa/mod.rs:921`. The greedy path then runs `run_greedy_pipeline`
(`pipeline/mod.rs:248`). The CP-SAT path runs its passes one at a time. `run_cpsat_pipeline`
(`pipeline/mod.rs:296`) exists, but the only caller found was the greedy one
(`tick_tock_ssa/mod.rs:800`).

T1 to T5 belong between `self.build()` and `PipelineContext::new`, as one function
`transform(program, spec) -> SsaProgram` that both entry points call. They should not run
inside `run_*_pipeline`, for two reasons:

- the pipelines begin with `ComputeDependenciesPass` and DCE on a fixed op list
  (`pipeline/mod.rs:249-253`);
- stage 1 needs to run the greedy pipeline on candidate programs, which is a call into
  `run_greedy_pipeline` on a cloned context.

After T5, stage 2 sees an ordinary `SsaProgram`. It cannot tell a rewritten program from one
emitted under today's flags.

## 2. Data model

| Term | Meaning | Today's nearest thing |
|---|---|---|
| **Region tag** | `(phase, round, tree_depth, iter, group)` on every op | `VarSource` (`types.rs:1340-1351`), set by `set_phase`/`set_round_context`/`set_iter_group` (`emit_ssa.rs:250-273`) |
| **Site** | One occurrence of a pattern: a set of op ids, its input VRegs (live-in), its output VRegs (live-out), and the facts it was matched under | none |
| **Rule** | `id`, `matcher(program, candidate) -> Option<Site>`, `precondition(site, facts) -> bool`, one or more `builder(site, resources) -> Replacement`, `proof_ids: [&str]` naming checks in `verification/rewrites.py` | the `if self.config.spec.<flag>` branches in `ssa_emit.rs` |
| **Alternative** | One builder's output for one site: new ops, the map from old live-outs to new VRegs, the resources it needs, and per-engine-class op counts | the per-encoding emitters (`emit_cached_level_select`, `emit_scattered_access`, ...) |
| **Resource** | Setup ops shared across sites, paid once if any chosen alternative needs them. Resources form a DAG: a resource can feed other resources | node table, leaf pairs, Mobius coefficients (`emit_cached_select_init`, `ssa_emit.rs:922-999`), derived constants `c[3..15]` (`ssa_emit.rs:242-265`), level loads (`ssa_emit.rs:335-344, 432-441, 493-502`) |
| **Global choice** | A choice that changes every site of a class at once, enumerated outside the stage-1 model as a context | `--index-form`; `scripts/emission_structure.py` already treats it as a context (`emission_structure.py:1036`) |
| **Assumption** | A fact taken from the problem contract, not derived from the program: "initial indices are 0", "the index output is not observed" | comments at `ssa_emit.rs:568-572` and `config.rs:298-302`; preconditions listed in `verification/memory_alias.py` docstring |

**Tags narrow candidates; they never decide.** A matcher gets candidates from the region
tags, for example all ops with `phase = Hash, round = r, group = g`. It then checks the shape
structurally: opcodes, operand wiring, constant operand values, and that the live-outs are
what the rule expects. If a tag is wrong, the matcher fails. It never silently rewrites the
wrong ops. This keeps the rules usable on input that has no tags (section 7).

**Facts.** Some preconditions are not local shape. Examples: "this index is at depth 3",
"this condition is 0/1", "this index is 0". Rules state them as named facts. Each fact comes
from one of three places:

- an analysis on plain SSA: constant propagation, interval range, boolean forms, def-use
  path recovery;
- the emitter's tags (a stepping stone);
- an assumption (the problem contract).

The catalogue lists the facts each rule needs and where each fact comes from today.

## 3. Cost by instantiation

Costs are counted on the instantiated alternative, not on a hand-written formula. The
existing calibration shows why. `scripts/emission_structure.py` fits integer coefficients
exactly, but it needs four product terms (`and:table:2+table:3`, ...). These exist because
the scalar constants 3 to 15 feed the node addresses of levels 2, 3 and 4. Whether a
constant survives DCE is therefore an OR over the levels that use it
(EMISSION_STRUCTURE_MODEL.md §3; code at `ssa_emit.rs:242-265, 335-344, 427-441, 474-491`).

Instantiation handles this directly:

1. Build each alternative once per site class. Sites of a class differ only in round, group
   and live-in VRegs, so one instantiation per class is enough *(inferred; to be checked by
   instantiating two sites of each class and comparing counts)*.
2. Represent each resource as a node in a DAG of setup ops. A resource's cost is counted
   once, after DCE of the setup DAG restricted to the chosen alternatives. In the stage-1
   model this becomes one boolean per resource and one per OR-node (the product terms of
   today's model).
3. Count by engine class as in EMISSION_STRUCTURE_MODEL.md §3: `vec_fixed`,
   `vec_shiftable`, `alu`, `load`, `store`, `flow`, `select1`, `select2`, `const`,
   `const_alu`. These are the categories the choice floor resolves.

`values_only` and index-form costs are not additive per site: what DCE removes after the
index stores are dropped depends on which tree-access encodings still read the index in the
last rounds (catalogue, coupling C3). These stay contexts, as `emission_structure.py` treats
them today (`--values-only`, `--index-form` at `emission_structure.py:1036-1038`). Within a
context, the per-site counts are additive once resources are separated, as the exact fits in
EMISSION_STRUCTURE_MODEL.md §3 show.

## 4. Stage 1: choosing alternatives

- **Variables.** For each site class and alternative, the number of sites using it. Sites
  in one class are interchangeable for the floor, which is why the model "collapses to
  counts". Plus one boolean per resource and per resource OR-node.
- **Objective.** `choice_floor` (EMISSION_STRUCTURE_MODEL.md §4): the smallest `T` such that
  some resolution of every ConstChoice and SelectChoice and of the shifted vector ops fits.
  The scheduler-level choices are resolved only inside the floor's capacity rows. Stage 1
  does not fix them.
- **Tie-breaks and caps.** These are as today: fewest encodings mixed per round, select work
  at a level's earliest visit, and optional `--max-util`.
- **Pins.** `--fix ROUND=ENCODING` (`emission_structure.py:1059-1076`) and
  `depth4_select_rounds=after-wrap` (`config.rs:396-400`) become equality constraints on
  site counts.
- **Greedy feedback.** The floor ignores time. Two plans with the same floor differed by 100
  greedy cycles (EMISSION_STRUCTURE_MODEL.md §5). The loop: solve for the floor; apply, DCE,
  and run `run_greedy_pipeline` on the candidate; add a no-good or a cap (for example
  `--max-util flow=F`) and repeat within a budget; keep the best greedy result. The design of
  this loop is in another agent's scope; this document only fixes its interface: a candidate
  is a map from site-class alternatives to counts, plus which concrete sites take which
  alternative.
- **Assigning counts to concrete sites.** Assignment is deterministic: lowest group first
  within a round. This matches today's emission plans (`solver/emission_plan.py`), so a
  count vector reproduces an existing plan's SSA program.

## 5. What stays in the scheduler, and why

| Choice | Where today | Why it stays |
|---|---|---|
| ConstChoice (load / add_imm / ALU-derived) | `emit_const_choice`, `emit_ssa.rs:696-...`; CP-SAT | Engine choice per op with no effect on dataflow. It interacts with slot timing, which only the scheduler sees |
| SelectChoice (flow `vselect` / FMA) | `emit_select_choice*`, `emit_ssa.rs:919-1052` | Same. Stage 1 only needs the class (`select1` / `select2`) to compute the floor |
| VALU→ALU shift | scheduler | Same |
| SelectChoice CSE groups (one shared SUB for selects with the same pair) | CP-SAT option, `python.rs:1068-1070` | Depends on which SelectChoices go to FMA |
| vbroadcast elimination | experimental scheduler flag | Same |

The rule of thumb: if two alternatives produce the same SSA dataflow and differ only in
which engine runs an op, the scheduler decides. If they produce different dataflow, it is a
rewrite.

Builders must keep the scheduling metadata the emitter attaches. Otherwise stage 2 sees a
different program even when the ops are the same:

- `VarType` (`set_next_var_type`);
- symmetry hints on select trees (`set_symmetry_hint`, `emit_ssa.rs:298-306`), which feed
  the declared CP-SAT symmetry rows;
- `keep_live` on the final XOR (`ssa_emit.rs:49-52, 73-78`);
- trace ids (`ssa_emit.rs:1577-1600`);
- barriers between rounds (`ssa_emit.rs:56-58`);
- `MemLocation` on every memory op (`memory_deps.rs:6-8` treats a missing one as an error).

## 6. Verification obligations

Three layers. Each layer is checked independently.

1. **Per-rule identity (Z3).** Each rule names the checks in `verification/rewrites.py` that
   prove its replacement equal to the matched ops under the rule's preconditions. Each
   precondition has a matching `*-needs-*` check where one exists, showing that the
   precondition is necessary. The existing ids are R1 to R15 (`rewrites.py:597-620`). A rule
   without an id is not ported until one is added (catalogue: "none: proof needed", with the
   obligation stated).
2. **Transformed SSA equals canonical SSA.** No such checker exists today.
   `verification/kernel_dataflow.py` compares a kernel against *its own* SSA program, using
   one hash-consed term store with a few normalisations (R4, R5: `Terms.fma`, `Terms.binop`,
   `kernel_dataflow.py:85-140`). It does not equate, for example, a merged hash with an
   unmerged one, because the terms differ. Two options; both should exist:
   - **Rewrite certificate (primary).** T4 writes a log of
     `(rule id, site ops, replacement ops, instantiated constants, facts used)`. A checker
     replays the log on the canonical program. For each entry it re-runs the matcher, checks
     that the replacement is exactly the builder's output for those constants, checks each
     fact against its source (analysis result or declared assumption), and checks that the
     instantiated constants are the ones the Z3 check was run with. This is sound if the Z3
     identities hold and the matcher is exact. Every step is mechanical.
   - **Differential (backstop).** `--validate` against `reference_kernel2` on random inputs
     (exists). Under `values_only` it compares values only.
3. **Kernel equals rewritten SSA.** `verification/kernel_dataflow.py` (exists; the
   `use-kernel-dataflow` skill). This is unchanged by the transform pipeline.

**Assumptions are explicit, not proved.** Each run records which assumptions its chosen
rules used (initial indices 0, index output not observed, tree memory may be overwritten).
The checker refuses a certificate that uses an assumption the run did not declare.

**Memory.** Every rule that adds or moves a memory op must add exactly the memory edges that
can alias, each proved with Z3 as in `verification/memory_alias.py`. It must never order
memory blanket-style. Today's `MemLocation::may_alias` is region equality plus interval
overlap (`types.rs:1583-1590`). That is exact for the current kernels, which store only
indices and values after their last reads (`memory_deps.rs:21-23`). It is not enough for the
two memory rules we lack (catalogue section "Rules we do not have yet").

## 7. Path to untagged input

The long-term goal is to transform arbitrary straight-line assembly, for example a kernel
lifted by `scripts/kernel_lift.py` with no emitter behind it. Emitter tags are a stepping
stone. Matchers verify shape structurally, so the gap is the facts and the SSA structure,
not the matching.

**What a lifted kernel gives us today** (read from `scripts/kernel_lift.py:1-25, 83-125` and
`verification/memory_alias.py:1-60, 554-566`):

- One `KOp` per slot: engine, the raw slot, scratch words read and written, and for memory
  ops the address term of the first word and the width.
- Scratch edges per word: RAW from the last writer, WAR and WAW. Lanes are exact.
- Memory edges from `memory_alias`. Each memory address is a symbolic term over the input
  (tree values, input values), with header words folded to constants under the generator's
  preconditions. Each pair is decided by value sets or Z3.
- Fixed scratch addresses and the original cycle of every op.
- Not given: SSA values (a word is reused by many values), vector groups (which 8 words form
  one value, which scalar writes assemble a vector), round, group or depth structure,
  `MemLocation` regions, ConstChoice/SelectChoice alternatives, and any notion of which
  memory is output.

**What the rules need**, and the analysis that would supply each:

| Need | Analysis | Existing pieces |
|---|---|---|
| SSA values | Straight-line renaming: each write of a word defines a new value. A read uses the last writer, which is already the RAW edge | `scratch_edges` in `kernel_lift.py:83-115` |
| Vector values and vector groups | A vector read whose 8 words have one writer (a vector op) is a vector value. If the 8 words have different scalar writers, it is a write group (`fresh_vector_group`). Scalar reads of lanes of one vector value form a read group (`decompose_vector`) | per-word last-writer data in the lift; group semantics in `emit_ssa.rs:1060-1110` |
| Constant operands | Constant propagation through `const`, `add_imm`, ALU arithmetic and `vbroadcast` | `Terms` folding (`kernel_dataflow.py:85-104`); `ValueSets` (`memory_alias.py:222`) |
| A condition is 0/1 | The boolean forms of R6: comparison, `& 1`, select of 0/1 values | `Terms.is_bool` (`kernel_dataflow.py:106`) |
| Index depth, index range | Interval analysis on the index chain from `idx = 0`: `idx' = 2 idx + {1, 2}` gives exactly the R7 ranges, and the mirrored form gives R11's. Needs the input assumption at the start | R7, R11 prove the ranges. `ValueSets` covers `& c`, `% c`, `>> c`, not affine chains *(inferred from its docstring)* |
| Which ops form a hash, an index step, a gather | Pattern matchers with constant operands, anchored on the hash constants (`HASH_STAGES`) and on loads whose address term is `forest_p + idx` | memory address terms from `memory_alias` |
| Round and group | Count hash chains along the def path from the value load (round); the input vload's offset gives the group | none |
| Branch bits of earlier rounds | Def-use along the index chain: the bit operand of each update | none (the emitter keeps `path_bits`, `ssa_emit.rs:1656-1665`) |
| Values flowing through memory (the 880-cycle kernel's store-select and repack) | Store-to-load forwarding over symbolic memory | `SymbolicKernel` exact memory model (`memory_alias.py:592`) |
| Memory regions for `MemLocation` | Classify each address term by base pointer and value set | value sets in `memory_alias` |
| Output observed or not | Not derivable. Comes from the harness contract | none |

**Order of work** *(inferred)*:

1. SSA renaming plus vector-group recovery over a lifted kernel, producing an `SsaProgram`.
   Check it by running `kernel_dataflow` on the original kernel against the recovered
   program.
2. Constant propagation and boolean forms (mostly exist).
3. Interval analysis for index depth.
4. Store-to-load forwarding, so memory-based selects are seen as selects.

Rules that need only local shape plus constants can run on lifted input after step 2. In the
catalogue these are marked "peephole on raw SSA": hash stage merge and FMA fusion. Rules
that need depth or bit-path facts need step 3 and the def-use path recovery.

## 8. Known hard parts

1. **Parity is not byte-for-byte for free.** Rewrites allocate new VRegs and op ids. Two
   things depend on op order:
   - The greedy scheduler (FIFO priority).
   - ConstChoice alternatives. `emit_const_choice` computes ALU-derived alternatives from
     the constants emitted *before* it (`emit_ssa.rs:703-711`). Adding or removing a
     constant changes the alternatives of later ones.

   Gate parity on the existing `program(sched)` comparison
   (`tests/test_emission_plan.py:48-69`, the SSA ops minus cycles), after renumbering VRegs
   and op ids in emission order. Have builders insert resources at fixed anchors equal to
   the emitter's positions. Where that is impossible, state the difference and fall back to
   equal per-engine counts plus PROVED and PASSED.
2. **The canonical form needs one emitter change.** Gather at depth 0 is never emitted
   today. `check_plannable` rejects it (`config.rs:221-223`), and the flag path always
   selects at depth 0 (`use_tree_reduce_at_depth(0)` is always true, `config.rs:382-384`).
   `emit_scattered_access` would accept it in the zero-based form (`ssa_emit.rs:686`), but
   its correctness there is not determined. With the mirrored index there is no index
   vector at depth 0 (`all_idx` is `None`, `ssa_emit.rs:1709-1711`), so canonical emission
   must be zero-based.
3. **The canonical form already carries one assumption.** The fixed wrap round with no
   bounds check (R7) relies on every initial index being 0. That is R7's base case, the
   same precondition `memory_alias.py` states, and the comment at `ssa_emit.rs:568-572`. So
   "initial indices 0" is a contract assumption of the canonical program, not something a
   rule adds.
4. **Global choices interact with the tree-access rule.** The mirrored index changes the
   gather address op, the compare constants, the level-position op and the bit extraction.
   It also makes `prune_dead_index` a no-op. Keep it a context (two stage-1 solves), not a
   variable.
5. **Mixed encodings change op order.** `emit_round` emits one encoder call per encoding in
   order of first use (`ssa_emit.rs:664-690`). A rewrite that replaces sites in place gives
   a different order. That is greedy-visible, so it needs an anchor rule, or a parity gate
   that accepts the reorder.
6. **Scratch is not in the floor.** `select_bit_source=path` costs 0 ops and 8 words per
   group per kept bit. Stage 1 cannot see that. A scratch-pressure term or the greedy
   feedback has to.
7. **Memory rules need IR the SSA lacks.** Data-dependent `vload` addresses, set-valued
   memory offsets for exact alias edges, a temporary memory region that exists in the
   upstream memory image, and kernel-level verification of store-to-load flow (the
   `kernel_dataflow` term store has `mem(addr)` over initial memory only, `Terms.mem`,
   `kernel_dataflow.py:140`). See the catalogue.
