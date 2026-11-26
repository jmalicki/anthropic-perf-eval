# Transform pipeline: what was built instead

Status: 2026-11-25. This note lists where the implementation departs from
[TRANSFORM_PIPELINE_DESIGN.md](TRANSFORM_PIPELINE_DESIGN.md),
[TRANSFORM_RULE_CATALOGUE.md](TRANSFORM_RULE_CATALOGUE.md) and
[BASELINE_TO_BEST_GAP.md](BASELINE_TO_BEST_GAP.md). Those three documents stay as written on
2026-11-24 (the design before implementation). Their rule numbers and coupling ids (C1-C18)
are frozen, because code comments cite them.

The implementation is the PR stack:

| PR | Branch | What it adds |
|---|---|---|
| #68 | `transform/region-tags` | Region tags on ops; merged into main |
| #71 | `transform/tree-access-costs` | Tree-access cost table derived by instantiation |
| #72 | `transform/apply-plan` | Applying a tree-access plan to the canonical program |
| #75 | `transform/hash-merge-rule` | Rule 3 |
| #77 | `transform/prune-values-rules` | Rules 5-7 and 9 |
| #80 | `transform/select-bits-mirrored-rules` | Rule 11 |
| #81 | `transform/mirror-index-rule` | Rule 8 |
| #70 | `ssa-equivalence-checker` | The SSA equivalence checker |

## Where the transforms run

- **Design (section 1):** one `transform(program, spec)` between `build()` and
  `PipelineContext::new`.
- **Built:** `TransformPass` (`pipeline/passes/transform.rs`) is the first pass inside
  `run_greedy_pipeline` and `run_cpsat_pipeline` (`pipeline/mod.rs`; on
  `transform/mirror-index-rule` at lines 254 and 309). It runs before
  `ComputeDependenciesPass`, on the program exactly as emitted.

## Tree access

- **Design:** tree access (rule 10) is found as sites in the canonical program.
- **Built:** its alternatives are instantiated by the emitter's own per-encoding builders
  (`instantiate_tree_access` in `transform/cost.rs`, #71). The emitter dispatches per
  encoding in `ssa_emit.rs` (dispatch ending around line 696 on main). A plan is applied to
  the canonical program by `apply_tree_access_plan` (#72).

## Selection

- **Design (section 4):** a stage-1 solver chooses one alternative per site, with the choice
  floor as the objective.
- **Built:** `--transform RULE` is an ordered list, and each rule rewrites every site it
  matches (`CompilerConfig::transforms`). No solver chooses per site. The strategy flags
  therefore remain, under a new name: `--transform hash-merge` stands for
  `--merge-hash-stages`, and so on.
- **Section 4's objective was found not to work as planned.** Stage-1 scores cannot rank
  plans near the optimum: the level-enumerated top-k search on `startup-and-stall-loop` (#83)
  found more than 100 plans that score at or below each best known plan.

## Evidence instead of a rewrite certificate

- **Design (section 6):** a replayable certificate of every rewrite.
- **Built:** no certificate exists. The evidence is:
  - parity tests: the rewritten program equals the flag emission op for op, after dead code
    elimination (`check_parity` in `transform/tests_index.rs`);
  - concrete interpreters: `transform/interp.rs`, and the full-memory one in
    `transform/tests_interp.rs`, compared against the reference kernel;
  - the separate SSA equivalence checker (`verification/ssa_equivalence.py`, #70);
  - `kernel_dataflow` PROVED and `--validate` PASSED on built kernels.

## Data model

- **Region:** `SsaOp::region` (a `RegionTag`, #68) instead of extending `VarSource`.
- **Rules:** the `Rule` trait (`transform/rule.rs`) has `propose` / `match_site` /
  `alternatives`, with a `Proposal` from tags or from ops alone, instead of
  matcher / precondition / builder.
- **Facts:** each fact names one of four sources (`FactSource`): `Structure`, `Program`,
  `Tag` and `Assumption`. The design had analysis, tags and assumptions. Assumptions are
  `values::Assumption` (`InitialIndicesZero`, `IndicesNotObserved`).

## Catalogue entries that turned out differently

- **Rules 8 (mirrored index) and 11 (`select_bit_source`).** The catalogue says
  "Peephole on raw SSA today: No".
  - Both were implemented as rewrites of the finished SSA: `transform/mirror_index.rs` (#81)
    and `transform/select_bits.rs` (#80).
  - The depths and bit paths they need come from bit-path forms proved on the ops (Pos/Zero
    forms, proof `R15-bit-path-steps`), not from the emitter's tags.
  - Rule 8 rewrites every index value or none.
- **Rule 7** is not a separate rule. It is the cascade of the prune-index rewrite
  (`transform/prune_index.rs`, #77).
- **The port order table** is the original plan. The order actually built:
  - tree access (#71, #72);
  - hash merge (#75);
  - prune-index and values-only (#77);
  - select-bits-path (#80);
  - mirror-index (#81).

## BASELINE_TO_BEST_GAP.md

- **"Our best kernel is about 1,100 cycles" is dated.** The best since then is 1,012 cycles
  with path bits and 1,033 with index bits: greedy plus justification, #93.
- **Row C1's "emitter-only (flags)" no longer holds for rules 3, 5-9 and 11.** Those are
  transforms on finished SSA (#75, #77, #80, #81).
