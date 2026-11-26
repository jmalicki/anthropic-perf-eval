# Transform rule catalogue

> **Status.** Written 2026-11-24 as the design before implementation. It is kept as written:
> the rule numbering and the coupling ids (C1-C18) are frozen, because code comments on the
> rule branches cite them. What was built differs in places; see
> [TRANSFORM_PIPELINE_AS_BUILT.md](TRANSFORM_PIPELINE_AS_BUILT.md).

This catalogue lists every strategy branch in today's emitter, read from the code, written
as a rule for the transform pipeline ([TRANSFORM_PIPELINE_DESIGN.md](TRANSFORM_PIPELINE_DESIGN.md)).

Files read:
- `scheduler_rs/src/kernel_builder/ssa_emit.rs`
- `tick_tock_ssa/mod.rs`
- `tree_select.rs`
- `config.rs`
- `emit_ssa.rs` (the emit helpers)
- `kernel_build_args.py`
- `scripts/emission_structure.py`
- `verification/rewrites.py`

Paths below are relative to `scheduler_rs/src/kernel_builder/` unless they start with a
top-level directory.

**Marks.** A claim with a file:line is read from code. *(inferred)* marks my reading. Nothing
here was measured by running anything; measured numbers are quoted from the docs named.

**Problem shape for the site counts.** 16 rounds, batch 256 (32 groups, `n_iters = 4`,
`UNROLL = 8`), forest height 10, so tree height 11. Round `r` visits depth `r % 11`
(`config.rs:404-406`):

| Round | 0 | 1 | 2 | 3 | 4 | 5–9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Depth | 0 | 1 | 2 | 3 | 4 | 5–9 | 10 (wrap) | 0 | 1 | 2 | 3 | 4 |

The wrap round is the one at depth `forest_height` = 10 (`ssa_emit.rs:1674`). The last round
is 15.

**Op notation.** `V` is a vector ALU op (`vec_shiftable`). `F` is multiply_add, `B` is
vbroadcast (`vec_fixed`). `A` is a scalar ALU op. `L` is a load, `S` a store.
`sel1`/`sel2` is a SelectChoice whose FMA form is 1 op or 2 ops (SUB + multiply_add).
`K` is a ConstChoice. These are the classes of EMISSION_STRUCTURE_MODEL.md §3.

**Fields of each entry.**
- **Branches**: file:line of every branch.
- **Site**: what a site is, and how many there are at 16 rounds / 32 groups.
- **Canonical** and **Alternatives**: op lists.
- **Preconditions**.
- **Facts beyond local shape**: each fact with its source, either an analysis on plain SSA
  (named) or emitter-only today.
- **Peephole on raw SSA today**: whether the rule matches with nothing but the ops and
  constant tracking.
- **Proofs**: ids in `verification/rewrites.py`.
- **Resources**: shared setup the alternatives need.
- **Class**.

---

## Classification summary

| # | Rule | Class | Sites (16 r, 32 g) | Proof |
|---|---|---|---|---|
| 1 | Hash FMA fusion (stages 0, 2, 4) | Local rule, already applied in canonical | 1,536 (512 hashes × 3 stages) | R1-hash-fma-fusion |
| 2 | Stage-5 XOR deferral (optimised-form value) | Global representation choice, already applied in canonical | 32 chains (entry, 512 stage-5s, every node XOR, branch, exit) | R2-stage5-xor-deferral, R3-branch-offset (+ needs-odd) |
| 3 | Hash stage 2+3 merge | Local rule | 512 | R8-hash-stage-merge (+ needs-left-shift) |
| 4 | Bounds check and wrap from depth (no compare against `n_nodes`) | Already applied in canonical (not a rewrite today) | 512 index steps | R7-index-range-by-depth |
| 5 | Wrap-round zero vector shared (prune, part 1) | Local rule (CSE) | 32, in round 10 | none needed beyond CSE (see entry) |
| 6 | Depth-0 index update is its offset (prune, part 2) | Local rule with a fact (index = 0) | 64, in rounds 0 and 11 | R12-index-update-from-root (+ needs-zero) |
| 7 | Initial index load removed | Not a rewrite: DCE consequence of 6, 8, 10 | 32 | — |
| 8 | Mirrored index `n = L_d - idx` | Global representation choice | 32 chains (512 steps, 320 cached-level reads, 192 gathers, 32 final stores) | R9, R10, R11 (+ R9 needs-odd) |
| 9 | `values_only` | Output-contract assumption | 32 index stores | none (output changes; see entry) |
| 10 | Tree access encoding (gather / standard / bits / bits-flow / mobius) | Gated local rule | 512 (320 at depths 0–4, of which 256 at depths 1–4 have a real choice; 192 gather only) | R13, R14, R4, R5, R6, R2 (+ needs-*) |
| 10a | Root vector shared at depth 0 (part of 10) | Gated local rule | 64 | R2 (2nd obligation), R7 base |
| 11 | `select_bit_source` (index / path) | Gated local rule (gated on 10 being bit-driven at depth ≥ 3) | up to 128 (rounds 3, 4, 14, 15) | R15-bits-from-index (+ needs-range) |
| 12 | `tree_reduce_max_depth` | Not a rewrite: restricts 10's alternatives per depth | — | — |
| 13 | `depth4_select_rounds` | Not a rewrite: stage-1 pin | 32 (round 4) | — |
| 14 | `emission_structure.py --fix` | Not a rewrite: stage-1 pin | per round | — |
| 15 | `access_plan` / `--emission-plan` | Not a rewrite: the stage-1 output | — | — |
| 16 | Depth-3 compare-chain path (no tree-reduce) | Not a rewrite: dead code | 0 | — |
| 17 | Partition stores, lazy partition, pseudo consts | Not rewrites: disabled or unused | 0 | — |
| 18 | ConstChoice, SelectChoice, VALU shift, CSE groups | Stay in the scheduler | — | R4, R5 (SelectChoice) |
| 19 | Barriers, `keep_live`, symmetry hints, trace ids | Not rewrites: scheduling metadata that builders must keep | — | — |
| N1 | Tree repack + packet `vload` gather | Not implemented: gated local rule + global memory resource | see entry | none: proofs needed |
| N2 | Store-select through memory | Not implemented: local rule (scheduler-level variant possible) | see entry | none: proofs needed |

---

## 1. Hash FMA fusion

- **Branches.**
  - `config.rs:935-948` (`can_use_fma`, `fma_multiplier`).
  - `tick_tock_ssa/mod.rs:681-687` (per-stage multipliers into `ctx.hash_fma_mult`).
  - `ssa_emit.rs:394-404` (multiplier constants).
  - `ssa_emit.rs:299-305, 307` (stages 0, 2 and 4 get `v_hc` but no `v_hs`).
  - `ssa_emit.rs:1575-1591` (emission).
  - There is no flag: it is always on.
- **Site.** One hash stage of form `(a + c) + (a << k)` for one group in one round: stages 0,
  2 and 4, so 3 × 512 = 1,536.
- **Canonical (reference form).** `t = V+(a, c)`, `s = V<<(a, k)`, `r = V+(t, s)`: 3 V.
- **Alternative.** `r = F(a, 1 + 2^k, c)`: 1 F.
- **Preconditions.** The stage has the operators `('+', '+', '<<')` (`config.rs:935-938`).
- **Facts beyond local shape.** The constant values of `c` and `k`. Source: constant
  propagation (ConstChoice → vbroadcast already carries `produces_constant`,
  `emit_ssa.rs:484-490`). No emitter-only fact.
- **Peephole on raw SSA today.** Yes, given constant tracking. It never fires on today's
  canonical program, which is already fused. It matters for lifted or reference-form input.
- **Proofs.** `R1-hash-fma-fusion` (`rewrites.py:81-94`).
- **Resources.** Multiplier constant `1 + 2^k` (1 K + 1 B per stage, shared by all sites);
  the shift constant is then dead.
- **Class.** Local rule. Kept applied in the canonical program (the "11-op hash").

## 2. Stage-5 XOR deferral

- **Branches.**
  - Constant and broadcast: `ssa_emit.rs:327-331`.
  - Entry conversion `val ^ C` per group: `ssa_emit.rs:599-605`.
  - Stage 5 as `a ^ (a >> 16)` without the constant: `ssa_emit.rs:1592-1608`.
  - Node XOR with `C` in every non-bit-driven access: `ssa_emit.rs:1151, 1177, 1206, 1283,
    1320, 1420, 1497`.
  - Bit-driven node tables pre-XORed: `ssa_emit.rs:949`.
  - Branch offset `2 - bit`: `ssa_emit.rs:1746`.
  - Exit conversion: `ssa_emit.rs:67-81`.
- **Site.** The whole value chain of one group: 32 chains.
- **Canonical (reference).** Stage 5 is 3 V. The node XOR is 1 V. The branch is the
  reference parity.
- **Alternative (today's canonical).**
  - Stage 5 is 2 V.
  - The node is pre-XORed with `C`: 1 V per group in gather/standard, 0 in bit-driven
    tables (paid once per table entry as A).
  - Branch bit inverted (`2 - bit`).
  - +1 V per group at entry and +1 V per group at exit.
- **Preconditions.** `C` is odd, which R3 and R9 need (`0xB55A4F09` is odd). Every use of the
  hashed value is one of: the next round's node XOR, the branch bit, or the exit.
- **Facts beyond local shape.** The full use set of the value chain: def-use on plain SSA
  over all rounds. `C` odd: a constant. Derivable on SSA, but not a local match.
- **Peephole on raw SSA today.** No (global).
- **Proofs.** `R2-stage5-xor-deferral` (stage, node XOR, exit), `R3-branch-offset`,
  `R3-branch-offset-needs-odd-constant` (SAT).
- **Resources.** `C` scalar and broadcast.
- **Class.** Global representation choice. Recommended: keep it in the canonical program and
  do not port it. Rules 6, 8 and 10 rely on it (couplings C11, C12).

## 3. Hash stage 2+3 merge (`--merge-hash-stages`)

- **Branches.**
  - Flag `config.rs:286-289`.
  - Constants skipped `ssa_emit.rs:294-298`.
  - Merged constants `ssa_emit.rs:406-419`, from `config.rs:950-978`.
  - Stage 3 skipped `ssa_emit.rs:1542-1545`.
  - Merged emission `ssa_emit.rs:1556-1565`.
  - CLI `kernel_build_args.py:220-222`.
- **Site.** Stages 2 and 3 of one group in one round: 512.
- **Canonical.** `x = F(a, m, C2)`; `t = V+(x, C3)`; `s = V<<(x, 9)`; `r = V^(t, s)`: 1 F + 3 V.
- **Alternative.** `l = F(a, m1, c1)`; `h = F(a, m2, c2)`; `r = V^(l, h)`: 2 F + 1 V. One op
  fewer per site. Measured 511 vector-op equivalents fewer at 16 rounds
  (BOUND_GAP_ANALYSIS.md §4.1).
- **Preconditions.** Stage 2 fuses (R1), and stage 3 has the form `(x + d) ^ (x << k)`, left
  shift only (`config.rs:966-978`). The constants are `m1 = m`, `c1 = c + d`, `m2 = m << k`,
  `c2 = c << k` mod 2^32.
- **Facts beyond local shape.** The constant values of the four vector operands, from
  constant propagation. Whether `x` has other users: not needed for correctness (keep `x` if
  it has), but needed for cost.
- **Peephole on raw SSA today.** Yes. The pattern is 4 ops with constant operands and no
  index or depth fact. *(inferred: no matcher exists; the shape is fixed by
  `ssa_emit.rs:1575-1636`.)*
- **Proofs.** `R8-hash-stage-merge` (`rewrites.py:116-152`); `R8-hash-stage-merge-needs-left-shift`
  (SAT).
- **Resources.** 3 K + 3 B for `c1, m2, c2`. When no site keeps the canonical form, the
  canonical's `C2`, `C3` and shift-9 constants and broadcasts become dead; the stage-2
  multiplier stays.
- **Class.** Local rule. Mixing merged and unmerged sites is valid; the emitter does all or
  none.

## 4. Bounds check and wrap from depth

- **Branches.**
  - Wrap only in the round at depth `forest_height`: `ssa_emit.rs:1674, 1704-1706, 1728-1736`.
  - No compare against `n_nodes` anywhere in the index step: `ssa_emit.rs:1743-1759`.
  - Tree depth counter with wrap: `ssa_emit.rs:60-64`.
- **Site.** Index step: 512.
- **Canonical (reference).** `idx' = 2 idx + off`; `idx'' = idx' < n_nodes ? idx' : 0`.
- **Alternative (today's canonical).** No compare. In the wrap round, `idx'' = 0`.
- **Preconditions.** Initial index 0 (R7's base case).
- **Facts beyond local shape.** The index range at each depth. Derivable by interval analysis
  from `idx = 0` (exactly R7's induction), given the input assumption.
- **Peephole on raw SSA today.** No.
- **Proofs.** `R7-index-range-by-depth`.
- **Resources.** None.
- **Class.** Already applied in canonical (not a rewrite today). It is the source of the
  canonical program's one input assumption (design doc §8.3).

## 5. Wrap-round zero vector shared (`--prune-dead-index`, part 1)

- **Branches.** `ssa_emit.rs:1728-1736` (`prune`: `v_constant(0)`, shared and CSE'd through
  `broadcast_cache`; otherwise `emit_vbroadcast_force(zero)` per group). The flag is at
  `config.rs:292-297`; the CLI at `kernel_build_args.py:229-234`.
- **Site.** One group's index in the wrap round: 32 (round 10).
- **Canonical.** 1 B per group (`emit_vbroadcast_force`, which deliberately skips CSE,
  `emit_ssa.rs:503-520`).
- **Alternative.** One shared B for all groups.
- **Preconditions.** None beyond identical op and operand.
- **Facts beyond local shape.** None.
- **Peephole on raw SSA today.** Yes (CSE of identical `vbroadcast(zero)`).
- **Proofs.** None needed beyond CSE: two `vbroadcast` ops of the same scalar define the same
  term (in `kernel_dataflow`'s hash-consed store they are one term). Why `_force` was used
  here originally is not determined.
- **Resources.** None.
- **Class.** Local rule.

## 6. Depth-0 index update is its offset (`--prune-dead-index`, part 2)

- **Branches.** `ssa_emit.rs:1737-1741` (`prune && tree_depth == 0`), against
  `ssa_emit.rs:1742-1759`.
- **Site.** One group's index step at depth 0: 64 (rounds 0 and 11).
- **Canonical.** `bit = V&(v, 1)`; `off = V-(2, bit)`; `idx' = F(idx, 2, off)`: 2 V + 1 F.
- **Alternative.** `bit`, `off`; `idx' = off`: 2 V.
- **Preconditions.** `idx = 0` on every lane.
- **Facts beyond local shape.**
  - `idx = 0` in round 11: derivable by constant propagation. The wrap round writes a
    broadcast of the zero VReg, which carries constant 0 (`emit_ssa.rs:191-192, 507-517`).
    `F(0, x, y) = y` is then constant folding.
  - `idx = 0` in round 0: only from the input assumption (initial indices 0). The canonical
    program reads the loaded index there.
- **Peephole on raw SSA today.** Round 11: yes, with constant propagation. Round 0: no, it
  needs the declared assumption.
- **Proofs.** `R12-index-update-from-root`; `R12-index-update-from-root-needs-zero` (SAT).
- **Resources.** None.
- **Class.** Local rule with a fact. Applies to the zero-based form only: the mirrored branch
  returns before it (`ssa_emit.rs:1699-1715`).

## 7. Initial index load removed

- **Branches.** `ssa_emit.rs:573-597`: `load_idx = !mirrored && !(prune && rounds > 0)`. This
  drops the per-group index address A and the index `vload`.
- **Site.** 32 groups.
- **Class.** Not a rewrite. Once rules 6/8/10 leave no reader of the loaded index (round 0's
  update and round 0's tree access), DCE removes the load and its address add. With
  canonical gather at depth 0, round 0's access reads the index, so rule 10 at depth 0 must
  also have been applied (coupling C6). The parity of op order is noted at
  `ssa_emit.rs:576-578`.

## 8. Mirrored index (`--index-form mirrored`)

- **Branches.**
  - Enum `config.rs:30-46`, predicate `ssa_emit.rs:111-113`.
  - Initial index not loaded: `ssa_emit.rs:568-574`.
  - Index step: `ssa_emit.rs:1699-1715`.
  - Compare constants `L_d - k`: `ssa_emit.rs:156-167`, used at `1159, 1191-1192`.
  - Level position `first - n`: `ssa_emit.rs:179-190`, used at `1250, 1364`.
  - Gather address `(forest_p + L_d) - n`: `ssa_emit.rs:1449-1477`.
  - Final store `L_d - n`: `ssa_emit.rs:201-213`, used at `90-94`.
  - Bit extraction skips its mirroring SUB: `ssa_emit.rs:1024-1030`.
  - Incompatible with partition stores (`ssa_emit.rs:1766-1769`) and with the dead depth-3
    path (`ssa_emit.rs:1289-1292`).
  - CLI `kernel_build_args.py:223-228`.
- **Site.** One group's whole index chain: 32 chains. They touch 512 index steps, every
  index read by tree access (320 at depths 0–4; standard reads at depths 1–4 and bit
  extraction at depths 3–4), 192 gathers at depths 5–10 (plus any gathered at 1–4), and 32
  final stores.
- **Canonical (zero-based, per group).**
  - Step: 2 V + 1 F (rule 6 alternative at depth 0: 2 V). Wrap: 1 B.
  - Gather address: `V+(idx, B(forest_p))`, with the broadcast shared globally.
  - Final store: stores `idx`.
- **Alternative (mirrored, per group).**
  - Step: `bit = V&(v, 1)`; `n' = F(n, 2, bit)`, so 1 V + 1 F. At depth 0, `n' = bit` (1 V).
    In the wrap round, nothing (`n = None`).
  - Gather: `V-(B(forest_p + L_d), n)`, plus per gathered round 1 K + 1 A + 1 B for the base.
    The A and the B are not CSE'd between the two visits of a level: `emit_alu` has no
    cache (`emit_ssa.rs:413-424`); `ssa_emit.rs:1451-1456`.
  - Compares: against `L_d - k` instead of `k`.
  - Level position: `V-(first, n)`.
  - Final store: `V-(B(L_d), n)` (+1 V per group).
  - Measured: 541 equivalents and 23 loads fewer at 16 rounds (BOUND_GAP_ANALYSIS.md §4.1,
    "Measured").
- **Preconditions.** `C` odd (R9; holds because of rule 2). Initial index 0 (R11 base; the
  canonical assumption). Every use of `idx` is one of the R10 forms: address add, compare
  with a constant, level position against `first = L_d / 2`, or final store.
- **Facts beyond local shape.**
  - The depth `d` of every index value, to pick `L_d`: derivable by interval analysis (R7
    and R11 give the exact ranges), or emitter-only today (`tree_depth` tag).
  - The full use set of the index chain: def-use on SSA.
  - `C` odd: a constant.
- **Peephole on raw SSA today.** No (global).
- **Proofs.** `R9-mirrored-index-step`, `R9-mirrored-index-step-needs-odd-constant` (SAT),
  `R10-mirrored-index-uses`, `R11-mirrored-index-range`.
- **Resources.**
  - `L_d` constants for gathered depths: K + A + B per gathered round.
  - `B(L_d)` for the final store.
  - Compare constants `v_constant(L_d - k)` (some coincide with existing vectors; `v_constant(0)`
    is a new B).
- **Class.** Global representation choice: a stage-1 context, not a variable.

## 9. `values_only` (`--values-only`)

- **Branches.**
  - Flag `config.rs:298-302`.
  - No final index computed `ssa_emit.rs:88-94`.
  - No index address add or `vstore` `ssa_emit.rs:633-642`.
  - CLI `kernel_build_args.py:235-239`.
  - The validator compares values only (`generate_ssa_kernel.py:286, 429-454`).
- **Site.** The 32 index `vstore`s (one per group).
- **Canonical.** Per group: K offset (shared with the value store), A address add, S `vstore`
  of the final index.
- **Alternative.** No index store. DCE then removes:
  - the index address A;
  - in the mirrored form, the final `L_d - n` SUB;
  - the index chain after the last op that reads it: the last round's update, and the
    previous round's update when the last round's access does not read the index (coupling
    C3).

  Measured: 100 equivalents alone, and 101 equivalents and 2 loads on top of the other
  switches (BOUND_GAP_ANALYSIS.md §4.1).
- **Preconditions.** The index output is not observed. This is an output-contract change,
  not an identity.
- **Facts beyond local shape.** "Index memory is not an observed output": a harness
  contract, not derivable. Everything after that is DCE.
- **Peephole on raw SSA today.** Yes, as "delete these stores" given the contract.
- **Proofs.** None: no identity holds, because the output changes. Obligation instead: the
  validator must compare values only (it does) and the run must declare the assumption.
- **Resources.** None.
- **Class.** Output-contract assumption.

## 10. Tree access encoding at depth d

The rule the other agent is building. Listed here for the couplings and the canonical form.

- **Branches.**
  - Dispatch `ssa_emit.rs:660-690`.
  - Encoding choice `config.rs:396-427`.
  - Plan check `config.rs:216-234, 442-470`.
  - Setup gating `tick_tock_ssa/mod.rs:666-677`.
  - Emitters: depth 0 `ssa_emit.rs:1136-1155`; depth 1 `1157-1181`; depth 2 `1183-1210`;
    depth 3 `1212-1285`; depth 4 `1326-1424`; gather `1426-1502`; bit-driven `1057-1130`.
  - Bit-driven setup `ssa_emit.rs:922-999`. Depth 3/4 setup `ssa_emit.rs:422-559`.
  - `CachedSelect` `config.rs:76-116`; `AccessEncoding` `config.rs:163-235`.
  - CLI `kernel_build_args.py:240-248`.
- **Site.** One group's node fetch in one round, from its index (or bits) to its node value
  XOR `C` (`ctx.v_nv`): 512. The 64 sites at depth 0 and the 192 at depths 5–10 have no real
  choice today. The choice is in the 256 sites at depths 1–4 (rounds 1–4, 12–15).
- **Canonical (gather, per group).** `a = V+(idx, B(forest_p))`, decomposed to 8 lanes;
  8 L (scalar, `MemLocation::word_in(Tree, 0, n_nodes)`); assembled; `V^(·, C)`. Total 2 V + 8 L.
- **Alternatives (per group; setup in Resources).**
  - Standard d0: `B_force(tc0)`, `V^(·, C)`: 1 B + 1 V. This is the per-group root broadcast.
  - Standard d1: `V==(idx, 1)`; `V+(tc2, td1)` (true value rebuilt per group, not CSE'd:
    `emit_ssa.rs:969-977`); sel1; `V^`. Total 3 V + 1 sel1.
  - Standard d2: 3 `V==`; 1 `V+` (as d1); 1 sel1; 2 sel2; `V^`. Total 5 V + 1 sel1 + 2 sel2.
  - Standard d3 (needs `tree_reduce_max_depth ≥ 3`): `V-` level position; `V&, V>>, V&, V>>`;
    4 F leaf; 3 sel2; `V^`. Total 6 V + 4 F + 3 sel2.
  - Standard d4 (needs `≥ 4`): `V-`; 6 bit ops; 8 F; 7 sel2; `V^`. Total 8 V + 8 F + 7 sel2.
  - Bits d≥1: bits `e(d)` V (rule 11); 2^(d-1) F leaves; 2^(d-1) − 1 sel2. No XOR.
  - Bits-flow d≥1: bits `e(d)`; 2^(d-1) sel1 leaves (shared delta); 2^(d-1) − 1 sel2.
  - Mobius d≥1: bits `e(d)`; 2^d − 1 F.
  - Any bit-driven encoding at d0: nothing per group (shared root vector, rule 10a).
  - Here `e(d) = 0` for `d ≤ 2`. For `d ≥ 3` with `select_bit_source=index`,
    `e(d) = 2(d − 2)` V, +1 V for the zero-based index (`ssa_emit.rs:1016-1042`).
- **Preconditions.**
  - Gather: index in `[0, n_nodes)` (R7; the `MemLocation` range at `ssa_emit.rs:1488-1492`).
  - Standard and bit-driven at depth d: every lane's index is at depth d (R7/R11), and the
    level's nodes are those loaded at init: tree memory is not written before the access.
    Today nothing stores to the tree region (`memory_deps.rs:21-23`).
  - Bit-driven: the bits are 0/1 (R6) and in path order, oldest first (R13-needs-bit-order).
  - Gather at depth 0 is rejected for plans (`config.rs:221-223`), and the mirrored index
    has no vector at depth 0.
- **Facts beyond local shape.**
  - Depth of the index: interval analysis (R7, R11) or the emitter's `tree_depth` tag.
  - Branch bits of earlier rounds and their order: def-use along the index chain (the bit
    operand of each update). Emitter-only today (`ctx.path_bits`, `ssa_emit.rs:1656-1665`).
  - Bits are 0/1: the boolean-forms analysis (R6; `Terms.is_bool`).
  - The tree is not written before the access: memory dependence analysis (`memory_deps.rs`).
  - `prev_idx` (the index before the last update) for bit extraction: def-use.
- **Peephole on raw SSA today.** No: it needs depth and bit-path facts.
- **Proofs.** `R13-select-by-path-bits`, `R13-select-by-path-bits-needs-bit-order` (SAT),
  `R14-mobius-select`, `R14-mobius-select-needs-boolean` (SAT), `R4-fma-select` (+
  needs-boolean), `R5-delta-roundtrip`, `R6-boolean-forms`, `R2` (node XOR with `C`
  pre-applied).
  - The Standard depth-1 and depth-2 compare chains: none specific. The compare and select
    is R10's compare form plus R4/R5. A direct proof that the depth-2 linear chain
    (`ssa_emit.rs:778-800`) equals `tree[idx] ^ C` for `idx ∈ [3, 6]` is missing (proof
    needed; obligation: for `idx` in the R7 range at depth 2, the 3-compare chain over
    `(tc3, tc3 + td4, tc5, tc6)` returns `tree[idx]`).
  - Standard depth 3 and 4: the bit tree is R13 with bits read from the level position. No
    check states that exact composition (proof needed: `idx − first` split by AND/SHR
    equals the R13 bit order).
- **Resources** (EMISSION_STRUCTURE_MODEL.md §2 and §3 table; code cited).
  - Level node loads: `tc[0..7]` at `ssa_emit.rs:335-344`, always emitted;
    `tc_r3[0..8]` if any round visits depth 3, `ssa_emit.rs:432-441`, with `uses_depth3`
    from `tick_tock_ssa/mod.rs:666`; `tc_r4[0..16]` if `uses_depth4`, `ssa_emit.rs:493-502`.
    Each node is 1 A address + 1 L.
  - Derived constants `c[3..15]` and `c_r4`: `ssa_emit.rs:242-265, 474-491`. Shared OR across
    levels 2, 3 and 4.
  - Standard: deltas and broadcasts at `ssa_emit.rs:346-392` (d1, d2), `444-458, 516-537`
    (d3), `504-512, 540-559` (d4).
  - Bit-driven, per level:
    - node table, 2^d A XOR: `ssa_emit.rs:941-950`;
    - root, 1 B: `953-957`;
    - pairs, 2^(d-1) A + 2^d B: `959-976`;
    - `high`, 2^(d-1) V (bits-flow only): `969-972`;
    - Mobius, d·2^(d-1) A + 2^d B: `977-996`.
  - Gather: `B(forest_p)` once (zero-based), or the per-round base of rule 8 (mirrored).
- **Class.** Gated local rule: per (round, group), with alternatives gated by depth
  (rule 12) and priced by the index form (rule 8) and the bit source (rule 11).

### 10a. Root vector shared at depth 0

- **Branches.** `ssa_emit.rs:953-957` (table root `B(tc0 ^ C)`, once) against
  `ssa_emit.rs:1136-1155` (per group `B_force(tc0)` + `V^`).
- **Site.** 64 (rounds 0 and 11).
- **Alternative.** Nothing per group.
- **Facts.** All lanes at the root. Round 11: constant propagation (the index is constant 0
  after the wrap, rule 6). Round 0: the input assumption.
- **Proofs.** R2 (pre-XORed node), R7 (base case).
- **Resources.** 1 A + 1 B.
- BOUND_GAP_ANALYSIS.md §4.3 lists the per-group root broadcast as waste (64 ops).

## 11. `select_bit_source` (index / path)

- **Branches.** `config.rs:118-155, 305-307`; `ssa_emit.rs:1016-1042`; CLI
  `kernel_build_args.py:249-255`.
- **Site.** One bit-driven access at depth 3 or 4: up to 128 (rounds 3, 4, 14, 15 × 32).
- **Alternative index (default).** Extract the `d − 1` older bits from `prev_idx`:
  `2(d − 2)` V, +1 V `L_{d-1} − idx` if zero-based.
- **Alternative path.** 0 ops. The recorded bits stay live: 8 words per group per bit, from
  the producing round to the select.
- **Preconditions.** Index: `n < 2^(d-1)` (R11; R15-needs-range shows it is needed).
- **Facts beyond local shape.** The range of `n`: interval analysis (R11). `prev_idx`: def-use.
- **Peephole on raw SSA today.** No.
- **Proofs.** `R15-bits-from-index`; `R15-bits-from-index-needs-range` (SAT).
- **Resources.** None. The cost is scratch, which the floor does not see.
- **Class.** Gated local rule (gated on rule 10 choosing a bit-driven encoding at that site).

## 12. `tree_reduce_max_depth`

- **Branches.**
  - `config.rs:262-264, 382-384` (`use_tree_reduce_at_depth`), `396-400`.
  - `ssa_emit.rs:444, 531, 1218` (depth-3 deltas and dispatch).
  - `tick_tock_ssa/mod.rs:676`.
  - CLI `kernel_build_args.py:417-418`.
- **What it does.** Decides which depths may select instead of gathering on the flag path.
- **Class.** Not a rewrite: the per-depth domain of rule 10's alternatives. With an emission
  plan it still gates the depth-3 delta setup (`ssa_emit.rs:444`). That setup is dead under
  a plan, because Standard is not plannable, so DCE removes it *(inferred)*.

## 13. `depth4_select_rounds` (all / after-wrap)

- **Branches.** `config.rs:308-312, 396-400`; CLI `kernel_build_args.py:256-261`; binding
  check `python.rs:1250-1258` (rejects it with a plan).
- **Class.** Not a rewrite: a stage-1 pin. Round 4 sites are fixed to gather.

## 14. `scripts/emission_structure.py --fix ROUND=ENCODING`

- **Branches.** `emission_structure.py:1059-1076`.
- **Class.** Not a rewrite: a stage-1 constraint (round `ROUND` uniform with one encoding).

## 15. `access_plan` / `--emission-plan`

- **Branches.** `config.rs:313-317, 414-417, 442-470`; `ssa_emit.rs:664-690`;
  `kernel_build_args.py:262-281`; format `solver/emission_plan.py`.
- **Class.** Not a rewrite: the stage-1 output today, for rule 10 only. Its parity test is the
  model for every port gate (`tests/test_emission_plan.py:57-69`).

## 16. Depth-3 compare chain without tree-reduce

- **Branches.** `ssa_emit.rs:1286-1323`.
- **Class.** Not a rewrite: dead code. `emit_round` reaches Standard at depth 3 only when
  `use_tree_reduce_at_depth(3)` (the comment at 1287-1288; `config.rs:396-427`).

## 17. Partition stores, lazy partition, pseudo consts

- **Branches.** `ssa_emit.rs:1764-1820` (`enable_partition_stores`); `config.rs:269-276`.
  The binding hard-codes these off (`python.rs:1279-1282`). No use of `enable_lazy_partition`,
  `partition_frequency` or `enable_pseudo_consts` was found outside config and the binding.
- **Class.** Not rewrites: disabled or unused.

## 18. Scheduler-level choices

ConstChoice (`emit_ssa.rs:696-...`), SelectChoice (`emit_ssa.rs:919-1052`), VALU shift, CSE
groups (`python.rs:1068-1070`) and vbroadcast elimination all stay in the scheduler. The
reason is in design doc §5. SelectChoice correctness is R4/R5.

## 19. Scheduling metadata

Builders must keep these. None is a rewrite:

- barriers between rounds (`ssa_emit.rs:54-58`);
- `keep_live` (`ssa_emit.rs:49-52, 73-78`);
- symmetry hints (`emit_ssa.rs:298-311`);
- trace ids (`ssa_emit.rs:1577-1600, 1632-1635`);
- `VarType` and `MemLocation`.

`tree_select.rs` is a pure test mirror of the depth-3/4 bit extraction
(`tree_select.rs:1-5`). It emits nothing.

---

## Couplings

Rule X's alternative changes what rule Y matches or costs.

| # | Coupling | Where |
|---|---|---|
| C1 | Index form changes rule 10's ops: compare constants, level position, gather address and base resource, bit-extraction SUB | `ssa_emit.rs:156-190, 1024-1030, 1449-1477` |
| C2 | Mirrored makes rule 6 and rule 7's prune path inapplicable (no zero-based step) | `ssa_emit.rs:573-574, 1699-1715`; `config.rs:292-297` |
| C3 | `values_only` × rule 10 in the last rounds: what DCE removes depends on whether the round-15 (and round-14) access reads the index. Standard, gather, and bit-driven at d ≥ 3 with the index source read it (`ssa_emit.rs:1171, 1201, 1250→184, 1364→184, 1469, 1022`). Bit-driven at d ≤ 2 and the path source do not. Not additive per site | `ssa_emit.rs:88-94, 1016-1042` |
| C4 | `values_only` × mirrored: the final `L_d − n` SUB is never emitted | `ssa_emit.rs:90-94, 201-213` |
| C5 | Rule 10 bit-driven reads `path_bits`, produced by the index step's AND (`emit_branch_bit`). That AND must survive even where `values_only` kills the rest of the step | `ssa_emit.rs:1017, 1656-1665, 1708, 1740, 1745` |
| C6 | Rule 7 (index load dead) needs every round-0 reader gone: round-0 tree access (canonical gather reads it) and round-0 update (rule 6 or 8) | `ssa_emit.rs:573-597` |
| C7 | Rule 11 × index form: the zero-based form adds 1 V per site | `ssa_emit.rs:1024-1030` |
| C8 | Rule 11 path × mirrored: at depth 2 the depth-0 bit *is* the mirrored index VReg (no extra scratch); zero-based keeps a separate vector | `config.rs:123-129`; `ssa_emit.rs:1708-1711` |
| C9 | Rule 10 resources: constants `c[3..15]` feed node addresses of levels 0–4. Survival is an OR across levels (the `and:` products of the cost model) | `ssa_emit.rs:242-265, 335-344, 427-441, 474-491`; EMISSION_STRUCTURE_MODEL.md §3 |
| C10 | `uses_depth3` / `uses_depth4` decide setup emission from round depths or the plan, not from what is chosen at depth 3 | `tick_tock_ssa/mod.rs:666-677`; `ssa_emit.rs:263-265, 423, 461` |
| C11 | Rules 6, 8 and 10 depend on rule 2: the branch parity inversion (R3, R9 need `C` odd) and node tables pre-XORed with `C` | `ssa_emit.rs:949, 1746`; `rewrites.py:194-215, 319-349` |
| C12 | Rule 10 bit-driven removes the per-group XOR with `C` (folded into the table); gather and Standard keep it | `ssa_emit.rs:949` vs `1151, 1177, 1206, 1283, 1420, 1497` |
| C13 | Rule 3 changes the constant set. ConstChoice ALU-derived alternatives depend on which constants were emitted before, so other constants' alternatives can change | `emit_ssa.rs:703-711`; `ssa_emit.rs:294-313, 406-419` |
| C14 | Mixed encodings in one round reorder emission (one call per encoding, first-use order). The greedy scheduler sees the order | `ssa_emit.rs:664-690` |
| C15 | Rule 12 gates rule 10's Standard depth-3 setup even under a plan | `ssa_emit.rs:444, 531` |
| C16 | Canonical gather at depth 0 requires the zero-based index (mirrored has no depth-0 vector) | `ssa_emit.rs:1469, 1709-1711`; `config.rs:221-223` |
| C17 | SelectChoice CSE groups (scheduler) depend on which selects share `(true, false)`, which rule 10 decides | `python.rs:1068-1070` |
| C18 | Partition stores require the zero-based index | `ssa_emit.rs:1766-1769` |

## Not determined

- Whether `emit_scattered_access` at depth 0 (zero-based) is correct. It is never emitted
  today, so it is untested. It is needed for the canonical program.
- Why the wrap round uses `emit_vbroadcast_force` (no CSE) without prune. I found no reader
  that depends on distinct VRegs.
- Whether any later pass removes the per-group `V+(base, delta)` that
  `emit_select_choice_with_delta` emits at Standard depths 1 and 2 (`emit_ssa.rs:969-977`).
  There is no CSE pass among `pipeline/passes/`. Whether CP-SAT CSE groups cover it is not
  determined (they are documented for SelectChoice SUBs only).
- The exact op count `values_only` removes under each last-round encoding. It is measured per
  context, not derived here.
- Whether mixing merged and unmerged hash sites, or mirrored and zero-based chains, is ever
  useful. Mixing is valid for rule 3 and not supported for rule 8 *(inferred)*.
- Whether one instantiation per site class gives exact counts for every site (design doc §3).
  The calibration's exact fit suggests it does within a context.

## Candidates found while reading

*(inferred, not proved or measured)*

- **Per-group true-value Add at Standard depths 1/2.** The Add is identical across groups
  (above). A CSE peephole would remove it: 64 V per visit pair. This matters only if
  Standard stays an alternative.
- **Mirrored gather base per round, not per level.** CSE across the two visits of a level
  would save 1 A + 1 B per revisited gathered level.
- **Round-0 entry XOR.** Round 0 is depth 0 for every lane. With the shared root (10a),
  `(v ^ C) ^ (tc0 ^ C) = v ^ tc0` (R2's second obligation). Today's 32 entry XORs plus the
  round-0 node XOR could become 32 XORs with the raw root broadcast: 32 V fewer. A Z3
  obligation beyond R2 is not needed, but the exit-side bookkeeping must stay unchanged
  (only round 0's input form changes).

---

## Rules we do not have yet

Both mechanisms are described in BOUND_GAP_ANALYSIS.md §4.4 and in PUBLIC_KERNEL_RESCHEDULE.md
("How the 880 and 899 kernels use main memory"). Only those descriptions are used here. No
third-party code or addresses are reproduced.

### N1. Tree repack + packet `vload` gather

- **Site.**
  - **Global resource.** A repacked tree: the nodes of a band of levels are rewritten into
    memory as packets. A packet is the subtree a lane can reach from one node over k levels.
    Packets are XORed with `C` once, so they are pre-XORed (R2), and each fits in one
    8-word `vload` window.
  - **Per-site part.** One lane's fetch for k consecutive rounds of one visit. The 880
    kernel uses levels 3+4 (3-word packets) and levels 5–7 (7-word packets), plus a read in
    place at levels 8–10.
  - At 16 rounds / 32 groups that is up to 256 lanes × number of packet bands per visit
    *(inferred count; the band choice is the stage-1 decision)*.
- **Canonical.** Per lane per round: 1 L scalar gather (plus the group's address V and XOR V).
- **Alternative.**
  - Per lane per packet: 1 `vload` at `packet_base + W·p(idx)`. The 8 words land in a
    per-lane scratch vector.
  - Per round inside the band: pick the word for that depth by the lane's bits, and move it
    into the group's lane. The 880 kernel does this with a scalar XOR that also moves the
    word (BOUND_GAP_ANALYSIS.md §4.4). The word choice within a packet is data-dependent,
    so it needs a select (scalar, vector, or N2). How the 880 kernel resolves it is marked
    inferred in that doc.
  - Setup: 1 `vload` + 1 `vstore` per packet block (31 + 31 in the 880 kernel), plus A XORs
    for pre-XOR.
- **Preconditions.**
  - Lane index at the band's first level is in the R7/R11 range (packet index in range).
  - The packet memory is not read with its original meaning after it is overwritten.
  - Tree memory may be overwritten: an output-contract assumption. BOUND_GAP_ANALYSIS.md §6
    says both harnesses allow it.
- **Facts beyond local shape.** Index depth (interval analysis, R7/R11). Branch bits of the
  band's rounds (def-use path recovery). The last read of each overwritten word (memory
  dependence).
- **Proof obligations (none exist).**
  1. Layout: word `w` of packet `p` equals `tree[f(p, w)] ^ C`, for the chosen `f`. Z3 per
     layout, or by construction from the repack loop.
  2. Address: `packet_base + W·p(idx)` is the packet of the lane's node, for `idx` in the R7
     range of the band's first depth.
  3. Word selection: the word picked by the bits equals `tree[idx_d] ^ C` at each depth of
     the band (an R13-style identity over packets).
  4. Memory alias, exact:
     - Each packet `vload` gets a RAW edge to exactly the repack stores whose window it can
       overlap. With a data-dependent packet index that is typically every store of the band,
       but that must be decided by Z3 over the address terms, as in `verification/memory_alias.py`.
     - Each repack store gets a WAR edge after every load that can read the word it
       overwrites. Gathers at depth d read only level d (R7), so stores over levels already
       passed need no edge to later gathers. That has to be proved, not assumed.
     - No edge where Z3 says disjoint.
  5. Kernel level: `kernel_dataflow` must model memory written mid-kernel. Today a load
     reads `mem(addr)` of initial memory only (`kernel_dataflow.py:140`, used at
     `kernel_dataflow.py:325-331`). It needs the store-forwarding model of
     `memory_alias.SymbolicKernel`.
- **What the IR and scheduler lack.**
  - **Data-dependent block loads.** `MemLocation::block` has a constant offset; `word_in`
    is width 1 (`types.rs:1557-1569`). A `vload` with a data-dependent address needs a
    `block_in(region, lo, hi, width)` (or set-valued) offset.
  - **A temporary memory region valid in the upstream image.** Partition stores use
    `ExtraRoom` (`MemRegion::ExtraRoom`, `types.rs:1520`) through header word 7. Under the
    upstream preconditions word 7 holds the root's value and memory ends at
    `inp_values_p + batch_size` (`verification/memory_alias.py` docstring). The packets must
    overwrite consumed tree, header or index memory, so each consumed word's last read must
    be known.
  - **Exact alias decisions beyond interval overlap.** `MemLocation::may_alias` is region
    equality plus interval overlap (`types.rs:1583-1590`), which is not exact for strided
    packet addresses.
  - **Scratch for per-lane packet vectors.** 64 words per group per band live at once
    *(inferred)*. The scratch model must price it.
  - **Lane moves.** `decompose_vector` / `fresh_vector_group` (`emit_ssa.rs:1060-1110`)
    already let scalar ops read one lane of a vector and write one lane of a group. They
    were built for gather and would be reused.
- **Class.** Gated local rule per lane-band, plus a global resource (the repacked band).
  It is gated on the output-contract assumption that tree memory may be overwritten.

### N2. Store-select through memory

- **Site.** A per-lane 2-way select whose condition is a 0/1 per-lane bit. Any `SelectChoice`
  in rule 10's trees with a branch-bit condition, or the packet-word choice of N1. Up to the
  number of SelectChoices emitted *(inferred)*.
- **Canonical.** 1 `vselect` (flow) or 1–2 vector ops (FMA form).
- **Alternative (per vector of 8 lanes).**
  - For each lane ℓ:
    - store `f[ℓ]` to bank word `B + ℓ` (fixed address);
    - store `t[ℓ]` to `addr_ℓ = bit_ℓ ? B + ℓ : D_ℓ`, a dead word (address from the bit:
      1 A, or a vector op for all 8 lanes plus decompose);
    - the second store must commit after the first.
  - Then 1 `vload` of `B..B+7`.
  - Cost: 16 S + 1 L + address ops, against 1 flow or 1–2 VALU. The store engine is idle
    in our kernels: 64 store slots in 1,099 cycles (PUBLIC_KERNEL_RESCHEDULE.md).
- **Preconditions.** The bit is 0/1 (R6). `D_ℓ ∉ {B .. B+7}`. No other access to `B..B+7`
  or `D_ℓ` between the stores and the `vload`.
- **Facts beyond local shape.** The bit is boolean (the boolean-forms analysis). The bank and
  dead words are never read elsewhere (memory dependence over the whole kernel).
- **Proof obligations (none exist).**
  1. Semantics: after `store(B+ℓ, f)`, then `store(bit ? B+ℓ : D, t)`, then load `B+ℓ`, the
     result is `bit ? t : f` for `bit ∈ {0, 1}` and `D ≠ B+ℓ`. A Z3 identity over a small
     array model, plus a `*-needs-boolean` and a `*-needs-distinct-dead-word` SAT check.
  2. Exact memory edges:
     - RAW `vload` after all 16 stores (the f-stores must alias; the t-stores may);
     - WAW t-store after f-store (may alias, when `bit = 1`);
     - WAR next use of the bank after this `vload`.
     - Dead words: two t-stores to one shared dead word may alias. Either give each lane
       its own `D_ℓ`, or prove the dead word is never loaded and drop the WAW. That drop is
       a new rule, "WAW to a never-read word needs no order", and needs its own proof.
     - The 880 kernel shares one dead word across many stores, which is where its large
       edge counts come from (PUBLIC_KERNEL_RESCHEDULE.md, memory table).
  3. Latency: our `MemEdgeKind` has WAW latency 1 (`memory_deps.rs:10-13`). `memory_alias.py`
     uses 0, with slot order within a bundle. Latency 1 is valid and slightly slower.
     Allowing 0 needs the lowering to keep program order inside a bundle.
  4. Kernel level: the same store-forwarding requirement on `kernel_dataflow` as N1.
- **What the IR and scheduler lack.**
  - **Set-valued memory offsets.** `{B+ℓ, D_ℓ}` is not an interval, so `MemOffset::Range`
    would order it against the whole span. A `MemOffset::OneOf` (or a term plus a Z3
    decision) is needed for exact edges.
  - **Stores with a data-dependent address and a lane value.** The address is a scalar VReg
    from the bit; the value is a lane from `decompose_vector`.
  - **A bank region.** Same issue as N1. With `values_only` and the initial-index-0
    assumption, the index region (256 words) is unused by the kernel and is a candidate
    *(inferred; it couples N2 to rule 9 and the canonical assumption)*.
  - **Scheduler choice or rewrite.** The dataflow is "select", so by the design doc §5 rule of
    thumb this could be a third SelectChoice encoding in the scheduler rather than a rewrite.
    But it adds 18+ ops and memory edges, which the CP-SAT model does not model as an
    optional encoding today. Treat it as a stage-1 rewrite first *(inferred)*.
- **Class.** Local rule (per select site), with memory resources (bank, dead words) shared
  per batch of sites.

---

## Proposed port order

The parity gate for every PR has the same form, modelled on
`tests/test_emission_plan.py:57-69`:

```
program(rewrite(emission(flags without X), choose X everywhere)) == program(emission(flags with X))
```

`program()` is the scheduler JSON's SSA ops without cycles, compared after renumbering VRegs
and op ids in emission order. Builders insert resource ops at the emitter's anchor positions
(design doc §8.1). This lets rules be ported against today's flag emission, before the
canonical emitter exists. Every PR also keeps the default emission unchanged
(`tests/test_emission_options.py`) and has PROVED (`kernel_dataflow`) and PASSED on the new
path.

| PR | Contents | Parity gate (must be identical) | Notes |
|---|---|---|---|
| 1 | Framework: `Site`, `Rule`, `Alternative`, `Resource`, certificate log, `transform()` hook between `build()` and `PipelineContext::new` (`tick_tock_ssa/mod.rs:782, 921`). No rules | Every existing flag combination unchanged with the hook on (identity transform) | Renumbering normaliser for `program()` |
| 2 | Rule 3 hash merge (peephole, constants only) | `--merge-hash-stages` on default flags, and with `--index-form mirrored --cached-select bits` | Anchor for `c1, m2, c2` at `ssa_emit.rs:406-419` (C13) |
| 3 | Rule 5 + rule 6 + DCE (prune) | `--prune-dead-index` | Round 0 needs the declared input assumption; round 11 by constant propagation |
| 4 | Rule 9 `values_only` (delete index stores, DCE) | `--values-only` with standard, bits, and mirrored | C3/C4 are DCE outcomes; no special casing |
| 5 | Rule 10 tree access (in progress elsewhere), including 10a | `test_plan_reproduces_flag_emission` cases, and `--cached-select standard` | Must reproduce first-use order (C14) |
| 6 | Rule 11 `select_bit_source` | `--select-bit-source path` with bits at `tree_reduce_max_depth` 3 and 4 | Gated on 5 |
| 7 | Rule 8 mirrored index (global; stage-1 context) | `--index-form mirrored` × each cached-select mode, ± `--values-only` | Largest coupling set (C1, C2, C4, C7, C8, C16) |
| 8 | Switch to canonical emission (gather at every depth including 0, zero-based, indices stored, 11-op hash); delete flag branches | Canonical + rules chosen to match each flag set == that flag set's emission, for every row of EMISSION_ABLATION.md | Needs the depth-0 gather check (Not determined, item 1) |
| 9 | Optional: reference-form hash (18 ops) and rule 1 as a peephole; SSA recovery from lifted kernels | Rule 1 output == today's canonical hash | First step on the untagged path (design doc §7) |
| 10 | N2, then N1 | No parity (new emission): PROVED with store forwarding in `kernel_dataflow`, PASSED, exact memory edges | Needs the IR additions above |

PRs 1–8 and 10 are what the benchmark number needs. PR 9 serves only the north-star goal
below.

### Later phase: input is the lifted baseline, no flags (goal only)

This phase is separate from, and after, the order above. Pass names refer to
[BASELINE_TO_BEST_GAP.md](BASELINE_TO_BEST_GAP.md) §3. None of these steps is needed for the
benchmark number.

| Step | Contents | Gate |
|---|---|---|
| L1 | B0 + B1: SSA recovery from `kernel_lift` output (straight-line renaming, vector-group recovery), producing an `SsaProgram` with `MemLocation`s from `memory_alias` address terms | `kernel_dataflow`: original kernel against the recovered program, on the baseline and on our own saved kernels |
| L2 | B2–B4: header specialisation, memory-to-register promotion of idx and val, dead-store elimination | SSA-to-SSA equivalence with memory (`verification/ssa_equivalence.py` or `SymbolicKernel` forwarding); `--validate` |
| L3 | B5–B9: if-conversion, bounds-check elimination, multiply-add formation, R1 fusion, R2 deferral | per-rule Z3 (MA2, R7, R1, R2, R3) + certificate |
| L4 | B10 + B11: SLP vectorisation into 32 groups of 8; CSE | SSA-to-SSA with a lane mapping; result equals today's canonical program after renumbering |
| L5 | Feed L4's output into PRs 1–10 instead of the emitter | lifted baseline + no flags == our best kernel's SSA program (or better on the floor) |
