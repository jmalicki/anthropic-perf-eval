# Stage-1 loop under the stall judge, and the start-up of `path_flow90.json`

Follow-up to docs/SCRATCH_CEILING_DIAGNOSIS.md. (A) Rerun the stage-1 loop with
`--greedy-budget-force stall --pressure-inflation-threshold 0.98` as the greedy that judges
plans, plus a plain top-k selection. (B) Attack the 24 start-up cycles between
`path_flow90.json` (1,033) and its static bound (1,009).

All numbers are greedy development measurements at 16 rounds, batch 256, `--merge-hash-stages
--index-form mirrored`, shaped greedy with the load-tail flags (below). Every kernel quoted is
PROVED (dataflow replay) and PASSED (`--validate`). "Measured by removal" means the effect was
measured by building with and without the change; "accounted" means it was attributed from a
schedule's slots without a counterfactual build.

## Answers

- **A, loop under the stall judge: negative.** No plan beats the baselines. Index bits: 73
  builds, best **1,039** (four plans tie, `flow90.json` among them). Path bits: 68 builds, best
  **1,034**, which is `path_flow90.json` itself under this judge (1,033 under the default one).
  The judge changes the path loop's own plans a lot (best non-seed plan 1,099 in the earlier
  loop, 1,042 now) but not the winner.
  **Top-k by score** (corrected after review: the first version built 8 plans that all shared
  the optimal score, i.e. members of one tie plateau, not ranks 1 to 8): one plan per score
  level for 14 levels per objective, each judged by the greedy. Index, floor 989 to 1,002:
  1,051 to 1,334 cycles; greedy model 1,030 to 1,043: 1,052 to 1,423. Path, floor 973 to 986:
  1,066 to 1,246; greedy model 1,011 to 1,025: 1,055 to 1,481. None beats the baselines. The
  best known plans score 997 / 1,043 (index, floor / model) and 980 / 1,023 (path), and **at
  least 100 plans score at or below each of them** (counted up to 100), so with these scores a
  top-k needs k > 100 to be guaranteed to include them: stage-1 scores do not rank plans finely
  enough.
  **Budget: stall + 0.98 is not safe on every plan.** 3 of the 28 new path-bit builds fail the
  scratch budget (peaks 1,761, 1,809, 1,817): the greedy reaches a cycle where nothing can be
  placed at 1,528 live words (a real deadlock), forces an op and stays over budget for 265 to
  874 cycles. 0 of the index builds and 0 of the 85 earlier builds fail. They are recorded as
  failed plans, not tuned around.
- **B, start-up: 1,033 -> 1,030** (1,032 simulated; PROVED, PASSED, `check_upstream.sh
  --strict`) with two wavefront orderings behind new flags, `--greedy-flow-priority chain-first`
  and `--greedy-alu-priority native-first`. Measured by removal: chain-first alone 1,031,
  native-first alone 1,033 (no change), both 1,030. Native-first helps only together with
  chain-first. On `flow90.json` (index) chain-first is worse (1,065; both 1,073) and
  native-first alone changes nothing (1,039). So the defaults stay.
  1. **Plan change (`bits` for the first groups in rounds 1 to 3): negative**, searched by hand
     (9 plans), not by stage 1. Every variant is slower: 1,038 to 1,063 alone, 1,036 to 1,037
     with both orderings. The flow admission rate it was meant to fix is fixed better by
     chain-first, and `bits` costs vector work.
  2. **Setup loads in the load window: 10 -> 1 by native-first** (measured by removal, with
     chain-first on: 10 with chain-first alone, 1 with both). They were the batch-value `vload`s
     of later groups, waiting for their address adds on a scalar ALU full of shifted vector ops.
     The ConstChoice route was not available: the greedy ignores `--const-choice` (identical
     kernel with `add_imm`), 34 ConstChoices stay on the load engine.
  3. **t = 51 is not reducible by a start-up encoding** (accounted from the dependency chain).
     The chain to the first gather is 6 setup ops, round 0 (10: xor, the 9-op hash chain, branch
     `&`), rounds 1 to 3 (11 each: one select, xor, hash chain, `&`), the index
     `multiply_add` and the address `-`. One select per round is on it whatever the encoding
     (`bits` and `bits-flow` both put a 1-cycle select there; gather would be longer). Only the
     6 setup ops (header pointer load, address add, root load, xor with the deferred constant,
     broadcast) or the hash depth could shorten it: emission changes, not scheduling.

What is left of the 21 cycles at 1,030 (exact split, then attribution with `scratch_ceiling_diagnosis
gap`): 20.5 cycles of idle load slots and 0.5 cycle of setup loads in the window; the idle slots
are attributed to 9.5 cycles of a delayed load with no load slot (setup loads before t: 110 in
cycles 0 to 44, 34 of them ConstChoices), 8 cycles of a delayed vector op with no room, and 3
cycles of a ready load waiting. Attribution is accounting, not a counterfactual: the 9.5 flow
cycles attributed at 1,033 were removed by chain-first and bought 2 measured cycles
(docs/GREEDY_CONST_CHOICE.md shows the same for the setup loads: moving 23 constants off the
load engine bought 0).

## A. The loop under the stall judge

Judge: shaped flags + load tail (`greedy_load_priority=final-last, greedy_groups_in_flight=32,
greedy_stagger_cycles=9`) + `greedy_budget_force=stall, pressure_inflation_threshold=0.98`
(`--greedy-option`, new; keys are validated: an option the build does not read, or one the loop
fixes such as `rounds` or `use_greedy`, is an error). Seeds: the six default seeds plus
`path_flow90.json` and `path_nogood_capped0.json`. Phases: modes `nogood,caps,caps-underfilled,
rows` (budget 8 each); `nogood` under `flow=0.92`; then the top-k rerun below. The first
phases' stage-1 solves had a 60 s wall-clock limit, so their plans past the first may depend on
machine load; the top-k rerun uses deterministic-time solves; the builds are deterministic.

| | Index bits | Path bits |
|---|---|---|
| builds | 73 | 68 |
| best | 1,039 (`flow90.json`, `caps-underfilled2/3`, `nogood-capped0`) | 1,034 (`path_flow90.json`) |
| best loop plan (not a seed) | 1,039 | 1,042 (`nogood-capped2`) |
| top-k, floor, 14 levels | scores 989 to 1,002: 1,051 to 1,334 | scores 973 to 986: 1,066 to 1,246 (1 failed) |
| top-k, greedy model, 14 levels | scores 1,030 to 1,043: 1,052 to 1,423 | scores 1,011 to 1,025: 1,055 to 1,481 (2 failed, 1 level skipped) |
| plans scoring at or below the best known plan | floor 997: 100+; model 1,043: 100+ | floor 980: 100+; model 1,023: 100+ |
| failed / over budget | 0 / 0 | 3 / 3 (peaks 1,761 to 1,817: deadlock at the budget) |
| peak live scratch (incl. the zero word) | 1,057 to 1,473 | 1,057 to 1,536 |

**Top-k, corrected.** The first version built the 8 unbuilt plans the solver returned in turn;
all 8 shared the optimal score (989 index floor, 973 path floor, 1,030 / 1,011 by the model),
so they were 8 members of one tie plateau, not ranks 1 to 8, and `flow90.json` (model score
1,043) was out of reach. `loop_topk` now enumerates score levels: for level L it asks for an
unbuilt plan with score <= L and objective >= L and builds it only if the plan's own score
(`plan_score`) is exactly L. The objective variable is only an upper relaxation of a plan's
score, so `objective >= L` alone returns plans below L (an earlier attempt asserted on exactly
that); a plan below L is excluded and the solve repeated, at most 20 times (one path level, 1,024
by the model, was skipped that way). No-goods are per model point (`plan_model_key`, the
per-(round, encoding) group counts): plans that differ only in which groups use an encoding
are one plan. With one plan per level and 14 levels, the greedy's cycles do not fall with the
score: the floor and the model do not order plans at this resolution, and at least 100 plans
(the count's limit) score at or below each best known plan.

Surrogate ranking (Spearman of score vs greedy cycles over all builds, full tables in
`docs/emission_plans/loop_report_stall_{index,path}.json`):

| Surrogate | Index all (73) | Index within 30 (38) | Path all (65) | Path within 30 (22) |
|---|---|---|---|---|
| floor | -0.07 | -0.21 | +0.18 | -0.29 |
| capped floor `flow=0.9` | +0.28 | +0.02 | +0.60 | +0.41 |
| greedy model (GREEDY_MODEL) | +0.67 | +0.34 | +0.35 | +0.41 |
| VALU cycles left after the greedy's shifts (post-build) | +0.98 | +0.91 | +0.97 | +0.89 |

**GREEDY_MODEL under this judge** (`scripts/greedy_model_fit.py` on
`docs/emission_plans/greedy_model_data.json`, judge `stall-0.98`: the earlier stall-judge loop
builds as the fit set, the top-k builds made afterwards as a held-out set). Refit on 41 index-bit plans
under 1,200 cycles: a_floor 52, a_load 13, b_flow 0.80, flow0 900 (checked-in, fitted on the
default judge: 32, 19, 0.75, 900). Leave-one-out MAE 5.6 (max 18), but the model's form was
chosen on earlier plans of the same kind, and leave-one-out uses the plans the constants are
scored on. **Held out**: index MAE 14.5 (optimistic by 11.6 on average, 23 plans under 1,200),
path MAE 62.3 (always optimistic, 21 plans). Checked-in constants on the earlier stall-judge
index plans: MAE 9.4; constants fitted on the default judge vs refitted on the stall judge
differ mainly in a_floor (+20). So: the judge change moves the constants; the model is a weak
ranker near the best under either judge and fails out of sample on path bits. The refit is checked
in as `es.GREEDY_MODEL_STALL` (reproduced by `greedy_model_fit.py --judge stall-0.98 --check`);
`es.GREEDY_MODEL` is unchanged (it describes the default judge).

## B. The start-up

### B.1 Orderings (measured by removal)

| Plan | default | native-first | chain-first | both |
|---|---|---|---|---|
| `path_flow90.json` path | 1,033 (peak 1,465) | 1,033 | 1,031 (1,484) | **1,030** / 1,032 (1,497) (U) |
| `flow90.json` index | 1,039 (1,431) | 1,039 | 1,065 | 1,073 |

(U): passes `./scripts/check_upstream.sh --strict` (9 x `CYCLES: 1032`). Other settings with
both orderings on `path_flow90.json`: stagger 8 1,030 (U), stagger 10 1,045, `stall` + 0.98
1,035; index with native-first + `stall` + 0.98 1,039.

Where the cycles go. Gap, idle and extra columns are an exact, checked split
(`scratch_ceiling_diagnosis gap`, row t 51, q 14, W 1,888); the cause columns **attribute** the
idle slots (accounting, not a counterfactual; deterministic, fractional; "no room": the op's
engine had no room in every cycle it waited). Cycles:

| `path_flow90.json` | gap (measured) | extra (setup) loads in window | flow no room | vector no room | load no room | ALU no room | ready load waited |
|---|---|---|---|---|---|---|---|
| default | 24 | 5.0 | 9.5 | 0 | 2.5 | 4.0 | 3.0 |
| chain-first | 22 | 5.0 | 0 | 7.5 | 5.5 | 0 | 4.0 |
| both | 21 | 0.5 | 0 | 8.0 | 9.5 | 0 | 3.0 |

The measured differences are 2 and 1 cycles; the attributed flow term was 9.5. chain-first
removes the flow attribution, but most of it reappears on the vector ALU and the load engine;
native-first removes the setup loads from the window, and they reappear as load starvation
before t. The start-up is close to a zero-sum among engines: the load engine has
about 120 setup loads for 102 slots before t = 51 (accounted), so some must land after t
unless setup loads leave the load engine.

### B.2 Plan family (hand search, not stage 1)

`bits` instead of `bits-flow` for groups 0..N-1 (the first in wavefront order) in the named
rounds of `path_flow90.json`; `bits` needs 0 + 1 + 3 flow selects in rounds 1 to 3 against 1 +
3 + 7. Stage 1 cannot express it: its variables are group counts per (round, encoding), not
which groups.

| Variant | default flags | both orderings |
|---|---|---|
| rounds 1-3, N = 4 / 8 / 12 / 16 / 32 | 1,043 / 1,040 / 1,040 / 1,045 / 1,063 | - / 1,036 / - / - / - |
| round 3, N = 4 / 8 / 16 | 1,038 / 1,038 / 1,042 | 1,037 / 1,036 / - |
| rounds 2-3, N = 8 | 1,041 | - |

All slower than the unchanged plan (1,033, 1,030 with both orderings). The hypothesis behind
it (one flow slot admits a group to the gathers only every 11 cycles; the observed spacing of
first gathers was 11 cycles) is consistent with the default schedule, but relieving it with
`bits` costs more vector work than it saves.

### B.3 The head t

Longest dependency path to the first gather load (group 0, round 4), unit latency, from the
`path_flow90.json` schedule: setup 0 to 5 (ConstChoice -> load header -> add -> load root -> xor
-> broadcast), round 0 6 to 15, round 1 16 to 26, round 2 27 to 37, round 3 38 to 48, index
`multiply_add` 49, address `-` 50, load 51. Each round 1 to 3 has exactly one select on the path;
with any cached encoding it is one cycle. So no start-up encoding of the first groups shortens
t (accounted, not built). Emission could: e.g. broadcasting the root before the deferred-constant
xor would take 1 cycle off the setup.

## Flags added

| Flag | Default | What |
|---|---|---|
| `--greedy-flow-priority chain-first` | `key` | Flow ops by chain rank (shared first), then key. Test: `test_flow_chain_first_lets_the_first_chain_through` (control: key is later) |
| (both orderings, and `--greedy-load-priority`) | | Wavefront only: with another `--greedy-priority` they are an error (they were silently ignored). Test: `tests/test_wavefront_only_options.py` |
| headline pin | | `tests/test_startup_orderings_pin.py` (CI only, 16-round builds): default 1,033, chain-first 1,031, native-first 1,033, both 1,030 |
| `--greedy-alu-priority {key,native-first}` | `key` | Native scalar-ALU ops are tried before the cycle's vector ops (lanes of a split shift carried from the previous cycle are still placed first and can take scalar slots). Test: `test_alu_native_first_keeps_scalar_slots_for_natives` (control: under the key shifts split the natives over two cycles). The reorder also puts natives first in the ASAP-inflation scan, whose last 3 ops are exempt: measured on `path_flow90.json` with chain-first, total inflation deferrals 299 -> 206; alone it gives the default kernel (231 -> 231) |
| `emission_structure loop --greedy-option KEY=VALUE` | none | Greedy build option for the judge, stored in the state |
| loop modes `topk-floor`, `topk-greedy-model` | - | Build the k best unbuilt plans by floor / greedy model |

Both orderings are also in `test_select_overflow_keeps_slot_limits` (slot limits with split
shifts and select overflow).

## Next

- Setup loads off the load engine: a greedy ConstChoice policy (flow `add_imm` or ALU
  derivation, today CP-SAT only) for the 34 consts that saturate the load engine before t. That is
  the largest remaining start-up term (9.5 cycles, accounted).
- The 7 VALU cycles under chain-first: early groups' hash work competing with later groups'
  work; a stagger profile (smaller for the first groups) is untried.

## Reproduce

```bash
F="--rounds 16 --use-greedy --validate --greedy-barriers none --greedy-priority wavefront \
  --greedy-split-shift --pressure-inflation-threshold 0.95 --greedy-load-priority final-last \
  --greedy-groups-in-flight 32 --greedy-stagger-cycles 9 --merge-hash-stages --index-form mirrored"
uv run generate_ssa_kernel.py $F --emission-plan docs/emission_plans/path_flow90.json --select-bit-source path \
  --greedy-alu-priority native-first --greedy-flow-priority chain-first \
  --output-json both.json --save-kernel both_kernel.json
./scripts/check_upstream.sh --kernel both_kernel.json --strict
uv run python -m scripts.scratch_ceiling_diagnosis gap both.json --t 51 --q 14 --w 1888
# Loop A (per bit source; one heavy job at a time):
J="--load-tail --greedy-option greedy_budget_force=stall --greedy-option pressure_inflation_threshold=0.98"
uv run python -m scripts.emission_structure loop run --select-bit-source path --state path.json $J \
  --modes nogood,caps,caps-underfilled,rows --budget 8   # + seeds, see section A
uv run python -m scripts.emission_structure loop run --select-bit-source path --state path.json $J \
  --nogood-caps flow=0.92 --modes nogood --budget 8
uv run python -m scripts.emission_structure loop run --select-bit-source path --state path.json $J \
  --modes topk-floor,topk-greedy-model --budget 14 --per-level 1 --deterministic-time 5 \
  --count-limit 100 --json report.json
# GREEDY_MODEL under this judge (fit set + held-out set as in section A):
uv run python -m scripts.greedy_model_fit --judge stall-0.98 --check
```

About 85 loop builds and 30 start-up builds, one at a time.
