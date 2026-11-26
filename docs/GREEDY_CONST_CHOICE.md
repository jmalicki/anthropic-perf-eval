# Greedy ConstChoice policy, forced encodings, and an early stagger

Follow-up to docs/STARTUP_AND_STALL_LOOP.md. The start-up of the best path-bit kernel
(`path_flow90.json`, 1,030 with `--greedy-flow-priority chain-first --greedy-alu-priority
native-first`) was accounted as 9.5 cycles of a load engine saturated by setup loads before
t = 51 (about 120 setup loads for 102 slots) and 7 cycles of VALU-full upstream of the early
gathers. This tries (1) taking constants off the load engine with a greedy ConstChoice
policy, and (2) a smaller stagger for the first groups.

All numbers are greedy development measurements at 16 rounds, batch 256, the shaped greedy
with the load-tail flags. "Measured" = built with and without the change; "accounted" =
attributed from a schedule's slots. Every kernel quoted is PROVED (dataflow replay, which
covers the ALU-derived constants) and PASSED.

## Answers

1. **Constants off the load engine: works, worth 0 cycles without justification
   (measured).** The new policy
   (`--greedy-const-policy alu-first`) moves 23 of 35 load-engine constants to the scalar ALU
   on `path_flow90.json` (24 ALU-derived, 18 `add_imm`, 12 loads) and 23 of 33 on
   `flow90.json`. Cycles: **1,030 -> 1,030** and **1,039 -> 1,039**; `alu-when-load-full`
   1,030 and 1,040. Both `alu-first` kernels pass `check_upstream.sh --strict` (1,032 and
   1,041 simulated). Why nothing moves (accounted): the setup loads then end at cycle 38
   instead of 50, and the load engine idles in cycles 39 to 50 instead, because no gather can
   start before t = 51; after t the ramp has the same idle slots (41 -> 42 on path). The
   "load engine full" blame in the earlier accounting was the wavefront order of the batch
   `vload`s of later groups, not load capacity. Measured without `--greedy-justify` (on #86 justification
   skipped any schedule with an ALU-derived constant). With derived constants modelled
   (#93, docs/CRITICAL_PATH.md), `alu-first` + `--reassociate-gather-xor` (R16) + justify 10
   gives 1,016 on path, the same as without `alu-first`; without R16 both const policies give
   1,019. Command (docs/emission_plans/justify_on_best_sweep.json, row "R16, path, alu-first"):
   `uv run generate_ssa_kernel.py --validate --rounds 16 --use-greedy --greedy-barriers none
   --greedy-priority wavefront --greedy-split-shift --pressure-inflation-threshold 0.95
   --greedy-load-priority final-last --greedy-groups-in-flight 32 --greedy-stagger-cycles 9
   --merge-hash-stages --index-form mirrored --greedy-justify 10 --emission-plan
   docs/emission_plans/path_flow90.json --select-bit-source path --greedy-flow-priority chain-first
   --greedy-alu-priority native-first --reassociate-gather-xor --greedy-const-policy alu-first`; #91 records 1,026 for `alu-first` + `--reassociate-gather-xor`
   (without justification) as a fragile interaction (1,067 at stagger 8).
2. **Forced encodings under `--use-greedy` now take effect** (they were dropped: the greedy
   read them only from the CP-SAT config). `--const-choice add_imm`: 54 of 54 constants on
   flow, 1,072 cycles on path; `--const-choice load`: 54 on load, 1,038 (static bound 1,017);
   `--select-choice fma` runs every select as a multiply_add. All PROVED and PASSED.
   Also tested (small sizes): `--select-choice vselect` (every select a flow vselect),
   `--valu-shift valu` (no shift) and `--valu-shift alu` (every shiftable vector op on the
   scalar ALU). The last is very slow under the greedy, about one shifted op per cycle on the
   12 scalar slots: 7,148 cycles at 16 rounds against 2,007 (shaped: 7,117 against 1,313;
   `valu`: 2,104 and 1,616), so a warning is printed. Options that another option would
   override or ignore are errors (CLI and binding, `check_encoding_option_combinations`): a
   forced encoding with the hint pass that would overwrite it, `--greedy-early-stagger`
   without the wavefront priority, and `--greedy-const-policy` under CP-SAT, a forced
   `--const-choice` or the const-choice hint pass.
3. **Setup loads needed before t (accounted, first consumer in the schedule):** path, 107
   of 120 (13 later: 12 batch `vload`s, 1 constant); index, 80 of 120. 107 loads need 54
   cycles of the load engine, above t = 51, so on the path plan some must land after t unless
   they leave the load engine; with 23 of them moved, they no longer do, and cycles still do
   not move (item 1).
4. **Smaller stagger for the first groups: negative (measured).** `--greedy-early-stagger
   K:S` (first K groups S cycles apart, the rest the normal 9 after them), 12 settings per
   plan: path 1,033 to 1,045 (K = 8, S = 0 fails the scratch budget at 1,778) against 1,030;
   index 1,039 to 1,135 against 1,039. Smaller early staggers shift fewer vector ops (1,194 to
   1,235 against 1,265 on path): more groups contend for the VALU at once, which is the 7
   VALU cycles, not admission. Larger early staggers (S = 12, 15) give 1,033 to 1,036 and 1,039
   to 1,043.
5. **Static bound: unchanged** at 1,009 (path) and 1,029 (index). The binding row is
   t 51 + q 14 + ceil(W / 2) with W the gather loads after t; constants are not in W. Only the
   tail row moves (path 1,008 -> 996, index 1,027 -> 1,015), and it was not binding.

So no kernel beats a baseline here (without justification): the best stay 1,030 / 1,032
(path) and 1,039 / 1,041 (index). The start-up loss after t is the ramp of group admission into the load stream under
the wavefront stagger, and the early-stagger and const experiments show it is not cheaply
changed in the greedy.

## 1. The ConstChoice policy

`--greedy-const-policy {default,alu-first,alu-when-load-full}` (greedy-only; the CP-SAT
pipeline keeps `default`, since its hint names only Load or AddImm). `GreedyShape::const_policy`
/ `ConstPolicy` in `scheduler_rs/src/kernel_builder/ssa_greedy/mod.rs`.

- `default` (unchanged): `add_imm` when the flow slot is free and no other flow op is ready,
  else a load.
- `alu-first`: ALU derivation when possible, else the default rule.
- `alu-when-load-full`: a load while a load slot is free, else ALU derivation, else default.

Derivation candidates: the emitter's own alternatives (`ConstEncoding::{Add, Sub, Mul,
Shift}`, offered for only 6 of 54 constants), then `a + b` for any two constants a, b > 0 with
a + b = value < 2^31 (e.g. the batch offsets 8g = 8(g - 1) + 8). A producer qualifies only if
it was scheduled in an earlier cycle and its value is still in the scheduler's live-word
counter, so the derive's read needs no lifetime extension: address assignment's AluDerive
pass (which extends producer intervals, the earlier VReg-lifetime bug) changes nothing, and
the counter-vs-interval agreement check holds. The chosen derivations go to lowering as
`ConstChoiceEncoding::AluDerive` through `GreedyDiagnostics::const_alu_derive`.

| Plan | policy | cycles | load / flow / ALU constants |
|---|---|---|---|
| `path_flow90.json` + both orderings | default | **1,030** | 35 / 19 / 0 |
| | alu-first | 1,030 (U) | 12 / 18 / 24 |
| | alu-when-load-full | 1,030 | 14 / 18 / 22 |
| | `--const-choice add_imm` | 1,072 | 0 / 54 / 0 |
| | `--const-choice load` | 1,038 | 54 / 0 / 0 |
| `flow90.json` (index) | default | **1,039** | 33 / 21 / 0 |
| | alu-first | 1,039 (U) | 10 / 21 / 23 |
| | alu-when-load-full | 1,040 | 9 / 21 / 24 |

(U): `check_upstream.sh --strict` passes (1,032 / 1,041 simulated).

Accounted (`scratch_ceiling_diagnosis gap`, cycles): path default 21 = 20.5 idle load slots in
[t, T - q) + 0.5 early loads; path alu-first 21 = 21.0 + 0. Load engine use per cycle, 0 to 70
(path): default full to cycle 50, then 0 to 2; alu-first full to cycle 37, idle 39 to 50, then
the same ramp.

**With a fixed-encodings rerun** (a hint pass or `--schedule-with-fixed-choices`, which rerun
the greedy with the first run's encodings fixed): the rerun schedules each derived constant as
the ALU op it lowers to, after both producers (visible and still live, as the policy itself
requires), or, where it cannot, as a load that is then recorded as `Load`
(`reconcile_const_choice_encodings`). Before the review of #86 the rerun always counted it as a
load while lowering still emitted the ALU derivation: 13 scalar ops in a cycle and a wrong
value at 4 rounds, batch 32 (`test_const_policy_with_a_fixed_choices_rerun`).

## 2. Early stagger

`--greedy-early-stagger K:S` (`EarlyStagger` in `ssa_greedy/wavefront.rs`): chain `rank` gets
the offset `S * min(rank, K) + 9 * max(rank - K, 0)`.

| K:S | path (both orderings) | index |
|---|---|---|
| none | **1,030** | **1,039** |
| 2:0 / 2:3 / 2:6 / 2:12 | 1,040 / 1,035 / 1,036 / 1,036 | 1,045 / 1,045 / 1,043 / 1,041 |
| 4:0 / 4:3 / 4:6 / 4:12 / 4:15 | 1,045 / 1,033 / 1,035 / 1,033 / 1,036 | 1,065 / 1,049 / 1,047 / 1,039 / 1,043 |
| 8:0 / 8:3 / 8:6 | over budget (1,778) / 1,044 / 1,033 | 1,135 / 1,060 / 1,047 |

## Tests

- Rust: `test_const_policy_derives_from_live_constants` (operands visible and still live at
  the derive, sum equals the value; control: `default` derives nothing),
  `test_early_stagger_chain_offsets`.
- Python `tests/test_greedy_const_choice.py`: forced `add_imm` / `load` / `fma` take effect
  under `--use-greedy`; `alu-first` and `alu-when-load-full` kernels are proved by the dataflow
  replay and validated (2 rounds, batch 16).

## Next

- The start-up after t is the admission ramp: group g reaches its first gather about 51 +
  11g cycles in, each gather round gives the load engine 4 cycles of work. Shortening it needs
  less work per group before the first gather (emission: the per-round hash chain and the
  rounds 1 to 3 selects), not a different order or encoding of the setup.
- CP-SAT on the start-up window (the window probe) with free encodings would show whether
  any schedule of this emission does better in cycles 0 to 110.

## Reproduce

```bash
F="--rounds 16 --use-greedy --validate --greedy-barriers none --greedy-priority wavefront \
  --greedy-split-shift --pressure-inflation-threshold 0.95 --greedy-load-priority final-last \
  --greedy-groups-in-flight 32 --greedy-stagger-cycles 9 --merge-hash-stages --index-form mirrored"
P="--emission-plan docs/emission_plans/path_flow90.json --select-bit-source path \
  --greedy-alu-priority native-first --greedy-flow-priority chain-first"
uv run generate_ssa_kernel.py $F $P --greedy-const-policy alu-first --save-kernel k.json --output-json s.json
./scripts/check_upstream.sh --kernel k.json --strict
uv run generate_ssa_kernel.py $F $P --greedy-early-stagger 4:3
uv run generate_ssa_kernel.py $F $P --const-choice load      # now takes effect under --use-greedy
```

About 40 sixteen-round builds, one at a time.
