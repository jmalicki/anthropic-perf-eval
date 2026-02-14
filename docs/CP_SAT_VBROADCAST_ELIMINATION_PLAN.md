# CP-SAT combined VALU shift / vbroadcast elimination

## Goal

Make the **combined** transform (shift consumers of a vbroadcast to ALU + eliminate the vbroadcast) a first-class choice the CP-SAT model can optimize. Quantification showed 44 of 47 vbroadcasts can be profitably eliminated and 3,101 chains with positive ASAP reduction ([scripts/quantify_schedule_hypotheses.py](scripts/quantify_schedule_hypotheses.py)).

## Current state

- **VALU shift**: In [scheduler_rs/src/cpsat_scheduler.rs](scheduler_rs/src/cpsat_scheduler.rs), shiftable VALU ops get a BoolVar and two optional intervals (VALU vs 8× ALU). Precedence for consumers uses `completion_cycle`. Shift decisions are in `shifted_ops` / `shifted_alu_cycles`.
- **Vbroadcast**: Non-shiftable VALU; fixed interval in VALU cumulative. Always emitted. Precedence: `cycle[C] >= cycle[vb]+1`.
- **Existing design**: [VALU_SHIFT_VBROADCAST_DEPENDENCY_OPTIMIZATION.md](VALU_SHIFT_VBROADCAST_DEPENDENCY_OPTIMIZATION.md).

## Design (summary)

- For each vbroadcast with at least one **shiftable** consumer and a single scalar producer: add `eliminate_vb[vb]` = AND(shift[C] for C in shiftable_consumers(vb)). Optional VALU interval for vb with presence = !eliminate_vb. Precedence: when consumer C is shifted, use scalar producer for C’s ALU precedence; when C not shifted, keep cycle[C] >= cycle[vb]+1 (and enforce vb present). Post-solve: record `eliminated_vbroadcast_ops`. Lowering: omit eliminated vb; when expanding shifted VALU that depended on eliminated vb, use scalar producer as dep.

---

## Tests

### Unit tests (Rust, `cpsat_scheduler.rs`)

- **Minimal fixture: vb → shiftable consumer → consumer**
  - Build SSA: const(0) → vbroadcast → valu1 (shiftable, e.g. Add) → valu2. One VALU slot, enough ALU so shifting valu1 is feasible. Run CP-SAT with `enable_valu_shift: true` and `enable_vbroadcast_elimination: true`. Assert: solution exists; either (a) vb is eliminated and valu1 is shifted and cycle(valu2) >= 1 + scalar_cycle, or (b) vb not eliminated and cycle(valu2) >= cycle(vb)+1. Optionally assert that when elimination is feasible, at least one solution has `eliminated_vbroadcast_ops` containing the vb’s op id.
- **Elimination implies shift**
  - Same fixture; assert that for every op id in `eliminated_vbroadcast_ops`, all its shiftable consumers are in `shifted_ops`.
- **Correctness of precedence**
  - Small SSA where vb has scalar producer at cycle 0; when vb is eliminated and consumer shifted, consumer’s ALU ops can be at cycle 1. Assert solution respects precedence (e.g. min(alu_cycles[valu1]) >= 1).
- **No elimination when consumer not shifted**
  - Force shift[vb’s consumer] = 0 (e.g. ForceValu for that op or a test that only shifts others). Assert vb is not in `eliminated_vbroadcast_ops`.

### Integration tests

- **Full pipeline**
  - Run `generate_ssa_kernel.py --enable-valu-shift --rounds 2 --output-json` (with vbroadcast elimination enabled). Validate kernel (reference comparison). Optionally assert `eliminated_vbroadcast_ops` in JSON is non-empty when quantification script reports profitably eliminable vbroadcasts.
- **Python interface**
  - In [tests/test_scheduler_rs_interface.py](tests/test_scheduler_rs_interface.py) or equivalent: build kernel with `enable_valu_shift=True`, then run Machine and compare to reference; ensure no regression when elimination is on.

### Regression and CI

- **Existing CP-SAT tests**
  - All existing `test_ssa_cpsat_*` and capacity-lower-bound tests must pass with `enable_vbroadcast_elimination: true` (and with `false` when gated). No change in behavior when the feature is disabled.
- **CI**
  - Run `cargo test --manifest-path scheduler_rs/Cargo.toml --features cpsat`; optional `ci-tests` feature for slower tests (e.g. full kernel run). Document in [CLAUDE.md](../CLAUDE.md) if new tests are under `ci-tests`.

### Optional property / stress

- **Proptest or small random SSA**
  - Generate small SSA graphs that include vb → shiftable → consumer; run CP-SAT with elimination enabled; assert solution is valid (precedence, capacity, and that eliminated vb set is consistent with shift set).

---

## Debugging tools

### JSON and Python output

- **Scheduler output JSON**
  - Add a top-level key, e.g. `eliminated_vbroadcast_ops: number[]` (op ids of vbroadcasts that were eliminated), written by the same path that writes `shifted_alu_cycles` / `schedule_constraints` ([scheduler_rs/src/python.rs](scheduler_rs/src/python.rs) `write_ssa_schedule_json`). This allows scripts and logs to inspect which vbroadcasts were eliminated without re-running the solver.
- **Usage**
  - Run `quantify_schedule_hypotheses.py` on a schedule JSON: it can compare “profitably eliminable” (from static analysis) with “actually eliminated” (from `eliminated_vbroadcast_ops`) and report agreement or divergence.

### Verbose logging (Rust)

- **At model build**
  - When `verbose` is true, log counts: number of vbroadcasts, number with at least one shiftable consumer, number given an optional interval (eliminable). Optionally list (vb op id, num shiftable consumers, scalar producer op id) for each eliminable vb.
- **At solve**
  - When `verbose` and a solution is found, log for each eliminable vb: op id, `eliminate_vb` value (true/false), and if true, list of consumer op ids that are shifted (should match all shiftable consumers). One line per eliminated vb is enough for debugging.

### Comparison with quantification script

- **Script enhancement**
  - In [scripts/quantify_schedule_hypotheses.py](scripts/quantify_schedule_hypotheses.py), if the input JSON contains `eliminated_vbroadcast_ops`, add a short section: “Actually eliminated (from solver): N vbroadcasts (ids …). Profitably eliminable (from static analysis): M. Overlap: …”. This gives a direct check that the solver’s decisions align with the script’s “combined transform” chains.
- **Discrepancies**
  - If solver eliminated fewer than “profitably eliminable”, possible causes: capacity/ALAP, time limit, or hinting. If solver eliminated a vb the script doesn’t list (e.g. different definition of “shiftable consumer”), that may indicate a bug or a difference in SSA structure (e.g. script uses op id, solver uses index).

### Bulk-move and schedule constraints

- **Bulk-move simulator**
  - Bulk-move uses `schedule_constraints` and `ssa_ops_bulk_move` / `ssa_cycle_assignments`; it does not need to know which vbroadcasts were eliminated (eliminated ops are simply not in the schedule). Ensure that when we omit eliminated vb from emission, the SSA indices / op ids in `schedule_constraints` and in bulk-move input remain consistent (e.g. constraints still refer to the same op indices; eliminated ops may have a “no-op” or placeholder cycle in assignments so indices line up). If the simulator expects one assignment per SSA op, document that eliminated vb ops still have a cycle assignment (e.g. from the solver’s cycle var) but are not emitted.
- **Validation**
  - After implementing, run bulk-move on a JSON produced with elimination enabled and confirm it doesn’t error (no missing indices, constraints still valid).

---

## Hinting the solver

### New hint field: `eliminate_vbroadcast_hints`

- **Add to `SsaCpsatHints`** ([scheduler_rs/src/cpsat_scheduler.rs](scheduler_rs/src/cpsat_scheduler.rs)):
  - `eliminate_vbroadcast_hints: Option<&'a HashSet<usize>>` — op ids of vbroadcasts that the solver should try to eliminate (hint `eliminate_vb[vb] = true` for those vb).
- **When building the model**
  - For each eliminable vb, if `eliminate_vbroadcast_hints` is `Some(hints)` and `hints.contains(&ops[vb].id)`, call `model.add_hint(eliminate_vb_var, 1)`. Otherwise no hint (or default 0). Hints are only applied when the vb is in the eliminable set (has shiftable consumers and scalar producer).
- **Semantics**
  - Hint means “prefer this vbroadcast eliminated”. The solver may still set eliminate_vb = 0 if it needs the vb (e.g. to satisfy capacity or other constraints). So hints are soft.

### Deriving hints from greedy

- **Greedy run first**
  - If the pipeline runs a greedy scheduler before CP-SAT (e.g. for warm start), the greedy can produce shift decisions. For each vbroadcast, check whether all its shiftable consumers are shifted in the greedy solution; if yes, add that vb’s op id to `eliminate_vbroadcast_hints`. Then pass this set into `SsaCpsatHints` along with `shift_hints` and cycle hints. This aligns “try to eliminate vb” with “try to shift these consumers” and can speed up convergence.
- **Implementation**
  - In the caller that builds hints (e.g. [kernel_builder/tick_tock_ssa.rs](scheduler_rs/src/kernel_builder/tick_tock_ssa.rs) or wherever greedy is run before CP-SAT): after greedy run, compute the set of vb op ids whose shiftable consumers are all in greedy’s shifted set; set `eliminate_vbroadcast_hints: Some(&that_set)`.

### Deriving hints from quantification script

- **Offline workflow**
  - User runs `quantify_schedule_hypotheses.py` on a previous JSON and gets “Vbroadcasts that can be profitably eliminated: …” and a table of (vb id, …). They can pass a subset of those vb ids as elimination hints for a re-run (e.g. via a small wrapper script or a future CLI flag that reads a “hint file”). No need to implement the file format in the first iteration; documenting “you can pass these op ids as eliminate_vbroadcast_hints” is enough.
- **Future**
  - If a `--hint-eliminate-vb` file or flag is added to the generator, it could read vb ids (one per line or JSON array) and pass them into the Rust builder as `eliminate_vbroadcast_hints`.

### Interaction with `shift_hints`

- **Consistency**
  - If the user provides `shift_hints` (e.g. “shift these op ids”) and also `eliminate_vbroadcast_hints` (“eliminate these vb”), the model should be consistent: for a vb to be eliminated, all its shiftable consumers must be shifted. So when applying hints, either (1) only hint eliminate_vb = 1 when the corresponding shift hints already imply all consumers shifted, or (2) hint both shift and eliminate and let the solver enforce the AND. Prefer (2) so that hinting eliminate_vb does not require the user to also pass full shift_hints; the solver will still enforce eliminate_vb = AND(shift[consumers]).
- **Override**
  - If `valu_shift_mode` is ForceValu for some ops, those ops cannot be shifted; so any vb that has such an op as its only shiftable consumer cannot be eliminated. The eliminable set is built after shift mode is known; hints for such a vb should be ignored (or the vb is not in the eliminable set).

---

## Tiebreaker objective

The CP-SAT model minimizes **makespan** with a **tie-breaker**; see the **canonical description** in `scheduler_rs/src/cpsat_scheduler/model.rs` (search for "CANONICAL DESCRIPTION" or "Objective = scale * makespan"). That block defines the formula and semantics; the tie-breaker rewards earlier placement and fills earlier slots even when that does not reduce makespan.

### No separate elimination term needed

**sum_completion already rewards vbroadcast elimination.** When a vb is eliminated, its shiftable consumers are forced to shift and use the scalar producer for precedence, so they can be scheduled **earlier** (e.g. completion at cycle 1 instead of vb_cycle+1). Their effective cycles (completion_cycle when shifted) therefore **decrease**. So minimizing sum_completion already favors solutions that eliminate vbroadcasts. **No separate objective term** for elimination is needed; the existing tie-breaker is sufficient. Use hints to steer when there are many equivalent options.


---

## Implementation checklist (recap)

| Area | Item |
|------|------|
| **Model** | Vbroadcast/scalar analysis; eliminate_vb vars and AND constraints; optional vb intervals; conditional precedence (section 5 + ALU block). |
| **Result** | Add `eliminated_vbroadcast_ops: HashSet<usize>`; extract from solution. |
| **Lowering** | Omit eliminated vb; rewrite deps to scalar for shifted consumers of eliminated vb. |
| **Config** | `enable_vbroadcast_elimination: bool` (default **false**; opt-in via `--enable-eliminate-vbroadcast`. Known buggy: scalar liveness not extended when shifted ALU reads it, can cause wrong results. For experimenting only.). |
| **Hints** | `eliminate_vbroadcast_hints: Option<&'a HashSet<usize>>`; apply to eliminate_vb vars; derive from greedy when available. |
| **Tiebreaker** | No change; existing `sum_completion` already rewards vbroadcast elimination (earlier consumer cycles). Use hints to steer when many equivalent options exist. |
| **JSON** | Write `eliminated_vbroadcast_ops` in scheduler output JSON. |
| **Tests** | Unit: minimal vb→shiftable→consumer; elimination implies shift; precedence; no elimination when consumer not shifted. Integration: full kernel + validate. Regression: all existing tests pass with feature on/off. |
| **Debug** | Verbose: log eliminable count and per-eliminated vb; script comparison section when JSON has eliminated set. |
| **Bulk-move** | Ensure constraint/assignment indices remain valid when vb are eliminated. |
