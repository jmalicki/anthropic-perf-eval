# Plan: Unify CP-SAT Scratch Model and Address Allocator Liveness

## Problem

We have two separate liveness/scratch models:

1. **Address allocator** (`scheduler_rs/src/address_assign.rs`): `compute_live_intervals()` + linear-scan allocation. Used *after* scheduling to assign physical scratch addresses.
2. **CP-SAT scratch model** (`scheduler_rs/src/cpsat_scheduler/model.rs`): Builds scratch intervals from `vreg_def_op`, `vreg_last_use`, `alu_cycle_vars`. Used *during* scheduling to enforce scratch budget.

These must be **consistent**: if CP-SAT says schedule X fits in budget, address assignment must also succeed with no conflicts. Today they are implemented separately and can drift (e.g. shifted VALU dest intervals, dead VRegs).

## Known Inconsistencies

| Scenario | Address assign | CP-SAT model | Impact |
|----------|----------------|--------------|--------|
| Shifted VALU dest (no consumer) | `[min_c, max_c]`; adjacent [22,23] and [23,24] incorrectly reuse address | Not modeled (dead VRegs excluded from `live_vregs`) | CP-SAT can find schedules that cause allocator conflict |
| Shifted VALU dest (with consumer) | `[min_c, consumer_cycle]`; start=min, end=consumer | start=min(alu_cycles), end=consumer_cycle+1 | Should match |
| Dead VRegs | Modeled; `[def, def+1]` | Excluded (DCE removes them in typical flow) | OK when DCE enabled |
| `sections_exceeding_budget` | N/A | Uses own logic (op.cycle, op.reads); no shifted_alu_cycles | Per-section budget check can mismatch |
| Half-open vs closed | Closed `[start, end]` | Half-open `[start, end)`; end = last_use_cycle + 1 | Documented; CP-SAT slightly more conservative |

## Goal

Single source of truth for liveness semantics. Both CP-SAT and address assign derive intervals from the same logic.

## Proposed Approach

### Option A: Shared interval computation

Extract liveness logic into a module used by both:

1. **New module** `scheduler_rs/src/liveness.rs` (or extend `address_assign`):
   - `compute_live_intervals(config, ops) -> HashMap<VReg, LiveInterval>`
   - Handles all special cases: SelectChoice FMA, shifted VALU, AluDerive, dead VRegs, vector groups, CSE.
   - Single implementation; both CP-SAT and address assign call it.

2. **CP-SAT model**:
   - After building cycle variables, call shared `compute_live_intervals` with the *scheduled* ops (cycles from solver or hints).
   - Convert closed intervals to half-open for cumulative constraint.
   - Problem: CP-SAT doesn't have cycle assignments until solve; intervals depend on cycle vars. So we can't call a single function with concrete cycles during model build.

### Option B: Shared semantics, separate construction

Keep separate construction but enforce the same rules:

1. **Define canonical semantics** in `types.rs` (extend AUTHORITATIVE section):
   - For each VReg type (normal, dead, shifted VALU dest, SelectChoice FMA, etc.): `[start, end]` rules.
   - "End" semantics: last read vs last write. Same-cycle reuse rule.

2. **CP-SAT model** explicitly implements those semantics:
   - For shifted VALU dests: when `def_idx` is shifted, `end >= completion_cycle + 1` (or `max(alu_cycles) + 1` in half-open) so write-at-boundary doesn't allow reuse.
   - Add dead shifted VALU dests to model when DCE is off (or document that DCE must be on).

3. **Address assign** uses same rules; add regression tests that CP-SAT solution + address assign yields no conflicts.

### Option C: Address assign drives CP-SAT feasibility check

Run address assign on greedy hint; if it fails, feed back "peak exceeded" or "conflict" to constrain CP-SAT. Complex; likely not worth it.

## Recommended: Option B + targeted fix for shifted VALU

1. **Fix shifted VALU bug correctly** in both places:
   - **Semantic rule:** For shifted VALU dest, storage is occupied until `max(alu_cycles) + 1` (reusable after last write completes).
   - **Address assign:** Use `end = max_c + 1` in `dest_write_interval` for shifted VALU.
   - **CP-SAT model:** For shifted VALU dests in `live_vregs`, ensure `end >= completion_cycle + 2` (half-open) when consumer is at completion+1. For shifted VALU dests with no consumer (if any), add interval `[min_c, max_c+2)`.

2. **Unify `sections_exceeding_budget`** with shared semantics:
   - Either call `AddressAssignConfig::compute_live_intervals` (requires building config with `shifted_alu_cycles`, etc.) and derive peak from that, or
   - Reimplement to use `shifted_alu_cycles` when computing per-section peaks so it matches address assign.

3. **Add cross-validation test:**
   - For a CP-SAT solution: run `assign_addresses`; assert no conflicts and `!budget_exceeded`.
   - Run on recorded configs (4r/256 depth-2, depth-3, etc.).

## Files to Touch

- `scheduler_rs/src/address_assign.rs` – canonical interval computation; fix shifted VALU.
- `scheduler_rs/src/cpsat_scheduler/model.rs` – scratch intervals must match address assign semantics.
- `scheduler_rs/src/kernel_builder/round_budget.rs` – `sections_exceeding_budget` should use same semantics (or delegate to address_assign).
- `scheduler_rs/src/types.rs` – extend AUTHORITATIVE liveliness docs with shifted VALU, end-is-write cases.

## Future: decompose_vector and independent lane liveness

Today, for read groups (decompose_vector), the vector VReg keeps one combined interval that we extend to cover all lane reads; lane VRegs have no intervals and share the group’s allocation. In the future we may want **decompose_vector to end the vector VReg’s liveness** and give each new scalar (lane) VReg **its own liveness interval** starting at the decomposition point. That would allow tighter packing when chains of ALU ops consume lanes at different times (e.g. scatter-load address lanes). See `emit_ssa.rs` `decompose_vector` doc comment and `address_assign.rs` vector group pass TODO.

## References

- BUGS.md: Same-cycle scratch write conflict (shifted VALU adjacent intervals)
- `types.rs`: AUTHORITATIVE LIVENESS MODEL DOCUMENTATION
- `model.rs` lines 1727–1755: CP-SAT scratch model comment block
- `address_assign.rs`: Liveness special cases table
