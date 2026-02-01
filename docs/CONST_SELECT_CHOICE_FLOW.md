# ConstChoice and SelectChoice: Flow Through the System

This document describes how **ConstChoice** (Load vs AddImm for constants) and **SelectChoice** (vselect vs FMA for conditional selection) flow from emission through scheduling to lowering, and why each exists.

## Why These Choices Exist

### ConstChoice (constant materialization)

- **Load**: `const dest value` — uses 1 **Load** slot (2 per cycle). No dependencies.
- **AddImm**: `add_imm dest s0 value` — uses 1 **Flow** slot (1 per cycle). Reads address 0 (always 0 at startup).

The solver can trade Load pressure for Flow pressure. When Load slots are full, AddImm can still materialize constants using the single Flow slot.

### SelectChoice (conditional selection)

- **vselect**: `vselect dest cond true_val false_val` — 1 **Flow** op.
- **FMA path**: `sub = true_val - false_val`, then `fma dest cond sub false_val` — 2 **VALU** ops (or 1 VALU if precomputed delta).

The solver can trade Flow pressure for VALU pressure. When Flow is saturated, FMA may fit; when VALU is saturated, vselect may fit.

---

## End-to-End Flow

```
Emission          DCE              Scheduling              Lowering
   │                │                    │                     │
   ▼                ▼                    ▼                     ▼
emit_const_choice  atomic          CP-SAT / Greedy      const_add_imm_ops
emit_const         (keep/remove    (cycle + choice      → emit const OR add_imm
emit_select_choice  as unit)        vars)               select_fma_ops
                                        │                → emit vselect OR sub+fma
                                        ▼
                               const_add_imm_ops (op ids)
                               select_fma_ops (op ids)
```

---

## ConstChoice

### Two emission paths

1. **emit_const_choice()** (single op)
   - Emits one SSA op with `Opcode::ConstChoice`, `dest`, `produces_constant`, and `const_alternatives: Some([Load, AddImm { value }])`.
   - Used in SSA kernel for bootstrap constants (e.g. c[1], c[2], c[4], partition_val_offset).
   - **Scheduling**: CP-SAT gets one op per constant and a boolean `const_choice[idx]` (true = AddImm, false = Load). Same `cycle[idx]` for both encodings; optional Load and Flow intervals.
   - **Lowering**: `lower_ssa_ops_with_shift` sees `Opcode::ConstChoice`. If `ssa_op.id ∈ const_add_imm_ops` → emit AddImm; else → emit Const. **Choice is made at lowering time.**

2. **emit_const()** (dual op — legacy)
   - Emits two SSA ops: one `Opcode::Const` (Load) and one `Opcode::AddImm` (Flow), same `dest`. Tracked in `const_alternatives` (load_op_idx, add_imm_op_idx).
   - Used for hash stages, offsets, etc. (see `ssa_emit.rs`).
   - **Why not ConstChoice here:** Per-section scratch budget model does not yet account for ConstChoice decisions; filtering avoids SSA violations (two writers to one VReg).
   - **Scheduling**: Before CP-SAT, AddImm ops are **filtered out** (tick_tock_ssa.rs). CP-SAT only sees Load ops. Greedy also skips AddImm alternatives.
   - **Lowering**: Only Load (Const) ops remain in the list; each is lowered to `const`. **No choice at lowering** — we fixed Load at filter time.

**DCE**: ConstChoice pairs (emit_const Load+AddImm) are treated as **atomic**: if the constant is live, both encodings are kept; if dead, both are removed. Single-op ConstChoice is one op and is kept or removed normally.

---

## SelectChoice

### Emission

- **emit_select_choice(cond, true_val, false_val)** and **emit_select_choice_with_delta(cond, delta, base)**.
- One SSA op with `Opcode::SelectChoice`, `dest`, and operands. Optionally `sub_temp_vreg` (for FMA path without delta) and `delta_vreg` (precomputed delta).
- No separate “vselect op” and “sub+fma ops” in the list — it’s a single op with two encodings chosen later.

### Scheduling

- CP-SAT has one op per SelectChoice and a boolean `select_choice[idx]` (true = FMA, false = vselect).
- Same `cycle[idx]`; optional Flow interval (vselect) vs optional VALU interval(s) (FMA: 1 or 2 ops, possibly sub at c and fma at c+1).
- Greedy can pass hints (`select_fma_ops`: op ids that chose FMA). Default is vselect.

### Lowering

- `lower_ssa_ops_with_shift` sees `Opcode::SelectChoice`.
- If `ssa_op.id ∈ select_fma_ops` → emit sub + fma (or single fma if delta_vreg set); else → emit vselect.
- **Choice is made at lowering time.**

**DCE**: SelectChoice is a single op (no “other encoding” in the SSA list). It is kept or removed like any other op; no special pairing.

---

## Summary Table

| Aspect              | ConstChoice (emit_const_choice) | ConstChoice (emit_const)     | SelectChoice        |
|---------------------|----------------------------------|------------------------------|----------------------|
| SSA ops emitted     | 1 (ConstChoice)                 | 2 (Const + AddImm)           | 1 (SelectChoice)     |
| CP-SAT models choice| Yes                              | No (AddImm filtered out)     | Yes                  |
| Choice made when    | Lowering (const_add_imm_ops)    | Never (always Load)          | Lowering (select_fma_ops) |
| DCE                 | Single op                        | Atomic pair                  | Single op            |

---

## Key Files

- **Emission**: `scheduler_rs/src/kernel_builder/emit_ssa.rs` — `emit_const_choice`, `emit_const`, `emit_select_choice`, `emit_select_choice_with_delta`.
- **DCE**: `emit_ssa.rs` — `eliminate_ssa_dead_code`; ConstChoice pairs kept/removed via `const_choice_partner` and `mark_live_atomic`.
- **Filtering (emit_const path)**: `scheduler_rs/src/kernel_builder/tick_tock_ssa.rs` — detect AddImm alternatives, filter before CP-SAT.
- **CP-SAT model**: `scheduler_rs/src/cpsat_scheduler.rs` — ConstChoice and SelectChoice decision variables, optional intervals.
- **Lowering**: `scheduler_rs/src/kernel_builder/ssa_lower.rs` — `lower_ssa_ops_with_shift`; ConstChoice → const or add_imm; SelectChoice → vselect or sub+fma.

---

## Unifying ConstChoice at Lowering

To have **all** constants choose Load vs AddImm at lowering (including those from emit_const()):

1. **Option A**: Use `emit_const_choice()` everywhere and fix the per-section scratch budget model so it accounts for ConstChoice. Then every constant is one op and we choose at lowering.
2. **Option B**: Keep emit_const() dual ops but model them in CP-SAT like ConstChoice (one logical op per pair, optional Load vs AddImm). Output which encoding was chosen per pair; at lowering emit only the chosen op (skip the other). Requires passing pair info and decisions into lowering.

See comments in `ssa_emit.rs` and `ssa_lower.rs` for the same options.
