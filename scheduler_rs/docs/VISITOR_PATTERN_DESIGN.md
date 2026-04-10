# Visitor/Trait Pattern Design for Compiler Module

**Status:** Design Phase (no code changes)
**Date:** 2026-04-10
**Scope:** `scheduler_rs/src/` — lowering, scheduling, address assignment, emission

---

## 1. Executive Summary

The scheduler/compiler module has several functions that exceed 300–500 lines, dominated by large `match`/`if-else` chains dispatching on opcode, engine, or encoding variant. This document proposes three visitor/trait interfaces — `OpLoweringVisitor`, `PriorityHeuristic`, and `OpEmitVisitor` — that replace these chains with extensible, testable trait implementations. The design is consistent with the existing `Pass` trait (pipeline) and `PressurePolicy` trait (scheduling heuristics) already in the codebase.

---

## 2. Current Pain Points

### 2.1 `lower_ssa_ops_with_shift()` — 450+ lines of opcode/encoding dispatch

**File:** `src/kernel_builder/ssa_lower.rs`, lines 184–630

The main loop iterates over SSA ops and dispatches on:

1. **Eliminated vbroadcasts** — skip
2. **Shifted VALU ops** — expand to 8 ALU ops via `expand_ssa_valu_to_alu`
3. **ConstChoice** — 3-way match on `ConstChoiceEncoding`:
   - `AluDerive(info)` → ALU op (~25 lines)
   - `AddImm` → Flow op (~20 lines)
   - `Load` / `None` → Load op (~15 lines)
4. **SelectChoice** — 2-way match on `SelectChoiceEncoding`:
   - `Fma` → sub + fma pair (~80 lines with CSE shared-sub logic)
   - `Vselect` / default → single Flow op (~25 lines)
5. **Default** — delegate to `lower_ssa_op` from `address_assign.rs`

**Why this is painful:**
- Adding a new encoding (e.g., `SubImm`, `ShiftLeft`) requires editing a 500-line function
- ConstChoice and SelectChoice lowering are interleaved with CSE pre-amble and vbroadcast elimination
- Hard to unit-test individual lowering paths in isolation
- The function mixes infrastructure (CSE map building, vbroadcast remapping) with per-opcode logic

### 2.2 `schedule_greedy_with_diagnostics()` — 527 lines with priority dispatch

**File:** `src/scheduler.rs`, lines 571–1097

The priority computation is in `compute_priority()` (lines 257–301), which is a match on `PriorityMode` with 8 arms. The main scheduling loop also has if/else chains for:
- Engine-based slot dispatch (`Valu` vs `Load` vs `Flow` etc.)
- VALU shift decisions
- Budget-deferred emergency fallback

**Why this is painful:**
- Adding a new priority mode requires editing `compute_priority` and potentially the main loop
- The `PriorityMode` enum and `compute_priority` function are tightly coupled — no way to add a heuristic without modifying the enum
- Some modes share logic (e.g., `LoadEarlySlack` = `LoadEarly` + `Slack`) but this is encoded implicitly

### 2.3 `lower_ssa_op()` — per-opcode slot building with implicit conventions

**File:** `src/address_assign.rs`, lines 1176–1300

This function builds the `slot` array for any `SsaOp` by:
1. Pushing `opcode.as_str()` as the first slot
2. Pushing dest address
3. Pushing source addresses/immediates
4. Building `reads`/`writes` sets based on vector lane expansion

**Why this is painful:**
- No per-opcode customization hook (e.g., `Vstore` has different slot layout)
- Reads/writes computation is formulaic but embedded inline
- Not extensible for future opcodes with non-standard slot layouts

### 2.4 `emit_valu_selection_init()`, `emit_hash()`, `emit_branch()` — long emission methods

**File:** `src/kernel_builder/ssa_emit.rs`

- `emit_valu_selection_init()` (lines 273–363, 91 lines) — tree cache loading and delta computation
- `emit_hash()` (lines 1084–1175, 92 lines) — 3-way dispatch per hash stage (FMA, stage-5 special, general)
- `emit_branch()` (lines 1181–1261, 81 lines) — wrap vs normal branching, partition stores

**Why these are painful:**
- `emit_hash` has a 3-level nested loop with conditional logic per stage type
- Adding a new hash stage pattern requires editing inside the loop body
- `emit_branch` mixes index computation with partition store emission

### 2.5 `compute_live_intervals()` — complex liveness with per-opcode special cases

**File:** `src/address_assign.rs`, lines 319–500+

This function has already been partially refactored with `LivenessDef` and `LivenessUse` traits (lines 181–284), but `OpLiveness` still uses if/else chains on opcode/context rather than a proper visitor.

---

## 3. Existing Trait Patterns in the Codebase

The codebase already uses trait-based patterns that the new design should follow:

### 3.1 `Pass` trait (`kernel_builder/pipeline/mod.rs`)

```rust
pub trait Pass {
    fn name(&self) -> &\'static str;
    fn run(&self, ctx: &mut PipelineContext) -> Result<String, String>;
}
```

- Unit structs implement `Pass` (e.g., `ComputeDependenciesPass`, `GreedySchedulePass`)
- `PipelineContext::run_pass()` handles timing and report recording
- Each pass is independently testable

### 3.2 `PressurePolicy` trait (`kernel_builder/ssa_greedy/pressure_policy/mod.rs`)

```rust
pub trait PressurePolicy: std::fmt::Debug {
    fn sort_ready(&self, candidates: &mut [SchedulingCandidate], ctx: &SchedulingContext);
    fn should_defer_for_pressure(&self, live_words: usize, words_added: usize, words_freed: usize, scratch_budget: usize) -> bool;
    fn allow_shift(&self, live_words: usize, alu_used: usize, scratch_budget: usize) -> bool;
    fn compute_asap_inflation(&self, candidate: &SchedulingCandidate, ctx: &SchedulingContext) -> Option<i32>;
}
```

- 5 implementations: `GreedyPolicy`, `PressureThrottledPolicy`, `PressureGatedPolicy`, `AdaptivePolicy`, `CascadePolicy`
- Context struct (`SchedulingContext`) provides read-only snapshot
- Default method implementations for backward compatibility

### 3.3 `LivenessDef` / `LivenessUse` traits (`address_assign.rs`)

```rust
pub trait LivenessDef {
    fn dest_write_interval(&self, op: &SsaOp, ctx: &LivenessContext<\'_>) -> Option<(i32, i32)>;
    fn aux_defs(&self, op: &SsaOp, ctx: &LivenessContext<\'_>) -> Vec<(VReg, i32, i32)>;
}
pub trait LivenessUse {
    fn use_cycles(&self, op: &SsaOp, ctx: &LivenessContext<\'_>) -> Vec<(VReg, i32)>;
}
```

- Partially refactored — `OpLiveness` dispatcher still uses if/else chains
- Demonstrates the pattern: trait + context struct + dispatcher

---

## 4. Proposed Trait Definitions

### 4.1 `OpLoweringVisitor` — replaces the big match in `lower_ssa_ops_with_shift`

**New file:** `src/kernel_builder/ssa_lower/lowering_visitor.rs` (or `src/kernel_builder/lowering/`)

```rust
/// Context available to all lowering methods.
/// Extracted from the current `LoweringContext` fields that are per-op.
pub struct OpLoweringContext<'a> {
    pub ssa_op: &'a SsaOp,
    pub vreg_to_addr: &'a HashMap<VReg, i64>,
    pub ssa_ops: &'a [SsaOp],              // For cross-op lookups (AluDerive)
    pub cse_sub_addr: Option<i64>,          // Shared CSE sub addr for this op (if in group)
    pub eliminated_vb_map: &'a HashMap<VReg, i64>,  // Eliminated vbroadcast remap
}

/// Result of lowering a single SSA op: zero or more concrete `Op`s.
pub type LoweringResult = Result<Vec<Op>, String>;

/// Visitor trait for SSA op lowering.
///
/// Each method handles one opcode/encoding category and returns zero or more
/// concrete `Op`s. The default implementations return `None` (unhandled),
/// causing the dispatcher to fall through to the next handler or the
/// generic `lower_ssa_op` fallback.
pub trait OpLoweringVisitor {
    /// Lower a shifted VALU op to 8 scalar ALU ops.
    fn lower_shifted_valu(&self, ctx: &OpLoweringContext<'_>) -> Option<LoweringResult> {
        None // default: not handled
    }

    /// Lower a ConstChoice op based on its encoding decision.
    fn lower_const_choice(
        &self,
        ctx: &OpLoweringContext<'_>,
        encoding: &ConstChoiceEncoding,
    ) -> Option<LoweringResult> {
        None
    }

    /// Lower a SelectChoice op based on its encoding decision.
    fn lower_select_choice(
        &self,
        ctx: &OpLoweringContext<'_>,
        encoding: &SelectChoiceEncoding,
    ) -> Option<LoweringResult> {
        None
    }

    /// Lower a standard (non-choice, non-shifted) SSA op.
    /// Called for all ops not handled by the specific methods.
    fn lower_default(&self, ctx: &OpLoweringContext<'_>) -> LoweringResult;
}
```

**Primary implementation:** `DefaultLoweringVisitor` — moves the current logic from `lower_ssa_ops_with_shift` into trait methods:

```rust
pub struct DefaultLoweringVisitor;

impl OpLoweringVisitor for DefaultLoweringVisitor {
    fn lower_shifted_valu(&self, ctx: &OpLoweringContext<'_>) -> Option<LoweringResult> {
        // Extract the current expand_ssa_valu_to_alu logic
        Some(expand_ssa_valu_to_alu(ctx.ssa_op, ctx.vreg_to_addr, ...))
    }

    fn lower_const_choice(
        &self,
        ctx: &OpLoweringContext<'_>,
        encoding: &ConstChoiceEncoding,
    ) -> Option<LoweringResult> {
        match encoding {
            ConstChoiceEncoding::AluDerive(info) => Some(Self::lower_alu_derive(ctx, info)),
            ConstChoiceEncoding::AddImm => Some(Self::lower_add_imm(ctx)),
            ConstChoiceEncoding::Load => Some(Self::lower_const_load(ctx)),
        }
    }

    fn lower_select_choice(
        &self,
        ctx: &OpLoweringContext<'_>,
        encoding: &SelectChoiceEncoding,
    ) -> Option<LoweringResult> {
        match encoding {
            SelectChoiceEncoding::Fma => Some(Self::lower_select_fma(ctx)),
            SelectChoiceEncoding::Vselect => Some(Self::lower_select_vselect(ctx)),
        }
    }

    fn lower_default(&self, ctx: &OpLoweringContext<'_>) -> LoweringResult {
        lower_ssa_op(ctx.ssa_op, ctx.vreg_to_addr).map(|op| vec![op])
    }
}
```

**Dispatcher function** (replaces the main loop of `lower_ssa_ops_with_shift`):

```rust
pub fn dispatch_lowering(
    visitor: &dyn OpLoweringVisitor,
    ctx: &LoweringContext<'_>,
) -> Result<Vec<Op>, String> {
    verify_vreg_mapping_complete(ctx)?;

    // Pre-compute CSE maps, eliminated vbroadcast maps (infrastructure)
    let cse_map = build_cse_sub_addr_map(ctx);
    let vb_remap = build_eliminated_vb_map(ctx);

    // Emit CSE shared SUBs first
    let mut result = emit_cse_shared_subs(ctx);

    for ssa_op in ctx.ssa_ops {
        if should_skip(ssa_op, ctx) { continue; }

        let op_ctx = OpLoweringContext {
            ssa_op, vreg_to_addr: ctx.vreg_to_addr, ssa_ops: ctx.ssa_ops,
            cse_sub_addr: cse_map.get(&ssa_op.id).copied(),
            eliminated_vb_map: &vb_remap,
        };

        let ops = if ctx.shifted_ops.contains(&ssa_op.id) {
            visitor.lower_shifted_valu(&op_ctx)
                .unwrap_or_else(|| visitor.lower_default(&op_ctx))
        } else if ssa_op.opcode == Opcode::ConstChoice {
            let enc = ctx.const_choice_encoding.get(&ssa_op.id);
            match enc {
                Some(e) => visitor.lower_const_choice(&op_ctx, e)
                    .unwrap_or_else(|| visitor.lower_default(&op_ctx)),
                None => visitor.lower_default(&op_ctx),
            }
        } else if ssa_op.opcode == Opcode::SelectChoice {
            let enc = ctx.select_choice_encoding.get(&ssa_op.id);
            match enc {
                Some(e) => visitor.lower_select_choice(&op_ctx, e)
                    .unwrap_or_else(|| visitor.lower_default(&op_ctx)),
                None => visitor.lower_default(&op_ctx),
            }
        } else {
            visitor.lower_default(&op_ctx)
        }?;

        result.extend(ops);
    }
    Ok(result)
}
```

---

### 4.2 `PriorityHeuristic` — replaces `compute_priority` + `PriorityMode` enum

**New file:** `src/scheduler/priority_heuristic.rs` (or extend existing `src/scheduler.rs`)

```rust
/// Context for priority computation (read-only snapshot).
pub struct PriorityContext<'a> {
    pub critical_path: &'a HashMap<usize, usize>,
    pub slack_map: &'a HashMap<usize, i32>,
    pub luc_map: &'a HashMap<usize, usize>,
    pub asap_map: &'a HashMap<usize, i32>,
    pub alap_map: &'a HashMap<usize, i32>,
}

/// Trait for scheduling priority heuristics.
///
/// Replaces the `PriorityMode` enum + `compute_priority` function.
/// Each implementation produces a comparable priority tuple for a ready op.
pub trait PriorityHeuristic {
    /// Compute priority for an op. Lower return value = higher scheduling priority.
    /// Returns a tuple for lexicographic comparison.
    fn priority(
        &self,
        op_id: usize,
        op_idx: usize,
        engine: Engine,
        ctx: &PriorityContext<'_>,
    ) -> (i32, i32, i32, usize);

    /// Human-readable name for diagnostics.
    fn name(&self) -> &'static str;
}
```

**Implementations** (one per current `PriorityMode` variant):

```rust
pub struct CriticalPathHeuristic;
pub struct CriticalPathValuHeuristic;
pub struct EngineBalanceHeuristic;
pub struct SlackHeuristic;
pub struct LoadEarlyHeuristic;
pub struct LoadEarlySlackHeuristic;
pub struct OriginalOrderHeuristic;
pub struct ScratchPressureHeuristic;
```

Each implementation extracts the relevant fields from `PriorityContext` and returns the same priority tuple as the current `compute_priority` function. For example:

```rust
impl PriorityHeuristic for CriticalPathHeuristic {
    fn priority(
        &self, op_id: usize, op_idx: usize, engine: Engine, ctx: &PriorityContext<'_>,
    ) -> (i32, i32, i32, usize) {
        let cp = *ctx.critical_path.get(&op_id).unwrap_or(&1) as i32;
        let luc = *ctx.luc_map.get(&op_id).unwrap_or(&0) as i32;
        let is_valu = if engine == Engine::Valu { 0 } else { 1 };
        (-cp, is_valu, -luc, op_id)
    }
    fn name(&self) -> &'static str { "critical_path" }
}
```

**Bridge:** To maintain backward compatibility, `PriorityMode` gains a `to_heuristic()` method that returns `Box<dyn PriorityHeuristic>`. The `schedule_greedy_with_diagnostics` function calls `heuristic.priority(...)` instead of `compute_priority(...)`.

---

### 4.3 `OpEmitVisitor` — simplifies hash/branch emission dispatch

**New file:** `src/kernel_builder/ssa_emit/emit_visitor.rs`

```rust
/// Trait for emitting a single hash stage for one group.
///
/// Replaces the 3-way if/else inside the emit_hash loop body.
pub trait HashStageEmitter {
    /// Emit ops for hash stage `hi` on group `g` with current value `val`.
    /// Returns the updated value VReg.
    fn emit_stage(
        &self,
        builder: &mut SsaTickTockKernelBuilder,
        hi: usize,
        g: usize,
        val: VReg,
    ) -> VReg;
}

/// Trait for branch phase emission.
pub trait BranchEmitter {
    /// Emit the index update for group `g`.
    fn emit_index_update(
        &self,
        builder: &mut SsaTickTockKernelBuilder,
        g: usize,
        is_wrap_round: bool,
    );
}
```

**Implementations:**

```rust
/// FMA-based hash stage (uses multiply_add).
pub struct FmaHashStage;

/// Stage-5 special handling (Shr + Xor).
pub struct Stage5HashStage;

/// General hash stage (Add/Xor + Shift).
pub struct GeneralHashStage {
    pub op1: char, pub op2: char, pub op3: char,
}
```

The `emit_hash` method selects the right emitter based on `hash_fma_mult[hi]` and stage index, then calls `emitter.emit_stage(self, hi, g, val)`.

---

### 4.4 Completing the `LivenessDef`/`LivenessUse` refactoring

The existing `LivenessDef`/`LivenessUse` traits in `address_assign.rs` already demonstrate the right pattern but `OpLiveness` uses if/else chains. The proposed refactoring:

```rust
/// Per-opcode liveness behavior. Each opcode category implements this trait.
pub trait OpLivenessBehavior {
    fn dest_write_interval(&self, op: &SsaOp, ctx: &LivenessContext<'_>) -> Option<(i32, i32)>;
    fn aux_defs(&self, op: &SsaOp, ctx: &LivenessContext<'_>) -> Vec<(VReg, i32, i32)>;
    fn use_cycles(&self, op: &SsaOp, ctx: &LivenessContext<'_>) -> Vec<(VReg, i32)>;
}

/// Registry: maps opcodes to their liveness behavior.
pub fn liveness_behavior_for(op: &SsaOp, ctx: &LivenessContext<'_>) -> &dyn OpLivenessBehavior {
    match op.opcode {
        Opcode::SelectChoice if ctx.select_fma_ops.contains(&op.id) => &SelectChoiceFmaLiveness,
        _ => &DefaultLiveness,
    }
}
```

---

## 5. Migration Strategy

The refactoring must be **incremental** — each step produces a compiling, passing codebase. No big-bang rewrite.

### Phase 1: `OpLoweringVisitor` (Highest Impact, Lowest Risk)

**Step 1.1:** Extract infrastructure from `lower_ssa_ops_with_shift`
- Move CSE map building (lines 193–219) to `build_cse_sub_addr_map(ctx) -> HashMap<usize, i64>`
- Move CSE shared SUB emission (lines 222–270) to `emit_cse_shared_subs(ctx) -> Vec<Op>`
- Move eliminated vbroadcast map building (lines 273–294) to `build_eliminated_vb_map(ctx) -> HashMap<VReg, i64>`
- **Tests:** All existing tests in `ssa_lower.rs` pass unchanged (these are integration tests calling `lower_ssa_ops_with_shift`)

**Step 1.2:** Define the `OpLoweringVisitor` trait and `OpLoweringContext`
- Create new module `src/kernel_builder/ssa_lower/` (move `ssa_lower.rs` → `ssa_lower/mod.rs`)
- Add `lowering_visitor.rs` with the trait definition
- **Tests:** No behavior change yet; trait exists but is not used

**Step 1.3:** Implement `DefaultLoweringVisitor`
- Extract each lowering path into a method on `DefaultLoweringVisitor`:
  - `lower_shifted_valu` ← current lines 300–314
  - `lower_const_choice(AluDerive)` ← current lines 324–368
  - `lower_const_choice(AddImm)` ← current lines 370–400
  - `lower_const_choice(Load)` ← current lines 402–426
  - `lower_select_choice(Fma)` ← current lines 465–630
  - `lower_select_choice(Vselect)` ← current lines ~430–464
- **Tests:** All existing tests still pass via `lower_ssa_ops_with_shift` which now delegates to `DefaultLoweringVisitor`

**Step 1.4:** Add `dispatch_lowering` and make `lower_ssa_ops_with_shift` a thin wrapper
- `lower_ssa_ops_with_shift(ctx)` becomes:
  ```rust
  pub fn lower_ssa_ops_with_shift(ctx: &LoweringContext<'_>) -> Result<Vec<Op>, String> {
      dispatch_lowering(&DefaultLoweringVisitor, ctx)
  }
  ```
- **Tests:** All existing tests pass; new unit tests for each `DefaultLoweringVisitor` method

**Step 1.5:** Add per-method unit tests
- Test `lower_const_choice(AddImm)` in isolation (construct minimal `OpLoweringContext`)
- Test `lower_select_choice(Fma)` in isolation
- Test `lower_shifted_valu` in isolation
- **Tests:** New test coverage for previously untestable code paths

### Phase 2: `PriorityHeuristic` (Medium Impact, Medium Risk)

**Step 2.1:** Define the `PriorityHeuristic` trait
- Add `src/scheduler/priority_heuristic.rs` (or section within `scheduler.rs`)
- Define trait with `priority()` and `name()` methods

**Step 2.2:** Implement one heuristic per `PriorityMode` variant
- Copy logic from `compute_priority` match arms into individual struct impls
- Add `PriorityMode::to_heuristic(&self) -> Box<dyn PriorityHeuristic>` bridge method

**Step 2.3:** Replace `compute_priority` calls with `heuristic.priority()`
- In `schedule_greedy_with_diagnostics`: call `mode.to_heuristic().priority(...)` instead of `compute_priority(...)`
- **Tests:** All existing `scheduler.rs` tests pass (they exercise `PriorityMode` variants)

**Step 2.4:** Deprecate `compute_priority` (mark `#[deprecated]`)
- New callers use `PriorityHeuristic` directly
- Old callers go through bridge

### Phase 3: `HashStageEmitter` + `BranchEmitter` (Lower Impact)

**Step 3.1:** Define `HashStageEmitter` trait
- Extract the 3-way dispatch in `emit_hash` into trait
- Implementations: `FmaHashStage`, `Stage5HashStage`, `GeneralHashStage`

**Step 3.2:** Refactor `emit_hash` to use emitter registry
- Build a `Vec<Box<dyn HashStageEmitter>>` indexed by stage number
- Replace inner loop body with `emitters[hi].emit_stage(self, hi, g, val)`

**Step 3.3:** Define `BranchEmitter` trait
- Separate `emit_index_update` from partition store emission
- **Tests:** Existing `tick_tock_ssa/tests.rs` and `ssa_greedy/tests.rs` pass

### Phase 4: Complete `LivenessDef`/`LivenessUse` refactoring (Lowest Priority)

**Step 4.1:** Replace `OpLiveness` if/else with `OpLivenessBehavior` trait objects
- Extract `SelectChoiceFmaLiveness` and `DefaultLiveness` as separate structs
- Add `liveness_behavior_for()` registry function

**Step 4.2:** Unit test per-opcode liveness behavior
- Test `SelectChoice` FMA def interval (`cycle+1`)
- Test shifted VALU def interval (`min(alu_cycles)`)
- Test default def interval (`op.cycle`)

---

## 6. Files and Modules Affected

| File | Change | Phase |
|------|--------|-------|
| `src/kernel_builder/ssa_lower.rs` → `src/kernel_builder/ssa_lower/mod.rs` | Rename to module directory | 1.2 |
| `src/kernel_builder/ssa_lower/lowering_visitor.rs` | **New:** `OpLoweringVisitor` trait + `DefaultLoweringVisitor` | 1.2–1.3 |
| `src/kernel_builder/ssa_lower/cse.rs` | **New:** CSE infrastructure extracted | 1.1 |
| `src/kernel_builder/ssa_lower/vb_elimination.rs` | **New:** Vbroadcast elimination map | 1.1 |
| `src/kernel_builder/mod.rs` | Update `mod ssa_lower` declaration | 1.2 |
| `src/kernel_builder/pipeline/passes/lower.rs` | May reference new module path | 1.2 |
| `src/scheduler.rs` | Add `PriorityHeuristic` implementations | 2.1–2.3 |
| `src/kernel_builder/ssa_emit.rs` → split | Hash/branch visitor extraction | 3.1–3.3 |
| `src/address_assign.rs` | Complete `OpLivenessBehavior` refactoring | 4.1 |

### No changes needed in:
- `src/types.rs` — `Opcode`, `Engine`, `Op`, `SsaOp` remain unchanged
- `src/kernel_builder/pipeline/mod.rs` — `Pass` trait is unaffected
- `src/kernel_builder/ssa_greedy/` — `PressurePolicy` is the model, not the target
- `src/cpsat_scheduler/` — CP-SAT model is independent

---

## 7. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Trait dispatch overhead (dyn dispatch in hot loop) | `PriorityHeuristic` can use monomorphization (`impl PriorityHeuristic`) instead of `dyn`; `OpLoweringVisitor` runs once per op (N=~200–1000), dyn overhead negligible |
| Refactoring breaks lowering correctness | Each phase has existing integration tests; new per-method unit tests add coverage |
| `ssa_lower.rs` → module directory rename breaks imports | Use `mod.rs` pattern; update `use crate::kernel_builder::ssa_lower::*` in one commit |
| `PriorityMode` enum is part of Python FFI | Bridge method `PriorityMode::to_heuristic()` preserves enum API; Python layer unchanged |
| `emit_hash` refactoring changes emission order | Hash stages emit sequentially within each group; visitor preserves ordering |

---

## 8. Design Principles

1. **Consistency with existing patterns:** Follow `Pass` trait style (unit structs + `run` method) and `PressurePolicy` style (context struct + default methods).
2. **Opt-in extensibility:** Default implementations return `None`/no-op, so new visitors only override methods they care about.
3. **Incremental migration:** Each step is independently mergeable. No phase depends on a later phase.
4. **No behavioral change:** All existing tests pass at every step. New trait methods produce identical results to the extracted code.
5. **Testability:** Per-method extraction enables unit tests that were impossible when logic was embedded in 500-line functions.

---

## 9. Future Extensions Enabled

Once the visitor pattern is in place:

- **New constant encodings:** Add `SubImm` or `ShiftLeft` as a new `ConstChoiceEncoding` variant and implement only `lower_const_choice` in a new visitor.
- **New selection strategies:** Add a third `SelectChoiceEncoding` variant (e.g., `CascadeSelect`) without modifying existing lowering code.
- **Custom priority heuristics:** Researchers can implement `PriorityHeuristic` and plug into the scheduler without touching `PriorityMode`.
- **New hash stages:** Add `NewHashStage` implementing `HashStageEmitter` without modifying `emit_hash`.
- **Instruction-set extensions:** New opcodes get their own lowering visitor method instead of being wedged into a catch-all match arm.
