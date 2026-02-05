# Plan: Greedy Compaction and CP-SAT Custom Subsolver

**Artifacts:** Plan docs under `docs/` only. “Build the plan” = write or update documentation in `docs/`; no implementation code is produced or checked in.

Ordered phases and options; no implementation detail beyond what’s needed to execute the plan.

---

## Phase 1: Greedy compaction in Rust (no C++ changes)

**Goal:** Greedily move ops to earlier cycles when dependencies are ready and slots are available; use as post-processing or to improve hints.

1. **Add a compaction routine** (e.g. in `schedule_constraints.rs` or a small new module):
   - Input: feasible schedule (cycle assignments), SSA ops, constraints (precedence, ordering, domain), per-engine slot capacities per cycle.
   - For each cycle `C` from 0 upward: if `C` has free slots, try moving ops from `C+1` to `C` when:
     - Precedence, ordering, and domain (ASAP/ALAP) allow it (reuse logic from `check_schedule_constraints` / bulk-move),
     - Moving the op to `C` does not exceed per-engine slot capacity at `C`.
   - Repeat until no moves or a fixed number of rounds.
   - Output: updated cycle assignments (and possibly new makespan).

2. **Integrate with CP-SAT path:**
   - **Option A (post-process):** After `solve_with_parameters`, extract cycle assignment from response; run compaction; use compacted schedule as final result.
   - **Option B (hint loop):** Use compaction to improve current solution; feed result as new hints; call CP-SAT again; optionally iterate.

3. **Validate:** Compare makespan and correctness before/after compaction; ensure capacity is never exceeded.

**References:** `schedule_constraints.rs` (`check_schedule_constraints`, `run_bulk_move_simulator`), `.cursor/skills/use-bulk-move/SKILL.md`. Bulk-move does not check slot capacity; compaction must.

---

## Phase 2 (optional): Custom subsolver inside OR-Tools

**Goal:** Run the same greedy compaction inside the solver’s parallel portfolio (shared time limit, solution pool).

1. **Implement a C++ SubSolver** (e.g. `GreedyCompactionSubSolver`):
   - Extend `SubSolver`; implement `Synchronize()`, `IsDone()`, `TaskIsAvailable()`, `GenerateTask()`.
   - In the task: read current best solution from `shared->response`, run compaction (dependencies + slot capacity), push improved solution if feasible and better.
   - Register in `SolveCpModelParallel()` in `cp_model_solver.cc` (e.g. append to `reentrant_interleaved_subsolvers` or `interleaved_subsolvers`).

2. **Compaction implementation:** Either reimplement compaction in C++ or call Phase 1 Rust compaction via FFI from the SubSolver task.

3. **Build:** OR-Tools must be built from source with this change; the Rust `cp_sat` crate does not expose subsolver registration.

**References:** `ortools/sat/subsolver.h`, `ortools/sat/cp_model_solver.cc` (e.g. `LnsSolver`, `FullProblemSolver`, `SolveCpModelParallel`).

---

## Phase 3 (optional): Expose subsolver hook from Rust crate

**Goal:** Allow a Rust-provided callback to run as a subsolver task inside the C++ solver (e.g. run Phase 1 compaction from the portfolio without reimplementing in C++).

1. **C++:** Add a way to register an “external” subsolver or callback (e.g. SubSolver that holds a function pointer / stub that calls into Rust; or a parameter/API that adds this callback to the subsolver list). Callback must be able to read current solution and push a new one (e.g. via `SharedClasses` or a thin wrapper).

2. **Rust (`cp_sat` fork):** Add API to pass a callback (e.g. at solve time); FFI so C++ invokes it in the task loop with solution read/write (and optionally time limit). Crate passes callback through to C++ registration.

3. **Use:** Pass Phase 1 compaction as the callback so it runs as one of the subsolvers.

**References:** Same as Phase 2; plus `cp_sat` crate (e.g. `solve_with_parameters` and how it calls into C++).

---

## Decision summary

| Phase | Scope | When to do it |
|-------|--------|----------------|
| **1** | Rust compaction + post-process or hint loop | First; unblocks better schedules without C++ or crate changes. |
| **2** | C++ SubSolver in OR-Tools | If compaction in portfolio and sharing time/solutions is required. |
| **3** | Rust crate subsolver hook | If Phase 1 logic should run inside the solver from Rust without reimplementing in C++. |

Recommendation: implement Phase 1 only for now; add Phase 2 or 3 only if portfolio integration becomes necessary.
