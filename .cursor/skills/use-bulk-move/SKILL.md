---
name: use-bulk-move
description: Run the bulk-move simulator to debug schedule makespan and identify which constraints block compaction. Use when optimizing the CP-SAT schedule, investigating empty cycles, or when the user asks which constraints are holding the schedule back.
---

# Using the Bulk-Move Tool

The **bulk-move simulator** checks whether empty cycles in a CP-SAT schedule could be removed by repeatedly "filling" empty cycle C by moving ops from C+1 to C. It uses the **same constraints as the CP-SAT model** (precedence, ordering, domain). When a move is impossible, it reports the **exact constraint** that blocks it—so you can see which constraint type is holding the schedule back.

## When to Use

- **Debugging high makespan** — See if empty cycles could be compressed (ReachedEnd) or which constraint blocks compaction (Blocked).
- **Finding tight constraints** — Blocked result gives you the constraint category (Precedence / Ordering / Domain) and op indices.
- **Checking solver quality** — ReachedEnd means the schedule had unnecessary empty cycles; the solver or tie-breaker left gaps.

## Prerequisites

1. **SSA schedule JSON** from a CP-SAT run with `--output-json`:
   ```bash
   uv run generate_ssa_kernel.py --output-json scheduler_output.json
   ```
   The file must contain `schedule_constraints` and `ssa_ops_bulk_move` / `ssa_cycle_assignments` (greedy output does not).

2. **scheduler_rs built** from project root:
   ```bash
   ./scripts/build_scheduler_rs.sh
   ```

## How to Run

```bash
# Default: scheduler_output.json
uv run scripts/simulate_bulk_move.py

# Custom path
uv run scripts/simulate_bulk_move.py path/to/schedule.json

# Step-by-step: print each bulk-move step (C and ops moved)
uv run scripts/simulate_bulk_move.py --verbose path.json
```

## Interpreting Results

| Result | Meaning | Next step |
|--------|--------|-----------|
| **NoEmptyCycles** | No empty cycles; schedule is already compact. | Nothing to compact; look at slot utilization or other optimizations. |
| **ReachedEnd** | All empty cycles were pushed to the end; makespan could have been lower. | Solver/tie-breaker left gaps. Consider improving objective or tie-breaking so the solver doesn't leave empty cycles. |
| **Blocked** | At some empty cycle C, moving ops from C+1 to C would violate a constraint. | Use `violation_message` to see which constraint blocked. |

### Blocked: Understanding the violation

The script prints:
- `Blocked at empty C=<cycle>: <violation_message>`
- `Steps before block: <n>`

**Violation message formats** (from `schedule_constraints.rs`):

- **Precedence** — `Precedence: op <consumer_idx> cycle <c> < dep <dep_idx> result_at ...`  
  Moving the consumer to C would run it before its dependency’s result is ready. The **precedence** constraint (data dependency) is holding the schedule.

- **Ordering** — `Ordering: op <before_idx> cycle <c> > op <after_idx> cycle ...`  
  Moving would break an ordering constraint (e.g. lex order, ordering group). The **ordering** constraint is holding the schedule.

- **Domain** — `Domain: op <idx> cycle <c> not in [lb, ub]`  
  Moving would put the op outside its [ASAP, ALAP] window. The **domain** (ALAP/ASAP) constraint is holding the schedule.

**Op indices** are into the SSA op list in the same order as `schedule_constraints`. Use the JSON `ops` / `ssa_ops_bulk_move` to map index → op id or instruction.

## Follow-Up: Which constraints are binding

When bulk-move **blocks**, the blocking constraint is one of the **binding** (tight) constraints at the CP-SAT solution. The CP-SAT path can log binding schedule constraints (precedence, ordering, domain) when building the schedule; search for "Binding schedule constraints" or "bulk-move" in `scheduler_rs/src/cpsat_scheduler.rs`. Comparing the violation’s constraint type and op indices to that list confirms which binding constraint is limiting compaction.

## Complementary Script

**analyze_empty_cycles.py** — Analyzes empty cycles from the JSON only (no Rust simulator). For each empty C, lists ops at C+1 and whether they’re "blocked by precedence" or "could run at C":
```bash
uv run scripts/analyze_empty_cycles.py scheduler_output.json
uv run scripts/analyze_empty_cycles.py --max-empty 20 --json-out report.json path.json
```
Use it to get a higher-level view of why ops sit at C+1; use bulk-move to get the **exact** constraint that blocks the move.

## Code References

- **Constraint definitions and bulk-move logic:** `scheduler_rs/src/schedule_constraints.rs`  
  - `ScheduleConstraint`: Precedence, Ordering, Domain  
  - `run_bulk_move_simulator`, `check_schedule_constraints`, `binding_schedule_constraints`
- **How constraints are built (CP-SAT):** `scheduler_rs/src/cpsat_scheduler.rs` (search for `build_ssa_schedule_constraints`, "bulk-move", "Binding schedule constraints")
- **Python wrapper:** `scripts/simulate_bulk_move.py`; Rust API: `scheduler_rs.run_bulk_move_simulator_from_json(path, verbose=...)`

## Limitation

Bulk-move checks **only** precedence, ordering, and domain. It does **not** check per-engine slot capacity. If compaction could violate capacity, that would indicate a bug in the model; use the CP-SAT "empty cycles + binding capacity" warning and active-set diagnostics for that.
