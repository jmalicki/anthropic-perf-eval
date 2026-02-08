# Commit plan (small, independent commits)

Execute in this order. Do **not** commit the kernel JSON files (`kernel_*.json`, `ssa_2round.json`) unless you want them tracked; they look like generated/test data.

1. **docs: add BUGS.md for bug tracking**
   - `BUGS.md`

2. **docs: add project skills and reference them in CLAUDE.md**
   - `.cursor/skills/use-bulk-move/SKILL.md`
   - `.cursor/skills/use-bugs-md/SKILL.md`
   - `.cursor/skills/use-bisection-tools/SKILL.md`
   - `CLAUDE.md` (Project skills section + 2-round note)

3. **feat(scheduler_rs): add schedule_constraints module**
   - `scheduler_rs/src/schedule_constraints.rs`
   - `scheduler_rs/src/lib.rs` (pub mod schedule_constraints)

4. **feat(scheduler_rs): CP-SAT capacity validation, bulk-move, full hints, makespan diagnostics**
   - `scheduler_rs/src/cpsat_scheduler/`
   - Includes: try_validate_ssa_slot_limits, capacity check before bulk-move, remove wrong redundant makespan bounds, makespan active set, full hints (chosen_cycle, select_fma_extra, fma_last), bulk-move integration and compaction, Unknown hint message, module docs.

5. **fix(scheduler_rs): concatenate section schedules; per-section ConstChoice hints; bulk-move result fields**
   - `scheduler_rs/src/kernel_builder/tick_tock_ssa/`
   - Rebase sections that skip per-section CP-SAT; segment const_add_imm hints; SsaScheduleResult fields and population for bulk-move JSON.

6. **feat(scheduler_rs): expose bulk-move from JSON, SSA schedule in JSON, empty-cycle logging**
   - `scheduler_rs/src/python.rs`
   - run_bulk_move_simulator_from_json, JSON output fields, empty_cycles(), emit one bundle per cycle, empty-cycle print.

7. **feat(scripts): add bulk-move simulator and empty-cycle analysis scripts**
   - `scripts/simulate_bulk_move.py`
   - `scripts/analyze_empty_cycles.py`

8. **fix(scheduler_rs): greedy ConstChoice as add_imm when Flow slot free**
   - `scheduler_rs/src/kernel_builder/ssa_greedy/`
   - Record const_add_imm_ops when scheduling ConstChoice as Flow; prefer add_imm when no other Flow op ready.
