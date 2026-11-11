---
name: use-schedule-report
description: Answer "how good is this schedule" from a scheduler output JSON - cycles, per-engine floors, the capacity floor for the ops this run emitted, the gap to it, per-round cycle spans and utilisation over time; diff two runs. Use after any build, when comparing schedulers or options, or whenever a lower bound is quoted (the floor is for one emission, never a bound on the problem).
---

# Schedule report

**Module:** `scripts/schedule_report.py`. The generator prints its summary after every build (three lines with `-q`, the whole report otherwise).

## How to run

```bash
uv run generate_ssa_kernel.py --rounds 16 --use-greedy -q --output-json sched.json

uv run python -m scripts.schedule_report sched.json                       # text
uv run python -m scripts.schedule_report sched.json --json                # same data, machine-readable
uv run python -m scripts.schedule_report sched.json --compare other.json  # cycles, floors, per-round spans
uv run python -m scripts.schedule_report sched.json --buckets 20          # finer utilisation over time
```

Run it as a module from the project root. Started from inside `scripts/`, `scripts/bisect.py` shadows the standard library's `bisect`.

## What it prints

| Block | Meaning |
|---|---|
| Header | Cycles achieved, scheduler (`run_options.use_cpsat`), rounds, batch |
| Engine table | Lowered ops, slots available (capacity x cycles), utilisation, and floor = ceil(ops / capacity) for that engine alone |
| Emitted | Vector ops (on `valu` plus those run as 8 scalar ops), native scalar ops, vector-op equivalents (vector + scalar/8) |
| Capacity floor for this emission | Fewest cycles the emitted ops fit in when vector ops may run as 8 scalar ops: minimise T with `vector - s <= 6T` and `scalar + 8s <= 12T` over the number shifted `s` (non-shiftable `multiply_add`/`vbroadcast` stay on `valu`), then the max with the load, flow and store floors. Names the binding engine |
| Gap | Achieved minus floor, in cycles and as a percent of the floor |
| CP-SAT bound | `lb_cycles` when present, labelled as a bound for this emission |
| Round table | First/last cycle, span, vector/scalar/load/flow ops per round (`ssa_ops[*].source_info.round`), and cycles shared with the previous round. Overlap 0 on every row means rounds run strictly one after another |
| Utilisation over time | Percent of slots used per engine in N equal slices of the schedule |

## Reading it

- **The floor is conditional.** It is slot arithmetic over the ops this compiler emitted, with the const/select encodings chosen in this run. A kernel that emits fewer ops has a lower floor. It is not a bound on the problem: for the 16-round kernel the floor is 1,271 while public kernels reach 863 ([docs/BOUND_GAP_ANALYSIS.md](../../../docs/BOUND_GAP_ANALYSIS.md)). Always quote it as "floor for this emission (N vector-op equivalents)". The same holds for `lb_cycles` and CP-SAT `best_bound`.
- **Gap large, floor fine:** the schedule is the problem (rounds serialised, an engine idle while another saturates). Look at the round table and utilisation over time.
- **Gap small, cycles too high:** the emission is the problem. Fewer ops are needed; see section 4 of the bound gap analysis.
- **Binding engine:** `alu+valu` means the two ALUs jointly (7.5 vector-op equivalents per cycle); `load` means 2 loads per cycle is the limit, and shifting more work to the scalar ALU will not help.

## Related scripts

All three compute the same joint ALU floor (1,271 for the 16-round greedy JSON).

- `scripts/analyze_bound_gap.py` reproduces `docs/BOUND_GAP_ANALYSIS.md`. Its bound also minimises over the const and select encodings, which changes the load and flow terms (1,088 and 448 instead of 1,068 and 488) but not the maximum.
- `scripts/theoretical_lower_bounds.py` counts SSA ops instead of lowered ops and breaks VALU down by opcode.
- `scripts/slot_usage_by_decile.py` ([use-schedule-debug](../use-schedule-debug/SKILL.md)) gives a more detailed utilisation view.

## Tests

`uv run pytest tests/test_schedule_report.py` checks the floor formula on hand-made inputs and, on real greedy and CP-SAT kernels, that engine counts sum to the lowered ops, the gap is non-negative and the round spans cover every scheduled op.
