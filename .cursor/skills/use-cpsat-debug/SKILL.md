---
name: use-cpsat-debug
description: Debug CP-SAT INFEASIBLE results and inspect active set (binding constraints). Use when investigating infeasible configs (e.g. 3r/40 with const=load or shift=alu) or which constraints limit the schedule.
---

# CP-SAT Debug Tools

Use these tools when CP-SAT reports **INFEASIBLE** or when you want to inspect the **active set** (binding constraints) that limit the schedule.

## INFEASIBLE: Model Export and Inspector

When CP-SAT reports INFEASIBLE, the same constraint bugs may cause wrong output in other configs. To diagnose:

1. **Export the model** (before returning):
   ```bash
   CPSAT_EXPORT_INFEASIBLE=1 uv run generate_ssa_kernel.py --rounds 3 --batch-size 40 \
     --tree-reduce-max-depth 3 --no-cse-groups --const-choice load --valu-shift alu -q
   ```
   Writes `cpsat_infeasible_model.pb` (or path from `CPSAT_EXPORT_INFEASIBLE_PATH`).

2. **Inspect the exported model**:
   ```bash
   uv run python -m scripts.inspect_cpsat_model cpsat_infeasible_model.pb
   ```
   Prints constraint count/type, cumulative capacity/demands. Use `--validate` to run ortools validation.

## Active Set (Binding Constraints)

Binding constraints are those that hold with equality at the CP-SAT solution—they limit the schedule.

- **During generation:** Run `generate_ssa_kernel.py` without `-q`; stderr prints `[Makespan active set]` and binding schedule constraints (precedence, ordering, domain).
- **From saved JSON:** Use `scripts/inspect_binding_constraints.py` on scheduler output (CP-SAT run with `--output-json`):
  ```bash
  uv run python -m scripts.inspect_binding_constraints scheduler_output.json
  ```

## Demand Audit

The model build includes `debug_assert!` checks: for each cumulative (valu, alu, load, store, flow, scratch), `max(demands) <= capacity`. Run with debug assertions enabled (`cargo test` or debug build) to catch invalid cumulative constraints at build time.
