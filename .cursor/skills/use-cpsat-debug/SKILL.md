---
name: use-cpsat-debug
description: Debug CP-SAT INFEASIBLE and inspect active set (binding constraints). We do this because we may have given CP-SAT incorrect constraints—e.g. if the proven optimum has 300+ empty cycles, our model is wrong. Use when investigating infeasible configs or which constraints are wrong (capacity, domain, makespan linkage).
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

## Budget Exceeded: Schedule JSON Export

When address assignment exceeds the scratch budget (after CP-SAT succeeds), the schedule JSON is **automatically exported** for debugging/reproduction:

- **Default path:** `scheduler_output_error.json`
- **Custom path:** Set `SCHEDULER_EXPORT_ON_ERROR_PATH=my_error.json`
- **Disable export:** Set `SCHEDULER_NO_EXPORT_ON_ERROR=1`

The exported JSON includes:
- CP-SAT schedule (cycle assignments, VALU shifts, ConstChoice/SelectChoice decisions)
- SSA ops (for debugging)
- Address assignment result (even though budget exceeded)
- Full liveness data
- Comment indicating the error

Use this JSON to reproduce the issue, analyze why budget was exceeded, or debug the address assignment algorithm.

## Active Set (Binding Constraints)

We inspect the active set (binding constraints) **because we suspect our constraints may be incorrect**. We trust CP-SAT: if the proven optimum has hundreds of empty cycles, the error is in the **model we gave it**—empty cycles can be removed, so a correct model would not have a proven minimum with 300+ empty cycles. We look at which constraints are binding to find what we did wrong.

Binding constraints are those that hold with equality at the CP-SAT solution—they limit the schedule. The same constraints also determine the **lower bound** (capacity_lower_bound, solver best_bound). When the optimum has many empty cycles, the constraints driving that bound are wrong (e.g. capacity formulas, domain bounds, or how we link makespan to choices).

- **During generation:** Run `generate_ssa_kernel.py` without `-q`; stderr prints `[Makespan active set]` and binding schedule constraints (precedence, ordering, domain).
- **From saved JSON:** Use `python -m scripts.inspect_binding_constraints` on scheduler output (CP-SAT run with `--output-json`):
  ```bash
  uv run python -m scripts.inspect_binding_constraints scheduler_output.json
  ```

## Demand Audit

The model build includes `debug_assert!` checks: for each cumulative (valu, alu, load, store, flow, scratch), `max(demands) <= capacity`. Run with debug assertions enabled (`cargo test` or debug build) to catch invalid cumulative constraints at build time.
