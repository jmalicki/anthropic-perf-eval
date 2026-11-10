---
name: use-kernel-dataflow
description: Prove an emitted kernel computes exactly what its SSA program computes, for all inputs, and locate the first cycle where it does not. Use first whenever a kernel fails validation, or after any change to scheduling, lowering, address assignment or liveness.
---

# Kernel dataflow replay

**Module:** `verification/kernel_dataflow.py`. Runs automatically under `--validate` (skip with `--no-dataflow-check`).

The kernel is executed symbolically: each scratch word holds a term such as `^(mem[+(p, idx)], c)` instead of a number. The same terms are built from the SSA ops alone (no cycles, addresses or liveness). If every store writes the same term to the same memory address on both sides, the kernel equals the SSA program for **every** input. It takes about half a second for the 16-round kernel.

## When to use

- **First step for any validation failure.** It tells you whether the bug is in scheduling/lowering/address assignment (replay fails) or in emission/SSA construction (replay proves, numeric validation fails).
- After changing the scheduler, the CP-SAT model, lowering, address assignment or liveness.
- Instead of `compare_kernel_scratch_cycle`, `find_wrong_scratch_use`, `trace_group20` and similar when the question is "which instruction read the wrong value".

## How to run

```bash
uv run generate_ssa_kernel.py --rounds 2 --use-greedy --output-json sched.json --save-kernel kern.json
uv run python -m verification.kernel_dataflow --kernel kern.json --scheduler sched.json

# What did every operand of every slot in cycle 436 hold, and who wrote it?
uv run python -m verification.kernel_dataflow --kernel kern.json --scheduler sched.json --explain-cycle 436
```

The scheduler JSON must come from the same build as the kernel and must contain `ssa_ops` and `vector_group_lanes` (any JSON written by the current `scheduler_rs`).

## Reading the output

`PROVED` means all stores match. Otherwise findings are listed earliest first; later ones are usually consequences of the first.

| Finding | Meaning | Usual cause |
|---|---|---|
| `write-conflict` | Two slots write one address in the same cycle | Address reused across adjacent intervals; op emitted in the wrong bundle |
| `foreign-value` | A slot computed a value no SSA op computes | Operand address wrong, or the address was overwritten before this read. The message names the wrong operand, what it should be, what the address held and who wrote it |
| `wrong-store` / `missing-store` / `extra-store` | Final memory writes differ | Consequence of an earlier finding, or a dropped op |

A `note:` line lists select conditions that were lowered to `multiply_add` but are not structurally 0/1. That lowering is only valid for 0/1 conditions.

## Limits

- It proves kernel = SSA program. It does not prove SSA program = reference kernel; `--validate` still checks that numerically.
- Algebraic identities it knows: constant folding, commutativity, `c*(t-f)+f = select(c,t,f)`, `(x-y)+y = x`. A new lowering that relies on another identity will show up as a `foreign-value` on a correct kernel; add the identity to `Terms` with a unit test in `tests/test_kernel_dataflow.py`.
- Straight-line kernels only (no jumps).

## Tests

`uv run --extra dev pytest tests/test_kernel_dataflow.py` builds real kernels and checks that clean ones are proved and that each corruption class (wrong bundle, wrong operand, premature overwrite, dropped op, wrong gather lane, wrong constant, write conflict) is caught.
