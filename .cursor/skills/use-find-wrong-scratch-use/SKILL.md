---
name: use-find-wrong-scratch-use
description: Find the first lowered op that reads from a scratch address not matching vreg_to_addr[SSA source]. Use when root-causing validation failure where scratch diverges (e.g. consumer of constant 5 reads from slot that got 4)—classify as wrong use (lowering/address) vs wrong assignment.
---

# Find Wrong Scratch Use

**Script:** `scripts/find_wrong_scratch_use.py`

Loads **scheduler output JSON** (from `--output-json`), walks the lowered `ops`, and for each op that reads from a target scratch address checks whether that read address equals the address assigned to the SSA source in `vreg_to_addr`. Reports the **first** op where the read address is wrong. Use when you already know **which addresses** diverge (e.g. from `compare_kernel_scratch_cycle.py`) and want to find the exact instruction and SSA op that perform the incorrect use.

## When to Use

- After `compare_kernel_scratch_cycle.py` (or similar) gives you a **(cycle, address)** where scratch differs.
- Root-cause: "consumer of constant X reads from slot that got Y" — this script finds that consumer op.
- Classifying bug as **(A)** address assignment (wrong/duplicate mapping) vs **(B)** lowering (emitted address not from vreg_to_addr[ssa_src]).

## Prerequisites

- Scheduler output JSON with `ssa_ops`, `vreg_to_addr`, and `ops` (from `generate_ssa_kernel.py --output-json`).
- Optional: know the divergent addresses (default script checks 1 and 2).

## How to Run

```bash
# Failing run's scheduler JSON; default checks addrs 1 and 2
uv run python -m scripts.find_wrong_scratch_use artifacts/scheduler_2r_16b_zero_gap.json

# Specific addresses (e.g. from compare_kernel_scratch_cycle output)
uv run python -m scripts.find_wrong_scratch_use artifacts/scheduler_2r_16b_zero_gap.json --addrs 1 2

# Verbose: print every op that reads target addrs and expected vs actual
uv run python -m scripts.find_wrong_scratch_use path/to/sched.json --addrs 1 2 -v
```

## Output

- **INCORRECT USE:** cycle, engine, opcode, op_id, slot, slot[index] (actual) vs expected (vreg_to_addr[SSA src]), and SSA op id/opcode/srcs.
- Or: `No incorrect use found among ops reading the given addrs.`

Supports binary ALU, load, vbroadcast, vstore (value slot), and const_choice ALU-derive slot layout. Use the reported op_id/cycle with `dump_cycle_instrs.py` or scheduler inspection to confirm the lowering path.
