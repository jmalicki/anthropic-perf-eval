---
name: use-inspect-schedule
description: Look up what lives at a scratch address (lane-aware), which SSA op a kernel slot belongs to, and an SSA op's operands, lowered ops and consumers, from a scheduler output JSON. Use after kernel_dataflow or a trace names a cycle, address or slot and you need to know which value and which op are involved.
---

# Inspect a schedule

**Script:** `scripts/inspect_schedule.py`. Replaces `inspect_vreg_at_addr.py`, `inspect_vreg_definer.py`, `inspect_cycle_addr.py` and `who_wrote_addr.py`, which matched only a value's base address (asking about address 30 when a vector sat at 27..34 printed "No VReg at address 30").

## When to use

- `verification.kernel_dataflow` (or a trace script) reported a cycle, a slot or an address, and you want the SSA op, its source info (round, phase, group) and its neighbours.
- "What is at address A at cycle C, who wrote it last, and who reads or writes it in C?"
- "Where did SSA op N end up?" Shifted ops become 8 scalar ALU slots (possibly in different cycles); a select lowered to the vector ALU becomes a subtract and a multiply_add (the subtract may be shared by a CSE group).

## How to run

Run from the project root as a module (`scripts/bisect.py` shadows the stdlib `bisect` when a script is run from inside `scripts/`).

```bash
uv run generate_ssa_kernel.py --rounds 3 --batch-size 16 --use-greedy -q --output-json /tmp/s.json --save-kernel /tmp/k.json

# Every value whose address range covers 30; with --cycle: who holds it, last write, ops touching it
uv run python -m scripts.inspect_schedule /tmp/s.json addr 30 --cycle 21

# An SSA op: result, operands (address, interval, definer), lowered ops with cycles, consumers
uv run python -m scripts.inspect_schedule /tmp/s.json op 123

# Which SSA op does the slot writing address 117 in cycle 28 belong to? (--kernel cross-checks the bundle)
uv run python -m scripts.inspect_schedule /tmp/s.json --kernel /tmp/k.json slot 28 117

# A value by name or id: address, interval, definer, readers
uv run python -m scripts.inspect_schedule /tmp/s.json value v123_v
```

## Reading the output

- **Cycles are schedule cycles** (the `cycle` fields in the JSON). The kernel starts with one `pause` bundle, so **kernel bundle index = schedule cycle + 1**. `kernel_dataflow` reports bundle indexes: pass `--bundle` to `addr --cycle` and `slot` to give the cycle that way. Every lowered op is printed with both numbers.
- **Intervals are `[def, last_use]`**: the value is written at the end of cycle `def` and last read in cycle `last_use`. It is readable in cycles `def+1 .. last_use`; another value may be written to the same address in cycle `last_use`. `addr --cycle C` marks the value that a read in C sees (`holds address`) and one being written in C.
- **Vector groups** (gather rounds): 8 scalar lanes alias the 8 words of a vector. A *decomposed* vector is defined by one op and its lanes are read by scalar loads; an *assembled* vector has no defining op, each lane is written by its own load. Lanes have no entry in the JSON's `vregs`; the script places lane k at the group's address + k and lists both the group and the lane.
- **Lowered-op roles**: `lane k of 8` (shifted vector op), `select`, `select subtract`, `select as multiply_add`, `constant as const|add_imm`, `shared select subtract of CSE group g` (belongs to no single SSA op; `slot` lists the selects it feeds, `op` on a select lists it).
- `slot` on a store matches DEST against the store's pointer operand (a store writes memory, not scratch).

## Requirements

A scheduler JSON written by the current `scheduler_rs` (it must contain `ssa_ops`). Intervals need `--liveness-detail full` (the default); without the `vregs` array only addresses are shown. `--kernel` must be from the same build.
