---
name: use-validation-trace
description: Use execution-trace scripts to find where a kernel's output diverges from the reference (wrong batch indices, cycles, instructions). Use when debugging validation failures, tracing wrong stores to producer ops, or inspecting scratch/cycle-level behavior.
---

# Using the Validation / Execution Trace Scripts

These scripts compare a **saved kernel**’s execution to the reference or trace scratch/cycles. Use them when you have a kernel that **runs but produces wrong output** (validation fails). If CP-SAT is **infeasible**, no kernel is saved—nothing to trace; use bisection tools instead.

Run all scripts from **project root** with `uv run python scripts/...`. Use the same `--rounds` and `--batch-size` as the run that produced the kernel.

---

## 1. Where did the kernel go wrong? (mismatches and groups)

**Script:** `scripts/debug_validation_trace.py`

Runs the saved kernel and the reference on the same input, then reports **per-batch got vs ref** and **all mismatching batch indices** with affected groups (lane-by-lane).

### Usage

```bash
# Default kernel: kernel_to_validate.json, rounds=4, batch_size=256, batch_idx=160
uv run python scripts/debug_validation_trace.py

# Match the config that produced the kernel
uv run python scripts/debug_validation_trace.py --kernel path/to/kernel.json --rounds 5 --batch-size 16

# Focus on a specific batch index in the printed reference trace
uv run python scripts/debug_validation_trace.py --kernel kernel_fail.json --rounds 4 --batch-size 32 --batch-idx 0
```

### Output

- Reference kernel trace for `--batch-idx` (per-round idx, node_val, xor, hashed, next_idx, wrapped).
- Final **Got vs Ref** for that batch index (value and idx; Match: YES/NO).
- **All mismatching batch indices** (up to 20 listed; total count).
- **Affected groups** and per-group lane table (got_idx, ref_idx, got_val, ref_val, MISMATCH/OK).
- Scratch memory dump (first 64 words) after the optimized kernel run.

Use this to see **which batch indices and groups** are wrong before diving into a single group.

---

## 2. Which cycle and instruction wrote the wrong value? (group trace)

**Script:** `scripts/trace_group20.py`

Traces **one group** (8 consecutive batch indices) through the kernel: reference trace, final comparison, **which cycles** wrote to that group’s output addresses, and for each **wrong** store the **instruction** at that cycle and optional producer-tree tracing.

### Usage

```bash
# Default: kernel_treeselect_4round.json or kernel_to_validate.json, rounds=4, batch_size=256, group=20
uv run python scripts/trace_group20.py

# Custom kernel and config; trace group 0 (batch indices 0–7)
uv run python scripts/trace_group20.py --kernel kernel_fail.json --rounds 4 --batch-size 32 --group 0

# Large batch: trace group 20 (batch 160–167)
uv run python scripts/trace_group20.py --kernel kernel_to_validate.json --rounds 4 --batch-size 256 --group 20
```

**Note:** `--group N` implies batch indices `N*8` to `N*8+7`. Ensure `(N+1)*8 <= batch_size`.

### Output

- **Reference trace** for that group (per round, per batch index: idx, val, node_val, xor, hash, next, wrapped).
- **Final memory comparison** for the group (got vs expected, WRONG/OK).
- **Memory writes to group addresses** — each store with **cycle number**, value, and ref (WRONG/OK).
- **Tracing wrong values backwards** — for each wrong store: cycle, batch index, kind (value/index), **instruction** at that cycle (e.g. `vstore` from scratch 53), store slot.
- Deeper **producer tree** tracing for wrong index stores (optional; script may trace back through scratch producers).

Use this to get the **exact cycle and instruction** that wrote a wrong value, then trace producers if needed.

---

## 3. Trace specific scratch addresses cycle-by-cycle

**Script:** `scripts/trace_scratch_addresses.py`

Runs the kernel step-by-step and logs **writes to monitored scratch addresses** and optional **snapshot** dumps at chosen cycles.

### Usage

```bash
# Default: kernel_treeselect_4round.json, rounds=4, batch_size=256; watches default scratch ranges
uv run python scripts/trace_scratch_addresses.py

# Custom kernel and config; watch specific addresses and snapshot cycles
uv run python scripts/trace_scratch_addresses.py --kernel kernel_to_validate.json --rounds 4 --batch-size 32 \
  --watch-addrs 0 8 16 24 --snapshot-cycles 10 20 30
```

Use when you need to see **when** a given scratch address (or vector base) is written during execution.

---

## 4. Dump instructions at specific cycles

**Script:** `scripts/dump_cycle_instrs.py`

Prints the **instruction bundle** (engine and slots) for given cycles. No execution; kernel JSON only.

### Usage

```bash
# Default: kernel_treeselect_4round.json, cycles 369..375
uv run python scripts/dump_cycle_instrs.py

# Custom kernel and cycles (e.g. cycles where trace_group20 reported wrong stores)
uv run python scripts/dump_cycle_instrs.py --kernel kernel_fail.json --cycles 560 561 562 563 564 565
```

Use to inspect **exactly what ops** run at a cycle you identified from the trace scripts.

---

## 5. Inspect schedule JSON: ops at cycle C reading address A, VRegs at address

When debugging **scratch reuse** or “vstore at cycle C reads address A”—use the **scheduler output JSON** (from `--output-json`), not the kernel.

**Script:** `scripts/inspect_cycle_addr.py` — Ops at a given cycle that read a given scratch address (or vector range), plus VRegs at that address. With `--vregs`, shows intervals and whether each VReg is live at that cycle.

```bash
uv run python scripts/inspect_cycle_addr.py sched_4r8b_d3.json --cycle 785 --addr 19 --vregs
```

**Script:** `scripts/inspect_vreg_at_addr.py` — Which VRegs were assigned a given scratch address; with `--vregs`, shows interval [start, end] from the JSON `vregs` array (requires `--liveness-detail full`).

```bash
uv run python scripts/inspect_vreg_at_addr.py scheduler_output.json --addr 10 --vregs
```

**Script:** `scripts/inspect_vreg_definer.py` — Given a VReg id, prints the **defining op** (the op that writes to that VReg’s scratch address) and, if present, interval/metadata from the `vregs` array. Use to confirm the value at an address comes from the expected SSA op (e.g. depth-3 tree-select).

```bash
uv run python scripts/inspect_vreg_definer.py sched_4r8b_d3.json --vreg 213
uv run python scripts/inspect_vreg_definer.py scheduler_output.json --vreg 213 --width vector
```

Use these when you know the **cycle and address** from the trace (e.g. “vstore at 786 from scratch 19”) and want to see which SSA op reads that address and which VRegs share it with their intervals.

---

## 6. Depth-3 wrong value: tree selection vs hash vs branch

**Script:** `scripts/compare_depth3_tree_value.py`

For **depth-3** configs (e.g. 4r/8b) where the kernel produces a wrong value, compares the wrong value to reference **node_val**, **val_after_xor**, and **hashed_val** for the depth-3 round. Tells you whether the bug is likely (A) wrong tree selection (got value matches another lane's node_val), (B) wrong XOR, or (C) wrong hash/branch. **Tip:** Generate the kernel with `--no-cse-groups` to simplify debugging (fewer moving parts); see BUGS.md for the minimal repro.

### Usage

```bash
uv run python scripts/compare_depth3_tree_value.py --kernel kernel_to_validate.json --rounds 4 --batch-size 8
uv run python scripts/compare_depth3_tree_value.py --rounds 4 --batch-size 8 --group 0
```

### Output

- **Reference** for the depth-3 round (round 3 for 4 rounds): node_val, val_after_xor, hashed_val per batch index (optionally for one group only with `--group`).
- **Final got vs ref** and list of mismatching batch indices.
- **Wrong value analysis**: for each wrong value store, whether got matches ref node_val (same or other lane), val_after_xor, or hashed_val — to distinguish tree selection mix-up vs hash/branch bug.

Use when debugging the "depth-3 (4r/8b): all 8 lanes wrong" failure to isolate tree-reduce selection vs hash vs branch.

---

## Workflow summary

| Goal | Script | Typical use |
|------|--------|-------------|
| Which batch indices / groups are wrong? | debug_validation_trace.py | After validation fails; use same rounds/batch-size as build. |
| Which cycle and instruction wrote a wrong value? | trace_group20.py | After debug_validation_trace; pick an affected group with `--group`. |
| Depth-3: tree selection vs hash vs branch? | compare_depth3_tree_value.py | 4r/8b wrong value; compare got to ref node_val/val_after_xor/hashed_val. |
| How does a scratch address change over time? | trace_scratch_addresses.py | When debugging scratch/register usage. |
| What instructions run at cycle C? | dump_cycle_instrs.py | After trace_group20 gives you a cycle number. |
| Ops at cycle C reading addr A / VRegs at addr? | inspect_cycle_addr.py, inspect_vreg_at_addr.py | Schedule JSON; debugging scratch reuse / vstore value interval. |
| Which SSA op defines VReg X? | inspect_vreg_definer.py | Schedule JSON; confirm producer of value at an address (e.g. VReg 213). |

**No kernel to trace?** If the failing config makes CP-SAT **infeasible**, no kernel file is produced. Use `scripts/bisect_cpsat_validation.py` or `scripts/bisect_minimal_fma.py` (see use-bisection-tools skill) to find which options or minimal config fail; trace scripts need a **saved kernel that runs but produces wrong output**.
