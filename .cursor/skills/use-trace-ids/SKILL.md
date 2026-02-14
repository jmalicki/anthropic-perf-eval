---
name: use-trace-ids
description: Use trace_input/trace_output debug ops with --enable-trace-ids to capture semantic trace data. The main use is to build both a passing and a failing kernel (e.g. different solver options) with trace support, run both on the same input, and compare their trace_io_logs to find the first divergence. Also use to compare one kernel's trace to the reference.
---

# Using Trace IDs for Semantic Debugging

Trace IDs enable **semantic debugging** by instrumenting the kernel with `trace_input` and `trace_output` debug ops that capture values at specific points in execution. Unlike execution traces that show all scratch writes, trace IDs let you mark **semantic points** (e.g., "round 0 group 0 hashed_val output") and see exactly what values flow through those points.

**Primary use:** Build **both** a passing and a failing kernel (e.g. default vs zero-gap, or different solver options) **with trace support** (`--enable-trace-ids`), run **both** on the **same input** with `enable_debug=True`, and **compare their two `trace_io_log`s** to find the **first** (cycle, trace_id, lane) where the values diverge. That tells you exactly which semantic point and round/group first differs between the two compiler/solver outcomes.

Use trace IDs when:
- You have **two kernels** (e.g. one that validates, one that fails) and want to see **where their behavior first diverges** (good vs bad options, same input, compare traces)
- You need to see **exact values** at semantic boundaries (e.g., hash outputs per round/group)
- You want to **compare optimized vs reference** at specific computation points
- You've localized a failure to a specific round/depth but need to see **where values diverge**

---

## 1. Good vs bad: build both with trace IDs and find first divergence

**Goal:** Find the first trace point where a passing kernel and a failing kernel (different options, same problem size) produce different values on the same input.

1. **Build passing kernel with trace IDs:** e.g. default options, no zero-gap.
   ```bash
   uv run generate_ssa_kernel.py --rounds 2 --batch-size 16 --enable-trace-ids \
     --validate --save-kernel kernel_pass_trace.json
   ```

2. **Build failing kernel with trace IDs:** same rounds/batch-size, different options (e.g. `--force-zero-gap`).
   ```bash
   uv run generate_ssa_kernel.py --rounds 2 --batch-size 16 --force-zero-gap \
     --enable-trace-ids --cpsat-time-limit 600 --save-kernel kernel_fail_trace.json
   ```

3. **Run both on the same input** with `enable_debug=True`, capture `trace_io_log` for each.

4. **Compare the two logs:** Walk both logs in order (by cycle, then trace_id). The **first** entry where `(trace_id, values)` differs between the two runs is the first divergence. That (cycle, trace_id, lane) pins the bug to a specific semantic point (e.g. round 0 group 0 hashed_val).

Use the same seed and mem image for both runs so the only difference is the kernel. Optionally compare to the reference as well (see section 4) to see whether the passing trace matches the reference and the failing trace diverges at that same point.

---

## 2. Generate a kernel with trace IDs enabled

**Flag:** `--enable-trace-ids`

```bash
# Generate kernel with trace IDs (adds trace_input/trace_output debug ops)
uv run generate_ssa_kernel.py --rounds 4 --batch-size 8 --tree-reduce-max-depth 3 \
  --no-cse-groups --cpsat-time-limit 60 --enable-trace-ids \
  --validate --save-kernel kernel_trace.json --output-json sched_trace.json
```

**What it does:** Adds `trace_input` and `trace_output` debug ops at semantic points (currently: hash outputs per round/group). These ops read/write scratch values and log them to `Machine.trace_io_log`.

**Current trace points:** Hash outputs (`r{round}_g{group}_hashed_val`) are traced. The emitter sets `next_trace_id` before emitting hash ops; lowering converts these to `trace_input`/`trace_output` debug ops.

---

## 3. Run kernel and capture trace data

Run the kernel with `enable_debug=True` to populate `trace_io_log`:

```python
import json
import random
from copy import copy
from problem import Machine, Tree, Input, build_mem_image, DebugInfo, N_CORES

# Load kernel
with open('kernel_trace.json') as f:
    kernel_data = json.load(f)
instrs = kernel_data.get('instrs', kernel_data)

# Build memory image
random.seed(123)  # Match validation seed
forest = Tree.generate(10)
inp = Input.generate(forest, 8, 4)
mem = build_mem_image(forest, inp)

# Run kernel with debug enabled
mem_copy = copy(mem)
machine = Machine(
    mem_copy,
    list(instrs),
    DebugInfo(scratch_map={}),
    n_cores=N_CORES,
    trace=False,
)
machine.enable_pause = False
machine.enable_debug = True  # Required for trace_io_log
machine.prints = False
machine.run()

# Access trace data
if hasattr(machine, 'trace_io_log') and machine.trace_io_log:
    for cycle, io_type, trace_id, values in machine.trace_io_log:
        print(f"Cycle {cycle}: {io_type} {trace_id} = {values}")
```

**Trace entry format:** `(cycle, io_type, trace_id, values)` where:
- `cycle`: Cycle number when trace op executed
- `io_type`: `"input"` or `"output"` 
- `trace_id`: Semantic identifier (e.g., `"r0_g0_hashed_val"`)
- `values`: Tuple of values captured (vector ops capture 8 elements)

---

## 4. Compare trace outputs to reference

Compare trace outputs to reference kernel to find where values diverge:

```python
from problem import reference_kernel2

# Run reference kernel
mem_ref = copy(mem)
trace_ref = {}
list(reference_kernel2(mem_ref, trace_ref))

# Compare trace outputs
for cycle, io_type, trace_id, values in machine.trace_io_log:
    if io_type == 'output':
        # Parse trace_id: r0_g0_hashed_val -> round 0, group 0
        parts = trace_id.split('_')
        if len(parts) >= 3 and parts[0].startswith('r') and parts[1].startswith('g'):
            round_num = int(parts[0][1:])
            group_num = int(parts[1][1:])
            
            print(f"Cycle {cycle}: {trace_id}")
            for lane in range(8):
                batch_i = group_num * 8 + lane
                got_val = values[lane] if lane < len(values) else None
                ref_key = (round_num, batch_i, 'hashed_val')
                ref_val = trace_ref.get(ref_key)
                got_str = hex(got_val) if got_val is not None else "<nil>"
                ref_str = hex(ref_val) if ref_val is not None else "<nil>"
                match = "OK" if got_val == ref_val else "WRONG"
                print(f"  Lane {lane} (batch {batch_i}): got={got_str:12} ref={ref_str:12} [{match}]")
```

**Key insight:** If round 0 outputs are correct but round 2 outputs are wrong, the bug is **not** in init phase or depth-1 computation—it's in the later rounds (e.g., depth-3 tree-reduce or branch computation).

---

## 5. Analyze trace inputs to find divergence point

Trace inputs show what values **enter** a computation. Compare to previous round's outputs:

```python
for cycle, io_type, trace_id, values in machine.trace_io_log:
    if io_type == 'input':
        parts = trace_id.split('_')
        if len(parts) >= 3 and parts[0].startswith('r'):
            round_num = int(parts[0][1:])
            group_num = int(parts[1][1:])
            
            print(f"Cycle {cycle}: {trace_id} INPUT")
            for lane in range(8):
                batch_i = group_num * 8 + lane
                input_val = values[lane] if lane < len(values) else None
                # Compare to previous round's output
                prev_round = round_num - 1
                ref_key = (prev_round, batch_i, 'hashed_val')
                ref_val = trace_ref.get(ref_key)
                input_str = hex(input_val) if input_val is not None else "<nil>"
                ref_str = hex(ref_val) if ref_val is not None else "<nil>"
                match = "OK" if input_val == ref_val else "WRONG"
                print(f"  Lane {lane}: input={input_str:12} prev_round={ref_str:12} [{match}]")
```

**Note:** Trace inputs may include multiple addresses (e.g., hash inputs read from 2 addresses = 16 values total). The first 8 are typically the main values; check the trace op definition to understand the layout.

---

## 6. Pinpoint root cause from trace patterns

**Pattern analysis:**

1. **All rounds correct:** Bug is in final store or output formatting
2. **Round 0 correct, later rounds wrong:** Bug is in branch computation or tree selection (not init phase)
3. **Specific lanes wrong consistently:** Lane-specific bug (e.g., wrong tree index for those lanes)
4. **Wrong inputs but correct outputs:** Bug is in how inputs are read (scratch reuse, wrong addresses)
5. **Correct inputs but wrong outputs:** Bug is in the computation itself (hash, tree selection, etc.)

**Example:** If round 0 (depth-1) outputs are **all correct** but round 2 (depth-3) outputs are wrong, the bug is:
- **NOT** in init phase `td_1` computation (depth-1 works fine)
- **NOT** in depth-1 tree selection (round 0 correct)
- **LIKELY** in depth-3 tree-reduce selection or branch computation for depth-3 rounds

---

## 7. Adding new trace points

To add trace IDs for other semantic points (e.g., tree selection outputs, branch offsets):

1. **Set trace ID before emitting op:**
```rust
// In ssa_emit.rs or tick_tock_ssa/mod.rs
self.emit.set_next_trace_id(format!("r{}_g{}_node_val", round, group));
let v_nv = self.select_2way_with_delta(cond, v_td_1, v_tc2);
// v_nv will have trace_id set
```

2. **Lowering automatically creates debug ops:** `ssa_lower.rs` converts ops with `trace_id` to `trace_input`/`trace_output` debug ops at lowering time.

3. **Debug ops are emitted:** `emit_trace_debug_ops()` creates:
   - `trace_input` at first cycle where trace_id appears (reads inputs)
   - `trace_output` at last cycle where trace_id appears (reads outputs after writes)

---

## Workflow summary

| Step | Command/Script | Purpose |
|------|----------------|---------|
| **Good vs bad (primary)** | Build both kernels with `--enable-trace-ids`, run both, diff `trace_io_log` | Find first (cycle, trace_id) where passing and failing diverge |
| Generate with trace IDs | `--enable-trace-ids` | Add semantic trace points |
| Run kernel | Python script with `enable_debug=True` | Capture `trace_io_log` |
| Compare outputs | Compare trace outputs to reference (or good vs bad) | Find which rounds/lanes are wrong |
| Compare inputs | Compare trace inputs to prev round | Find where values diverge |
| Analyze pattern | Pattern analysis (see section 6) | Pinpoint root cause |

---

## Limitations

- **Current coverage:** Only hash outputs are traced (`r{round}_g{group}_hashed_val`). Tree selection outputs, branch offsets, etc. are not traced by default.
- **Performance:** Trace ops add cycles and scratch reads/writes. Use for debugging, not production.
- **Debug ops:** Require `enable_debug=True` in Machine. Trace ops are skipped if debug is disabled.

---

## Quick reference

```bash
# Generate kernel with trace IDs
uv run generate_ssa_kernel.py --rounds 4 --batch-size 8 --enable-trace-ids \
  --validate --save-kernel kernel_trace.json

# Run and analyze (Python)
python3 << 'EOF'
import json, random
from copy import copy
from problem import Machine, Tree, Input, build_mem_image, DebugInfo, N_CORES, reference_kernel2

# Load kernel, build mem, run optimized
# ... (see section 3)

# Run reference, compare
# ... (see section 4)
EOF
```

Use trace IDs to **understand the flow** of values through semantic boundaries, complementing execution traces that show scratch-level behavior.
