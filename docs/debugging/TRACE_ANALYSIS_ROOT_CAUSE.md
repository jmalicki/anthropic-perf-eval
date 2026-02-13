# Root Cause Analysis: Validation Failure (4r/8b depth-3)

**Date:** 2026-02-10  
**Method:** Trace IDs + Scheduler JSON correlation  
**Kernel:** `kernel_4r_8b_d3_trace.json` (generated with `--enable-trace-ids`)

## Debugging Methodology

This document demonstrates a complete debugging workflow using trace IDs, scheduler JSON, and liveness analysis to pinpoint root causes.

### Step 1: Generate kernel with trace IDs

```bash
uv run generate_ssa_kernel.py --rounds 4 --batch-size 8 --tree-reduce-max-depth 3 \
  --no-cse-groups --cpsat-time-limit 60 --enable-trace-ids \
  --validate --save-kernel kernel_4r_8b_d3_trace.json \
  --output-json sched_4r_8b_d3_trace.json
```

**Result:** Kernel fails validation; trace IDs added to hash output ops.

### Step 2: Run kernel and capture trace data

```python
import json
import random
from copy import copy
from problem import Machine, Tree, Input, build_mem_image, DebugInfo, N_CORES, reference_kernel2

# Load kernel
with open('kernel_4r_8b_d3_trace.json') as f:
    kernel_data = json.load(f)
instrs = kernel_data.get('instrs', kernel_data)

# Build memory and run optimized kernel
random.seed(123)
forest = Tree.generate(10)
inp = Input.generate(forest, 8, 4)
mem = build_mem_image(forest, inp)

mem_opt = copy(mem)
machine = Machine(
    mem_opt,
    list(instrs),
    DebugInfo(scratch_map={}),
    n_cores=N_CORES,
    trace=False,
)
machine.enable_pause = False
machine.enable_debug = True  # Required for trace_io_log
machine.prints = False
machine.run()

# Run reference kernel
mem_ref = copy(mem)
trace_ref = {}
list(reference_kernel2(mem_ref, trace_ref))

# Access trace data
trace_io_log = machine.trace_io_log  # List of (cycle, io_type, trace_id, values)
```

**Result:** `trace_io_log` contains semantic trace points (hash inputs/outputs per round/group).

### Step 3: Compare trace outputs to reference

Compare trace outputs round-by-round to find where values diverge:

```python
for entry in machine.trace_io_log:
    cycle, io_type, trace_id, values = entry
    if io_type == 'output':
        # Parse trace_id: r0_g0_hashed_val -> round 0, group 0
        parts = trace_id.split('_')
        round_num = int(parts[0][1:])
        group_num = int(parts[1][1:])
        
        for lane in range(8):
            batch_i = group_num * 8 + lane
            got_val = values[lane]
            ref_key = (round_num, batch_i, 'hashed_val')
            ref_val = trace_ref.get(ref_key)
            if got_val != ref_val:
                print(f"Round {round_num} batch {batch_i}: WRONG")
```

**Finding:** Round 0 and 1 are correct; Round 2 is wrong for batches [0,2,4,5,6,7].

### Step 4: Correlate trace outputs with scheduler JSON

Load scheduler JSON and find VRegs corresponding to trace outputs:

```python
with open('sched_4r_8b_d3_trace.json') as f:
    sched_data = json.load(f)

ops = sched_data.get('ops', [])
vreg_to_addr = sched_data.get('vreg_to_addr', {})
vregs = sched_data.get('vregs', [])

# Find ops at trace output cycle
r2_output_cycle = 50  # From trace_io_log
cycle_ops = [o for o in ops if o.get('cycle') == r2_output_cycle]

# Find hash output VRegs
for op in cycle_ops:
    if op.get('opcode') == '^':  # XOR produces hash
        writes = op.get('writes', [])  # Addresses written to
        # Find VRegs at those addresses
        for addr in writes:
            # Look up VReg in vreg_to_addr and vregs array
```

**Finding:** Hash outputs written to addresses 13-20, but need to check what VRegs are actually there.

### Step 5: Trace dependency chain backwards

Find what produces the wrong values by tracing dependencies:

```python
# Find multiply_add that computes round 2 idx
r2_idx_mult = next((o for o in ops if o.get('opcode') == 'multiply_add' and 43 in o.get('writes', [])), None)

# Check offset computation
reads = r2_idx_mult.get('reads', [])
offset_addrs = reads[16:24]  # Offset addresses for 8 lanes

# Find what produces each offset
for lane, offset_addr in enumerate(offset_addrs):
    offset_prods = [o for o in ops if offset_addr in o.get('writes', []) and o.get('cycle', 999) < r2_idx_mult.get('cycle')]
    offset_prod = max(offset_prods, key=lambda x: x.get('cycle', 0))
    # Check if offset_prod reads from round 1 hash
```

**Finding:** Offset computation reads from wrong addresses.

### Step 6: Check liveness intervals

Verify which VRegs are live at the problematic cycle:

```python
op159_cycle = 50
hash_base = 13
hash_addrs = list(range(hash_base, hash_base + 8))

for addr in hash_addrs:
    # Find VRegs live at this address at cycle 50
    live_vregs = []
    for vreg_key, vreg_addr in vreg_to_addr.items():
        if vreg_addr == addr:
            for v in vregs:
                if f"id: {v.get('id')}" in vreg_key and "Vector" in vreg_key:
                    start = v.get('interval_start', 999)
                    end = v.get('interval_end', 0)
                    if start <= op159_cycle < end:
                        live_vregs.append((v.get('id'), v.get('phase'), v.get('round'), start, end))
    
    if not live_vregs:
        print(f"Address {addr}: NO VREG LIVE - expired or reused!")
```

**Finding:** Round 1 hash VRegs expired before cycle 50; addresses reused for round 2 hash.

### Step 7: Identify root cause

Correlate findings:
- Round 1 hash produced at cycle 32
- Op 159 (offset computation) at cycle 50 needs round 1 hash
- Round 1 hash VRegs expired before cycle 50
- Addresses reused → Op 159 reads wrong values

**Root cause:** Liveness extension bug - hash VRegs not extended to cover next round's branch computation.

## Summary

**Root cause:** **Liveness extension bug** - Round 1 hash VRegs expire before they're needed for round 2 branch offset computation, causing Op 159 (AND computing `hash & 1`) to read from wrong addresses.

## Trace Analysis Findings

### 1. Round-by-Round Correctness
- **Round 0 (depth-1):** ALL CORRECT ✓
- **Round 1 (depth-2):** ALL CORRECT ✓  
- **Round 2 (depth-2):** WRONG for batches 0,2,4,5,6,7
- **Round 3 (depth-3):** WRONG for batches 0,2,4,5,6,7

**Key insight:** Since rounds 0 and 1 are correct, the bug is **NOT** in:
- Init phase (`td_1` computation)
- Depth-1 tree selection
- Depth-2 tree selection for round 1

### 2. Pattern Analysis

**Wrong batches:** [0, 2, 4, 5, 6, 7]  
**Correct batches:** [1, 3]

**Round 1 next_idx (feeds into round 2):**
- Wrong batches: next_idx = 6 or 5 (even/5 indices)
- Correct batches: next_idx = 3 (odd index)

### 3. Root Cause: Op 159 Address Mismatch

**Op 159** (cycle 50): AND operation computing `hash & 1` for offset computation

**Expected:** Read round 1 hash from addresses `[13, 14, 15, 16, 17, 18, 19, 20]`  
**Actual:** Reads from addresses `[23, 17, 22, 28, 19, 16, 13, 20]`

**Lane-by-lane mismatch:**
- Lane 0: reads from addr 23 instead of 13
- Lane 1: reads from addr 17 instead of 14  
- Lane 2: reads from addr 22 instead of 15
- Lane 3: reads from addr 28 instead of 16
- Lane 4: reads from addr 19 instead of 17
- Lane 5: reads from addr 16 instead of 18
- Lane 6: reads from addr 13 instead of 19
- Lane 7: reads from addr 20 (CORRECT)

### 4. Liveness Analysis

**Round 1 hash output:** Cycle 32  
**Op 159 (offset computation):** Cycle 50  
**Gap:** 18 cycles

**At cycle 50:**
- Address 13: VReg 159 (round 2 hash) is live - **WRONG VReg!**
- Addresses 14-20: **NO VRegs live** - round 1 hash VRegs have expired

**Conclusion:** Round 1 hash VRegs expire before cycle 50, so addresses 13-20 are reused for round 2 hash VRegs. When Op 159 tries to read round 1 hash, it reads from whatever VRegs are currently at those addresses (wrong values).

## Root Cause

**Liveness extension bug:** The round 1 hash VRegs are not extended to cover the branch offset computation that happens in the next round. The liveness computation doesn't track that hash values are needed for `offset = 2 - (hash & 1)` in the branch step.

**Chain of failure:**
1. Round 1 hash produced at cycle 32
2. Round 1 hash VRegs expire (intervals end before cycle 50)
3. Addresses 13-20 reused for round 2 hash VRegs
4. Op 159 (cycle 50) tries to read round 1 hash from addresses 13-20
5. Reads wrong values (round 2 hash or other VRegs)
6. Wrong offset computed → wrong idx → wrong tree selection → wrong output

## Fix Location

The bug is in **liveness computation** - specifically, hash VRegs need to be extended to cover the branch computation in the next round. This is likely in:

- `scheduler_rs/src/address_assign.rs` - `compute_live_intervals()` function
- The dependency tracking should include hash values → branch offset computation dependency

**Fix approach:**
1. Extend hash VReg liveness to cover branch computation in next round
2. Or: Ensure branch offset computation reads hash values before they expire
3. Or: Track cross-round dependencies (hash values used in next round's branch)

## Verification

To verify the fix:
1. Generate kernel with `--enable-trace-ids`
2. Check that round 1 hash VRegs are live at cycle 50
3. Verify Op 159 reads from correct addresses [13-20]
4. Confirm round 2 outputs match reference

## Tools Used

- **Trace IDs:** `--enable-trace-ids` to capture semantic trace points
- **Scheduler JSON:** Correlated trace outputs with VReg assignments and intervals
- **Liveness analysis:** Checked which VRegs are live at which cycles
- **Dependency tracing:** Traced offset computation back to hash values

This demonstrates the power of combining trace IDs with scheduler JSON to pinpoint exact root causes.
