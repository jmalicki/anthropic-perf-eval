# Big-O Analysis: Scratch Allocation (Address Assignment)

This document characterizes the time and space complexity of **scratch allocation** — assigning physical scratch addresses to VRegs after scheduling. The implementation lives in `scheduler_rs/src/address_assign.rs` and is invoked from the pipeline via `scheduler_rs/src/kernel_builder/pipeline/passes/assign_addrs.rs`.

## Overview

1. **Compute live intervals** (`compute_live_intervals`): For each VReg, determine [start, end] cycles (definition to last use, with special cases for SelectChoice FMA, shifted ALU, AluDerive, dead VRegs, CSE, vector groups).
2. **Assign addresses** (`assign_addresses`): Sort intervals by start, then linear-scan allocate: expire intervals that end before current start, maintain a **rangemap::RangeSet** of used address ranges and a free list; try freed ranges first (overlap check via RangeSet), then first-fit from **gaps()** in address order; fallback to sorted used-address list when needed.
3. **Verify** (`verify_no_address_conflicts_fast`): Ensure no two simultaneously-live VRegs have overlapping address ranges (O(V log V)).

## Notation

| Symbol | Meaning |
|--------|--------|
| **N** | Number of SSA ops |
| **V** | Number of VRegs (live intervals). **V = O(N)** (one dest per op plus aux defs like sub_temp; vector lanes folded into groups). |
| **U** | Max uses per op (bounded constant, e.g. 4–8) |
| **VG** | Number of vector groups (`config.vector_group_lanes.len()`) |
| **D** | Number of ConstChoice AluDerive ops |
| **B** | Scratch budget (1536 words; constant) |

---

## 1. `compute_live_intervals`

- **First pass (defs):** For each op: `dest_write_interval`, `aux_defs` → O(1) each. **O(N).**
- **Second pass (extend to last use):** For each op, `use_cycles` returns O(U) uses; for each, HashMap get_mut. **O(N · U) = O(N).**
- **AluDerive:** Build `id_to_info` O(N). For each of D derive ops, look up and extend two producer intervals. **O(N + D).**
- **Dead VReg extension:** Build `used_vregs` O(N). Iterate intervals, extend dead. **O(N + V) = O(N).**
- **CSE shared sub_temp:** Insert O(|cse_shared_sub_temp_intervals|). **O(CSE).**
- **Vector groups:** For each of VG groups: min/max over lane intervals O(1), `ops.iter()` for consumer_end and lane_use_end **O(N)** each. **O(VG · N).**
- **zero_vreg, A4 sanity:** O(N) and O(N · U + V) = **O(N).**

**Time: O(N + VG · N) = O(N · (1 + VG)).**  
With VG small in practice, this is **O(N)**.

**Space:** O(V) for the intervals map.

---

## 2. `assign_addresses`

- **compute_live_intervals:** **O(N · (1 + VG)).**
- **Use-def check:** For each op, use_cycles O(U); for each use, interval lookup O(1) or vector_group_lanes find O(VG). **O(N · U · (1 + VG)) = O(N · (1 + VG)).**
- **compute_peak_liveness:** Collect and sort O(V) intervals → **O(V log V)**. One pass with heap expire/insert → **O(V log V)**.
- **Sort intervals:** **O(V log V).**
- **Linear scan** (main loop over V intervals):
  - **Expire:** Amortized each interval pushed once, popped once from `active` (BinaryHeap). **O(V log V)** total. On expire, remove range from **rangemap::RangeSet** and push to `free_ranges` → **O(log V)** per expire.
  - **Find address:** Try `free_ranges` first: pop until a range fits; overlap check via **RangeSet::overlaps** → **O(log V)** per try. Worst case O(V) pops per interval → **O(V log V)** per interval. If no free range fits, use **RangeSet::gaps()** (iterator over maximally-sized gaps); scan for first gap with length ≥ size → **O(V)** per interval. Fallback: build sorted used list from RangeSet + `first_gap_of_length` → **O(V log V)** per interval.
- **Post-allocation vector lanes:** O(VG) groups, 8 lanes each. **O(VG).**
- **Statistics:** O(V).

**Dominant term in the linear scan:** Free-range tries **O(V log V)** per interval or gap scan **O(V)** per interval → **O(V² log V)** or **O(V²)** worst case. The implementation uses the **rangemap** crate (`RangeSet<i64>`) for used ranges and gap iteration.

**Time for assign_addresses:** **O(N · (1 + VG) + V²)** (or **O(V² log V)** for the allocation loop when free-range path dominates). With **V = O(N)**:

- **O(N · (1 + VG) + N²).**
- When VG is small: **O(N + N²) = O(N²)** worst case.

**Space:** O(V) for intervals, sorted list, active heap, **RangeSet** (used ranges), free_ranges, addrs_written_this_cycle, and temporary clones for gap search. **O(V) = O(N).**

---

## 3. `verify_no_address_conflicts_fast`

- Build entries from intervals + vreg_to_addr: **O(V).**
- Sort by address: **O(V log V).**
- Step 1 (adjacent overlap check): **O(V).**
- Step 2 (group by (addr, size), sort by interval.start, check overlaps): **O(V log V).**

**Time: O(V log V).**  
**Space: O(V).**

---

## 4. Pipeline pass (`AssignAddressesPass`)

The pass builds `AddressAssignConfig` (CSE, const_choice, select_fma, shifted_alu, vector_group_lanes, vreg_def) from context — all **O(N)** or O(config sizes) — then calls `assign_addresses` and `verify_no_address_conflicts_fast`.

**Total time:** **O(N · (1 + VG) + V²)** = **O(N²)** when V = O(N) and VG is small.  
**Total space:** **O(N).**

---

## Summary table

| Phase | Time | Space |
|--------|------|--------|
| compute_live_intervals | O(N · (1 + VG)) | O(V) |
| assign_addresses (total) | O(N · (1 + VG) + V²) | O(V) |
| └ linear scan (worst case) | O(V²) | O(V) |
| verify_no_address_conflicts_fast | O(V log V) | O(V) |
| **End-to-end scratch allocation** | **O(N · (1 + VG) + N²)** | **O(N)** |

Scratch allocation is **quadratic in V (and hence in N)** in the worst case. The implementation uses the **rangemap** crate (`RangeSet<i64>`) to maintain used address ranges and to iterate **gaps** for first-fit; overlap checks are O(log V), and the free list is tried before the gap iterator so freed slots are reused (matching previous allocation behavior).
