# Public kernels on our schedulers: emission versus scheduling

November 2026. This doc takes the exact ops of public leaderboard kernels, rebuilds their
dependency graph and schedules those ops again. The point is to separate two questions: how
good is their emission (which ops, which encodings), and how good is their schedule
(which cycle each op issues in).

## What this means

**The public kernels' remaining advantage is emission, not scheduling.** All three public
kernels sit 4 to 8 cycles (0.4 to 0.8%) above the capacity floor of their own ops. Their
schedules are essentially optimal for their op sets: there is nothing left for a scheduler to
win. Rescheduling their exact ops with fixed addresses, our tools reproduce their cycle
counts but do not beat them. CP-SAT proves the 880 and 899 schedules optimal for their ops at
their addresses (bounds 880 and 899). It proves 982 is within 3 cycles of the best possible
schedule (bound 979). Their VALU/ALU splits are also already balanced: letting any VALU op
move to the ALU lowers their floors by only 1 to 2 cycles (874 → 873, 895 → 893, 974 → 973).
So the optional VALU-shift experiment on their ops has at most 2 cycles to find, and was not
run.

**Our 1,099-cycle kernel is the opposite case.** It sits 105 cycles above the floor of its own
ops (994 with the VALU→ALU split free). That breaks into two parts:

- 56 cycles of VALU/ALU imbalance. Our ALU/VALU split is fixed at emission, and with that
  split the VALU floor is 1,050. They run far more work on the scalar ALU: 10,418 to 11,585
  ALU ops against our 8,384, at about the same total work.
- 49 cycles of schedule above that floor: 16 under-filled start-up cycles, a 62-cycle tail,
  and 377 cycles below 90% of the joint VALU+ALU peak, where the public kernels have 11 to
  28.

How much of that a better schedule could recover, on our ops with the split free and the
addresses re-allocated (value-dependency schedule, CP-SAT, dtime 200):
- CP-SAT proves a lower bound of 1,009 cycles. So at least 15 cycles above the 994 floor are
  forced by the dependency structure (start-up and tail), and at most 89 of the 105 are
  recoverable by scheduling.
- CP-SAT itself found no better schedule than the 1,098 it was warm-started from, so how many
  of those 89 a real scheduler would recover is still open.

The gap to the leaders' level is therefore mostly emission:

| | ours (1,099) | 880 | difference |
|---|---:|---:|---:|
| vector-op equivalents | 7,343 | 6,541 | +802 (+12%) |
| loads | 1,988 | 1,724 | +264 |
| floor with free VALU→ALU split | 994 | 873 | +121 |
| cycles above that floor | 105 | 7 | +98 |

1,099 − 880 = 219 cycles: 121 from the floor (more ops, mostly loads and vector work) and
98 from scheduling and the VALU/ALU split.

**Does this make a CP-SAT-based design credible?** Partly.
- As a source of proofs it works. It proved two hand-tuned 20,000-op leaderboard schedules
  optimal: 880 in 10 s, and 899 within its budget. Its LP bound reached the capacity floor
  for 982 (979) in a few minutes, so it can say how close a schedule is to optimal. It is also
  exact about constraints: the same model carries exact memory aliasing, latency-0
  anti-dependences and co-issue groups.
- As a stand-alone improver of 20,000-op schedules at makespan granularity it is weak. It
  never improved a warm start by a single cycle in these budgets: from the greedy's 996 for
  982 it stayed at 996, and from 982 it stayed at 982. Construction has to come from
  something like the public kernels' list schedulers. A plain critical-path list scheduler
  is 4 to 5% worse (919 / 924 / 996). Replaying their issue order reproduces 880 / 899 / 982.
- So the design that matches the evidence is: a good constructive scheduler and emission, then
  CP-SAT for bounds, repair and local improvement.

## Results

### Op counts, floors, critical path

Op counts are slots in the emitted kernel (our kernel's flow count includes its 2 `pause`
bundles). Floors use `scripts/schedule_report.py`'s arithmetic, `emission_floor`:
- "fixed": each engine on its own, with the kernel's own VALU/ALU split;
- "shift": any VALU op other than `multiply_add` or `vbroadcast` may run as 8 ALU ops.

The critical path is counted two ways:
- "with addresses": the longest path of the full lifted graph, with the kernel's own scratch
  addresses;
- "values": value (read-after-write) edges only, scratch and memory.

The gap is simulator cycles minus the fixed floor.

| kernel | VALU | ALU | load | store | flow | vec-op eq. | floor fixed | floor shift | crit. path (addresses / values) | cycles | gap | start-up / tail | cycles < 90% fill |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 880 (Sakauma) | 5,239 | 10,418 | 1,724 | 1,591 | 843 | 6,541 | 874 | 873 | 873 / 441 | 880 | 6 | 3 / 6 | 28 |
| 899 (Sakauma) | 5,365 | 10,655 | 1,756 | 832 | 862 | 6,697 | 895 | 893 | 866 / 195 | 899 | 4 | 5 / 3 | 11 |
| 982 (BasicCoder) | 5,843 | 11,585 | 1,863 | 38 | 857 | 7,291 | 974 | 973 | 871 / 232 | 982 | 8 | 8 / 3 | 18 |
| ours, PR #51 path bits | 6,295 | 8,384 | 1,988 | 64 | 866 | 7,343 | 1,050 | 994 | 1,068 / 206 | 1,099 | 49 | 16 / 62 | 377 |

The binding engine is VALU+ALU for all four on the fixed floor; with the split free, ours is
load-bound (1,988 / 2 = 994).

What the critical-path columns show:
- The 880 kernel is bound twice over. Its floor (874) and its critical path at its own
  addresses (873) both sit within 7 cycles of its 880. Its value-only critical path (441) is
  long because values flow through memory (store, then load).
- Our 1,068 "with addresses" is an artifact of our pipeline: it assigns addresses after
  scheduling, so its address reuse follows its own schedule order (see "Which treatment is
  fair"). The relevant number for us is 206.

### Rescheduling

"Fixed addresses" keeps every op, encoding and scratch/memory address; the kernel is emitted
and validated on the upstream simulator (8 random inputs, plus `./scripts/check_upstream.sh
--kernel` for the CP-SAT and greedy outputs). "Value deps" is the value-dependency schedule
with addresses re-allocated: a relaxation, not emitted; its CP-SAT bound holds for any
re-allocation.

Both greedy columns are our Python list scheduler; both CP-SAT columns are "CP-SAT on their
ops" (`scripts/reschedule_kernel.py`), not our Rust SSA pipeline.

| kernel | treatment | their cycles | greedy, critical-path priority | greedy, original-order priority | CP-SAT (warm start, dtime) | CP-SAT lower bound |
|---|---|---:|---:|---:|---:|---:|
| 880 | fixed addresses | 880 | 919 | 880 | 880 (warm start 880, dtime 100; proved optimal in 10 s) | 880 |
| 880 | value deps | 880 | 919 | 880 | — | — |
| 899 | fixed addresses | 899 | 924 | 899 | 899 (warm start 899, dtime 100; proved optimal, dtime 260) | 899 |
| 899 | value deps | 899 | 1,264 | 899 | — | — |
| 982 | fixed addresses | 982 | 996 | 982 | 982 (warm start 982, dtime 100) | 979 |
| 982 | value deps | 982 | 1,524 | 982 | — | — |
| ours | fixed addresses | 1,099 | 1,120 | 1,098 | — | — |
| ours | value deps, VALU shift free | 1,099 | 1,546 | 1,098 | 1,098 (warm start 1,098, dtime 200; 1 op shifted) | 1,009 |

Notes:
- The first 982 CP-SAT run started from the critical-path greedy (996). It ended on its
  wall-clock backstop (5,400 s), so it is not deterministic. Its result, 996 with bound 979,
  is reported only as context. All recorded rows use `interleave_search` and a
  deterministic-time budget, and end on that budget.
- The value-dependency greedy keeps the peak of live scratch words close to the kernel's own:
  880 1,231 (own 1,230), 899 1,214 (own 1,241), 982 1,596 (own 1,503; over 1,536, so not
  allocatable as is), ours 1,185 (own 1,033).

### Memory dependencies

| kernel | memory ops | ordered op pairs with a store | memory edges | word pairs Z3-proved disjoint | Z3 calls (rest decided by a sampled model) | aliasing pairs seen on 8 random inputs | of which ordered through a must-alias store | seen without ordering | edges never seen to alias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 880 | 3,266 | 3,929,770 | 441,668 | 16 | 4,433 | 510,091 | 107,355 | 0 | 38,932 |
| 899 | 2,480 | 1,716,832 | 23,077 | 0 | 1,828 | 23,103 | 9,483 | 0 | 9,457 |
| 982 | 1,886 | 70,927 | 1,607 | 0 | 0 | 1,505 | 0 | 0 | 102 |
| ours | 2,018 | 127,072 | 32 | 0 | 0 | 32 | 0 | 0 | 0 |

How to read the table:
- Pairs not listed as edges were proved disjoint, either by value sets (nearly all of them)
  or by Z3, or they are ordered through a store that must alias both.
- No address fell outside its value set, and every kernel's outputs matched the reference on
  all inputs.
- Edges never seen to alias are mostly pairs that alias only for particular branch-bit
  combinations. In the 880 kernel many bit-addressed stores share one dead word, so they can
  alias on any input whose bits send both there.

### How the 880 and 899 kernels use main memory mid-kernel

Main memory is the only place this ISA addresses by a register value. The patterns below come
from the exact store-to-load flow of the alias analysis (`scripts/kernel_memory_use.py`). They
are described at the level the authors describe them; no addresses or code are reproduced.

1. **Tree repacking at start-up** (pattern (a)).
   - What it does: early `vstore`s write tree values, XOR-adjusted by the deferred hash
     constant, back into the tree region in a new layout. Later data-dependent `vload`s read
     them.
   - 880: 31 vstores in cycles 31 to 59, read by 771 vloads.
   - 899: 30 vstores in cycles 52 to 109, read by 512 vloads, plus two mid-kernel vstores
     read by 256 `load_offset`s.
   - 982 does the same in place: 6 vstores read by its scalar gathers.
   - Effect: one vload fetches a contiguous packet holding several candidate nodes, so one
     load slot serves more than one round. The XOR is also folded in once, not per gather.
   - Scratch-only equivalent: per-lane scalar gathers, one load slot per lane per level. For
     the 880 kernel that is the difference between its 1,724 loads and the 2,000 or more a
     scalar-gather design needs (docs/BOUND_GAP_ANALYSIS.md section 1).
2. **Select through memory (lane bucketing by branch bit)**, the main pattern (a form of
   (c)).
   - What it does: for each lane, one scalar store writes a candidate value to a fixed bank
     word. A second scalar store's address is chosen by the lane's branch bit: either the same
     bank word (overwriting it) or a dead word. One `vload` of the bank then returns the
     selected value for all 8 lanes as a vector. Store order within the bank decides which
     candidate wins.
   - 880: 464 fixed-address stores and 976 bit-addressed stores, read by 314 vloads, in cycles
     68 to 868.
   - 899: 384 and 384, read by 304 vloads, in cycles 66 to 525.
   - Effect: a per-lane select, and the scalar-to-vector assembly, done by the store engine.
     The store engine is otherwise idle (2 slots per cycle; the 880 kernel uses 90% of them).
     The scratch-only alternative is a `vselect` on the flow engine, which is about 96% busy in
     both kernels, or a `multiply_add` select on VALU, the binding engine.
   - Accounting: at one vector select per 8 bit-addressed stores, about 122 selects (880) and
     48 (899) are moved off flow and VALU, at the price of 16 store slots and one load slot
     per vector. On the VALU that is about 16 and 6 cycles at the joint peak. The real saving
     is larger where flow is the alternative, because flow has no slack left.
3. **Broadcast through memory** (880 only).
   - What it does: 8 scalar stores of one scalar (a constant or a tree value) to words 0 to 7,
     then one vload: a `vbroadcast` without a VALU slot.
   - Count: 76 stores, 13 vloads, all before cycle 90.
4. **One lane permutation at the tail** (880 only; (c)).
   - What it does: 8 stores of working values into header words and one vload realign one
     group's lanes, as the author's notes describe.
5. **Output stores** (all kernels): 32 `vstore`s of final values, never read back.

**Spilling (pattern (b)) does not occur.** No working value is stored and reloaded long
after.
- The working values parked in memory at the peak are short-lived bank entries: 108 words
  (880) and 50 (899).
- The peak of live scratch words plus those parked working values is 1,301 (880) and 1,275
  (899), well under 1,536. Both would fit in scratch without the round trips; the round trips
  are there for addressing, not capacity.
- The other parked words (260 and 256) are the repacked tree.

**For our emitter:**
- Overflow scratch is not what main memory buys here.
- Two memory-movement primitives are worth having:
  - a tree-repack plus packet-`vload` gather, the larger effect (loads);
  - a store-select that uses the idle store engine for per-lane selects and lane-to-vector
    assembly.
- Our kernel today uses 64 store slots in 1,099 cycles, so the store engine is almost
  entirely free.


## Sources and credit

Third-party code was never committed. Each builder was run once, in a sandbox, and only its
emitted instruction list was kept (see "Capture" below). Measurements and op counts derived
from those lists are recorded here.

| Kernel | Author | Source (fetched 2026-11-21) | sha256 of the file run |
|---|---|---|---|
| 880 | Sakauma | `github.com/Sakauma/my_original_performance_takehome`, `perf_takehome.py` at HEAD | `c955068f8ea8c618…31f710` |
| 899 | Sakauma | same repository, commit `ab9697d6`, `perf_takehome.py` | `ab5c199d255bab0d…a671156` |
| 982 | BasicCoder | `github.com/BasicCoder/original_performance_takehome_work`, `perf_takehome.py` at HEAD | `14207a86ac9a8832…49ba4d7` |
| 1,099 (ours) | this repo, PR #51 "path bits" | `emission/vector-work`, plan `docs/emission_plans/path_flow90.json` | — |

The 913 kernel (Sakauma, commit `c5c8d3e3`) dispatches through `jump_indirect` and is out of
scope: the lifter handles straight-line code only.

## Method

### Capture (`scripts/capture_public_kernel.py`)

The builder file is copied into a fresh temporary directory as `perf_takehome.py`, next to the
unmodified upstream simulator (`upstream/tests/frozen_problem.py`, importable as `problem`).
It runs as `python -I -S -B` (no site-packages, no `PYTHON*` variables, no script directory
on the path) with a minimal environment, under `sandbox-exec`: no network, no writes outside
the temporary directory, no reads under `$HOME` except the Python installation. The driver
calls `KernelBuilder().build_kernel(10, 2047, 256, 16)` and writes `kb.instrs` as JSON; the
result is checked to be plain data (ISA engine names, slots of strings and integers).
`debug` slots are dropped. All three builders ran cleanly; their cycle counts on the upstream
simulator match their published numbers.

### Lifting (`scripts/kernel_lift.py`)

Every slot is one op. An edge `src -> dst` with latency `L` means `cycle[dst] >= cycle[src] + L`.

- **Scratch, per word.** Vector ops touch 8 words, so lanes are exact: a scalar write to lane 3
  only conflicts with readers of lane 3. Read-after-write from the last writer, latency 1
  (results are visible the next cycle). Write-after-read from every read since that write,
  latency 0: a slot reads before the cycle's writes commit, so the overwrite may share the
  reader's cycle. Write-after-write from the last writer, latency 1. A word that is read before
  any write reads the machine's initial 0, and its first writer is ordered after those reads.
  The edges are the minimal exact set: only the last writer, never blanket ordering.
- **Main memory**, exact, from `verification/memory_alias.py` (next section). Same latencies:
  store to load 1, load to store 0, store to store 0 (a cycle's memory writes commit in
  slot order, so a later store may share the cycle if it stays later in the bundle; the
  rescheduler emits each cycle in program order).
- **Co-issue groups.** Keeping the kernel's scratch addresses can force two ops into one
  cycle: op A reads `w` and writes `v` while op B reads `v` and writes `w`. Then each has a
  latency-0 edge to the other. Such strongly connected components of latency-0 edges are
  scheduled as one unit. Our own kernel has them (for example a `load` and an `alu +` in
  cycle 9 swapping roles on s1/s83).

The kernel's own schedule satisfies its lifted graph for all four kernels (checked by the
report: `original_schedule_violations = 0`).

### Exact memory dependencies (`verification/memory_alias.py`)

A memory edge is added only if the two addresses **can be equal for some valid input**,
never because they cannot be told apart.

1. The kernel is executed symbolically with the term store of `verification/kernel_dataflow.py`,
   extended with an exact memory model. A load returns the value of the latest earlier store
   that must alias it, wrapped in `ite(addr == store_addr, store_value, …)` for every later
   store that only may alias it, over the initial memory. Addresses computed from loaded
   values are therefore exact too, which matters for kernels that store pointers.
2. **Preconditions**, from the upstream generator (`build_mem_image`, `Tree.generate`,
   `Input.generate` in `upstream/tests/frozen_problem.py`):
   - header words 0..6 are `rounds=16, n_nodes=2047, batch_size=256, forest_height=10,
     forest_values_p=7, inp_indices_p=2054, inp_values_p=2310`; word 7 holds the root's value
     (the generator's write of `extra_room` to word 7 is overwritten by the tree values, and
     the memory list is truncated to `inp_values_p + batch_size = 2566` words);
   - tree values and input values are arbitrary integers in [0, 2^30);
   - every input index is 0.
   So constant-address loads of the header and index regions fold to constants, and every
   memory address becomes a term over tree values and input values.
3. **Per pair.** First a value-set over-approximation (constants, `x & c`, `x % c`, `x >> c`,
   selects, arithmetic on small sets): disjoint sets prove the pair disjoint. Otherwise Z3
   over 32-bit bit-vectors: SAT means the addresses can be equal (edge), UNSAT means they never
   are (no edge). `unknown` raises an error; it is never turned into an edge. To keep this
   fast, the query's formula is first evaluated on 1,024 fixed assignments of its free
   variables with the same operations as the Z3 expression; a satisfying assignment is a Z3
   model, so SAT is decided without the solver, and only the remaining queries (in practice
   the UNSAT ones) go to Z3. A test checks that both routes give identical edges. Each load lane
   scans earlier stores newest first and stops at the first store that must alias it (older
   stores are ordered before that one by store-to-store edges); each store scans earlier loads
   and stores the same way.
4. **Assumption A.** A subterm whose value set is unbounded (a hash value, a tree value) is
   replaced by a fresh 32-bit variable, shared by all occurrences of that subterm. This only
   adds behaviours, so UNSAT is a proof for the real kernel. A SAT witness is an assignment of
   those variables, in practice of the branch bits `h & 1` of different hashes; assuming those
   bits are independent is the one assumption. In practice the addresses depend on the hashes
   only through such bits.
5. **Lemmas** (Z3, in `./scripts/test.sh proofs`): MA1, the word-range overlap test
   `a - b + (wa - 1) <u wa + wb - 1` is exactly "some word of one range equals some word of the
   other", with wrap-around, for widths 1 and 8; MA2, the value-set rules (`x & c` is a submask
   of `c`, `x % c < c` for each `c <= 64`, `x >> s < 2^(32-s)` for `s` in [20, 32),
   `x % 2 == x & 1`); MA3, an instance of the branch-bit abstraction.
6. **Cross-check.** Each kernel is run on random inputs on the upstream simulator with address
   tracing. Every pair seen to alias must have an edge (a missing one would mean the
   preconditions are wrong), and every concrete address must lie in its value set. Memory edges
   never seen to alias are counted: they are the slack of Assumption A (or aliasing that random
   inputs rarely hit).

### Rescheduling (`scripts/reschedule_kernel.py`)

The ops, their encodings and their scratch and memory addresses are kept. Only issue cycles
change.

- **Greedy**: cycle-by-cycle list scheduling over co-issue groups, highest remaining critical
  path first (ties by original cycle), with latency-0 successors admitted into the same cycle.
  This is the priority family of our Rust greedy's critical-path mode, but a Python
  implementation: our Rust greedy and CP-SAT work on SSA values and assign addresses after
  scheduling, and cannot take an externally supplied graph with fixed addresses and latency-0
  edges without a new model.
- **CP-SAT**: one start variable per op in [ASAP, horizon − height], precedence with latencies,
  one cumulative per engine (ALU 12, VALU 6, load 2, store 2, flow 1), minimise makespan,
  warm-started from the greedy schedule (or, with `--hint original`, from the kernel's own).
  The budget is `max_deterministic_time` (quoted per row), with a wall-clock backstop. This is
  "CP-SAT on their ops": the same solver and constraint form as our model, not our SSA pipeline.
- **Value-dependency schedule, addresses re-allocated** (`--value-deps`): the same schedulers on value edges and memory edges only,
  without the edges that reusing scratch addresses adds. Such a schedule needs a new address
  assignment, so no kernel is emitted. It is a relaxation: its CP-SAT bound is a lower bound
  for any reschedule with re-allocated addresses. The schedule's peak live scratch words show
  whether it could plausibly be allocated in 1,536 words.
- Every emitted schedule is checked against the graph and slot limits, run on the upstream
  simulator against the reference on 8 random inputs, and run through the unmodified upstream
  tests with `./scripts/check_upstream.sh --kernel`.

Which treatment is fair depends on when the author chose the addresses. Our pipeline assigns
addresses after scheduling, so its address reuse follows its own schedule order: the
"critical path with these addresses" of our kernel (1,068) is an artifact of that order and
never constrained our scheduler. For our kernel the fair comparisons are the value-dependency
rows. By its author's description, the 880 kernel ships a fixed address table and a fixed issue
plan chosen together; the 899 and 982 builders allocate during or after their own scheduling. Both treatments are reported for every kernel.

## Reproduce

```bash
# capture (builder files in their own empty directory, outside the repo)
uv run python -I scripts/capture_public_kernel.py ext/sakauma_880.py --out k/880.json
# lift, report, memory use, concrete cross-check; writes graphs/NAME.graph.json
uv run python -I scripts/kernel_lift.py k880=k/880.json k899=k/899.json k982=k/982.json \
    ours=kernel_path_flow90.json --concrete-runs 4 --save-graph-dir graphs --json reports.json
# reschedule
uv run python -I scripts/reschedule_kernel.py k/982.json --graph graphs/k982.graph.json --scheduler greedy
uv run python -I scripts/reschedule_kernel.py k/982.json --graph graphs/k982.graph.json \
    --cpsat-deterministic-time 300 --save-kernel k/982_cpsat.json
uv run python -I scripts/reschedule_kernel.py k/982.json --graph graphs/k982.graph.json --value-deps \
    --cpsat-deterministic-time 300
./scripts/check_upstream.sh --kernel k/982_cpsat.json
```
