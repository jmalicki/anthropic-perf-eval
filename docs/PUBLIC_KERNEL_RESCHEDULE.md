# Public kernels on our schedulers: emission versus scheduling

November 2026. This doc takes the exact ops of public leaderboard kernels, rebuilds their
dependency graph and schedules those ops again. The point is to separate two questions: how
good is their emission (which ops, which encodings), and how good is their schedule
(which cycle each op issues in).

RESULTS_PLACEHOLDER

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
  store to load 1, load to store 0, store to store 1.
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
   are (no edge). `unknown` raises an error; it is never turned into an edge. Each load lane
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
