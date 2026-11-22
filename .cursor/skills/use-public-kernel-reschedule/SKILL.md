---
name: use-public-kernel-reschedule
description: Separate emission quality from scheduling quality - lift any straight-line kernel (ours or a public leaderboard kernel) into a dependency graph with exact scratch and Z3-exact memory edges, report op counts, floors, critical path and fill, and reschedule the same ops with greedy or CP-SAT. Use when comparing our kernel with a public one, asking "is the gap emission or scheduling?", or capturing a third-party kernel builder safely.
---

# Public kernel lift and reschedule

**Modules:** `scripts/capture_public_kernel.py` (sandboxed capture), `scripts/kernel_lift.py`
(lifter and report), `verification/memory_alias.py` (exact memory aliasing, Z3),
`scripts/reschedule_kernel.py` (greedy and CP-SAT on a lifted graph).
Results and method: [docs/PUBLIC_KERNEL_RESCHEDULE.md](../../../docs/PUBLIC_KERNEL_RESCHEDULE.md).

## How to run

```bash
# 1. Capture a third-party builder's instructions (untrusted code). Keep the builder in its own
#    empty directory, outside the repo. It runs under sandbox-exec: no network, no writes outside
#    a temp dir, no reads under $HOME, only the stdlib and upstream frozen_problem.py importable.
uv run python -I scripts/capture_public_kernel.py /path/to/their_perf_takehome.py --out k880.json --source URL

# 2. Lift and report (any kernel JSON: ours from --save-kernel, or a captured one). NAME=PATH names rows.
uv run python -I scripts/kernel_lift.py k880=k880.json ours=kernel_to_validate.json \
    --concrete-runs 4 --save-graph-dir graphs/ --json reports.json
uv run python -I scripts/kernel_lift.py k880=k880.json --table        # one Markdown row per kernel

# 3. Reschedule the same ops (encodings and scratch addresses fixed), validate on the simulator.
uv run python -I scripts/reschedule_kernel.py k880.json --graph graphs/k880.graph.json --scheduler greedy
uv run python -I scripts/reschedule_kernel.py k880.json --graph graphs/k880.graph.json \
    --cpsat-deterministic-time 300 --hint greedy --save-kernel k880_cpsat.json
./scripts/check_upstream.sh --kernel k880_cpsat.json                  # unmodified upstream tests
# Value-dependency schedule (addresses re-allocated): drop the WAR/WAW scratch edges. A relaxation:
# no kernel is emitted; reports cycles, CP-SAT bound and the schedule's peak live scratch words.
# Use it for kernels whose addresses were assigned after scheduling (ours).
uv run python -I scripts/reschedule_kernel.py ours.json --graph graphs/ours.graph.json --value-deps \
    --valu-shift --cpsat-deterministic-time 300

# Memory-alias analysis on its own (edges, Z3 counts, concrete cross-check)
uv run python -I verification/memory_alias.py k880.json --concrete-runs 8
```

`-I` keeps the interpreter from importing anything from the current or the kernel's directory.
Lifting a kernel that uses memory as scratch (880, 899) takes minutes: most of it is Z3.

## What the report means

| Field | Meaning |
|---|---|
| ops | Slots per engine; vector-op equivalents = VALU + ALU/8 |
| floor (fixed) | max over engines of ceil(ops / slots): these ops, these encodings |
| floor (shift) | same, if any VALU op other than multiply_add/vbroadcast could run as 8 ALU ops |
| critical path | longest path of the full graph (RAW + WAR + WAW on scratch words, memory edges): a bound for any schedule that keeps these addresses |
| dataflow critical path | longest chain of value edges only: what a different address assignment could approach |
| gap | simulator cycles minus floor (fixed) |
| start-up / tail | leading / trailing cycles below 90% of the joint VALU+ALU peak (7.5 vector-op equivalents) |
| concrete check | random inputs, address tracing: every pair seen to alias must have an edge; edges never seen are Assumption A's slack |

## Rules (see the doc for proofs)

- Scratch edges are exact per word: RAW latency 1, WAR latency 0 (reads precede the cycle's writes), WAW latency 1.
- Memory edges only where Z3 finds the two addresses can be equal under the upstream preconditions; `unknown` is an error, never an edge.
- Ops that must share a cycle (two ops each reading what the other overwrites) form co-issue groups; schedulers keep them together.
- CP-SAT here is "CP-SAT on their ops" (same solver, cumulative form and search settings as our model, `interleave_search` on), not our SSA pipeline. Quote its deterministic-time budget; a run that prints `stopped on the wall-clock limit` is not reproducible.
- The greedy runs two priorities (critical path, and original issue order) and keeps the better; both are printed.
- Do not commit third-party kernels, their lifted graphs or reschedules (the take-home notice forbids publishing solutions). Keep them in the scratchpad or the git-ignored `build/`; commit aggregate numbers only.

## Tests

`uv run pytest tests/test_kernel_lift.py` (hand-made kernels: edge latencies, lanes, alias decisions by value sets and by Z3, must-alias forwarding, co-issue groups, greedy/CP-SAT validity, sandbox). Z3 lemmas MA1-MA3 run in `./scripts/test.sh proofs`.
