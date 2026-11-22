#!/usr/bin/env python3
"""Reschedule the exact ops of a kernel (e.g. a public leaderboard kernel) and validate it.

    uv run python -m scripts.reschedule_kernel kernel.json --graph NAME.graph.json \\
        --cpsat-deterministic-time 300 --save-kernel out.json

The ops, their encodings and their scratch addresses are kept: only the issue cycles
change. The dependency graph comes from ``scripts/kernel_lift.py`` (exact scratch RAW/WAR/WAW
edges on lane words and Z3-exact memory edges); pass ``--graph`` to reuse one written by
``kernel_lift.py --save-graph-dir``, otherwise the kernel is lifted here.

Two schedulers, both written for an arbitrary externally supplied op graph:

* greedy: cycle-by-cycle list scheduling, highest remaining critical path first (the same
  priority family as our Rust greedy's critical-path mode), with zero-latency successors
  allowed into the same cycle;
* CP-SAT: one start variable per op in [ASAP, horizon - height], precedence with latencies,
  one cumulative per engine (ALU 12, VALU 6, load 2, store 2, flow 1), minimise makespan,
  warm-started from a hint schedule (greedy, or the kernel's own schedule with
  ``--hint original``). This is "CP-SAT on their ops", the same solver and constraint
  form as our model, not our SSA pipeline (which needs SSA values, not fixed addresses).

``--value-deps`` schedules on value dependencies only (addresses re-allocated): the
write-after-read/write-after-write scratch edges are dropped, so no kernel is emitted; it is a
relaxation whose CP-SAT bound holds for any re-allocation.

``--valu-shift`` lets CP-SAT run any VALU op except multiply_add/vbroadcast as 8 scalar ALU
ops in the same cycle (our VALU->ALU shift): an encoding freedom, so off by default.

CP-SAT here is the OR-Tools Python wheel. Do not import it into a process that has imported
``scheduler_rs`` (which links the Homebrew OR-Tools library): the two crash together; run
this script as its own process.

Every schedule is checked against the graph and run on the upstream simulator against the
reference for ``--seeds`` random inputs. ``--save-kernel`` writes a kernel JSON for
``scripts/check_upstream.sh --kernel``.
"""

from __future__ import annotations

import argparse
import collections
import heapq
import json
import os
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.kernel_lift import (  # noqa: E402
    CAP, ENGINES, NON_SHIFTABLE, VLEN, asap, check_schedule, condense, heights, lift, load_graph, value_dependency_graph,
)
from scripts.kernel_memory_use import peak_live_words, scratch_values  # noqa: E402
from verification.memory_alias import MemLayout, load_bundles, trace_memory  # noqa: E402


# =============================================================================
# Greedy list scheduler
# =============================================================================


def greedy(g, priority: str = "height") -> list[int]:
    """List-schedule the graph, one co-issue group (usually a single op) at a time.
    `priority`: "height" (longest path to the end first, ties by original cycle) or
    "original" (original cycle first, i.e. replay the kernel's own order)."""
    c = condense(g)
    h = heights(g, c)
    m = len(c.members)
    need = [collections.Counter(g.ops[i].engine for i in c.members[x]) for x in range(m)]
    remaining = [len(p) for p in c.preds]
    earliest = [0] * m
    start = [-1] * g.n

    def key(x: int):
        first = c.members[x][0]
        if priority == "original":
            return (g.ops[first].cycle, first)
        return (-h[first], g.ops[first].cycle, first)

    waiting: dict[int, list[int]] = collections.defaultdict(list)  # cycle -> groups ready then
    waiting[0] = [x for x in range(m) if remaining[x] == 0]
    heap: list = []
    placed = 0
    t = 0
    while placed < m:
        for x in waiting.pop(t, []):
            heapq.heappush(heap, (key(x), x))
        used = collections.Counter()
        deferred = []
        while heap:
            _, x = heapq.heappop(heap)
            if any(used[e] + k > CAP[e] for e, k in need[x].items()):
                deferred.append(x)
                continue
            used.update(need[x])
            for i in c.members[x]:
                start[i] = t
            placed += 1
            for y, lat in c.succs[x]:
                earliest[y] = max(earliest[y], t + lat)
                remaining[y] -= 1
                if remaining[y] == 0:
                    if earliest[y] <= t:
                        heapq.heappush(heap, (key(y), y))
                    else:
                        waiting[earliest[y]].append(y)
        waiting[t + 1].extend(deferred)
        t += 1
        if t > 10 * g.n + 10:
            raise RuntimeError("greedy made no progress")
    return start


# =============================================================================
# CP-SAT
# =============================================================================


def cpsat(g, hint: list[int], deterministic_time: float, time_limit: float, workers: int = 8,
          valu_shift: bool = False, log: bool = True, seed: int = 0):
    """Minimise makespan over the graph; returns (start, shifted op set, info dict)."""
    from ortools.sat.python import cp_model

    c = condense(g)
    lo = asap(g, c)
    h = heights(g, c)
    horizon = max(s + 1 for s in hint)
    m = cp_model.CpModel()
    start = [m.NewIntVar(lo[i], horizon - h[i], f"s{i}") for i in range(g.n)]
    for (s, d), lat in sorted(g.edges.items()):
        m.Add(start[d] >= start[s] + lat)
    makespan = m.NewIntVar(max(lo[i] + h[i] for i in range(g.n)), horizon, "makespan")
    succ = g.succs()
    for i in range(g.n):
        if not succ[i]:
            m.Add(makespan >= start[i] + 1)
    by_engine: dict[str, list] = collections.defaultdict(list)
    shift_vars = {}
    for i, op in enumerate(g.ops):
        e = op.engine
        if valu_shift and e == "valu" and op.slot[0] not in NON_SHIFTABLE:
            s = m.NewBoolVar(f"shift{i}")
            shift_vars[i] = s
            by_engine["valu"].append((m.NewOptionalFixedSizeIntervalVar(start[i], 1, ~s, f"v{i}"), 1))
            by_engine["alu"].append((m.NewOptionalFixedSizeIntervalVar(start[i], 1, s, f"a{i}"), VLEN))
            m.AddHint(s, 0)
        else:
            by_engine[e].append((m.NewFixedSizeIntervalVar(start[i], 1, f"i{i}"), 1))
    for e in ENGINES:
        if by_engine[e]:
            m.AddCumulative([iv for iv, _ in by_engine[e]], [d for _, d in by_engine[e]], CAP[e])
    for i in range(g.n):
        m.AddHint(start[i], hint[i])
    m.AddHint(makespan, horizon)
    m.Minimize(makespan)
    solver = cp_model.CpSolver()
    p = solver.parameters
    p.max_deterministic_time = deterministic_time
    p.max_time_in_seconds = time_limit
    p.num_workers = workers
    p.random_seed = seed
    p.log_search_progress = log
    # Same search settings as our Rust model (cpsat_scheduler/model.rs build_sat_parameters):
    # interleave_search makes the result independent of thread timing, so with a
    # deterministic-time budget the same command gives the same schedule.
    p.interleave_search = True
    p.linearization_level = 2
    p.cp_model_probing_level = 2
    p.randomize_search = True
    t0 = time.perf_counter()
    status = solver.Solve(m)
    info = {
        "status": solver.StatusName(status),
        "objective": solver.ObjectiveValue() if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None,
        "bound": solver.BestObjectiveBound(),
        "deterministic_time": solver.deterministic_time,
        "wall_seconds": round(time.perf_counter() - t0, 1),
        "hit_wall_clock": (time.perf_counter() - t0 >= time_limit * 0.98
                           and solver.deterministic_time < deterministic_time),
        "horizon": horizon,
    }
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, set(), info
    sched = [solver.Value(v) for v in start]
    shifted = {i for i, s in shift_vars.items() if solver.Value(s)}
    return sched, shifted, info


# =============================================================================
# Emission and validation
# =============================================================================


def emit(g, start: list[int], shifted: set[int] = frozenset()) -> list[dict]:
    """Bundles from issue cycles (empty cycles dropped: the simulator does not count them)."""
    T = max(start) + 1
    bundles: list[dict] = [collections.defaultdict(list) for _ in range(T)]
    for i, op in enumerate(g.ops):
        b = bundles[start[i]]
        if i in shifted:
            name, d, a, c = op.slot
            for k in range(VLEN):
                b["alu"].append([name, d + k, a + k, c + k])
        else:
            b[op.engine].append(list(op.slot))
    return [dict(b) for b in bundles if b]


def validate(bundles: list[dict], seeds: int, layout: MemLayout | None = None) -> bool:
    """Run on the upstream simulator for `seeds` random inputs; compare output values."""
    layout = layout or MemLayout()
    for seed in range(1, seeds + 1):
        _, machine, ref = trace_memory(bundles, 1000 + seed, layout)
        if machine.mem[layout.val_p:layout.size] != ref[layout.val_p:layout.size]:
            return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("kernel", help="kernel JSON (bundles under 'instrs')")
    ap.add_argument("--graph", help="graph JSON from kernel_lift.py --save-graph-dir (else lift here)")
    ap.add_argument("--scheduler", choices=("greedy", "cpsat"), default="cpsat")
    ap.add_argument("--hint", choices=("greedy", "original"), default="greedy",
                    help="CP-SAT warm start: our greedy schedule or the kernel's own")
    ap.add_argument("--cpsat-deterministic-time", type=float, default=120.0)
    ap.add_argument("--cpsat-time-limit", type=float, default=3600.0, help="wall-clock backstop")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--valu-shift", action="store_true", help="allow VALU->ALU shift (encoding change)")
    ap.add_argument("--value-deps", action="store_true",
                    help="drop the write-after-read/write-after-write scratch edges (needs --graph): "
                         "a relaxation that would need new addresses, so no kernel is emitted; reports "
                         "cycles and the schedule's peak live scratch words")
    ap.add_argument("--seeds", type=int, default=8, help="random inputs to validate on")
    ap.add_argument("--save-kernel", help="write the rescheduled kernel JSON here")
    ap.add_argument("--json", help="write a result summary JSON here")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args()

    data = json.load(open(args.kernel))
    bundles = data["instrs"] if isinstance(data, dict) else data
    g = load_graph(args.graph) if args.graph else lift(bundles)
    original = [o.cycle for o in g.ops]
    if args.value_deps:
        if not args.graph:
            ap.error("--value-deps needs --graph")
        g = value_dependency_graph(g)
    if check_schedule(g, original):
        print("the kernel's own schedule violates its lifted graph; refusing to reschedule", file=sys.stderr)
        return 2
    result = {"kernel": args.kernel, "ops": g.n, "edges": len(g.edges),
              "original_cycles": len({c for c in original})}
    t0 = time.perf_counter()
    runs = {p: greedy(g, p) for p in ("height", "original")}
    for p, s in runs.items():
        result[f"greedy_{p}_cycles"] = max(s) + 1
    best = min(runs, key=lambda p: max(runs[p]))
    gstart = runs[best]
    result["greedy_cycles"] = max(gstart) + 1
    result["greedy_priority"] = best
    result["greedy_seconds"] = round(time.perf_counter() - t0, 1)
    print(f"greedy: critical-path priority {result['greedy_height_cycles']} cycles, original-order priority "
          f"{result['greedy_original_cycles']} (kernel's own: {result['original_cycles']})", flush=True)
    sched, shifted = gstart, set()
    if args.scheduler == "cpsat":
        hint = original if args.hint == "original" else gstart
        sched, shifted, info = cpsat(g, hint, args.cpsat_deterministic_time, args.cpsat_time_limit,
                                     args.workers, args.valu_shift, log=not args.quiet)
        result["cpsat"] = info | {"hint": args.hint, "valu_shift": args.valu_shift, "shifted": len(shifted),
                                  "deterministic_time_budget": args.cpsat_deterministic_time}
        if sched is None:
            print(f"CP-SAT found no schedule: {info}")
            json.dump(result, open(args.json, "w"), indent=1) if args.json else None
            return 1
        print(f"CP-SAT: {max(sched) + 1} cycles, bound {info['bound']:.0f}, status {info['status']}, "
              f"dtime {info['deterministic_time']:.0f}, {info['wall_seconds']}s, shifted {len(shifted)}"
              + ("  WARNING: CP-SAT stopped on the wall-clock limit; result depends on machine load"
                 if info["hit_wall_clock"] else ""))
    vals = scratch_values(g.ops)
    result["peak_live_words"] = peak_live_words(vals, sched)[0]
    result["original_peak_live_words"] = peak_live_words(vals, original)[0]
    if args.value_deps:
        result.update(cycles=max(sched) + 1, value_dependency_schedule=True)
        print(f"value-dependency schedule (addresses re-allocated): {result['cycles']} cycles, peak live scratch words {result['peak_live_words']} "
              f"(kernel's own schedule: {result['original_peak_live_words']}); not emitted (needs new addresses)")
        if args.json:
            json.dump(result, open(args.json, "w"), indent=1)
        return 0
    if not shifted:
        bad = check_schedule(g, sched)
        if bad:
            print(f"INVALID schedule: {bad[:5]}", file=sys.stderr)
            return 1
    out = emit(g, sched, shifted)
    ok = validate(out, args.seeds)
    result.update(cycles=len(out), validated_seeds=args.seeds, valid=ok)
    print(f"rescheduled: {len(out)} cycles; simulator vs reference on {args.seeds} inputs: {'PASS' if ok else 'FAIL'}")
    if args.save_kernel:
        meta = {k: data.get(k) for k in ("forest_height", "batch_size", "rounds")} if isinstance(data, dict) else {}
        json.dump(meta | {"instrs": out}, open(args.save_kernel, "w"))
    if args.json:
        json.dump(result, open(args.json, "w"), indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
