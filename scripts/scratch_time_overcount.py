#!/usr/bin/env python3
"""Scratch liveness per cycle: the CP-SAT "cycle" count against the exact phase count.

Within a VLIW cycle every read happens before every write, so a word whose value is last
read at cycle C can be rewritten at cycle C. Address assignment uses that (closed interval
[def, last_read] with same-cycle reuse; docs/ALLOCATION_CONJECTURES.md, "Real semantics").
The two counts compared here:

- cycle: what the CP-SAT scratch cumulative counted with `--cpsat-scratch-time cycle`, each
  value live over the half-open [def, last_read + 1), i.e. every cycle t with
  def <= t <= last_read. At cycle C it counts both the values read for the last time at C
  and the values written at C.
- phase: the exact load. Cycle t has a read phase (2t) and a write phase (2t + 1). A value
  with def < last_read holds its words over [2 def + 1, 2 last_read]; a point value
  (def == last_read) over [2 def, 2 def + 1]. This is `address_assign::interval_span`, the
  rule the allocator uses. The per-cycle phase count is the larger of the two phases.
  `--cpsat-scratch-time phase` (the default) counts each value over [def, last_read), whose
  peak equals this phase peak (verification/allocation_conjectures.py, A8).

Reads the per-VReg intervals of a scheduler JSON written with `--liveness-detail full`
(`vregs[*].interval_start/interval_end/size_words`).

  uv run python -m scripts.scratch_time_overcount sched.json [more.json ...]
  uv run python -m scripts.scratch_time_overcount sched.json --json
"""

import argparse
import json
import sys


def value_intervals(sched):
    """(start, end, size) per VReg of a scheduler JSON, closed [def, last_read]."""
    vregs = sched.get("vregs")
    if not vregs:
        raise ValueError("scheduler JSON has no 'vregs' (generate with --liveness-detail full)")
    return [(v["interval_start"], v["interval_end"], v["size_words"]) for v in vregs]


def span(start, end):
    """Half-cycle span [lo, hi] (inclusive) of a closed interval, as address_assign::interval_span."""
    if end > start:
        return 2 * start + 1, 2 * end
    return 2 * start, 2 * start + 1


def per_cycle_counts(intervals):
    """Per cycle t: (cycle count, phase count) in words.

    cycle count: sum of sizes with start <= t <= end.
    phase count: max over the read phase 2t and the write phase 2t + 1 of the sum of sizes
    whose span covers that phase.
    """
    if not intervals:
        return []
    n = max(max(s, e) for s, e, _ in intervals) + 2
    cyc = [0] * (n + 1)
    half = [0] * (2 * n + 2)
    for s, e, size in intervals:
        cyc[s] += size
        cyc[max(s, e) + 1] -= size
        lo, hi = span(s, e)
        half[lo] += size
        half[hi + 1] -= size
    out = []
    c = 0
    h = [0] * (2 * n + 2)
    acc = 0
    for i in range(2 * n + 2):
        acc += half[i]
        h[i] = acc
    for t in range(n):
        c += cyc[t]
        out.append((c, max(h[2 * t], h[2 * t + 1])))
    return out


def report(sched):
    """Peaks of both counts and the largest per-cycle overcount (cycle - phase)."""
    counts = per_cycle_counts(value_intervals(sched))
    over = [(c - p, t) for t, (c, p) in enumerate(counts)]
    max_over, at = max(over) if over else (0, None)
    return {
        "cycles": sched.get("total_cycles"),
        "peak_cycle_count": max((c for c, _ in counts), default=0),
        "peak_phase_count": max((p for _, p in counts), default=0),
        "max_overcount": max_over,
        "max_overcount_at": at,
        "cycles_overcounted": sum(1 for d, _ in over if d > 0),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("json", nargs="+", help="scheduler output JSON(s) with --liveness-detail full")
    ap.add_argument("--json", dest="as_json", action="store_true", help="print machine-readable results")
    args = ap.parse_args(argv)
    results = {path: report(json.load(open(path))) for path in args.json}
    if args.as_json:
        print(json.dumps(results, indent=2))
        return 0
    print(f"{'schedule':<40} {'cycles':>6} {'peak cycle':>10} {'peak phase':>10} {'max over':>8} {'at':>6} {'#cyc over':>9}")
    for path, r in results.items():
        print(f"{path[-40:]:<40} {r['cycles']!s:>6} {r['peak_cycle_count']:>10} {r['peak_phase_count']:>10} "
              f"{r['max_overcount']:>8} {r['max_overcount_at']!s:>6} {r['cycles_overcounted']:>9}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
