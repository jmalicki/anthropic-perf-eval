#!/usr/bin/env python3
"""How good is this schedule? One report from a scheduler output JSON.

    uv run python -m scripts.schedule_report sched.json
    uv run python -m scripts.schedule_report sched.json --json
    uv run python -m scripts.schedule_report sched.json --compare other.json

Prints cycles achieved, per-engine op counts and floors, the joint capacity floor for the
ops THIS run emitted, the gap to it, per-round cycle spans and utilisation over time.

The floor is slot arithmetic over the emitted ops. It is conditional on the emission: a
kernel that emits fewer ops has a lower floor. It is not a bound on the problem (see
docs/BOUND_GAP_ANALYSIS.md: 1,271 for our 16-round emission, public kernels reach 863).

Run it as a module from the project root (scripts/bisect.py shadows the standard library's
bisect when a script is started from inside scripts/).
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys

from scripts.analyze_bound_gap import CAP, VEQ_PER_CYCLE
from scripts.theoretical_lower_bounds import compute_capacity_bound

ENGINES = ("valu", "alu", "load", "store", "flow")
VLEN = 8
NON_SHIFTABLE_OPCODES = ("multiply_add", "vbroadcast")
SCOPE_NOTE = "Fewer emitted ops lower it; it is not a bound on the problem."


# ----------------------------------------------------------------------------
# Floor arithmetic (pure functions; unit-tested on hand-made inputs)
# ----------------------------------------------------------------------------
def joint_alu_floor(vector_ops, scalar_ops, max_shift=None):
    """Fewest cycles T for the two ALUs when a vector op may run as 8 scalar ops.

    Minimise T subject to, for some number shifted s in [0, max_shift]:
        vector_ops - s      <= 6 * T     (vector ALU)
        scalar_ops + 8 * s  <= 12 * T    (scalar ALU)

    Returns (T, s) with s the largest feasible shift at T. `max_shift` defaults to
    `vector_ops` (every vector op may shift).
    """
    max_shift = vector_ops if max_shift is None else max(0, min(max_shift, vector_ops))
    # 8*(V - s) + (S + 8*s) <= 48T + 12T, so T >= (8V + S) / 60 whatever s is.
    t = math.ceil((VLEN * vector_ops + scalar_ops) / (VLEN * CAP["valu"] + CAP["alu"]))
    while True:
        lo = max(0, vector_ops - CAP["valu"] * t)
        hi = min(max_shift, (CAP["alu"] * t - scalar_ops) // VLEN) if CAP["alu"] * t >= scalar_ops else -1
        if lo <= hi:
            return t, hi
        t += 1


def emission_floor(vector_ops, scalar_ops, load_ops=0, flow_ops=0, store_ops=0, max_shift=None):
    """Capacity floor for one emission: max of the joint ALU floor and the load, flow and
    store floors. Ignores precedence and scratch, so a real schedule is never below it."""
    joint, shift = joint_alu_floor(vector_ops, scalar_ops, max_shift)
    terms = {
        "alu+valu": joint,
        "load": compute_capacity_bound(load_ops, CAP["load"]),
        "flow": compute_capacity_bound(flow_ops, CAP["flow"]),
        "store": compute_capacity_bound(store_ops, CAP["store"]),
    }
    floor = max(terms.values())
    veq = vector_ops + scalar_ops / VLEN
    return {
        "cycles": floor,
        "binding": [name for name, v in terms.items() if v == floor and floor > 0],
        "terms": terms,
        "shift_at_floor": shift,
        "valu_cycles_at_floor": compute_capacity_bound(vector_ops - shift, CAP["valu"]),
        "alu_cycles_at_floor": compute_capacity_bound(scalar_ops + VLEN * shift, CAP["alu"]),
        "vector_op_equivalents": veq,
        "continuous_alu_floor": veq / VEQ_PER_CYCLE,
    }


# ----------------------------------------------------------------------------
# Reading the scheduler JSON
# ----------------------------------------------------------------------------
def load(path):
    with open(path) as f:
        return json.load(f)


def lowered_with_ssa(d):
    """Yield (lowered op, SSA op or None, is_shift_lane) for every lowered op.

    A vector op run on the scalar ALU appears as 8 lowered `alu` ops with
    id = ssa_id * 1000 + lane. Such an id can coincide with an unrelated SSA id in a large
    kernel, so a scalar op whose id resolves to a vector SSA op is also treated as a lane.
    """
    ssa = {o["id"]: o for o in d.get("ssa_ops") or []}
    for o in d["ops"]:
        s = ssa.get(o["id"])
        lane = False
        if ssa and o["engine"] == "alu" and (s is None or s["engine"] == "valu"):
            parent = ssa.get(o["id"] // 1000)
            if parent is not None and parent["engine"] == "valu":
                s, lane = parent, True
        yield o, s, lane


def shifted_count(d):
    """Vector ops executed as 8 scalar ops in this schedule."""
    if d.get("valu_shifted") is not None:
        return d["valu_shifted"]
    return len(d.get("shifted_alu_cycles") or {})


def analyze(d, buckets=10):
    """Build the whole report as a plain dict (what --json prints)."""
    cycles = d["total_cycles"]
    ops = d["ops"]
    counts = collections.Counter(o["engine"] for o in ops)
    shifted = shifted_count(d)
    vector_ops = counts["valu"] + shifted
    native_scalar = counts["alu"] - VLEN * shifted
    if native_scalar < 0:
        raise ValueError(f"{counts['alu']} scalar ops cannot hold {shifted} shifted vector ops")
    ssa_valu = [o for o in d.get("ssa_ops") or [] if o["engine"] == "valu"]
    non_shiftable = sum(1 for o in ssa_valu if o["opcode"] in NON_SHIFTABLE_OPCODES) if ssa_valu else None
    max_shift = None if non_shiftable is None else max(shifted, vector_ops - non_shiftable)

    engines = {}
    for e in ENGINES:
        slots = CAP[e] * cycles
        engines[e] = {
            "ops": counts[e],
            "capacity": CAP[e],
            "slots": slots,
            "utilisation_pct": 100.0 * counts[e] / slots if slots else 0.0,
            "floor": compute_capacity_bound(counts[e], CAP[e]),
        }

    floor = emission_floor(vector_ops, native_scalar, counts["load"], counts["flow"], counts["store"], max_shift)
    gap = cycles - floor["cycles"]
    opts = d.get("run_options") or {}
    rounds = per_round(d)
    # JSONs written through the binding without run_options (tests, older scripts): only
    # CP-SAT writes lb_cycles, and the rounds are visible in the SSA ops.
    use_cpsat = opts.get("use_cpsat", d.get("lb_cycles") is not None)
    n_rounds = opts.get("rounds", sum(1 for r in rounds if r["round"] is not None) or None)
    return {
        "run": {
            "cycles": cycles,
            "scheduler": "cpsat" if use_cpsat else "greedy",
            "rounds": n_rounds,
            "batch_size": opts.get("batch_size"),
            "git_sha": d.get("git_sha"),
            "total_ops": len(ops),
            "solver_bound_for_emission": d.get("lb_cycles"),
            "peak_live_words": d.get("peak_live_words"),
        },
        "engines": engines,
        "emission": {
            "vector_ops": vector_ops,
            "vector_ops_on_valu": counts["valu"],
            "vector_ops_shifted_to_alu": shifted,
            "native_scalar_ops": native_scalar,
            "non_shiftable_vector_ops": non_shiftable,
            "vector_op_equivalents": floor["vector_op_equivalents"],
        },
        "floor": floor,
        "gap": {"cycles": gap, "pct_of_floor": 100.0 * gap / floor["cycles"] if floor["cycles"] else 0.0},
        "rounds": rounds,
        "utilisation": utilisation_over_time(d, buckets),
    }


def per_round(d):
    """First/last cycle, span and op counts per round, from ssa_ops[*].source_info.round.

    `round` is None for ops outside the rounds (init, input load, output store). Overlap is
    with the previous row: cycles shared with it (0 when this round starts after it ends)."""
    rows = {}
    for o, s, lane in lowered_with_ssa(d):
        if s is None:
            continue
        r = (s.get("source_info") or {}).get("round")
        row = rows.setdefault(r, {"round": r, "first_cycle": o["cycle"], "last_cycle": o["cycle"],
                                  "vector_ops": 0, "scalar_ops": 0, "load_ops": 0, "flow_ops": 0,
                                  "store_ops": 0, "lowered_ops": 0})
        row["first_cycle"] = min(row["first_cycle"], o["cycle"])
        row["last_cycle"] = max(row["last_cycle"], o["cycle"])
        row["lowered_ops"] += 1
        if lane:
            row["vector_ops"] += 1 / VLEN  # 8 lanes make one vector op
        elif o["engine"] == "valu":
            row["vector_ops"] += 1
        elif o["engine"] == "alu":
            row["scalar_ops"] += 1
        else:
            row[o["engine"] + "_ops"] += 1
    out = [rows[r] for r in sorted(rows, key=lambda r: (r is not None, r if r is not None else -1))]
    prev = None
    for row in out:
        row["vector_ops"] = round(row["vector_ops"])
        row["span"] = row["last_cycle"] - row["first_cycle"] + 1
        if row["round"] is None:
            row["overlap_prev"] = None
            continue
        row["overlap_prev"] = None if prev is None else max(0, prev["last_cycle"] - row["first_cycle"] + 1)
        row["start_after_prev_end"] = None if prev is None else row["first_cycle"] - prev["last_cycle"]
        prev = row
    return out


def utilisation_over_time(d, buckets=10):
    """Percent of slots used per engine in `buckets` equal slices of the schedule."""
    cycles = d["total_cycles"]
    buckets = max(1, min(buckets, cycles)) if cycles else 1
    edges = [cycles * i // buckets for i in range(buckets + 1)]
    used = {e: [0] * buckets for e in ENGINES}
    for o in d["ops"]:
        if o["engine"] in used and cycles:
            used[o["engine"]][min(buckets - 1, o["cycle"] * buckets // cycles)] += 1
    # A cycle c lands in bucket c*buckets//cycles; count the cycles that land in each.
    width = collections.Counter(c * buckets // cycles for c in range(cycles))
    return {
        "bucket_first_cycle": [min((c for c in range(edges[i], cycles) if c * buckets // cycles == i), default=edges[i])
                               for i in range(buckets)],
        "bucket_cycles": [width[i] for i in range(buckets)],
        "pct": {e: [100.0 * used[e][i] / (CAP[e] * width[i]) if width[i] else 0.0 for i in range(buckets)]
                for e in ENGINES},
    }


# ----------------------------------------------------------------------------
# Text output
# ----------------------------------------------------------------------------
def _binding(floor):
    return ", ".join(floor["binding"]) or "none"


def floor_sentence(rep):
    f = rep["floor"]
    return (f"floor for this emission ({f['vector_op_equivalents']:,.0f} vector-op equivalents): "
            f"{f['cycles']:,}. {SCOPE_NOTE}")


def summary_lines(rep):
    """The few lines generate_ssa_kernel.py prints after every build."""
    f, g = rep["floor"], rep["gap"]
    return [
        f"Emission floor: {f['cycles']} cycles ({_binding(f)} binds; "
        f"{f['vector_op_equivalents']:,.0f} vector-op equivalents, {rep['engines']['load']['ops']} loads)",
        f"Gap to floor: {g['cycles']} cycles ({g['pct_of_floor']:.1f}% above the floor)",
        f"The floor is for the ops this run emitted. {SCOPE_NOTE}",
    ]


def format_report(rep, verbose_rounds=True):
    run, f, g, em = rep["run"], rep["floor"], rep["gap"], rep["emission"]
    out = []
    out.append(f"Schedule: {run['cycles']} cycles, scheduler={run['scheduler']}, rounds={run['rounds']}, "
               f"batch={run['batch_size']}, lowered ops={run['total_ops']}, sha={run['git_sha']}")
    out.append("")
    out.append("Engine   ops    cap  slots   util   floor=ceil(ops/cap)")
    for e in ENGINES:
        x = rep["engines"][e]
        out.append(f"{e:<6} {x['ops']:>6} {x['capacity']:>5} {x['slots']:>6} {x['utilisation_pct']:>5.1f}% {x['floor']:>6}")
    out.append("")
    ns = "" if em["non_shiftable_vector_ops"] is None else f", {em['non_shiftable_vector_ops']} of them not shiftable"
    out.append(f"Emitted: {em['vector_ops']} vector ops ({em['vector_ops_on_valu']} on valu + "
               f"{em['vector_ops_shifted_to_alu']} run as 8 scalar ops{ns}), {em['native_scalar_ops']} native scalar ops")
    out.append(f"  vector-op equivalents: {em['vector_ops']} + {em['native_scalar_ops']}/8 = "
               f"{f['vector_op_equivalents']:.2f}; at {VEQ_PER_CYCLE} per cycle = {f['continuous_alu_floor']:.2f} cycles")
    t = f["terms"]
    out.append(f"Capacity {floor_sentence(rep)}")
    out.append(f"  binds: {_binding(f)}. Terms: alu+valu {t['alu+valu']} (with {f['shift_at_floor']} vector ops shifted: "
               f"valu {f['valu_cycles_at_floor']}, alu {f['alu_cycles_at_floor']}), load {t['load']}, flow {t['flow']}, store {t['store']}")
    out.append(f"  encodings (const, select) as chosen in this run; precedence and scratch ignored")
    out.append(f"Gap: achieved {run['cycles']} - floor {f['cycles']} = {g['cycles']} cycles ({g['pct_of_floor']:.1f}% above the floor)")
    if run["solver_bound_for_emission"] is not None:
        lb = run["solver_bound_for_emission"]
        status = "optimal for this emission" if lb == run["cycles"] else "not proven optimal for this emission"
        out.append(f"CP-SAT bound for this emission (lb_cycles): {lb} ({status}; not a bound on the problem)")
    if rep["rounds"] and verbose_rounds:
        out.append("")
        out.append("Round  first   last  span  overlap  vector  scalar   load   flow")
        for r in rep["rounds"]:
            name = "setup" if r["round"] is None else str(r["round"])
            ov = "-" if r["overlap_prev"] is None else str(r["overlap_prev"])
            out.append(f"{name:>5} {r['first_cycle']:>6} {r['last_cycle']:>6} {r['span']:>5} {ov:>8} "
                       f"{r['vector_ops']:>7} {r['scalar_ops']:>7} {r['load_ops']:>6} {r['flow_ops']:>6}")
        timed = [r for r in rep["rounds"] if r["round"] is not None]
        if len(timed) > 1:
            total_span = sum(r["span"] for r in timed)
            total_ov = sum(r["overlap_prev"] or 0 for r in timed)
            out.append(f"  sum of round spans {total_span}, total overlap with the previous round {total_ov} cycles "
                       f"(0 = rounds run strictly one after another)")
    u = rep["utilisation"]
    out.append("")
    out.append(f"Utilisation over time (% of slots, {len(u['bucket_cycles'])} buckets)")
    out.append("from cycle " + " ".join(f"{c:>5}" for c in u["bucket_first_cycle"]))
    for e in ENGINES:
        out.append(f"{e:<10} " + " ".join(f"{p:>5.0f}" for p in u["pct"][e]))
    return "\n".join(out)


def format_compare(a, b, name_a, name_b):
    out = [f"A = {name_a}", f"B = {name_b}", "", f"{'':<28} {'A':>9} {'B':>9} {'B-A':>8}"]

    def row(label, x, y, fmt="{:>9}"):
        delta = f"{y - x:>+8.2f}" if isinstance(y - x, float) else f"{y - x:>+8}"
        out.append(f"{label:<28} {fmt.format(x)} {fmt.format(y)} {delta}")

    row("cycles", a["run"]["cycles"], b["run"]["cycles"])
    row("emission floor", a["floor"]["cycles"], b["floor"]["cycles"])
    row("gap to floor", a["gap"]["cycles"], b["gap"]["cycles"])
    row("vector-op equivalents", a["floor"]["vector_op_equivalents"], b["floor"]["vector_op_equivalents"], "{:>9.2f}")
    for e in ENGINES:
        row(f"{e} ops", a["engines"][e]["ops"], b["engines"][e]["ops"])
    for e in ENGINES:
        row(f"{e} floor", a["engines"][e]["floor"], b["engines"][e]["floor"])
    out.append(f"{'binding':<28} {_binding(a['floor']):>9} {_binding(b['floor']):>9}")
    out.append(f"{'scheduler':<28} {a['run']['scheduler']:>9} {b['run']['scheduler']:>9}")
    ra = {r["round"]: r for r in a["rounds"]}
    rb = {r["round"]: r for r in b["rounds"]}
    keys = sorted(set(ra) | set(rb), key=lambda r: (r is not None, r if r is not None else -1))
    if keys:
        out.append("")
        out.append("Round   span A   span B      B-A   first A  first B")
        for k in keys:
            x, y = ra.get(k), rb.get(k)
            name = "setup" if k is None else str(k)
            sa = "-" if x is None else str(x["span"])
            sb = "-" if y is None else str(y["span"])
            dd = "-" if x is None or y is None else f"{y['span'] - x['span']:+d}"
            fa = "-" if x is None else str(x["first_cycle"])
            fb = "-" if y is None else str(y["first_cycle"])
            out.append(f"{name:>5} {sa:>8} {sb:>8} {dd:>8} {fa:>9} {fb:>8}")
    out.append("")
    out.append(f"Floors are per emission. {SCOPE_NOTE}")
    return "\n".join(out)


def main(argv=None):
    p = argparse.ArgumentParser(description="Schedule quality report from a scheduler output JSON")
    p.add_argument("json_file", nargs="?", default="scheduler_output.json")
    p.add_argument("--json", action="store_true", help="Emit the report as JSON")
    p.add_argument("--compare", metavar="OTHER_JSON", help="Diff against a second run (cycles, floors, per-round spans)")
    p.add_argument("--buckets", type=int, default=10, help="Time buckets for utilisation (default 10)")
    args = p.parse_args(argv)

    rep = analyze(load(args.json_file), args.buckets)
    other = analyze(load(args.compare), args.buckets) if args.compare else None
    if args.json:
        payload = rep if other is None else {"a": rep, "b": other}
        json.dump(payload, sys.stdout, indent=2)
        print()
    elif other is not None:
        print(format_compare(rep, other, args.json_file, args.compare))
    else:
        print(format_report(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
