"""Why a greedy schedule shifts the vector ops it does: accounting, idle-slot classes, phases.

From a scheduler output JSON this prints:

1. **Shift accounting.** Vector ops V, shifted s, cycles T and idle vector slots I satisfy
   V - s = 6 T - I exactly (every vector op not shifted takes one vector slot). So when the
   vector ALU is busy in every steady-state cycle, the shift count is fixed by the cycle
   count: a schedule that is longer for any reason shifts fewer ops. "Missing" shifts
   against the capacity floor's split are 6 (T - floor) - I, and the cycles above the floor
   split into those any schedule of the emission needs (the static head/tail bound,
   `scripts.static_bounds`, minimum over encodings) and the rest.
2. **Idle scalar-ALU slots by class**, per cycle: the vector ALU not full (nothing to shift);
   full with a ready shiftable op waiting (a declined shift); full with a non-shiftable op
   (multiply_add, vbroadcast) waiting while shiftable ops hold vector slots (priority
   inversion); full with nothing waiting. In shift equivalents (slots / 8).
3. **Round bands** (setup, rounds 0-4, 5-10, 11-14, 15): per band, the shiftable ops, the
   shifts taken, the op-cycles shiftable ops spent ready and waiting, and the idle
   scalar-ALU slots in the cycles whose vector work belongs to that band. Shifts help only
   where idle scalar slots and ready shiftable work coincide.
4. **Select resolution by round**: SelectChoices run as a flow vselect or a multiply_add
   (`fma1` one op, `fma2` subtract + multiply_add), and the mean wait (issue - ready) of
   flow ops per round: a long wait means the one flow slot is the queue.

"Ready" is operand availability (the producer's last lowered op + 1: the last lane of a
split-shifted producer, the multiply_add of a two-op select); it ignores the wavefront window.

Usage (from the project root):
    uv run python -m scripts.shift_limit_diagnosis sched.json [more.json ...] [--floor N]

docs/GREEDY_SHAPE_ABLATION.md section 12 uses it.
"""
import argparse
import collections
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
sys.path.insert(0, os.path.dirname(_HERE))

from scripts.schedule_report import lowered_with_ssa  # noqa: E402

FIXED_VECTOR = {"multiply_add", "vbroadcast"}
CAP = {"load": 2, "valu": 6, "alu": 12, "flow": 1, "store": 2}
BANDS = (("setup", None, None), ("r0-4", 0, 4), ("r5-10", 5, 10), ("r11-14", 11, 14), ("r15", 15, 10 ** 9))
IDLE_CLASSES = ("valu not full", "declined (shiftable waiting)", "inversion (fixed waiting)",
                "full, nothing waiting")


def band_of(rnd):
    for name, lo, hi in BANDS:
        if lo is None and rnd is None:
            return name
        if lo is not None and rnd is not None and lo <= rnd <= hi:
            return name
    return "setup"


def analyze(d: dict) -> dict:
    """Everything the report prints, as numbers."""
    total = d["total_cycles"]
    ssa = {o["id"]: o for o in d["ssa_ops"]}
    use = {e: [0] * total for e in CAP}
    shifted = set()
    start, finish = {}, {}
    for low, s, lane in lowered_with_ssa(d):
        if low["engine"] in use:
            use[low["engine"]][low["cycle"]] += 1
        if s is not None:
            start[s["id"]] = min(start.get(s["id"], 10 ** 9), low["cycle"])
            finish[s["id"]] = max(finish.get(s["id"], -1), low["cycle"])
            if lane:
                shifted.add(s["id"])
    valu_ssa = [o for o in d["ssa_ops"] if o["engine"] == "valu"]
    on_valu = collections.defaultdict(list)
    for o in valu_ssa:
        if o["id"] not in shifted:
            on_valu[o["cycle"]].append(o)

    def ready(o):
        # A split-shifted producer's lanes straddle two cycles: its result is complete only
        # after the last one.
        return max((finish[x] + 1 for x in o["dep_ids"] if x in finish), default=0)

    w_shift, w_fixed = [0] * total, [0] * total
    wait_by_band = collections.Counter()
    flow_wait = collections.defaultdict(list)
    for o in d["ssa_ops"]:
        if o["id"] not in start:
            continue
        r, s = ready(o), start[o["id"]]
        if o["engine"] == "valu":
            arr = w_fixed if o["opcode"] in FIXED_VECTOR else w_shift
            for t in range(r, min(s, total)):
                arr[t] += 1
            if o["opcode"] not in FIXED_VECTOR:
                wait_by_band[band_of(o["source_info"]["round"])] += max(0, s - r)
        elif o["engine"] == "flow":
            flow_wait[o["source_info"]["round"]].append(s - r)

    idle = collections.Counter()
    band_alu_idle = collections.Counter()
    band_cycles = collections.Counter()
    for t in range(total):
        free = CAP["alu"] - use["alu"][t]
        if use["valu"][t] < CAP["valu"]:
            k = IDLE_CLASSES[0]
        elif w_shift[t]:
            k = IDLE_CLASSES[1]
        elif w_fixed[t] and any(o["opcode"] not in FIXED_VECTOR for o in on_valu[t]):
            k = IDLE_CLASSES[2]
        else:
            k = IDLE_CLASSES[3]
        idle[k] += free
        bands = collections.Counter(band_of(o["source_info"]["round"]) for o in on_valu[t])
        b = bands.most_common(1)[0][0] if bands else "setup"
        band_alu_idle[b] += free
        band_cycles[b] += 1

    band = {name: collections.Counter() for name, _, _ in BANDS}
    for o in valu_ssa:
        if o["opcode"] in FIXED_VECTOR:
            continue
        b = band_of(o["source_info"]["round"])
        band[b]["shiftable"] += 1
        band[b]["shifted"] += o["id"] in shifted
    for name in band:
        band[name]["ready_wait"] = wait_by_band[name]
        band[name]["alu_idle"] = band_alu_idle[name]
        band[name]["cycles"] = band_cycles[name]

    selects = collections.defaultdict(collections.Counter)
    for low in d["ops"]:
        sid = low.get("from_select_choice")
        if sid is None or sid not in ssa or low["opcode"] == "-":
            continue
        form = "flow" if low["engine"] == "flow" else f"fma{ssa[sid].get('select_fma_vector_ops') or ''}"
        selects[ssa[sid]["source_info"]["round"]][form] += 1

    # Selects run in their multiply_add form take vector slots too (subtract + multiply_add).
    select_vector = sum(1 for low in d["ops"] if low.get("from_select_choice") is not None
                        and low["engine"] == "valu")
    vector_ops = len(valu_ssa) + select_vector
    valu_idle = sum(CAP["valu"] - x for x in use["valu"])
    return {
        "cycles": total, "vector_ops": vector_ops, "shifted": len(shifted), "valu_idle": valu_idle,
        "identity_holds": vector_ops - len(shifted) == CAP["valu"] * total - valu_idle,
        "alu_idle": dict(idle), "bands": {k: dict(v) for k, v in band.items()},
        "selects": {r: dict(v) for r, v in selects.items()},
        "flow_wait": {r: sum(v) / len(v) for r, v in flow_wait.items() if v},
        "peak_live": d.get("peak_live_words"),
    }


def static_bound(d: dict):
    """The static head/tail bound, or None with a warning on stderr when it cannot be
    computed (scripts.static_bounds needs numpy)."""
    try:
        from scripts.static_bounds import static_makespan_bound
    except ImportError as e:
        print(f"warning: static head/tail bound not computed ({type(e).__name__}: {e}); the cycles above "
              "the floor are not split into forced and scheduling loss", file=sys.stderr)
        return None
    return static_makespan_bound(d, "any")["bound"]


def report(name: str, a: dict, floor: int | None, bound: int | None) -> str:
    out = [f"== {name}: {a['cycles']} cycles, {a['shifted']} of {a['vector_ops']} vector ops shifted, "
           f"{a['valu_idle']} vector slots idle, peak live {a['peak_live']}"]
    out.append(f"  identity V - s = 6T - I: {'holds' if a['identity_holds'] else 'FAILS'}")
    if floor is not None:
        missing = 6 * (a["cycles"] - floor) - a["valu_idle"]
        line = f"  vs floor {floor}: {a['cycles'] - floor} cycles above it = {missing} shifts 'missing'"
        if bound is not None:
            forced = max(0, min(bound, a["cycles"]) - floor)
            line += (f"; static head/tail bound {bound}: {forced} of those cycles are needed by any schedule, "
                     f"{a['cycles'] - max(bound, floor)} are scheduling loss")
        else:
            line += "; static head/tail bound not computed"
        out.append(line)
    out.append("  idle scalar-ALU slots (shift equivalents = slots / 8):")
    for k in IDLE_CLASSES:
        v = a["alu_idle"].get(k, 0)
        out.append(f"    {k:30s} {v:6d}  ({v / 8:6.1f})")
    out.append("  band     cycles  shiftable  shifted  ready-wait op-cycles  alu idle in band cycles")
    for name, _, _ in BANDS:
        b = a["bands"][name]
        out.append(f"    {name:7s} {b['cycles']:6d} {b.get('shiftable', 0):10d} {b.get('shifted', 0):8d} "
                   f"{b['ready_wait']:21d} {b['alu_idle']:24d}")
    rounds = sorted(set(a["selects"]) | set(a["flow_wait"]), key=lambda r: -1 if r is None else r)
    if rounds:
        out.append("  round  selects (form: count)              mean flow wait")
        for r in rounds:
            sel = ", ".join(f"{k} {v}" for k, v in sorted(a["selects"].get(r, {}).items()))
            out.append(f"    {str(r):5s} {sel:34s} {a['flow_wait'].get(r, 0):8.1f}")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("schedules", nargs="+", help="scheduler output JSONs (--output-json)")
    ap.add_argument("--floor", type=int, default=None,
                    help="the emission's capacity floor (e.g. emission_structure choice_floor) for the "
                         "missing-shift accounting")
    ap.add_argument("--no-static-bound", action="store_true", help="skip the static head/tail bound")
    args = ap.parse_args(argv)
    for path in args.schedules:
        with open(path) as f:
            d = json.load(f)
        a = analyze(d)
        bound = None if args.no_static_bound or args.floor is None else static_bound(d)
        print(report(os.path.basename(path), a, args.floor, bound))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
