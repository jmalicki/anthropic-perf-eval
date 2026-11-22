#!/usr/bin/env python3
"""Window optimality probe: can CP-SAT improve the warm start one window at a time?

    # Build the 16-round shaped greedy warm start, sweep W=50 windows (free and fixed
    # encodings) and the rolling pass, validate the resulting kernel:
    uv run python -m scripts.window_optimality_probe --rounds 16 --window-sizes 50 \\
        --report window_probe_w50.json

    # Summarise an existing report (Markdown tables), or several merged:
    uv run python -m scripts.window_optimality_probe --summarize window_probe_w50.json

The generator builds its usual warm start (by default the shaped greedy, `--greedy-barriers
none --greedy-priority wavefront`) and, instead of the whole-program CP-SAT solve, runs the
probe in `scheduler_rs` (`CPSAT_WINDOW_PROBE=spec.json`): every op outside a window of W
cycles is pinned at its warm-start cycle, the ops after the window move as one block, and
CP-SAT minimises the makespan, i.e. maximises how far that block moves earlier. Every
window is solved against the unchanged warm start ("sweep"); then non-overlapping windows
are applied in sequence, each solved against the schedule the earlier ones improved
("rolling"). The generator then lowers and validates the best rolling schedule.

Extra generator options go after `--` (e.g. `-- --tree-reduce-max-depth 2`). The report is
JSON (`--report`), rewritten after every window, so it can be summarised while the run is
still going. See docs/WINDOW_OPTIMALITY_PROBE.md.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHAPED_GREEDY = ["--greedy-barriers", "none", "--greedy-priority", "wavefront"]


# ----------------------------------------------------------------------------
# Summary (pure functions on report dicts)
# ----------------------------------------------------------------------------
def phase_of(start: int, w: int, makespan: int, edge_fraction: float = 0.1) -> str:
    """Which part of the schedule a window [start, start + w) sits in.

    start-up: it starts in the first `edge_fraction` of the schedule; tail: it ends in the
    last `edge_fraction`; steady state: otherwise. A window covering both edges is "whole".
    """
    edge = edge_fraction * makespan
    early = start < edge
    late = start + w > makespan - edge
    if early and late:
        return "whole"
    if early:
        return "start-up"
    if late:
        return "tail"
    return "steady"


def summarize(report: dict) -> dict:
    """Aggregate a probe report: per (W, encodings) sweep rows, rolling rows, phases."""
    base = report["base_makespan"]
    sweep_rows = []
    groups: dict[tuple[int, str], list[dict]] = {}
    for e in report["entries"]:
        if e["phase"] == "sweep":
            groups.setdefault((e["w"], e["encodings"]), []).append(e)
    for (w, enc), es in sorted(groups.items()):
        solved = [e for e in es if e["makespan"] is not None]
        optimal = [e for e in es if e["status"] == "optimal"]
        improved = [e for e in es if e["saved"] > 0]
        open_bounds = [e for e in solved if e["status"] != "optimal" and (e["saved_bound"] or 0) > e["saved"]]
        by_phase: dict[str, int] = {}
        for e in improved:
            p = phase_of(e["start"], e["w"], base)
            by_phase[p] = by_phase.get(p, 0) + 1
        times = [e["wall_secs"] for e in es]
        sweep_rows.append({
            "w": w,
            "encodings": enc,
            "windows": len(es),
            "solved": len(solved),
            "optimal": len(optimal),
            "improved": len(improved),
            "max_saved": max((e["saved"] for e in es), default=0),
            "total_saved": sum(e["saved"] for e in es),
            "open_gap_windows": len(open_bounds),
            "max_saved_bound": max((e["saved_bound"] or 0 for e in solved), default=0),
            "improved_by_phase": by_phase,
            "encoding_change_windows": sum(
                1 for e in improved if e["shift_changes"] or e["select_changes"] or e["const_changes"]
            ),
            "median_secs": statistics.median(times) if times else 0.0,
            "max_secs": max(times, default=0.0),
            "median_inside": statistics.median([e["inside"] for e in es]) if es else 0,
        })
    rolling_rows = []
    for w, final in report.get("rolling_final", []):
        es = [e for e in report["entries"] if e["phase"] == "rolling" and e["w"] == w]
        rolling_rows.append({
            "w": w,
            "windows": len(es),
            "improved": sum(1 for e in es if e["saved"] > 0),
            "optimal": sum(1 for e in es if e["status"] == "optimal"),
            "final": final,
            "saved": base - final,
            "median_secs": statistics.median([e["wall_secs"] for e in es]) if es else 0.0,
        })
    return {"base_makespan": base, "sweep": sweep_rows, "rolling": rolling_rows}


def markdown(summary: dict) -> str:
    """Markdown tables for a summary."""
    out = [f"Reference makespan: {summary['base_makespan']} cycles.", ""]
    out.append("| W | encodings | windows | optimal | improved | max saved | max saved bound "
               "| windows with open gap | improved by phase | improved with encoding changes "
               "| median ops inside | median s | max s |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in summary["sweep"]:
        phases = ", ".join(f"{k} {v}" for k, v in sorted(r["improved_by_phase"].items())) or "-"
        out.append(
            f"| {r['w']} | {r['encodings']} | {r['windows']} | {r['optimal']} | {r['improved']} "
            f"| {r['max_saved']} | {r['max_saved_bound']} | {r['open_gap_windows']} | {phases} "
            f"| {r['encoding_change_windows']} | {r['median_inside']:.0f} | {r['median_secs']:.1f} "
            f"| {r['max_secs']:.1f} |"
        )
    if summary["rolling"]:
        out += ["", "| W | rolling windows | improved | optimal | final makespan | saved | median s |",
                "|---|---|---|---|---|---|---|"]
        for r in summary["rolling"]:
            out.append(
                f"| {r['w']} | {r['windows']} | {r['improved']} | {r['optimal']} | {r['final']} "
                f"| {r['saved']} | {r['median_secs']:.1f} |"
            )
    return "\n".join(out)


def merge_reports(reports: list[dict]) -> dict:
    """Merge reports of one reference (e.g. one run per W) into one."""
    if not reports:
        raise ValueError("no reports")
    bases = {r["base_makespan"] for r in reports}
    if len(bases) != 1:
        raise ValueError(f"reports have different reference makespans: {sorted(bases)}")
    merged = dict(reports[0])
    merged["entries"] = [e for r in reports for e in r["entries"]]
    merged["rolling_final"] = [x for r in reports for x in r.get("rolling_final", [])]
    return merged


# ----------------------------------------------------------------------------
# Running the generator
# ----------------------------------------------------------------------------
def build_spec(args: argparse.Namespace) -> dict:
    spec = {
        "window_sizes": args.window_sizes,
        "stride_fraction": args.stride_fraction,
        "encodings": args.encodings,
        "sweep": not args.no_sweep,
        "rolling": not args.no_rolling,
        "deterministic_time": args.deterministic_time,
        "time_limit_secs": args.time_limit,
        "output": str(Path(args.report).resolve()),
    }
    if args.cycle_range:
        spec["cycle_range"] = list(args.cycle_range)
    if args.workers is not None:
        spec["workers"] = args.workers
    if args.ignore_subsolvers is not None:
        spec["ignore_subsolvers"] = args.ignore_subsolvers
    if args.tightening is not None:
        spec["asap_alap_tightening"] = args.tightening
    return spec


def generator_command(args: argparse.Namespace, extra: list[str]) -> list[str]:
    cmd = [sys.executable, str(ROOT / "generate_ssa_kernel.py"), "--rounds", str(args.rounds)]
    if not args.no_shaped_greedy:
        cmd += SHAPED_GREEDY
    # The whole-program solve does not run, but the per-section repair would: give it the
    # same deterministic budget so the warm start is reproducible.
    cmd += ["--cpsat-deterministic-time", str(args.deterministic_time),
            "--cpsat-time-limit", str(args.time_limit)]
    if args.output_json:
        cmd += ["--output-json", args.output_json]
    if not args.no_validate:
        cmd += ["--validate"]
        if args.save_kernel:
            cmd += ["--save-kernel", args.save_kernel]
    return cmd + extra


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    extra: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summarize", nargs="+", metavar="REPORT", help="Summarise existing report(s) and exit")
    p.add_argument("--json", action="store_true", help="With --summarize: print the summary as JSON")
    p.add_argument("--rounds", type=int, default=16)
    p.add_argument("--window-sizes", type=int, nargs="+", default=[50], metavar="W")
    p.add_argument("--stride-fraction", type=float, default=0.5, help="Sweep stride as a fraction of W (default 0.5)")
    p.add_argument("--encodings", nargs="+", choices=["free", "fixed"], default=["free", "fixed"])
    p.add_argument("--no-sweep", action="store_true", help="Skip the independent sweep")
    p.add_argument("--no-rolling", action="store_true", help="Skip the rolling pass")
    p.add_argument("--cycle-range", type=int, nargs=2, metavar=("LO", "HI"), help="Only windows starting in [LO, HI)")
    p.add_argument("--deterministic-time", type=float, default=30.0, help="CP-SAT deterministic time per window")
    p.add_argument("--time-limit", type=float, default=900.0, help="Wall-clock backstop per window (s)")
    p.add_argument("--workers", type=int, default=None, help="CP-SAT workers per window (default: tuning's)")
    p.add_argument("--tightening", choices=["none", "simple", "symmetry-aware", "energetic"], default=None,
                   help="ASAP/ALAP tightening in the window models (default: the generator's, energetic)")
    p.add_argument("--ignore-subsolvers", nargs="*", default=None, metavar="NAME",
                   help="CP-SAT subsolvers to leave out of window solves (default: max_lp max_lp_sym "
                        "reduced_costs pseudo_costs; pass with no names to keep all)")
    p.add_argument("--report", default="window_probe.json", help="Report JSON path")
    p.add_argument("--output-json", default="scheduler_output_window_probe.json", help="Scheduler output JSON")
    p.add_argument("--save-kernel", default="kernel_window_probe.json", help="Kernel JSON (with validation)")
    p.add_argument("--no-validate", action="store_true", help="Do not validate the final kernel")
    p.add_argument("--no-shaped-greedy", action="store_true",
                   help="Use the default greedy shape instead of --greedy-barriers none --greedy-priority wavefront")
    args = p.parse_args(argv)

    if args.summarize:
        reports = [json.loads(Path(f).read_text()) for f in args.summarize]
        summary = summarize(merge_reports(reports))
        print(json.dumps(summary, indent=2) if args.json else markdown(summary))
        return 0

    spec = build_spec(args)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(spec, f)
        spec_path = f.name
    cmd = generator_command(args, extra)
    print("Window probe spec:", json.dumps(spec), flush=True)
    print("Command: CPSAT_WINDOW_PROBE=" + spec_path + " " + " ".join(cmd), flush=True)
    try:
        rc = subprocess.call(cmd, cwd=ROOT, env={**os.environ, "CPSAT_WINDOW_PROBE": spec_path})
    finally:
        os.unlink(spec_path)
    if Path(args.report).exists():
        print()
        print(markdown(summarize(json.loads(Path(args.report).read_text()))))
    return rc


if __name__ == "__main__":
    sys.exit(main())
