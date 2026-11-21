#!/usr/bin/env python3
"""Ablate CP-SAT configuration and modelling switches over a benchmark set.

Runs every configuration on every instance with the same deterministic-time budget
(--cpsat-deterministic-time, not wall clock), validates every kernel, and writes a results
table as JSON and Markdown. See .cursor/skills/use-cpsat-ablation/SKILL.md.

    # The standard study: every switch alone, on the standard instances
    uv run python -m scripts.run_cpsat_ablation --out-dir build/cpsat_ablation

    # Own configurations ("name:generate_ssa_kernel.py options"; the first is the baseline)
    uv run python -m scripts.run_cpsat_ablation --instances 2x256,4x64 --budget 10 \\
        --config "baseline:" --config "makespan:--cpsat-objective makespan"

    # Rebuild the tables from finished runs without solving again
    uv run python -m scripts.run_cpsat_ablation --out-dir build/cpsat_ablation --report-only

An instance is ROUNDSxBATCH, optionally with its own budget: 16x256@60.

Finished cells are kept in the output directory and reused on the next invocation (same
instance, configuration options and budget), so a study can be extended or resumed.
Configurations that switch interleaved search off with more than one worker are not
reproducible; they are run --nondeterministic-repeats times and the spread is reported.

For comparing arbitrary generate_ssa_kernel.py options on one instance with a wall-clock
limit, use scripts/run_ablation.py instead.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import schedule_report  # noqa: E402
from scripts.run_ablation import _parse_validation, parse_arm  # noqa: E402

DEFAULT_INSTANCES = "2x256,3x256,4x64,4x256,6x64"
DEFAULT_BUDGET = 20.0

# One switch per configuration, baseline first. "best" combinations belong on the command
# line (--config), since which switches combine well is what a study finds out.
STANDARD_CONFIGS = [
    ("baseline", ""),
    # 1. Search mode / workers
    ("workers-16", "--cpsat-workers 16"),
    ("no-interleave", "--no-cpsat-interleave"),
    ("no-interleave-w12", "--no-cpsat-interleave --cpsat-workers 12"),
    ("no-interleave-w12-lb",
     "--no-cpsat-interleave --cpsat-workers 12 --cpsat-extra-subsolvers objective_lb_search,objective_shaving"),
    # 2. Objective
    ("makespan-only", "--cpsat-objective makespan"),
    # 3. Greedy ALAP cap and horizon
    ("alap-slack-5", "--cpsat-greedy-alap-slack 5"),
    ("alap-slack-50", "--cpsat-greedy-alap-slack 50"),
    ("alap-slack-none", "--cpsat-greedy-alap-slack none"),
    ("tight-horizon", "--no-cpsat-horizon-num-ops --no-cpsat-horizon-max-latency"),
    ("alap-slack-none-tight-horizon",
     "--cpsat-greedy-alap-slack none --no-cpsat-horizon-num-ops --no-cpsat-horizon-max-latency"),
    # 4. Makespan interval
    ("makespan-interval", "--cpsat-makespan-interval"),
    ("push-toward-start", "--cpsat-push-tasks-toward-start"),
    # 5. Cumulative propagators
    ("propagators-all", "--cpsat-cumulative-propagators all --cpsat-ttef-max-intervals 1000000"),
    # 6. Redundant rows
    ("no-redundant-rows", "--disable-cpsat-constraints conditional_makespan,flow_cumulative"),
]


# ----------------------------------------------------------------------------
# Instances and configurations
# ----------------------------------------------------------------------------

def parse_instance(text: str, default_budget: float) -> dict:
    """'4x64' or '16x256@60' -> {'rounds': 4, 'batch_size': 64, 'budget': 20.0, 'name': '4r_b64'}."""
    match = re.fullmatch(r"(\d+)x(\d+)(?:@([\d.]+))?", text.strip())
    if not match:
        raise argparse.ArgumentTypeError(f"instance must be ROUNDSxBATCH[@BUDGET], got {text!r}")
    rounds, batch = int(match.group(1)), int(match.group(2))
    if batch % 8 != 0 or batch == 0:
        raise argparse.ArgumentTypeError(f"batch size must be a positive multiple of 8, got {batch}")
    budget = float(match.group(3)) if match.group(3) else default_budget
    return {"rounds": rounds, "batch_size": batch, "budget": budget, "name": f"{rounds}r_b{batch}"}


def is_nondeterministic(options: str) -> bool:
    """Parallel CP-SAT without interleaved search depends on thread timing (one worker does not)."""
    args = shlex.split(options)
    if "--no-cpsat-interleave" not in args:
        return False
    if "--cpsat-workers" in args:
        return args[args.index("--cpsat-workers") + 1] != "1"
    return True


def safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


# ----------------------------------------------------------------------------
# One run
# ----------------------------------------------------------------------------

def build_command(instance: dict, options: str, time_limit: float, paths: dict) -> list:
    return [
        sys.executable, str(ROOT / "generate_ssa_kernel.py"),
        "--rounds", str(instance["rounds"]), "--batch-size", str(instance["batch_size"]),
        "--cpsat-deterministic-time", str(instance["budget"]), "--cpsat-time-limit", str(time_limit),
        "--validate", "--output-json", str(paths["schedule"]), "--save-kernel", str(paths["kernel"]),
        *shlex.split(options),
    ]


def last_solve(log: str) -> str:
    """The part of the log that belongs to the last CP-SAT solve (the full-kernel one)."""
    start = log.rfind("Starting CP-SAT solver")
    return log[start:] if start >= 0 else ""


def parse_response(solve_log: str) -> dict:
    """Fields of CP-SAT's 'CpSolverResponse summary' block."""
    out = {}
    block = solve_log[solve_log.rfind("CpSolverResponse summary"):] if "CpSolverResponse summary" in solve_log else ""
    for key, cast in [("status", str), ("objective", float), ("best_bound", float), ("walltime", float),
                      ("deterministic_time", float), ("conflicts", int), ("branches", int)]:
        match = re.search(rf"^{key}: (\S+)$", block, re.MULTILINE)
        if match:
            try:
                out[key] = cast(match.group(1))
            except ValueError:
                pass
    return out


def parse_incumbents(solve_log: str, scale: int) -> list:
    """[(wall seconds, makespan)] each time the makespan of the incumbent improved.

    Solution lines look like '#3   5.88s best:1.93013167e+09 next:[...] rnd_var_lns'. The
    objective is scale * makespan + tie-breaker (scale 1 when the objective is the makespan).
    """
    trace = []
    for match in re.finditer(r"^#\d+\s+([\d.]+)s\s+best:([\d.e+]+)", solve_log, re.MULTILINE):
        makespan = int(float(match.group(2)) // scale)
        if not trace or makespan < trace[-1][1]:
            trace.append((float(match.group(1)), makespan))
    return trace


def parse_last_bound_time(solve_log: str) -> float | None:
    times = re.findall(r"^#Bound\s+([\d.]+)s", solve_log, re.MULTILINE)
    return float(times[-1]) if times else None


def parse_subsolvers(solve_log: str) -> list:
    match = re.search(r"^\d+ full problem subsolvers: \[(.*)\]$", solve_log, re.MULTILINE)
    return [name.strip() for name in match.group(1).split(",")] if match else []


def analyse_run(log: str, schedule_path: Path, returncode: int, elapsed: float) -> dict:
    """Everything the tables need from one finished run."""
    solve_log = last_solve(log)
    response = parse_response(solve_log)
    # Printed after a solve whose objective has the tie-breaker; absent for makespan-only.
    scales = re.findall(r"Objective:\s*scale=(\d+)", solve_log)
    scale = int(scales[-1]) if scales else 1
    incumbents = parse_incumbents(solve_log, scale)
    result = {
        "returncode": returncode,
        "elapsed_secs": round(elapsed, 1),
        "status": response.get("status"),
        "solver_walltime": response.get("walltime"),
        "deterministic_time": response.get("deterministic_time"),
        "objective_scale": scale,
        "incumbents": incumbents,
        "last_bound_improvement_wall": parse_last_bound_time(solve_log),
        "full_subsolvers": parse_subsolvers(solve_log),
        "wall_clock_limit_hit": "WARNING: CP-SAT stopped on the wall-clock limit" in log,
        "proved": "PROVED - every store matches the SSA program" in log,
        "passed": _parse_validation(log) is True,
        "hint_complete": "The solution hint is complete and is feasible" in solve_log,
    }
    greedy = re.search(r"\(greedy was (\d+)\)", log)
    result["greedy_cycles"] = int(greedy.group(1)) if greedy else None
    if schedule_path.exists() and returncode == 0:
        with open(schedule_path) as f:
            data = json.load(f)
        report = schedule_report.analyze(data)
        result.update({
            "cycles": report["run"]["cycles"],
            "lb_cycles": data.get("lb_cycles"),
            "capacity_floor": report["floor"]["cycles"],
            "peak_live_words": data.get("peak_live_words"),
            "valu_shifted": data.get("valu_shifted"),
            "cpsat_tuning": (data.get("run_options") or {}).get("cpsat_tuning"),
            "disabled_cpsat_constraints": (data.get("run_options") or {}).get("disabled_cpsat_constraints"),
        })
    # When the final makespan was first reached. Wall seconds come from the log; the solver
    # does not print deterministic time per solution, so that figure is an estimate that
    # assumes deterministic time advanced evenly over the solve.
    if incumbents and result.get("cycles") is not None:
        reached = [t for t, makespan in incumbents if makespan <= result["cycles"]]
        wall = min(reached) if reached else incumbents[-1][0]
        result["final_incumbent_wall"] = wall
        total_wall, total_dtime = response.get("walltime"), response.get("deterministic_time")
        if total_wall and total_dtime:
            result["final_incumbent_dtime_estimate"] = round(total_dtime * min(1.0, wall / total_wall), 1)
    return result


def run_cell(instance: dict, config_name: str, options: str, repeat: int, out_dir: Path,
             time_limit: float, force: bool) -> dict:
    """Run one (instance, configuration, repeat), or reuse its saved result."""
    stem = f"{instance['name']}__{safe(config_name)}" + (f"__run{repeat}" if repeat else "")
    paths = {
        "schedule": out_dir / f"{stem}.schedule.json",
        "kernel": out_dir / f"{stem}.kernel.json",
        "log": out_dir / f"{stem}.log",
        "result": out_dir / f"{stem}.result.json",
    }
    key = hashlib.sha256(json.dumps([instance, options], sort_keys=True).encode()).hexdigest()[:16]
    if paths["result"].exists() and not force:
        with open(paths["result"]) as f:
            saved = json.load(f)
        if saved.get("key") == key:
            return saved
    cmd = build_command(instance, options, time_limit, paths)
    start = time.time()
    with open(paths["log"], "w") as log_file:
        log_file.write("Command: " + " ".join(shlex.quote(c) for c in cmd) + "\n\n")
        log_file.flush()
        proc = subprocess.run(cmd, cwd=ROOT, stdout=log_file, stderr=subprocess.STDOUT, text=True)
    elapsed = time.time() - start
    log = paths["log"].read_text(errors="replace")
    result = {
        "key": key,
        "instance": instance["name"],
        "rounds": instance["rounds"],
        "batch_size": instance["batch_size"],
        "budget": instance["budget"],
        "config": config_name,
        "options": options,
        "repeat": repeat,
        "deterministic": not is_nondeterministic(options),
        "log": paths["log"].name,
        "schedule": paths["schedule"].name,
    }
    result.update(analyse_run(log, paths["schedule"], proc.returncode, elapsed))
    with open(paths["result"], "w") as f:
        json.dump(result, f, indent=1)
    return result


# ----------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------

def _spread(values: list, fmt: str = "{}", median: bool = False) -> str:
    """One value, or 'min..max' ('min / median / max' with median=True) over repeated runs."""
    values = [v for v in values if v is not None]
    if not values:
        return "-"
    if len(set(values)) == 1:
        return fmt.format(values[0])
    if median:
        return f"{fmt.format(min(values))} / {fmt.format(statistics.median(values))} / {fmt.format(max(values))}"
    return f"{fmt.format(min(values))}..{fmt.format(max(values))}"


def summarise(cells: list) -> dict:
    """Collapse the repeats of one (instance, configuration) into one table row."""
    ok = [c for c in cells if c.get("cycles") is not None]
    first = cells[0]
    return {
        "config": first["config"],
        "options": first["options"],
        "runs": len(cells),
        "deterministic": first["deterministic"],
        "cycles": [c.get("cycles") for c in ok],
        "best_cycles": min((c["cycles"] for c in ok), default=None),
        "status": sorted({c.get("status") or "?" for c in cells}),
        "lb_cycles": [c.get("lb_cycles") for c in ok],
        "capacity_floor": [c.get("capacity_floor") for c in ok],
        "greedy_cycles": first.get("greedy_cycles"),
        "final_incumbent_dtime_estimate": [c.get("final_incumbent_dtime_estimate") for c in ok],
        "final_incumbent_wall": [c.get("final_incumbent_wall") for c in ok],
        "deterministic_time": [c.get("deterministic_time") for c in cells],
        "solver_walltime": [c.get("solver_walltime") for c in cells],
        "peak_live_words": [c.get("peak_live_words") for c in ok],
        "all_valid": bool(ok) and len(ok) == len(cells) and all(c["proved"] and c["passed"] for c in cells),
        "failed_runs": len(cells) - len(ok),
        "wall_clock_limit_hit": any(c.get("wall_clock_limit_hit") for c in cells),
    }


def summary_matrix(by_cell: dict, instances: list, configs: list) -> list:
    """Cycles of every configuration on every instance, as Markdown lines (one row per configuration)."""
    present = [i for i in instances if any((i["name"], name) in by_cell for name, _ in configs)]
    header = [f"{i['rounds']}r / b{i['batch_size']} (budget {i['budget']:g})" for i in present]
    lines = ["| Configuration | " + " | ".join(header) + " |", "|---|" + "---|" * len(present)]
    greedy, floor = [], []
    for instance in present:
        cells = [c for name, _ in configs for c in by_cell.get((instance["name"], name), [])]
        greedy.append(next((str(c["greedy_cycles"]) for c in cells if c.get("greedy_cycles")), "-"))
        floor.append(_spread([c.get("capacity_floor") for c in cells]))
    lines.append("| *greedy hint* | " + " | ".join(greedy) + " |")
    for name, options in configs:
        if not any((i["name"], name) in by_cell for i in present):
            continue
        row = []
        for instance in present:
            cells = by_cell.get((instance["name"], name))
            if not cells:
                row.append("-")
                continue
            summary = summarise(cells)
            text = _spread(summary["cycles"], median=True)
            if "OPTIMAL" in summary["status"]:
                text += " (optimal)" if summary["status"] == ["OPTIMAL"] else " (some optimal)"
            if not summary["all_valid"]:
                text += " INVALID"
            row.append(text)
        label = name + ("" if not is_nondeterministic(options) else " (not reproducible)")
        lines.append(f"| {label} | " + " | ".join(row) + " |")
    lines.append("| *capacity floor* | " + " | ".join(floor) + " |")
    return lines


def markdown(results: list, instances: list, configs: list) -> str:
    """A summary matrix, then one table per instance; rows in configuration order, baseline first."""
    by_cell: dict = {}
    for cell in results:
        by_cell.setdefault((cell["instance"], cell["config"]), []).append(cell)
    lines = ["### Cycles by configuration and instance", ""]
    lines += summary_matrix(by_cell, instances, configs)
    lines += ["", "Non-reproducible configurations show min / median / max over their runs. \"optimal\" is "
              "optimal for the emitted ops inside the model's domains, not for the problem.", ""]
    for instance in instances:
        rows = [summarise(by_cell[(instance["name"], name)]) for name, _ in configs
                if (instance["name"], name) in by_cell]
        if not rows:
            continue
        baseline = rows[0]["best_cycles"]
        greedy = next((r["greedy_cycles"] for r in rows if r["greedy_cycles"]), None)
        lines.append(f"### {instance['rounds']} rounds, batch {instance['batch_size']} "
                     f"(budget {instance['budget']:g} deterministic time; greedy hint {greedy} cycles)")
        lines.append("")
        lines.append("| Configuration | Cycles | vs baseline | Status | Bound (lb_cycles) | Floor | "
                     "Final incumbent at: dtime (est) | wall s | Solver dtime | Solver wall s | Peak live | Valid |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for row in rows:
            cycles = _spread(row["cycles"], median=True)
            if row["runs"] > 1:
                cycles += f" (min / median / max of {row['runs']} runs)" if len(set(row["cycles"])) > 1 \
                    else f" (all {row['runs']} runs)"
            delta = "-"
            if baseline is not None and row["cycles"]:
                deltas = [c - baseline for c in row["cycles"]]
                delta = "0" if set(deltas) == {0} else _spread(deltas, "{:+d}")
            valid = "PROVED, PASSED" if row["all_valid"] else f"NO ({row['failed_runs']} failed)"
            notes = " (wall-clock limit hit: invalid)" if row["wall_clock_limit_hit"] else ""
            name = row["config"] + ("" if row["deterministic"] else " (not reproducible)")
            lines.append(
                f"| {name} | {cycles} | {delta} | {', '.join(row['status'])} | {_spread(row['lb_cycles'])} "
                f"| {_spread(row['capacity_floor'])} "
                f"| {_spread(row['final_incumbent_dtime_estimate'], '{:.1f}')} "
                f"| {_spread(row['final_incumbent_wall'], '{:.1f}')} "
                f"| {_spread(row['deterministic_time'], '{:.1f}')} | {_spread(row['solver_walltime'], '{:.1f}')} "
                f"| {_spread(row['peak_live_words'])} | {valid}{notes} |")
        lines.append("")
    lines.append("Configurations:")
    lines.append("")
    for name, options in configs:
        lines.append(f"- `{name}`: `{options or '(defaults)'}`")
    lines.append("")
    lines.append("Columns: *Cycles* is the schedule length of the validated kernel. *Bound* is the solver's "
                 "lower bound for the ops this run emitted, inside the model's domains; it is not a bound on "
                 "the problem. *Floor* is the capacity floor of scripts.schedule_report for the emitted ops. "
                 "*Final incumbent at* is when the final makespan was first reached: wall seconds from the "
                 "solver log, and a deterministic-time estimate (the log has no per-solution deterministic "
                 "time; the estimate assumes it advanced evenly). *Solver dtime* is the total the solver "
                 "reports, summed over workers, so it exceeds the per-worker budget.")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--instances", default=DEFAULT_INSTANCES,
                        help=f"Comma-separated ROUNDSxBATCH[@BUDGET] (default: {DEFAULT_INSTANCES})")
    parser.add_argument("--budget", type=float, default=DEFAULT_BUDGET,
                        help=f"Deterministic-time budget for instances without their own (default: {DEFAULT_BUDGET:g})")
    parser.add_argument("--config", action="append", default=[], metavar="NAME:OPTIONS",
                        help="Configuration as 'name:generate_ssa_kernel.py options'. Repeatable; the first is the "
                             "baseline. Without any, the standard list is run (--list-configs shows it).")
    parser.add_argument("--add-config", action="append", default=[], metavar="NAME:OPTIONS",
                        help="Add a configuration to the standard list (e.g. a combination)")
    parser.add_argument("--only", default=None, metavar="NAME,...",
                        help="Run only these configurations of the list (the baseline is always kept)")
    parser.add_argument("--shared-args", default="", help="Options added to every configuration")
    parser.add_argument("--time-limit", type=float, default=3600.0,
                        help="Wall-clock backstop per run in seconds (default: 3600). A run that ends on it is "
                             "flagged invalid.")
    parser.add_argument("--nondeterministic-repeats", type=int, default=3,
                        help="Runs per non-reproducible configuration (default: 3)")
    parser.add_argument("--jobs", type=int, default=1,
                        help="Runs in parallel (default: 1). Deterministic-time budgets make reproducible "
                             "configurations independent of load; non-reproducible ones are not, so they always "
                             "run one at a time, after the others.")
    parser.add_argument("--out-dir", default="build/cpsat_ablation",
                        help="Directory for logs, schedules and results (default: build/cpsat_ablation, git-ignored)")
    parser.add_argument("--name", default="cpsat_ablation", help="Basename of the result files (.json, .md)")
    parser.add_argument("--force", action="store_true", help="Rerun cells that already have a saved result")
    parser.add_argument("--report-only", action="store_true", help="Only rebuild the tables from saved results")
    parser.add_argument("--list-configs", action="store_true", help="Print the configurations and exit")
    args = parser.parse_args(argv)

    configs = [parse_arm(text) for text in args.config] if args.config else list(STANDARD_CONFIGS)
    configs += [parse_arm(text) for text in args.add_config]
    names = [name for name, _ in configs]
    if len(set(names)) != len(names):
        parser.error("configuration names must be unique")
    if args.only:
        wanted = {name.strip() for name in args.only.split(",")}
        unknown = wanted - set(names)
        if unknown:
            parser.error(f"--only names unknown configurations: {sorted(unknown)}")
        configs = [c for i, c in enumerate(configs) if i == 0 or c[0] in wanted]
    if args.shared_args:
        configs = [(name, f"{args.shared_args} {options}".strip()) for name, options in configs]
    if args.list_configs:
        for name, options in configs:
            print(f"{name}: {options or '(defaults)'}")
        return 0
    try:
        instances = [parse_instance(text, args.budget) for text in args.instances.split(",") if text.strip()]
    except argparse.ArgumentTypeError as err:
        parser.error(str(err))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cells = []
    for instance in instances:
        for name, options in configs:
            repeats = args.nondeterministic_repeats if is_nondeterministic(options) else 1
            for repeat in range(repeats):
                cells.append((instance, name, options, repeat))

    results = []
    if args.report_only:
        for instance, name, options, repeat in cells:
            stem = f"{instance['name']}__{safe(name)}" + (f"__run{repeat}" if repeat else "")
            path = out_dir / f"{stem}.result.json"
            if path.exists():
                with open(path) as f:
                    results.append(json.load(f))
    else:
        def run(cell):
            instance, name, options, repeat = cell
            result = run_cell(instance, name, options, repeat, out_dir, args.time_limit, args.force)
            valid = "ok" if result.get("proved") and result.get("passed") else "NOT VALIDATED"
            print(f"{result['instance']:>9} {name:<32} run {repeat}: cycles={result.get('cycles')} "
                  f"status={result.get('status')} lb={result.get('lb_cycles')} "
                  f"dtime={result.get('deterministic_time')} wall={result.get('solver_walltime')} {valid}",
                  flush=True)
            return result

        reproducible = [c for c in cells if not is_nondeterministic(c[2])]
        timing_dependent = [c for c in cells if is_nondeterministic(c[2])]
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
            results.extend(pool.map(run, reproducible))
        results.extend(run(cell) for cell in timing_dependent)

    order = {(instance["name"], name, repeat): i for i, (instance, name, _, repeat) in enumerate(cells)}
    results.sort(key=lambda r: order.get((r["instance"], r["config"], r["repeat"]), len(order)))
    json_path, md_path = out_dir / f"{args.name}.json", out_dir / f"{args.name}.md"
    with open(json_path, "w") as f:
        json.dump({"instances": instances, "configs": [{"name": n, "options": o} for n, o in configs],
                   "results": results}, f, indent=1)
    table = markdown(results, instances, configs)
    md_path.write_text(table)
    print()
    print(table)
    print(f"Wrote {json_path} and {md_path}")
    invalid = [r for r in results if not (r.get("proved") and r.get("passed"))]
    if invalid:
        print(f"{len(invalid)} run(s) did not produce a validated kernel:")
        for r in invalid:
            print(f"  {r['instance']} {r['config']} run {r['repeat']}: see {out_dir / r['log']}")
    return 1 if invalid else 0


if __name__ == "__main__":
    sys.exit(main())
