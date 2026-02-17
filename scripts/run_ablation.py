#!/usr/bin/env python3
"""General-purpose ablation runner for generate_ssa_kernel.py options.

Compare any set of generate_ssa_kernel.py configurations by specifying
shared options and per-arm differences.

Examples:
    # Tree-reduce depth 3 vs 4
    uv run python scripts/run_ablation.py \\
      --rounds 5 --time-limit 600 \\
      --arm "depth-3:--tree-reduce-max-depth 3" \\
      --arm "depth-4:--tree-reduce-max-depth 4"

    # Const choice comparison
    uv run python scripts/run_ablation.py \\
      --rounds 4 --time-limit 600 \\
      --arm "optimize:--const-choice optimize" \\
      --arm "add_imm:--const-choice add_imm" \\
      --arm "load:--const-choice load"

    # VALU shift on vs off
    uv run python scripts/run_ablation.py \\
      --rounds 4 --time-limit 600 \\
      --arm "valu-shift:--enable-valu-shift" \\
      --arm "no-valu-shift:--no-valu-shift"

Output: ablation_general_results.json (configurable with --output)
"""

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path


def parse_arm(arm_str):
    """Parse an arm specification 'name:options'.

    Splits on the first colon to separate the arm name from extra CLI args.
    If no colon is present, the entire string is treated as the arm name
    with no extra args.
    """
    if ':' in arm_str:
        name, extra = arm_str.split(':', 1)
        return name.strip(), extra.strip()
    return arm_str.strip(), ""


def run_arm(arm_name, extra_args_str, rounds, batch_size, time_limit, shared_args_str, quiet=False):
    """Run a single ablation arm.

    Constructs and runs a generate_ssa_kernel.py command from shared options
    plus arm-specific options. By default streams output to both stdout and a
    log file; use quiet=True to only write to the log file.

    Returns a result dict with makespan, timing, solver stats, and the
    loaded scheduler output JSON.
    """
    # Sanitize arm name for filenames
    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', arm_name.lower())
    output_json = f"scheduler_output_{safe_name}.json"
    log_file = f"ablation_log_{safe_name}.txt"

    print(f"\n{'='*80}")
    print(f"Running arm: {arm_name}")
    print(f"  Extra args: {extra_args_str or '(none)'}")
    print(f"  Rounds: {rounds}, Batch size: {batch_size}")
    print(f"  Time limit: {time_limit}s")
    print(f"{'='*80}\n")

    # Build command
    cmd = [
        sys.executable, "generate_ssa_kernel.py",
        "--rounds", str(rounds),
        "--batch-size", str(batch_size),
        "--cpsat-time-limit", str(time_limit),
        "--output-json", output_json,
        "--comment", f"ablation: {arm_name}",
    ]

    # Add shared args
    if shared_args_str:
        cmd.extend(shlex.split(shared_args_str))

    # Add arm-specific args
    if extra_args_str:
        cmd.extend(shlex.split(extra_args_str))

    # Run and stream output
    start_time = time.time()
    with open(log_file, 'w') as log_f:
        log_f.write(f"{'='*80}\n")
        log_f.write(f"Arm: {arm_name}\n")
        log_f.write(f"Command: {' '.join(cmd)}\n")
        log_f.write(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n")
        log_f.write(f"{'='*80}\n\n")
        log_f.flush()

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        lines = []
        for line in proc.stdout or []:
            lines.append(line)
            log_f.write(line)
            log_f.flush()
            if not quiet:
                print(line, end='')
        proc.wait()
        stdout_output = ''.join(lines)

        log_f.write(f"\n\n{'='*80}\n")
        log_f.write(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
        log_f.write(f"Return code: {proc.returncode}\n")
        log_f.write(f"{'='*80}\n")
        log_f.flush()

    elapsed_time = time.time() - start_time

    # Parse solver status from output
    solver_status = _parse_solver_status(stdout_output)

    # On failure, still record result
    if proc.returncode != 0:
        print(f"Command failed with return code {proc.returncode}")
        print(f"  Solver status: {solver_status or 'not parsed'}")
        return _make_result(
            arm_name, extra_args_str, rounds, batch_size,
            makespan=None,
            elapsed_time=elapsed_time,
            solver_status=solver_status or 'UNKNOWN',
            return_code=proc.returncode,
            output_json=output_json,
            log_file=log_file,
            lb_cycles=None,
            optimization_milestones=[],
        )

    # Parse makespan
    makespan = _parse_makespan(stdout_output)

    # Load scheduler output JSON
    json_data = None
    if Path(output_json).exists():
        try:
            with open(output_json, 'r') as f:
                json_data = json.load(f)
        except Exception as e:
            print(f"Warning: Could not parse JSON: {e}")

    # Parse solver statistics from output and JSON
    solver_stats = _parse_solver_stats(stdout_output, json_data)

    # Parse lower bound and optimization milestones
    lb_cycles = _parse_lower_bound(json_data)
    scale = _parse_objective_scale(stdout_output)
    optimization_milestones = _parse_optimization_milestones(stdout_output, scale=scale)

    # Use the more detailed solver_status from solver_stats if available
    final_status = solver_stats.pop('solver_status', None) or solver_status

    return _make_result(
        arm_name, extra_args_str, rounds, batch_size,
        makespan=makespan,
        elapsed_time=elapsed_time,
        solver_status=final_status,
        return_code=proc.returncode,
        output_json=output_json,
        log_file=log_file,
        json_data=json_data,
        lb_cycles=lb_cycles,
        optimization_milestones=optimization_milestones,
        **solver_stats,
    )


def _parse_solver_status(stdout_output):
    """Parse solver status from stdout."""
    solver_status = None
    for line in stdout_output.split('\n'):
        line_lower = line.lower()
        if 'status:' in line_lower:
            if 'optimal' in line_lower:
                return 'OPTIMAL'
            if 'feasible' in line_lower:
                return 'FEASIBLE'
            if 'infeasible' in line_lower:
                return 'INFEASIBLE'
            if 'unknown' in line_lower:
                return 'UNKNOWN'
        if 'no solution found' in line_lower and 'status' in line_lower:
            if 'infeasible' in line_lower:
                return 'INFEASIBLE'
            if 'unknown' in line_lower:
                return 'UNKNOWN'
    return solver_status


def _parse_makespan(stdout_output):
    """Parse makespan (cycle count) from stdout."""
    makespan = None
    for line in stdout_output.split('\n'):
        if 'Cycles:' in line and 'Results:' not in line:
            try:
                parts = line.split('Cycles:')
                if len(parts) > 1:
                    makespan = int(parts[1].strip().split()[0])
            except (ValueError, IndexError):
                pass
        if makespan is None and '  Cycles:' in line:
            try:
                makespan = int(line.split('Cycles:')[1].strip().split()[0])
            except (ValueError, IndexError):
                pass
    return makespan


def _parse_objective_scale(stdout_output):
    """Parse CP-SAT objective scale from log. floor(obj/scale) = makespan cycles."""
    match = re.search(r'Objective:\s*scale=(\d+)', stdout_output, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 1_000_000  # default for typical larger problems


def _parse_lower_bound(json_data):
    """Extract proven lower bound on cycles from scheduler output JSON."""
    if not json_data:
        return None
    return json_data.get('lb_cycles')


def _parse_optimization_milestones(stdout_output, scale=None):
    """Parse CP-SAT progress log for time-to-cycle milestones.

    CP-SAT logs have lines like:
      #1       0.77s best:96026397 next:[...] quick_restart_no_lp [hint]
      #2       4.22s best:96026185 next:[...] rnd_var_lns

    Objective is scaled (scale from log, or 1e6 default), so cycles = floor(obj / scale).
    Returns list of {'cycles': N, 'time_secs': T} for each cycle value first achieved,
    in chronological order.
    """
    if scale is None:
        scale = _parse_objective_scale(stdout_output)
    seen_cycles = set()
    milestones = []
    for line in stdout_output.split('\n'):
        # Match solution improvement lines: #N TIME best:OBJ
        match = re.search(r'#\d+\s+([\d.]+)s\s+best:(\d+)', line)
        if match:
            time_secs = float(match.group(1))
            obj = int(match.group(2))
            cycles = obj // scale
            if cycles not in seen_cycles:
                seen_cycles.add(cycles)
                milestones.append({'cycles': cycles, 'time_secs': time_secs})
    return milestones


def _parse_solver_stats(stdout_output, json_data):
    """Parse detailed solver statistics from stdout and JSON."""
    time_to_feasibility = None
    time_to_optimum = None
    solver_status = None
    solve_time_secs = None
    optimality_gap = None
    solver_iterations = None
    solution_count = 0
    first_feasible_time = None

    for line in stdout_output.split('\n'):
        line_lower = line.lower()

        # Parse "Solution #X found at time Y"
        solution_match = re.search(r'solution\s+#(\d+)\s+found\s+at\s+time\s+([\d.]+)', line_lower)
        if solution_match:
            solution_count = int(solution_match.group(1))
            solution_time = float(solution_match.group(2))
            if first_feasible_time is None:
                first_feasible_time = solution_time

        # Parse "best: X"
        if 'best:' in line_lower:
            try:
                parts = line.split('best:')
                if len(parts) > 1:
                    best_val = parts[1].strip().split()[0]
                    if best_val.lower() not in ('inf', '-', 'infinity'):
                        if first_feasible_time is None:
                            time_match = re.search(r'time[:\s]+([\d.]+)', line_lower)
                            if time_match:
                                first_feasible_time = float(time_match.group(1))
            except (ValueError, IndexError):
                pass

        # Parse solver status
        if 'status:' in line_lower or ('cp-sat' in line_lower and 'status' in line_lower):
            if 'optimal' in line_lower:
                solver_status = 'OPTIMAL'
            elif 'feasible' in line_lower:
                solver_status = 'FEASIBLE'
            elif 'infeasible' in line_lower:
                solver_status = 'INFEASIBLE'
            elif 'unknown' in line_lower:
                solver_status = 'UNKNOWN'

        # Parse "SSA CP-SAT result" line
        if 'ssa cp-sat result' in line_lower:
            status_match = re.search(r'status\s+(\w+)', line_lower)
            if status_match:
                solver_status = status_match.group(1).upper()
            time_match = re.search(r'time\s+([\d.]+)s', line_lower)
            if time_match:
                solve_time_secs = float(time_match.group(1))
            gap_match = re.search(r'gap\s+([\d.]+)%', line_lower)
            if gap_match:
                optimality_gap = float(gap_match.group(1)) / 100.0

        # Parse solver iterations
        if 'branches:' in line_lower or 'conflicts:' in line_lower:
            branches_match = re.search(r'branches[:\s]+(\d+)', line_lower)
            if branches_match:
                solver_iterations = int(branches_match.group(1))

    # Supplement from JSON
    if json_data and 'stats' in json_data:
        stats = json_data['stats']
        if solve_time_secs is None:
            solve_time_secs = stats.get('solve_time_secs')
        if optimality_gap is None:
            optimality_gap = stats.get('gap')

    # Compute derived timing
    if first_feasible_time is not None:
        time_to_feasibility = first_feasible_time
    elif solve_time_secs is not None and solver_status in ('OPTIMAL', 'FEASIBLE'):
        time_to_feasibility = solve_time_secs

    if solver_status == 'OPTIMAL' and solve_time_secs is not None:
        time_to_optimum = solve_time_secs

    return {
        'solve_time_secs': solve_time_secs,
        'time_to_feasibility': time_to_feasibility,
        'time_to_optimum': time_to_optimum,
        'solver_status': solver_status,
        'optimality_gap': optimality_gap,
        'solver_iterations': solver_iterations,
        'solution_count': solution_count,
    }


def _make_result(arm_name, extra_args, rounds, batch_size, *, makespan,
                 elapsed_time, solver_status, return_code, output_json,
                 log_file, json_data=None, solve_time_secs=None,
                 time_to_feasibility=None, time_to_optimum=None,
                 optimality_gap=None, solver_iterations=None,
                 solution_count=0, lb_cycles=None, optimization_milestones=None):
    """Build a result dict for one arm."""
    return {
        'arm_name': arm_name,
        'extra_args': extra_args,
        'rounds': rounds,
        'batch_size': batch_size,
        'makespan': makespan,
        'elapsed_time': elapsed_time,
        'solve_time_secs': solve_time_secs,
        'time_to_feasibility': time_to_feasibility,
        'time_to_optimum': time_to_optimum,
        'solver_status': solver_status,
        'optimality_gap': optimality_gap,
        'solver_iterations': solver_iterations,
        'solution_count': solution_count,
        'return_code': return_code,
        'output_json': output_json,
        'log_file': log_file,
        'json_data': json_data,
        'lb_cycles': lb_cycles,
        'optimization_milestones': optimization_milestones or [],
    }


def _do_reanalyze(path: str) -> int:
    """Re-parse existing ablation logs/JSON; extract lb_cycles and milestones; save in-place."""
    p = Path(path)
    if not p.exists():
        print(f"ERROR: --reanalyze path not found: {path}")
        return 1

    with open(p, 'r') as f:
        data = json.load(f)

    results = data.get('results', [])
    if not results:
        print("ERROR: No results in ablation file")
        return 1

    for r in results:
        log_file = r.get('log_file')
        output_json = r.get('output_json')
        if log_file and Path(log_file).exists():
            with open(log_file, 'r') as f:
                log_content = f.read()
            scale = _parse_objective_scale(log_content)
            r['optimization_milestones'] = _parse_optimization_milestones(log_content, scale=scale)
        else:
            r['optimization_milestones'] = r.get('optimization_milestones') or []

        if output_json and Path(output_json).exists():
            try:
                with open(output_json, 'r') as f:
                    sched = json.load(f)
                r['lb_cycles'] = _parse_lower_bound(sched)
            except Exception as e:
                print(f"Warning: Could not parse {output_json}: {e}")
                r['lb_cycles'] = r.get('lb_cycles')

    print_summary(results)
    with open(p, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to: {p}")
    return 0


def print_summary(results):
    """Print a quick summary table after all arms complete."""
    print(f"\n{'='*80}")
    print("ABLATION RESULTS SUMMARY")
    print(f"{'='*80}")

    baseline = results[0]

    for i, r in enumerate(results):
        label = "BASELINE" if i == 0 else f"arm {i}"
        status = r.get('solver_status', '?')
        makespan = r.get('makespan')

        print(f"\n  [{label}] {r['arm_name']}")
        if r['extra_args']:
            print(f"    Options: {r['extra_args']}")
        if makespan is not None:
            print(f"    Makespan: {makespan} cycles")
        else:
            print(f"    Makespan: -- ({status})")

        if r.get('elapsed_time'):
            print(f"    Elapsed: {r['elapsed_time']:.1f}s")
        if r.get('solve_time_secs'):
            print(f"    Solve time: {r['solve_time_secs']:.2f}s")
        if r.get('solver_status'):
            print(f"    Status: {r['solver_status']}")
        if r.get('optimality_gap') is not None:
            print(f"    Gap: {r['optimality_gap']:.4f}")
        if r.get('lb_cycles') is not None:
            print(f"    Lower bound: {r['lb_cycles']} cycles")
        if r.get('solver_iterations'):
            print(f"    Branches: {r['solver_iterations']}")
        if r.get('solution_count'):
            print(f"    Solutions: {r['solution_count']}")

        # Optimization milestones: "reached X cycles after Ys"
        milestones = r.get('optimization_milestones') or []
        if milestones:
            ms_strs = [f"{m['cycles']}@{m['time_secs']:.1f}s" for m in milestones]
            if len(ms_strs) <= 10:
                print(f"    Milestones: {' → '.join(ms_strs)}")
            else:
                head = ms_strs[:3]
                tail = ms_strs[-3:]
                print(f"    Milestones: {' → '.join(head)} ... {' → '.join(tail)}")

        # Compare to baseline
        if i > 0 and baseline.get('makespan') and makespan:
            diff = makespan - baseline['makespan']
            pct = (diff / baseline['makespan']) * 100
            print(f"    vs baseline: {diff:+d} cycles ({pct:+.1f}%)")

        if i > 0 and baseline.get('time_to_feasibility') and r.get('time_to_feasibility'):
            tdiff = r['time_to_feasibility'] - baseline['time_to_feasibility']
            print(f"    Feasibility vs baseline: {tdiff:+.2f}s")


def main():
    parser = argparse.ArgumentParser(
        description="General-purpose ablation runner for generate_ssa_kernel.py options.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Tree-reduce depth comparison
  %(prog)s --rounds 5 --time-limit 600 \\
    --arm "depth-3:--tree-reduce-max-depth 3" \\
    --arm "depth-4:--tree-reduce-max-depth 4"

  # VALU shift on vs off
  %(prog)s --rounds 4 --time-limit 600 \\
    --arm "valu-shift:--enable-valu-shift" \\
    --arm "no-valu-shift:--no-valu-shift"

  # With shared args applied to all arms
  %(prog)s --rounds 4 --time-limit 600 \\
    --shared-args "--no-valu-shift" \\
    --arm "optimize:--const-choice optimize" \\
    --arm "load:--const-choice load"

  # Re-parse existing ablation logs (extract lb_cycles, milestones) without re-running
  %(prog)s --reanalyze ablation_tightening_extensions_3r_64b_240s.json
""")

    parser.add_argument("--rounds", type=int, default=4,
                        help="Number of rounds (default: 4)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size (default: 256, must be multiple of 8)")
    parser.add_argument("--time-limit", type=float, default=600.0,
                        help="CP-SAT time limit per arm in seconds (default: 600)")
    parser.add_argument("--arm", action="append", default=None,
                        metavar='"name:options"',
                        help='Arm specification as "name:extra_args". '
                             'Repeat for each arm. First arm is treated as baseline. '
                             'Not required with --reanalyze.')
    parser.add_argument("--shared-args", type=str, default="",
                        help="Extra args applied to ALL arms (e.g. '--no-valu-shift')")
    parser.add_argument("--output", type=str, default="ablation_general_results.json",
                        help="Output JSON path (default: ablation_general_results.json)")
    parser.add_argument("--quiet", action="store_true",
                        help="Only write scheduler output to log files; do not print to terminal")
    parser.add_argument("--reanalyze", type=str, metavar="PATH",
                        help="Re-parse existing ablation logs/JSON; skip running. Loads PATH, "
                             "re-extracts lb_cycles and optimization_milestones, prints summary, saves.")

    args = parser.parse_args()

    # Reanalyze mode: load existing results, re-parse logs, update and save (in-place)
    if args.reanalyze:
        return _do_reanalyze(args.reanalyze)

    if args.batch_size % 8 != 0:
        print(f"ERROR: batch_size must be a multiple of 8 (got {args.batch_size})")
        return 1

    if not args.arm or len(args.arm) < 2:
        print("ERROR: need at least 2 arms for an ablation study")
        return 1

    # Parse arms
    arms = []
    for arm_str in args.arm:
        name, extra = parse_arm(arm_str)
        arms.append({'name': name, 'extra_args': extra})

    # Check for duplicate arm names
    names = [a['name'] for a in arms]
    if len(names) != len(set(names)):
        print("ERROR: duplicate arm names detected")
        return 1

    print("General-Purpose Ablation Study")
    print("=" * 80)
    print(f"Rounds: {args.rounds}, Batch size: {args.batch_size}")
    print(f"Time limit: {args.time_limit}s per arm")
    if args.shared_args:
        print(f"Shared args: {args.shared_args}")
    print(f"Arms ({len(arms)}):")
    for i, arm in enumerate(arms):
        label = " (baseline)" if i == 0 else ""
        print(f"  {i+1}. {arm['name']}{label}: {arm['extra_args'] or '(no extra args)'}")
    print()

    # Run each arm
    results = []
    for arm in arms:
        result = run_arm(
            arm['name'],
            arm['extra_args'],
            rounds=args.rounds,
            batch_size=args.batch_size,
            time_limit=args.time_limit,
            shared_args_str=args.shared_args,
            quiet=args.quiet,
        )
        results.append(result)

    # Print summary
    print_summary(results)

    # Save results
    output_data = {
        'config': {
            'rounds': args.rounds,
            'batch_size': args.batch_size,
            'time_limit': args.time_limit,
            'shared_args': args.shared_args,
            'arms': arms,
        },
        'results': results,
    }
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {args.output}")

    log_files = [r.get('log_file') for r in results if r.get('log_file')]
    if log_files:
        print(f"\nSolver logs:")
        for lf in log_files:
            print(f"  - {lf}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
