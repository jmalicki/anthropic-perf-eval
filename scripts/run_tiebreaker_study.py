#!/usr/bin/env python3
"""Run the experiments of docs/TIEBREAKER_OBJECTIVE_STUDY.md (why the tie-breaker costs cycles).

Three stages, each resumable (a run whose output file exists is skipped):

  export     build every instance once with a tiny budget to export its CP-SAT model
             (OUT/models/r<rounds>_b<batch>.pb and .pb.params)
  pipeline   full builds through generate_ssa_kernel.py with --validate, under each objective
             and several solver seeds: the effect on validated kernels, and the safety record
  variants   solve the exported models under objective variants with scripts/objective_variants.py
             (experiment sets: replicate, mechanism, dose, scale)

Every solve is budgeted in deterministic time; a run that hit the wall-clock backstop is
recorded as wall_limited and must not be used.

Usage:
  uv run python -m scripts.run_tiebreaker_study export   --out study_runs/tb
  uv run python -m scripts.run_tiebreaker_study pipeline --out study_runs/tb --seeds 1 2 3
  uv run python -m scripts.run_tiebreaker_study variants --out study_runs/tb --set replicate --seeds 1 2 3 4 5
  uv run python -m scripts.run_tiebreaker_study variants --out study_runs/tb --set mechanism --instances 4x64 6x64
"""

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (rounds, batch size). Model size grows with both.
INSTANCES = [(2, 16), (3, 16), (4, 16), (6, 16), (2, 64), (3, 64), (4, 64), (6, 64), (2, 256)]

# name -> (variant, mode, solver parameter overrides)
EXPERIMENT_SETS = {
    # The effect itself, with a distribution over seeds.
    "replicate": {
        "tiebreak": ("tiebreak", "single", []),
        "makespan": ("makespan", "single", []),
    },
    # One change at a time.
    "mechanism": {
        "tiebreak-nolns": ("tiebreak", "single", ["use_lns=false"]),
        "makespan-nolns": ("makespan", "single", ["use_lns=false"]),
        "tiebreak-nocore": ("tiebreak", "single", ["ignore_subsolvers=core"]),
        "makespan-nocore": ("makespan", "single", ["ignore_subsolvers=core"]),
        "tiebreak-destructive": ("tiebreak", "destructive", []),
        "makespan-destructive": ("makespan", "destructive", []),
        "twophase": ("tiebreak", "twophase", []),
        "late": ("late", "single", []),
        "weight1000": ("weight:1000", "single", []),
        "tiebreak-nogaplimit": ("tiebreak", "single", ["absolute_gap_limit=0"]),
    },
    # Dose-response: how many tie-breaker terms does it take?
    "dose": {
        "frac01": ("frac:1", "single", []),
        "frac10": ("frac:10", "single", []),
        "frac50": ("frac:50", "single", []),
    },
}


def instance_tag(rounds, batch):
    return f"r{rounds}_b{batch}"


def parse_instances(names):
    """['4x64', '2x256'] -> [(4, 64), (2, 256)]; None -> the default list."""
    if not names:
        return INSTANCES
    out = []
    for name in names:
        rounds, batch = name.lower().split("x")
        out.append((int(rounds), int(batch)))
    return out


def run_logged(cmd, log_path, env=None, timeout=None):
    """Run cmd with stdout+stderr to log_path; returns (exit code, seconds)."""
    started = time.time()
    with open(log_path, "w") as log:
        try:
            code = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                  timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            code = -9
    return code, time.time() - started


def generator_cmd(rounds, batch, extra):
    return [sys.executable, os.path.join(ROOT, "generate_ssa_kernel.py"), "--rounds", str(rounds),
            "--batch-size", str(batch), *extra]


# ----------------------------------------------------------------------------
# export
# ----------------------------------------------------------------------------

def stage_export(args):
    os.makedirs(os.path.join(args.out, "models"), exist_ok=True)
    for rounds, batch in parse_instances(args.instances):
        tag = instance_tag(rounds, batch)
        model = os.path.join(args.out, "models", f"{tag}.pb")
        if os.path.exists(model) and os.path.exists(model + ".params"):
            print(f"export {tag}: exists")
            continue
        env = dict(os.environ, CPSAT_EXPORT_MODEL=model)
        env.pop("CPSAT_MAX_DETERMINISTIC_TIME", None)
        cmd = generator_cmd(rounds, batch, [
            "--cpsat-deterministic-time", "0.05", "--cpsat-time-limit", str(args.time_limit), "--no-validate",
            "--output-json", os.path.join(args.out, "models", f"{tag}.sched.json"),
            "--save-kernel", os.path.join(args.out, "models", f"{tag}.kernel.json"), *args.generator_args])
        code, seconds = run_logged(cmd, os.path.join(args.out, "models", f"{tag}.log"), env=env)
        ok = code == 0 and os.path.exists(model)
        print(f"export {tag}: {'ok' if ok else f'FAILED (exit {code})'} in {seconds:.0f}s", flush=True)


# ----------------------------------------------------------------------------
# pipeline
# ----------------------------------------------------------------------------

def parse_pipeline_log(text):
    """What a generate_ssa_kernel.py --validate log says about the full-kernel solve."""
    def last(pattern, cast: type = str):
        found = re.findall(pattern, text, flags=re.M)
        return cast(found[-1]) if found else None

    return {
        "cycles": last(r"^SSA CP-SAT result: (\d+) cycles \(best_bound", int),
        "status": last(r"^SSA CP-SAT result: \d+ cycles \(best_bound: [\d.]+\), status (\w+)"),
        "dtime": last(r"^deterministic_time: ([\d.eE+-]+)", float),
        "peak_live": last(r"peak_live=(\d+)", int),
        "valu_shifted": last(r"valu_shifted=(\d+)", int),
        "validated": "PASSED - kernel produces correct output" in text,
        "dataflow_replay": bool(re.search(r"dataflow replay: \d+ kernel ops", text)),
        "wall_limited": "WARNING: CP-SAT stopped on the wall-clock limit" in text,
        "errors": sorted(set(re.findall(r"^(?:Error|ERROR|thread '.*' panicked|.*budget exceeded).*$", text,
                                        flags=re.M | re.I)))[:5],
    }


def pipeline_job(args, rounds, batch, objective, seed):
    tag = f"{instance_tag(rounds, batch)}_{objective.replace('+', '-')}_s{seed}"
    base = os.path.join(args.out, "pipeline", tag)
    if os.path.exists(base + ".json"):
        return tag, None
    env = dict(os.environ)
    env.pop("CPSAT_MAX_DETERMINISTIC_TIME", None)
    env.pop("CPSAT_EXPORT_MODEL", None)
    cmd = generator_cmd(rounds, batch, [
        "--cpsat-deterministic-time", str(args.dtime), "--cpsat-time-limit", str(args.time_limit),
        "--cpsat-objective", objective, "--cpsat-seed", str(seed), "--validate",
        "--output-json", base + ".sched.json", "--save-kernel", base + ".kernel.json", *args.generator_args])
    code, seconds = run_logged(cmd, base + ".log", env=env)
    with open(base + ".log") as f:
        record = parse_pipeline_log(f.read())
    record.update({"rounds": rounds, "batch": batch, "objective": objective, "seed": seed,
                   "budget": args.dtime, "exit_code": code, "wall": seconds})
    with open(base + ".json", "w") as f:
        json.dump(record, f, indent=1)
    return tag, record


def stage_pipeline(args):
    os.makedirs(os.path.join(args.out, "pipeline"), exist_ok=True)
    jobs = [(rounds, batch, objective, seed)
            for rounds, batch in parse_instances(args.instances)
            for seed in args.seeds
            for objective in ("makespan+tiebreak", "makespan")]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for tag, record in pool.map(lambda job: pipeline_job(args, *job), jobs):
            if record is None:
                print(f"pipeline {tag}: exists")
            else:
                print(f"pipeline {tag}: {record['cycles']} cycles, {record['status']}, validated="
                      f"{record['validated']}, dtime {record['dtime']}, exit {record['exit_code']}, "
                      f"wall {record['wall']:.0f}s" + (" WALL-LIMITED" if record["wall_limited"] else ""),
                      flush=True)


# ----------------------------------------------------------------------------
# variants
# ----------------------------------------------------------------------------

def variant_job(args, rounds, batch, name, spec, seed):
    variant, mode, overrides = spec
    tag = f"{instance_tag(rounds, batch)}_{name}_s{seed}"
    base = os.path.join(args.out, "variants", tag)
    if os.path.exists(base + ".json"):
        return tag, None
    model = os.path.join(args.out, "models", f"{instance_tag(rounds, batch)}.pb")
    cmd = [sys.executable, "-m", "scripts.objective_variants", model, "--variant", variant, "--mode", mode,
           "--dtime", str(args.dtime), "--time-limit", str(args.time_limit), "--seed", str(seed),
           "--out", base + ".json", "--log", base + ".solver.log"]
    for override in overrides:
        cmd += ["--param", override]
    if os.path.exists(base + ".solver.log"):
        os.remove(base + ".solver.log")
    code, seconds = run_logged(cmd, base + ".log")
    line = ""
    if os.path.exists(base + ".log"):
        with open(base + ".log") as f:
            line = (f.read().strip().splitlines() or [""])[-1]
    if not args.keep_logs and os.path.exists(base + ".solver.log"):
        os.remove(base + ".solver.log")
    if os.path.exists(base + ".json"):
        with open(base + ".json") as f:
            record = json.load(f)
        record.update({"rounds": rounds, "batch": batch, "name": name})
        with open(base + ".json", "w") as f:
            json.dump(record, f, indent=1)
    return tag, f"{line} [exit {code}, {seconds:.0f}s]"


def stage_variants(args):
    os.makedirs(os.path.join(args.out, "variants"), exist_ok=True)
    specs = {}
    for set_name in args.set:
        specs.update(EXPERIMENT_SETS[set_name])
    if args.only:
        specs = {name: spec for name, spec in specs.items() if name in args.only}
    jobs = [(rounds, batch, name, spec, seed)
            for rounds, batch in parse_instances(args.instances)
            for seed in args.seeds
            for name, spec in specs.items()]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for tag, line in pool.map(lambda job: variant_job(args, *job), jobs):
            print(f"variants {tag}: {line or 'exists'}", flush=True)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["export", "pipeline", "variants"])
    parser.add_argument("--out", required=True, help="directory for models, logs and records")
    parser.add_argument("--instances", nargs="*", metavar="RxB", help="e.g. 4x64 2x256 (default: all nine)")
    parser.add_argument("--seeds", nargs="*", type=int, default=[1, 2, 3])
    parser.add_argument("--dtime", type=float, default=20.0, help="deterministic-time budget per run")
    parser.add_argument("--time-limit", type=float, default=14400.0, help="wall-clock backstop per run")
    parser.add_argument("--jobs", type=int, default=2, help="runs in parallel (each uses 8 solver workers)")
    parser.add_argument("--set", nargs="*", default=["replicate"], choices=sorted(EXPERIMENT_SETS))
    parser.add_argument("--only", nargs="*", help="variants stage: run only these names from the chosen sets")
    parser.add_argument("--keep-logs", action="store_true", help="variants stage: keep the full solver logs")
    parser.add_argument("--generator-args", nargs=argparse.REMAINDER, default=[],
                        help="everything after this goes to generate_ssa_kernel.py (export and pipeline)")
    args = parser.parse_args(argv)
    args.out = os.path.abspath(args.out)
    os.makedirs(args.out, exist_ok=True)
    {"export": stage_export, "pipeline": stage_pipeline, "variants": stage_variants}[args.stage](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
