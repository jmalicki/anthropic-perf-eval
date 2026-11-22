#!/usr/bin/env python3
"""Check that a change to the CP-SAT model builder leaves every model it builds unchanged.

A refactor of `scheduler_rs/src/cpsat_scheduler/` that claims not to change the model must
hand CP-SAT byte-identical models (same variables, constraints, hints, strategy and
objective, in the same order) and so produce byte-identical kernels. This script records
the models and kernels of a matrix of builds with the installed extension, and compares two
such records.

Run it from the change's checkout and swap only the Rust sources, so this script (and the
Python it imports) stays the change's version while the base extension is built. The base
must export models with `CPSAT_EXPORT_MODEL_DIR`; on a base that predates it, cherry-pick the
commit that added it.

    # 1. Build the base's Rust sources and record them
    git checkout <base> -- scheduler_rs/ && ./scripts/build_scheduler_rs.sh --release
    uv run python -m scripts.cpsat_model_identity record --out build/model_identity/before

    # 2. Restore the change's Rust sources, build and record
    git checkout HEAD -- scheduler_rs/ && ./scripts/build_scheduler_rs.sh --release
    uv run python -m scripts.cpsat_model_identity record --out build/model_identity/after

    # 3. Compare (exit status 1 if any model or kernel differs)
    uv run python -m scripts.cpsat_model_identity compare build/model_identity/before build/model_identity/after

(`git checkout <base> -- scheduler_rs/` leaves files that exist only in the change in place;
they are not compiled unless a module declares them.)

`record` exits 1 if any cell failed (nonzero return code, no exported model or no kernel),
and `compare` exits 1 unless both records have the same, nonempty set of cells, every cell
is valid in both, and every model and kernel is identical.

Every model built in a run is kept (`CPSAT_EXPORT_MODEL_DIR`), per-section feasibility
solves included, not only the last one. Each build uses a deterministic-time budget, so the
kernels are comparable as long as the solves end on that budget (the wall-clock limit is a
generous backstop).

The matrix covers default options, each forced encoding mode, the makespan-only objective,
the makespan interval and the shaped greedy start, on several instance sizes. Restrict it
with --instances / --configs (names as printed by --list).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import diff_cpsat_models  # noqa: E402

DEFAULT_INSTANCES = ["2x16", "3x64", "4x64", "3x256"]

CONFIGS: list[tuple[str, str]] = [
    ("default", ""),
    ("shift-alu", "--valu-shift alu"),
    ("shift-valu", "--valu-shift valu"),
    ("select-fma", "--select-choice fma"),
    ("select-vselect", "--select-choice vselect"),
    ("const-load", "--const-choice load"),
    ("const-addimm", "--const-choice add_imm"),
    ("obj-makespan", "--cpsat-objective makespan"),
    ("makespan-interval", "--cpsat-makespan-interval"),
    ("shaped-greedy", "--greedy-barriers none --greedy-priority wavefront --greedy-split-shift"),
    ("vb-elim", "--enable-eliminate-vbroadcast"),
    ("no-cse", "--no-cse-groups"),
]

# Cells that exercise the per-section (segment-mode) model: the greedy's "greedy" pressure
# policy overflows the scratch budget in one section, so a per-section feasibility solve
# runs and its model is exported first (model_000.pb). That solve finds no schedule within a
# small deterministic budget, so the run fails; the cell is valid when it fails the same way
# (same return code, models exported, no kernel). (instance, name, options)
SEGMENT_CELLS: list[tuple[str, str, str]] = [
    ("3x256", "segment-greedy-policy", "--scheduling-pressure-policy greedy"),
]


def _cell_dir(out: Path, instance: str, config: str) -> Path:
    return out / f"{instance}__{config}"


def _run_cell(out: Path, instance: str, config: str, options: str, budget: float, time_limit: float) -> dict:
    rounds, batch = instance.split("x")
    cell = _cell_dir(out, instance, config)
    models = cell / "models"
    if cell.exists():
        for old in sorted(models.glob("*.pb")) if models.exists() else []:
            old.unlink()
    models.mkdir(parents=True, exist_ok=True)
    kernel = cell / "kernel.json"
    if kernel.exists():
        kernel.unlink()
    cmd = [
        sys.executable, str(ROOT / "generate_ssa_kernel.py"), "-q", "--no-validate",
        "--rounds", rounds, "--batch-size", batch,
        "--cpsat-deterministic-time", str(budget), "--cpsat-time-limit", str(time_limit),
        "--output-json", str(cell / "sched.json"), "--save-kernel", str(kernel),
        *shlex.split(options),
    ]
    env = dict(os.environ, CPSAT_EXPORT_MODEL_DIR=str(models))
    env.pop("CPSAT_EXPORT_MODEL", None)
    result = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    (cell / "log.txt").write_text(result.stdout + "\n--- stderr ---\n" + result.stderr)
    wall_clock_stop = "stopped on the wall-clock limit" in result.stderr + result.stdout
    record = {
        "instance": instance,
        "config": config,
        "expect_failure": any(i == instance and n == config for i, n, _ in SEGMENT_CELLS),
        "options": options,
        "command": shlex.join(cmd),
        "returncode": result.returncode,
        "models": len(list(models.glob("*.pb"))),
        "kernel_sha256": hashlib.sha256(kernel.read_bytes()).hexdigest() if kernel.exists() else None,
        "wall_clock_stop": wall_clock_stop,
    }
    (cell / "record.json").write_text(json.dumps(record, indent=2))
    return record


def record(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    configs = [(n, o) for n, o in CONFIGS if not args.configs or n in args.configs]
    cells = [(i, n, o) for i in args.instances for n, o in configs]
    cells += [(i, n, o) for i, n, o in SEGMENT_CELLS if not args.configs or n in args.configs]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(_run_cell, out, i, n, o, args.budget, args.time_limit): (i, n) for i, n, o in cells
        }
        failed = 0
        for future in concurrent.futures.as_completed(futures):
            r = future.result()
            problem = _record_problem(r)
            failed += bool(problem)
            print(
                f"{r['instance']:>6} {r['config']:<18} rc={r['returncode']} models={r['models']} "
                f"kernel={(r['kernel_sha256'] or '-')[:12]}{' WALL-CLOCK STOP' if r['wall_clock_stop'] else ''}"
                f"{'  FAILED: ' + problem if problem else ''}",
                flush=True,
            )
    if failed:
        print(f"\n{failed} of {len(cells)} cells failed (see log.txt in each cell)")
    return 1 if failed else 0


def _record_problem(r: dict) -> str | None:
    """Why a cell's record cannot be compared (None when it is valid)."""
    if r.get("expect_failure"):
        if r["models"] < 1:
            return "no model exported (does the extension support CPSAT_EXPORT_MODEL_DIR?)"
        return None
    if r["returncode"] != 0:
        return f"return code {r['returncode']}"
    if r["models"] < 1:
        return "no model exported (does the extension support CPSAT_EXPORT_MODEL_DIR?)"
    if not r["kernel_sha256"]:
        return "no kernel written"
    return None


def compare(args: argparse.Namespace) -> int:
    a_root, b_root = Path(args.a), Path(args.b)
    def cell_names(root: Path) -> set[str]:
        return {p.name for p in root.iterdir() if (p / "record.json").exists()} if root.is_dir() else set()

    names_a, names_b = cell_names(a_root), cell_names(b_root)
    if not names_a or not names_b:
        print(f"no recorded cells in {a_root if not names_a else b_root}")
        return 1
    failures = 0
    for name in sorted(names_a ^ names_b):
        print(f"{name:<32} MISSING in {b_root if name in names_a else a_root}")
        failures += 1
    cells = sorted(names_a & names_b)
    for name in cells:
        ra = json.loads((a_root / name / "record.json").read_text())
        rb = json.loads((b_root / name / "record.json").read_text())
        models_a = sorted((a_root / name / "models").glob("*.pb"))
        models_b = sorted((b_root / name / "models").glob("*.pb"))
        problems: list[str] = []
        for label, r, models in (("a", ra, models_a), ("b", rb, models_b)):
            problem = _record_problem(r)
            if problem:
                problems.append(f"invalid record in {label}: {problem}")
            elif len(models) != r["models"]:
                # An interrupted re-recording deletes the models but leaves the old record.
                problems.append(f"invalid record in {label}: {len(models)} model files, record says {r['models']}")
        if len(models_a) != len(models_b):
            problems.append(f"{len(models_a)} vs {len(models_b)} models")
        for ma, mb in zip(models_a, models_b):
            diffs = diff_cpsat_models.compare(diff_cpsat_models.load(str(ma)), diff_cpsat_models.load(str(mb)))
            if diffs:
                problems.append(f"{ma.name}:\n      " + "\n      ".join(diffs))
        if ra["returncode"] != rb["returncode"]:
            problems.append(f"return code {ra['returncode']} vs {rb['returncode']}")
        if ra["kernel_sha256"] != rb["kernel_sha256"]:
            problems.append(f"kernel {str(ra['kernel_sha256'])[:12]} vs {str(rb['kernel_sha256'])[:12]}")
        if ra["wall_clock_stop"] or rb["wall_clock_stop"]:
            problems.append("a solve stopped on the wall-clock limit: kernels are not comparable")
        status = "IDENTICAL" if not problems else "DIFFERS"
        print(f"{name:<32} {status}  models={len(models_a)} rc={ra['returncode']} "
              f"kernel={(ra['kernel_sha256'] or '-')[:12]}")
        for p in problems:
            print(f"    {p}")
        failures += bool(problems)
    total = len(names_a | names_b)
    print(f"\n{total - failures} of {total} cells identical")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("record", help="build the matrix with the installed extension and keep models and kernels")
    rec.add_argument("--out", required=True)
    rec.add_argument("--instances", type=lambda s: s.split(","), default=DEFAULT_INSTANCES)
    rec.add_argument("--configs", type=lambda s: s.split(","), default=None)
    rec.add_argument("--budget", type=float, default=1.0, help="--cpsat-deterministic-time per solve")
    rec.add_argument("--time-limit", type=float, default=900.0, help="wall-clock backstop per solve")
    rec.add_argument("--jobs", type=int, default=4)
    cmp_ = sub.add_parser("compare", help="compare two records")
    cmp_.add_argument("a")
    cmp_.add_argument("b")
    sub.add_parser("list", help="print the configurations")
    args = parser.parse_args()
    if args.command == "record":
        return record(args)
    if args.command == "compare":
        return compare(args)
    for name, options in CONFIGS:
        print(f"{name:<18} {options}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
