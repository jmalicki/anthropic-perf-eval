#!/usr/bin/env python3
"""Measurement for --cpsat-symmetry-rows (docs/CPSAT_CONFIG_ABLATION.md, symmetry section).

declared, none and checked, each with three CP-SAT seeds, with the default objective and with
the makespan-only objective, at 4x64 and 6x64, budget 20. Single runs differ by up to 18
cycles at 4x64 between configurations that should be equivalent, so one run per cell says
nothing. Rebuild the extension first (./scripts/build_scheduler_rs.sh --release).

    uv run python -m scripts.run_symmetry_measurement [--jobs 3]
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_cpsat_ablation  # noqa: E402


def configs() -> list[str]:
    """`name:options` for every cell."""
    out = []
    for mode in ("declared", "none", "checked"):
        for objective, extra in (("", ""), ("-mk", " --cpsat-objective makespan")):
            for seed in (1, 2, 3):
                out.append(f"{mode}{objective}-s{seed}:--cpsat-symmetry-rows {mode} --cpsat-random-seed {seed}{extra}")
    return out


def main(argv: list[str]) -> int:
    args = ["--instances", "4x64,6x64", "--budget", "20", "--out-dir", "build/cpsat_ablation_sym",
            "--name", "symmetry_seeds"]
    for c in configs():
        args += ["--config", c]
    return run_cpsat_ablation.main(args + argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
