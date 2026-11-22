"""The greedy schedule must reach CP-SAT as a complete solution hint.

CP-SAT starts from a hint directly only when every non-fixed variable is hinted ("The
solution hint is complete and is feasible", first solution `complete_hint`). With even one
family of variables unhinted it has to repair the hint first, so the greedy schedule is no
longer the solver's starting point. The scratch interval of the shared CSE SUBs
(`cse_sub_end_N`, `cse_sub_size_N`) used to be unhinted from 3 rounds up, and under forced
`--const-choice load` / `--valu-shift alu` some hints were skipped even when the greedy had
already used the forced encoding. The test checks both that every variable is hinted and that
CP-SAT reports the hint feasible.

Each build exports its model (CPSAT_EXPORT_MODEL) and the test reads the hint from it.
"""

import collections
import os
import re
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts import diff_cpsat_models  # noqa: E402

BASE = ["--batch-size", "32", "--cpsat-deterministic-time", "0.2"]


def unhinted_variables(model):
    """Names of the non-fixed variables of an exported model that have no hint."""
    hinted = set(model.solution_hint.vars)
    return [
        var.name or f"var_{index}"
        for index, var in enumerate(model.variables)
        if index not in hinted and not (len(var.domain) == 2 and var.domain[0] == var.domain[1])
    ]


def _export_model(tmp_path, extra):
    model_path = tmp_path / "model.pb"
    env = dict(os.environ, CPSAT_EXPORT_MODEL=str(model_path))
    cmd = [sys.executable, os.path.join(ROOT, "generate_ssa_kernel.py"), "--no-validate",
           "--output-json", str(tmp_path / "sched.json"), "--save-kernel", str(tmp_path / "kernel.json"),
           *BASE, *extra]
    result = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=110)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    return diff_cpsat_models.load(str(model_path)), result.stdout + result.stderr


@pytest.mark.parametrize(
    "extra",
    [
        ["--rounds", "2"],
        # From 3 rounds up the selects share subtractions (CSE groups).
        ["--rounds", "3"],
        ["--rounds", "4"],
        # A gather round.
        ["--rounds", "5"],
        ["--rounds", "3", "--no-valu-shift"],
        ["--rounds", "3", "--select-choice", "vselect"],
        ["--rounds", "3", "--const-choice", "load"],
        ["--rounds", "3", "--valu-shift", "alu"],
        ["--rounds", "3", "--valu-shift", "valu"],
        ["--rounds", "3", "--select-choice", "fma"],
        ["--rounds", "3", "--const-choice", "add_imm"],
        ["--rounds", "3", "--cpsat-makespan-interval"],
        ["--rounds", "3", "--cpsat-objective", "makespan"],
    ],
    ids=lambda extra: " ".join(extra).replace("--", ""),
)
def test_greedy_hint_is_complete(tmp_path, extra):
    model, log = _export_model(tmp_path, extra)
    assert len(model.solution_hint.vars) > 0, "the model carries no hint at all"
    missing = unhinted_variables(model)
    families = collections.Counter(re.sub(r"\d+", "N", name) for name in missing)
    assert not missing, (
        f"{len(missing)} of {len(model.variables)} variables have no hint; by family: "
        + ", ".join(f"{count} x {family}" for family, count in families.most_common(8))
    )
    # Complete is not enough: the greedy schedule must also satisfy the model, or CP-SAT
    # repairs it instead of starting from it.
    assert "The solution hint is complete and is feasible" in log, (
        "CP-SAT did not accept the greedy hint as a solution: "
        + " | ".join(line for line in log.splitlines() if "solution hint" in line)
    )


def test_cse_groups_are_present_in_the_three_round_model(tmp_path):
    """The 3-round case above does exercise the shared-SUB variables."""
    model, _ = _export_model(tmp_path, ["--rounds", "3"])
    names = [var.name for var in model.variables]
    assert any(name.startswith("cse_sub_end_") for name in names)
    assert any(name.startswith("cse_sub_size_") for name in names)
