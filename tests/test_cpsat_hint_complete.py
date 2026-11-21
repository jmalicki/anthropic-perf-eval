"""The greedy schedule must reach CP-SAT as a complete solution hint.

CP-SAT starts from a hint directly only when every non-fixed variable is hinted ("The
solution hint is complete and is feasible", first solution `complete_hint`). With even one
family of variables unhinted it has to repair the hint first, so the greedy schedule is no
longer the solver's starting point. The scratch interval of the shared CSE SUBs
(`cse_sub_end_N`, `cse_sub_size_N`) used to be unhinted from 3 rounds up.

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
    cmd = [sys.executable, os.path.join(ROOT, "generate_ssa_kernel.py"), "-q", "--no-validate",
           "--output-json", str(tmp_path / "sched.json"), "--save-kernel", str(tmp_path / "kernel.json"),
           *BASE, *extra]
    result = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=110)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    return diff_cpsat_models.load(str(model_path))


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
        pytest.param(
            ["--rounds", "3", "--const-choice", "load"],
            marks=pytest.mark.xfail(
                strict=True,
                reason="known gap: with --const-choice load 18 cycle variables (cN) are left unhinted",
            ),
        ),
        ["--rounds", "3", "--const-choice", "add_imm"],
        ["--rounds", "3", "--cpsat-makespan-interval"],
        ["--rounds", "3", "--cpsat-objective", "makespan"],
    ],
    ids=lambda extra: " ".join(extra).replace("--", "") if isinstance(extra, list) else None,
)
def test_greedy_hint_is_complete(tmp_path, extra):
    model = _export_model(tmp_path, extra)
    assert len(model.solution_hint.vars) > 0, "the model carries no hint at all"
    missing = unhinted_variables(model)
    families = collections.Counter(re.sub(r"\d+", "N", name) for name in missing)
    assert not missing, (
        f"{len(missing)} of {len(model.variables)} variables have no hint; by family: "
        + ", ".join(f"{count} x {family}" for family, count in families.most_common(8))
    )


def test_cse_groups_are_present_in_the_three_round_model(tmp_path):
    """The 3-round case above does exercise the shared-SUB variables."""
    model = _export_model(tmp_path, ["--rounds", "3"])
    names = [var.name for var in model.variables]
    assert any(name.startswith("cse_sub_end_") for name in names)
    assert any(name.startswith("cse_sub_size_") for name in names)
