"""CP-SAT tuning switches: each flag must reach the solver, and defaults must change nothing.

The switches travel as one JSON object (``cpsat_tuning_json``) from kernel_build_args to
``CpsatTuning`` in Rust. Defaults live in Rust only; ``scheduler_rs.resolve_cpsat_tuning``
reports the effective values, which is what run_options records.
"""

import json
import os
import re
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import kernel_build_args  # noqa: E402
import scheduler_rs  # noqa: E402
from scripts import diff_cpsat_models  # noqa: E402

SMALL = ["--rounds", "2", "--batch-size", "16", "--cpsat-deterministic-time", "0.3", "--cpsat-time-limit", "100"]


def _parse(argv):
    import argparse

    parser = argparse.ArgumentParser()
    kernel_build_args.add_kernel_build_args(parser)
    return parser.parse_args(argv)


def _run(tmp_path, tag, extra, quiet=True):
    """Build in a fresh process; return (stdout+stderr, model path, schedule JSON path, kernel path)."""
    model = tmp_path / f"model_{tag}.pb"
    sched = tmp_path / f"sched_{tag}.json"
    kernel = tmp_path / f"kernel_{tag}.json"
    env = dict(os.environ, CPSAT_EXPORT_MODEL=str(model))
    env.pop("CPSAT_MAX_DETERMINISTIC_TIME", None)
    cmd = [sys.executable, os.path.join(ROOT, "generate_ssa_kernel.py"), "--no-validate",
           "--output-json", str(sched), "--save-kernel", str(kernel), *SMALL, *extra]
    if quiet:
        cmd.append("-q")
    result = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=110)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    return result.stdout + result.stderr, model, sched, kernel


def _solver_parameters(output):
    """The 'Parameters: ...' line CP-SAT prints for the full-kernel solve (the last solve)."""
    lines = [line for line in output.splitlines() if line.startswith("Parameters:")]
    assert lines, "no CP-SAT parameter line in the output"
    return lines[-1]


def _model_diff(a, b):
    return diff_cpsat_models.compare(diff_cpsat_models.load(str(a)), diff_cpsat_models.load(str(b)))


# ----------------------------------------------------------------------------
# Defaults and plumbing
# ----------------------------------------------------------------------------

def test_defaults_come_from_rust_and_are_recorded():
    defaults = json.loads(scheduler_rs.resolve_cpsat_tuning())
    assert defaults["workers"] == 8
    assert defaults["interleave_search"] is True
    assert defaults["subsolvers"] == [] and defaults["extra_subsolvers"] == [] and defaults["ignore_subsolvers"] == []

    kwargs = kernel_build_args.args_to_build_kwargs(_parse([]), use_git_sha=False)
    assert kwargs["cpsat_tuning_json"] is None, "nothing set: nothing is sent, Rust defaults apply"
    assert json.loads(kwargs["run_options_json"])["cpsat_tuning"] == defaults


def test_every_tuning_option_has_a_cli_flag_and_a_rust_key():
    defaults = json.loads(scheduler_rs.resolve_cpsat_tuning())
    args = _parse([])
    keys = [key for _, key in kernel_build_args.CPSAT_TUNING_OPTIONS]
    assert sorted(keys) == sorted(defaults), "CPSAT_TUNING_OPTIONS and CpsatTuning must list the same keys"
    for dest, _ in kernel_build_args.CPSAT_TUNING_OPTIONS:
        assert getattr(args, dest) is None, f"--{dest.replace('_', '-')} must default to None (Rust owns the default)"


def test_search_flags_travel_in_kwargs_and_run_options():
    args = _parse(["--cpsat-workers", "12", "--no-cpsat-interleave",
                   "--cpsat-extra-subsolvers", "objective_lb_search, reduced_costs",
                   "--cpsat-ignore-subsolvers", "rnd_cst_lns"])
    kwargs = kernel_build_args.args_to_build_kwargs(args, use_git_sha=False)
    sent = json.loads(kwargs["cpsat_tuning_json"])
    assert sent == {"workers": 12, "interleave_search": False,
                    "extra_subsolvers": ["objective_lb_search", "reduced_costs"],
                    "ignore_subsolvers": ["rnd_cst_lns"]}
    recorded = json.loads(kwargs["run_options_json"])["cpsat_tuning"]
    assert recorded["workers"] == 12 and recorded["interleave_search"] is False
    assert recorded["extra_subsolvers"] == ["objective_lb_search", "reduced_costs"]
    assert recorded["subsolvers"] == [], "unset keys are recorded with their default"

    overridden = kernel_build_args.args_to_build_kwargs(None, use_git_sha=False, cpsat_workers=3)
    assert json.loads(overridden["cpsat_tuning_json"]) == {"workers": 3}


@pytest.mark.parametrize("bad", ['{"wrokers": 4}', '{"workers": 0}', '{"workers": "8"}', "[1]",
                                 '{"extra_subsolvers": ["objective_lb"]}'])
def test_binding_rejects_bad_tuning(bad):
    with pytest.raises(ValueError, match="cpsat"):
        scheduler_rs.resolve_cpsat_tuning(bad)
    with pytest.raises(ValueError, match="cpsat"):
        scheduler_rs.build_tick_tock_kernel_ssa(
            forest_height=10, n_nodes=2047, batch_size=16, rounds=2, verbose=False,
            use_cpsat=True, cpsat_tuning_json=bad,
        )


@pytest.mark.parametrize("argv", [["--cpsat-workers", "0"], ["--cpsat-workers", "many"]])
def test_cli_rejects_bad_worker_counts(argv):
    with pytest.raises(SystemExit):
        _parse(argv)


# ----------------------------------------------------------------------------
# The flags reach the solver
# ----------------------------------------------------------------------------

def test_default_solver_parameters_are_unchanged(tmp_path):
    output, _, _, _ = _run(tmp_path, "default", [], quiet=False)
    params = _solver_parameters(output)
    assert "num_workers: 8" in params
    assert "interleave_search: true" in params
    assert "subsolvers" not in params


def test_search_flags_reach_the_solver(tmp_path):
    output, _, sched, _ = _run(
        tmp_path, "tuned",
        ["--cpsat-workers", "3", "--no-cpsat-interleave", "--cpsat-extra-subsolvers", "objective_lb_search",
         "--cpsat-ignore-subsolvers", "rnd_cst_lns"],
        quiet=False,
    )
    params = _solver_parameters(output)
    assert "num_workers: 3" in params
    assert "interleave_search: false" in params
    assert re.search(r'extra_subsolvers: "objective_lb_search"', params)
    assert re.search(r'ignore_subsolvers: "rnd_cst_lns"', params)
    with open(sched) as f:
        recorded = json.load(f)["run_options"]["cpsat_tuning"]
    assert recorded["workers"] == 3 and recorded["interleave_search"] is False


def test_search_flags_do_not_change_the_model(tmp_path):
    """Workers and search mode are solver parameters; the model handed over stays the same."""
    _, base, _, _ = _run(tmp_path, "base", [])
    _, tuned, _, _ = _run(tmp_path, "tuned", ["--cpsat-workers", "4", "--no-cpsat-interleave"])
    assert not _model_diff(base, tuned)


def test_explicit_defaults_build_the_default_kernel(tmp_path):
    """Passing every default explicitly is the same build as passing nothing."""
    _, base_model, _, base_kernel = _run(tmp_path, "base", [])
    _, same_model, _, same_kernel = _run(tmp_path, "explicit", ["--cpsat-workers", "8", "--cpsat-interleave"])
    assert not _model_diff(base_model, same_model)
    assert base_kernel.read_bytes() == same_kernel.read_bytes()


def _variables(model):
    """name -> (index, lower bound, upper bound) for the named variables of an exported model."""
    return {v.name: (i, v.domain[0], v.domain[-1]) for i, v in enumerate(model.variables) if v.name}


# ----------------------------------------------------------------------------
# Objective
# ----------------------------------------------------------------------------

def test_default_objective_has_the_tie_breaker(tmp_path):
    _, model_path, _, _ = _run(tmp_path, "base", [])
    model = diff_cpsat_models.load(str(model_path))
    variables = _variables(model)
    assert len(model.objective.vars) > 1, "makespan plus one completion term per op"
    assert any(name.startswith("shift_chosen_cycle_") for name in variables)
    assert any(c.name == "tie_sum_completion" for c in model.constraints)


def test_makespan_only_objective_reaches_the_solver(tmp_path):
    output, model_path, sched, _ = _run(tmp_path, "makespan", ["--cpsat-objective", "makespan"], quiet=False)
    model = diff_cpsat_models.load(str(model_path))
    variables = _variables(model)
    assert list(model.objective.vars) == [variables["makespan"][0]]
    assert list(model.objective.coeffs) == [1]
    assert not any("chosen_cycle" in name for name in variables), "tie-breaker auxiliaries must be gone"
    assert not any(c.name == "tie_sum_completion" for c in model.constraints)
    assert "absolute_gap_limit" not in _solver_parameters(output), "that limit is a fraction of the tie-breaker scale"
    with open(sched) as f:
        data = json.load(f)
    assert data["run_options"]["cpsat_tuning"]["objective"] == "makespan"
    # The bound is reported in cycles (no scale factor to divide out).
    assert 0 < data["lb_cycles"] <= data["total_cycles"]


def test_explicit_default_objective_is_the_default_model(tmp_path):
    _, base, _, _ = _run(tmp_path, "base", [])
    _, same, _, _ = _run(tmp_path, "explicit", ["--cpsat-objective", "makespan+tiebreak"])
    assert not _model_diff(base, same)
