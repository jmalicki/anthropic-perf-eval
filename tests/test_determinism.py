"""Two builds of the same configuration must produce the same solver model and kernel.

Rust's HashMap/HashSet iteration order is randomised per process. If any loop that creates
solver variables, constraints or hints, or that allocates addresses, iterates such a map,
CP-SAT is handed a different model on every run and returns a different schedule, so no two
runs can be compared. Each build therefore runs in its own process.
"""

import hashlib
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts import diff_cpsat_models  # noqa: E402


def _build(tmp_path, tag, extra):
    kernel = tmp_path / f"kernel_{tag}.json"
    sched = tmp_path / f"sched_{tag}.json"
    model = tmp_path / f"model_{tag}.pb"
    env = dict(os.environ, CPSAT_EXPORT_MODEL=str(model))
    cmd = [sys.executable, os.path.join(ROOT, "generate_ssa_kernel.py"), "-q", "--no-validate",
           "--output-json", str(sched), "--save-kernel", str(kernel), *extra]
    result = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=110)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    return kernel, model


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "extra",
    [
        ["--rounds", "2", "--batch-size", "16", "--cpsat-deterministic-time", "0.5"],
        # Gather round, CSE groups and symmetry ordering all contribute constraints here.
        ["--rounds", "5", "--batch-size", "16", "--cpsat-deterministic-time", "0.5"],
        ["--rounds", "3", "--batch-size", "16", "--cpsat-deterministic-time", "0.5", "--valu-shift", "alu"],
    ],
    ids=["2r16", "5r16", "3r16-force-alu"],
)
def test_cpsat_build_is_reproducible(tmp_path, extra):
    kernel_a, model_a = _build(tmp_path, "a", extra)
    kernel_b, model_b = _build(tmp_path, "b", extra)
    diffs = diff_cpsat_models.compare(diff_cpsat_models.load(str(model_a)), diff_cpsat_models.load(str(model_b)))
    assert not diffs, "solver model differs between two identical builds:\n" + "\n".join(diffs)
    assert _sha(kernel_a) == _sha(kernel_b), "same model and deterministic budget, different kernel"


def test_greedy_build_is_reproducible(tmp_path):
    extra = ["--rounds", "5", "--batch-size", "32", "--use-greedy"]
    kernel_a, _ = _build(tmp_path, "a", extra)
    kernel_b, _ = _build(tmp_path, "b", extra)
    assert _sha(kernel_a) == _sha(kernel_b)


def test_diff_reports_reordered_constraints(tmp_path):
    """The diff tool must tell reordering apart from a real difference."""
    _, model_path = _build(tmp_path, "a", ["--rounds", "2", "--batch-size", "16", "--cpsat-deterministic-time", "0.2"])
    a = diff_cpsat_models.load(str(model_path))
    b = diff_cpsat_models.load(str(model_path))
    assert diff_cpsat_models.compare(a, b) == []
    first, second = b.constraints[0], b.constraints[1]
    tmp = type(first)()
    tmp.CopyFrom(first)
    first.CopyFrom(second)
    second.CopyFrom(tmp)
    diffs = diff_cpsat_models.compare(a, b)
    assert diffs and "different order" in diffs[0] and "index 0" in diffs[0]
    del b.constraints[5]
    assert "different entries" in "\n".join(diff_cpsat_models.compare(a, b))


def test_deterministic_budget_reaches_the_solver_through_kwargs(tmp_path, monkeypatch):
    """The documented API is scheduler_rs.build_tick_tock_kernel_ssa(**args_to_build_kwargs(...)).

    The budget must travel in those kwargs (not through a side channel), be recorded in
    run_options, and stop the solve.
    """
    import json

    import kernel_build_args
    import scheduler_rs

    monkeypatch.delenv("CPSAT_MAX_DETERMINISTIC_TIME", raising=False)
    sched = tmp_path / "sched.json"
    kwargs = kernel_build_args.args_to_build_kwargs(
        None, output_json=str(sched), use_git_sha=False,
        rounds=4, batch_size=64, cpsat_time_limit=300.0, cpsat_deterministic_time=0.5,
    )
    assert kwargs["cpsat_deterministic_time"] == 0.5
    kwargs["verbose"] = False
    scheduler_rs.build_tick_tock_kernel_ssa(**kwargs)
    with open(sched) as f:
        assert json.load(f)["run_options"]["cpsat_deterministic_time"] == 0.5


def test_environment_budget_is_recorded_when_no_option_is_given(monkeypatch):
    import json

    import kernel_build_args

    monkeypatch.setenv("CPSAT_MAX_DETERMINISTIC_TIME", "3.5")
    kwargs = kernel_build_args.args_to_build_kwargs(None, use_git_sha=False, rounds=2, batch_size=16)
    assert kwargs["cpsat_deterministic_time"] == 3.5
    assert json.loads(kwargs["run_options_json"])["cpsat_deterministic_time"] == 3.5
    assert os.environ["CPSAT_MAX_DETERMINISTIC_TIME"] == "3.5", "the inherited value must be left in place"


@pytest.mark.parametrize("bad", ["0", "-1", "nan", "inf", "abc"])
def test_cli_rejects_invalid_deterministic_budget(bad):
    import argparse

    import kernel_build_args

    parser = argparse.ArgumentParser()
    kernel_build_args.add_kernel_build_args(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(["--cpsat-deterministic-time", bad])


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_binding_rejects_invalid_deterministic_budget(bad):
    import scheduler_rs

    with pytest.raises(ValueError, match="cpsat_deterministic_time"):
        scheduler_rs.build_tick_tock_kernel_ssa(
            forest_height=10, n_nodes=2047, batch_size=16, rounds=2, verbose=False,
            use_cpsat=True, cpsat_deterministic_time=bad,
        )
