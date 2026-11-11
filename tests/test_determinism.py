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
