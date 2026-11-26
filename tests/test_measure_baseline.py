"""scripts/measure_baseline.py: the original scalar baseline, measured on the vendored upstream
simulator, reproduces upstream's BASELINE = 147734; the script pins the file by full sha and
sha256, loads the simulator privately and leaves the global random state alone."""

import os
import random
import subprocess
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import measure_baseline as mb  # noqa: E402


def _have_commit() -> bool:
    return subprocess.run(["git", "-C", mb.REPO, "cat-file", "-e", f"{mb.BASELINE_COMMIT}^{{commit}}"],
                          capture_output=True).returncode == 0


needs_history = pytest.mark.skipif(not _have_commit(), reason="baseline commit not in this clone (shallow checkout)")


@needs_history
def test_reproduces_the_upstream_baseline():
    report, _ = mb.measure(16, 256)
    assert report["cycles"] == 147734
    assert report["values_match_reference"] and report["indices_match_reference"]


@needs_history
def test_global_random_state_is_restored():
    random.seed(99)
    before = random.getstate()
    mb.measure(1, 8)
    assert random.getstate() == before


@needs_history
def test_file_is_checked_by_sha256():
    with pytest.raises(RuntimeError, match="sha256"):
        mb.baseline_source(commit="HEAD")  # today's perf_takehome.py is a different file


def test_a_file_without_the_builder_marker_is_not_executed(tmp_path):
    problem = types.ModuleType("problem")
    with pytest.raises(RuntimeError, match="marker"):
        mb.load_builder("raise SystemExit('executed')\n", problem, str(tmp_path))


def test_simulator_is_the_vendored_upstream_one_loaded_privately(monkeypatch):
    impostor = types.ModuleType("frozen_problem")
    monkeypatch.setitem(sys.modules, "frozen_problem", impostor)
    module = mb.upstream_problem()
    assert module is not impostor
    assert os.path.samefile(module.__file__, os.path.join(mb.REPO, "upstream", "tests", "frozen_problem.py"))
    assert sys.modules["frozen_problem"] is impostor
    assert hasattr(module, "Machine")
