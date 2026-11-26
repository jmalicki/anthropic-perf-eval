"""`./scripts/test.sh python` must run every test file in the repository (CI runs that suite).

It collects `tests/` by discovery (pytest's `test_*.py` pattern), so a new file there runs
without editing anything. A test file elsewhere is not collected unless the script names it:
this test lists every tracked `test_*.py` / `*_test.py` outside `tests/` and fails on one that
is neither run by the suite nor excluded below with a reason.
"""
import os
import re
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Outside tests/ and run by `run_python` (named in scripts/test.sh).
RUN_EXPLICITLY = {"solver/test_slot_utilization.py"}
# Outside tests/ and deliberately not run, with the reason.
EXCLUDED = {
    "scripts/test_slot.py": "a tool (machine-wide test slots), not tests; tested by tests/test_test_slot.py",
    "test_optimized.py": "manual 16-round performance harness (`uv run test_optimized.py`), "
                         "its test_ function takes build parameters; covered by the kernel/upstream suites",
}


def _tracked_test_files():
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    pattern = re.compile(r"(^|/)(test_[^/]*\.py|[^/]*_test\.py)$")
    return sorted(f for f in out.splitlines() if pattern.search(f) and not f.startswith("upstream/"))


def test_every_test_file_is_run_or_excluded():
    outside = [f for f in _tracked_test_files() if not f.startswith("tests/")]
    unknown = [f for f in outside if f not in RUN_EXPLICITLY and f not in EXCLUDED]
    assert not unknown, (
        f"test files outside tests/ that ./scripts/test.sh python does not run: {unknown}; "
        "move them under tests/, or name them in scripts/test.sh run_python and RUN_EXPLICITLY here"
    )
    assert not [f for f in EXCLUDED if f not in outside], "stale exclusion: file no longer exists"


def test_python_suite_collects_tests_and_the_named_files():
    script = open(os.path.join(ROOT, "scripts", "test.sh")).read()
    body = script[script.index("run_python() {"):script.index("run_ci_only() {")]
    default = re.search(r"set -- (.*)\n", body).group(1).split()
    assert default[0] == "tests", "the python suite must collect tests/ by discovery"
    assert set(default[1:]) == RUN_EXPLICITLY
