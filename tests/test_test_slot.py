"""Tests for scripts/test_slot.py (machine-wide test slots for scripts/test.sh)."""

import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLOT = os.path.join(ROOT, "scripts", "test_slot.py")


def _env(tmp_path, slots):
    env = dict(os.environ, PERF_EVAL_TEST_LOCK_DIR=str(tmp_path), PERF_EVAL_TEST_SLOTS=str(slots))
    env.pop("CI", None)
    return env


def _hold(tmp_path, slots, seconds, marker):
    """Start a slot holder that records its start and end time in marker."""
    code = (f"import time; open({str(marker)!r}, 'a').write(f'start {{time.time()}}\\n'); "
            f"time.sleep({seconds}); open({str(marker)!r}, 'a').write(f'end {{time.time()}}\\n')")
    return subprocess.Popen([sys.executable, SLOT, "t", "--", sys.executable, "-c", code],
                            env=_env(tmp_path, slots), stderr=subprocess.PIPE, text=True)


def _times(marker):
    lines = dict(line.split() for line in marker.read_text().splitlines())
    return float(lines["start"]), float(lines["end"])


def test_one_slot_serialises_runs(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    first = _hold(tmp_path, 1, 1.5, a)
    time.sleep(0.5)  # let the first take the slot
    second = _hold(tmp_path, 1, 0.1, b)
    assert first.wait(30) == 0 and second.wait(30) == 0
    assert _times(b)[0] >= _times(a)[1], "second run started before the first released the slot"
    assert "busy" in second.stderr.read()


def test_two_slots_allow_two_runs_at_once(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    first = _hold(tmp_path, 2, 1.5, a)
    time.sleep(0.5)
    second = _hold(tmp_path, 2, 0.1, b)
    assert first.wait(30) == 0 and second.wait(30) == 0
    assert _times(b)[0] < _times(a)[1], "second run waited although a slot was free"


def test_exit_code_is_passed_through_and_zero_slots_takes_none(tmp_path):
    for slots in (0, 1):
        r = subprocess.run([sys.executable, SLOT, "t", "--", sys.executable, "-c", "raise SystemExit(3)"],
                           env=_env(tmp_path / str(slots), slots))
        assert r.returncode == 3
    assert not (tmp_path / "0").exists(), "PERF_EVAL_TEST_SLOTS=0 must not create lock files"


def test_a_killed_holder_releases_its_slot(tmp_path):
    marker = tmp_path / "m"
    holder = _hold(tmp_path, 1, 60, marker)
    time.sleep(0.5)
    holder.kill()
    holder.wait(10)
    start = time.time()
    r = subprocess.run([sys.executable, SLOT, "t", "--", "true"], env=_env(tmp_path, 1), timeout=30)
    assert r.returncode == 0 and time.time() - start < 10


def test_ci_takes_no_slot(tmp_path):
    env = dict(_env(tmp_path, 1), CI="true")
    r = subprocess.run([sys.executable, SLOT, "t", "--", "true"], env=env)
    assert r.returncode == 0
    assert not any(tmp_path.iterdir())
