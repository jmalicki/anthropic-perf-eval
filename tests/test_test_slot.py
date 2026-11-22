"""Tests for scripts/test_slot.py (machine-wide test slots for scripts/test.sh)."""

import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLOT = os.path.join(ROOT, "scripts", "test_slot.py")
WAIT = 30  # seconds; every wait below is a bounded poll, not a fixed delay


def _env(lock_dir, slots):
    env = dict(os.environ, PERF_EVAL_TEST_LOCK_DIR=str(lock_dir), PERF_EVAL_TEST_SLOTS=str(slots))
    env.pop("CI", None)
    return env


def _poll(condition, what):
    deadline = time.time() + WAIT
    while not condition():
        if time.time() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(0.05)


class Holder:
    """A slot holder: records its pid in `started` once running, then waits for `release`."""

    def __init__(self, tmp_path, name, lock_dir, slots):
        self.started = tmp_path / f"{name}.started"
        self.release = tmp_path / f"{name}.release"
        self.log = tmp_path / f"{name}.stderr"
        code = ("import os, time, pathlib; "
                f"pathlib.Path({str(self.started)!r}).write_text(str(os.getpid())); "
                f"r = pathlib.Path({str(self.release)!r})\n"
                "while not r.exists(): time.sleep(0.05)")
        with open(self.log, "w") as log:  # a file, so the wrapper's messages can be polled
            self.proc = subprocess.Popen([sys.executable, SLOT, name, "--", sys.executable, "-c", code],
                                         env=_env(lock_dir, slots), stderr=log)

    def stderr(self):
        return self.log.read_text()

    def is_running(self):
        return self.started.exists()

    def wait_running(self):
        _poll(self.is_running, f"{self.started.name}")

    def finish(self):
        self.release.write_text("go")
        assert self.proc.wait(WAIT) == 0
        return self.stderr()


def test_one_slot_serialises_runs(tmp_path):
    first = Holder(tmp_path, "a", tmp_path / "locks", 1)
    first.wait_running()
    second = Holder(tmp_path, "b", tmp_path / "locks", 1)
    _poll(lambda: "busy" in second.stderr(), "the second run to find the slot busy")
    assert not second.is_running(), "second run started while the only slot was held"
    first.finish()
    second.wait_running()
    second.finish()


def test_two_slots_allow_two_runs_at_once(tmp_path):
    first = Holder(tmp_path, "a", tmp_path / "locks", 2)
    first.wait_running()
    second = Holder(tmp_path, "b", tmp_path / "locks", 2)
    second.wait_running()  # while the first still holds its slot
    second.finish()
    first.finish()


def test_the_command_is_the_slot_holder(tmp_path):
    """The wrapper execs the command: same process, so the slot lives exactly as long as it."""
    holder = Holder(tmp_path, "a", tmp_path / "locks", 1)
    holder.wait_running()
    assert int(holder.started.read_text()) == holder.proc.pid
    holder.finish()


def test_a_killed_holder_releases_its_slot(tmp_path):
    holder = Holder(tmp_path, "a", tmp_path / "locks", 1)
    holder.wait_running()  # the slot is taken
    holder.proc.kill()
    holder.proc.wait(WAIT)
    r = subprocess.run([sys.executable, SLOT, "t", "--", "true"], env=_env(tmp_path / "locks", 1), timeout=WAIT)
    assert r.returncode == 0


def test_exit_code_is_passed_through_and_zero_slots_takes_none(tmp_path):
    for slots in (0, 1):
        r = subprocess.run([sys.executable, SLOT, "t", "--", sys.executable, "-c", "raise SystemExit(3)"],
                           env=_env(tmp_path / str(slots), slots), timeout=WAIT)
        assert r.returncode == 3
    assert not (tmp_path / "0").exists(), "PERF_EVAL_TEST_SLOTS=0 must not create lock files"


def test_missing_command_is_reported(tmp_path):
    r = subprocess.run([sys.executable, SLOT, "t", "--", "no-such-command-xyz"], env=_env(tmp_path, 1),
                       capture_output=True, text=True, timeout=WAIT)
    assert r.returncode == 127 and "cannot run" in r.stderr


def test_ci_takes_no_slot(tmp_path):
    env = dict(_env(tmp_path, 1), CI="true")
    r = subprocess.run([sys.executable, SLOT, "t", "--", "true"], env=env, timeout=WAIT)
    assert r.returncode == 0
    assert not any(tmp_path.iterdir())
