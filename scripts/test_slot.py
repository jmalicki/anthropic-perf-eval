#!/usr/bin/env python3
"""Run a command while holding one of N machine-wide test slots.

Several agents or worktrees running test suites at once can load the machine so heavily that
timing-sensitive tests time out (pytest-timeout, subprocess timeouts, Z3 timeouts). This
wrapper limits how many suites run at the same time on one machine. A slot is an exclusive
`fcntl.flock` on one of N lock files in a directory shared by every worktree.

The wrapper takes a slot and then replaces itself with the command (`exec`), passing the
locked descriptor on. So the slot is held for exactly as long as the command runs, not as
long as some wrapper process lives: a killed or crashed command frees it (the kernel
releases the lock), and a command cannot keep running after its slot was freed. Processes
the command starts inherit the descriptor too, so a background child that outlives the
command keeps the slot until it exits as well; the limit can therefore be held longer than
needed, but never exceeded.

    scripts/test_slot.py SUITE -- COMMAND [ARGS...]

Environment:
    PERF_EVAL_TEST_SLOTS     number of slots (default 2); 0 runs the command without a slot
    PERF_EVAL_TEST_LOCK_DIR  lock directory (default ~/.cache/anthropic-perf-eval/test-slots)
    CI                       when set (GitHub Actions sets it), no slot is taken

Used by scripts/test.sh for every suite.
"""

from __future__ import annotations

import fcntl
import os
import sys
import time

DEFAULT_SLOTS = 2
POLL_SECONDS = 5


def lock_dir() -> str:
    return os.environ.get("PERF_EVAL_TEST_LOCK_DIR",
                          os.path.join(os.path.expanduser("~"), ".cache", "anthropic-perf-eval", "test-slots"))


def slots() -> int:
    if os.environ.get("CI"):
        return 0
    raw = os.environ.get("PERF_EVAL_TEST_SLOTS", str(DEFAULT_SLOTS))
    try:
        n = int(raw)
    except ValueError:
        sys.exit(f"test_slot.py: PERF_EVAL_TEST_SLOTS must be an integer, got {raw!r}")
    if n < 0:
        sys.exit(f"test_slot.py: PERF_EVAL_TEST_SLOTS must be 0 or more, got {n}")
    return n


def describe(path: str) -> str:
    try:
        with open(path) as f:
            return f.read().strip() or "?"
    except OSError:
        return "?"


def acquire(n: int, label: str):
    """Block until one of the n slots is free; return its open, locked file."""
    directory = lock_dir()
    os.makedirs(directory, exist_ok=True)
    paths = [os.path.join(directory, f"slot-{i}.lock") for i in range(n)]
    announced = False
    while True:
        for path in paths:
            f = open(path, "a+")
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                f.close()
                continue
            f.seek(0)
            f.truncate()
            f.write(f"{label} pid {os.getpid()} since {time.strftime('%H:%M:%S')} in {os.getcwd()}\n")
            f.flush()
            if announced:
                print(f"test_slot.py: got {os.path.basename(path)} for {label}", file=sys.stderr, flush=True)
            return f
        if not announced:
            holders = "; ".join(describe(p) for p in paths)
            print(f"test_slot.py: all {n} test slot(s) busy, waiting for {label} (holders: {holders})",
                  file=sys.stderr, flush=True)
            announced = True
        time.sleep(POLL_SECONDS)


def main(argv: list[str]) -> int:
    if "--" not in argv or argv.index("--") == len(argv) - 1:
        print(__doc__, file=sys.stderr)
        return 2
    sep = argv.index("--")
    label = " ".join(argv[:sep]) or "test"
    command = argv[sep + 1:]
    n = slots()
    if n > 0:
        held = acquire(n, label)
        os.set_inheritable(held.fileno(), True)  # the lock travels with the command
    try:
        os.execvp(command[0], command)
    except OSError as e:
        print(f"test_slot.py: cannot run {command[0]!r}: {e}", file=sys.stderr)
        return 127


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
