#!/usr/bin/env python3
"""Capture the instruction stream of a third-party kernel builder, in a sandbox.

    uv run python -m scripts.capture_public_kernel BUILDER.py --out kernel.json

BUILDER.py is a public take-home solution: a `perf_takehome.py` replacement that defines
`KernelBuilder` with `build_kernel(forest_height, n_nodes, batch_size, rounds)` and fills
`.instrs`. It is untrusted code, so it never runs in this process:

* It is copied, with the unmodified upstream simulator (`upstream/tests/frozen_problem.py`,
  importable as `problem`), into a fresh temporary directory. The interpreter uses
  `-I -S -B` and a minimal environment, with an import path for the standard library
  and the copied sources.
* The macOS profile restricts file reads to the capture directory, Python installation,
  and selected system runtime paths. It restricts file writes to the capture directory
  and `/dev/null`, denies networking and fork, and allows only the configured Python executable.
* The parent reads a bounded JSON result from a regular file and checks the bundle schema
  before writing the capture.

The output has the format `generate_ssa_kernel.py --save-kernel` writes, so
`scripts/check_upstream.sh --kernel` and `scripts/kernel_report.py` read it directly.
`debug` slots are dropped (the upstream test disables them).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FROZEN_PROBLEM = REPO / "upstream" / "tests" / "frozen_problem.py"
ENGINES = ("alu", "valu", "load", "store", "flow")
DEFAULT_PARAMS = (10, 2047, 256, 16)  # forest_height, n_nodes, batch_size, rounds
MAX_CAPTURE_BYTES = 16 * 1024 * 1024

DRIVER = """\
import json, resource, sys
resource.setrlimit(resource.RLIMIT_FSIZE, ({max_bytes}, {max_bytes}))
sys.path.insert(0, {sandbox!r})
from perf_takehome import KernelBuilder
kb = KernelBuilder()
kb.build_kernel({fh}, {nn}, {bs}, {r})
out = []
for bundle in kb.instrs:
    out.append({{str(e): [list(s) for s in slots] for e, slots in bundle.items() if e != "debug"}})
with open({result!r}, "w") as f:
    json.dump(out, f)
"""


def sandbox_profile(sandbox: str, python_prefix: str, python: str) -> str:
    """Allow Python to read its runtime and the capture directory on macOS."""
    return "\n".join([
        "(version 1)",
        "(allow default)",
        "(deny network*)",
        "(deny process-fork)",
        "(deny process-exec)",
        f"(allow process-exec (literal {json.dumps(python, ensure_ascii=False)}))",
        "(deny file-write*)",
        f"(allow file-write* (subpath {json.dumps(sandbox, ensure_ascii=False)}))",
        '(allow file-write* (literal "/dev/null"))',
        "(deny file-read*)",
        # Python startup needs an operation on the root directory itself.
        '(allow file-read-data (literal "/"))',
        '(allow file-read* (subpath "/System/Library") (subpath "/usr/lib")'
        ' (literal "/dev/null") (literal "/dev/urandom"))',
        f"(allow file-read* (subpath {json.dumps(python_prefix, ensure_ascii=False)}))",
        f"(allow file-read* (subpath {json.dumps(sandbox, ensure_ascii=False)}))",
    ])


def check_bundles(bundles) -> list[dict]:
    """Reject anything that is not a list of {engine: [[str|int, ...], ...]} bundles."""
    if not isinstance(bundles, list):
        raise ValueError("builder output is not a list of bundles")
    for i, bundle in enumerate(bundles):
        if not isinstance(bundle, dict):
            raise ValueError(f"bundle {i} is not a dict")
        for engine, slots in bundle.items():
            if engine not in ENGINES:
                raise ValueError(f"bundle {i}: unknown engine {engine!r}")
            if not isinstance(slots, list):
                raise ValueError(f"bundle {i}: {engine} slots are not a list")
            for slot in slots:
                if not isinstance(slot, list) or not slot or not isinstance(slot[0], str) or not all(
                        isinstance(x, int) and not isinstance(x, bool) for x in slot[1:]):
                    raise ValueError(f"bundle {i}: malformed {engine} slot {slot!r}")
    return bundles


def _read_capture_result(result: str) -> list[dict]:
    """Read only a bounded regular file from the builder's capture directory."""
    fd = os.open(result, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as output:
        info = os.fstat(output.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("builder result is not a regular file with one link")
        if info.st_size > MAX_CAPTURE_BYTES:
            raise ValueError("builder result exceeds the capture limit")
        data = output.read(MAX_CAPTURE_BYTES + 1)
    if len(data) > MAX_CAPTURE_BYTES:
        raise ValueError("builder result exceeds the capture limit")
    return check_bundles(json.loads(data))


def capture(builder: Path, params=DEFAULT_PARAMS, timeout: float = 600.0, sandbox_exec: bool = True) -> list[dict]:
    """Run `builder` in a sandboxed subprocess and return its bundles (debug slots dropped)."""
    python = os.path.realpath(getattr(sys, "_base_executable", sys.executable))
    python_prefix = os.path.realpath(sysconfig.get_config_var("prefix") or sys.base_prefix)
    use_sbx = sandbox_exec and shutil.which("sandbox-exec") is not None
    if sandbox_exec and not use_sbx:
        raise RuntimeError("sandbox-exec not found; refusing to run untrusted code unsandboxed "
                           "(pass --no-sandbox-exec only inside another sandbox)")
    with tempfile.TemporaryDirectory(prefix="pubkernel-") as tmp:
        sandbox = os.path.realpath(tmp)
        shutil.copyfile(builder, os.path.join(sandbox, "perf_takehome.py"))
        shutil.copyfile(FROZEN_PROBLEM, os.path.join(sandbox, "problem.py"))
        result = os.path.join(sandbox, "result.json")
        driver = os.path.join(sandbox, "_capture_driver.py")
        fh, nn, bs, r = params
        Path(driver).write_text(DRIVER.format(
            sandbox=sandbox, fh=fh, nn=nn, bs=bs, r=r, result=result, max_bytes=MAX_CAPTURE_BYTES,
        ))
        cmd = [python, "-I", "-S", "-B", driver]
        if use_sbx:
            cmd = ["sandbox-exec", "-p", sandbox_profile(sandbox, python_prefix, python)] + cmd
        env = {"PATH": "/usr/bin:/bin", "HOME": sandbox, "TMPDIR": sandbox, "LANG": "C.UTF-8"}
        with tempfile.TemporaryFile() as diagnostics:
            proc = subprocess.run(
                cmd, cwd=sandbox, env=env, timeout=timeout, stdout=subprocess.DEVNULL, stderr=diagnostics,
            )
            if proc.returncode != 0:
                size = diagnostics.seek(0, os.SEEK_END)
                diagnostics.seek(max(0, size - 4000))
                error = diagnostics.read(4000).decode(errors="replace")
                raise RuntimeError(f"builder failed (exit {proc.returncode}):\n{error}")
        return _read_capture_result(result)


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("builder", type=Path, help="third-party perf_takehome.py replacement")
    ap.add_argument("--out", type=Path, required=True, help="kernel JSON to write")
    ap.add_argument("--params", type=int, nargs=4, default=DEFAULT_PARAMS,
                    metavar=("HEIGHT", "NODES", "BATCH", "ROUNDS"))
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--source", default="", help="provenance (URL/author) stored in the JSON")
    ap.add_argument("--no-sandbox-exec", action="store_true",
                    help="skip sandbox-exec (only when already inside another sandbox)")
    args = ap.parse_args()
    bundles = capture(args.builder, tuple(args.params), args.timeout, not args.no_sandbox_exec)
    fh, _, bs, r = args.params
    args.out.write_text(json.dumps({"forest_height": fh, "batch_size": bs, "rounds": r,
                                    "source": args.source, "instrs": bundles}))
    n_ops = sum(len(s) for b in bundles for s in b.values())
    print(f"captured {len(bundles)} bundles, {n_ops} slots -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
