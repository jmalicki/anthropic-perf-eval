#!/usr/bin/env python3
"""Measure the original scalar baseline kernel on the vendored upstream simulator.

    uv run python -m scripts.measure_baseline [--rounds 16] [--batch-size 256] [--save KERNEL.json]

The baseline builder (``KernelBuilder.build_kernel``) is no longer in the working tree: the
root ``perf_takehome.py`` now imports this repository's builder. The script reads the
original from git (``perf_takehome.py`` at commit ``dceff1151fb8…``, the first commit that
contains the file; the root commit 9da5787 does not), checks its sha256, runs its builder
part, and simulates the kernel. The commit is named by its full sha, so a branch or tag that
happens to be called ``dceff11`` cannot be read instead. The simulator is
``upstream/tests/frozen_problem.py`` (vendored unchanged), checked against
``upstream/SHA256SUMS`` and imported under a private module name, bound as ``problem``, so
neither this repository's ``problem.py`` nor the tests in the original file are used. The
global ``random`` state is restored after the inputs are generated.

Printed (JSON): cycles, bundles (all and with a non-debug slot), ops per engine, ops per
opcode, scratch words used, and whether values and indices match ``reference_kernel2``.
``--save`` writes the kernel without debug slots, with layout metadata
(``rounds``, ``batch_size``, ``forest_height``) for ``scripts/kernel_lift.py``.

Numbers and their use: docs/BASELINE_TO_BEST_GAP.md.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import os
import random
import subprocess
import sys
import tempfile
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The first commit containing perf_takehome.py (full sha: an abbreviation could resolve to a
# branch or tag of that name), and the sha256 of the file there.
BASELINE_COMMIT = "dceff1151fb8372e26ed19e7c1c490b92bd09c1d"
BASELINE_PATH = "perf_takehome.py"
BASELINE_SHA256 = "f9c1de0945031d3333a0cae78cad2e93c93fe67fa0baaee6a3e9a9cc8e6267eb"
# The part of the file that is the builder ends where its test harness begins.
BUILDER_END_MARKER = "BASELINE = "
FOREST_HEIGHT = 10


def baseline_source(repo: str = REPO, commit: str = BASELINE_COMMIT) -> str:
    """Text of the original baseline builder file, read from git and checked by sha256."""
    data = subprocess.run(
        ["git", "-C", repo, "show", f"{commit}:{BASELINE_PATH}"],
        check=True, capture_output=True,
    ).stdout
    digest = hashlib.sha256(data).hexdigest()
    if digest != BASELINE_SHA256:
        raise RuntimeError(f"{commit}:{BASELINE_PATH} has sha256 {digest}, expected {BASELINE_SHA256}")
    return data.decode()


def _expected_sha256(repo: str, relpath: str) -> str:
    """The sha256 that upstream/SHA256SUMS records for `relpath` (relative to upstream/)."""
    with open(os.path.join(repo, "upstream", "SHA256SUMS")) as f:
        for line in f:
            digest, name = line.split()
            if name == relpath:
                return digest
    raise RuntimeError(f"upstream/SHA256SUMS has no entry for {relpath}")


def upstream_problem(repo: str = REPO) -> types.ModuleType:
    """The vendored upstream simulator (upstream/tests/frozen_problem.py), checked against
    upstream/SHA256SUMS and loaded under a private module name, so a ``frozen_problem``
    already imported from elsewhere (the root one re-exports this repository's simulator)
    can be neither used nor replaced."""
    path = os.path.join(repo, "upstream", "tests", "frozen_problem.py")
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    want = _expected_sha256(repo, "tests/frozen_problem.py")
    if digest != want:
        raise RuntimeError(f"{path} has sha256 {digest}, expected {want} (upstream/SHA256SUMS)")
    spec = importlib.util.spec_from_file_location("_measure_baseline_upstream_problem", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_builder(source: str, problem: types.ModuleType, workdir: str) -> type:
    """KernelBuilder class from the baseline file, with `problem` bound to `problem`.

    Only the part before ``BASELINE = `` (the builder) is executed; the file's own test
    harness is not.
    """
    if BUILDER_END_MARKER not in source:
        raise RuntimeError(f"the baseline file has no {BUILDER_END_MARKER!r} marker; refusing to execute all of it")
    builder_part = source.split(BUILDER_END_MARKER)[0]
    path = os.path.join(workdir, "baseline_perf_takehome.py")
    with open(path, "w") as f:
        f.write(builder_part)
    saved = sys.modules.get("problem")
    sys.modules["problem"] = problem
    try:
        module = types.ModuleType("baseline_perf_takehome")
        module.__file__ = path
        exec(compile(builder_part, path, "exec"), module.__dict__)
    finally:
        if saved is None:
            sys.modules.pop("problem", None)
        else:
            sys.modules["problem"] = saved
    return module.KernelBuilder


def measure(rounds: int, batch_size: int, seed: int = 123) -> tuple[dict, list[dict]]:
    """Build the baseline kernel, count its ops and run it.

    Returns:
        (report, instrs): the counts as printed, and the kernel's bundles.
    """
    problem = upstream_problem()
    with tempfile.TemporaryDirectory() as tmp:
        builder_cls = load_builder(baseline_source(), problem, tmp)
    # The simulator draws from the global `random`: seed it for this run and restore the
    # caller's state afterwards.
    state = random.getstate()
    try:
        random.seed(seed)
        forest = problem.Tree.generate(FOREST_HEIGHT)
        inp = problem.Input.generate(forest, batch_size, rounds)
    finally:
        random.setstate(state)
    mem = problem.build_mem_image(forest, inp)
    kb = builder_cls()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)

    per_engine, per_opcode, non_debug = collections.Counter(), collections.Counter(), 0
    for bundle in kb.instrs:
        if any(engine != "debug" for engine in bundle):
            non_debug += 1
        for engine, slots in bundle.items():
            for slot in slots:
                per_engine[engine] += 1
                per_opcode[f"{engine}:{slot[0]}"] += 1

    machine = problem.Machine(list(mem), kb.instrs, kb.debug_info(), n_cores=problem.N_CORES,
                              value_trace={})
    machine.enable_pause = False
    machine.enable_debug = False
    machine.run()
    ref = None
    for ref in problem.reference_kernel2(list(mem), {}):
        pass
    idx_p, val_p = ref[5], ref[6]
    report = {
        "baseline": f"{BASELINE_COMMIT}:{BASELINE_PATH}",
        "rounds": rounds,
        "batch_size": batch_size,
        "forest_height": FOREST_HEIGHT,
        "seed": seed,
        "cycles": machine.cycle,
        "bundles": len(kb.instrs),
        "bundles_with_non_debug_slot": non_debug,
        "ops_per_engine": dict(sorted(per_engine.items())),
        "ops_per_opcode": dict(sorted(per_opcode.items())),
        "scratch_words_used": kb.scratch_ptr,
        "values_match_reference": machine.mem[val_p:val_p + batch_size] == ref[val_p:val_p + batch_size],
        "indices_match_reference": machine.mem[idx_p:idx_p + batch_size] == ref[idx_p:idx_p + batch_size],
    }
    return report, kb.instrs


def save_kernel(instrs: list[dict], path: str, rounds: int, batch_size: int) -> None:
    """Write the kernel for scripts/kernel_lift.py: debug slots dropped, layout metadata set."""
    bundles = []
    for bundle in instrs:
        kept = {engine: [list(slot) for slot in slots] for engine, slots in bundle.items() if engine != "debug"}
        if kept:
            bundles.append(kept)
    with open(path, "w") as f:
        json.dump({"instrs": bundles, "rounds": rounds, "batch_size": batch_size,
                   "forest_height": FOREST_HEIGHT}, f)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--rounds", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--save", metavar="KERNEL.json", help="write the kernel for scripts/kernel_lift.py")
    args = ap.parse_args(argv)
    if args.rounds < 0 or args.batch_size < 1:
        ap.error("--rounds must be >= 0 and --batch-size >= 1")
    report, instrs = measure(args.rounds, args.batch_size, args.seed)
    print(json.dumps(report, indent=1))
    if args.save:
        save_kernel(instrs, args.save, args.rounds, args.batch_size)
    ok = report["values_match_reference"] and report["indices_match_reference"]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
