#!/usr/bin/env python3
"""Measure the original scalar baseline kernel on the vendored upstream simulator.

    uv run python -m scripts.measure_baseline [--rounds 16] [--batch-size 256] [--save KERNEL.json]

The baseline builder (``KernelBuilder.build_kernel``) is no longer in the working tree: the
root ``perf_takehome.py`` now imports this repository's builder. The script reads the
original from git (``git show dceff11:perf_takehome.py``, the repository's first commit) into
a temporary directory, runs its builder part, and simulates the kernel. The simulator is
``upstream/tests/frozen_problem.py`` (vendored unchanged; ``upstream/SHA256SUMS``), bound
as ``problem``, so neither this repository's ``problem.py`` nor the tests in the original
file are used.

Printed (JSON): cycles, bundles (all and with a non-debug slot), ops per engine, ops per
opcode, scratch words used, and whether values and indices match ``reference_kernel2``.
``--save`` writes the kernel without debug slots, with layout metadata
(``rounds``, ``batch_size``, ``forest_height``) for ``scripts/kernel_lift.py``.

Numbers and their use: docs/BASELINE_TO_BEST_GAP.md.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import subprocess
import sys
import tempfile
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE_COMMIT = "dceff11"
BASELINE_PATH = "perf_takehome.py"
FOREST_HEIGHT = 10


def baseline_source(repo: str = REPO, commit: str = BASELINE_COMMIT) -> str:
    """Text of the original baseline builder file, read from git."""
    return subprocess.run(
        ["git", "-C", repo, "show", f"{commit}:{BASELINE_PATH}"],
        check=True, capture_output=True, text=True,
    ).stdout


def upstream_problem(repo: str = REPO) -> types.ModuleType:
    """The vendored upstream simulator module (upstream/tests/frozen_problem.py)."""
    sys.path.insert(0, os.path.join(repo, "upstream", "tests"))
    try:
        import frozen_problem
    finally:
        sys.path.pop(0)
    return frozen_problem


def load_builder(source: str, problem: types.ModuleType, workdir: str) -> type:
    """KernelBuilder class from the baseline file, with `problem` bound to `problem`.

    Only the part before ``BASELINE = `` (the builder) is executed; the file's own test
    harness is not.
    """
    builder_part = source.split("BASELINE = ")[0]
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
    random.seed(seed)
    forest = problem.Tree.generate(FOREST_HEIGHT)
    inp = problem.Input.generate(forest, batch_size, rounds)
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
