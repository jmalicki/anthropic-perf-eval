#!/usr/bin/env python3
"""Compare trace_io_log from two kernels (good vs bad) on same input; report first divergence.

Usage:
  uv run python scripts/compare_trace_logs.py kernel_pass.json kernel_fail.json --rounds 2 --batch-size 16
"""

import argparse
import json
import random
from copy import copy

from problem import (
    Machine,
    Tree,
    Input,
    build_mem_image,
    N_CORES,
    DebugInfo,
)


def load_instrs(path):
    with open(path) as f:
        data = json.load(f)
    return data.get("instrs", data) if isinstance(data, dict) else data


def run_and_capture_trace(instrs, mem, seed=123):
    machine = Machine(
        copy(mem),
        list(instrs),
        DebugInfo(scratch_map={}),
        n_cores=N_CORES,
        trace=False,
    )
    machine.enable_pause = False
    machine.enable_debug = True
    machine.prints = False
    machine.run()
    return getattr(machine, "trace_io_log", []) or []


def main():
    ap = argparse.ArgumentParser(description="Compare trace_io_log of two kernels on same input")
    ap.add_argument("kernel_pass", help="Passing kernel JSON")
    ap.add_argument("kernel_fail", help="Failing kernel JSON")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    random.seed(args.seed)
    forest = Tree.generate(10)
    inp = Input.generate(forest, args.batch_size, args.rounds)
    mem = build_mem_image(forest, inp)

    instrs_pass = load_instrs(args.kernel_pass)
    instrs_fail = load_instrs(args.kernel_fail)

    print("Running passing kernel...")
    log_pass = run_and_capture_trace(instrs_pass, mem, args.seed)
    print(f"  {len(log_pass)} trace entries")

    print("Running failing kernel...")
    log_fail = run_and_capture_trace(instrs_fail, mem, args.seed)
    print(f"  {len(log_fail)} trace entries")

    # Build comparable keys: (cycle, io_type, trace_id) -> values
    def key(e):
        return (e[0], e[1], e[2])

    by_pass = {key(e): e[3] for e in log_pass}
    by_fail = {key(e): e[3] for e in log_fail}

    # Order by cycle then trace_id (use pass order)
    ordered_keys = [key(e) for e in log_pass]
    for k in (key(e) for e in log_fail):
        if k not in by_pass:
            ordered_keys.append(k)
    ordered_keys.sort(key=lambda x: (x[0], x[2]))

    print("\n--- First divergence ---")
    for k in ordered_keys:
        cycle, io_type, trace_id = k
        vp = by_pass.get(k)
        vf = by_fail.get(k)
        if vp is None:
            print(f"  Only in FAIL: cycle={cycle} {io_type} {trace_id}")
            continue
        if vf is None:
            print(f"  Only in PASS: cycle={cycle} {io_type} {trace_id}")
            continue
        vp = tuple(vp) if isinstance(vp, (list, tuple)) else (vp,)
        vf = tuple(vf) if isinstance(vf, (list, tuple)) else (vf,)
        if vp != vf:
            print(f"  FIRST DIVERGENCE: cycle={cycle} {io_type} {trace_id}")
            print(f"    PASS: {vp}")
            print(f"    FAIL: {vf}")
            if len(vp) == len(vf):
                for i, (a, b) in enumerate(zip(vp, vf)):
                    if a != b:
                        print(f"    Lane {i}: PASS={a} (0x{a&0xffffffff:x})  FAIL={b} (0x{b&0xffffffff:x})")
            return 0

    print("  No divergence in trace_io_log (same (cycle, io_type, trace_id) values).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
