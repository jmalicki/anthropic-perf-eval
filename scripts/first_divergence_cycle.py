#!/usr/bin/env python3
"""Find the first cycle at which scratch/memory diverges from reference.

Runs the optimized kernel cycle-by-cycle and compares the output region
(inp_values, inp_indices) to reference round states. Reports the first cycle
where the output region does not match any reference state (divergence).

  uv run python scripts/first_divergence_cycle.py --kernel kern.json [--seed 123]

Output directory (for plan): artifacts/root_cause/1D_first_divergence/
"""

import sys
import os

# Avoid scripts/ shadowing stdlib (e.g. bisect used by random); do before other imports
_scripts = os.path.dirname(os.path.abspath(__file__))
if _scripts in sys.path:
    sys.path.remove(_scripts)
sys.path.insert(0, os.path.dirname(_scripts))

import argparse
import json
import random


def load_kernel_and_params(path):
    with open(path) as f:
        data = json.load(f)
    if "instrs" in data:
        instrs = data["instrs"]
        params = {
            "forest_height": data.get("forest_height", 10),
            "batch_size": data.get("batch_size", 256),
            "rounds": data.get("rounds", 16),
            "seed": data.get("seed", 123),
        }
        return instrs, params
    return data, {"forest_height": 10, "batch_size": 256, "rounds": 16, "seed": 123}


def run_one_cycle(machine):
    """Run one cycle of the machine (one bundle per core). Returns True if a cycle was executed."""
    from problem import CoreState

    has_non_debug = False
    for core in machine.cores:
        if core.state != CoreState.RUNNING:
            continue
        if core.pc >= len(machine.program):
            core.state = CoreState.STOPPED
            continue
        instr = machine.program[core.pc]
        core.pc += 1
        machine.step(instr, core)
        if any(name != "debug" for name in instr.keys()):
            has_non_debug = True
    if has_non_debug:
        machine.cycle += 1
    return has_non_debug


def run_until_cycle(machine, max_cycle):
    """Run until machine.cycle reaches max_cycle or all cores stop."""
    from problem import CoreState

    while any(c.state == CoreState.RUNNING for c in machine.cores) and machine.cycle < max_cycle:
        run_one_cycle(machine)


def first_divergence_cycle(kernel_path, seed=None, output_dir=None):
    from problem import (
        Machine,
        Tree,
        Input,
        build_mem_image,
        reference_kernel2,
        N_CORES,
        DebugInfo,
    )

    instrs, params = load_kernel_and_params(kernel_path)
    if isinstance(instrs, dict):
        raise SystemExit("Kernel JSON must contain 'instrs' key.")
    seed = seed if seed is not None else params.get("seed", 123)
    forest_height = params["forest_height"]
    batch_size = params["batch_size"]
    rounds = params["rounds"]

    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    ref_states = list(reference_kernel2(mem))
    inp_values_p = ref_states[0][6]
    inp_indices_p = ref_states[0][5]
    n_vals = batch_size
    n_idx = batch_size

    def output_region_from_mem(mem_arr):
        return (
            tuple(mem_arr[inp_values_p : inp_values_p + n_vals]),
            tuple(mem_arr[inp_indices_p : inp_indices_p + n_idx]),
        )

    ref_outputs = [output_region_from_mem(ref_states[i]) for i in range(len(ref_states))]

    def output_region(m):
        return (
            tuple(m.mem[inp_values_p : inp_values_p + n_vals]),
            tuple(m.mem[inp_indices_p : inp_indices_p + n_idx]),
        )

    machine = Machine(
        list(mem),
        list(instrs),
        DebugInfo(scratch_map={}),
        n_cores=N_CORES,
        trace=False,
    )
    machine.enable_pause = False
    machine.enable_debug = False
    machine.prints = False

    from problem import CoreState

    for c in machine.cores:
        if c.state == CoreState.PAUSED:
            c.state = CoreState.RUNNING

    # Track which round we're in; output may be ref_outputs[r-1] or ref_outputs[r] during round r.
    first_div = None
    current_round = 0
    last_cycle = 0
    while any(c.state == CoreState.RUNNING for c in machine.cores):
        run_one_cycle(machine)
        last_cycle = machine.cycle
        out = output_region(machine)
        if current_round < len(ref_outputs):
            allowed = {ref_outputs[current_round]}
            if current_round > 0:
                allowed.add(ref_outputs[current_round - 1])
            if out not in allowed:
                first_div = last_cycle
                break
            if out == ref_outputs[current_round]:
                current_round += 1

    out = output_region(machine)
    match = (first_div is None and out == ref_outputs[-1])

    report = [
        f"Kernel: {kernel_path}",
        f"Params: forest_height={forest_height} batch_size={batch_size} rounds={rounds} seed={seed}",
        f"First divergence cycle: {first_div}",
        f"Last cycle: {last_cycle}",
        f"Final output matches reference: {match if first_div is None else False}",
    ]
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "report.txt"), "w") as f:
            f.write("\n".join(report) + "\n")

    for line in report:
        print(line)
    return first_div, last_cycle, match


def main():
    parser = argparse.ArgumentParser(
        description="Find first cycle where optimized kernel output diverges from reference."
    )
    parser.add_argument("--kernel", "-k", required=True, help="Path to kernel JSON (--save-kernel)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default from kernel JSON)")
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help="Write report to this dir (e.g. artifacts/root_cause/1D_first_divergence/)",
    )
    args = parser.parse_args()
    first_div, _, match = first_divergence_cycle(
        args.kernel, seed=args.seed, output_dir=args.output_dir
    )
    return 0 if first_div is None and match else 1


if __name__ == "__main__":
    sys.exit(main())
