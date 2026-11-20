#!/usr/bin/env python3
"""Emission ablation: build the kernel with each emission option and tabulate what it emits.

For every configuration this builds a kernel, proves it equal to its SSA program
(verification/kernel_dataflow.py), validates it numerically against the reference
(values and indices, or values only for configurations that set values_only), and reports
vector-op equivalents, loads, flow ops, the capacity floor for the emitted ops and the
schedule length. With --upstream each kernel also runs on the unmodified upstream simulator
(scripts/check_upstream.sh --kernel).

    uv run python -m scripts.emission_ablation                     # 16 rounds, greedy
    uv run python -m scripts.emission_ablation --rounds 4 --only baseline,merge-hash
    uv run python -m scripts.emission_ablation --upstream --markdown
    uv run python -m scripts.emission_ablation --cpsat-deterministic-time 10 --rounds 3 --batch-size 64

Two floors are reported, both capacity-only (precedence and scratch ignored) and both for
the ops the run emitted, never a bound on the problem:

  Floor               scripts.schedule_report's floor, with every select encoded as the
                      scheduler chose in this run (greedy puts every SelectChoice on flow).
  Floor, best split   the same with the selects re-divided between the flow engine and the
                      vector ALU in the best way (best_select_split_floor): what a scheduler
                      that balances the two could reach for this emission.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HERE = os.path.dirname(os.path.abspath(__file__))
# scripts/bisect.py shadows the standard library's bisect; keep this directory off the path.
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
sys.path.insert(0, ROOT)

from generate_ssa_kernel import save_kernel_to_json, simulator_cycles, validate_kernel  # noqa: E402
from kernel_build_args import build_kernel_with_args  # noqa: E402
from scripts import schedule_report as sr  # noqa: E402
from verification.kernel_dataflow import check  # noqa: E402

# name -> emission options (kernel_build_args.py "Emission options"). Order is the table order.
_SAFE = {"merge_hash_stages": True, "index_form": "mirrored"}
CONFIGS: dict[str, dict] = {
    "baseline": {},
    # One option at a time.
    "merge-hash": {"merge_hash_stages": True},
    "mirrored-index": {"index_form": "mirrored"},
    "prune-dead-index": {"prune_dead_index": True},
    "select-bits": {"cached_select": "bits"},
    "select-bits-flow": {"cached_select": "bits-flow"},
    "select-mobius": {"cached_select": "mobius"},
    "values-only": {"values_only": True},
    # Combinations whose output is unchanged (values and indices validated).
    "select-bits+mirrored": {"cached_select": "bits", "index_form": "mirrored"},
    "select-mobius+mirrored": {"cached_select": "mobius", "index_form": "mirrored"},
    "all-safe-bits": {**_SAFE, "cached_select": "bits"},
    "all-safe-bits-flow": {**_SAFE, "cached_select": "bits-flow"},
    "all-safe-mobius": {**_SAFE, "cached_select": "mobius"},
    # Level 4 cached as well (16-way select): on both visits, or only after the wrap (round 15).
    "depth4": {"tree_reduce_max_depth": 4},
    "depth4-after-wrap": {"tree_reduce_max_depth": 4, "depth4_select_rounds": "after-wrap"},
    "all-safe-bits+depth4": {**_SAFE, "cached_select": "bits", "tree_reduce_max_depth": 4},
    "all-safe-bits+depth4-after-wrap": {**_SAFE, "cached_select": "bits", "tree_reduce_max_depth": 4,
                                        "depth4_select_rounds": "after-wrap"},
    "all-safe-bits-flow+depth4-after-wrap": {**_SAFE, "cached_select": "bits-flow", "tree_reduce_max_depth": 4,
                                             "depth4_select_rounds": "after-wrap"},
    "all-safe-mobius+depth4-after-wrap": {**_SAFE, "cached_select": "mobius", "tree_reduce_max_depth": 4,
                                          "depth4_select_rounds": "after-wrap"},
    # The same with the final indices not stored (values only).
    "mirrored-index+values-only": {"index_form": "mirrored", "values_only": True},
    "all-bits+values-only": {**_SAFE, "cached_select": "bits", "values_only": True},
    "all-bits-flow+values-only": {**_SAFE, "cached_select": "bits-flow", "values_only": True},
    "all-mobius+values-only": {**_SAFE, "cached_select": "mobius", "values_only": True},
    "all-bits+depth4-after-wrap+values-only": {**_SAFE, "cached_select": "bits", "tree_reduce_max_depth": 4,
                                               "depth4_select_rounds": "after-wrap", "values_only": True},
    "all-mobius+depth4-after-wrap+values-only": {**_SAFE, "cached_select": "mobius", "tree_reduce_max_depth": 4,
                                                 "depth4_select_rounds": "after-wrap", "values_only": True},
    # Bits kept live instead of extracted: fewest ops, most scratch (may not fit).
    "select-bits-path": {"cached_select": "bits", "select_bit_source": "path"},
    "select-mobius-path": {"cached_select": "mobius", "select_bit_source": "path"},
    "all-safe-bits-path": {**_SAFE, "cached_select": "bits", "select_bit_source": "path"},
    # Further combinations for scheduler studies (docs/EMISSION_ABLATION.md section 3.5).
    "merge+mirrored": dict(_SAFE),
    "merge+bits": {"merge_hash_stages": True, "cached_select": "bits"},
    "merge+prune+bits": {"merge_hash_stages": True, "prune_dead_index": True, "cached_select": "bits"},
    "merge+prune+bits-flow": {"merge_hash_stages": True, "prune_dead_index": True, "cached_select": "bits-flow"},
    "merge+prune+mobius": {"merge_hash_stages": True, "prune_dead_index": True, "cached_select": "mobius"},
    "all-safe-bits+prune": {**_SAFE, "cached_select": "bits", "prune_dead_index": True},
}
# Configurations left out unless named with --only: the `path` ones are known not to fit the
# scratch budget with the default greedy scheduler at 16 rounds; the rest are extra
# combinations that the default table does not need.
NOT_BY_DEFAULT = ("select-bits-path", "select-mobius-path", "all-safe-bits-path",
                  "merge+mirrored", "merge+bits", "merge+prune+bits", "merge+prune+bits-flow",
                  "merge+prune+mobius", "all-safe-bits+prune")

SEED = 123
VEQ_PER_CYCLE = sr.CAP["valu"] + sr.CAP["alu"] / sr.VLEN


def select_costs(sched):
    """Vector-ALU cost of each SelectChoice if it is not a flow vselect: 1 multiply_add when
    its true value is `base + delta` (precomputed delta), else 2 (SUB + multiply_add)."""
    def_op = {}
    for op in sched["ssa_ops"]:
        if op.get("dest"):
            def_op.setdefault(int(op["dest"][1:].split("_")[0]), op)
    costs = {}
    for op in sched["ssa_ops"]:
        if op["opcode"] != "select_choice":
            continue
        _, true_val, base = (src["VReg"]["id"] for src in op["srcs"])
        producer = def_op.get(true_val)
        with_delta = (producer is not None and producer["opcode"] == "+"
                      and base in [src["VReg"]["id"] for src in producer["srcs"] if "VReg" in src])
        costs[op["id"]] = 1 if with_delta else 2
    return costs


def best_select_split_floor(sched, rep):
    """Capacity floor with the selects divided between flow and the vector ALU in the best way.

    Smallest T with loads <= 2T, stores <= 2T and, putting as many selects on the T flow
    slots as fit (the 2-op ones first), vector-op equivalents <= 7.5 T. Continuous in the
    ALU term (the split between the two ALUs is not rounded), so it can be one cycle below
    an integral floor; constants keep the encoding chosen in this run.
    """
    costs = select_costs(sched)
    lowered = [op for op in sched["ops"] if op.get("from_select_choice") is not None]
    on_valu = sum(1 for op in lowered if op["engine"] == "valu")
    on_flow = sum(1 for op in lowered if op["engine"] == "flow")
    base_veq = rep["emission"]["vector_op_equivalents"] - on_valu
    flow_fixed = rep["engines"]["flow"]["ops"] - on_flow
    one_op = sum(1 for c in costs.values() if c == 1)
    two_op = len(costs) - one_op
    t = max(1, rep["floor"]["terms"]["load"], rep["floor"]["terms"]["store"], flow_fixed)
    while True:
        free = t - flow_fixed
        flow_two = min(two_op, free)
        flow_one = min(one_op, free - flow_two)
        if base_veq + (one_op - flow_one) + 2 * (two_op - flow_two) <= VEQ_PER_CYCLE * t:
            return t
        t += 1


def run_config(name, options, args, workdir):
    """Build, prove, validate and measure one configuration. Returns a row dict."""
    sched_path = os.path.join(workdir, f"sched_{name}.json")
    kernel_path = os.path.join(workdir, f"kernel_{name}.json")
    overrides = dict(options)
    overrides.update(rounds=args.rounds, batch_size=args.batch_size, forest_height=args.forest_height,
                     use_greedy=args.cpsat_deterministic_time is None)
    if args.cpsat_deterministic_time is not None:
        overrides.update(cpsat_deterministic_time=args.cpsat_deterministic_time,
                         cpsat_time_limit=args.cpsat_time_limit)
    for key in ("select_choice", "const_choice", "valu_shift"):
        if getattr(args, key) != "optimize":
            overrides[key] = getattr(args, key)
    overrides.update(args.build_options)
    try:
        instrs, stats = build_kernel_with_args(None, verbose=False, output_json=sched_path, use_git_sha=False,
                                               **overrides)
    except RuntimeError as e:  # e.g. scratch budget exceeded in address assignment
        return {"name": name, "options": options, "error": str(e)}
    instrs = json.loads(json.dumps(list(instrs)))
    with open(sched_path) as f:
        sched = json.load(f)
    rep = sr.analyze(sched)
    proved = check(instrs, sched).ok
    values_only = bool(options.get("values_only"))
    validate_kwargs = {"values_only": True} if values_only else {}
    try:
        passed = (not stats["budget_exceeded"]) and validate_kernel(
            instrs, args.forest_height, args.batch_size, args.rounds, SEED, **validate_kwargs)
    except Exception as e:  # the simulator rejects the kernel, e.g. a write outside the scratch space
        print(f"  {name}: kernel crashed in the simulator: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        passed = False
    row = {
        "name": name,
        "options": options,
        "vector_op_equivalents": rep["emission"]["vector_op_equivalents"],
        "loads": rep["engines"]["load"]["ops"],
        "flow": rep["engines"]["flow"]["ops"],
        "stores": rep["engines"]["store"]["ops"],
        "floor": rep["floor"]["cycles"],
        "floor_best_select_split": best_select_split_floor(sched, rep),
        "floor_binds": rep["floor"].get("binding"),
        "cycles": stats["cycles"],
        "simulator_cycles": simulator_cycles(instrs)[0],
        "peak_live": stats.get("peak_live"),
        "budget_exceeded": stats["budget_exceeded"],
        "proved": proved,
        "passed": passed,
        "validated": "values only" if values_only else "values and indices",
        "upstream": None,
    }
    if args.upstream:
        save_kernel_to_json(instrs, kernel_path, args.forest_height, args.batch_size, args.rounds, SEED)
        out = subprocess.run([os.path.join(ROOT, "scripts", "check_upstream.sh"), "--kernel", kernel_path],
                             cwd=ROOT, capture_output=True, text=True)
        row["upstream"] = out.returncode == 0
    return row


def _yes(flag):
    return "-" if flag is None else ("yes" if flag else "NO")


def format_table(rows, markdown):
    """Render rows as an aligned text table or a Markdown table."""
    header = ["Configuration", "Vector-op equiv.", "Loads", "Flow", "Floor", "Binds", "Floor, best split", "Cycles (sched / sim)",
              "Peak live", "PROVED", "PASSED", "Upstream"]
    body = []
    for r in rows:
        if "error" in r:
            body.append([r["name"], "build failed: " + r["error"]] + [""] * (len(header) - 2))
            continue
        body.append([
            r["name"], f"{r['vector_op_equivalents']:,.2f}", f"{r['loads']:,}", f"{r['flow']:,}", f"{r['floor']:,}",
            "+".join(r["floor_binds"] or []), f"{r['floor_best_select_split']:,}",
            f"{r['cycles']:,} / {r['simulator_cycles']:,}"
            + (" (over budget)" if r["budget_exceeded"] else ""),
            str(r["peak_live"]), _yes(r["proved"]), _yes(r["passed"]) + (" (values only)" if r["validated"] == "values only" else ""),
            _yes(r["upstream"]),
        ])
    if markdown:
        lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
        lines += ["| " + " | ".join(row) + " |" for row in body]
        return "\n".join(lines)
    widths = [max(len(header[i]), *(len(row[i]) for row in body)) for i in range(len(header))]
    lines = ["  ".join(h.ljust(w) for h, w in zip(header, widths))]
    lines += ["  ".join(c.ljust(w) for c, w in zip(row, widths)) for row in body]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--forest-height", type=int, default=10)
    ap.add_argument("--only", help="comma-separated configuration names (default: all): " + ", ".join(CONFIGS))
    ap.add_argument("--cpsat-deterministic-time", type=float, default=None,
                    help="schedule with CP-SAT under this deterministic-time budget instead of greedy")
    ap.add_argument("--cpsat-time-limit", type=float, default=3600.0, help="wall-clock backstop for CP-SAT")
    ap.add_argument("--select-choice", choices=["optimize", "fma", "vselect"], default="optimize")
    ap.add_argument("--const-choice", choices=["optimize", "add_imm", "load"], default="optimize")
    ap.add_argument("--valu-shift", choices=["optimize", "alu", "valu"], default="optimize")
    ap.add_argument("--build-option", action="append", default=[], metavar="NAME=VALUE",
                    help="extra build option applied to every configuration, by its kernel_build_args.py name "
                         "(e.g. greedy_barriers=none, greedy_split_shift=true); VALUE is parsed as JSON when it "
                         "can be, else taken as a string. Repeatable")
    ap.add_argument("--upstream", action="store_true",
                    help="also run each kernel on the unmodified upstream simulator (needs height 10, batch 256, 16 rounds)")
    ap.add_argument("--markdown", action="store_true", help="print a Markdown table")
    ap.add_argument("--json", help="also write the rows to this JSON file")
    ap.add_argument("--keep", help="directory to keep scheduler JSONs and kernels in (default: a temporary directory)")
    args = ap.parse_args(argv)

    args.build_options = {}
    for item in args.build_option:
        name, sep, value = item.partition("=")
        if not sep or not name:
            ap.error(f"--build-option needs NAME=VALUE, got {item!r}")
        try:
            args.build_options[name] = json.loads(value)
        except ValueError:
            args.build_options[name] = value
    names = ([n for n in CONFIGS if n not in NOT_BY_DEFAULT] if not args.only
             else [n.strip() for n in args.only.split(",")])
    unknown = [n for n in names if n not in CONFIGS]
    if unknown:
        ap.error(f"unknown configuration(s) {unknown}; choose from {list(CONFIGS)}")
    if args.upstream and (args.rounds, args.batch_size, args.forest_height) != (16, 256, 10):
        ap.error("--upstream needs --rounds 16 --batch-size 256 --forest-height 10 (the only shape upstream tests)")

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        workdir = args.keep or tmp
        os.makedirs(workdir, exist_ok=True)
        for name in names:
            rows.append(run_config(name, CONFIGS[name], args, workdir))
            print(f"  done: {name}", file=sys.stderr, flush=True)
    scheduler = "greedy" if args.cpsat_deterministic_time is None else f"CP-SAT (deterministic time {args.cpsat_deterministic_time:g})"
    print(f"rounds={args.rounds} batch={args.batch_size} height={args.forest_height} scheduler={scheduler}"
          + "".join(f" {k}={getattr(args, k)}" for k in ("select_choice", "const_choice", "valu_shift")
                    if getattr(args, k) != "optimize")
          + "".join(f" {k}={v}" for k, v in args.build_options.items()))
    print(format_table(rows, args.markdown))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2)
    ok = all("error" not in r and r["proved"] and r["passed"] and r["upstream"] is not False for r in rows)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
