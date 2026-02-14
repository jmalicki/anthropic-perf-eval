"""Shared kernel build argument group and kwargs for Rust.

Use add_kernel_build_args(parser) to add options to any argparse parser.
Use args_to_build_kwargs(args, **overrides) to get kwargs for scheduler_rs.build_tick_tock_kernel_ssa.
Callers do: instrs, stats = scheduler_rs.build_tick_tock_kernel_ssa(**args_to_build_kwargs(args))

Rust rejects unknown kwargs via PyO3. Both generate_ssa_kernel.py and bisection scripts import from here.
"""

import argparse
import json
import os
import subprocess
import sys
from typing import Any, Optional

DEFAULT_SCHEDULER_JSON = "scheduler_output.json"


def _n_nodes(forest_height: int) -> int:
    return (1 << forest_height) - 1


def _get_git_sha() -> Optional[str]:
    """Return git HEAD SHA (short) or None."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if out.returncode == 0 and out.stdout:
            return out.stdout.strip()[:12]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def add_kernel_build_args(parser: argparse.ArgumentParser) -> argparse._ArgumentGroup:
    """Add kernel build options to parser. Returns the argument group."""
    group = parser.add_argument_group("Kernel build")

    # Problem parameters
    group.add_argument("--forest-height", type=int, default=10,
                       help="Tree depth (default: 10, gives 2047 nodes)")
    group.add_argument("--n-nodes", type=int, default=None,
                       help="Number of tree nodes (default: 2^forest_height - 1)")
    group.add_argument("--batch-size", type=int, default=256,
                       help="Parallel traversals; must be multiple of 8 (default: 256)")
    group.add_argument("--rounds", type=int, default=16,
                       help="Hash rounds per traversal (default: 16)")

    # Scheduler options
    group.add_argument("--cpsat-time-limit", type=float, default=3600.0,
                       help="CP-SAT time limit in seconds (default: 3600)")
    group.add_argument("--per-section-optimize-secs", type=float, default=None, metavar="SECS",
                       help="Per-section CP-SAT optimize time; 0 or omit = feasibility only")
    group.add_argument("--use-greedy", action="store_true",
                       help="Use greedy scheduler instead of CP-SAT")
    group.add_argument("--enable-valu-shift", action="store_true", default=True,
                       help="Enable VALU->ALU shifting (default: True)")
    group.add_argument("--no-valu-shift", action="store_false", dest="enable_valu_shift",
                       help="Disable VALU->ALU shifting")
    group.add_argument("--no-eliminate-vbroadcast", action="store_true", dest="no_eliminate_vbroadcast",
                       default=False,
                       help="Disable vbroadcast elimination: always emit vbroadcast, shifted ALU ops read vector (default: elimination enabled when VALU shift enabled)")
    group.add_argument("--tree-reduce-max-depth", type=int, default=3, metavar="N",
                       help="Max tree depth for tree-reduce (2..4, default: 3)")
    group.add_argument("--const-choice", choices=["optimize", "add_imm", "load"], default="optimize",
                       help="CP-SAT ConstChoice (default: optimize)")
    group.add_argument("--select-choice", choices=["optimize", "fma", "vselect"], default="optimize",
                       help="CP-SAT SelectChoice (default: optimize)")
    group.add_argument("--valu-shift", choices=["optimize", "alu", "valu"], default="optimize",
                       help="CP-SAT VALU shift (default: optimize)")
    group.add_argument("--force-zero-gap", action="store_true",
                       help="Force 0 integral gap for proven optimality")
    group.add_argument("--enable-asap-alap-tightening", action=argparse.BooleanOptionalAction,
                       default=True,
                       help="Enable ASAP/ALAP slot tightening (default: enabled)")
    group.add_argument("--enable-scratch-tightening", action="store_true",
                       help="Enable scratch-aware tightening (default: disabled; non-default debugging option)")
    group.add_argument("--scheduling-pressure-policy",
                       choices=["greedy", "throttled", "gated", "adaptive", "cascade"],
                       default="adaptive",
                       help="Scheduling pressure policy (default: adaptive)")

    # Constraint / CSE options
    import scheduler_rs
    _constraint_names = ", ".join(scheduler_rs.optional_constraint_group_names())
    group.add_argument("--disable-cpsat-constraints", type=str, default=None,
                       metavar="GROUP1,GROUP2,...",
                       help=f"Disable CP-SAT constraint groups. Valid: {_constraint_names}")
    group.add_argument("--no-cse-groups", action="store_true",
                       help="Disable CSE groups (debugging depth-3)")
    group.add_argument("--enable-trace-ids", action="store_true",
                       help="Enable trace IDs for semantic debugging (trace_input/trace_output)")
    group.add_argument("--no-verify-scratch-conflict", action="store_false", dest="verify_same_cycle_scratch",
                       default=True,
                       help="Disable same-cycle scratch write conflict check (default: enabled)")
    group.add_argument("--bulk-move", action="store_true", dest="enable_bulk_move",
                       default=False,
                       help="Enable bulk-move compaction after CP-SAT (off by default: known buggy, can cause same-cycle scratch conflicts)")

    # Output (passed to Rust)
    group.add_argument("--output-json", type=str, default=None,
                       help=f"Schedule JSON path (default: {DEFAULT_SCHEDULER_JSON})")
    group.add_argument("--comment", type=str, default=None, help="Comment for schedule JSON")
    group.add_argument("--liveness-detail", type=str, default=None,
                       choices=["none", "summary", "full"],
                       help="Liveness detail in JSON")

    return group


def args_to_build_kwargs(
    args=None,
    *,
    output_json: Optional[str] = None,
    use_git_sha: bool = True,
    **overrides,
) -> dict:
    """
    Produce kwargs for scheduler_rs.build_tick_tock_kernel_ssa from args and/or overrides.

    args: parsed namespace from add_kernel_build_args, or None to use overrides only.
    output_json: override output path.
    use_git_sha: include git SHA in run_options (default True).
    **overrides: kwargs that override args (e.g. rounds=4, batch_size=8).

    Returns dict suitable for scheduler_rs.build_tick_tock_kernel_ssa(**kwargs).
    Rust will reject unknown keys.
    """
    import scheduler_rs

    def get(name, default=None):
        if name in overrides:
            return overrides[name]
        if args is not None:
            return getattr(args, name, default)
        return default

    forest_height = get("forest_height", 10)
    n_nodes_val = get("n_nodes")
    n_nodes = n_nodes_val if n_nodes_val is not None else _n_nodes(forest_height)
    batch_size = get("batch_size", 256)
    rounds = get("rounds", 16)
    use_cpsat = not get("use_greedy", False)
    cpsat_time_limit = get("cpsat_time_limit", 3600.0)
    # force_zero_gap defaults to 2h time limit, but allow explicit override
    force_zero_gap = get("force_zero_gap", False)
    if force_zero_gap and cpsat_time_limit == 3600.0:  # Only auto-set if using default
        cpsat_time_limit = 7200
    per_section = get("per_section_optimize_secs")
    per_section_optimize = None if per_section is None or per_section <= 0 else per_section
    tree_reduce_max_depth = max(2, min(4, get("tree_reduce_max_depth", 3)))

    disable_raw = get("disable_cpsat_constraints") or os.environ.get("CPSAT_DISABLE_OPTIONAL_CONSTRAINTS")
    optional_constraint_groups_to_disable = None
    if disable_raw:
        optional_constraint_groups_to_disable = []
        for s in disable_raw.split(","):
            t = s.strip()
            if t:
                optional_constraint_groups_to_disable.append(scheduler_rs.OptionalConstraintGroup.from_str(t))

    out_path = output_json if output_json is not None else get("output_json")
    if out_path is None:
        out_path = DEFAULT_SCHEDULER_JSON

    run_options = {
        "forest_height": forest_height,
        "n_nodes": n_nodes,
        "batch_size": batch_size,
        "rounds": rounds,
        "use_cpsat": use_cpsat,
        "cpsat_time_limit": cpsat_time_limit,
        "per_section_optimize_secs": per_section_optimize,
        "enable_valu_shift": get("enable_valu_shift", True),
        "tree_reduce_max_depth": tree_reduce_max_depth,
        "const_choice": get("const_choice", "optimize"),
        "select_choice": get("select_choice", "optimize"),
        "valu_shift": get("valu_shift", "optimize"),
        "force_zero_gap": get("force_zero_gap", False) and use_cpsat,
        "enable_asap_alap_tightening": get("enable_asap_alap_tightening", True) and use_cpsat,
        "enable_scratch_tightening": get("enable_scratch_tightening", False) and use_cpsat,
        "scheduling_pressure_policy": get("scheduling_pressure_policy", "adaptive"),
        "enable_cse_groups": not get("no_cse_groups", False),
        "enable_trace_ids": get("enable_trace_ids", False),
        "enable_bulk_move": get("enable_bulk_move", False),
        "enable_vbroadcast_elimination": not get("no_eliminate_vbroadcast", False),
    }
    git_sha = _get_git_sha() if use_git_sha else None
    enable_cse = False if get("no_cse_groups", False) else None

    return {
        "forest_height": forest_height,
        "n_nodes": n_nodes,
        "batch_size": batch_size,
        "rounds": rounds,
        "enable_valu_shift": get("enable_valu_shift", True),
        "verbose": True,  # caller can override
        "use_cpsat": use_cpsat,
        "cpsat_time_limit": cpsat_time_limit,
        "per_section_optimize_secs": per_section_optimize,
        "output_json": out_path,
        "tree_reduce_max_depth": tree_reduce_max_depth,
        "const_choice": get("const_choice", "optimize"),
        "select_choice": get("select_choice", "optimize"),
        "valu_shift": get("valu_shift", "optimize"),
        "force_zero_gap": get("force_zero_gap", False) if use_cpsat else None,
        "enable_asap_alap_tightening": get("enable_asap_alap_tightening", True) if use_cpsat else None,
        "enable_scratch_tightening": get("enable_scratch_tightening", False) if use_cpsat else None,
        "run_options_json": json.dumps(run_options),
        "git_sha": git_sha,
        "comment": get("comment"),
        "liveness_detail": get("liveness_detail"),
        "scheduling_pressure_policy": get("scheduling_pressure_policy", "adaptive"),
        "optional_constraint_groups_to_disable": optional_constraint_groups_to_disable,
        "enable_cse_groups": enable_cse,
        "enable_trace_ids": get("enable_trace_ids", False),
        "verify_same_cycle_scratch": get("verify_same_cycle_scratch", True),
        "enable_bulk_move": get("enable_bulk_move", False),
        "enable_vbroadcast_elimination": False if get("no_eliminate_vbroadcast", False) else None,
    }


def build_kernel_with_args(
    args: Optional[Any] = None,
    *,
    verbose: bool = True,
    output_json: Optional[str] = None,
    use_git_sha: bool = True,
    **overrides: Any,
):
    """
    Call scheduler_rs.build_tick_tock_kernel_ssa with options from args and/or overrides.

    Convenience wrapper: uses args_to_build_kwargs then calls Rust.
    """
    import scheduler_rs

    kwargs = args_to_build_kwargs(args, output_json=output_json, use_git_sha=use_git_sha, **overrides)
    kwargs["verbose"] = verbose
    return scheduler_rs.build_tick_tock_kernel_ssa(**kwargs)
