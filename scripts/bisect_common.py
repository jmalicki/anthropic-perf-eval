#!/usr/bin/env python3
"""Shared helper for bisection scripts: run kernel build and validation.

Used by bisect.py, bisect_cpsat_validation.py, and bisect_minimal_fma.py.
Calls scheduler_rs in-process via kernel_build_args.build_kernel_with_args.
"""

import os
import re
import subprocess
import sys
from types import SimpleNamespace
from typing import Optional

# Import from project root (scripts/ is a subdir)
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from kernel_build_args import build_kernel_with_args


def parse_solver_status(stdout: str, stderr: str) -> Optional[str]:
    """Parse CP-SAT / Rust solver status from combined output.

    Returns one of: 'OPTIMAL', 'FEASIBLE', 'INFEASIBLE', 'UNKNOWN', or None if not found.
    Rust prints to stderr: "SSA CP-SAT result: ... status Optimal" or
    "SSA CP-SAT: no solution found, status Infeasible". CP-SAT may print "status: FEASIBLE" etc.
    """
    text = (stdout or "") + "\n" + (stderr or "")
    line_lower = text.lower()
    # Prefer Rust-style "status Infeasible" / "status Optimal" (Debug format)
    rust_status = re.search(
        r"status\s+(optimal|feasible|infeasible|unknown)",
        line_lower,
        re.IGNORECASE,
    )
    if rust_status:
        return rust_status.group(1).upper()
    # CP-SAT / log style "status: FEASIBLE"
    if "status:" in line_lower:
        if "optimal" in line_lower:
            return "OPTIMAL"
        if "feasible" in line_lower:
            return "FEASIBLE"
        if "infeasible" in line_lower:
            return "INFEASIBLE"
        if "unknown" in line_lower:
            return "UNKNOWN"
    return None


def n_nodes(forest_height: int) -> int:
    """Return tree node count for a given forest height."""
    return (1 << forest_height) - 1


def run_one(
    project_root: str,
    script_path: str,
    *,
    rounds: int,
    batch_size: int,
    forest_height: int,
    n_nodes_val: Optional[int] = None,
    use_greedy: bool = False,
    cpsat_time_limit: float = 60.0,
    seed: int = 123,
    const_choice: str = "optimize",
    select_choice: str = "optimize",
    valu_shift: str = "optimize",
    disable_constraints: Optional[list[str] | str] = None,
    validate: bool = True,
    quiet: bool = True,
    timeout_sec: Optional[float] = None,
    return_full_output: bool = False,
    tree_reduce_max_depth: Optional[int] = None,
    no_cse_groups: bool = False,
):
    """Build kernel and optionally validate. Calls Rust in-process via kernel_build_args.

    Returns (success, stderr_snippet) by default. If return_full_output=True,
    returns (success, stderr_snippet, stdout, stderr) - stdout/stderr are empty for in-process.

    script_path: unused (kept for backward compat); build is in-process.
    disable_constraints: None, or list of group names, or comma-separated string.
    """
    import random

    nn = n_nodes_val if n_nodes_val is not None else n_nodes(forest_height)
    disable_str = None
    if disable_constraints is not None:
        if isinstance(disable_constraints, str):
            disable_str = disable_constraints
        else:
            disable_str = ",".join(disable_constraints)

    try:
        instrs, stats = build_kernel_with_args(
            args=None,
            verbose=not quiet,
            output_json=None,
            use_git_sha=False,
            forest_height=forest_height,
            n_nodes=nn,
            batch_size=batch_size,
            rounds=rounds,
            use_greedy=use_greedy,
            cpsat_time_limit=cpsat_time_limit,
            const_choice=const_choice,
            select_choice=select_choice,
            valu_shift=valu_shift,
            disable_cpsat_constraints=disable_str,
            tree_reduce_max_depth=tree_reduce_max_depth,
            no_cse_groups=no_cse_groups,
        )
    except Exception as e:
        snippet = str(e)
        if return_full_output:
            return False, snippet, "", snippet
        return False, snippet

    if not validate:
        if return_full_output:
            return True, "", "", ""
        return True, ""

    # Validate
    from generate_ssa_kernel import validate_kernel

    random.seed(seed)
    correct = validate_kernel(instrs, forest_height, batch_size, rounds, seed)
    snippet = "Validation failed" if not correct else ""
    if return_full_output:
        return correct, snippet, "", ""
    return correct, snippet
