#!/usr/bin/env python3
"""Shared helper for bisection scripts: build generate_ssa_kernel.py argv and run.

Used by bisect.py, bisect_cpsat_validation.py, and bisect_minimal_fma.py so
CLI construction and subprocess invocation live in one place.
"""

import re
import subprocess
import sys
from typing import Optional


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
):
    """Run generate_ssa_kernel.py with given params.

    Returns (success, stderr_snippet) by default. If return_full_output=True,
    returns (success, stderr_snippet, stdout, stderr) so callers can parse solver status.

    disable_constraints: None (no disable), or list of group names (e.g. ["capacity_valu"]),
    or comma-separated string (parsed and passed as --disable-cpsat-constraints).
    """
    nn = n_nodes_val if n_nodes_val is not None else n_nodes(forest_height)
    cmd = [
        sys.executable,
        script_path,
        "--rounds",
        str(rounds),
        "--batch-size",
        str(batch_size),
        "--forest-height",
        str(forest_height),
        "--n-nodes",
        str(nn),
        "--seed",
        str(seed),
        "--const-choice",
        const_choice,
        "--select-choice",
        select_choice,
        "--valu-shift",
        valu_shift,
    ]
    if validate:
        cmd.append("--validate")
    if quiet:
        cmd.append("-q")
    if disable_constraints is not None:
        if isinstance(disable_constraints, str):
            groups = [s.strip() for s in disable_constraints.split(",") if s.strip()]
        else:
            groups = list(disable_constraints)
        if groups:
            cmd.extend(["--disable-cpsat-constraints", ",".join(groups)])
    if use_greedy:
        cmd.append("--use-greedy")
    else:
        cmd.extend(["--cpsat-time-limit", str(cpsat_time_limit)])
    timeout = timeout_sec if timeout_sec is not None else (cpsat_time_limit + 30 if not use_greedy else 60)
    result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True, timeout=timeout)
    snippet = ""
    if result.stderr:
        lines = result.stderr.strip().splitlines()
        snippet = "\n".join(lines[-8:]) if len(lines) > 8 else result.stderr.strip()
    if return_full_output:
        return result.returncode == 0, snippet, result.stdout or "", result.stderr or ""
    return result.returncode == 0, snippet
