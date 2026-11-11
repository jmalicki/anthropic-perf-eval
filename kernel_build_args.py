"""Shared kernel build argument group and kwargs for Rust.

Use add_kernel_build_args(parser) to add options to any argparse parser.
Use args_to_build_kwargs(args, **overrides) to get kwargs for scheduler_rs.build_tick_tock_kernel_ssa.
Callers do: instrs, stats = scheduler_rs.build_tick_tock_kernel_ssa(**args_to_build_kwargs(args))

Rust rejects unknown kwargs via PyO3. Both generate_ssa_kernel.py and bisection scripts import from here.
"""

import argparse
import json
import math
import os
import subprocess
import sys
from typing import Any, Optional

DEFAULT_SCHEDULER_JSON = "scheduler_output.json"


def n_nodes_for_height(forest_height: int) -> int:
    """Node count of the forest, matching problem.Tree.generate(height).

    A tree of height h has levels 0..h, so 2**(h+1) - 1 nodes (2047 for height 10).
    """
    return (1 << (forest_height + 1)) - 1


_n_nodes = n_nodes_for_height


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


def _positive_finite_float(text: str) -> float:
    """argparse type: a float that is finite and greater than zero."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not (math.isfinite(value) and value > 0):
        raise argparse.ArgumentTypeError(f"must be a positive finite number, got {text!r}")
    return value


def _positive_int(text: str) -> int:
    """argparse type: an integer greater than zero."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {text!r}")
    return value


def _slack_or_none(text: str):
    """argparse type: a non-negative integer, or 'none' to switch the cap off."""
    if text.strip().lower() in ("none", "off"):
        return "none"
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer or 'none', got {text!r}") from None
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be 0 or more (or 'none'), got {text!r}")
    return value


def _name_list(text: str) -> list:
    """argparse type: comma-separated names, e.g. subsolver names. Blank entries are dropped."""
    return [name.strip() for name in text.split(",") if name.strip()]


# CP-SAT tuning switches for the full-kernel solve: (argparse dest / override name, CpsatTuning key).
# An option left at None is not sent, so the Rust default applies; scheduler_rs.resolve_cpsat_tuning
# fills in the defaults for run_options. Defaults live in one place (CpsatTuning::default in Rust).
# The string "none" is sent as JSON null (an option that is switched off rather than left unset).
CPSAT_TUNING_OPTIONS = [
    ("cpsat_workers", "workers"),
    ("cpsat_interleave", "interleave_search"),
    ("cpsat_subsolvers", "subsolvers"),
    ("cpsat_extra_subsolvers", "extra_subsolvers"),
    ("cpsat_ignore_subsolvers", "ignore_subsolvers"),
    ("cpsat_objective", "objective"),
    ("cpsat_greedy_alap_slack", "greedy_alap_slack"),
    ("cpsat_horizon_max_latency", "horizon_at_least_max_latency_path"),
    ("cpsat_horizon_num_ops", "horizon_at_least_num_ops"),
    ("cpsat_makespan_interval", "makespan_interval"),
    ("cpsat_push_tasks_toward_start", "push_all_tasks_toward_start"),
]


def add_kernel_build_args(parser: argparse.ArgumentParser) -> argparse._ArgumentGroup:
    """Add kernel build options to parser. Returns the argument group."""
    group = parser.add_argument_group("Kernel build")

    # Problem parameters
    group.add_argument("--forest-height", type=int, default=10,
                       help="Tree depth (default: 10, gives 2047 nodes)")
    group.add_argument("--n-nodes", type=int, default=None,
                       help="Number of tree nodes (default: 2^(forest_height+1) - 1, i.e. 2047 for height 10)")
    group.add_argument("--batch-size", type=int, default=256,
                       help="Parallel traversals; must be multiple of 8 (default: 256)")
    group.add_argument("--rounds", type=int, default=16,
                       help="Hash rounds per traversal (default: 16)")

    # Scheduler options
    group.add_argument("--cpsat-time-limit", type=float, default=3600.0,
                       help="CP-SAT time limit in seconds (default: 3600)")
    group.add_argument("--cpsat-deterministic-time", type=_positive_finite_float, default=None, metavar="DTIME",
                       help="CP-SAT deterministic-time budget per solve. Unlike --cpsat-time-limit it does not "
                            "depend on machine load, so the same command gives the same kernel as long as the "
                            "solve ends on this budget. --cpsat-time-limit remains as a wall-clock backstop; if "
                            "it fires first the result depends on load again (a warning is printed), so keep "
                            "it generous. Defaults to CPSAT_MAX_DETERMINISTIC_TIME if that is set.")
    # CP-SAT tuning switches (full-kernel solve only). Unset = the default named in the help.
    group.add_argument("--cpsat-workers", type=_positive_int, default=None, metavar="N",
                       help="CP-SAT num_workers (default: 8)")
    group.add_argument("--cpsat-interleave", action=argparse.BooleanOptionalAction, default=None,
                       help="CP-SAT interleave_search (default: on). On = deterministic round-robin of "
                            "subsolvers; OR-Tools 9.15 then starts no probing, shaving, lb_tree_search or "
                            "objective_lb_search worker. --no-cpsat-interleave runs the usual parallel "
                            "portfolio, whose result is NOT reproducible from run to run.")
    group.add_argument("--cpsat-subsolvers", type=_name_list, default=None, metavar="NAME,...",
                       help="Replace CP-SAT's list of full-problem subsolvers (default: CP-SAT's own list)")
    group.add_argument("--cpsat-extra-subsolvers", type=_name_list, default=None, metavar="NAME,...",
                       help="Add full-problem subsolvers, e.g. objective_lb_search,objective_shaving")
    group.add_argument("--cpsat-ignore-subsolvers", type=_name_list, default=None, metavar="NAME,...",
                       help="Remove subsolvers (full-problem or LNS) by name")
    group.add_argument("--cpsat-objective", choices=["makespan+tiebreak", "makespan"], default=None,
                       help="CP-SAT objective (default: makespan+tiebreak = scale*makespan + sum of op "
                            "completion cycles). 'makespan' drops the tie-breaker: the solve stops once the "
                            "makespan is proved, and the schedule returned is not left-packed.")
    group.add_argument("--cpsat-greedy-alap-slack", type=_slack_or_none, default=None, metavar="K|none",
                       help="Cap each op's latest cycle at its greedy cycle + K (default: 15). The cap confines "
                            "the search to a neighbourhood of the greedy schedule; 'none' removes it.")
    group.add_argument("--cpsat-horizon-max-latency", action=argparse.BooleanOptionalAction, default=None,
                       help="Raise the horizon (greedy makespan + 20) to the max-latency critical path, i.e. "
                            "every shiftable op at its worst-case latency (default: on). "
                            "--no-cpsat-horizon-max-latency drops the guard; it relies on the greedy schedule "
                            "being feasible.")
    group.add_argument("--cpsat-horizon-num-ops", action=argparse.BooleanOptionalAction, default=None,
                       help="Raise the horizon (greedy makespan + 20) to the number of ops (default: on). "
                            "--no-cpsat-horizon-num-ops keeps the tight horizon; it relies on the greedy "
                            "schedule being feasible.")
    group.add_argument("--cpsat-makespan-interval", action=argparse.BooleanOptionalAction, default=None,
                       help="Add an interval [makespan, horizon) with demand = capacity to each engine "
                            "cumulative, as OR-Tools' rcpsp_sat.py does, so the LP relaxation and energy cuts "
                            "see the makespan (default: off)")
    group.add_argument("--cpsat-push-tasks-toward-start", action=argparse.BooleanOptionalAction, default=None,
                       help="CP-SAT push_all_tasks_toward_start (experimental solver parameter; default: off)")
    group.add_argument("--per-section-optimize-secs", type=float, default=None, metavar="SECS",
                       help="Per-section CP-SAT optimize time; 0 or omit = feasibility only")
    group.add_argument("--use-greedy", action="store_true",
                       help="Use greedy scheduler instead of CP-SAT")
    group.add_argument("--enable-valu-shift", action="store_true", default=True,
                       help="Enable VALU->ALU shifting (default: True)")
    group.add_argument("--no-valu-shift", action="store_false", dest="enable_valu_shift",
                       help="Disable VALU->ALU shifting")
    group.add_argument("--enable-eliminate-vbroadcast", action="store_true", dest="enable_eliminate_vbroadcast",
                       default=False,
                       help="Enable vbroadcast elimination (experimental, buggy: can cause wrong results when shifted ALU reads scalar; off by default)")
    group.add_argument("--tree-reduce-max-depth", type=int, default=3, metavar="N",
                       help="Max tree depth for tree-reduce (2..4, default: 3)")
    group.add_argument("--const-choice", choices=["optimize", "add_imm", "load"], default="optimize",
                       help="CP-SAT ConstChoice (default: optimize)")
    group.add_argument("--select-choice", choices=["optimize", "fma", "vselect"], default="optimize",
                       help="CP-SAT SelectChoice (default: optimize)")
    group.add_argument("--valu-shift", choices=["optimize", "alu", "valu"], default="optimize",
                       help="CP-SAT VALU shift (default: optimize)")
    group.add_argument("--force-zero-gap", action="store_true",
                       help="Force 0 integral gap: prove the schedule optimal for the emitted ops (not for the problem)")
    group.add_argument("--asap-alap-tightening", choices=["none", "simple", "symmetry-aware", "energetic"],
                       default="energetic",
                       help="ASAP/ALAP tightening mode: none, simple (precedence only), symmetry-aware, or energetic (default)")
    group.add_argument("--enable-scratch-tightening", action="store_true",
                       help="Enable scratch-aware tightening (default: disabled; non-default debugging option)")
    group.add_argument("--symmetry-precedence-integrated", action="store_true",
                       help="Integrate precedence into symmetry ordering tie-breaking (experimental)")
    group.add_argument("--energetic-alap-pulls", action="store_true",
                       help="Enable ALAP pulls in energetic tightening (experimental; T7-two-step risk)")
    group.add_argument("--energetic-bounded-push", action="store_true", dest="energetic_bounded_push",
                       help="Cap ASAP push / ALAP pull magnitude per iteration (experimental)")
    group.add_argument("--energetic-bounded-push-k", type=int, default=None, metavar="K",
                       help="Cap push/pull to K cycles when --energetic-bounded-push (default: 2)")
    group.add_argument("--precedence-neutral-horizon", action="store_true",
                       help="Use precedence-neutral horizon (2x critical path) instead of greedy-based")
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
                       help="Enable bulk-move compaction after CP-SAT. Off by default: CP-SAT should do it; bulk-move masks bugs and is buggy (same-cycle scratch conflicts). Experimentation only.")
    group.add_argument("--greedy-full-bounds", action="store_true", dest="greedy_use_full_bounds",
                       default=False,
                       help="With --use-greedy, use Full bounds (ASAP deferral, clamping, capacity deferral). For comparing PolicyOnly vs Full greedy schedules (feasibility investigation).")

    # Hint passes (run after GreedySchedule; overwrite greedy_decisions only). Off by default.
    group.add_argument("--hint-passes", action="store_true", dest="hint_passes",
                       default=False,
                       help="Enable all three hint passes (ConstChoice, SelectChoice, VALU shift). For ablation.")
    group.add_argument("--const-choice-hint-pass", action="store_true", dest="enable_const_choice_hint_pass",
                       default=False,
                       help="ConstChoice hint pass: set add_imm where Flow has slack and Load is tight.")
    group.add_argument("--select-choice-hint-pass", action="store_true", dest="enable_select_choice_hint_pass",
                       default=False,
                       help="SelectChoice hint pass: prefer FMA over vselect when VALU saturated.")
    group.add_argument("--valu-shift-hint-pass", action="store_true", dest="enable_valu_shift_hint_pass",
                       default=False,
                       help="VALU shift hint pass: add ops to shifted_ops where ALU has slack.")
    group.add_argument("--schedule-with-fixed-choices", action="store_true", dest="enable_schedule_with_fixed_choices",
                       default=False,
                       help="After hint passes, re-run greedy with encodings fixed to fill freed slots.")

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
    # Effective deterministic budget: option/override first, else the environment variable
    # the solver would fall back to, so run_options records what was actually applied.
    cpsat_deterministic_time = get("cpsat_deterministic_time")
    if cpsat_deterministic_time is None and os.environ.get("CPSAT_MAX_DETERMINISTIC_TIME"):
        cpsat_deterministic_time = _positive_finite_float(os.environ["CPSAT_MAX_DETERMINISTIC_TIME"])
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

    # CP-SAT tuning: send only what was set; record the effective values (defaults filled in by Rust).
    cpsat_tuning = {
        key: (None if get(dest) == "none" else get(dest))
        for dest, key in CPSAT_TUNING_OPTIONS
        if get(dest) is not None
    }
    cpsat_tuning_json = json.dumps(cpsat_tuning) if cpsat_tuning else None
    cpsat_tuning_effective = json.loads(scheduler_rs.resolve_cpsat_tuning(cpsat_tuning_json))

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
        "cpsat_deterministic_time": cpsat_deterministic_time,
        "cpsat_tuning": cpsat_tuning_effective,
        "per_section_optimize_secs": per_section_optimize,
        "enable_valu_shift": get("enable_valu_shift", True),
        "tree_reduce_max_depth": tree_reduce_max_depth,
        "const_choice": get("const_choice", "optimize"),
        "select_choice": get("select_choice", "optimize"),
        "valu_shift": get("valu_shift", "optimize"),
        "force_zero_gap": get("force_zero_gap", False) and use_cpsat,
        "asap_alap_tightening": get("asap_alap_tightening", "energetic"),
        "enable_scratch_tightening": get("enable_scratch_tightening", False) and use_cpsat,
        "symmetry_precedence_integrated": get("symmetry_precedence_integrated", False),
        "energetic_alap_pulls": get("energetic_alap_pulls", False),
        "energetic_bounded_push": get("energetic_bounded_push", False),
        "energetic_bounded_push_k": get("energetic_bounded_push_k"),
        "precedence_neutral_horizon": get("precedence_neutral_horizon", False),
        "scheduling_pressure_policy": get("scheduling_pressure_policy", "adaptive"),
        "enable_cse_groups": not get("no_cse_groups", False),
        "enable_trace_ids": get("enable_trace_ids", False),
        "enable_bulk_move": get("enable_bulk_move", False),
        "enable_vbroadcast_elimination": get("enable_eliminate_vbroadcast", False),
        "greedy_use_full_bounds": get("greedy_use_full_bounds", False),
        "enable_const_choice_hint_pass": get("hint_passes", False) or get("enable_const_choice_hint_pass", False),
        "enable_select_choice_hint_pass": get("hint_passes", False) or get("enable_select_choice_hint_pass", False),
        "enable_valu_shift_hint_pass": get("hint_passes", False) or get("enable_valu_shift_hint_pass", False),
        "enable_schedule_with_fixed_choices": get("enable_schedule_with_fixed_choices", False),
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
        "cpsat_deterministic_time": cpsat_deterministic_time,
        "cpsat_tuning_json": cpsat_tuning_json,
        "per_section_optimize_secs": per_section_optimize,
        "output_json": out_path,
        "tree_reduce_max_depth": tree_reduce_max_depth,
        "const_choice": get("const_choice", "optimize"),
        "select_choice": get("select_choice", "optimize"),
        "valu_shift": get("valu_shift", "optimize"),
        "force_zero_gap": get("force_zero_gap", False) if use_cpsat else None,
        "asap_alap_tightening": get("asap_alap_tightening", "energetic"),
        "enable_scratch_tightening": get("enable_scratch_tightening", False) if use_cpsat else None,
        "symmetry_precedence_integrated": get("symmetry_precedence_integrated", False),
        "energetic_alap_pulls": get("energetic_alap_pulls", False),
        "energetic_bounded_push": get("energetic_bounded_push", False),
        "energetic_bounded_push_k": get("energetic_bounded_push_k"),
        "precedence_neutral_horizon": get("precedence_neutral_horizon", False),
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
        "enable_vbroadcast_elimination": True if get("enable_eliminate_vbroadcast", False) else None,
        "greedy_use_full_bounds": get("greedy_use_full_bounds", False),
        "enable_const_choice_hint_pass": get("hint_passes", False) or get("enable_const_choice_hint_pass", False),
        "enable_select_choice_hint_pass": get("hint_passes", False) or get("enable_select_choice_hint_pass", False),
        "enable_valu_shift_hint_pass": get("hint_passes", False) or get("enable_valu_shift_hint_pass", False),
        "enable_schedule_with_fixed_choices": get("enable_schedule_with_fixed_choices", False),
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
