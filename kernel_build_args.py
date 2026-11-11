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

    # Emission options (what ops the kernel contains). Defaults leave the emission unchanged;
    # see docs/EMISSION_ABLATION.md.
    group.add_argument("--merge-hash-stages", action="store_true", default=False,
                       help="Emit hash stages 2 and 3 as two multiply_adds and an XOR: 10 vector ops per "
                            "hash instead of 11 (proof: R8-hash-stage-merge in verification/rewrites.py)")
    group.add_argument("--index-form", choices=["zero-based", "mirrored"], default="zero-based",
                       help="Tree index representation between rounds. mirrored tracks n = L_d - idx (position "
                            "in the level from the right): index update in 2 vector ops instead of 3, none at "
                            "depth 0 and in the wrap round; initial indices are assumed 0 and not loaded "
                            "(proofs: R9 to R11 in verification/rewrites.py)")
    group.add_argument("--prune-dead-index", action="store_true", default=False,
                       help="Zero-based index form: do not emit index work whose result is known. At depth 0 "
                            "every index is 0, so the update needs no multiply_add and the initial indices are "
                            "not loaded; the wrap round shares one zero vector. Output unchanged (proof: "
                            "R12-index-update-from-root). No effect with --index-form mirrored, which has none "
                            "of this work")
    group.add_argument("--values-only", action="store_true", default=False,
                       help="CHANGES THE OUTPUT: do not store the final indices, only the values (which is all "
                            "the upstream tests compare). The last round's index update and everything else "
                            "that only fed the index store is then dead and removed. --validate checks values "
                            "only under this flag")
    group.add_argument("--cached-select", choices=["standard", "bits", "bits-flow", "mobius"], default="standard",
                       help="Node selection on the cached tree levels (depth 0 to --tree-reduce-max-depth). bits: "
                            "select tree driven by the branch bits over node constants that already carry the "
                            "deferred hash constant (no compare, root broadcast or XOR per group); leaf selects "
                            "are multiply_adds, the selects above them go to flow or the vector ALU as the "
                            "scheduler chooses. bits-flow: the leaf selects are scheduler's choice too, so all "
                            "2^d - 1 can use the flow engine. mobius: the same table as 2^d - 1 multiply_adds, "
                            "no flow ops (proofs: R13, R14 in verification/rewrites.py)")
    group.add_argument("--select-bit-source", choices=["index", "path"], default="index",
                       help="With --cached-select bits|bits-flow|mobius: where the bits of earlier rounds come from. index "
                            "(default) extracts them from the index (2 vector ops per group at depth 3, 4 at "
                            "depth 4, none below; one more for a zero-based index). path keeps every bit live "
                            "until the select: no ops, but 8 scratch words per group and earlier round, which "
                            "can exceed the scratch budget (proof: R15-bits-from-index)")

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
        "merge_hash_stages": get("merge_hash_stages", False),
        "index_form": get("index_form", "zero-based"),
        "prune_dead_index": get("prune_dead_index", False),
        "values_only": get("values_only", False),
        "cached_select": get("cached_select", "standard"),
        "select_bit_source": get("select_bit_source", "index"),
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
        "merge_hash_stages": get("merge_hash_stages", False),
        "index_form": get("index_form", "zero-based"),
        "prune_dead_index": get("prune_dead_index", False),
        "values_only": get("values_only", False),
        "cached_select": get("cached_select", "standard"),
        "select_bit_source": get("select_bit_source", "index"),
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
