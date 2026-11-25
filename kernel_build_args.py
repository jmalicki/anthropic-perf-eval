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
RELAXED_SCHEDULER_JSON = "scheduler_output_relaxed_scratch.json"
NORMAL_KERNEL_PATHS = {
    "kernel_to_validate.json",
    "kernel_upstream_check.json",
    "scheduler_output_upstream_check.json",
    DEFAULT_SCHEDULER_JSON,
}

# Mirrors of DEFAULT_GREEDY_* in scheduler_rs/src/kernel_builder/config.rs.
DEFAULT_GREEDY_STAGGER_CYCLES = 12
DEFAULT_GREEDY_GROUPS_IN_FLIGHT = 16

# Machine scratch size (problem.SCRATCH_SIZE) and the size used for --no-scratch-constraint
# (UNBOUNDED_SCRATCH_WORDS in scheduler_rs/src/kernel_builder/config.rs).
MACHINE_SCRATCH_SIZE = 1536
UNBOUNDED_SCRATCH_WORDS = 1 << 20


def scratch_settings(get) -> tuple:
    """Effective (scratch_size, scratch_constraint) for the relaxed-scratch options.

    get: option lookup (name, default) -> value, as in args_to_build_kwargs.

    Returns (words the greedy and the allocator may use, whether CP-SAT models scratch).
    """
    constraint = get("scratch_constraint", True)
    size = get("scratch_budget_override")
    if size is None:
        size = MACHINE_SCRATCH_SIZE if constraint else UNBOUNDED_SCRATCH_WORDS
    return size, constraint


def is_relaxed_scratch(scratch_size: int, scratch_constraint: bool) -> bool:
    """True when the kernel would not be valid on the machine's 1,536-word scratch."""
    return scratch_size > MACHINE_SCRATCH_SIZE or not scratch_constraint


def n_nodes_for_height(forest_height: int) -> int:
    """Node count of the forest, matching problem.Tree.generate(height).

    A tree of height h has levels 0..h, so 2**(h+1) - 1 nodes (2047 for height 10).
    """
    return (1 << (forest_height + 1)) - 1


_n_nodes = n_nodes_for_height


# ----------------------------------------------------------------------------
# CP-SAT tuning switches (kept together, apart from the other option helpers)
# ----------------------------------------------------------------------------

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


def _worker_count(text: str) -> int:
    """argparse type: a CP-SAT worker count (OR-Tools 9.15 accepts 1 to 10000)."""
    value = _positive_int(text)
    if value > 10000:
        raise argparse.ArgumentTypeError(f"must be at most 10000, got {text!r}")
    return value


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
    ("cpsat_cumulative_propagators", "cumulative_propagators"),
    ("cpsat_ttef_max_intervals", "timetable_edge_finding_max_intervals"),
    ("cpsat_lane_model", "lane_model"),
    ("cpsat_linearization_level", "linearization_level"),
    ("cpsat_probing_level", "probing_level"),
    ("cpsat_makespan_rows", "makespan_rows"),
    ("cpsat_symmetry_rows", "symmetry_rows"),
    ("cpsat_random_seed", "random_seed"),
]

# Tuning keys whose value "none" is an enum variant, not "switched off" (JSON null).
ENUM_TUNING_KEYS = {"makespan_rows", "symmetry_rows"}

CPSAT_LANE_MODELS = ["free", "sorted", "pinned", "sorted-pinned"]

CUMULATIVE_PROPAGATORS = ["overload_checker", "timetable_edge_finding", "hard_precedences"]


def _propagator_list(text: str) -> list:
    """argparse type: comma-separated cumulative propagator names; 'all' means every one."""
    names = _name_list(text)
    if names == ["all"]:
        return list(CUMULATIVE_PROPAGATORS)
    unknown = [name for name in names if name not in CUMULATIVE_PROPAGATORS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown cumulative propagator(s) {unknown}; valid: all, {', '.join(CUMULATIVE_PROPAGATORS)}")
    return names


# ----------------------------------------------------------------------------

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


def _non_negative_int(text: str) -> int:
    """argparse type: an integer greater than or equal to zero."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {text!r}") from None
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be zero or more, got {text!r}")
    return value


# =============================================================================
# Emission options
# =============================================================================
# What ops the kernel contains (docs/EMISSION_ABLATION.md). Kept apart from the scheduler
# options: this dict is the single list of them (name -> default) for the CLI, run_options,
# the binding kwargs and the run header, and add_emission_args is their only argparse block.
EMISSION_DEFAULTS = {
    "merge_hash_stages": False,
    "index_form": "zero-based",
    "prune_dead_index": False,
    "values_only": False,
    "cached_select": "standard",
    "select_bit_source": "index",
    "depth4_select_rounds": "all",
    # Per round and group tree-access encodings (solver/emission_plan.py): a path, a plan dict,
    # or an expanded list. Passed to the binding expanded; None = choose from the flags above.
    "emission_plan": None,
    # Rewrite rules applied to the finished SSA program before scheduling (list of names, in
    # order); None = none. They read only the ops (scheduler_rs/src/kernel_builder/transform/).
    "transform": None,
}


def add_emission_args(parser: argparse.ArgumentParser) -> argparse._ArgumentGroup:
    """Add the emission options to parser. Returns the argument group."""
    group = parser.add_argument_group(
        "Emission options",
        "Change which ops the kernel contains. Defaults leave the emission unchanged; each option is "
        "backed by a proof in verification/rewrites.py. See docs/EMISSION_ABLATION.md.")
    group.add_argument("--merge-hash-stages", action="store_true", default=EMISSION_DEFAULTS["merge_hash_stages"],
                       help="Emit hash stages 2 and 3 as two multiply_adds and an XOR: 10 vector ops per "
                            "hash instead of 11 (proof: R8-hash-stage-merge)")
    group.add_argument("--index-form", choices=["zero-based", "mirrored"], default=EMISSION_DEFAULTS["index_form"],
                       help="Tree index representation between rounds. mirrored tracks n = L_d - idx (position "
                            "in the level from the right): index update in 2 vector ops instead of 3, 1 at depth 0 "
                            "(no multiply_add) and none in the wrap round; initial indices are assumed 0 and "
                            "not loaded "
                            "(proofs: R9 to R11)")
    group.add_argument("--prune-dead-index", action="store_true", default=EMISSION_DEFAULTS["prune_dead_index"],
                       help="Zero-based index form: do not emit index work whose result is known. At depth 0 "
                            "every index is 0, so the update needs no multiply_add and the initial indices are "
                            "not loaded; the wrap round shares one zero vector. Output unchanged (proof: "
                            "R12-index-update-from-root). No effect with --index-form mirrored, which has none "
                            "of this work")
    group.add_argument("--values-only", action="store_true", default=EMISSION_DEFAULTS["values_only"],
                       help="CHANGES THE OUTPUT: do not store the final indices, only the values (which is all "
                            "the upstream tests compare). The last round's index update and everything else "
                            "that only fed the index store is then dead and removed. --validate checks values "
                            "only under this flag")
    group.add_argument("--cached-select", choices=["standard", "bits", "bits-flow", "mobius"],
                       default=EMISSION_DEFAULTS["cached_select"],
                       help="Node selection on the cached tree levels (depth 0 to --tree-reduce-max-depth). bits: "
                            "select tree driven by the branch bits over node constants that already carry the "
                            "deferred hash constant (no compare, root broadcast or XOR per group); leaf selects "
                            "are multiply_adds, the selects above them go to flow or the vector ALU as the "
                            "scheduler chooses. bits-flow: the leaf selects are scheduler's choice too, so all "
                            "2^d - 1 can use the flow engine. mobius: the same table as 2^d - 1 multiply_adds, "
                            "no flow ops (proofs: R13, R14)")
    group.add_argument("--select-bit-source", choices=["index", "path"], default=EMISSION_DEFAULTS["select_bit_source"],
                       help="With --cached-select bits|bits-flow|mobius: where the bits of earlier rounds come "
                            "from. index (default) extracts the older bits used at depths 3 and 4 from the index (2 "
                            "vector ops per group at depth 3, 4 at depth 4; one more for a zero-based index); the "
                            "newest bit and, at depth 2, the depth-0 bit are still read directly. path keeps "
                            "every bit live until the select: no ops, but 8 scratch words per group and earlier "
                            "round, which can exceed the scratch budget (proof: R15-bits-from-index)")
    group.add_argument("--depth4-select-rounds", choices=["all", "after-wrap"],
                       default=EMISSION_DEFAULTS["depth4_select_rounds"],
                       help="With --tree-reduce-max-depth 4: which visits of tree level 4 select among the 16 "
                            "cached nodes instead of gathering. all (default): every visit (rounds 4 and 15 of "
                            "16). after-wrap: only visits after the first wrap (round 15); round 4 gathers. "
                            "Selecting costs 15 selects per group and saves 8 loads per group")
    group.add_argument("--emission-plan", default=EMISSION_DEFAULTS["emission_plan"], metavar="PATH",
                       help="JSON plan of the tree-access encoding (gather, bits, bits-flow, mobius) of every "
                            "group in every round, as written by scripts/emission_structure.py (format: "
                            "solver/emission_plan.py). Replaces --cached-select, --tree-reduce-max-depth and "
                            "--depth4-select-rounds for the tree access. See docs/EMISSION_STRUCTURE_MODEL.md")
    group.add_argument("--transform", action="append", choices=["hash-merge", "prune-index", "values-only", "select-bits-path"], default=EMISSION_DEFAULTS["transform"],
                       metavar="RULE",
                       help="Rewrite the finished SSA program with RULE before scheduling (repeatable; applied in "
                            "order). hash-merge: find hash stages x = a*m + c; (x + d) ^ (x << k) from the ops "
                            "alone and rewrite them to (a*m + c+d) ^ (a*(m<<k) + (c<<k)), the program "
                            "--merge-hash-stages emits (proof: R8-hash-stage-merge-general; with --merge-hash-stages "
                            "there is nothing left to merge and it warns). prune-index: share the wrap "
                            "round's zero vector and replace multiply_adds with a factor shown to be 0 (the index "
                            "update at depth 0) by their addend, assuming the input indices start at 0; the initial "
                            "index load then dies (proofs: R12). values-only: declare the final indices unobserved, "
                            "so the index stores stop being roots and dead code elimination does the rest; "
                            "validation then compares values only. select-bits-path: a bit-driven tree access "
                            "at depth 3 or more reads the earlier branch bits instead of extracting them from the "
                            "index (the program --select-bit-source path emits; fewer ops, more scratch), from "
                            "forms proved on the ops, assuming the input indices start at 0 (proofs: R15). "
                            "Off by default")
    return group


def resolve_emission_plan(plan, rounds: int, batch_size: int, forest_height: int, cached_select: str,
                          depth4_select_rounds: str):
    """The `emission_plan` value for the binding: `plan` (a path, plan dict or list) expanded to
    one encoding name per round and group, or None. Rejects combining a plan with the flags it
    replaces."""
    if plan is None:
        return None
    from solver.emission_plan import expand_plan, load_plan

    if cached_select != EMISSION_DEFAULTS["cached_select"] or depth4_select_rounds != EMISSION_DEFAULTS["depth4_select_rounds"]:
        raise ValueError("--emission-plan chooses the tree access of every round and group; "
                         "do not combine it with --cached-select or --depth4-select-rounds")
    if isinstance(plan, str):
        plan = load_plan(plan)
    return expand_plan(plan, rounds, batch_size // 8, forest_height)


def emission_options(get) -> dict:
    """Every emission option with its value, read through `get(name, default)`. The emission
    plan comes back expanded (resolve_emission_plan)."""
    options = {name: get(name, default) for name, default in EMISSION_DEFAULTS.items()}
    options["emission_plan"] = resolve_emission_plan(
        options["emission_plan"], get("rounds", 16), get("batch_size", 256), get("forest_height", 10),
        options["cached_select"], options["depth4_select_rounds"])
    return options


def non_default_emission_options(build_kwargs: dict) -> dict:
    """Emission options in `build_kwargs` that differ from their defaults (for the run header).
    An emission plan is shown as a one-line summary."""
    options = {name: build_kwargs[name] for name, default in EMISSION_DEFAULTS.items()
               if build_kwargs.get(name, default) != default}
    if options.get("emission_plan") is not None:
        from solver.emission_plan import describe_plan

        options["emission_plan"] = describe_plan(options["emission_plan"])
    return options


def add_kernel_build_args(parser: argparse.ArgumentParser) -> argparse._ArgumentGroup:
    """Add kernel build options to parser. Returns the argument group."""
    add_emission_args(parser)
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
    group.add_argument("--cpsat-workers", type=_worker_count, default=None, metavar="N",
                       help="CP-SAT num_workers, 1 to 10000 (default: 8)")
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
    group.add_argument("--cpsat-scratch-time", choices=["phase", "cycle"], default="phase",
                       help="Timeline of the CP-SAT scratch cumulative. 'phase' (default): read and write "
                            "phases of each cycle, so a word last read at cycle C can hold a value written "
                            "at C, exactly as address assignment reuses it. 'cycle': each value counted over "
                            "[write, last read + 1), which also counts values dying at C together with "
                            "values born at C (the earlier overcount, kept for ablation)")
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
    group.add_argument("--cpsat-cumulative-propagators", type=_propagator_list, default=None, metavar="NAME,...",
                       help="Extra CP-SAT cumulative propagators for every worker: all, or a list of "
                            + ", ".join(CUMULATIVE_PROPAGATORS) + " (default: none). hard_precedences sets both "
                            "use_hard_precedences_in_cumulative and exploit_all_precedences: OR-Tools 9.15 "
                            "ignores the first without the second.")
    group.add_argument("--cpsat-ttef-max-intervals", type=_non_negative_int, default=None, metavar="N",
                       help="max_num_intervals_for_timetable_edge_finding (CP-SAT default: 100). Timetable edge "
                            "finding is skipped on larger cumulatives, i.e. on all of ours unless this is raised.")
    group.add_argument("--cpsat-lane-model", choices=CPSAT_LANE_MODELS, default=None,
                       help="Lane variables of a shiftable vector op's 8 scalar ops (default: free). sorted: "
                            "lane_0 <= ... <= lane_7 (removes the 8! lane permutations of each op). pinned: an "
                            "unshifted op's lanes equal its cycle, and lane precedences carry no shift literal. "
                            "sorted-pinned: both. Neither loses a schedule (verification/lane_model.py).")
    group.add_argument("--cpsat-linearization-level", type=int, choices=[0, 1, 2], default=None,
                       help="CP-SAT linearization_level (default: 2; CP-SAT's own default is 1). 2 adds the "
                            "LP relaxations and cuts of cumulatives; see docs/CPSAT_CONFIG_ABLATION.md.")
    group.add_argument("--cpsat-probing-level", type=int, choices=[0, 1, 2, 3], default=None,
                       help="CP-SAT cp_model_probing_level (default: 2)")
    group.add_argument("--cpsat-makespan-rows", choices=["none", "static", "bucketed"], default=None,
                       help="Extra makespan lower-bound rows (default: none). static: per-engine head/tail "
                            "energy rows from static longest-path heads and tails. bucketed: static plus "
                            "prefix-end/suffix-start variables per bucket of heads that follow the solution. "
                            "See scheduler_rs/src/cpsat_scheduler/makespan_rows.rs.")
    group.add_argument("--cpsat-random-seed", type=int, default=None, metavar="N",
                       help="Set the CP-SAT solver seed (default: CP-SAT's own). Reproducible parallel search "
                            "requires deterministic interleaving and must finish before the wall-clock limit.")
    group.add_argument("--cpsat-symmetry-rows", choices=["declared", "none", "checked"], default=None,
                       help="Symmetry-breaking ordering rows (default: declared = the sym_ord/sym_lex rows emission "
                            "declares, unchecked). none = no ordering rows. checked = only orderings justified by a "
                            "swap of structurally identical vector groups (one anchor per group; "
                            "scheduler_rs/src/cpsat_scheduler/group_symmetry.rs)")
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
                       help="ASAP/ALAP tightening mode: none, simple (precedence only), symmetry-aware, or energetic (default: symmetry-aware plus resource-aware earliest/latest cycles). "
                            "The resource-aware part excludes no schedule of the model; the symmetry-aware part "
                            "is valid only with the declared ordering rows and is skipped unless "
                            "--cpsat-symmetry-rows is declared")
    group.add_argument("--enable-scratch-tightening", action="store_true",
                       help="Enable scratch-aware tightening (default: disabled; non-default debugging option)")
    group.add_argument("--symmetry-precedence-integrated", action="store_true",
                       help="Integrate precedence into symmetry ordering tie-breaking (experimental)")
    group.add_argument("--precedence-neutral-horizon", action="store_true",
                       help="Use precedence-neutral horizon (2x critical path) instead of greedy-based")
    group.add_argument("--scheduling-pressure-policy",
                       choices=["greedy", "throttled", "gated", "adaptive", "cascade"],
                       default="adaptive",
                       help="Scheduling pressure policy (default: adaptive)")

    group.add_argument("--greedy-barriers", choices=["round", "none"], default="round",
                       help="Round barriers in the greedy scheduler: 'round' (default) runs rounds strictly one "
                            "after another; 'none' drops the barriers so the ready queue may mix rounds (the "
                            "kernel is then one section for the per-section budget check)")
    group.add_argument("--greedy-priority", choices=["fifo", "wavefront"], default="fifo",
                       help="Greedy ready-queue order: 'fifo' (default) or 'wavefront', which staggers the "
                            "independent group chains so some gather while others hash or select. Use with "
                            "--greedy-barriers none")
    group.add_argument("--greedy-stagger-cycles", type=_non_negative_int, default=DEFAULT_GREEDY_STAGGER_CYCLES,
                       metavar="CYCLES",
                       help="Wavefront: virtual start delay between consecutive group chains in cycles "
                            f"(default: {DEFAULT_GREEDY_STAGGER_CYCLES}; 0 = all chains abreast)")
    group.add_argument("--greedy-groups-in-flight", type=_non_negative_int,
                       default=DEFAULT_GREEDY_GROUPS_IN_FLIGHT, metavar="N",
                       help="Wavefront: group chains allowed to run at once, which bounds scratch use "
                            f"(default: {DEFAULT_GREEDY_GROUPS_IN_FLIGHT}; 0 = no limit)")
    group.add_argument("--greedy-split-shift", action="store_true", default=False,
                       help="With --use-greedy: let a vector op run as 8 scalar ops spread over two cycles, so "
                            "all 12 scalar slots carry shifted work (3 shifted ops per 2 cycles instead of 2). "
                            "Ignored with CP-SAT unless --cpsat-split-shift-hint is set")
    group.add_argument("--greedy-justify", type=_non_negative_int, default=0, metavar="N",
                       help="Shorten the greedy schedule by forward/backward justification, at most N "
                            "iterations (default 0 = off). Keeps the greedy's const/select encodings; a "
                            "candidate that breaks a slot limit, a precedence or the scratch budget is dropped")
    group.add_argument("--pressure-soft-threshold", type=_positive_finite_float, default=None, metavar="FRAC",
                       help="Pressure policy: fraction of the scratch budget above which the ready queue blends "
                            "in net pressure and busy-ALU shifts are suppressed (policy default 0.85)")
    group.add_argument("--pressure-critical-threshold", type=_positive_finite_float, default=None, metavar="FRAC",
                       help="Pressure policy: fraction of the scratch budget above which no vector op is "
                            "shifted (policy default 0.95)")
    group.add_argument("--pressure-inflation-threshold", type=_positive_finite_float, default=None, metavar="FRAC",
                       help="Pressure policy: fraction of the scratch budget above which producers with slack "
                            "are deferred (policy default 0.80; 1.0 or more turns deferral off)")
    group.add_argument("--greedy-budget-headroom", type=_non_negative_int, default=0, metavar="WORDS",
                       help="Scratch words the greedy keeps free below the budget (default 0)")
    group.add_argument("--scratch-model", choices=["words", "pools"], default="words",
                       help="What the schedulers' scratch budget bounds: 'words' (default) = live words at "
                            "every point; 'pools' = 8 * (most vectors live at once) + (most scalars live at "
                            "once) <= budget, under which address assignment always fits (a per-point word "
                            "count does not guarantee that; see docs/ALLOCATION_CONJECTURES.md)")
    group.add_argument("--cpsat-split-shift-hint", action="store_true", default=False,
                       help="With --greedy-split-shift and CP-SAT: keep the greedy's split shifts and hint "
                            "CP-SAT each lane at its own cycle, so the solver starts from the split-shift "
                            "schedule (default: the greedy that feeds CP-SAT makes no split shifts)")

    # Relaxed-scratch reference (analysis only; docs/RELAXED_SCRATCH_REFERENCE.md)
    group.add_argument("--scratch-budget-override", type=_positive_int, default=None, metavar="WORDS",
                       help="ANALYSIS ONLY: scratch words the greedy, the per-section repair, CP-SAT and "
                            "address assignment may use instead of the machine's 1536. Above 1536 the kernel "
                            "does not run on the real machine; it is saved as a marked relaxed-scratch kernel "
                            "and validated on a simulator with that much scratch.")
    group.add_argument("--no-scratch-constraint", action="store_false", dest="scratch_constraint", default=True,
                       help="ANALYSIS ONLY: drop the scratch constraint from the CP-SAT model and give the greedy "
                            f"and the allocator {UNBOUNDED_SCRATCH_WORDS} words (unless --scratch-budget-override "
                            "is given). Measures what any scratch-saving technique could buy at most.")

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
        key: (None if get(dest) == "none" and key not in ENUM_TUNING_KEYS else get(dest))
        for dest, key in CPSAT_TUNING_OPTIONS
        if get(dest) is not None
    }
    cpsat_tuning_json = json.dumps(cpsat_tuning) if cpsat_tuning else None
    cpsat_tuning_effective = json.loads(scheduler_rs.resolve_cpsat_tuning(cpsat_tuning_json))

    scratch_size, scratch_constraint = scratch_settings(get)
    relaxed = is_relaxed_scratch(scratch_size, scratch_constraint)

    out_path = output_json if output_json is not None else get("output_json")
    if out_path is None:
        out_path = RELAXED_SCHEDULER_JSON if relaxed else DEFAULT_SCHEDULER_JSON
    elif relaxed and os.path.basename(out_path) in NORMAL_KERNEL_PATHS:
        raise ValueError(f"refusing to write a relaxed-scratch build to {out_path}, a normal kernel/result path")

    emission = emission_options(get)
    run_options = {
        **emission,
        "forest_height": forest_height,
        "n_nodes": n_nodes,
        "batch_size": batch_size,
        "rounds": rounds,
        "use_cpsat": use_cpsat,
        "cpsat_time_limit": cpsat_time_limit,
        "cpsat_deterministic_time": cpsat_deterministic_time,
        "cpsat_scratch_time": get("cpsat_scratch_time", "phase"),
        "cpsat_tuning": cpsat_tuning_effective,
        "disabled_cpsat_constraints": [group.name for group in optional_constraint_groups_to_disable or []],
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
        "greedy_barriers": get("greedy_barriers", "round"),
        "greedy_priority": get("greedy_priority", "fifo"),
        "greedy_stagger_cycles": get("greedy_stagger_cycles", DEFAULT_GREEDY_STAGGER_CYCLES),
        "greedy_groups_in_flight": get("greedy_groups_in_flight", DEFAULT_GREEDY_GROUPS_IN_FLIGHT),
        "greedy_split_shift": get("greedy_split_shift", False),
        "greedy_justify": get("greedy_justify", 0),
        "cpsat_split_shift_hint": get("cpsat_split_shift_hint", False),
        "pressure_soft_threshold": get("pressure_soft_threshold"),
        "pressure_critical_threshold": get("pressure_critical_threshold"),
        "pressure_inflation_threshold": get("pressure_inflation_threshold"),
        "greedy_budget_headroom": get("greedy_budget_headroom", 0),
        "scratch_size": scratch_size,
        "scratch_constraint": scratch_constraint,
        "relaxed_scratch": relaxed,
        "scratch_model": get("scratch_model", "words"),
    }
    git_sha = _get_git_sha() if use_git_sha else None
    enable_cse = False if get("no_cse_groups", False) else None

    return {
        **emission,
        "forest_height": forest_height,
        "n_nodes": n_nodes,
        "batch_size": batch_size,
        "rounds": rounds,
        "enable_valu_shift": get("enable_valu_shift", True),
        "verbose": True,  # caller can override
        "use_cpsat": use_cpsat,
        "cpsat_time_limit": cpsat_time_limit,
        "cpsat_deterministic_time": cpsat_deterministic_time,
        "cpsat_scratch_time": get("cpsat_scratch_time", "phase"),
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
        "greedy_barriers": get("greedy_barriers", "round"),
        "greedy_priority": get("greedy_priority", "fifo"),
        "greedy_stagger_cycles": get("greedy_stagger_cycles", DEFAULT_GREEDY_STAGGER_CYCLES),
        "greedy_groups_in_flight": get("greedy_groups_in_flight", DEFAULT_GREEDY_GROUPS_IN_FLIGHT),
        "greedy_split_shift": get("greedy_split_shift", False),
        "greedy_justify": get("greedy_justify", 0),
        "cpsat_split_shift_hint": get("cpsat_split_shift_hint", False),
        "pressure_soft_threshold": get("pressure_soft_threshold"),
        "pressure_critical_threshold": get("pressure_critical_threshold"),
        "pressure_inflation_threshold": get("pressure_inflation_threshold"),
        "greedy_budget_headroom": get("greedy_budget_headroom", 0),
        "scratch_size": scratch_size,
        "scratch_constraint": scratch_constraint,
        "scratch_model": get("scratch_model", "words"),
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
