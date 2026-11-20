#!/usr/bin/env python3
"""Solve an exported CP-SAT model under a variant of its objective and record the incumbent trace.

The scheduler's full-kernel model minimises ``scale * makespan + sum(op completion cycles)``.
This script takes the exact model the scheduler built (``CPSAT_EXPORT_MODEL=m.pb`` writes
``m.pb`` and ``m.pb.params``), rewrites only the objective or the solve procedure, solves it
with the same OR-Tools version, and writes one JSON record: every improving solution with
its deterministic time, makespan, tie-breaker sum and the subsolver that found it.

Variants (``--variant``):
  tiebreak     the model as exported
  makespan     objective = makespan (tie-breaker terms removed; nothing else changes)
  frac:P       tie-breaker over a fixed P percent of its terms (0 < P < 100)
  late         scale * makespan - sum: the secondary objective pushes ops late, not early
  weight:K     makespan coefficient multiplied by K (tie-breaker K times lighter)

Modes (``--mode``):
  single       one solve
  destructive  after every solution that lowers the makespan to M, stop, impose
               makespan <= M - 1 and solve again (hinted with that solution) on what is left
               of the budget; the objective itself is whatever --variant says
  twophase     solve the makespan variant on --phase1-fraction of the budget, then fix
               makespan <= M and minimise the tie-breaker sum on the rest

Usage:
  uv run python -m scripts.objective_variants m.pb --variant makespan --dtime 20 --seed 3 --out run.json
  uv run python -m scripts.objective_variants m.pb --variant tiebreak --param use_lns=false --out run.json
"""

import argparse
import json
import re
import sys
import time

from google.protobuf import text_format
from ortools.sat import cp_model_pb2, sat_parameters_pb2
from ortools.sat.python import cp_model

LOG_SOLUTION = re.compile(r"^#(\d+)\s+([\d.]+)s\s+best:(-?[\d']+)\s+next:\[[^\]]*\]\s+(\S+)")


# ----------------------------------------------------------------------------
# Model access
# ----------------------------------------------------------------------------

def load_model(path):
    """Read a CpModelProto written by CPSAT_EXPORT_MODEL."""
    model = cp_model_pb2.CpModelProto()
    with open(path, "rb") as f:
        model.ParseFromString(f.read())
    return model


def load_params(model_path, params_path=None):
    """The SatParameters exported next to the model, or the scheduler's defaults if absent."""
    params = sat_parameters_pb2.SatParameters()
    path = params_path or model_path + ".params"
    try:
        with open(path, "rb") as f:
            params.ParseFromString(f.read())
    except FileNotFoundError:
        if params_path:
            raise
        # build_sat_parameters with CpsatTuning::default(); the gap limit is set by the caller.
        text_format.Parse(
            "log_search_progress: true cp_model_presolve: true linearization_level: 2 "
            "randomize_search: true cp_model_probing_level: 2 interleave_search: true num_workers: 8",
            params)
    return params


class Objective:
    """The exported objective split into its makespan term and its tie-breaker terms."""

    def __init__(self, model):
        names = {v.name: i for i, v in enumerate(model.variables) if v.name}
        self.makespan = names["makespan"]
        terms = dict(zip(model.objective.vars, model.objective.coeffs))
        if self.makespan not in terms:
            raise ValueError("the makespan variable is not in the objective")
        self.scale = terms.pop(self.makespan)
        self.tie_vars = sorted(terms)
        if any(c != 1 for c in terms.values()):
            raise ValueError("expected unit coefficients on the tie-breaker terms")
        if not self.tie_vars:
            raise ValueError("the model has no tie-breaker; export it with the default objective")

    def set(self, model, makespan_coeff, tie_vars, tie_coeff=1, offset=0):
        del model.objective.vars[:]
        del model.objective.coeffs[:]
        model.objective.vars.append(self.makespan)
        model.objective.coeffs.append(makespan_coeff)
        model.objective.vars.extend(tie_vars)
        model.objective.coeffs.extend([tie_coeff] * len(tie_vars))
        model.objective.offset = offset


def apply_variant(model, params, objective, variant):
    """Rewrite the objective of `model` (and the gap limit that depends on it) in place."""
    scale = objective.scale
    if variant == "tiebreak":
        pass
    elif variant == "makespan":
        objective.set(model, 1, [])
        params.ClearField("absolute_gap_limit")  # a fraction of the tie-breaker scale
    elif variant.startswith("frac:"):
        percent = float(variant.split(":", 1)[1])
        if not 0 < percent < 100:
            raise ValueError("frac:P needs 0 < P < 100")
        # Evenly spaced through the sorted term list: the same subset on every run.
        keep = max(1, round(len(objective.tie_vars) * percent / 100))
        step = len(objective.tie_vars) / keep
        objective.set(model, scale, [objective.tie_vars[int(i * step)] for i in range(keep)])
    elif variant == "late":
        # sum <= scale - 1 is a constraint of the model, so scale - 1 - sum >= 0.
        objective.set(model, scale, objective.tie_vars, tie_coeff=-1, offset=scale - 1)
    elif variant.startswith("weight:"):
        factor = int(variant.split(":", 1)[1])
        objective.set(model, scale * factor, objective.tie_vars)
        if params.HasField("absolute_gap_limit"):
            params.absolute_gap_limit *= factor
    else:
        raise ValueError(f"unknown variant {variant!r}")


def set_param(params, assignment):
    """Apply NAME=VALUE (text-format value) to the solver parameters."""
    name, _, value = assignment.partition("=")
    field = params.DESCRIPTOR.fields_by_name.get(name)
    if field is None:
        raise ValueError(f"unknown solver parameter {name!r}")
    if field.label == field.LABEL_REPEATED:
        params.ClearField(name)
        for item in filter(None, value.split(",")):
            text_format.Parse(f'{name}: "{item}"', params)
    else:
        params.ClearField(name)
        text_format.Parse(f"{name}: {value}", params)


# ----------------------------------------------------------------------------
# One solve with a trace
# ----------------------------------------------------------------------------

class Outcome:
    """What one solve returned besides its solutions."""

    def __init__(self, status, deterministic_time, best_objective_bound):
        self.status = status
        self.deterministic_time = deterministic_time
        self.best_objective_bound = best_objective_bound


class Trace(cp_model.CpSolverSolutionCallback):
    """Records every improving solution; optionally stops when the makespan drops."""

    def __init__(self, objective, dtime_offset, stop_below=None):
        super().__init__()
        self.objective = objective
        self.dtime_offset = dtime_offset
        self.stop_below = stop_below
        self.events = []
        self.solution = None
        self.started = time.time()

    def on_solution_callback(self):
        response = self.response_proto
        makespan = response.solution[self.objective.makespan]
        self.events.append({
            "dtime": self.dtime_offset + self.deterministic_time,
            "wall": time.time() - self.started,
            "objective": self.objective_value,
            "makespan": makespan,
            "tie": sum(response.solution[i] for i in self.objective.tie_vars),
            "subsolver": response.solution_info.split(" ")[0] if response.solution_info else None,
        })
        self.solution = list(response.solution)
        if self.stop_below is not None and makespan < self.stop_below:
            self.stop_search()


def solve_once(model, params, objective, dtime_offset=0.0, stop_below=None, log_path=None):
    """Solve `model`; returns (Outcome, trace events, last solution, log lines).

    The model and parameters are protobuf messages; OR-Tools 9.15's Python solver has its own
    message classes, which read the text format.
    """
    solver = cp_model.CpSolver()
    solver.parameters.parse_text_format(text_format.MessageToString(params))
    solver.parameters.log_to_stdout = False
    lines = []
    solver.log_callback = lines.append
    trace = Trace(objective, dtime_offset, stop_below)
    cp = cp_model.CpModel()
    cp.Proto().parse_text_format(text_format.MessageToString(model))
    status = solver.solve(cp, trace)
    if log_path:
        with open(log_path, "a") as f:
            f.write("\n".join(lines) + "\n")
    # The log names the subsolver of every solution; use it where the response did not.
    logged = [m.group(4) for m in map(LOG_SOLUTION.match, lines) if m]
    if len(logged) == len(trace.events):
        for event, name in zip(trace.events, logged):
            event["subsolver"] = name
    response = solver.response_proto
    outcome = Outcome(getattr(status, "name", str(status)), response.deterministic_time, response.best_objective_bound)
    return outcome, trace.events, trace.solution, lines


def parse_stat_tables(lines, wanted=("LNS stats", "Search stats")):
    """The per-subsolver tables CP-SAT prints at the end of a solve: {table: {row: {column: text}}}.

    "LNS stats" gives, per neighbourhood generator, improving/total calls, the share of
    sub-solves that were closed (OPTIMAL or INFEASIBLE) and the adapted difficulty;
    "Search stats" gives conflicts and branches per full-problem worker.
    """
    tables, current, columns = {}, None, []
    for line in "\n".join(lines).splitlines():
        header = next((name for name in wanted if line.startswith(name)), None)
        if header:
            current, columns = tables.setdefault(header, {}), line[len(header):].split()
            continue
        row = re.match(r"^\s*'([^']+)':\s+(.*)$", line)
        if current is not None and row:
            current[row.group(1)] = dict(zip(columns, row.group(2).replace("'", "").split()))
        else:
            current = None
    return tables


def set_hint(model, solution):
    del model.solution_hint.vars[:]
    del model.solution_hint.values[:]
    model.solution_hint.vars.extend(range(len(solution)))
    model.solution_hint.values.extend(solution)


def cap_makespan(model, objective, upper):
    """makespan <= upper, as a domain change. Returns False if that empties the domain."""
    variable = model.variables[objective.makespan]
    lower = variable.domain[0]
    if upper < lower:
        return False
    del variable.domain[:]
    variable.domain.extend([lower, upper])
    return True


# ----------------------------------------------------------------------------
# Solve procedures
# ----------------------------------------------------------------------------

def status_name(outcome):
    return outcome.status


def run_single(model, params, objective, args):
    response, events, solution, lines = solve_once(model, params, objective, log_path=args.log)
    return {"events": events, "solution": solution, "status": status_name(response),
            "dtime": response.deterministic_time, "solves": 1,
            "best_bound": response.best_objective_bound, "stats": parse_stat_tables(lines)}


def run_destructive(model, params, objective, args):
    """Impose makespan <= incumbent - 1 between solves instead of relying on the objective cutoff."""
    budget = params.max_deterministic_time
    used, events, solution, solves = 0.0, [], None, 0
    status = "UNKNOWN"
    while used < budget:
        step = sat_parameters_pb2.SatParameters()
        step.CopyFrom(params)
        step.max_deterministic_time = budget - used
        incumbent = events[-1]["makespan"] if events else None
        response, new_events, new_solution, _ = solve_once(
            model, step, objective, dtime_offset=used,
            stop_below=incumbent if incumbent is not None else 1 << 60, log_path=args.log)
        solves += 1
        used += response.deterministic_time
        if not new_events:
            # No schedule under the cap: proved (INFEASIBLE) or out of budget (UNKNOWN).
            status = "OPTIMAL" if status_name(response) == "INFEASIBLE" and events else status_name(response)
            if events and status != "OPTIMAL":
                status = "FEASIBLE"
            break
        events.extend(new_events)
        solution = new_solution
        status = "FEASIBLE"
        if not cap_makespan(model, objective, events[-1]["makespan"] - 1):
            status = "OPTIMAL"
            break
        set_hint(model, solution)
    return {"events": events, "solution": solution, "status": status, "dtime": used, "solves": solves,
            "best_bound": None}


def run_twophase(model, params, objective, args):
    """Lexicographic: makespan alone, then the tie-breaker sum with the makespan capped."""
    budget = params.max_deterministic_time
    first = cp_model_pb2.CpModelProto()
    first.CopyFrom(model)
    first_params = sat_parameters_pb2.SatParameters()
    first_params.CopyFrom(params)
    apply_variant(first, first_params, objective, "makespan")
    first_params.max_deterministic_time = budget * args.phase1_fraction
    response, events, solution, _ = solve_once(first, first_params, objective, log_path=args.log)
    used = response.deterministic_time
    result = {"events": events, "solution": solution, "status": status_name(response), "dtime": used,
              "solves": 1, "best_bound": response.best_objective_bound,
              "phase1": {"status": status_name(response), "dtime": used,
                         "makespan": events[-1]["makespan"] if events else None}}
    if not events or budget - used <= 0:
        return result
    second = cp_model_pb2.CpModelProto()
    second.CopyFrom(model)
    cap_makespan(second, objective, events[-1]["makespan"])
    del second.objective.vars[:]
    del second.objective.coeffs[:]
    second.objective.vars.extend(objective.tie_vars)
    second.objective.coeffs.extend([1] * len(objective.tie_vars))
    set_hint(second, solution)
    second_params = sat_parameters_pb2.SatParameters()
    second_params.CopyFrom(params)
    second_params.ClearField("absolute_gap_limit")
    # Whatever phase 1 left unused (it stops early when it proves the makespan) goes to phase 2.
    second_params.max_deterministic_time = budget - used
    response2, events2, solution2, _ = solve_once(second, second_params, objective, dtime_offset=used,
                                                  log_path=args.log)
    result["events"] = events + events2
    result["solution"] = solution2 or solution
    result["dtime"] = used + response2.deterministic_time
    result["solves"] = 2
    result["phase2"] = {"status": status_name(response2), "dtime": response2.deterministic_time}
    return result


# ----------------------------------------------------------------------------
# What the returned schedule looks like
# ----------------------------------------------------------------------------

def auxiliary_drift(model, solution):
    """How far inequality-bounded auxiliaries sit from the value they stand for.

    For every shiftable op: `shifted_dest_start` should be the first lane cycle when the op is
    shifted (it is only bounded above by the lanes), and `completion` should be the last lane
    cycle. Counts the ops where the solution differs and by how much.
    """
    names = {v.name: i for i, v in enumerate(model.variables) if v.name}
    value = lambda name: solution[names[name]]
    lanes = {}
    for name in names:
        m = re.fullmatch(r"alu_c(\d+)_(\d+)", name)
        if m:
            lanes.setdefault(int(m.group(1)), []).append(name)
    out = {"shiftable": len(lanes), "shifted": 0,
           "start_below_first_lane": 0, "start_below_first_lane_cycles": 0, "start_above_first_lane": 0,
           "completion_above_last_lane": 0, "completion_above_last_lane_cycles": 0,
           "completion_below_last_lane": 0,
           "unshifted_completion_not_cycle": 0, "unshifted_lanes_not_cycle": 0}
    for idx, lane_names in sorted(lanes.items()):
        lane_values = [value(n) for n in lane_names]
        shifted = bool(value(f"shift_{idx}")) if f"shift_{idx}" in names else True
        completion = value(f"completion_c{idx}") if f"completion_c{idx}" in names else None
        start = value(f"shifted_dest_start_{idx}") if f"shifted_dest_start_{idx}" in names else None
        if shifted:
            out["shifted"] += 1
            if start is not None and start < min(lane_values):
                out["start_below_first_lane"] += 1
                out["start_below_first_lane_cycles"] += min(lane_values) - start
            if start is not None and start > min(lane_values):
                out["start_above_first_lane"] += 1
            if completion is not None and completion > max(lane_values):
                out["completion_above_last_lane"] += 1
                out["completion_above_last_lane_cycles"] += completion - max(lane_values)
            if completion is not None and completion < max(lane_values):
                out["completion_below_last_lane"] += 1
        else:
            cycle = value(f"c{idx}")
            if completion is not None and completion != cycle:
                out["unshifted_completion_not_cycle"] += 1
            if any(v != cycle for v in lane_values):
                out["unshifted_lanes_not_cycle"] += 1
    return out


def summarise(record):
    """Fields derived from the trace: what a table needs without re-reading the events."""
    events = record["events"]
    if not events:
        return {"final_makespan": None}
    final = events[-1]["makespan"]
    drops = [e for i, e in enumerate(events) if i == 0 or e["makespan"] < events[i - 1]["makespan"]]
    by_solver = {}
    for i, event in enumerate(events):
        entry = by_solver.setdefault(event["subsolver"] or "?", {"makespan": 0, "tie_only": 0})
        entry["makespan" if i == 0 or event["makespan"] < events[i - 1]["makespan"] else "tie_only"] += 1
    return {
        "first_makespan": events[0]["makespan"],
        "final_makespan": final,
        "final_tie": events[-1]["tie"],
        "dtime_to_final_makespan": next(e["dtime"] for e in events if e["makespan"] == final),
        "solutions": len(events),
        "makespan_improvements": len(drops) - 1,
        "tie_only_improvements": len(events) - len(drops),
        "by_subsolver": by_solver,
    }


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help="CpModelProto written by CPSAT_EXPORT_MODEL (default objective)")
    parser.add_argument("--params", help="SatParameters file (default: MODEL.params, else scheduler defaults)")
    parser.add_argument("--variant", default="tiebreak")
    parser.add_argument("--mode", choices=["single", "destructive", "twophase"], default="single")
    parser.add_argument("--dtime", type=float, help="max_deterministic_time (default: as exported)")
    parser.add_argument("--time-limit", type=float, default=7200.0, help="wall-clock backstop in seconds")
    parser.add_argument("--seed", type=int, help="random_seed (default: as exported)")
    parser.add_argument("--param", action="append", default=[], metavar="NAME=VALUE",
                        help="override a solver parameter, e.g. use_lns=false (repeatable)")
    parser.add_argument("--phase1-fraction", type=float, default=0.75)
    parser.add_argument("--out", required=True, help="JSON record to write")
    parser.add_argument("--log", help="append the solver log(s) to this file")
    parser.add_argument("--keep-solution", action="store_true", help="store the final solution vector")
    args = parser.parse_args(argv)

    model = load_model(args.model)
    params = load_params(args.model, args.params)
    objective = Objective(model)
    if not params.HasField("absolute_gap_limit"):
        params.absolute_gap_limit = objective.scale // 3
    if args.dtime is not None:
        params.max_deterministic_time = args.dtime
    if not params.HasField("max_deterministic_time"):
        parser.error("no deterministic-time budget: pass --dtime")
    params.max_time_in_seconds = args.time_limit
    if args.seed is not None:
        params.random_seed = args.seed
    for assignment in args.param:
        set_param(params, assignment)
    if args.mode != "twophase":
        apply_variant(model, params, objective, args.variant)

    started = time.time()
    result = {"single": run_single, "destructive": run_destructive, "twophase": run_twophase}[args.mode](
        model, params, objective, args)
    wall = time.time() - started
    solution = result.pop("solution")
    record = {
        "model": args.model, "variant": args.variant, "mode": args.mode,
        "seed": params.random_seed if params.HasField("random_seed") else 1,
        "budget": params.max_deterministic_time, "param_overrides": args.param,
        "parameters": text_format.MessageToString(params, as_one_line=True),
        "scale": objective.scale, "tie_terms": len(objective.tie_vars),
        "variables": len(model.variables), "constraints": len(model.constraints),
        "wall": wall,
        # A run that ended on the wall clock got as far as machine load allowed: not a measurement.
        "wall_limited": wall >= args.time_limit * 0.999 and result["dtime"] < params.max_deterministic_time,
        **result,
    }
    record.update(summarise(record))
    if solution:
        record["auxiliary_drift"] = auxiliary_drift(model, solution)
        if args.keep_solution:
            record["solution"] = solution
    with open(args.out, "w") as f:
        json.dump(record, f, indent=1)
    print(f"{args.variant} {args.mode} seed={record['seed']}: makespan {record.get('final_makespan')} "
          f"({record['status']}), {record.get('solutions', 0)} solutions "
          f"({record.get('makespan_improvements', 0)} lowered the makespan), "
          f"dtime {record['dtime']:.2f} of {record['budget']}, wall {wall:.0f}s"
          + (" WALL-LIMITED" if record["wall_limited"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
