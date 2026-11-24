#!/usr/bin/env python3
"""Closed loop between the structure model (stage 1) and the shaped greedy (stage 2).

    uv run python -m scripts.emission_structure loop run --modes nogood,caps,rows --budget 8 \
        --state artifacts/benders/state.json --json artifacts/benders/trace.json
    uv run python -m scripts.emission_structure loop report --state artifacts/benders/state.json

Logic-based Benders-style feedback (docs/EMISSION_STRUCTURE_MODEL.md section 10). Each
iteration solves stage 1 for a plan, builds and schedules it with the shaped greedy (one
greedy run, proved by dataflow replay and validated), reads the scheduler JSON and feeds back
into stage 1. Three kinds of feedback, kept apart because they differ in what they promise:

* ``nogood`` (a): a no-good cut on every built plan plus the incumbent bound "floor <
  best cycles". Sound (the floor is a lower bound on any schedule) and weak: it enumerates
  plans in floor order.
* ``caps`` (b): learned per-engine utilisation caps (`solve_plan(max_util=...)`). The engine
  the plan saturates and the schedule runs full in its steady state gets a cap below the
  plan's use of it; kept when the measured cycles improve, reverted otherwise. A heuristic,
  not a bound: it generalises the manual `--max-util flow=0.9`.
* ``rows`` (c): time-aware rows added to the stage-1 objective, M >= t + q +
  window_floor(work with head >= t and tail >= q), with the work, heads and tails of each
  op class (round, encoding, phase, op kind) learned from the built plans (`RowTable`), plus
  no-goods and the incumbent bound "M < best cycles". The row form is proved
  (verification/window_rows_proof.py); the per-class table it is fed is observed, and every
  build is checked against the table it was solved with (violations are reported).

Every number here is a greedy development measurement, not a recorded benchmark result.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from typing import Mapping, Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
sys.path.insert(0, ROOT)

from scripts import emission_structure as es  # noqa: E402
from solver.emission_plan import describe_plan, expand_plan, load_plan  # noqa: E402

CAPPABLE = ("load", "flow", "valu")
MODES = ("nogood", "caps", "caps-underfilled", "rows")
DEFAULT_SEEDS = ("gather", "bits+depth4", "bits-flow+depth4", "mobius+depth4",
                 "docs/emission_plans/floor_optimal.json", "docs/emission_plans/flow90.json")


def plan_key(rows: Sequence[Sequence[str]]) -> str:
    return json.dumps([list(r) for r in rows])


def counts_key(rows, ctx: es.Context) -> dict[str, int]:
    """`plan_group_counts` with JSON-friendly "round:encoding" keys."""
    return {f"{r}:{e}": v for (r, e), v in es.plan_group_counts(rows, ctx).items()}


# ----------------------------------------------------------------------------
# What a built plan shows: work, heads and tails per op class
# ----------------------------------------------------------------------------
def op_window_category(op: Mapping, shiftable: set, eliminable: set) -> str | None:
    """WINDOW_CATEGORIES kind of one SSA op, or None if a window does not charge it
    (ConstChoice, eliminable vbroadcast, pseudo-ops)."""
    engine, opcode = op["engine"], op["opcode"]
    if opcode == "select_choice":
        return "select"
    if opcode == "const_choice":
        return None
    if engine == "valu":
        if op["id"] in eliminable:
            return None
        return "vec_shiftable" if op["id"] in shiftable else "vec_fixed"
    if engine in ("alu", "load", "store", "flow"):
        return engine
    return None


def observe(rows: Sequence[Sequence[str]], sched: Mapping, ctx: es.Context) -> dict[str, dict]:
    """Op classes of a built plan with their work and minimum head and tail.

    Classes (string keys, for JSON):
        "g|r|e|phase|cat"  ops of round r that belong to one group, for groups using encoding e
                           in round r: "count" is the fewest such ops in any one group
        "r|r|phase|cat"    ops of round r that belong to no group (the hash, the index update,
                           shared per-round constants): "count" is their number
    "head" / "tail" are the smallest head and tail (scripts/static_bounds.py, minimum latency
    over every encoding) of any op of the class. Ops outside the rounds (setup, input, output)
    are not observed: leaving ops out of a window only lowers it.
    """
    from scripts import static_bounds as sb

    bound_ops = sb.ops_from_schedule_json(sched, "any")
    heads, tails = sb.heads(bound_ops), sb.tails(bound_ops)
    ssa = {o["id"]: o for o in sched["ssa_ops"] if o["engine"] != "barrier"}
    opts = sched.get("run_options") or {}
    shift_on = opts.get("enable_valu_shift", True) and opts.get("valu_shift", "optimize") != "valu"
    shiftable = {i for i, o in ssa.items()
                 if shift_on and o["engine"] == "valu" and o["opcode"] in sb.SHIFTABLE_OPCODES}
    consumers = collections.defaultdict(list)
    for o in ssa.values():
        for p in o["dep_ids"]:
            consumers[p].append(o["id"])
    eliminable = (sb._eliminable_vbroadcasts(ssa, consumers, shiftable)
                  if shift_on and opts.get("enable_vbroadcast_elimination", True) else set())
    per_group = collections.defaultdict(collections.Counter)  # (r, e, phase, cat) -> group -> n
    out: dict[str, dict] = {}

    def note(key, head, tail):
        cls = out.setdefault(key, {"count": 0, "head": head, "tail": tail})
        cls["head"] = min(cls["head"], head)
        cls["tail"] = min(cls["tail"], tail)
        return cls

    for i, o in ssa.items():
        cat = op_window_category(o, shiftable, eliminable)
        info = o.get("source_info") or {}
        r, g, phase = info.get("round"), info.get("group"), info.get("phase")
        if cat is None or r is None:
            continue
        if g is None:
            note(f"r|{r}|{phase}|{cat}", heads[i], tails[i])["count"] += 1
        else:
            e = rows[r][g]
            note(f"g|{r}|{e}|{phase}|{cat}", heads[i], tails[i])
            per_group[r, e, phase, cat][g] += 1
    for (r, e, phase, cat), by_group in per_group.items():
        groups = [g for g in range(ctx.n_groups) if rows[r][g] == e]
        out[f"g|{r}|{e}|{phase}|{cat}"]["count"] = min(by_group.get(g, 0) for g in groups)
    # A group-free class missing from this plan has count 0 here, which matters when merging.
    return out


@dataclass
class RowTable:
    """Per op class: work, head and tail, each the minimum over every observed plan. The
    window rows use them as lower bounds in every plan: that is the observed (not proved)
    part of the rows; `check` reports plans that contradict it."""
    context: es.Context
    classes: dict[str, dict] = field(default_factory=dict)
    observed_plans: int = 0
    max_axis: int = 256

    def _seen(self, key: str, obs: Mapping[str, dict], rows) -> dict | None:
        """What a plan shows for a table class: its observation, a zero count when the class
        should be there (a round class; a group class whose encoding the plan uses in that
        round), or None when the plan says nothing about it."""
        if key in obs:
            return obs[key]
        parts = key.split("|")
        if parts[0] == "r" or (rows is not None and parts[2] in rows[int(parts[1])]):
            return {"count": 0}
        return None

    def check(self, obs: Mapping[str, dict], rows=None) -> list[dict]:
        """Classes of the table that one plan (`observe(rows, ...)`) shows to be too high."""
        bad = []
        for key, cls in self.classes.items():
            seen = self._seen(key, obs, rows)
            if seen is None:
                continue
            for f in ("count", "head", "tail"):
                if f in seen and seen[f] < cls[f]:
                    bad.append({"class": key, "field": f, "table": cls[f], "observed": seen[f]})
        return bad

    def merge(self, obs: Mapping[str, dict], rows=None) -> list[dict]:
        """Lower the table to what one plan shows; returns `check` from before the merge."""
        bad = self.check(obs, rows)
        for key, cls in self.classes.items():
            seen = self._seen(key, obs, rows)
            if seen is not None:
                for f in ("count", "head", "tail"):
                    if f in seen:
                        cls[f] = min(cls[f], seen[f])
        for key, seen in obs.items():
            if key not in self.classes:
                # A round class first seen now was absent (count 0) from every earlier plan.
                count = seen["count"] if (key.startswith("g|") or not self.observed_plans) else 0
                self.classes[key] = {"count": count, "head": seen["head"], "tail": seen["tail"]}
        self.observed_plans += 1
        return bad

    def _axis(self, values) -> list[int]:
        vals = sorted(set(values))
        if len(vals) <= self.max_axis:
            return vals
        idx = sorted({round(i * (len(vals) - 1) / (self.max_axis - 1)) for i in range(self.max_axis)})
        return [vals[i] for i in idx]

    def windows(self) -> list[tuple[int, int]]:
        live = [c for c in self.classes.values() if c["count"] > 0]
        return [(t, q) for t in self._axis(c["head"] for c in live) for q in self._axis(c["tail"] for c in live)]

    def window_specs(self, base_rows: Sequence[Sequence[str]]) -> list[dict]:
        """The windows as `solve_plan(windows=...)` takes them. Rounds the model does not choose
        are taken from `base_rows`. A window is kept only if work that every plan has (round
        classes, fixed rounds) lies in it: only then is M >= t + q valid for every plan."""
        ctx = self.context
        choice = set(ctx.choice_rounds())
        specs = []
        for t, q in self.windows():
            const = collections.Counter()
            per_group: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
            for key, cls in self.classes.items():
                if cls["count"] <= 0 or cls["head"] < t or cls["tail"] < q:
                    continue
                parts = key.split("|")
                if parts[0] == "r":
                    const[parts[3]] += cls["count"]
                    continue
                r, e, cat = int(parts[1]), parts[2], parts[4]
                if r in choice:
                    per_group[f"{r}:{e}"][cat] += cls["count"]
                else:
                    const[cat] += cls["count"] * list(base_rows[r]).count(e)
            if sum(const.values()) > 0:
                specs.append({"t": t, "q": q, "const": dict(const),
                              "per_group": {k: dict(v) for k, v in per_group.items()}})
        return specs

    def bound(self, rows: Sequence[Sequence[str]], specs: Sequence[Mapping] | None = None) -> dict:
        """The strongest window row for a plan: max over windows of t + q + window_floor."""
        specs = self.window_specs(rows) if specs is None else specs
        n = counts_key(rows, self.context)
        best = {"bound": 0, "t": None, "q": None, "window_floor": 0}
        for w in specs:
            c = collections.Counter(w["const"])
            for key, coef in w["per_group"].items():
                for cat, v in coef.items():
                    c[cat] += v * n.get(key, 0)
            value = w["t"] + w["q"] + es.window_floor(c)
            if value > best["bound"]:
                best = {"bound": value, "t": w["t"], "q": w["q"], "window_floor": value - w["t"] - w["q"]}
        return best

    def union(self, other: "RowTable") -> "RowTable":
        """The table both sets of observations give (minimum per field)."""
        out = RowTable(self.context, {}, self.observed_plans + other.observed_plans, self.max_axis)
        for key in set(self.classes) | set(other.classes):
            a, b = self.classes.get(key), other.classes.get(key)
            if a is None or b is None:
                have, missing_in = (a, other) if b is None else (b, self)
                cls = dict(have)
                if key.startswith("r|") and missing_in.observed_plans:
                    cls["count"] = 0
                out.classes[key] = cls
            else:
                out.classes[key] = {f: min(a[f], b[f]) for f in ("count", "head", "tail")}
        return out

    def to_json(self) -> dict:
        return {"context": asdict(self.context), "classes": self.classes,
                "observed_plans": self.observed_plans, "max_axis": self.max_axis}

    @classmethod
    def from_json(cls, d: Mapping) -> "RowTable":
        return cls(es.Context(**d["context"]), {k: dict(v) for k, v in d["classes"].items()},
                   d.get("observed_plans", 0), d.get("max_axis", 256))


# ----------------------------------------------------------------------------
# Capped floor (what `solve_plan(max_util=caps)` minimises), for any plan
# ----------------------------------------------------------------------------
def _fits_budgets(c: Mapping[str, int], b: Mapping[str, int]) -> bool:
    """es._feasible_at with absolute slot budgets per engine instead of cap * T."""
    if c["store"] > b["store"] or c["flow"] > b["flow"] or c["load"] > b["load"]:
        return False
    off_load = max(0, c["load"] + c["const"] + c["const_alu"] - b["load"])
    for on_alu in range(min(c["const_alu"], off_load) + 1):
        on_flow = off_load - on_alu
        room = b["flow"] - c["flow"] - on_flow
        if room < 0:
            continue
        s2 = min(c["select2"], room)
        s1 = min(c["select1"], room - s2)
        vector = c["vec_fixed"] + c["vec_shiftable"] + (c["select1"] - s1) + 2 * (c["select2"] - s2)
        shiftable = c["vec_shiftable"] + (c["select2"] - s2)
        alu_room = b["alu"] - c["alu"] - on_alu
        if alu_room < 0:
            continue
        if max(0, vector - b["valu"]) <= min(shiftable, alu_room // es.VLEN):
            return True
    return False


def capped_floor(counts: Mapping[str, int], caps: Mapping[str, float] | None = None) -> int:
    """Smallest T such that the counts fit with engine e limited to floor(caps[e] * cap_e * T)
    slots: `choice_floor` for no caps, the capped objective of `solve_plan` otherwise."""
    ratios = {e: Fraction(str(v)) for e, v in (caps or {}).items()}
    c = {k: int(counts[k]) for k in es.CATEGORIES}

    def budgets(t):
        return {e: math.floor(ratios.get(e, 1) * es.CAP[e] * t) for e in es.CAP}

    t = 1
    while not _fits_budgets(c, budgets(t)):
        t += 1
    return t


def plan_use(counts: Mapping[str, int], t: int) -> dict[str, float]:
    """Share of each cappable engine's capacity over `t` cycles the plan needs at its best
    resolution in `t` (VALU: shifting only what does not fit)."""
    res = es._feasible_at({k: int(counts[k]) for k in es.CATEGORIES}, t)
    if res is None:
        raise ValueError(f"counts do not fit in {t} cycles")
    c = counts
    loads = c["load"] + c["const"] + c["const_alu"] - res["const_on_flow"] - res["const_alu_on_alu"]
    flow = c["flow"] + res["select1_flow"] + res["select2_flow"] + res["const_on_flow"]
    vector = (c["vec_fixed"] + c["vec_shiftable"] + (c["select1"] - res["select1_flow"])
              + 2 * (c["select2"] - res["select2_flow"]))
    on_valu = min(vector, es.CAP["valu"] * t)
    return {"load": loads / (es.CAP["load"] * t), "flow": flow / (es.CAP["flow"] * t),
            "valu": on_valu / (es.CAP["valu"] * t)}


# ----------------------------------------------------------------------------
# Stage 2: build one plan and read its schedule
# ----------------------------------------------------------------------------
def schedule_summary(sched: Mapping) -> dict:
    """What the feedback reads: per-engine utilisation in 10 buckets, start-up / steady /
    tail averages, the scheduler's own floor and binding engine, and the exact static
    head-body-tail bound of the emitted ops."""
    from scripts import schedule_report as sr

    rep = sr.analyze(sched)
    pct = rep["utilisation"]["pct"]
    steady = {e: sum(v[1:-1]) / (len(v) - 2) for e, v in pct.items()}
    return {
        "utilisation_pct": pct,
        "startup_pct": {e: v[0] for e, v in pct.items()},
        "steady_pct": steady,
        "tail_pct": {e: v[-1] for e, v in pct.items()},
        "scheduled_floor": rep["floor"]["cycles"],
        "scheduled_floor_binds": rep["floor"]["binding"],
        "static_bound_any": (rep.get("static_bound") or {}).get("any", {}).get("bound"),
        "valu_shifted": rep["emission"]["vector_ops_shifted_to_alu"],
    }


class Stage2:
    """Builds plans (each at most once), proves and validates them, and keeps what the loop
    needs in a state file so that later runs reuse the builds."""

    def __init__(self, table: es.CostTable, workdir: str, state_path: str | None = None, log=print,
                 keep_files: bool = False, greedy_extra: Mapping | None = None):
        self.table = table
        self.ctx = table.context
        self.workdir = workdir
        self.state_path = state_path
        self.log = log or (lambda *a, **k: None)
        self.keep_files = keep_files
        # Scheduler options on top of the shaped flags. Part of the state: builds made with
        # other options are not comparable, so a state file refuses a different setting.
        self.greedy_extra = dict(greedy_extra or {})
        self.builds: dict[str, dict] = {}
        self.rowtable = RowTable(self.ctx)
        self.trace: list[dict] = []
        self.built_now = 0
        if state_path and os.path.exists(state_path):
            with open(state_path) as f:
                state = json.load(f)
            if es.Context(**state["context"]) != self.ctx:
                raise ValueError(f"{state_path} is for another context")
            if greedy_extra is not None and state.get("greedy_extra", {}) != self.greedy_extra:
                raise ValueError(f"{state_path} was built with greedy options {state.get('greedy_extra', {})}, "
                                 f"not {self.greedy_extra}")
            self.greedy_extra = state.get("greedy_extra", {})
            self.builds = state["builds"]
            self.rowtable = RowTable.from_json(state["row_table"])
            self.trace = state.get("trace", [])

    def save(self):
        if self.state_path:
            es.write_json_atomic(self.state_path, {"context": asdict(self.ctx), "greedy_extra": self.greedy_extra,
                                                   "builds": self.builds, "row_table": self.rowtable.to_json(),
                                                   "trace": self.trace})

    def best(self) -> dict | None:
        ok = [b for b in self.builds.values() if b["valid"]]
        return min(ok, key=lambda b: (b["cycles"], b["simulator_cycles"])) if ok else None

    def _failed(self, rows, name: str, key: str, error: str) -> dict:
        """Record a plan the scheduler could not build (e.g. it overflows the scratch budget)."""
        import re

        m = re.search(r"peak[_ ]live[^0-9]*(\d+)", error) or re.search(r"(\d+) (?:live )?words", error)
        counts = self.table.predict(rows)
        rec = {
            "name": name, "plan": describe_plan(rows), "rows": [list(r) for r in rows],
            "group_counts": counts_key(rows, self.ctx), "counts": counts,
            "predicted_floor": es.choice_floor(counts)["cycles"], "emitted_floor": None,
            "counts_match": None, "cycles": None, "simulator_cycles": None, "proved": False, "passed": False,
            "valid": False, "peak_live": int(m.group(1)) if m else None, "budget_exceeded": True,
            "error": error[:2000], "summary": None, "row_violations_on_merge": [], "kernel": None,
        }
        self.builds[key] = rec
        self.built_now += 1
        self.save()
        self.log(f"    FAILED {name}: {rec['plan']} -> floor {rec['predicted_floor']}: {error[:300]}")
        return rec

    def evaluate(self, rows, name: str, row_table_for_check: RowTable | None = None) -> tuple[dict, bool]:
        """(build record, built now?). Builds, proves (dataflow replay) and validates a plan
        not built before, observes its op classes and merges them into the row table."""
        key = plan_key(rows)
        if key in self.builds:
            return self.builds[key], False
        stem = os.path.splitext(os.path.basename(self.state_path or "loop"))[0]
        try:
            row = es.measure(rows, self.table, self.workdir, name=f"{stem}_b{len(self.builds)}", shaped=True,
                             upstream=False, greedy_extra=self.greedy_extra)
        except Exception as e:  # noqa: BLE001 - only a scratch-budget failure is a plan outcome
            if "budget" not in str(e).lower():
                raise
            return self._failed(rows, name, key, f"{type(e).__name__}: {e}"), True
        with open(row["sched_json"]) as f:
            sched = json.load(f)
        obs = observe(rows, sched, self.ctx)
        bound_before = self.rowtable.bound(rows)["bound"] if self.rowtable.classes else 0
        violations = self.rowtable.merge(obs, rows)
        bound_after = self.rowtable.bound(rows)["bound"]
        summary = schedule_summary(sched)
        del sched
        counts = self.table.predict(rows)
        rec = {
            "name": name, "plan": describe_plan(rows), "rows": [list(r) for r in rows],
            "group_counts": counts_key(rows, self.ctx), "counts": counts,
            "predicted_floor": row["predicted_floor"], "emitted_floor": row["emitted_floor"],
            "counts_match": row["counts_match"], "cycles": row["cycles"],
            "simulator_cycles": row["simulator_cycles"], "proved": row["proved"], "passed": row["passed"],
            "valid": bool(row["counts_match"] and row["proved"] and row["passed"]),
            "peak_live": row["peak_live"], "max_addr": row.get("max_addr"), "budget_exceeded": False,
            "summary": summary,
            "row_violations_on_merge": violations, "kernel": row["kernel"],
            # The plan's window bound with the row table before it was merged (what a solve
            # would have used) and after; before > after means the table overestimated it.
            "rows_bound_before_merge": bound_before, "rows_bound_after_merge": bound_after,
        }
        self.builds[key] = rec
        self.built_now += 1
        best = self.best()
        if not self.keep_files:  # 17 MB per scheduler JSON on a nearly full disk
            os.unlink(row["sched_json"])
            if best is not rec and os.path.exists(row["kernel"]):
                os.unlink(row["kernel"])
        self.save()
        self.log(f"    built {name}: {rec['plan']} -> floor {rec['predicted_floor']}, "
                 f"{rec['cycles']} / {rec['simulator_cycles']} cycles, peak live {rec['peak_live']}, "
                 f"proved={rec['proved']} passed={rec['passed']}")
        return rec, True


def solve(table: es.CostTable, **kwargs) -> dict:
    return es.solve_plan_isolated(table, **kwargs)


# ----------------------------------------------------------------------------
# The loops
# ----------------------------------------------------------------------------
def _record(stage2: Stage2, mode: str, it: int, res: dict, rec: dict | None, built: bool, feedback: str,
            extra: Mapping | None = None) -> dict:
    entry = {"mode": mode, "iter": it, "status": res.get("status"), "feedback": feedback, "built": built}
    if rec is not None:
        entry.update(plan=rec["plan"], floor=rec["predicted_floor"], cycles=rec["cycles"],
                     simulator_cycles=rec["simulator_cycles"], proved=rec["proved"], passed=rec["passed"])
    if "objective" in res:
        entry["objective"] = res["objective"]
        entry["capped_cycles"] = res["capped_cycles"]
    entry.update(extra or {})
    stage2.trace.append(entry)
    stage2.save()
    stage2.log(f"  [{mode} {it}] " + ", ".join(f"{k}={v}" for k, v in entry.items()
                                              if k not in ("mode", "iter")))
    return entry


def _built_nogoods(stage2: Stage2) -> list[dict]:
    return [b["group_counts"] for b in stage2.builds.values()]


def loop_nogood(stage2: Stage2, budget: int, patience: int, time_limit: float,
                max_util: Mapping[str, float] | None = None) -> dict:
    """(a): next plan by floor, every built plan cut off, floor < incumbent. With `max_util`
    (e.g. the caps loop's result) the objective is the capped floor and the incumbent bound is
    dropped (a capped floor is not a lower bound): a no-good neighbourhood search."""
    mode = "nogood" if not max_util else "nogood-capped"
    best = stage2.best()
    stale = 0
    for it in range(budget):
        incumbent = best["cycles"] if best and not max_util else None
        res = solve(stage2.table, nogoods=_built_nogoods(stage2), incumbent=incumbent, time_limit=time_limit,
                    max_util=dict(max_util or {}))
        if res["status"] == "infeasible":
            _record(stage2, mode, it, res, None, False, "exhausted: no unbuilt plan with floor < incumbent")
            break
        rec, built = stage2.evaluate(res["rows"], f"{mode}{it}")
        improved = rec["valid"] and (best is None or rec["cycles"] < best["cycles"])
        best = rec if improved else best
        stale = 0 if improved else stale + 1
        _record(stage2, mode, it, res, rec, built,
                "no-good on this plan" + (f"; caps {dict(max_util)}" if max_util else
                                          f"; incumbent floor < {best['cycles']}")
                + (" (new best)" if improved else ""))
        if stale >= patience:
            break
    return best


def loop_caps(stage2: Stage2, budget: int, patience: int, time_limit: float, step: float = 0.04,
              min_step: float = 0.01, saturated: float = 97.0, near_full: float = 0.9,
              rule: str = "saturated", mode: str = "caps") -> dict:
    """(b): learn per-engine caps. From the best plan so far, cap one engine the plan uses at
    >= `near_full` of capacity (its use under the plan's best resolution at its capped floor),
    at that use minus `step`. Which engine (steady state = the 10 utilisation buckets without
    the first and last):

    * rule "saturated": an engine the schedule runs at >= `saturated`% in its steady state,
      least-used first (it costs the floor least);
    * rule "underfilled": an engine the schedule fills less than the plan needs (steady
      utilisation below the plan's use by more than 2 points), largest shortfall first: the
      plan counts on slots the scheduler does not deliver.

    Keep the caps if cycles improve; otherwise try the next engine, then halve the step."""
    if rule not in ("saturated", "underfilled"):
        raise ValueError(f"caps rule {rule!r}")

    def pick(use, steady, skip):
        if rule == "saturated":
            es_ = [e for e in CAPPABLE if use[e] >= near_full and steady[e] >= saturated and e not in skip]
            return sorted(es_, key=lambda e: use[e])
        short = {e: use[e] - steady[e] / 100 for e in CAPPABLE}
        es_ = [e for e in CAPPABLE if use[e] >= near_full and short[e] > 0.02 and e not in skip]
        return sorted(es_, key=lambda e: -short[e])

    caps: dict[str, float] = {}
    best_caps: dict[str, float] = {}
    best_rec = None
    tried: set[str] = set()
    stale = 0
    for it in range(budget):
        res = solve(stage2.table, max_util=caps, time_limit=time_limit)
        rec, built = stage2.evaluate(res["rows"], f"{mode}{it}")
        improved = rec["valid"] and (best_rec is None or rec["cycles"] < best_rec["cycles"])
        if improved:
            best_rec, best_caps, tried, stale = rec, dict(caps), set(), 0
        else:
            stale += 1
        if best_rec is None:  # nothing valid yet (e.g. the plan overflowed the scratch budget)
            _record(stage2, mode, it, res, rec, built, f"caps {caps or 'none'}: no valid build; stop",
                    {"caps": dict(caps)})
            break
        use = plan_use(best_rec["counts"], capped_floor(best_rec["counts"], best_caps))
        steady = best_rec["summary"]["steady_pct"]
        candidates = pick(use, steady, tried)
        if not candidates and step / 2 >= min_step:
            step, tried = step / 2, set()
            candidates = pick(use, steady, set())
        feedback = (f"caps {caps or 'none'} -> {rec['cycles']}" + (" (new best)" if improved else "")
                    + "; best plan use " + ", ".join(f"{e} {use[e]:.3f}" for e in CAPPABLE)
                    + "; steady " + ", ".join(f"{e} {steady[e]:.0f}%" for e in CAPPABLE))
        if not candidates:
            _record(stage2, mode, it, res, rec, built, feedback + "; no engine left to cap",
                    {"caps": dict(caps)})
            break
        engine = candidates[0]
        tried.add(engine)
        new_cap = round(min(best_caps.get(engine, 1.0), use[engine]) - step, 4)
        next_caps = {**best_caps, engine: new_cap}
        _record(stage2, mode, it, res, rec, built, feedback + f"; next: cap {engine} at {new_cap}",
                {"caps": dict(caps)})
        caps = next_caps
        if stale >= patience:
            break
    return {"best": best_rec, "caps": best_caps}


def loop_rows(stage2: Stage2, budget: int, patience: int, time_limit: float) -> dict:
    """(c): minimise M (floor and window rows), every built plan cut off, M < incumbent.
    After each build the row table is lowered to what the plan showed (a refined cut), and
    the build is checked against the table its plan was solved with."""
    best = stage2.best()
    stale = 0
    base = es.baseline_rows(stage2.ctx)
    for it in range(budget):
        table_then = RowTable.from_json(stage2.rowtable.to_json())
        specs = table_then.window_specs(base)
        incumbent = best["cycles"] if best else None
        res = solve(stage2.table, windows=specs, nogoods=_built_nogoods(stage2), incumbent=incumbent,
                    time_limit=time_limit)
        if res["status"] == "infeasible":
            _record(stage2, "rows", it, res, None, False,
                    "exhausted: no unbuilt plan with M < incumbent (given the observed row table)",
                    {"windows": len(specs)})
            break
        predicted = table_then.bound(res["rows"])
        rec, built = stage2.evaluate(res["rows"], f"rows{it}")
        # The plan's own classes against the table its plan was solved with.
        violations = rec["row_violations_on_merge"] if built else []
        improved = rec["valid"] and (best is None or rec["cycles"] < best["cycles"])
        best = rec if improved else best
        stale = 0 if improved else stale + 1
        _record(stage2, "rows", it, res, rec, built,
                f"no-good on this plan; row table refined from it; M < {best['cycles']}"
                + (" (new best)" if improved else ""),
                {"windows": len(specs), "rows_bound": predicted["bound"], "rows_window": [predicted["t"], predicted["q"]],
                 "violations": len(violations),
                 "static_bound_any": (rec["summary"] or {}).get("static_bound_any")})
        if stale >= patience:
            break
    return best


# ----------------------------------------------------------------------------
# Surrogate quality
# ----------------------------------------------------------------------------
def _ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2
        i = j + 1
    return ranks


def pearson(xs, ys) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(sxx * syy)


def spearman(xs, ys) -> float | None:
    return pearson(_ranks(xs), _ranks(ys))


def surrogates(stage2: Stage2, caps: Mapping[str, float] | None) -> list[dict]:
    """Per valid build: each candidate objective next to its scheduled cycles."""
    out = []
    for b in stage2.builds.values():
        if not b["valid"]:
            continue
        rows = b["rows"]
        rb = stage2.rowtable.bound(rows)
        out.append({
            "plan": b["plan"], "name": b["name"], "cycles": b["cycles"],
            "floor": b["predicted_floor"],
            "floor_plus_rows": max(b["predicted_floor"], rb["bound"]),
            "capped_floor": capped_floor(b["counts"], caps) if caps else None,
            "capped_floor_flow90": capped_floor(b["counts"], {"flow": 0.9}),
            "static_bound_any": b["summary"]["static_bound_any"],
            # Calibrated heuristic of the load-tail greedy's cycles (es.GREEDY_MODEL), not a bound.
            "greedy_model": es.greedy_model_cycles(b["counts"]),
            "rows_le_cycles": max(b["predicted_floor"], rb["bound"]) <= b["cycles"],
            "peak_live": b.get("peak_live"), "max_addr": b.get("max_addr"),
            # Post-build diagnostic, not a stage-1 surrogate: vector work the greedy left on the
            # VALU (after its own shifts), in VALU cycles. Stage 1 assumes the best split instead.
            "valu_after_greedy_shift": math.ceil((b["counts"]["vec_fixed"] + b["counts"]["vec_shiftable"]
                                                  - b["summary"]["valu_shifted"]) / es.CAP["valu"]),
        })
    return out


def failed_builds(stage2: Stage2) -> list[dict]:
    """Plans the scheduler could not build within the scratch budget (or that failed a check)."""
    return [{"name": b["name"], "plan": b["plan"], "floor": b["predicted_floor"], "peak_live": b.get("peak_live"),
             "budget_exceeded": b.get("budget_exceeded"), "error": (b.get("error") or "")[:300]}
            for b in stage2.builds.values() if not b["valid"]]


def correlation_table(points: Sequence[Mapping], subset: str = "all") -> list[dict]:
    rows = []
    for name in ("floor", "floor_plus_rows", "capped_floor", "capped_floor_flow90", "static_bound_any",
                 "greedy_model", "valu_after_greedy_shift"):
        pts = [(p[name], p["cycles"]) for p in points if p.get(name) is not None]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        rows.append({"surrogate": name, "subset": subset, "n": len(pts), "pearson": pearson(xs, ys),
                     "spearman": spearman(xs, ys)})
    return rows


def format_correlations(rows) -> str:
    f = lambda v: "-" if v is None else f"{v:+.2f}"  # noqa: E731
    lines = ["| Surrogate | Plans | Pearson r | Spearman rho |", "|---|---|---|---|"]
    lines += [f"| {r['surrogate']} ({r['subset']}) | {r['n']} | {f(r['pearson'])} | {f(r['spearman'])} |" for r in rows]
    return "\n".join(lines)


def format_trace(trace) -> str:
    lines = ["| Mode | Iter | Plan | Floor | Objective | Cycles (sched / sim) | Built | Feedback |",
             "|---|---|---|---|---|---|---|---|"]
    for e in trace:
        cyc = f"{e['cycles']} / {e['simulator_cycles']}" if "cycles" in e else "-"
        lines.append(f"| {e['mode']} | {e['iter']} | {e.get('plan', '-')} | {e.get('floor', '-')} | "
                     f"{e.get('objective', '-')} | {cyc} | {'yes' if e['built'] else 'no'} | {e['feedback']} |")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def resolve_plan(spec: str, ctx: es.Context):
    if spec in es.NAMED_PLANS:
        return spec, es.flag_plan(ctx, *es.NAMED_PLANS[spec])
    if spec == "gather":
        return spec, es.baseline_rows(ctx)
    path = spec if os.path.isabs(spec) else os.path.join(ROOT, spec)
    return os.path.basename(spec), expand_plan(load_plan(path), ctx.rounds, ctx.n_groups, ctx.forest_height)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="emission_structure loop", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["run", "report"])
    ap.add_argument("--rounds", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--forest-height", type=int, default=10)
    ap.add_argument("--select-bit-source", choices=["index", "path"], default="index")
    ap.add_argument("--load-tail", action="store_true",
                    help="add the load-tail greedy flags to the shaped ones: " +
                    " ".join(f"{k}={v}" for k, v in es.LOAD_TAIL_GREEDY.items()) +
                    " (stored in the state; a state refuses a different setting)")
    ap.add_argument("--near", type=int, default=30,
                    help="report: the near-optimal subset is the valid builds within this many cycles of the best")
    ap.add_argument("--table", help="cost table JSON (default: the calibrated cache of the context)")
    ap.add_argument("--state", required=True, help="state JSON: builds, row table, trace (reused across runs)")
    ap.add_argument("--workdir", help="scheduler JSONs and kernels (default: next to --state)")
    ap.add_argument("--modes", default="nogood,caps,rows", help=f"comma-separated, from {MODES}")
    ap.add_argument("--seed-plan", action="append", default=None,
                    help=f"plans built before the loops (default: {', '.join(DEFAULT_SEEDS)})")
    ap.add_argument("--budget", type=int, default=8, help="iterations per mode")
    ap.add_argument("--patience", type=int, default=4, help="stop a mode after this many iterations without improvement")
    ap.add_argument("--nogood-caps", help="nogood: enumerate under these caps (e.g. flow=0.92), no incumbent bound")
    ap.add_argument("--max-builds", type=int, default=60, help="refuse to build more than this many plans in one run")
    ap.add_argument("--time-limit", type=float, default=60.0, help="stage-1 CP-SAT seconds per solve")
    ap.add_argument("--max-axis", type=int, default=None, help="window row grid: at most this many t and q values")
    ap.add_argument("--caps", help="report: caps for the capped-floor surrogate, e.g. flow=0.9 (default: the caps loop's best)")
    ap.add_argument("--keep-files", action="store_true", help="keep every scheduler JSON and kernel")
    ap.add_argument("--json", help="also write the trace and correlations here")
    ap.add_argument("--extra-state", action="append", default=[],
                    help="report: also count the builds (and row observations) of these state files")
    args = ap.parse_args(argv)
    ctx = es.Context(rounds=args.rounds, batch_size=args.batch_size, forest_height=args.forest_height,
                     select_bit_source=args.select_bit_source)
    workdir = args.workdir or os.path.join(os.path.dirname(os.path.abspath(args.state)), "builds")
    os.makedirs(workdir, exist_ok=True)
    table = es.load_or_calibrate(ctx, args.table, workdir)
    greedy_extra = es.LOAD_TAIL_GREEDY if args.load_tail else ({} if args.command == "run" else None)
    stage2 = Stage2(table, workdir, args.state, keep_files=args.keep_files, greedy_extra=greedy_extra)
    if args.max_axis:
        stage2.rowtable.max_axis = args.max_axis

    caps_found = None
    if args.command == "run":
        modes = [m for m in args.modes.split(",") if m]
        for m in modes:
            if m not in MODES:
                ap.error(f"unknown mode {m!r}")
        for spec in args.seed_plan if args.seed_plan is not None else DEFAULT_SEEDS:
            name, rows = resolve_plan(spec, ctx)
            stage2.evaluate(rows, f"seed {name}")
        budget_left = lambda: args.max_builds - stage2.built_now  # noqa: E731
        for m in modes:
            if budget_left() <= 0:
                print(f"build limit {args.max_builds} reached; skipping {m}")
                continue
            n_iter = min(args.budget, budget_left())
            print(f"== mode {m}: up to {n_iter} iterations ==", flush=True)
            if m == "nogood":
                loop_nogood(stage2, n_iter, args.patience, args.time_limit,
                            es.parse_max_util(args.nogood_caps) if args.nogood_caps else None)
            elif m in ("caps", "caps-underfilled"):
                rule = "saturated" if m == "caps" else "underfilled"
                caps_found = loop_caps(stage2, n_iter, args.patience, args.time_limit, rule=rule, mode=m)["caps"]
            else:
                loop_rows(stage2, n_iter, args.patience, args.time_limit)
        print(f"{stage2.built_now} plans built in this run, {len(stage2.builds)} in the state")
    if args.caps:
        caps_found = es.parse_max_util(args.caps)
    elif caps_found is None:
        caps_found = next((e["caps"] for e in reversed(stage2.trace)
                           if e["mode"].startswith("caps") and "(new best)" in e["feedback"]), None)
    for path in args.extra_state:
        other = Stage2(table, workdir, path, log=None)
        stage2.builds = {**other.builds, **stage2.builds}
        stage2.rowtable = stage2.rowtable.union(other.rowtable)
        stage2.trace = stage2.trace + other.trace
        stage2.state_path = None  # a merged view: never written back
    points = surrogates(stage2, caps_found)
    loop_names = [p for p in points if not p["name"].startswith("seed")]
    best = stage2.best()
    near = [p for p in points if best and p["cycles"] <= best["cycles"] + args.near]
    corr = (correlation_table(points, "all builds") + correlation_table(loop_names, "loop plans")
            + correlation_table(near, f"within {args.near} of best"))
    failed = failed_builds(stage2)
    print(format_trace(stage2.trace))
    print()
    print(f"capped floor uses caps {caps_found}")
    print(format_correlations(corr))
    bad = [p for p in points if not p["rows_le_cycles"]]
    loop_builds = [b for b in stage2.builds.values() if not b["name"].startswith("seed")]
    viol_plans = [b for b in loop_builds if b["row_violations_on_merge"]]
    over = [b for b in loop_builds if b.get("rows_bound_before_merge", 0) > b.get("rows_bound_after_merge", 0)]
    print(f"row bound (final table) above the scheduled cycles on {len(bad)} of {len(points)} plans")
    print(f"loop plans contradicting the row table learned before them: {len(viol_plans)} of {len(loop_builds)} "
          f"({sum(len(b['row_violations_on_merge']) for b in viol_plans)} class fields); "
          f"of those, the plan's own window bound was overestimated on {len(over)}")
    print(f"greedy options on top of the shaped flags: {stage2.greedy_extra or 'none'}")
    hist = collections.Counter(p["cycles"] for p in points)
    print("cycles over valid plans (cycles: plans): " + ", ".join(f"{c}: {n}" for c, n in sorted(hist.items())))
    peaks = [p["peak_live"] for p in points if p.get("peak_live") is not None]
    if peaks:
        print(f"peak live scratch over valid plans: {min(peaks)} to {max(peaks)} words")
    print(f"plans that failed (scratch budget or a check): {len(failed)}")
    for b in failed:
        print(f"  {b['name']}: {b['plan']} (floor {b['floor']}, peak {b['peak_live']}): {b['error'][:160]}")
    if best:
        print(f"best: {best['plan']} -> {best['cycles']} / {best['simulator_cycles']} cycles "
              f"(floor {best['predicted_floor']}, kernel {best['kernel']})")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"context": asdict(stage2.ctx), "greedy_extra": stage2.greedy_extra, "trace": stage2.trace,
                       "correlations": corr, "points": points, "failed": failed, "best": best,
                       "caps": caps_found}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
