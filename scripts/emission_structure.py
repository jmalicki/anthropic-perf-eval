#!/usr/bin/env python3
"""Emission-level structure model: choose the tree-access encoding of every group in every
round to minimise the capacity floor, then emit and schedule the chosen plan.

    uv run python -m scripts.emission_structure calibrate             # cost table from the emitter
    uv run python -m scripts.emission_structure check --plans 20      # predicted == emitted counts?
    uv run python -m scripts.emission_structure solve --out plan.json # best plan for the floor
    uv run python -m scripts.emission_structure predict --plan plan.json
    uv run python -m scripts.emission_structure measure --plan plan.json --shaped --upstream
    uv run python -m scripts.emission_structure derive                # table by instantiation == calibrated?

Design: docs/EMISSION_STRUCTURE_MODEL.md. In short:

* Counts. An emission is summarised by ten op counts (CATEGORIES): fixed vector ops, shiftable
  vector ops, scalar ALU, load, store and flow ops, and the choice ops whose engine the
  scheduler picks (SelectChoice with a 1- or 2-op multiply_add form, ConstChoice with or without
  an ALU form). `emission_counts` reads them from a scheduler JSON's SSA ops.
* Floor. `choice_floor` is the capacity floor of those counts with every choice resolved in
  the best way: the smallest T with loads, stores, flow, vector and scalar ALU work all fitting
  in T cycles (vector ops may run as 8 scalar ops). It ignores dependencies, latency and scratch.
* Cost model. Counts are linear in features of the plan (`plan_features`): how many groups
  use each bit-driven encoding at each depth, which per-level tables some group needs, and
  which rounds and depths gather at all. The coefficients are not written by hand: `calibrate`
  builds probe plans with the real emitter and solves for them exactly, and `check` builds
  further random plans and asserts predicted == emitted counts.
* Solve. A CP-SAT model over group counts per (round, encoding) minimises the floor exactly as
  `choice_floor` defines it; tie-breaks: fewest encodings mixed into a round, then fewest
  vector-op equivalents.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import math
import os
import random
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from xml.sax.saxutils import escape as xml_escape
from typing import Iterable, Mapping, Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HERE = os.path.dirname(os.path.abspath(__file__))
# scripts/bisect.py shadows the standard library's bisect; keep this directory off the path.
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
sys.path.insert(0, ROOT)

from solver.emission_plan import (  # noqa: E402
    BIT_DRIVEN, ENCODINGS, MAX_CACHED_DEPTH, allowed_encodings, describe_plan, expand_plan, load_plan,
    plan_document, round_depth,
)

# ----------------------------------------------------------------------------
# Counts and the floor
# ----------------------------------------------------------------------------
CATEGORIES = (
    "vec_fixed",      # vector ops that cannot run on the scalar ALU: multiply_add, vbroadcast
    "vec_shiftable",  # other vector ops: 1 VALU slot, or 8 scalar ALU slots
    "alu",            # scalar ALU ops
    "load",           # load, vload (not ConstChoice)
    "store",
    "flow",           # flow ops that are not SelectChoice
    "select1",        # SelectChoice: 1 flow vselect, or 1 multiply_add
    "select2",        # SelectChoice: 1 flow vselect, or SUB + multiply_add
    "const",          # ConstChoice: 1 load or 1 flow add_imm
    "const_alu",      # ConstChoice: 1 load, 1 flow add_imm, or 1 scalar ALU op
)
# Issue slots per cycle: the one Python copy (scripts/analyze_bound_gap.py, as schedule_report
# uses); scheduler_rs/src/architecture.rs holds the same numbers for AnthropicPerfEval.
from scripts.analyze_bound_gap import CAP  # noqa: E402
VLEN = 8
# Utilisation caps are used as exact ratios; at most 6 decimal places keeps CP-SAT's coefficients
# within 64 bits.
MAX_CAP_DENOMINATOR = 10 ** 6
# values_only: the cost model has one feature per gather/select pattern over the tail rounds.
MAX_TAIL_ROUNDS = 10
FIXED_VECTOR_OPCODES = ("multiply_add", "vbroadcast")


def emission_counts(sched: Mapping) -> dict[str, int]:
    """The ten op counts (CATEGORIES) of an emission, from a scheduler JSON's SSA ops.

    Needs a JSON written by a scheduler_rs that exports `select_fma_vector_ops` and
    `const_alu_derivable` on choice ops.
    """
    c = dict.fromkeys(CATEGORIES, 0)
    for op in sched["ssa_ops"]:
        opcode, engine = op["opcode"], op["engine"]
        if opcode == "select_choice":
            cost = op.get("select_fma_vector_ops")
            if cost not in (1, 2):
                raise ValueError("scheduler JSON lacks select_fma_vector_ops; rebuild scheduler_rs")
            c["select1" if cost == 1 else "select2"] += 1
        elif opcode == "const_choice":
            if "const_alu_derivable" not in op:
                raise ValueError("scheduler JSON lacks const_alu_derivable; rebuild scheduler_rs")
            c["const_alu" if op["const_alu_derivable"] else "const"] += 1
        elif engine == "valu":
            c["vec_fixed" if opcode in FIXED_VECTOR_OPCODES else "vec_shiftable"] += 1
        elif engine in ("alu", "load", "store", "flow"):
            c[engine] += 1
        else:
            raise ValueError(f"unexpected SSA op engine {engine!r} ({opcode})")
    return c


def vector_op_equivalents(counts: Mapping[str, int], resolution: Mapping[str, int] | None = None) -> float:
    """Vector ops plus scalar ALU ops / 8 for counts with choices resolved as in `resolution`
    (selects not on flow take their multiply_add form; consts on the ALU count 1/8)."""
    r = resolution or {"select1_flow": 0, "select2_flow": 0, "const_alu_on_alu": 0}
    vec = (counts["vec_fixed"] + counts["vec_shiftable"] + (counts["select1"] - r["select1_flow"])
           + 2 * (counts["select2"] - r["select2_flow"]))
    return vec + (counts["alu"] + r["const_alu_on_alu"]) / VLEN


def _feasible_at(c: Mapping[str, int], t: int, cap: Mapping[str, int] = CAP):
    """Best choice resolution with everything in `t` cycles under issue capacities `cap`, or
    None. See choice_floor."""
    if c["load"] > cap["load"] * t or c["store"] > cap["store"] * t or c["flow"] > cap["flow"] * t:
        return None
    consts = c["const"] + c["const_alu"]
    off_load = max(0, c["load"] + consts - cap["load"] * t)  # consts that must leave the load engine
    for on_alu in range(min(c["const_alu"], off_load) + 1):
        on_flow = off_load - on_alu
        room = cap["flow"] * t - c["flow"] - on_flow
        if room < 0:
            continue
        # A 2-op select moved to flow saves a fixed and a shiftable vector op, a 1-op select a
        # fixed one: 2-op first is never worse.
        s2 = min(c["select2"], room)
        s1 = min(c["select1"], room - s2)
        vector = (c["vec_fixed"] + c["vec_shiftable"] + (c["select1"] - s1) + 2 * (c["select2"] - s2))
        shiftable = c["vec_shiftable"] + (c["select2"] - s2)
        alu_room = cap["alu"] * t - c["alu"] - on_alu
        if alu_room < 0:
            continue
        lo = max(0, vector - cap["valu"] * t)
        hi = min(shiftable, alu_room // VLEN)
        if lo <= hi:
            return {"select1_flow": s1, "select2_flow": s2, "const_on_flow": on_flow,
                    "const_alu_on_alu": on_alu, "shifted": hi}
    return None


def choice_floor(counts: Mapping[str, int]) -> dict:
    """Capacity floor of an emission with every choice resolved in the best way.

    The smallest integer T such that, for some resolution of the choice ops and some number
    s of shiftable vector ops run as 8 scalar ops each:
        loads + consts on load      <= 2T      stores <= 2T
        flow + selects on flow + consts on flow <= T
        vector ops - s              <= 6T      scalar ops + consts on ALU + 8s <= 12T
    Ignores dependencies, latency and scratch, so no schedule of this emission is shorter. It is
    a floor for this emission only, never a bound on the problem.

    Returns:
        {"cycles", "binding", "resolution", "vector_op_equivalents"}; "binding" names the
        engines at the limit (load / flow: T - 1 does not fit them alone).
    """
    c = {k: int(counts[k]) for k in CATEGORIES}
    t = max(1, math.ceil(c["store"] / CAP["store"]), math.ceil(c["flow"] / CAP["flow"]),
            math.ceil(c["load"] / CAP["load"]))
    while _feasible_at(c, t) is None:
        t += 1
    res = _feasible_at(c, t)
    binding = []
    if t > 1:
        # An engine binds if giving it unlimited capacity would let T - 1 fit.
        for engine in ("load", "flow", "valu+alu"):
            lifted = ("valu", "alu") if engine == "valu+alu" else (engine,)
            relaxed = {k: (10 ** 9 if k in lifted else v) for k, v in CAP.items()}
            if _feasible_at(c, t - 1, relaxed) is not None:
                binding.append(engine)
    return {"cycles": t, "binding": binding, "resolution": res,
            "vector_op_equivalents": vector_op_equivalents(c, res)}


# Op kinds of a window row (scripts/emission_loop.py). A SelectChoice is charged as 1 flow slot
# or 1 fixed vector op whatever its FMA form (the 2-op form's subtract may be shared, as the
# proved reference model of verification/static_bounds_proof.py assumes); ConstChoices and
# eliminable vbroadcasts are not charged.
WINDOW_CATEGORIES = ("vec_fixed", "vec_shiftable", "alu", "load", "store", "flow", "select")


def _window_fits(c: Mapping[str, int], span: int, cap: Mapping[str, int], vlen: int) -> bool:
    if span < 0:
        return False
    if c["load"] > cap["load"] * span or c["store"] > cap["store"] * span or c["flow"] > cap["flow"] * span:
        return False
    on_flow = min(c["select"], cap["flow"] * span - c["flow"])  # a select on flow frees a VALU slot
    vector = c["vec_fixed"] + c["vec_shiftable"] + c["select"] - on_flow
    alu_room = cap["alu"] * span - c["alu"]
    if alu_room < 0:
        return False
    return max(0, vector - cap["valu"] * span) <= min(c["vec_shiftable"], alu_room // vlen)


def window_floor(counts: Mapping[str, int], cap: Mapping[str, int] = CAP, vlen: int = VLEN) -> int:
    """Fewest cycles that can hold ops with these WINDOW_CATEGORIES counts, every choice
    resolved in the best way (selects on flow or the VALU, shiftable vector ops on the VALU or
    as `vlen` scalar slots). 0 for no ops. Monotone in every count.

    A window row (docs/EMISSION_STRUCTURE_MODEL.md section 10) is makespan >= t + q +
    window_floor(ops with head >= t and tail >= q): their slots all lie in cycles
    [t, makespan - 1 - q]. verification/window_rows_proof.py: the coupling of engines in
    `_window_fits` is proved by Z3 for all counts (WR3); the whole row is a randomized check (WR1).
    """
    c = {k: int(counts.get(k, 0)) for k in WINDOW_CATEGORIES}
    if not any(c.values()):
        return 0
    lo, hi = 1, max(1, sum(c.values()))
    while not _window_fits(c, hi, cap, vlen):
        hi *= 2
    while lo < hi:
        mid = (lo + hi) // 2
        if _window_fits(c, mid, cap, vlen):
            hi = mid
        else:
            lo = mid + 1
    return lo


# ----------------------------------------------------------------------------
# Building probes with the real emitter
# ----------------------------------------------------------------------------
SHAPED_GREEDY = {"greedy_barriers": "none", "greedy_priority": "wavefront", "greedy_split_shift": True,
                 "pressure_inflation_threshold": 0.95}


# Calibration and check builds only read the SSA program, which the scheduler does not change;
# the shaped greedy is used because it fits the scratch budget for every plan tried.
PROBE_OPTIONS = {**SHAPED_GREEDY, "liveness_detail": "none"}


@dataclass(frozen=True)
class Context:
    """Everything outside the plan that changes the op counts."""
    rounds: int = 16
    batch_size: int = 256
    forest_height: int = 10
    merge_hash_stages: bool = True
    index_form: str = "mirrored"
    select_bit_source: str = "index"
    values_only: bool = False

    @property
    def n_groups(self) -> int:
        return self.batch_size // VLEN

    def build_options(self) -> dict:
        return asdict(self)

    def key(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]

    def depth(self, rnd: int) -> int:
        return round_depth(rnd, self.forest_height)

    def tail_rounds(self) -> list[int]:
        """Choice rounds after the last round that must gather. Only with `values_only` does
        their structure change the index work: no index is stored at the end, so a group's
        index updates after its last index read in these rounds are dead."""
        forced = [r for r in range(self.rounds) if self.depth(r) > MAX_CACHED_DEPTH]
        last = max(forced, default=-1)
        return [r for r in self.choice_rounds() if r > last]

    def tail_patterns(self) -> list[str]:
        """Per-group gather ('g') / select ('s') patterns over `tail_rounds`, when they matter.

        Raises:
            ValueError: for more than MAX_TAIL_ROUNDS tail rounds (2^k patterns, each a feature).
        """
        if not self.values_only:
            return []
        k = len(self.tail_rounds())
        if k > MAX_TAIL_ROUNDS:
            raise ValueError(f"values_only with {k} rounds after the last forced gather: the model enumerates "
                             f"2^{k} tail patterns; at most {MAX_TAIL_ROUNDS} rounds are supported")
        return ["".join(p) for p in itertools.product("gs", repeat=k)]

    def choice_rounds(self) -> list[int]:
        """Rounds whose encoding can be chosen (depth 1 to 4); depth 0 is the shared root and
        deeper levels can only gather."""
        return [r for r in range(self.rounds) if 1 <= self.depth(r) <= MAX_CACHED_DEPTH]


def build_emission(rows: Sequence[Sequence[str]], ctx: Context, workdir: str, name: str = "probe",
                   extra: Mapping | None = None):
    """Build a plan with the greedy scheduler. Returns (instrs, stats, sched dict, sched path)."""
    from kernel_build_args import build_kernel_with_args

    path = os.path.join(workdir, f"sched_{name}.json")
    options = dict(ctx.build_options(), use_greedy=True, emission_plan=[list(r) for r in rows])
    options.update(extra or {})
    instrs, stats = build_kernel_with_args(None, verbose=False, output_json=path, use_git_sha=False, **options)
    with open(path) as f:
        sched = json.load(f)
    return instrs, stats, sched, path


# ----------------------------------------------------------------------------
# Plan features (the hand-written structure; the coefficients come from the emitter)
# ----------------------------------------------------------------------------
def interactions(ctx: Context) -> list[tuple[str, ...]]:
    """Products of boolean features whose resources the emitter shares, so that their cost is
    not additive (each product is a feature "and:a+b[+c]"):

    * table:2, table:3, table:4 - the scalar constants 3 to 15 are derived from each other
      with ALU adds and feed the node addresses of levels 2, 3 and 4; one survives dead code
      elimination iff some level that needs it, directly or through another constant, is
      selected. Every subset of two or three of these levels gets a product term (the full
      multilinear form of a function of three booleans).

    Within one level no product is needed although pairs and Mobius coefficients share the
    broadcast of table entry 0: table:d = pairs:d OR mobius:d, so their product is already
    table:d - pairs:d - mobius:d and the shared op is in those three coefficients.
    """
    depths = sorted({ctx.depth(r) for r in ctx.choice_rounds()})
    tables = [f"table:{d}" for d in depths if d >= 2]
    return [combo for k in range(2, len(tables) + 1) for combo in itertools.combinations(tables, k)]


def interaction_name(combo: Sequence[str]) -> str:
    return "and:" + "+".join(combo)


def feature_names(ctx: Context) -> list[str]:
    names = ["base"]
    depths = sorted({ctx.depth(r) for r in ctx.choice_rounds()})
    for d in depths:
        names += [f"groups:{d}:{e}" for e in BIT_DRIVEN]
        names += [f"table:{d}", f"pairs:{d}", f"pairs_high:{d}", f"mobius:{d}", f"gather_depth:{d}"]
    names += [f"gather_round:{r}" for r in ctx.choice_rounds()]
    names += [interaction_name(c) for c in interactions(ctx)]
    # One pattern is the reference: the pattern counts always sum to the number of groups.
    names += [f"tail:{p}" for p in ctx.tail_patterns()[1:]]
    return names


def plan_features(rows: Sequence[Sequence[str]], ctx: Context) -> dict[str, int]:
    """Feature values of a plan. Only rounds at depth 1 to 4 matter: depth 0 always selects the
    shared root (every bit-driven encoding is the same there) and deeper rounds always gather.

        groups:d:e        groups using bit-driven encoding e at depth d, over all rounds
        table:d           1 if any group selects level d with the branch bits (node loads,
                          node table XORed with the hash constant, level constants)
        pairs:d           1 if any uses bits or bits-flow (leaf pair base/delta vectors)
        pairs_high:d      1 if any uses bits-flow (the pair's high vector)
        mobius:d          1 if any uses mobius (Mobius coefficients)
        gather_depth:d    1 if any group gathers at depth d (the level's end-pointer constant)
        gather_round:r    1 if any group gathers in round r (the round's address base vector)
        and:a+b[+c]       products for resources shared between features (`interactions`)
        tail:p            with values_only: groups whose gather/select pattern over the rounds
                          after the last forced gather is p (which index updates are dead)
    Gathering groups have no feature of their own: per depth the group counts sum to a
    constant, so their cost is in the other coefficients and in "base".
    """
    f = dict.fromkeys(feature_names(ctx), 0)
    f["base"] = 1
    for r in ctx.choice_rounds():
        d = ctx.depth(r)
        row = rows[r]
        for e in BIT_DRIVEN:
            f[f"groups:{d}:{e}"] += row.count(e)
        used = set(row)
        if used & set(BIT_DRIVEN):
            f[f"table:{d}"] = 1
        if used & {"bits", "bits-flow"}:
            f[f"pairs:{d}"] = 1
        if "bits-flow" in used:
            f[f"pairs_high:{d}"] = 1
        if "mobius" in used:
            f[f"mobius:{d}"] = 1
        if "gather" in used:
            f[f"gather_depth:{d}"] = 1
            f[f"gather_round:{r}"] = 1
    for combo in interactions(ctx):
        f[interaction_name(combo)] = int(all(f[name] for name in combo))
    if ctx.tail_patterns():
        tail = ctx.tail_rounds()
        for g in range(ctx.n_groups):
            key = "tail:" + "".join("g" if rows[r][g] == "gather" else "s" for r in tail)
            if key in f:  # the all-gather reference pattern has no feature
                f[key] += 1
    return f


# ----------------------------------------------------------------------------
# Cost table: calibration and prediction
# ----------------------------------------------------------------------------
@dataclass
class CostTable:
    context: Context
    coefficients: dict[str, dict[str, int]]          # feature -> category -> count
    calibration_plans: int = 0
    checked_plans: int = 0
    notes: list[str] = field(default_factory=list)

    def predict(self, rows: Sequence[Sequence[str]]) -> dict[str, int]:
        f = plan_features(rows, self.context)
        return {c: sum(f[name] * coef[c] for name, coef in self.coefficients.items()) for c in CATEGORIES}

    def to_json(self) -> dict:
        return {"context": asdict(self.context), "coefficients": self.coefficients,
                "calibration_plans": self.calibration_plans, "checked_plans": self.checked_plans,
                "notes": self.notes}

    @classmethod
    def from_json(cls, d: Mapping) -> "CostTable":
        return cls(Context(**d["context"]), {k: dict(v) for k, v in d["coefficients"].items()},
                   d.get("calibration_plans", 0), d.get("checked_plans", 0), list(d.get("notes", [])))


def uniform_rows(ctx: Context, rule) -> list[list[str]]:
    """Plan with one encoding per round: rule(round, depth) -> name."""
    return [[rule(r, ctx.depth(r))] * ctx.n_groups for r in range(ctx.rounds)]


def baseline_rows(ctx: Context) -> list[list[str]]:
    """Root selected with bits, every other round gathers."""
    return uniform_rows(ctx, lambda r, d: "bits" if d == 0 else "gather")


def random_rows(ctx: Context, rng: random.Random) -> list[list[str]]:
    """A random plan: per choice round, either one encoding for all groups or a random mix of
    two to four encodings in contiguous runs (or shuffled)."""
    rows = baseline_rows(ctx)
    for r in ctx.choice_rounds():
        options = list(allowed_encodings(ctx.depth(r)))
        kind = rng.random()
        if kind < 0.2:
            rows[r] = ["gather"] * ctx.n_groups
        elif kind < 0.45:
            rows[r] = [rng.choice(options)] * ctx.n_groups
        else:
            k = rng.randint(2, len(options))
            chosen = rng.sample(options, k)
            row = [rng.choice(chosen) for _ in range(ctx.n_groups)]
            for i, e in enumerate(chosen[:ctx.n_groups]):  # every chosen encoding appears
                row[i] = e
            if rng.random() < 0.5:
                row.sort(key=chosen.index)
            rows[r] = row
    return rows


def design_plans(ctx: Context, n_random: int, seed: int) -> list[list[list[str]]]:
    """Calibration plans: the baseline, each (round, encoding) alone over the baseline for 1 and
    all groups, each encoding on every subset of the cached levels, pairs and Mobius together
    at each level, and random mixes."""
    plans = [baseline_rows(ctx)]
    for r in ctx.choice_rounds():
        for e in BIT_DRIVEN:
            for k in (1, ctx.n_groups):
                rows = baseline_rows(ctx)
                rows[r] = [e] * k + ["gather"] * (ctx.n_groups - k)
                plans.append(rows)
    # Every subset of cached levels selected (others gathered), with each bit-driven encoding:
    # identifies the shared-constant products of `interactions`.
    depths = sorted({ctx.depth(r) for r in ctx.choice_rounds()})
    for k in range(len(depths) + 1):
        for subset in itertools.combinations(depths, k):
            for e in BIT_DRIVEN:
                plans.append(uniform_rows(ctx, lambda r, d, e=e, s=subset: e if d == 0 or d in s else "gather"))
    # Pairs and Mobius coefficients at the same level, alone (needs two groups).
    for d in depths if ctx.n_groups >= 2 else []:
        r = min(r for r in ctx.choice_rounds() if ctx.depth(r) == d)
        for pair_mode in ("bits", "bits-flow"):
            rows = baseline_rows(ctx)
            rows[r] = [pair_mode, "mobius"] + ["gather"] * (ctx.n_groups - 2)
            plans.append(rows)
    rng = random.Random(seed)
    plans += [random_rows(ctx, rng) for _ in range(n_random)]
    unique, seen = [], set()
    for p in plans:
        key = json.dumps(p)
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def _solve_exact(xs: list[list[int]], ys: list[int]) -> tuple[list[int] | None, list[int]]:
    """Integer coefficients w with xs @ w == ys exactly, by Gaussian elimination over the
    rationals. Returns (w or None if inconsistent or not integral, free columns)."""
    from fractions import Fraction

    m, n = len(xs), len(xs[0])
    a = [[Fraction(v) for v in row] + [Fraction(y)] for row, y in zip(xs, ys)]
    pivots, row = [], 0
    for col in range(n):
        piv = next((i for i in range(row, m) if a[i][col] != 0), None)
        if piv is None:
            continue
        a[row], a[piv] = a[piv], a[row]
        inv = 1 / a[row][col]
        a[row] = [v * inv for v in a[row]]
        for i in range(m):
            if i != row and a[i][col] != 0:
                factor = a[i][col]
                a[i] = [vi - factor * vr for vi, vr in zip(a[i], a[row])]
        pivots.append(col)
        row += 1
    if any(all(v == 0 for v in a[i][:n]) and a[i][n] != 0 for i in range(row, m)):
        return None, []
    free = [c for c in range(n) if c not in pivots]
    w = [Fraction(0)] * n
    for i, col in enumerate(pivots):
        w[col] = a[i][n]
    if any(v.denominator != 1 for v in w):
        return None, free
    return [int(v) for v in w], free


def calibrate(ctx: Context, workdir: str, n_random: int = 40, seed: int = 1, log=print) -> CostTable:
    """Fit the cost table from emitter builds of the design plans; exact or an error."""
    names = feature_names(ctx)
    plans = design_plans(ctx, n_random, seed)
    xs, ys, built, skipped = [], [], [], []
    for i, rows in enumerate(plans):
        try:
            _, _, sched, _ = build_emission(rows, ctx, workdir, name=f"cal{i}", extra=PROBE_OPTIONS)
        except RuntimeError as e:  # e.g. scratch budget exceeded: no SSA JSON for this plan
            skipped.append(f"cal{i}: {e}")
            continue
        f = plan_features(rows, ctx)
        xs.append([f[n] for n in names])
        ys.append(emission_counts(sched))
        built.append(rows)
        if log and (i + 1) % 10 == 0:
            log(f"  calibration: {i + 1}/{len(plans)} plans built")
    if not built:
        raise RuntimeError(f"calibration: none of the {len(plans)} design plans built; first errors: "
                           + "; ".join(skipped[:3]))
    coefficients = {n: {} for n in names}
    notes = []
    for c in CATEGORIES:
        w, free = _solve_exact(xs, [y[c] for y in ys])
        if w is None:
            raise RuntimeError(f"no exact integer fit for {c}: the emission is not linear in the plan "
                               f"features (feature_names); refine the structure")
        if free:
            notes.append(f"{c}: features {[names[j] for j in free]} not identified by the design plans (set to 0)")
        for n, v in zip(names, w):
            coefficients[n][c] = v
    if skipped:
        notes.append(f"{len(skipped)} design plans skipped (did not build): " + "; ".join(skipped[:3]))
    table = CostTable(ctx, coefficients, calibration_plans=len(built), notes=notes)
    for rows, y in zip(built, ys):  # exactness on the design itself
        assert table.predict(rows) == y
    return table


def check_table(table: CostTable, workdir: str, n_plans: int, seed: int, log=print) -> dict:
    """Build `n_plans` fresh random plans and compare emitted with predicted counts.

    Returns:
        {"compared": plans built and compared, "bad": the ones whose counts differ (empty = every
        prediction exact), "skipped": build errors of plans that did not build}.
    """
    rng = random.Random(seed)
    bad, skipped, compared = [], [], 0
    for i in range(n_plans):
        rows = random_rows(table.context, rng)
        try:
            _, _, sched, _ = build_emission(rows, table.context, workdir, name=f"chk{i}", extra=PROBE_OPTIONS)
        except RuntimeError as e:  # e.g. does not fit the scratch budget: nothing to compare
            skipped.append(f"chk{i}: {e}")
            continue
        compared += 1
        got, want = emission_counts(sched), table.predict(rows)
        if got != want:
            bad.append({"plan": rows, "emitted": got, "predicted": want})
    table.checked_plans += compared
    return {"compared": compared, "bad": bad, "skipped": skipped}


def write_json_atomic(path: str, obj) -> None:
    """Write JSON to `path` through a temporary file in the same directory, so a concurrent
    reader sees either the old file or the complete new one."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=1)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def default_table_path(ctx: Context) -> str:
    return os.path.join(ROOT, "artifacts", "emission_structure", f"costs_{ctx.key()}.json")


def derive_table(ctx: Context) -> tuple[CostTable, dict]:
    """Cost table derived by instantiation in scheduler_rs (`derive_tree_access_cost_table`):
    the canonical program with every alternative of every site built next to it, and each
    op's liveness reduced to the features of `feature_names`. No probe builds, no fit.

    Returns:
        (the table, the full JSON document: sites, resources, facts, problems).

    Raises:
        RuntimeError: if the derivation reports problems (something that does not fit the
            features) or produces a feature the model does not have.
    """
    import scheduler_rs
    from kernel_build_args import n_nodes_for_height

    doc = json.loads(scheduler_rs.derive_tree_access_cost_table(
        ctx.forest_height, n_nodes_for_height(ctx.forest_height), ctx.batch_size, ctx.rounds,
        merge_hash_stages=ctx.merge_hash_stages, index_form=ctx.index_form,
        select_bit_source=ctx.select_bit_source, values_only=ctx.values_only))
    if doc["problems"]:
        raise RuntimeError(f"derived cost table does not fit the features ({len(doc['problems'])} problems): "
                           + "; ".join(doc["problems"][:5]))
    names = feature_names(ctx)
    unknown = sorted(set(doc["coefficients"]) - set(names))
    if unknown:
        raise RuntimeError(f"derived cost table has features the model lacks: {unknown}")
    not_derived = [n for n in names if n not in doc["coefficients"]]
    if not_derived:  # e.g. tail:* (values_only): never zero-filled
        raise RuntimeError(f"derived cost table does not derive {len(not_derived)} features of the model: "
                           f"{not_derived[:5]}")
    coefficients = {n: {c: int(doc["coefficients"].get(n, {}).get(c, 0)) for c in CATEGORIES} for n in names}
    return CostTable(ctx, coefficients, notes=["derived by instantiation (scheduler_rs)"]), doc


def table_differences(a: CostTable, b: CostTable) -> list[str]:
    """Entries (feature, category) where two cost tables differ, as readable lines."""
    diffs = []
    for name in sorted(set(a.coefficients) | set(b.coefficients)):
        for c in CATEGORIES:
            va, vb = a.coefficients.get(name, {}).get(c, 0), b.coefficients.get(name, {}).get(c, 0)
            if va != vb:
                diffs.append(f"{name} {c}: {va} vs {vb}")
    return diffs


def load_or_calibrate(ctx: Context, path: str | None, workdir: str, log=print) -> CostTable:
    path = path or default_table_path(ctx)
    if os.path.exists(path):
        with open(path) as f:
            table = CostTable.from_json(json.load(f))
        if table.context == ctx:
            return table
        log(f"  {path} is for another context; recalibrating")
    log(f"  calibrating the cost table for {asdict(ctx)} ...")
    table = calibrate(ctx, workdir, log=log)
    write_json_atomic(path, table.to_json())
    log(f"  cost table written to {path}")
    return table


# ----------------------------------------------------------------------------
# The optimisation model
# ----------------------------------------------------------------------------
def plan_group_counts(rows: Sequence[Sequence[str]], ctx: Context) -> dict[tuple[int, str], int]:
    """Groups per (choice round, encoding) of a plan: what `solve_plan` decides (its rows are
    rebuilt from these counts in a fixed order, so two plans with equal counts are one plan
    to the model; with `values_only` the tail patterns are part of the decision too, see
    `plan_model_key`)."""
    return {(r, e): rows[r].count(e) for r in ctx.choice_rounds() for e in set(rows[r])}


def plan_tail_pattern_counts(rows: Sequence[Sequence[str]], ctx: Context) -> dict[str, int]:
    """Groups per tail pattern ('g'/'s' per tail round), {} when patterns do not matter."""
    if not ctx.tail_patterns():
        return {}
    tail = ctx.tail_rounds()
    out: dict[str, int] = collections.Counter(
        "".join("g" if rows[r][g] == "gather" else "s" for r in tail) for g in range(ctx.n_groups))
    return dict(out)


def plan_model_key(rows: Sequence[Sequence[str]], ctx: Context) -> dict[str, int]:
    """Everything `solve_plan` decides about a plan, as JSON-friendly keys: "round:encoding"
    group counts and, with `values_only`, "tail:pattern" counts. Two plans with the same key
    are the same plan to the model; `canonical_rows` is the arrangement it emits for it."""
    key = {f"{r}:{e}": v for (r, e), v in plan_group_counts(rows, ctx).items()}
    key.update({f"tail:{p}": v for p, v in plan_tail_pattern_counts(rows, ctx).items()})
    return key


def rows_from_counts(ctx: Context, chosen: Mapping[tuple[int, str], int], pattern_values: Mapping[str, int],
                     fixed: Mapping[int, str] | None = None) -> list[list[str]]:
    """The rows `solve_plan` emits for solved counts: per choice round, contiguous runs in
    ENCODINGS order; with tail patterns, each group its pattern (in pattern order) and the
    select groups of each tail round their bit-driven encodings in a fixed order."""
    g = ctx.n_groups
    rows = baseline_rows(ctx)
    for r, e in (fixed or {}).items():  # rounds outside the choice rounds (e.g. the root) keep their fixed encoding
        rows[r] = [e] * g
    for r in ctx.choice_rounds():
        rows[r] = [e for e in ENCODINGS for _ in range(chosen.get((r, e), 0))]
    patterns = ctx.tail_patterns()
    if patterns:
        group_pattern = [p for p in patterns for _ in range(pattern_values.get(p, 0))]
        for i, r in enumerate(ctx.tail_rounds()):
            selects = [e for e in BIT_DRIVEN for _ in range(chosen.get((r, e), 0))]
            rows[r] = ["gather" if p[i] == "g" else selects.pop(0) for p in group_pattern]
    return rows


def canonical_rows(rows: Sequence[Sequence[str]], ctx: Context) -> list[list[str]]:
    """The arrangement `solve_plan` would emit for this plan's `plan_model_key`. A plan equal
    to it is the one the model can return; any other arrangement with the same key (a
    hand-made seed, say) is a different build but the same model point."""
    return rows_from_counts(ctx, plan_group_counts(rows, ctx), plan_tail_pattern_counts(rows, ctx))


def solve_plan(table: CostTable, encodings: Iterable[str] = ENCODINGS, fixed: Mapping[int, str] | None = None,
               time_limit: float = 60.0, tie_break: bool = True, max_util: Mapping[str, float] | None = None,
               lateness: bool = True, nogoods: Sequence[Mapping] | None = None, incumbent: int | None = None,
               windows: Sequence[Mapping] | None = None, floor_incumbent: int | None = None,
               deterministic_time: float | None = None) -> dict:
    """Plan minimising choice_floor of the predicted counts (CP-SAT, exact, single worker so
    that the same inputs give the same plan).

    Args:
        table: Calibrated cost table (its context fixes rounds, groups, global options).
        encodings: Encodings the plan may use.
        fixed: round -> encoding for rounds to keep uniform with that encoding.
        time_limit: Per solve (two solves: floor, then tie-breaks at that floor).
        tie_break: After the floor, minimise (lexicographically) the number of extra encodings
            mixed into rounds, then the round-weighted count of groups that select instead of
            gathering (select work goes to the earliest visit of a level: the last rounds form
            the schedule's tail, with nothing after them to overlap), then vector-op
            equivalents (x8, in scalar units).
        lateness: Include the lateness term in the tie-break (off reproduces the plans of
            docs/EMISSION_STRUCTURE_MODEL.md section 5 "without the tie-break").
        max_util: engine ("load", "flow", "valu") -> fraction of its T-cycle capacity the plan
            may use. Minimises T under that headroom instead of the bare floor: a greedy
            schedule lands far above the floor when two engines saturate at once
            (docs/EMISSION_STRUCTURE_MODEL.md). The plan's own floor is reported too.
        nogoods: plans to exclude, each as `plan_model_key` (keys "round:encoding" strings or
            (round, encoding) tuples, plus "tail:pattern" with values_only); one no-good cut
            per plan. It excludes every arrangement with that key, so only give it plans whose
            rows are `canonical_rows` (the ones this model emits for the key).
        incumbent: the objective (T, or M with `windows`) must be below this. Prunes soundly
            only when the objective is a lower bound on the cycles being compared (with
            `windows` built from an observed row table it is not: a heuristic cut).
        floor_incumbent: the capacity floor T must be below this. Sound when compared with
            scheduled cycles of the same emission (the floor is a lower bound on any schedule).
        deterministic_time: CP-SAT deterministic-time budget per solve (the solver's own work
            counter), so that the same inputs give the same plan whatever the machine load;
            `time_limit` stays as a wall-clock backstop. Each solve's status and whether it
            stopped on the wall clock are returned under "solves".
        windows: time-aware rows (scripts/emission_loop.py, `RowTable.window_specs`), each
            {"t", "q", "const": {cat: n}, "per_group": {(round, encoding) or "r:e": {cat: n}}}.
            With them the objective is M >= T and, per window, M >= t + q + the window's
            choice floor (`window_floor`) of the counts it holds; otherwise T.

    Returns:
        {"rows", "floor" (choice_floor of the plan), "capped_cycles" (T under max_util),
        "objective" (T, or M with windows), "counts", "status", "tie_break", "solves" (per
        CP-SAT solve: phase, status, wall and deterministic time, stopped_on_wall_clock)};
        with nogoods or an incumbent, {"status": "infeasible", "solves"} when no plan is left.
    """
    ctx = table.context
    g = ctx.n_groups
    wanted = set(encodings)  # once: `encodings` may be an iterator
    encodings = [e for e in ENCODINGS if e in wanted]
    fixed = dict(fixed or {})
    for r, e in fixed.items():
        if not 0 <= r < ctx.rounds or e not in allowed_encodings(ctx.depth(r)):
            raise ValueError(f"fixed: {e!r} is not available in round {r} (depth {ctx.depth(r)})"
                             if 0 <= r < ctx.rounds else f"fixed: no round {r}")

    # Imported only after the arguments are checked: in a process that has loaded scheduler_rs
    # this import crashes (see solve_plan_isolated), and a bad argument should fail cleanly.
    from ortools.sat.python import cp_model

    m = cp_model.CpModel()
    n, u = {}, {}
    for r in ctx.choice_rounds():
        allowed = [e for e in allowed_encodings(ctx.depth(r)) if e in encodings]
        if r in fixed:
            allowed = [fixed[r]]
        if not allowed:
            raise ValueError(f"no allowed encoding for round {r}")
        for e in allowed:
            n[r, e] = m.NewIntVar(0, g, f"n_{r}_{e}")
            u[r, e] = m.NewBoolVar(f"u_{r}_{e}")
            m.Add(n[r, e] <= g * u[r, e])
            m.Add(n[r, e] >= u[r, e])
        m.Add(sum(n[r, e] for e in allowed) == g)

    def any_of(name, lits):
        lits = list(lits)
        b = m.NewBoolVar(name)
        if lits:
            m.AddMaxEquality(b, lits)
        else:
            m.Add(b == 0)
        return b

    feat = {"base": 1}
    for r in ctx.choice_rounds():
        feat[f"gather_round:{r}"] = u.get((r, "gather"), 0)
    for d in sorted({ctx.depth(r) for r in ctx.choice_rounds()}):
        rounds = [r for r in ctx.choice_rounds() if ctx.depth(r) == d]
        for e in BIT_DRIVEN:
            feat[f"groups:{d}:{e}"] = sum(n[r, e] for r in rounds if (r, e) in n)
        lit = lambda es: [u[r, e] for r in rounds for e in es if (r, e) in u]  # noqa: E731
        feat[f"table:{d}"] = any_of(f"table_{d}", lit(BIT_DRIVEN))
        feat[f"pairs:{d}"] = any_of(f"pairs_{d}", lit(("bits", "bits-flow")))
        feat[f"pairs_high:{d}"] = any_of(f"high_{d}", lit(("bits-flow",)))
        feat[f"mobius:{d}"] = any_of(f"mobius_{d}", lit(("mobius",)))
        feat[f"gather_depth:{d}"] = any_of(f"gdepth_{d}", lit(("gather",)))
    for combo in interactions(ctx):
        b = m.NewBoolVar(interaction_name(combo))
        m.AddMinEquality(b, [feat[name] for name in combo])
        feat[interaction_name(combo)] = b
    # Groups are interchangeable, so pattern counts consistent with the per-round gather counts
    # can always be realised by some assignment of patterns to groups.
    tail, patterns = ctx.tail_rounds(), ctx.tail_patterns()
    pattern_count = {}
    if patterns:
        for p in patterns:
            pattern_count[p] = m.NewIntVar(0, g, f"tail_{p}")
            feat[f"tail:{p}"] = pattern_count[p]  # the reference pattern's entry is unused
        m.Add(sum(pattern_count.values()) == g)
        for i, r in enumerate(tail):
            m.Add(sum(pattern_count[p] for p in patterns if p[i] == "g") == n.get((r, "gather"), 0))
    cnt = {c: sum(feat[name] * coef[c] for name, coef in table.coefficients.items() if coef[c]) for c in CATEGORIES}

    big = 10 * (sum(abs(v) for coef in table.coefficients.values() for v in coef.values()) * g + 1)
    # Caps as exact ratios (no rounding); T may need up to big / (smallest cap) cycles.
    caps = {}
    for engine, frac in (max_util or {}).items():
        ratio = Fraction(str(frac))
        if engine not in ("load", "flow", "valu") or not 0 < ratio <= 1:
            raise ValueError(f"max_util: {engine}={frac}; engines load, flow, valu, fraction in (0, 1]")
        if ratio.denominator > MAX_CAP_DENOMINATOR:  # e.g. 0.3333333333333333: coefficients would overflow
            raise ValueError(f"max_util: {engine}={frac} has more than 6 decimal places; give the cap with at "
                             f"most 6 (it is used exactly, never rounded)")
        caps[engine] = ratio
    t_max = big if not caps else math.ceil(big / min(caps.values())) + 1
    for engine, ratio in caps.items():
        # CP-SAT rejects a linear constraint whose terms can exceed 64 bits.
        if max(ratio.denominator * big * 4, ratio.numerator * CAP[engine] * t_max) >= 2 ** 62:
            raise ValueError(f"max_util: {engine}={ratio} makes the cap constraint too large for CP-SAT "
                             f"(T up to {t_max:,}); use a larger cap")
    t = m.NewIntVar(1, t_max, "T")
    s1 = m.NewIntVar(0, big, "select1_flow")
    s2 = m.NewIntVar(0, big, "select2_flow")
    cf = m.NewIntVar(0, big, "const_flow")
    ca = m.NewIntVar(0, big, "const_alu_on_alu")
    cfa = m.NewIntVar(0, big, "const_alu_flow")
    sh = m.NewIntVar(0, big, "shifted")
    m.Add(s1 <= cnt["select1"])
    m.Add(s2 <= cnt["select2"])
    m.Add(cf <= cnt["const"])
    m.Add(ca + cfa <= cnt["const_alu"])
    m.Add(cnt["load"] + (cnt["const"] - cf) + (cnt["const_alu"] - ca - cfa) <= CAP["load"] * t)
    m.Add(cnt["store"] <= CAP["store"] * t)
    m.Add(cnt["flow"] + s1 + s2 + cf + cfa <= CAP["flow"] * t)
    vector = cnt["vec_fixed"] + cnt["vec_shiftable"] + (cnt["select1"] - s1) + 2 * (cnt["select2"] - s2)
    m.Add(vector - sh <= CAP["valu"] * t)
    m.Add(cnt["alu"] + ca + VLEN * sh <= CAP["alu"] * t)
    m.Add(sh <= cnt["vec_shiftable"] + (cnt["select2"] - s2))
    used = {"load": cnt["load"] + (cnt["const"] - cf) + (cnt["const_alu"] - ca - cfa),
            "flow": cnt["flow"] + s1 + s2 + cf + cfa, "valu": vector - sh}
    for engine, ratio in caps.items():
        m.Add(ratio.denominator * used[engine] <= ratio.numerator * CAP[engine] * t)

    def n_key(key):
        if isinstance(key, str):
            r, _, e = key.partition(":")
            return int(r), e
        return int(key[0]), key[1]

    for i, plan in enumerate(nogoods or []):
        want = {n_key(k): int(v) for k, v in plan.items() if not (isinstance(k, str) and k.startswith("tail:"))}
        want_tail = {k[len("tail:"):]: int(v) for k, v in plan.items() if isinstance(k, str) and k.startswith("tail:")}
        differs = []
        for key, var in n.items():
            b = m.NewBoolVar(f"nogood_{i}_{key[0]}_{key[1]}")
            m.Add(var != want.get(key, 0)).OnlyEnforceIf(b)
            differs.append(b)
        if want_tail:  # with values_only the tail patterns are part of the plan
            for p, var in pattern_count.items():
                b = m.NewBoolVar(f"nogood_{i}_tail_{p}")
                m.Add(var != want_tail.get(p, 0)).OnlyEnforceIf(b)
                differs.append(b)
        # A key the model cannot express (an encoding not allowed here) already differs.
        if all(k in n or v == 0 for k, v in want.items()) and all(p in pattern_count or v == 0
                                                                  for p, v in want_tail.items()):
            m.AddBoolOr(differs)

    objective = t
    if windows:
        objective = m.NewIntVar(1, t_max + 10 ** 6, "M")
        m.Add(objective >= t)
        for w_i, w in enumerate(windows):
            per = {n_key(k): v for k, v in w["per_group"].items()}
            cnt_w = {}
            for c in WINDOW_CATEGORIES:
                expr = int(w["const"].get(c, 0))
                for key, coef in per.items():
                    if coef.get(c) and key in n:
                        expr = expr + int(coef[c]) * n[key]
                cnt_w[c] = expr
            span = objective - int(w["t"]) - int(w["q"])
            sf = m.NewIntVar(0, big, f"w{w_i}_select_flow")
            sh = m.NewIntVar(0, big, f"w{w_i}_shifted")
            m.Add(sf <= cnt_w["select"])
            m.Add(sh <= cnt_w["vec_shiftable"])
            m.Add(cnt_w["load"] <= CAP["load"] * span)
            m.Add(cnt_w["store"] <= CAP["store"] * span)
            m.Add(cnt_w["flow"] + sf <= CAP["flow"] * span)
            m.Add(cnt_w["vec_fixed"] + cnt_w["vec_shiftable"] + cnt_w["select"] - sf - sh <= CAP["valu"] * span)
            m.Add(cnt_w["alu"] + VLEN * sh <= CAP["alu"] * span)
    if incumbent is not None:
        m.Add(objective <= int(incumbent) - 1)
    if floor_incumbent is not None:
        m.Add(t <= int(floor_incumbent) - 1)
    m.Minimize(objective)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    if deterministic_time is not None:
        solver.parameters.max_deterministic_time = deterministic_time
    solver.parameters.num_workers = 1
    solves = []

    def run(phase):
        st = solver.Solve(m)
        wall = solver.WallTime()
        solves.append({
            "phase": phase, "status": solver.StatusName(st), "wall_time": round(wall, 3),
            "deterministic_time": round(solver.deterministic_time, 4),
            # Ended on the wall clock rather than on optimality/infeasibility or the
            # deterministic budget: the result then depends on machine load.
            "stopped_on_wall_clock": st not in (cp_model.OPTIMAL, cp_model.INFEASIBLE) and wall >= 0.98 * time_limit,
        })
        return st

    status = run("main")
    if status == cp_model.INFEASIBLE and (nogoods or incumbent is not None or floor_incumbent is not None):
        return {"status": "infeasible", "solves": solves}
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"structure model: {solver.StatusName(status)}")
    best_obj = solver.Value(objective)
    best_t = solver.Value(t)
    proven = status == cp_model.OPTIMAL
    # Snapshots of the solved values: rows are rebuilt only from these, so a tie-break that ends
    # UNKNOWN (no incumbent) cannot leak into the plan.
    chosen = {key: solver.Value(var) for key, var in n.items()}
    pattern_values = {p: solver.Value(var) for p, var in pattern_count.items()}
    tie_break_status = "not run"
    if windows and tie_break:
        # Among plans with the best M, the lowest floor T, then the usual tie-breaks.
        m.Add(objective == best_obj)
        m.Minimize(t)
        for key, val in chosen.items():
            m.AddHint(n[key], val)
        status_t = run("windows_floor")
        if status_t in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            best_t = solver.Value(t)
            chosen = {key: solver.Value(var) for key, var in n.items()}
            pattern_values = {p: solver.Value(var) for p, var in pattern_count.items()}
        proven = proven and status_t == cp_model.OPTIMAL
    if tie_break:
        m.Add(t == best_t)
        mixes = sum(u.values()) - len(ctx.choice_rounds())
        late = sum(r * n[r, e] for (r, e) in n if e != "gather") if lateness else 0
        work = VLEN * vector + cnt["alu"] + ca
        work_max = VLEN * big * 4
        late_max = ctx.rounds * ctx.rounds * g + 1
        # Lexicographic: mixes, then lateness, then work in scalar-op units (a vector op is 8).
        m.Minimize(mixes * (late_max * work_max) + late * work_max + work)
        m.ClearHints()  # the windows' T solve may have hinted these already
        for var, val in zip(list(n.values()), [solver.Value(v) for v in n.values()]):
            m.AddHint(var, val)
        status2 = run("tie_break")
        tie_break_status = solver.StatusName(status2)
        if status2 in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            chosen = {key: solver.Value(var) for key, var in n.items()}
            pattern_values = {p: solver.Value(var) for p, var in pattern_count.items()}
        elif status2 != cp_model.UNKNOWN:  # MODEL_INVALID / INFEASIBLE would be a bug
            raise RuntimeError(f"structure model tie-break: {tie_break_status}")
        # UNKNOWN (time limit without an incumbent): keep the first solve's plan.

    rows = rows_from_counts(ctx, chosen, pattern_values, fixed)
    counts = table.predict(rows)
    floor = choice_floor(counts)
    assert floor["cycles"] <= best_t, (floor["cycles"], best_t)
    if proven and not caps and (tie_break or not windows):
        assert floor["cycles"] == best_t, (floor["cycles"], best_t)
    return {"rows": rows, "floor": floor, "capped_cycles": best_t, "counts": counts,
            "objective": best_obj, "status": "optimal" if proven else "feasible", "tie_break": tie_break_status,
            "solves": solves}


def _solve_plan_child(table_json, kwargs):
    return solve_plan(CostTable.from_json(table_json), **kwargs)


def solve_plan_isolated(table: CostTable, **kwargs) -> dict:
    """`solve_plan` in a spawned child process.

    The pip OR-Tools CP-SAT used here and the OR-Tools that scheduler_rs links cannot be loaded
    into one process (it crashes in either import order), so a command that both solves and
    builds kernels solves in a child that never imports scheduler_rs.
    """
    import multiprocessing

    with multiprocessing.get_context("spawn").Pool(1) as pool:
        return pool.apply(_solve_plan_child, (table.to_json(), kwargs))


# ----------------------------------------------------------------------------
# What-ifs: hand-adjusted tables for encodings the emitter does not have yet
# ----------------------------------------------------------------------------
def what_if_gather_without_address_op(table: CostTable) -> CostTable:
    """The table as if every gathering group needed no vector address op (one shiftable vector
    op fewer per group and gather round), as `vload` at the 1-based index would give
    (docs/BOUND_GAP_ANALYSIS.md section 4.4). An estimate, not calibrated: it assumes the node
    reaches the XOR at no other cost, which the real encoding has to show.
    """
    ctx = table.context
    coef = {k: dict(v) for k, v in table.coefficients.items()}
    gather_rounds = sum(1 for r in range(ctx.rounds) if ctx.depth(r) >= 1)
    coef["base"]["vec_shiftable"] -= gather_rounds * ctx.n_groups
    for name in coef:
        if name.startswith("groups:"):  # these groups no longer gather: give the op back
            coef[name]["vec_shiftable"] += 1
    return CostTable(ctx, coef, table.calibration_plans, table.checked_plans,
                     table.notes + ["what-if: gather without the vector address op (not calibrated)"])


def what_if_packets(table: CostTable, pairs: Sequence[int] = (5, 7)) -> CostTable:
    """The table as if levels L and L+1 were fetched with one per-lane `vload` of a 3-word
    packet [node, left child, right - left] (pre-XORed with the hash constant, written in place
    during setup), for every L in `pairs` (each level visited once, so the gather rounds of L
    and L+1 are replaced). An estimate from docs/EMISSION_STRUCTURE_MODEL.md section 6, not
    calibrated. Per group and pair: 8 loads instead of 16; 1 multiply_add for the packet address
    instead of 2 address ops and 2 constant XORs; 8 scalar copies (level L) and 16 scalar ops
    (level L+1: left + bit * diff). Setup per pair: 3 * 2^L / 8 vloads and vector XORs, 2^L
    scalar subtracts, 3 * 2^L scalar stores and their address adds.
    """
    ctx = table.context
    coef = {k: dict(v) for k, v in table.coefficients.items()}
    b = coef["base"]
    for level in pairs:
        if level <= MAX_CACHED_DEPTH:
            raise ValueError(f"packet pair {level}+{level + 1} must be deeper than the cached levels")
        if level + 1 > ctx.forest_height:
            raise ValueError(f"packet pair {level}+{level + 1}: level {level + 1} does not exist "
                             f"(deepest level {ctx.forest_height})")
        visits = [sum(1 for r in range(ctx.rounds) if ctx.depth(r) == d) for d in (level, level + 1)]
        if visits != [1, 1]:
            raise ValueError(f"levels {level} and {level + 1} must each be visited exactly once (visits {visits}): "
                             f"the estimate replaces one gather pair per group")
        g = ctx.n_groups
        b["load"] += -8 * g + 3 * 2 ** level // 8
        b["vec_shiftable"] += -4 * g + 3 * 2 ** level // 8
        b["vec_fixed"] += g
        b["alu"] += 24 * g + 2 ** level + 3 * 2 ** level
        b["store"] += 3 * 2 ** level
    return CostTable(ctx, coef, table.calibration_plans, table.checked_plans,
                     table.notes + [f"what-if: packets for level pairs {list(pairs)} (not calibrated)"])


WHAT_IFS = {
    "gather-without-address-op": what_if_gather_without_address_op,
    "packets-5-7": lambda t: what_if_packets(t, (5, 7)),
    "packets-5-7-9": lambda t: what_if_packets(t, (5, 7, 9)),
    "packets-5": lambda t: what_if_packets(t, (5,)),
}


# ----------------------------------------------------------------------------
# Named plans for today's flag configurations
# ----------------------------------------------------------------------------
def flag_plan(ctx: Context, mode: str, depth4: str = "none") -> list[list[str]]:
    """The plan a flag configuration emits: `--cached-select mode` on levels 0 to 3, and level 4
    gathered ("none"), selected after the wrap ("after-wrap") or on every visit ("all")."""
    def rule(r, d):
        if d <= 3:
            return mode
        if d == 4 and (depth4 == "all" or (depth4 == "after-wrap" and r > ctx.forest_height)):
            return mode
        return "gather"
    return uniform_rows(ctx, rule)


NAMED_PLANS = {
    "bits": ("bits", "none"),
    "bits-flow": ("bits-flow", "none"),
    "mobius": ("mobius", "none"),
    "bits+depth4-after-wrap": ("bits", "after-wrap"),
    "bits-flow+depth4-after-wrap": ("bits-flow", "after-wrap"),
    "mobius+depth4-after-wrap": ("mobius", "after-wrap"),
    "bits+depth4": ("bits", "all"),
    "bits-flow+depth4": ("bits-flow", "all"),
    "mobius+depth4": ("mobius", "all"),
}


# ----------------------------------------------------------------------------
# Evaluation: emit, schedule, prove, validate
# ----------------------------------------------------------------------------
def measure(rows, table: CostTable, workdir: str, name: str, shaped: bool, upstream: bool,
             cpsat_deterministic_time: float | None = None, keep_kernel: str | None = None) -> dict:
    """Build a plan, prove and validate it, and compare predicted with emitted numbers."""
    from generate_ssa_kernel import save_kernel_to_json, simulator_cycles, validate_kernel
    from scripts import schedule_report as sr
    from verification.kernel_dataflow import check

    ctx = table.context
    extra = dict(SHAPED_GREEDY) if shaped else {}
    if cpsat_deterministic_time is not None:
        extra.update(use_greedy=False, cpsat_deterministic_time=cpsat_deterministic_time)
    instrs, stats, sched, path = build_emission(rows, ctx, workdir, name=name, extra=extra)
    instrs = json.loads(json.dumps(list(instrs)))
    emitted = emission_counts(sched)
    predicted = table.predict(rows)
    rep = sr.analyze(sched)
    proved = check({"instrs": instrs}, sched).ok
    passed = (not stats["budget_exceeded"]) and validate_kernel(
        instrs, ctx.forest_height, ctx.batch_size, ctx.rounds, 123, values_only=ctx.values_only)
    row = {
        "name": name,
        "plan": describe_plan(rows),
        "predicted_floor": choice_floor(predicted)["cycles"],
        "emitted_floor": choice_floor(emitted)["cycles"],
        "counts_match": emitted == predicted,
        "scheduled_floor": rep["floor"]["cycles"],
        "scheduled_floor_binds": rep["floor"]["binding"],
        "cycles": stats["cycles"],
        "simulator_cycles": simulator_cycles(instrs)[0],
        "peak_live": stats.get("peak_live"),
        "proved": proved,
        "passed": passed,
        "upstream": None,
        "scheduler": "greedy (shaped)" if shaped and cpsat_deterministic_time is None else (
            "greedy" if cpsat_deterministic_time is None else f"CP-SAT dtime {cpsat_deterministic_time:g}"),
    }
    # Always saved (in the working directory unless keep_kernel names a file), so --keep keeps it.
    kernel_path = os.path.abspath(keep_kernel or os.path.join(workdir, f"kernel_{name}.json"))
    save_kernel_to_json(instrs, kernel_path, ctx.forest_height, ctx.batch_size, ctx.rounds, 123,
                        values_only=ctx.values_only)
    row["kernel"] = kernel_path
    row["sched_json"] = os.path.abspath(path)
    row["upstream_strict"] = None
    if upstream:
        import subprocess

        def upstream_ok(*flags):
            out = subprocess.run([os.path.join(ROOT, "scripts", "check_upstream.sh"), *flags, "--kernel", kernel_path],
                                 cwd=ROOT, capture_output=True, text=True)
            return out.returncode == 0

        # Correctness on the unmodified upstream simulator and tests; --strict adds the upstream
        # speed thresholds, which a slow kernel fails by design.
        row["upstream"] = upstream_ok()
        row["upstream_strict"] = upstream_ok("--strict")
    return row


def scatter_svg(series: Sequence[tuple[str, Sequence[Mapping]]], width: int = 640, height: int = 420) -> str:
    """SVG scatter of simulator cycles against the emitted (choice) floor, one colour per series,
    with the line cycles = floor for reference. No plotting dependency."""
    points = [(label, r["emitted_floor"], r["simulator_cycles"]) for label, rows in series for r in rows]
    xs = [p[1] for p in points]
    ys = [p[2] for p in points]
    lo = min(xs + ys) - 10
    x_hi, y_hi = max(xs) + 10, max(ys) + 10
    left, right, top, bottom = 64, 16, 16, 48
    sx = lambda x: left + (x - lo) * (width - left - right) / (x_hi - lo)  # noqa: E731
    sy = lambda y: height - bottom - (y - lo) * (height - top - bottom) / (y_hi - lo)  # noqa: E731
    colours = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           f'font-family="sans-serif" font-size="12">',
           f'<rect width="{width}" height="{height}" fill="white"/>']
    for v in range(int(lo // 50 + 1) * 50, int(max(x_hi, y_hi)) + 1, 50):
        if v <= x_hi:
            out.append(f'<line x1="{sx(v):.1f}" y1="{sy(lo):.1f}" x2="{sx(v):.1f}" y2="{top}" stroke="#eee"/>')
            out.append(f'<text x="{sx(v):.1f}" y="{height - bottom + 16}" text-anchor="middle">{v}</text>')
        if v <= y_hi:
            out.append(f'<line x1="{left}" y1="{sy(v):.1f}" x2="{width - right}" y2="{sy(v):.1f}" stroke="#eee"/>')
            out.append(f'<text x="{left - 6}" y="{sy(v) + 4:.1f}" text-anchor="end">{v}</text>')
    diag = min(x_hi, y_hi)
    out.append(f'<line x1="{sx(lo):.1f}" y1="{sy(lo):.1f}" x2="{sx(diag):.1f}" y2="{sy(diag):.1f}" '
               f'stroke="#999" stroke-dasharray="4 3"/>')
    out.append(f'<text x="{(left + width - right) / 2}" y="{height - 10}" text-anchor="middle">'
               f'Emitted floor (cycles, choices resolved best)</text>')
    out.append(f'<text transform="translate(16 {(top + height - bottom) / 2}) rotate(-90)" '
               f'text-anchor="middle">Simulator cycles</text>')
    for i, (label, rows) in enumerate(series):
        colour = colours[i % len(colours)]
        for r in rows:
            out.append(f'<circle cx="{sx(r["emitted_floor"]):.1f}" cy="{sy(r["simulator_cycles"]):.1f}" r="5" '
                       f'fill="{colour}" fill-opacity="0.8"><title>{xml_escape(str(r["name"]))}: floor '
                       f'{r["emitted_floor"]}, {r["simulator_cycles"]} cycles</title></circle>')
        y = top + 14 + 18 * i
        out.append(f'<circle cx="{left + 14}" cy="{y - 4}" r="5" fill="{colour}"/>')
        out.append(f'<text x="{left + 24}" y="{y}">{xml_escape(label)}</text>')
    out.append(f'<text x="{sx(diag) - 4:.1f}" y="{sy(diag) + 14:.1f}" text-anchor="end" fill="#777">'
               f'cycles = floor</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def rows_ok(rows: Sequence[Mapping]) -> bool:
    """Every measured row predicted exactly, PROVED, PASSED and (when run) correct upstream.
    The upstream speed thresholds (--strict) are reported, not required."""
    return all(r["counts_match"] and r["proved"] and r["passed"] and r["upstream"] is not False for r in rows)


def format_rows(rows: list[dict]) -> str:
    header = ["Structure", "Predicted floor", "Emitted floor", "Counts match", "Scheduled floor",
              "Cycles (sched / sim)", "Peak live", "PROVED", "PASSED", "Upstream correct", "Upstream --strict"]
    yes = lambda v: "-" if v is None else ("yes" if v else "NO")  # noqa: E731
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        lines.append("| " + " | ".join([
            r["name"], f"{r['predicted_floor']:,}", f"{r['emitted_floor']:,}", yes(r["counts_match"]),
            f"{r['scheduled_floor']:,} ({'+'.join(r['scheduled_floor_binds'])})",
            f"{r['cycles']:,} / {r['simulator_cycles']:,}", str(r["peak_live"]), yes(r["proved"]),
            yes(r["passed"]), yes(r["upstream"]), yes(r.get("upstream_strict"))]) + " |")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def _context_from_args(args) -> Context:
    return Context(rounds=args.rounds, batch_size=args.batch_size, forest_height=args.forest_height,
                   merge_hash_stages=not args.no_merge_hash_stages, index_form=args.index_form,
                   select_bit_source=args.select_bit_source, values_only=args.values_only)


def parse_max_util(text: str) -> dict[str, float]:
    """'flow=0.8,load=0.95' -> {"flow": 0.8, "load": 0.95}; 'none' or '' -> {}."""
    if text in ("", "none"):
        return {}
    return {k: float(v) for k, v in (item.split("=") for item in text.split(","))}


def _print_counts(label, counts):
    floor = choice_floor(counts)
    res = floor["resolution"]
    print(f"{label}: floor {floor['cycles']:,} (binds: {', '.join(floor['binding']) or 'ALUs'}), "
          f"{floor['vector_op_equivalents']:,.2f} vector-op equivalents, loads "
          f"{counts['load'] + counts['const'] + counts['const_alu'] - res['const_on_flow'] - res['const_alu_on_alu']:,}, "
          f"flow {counts['flow'] + res['select1_flow'] + res['select2_flow'] + res['const_on_flow']:,}")
    print("  counts: " + ", ".join(f"{k}={v:,}" for k, v in counts.items()))
    print("  resolution: " + ", ".join(f"{k}={v:,}" for k, v in res.items()))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["loop"]:  # the stage 1 / stage 2 feedback loop lives in its own module
        from scripts import emission_loop
        return emission_loop.main(argv[1:])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["calibrate", "check", "predict", "solve", "sweep", "measure", "table",
                                        "scatter", "derive", "loop"],
                    help="loop: closed loop with the shaped greedy (scripts/emission_loop.py; loop --help)")
    ap.add_argument("--rounds", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--forest-height", type=int, default=10)
    ap.add_argument("--no-merge-hash-stages", action="store_true", help="context: 11-op hash")
    ap.add_argument("--index-form", choices=["zero-based", "mirrored"], default="mirrored")
    ap.add_argument("--select-bit-source", choices=["index", "path"], default="index")
    ap.add_argument("--values-only", action="store_true", help="context: CHANGES THE OUTPUT (indices not stored)")
    ap.add_argument("--table", help="cost table JSON (default: artifacts/emission_structure/costs_<context>.json, "
                                    "calibrated on first use)")
    ap.add_argument("--plans", type=int, default=20, help="check: random plans to build")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--plan", action="append", default=[],
                    help="plan file or named flag plan (" + ", ".join(NAMED_PLANS) + "); repeatable for measure")
    ap.add_argument("--encodings", default=",".join(ENCODINGS), help="solve: encodings allowed")
    ap.add_argument("--time-limit", type=float, default=60.0, help="solve: CP-SAT seconds per solve")
    ap.add_argument("--max-util", action="append", default=[], metavar="ENGINE=FRAC[,ENGINE=FRAC]",
                    help="solve: cap the share of an engine's capacity the plan may use (load, flow, valu), "
                         "e.g. flow=0.8; minimises T under the caps. sweep: repeat for each setting to try "
                         "('none' = no cap)")
    ap.add_argument("--out", help="solve: write the plan here")
    ap.add_argument("--shaped", action="store_true", help="measure: shaped greedy flags (" +
                    " ".join(f"{k}={v}" for k, v in SHAPED_GREEDY.items()) + ")")
    ap.add_argument("--cpsat-deterministic-time", type=float, default=None, help="measure: CP-SAT instead of greedy")
    ap.add_argument("--upstream", action="store_true",
                    help="measure/sweep: also ./scripts/check_upstream.sh, without and with --strict")
    ap.add_argument("--keep", help="directory for scheduler JSONs and kernels (default: temporary)")
    ap.add_argument("--json", help="measure/table: also write rows here")
    ap.add_argument("--fix", action="append", default=[], metavar="ROUND=ENCODING",
                    help="solve/sweep: keep round ROUND uniform with ENCODING (e.g. 15=gather keeps select work "
                         "out of the last round, the schedule's tail); repeatable")
    ap.add_argument("--no-lateness", action="store_true",
                    help="solve/sweep: drop the tie-break that puts select work at a level's earliest visit")
    ap.add_argument("--input", action="append", default=[], metavar="LABEL=ROWS.json",
                    help="scatter: rows written by measure/sweep --json, under a legend label; repeatable")
    ap.add_argument("--what-if", choices=sorted(WHAT_IFS),
                    help="predict/solve: use a hand-adjusted table for an encoding the emitter does not have "
                         "yet (an estimate, never compared with an emission)")
    args = ap.parse_args(argv)
    ctx = _context_from_args(args)
    fixed = {}
    for item in args.fix:
        rnd, sep, enc = item.partition("=")
        if not sep or not rnd.isdigit() or enc not in ENCODINGS:
            ap.error(f"--fix needs ROUND=ENCODING with ENCODING in {ENCODINGS}, got {item!r}")
        fixed[int(rnd)] = enc
    if args.command == "scatter":
        series = []
        for item in args.input:
            label, sep, path = item.partition("=")
            if not sep:
                ap.error(f"--input needs LABEL=ROWS.json, got {item!r}")
            with open(path) as f:
                series.append((label, json.load(f)))
        svg = scatter_svg(series)
        if args.out:
            with open(args.out, "w") as f:
                f.write(svg)
            print(f"scatter written to {args.out}")
        else:
            print(svg)
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        workdir = args.keep or tmp
        os.makedirs(workdir, exist_ok=True)
        if args.command == "calibrate":
            path = args.table or default_table_path(ctx)
            table = calibrate(ctx, workdir)
            write_json_atomic(path, table.to_json())
            print(f"calibrated from {table.calibration_plans} plans (exact); written to {path}")
            for note in table.notes:
                print("  note: " + note)
            return 0
        if args.command == "derive":
            # The derivation needs no probe builds: run and report it first, so a cold cache or
            # a calibration failure cannot hide it. Its report must not overwrite the
            # reference it is compared with (a later run would then compare it with itself).
            reference = args.table or default_table_path(ctx)
            if args.json and os.path.realpath(args.json) == os.path.realpath(reference):
                ap.error(f"--json {args.json} is the reference cost table ({reference}); "
                         "write the derivation elsewhere")
            derived, doc = derive_table(ctx)
            if args.json:
                write_json_atomic(args.json, doc)
            print(f"Derived by instantiation (scheduler_rs): {len(doc['sites'])} sites, "
                  f"{len(doc['resources'])} shared-resource groups")
            print("| Feature | " + " | ".join(CATEGORIES) + " |")
            print("|---|" + "---|" * len(CATEGORIES))
            for name, coef in derived.coefficients.items():
                if any(coef.values()):
                    print(f"| {name} | " + " | ".join(f"{coef[c]:,}" if coef[c] else "" for c in CATEGORIES) + " |")
            try:
                table = load_or_calibrate(ctx, args.table, workdir)
            except Exception as e:  # noqa: BLE001 - reported; the derivation above stands
                print(f"no calibrated table to compare with: {type(e).__name__}: {e}")
                return 1
            diffs = table_differences(derived, table)
            entries = len(derived.coefficients) * len(CATEGORIES)
            if diffs:
                print(f"DIFFERS from the calibrated table in {len(diffs)} of {entries} entries (derived vs calibrated):")
                for d in diffs:
                    print("  " + d)
                return 1
            print(f"Equal to the calibrated table in all {entries} entries "
                  f"({len(derived.coefficients)} features x {len(CATEGORIES)} categories).")
            return 0
        table = load_or_calibrate(ctx, args.table, workdir)
        if args.what_if:
            if args.command not in ("predict", "solve"):
                ap.error("--what-if applies to predict and solve only: there is no emission to measure")
            table = WHAT_IFS[args.what_if](table)
            print(f"WHAT-IF {args.what_if}: hand-adjusted cost table, not calibrated against the emitter")

        def resolve(spec):
            if spec in NAMED_PLANS:
                return spec, flag_plan(ctx, *NAMED_PLANS[spec])
            return os.path.basename(spec), expand_plan(load_plan(spec), ctx.rounds, ctx.n_groups, ctx.forest_height)

        if args.command == "check":
            result = check_table(table, workdir, args.plans, args.seed)
            bad = result["bad"]
            print(f"{result['compared']} random plans compared: {result['compared'] - len(bad)} predicted exactly, "
                  f"{len(bad)} not; {len(result['skipped'])} did not build")
            for b in bad[:3]:
                print(json.dumps(b))
            for e in result["skipped"][:3]:
                print("  did not build: " + e)
            return 1 if bad or not result["compared"] else 0
        if args.command == "table":
            print("| Feature | " + " | ".join(CATEGORIES) + " |")
            print("|---|" + "---|" * len(CATEGORIES))
            for name, coef in table.coefficients.items():
                if any(coef.values()):
                    print(f"| {name} | " + " | ".join(f"{coef[c]:,}" if coef[c] else "" for c in CATEGORIES) + " |")
            return 0
        if args.command == "predict":
            for spec in args.plan or list(NAMED_PLANS):
                name, rows = resolve(spec)
                _print_counts(f"{name} [{describe_plan(rows)}]", table.predict(rows))
            return 0
        if args.command == "solve":
            max_util = parse_max_util(args.max_util[0] if args.max_util else "none")
            # Isolated: on a cold cache the calibration above has loaded scheduler_rs here.
            result = solve_plan_isolated(table, encodings=args.encodings.split(","), time_limit=args.time_limit,
                                         max_util=max_util, lateness=not args.no_lateness, fixed=fixed)
            capped = f", T under caps {result['capped_cycles']:,}" if max_util else ""
            _print_counts(f"best plan ({result['status']}{capped}) [{describe_plan(result['rows'])}]", result["counts"])
            if args.out:
                doc = plan_document(result["rows"], ctx.forest_height, context=asdict(ctx),
                                    predicted={"floor": result["floor"]["cycles"], "counts": result["counts"],
                                               "max_util": max_util, "capped_cycles": result["capped_cycles"]},
                                    comment="written by scripts/emission_structure.py solve")
                with open(args.out, "w") as f:
                    json.dump(doc, f, indent=1)
                print(f"plan written to {args.out}")
            return 0
        if args.command == "sweep":
            rows_out = []
            for text in args.max_util or ["none"]:
                max_util = parse_max_util(text)
                result = solve_plan_isolated(table, encodings=args.encodings.split(","),
                                             time_limit=args.time_limit, max_util=max_util,
                                             lateness=not args.no_lateness, fixed=fixed)
                row = measure(result["rows"], table, workdir, f"caps {text}", args.shaped, args.upstream,
                              args.cpsat_deterministic_time)
                row["capped_cycles"] = result["capped_cycles"]
                rows_out.append(row)
                print(f"  done: caps {text}", file=sys.stderr, flush=True)
            print(format_rows(rows_out))
            if args.json:
                with open(args.json, "w") as f:
                    json.dump(rows_out, f, indent=1)
            return 0 if rows_ok(rows_out) else 1
        if args.command == "measure":
            rows_out = []
            for spec in args.plan or ["bits"]:
                name, rows = resolve(spec)
                rows_out.append(measure(rows, table, workdir, name, args.shaped, args.upstream,
                                         args.cpsat_deterministic_time))
                print(f"  done: {name}", file=sys.stderr, flush=True)
            print(format_rows(rows_out))
            if args.json:
                with open(args.json, "w") as f:
                    json.dump(rows_out, f, indent=1)
            return 0 if rows_ok(rows_out) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
