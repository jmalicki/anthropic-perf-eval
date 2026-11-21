"""Lane models of shiftable ops (--cpsat-lane-model): the model rows that mention lanes.

A shiftable vector op has eight lane variables ``alu_c{op}_{lane}`` (its scalar ops' cycles
when shifted). ``sorted`` orders them, ``pinned`` sets an unshifted op's lanes to its cycle;
``verification/lane_model.py`` proves, per row shape, that neither loses a schedule. These
tests tie the proof to the real model: every row of an exported model that mentions a lane
must have one of the proved shapes, and the rows the pinning proof relies on must exist.

Shapes (see the Z3 module for why each is safe):

* uniform: one lane variable, and the same row exists for all eight lanes of that op;
* lane-wise: lane i of a reader against lane i of a producer (same index), for all eight;
* the interval binding ``x - x == 0`` and the size-1 lane intervals (one presence literal
  for all eight lanes), in a cumulative with the same demand for all eight lanes;
* lane order ``lane_i <= lane_i+1`` (sorted models only).

A row that mentions lanes in any other way (for example lane i of a reader waiting for its
own scalar producer, which per-op sorting would break) fails the test.
"""

import collections
import os
import re
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts import diff_cpsat_models  # noqa: E402

BASE = ["--batch-size", "32", "--cpsat-deterministic-time", "0.2", "--cpsat-time-limit", "100"]
LANES = 8
LANE_VAR = re.compile(r"alu_c(\d+)_(\d+)$")
CYCLE_VAR = re.compile(r"c(\d+)$")


def _export(tmp_path, tag, extra):
    model_path = tmp_path / f"model_{tag}.pb"
    env = dict(os.environ, CPSAT_EXPORT_MODEL=str(model_path))
    env.pop("CPSAT_MAX_DETERMINISTIC_TIME", None)
    cmd = [sys.executable, os.path.join(ROOT, "generate_ssa_kernel.py"), "-q", "--no-validate",
           "--output-json", str(tmp_path / f"sched_{tag}.json"), "--save-kernel", str(tmp_path / f"kernel_{tag}.json"),
           *BASE, *extra]
    result = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=170)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    return diff_cpsat_models.load(str(model_path))


def _vars_in(message) -> set:
    """Every variable index a proto message refers to (fields named vars/literals, recursively)."""
    found = set()
    for field, value in message.ListFields():
        if field.type == field.TYPE_MESSAGE:
            items = value if field.label == field.LABEL_REPEATED else [value]
            for item in items:
                found |= _vars_in(item)
        elif field.name in ("vars", "literals", "enforcement_literal", "variables", "target", "index"):
            items = value if field.label == field.LABEL_REPEATED else [value]
            found |= {v if v >= 0 else -v - 1 for v in items}
    return found


class LaneRows:
    """Classification of every constraint of one exported model that mentions a lane."""

    def __init__(self, model):
        self.model = model
        self.names = [v.name for v in model.variables]
        self.index = {name: i for i, name in enumerate(self.names) if name}
        self.lane = {}  # var index -> (op, lane)
        for i, name in enumerate(self.names):
            m = LANE_VAR.match(name or "")
            if m:
                self.lane[i] = (int(m.group(1)), int(m.group(2)))
        self.ops = sorted({op for op, _ in self.lane.values()})
        self.unverified = []
        self.uniform = collections.defaultdict(set)  # key -> lanes
        self.lanewise = collections.defaultdict(set)  # (reader, producer, domain, enforcement) -> lanes
        self.order = collections.defaultdict(set)  # op -> i of lane_i <= lane_i+1
        self.lane_intervals = {}  # interval constraint index -> (op, lane, presence)
        self.rows_by_var = collections.defaultdict(list)  # var -> (terms, domain, enforcement) of its rows
        self.by_kind = collections.Counter()
        for c in model.constraints:
            if c.WhichOneof("constraint") == "linear":
                key = self._row_key(c)
                for v, _ in key[0]:
                    self.rows_by_var[v].append(key)
        for ci, c in enumerate(model.constraints):
            self._classify(ci, c)
        for key, lanes in self.uniform.items():
            if lanes != set(range(LANES)):
                self.unverified.append(f"uniform row on op {key[0]} exists for lanes {sorted(lanes)} only")
        for key, lanes in self.lanewise.items():
            if lanes != set(range(LANES)):
                self.unverified.append(f"lane-wise row {key[:2]} exists for lanes {sorted(lanes)} only")
        by_op = collections.defaultdict(dict)
        for op, lane, presence in self.lane_intervals.values():
            by_op[op][lane] = presence
        for op, lanes in by_op.items():
            if set(lanes) != set(range(LANES)) or len(set(lanes.values())) != 1:
                self.unverified.append(f"lane intervals of op {op}: {lanes}")
        for name, field in [("objective", model.objective), ("floating objective", model.floating_point_objective)]:
            if _vars_in(field) & set(self.lane):
                self.unverified.append(f"the {name} mentions a lane")
        for strategy in model.search_strategy:
            if _vars_in(strategy) & set(self.lane):
                self.unverified.append("a search strategy mentions a lane")

    @staticmethod
    def _row_key(c):
        terms = collections.Counter()
        for v, k in zip(c.linear.vars, c.linear.coeffs):
            terms[v] += k
        return (tuple(sorted((v, k) for v, k in terms.items() if k)), tuple(c.linear.domain),
                tuple(sorted(c.enforcement_literal)))

    def _classify(self, ci, c):
        kind = c.WhichOneof("constraint")
        mentioned = _vars_in(c) & set(self.lane)
        if kind == "interval":
            starts = list(c.interval.start.vars)
            if not mentioned:
                return
            ok = (len(starts) == 1 and starts[0] in self.lane and list(c.interval.start.coeffs) == [1]
                  and not c.interval.size.vars and c.interval.size.offset == 1
                  and len(c.enforcement_literal) == 1)
            if ok:
                op, lane = self.lane[starts[0]]
                self.lane_intervals[ci] = (op, lane, c.enforcement_literal[0])
                self.by_kind["lane interval"] += 1
            else:
                self.unverified.append(f"interval #{ci} {c.name!r}")
            return
        if kind in ("cumulative", "no_overlap"):
            intervals = list(getattr(c, kind).intervals)
            lane_positions = [p for p, iv in enumerate(intervals) if iv in self.lane_intervals]
            if not lane_positions:
                return
            if kind == "no_overlap":
                self.unverified.append(f"no_overlap #{ci} {c.name!r} contains lane intervals")
                return
            demands = collections.defaultdict(set)
            for p in lane_positions:
                demand = c.cumulative.demands[p]
                op = self.lane_intervals[intervals[p]][0]
                demands[op].add((tuple(demand.vars), tuple(demand.coeffs), demand.offset))
            for op, kinds in demands.items():
                if len(kinds) != 1 or kinds != {((), (), 1)}:
                    self.unverified.append(f"cumulative {c.name!r}: lane demands of op {op} are {kinds}")
            self.by_kind["cumulative"] += 1
            return
        if not mentioned:
            return
        if kind != "linear":
            self.unverified.append(f"{kind} #{ci} {c.name!r}")
            return
        terms = list(zip(c.linear.vars, c.linear.coeffs))
        lane_terms = [(self.lane[v], k) for v, k in terms if v in self.lane]
        others = tuple(sorted((v, k) for v, k in terms if v not in self.lane))
        domain = tuple(c.linear.domain)
        enforcement = tuple(sorted(c.enforcement_literal))
        if len(terms) == 2 and terms[0][0] == terms[1][0] and terms[0][1] + terms[1][1] == 0:
            self.by_kind["interval binding"] += 1
        elif len(lane_terms) == 1:
            (op, lane), k = lane_terms[0]
            self.uniform[(op, k, others, domain, enforcement)].add(lane)
            self.by_kind["uniform"] += 1
        elif len(lane_terms) == 2 and not others:
            ((op_a, lane_a), k_a), ((op_b, lane_b), k_b) = lane_terms
            if {k_a, k_b} == {1, -1} and op_a != op_b and lane_a == lane_b:
                reader, producer = (op_a, op_b) if k_a == 1 else (op_b, op_a)
                self.lanewise[(reader, producer, domain, enforcement)].add(lane_a)
                self.by_kind["lane-wise"] += 1
            elif (op_a == op_b and {k_a, k_b} == {1, -1} and abs(lane_a - lane_b) == 1 and not enforcement
                  and domain[1] == 0 and (lane_a < lane_b) == (k_a == 1)):
                self.order[op_a].add(min(lane_a, lane_b))
                self.by_kind["lane order"] += 1
            else:
                self.unverified.append(f"two-lane row #{ci} {c.name!r}")
        else:
            self.unverified.append(f"linear #{ci} {c.name!r} with {len(lane_terms)} lanes")

    # --- premises of the pinning proof ------------------------------------------------

    def var(self, name):
        return self.index.get(name)

    def has_row(self, terms, domain_lo, enforcement=()):
        """Some row implies ``sum(terms) >= domain_lo`` whenever ``enforcement`` holds.

        It does if its enforcement literals are among ``enforcement``, its lower bound is at
        least ``domain_lo``, and it has the same terms plus possibly extra terms ``-k * y``
        with ``k > 0`` and ``y >= 0`` (such as the multiply-add extra cycle of a select).
        """
        want = dict(terms)
        for row_terms, row_domain, row_enf in self.rows_by_var.get(terms[0][0], ()):
            if not set(row_enf) <= set(enforcement):
                continue
            # lo <= row <= hi gives row >= lo and -row >= -hi.
            for sign, lo in ((1, row_domain[0]), (-1, -row_domain[-1])):
                if lo < domain_lo:
                    continue
                row = {v: sign * k for v, k in row_terms}
                if any(row.get(v) != k for v, k in want.items()):
                    continue
                extra = [(v, k) for v, k in row.items() if v not in want]
                if all(k < 0 and self.model.variables[v].domain[0] >= 0 for v, k in extra):
                    return True
        return False

    def pinning_gaps(self):
        """Uniform lane rows whose twin on the op's cycle is missing (see LANE-pin-dag)."""
        gaps = []
        for (op, k, others, domain, enforcement) in self.uniform:
            c_op, shift = self.var(f"c{op}"), self.var(f"shift_{op}")
            neg_shift = -shift - 1
            if shift in enforcement:
                continue  # inactive when the op is not shifted
            if enforcement and all(self.names[e].startswith("eliminate_vb_") for e in enforcement if e >= 0):
                # eliminate_vb => shift of every consumer (vb_elim_*_shift_ge_c*)
                if all(self.has_row([(shift, 1), (e, -1)], 0) for e in enforcement):
                    continue
            if [v for v, _ in others] == [c_op]:
                continue  # lane_pin itself
            if self.has_row([(c_op, k), *others], domain[0], enforcement):
                continue  # the same row on the cycle variable
            if len(others) == 1 and k == 1 and others[0][1] == -1 and domain[0] == 0:
                # start <= lane: scratch start equals the cycle when not shifted
                start = others[0][0]
                if self.has_row([(c_op, 1), (start, -1)], 0, [neg_shift]):
                    continue
            if len(others) == 1 and k == 1 and others[0][1] == -1:
                # lane >= c_dep + lo when dep is not shifted: c_op >= completion_dep + 1 >= c_dep + 1
                m = CYCLE_VAR.match(self.names[others[0][0]] or "")
                completion = self.var(f"completion_c{m.group(1)}") if m else None
                if completion is not None and self.has_row([(c_op, 1), (completion, -1)], domain[0]) and \
                        self.has_row([(completion, 1), (others[0][0], -1)], 0):
                    continue
            gaps.append((op, [(self.names[v], w) for v, w in others], domain, enforcement))
        for (reader, producer, domain, _) in self.lanewise:
            if not self.has_row([(self.var(f"c{reader}"), 1), (self.var(f"completion_c{producer}"), -1)], domain[0]):
                gaps.append(("lane-wise without consumer_after_completion", reader, producer))
        return gaps

    def domain_gaps(self):
        """Ops whose cycle can go below their lanes' lower bound (pinning would cut it)."""
        out = []
        for op in self.ops:
            c_dom = self.model.variables[self.var(f"c{op}")].domain
            lane_dom = self.model.variables[self.var(f"alu_c{op}_0")].domain
            if c_dom[0] < lane_dom[0]:
                out.append((op, list(c_dom), list(lane_dom)))
        return out


CONFIGS = {
    "3r": ["--rounds", "3"],
    # A gather round: lanes of a write group come from scalar loads.
    "5r": ["--rounds", "5"],
    # Forced multiply-add selects: the select_prec_*_fma rows on lanes.
    "3r-fma": ["--rounds", "3", "--select-choice", "fma"],
    # Eliminable broadcasts: the vb_elim_* rows on lanes.
    "3r-vb-elim": ["--rounds", "3", "--enable-eliminate-vbroadcast"],
}
CASES = [(cfg, lane_model) for cfg in ["3r", "5r"] for lane_model in ["free", "sorted", "pinned", "sorted-pinned"]]
CASES += [(cfg, lane_model) for cfg in ["3r-fma", "3r-vb-elim"] for lane_model in ["free", "sorted-pinned"]]


@pytest.mark.parametrize("cfg,lane_model", CASES, ids=[f"{c}-{m}" for c, m in CASES])
def test_every_lane_row_has_a_proved_shape(tmp_path, cfg, lane_model):
    model = _export(tmp_path, "m", [*CONFIGS[cfg], "--cpsat-lane-model", lane_model])
    rows = LaneRows(model)
    assert rows.ops, "no shiftable op: the test checks nothing"
    assert not rows.unverified, "\n".join(rows.unverified[:20])
    names = [c.name for c in model.constraints]
    n_order = sum(name.startswith("lane_order_") for name in names)
    n_pin = sum(name.startswith("lane_pin_") for name in names)
    n_on_off = sum(bool(re.match(r"shift_alu_prec_\d+_\d+_dep_\d+_(on|off)$", name)) for name in names)
    n_single = sum(bool(re.match(r"shift_alu_prec_\d+_\d+_dep_\d+$", name)) for name in names)
    sorted_, pinned = "sorted" in lane_model, "pinned" in lane_model
    assert n_order == (LANES - 1) * len(rows.ops) * sorted_
    assert sum(len(v) for v in rows.order.values()) == n_order
    assert n_pin == LANES * len(rows.ops) * pinned
    if pinned:
        assert n_on_off == 0 and n_single > 0
        for c in model.constraints:
            if c.name.startswith("lane_pin_"):
                op = int(c.name.split("_")[2])
                assert list(c.enforcement_literal) == [-rows.var(f"shift_{op}") - 1], c.name
    else:
        assert n_single == 0 and n_on_off > 0
    if not pinned:
        # Premises of LANE-pin-dag, read off the free model.
        assert not rows.pinning_gaps(), rows.pinning_gaps()[:10]
        assert not rows.domain_gaps(), rows.domain_gaps()[:10]


def test_free_is_the_default_model(tmp_path):
    default = _export(tmp_path, "default", ["--rounds", "3"])
    explicit = _export(tmp_path, "free", ["--rounds", "3", "--cpsat-lane-model", "free"])
    assert not diff_cpsat_models.compare(default, explicit)


def test_lane_specific_producer_would_be_rejected():
    """The classifier itself: a row tying lane 0 of an op to one producer only is unverified."""
    from ortools.sat import cp_model_pb2

    model = cp_model_pb2.CpModelProto()
    for name in [*(f"alu_c7_{i}" for i in range(LANES)), "c3"]:
        var = model.variables.add(name=name)
        var.domain.extend([0, 10])
    row = model.constraints.add(name="lane_specific")
    row.linear.vars.extend([0, LANES])
    row.linear.coeffs.extend([1, -1])
    row.linear.domain.extend([1, 2**62])
    rows = LaneRows(model)
    assert any("lanes [0] only" in message for message in rows.unverified), rows.unverified
