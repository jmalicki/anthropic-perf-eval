"""Emission structure model (scripts/emission_structure.py, docs/EMISSION_STRUCTURE_MODEL.md).

- choice_floor equals a brute-force search over every choice resolution on small counts.
- The cost table calibrated from the emitter predicts the emitted op counts exactly, on its
  design plans and on random plans it was not fitted to (16 rounds, 4 groups).
- The CP-SAT plan's floor equals choice_floor of its predicted counts, and building that plan
  emits exactly those counts.
"""

import itertools
import json
import math
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import emission_structure as es  # noqa: E402

CTX = es.Context(rounds=16, batch_size=32)


def brute_force_floor(c):
    """Smallest T for which some resolution of every choice fits (exhaustive)."""
    for t in itertools.count(1):
        for s1 in range(c["select1"] + 1):
            for s2 in range(c["select2"] + 1):
                for cf in range(c["const"] + 1):
                    for af, aa in itertools.product(range(c["const_alu"] + 1), repeat=2):
                        if af + aa > c["const_alu"]:
                            continue
                        loads = c["load"] + (c["const"] - cf) + (c["const_alu"] - af - aa)
                        flow = c["flow"] + s1 + s2 + cf + af
                        vector = c["vec_fixed"] + c["vec_shiftable"] + (c["select1"] - s1) + 2 * (c["select2"] - s2)
                        shiftable = c["vec_shiftable"] + c["select2"] - s2
                        if loads > 2 * t or c["store"] > 2 * t or flow > t:
                            continue
                        if any(vector - sh <= 6 * t and c["alu"] + aa + 8 * sh <= 12 * t
                               for sh in range(shiftable + 1)):
                            return t


def test_choice_floor_matches_brute_force():
    rng = random.Random(5)
    for _ in range(300):
        c = {"vec_fixed": rng.randint(0, 30), "vec_shiftable": rng.randint(0, 30), "alu": rng.randint(0, 40),
             "load": rng.randint(0, 12), "store": rng.randint(0, 6), "flow": rng.randint(0, 4),
             "select1": rng.randint(0, 5), "select2": rng.randint(0, 5), "const": rng.randint(0, 3),
             "const_alu": rng.randint(0, 3)}
        assert es.choice_floor(c)["cycles"] == brute_force_floor(c), c


def test_choice_floor_reports_binding_engine():
    loads_only = dict.fromkeys(es.CATEGORIES, 0) | {"load": 100}
    assert es.choice_floor(loads_only) == {**es.choice_floor(loads_only), "cycles": 50, "binding": ["load"]}
    vector_only = dict.fromkeys(es.CATEGORIES, 0) | {"vec_shiftable": 75}
    assert es.choice_floor(vector_only)["cycles"] == 10  # 60 on the VALU, 15 x 8 on the ALU
    assert es.choice_floor(vector_only)["binding"] == ["valu+alu"]
    # A 2-op select is worth moving to flow first: 1 flow slot, 12 vector ops otherwise.
    selects = dict.fromkeys(es.CATEGORIES, 0) | {"select2": 6}
    assert es.choice_floor(selects)["cycles"] == 2


def test_emission_counts_needs_choice_costs():
    with pytest.raises(ValueError, match="select_fma_vector_ops"):
        es.emission_counts({"ssa_ops": [{"opcode": "select_choice", "engine": "flow"}]})


@pytest.fixture(scope="module")
def table(tmp_path_factory):
    return es.calibrate(CTX, str(tmp_path_factory.mktemp("cal")), n_random=20, seed=3, log=None)


def test_calibration_is_exact_on_unseen_plans(table, tmp_path):
    assert table.calibration_plans > len(es.feature_names(CTX))
    assert es.check_table(table, str(tmp_path), n_plans=12, seed=11, log=None) == []


def test_counts_cover_every_ssa_op(tmp_path):
    rows = es.flag_plan(CTX, "bits-flow", "after-wrap")
    _, _, sched, _ = es.build_emission(rows, CTX, str(tmp_path))
    counts = es.emission_counts(sched)
    assert sum(counts.values()) == len(sched["ssa_ops"])
    assert counts["select1"] > 0 and counts["select2"] > 0


def test_cost_table_round_trips(table):
    again = es.CostTable.from_json(json.loads(json.dumps(table.to_json())))
    rows = es.flag_plan(CTX, "bits")
    assert again.predict(rows) == table.predict(rows)


@pytest.mark.parametrize("max_util", [{}, {"flow": 0.8}])
def test_solved_plan_floor_is_emitted(table, tmp_path, max_util):
    # The pip OR-Tools solver cannot share a process with scheduler_rs: solve in a child.
    result = es.solve_plan_isolated(table, max_util=max_util, time_limit=30)
    assert result["floor"]["cycles"] == es.choice_floor(result["counts"])["cycles"]
    if not max_util:
        assert result["capped_cycles"] == result["floor"]["cycles"]
    # Never worse than any flag configuration the plan language can express.
    for mode, depth4 in es.NAMED_PLANS.values():
        flags = es.choice_floor(table.predict(es.flag_plan(CTX, mode, depth4)))["cycles"]
        assert result["capped_cycles"] <= flags or max_util
    _, _, sched, _ = es.build_emission(result["rows"], CTX, str(tmp_path))
    assert es.emission_counts(sched) == result["counts"]
    assert math.isclose(es.choice_floor(es.emission_counts(sched))["cycles"], result["floor"]["cycles"])
