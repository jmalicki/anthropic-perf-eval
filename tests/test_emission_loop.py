"""Closed loop between the structure model and the shaped greedy (scripts/emission_loop.py,
docs/EMISSION_STRUCTURE_MODEL.md section 10). 16 rounds, 4 groups.

- window_floor equals a brute force over every choice resolution and is monotone.
- capped_floor is choice_floor without caps and the solver's capped objective with them.
- No-good cuts and the incumbent bound exclude what they should.
- With window rows the solver's M is max(floor, the row table's bound) of the plan it returns.
- The row table learned from built plans never puts a plan's bound above its cycles, and a
  plan that shows a lower head than the table is reported.
- The loop end to end: every plan it builds is PROVED and PASSED, and the state is reusable.
"""

import itertools
import json
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import emission_loop as el  # noqa: E402
from scripts import emission_structure as es  # noqa: E402

CTX = es.Context(rounds=16, batch_size=32)


def brute_window_floor(c, cap=es.CAP, vlen=es.VLEN):
    for span in itertools.count(0):
        if c["load"] > cap["load"] * span or c["store"] > cap["store"] * span:
            continue
        for sf in range(c["select"] + 1):
            if c["flow"] + sf > cap["flow"] * span:
                continue
            vector = c["vec_fixed"] + c["vec_shiftable"] + c["select"] - sf
            if any(vector - sh <= cap["valu"] * span and c["alu"] + vlen * sh <= cap["alu"] * span
                   for sh in range(c["vec_shiftable"] + 1)):
                return span


def random_window_counts(rng):
    return {"vec_fixed": rng.randint(0, 20), "vec_shiftable": rng.randint(0, 20), "alu": rng.randint(0, 30),
            "load": rng.randint(0, 10), "store": rng.randint(0, 4), "flow": rng.randint(0, 4),
            "select": rng.randint(0, 8)}


def test_window_floor_matches_brute_force_and_is_monotone():
    rng = random.Random(3)
    for _ in range(300):
        c = random_window_counts(rng)
        assert es.window_floor(c) == brute_window_floor(c), c
        bigger = dict(c)
        bigger[rng.choice(es.WINDOW_CATEGORIES)] += 1
        assert es.window_floor(bigger) >= es.window_floor(c)
    assert es.window_floor(dict.fromkeys(es.WINDOW_CATEGORIES, 0)) == 0


def test_capped_floor_is_choice_floor_without_caps():
    rng = random.Random(4)
    for _ in range(200):
        c = {"vec_fixed": rng.randint(0, 30), "vec_shiftable": rng.randint(0, 30), "alu": rng.randint(0, 40),
             "load": rng.randint(0, 12), "store": rng.randint(0, 6), "flow": rng.randint(0, 4),
             "select1": rng.randint(0, 5), "select2": rng.randint(0, 5), "const": rng.randint(0, 3),
             "const_alu": rng.randint(0, 3)}
        assert el.capped_floor(c) == es.choice_floor(c)["cycles"], c
        assert el.capped_floor(c, {"flow": 0.5}) >= el.capped_floor(c)


def flat_table(ctx, **base):
    coef = {name: dict.fromkeys(es.CATEGORIES, 0) for name in es.feature_names(ctx)}
    coef["base"].update(base)
    return es.CostTable(ctx, coef)


@pytest.fixture(scope="module")
def table(tmp_path_factory):
    return es.calibrate(CTX, str(tmp_path_factory.mktemp("cal")), n_random=10, seed=3, log=None)


def test_nogood_and_incumbent(table):
    first = es.solve_plan_isolated(table, time_limit=20)
    key = es.plan_group_counts(first["rows"], CTX)
    second = es.solve_plan_isolated(table, time_limit=20, nogoods=[{f"{r}:{e}": v for (r, e), v in key.items()}])
    assert second["status"] != "infeasible"
    assert es.plan_group_counts(second["rows"], CTX) != key
    assert second["floor"]["cycles"] >= first["floor"]["cycles"]
    # Nothing has a floor below the optimum.
    assert es.solve_plan_isolated(table, time_limit=20, incumbent=first["floor"]["cycles"])["status"] == "infeasible"


def test_capped_floor_matches_the_solver(table):
    caps = {"flow": 0.8}
    res = es.solve_plan_isolated(table, max_util=caps, time_limit=20)
    assert el.capped_floor(res["counts"], caps) == res["capped_cycles"]


@pytest.fixture(scope="module")
def built(table, tmp_path_factory):
    """Stage 2 with three plans built (state in a temporary file)."""
    d = tmp_path_factory.mktemp("loop")
    stage2 = el.Stage2(table, str(d), str(d / "state.json"), log=None)
    for spec in ("gather", "bits+depth4", "bits"):
        stage2.evaluate(el.resolve_plan(spec, CTX)[1], f"seed {spec}")
    return stage2


def test_builds_are_proved_and_rows_stay_below_cycles(built):
    assert len(built.builds) == 3
    for b in built.builds.values():
        assert b["valid"] and b["proved"] and b["passed"] and b["counts_match"]
        assert max(b["predicted_floor"], built.rowtable.bound(b["rows"])["bound"]) <= b["cycles"]
        assert b["rows_bound_after_merge"] <= b["cycles"]


def test_row_table_reports_a_plan_below_it(built):
    rows = el.resolve_plan("bits", CTX)[1]
    key = next(k for k in built.rowtable.classes if k.startswith("r|") and built.rowtable.classes[k]["head"] > 0)
    obs = {k: dict(v) for k, v in built.rowtable.classes.items()}
    obs[key]["head"] -= 1
    bad = built.rowtable.check(obs, rows)
    assert {"class": key, "field": "head"}.items() <= bad[0].items()
    # A round class missing from a plan counts as zero of it.
    del obs[key]
    assert any(b["class"] == key and b["field"] == "count" for b in built.rowtable.check(obs, rows))


def test_windows_objective_is_the_plans_row_bound(table, built):
    specs = built.rowtable.window_specs(es.baseline_rows(CTX))
    assert specs
    res = es.solve_plan_isolated(table, windows=specs, time_limit=30)
    bound = built.rowtable.bound(res["rows"], specs)["bound"]
    assert res["objective"] == max(res["floor"]["cycles"], bound)
    assert res["objective"] >= es.solve_plan_isolated(table, time_limit=20)["floor"]["cycles"]


def test_loop_end_to_end(table, tmp_path):
    state = tmp_path / "state.json"
    argv = ["run", "--batch-size", "32", "--state", str(state), "--budget", "1", "--patience", "1",
            "--time-limit", "20", "--modes", "nogood,caps-underfilled,rows", "--seed-plan", "bits",
            "--json", str(tmp_path / "out.json")]
    # Use the calibrated fixture rather than the context's cache.
    table_path = tmp_path / "table.json"
    table_path.write_text(json.dumps(table.to_json()))
    assert el.main(argv + ["--table", str(table_path)]) == 0
    out = json.loads((tmp_path / "out.json").read_text())
    modes = {e["mode"] for e in out["trace"]}
    assert {"nogood", "caps-underfilled", "rows"} <= modes
    for e in out["trace"]:
        if "cycles" in e:
            assert e["proved"] and e["passed"]
    saved = json.loads(state.read_text())
    n = len(saved["builds"])
    # A second run reuses the state: the seed is not rebuilt.
    assert el.main(["report", "--batch-size", "32", "--state", str(state), "--table", str(table_path)]) == 0
    assert len(json.loads(state.read_text())["builds"]) == n


def test_greedy_options_are_part_of_the_state(table, tmp_path):
    """Builds with the load-tail flags are recorded with them, and a state refuses other flags."""
    state = str(tmp_path / "state.json")
    stage2 = el.Stage2(table, str(tmp_path), state, log=None, greedy_extra=es.LOAD_TAIL_GREEDY)
    rec, built = stage2.evaluate(el.resolve_plan("bits", CTX)[1], "seed bits")
    assert built and rec["valid"] and rec["proved"] and rec["passed"]
    assert rec["peak_live"] and rec["peak_live"] <= 1536 and rec["budget_exceeded"] is False
    assert json.loads((tmp_path / "state.json").read_text())["greedy_extra"] == es.LOAD_TAIL_GREEDY
    with pytest.raises(ValueError, match="greedy options"):
        el.Stage2(table, str(tmp_path), state, log=None, greedy_extra={})
    # None (report) accepts the stored options.
    assert el.Stage2(table, str(tmp_path), state, log=None).greedy_extra == es.LOAD_TAIL_GREEDY


def test_scratch_budget_failure_is_a_recorded_outcome(table, tmp_path, monkeypatch):
    def over_budget(*a, **k):
        raise RuntimeError("Scratch budget exceeded: peak_live 1600 > budget 1536")

    monkeypatch.setattr(es, "measure", over_budget)
    stage2 = el.Stage2(table, str(tmp_path), str(tmp_path / "s.json"), log=None)
    rec, built = stage2.evaluate(el.resolve_plan("bits", CTX)[1], "seed bits")
    assert built and not rec["valid"] and rec["budget_exceeded"] and rec["peak_live"] == 1600
    assert stage2.best() is None
    assert el.failed_builds(stage2)[0]["peak_live"] == 1600

    def other_error(*a, **k):
        raise RuntimeError("something else")

    monkeypatch.setattr(es, "measure", other_error)
    with pytest.raises(RuntimeError, match="something else"):
        stage2.evaluate(el.resolve_plan("gather", CTX)[1], "seed gather")
