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


def test_scratch_budget_failure_is_a_recorded_outcome(table, tmp_path):
    """A real over-budget build (batch 32 needs about 440 words; give it 300) is recorded as a
    failed plan with the builder's own peak, from the typed exception."""
    stage2 = el.Stage2(table, str(tmp_path), str(tmp_path / "s.json"), log=None,
                       greedy_extra={"scratch_budget_override": 300})
    rec, built = stage2.evaluate(el.resolve_plan("bits", CTX)[1], "seed bits")
    assert built and not rec["valid"] and rec["budget_exceeded"]
    assert rec["peak_live"] > 300 and "scratch budget exceeded" in rec["error"]
    assert rec["error"].startswith("ScratchBudgetExceeded")
    assert stage2.best() is None
    assert el.failed_builds(stage2)[0]["peak_live"] == rec["peak_live"]


def test_only_the_typed_budget_error_is_a_plan_outcome(table, tmp_path, monkeypatch):
    import scheduler_rs

    assert issubclass(scheduler_rs.ScratchBudgetExceeded, RuntimeError)
    stage2 = el.Stage2(table, str(tmp_path), str(tmp_path / "s.json"), log=None)

    def untyped(*a, **k):  # mentions the budget, but is not the builder's budget failure
        raise RuntimeError("scratch budget exceeded: peak_live 1600 > budget 1536")

    monkeypatch.setattr(es, "measure", untyped)
    with pytest.raises(RuntimeError, match="1600"):
        stage2.evaluate(el.resolve_plan("gather", CTX)[1], "seed gather")
    assert el.scratch_budget_error(scheduler_rs.ScratchBudgetExceeded("m", 1600, 1536)) == {
        "peak_live": 1600, "budget": 1536}


# ----------------------------------------------------------------------------
# Review follow-ups: canonical no-goods, sound rows cut, reproducibility records
# ----------------------------------------------------------------------------
def _fake_build(rows, cycles=1000):
    return {"name": "fake", "plan": "fake", "rows": [list(r) for r in rows], "valid": True, "cycles": cycles,
            "simulator_cycles": cycles, "predicted_floor": 900, "rows_bound_before_merge": 0,
            "rows_bound_after_merge": 0}


def _non_canonical(rows):
    """Same model point, another arrangement: the root (not a choice round) selected with mobius."""
    other = [list(r) for r in rows]
    other[0] = ["mobius"] * len(other[0])
    return other


def test_solver_rows_are_canonical_and_seeds_need_not_be(table):
    res = es.solve_plan_isolated(table, time_limit=20, deterministic_time=5)
    assert el.is_canonical(res["rows"], CTX)
    other = _non_canonical(res["rows"])
    assert es.plan_model_key(other, CTX) == es.plan_model_key(res["rows"], CTX)
    assert not el.is_canonical(other, CTX)
    # A mixed round arranged in another order is the same point but not canonical either.
    mixed = es.baseline_rows(CTX)
    r = CTX.choice_rounds()[0]
    mixed[r] = ["gather", "bits"] * (CTX.n_groups // 2)
    assert es.canonical_rows(mixed, CTX)[r] == sorted(mixed[r], key=es.ENCODINGS.index)
    assert not el.is_canonical(mixed, CTX)


def test_a_non_canonical_build_does_not_cut_its_model_point(table, tmp_path):
    stage2 = el.Stage2(table, str(tmp_path), None, log=None, fingerprint={"extension_sha256": "test"})
    first = es.solve_plan_isolated(table, time_limit=20, deterministic_time=5)
    seed = _non_canonical(first["rows"])
    stage2.builds[el.plan_key(seed)] = _fake_build(seed)
    assert el._built_nogoods(stage2) == []
    # So stage 1 can still return the model's own arrangement of those counts, unbuilt.
    res = el.solve(stage2, nogoods=el._built_nogoods(stage2), time_limit=20)
    assert res["rows"] == first["rows"] and el.plan_key(res["rows"]) not in stage2.builds
    # Once the canonical arrangement is built, the point is cut.
    stage2.builds[el.plan_key(first["rows"])] = _fake_build(first["rows"])
    assert el._built_nogoods(stage2) == [es.plan_model_key(first["rows"], CTX)]
    res = el.solve(stage2, nogoods=el._built_nogoods(stage2), time_limit=20)
    assert es.plan_model_key(res["rows"], CTX) != es.plan_model_key(first["rows"], CTX)


def test_floor_incumbent_is_the_sound_cut(table):
    first = es.solve_plan_isolated(table, time_limit=20, deterministic_time=5)
    res = es.solve_plan_isolated(table, time_limit=20, floor_incumbent=first["floor"]["cycles"])
    assert res["status"] == "infeasible" and res["solves"][0]["status"] == "INFEASIBLE"


@pytest.mark.parametrize("cut", ["sound", "heuristic"])
def test_rows_cut_is_sound_by_default_and_never_claims_exhaustion_when_heuristic(table, tmp_path, monkeypatch, cut):
    stage2 = el.Stage2(table, str(tmp_path), None, log=None, fingerprint={"extension_sha256": "test"})
    rows = es.baseline_rows(CTX)
    rec = _fake_build(rows, cycles=1000)
    rec["rows_bound_before_merge"], rec["rows_bound_after_merge"] = 990, 981  # overestimated by 9
    stage2.builds[el.plan_key(rows)] = rec
    seen = {}

    def fake_solve(_stage2, **kwargs):
        seen.update(kwargs)
        return {"status": "infeasible", "solves": [{"phase": "main", "status": "INFEASIBLE"}]}

    monkeypatch.setattr(el, "solve", fake_solve)
    el.loop_rows(stage2, 1, 1, 1.0, cut=cut)
    feedback = stage2.trace[-1]["feedback"]
    assert seen["floor_incumbent"] == 1000
    if cut == "sound":
        assert seen["incumbent"] is None and feedback.startswith("exhausted")
    else:
        assert seen["incumbent"] == 1000 + 9 and "exhausted" not in feedback and "heuristic" in feedback
    assert stage2.trace[-1]["solves"] == [{"phase": "main", "status": "INFEASIBLE"}]


def test_solves_are_recorded_and_deterministic(table):
    a = es.solve_plan_isolated(table, time_limit=60, deterministic_time=5)
    b = es.solve_plan_isolated(table, time_limit=60, deterministic_time=5)
    assert a["rows"] == b["rows"]
    phases = [s["phase"] for s in a["solves"]]
    assert phases[0] == "main" and phases[-1] == "tie_break"
    for s in a["solves"]:
        assert s["status"] in ("OPTIMAL", "FEASIBLE", "UNKNOWN") and not s["stopped_on_wall_clock"]
        assert s["deterministic_time"] <= 5 + 1e-6


def test_state_from_another_scheduler_build_is_refused(table, tmp_path):
    state = tmp_path / "state.json"
    stage2 = el.Stage2(table, str(tmp_path), str(state), log=None, fingerprint={"extension_sha256": "a"})
    rows = es.baseline_rows(CTX)
    stage2.builds[el.plan_key(rows)] = _fake_build(rows)
    stage2.save()
    assert el.Stage2(table, str(tmp_path), str(state), log=None, fingerprint={"extension_sha256": "a"}).builds
    with pytest.raises(ValueError, match="not comparable"):
        el.Stage2(table, str(tmp_path), str(state), log=None, fingerprint={"extension_sha256": "b"})
    # Reports only read it.
    assert el.Stage2(table, str(tmp_path), str(state), log=None, fingerprint={"extension_sha256": "b"},
                     check_scheduler=False).scheduler == {"extension_sha256": "a"}
    assert "extension_sha256" in el.scheduler_fingerprint() or "error" in el.scheduler_fingerprint()


def test_parse_greedy_options():
    assert el.parse_greedy_options(["greedy_budget_force=stall", "pressure_inflation_threshold=0.98",
                                    "greedy_groups_in_flight=32", "greedy_split_shift=true"]) == {
        "greedy_budget_force": "stall", "pressure_inflation_threshold": 0.98,
        "greedy_groups_in_flight": 32, "greedy_split_shift": True}
    with pytest.raises(ValueError):
        el.parse_greedy_options(["stall"])


@pytest.mark.parametrize("objective", ["floor", "greedy-model"])
def test_topk_enumerates_score_levels(table, tmp_path, objective):
    """topk builds distinct plans with non-decreasing scores, moving up a level after
    `per_level` plans, each PROVED and PASSED under the stall judge and within the 1,535
    assignable words (plus the reserved zero word); the recorded score is the plan's own score. Deterministic-time solves."""
    judge = dict(es.LOAD_TAIL_GREEDY, greedy_budget_force="stall", pressure_inflation_threshold=0.98)
    stage2 = el.Stage2(table, str(tmp_path), str(tmp_path / "s.json"), log=None, greedy_extra=judge)
    stage2.deterministic_time = 2.0
    el.loop_topk(stage2, 3, 60.0, objective, per_level=1)
    entries = [e for e in stage2.trace if e["mode"] == f"topk-{objective}"]
    assert len(entries) == 3
    assert len({e["plan"] for e in entries}) == 3
    scores = [e["score"] for e in entries]
    assert scores == sorted(scores) and len(set(scores)) == 3, scores  # one plan per level
    assert [e["level"] for e in entries] == [1, 2, 3]
    for b in stage2.builds.values():
        # peak_live counts the reserved zero word: 1,535 assignable words + it = 1,536.
        assert b["valid"] and b["peak_live"] - 1 <= 1535 and b["budget_exceeded"] is False
        assert es.plan_score(table, b["rows"], objective) in scores


def test_budget_exceeded_is_recorded_from_the_build(table, tmp_path, monkeypatch):
    """Stage2 records the build's own budget flag (it used to write False)."""
    real = es.measure

    def flagged(*a, **k):
        row = real(*a, **k)
        row["budget_exceeded"] = True
        return row

    monkeypatch.setattr(es, "measure", flagged)
    stage2 = el.Stage2(table, str(tmp_path), str(tmp_path / "s.json"), log=None)
    rec, _ = stage2.evaluate(el.resolve_plan("bits", CTX)[1], "seed bits")
    assert rec["budget_exceeded"] is True


def test_greedy_options_are_validated():
    el.validate_greedy_options({"greedy_budget_force": "stall", "pressure_inflation_threshold": 0.98}, CTX)
    with pytest.raises(ValueError, match="unknown"):
        el.validate_greedy_options({"greedy_budgt_force": "stall"}, CTX)  # typo
    for fixed in ({"rounds": 4}, {"use_greedy": False}, {"select_bit_source": "path"}):
        with pytest.raises(ValueError, match="fixed"):
            el.validate_greedy_options(fixed, CTX)
    # Read by the build but no effect on a greedy build (review of #83).
    for no_effect in ({"cpsat_workers": 4}, {"force_zero_gap": True}, {"per_section_optimize_secs": 10},
                      {"asap_alap_tightening": "none"}, {"liveness_detail": "none"}, {"no_cse_groups": True}):
        with pytest.raises(ValueError, match="do not affect a greedy build"):
            el.validate_greedy_options(no_effect, CTX)
    for ok in ({"greedy_justify": 10}, {"scheduling_pressure_policy": "cascade"}, {"enable_schedule_with_fixed_choices": True}):
        el.validate_greedy_options(ok, CTX)
    for name in el.GREEDY_BUILD_OPTIONS:
        assert name in el.build_option_names(), name


def test_topk_stops_when_the_plan_space_is_exhausted(tmp_path, monkeypatch):
    """Review of #83: 2 rounds, one group: the only choice round (depth 1) has 4 encodings, so 4
    plans, all with the same score under a flat table. One plan per level is built, the other
    three are rejected at the next level, and then nothing is left: the loop must stop after
    a few solves (before, each further level cost a solve, up to 10^6 levels)."""
    ctx = es.Context(rounds=2, batch_size=8)
    table = flat_table(ctx, load=100)
    stage2 = el.Stage2(table, str(tmp_path), None, log=None, fingerprint={"extension_sha256": "test"})

    def fake_evaluate(rows, name, *a, **k):
        rec = dict(_fake_build(rows), name=name, counts=table.predict(rows), proved=True, passed=True,
                   predicted_floor=es.choice_floor(table.predict(rows))["cycles"])
        stage2.builds[el.plan_key(rows)] = rec
        return rec, True

    calls = []
    real_solve = el.solve

    def counting_solve(*a, **k):
        calls.append(k)
        return real_solve(*a, **k)

    monkeypatch.setattr(stage2, "evaluate", fake_evaluate)
    monkeypatch.setattr(el, "solve", counting_solve)
    el.loop_topk(stage2, 10, 20.0, "floor", per_level=1)
    assert len(stage2.builds) == 1, "one plan per score level, and there is one level"
    assert stage2.trace[-1]["feedback"].startswith("exhausted")
    assert len(calls) <= 8, f"{len(calls)} solves"


def test_error_fingerprints_never_match(table, tmp_path):
    """Review of #76: a fingerprint that failed is a mismatch, even against an equal failure."""
    assert el.same_scheduler({"extension_sha256": "a"}, {"extension_sha256": "a"})
    assert not el.same_scheduler({"extension_sha256": "a"}, {"extension_sha256": "b"})
    err = {"error": "scheduler_rs not installed"}
    assert not el.same_scheduler(err, dict(err))
    assert not el.same_scheduler(None, {"extension_sha256": "a"})
    state = tmp_path / "state.json"
    stage2 = el.Stage2(table, str(tmp_path), str(state), log=None, fingerprint=err)
    rows = es.baseline_rows(CTX)
    stage2.builds[el.plan_key(rows)] = _fake_build(rows)
    stage2.save()
    with pytest.raises(ValueError, match="not comparable"):
        el.Stage2(table, str(tmp_path), str(state), log=None, fingerprint=dict(err))


# ----------------------------------------------------------------------------
# Review follow-ups (CodeRabbit on #74)
# ----------------------------------------------------------------------------
def test_rows_loop_survives_a_failed_build_without_incumbent(table, tmp_path, monkeypatch):
    """No valid incumbent yet and the candidate fails the scratch budget: the loop records it
    and goes on (before, the feedback read best['cycles'] from None and aborted the run)."""
    stage2 = el.Stage2(table, str(tmp_path), None, log=None, fingerprint={"extension_sha256": "test"})
    rows = es.baseline_rows(CTX)
    monkeypatch.setattr(el, "solve", lambda *a, **k: {"status": "optimal", "rows": rows, "solves": []})

    def failing_evaluate(r, name, *a, **k):
        return stage2._failed(r, name, el.plan_key(r), "ScratchBudgetExceeded: test", peak_live=1600), True

    monkeypatch.setattr(stage2, "evaluate", failing_evaluate)
    assert el.loop_rows(stage2, 2, 5, 1.0) is None
    rows_entries = [e for e in stage2.trace if e["mode"] == "rows"]
    assert len(rows_entries) == 2 and all("no valid incumbent" in e["feedback"] for e in rows_entries)


def test_max_axis_below_two_is_rejected(table, tmp_path):
    with pytest.raises(SystemExit):
        el.main(["report", "--state", str(tmp_path / "s.json"), "--max-axis", "1"])
    with pytest.raises(ValueError, match="max_axis"):
        el.RowTable(CTX, {}, 0, max_axis=1)._axis([1, 2, 3])


def test_valu_diagnostic_counts_selects_on_the_valu(table):
    """valu_after_greedy_shift counts native vector ops left on the VALU plus selects lowered
    to multiply_adds there; a build without the field counts none."""
    stage2 = el.Stage2(table, "/nonexistent", None, log=None, fingerprint={"extension_sha256": "test"})
    rows = es.baseline_rows(CTX)
    counts = table.predict(rows)
    base = dict(_fake_build(rows), counts=counts, predicted_floor=es.choice_floor(counts)["cycles"],
                summary={"static_bound_any": 0, "valu_shifted": 0})
    native = counts["vec_fixed"] + counts["vec_shiftable"]
    stage2.builds = {"a": base}
    without = el.surrogates(stage2, None)[0]["valu_after_greedy_shift"]
    assert without == -(-native // es.CAP["valu"])
    with_selects = dict(base, summary=dict(base["summary"], select_ops_on_valu=6 * 7))
    stage2.builds = {"a": with_selects}
    assert el.surrogates(stage2, None)[0]["valu_after_greedy_shift"] == -(-(native + 42) // es.CAP["valu"])
