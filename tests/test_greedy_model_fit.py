"""es.GREEDY_MODEL is reproducible from checked-in data (scripts/greedy_model_fit.py,
docs/emission_plans/greedy_model_data.json, docs/GREEDY_SHAPE_ABLATION.md section 12.3).

- The grid fit over the 24 index-bit loop builds under 1,200 cycles gives exactly the
  constants in es.GREEDY_MODEL, and leave-one-out gives MAE 6.2 and max 18.
- The rankings quoted in the doc are what the data give (in-sample, and leave-one-out for
  the constants only).
- The out-of-sample plans (chosen by stage 1 with the model) are all predicted optimistic by
  20 cycles or more: the model fails out of sample, as the doc says.
- es.greedy_model_cycles is the formula the fit uses.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import emission_structure as es  # noqa: E402
from scripts import greedy_model_fit as gf  # noqa: E402


def test_fit_reproduces_greedy_model_and_quoted_numbers():
    data = gf.load()
    res = gf.analyse(data)
    assert gf.check(res) == []
    assert res["train_plans"] == 24
    assert round(res["loo_spearman"]["all"], 2) == 0.96 and round(res["loo_spearman"]["near"], 2) == 0.71
    assert round(res["in_sample"]["index"]["spearman_all"], 2) == 0.95
    assert round(res["in_sample"]["index"]["spearman_near"], 2) == 0.71
    assert res["in_sample"]["index"]["n_loop"] == 22 and round(res["in_sample"]["index"]["spearman_loop"], 2) == 0.94
    assert round(res["in_sample"]["path"]["spearman_all"], 2) == 0.05
    assert res["in_sample"]["index"]["n_all"] == 28 and res["in_sample"]["path"]["n_all"] == 16


def test_out_of_sample_plans_are_predicted_too_low():
    res = gf.analyse(gf.load())
    oos = {o["name"]: o for o in res["out_of_sample"]}
    assert set(oos) == {"gm_idx", "gm_idx_d1bits", "gm_path", "gm_path_r4bits"}
    assert all(o["error"] <= -20 for o in oos.values())
    assert (oos["gm_idx"]["cycles"], oos["gm_idx_d1bits"]["cycles"]) == (1051, 1058)


def test_es_formula_is_the_fitted_formula():
    th = gf.GRID[[gf.as_model(t) for t in gf.GRID].index(
        {k: es.GREEDY_MODEL[k] for k in ("a_load", "flow0", "b_flow_x100", "a_floor")})]
    for b in gf.load()["builds"]:
        assert es.greedy_model_cycles(b["counts"]) == gf.predict(b["features"], th)


def test_collect_requires_out(tmp_path):
    """Review of #82: `collect` without --out must not overwrite the checked-in dataset."""
    import pytest
    from scripts import greedy_model_fit as gmf

    data = tmp_path / "data.json"
    data.write_text('{"builds": [1]}')
    with pytest.raises(SystemExit):
        gmf.main(["collect", "--data", str(data)])
    assert data.read_text() == '{"builds": [1]}'
