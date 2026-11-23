"""Tests for scripts.emission_ablation (the table in docs/EMISSION_ABLATION.md)."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernel_build_args import args_to_build_kwargs  # noqa: E402
from scripts import emission_ablation as ea  # noqa: E402


def _report(vector_op_equivalents, flow, load_floor=0, store_floor=0):
    return {"emission": {"vector_op_equivalents": vector_op_equivalents},
            "engines": {"flow": {"ops": flow}},
            "floor": {"terms": {"load": load_floor, "store": store_floor}}}


def _select(op_id, true_val, base):
    return {"id": op_id, "opcode": "select_choice", "dest": f"v{op_id}_v",
            "srcs": [{"VReg": {"id": 1}}, {"VReg": {"id": true_val}}, {"VReg": {"id": base}}]}


def _sched(n_delta, n_plain, on_flow=True):
    """n_delta selects whose true value is base + delta, n_plain without; all lowered to flow or to valu."""
    ssa = [{"id": 900, "opcode": "+", "dest": "v50_v", "srcs": [{"VReg": {"id": 40}}, {"VReg": {"id": 41}}]},
           {"id": 901, "opcode": "vbroadcast", "dest": "v60_v", "srcs": [{"VReg": {"id": 7}}]}]
    lowered = []
    for i in range(n_delta + n_plain):
        delta = i < n_delta
        ssa.append(_select(i, 50 if delta else 60, 40))
        if on_flow:
            lowered.append({"engine": "flow", "from_select_choice": i})
        else:
            lowered += [{"engine": "valu", "from_select_choice": i}] * (1 if delta else 2)
    return {"ssa_ops": ssa, "ops": lowered}


def test_select_costs_distinguish_precomputed_deltas():
    costs = ea.select_costs(_sched(n_delta=2, n_plain=3))
    assert sorted(costs.values()) == [1, 1, 2, 2, 2]


@pytest.mark.parametrize("veq, n_delta, n_plain, expected", [
    (75, 0, 0, 10),     # no selects: 75 vector-op equivalents at 7.5 per cycle
    (75, 10, 0, 10),    # ten selects fit on the ten flow slots
    (75, 0, 100, 29),   # T flow slots take T selects, the other 100 - T cost 2 each: 75 + 2(100 - T) <= 7.5 T
    (0, 100, 0, 12),    # 1-op selects: 100 - T <= 7.5 T
])
def test_best_select_split_floor(veq, n_delta, n_plain, expected):
    sched = _sched(n_delta, n_plain)
    assert ea.best_select_split_floor(sched, _report(veq, flow=n_delta + n_plain)) == expected


def test_best_select_split_floor_is_the_same_however_the_run_encoded_the_selects():
    on_flow = ea.best_select_split_floor(_sched(4, 6, on_flow=True), _report(100, flow=10))
    # Encoded on the vector ALU: 4 one-op and 6 two-op selects are 16 more vector ops, no flow.
    on_valu = ea.best_select_split_floor(_sched(4, 6, on_flow=False), _report(116, flow=0))
    assert on_flow == on_valu


def test_best_select_split_floor_respects_loads_and_fixed_flow():
    assert ea.best_select_split_floor(_sched(0, 0), _report(15, flow=0, load_floor=40)) == 40
    # 30 flow ops that are not selects leave no flow slot for the 5 selects before cycle 31.
    assert ea.best_select_split_floor(_sched(0, 5), _report(0, flow=35)) == 30


def test_every_configuration_names_real_build_options():
    """A typo in CONFIGS would otherwise only show up as a PyO3 error at run time."""
    known = set(args_to_build_kwargs(None, use_git_sha=False))
    for name, options in ea.CONFIGS.items():
        assert set(options) <= known, name
    assert set(ea.NOT_BY_DEFAULT) <= set(ea.CONFIGS)
    assert ea.CONFIGS["baseline"] == {}


def test_cli_builds_proves_and_validates(tmp_path, capsys):
    out = tmp_path / "rows.json"
    status = ea.main(["--rounds", "3", "--batch-size", "16", "--markdown", "--json", str(out),
                      "--only", "baseline,all-safe-bits,all-bits+values-only"])
    assert status == 0
    rows = json.loads(out.read_text())
    assert [r["name"] for r in rows] == ["baseline", "all-safe-bits", "all-bits+values-only"]
    assert all(r["proved"] and r["passed"] for r in rows)
    assert rows[0]["validated"] == "values and indices" and rows[2]["validated"] == "values only"
    assert rows[1]["vector_op_equivalents"] < rows[0]["vector_op_equivalents"]
    assert rows[2]["vector_op_equivalents"] < rows[1]["vector_op_equivalents"]
    assert all(r["floor_best_select_split"] <= r["floor"] for r in rows)
    table = capsys.readouterr().out
    assert "| Configuration |" in table and "yes (values only)" in table


def test_cli_rejects_unknown_configuration_and_wrong_shape_for_upstream():
    with pytest.raises(SystemExit):
        ea.main(["--only", "no-such-config"])
    with pytest.raises(SystemExit):
        ea.main(["--rounds", "4", "--upstream"])


def test_build_options_override_validation_and_saved_kernel(tmp_path):
    """A --build-option that changes the shape or values_only is validated as built, and --keep
    keeps the kernel (CodeRabbit on #48: validation used the configuration and args instead)."""
    out = tmp_path / "rows.json"
    status = ea.main(["--rounds", "3", "--batch-size", "16", "--json", str(out), "--keep", str(tmp_path),
                      "--only", "baseline", "--build-option", "rounds=2", "--build-option", "values_only=true"])
    assert status == 0
    (row,) = json.loads(out.read_text())
    assert row["shape"] == {"forest_height": 10, "batch_size": 16, "rounds": 2}
    assert row["validated"] == "values only" and row["passed"]
    kernel = json.loads((tmp_path / "kernel_baseline.json").read_text())
    assert (kernel["rounds"], kernel["batch_size"], kernel["values_only"]) == (2, 16, True)
    assert row["kernel"] == str(tmp_path / "kernel_baseline.json")


def test_upstream_shape_check_uses_build_options():
    with pytest.raises(SystemExit):
        ea.main(["--upstream", "--build-option", "rounds=4"])
