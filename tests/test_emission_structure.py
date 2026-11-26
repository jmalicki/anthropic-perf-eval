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
    flow_only = dict.fromkeys(es.CATEGORIES, 0) | {"flow": 100}
    assert es.choice_floor(flow_only)["cycles"] == 100
    assert es.choice_floor(flow_only)["binding"] == ["flow"]
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
    result = es.check_table(table, str(tmp_path), n_plans=12, seed=11, log=None)
    assert result["compared"] >= 1 and result["bad"] == []


def test_table_by_instantiation_equals_calibrated(table):
    """The binding's cost table, derived from one program holding every alternative, equals
    the table fitted from probe builds entry for entry, products of shared constants included."""
    derived, doc = es.derive_table(CTX)
    assert es.table_differences(derived, table) == []
    assert any(derived.coefficients[name]["alu"] for name in es.feature_names(CTX) if name.startswith("and:"))
    assert doc["problems"] == [] and doc["rule"] == "tree-access"
    choice_rounds = len(CTX.choice_rounds())
    assert len(doc["sites"]) == choice_rounds * CTX.n_groups
    site = doc["sites"][0]
    assert set(site["alternatives"]) == {"gather", "bits", "bits-flow", "mobius"}
    assert {f["name"]: f["source"] for f in site["facts"]}["depth"] == "structure"


@pytest.mark.parametrize("context", [
    es.Context(rounds=16, batch_size=32, index_form="zero-based"),  # bits not matched on the zero-based chain
    es.Context(rounds=16, batch_size=32, values_only=True),         # tail:* liveness is per group
])
def test_table_by_instantiation_refuses_unsupported_contexts(context):
    with pytest.raises(RuntimeError, match="problems"):
        es.derive_table(context)


@pytest.mark.ci_only
@pytest.mark.timeout(1800)
def test_table_by_instantiation_equals_calibrated_32_groups(tmp_path):
    """The parity gate at the shape of the recorded results (144 probe builds, a few minutes)."""
    ctx = es.Context(rounds=16, batch_size=256)
    calibrated = es.calibrate(ctx, str(tmp_path), log=None)
    derived, _ = es.derive_table(ctx)
    assert es.table_differences(derived, calibrated) == []


def test_counts_cover_every_ssa_op(tmp_path):
    rows = es.flag_plan(CTX, "bits-flow", "after-wrap")
    _, _, sched, _ = es.build_emission(rows, CTX, str(tmp_path))
    counts = es.emission_counts(sched)
    assert sum(counts.values()) == len(sched["ssa_ops"])
    assert counts["select1"] > 0 and counts["select2"] > 0


def test_what_ifs_adjust_the_table_and_are_labelled(table):
    rows = es.flag_plan(CTX, "bits")
    base = table.predict(rows)
    no_address = es.what_if_gather_without_address_op(table).predict(rows)
    gather_group_rounds = 14 * CTX.n_groups - 6 * CTX.n_groups  # rounds 1-10, 15 gather; 1-3, 12-14 select
    assert base["vec_shiftable"] - no_address["vec_shiftable"] == gather_group_rounds
    packets = es.what_if_packets(table, (5,))
    assert table.predict(rows)["load"] - packets.predict(rows)["load"] == 8 * CTX.n_groups - 3 * 32 // 8
    assert any("not calibrated" in n for n in packets.notes)
    with pytest.raises(ValueError, match="deeper than the cached levels"):
        es.what_if_packets(table, (3,))
    with pytest.raises(ValueError, match="level 11 does not exist"):
        es.what_if_packets(table, (10,))


def flat_table(ctx, **base):
    """A cost table whose counts are `base` whatever the plan (for solver edge cases)."""
    coef = {name: dict.fromkeys(es.CATEGORIES, 0) for name in es.feature_names(ctx)}
    coef["base"].update(base)
    return es.CostTable(ctx, coef)


def test_caps_are_exact_and_bound_t_by_the_cap():
    ctx = es.Context(rounds=16, batch_size=8)
    # 1 flow op under a 0.001 flow cap needs 1,000 cycles (beyond the uncapped bound on T).
    assert es.solve_plan_isolated(flat_table(ctx, flow=1), max_util={"flow": 0.001},
                                  time_limit=10)["capped_cycles"] == 1000
    # 901 flow ops under 0.9006 need 1,001 cycles; rounding the cap to 0.901 would allow 1,000.
    assert es.solve_plan_isolated(flat_table(ctx, flow=901), max_util={"flow": 0.9006},
                                  time_limit=10)["capped_cycles"] == 1001
    # A cap with more precision than CP-SAT's coefficients allow is rejected, naming the cap.
    with pytest.raises(ValueError, match="0.3333333333333333 has more than 6 decimal places"):
        es.solve_plan_isolated(flat_table(ctx, flow=901), max_util={"flow": 0.3333333333333333}, time_limit=10)


def test_fixed_root_and_encoding_iterators():
    ctx = es.Context(rounds=16, batch_size=8)
    result = es.solve_plan_isolated(flat_table(ctx, flow=1), encodings=iter(es.ENCODINGS),
                                    fixed={0: "mobius"}, time_limit=10)
    assert result["rows"][0] == ["mobius"]
    with pytest.raises(ValueError, match="not available"):
        es.solve_plan(flat_table(ctx, flow=1), fixed={5: "bits"})


def test_one_group_design_plans_are_valid():
    ctx = es.Context(rounds=16, batch_size=8)
    for rows in es.design_plans(ctx, n_random=5, seed=1):
        assert all(len(row) == 1 for row in rows)


def test_packet_what_if_needs_single_visits(table):
    long_ctx = es.Context(rounds=18, batch_size=32)
    with pytest.raises(ValueError, match="exactly once"):
        es.what_if_packets(es.CostTable(long_ctx, table.coefficients), (5,))


def test_scatter_escapes_text():
    import xml.etree.ElementTree as ET

    row = {"name": "a & b", "emitted_floor": 990, "simulator_cycles": 1100}
    ET.fromstring(es.scatter_svg([("Flow & Load", [row])]))


def test_values_only_tail_patterns_make_the_fit_exact(tmp_path):
    """With values_only, a group's index updates after its last index read are dead, so the
    counts depend on each group's gather/select pattern over rounds 12 to 15. Without the
    tail:* features calibration has no exact fit; with them it predicts unseen plans exactly and
    the solved plan emits exactly its predicted counts."""
    ctx = es.Context(rounds=16, batch_size=32, values_only=True)
    assert ctx.tail_rounds() == [12, 13, 14, 15]
    table = es.calibrate(ctx, str(tmp_path), n_random=20, seed=3, log=None)
    checked = es.check_table(table, str(tmp_path), n_plans=8, seed=13, log=None)
    assert checked["compared"] >= 1 and checked["bad"] == []
    result = es.solve_plan_isolated(table, time_limit=30)
    _, _, sched, _ = es.build_emission(result["rows"], ctx, str(tmp_path), name="vo_solved")
    assert es.emission_counts(sched) == result["counts"]
    # Fixing the last round to gather keeps select work out of the schedule's tail.
    fixed = es.solve_plan_isolated(table, time_limit=30, fixed={15: "gather"})
    assert fixed["rows"][15] == ["gather"] * ctx.n_groups
    assert fixed["floor"]["cycles"] >= result["floor"]["cycles"]
    _, _, sched, _ = es.build_emission(fixed["rows"], ctx, str(tmp_path), name="vo_fixed")
    assert es.emission_counts(sched) == fixed["counts"]


def test_tail_pattern_enumeration_is_bounded():
    """Height 4, 40 rounds: every non-root round is a tail round (2^32 patterns); refuse it
    before enumerating instead of exhausting memory. The 16-round default (4 tail rounds) works."""
    with pytest.raises(ValueError, match="2\\^32 tail patterns"):
        es.Context(rounds=40, forest_height=4, values_only=True).tail_patterns()
    assert len(es.Context(rounds=16, values_only=True).tail_patterns()) == 16


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


def test_derive_reports_before_calibrating(monkeypatch, capsys):
    """`derive` runs and prints the derivation before it loads (or calibrates) the reference:
    a calibration failure is reported after the derivation, not instead of it."""
    called = []

    def fake_derive(ctx):
        called.append("derive")
        return es.CostTable(ctx, {}, notes=[]), {"sites": [1, 2, 3], "resources": [1]}

    def failing_calibration(*args, **kwargs):
        called.append("calibrate")
        raise RuntimeError("probe build failed")

    monkeypatch.setattr(es, "derive_table", fake_derive)
    monkeypatch.setattr(es, "load_or_calibrate", failing_calibration)
    assert es.main(["derive", "--rounds", "2", "--batch-size", "16"]) == 1
    out = capsys.readouterr().out
    assert called == ["derive", "calibrate"]
    assert "3 sites" in out and "probe build failed" in out


@pytest.mark.parametrize("default", [False, True])
def test_derive_refuses_to_overwrite_its_reference(tmp_path, monkeypatch, default):
    """--json naming the reference table (given, or the default when --table is omitted) is
    refused before anything runs."""
    monkeypatch.setattr(es, "derive_table", lambda ctx: pytest.fail("derived despite the refusal"))
    if default:
        ref = es.default_table_path(es.Context(rounds=2, batch_size=16))
        argv = ["derive", "--rounds", "2", "--batch-size", "16", "--json", ref]
    else:
        ref = str(tmp_path / "costs.json")
        argv = ["derive", "--rounds", "2", "--batch-size", "16", "--table", ref,
                "--json", str(tmp_path / "." / "costs.json")]
    with pytest.raises(SystemExit) as e:
        es.main(argv)
    assert e.value.code == 2
