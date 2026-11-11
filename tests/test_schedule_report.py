"""Tests for scripts.schedule_report (per-run schedule quality report).

The floor formula is checked on hand-made inputs; the report is checked on real kernels
built through the scheduler_rs binding.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scheduler_rs  # noqa: E402
from scripts import schedule_report as sr  # noqa: E402


def _build(tmp_path_factory, name, **kwargs):
    path = tmp_path_factory.mktemp(name) / "sched.json"
    args = dict(forest_height=10, n_nodes=2047, batch_size=16, rounds=5, enable_valu_shift=True,
                verbose=False, use_cpsat=False, output_json=str(path))
    args.update(kwargs)
    # The CLI records its options in the JSON; do the same for the fields the report reads.
    args["run_options_json"] = json.dumps(
        {"use_cpsat": args["use_cpsat"], "rounds": args["rounds"], "batch_size": args["batch_size"]})
    scheduler_rs.build_tick_tock_kernel_ssa(**args)
    with open(path) as f:
        return json.load(f), str(path)


def _ssa_vector_ops(sched):
    """Vector ops as emitted: SSA `valu` ops plus selects the scheduler encoded as multiply_add."""
    fma_selects = sum(1 for o in sched["ops"] if o["engine"] == "valu" and "from_select_choice" in o)
    return sum(1 for o in sched["ssa_ops"] if o["engine"] == "valu") + fma_selects


@pytest.fixture(scope="module")
def greedy(tmp_path_factory):
    """5 rounds: cached levels, tree-reduce selects and one gather round."""
    return _build(tmp_path_factory, "greedy")


@pytest.fixture(scope="module")
def cpsat(tmp_path_factory):
    """CP-SAT exercises shifted ops, alternative encodings and lb_cycles."""
    return _build(tmp_path_factory, "cpsat", rounds=2, use_cpsat=True, cpsat_time_limit=20.0)


# ---------------------------------------------------------------------------
# Floor formula on hand-made inputs
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("vector, scalar, expected", [
    (0, 0, 0),
    (60, 0, 8),      # 7.5 vector-op equivalents per cycle: 48 on valu, 12 as 96 scalar ops
    (61, 0, 9),
    (6, 12, 1),      # both ALUs exactly full
    (7, 0, 1),       # one vector op runs as 8 scalar ops
    (7, 12, 2),      # ... unless the scalar ALU is already full
    (0, 13, 2),
    (9506, 162, 1271),  # the 16-round emission (docs/BOUND_GAP_ANALYSIS.md section 2)
])
def test_joint_alu_floor(vector, scalar, expected):
    t, shift = sr.joint_alu_floor(vector, scalar)
    assert t == expected
    assert 0 <= shift <= vector
    assert vector - shift <= 6 * t and scalar + 8 * shift <= 12 * t
    if t > 0:  # one cycle fewer is infeasible for every shift count
        assert not any(vector - s <= 6 * (t - 1) and scalar + 8 * s <= 12 * (t - 1) for s in range(vector + 1))


def test_joint_alu_floor_respects_non_shiftable_ops():
    assert sr.joint_alu_floor(60, 0, max_shift=0) == (10, 0)
    assert sr.joint_alu_floor(60, 0, max_shift=6)[0] == 9


def test_emission_floor_names_the_binding_engine():
    assert sr.emission_floor(60, 0)["binding"] == ["alu+valu"]
    f = sr.emission_floor(60, 0, load_ops=17)
    assert (f["cycles"], f["binding"]) == (9, ["load"])
    f = sr.emission_floor(60, 0, load_ops=16, flow_ops=8, store_ops=3)
    assert f["cycles"] == 8 and f["binding"] == ["alu+valu", "load", "flow"]
    assert f["terms"]["store"] == 2
    assert f["vector_op_equivalents"] == 60


# ---------------------------------------------------------------------------
# Report on real kernels
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fixture", ["greedy", "cpsat"])
def test_engine_counts_sum_to_lowered_ops(fixture, request):
    sched, _ = request.getfixturevalue(fixture)
    rep = sr.analyze(sched)
    assert sum(e["ops"] for e in rep["engines"].values()) == len(sched["ops"]) == sched["total_ops"]
    em = rep["emission"]
    assert em["vector_ops_on_valu"] + 8 * em["vector_ops_shifted_to_alu"] + em["native_scalar_ops"] == (
        rep["engines"]["valu"]["ops"] + rep["engines"]["alu"]["ops"])
    assert em["vector_ops"] == _ssa_vector_ops(sched)


@pytest.mark.parametrize("fixture", ["greedy", "cpsat"])
def test_gap_is_non_negative_and_floor_is_consistent(fixture, request):
    sched, _ = request.getfixturevalue(fixture)
    rep = sr.analyze(sched)
    assert rep["gap"]["cycles"] >= 0
    assert rep["floor"]["cycles"] + rep["gap"]["cycles"] == sched["total_cycles"]
    assert rep["floor"]["cycles"] == max(rep["floor"]["terms"].values())
    assert rep["floor"]["cycles"] >= rep["floor"]["continuous_alu_floor"]
    for e in ("load", "flow", "store"):  # a single engine's floor never exceeds the joint one
        assert rep["engines"][e]["floor"] <= rep["floor"]["cycles"]


@pytest.mark.parametrize("fixture", ["greedy", "cpsat"])
def test_round_spans_cover_all_scheduled_ops(fixture, request):
    sched, _ = request.getfixturevalue(fixture)
    rows = {r["round"]: r for r in sr.per_round(sched)}
    assert set(rows) - {None} == set(range(sched["run_options"]["rounds"]))
    assert sum(r["lowered_ops"] for r in rows.values()) == len(sched["ops"])
    seen = 0
    for o, s, _lane in sr.lowered_with_ssa(sched):
        row = rows[s["source_info"]["round"]]
        assert row["first_cycle"] <= o["cycle"] <= row["last_cycle"]
        seen += 1
    assert seen == len(sched["ops"])
    for r in rows.values():
        assert r["span"] == r["last_cycle"] - r["first_cycle"] + 1
        assert 0 <= r["first_cycle"] and r["last_cycle"] < sched["total_cycles"]
    assert sum(r["vector_ops"] for r in rows.values()) == _ssa_vector_ops(sched)


def test_utilisation_buckets_account_for_every_slot(greedy):
    sched, _ = greedy
    rep = sr.analyze(sched, buckets=7)
    u = rep["utilisation"]
    assert sum(u["bucket_cycles"]) == sched["total_cycles"]
    for e, eng in rep["engines"].items():
        used = sum(p / 100 * eng["capacity"] * w for p, w in zip(u["pct"][e], u["bucket_cycles"]))
        assert round(used) == eng["ops"]
        assert all(0 <= p <= 100 + 1e-9 for p in u["pct"][e])


def test_wording_states_the_floor_is_conditional(greedy):
    sched, _ = greedy
    rep = sr.analyze(sched)
    text = sr.format_report(rep)
    assert "floor for this emission (" in text and "not a bound on the problem" in text
    lines = sr.summary_lines(rep)
    assert len(lines) <= 3
    assert "not a bound on the problem" in " ".join(lines)
    assert f"{rep['floor']['cycles']} cycles" in lines[0]


def test_cpsat_bound_is_labelled_as_emission_scoped(cpsat):
    sched, _ = cpsat
    assert sched["run_options"]["use_cpsat"] is True
    assert sched["lb_cycles"] <= sched["total_cycles"]
    assert sched["lb_scope"] == "emission"
    rep = sr.analyze(sched)
    assert rep["run"]["scheduler"] == "cpsat"
    assert "CP-SAT bound for this emission (lb_cycles)" in sr.format_report(rep)


def test_greedy_json_has_no_bound_fields(greedy):
    sched, _ = greedy
    assert "lb_cycles" not in sched and "lb_scope" not in sched
    assert sr.analyze(sched)["run"]["scheduler"] == "greedy"


def test_scheduler_and_rounds_are_inferred_without_run_options(greedy, cpsat):
    for (sched, _), scheduler, rounds in ((greedy, "greedy", 5), (cpsat, "cpsat", 2)):
        bare = {k: v for k, v in sched.items() if k != "run_options"}
        run = sr.analyze(bare)["run"]
        assert (run["scheduler"], run["rounds"], run["batch_size"]) == (scheduler, rounds, None)


def test_cli_json_and_compare(greedy, cpsat, capsys):
    (_, a), (_, b) = greedy, cpsat
    assert sr.main([a, "--json"]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert set(rep) == {"run", "engines", "emission", "floor", "gap", "rounds", "round_overlap", "utilisation"}
    assert sr.main([a, "--compare", b]) == 0
    out = capsys.readouterr().out
    assert "emission floor" in out and "span A" in out and "not a bound on the problem" in out
    assert sr.main([a, "--compare", b, "--json"]) == 0
    both = json.loads(capsys.readouterr().out)
    assert both["a"]["run"]["rounds"] == 5 and both["b"]["run"]["rounds"] == 2


# ---------------------------------------------------------------------------
# Simulator cycles, round overlap and the --table mode
# ---------------------------------------------------------------------------
def _simulated_cycles(rounds, batch_size, **kwargs):
    """Build through the binding and run the kernel on the simulator: (machine cycles, schedule JSON)."""
    import random
    import tempfile

    from problem import DebugInfo, Input, Machine, N_CORES, Tree, build_mem_image

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "sched.json")
        instrs, _ = scheduler_rs.build_tick_tock_kernel_ssa(
            forest_height=10, n_nodes=2047, batch_size=batch_size, rounds=rounds, verbose=False,
            use_cpsat=False, output_json=path, **kwargs)
        with open(path) as f:
            sched = json.load(f)
    random.seed(123)
    forest = Tree.generate(10)
    mem = build_mem_image(forest, Input.generate(forest, batch_size, rounds))
    machine = Machine(mem, json.loads(json.dumps(list(instrs))), DebugInfo(scratch_map={}), n_cores=N_CORES)
    machine.enable_pause = False
    machine.enable_debug = False
    machine.run()
    return machine.cycle, sched


@pytest.mark.parametrize("kwargs", [{}, {"greedy_barriers": "none", "greedy_priority": "wavefront"}],
                         ids=["default", "wavefront"])
def test_simulator_cycles_match_the_machine(kwargs):
    """simulator_cycles = schedule cycles + pause bundles - empty cycles, for a schedule with
    empty barrier cycles (default) and one without (no barriers)."""
    machine_cycles, sched = _simulated_cycles(3, 32, **kwargs)
    run = sr.analyze(sched)["run"]
    assert run["simulator_cycles"] == machine_cycles
    assert run["simulator_cycles"] == run["cycles"] + sr.PAUSE_BUNDLES - run["empty_cycles"]
    if kwargs:
        assert run["empty_cycles"] == 0
    else:
        assert run["empty_cycles"] == 2, "one empty cycle per round barrier"


def test_round_overlap_is_zero_by_default_and_positive_without_barriers():
    _, default = _simulated_cycles(3, 32)
    _, wavefront = _simulated_cycles(3, 32, greedy_barriers="none", greedy_priority="wavefront")
    assert sr.analyze(default)["round_overlap"]["total_overlap"] == 0
    overlap = sr.analyze(wavefront)["round_overlap"]
    assert overlap["total_overlap"] > 0
    rounds = [r for r in sr.analyze(wavefront)["rounds"] if r["round"] is not None]
    assert overlap["sum_of_spans"] == sum(r["span"] for r in rounds)
    assert sr.round_overlap([r for r in rounds if r["round"] == 0]) is None


def test_table_mode(greedy, cpsat, capsys):
    (_, a), (_, b) = greedy, cpsat
    assert sr.main(["--table", f"five rounds={a}", b]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == sr.TABLE_HEADER
    assert set(lines[1]) == {"|", "-"} and lines[1].count("|") == lines[0].count("|")
    rep_a = sr.analyze(sr.load(a))
    first = [c.strip() for c in lines[2].strip("|").split("|")]
    assert first[:5] == ["five rounds", "5", str(rep_a["run"]["cycles"]), str(rep_a["run"]["simulator_cycles"]),
                         str(rep_a["floor"]["cycles"])]
    assert first[-1] == str(rep_a["round_overlap"]["total_overlap"])
    assert lines[3].startswith("| sched |"), "an unlabelled file is named after its stem"
    assert all(line.count("|") == lines[0].count("|") for line in lines[2:4])
    assert "not a bound on the problem" in lines[-1]

    assert sr.main(["--table", f"x={a}", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["x"]["run"]["cycles"] == rep_a["run"]["cycles"]
    assert sr.parse_table_arg("name=/tmp/a=b.json") == ("name", "/tmp/a=b.json")
    assert sr.parse_table_arg("dir/run1.json") == ("run1", "dir/run1.json")
