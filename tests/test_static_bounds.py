"""Tests for scripts.static_bounds (static head-body-tail makespan rows).

Hand-made graphs check heads, tails and the rows; random op counts check that the t = q = 0
cells reproduce the joint ALU floor; real kernels built through the binding check that no
achieved schedule is below the bound. The Z3 soundness checks of
verification/static_bounds_proof.py run here too.
"""

import json
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scheduler_rs
from scripts import schedule_report as sr
from scripts import static_bounds as sb


def op(i, deps=(), **work):
    return sb.BoundOp(i, tuple((d, 1) for d in deps), {k.replace("joint", sb.JOINT): v for k, v in work.items()})


# ---------------------------------------------------------------------------
# Heads, tails and rows on hand-made graphs
# ---------------------------------------------------------------------------
def test_heads_and_tails_of_a_diamond():
    ops = [op(0), op(1, [0]), op(2, [0]), op(3, [1, 2]), op(4, [2])]
    assert sb.heads(ops) == {0: 0, 1: 1, 2: 1, 3: 2, 4: 2}
    assert sb.tails(ops) == {0: 2, 1: 1, 2: 1, 3: 0, 4: 0}


def test_a_chain_is_bound_by_its_length():
    ops = [op(i, [i - 1] if i else [], alu=1) for i in range(5)]
    r = sb.makespan_rows(ops)
    assert r["bound"] == 5 == r["critical_path"]
    assert r["rows"]["alu"]["floor"] == 1


def test_head_row_charges_the_ramp():
    """One root, then 24 independent loads: the loads cannot start before cycle 1, so
    12 cycles of loads plus the root's cycle; the plain floor says 13 too, but with the root
    on another engine only the head row sees it."""
    ops = [op(0, alu=1)] + [op(i, [0], load=1) for i in range(1, 25)]
    r = sb.makespan_rows(ops)
    assert r["rows"]["load"]["floor"] == 12
    assert r["rows"]["load"]["head_row"] == {"bound": 13, "t": 1}
    assert (r["bound"], r["engine"], r["t"], r["q"]) == (13, "load", 1, 0)


def test_pair_row_beats_head_and_tail_rows():
    """Loads between a root and a sink: they fit in [1, M - 2] only."""
    ops = [op(0, alu=1)] + [op(i, [0], load=1) for i in range(1, 7)] + [op(7, range(1, 7), alu=1)]
    r = sb.makespan_rows(ops)
    row = r["rows"]["load"]
    assert (row["floor"], row["head_row"]["bound"], row["tail_row"]["bound"]) == (3, 4, 4)
    assert row["best"] == {"bound": 5, "t": 1, "q": 1, "work": 6}
    assert r["bound"] == 5


def test_zero_latency_edges_do_not_add_depth():
    ops = [op(0, alu=1), sb.BoundOp(1, ((0, 1),), {}), sb.BoundOp(2, ((1, 0),), {sb.JOINT: 8})]
    assert sb.heads(ops) == {0: 0, 1: 1, 2: 1}
    assert sb.tails(ops) == {0: 1, 1: 0, 2: 0}


def test_cycle_is_rejected():
    with pytest.raises(ValueError, match="cycle"):
        sb.heads([op(0, [1]), op(1, [0])])


def test_empty():
    assert sb.makespan_rows([])["bound"] == 0


@pytest.mark.parametrize("seed", range(20))
def test_whole_schedule_cells_equal_the_joint_floor(seed):
    """With no dependencies every op has head = tail = 0, so the bound is the capacity floor:
    max(ceil(N/6), ceil(S/12), ceil((8V+S)/60)) = schedule_report.joint_alu_floor."""
    rng = random.Random(seed)
    non_shift, shift, scalar = rng.randrange(200), rng.randrange(400), rng.randrange(300)
    ops = ([op(i, valu=1, joint=8) for i in range(non_shift)]
           + [op(10_000 + i, joint=8) for i in range(shift)]
           + [op(20_000 + i, alu=1, joint=1) for i in range(scalar)])
    t, _ = sr.joint_alu_floor(non_shift + shift, scalar, max_shift=shift)
    assert sb.makespan_rows(ops)["bound"] == t


# ---------------------------------------------------------------------------
# Scheduler JSON adapter
# ---------------------------------------------------------------------------
def _json(ssa, run_options=None):
    return {"ssa_ops": [dict(id=i, engine=e, opcode=o, dep_ids=list(d)) for i, (e, o, d) in enumerate(ssa)],
            "ops": [], "run_options": run_options or {}}


def test_min_work_over_encodings():
    d = _json([("load", "const_choice", []), ("flow", "select_choice", [0]), ("valu", "^", [1]),
               ("valu", "multiply_add", [2]), ("alu", "+", [3]), ("store", "store", [4]), ("valu", "cdiv", [])])
    work = {o.id: o.work for o in sb.ops_from_schedule_json(d)}
    assert work == {0: {}, 1: {}, 2: {sb.JOINT: 8}, 3: {"valu": 1, sb.JOINT: 8}, 4: {"alu": 1, sb.JOINT: 1},
                    5: {"store": 1}, 6: {"valu": 1, sb.JOINT: 8}}


def test_shift_disabled_keeps_vector_ops_on_valu():
    d = _json([("valu", "^", [])], {"enable_valu_shift": False})
    assert sb.ops_from_schedule_json(d)[0].work == {"valu": 1, sb.JOINT: 8}


def test_barriers_are_dropped_with_their_edges():
    d = _json([("alu", "+", []), ("barrier", "barrier", [0]), ("alu", "+", [1])])
    ops = sb.ops_from_schedule_json(d)
    assert [o.id for o in ops] == [0, 2]
    assert sb.heads(ops) == {0: 0, 2: 0}


@pytest.mark.parametrize("elim, latency, work", [(True, 0, {}), (False, 1, {"valu": 1, sb.JOINT: 8})])
def test_eliminable_vbroadcast(elim, latency, work):
    d = _json([("alu", "+", []), ("valu", "vbroadcast", [0]), ("valu", "^", [1])],
              {"enable_vbroadcast_elimination": elim})
    ops = {o.id: o for o in sb.ops_from_schedule_json(d)}
    assert ops[1].work == work
    assert ops[2].deps == ((1, latency),)


def test_run_work_preserves_joint_work_for_an_ambiguous_shift_lane_id():
    d = _json([("valu", "^", []), ("alu", "^", [0])])
    d["ssa_ops"][1]["id"] = 3
    d["ops"] = [
        {"id": 3, "engine": "alu", "opcode": "^", "cycle": 2},
        {"id": 3, "engine": "alu", "opcode": "^", "cycle": 1},
        {"id": 4, "engine": "alu", "opcode": "^", "cycle": 2},
    ]
    d["valu_shifted"] = 1
    # Old greedy output can report a shift without exporting lane cycles. Both id=3 slots
    # could be either the scalar op or a lane of op 0, so their depth is unknown.
    work = {o.id: o.work for o in sb.ops_from_schedule_json(d, "run")}
    assert work == {0: {sb.JOINT: 1}, 3: {}, -1: {sb.JOINT: 2}}

    # A non-matching lane cycle identifies the native scalar slot.
    d["shifted_alu_cycles"] = {"0": [1, 1, 1, 1, 2, 2, 2, 2]}
    work = {o.id: o.work for o in sb.ops_from_schedule_json(d, "run")}
    assert work == {0: {sb.JOINT: 1}, 3: {"alu": 1, sb.JOINT: 1}, -1: {sb.JOINT: 1}}


def test_run_work_keeps_choice_provenance_and_leaves_shared_sub_unattributed():
    d = _json([("valu", "^", []), ("load", "const_choice", []), ("flow", "select_choice", [])])
    d["ops"] = [
        {"id": 1, "engine": "alu", "opcode": "^", "from_const_choice": 1},
        {"id": 2, "engine": "valu", "opcode": "-", "from_select_choice": 2},
        {"id": 1_000_002, "engine": "valu", "opcode": "multiply_add", "from_select_choice": 2},
        {"id": 0xC5E0_0000, "engine": "valu", "opcode": "-"},
    ]
    work = {o.id: o.work for o in sb.ops_from_schedule_json(d, "run")}
    assert work == {
        0: {},
        1: {"alu": 1, sb.JOINT: 1},
        2: {"valu": 2, sb.JOINT: 16},
        -1: {"valu": 1, sb.JOINT: 8},
    }


def test_json_without_ssa_ops_is_rejected():
    with pytest.raises(ValueError, match="ssa_ops"):
        sb.ops_from_schedule_json({"ops": []})


# ---------------------------------------------------------------------------
# Real kernels
# ---------------------------------------------------------------------------
def _build(tmp_path_factory, name, **kwargs):
    path = tmp_path_factory.mktemp(name) / "sched.json"
    args = dict(forest_height=10, n_nodes=2047, batch_size=16, rounds=5, enable_valu_shift=True,
                verbose=False, use_cpsat=False, output_json=str(path))
    args.update(kwargs)
    args["run_options_json"] = json.dumps({"use_cpsat": args["use_cpsat"], "rounds": args["rounds"],
                                           "batch_size": args["batch_size"]})
    scheduler_rs.build_tick_tock_kernel_ssa(**args)
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="module", params=["greedy", "cpsat"])
def kernel(request, tmp_path_factory):
    if request.param == "greedy":
        return _build(tmp_path_factory, "greedy")
    return _build(tmp_path_factory, "cpsat", rounds=2, use_cpsat=True, cpsat_time_limit=20.0)


def test_no_schedule_is_below_the_bound(kernel):
    rep = sr.analyze(kernel)
    s = rep["static_bound"]
    assert s["any"]["bound"] <= s["run"]["bound"] <= kernel["total_cycles"]
    assert s["run"]["bound"] >= rep["floor"]["cycles"]
    assert s["explained_cycles"] + s["gap_cycles"] == rep["gap"]["cycles"]
    if kernel.get("lb_cycles") is not None:
        assert s["any"]["bound"] <= kernel["total_cycles"]


def test_run_rows_count_the_lowered_work(kernel):
    """The run encoding attributes every lowered op to an SSA op: the joint row's total is
    8 * vector ops + native scalar ops, exactly what the emission floor divides."""
    rep = sr.analyze(kernel)
    rows = sb.static_makespan_bound(kernel, "run")["rows"]
    em = rep["emission"]
    assert rows[sb.JOINT]["work"] == 8 * em["vector_ops"] + em["native_scalar_ops"]
    for e in ("load", "store", "flow"):
        assert rows[e]["work"] == rep["engines"][e]["ops"]


def test_report_prints_the_bound_and_its_scope(kernel):
    rep = sr.analyze(kernel)
    text = sr.format_report(rep)
    assert "Static head/tail bound for this emission" in text and sb.SCOPE_NOTE in text
    assert "static head/tail bound" in sr.summary_lines(rep)[1]
    assert "static head/tail bound" in sr.format_compare(rep, rep, "a", "b")


# ---------------------------------------------------------------------------
# Z3 soundness (verification/static_bounds_proof.py)
# ---------------------------------------------------------------------------
pytest.importorskip("z3")
from verification import static_bounds_proof as proof


@pytest.mark.parametrize("name,desc,check", proof.STATIC_BOUND_CHECKS, ids=[c[0] for c in proof.STATIC_BOUND_CHECKS])
def test_static_bound_check(name, desc, check):
    _, result, _, stats = check()
    assert result == ("sat" if name in proof.EXPECT_SAT else "unsat"), (desc, stats)


def test_soundness_check_catches_an_inflated_bound(monkeypatch):
    """SB1 has teeth: one cycle added to every row that uses t > 0 or q > 0 is caught."""
    real = sb.makespan_rows

    def inflated(ops, caps=None):
        r = real(ops, caps)
        r["bound"] += r["t"] > 0 or r["q"] > 0
        return r

    monkeypatch.setattr(sb, "makespan_rows", inflated)
    violations, _ = proof.check_random(60, seed=3)
    assert violations
