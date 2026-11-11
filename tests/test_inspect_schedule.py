"""Tests for scripts/inspect_schedule.py (lane-aware schedule inspection).

Builds small kernels through the PyO3 binding and checks each subcommand against facts read
directly from the scheduler JSON, including the case the old scripts got wrong: asking about
an address in the middle of a vector.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scheduler_rs  # noqa: E402
from scripts import inspect_schedule as ins  # noqa: E402


def _build(tmp_path_factory, name, **kwargs):
    path = tmp_path_factory.mktemp(name) / "sched.json"
    args = dict(forest_height=10, n_nodes=2047, batch_size=16, rounds=5, enable_valu_shift=True,
                verbose=False, use_cpsat=False, output_json=str(path))
    args.update(kwargs)
    instrs, _ = scheduler_rs.build_tick_tock_kernel_ssa(**args)
    with open(path) as f:
        sched = json.load(f)
    kernel = {"instrs": json.loads(json.dumps(list(instrs)))}
    return ins.Schedule(sched, kernel), sched, str(path)


@pytest.fixture(scope="module")
def greedy(tmp_path_factory):
    """5 rounds: shifted ops, selects and one gather round (vector groups)."""
    return _build(tmp_path_factory, "greedy")


@pytest.fixture(scope="module")
def fma(tmp_path_factory):
    """Selects forced onto the vector ALU: subtract + multiply_add lowering."""
    return _build(tmp_path_factory, "fma", rounds=3, use_cpsat=True, cpsat_time_limit=30.0,
                  select_choice="fma")


def _vector_with_definer(sched):
    for v in sched["vregs"]:
        if v["width"] == "vector" and v["address"] is not None and v["interval_end"] > v["interval_start"] + 1:
            name = f"v{v['id']}_v"
            if any(op.get("dest") == name for op in sched["ssa_ops"]):
                return v, name
    raise AssertionError("no vector value with a defining op")


def test_addr_finds_vector_by_inner_lane(greedy):
    s, sched, _ = greedy
    v, name = _vector_with_definer(sched)
    definer = next(op for op in sched["ssa_ops"] if op.get("dest") == name)
    inner = v["address"] + 3  # not the base address: the old scripts printed "No VReg at address"
    cycle = v["interval_start"] + 1
    text = "\n".join(ins.cmd_addr(s, inner, cycle))
    line = next(l for l in text.splitlines() if l.strip().startswith(name + " "))
    assert f"addresses {v['address']}..{v['address'] + 7}" in line
    assert f"lane 3 is address {inner}" in line
    assert f"interval [{v['interval_start']}, {v['interval_end']}]" in line
    assert f"holds address {inner} for reads in cycle {cycle}" in line
    assert f"defined by SSA op {definer['id']} {definer['opcode']} at cycle {definer['cycle']}" in text
    # Exactly one non-lane value can hold an address at a time.
    holders = [l for l in text.splitlines() if "<== holds" in l and "of vector group" not in l]
    assert len(holders) == 1, holders
    # The last write before `cycle` is the definer's lowered op.
    after = text.split(f"Last write to address {inner} before cycle {cycle}:")[1]
    assert f"SSA op {definer['id']}" in after.splitlines()[1]


def test_addr_without_cycle_and_unknown_address(greedy):
    s, sched, _ = greedy
    v, name = _vector_with_definer(sched)
    lines = ins.cmd_addr(s, v["address"] + 7, None)
    assert any(l.strip().startswith(name + " ") for l in lines)
    assert not any("<==" in l for l in lines)
    assert ins.cmd_addr(s, 10**6, None) == ["No value's address range covers address 1000000."]


def test_addr_vector_group_lanes(greedy):
    """Gather lanes have no entry in vregs; they are found through their group's address."""
    s, sched, _ = greedy
    groups = sched["vector_group_lanes"]
    assert groups, "5 rounds must contain a gather round"
    by_id = {v["id"]: v for v in sched["vregs"]}
    gid, lanes = next((int(g), l) for g, l in groups.items()
                      if not any(op.get("dest") == f"v{g}_v" for op in sched["ssa_ops"]))
    base = by_id[gid]["address"]
    lane_op = next(op for op in sched["ssa_ops"] if op.get("dest") == f"v{lanes[5]}_s")
    text = "\n".join(ins.cmd_addr(s, base + 5, None))
    assert f"v{lanes[5]}_s  address {base + 5}" in text
    assert f"lane 5 of vector group v{gid}_v" in text
    assert f"defined by SSA op {lane_op['id']} load" in text
    assert "vector group assembled from 8 scalar lanes" in text
    # The op that reads the assembled vector is a consumer of the lane's load.
    reader = next(op for op in sched["ssa_ops"]
                  if any(isinstance(x, dict) and x.get("VReg", {}).get("id") == gid for x in op["srcs"]))
    assert f"SSA op {reader['id']} " in "\n".join(ins.cmd_op(s, lane_op["id"])).split("Consumers")[1]


def test_op_shows_operands_lowered_and_consumers(greedy):
    s, sched, _ = greedy
    op = next(o for o in sched["ssa_ops"] if o["engine"] == "valu" and o["opcode"] == "^" and len(o["dep_ids"]) == 2)
    text = "\n".join(ins.cmd_op(s, op["id"]))
    assert text.startswith(f"SSA op {op['id']}: ^ on valu, cycle {op['cycle']}")
    for src in op["srcs"]:
        assert f"v{src['VReg']['id']}_v" in text.split("Operands:")[1].split("Lowered ops")[0]
    for dep in op["dep_ids"]:
        assert f"defined by SSA op {dep} " in text
    consumers = [o["id"] for o in sched["ssa_ops"] if op["id"] in o["dep_ids"]]
    tail = text.split("Consumers")[1]
    assert tail.startswith(f" ({len(consumers)}):")
    for c in consumers:
        assert f"SSA op {c} " in tail
    assert ins.cmd_op(s, 10**9)[0].startswith("No SSA op with id")


def test_op_shifted_lists_eight_scalar_lanes(greedy):
    s, sched, _ = greedy
    assert sched["valu_shifted"] > 0
    lowered_ids = {o["id"] for o in sched["ops"]}
    shifted = next(o for o in sched["ssa_ops"] if o["engine"] == "valu" and o["id"] * 1000 in lowered_ids
                   and o["id"] not in lowered_ids)
    lines = ins.cmd_op(s, shifted["id"])
    lanes = [l for l in lines if "vector op shifted to the scalar ALU" in l]
    assert len(lanes) == 8
    assert sorted(int(l.split("[lane ")[1].split(" ")[0]) for l in lanes) == list(range(8))
    assert all(" alu [" in l for l in lanes)
    assert "Lowered ops (8):" in lines

    # slot: a lane's destination address maps back to the vector op and names the lane.
    low = next(o for o in sched["ops"] if o["id"] == shifted["id"] * 1000 + 6)
    text = "\n".join(ins.cmd_slot(s, low["cycle"], int(low["slot"][1])))
    assert f"belongs to: SSA op {shifted['id']} {shifted['opcode']}" in text
    assert "lane 6 of 8" in text
    assert "kernel: bundle contains this slot" in text


def test_slot_maps_every_lowered_op_to_an_ssa_op(greedy):
    """Each scratch-writing slot of the kernel is attributed to exactly the op that made it."""
    s, sched, _ = greedy
    checked = 0
    for i, low in enumerate(s.lowered):
        writes, _ = ins.slot_effects(low["engine"], low["slot"])
        if not writes:
            continue
        ssa_id, _role = s.owner[i]
        assert ssa_id is not None, low
        dest = s.values[s.ssa[ssa_id]["dest"]]
        lo, n = writes[0]
        assert dest.addr <= lo and lo + n <= dest.addr + dest.size, (low, dest)
        text = "\n".join(ins.cmd_slot(s, low["cycle"], lo + n - 1))  # last word, not the base
        assert f"belongs to: SSA op {ssa_id} " in text
        assert "WARNING" not in text
        checked += 1
    assert checked > 300
    assert ins.cmd_slot(s, 0, 10**6)[0].startswith("No slot in schedule cycle 0")


def test_bundle_flag_is_schedule_cycle_plus_one(greedy, capsys):
    s, sched, path = greedy
    low = next(o for o in sched["ops"] if o["engine"] == "valu")
    dest = low["slot"][1]
    assert ins.main([path, "slot", str(low["cycle"]), dest]) == 0
    by_cycle = capsys.readouterr().out
    assert ins.main([path, "slot", str(low["cycle"] + 1), dest, "--bundle"]) == 0
    assert capsys.readouterr().out == by_cycle
    assert f"schedule cycle {low['cycle']} (kernel bundle {low['cycle'] + 1})" in by_cycle
    # The kernel really has that slot at bundle cycle + 1.
    bundle = s.instrs[low["cycle"] + 1]
    assert [low["slot"][0]] + [int(x) for x in low["slot"][1:]] in bundle["valu"]


def test_select_lowered_to_subtract_and_multiply_add(fma):
    s, sched, _ = fma
    fmas = [o for o in sched["ops"] if o["opcode"] == "multiply_add" and o.get("from_select_choice") is not None]
    assert fmas, "select_choice=fma must lower selects to multiply_add"
    with_subtract = precomputed = 0
    for low in fmas:
        parent = low["from_select_choice"]
        lines = ins.cmd_op(s, parent)
        lowered = "\n".join(lines).split("Lowered ops")[1].split("Consumers")[0]
        assert "select as multiply_add" in lowered
        fma_line = next(l for l in lowered.splitlines() if "select as multiply_add" in l)
        assert f"cycle {low['cycle']} " in fma_line
        if "select subtract: true - false" in lowered:
            with_subtract += 1
            sub_line = next(l for l in lowered.splitlines() if "select subtract" in l)
            sub_cycle = int(sub_line.split("cycle ")[1].split(" ")[0])
            assert sub_cycle < low["cycle"]
            # The multiply_add reads the subtract's result (operand 3 = true - false).
            sub_dest = int(sub_line.split("['-', ")[1].split(",")[0])
            assert int(low["slot"][3]) == sub_dest
        else:
            # (true - false) was emitted as an SSA value of its own; the op names it.
            precomputed += 1
            note = next(l for l in lowered.splitlines() if "no subtract: (true - false) is precomputed in v" in l)
            delta = s.find_value(note.split("precomputed in ")[1].split(",")[0])
            assert delta.addr == int(low["slot"][3])
        # slot on the multiply_add's last result word names the select.
        text = "\n".join(ins.cmd_slot(s, low["cycle"], int(low["slot"][1]) + 7))
        assert f"belongs to: SSA op {parent} select_choice" in text

    assert with_subtract > 0 and precomputed > 0, (with_subtract, precomputed)

    shared = [(i, o) for i, o in enumerate(s.lowered) if o["id"] >= ins.CSE_SHARED_SUB_PREFIX]
    for i, low in shared:
        users = s.shared_subtract_users(i)
        text = "\n".join(ins.cmd_slot(s, low["cycle"], int(low["slot"][1])))
        assert "shared select subtract of CSE group" in text
        for user in users:
            assert f"feeds the multiply_add of SSA op {user} " in text


def test_value_subcommand(greedy):
    s, sched, _ = greedy
    v, name = _vector_with_definer(sched)
    readers = [o["id"] for o in sched["ssa_ops"]
               if any(isinstance(x, dict) and x.get("VReg", {}).get("id") == v["id"] for x in o["srcs"])]
    text = "\n".join(ins.cmd_value(s, str(v["id"])))
    assert text.startswith(f"{name}  addresses {v['address']}..{v['address'] + 7}")
    assert f"Readers ({len(set(readers))}):" in text
    assert ins.cmd_value(s, "v99999999_v")[0].startswith("No value named")
