"""Lifter, exact memory-alias analysis, rescheduler and sandboxed capture, on hand-made kernels.

The public kernels themselves are third-party code and are not in the repository; these
tests pin the rules (docs/PUBLIC_KERNEL_RESCHEDULE.md) on small kernels whose answers are
known by hand.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import kernel_lift, reschedule_kernel
from scripts.capture_public_kernel import capture, check_bundles
from scripts.kernel_memory_use import memory_use, peak_live_words, scratch_report, scratch_values
from scripts.kernel_lift import check_schedule, condense, heights, lift, load_graph, report, save_graph, scratch_edges, value_dependency_graph
from scripts.reschedule_kernel import emit, greedy
from verification.memory_alias import MemLayout, SymbolicKernel

REPO = Path(__file__).resolve().parent.parent
L = MemLayout()
VAL_P = L.val_p  # 2310: first input value


def _kernel(*cycles):
    """Bundles from lists of (engine, slot) per cycle."""
    out = []
    for cyc in cycles:
        b = {}
        for engine, slot in cyc:
            b.setdefault(engine, []).append(list(slot))
        out.append(b)
    return out


# -----------------------------------------------------------------------------
# Scratch edges
# -----------------------------------------------------------------------------


def test_scratch_raw_war_waw_latencies():
    k = _kernel(
        [("load", ("const", 10, 5))],                # 0: write s10
        [("alu", ("+", 11, 10, 10))],                # 1: read s10 -> RAW 0->1 (lat 1)
        [("load", ("const", 10, 6))],                # 2: overwrite s10 -> WAR 1->2 (lat 0), WAW 0->2 (lat 1)
    )
    sym = SymbolicKernel(k).run()
    e, raw = scratch_edges(sym.ops)
    assert raw == {(0, 1)}
    assert e == {(0, 1): 1, (1, 2): 0, (0, 2): 1}


def test_same_cycle_reader_and_writer_get_latency_zero():
    # A reader and the overwrite of its operand in one cycle: the read sees the old value.
    k = _kernel(
        [("load", ("const", 10, 5))],
        [("alu", ("+", 11, 10, 10)), ("load", ("const", 10, 6))],
    )
    e, _ = scratch_edges(SymbolicKernel(k).run().ops)
    assert e[(1, 2)] == 0 and e[(0, 1)] == 1 and e[(0, 2)] == 1


def test_vector_lanes_are_exact():
    # A scalar write to lane 3 of a vector only conflicts with readers of that lane.
    k = _kernel(
        [("valu", ("vbroadcast", 16, 0))],           # 0: writes s16..s23
        [("alu", ("+", 40, 19, 19)), ("alu", ("+", 41, 20, 20))],  # 1 reads lane 3, 2 reads lane 4
        [("load", ("const", 19, 7))],                # 3: overwrites lane 3 only
    )
    e, _ = scratch_edges(SymbolicKernel(k).run().ops)
    assert (1, 3) in e and (2, 3) not in e


# -----------------------------------------------------------------------------
# Memory aliasing
# -----------------------------------------------------------------------------


def _bit_kernel(store_addr_slots, load_addr_slots):
    """s1 = mem[VAL_P] (an input value), s2 = s1 & 1 (a free bit); then the given address
    computations, a store of s1 to s30 and a load from s31."""
    return _kernel(
        [("load", ("const", 3, VAL_P))],
        [("load", ("load", 1, 3)), ("load", ("const", 4, 1))],
        [("alu", ("&", 2, 1, 4))],
        store_addr_slots,
        load_addr_slots,
        [("store", ("store", 30, 1))],
        [("load", ("load", 50, 31))],
    )


def _mem_edge(k):
    sym = SymbolicKernel(k).run()
    store = next(o.index for o in sym.ops if o.mem == "store")
    load = max(o.index for o in sym.ops if o.mem == "load")
    return sym, (store, load) in sym.mem_edges


def test_constant_vs_bit_address_may_alias():
    # store to VAL_P, load from VAL_P + bit: equal when the bit is 0 -> edge.
    k = _bit_kernel([("load", ("const", 30, VAL_P)), ("load", ("const", 5, VAL_P))],
                    [("alu", ("+", 31, 5, 2))])
    sym, edge = _mem_edge(k)
    assert edge


def test_value_sets_prove_disjoint():
    # store to VAL_P + 2, load from VAL_P + bit in {VAL_P, VAL_P + 1}: never equal.
    k = _bit_kernel([("load", ("const", 30, VAL_P + 2)), ("load", ("const", 5, VAL_P))],
                    [("alu", ("+", 31, 5, 2))])
    sym, edge = _mem_edge(k)
    assert not edge and sym.stats.disjoint_z3 == 0


def test_z3_proves_disjoint_where_value_sets_overlap():
    # store to VAL_P + 1 - bit, load from VAL_P + bit: both in {VAL_P, VAL_P + 1}, but equal
    # only if 2 * bit == 1, which has no solution.
    k = _bit_kernel(
        [("load", ("const", 5, VAL_P + 1)), ("load", ("const", 6, VAL_P))],
        [("alu", ("-", 30, 5, 2)), ("alu", ("+", 31, 6, 2))],
    )
    sym, edge = _mem_edge(k)
    assert not edge and sym.stats.disjoint_z3 == 1


def test_must_alias_load_reads_the_stored_value():
    # Store s1 to an address, load it back from the same address term: the loaded word is
    # the stored term itself, so a later address computed from it is exact.
    k = _bit_kernel([("load", ("const", 5, VAL_P))],
                    [("alu", ("+", 30, 5, 2)), ("alu", ("+", 31, 5, 2))])
    sym = SymbolicKernel(k).run()
    load = max(o.index for o in sym.ops if o.mem == "load")
    store = next(o.index for o in sym.ops if o.mem == "store")
    assert sym.mem_edges[(store, load)] == 1
    assert sym.stats.must_alias == 0  # identical terms are decided without Z3


def test_header_loads_fold_to_constants():
    # mem[4] is forest_values_p = 7 by the preconditions; a store through it hits word 7.
    k = _kernel(
        [("load", ("const", 3, 4))],
        [("load", ("load", 5, 3))],           # s5 = mem[4] = 7
        [("store", ("store", 5, 5))],         # mem[7] = 7
        [("load", ("const", 6, L.forest_p))],
        [("load", ("load", 8, 6))],           # load mem[7]: must alias
    )
    sym = SymbolicKernel(k).run()
    assert (2, 4) in sym.mem_edges


def test_load_then_store_same_cycle_orders_load_first():
    k = _kernel(
        [("load", ("const", 3, VAL_P))],
        [("load", ("load", 1, 3)), ("store", ("store", 3, 3))],
    )
    sym = SymbolicKernel(k).run()
    load = next(o.index for o in sym.ops if o.mem == "load")
    store = next(o.index for o in sym.ops if o.mem == "store")
    assert sym.mem_edges == {(load, store): 0}


# -----------------------------------------------------------------------------
# Graph, report, rescheduling
# -----------------------------------------------------------------------------


def _chain_kernel():
    # Two independent chains issued serially; a scheduler can interleave them.
    cycles = [[("load", ("const", 10, 1))], [("load", ("const", 20, 2))]]
    for _ in range(3):
        cycles.append([("alu", ("+", 10, 10, 10))])
    for _ in range(3):
        cycles.append([("alu", ("+", 20, 20, 20))])
    return _kernel(*cycles)


def test_original_schedule_satisfies_its_graph_and_reports():
    k = _chain_kernel()
    lk = lift(k)
    assert check_schedule(lk, [o.cycle for o in lk.ops]) == []
    r = report(lk, k, "chain")
    assert r["simulator_cycles"] == 8 and r["critical_path"] == 4 and r["dataflow_critical_path"] == 4 and r["ops"] == {"alu": 6, "load": 2}


def test_greedy_and_cpsat_reach_the_critical_path(tmp_path):
    k = _chain_kernel()
    lk = lift(k)
    path = tmp_path / "g.json"
    save_graph(lk, str(path), "chain")
    g = load_graph(str(path))
    assert g.edges == lk.edges
    gs = greedy(g)
    assert check_schedule(g, gs) == [] and max(gs) + 1 == max(heights(g))
    assert len(emit(g, gs)) == 4


@pytest.mark.deep
def test_cpsat_reaches_the_critical_path(tmp_path):
    # Run in a subprocess: the OR-Tools Python wheel and the OR-Tools library that
    # scheduler_rs links cannot share one process (importing both segfaults).
    kernel = tmp_path / "k.json"
    kernel.write_text(json.dumps({"instrs": _chain_kernel()}))
    out = tmp_path / "r.json"
    proc = subprocess.run(
        [sys.executable, "-I", str(REPO / "scripts" / "reschedule_kernel.py"), str(kernel), "--scheduler", "cpsat",
         "--cpsat-deterministic-time", "5", "--workers", "1", "--seeds", "0", "-q", "--json", str(out)],
        capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    r = json.loads(out.read_text())
    assert r["cycles"] == 4 and r["cpsat"]["status"] == "OPTIMAL" and r["greedy_cycles"] == 4


def test_swap_in_one_cycle_is_a_co_issue_group():
    # Op 2 reads s1 and writes s2; op 3 reads s2 and writes s1, in one cycle (a swap). Each
    # must not be later than the other, so they form one group and stay together.
    k = _kernel(
        [("load", ("const", 1, 5)), ("load", ("const", 2, 6))],
        [("alu", ("+", 2, 1, 1)), ("alu", ("+", 1, 2, 2))],
    )
    g = lift(k)
    assert g.edges[(2, 3)] == 0 and g.edges[(3, 2)] == 0
    c = condense(g)
    assert c.group[2] == c.group[3]
    s = greedy(g)
    assert s[2] == s[3] and check_schedule(g, s) == []


def test_cpsat_makespan_includes_terminal_co_issue_groups(tmp_path):
    # Seven independent swaps use 14 scalar slots. Their internal WAR edges are
    # mutual, so no individual operation is a sink. Capacity 12 needs two cycles.
    bundles = []
    for base in range(0, 14, 2):
        if base in (0, 12):
            bundles.append({"alu": []})
        bundles[-1]["alu"].extend([["+", base, base + 1, base + 1], ["+", base + 1, base, base]])
    kernel = tmp_path / "swaps.json"
    kernel.write_text(json.dumps(bundles))
    summary = tmp_path / "summary.json"
    subprocess.run(
        [sys.executable, "-I", str(REPO / "scripts/reschedule_kernel.py"), str(kernel),
         "--workers", "1", "--seeds", "0", "-q", "--json", str(summary)],
        check=True, capture_output=True, timeout=60,
    )
    result = json.loads(summary.read_text())
    assert result["cpsat"]["objective"] == result["cycles"] == 2


def test_rescheduler_rejects_graph_for_another_kernel(tmp_path):
    bundles = _chain_kernel()
    path = tmp_path / "graph.json"
    save_graph(lift(bundles), str(path))
    kernel = tmp_path / "kernel.json"
    command = [sys.executable, "-I", str(REPO / "scripts/reschedule_kernel.py"), str(kernel),
               "--graph", str(path), "--scheduler", "greedy", "--seeds", "0"]
    kernel.write_text(json.dumps(bundles))
    subprocess.run(command, capture_output=True, check=True, timeout=60)
    bundles[0]["load"][0][-1] += 1
    kernel.write_text(json.dumps(bundles))
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 2
    assert "graph does not match" in result.stderr


@pytest.mark.parametrize("rounds", [0, 1])
def test_lift_table_checks_use_kernel_metadata(tmp_path, monkeypatch, capsys, rounds):
    # The batch size in header word 2 is 8, so the store and final load both touch word 8.
    bundles = _kernel(
        [("load", ("const", 3, 2))],
        [("load", ("load", 5, 3))],
        [("store", ("store", 5, 5))],
        [("load", ("const", 6, 8))],
        [("load", ("load", 8, 6))],
    )
    layout = MemLayout(forest_height=3, batch_size=8, rounds=rounds)
    kernel = tmp_path / "kernel.json"
    kernel.write_text(json.dumps(vars(layout) | {"instrs": bundles}))
    monkeypatch.setattr(sys, "argv", ["kernel_lift", f"sample={kernel}", "--table", "--concrete-runs", "1",
                                     "--save-graph-dir", str(tmp_path)])
    assert kernel_lift.main() == int(rounds != 0)
    output = capsys.readouterr()
    assert "| sample |" in output.out and "concrete check" not in output.out
    assert f"sample: concrete check (1 random inputs): {'PASS' if rounds == 0 else 'FAIL'}" in output.err
    graph = load_graph(str(tmp_path / "sample.graph.json"))
    assert graph.layout == layout and graph.memory == {(2, 4): 1}


def test_rescheduler_requires_matching_graph_layout(tmp_path, monkeypatch, capsys):
    bundles = _chain_kernel()
    layout = MemLayout(forest_height=4, batch_size=16, rounds=0)
    path = tmp_path / "graph.json"
    save_graph(lift(bundles, layout), str(path))
    graph_data = json.loads(path.read_text())
    kernel = tmp_path / "kernel.json"
    kernel.write_text(json.dumps(vars(layout) | {"instrs": bundles}))
    monkeypatch.setattr(sys, "argv", ["reschedule_kernel", str(kernel), "--graph", str(path), "--scheduler", "greedy",
                                     "--seeds", "1"])
    assert reschedule_kernel.main() == 0
    assert "on 1 inputs: PASS" in capsys.readouterr().out
    for incompatible in (vars(MemLayout()), None):
        if incompatible is None:
            del graph_data["layout"]
        else:
            graph_data["layout"] = incompatible
        path.write_text(json.dumps(graph_data))
        with pytest.raises(SystemExit, match="2"):
            reschedule_kernel.main()
        assert "graph layout does not match" in capsys.readouterr().err
    # An old graph remains usable for a kernel with the old default layout.
    kernel.write_text(json.dumps(bundles))
    monkeypatch.setattr(sys, "argv", [*sys.argv[:-1], "0"])
    assert reschedule_kernel.main() == 0


def test_rescheduler_reports_skipped_validation_and_keeps_previous_output_on_failure(tmp_path, monkeypatch, capsys):
    kernel = tmp_path / "kernel.json"
    out = tmp_path / "out.json"
    summary = tmp_path / "summary.json"
    layout = {"forest_height": 3, "batch_size": 8, "rounds": 0}
    kernel.write_text(json.dumps(layout | {"instrs": _chain_kernel()}))
    base = ["reschedule_kernel", str(kernel), "--scheduler", "greedy", "--save-kernel", str(out), "--json", str(summary)]
    monkeypatch.setattr(sys, "argv", [*base, "--seeds", "1"])
    assert reschedule_kernel.main() == 0
    assert "on 1 inputs: PASS" in capsys.readouterr().out
    previous = out.read_bytes()

    layout["rounds"] = 1
    kernel.write_text(json.dumps(layout | {"instrs": _chain_kernel()}))
    assert reschedule_kernel.main() == 1
    assert "on 1 inputs: FAIL" in capsys.readouterr().out
    assert out.read_bytes() == previous
    result = json.loads(summary.read_text())
    assert result["validation"] == "failed" and result["valid"] is False

    monkeypatch.setattr(sys, "argv", [*base, "--seeds", "0"])
    assert reschedule_kernel.main() == 0
    assert "validation SKIPPED" in capsys.readouterr().out
    result = json.loads(summary.read_text())
    assert result["validation"] == "skipped" and result["valid"] is None and result["validated_seeds"] == 0
    saved = json.loads(out.read_text())
    assert {k: saved[k] for k in layout} == layout and saved["instrs"]

    monkeypatch.setattr(sys, "argv", [*base, "--seeds", "-1"])
    with pytest.raises(SystemExit, match="2"):
        reschedule_kernel.main()
    assert "--seeds must be nonnegative" in capsys.readouterr().err


def test_valu_shift_emits_eight_alu_lanes():
    k = _kernel([("valu", ("vbroadcast", 16, 0))], [("valu", ("^", 24, 16, 16))])
    g = lift(k)
    out = emit(g, [0, 1], shifted={1})
    assert len(out[1]["alu"]) == 8 and out[1]["alu"][3] == ["^", 27, 19, 19]


# -----------------------------------------------------------------------------
# Sandboxed capture
# -----------------------------------------------------------------------------

BUILDER = '''
from problem import VLEN
class KernelBuilder:
    def __init__(self):
        self.instrs = []
    def build_kernel(self, forest_height, n_nodes, batch_size, rounds):
        self.instrs = [{"load": [("const", 0, VLEN)], "debug": [("comment", "x")]}]
'''


def test_check_bundles_rejects_non_data():
    with pytest.raises(ValueError):
        check_bundles([{"alu": [["+", 1, "2", 3]]}])
    with pytest.raises(ValueError):
        check_bundles([{"gpu": [["+", 1, 2, 3]]}])
    for malformed in ({"load": "a"}, {"load": ["a"]}, {"load": None}):
        with pytest.raises(ValueError):
            check_bundles([malformed])


def test_capture_result_read_is_bounded(tmp_path, monkeypatch):
    from scripts import capture_public_kernel

    result = tmp_path / "result.json"
    bundles = [{"load": [["const", 0, 8]]}]
    result.write_text(json.dumps(bundles))
    assert capture_public_kernel._read_capture_result(str(result)) == bundles

    monkeypatch.setattr(capture_public_kernel, "MAX_CAPTURE_BYTES", 1)
    with pytest.raises(ValueError, match="capture limit"):
        capture_public_kernel._read_capture_result(str(result))


@pytest.mark.skipif(shutil.which("sandbox-exec") is None, reason="needs macOS sandbox-exec")
def test_capture_runs_builder_in_sandbox(tmp_path):
    builder = tmp_path / "builder.py"
    builder.write_text(BUILDER)
    assert capture(builder, timeout=120) == [{"load": [["const", 0, 8]]}]


@pytest.mark.skipif(shutil.which("sandbox-exec") is None, reason="needs macOS sandbox-exec")
def test_capture_sandbox_blocks_writes_and_repo_imports(tmp_path):
    outside = tmp_path / "escaped.txt"
    builder = tmp_path / "builder.py"
    builder.write_text(f"open({str(outside)!r}, 'w').write('x')\n" + BUILDER)
    with pytest.raises(RuntimeError):
        capture(builder, timeout=120)
    assert not outside.exists()
    builder.write_text("open('extra.txt', 'w').write('x')\n" + BUILDER)
    with pytest.raises(RuntimeError):
        capture(builder, timeout=120)
    builder.write_text("import scripts.kernel_lift\n" + BUILDER)
    with pytest.raises(RuntimeError):
        capture(builder, timeout=120)


def test_graph_json_roundtrip_is_plain_data(tmp_path):
    lk = lift(_chain_kernel())
    path = tmp_path / "g.json"
    save_graph(lk, str(path))
    data = json.loads(path.read_text())
    assert set(data) == {"name", "ops", "edges", "memory_edges", "raw", "memory", "layout"}
    g = load_graph(str(path))
    assert g.raw == lk.raw and [o.reads for o in g.ops] == [o.reads for o in lk.ops]


def test_value_dependency_graph_keeps_value_and_memory_edges():
    # WAR 1->2 and WAW 0->2 on s10 disappear; RAW 0->1 stays.
    k = _kernel(
        [("load", ("const", 10, 5))],
        [("alu", ("+", 11, 10, 10))],
        [("load", ("const", 10, 6))],
    )
    g = value_dependency_graph(lift(k))
    assert g.edges == {(0, 1): 1}


def test_memory_use_classifies_a_scatter_then_load():
    # Store an input value to VAL_P + bit (data-dependent, 2 words), load it back.
    k = _bit_kernel([("load", ("const", 5, VAL_P))],
                    [("alu", ("+", 30, 5, 2)), ("alu", ("+", 31, 5, 2))])
    sym = SymbolicKernel(k).run()
    pats = memory_use(sym)["patterns"]
    assert len(pats) == 1
    p = pats[0]
    assert p["address"].startswith("data-dependent (2 bases")
    assert p["value"].startswith("working value") and p["readers"] == ["load"] and p["loads"] == 1
    assert scratch_report(sym)["peak_parked_working_words"] == 1


@pytest.mark.parametrize("base, working", [(VAL_P, True), (L.forest_p, False)])
def test_memory_use_classifies_dynamic_loads_by_their_source_region(base, working):
    k = _kernel(
        [("load", ("const", 3, VAL_P)), ("load", ("const", 4, 1))],
        [("load", ("load", 1, 3)), ("load", ("const", 5, base))],
        [("alu", ("&", 2, 1, 4)), ("load", ("const", 6, VAL_P + 4))],
        [("alu", ("+", 7, 5, 2))],
        [("load", ("load", 8, 7))],
        [("store", ("store", 6, 8))],
        [("load", ("load", 9, 6))],
    )
    sym = SymbolicKernel(k).run()
    value = memory_use(sym)["patterns"][0]["value"]
    assert value.startswith("working value" if working else "function of tree values")
    scratch = scratch_report(sym)
    assert scratch["peak_parked_working_words"] == int(working)
    assert scratch["peak_parked_tree_or_constant_words"] == int(not working)


def test_peak_live_words_counts_value_lifetimes():
    k = _chain_kernel()
    sym = SymbolicKernel(k).run()
    vals = scratch_values(sym.ops)
    # s10 is live from cycle 1 to 4 while s20 waits from cycle 2 to its first read at 5.
    assert peak_live_words(vals, [o.cycle for o in sym.ops])[0] == 2
    r = scratch_report(sym)
    assert r["words_touched"] == 2 and r["peak_live_plus_parked_working"] == 2


def test_sampled_witnesses_agree_with_z3(monkeypatch):
    # Every decision taken from a sampled model must be the one Z3 gives on its own.
    import verification.memory_alias as ma
    kernels = [
        _bit_kernel([("load", ("const", 30, VAL_P)), ("load", ("const", 5, VAL_P))], [("alu", ("+", 31, 5, 2))]),
        _bit_kernel([("load", ("const", 5, VAL_P + 1)), ("load", ("const", 6, VAL_P))],
                    [("alu", ("-", 30, 5, 2)), ("alu", ("+", 31, 6, 2))]),
    ]
    with_samples = [SymbolicKernel(k).run() for k in kernels]
    monkeypatch.setattr(ma, "N_SAMPLES", 0)
    z3_only = [SymbolicKernel(k).run() for k in kernels]
    assert [s.mem_edges for s in with_samples] == [s.mem_edges for s in z3_only]
    assert with_samples[0].stats.sampled_witnesses > 0 and z3_only[0].stats.sampled_witnesses == 0


def test_two_stores_to_one_word_may_share_a_cycle():
    # Memory writes of a cycle commit in slot order, so store -> store has latency 0 and the
    # emitter keeps the later store later in the bundle.
    k = _kernel(
        [("load", ("const", 3, VAL_P)), ("load", ("const", 4, 9))],
        [("store", ("store", 3, 3)), ("store", ("store", 3, 4))],
    )
    sym = SymbolicKernel(k).run()
    assert sym.mem_edges == {(2, 3): 0}
    g = lift(k)
    out = emit(g, [0, 0, 1, 1])
    assert out[1]["store"] == [["store", 3, 3], ["store", 3, 4]]
