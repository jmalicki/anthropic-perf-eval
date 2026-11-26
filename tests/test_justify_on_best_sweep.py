"""The justification sweep of docs/CRITICAL_PATH.md is reproducible: rebuilding every row of
docs/emission_plans/justify_on_best_sweep.json (scripts/justify_on_best_sweep.py; 16-round
greedy builds, each proved and validated) gives the recorded cycles, and failures stay
failures. 62 builds: CI only."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import justify_on_best_sweep as sweep  # noqa: E402


@pytest.mark.ci_only
@pytest.mark.timeout(600)  # 62 builds, about 70 s in CI: above the global 120 s on a slow runner
def test_justify_on_best_sweep_reproduces_the_recorded_rows():
    with open(sweep.DATA) as f:
        recorded = json.load(f)["tables"]
    rebuilt = sweep.run(log=lambda *a, **k: None)
    assert set(rebuilt) == set(recorded)
    for table, rows in rebuilt.items():
        assert [r["row"] for r in rows] == [r["row"] for r in recorded[table]], table
        assert [r["command"] for r in rows] == [r["command"] for r in recorded[table]], table
        assert [sweep.outcome(r) for r in rows] == [sweep.outcome(r) for r in recorded[table]], table


def test_recorded_sweep_holds_the_doc_headline():
    """Cheap (no build): the JSON has the numbers docs/CRITICAL_PATH.md quotes."""
    with open(sweep.DATA) as f:
        tables = json.load(f)["tables"]
    best = {r["row"]: sweep.outcome(r) for r in tables["neighbours"]}
    assert (best["best, path"], best["best, index"]) == (1012, 1033)
    assert (best["justify 0, path"], best["justify 0, index"]) == (1044, 1047)
    assert sum(sweep.outcome(r) == "fail" for r in tables["sweep"]) == 2


def test_compare_reports_a_recorded_table_missing_its_last_row():
    """Cheap (no build): a recorded table with fewer rows than the rebuilt one is a change, not
    silently truncated by zip (review of #93)."""
    with open(sweep.DATA) as f:
        recorded = json.load(f)["tables"]
    assert sweep.compare(recorded, recorded) == []
    table = next(iter(recorded))
    short = dict(recorded, **{table: recorded[table][:-1]})
    changes = sweep.compare(recorded, short)
    assert changes == [f"CHANGED {table}: {len(recorded[table])} rows rebuilt, {len(recorded[table]) - 1} recorded"]
    flipped = dict(recorded, **{table: [dict(recorded[table][0], cycles=-1)] + recorded[table][1:]})
    assert any("recorded -1" in c for c in sweep.compare(recorded, flipped))
