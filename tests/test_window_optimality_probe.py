"""Tests for the window optimality probe script's summary (no solver needed)."""

import pytest

from scripts.window_optimality_probe import markdown, merge_reports, phase_of, summarize


def entry(phase, w, start, saved, status="optimal", encodings="free", bound=None, secs=1.0, shift=0):
    return {
        "phase": phase, "w": w, "start": start, "encodings": encodings, "status": status,
        "reference_makespan": 100, "makespan": 100 - saved, "saved": saved, "lb": None,
        "saved_bound": saved if bound is None else bound, "wall_secs": secs,
        "before": 0, "inside": 10, "after": 0,
        "shift_changes": shift, "select_changes": 0, "const_changes": 0,
    }


def test_phase_of_splits_schedule_into_start_tail_and_steady():
    assert phase_of(0, 10, 1000) == "start-up"
    assert phase_of(500, 10, 1000) == "steady"
    assert phase_of(950, 50, 1000) == "tail"
    assert phase_of(0, 1000, 1000) == "whole"


def test_summarize_counts_improvements_bounds_and_encoding_changes():
    report = {
        "base_makespan": 100,
        "entries": [
            entry("sweep", 20, 0, 2, shift=1),
            entry("sweep", 20, 10, 0, status="feasible", bound=3),
            entry("sweep", 20, 90, 1),
            entry("sweep", 20, 0, 0, encodings="fixed"),
            entry("rolling", 20, 0, 2),
            entry("rolling", 20, 18, 1),
        ],
        "rolling_final": [[20, 97]],
    }
    s = summarize(report)
    free = next(r for r in s["sweep"] if r["encodings"] == "free")
    assert free["windows"] == 3
    assert free["optimal"] == 2
    assert free["improved"] == 2
    assert free["max_saved"] == 2
    assert free["open_gap_windows"] == 1
    assert free["max_saved_bound"] == 3
    assert free["improved_by_phase"] == {"start-up": 1, "tail": 1}
    assert free["encoding_change_windows"] == 1
    fixed = next(r for r in s["sweep"] if r["encodings"] == "fixed")
    assert fixed["improved"] == 0
    assert s["rolling"] == [{"w": 20, "windows": 2, "improved": 2, "optimal": 2, "final": 97,
                             "saved": 3, "complete": True, "reached_cycle": 38,
                             "full_model_check": None, "median_secs": 1.0}]
    text = markdown(s)
    assert "| 20 | free | 3 | 2 | 2 | 2 | 3 | 1 |" in text
    assert "| 20 | 2 | 2 | 2 | 97 | 3 | yes | - |" in text


def test_summarize_reports_an_interrupted_rolling_pass():
    report = {
        "base_makespan": 100,
        "entries": [entry("rolling", 20, 0, 1), {**entry("rolling", 20, 19, 1), "makespan": 98}],
        "rolling_final": [],
    }
    (row,) = summarize(report)["rolling"]
    assert row["final"] == 98 and row["saved"] == 2
    assert row["complete"] is False and row["reached_cycle"] == 39
    assert "no (reached cycle 39)" in markdown(summarize(report))


def test_merge_reports_requires_one_reference():
    a = {"base_makespan": 100, "entries": [entry("sweep", 20, 0, 1)], "rolling_final": []}
    b = {"base_makespan": 100, "entries": [entry("sweep", 50, 0, 2)], "rolling_final": [[50, 98]]}
    merged = merge_reports([a, b])
    assert len(merged["entries"]) == 2
    assert merged["rolling_final"] == [[50, 98]]
    with pytest.raises(ValueError):
        merge_reports([a, {**b, "base_makespan": 99}])
