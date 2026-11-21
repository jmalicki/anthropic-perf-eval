"""scripts/run_cpsat_ablation.py: instance and configuration parsing, log analysis, tables."""

import argparse
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts import run_cpsat_ablation as ablation  # noqa: E402

SOLVE_LOG = """\
Starting CP-SAT solver v9.15.9999
Parameters: max_deterministic_time: 0.1 num_workers: 1
#1       0.10s best:999 next:[1,2] section_solve
CpSolverResponse summary:
status: OPTIMAL
walltime: 0.2
deterministic_time: 0.05

Starting CP-SAT solver v9.15.9999
Parameters: max_deterministic_time: 20 interleave_search: true num_workers: 8
The solution hint is complete and is feasible.
#Bound   1.05s best:inf   next:[1.6300251e+09,6.09015673e+09] initial_domain
#1       1.05s best:1.94013358e+09 next:[1.6300251e+09,1.93013358e+09] complete_hint
10 full problem subsolvers: [core, default_lp, fixed, max_lp, max_lp_sym, no_lp, pseudo_costs, quick_restart]
#Bound   2.43s best:1.94013358e+09 next:[1.63869177e+09,1.93013358e+09] max_lp_sym
#2       2.77s best:1.93013351e+09 next:[1.6400251e+09,1.93013351e+09] no_lp [hint]
#3       5.88s best:1.93013167e+09 next:[1.64003054e+09,1.93013167e+09] rnd_var_lns (d=5.00e-01)
#Bound  40.00s best:1.93013167e+09 next:[1.64003054e+09,1.93013167e+09] reduced_costs
CpSolverResponse summary:
status: FEASIBLE
objective: 1930131671
best_bound: 1640030542
conflicts: 0
branches: 8898
walltime: 80.0
usertime: 80.0
deterministic_time: 60.0

  Objective: scale=10000000, so makespan is leading digits (floor(1930131671/10000000)= 193), tie-breaker in lower digits
SSA CP-SAT result: 193 cycles (greedy was 194), 180 VALU ops shifted
    PROVED - every store matches the SSA program for all inputs
  PASSED - kernel produces correct output
"""


def test_parse_instance():
    assert ablation.parse_instance("4x64", 20.0) == {"rounds": 4, "batch_size": 64, "budget": 20.0, "name": "4r_b64"}
    assert ablation.parse_instance("16x256@60", 20.0)["budget"] == 60.0
    for bad in ["4", "4x12", "x64", "4x0", "4x64@"]:
        with pytest.raises(argparse.ArgumentTypeError):
            ablation.parse_instance(bad, 20.0)


def test_only_parallel_search_without_interleave_is_nondeterministic():
    assert not ablation.is_nondeterministic("")
    assert not ablation.is_nondeterministic("--cpsat-workers 16")
    assert ablation.is_nondeterministic("--no-cpsat-interleave")
    assert ablation.is_nondeterministic("--no-cpsat-interleave --cpsat-workers 12")
    assert not ablation.is_nondeterministic("--no-cpsat-interleave --cpsat-workers 1")


def test_log_analysis_uses_the_last_solve(tmp_path):
    result = ablation.analyse_run(SOLVE_LOG, tmp_path / "missing.json", 0, 81.0)
    assert result["status"] == "FEASIBLE"
    assert result["solver_walltime"] == 80.0 and result["deterministic_time"] == 60.0
    assert result["objective_scale"] == 10_000_000
    # Makespan 194 at the hint, 193 from the second solution; the third only improves the tie-breaker.
    assert result["incumbents"] == [(1.05, 194), (2.77, 193)]
    assert result["last_bound_improvement_wall"] == 40.0
    assert result["full_subsolvers"][:2] == ["core", "default_lp"]
    assert result["greedy_cycles"] == 194
    assert result["proved"] and result["passed"] and result["hint_complete"]
    assert not result["wall_clock_limit_hit"]
    assert "cycles" not in result, "no schedule JSON: no cycle count is invented from the log"


def test_makespan_only_log_has_scale_one():
    log = SOLVE_LOG.split("\n\n")[1].replace("best:1.94013358e+09", "best:194").replace(
        "best:1.93013351e+09", "best:193").replace("best:1.93013167e+09", "best:193")
    assert ablation.parse_incumbents(ablation.last_solve(log), 1) == [(1.05, 194), (2.77, 193)]


def test_wall_clock_warning_and_failed_validation_are_flagged(tmp_path):
    log = SOLVE_LOG.replace("  PASSED - kernel produces correct output",
                            "  FAILED - kernel produces incorrect output")
    log += "  WARNING: CP-SAT stopped on the wall-clock limit (10s) after 3.0 of 20 deterministic time\n"
    result = ablation.analyse_run(log, tmp_path / "missing.json", 1, 12.0)
    assert result["wall_clock_limit_hit"] and not result["passed"]


def _cell(config, cycles, repeat=0, deterministic=True, **extra):
    cell = {"instance": "2r_b256", "config": config, "options": "", "repeat": repeat, "deterministic": deterministic,
            "cycles": cycles, "status": "FEASIBLE", "lb_cycles": 164, "capacity_floor": 163, "greedy_cycles": 194,
            "final_incumbent_dtime_estimate": 2.1, "final_incumbent_wall": 2.8, "deterministic_time": 60.0,
            "solver_walltime": 80.0, "peak_live_words": 1129, "proved": True, "passed": True,
            "wall_clock_limit_hit": False}
    cell.update(extra)
    return cell


def test_markdown_table_reports_spread_and_invalid_runs():
    instances = [ablation.parse_instance("2x256", 20.0)]
    configs = [("baseline", ""), ("fast", "--no-cpsat-interleave"), ("broken", "--x")]
    results = [
        _cell("baseline", 193),
        _cell("fast", 190, 0, False), _cell("fast", 192, 1, False), _cell("fast", 191, 2, False),
        _cell("broken", 180, passed=False),
    ]
    table = ablation.markdown(results, instances, configs)
    assert "### 2 rounds, batch 256 (budget 20 deterministic time; greedy hint 194 cycles)" in table
    rows = {line.split("|")[1].strip(): [c.strip() for c in line.split("|")] for line in table.splitlines()
            if line.startswith("| ") and "Configuration" not in line}
    assert rows["baseline"][2] == "193" and rows["baseline"][3] == "0"
    assert rows["baseline"][-2] == "PROVED, PASSED"
    fast = rows["fast (not reproducible)"]
    assert fast[2] == "190 / 191 / 192 (min / median / max of 3 runs)"
    assert fast[3] == "-3..-1"
    assert rows["broken"][-2].startswith("NO")
    assert "- `fast`: `--no-cpsat-interleave`" in table


def test_standard_configs_are_valid_cli_options():
    """Every standard configuration must parse with the generator's own argument parser."""
    import shlex

    import kernel_build_args

    parser = argparse.ArgumentParser()
    kernel_build_args.add_kernel_build_args(parser)
    names = [name for name, _ in ablation.STANDARD_CONFIGS]
    assert names[0] == "baseline" and ablation.STANDARD_CONFIGS[0][1] == ""
    assert len(set(names)) == len(names)
    for name, options in ablation.STANDARD_CONFIGS:
        args = parser.parse_args(shlex.split(options))
        kwargs = kernel_build_args.args_to_build_kwargs(args, use_git_sha=False)
        json.loads(kwargs["run_options_json"])  # tuning accepted by Rust


def test_end_to_end_small_instance(tmp_path):
    """Two configurations on a tiny instance: results are saved, reused, and tabulated."""
    argv = ["--instances", "2x16@0.3", "--config", "baseline:", "--config", "makespan:--cpsat-objective makespan",
            "--out-dir", str(tmp_path), "--time-limit", "100"]
    assert ablation.main(argv) == 0
    with open(tmp_path / "cpsat_ablation.json") as f:
        data = json.load(f)
    by_config = {r["config"]: r for r in data["results"]}
    assert set(by_config) == {"baseline", "makespan"}
    for result in by_config.values():
        assert result["proved"] and result["passed"] and result["cycles"] > 0
        assert result["lb_cycles"] <= result["cycles"]
    assert by_config["makespan"]["cpsat_tuning"]["objective"] == "makespan"
    assert by_config["makespan"]["objective_scale"] == 1
    table = (tmp_path / "cpsat_ablation.md").read_text()
    assert "| baseline |" in table and "| makespan |" in table
    # Second invocation reuses the saved cells (no solve: the logs are not rewritten).
    stamp = (tmp_path / "2r_b16__baseline.log").stat().st_mtime_ns
    assert ablation.main(argv) == 0
    assert (tmp_path / "2r_b16__baseline.log").stat().st_mtime_ns == stamp
    # A different scheduler_rs build invalidates the saved cells.
    real = ablation.extension_fingerprint
    try:
        ablation.extension_fingerprint = lambda: "another-build"
        assert ablation.main(argv[:2] + ["--config", "baseline:"] + argv[6:]) == 0
    finally:
        ablation.extension_fingerprint = real
    assert (tmp_path / "2r_b16__baseline.log").stat().st_mtime_ns != stamp
    assert ablation.main(argv + ["--report-only"]) == 0
