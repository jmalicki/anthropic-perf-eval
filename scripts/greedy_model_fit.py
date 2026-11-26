#!/usr/bin/env python3
"""Fit, check and score the greedy pipeline model `es.GREEDY_MODEL` from checked-in data.

    uv run python -m scripts.greedy_model_fit                 # fit, leave-one-out, rankings, out-of-sample
    uv run python -m scripts.greedy_model_fit --check         # exit 1 unless it reproduces GREEDY_MODEL
                                                              # and the numbers below
    uv run python -m scripts.greedy_model_fit --judge stall-0.98 --check   # GREEDY_MODEL_STALL
    uv run python -m scripts.greedy_model_fit collect --state S.json [...] [--measured M.json] --out D.json

`--measured` is a JSON list of build records in the dataset's own format, each with `name`,
`bit_source` ("index" or "path"), `rows`, `counts` (predicted op counts, `es.Table.predict`) and
`cycles`; `collect` refuses a record that lacks one. No script writes it: the records are
assembled from the builds (`emission_structure measure` output has no bit source or counts).

The model (scripts/emission_structure.py, docs/GREEDY_SHAPE_ABLATION.md section 12.3) is

    predicted = max(floor + a_floor,  loads / 2 + a_load + b_flow * max(0, flow - flow0))

with `loads` = load ops + ConstChoices and `flow` = flow ops + every SelectChoice. It is a
calibrated HEURISTIC of the load-tail greedy (shaped flags + LOAD_TAIL_GREEDY), never a bound.

Data: docs/emission_plans/greedy_model_data.json, one record per distinct build: rows, the
predicted op counts (equal to the emitted counts: every build was checked), scheduled and
simulator cycles, the bit source and where it came from. Records with use "loop" are the
PR #74 loop builds (the 28 index-bit and 16 path-bit plans of loop_report_load_tail.json);
"out_of_sample" are plans built after the fit to test it (section 12.3(c)).

Fit: grid search minimising the sum of absolute errors over the index-bit loop builds under
1,200 cycles (24 plans; the four slower ones are uniform flag plans and gather, far from the
region), the grid of GRID. Leave-one-out refits the grid without each plan. Both use the same
plans the model is then scored on, so every ranking computed from them is in-sample; the
leave-one-out numbers cover the constants, not the choice of the model's form, which was made
after looking at these plans. The "near" subsets (cycles within 30 of the best) are selected
on the outcome. The only out-of-sample test is the plans stage 1 chose with the model: there
it is optimistic by 20 to 42 cycles on all four, i.e. it fails.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
sys.path.insert(0, ROOT)

from scripts import emission_structure as es  # noqa: E402
from scripts.emission_loop import pearson, spearman  # noqa: E402

DATA = os.path.join(ROOT, "docs", "emission_plans", "greedy_model_data.json")
# (a_load, flow0, b_flow, a_floor): the grid the constants were chosen from.
GRID = list(itertools.product(range(10, 41), range(800, 921, 5), [x / 100 for x in range(0, 101, 5)],
                              range(20, 81, 4)))
TRAIN_MAX_CYCLES = 1200
NEAR = 30

# Judges: the greedy configuration whose cycles the model predicts. Each build record carries
# its judge ("default" when absent): "default" = shaped flags + LOAD_TAIL_GREEDY (the PR #74
# loop); "stall-0.98" = that + greedy_budget_force=stall, pressure_inflation_threshold=0.98.
JUDGES = {"default": "GREEDY_MODEL", "stall-0.98": "GREEDY_MODEL_STALL"}

# What --check requires per judge (docs/GREEDY_SHAPE_ABLATION.md section 12.3 and
# docs/STARTUP_AND_STALL_LOOP.md section A quote these); the fit must also equal the judge's
# constants in scripts/emission_structure.py.
EXPECTED = {
    "default": {"train_plans": 24, "fit": {"a_load": 19, "flow0": 900, "b_flow_x100": 75, "a_floor": 32},
                "loo_mae": 6.2, "loo_max": 18},
    "stall-0.98": {"train_plans": 41, "fit": {"a_load": 13, "flow0": 900, "b_flow_x100": 80, "a_floor": 52},
                   "loo_mae": 5.6, "loo_max": 18},
}


def features(counts):
    loads = counts["load"] + counts["const"] + counts["const_alu"]
    return {"floor": es.choice_floor(counts)["cycles"], "half_loads": loads / 2,
            "flow": counts["flow"] + counts["select1"] + counts["select2"]}


def predict_grid(f):
    """Predictions of every grid point for one plan's features (array over GRID)."""
    g = predict_grid.arr
    return np.maximum(f["half_loads"] + g[:, 0] + g[:, 2] * np.maximum(0, f["flow"] - g[:, 1]),
                      f["floor"] + g[:, 3])


predict_grid.arr = np.array(GRID, dtype=float)


def predict(f, th):
    a_load, flow0, b_flow, a_floor = th
    return max(f["half_loads"] + a_load + b_flow * max(0, f["flow"] - flow0), f["floor"] + a_floor)


def as_model(th):
    a_load, flow0, b_flow, a_floor = th
    return {"a_load": int(a_load), "flow0": int(flow0), "b_flow_x100": int(round(b_flow * 100)), "a_floor": int(a_floor)}


def load(path=DATA, judge="default"):
    """The builds of one judge, with their model features."""
    if judge not in JUDGES:
        raise ValueError(f"judge {judge!r}; one of {sorted(JUDGES)}")
    with open(path) as f:
        data = json.load(f)
    data["builds"] = [b for b in data["builds"] if b.get("judge", "default") == judge]
    for b in data["builds"]:
        b["features"] = features(b["counts"])
    data["judge"] = judge
    return data


def analyse(data) -> dict:
    builds = data["builds"]
    train = [b for b in builds if b["use"] == "loop" and b["bit_source"] == "index"
             and b["cycles"] < TRAIN_MAX_CYCLES]
    errs = np.stack([np.abs(predict_grid(b["features"]) - b["cycles"]) for b in train], axis=1)
    total = errs.sum(axis=1)
    th = GRID[int(np.argmin(total))]  # first minimum in GRID order, as a sequential search
    loo_err, loo_pred = [], []
    for i, b in enumerate(train):
        th_i = GRID[int(np.argmin(total - errs[:, i]))]
        p = predict(b["features"], th_i)
        loo_pred.append(p)
        loo_err.append(p - b["cycles"])
    out = {"train_plans": len(train), "fit": as_model(th), "fit_tuple": th,
           "loo_mae": round(float(np.mean(np.abs(loo_err))), 1), "loo_max": int(round(max(map(abs, loo_err))))}
    y = [b["cycles"] for b in train]
    near = [i for i, v in enumerate(y) if v <= min(y) + NEAR]
    out["loo_spearman"] = {"all": spearman(loo_pred, y), "near": spearman([loo_pred[i] for i in near],
                                                                          [y[i] for i in near]),
                           "n_all": len(y), "n_near": len(near)}
    ranks = {}
    for src in ("index", "path"):
        loop = [b for b in builds if b["use"] == "loop" and b["bit_source"] == src]
        x = [predict(b["features"], th) for b in loop]
        yy = [b["cycles"] for b in loop]
        nr = [i for i, v in enumerate(yy) if v <= min(yy) + NEAR]
        lp = [i for i, b in enumerate(loop) if not b["name"].startswith("seed")]
        ranks[src] = {"n_all": len(loop), "spearman_all": spearman(x, yy), "pearson_all": pearson(x, yy),
                      "n_loop": len(lp), "spearman_loop": spearman([x[i] for i in lp], [yy[i] for i in lp]),
                      "n_near": len(nr), "spearman_near": spearman([x[i] for i in nr], [yy[i] for i in nr])}
    out["in_sample"] = ranks
    out["out_of_sample"] = [{"name": b["name"], "bit_source": b["bit_source"], "predicted": predict(b["features"], th),
                             "cycles": b["cycles"], "error": predict(b["features"], th) - b["cycles"]}
                            for b in builds if b["use"] == "out_of_sample"]
    summary = {}
    for src in ("index", "path"):
        err = [o["error"] for o in out["out_of_sample"] if o["bit_source"] == src and o["cycles"] < TRAIN_MAX_CYCLES]
        if err:
            summary[src] = {"n": len(err), "mae": round(float(np.mean(np.abs(err))), 1),
                            "mean_error": round(float(np.mean(err)), 1)}
    out["out_of_sample_summary"] = summary  # plans under TRAIN_MAX_CYCLES
    out["judge"] = data.get("judge", "default")
    return out


def check(res) -> list[str]:
    judge = res.get("judge", "default")
    bad = []
    for k, v in EXPECTED[judge].items():
        if res[k] != v:
            bad.append(f"{k}: computed {res[k]}, expected {v}")
    name = JUDGES[judge]
    gm = {k: getattr(es, name)[k] for k in ("a_load", "flow0", "b_flow_x100", "a_floor")}
    if res["fit"] != gm:
        bad.append(f"fit {res['fit']} != es.{name} {gm}")
    return bad


def report(res):
    f = lambda v: "-" if v is None else f"{v:+.2f}"  # noqa: E731
    name = JUDGES[res.get("judge", "default")]
    print(f"judge: {res.get('judge', 'default')}")
    print(f"training plans: {res['train_plans']} (index-bit loop builds under {TRAIN_MAX_CYCLES} cycles)")
    print(f"fit: {res['fit']}  (es.{name} {getattr(es, name)})")
    ls = res["loo_spearman"]
    print(f"leave-one-out (constants refitted per plan; the form was chosen on these plans): MAE {res['loo_mae']}, "
          f"max {res['loo_max']}; Spearman all({ls['n_all']}) {f(ls['all'])}, near({ls['n_near']}) {f(ls['near'])}")
    for src, r in res["in_sample"].items():
        print(f"in-sample {src}: all({r['n_all']}) Spearman {f(r['spearman_all'])} Pearson {f(r['pearson_all'])}; "
              f"loop plans({r['n_loop']}) Spearman {f(r['spearman_loop'])}; "
              f"near({r['n_near']}, selected on cycles) Spearman {f(r['spearman_near'])}")
    print("out of sample (plans built after the fit: default judge, stage 1 chose them with the model; "
          "stall-0.98, the top-k rerun):")
    for o in res["out_of_sample"]:
        print(f"  {o['name']} ({o['bit_source']}): predicted {o['predicted']:.0f}, ran {o['cycles']} "
              f"(error {o['error']:+.0f})")
    for src, s in res.get("out_of_sample_summary", {}).items():
        print(f"out of sample {src}, under {TRAIN_MAX_CYCLES} cycles: {s['n']} plans, MAE {s['mae']}, "
              f"mean error {s['mean_error']:+}")


NOTE = ("Builds behind es.GREEDY_MODEL (judge 'default') and es.GREEDY_MODEL_STALL (judge 'stall-0.98') "
        "(scripts/greedy_model_fit.py). 16 rounds, batch 256, --merge-hash-stages --index-form mirrored, shaped "
        "greedy + LOAD_TAIL_GREEDY (stall-0.98: + greedy_budget_force=stall, pressure_inflation_threshold=0.98); "
        "greedy development measurements, every build PROVED and PASSED.")


# What the fit reads from a measured (out-of-sample) record (`load`, `analyse`), plus its plan rows.
MEASURED_FIELDS = ("name", "bit_source", "rows", "counts", "cycles")


def collect(states, measured, out, judge="default"):
    """Records for one judge from loop state files (use "loop") and measured plans (use
    "out_of_sample"), written into `out`: records of other judges already there are kept,
    those of this judge are replaced."""
    if judge not in JUDGES:
        raise ValueError(f"judge {judge!r}; one of {sorted(JUDGES)}")
    builds, seen = [], set()
    for path in states:
        with open(path) as f:
            st = json.load(f)
        ctx = es.Context(**st["context"])
        for key, b in st["builds"].items():
            if not b.get("valid") or (ctx.select_bit_source, key) in seen:
                continue
            seen.add((ctx.select_bit_source, key))
            builds.append({"name": b["name"], "use": "loop", "bit_source": ctx.select_bit_source,
                           "source": os.path.basename(path), "plan": b["plan"], "rows": b["rows"],
                           "counts": b["counts"], "cycles": b["cycles"], "simulator_cycles": b["simulator_cycles"],
                           "greedy_extra": st.get("greedy_extra"), "judge": judge})
    for path in measured:
        with open(path) as f:
            for i, m in enumerate(json.load(f)):
                missing = [k for k in MEASURED_FIELDS if k not in m]
                if missing:
                    raise ValueError(f"{path}: measured record {m.get('name', f'#{i}')!r} lacks {', '.join(missing)} "
                                     f"(each needs {', '.join(MEASURED_FIELDS)})")
                if m["bit_source"] not in ("index", "path"):
                    raise ValueError(f"{path}: measured record {m['name']!r}: bit_source {m['bit_source']!r} "
                                     "is not index or path")
                builds.append(dict(m, use="out_of_sample", judge=judge))
    if os.path.exists(out):
        with open(out) as f:
            builds = [b for b in json.load(f)["builds"] if b.get("judge", "default") != judge] + builds
    es.write_json_atomic(out, {
        "note": NOTE,
        "builds": builds})
    print(f"{len(builds)} builds written to {out}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", default="fit", choices=["fit", "collect"])
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--judge", default="default", choices=sorted(JUDGES),
                    help="which judge's records to fit (collect: which judge to tag them with)")
    ap.add_argument("--json", help="fit: also write the results here")
    ap.add_argument("--state", action="append", default=[], help="collect: loop state files")
    ap.add_argument("--measured", action="append", default=[], help="collect: JSON lists of out-of-sample builds")
    ap.add_argument("--out", help="collect: dataset to write")
    args = ap.parse_args(argv)
    if args.command == "collect":
        if not args.out:
            ap.error("collect needs --out (it rewrites this judge's records in that file, so the "
                     "checked-in dataset is only written when named)")
        collect(args.state, args.measured, args.out, judge=args.judge)
        return 0
    res = analyse(load(args.data, args.judge))
    report(res)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({k: v for k, v in res.items() if k != "fit_tuple"}, f, indent=1)
    if args.check:
        bad = check(res)
        for b in bad:
            print("CHECK FAILED:", b)
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
