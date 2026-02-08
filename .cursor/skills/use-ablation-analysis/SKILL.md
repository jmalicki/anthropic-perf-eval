---
name: use-ablation-analysis
description: Run ablation studies comparing ASAP/ALAP tightening configurations (baseline, slot-aware, scratch-aware, both) and analyze results. Use when measuring the impact of domain tightening on CP-SAT solve time, makespan, and solver statistics.
---

# Using the Ablation Analysis Tools

Two scripts work together for ASAP/ALAP tightening ablation studies:

1. **run_asap_alap_ablation.py** — Run a **4-arm ablation study** (baseline, slot-only, scratch-only, slot+scratch).
2. **analyze_ablation_results.py** — **Analyze** saved results: makespan, solve time, solver stats, liveness, VALU shift, slot usage.

Run both from **project root**. Ensure **scheduler_rs** is built (`./scripts/build_scheduler_rs.sh`).

---

## 1. Run an ablation study

**Script:** `scripts/run_asap_alap_ablation.py`

Runs **4 sequential CP-SAT experiments** with the same rounds/batch/time-limit, varying tightening flags:

| Arm | `--enable-asap-alap-tightening` | `--enable-scratch-tightening` |
|-----|:-------------------------------:|:-----------------------------:|
| Baseline | off | off |
| Slot-aware | **on** | off |
| Scratch-aware | off | **on** |
| Slot + scratch | **on** | **on** |

Each arm invokes `generate_ssa_kernel.py` with `--output-json` and streams full solver logs to `ablation_log_*.txt`.

### Usage

```bash
# Default: 4 rounds, batch 64, 600s per arm (~40 min total)
uv run python scripts/run_asap_alap_ablation.py

# Quick dev run (2 rounds, small batch, 120s per arm)
uv run python scripts/run_asap_alap_ablation.py --rounds 2 --batch-size 64 --time-limit 120

# Longer study (4 rounds, 1800s per arm for better convergence)
uv run python scripts/run_asap_alap_ablation.py --rounds 4 --batch-size 64 --time-limit 1800

# Custom output file
uv run python scripts/run_asap_alap_ablation.py --output my_ablation.json
```

### Options

- `--rounds N` — Number of rounds (default: 4).
- `--batch-size N` — Batch size, must be multiple of 8 (default: 64).
- `--time-limit SECS` — CP-SAT time limit per arm (default: 600).
- `--output PATH` — Output JSON file (default: `ablation_results.json`).

### Output files

- **`ablation_results.json`** — Structured results for all 4 arms (makespan, timing, solver status, embedded scheduler JSON).
- **`scheduler_output_*.json`** — Per-arm scheduler output JSON (for slot/liveness analysis).
- **`ablation_log_*.txt`** — Full CP-SAT solver logs per arm (for post-hoc analysis).

---

## 2. Analyze ablation results

**Script:** `scripts/analyze_ablation_results.py`

Reads `ablation_results.json` (or a custom path) and prints a detailed comparison.

### Usage

```bash
# Analyze default results file
uv run python scripts/analyze_ablation_results.py

# Analyze a specific results file
uv run python scripts/analyze_ablation_results.py ablation_results_3r_64b_1800s.json
```

### What it reports

For **each arm**:
- Makespan, peak liveness, VALU shifted, budget exceeded
- Lower bound cycles and optimality gap
- Liveness curve analysis (peak, average, min, peak cycle)
- Usage accounting by round and by variable type
- VALU shift breakdown by operation type
- Solver statistics (time, branches, conflicts, restarts)
- Slot usage per engine (ALU, VALU, Load, Store, Flow)

**Comparison** (baseline vs tightening):
- Makespan difference (cycles and %)
- Solve time difference
- Solver statistics (branches, conflicts)

---

## 3. Interpreting results

### Makespan

- **Lower is better.** Compare each arm to baseline.
- If an arm **hits the time limit** (solver status FEASIBLE, not OPTIMAL), the makespan may improve with more time.
- Arms returning **UNKNOWN** or **INFEASIBLE** mean the solver failed — check the log file.

### Solve time & solver statistics

- **Time to feasibility** — How fast does the solver find *any* valid schedule? Lower = tighter domains help propagation.
- **Branches / conflicts** — Fewer = tighter domains pruned search space effectively.
- **Optimality gap** — How far from proven optimum. 0% = proven optimal.

### Liveness & slot usage

- Compare **peak liveness** across arms — tightening shouldn't increase it significantly.
- Compare **slot utilization** — tightening may change how the solver packs operations.
- **VALU shift count** — tighter domains may change which ops the solver decides to shift.

### Failure cases

| Solver status | Meaning | Action |
|---------------|---------|--------|
| OPTIMAL | Proven optimal within time limit | Best result |
| FEASIBLE | Valid schedule found, not proven optimal | Increase time limit |
| UNKNOWN | No solution found within time limit | Increase time limit or check model |
| INFEASIBLE | Model proven to have no solution | Bug in tightening constraints |

If **baseline passes but tightening arm is INFEASIBLE**, the tightening over-pruned domains — this is a bug in bounds computation.

---

## 4. Existing results

Previous ablation studies are documented in:

- `docs/ABLATION_ANALYSIS.md` — 3-round, batch 64 study
- `docs/ABLATION_3R_64B_1800S.md` — 3-round, batch 64, 1800s study
- `docs/ABLATION_4ARM_2R_64B_600S.md` — 4-arm, 2-round, batch 64, 600s study
- `docs/ABLATION_STUDY_RESULTS.md` — General results summary
- `ablation_results.json` / `ablation_results_3r_64b_1800s.json` — Raw JSON results

---

## Workflow summary

| Goal | Script | Typical use |
|------|--------|-------------|
| Run full 4-arm ablation study | run_asap_alap_ablation.py | Compare baseline vs slot vs scratch vs both |
| Analyze saved results | analyze_ablation_results.py | After a run completes; compare arms |
| Quick dev check | run_asap_alap_ablation.py --rounds 2 --time-limit 120 | Fast iteration on tightening changes |
| Production ablation | run_asap_alap_ablation.py --rounds 4 --time-limit 1800 | Thorough study for documentation |

After running an ablation, use the per-arm `scheduler_output_*.json` files with other analysis tools (slot decile, bulk-move, what-if shift) for deeper investigation.
