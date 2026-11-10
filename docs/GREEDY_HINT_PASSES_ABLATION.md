# Greedy Hint Passes and Ablation

## Hint passes and lowering when CP-SAT is not run

When CP-SAT is **not** run (greedy-only pipeline), **lowering uses `greedy_decisions`** as the source of encodings. The pipeline does:

1. `GreedySchedulePass` → fills `ctx.greedy_decisions`
2. Optional hint passes (ConstChoice, SelectChoice, VALU shift) → overwrite only the corresponding fields in `ctx.greedy_decisions`
3. Optional `ScheduleWithFixedChoicesPass` → re-runs greedy with those encodings fixed, updates `op.cycle` and `greedy_metrics`; does **not** change `greedy_decisions`
4. `final_decisions = greedy_decisions.clone()`
5. `LowerToOpsPass` reads `ctx.final_decisions` for ConstChoice encoding, SelectChoice encoding, and shifted_ops

So **the hint passes directly drive lowering in greedy-only mode**: ConstChoice → add_imm vs Load, SelectChoice → FMA vs vselect, and which VALU ops are shifted come from the (possibly hint-modified) `greedy_decisions`. When CP-SAT **is** run, it overwrites `final_decisions` from the solver, so lowering uses CP-SAT’s solution; the hints only warm-start CP-SAT.

## Ablation configuration

Use a **fixed config** for hint-pass ablations so results are comparable:

| Parameter         | Value   | Note                                   |
|------------------|---------|----------------------------------------|
| Rounds           | 2       | Fast, dev-friendly                     |
| Batch size       | 24      | Multiple of 8                          |
| CP-SAT time limit| 300 s   | Enough for solver to improve from hints |
| Validation       | On      | Every arm must pass reference validation |

## Pass criteria

**Phase 1 (hint passes only)**  
- Every arm: return code 0, kernel passes validation.  
- Solver status: FEASIBLE or OPTIMAL (no INFEASIBLE/UNKNOWN from bad hints).  
- Cycles: hint-pass arm ≤ baseline (no regression).

**Phase 2 (hint passes + schedule-with-fixed-choices)**  
- All arms: validation pass, return code 0.  
- Solver status: FEASIBLE or OPTIMAL.  
- Cycles: Arm 2 ≤ Arm 1; Arm 3 ≤ Arm 2 (no regression from adding re-schedule).

## Commands

**Phase 1 ablation (no-hint vs hint-passes)**  
```bash
uv run python scripts/run_ablation.py --rounds 2 --batch-size 24 --time-limit 300 \
  --validate \
  --arm "no-hint-passes:" \
  --arm "hint-passes:--hint-passes" \
  --output ablation_phase1_hint_passes.json
```

**Phase 2 ablation (baseline vs hint-only vs hint+reschedule)**  
```bash
uv run python scripts/run_ablation.py --rounds 2 --batch-size 24 --time-limit 300 \
  --validate \
  --arm "baseline:" \
  --arm "hint-passes:--hint-passes" \
  --arm "hint-and-reschedule:--hint-passes --schedule-with-fixed-choices" \
  --output ablation_phase2.json
```

**Optional: fail run if any arm fails validation**  
```bash
uv run python scripts/run_ablation.py ... --validate --fail-on-validation-error
```

## CLI flags

- `--hint-passes` — enable all three hint passes (ConstChoice, SelectChoice, VALU shift).
- `--const-choice-hint-pass`, `--select-choice-hint-pass`, `--valu-shift-hint-pass` — enable individually (all off by default).
- `--schedule-with-fixed-choices` — after hint passes, re-run greedy with encodings fixed to assign new cycles.

See `generate_ssa_kernel.py --help` and `scripts/run_ablation.py --help`.
