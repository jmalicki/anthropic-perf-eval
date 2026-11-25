# Research notes: why CP-SAT stalls on this scheduling problem (November 2026)

Working notes of the investigation summarised in
[../ENERGETIC_REASONING_FOR_UNIT_TASKS.md](../ENERGETIC_REASONING_FOR_UNIT_TASKS.md). They are
kept as written, because each claim carries a tag saying how it was established (read in the
source, read in a paper in full or in part, abstract only, recollection, or our own inference)
and those tags are the value of the notes. Read a claim's tag before relying on it.

| File | What it covers |
|---|---|
| [cpsat_9_15_internals.md](cpsat_9_15_internals.md) | How OR-Tools CP-SAT 9.15 handles a cumulative over thousands of unit intervals: presolve, propagators, the linear relaxation, cut generators, the bound workers, LNS under interleaved search, what changed in 10.0-rc.1, and eight ranked changes with a test for each |
| [optimal_instruction_scheduling_literature.md](optimal_instruction_scheduling_literature.md) | Exact instruction scheduling in compilers (Malik, McInnes, van Beek; Wilken, Heffernan; Unison; Shobaki): implied constraints, dominance rules, lower bounds, identical sub-graphs, register pressure, and the proof obligations each technique carries for our model |
| [recent_or_survey.md](recent_or_survey.md) | Operations-research and constraint-programming work since about 2015 on this problem class: search in other solvers, propagation for unit-duration cumulatives, time-indexed and decomposition bounds, structure exploitation, hardware scheduling; ranked shortlist |
| [recent_or_parts/](recent_or_parts/) | The four per-topic files behind the survey, with per-paper detail |
| [energetic_tightening_root_cause.md](energetic_tightening_root_cause.md) | Why the previous `--asap-alap-tightening energetic` rule excluded feasible schedules: the rule, the 6 rounds / batch 64 case, the minimal counterexample, and why the existing proofs did not cover it |

Notes on reading them:

- They were written before the fix. Where the root-cause note says "the rule" or cites
  `bounds.rs:1187-1265`, it means `tighten_asap_alap_by_slots`, which has since been removed and
  replaced by `scheduler_rs/src/cpsat_scheduler/resource_bounds.rs`.
- Scripts named `rc/...` and files "in the session scratchpad" were throwaway and are not in the
  repository. The checks that matter were redone as `verification/resource_bounds.py` and the
  tests in `resource_bounds.rs` and `cpsat_scheduler/tests.rs`.
- Several papers could not be fetched (publisher sites refused automated download). Each note
  lists what was and was not read; statements about unread papers are marked as recollection.
- One result in the survey's section 3.2 (a counting time-indexed model) was attempted during
  the investigation with a wrong op-to-variable mapping; the bound it printed was withdrawn and
  appears nowhere in these notes.
