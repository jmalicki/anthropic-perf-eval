# CP-SAT Subsolver Parameter Evidence and Rationale

## Summary

After researching academic literature, OR-Tools documentation, and the CP-SAT Primer, here's what the evidence says about tuning CP-SAT subsolver parameters for our constraint structure:

## Key Findings

### 1. LNS Tuning (Moderate Evidence)

**What we know:**
- CP-SAT uses LNS as part of its portfolio of incomplete subsolvers for finding primal solutions [[CP-SAT Primer LNS](https://d-krupke.github.io/cpsat-primer/09_lns.html)]
- The portfolio approach with "continuous sharing of information between workers has produced massive speedups" [[Perron et al. 2023](https://drops.dagstuhl.de/storage/00lipics/lipics-vol280-cp2023/LIPIcs.CP.2023.3/LIPIcs.CP.2023.3.pdf)]
- `use_lns_only` exists but "needs to be combined with a time limit, as the incomplete subsolvers do not know when to stop"

**Recommendation:** Worth testing `use_lns_only` if LNS workers are finding solutions but core MIP is slow. However, the CP-SAT Primer warns that "most solver parameters beyond basic settings like `max_time_in_seconds` are best left at their defaults."

### 2. Cumulative-Specific Parameters (Weak Evidence)

**What we know:**
- Cumulative constraints have weak LP relaxations in CP-SAT [[Perron et al. 2023](https://drops.dagstuhl.de/storage/00lipics/lipics-vol280-cp2023/LIPIcs.CP.2023.3/LIPIcs.CP.2023.3.pdf)]
- Research shows redundant constraints (cutting planes) can strengthen cumulative propagation [[Thesis HAL](https://theses.hal.science/tel-01745888/document)]
- Energetic reasoning algorithms exist for cumulative constraints [[IJCAI 2016](https://ijcai.org/Proceedings/16/Papers/440.pdf)]
- `use_cumulative_in_no_overlap_2d` provides "an additional level of reasoning" for 2D packing [[GitHub Discussion](https://github.com/google/or-tools/discussions/3107)]

**Recommendation:** Limited evidence for cumulative-specific parameters improving general scheduling. Most parameters may not exist in the Rust CP-SAT API or may not apply to single-dimensional cumulative constraints. **Better approach: add redundant constraints to the model** (per-cycle bounds, profile bounds) rather than tuning solver parameters.

### 3. Optional Variable Handling (Mixed Evidence)

**What we know:**
- Optional intervals "defer early commitment to alternatives, allowing the solver to maintain multiple possibilities until more information constrains the search space" [[OR StackExchange](https://or.stackexchange.com/questions/7563/why-most-of-constraint-solvers-do-not-support-optional-intervals)]
- CP-SAT shows "drastically worse performance when optional intervals must be left unscheduled compared to fully scheduled scenarios" [[OR-Tools Discuss](https://groups.google.com/g/or-tools-discuss/c/tmF6MqBYU4)]
- Limited evidence that prioritizing optional variables in search helps—the CP-SAT Primer warns that "inappropriate heuristics can degrade performance" [[CP-SAT Primer Parameters](https://d-krupke.github.io/cpsat-primer/05_parameters.html)]

**Recommendation:** The evidence is mixed. Optional intervals help propagation by deferring commitment, but many unscheduled optional intervals hurt performance. Fixing them early might help, but there's no strong evidence that search strategies prioritizing optional variables improve performance.

### 4. Search Strategy for Choice Variables (Strong Evidence Against)

**What we know:**
- The CP-SAT Primer explicitly warns that "inappropriate heuristics can degrade performance" and decision strategies "are best left at their default settings, as they are well-chosen and tampering with them could disrupt optimizations" [[CP-SAT Primer Parameters](https://d-krupke.github.io/cpsat-primer/05_parameters.html)]
- The primer author's experience: "I played around a little with selecting a manual search strategy. But even for the coloring, where this may even seem smart, it only gave an advantage for a bad model and after improving the model by symmetry breaking, it performed worse"
- "CP-SAT can learn the best strategy (Gurobi does such a thing, too) much better dynamically on its own"

**Recommendation:** **Do not manually configure search strategies.** There's strong evidence that manual search strategies often hurt performance. CP-SAT's default strategies are well-tuned, and the solver learns better strategies dynamically.

### 5. Interval Variable Parameters (Weak Evidence)

**What we know:**
- Minimal evidence in literature or documentation that interval-specific parameters significantly improve performance
- The CP-SAT Primer emphasizes that "for better performance, focus on improving your model" rather than tuning parameters

**Recommendation:** These parameters may not exist or may not be documented. The general guidance is to focus on model improvements rather than parameter tuning.

## Overall Recommendation

**Focus on model improvements rather than parameter tuning:**

1. **Add redundant constraints** (per-cycle bounds, profile bounds) - Strong evidence this helps cumulative propagation
2. **Improve tie-breaker objective** - Guides search toward feasibility without disrupting solver's internal strategies
3. **Test `use_lns_only`** - Moderate evidence this might help if LNS is finding solutions but core MIP is slow
4. **Avoid manual search strategies** - Strong evidence this hurts performance

## References

- [CP-SAT Primer - LNS](https://d-krupke.github.io/cpsat-primer/09_lns.html)
- [CP-SAT Primer - Parameters](https://d-krupke.github.io/cpsat-primer/05_parameters.html)
- [Perron et al. 2023 - CP-SAT-LP Solver](https://drops.dagstuhl.de/storage/00lipics/lipics-vol280-cp2023/LIPIcs.CP.2023.3/LIPIcs.CP.2023.3.pdf)
- [IJCAI 2016 - Energetic Reasoning](https://ijcai.org/Proceedings/16/Papers/440.pdf)
- [OR StackExchange - Optional Intervals](https://or.stackexchange.com/questions/7563/why-most-of-constraint-solvers-do-not-support-optional-intervals)
- [OR-Tools GitHub Discussion - Cumulative](https://github.com/google/or-tools/discussions/3107)
- [OR-Tools Mailing List - Optional Intervals Performance](https://groups.google.com/g/or-tools-discuss/c/tmF6MqBYU4)
