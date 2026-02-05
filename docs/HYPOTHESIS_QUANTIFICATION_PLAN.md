# Hypothesis Quantification Plan

Use the scheduler output JSON (`--output-json` from `generate_ssa_kernel.py` with CP-SAT) as the single source of truth. The JSON contains `ops` (id, cycle, engine, opcode, dep_ids), `shifted_alu_cycles` (op_id → [cycle0..7] when VALU shift enabled), `schedule_constraints`, and optionally `ssa_ops_bulk_move` / `ssa_cycle_assignments`. ConstChoice / SelectChoice are **not** stored explicitly; infer from engine/opcode (e.g. add_imm vs const, vselect vs FMA).

---

## 1. Hypothesis summary and quantification

| Hypothesis | Improvement idea | Quantification approach | Main metrics |
|------------|------------------|-------------------------|--------------|
| VALU shift | Enable shift / solver chooses more shift | A/B: with vs without `--enable-valu-shift` | Δcycles, Δpeak_live, decile ΔVALU/ALU |
| Vbroadcast elimination | Emit vbroadcast only when needed | Script on JSON: consumers all shifted → eliminable | # eliminable vbroadcasts, VALU slots saved per cycle |
| Element-wise chaining | Per-lane completion for shifted deps | Script on JSON: slack = max−min ALU cycles per dep | Slack distribution, theoretical cycle bound |
| ConstChoice | Prefer add_imm in early deciles | Decile counts from JSON; optional A/B with hints/seeds | add_imm vs const in decile 0, correlation with cycles |
| Precedence | Move "VALU other" earlier where ALU free | Bulk-move simulator + what-if shift | Compacted makespan, blocking constraint, Δcycles per what-if |

---

## 2. Per-hypothesis details

### 2.1 VALU shift

**Idea:** More VALU→ALU shifting frees VALU slots in mid-deciles and can reduce makespan if ALU has slack.

**Quantify:** A/B runs — generate two kernels (same params): one with `--enable-valu-shift`, one without. Compare: `total_cycles`, `peak_live_words`, `budget_exceeded`, and decile table (`scripts/slot_usage_by_decile.py --markdown` on both JSONs). Metric: Δcycles, Δpeak_live, and per-decile Δ(VALU other) and Δ(ALU).

### 2.2 Vbroadcast elimination

**Idea:** If all consumers of a vbroadcast are shifted to ALU, the vbroadcast could be replaced by 8× scalar constant loads or add_imm; the current implementation may still emit the vbroadcast ([VALU_SHIFT_VBROADCAST_DEPENDENCY_OPTIMIZATION.md](VALU_SHIFT_VBROADCAST_DEPENDENCY_OPTIMIZATION.md)).

**Quantify (post-hoc on JSON):** Build consumer map from `ops` (dep_ids → consumers). For each op with engine VALU and opcode `vbroadcast`: if every consumer has id in `shifted_alu_cycles`, mark as "could_eliminate". Metrics: count and list of eliminable vbroadcasts; per-cycle VALU slot saving if removed (1 slot per vbroadcast at its cycle).

### 2.3 Element-wise ALU chaining

**Idea:** Shifted VALU ops have 8 ALU cycles (one per lane). Dependencies could be satisfied per-lane when that lane's ALU op completes, instead of when the whole vector "completes" (max of 8 cycles).

**Quantify (post-hoc on JSON):** For each op in `shifted_alu_cycles`: completion_cycle = max(alu_cycles). For each dependent, current model uses completion_cycle+1 as earliest start. Per-lane slack: max(alu_cycles) - min(alu_cycles) per dependency. Metrics: distribution of per-op "wasted" slack for shifted ops that have dependents; theoretical cycle saving if dependents were scheduled per-lane (upper bound).

### 2.4 ConstChoice

**Idea:** Prefer add_imm over Load for constants in the first deciles to free Load slots.

**Quantify:** From JSON, in the first decile count Flow `add_imm` and Load `const`. Optional: A/B with solver hints or seeds; correlate decile-0 add_imm vs const ratio with cycles.

### 2.5 Precedence

**Idea:** Some "VALU other" ops may be placed in mid-deciles by precedence; moving them earlier could use empty ALU slots.

**Quantify:** Use [use-bulk-move](.cursor/skills/use-bulk-move/SKILL.md) (`scripts/simulate_bulk_move.py`) and [use-whatif-shift](.cursor/skills/use-whatif-shift/SKILL.md) (`scripts/what_if_shift.py`). Metrics: bulk-move original vs final makespan and blocking constraint; what-if Δcycles for selected ops.

---

## 3. New tools and skills

None of the existing scripts or skills implement the **post-hoc JSON analysis** for vbroadcast eliminability or element-wise slack. The following additions are recommended.

### 3.1 New tool: `scripts/quantify_schedule_hypotheses.py`

**Purpose:** One script that reads a scheduler output JSON and reports metrics for the hypotheses that can be computed without re-running the solver.

**Input:** Scheduler output JSON path (default: `scheduler_output.json`). Optional: `--markdown` for renderable output.

**Output (sections):**

1. **Vbroadcast eliminability**
   - Build consumer map: for each op, which ops list it in `dep_ids`.
   - For each op with `engine == "valu"` and `opcode == "vbroadcast"`: if `shifted_alu_cycles` is present and non-empty, check whether every consumer op has its id in `shifted_alu_cycles`. If yes, mark vbroadcast as eliminable.
   - Report: count of eliminable vbroadcasts; list (id, cycle); optional summary of VALU slot saving per cycle (1 slot per eliminable vbroadcast at its cycle).
   - If `shifted_alu_cycles` is missing or empty, report "N/A (run with --enable-valu-shift and output-json)".

2. **Element-wise slack (shifted ops)**
   - For each op_id in `shifted_alu_cycles`, get `alu_cycles` (length 8). Completion = max(alu_cycles). Find all consumer ops (ops that have op_id in `dep_ids`). For each consumer, "wasted" slack = completion - min(alu_cycles).
   - Report: number of shifted ops that have at least one consumer; distribution of slack (e.g. min/max/mean, or histogram); theoretical cycle saving: for each consumer, if it could start at min(dep_alu_cycles)+1 instead of completion+1, sum max(0, consumer_cycle - (min(dep_alu_cycles)+1)). Use consumer's cycle from `ops` (or min(consumer's alu_cycles) if consumer is also shifted). Keep formula simple and document in script docstring.

3. **Decile 0 ConstChoice summary**
   - First decile: cycles 0..total_cycles//10 (or 0..(total_cycles*10)//100). Sum over those cycles: Flow add_imm count, Load const count (from scanning `ops` by cycle, engine, opcode).
   - Report: "Decile 0: add_imm N, Load const M". Helps correlate with ConstChoice hypothesis without solver changes.

**Dependencies:** Only standard library + reading JSON; same style as `slot_usage_by_decile.py` (argparse, no heavy deps).

**Placement:** New file `scripts/quantify_schedule_hypotheses.py`. Referenced by the new skill and by this plan.

### 3.2 Optional tool: A/B VALU shift runner

**Purpose:** Run generator twice (with and without `--enable-valu-shift`), same time limit and rounds, then compare cycles and optionally run decile/hypothesis script on both outputs.

**Options:**
- **A) Small script `scripts/run_valu_shift_ab.py`:** Invokes `generate_ssa_kernel.py` twice (e.g. with `--output-json scheduler_output_shift.json` and `..._no_shift.json`), parses cycle/peak_live from generator stdout or from JSON, runs `slot_usage_by_decile.py` on both, prints Δ. Pros: one command. Cons: duplicates invocation logic and may be brittle if generator output format changes.
- **B) Document in skill only:** "For VALU shift A/B: run `uv run generate_ssa_kernel.py --output-json out_shift.json --enable-valu-shift` and `... --output-json out_noshift.json` (no flag), then compare JSONs (total_cycles, peak_live_words) and run `slot_usage_by_decile.py` on each." No new script.

**Recommendation:** Start with **B** (skill-only). Add script **A** later if users run A/B often and want automation.

### 3.3 New skill: `use-schedule-hypotheses`

**Purpose:** When estimating impact of vbroadcast elimination, element-wise chaining, ConstChoice, or VALU shift (before changing the solver or encoder), or when analyzing an optimal/hand-tuned schedule to see which hypotheses might help.

**When to use:**
- Before changing the solver or encoder — estimate whether a strategy could improve cycles.
- After producing an optimal or hand-tuned schedule — run hypothesis metrics to see eliminable vbroadcasts, slack, and decile-0 mix.

**How to use:**
1. **Prerequisite:** Scheduler output JSON from a run you care about (e.g. `uv run generate_ssa_kernel.py --output-json scheduler_output.json`).
2. **Run hypothesis quantification script:**  
   `uv run scripts/quantify_schedule_hypotheses.py [scheduler_output.json]`  
   Optionally `--markdown` for renderable output. Interpret: eliminable vbroadcast count (and VALU slot saving), element-wise slack distribution and theoretical cycle bound, decile 0 add_imm vs const.
3. **VALU shift A/B:** Run generator with and without `--enable-valu-shift` (same `--cpsat-time-limit`, `--rounds`), save to two JSONs. Compare `total_cycles`, `peak_live_words`, `budget_exceeded`. Run `slot_usage_by_decile.py` on both to compare decile utilization.
4. **Precedence / movement:** Use [use-bulk-move](.cursor/skills/use-bulk-move/SKILL.md) (`scripts/simulate_bulk_move.py`) to see if empty cycles can be compacted and which constraint blocks. Use [use-whatif-shift](.cursor/skills/use-whatif-shift/SKILL.md) (`scripts/what_if_shift.py`) to force shift for specific ops and compare cycles.

**References:**  
- Script: [scripts/quantify_schedule_hypotheses.py](../scripts/quantify_schedule_hypotheses.py)  
- Plan: [docs/HYPOTHESIS_QUANTIFICATION_PLAN.md](HYPOTHESIS_QUANTIFICATION_PLAN.md)  
- Schedule debug (per-decile): [.cursor/skills/use-schedule-debug/SKILL.md](../.cursor/skills/use-schedule-debug/SKILL.md)  
- Bulk-move: [.cursor/skills/use-bulk-move/SKILL.md](../.cursor/skills/use-bulk-move/SKILL.md)  
- What-if shift: [.cursor/skills/use-whatif-shift/SKILL.md](../.cursor/skills/use-whatif-shift/SKILL.md)

**Skill file:** Create `.cursor/skills/use-schedule-hypotheses/SKILL.md` with the above content (YAML frontmatter: name `use-schedule-hypotheses`, description one line for when to use).

### 3.4 CLAUDE.md update

Add one bullet under **Project skills** (after the existing schedule-debug / what-if / bugs / bisection bullets):

- **Hypothesis quantification (schedule optimization ideas)** — Read [.cursor/skills/use-schedule-hypotheses/SKILL.md](.cursor/skills/use-schedule-hypotheses/SKILL.md). Use `scripts/quantify_schedule_hypotheses.py` on scheduler output JSON to estimate impact of vbroadcast elimination, element-wise chaining, and ConstChoice; use A/B runs and bulk-move/what-if for VALU shift and precedence.

---

## 4. Implementation order

1. **Implement** `scripts/quantify_schedule_hypotheses.py` (vbroadcast + slack + decile-0).
2. **Create** `.cursor/skills/use-schedule-hypotheses/SKILL.md`.
3. **Update** `CLAUDE.md` with the new skill bullet.
4. **Optional later:** Add `scripts/run_valu_shift_ab.py` if A/B runs are frequent; extend skill to mention it.

No changes to the Rust scheduler are required for quantification; only the new script and skill (and CLAUDE.md).
