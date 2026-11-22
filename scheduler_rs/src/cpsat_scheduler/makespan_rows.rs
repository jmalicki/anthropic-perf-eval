//! Makespan lower-bound rows from static heads and tails (`--cpsat-makespan-rows`).
//!
//! # Static heads and tails
//!
//! `head(o)` is the longest dependency path from a source to `o`, every edge of latency 1
//! (the minimum over every encoding: vselect, a VALU op that is not shifted, a shifted op
//! lane by lane), except edges out of an eliminable vbroadcast, which have latency 0 (when it
//! is eliminated its shifted readers read the scalar one cycle after the scalar's cycle).
//! `tail(o)` is the longest path from `o` to a sink, the op's own cycle excluded: a sink has
//! tail 0. These are not the model's ASAP/ALAP, which are tightened by slot reasoning and the
//! greedy warm start.
//!
//! In every schedule of the model (`op_times.rs`), every slot op `o` occupies (its issue
//! cycle, a lane of a shifted op, the multiply_add cycle of a select) lies in
//! `[head(o), makespan - 1 - tail(o)]`. A set of ops with `head >= t` and `tail >= q` that
//! needs `W` slots of an engine with `cap` slots per cycle therefore needs
//! `cap * (makespan - t - q) >= W`. (Counting the op's own cycle in the tail, as a first draft
//! of this definition did, gives a row that is one cycle too strong; the proof in
//! `verification/makespan_rows.py` fails for that variant.)
//!
//! # Rows
//!
//! * **Static constant row** (`Static`, `Bucketed`): `makespan >= max over engines and (t, q)
//!   of t + q + ceil(W_min / cap)`, where `W_min` takes the cheapest encoding of every op
//!   (shiftable ops count 8 in the joint VALU+ALU row of capacity `8 * VALU + ALU`; selects
//!   and constants count 0). This equals `scripts/static_bounds.py` with `encodings="any"`.
//! * **Static literal rows** (`Static`, `Bucketed`): for VALU, ALU, Load and Flow, the work of
//!   a head window (`head >= t`) or a tail window (`tail >= q`) as a linear expression of the
//!   shift, SelectChoice and ConstChoice literals, at up to [`LITERAL_ROWS_PER_SIDE`]
//!   thresholds per side: those whose cheapest-encoding bound is highest.
//! * **Bucketed rows** (`Bucketed`): ops are put in buckets of [`BUCKET_CYCLES`] static-head
//!   cycles. `prefix_end_k` is at least every finish term (issue cycle, completion cycle,
//!   multiply_add cycle) of the ops in buckets `0..=k`; `suffix_start_k` is at most the
//!   start (issue cycle, or each lane when shifted) of the ops in buckets `k..`. Per engine
//!   `cap * (prefix_end_k + 1) >= work(prefix)` and `cap * (makespan - suffix_start_k) >=
//!   work(suffix)`, with the work accumulated in chained auxiliaries so that each op appears
//!   in O(1) rows. Size: `2 * buckets` boundary variables, `5 * 2 * buckets` work
//!   auxiliaries, and per op one row to its bucket's `prefix_end` (three for an op with a
//!   completion or multiply_add cycle) and one to its `suffix_start` (nine for a shiftable
//!   op: eight lanes enforced by the shift literal, the issue cycle by its negation).

use std::collections::{BTreeMap, HashSet};

#[cfg(feature = "cpsat")]
use cp_sat::builder::{BoolVar, CpModelBuilder, IntVar, LinearExpr};

#[cfg(feature = "cpsat")]
use super::config::ConstChoiceMode;
#[cfg(feature = "cpsat")]
use super::op_times::{ConstChoiceVars, FmaPath, OpTimes};
use crate::types::{Engine, Opcode, SsaOp, VLEN};

/// Static-head cycles per bucket in `Bucketed` mode.
pub(crate) const BUCKET_CYCLES: i64 = 32;
/// Literal rows per engine and side (head windows, tail windows) in `Static` mode.
pub(crate) const LITERAL_ROWS_PER_SIDE: usize = 8;

/// Rows of the static bound: one per engine, plus the joint VALU+ALU row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum RowEngine {
    Valu,
    Alu,
    Load,
    Store,
    Flow,
    /// VALU and ALU together: a VALU op counts VLEN, a scalar op 1, capacity VLEN*VALU + ALU.
    Joint,
}

impl RowEngine {
    pub(crate) const ALL: [RowEngine; 6] =
        [RowEngine::Valu, RowEngine::Alu, RowEngine::Load, RowEngine::Store, RowEngine::Flow, RowEngine::Joint];

    /// Name used in the scheduler JSON and in `scripts/static_bounds.py`.
    pub(crate) fn name(self) -> &'static str {
        match self {
            RowEngine::Valu => "valu",
            RowEngine::Alu => "alu",
            RowEngine::Load => "load",
            RowEngine::Store => "store",
            RowEngine::Flow => "flow",
            RowEngine::Joint => "alu+valu",
        }
    }
}

/// Slots per cycle of each row engine.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RowCaps {
    pub(crate) valu: i64,
    pub(crate) alu: i64,
    pub(crate) load: i64,
    pub(crate) store: i64,
    pub(crate) flow: i64,
}

impl RowCaps {
    pub(crate) fn of(&self, engine: RowEngine) -> i64 {
        match engine {
            RowEngine::Valu => self.valu,
            RowEngine::Alu => self.alu,
            RowEngine::Load => self.load,
            RowEngine::Store => self.store,
            RowEngine::Flow => self.flow,
            RowEngine::Joint => VLEN as i64 * self.valu + self.alu,
        }
    }
}

/// How each op can be encoded, for the cheapest-encoding work.
pub(crate) struct OpClasses<'a> {
    /// Ops that may be shifted to scalar ALU (empty when the shift is off)
    pub(crate) shiftable: &'a HashSet<usize>,
    /// Vbroadcasts that may be eliminated (empty when elimination is off)
    pub(crate) eliminable_vbs: &'a HashSet<usize>,
}

/// Static heads and tails of every op (see the module doc).
#[derive(Debug, Clone)]
pub(crate) struct StaticPaths {
    pub(crate) head: Vec<i64>,
    pub(crate) tail: Vec<i64>,
}

/// Longest-path heads and tails over the dependency edges of `ops`.
///
/// # Arguments
/// * `ops` - SSA ops (edges: `dep_ids` that name an op of the slice)
/// * `zero_latency_out` - Ops whose out-edges have latency 0 (eliminable vbroadcasts)
///
/// # Returns
/// Heads and tails indexed like `ops`.
pub(crate) fn static_paths(ops: &[SsaOp], zero_latency_out: &HashSet<usize>) -> StaticPaths {
    let n = ops.len();
    let id_to_idx: BTreeMap<usize, usize> = ops.iter().enumerate().map(|(i, op)| (op.id, i)).collect();
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut succs: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, op) in ops.iter().enumerate() {
        for dep in &op.dep_ids {
            if let Some(&d) = id_to_idx.get(dep) {
                if d != i {
                    preds[i].push(d);
                    succs[d].push(i);
                }
            }
        }
    }
    // Kahn order (dependencies form a DAG); ties by index for determinism.
    let mut indegree: Vec<usize> = preds.iter().map(|p| p.len()).collect();
    let mut ready: std::collections::BTreeSet<usize> = (0..n).filter(|&i| indegree[i] == 0).collect();
    let mut order = Vec::with_capacity(n);
    while let Some(i) = ready.pop_first() {
        order.push(i);
        for &s in &succs[i] {
            indegree[s] -= 1;
            if indegree[s] == 0 {
                ready.insert(s);
            }
        }
    }
    assert_eq!(order.len(), n, "dependency graph has a cycle");
    let latency = |from: usize| -> i64 { if zero_latency_out.contains(&from) { 0 } else { 1 } };
    let mut head = vec![0i64; n];
    for &i in &order {
        for &p in &preds[i] {
            head[i] = head[i].max(head[p] + latency(p));
        }
    }
    let mut tail = vec![0i64; n];
    for &i in order.iter().rev() {
        for &s in &succs[i] {
            tail[i] = tail[i].max(tail[s] + latency(i));
        }
    }
    StaticPaths { head, tail }
}

/// Slots of `engine` op `idx` needs under its cheapest encoding.
pub(crate) fn min_work(ops: &[SsaOp], idx: usize, engine: RowEngine, classes: &OpClasses<'_>) -> i64 {
    let op = &ops[idx];
    if matches!(op.opcode, Opcode::SelectChoice | Opcode::ConstChoice) {
        return 0;
    }
    let vector = op.engine == Engine::Valu && !classes.eliminable_vbs.contains(&idx);
    let shiftable = classes.shiftable.contains(&idx);
    match engine {
        RowEngine::Valu => i64::from(vector && !shiftable),
        RowEngine::Alu => i64::from(op.engine == Engine::Alu),
        RowEngine::Load => i64::from(op.engine == Engine::Load),
        RowEngine::Store => i64::from(op.engine == Engine::Store),
        RowEngine::Flow => i64::from(op.engine == Engine::Flow),
        RowEngine::Joint => {
            if vector {
                VLEN as i64
            } else {
                i64::from(op.engine == Engine::Alu)
            }
        }
    }
}

/// The strongest constant row: `makespan >= bound`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StaticBound {
    pub(crate) bound: i64,
    pub(crate) engine: RowEngine,
    pub(crate) t: i64,
    pub(crate) q: i64,
}

/// `max over engines, t, q of t + q + ceil(W(head >= t, tail >= q) / cap)` with the cheapest
/// encoding of every op (0 when there are no ops).
pub(crate) fn static_bound(
    ops: &[SsaOp],
    paths: &StaticPaths,
    classes: &OpClasses<'_>,
    caps: &RowCaps,
) -> Option<StaticBound> {
    let max_head = paths.head.iter().copied().max()? as usize;
    let max_tail = paths.tail.iter().copied().max()? as usize;
    let mut best: Option<StaticBound> = None;
    for engine in RowEngine::ALL {
        let cap = caps.of(engine);
        if cap <= 0 {
            continue;
        }
        // grid[t][q] = work with head == t and tail == q, then 2-D suffix sums.
        let (h, w) = (max_head + 2, max_tail + 2);
        let mut grid = vec![0i64; h * w];
        for idx in 0..ops.len() {
            let work = min_work(ops, idx, engine, classes);
            if work > 0 {
                grid[paths.head[idx] as usize * w + paths.tail[idx] as usize] += work;
            }
        }
        for t in (0..h - 1).rev() {
            for q in (0..w - 1).rev() {
                grid[t * w + q] += grid[(t + 1) * w + q] + grid[t * w + q + 1] - grid[(t + 1) * w + q + 1];
            }
        }
        for t in 0..h - 1 {
            for q in 0..w - 1 {
                let work = grid[t * w + q];
                if work <= 0 {
                    continue;
                }
                let bound = t as i64 + q as i64 + (work + cap - 1) / cap;
                let candidate = StaticBound { bound, engine, t: t as i64, q: q as i64 };
                if best.map_or(true, |b| (bound, std::cmp::Reverse(candidate.engine)) > (b.bound, std::cmp::Reverse(b.engine))) {
                    best = Some(candidate);
                }
            }
        }
    }
    best
}

// ============================================================================
// Model rows
// ============================================================================

/// Work as `constant + sum(coeff * literal)`.
#[cfg(feature = "cpsat")]
#[derive(Debug, Clone, Default)]
pub(crate) struct LinearWork {
    pub(crate) constant: i64,
    pub(crate) terms: Vec<(i64, BoolVar)>,
}

#[cfg(feature = "cpsat")]
impl LinearWork {
    fn add(&mut self, other: &LinearWork) {
        self.constant += other.constant;
        self.terms.extend_from_slice(&other.terms);
    }
    /// `coeff * b`
    fn lit(coeff: i64, b: BoolVar) -> Self {
        LinearWork { constant: 0, terms: vec![(coeff, b)] }
    }
    /// `coeff * (1 - b)`
    fn not_lit(coeff: i64, b: BoolVar) -> Self {
        LinearWork { constant: coeff, terms: vec![(-coeff, b)] }
    }
    fn constant(c: i64) -> Self {
        LinearWork { constant: c, terms: Vec::new() }
    }
    fn has_literals(&self) -> bool {
        !self.terms.is_empty()
    }
    /// Largest value over all literal assignments.
    fn max_value(&self) -> i64 {
        self.constant + self.terms.iter().map(|&(c, _)| c.max(0)).sum::<i64>()
    }
    /// Value under a hint (`None` if a literal is unhinted).
    fn eval(&self, hint: &dyn Fn(IntVar) -> Option<i64>) -> Option<i64> {
        let mut v = self.constant;
        for &(c, b) in &self.terms {
            v += c * hint(IntVar::from(b))?;
        }
        Some(v)
    }
}

/// Decision literals the work of an op depends on.
#[cfg(feature = "cpsat")]
pub(crate) struct WorkContext<'a> {
    pub(crate) ops: &'a [SsaOp],
    pub(crate) times: &'a [OpTimes],
    pub(crate) classes: OpClasses<'a>,
    pub(crate) const_choice: &'a std::collections::HashMap<usize, ConstChoiceVars>,
    pub(crate) const_choice_mode: ConstChoiceMode,
}

/// Slots of `engine` op `idx` uses, as a function of its encoding literals. Never more than
/// what the op uses in the encoding the literals select, so every row built from it holds
/// in every schedule. SelectChoice VALU slots are left out (CSE groups share subtracts) and
/// an eliminable vbroadcast counts 0; both only weaken the rows.
#[cfg(feature = "cpsat")]
pub(crate) fn op_work(ctx: &WorkContext<'_>, idx: usize, engine: RowEngine) -> LinearWork {
    let op = &ctx.ops[idx];
    let none = LinearWork::default();
    match op.opcode {
        Opcode::SelectChoice => {
            return match (engine, ctx.times[idx].fma_path()) {
                (RowEngine::Flow, Some(FmaPath::Never)) => LinearWork::constant(1),
                (RowEngine::Flow, Some(FmaPath::Chosen(b))) => LinearWork::not_lit(1, b),
                _ => none,
            };
        }
        Opcode::ConstChoice => {
            return match (engine, ctx.const_choice_mode, ctx.const_choice.get(&idx)) {
                (RowEngine::Load, ConstChoiceMode::ForceLoad, _) => LinearWork::constant(1),
                (RowEngine::Flow, ConstChoiceMode::ForceAddImm, _) => LinearWork::constant(1),
                (RowEngine::Load, ConstChoiceMode::Optimize, Some(ConstChoiceVars::AddImm(b))) => {
                    LinearWork::not_lit(1, *b)
                }
                (RowEngine::Flow, ConstChoiceMode::Optimize, Some(ConstChoiceVars::AddImm(b))) => {
                    LinearWork::lit(1, *b)
                }
                (RowEngine::Load, ConstChoiceMode::Optimize, Some(ConstChoiceVars::Encoding(v))) => {
                    LinearWork::lit(1, v[0])
                }
                (RowEngine::Flow, ConstChoiceMode::Optimize, Some(ConstChoiceVars::Encoding(v))) => {
                    LinearWork::lit(1, v[1])
                }
                (RowEngine::Alu, ConstChoiceMode::Optimize, Some(ConstChoiceVars::Encoding(v))) => LinearWork {
                    constant: 0,
                    terms: v[2..].iter().map(|&b| (1, b)).collect(),
                },
                _ => none,
            };
        }
        _ => {}
    }
    match (engine, ctx.times[idx].shift()) {
        (RowEngine::Valu, Some(s)) => LinearWork::not_lit(1, s),
        (RowEngine::Alu, Some(s)) => LinearWork::lit(VLEN as i64, s),
        _ => LinearWork::constant(min_work(ctx.ops, idx, engine, &ctx.classes)),
    }
}

/// One row added to the model, kept so it can be checked against the hint:
/// `sum(coeff * var) >= rhs`, enforced by `enforce` when present.
#[cfg(feature = "cpsat")]
#[derive(Debug, Clone)]
pub(crate) struct Row {
    pub(crate) name: String,
    pub(crate) terms: Vec<(i64, IntVar)>,
    pub(crate) rhs: i64,
    pub(crate) enforce: Option<BoolVar>,
}

/// What [`add_makespan_rows`] added.
#[cfg(feature = "cpsat")]
#[derive(Debug, Default)]
pub(crate) struct AddedRows {
    pub(crate) static_bound: Option<StaticBound>,
    pub(crate) literal_rows: usize,
    pub(crate) buckets: usize,
    pub(crate) new_vars: usize,
    pub(crate) rows: Vec<Row>,
    /// Hints for the new variables, in creation order
    pub(crate) hints: Vec<(IntVar, i64)>,
}

#[cfg(feature = "cpsat")]
impl AddedRows {
    fn ge(&mut self, model: &mut CpModelBuilder, name: String, terms: Vec<(i64, IntVar)>, rhs: i64, enforce: Option<BoolVar>) {
        let mut lhs = LinearExpr::default();
        for &(c, v) in &terms {
            lhs = lhs + LinearExpr::from((c, v));
        }
        let ct = model.add_ge(lhs, rhs);
        if let Some(lit) = enforce {
            model.only_enforce_if(ct, [lit]);
        }
        model.set_constraint_name(ct, &name);
        self.rows.push(Row { name, terms, rhs, enforce });
    }

    /// Rows the hint (the model's hints plus [`Self::hints`]) violates. Rows whose
    /// enforcement literal is hinted false, or with an unhinted variable, are skipped.
    pub(crate) fn hint_violations(&self, model_hint: &dyn Fn(IntVar) -> Option<i64>) -> Vec<String> {
        let own: std::collections::HashMap<IntVar, i64> = self.hints.iter().copied().collect();
        let value = |v: IntVar| own.get(&v).copied().or_else(|| model_hint(v));
        let literal = |b: BoolVar| -> Option<bool> {
            let positive = b.index() >= 0;
            let var = if positive { IntVar::from(b) } else { IntVar::from(!b) };
            value(var).map(|h| (h != 0) == positive)
        };
        self.rows
            .iter()
            .filter(|row| {
                if let Some(lit) = row.enforce {
                    if literal(lit) != Some(true) {
                        return false;
                    }
                }
                let mut lhs = 0i64;
                for &(c, v) in &row.terms {
                    match value(v) {
                        Some(h) => lhs += c * h,
                        None => return false,
                    }
                }
                lhs < row.rhs
            })
            .map(|row| row.name.clone())
            .collect()
    }
}

/// `coeff * work` as terms, and its constant times `coeff`.
#[cfg(feature = "cpsat")]
fn work_terms(w: &LinearWork, coeff: i64) -> (Vec<(i64, IntVar)>, i64) {
    (w.terms.iter().map(|&(c, b)| (c * coeff, IntVar::from(b))).collect(), w.constant * coeff)
}

/// Up to `k` thresholds `t >= 1` of one side (head or tail values), highest
/// `t + ceil(work(value >= t) / cap)` first (ties: smaller threshold).
fn top_thresholds(values: &[i64], work: &[i64], cap: i64, k: usize) -> Vec<i64> {
    let max = values.iter().copied().max().unwrap_or(0).max(0) as usize;
    let mut suffix = vec![0i64; max + 2];
    for (&v, &w) in values.iter().zip(work) {
        suffix[v as usize] += w;
    }
    for t in (0..=max).rev() {
        suffix[t] += suffix[t + 1];
    }
    let mut scored: Vec<(i64, i64)> = (1..=max as i64)
        .filter(|&t| suffix[t as usize] > 0)
        .map(|t| (t + (suffix[t as usize] + cap - 1) / cap, t))
        .collect();
    scored.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    scored.into_iter().take(k).map(|(_, t)| t).collect()
}

/// Add the rows of `--cpsat-makespan-rows` to the model (see the module doc).
///
/// # Arguments
/// * `model` - Model under construction, after its hints were added
/// * `ctx` - Ops, times and literals
/// * `paths` - Static heads and tails
/// * `caps` - Slots per cycle
/// * `makespan` - The makespan variable
/// * `horizon` - Upper bound of cycle domains
/// * `bucketed` - Also add the bucketed prefix/suffix rows
/// * `hint` - Hinted value of a model variable (to hint the new auxiliaries)
///
/// # Returns
/// What was added; the caller adds [`AddedRows::hints`] to the model.
#[cfg(feature = "cpsat")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn add_makespan_rows(
    model: &mut CpModelBuilder,
    ctx: &WorkContext<'_>,
    paths: &StaticPaths,
    caps: &RowCaps,
    makespan: IntVar,
    horizon: i64,
    bucketed: bool,
    hint: &dyn Fn(IntVar) -> Option<i64>,
) -> AddedRows {
    let mut added = AddedRows::default();
    let n = ctx.ops.len();
    let m = makespan;

    // 1. Constant row.
    added.static_bound = static_bound(ctx.ops, paths, &ctx.classes, caps);
    if let Some(sb) = added.static_bound {
        added.ge(model, "makespan_ge_static_headtail".to_string(), vec![(1, m)], sb.bound, None);
    }

    // 2. Literal rows over head windows (head >= t) and tail windows (tail >= q).
    for engine in [RowEngine::Valu, RowEngine::Alu, RowEngine::Load, RowEngine::Flow] {
        let cap = caps.of(engine);
        let works: Vec<LinearWork> = (0..n).map(|i| op_work(ctx, i, engine)).collect();
        if cap <= 0 || !works.iter().any(LinearWork::has_literals) {
            continue;
        }
        // VALU and ALU thresholds are ranked by the joint cheapest-encoding bound (the two
        // rows together are never weaker than it); Load and Flow by their largest work.
        let (score_work, score_cap): (Vec<i64>, i64) = match engine {
            RowEngine::Valu | RowEngine::Alu => (
                (0..n).map(|i| min_work(ctx.ops, i, RowEngine::Joint, &ctx.classes)).collect(),
                caps.of(RowEngine::Joint),
            ),
            _ => (works.iter().map(LinearWork::max_value).collect(), cap),
        };
        for (side, values) in [("head", &paths.head), ("tail", &paths.tail)] {
            for threshold in top_thresholds(values, &score_work, score_cap, LITERAL_ROWS_PER_SIDE) {
                let mut work = LinearWork::default();
                for i in (0..n).filter(|&i| values[i] >= threshold) {
                    work.add(&works[i]);
                }
                if !work.has_literals() {
                    continue;
                }
                // cap * (makespan - threshold) >= work
                let (wt, wc) = work_terms(&work, -1);
                let mut terms = vec![(cap, m)];
                terms.extend(wt);
                let name = format!("makespan_rows_{}_{}_{}", engine.name(), side, threshold);
                added.ge(model, name, terms, cap * threshold - wc, None);
                added.literal_rows += 1;
            }
        }
    }

    if !bucketed {
        return added;
    }

    // 3. Bucketed prefix/suffix rows.
    let max_head = paths.head.iter().copied().max().unwrap_or(0);
    let buckets = (max_head / BUCKET_CYCLES + 1) as usize;
    if buckets < 2 {
        return added;
    }
    added.buckets = buckets;
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); buckets];
    for i in 0..n {
        members[(paths.head[i] / BUCKET_CYCLES) as usize].push(i);
    }
    let hinted_true = |b: BoolVar| hint(IntVar::from(b)).map(|h| h != 0);

    // prefix_end_k (k = 0..buckets-2): at least every finish term of buckets 0..=k.
    let mut prefix_end: Vec<IntVar> = Vec::new();
    let mut prefix_hint: Option<i64> = None;
    let mut min_tail = i64::MAX;
    for k in 0..buckets - 1 {
        let e = model.new_int_var_with_name([(0, horizon - 1)], format!("prefix_end_{}", k));
        added.new_vars += 1;
        if let Some(&prev) = prefix_end.last() {
            added.ge(model, format!("prefix_end_chain_{}", k), vec![(1, e), (-1, prev)], 0, None);
        }
        for &i in &members[k] {
            let t = &ctx.times[i];
            // (variable, cycles by which a reader can be early: a reader on the vselect path
            // may run in the multiply_add cycle of a select)
            let mut finish: Vec<(IntVar, i64)> = vec![(t.issue(), 0)];
            finish.extend(t.completion().map(|c| (c, 0)));
            finish.extend(t.fma_cycle().map(|f| (f, 1)));
            for (j, (v, early)) in finish.into_iter().enumerate() {
                added.ge(model, format!("prefix_end_{}_ge_op{}_{}", k, i, j), vec![(1, e), (-1, v)], 0, None);
                min_tail = min_tail.min((paths.tail[i] - early).max(0));
                if let Some(h) = hint(v) {
                    prefix_hint = Some(prefix_hint.map_or(h, |p| p.max(h)));
                }
            }
        }
        // The op whose finish term is the largest has at least min_tail cycles after it.
        let mt = if min_tail == i64::MAX { 0 } else { min_tail };
        added.ge(model, format!("makespan_ge_prefix_end_{}", k), vec![(1, m), (-1, e)], 1 + mt, None);
        if let Some(h) = prefix_hint {
            added.hints.push((e, h.clamp(0, horizon - 1)));
        }
        prefix_end.push(e);
    }
    // suffix_start_k (k = 1..buckets-1): at most every start of buckets k.. (created last
    // bucket first so each chain row refers to an existing variable).
    let mut suffix_start_rev: Vec<IntVar> = Vec::new();
    let mut suffix_hint: Option<i64> = None;
    for k in (1..buckets).rev() {
        let lo = k as i64 * BUCKET_CYCLES;
        let s = model.new_int_var_with_name([(lo, horizon - 1)], format!("suffix_start_{}", k));
        added.new_vars += 1;
        if let Some(&next) = suffix_start_rev.last() {
            added.ge(model, format!("suffix_start_chain_{}", k), vec![(1, next), (-1, s)], 0, None);
        }
        for &i in &members[k] {
            let t = &ctx.times[i];
            match t.shift() {
                Some(shift) => {
                    for (lane_i, &lane) in t.lanes().iter().enumerate() {
                        added.ge(model, format!("suffix_start_{}_le_op{}_lane{}", k, i, lane_i), vec![(1, lane), (-1, s)], 0, Some(shift));
                    }
                    added.ge(model, format!("suffix_start_{}_le_op{}", k, i), vec![(1, t.issue()), (-1, s)], 0, Some(!shift));
                    let start = match hinted_true(shift) {
                        Some(true) => t.lanes().iter().filter_map(|&l| hint(l)).min(),
                        Some(false) => hint(t.issue()),
                        None => None,
                    };
                    if let Some(h) = start {
                        suffix_hint = Some(suffix_hint.map_or(h, |p| p.min(h)));
                    }
                }
                None => {
                    added.ge(model, format!("suffix_start_{}_le_op{}", k, i), vec![(1, t.issue()), (-1, s)], 0, None);
                    if let Some(h) = hint(t.issue()) {
                        suffix_hint = Some(suffix_hint.map_or(h, |p| p.min(h)));
                    }
                }
            }
        }
        if let Some(h) = suffix_hint {
            added.hints.push((s, h.clamp(lo, horizon - 1)));
        }
        suffix_start_rev.push(s);
    }
    let suffix_start = |k: usize| suffix_start_rev[buckets - 1 - k];

    // Energy rows, the work of a prefix or suffix accumulated in chained auxiliaries.
    for engine in RowEngine::ALL {
        let cap = caps.of(engine);
        if cap <= 0 {
            continue;
        }
        let bucket_work: Vec<LinearWork> = members
            .iter()
            .map(|ms| {
                let mut w = LinearWork::default();
                for &i in ms {
                    if engine == RowEngine::Joint {
                        w.add(&LinearWork::constant(min_work(ctx.ops, i, engine, &ctx.classes)));
                    } else {
                        w.add(&op_work(ctx, i, engine));
                    }
                }
                w
            })
            .collect();
        let max_total: i64 = bucket_work.iter().map(LinearWork::max_value).sum();
        if max_total == 0 {
            continue;
        }
        let constant_only = bucket_work.iter().all(|w| !w.has_literals());
        let name = engine.name().replace('+', "_");
        for (side, range) in [("prefix", (0..buckets - 1).collect::<Vec<_>>()), ("suffix", (1..buckets).rev().collect())] {
            let mut acc = 0i64; // constant work so far
            let mut prev: Option<(IntVar, Option<i64>)> = None;
            for k in range {
                // Span rows: prefix: cap * (prefix_end_k + 1) >= work; suffix: cap * (makespan - suffix_start_k) >= work.
                let span: Vec<(i64, IntVar)> = if side == "prefix" {
                    vec![(cap, prefix_end[k])]
                } else {
                    vec![(cap, m), (-cap, suffix_start(k))]
                };
                let span_const = if side == "prefix" { cap } else { 0 };
                if constant_only {
                    acc += bucket_work[k].constant;
                    added.ge(model, format!("{}_energy_{}_{}", side, name, k), span, acc - span_const, None);
                    continue;
                }
                // aux_k = aux_prev + work_k
                let aux = model.new_int_var_with_name([(0, max_total)], format!("{}_work_{}_{}", side, name, k));
                added.new_vars += 1;
                let (bt, bc) = work_terms(&bucket_work[k], -1);
                let mut terms = vec![(1, aux)];
                if let Some((p, _)) = prev {
                    terms.push((-1, p));
                }
                terms.extend(bt);
                let negated: Vec<(i64, IntVar)> = terms.iter().map(|&(c, v)| (-c, v)).collect();
                // aux - prev - work_terms >= work_const and <= work_const
                added.ge(model, format!("{}_work_{}_{}_ge", side, name, k), terms, -bc, None);
                added.ge(model, format!("{}_work_{}_{}_le", side, name, k), negated, bc, None);
                let mut energy = span;
                energy.push((-1, aux));
                added.ge(model, format!("{}_energy_{}_{}", side, name, k), energy, -span_const, None);
                let aux_hint = match (prev, bucket_work[k].eval(hint)) {
                    (None, Some(w)) => Some(w),
                    (Some((_, Some(p))), Some(w)) => Some(p + w),
                    _ => None,
                };
                if let Some(h) = aux_hint {
                    added.hints.push((aux, h.clamp(0, max_total)));
                }
                prev = Some((aux, aux_hint));
            }
        }
    }
    added
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Operand, VReg};

    fn op(id: usize, engine: Engine, opcode: Opcode, deps: &[usize]) -> SsaOp {
        let mut o = SsaOp::new(id, engine, opcode, Some(VReg::scalar(id as u32)), vec![Operand::VReg(VReg::scalar(1000))]);
        o.dep_ids = deps.iter().copied().collect();
        o
    }

    /// Heads count the longest path, tails exclude the op's own cycle, and an eliminable
    /// vbroadcast passes its head on with latency 0.
    #[test]
    fn test_static_paths() {
        let ops = vec![
            op(0, Engine::Load, Opcode::Load, &[]),
            op(1, Engine::Valu, Opcode::Vbroadcast, &[0]),
            op(2, Engine::Valu, Opcode::Add, &[1]),
            op(3, Engine::Store, Opcode::Store, &[2]),
        ];
        let p = static_paths(&ops, &HashSet::new());
        assert_eq!(p.head, vec![0, 1, 2, 3]);
        assert_eq!(p.tail, vec![3, 2, 1, 0]);
        let p = static_paths(&ops, &[1].into_iter().collect());
        assert_eq!(p.head, vec![0, 1, 1, 2]);
        assert_eq!(p.tail, vec![2, 1, 1, 0]);
    }

    /// One op: makespan 1 (tail 0, head 0, one slot), not 2.
    #[test]
    fn test_static_bound_single_op() {
        let ops = vec![op(0, Engine::Alu, Opcode::Add, &[])];
        let empty = HashSet::new();
        let classes = OpClasses { shiftable: &empty, eliminable_vbs: &empty };
        let caps = RowCaps { valu: 6, alu: 12, load: 2, store: 2, flow: 1 };
        let p = static_paths(&ops, &empty);
        assert_eq!(static_bound(&ops, &p, &classes, &caps).map(|b| b.bound), Some(1));
    }

    /// Two chained loads, then four loads that need both: the four start at cycle 2 and need
    /// two cycles on two load slots, so the bound is 2 + 2 = 4, above the global ceil(6/2) = 3.
    #[test]
    fn test_static_bound_head_window() {
        let mut ops = vec![op(0, Engine::Load, Opcode::Load, &[]), op(1, Engine::Load, Opcode::Load, &[0])];
        for i in 2..6 {
            ops.push(op(i, Engine::Load, Opcode::Load, &[1]));
        }
        let empty = HashSet::new();
        let classes = OpClasses { shiftable: &empty, eliminable_vbs: &empty };
        let caps = RowCaps { valu: 6, alu: 12, load: 2, store: 2, flow: 1 };
        let p = static_paths(&ops, &empty);
        let b = static_bound(&ops, &p, &classes, &caps).unwrap();
        // head >= 1 (five loads, 1 + 3) ties with head >= 2 (2 + 2); the first one found is kept.
        assert_eq!((b.bound, b.engine, b.t, b.q), (4, RowEngine::Load, 1, 0));
    }
}
