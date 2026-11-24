//! Rule `mirror-index`: what `--index-form mirrored` emits, as a rewrite of the finished
//! zero-based program (catalogue rule 8). Reads no emitter state. A global representation
//! choice: it rewrites every index value of the program or none.
//!
//! **Which values are indices, and at which depth, from the ops.** The bit-path analysis of
//! `select_bits` (`BitPaths`) gives a value the form `Zero(b_0 .. b_(k-1))` when it is shown
//! to equal `L_k - Pos(b)`, `L_k = 2^(k+1) - 2`, with `Pos(b) = sum b_j 2^(k-1-j)` over
//! branch bits `b_j = x_j & 1`. An *index step* is `y = multiply_add(x, 2, o)` (factors in
//! either order) with `x = Zero(L)` (a value shown to be 0 is `Zero([])`) and `o = Zero([b])`
//! (the offset `2 - b`), so `y = Zero(L ++ [b])`. The index values are the steps' first
//! factors and results, and `2 - b` values read as anything other than a step's addend (the
//! depth-0 index of `--prune-dead-index`). The depth of an index value is its `k`: the number
//! of steps since the zero it starts from, counted on the ops, not read from a tag.
//!
//! The zero at the start of the first pass is the load of the input indices: 0 only under the
//! declared input precondition `Assumption::InitialIndicesZero` (without it no chain starts
//! there and the rule finds nothing). After the wrap it is the broadcast of the zero register.
//!
//! **The mirrored index.** `n(y) = Pos(L)` for an index value `y = Zero(L)`: `n = b` after the
//! first step from zero, `n' = multiply_add(n, 2, b)` after a later one, and no value (0) at
//! depth 0. Every use of an index value `idx = L_k - n` must be one of these forms, each
//! rewritten to read `n` (`R10-mirror-uses-general`, symbolic in every constant):
//!
//! | use of `idx` (depth `k`)        | rewritten                                    |
//! |---------------------------------|----------------------------------------------|
//! | first factor of the next step   | `multiply_add(n, 2, b)` (or `b` from depth 0)|
//! | `idx + vbroadcast(s)` (gather)  | `vbroadcast(s + L_k) - n`, one base per visit |
//! | `idx == c`, `c` constant        | `n == L_k - c`                               |
//! | `idx - c`, `c` constant         | `(L_k - c) - n`                              |
//! | `L_k - idx`                     | `n`                                          |
//! | stored to the input indices     | `L_k - n` (`vbroadcast(0)` at depth 0)       |
//!
//! Any other use (another opcode, an index used as a step's addend, a gather or compare at
//! depth 0, a vector group, `keep_live`, a select's delta) refuses the whole program with the
//! reason, so the rewrite never leaves a mix of the two forms.
//!
//! New ops are placed where the emitter places them, so the result can be compared op for op
//! with the `--index-form mirrored` emission: the mirrored step at the zero-based step; the
//! gather base (constant `L_k`, `s + L_k`, its broadcast) before the first gather address of
//! each visit (a visit: consecutive gather addresses of one depth with no index step between
//! them; the emitter does not share a base between visits); constants before their first use
//! (as the emitter's `v_constant`); the final `L_k - n` after the last op computing a stored
//! value, in store order.

use std::collections::BTreeMap;

use crate::types::{Engine, Opcode, Operand, SsaOp, VReg, VarType};

use super::super::SsaProgram;
use super::graph::SsaGraph;
use super::report::{dce_counts, RuleReport};
use super::rule::{Fact, FactSource};
use super::select_bits::{level_last, BitForm, BitPaths};
use super::surgery::{Stamp, Surgery};
use super::values::{Assumption, MemoryLayout, ValueFacts};

/// Rule id.
pub const MIRROR_INDEX_RULE: &str = "mirror-index";

/// Proofs the rewrite relies on.
pub const MIRROR_INDEX_PROOFS: [&str; 4] =
    ["R15-bit-path-steps", "R10-mirror-uses-general", "R10-mirrored-index-uses", "R11-mirrored-index-range"];

/// The mirror-index rule.
#[derive(Clone, Debug)]
pub struct MirrorIndexRule {
    /// Where the input indices and values are.
    pub layout: MemoryLayout,
    /// Declared assumptions (`InitialIndicesZero`: the first pass starts at the loaded index).
    pub assumptions: Vec<Assumption>,
}

impl MirrorIndexRule {
    /// The rule for the problem's layout, with the input precondition declared.
    pub fn for_problem(batch_size: usize) -> Self {
        MirrorIndexRule { layout: MemoryLayout::problem(batch_size), assumptions: vec![Assumption::InitialIndicesZero] }
    }
}

/// An index step `y = multiply_add(x, 2, o)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexStep {
    /// The op.
    pub op: usize,
    /// First factor (the index before).
    pub index: VReg,
    /// The constant-2 factor.
    pub two: VReg,
    /// The offset `o = 2 - b` (the addend).
    pub offset: VReg,
    /// The branch bit the offset carries.
    pub bit: VReg,
    /// Depth of the index before (steps since its zero).
    pub depth: usize,
}

/// How one reader uses an index value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IndexUse {
    /// First factor of an index step.
    Step,
    /// `idx + vbroadcast(s)`: a gather address; `s` is the scalar.
    Gather { scalar: VReg },
    /// `idx == c` with `idx` at operand `pos`.
    Compare { pos: usize, constant: i64 },
    /// `idx - c`.
    LevelPosition { constant: i64 },
    /// `L_k - idx`.
    Mirror,
    /// Stored to the input indices.
    Store,
    /// A reader of a depth-0 index (a value shown to be 0) that is neither a step nor an index
    /// store: it reads a constant 0, which the rewrite does not change.
    Constant,
}

/// Everything the rewrite needs: every index value, its depth and bits, every use.
#[derive(Clone, Debug, Default)]
pub struct MirrorPlan {
    /// The index steps, in op order.
    pub steps: Vec<IndexStep>,
    /// Index value -> (depth, bits oldest first).
    pub index_values: BTreeMap<VReg, (usize, Vec<VReg>)>,
    /// (reader op, index value, use), in op order.
    pub uses: Vec<(usize, VReg, IndexUse)>,
    /// Whether some chain starts at the loaded input index (relies on `InitialIndicesZero`).
    pub assumed: bool,
    /// Facts the rewrite relies on.
    pub facts: Vec<Fact>,
}

impl MirrorIndexRule {
    /// Find every index value and classify every use, or refuse.
    ///
    /// # Arguments
    /// * `graph` - The program.
    /// * `zero` - Its zero register.
    ///
    /// # Returns
    /// The plan (empty when the program has no zero-based index step), or why the program
    /// cannot be rewritten as a whole.
    pub fn plan(&self, graph: &SsaGraph, zero: Option<VReg>) -> Result<MirrorPlan, String> {
        let values = ValueFacts::compute(graph, zero, &self.layout, &self.assumptions);
        let paths = BitPaths::compute(graph, &values, &self.layout, &self.assumptions);
        let zero_form = |v: VReg| -> Option<Vec<VReg>> {
            if values.value(graph, v) == Some(0) {
                return Some(vec![]);
            }
            match paths.form(graph, &values, v) {
                Some(BitForm::Zero(l)) => Some(l),
                _ => None,
            }
        };
        let mut plan = MirrorPlan::default();
        // 1. Steps.
        for (i, op) in graph.ops.iter().enumerate() {
            if op.engine != Engine::Valu || op.opcode != Opcode::MultiplyAdd || op.srcs.len() != 3 {
                continue;
            }
            let Some(s) = op.srcs.iter().map(|o| o.as_vreg()).collect::<Option<Vec<_>>>() else { continue };
            let Some(dest) = op.dest else { continue };
            let (x, two) = if values.value(graph, s[1]) == Some(2) {
                (s[0], s[1])
            } else if values.value(graph, s[0]) == Some(2) {
                (s[1], s[0])
            } else {
                continue;
            };
            let (Some(l), Some(o)) = (zero_form(x), zero_form(s[2])) else {
                // Shaped like a step (times 2 plus an offset 2 - b) but the index is not shown to
                // be L_k - Pos: rewriting the other chains would leave this one zero-based, so
                // the program is refused as a whole.
                if zero_form(s[2]).map_or(false, |o| o.len() == 1) && zero_form(x).is_none() {
                    return Err(format!(
                        "op {} is an index step whose index is not shown to be a zero-based index: {}",
                        i,
                        paths.why_no_form(graph, &values, x)
                    ));
                }
                continue;
            };
            if o.len() != 1 || !matches!(paths.form(graph, &values, dest), Some(BitForm::Zero(_))) {
                continue;
            }
            plan.steps.push(IndexStep { op: i, index: x, two, offset: s[2], bit: o[0], depth: l.len() });
            plan.index_values.insert(x, (l.len(), l.clone()));
            let mut l2 = l;
            l2.push(o[0]);
            plan.index_values.insert(dest, (l2.len(), l2));
            if values.relies_on_assumption(graph, x) {
                plan.assumed = true;
            }
        }
        if plan.steps.is_empty() {
            return Ok(plan);
        }
        // `2 - b` values read as anything but a step addend: depth-1 indices (prune form).
        // Zero values stored to the indices: depth-0 final indices.
        let index_stores = self.index_stores(graph, &values);
        for (i, op) in graph.ops.iter().enumerate() {
            let Some(d) = op.dest else { continue };
            if plan.index_values.contains_key(&d) {
                continue;
            }
            let Some(l) = zero_form(d) else { continue };
            let readers = graph.readers(i);
            let stored = readers.iter().any(|r| index_stores.contains(r) && graph.ops[*r].srcs.get(1).and_then(|s| s.as_vreg()) == Some(d));
            // A reader other than a step that reads `d` as its offset.
            let not_offset = readers.iter().any(|&r| !plan.steps.iter().any(|s| s.op == r && s.offset == d));
            if l.len() == 1 && not_offset {
                plan.index_values.insert(d, (1, l));
            } else if l.is_empty() && stored {
                plan.index_values.insert(d, (0, l));
            }
        }
        // 2. Classify every use.
        for (&v, (k, _)) in &plan.index_values {
            if graph.is_kept_live(v) {
                return Err(format!("index value {:?} (depth {}) is kept live by another op", v, k));
            }
            if graph.group_lanes(v).is_some() || graph.element_of(v).is_some() {
                return Err(format!("index value {:?} is a vector group or a lane of one", v));
            }
            if graph.is_redefined(v) {
                return Err(format!("index value {:?} is defined more than once", v));
            }
        }
        // Every index value of depth >= 2 must be a step's result (its n is built there).
        for (&v, &(k, _)) in &plan.index_values {
            if k >= 2 && !plan.steps.iter().any(|s| graph.ops[s.op].dest == Some(v)) {
                return Err(format!("index value {:?} (depth {}) is not the result of an index step", v, k));
            }
        }
        let def_index = |v: VReg| graph.def_index(v);
        for (&v, &(k, _)) in &plan.index_values {
            let Some(d) = def_index(v) else { return Err(format!("index value {:?} has no definer", v)) };
            for &r in graph.readers(d) {
                let op = &graph.ops[r];
                if op.delta_vreg == Some(v) {
                    return Err(format!("index value {:?} is the delta of select op {}", v, r));
                }
                let u = self.classify(graph, &values, &plan, &index_stores, r, v, k)?;
                plan.uses.push((r, v, u));
            }
        }
        plan.uses.sort_by_key(|(r, v, _)| (*r, *v));
        plan.uses.dedup_by_key(|(r, v, _)| (*r, *v));
        plan.facts.push(Fact {
            name: "index_chain",
            value: format!("{} steps, {} index values, depth from the step count", plan.steps.len(), plan.index_values.len()),
            source: FactSource::Structure,
        });
        plan.facts.push(Fact { name: "proofs", value: MIRROR_INDEX_PROOFS.join(", "), source: FactSource::Structure });
        if plan.assumed {
            plan.facts.push(self.layout.fact());
            plan.facts.push(Assumption::InitialIndicesZero.fact());
        }
        Ok(plan)
    }

    /// Indices of vector stores to the input indices (`mem[indices_word] + offset`).
    fn index_stores(&self, graph: &SsaGraph, values: &ValueFacts) -> std::collections::BTreeSet<usize> {
        self.stores_to(graph, values, self.layout.indices_word)
    }

    fn stores_to(&self, graph: &SsaGraph, values: &ValueFacts, word: i64) -> std::collections::BTreeSet<usize> {
        graph
            .ops
            .iter()
            .enumerate()
            .filter(|(_, o)| o.opcode == Opcode::Vstore)
            .filter(|(_, o)| {
                // Inside the array `word` points to (`MemoryLayout::located`).
                o.srcs.first().and_then(|s| s.as_vreg()).and_then(|a| values.address(graph, a)).map_or(false, |a| {
                    a.word == word && self.layout.located(a, 8)
                })
            })
            .map(|(i, _)| i)
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn classify(
        &self,
        graph: &SsaGraph,
        values: &ValueFacts,
        plan: &MirrorPlan,
        index_stores: &std::collections::BTreeSet<usize>,
        r: usize,
        v: VReg,
        k: usize,
    ) -> Result<IndexUse, String> {
        let op = &graph.ops[r];
        let refuse = |why: &str| Err(format!("index value {:?} (depth {}) read by op {} ({:?} {:?}): {}", v, k, r, op.engine, op.opcode, why));
        if let Some(s) = plan.steps.iter().find(|s| s.op == r) {
            return if s.index == v { Ok(IndexUse::Step) } else { refuse("an index used as a step's addend") };
        }
        let srcs: Vec<Option<VReg>> = op.srcs.iter().map(|s| s.as_vreg()).collect();
        let pos: Vec<usize> = srcs.iter().enumerate().filter(|(_, s)| **s == Some(v)).map(|(i, _)| i).collect();
        if pos.len() != 1 {
            return refuse("reads the index more than once or not as an operand");
        }
        let p = pos[0];
        let other = |q: usize| srcs.get(q).copied().flatten();
        match (op.engine, op.opcode, op.srcs.len()) {
            (Engine::Store, Opcode::Vstore, 2) if p == 1 && index_stores.contains(&r) => Ok(IndexUse::Store),
            _ if k == 0 => Ok(IndexUse::Constant),
            (Engine::Valu, Opcode::Add, 2) => {
                match other(1 - p).and_then(|w| graph.broadcast_source(w)) {
                    Some(s) => Ok(IndexUse::Gather { scalar: s }),
                    None => refuse("added to something that is not a vbroadcast"),
                }
            }
            (Engine::Valu, Opcode::Eq, 2) => match other(1 - p).and_then(|w| values.value(graph, w)) {
                Some(c) => Ok(IndexUse::Compare { pos: p, constant: c }),
                None => refuse("compared with a value that is not a constant"),
            },
            (Engine::Valu, Opcode::Sub, 2) if p == 0 => match other(1).and_then(|w| values.value(graph, w)) {
                Some(c) => Ok(IndexUse::LevelPosition { constant: c }),
                None => refuse("minus a value that is not a constant"),
            },
            (Engine::Valu, Opcode::Sub, 2) => match other(0).and_then(|w| values.value(graph, w)) {
                Some(c) if c == level_last(k) => Ok(IndexUse::Mirror),
                Some(c) => refuse(&format!("{} - idx with {} != L_{} = {}", c, c, k, level_last(k))),
                None => refuse("subtracted from a value that is not a constant"),
            },
            _ => refuse("not an R10 form (step, gather address, compare, level position, mirror, index store)"),
        }
    }
}

/// Metadata of an op the emitter creates once per visit, before its groups: the first use's
/// round, depth, phase and region (without the group), and the iteration and group labels the
/// emitter still holds from the op before (it sets them per group and does not always clear
/// them; labels only, never semantics).
fn visit_stamp(op: &SsaOp, prev: Option<&SsaOp>) -> Stamp {
    let stamp = Stamp::of(op);
    let held = prev.and_then(|p| p.source_info.clone());
    Stamp {
        source_info: stamp.source_info.clone().map(|mut s| {
            s.iter = held.as_ref().and_then(|h| h.iter);
            s.group = held.as_ref().and_then(|h| h.group);
            s
        }),
        region: stamp.region.map(|mut t| {
            t.group = None;
            t
        }),
        ..stamp
    }
}

/// The emitter's `v_constant(value)` as seen from op `after`: the scalar is the first scalar
/// defined at or before `after` that holds `value` (the init constants `c[0..15]` come
/// first) when `value <= 15`, else the first `ConstChoice`/`Const` of `value` (the
/// emitter's CSE); the vector is the first broadcast of it.
fn v_constant(surgery: &mut Surgery, graph_values: &[Option<i64>], value: i64, after: usize, stamp: &Stamp) -> VReg {
    let ops = surgery.ops();
    let scalar = if value == 0 {
        None
    } else if value <= 15 {
        (0..=after).find(|&i| ops[i].dest.map_or(false, |d| d.is_scalar()) && graph_values[i] == Some(value)).and_then(|i| ops[i].dest)
    } else {
        None
    };
    let s = match scalar {
        Some(s) => s,
        None => surgery.constant(value, after, &stamp.clone().with_var_type(Some(VarType::Constant))),
    };
    surgery.broadcast(s, after, &stamp.clone().with_var_type(Some(VarType::Constant)))
}

/// Rewrite every index value of `program` to the mirrored form, or none.
///
/// # Arguments
/// * `program` - A finished zero-based program (before dependency computation).
/// * `rule` - The rule.
///
/// # Returns
/// The rewritten program and the report; the program unchanged with `refused` set when some
/// index use is not an R10 form (or the program is not single assignment).
pub fn apply_mirror_index(program: SsaProgram, rule: &MirrorIndexRule) -> Result<(SsaProgram, RuleReport), String> {
    let mut report = RuleReport { rule: MIRROR_INDEX_RULE, ..Default::default() };
    let (plan, op_values, value_store_defs) = {
        let graph = SsaGraph::new(&program.ops, &program.vector_group_lanes);
        let plan = match rule.plan(&graph, program.zero_vreg) {
            Ok(p) => p,
            Err(e) => {
                report.refused = Some(e);
                return Ok((program, report));
            }
        };
        let values = ValueFacts::compute(&graph, program.zero_vreg, &rule.layout, &rule.assumptions);
        let op_values: Vec<Option<i64>> =
            program.ops.iter().map(|o| o.dest.and_then(|d| values.value(&graph, d))).collect();
        let value_stores = rule.stores_to(&graph, &values, rule.layout.values_word);
        let defs: Vec<usize> = value_stores
            .iter()
            .filter_map(|&s| program.ops[s].srcs.get(1).and_then(|x| x.as_vreg()).and_then(|v| graph.def_index(v)))
            .collect();
        (plan, op_values, defs)
    };
    if plan.steps.is_empty() {
        return Ok((program, report));
    }
    if let Err(e) = Surgery::check(&program) {
        report.refused = Some(e);
        return Ok((program, report));
    }
    let before = dce_counts(&program).map_err(|e| report.cost_error = Some(e)).ok();
    let mut surgery = Surgery::new(program)?;
    // n of every index value: None = 0 (depth 0).
    let mut n: BTreeMap<VReg, Option<VReg>> = BTreeMap::new();
    for (&v, (k, bits)) in &plan.index_values {
        if *k == 0 {
            n.insert(v, None);
        } else if *k == 1 {
            n.insert(v, Some(bits[0]));
        }
    }
    // Depth-0 index values (the loaded input index, the wrap round's zeros) that only steps and
    // index stores read go with them. Removing them now also keeps the constant lookups below
    // from returning one: the emitter's CSE never returns the wrap round's forced broadcasts.
    for (&v, &(k, _)) in &plan.index_values {
        if k != 0 {
            continue;
        }
        let only_index = plan.uses.iter().filter(|(_, u, _)| *u == v).all(|(_, _, u)| matches!(u, IndexUse::Step | IndexUse::Store));
        if let (true, Some(d)) = (only_index, surgery.ops().iter().position(|o| o.dest == Some(v))) {
            surgery.remove(d);
        }
    }
    // Steps in op order: their results' n.
    for s in &plan.steps {
        let op = surgery.ops()[s.op].clone();
        let dest = op.dest.unwrap();
        surgery.remove(s.op);
        let prev = n.get(&s.index).copied().flatten();
        let m = match prev {
            None => s.bit,
            Some(p) => {
                let new = surgery.fresh(true);
                let fma = SsaOp::new(0, Engine::Valu, Opcode::MultiplyAdd, Some(new), vec![Operand::VReg(p), Operand::VReg(s.two), Operand::VReg(s.bit)]);
                surgery.insert_before(s.op, fma, &Stamp::of(&op));
                new
            }
        };
        n.insert(dest, Some(m));
        *report.sites.entry("step").or_default() += 1;
    }
    // Visits: consecutive uses of one kind (gather, compare, level position) at one depth with
    // no index step between them. The emitter creates a visit's base and constants once, before
    // its first group (`emit_scattered_access`, `v_index_compare` in the access emitters).
    let step_ops_all: std::collections::BTreeSet<usize> = plan.steps.iter().map(|s| s.op).collect();
    let mut visit_of: Vec<usize> = Vec::with_capacity(plan.uses.len());
    let mut visit_first: Vec<usize> = Vec::new();
    {
        let mut last: Option<(u8, usize, usize)> = None; // (kind, depth, last op)
        for (r, v, u) in &plan.uses {
            let kind = match u {
                IndexUse::Gather { .. } => 1,
                IndexUse::Compare { .. } => 2,
                IndexUse::LevelPosition { .. } => 3,
                _ => 0,
            };
            let k = plan.index_values[v].0;
            let same = last.map_or(false, |(lk, ld, lo)| {
                lk == kind && ld == k && !step_ops_all.iter().any(|&o| o > lo && o < *r)
            });
            if !same {
                visit_first.push(*r);
            }
            visit_of.push(visit_first.len() - 1);
            last = Some((kind, k, *r));
        }
    }
    // Constants of each compare / level-position visit, created at its first use in order of
    // first appearance: (visit, value) -> vector.
    let mut visit_consts: BTreeMap<(usize, i64), VReg> = BTreeMap::new();
    for (j, (_, v, u)) in plan.uses.iter().enumerate() {
        let value = match u {
            IndexUse::Compare { constant, .. } | IndexUse::LevelPosition { constant } => {
                (level_last(plan.index_values[v].0) - constant) & 0xFFFF_FFFF
            }
            _ => continue,
        };
        let vis = visit_of[j];
        if visit_consts.contains_key(&(vis, value)) {
            continue;
        }
        let first = visit_first[vis];
        let stamp = visit_stamp(&surgery.ops()[first], first.checked_sub(1).map(|p| &surgery.ops()[p]));
        let c = v_constant(&mut surgery, &op_values, value, first - 1, &stamp);
        visit_consts.insert((vis, value), c);
    }
    // Uses.
    let anchor = value_store_defs.iter().copied().max();
    let mut visit: Option<(usize, VReg, usize, VReg)> = None; // (visit, scalar, last use op, base)
    for (j, (r, v, u)) in plan.uses.iter().enumerate() {
        let (r, v) = (*r, *v);
        let vis = visit_of[j];
        let k = plan.index_values[&v].0;
        let nv = n.get(&v).copied().flatten();
        let op = surgery.ops()[r].clone();
        let stamp = Stamp::of(&op);
        // The replacement defines the replaced op's VReg (a gather address is decomposed into
        // lanes keyed by it), so readers need no renaming.
        let replace = |surgery: &mut Surgery, opcode: Opcode, srcs: Vec<VReg>| {
            let mut new = op.clone();
            new.opcode = opcode;
            new.srcs = srcs.into_iter().map(Operand::VReg).collect();
            surgery.remove(r);
            surgery.insert_before(r, new, &stamp);
        };
        match u {
            IndexUse::Step | IndexUse::Constant => continue,
            IndexUse::Gather { scalar } => {
                let nv = nv.expect("gather at depth >= 1");
                let same_visit = visit.map_or(false, |(v0, s, _, _)| v0 == vis && s == *scalar);
                let base = if same_visit {
                    visit.unwrap().3
                } else {
                    let base_stamp = visit_stamp(&op, r.checked_sub(1).map(|p| &surgery.ops()[p]));
                    let kst = surgery.constant(level_last(k), r - 1, &base_stamp.clone().with_var_type(Some(VarType::Constant)));
                    let a = surgery.fresh(false);
                    let add = SsaOp::new(0, Engine::Alu, Opcode::Add, Some(a), vec![Operand::VReg(*scalar), Operand::VReg(kst)]);
                    surgery.insert_before(r, add, &base_stamp.clone().with_var_type(Some(VarType::Pointer)));
                    let b = surgery.fresh(true);
                    let bc = SsaOp::new(0, Engine::Valu, Opcode::Vbroadcast, Some(b), vec![Operand::VReg(a)]);
                    surgery.insert_before(r, bc, &base_stamp.clone().with_var_type(None));
                    b
                };
                visit = Some((vis, *scalar, r, base));
                replace(&mut surgery, Opcode::Sub, vec![base, nv]);
                *report.sites.entry("gather").or_default() += 1;
            }
            IndexUse::Compare { pos, constant } => {
                let nv = nv.expect("compare at depth >= 1");
                let c = visit_consts[&(vis, (level_last(k) - constant) & 0xFFFF_FFFF)];
                let srcs = if *pos == 0 { vec![nv, c] } else { vec![c, nv] };
                replace(&mut surgery, Opcode::Eq, srcs);
                *report.sites.entry("compare").or_default() += 1;
            }
            IndexUse::LevelPosition { constant } => {
                let nv = nv.expect("level position at depth >= 1");
                let c = visit_consts[&(vis, (level_last(k) - constant) & 0xFFFF_FFFF)];
                replace(&mut surgery, Opcode::Sub, vec![c, nv]);
                *report.sites.entry("level-position").or_default() += 1;
            }
            IndexUse::Mirror => {
                surgery.remove(r);
                surgery.replace_uses(op.dest.unwrap(), nv.expect("mirror at depth >= 1"));
                *report.sites.entry("mirror").or_default() += 1;
            }
            IndexUse::Store => {
                let at = anchor.ok_or("index store without value stores to anchor the final index")?;
                let fstamp = Stamp::of(&surgery.ops()[at]).with_var_type(Some(VarType::BatchData));
                let value = match nv {
                    None => v_constant(&mut surgery, &op_values, 0, at, &fstamp),
                    Some(nv) => {
                        let last = v_constant(&mut surgery, &op_values, level_last(k), at, &fstamp);
                        let d = surgery.fresh(true);
                        surgery.insert_after(at, SsaOp::new(0, Engine::Valu, Opcode::Sub, Some(d), vec![Operand::VReg(last), Operand::VReg(nv)]), &fstamp);
                        d
                    }
                };
                let mut store = op.clone();
                store.srcs[1] = Operand::VReg(value);
                surgery.insert_before(r, store, &stamp);
                surgery.remove(r);
                *report.sites.entry("store").or_default() += 1;
            }
        }
    }
    let (program, surgery_report) = surgery.finish()?;
    report.surgery = surgery_report;
    if let (Some(before), None) = (before, &report.cost_error) {
        match dce_counts(&program) {
            Ok(after) => report.close(&before, &after),
            Err(e) => report.cost_error = Some(e),
        }
    }
    Ok((program, report))
}
