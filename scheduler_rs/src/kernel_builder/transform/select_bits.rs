//! Rule `select-bits-path`: what `--select-bit-source path` emits, as a rewrite of the
//! finished program (catalogue rule 11). Reads no emitter state.
//!
//! With `--select-bit-source index` (the default) a bit-driven tree access at depth `d >= 3`
//! reads its `d - 1` older branch bits back from the index before the last update (AND and
//! shift right by 1, after one SUB `L_(d-1) - idx` for a zero-based index;
//! `R15-bits-from-index`). With `path` it reads the branch bits themselves (the `v & 1` of each
//! earlier index step), which stay live. This rule finds every value that the ops prove equal
//! to an earlier branch bit and makes its readers read that bit; what only fed those values
//! (the extraction, and the SUB) dies with them.
//!
//! **What is proved, from the ops alone.** A forward pass gives each vector value one of two
//! forms, built only from constants shown by propagation (`values::ValueFacts`: the vectors
//! 1, 2 and `L_k = 2^(k+1) - 2`, whether loaded or derived by ALU ops; an annotation is used
//! only where it agrees with the immediate):
//!
//! * a *bit*: `x & 1` (either operand order) is 0 or 1 in every lane (`R6-boolean-forms`),
//!   whatever `x` is; it starts the form `Pos([b])` with `b` its own result;
//! * `Pos(b_0 .. b_(k-1))`: the value is `sum b_j 2^(k-1-j)` (the mirrored index after `k`
//!   steps, R9), each `b_j` a bit's VReg, `k <= 31`:
//!   - a value shown to be 0 (`values::ValueFacts`) is `Pos([])`;
//!   - `multiply_add(P, 2, B)` (factors in either order) with `P = Pos(L)` and the addend
//!     `B = Pos([b])` is `Pos(L ++ [b])`;
//!   - `Pos(L) >> 1` is `Pos(L` without its last bit`)`;
//!   - `Pos(L) & 1` is `Pos([last of L])`;
//!   - `L_k - Zero(L)` with `k = len(L)` is `Pos(L)`;
//! * `Zero(b_0 .. b_(k-1))`: the value is `L_k - Pos(L)` (the zero-based index after `k` steps
//!   from 0, R10):
//!   - a value shown to be 0 is `Zero([])`;
//!   - `L_k - Pos(L)` with `k = len(L)` is `Zero(L)` (the index offset `2 - b` is `k = 1`);
//!   - `multiply_add(Z, 2, O)` with `Z = Zero(L)` and `O = Zero([b])` is `Zero(L ++ [b])`.
//!
//! Each step is an identity on 32-bit words: `R15-bit-path-steps` (with `R15-bit-path-steps-
//! needs-boolean` showing the 0/1 fact is needed). The forms then follow by induction over the
//! ops in order. A *site* is an op whose form is `Pos([b])` and whose result is not `b`
//! itself: its value equals `b` in every lane, so its readers can read `b`.
//!
//! The zero value at the start of a zero-based chain is, in round 0, the load of the input
//! indices: 0 only under the declared input precondition `Assumption::InitialIndicesZero`
//! (the site then records it); after the wrap it is the broadcast of the zero register
//! (constant propagation). The mirrored chain starts at the bit itself and needs no
//! assumption.
//!
//! Sites whose result is a vector group (or a lane of one), is kept live by another op, or
//! carries a trace id are rejected: rewiring would change what those mean. A program that is
//! not in single assignment is refused as a whole (`Surgery::check`).
//!
//! Cost: the rewrite removes ops and keeps the bits live longer. The ops are in the report;
//! the scratch is not (8 words per group per kept bit, from the round that produced it to the
//! select), which the schedulers see and the floor does not.

use std::collections::BTreeMap;

use crate::types::{Engine, Opcode, VReg};

use super::super::SsaProgram;
use super::graph::SsaGraph;
use super::report::{dce_counts, RuleReport};
use super::rule::{AlternativeSpec, Fact, FactSource, Proposal, Rejection, Rule};
use super::surgery::Surgery;
use super::values::{Assumption, MemoryLayout, ValueFacts};

/// Rule id.
pub const SELECT_BITS_RULE: &str = "select-bits-path";

/// Alternatives of a site: the value recomputed from the index, or the branch bit read.
pub const SELECT_BITS_ALTERNATIVES: [AlternativeSpec; 2] = [
    AlternativeSpec { name: "index", proofs: &["R15-bits-from-index", "R15-bits-from-index-needs-range"] },
    AlternativeSpec {
        name: "path",
        proofs: &["R15-bit-path-steps", "R15-bit-path-steps-needs-boolean", "R6-boolean-forms"],
    },
];

/// Longest bit path a form tracks: `Pos` stays below `2^31`, so a shift right by 1 drops
/// exactly the last bit and nothing wraps.
pub const MAX_PATH_BITS: usize = 31;

/// `L_k = 2^(k+1) - 2`: the last index of level `k` (zero-based), the mirroring constant.
pub fn level_last(k: usize) -> i64 {
    (1i64 << (k + 1)) - 2
}

/// The form of a value: what it is in terms of earlier branch bits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BitForm {
    /// `sum b_j 2^(k-1-j)` over the listed bits, oldest first.
    Pos(Vec<VReg>),
    /// `L_k - Pos(bits)`.
    Zero(Vec<VReg>),
}

/// A form and whether it relies on `InitialIndicesZero`.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Known {
    form: BitForm,
    assumed: bool,
}

/// The bit-path analysis of a program: the form of every op's result, if any.
pub struct BitPaths {
    forms: Vec<Option<Known>>,
    /// For rejection reasons at the base of a chain.
    layout: MemoryLayout,
    assumptions: Vec<Assumption>,
}

impl BitPaths {
    /// Compute the forms in op order.
    ///
    /// # Arguments
    /// * `graph` - The program.
    /// * `values` - Its value facts (zero values, with the assumption they rely on).
    /// * `layout`, `assumptions` - What `values` was computed with (for rejection reasons).
    ///
    /// # Returns
    /// The analysis.
    pub fn compute(graph: &SsaGraph, values: &ValueFacts, layout: &MemoryLayout, assumptions: &[Assumption]) -> Self {
        let mut paths =
            BitPaths { forms: vec![None; graph.ops.len()], layout: *layout, assumptions: assumptions.to_vec() };
        for i in 0..graph.ops.len() {
            paths.forms[i] = paths.evaluate(graph, values, i).ok();
        }
        paths
    }

    /// The form of `v` (`None` if unknown or `v` is not single assignment).
    pub fn form(&self, graph: &SsaGraph, values: &ValueFacts, v: VReg) -> Option<BitForm> {
        self.known(graph, values, v).map(|k| k.form)
    }

    fn known(&self, graph: &SsaGraph, values: &ValueFacts, v: VReg) -> Option<Known> {
        if graph.is_redefined(v) || !v.is_vector() {
            return None;
        }
        if values.value(graph, v) == Some(0) {
            // A zero is Pos([]) and Zero([]) at once; `Pos` is returned and `zero_form` below
            // accepts it as `Zero([])` too.
            return Some(Known { form: BitForm::Pos(vec![]), assumed: values.relies_on_assumption(graph, v) });
        }
        graph.def_index(v).and_then(|i| self.forms[i].clone())
    }

    /// `v` as `Pos`, with the assumption flag.
    fn pos(&self, graph: &SsaGraph, values: &ValueFacts, v: VReg) -> Option<(Vec<VReg>, bool)> {
        match self.known(graph, values, v)? {
            Known { form: BitForm::Pos(l), assumed } => Some((l, assumed)),
            _ => None,
        }
    }

    /// `v` as `Zero` (a zero value is `Zero([])`), with the assumption flag.
    fn zero_form(&self, graph: &SsaGraph, values: &ValueFacts, v: VReg) -> Option<(Vec<VReg>, bool)> {
        match self.known(graph, values, v)? {
            Known { form: BitForm::Zero(l), assumed } => Some((l, assumed)),
            Known { form: BitForm::Pos(l), assumed } if l.is_empty() => Some((l, assumed)),
            _ => None,
        }
    }

    /// The form of op `i`'s result from the forms before it, or why it has none (the first
    /// failing step, for rejections).
    fn evaluate(&self, graph: &SsaGraph, values: &ValueFacts, i: usize) -> Result<Known, String> {
        let op = &graph.ops[i];
        let Some(dest) = op.dest else { return Err("op without a dest".into()) };
        if graph.is_redefined(dest) {
            return Err(format!("{:?} is defined more than once", dest));
        }
        if op.engine != Engine::Valu || !dest.is_vector() {
            return Err(format!("{:?} {:?} is not a vector ALU op", op.engine, op.opcode));
        }
        let srcs: Vec<VReg> = match op.srcs.iter().map(|s| s.as_vreg()).collect::<Option<Vec<_>>>() {
            Some(s) => s,
            None => return Err(format!("{:?} has an immediate operand", op.opcode)),
        };
        // Constants by propagation (`ValueFacts`): folds the ALU-derived constants the emitter
        // builds (e.g. `L_2 = 6` as `3 + 3`) and trusts an annotation only where it agrees
        // with the immediate.
        let konst = |v: VReg| values.value(graph, v);
        let why = |v: VReg| self.why_no_form(graph, values, v);
        match (op.opcode, srcs.as_slice()) {
            (Opcode::And, &[a, b]) => {
                let x = if konst(b) == Some(1) {
                    a
                } else if konst(a) == Some(1) {
                    b
                } else {
                    return Err(format!("AND with neither operand the constant 1 ({:?}, {:?})", konst(a), konst(b)));
                };
                match self.pos(graph, values, x) {
                    // Pos(L) & 1 is the last bit of L.
                    Some((l, assumed)) if !l.is_empty() => {
                        Ok(Known { form: BitForm::Pos(vec![*l.last().unwrap()]), assumed })
                    }
                    // Anything & 1 is a bit (R6): its own result.
                    _ => Ok(Known { form: BitForm::Pos(vec![dest]), assumed: false }),
                }
            }
            (Opcode::Shr, &[x, k]) => {
                if konst(k) != Some(1) {
                    return Err(format!("shift right by {:?}, not by the constant 1", konst(k)));
                }
                match self.pos(graph, values, x) {
                    Some((mut l, assumed)) if !l.is_empty() => {
                        l.pop();
                        Ok(Known { form: BitForm::Pos(l), assumed })
                    }
                    Some(_) => Err("shift right of a zero".into()),
                    None => Err(format!("shifted value is not a position of branch bits: {}", why(x))),
                }
            }
            (Opcode::Sub, &[c, x]) => {
                let Some(c) = konst(c) else { return Err("SUB minuend is not a broadcast constant".into()) };
                if let Some((l, assumed)) = self.zero_form(graph, values, x).filter(|(l, _)| !l.is_empty()) {
                    return if c == level_last(l.len()) {
                        Ok(Known { form: BitForm::Pos(l), assumed })
                    } else {
                        Err(format!("SUB constant {} is not L_{} = {}", c, l.len(), level_last(l.len())))
                    };
                }
                match self.pos(graph, values, x) {
                    Some((l, assumed)) if !l.is_empty() => {
                        if c == level_last(l.len()) {
                            Ok(Known { form: BitForm::Zero(l), assumed })
                        } else {
                            Err(format!("SUB constant {} is not L_{} = {}", c, l.len(), level_last(l.len())))
                        }
                    }
                    _ => Err(format!("SUB subtrahend has no bit form: {}", why(x))),
                }
            }
            (Opcode::MultiplyAdd, &[a, b, c]) => {
                let x = if konst(b) == Some(2) {
                    a
                } else if konst(a) == Some(2) {
                    b
                } else {
                    return Err(format!("multiply_add with no factor the constant 2 ({:?}, {:?})", konst(a), konst(b)));
                };
                // Mirrored: 2 Pos(L) + Pos([b]).
                if let (Some((l, a1)), Some((bit, a2))) = (self.pos(graph, values, x), self.pos(graph, values, c)) {
                    if bit.len() == 1 && l.len() < MAX_PATH_BITS {
                        let mut l = l;
                        l.push(bit[0]);
                        return Ok(Known { form: BitForm::Pos(l), assumed: a1 || a2 });
                    }
                }
                // Zero-based: 2 Zero(L) + Zero([b]).
                if let (Some((l, a1)), Some((bit, a2))) =
                    (self.zero_form(graph, values, x), self.zero_form(graph, values, c))
                {
                    if bit.len() == 1 && l.len() < MAX_PATH_BITS {
                        let mut l = l;
                        l.push(bit[0]);
                        return Ok(Known { form: BitForm::Zero(l), assumed: a1 || a2 });
                    }
                }
                Err(format!(
                    "multiply_add is neither 2 Pos + bit nor 2 Zero + (2 - bit): factor: {}; addend: {}",
                    why(x),
                    why(c)
                ))
            }
            _ => Err(format!("{:?} is not AND, SHR, SUB or multiply_add", op.opcode)),
        }
    }

    /// Why `v` has no form: the first failing step down its definition.
    pub fn why_no_form(&self, graph: &SsaGraph, values: &ValueFacts, v: VReg) -> String {
        if graph.is_redefined(v) {
            return format!("{:?} is defined more than once", v);
        }
        if let Some(k) = self.known(graph, values, v) {
            return format!("has the form {:?}", k.form);
        }
        let Some(i) = graph.def_index(v) else { return "no definer (vector group or lane)".into() };
        match self.evaluate(graph, values, i) {
            Ok(_) => "has a form".into(),
            Err(e) => {
                let op = &graph.ops[i];
                if matches!(op.opcode, Opcode::Vload | Opcode::Load) {
                    // The base of a zero-based chain: say why it is not shown to be 0.
                    return format!("chain base: {}", values.why_not_zero(graph, v, &self.layout, &self.assumptions));
                }
                e
            }
        }
    }
}

/// The select-bits-path rule.
#[derive(Clone, Debug)]
pub struct SelectBitsRule {
    /// Where the input indices are.
    pub layout: MemoryLayout,
    /// Declared assumptions (`InitialIndicesZero` lets a zero-based chain start at the
    /// loaded input index).
    pub assumptions: Vec<Assumption>,
}

impl SelectBitsRule {
    /// The rule for the problem's layout, with the input precondition declared.
    ///
    /// # Arguments
    /// * `batch_size` - Lanes of the problem.
    ///
    /// # Returns
    /// The rule, assuming `InitialIndicesZero`.
    pub fn for_problem(batch_size: usize) -> Self {
        SelectBitsRule { layout: MemoryLayout::problem(batch_size), assumptions: vec![Assumption::InitialIndicesZero] }
    }
}

/// A matched site: an op whose value is proved equal to an earlier branch bit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectBitsSite {
    /// The op.
    pub op: usize,
    /// Its result.
    pub output: VReg,
    /// The branch bit it equals (readers of `output` read this instead).
    pub bit: VReg,
    /// The op defining `bit`.
    pub bit_def: usize,
    /// Round label (tag), if any.
    pub round: Option<u32>,
    /// Group label (tag), if any.
    pub group: Option<u32>,
    /// Facts the match relies on.
    pub facts: Vec<Fact>,
}

impl SelectBitsRule {
    /// Match a site at op `i`.
    ///
    /// # Arguments
    /// * `graph` - The program.
    /// * `values` - Its value facts (with this rule's layout and assumptions).
    /// * `paths` - Its bit-path forms.
    /// * `i` - Candidate op.
    ///
    /// # Returns
    /// The site, or why op `i` is not one.
    pub fn match_at(&self, graph: &SsaGraph, values: &ValueFacts, paths: &BitPaths, i: usize) -> Result<SelectBitsSite, Rejection> {
        let reject = |s: String| Err(Rejection(s));
        let op = &graph.ops[i];
        let Some(output) = op.dest else { return reject("op without a dest".into()) };
        let known = match paths.evaluate(graph, values, i) {
            Ok(k) => k,
            Err(e) => return reject(e),
        };
        let bit = match &known.form {
            BitForm::Pos(l) if l.len() == 1 => l[0],
            BitForm::Pos(l) => return reject(format!("a position of {} bits, not one bit", l.len())),
            BitForm::Zero(l) => return reject(format!("a zero-based index of {} bits, not a bit", l.len())),
        };
        if bit == output {
            return reject("a branch bit itself (x & 1)".into());
        }
        if graph.group_lanes(output).is_some() || graph.element_of(output).is_some() {
            return reject("result is a vector group (surgery cannot rewire lanes)".into());
        }
        if graph.is_kept_live(output) {
            return reject("result is kept live by another op (keep_live)".into());
        }
        if op.trace_id.is_some() {
            return reject("op carries a trace id".into());
        }
        if op.keep_live.is_some() {
            return reject("op keeps other values live (keep_live)".into());
        }
        let Some(bit_def) = graph.def_index(bit) else { return reject("the bit has no definer".into()) };
        let label = |f: fn(&crate::types::RegionTag) -> Option<u32>| op.region.and_then(|t| f(&t));
        let mut facts = vec![
            Fact { name: "equals_bit", value: format!("{:?} op {} equals the bit of op {}", op.opcode, i, bit_def), source: FactSource::Structure },
            Fact { name: "boolean", value: format!("op {} is x & 1 (R6)", bit_def), source: FactSource::Structure },
        ];
        if known.assumed {
            facts.push(self.layout.fact());
            facts.push(Assumption::InitialIndicesZero.fact());
        }
        Ok(SelectBitsSite { op: i, output, bit, bit_def, round: label(|t| t.round), group: label(|t| t.group), facts })
    }

    /// Candidate ops: vector AND and SHR whose non-constant operand is defined by a
    /// multiply_add, SUB, SHR or AND (an index or a value derived from one), and SUB of a
    /// constant (the zero-based mirroring). Branch bits `v & 1` of a hash output are not
    /// candidates.
    fn is_candidate(graph: &SsaGraph, i: usize) -> bool {
        let op = &graph.ops[i];
        if op.engine != Engine::Valu || !matches!(op.opcode, Opcode::And | Opcode::Shr | Opcode::Sub | Opcode::MultiplyAdd) {
            return false;
        }
        op.srcs.iter().filter_map(|s| s.as_vreg()).any(|v| {
            graph.def_op(v).map_or(false, |d| {
                d.engine == Engine::Valu && matches!(d.opcode, Opcode::MultiplyAdd | Opcode::Sub | Opcode::Shr | Opcode::And)
            })
        })
    }

    /// Every site of `graph` in op order, and rejected candidates counted by reason.
    ///
    /// # Arguments
    /// * `graph` - The program.
    /// * `zero` - The program's zero register.
    ///
    /// # Returns
    /// `(sites, rejections)`. Candidates whose form is a position or a zero-based index of
    /// more than one bit are intermediate steps, not rejections, and are not counted.
    pub fn find_sites(&self, graph: &SsaGraph, zero: Option<VReg>) -> (Vec<SelectBitsSite>, BTreeMap<String, usize>) {
        let values = ValueFacts::compute(graph, zero, &self.layout, &self.assumptions);
        let paths = BitPaths::compute(graph, &values, &self.layout, &self.assumptions);
        let mut sites = Vec::new();
        let mut rejected: BTreeMap<String, usize> = BTreeMap::new();
        for p in self.propose(graph) {
            for &i in &p.ops {
                match self.match_at(graph, &values, &paths, i) {
                    Ok(s) => sites.push(s),
                    Err(e) if e.0.starts_with("a position of") || e.0.starts_with("a zero-based index of") => {}
                    Err(e) => *rejected.entry(e.0).or_default() += 1,
                }
            }
        }
        (sites, rejected)
    }
}

impl Rule for SelectBitsRule {
    type Site = SelectBitsSite;

    fn id(&self) -> &'static str {
        SELECT_BITS_RULE
    }

    fn alternatives(&self) -> &'static [AlternativeSpec] {
        &SELECT_BITS_ALTERNATIVES
    }

    /// One proposal per candidate op (`is_candidate`), in op order, from the ops alone.
    fn propose(&self, graph: &SsaGraph) -> Vec<Proposal> {
        (0..graph.ops.len())
            .filter(|&i| Self::is_candidate(graph, i))
            .map(|i| {
                let t = graph.ops[i].region;
                Proposal {
                    region_id: t.map(|t| t.id),
                    round: t.and_then(|t| t.round),
                    group: t.and_then(|t| t.group),
                    depth: t.and_then(|t| t.depth),
                    ops: vec![i],
                }
            })
            .collect()
    }

    /// Computes the facts for every call; use `find_sites` for a whole program.
    fn match_site(&self, graph: &SsaGraph, proposal: &Proposal) -> Result<SelectBitsSite, Rejection> {
        let values = ValueFacts::compute(graph, None, &self.layout, &self.assumptions);
        let paths = BitPaths::compute(graph, &values, &self.layout, &self.assumptions);
        let mut last = Rejection("empty proposal".into());
        for &i in &proposal.ops {
            match self.match_at(graph, &values, &paths, i) {
                Ok(s) => return Ok(s),
                Err(e) => last = e,
            }
        }
        Err(last)
    }
}

/// Rewrite every select-bits-path site of `program` chosen by `select`: readers of the site's
/// value read the branch bit it equals; what only fed the site dies.
///
/// # Arguments
/// * `program` - A finished program (before dependency computation).
/// * `rule` - The rule.
/// * `select` - Which matched sites to rewrite.
///
/// # Returns
/// The rewritten program and the report (per site the op count delta is 0: the site's op dies
/// in the cascade, with the extraction ops that only fed it).
pub fn apply_select_bits(
    program: SsaProgram,
    rule: &SelectBitsRule,
    select: impl Fn(&SelectBitsSite) -> bool,
) -> Result<(SsaProgram, RuleReport), String> {
    let (sites, rejections) = {
        let graph = SsaGraph::new(&program.ops, &program.vector_group_lanes);
        rule.find_sites(&graph, program.zero_vreg)
    };
    let sites: Vec<SelectBitsSite> = sites.into_iter().filter(|s| select(s)).collect();
    let mut report = RuleReport { rule: SELECT_BITS_RULE, rejections, ..Default::default() };
    if sites.is_empty() {
        return Ok((program, report));
    }
    if let Err(e) = Surgery::check(&program) {
        report.refused = Some(e);
        return Ok((program, report));
    }
    let before = dce_counts(&program).map_err(|e| report.cost_error = Some(e)).ok();
    let mut surgery = Surgery::new(program)?;
    for s in &sites {
        *report.sites.entry("bit-read").or_default() += 1;
        // The site's op is not removed by the rule but by the cascade, with the extraction
        // ops that only fed it: the per-site delta is 0 and the saving is in the cascade.
        *report.site_classes.entry(("bit-read", [0; 10])).or_default() += 1;
        surgery.replace_uses(s.output, s.bit);
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
