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
use super::surgery::{Finished, Surgery};
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

/// The bit-path analysis of a program: the form of every op's result, or why it has none.
///
/// Computed once, in op order: an op's result is evaluated from its operands' stored results,
/// so the cost is linear in the program (no recursion). A failure that comes from an operand
/// carries that operand's reason unchanged (the first failing step down the chain); reasons
/// are short and name no VReg, so rejection counts group by cause.
pub struct BitPaths {
    results: Vec<Result<Known, String>>,
}

impl BitPaths {
    /// Compute the forms in op order.
    ///
    /// # Arguments
    /// * `graph` - The program.
    /// * `values` - Its value facts (constants and zero values, with the assumption they rely on).
    /// * `layout`, `assumptions` - What `values` was computed with (for the reason at the base
    ///   of a chain).
    ///
    /// # Returns
    /// The analysis.
    pub fn compute(graph: &SsaGraph, values: &ValueFacts, layout: &MemoryLayout, assumptions: &[Assumption]) -> Self {
        let mut paths = BitPaths { results: Vec::with_capacity(graph.ops.len()) };
        for i in 0..graph.ops.len() {
            let r = paths.evaluate(graph, values, layout, assumptions, i);
            paths.results.push(r);
        }
        paths
    }

    /// The form of `v` (`None` if unknown or `v` is not single assignment).
    pub fn form(&self, graph: &SsaGraph, values: &ValueFacts, v: VReg) -> Option<BitForm> {
        self.known(graph, values, v).ok().map(|k| k.form)
    }

    /// The stored result of op `i`: its form, or why it has none.
    pub fn result(&self, i: usize) -> Result<BitForm, &str> {
        match &self.results[i] {
            Ok(k) => Ok(k.form.clone()),
            Err(e) => Err(e.as_str()),
        }
    }

    /// Whether op `i`'s form relies on `InitialIndicesZero`.
    pub fn assumed(&self, i: usize) -> bool {
        matches!(&self.results[i], Ok(k) if k.assumed)
    }

    fn known(&self, graph: &SsaGraph, values: &ValueFacts, v: VReg) -> Result<Known, String> {
        if graph.is_redefined(v) {
            return Err("a VReg is defined more than once".into());
        }
        if !v.is_vector() {
            return Err("a scalar, not a vector".into());
        }
        if values.value(graph, v) == Some(0) {
            // A zero is Pos([]) and Zero([]) at once; `Pos` is returned and `zero_form` below
            // accepts it as `Zero([])` too.
            return Ok(Known { form: BitForm::Pos(vec![]), assumed: values.relies_on_assumption(graph, v) });
        }
        match graph.def_index(v) {
            Some(i) if i < self.results.len() => self.results[i].clone(),
            Some(_) => Err("read before its definition".into()),
            None => Err("no definer (vector group or lane)".into()),
        }
    }

    /// `v` as `Pos`, with the assumption flag, or why not.
    fn pos(&self, graph: &SsaGraph, values: &ValueFacts, v: VReg) -> Result<(Vec<VReg>, bool), String> {
        match self.known(graph, values, v)? {
            Known { form: BitForm::Pos(l), assumed } => Ok((l, assumed)),
            Known { form: BitForm::Zero(_), .. } => Err("a zero-based index where a position is needed".into()),
        }
    }

    /// `v` as `Zero` (a zero value is `Zero([])`), with the assumption flag, or why not.
    fn zero_form(&self, graph: &SsaGraph, values: &ValueFacts, v: VReg) -> Result<(Vec<VReg>, bool), String> {
        match self.known(graph, values, v)? {
            Known { form: BitForm::Zero(l), assumed } => Ok((l, assumed)),
            Known { form: BitForm::Pos(l), assumed } if l.is_empty() => Ok((l, assumed)),
            Known { form: BitForm::Pos(_), .. } => Err("a position where a zero-based index is needed".into()),
        }
    }

    /// The form of op `i`'s result from the stored results before it, or why it has none.
    fn evaluate(
        &self,
        graph: &SsaGraph,
        values: &ValueFacts,
        layout: &MemoryLayout,
        assumptions: &[Assumption],
        i: usize,
    ) -> Result<Known, String> {
        let op = &graph.ops[i];
        let Some(dest) = op.dest else { return Err("op without a dest".into()) };
        if graph.is_redefined(dest) {
            return Err("a VReg is defined more than once".into());
        }
        if matches!(op.opcode, Opcode::Vload | Opcode::Load) {
            // The base of a zero-based chain (when its value is 0, `known` answers before
            // this): say why it is not shown to be 0.
            return Err(format!("chain base: {}", values.why_not_zero(graph, dest, layout, assumptions)));
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
        // with the immediate. A constant can itself rely on `InitialIndicesZero` (e.g.
        // `loaded_idx + 1`), so every form built with one carries that flag too.
        let konst = |v: VReg| values.value(graph, v);
        let konst_assumed = |v: VReg| values.relies_on_assumption(graph, v);
        match (op.opcode, srcs.as_slice()) {
            (Opcode::And, &[a, b]) => {
                let (x, one) = if konst(b) == Some(1) {
                    (a, b)
                } else if konst(a) == Some(1) {
                    (b, a)
                } else {
                    return Err(format!("AND with neither operand the constant 1 ({:?}, {:?})", konst(a), konst(b)));
                };
                match self.pos(graph, values, x) {
                    // Pos(L) & 1 is the last bit of L.
                    Ok((l, assumed)) if !l.is_empty() => {
                        Ok(Known { form: BitForm::Pos(vec![*l.last().unwrap()]), assumed: assumed || konst_assumed(one) })
                    }
                    // Anything & 1 is a bit (R6): its own result.
                    _ => Ok(Known { form: BitForm::Pos(vec![dest]), assumed: konst_assumed(one) }),
                }
            }
            (Opcode::Shr, &[x, k]) => {
                if konst(k) != Some(1) {
                    return Err(format!("shift right by {:?}, not by the constant 1", konst(k)));
                }
                let (mut l, assumed) = self.pos(graph, values, x)?;
                if l.is_empty() {
                    return Err("shift right of a zero".into());
                }
                l.pop();
                Ok(Known { form: BitForm::Pos(l), assumed: assumed || konst_assumed(k) })
            }
            (Opcode::Sub, &[c, x]) => {
                let Some(cv) = konst(c) else { return Err("SUB minuend is not a constant".into()) };
                let ca = konst_assumed(c);
                match self.known(graph, values, x)? {
                    Known { form: BitForm::Zero(l), assumed } if !l.is_empty() => {
                        if cv == level_last(l.len()) {
                            Ok(Known { form: BitForm::Pos(l), assumed: assumed || ca })
                        } else {
                            Err(format!("SUB constant {} is not L_{} = {}", cv, l.len(), level_last(l.len())))
                        }
                    }
                    Known { form: BitForm::Pos(l), assumed } if !l.is_empty() => {
                        if cv == level_last(l.len()) {
                            Ok(Known { form: BitForm::Zero(l), assumed: assumed || ca })
                        } else {
                            Err(format!("SUB constant {} is not L_{} = {}", cv, l.len(), level_last(l.len())))
                        }
                    }
                    _ => Err("SUB of a zero".into()),
                }
            }
            (Opcode::MultiplyAdd, &[a, b, c]) => {
                let (x, two) = if konst(b) == Some(2) {
                    (a, b)
                } else if konst(a) == Some(2) {
                    (b, a)
                } else {
                    return Err(format!("multiply_add with no factor the constant 2 ({:?}, {:?})", konst(a), konst(b)));
                };
                let ta = konst_assumed(two);
                let extend = |mut l: Vec<VReg>, bit: Vec<VReg>| -> Option<Vec<VReg>> {
                    (bit.len() == 1 && l.len() < MAX_PATH_BITS).then(|| {
                        l.push(bit[0]);
                        l
                    })
                };
                // An operand without a form: its reason is the cause.
                self.known(graph, values, x)?;
                self.known(graph, values, c)?;
                // Mirrored: 2 Pos(L) + Pos([b]).
                if let (Ok((l, a1)), Ok((bit, a2))) = (self.pos(graph, values, x), self.pos(graph, values, c)) {
                    if let Some(l) = extend(l, bit) {
                        return Ok(Known { form: BitForm::Pos(l), assumed: a1 || a2 || ta });
                    }
                }
                // Zero-based: 2 Zero(L) + Zero([b]).
                if let (Ok((l, a1)), Ok((bit, a2))) = (self.zero_form(graph, values, x), self.zero_form(graph, values, c)) {
                    if let Some(l) = extend(l, bit) {
                        return Ok(Known { form: BitForm::Zero(l), assumed: a1 || a2 || ta });
                    }
                }
                Err("multiply_add is neither 2 Pos + bit nor 2 Zero + (2 - bit)".into())
            }
            _ => Err(format!("{:?} is not AND, SHR, SUB or multiply_add", op.opcode)),
        }
    }

    /// Why `v` has no form (its stored reason), or that it has one.
    pub fn why_no_form(&self, graph: &SsaGraph, values: &ValueFacts, v: VReg) -> String {
        match self.known(graph, values, v) {
            Ok(k) => format!("has the form {:?}", k.form),
            Err(e) => e,
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
    /// The program's zero register, for `Rule::match_site` (which gets only the graph);
    /// `find_sites` takes it as an argument and falls back to this.
    pub zero: Option<VReg>,
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
        SelectBitsRule { layout: MemoryLayout::problem(batch_size), assumptions: vec![Assumption::InitialIndicesZero], zero: None }
    }

    /// The same rule for a program whose zero register is `zero` (so `match_site` sees the
    /// zeros `find_sites` sees, e.g. the wrap round's broadcasts of it).
    pub fn with_zero(mut self, zero: Option<VReg>) -> Self {
        self.zero = zero;
        self
    }

    /// The value facts and bit-path forms of `graph`, as every entry point computes them.
    fn analyses(&self, graph: &SsaGraph, zero: Option<VReg>) -> (ValueFacts, BitPaths) {
        let values = ValueFacts::compute(graph, zero, &self.layout, &self.assumptions);
        let paths = BitPaths::compute(graph, &values, &self.layout, &self.assumptions);
        (values, paths)
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
    pub fn match_at(&self, graph: &SsaGraph, _values: &ValueFacts, paths: &BitPaths, i: usize) -> Result<SelectBitsSite, Rejection> {
        let reject = |s: String| Err(Rejection(s));
        let op = &graph.ops[i];
        let Some(output) = op.dest else { return reject("op without a dest".into()) };
        let form = match paths.result(i) {
            Ok(f) => f,
            Err(e) => return reject(e.to_string()),
        };
        let assumed = paths.assumed(i);
        let bit = match &form {
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
        // The rewrite can make the index-derived ops behind the site dead. If one of them is
        // named in another op's keep_live, removing it would leave that list dangling: refuse
        // (conservatively: any index-derived ancestor, whether or not it would die).
        {
            let mut stack: Vec<usize> = vec![i];
            let mut seen = std::collections::BTreeSet::new();
            while let Some(j) = stack.pop() {
                if !seen.insert(j) || j == bit_def {
                    continue;
                }
                let derived = matches!(paths.result(j), Ok(BitForm::Pos(l)) | Ok(BitForm::Zero(l)) if !l.is_empty());
                if !derived {
                    continue;
                }
                if j != i && graph.ops[j].dest.map_or(false, |d| graph.is_kept_live(d)) {
                    return reject("an index-derived op the rewrite may make dead is kept live by another op".into());
                }
                stack.extend(graph.ops[j].srcs.iter().filter_map(|s| s.as_vreg()).filter_map(|v| graph.def_index(v)));
            }
        }
        let label = |f: fn(&crate::types::RegionTag) -> Option<u32>| op.region.and_then(|t| f(&t));
        let mut facts = vec![
            Fact { name: "equals_bit", value: format!("{:?} op {} equals the bit of op {}", op.opcode, i, bit_def), source: FactSource::Structure },
            Fact { name: "boolean", value: format!("op {} is x & 1 (R6)", bit_def), source: FactSource::Structure },
        ];
        if assumed {
            facts.push(self.layout.fact());
            facts.push(Assumption::InitialIndicesZero.fact());
        }
        Ok(SelectBitsSite { op: i, output, bit, bit_def, round: label(|t| t.round), group: label(|t| t.group), facts })
    }

    /// Candidate ops: vector AND, SHR, SUB and multiply_add with an operand defined by one of
    /// those (a cheap structural filter; hash ops pass it too). `find_sites` counts a
    /// rejection only for a candidate with an operand shown to be index-derived
    /// (`index_derived`), so rejections are about index-derived ops, not hash stages.
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

    /// Whether some operand of op `i` is index-derived: a position or zero-based index of at
    /// least one bit, or a chain that fails only at its base (a loaded index not shown to be 0).
    fn index_derived(graph: &SsaGraph, values: &ValueFacts, paths: &BitPaths, i: usize) -> bool {
        graph.ops[i].srcs.iter().filter_map(|s| s.as_vreg()).any(|v| match paths.known(graph, values, v) {
            Ok(Known { form: BitForm::Pos(l), .. }) | Ok(Known { form: BitForm::Zero(l), .. }) => !l.is_empty(),
            Err(e) => e.starts_with("chain base"),
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
        let (values, paths) = self.analyses(graph, zero.or(self.zero));
        let mut sites = Vec::new();
        let mut rejected: BTreeMap<String, usize> = BTreeMap::new();
        for p in self.propose(graph) {
            for &i in &p.ops {
                match self.match_at(graph, &values, &paths, i) {
                    Ok(s) => sites.push(s),
                    Err(e) if e.0.starts_with("a position of") || e.0.starts_with("a zero-based index of") => {}
                    Err(e) if Self::index_derived(graph, &values, &paths, i) => *rejected.entry(e.0).or_default() += 1,
                    Err(_) => {}
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

    /// Computes the analyses for every call (with `self.zero`, as `find_sites` does); use
    /// `find_sites` for a whole program.
    fn match_site(&self, graph: &SsaGraph, proposal: &Proposal) -> Result<SelectBitsSite, Rejection> {
        let (values, paths) = self.analyses(graph, self.zero);
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
        report.add_facts(&s.facts);
        // The site's op is not removed by the rule but by the cascade, with the extraction
        // ops that only fed it: the per-site delta is 0 and the saving is in the cascade.
        *report.site_classes.entry(("bit-read", [0; 10])).or_default() += 1;
        surgery.replace_uses(s.output, s.bit);
    }
    let (program, surgery_report) = match surgery.finish_or_refuse()? {
        Finished::Done(program, report) => (program, report),
        Finished::Refused(program, why) => {
            report.refused = Some(why);
            // Nothing was rewritten, so nothing relied on the sites' facts (assumptions).
            report.facts.clear();
            report.sites = Default::default();
            report.site_classes = Default::default();
            return Ok((program, report));
        }
    };
    report.surgery = surgery_report;
    if let (Some(before), None) = (before, &report.cost_error) {
        match dce_counts(&program) {
            Ok(after) => report.close(&before, &after),
            Err(e) => report.cost_error = Some(e),
        }
    }
    Ok((program, report))
}
