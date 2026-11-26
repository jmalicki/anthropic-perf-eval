//! Facts about values that rules need beyond local shape: a small constant propagation over
//! the SSA graph, the problem's memory layout, and the assumptions a run declares about its
//! inputs and observed outputs.
//!
//! * `MemoryLayout`: where the problem keeps its arrays (`problem.py` `build_mem_image`): the
//!   header word holding the pointer to the tree (4), the indices (5) and the values (6), and
//!   their extents (`n_nodes`, `batch_size`, `batch_size`). The three arrays lie back to back
//!   after the header, so two accesses are disjoint when each is *located* (inside the extent
//!   of the array its header word points to) and they are in different arrays or in
//!   non-overlapping ranges of one (`MemoryLayout::may_overlap`); an access that is not located
//!   may alias anything. A whole-program fact (`FactSource::Program`).
//! * Addresses (`ValueFacts::address`): `mem[word] + offset`, a pointer loaded from a constant
//!   header word plus constants, offsets mod 2^32 as the machine adds. A header word is read as
//!   a pointer only while no store with an address that is not located came before the load
//!   (such a store could have rewritten the header); after one, addresses are unknown.
//! * `Assumption`: what the run declares and the ops cannot show. `InitialIndicesZero` is an
//!   input precondition (`problem.py` `Input.generate` starts every index at 0; the canonical
//!   program already relies on it, R7's base case). `IndicesNotObserved` is an output
//!   contract (the harness compares values only). A rule relying on one names it on its
//!   site with `FactSource::Assumption` and refuses the site when it is not declared.
//! * `ValueFacts`: for every op, the constant its result holds in every lane, if one can be
//!   shown: constants, broadcasts, folding of ALU / VALU ops whose operands are known,
//!   `x * 0 = 0`, `x & 0 = 0`, `multiply_add(0, m, c) = c`, and a load of the input indices
//!   under `InitialIndicesZero` (address `mem[layout.indices_word] + k` with the 8 or 1 words
//!   inside the batch, and no store anywhere before the load). Lanes of decomposed vectors
//!   and vector group aliases are not tracked (unknown).
//!
//! Every value is a 32-bit word, as the machine computes it (`problem.py`: results mod 2^32).
//! Maps are not used; results are vectors indexed by op, so nothing depends on hash order.

use crate::types::{Engine, Opcode, Operand, SsaOp, VReg};

use super::graph::SsaGraph;
use super::rule::{Fact, FactSource};

const MASK: i64 = 0xFFFF_FFFF;

/// The problem's memory layout (`problem.py` `build_mem_image`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryLayout {
    /// Header word holding the pointer to the tree values.
    pub tree_word: i64,
    /// Header word holding the pointer to the input / output indices.
    pub indices_word: i64,
    /// Header word holding the pointer to the input / output values.
    pub values_word: i64,
    /// Words in the indices (and in the values) array.
    pub batch_size: i64,
    /// Words in the tree array, when known; `None` makes no tree access located (sound: every
    /// tree access may alias).
    pub n_nodes: Option<i64>,
}

impl MemoryLayout {
    /// The layout of `problem.py` for a batch.
    ///
    /// # Arguments
    /// * `batch_size` - Number of lanes (words per array).
    ///
    /// # Returns
    /// Header words 4 (tree), 5 (indices), 6 (values).
    pub fn problem(batch_size: usize) -> Self {
        MemoryLayout { tree_word: 4, indices_word: 5, values_word: 6, batch_size: batch_size as i64, n_nodes: None }
    }

    /// The layout with the tree's extent known.
    ///
    /// # Arguments
    /// * `n_nodes` - Words in the tree array.
    /// * `batch_size` - Number of lanes.
    ///
    /// # Returns
    /// `problem(batch_size)` with `n_nodes` set.
    pub fn problem_with_tree(n_nodes: usize, batch_size: usize) -> Self {
        MemoryLayout { n_nodes: Some(n_nodes as i64), ..Self::problem(batch_size) }
    }

    /// Words in the array header word `word` points to, if it points to one of known extent.
    pub fn extent(&self, word: i64) -> Option<i64> {
        if word == self.indices_word || word == self.values_word {
            Some(self.batch_size)
        } else if word == self.tree_word {
            self.n_nodes
        } else {
            None
        }
    }

    /// Whether `width` words at `a` lie inside the array `a.word` points to.
    pub fn located(&self, a: Address, width: i64) -> bool {
        self.extent(a.word).map_or(false, |n| a.offset >= 0 && a.offset + width <= n)
    }

    /// Whether two accesses may touch a common word: `false` only when both are located and
    /// they are in different arrays (laid out disjoint by `build_mem_image`) or in
    /// non-overlapping ranges of one.
    ///
    /// # Arguments
    /// * `a`, `wa` - First address (`None`: unknown) and width.
    /// * `b`, `wb` - Second address and width.
    pub fn may_overlap(&self, a: Option<Address>, wa: i64, b: Option<Address>, wb: i64) -> bool {
        match (a, b) {
            (Some(a), Some(b)) if self.located(a, wa) && self.located(b, wb) => {
                a.word == b.word && a.offset < b.offset + wb && b.offset < a.offset + wa
            }
            _ => true,
        }
    }

    /// The fact a rule records when it relies on the layout.
    pub fn fact(&self) -> Fact {
        Fact {
            name: "memory_layout",
            value: format!(
                "indices at mem[{}] (batch {}), values at mem[{}] (problem.py build_mem_image)",
                self.indices_word, self.batch_size, self.values_word
            ),
            source: FactSource::Program,
        }
    }

    /// Whether `width` words at `offset` from the indices pointer lie inside the indices.
    pub fn in_indices(&self, a: Address, width: i64) -> bool {
        a.word == self.indices_word && self.located(a, width)
    }
}

/// A declared assumption: something about the inputs or the observed outputs that the ops
/// cannot establish.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Assumption {
    /// Input precondition: every input index is 0 when the kernel starts.
    InitialIndicesZero,
    /// Output contract: the final indices are not observed (only the values are compared).
    IndicesNotObserved,
}

impl Assumption {
    /// Stable name.
    pub fn name(&self) -> &'static str {
        match self {
            Assumption::InitialIndicesZero => "initial-indices-zero",
            Assumption::IndicesNotObserved => "indices-not-observed",
        }
    }

    /// What is assumed and who states it.
    pub fn statement(&self) -> &'static str {
        match self {
            Assumption::InitialIndicesZero => {
                "every input index is 0 at entry (problem.py Input.generate; the canonical program's \
                 R7 base case; declared by the run)"
            }
            Assumption::IndicesNotObserved => {
                "the final indices are not an observed output; only the values are compared \
                 (declared by the run with --transform values-only; the validator then compares values only)"
            }
        }
    }

    /// The fact a site records when it relies on this assumption.
    pub fn fact(&self) -> Fact {
        Fact { name: self.name(), value: self.statement().into(), source: FactSource::Assumption }
    }
}

/// An address `mem[word] + offset`: an offset from a pointer read from a header word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Address {
    /// Header word the pointer is loaded from.
    pub word: i64,
    /// Constant offset added to it.
    pub offset: i64,
}

/// Known constants of an SSA program.
pub struct ValueFacts {
    /// Per op: the constant its dest holds in every lane, if shown.
    known: Vec<Option<i64>>,
    /// Per op: whether that constant relies on `InitialIndicesZero`.
    assumed: Vec<bool>,
    /// Per op: the address its (scalar) result holds, `mem[word] + offset`, if shown.
    addresses: Vec<Option<Address>>,
    /// The reserved zero register (holds 0, no definer).
    zero: Option<VReg>,
}

fn fold(opcode: Opcode, a: i64, b: i64) -> Option<i64> {
    let r = match opcode {
        Opcode::Add => a + b,
        Opcode::Sub => a - b,
        Opcode::Mul => a.wrapping_mul(b),
        Opcode::Xor => a ^ b,
        Opcode::And => a & b,
        Opcode::Or => a | b,
        Opcode::Shl => {
            if b >= 32 {
                0
            } else {
                a << b
            }
        }
        Opcode::Shr => {
            if b >= 32 {
                0
            } else {
                a >> b
            }
        }
        Opcode::Div if b != 0 => a / b,
        Opcode::Mod if b != 0 => a % b,
        Opcode::Lt => (a < b) as i64,
        Opcode::Eq => (a == b) as i64,
        _ => return None,
    };
    Some(r & MASK)
}

impl ValueFacts {
    /// Propagate constants through the program in order.
    ///
    /// # Arguments
    /// * `graph` - The program.
    /// * `zero` - The reserved zero register, if any.
    /// * `layout` - Where the input indices are.
    /// * `assumptions` - Declared assumptions; only `InitialIndicesZero` is used here.
    ///
    /// # Returns
    /// The facts.
    pub fn compute(graph: &SsaGraph, zero: Option<VReg>, layout: &MemoryLayout, assumptions: &[Assumption]) -> Self {
        let n = graph.ops.len();
        let mut facts = ValueFacts { known: vec![None; n], assumed: vec![false; n], addresses: vec![None; n], zero };
        let initial_zero = assumptions.contains(&Assumption::InitialIndicesZero);
        let mut stored = false;
        // Set once a store whose address is not located has run: it may have rewritten a
        // header word, so no header word is read as a pointer after it.
        let mut header_clobbered = false;
        for (i, op) in graph.ops.iter().enumerate() {
            let (value, assumed) = facts.evaluate(graph, op, layout, initial_zero && !stored);
            facts.known[i] = value.map(|v| v & MASK);
            facts.assumed[i] = value.is_some() && assumed;
            facts.addresses[i] = facts.address_of(graph, op, header_clobbered);
            if op.writes_memory() {
                stored = true;
                let w = if matches!(op.opcode, Opcode::Vstore) { 8 } else { 1 };
                let a = op.srcs.first().and_then(|s| s.as_vreg()).and_then(|v| facts.address(graph, v));
                if !a.map_or(false, |a| layout.located(a, w)) {
                    header_clobbered = true;
                }
            }
        }
        facts
    }

    /// Value of one op from the facts of the ops before it.
    fn evaluate(&self, graph: &SsaGraph, op: &SsaOp, layout: &MemoryLayout, initial_zero: bool) -> (Option<i64>, bool) {
        let operand = |s: &Operand| -> (Option<i64>, bool) {
            match s {
                Operand::Imm(k) => (Some(*k & MASK), false),
                Operand::VReg(v) => (self.value(graph, *v), self.relies_on_assumption(graph, *v)),
            }
        };
        if op.dest.is_none() {
            return (None, false);
        }
        match op.opcode {
            // The immediate, when `produces_constant` agrees with it (else unknown).
            Opcode::ConstChoice | Opcode::Const => match (op.srcs.first(), op.produces_constant) {
                (Some(Operand::Imm(imm)), Some(k)) if *imm == k => (Some(k), false),
                _ => (None, false),
            },
            Opcode::Vbroadcast => op.srcs.first().map_or((None, false), operand),
            Opcode::Vload | Opcode::Load => {
                let width = if op.opcode == Opcode::Vload { 8 } else { 1 };
                let a = op.srcs.first().and_then(|s| s.as_vreg()).and_then(|v| self.address(graph, v));
                match a {
                    Some(a) if initial_zero && layout.in_indices(a, width) => (Some(0), true),
                    _ => (None, false),
                }
            }
            Opcode::MultiplyAdd if op.srcs.len() == 3 => {
                let (a, b, c) = (operand(&op.srcs[0]), operand(&op.srcs[1]), operand(&op.srcs[2]));
                // A zero factor: the result is the addend, whatever the other factor is.
                for f in [a, b] {
                    if f.0 == Some(0) {
                        return (c.0, c.0.is_some() && (f.1 || c.1));
                    }
                }
                match (a.0, b.0, c.0) {
                    (Some(x), Some(y), Some(z)) => (Some(x.wrapping_mul(y).wrapping_add(z) & MASK), a.1 || b.1 || c.1),
                    _ => (None, false),
                }
            }
            _ if matches!(op.engine, Engine::Alu | Engine::Valu) && op.srcs.len() == 2 => {
                let (a, b) = (operand(&op.srcs[0]), operand(&op.srcs[1]));
                // Absorbing zero: known without the other operand.
                if matches!(op.opcode, Opcode::Mul | Opcode::And) {
                    for f in [a, b] {
                        if f.0 == Some(0) {
                            return (Some(0), f.1);
                        }
                    }
                }
                match (a.0, b.0) {
                    (Some(x), Some(y)) => (fold(op.opcode, x, y), a.1 || b.1),
                    _ => (None, false),
                }
            }
            _ => (None, false),
        }
    }

    /// The constant `v` holds in every lane, if shown.
    ///
    /// # Arguments
    /// * `graph` - The program the facts were computed on.
    /// * `v` - A VReg.
    ///
    /// # Returns
    /// The value (a 32-bit word), or `None` if unknown.
    pub fn value(&self, graph: &SsaGraph, v: VReg) -> Option<i64> {
        if Some(v) == self.zero {
            return Some(0);
        }
        // A VReg defined more than once has no single value.
        if graph.is_redefined(v) {
            return None;
        }
        graph.def_index(v).and_then(|i| self.known.get(i).copied().flatten())
    }

    /// Whether the value of `v` relies on `InitialIndicesZero`.
    pub fn relies_on_assumption(&self, graph: &SsaGraph, v: VReg) -> bool {
        graph.def_index(v).map_or(false, |i| self.assumed.get(i).copied().unwrap_or(false))
    }

    /// The address `v` holds as `mem[word] + offset`, if it is a pointer loaded from a constant
    /// header word plus a constant.
    ///
    /// # Arguments
    /// * `graph` - The program.
    /// * `v` - A scalar VReg used as an address.
    ///
    /// # Returns
    /// The address, or `None` if it is not of that form.
    pub fn address(&self, graph: &SsaGraph, v: VReg) -> Option<Address> {
        if graph.is_redefined(v) {
            return None;
        }
        graph.def_index(v).and_then(|i| self.addresses.get(i).copied().flatten())
    }

    /// The address op `op`'s result holds, from the addresses and constants of the ops before
    /// it (memoised per op by `compute`, so add chains cost linear time).
    fn address_of(&self, graph: &SsaGraph, op: &SsaOp, header_clobbered: bool) -> Option<Address> {
        let dest = op.dest?;
        if !dest.is_scalar() || graph.is_redefined(dest) {
            return None;
        }
        match (op.engine, op.opcode) {
            (Engine::Load, Opcode::Load) if !header_clobbered => {
                let word = op.srcs.first().and_then(|s| match s {
                    Operand::Imm(k) => Some(*k),
                    Operand::VReg(w) => self.value(graph, *w),
                })?;
                Some(Address { word, offset: 0 })
            }
            (Engine::Alu, Opcode::Add) if op.srcs.len() == 2 => {
                let (a, b) = (op.srcs[0].as_vreg()?, op.srcs[1].as_vreg()?);
                for (p, k) in [(a, b), (b, a)] {
                    if let (Some(base), Some(k)) = (self.address(graph, p), self.value(graph, k)) {
                        // The machine adds mod 2^32.
                        return Some(Address { word: base.word, offset: (base.offset + k) & MASK });
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// Why `v` is not shown to be 0, for rejections.
    ///
    /// # Arguments
    /// * `graph` - The program.
    /// * `v` - The VReg.
    /// * `layout` - Memory layout.
    /// * `assumptions` - Declared assumptions.
    ///
    /// # Returns
    /// A short reason.
    pub fn why_not_zero(&self, graph: &SsaGraph, v: VReg, layout: &MemoryLayout, assumptions: &[Assumption]) -> String {
        if let Some(k) = self.value(graph, v) {
            return format!("factor is the constant {}, not 0", k);
        }
        let Some(op) = graph.def_op(v) else { return "factor has no definer (vector group or lane)".into() };
        if matches!(op.opcode, Opcode::Vload | Opcode::Load) {
            let width = if op.opcode == Opcode::Vload { 8 } else { 1 };
            let a = op.srcs.first().and_then(|s| s.as_vreg()).and_then(|a| self.address(graph, a));
            return match a {
                Some(a) if layout.in_indices(a, width) && !assumptions.contains(&Assumption::InitialIndicesZero) => {
                    "factor is a load of the input indices and initial-indices-zero is not declared".into()
                }
                Some(a) if layout.in_indices(a, width) => "factor is a load of the indices after a store".into(),
                Some(a) => format!("factor is a load of mem[mem[{}] + {}], not of the input indices", a.word, a.offset),
                None => "factor is a load from an address that is not a header pointer plus a constant".into(),
            };
        }
        format!("factor ({:?}) is not provably 0", op.opcode)
    }
}
