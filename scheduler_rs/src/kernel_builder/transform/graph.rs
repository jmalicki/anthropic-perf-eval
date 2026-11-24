//! Read-only def/use view of an SSA program, for matchers and cost analysis.
//!
//! Works on a complete program (`SsaProgram`) and on the ops emitted so far (the emitter's
//! `ops`, whose op ids equal their indices). Dependencies follow dead code elimination
//! (`eliminate_ssa_dead_code`): a read of a vector group alias depends on the 8 lane
//! definers, a read of a decomposed lane on the decomposed vector's definer, and a load on
//! the stores of its memory RAW edges; roots are `is_dce_root`. Two differences, neither of
//! which occurs in programs emitted with `emit_const_choice` (every program today):
//! DCE keeps the Load/AddImm encodings of an `emit_const` constant together through its
//! `const_alternatives` list, which this view approximates by keeping ops that define the
//! same VReg together (`live`); and `producers` sees only the first definer of a VReg.
//! `live` is tested against `eliminate_ssa_dead_code` on built programs.
//! Maps are used for lookup only and never iterated, so results do not depend on hash order.

use std::collections::HashMap;

use crate::types::{MemEdgeKind, Opcode, SsaOp, VReg};

use super::super::emit_ssa::is_dce_root;

/// Def/use view of a slice of SSA ops.
pub struct SsaGraph<'a> {
    /// The ops, in emission (topological) order.
    pub ops: &'a [SsaOp],
    /// VReg -> index of its (first) defining op.
    def: HashMap<VReg, usize>,
    /// Vector group key -> its 8 lane VRegs (`SsaEmitContext::vector_group_lanes`).
    lanes: &'a HashMap<VReg, Vec<VReg>>,
    /// Lane VReg of a decomposed vector (read group) -> (vector, lane).
    element_of: HashMap<VReg, (VReg, usize)>,
}

impl<'a> SsaGraph<'a> {
    /// Build the view.
    ///
    /// # Arguments
    /// * `ops` - The ops, in emission order.
    /// * `lanes` - Vector groups of the program (alias or decomposed vector -> lanes).
    ///
    /// # Returns
    /// The graph; nothing is copied but two lookup maps.
    pub fn new(ops: &'a [SsaOp], lanes: &'a HashMap<VReg, Vec<VReg>>) -> Self {
        let mut def = HashMap::with_capacity(ops.len());
        for (i, op) in ops.iter().enumerate() {
            if let Some(dest) = op.dest {
                def.entry(dest).or_insert(i);
            }
        }
        // Lanes of a vector that has a definer are decomposed elements (read group); lanes of
        // an alias without one are written by loads (write group). Ops are scanned in order
        // so the map is built deterministically (only looked up afterwards).
        let mut element_of = HashMap::new();
        for op in ops {
            for v in op.reads() {
                if let Some(ls) = lanes.get(&v) {
                    if def.contains_key(&v) {
                        for (k, &e) in ls.iter().enumerate() {
                            element_of.entry(e).or_insert((v, k));
                        }
                    }
                }
            }
            if let Some(dest) = op.dest {
                if let Some(ls) = lanes.get(&dest) {
                    for (k, &e) in ls.iter().enumerate() {
                        element_of.entry(e).or_insert((dest, k));
                    }
                }
            }
        }
        SsaGraph { ops, def, lanes, element_of }
    }

    /// Index of the op defining `v`, if any (a write-group alias and the zero VReg have none).
    pub fn def_index(&self, v: VReg) -> Option<usize> {
        self.def.get(&v).copied()
    }

    /// The op defining `v`, if any.
    pub fn def_op(&self, v: VReg) -> Option<&'a SsaOp> {
        self.def_index(v).map(|i| &self.ops[i])
    }

    /// Lanes of a vector group (`v` is an alias written lane by lane, or a decomposed vector).
    pub fn group_lanes(&self, v: VReg) -> Option<&'a [VReg]> {
        self.lanes.get(&v).map(|l| l.as_slice())
    }

    /// `(vector, lane)` if `v` is a lane of a decomposed vector.
    pub fn element_of(&self, v: VReg) -> Option<(VReg, usize)> {
        self.element_of.get(&v).copied().filter(|(vec, _)| self.def.contains_key(vec))
    }

    /// Constant produced by the op defining `v` (`ConstChoice` / `Const`), if any.
    pub fn const_value(&self, v: VReg) -> Option<i64> {
        self.def_op(v).filter(|op| matches!(op.opcode, Opcode::ConstChoice | Opcode::Const))
            .and_then(|op| op.produces_constant)
    }

    /// The scalar `s` if `v` is `vbroadcast(s)`.
    pub fn broadcast_source(&self, v: VReg) -> Option<VReg> {
        let op = self.def_op(v)?;
        if op.opcode != Opcode::Vbroadcast {
            return None;
        }
        op.srcs.first().and_then(|s| s.as_vreg())
    }

    /// Constant value broadcast into `v` (`vbroadcast(const)`), if it is one.
    pub fn broadcast_const(&self, v: VReg) -> Option<i64> {
        self.broadcast_source(v).and_then(|s| self.const_value(s))
    }

    /// Indices of the ops op `idx` depends on, as dead code elimination sees it: the definer
    /// of each VReg it reads, the 8 lane definers for a vector group alias, the decomposed
    /// vector's definer for a lane. Sorted, without duplicates.
    ///
    /// # Arguments
    /// * `idx` - Op index.
    ///
    /// # Returns
    /// The producer indices.
    pub fn producers(&self, idx: usize) -> Vec<usize> {
        let mut out = Vec::new();
        for v in self.ops[idx].reads() {
            if let Some(ls) = self.lanes.get(&v) {
                if !self.def.contains_key(&v) {
                    out.extend(ls.iter().filter_map(|l| self.def_index(*l)));
                    continue;
                }
            }
            if let Some(d) = self.def_index(v) {
                out.push(d);
            } else if let Some((vec, _)) = self.element_of(v) {
                out.extend(self.def_index(vec));
            }
        }
        // A load needs the stores its memory RAW edges come from (`mem_deps` holds op ids;
        // ids equal indices in emission order before DCE, which is where this view is used).
        for &(src, kind) in &self.ops[idx].mem_deps {
            if kind == MemEdgeKind::Raw {
                if let Some(p) = self.ops.iter().position(|o| o.id == src) {
                    out.push(p);
                }
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Which ops dead code elimination keeps: roots (`is_dce_root`) and, transitively, their
    /// producers; ops defining the same VReg (alternative encodings) are kept together.
    ///
    /// # Returns
    /// One flag per op.
    pub fn live(&self) -> Vec<bool> {
        let n = self.ops.len();
        let mut live = vec![false; n];
        let mut stack: Vec<usize> = (0..n).filter(|&i| is_dce_root(&self.ops[i])).collect();
        for &i in &stack {
            live[i] = true;
        }
        // Alternative encodings: later definers of a VReg follow the first one.
        let mut partners: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (i, op) in self.ops.iter().enumerate() {
            if let Some(d) = op.dest {
                if let Some(first) = self.def_index(d) {
                    if first != i {
                        partners[first].push(i);
                        partners[i].push(first);
                    }
                }
            }
        }
        while let Some(i) = stack.pop() {
            for p in self.producers(i).into_iter().chain(partners[i].iter().copied()) {
                if !live[p] {
                    live[p] = true;
                    stack.push(p);
                }
            }
        }
        live
    }

    /// Users of every op (inverse of `producers`), each list in increasing op order.
    pub fn users(&self) -> Vec<Vec<usize>> {
        let mut users = vec![Vec::new(); self.ops.len()];
        for u in 0..self.ops.len() {
            for p in self.producers(u) {
                users[p].push(u);
            }
        }
        users
    }
}
