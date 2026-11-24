//! Read-only def/use view of an SSA program, for matchers and cost analysis.
//!
//! Works on a complete program (`SsaProgram`) and on the ops emitted so far (the emitter's
//! `ops`, whose op ids equal their indices). Dependencies follow dead code elimination
//! (`eliminate_ssa_dead_code`): a read of a vector group alias depends on the 8 lane
//! definers, and a read of a decomposed lane depends on the decomposed vector's definer. One
//! addition: a `SelectChoice`'s `delta_vreg` (read by its multiply_add lowering, not in
//! `srcs`) counts as a read (`op_reads`), so no rewrite can remove or move a delta under its
//! select. In emitted programs the delta's definer is also reached through the select's true
//! value, so liveness is the same as DCE's.
//!
//! The view assumes single assignment; `redefined` names the VRegs with more than one
//! definer (`def_index` gives the first, `const_value` none), so callers can refuse them.
//! Maps are used for lookup only and never iterated, so results do not depend on hash order.

use std::cell::OnceCell;
use std::collections::{BTreeSet, HashMap, HashSet};

use crate::types::{Opcode, Operand, SsaOp, VReg};

/// The VRegs `op` reads: its VReg operands and a `SelectChoice`'s `delta_vreg`.
///
/// # Arguments
/// * `op` - The op.
///
/// # Returns
/// The VRegs, operands first (duplicates kept).
pub fn op_reads(op: &SsaOp) -> impl Iterator<Item = VReg> + '_ {
    op.reads().chain(op.delta_vreg)
}

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
    /// `users()`, computed on first use of `readers`.
    users_cache: OnceCell<Vec<Vec<usize>>>,
    /// VRegs some op names in `keep_live`, computed on first use of `is_kept_live`.
    kept_live_cache: OnceCell<HashSet<VReg>>,
    /// VRegs defined by more than one op.
    redefined: BTreeSet<VReg>,
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
        let mut redefined = BTreeSet::new();
        for (i, op) in ops.iter().enumerate() {
            if let Some(dest) = op.dest {
                if def.contains_key(&dest) {
                    redefined.insert(dest);
                } else {
                    def.insert(dest, i);
                }
            }
        }
        // Lanes of a vector that has a definer are decomposed elements (read group); lanes of
        // an alias without one are written by loads (write group). Ops are scanned in order
        // so the map is built deterministically (only looked up afterwards).
        let mut element_of = HashMap::new();
        for op in ops {
            for v in op_reads(op) {
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
        SsaGraph { ops, def, lanes, element_of, users_cache: OnceCell::new(), kept_live_cache: OnceCell::new(), redefined }
    }

    /// VRegs with more than one definer (not single assignment), in increasing order.
    pub fn redefined(&self) -> &BTreeSet<VReg> {
        &self.redefined
    }

    /// Whether `v` has more than one definer.
    pub fn is_redefined(&self, v: VReg) -> bool {
        self.redefined.contains(&v)
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
    ///
    /// The `produces_constant` annotation is trusted only when it agrees with the op's
    /// immediate (`srcs[0]`), which is what lowering emits; a mismatch, a missing immediate or
    /// a redefined `v` gives `None`.
    pub fn const_value(&self, v: VReg) -> Option<i64> {
        if self.is_redefined(v) {
            return None;
        }
        let op = self.def_op(v).filter(|op| matches!(op.opcode, Opcode::ConstChoice | Opcode::Const))?;
        match (op.srcs.first(), op.produces_constant) {
            (Some(Operand::Imm(imm)), Some(k)) if *imm == k => Some(k),
            _ => None,
        }
    }

    /// The scalar `s` if `v` is `vbroadcast(s)` (and `v` is defined once).
    pub fn broadcast_source(&self, v: VReg) -> Option<VReg> {
        if self.is_redefined(v) {
            return None;
        }
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
    /// of each VReg it reads (`op_reads`, so including a select's delta), the 8 lane
    /// definers for a vector group alias, the decomposed vector's definer for a lane. Sorted,
    /// without duplicates.
    ///
    /// # Arguments
    /// * `idx` - Op index.
    ///
    /// # Returns
    /// The producer indices.
    pub fn producers(&self, idx: usize) -> Vec<usize> {
        let mut out = Vec::new();
        for v in op_reads(&self.ops[idx]) {
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
        out.sort_unstable();
        out.dedup();
        out
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

    /// Readers of op `idx` (its `users()` entry), from a cache built on first call.
    ///
    /// # Arguments
    /// * `idx` - Op index.
    ///
    /// # Returns
    /// Indices of the ops depending on op `idx`, increasing.
    pub fn readers(&self, idx: usize) -> &[usize] {
        &self.users_cache.get_or_init(|| self.users())[idx]
    }

    /// Whether some op lists `v`, or a lane of `v` when `v` is a vector group, in its
    /// `keep_live` (lifetime extension, not a read).
    ///
    /// # Arguments
    /// * `v` - The VReg.
    ///
    /// # Returns
    /// True if a rewrite that removes `v` would leave a dangling `keep_live` entry.
    pub fn is_kept_live(&self, v: VReg) -> bool {
        let kept = self
            .kept_live_cache
            .get_or_init(|| self.ops.iter().filter_map(|o| o.keep_live.as_ref()).flatten().copied().collect());
        kept.contains(&v) || self.group_lanes(v).map_or(false, |ls| ls.iter().any(|l| kept.contains(l)))
    }
}
